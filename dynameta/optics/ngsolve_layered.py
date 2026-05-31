"""
Default Stage-3 builder: a 3D periodic OCC/NGSolve unit cell from the layered
Design, implementing the core OpticalGeometryBuilder Protocol. Critically, it
emits a GeometryAlignment (the bridge keystone) so the carrier-derived eps is
placed on the right semiconductor subdomains -- the bridge never touches the mesh.

Bands (bottom->top): pml_bot, substrate_buffer (substrate material), the device
layers, superstrate_buffer, pml_top (superstrate material). Super/substrate
materials come from the Stack (not hardcoded). Inclusions are extruded OCC
solids (Rectangle->Box, Circle->Cylinder); the layer background fills the rest.
Full-cell cavity dielectric/semiconductor layers are split into an
inclusion-footprint column + outside annulus for local refinement; both
semiconductor sub-columns are aligned to the same carrier source region.

Interior-only inclusions (Phase 2/3): inclusions must lie strictly inside the
open cell so the four periodic boundary faces stay clean rectangles (keeps the
proven face Identify working).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import netgen.occ as occ
import ngsolve as ng
from netgen.meshing import BoundaryLayerParameters, MeshingStep, IdentificationType

from dynameta.core.units import NM
from dynameta.core.alignment import GeometryAlignment, RegionAlignment
from dynameta.geometry.design import Design

S = 1e9   # m -> nm (OCC works in nm); == 1/NM.metres_per_unit


@dataclass
class OpticalGeometry:
    mesh:                 ng.Mesh
    z_intervals_nm:       Dict[str, Tuple[float, float]]
    period_x_nm:          float
    period_y_nm:          float
    z_super_interface_nm: float    # device/superstrate-buffer top = air/PML start
    z_sub_interface_nm:   float    # bottom PML/substrate interface
    material_by_region:   Dict[str, str] = field(default_factory=dict)
    source_by_region:     Dict[str, str] = field(default_factory=dict)   # semi region -> carrier region
    n_px:                 int = 0        # # x-periodic identifications (Bloch phase order)
    n_py:                 int = 0        # # y-periodic identifications


class LayeredOpticalBuilder:
    def __init__(self, design: Design) -> None:
        self.design = design
        self._geo: Optional[OpticalGeometry] = None

    # ---- helpers ----
    def _refinement_footprint_nm(self) -> Optional[Tuple[float, float, float, float]]:
        """Lateral bbox (nm) of the principal (top-most) inclusion -- the
        cavity layers are locally refined under it."""
        for L in reversed(self.design.stack.layers):
            if L.inclusions:
                xlo, xhi, ylo, yhi = L.inclusions[0].shape.bbox_m()
                return (xlo * S, xhi * S, ylo * S, yhi * S)
        return None

    def _inclusion_solid(self, inc_shape, z_lo, z_hi):
        k = inc_shape.kind
        if k == "rectangle":
            xlo, xhi, ylo, yhi = [v * S for v in inc_shape.bbox_m()]
            return occ.Box(occ.Pnt(xlo, ylo, z_lo), occ.Pnt(xhi, yhi, z_hi))
        if k == "circle":
            cx, cy = inc_shape.center_m()
            return occ.Cylinder(occ.Pnt(cx * S, cy * S, z_lo), occ.Z,
                                  r=inc_shape.radius_m * S, h=(z_hi - z_lo))
        raise NotImplementedError(
            "inclusion shape '{}' not yet supported by the default OCC builder "
            "(Phase 3)".format(k))

    # ---- build ----
    def build(self) -> OpticalGeometry:
        d = self.design
        spec = d.mesh_3d
        Px = d.unit_cell.period_x_m * S
        Py = d.unit_cell.period_y_m * S
        z_iv_m = d.z_intervals()
        sub_mat = d.stack.substrate_material
        sup_mat = d.stack.superstrate_material

        solids: List = []
        z_intervals_nm: Dict[str, Tuple[float, float]] = {}
        material_by_region: Dict[str, str] = {}
        source_by_region: Dict[str, str] = {}
        region_align: List[RegionAlignment] = []

        footprint = self._refinement_footprint_nm()
        layers = d.stack.layers
        metal_idx = [i for i, L in enumerate(layers)
                      if d.material_role(L.background_material) == "metal"
                      or any(d.material_role(inc.material) == "metal" for inc in L.inclusions)]
        first_metal = metal_idx[0] if metal_idx else None
        last_metal = metal_idx[-1] if metal_idx else None

        def add_box(name, mat, z_lo, z_hi, xl=0.0, xh=None, yl=0.0, yh=None):
            xh = Px if xh is None else xh
            yh = Py if yh is None else yh
            b = occ.Box(occ.Pnt(xl, yl, z_lo), occ.Pnt(xh, yh, z_hi))
            b.name = name
            b.bc("default")
            solids.append(b)
            z_intervals_nm[name] = (z_lo, z_hi)
            material_by_region[name] = mat

        # bottom: PML + substrate buffer
        z = -(spec.pml_thk_m + spec.substrate_buffer_m) * S
        add_box("pml_bot", sub_mat, z, z + spec.pml_thk_m * S); z += spec.pml_thk_m * S
        z_sub_interface_nm = z
        add_box("substrate", sub_mat, z, z + spec.substrate_buffer_m * S)
        z += spec.substrate_buffer_m * S

        # device layers
        for i, L in enumerate(layers):
            thk = L.thickness_m * S
            z_lo, z_hi = z, z + thk
            bg_role = d.material_role(L.background_material)
            is_semi_bg = bg_role == "semiconductor"
            is_cavity = (bg_role in ("dielectric", "semiconductor")) and footprint is not None and not L.inclusions

            if L.inclusions:
                # inclusion solid(s) + background-minus-inclusions
                for j, inc in enumerate(L.inclusions):
                    inc_solid = self._inclusion_solid(inc.shape, z_lo, z_hi)
                    iname = "{}__incl{}".format(L.name, j)
                    inc_solid.name = iname; inc_solid.bc("default")
                    solids.append(inc_solid)
                    z_intervals_nm[iname] = (z_lo, z_hi)
                    material_by_region[iname] = inc.material
                # background = full cell minus inclusions
                bg = occ.Box(occ.Pnt(0, 0, z_lo), occ.Pnt(Px, Py, z_hi))
                for inc in L.inclusions:
                    bg = bg - self._inclusion_solid(inc.shape, z_lo, z_hi)
                for k_idx, s in enumerate(bg.solids):
                    bn = L.name if k_idx == 0 else "{}__bg{}".format(L.name, k_idx)
                    s.name = bn; s.bc("default")
                    solids.append(s)
                    z_intervals_nm[bn] = (z_lo, z_hi)
                    material_by_region[bn] = L.background_material
            elif is_cavity:
                fx0, fx1, fy0, fy1 = footprint
                inp = occ.Box(occ.Pnt(fx0, fy0, z_lo), occ.Pnt(fx1, fy1, z_hi))
                out_full = occ.Box(occ.Pnt(0, 0, z_lo), occ.Pnt(Px, Py, z_hi))
                out_pieces = (out_full - inp).solids
                inp.name = L.name + "_inpatch"; inp.bc("default")
                solids.append(inp)
                z_intervals_nm[inp.name] = (z_lo, z_hi)
                material_by_region[inp.name] = L.background_material
                if is_semi_bg:
                    source_by_region[inp.name] = L.name
                    region_align.append(RegionAlignment(
                        inp.name, L.name, (0.0, Px / S, 0.0, Py / S, z_lo / S, z_hi / S)))
                    # name in/out interface faces for optional prisms
                    if spec.semi_prism_thk_m:
                        for _f in inp.faces:
                            if abs(_f.center.z - z_hi) < 1e-2:
                                _f.name = "semi_bl_top"
                            elif abs(_f.center.z - z_lo) < 1e-2:
                                _f.name = "semi_bl_bot"
                for k_idx, s in enumerate(out_pieces):
                    on = "{}_outside".format(L.name) if k_idx == 0 else "{}_outside{}".format(L.name, k_idx)
                    s.name = on; s.bc("default")
                    solids.append(s)
                    z_intervals_nm[on] = (z_lo, z_hi)
                    material_by_region[on] = L.background_material
                    if is_semi_bg:
                        source_by_region[on] = L.name
                        region_align.append(RegionAlignment(
                            on, L.name, (0.0, Px / S, 0.0, Py / S, z_lo / S, z_hi / S)))
            else:
                # plain full-cell band; metal skin/bulk split for first/last metal
                if i in (first_metal, last_metal) and spec.metal_skin_thk_m > 0 \
                        and bg_role == "metal":
                    skin = min(spec.metal_skin_thk_m * S, thk)
                    if i == first_metal:   # mirror: bulk below, skin on top
                        if thk - skin > 0:
                            add_box(L.name + "_bulk", L.background_material, z_lo, z_hi - skin)
                        add_box(L.name + "_skin", L.background_material, z_hi - skin, z_hi)
                    else:                  # top metal: skin on bottom, bulk above
                        add_box(L.name + "_skin", L.background_material, z_lo, z_lo + skin)
                        if thk - skin > 0:
                            add_box(L.name + "_bulk", L.background_material, z_lo + skin, z_hi)
                else:
                    add_box(L.name, L.background_material, z_lo, z_hi)
                    if is_semi_bg:
                        source_by_region[L.name] = L.name
                        region_align.append(RegionAlignment(
                            L.name, L.name, (0.0, Px / S, 0.0, Py / S, z_lo / S, z_hi / S)))
            z += thk

        # top: superstrate buffer + PML
        z_super_interface_nm = z + spec.superstrate_buffer_m * S
        add_box("superstrate", sup_mat, z, z + spec.superstrate_buffer_m * S)
        z += spec.superstrate_buffer_m * S
        add_box("pml_top", sup_mat, z, z + spec.pml_thk_m * S)

        # glue + periodic identify (before OCCGeometry)
        glued = occ.Glue(solids)
        n_px, n_py = _identify_periodic(glued, Px, Py)
        geo = occ.OCCGeometry(glued)
        for face in geo.shape.faces:
            c = face.center
            if abs(c.x) < 1e-6:        face.bc("periodic_x_lo")
            elif abs(c.x - Px) < 1e-6: face.bc("periodic_x_hi")
            elif abs(c.y) < 1e-6:      face.bc("periodic_y_lo")
            elif abs(c.y - Py) < 1e-6: face.bc("periodic_y_hi")
        for solid in geo.shape.solids:
            solid.maxh = self._maxh(solid.name, material_by_region.get(solid.name, ""))

        gen_kwargs = dict(maxh=min(spec.maxh_superstrate_m, spec.maxh_pml_m) * S,
                            perfstepsend=MeshingStep.MESHVOLUME)
        if spec.semi_prism_thk_m and any("_inpatch" in r and r in source_by_region
                                          for r in source_by_region):
            semi_inp = next((r for r in source_by_region if r.endswith("_inpatch")), None)
            if semi_inp:
                prism_nm = [t * S for t in spec.semi_prism_thk_m]
                gen_kwargs["boundary_layers"] = [
                    BoundaryLayerParameters(boundary="semi_bl_bot", thickness=prism_nm,
                                              new_material=semi_inp, domain=semi_inp, outside=False),
                    BoundaryLayerParameters(boundary="semi_bl_top", thickness=prism_nm,
                                              new_material=semi_inp, domain=semi_inp, outside=False)]
        mesh = ng.Mesh(geo.GenerateMesh(**gen_kwargs))

        self._geo = OpticalGeometry(
            mesh=mesh, z_intervals_nm=z_intervals_nm, period_x_nm=Px, period_y_nm=Py,
            z_super_interface_nm=z_super_interface_nm, z_sub_interface_nm=z_sub_interface_nm,
            material_by_region=material_by_region, source_by_region=source_by_region,
            n_px=n_px, n_py=n_py)
        self._region_align = region_align
        return self._geo

    def _maxh(self, region_name: str, material: str) -> float:
        spec = self.design.mesh_3d
        role = self.design.material_role(material) if material in self.design.materials else ""
        if "pml" in region_name:        return spec.maxh_pml_m * S
        if "substrate" in region_name:  return spec.maxh_substrate_m * S
        if "superstrate" in region_name: return spec.maxh_superstrate_m * S
        if region_name.endswith("_skin"):
            return (spec.maxh_metal_skin_m or spec.maxh_metal_m) * S
        if region_name.endswith("_bulk"): return spec.maxh_metal_bulk_m * S
        if "_inpatch" in region_name:   return spec.maxh_inclusion_m * S
        if "_outside" in region_name:   return spec.maxh_background_m * S
        if role == "metal":             return spec.maxh_metal_m * S
        return spec.maxh_inclusion_m * S

    # ---- OpticalGeometryBuilder Protocol ----
    def mesh_regions(self) -> List[str]:
        if self._geo is None:
            self.build()
        return list(self._geo.mesh.GetMaterials())

    def alignment(self) -> GeometryAlignment:
        if self._geo is None:
            self.build()
        spatial = {ra.mesh_region for ra in self._region_align}
        fixed = {r: self._geo.material_by_region.get(r, self.design.stack.superstrate_material)
                  for r in self._geo.mesh.GetMaterials() if r not in spatial}
        return GeometryAlignment(unit_scale=NM, region_alignments=list(self._region_align),
                                  fixed_eps_regions=fixed)


def _identify_periodic(shape, Px: float, Py: float) -> Tuple[int, int]:
    """Returns (n_px, n_py): the count of x- then y- periodic identifications, in
    creation order -- the order a Floquet/Bloch phase list keys off."""
    tol = max(Px, Py) * 1e-4
    x0, xP, y0, yP = [], [], [], []
    for f in shape.faces:
        c = f.center
        if abs(c.x) < tol:        x0.append(f)
        elif abs(c.x - Px) < tol: xP.append(f)
        elif abs(c.y) < tol:      y0.append(f)
        elif abs(c.y - Py) < tol: yP.append(f)
    sig_yz = lambda f: (round(f.center.y * 1e3), round(f.center.z * 1e3))
    sig_xz = lambda f: (round(f.center.x * 1e3), round(f.center.z * 1e3))
    xP_by = {sig_yz(f): f for f in xP}
    yP_by = {sig_xz(f): f for f in yP}
    tx = occ.gp_Trsf.Translation(occ.Vec(Px, 0, 0))
    ty = occ.gp_Trsf.Translation(occ.Vec(0, Py, 0))
    n_px = 0
    for f0 in x0:
        p = xP_by.get(sig_yz(f0))
        if p is not None:
            f0.Identify(p, "px_{}".format(n_px), IdentificationType.PERIODIC, tx); n_px += 1
    n_py = 0
    for f0 in y0:
        p = yP_by.get(sig_xz(f0))
        if p is not None:
            f0.Identify(p, "py_{}".format(n_py), IdentificationType.PERIODIC, ty); n_py += 1
    return n_px, n_py
