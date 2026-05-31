"""
Build a 3D periodic unit-cell NGSolve mesh from a dynameta.Design.

Maps each Design.Layer to an OCC band; layers with
lateral_extent='patch_footprint' are also split into an "in-patch column"
and an "outside-patch annulus" so the cavity can be locally refined.

Coordinate convention: OCC kernel works in nm units. UNIT_SCALE = 1e9
multiplies SI metres to get OCC nm.

Returns a Cell3DGeometry holding:
  - the Netgen mesh
  - per-region material name lookup
  - per-region z-interval lookup (in nm)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import netgen.occ as occ
import ngsolve as ng
from netgen.meshing import BoundaryLayerParameters, MeshingStep, IdentificationType

from dynameta.design import Design, Layer


UNIT_SCALE = 1e9       # m -> nm (OCC works in nm)


@dataclass
class Cell3DGeometry:
    mesh:                ng.Mesh
    material_by_region:  Dict[str, str]
    z_intervals:         Dict[str, Tuple[float, float]]   # nm
    period_nm:           float
    patch_side_nm:       float
    layer_name_by_region: Dict[str, str] = field(default_factory=dict)


def build_unit_cell_3d(design: Design) -> Cell3DGeometry:
    """Build the 3D periodic unit-cell mesh from the Design."""
    spec = design.mesh_3d
    S    = UNIT_SCALE
    P_nm = design.period_m * S
    L_nm = design.patch_side_m * S

    # ---- Layer stack: bottom is PML + substrate; then Design layers; then air + PML ----
    bands: List[Tuple[str, float, float, str]] = []   # name, z_lo_nm, z_hi_nm, default_material
    z = -(spec.pml_thk_m + spec.substrate_thk_m) * S
    bands.append(("pml_bot",   z, z + spec.pml_thk_m * S, "Si"))
    z += spec.pml_thk_m * S
    bands.append(("substrate", z, z + spec.substrate_thk_m * S, "Si"))
    z += spec.substrate_thk_m * S

    # For each Design layer, push a band. If the layer is a metal AND it's
    # the FIRST or LAST layer (mirror or patch) we split it into skin+bulk
    # for FEM efficiency.
    for i, L in enumerate(design.layers):
        thk_nm = L.thickness_m * S
        is_first_metal = (i == 0) and (L.role == "metal")
        is_last_metal  = (i == len(design.layers) - 1) and (L.role == "metal")
        if is_first_metal and spec.mirror_skin_thk_m > 0:
            bulk_thk = thk_nm - spec.mirror_skin_thk_m * S
            if bulk_thk > 0:
                bands.append((L.name + "_bulk", z, z + bulk_thk, L.material))
                z += bulk_thk
            bands.append((L.name + "_skin", z, z + spec.mirror_skin_thk_m * S,
                            L.material))
            z += spec.mirror_skin_thk_m * S
        elif is_last_metal and spec.patch_skin_thk_m > 0:
            skin_thk = spec.patch_skin_thk_m * S
            if skin_thk > thk_nm:
                skin_thk = thk_nm
            bands.append((L.name + "_skin", z, z + skin_thk, L.material))
            z += skin_thk
            bulk_thk = thk_nm - skin_thk
            if bulk_thk > 0:
                bands.append((L.name + "_bulk", z, z + bulk_thk, L.material))
                z += bulk_thk
        else:
            bands.append((L.name, z, z + thk_nm, L.material))
            z += thk_nm

    # Air buffer + top PML
    bands.append(("air_buffer", z, z + spec.air_buffer_m * S, "air"))
    z += spec.air_buffer_m * S
    bands.append(("pml_top",    z, z + spec.pml_thk_m * S, "air"))
    z += spec.pml_thk_m * S

    # Patch footprint mapping: layers with lateral_extent='patch_footprint' get
    # an in-patch column (square at center) + outside annulus.
    patch_layers = {L.name for L in design.layers
                      if L.lateral_extent.kind == "patch_footprint"}
    # Patch footprint bbox in OCC nm
    px_lo = (P_nm - L_nm) / 2.0
    px_hi = px_lo + L_nm
    py_lo = px_lo
    py_hi = px_hi

    # Cavity layers (full_period layers between mirror and patch) get
    # the in-patch / outside split too, to allow local refinement.
    cavity_layer_names = {L.name for L in design.layers
                            if L.lateral_extent.kind == "full_period"
                            and L.role in ("dielectric", "semiconductor")}
    # Semiconductor (ITO) layers -- targets for optional prismatic boundary
    # layers (sub-nm z at the accumulation interfaces).
    semi_layer_names = {L.name for L in design.layers if L.role == "semiconductor"}

    # ---- Build OCC solids ----
    z_intervals: Dict[str, Tuple[float, float]] = {}
    material_by_region: Dict[str, str] = {}
    layer_name_by_region: Dict[str, str] = {}
    solids: List[occ.Solid] = []

    for band_name, z_lo, z_hi, default_mat in bands:
        layer_owner = band_name
        for ln in (L.name for L in design.layers):
            if band_name == ln or band_name.startswith(ln + "_skin") \
                    or band_name.startswith(ln + "_bulk"):
                layer_owner = ln
                break
        # If band belongs to a patch_footprint layer -> single in-patch column only
        if layer_owner in patch_layers:
            inp = occ.Box(occ.Pnt(px_lo, py_lo, z_lo),
                            occ.Pnt(px_hi, py_hi, z_hi))
            inp.name = layer_owner + "_inpatch" if layer_owner in cavity_layer_names else band_name
            inp.bc("default")
            solids.append(inp)
            z_intervals[inp.name] = (z_lo, z_hi)
            material_by_region[inp.name] = default_mat
            layer_name_by_region[inp.name] = layer_owner
            continue
        # If band belongs to a cavity layer -> split inpatch / outside
        if layer_owner in cavity_layer_names:
            inp = occ.Box(occ.Pnt(px_lo, py_lo, z_lo),
                            occ.Pnt(px_hi, py_hi, z_hi))
            out_full = occ.Box(occ.Pnt(0.0, 0.0, z_lo),
                                  occ.Pnt(P_nm, P_nm, z_hi))
            # Boolean subtraction returns a Compound in newer netgen
            # (>=6.2.25xx); extract the single frame solid so .name/.bc can be
            # set (a Compound rejects property queries).
            out_annulus = (out_full - inp).solids[0]
            for s, suffix in ((inp, "_inpatch"), (out_annulus, "_outside")):
                s.name = layer_owner + suffix
                s.bc("default")
                z_intervals[s.name] = (z_lo, z_hi)
                material_by_region[s.name] = default_mat
                layer_name_by_region[s.name] = layer_owner
                solids.append(s)
            # Name the in-patch semiconductor (ITO) interface faces so prismatic
            # boundary layers can be grown from them. The in-patch column is
            # strictly interior (centred), so these prisms never touch the x/y
            # periodic faces. Names set on the solid's faces survive Glue().
            if spec.ito_prism_thk_m and layer_owner in semi_layer_names:
                for _f in inp.faces:
                    if abs(_f.center.z - z_hi) < 1e-2:
                        _f.name = "ito_bl_top"
                    elif abs(_f.center.z - z_lo) < 1e-2:
                        _f.name = "ito_bl_bot"
            continue
        # Default: full unit-cell band
        b = occ.Box(occ.Pnt(0.0, 0.0, z_lo), occ.Pnt(P_nm, P_nm, z_hi))
        b.name = band_name
        b.bc("default")
        solids.append(b)
        z_intervals[band_name] = (z_lo, z_hi)
        material_by_region[band_name] = default_mat
        layer_name_by_region[band_name] = layer_owner

    # Glue solids, then create OCC PERIODIC IDENTIFICATIONS (x=0<->x=P,
    # y=0<->y=P) so ng.Periodic(HCurl) finds them. Just naming the faces
    # "periodic_x_lo/hi" is NOT enough -- without face.Identify the FE space
    # has no periodic dofs and the Bloch BC is silently dropped (NGSolve also
    # warns that periodic= is not a valid HCurl kwarg). Identification MUST be
    # done on the glued shape's faces BEFORE wrapping in OCCGeometry. Matches
    # the proven Modulator geometry_3d.py.
    glued = occ.Glue(solids)
    _identify_periodic_faces(glued, P_nm)
    geo = occ.OCCGeometry(glued)

    # Boundary names (after identification): periodic x/y faces + the rest.
    for face in geo.shape.faces:
        c = face.center
        if abs(c.x - 0.0) < 1e-6:
            face.bc("periodic_x_lo")
        elif abs(c.x - P_nm) < 1e-6:
            face.bc("periodic_x_hi")
        elif abs(c.y - 0.0) < 1e-6:
            face.bc("periodic_y_lo")
        elif abs(c.y - P_nm) < 1e-6:
            face.bc("periodic_y_hi")

    # Per-region maxh assignments
    for solid in geo.shape.solids:
        n = solid.name
        maxh = _maxh_for_region(n, spec)
        solid.maxh = maxh

    gen_kwargs = dict(
        maxh=min(spec.maxh_air_m * S, spec.maxh_pml_m * S),
        perfstepsend=MeshingStep.MESHVOLUME,   # stop after volume mesh (skip
                                               # surface optimization for speed;
                                               # boundary layers are inserted in
                                               # the volume step, so still applied)
    )
    if spec.ito_prism_thk_m:
        # Grow prisms from BOTH in-patch ITO interfaces into the semiconductor
        # column. Structured prisms can be arbitrarily thin in z without
        # slivering, so order-2 HCurl assembles (unlike solid sub-layer splits).
        semi_inpatch = next((ln + "_inpatch" for ln in semi_layer_names
                              if ln in cavity_layer_names), None)
        if semi_inpatch is not None:
            prism_nm = [t * S for t in spec.ito_prism_thk_m]   # metres -> OCC nm
            gen_kwargs["boundary_layers"] = [
                BoundaryLayerParameters(
                    boundary="ito_bl_bot", thickness=prism_nm,
                    new_material=semi_inpatch, domain=semi_inpatch, outside=False),
                BoundaryLayerParameters(
                    boundary="ito_bl_top", thickness=prism_nm,
                    new_material=semi_inpatch, domain=semi_inpatch, outside=False),
            ]
    mesh = ng.Mesh(geo.GenerateMesh(**gen_kwargs))

    return Cell3DGeometry(
        mesh=mesh,
        material_by_region=material_by_region,
        z_intervals=z_intervals,
        period_nm=P_nm,
        patch_side_nm=L_nm,
        layer_name_by_region=layer_name_by_region,
    )


def _identify_periodic_faces(shape, P_nm: float) -> None:
    """Create OCC PERIODIC identifications on the glued unit-cell shape:
    every x=0 face -> its x=P counterpart (translation +P in x), and every
    y=0 face -> its y=P counterpart (+P in y). Faces are matched by their
    (transverse, z) center signature so multi-region stacks pair correctly.
    ng.Periodic(HCurl) consumes these identifications. Mirrors the proven
    Modulator geometry_3d.py.
    """
    tol = P_nm * 1e-4
    x0, xP, y0, yP = [], [], [], []
    for f in shape.faces:
        c = f.center
        if abs(c.x - 0.0) < tol:
            x0.append(f)
        elif abs(c.x - P_nm) < tol:
            xP.append(f)
        elif abs(c.y - 0.0) < tol:
            y0.append(f)
        elif abs(c.y - P_nm) < tol:
            yP.append(f)

    def sig_yz(f):
        c = f.center
        return (round(c.y * 1e3), round(c.z * 1e3))

    def sig_xz(f):
        c = f.center
        return (round(c.x * 1e3), round(c.z * 1e3))

    xP_by = {sig_yz(f): f for f in xP}
    yP_by = {sig_xz(f): f for f in yP}
    trans_x = occ.gp_Trsf.Translation(occ.Vec(P_nm, 0, 0))
    trans_y = occ.gp_Trsf.Translation(occ.Vec(0, P_nm, 0))
    nx = ny = 0
    for f0 in x0:
        partner = xP_by.get(sig_yz(f0))
        if partner is not None:
            f0.Identify(partner, "periodic_x_{}".format(nx),
                          IdentificationType.PERIODIC, trans_x)
            nx += 1
    for f0 in y0:
        partner = yP_by.get(sig_xz(f0))
        if partner is not None:
            f0.Identify(partner, "periodic_y_{}".format(ny),
                          IdentificationType.PERIODIC, trans_y)
            ny += 1


def _maxh_for_region(region_name: str, spec) -> float:
    S = UNIT_SCALE
    if "pml" in region_name:
        return spec.maxh_pml_m * S
    if "substrate" in region_name:
        return spec.maxh_substrate_m * S
    if "air" in region_name:
        return spec.maxh_air_m * S
    if region_name.endswith("_skin") and "mirror" in region_name:
        return (spec.maxh_mirror_skin_m or spec.maxh_metal_m) * S
    if region_name.endswith("_bulk") and "mirror" in region_name:
        return spec.maxh_mirror_bulk_m * S
    if region_name.endswith("_skin"):     # patch skin
        return (spec.maxh_patch_skin_m or spec.maxh_metal_m) * S
    if region_name.endswith("_bulk"):     # patch bulk
        return spec.maxh_patch_bulk_m * S
    if region_name.endswith("_inpatch"):
        return spec.maxh_cavity_in_patch_m * S
    if region_name.endswith("_outside"):
        return spec.maxh_cavity_outside_m * S
    return spec.maxh_metal_m * S
