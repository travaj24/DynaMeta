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
inclusion-footprint column + outside annulus (a mesh plane under the footprint,
and its own region for the eps/carrier alignment); both semiconductor
sub-columns are aligned to the same carrier source region.

Boundary-spanning inclusions ARE supported: an inclusion is clipped to the cell
and unioned with its periodic translates (_inclusion_solids_clipped), and the
cavity refinement column follows the same construction, so the four periodic
boundary faces always carry matching sub-faces. Two gates make that claim
checkable rather than hopeful (audit F-1/F-2): _identify_periodic REFUSES a face
it cannot pair, and _assert_periodic_complete re-checks the BUILT mesh's
periodic entity table before the geometry is handed out -- a mesh that netgen
identified only partially raises instead of solving non-periodically in silence.

POLARIZATION VOCABULARY (audit V-8): this module speaks {'x', 'y', 'p'} -- the LAB AXIS of the
incident E ('y' = s-pol, 'p' = p-pol, 'x' = E along lab x, transverse only at normal incidence). It
is one of five spellings in the repo -- {'s','p'} is the PLANE-OF-INCIDENCE spelling (tmm_reference,
resonance, nonlocal_tmm, shg_fem's closed forms, the oblique 2-D FDTD), {'te','tm'} the lumenairy
grating bridge's, the integer `row` 0/1 the differentiable Berreman/RCWA/PMM forwards', and
`pol_axis` hydro_fem's 2-D in-plane axis. The map, the `normalize_pol` converter and the
normal-incidence / azimuth caveats live in `dynameta.core.polarization`. The set ACCEPTED here is
UNCHANGED; acceptance unification (b), the V-8 follow-on, widened only the two PLANE-OF-INCIDENCE
families ({'s','p'} and {'te','tm'}), whose aliases name the same physical mode in every geometry
they cover; this vocabulary's crossings depend on the azimuth (or have no image at all), so they
stay STRICT and are made explicitly, through normalize_pol.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import netgen.occ as occ
import ngsolve as ng
from netgen.meshing import BoundaryLayerParameters, MeshingStep, IdentificationType

from dynameta.core.units import NM, NM_PER_M
from dynameta.core.alignment import GeometryAlignment, RegionAlignment
from dynameta.geometry.design import Design

S = NM_PER_M   # m -> nm (OCC works in nm); single source = core.units.NM_PER_M


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
    # Structural role of every region, recorded where the region is CREATED (audit F-7):
    # 'pml' | 'substrate' | 'superstrate' | 'metal_skin' | 'metal_bulk' | 'inclusion' | 'layer'.
    # Consumers must route on this, never on the region NAME: a user layer may legally be called
    # 'pml_calibration_film' or 'ito_substrate_cap'. Empty for a geometry built by other means, in
    # which case consumers fall back to the builder's own naming (documented at each site).
    role_by_region:       Dict[str, str] = field(default_factory=dict)
    n_px:                 int = 0        # # x-periodic identifications (Bloch phase order)
    n_py:                 int = 0        # # y-periodic identifications
    # Mirror-symmetry reduction: True when the x (resp. y) axis is a HALF-cell with symmetry walls
    # (faces named 'sym_x'/'sym_y') instead of a periodic boundary. period_x_nm/period_y_nm then carry
    # the REDUCED meshed extent (so the 0-order cell-average + area normalization follow automatically).
    # The solver maps these to PEC (tangential-E=0, Dirichlet) / PMC (natural) walls per polarization.
    sym_x:                bool = False
    sym_y:                bool = False


class LayeredOpticalBuilder:
    _symmetry_hinted = False     # emit the "you could reduce" advisory at most once per process

    def __init__(self, design: Design) -> None:
        self.design = design
        self._geo: Optional[OpticalGeometry] = None
        self._maxh_by_region: Dict[str, float] = {}   # per-region maxh (nm) that reached the mesher

    # ---- helpers ----
    def _refinement_footprint_nm(self) -> Optional[Tuple[float, float, float, float]]:
        """RAW lateral bbox (nm) of the principal (top-most) inclusion -- the cavity layers are
        split under it. NOTE: raw = possibly OUTSIDE the unit cell (a boundary-crossing
        inclusion's bbox overhangs it); every consumer must go through
        _footprint_solids_clipped, which clips + periodically wraps it."""
        for L in reversed(self.design.stack.layers):
            if L.inclusions:
                xlo, xhi, ylo, yhi = L.inclusions[0].shape.bbox_m()
                return (xlo * S, xhi * S, ylo * S, yhi * S)
        return None

    def _footprint_solids_clipped(self, fp, z_lo, z_hi, Px, Py, sym_x=False, sym_y=False):
        """The cavity-layer refinement column: the (raw) inclusion footprint clipped to the unit
        cell and unioned with its periodic translates -- exactly the construction
        _inclusion_solids_clipped applies to the inclusion itself, so the in-patch column sits
        under the inclusion (wrapped pieces included) and NEVER overhangs the cell.

        Audit F-2: the raw, UNCLIPPED footprint used to be handed straight to occ.Box, so for a
        boundary-crossing inclusion the cavity box stuck out of the cell; its side faces then had
        no partner on the opposite periodic boundary, _identify_periodic dropped them silently,
        and the solve was non-periodic with every upstream count still self-consistent. Clipping
        is a NO-OP for a strictly-interior footprint (the default path), which therefore reduces
        exactly to the single box built before. Returns a list of OCC solids (one per surviving
        translate piece); the caller names them all with ONE region name."""
        fx0, fx1, fy0, fy1 = fp
        eps = 1e-9 * max(Px, Py)
        dxs = (0.0,) if sym_x else (-Px, 0.0, Px)
        dys = (0.0,) if sym_y else (-Py, 0.0, Py)
        out = []
        for dx in dxs:
            for dy in dys:
                xa, xb = max(fx0 + dx, 0.0), min(fx1 + dx, Px)
                ya, yb = max(fy0 + dy, 0.0), min(fy1 + dy, Py)
                if xb - xa <= eps or yb - ya <= eps:
                    continue
                out.append(occ.Box(occ.Pnt(xa, ya, z_lo), occ.Pnt(xb, yb, z_hi)))
        return out

    def _check_symmetry_supported(self, d, sym):
        """Refuse a symmetry-reduced build for cases the reduced-mesh path does not yet handle, so an
        unsupported design fails LOUDLY rather than mis-meshing. Scope: a mirror-symmetric (centered)
        dielectric/metal cell at NORMAL incidence. The carrier-coupled (semiconductor), prismatic-
        boundary-layer, and rotated-polygon paths are out of scope for now (a mirror does not, in
        general, respect a carrier alignment / a rotated footprint)."""
        devsym = d.device_symmetry()
        if devsym == "none":
            raise NotImplementedError(
                "mesh_3d.symmetry={!r} needs a mirror-symmetric (centered) cell, but device_symmetry() "
                "is 'none' (off-center or asymmetric inclusion). Use symmetry='none'.".format(sym))
        if sym == "quarter" and devsym != "c4v":
            raise NotImplementedError(
                "mesh_3d.symmetry='quarter' needs a c4v (4-fold) cell; device_symmetry()={!r} supports "
                "a half-cell only ('half_x'/'half_y').".format(devsym))
        opt = getattr(d, "optical", None)
        if opt is not None and (abs(opt.incidence_angle_deg) > 1e-9 or abs(opt.azimuth_deg) > 1e-9):
            raise NotImplementedError(
                "mesh_3d.symmetry reduction is NORMAL-incidence only (an oblique/conical in-plane "
                "wavevector breaks the mirror symmetry); got incidence_angle_deg={:.3g}, azimuth_deg="
                "{:.3g}.".format(opt.incidence_angle_deg, opt.azimuth_deg))
        if opt is not None and opt.polarization not in ("x", "y", "p"):
            # audit V-8: an OFF-VOCABULARY label (e.g. the plane-of-incidence 's') is a different
            # failure from the in-vocabulary-but-unsupported 'p' below, and used to be reported as
            # the latter.  LAZY import, failure path only.
            from dynameta.core.polarization import pol_vocabulary_error
            raise pol_vocabulary_error(
                opt.polarization, "lab_xyp",
                where="LayeredOpticalBuilder._check_symmetry_supported", param="polarization")
        if opt is not None and opt.polarization not in ("x", "y"):
            # mirror the solver's wall-type guard at BUILD time, so requesting a reduction with an
            # unsupported polarization fails early/clearly rather than after the reduced mesh is built.
            # (This is a CAPABILITY refusal of the in-vocabulary 'p', not a vocabulary error: the
            # mirror wall type is PEC/PMC keyed to a linear x/y E axis and p-pol has no single one.)
            raise NotImplementedError(
                "mesh_3d.symmetry reduction requires polarization 'x' or 'y' (the wall type is keyed to "
                "the linear-polarization axis); got {!r}. Use symmetry='none'. ('x'/'y'/'p' is the "
                "OpticalSpec lab-axis vocabulary -- see dynameta.core.polarization, audit V-8.)"
                .format(opt.polarization))
        if d.mesh_3d.semi_prism_thk_m:
            raise NotImplementedError(
                "mesh_3d.symmetry reduction is not yet supported with semi_prism_thk_m (the prismatic "
                "boundary-layer faces on a reduced cell are unvalidated).")
        for L in d.stack.layers:
            if d.material_role(L.background_material) == "semiconductor" or any(
                    d.material_role(inc.material) == "semiconductor" for inc in L.inclusions):
                raise NotImplementedError(
                    "mesh_3d.symmetry reduction is not yet supported for a carrier-coupled "
                    "(semiconductor) layer -- the carrier->eps alignment is not symmetry-aware; "
                    "layer '{}'.".format(L.name))
            for inc in L.inclusions:
                if inc.shape.kind not in ("rectangle", "circle"):
                    raise NotImplementedError(
                        "mesh_3d.symmetry reduction supports rectangle/circle inclusions only (the "
                        "Box/Cylinder OCC primitives validated for a clean symmetry-plane cut); got "
                        "'{}'. Use symmetry='none' for this shape.".format(inc.shape.kind))

    def _inclusion_solid(self, inc_shape, z_lo, z_hi):
        k = inc_shape.kind
        if k == "rectangle":
            xlo, xhi, ylo, yhi = [v * S for v in inc_shape.bbox_m()]
            return occ.Box(occ.Pnt(xlo, ylo, z_lo), occ.Pnt(xhi, yhi, z_hi))
        if k == "circle":
            cx, cy = inc_shape.center_m()
            return occ.Cylinder(occ.Pnt(cx * S, cy * S, z_lo), occ.Z,
                                  r=inc_shape.radius_m * S, h=(z_hi - z_lo))
        if k in ("polygon", "regular_polygon"):
            return self._polygon_prism([(x * S, y * S) for x, y in inc_shape.vertices_m()],
                                        z_lo, z_hi)
        if k == "ellipse":
            # A TRUE ellipse (OCC analytic curve), not the inscribed 72-gon this used to build:
            # that under-filled the area by a systematic 0.127% (audit GEO-3), always in the same
            # direction, whereas the primitive reproduces pi*rx*ry*h to ~4e-7 -- and it makes an
            # ellipse with rx == ry agree with the `circle` branch instead of quietly disagreeing.
            #
            # occ.WorkPlane.Ellipse(major, minor) SEGFAULTS -- SIGSEGV, not a Python exception, so
            # it cannot be caught or reported -- whenever minor > major (verified on netgen's OCC
            # build). A tall inclusion (ry > rx) is therefore never expressed by passing the radii
            # in their natural order. Instead the major axis is chosen BY CONSTRUCTION: point the
            # workplane's local x along global Y and swap the radii, so `major >= minor` always
            # holds by the shape of the code rather than by the caller's luck.
            cx, cy = inc_shape.center_m()
            rx, ry = inc_shape.rx_m * S, inc_shape.ry_m * S
            origin = (cx * S, cy * S, z_lo)
            if rx >= ry:
                axes, major, minor = occ.Axes(origin, occ.Z), rx, ry
            else:
                axes, major, minor = occ.Axes(origin, occ.Z, occ.Y), ry, rx
            return occ.WorkPlane(axes).Ellipse(major, minor).Face().Extrude(z_hi - z_lo)
        raise NotImplementedError(
            "inclusion shape '{}' not supported by the default OCC builder".format(k))

    def _polygon_prism(self, pts_nm, z_lo, z_hi):
        """A vertical prism over a closed polygon (vertices in nm) -- the OCC primitive for
        polygon/regular_polygon inclusions and (via a fine vertex sampling) ellipses.

        The vertex list is normalized to counter-clockwise (positive signed area) so the
        extruded face is positively oriented regardless of the caller's winding. A
        clockwise (negative-area) face would extrude to a negative-volume solid whose
        cell-intersection captures the COMPLEMENT of the footprint, silently swapping the
        inclusion and background regions (audit GEO-1)."""
        pts = list(pts_nm)
        area2 = sum(x0 * y1 - x1 * y0
                    for (x0, y0), (x1, y1) in zip(pts, pts[1:] + pts[:1]))
        if area2 < 0.0:
            pts = pts[::-1]
        wp = occ.WorkPlane(occ.Axes((0.0, 0.0, z_lo), occ.Z))
        wp.MoveTo(*pts[0])
        for p in pts[1:]:
            wp.LineTo(*p)
        wp.Close()
        return wp.Face().Extrude(z_hi - z_lo)

    def _inclusion_solids_clipped(self, inc_shape, z_lo, z_hi, Px, Py, sym_x=False, sym_y=False):
        """The inclusion intersected with the unit cell, UNIONED with its periodic
        translates (+/-Px, +/-Py) each also intersected with the cell. An inclusion
        that crosses a cell boundary therefore contributes its wrapped piece(s) at the
        OPPOSITE boundary, so the periodic faces carry matching inclusion sub-faces
        that _identify_periodic pairs by (y,z)/(x,z) signature -- this is what makes a
        boundary-spanning (e.g. a connected grating stripe) inclusion periodic-correct.
        For a strictly-interior inclusion only the (0,0) translate survives the clip and
        this reduces exactly to the plain solid. Returns one (possibly multi-piece) OCC
        solid; rebuild the base per translate so an in-place .Move cannot alias.

        Px/Py here are the MESHED extents. On a symmetry-reduced axis (sym_x/sym_y) the cell
        is a half-cell and a MIRROR wall does NOT wrap, so the +/- periodic translate on that
        axis is SUPPRESSED (the centered inclusion is simply cut by the symmetry plane at the
        reduced extent -- e.g. a centered disk -> a quarter-disk whose flat edges lie on the
        walls)."""
        cell = occ.Box(occ.Pnt(0.0, 0.0, z_lo), occ.Pnt(Px, Py, z_hi))
        dxs = (0.0,) if sym_x else (-Px, 0.0, Px)
        dys = (0.0,) if sym_y else (-Py, 0.0, Py)
        pieces = []
        for dx in dxs:
            for dy in dys:
                t = self._inclusion_solid(inc_shape, z_lo, z_hi)
                if dx != 0.0 or dy != 0.0:
                    t = t.Move(occ.Vec(dx, dy, 0.0))
                clipped = t * cell
                if len(clipped.solids) > 0:
                    pieces.append(clipped)
        if not pieces:
            raise ValueError(
                "inclusion '{}' does not intersect the unit cell "
                "[0,{:.3g}]x[0,{:.3g}] nm; check its center/size".format(
                    inc_shape.kind, Px, Py))
        out = pieces[0]
        for p in pieces[1:]:
            out = out + p
        return out

    # ---- build ----
    def build(self) -> OpticalGeometry:
        d = self.design
        self._check_layer_names(d)          # audit F-7 (reserved region names)
        self._check_features(d)             # audit F-14 (silently ignored Stack.features)
        spec = d.mesh_3d
        Px = d.unit_cell.period_x_m * S
        Py = d.unit_cell.period_y_m * S
        # Mirror-symmetry reduction: a reduced axis is meshed at HALF extent with symmetry walls
        # (named 'sym_x'/'sym_y') replacing the periodic boundary. Only the meshed extent shrinks;
        # the physics is the full mirror-periodic tiling. Scope (gated below): plain + centered-
        # inclusion dielectric/metal layers; the carrier-coupled (semiconductor), prism, and metal
        # skin/bulk paths are NOT yet symmetry-aware, so refuse rather than mis-mesh.
        sym = getattr(spec, "symmetry", "none")
        sym_x = sym in ("half_x", "quarter")
        sym_y = sym in ("half_y", "quarter")
        if sym_x or sym_y:
            self._check_symmetry_supported(d, sym)
        elif not LayeredOpticalBuilder._symmetry_hinted:
            # ADVISORY only (never auto-applies): if this full-cell build is eligible for a mirror-
            # symmetry reduction, tell the user once they can opt in. Detection is non-binding.
            avail = d.detect_symmetry_reduction()
            if avail != "none":
                LayeredOpticalBuilder._symmetry_hinted = True
                frac = "~1/4 the lateral DOFs" if avail == "quarter" else "~1/2 the lateral DOFs"
                warnings.warn(
                    "this design is {} at normal incidence -- you can set mesh_3d.symmetry={!r} to "
                    "solve a symmetry-reduced cell ({}) at the SAME R/T. It is OPT-IN; the full "
                    "periodic cell is being meshed.".format(d.device_symmetry(), avail, frac),
                    stacklevel=2)
        Px_mesh = 0.5 * Px if sym_x else Px
        Py_mesh = 0.5 * Py if sym_y else Py
        _z_iv_m = d.z_intervals()          # call kept (it validates the stack); result unused here
        sub_mat = d.stack.substrate_material
        sup_mat = d.stack.superstrate_material

        solids: List = []
        z_intervals_nm: Dict[str, Tuple[float, float]] = {}
        material_by_region: Dict[str, str] = {}
        source_by_region: Dict[str, str] = {}
        region_align: List[RegionAlignment] = []
        # audit F-7: the STRUCTURAL role of every region, recorded where the region is created so
        # _maxh never has to infer it from a user-chosen layer name. Values:
        # 'pml' | 'substrate' | 'superstrate' | 'metal_skin' | 'metal_bulk' | 'inclusion' | 'layer'.
        role_by_region: Dict[str, str] = {}

        footprint = self._refinement_footprint_nm()
        layers = d.stack.layers
        metal_idx = [i for i, L in enumerate(layers)
                      if d.material_role(L.background_material) == "metal"
                      or any(d.material_role(inc.material) == "metal" for inc in L.inclusions)]
        first_metal = metal_idx[0] if metal_idx else None
        last_metal = metal_idx[-1] if metal_idx else None

        def add_box(name, mat, z_lo, z_hi, xl=0.0, xh=None, yl=0.0, yh=None, role="layer"):
            xh = Px_mesh if xh is None else xh
            yh = Py_mesh if yh is None else yh
            b = occ.Box(occ.Pnt(xl, yl, z_lo), occ.Pnt(xh, yh, z_hi))
            b.name = name
            b.bc("default")
            solids.append(b)
            z_intervals_nm[name] = (z_lo, z_hi)
            material_by_region[name] = mat
            role_by_region[name] = role

        # bottom: PML + substrate buffer
        z = -(spec.pml_thk_m + spec.substrate_buffer_m) * S
        add_box("pml_bot", sub_mat, z, z + spec.pml_thk_m * S, role="pml"); z += spec.pml_thk_m * S
        z_sub_interface_nm = z
        add_box("substrate", sub_mat, z, z + spec.substrate_buffer_m * S, role="substrate")
        z += spec.substrate_buffer_m * S

        # device layers
        for i, L in enumerate(layers):
            thk = L.thickness_m * S
            z_lo, z_hi = z, z + thk
            bg_role = d.material_role(L.background_material)
            is_semi_bg = bg_role == "semiconductor"
            is_cavity = (bg_role in ("dielectric", "semiconductor")) and footprint is not None and not L.inclusions

            if L.inclusions:
                # BI-1: a semiconductor in an inclusion layer (as background OR as an
                # inclusion) would be SILENTLY frozen at its nominal eps -- this branch
                # registers no carrier alignment, and the inclusion vs background region
                # naming diverges from the DEVSIM builder. Fail loudly until the
                # charge->optics bridge supports it (move the semiconductor to its own
                # full-cell layer, or supply a manual GeometryAlignment).
                if is_semi_bg or any(d.material_role(inc.material) == "semiconductor"
                                      for inc in L.inclusions):
                    raise NotImplementedError(
                        "layer '{}' has inclusions AND a semiconductor; the carrier->eps "
                        "bridge cannot align an inclusion-layer semiconductor (it would be "
                        "frozen at nominal eps). Put the semiconductor in its own full-cell "
                        "layer, or build a manual GeometryAlignment.".format(L.name))
                # inclusion solid(s) + background-minus-inclusions. Each inclusion is
                # clipped to the cell and unioned with its periodic translates, so a
                # boundary-spanning inclusion contributes >1 sub-solid (the wrapped
                # pieces); name every sub-solid the same region name (one material).
                for j, inc in enumerate(L.inclusions):
                    inc_solid = self._inclusion_solids_clipped(inc.shape, z_lo, z_hi, Px_mesh,
                                                               Py_mesh, sym_x, sym_y)
                    iname = "{}__incl{}".format(L.name, j)
                    for s in inc_solid.solids:
                        s.name = iname; s.bc("default")
                        solids.append(s)
                    z_intervals_nm[iname] = (z_lo, z_hi)
                    material_by_region[iname] = inc.material
                    role_by_region[iname] = "inclusion"
                # background = (meshed) cell minus the (cell-clipped) inclusions
                bg = occ.Box(occ.Pnt(0, 0, z_lo), occ.Pnt(Px_mesh, Py_mesh, z_hi))
                for inc in L.inclusions:
                    bg = bg - self._inclusion_solids_clipped(inc.shape, z_lo, z_hi, Px_mesh,
                                                             Py_mesh, sym_x, sym_y)
                if len(bg.solids) == 0:
                    raise ValueError(
                        "layer '{}' inclusion(s) leave no background region -- they cover "
                        "the entire unit cell (check inclusion size/winding).".format(L.name))
                for k_idx, s in enumerate(bg.solids):
                    bn = L.name if k_idx == 0 else "{}__bg{}".format(L.name, k_idx)
                    s.name = bn; s.bc("default")
                    solids.append(s)
                    z_intervals_nm[bn] = (z_lo, z_hi)
                    material_by_region[bn] = L.background_material
                    role_by_region[bn] = "layer"
            elif is_cavity:
                # in-patch = the inclusion footprint CLIPPED to the meshed cell (+ its periodic
                # wraps): an unclipped box overhangs the cell and breaks the periodic pairing (F-2).
                # Reduces to the plain single box for a strictly-interior footprint.
                inp_pieces = self._footprint_solids_clipped(footprint, z_lo, z_hi, Px_mesh,
                                                            Py_mesh, sym_x, sym_y)
                if not inp_pieces:
                    raise ValueError(
                        "layer '{}': the refinement footprint {} nm does not intersect the meshed "
                        "cell [0,{:.3g}]x[0,{:.3g}] nm".format(
                            L.name, tuple(round(v, 3) for v in footprint), Px_mesh, Py_mesh))
                out_full = occ.Box(occ.Pnt(0, 0, z_lo), occ.Pnt(Px_mesh, Py_mesh, z_hi))
                for _s in self._footprint_solids_clipped(footprint, z_lo, z_hi, Px_mesh,
                                                          Py_mesh, sym_x, sym_y):
                    out_full = out_full - _s      # fresh copies: the named pieces cannot alias
                out_pieces = out_full.solids
                # ONE region name for all in-patch pieces (the same idiom the inclusion branch
                # uses for a wrapped inclusion), so no new region name reaches the consumers.
                iname = L.name + "_inpatch"
                for inp in inp_pieces:
                    inp.name = iname; inp.bc("default")
                    solids.append(inp)
                z_intervals_nm[iname] = (z_lo, z_hi)
                material_by_region[iname] = L.background_material
                role_by_region[iname] = "layer"    # cavity in-patch is BACKGROUND material (see _maxh)
                if is_semi_bg:
                    source_by_region[iname] = L.name
                    region_align.append(RegionAlignment(
                        iname, L.name, (0.0, Px / S, 0.0, Py / S, z_lo / S, z_hi / S)))
                    # name in/out interface faces for optional prisms
                    if spec.semi_prism_thk_m:
                        for inp in inp_pieces:
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
                    role_by_region[on] = "layer"
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
                            add_box(L.name + "_bulk", L.background_material, z_lo, z_hi - skin,
                                    role="metal_bulk")
                        add_box(L.name + "_skin", L.background_material, z_hi - skin, z_hi,
                                role="metal_skin")
                    else:                  # top metal: skin on bottom, bulk above
                        add_box(L.name + "_skin", L.background_material, z_lo, z_lo + skin,
                                role="metal_skin")
                        if thk - skin > 0:
                            add_box(L.name + "_bulk", L.background_material, z_lo + skin, z_hi,
                                    role="metal_bulk")
                else:
                    add_box(L.name, L.background_material, z_lo, z_hi)
                    if is_semi_bg:
                        source_by_region[L.name] = L.name
                        region_align.append(RegionAlignment(
                            L.name, L.name, (0.0, Px / S, 0.0, Py / S, z_lo / S, z_hi / S)))
            z += thk

        # top: superstrate buffer + PML
        z_super_interface_nm = z + spec.superstrate_buffer_m * S
        add_box("superstrate", sup_mat, z, z + spec.superstrate_buffer_m * S, role="superstrate")
        z += spec.superstrate_buffer_m * S
        add_box("pml_top", sup_mat, z, z + spec.pml_thk_m * S, role="pml")

        # glue + periodic identify (before OCCGeometry). A symmetry-reduced axis is NOT identified
        # (its boundary becomes a symmetry wall, not a periodic pair) -> n_px/n_py = 0 on that axis.
        glued = occ.Glue(solids)
        n_px, n_py = _identify_periodic(glued, Px_mesh, Py_mesh, sym_x=sym_x, sym_y=sym_y)
        # Name horizontal full-cell layer INTERFACES (interior faces) by z (nm) here, PRE-OCCGeometry,
        # via f.name: an interior interface is not an exterior boundary, so it must be named on the glued
        # shape BEFORE OCCGeometry to reach the mesh (an exterior boundary can be (re)labelled after, as
        # the periodic faces are below). This lets a surface boundary condition (e.g. a graphene sheet,
        # solver.solve_fem(sheet_bcs={'iface_z<nm>': sigma})) target an interior interface. The tight
        # center tolerance matches only true full-cell horizontal faces; semi-prism is skipped to avoid
        # clobbering its named boundary-layer faces.
        if not spec.semi_prism_thk_m:
            for f in glued.faces:
                c = f.center
                if abs(c.x - 0.5 * Px_mesh) < 1e-3 * Px_mesh and abs(c.y - 0.5 * Py_mesh) < 1e-3 * Py_mesh:
                    f.name = "iface_z{}".format(int(round(c.z)))
        # Name the SYMMETRY-WALL faces here, PRE-OCCGeometry, via f.name -- the SAME mechanism as the
        # iface_z naming above (an exterior face .bc() set AFTER OCCGeometry does NOT propagate to
        # mesh.GetBoundaries(), so the solver's dirichlet='sym_y' would silently match nothing -- the
        # PEC wall would be dropped to a natural BC, a silent wrong-physics footgun). Both walls of a
        # reduced axis (the x=0/y=0 origin plane AND the x=Px_mesh/y=Py_mesh cut plane -- both mirror
        # planes for a centered cell) share one name; the solver applies the PEC/PMC type per pol.
        tol_x, tol_y = 1e-6 * Px_mesh, 1e-6 * Py_mesh
        if sym_x or sym_y:
            for f in glued.faces:
                c = f.center
                if sym_x and (abs(c.x) < tol_x or abs(c.x - Px_mesh) < tol_x):
                    f.name = "sym_x"
                elif sym_y and (abs(c.y) < tol_y or abs(c.y - Py_mesh) < tol_y):
                    f.name = "sym_y"
        # Per-region maxh, set on the GLUED shape PRE-OCCGeometry -- the same mechanism (and the same
        # trap) as the iface_z / sym-wall naming above: OCCGeometry.shape returns a FRESH wrapper on
        # every access (`g.shape is g.shape` is False), so a `solid.maxh` written through
        # `geo.shape.solids` lands on a throwaway copy that is discarded before GenerateMesh and every
        # per-region refinement knob is silently inert (audit F-23: maxh_metal/_skin/_bulk/_inclusion/
        # _background/_substrate all moved mesh.nv by exactly 0; raw-netgen A/B nv = 33 post vs 1215
        # pre). Set it here, where it reaches the mesher.
        maxh_by_region: Dict[str, float] = {}
        for solid in glued.solids:
            h = self._maxh(solid.name, material_by_region.get(solid.name, ""),
                           role_by_region.get(solid.name))
            solid.maxh = h
            maxh_by_region[solid.name] = h
        self._maxh_by_region = maxh_by_region     # what actually reached the mesher (F-7 gate)
        geo = occ.OCCGeometry(glued)
        # Diagnostic labels for the exterior PERIODIC side faces (the Bloch periodicity itself is driven
        # by the _identify_periodic Identify() calls, keyed by idnr -- not these names); skip a reduced
        # axis, whose walls are already named 'sym_x'/'sym_y' pre-OCCGeometry above.
        for face in geo.shape.faces:
            c = face.center
            if not sym_x and abs(c.x) < 1e-6:             face.bc("periodic_x_lo")
            elif not sym_x and abs(c.x - Px_mesh) < 1e-6: face.bc("periodic_x_hi")
            elif not sym_y and abs(c.y) < 1e-6:           face.bc("periodic_y_lo")
            elif not sym_y and abs(c.y - Py_mesh) < 1e-6: face.bc("periodic_y_hi")

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
        # POST-BUILD periodicity gate (F-1): the pre-mesh Identify counts are self-consistent even
        # when netgen's periodic entity table comes out incomplete, so check the built mesh itself.
        _assert_periodic_complete(mesh, Px_mesh, Py_mesh, sym_x=sym_x, sym_y=sym_y)

        self._geo = OpticalGeometry(
            mesh=mesh, z_intervals_nm=z_intervals_nm, period_x_nm=Px_mesh, period_y_nm=Py_mesh,
            z_super_interface_nm=z_super_interface_nm, z_sub_interface_nm=z_sub_interface_nm,
            material_by_region=material_by_region, source_by_region=source_by_region,
            role_by_region=role_by_region,
            n_px=n_px, n_py=n_py, sym_x=sym_x, sym_y=sym_y)
        self._region_align = region_align
        return self._geo

    # Region names the BUILDER itself emits verbatim; anything else in a name is user text.
    _RESERVED_REGIONS = {"pml_bot": "pml", "pml_top": "pml",
                         "substrate": "substrate", "superstrate": "superstrate"}

    @classmethod
    def _check_layer_names(cls, d) -> None:
        """RAISE if a Layer name is EXACTLY a region name this builder emits (audit F-7).

        `Stack.__post_init__` enforces only that layer names are unique AMONG THEMSELVES -- nothing
        stops `Layer("substrate", ...)`. `add_box` then writes into the same `z_intervals_nm` /
        `material_by_region` / `role_by_region` dicts the buffers use, so the layer OVERWRITES the
        buffer's entry: both solids end up sharing one mesh material, the layer's eps lands on the
        buffer, and downstream `solve_fem` silently moves its Fresnel interface z_int
        (`z_intervals_nm['substrate']`), its transmission probe band and its flux band. Every
        symptom is a plausible-looking number, which is why this is a raise and not a warning.

        The rule is EXACT-match only. A name that merely contains or is prefixed by a reserved word
        ('ito_substrate_cap', 'pml_calibration_film') is legal: every consumer that used to infer
        structure from the name now routes on `role_by_region` instead (`_maxh` here, the PML
        exclusion in `solver._absorbed_fraction` / `_per_region_absorption`).

        Design-time would be earlier, but the reserved set is the OPTICAL builder's, not the data
        model's: a Stack meshed by some other builder is entitled to those names."""
        bad = [L.name for L in d.stack.layers if str(L.name) in cls._RESERVED_REGIONS]
        if bad:
            raise ValueError(
                "Layer name(s) {} are region names LayeredOpticalBuilder emits itself, so the "
                "layer would OVERWRITE the corresponding buffer/PML region: the two solids would "
                "share one mesh material, the layer's eps would land on the buffer, and solve_fem "
                "would silently take its Fresnel interface / probe / flux bands from the wrong "
                "z-interval. Reserved (exact names only): {}. Rename the layer(s) -- a name that "
                "merely CONTAINS a reserved word (e.g. 'ito_substrate_cap') is fine."
                .format(", ".join(repr(b) for b in bad), ", ".join(sorted(cls._RESERVED_REGIONS))))

    @staticmethod
    def _check_features(d) -> None:
        """RAISE if the Stack carries `Feature`s (audit F-14).

        `Design.__post_init__` validates every Feature's material and `Feature.__post_init__`
        validates its z-span, so a via / T-patch stem passes construction cleanly -- and then this
        builder never reads `d.stack.features` (it lays down layers only), so the feature simply is
        not in the mesh. A validated Design that silently loses a solid is worse than one that
        refuses to build. `Feature`'s own docstring promises support "in a later phase"; until then
        say so at runtime instead of meshing a different device than the one described."""
        feats = list(getattr(d.stack, "features", ()) or ())
        if feats:
            raise NotImplementedError(
                "LayeredOpticalBuilder does not implement Stack.features yet, and would mesh the "
                "layer stack WITHOUT the {} feature(s) ({}) -- a device different from the one the "
                "Design describes. Model the feature as a Layer inclusion (Stack.layers[].inclusions), "
                "or subclass the builder to add the feature solids. (Design validates Feature "
                "materials and z-spans, so this cannot be caught at construction.)"
                .format(len(feats), ", ".join(repr(f.name) for f in feats)))

    def _maxh(self, region_name: str, material: str, mesh_role: Optional[str] = None) -> float:
        """Per-region target element size (nm). Routing, most specific first:
        pml/substrate/superstrate bands; the metal skin/bulk split; any METAL region
        (background OR inclusion) by maxh_metal_m; the INCLUSION solids by maxh_inclusion_m;
        everything else -- a plain full-cell device band, an inclusion layer's background /
        `__bg<k>` remainder, and BOTH halves of a cavity split (`_inpatch` column + `_outside`
        annulus, which are background material by construction) -- by maxh_background_m.

        ROUTE ON THE ROLE, NOT ON THE NAME (audit F-7). `mesh_role` is recorded by build() where
        each solid is created ('pml' | 'substrate' | 'superstrate' | 'metal_skin' | 'metal_bulk' |
        'inclusion' | 'layer'); the material's own role decides 'metal'. The previous SUBSTRING
        routing (`"substrate" in region_name`, `"pml" in region_name`, ...) was harmless while
        F-23 kept the whole loop inert, but it is now live and mis-sizes any user layer whose name
        merely CONTAINS a reserved word: 'ito_substrate_cap' took maxh_substrate_m,
        'pml_calibration_film' took maxh_pml_m, and a device layer literally named 'au_bulk' took
        maxh_metal_bulk_m -- silently, in the wrong direction (the substrate/pml knobs are the
        COARSEST in the spec, so a device layer routed there is under-resolved, not just slow).

        `mesh_role=None` (a direct call, or a subclass adding its own solids) falls back to name
        matching -- but ANCHORED: the reserved band names must match EXACTLY, and only the
        terminal `_skin` / `_bulk` tokens and the builder-only `__incl` marker (double underscore,
        unreachable from a Layer name via this builder) are treated as structural.

        Two routing rules exist only because F-23 made the values live; while the loop was inert
        the distinction was invisible and both defaults were landmines. (i) The catch-all is
        maxh_background_m, not maxh_inclusion_m: a 5 nm default (the shipped maxh_inclusion_m) on
        every plain dielectric film is a ~24x over-refinement of a region nobody asked to refine
        -- the repo's designs uniformly set maxh_background_m for their plain bands and
        maxh_inclusion_m only when they own an inclusion. (ii) The cavity `_inpatch` column
        follows the background, not the inclusion: it is a full-LAYER-thickness column of
        background material, so maxh_inclusion_m's 5 nm default put 5.1M elements (493 s of
        meshing) into e.g. tests/test_shg_grating's 250 nm air cap. The split still does its
        structural job (a mesh plane under the footprint + its own region for the eps/carrier
        alignment). The metal-role test sits ABOVE the inclusion test so a metal tooth keeps
        following maxh_metal_m."""
        spec = self.design.mesh_3d
        role = self.design.material_role(material) if material in self.design.materials else ""
        r = mesh_role
        if r is None:                                     # anchored name fallback (no substrings)
            r = self._RESERVED_REGIONS.get(region_name)
            if r is None and region_name.endswith("_skin"):   r = "metal_skin"
            elif r is None and region_name.endswith("_bulk"): r = "metal_bulk"
            elif r is None and "__incl" in region_name:       r = "inclusion"
            else:                                             r = r or "layer"
        if r == "pml":                  return spec.maxh_pml_m * S
        if r == "substrate":            return spec.maxh_substrate_m * S
        if r == "superstrate":          return spec.maxh_superstrate_m * S
        if r == "metal_skin":
            return (spec.maxh_metal_skin_m or spec.maxh_metal_m) * S
        if r == "metal_bulk":           return spec.maxh_metal_bulk_m * S
        if role == "metal":             return spec.maxh_metal_m * S
        if r == "inclusion":            return spec.maxh_inclusion_m * S
        return spec.maxh_background_m * S

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


def _identify_periodic(shape, Px: float, Py: float, sym_x: bool = False,
                       sym_y: bool = False) -> Tuple[int, int]:
    """Returns (n_px, n_py): the number of periodic IDENTIFICATIONS created on the x- then the
    y-axis (0 or 1 each, in creation order) -- i.e. exactly the length and order of the
    Floquet/Bloch `phase` list ng.Periodic keys by idnr. On a symmetry-reduced axis
    (sym_x/sym_y) the boundary is a mirror WALL, not a periodic pair, so that axis is NOT
    identified and its count is 0 (the wall faces are named 'sym_x'/'sym_y' instead).

    ONE identification per axis (all face pairs share the name 'px'/'py'), NOT one per face
    pair. Audit F-1: netgen expands a PER-FACE Face.Identify inconsistently on a glued
    multi-band stack -- for a full-width inclusion stripe next to a cavity-split full-cell
    layer one identification expands to that face's bottom EDGE only, leaving edge/face dofs
    off the periodic entity table (measured: 10 unpaired x-edges + 8 unpaired x-faces on a
    provably CONGRUENT pair of surface triangulations). ng.Periodic then leaves those dofs
    unconstrained and the solve is silently non-periodic. Sharing one identification name per
    axis makes netgen expand it over the whole boundary (measured: 0 unpaired entities on the
    same geometry), and it also collapses the interleaved per-layer idnr list the solver's
    Bloch-direction detection has to resolve down to (at most) two.

    Every candidate face MUST find a partner: an unpairable face used to be dropped without a
    word (audit F-2) while n_px/n_py stayed self-consistent with the successful pairs, so the
    band's boundary silently fell back to a natural BC. Partners are matched on the transverse
    centroid with an ABSOLUTE tolerance (not 0.001 nm integer buckets)."""
    tol = max(Px, Py) * 1e-4
    x0, xP, y0, yP = [], [], [], []
    for f in shape.faces:
        c = f.center
        if not sym_x and abs(c.x) < tol:        x0.append(f)
        elif not sym_x and abs(c.x - Px) < tol: xP.append(f)
        elif not sym_y and abs(c.y) < tol:      y0.append(f)
        elif not sym_y and abs(c.y - Py) < tol: yP.append(f)

    def sig_yz(f):
        return (f.center.y, f.center.z)

    def sig_xz(f):
        return (f.center.x, f.center.z)

    stol = max(Px, Py) * 1e-6

    def _diagnose(s0, other, sig, this_plane, other_plane):
        """Which of the two real causes produced this orphan -- they need DIFFERENT fixes, and the
        message used to assert the first one unconditionally.

          * OVERHANG / CROSSING (the F-2 mode): some solid extends outside [0,Px]x[0,Py], so its
            boundary never lands on the periodic planes where a partner could be found. Decided by
            the glued shape's own bounding box, which is unambiguous. Fix: clip + periodically wrap
            the feature (or move it inside).
          * TOUCHING WITHOUT SPANNING (the likelier user trigger): everything is inside the cell,
            but a feature ABUTS one periodic plane and does not reach the other, so the two planes
            are cut into different pieces. Fix: move the feature strictly inside the cell, or
            centre it on the boundary so it crosses and wraps. The far-plane faces sharing this
            z-band are reported as supporting evidence.
        """
        otol = 1e-3 * max(Px, Py)
        (blo, bhi) = shape.bounding_box
        span = []
        if blo[0] < -otol or bhi[0] > Px + otol:
            span.append("x in [{:.4g}, {:.4g}]".format(blo[0], bhi[0]))
        if blo[1] < -otol or bhi[1] > Py + otol:
            span.append("y in [{:.4g}, {:.4g}]".format(blo[1], bhi[1]))
        if span:
            return ("a geometry feature OVERHANGS or CROSSES the unit cell -- the meshed shape "
                    "spans {} nm, outside [0,{:.4g}]x[0,{:.4g}] nm -- so the wrapped piece that "
                    "would carry the partner face is missing or misplaced (typically an inclusion "
                    "or refinement footprint built from a RAW, unclipped bbox). Clip it to the "
                    "cell and union its periodic translates, or move it inside".format(
                        " and ".join(span), Px, Py))
        same_z = [c for c in (sig(f) for f in other) if abs(c[1] - s0[1]) < 1e3 * stol]
        return ("a geometry feature TOUCHES {} without spanning the unit cell: nothing lies "
                "outside [0,{:.4g}]x[0,{:.4g}] nm, but this z-band meets {} with {} face(s) "
                "(transverse offsets {} nm), so the two planes are cut into different pieces. "
                "Move the feature strictly inside the cell, or centre it on the boundary so it "
                "crosses and wraps".format(this_plane, Px, Py, other_plane, len(same_z),
                                            [round(c[0], 4) for c in same_z[:4]]))

    def _pairs(lo, hi, sig, axis, lo_plane, hi_plane):
        """Pair every lo-face with the hi-face at the same transverse centroid. RAISE on an
        orphan (F-2) or an ambiguous match (BI-5) -- both would otherwise drop a boundary to a
        natural BC with no error. The orphan message names WHICH cause applies (see _diagnose)."""
        out, used = [], set()
        for f0 in lo:
            s0 = sig(f0)
            cand = [j for j, fp in enumerate(hi)
                    if abs(sig(fp)[0] - s0[0]) < stol and abs(sig(fp)[1] - s0[1]) < stol]
            if len(cand) > 1:
                raise RuntimeError(
                    "periodic {}-boundary face-centroid collision at {}: {} faces on {} share a "
                    "transverse centroid, so the periodic partner is ambiguous. Refine the "
                    "inclusion topology.".format(axis, tuple(round(v, 4) for v in s0),
                                                  len(cand), hi_plane))
            if not cand or cand[0] in used:
                raise RuntimeError(
                    "periodic {}-boundary: the face on {} with transverse centroid {} nm has NO "
                    "partner on {} ({} faces on {} vs {} on {}). An unpaired periodic face is "
                    "left at a natural boundary condition and the solve is silently "
                    "non-periodic. Diagnosis: {}.".format(
                        axis, lo_plane, tuple(round(v, 4) for v in s0), hi_plane,
                        len(lo), lo_plane, len(hi), hi_plane,
                        _diagnose(s0, hi, sig, lo_plane, hi_plane)))
            used.add(cand[0])
            out.append((f0, hi[cand[0]]))
        if len(used) != len(hi):
            j0 = next(j for j in range(len(hi)) if j not in used)
            orphan = [tuple(round(v, 4) for v in sig(hi[j]))
                      for j in range(len(hi)) if j not in used]
            raise RuntimeError(
                "periodic {}-boundary: {} face(s) on {} have no partner on {} (centroids {} nm). "
                "An unpaired periodic face is left at a natural boundary condition and the solve "
                "is silently non-periodic. Diagnosis: {}.".format(
                    axis, len(orphan), hi_plane, lo_plane, orphan,
                    _diagnose(sig(hi[j0]), lo, sig, hi_plane, lo_plane)))
        return out

    tx = occ.gp_Trsf.Translation(occ.Vec(Px, 0, 0))
    ty = occ.gp_Trsf.Translation(occ.Vec(0, Py, 0))
    n_px = n_py = 0
    if x0 or xP:
        for f0, fp in _pairs(x0, xP, sig_yz, "x", "x=0", "x=Px"):
            f0.Identify(fp, "px", IdentificationType.PERIODIC, tx)
        n_px = 1
    if y0 or yP:
        for f0, fp in _pairs(y0, yP, sig_xz, "y", "y=0", "y=Py"):
            f0.Identify(fp, "py", IdentificationType.PERIODIC, ty)
        n_py = 1
    return n_px, n_py


def _assert_periodic_complete(mesh, Px: float, Py: float, sym_x: bool = False,
                              sym_y: bool = False) -> None:
    """POST-BUILD gate (audit F-1): after GenerateMesh, verify netgen actually recorded a
    COMPLETE periodic entity table -- every VERTEX, EDGE and FACE that lies on a periodic
    boundary plane must appear in mesh.GetPeriodicNodePairs for that axis. ng.Periodic
    silently leaves an unlisted boundary dof UNCONSTRAINED, so an incomplete table is a
    non-periodic solve with no error, no warning, and an R/T error that does not converge away
    (measured 9.4 points of T on the trigger geometry).

    Cost is O(boundary elements) but NOT free: measured 0.16-0.27 s on a ~33k-element mesh
    (nv ~ 7k, 70k boundary faces), i.e. ~8 % of that mesh's 2.1-3.0 s build -- not the
    "microseconds" the audit ledger's fix sketch assumed. It stays unconditional because it is
    O(1) in the SOLVE that follows and the failure it catches is silent, but it is a visible
    fraction of a build-only workload (a mesh-convergence sweep).

    The entity-pair COUNT is the artefact-free signal: a Periodic(H1).Set() point probe has
    both false positives (~1e-1 'violations' on provably complete tables) and false negatives,
    so it is deliberately not used here."""
    from ngsolve.fem import NODE_TYPE
    axes = ([] if sym_x else [("x", 0, Px)]) + ([] if sym_y else [("y", 1, Py)])
    if not axes:
        return                                   # quarter cell: no periodic identification at all
    ptol = 1e-6 * max(Px, Py)
    _pt: Dict[int, Tuple[float, float, float]] = {}

    def point(nr):
        p = _pt.get(nr)
        if p is None:
            p = mesh[ng.NodeId(NODE_TYPE.VERTEX, nr)].point
            _pt[nr] = p
        return p

    kinds = (("vertex", NODE_TYPE.VERTEX), ("edge", NODE_TYPE.EDGE), ("face", NODE_TYPE.FACE))
    need = {a[0]: {k: set() for k, _ in kinds} for a in axes}
    for el in mesh.Elements(ng.BND):
        vs = [point(v.nr) for v in el.vertices]
        for name, i, P in axes:
            if all(abs(p[i]) < ptol for p in vs) or all(abs(p[i] - P) < ptol for p in vs):
                need[name]["vertex"].update(v.nr for v in el.vertices)
                need[name]["edge"].update(e.nr for e in el.edges)
                need[name]["face"].update(f.nr for f in el.faces)
    def centroid(nt, nr):
        """Entity centroid from its vertices (VERTEX: the point itself). Only ever called on
        entities that appear in the periodic table, i.e. boundary ones."""
        if nt == NODE_TYPE.VERTEX:
            return point(nr)
        vs = [point(v.nr) for v in mesh[ng.NodeId(nt, nr)].vertices]
        return [sum(p[k] for p in vs) / len(vs) for k in range(3)]

    # classify each PAIR by its own translation vector (not by idnr: an idnr with no vertex
    # pair would otherwise go unclassified and its entities would read as missing)
    have = {a[0]: {k: set() for k, _ in kinds} for a in axes}
    for k, nt in kinds:
        for (a, b), _idnr in mesh.GetPeriodicNodePairs(nt):
            ca, cb = centroid(nt, a), centroid(nt, b)
            d = [cb[j] - ca[j] for j in range(3)]
            for name, i, P in axes:
                if abs(abs(d[i]) - P) < 1e-3 * P and all(abs(d[j]) < 1e-3 * P
                                                          for j in range(3) if j != i):
                    have[name][k].update((a, b))
    for name, i, P in axes:
        miss = {k: need[name][k] - have[name][k] for k, _ in kinds}
        if not any(miss.values()):
            continue
        where = sorted({tuple(round(c, 3) for c in point(v)) for v in miss["vertex"]})[:3]
        if not where:                            # edge/face-level only: locate via their element
            for el in mesh.Elements(ng.BND):
                if any(e.nr in miss["edge"] for e in el.edges) or \
                        any(f.nr in miss["face"] for f in el.faces):
                    where.append(tuple(round(c, 3) for c in point(el.vertices[0].nr)))
                if len(where) >= 3:
                    break
        raise RuntimeError(
            "periodic mesh INCOMPLETE on the {} axis: {} vertex / {} edge / {} face entities on "
            "the {}=0 / {}={:.4g} nm boundary planes are missing from netgen's periodic entity "
            "table (of {} / {} / {} present). ng.Periodic leaves those dofs unconstrained, so the "
            "solve would be silently non-periodic (no warning, and the error does NOT converge "
            "away under refinement). Offending mesh points near {} nm. This is a geometry feature "
            "netgen cannot identify as a whole -- typically a full-width inclusion stripe adjacent "
            "to a cavity-split full-cell layer, or a feature crossing the cell boundary. Restrict "
            "the design to interior inclusions (see the module docstring) or split the offending "
            "layer.".format(name, len(miss["vertex"]), len(miss["edge"]), len(miss["face"]),
                            name, name, P, len(need[name]["vertex"]), len(need[name]["edge"]),
                            len(need[name]["face"]), where))
