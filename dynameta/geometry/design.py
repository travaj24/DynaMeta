"""
Design: the complete declarative device = UnitCell + Stack + electrodes +
materials + solver specs. Validated at construction. This is what the default
builders and the pipeline consume; it carries NO solver state.

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

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Tuple

from dynameta.geometry.unit_cell import UnitCell
from dynameta.geometry.stack import Stack, Layer
from dynameta.geometry.electrode import Electrode
from dynameta.geometry.specs import CarrierSpec, Mesh2DSpec, Mesh3DSpec, OpticalSpec
from dynameta.materials.material import MaterialRegistry


Role = Literal["metal", "semiconductor", "dielectric"]
_SYM_ORDER = {"none": 0, "c2v": 1, "c4v": 2}


@dataclass
class Design:
    name:        str
    unit_cell:   UnitCell
    stack:       Stack
    electrodes:  List[Electrode]
    materials:   MaterialRegistry
    mesh_2d:     Mesh2DSpec = field(default_factory=Mesh2DSpec)
    mesh_3d:     Mesh3DSpec = field(default_factory=Mesh3DSpec)
    optical:     OpticalSpec = field(default_factory=OpticalSpec)
    pretty_name: str = ""
    # v0.15 (plan 4.2.6, decision 37): carrier-solver settings (extended precision). APPENDED after
    # pretty_name so positional Design(...) calls are unchanged; the default (all unstated) is v0.14.
    carriers:    CarrierSpec = field(default_factory=CarrierSpec)

    def __post_init__(self) -> None:
        if not self.pretty_name:
            self.pretty_name = self.name
        if not isinstance(self.carriers, CarrierSpec):
            raise TypeError("Design.carriers must be a CarrierSpec, got {!r}".format(self.carriers))
        # Every referenced material must be registered.
        used = {self.stack.superstrate_material, self.stack.substrate_material}
        for L in self.stack.layers:
            used.update(L.materials_used())
        for f in self.stack.features:
            used.add(f.material)
        for mat in used:
            if mat not in self.materials:
                raise ValueError("Material '{}' used in stack is not registered "
                                  "(known: {})".format(mat, self.materials.names()))
        # Electrodes reference existing layers; names unique.
        layer_names = {L.name for L in self.stack.layers}
        for E in self.electrodes:
            if E.layer not in layer_names:
                raise ValueError("Electrode '{}' references unknown layer '{}'"
                                  .format(E.name, E.layer))
        enames = [E.name for E in self.electrodes]
        if len(set(enames)) != len(enames):
            raise ValueError("Duplicate electrode names: {}".format(enames))

    # ---- geometry helpers ----
    def z_intervals(self) -> Dict[str, Tuple[float, float]]:
        return self.stack.z_intervals()

    def material_role(self, material_name: str) -> Role:
        m = self.materials.get(material_name)
        if m.is_metal:
            return "metal"
        if m.is_semiconductor:
            return "semiconductor"
        return "dielectric"

    def semiconductor_layers(self) -> List[Layer]:
        """Layers whose background or any inclusion is a semiconductor."""
        out = []
        for L in self.stack.layers:
            if any(self.material_role(m) == "semiconductor"
                    for m in L.materials_used()):
                out.append(L)
        return out

    def device_symmetry(self) -> str:
        """Coarse device point-group ('c4v' | 'c2v' | 'none'), the intersection
        of the lattice symmetry with every inclusion's AND every Stack.feature's centered symmetry. Used
        to gate the carrier-field lift (SeparableXYLift requires 'c4v')."""
        sym = self.unit_cell.lattice_symmetry()
        cx, cy = self.unit_cell.center_m
        # audit S4-27 (wave-7, 2026-11-02): Stack.features are solids of the device too (a via, a
        # T-patch stem) -- they used to be skipped, so an OFF-CENTRE via left a c4v verdict
        # standing (and with it the SeparableXY lift and the FEM quarter-cell hint). A feature
        # demotes exactly like an inclusion: off-centre -> 'none', else its shape's symmetry.
        for ft in self.stack.features:
            fcx, fcy = ft.shape.center_m()
            if abs(fcx - cx) > 1e-12 or abs(fcy - cy) > 1e-12:
                return "none"
            sf = ft.shape.intrinsic_symmetry()
            if _SYM_ORDER[sf] < _SYM_ORDER[sym]:
                sym = sf
        for L in self.stack.layers:
            for inc in L.inclusions:
                scx, scy = inc.shape.center_m()
                # ABSOLUTE 1e-12 m (1 pm) on purpose, vs UnitCell.is_square's RELATIVE 1e-15 (audit
                # R-34): a centre is a sum of user offsets, so its round-off scales with the cell and
                # 1 pm is ~1e-6 of any optical period (a "centred" inclusion typed as 150e-9 in a
                # 300e-9 cell is bit-exact; one built as 1000e-9 - 850e-9 is 5e-23 m off), while the
                # squareness test compares two typed literals and only needs to absorb ~ulp noise.
                if abs(scx - cx) > 1e-12 or abs(scy - cy) > 1e-12:
                    return "none"
                s = inc.shape.intrinsic_symmetry()
                if _SYM_ORDER[s] < _SYM_ORDER[sym]:
                    sym = s
                # a tapered inclusion is as symmetric as the LESS symmetric of its two faces (a square
                # top over a rectangular base is c2v); its bottom shares the top's centre (Taper
                # validation), so only the shape symmetry can drop. taper=None: unchanged.
                if getattr(inc, "taper", None) is not None:
                    sb = inc.taper.bottom.intrinsic_symmetry()
                    if _SYM_ORDER[sb] < _SYM_ORDER[sym]:
                        sym = sb
        return sym

    def detect_symmetry_reduction(self) -> str:
        """The best mirror-symmetry FEM domain reduction this design is ELIGIBLE for, WITHOUT applying
        it: 'quarter' (c4v), 'half_x'/'half_y' (c2v), or 'none'. Purely advisory -- it never changes a
        solve; opt in by SETTING mesh_3d.symmetry to the returned value (the FEM build then meshes a
        half/quarter cell with symmetry walls, ~1/2 or ~1/4 the DOFs, for the SAME 0-order R/T). The
        optical builder emits a one-time hint when an eligible reduction is available but unused.
        Eligible = a centered, mirror-symmetric cell (device_symmetry c2v/c4v) at NORMAL incidence,
        a single linear x/y polarization (the OpticalSpec LAB-AXIS vocabulary of audit V-8 -- the
        wall type is keyed to the E axis, so 'p' is excluded), no
        carrier-coupled (semiconductor) or prismatic-boundary-layer feature, and only rectangle/circle
        inclusions (the Box/Cylinder OCC primitives validated for a clean symmetry-plane cut) -- the
        reduced-mesh path's current scope. The opt-in solve additionally requires a SCALAR eps (an
        anisotropic/gyrotropic tensor eps breaks the mirror parity and the solve raises); that is not
        knowable from the Design alone (it depends on the n_to_eps map) so it is enforced at solve
        time, not here. Pure inspection -- no solver/meshing."""
        devsym = self.device_symmetry()
        if devsym not in ("c2v", "c4v"):
            return "none"
        if abs(self.optical.incidence_angle_deg) > 1e-9 or abs(self.optical.azimuth_deg) > 1e-9:
            return "none"
        if self.optical.polarization not in ("x", "y"):
            return "none"          # the symmetry wall type is keyed to a linear x/y E axis ('p' rejected)
        if self.mesh_3d.semi_prism_thk_m:
            return "none"
        for L in self.stack.layers:
            if self.material_role(L.background_material) == "semiconductor" or any(
                    self.material_role(inc.material) == "semiconductor" for inc in L.inclusions):
                return "none"
            for inc in L.inclusions:
                if inc.shape.kind not in ("rectangle", "circle"):
                    return "none"
        for ft in self.stack.features:              # audit S4-27: the same scope rules for features
            if self.material_role(ft.material) == "semiconductor":
                return "none"
            if ft.shape.kind not in ("rectangle", "circle"):
                return "none"
        return "quarter" if devsym == "c4v" else "half_x"
