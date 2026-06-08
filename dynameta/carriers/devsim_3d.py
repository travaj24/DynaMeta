"""
Native 3D DEVSIM carriers (equilibrium AND drift-diffusion).

Builds a 3D gmsh mesh of a stacked gated capacitor (semiconductor + gate oxide,
gate contact on top, body contact on bottom) and solves, on the 3D mesh, either:
  * EQUILIBRIUM (default): single-variable Poisson + Aymerich-Humet F_1/2
    (`physics_equilibrium`); or
  * DRIFT-DIFFUSION (`Stacked3DSpec.physics='drift_diffusion'`): FD-enhanced
    Scharfetter-Gummel electron continuity + Poisson (`physics_drift_diffusion`),
    body contact pinning Electrons (= N_D), abs_tol scaled to n_bg, staged
    zero-bias-seed -> gate-ramp Newton.
Emits a `CarrierField(ndim=3)` the bridge consumes DIRECTLY: the native 3D-grid branch of
assemble_eps places the (x,y,z) density with NO FieldLift synthesis (a lift applies to 2D
fields only) -- the physically-correct route for non-separable topologies (vs 2D + `SeparableXYLift`).

Validated: `validation/carriers_3d.py` (equilibrium: RelError~1e-8, +Vg accumulates/
-Vg depletes, Gauss to ~12%, lateral invariance ~1e-13); `validation/carriers_3d_dd.py`
(DD: converges, sign-correct, reduces to the equilibrium accumulation to 0.8% at +1V).

A centered GATE PATCH is supported (`gate_patch_frac` < 1: gated under the patch, free
surface in the gap -> laterally-varying accumulation), and `Stacked3DSpec.from_design`
derives the spec from a Design and names the region to match the optics alignment
(-> run_pipeline, no hand-alignment). A MULTI-DIELECTRIC gate stack is meshed as DISTINCT
regions (semi | oxide | diel1 | ... | gate on top) via `extra_dielectrics`, so the gate
voltage division is the exact series capacitance (from_design now keeps ALL gate-side
dielectric layers instead of collapsing to the nearest one). Arbitrary LATERAL material
INCLUSIONS (centered pillars of a different material in a layer, beyond the centered gate
patch) are also supported (Stacked3DSpec.inclusions / Inclusion3D): a separate adjacency-based
OCC build fragments them in as distinct regions and finds every region-region interface +
the contacts from surface->volume adjacency. SCOPE / remaining: a single gated semiconductor
(one semiconductor layer).

gmsh notes: its OCC kernel cannot build at 1e-9-metre scale, so the geometry is
built in NM and the mesh emitted SCALED to metres (Mesh.ScalingFactor); DEVSIM
reads MSH 2.2.
"""
from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from dynameta.core.carrier_field import (
    CarrierField, CarrierRegion, ELECTRON_DENSITY, POTENTIAL)
from dynameta.core.interfaces import RegionInfo
from dynameta.core.resample import resample_to_grid
from dynameta.carriers import physics_equilibrium as PE
from dynameta.carriers import physics_drift_diffusion as DD
from dynameta.carriers.dc_solve import solve_dc
from dynameta.carriers.physics_equilibrium import M_E


@dataclass
class Inclusion3D:
    """A lateral material INCLUSION embedded in one layer of the 3D stack: a centered rectangular
    pillar (x_frac x y_frac of the cell) of a different material spanning the FULL z-range of `in_layer`
    (a region name from the stack: 'semi', 'oxide', 'diel1', ...). role in {dielectric, semiconductor,
    metal}; eps = its DC permittivity (dielectric / semiconductor). The builder OCC-fragments it into the
    mesh as its own region, so the gate field couples differently through it -> laterally-varying
    accumulation (the non-separable carrier topology a 2D+symmetrization path cannot capture)."""
    name: str
    material: str
    role: str = "dielectric"
    eps: float = 1.0
    in_layer: str = "oxide"
    x_frac: float = 0.5
    y_frac: float = 0.5


@dataclass
class Stacked3DSpec:
    """A simple stacked gated capacitor for the 3D equilibrium solve (all SI)."""
    semi_material:   str = "ITO"
    oxide_material:  str = "HfO2"
    lateral_m:       float = 12e-9       # square lateral extent (x = y)
    semi_thk_m:      float = 12e-9
    oxide_thk_m:     float = 8e-9
    n_bg_m3:         float = 4e26
    eps_semi:        float = 9.5
    eps_oxide:       float = 18.0
    dos_mass_kg:     float = 0.35 * M_E
    mobility_m2Vs:   float = 0.004       # electron mobility (DD only); ITO ~40 cm^2/Vs
    physics:         str = "equilibrium" # "equilibrium" or "drift_diffusion"
    gate_patch_frac: float = 1.0         # gate footprint as a fraction of the cell (centered
                                          # square). 1.0 = full-cell gate; <1 = a PATCH gate
                                          # (laterally-varying accumulation: gated under the
                                          # patch, free surface in the gap -- the non-separable
                                          # topology the 2D+symmetrization path approximates).
    mesh_min_nm:     float = 0.5         # near the semi/oxide interface
    mesh_max_nm:     float = 3.0
    grid_n:          Tuple[int, int, int] = (16, 16, 33)   # (nx, ny, nz) output grid
    field_region_name: str = "semi"      # emitted CarrierField region key (match the optics
                                          # alignment source_region for run_pipeline integration)
    gate_name:       str = "gate"        # Design electrode name driving the gate -> the key
                                          # solve() reads from BiasPoint.voltages (the internal
                                          # DEVSIM contact stays "gate"). from_design sets this.
    body_name:       str = "body"        # Design electrode name for the body/back contact
    # EXTRA gate-side dielectric layers BEYOND the (nearest-semi) gate oxide, ordered from the oxide
    # OUTWARD toward the gate: each (material, thickness_m, eps_static). Empty (default) = the single
    # semi+oxide stack (byte-identical to before). With entries the builder meshes the FULL
    # multi-dielectric stack as DISTINCT regions (semi | oxide | extra0 | ... | gate on top), so the
    # gate voltage division across the series dielectric capacitance is exact (vs collapsing to one
    # oxide). from_design populates this from all the gate-side dielectric layers of a Design.
    extra_dielectrics: List[Tuple[str, float, float]] = field(default_factory=list)
    # lateral material INCLUSIONS embedded in a layer (centered pillars of a different material) -- the
    # non-separable carrier topology beyond the centered gate patch. Empty = a laterally-uniform stack.
    inclusions: List[Inclusion3D] = field(default_factory=list)

    def dielectric_stack_m(self) -> "List[Tuple[str, str, float, float]]":
        """Ordered gate-side dielectric layers from the semiconductor outward, as
        (region_name, material, thickness_m, eps_static). Region names are unique + stable."""
        out = [("oxide", self.oxide_material, float(self.oxide_thk_m), float(self.eps_oxide))]
        for k, (mat, thk, eps) in enumerate(self.extra_dielectrics):
            out.append(("diel{}".format(k + 1), str(mat), float(thk), float(eps)))
        return out

    def __post_init__(self):
        # Sanity-check the relative permittivities so a bogus/omitted value fails at
        # construction instead of silently producing wrong gate capacitance. eps_oxide
        # DEFAULTS to HfO2 (18.0); set it for a different gate dielectric. (Consistency
        # with the 2D builder, which raises, and from_design, which derives it.)
        if not (self.eps_oxide >= 1.0):
            raise ValueError("Stacked3DSpec.eps_oxide must be >= 1 (a DC relative "
                              "permittivity); got {}".format(self.eps_oxide))
        if not (self.eps_semi >= 1.0):
            raise ValueError("Stacked3DSpec.eps_semi must be >= 1; got {}".format(self.eps_semi))
        for mat, thk, eps in self.extra_dielectrics:
            if not (eps >= 1.0) or not (thk > 0.0):
                raise ValueError("extra_dielectrics entry '{}' needs thickness>0 and eps>=1; got "
                                  "thk={}, eps={}".format(mat, thk, eps))

    def layer_stack_nm(self) -> "List[Tuple[str, str, str, float, float, float]]":
        """The full ordered stack from the body up to the gate as
        (region_name, material, role, z_lo_nm, z_hi_nm, eps_static): the semiconductor then each
        gate-side dielectric. z in NANOMETRES (the gmsh-OCC build scale). Reduces to [semi, oxide]
        when there are no extra dielectrics."""
        z = 0.0
        out = [("semi", self.semi_material, "semiconductor", 0.0, self.semi_thk_m * 1e9,
                float(self.eps_semi))]
        z = self.semi_thk_m * 1e9
        for name, mat, thk_m, eps in self.dielectric_stack_m():
            out.append((name, mat, "dielectric", z, z + thk_m * 1e9, eps))
            z += thk_m * 1e9
        return out

    @classmethod
    def from_design(cls, design, *, gate_patch_frac=None, grid_n=(16, 16, 33),
                     mesh_min_nm: float = 0.5, mesh_max_nm: float = 3.0) -> "Stacked3DSpec":
        """Derive a stacked 3D spec from a `Design`. Handles a MULTI-LAYER stack (e.g. the
        full Park mirror/Al2O3/HfO2/ITO/HfO2/Al2O3/patch): finds the semiconductor layer and
        ALL the dielectric layers on the GATE side (the direction of the gate electrode), and
        meshes them as distinct regions (the nearest -> the gate "oxide", the rest ->
        extra_dielectrics) so the gate voltage division is the exact SERIES capacitance.
        Layers on the body side of the semiconductor (e.g. the mirror/back oxides) are not
        meshed (they do not set the gate accumulation). gate_patch_frac is derived from the
        gate electrode footprint. The emitted CarrierField region is named after the Design's
        semiconductor layer so it matches the optics alignment source_region (-> run_pipeline,
        no hand alignment). Inclusions are allowed in non-semiconductor layers (e.g. the
        patch); the semiconductor layer itself must be laterally uniform."""
        layers = design.stack.layers
        semi_idxs = [i for i, L in enumerate(layers)
                      if design.material_role(L.background_material) == "semiconductor"]
        if not semi_idxs:
            raise ValueError("from_design needs a semiconductor layer (a material with a transport model)")
        if len(semi_idxs) > 1:
            # The stacked builder models ONE gated semiconductor; with several it cannot
            # know which the gate drives (audit F2). Fail loudly rather than pick index 0.
            raise ValueError(
                "from_design: {} semiconductor layers ({}); the stacked 3D builder models a "
                "single gated semiconductor. Build a manual Stacked3DSpec for the one the gate "
                "drives.".format(len(semi_idxs), [layers[i].name for i in semi_idxs]))
        semi_idx = semi_idxs[0]
        semi_L = layers[semi_idx]
        if semi_L.inclusions:
            raise ValueError("from_design: the semiconductor layer '{}' must be laterally uniform "
                              "(no material inclusions); a gate PATCH is modeled via gate_patch_frac"
                              .format(semi_L.name))
        # gate electrode = the BIASED electrode (audit F1: do NOT pick the first electrode
        # with a CrossSection footprint, which could be a ground pad). Prefer a biased
        # electrode with a patch (CrossSection) footprint for the frac; else any biased one.
        biased = [e for e in design.electrodes if getattr(e, "role", "biased") == "biased"]
        gate_e = next((e for e in biased if not isinstance(e.footprint, str)),
                       (biased[0] if biased else None))
        gate_idx = (next((i for i, L in enumerate(layers) if L.name == gate_e.layer), len(layers))
                     if gate_e is not None else len(layers))   # default: gate above the semiconductor
        step = 1 if gate_idx > semi_idx else -1
        # Collect ALL gate-side dielectric layers from the semiconductor outward toward the gate (the
        # full multi-dielectric stack, e.g. Park's upper HfO2 + Al2O3). Stop at the gate layer or a
        # non-dielectric (metal/ambient). The nearest becomes the gate "oxide"; the rest are meshed as
        # distinct regions via extra_dielectrics -> the gate voltage division is the exact series
        # capacitance (the old code collapsed to just the nearest oxide).
        diel_layers = []
        j = semi_idx + step
        while 0 <= j < len(layers):
            Lj = layers[j]
            # a REAL gate dielectric is a dielectric WITH a DC permittivity (so the ambient 'air'
            # patch layer -- role 'dielectric' but no eps_static_dc -- stops the collection rather than
            # being meshed as a gate oxide).
            is_diel = (design.material_role(Lj.background_material) == "dielectric"
                       and design.materials.get(Lj.background_material).dc_permittivity() is not None)
            if not is_diel:
                break
            diel_layers.append(Lj)
            if j == gate_idx:        # the gate sits ON this dielectric -> it is the topmost gate oxide
                break
            j += step
        if not diel_layers:
            raise ValueError("from_design found no gate-side dielectric layer adjacent to "
                              "semiconductor '{}'".format(semi_L.name))
        ox_L = diel_layers[0]
        extra_diels = []
        for L in diel_layers[1:]:
            e = design.materials.get(L.background_material).dc_permittivity()
            if e is None:
                raise ValueError("gate dielectric '{}' has no eps_static_dc".format(L.background_material))
            extra_diels.append((L.background_material, float(L.thickness_m), float(e)))
        tr = design.materials.get(semi_L.background_material).transport
        if tr is None:
            raise ValueError("semiconductor '{}' has no transport model".format(semi_L.background_material))
        eps_ox = design.materials.get(ox_L.background_material).dc_permittivity()
        if eps_ox is None:
            raise ValueError("gate dielectric '{}' has no eps_static_dc".format(ox_L.background_material))
        cell = design.unit_cell
        # F3: the stacked carrier box is SQUARE (lateral_m is one side). A non-square cell
        # would mis-place the accumulation laterally in the optics (the carrier eps would
        # span only min(px,py)); refuse it rather than silently clamp.
        if abs(cell.period_x_m - cell.period_y_m) > 1e-3 * max(cell.period_x_m, cell.period_y_m):
            raise ValueError(
                "from_design: non-square unit cell (px={:.3g}, py={:.3g} m) is not supported "
                "by the square 3D carrier box; build a manual Stacked3DSpec.".format(
                    cell.period_x_m, cell.period_y_m))
        frac = gate_patch_frac
        if frac is None:                                  # derive from the gate-patch footprint
            frac = 1.0
            if gate_e is not None and not isinstance(gate_e.footprint, str):
                xlo, xhi, ylo, yhi = gate_e.footprint.bbox_m()
                wx, wy = xhi - xlo, yhi - ylo
                if abs(wx - wy) > 0.05 * max(wx, wy):     # F4: the patch model is a centered SQUARE
                    warnings.warn(
                        "from_design: gate footprint {:.3g}x{:.3g} m is not square; it is "
                        "collapsed to a centered square of the larger side (gate_patch_frac is "
                        "a single scalar).".format(wx, wy))
                frac = min(1.0, max(wx / cell.period_x_m, wy / cell.period_y_m))
        # gate/body electrode names so run_pipeline's BiasPoint (keyed by electrode name)
        # maps to the right contact instead of silently defaulting Vg=0 (verifier HIGH).
        gate_name = gate_e.name if gate_e is not None else "gate"
        ground_e = next((e for e in design.electrodes if getattr(e, "role", "") == "ground"), None)
        body_name = ground_e.name if ground_e is not None else "body"
        n_bg = tr.n_bg_m3
        mob = (float(tr.mobility_m2Vs_of_n_m3(n_bg)) if tr.mobility_m2Vs_of_n_m3 is not None else 0.004)
        return cls(semi_material=semi_L.background_material, oxide_material=ox_L.background_material,
                    lateral_m=min(cell.period_x_m, cell.period_y_m),
                    semi_thk_m=semi_L.thickness_m, oxide_thk_m=ox_L.thickness_m,
                    n_bg_m3=n_bg, eps_semi=tr.eps_static, eps_oxide=float(eps_ox),
                    dos_mass_kg=float(tr.dos_mass_kg_of_n_m3(n_bg)), mobility_m2Vs=mob,
                    physics=tr.physics, gate_patch_frac=float(frac),
                    field_region_name=semi_L.name, gate_name=gate_name, body_name=body_name,
                    grid_n=grid_n, mesh_min_nm=mesh_min_nm, mesh_max_nm=mesh_max_nm,
                    extra_dielectrics=extra_diels)


class Devsim3DEquilibrium:
    """A 3D-DEVSIM CarrierSolver (equilibrium) over a Stacked3DSpec. Implements the
    `regions()` + `solve(bias)` CarrierSolver Protocol, emitting CarrierField(ndim=3).

    bias.voltages: {"gate": V_gate, "body": V_body(=0)} -- gate on top of the oxide,
    body on the bottom of the semiconductor.
    """

    def __init__(self, spec: Stacked3DSpec, *, mesh_name: str = "ms3d_mesh",
                  device_name: str = "ms3d_device", msh_path: Optional[str] = None) -> None:
        self.spec = spec
        self.mesh_name = mesh_name
        self.device = device_name
        self.msh_path = msh_path or os.path.join(
            os.path.expanduser("~"), ".dynameta", "_devsim3d.msh")
        self._built = False
        s = spec
        self._z_semi = (0.0, s.semi_thk_m)
        self._z_ox = (s.semi_thk_m, s.semi_thk_m + s.oxide_thk_m)

    # ---- CarrierSolver Protocol ----
    def regions(self) -> List[RegionInfo]:
        s = self.spec
        return [RegionInfo(name=s.field_region_name, role="semiconductor", material=s.semi_material,
                            bbox_m=(0.0, s.lateral_m, 0.0, s.lateral_m, *self._z_semi),
                            ndim=3)]

    def build_device(self) -> None:
        import devsim as ds
        self._build_mesh()                                      # sets _mesh_regions, _iface_pairs, _top_diel
        ds.create_gmsh_mesh(mesh=self.mesh_name, file=self.msh_path)
        s = self.spec
        for (name, mat, _role, _eps) in self._mesh_regions:     # layers + any lateral inclusions
            ds.add_gmsh_region(mesh=self.mesh_name, gmsh_name=name, region=name, material=mat)
        ds.add_gmsh_contact(mesh=self.mesh_name, gmsh_name="gate", region=self._top_diel,
                              name="gate", material="metal")     # gate on the topmost (stack) dielectric
        ds.add_gmsh_contact(mesh=self.mesh_name, gmsh_name="body", region="semi",
                              name="body", material="metal")
        for (ifn, r0, r1) in self._iface_pairs:                 # one interface per adjacent region pair
            ds.add_gmsh_interface(mesh=self.mesh_name, gmsh_name=ifn, region0=r0, region1=r1, name=ifn)
        ds.finalize_mesh(mesh=self.mesh_name)
        ds.create_device(mesh=self.mesh_name, device=self.device)
        self._dd = (s.physics == "drift_diffusion")
        if self._dd:
            # full drift-diffusion: electron continuity (FD-enhanced Scharfetter-
            # Gummel) + Poisson on the 3D semi region (dimension-agnostic models).
            DD.setup_semiconductor_region_dd(self.device, "semi", n_bg_m3=s.n_bg_m3,
                                              eps_static=s.eps_semi, dos_mass_kg=s.dos_mass_kg,
                                              mobility_m2Vs=s.mobility_m2Vs)
        else:
            PE.setup_semiconductor_region(self.device, "semi", n_bg_m3=s.n_bg_m3,
                                           eps_static=s.eps_semi, dos_mass_kg=s.dos_mass_kg)
        for (name, mat, role, eps) in self._mesh_regions:       # dielectric layers + dielectric inclusions
            if name == "semi":
                continue                                         # the gated semiconductor, set above
            if role == "dielectric":
                PE.setup_dielectric_region(self.device, name, eps)
            elif role == "semiconductor":
                PE.setup_semiconductor_region(self.device, name, n_bg_m3=s.n_bg_m3,
                                               eps_static=eps, dos_mass_kg=s.dos_mass_kg)
            # metal inclusions are inert (no physics) -- their Dirichlet-free boundary is fine
        for itf in ds.get_interface_list(device=self.device):
            PE.setup_interface(self.device, itf)
        for c in ds.get_contact_list(device=self.device):
            # the "body" contact is on the semiconductor; for DD it must also pin the
            # electron density (Electrons = N_D). "gate" is on the oxide (Potential only).
            if self._dd and c == "body":
                DD.setup_contact_ohmic_dd(self.device, c)
            else:
                PE.setup_contact(self.device, c)
        self._built = True

    def solve(self, bias) -> CarrierField:
        import devsim as ds
        if not self._built:
            self.build_device()
        gn, bn = self.spec.gate_name, self.spec.body_name
        # Map the Design electrode names (BiasPoint keys) to the internal gate/body
        # contacts. The stacked builder drives ONLY these two, so warn on (a) a bias that
        # matches NEITHER name (would solve at Vg=0) and (b) any EXTRA key that matches
        # neither and is therefore silently dropped (e.g. a biased back contact on a
        # collapsed layer -- audit AD-2 extends the original verifier-HIGH guard).
        if bias.voltages:
            if gn not in bias.voltages and bn not in bias.voltages:
                warnings.warn(
                    "Devsim3DEquilibrium.solve: BiasPoint has voltages {} but neither gate_name "
                    "'{}' nor body_name '{}' is among them -- solving at Vg=0. Check the electrode "
                    "names.".format(sorted(bias.voltages), gn, bn))
            else:
                unknown = [k for k in bias.voltages if k not in (gn, bn)]
                if unknown:
                    warnings.warn(
                        "Devsim3DEquilibrium.solve: BiasPoint voltage key(s) {} match neither "
                        "gate_name '{}' nor body_name '{}' and are SILENTLY IGNORED -- the stacked "
                        "builder drives only the gate and body contact. Check the electrode "
                        "names.".format(unknown, gn, bn))
        vg = float(bias.voltages.get(gn, 0.0))
        vb = float(bias.voltages.get(bn, 0.0))
        ds.set_parameter(device=self.device, name="body_bias", value=vb)
        if getattr(self, "_dd", False):
            # 3D drift-diffusion: abs_tol scaled to the carrier density (SI continuity
            # residual ~n_bg; the _dc_abs_tol lesson), zero-bias seed, then ramp the
            # gate in 0.25 V steps (coupled Newton at each step).
            abs_tol = max(1e10, self.spec.n_bg_m3 * 1e-12)
            ds.set_parameter(device=self.device, name="gate_bias", value=0.0)
            solve_dc(self.device, method="newton", abs_tol=abs_tol, rel_tol=1e-5,
                      max_iter=100, semiconductor_regions=["semi"])
            n_steps = max(1, int(abs(vg) / 0.25 + 0.5))
            for k in range(1, n_steps + 1):
                ds.set_parameter(device=self.device, name="gate_bias", value=vg * k / n_steps)
                solve_dc(self.device, method="newton", abs_tol=abs_tol, rel_tol=1e-5,
                          max_iter=100, semiconductor_regions=["semi"])
        else:
            ds.set_parameter(device=self.device, name="gate_bias", value=vg)
            ds.solve(type="dc", solver_type="direct", absolute_error=1e10,
                      relative_error=1e-5, maximum_iterations=80)
        def g(nm):
            return np.array(ds.get_node_model_values(device=self.device, region="semi", name=nm))
        x, y, z = g("x"), g("y"), g("z")
        n, pot = g("Electrons"), g("Potential")
        nodes = np.column_stack([x, y, z])
        grid = resample_to_grid(nodes, {ELECTRON_DENSITY: n, POTENTIAL: pot},
                                  self.spec.grid_n)            # ndim-general resampler
        rname = self.spec.field_region_name
        reg = CarrierRegion(
            name=rname, role="semiconductor", material=self.spec.semi_material,
            nodes_m=nodes, node_fields={ELECTRON_DENSITY: n, POTENTIAL: pot},
            grid_axes_m={"x": grid["axis_0"], "y": grid["axis_1"], "z": grid["axis_2"]},
            grid_fields={ELECTRON_DENSITY: grid[ELECTRON_DENSITY], POTENTIAL: grid[POTENTIAL]})
        return CarrierField(
            bias_label=bias.label, voltages=dict(bias.voltages), ndim=3,
            temperature_K=PE.T_REF, regions={rname: reg},
            n_bg_by_region={rname: self.spec.n_bg_m3},
            unit_cell_m=(self.spec.lateral_m, self.spec.lateral_m))

    def teardown(self) -> None:
        import devsim as ds
        from dynameta.carriers import eq_registry as _R
        _R.clear(self.device)
        for dv in list(ds.get_device_list()):
            ds.delete_device(device=dv)
        for m in list(ds.get_mesh_list()):
            ds.delete_mesh(mesh=m)
        self._built = False

    # ---- gmsh mesh (nm geometry -> metre mesh) ----
    def _build_mesh(self) -> None:
        if self.spec.inclusions:                                # lateral inclusions -> adjacency-based build
            return self._build_mesh_incl()
        import gmsh
        s = self.spec
        Lnm = s.lateral_m * 1e9
        tsemi, tox = s.semi_thk_m * 1e9, s.oxide_thk_m * 1e9
        os.makedirs(os.path.dirname(self.msh_path), exist_ok=True)
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("ms3d")
        occ = gmsh.model.occ
        stack = s.layer_stack_nm()                              # [(name,mat,role,zlo,zhi,eps)] body->gate
        ztop = stack[-1][4]                                     # top of the topmost dielectric
        z_semi_top = stack[0][4]                                # semi/oxide interface (= tsemi)
        boundaries = [stack[k][3] for k in range(1, len(stack))]   # interior z-boundaries (semi/ox, ox/diel1,...)
        frac = float(s.gate_patch_frac)
        patterned = frac < 1.0 - 1e-9
        ph = Lnm * frac / 2.0                                   # patch half-width (centered)
        # one box per layer (semiconductor + each gate-side dielectric), fragmented into a conformal stack
        boxes = [occ.addBox(0, 0, zlo, Lnm, Lnm, zhi - zlo) for (_n, _m, _r, zlo, zhi, _e) in stack]
        occ.synchronize()
        if len(boxes) > 1:
            occ.fragment([(3, boxes[0])], [(3, b) for b in boxes[1:]])
            occ.synchronize()
        if patterned:
            # imprint the gate-patch rectangle onto the topmost dielectric top face so it splits into
            # the patch (gate contact) + the surrounding free surface (ungated gap).
            rect = occ.addRectangle(Lnm / 2 - ph, Lnm / 2 - ph, ztop, 2 * ph, 2 * ph)
            occ.synchronize()
            occ.fragment([(d, t) for d, t in gmsh.model.getEntities(3)], [(2, rect)])
            occ.synchronize()
        # classify each fragmented volume into its layer by z-centre -> a physical group per region
        for (name, mat, role, zlo, zhi, eps) in stack:
            tags = [t for (d, t) in gmsh.model.getEntities(3)
                    if zlo - 1e-6 < occ.getCenterOfMass(d, t)[2] < zhi + 1e-6]
            if tags:
                gmsh.model.addPhysicalGroup(3, tags, name=name)
        # surfaces: gate (topmost top), body (z=0), one interface per interior boundary
        gate, body = [], []
        ifaces = {round(zb, 6): [] for zb in boundaries}
        for dim, tag in gmsh.model.getEntities(2):
            zc = occ.getCenterOfMass(dim, tag)[2]
            if abs(zc - ztop) < 1e-4:
                if patterned:
                    bb = occ.getBoundingBox(dim, tag)
                    if (bb[3] - bb[0]) < Lnm - 1.0:           # narrower than the full cell -> the patch
                        gate.append(tag)                      # else: ungated free surface
                else:
                    gate.append(tag)
            elif abs(zc) < 1e-4:
                body.append(tag)
            else:
                for zb in boundaries:
                    if abs(zc - zb) < 1e-4:
                        ifaces[round(zb, 6)].append(tag); break
        gmsh.model.addPhysicalGroup(2, gate, name="gate")
        gmsh.model.addPhysicalGroup(2, body, name="body")
        semi_ox = []
        for k in range(1, len(stack)):
            zb = round(stack[k][3], 6)
            ifname = "if_{}".format(k)                         # interface between stack[k-1] and stack[k]
            if ifaces[zb]:
                gmsh.model.addPhysicalGroup(2, ifaces[zb], name=ifname)
            if k == 1:
                semi_ox = ifaces[zb]                           # the semi/oxide interface (mesh refinement)
        # device-build handoff (consumed by build_device): regions, interfaces, the gate dielectric
        self._mesh_regions = [(nm, mt, rl, ep) for (nm, mt, rl, _zl, _zh, ep) in stack]
        self._iface_pairs = [("if_{}".format(k), stack[k - 1][0], stack[k][0])
                              for k in range(1, len(stack))]
        self._top_diel = stack[-1][0]
        fd = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(fd, "SurfacesList", semi_ox)
        ft = gmsh.model.mesh.field.add("Threshold")
        gmsh.model.mesh.field.setNumber(ft, "InField", fd)
        gmsh.model.mesh.field.setNumber(ft, "SizeMin", s.mesh_min_nm)
        gmsh.model.mesh.field.setNumber(ft, "SizeMax", s.mesh_max_nm)
        gmsh.model.mesh.field.setNumber(ft, "DistMin", 1.0)
        gmsh.model.mesh.field.setNumber(ft, "DistMax", 6.0)
        gmsh.model.mesh.field.setAsBackgroundMesh(ft)
        for opt in ("Mesh.MeshSizeExtendFromBoundary", "Mesh.MeshSizeFromPoints",
                     "Mesh.MeshSizeFromCurvature"):
            gmsh.option.setNumber(opt, 0)
        gmsh.model.mesh.generate(3)
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
        gmsh.option.setNumber("Mesh.ScalingFactor", 1e-9)     # nm -> metre
        gmsh.write(self.msh_path)
        gmsh.finalize()

    def _build_mesh_incl(self) -> None:
        """Adjacency-based 3D OCC build supporting lateral material INCLUSIONS (centered pillars in a
        layer). Builds a box per layer + per inclusion, fragments them all, classifies each resulting
        volume into a region (inclusion footprint first, else the layer by z-centre), then finds every
        region-region INTERFACE and the gate/body CONTACTS from surface->volume adjacency (so the
        inclusions' LATERAL interfaces are captured, not just the stacked z-interfaces). Sets
        _mesh_regions / _iface_pairs / _top_diel for build_device."""
        import gmsh
        from collections import defaultdict
        s = self.spec
        Lnm = s.lateral_m * 1e9
        os.makedirs(os.path.dirname(self.msh_path), exist_ok=True)
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("ms3d")
        occ = gmsh.model.occ
        stack = s.layer_stack_nm()                              # [(name,mat,role,zlo,zhi,eps)]
        z_of = {nm: (zlo, zhi) for (nm, _m, _r, zlo, zhi, _e) in stack}
        ztop = stack[-1][4]
        frac = float(s.gate_patch_frac); patterned = frac < 1.0 - 1e-9; ph = Lnm * frac / 2.0
        for (_n, _m, _r, zlo, zhi, _e) in stack:               # a box per layer
            occ.addBox(0, 0, zlo, Lnm, Lnm, zhi - zlo)
        inc_geo = []                                            # (inc, x0,x1,y0,y1,zlo,zhi) nm footprints
        for inc in s.inclusions:                               # a centered pillar per inclusion
            if inc.in_layer not in z_of:
                raise ValueError("Inclusion3D.in_layer '{}' is not a stack region {}".format(
                    inc.in_layer, sorted(z_of)))
            zlo, zhi = z_of[inc.in_layer]
            wx, wy = inc.x_frac * Lnm, inc.y_frac * Lnm
            x0, y0 = Lnm / 2 - wx / 2, Lnm / 2 - wy / 2
            occ.addBox(x0, y0, zlo, wx, wy, zhi - zlo)
            inc_geo.append((inc, x0, x0 + wx, y0, y0 + wy, zlo, zhi))
        occ.synchronize()
        vols = [t for (_d, t) in gmsh.model.getEntities(3)]
        occ.fragment([(3, vols[0])], [(3, t) for t in vols[1:]])
        occ.synchronize()
        if patterned:
            rect = occ.addRectangle(Lnm / 2 - ph, Lnm / 2 - ph, ztop, 2 * ph, 2 * ph)
            occ.synchronize()
            occ.fragment([(d, t) for d, t in gmsh.model.getEntities(3)], [(2, rect)])
            occ.synchronize()

        def region_of(tag):                                     # classify by the volume's BOUNDING BOX
            # (NOT its centre of mass -- the oxide-minus-pillar FRAME has its COM at the cell centre,
            # inside the pillar footprint, so a COM test would misclassify the frame as the inclusion).
            bb = occ.getBoundingBox(3, tag)                     # xmin,ymin,zmin, xmax,ymax,zmax (nm)
            for (inc, x0, x1, y0, y1, zl, zh) in inc_geo:       # the inclusion = a box CONTAINED in its footprint
                if (bb[0] >= x0 - 1e-3 and bb[3] <= x1 + 1e-3 and bb[1] >= y0 - 1e-3 and
                        bb[4] <= y1 + 1e-3 and bb[2] >= zl - 1e-3 and bb[5] <= zh + 1e-3):
                    return inc.name
            cz = 0.5 * (bb[2] + bb[5])
            for (nm, _m, _r, zl, zh, _e) in stack:
                if zl - 1e-6 < cz < zh + 1e-6:
                    return nm
            return None
        vol_region = {}
        reg_tags = defaultdict(list)
        for (d, t) in gmsh.model.getEntities(3):
            r = region_of(t)
            vol_region[t] = r
            if r is not None:
                reg_tags[r].append(t)
        for r, tags in reg_tags.items():
            gmsh.model.addPhysicalGroup(3, tags, name=r)
        # surfaces via adjacency: 2 bounding vols of DIFFERENT regions -> interface; 1 vol -> a boundary
        gate, body, semi_ox = [], [], []
        ifsurf = defaultdict(list)
        for (d, t) in gmsh.model.getEntities(2):
            up, _down = gmsh.model.getAdjacencies(2, t)
            if len(up) == 2:
                r0, r1 = vol_region.get(int(up[0])), vol_region.get(int(up[1]))
                if r0 is not None and r1 is not None and r0 != r1:
                    ifsurf[tuple(sorted((r0, r1)))].append(t)
            else:
                cz = occ.getCenterOfMass(d, t)[2]
                if abs(cz - ztop) < 1e-4:
                    if patterned:
                        bb = occ.getBoundingBox(d, t)
                        if (bb[3] - bb[0]) < Lnm - 1.0:
                            gate.append(t)
                    else:
                        gate.append(t)
                elif abs(cz) < 1e-4:
                    body.append(t)
                # else: a lateral domain-boundary face -> natural zero-flux, no contact needed
        gmsh.model.addPhysicalGroup(2, gate, name="gate")
        gmsh.model.addPhysicalGroup(2, body, name="body")
        iface_pairs = []
        for i, (key, tags) in enumerate(sorted(ifsurf.items())):
            ifn = "if_{}".format(i)
            gmsh.model.addPhysicalGroup(2, tags, name=ifn)
            iface_pairs.append((ifn, key[0], key[1]))
            if {"semi", "oxide"} == set(key):
                semi_ox = tags
        self._mesh_regions = [(nm, mt, rl, ep) for (nm, mt, rl, _zl, _zh, ep) in stack] + \
                             [(inc.name, inc.material, inc.role, inc.eps) for inc in s.inclusions]
        self._iface_pairs = iface_pairs
        self._top_diel = stack[-1][0]
        if semi_ox:                                             # refine at the semi/oxide interface
            fd = gmsh.model.mesh.field.add("Distance")
            gmsh.model.mesh.field.setNumbers(fd, "SurfacesList", semi_ox)
            ft = gmsh.model.mesh.field.add("Threshold")
            gmsh.model.mesh.field.setNumber(ft, "InField", fd)
            gmsh.model.mesh.field.setNumber(ft, "SizeMin", s.mesh_min_nm)
            gmsh.model.mesh.field.setNumber(ft, "SizeMax", s.mesh_max_nm)
            gmsh.model.mesh.field.setNumber(ft, "DistMin", 1.0)
            gmsh.model.mesh.field.setNumber(ft, "DistMax", 6.0)
            gmsh.model.mesh.field.setAsBackgroundMesh(ft)
        for opt in ("Mesh.MeshSizeExtendFromBoundary", "Mesh.MeshSizeFromPoints", "Mesh.MeshSizeFromCurvature"):
            gmsh.option.setNumber(opt, 0)
        gmsh.option.setNumber("Mesh.MeshSizeMax", s.mesh_max_nm)   # cap everywhere (the inclusion has no Distance field)
        gmsh.model.mesh.generate(3)
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
        gmsh.option.setNumber("Mesh.ScalingFactor", 1e-9)
        gmsh.write(self.msh_path)
        gmsh.finalize()
