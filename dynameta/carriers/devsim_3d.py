"""
Native 3D DEVSIM carriers (equilibrium AND drift-diffusion).

Builds a 3D gmsh mesh of a stacked gated capacitor (semiconductor + gate oxide,
gate contact on top, body contact on bottom) and solves, on the 3D mesh, either:
  * EQUILIBRIUM (default): single-variable Poisson + Aymerich-Humet F_1/2
    (`physics_equilibrium`); or
  * DRIFT-DIFFUSION (`Stacked3DSpec.physics='drift_diffusion'`): FD-enhanced
    Scharfetter-Gummel electron continuity + Poisson (`physics_drift_diffusion`),
    body contact pinning the electron QFL, abs_tol scaled to n_bg, staged
    zero-bias-seed -> gate-ramp Newton.
Emits a `CarrierField(ndim=3)` the bridge consumes via `IdentityLift` -- the
physically-correct route for non-separable topologies (vs 2D + `SeparableXYLift`).

Validated: `validation/carriers_3d.py` (equilibrium: RelError~1e-8, +Vg accumulates/
-Vg depletes, Gauss to ~12%, lateral invariance ~1e-13); `validation/carriers_3d_dd.py`
(DD: converges, sign-correct, reduces to the equilibrium accumulation to 0.8% at +1V).

SCOPE / remaining (see docs/roadmap_phase5_stretch.md): builds a STACKED (full-cell,
no-inclusion) geometry; a general Design -> gmsh builder (lateral inclusions, arbitrary
electrodes, shared optics lateral extent/region naming) is the last pipeline piece.

gmsh notes: its OCC kernel cannot build at 1e-9-metre scale, so the geometry is
built in NM and the mesh emitted SCALED to metres (Mesh.ScalingFactor); DEVSIM
reads MSH 2.2.
"""
from __future__ import annotations

import os
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
    mesh_min_nm:     float = 0.5         # near the semi/oxide interface
    mesh_max_nm:     float = 3.0
    grid_n:          Tuple[int, int, int] = (16, 16, 33)   # (nx, ny, nz) output grid


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
        return [RegionInfo(name="semi", role="semiconductor", material=s.semi_material,
                            bbox_m=(0.0, s.lateral_m, 0.0, s.lateral_m, *self._z_semi),
                            ndim=3)]

    def build_device(self) -> None:
        import devsim as ds
        self._build_mesh()
        ds.create_gmsh_mesh(mesh=self.mesh_name, file=self.msh_path)
        ds.add_gmsh_region(mesh=self.mesh_name, gmsh_name="semi", region="semi",
                            material=self.spec.semi_material)
        ds.add_gmsh_region(mesh=self.mesh_name, gmsh_name="oxide", region="oxide",
                            material=self.spec.oxide_material)
        ds.add_gmsh_contact(mesh=self.mesh_name, gmsh_name="gate", region="oxide",
                              name="gate", material="metal")
        ds.add_gmsh_contact(mesh=self.mesh_name, gmsh_name="body", region="semi",
                              name="body", material="metal")
        ds.add_gmsh_interface(mesh=self.mesh_name, gmsh_name="semi_oxide",
                                region0="semi", region1="oxide", name="si_ox")
        ds.finalize_mesh(mesh=self.mesh_name)
        ds.create_device(mesh=self.mesh_name, device=self.device)
        s = self.spec
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
        PE.setup_dielectric_region(self.device, "oxide", s.eps_oxide)
        for itf in ds.get_interface_list(device=self.device):
            PE.setup_interface(self.device, itf)
        for c in ds.get_contact_list(device=self.device):
            # the "body" contact is on the semiconductor; for DD it must also pin the
            # electron quasi-Fermi level. "gate" is on the oxide (Potential only).
            if self._dd and c == "body":
                DD.setup_contact_ohmic_dd(self.device, c)
            else:
                PE.setup_contact(self.device, c)
        self._built = True

    def solve(self, bias) -> CarrierField:
        import devsim as ds
        if not self._built:
            self.build_device()
        vg = float(bias.voltages.get("gate", 0.0))
        vb = float(bias.voltages.get("body", 0.0))
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
        g = lambda nm: np.array(ds.get_node_model_values(device=self.device,
                                                          region="semi", name=nm))
        x, y, z = g("x"), g("y"), g("z")
        n, pot = g("Electrons"), g("Potential")
        nodes = np.column_stack([x, y, z])
        grid = resample_to_grid(nodes, {ELECTRON_DENSITY: n, POTENTIAL: pot},
                                  self.spec.grid_n)            # ndim-general resampler
        reg = CarrierRegion(
            name="semi", role="semiconductor", material=self.spec.semi_material,
            nodes_m=nodes, node_fields={ELECTRON_DENSITY: n, POTENTIAL: pot},
            grid_axes_m={"x": grid["axis_0"], "y": grid["axis_1"], "z": grid["axis_2"]},
            grid_fields={ELECTRON_DENSITY: grid[ELECTRON_DENSITY], POTENTIAL: grid[POTENTIAL]})
        return CarrierField(
            bias_label=bias.label, voltages=dict(bias.voltages), ndim=3,
            temperature_K=PE.T_REF, regions={"semi": reg},
            n_bg_by_region={"semi": self.spec.n_bg_m3},
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
        import gmsh
        s = self.spec
        Lnm = s.lateral_m * 1e9
        tsemi, tox = s.semi_thk_m * 1e9, s.oxide_thk_m * 1e9
        os.makedirs(os.path.dirname(self.msh_path), exist_ok=True)
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("ms3d")
        occ = gmsh.model.occ
        vb = occ.addBox(0, 0, 0, Lnm, Lnm, tsemi)
        vt = occ.addBox(0, 0, tsemi, Lnm, Lnm, tox)
        occ.synchronize()
        occ.fragment([(3, vb)], [(3, vt)])
        occ.synchronize()
        for dim, tag in gmsh.model.getEntities(3):
            zc = occ.getCenterOfMass(dim, tag)[2]
            gmsh.model.addPhysicalGroup(3, [tag], name=("semi" if zc < tsemi else "oxide"))
        gate, body, iface = [], [], []
        for dim, tag in gmsh.model.getEntities(2):
            zc = occ.getCenterOfMass(dim, tag)[2]
            if abs(zc - (tsemi + tox)) < 1e-4: gate.append(tag)
            elif abs(zc) < 1e-4: body.append(tag)
            elif abs(zc - tsemi) < 1e-4: iface.append(tag)
        gmsh.model.addPhysicalGroup(2, gate, name="gate")
        gmsh.model.addPhysicalGroup(2, body, name="body")
        gmsh.model.addPhysicalGroup(2, iface, name="semi_oxide")
        fd = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(fd, "SurfacesList", iface)
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
