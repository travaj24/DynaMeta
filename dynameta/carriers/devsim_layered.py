"""
Default Stage-1 builder: a 2D (x, z) DEVSIM device from the layered Design,
implementing the core CarrierSolver Protocol.

Region rule (generalizes the old patch-specific build):
  - A layer's BACKGROUND is meshed as a DC region unless its material is the
    ambient (super/substrate) material -- e.g. the air above a patch is skipped.
  - Each INCLUSION is meshed over its x-extent as its own region.
  - Metals are meshed but INERT (no equation) so their Dirichlet contacts can
    realize (this session's finding); dielectrics/semiconductors get physics.
Contacts:
  - edge-footprint electrodes -> thin x-slab at the cell edge, full layer
    z-range, on the layer's region (the ITO peripheral grounds).
  - metal-gate electrodes -> slab spanning the gate metal's z-extent over the
    footprint x-range, attached to the nearest meshed non-metal neighbour.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import devsim as ds

from dynameta.core.carrier_field import (
    CarrierField, CarrierRegion, ELECTRON_DENSITY, POTENTIAL)
from dynameta.core.interfaces import RegionInfo
from dynameta.core.resample import resample_to_grid
from dynameta.carriers import physics_equilibrium as PE
from dynameta.carriers import physics_drift_diffusion as DD
from dynameta.carriers import eq_registry as _R
from dynameta.carriers.dc_solve import solve_dc
from dynameta.geometry.design import Design
from dynameta.geometry.electrode import Electrode


@dataclass
class _RegionSpec:
    name:     str
    material: str
    role:     str            # metal | dielectric | semiconductor
    x_lo:     float
    x_hi:     float
    z_lo:     float
    z_hi:     float


class LayeredDevsimBuilder:
    def __init__(self, design: Design, *, mesh_name: str = "ms_mesh",
                  device_name: str = "ms_device") -> None:
        self.design = design
        self.mesh_name = mesh_name
        self.device = device_name
        self._specs: List[_RegionSpec] = self._region_specs()
        self._contact_region: Dict[str, str] = {}
        self._built = False

    # ---- region planning ----
    def _ambient(self) -> set:
        s = self.design.stack
        return {s.superstrate_material, s.substrate_material}

    def _region_specs(self) -> List[_RegionSpec]:
        d = self.design
        P = d.unit_cell.period_x_m
        z_iv = d.z_intervals()
        ambient = self._ambient()
        specs: List[_RegionSpec] = []
        for L in d.stack.layers:
            zlo, zhi = z_iv[L.name]
            bg_role = d.material_role(L.background_material)
            bg_ambient = L.background_material in ambient
            n_incl = len(L.inclusions)
            if not bg_ambient:
                # background region over the full cell (x split by inclusions is
                # ignored for the background's DC role; inclusions overlay it)
                specs.append(_RegionSpec(L.name, L.background_material, bg_role,
                                          0.0, P, zlo, zhi))
            for i, inc in enumerate(L.inclusions):
                xlo, xhi, _, _ = inc.shape.bbox_m()
                role = d.material_role(inc.material)
                # name: if background is ambient and a single inclusion, the
                # inclusion *is* the layer's region (keeps names like "patch")
                name = L.name if (bg_ambient and n_incl == 1) \
                    else "{}__incl{}".format(L.name, i)
                specs.append(_RegionSpec(name, inc.material, role,
                                          xlo, xhi, zlo, zhi))
        return specs

    def _meshed(self) -> List[_RegionSpec]:
        return self._specs

    def _nearest_nonmetal(self, layer_name: str) -> Tuple[str, float]:
        """For a metal-gate electrode on `layer_name`, find the nearest meshed
        non-metal region and the z where the gate touches it."""
        d = self.design
        names = [L.name for L in d.stack.layers]
        idx = names.index(layer_name)
        z_iv = d.z_intervals()
        nonmetal = {s.name: s for s in self._specs if s.role != "metal"}
        for j in range(idx + 1, len(names)):                 # search upward
            if names[j] in nonmetal:
                return names[j], z_iv[names[j]][0]
        for j in range(idx - 1, -1, -1):                     # search downward
            if names[j] in nonmetal:
                return names[j], z_iv[names[j]][1]
        raise ValueError("gate layer '{}' has no meshed non-metal neighbour"
                          .format(layer_name))

    # ---- mesh build ----
    def build_device(self) -> None:
        d = self.design
        spec = d.mesh_2d
        P = d.unit_cell.period_x_m
        z_iv = d.z_intervals()
        ds.create_2d_mesh(mesh=self.mesh_name)

        # x mesh lines: cell edges + inclusion feature edges + cell midline
        x_lines = {0.0: spec.x_spacing_edge_m, P: spec.x_spacing_edge_m,
                    P / 2.0: spec.x_spacing_feature_mid_m}
        for L in d.stack.layers:
            for inc in L.inclusions:
                xlo, xhi, _, _ = inc.shape.bbox_m()
                x_lines[xlo] = spec.x_spacing_feature_edge_m
                x_lines[xhi] = spec.x_spacing_feature_edge_m
        for pos in sorted(x_lines):
            ds.add_2d_mesh_line(mesh=self.mesh_name, dir="x", pos=pos,
                                  ns=x_lines[pos], ps=x_lines[pos])

        # y mesh lines: interface-zone refinement over meshed layers
        izone = spec.interface_zone_m
        meshed_layer_names = sorted({s.name.split("__incl")[0] for s in self._specs},
                                     key=lambda nm: z_iv[nm][0])
        for nm in meshed_layer_names:
            zlo, zhi = z_iv[nm]
            thk = zhi - zlo
            ds.add_2d_mesh_line(mesh=self.mesh_name, dir="y", pos=zlo,
                                  ns=spec.interface_ps_m, ps=spec.interface_ps_m)
            zone = min(izone, 0.45 * thk)
            if zone > 0:
                ds.add_2d_mesh_line(mesh=self.mesh_name, dir="y", pos=zlo + zone,
                                      ns=spec.coarse_ps_m, ps=spec.coarse_ps_m)
                ds.add_2d_mesh_line(mesh=self.mesh_name, dir="y", pos=zhi - zone,
                                      ns=spec.coarse_ps_m, ps=spec.coarse_ps_m)
            if thk > 4 * izone:
                ds.add_2d_mesh_line(mesh=self.mesh_name, dir="y", pos=0.5 * (zlo + zhi),
                                      ns=spec.coarse_ps_m, ps=spec.coarse_ps_m)
        top_z = max(z_iv[nm][1] for nm in meshed_layer_names)
        ds.add_2d_mesh_line(mesh=self.mesh_name, dir="y", pos=top_z,
                              ns=spec.interface_ps_m, ps=spec.interface_ps_m)

        # regions
        for s in self._specs:
            ds.add_2d_region(mesh=self.mesh_name, region=s.name, material=s.material,
                              xl=s.x_lo, xh=s.x_hi, yl=s.z_lo, yh=s.z_hi, bloat=1e-12)

        # interfaces between z-adjacent meshed regions sharing x-overlap. ONLY
        # between regions that both carry Potential (skip metals -- they are
        # inert with no Potential solution, so a continuity interface would fail;
        # their boundaries are handled by Dirichlet contacts instead).
        for a in self._specs:
            for b in self._specs:
                if a.name >= b.name:
                    continue
                if a.role == "metal" or b.role == "metal":
                    continue
                if abs(a.z_hi - b.z_lo) < 1e-15 or abs(b.z_hi - a.z_lo) < 1e-15:
                    xl = max(a.x_lo, b.x_lo); xh = min(a.x_hi, b.x_hi)
                    if xh <= xl:
                        continue
                    z_if = a.z_hi if abs(a.z_hi - b.z_lo) < 1e-15 else b.z_hi
                    r0, r1 = (a.name, b.name) if a.z_hi <= b.z_lo + 1e-15 else (b.name, a.name)
                    ds.add_2d_interface(mesh=self.mesh_name,
                                          name="{}__{}".format(r0, r1),
                                          region0=r0, region1=r1,
                                          xl=xl, xh=xh, yl=z_if, yh=z_if, bloat=1e-12)

        # contacts
        for E in d.electrodes:
            zlo, zhi = z_iv[E.layer]
            if E.is_edge:
                region = E.layer
                # Thin x-slab at the cell edge, full layer z-range. NOTE: DEVSIM
                # captures only ~2 box-corner nodes here -- full-edge lateral
                # capture is impossible at a domain boundary (it needs an adjacent
                # region, as a horizontal-face contact like bot_contact has). This
                # weak 2-node carrier pin is fine for the equilibrium solve (n is
                # local) but is why gated drift-diffusion does not converge: the
                # continuity equation needs strong carrier pinning, which a lateral
                # edge ground cannot provide in DEVSIM.
                xlo = (0.0 - 1e-10) if E.footprint == "x_lo" else (P - 1e-10)
                xhi = (0.0 + 1e-10) if E.footprint == "x_lo" else (P + 1e-10)
                yl_c, yh_c = zlo, zhi
            else:
                # metal gate: footprint x-range + nearest non-metal neighbour
                region, ybnd = self._nearest_nonmetal(E.layer)
                if E.footprint == "full":
                    xlo, xhi = 0.0, P
                else:
                    bx = E.footprint.bbox_m()
                    xlo, xhi = bx[0], bx[1]
                yl_c, yh_c = min(ybnd, zlo), max(ybnd, zhi)
            ds.add_2d_contact(mesh=self.mesh_name, name=E.name, material="metal",
                                region=region, xl=xlo, xh=xhi, yl=yl_c, yh=yh_c,
                                bloat=1e-10)
            self._contact_region[E.name] = region

        ds.finalize_mesh(mesh=self.mesh_name)
        ds.create_device(mesh=self.mesh_name, device=self.device)

        # per-region physics. Track drift-diffusion semiconductor regions so
        # their ohmic contacts also pin the electron quasi-Fermi level.
        self._dd_regions = set()
        for s in self._specs:
            if s.role == "semiconductor":
                tr = self.design.materials.get(s.material).transport
                dos = float(tr.dos_mass_kg_of_n_m3(tr.n_bg_m3))
                if tr.physics == "drift_diffusion":
                    # The 2D-layered DD path is NOT validated for a GATED capacitor with
                    # weak edge-only ohmic grounds -- it is ill-conditioned and may not
                    # converge (physics_drift_diffusion KNOWN LIMITATION). Equilibrium is
                    # the validated tool for DC gate accumulation (audit F4).
                    warnings.warn(
                        "layered drift-diffusion on semiconductor '{}': a gated device with "
                        "weak edge-only ohmic grounds may not converge; use the equilibrium "
                        "physics mode for DC gate accumulation.".format(s.name))
                    DD.setup_semiconductor_region_dd(
                        self.device, s.name, n_bg_m3=tr.n_bg_m3,
                        eps_static=tr.eps_static, dos_mass_kg=dos,
                        mobility_m2Vs=float(tr.mobility_m2Vs_of_n_m3(tr.n_bg_m3)))
                    self._dd_regions.add(s.name)
                else:
                    PE.setup_semiconductor_region(
                        self.device, s.name, n_bg_m3=tr.n_bg_m3,
                        eps_static=tr.eps_static, dos_mass_kg=dos)
            elif s.role == "dielectric":
                eps_r = self._dielectric_eps_static(s.material)
                PE.setup_dielectric_region(self.device, s.name, eps_r)
            # metals: inert, no setup
        for iface in ds.get_interface_list(device=self.device):
            PE.setup_interface(self.device, iface)
        for c in ds.get_contact_list(device=self.device):
            if self._contact_region.get(c) in self._dd_regions:
                DD.setup_contact_ohmic_dd(self.device, c)   # pin Potential + Electrons (=N_D)
            else:
                PE.setup_contact(self.device, c)            # pin Potential only
        self._built = True

    def _dielectric_eps_static(self, material_name: str) -> float:
        mat = self.design.materials.get(material_name)
        eps_dc = mat.dc_permittivity()
        if eps_dc is None:
            # The old code fell back to the OPTICAL eps with only a printed warning --
            # WRONG for a gate dielectric (HfO2 optical ~4 vs DC ~18) and a silent-physics
            # path. RAISE instead, matching the 3D builder (audit F4 / cross-cutting F4).
            raise ValueError(
                "dielectric '{}' has no eps_static_dc; the Stage-1 Poisson solve needs the "
                "DC permittivity (the optical eps would under-predict gate accumulation). "
                "Set Material.eps_static_dc.".format(material_name))
        return float(eps_dc)

    # ---- CarrierSolver Protocol ----
    def regions(self) -> List[RegionInfo]:
        out = []
        for s in self._specs:
            out.append(RegionInfo(name=s.name, role=s.role, material=s.material,
                                    bbox_m=(s.x_lo, s.x_hi, s.x_lo, s.x_hi, s.z_lo, s.z_hi),
                                    ndim=2))
        return out

    def solve(self, bias, *, grid_n_x: int = 256, grid_n_z: int = 32,
                rel_tol: float = 1e-5, max_iter: int = 60,
                v_step: float = 0.25, abs_tol: float = None,
                verbose: bool = False) -> CarrierField:
        if not self._built:
            self.build_device()
        d = self.design
        method = d.mesh_2d.dc_method
        semi = sorted(self._dd_regions)
        abs_tol = self._dc_abs_tol() if abs_tol is None else abs_tol
        # zero-bias seed (grounds at fixed_voltage, biased at 0)
        for E in d.electrodes:
            v = E.fixed_voltage_V if E.role == "ground" else 0.0
            ds.set_parameter(device=self.device, name="{}_bias".format(E.name), value=v)
        solve_dc(self.device, method=method, abs_tol=abs_tol, rel_tol=rel_tol,
                  max_iter=max_iter, semiconductor_regions=semi, verbose=verbose)
        # ramp biased electrodes to their target
        for E in d.electrodes:
            target = bias.voltages.get(E.name,
                        E.fixed_voltage_V if E.role == "ground" else 0.0)
            v_now = E.fixed_voltage_V if E.role == "ground" else 0.0
            n_steps = max(1, int(abs(target - v_now) / v_step + 0.5))
            dv = (target - v_now) / n_steps
            for _ in range(n_steps):
                v_now += dv
                ds.set_parameter(device=self.device, name="{}_bias".format(E.name),
                                  value=v_now)
                solve_dc(self.device, method=method, abs_tol=abs_tol, rel_tol=rel_tol,
                          max_iter=max_iter, semiconductor_regions=semi, verbose=verbose)
        return self._to_carrier_field(bias, grid_n_x, grid_n_z)

    def _dc_abs_tol(self) -> float:
        """Absolute Newton tolerance, scaled to the carrier density for DD.

        In SI the electron-continuity residual is in carrier-density units
        (~n_bg, ~1e26 m^-3). The absolute Newton update floors near
        n_bg*machine_eps (~1e11 for n_bg~1e26), so the Boltzmann-era abs_tol=1e10
        sits BELOW that precision floor and can never be satisfied -- the solve
        spins to max_iter and raises Convergence failure even though the relative
        update is ~1e-9. Scaling abs_tol to the carrier density puts it safely
        above the floor so rel_tol becomes the binding (meaningful) gate.
        Equilibrium mode (Poisson-only, V-unit residual) keeps the tight 1e10.
        """
        if not self._dd_regions:
            return 1e10
        n_scale = 0.0
        for s in self._specs:
            if s.name in self._dd_regions:
                t = self.design.materials.get(s.material).transport
                n_scale = max(n_scale, float(t.n_bg_m3))
        return max(1e10, n_scale * 1e-12)

    def _to_carrier_field(self, bias, grid_n_x, grid_n_z) -> CarrierField:
        d = self.design
        regions: Dict[str, CarrierRegion] = {}
        n_bg_by_region: Dict[str, float] = {}
        for s in self._specs:
            if s.role != "semiconductor":
                continue
            x = np.array(ds.get_node_model_values(device=self.device, region=s.name, name="x"))
            y = np.array(ds.get_node_model_values(device=self.device, region=s.name, name="y"))
            n = np.array(ds.get_node_model_values(device=self.device, region=s.name, name="Electrons"))
            pot = np.array(ds.get_node_model_values(device=self.device, region=s.name, name="Potential"))
            nodes = np.column_stack([x, y])
            grid = resample_to_grid(nodes, {ELECTRON_DENSITY: n, POTENTIAL: pot},
                                      (grid_n_x, grid_n_z))
            regions[s.name] = CarrierRegion(
                name=s.name, role=s.role, material=s.material, nodes_m=nodes,
                node_fields={ELECTRON_DENSITY: n, POTENTIAL: pot},
                grid_axes_m={"x": grid["axis_0"], "y": grid["axis_1"]},
                grid_fields={ELECTRON_DENSITY: grid[ELECTRON_DENSITY],
                              POTENTIAL: grid[POTENTIAL]})
            n_bg_by_region[s.name] = float(d.materials.get(s.material).transport.n_bg_m3)
        return CarrierField(
            bias_label=bias.label, voltages=dict(bias.voltages), ndim=2,
            temperature_K=PE.T_REF, regions=regions, n_bg_by_region=n_bg_by_region,
            unit_cell_m=(d.unit_cell.period_x_m, d.unit_cell.period_y_m))

    def teardown(self) -> None:
        _R.clear(self.device)
        for dv in list(ds.get_device_list()):
            ds.delete_device(device=dv)
        for m in list(ds.get_mesh_list()):
            ds.delete_mesh(mesh=m)
        self._built = False
