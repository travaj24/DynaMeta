"""
Schrodinger-Poisson CarrierSolver: a quantum-corrected alternative to the classical
DEVSIM Stage-1, for the degenerate ITO accumulation layer. Wraps the validated
`SchrodingerPoisson1D` (BenDaniel-Duke + degenerate 2D filling + Trellakis self-
consistency) and emits a `CarrierField(ndim=3)` the bridge consumes via IdentityLift
(laterally uniform: the through-stack quantum profile broadcast over the cell -- the
right first-order model for the vertically-gated accumulation layer).

Degenerate-bulk handling: E_F is set from the bulk 3D degenerate relation
E_F - E_c = (hbar^2/2m*)(3 pi^2 n_bg)^(2/3), and the sub-band rejection is disabled
(bound_tol=1e9) so ALL sub-bands up to E_F are kept -- they carry the bulk continuum
of a degenerate semiconductor (rejecting them, as for an isolated well, collapses the
bulk density to ~0). Validated to recover n_bg in the bulk.

The QUANTUM signature vs the classical solve: the accumulation density peak is
displaced ~1 nm from the oxide interface (the quantum "dead layer"), where the
classical Poisson/DD peaks AT the interface -- this shifts the ENZ-region eps profile.

Surface potential: `surface_potential_of_gate(Vg)` maps the gate voltage to the
semiconductor surface potential at the oxide interface (default: identity, i.e. the
full gate drop -- a simplification; supply a callable folding the oxide capacitance
voltage division for quantitative device matching).
"""

from __future__ import annotations

from typing import Callable, List, Optional

import numpy as np

from dynameta.core.carrier_field import (
    CarrierField, CarrierRegion, ELECTRON_DENSITY, POTENTIAL)
from dynameta.core.interfaces import RegionInfo
from dynameta.carriers.schrodinger_poisson import SchrodingerPoisson1D, HBAR, M_E, Q


class SchrodingerPoissonCarrier:
    """A quantum (Schrodinger-Poisson) CarrierSolver over a laterally-uniform
    semiconductor slab. Implements regions() + solve(bias), emitting CarrierField(ndim=3).
    bias.voltages: {"gate": Vg, "body": 0}."""

    def __init__(self, *, semi_thk_m: float = 12e-9, n_bg_m3: float = 4e26,
                 m_eff_kg: float = 0.35 * M_E, eps_static: float = 9.5,
                 T_K: float = 300.0, lateral_m: float = 12e-9, semi_material: str = "ITO",
                 nz: int = 601, n_lateral: int = 4, n_states: int = 80,
                 surface_potential_of_gate: Optional[Callable[[float], float]] = None) -> None:
        self.semi_thk_m = float(semi_thk_m)
        self.n_bg_m3 = float(n_bg_m3)
        self.m_eff_kg = float(m_eff_kg)
        self.eps_static = float(eps_static)
        self.T_K = float(T_K)
        self.lateral_m = float(lateral_m)
        self.semi_material = semi_material
        self.nz = int(nz)
        self.n_lateral = int(n_lateral)
        self.n_states = int(n_states)
        self._psi_s = surface_potential_of_gate or (lambda vg: vg)
        # bulk degenerate Fermi level (relative to the conduction-band edge E_c = 0)
        self.E_F_J = (HBAR ** 2 / (2.0 * self.m_eff_kg)) * (3.0 * np.pi ** 2 * self.n_bg_m3) ** (2.0 / 3.0)

    # ---- CarrierSolver Protocol ----
    def regions(self) -> List[RegionInfo]:
        L, t = self.lateral_m, self.semi_thk_m
        return [RegionInfo(name="semi", role="semiconductor", material=self.semi_material,
                            bbox_m=(0.0, L, 0.0, L, 0.0, t), ndim=3)]

    def solve(self, bias) -> CarrierField:
        vg = float(bias.voltages.get("gate", 0.0))
        psi_s = float(self._psi_s(vg))                    # surface potential at the oxide side
        z = np.linspace(0.0, self.semi_thk_m, self.nz)    # z=0 body, z=t gate/oxide interface
        sp = SchrodingerPoisson1D(z, self.m_eff_kg, T_K=self.T_K)
        Nd = np.full_like(z, self.n_bg_m3)
        # phi=0 at body (z=0), psi_s at the gate side (z=t): +psi_s accumulates electrons.
        phi, n_z, _res = sp.solve_self_consistent(
            eps_r=self.eps_static, doping_m3=Nd, E_F_J=self.E_F_J,
            phi_left_V=0.0, phi_right_V=psi_s, n_states=self.n_states,
            bound_tol=1e9, max_outer=80, tol_V=1e-5)        # slab mode: keep all sub-bands

        # broadcast the through-stack quantum profile over the (x, y) cell (laterally uniform)
        xs = np.linspace(0.0, self.lateral_m, self.n_lateral)
        ys = np.linspace(0.0, self.lateral_m, self.n_lateral)
        nx, ny, nz = xs.size, ys.size, z.size
        n3d = np.broadcast_to(n_z[None, None, :], (nx, ny, nz)).copy()
        pot3d = np.broadcast_to(phi[None, None, :], (nx, ny, nz)).copy()
        X, Y, Z = np.meshgrid(xs, ys, z, indexing="ij")
        nodes = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
        node_fields = {ELECTRON_DENSITY: n3d.ravel(), POTENTIAL: pot3d.ravel()}
        reg = CarrierRegion(
            name="semi", role="semiconductor", material=self.semi_material,
            nodes_m=nodes, node_fields=node_fields,
            grid_axes_m={"x": xs, "y": ys, "z": z},
            grid_fields={ELECTRON_DENSITY: n3d, POTENTIAL: pot3d})
        return CarrierField(
            bias_label=bias.label, voltages=dict(bias.voltages), ndim=3,
            temperature_K=self.T_K, regions={"semi": reg},
            n_bg_by_region={"semi": self.n_bg_m3},
            unit_cell_m=(self.lateral_m, self.lateral_m),
            extras={"quantum": True, "E_F_eV": self.E_F_J / Q,
                    "surface_potential_V": psi_s})

    def teardown(self) -> None:
        pass
