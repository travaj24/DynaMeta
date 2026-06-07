"""
Schrodinger-Poisson CarrierSolver: a quantum-corrected alternative to the classical
DEVSIM Stage-1, for the degenerate ITO accumulation layer. Wraps the validated
`SchrodingerPoisson1D` (BenDaniel-Duke + degenerate 2D filling + Trellakis self-
consistency) and emits a `CarrierField(ndim=3)` the bridge consumes DIRECTLY -- the native
3D-grid branch of assemble_eps places the (x,y,z) density without any FieldLift synthesis (the
lift applies to 2D fields only). The emitted field is laterally uniform (the through-stack
quantum profile broadcast over the cell -- the right first-order model for the vertically-gated
accumulation layer).

Degenerate-bulk handling: E_F is set from the bulk 3D degenerate relation
E_F - E_c = (hbar^2/2m*)(6 pi^2 n_bg/(g_s g_v))^(2/3) (the (3 pi^2 n)^(2/3) special case is the
g_s=2, g_v=1 ITO default; g_s/g_v are threaded to the SP1D fill), and the sub-band rejection is disabled
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

Known limitations (audit SP-3/SP-4):
  - The self-consistent solve is PARABOLIC (m* constant). Kane nonparabolicity exists
    only at the SchrodingerPoisson1D.density() level, not through this CarrierSolver,
    and the bulk E_F here uses the parabolic (3 pi^2 n)^(2/3) relation. Do not treat the
    accumulation magnitude as nonparabolic-accurate for ITO at >1e26 m^-3.
  - The body side (z=0) is a Dirichlet hard wall, so the emitted density carries a small
    (~0.4 nm) unphysical depletion at z=0 at every bias (a real device body is a contact
    into the bulk, not an infinite barrier). The gate-side dead layer IS the physical
    quantum effect; the body-side one is a boundary-condition artifact.
"""

from __future__ import annotations

import warnings
from typing import Callable, List, Optional

import numpy as np

from dynameta.core.carrier_field import (
    CarrierField, CarrierRegion, ELECTRON_DENSITY, POTENTIAL)
from dynameta.core.interfaces import RegionInfo
from dynameta.core.numerics import trapz
from dynameta.carriers.schrodinger_poisson import SchrodingerPoisson1D, HBAR, M_E, Q, EPS0


class SchrodingerPoissonCarrier:
    """A quantum (Schrodinger-Poisson) CarrierSolver over a laterally-uniform
    semiconductor slab. Implements regions() + solve(bias), emitting CarrierField(ndim=3).
    bias.voltages: {"gate": Vg, "body": 0}."""

    def __init__(self, *, semi_thk_m: float = 12e-9, n_bg_m3: float = 4e26,
                 m_eff_kg: float = 0.35 * M_E, eps_static: float = 9.5,
                 T_K: float = 300.0, lateral_m: float = 12e-9, semi_material: str = "ITO",
                 nz: int = 601, n_lateral: int = 4, n_states: int = 80,
                 oxide_thk_m: Optional[float] = None, eps_oxide: float = 18.0,
                 alpha_np_per_eV: float = 0.0, g_s: int = 2, g_v: int = 1,
                 surface_potential_of_gate: Optional[Callable[[float], float]] = None,
                 surface_potential_xy: Optional[Callable[[float, float, float], float]] = None) -> None:
        self.semi_thk_m = float(semi_thk_m)
        self.n_bg_m3 = float(n_bg_m3)
        self.m_eff_kg = float(m_eff_kg)
        self.g_s, self.g_v = int(g_s), int(g_v)        # spin + valley degeneracy (ITO: 2, 1)
        self.eps_static = float(eps_static)
        self.T_K = float(T_K)
        self.lateral_m = float(lateral_m)
        self.semi_material = semi_material
        self.nz = int(nz)
        self.n_lateral = int(n_lateral)
        self.n_states = int(n_states)
        # Kane in-plane nonparabolicity (eV^-1) applied to the emitted density as a
        # POST-HOC nonparabolic 2D fill on the converged (parabolic) self-consistent
        # potential -- captures ITO's DOS-mass flattening where it matters for the optical
        # eps. The self-consistent potential and the bulk E_F stay parabolic (so the bulk
        # density may shift ~by the DOS enhancement); 0 = parabolic (default).
        self.alpha_np = float(alpha_np_per_eV)
        # gate -> semiconductor surface-potential map. Priority:
        #   1. an explicit surface_potential_of_gate callable;
        #   2. else, if a gate oxide is given, the physical series-capacitor division
        #      Vg = psi_s + q*N_excess(psi_s)/C_ox (C_ox = eps_ox*eps0/t_ox) -- solved
        #      for psi_s. This is the CALIBRATED map: most of Vg drops across the oxide
        #      once the channel accumulates, so psi_s << Vg (the old identity map
        #      psi_s=Vg grossly over-estimated the accumulation).
        #   3. else identity psi_s=Vg (qualitative only; documented over-estimate).
        self._C_ox = (eps_oxide * EPS0 / float(oxide_thk_m)) if oxide_thk_m else None
        if surface_potential_of_gate is not None:
            self._psi_s = surface_potential_of_gate
        elif self._C_ox is not None:
            self._psi_s = self._gate_to_psi_s
        else:
            self._psi_s = lambda vg: vg
        # optional LATERAL surface-potential map psi_s(x_m, y_m, Vg) -> V for a
        # laterally-VARYING device (e.g. under a patch vs the gap). When given, the
        # solver runs a 1D SP per lateral column (caching by psi_s value); when None it
        # is laterally uniform (the through-stack profile broadcast over the cell).
        self._psi_xy = surface_potential_xy
        # SP-5: the per-column path uses _psi_xy directly and does NOT apply the oxide
        # series-cap division, so supplying both is ambiguous -- warn that psi_xy is taken
        # as the FINAL surface potential (it must already include any oxide division).
        if self._psi_xy is not None and self._C_ox is not None:
            warnings.warn(
                "SchrodingerPoissonCarrier: surface_potential_xy is used as the final "
                "per-column surface potential; the oxide voltage division (oxide_thk_m/"
                "eps_oxide) is NOT applied to it. Fold any oxide division into psi_xy.",
                stacklevel=2)
        # bulk degenerate Fermi level (relative to the conduction-band edge E_c = 0). General form
        # k_F = (6 pi^2 n / (g_s g_v))^(1/3); the (3 pi^2 n)^(1/3) special case is g_s=2, g_v=1 (the
        # ITO default). Threading g_s/g_v keeps it consistent with the SchrodingerPoisson1D filling.
        self.E_F_J = (HBAR ** 2 / (2.0 * self.m_eff_kg)) * (
            6.0 * np.pi ** 2 * self.n_bg_m3 / (self.g_s * self.g_v)) ** (2.0 / 3.0)

    # ---- CarrierSolver Protocol ----
    def regions(self) -> List[RegionInfo]:
        L, t = self.lateral_m, self.semi_thk_m
        return [RegionInfo(name="semi", role="semiconductor", material=self.semi_material,
                            bbox_m=(0.0, L, 0.0, L, 0.0, t), ndim=3)]

    def _solve_column(self, sp, Nd, psi_s):
        """One 1D self-consistent SP solve at gate-side surface potential psi_s
        (phi=0 at body z=0, psi_s at the gate/oxide side z=t). Returns (phi, n_z)."""
        from dynameta.carriers.schrodinger_poisson import Q
        phi, n_z, _res = sp.solve_self_consistent(
            eps_r=self.eps_static, doping_m3=Nd, E_F_J=self.E_F_J,
            phi_left_V=0.0, phi_right_V=psi_s, n_states=self.n_states,
            bound_tol=1e9, max_outer=80, tol_V=1e-5)          # slab mode: keep all sub-bands
        if self.alpha_np > 0.0:
            # post-hoc nonparabolic 2D fill on the converged (parabolic) potential
            res_np = sp.density(-Q * phi, self.E_F_J, n_states=self.n_states,
                                 bound_tol=1e9, alpha_np_per_eV=self.alpha_np)
            n_z = np.zeros_like(phi)
            n_z[1:-1] = res_np.density_m3
        return phi, n_z

    def _gate_to_psi_s(self, vg: float) -> float:
        """Resolve the semiconductor surface potential psi_s from the gate voltage via
        the oxide series capacitance: Vg = psi_s + q*N_excess(psi_s)/C_ox, solved by
        bisection. N_excess(psi_s) is the accumulated sheet density RELATIVE TO THE
        FLAT-BAND (psi_s=0) self-consistent profile -- NOT relative to the flat doping
        n_bg. This matters because the hard-wall Dirichlet boundary conditions make the
        flat-band slab itself deficient near the walls (a ~0.4 nm dead layer per wall),
        so baselining against n_bg injected a spurious negative offset that biased psi_s
        several hundred mV high (audit SP-2). With the flat-band baseline, N_excess(0)=0
        exactly, so f(0)=-Vg<0 and f(Vg)>0 genuinely bracket the root; psi_s comes out
        the true accumulation potential (most of Vg drops across a thin high-k oxide)."""
        if self._C_ox is None or abs(vg) < 1e-12:
            return float(vg)
        s = 1.0 if vg > 0 else -1.0
        Vg = abs(float(vg))
        z = np.linspace(0.0, self.semi_thk_m, self.nz)
        sp = SchrodingerPoisson1D(z, self.m_eff_kg, T_K=self.T_K, g_s=self.g_s, g_v=self.g_v)
        Nd = np.full_like(z, self.n_bg_m3)
        _, n0_z = self._solve_column(sp, Nd, 0.0)             # flat-band baseline density

        def residual(psi):
            _, n_z = self._solve_column(sp, Nd, s * psi)
            # excess vs the FLAT-BAND profile (removes the hard-wall dead-layer offset)
            n_exc = trapz(n_z - n0_z, z)
            return psi + Q * n_exc / self._C_ox - Vg          # Vg residual at trial psi_s

        lo, hi = 0.0, Vg
        r_lo, r_hi = residual(lo), residual(hi)   # r_lo = -Vg < 0 (flat-band baseline -> N_exc(0)=0)
        grow = 0
        while r_lo * r_hi > 0.0 and grow < 6:      # depletion root lies beyond psi=Vg: expand bracket
            hi *= 2.0
            r_hi = residual(hi)
            grow += 1
        if r_lo * r_hi > 0.0:                       # never bracketed -> warn, don't silently lie (SP-NEG-1)
            warnings.warn(
                "SchrodingerPoissonCarrier: gate->psi_s bisection could not bracket a root for "
                "Vg={:.3g} V. The oxide series-cap map is calibrated for accumulation; a depletion "
                "bias on a degenerate film may not converge, so the surface potential (hence the "
                "returned density) may be unreliable.".format(vg), stacklevel=2)
        for _ in range(8):                          # ~bracket/256 resolution
            mid = 0.5 * (lo + hi)
            if residual(mid) < 0.0:
                lo = mid
            else:
                hi = mid
        return s * 0.5 * (lo + hi)

    def solve(self, bias) -> CarrierField:
        vg = float(bias.voltages.get("gate", 0.0))
        z = np.linspace(0.0, self.semi_thk_m, self.nz)        # z=0 body, z=t gate/oxide interface
        sp = SchrodingerPoisson1D(z, self.m_eff_kg, T_K=self.T_K, g_s=self.g_s, g_v=self.g_v)
        Nd = np.full_like(z, self.n_bg_m3)
        xs = np.linspace(0.0, self.lateral_m, self.n_lateral)
        ys = np.linspace(0.0, self.lateral_m, self.n_lateral)
        nx, ny, nz = xs.size, ys.size, z.size

        if self._psi_xy is None:
            # laterally uniform: one column broadcast over the cell
            psi_s = float(self._psi_s(vg))
            phi, n_z = self._solve_column(sp, Nd, psi_s)
            n3d = np.broadcast_to(n_z[None, None, :], (nx, ny, nz)).copy()
            pot3d = np.broadcast_to(phi[None, None, :], (nx, ny, nz)).copy()
            psi_extra = {"surface_potential_V": psi_s}
        else:
            # per-column: solve a 1D SP at each lateral psi_s, caching by value (a
            # patch is ~equipotential -> few distinct psi_s -> few solves)
            n3d = np.empty((nx, ny, nz)); pot3d = np.empty((nx, ny, nz))
            cache = {}
            for i, xv in enumerate(xs):
                for j, yv in enumerate(ys):
                    psi_s = float(self._psi_xy(float(xv), float(yv), vg))
                    key = round(psi_s, 4)                     # ~mV resolution
                    if key not in cache:
                        cache[key] = self._solve_column(sp, Nd, psi_s)
                    phi, n_z = cache[key]
                    pot3d[i, j, :] = phi; n3d[i, j, :] = n_z
            keys = sorted(cache)
            psi_extra = {"surface_potential_range_V": [keys[0], keys[-1]],
                          "n_distinct_columns": len(cache), "laterally_varying": True}
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
            extras=dict({"quantum": True, "E_F_eV": self.E_F_J / Q}, **psi_extra))

    def teardown(self) -> None:
        pass
