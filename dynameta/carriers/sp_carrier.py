"""
Schrodinger-Poisson CarrierSolver: a quantum-corrected alternative to the classical
DEVSIM Stage-1, for the accumulation layer of a degenerate semiconductor film. Wraps the validated
`SchrodingerPoisson1D` (BenDaniel-Duke + degenerate 2D filling + Trellakis self-
consistency) and emits a `CarrierField(ndim=3)` the bridge consumes DIRECTLY -- the native
3D-grid branch of assemble_eps places the (x,y,z) density without any FieldLift synthesis (the
lift applies to 2D fields only). The emitted field is laterally uniform (the through-stack
quantum profile broadcast over the cell -- the right first-order model for the vertically-gated
accumulation layer).

Degenerate-bulk handling: E_F is the Fermi level at which the SP model's OWN flat-band bulk holds n_bg
(`fermi_level="model"`, the default): the root of n_sep(E_F) = n_bg with
n_sep = (1/pi) Int dk_z pref0 [kT F_0(eta) + 2 alpha kT^2 F_1(eta)] -- parabolic z dispersion, in-plane
Kane DOS, finite temperature (SchrodingerPoisson1D.bulk_fermi_level; at alpha = 0 it is the parabolic
N_c F_1/2(E_F/kT) = n_bg). `fermi_level="bulk3d_T0"` keeps the pre-0.16 T = 0 3D relation
E_F - E_c = (hbar^2/2m*)(6 pi^2 n_bg/(g_s g_v))^(2/3), Kane-inverted for alpha > 0; it is not the model's
bulk limit (a 40 nm flat-band slab then holds 1.0029 n_bg at alpha = 0 and 0.896 n_bg at alpha = 0.4 eV^-1
for 3e26 m^-3, 0.30 m_e, 300 K, and the self-consistent solve restores n_bg with a spurious band bending of
-1.0 mV / +32 mV). The sub-band rejection is disabled (bound_tol=1e9) so ALL sub-bands up to E_F are kept
-- they carry the bulk continuum of a degenerate semiconductor (rejecting them, as for an isolated well,
collapses the bulk density to ~0).

The QUANTUM signature vs the classical solve: the accumulation density peak is
displaced ~1 nm from the oxide interface (the quantum "dead layer"), where the
classical Poisson/DD peaks AT the interface -- this shifts the near-interface eps profile.

Surface potential (gate -> psi_s map), in priority order: an explicit `surface_potential_of_gate(Vg)`
callable; else, with a gate oxide (oxide_thk_m, eps_oxide), Gauss's law at the oxide face selected by
`gate_map`; else the identity psi_s = Vg (the full gate drop -- qualitative only).
  - `gate_map="face_field"` (default): Vg = psi_s + F_face(psi_s)/C_ox, C_ox = eps_ox eps0/t_ox, with
    F_face = eps_s eps0 dphi/dz at the oxide face from the conservative flux of the solver's own Poisson
    stencil (schrodinger_poisson.sp_face_flux_C_m2). This is the continuity of the normal displacement
    at a charge-free semiconductor/oxide interface (no work-function difference: Vg is the gate's
    potential relative to the body). It includes the field of the gate-side quantum dead layer, so
    psi_s != 0 at Vg = 0.
  - `gate_map="flat_band_excess"`: Vg = psi_s + q N_exc(psi_s)/C_ox with N_exc the sheet excess RELATIVE
    TO THE psi_s = 0 PROFILE (signed). It is the face-field law with the flat-band face field and the
    change of the body-side flux both dropped, i.e. it assumes the dead-layer field at psi_s = 0 ends
    on the body; psi_s = 0 at Vg = 0 by construction.
  Both maps solve their residual for the SIGNED psi_s by brentq (xtol 1e-9 V).

Capabilities and limitations (docstring corrected per audit C6-5 -- the old text
claimed parabolic-only for code that is fully nonparabolic):
  - With alpha_np_per_eV > 0 the self-consistent solve IS fully NONPARABOLIC in-plane: the
    Trellakis Newton's a-priori density + Jacobian use the Kane DOS through
    _solve_column, and the bulk E_F is calibrated on the same band. The z dispersion is parabolic
    (mass m_eff_kg) throughout.
  - The body side (z=0) is a Dirichlet hard wall BY DEFAULT, so the emitted density
    carries a small (~0.4 nm) unphysical depletion at z=0 at every bias (a real device
    body is a contact into the bulk, not an infinite barrier); neumann_body=True removes
    it. The gate-side dead layer IS the physical quantum effect. Only ONE gate is modelled; the
    body is a fixed-potential reference.
"""

from __future__ import annotations

import warnings
from typing import Callable, List, Optional

import numpy as np

from dynameta.core.carrier_field import (
    CarrierField, CarrierRegion, ELECTRON_DENSITY, POTENTIAL)
from dynameta.core.interfaces import RegionInfo
from dynameta.core.numerics import trapz
from dynameta.carriers.schrodinger_poisson import (
    SchrodingerPoisson1D, Q, EPS0, bulk_fermi_level_3d_T0_J, sp_bulk_fermi_level_J,
    sp_face_flux_C_m2)
from dynameta.warning_types import AdvisoryWarning, UnreliableResultWarning
from dynameta.constants import T_REF   # audit X-24: one reference temperature

_FERMI_LEVELS = ("model", "bulk3d_T0")
_GATE_MAPS = ("face_field", "flat_band_excess")
_PSI_XTOL_V = 1e-9          # brentq tolerance on psi_s for both gate maps


class SchrodingerPoissonCarrier:
    """A quantum (Schrodinger-Poisson) CarrierSolver over a laterally-uniform
    semiconductor slab. Implements regions() + solve(bias), emitting CarrierField(ndim=3).
    bias.voltages: {"gate": Vg, "body": 0}.

    The film's material parameters have no defaults: `n_bg_m3` (background density [m^-3]),
    `m_eff_kg` (band-edge effective mass [kg]) and `eps_static` (static relative permittivity) are
    required keywords, and `eps_oxide` is required whenever `oxide_thk_m` is given. For a preset film,
    `materials.tco_presets.dual_gate_film_kwargs(material)` returns the first three plus its
    `alpha_np_per_eV`, ready to pass as `**kwargs`."""

    def __init__(self, *, n_bg_m3: float, m_eff_kg: float, eps_static: float,
                 semi_thk_m: float = 12e-9,
                 T_K: float = T_REF, lateral_m: float = 12e-9, semi_material: str = "film",
                 nz: int = 601, n_lateral: int = 4, n_states: int = 80,
                 oxide_thk_m: Optional[float] = None, eps_oxide: Optional[float] = None,
                 alpha_np_per_eV: float = 0.0, g_s: int = 2, g_v: int = 1,
                 neumann_body: bool = False,
                 surface_potential_of_gate: Optional[Callable[[float], float]] = None,
                 surface_potential_xy: Optional[Callable[[float, float, float], float]] = None,
                 fermi_level: str = "model", gate_map: str = "face_field") -> None:
        if fermi_level not in _FERMI_LEVELS:
            raise ValueError("SchrodingerPoissonCarrier: fermi_level must be one of {}, got "
                             "{!r}".format(_FERMI_LEVELS, fermi_level))
        if gate_map not in _GATE_MAPS:
            raise ValueError("SchrodingerPoissonCarrier: gate_map must be one of {}, got "
                             "{!r}".format(_GATE_MAPS, gate_map))
        if oxide_thk_m and eps_oxide is None:
            raise ValueError("SchrodingerPoissonCarrier: eps_oxide (the gate oxide's static relative "
                             "permittivity) is required with oxide_thk_m; there is no material default")
        self.fermi_level = fermi_level
        self.gate_map = gate_map
        self.semi_thk_m = float(semi_thk_m)
        self.n_bg_m3 = float(n_bg_m3)
        self.m_eff_kg = float(m_eff_kg)
        self.g_s, self.g_v = int(g_s), int(g_v)        # spin + valley degeneracy (single-valley oxide: 2, 1)
        self.eps_static = float(eps_static)
        self.T_K = float(T_K)
        self.lateral_m = float(lateral_m)
        self.semi_material = semi_material
        self.nz = int(nz)
        self.n_lateral = int(n_lateral)
        self.n_states = int(n_states)
        # Kane in-plane nonparabolicity (eV^-1) applied to the emitted density as a
    # alpha_np > 0 makes the SELF-CONSISTENT solve fully Kane-nonparabolic: the Newton's
    # density AND Jacobian use the Kane DOS and the bulk E_F is Kane-inverted (see the
    # class docstring / audit C6-5; the old comment here wrongly described a post-hoc
    # parabolic re-fill -- audit S1-16).
        self.alpha_np = float(alpha_np_per_eV)
        # Neumann (zero-flux) BODY boundary at z=0 for the Schrodinger solve: psi need not vanish at
        # the body (a contact into the bulk, not an infinite wall), removing the ~0.4nm Dirichlet
        # dead layer; the Poisson keeps the Dirichlet body reference so n[body] is the bulk n_bg. (A
        # hard Neumann wall has its own boundary pile-up; both are far from the gate-side ENZ.)
        self.neumann_body = bool(neumann_body)
        # gate -> semiconductor surface-potential map. Priority:
        #   1. an explicit surface_potential_of_gate callable;
        #   2. else, if a gate oxide is given, Gauss's law at the oxide face (C_ox = eps_ox*eps0/t_ox),
        #      solved for psi_s: gate_map="face_field" Vg = psi_s + F_face(psi_s)/C_ox (default) or
        #      gate_map="flat_band_excess" Vg = psi_s + q*N_exc(psi_s)/C_ox (see the module docstring).
        #      Most of Vg drops across the oxide once the channel accumulates, so psi_s << Vg.
        #   3. else identity psi_s=Vg (qualitative only; documented over-estimate).
        self._C_ox = (float(eps_oxide) * EPS0 / float(oxide_thk_m)) if oxide_thk_m else None
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
                "eps_oxide) is NOT applied to it. Fold any oxide division into psi_xy.", AdvisoryWarning,
                stacklevel=2)
        # bulk Fermi level (relative to the conduction-band edge E_c = 0). "model": the level at which
        # the SP model's own flat-band bulk (parabolic z, in-plane Kane, finite T) holds n_bg -- the one
        # kernel in schrodinger_poisson. "bulk3d_T0": the pre-0.16 T = 0 3D (Kane-inverted) relation.
        if self.fermi_level == "model":
            self.E_F_J = sp_bulk_fermi_level_J(self.n_bg_m3, self.m_eff_kg, self.T_K, g_s=self.g_s,
                                               g_v=self.g_v, alpha_np_per_eV=self.alpha_np)
        else:
            self.E_F_J = bulk_fermi_level_3d_T0_J(self.n_bg_m3, self.m_eff_kg, g_s=self.g_s,
                                                  g_v=self.g_v, alpha_np_per_eV=self.alpha_np)

    # ---- CarrierSolver Protocol ----
    def regions(self) -> List[RegionInfo]:
        L, t = self.lateral_m, self.semi_thk_m
        return [RegionInfo(name="semi", role="semiconductor", material=self.semi_material,
                            bbox_m=(0.0, L, 0.0, L, 0.0, t), ndim=3)]

    def _solve_column(self, sp, Nd, psi_s):
        """One 1D self-consistent SP solve at gate-side surface potential psi_s
        (phi=0 at body z=0, psi_s at the gate/oxide side z=t). Returns (phi, n_z). When
        alpha_np>0 the solve is FULLY self-consistent nonparabolic: the Trellakis Newton's
        a-priori density + Jacobian use the Kane DOS, so the converged potential AND density
        (and the bulk E_F above) all carry the nonparabolicity (no longer a post-hoc re-fill)."""
        phi, n_z, _res = sp.solve_self_consistent(
            eps_r=self.eps_static, doping_m3=Nd, E_F_J=self.E_F_J,
            phi_left_V=0.0, phi_right_V=psi_s, n_states=self.n_states,
            bound_tol=1e9, max_outer=80, tol_V=1e-5,          # slab mode: keep all sub-bands
            alpha_np_per_eV=self.alpha_np, neumann_left=self.neumann_body)
        return phi, n_z

    def _column_setup(self):
        """(z, sp, Nd, col): the column grid, its solver, the doping, and a memoized column solve
        col(psi_s) -> (phi, n_z) (brentq re-evaluates its bracket ends; each SP solve is costly)."""
        z = np.linspace(0.0, self.semi_thk_m, self.nz)
        sp = SchrodingerPoisson1D(z, self.m_eff_kg, T_K=self.T_K, g_s=self.g_s, g_v=self.g_v)
        Nd = np.full_like(z, self.n_bg_m3)
        cache = {}

        def col(psi_s):
            key = float(psi_s)
            if key not in cache:
                cache[key] = self._solve_column(sp, Nd, key)
            return cache[key]
        return z, sp, Nd, col

    def gate_face_flux(self, phi_V: np.ndarray, n_m3: np.ndarray) -> float:
        """eps_s eps0 dphi/dz [C/m^2] at the gate-side (oxide) face of a column profile returned by the
        column solve on this carrier's grid (conservative flux of the solver's Poisson stencil,
        schrodinger_poisson.sp_face_flux_C_m2). Gauss's law at the oxide face reads
        Vg = psi_s + gate_face_flux / C_ox."""
        h = self.semi_thk_m / (self.nz - 1)
        Nd = np.full(self.nz, self.n_bg_m3)
        return sp_face_flux_C_m2(phi_V, n_m3, Nd, h, self.eps_static, side="right")

    def _gate_to_psi_s(self, vg: float) -> float:
        """Resolve the semiconductor surface potential psi_s from the gate voltage by Gauss's law at
        the oxide face, with the map selected by `gate_map` (see the module docstring)."""
        if self._C_ox is None:
            return float(vg)
        if self.gate_map == "flat_band_excess":
            return self._gate_to_psi_s_flat_band(vg)
        return self._gate_to_psi_s_face_field(vg)

    def _gate_to_psi_s_face_field(self, vg: float) -> float:
        """Vg = psi_s + F(psi_s)/C_ox with F = eps_s eps0 dphi/dz at the oxide face (continuity of the
        normal displacement at the charge-free semiconductor/oxide interface; the oxide field is
        (Vg - psi_s)/t_ox). F is the conservative face flux of the solver's own Poisson stencil.

        Bracket: g(psi) = psi + F(psi)/C_ox - Vg has slope >= 1 because F does not decrease when the
        surface potential rises (more accumulation / less depletion). Hence g(x) - g(y) >= x - y, and
        the root lies between 0 and -g(0): one column solve at psi = 0 brackets it exactly. A failed
        sign check (a non-monotone column) widens the bracket symmetrically and, if that fails too,
        warns."""
        from scipy.optimize import brentq
        vg = float(vg)
        _z, _sp, _Nd, col = self._column_setup()
        C = self._C_ox

        def g(psi_s):
            phi, n_z = col(psi_s)
            return psi_s + self.gate_face_flux(phi, n_z) / C - vg

        g0 = g(0.0)
        if g0 == 0.0:
            return 0.0
        a, b = (0.0, -g0) if g0 < 0.0 else (-g0, 0.0)
        ga, gb = (g0, g(b)) if g0 < 0.0 else (g(a), g0)
        w = abs(g0)
        grow = 0
        while not (ga <= 0.0 <= gb) and grow < 6:
            a, b = a - w, b + w
            ga, gb = g(a), g(b)
            w *= 2.0
            grow += 1
        if not (ga <= 0.0 <= gb):
            warnings.warn(
                "SchrodingerPoissonCarrier: the face-field gate map could not bracket psi_s for "
                "Vg={:.4g} V (g({:.4g})={:.3g}, g({:.4g})={:.3g}); the column response is not monotone in "
                "the surface potential, so the returned surface potential (hence the density) is "
                "unreliable.".format(vg, a, ga, b, gb), UnreliableResultWarning, stacklevel=3)
            return float(a if abs(ga) < abs(gb) else b)
        if ga == 0.0:
            return float(a)
        if gb == 0.0:
            return float(b)
        return float(brentq(g, a, b, xtol=_PSI_XTOL_V, maxiter=200))

    def _gate_to_psi_s_flat_band(self, vg: float) -> float:
        """Vg = psi_s + q*N_exc(psi_s)/C_ox for the SIGNED psi_s, N_exc the signed sheet excess
        RELATIVE TO THE FLAT-BAND (psi_s=0) self-consistent profile -- NOT relative to the flat doping
        n_bg (the hard walls make the flat-band slab itself deficient near the walls, so baselining
        against n_bg injected a spurious offset). Written for psi = |psi_s|, s = sign(Vg):
        r(psi) = psi + s q N_exc(s psi)/C_ox - |Vg|. r(0) = -|Vg| < 0 (N_exc(0) = 0), and N_exc(s|Vg|)
        has the sign s (accumulation adds, depletion removes electrons), so r(|Vg|) > 0: [0, |Vg|]
        brackets the root for BOTH signs. (Without the factor s the depletion residual never crossed
        zero there and the bracket expansion converged on a spurious root near -17 V.) The expansion
        is kept for a column whose excess has the wrong sign; if even that fails it warns."""
        from scipy.optimize import brentq
        if abs(vg) < 1e-12:
            return float(vg)
        s = 1.0 if vg > 0 else -1.0
        Vg = abs(float(vg))
        z, _sp, _Nd, col = self._column_setup()
        _, n0_z = col(0.0)                                    # flat-band baseline density

        def residual(psi):
            if psi == 0.0:
                return -Vg                                    # N_exc(0) = 0 exactly
            _, n_z = col(s * psi)
            n_exc = trapz(n_z - n0_z, z)                      # signed excess vs the flat-band profile
            return psi + s * Q * n_exc / self._C_ox - Vg

        lo, hi = 0.0, Vg
        r_lo, r_hi = -Vg, residual(hi)
        grow = 0
        while r_lo * r_hi > 0.0 and grow < 6:      # wrong-sign column: the root lies beyond psi=Vg
            lo, r_lo = hi, r_hi
            hi *= 2.0
            r_hi = residual(hi)
            grow += 1
        if r_lo * r_hi > 0.0:                       # never bracketed -> warn, don't silently lie (SP-NEG-1)
            warnings.warn(
                "SchrodingerPoissonCarrier: gate->psi_s map could not bracket a root for Vg={:.3g} V "
                "(the column's sheet excess does not have the sign of the surface potential), so the "
                "surface potential (hence the returned density) is unreliable.".format(vg),
                UnreliableResultWarning, stacklevel=3)
            return s * (lo if abs(r_lo) < abs(r_hi) else hi)
        if r_hi == 0.0:
            return s * hi
        return s * float(brentq(residual, lo, hi, xtol=_PSI_XTOL_V, maxiter=200))

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
            extras=dict({"quantum": True, "E_F_eV": self.E_F_J / Q, "fermi_level": self.fermi_level,
                         "gate_map": self.gate_map}, **psi_extra))

    def teardown(self) -> None:
        pass
