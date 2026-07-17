"""Steady-state z-resolved fiber-amplifier solve: the two-point boundary value problem for the
coupled pump / signal / forward+backward-ASE powers along the fiber (docs sec.1-2). Forward
channels are seeded at z=0, backward channels at z=L; the local metastable fraction nbar2(z) is
algebraic in the local powers, so the state is P(z) alone and the ODE set is first order.

Solved by RELAXATION (the standard EDFA numerical method): alternately integrate the
forward-propagating channels 0->L and the backward-propagating channels L->0 as initial-value
problems (scipy.integrate.solve_ivp), freezing the other direction's z-profile each half-step,
until the endpoint powers converge. Each pass is a stable IVP -- unlike a single two-point
solve_bvp over all channels, whose Newton iteration overflows on the ASE that grows from the
spontaneous floor through tens of dB of gain. Pure numpy/scipy; SI units.
docs/fiber_amp_model_spec.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from dynameta.constants import C_LIGHT, H_PLANCK
from dynameta.optics.fiber_amp.rare_earth import ChannelSet
from dynameta.optics.fiber_amp.spectroscopy import RareEarthIon
from dynameta.optics.fiber_amp.waveguide import FiberSpec

__all__ = ["Pump", "Signal", "AseBand", "FiberAmplifier", "SteadyStateResult"]

@dataclass(frozen=True)
class Pump:
    """A pump beam: power [W], wavelength [m], direction 'fwd' (co, seeded at z=0) or 'bwd'
    (counter, seeded at z=L). For double-clad pumping the FiberSpec carries clad_radius_m and
    the overlap is A_core/A_clad -- set cladding=True to use it for this pump."""
    power_W: float
    lambda_m: float
    direction: str = "fwd"
    cladding: bool = False


@dataclass(frozen=True)
class Signal:
    """A forward signal: power [W] at z=0, wavelength [m]."""
    power_W: float
    lambda_m: float


@dataclass(frozen=True)
class AseBand:
    """The spectral band over which ASE is resolved (both propagation directions): [lambda_min,
    lambda_max] in n_bins bins. m_modes = 2 (both polarizations). Set n_bins=0 to disable ASE
    (a fast ASE-free gain estimate)."""
    lambda_min_m: float
    lambda_max_m: float
    n_bins: int = 40
    m_modes: int = 2


@dataclass
class SteadyStateResult:
    z_m: np.ndarray                            # (M,) mesh
    power_W: np.ndarray                        # (K, M) power of each channel along z
    lambda_m: np.ndarray                       # (K,)
    u: np.ndarray                              # (K,) +1/-1
    is_ase: np.ndarray                         # (K,)
    kind: List[str]                            # (K,) 'pump'|'signal'|'ase'
    nbar2_z: np.ndarray                        # (M,) metastable fraction profile
    signal_gain_dB: np.ndarray                 # per signal channel
    meta: dict = field(default_factory=dict)


class FiberAmplifier:
    """A rare-earth fiber amplifier: an ion + fiber + a channel plan (pumps, signals, an ASE
    band). solve() returns the steady-state z-profiles, signal gain, and the output ASE
    spectrum. Concentration/degradation effects (upconversion, pair-induced quenching,
    photodarkening -- Phase 5, via a ConcentrationModel or the raw upconversion_C_up) and
    cladding pumping (via FiberSpec.clad_radius_m + Pump.cladding) are opt-in; with
    concentration=None and upconversion_C_up=0 the solve is the ideal model."""

    def __init__(self, ion: RareEarthIon, fiber: FiberSpec, pumps: List[Pump],
                 signals: List[Signal], ase: Optional[AseBand] = None, *,
                 upconversion_C_up: float = 0.0, concentration=None):
        self.ion, self.fiber = ion, fiber
        self.pumps, self.signals = list(pumps), list(signals)
        self.ase = ase
        # an all-default (identity) model collapses to the None path: truly byte-identical
        if concentration is not None and getattr(concentration, "is_identity", False):
            concentration = None
        self.concentration = concentration
        if concentration is not None:
            self.upconversion_C_up = float(concentration.c_up_m3_s)
            self._n_active = concentration.active_density(fiber.n_t_m3)
            self._n_dark = concentration.dark_density(fiber.n_t_m3)
        else:
            self.upconversion_C_up = float(upconversion_C_up)
            self._n_active = fiber.n_t_m3
            self._n_dark = 0.0

    # ---- channel plan --------------------------------------------------------------------
    def _plan(self) -> Tuple[ChannelSet, np.ndarray, np.ndarray, np.ndarray, List[str]]:
        lam, u, is_ase, dnu, kind, bc, cladding = [], [], [], [], [], [], []
        for p in self.pumps:
            lam.append(p.lambda_m); u.append(+1.0 if p.direction == "fwd" else -1.0)
            is_ase.append(False); dnu.append(0.0); kind.append("pump")
            bc.append(p.power_W); cladding.append(p.cladding)
        for s in self.signals:
            lam.append(s.lambda_m); u.append(+1.0); is_ase.append(False); dnu.append(0.0)
            kind.append("signal"); bc.append(s.power_W); cladding.append(False)
        if self.ase is not None and self.ase.n_bins > 0:
            edges = np.linspace(self.ase.lambda_min_m, self.ase.lambda_max_m,
                                self.ase.n_bins + 1)
            centres = 0.5 * (edges[:-1] + edges[1:])
            # bin width in FREQUENCY (the spontaneous term is per unit optical frequency)
            nu_edges = C_LIGHT / edges
            dnu_bins = np.abs(nu_edges[:-1] - nu_edges[1:])
            for direction in (+1.0, -1.0):
                for c, dv in zip(centres, dnu_bins):
                    lam.append(float(c)); u.append(direction); is_ase.append(True)
                    dnu.append(float(dv)); kind.append("ase"); bc.append(0.0)
                    cladding.append(False)
        lam = np.asarray(lam); u = np.asarray(u); is_ase = np.asarray(is_ase, bool)
        ch = ChannelSet.build(self.ion, self.fiber, lam, u, is_ase=is_ase, dnu_hz=np.asarray(dnu))
        # cladding pumps: replace the (core) overlap by A_core/A_clad
        gamma = ch.gamma.copy()
        for k, cl in enumerate(cladding):
            if cl:
                from dynameta.optics.fiber_amp.waveguide import cladding_pump_overlap
                gamma[k] = cladding_pump_overlap(self.fiber)
        ch = ChannelSet(ch.lambda_m, ch.u, ch.is_ase, ch.dnu_hz, ch.sigma_a, ch.sigma_e,
                        gamma, ch.loss_per_m, ch.tau_s, ch.sigma_esa)
        return ch, np.asarray(bc), u, is_ase, kind

    # ---- pointwise physics used by the IVP passes ----------------------------------------
    # Per-channel constants (overlap/cross-section/frequency products) are hoisted into a
    # coefficient bundle ONCE per solve (audit S6-13: they were recomputed on each of the
    # ~25k RHS calls). Same algebra, tolerance-neutral regrouping.
    def _coeffs(self, ch: ChannelSet):
        A = self.fiber.a_dope_m2
        na = self._n_active
        m = self.ase.m_modes if self.ase else 2
        c = {
            "flux_a": ch.gamma * ch.sigma_a / (H_PLANCK * ch.nu_hz * A),   # R_a = sum(flux_a P)
            "flux_e": ch.gamma * ch.sigma_e / (H_PLANCK * ch.nu_hz * A),
            "g_e": ch.gamma * na * ch.sigma_e,
            "g_a": ch.gamma * na * ch.sigma_a,
            "g_esa": ch.gamma * na * ch.sigma_esa,
            "loss": ch.loss_per_m.copy(),
            "s_pref": np.where(ch.is_ase, ch.gamma * na * ch.sigma_e * m
                               * H_PLANCK * ch.nu_hz * ch.dnu_hz, 0.0),
            "m_modes": m,
        }
        if self.concentration is not None:
            c["loss"] = c["loss"] + ch.gamma * self._n_dark * ch.sigma_a   # unbleachable PIQ
        return c

    def _nbar2_c(self, c, P):
        """Metastable fraction from the local power vector P (K,) via the coefficient bundle."""
        R_a = float(np.dot(c["flux_a"], P))
        R_e = float(np.dot(c["flux_e"], P))
        tau = self._tau_s
        if self.upconversion_C_up <= 0.0:
            return tau * R_a / (1.0 + tau * (R_a + R_e))
        A2 = self.upconversion_C_up * self._n_active     # upconversion among active excited ions
        B = 1.0 / tau + R_a + R_e
        return (-B + np.sqrt(B * B + 4.0 * A2 * R_a)) / (2.0 * A2)

    def _dP_full_c(self, c, u, P):
        """dP_k/dz [W/m] for every channel from the local power vector P (K,)."""
        P = np.maximum(P, 0.0)
        n2 = self._nbar2_c(c, P)
        g = c["g_e"] * n2 - c["g_a"] * (1.0 - n2) - c["g_esa"] * n2 - c["loss"]
        if self.concentration is not None:
            g = g - self.concentration.photodarkening_loss_per_m(n2)   # inversion-dependent gray
        return u * (g * P + c["s_pref"] * n2)

    # back-compat single-call forms (reference/diagnostic surface)
    def _nbar2(self, ch: ChannelSet, P):
        """Metastable fraction from the full local power vector P (K,) at one z (scalar)."""
        self._tau_s = ch.tau_s
        return self._nbar2_c(self._coeffs(ch), np.asarray(P, float))

    def _dP_full(self, ch: ChannelSet, P):
        """dP_k/dz [W/m] for every channel from the full local power vector P (K,)."""
        self._tau_s = ch.tau_s
        return self._dP_full_c(self._coeffs(ch), ch.u, np.asarray(P, float))

    def _nbar2_profile(self, ch: ChannelSet, P):
        """nbar2 at each z given the full power profile P (K, M)."""
        self._tau_s = ch.tau_s
        c = self._coeffs(ch)
        return np.array([self._nbar2_c(c, P[:, j]) for j in range(P.shape[1])])

    def solve(self, *, n_nodes: int = 201, max_iter: int = 200, tol: float = 1e-6,
              method: str = "LSODA") -> SteadyStateResult:
        from scipy.integrate import solve_ivp
        ch, bc, u, is_ase, kind = self._plan()
        self._tau_s = ch.tau_s
        c = self._coeffs(ch)
        L = self.fiber.length_m
        z = np.linspace(0.0, L, n_nodes)
        fwd = np.where(u > 0)[0]
        bwd = np.where(u < 0)[0]
        u_fwd, u_bwd = u[fwd], u[bwd]

        # backward-channel profiles (K_bwd, M), initialised to their z=L seed everywhere
        P_bwd = np.repeat(bc[bwd][:, None], n_nodes, axis=1) if bwd.size else np.zeros((0, n_nodes))
        P_fwd = np.repeat(bc[fwd][:, None], n_nodes, axis=1)

        def _assemble(Pf, Pb):
            P = np.empty(ch.lambda_m.size)
            P[fwd] = Pf
            if bwd.size:
                P[bwd] = Pb
            return P

        # Lean frozen-profile interpolator (audit S6-2): the mesh is uniform, so scipy interp1d's
        # per-call validation machinery (~43% of solve runtime) is pure overhead. Endpoint clamps
        # reproduce interp1d's fill_value=(left, right) exactly; interior uses the identical
        # linear form, and LSODA samples strictly inside (0, L), so results are unchanged.
        inv_dz = (n_nodes - 1) / L

        def _make_interp(Y):
            slopes = (Y[:, 1:] - Y[:, :-1]) * inv_dz
            ncap = n_nodes - 2

            def f(zz):
                if zz <= 0.0:
                    return Y[:, 0]
                if zz >= L:
                    return Y[:, -1]
                j = int(zz * inv_dz)
                if j > ncap:
                    j = ncap
                return Y[:, j] + slopes[:, j] * (zz - z[j])
            return f

        last_out = None
        last_prof = None
        converged = False
        for it in range(max_iter):
            bwd_of = _make_interp(P_bwd) if bwd.size else None

            def rhs_f(zz, Pf):
                Pb = bwd_of(zz) if bwd.size else np.zeros(0)
                return self._dP_full_c(c, u, _assemble(Pf, Pb))[fwd]

            sf = solve_ivp(rhs_f, (0.0, L), bc[fwd], t_eval=z, method=method,
                           rtol=1e-7, atol=1e-15)
            P_fwd = sf.y

            if bwd.size:
                fwd_of = _make_interp(P_fwd)

                def rhs_b(zz, Pb):
                    return self._dP_full_c(c, u, _assemble(fwd_of(zz), Pb))[bwd]

                sb = solve_ivp(rhs_b, (L, 0.0), bc[bwd], t_eval=z[::-1], method=method,
                               rtol=1e-7, atol=1e-15)
                P_bwd = sb.y[:, ::-1]

            # convergence: endpoint powers AND the full interior profile, each channel measured
            # against its own peak power (audit S3-34: the old endpoint-only test with a 1e-15 W
            # floor could declare victory while the interior was still moving, and amplified
            # noise on strongly-absorbed channels).
            out = np.concatenate([P_fwd[:, -1], (P_bwd[:, 0] if bwd.size else [])])
            prof = np.concatenate([P_fwd, P_bwd], axis=0) if bwd.size else P_fwd.copy()
            if last_out is not None:
                denom = np.maximum(np.abs(out), 1e-15)
                end_ok = float(np.max(np.abs(out - last_out) / denom)) < tol
                ch_peak = np.maximum(np.max(np.abs(prof), axis=1, keepdims=True), 1e-300)
                prof_ok = float(np.max(np.abs(prof - last_prof) / ch_peak)) < tol
                if end_ok and prof_ok:
                    converged = True
                    last_out = out
                    break
            last_out = out
            last_prof = prof

        P = np.empty((ch.lambda_m.size, n_nodes))
        P[fwd] = P_fwd
        if bwd.size:
            P[bwd] = P_bwd
        n2 = self._nbar2_profile(ch, P)
        sig_idx = [i for i, k in enumerate(kind) if k == "signal"]
        gains_dB = np.array([10.0 * np.log10(P[i, -1] / bc[i]) for i in sig_idx])
        return SteadyStateResult(z, P, ch.lambda_m, u, is_ase, kind, n2, gains_dB,
                                 meta={"converged": converged, "iterations": it + 1,
                                       "dnu_hz": ch.dnu_hz.copy(),
                                       "sigma_a": ch.sigma_a.copy(),
                                       "sigma_e": ch.sigma_e.copy(),
                                       "sigma_esa": ch.sigma_esa.copy(),
                                       "gamma": ch.gamma.copy(),
                                       "m_modes": c["m_modes"]})
