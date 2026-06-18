"""QD-SOA gain core (roadmap SOA Phase 1): group-resolved WL -> ES -> GS rate equations.

This is the semiconductor quantum-dot gain model that replaces the four-level ATOMIC scheme
of optics.laser_gain for an injection-pumped traveling-wave amplifier. The state is a wetting
-layer carrier DENSITY N_w plus, for each inhomogeneous dot-size GROUP j, the excited- and
ground-state occupations rho_ES_j, rho_GS_j in [0, 1]. Group resolution is what produces
spectral hole burning (each group bleaches under the LOCAL photon density, not in lockstep).

Conventions and the deep-physics corrections applied (docs/DynaMeta_QD_SOA_extension_spec.md
Section 8; every central formula hand-derived, not copied):

State multiplicities mu_GS = 2 (spin), mu_ES = 4 (2 orbital x spin). N_q is the dot volume
density [m^-3]; group j carries N_q_j = N_q * w_j with the inhomogeneous weights sum w_j = 1.

Rate equations (per unit time):

  dN_w/dt   = I/(q V_a)                                     [injection pump]
              - (N_w/tau_cap) sum_j w_j (1 - rho_ES_j)      [capture out of WL, Pauli-blocked]
              + (mu_ES N_q/tau_esc) sum_j w_j rho_ES_j      [thermal escape back to WL]
              - B N_w^2 - C N_w^3                           [WL bimolecular + Auger recomb]

  drho_ES_j/dt = + N_w (1 - rho_ES_j)/(tau_cap mu_ES N_q)   [capture in (conjugate to WL loss)]
                 - rho_ES_j/tau_esc                         [escape out]
                 - rho_ES_j (1 - rho_GS_j)/tau_ES_GS        [relax ES -> GS]
                 + (mu_GS/mu_ES) rho_GS_j (1 - rho_ES_j)/tau_GS_ES   [thermal back-transfer in]
                 - rho_ES_j^2/tau_sp                        [spontaneous (excitonic rho_e=rho_h)]

  drho_GS_j/dt = + (mu_ES/mu_GS) rho_ES_j (1 - rho_GS_j)/tau_ES_GS   [relax in]
                 - rho_GS_j (1 - rho_ES_j)/tau_GS_ES        [back-transfer out]
                 - v_g sigma_pk L_hom(nu_s - nu_j) (2 rho_GS_j - 1) S_conf   [stimulated]
                 - rho_GS_j^2/tau_sp                        [spontaneous]

The capture/escape and ES<->GS exchange are written as conjugate NUMBER-flux pairs (the
mu_ES N_q normalization on capture, the mu_ES/mu_GS ratio on the ES<->GS cross-terms) so the
total carrier number is conserved by every internal transition -- only injection, the two
recombination channels, and the stimulated term change it (verified to ~1e-12 by the
conservation gate). The per-dot stimulated rate is intrinsically per-dot (no 1/N_q), and the
depleting photon density is the CONFINED density S_conf = Gamma P/(v_g h nu A_mode); the same
sigma_pk * L_hom and mu_GS-weighted N_q build the modal gain, so photon number is conserved.

Spectral modal gain (sum over groups; intensity gain per metre):

  g(nu) = sum_j N_q w_j mu_GS sigma_pk L_hom(nu - nu_j) (2 rho_GS_j - 1)        [1/m]

Excitonic / charge-neutral single-occupation convention (rho_e = rho_h = rho per state):
gain ~ (2 rho_GS - 1) and the spontaneous rate ~ rho^2. Separate electron/hole occupations
(the higher-fidelity pattern-effect model) are a documented Phase-2+ upgrade -- with it the
inversion factor becomes (f_c + f_v - 1) and n_sp = f_c(1-f_v)/(f_c - f_v) (kept consistent).

Pure numpy/scipy; no FDTD, no DEVSIM, no metasurface seam.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy.integrate import solve_ivp

from dynameta.constants import HBAR, KB, Q_E

H_PLANCK = 2.0 * np.pi * HBAR                                  # J s

__all__ = ["QDGainParams", "QDGainModel"]


@dataclass(frozen=True)
class QDGainParams:
    """QD-SOA active-region + kinetic parameters (SI). Defaults are a generic 1550 nm
    InAs/InGaAs QD-SOA in the published range (Sugawara 2002; Berg & Mork 2004) -- a starting
    point, NOT a calibrated datasheet model (no Innolume set exists in the repo yet).

    nu0/inhomogeneous: the ground-state transition is inhomogeneously broadened by the dot-size
    distribution (Gaussian FWHM fwhm_inhom_Hz about nu0) and homogeneously by fwhm_hom_Hz.
    """
    # active region / mode
    N_q_m3: float = 5.0e22            # dot volume density [m^-3]
    V_a_m3: float = 3.0e-16           # active volume of the single section [m^3]
    Gamma: float = 0.06               # optical confinement factor of the dot layer
    A_mode_m2: float = 0.4e-12        # transverse modal area [m^2]
    v_g_m_s: float = 8.5e7            # group velocity in the waveguide [m/s] (n_g ~ 3.5)
    # level structure
    mu_GS: int = 2                    # GS state multiplicity (spin)
    mu_ES: int = 4                    # ES state multiplicity (2 orbital x spin)
    dE_ES_GS_eV: float = 0.060        # ES - GS energy separation [eV] (sets detailed balance)
    # kinetics (times in s)
    tau_cap_s: float = 1.0e-12        # WL -> ES capture
    tau_esc_s: float = 8.0e-12        # ES -> WL escape (override or derive via detailed balance)
    tau_ES_GS_s: float = 1.5e-12      # ES -> GS relaxation
    tau_GS_ES_s: float = 6.0e-12      # GS -> ES thermal back-transfer (derive via detailed bal.)
    tau_sp_s: float = 1.0e-9          # confined-state spontaneous (radiative) lifetime
    B_wl_m3_s: float = 1.0e-16        # WL bimolecular recombination [m^3/s]
    C_wl_m6_s: float = 1.0e-41        # WL Auger recombination [m^6/s]
    # optical transition
    sigma_pk_m2: float = 5.0e-19      # peak GS stimulated cross-section [m^2]
    nu0_Hz: float = 1.934e14          # GS transition centre (~1550 nm) [Hz]
    fwhm_inhom_Hz: float = 5.0e12     # inhomogeneous (dot-size) Gaussian FWHM [Hz]
    fwhm_hom_Hz: float = 1.0e12       # homogeneous Lorentzian FWHM [Hz]
    # spectral discretization
    n_groups: int = 41               # inhomogeneous size groups (odd -> a group sits on nu0)
    span_sigma: float = 3.0          # half-width of the group grid in inhomogeneous sigmas
    T_K: float = 300.0

    def __post_init__(self):
        pos = ("N_q_m3", "V_a_m3", "A_mode_m2", "v_g_m_s", "tau_cap_s", "tau_esc_s",
               "tau_ES_GS_s", "tau_GS_ES_s", "tau_sp_s", "sigma_pk_m2", "nu0_Hz",
               "fwhm_inhom_Hz", "fwhm_hom_Hz", "T_K")
        for nm in pos:
            if not (getattr(self, nm) > 0.0):
                raise ValueError("QDGainParams: {} must be > 0".format(nm))
        if not (0.0 < self.Gamma <= 1.0):
            raise ValueError("QDGainParams: Gamma must be in (0, 1]")
        if self.mu_GS < 1 or self.mu_ES < 1:
            raise ValueError("QDGainParams: mu_GS, mu_ES must be >= 1")
        if self.B_wl_m3_s < 0.0 or self.C_wl_m6_s < 0.0:
            raise ValueError("QDGainParams: B, C must be >= 0")
        if self.n_groups < 1 or self.span_sigma <= 0.0:
            raise ValueError("QDGainParams: n_groups >= 1 and span_sigma > 0")

    def with_detailed_balance_taus(self) -> "QDGainParams":
        """Return a copy with tau_GS_ES fixed by detailed balance, so the dark
        (no-injection, no-signal) relaxation goes to the correct quasi-equilibrium occupancy:

            tau_GS_ES/tau_ES_GS = (mu_GS/mu_ES) exp(+dE/kT)    (GS deeper -> slower to leave up)

        (hand-derived from setting the net ES<->GS exchange to zero with Fermi-Dirac
        occupancy ratios; the mu ratio is mu_GS/mu_ES, NOT its inverse). tau_esc is left as
        supplied (the WL is a continuum reservoir; its detailed-balance link needs the WL
        effective DOS, a Phase-2 refinement)."""
        from dataclasses import replace
        x = self.dE_ES_GS_eV * Q_E / (KB * self.T_K)
        tau_gs_es = self.tau_ES_GS_s * (self.mu_GS / self.mu_ES) * np.exp(x)
        return replace(self, tau_GS_ES_s=float(tau_gs_es))


class QDGainModel:
    """Group-resolved QD-SOA gain core. State vector y = [N_w, rho_ES_0..G-1, rho_GS_0..G-1]."""

    def __init__(self, params: Optional[QDGainParams] = None):
        self.p = params if params is not None else QDGainParams()
        p = self.p
        # inhomogeneous size groups: Gaussian-weighted frequencies about nu0
        sig = p.fwhm_inhom_Hz / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        if p.n_groups == 1:
            self.nu_j = np.array([p.nu0_Hz])
            self.w_j = np.array([1.0])
        else:
            self.nu_j = p.nu0_Hz + sig * np.linspace(-p.span_sigma, p.span_sigma, p.n_groups)
            wj = np.exp(-0.5 * ((self.nu_j - p.nu0_Hz) / sig) ** 2)
            self.w_j = wj / wj.sum()                          # normalized: sum w_j = 1
        self._sig_inhom = sig
        self.ng = int(p.n_groups)

    # ---- lineshape + gain ----
    def _lorentzian(self, dnu):
        """Homogeneous Lorentzian normalized to 1 at line centre; FWHM = fwhm_hom_Hz."""
        hw = 0.5 * self.p.fwhm_hom_Hz
        return hw * hw / (np.asarray(dnu) ** 2 + hw * hw)

    def material_gain_per_m(self, rho_GS, nu_Hz) -> np.ndarray:
        """Spectral intensity gain g(nu) [1/m] from the per-group GS occupations rho_GS
        (length n_groups): g(nu) = sum_j N_q w_j mu_GS sigma_pk L_hom(nu-nu_j)(2 rho_GS_j-1)."""
        p = self.p
        rho = np.asarray(rho_GS, dtype=np.float64)
        nu = np.atleast_1d(np.asarray(nu_Hz, dtype=np.float64))
        # (n_nu, n_groups)
        L = self._lorentzian(nu[:, None] - self.nu_j[None, :])
        inv = (2.0 * rho - 1.0) * self.w_j                    # per-group inversion x weight
        g = p.N_q_m3 * p.mu_GS * p.sigma_pk_m2 * (L @ inv)
        return g if np.ndim(nu_Hz) else float(g[0])

    def power_to_photon_density(self, P_W: float, nu_Hz: float) -> float:
        """Confined photon density S_conf = Gamma P/(v_g h nu A_mode) [m^-3] for guided
        power P at frequency nu -- the density that actually depletes the dots."""
        p = self.p
        return float(p.Gamma * P_W / (p.v_g_m_s * H_PLANCK * nu_Hz * p.A_mode_m2))

    # ---- rate equations ----
    def rhs_fields(self, N_w, rho_ES, rho_GS, I_A, S_conf_m3, nu_s_Hz):
        """Vectorized rate equations over a stack of z-slices -- the single source of truth
        for the carrier dynamics (the scalar rhs and the traveling-wave engine both call it).
        Shapes: N_w (Nz,); rho_ES, rho_GS (Nz, ng); I_A and S_conf_m3 scalar or (Nz,).
        Returns (dN_w (Nz,), drho_ES (Nz, ng), drho_GS (Nz, ng))."""
        p = self.p
        w = self.w_j                                          # (ng,)
        Nw = np.asarray(N_w, dtype=np.float64)                # (Nz,)
        S = np.asarray(S_conf_m3, dtype=np.float64)
        Sb = S[:, None] if S.ndim else S                      # (Nz,1) or scalar
        Ib = np.asarray(I_A, dtype=np.float64)                # (Nz,) or scalar
        L = self._lorentzian(nu_s_Hz - self.nu_j)[None, :]    # (1, ng)

        cap_occ = Nw[:, None] * (1.0 - rho_ES) / (p.tau_cap_s * p.mu_ES * p.N_q_m3)
        esc_occ = rho_ES / p.tau_esc_s
        fwd = rho_ES * (1.0 - rho_GS) / p.tau_ES_GS_s
        bwd = rho_GS * (1.0 - rho_ES) / p.tau_GS_ES_s
        stim = p.v_g_m_s * p.sigma_pk_m2 * L * (2.0 * rho_GS - 1.0) * Sb
        sp_ES = rho_ES * rho_ES / p.tau_sp_s
        sp_GS = rho_GS * rho_GS / p.tau_sp_s

        dN_w = (Ib / (Q_E * p.V_a_m3)
                - (Nw / p.tau_cap_s) * np.sum(w * (1.0 - rho_ES), axis=1)
                + (p.mu_ES * p.N_q_m3 / p.tau_esc_s) * np.sum(w * rho_ES, axis=1)
                - p.B_wl_m3_s * Nw * Nw - p.C_wl_m6_s * Nw ** 3)
        drho_ES = cap_occ - esc_occ - fwd + (p.mu_GS / p.mu_ES) * bwd - sp_ES
        drho_GS = (p.mu_ES / p.mu_GS) * fwd - bwd - stim - sp_GS
        return dN_w, drho_ES, drho_GS

    def rhs(self, y, I_A: float, S_conf_m3: float, nu_s_Hz: float) -> np.ndarray:
        """dy/dt for the single-section state y = [N_w, rho_ES(ng), rho_GS(ng)] -- a thin
        wrapper over rhs_fields with one slice."""
        ng = self.ng
        dNw, dES, dGS = self.rhs_fields(y[0:1], y[1:1 + ng][None, :], y[1 + ng:][None, :],
                                        I_A, S_conf_m3, nu_s_Hz)
        out = np.empty_like(y)
        out[0] = dNw[0]
        out[1:1 + ng] = dES[0]
        out[1 + ng:] = dGS[0]
        return out

    def _initial_state(self) -> np.ndarray:
        y0 = np.zeros(1 + 2 * self.ng)
        y0[0] = 1.0e21                                        # modest WL seed [m^-3]
        y0[1:] = 0.05                                         # nearly empty dots
        return y0

    def steady_state(self, I_A: float, *, S_conf_m3: float = 0.0,
                     nu_s_Hz: Optional[float] = None, t_end_s: float = 5.0e-9,
                     y0=None) -> np.ndarray:
        """Integrate to steady state under constant injection I and signal (S_conf at nu_s).
        Returns y = [N_w, rho_ES(ng), rho_GS(ng)]. Raises if the residual is not small."""
        if I_A < 0.0 or S_conf_m3 < 0.0:
            raise ValueError("steady_state: I and S_conf must be >= 0")
        nu_s = float(nu_s_Hz) if nu_s_Hz is not None else self.p.nu0_Hz
        y0 = self._initial_state() if y0 is None else np.asarray(y0, dtype=np.float64).copy()
        sol = solve_ivp(lambda t, y: self.rhs(y, I_A, S_conf_m3, nu_s),
                        (0.0, t_end_s), y0, method="BDF", rtol=1e-9, atol=1e-12,
                        dense_output=False, t_eval=[t_end_s])
        if not sol.success:
            raise RuntimeError("QDGainModel.steady_state: integration failed ({})".format(
                sol.message))
        y = sol.y[:, -1]
        # convergence + physicality checks
        res = self.rhs(y, I_A, S_conf_m3, nu_s)
        scale = max(abs(y[0]), 1.0) / self.p.tau_sp_s
        if np.max(np.abs(res[1:])) > 1e-6 or abs(res[0]) > 1e-3 * scale:
            # one more relaxation leg if not yet converged
            sol = solve_ivp(lambda t, yy: self.rhs(yy, I_A, S_conf_m3, nu_s),
                            (0.0, 10.0 * t_end_s), y, method="BDF", rtol=1e-10, atol=1e-13,
                            t_eval=[10.0 * t_end_s])
            if not sol.success:
                raise RuntimeError("QDGainModel.steady_state: relaxation leg failed "
                                   "({})".format(sol.message))
            y = sol.y[:, -1]
        occ = y[1:]                                          # re-read AFTER any extra leg
        if np.any(occ < -1e-6) or np.any(occ > 1.0 + 1e-6):
            raise RuntimeError("QDGainModel.steady_state: occupation left [0,1] "
                               "(min {:.3e}, max {:.3e})".format(float(occ.min()),
                                                                 float(occ.max())))
        return y

    def rho_GS(self, y) -> np.ndarray:
        return np.asarray(y)[1 + self.ng:1 + 2 * self.ng]

    def rho_ES(self, y) -> np.ndarray:
        return np.asarray(y)[1:1 + self.ng]

    # ---- small-signal + saturation ----
    def small_signal_gain_per_m(self, I_A: float, nu_Hz=None) -> np.ndarray:
        """Unsaturated (S_conf -> 0) modal gain spectrum at injection I."""
        y = self.steady_state(I_A, S_conf_m3=0.0)
        nu = self.p.nu0_Hz if nu_Hz is None else nu_Hz
        return self.material_gain_per_m(self.rho_GS(y), nu)

    def saturation_curve(self, I_A: float, P_in_W, *, nu_s_Hz: Optional[float] = None
                         ) -> Tuple[np.ndarray, np.ndarray]:
        """Static gain at the signal frequency vs guided signal power. Returns
        (g_per_m, S_conf) over the supplied P_in_W array (single-section, steady state at each
        power -- the compression curve). g is the line-resolved modal gain at nu_s."""
        nu_s = float(nu_s_Hz) if nu_s_Hz is not None else self.p.nu0_Hz
        P = np.atleast_1d(np.asarray(P_in_W, dtype=np.float64))
        g = np.empty(P.size)
        S = np.empty(P.size)
        y = None
        for i, Pi in enumerate(P):
            Si = self.power_to_photon_density(float(Pi), nu_s)
            y = self.steady_state(I_A, S_conf_m3=Si, nu_s_Hz=nu_s, y0=y)   # warm-start
            g[i] = self.material_gain_per_m(self.rho_GS(y), nu_s)
            S[i] = Si
        return g, S

    def total_carrier_density(self, y) -> float:
        """Total carrier density n_tot = N_w + N_q sum_j w_j (mu_ES rho_ES_j + mu_GS rho_GS_j)
        [m^-3] -- the conserved quantity (only injection/recomb/stim change it)."""
        p = self.p
        return float(y[0] + p.N_q_m3 * np.sum(
            self.w_j * (p.mu_ES * self.rho_ES(y) + p.mu_GS * self.rho_GS(y))))

    # ---- traveling-wave slice protocol (consumed by soa.traveling_wave.TravelingWaveSOA) ----
    @property
    def v_g(self) -> float:
        return self.p.v_g_m_s

    @property
    def gamma_confinement(self) -> float:
        return self.p.Gamma

    @property
    def relaxation_time_s(self) -> float:
        """Slowest carrier relaxation timescale (the spontaneous lifetime) -- sets how long a
        CW solve must settle to steady state (>> transit time)."""
        return self.p.tau_sp_s

    def photon_density(self, P_W, nu_Hz):
        """Confined photon density S_conf [m^-3] for guided power P (scalar or array)."""
        p = self.p
        return p.Gamma * np.asarray(P_W, dtype=np.float64) / (
            p.v_g_m_s * H_PLANCK * nu_Hz * p.A_mode_m2)

    def init_slices(self, n_slices: int, I_A: float):
        """Per-slice carrier state (N_w (Nz,), rho_ES (Nz, ng), rho_GS (Nz, ng)) initialized to
        the unsaturated steady state at injection I (uniform along z)."""
        y = self.steady_state(float(I_A))
        nz = int(n_slices)
        return (np.full(nz, y[0]),
                np.tile(self.rho_ES(y), (nz, 1)),
                np.tile(self.rho_GS(y), (nz, 1)))

    def gain_per_m_slices(self, state, nu_Hz) -> np.ndarray:
        """Material intensity gain g(nu) [1/m] per slice from the slice state (uses rho_GS)."""
        p = self.p
        rho_GS = state[2]
        wl = self.w_j * self._lorentzian(nu_Hz - self.nu_j)   # (ng,)
        return p.N_q_m3 * p.mu_GS * p.sigma_pk_m2 * np.sum(
            (2.0 * np.asarray(rho_GS) - 1.0) * wl[None, :], axis=1)

    def step_slices(self, state, P_local_W, dt_s: float, nu_s_Hz: float, I_A: float):
        """Advance the per-slice carrier state by dt driven by the local guided POWER P (Nz,)
        held fixed across the step (operator splitting); explicit RK4. Power is converted to
        the confined photon density internally so the traveling-wave engine speaks one
        currency (power) to every slab model."""
        Nw, rES, rGS = state
        S_conf = self.photon_density(P_local_W, nu_s_Hz)

        def f(nw, es, gs):
            return self.rhs_fields(nw, es, gs, I_A, S_conf, nu_s_Hz)

        k1 = f(Nw, rES, rGS)
        k2 = f(Nw + 0.5 * dt_s * k1[0], rES + 0.5 * dt_s * k1[1], rGS + 0.5 * dt_s * k1[2])
        k3 = f(Nw + 0.5 * dt_s * k2[0], rES + 0.5 * dt_s * k2[1], rGS + 0.5 * dt_s * k2[2])
        k4 = f(Nw + dt_s * k3[0], rES + dt_s * k3[1], rGS + dt_s * k3[2])
        Nw_n = Nw + dt_s / 6.0 * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0])
        rES_n = rES + dt_s / 6.0 * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1])
        rGS_n = rGS + dt_s / 6.0 * (k1[2] + 2.0 * k2[2] + 2.0 * k3[2] + k4[2])
        # physical bounds (occupations in [0,1], densities >= 0): a stability guard against
        # explicit-step overshoot under very strong/fast transients -- it only acts at the
        # bounds, where the unclamped value is already unphysical.
        return (np.maximum(Nw_n, 0.0), np.clip(rES_n, 0.0, 1.0), np.clip(rGS_n, 0.0, 1.0))
