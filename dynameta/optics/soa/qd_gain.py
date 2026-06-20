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

# Optional numba fast path for the per-step carrier RK4 (the dominant traveling-wave cost ~70%).
# Self-contained guarded import (mirrors optics.fdtd_nd.backends) so the module imports without
# numba; selected via QDGainModel(fast=True). The numpy rhs_fields stays the reference and the
# parity is asserted by validation/qd_soa_numba_parity.py.
try:                                                          # pragma: no cover - env dependent
    from numba import njit as _njit
    _HAVE_NUMBA = True
except Exception:                                             # pragma: no cover
    _HAVE_NUMBA = False

    def _njit(*a, **k):
        def _wrap(f):
            return f
        return _wrap if not (len(a) == 1 and callable(a[0])) else a[0]

__all__ = ["QDGainParams", "QDGainModel", "SelfHeating"]


@_njit(cache=True, fastmath=True)
def _qd_carrier_rk4_numba(Nw, rES, rGS, S, I_A, dt, Lrow, w, cap_den, esc_pref, stim_pref,
                          tau_cap, tau_esc, tau_ES_GS, tau_GS_ES, tau_sp, B, C,
                          mGS_over_mES, mES_over_mGS, qVa, stim_pref_ES, LErow, es_active,
                          leak_rate):
    """Compiled twin of step_slices: explicit-loop RK4 of the group-resolved QD rate equations
    over all z-slices, MIRRORING rhs_fields term-for-term (so it stays bit-parity with the numpy
    reference -- validated). One source of truth is rhs_fields; this is the optional accelerator.
    Returns the advanced (Nw, rho_ES, rho_GS) with the same [0,1]/>=0 physical clamp."""
    nz, ng = rES.shape
    Nw_o = np.empty(nz)
    rES_o = np.empty((nz, ng))
    rGS_o = np.empty((nz, ng))
    for k in range(nz):
        Sk = S[k]
        nw0 = Nw[k]
        es0 = rES[k]
        gs0 = rGS[k]
        nw_acc = 0.0
        es_acc = np.zeros(ng)
        gs_acc = np.zeros(ng)
        nw_y = nw0
        es_y = es0.copy()
        gs_y = gs0.copy()
        for stage in range(4):
            sum1 = 0.0                                        # sum_j w (1 - rho_ES)
            sum2 = 0.0                                        # sum_j w rho_ES
            des = np.empty(ng)
            dgs = np.empty(ng)
            for j in range(ng):
                ej = es_y[j]
                gj = gs_y[j]
                cap = nw_y * (1.0 - ej) / cap_den
                esc = ej / tau_esc
                fwd = ej * (1.0 - gj) / tau_ES_GS
                bwd = gj * (1.0 - ej) / tau_GS_ES
                stim = stim_pref * Lrow[j] * (2.0 * gj - 1.0) * Sk
                spE = ej * ej / tau_sp
                spG = gj * gj / tau_sp
                des[j] = cap - esc - fwd + mGS_over_mES * bwd - spE
                if es_active:                               # ES optical channel (depletes rho_ES)
                    des[j] -= stim_pref_ES * LErow[j] * (2.0 * ej - 1.0) * Sk
                dgs[j] = mES_over_mGS * fwd - bwd - stim - spG
                sum1 += w[j] * (1.0 - ej)
                sum2 += w[j] * ej
            dnw = (I_A / qVa - (nw_y / tau_cap) * sum1 + esc_pref * sum2
                   - B * nw_y * nw_y - C * nw_y ** 3 - leak_rate * nw_y)
            cw = 2.0 if 0 < stage < 3 else 1.0
            nw_acc += cw * dnw
            for j in range(ng):
                es_acc[j] += cw * des[j]
                gs_acc[j] += cw * dgs[j]
            if stage < 3:
                h = 0.5 * dt if stage < 2 else dt
                nw_y = nw0 + h * dnw
                for j in range(ng):
                    es_y[j] = es0[j] + h * des[j]
                    gs_y[j] = gs0[j] + h * dgs[j]
        nw_n = nw0 + dt / 6.0 * nw_acc
        Nw_o[k] = nw_n if nw_n > 0.0 else 0.0
        for j in range(ng):
            e = es0[j] + dt / 6.0 * es_acc[j]
            g = gs0[j] + dt / 6.0 * gs_acc[j]
            rES_o[k, j] = 0.0 if e < 0.0 else (1.0 if e > 1.0 else e)
            rGS_o[k, j] = 0.0 if g < 0.0 else (1.0 if g > 1.0 else g)
    return Nw_o, rES_o, rGS_o


@_njit(cache=True, fastmath=True)
def _qd_carrier_rk4_eh_numba(Nwe, Nwh, fcES, fvES, fcGS, fvGS, S, I_A, dt, Lrow, w,
                             cap_den_e, cap_den_h, esc_pref_e, esc_pref_h, stim_pref,
                             tau_cap_e, tau_cap_h, tau_esc_e, tau_esc_h, tau_rel_e, tau_rel_h,
                             tau_back_e, tau_back_h, tau_sp, B, C, mGS_over_mES, mES_over_mGS, qVa,
                             stim_pref_ES, LErow, es_active, leak_rate):
    """Compiled twin of _step_slices_eh: explicit-loop RK4 of the electron/hole-split rate
    equations over all z-slices, MIRRORING rhs_fields_eh term-for-term (bit-parity, validated).
    Stimulated + spontaneous are the SAME scalar into both bands; WL recomb is the pair form."""
    nz, ng = fcES.shape
    Nwe_o = np.empty(nz)
    Nwh_o = np.empty(nz)
    fcES_o = np.empty((nz, ng))
    fvES_o = np.empty((nz, ng))
    fcGS_o = np.empty((nz, ng))
    fvGS_o = np.empty((nz, ng))
    for k in range(nz):
        Sk = S[k]
        nwe0 = Nwe[k]
        nwh0 = Nwh[k]
        ce0 = fcES[k]
        ve0 = fvES[k]
        cg0 = fcGS[k]
        vg0 = fvGS[k]
        nwe_acc = 0.0
        nwh_acc = 0.0
        ce_acc = np.zeros(ng)
        ve_acc = np.zeros(ng)
        cg_acc = np.zeros(ng)
        vg_acc = np.zeros(ng)
        nwe_y = nwe0
        nwh_y = nwh0
        ce_y = ce0.copy()
        ve_y = ve0.copy()
        cg_y = cg0.copy()
        vg_y = vg0.copy()
        for stage in range(4):
            s_ce = 0.0
            s_ce1 = 0.0
            s_ve = 0.0
            s_ve1 = 0.0
            dce = np.empty(ng)
            dve = np.empty(ng)
            dcg = np.empty(ng)
            dvg = np.empty(ng)
            for j in range(ng):
                cej = ce_y[j]
                vej = ve_y[j]
                cgj = cg_y[j]
                vgj = vg_y[j]
                inv = cgj + vgj - 1.0
                stim = stim_pref * Lrow[j] * inv * Sk
                spG = cgj * vgj / tau_sp
                spE = cej * vej / tau_sp
                dce[j] = (nwe_y * (1.0 - cej) / cap_den_e - cej / tau_esc_e
                          - cej * (1.0 - cgj) / tau_rel_e
                          + mGS_over_mES * cgj * (1.0 - cej) / tau_back_e - spE)
                dve[j] = (nwh_y * (1.0 - vej) / cap_den_h - vej / tau_esc_h
                          - vej * (1.0 - vgj) / tau_rel_h
                          + mGS_over_mES * vgj * (1.0 - vej) / tau_back_h - spE)
                if es_active:                               # ES channel: SAME scalar into both bands
                    stim_ES = stim_pref_ES * LErow[j] * (cej + vej - 1.0) * Sk
                    dce[j] -= stim_ES
                    dve[j] -= stim_ES
                dcg[j] = (mES_over_mGS * cej * (1.0 - cgj) / tau_rel_e
                          - cgj * (1.0 - cej) / tau_back_e - stim - spG)
                dvg[j] = (mES_over_mGS * vej * (1.0 - vgj) / tau_rel_h
                          - vgj * (1.0 - vej) / tau_back_h - stim - spG)
                s_ce1 += w[j] * (1.0 - cej)
                s_ce += w[j] * cej
                s_ve1 += w[j] * (1.0 - vej)
                s_ve += w[j] * vej
            R_wl = B * nwe_y * nwh_y + C * nwe_y * nwh_y * (nwe_y + nwh_y) / 2.0
            dnwe = (I_A / qVa - (nwe_y / tau_cap_e) * s_ce1 + esc_pref_e * s_ce - R_wl
                    - leak_rate * nwe_y)
            dnwh = (I_A / qVa - (nwh_y / tau_cap_h) * s_ve1 + esc_pref_h * s_ve - R_wl
                    - leak_rate * nwh_y)
            cw = 2.0 if 0 < stage < 3 else 1.0
            nwe_acc += cw * dnwe
            nwh_acc += cw * dnwh
            for j in range(ng):
                ce_acc[j] += cw * dce[j]
                ve_acc[j] += cw * dve[j]
                cg_acc[j] += cw * dcg[j]
                vg_acc[j] += cw * dvg[j]
            if stage < 3:
                h = 0.5 * dt if stage < 2 else dt
                nwe_y = nwe0 + h * dnwe
                nwh_y = nwh0 + h * dnwh
                for j in range(ng):
                    ce_y[j] = ce0[j] + h * dce[j]
                    ve_y[j] = ve0[j] + h * dve[j]
                    cg_y[j] = cg0[j] + h * dcg[j]
                    vg_y[j] = vg0[j] + h * dvg[j]
        nwe_n = nwe0 + dt / 6.0 * nwe_acc
        nwh_n = nwh0 + dt / 6.0 * nwh_acc
        Nwe_o[k] = nwe_n if nwe_n > 0.0 else 0.0
        Nwh_o[k] = nwh_n if nwh_n > 0.0 else 0.0
        for j in range(ng):
            ce = ce0[j] + dt / 6.0 * ce_acc[j]
            ve = ve0[j] + dt / 6.0 * ve_acc[j]
            cg = cg0[j] + dt / 6.0 * cg_acc[j]
            vg = vg0[j] + dt / 6.0 * vg_acc[j]
            fcES_o[k, j] = 0.0 if ce < 0.0 else (1.0 if ce > 1.0 else ce)
            fvES_o[k, j] = 0.0 if ve < 0.0 else (1.0 if ve > 1.0 else ve)
            fcGS_o[k, j] = 0.0 if cg < 0.0 else (1.0 if cg > 1.0 else cg)
            fvGS_o[k, j] = 0.0 if vg < 0.0 else (1.0 if vg > 1.0 else vg)
    return Nwe_o, Nwh_o, fcES_o, fvES_o, fcGS_o, fvGS_o


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
    # excited-state (ES) optical band (opt-in two-state gain). The ES transition sits dE_ES_GS
    # ABOVE the GS (nu_ES_j = nu_j + dE_ES_GS/h, derived -- no separate nu param), carries mu_ES.
    # sigma_pk_ES_m2 = 0 (default) -> GS-only, byte-identical. Lets a signal near the ES band see
    # ES gain and exposes the GS/ES two-state crossover at high injection.
    sigma_pk_ES_m2: float = 0.0       # peak ES stimulated cross-section [m^2]; 0 -> GS-only
    fwhm_hom_ES_Hz: Optional[float] = None  # ES homogeneous FWHM [Hz]; None -> reuse fwhm_hom_Hz
    # coherent / phase
    alpha_lef: float = 2.0           # linewidth enhancement factor (carrier-induced index;
                                     # QD ~ 1-3 near the GS peak) -- drives FWM + its asymmetry
    alpha_lef_density_slope: float = 0.0  # d(alpha)/d(rho_GS): linewidth factor rises with inversion
                                     # as the gain clamps (dg/dN drops, dn/dN persists); 0 -> constant
    beta2_s2_per_m: float = 0.0      # background (waveguide) group-velocity dispersion d2beta/domega2
                                     # [s^2/m]; broadband non-resonant index (0 -> no GVD)
    # spectral discretization
    n_groups: int = 41               # inhomogeneous size groups (odd -> a group sits on nu0)
    span_sigma: float = 3.0          # half-width of the group grid in inhomogeneous sigmas
    T_K: float = 300.0
    # electron/hole occupation split (opt-in, Phase-6 gain fidelity). With eh_split the dots carry
    # SEPARATE electron f_c and hole f_v occupations per state; gain -> (f_c+f_v-1), spontaneous ->
    # f_c f_v, n_sp -> f_c f_v/(f_c+f_v-1). The tau_*_s above are the ELECTRON times; the hole times
    # default (None) to the electron value, so eh_split with all hole-times None reduces to the
    # excitonic model. The physical asymmetry is holes-faster (heavier -> faster capture/relax).
    eh_split: bool = False
    tau_cap_h_s: Optional[float] = None   # WL -> ES hole capture (None -> tau_cap_s)
    tau_esc_h_s: Optional[float] = None   # ES -> WL hole escape  (None -> tau_esc_s)
    tau_rel_h_s: Optional[float] = None   # ES -> GS hole relaxation (None -> tau_ES_GS_s)
    tau_back_h_s: Optional[float] = None  # GS -> ES hole back-transfer (None -> tau_GS_ES_s)

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
        for nm in ("tau_cap_h_s", "tau_esc_h_s", "tau_rel_h_s", "tau_back_h_s"):
            v = getattr(self, nm)
            if v is not None and not (v > 0.0):
                raise ValueError("QDGainParams: {} must be > 0 when set".format(nm))
        if self.sigma_pk_ES_m2 < 0.0:                         # 0 is the GS-only disable switch
            raise ValueError("QDGainParams: sigma_pk_ES_m2 must be >= 0")
        if self.fwhm_hom_ES_Hz is not None and not (self.fwhm_hom_ES_Hz > 0.0):
            raise ValueError("QDGainParams: fwhm_hom_ES_Hz must be > 0 when set")

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
        ratio = (self.mu_GS / self.mu_ES) * np.exp(x)
        updates = {"tau_GS_ES_s": float(self.tau_ES_GS_s * ratio)}
        if self.eh_split:                                     # holes get the same detailed-balance rule
            tau_rel_h = self.tau_rel_h_s if self.tau_rel_h_s is not None else self.tau_ES_GS_s
            updates["tau_back_h_s"] = float(tau_rel_h * ratio)
        return replace(self, **updates)


@dataclass(frozen=True)
class SelfHeating:
    """Lumped self-heating coupled into the gain (opt-in ENOB-budget physics). The junction
    dissipates P_diss = I V_j - eta_extraction (P_out - P_in) and heats to T = T0 + Rth P_diss
    (steady) / Cth dT/dt = P_diss - (T-T0)/Rth (transient, tau_th = Rth Cth). Temperature couples
    into the gain two ways (the minimal well-posed choice -> a monotone dg/dT < 0, unique stable
    fixed point):
      - rigid red-shift of the whole comb: nu_j(T) = nu_j0 - dnu0_dT (T - T0)  (dnu0_dT > 0)
      - fractional peak-gain coefficient:  gain_scale(T) = max(0, 1 + dg_dT_frac (T - T0)), applied
        to BOTH the modal-gain emission and the per-dot stimulated depletion (photon-number safe).
    Rth_K_W = 0 (default) -> isothermal, byte-identical to the current engine. The derived
    single-pass dG/dT [dB/K] feeds metrics.thermal_drift_budget_K -> the predistortion ENOB ceiling.
    SI; ASCII. (Sugawara/Coldren bandgap red-shift; lumped RC after e.g. Ning & Lippi SOA-thermal.)"""
    Rth_K_W: float = 0.0              # thermal resistance [K/W]; 0 -> isothermal (OFF)
    Cth_J_K: float = 1.0e-9           # heat capacity [J/K] (transient only; tau_th = Rth Cth)
    dnu0_dT_Hz_K: float = 0.0         # gain-peak red-shift coef [Hz/K] (>=0: nu falls as T rises)
    dg_dT_frac_per_K: float = 0.0     # fractional peak-gain coef (1/g) dg/dT [1/K] (<=0 physical)
    T0_K: float = 300.0               # ambient / heat-sink reference [K]
    V_j_V: float = 0.9                # junction bias for P_elec = I V_j [V]
    eta_extraction: float = 1.0       # optical-extraction fraction removed from P_diss [0,1]
    w_relax: float = 0.5              # steady Picard under-relaxation in (0,1]
    tol_T_K: float = 1.0e-3
    max_iter: int = 60

    def __post_init__(self):
        if self.Rth_K_W < 0.0 or self.Cth_J_K <= 0.0 or self.T0_K <= 0.0:
            raise ValueError("SelfHeating: need Rth >= 0, Cth > 0, T0 > 0")
        if self.dnu0_dT_Hz_K < 0.0:
            raise ValueError("SelfHeating: dnu0_dT_Hz_K must be >= 0 (red-shift as T rises)")
        if self.dg_dT_frac_per_K > 0.0:
            raise ValueError("SelfHeating: dg_dT_frac_per_K must be <= 0 (gain drops with T)")
        if not (0.0 <= self.eta_extraction <= 1.0) or not (0.0 < self.w_relax <= 1.0):
            raise ValueError("SelfHeating: eta_extraction in [0,1], w_relax in (0,1]")

    @property
    def active(self) -> bool:
        return self.Rth_K_W > 0.0

    @property
    def tau_th_s(self) -> float:
        return self.Rth_K_W * self.Cth_J_K


@dataclass(frozen=True)
class Leakage:
    """Thermally-activated carrier leakage (phenomenological) out of the wetting-layer reservoir over
    the confining barrier -- a temperature-activated escape the closed capture/escape/recombination
    ladder omits. Adds a linear loss -N_w / tau_leak(T) to the WL rate equation, with an Arrhenius rate

        1 / tau_leak(T) = (1 / tau_leak0_s) exp(-E_barrier_eV q / (k_B T)),

    so the escape rate rises steeply as the device heats (more carriers clear the barrier). TWO
    distinct effects, NOT the same thing:
      - the DIVERTED CURRENT (carriers lost to escape, N_w/tau_leak) rises with both pump and T;
      - the GAIN SUPPRESSION it causes is LARGEST near threshold / below saturation and SHRINKS with
        drive -- the clamped high-injection gain is barely affected (the % AND absolute gain drop fall
        monotonically as current rises). So this is a below-saturation/threshold and high-T effect, NOT
        a "high-injection gain rolloff."
    T is the model junction temperature (set via set_temperature / SelfHeating.T0_K; default p.T_K).
    tau_leak0_s <= 0 (default) disables leakage -> byte-identical. The escaped carriers LEAVE the model
    permanently -- there is no leakage-recapture return path. In the e/h-split layout BOTH the electron
    and hole WL reservoirs leak at the same rate (SIMPLIFICATION: a single shared rate enforces
    charge-neutral bipolar leakage; the model does not support distinct e/h barrier heights, unlike the
    e/h-asymmetric capture/relax times). SI; ASCII. (PHENOMENOLOGICAL: a linear -N_w/tau_leak loss with
    an Arrhenius PREFACTOR exp(-E_b q/kT), NOT a computed thermionic-emission / barrier-integral flux;
    E_barrier_eV is a fitted activation energy, tau_leak0_s a fitted prefactor; no T-dependent
    attempt-rate prefactor.)"""
    tau_leak0_s: float = 0.0          # escape-time prefactor [s]; <= 0 -> OFF
    E_barrier_eV: float = 0.30        # barrier activation energy [eV]

    def __post_init__(self):
        if self.E_barrier_eV < 0.0:
            raise ValueError("Leakage: E_barrier_eV must be >= 0")

    @property
    def active(self) -> bool:
        return self.tau_leak0_s > 0.0

    def rate_at(self, T_K: float) -> float:
        """Thermionic leakage rate 1/tau_leak(T) [1/s] (0 when disabled)."""
        if self.tau_leak0_s <= 0.0:
            return 0.0
        return np.exp(-self.E_barrier_eV * Q_E / (KB * float(T_K))) / self.tau_leak0_s


@dataclass(frozen=True)
class ManyBody:
    """Closed-form many-body-corrected QD gain (opt-in screened-Hartree-Fock-FLAVOURED physics; NOT a
    solved self-consistent k-resolved semiconductor-Bloch-equation gain -- there is no k-summation,
    screened-Coulomb matrix, or self-energy iteration, only the Haug-Koch closed-form universal forms
    with input coefficients). STATUS: this is a STANDALONE chi(nu)/alpha analysis accessor
    (material_gain_index_mb); it is NOT wired into the traveling-wave marcher (gain_per_m_slices /
    rhs_fields / step_slices), so the simulated amplifier's gain and dynamics are UNCHANGED whether or
    not ManyBody is enabled. The phenomenological free-carrier gain is a sum of complex Lorentzians
    whose REAL part is the gain and IMAGINARY part its Kramers-Kronig index partner (one analytic
    chi(nu)). This adds the three dominant finite-density many-body corrections, all functions of the
    carrier density N (and T), so the gain, the carrier-induced index, and hence alpha become a single
    CONSISTENT (one-chi) object rather than three separately tuned knobs:

      - Bandgap renormalization (BGR): the screened-exchange + Coulomb-hole self-energy red-shifts
        every transition, dE_BGR(N) = -bgr_coeff * E_R * (a_B^3 N)^(1/3)  (Haug-Koch universal 3D
        form; bgr_coeff ~ 1.9). E_R (exciton Rydberg) and a_B (exciton Bohr radius) carry the
        material (m*, eps). nu_j -> nu_j + dE_BGR(N)/h  (dE_BGR < 0 -> red-shift).
      - Excitation-induced dephasing (EID) + phonon dephasing: the homogeneous HWHM grows with
        carrier density (carrier-carrier scattering) and temperature (LO-phonon, Bose-occupied):
        gamma(N,T) = gamma0 + gamma_eid (N/N_ref) + gamma_phonon [(2 n_LO(T)+1) - (2 n_LO(T0)+1)].
        The EID broadening CONSERVES oscillator strength (the line area), so the peak drops as
        gamma0/gamma -- the physically correct invariant (the free-carrier model holds the peak,
        which over-counts the integrated gain when broadened).
      - Coulomb / excitonic enhancement, screened away toward the Mott density:
        C_enh(N) = 1 + coulomb_enh * exp(-N / N_mott)  (-> 1 at high N).

    enabled = False (default) -> every correction is the identity and the gain reduces EXACTLY to
    material_gain_per_m. SCOPE: per-group (each inhomogeneous group renormalized by the SAME N) --
    the QD-appropriate limit since dots are spatially separated; a full k-resolved multiband SBE with
    the wetting-layer continuum and inter-group Coulomb coupling is the deeper (continuum) refinement.
    SI; ASCII. (Haug & Koch, Quantum Theory of the Optical and Electronic Properties of Semiconductors.)"""
    enabled: bool = False
    exciton_rydberg_meV: float = 10.0     # exciton Rydberg E_R [meV] (sets the BGR + enhancement scale)
    exciton_bohr_nm: float = 12.0         # exciton Bohr radius a_B [nm]
    bgr_coeff: float = 1.9                # universal Haug-Koch BGR coefficient (~1.9, 3D)
    gamma_eid_Hz: float = 0.0             # EID HWHM at N = N_ref_eid [Hz]; 0 -> no density broadening
    N_ref_eid_m3: float = 1.0e24          # reference density for the EID slope
    gamma_phonon_Hz: float = 0.0          # LO-phonon dephasing HWHM scale [Hz]; 0 -> no T broadening
    E_LO_meV: float = 36.0                # LO-phonon energy [meV] (Bose occupation)
    coulomb_enh: float = 0.0              # excitonic peak enhancement at low density; 0 -> none
    N_mott_m3: float = 5.0e24             # Mott density (enhancement screened to 1 above this)

    def __post_init__(self):
        if self.exciton_rydberg_meV <= 0.0 or self.exciton_bohr_nm <= 0.0:
            raise ValueError("ManyBody: exciton_rydberg_meV and exciton_bohr_nm must be > 0")
        if self.bgr_coeff < 0.0 or self.gamma_eid_Hz < 0.0 or self.gamma_phonon_Hz < 0.0:
            raise ValueError("ManyBody: bgr_coeff, gamma_eid_Hz, gamma_phonon_Hz must be >= 0")
        if self.N_ref_eid_m3 <= 0.0 or self.N_mott_m3 <= 0.0:
            raise ValueError("ManyBody: N_ref_eid_m3 and N_mott_m3 must be > 0")

    @property
    def active(self) -> bool:
        return bool(self.enabled)


class QDGainModel:
    """Group-resolved QD-SOA gain core. State vector y = [N_w, rho_ES_0..G-1, rho_GS_0..G-1]."""

    def __init__(self, params: Optional[QDGainParams] = None, *, fast: bool = False,
                 self_heating: "Optional[SelfHeating]" = None,
                 many_body: "Optional[ManyBody]" = None,
                 leakage: "Optional[Leakage]" = None):
        self.p = params if params is not None else QDGainParams()
        p = self.p
        self.mb = many_body                                   # microscopic many-body gain (opt-in)
        self.leak = leakage                                   # thermionic WL leakage (opt-in)
        # Optional numba carrier-step accelerator (4.7x at ng=1, 7x at ng=41; bit-parity with the
        # numpy reference). Default OFF so results are byte-stable across machines (numba-present or
        # not); opt in with fast=True for long transient runs. Explicit fast=True without numba is
        # an error (don't silently fall back to slow numpy when speed was requested).
        if fast and not _HAVE_NUMBA:
            raise RuntimeError("QDGainModel(fast=True) requires numba (pip install numba)")
        self._use_numba = bool(fast) and _HAVE_NUMBA
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
        # Precomputed constant prefactors (frozen params) -- pulled out of the per-step hot loop.
        # Each is the SAME product the inline expression formed, so results stay bit-identical.
        self._cap_den = p.tau_cap_s * p.mu_ES * p.N_q_m3            # capture denominator
        self._esc_pref = p.mu_ES * p.N_q_m3 / p.tau_esc_s          # WL escape-in prefactor
        self._stim_pref = p.v_g_m_s * p.sigma_pk_m2                # per-dot stimulated prefactor
        self._gain_pref = p.N_q_m3 * p.mu_GS * p.sigma_pk_m2       # modal-gain prefactor
        self._qVa = Q_E * p.V_a_m3                                 # injection charge*volume
        # Single-entry caches for the homogeneous Lorentzian evaluated at the (fixed) signal
        # frequency -- rhs_fields and gain_per_m_slices are called O(nt) times with the SAME nu_s,
        # so caching L(nu_s - nu_j) removes a redundant recompute per call (byte-identical values).
        self._L_nu = None        # cached nu for the rate-equation Lorentzian row (1, ng)
        self._L_row = None
        self._gw_nu = None       # cached nu for the gain line weights w_j * L (ng,)
        self._gw = None
        # electron/hole occupation split: resolve hole times (None -> electron value -> symmetric
        # reduction) into plain floats so the hot loop sees scalars, and precompute the per-band
        # capture/escape prefactors mirroring the electron ones.
        self.eh = bool(p.eh_split)
        self._tcap_h = float(p.tau_cap_h_s) if p.tau_cap_h_s is not None else p.tau_cap_s
        self._tesc_h = float(p.tau_esc_h_s) if p.tau_esc_h_s is not None else p.tau_esc_s
        self._trel_h = float(p.tau_rel_h_s) if p.tau_rel_h_s is not None else p.tau_ES_GS_s
        self._tback_h = float(p.tau_back_h_s) if p.tau_back_h_s is not None else p.tau_GS_ES_s
        self._cap_den_h = self._tcap_h * p.mu_ES * p.N_q_m3        # hole capture denominator
        self._esc_pref_h = p.mu_ES * p.N_q_m3 / self._tesc_h       # hole escape-in prefactor
        # excited-state optical band (opt-in). The ES comb is the GS comb rigidly blue-shifted by
        # the ES-GS separation (so optical + detailed-balance spacings stay consistent); carries
        # mu_ES. sigma_pk_ES = 0 -> _es_active False -> every ES term short-circuits to 0 (GS-only,
        # byte-identical). Its own cached Lorentzian row + gain line weights mirror the GS ones.
        self.nu_ES_j = self.nu_j + p.dE_ES_GS_eV * Q_E / H_PLANCK  # (ng,) ES line comb
        self._es_active = p.sigma_pk_ES_m2 > 0.0
        self._stim_pref_ES = p.v_g_m_s * p.sigma_pk_ES_m2          # per-dot ES stimulated prefactor
        self._gain_pref_ES = p.N_q_m3 * p.mu_ES * p.sigma_pk_ES_m2  # ES modal-gain prefactor (mu_ES)
        self._hw_ES = 0.5 * (p.fwhm_hom_ES_Hz if p.fwhm_hom_ES_Hz is not None else p.fwhm_hom_Hz)
        self._zeros_ng = np.zeros(self.ng)                        # inactive-ES Lorentzian row
        self._LE_nu = None
        self._LE_row = None
        self._gwE_nu = None
        self._gwE = None
        # lumped self-heating (opt-in). The gain emitters/stim multiply by _gain_scale (1.0 ->
        # isothermal, byte-identical) and set_temperature rigidly red-shifts the combs off the
        # cached cold-grid _nu_j0/_nu_ES_j0. dg_dT_frac is taken from SelfHeating directly.
        self.sh = self_heating
        self._nu_j0 = self.nu_j.copy()
        self._nu_ES_j0 = self.nu_ES_j.copy()
        self._gain_scale = 1.0
        self._dg_dT_frac = self.sh.dg_dT_frac_per_K if self.sh is not None else 0.0
        self._T = self.sh.T0_K if self.sh is not None else p.T_K

    def _leak_rate(self) -> float:
        """Instantaneous thermionic WL leakage rate 1/tau_leak(T) [1/s] at the model temperature
        self._T (0.0 when no Leakage -> the rate-equation term vanishes, byte-identical)."""
        return self.leak.rate_at(self._T) if self.leak is not None else 0.0

    # ---- lineshape + gain ----
    def _lorentzian(self, dnu):
        """Homogeneous Lorentzian normalized to 1 at line centre; FWHM = fwhm_hom_Hz."""
        hw = 0.5 * self.p.fwhm_hom_Hz
        return hw * hw / (np.asarray(dnu) ** 2 + hw * hw)

    def _L_at(self, nu_s_Hz):
        """Cached homogeneous Lorentzian row L(nu_s - nu_j), shape (1, ng) -- constant across the
        time march (fixed nu_s), so computed once and reused (identical values)."""
        if self._L_nu != nu_s_Hz:
            self._L_row = self._lorentzian(nu_s_Hz - self.nu_j)[None, :]
            self._L_nu = nu_s_Hz
        return self._L_row

    def _gain_line_weights(self, nu_Hz):
        """Cached gain line weights w_j * L(nu - nu_j), shape (ng,) -- constant across the march."""
        if self._gw_nu != nu_Hz:
            self._gw = self.w_j * self._lorentzian(nu_Hz - self.nu_j)
            self._gw_nu = nu_Hz
        return self._gw

    # ---- excited-state (ES) band lineshape + inversion (mirror the GS helpers, retargeted to
    # the blue-shifted ES comb nu_ES_j and the ES homogeneous HWHM _hw_ES) ----
    def _lorentzian_ES(self, dnu):
        hw = self._hw_ES
        return hw * hw / (np.asarray(dnu) ** 2 + hw * hw)

    def _LE_at(self, nu_s_Hz):
        """Cached ES Lorentzian row L_ES(nu_s - nu_ES_j), shape (1, ng)."""
        if self._LE_nu != nu_s_Hz:
            self._LE_row = self._lorentzian_ES(nu_s_Hz - self.nu_ES_j)[None, :]
            self._LE_nu = nu_s_Hz
        return self._LE_row

    def _gain_line_weights_ES(self, nu_Hz):
        """Cached ES gain line weights w_j * L_ES(nu - nu_ES_j), shape (ng,)."""
        if self._gwE_nu != nu_Hz:
            self._gwE = self.w_j * self._lorentzian_ES(nu_Hz - self.nu_ES_j)
            self._gwE_nu = nu_Hz
        return self._gwE

    def _es_inversion(self, state) -> np.ndarray:
        """ES inversion: (2 rho_ES - 1) excitonic [state[1]], or (f_c_ES + f_v_ES - 1) split
        [state[2]+state[3]]. The ES analogue of _gs_inversion."""
        if self.eh:
            return np.asarray(state[2]) + np.asarray(state[3]) - 1.0
        return 2.0 * np.asarray(state[1]) - 1.0

    def set_temperature(self, T_K: float) -> None:
        """Set the junction temperature [K] (self-heating coupling site). Rigidly red-shifts the
        GS+ES combs off the cold grids (nu_j = nu_j0 - dnu0_dT (T-T0)) and sets the fractional
        peak-gain scale (1 + dg_dT_frac (T-T0), floored at 0); invalidates the cached Lorentzian
        rows. No-op (and no cache touch) when self-heating is disabled -> isothermal byte-identity."""
        self._T = float(T_K)
        if self.sh is None:
            return
        dT = self._T - self.sh.T0_K
        self.nu_j = self._nu_j0 - self.sh.dnu0_dT_Hz_K * dT
        self.nu_ES_j = self._nu_ES_j0 - self.sh.dnu0_dT_Hz_K * dT
        self._L_nu = self._gw_nu = self._LE_nu = self._gwE_nu = None     # invalidate stale rows
        s = 1.0 + self._dg_dT_frac * dT
        self._gain_scale = s if s > 0.0 else 0.0

    def material_gain_per_m(self, rho_GS, nu_Hz) -> np.ndarray:
        """Spectral intensity gain g(nu) [1/m] from the per-group GS occupations rho_GS
        (length n_groups): g(nu) = sum_j N_q w_j mu_GS sigma_pk L_hom(nu-nu_j)(2 rho_GS_j-1)."""
        p = self.p
        rho = np.asarray(rho_GS, dtype=np.float64)
        nu = np.atleast_1d(np.asarray(nu_Hz, dtype=np.float64))
        # (n_nu, n_groups)
        L = self._lorentzian(nu[:, None] - self.nu_j[None, :])
        inv = (2.0 * rho - 1.0) * self.w_j                    # per-group inversion x weight
        g = self._gain_scale * p.N_q_m3 * p.mu_GS * p.sigma_pk_m2 * (L @ inv)
        return g if np.ndim(nu_Hz) else float(g[0])

    def total_material_gain(self, rho_ES, rho_GS, nu_Hz) -> np.ndarray:
        """Full GS + ES material gain spectrum g(nu) [1/m] over nu (scalar or array) from the
        EXCITONIC occupations. material_gain_per_m is the GS-only sibling (back-compat). The ES
        term g_ES = sum_j N_q w_j mu_ES sigma_pk_ES L_ES(nu - nu_ES_j)(2 rho_ES-1) is added only
        when sigma_pk_ES > 0 -> identical to material_gain_per_m for the GS-only default."""
        p = self.p
        rES = np.asarray(rho_ES, dtype=np.float64)
        nu = np.atleast_1d(np.asarray(nu_Hz, dtype=np.float64))
        g = self.material_gain_per_m(rho_GS, nu)              # GS band (array-safe)
        if self._es_active:
            LE = self._lorentzian_ES(nu[:, None] - self.nu_ES_j[None, :])
            invE = (2.0 * rES - 1.0) * self.w_j
            g = g + self._gain_scale * p.N_q_m3 * p.mu_ES * p.sigma_pk_ES_m2 * (LE @ invE)
        return g if np.ndim(nu_Hz) else float(np.atleast_1d(g)[0])

    def emission_gain_per_m(self, rho_GS, nu_Hz) -> np.ndarray:
        """Spontaneous-EMISSION gain spectrum g_sp(nu) [1/m] (GS band, excitonic): sum_j N_q w_j
        mu_GS sigma_pk L(nu-nu_j) rho_GS_j^2 -- proportional to the upper-state population, the ASE
        SOURCE amplitude (the bidirectional-ASE source is q = Gamma g_sp h nu). It equals
        g(nu)*n_sp(nu) but is POLE-FREE (no n_sp division at net transparency); for a single group
        g_sp = g * rho^2/(2 rho-1), so Gamma g_sp h nu reproduces ase_output_psd's source exactly.
        Scaled by the self-heating gain factor like the net gain."""
        p = self.p
        rho = np.asarray(rho_GS, dtype=np.float64)
        nu = np.atleast_1d(np.asarray(nu_Hz, dtype=np.float64))
        L = self._lorentzian(nu[:, None] - self.nu_j[None, :])
        em = (rho * rho) * self.w_j                          # per-group emission (upper population)
        g = self._gain_scale * p.N_q_m3 * p.mu_GS * p.sigma_pk_m2 * (L @ em)
        return g if np.ndim(nu_Hz) else float(g[0])

    # ---- closed-form many-body-corrected gain (screened-HF-flavoured, opt-in; NOT a solved SBE) ----
    def _mb_bgr_shift_Hz(self, N_m3) -> float:
        """Bandgap-renormalization shift dnu(N) = dE_BGR(N)/h [Hz] (<= 0, a red-shift). Universal
        screened-exchange + Coulomb-hole form dE_BGR = -bgr_coeff E_R (a_B^3 N)^(1/3)."""
        mb = self.mb
        E_R = mb.exciton_rydberg_meV * 1.0e-3 * Q_E          # exciton Rydberg [J]
        a_B = mb.exciton_bohr_nm * 1.0e-9                    # exciton Bohr radius [m]
        N = max(float(N_m3), 0.0)
        return -mb.bgr_coeff * E_R * (a_B ** 3 * N) ** (1.0 / 3.0) / H_PLANCK

    def _mb_hwhm_Hz(self, N_m3, T_K) -> float:
        """Density + temperature dependent homogeneous HWHM gamma(N,T) [Hz] = gamma0 + EID(N) +
        phonon-excess(T). Reduces to gamma0 = 0.5 fwhm_hom at N=0 and T=T_nominal."""
        mb, p = self.mb, self.p
        hw0 = 0.5 * p.fwhm_hom_Hz
        eid = mb.gamma_eid_Hz * (max(float(N_m3), 0.0) / mb.N_ref_eid_m3)
        ph = 0.0
        if mb.gamma_phonon_Hz > 0.0:
            elo = mb.E_LO_meV * 1.0e-3 * Q_E
            nT = 1.0 / np.expm1(elo / (KB * float(T_K)))     # LO-phonon Bose occupation at T
            n0 = 1.0 / np.expm1(elo / (KB * p.T_K))          # ... at the nominal T (excess only)
            ph = mb.gamma_phonon_Hz * 2.0 * (nT - n0)
        return hw0 + eid + ph

    def material_gain_index_mb(self, rho_GS, nu_Hz, N_carrier_m3, T_K=None) -> Tuple:
        """Microscopic many-body GS gain g(nu) [1/m] AND its carrier-induced index partner gi(nu)
        [1/m -- the complex gain coefficient is g + i gi, gi the dispersive Kramers-Kronig partner]
        from the renormalized complex susceptibility. BGR red-shifts the comb by dnu_BGR(N), EID +
        phonon set the HWHM gamma(N,T) with OSCILLATOR-STRENGTH conservation (the peak scales as
        gamma0/gamma so the line area is invariant under broadening), and the Coulomb enhancement
        C_enh(N) scales the peak. g and gi are the Re and Im of the SAME analytic complex Lorentzian
        sum, so gi == Hilbert(g) by construction (KK-consistent) and the local alpha = -gi/g is the
        KK-consistent ratio of the renormalized chi (no separate alpha knob; the coefficients are
        still input parameters, not solved). Reduces EXACTLY to (material_gain_per_m, its KK index)
        when many_body is
        disabled OR all corrections are zero (bgr_coeff=gamma_eid=gamma_phonon=coulomb_enh=0).

        INTEGRATION SCOPE: this is the standalone many-body susceptibility ACCESSOR (pass the carrier
        density N explicitly). It is NOT yet wired into the traveling-wave marcher's per-slice gain
        (gain_per_m_slices still uses the free-carrier material_gain_per_m), so the device dynamics
        are unchanged until a caller drives the per-slice gain through this method with the local
        wetting-layer density N_w(z) -- the straightforward integration follow-on."""
        p = self.p
        rho = np.asarray(rho_GS, dtype=np.float64)
        nu = np.atleast_1d(np.asarray(nu_Hz, dtype=np.float64))
        inv = (2.0 * rho - 1.0) * self.w_j
        pref0 = self._gain_scale * p.N_q_m3 * p.mu_GS * p.sigma_pk_m2
        if self.mb is None or not self.mb.active:            # free-carrier (un-renormalized chi)
            x = (nu[:, None] - self.nu_j[None, :]) / (0.5 * p.fwhm_hom_Hz)
            g = pref0 * ((1.0 / (1.0 + x * x)) @ inv)
            gi = pref0 * ((x / (1.0 + x * x)) @ inv)
            return (g if np.ndim(nu_Hz) else float(g[0]), gi if np.ndim(nu_Hz) else float(gi[0]))
        mb = self.mb
        T = float(T_K) if T_K is not None else p.T_K
        hw = self._mb_hwhm_Hz(N_carrier_m3, T)
        nu_t = self.nu_j + self._mb_bgr_shift_Hz(N_carrier_m3)         # BGR-shifted comb
        cenh = 1.0 + mb.coulomb_enh * np.exp(-max(float(N_carrier_m3), 0.0) / mb.N_mott_m3)
        osc = (0.5 * p.fwhm_hom_Hz) / hw                              # area-conserving peak factor
        x = (nu[:, None] - nu_t[None, :]) / hw
        pref = pref0 * cenh
        g = pref * ((osc / (1.0 + x * x)) @ inv)                     # gain (Re of the complex line)
        gi = pref * ((osc * x / (1.0 + x * x)) @ inv)               # index partner (Im, KK)
        return (g if np.ndim(nu_Hz) else float(g[0]), gi if np.ndim(nu_Hz) else float(gi[0]))

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
        L = self._L_at(nu_s_Hz)                               # cached (1, ng)

        cap_occ = Nw[:, None] * (1.0 - rho_ES) / self._cap_den
        esc_occ = rho_ES / p.tau_esc_s
        fwd = rho_ES * (1.0 - rho_GS) / p.tau_ES_GS_s
        bwd = rho_GS * (1.0 - rho_ES) / p.tau_GS_ES_s
        gsc = self._gain_scale                                # self-heating gain factor (1.0 off)
        stim = gsc * self._stim_pref * L * (2.0 * rho_GS - 1.0) * Sb
        stim_ES = (gsc * self._stim_pref_ES * self._LE_at(nu_s_Hz) * (2.0 * rho_ES - 1.0) * Sb
                   if self._es_active else 0.0)        # ES optical channel (0 -> GS-only)
        sp_ES = rho_ES * rho_ES / p.tau_sp_s
        sp_GS = rho_GS * rho_GS / p.tau_sp_s

        dN_w = (Ib / self._qVa
                - (Nw / p.tau_cap_s) * np.sum(w * (1.0 - rho_ES), axis=1)
                + self._esc_pref * np.sum(w * rho_ES, axis=1)
                - p.B_wl_m3_s * Nw * Nw - p.C_wl_m6_s * Nw ** 3)
        if self.leak is not None:                             # thermionic WL leakage -N_w/tau_leak(T)
            dN_w = dN_w - self._leak_rate() * Nw
        drho_ES = cap_occ - esc_occ - fwd + (p.mu_GS / p.mu_ES) * bwd - sp_ES - stim_ES
        drho_GS = (p.mu_ES / p.mu_GS) * fwd - bwd - stim - sp_GS
        return dN_w, drho_ES, drho_GS

    def _gs_inversion(self, state) -> np.ndarray:
        """GS inversion used by the gain and the line filter: (2 rho_GS - 1) excitonic, or the
        e/h sum-minus-one (f_c_GS + f_v_GS - 1) in the split. The product f_c f_v cancels exactly
        in (downward f_c f_v) - (upward (1-f_c)(1-f_v)) -> f_c + f_v - 1, so the inversion is
        LINEAR in occupations (transparency at f_c+f_v=1, i.e. rho=1/2)."""
        if self.eh:
            return np.asarray(state[4]) + np.asarray(state[5]) - 1.0
        return 2.0 * np.asarray(state[2]) - 1.0

    def rhs_fields_eh(self, N_w_e, N_w_h, f_c_ES, f_v_ES, f_c_GS, f_v_GS, I_A, S_conf_m3, nu_s_Hz):
        """Vectorized electron/hole-split rate equations (single source of truth for the split
        carrier dynamics; the numba twin _qd_carrier_rk4_eh_numba mirrors it). f_c = electron
        occupation of the conduction confined state, f_v = HOLE occupation of the valence state.
        ONLY the stimulated and spontaneous terms couple the two bands (the SAME scalar subtracted
        from both); everything else is two independent capture/escape/relax ladders. Reduces to
        rhs_fields term-for-term when f_c=f_v=rho, N_w_e=N_w_h, and all hole times = electron times.
        Returns (dN_w_e, dN_w_h, df_c_ES, df_v_ES, df_c_GS, df_v_GS)."""
        p = self.p
        w = self.w_j
        Nwe = np.asarray(N_w_e, dtype=np.float64)
        Nwh = np.asarray(N_w_h, dtype=np.float64)
        S = np.asarray(S_conf_m3, dtype=np.float64)
        Sb = S[:, None] if S.ndim else S
        Ib = np.asarray(I_A, dtype=np.float64)
        L = self._L_at(nu_s_Hz)                               # cached (1, ng)
        inj = Ib / self._qVa
        # --- shared band-coupling scalars (subtracted into BOTH bands of a state) ---
        gsc = self._gain_scale                                # self-heating gain factor (1.0 off)
        inv = f_c_GS + f_v_GS - 1.0                           # GS inversion (sum-minus-one)
        stim = gsc * self._stim_pref * L * inv * Sb           # one e + one h removed per event
        stim_ES = (gsc * self._stim_pref_ES * self._LE_at(nu_s_Hz) * (f_c_ES + f_v_ES - 1.0) * Sb
                   if self._es_active else 0.0)               # ES channel (SAME scalar into both)
        sp_GS = f_c_GS * f_v_GS / p.tau_sp_s                  # spontaneous PRODUCT (not square)
        sp_ES = f_c_ES * f_v_ES / p.tau_sp_s
        R_wl = (p.B_wl_m3_s * Nwe * Nwh                       # pair recomb (charge-neutral)
                + p.C_wl_m6_s * Nwe * Nwh * (Nwe + Nwh) / 2.0)
        # --- WL electrons / holes (pair injection; same R_wl removes one e and one h) ---
        dNwe = (inj - (Nwe / p.tau_cap_s) * np.sum(w * (1.0 - f_c_ES), axis=1)
                + self._esc_pref * np.sum(w * f_c_ES, axis=1) - R_wl)
        dNwh = (inj - (Nwh / self._tcap_h) * np.sum(w * (1.0 - f_v_ES), axis=1)
                + self._esc_pref_h * np.sum(w * f_v_ES, axis=1) - R_wl)
        if self.leak is not None:                             # neutral bipolar WL leakage (e + h)
            lr = self._leak_rate()
            dNwe = dNwe - lr * Nwe
            dNwh = dNwh - lr * Nwh
        # --- ES electrons / holes (independent ladders; back-transfer Pauli on (1-f_*_ES)) ---
        df_c_ES = (Nwe[:, None] * (1.0 - f_c_ES) / self._cap_den - f_c_ES / p.tau_esc_s
                   - f_c_ES * (1.0 - f_c_GS) / p.tau_ES_GS_s
                   + (p.mu_GS / p.mu_ES) * f_c_GS * (1.0 - f_c_ES) / p.tau_GS_ES_s - sp_ES - stim_ES)
        df_v_ES = (Nwh[:, None] * (1.0 - f_v_ES) / self._cap_den_h - f_v_ES / self._tesc_h
                   - f_v_ES * (1.0 - f_v_GS) / self._trel_h
                   + (p.mu_GS / p.mu_ES) * f_v_GS * (1.0 - f_v_ES) / self._tback_h - sp_ES - stim_ES)
        # --- GS electrons / holes (carry the SAME stimulated + spontaneous scalar) ---
        df_c_GS = ((p.mu_ES / p.mu_GS) * f_c_ES * (1.0 - f_c_GS) / p.tau_ES_GS_s
                   - f_c_GS * (1.0 - f_c_ES) / p.tau_GS_ES_s - stim - sp_GS)
        df_v_GS = ((p.mu_ES / p.mu_GS) * f_v_ES * (1.0 - f_v_GS) / self._trel_h
                   - f_v_GS * (1.0 - f_v_ES) / self._tback_h - stim - sp_GS)
        return dNwe, dNwh, df_c_ES, df_v_ES, df_c_GS, df_v_GS

    def rhs(self, y, I_A: float, S_conf_m3: float, nu_s_Hz: float) -> np.ndarray:
        """dy/dt for the single-section state -- a thin wrapper over rhs_fields(_eh) with one
        slice. Excitonic y = [N_w, rho_ES(ng), rho_GS(ng)]; split
        y = [N_w_e, N_w_h, f_c_ES(ng), f_v_ES(ng), f_c_GS(ng), f_v_GS(ng)]."""
        ng = self.ng
        if self.eh:
            b = [y[0:1], y[1:2], y[2:2 + ng][None, :], y[2 + ng:2 + 2 * ng][None, :],
                 y[2 + 2 * ng:2 + 3 * ng][None, :], y[2 + 3 * ng:][None, :]]
            d = self.rhs_fields_eh(b[0], b[1], b[2], b[3], b[4], b[5], I_A, S_conf_m3, nu_s_Hz)
            out = np.empty_like(y)
            out[0] = d[0][0]
            out[1] = d[1][0]
            out[2:2 + ng] = d[2][0]
            out[2 + ng:2 + 2 * ng] = d[3][0]
            out[2 + 2 * ng:2 + 3 * ng] = d[4][0]
            out[2 + 3 * ng:] = d[5][0]
            return out
        dNw, dES, dGS = self.rhs_fields(y[0:1], y[1:1 + ng][None, :], y[1 + ng:][None, :],
                                        I_A, S_conf_m3, nu_s_Hz)
        out = np.empty_like(y)
        out[0] = dNw[0]
        out[1:1 + ng] = dES[0]
        out[1 + ng:] = dGS[0]
        return out

    def _initial_state(self) -> np.ndarray:
        if self.eh:
            y0 = np.zeros(2 + 4 * self.ng)
            y0[0] = y0[1] = 1.0e21                            # WL electron + hole seeds [m^-3]
            y0[2:] = 0.05                                     # nearly-empty dots (all 4 blocks)
            return y0
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
        nd = 2 if self.eh else 1                              # number of WL density entries
        # convergence + physicality checks
        res = self.rhs(y, I_A, S_conf_m3, nu_s)
        scale = max(abs(y[0]), 1.0) / self.p.tau_sp_s
        if np.max(np.abs(res[nd:])) > 1e-6 or np.max(np.abs(res[:nd])) > 1e-3 * scale:
            # one more relaxation leg if not yet converged
            sol = solve_ivp(lambda t, yy: self.rhs(yy, I_A, S_conf_m3, nu_s),
                            (0.0, 10.0 * t_end_s), y, method="BDF", rtol=1e-10, atol=1e-13,
                            t_eval=[10.0 * t_end_s])
            if not sol.success:
                raise RuntimeError("QDGainModel.steady_state: relaxation leg failed "
                                   "({})".format(sol.message))
            y = sol.y[:, -1]
        occ = y[nd:]                                          # re-read AFTER any extra leg
        if np.any(occ < -1e-6) or np.any(occ > 1.0 + 1e-6):
            raise RuntimeError("QDGainModel.steady_state: occupation left [0,1] "
                               "(min {:.3e}, max {:.3e})".format(float(occ.min()),
                                                                 float(occ.max())))
        return y

    def rho_GS(self, y) -> np.ndarray:
        return np.asarray(y)[1 + self.ng:1 + 2 * self.ng]

    def rho_ES(self, y) -> np.ndarray:
        return np.asarray(y)[1:1 + self.ng]

    # ---- e/h-split accessors (state y = [N_w_e, N_w_h, f_c_ES, f_v_ES, f_c_GS, f_v_GS]) ----
    def f_c_ES(self, y) -> np.ndarray:
        return np.asarray(y)[2:2 + self.ng]

    def f_v_ES(self, y) -> np.ndarray:
        return np.asarray(y)[2 + self.ng:2 + 2 * self.ng]

    def f_c_GS(self, y) -> np.ndarray:
        return np.asarray(y)[2 + 2 * self.ng:2 + 3 * self.ng]

    def f_v_GS(self, y) -> np.ndarray:
        return np.asarray(y)[2 + 3 * self.ng:2 + 4 * self.ng]

    def total_electron_density(self, y) -> float:
        """n_tot_e = N_w_e + N_q sum_j w_j (mu_ES f_c_ES_j + mu_GS f_c_GS_j) -- conserved by
        internal transitions (only injection/recomb/stim change it). e/h-split state only."""
        p = self.p
        return float(y[0] + p.N_q_m3 * np.sum(
            self.w_j * (p.mu_ES * self.f_c_ES(y) + p.mu_GS * self.f_c_GS(y))))

    def total_hole_density(self, y) -> float:
        """n_tot_h = N_w_h + N_q sum_j w_j (mu_ES f_v_ES_j + mu_GS f_v_GS_j). e/h-split state."""
        p = self.p
        return float(y[1] + p.N_q_m3 * np.sum(
            self.w_j * (p.mu_ES * self.f_v_ES(y) + p.mu_GS * self.f_v_GS(y))))

    # ---- self-heating: self-consistent steady operating point + ENOB tie-in ----
    def _y_to_slice(self, y):
        """Wrap a steady-state vector y as a 1-slice state tuple (so gain_per_m_slices, which
        handles excitonic/e-h/ES/gain-scale uniformly, can read it)."""
        if self.eh:
            return (np.atleast_1d(y[0]), np.atleast_1d(y[1]),
                    self.f_c_ES(y)[None, :], self.f_v_ES(y)[None, :],
                    self.f_c_GS(y)[None, :], self.f_v_GS(y)[None, :])
        return (np.atleast_1d(y[0]), self.rho_ES(y)[None, :], self.rho_GS(y)[None, :])

    def steady_gain_self_consistent(self, I_A, P_in_W, L_m, *, nu_s_Hz=None):
        """Self-consistent steady single-pass gain under lumped self-heating: iterate the thermal
        fixed point T = T0 + Rth*P_diss, P_diss = I V_j - eta(P_out - P_in), with the gain (hence
        P_out = P_in exp(Gamma g(nu_s) L), lumped input-end saturation; ignores z-saturation, an
        ENOB-budget estimate) re-solved at each T via set_temperature. The feedback is mildly
        DESTABILIZING -- heating cuts the gain, so less optical power is extracted and P_diss rises
        -- with loop gain k = Rth eta |dP_out/dT|; the damped Picard (w_relax) converges to the
        unique fixed point for k < 1 and the max_iter cap RAISES (thermal-runaway guard) otherwise.
        Returns (g_per_m, T_star, G_dB). Isothermal (self_heating None/inactive) -> a single cold
        solve at T0, no iteration."""
        nu_s = float(nu_s_Hz) if nu_s_Hz is not None else self.p.nu0_Hz
        gam = self.p.Gamma

        def cold_gain(T):
            self.set_temperature(T)
            y = self.steady_state(I_A, S_conf_m3=self.photon_density(P_in_W, nu_s), nu_s_Hz=nu_s)
            return float(self.gain_per_m_slices(self._y_to_slice(y), nu_s)[0])

        if self.sh is None or not self.sh.active:
            T0 = self.sh.T0_K if self.sh is not None else self.p.T_K
            g = cold_gain(T0)
            return g, T0, float((10.0 / np.log(10.0)) * gam * g * L_m)
        sh = self.sh
        T = sh.T0_K
        for _ in range(sh.max_iter):
            g = cold_gain(T)
            P_out = P_in_W * np.exp(gam * g * L_m)
            P_diss = I_A * sh.V_j_V - sh.eta_extraction * (P_out - P_in_W)
            T_target = sh.T0_K + sh.Rth_K_W * P_diss
            T_new = (1.0 - sh.w_relax) * T + sh.w_relax * T_target
            if abs(T_new - T) < sh.tol_T_K:
                T = T_new
                self.set_temperature(T)
                g = cold_gain(T)
                return g, T, float((10.0 / np.log(10.0)) * gam * g * L_m)
            T = T_new
        raise RuntimeError("steady_gain_self_consistent: thermal fixed point did not converge in "
                           "{} iterates (raise Rth/lower w_relax, or thermal runaway)".format(
                               sh.max_iter))

    def dGdT_dB_per_K(self, I_A, P_in_W, L_m, T_K, *, nu_s_Hz=None, dT=0.5):
        """Single-pass gain temperature sensitivity dG/dT [dB/K] at T_K (central difference via
        set_temperature). Feeds metrics.thermal_drift_budget_K -> the predistortion ENOB ceiling.
        Requires self_heating (the coupling coefficients). Restores T_K on exit."""
        nu_s = float(nu_s_Hz) if nu_s_Hz is not None else self.p.nu0_Hz
        c = (10.0 / np.log(10.0)) * self.p.Gamma * L_m

        def G_dB(T):
            self.set_temperature(T)
            y = self.steady_state(I_A, S_conf_m3=self.photon_density(P_in_W, nu_s), nu_s_Hz=nu_s)
            return c * float(self.gain_per_m_slices(self._y_to_slice(y), nu_s)[0])

        hi, lo = G_dB(T_K + dT), G_dB(T_K - dT)
        self.set_temperature(T_K)
        return (hi - lo) / (2.0 * dT)

    # ---- small-signal + saturation ----
    def small_signal_gain_per_m(self, I_A: float, nu_Hz=None) -> np.ndarray:
        """Unsaturated (S_conf -> 0) modal gain spectrum at injection I, GS + (if active) ES band.
        Excitonic state only (the two-state crossover oracle uses the excitonic model); for the
        e/h split read spectra via gain_per_m_slices."""
        y = self.steady_state(I_A, S_conf_m3=0.0)
        nu = self.p.nu0_Hz if nu_Hz is None else nu_Hz
        return self.total_material_gain(self.rho_ES(y), self.rho_GS(y), nu)

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

    @property
    def alpha_lef(self) -> float:
        """Linewidth enhancement factor (carrier-induced index / gain) -- the coherent
        propagation phase term and the FWM asymmetry."""
        return self.p.alpha_lef

    @property
    def beta2_s2_per_m(self) -> float:
        """Background (waveguide) group-velocity dispersion d2 beta / d omega^2 [s^2/m] -- the
        broadband non-resonant index that amplify_coherent applies as a Fourier split-step."""
        return self.p.beta2_s2_per_m

    def alpha_lef_slices(self, state):
        """Per-slice linewidth enhancement factor alpha(rho_GS) = alpha_lef +
        alpha_lef_density_slope * (rho_GS - 1/2), with rho_GS the ensemble-mean GS occupation of each
        slice (rho_GS - 1/2 = inversion/2). alpha rises with carrier density as the gain clamps
        (dg/dN falls while the carrier-induced index dn/dN persists), the measured behaviour in QD
        SOAs. Returns the SCALAR alpha_lef when the slope is 0 (so the coherent engine is
        byte-identical), else a (nz,) array. The frequency dependence of the carrier-induced index is
        carried separately by the resonant Kramers-Kronig line filter (amplify_coherent line_filter)."""
        sl = self.p.alpha_lef_density_slope
        if sl == 0.0:
            return self.p.alpha_lef
        inv = self._gs_inversion(state)                       # (nz, ng) = 2 rho_GS - 1
        inv_mean = np.tensordot(inv, self.w_j, axes=([-1], [0])) / np.sum(self.w_j)   # (nz,)
        return self.p.alpha_lef + 0.5 * sl * inv_mean         # alpha(rho_mean), slope d(alpha)/d(rho)

    def photon_density(self, P_W, nu_Hz):
        """Confined photon density S_conf [m^-3] for guided power P (scalar or array)."""
        p = self.p
        return p.Gamma * np.asarray(P_W, dtype=np.float64) / (
            p.v_g_m_s * H_PLANCK * nu_Hz * p.A_mode_m2)

    def init_slices(self, n_slices: int, I_A):
        """Per-slice carrier state initialized to the unsaturated steady state at injection I.
        Excitonic: (N_w, rho_ES, rho_GS); e/h split: (N_w_e, N_w_h, f_c_ES, f_v_ES, f_c_GS, f_v_GS).
        I_A scalar -> uniform along z (byte-identical). I_A as a (n_slices,) array is a NON-UNIFORM
        injection PROFILE I(z) (e.g. from a drift-diffusion / DEVSIM solve, or current crowding) --
        each slice is seeded at its OWN local steady state (rhs_fields already accepts a per-slice
        I_A, so the marcher carries the profile through the dynamics)."""
        nz = int(n_slices)
        Iarr = np.asarray(I_A, dtype=np.float64)
        if Iarr.ndim == 0:                                    # uniform (the original path)
            y = self.steady_state(float(Iarr))
            if self.eh:
                return (np.full(nz, y[0]), np.full(nz, y[1]),
                        np.tile(self.f_c_ES(y), (nz, 1)), np.tile(self.f_v_ES(y), (nz, 1)),
                        np.tile(self.f_c_GS(y), (nz, 1)), np.tile(self.f_v_GS(y), (nz, 1)))
            return (np.full(nz, y[0]), np.tile(self.rho_ES(y), (nz, 1)),
                    np.tile(self.rho_GS(y), (nz, 1)))
        if Iarr.size != nz:
            raise ValueError("init_slices: injection profile I_A must be scalar or length n_slices")
        ys = [self.steady_state(float(ii)) for ii in Iarr]    # per-slice local steady state
        if self.eh:
            return (np.array([y[0] for y in ys]), np.array([y[1] for y in ys]),
                    np.array([self.f_c_ES(y) for y in ys]), np.array([self.f_v_ES(y) for y in ys]),
                    np.array([self.f_c_GS(y) for y in ys]), np.array([self.f_v_GS(y) for y in ys]))
        return (np.array([y[0] for y in ys]),
                np.array([self.rho_ES(y) for y in ys]),
                np.array([self.rho_GS(y) for y in ys]))

    def gain_per_m_slices(self, state, nu_Hz) -> np.ndarray:
        """Material intensity gain g(nu) [1/m] per slice = GS band + (if sigma_pk_ES>0) the ES
        band g_ES = sum_j N_q w_j mu_ES sigma_pk_ES L_ES(nu-nu_ES_j)(INV_ES). GS inversion via
        _gs_inversion (excitonic 2 rho_GS-1 / e/h f_c_GS+f_v_GS-1); ES via _es_inversion. With
        sigma_pk_ES=0 the ES branch is skipped -> byte-identical GS-only gain."""
        wl = self._gain_line_weights(nu_Hz)                   # cached (ng,) = w_j * L(nu - nu_j)
        g = self._gain_scale * self._gain_pref * np.sum(self._gs_inversion(state) * wl[None, :], axis=1)
        if self._es_active:
            wlE = self._gain_line_weights_ES(nu_Hz)
            g = g + self._gain_scale * self._gain_pref_ES * np.sum(
                self._es_inversion(state) * wlE[None, :], axis=1)
        return g

    def wl_density_slices(self, state) -> np.ndarray:
        """Wetting-layer (reservoir) carrier density N_w [m^-3] per slice -- state[0] for both the
        excitonic (N_w) and e/h-split (N_w_e) layouts. The free-carrier reservoir that drives the
        dynamic free-carrier absorption (TravelingWaveSOA NonlinearLoss): alpha_FCA = sigma_FCA N_w
        makes the internal loss depend on pumping/saturation rather than a fixed alpha_i.

        SINGLE-RESERVOIR proxy: this is the WL density ONLY (in the e/h-split layout the ELECTRON WL
        N_w_e alone; the hole WL N_w_h = state[1] and the confined ES/GS carriers are NOT summed in).
        sigma_FCA is therefore an effective lumped cross-section calibrated to this one density, not a
        first-principles per-species sigma."""
        return np.asarray(state[0], dtype=np.float64)

    def emission_gain_per_m_slices(self, state, nu_Hz) -> np.ndarray:
        """Spontaneous-EMISSION gain g_sp(nu) [1/m] per slice (GS band) = sum_j N_q w_j mu_GS sigma_pk
        L(nu-nu_j) rho_GS_j^2 (excitonic) or f_c_GS f_v_GS (e/h split) -- the per-slice upper-state
        population, hence the LANGEVIN / ASE spontaneous-source amplitude. Mirrors emission_gain_per_m
        for the slice state layout; always >= 0 (pole-free)."""
        wl = self._gain_line_weights(nu_Hz)                   # cached (ng,) = w_j * L(nu - nu_j)
        if self.eh:
            em = np.asarray(state[4]) * np.asarray(state[5])  # f_c_GS * f_v_GS  (nz, ng)
        else:
            rho = np.asarray(state[2])
            em = rho * rho                                    # rho_GS^2  (nz, ng)
        return self._gain_scale * self._gain_pref * np.sum(em * wl[None, :], axis=1)

    def gain_per_m_thermal(self, state, nu_Hz, T_z) -> np.ndarray:
        """Per-slice GS gain g(nu_s) [1/m] at a SPATIALLY-RESOLVED temperature profile T_z (n_slices):
        each slice's comb is red-shifted by dnu0_dT (T_z - T0) and the gain scaled by 1 + dg_dT_frac
        (T_z - T0) -- the per-slice generalization of the lumped set_temperature. Includes the ES band
        (same red-shift + scale on the blue-shifted ES comb) when it is active (sigma_pk_ES > 0).
        Reduces to gain_per_m_slices when T_z == the nominal T0, and to the lumped set_temperature(T)
        gain when T_z is uniform. A reduced 1-D fin (optics.soa.thermal) OR the thermal FEM
        (carriers.thermal_fem, sampled via sample_T_along_axis) supplies T_z through this interface.
        Requires self_heating (the dnu0_dT / dg_dT_frac coefficients).

        Precondition -- MUTUALLY EXCLUSIVE with the lumped set_temperature(): this method reads the COLD
        combs _nu_j0/_nu_ES_j0 and computes its own per-slice scale, deliberately ignoring any
        set_temperature() state (so it never double-counts). Consequently the identity
        gain_per_m_thermal(., ., T0) == gain_per_m_slices holds only on a model whose set_temperature()
        has NOT been engaged (self._gain_scale == 1). Do not mix the two thermal representations on one
        model instance -- pick the lumped single-T path or this per-slice path, not both."""
        if self.sh is None:
            raise ValueError("gain_per_m_thermal requires a SelfHeating (the dnu0_dT/dg_dT_frac coeffs)")
        p, sh = self.p, self.sh
        nu = float(nu_Hz)
        dT = np.atleast_1d(np.asarray(T_z, dtype=np.float64)) - sh.T0_K       # (nz,)
        scale = np.clip(1.0 + sh.dg_dT_frac_per_K * dT, 0.0, None)            # (nz,) per-slice scale
        nu_j_z = self._nu_j0[None, :] - sh.dnu0_dT_Hz_K * dT[:, None]         # (nz, ng) shifted GS comb
        L = self._lorentzian(nu - nu_j_z)                                    # (nz, ng)
        g = scale * p.N_q_m3 * p.mu_GS * p.sigma_pk_m2 * np.sum(
            self._gs_inversion(state) * L * self.w_j[None, :], axis=1)
        if self._es_active:                          # ES band: same red-shift + scale on the ES comb
            nu_ES_j_z = self._nu_ES_j0[None, :] - sh.dnu0_dT_Hz_K * dT[:, None]
            LE = self._lorentzian_ES(nu - nu_ES_j_z)
            g = g + scale * self._gain_pref_ES * np.sum(
                self._es_inversion(state) * LE * self.w_j[None, :], axis=1)
        return g

    def gain_per_m_nonmarkovian(self, state, nu_Hz, *, gamma2_factor, w1) -> np.ndarray:
        """GS gain g(nu) [1/m] per slice with a TWO-TIMESCALE (heterogeneous-rate) biexponential
        homogeneous line: the single Lorentzian (HWHM hw = fwhm_hom/2, single dephasing T2) is replaced
        by the two-channel line w1 L(hw) + (1-w1) L(hw*gamma2_factor) -- a multi-rate dephasing that
        gives a SUPER-Lorentzian (sharper core + HEAVIER wing) gain line. Reduces EXACTLY to
        gain_per_m_slices (GS band) when w1 = 1 or gamma2_factor = 1 (the single-rate Lorentzian limit).
        GS band only (the inhomogeneous comb + inversion are unchanged).

        CONVENTION: this uses the gain's PEAK-normalized Lorentzian (hw^2/(dnu^2+hw^2)=1 at line centre,
        sigma_pk the peak cross-section), so broadening the wing channel (gamma2_factor>1) INFLATES the
        integrated gain (sum w1 + (1-w1)gamma2_factor) -- it is NOT oscillator-strength conserving. The
        AREA-normalized lineshape <-> biexp-memory-kernel Fourier pair (optics.soa.lineshape) is the
        spectroscopy-convention twin; the two normalizations conserve peak vs area respectively."""
        if not (0.0 <= float(w1) <= 1.0):
            raise ValueError("gain_per_m_nonmarkovian: w1 must be in [0, 1]")
        p = self.p
        dnu = float(nu_Hz) - self.nu_j                       # (ng,)
        hw = 0.5 * p.fwhm_hom_Hz
        hw2 = hw * float(gamma2_factor)
        L = float(w1) * (hw * hw) / (dnu * dnu + hw * hw) + (1.0 - float(w1)) * (hw2 * hw2) / (
            dnu * dnu + hw2 * hw2)
        return self._gain_scale * p.N_q_m3 * p.mu_GS * p.sigma_pk_m2 * np.sum(
            self._gs_inversion(state) * (self.w_j * L)[None, :], axis=1)

    def line_kappa_slices(self, state, nu_s_Hz, hw_Hz) -> np.ndarray:
        """Per-slice, per-group complex-Lorentzian line-filter DRIVE kappa_j[k] (real), the source
        term of the Maxwell-Bloch polarization ADE used by the spectral-dispersion path of
        TravelingWaveSOA.amplify_coherent(line_filter=True). With the per-group pole
            lam_j = -2 pi hw + 1j 2 pi (nu_s - nu_j),
        drive kappa_j[k] = (2 pi hw) * A_j[k], and readout (1/A) sum_j p_j, the steady-state band
        response of a tone at offset f reproduces EXACTLY 2 * Gamma_field(nu_s + f), where
            Gamma_field(nu) = 0.5 sum_j A_j / (1 - 1j (nu - nu_j)/hw),  2 Re == g(nu),
            A_j[k] = N_q w_j mu_GS sigma_pk (2 rho_GS_j[k] - 1)  [the SAME assembly as the gain],
        so the line shape AND its Kramers-Kronig dispersive partner are carried by one pole per
        group (no FFT, no tone comb). Returns kappa, shape (nz, ng) [s^-1], from the LIVE GS
        inversion (so carrier-density pulsation rides on top -> dispersion-enlarged FWM/XGM).
        Uses _gs_inversion so the e/h split (f_c_GS+f_v_GS-1) drives the line filter too. Scaled by
        the self-heating gain factor so the dispersive Kramers-Kronig partner stays consistent."""
        return ((2.0 * np.pi * hw_Hz) * self._gain_scale * self._gain_pref
                * self.w_j[None, :] * self._gs_inversion(state))

    def step_slices(self, state, P_local_W, dt_s: float, nu_s_Hz: float, I_A: float):
        """Advance the per-slice carrier state by dt driven by the local guided POWER P (Nz,)
        held fixed across the step (operator splitting); explicit RK4. Power is converted to
        the confined photon density internally so the traveling-wave engine speaks one
        currency (power) to every slab model."""
        if self.eh:
            return self._step_slices_eh(state, P_local_W, dt_s, nu_s_Hz, I_A)
        Nw, rES, rGS = state
        S_conf = self.photon_density(P_local_W, nu_s_Hz)

        if self._use_numba:                                   # compiled twin (bit-parity, ~5-7x)
            p = self.p
            Sa = np.asarray(S_conf, dtype=np.float64)         # photon density (nz,) or scalar
            # Nw/rES/rGS are already C-contiguous (kernel output or init_slices) -> no copy; the
            # confined density only needs broadcasting/copying when it is a scalar; the Lorentzian
            # row is the cached constant (avoid recomputing every step).
            S = (np.ascontiguousarray(np.broadcast_to(Sa, Nw.shape))
                 if Sa.shape != Nw.shape else Sa)
            LE = self._LE_at(nu_s_Hz)[0] if self._es_active else self._zeros_ng
            gsc = self._gain_scale                            # self-heating gain factor (1.0 off)
            return _qd_carrier_rk4_numba(
                Nw, rES, rGS, S, float(I_A), float(dt_s),
                self._L_at(nu_s_Hz)[0], self.w_j,
                self._cap_den, self._esc_pref, gsc * self._stim_pref, p.tau_cap_s, p.tau_esc_s,
                p.tau_ES_GS_s, p.tau_GS_ES_s, p.tau_sp_s, p.B_wl_m3_s, p.C_wl_m6_s,
                p.mu_GS / p.mu_ES, p.mu_ES / p.mu_GS, self._qVa,
                gsc * self._stim_pref_ES, LE, self._es_active, self._leak_rate())

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

    def _step_slices_eh(self, state, P_local_W, dt_s, nu_s_Hz, I_A):
        """RK4 advance of the e/h-split slice state (N_w_e, N_w_h, f_c_ES, f_v_ES, f_c_GS,
        f_v_GS) by dt at fixed local power; numba twin when fast=True, else numpy rhs_fields_eh."""
        S_conf = self.photon_density(P_local_W, nu_s_Hz)
        if self._use_numba:                                   # compiled e/h twin (bit-parity)
            p = self.p
            Sa = np.asarray(S_conf, dtype=np.float64)
            base = state[0]
            S = (np.ascontiguousarray(np.broadcast_to(Sa, base.shape))
                 if Sa.shape != base.shape else Sa)
            LE = self._LE_at(nu_s_Hz)[0] if self._es_active else self._zeros_ng
            gsc = self._gain_scale                            # self-heating gain factor (1.0 off)
            return _qd_carrier_rk4_eh_numba(
                state[0], state[1], state[2], state[3], state[4], state[5], S,
                float(I_A), float(dt_s), self._L_at(nu_s_Hz)[0], self.w_j,
                self._cap_den, self._cap_den_h, self._esc_pref, self._esc_pref_h,
                gsc * self._stim_pref,
                p.tau_cap_s, self._tcap_h, p.tau_esc_s, self._tesc_h, p.tau_ES_GS_s, self._trel_h,
                p.tau_GS_ES_s, self._tback_h, p.tau_sp_s, p.B_wl_m3_s, p.C_wl_m6_s,
                p.mu_GS / p.mu_ES, p.mu_ES / p.mu_GS, self._qVa,
                gsc * self._stim_pref_ES, LE, self._es_active, self._leak_rate())

        def f(s):
            return self.rhs_fields_eh(s[0], s[1], s[2], s[3], s[4], s[5], I_A, S_conf, nu_s_Hz)

        s0 = state
        k1 = f(s0)
        k2 = f([s0[i] + 0.5 * dt_s * k1[i] for i in range(6)])
        k3 = f([s0[i] + 0.5 * dt_s * k2[i] for i in range(6)])
        k4 = f([s0[i] + dt_s * k3[i] for i in range(6)])
        out = [s0[i] + dt_s / 6.0 * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]) for i in range(6)]
        # densities >= 0, occupations in [0, 1] (overshoot guard at the bounds)
        return (np.maximum(out[0], 0.0), np.maximum(out[1], 0.0),
                np.clip(out[2], 0.0, 1.0), np.clip(out[3], 0.0, 1.0),
                np.clip(out[4], 0.0, 1.0), np.clip(out[5], 0.0, 1.0))
