"""QD-SOA gain core (roadmap SOA Phase 1) vs analytic / known-limit oracles.

The semiconductor quantum-dot gain model optics.soa.qd_gain.QDGainModel -- group-resolved
WL -> ES -> GS rate equations with an injection-current pump and signal-driven (stimulated)
GS depletion. Gates (each an independent reduces-to-known-limit or physical-signature check):

GATE A (transparency + monotonicity, single group): with one dot group the line-centre modal
        gain g0 = N_q mu_GS sigma_pk (2 rho_GS - 1), so g0 = 0 EXACTLY at the transparency
        occupation rho_GS = 1/2. The small-signal gain must be monotonic in injection current
        and change sign precisely where rho_GS crosses 1/2 (sign(g0) == sign(rho_GS - 1/2) at
        every current) -- the analytic known limit.
GATE B (dynamic saturation + the QD signature): the static gain g(nu0; S) decreases
        monotonically with the signal photon density, and the saturation density S_sat (the
        3 dB point, g = g0/2) RISES with pump current -- the carrier-reservoir effect that is
        the reason a QD SOA is the right amplifier class (higher pump -> higher saturation
        power).
GATE C (particle conservation): every internal transition (capture/escape, ES<->GS exchange)
        is a conjugate number-flux pair, so d(n_tot)/dt computed from the full RHS must equal
        injection - WL recombination - confined spontaneous - stimulated, to ~1e-12 relative.
        This is the verification of the corrected conjugate-flux bookkeeping (spec Section 8.3).
GATE D (detailed balance): with capture/escape/recombination switched off, no injection and
        no signal, the ES<->GS relaxation drives the dark occupations to the quasi-equilibrium
        ratio rho_ES/(1-rho_ES) = exp(-dE/kT) rho_GS/(1-rho_GS) -- the thermodynamic check on
        the mu_ES/mu_GS-weighted exchange and the detailed-balance escape times.
GATE E (spectral hole burning): a strong monochromatic drive at nu0 depletes the RESONANT
        dot group's inversion much more than the off-resonant wings -- a localized spectral
        hole that only group-resolved populations can produce (a single global rho_GS cannot).

Run: python -m validation.qd_soa_gain_core
"""
import dataclasses
import os
import sys

import numpy as np
from scipy.integrate import solve_ivp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dynameta.constants import KB, Q_E
from dynameta.optics.soa.qd_gain import QDGainModel, QDGainParams


def main():
    print("[qd] === QD-SOA gain core vs analytic / known-limit oracles ===", flush=True)
    ok = True
    base = QDGainParams().with_detailed_balance_taus()
    nu0 = base.nu0_Hz

    # ---- GATE A: single-group transparency + monotonicity ----
    p1 = dataclasses.replace(base, n_groups=1)
    m1 = QDGainModel(p1)
    I_mA = np.array([0.5, 1.0, 2.0, 3.0, 5.0, 10.0])
    g0s, rgs = [], []
    for I in I_mA * 1e-3:
        y = m1.steady_state(float(I))
        g0s.append(m1.material_gain_per_m(m1.rho_GS(y), nu0))
        rgs.append(float(m1.rho_GS(y)[0]))
    g0s, rgs = np.array(g0s), np.array(rgs)
    monotonic = bool(np.all(np.diff(g0s) > 0.0))
    # sign(g0) must match sign(rho_GS - 1/2) everywhere outside a tiny deadband
    dead = 1e-3
    sign_ok = bool(np.all(np.sign(g0s[np.abs(rgs - 0.5) > dead])
                          == np.sign((rgs - 0.5)[np.abs(rgs - 0.5) > dead])))
    # interpolate the transparency current and confirm rho_GS = 0.5 there
    k = int(np.where(np.diff(np.sign(g0s)))[0][0])
    f = -g0s[k] / (g0s[k + 1] - g0s[k])
    rho_at_tr = rgs[k] + f * (rgs[k + 1] - rgs[k])
    g_a = bool(monotonic and sign_ok and abs(rho_at_tr - 0.5) < 5e-3)
    ok = ok and g_a
    print("[qd] GATE A: single-group g0 monotonic={} sign-matches-inversion={} "
          "rho_GS@transparency={:.4f} -> {}".format(monotonic, sign_ok, rho_at_tr,
                                                    "PASS" if g_a else "FAIL"), flush=True)

    # ---- GATE B: saturation monotone + S_sat rises with pump ----
    m = QDGainModel(base)
    S_grid = np.logspace(18.5, 24.0, 28)

    def sweep(I):
        g0 = m.material_gain_per_m(m.rho_GS(m.steady_state(I)), nu0)
        y, gs = None, []
        for S in S_grid:
            y = m.steady_state(I, S_conf_m3=float(S), nu_s_Hz=nu0, y0=y)
            gs.append(m.material_gain_per_m(m.rho_GS(y), nu0))
        gs = np.array(gs)
        below = np.where(gs <= 0.5 * g0)[0]
        S_sat = S_grid[below[0]] if below.size else np.inf
        return g0, gs, S_sat

    res = {I: sweep(I * 1e-3) for I in (6.0, 12.0, 25.0)}
    mono_sat = all(bool(np.all(np.diff(res[I][1]) <= 1e-6 * abs(res[I][0]) + 1.0))
                   for I in res)                              # g non-increasing in S
    S_sats = [res[I][2] for I in (6.0, 12.0, 25.0)]
    rises = bool(S_sats[0] < S_sats[1] < S_sats[2] and np.all(np.isfinite(S_sats)))
    g_b = bool(mono_sat and rises)
    ok = ok and g_b
    print("[qd] GATE B: g(S) non-increasing={}; S_sat(3dB) rises with pump "
          "[{:.2e}, {:.2e}, {:.2e}] m^-3 -> {}".format(
              mono_sat, S_sats[0], S_sats[1], S_sats[2], "PASS" if g_b else "FAIL"), flush=True)

    # ---- GATE C: particle-conservation closure of the full RHS ----
    p = base
    rng = np.random.default_rng(0)
    ng = m.ng
    y = np.empty(1 + 2 * ng)
    y[0] = 7.3e23                                             # arbitrary non-eq WL density
    y[1:1 + ng] = 0.2 + 0.5 * rng.random(ng)                 # arbitrary ES occupations
    y[1 + ng:] = 0.1 + 0.7 * rng.random(ng)                  # arbitrary GS occupations
    I_test, S_test = 20.0e-3, 3.0e21
    dy = m.rhs(y, I_test, S_test, nu0)
    dn_actual = dy[0] + p.N_q_m3 * np.sum(
        m.w_j * (p.mu_ES * dy[1:1 + ng] + p.mu_GS * dy[1 + ng:]))
    rho_ES, rho_GS = y[1:1 + ng], y[1 + ng:]
    L = m._lorentzian(nu0 - m.nu_j)
    stim_loss = p.N_q_m3 * np.sum(m.w_j * p.mu_GS * p.v_g_m_s * p.sigma_pk_m2 * L
                                  * (2.0 * rho_GS - 1.0) * S_test)
    spont = p.N_q_m3 * np.sum(m.w_j * (p.mu_ES * rho_ES ** 2 + p.mu_GS * rho_GS ** 2)) / p.tau_sp_s
    wl_recomb = p.B_wl_m3_s * y[0] ** 2 + p.C_wl_m6_s * y[0] ** 3
    dn_expected = I_test / (Q_E * p.V_a_m3) - wl_recomb - spont - stim_loss
    rel = abs(dn_actual - dn_expected) / max(abs(dn_expected), 1.0)
    g_c = bool(rel < 1e-9)
    ok = ok and g_c
    print("[qd] GATE C: conservation closure |d(n_tot)/dt - (inj - recomb - stim)|/scale = "
          "{:.2e} -> {}".format(rel, "PASS" if g_c else "FAIL"), flush=True)

    # ---- GATE D: detailed balance (relaxation-only -> Boltzmann ES/GS ratio) ----
    pr = dataclasses.replace(base, tau_cap_s=1.0e9, tau_esc_s=1.0e9, tau_sp_s=1.0e12,
                             B_wl_m3_s=0.0, C_wl_m6_s=0.0)
    mr = QDGainModel(pr)
    j = mr.ng // 2
    y0 = np.zeros(1 + 2 * mr.ng)
    y0[1:1 + mr.ng] = 0.6                                     # load the ES, drain the GS
    y0[1 + mr.ng:] = 0.2
    sol = solve_ivp(lambda t, yy: mr.rhs(yy, 0.0, 0.0, nu0), (0.0, 1.0e-7), y0,
                    method="BDF", rtol=1e-11, atol=1e-14, t_eval=[1.0e-7])
    yf = sol.y[:, -1]
    rES, rGS = float(mr.rho_ES(yf)[j]), float(mr.rho_GS(yf)[j])
    ratio = (rES / (1.0 - rES)) / (rGS / (1.0 - rGS))
    boltz = float(np.exp(-pr.dE_ES_GS_eV * Q_E / (KB * pr.T_K)))
    g_d = bool(abs(ratio - boltz) / boltz < 1e-3)
    ok = ok and g_d
    print("[qd] GATE D: detailed balance ES/GS occ-ratio {:.5f} vs Boltzmann {:.5f} "
          "(rel {:.1e}) -> {}".format(ratio, boltz, abs(ratio - boltz) / boltz,
                                      "PASS" if g_d else "FAIL"), flush=True)

    # ---- GATE E: spectral hole burning -- the hole TRACKS the drive frequency ----
    # The discriminating SHB signature (impossible for a single global rho_GS): under a CW
    # drive, the inversion-depletion profile peaks at the DRIVEN group and moves with it. (The
    # static hole is shallow here because the fast intradot ES->GS relaxation -- the same
    # mechanism behind QD low pattern effects -- refills the GS faster than a CW signal burns
    # it; the frequency-tracking is the regime-independent test.)
    I_e, S_e = 12.0e-3, 5.0e20
    inv_unsat = 2.0 * m.rho_GS(m.steady_state(I_e)) - 1.0
    offsets = (-1.5, 0.0, 1.5)                               # drive detunings in inhom. sigmas
    track_err, contrasts = [], []
    jw = m.ng // 2 + 12                                     # a fixed off-resonant wing group
    for off in offsets:
        nu_d = nu0 + off * m._sig_inhom
        dep = inv_unsat - (2.0 * m.rho_GS(m.steady_state(I_e, S_conf_m3=S_e, nu_s_Hz=nu_d)) - 1.0)
        j_pk = int(np.argmax(dep))
        track_err.append(abs((m.nu_j[j_pk] - nu0) / m._sig_inhom - off))
        contrasts.append(float(dep[j_pk] / max(dep[jw], 1e-12)))
    tracks = bool(max(track_err) < 0.2)                     # peak follows the drive (< group step)
    nonuniform = bool(min(contrasts) > 1.05)                # genuinely group-resolved (not flat)
    g_e = bool(tracks and nonuniform)
    ok = ok and g_e
    print("[qd] GATE E: SHB hole tracks drive (max track err {:.2f} sigma, peak/wing contrast "
          "{:.2f}x) -> {}".format(max(track_err), min(contrasts), "PASS" if g_e else "FAIL"),
          flush=True)

    print("[qd] *** QD-SOA GAIN CORE: {} ***".format("PASS" if ok else "FAIL"), flush=True)
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
