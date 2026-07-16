"""LASING: the dynamic four-level gain medium (R20 follow-on) inside a Fabry-Perot cavity,
vs the hand-derived class-B laser rate-equation oracle (laser_gain cavity-design formulas).

Testbed: a 10.6 um slab of n_c = sqrt(2) host (gain window = the whole slab, Gamma = 1)
between eps = 100 (n_out = 10) pads index-matched into the CPML -- the two Fresnel steps are
the mirrors (R = 0.566 from inside). L_c = 25 half-waves of lambda_a/(2 n_c), so the m = 25
longitudinal mode sits on the line center w_a to ~1e-3 of an FSR, and FSR = dw/2 biases
single-mode operation. Lifetimes re-pinned vs the saturation testbed (tau21 = 5 ps >> tau_p
~ 88 fs >> T2 = 16 fs) so relaxation oscillations are underdamped class-B. There is no
spontaneous emission in the kernel: every run is seeded deterministically by a weak in-cavity
burst; below threshold the field DECAYS (no self-oscillation), above it grows, rings, and
clamps -- exactly the rate-equation picture.

Gate sequencing (each consumes the previous gate's MEASURED quantity, per the oracle pin):
GATE A (cold cavity): unpumped ring-down -> tau_p measured == cavity_photon_lifetime_s to 5%.
GATE B (threshold): the seeded field's net envelope rate gamma_f(Wp) = (v_g g0(dN_ss) -
        1/tau_p)/2 at four pumps straddling W_p_th -- measured vs predicted (using the
        MEASURED tau_p) to 5% of the loss rate, and the sign flips across threshold
        (pump_threshold_per_s).
GATE C (clamping/saturation, STANDING-WAVE form): the saturation law is local, so in the
        cavity dN(z) = dN0/(1 + u sin^2(kz)) with u = E_antinode^2/A_sat^2, and the
        mode-weighted clamp <dN sin^2>/<sin^2> = dN_th gives the hand-derived condition

            r * (2/u) * (1 - 1/sqrt(1+u)) = 1     (u = 1.448 at r = 2; u = 3 EXACTLY at r = 3)

        -- the no-grating LOWER bound on the intracavity intensity (inversion-grating /
        side-mode effects only RAISE the intensity needed to clamp). Gates: the r3/r2 ratio
        of measured u within 15% of the predicted ratio (2.07; systematics largely cancel),
        and each absolute u within the declared [0.95, 2.0] x floor band.
GATE D (relaxation oscillations): the ring-in envelope oscillation frequency at r = 2 and
        r = 3 matches relaxation_oscillation_rad_s within the declared [hot, cold] tau_p
        bracket +- 20% (the gain line adds slow light Delta n_g = c alpha_m / dw at clamp).
GATE E (conservation): sum(N) == NTOT in the gain window to < 1e-9 after the longest run.

Run: python -m validation.fdtd_lasing_cavity   (numpy kernel; ~6-8 min)
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dynameta.constants import C_LIGHT, EPS0, HBAR, M_E, Q_E
from dynameta.optics.fdtd_nd import cpml_z, run_2d_te
from dynameta.optics.laser_gain import (FourLevelSystem, cavity_photon_lifetime_s,
                                         pump_threshold_per_s, relaxation_oscillation_rad_s,
                                         small_signal_gain_per_m, threshold_inversion_m3)

# ---- testbed pin (module docstring) ----
N_C = np.sqrt(2.0)
N_OUT = 10.0
NX = 4
DZ = 10e-9
N_STR = 1060                     # L_c = 10.6 um = 25 half-waves of lambda_a/(2 n_c)
N_PAD = 80
NPML = 12
KAPPA = Q_E ** 2 / M_E
W_A = 2.0 * np.pi * 2.5e14
DW = 2.0 * np.pi * 2.0e13
T32, T21, T10 = 1.0e-14, 5.0e-12, 5.0e-15
NTOT = 1.0e25

NZ = 2 * N_PAD + N_STR
DX = 4.0 * DZ
DT = 0.5 / (C_LIGHT * np.sqrt(1.0 / DX ** 2 + 1.0 / DZ ** 2))
L_C = N_STR * DZ
R_MIRROR = ((N_C - N_OUT) / (N_C + N_OUT)) ** 2
A_SAT2 = 2.0 * DW * HBAR * W_A / (KAPPA * T21)

K_PROBE = N_PAD + N_STR // 2                       # mid-cavity antinode (m = 25 is odd)
K_SRC = K_PROBE - 42                               # ~one half-wave away (adjacent antinode)


def _grids():
    eps = np.full((NX, NZ), N_OUT ** 2)
    eps[:, N_PAD:N_PAD + N_STR] = N_C ** 2
    win = np.zeros((NX, NZ))
    win[:, N_PAD:N_PAD + N_STR] = 1.0
    den = 1.0 + DW * DT / 2.0
    G1 = np.full((NX, NZ), (2.0 - W_A ** 2 * DT ** 2) / den)
    G2 = np.full((NX, NZ), (DW * DT / 2.0 - 1.0) / den)
    kapfac = KAPPA * DT ** 2 / den * win
    return eps, win, G1, G2, kapfac


def _run(Wp_per_s, nsteps, seed_amp, out=None, snap_step=10):
    eps, win, G1, G2, kapfac = _grids()
    zeros = np.zeros((NX, NZ))
    sysm = FourLevelSystem(tau_32_s=T32, tau_21_s=T21, tau_10_s=T10,
                           W_p_per_s=max(Wp_per_s, 0.0), N_total_m3=NTOT)
    ss = sysm.steady_state()
    Npop0 = np.stack([np.where(win > 0, ss[0], NTOT), win * ss[1], win * ss[2], win * ss[3]])
    t = np.arange(nsteps) * DT
    t0, tau_s = 30e-15 * 6.0, 30e-15
    src = seed_amp * np.exp(-((t - t0) / tau_s) ** 2) * np.cos(W_A * (t - t0))
    cpml = cpml_z(NZ, DZ, DT, NPML, N_OUT, N_OUT)
    gd = (G1, G2, kapfac, Wp_per_s * win, Npop0, T32, T21, T10, HBAR * W_A, snap_step)
    _, _, eyR, _ = run_2d_te(eps, zeros, zeros, zeros, DX, DZ, DT, nsteps, K_SRC,
                              N_PAD // 2, K_PROBE, src, cpml, np, None,
                              gain_dyn=gd, gain_dyn_out=out)
    return eyR.mean(axis=1)                        # uniform in x -> average the 4 columns


def _envelope(sig):
    from scipy.signal import hilbert
    return np.abs(hilbert(sig))


def _fit_rate(env, t, t_lo, t_hi):
    """ln-linear envelope rate [1/s] over the window [t_lo, t_hi]."""
    m = (t >= t_lo) & (t <= t_hi) & (env > 0.0)
    p = np.polyfit(t[m], np.log(env[m]), 1)
    return float(p[0])


def main():
    print("[las] === four-level gain in a Fabry-Perot cavity vs rate-equation oracle ===",
          flush=True)
    ok = True
    t_of = lambda n: np.arange(n) * DT

    # oracle pins (cold)
    tau_p_pred = cavity_photon_lifetime_s(L_C, N_C, R_MIRROR, R_MIRROR)
    dN_th = threshold_inversion_m3(KAPPA, N_C, DW, L_C, R_MIRROR, R_MIRROR, Gamma=1.0)
    sys0 = FourLevelSystem(tau_32_s=T32, tau_21_s=T21, tau_10_s=T10, N_total_m3=NTOT)
    Wp_th = pump_threshold_per_s(dN_th, sys0)
    alpha_m = np.log(1.0 / R_MIRROR) / L_C
    n_g_hot = N_C + C_LIGHT * alpha_m / DW         # clamped-line slow light (declared bracket)
    print("[las] pins: R = {:.4f}, tau_p = {:.2f} fs, dN_th = {:.3e} m^-3 ({:.3f} NTOT), "
          "W_p_th = {:.3e} 1/s".format(R_MIRROR, tau_p_pred * 1e15, dN_th, dN_th / NTOT,
                                       Wp_th), flush=True)

    # ---- GATE A: cold-cavity ring-down ----
    nA = int(round(2.0e-12 / DT))
    eyA = _run(0.0, nA, seed_amp=1.0e3)
    tA = t_of(nA)
    envA = _envelope(eyA)
    rate = _fit_rate(envA, tA, 0.5e-12, 1.6e-12)   # after the seed burst dies
    tau_p_meas = -1.0 / (2.0 * rate)               # field envelope decays at 1/(2 tau_p)
    dA = abs(tau_p_meas - tau_p_pred) / tau_p_pred
    g_a = bool(rate < 0.0 and dA < 0.05)
    ok = ok and g_a
    print("[las] GATE A: ring-down tau_p = {:.2f} fs vs predicted {:.2f} fs (rel {:.1e}) "
          "-> {}".format(tau_p_meas * 1e15, tau_p_pred * 1e15, dA, "PASS" if g_a else "FAIL"),
          flush=True)

    # ---- GATE B: net envelope rate vs pump straddles threshold ----
    nB = int(round(6.0e-12 / DT))
    tB = t_of(nB)
    v_g = C_LIGHT / N_C
    worstB, signs = 0.0, []
    for fac in (0.5, 0.75, 1.25, 1.5):
        Wp = fac * Wp_th
        s = FourLevelSystem(tau_32_s=T32, tau_21_s=T21, tau_10_s=T10, W_p_per_s=Wp,
                            N_total_m3=NTOT)
        g0 = small_signal_gain_per_m(KAPPA, s.inversion_ss_m3(), N_C, DW)
        gam_pred = 0.5 * (v_g * g0 - 1.0 / tau_p_meas)
        ey = _run(Wp, nB, seed_amp=2.0e2)
        gam_meas = _fit_rate(_envelope(ey), tB, 1.0e-12, 5.0e-12)
        signs.append(gam_meas > 0.0)
        worstB = max(worstB, abs(gam_meas - gam_pred) * tau_p_meas)   # normalized by loss rate
        print("[las]   Wp = {:.2f} W_th: envelope rate meas {:+.3e} / pred {:+.3e} 1/s".format(
            fac, gam_meas, gam_pred), flush=True)
    g_b = bool(signs == [False, False, True, True] and worstB < 0.05)
    ok = ok and g_b
    print("[las] GATE B: decay below / growth above threshold, worst |d(rate)|*tau_p = "
          "{:.1e} -> {}".format(worstB, "PASS" if g_b else "FAIL"), flush=True)

    # ---- GATE C + D: above-threshold clamping + relaxation oscillations ----
    nL = int(round(45.0e-12 / DT))
    tL = t_of(nL)

    def _u_clamp(r):
        """Solve r (2/u)(1 - 1/sqrt(1+u)) = 1 for u (the no-grating standing-wave clamp)."""
        lo, hi = 1e-6, 1e3
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if r * (2.0 / mid) * (1.0 - 1.0 / np.sqrt(1.0 + mid)) > 1.0:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    u_meas, fRO, outs = {}, {}, {}
    for r in (2.0, 3.0):
        dN0 = r * dN_th
        Wp = dN0 / (NTOT * (T21 - T10) - dN0 * (T10 + T21 + T32))
        out = {}
        ey = _run(Wp, nL, seed_amp=2.0e2, out=out, snap_step=nL - 2)
        outs[r] = out
        env = _envelope(ey)
        late = env[(tL >= 38e-12) & (tL <= 44e-12)]
        u_meas[r] = float(np.mean(late) ** 2) / A_SAT2     # u = E_antinode^2 / A_sat^2
        # RO: oscillation of the envelope about steady state in the ring-in window
        m = (tL >= 8e-12) & (tL <= 38e-12)
        fluct = env[m] - np.mean(env[m])
        fr = np.fft.rfftfreq(fluct.size, DT)
        F = np.abs(np.fft.rfft(fluct * np.hanning(fluct.size)))
        F[fr < 5e10] = 0.0                          # drop the DC/drift bin
        fRO[r] = float(fr[np.argmax(F)])
        print("[las]   r = {:g}: u = E_anti^2/A_sat^2 = {:.3f} (no-grating floor {:.3f}, "
              "x{:.2f}), f_RO = {:.1f} GHz".format(r, u_meas[r], _u_clamp(r),
                                                   u_meas[r] / _u_clamp(r), fRO[r] / 1e9),
              flush=True)

    ratio = u_meas[3.0] / u_meas[2.0]
    ratio_pred = _u_clamp(3.0) / _u_clamp(2.0)
    in_band = all(0.95 <= u_meas[r] / _u_clamp(r) <= 2.0 for r in (2.0, 3.0))
    g_c = bool(abs(ratio / ratio_pred - 1.0) < 0.15 and in_band)
    ok = ok and g_c
    print("[las] GATE C: u(r=3)/u(r=2) = {:.3f} vs standing-wave clamp ratio {:.3f} "
          "(15% gate); absolute u in the [0.95, 2.0] x floor band -> {}".format(
              ratio, ratio_pred, "PASS" if g_c else "FAIL"), flush=True)

    g_d = True
    for r in (2.0, 3.0):
        w_cold, _ = relaxation_oscillation_rad_s(r, tau_p_meas, T21)
        w_hot, _ = relaxation_oscillation_rad_s(r, tau_p_meas * n_g_hot / N_C, T21)
        lo, hi = 0.8 * min(w_hot, w_cold), 1.2 * max(w_hot, w_cold)
        w_meas = 2.0 * np.pi * fRO[r]
        inside = bool(lo <= w_meas <= hi)
        g_d = g_d and inside
        print("[las]   r = {:g}: omega_RO meas {:.3e} in [{:.3e}, {:.3e}] rad/s -> {}".format(
            r, w_meas, lo, hi, "ok" if inside else "OUT"), flush=True)
    ok = ok and g_d
    print("[las] GATE D: relaxation-oscillation frequency inside the [hot, cold] tau_p "
          "bracket +-20% -> {}".format("PASS" if g_d else "FAIL"), flush=True)

    Nf = outs[2.0]["Npop_final"]
    tot = Nf.sum(axis=0)[:, N_PAD:N_PAD + N_STR]
    cons = float(np.max(np.abs(tot - NTOT)) / NTOT)
    g_e = bool(cons < 1e-9)
    ok = ok and g_e
    print("[las] GATE E: sum(N) conserved to {:.1e} after the 45 ps run -> {}".format(
        cons, "PASS" if g_e else "FAIL"), flush=True)

    print("[las] *** LASING vs RATE-EQUATION ORACLE: {} ***".format(
        "PASS" if ok else "FAIL"), flush=True)
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
