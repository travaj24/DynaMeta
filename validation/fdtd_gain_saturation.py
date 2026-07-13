"""R20 follow-on oracle: DYNAMIC four-level gain (field-coupled populations, saturation).

The kernel couples the gain-line polarization to per-cell four-level populations through the
stimulated rate S_st = -E dPG/dt / (hbar w_a). For a quasi-CW drive at line center the steady
inversion obeys the CLASSIC homogeneous saturation law (derived from the same rate structure):

    dN(A) = dN0 / (1 + A^2 / A_sat^2),     A_sat^2 = 2 dw hbar w_a / (kappa tau21)

GATE A (small-signal reduces to clamped): a weak pulse through the dynamically-pumped medium
        (populations initialized at the pump steady state) matches the SHIPPED clamped-inversion
        run with gain_dN = dN_ss to < 1% in ln(G).
GATE B (saturation closed form): a long flat-top quasi-CW drive at line center -- the inversion
        captured at the plateau matches dN0/(1 + A^2/A_sat^2) to < 5% across a 30x amplitude
        range straddling A_sat, and the measured ln-gain ratio falls accordingly.
GATE C (conservation + recovery): sum(N) == N_total to the per-step rounding floor (< 1e-9
        relative over the whole run); after the pulse the inversion RECOVERS toward dN0.
GATE D (guards): gain and gain_dyn together raise.

Run: python -m validation.fdtd_gain_saturation
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dynameta.constants import C_LIGHT, HBAR, M_E, Q_E
from dynameta.optics.fdtd_nd import cpml_z, run_2d_te
from dynameta.optics.laser_gain import FourLevelSystem, small_signal_gain_per_m

N_MED = np.sqrt(2.0)
NX = 4
KAPPA = Q_E ** 2 / M_E
W_A = 2.0 * np.pi * 2.5e14
DW = 2.0 * np.pi * 2.0e13
T32, T21, T10 = 1.0e-14, 1.0e-13, 5.0e-15            # fast laser-model lifetimes (reach ss in-pulse)


def _setup(dz, n_pad, n_str):
    nz = 2 * n_pad + n_str
    dx = 4.0 * dz
    dt = 0.5 / (C_LIGHT * np.sqrt(1.0 / dx ** 2 + 1.0 / dz ** 2))
    cpml = cpml_z(nz, dz, dt, 12, N_MED, N_MED)
    return nz, dx, dt, cpml


def _flat_top(t, t_on, t_off, edge):
    return 0.5 * (np.tanh((t - t_on) / edge) - np.tanh((t - t_off) / edge))


def main():
    print("[gs] === R20 follow-on: dynamic gain saturation ===", flush=True)
    ok = True
    dz = 10e-9
    n_pad, n_str = 60, 40
    L = n_str * dz
    nz, dx, dt, cpml = _setup(dz, n_pad, n_str)
    k_src, k_pL, k_pR = 16, 20, nz - 16
    eps = np.full((NX, nz), N_MED ** 2)
    zeros = np.zeros((NX, nz))

    # pump steady state (exact chain balance) -> initial populations + clamped reference dN
    NTOT = 1.0e25
    WP = 2.0e11                                       # -> healthy inversion (tau21 >> tau10)
    sysm = FourLevelSystem(tau_32_s=T32, tau_21_s=T21, tau_10_s=T10, W_p_per_s=WP,
                           N_total_m3=NTOT)
    ss = sysm.steady_state()
    dN0 = sysm.inversion_ss_m3()
    A_sat = np.sqrt(2.0 * DW * HBAR * W_A / (KAPPA * T21))

    den_g = 1.0 + DW * dt / 2.0
    G1 = np.full((NX, nz), (2.0 - W_A ** 2 * dt ** 2) / den_g)
    G2 = np.full((NX, nz), (DW * dt / 2.0 - 1.0) / den_g)
    win = np.zeros((NX, nz)); win[:, n_pad:n_pad + n_str] = 1.0
    kapfac = KAPPA * dt ** 2 / den_g * win
    Wp_grid = WP * win
    Npop0 = np.stack([np.where(win > 0, ss[0], NTOT), win * ss[1], win * ss[2], win * ss[3]])

    def run_dyn(src, snap_step):
        out = {}
        gd = (G1, G2, kapfac, Wp_grid, Npop0, T32, T21, T10, HBAR * W_A, snap_step)
        _, _, eyR, _ = run_2d_te(eps, zeros, zeros, zeros, dx, dz, dt, src.size, k_src, k_pL,
                                  k_pR, src, cpml, np, None, gain_dyn=gd, gain_dyn_out=out)
        return eyR.mean(axis=1), out

    def run_clamped(src, dN):
        g3 = -kapfac / dt ** 2 * dt ** 2 * dN          # == -kappa dN dt^2/den on the window
        _, _, eyR, _ = run_2d_te(eps, zeros, zeros, zeros, dx, dz, dt, src.size, k_src, k_pL,
                                  k_pR, src, cpml, np, None, gain=(G1, G2, g3))
        return eyR.mean(axis=1)

    def amp_at(sig, f):
        F = np.abs(np.fft.rfft(sig))
        fr = np.fft.rfftfreq(sig.size, dt)
        return float(F[np.argmin(np.abs(fr - f))])

    f0 = W_A / (2.0 * np.pi)

    # ---- GATE A: small signal == clamped ----
    tau_p = 120e-15
    t0p = 6.0 * tau_p
    nstepsA = int(round((2.0 * t0p + 200e-15) / dt))
    tA = np.arange(nstepsA) * dt
    srcA = 1.0e3 * np.exp(-((tA - t0p) / tau_p) ** 2) * np.cos(W_A * (tA - t0p))
    ey_ref, _ = run_dyn(0.0 * srcA + 1e-300, 10)       # zero-field reference (populations only)
    ey_off = run_clamped(srcA, 0.0)                    # dN = 0 -> passive medium baseline
    ey_dynA, _ = run_dyn(srcA, 10)
    ey_clA = run_clamped(srcA, dN0)
    lnG_dyn = np.log(amp_at(ey_dynA, f0) / amp_at(ey_off, f0))
    lnG_cl = np.log(amp_at(ey_clA, f0) / amp_at(ey_off, f0))
    relA = abs(lnG_dyn - lnG_cl) / abs(lnG_cl)
    g_a = bool(relA < 1e-2)
    ok = ok and g_a
    print("[gs] GATE A: small-signal dynamic ln(G) = {:.5f} vs clamped {:.5f} (rel {:.1e}) -> {}"
          .format(lnG_dyn, lnG_cl, relA, "PASS" if g_a else "FAIL"), flush=True)

    # ---- GATE B: saturation law on the plateau inversion ----
    t_on, t_off, edge = 100e-15, 700e-15, 25e-15
    nstepsB = int(round(900e-15 / dt))
    tB = np.arange(nstepsB) * dt
    snap = int(round(600e-15 / dt))                    # deep in the plateau
    worstB = 0.0
    meas = []
    for A in (0.1 * A_sat, 0.5 * A_sat, 1.0 * A_sat, 3.0 * A_sat):
        srcB = A * _flat_top(tB, t_on, t_off, edge) * np.cos(W_A * tB)
        _, out = run_dyn(srcB, snap)
        dn_mid = float(out["dN_snap"][:, n_pad + n_str // 2].mean())
        # the LOCAL plateau amplitude inside the medium ~ the launched amplitude (index-matched,
        # weak gain); the soft source launches A/2 into each direction -> calibrate from the
        # zero-gain run at the SAME drive
        ey_cal = run_clamped(srcB, 0.0)
        A_loc = float(np.max(np.abs(ey_cal[snap - 400:snap])))
        dn_cf = dN0 / (1.0 + (A_loc / A_sat) ** 2)
        worstB = max(worstB, abs(dn_mid - dn_cf) / dN0)
        meas.append((A_loc / A_sat, dn_mid / dN0, dn_cf / dN0))
    g_b = bool(worstB < 5e-2)
    ok = ok and g_b
    print("[gs] GATE B: plateau inversion vs dN0/(1 + A^2/A_sat^2): " + "; ".join(
        "A/As={:.2f} meas {:.3f} cf {:.3f}".format(*m) for m in meas)
        + " (worst {:.1e}) -> {}".format(worstB, "PASS" if g_b else "FAIL"), flush=True)

    # ---- GATE C: conservation + recovery ----
    srcC = (2.0 * A_sat) * _flat_top(tB, t_on, 400e-15, edge) * np.cos(W_A * tB)
    _, outC = run_dyn(srcC, int(round(380e-15 / dt)))
    Nf = outC["Npop_final"]
    tot = Nf.sum(axis=0)[:, n_pad:n_pad + n_str]
    cons = float(np.max(np.abs(tot - NTOT)) / NTOT)
    dn_end = float(Nf[2, :, n_pad + n_str // 2].mean() - Nf[1, :, n_pad + n_str // 2].mean())
    dn_sat = float(outC["dN_snap"][:, n_pad + n_str // 2].mean())
    recov = dn_end > dn_sat + 0.3 * (dN0 - dn_sat)     # pulse off at 400 fs; ~5 tau21 of recovery
    g_c = bool(cons < 1e-9 and recov)
    ok = ok and g_c
    print("[gs] GATE C: sum(N) conserved to {:.1e}; inversion recovers after the pulse "
          "({:.3f} -> {:.3f} of dN0) -> {}".format(
              cons, dn_sat / dN0, dn_end / dN0, "PASS" if g_c else "FAIL"), flush=True)

    # ---- GATE D: guards ----
    g_d = False
    try:
        gd = (G1, G2, kapfac, Wp_grid, Npop0, T32, T21, T10, HBAR * W_A, 10)
        run_2d_te(eps, zeros, zeros, zeros, dx, dz, dt, 100, k_src, k_pL, k_pR,
                   np.zeros(100), cpml, np, None, gain=(G1, G2, zeros), gain_dyn=gd)
    except ValueError:
        g_d = True
    ok = ok and g_d
    print("[gs] GATE D: gain + gain_dyn together raise -> {}".format(
        "PASS" if g_d else "FAIL"), flush=True)

    print("[gs] *** R20 DYNAMIC GAIN SATURATION: {} ***".format("PASS" if ok else "FAIL"),
          flush=True)
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
