"""R11 macrospin LLG oracle.

GATE A (pure precession, the alpha = 0 reduces-to limit): no anisotropy/demag, H = H z-hat, m0 = x-hat
        -> m precesses in the xy-plane at EXACTLY omega = gamma0 mu0 H; phase vs cos/sin to < 1e-4
        over 3 periods; |m| = 1 to machine; m_z stays 0; energy constant.
GATE B (EXACT nonlinear ring-down): with H along z only, the polar angle obeys
        d(theta)/dt = -lambda sin(theta), whose exact solution is tan(theta/2)(t) =
        tan(theta0/2) exp(-lambda t), lambda = alpha gamma0 mu0 H / (1 + alpha^2) -- the fitted
        log-slope must match the analytic lambda (NOT a linearized small-angle approximation).
GATE C (Lyapunov): with anisotropy + demag + a tilted constant field and alpha > 0, the energy
        density U(m(t)) is monotonically non-increasing.
GATE D (Stoner-Wohlfarth astroid): the 45-degree switching field is EXACTLY H_K/2 -- a field at
        0.45 H_K (45 deg from the easy axis, opposing m) does NOT switch; 0.55 H_K DOES.

Run: python -m validation.llg_macrospin
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dynameta.constants import MU0
from dynameta.carriers.llg import LLGMacrospin, GAMMA_ELECTRON_RAD_ST

MS = 1.0e5                                                  # A/m


def main():
    print("[lg] === R11 Landau-Lifshitz-Gilbert macrospin ===", flush=True)
    ok = True
    H0 = 1.0e4                                              # A/m -> omega = gamma0 mu0 H ~ 2.21e9 rad/s

    # ---- GATE A: pure precession ----
    llg = LLGMacrospin(Ms_A_m=MS, alpha=0.0, H_applied_A_m=lambda t: np.array([0.0, 0.0, H0]))
    w = GAMMA_ELECTRON_RAD_ST * MU0 * H0
    T = 2.0 * np.pi / w
    t = np.linspace(0.0, 3.0 * T, 1501)
    r = llg.simulate(t, m0=[1.0, 0.0, 0.0])
    norm_err = float(np.max(np.abs(np.linalg.norm(r.m_t, axis=1) - 1.0)))
    # precession sense: dm/dt = -gamma' m x H; m=x, H=z -> m x H = -y -> dm/dt = +y
    mx_an, my_an = np.cos(w * t), np.sin(w * t)
    phase_err = float(max(np.max(np.abs(r.m_t[:, 0] - mx_an)), np.max(np.abs(r.m_t[:, 1] - my_an))))
    mz_err = float(np.max(np.abs(r.m_t[:, 2])))
    en_var = float(np.max(np.abs(r.energy_J_m3 - r.energy_J_m3[0])))
    g_a = bool(norm_err < 1e-12 and phase_err < 1e-4 and mz_err < 1e-9 and en_var < 1e-9
               and abs(r.precession_rad_s - w) < 1e-3 * w)
    ok = ok and g_a
    print("[lg] GATE A: omega = {:.4e} rad/s; |m|-1 max {:.1e}; phase err {:.1e} over 3 periods; "
          "m_z {:.1e}; dE {:.1e} -> {}".format(w, norm_err, phase_err, mz_err, en_var,
                                               "PASS" if g_a else "FAIL"), flush=True)

    # ---- GATE B: exact tan(theta/2) ring-down ----
    alpha = 0.05
    llg_d = LLGMacrospin(Ms_A_m=MS, alpha=alpha, H_applied_A_m=lambda t: np.array([0.0, 0.0, H0]))
    lam_an = alpha * GAMMA_ELECTRON_RAD_ST * MU0 * H0 / (1.0 + alpha ** 2)
    th0 = np.deg2rad(30.0)
    t2 = np.linspace(0.0, 3.0 / lam_an, 4001)
    r2 = llg_d.simulate(t2, m0=[np.sin(th0), 0.0, np.cos(th0)])
    theta = np.arccos(np.clip(r2.m_t[:, 2], -1.0, 1.0))
    y = np.log(np.tan(theta / 2.0))
    lam_fit = -float(np.polyfit(t2, y, 1)[0])
    relB = abs(lam_fit - lam_an) / lam_an
    g_b = bool(relB < 1e-3)
    ok = ok and g_b
    print("[lg] GATE B: exact tan(theta/2) decay -- fitted lambda {:.6e} vs analytic "
          "alpha gamma' H/(1+alpha^2) = {:.6e} (rel {:.1e}) -> {}".format(
              lam_fit, lam_an, relB, "PASS" if g_b else "FAIL"), flush=True)

    # ---- GATE C: Lyapunov with anisotropy + demag + tilted field ----
    llg_c = LLGMacrospin(Ms_A_m=MS, alpha=0.2, K_u_J_m3=1.0e3, N_demag=np.array([0.0, 0.0, 1.0]),
                         H_applied_A_m=lambda t: np.array([6e3, 0.0, 6e3]))
    t3 = np.linspace(0.0, 30e-9, 4001)
    r3 = llg_c.simulate(t3, m0=[0.3, 0.2, 0.93])
    dE = np.diff(r3.energy_J_m3)
    tol = 1e-9 * max(abs(float(r3.energy_J_m3[0] - r3.energy_J_m3[-1])), 1e-30)
    g_c = bool(np.all(dE <= tol) and r3.energy_J_m3[-1] < r3.energy_J_m3[0])
    ok = ok and g_c
    print("[lg] GATE C: Lyapunov U(m(t)) monotone non-increasing (max dU = {:+.2e} J/m^3, total "
          "drop {:.3e}) -> {}".format(float(np.max(dE)), float(r3.energy_J_m3[0] - r3.energy_J_m3[-1]),
                                      "PASS" if g_c else "FAIL"), flush=True)

    # ---- GATE D: Stoner-Wohlfarth 45-degree astroid (h* = 1/2 exactly) ----
    K_u = 1.0e3
    H_K = 2.0 * K_u / (MU0 * MS)                            # ~ 1.59e4 A/m
    res = {}
    for frac in (0.45, 0.55):
        Hvec = frac * H_K * np.array([np.sin(np.deg2rad(45.0)), 0.0, -np.cos(np.deg2rad(45.0))])
        llg_sw = LLGMacrospin(Ms_A_m=MS, alpha=1.0, K_u_J_m3=K_u,
                              H_applied_A_m=lambda t, _H=Hvec: _H)
        t4 = np.linspace(0.0, 30e-9, 3001)
        r4 = llg_sw.simulate(t4, m0=[1e-3, 0.0, 1.0])       # tiny tilt breaks the unstable symmetry
        res[frac] = float(r4.m_t[-1, 2])
    g_d = bool(res[0.45] > 0.4 and res[0.55] < 0.0)
    ok = ok and g_d
    print("[lg] GATE D: Stoner-Wohlfarth 45 deg -- m_z(final) at 0.45 H_K = {:+.3f} (no switch), at "
          "0.55 H_K = {:+.3f} (switched); threshold H_K/2 -> {}".format(
              res[0.45], res[0.55], "PASS" if g_d else "FAIL"), flush=True)

    print("[lg] *** R11 LLG MACROSPIN: {} ***".format("PASS" if ok else "FAIL"), flush=True)
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
