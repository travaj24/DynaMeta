"""
Reconfigurable switching drivers oracle (roadmap Phase-4 REMAINING): the PCM crystallization driver
(JMAK/Arrhenius + melt-quench) and the LC director relaxation driver (carriers.switching). Pure numpy.

GATE A (PCM): (i) at constant T the integrator reproduces the closed-form Avrami curve
        x = 1 - exp(-(K(T) t)^n) (machine precision -- the isokinetic-additivity integrate() reduces
        to it); (ii) a sustained in-window pulse drives x monotonically 0 -> ~1; (iii) a melt spike
        (T >= T_melt) RESETS x to 0 (melt-quench), then crystallization restarts; (iv) a sub-glass
        segment leaves x frozen.
GATE B (LC): the relaxation time tau = gamma d^2 / (K pi^2) is sane (ms-scale), theta(t) decays
        EXACTLY exponentially (theta(tau) = theta0/e), and is monotonically decreasing.

Run: python -m validation.switching_drivers
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dynameta.carriers.switching import PCMSwitching, LCRelaxation

EV = 1.602176634e-19


def main():
    print("[sw] === Reconfigurable switching drivers (PCM JMAK + LC relaxation) ===", flush=True)
    pcm = PCMSwitching(K0_per_s=1.0e22, E_a_J=2.0 * EV, T_glass_K=425.0, T_melt_K=900.0, avrami_n=3.0)

    # (i) constant-T integrate == closed-form Avrami
    t = np.linspace(0.0, 2.0e-7, 400)
    x_int = pcm.integrate(t, np.full_like(t, 700.0))
    x_cf = pcm.fraction_isothermal(t, 700.0)
    avrami_ok = bool(np.max(np.abs(x_int - x_cf)) < 1e-9)
    # (ii) monotonic 0 -> ~1
    mono_ok = bool(np.all(np.diff(x_int) >= -1e-12) and x_int[0] < 1e-6 and x_int[-1] > 0.999)

    # (iii) melt-quench: crystallize, spike above T_melt (reset), then re-crystallize
    tp = np.linspace(0.0, 3.0e-7, 600)
    Tp = np.full_like(tp, 700.0)
    Tp[(tp >= 1.0e-7) & (tp < 1.2e-7)] = 1000.0          # melt spike (> T_melt)
    xp = pcm.integrate(tp, Tp)
    i_pre = np.searchsorted(tp, 1.0e-7) - 1
    i_melt = np.searchsorted(tp, 1.19e-7)
    quench_ok = bool(xp[i_pre] > 0.2 and xp[i_melt] < 1e-9 and xp[-1] > 0.5)

    # (iv) frozen below the glass onset
    xf = pcm.integrate(np.array([0.0, 1.0, 2.0]), np.array([300.0, 300.0, 300.0]), x0=0.3)
    frozen_ok = bool(np.allclose(xf, 0.3))

    gate_a = avrami_ok and mono_ok and quench_ok and frozen_ok
    print("[sw] PCM: Avrami(int==closed-form) max|d|={:.1e}; mono 0->{:.4f}; melt-quench "
          "pre={:.3f}->melt={:.1e}->end={:.3f}; frozen={}".format(
              float(np.max(np.abs(x_int - x_cf))), x_int[-1], xp[i_pre], xp[i_melt], xp[-1],
              frozen_ok), flush=True)
    print("[sw] GATE A (PCM Avrami + monotonic + melt-quench reset + frozen): {}".format(
        "PASS" if gate_a else "FAIL"), flush=True)

    # LC relaxation
    lc = LCRelaxation(K_elastic_N=1.0e-11, gamma_visc_Pa_s=0.1, d_m=5.0e-6)
    tau = lc.tau_s()
    tau_an = 0.1 * (5.0e-6) ** 2 / (1.0e-11 * np.pi ** 2)
    tl = np.linspace(0.0, 3.0 * tau, 200)
    th = lc.relax(tl, np.radians(30.0))
    tau_ok = bool(abs(tau - tau_an) < 1e-12 * tau_an and 1e-3 < tau < 1e-1)
    e_fold = float(lc.relax(np.array([tau]), np.radians(30.0))[0])
    exp_ok = bool(abs(e_fold - np.radians(30.0) / np.e) < 1e-9)
    mono_lc = bool(np.all(np.diff(th) <= 1e-15) and abs(th[0] - np.radians(30.0)) < 1e-12)
    gate_b = tau_ok and exp_ok and mono_lc
    print("[sw] LC: tau={:.4e} s (analytic {:.4e}); theta(tau)/theta0={:.4f} (1/e={:.4f}); "
          "monotonic={}".format(tau, tau_an, e_fold / np.radians(30.0), 1.0 / np.e, mono_lc),
          flush=True)
    print("[sw] GATE B (LC tau = gamma d^2/(K pi^2), exponential decay): {}".format(
        "PASS" if gate_b else "FAIL"), flush=True)

    overall = gate_a and gate_b
    print("[sw] *** RECONFIGURABLE SWITCHING DRIVERS: {} ***".format("PASS" if overall else "FAIL"),
          flush=True)
    return overall


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
