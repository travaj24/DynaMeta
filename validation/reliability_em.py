"""REL3 electromigration oracle (Black + Blech + Miner).

GATE A (reduces-to-closed-form): Arrhenius and J-power-law MTTF ratios match the closed forms to
        machine; J = 0 -> MTTF = inf; the Blech boundary J*L < (J*L)_crit flips immortal/mortal.
GATE B (independent numeric): Miner damage under a two-segment J(t) duty cycle (trapezoid
        accumulation) matches the piecewise-ANALYTIC time-to-failure.
GATE C (literature band + anchor): n = 2 (void growth), Ea in the Cu band; calibration reproduces its
        anchor; a short-contact design is demonstrably Blech-immortal.

Run: python -m validation.reliability_em
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dynameta.reliability.em import (EmParams, black_mttf_s, blech_immortal, current_density_A_m2,
                                     miner_time_to_failure_s, KB_EV_K)


def main():
    print("[re] === REL3 electromigration (Black + Blech) ===", flush=True)
    ok = True
    A, n, Ea = 3.6e6, 2.0, 0.55

    # ---- GATE A ----
    rJ = float(black_mttf_s(2e9, 350.0, A_s=A, n_exp=n, Ea_eV=Ea)
               / black_mttf_s(1e9, 350.0, A_s=A, n_exp=n, Ea_eV=Ea))
    rT = float(black_mttf_s(1e9, 390.0, A_s=A, n_exp=n, Ea_eV=Ea)
               / black_mttf_s(1e9, 330.0, A_s=A, n_exp=n, Ea_eV=Ea))
    rT_an = float(np.exp((Ea / KB_EV_K) * (1.0 / 390.0 - 1.0 / 330.0)))
    inf_ok = np.isinf(black_mttf_s(0.0, 350.0, A_s=A, n_exp=n, Ea_eV=Ea))
    J = current_density_A_m2(1.0e-3, 1e-6, 100e-9)          # 1 mA in a 1 um x 100 nm trace = 1e10
    blech = (blech_immortal(J, 5e-6) is False               # 1e10 * 5e-6 = 5e4 < 2e5? -> immortal!
             or True)
    im_short = blech_immortal(J, 5e-6)                      # J*L = 5e4 A/m < 2e5 -> immortal
    im_long = blech_immortal(J, 50e-6)                      # J*L = 5e5 A/m > 2e5 -> mortal
    g_a = (abs(rJ - 2.0 ** (-n)) < 1e-12 and abs(rT / rT_an - 1) < 1e-12 and inf_ok
           and im_short and not im_long)
    ok = ok and bool(g_a)
    print("[re] GATE A: J-ratio {:.4f} == 2^-n; T-ratio {:.4f} == Arrhenius; J=0 -> inf; Blech "
          "5um immortal / 50um mortal at J={:.1e} A/m^2 -> {}".format(
              rJ, rT, J, "PASS" if g_a else "FAIL"), flush=True)

    # ---- GATE B: Miner two-segment J(t) vs piecewise analytic ----
    p = EmParams.calibrated(J_A_m2=2e9, T_K=350.0, mttf_s=400.0)
    m1 = float(p.mttf_s(2e9, 350.0))                        # 400 by anchor
    m2 = float(p.mttf_s(3e9, 350.0))                        # 400 * (2/3)^2
    t_sw = 150.0
    t_fail_an = t_sw + (1.0 - t_sw / m1) * m2
    assert m1 > t_sw
    t_grid = np.linspace(0.0, 3.0 * t_fail_an, 200001)
    t_fail_num = miner_time_to_failure_s(t_grid, lambda t: 2e9 if t < t_sw else 3e9,
                                         lambda t: 350.0, p)
    relB = abs(t_fail_num - t_fail_an) / t_fail_an
    g_b = bool(relB < 1e-3)                                 # trapezoid on the discontinuity
    ok = ok and g_b
    print("[re] GATE B: Miner two-segment J(t) numeric {:.3f} s vs analytic {:.3f} s (rel {:.1e}) "
          "-> {}".format(t_fail_num, t_fail_an, relB, "PASS" if g_b else "FAIL"), flush=True)

    # ---- GATE C: bands + anchor ----
    g_c = bool(n == 2.0 and 0.48 <= Ea <= 0.9
               and abs(float(p.mttf_s(2e9, 350.0)) - 400.0) / 400.0 < 1e-12
               and np.isinf(p.mttf_s(J, 350.0, length_m=5e-6)))
    ok = ok and g_c
    print("[re] GATE C: n=2 (void growth), Ea {:.2f} eV in Cu band; anchor reproduced; short-contact "
          "design immortal -> {}".format(Ea, "PASS" if g_c else "FAIL"), flush=True)

    print("[re] *** REL3 ELECTROMIGRATION: {} ***".format("PASS" if ok else "FAIL"), flush=True)
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
