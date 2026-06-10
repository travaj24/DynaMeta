"""REL1 gate-oxide TDDB oracle. The separable E-model tBD = tau0 exp(-gamma_E E) exp(Ea/kBT).

GATE A (reduces-to-closed-form): the field-acceleration ratio tBD(E2)/tBD(E1) = exp(-gamma_E (E2-E1))
        and the Arrhenius ratio tBD(T2)/tBD(T1) = exp((Ea/kB)(1/T2 - 1/T1)) match the closed forms to
        machine; Weibull area scaling t63(A2) = t63(A1) (A1/A2)^(1/beta) exact; the 1/E model's
        G/E acceleration likewise.
GATE B (independent numeric reference): TIME-VARYING stress -- with Miner damage accumulation
        (failure when integral dt/tBD(E, T(t)) = 1) under a two-segment T profile, the scipy ODE
        integration of the damage must match the piecewise-ANALYTIC time-to-failure (a composite the
        closed form alone cannot produce -- numeric-vs-analytic, not algebra inversion).
GATE C (literature anchor band): the default slopes sit in the published windows (gamma_E/ln10 in
        1-2 decades/(MV/cm); Ea in 0.6-0.9 eV); TddbParams.calibrated reproduces its anchor point
        exactly and extrapolates with the documented slope.

Run: python -m validation.reliability_tddb
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dynameta.reliability.tddb import (TddbParams, tbd_e_model, tbd_one_over_e, weibull_area_scale,
                                       KB_EV_K)

MV = 1.0e8                                                   # 1 MV/cm in V/m


def main():
    print("[rt] === REL1 gate-oxide TDDB (separable E-model) ===", flush=True)
    ok = True
    tau0, gam, Ea = 2.0e-7, 3.0, 0.7

    # ---- GATE A: closed-form acceleration ratios ----
    r_field = float(tbd_e_model(8 * MV, 300.0, tau0_s=tau0, gamma_E_per_MV_cm=gam, Ea_eV=Ea)
                    / tbd_e_model(5 * MV, 300.0, tau0_s=tau0, gamma_E_per_MV_cm=gam, Ea_eV=Ea))
    r_field_an = float(np.exp(-gam * 3.0))
    r_T = float(tbd_e_model(6 * MV, 398.0, tau0_s=tau0, gamma_E_per_MV_cm=gam, Ea_eV=Ea)
                / tbd_e_model(6 * MV, 358.0, tau0_s=tau0, gamma_E_per_MV_cm=gam, Ea_eV=Ea))
    r_T_an = float(np.exp((Ea / KB_EV_K) * (1.0 / 398.0 - 1.0 / 358.0)))
    t63 = float(weibull_area_scale(1.0e6, 1e-12, 64e-12, beta=1.5))
    t63_an = 1.0e6 * (1.0 / 64.0) ** (1.0 / 1.5)
    r_inv = float(tbd_one_over_e(5 * MV, 300.0, tau0_s=1e-12, G_MV_cm=350.0)
                  / tbd_one_over_e(7 * MV, 300.0, tau0_s=1e-12, G_MV_cm=350.0))
    r_inv_an = float(np.exp(350.0 / 5.0 - 350.0 / 7.0))
    g_a = (abs(r_field / r_field_an - 1) < 1e-12 and abs(r_T / r_T_an - 1) < 1e-12
           and abs(t63 / t63_an - 1) < 1e-12 and abs(r_inv / r_inv_an - 1) < 1e-12)
    ok = ok and bool(g_a)
    print("[rt] GATE A: field ratio {:.6e} (=e^-3g), Arrhenius ratio {:.4f}, Weibull t63 {:.4g}, "
          "1/E ratio {:.4g} -- all == closed form -> {}".format(
              r_field, r_T, t63, r_inv, "PASS" if g_a else "FAIL"), flush=True)

    # ---- GATE B: Miner damage under a two-segment T(t) vs piecewise analytic ----
    from scipy.integrate import solve_ivp
    E = 6 * MV
    T1, T2, t_switch = 330.0, 390.0, 200.0                   # cool then hot
    # calibrate so the failure genuinely STRADDLES the switch (tb1 > t_switch > damage-to-go)
    p = TddbParams.calibrated(E_ox_V_m=E, T_K=T1, tbd_s=500.0, gamma_E_per_MV_cm=gam, Ea_eV=Ea)
    tb1, tb2 = float(p.tbd_s(E, T1)), float(p.tbd_s(E, T2))
    assert tb1 > t_switch, "gate setup: failure must straddle the T switch"
    # analytic: damage D(t) = t/tb1 (t<ts); fail when ts/tb1 + (t-ts)/tb2 = 1
    t_fail_an = t_switch + (1.0 - t_switch / tb1) * tb2
    T_of_t = lambda t: T1 if t < t_switch else T2
    rhs = lambda t, D: [1.0 / float(p.tbd_s(E, T_of_t(t)))]
    hit = lambda t, D: D[0] - 1.0
    hit.terminal, hit.direction = True, 1.0
    sol = solve_ivp(rhs, (0.0, 5.0 * t_fail_an), [0.0], events=hit, rtol=1e-10, atol=1e-12,
                    max_step=t_switch / 8.0)
    t_fail_num = float(sol.t_events[0][0])
    relB = abs(t_fail_num - t_fail_an) / t_fail_an
    g_b = bool(relB < 1e-6)
    ok = ok and g_b
    print("[rt] GATE B: Miner two-segment T(t) time-to-failure numeric {:.6f} s vs analytic {:.6f} s "
          "(rel {:.1e}) -> {}".format(t_fail_num, t_fail_an, relB, "PASS" if g_b else "FAIL"),
          flush=True)

    # ---- GATE C: literature slope band + calibration anchor ----
    decades = gam / np.log(10.0)
    cal = TddbParams.calibrated(E_ox_V_m=7 * MV, T_K=398.0, tbd_s=500.0,
                                gamma_E_per_MV_cm=gam, Ea_eV=Ea)
    anchor = float(cal.tbd_s(7 * MV, 398.0))
    use = float(cal.tbd_s(5 * MV, 358.0))
    g_c = bool(1.0 <= decades <= 2.0 and 0.6 <= Ea <= 0.9 and abs(anchor - 500.0) / 500.0 < 1e-12
               and use > anchor)
    ok = ok and g_c
    print("[rt] GATE C: slope {:.2f} decades/(MV/cm) in [1,2]; Ea {:.2f} eV in [0.6,0.9]; anchor "
          "reproduced ({:.1f} s); use-condition (5 MV/cm, 358K) extrapolates to {:.3e} s ({:.0f} h) "
          "-> {}".format(decades, Ea, anchor, use, use / 3600.0, "PASS" if g_c else "FAIL"),
          flush=True)

    print("[rt] *** REL1 TDDB: {} ***".format("PASS" if ok else "FAIL"), flush=True)
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
