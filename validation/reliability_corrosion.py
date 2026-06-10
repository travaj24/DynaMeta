"""REL9 corrosion / oxidation / humidity-life oracle (Deal-Grove + Peck).

GATE A (closed form vs INDEPENDENT ODE): the Deal-Grove thickness x(t) = (A/2)(sqrt(1 + 4B(t+tau)/
        A^2) - 1) must match a scipy integration of the rate ODE dx/dt = B/(2x + A); x(0) = x0
        exact; the thin-film LINEAR limit x ~ (B/A) t and the thick-film PARABOLIC limit
        x ~ sqrt(B t) both recovered.
GATE B (Peck): the acceleration factor matches its closed form to machine; humidity monotonicity;
        the canonical 85C/85%RH vs office-ambient acceleration is large (>> 10x, the reason the
        85/85 chamber exists).

Run: python -m validation.reliability_corrosion
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dynameta.reliability.corrosion import (deal_grove_thickness_m, deal_grove_rate_arrhenius,
                                            peck_time_to_failure_s, peck_af, KB_EV_K)


def main():
    print("[rc] === REL9 corrosion / oxidation / humidity life ===", flush=True)
    ok = True

    # ---- GATE A: Deal-Grove closed form vs the rate ODE + limits ----
    from scipy.integrate import solve_ivp
    A, B, x0 = 50e-9, 1.0e-19, 2e-9
    t = np.linspace(0.0, 5.0e4, 21)
    x_cf = deal_grove_thickness_m(t, A_m=A, B_m2_s=B, x0_m=x0)
    sol = solve_ivp(lambda tt, y: [B / (2.0 * y[0] + A)], (0.0, float(t[-1])), [x0], t_eval=t,
                    rtol=1e-12, atol=1e-18)
    rel_ode = float(np.max(np.abs(sol.y[0] - x_cf) / x_cf[-1]))
    x_zero = float(deal_grove_thickness_m(0.0, A_m=A, B_m2_s=B, x0_m=x0))
    # linear limit: tiny t, no initial oxide -> x ~ (B/A) t
    t_lin = 1.0
    x_lin = float(deal_grove_thickness_m(t_lin, A_m=A, B_m2_s=B))
    rel_lin = abs(x_lin - (B / A) * t_lin) / ((B / A) * t_lin)
    # parabolic limit: t >> A^2/B -> x ~ sqrt(B t)
    t_par = 1.0e9 * A ** 2 / B
    x_par = float(deal_grove_thickness_m(t_par, A_m=A, B_m2_s=B))
    rel_par = abs(x_par - np.sqrt(B * t_par)) / np.sqrt(B * t_par)
    g_a = bool(rel_ode < 1e-8 and abs(x_zero - x0) < 1e-18 and rel_lin < 1e-3 and rel_par < 1e-3)
    ok = ok and g_a
    print("[rc] GATE A: closed form vs rate ODE rel {:.1e}; x(0)=x0 exact; linear limit rel {:.1e}; "
          "parabolic limit rel {:.1e} -> {}".format(rel_ode, rel_lin, rel_par,
                                                    "PASS" if g_a else "FAIL"), flush=True)

    # ---- GATE B: Peck humidity life ----
    af = peck_af(RH_use=40.0, RH_stress=85.0, T_use_K=298.15, T_stress_K=358.15)
    af_an = (85.0 / 40.0) ** 2.7 * np.exp((0.8 / KB_EV_K) * (1.0 / 298.15 - 1.0 / 358.15))
    mono = (float(peck_time_to_failure_s(85.0, 358.15, A_s=1e12))
            < float(peck_time_to_failure_s(40.0, 358.15, A_s=1e12)))
    g_b = bool(abs(af / af_an - 1) < 1e-12 and af > 10.0 and mono)
    ok = ok and g_b
    print("[rc] GATE B: Peck AF(85C/85%RH vs 25C/40%RH) = {:.0f}x == closed form (>> 10x, the 85/85 "
          "chamber rationale); RH-monotone -> {}".format(af, "PASS" if g_b else "FAIL"), flush=True)

    # Arrhenius-rate helper sanity (B and B/A both Arrhenius)
    B1, BA1 = deal_grove_rate_arrhenius(1100.0, B0_m2_s=1e-10, Ea_B_eV=1.23, BA0_m_s=1e3,
                                        Ea_BA_eV=2.0)
    print("[rc] INFO: Deal-Grove at 1100 K: B = {:.2e} m^2/s, B/A = {:.2e} m/s".format(B1, BA1),
          flush=True)

    print("[rc] *** REL9 CORROSION / HUMIDITY: {} ***".format("PASS" if ok else "FAIL"), flush=True)
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
