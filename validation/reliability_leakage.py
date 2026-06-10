"""R16 gate-oxide tunneling leakage oracle (Fowler-Nordheim + direct tunneling).

GATE A (FN reduces-to-closed-form): ln(J/E^2) vs 1/E is EXACTLY linear with slope -B_FN
        (machine); the two-field ratio matches (E2/E1)^2 exp(-B(1/E2 - 1/E1)) analytically.
GATE B (literature anchors, SiO2 phi_b = 3.1 eV, m_ox = 0.42 m0): B_FN in the 230-260 MV/cm
        band; A_FN matches the textbook 1.54e-6/(m_r phi_eV) A/V^2; B_FN scales EXACTLY as
        phi_b^(3/2); J_FN(1e9 V/m) lands in the known 1e-3..1e-1 A/cm^2 FN window.
GATE C (DT <-> FN continuity, EXACT): the trapezoidal-WKB exponent g(u >= 1) = 1 makes
        J_DT(V = phi_b) IDENTICAL to J_FN(E = phi_b/t_ox) (0.0 difference, not a tolerance);
        J_DT is monotone in V; for V < phi_b it EXCEEDS the naive FN extrapolation (the
        thin-oxide leakage excess); V <= 0 gives exactly 0.
GATE D (off-switch + guards): enabled=False -> 0.0 EXACTLY (scalar and array); t_ox = 0
        raises; joule_W_m3 = J*E consistency.

Run: python -m validation.reliability_leakage
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dynameta.reliability.leakage import (OxideLeakageParams, direct_tunneling_current,
                                          fn_coefficients, fowler_nordheim_current)


def main():
    print("[lk] === R16 gate-oxide tunneling leakage ===", flush=True)
    ok = True
    a_fn, b_fn = fn_coefficients(0.42, 3.1)

    # ---- GATE A: FN linearity in ln(J/E^2) vs 1/E ----
    E = np.array([5e8, 7e8, 1e9, 1.3e9, 1.6e9])
    J = fowler_nordheim_current(E)
    y = np.log(J / E ** 2)
    slope, icpt = np.polyfit(1.0 / E, y, 1)
    resid = float(np.max(np.abs(y - (slope / E + icpt))))
    relA = abs(slope + b_fn) / b_fn
    r12 = J[2] / J[0]
    r12_an = (E[2] / E[0]) ** 2 * math.exp(-b_fn * (1.0 / E[2] - 1.0 / E[0]))
    g_a = bool(relA < 1e-12 and resid < 1e-12 and abs(r12 / r12_an - 1.0) < 1e-12)
    ok = ok and g_a
    print("[lk] GATE A: ln(J/E^2) vs 1/E slope = -B_FN (rel {:.1e}, fit resid {:.1e}); "
          "two-field ratio analytic (rel {:.1e}) -> {}".format(
              relA, resid, abs(r12 / r12_an - 1.0), "PASS" if g_a else "FAIL"), flush=True)

    # ---- GATE B: SiO2 literature anchors ----
    band = 2.3e10 <= b_fn <= 2.6e10
    a_txt = 1.54e-6 / (0.42 * 3.1)                       # the textbook A = 1.54e-6/(m_r phi) A/V^2
    a_ok = abs(a_fn - a_txt) / a_txt < 0.02
    _, b28 = fn_coefficients(0.42, 2.8)
    scale_ok = abs(b28 / b_fn - (2.8 / 3.1) ** 1.5) < 1e-12
    J10 = fowler_nordheim_current(1.0e9) * 1e-4          # A/m^2 -> A/cm^2 at 10 MV/cm
    window = 1e-3 < J10 < 1e-1
    g_b = bool(band and a_ok and scale_ok and window)
    ok = ok and g_b
    print("[lk] GATE B: B_FN = {:.3e} V/m ({:.0f} MV/cm, band 230-260); A_FN = {:.3e} vs "
          "textbook {:.3e} A/V^2; phi^1.5 scaling exact; J(10 MV/cm) = {:.2e} A/cm^2 -> {}"
          .format(b_fn, b_fn / 1e8, a_fn, a_txt, J10, "PASS" if g_b else "FAIL"), flush=True)

    # ---- GATE C: DT <-> FN exact continuity + shape ----
    t_ox = 3e-9
    j_join_dt = direct_tunneling_current(3.1, t_ox)
    j_join_fn = fowler_nordheim_current(3.1 / t_ox)
    V = np.linspace(0.3, 3.1, 60)
    j_dt = direct_tunneling_current(V, t_ox)
    mono = bool(np.all(np.diff(j_dt) > 0.0))
    excess = bool(np.all(j_dt[:-1] > fowler_nordheim_current(V[:-1] / t_ox)))
    zero = direct_tunneling_current(0.0, t_ox) == 0.0
    g_c = bool(j_join_dt == j_join_fn and mono and excess and zero)
    ok = ok and g_c
    print("[lk] GATE C: J_DT(phi_b) == J_FN IDENTICALLY (|d| = {:.1e}); monotone in V; "
          "exceeds naive FN below phi_b; V = 0 -> 0 -> {}".format(
              abs(j_join_dt - j_join_fn), "PASS" if g_c else "FAIL"), flush=True)

    # ---- GATE D: off-switch + guards + Joule consistency ----
    off = OxideLeakageParams(t_ox_m=t_ox)                # enabled defaults False
    on = OxideLeakageParams(t_ox_m=t_ox, enabled=True)
    j_on = on.leakage_J_A_m2(2.0)
    q_on = on.joule_W_m3(2.0)
    guards = False
    try:
        direct_tunneling_current(1.0, 0.0)
    except ValueError:
        try:
            OxideLeakageParams(t_ox_m=-1e-9)
        except ValueError:
            guards = True
    g_d = bool(off.leakage_J_A_m2(2.0) == 0.0 and np.all(off.leakage_J_A_m2(V) == 0.0)
               and q_on == j_on * 2.0 / t_ox and j_on > 0.0 and guards)
    ok = ok and g_d
    print("[lk] GATE D: enabled=False -> 0.0 exactly; Q = J*V/t_ox consistent ({:.3e} W/m^3 at "
          "2 V); t_ox guards raise -> {}".format(q_on, "PASS" if g_d else "FAIL"), flush=True)

    print("[lk] *** R16 OXIDE TUNNELING LEAKAGE: {} ***".format("PASS" if ok else "FAIL"),
          flush=True)
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
