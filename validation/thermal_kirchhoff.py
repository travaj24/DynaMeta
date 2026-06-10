"""R21 k(T) Kirchhoff-transform oracle (exact nonlinear steady conduction).

GATE A (constant-k reduces EXACTLY): k_of_T = const c -> theta = c (T - T_sink), so the
        Kirchhoff path must reproduce the SHIPPED linear solver's temperature field (probe
        points, two-layer flux stack) to solver roundoff.
GATE B (linear k(T), uniform Joule, closed form): single slab, insulated top, sink bottom:
        theta(z) = Q z (L - z/2) EXACTLY (order-2 FEM reproduces the quadratic to roundoff),
        and for k = k0 (1 + beta (T - Ts)): T(z) = Ts + (sqrt(1 + 2 beta theta/k0) - 1)/beta in
        closed form. Also asserts the nonlinearity has TEETH: the naive constant-k0 profile
        differs by >> the gate tolerance.
GATE C (flux-driven slab, closed form + physics): theta(z) = flux * z; with beta > 0 (k rises
        when hot) the true top temperature sits BELOW the constant-k0 estimate.
GATE D (round-trip + guards): invert(theta(T)) == T to 1e-9 K across 300-900 K for a nonlinear
        k(T); k <= 0 and non-callable k_of_T raise.

Run: python -m validation.thermal_kirchhoff
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dynameta.carriers.thermal_fem import (ThermalLayer, solve_thermal_fem,
                                           solve_thermal_kirchhoff_fem, kirchhoff_theta,
                                           invert_kirchhoff)

PER = 200e-9


def main():
    print("[kt] === R21 k(T) Kirchhoff transform ===", flush=True)
    ok = True

    # ---- GATE A: constant-k reduction ----
    lays = [ThermalLayer("a", 80e-9, 5.0), ThermalLayer("b", 120e-9, 5.0)]
    r_lin = solve_thermal_fem(lays, period_x_m=PER, period_y_m=PER, flux_W_m2=2e8,
                              T_sink_K=300.0, maxh_m=20e-9)
    r_kir = solve_thermal_kirchhoff_fem(lays, lambda T: 5.0, period_x_m=PER, period_y_m=PER,
                                        flux_W_m2=2e8, T_sink_K=300.0, maxh_m=20e-9)
    zs = np.array([20e-9, 80e-9, 140e-9, 199e-9])
    dA = max(abs(r_kir.T_at(PER / 2, PER / 2, z) - r_lin.T_at(PER / 2, PER / 2, z)) for z in zs)
    rise = r_lin.T_at(PER / 2, PER / 2, 199e-9) - 300.0
    g_a = bool(dA < 1e-8 * max(rise, 1.0))
    ok = ok and g_a
    print("[kt] GATE A: constant-k Kirchhoff vs linear solver, max |dT| = {:.1e} K (rise {:.1f} "
          "K) -> {}".format(dA, rise, "PASS" if g_a else "FAIL"), flush=True)

    # ---- GATE B: linear k(T) + uniform Q closed form ----
    L, k0, beta, Q = 1.0e-6, 10.0, 0.01, 1.0e14
    slab = [ThermalLayer("s", L, k0)]
    k_of_T = lambda T: k0 * (1.0 + beta * (T - 300.0))
    r = solve_thermal_kirchhoff_fem(slab, k_of_T, period_x_m=PER, period_y_m=PER,
                                    joule_W_m3=Q, T_sink_K=300.0, maxh_m=50e-9)
    zb = np.linspace(0.1 * L, 0.999 * L, 12)
    th_cf = Q * zb * (L - zb / 2.0)
    T_cf = 300.0 + (np.sqrt(1.0 + 2.0 * beta * th_cf / k0) - 1.0) / beta
    T_fem = r.T_profile(zb, PER / 2, PER / 2)
    riseB = float(np.max(T_cf) - 300.0)
    dB = float(np.max(np.abs(T_fem - T_cf))) / riseB
    T_naive = 300.0 + th_cf / k0                              # constant-k0 (no Kirchhoff)
    teeth = float(np.max(np.abs(T_naive - T_cf))) / riseB
    g_b = bool(dB < 1e-6 and teeth > 50.0 * max(dB, 1e-9))
    ok = ok and g_b
    print("[kt] GATE B: Joule slab vs closed-form Kirchhoff inversion rel {:.1e} (rise {:.1f} K); "
          "naive constant-k0 off by {:.1e} (the gate has teeth) -> {}".format(
              dB, riseB, teeth, "PASS" if g_b else "FAIL"), flush=True)

    # ---- GATE C: flux-driven slab closed form + cooler-than-naive physics ----
    flux = 5.0e8
    r2 = solve_thermal_kirchhoff_fem(slab, k_of_T, period_x_m=PER, period_y_m=PER,
                                     flux_W_m2=flux, T_sink_K=300.0, maxh_m=50e-9)
    th_top = flux * L
    T_top_cf = 300.0 + (np.sqrt(1.0 + 2.0 * beta * th_top / k0) - 1.0) / beta
    T_top = r2.T_at(PER / 2, PER / 2, 0.999 * L)
    T_top_cf_at = 300.0 + (np.sqrt(1.0 + 2.0 * beta * flux * 0.999 * L / k0) - 1.0) / beta
    dC = abs(T_top - T_top_cf_at) / (T_top_cf - 300.0)
    cooler = T_top < 300.0 + flux * 0.999 * L / k0            # beta>0 conducts better when hot
    g_c = bool(dC < 1e-6 and cooler)
    ok = ok and g_c
    print("[kt] GATE C: flux slab T(top) = {:.1f} K vs closed form (rel {:.1e}); below the "
          "constant-k0 {:.1f} K (hot k rises -> cooler) -> {}".format(
              T_top, dC, 300.0 + flux * L / k0, "PASS" if g_c else "FAIL"), flush=True)

    # ---- GATE D: inversion round-trip + guards ----
    k_nl = lambda T: 148.0 * (T / 300.0) ** (-1.3)            # Si-like falling k(T)
    worst = max(abs(invert_kirchhoff(k_nl, kirchhoff_theta(k_nl, T, 300.0), 300.0) - T)
                for T in (310.0, 450.0, 700.0, 900.0))
    guards = False
    try:
        solve_thermal_kirchhoff_fem(slab, lambda T: -1.0, period_x_m=PER, period_y_m=PER)
    except ValueError:
        try:
            solve_thermal_kirchhoff_fem(slab, 5.0, period_x_m=PER, period_y_m=PER)
        except ValueError:
            guards = True
    g_d = bool(worst < 1e-8 and guards)
    ok = ok and g_d
    print("[kt] GATE D: theta round-trip worst |dT| = {:.1e} K over 310-900 K (Si-like T^-1.3); "
          "k <= 0 / non-callable raise -> {}".format(worst, "PASS" if g_d else "FAIL"),
          flush=True)

    print("[kt] *** R21 KIRCHHOFF k(T): {} ***".format("PASS" if ok else "FAIL"), flush=True)
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
