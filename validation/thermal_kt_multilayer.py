"""R21 follow-on oracle: PER-LAYER k(T) (exact 1D Kirchhoff with interface theta-jumps) +
transient k(T(x)) FEM.

GATE A (constant-k reduces EXACTLY): the exact 1D multi-layer solver with constant callables
        reproduces carriers.thermal.steady_layered_temperature interface temperatures to
        machine (the series-resistance closed form).
GATE B (cross-check vs the shipped single-material Kirchhoff FEM): single layer, k(T) linear,
        uniform Joule -- the 1D solver's profile matches solve_thermal_kirchhoff_fem pointwise
        (< 1e-5 rel-to-rise; two completely different methods, one exact answer).
GATE C (the genuinely-new theta-jump case): TWO layers with DIFFERENT k(T) under a top flux --
        the transient k(T(x)) FEM run to steady state matches the exact 1D solver's interface
        and mid-layer temperatures (< 1e-3 rel-to-rise); interface flux continuity
        k_1(T_i) dT/dz|_1 == k_2(T_i) dT/dz|_2 holds in the exact solution by construction
        (checked numerically).
GATE D (constant-k transient byte-equivalence): solve_thermal_transient_kt_fem with constant
        callables reproduces the SHIPPED solve_thermal_transient_fem trace to solver roundoff.
GATE E (guards): missing layer coverage and non-positive k(T) raise.

Run: python -m validation.thermal_kt_multilayer
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dynameta.carriers.thermal import steady_layered_temperature
from dynameta.carriers.thermal_fem import (ThermalLayer, solve_thermal_kirchhoff_fem,
                                           solve_thermal_kirchhoff_layered_1d,
                                           solve_thermal_transient_fem,
                                           solve_thermal_transient_kt_fem)

PER = 200e-9


def main():
    print("[kt2] === R21 follow-on: per-layer k(T) + transient k(T(x)) ===", flush=True)
    ok = True

    # ---- GATE A: constant-k multi-layer == series-resistance closed form ----
    lays = [ThermalLayer("a", 100e-9, 140.0), ThermalLayer("b", 60e-9, 1.4),
            ThermalLayer("c", 40e-9, 30.0)]
    flux = 1.0e8
    r1d = solve_thermal_kirchhoff_layered_1d(
        lays, {"a": lambda T: 140.0, "b": lambda T: 1.4, "c": lambda T: 30.0}, flux_W_m2=flux)
    T_ref = steady_layered_temperature([140.0, 1.4, 30.0], [100e-9, 60e-9, 40e-9], flux, 300.0)
    # steady_layered_temperature returns the per-layer MEAN == MIDPOINT (linear profile at
    # constant k), so compare the exact solver at each layer's midpoint
    mids = 0.5 * (r1d.interface_z_m[:-1] + r1d.interface_z_m[1:])
    dA = float(np.max(np.abs(np.array([r1d.T_at_z(z) for z in mids]) - T_ref)))
    g_a = bool(dA < 1e-9)
    ok = ok and g_a
    print("[kt2] GATE A: constant-k interfaces vs series-resistance closed form, max |dT| = "
          "{:.1e} K -> {}".format(dA, "PASS" if g_a else "FAIL"), flush=True)

    # ---- GATE B: single-layer k(T) vs the shipped Kirchhoff FEM ----
    L, k0, beta, Q = 1.0e-6, 10.0, 0.01, 1.0e14
    slab = [ThermalLayer("s", L, k0)]
    k_lin = lambda T: k0 * (1.0 + beta * (T - 300.0))
    r1 = solve_thermal_kirchhoff_layered_1d(slab, k_lin, joule_W_m3={"s": Q})
    rfem = solve_thermal_kirchhoff_fem(slab, k_lin, period_x_m=PER, period_y_m=PER,
                                       joule_W_m3=Q, T_sink_K=300.0, maxh_m=50e-9)
    zs = np.linspace(0.05 * L, 0.999 * L, 15)
    prof_1d = np.array([r1.T_at_z(z) for z in zs])
    prof_fem = np.array([rfem.T_at(PER / 2, PER / 2, z) for z in zs])
    rise = float(np.max(prof_1d) - 300.0)
    dB = float(np.max(np.abs(prof_1d - prof_fem))) / rise
    g_b = bool(dB < 1e-5)
    ok = ok and g_b
    print("[kt2] GATE B: 1D exact vs shipped Kirchhoff FEM profile, rel {:.1e} (rise {:.1f} K) "
          "-> {}".format(dB, rise, "PASS" if g_b else "FAIL"), flush=True)

    # ---- GATE C: two layers, DIFFERENT k(T), flux-driven: exact 1D vs transient-FEM steady ----
    two = [ThermalLayer("lo", 120e-9, 8.0, rho_kg_m3=2e3, Cp_J_kgK=700.0),
           ThermalLayer("hi", 80e-9, 2.0, rho_kg_m3=2e3, Cp_J_kgK=700.0)]
    k_by = {"lo": lambda T: 8.0 * (300.0 / T) ** 1.3,        # Si-like falling k(T)
            "hi": lambda T: 2.0 * (1.0 + 0.004 * (T - 300.0))}
    flux2 = 6.0e8
    r2 = solve_thermal_kirchhoff_layered_1d(two, k_by, flux_W_m2=flux2)
    rt = solve_thermal_transient_kt_fem(two, k_by, period_x_m=PER, period_y_m=PER,
                                        t_end_s=6e-7, dt_s=4e-9, flux_W_m2=flux2,
                                        T_sink_K=300.0, maxh_m=20e-9, store_every=30)
    z_chk = np.array([60e-9, 119e-9, 121e-9, 160e-9, 199e-9])
    T_1d = np.array([r2.T_at_z(z) for z in z_chk])
    T_fem = np.array([rt.T_at(PER / 2, PER / 2, z) for z in z_chk])
    rise2 = float(r2.interface_T_K[-1] - 300.0)
    dC = float(np.max(np.abs(T_1d - T_fem))) / rise2
    # flux-continuity self-check on the exact solution (numeric dT/dz each side of z = 120 nm)
    h = 0.05e-9
    g_lo = (r2.T_at_z(120e-9 - h) - r2.T_at_z(120e-9 - 3 * h)) / (2 * h)
    g_hi = (r2.T_at_z(120e-9 + 3 * h) - r2.T_at_z(120e-9 + h)) / (2 * h)
    T_i = r2.interface_T_K[1]
    fcont = abs(k_by["lo"](T_i) * g_lo - k_by["hi"](T_i) * g_hi) / flux2
    g_c = bool(dC < 1e-3 and fcont < 1e-3)
    ok = ok and g_c
    print("[kt2] GATE C: theta-jump two-layer -- transient k(T(x)) FEM steady vs exact 1D, rel "
          "{:.1e} (rise {:.1f} K); interface flux continuity {:.1e} -> {}".format(
              dC, rise2, fcont, "PASS" if g_c else "FAIL"), flush=True)

    # ---- GATE D: constant-k kt-transient == shipped transient ----
    cons = [ThermalLayer("s", 150e-9, 5.0, rho_kg_m3=2.2e3, Cp_J_kgK=730.0)]
    rk = solve_thermal_transient_kt_fem(cons, lambda T: 5.0, period_x_m=PER, period_y_m=PER,
                                        t_end_s=4e-8, dt_s=1e-9, flux_W_m2=2e8, T_sink_K=300.0,
                                        maxh_m=30e-9, store_every=5)
    rc = solve_thermal_transient_fem(cons, period_x_m=PER, period_y_m=PER, t_end_s=4e-8,
                                     dt_s=1e-9, flux_W_m2=2e8, T_sink_K=300.0, maxh_m=30e-9,
                                     store_every=5)
    riseD = float(np.max(rc.mean_T_per_layer_t) - 300.0)
    dD = float(np.max(np.abs(rk.mean_T_per_layer_t - rc.mean_T_per_layer_t))) / riseD
    g_d = bool(dD < 1e-10)
    ok = ok and g_d
    print("[kt2] GATE D: constant-k kt-transient vs shipped transient trace, rel {:.1e} -> {}"
          .format(dD, "PASS" if g_d else "FAIL"), flush=True)

    # ---- GATE E: guards ----
    guards = False
    try:
        solve_thermal_kirchhoff_layered_1d(two, {"lo": lambda T: 8.0})   # missing 'hi'
    except ValueError:
        try:
            solve_thermal_kirchhoff_layered_1d(slab, lambda T: -1.0)
        except ValueError:
            guards = True
    g_e = bool(guards)
    ok = ok and g_e
    print("[kt2] GATE E: coverage + positivity guards raise -> {}".format(
        "PASS" if g_e else "FAIL"), flush=True)

    print("[kt2] *** R21 FOLLOW-ON PER-LAYER/TRANSIENT k(T): {} ***".format(
        "PASS" if ok else "FAIL"), flush=True)
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
