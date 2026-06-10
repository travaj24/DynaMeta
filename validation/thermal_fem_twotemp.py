"""R14 spatially-resolved two-temperature FEM oracle.

GATE A (reduces to the lumped R9 TTM): all-insulated BCs + uniform source -> the spatial PDE
        collapses to the lumped ODE; FEM mean traces must match BOTH the exact closed form
        (lambda = G(1/C_e + 1/C_l); d(t) = (S/(C_e lambda))(1 - e^{-lambda t});
        C_e Te + C_l Tl = (C_e + C_l) T0 + S t) AND carrier_heating.two_temperature_response.
GATE B (reduces to single-T at large G): G = 1e20 equilibrates Te ~= Tl instantly; the summed
        system is the single-T transient with k = k_e + k_l, C = C_e + C_l -- the FEM traces must
        match solve_thermal_transient_fem on the same mesh/steps.
GATE C (steady analytic two-field profile): single uniform-source layer, sink bottom, insulated
        top. EXACT 1-D closed form: w = k_e Te + k_l Tl = (k_e + k_l) T_sink + S z (L - z/2);
        d = Te - Tl = d_p [1 - cosh(chi (L - z)) / cosh(chi L)], d_p = S k_l / (G (k_e + k_l)),
        chi^2 = G (1/k_e + 1/k_l); then Te = (w + k_l d)/(k_e + k_l), Tl = (w - k_e d)/(k_e + k_l).
GATE D (G = 0 decouples): lattice stays EXACTLY at T_sink (no source, no coupling); the electron
        trace equals the independent single-T solve with (k_e, C_e) to solver precision.

Run: python -m validation.thermal_fem_twotemp
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dynameta.carriers.thermal_fem import (ThermalLayer, ThermalLayerTwoTemp,
                                           solve_thermal_fem, solve_thermal_transient_fem,
                                           solve_thermal_twotemp_fem,
                                           solve_thermal_transient_twotemp_fem)
from dynameta.carriers.carrier_heating import TwoTempParams, two_temperature_response


def gate_a():
    """Uniform source + insulated everywhere == lumped TTM ODE (closed form + R9 cross-check)."""
    C_e, C_l, G, S, T0 = 1.0e4, 2.0e6, 1.0e16, 1.0e18, 300.0
    lay = ThermalLayerTwoTemp(name="m", thickness_m=100e-9, k_thermal=1.0, rho_kg_m3=2.0e3,
                              Cp_J_kgK=1.0e3, G_e_l=G, C_e_J_m3K=C_e, k_electron=10.0)
    r = solve_thermal_transient_twotemp_fem(
        [lay], period_x_m=100e-9, period_y_m=100e-9, t_end_s=5e-12, dt_s=1e-14,
        source_e_W_m3=S, T_sink_K=T0, theta=0.5, maxh_m=50e-9, order=2,
        bottom_bc="insulated", store_every=10)
    t = r.t_s
    lam = G * (1.0 / C_e + 1.0 / C_l)
    d = (S / (C_e * lam)) * (1.0 - np.exp(-lam * t))
    Te_cf = T0 + (S * t + C_l * d) / (C_e + C_l)
    Tl_cf = T0 + (S * t - C_e * d) / (C_e + C_l)
    Te, Tl = r.mean_Te_per_layer_t[:, 0], r.mean_Tl_per_layer_t[:, 0]
    rise = max(float(np.max(np.abs(Te_cf - T0))), 1e-30)
    err_cf = max(float(np.max(np.abs(Te - Te_cf))), float(np.max(np.abs(Tl - Tl_cf)))) / rise
    # independent in-house reference: the lumped R9 integrator
    _, Te_ode, Tl_ode = two_temperature_response(
        t, lambda tt: S, TwoTempParams(C_e=C_e, C_l=C_l, G_e_l=G, alpha_abs=1.0), T0_K=T0)
    err_ode = max(float(np.max(np.abs(Te - Te_ode))),
                  float(np.max(np.abs(Tl - Tl_ode)))) / rise
    ok = bool(err_cf < 1e-3 and err_ode < 1e-3)
    print("[t2] GATE A: insulated+uniform == lumped TTM -- vs closed form rel {:.1e}, vs R9 "
          "two_temperature_response rel {:.1e} (rise {:.1f} K, Te-Tl final {:.1f} K) -> {}".format(
              err_cf, err_ode, rise, float(Te[-1] - Tl[-1]), "PASS" if ok else "FAIL"), flush=True)
    return ok


def gate_b():
    """G = 1e20 (instant equilibration) == single-T transient with k_e + k_l, C_e + C_l."""
    k_e, k_l, C_e, C_l = 5.0, 5.0, 1.0e4, 2.0e6
    flux, T0 = 1.0e9, 300.0
    lay2 = ThermalLayerTwoTemp(name="m", thickness_m=200e-9, k_thermal=k_l, rho_kg_m3=2.0e3,
                               Cp_J_kgK=1.0e3, G_e_l=1.0e20, C_e_J_m3K=C_e, k_electron=k_e)
    r2 = solve_thermal_transient_twotemp_fem(
        [lay2], period_x_m=100e-9, period_y_m=100e-9, t_end_s=10e-9, dt_s=0.05e-9,
        flux_W_m2=flux, T_sink_K=T0, theta=1.0, maxh_m=25e-9, order=2, store_every=20)
    lay1 = ThermalLayer(name="m", thickness_m=200e-9, k_thermal=k_e + k_l, rho_kg_m3=1.0,
                        Cp_J_kgK=C_e + C_l)
    r1 = solve_thermal_transient_fem(
        [lay1], period_x_m=100e-9, period_y_m=100e-9, t_end_s=10e-9, dt_s=0.05e-9,
        flux_W_m2=flux, T_sink_K=T0, theta=1.0, maxh_m=25e-9, order=2, store_every=20)
    Te, Tl = r2.mean_Te_per_layer_t[:, 0], r2.mean_Tl_per_layer_t[:, 0]
    T1 = r1.mean_T_per_layer_t[:, 0]
    rise = max(float(np.max(np.abs(T1 - T0))), 1e-30)
    err = max(float(np.max(np.abs(Te - T1))), float(np.max(np.abs(Tl - T1)))) / rise
    dmax = float(np.max(np.abs(Te - Tl)))
    ok = bool(err < 5e-3 and dmax < 5e-3 * rise)
    print("[t2] GATE B: G=1e20 vs single-T (k_e+k_l, C_e+C_l) -- traj rel {:.1e}, max|Te-Tl| "
          "{:.1e} K (rise {:.1f} K) -> {}".format(err, dmax, rise, "PASS" if ok else "FAIL"),
          flush=True)
    return ok


def gate_c():
    """Steady uniform-source profile vs the exact 1-D two-field closed form."""
    k_e, k_l, G, S, T0, L = 10.0, 1.0, 1.0e12, 1.0e15, 300.0, 1.0e-6
    lay = ThermalLayerTwoTemp(name="m", thickness_m=L, k_thermal=k_l, rho_kg_m3=2.0e3,
                              Cp_J_kgK=1.0e3, G_e_l=G, C_e_J_m3K=1.0e4, k_electron=k_e)
    r = solve_thermal_twotemp_fem([lay], period_x_m=100e-9, period_y_m=100e-9,
                                  source_e_W_m3=S, T_sink_K=T0, maxh_m=60e-9, order=2)
    chi = np.sqrt(G * (1.0 / k_e + 1.0 / k_l))
    d_p = S * k_l / (G * (k_e + k_l))
    z = np.linspace(0.05 * L, 0.999 * L, 25)
    w = (k_e + k_l) * T0 + S * z * (L - z / 2.0)
    d = d_p * (1.0 - np.cosh(chi * (L - z)) / np.cosh(chi * L))
    Te_an = (w + k_l * d) / (k_e + k_l)
    Tl_an = (w - k_e * d) / (k_e + k_l)
    Te = np.array([r.Te_at(50e-9, 50e-9, zz) for zz in z])
    Tl = np.array([r.Tl_at(50e-9, 50e-9, zz) for zz in z])
    rise = max(float(np.max(Te_an - T0)), 1e-30)
    err = max(float(np.max(np.abs(Te - Te_an))), float(np.max(np.abs(Tl - Tl_an)))) / rise
    ok = bool(err < 1e-2)
    print("[t2] GATE C: steady profile vs exact cosh closed form (chi*L = {:.2f}) -- rel {:.1e} "
          "(rise {:.1f} K, Te-Tl top {:.1f} K analytic {:.1f}) -> {}".format(
              chi * L, err, rise, float(Te[-1] - Tl[-1]), float(d[-1]),
              "PASS" if ok else "FAIL"), flush=True)
    return ok


def gate_d():
    """G = 0: lattice pinned at T_sink; electron channel == independent single-T solve."""
    k_e, C_e, S, T0 = 10.0, 3.0e5, 1.0e15, 300.0
    lay2 = ThermalLayerTwoTemp(name="m", thickness_m=200e-9, k_thermal=1.0, rho_kg_m3=2.0e3,
                               Cp_J_kgK=1.0e3, G_e_l=0.0, C_e_J_m3K=C_e, k_electron=k_e)
    r2 = solve_thermal_transient_twotemp_fem(
        [lay2], period_x_m=100e-9, period_y_m=100e-9, t_end_s=2e-9, dt_s=0.02e-9,
        source_e_W_m3=S, T_sink_K=T0, theta=1.0, maxh_m=25e-9, order=2, store_every=10)
    lay1 = ThermalLayer(name="m", thickness_m=200e-9, k_thermal=k_e, rho_kg_m3=1.0, Cp_J_kgK=C_e)
    r1 = solve_thermal_transient_fem(
        [lay1], period_x_m=100e-9, period_y_m=100e-9, t_end_s=2e-9, dt_s=0.02e-9,
        joule_W_m3=S, T_sink_K=T0, theta=1.0, maxh_m=25e-9, order=2, store_every=10)
    Te, Tl = r2.mean_Te_per_layer_t[:, 0], r2.mean_Tl_per_layer_t[:, 0]
    T1 = r1.mean_T_per_layer_t[:, 0]
    rise = max(float(np.max(np.abs(T1 - T0))), 1e-30)
    err_e = float(np.max(np.abs(Te - T1))) / rise
    err_l = float(np.max(np.abs(Tl - T0)))
    ok = bool(err_e < 1e-8 and err_l < 1e-8)
    print("[t2] GATE D: G=0 decoupled -- Te vs independent single-T rel {:.1e}; max|Tl - T_sink| "
          "= {:.1e} K -> {}".format(err_e, err_l, "PASS" if ok else "FAIL"), flush=True)
    return ok


def main():
    print("[t2] === R14 spatially-resolved two-temperature FEM ===", flush=True)
    ok = True
    for g in (gate_a, gate_b, gate_c, gate_d):
        ok = g() and ok
    print("[t2] *** R14 TWO-TEMPERATURE FEM: {} ***".format("PASS" if ok else "FAIL"), flush=True)
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
