"""
Self-consistent electro-thermo-optic Picard driver oracle (roadmap R6). solve_electrothermal_picard
closes E -> Joule(sigma(T)|E|^2) -> T -> sigma(T) over the existing steady electrostatics + thermal
FEM solvers.

GATE A -- REDUCES TO ONE PASS AT WEAK COUPLING: with sigma CONSTANT (no T-dependence) the loop is a
        no-op after the first thermal solve. On a fixed mesh + factorization the second iteration
        reproduces the first to machine precision (residual_history[-1] ~ 0, converged, n_iter == 2);
        and the converged operating point matches an INDEPENDENT manual one-pass
        (solve_electrostatics_fem -> Q = sigma|E|^2 -> solve_thermal_fem) to mesh-noise tolerance.

GATE B -- ENERGY BALANCE: at the converged steady state the total Joule power generated equals the
        conductive power leaving the sink face, |P_in - P_out|/P_in < 1e-2 (independent FEM boundary-
        flux integral vs the analytic volume sum); and a single uniformly-heated slab recovers the
        analytic mean rise T_mean = T_sink + Q L^2/(3k).

GATE C -- COUPLING CONVERGES AND MATTERS: with a physical sigma(T) (mobility ~ T^-1.5, conductivity
        FALLS with heating = negative feedback) the Picard residual decreases monotonically and
        converges within max_iter at relax=1.0; the self-consistent operating point DIFFERS from the
        one-pass estimate by more than tol_T_K (the feature demonstrably does something); T > T_sink.

Run: python -m validation.electrothermal_picard
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dynameta.carriers.electrothermal import (ElectroThermalLayer, solve_electrothermal_picard)
from dynameta.carriers.electrostatics_fem import ElectrostaticLayer, solve_electrostatics_fem
from dynameta.carriers.thermal_fem import ThermalLayer, solve_thermal_fem

NM = 1e-9
PERIOD = 400 * NM
T_SINK = 300.0


def main():
    print("[et] === self-consistent electro-thermo-optic Picard ===", flush=True)
    ok = True

    # ---- GATE A: constant sigma -> loop is a no-op; matches the manual one-pass ----
    SIG = 1.0e2
    layers_a = [ElectroThermalLayer("ox", 60 * NM, 4.0, 1.4, sigma_S_m=0.0),
                ElectroThermalLayer("cond", 40 * NM, 9.0, 5.0, sigma_S_m=SIG)]
    resA = solve_electrothermal_picard(layers_a, 1.0, period_x_m=PERIOD, period_y_m=PERIOD,
                                       T_sink_K=T_SINK, max_iter=8, tol_T_K=1e-3, maxh_m=40 * NM)
    # manual one-pass (independent reference): E -> Q = sigma |Ez|^2 -> thermal
    estack = [ElectrostaticLayer("ox", 60 * NM, 4.0), ElectrostaticLayer("cond", 40 * NM, 9.0)]
    eres = solve_electrostatics_fem(estack, 1.0, period_x_m=PERIOD, period_y_m=PERIOD, maxh_m=40 * NM)
    ez = eres.mean_Ez_per_layer()
    Qman = SIG * ez[1] ** 2
    tstack = [ThermalLayer("ox", 60 * NM, 1.4), ThermalLayer("cond", 40 * NM, 5.0)]
    tman = solve_thermal_fem(tstack, period_x_m=PERIOD, period_y_m=PERIOD, T_sink_K=T_SINK,
                             joule_W_m3={"cond": Qman}, maxh_m=40 * NM).mean_T_per_layer()
    noop = resA.residual_history[-1]
    rise = max(np.max(np.abs(tman - T_SINK)), 1e-30)
    rel_man = float(np.max(np.abs(resA.T_per_layer - tman)) / rise)
    g_a = bool(noop < 1e-9 and resA.converged and resA.n_iter == 2 and rel_man < 2e-2)
    ok = ok and g_a
    print("[et] GATE A: const-sigma no-op resid[-1]={:.1e}, n_iter={}, vs manual one-pass T_cond "
          "{:.4f}/{:.4f} K (rel {:.2e}): {}".format(noop, resA.n_iter, resA.T_per_layer[1], tman[1],
                                                     rel_man, "PASS" if g_a else "FAIL"), flush=True)

    # ---- GATE B: energy balance + analytic slab rise ----
    # single conductive slab, flux=0, const sigma -> uniform Joule; mean rise = T_sink + Q L^2/(3k)
    Lc, kc = 80 * NM, 5.0
    slab = [ElectroThermalLayer("slab", Lc, 9.0, kc, sigma_S_m=SIG)]
    resB = solve_electrothermal_picard(slab, 1.0, period_x_m=PERIOD, period_y_m=PERIOD,
                                       T_sink_K=T_SINK, max_iter=8, tol_T_K=1e-4, maxh_m=20 * NM)
    Qslab = float(resB.joule_per_layer[0])
    t_an = T_SINK + Qslab * Lc ** 2 / (3.0 * kc)
    t_fem = float(resB.T_per_layer[0])
    rel_slab = abs(t_fem - t_an) / max(abs(t_an - T_SINK), 1e-30)
    p_in, p_out = resB.total_joule_W, resB.total_sink_outflux_W
    rel_bal = abs(p_in - p_out) / max(abs(p_in), 1e-30) if np.isfinite(p_out) else float("inf")
    g_b = bool(rel_slab < 1e-2 and rel_bal < 1e-2)
    ok = ok and g_b
    print("[et] GATE B: slab T_mean {:.4f}/{:.4f} K (rel {:.2e}); energy P_in={:.4e} P_out={:.4e} W "
          "(rel {:.2e}): {}".format(t_fem, t_an, rel_slab, p_in, p_out, rel_bal,
                                    "PASS" if g_b else "FAIL"), flush=True)

    # ---- GATE C: sigma(T) feedback converges AND shifts the operating point ----
    sig0, T0, p_exp = 5.0e2, T_SINK, 1.5         # mobility ~ T^-1.5 -> sigma(T) = sig0 (T0/T)^1.5
    sigma_T = lambda T: sig0 * (T0 / T) ** p_exp
    layers_c = [ElectroThermalLayer("ox", 60 * NM, 4.0, 1.4, sigma_S_m=0.0),
                ElectroThermalLayer("cond", 40 * NM, 9.0, 5.0, sigma_S_m=sigma_T)]
    resC = solve_electrothermal_picard(layers_c, 1.0, period_x_m=PERIOD, period_y_m=PERIOD,
                                       T_sink_K=T_SINK, max_iter=40, tol_T_K=1e-4, relax=1.0,
                                       maxh_m=40 * NM)
    # one-pass estimate: sigma evaluated at T_sink (no feedback)
    one_pass = solve_electrothermal_picard(layers_c, 1.0, period_x_m=PERIOD, period_y_m=PERIOD,
                                           T_sink_K=T_SINK, max_iter=1, tol_T_K=1e-4, maxh_m=40 * NM)
    rh = resC.residual_history
    monotone = all(rh[i + 1] <= rh[i] + 1e-12 for i in range(len(rh) - 1))
    shift = abs(resC.T_per_layer[1] - one_pass.T_per_layer[1])
    g_c = bool(resC.converged and monotone and shift > 1e-4 and resC.T_per_layer[1] > T_SINK)
    ok = ok and g_c
    print("[et] GATE C: sigma(T) converged in {} iters (monotone={}), T_cond_sc={:.4f} vs one-pass "
          "{:.4f} K (shift {:.4f} K > tol): {}".format(resC.n_iter, monotone, resC.T_per_layer[1],
                                                        one_pass.T_per_layer[1], shift,
                                                        "PASS" if g_c else "FAIL"), flush=True)

    print("[et] *** ELECTRO-THERMO-OPTIC PICARD: {} ***".format("PASS" if ok else "FAIL"), flush=True)
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
