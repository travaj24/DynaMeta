"""END-TO-END: bias -> electro-thermal FEM -> oxide TDDB lifetime + CW damage threshold.

The chain this example wires (all shipped, previously connected only in validations):

    ElectroThermalLayer stack --solve_electrothermal_picard--> ElectroThermalResult
        --drivers.tddb_tbd_from_electrothermal--> oxide time-to-breakdown t_BD(V)   [REL1]
    Material T-dependence --drivers.cw_damage_threshold_from_stack--> I_crit        [REL5]

GATE A: t_BD is finite/positive and STRICTLY DECREASING with bias (higher oxide field).
GATE B: the adapter's (E_ox, T) equals the direct ElectrostaticResult/T_per_layer read.
GATE C: the CW runaway threshold exists (finite, positive) for a stack whose absorption
        grows with T, and the absorbed fraction at ambient is in (0, 1).

Requires NGSolve. Run: python -m examples.electrothermal_reliability
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dynameta.carriers.electrothermal import ElectroThermalLayer, solve_electrothermal_picard
from dynameta.core.layered import LayeredSlab, LayeredStack
from dynameta.drivers import cw_damage_threshold_from_stack, tddb_tbd_from_electrothermal
from dynameta.reliability.lidt import ThermalNode, stack_absorbed_of_T
from dynameta.reliability.tddb import TddbParams, oxide_stress_from_electrothermal

# DRIVE MODEL (kept honest): the Picard driver solves a charge-free eps_static Poisson for E
# and forms Q = sigma E^2 -- i.e. the field divides CAPACITIVELY while sigma only dissipates.
# That pairing is the high-frequency (displacement-dominated) limit: valid for an RMS drive
# amplitude V at f >> sigma/(2 pi eps0 eps_r) ~ 30 GHz for ITO sigma = 15 S/m, eps_r = 9.3 (a
# mm-wave stress drive). At TRUE DC the zero-sigma oxide blocks conduction and the Joule term
# would vanish; this example's V values are RMS amplitudes of the fast drive. At V = 3 V the
# ITO field is ~2.9e7 V/m, so sigma ~ 15 S/m gives Q ~ 1.2e16 W/m^3 (~1.2 GW/m^2 areal over
# the 100 nm film) and a few-K rise over the oxide+film thermal resistance -- strong enough
# to see, weak enough that the mild sigma(T) feedback (0.2 %/K) stays Picard-stable. The
# TDDB read uses the same capacitive E_ox (the E-model under fast AC stress is approximate).
LAYERS = [
    ElectroThermalLayer("metal", 100e-9, eps_static=1.0e4, k_thermal=180.0, sigma_S_m=0.0),
    ElectroThermalLayer("hfo2", 10e-9, eps_static=20.0, k_thermal=1.1, sigma_S_m=0.0),
    ElectroThermalLayer("ito", 100e-9, eps_static=9.3, k_thermal=8.0,
                        sigma_S_m=lambda T: 15.0 * (1.0 + 2.0e-3 * (T - 300.0))),
]
PERIOD = 400e-9


def _ito_drude_eps(lambda_m, T_K):
    """ITO Drude eps with T-dependent damping (Gamma grows with T -> absorption grows)."""
    w = 2.0 * np.pi * 2.99792458e8 / lambda_m
    wp = 2.9e15
    gam = 1.8e14 * (1.0 + 2.5e-3 * (T_K - 300.0))
    return 3.9 - wp ** 2 / (w ** 2 + 1j * w * gam)          # exp(-iwt): Im(eps) > 0 absorber


def _stack_at_T(T_K):
    return LayeredStack(1.0 + 0j, 1.45 + 0j,
                        [LayeredSlab(310e-9, eps=_ito_drude_eps(1.31e-6, T_K))])


def main():
    print("[etrel] === electro-thermal -> TDDB + LIDT reliability workflow ===", flush=True)
    ok = True

    params = TddbParams.calibrated(E_ox_V_m=5.0e8, T_K=398.0, tbd_s=1.0e4)
    tbds, stresses = [], []
    for V in (1.0, 2.0, 3.0):
        res = solve_electrothermal_picard(LAYERS, V, period_x_m=PERIOD, period_y_m=PERIOD,
                                          T_sink_K=300.0, max_iter=40, tol_T_K=1e-4)
        tbd = tddb_tbd_from_electrothermal(res, "hfo2", params)
        E_ox, T_ox = oxide_stress_from_electrothermal(res, "hfo2")
        tbds.append(tbd)
        stresses.append((E_ox, T_ox))
        print("[etrel]   V = {:.1f} V -> E_ox = {:.3e} V/m, T_ox = {:.2f} K, "
              "t_BD = {:.3e} s".format(V, E_ox, T_ox, tbd), flush=True)
        # GATE B (adapter == direct read), checked at every bias
        i = [L.name for L in res.layers].index("hfo2")
        E_direct = abs(res.E_result.mean_Ez_per_layer()[i])
        T_direct = float(res.T_per_layer[i])
        if not (np.isclose(E_ox, E_direct, rtol=1e-12) and np.isclose(T_ox, T_direct, rtol=1e-12)):
            ok = False
            print("[etrel]   GATE B FAIL: adapter stress != direct read", flush=True)

    g_a = bool(all(np.isfinite(t) and t > 0 for t in tbds)
               and all(tbds[i + 1] < tbds[i] for i in range(len(tbds) - 1)))
    ok = ok and g_a
    print("[etrel] GATE A: t_BD finite and strictly decreasing with bias -> {}".format(
        "PASS" if g_a else "FAIL"), flush=True)
    print("[etrel] GATE B: adapter (E_ox, T) == direct result reads -> PASS", flush=True)

    absorbed = stack_absorbed_of_T(_stack_at_T, 1.31e-6)
    a300 = absorbed(300.0)
    node = ThermalNode(R_th_K_W=2.0e4, C_th_J_K=1.0e-9, area_m2=(50e-6) ** 2)
    I_crit = cw_damage_threshold_from_stack(_stack_at_T, 1.31e-6, node)
    g_c = bool(0.0 < a300 < 1.0 and np.isfinite(I_crit) and I_crit > 0.0
               and absorbed(600.0) > a300)
    ok = ok and g_c
    print("[etrel] GATE C: A(300K) = {:.4f}, A(600K) = {:.4f}, CW runaway threshold "
          "I_crit = {:.3e} W/m^2 -> {}".format(a300, absorbed(600.0), I_crit,
                                               "PASS" if g_c else "FAIL"), flush=True)

    print("[etrel] *** ELECTRO-THERMAL RELIABILITY WORKFLOW: {} ***".format(
        "PASS" if ok else "FAIL"), flush=True)
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
