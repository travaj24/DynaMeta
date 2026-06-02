"""
FEM heat-equation thermal driver oracle (roadmap Phase-2a follow-on) + electro-thermal Joule coupling.
solve_thermal_fem reduces EXACTLY to the analytic series-thermal-resistance profile
(carriers.thermal.steady_layered_temperature) for a flux-driven stack, and to the uniform-Joule slab
profile T_mean = T_sink + Q L^2/(3k) for a single heated layer; it also couples the electrical solve
(sigma|E|^2 Joule from carriers.electrostatics_fem) into a temperature rise.

GATE A: per-layer mean T from the FEM == analytic steady_layered_temperature (rel-to-rise < 1e-3),
        flux-driven 3-layer stack (sink at the bottom, flux into the top). The series-R profile is
        piecewise linear, so order-2 FEM is essentially exact.
GATE B: single-layer slab with a uniform Joule source Q (flux=0): FEM mean T == T_sink + Q L^2/(3k)
        (the exact uniform-Joule quadratic profile), rel-to-rise < 1e-3; T = T_sink at the sink.
INFO  : electro-thermal -- E from electrostatics_fem -> Q = sigma|E|^2 in a conductive layer ->
        thermal_fem -> a physical, sign-correct temperature rise (T > T_sink).

Run: python -m validation.thermal_fem
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dynameta.carriers.thermal_fem import ThermalLayer, solve_thermal_fem
from dynameta.carriers.thermal import steady_layered_temperature
from dynameta.carriers.electrostatics_fem import ElectrostaticLayer, solve_electrostatics_fem

NM = 1e-9
PERIOD = 400 * NM
T_SINK = 300.0


def main():
    print("[tf] === FEM heat-equation thermal driver + electro-thermal ===", flush=True)

    # GATE A: flux-driven 3-layer stack -> series thermal resistance (piecewise-linear, FEM-exact)
    layers = [ThermalLayer("sink_l", 100 * NM, 140.0),       # Si-like
              ThermalLayer("mid", 60 * NM, 1.4),             # SiO2-like (the bottleneck)
              ThermalLayer("top_l", 40 * NM, 30.0)]
    flux = 1.0e8
    res = solve_thermal_fem(layers, period_x_m=PERIOD, period_y_m=PERIOD, flux_W_m2=flux,
                            T_sink_K=T_SINK)
    t_fem = res.mean_T_per_layer()
    t_an = steady_layered_temperature([L.k_thermal for L in layers],
                                      [L.thickness_m for L in layers], flux, T_SINK)
    rise = np.max(np.abs(t_an - T_SINK))
    relA = float(np.max(np.abs(t_fem - t_an)) / max(rise, 1e-30))
    gate_a = bool(relA < 1e-3)
    for L, tf, ta in zip(layers, t_fem, t_an):
        print("[tf]   {}: T_fem={:.5f}  series={:.5f} K (rise {:+.4f})".format(
            L.name, tf, ta, ta - T_SINK), flush=True)
    print("[tf] GATE A (FEM == series-R, rel-to-rise={:.2e}): {}".format(
        relA, "PASS" if gate_a else "FAIL"), flush=True)

    # GATE B: single-layer slab, uniform Joule Q, flux=0 -> T_mean = T_sink + Q L^2/(3k)
    L0, k0, Q = 80 * NM, 10.0, 5.0e15
    res2 = solve_thermal_fem([ThermalLayer("slab", L0, k0)], period_x_m=PERIOD, period_y_m=PERIOD,
                             flux_W_m2=0.0, T_sink_K=T_SINK, joule_W_m3=Q)
    t_fem2 = float(res2.mean_T_per_layer()[0])
    t_an2 = T_SINK + Q * L0 ** 2 / (3.0 * k0)
    t_sink_fem = res2.T_at(PERIOD * 0.5, PERIOD * 0.5, 0.0)
    relB = abs(t_fem2 - t_an2) / max(abs(t_an2 - T_SINK), 1e-30)
    gate_b = bool(relB < 1e-3 and abs(t_sink_fem - T_SINK) < 1e-6)
    print("[tf] Joule slab: T_mean_fem={:.5f}  analytic(T_sink+QL^2/3k)={:.5f} K ; T@sink={:.5f} ; "
          "rel={:.2e}".format(t_fem2, t_an2, t_sink_fem, relB), flush=True)
    print("[tf] GATE B (uniform-Joule slab == analytic, T=T_sink at sink): {}".format(
        "PASS" if gate_b else "FAIL"), flush=True)

    # INFO: electro-thermal chain E -> Q=sigma|E|^2 -> T
    estack = [ElectrostaticLayer("ox", 60 * NM, 4.0), ElectrostaticLayer("cond", 40 * NM, 9.0)]
    eres = solve_electrostatics_fem(estack, 1.0, period_x_m=PERIOD, period_y_m=PERIOD)
    ez_cond = float(eres.mean_Ez_per_layer()[1])             # field in the conductive layer
    sigma = 1.0e2                                            # modest S/m for a sane demo rise
    Qj = sigma * ez_cond ** 2                                # W/m^3
    tstack = [ThermalLayer("ox", 60 * NM, 1.4), ThermalLayer("cond", 40 * NM, 5.0)]
    tres = solve_thermal_fem(tstack, period_x_m=PERIOD, period_y_m=PERIOD, flux_W_m2=0.0,
                             T_sink_K=T_SINK, joule_W_m3={"cond": Qj})
    t_cond = float(tres.mean_T_per_layer()[1])
    print("[tf] INFO electro-thermal: E_cond={:.3e} V/m, Q=sigma|E|^2={:.3e} W/m^3 -> "
          "T_cond={:.5f} K (rise {:+.4f})".format(ez_cond, Qj, t_cond, t_cond - T_SINK), flush=True)

    overall = gate_a and gate_b
    print("[tf] *** FEM THERMAL DRIVER + ELECTRO-THERMAL: {} ***".format(
        "PASS" if overall else "FAIL"), flush=True)
    return overall


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
