"""
Transient FEM heat-equation oracle (roadmap R5): solve_thermal_transient_fem integrates
rho*Cp*dT/dt = div(k grad T) + Q by the theta-method. Two independent gates.

GATE T1 -- semi-infinite-solid CONSTANT-SURFACE-FLUX erfc (Carslaw and Jaeger): start a thick single
        layer uniform at T_sink, drive a constant flux q0 into the top face, and compare the FEM
        temperature at several depths/times to the closed form
          T(x,t)-T_sink = (2 q0 sqrt(alpha t/pi)/k) exp(-x^2/(4 alpha t)) - (q0 x/k) erfc(x/(2 sqrt(alpha t)))
        (x = depth below the heated face, alpha = k/(rho Cp)), at probe times with Fourier number
        alpha t / L^2 < 0.05 so the thermal wave has NOT reached the sink (the semi-infinite premise).
        This exercises rho*Cp directly through alpha. Independent reference = scipy.special.erfc.

GATE T2 -- steady recovered at large t (reduces-to-known-limit, mandatory house rule): a finite
        multilayer under a constant flux, uniform IC = T_sink, integrated backward-Euler to
        t_end >> the stack thermal time constant. The final per-layer mean T must equal BOTH the
        steady FEM (solve_thermal_fem, same a,f) AND the analytic series-resistance profile
        (carriers.thermal.steady_layered_temperature) to rel-to-rise < 1e-3; the approach is
        monotone (no overshoot for theta=1) and T = T_sink at the sink face throughout.

Run: python -m validation.thermal_transient_fem
"""

import os
import sys

import numpy as np
from scipy.special import erfc

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dynameta.carriers.thermal_fem import (ThermalLayer, solve_thermal_fem,
                                           solve_thermal_transient_fem)
from dynameta.carriers.thermal import steady_layered_temperature

NM = 1e-9
PERIOD = 400 * NM
T_SINK = 300.0


def _erfc_surface_flux(x, t, q0, k, alpha):
    """Carslaw & Jaeger constant-flux-into-a-semi-infinite-solid temperature rise at depth x, time t."""
    s = np.sqrt(alpha * t)
    return (2.0 * q0 * np.sqrt(alpha * t / np.pi) / k) * np.exp(-x ** 2 / (4.0 * alpha * t)) \
        - (q0 * x / k) * erfc(x / (2.0 * s))


def main():
    print("[ttf] === transient FEM heat equation (theta-method) ===", flush=True)
    ok = True

    # ---- GATE T1: constant-surface-flux erfc, semi-infinite (Fourier alpha t / L^2 << 1) ----
    L = 2000 * NM
    k, rho, Cp = 1.4, 2200.0, 730.0                 # SiO2-like
    alpha = k / (rho * Cp)
    q0 = 1.0e8
    t_probe = 1.0e-7
    Fo = alpha * t_probe / L ** 2
    assert Fo < 0.05, "semi-infinite premise violated (Fourier number {:.3f} >= 0.05)".format(Fo)
    dt = t_probe / 200.0
    slab = [ThermalLayer("slab", L, k, rho_kg_m3=rho, Cp_J_kgK=Cp)]
    tr = solve_thermal_transient_fem(slab, period_x_m=PERIOD, period_y_m=PERIOD, t_end_s=t_probe,
                                     dt_s=dt, flux_W_m2=q0, T_sink_K=T_SINK, theta=1.0,
                                     maxh_m=25 * NM, order=2)
    # probe STRICTLY interior depths (avoid the exact top face z=L, where point-location can fall just
    # outside the meshed boundary and return 0); still spans the ~sqrt(alpha t)~295 nm penetration zone.
    depths = np.array([20.0, 60.0, 100.0, 150.0, 200.0]) * NM     # below the heated top face
    rise_ref = _erfc_surface_flux(depths, t_probe, q0, k, alpha)
    cx = PERIOD * 0.5
    rise_fem = np.array([tr.T_at(cx, cx, L - x) - T_SINK for x in depths])
    relT1 = float(np.max(np.abs(rise_fem - rise_ref)) / max(np.max(np.abs(rise_ref)), 1e-30))
    g_t1 = bool(relT1 < 1e-2)
    ok = ok and g_t1
    for x, rf, rr in zip(depths, rise_fem, rise_ref):
        print("[ttf]   x={:5.0f} nm: dT_fem={:+.4f}  erfc={:+.4f} K".format(x / NM, rf, rr), flush=True)
    print("[ttf] GATE T1 (constant-flux erfc, Fo={:.3f}, rel-to-rise={:.2e}): {}".format(
        Fo, relT1, "PASS" if g_t1 else "FAIL"), flush=True)

    # ---- GATE T2: steady recovered at large t (reduces-to-known-limit) ----
    stack = [ThermalLayer("sink_l", 100 * NM, 140.0, rho_kg_m3=2329.0, Cp_J_kgK=700.0),   # Si
             ThermalLayer("mid", 60 * NM, 1.4, rho_kg_m3=2200.0, Cp_J_kgK=730.0),         # SiO2 bottleneck
             ThermalLayer("top_l", 40 * NM, 30.0, rho_kg_m3=7140.0, Cp_J_kgK=340.0)]      # ITO-ish
    flux = 1.0e8
    t_end, dt2 = 2.0e-6, 1.0e-8
    trS = solve_thermal_transient_fem(stack, period_x_m=PERIOD, period_y_m=PERIOD, t_end_s=t_end,
                                      dt_s=dt2, flux_W_m2=flux, T_sink_K=T_SINK, theta=1.0,
                                      maxh_m=30 * NM, order=2)
    t_final = trS.mean_T_per_layer_t[-1]
    steady_fem = solve_thermal_fem(stack, period_x_m=PERIOD, period_y_m=PERIOD, flux_W_m2=flux,
                                   T_sink_K=T_SINK, maxh_m=30 * NM, order=2).mean_T_per_layer()
    steady_an = steady_layered_temperature([L.k_thermal for L in stack],
                                           [L.thickness_m for L in stack], flux, T_SINK)
    rise = max(np.max(np.abs(steady_an - T_SINK)), 1e-30)
    rel_fem = float(np.max(np.abs(t_final - steady_fem)) / rise)
    rel_an = float(np.max(np.abs(t_final - steady_an)) / rise)
    # monotone approach (theta=1, no overshoot) + sink pinned
    dmono = float(np.min(np.diff(trS.mean_T_per_layer_t, axis=0)))
    t_sink_final = trS.T_at(PERIOD * 0.5, PERIOD * 0.5, 0.0)
    g_t2 = bool(rel_fem < 1e-3 and rel_an < 1e-3 and dmono > -1e-6 and abs(t_sink_final - T_SINK) < 1e-6)
    ok = ok and g_t2
    for L_, tf, sf, sa in zip(stack, t_final, steady_fem, steady_an):
        print("[ttf]   {}: T(t_end)={:.5f}  steady_fem={:.5f}  series={:.5f} K".format(
            L_.name, tf, sf, sa), flush=True)
    print("[ttf] GATE T2 (large-t == steady FEM rel={:.2e}, == series-R rel={:.2e}, monotone d>={:+.1e}, "
          "T@sink={:.5f}): {}".format(rel_fem, rel_an, dmono, t_sink_final,
                                      "PASS" if g_t2 else "FAIL"), flush=True)

    # INFO: 10-90 rise time of the top-layer mean-T trace vs the lumped tau ~ rhoCp Rtot L
    top = trS.mean_T_per_layer_t[:, -1]
    rise_top = top - top[0]
    tgt = rise_top[-1]
    if tgt > 0:
        ts = trS.t_s
        t10 = float(np.interp(0.10 * tgt, rise_top, ts))
        t90 = float(np.interp(0.90 * tgt, rise_top, ts))
        print("[ttf] INFO top-layer 10-90 rise = {:.3e} s (final rise {:+.4f} K)".format(
            t90 - t10, tgt), flush=True)

    print("[ttf] *** TRANSIENT THERMAL FEM: {} ***".format("PASS" if ok else "FAIL"), flush=True)
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
