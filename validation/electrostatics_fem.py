"""
FEM electrostatics (Poisson) E-field driver oracle (roadmap Phase-1a follow-on). The H1 Laplace solve
carriers.electrostatics_fem.solve_electrostatics_fem reduces EXACTLY to the analytic series-capacitor
field (carriers.electrostatics.layered_static_field_z) for a laterally uniform stack, and produces a
genuine LATERAL field for a laterally-varying (patterned) gate where the 1D series-cap does not apply.

GATE A: the per-layer mean E_z from the FEM == the analytic series-cap field (rel < 1e-3) on a 3-layer
        stack -- the FEM generalizes, but must match the exact result where the exact result holds.
GATE B: a laterally-varying top electrode (a -V..+V ramp across the cell) produces a significant
        LATERAL field (|Ex| > 1e6 V/m) -- the genuinely-2D effect the 1D series-cap (Ex = 0) cannot
        represent -- and the Laplace solve still closes cleanly.
INFO  : the FEM E_z drives PockelsEffect to the SAME tensor eps as the analytic E_z (the driver feeds
        the field-effect EffectModels).

Run: python -m validation.electrostatics_fem
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ngsolve as ng

from dynameta.carriers.electrostatics_fem import ElectrostaticLayer, solve_electrostatics_fem
from dynameta.carriers.electrostatics import layered_static_field_z
from dynameta.core.effects import PockelsEffect

NM = 1e-9
S = 1e9
LAYERS = [ElectrostaticLayer("l0", 50 * NM, 4.0), ElectrostaticLayer("l1", 30 * NM, 9.0),
          ElectrostaticLayer("l2", 20 * NM, 2.25)]
V = 2.0
PERIOD = 300 * NM


def main():
    print("[ef] === FEM electrostatics (Poisson) E-driver ===", flush=True)

    # GATE A: laterally uniform stack -> exact series-capacitor field
    res = solve_electrostatics_fem(LAYERS, V, period_x_m=PERIOD, period_y_m=PERIOD)
    ez_fem = res.mean_Ez_per_layer()
    ez_ser = layered_static_field_z([L.eps_static for L in LAYERS],
                                    [L.thickness_m for L in LAYERS], V)
    rel = float(np.max(np.abs(ez_fem - ez_ser) / np.maximum(np.abs(ez_ser), 1e-30)))
    gate_a = bool(rel < 1e-3)
    for L, ef, es in zip(LAYERS, ez_fem, ez_ser):
        print("[ef]   {}: E_z_fem={:+.4e}  series={:+.4e} V/m".format(L.name, ef, es), flush=True)
    print("[ef] GATE A (FEM == series-cap, max rel={:.2e}): {}".format(
        rel, "PASS" if gate_a else "FAIL"), flush=True)

    # GATE B: laterally-varying top electrode (-V .. +V ramp across x) -> lateral field
    ramp = V * (2.0 * ng.x / (PERIOD * S) - 1.0)               # -V at x=0 -> +V at x=PERIOD (nm coords)
    res2 = solve_electrostatics_fem(LAYERS, V, period_x_m=PERIOD, period_y_m=PERIOD,
                                    top_voltage_cf=ramp)
    z_top = sum(L.thickness_m for L in LAYERS) * 0.9           # near the patterned electrode
    ex_max = 0.0
    for xf in (0.25, 0.5, 0.75):
        E = res2.E_at(PERIOD * xf, PERIOD * 0.5, z_top)
        ex_max = max(ex_max, abs(E[0]))
    finite = bool(np.isfinite(ex_max))
    gate_b = bool(finite and ex_max > 1e6)
    print("[ef] split/ramp gate: max|Ex|={:.3e} V/m (1D series-cap gives Ex=0)".format(ex_max),
          flush=True)
    print("[ef] GATE B (laterally-varying gate -> lateral Ex > 1e6 V/m): {}".format(
        "PASS" if gate_b else "FAIL"), flush=True)

    # INFO: the FEM E_z feeds PockelsEffect to the same eps as the analytic E_z
    no, ne, r33 = 2.21, 2.14, 30.9e-12
    pk = PockelsEffect(eps_bg=np.diag([no ** 2, no ** 2, ne ** 2]).astype(complex),
                       r_voigt=np.array([[0, 0, 0], [0, 0, 0], [0, 0, r33], [0, 0, 0],
                                         [0, 0, 0], [0, 0, 0]], dtype=float))
    e_fem = pk.eps({"E": [0.0, 0.0, float(ez_fem[-1])]}, 1550e-9)
    e_an = pk.eps({"E": [0.0, 0.0, float(ez_ser[-1])]}, 1550e-9)
    print("[ef] INFO: Pockels eps via FEM-Ez vs analytic-Ez max|d|={:.2e}".format(
        float(np.max(np.abs(e_fem - e_an)))), flush=True)

    overall = gate_a and gate_b
    print("[ef] *** FEM ELECTROSTATICS E-DRIVER: {} ***".format("PASS" if overall else "FAIL"),
          flush=True)
    return overall


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
