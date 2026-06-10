"""D4 Chynoweth impact-ionization / I_sub oracle (post-processor on solved DD fields).

GATE A (closed form + masking): ChynowethParams.alpha_n/p against independent math.exp at
        sample fields (machine), EXACTLY 0 at E <= 0, SI sanity of the Si constants
        (alpha_n(3e7 V/m) ~ 1.2e6 1/m -- the literature cm-units trap is x100 either way).
GATE B (constant-field EXACT oracle): the uniform n-type bar (validated against Ohm's law in
        D1) has E = V/L and J = sigma E everywhere, so I_sub == alpha_n(E) * J * Area in
        closed form; the directional FV edge quadrature must match (the zero-couple-diagonal
        mesh identity probed during development).
GATE C (reverse-bias junction physics): on the p-n diode, I_sub grows strongly with reverse
        bias (the depletion field rises ~ sqrt(Vbi + V) and alpha is exponential in -1/E);
        forward/zero bias produce comparatively negligible I_sub; zeroed coefficients give
        EXACTLY 0.
GATE D (discretization convergence): refining the junction mesh changes I_sub by < 5%
        (the FV quadrature is converged, not mesh-noise).

Run: python -m validation.impact_ionization_isub
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import devsim as ds

from dynameta.constants import Q_E
from dynameta.carriers.devsim_layered import LayeredDevsimBuilder, _EDGE_METAL_W_M
from dynameta.carriers.impact_ionization import (ChynowethParams, SILICON_VANOVERSTRAETEN,
                                                 substrate_current)
from dynameta.carriers import eq_registry as _R
from dynameta.sweep import BiasPoint
from validation.contact_current_drivers import _bar_design, P, T_SI, ND, MU_N, V_BAR


def _teardown(b):
    try:
        ds.delete_device(device=b.device); ds.delete_mesh(mesh=b.mesh_name); _R.clear(b.device)
    except Exception:
        pass


def gate_a():
    si = SILICON_VANOVERSTRAETEN
    ok = True
    for E in (3.0e7, 6.0e7, 1.2e8):
        ok = ok and abs(si.alpha_n(E) - 7.03e7 * math.exp(-1.231e8 / E)) <= 1e-9 * si.alpha_n(E)
        ok = ok and abs(si.alpha_p(E) - 1.582e8 * math.exp(-2.036e8 / E)) <= 1e-9 * si.alpha_p(E)
    masked = (si.alpha_n(0.0) == 0.0 and si.alpha_p(-5.0) == 0.0
              and np.all(si.alpha_n(np.array([0.0, 1.0e7, -1.0])) ==
                         np.array([0.0, si.alpha_n(1.0e7), 0.0])))
    a30 = si.alpha_n(3.0e7)
    sane = 1.0e5 < a30 < 1.0e7                     # ~1.2e6 1/m; a cm-units slip is x100 off
    ok = bool(ok and masked and sane)
    print("[ii] GATE A: Chynoweth closed form (machine), E<=0 masked EXACTLY 0, alpha_n(3e7 "
          "V/m) = {:.3e} 1/m in the SI sanity band -> {}".format(
              a30, "PASS" if ok else "FAIL"), flush=True)
    return ok


def gate_b():
    pars = ChynowethParams(a_n_per_m=1.0e6, b_n_V_per_m=2.0e5, a_p_per_m=0.0, b_p_V_per_m=1.0)
    d = _bar_design("bipolar_dd")
    b = LayeredDevsimBuilder(d, mesh_name="iibar_m", device_name="iibar_d")
    b.solve(BiasPoint({"anode": V_BAR}, "v"))
    i_sub = substrate_current(b.device, "si", pars)            # per-unit-depth [A/m]
    i_off = substrate_current(b.device, "si", ChynowethParams(0.0, 1.0, 0.0, 1.0))
    _teardown(b)
    L = P - 2.0 * _EDGE_METAL_W_M
    E_cf = V_BAR / L
    J_cf = Q_E * ND * MU_N * E_cf
    i_cf = pars.alpha_n(E_cf) * J_cf * (L * T_SI)              # alpha * J * Area, per depth
    rel = abs(i_sub - i_cf) / i_cf
    ok = bool(rel < 1e-3 and i_off == 0.0)
    print("[ii] GATE B: constant-field bar I_sub = {:.6e} A/m vs closed form alpha(E) J Area = "
          "{:.6e} A/m (rel {:.1e}); zero coefficients -> {!r} -> {}".format(
              i_sub, i_cf, rel, i_off, "PASS" if ok else "FAIL"), flush=True)
    return ok


def _diode_isub(v_anode, mid_spacing_m=2.0e-9):
    from validation.bipolar_diode_2d import build_diode_design
    d = build_diode_design()
    d.mesh_2d.x_spacing_feature_mid_m = mid_spacing_m
    tag = "ii{}_{}".format("m" if v_anode < 0 else "p",
                           "{:.0f}_{:.0f}".format(abs(v_anode) * 100, mid_spacing_m * 1e10))
    b = LayeredDevsimBuilder(d, mesh_name=tag + "_m", device_name=tag + "_d")
    b.solve(BiasPoint({"anode": v_anode}, "v"))
    i_sub = substrate_current(b.device, "si", SILICON_VANOVERSTRAETEN,
                              depth_m=d.unit_cell.period_y_m)
    _teardown(b)
    return i_sub


def gate_c():
    i_m5, i_m1, i_0 = _diode_isub(-5.0), _diode_isub(-1.0), _diode_isub(0.0)
    grows = i_m5 > 3.0 * i_m1 > 0.0
    fwd_small = i_0 < 1e-2 * i_m5
    ok = bool(grows and fwd_small)
    print("[ii] GATE C: reverse-bias growth I_sub(-5 V) = {:.3e} A > 3x I_sub(-1 V) = {:.3e} A; "
          "I_sub(0 V) = {:.3e} A negligible -> {}".format(
              i_m5, i_m1, i_0, "PASS" if ok else "FAIL"), flush=True)
    return ok


def gate_d():
    i_coarse = _diode_isub(-5.0, mid_spacing_m=2.0e-9)
    i_fine = _diode_isub(-5.0, mid_spacing_m=1.0e-9)
    rel = abs(i_fine - i_coarse) / max(abs(i_fine), 1e-300)
    ok = bool(rel < 5e-2)
    print("[ii] GATE D: junction-mesh refinement 2 nm -> 1 nm changes I_sub by {:.1e} (< 5e-2) "
          "-> {}".format(rel, "PASS" if ok else "FAIL"), flush=True)
    return ok


def main():
    print("[ii] === D4 Chynoweth impact ionization / I_sub ===", flush=True)
    ok = True
    for g in (gate_a, gate_b, gate_c, gate_d):
        ok = g() and ok
    print("[ii] *** D4 IMPACT IONIZATION: {} ***".format("PASS" if ok else "FAIL"), flush=True)
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
