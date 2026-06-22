"""Lumenairy EMT (Rytov) fast-screen bridge (roadmap v0.5 A5) vs the rigorous RCWA bridge.

The EMT screen homogenizes a sub-wavelength 1-D lamellar grating into a uniaxial (3,3) tensor
(Rytov) and solves it with the Berreman backend in microseconds -- a SCREEN that must converge
monotonically onto the rigorous RCWA/PMM as the period shrinks, NOT a replacement.

GATE A (convergence onto RCWA): a binary grating (eps 6 / 1, fill 0.5) at fixed thickness, swept
        period lambda/2 -> lambda/100. The EMT screen (period-independent) vs the rigorous RCWA
        bridge: x-pol (TM/perp/harmonic) AND y-pol (TE/par/arithmetic) errors DECREASE
        MONOTONICALLY and reach < 3e-3 at lambda/100. WRONG-MODEL GUARD: at lambda/2 the error is
        LARGE (> 3e-2) -- the screen is genuinely an approximation, so a trivial pass is excluded.
GATE B (Rytov tensor correctness): rytov_tensor_for_layer on a known binary grating equals the
        hand-computed diag(harmonic, arithmetic, arithmetic) means < 1e-12; the x/y axes carry
        DIFFERENT means (form birefringence is real, harmonic < arithmetic); order=2 sharpens the
        bulk index (differs from order=0); and order=2 on a 3-segment cell RAISES (binary-only).
GATE C (screen plumbing): homogenize_lamellar_layers replaces ONLY the lamellar inclusion layer
        (a uniform layer is left untouched), and make_lumenairy_emt_screen_solver == feeding the
        hand-built Rytov EpsField straight to the Berreman solver < 1e-12 (the screen IS just
        homogenize-then-Berreman).
GATE D (2-D mixing rules): maxwell_garnett_eps / bruggeman_eps endpoints are EXACT (fill 0 -> host,
        fill 1 -> inclusion), a dilute screen lies strictly BETWEEN the two constituents, and both
        are passive (Im >= 0); Bruggeman is symmetric under (swap phases, 1 - fill).

Honest SKIP (exit 0 + banner) when lumenairy is not importable.

Run: python -m validation.lumenairy_emt_screen
"""
import importlib.util
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dynameta.core.eps_field import EpsField
from dynameta.geometry import Design, Inclusion, Layer, Stack, UnitCell
from dynameta.geometry.cross_section import Rectangle
from dynameta.geometry.specs import OpticalSpec
from dynameta.materials import ConstantOptical, Material, MaterialRegistry

LAM = 1.55e-6
EPS_RIDGE = 6.0 + 0j
EPS_GROOVE = 1.0 + 0j
FILL = 0.5
THK = 180e-9


def _registry():
    reg = MaterialRegistry()
    reg.add(Material("air", ConstantOptical(1.0 + 0j)))
    reg.add(Material("glass", ConstantOptical(complex(1.5 ** 2))))
    reg.add(Material("ridge", ConstantOptical(EPS_RIDGE)))
    reg.add(Material("lo", ConstantOptical(2.1 + 0j)))
    return reg


def _grating_design(period, *, pol="x"):
    """A binary lamellar grating: a ridge Rectangle of width FILL*period (groove = air bg)."""
    w = FILL * period
    ridge = Inclusion(shape=Rectangle(period / 2.0, period / 2.0, w, period), material="ridge")
    return Design(name="emt", unit_cell=UnitCell.square(period),
                  stack=Stack(layers=[Layer("grating", THK, "air", inclusions=[ridge])],
                              superstrate_material="air", substrate_material="glass"),
                  electrodes=[], materials=_registry(),
                  optical=OpticalSpec(polarization=pol, incidence_angle_deg=0.0))


def main():
    if importlib.util.find_spec("lumenairy") is None:
        print("[emt] *** SKIP: lumenairy not installed -- EMT screen gates not run ***",
              flush=True)
        return True
    from dynameta.optics.lumenairy_bridge import (bruggeman_eps, homogenize_lamellar_layers,
                                                  make_lumenairy_berreman_solver,
                                                  make_lumenairy_emt_screen_solver,
                                                  make_lumenairy_rcwa_solver, maxwell_garnett_eps,
                                                  rytov_tensor_for_layer)

    print("[emt] === Lumenairy EMT (Rytov) screen vs rigorous RCWA ===", flush=True)
    ok = True
    n_sup, n_sub = 1.0 + 0j, 1.5 + 0j

    # ---- GATE A: monotone convergence onto rigorous RCWA as period -> lambda/100 ----
    screen = make_lumenairy_emt_screen_solver()
    rcwa = make_lumenairy_rcwa_solver(n_orders=20)
    fracs = [2.0, 4.0, 10.0, 25.0, 100.0]                # period = lambda / frac
    err_x, err_y = [], []
    for pol, store in (("x", err_x), ("y", err_y)):
        d_screen = _grating_design(LAM / 100.0, pol=pol)  # screen is period-independent
        R_emt = screen(d_screen, None, {}, LAM, n_sup, n_sub).R
        for fr in fracs:
            d = _grating_design(LAM / fr, pol=pol)
            R_rig = rcwa(d, None, {}, LAM, n_sup, n_sub).R
            store.append(abs(R_rig - R_emt))
    mono_x = all(err_x[i + 1] <= err_x[i] + 1e-9 for i in range(len(err_x) - 1))
    mono_y = all(err_y[i + 1] <= err_y[i] + 1e-9 for i in range(len(err_y) - 1))
    converged = err_x[-1] < 3e-3 and err_y[-1] < 3e-3
    large_at_half = err_x[0] > 3e-2 or err_y[0] > 3e-2     # NOT a trivial pass
    g_a = bool(mono_x and mono_y and converged and large_at_half)
    ok = ok and g_a
    print("[emt] GATE A: EMT->RCWA monotone {}/{}, |dR| at L/2 {:.2e}/{:.2e} -> L/100 {:.2e}/{:.2e} "
          "(<3e-3) -> {}".format(mono_x, mono_y, err_x[0], err_y[0], err_x[-1], err_y[-1],
                                 "PASS" if g_a else "FAIL"), flush=True)

    # ---- GATE B: Rytov tensor == hand-computed harmonic/arithmetic means ----
    d = _grating_design(LAM / 50.0)
    L = d.stack.layers[0]
    tens = rytov_tensor_for_layer(L, d, LAM, d.unit_cell.period_x_m, d.unit_cell.period_y_m)
    eps_par = FILL * EPS_RIDGE + (1.0 - FILL) * EPS_GROOVE          # arithmetic (y/z, along)
    eps_perp = 1.0 / (FILL / EPS_RIDGE + (1.0 - FILL) / EPS_GROOVE)  # harmonic (x, across)
    rytov_err = max(abs(tens[0, 0] - eps_perp), abs(tens[1, 1] - eps_par),
                    abs(tens[2, 2] - eps_par))
    off_diag = max(abs(tens[i, j]) for i in range(3) for j in range(3) if i != j)
    form_biref = abs(tens[0, 0] - tens[1, 1]) > 0.5 and (tens[0, 0].real < tens[1, 1].real)
    tens2 = rytov_tensor_for_layer(L, d, LAM, d.unit_cell.period_x_m, d.unit_cell.period_y_m,
                                   order=2)
    order2_sharpens = abs(tens2[1, 1] - tens[1, 1]) > 1e-6
    # binary-only order=2 guard: a cell with 3 DISTINCT media must raise
    reg3 = _registry()
    reg3.add(Material("mid", ConstantOptical(3.0 + 0j)))
    incs = [Inclusion(shape=Rectangle(0.15 * (LAM / 50.0), 0.5 * (LAM / 50.0),
                                      0.2 * (LAM / 50.0), LAM / 50.0), material="ridge"),
            Inclusion(shape=Rectangle(0.6 * (LAM / 50.0), 0.5 * (LAM / 50.0),
                                      0.3 * (LAM / 50.0), LAM / 50.0), material="mid")]
    d3 = Design(name="emt3", unit_cell=UnitCell.square(LAM / 50.0),
                stack=Stack(layers=[Layer("g3", THK, "air", inclusions=incs)],
                            superstrate_material="air", substrate_material="glass"),
                electrodes=[], materials=reg3,
                optical=OpticalSpec(polarization="x", incidence_angle_deg=0.0))
    raised3 = False
    try:
        rytov_tensor_for_layer(d3.stack.layers[0], d3, LAM, d3.unit_cell.period_x_m,
                               d3.unit_cell.period_y_m, order=2)
    except ValueError:
        raised3 = True
    g_b = bool(rytov_err < 1e-12 and off_diag < 1e-15 and form_biref and order2_sharpens
               and raised3)
    ok = ok and g_b
    print("[emt] GATE B: Rytov tensor err {:.1e}, off-diag {:.1e}, form-birefringent {}, order=2 "
          "sharpens {}, 3-seg order=2 raises {} -> {}".format(
              rytov_err, off_diag, form_biref, order2_sharpens, raised3,
              "PASS" if g_b else "FAIL"), flush=True)

    # ---- GATE C: homogenize plumbing (replaces only lamellar; screen == direct) ----
    d = _grating_design(LAM / 50.0, pol="x")
    # add a uniform film that must be left untouched by the homogenizer
    d.stack.layers.append(Layer("film", 120e-9, "lo"))
    ebr = homogenize_lamellar_layers(d, LAM)
    only_grating = set(ebr) == {"grating"} and getattr(ebr["grating"], "is_tensor", False)
    # screen solver == hand-built Rytov EpsField -> Berreman solver
    r_screen = make_lumenairy_emt_screen_solver()(d, None, {}, LAM, n_sup, n_sub)
    r_direct = make_lumenairy_berreman_solver()(d, None, ebr, LAM, n_sup, n_sub)
    plumb_err = max(abs(r_screen.R - r_direct.R), abs(r_screen.T - r_direct.T),
                    abs(r_screen.r - r_direct.r))
    # order=2 on a MULTI-MEDIA lamellar grating must NOT be misrouted/dropped: it falls back to
    # order=0 and still homogenizes (the binary-only order=2 error is no longer swallowed as
    # "not lamellar"). d3 is a 3-distinct-media lamellar cell.
    ebr2 = homogenize_lamellar_layers(d3, LAM, order=2)
    multimedia_homogenized = ("g3" in ebr2 and getattr(ebr2["g3"], "is_tensor", False))
    g_c = bool(only_grating and plumb_err < 1e-12 and multimedia_homogenized)
    ok = ok and g_c
    print("[emt] GATE C: homogenize replaces only lamellar {}, screen==direct {:.1e}, multi-media "
          "order=2 falls back (homogenized {}) -> {}".format(
              only_grating, plumb_err, multimedia_homogenized, "PASS" if g_c else "FAIL"),
          flush=True)

    # ---- GATE D: 2-D scalar mixing rules (endpoints, bounds, passivity, symmetry) ----
    eh, ei = 2.1 + 0j, 6.0 + 0j
    mg0 = maxwell_garnett_eps(eh, ei, 0.0)
    mg1 = maxwell_garnett_eps(eh, ei, 1.0)
    mg_dil = maxwell_garnett_eps(eh, ei, 0.2)
    endpoints = abs(mg0 - eh) < 1e-12 and abs(mg1 - ei) < 1e-12
    between = eh.real < mg_dil.real < ei.real
    br = bruggeman_eps(eh, ei, 0.4)
    br_swap = bruggeman_eps(ei, eh, 0.6)                  # symmetric: same effective medium
    passive = mg_dil.imag >= -1e-12 and br.imag >= -1e-12
    symmetric = abs(br - br_swap) < 1e-9
    g_d = bool(endpoints and between and passive and symmetric)
    ok = ok and g_d
    print("[emt] GATE D: MG endpoints {}, dilute between {}, passive {}, Bruggeman symmetric {} "
          "-> {}".format(endpoints, between, passive, symmetric, "PASS" if g_d else "FAIL"),
          flush=True)

    print("[emt] *** LUMENAIRY EMT SCREEN: {} ***".format("PASS" if ok else "FAIL"), flush=True)
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
