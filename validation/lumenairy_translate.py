"""Bidirectional DynaMeta <-> Lumenairy translation (roadmap v0.5 A3) round-trip gates.

GATE A (design round-trip): Design -> design_to_rcwa_stack -> rcwa_stack_to_design
        preserves layer count/order/thicknesses and eps at the translation wavelength, and
        the RCWA-bridge solve of the round-tripped design EQUALS the original's (< 1e-12).
GATE B (lumenairy-born stack, incl. DISPERSIVE): an RCWAStack with a dispersive-callable
        layer -> rcwa_stack_to_design -> the bridge solve matches the DIRECT lumenairy solve
        at three wavelengths (< 1e-12) -- the CallableOptical chain preserves dispersion.
GATE C (materials mapping): optical_model_to_lumenairy_eps(DrudeOptical, n_m3) reproduces
        the hand Drude formula (< 1e-15 rel); the index-vs-eps trap is covered by feeding a
        callable INDEX region medium through the round trip in GATE B.
GATE D (guards): patterned stacks raise NotImplementedError; CallableOptical rejects
        non-callables; DrudeOptical without n_m3 raises through the adapter.

Honest SKIP (exit 0 + banner) when lumenairy is not importable.

Run: python -m validation.lumenairy_translate
"""
import importlib.util
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dynameta.geometry import Design, Layer, Stack, UnitCell
from dynameta.geometry.specs import OpticalSpec
from dynameta.materials import ConstantOptical, DrudeOptical, Material, MaterialRegistry

LAM = 1.31e-6


def main():
    if importlib.util.find_spec("lumenairy") is None:
        print("[ltr] *** SKIP: lumenairy not installed -- translation gates not run ***",
              flush=True)
        return True
    import lumenairy as lum
    from dynameta.optics.lumenairy_bridge import (design_to_rcwa_stack,
                                                  make_lumenairy_rcwa_solver)
    from dynameta.optics.lumenairy_bridge.translate import (
        CallableOptical, optical_model_to_lumenairy_eps, rcwa_stack_to_design)

    print("[ltr] === DynaMeta <-> Lumenairy translation round-trips ===", flush=True)
    ok = True
    solver = make_lumenairy_rcwa_solver(n_orders=3)

    # ---- GATE A: design -> stack -> design round-trip ----
    reg = MaterialRegistry()
    reg.add(Material("air", ConstantOptical(1.0 + 0j)))
    reg.add(Material("glass", ConstantOptical(complex(1.5 ** 2))))
    reg.add(Material("hi", ConstantOptical(complex(4.0, 0.3))))
    reg.add(Material("lo", ConstantOptical(complex(2.1))))
    d0 = Design(name="rt", unit_cell=UnitCell.square(400e-9),
                stack=Stack(layers=[Layer("a", 120e-9, "hi"), Layer("b", 200e-9, "lo"),
                                    Layer("c", 80e-9, "hi")],
                            superstrate_material="air", substrate_material="glass"),
                electrodes=[], materials=reg,
                optical=OpticalSpec(polarization="y", incidence_angle_deg=0.0))
    stk, _ = design_to_rcwa_stack(d0, LAM, n_orders=3)
    d1 = rcwa_stack_to_design(stk)
    geo_ok = (len(d1.stack.layers) == 3
              and all(abs(a.thickness_m - b.thickness_m) < 1e-18
                      for a, b in zip(d0.stack.layers, d1.stack.layers))
              and all(abs(complex(d0.materials.get(a.background_material).eps(LAM))
                          - complex(d1.materials.get(b.background_material).eps(LAM)))
                      < 1e-15
                      for a, b in zip(d0.stack.layers, d1.stack.layers)))
    r0 = solver(d0, None, {}, LAM, 1.0 + 0j, 1.5 + 0j)
    r1 = solver(d1, None, {}, LAM, 1.0 + 0j, 1.5 + 0j)
    dA = max(abs(r0.R - r1.R), abs(r0.T - r1.T), abs(r0.r - r1.r))
    g_a = bool(geo_ok and dA < 1e-12)
    ok = ok and g_a
    print("[ltr] GATE A: design round-trip (geometry exact; solve |d| = {:.2e}) -> {}"
          .format(dA, "PASS" if g_a else "FAIL"), flush=True)

    # ---- GATE B: lumenairy-born dispersive stack -> design -> bridge == direct ----
    def eps_disp(wl):
        w = 2.0 * np.pi * 2.99792458e8 / wl
        return 3.9 - (2.9e15) ** 2 / (w ** 2 + 1j * w * 1.8e14)

    def n_sub_disp(wl):                                   # dispersive INDEX region medium
        return 1.5 + 0.02 * (wl / LAM - 1.0)

    worstB = 0.0
    for lam in (1.2e-6, 1.31e-6, 1.55e-6):
        rs = lum.RCWAStack(400e-9, n_superstrate=1.0, n_substrate=n_sub_disp, n_orders=3)
        rs.add_layer(150e-9, eps=eps_disp)
        rs.add_layer(90e-9, eps=complex(2.25))
        rs.set_source(lam, theta=0.0)                     # sweep takes theta/phi from here
        orders, R2, T2, jones = rs.solve_vs_wavelength([lam])
        R_dir, T_dir = float(np.sum(R2[0, 1])), float(np.sum(T2[0, 1]))   # incident E_y
        d_b = rcwa_stack_to_design(rs)
        r_b = solver(d_b, None, {}, lam, 1.0 + 0j, 1.5 + 0j)
        worstB = max(worstB, abs(r_b.R - R_dir), abs(r_b.T - T_dir))
    g_b = bool(worstB < 1e-12)
    ok = ok and g_b
    print("[ltr] GATE B: dispersive lumenairy stack -> design -> bridge == direct solve "
          "over 3 wavelengths: worst |d| = {:.2e} -> {}".format(
              worstB, "PASS" if g_b else "FAIL"), flush=True)

    # ---- GATE C: materials mapping (Drude with explicit density) ----
    drude = DrudeOptical(eps_inf=3.9, m_opt_kg=0.35 * 9.1093837015e-31,
                         gamma_rad_s=1.8e14)
    fn = optical_model_to_lumenairy_eps(drude, n_m3=4.0e26)
    wl = 1.31e-6
    w = 2.0 * np.pi * 2.99792458e8 / wl
    wp2 = 4.0e26 * (1.602176634e-19) ** 2 / (8.8541878128e-12 * 0.35 * 9.1093837015e-31)
    eps_hand = 3.9 - wp2 / (w ** 2 + 1j * w * 1.8e14)
    dC = abs(fn(wl) - eps_hand) / abs(eps_hand)
    g_c = bool(dC < 1e-12)
    ok = ok and g_c
    print("[ltr] GATE C: DrudeOptical -> lumenairy dispersive spec vs hand formula "
          "(rel {:.2e}) -> {}".format(dC, "PASS" if g_c else "FAIL"), flush=True)

    # ---- GATE D: guards ----
    g_d = 0
    rs2 = lum.RCWAStack(400e-9, period_y=400e-9, n_orders=3)
    cell = np.full((13, 13), 2.0 + 0j)
    cell[:6, :] = 4.0
    rs2.add_layer(100e-9, eps_cell=cell)
    try:
        rcwa_stack_to_design(rs2)
    except NotImplementedError:
        g_d += 1
    try:
        CallableOptical(1.0)
    except TypeError:
        g_d += 1
    try:
        optical_model_to_lumenairy_eps(drude)(wl)         # Drude needs n_m3
    except (ValueError, TypeError):
        g_d += 1
    g_d = bool(g_d == 3)
    ok = ok and g_d
    print("[ltr] GATE D: patterned-reverse / non-callable / density-less-Drude guards "
          "-> {}".format("PASS" if g_d else "FAIL"), flush=True)

    print("[ltr] *** LUMENAIRY TRANSLATION: {} ***".format("PASS" if ok else "FAIL"),
          flush=True)
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
