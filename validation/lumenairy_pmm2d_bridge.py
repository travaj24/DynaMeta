"""Lumenairy 2-D crossed-patterned PMM bridge (audit 8.1-4) vs TMM / cross-engine oracles.

GATE A (uniform == TMM, both engines): a 2-layer ASYMMETRIC lossy stack at normal +
        30 deg y/p -- R/T/r match tmm_reference at machine level (pins the stack layer
        ORDER, the pol-row mapping, and the shared lab-basis -> Byrnes p-pol r
        conversion on BOTH the pure and hybrid engines).
GATE B (the 2-D REFEREE role): an ITO-like NEAR-ENZ METAL patch (eps = -3 + 1j, the
        modulator regime; harder metals converge too slowly for a validation-scale
        gate on EVERY engine -- measured during bring-up) on a dielectric spacer.
        PMM2DStackPure (no Fourier floor, wall-resolved staggered basis) is the
        reference; the RCWA bridge on the SAME Design must CONVERGE TOWARD it: |R -
        R_pure| strictly shrinking over n_orders 4 -> 8 -> 12 -> 16 and < 1.5e-3 at
        n=16 (direction-sensitive, the validation/lumenairy_pmm_bridge GATE B/E
        pattern), with the pure reference degree-stable (|R(M=6) - R(M=5)| below the
        final RCWA gap).  The hybrid engine must land within 5e-3 of pure -- three
        methods (staggered-modal, Fourier-projected-modal, FMM) on one geometry.
GATE C (slab order, audit C5-1): an asymmetric LOSSY linear-in-z graded profile --
        slice_eps_field returns SUBSTRATE-first slabs and both 2-D PMM engines build
        superstrate-first, so an unreversed translation vertically flips the profile.
        Both engines match TMM and the RCWA bridge on the same design; the FLIPPED
        profile differs by > 1e-3 (the fixture keeps its discriminating power).
GATE D (scope guards raise): conical azimuth, incidence_side='bottom', a Circle
        inclusion on the pure engine, non-commensurate Rectangle walls (pure), a
        laterally-structured gridded EpsField (pure), tensor EpsField (both), and a
        bogus engine name.
GATE E (per-layer absorption, audit C3): a LOSSY ITO patch (eps -3 + 1j) on a LOSSY
        'hi' spacer (eps 4 + 0.3j).  The PURE engine (internal retention new in
        lumenairy 5.22) must close its own budget |A_independent - (1 - R - T)| <
        1e-6, be genuinely absorbing (A_independent > 1e-3), key
        per_region_absorption by DESIGN layer name, and vanish (< 1e-6) on the
        lossless twin -- and absorption=False must leave per_region_absorption=None
        (byte-identical off-switch).  Hybrid-vs-Pure cross-check: both engines close
        their OWN budget and their TOTAL A_independent agree (staggered-modal vs
        Fourier-projected internal quadrature -- different bases, so gated at the
        convergence-appropriate tolerance measured at bring-up).

Honest SKIP (exit 0 + banner) when lumenairy is not importable.

Run: python -m validation.lumenairy_pmm2d_bridge     (~2-3 min: the pure staggered
     eig on the patch union grid dominates; cases are deliberately small, see the
     budget note in each gate)
"""
import importlib.util
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dynameta.core.eps_field import EpsField
from dynameta.geometry import Design, Inclusion, Layer, Stack, UnitCell
from dynameta.geometry.cross_section import Circle, Rectangle
from dynameta.geometry.specs import OpticalSpec
from dynameta.materials import ConstantOptical, Material, MaterialRegistry

LAM = 1.31e-6
PER = 600e-9


def _design(layers, *, pol="y", theta=0.0, extra=()):
    reg = MaterialRegistry()
    reg.add(Material("air", ConstantOptical(1.0 + 0j)))
    reg.add(Material("glass", ConstantOptical(complex(1.5 ** 2))))
    reg.add(Material("hi", ConstantOptical(complex(4.0, 0.3))))
    reg.add(Material("lo", ConstantOptical(complex(2.1))))
    # near-ENZ metal (ITO-like at 1.31 um): genuinely Re(eps) < 0 but soft enough
    # that all three engines converge at validation scale (see GATE B header)
    reg.add(Material("ito", ConstantOptical(complex(-3.0, 1.0))))
    for nm, eps in extra:
        reg.add(Material(nm, ConstantOptical(eps)))
    return Design(name="pmm2d", unit_cell=UnitCell.square(PER),
                  stack=Stack(layers=layers, superstrate_material="air",
                              substrate_material="glass"),
                  electrodes=[], materials=reg,
                  optical=OpticalSpec(polarization=pol, incidence_angle_deg=theta))


def _patch_design(pol="x"):
    """Centered 300 nm ITO patch (walls at PER/4 and 3 PER/4 -> exact on the pure
    (4, 4) union grid AND on the bridge's 128-pixel raster) ON TOP of a lo spacer.
    Design layers are BOTTOM -> TOP, so the spacer comes first (the patch-on-top
    geometry diffracts strongly, R ~ 0.07 -- a robustly discriminating referee
    case; buried-patch R ~ 0.009 left only a 2% monotonicity margin)."""
    patch = Inclusion(shape=Rectangle(PER / 2.0, PER / 2.0, PER / 2.0, PER / 2.0),
                      material="ito")
    return _design([Layer("spacer", 150e-9, "lo"),
                    Layer("patch", 60e-9, "air", inclusions=[patch])], pol=pol)


def _graded_design_and_eps():
    """Asymmetric LOSSY graded fixture (audit C5-1): eps(top) >> eps(bottom), loss
    concentrated toward the top -- R differs measurably from the flipped copy."""
    d = _design([Layer("a", 120e-9, "hi")])
    z_nm = np.linspace(0.0, 120.0, 25)           # ascending = substrate-first
    u = z_nm / 120.0
    eps_z = 2.0 + 6.7 * u ** 2 + 1.0j * u ** 3
    ef = EpsField(z_axis_u=z_nm, y_axis_u=np.zeros(1), x_axis_u=np.zeros(1),
                  values_zyx=eps_z.reshape(-1, 1, 1).astype(complex))
    return d, {"a": ef}


def _lossy_patch_design(pol="x"):
    """A LOSSY ITO patch (eps -3 + 1j, walls at PER/4 & 3 PER/4 -> pure (4, 4) union
    grid) ON TOP of a LOSSY 'hi' spacer (eps 4 + 0.3j) -- BOTH Im(eps) > 0, the
    per-layer absorption fixture (GATE E).  Design layers are BOTTOM -> TOP."""
    patch = Inclusion(shape=Rectangle(PER / 2.0, PER / 2.0, PER / 2.0, PER / 2.0),
                      material="ito")
    return _design([Layer("spacer", 150e-9, "hi"),
                    Layer("patch", 60e-9, "air", inclusions=[patch])], pol=pol)


def _lossless_patch_design(pol="x"):
    """The lossless twin of _lossy_patch_design (real-eps materials throughout, same
    geometry): a 'glass' patch (eps 2.25) on a 'lo' spacer (eps 2.1).  A_independent
    must vanish -- the honest zero check for the absorption path."""
    patch = Inclusion(shape=Rectangle(PER / 2.0, PER / 2.0, PER / 2.0, PER / 2.0),
                      material="glass")
    return _design([Layer("spacer", 150e-9, "lo"),
                    Layer("patch", 60e-9, "air", inclusions=[patch])], pol=pol)


def main():
    if importlib.util.find_spec("lumenairy") is None:
        print("[lp2b] *** SKIP: lumenairy not installed -- PMM2D gates not run ***",
              flush=True)
        return True
    from dynameta.optics.lumenairy_bridge import (make_lumenairy_pmm2d_solver,
                                                  make_lumenairy_rcwa_solver,
                                                  pure_union_grid_n)
    from dynameta.optics.tmm_reference import make_layered_tmm_solver

    print("[lp2b] === Lumenairy PMM2D bridge vs TMM / cross-engine referee ===",
          flush=True)
    ok = True
    tmm = make_layered_tmm_solver()

    # ---- GATE A: unstructured asymmetric lossy stack vs TMM (both engines) ----
    # pure n_modes=6 measured 5.0e-13 worst; hybrid is Rayleigh-exact on uniform
    # films (measured ~1e-15).  Threshold 1e-10 = two decades of margin.
    lays = [Layer("a", 120e-9, "hi"), Layer("b", 200e-9, "lo")]
    worst = {"pure": 0.0, "hybrid": 0.0}
    solv = {"pure": make_lumenairy_pmm2d_solver(engine="pure", n_modes=6, n_orders=3),
            "hybrid": make_lumenairy_pmm2d_solver(engine="hybrid", degree=7,
                                                  n_orders=3)}
    for pol, th in (("y", 0.0), ("y", 30.0), ("p", 30.0)):
        d = _design(lays, pol=pol, theta=th)
        r_t = tmm(d, None, {}, LAM, 1.0 + 0j, 1.5 + 0j)
        for eng in ("pure", "hybrid"):
            r_p = solv[eng](d, None, {}, LAM, 1.0 + 0j, 1.5 + 0j)
            worst[eng] = max(worst[eng], abs(r_p.R - r_t.R), abs(r_p.T - r_t.T),
                             abs(r_p.r - r_t.r))
    g_a = bool(worst["pure"] < 1e-10 and worst["hybrid"] < 1e-10)
    ok = ok and g_a
    print("[lp2b] GATE A: uniform vs TMM (normal + 30deg y/p): worst |d| pure = "
          "{:.2e}, hybrid = {:.2e} -> {}".format(worst["pure"], worst["hybrid"],
                                                 "PASS" if g_a else "FAIL"),
          flush=True)

    # ---- GATE B: pure PMM2D as the 2-D convergence referee for RCWA ----
    d = _patch_design(pol="x")
    n_grid = pure_union_grid_n(d)
    ref_lo = make_lumenairy_pmm2d_solver(engine="pure", n_modes=5, n_orders=3)(
        d, None, {}, LAM, 1.0 + 0j, 1.5 + 0j)
    ref = make_lumenairy_pmm2d_solver(engine="pure", n_modes=6, n_orders=3)(
        d, None, {}, LAM, 1.0 + 0j, 1.5 + 0j)
    self_conv = abs(ref.R - ref_lo.R)
    dists = []
    for n in (4, 8, 12, 16):
        rc = make_lumenairy_rcwa_solver(n_orders=n)(d, None, {}, LAM,
                                                    1.0 + 0j, 1.5 + 0j)
        dists.append(abs(rc.R - ref.R))
        print("[lp2b]   RCWA n_orders {:2d}: R = {:.5f} (pure ref {:.5f}, "
              "|d| = {:.1e})".format(n, rc.R, ref.R, dists[-1]), flush=True)
    r_hyb = make_lumenairy_pmm2d_solver(engine="hybrid", degree=9, n_orders=9)(
        d, None, {}, LAM, 1.0 + 0j, 1.5 + 0j)
    d_hyb = abs(r_hyb.R - ref.R)
    # measured at bring-up: dists 5.7e-3 -> 4.0e-3 -> 1.6e-3 -> 5.8e-4 (strictly
    # decreasing TOWARD the pure value); pure self-convergence 3.1e-4 sits BELOW
    # the final RCWA gap, so the approach direction is resolvable, not noise
    g_b = bool(n_grid == 4
               and dists[3] < dists[2] < dists[1] < dists[0]
               and dists[3] < 1.5e-3 and self_conv < 1e-3
               and self_conv < dists[2] and d_hyb < 5e-3)
    ok = ok and g_b
    print("[lp2b] GATE B: union grid N = {}; RCWA converges TOWARD pure "
          "({:.1e} -> {:.1e} -> {:.1e} -> {:.1e}, final < 1.5e-3); pure degree-"
          "stability {:.1e}; hybrid within {:.1e} (< 5e-3) -> {}".format(
              n_grid, dists[0], dists[1], dists[2], dists[3], self_conv, d_hyb,
              "PASS" if g_b else "FAIL"), flush=True)

    # ---- GATE C: graded slab ORDER (audit C5-1) on an asymmetric lossy profile ----
    d, ebr = _graded_design_and_eps()
    r_t = tmm(d, None, ebr, LAM, 1.0 + 0j, 1.5 + 0j)
    r_rc = make_lumenairy_rcwa_solver(n_orders=2)(d, None, ebr, LAM,
                                                  1.0 + 0j, 1.5 + 0j)
    worst_c = 0.0
    for eng in ("pure", "hybrid"):
        r_p = solv[eng](d, None, ebr, LAM, 1.0 + 0j, 1.5 + 0j)
        worst_c = max(worst_c, abs(r_p.R - r_t.R), abs(r_p.T - r_t.T),
                      abs(r_p.R - r_rc.R))
    # the fixture must actually discriminate the flip: the same profile reversed
    # in z gives a materially different R (else a future regression could hide)
    d2, ebr2 = _graded_design_and_eps()
    ef = ebr2["a"]
    ebr_flip = {"a": EpsField(z_axis_u=ef.z_axis_u, y_axis_u=ef.y_axis_u,
                              x_axis_u=ef.x_axis_u,
                              values_zyx=ef.values_zyx[::-1].copy())}
    r_flip = solv["pure"](d2, None, ebr_flip, LAM, 1.0 + 0j, 1.5 + 0j)
    disc = abs(r_flip.R - r_t.R)
    g_c = bool(worst_c < 1e-8 and disc > 1e-3)
    ok = ok and g_c
    print("[lp2b] GATE C: graded slab order vs TMM + RCWA bridge: worst |d| = "
          "{:.2e} (< 1e-8); flipped-profile discrimination |dR| = {:.2e} "
          "(> 1e-3) -> {}".format(worst_c, disc, "PASS" if g_c else "FAIL"),
          flush=True)

    # ---- GATE D: scope guards raise ----
    g_d = 0
    pure = solv["pure"]
    # 1) conical azimuth (audit C4-2)
    d = _design(lays, pol="y", theta=30.0)
    object.__setattr__(d.optical, "azimuth_deg", 20.0)
    try:
        pure(d, None, {}, LAM, 1.0 + 0j, 1.5 + 0j)
    except NotImplementedError:
        g_d += 1
    # 2) bottom incidence
    d = _design(lays, pol="y")
    object.__setattr__(d.optical, "incidence_side", "bottom")
    try:
        pure(d, None, {}, LAM, 1.0 + 0j, 1.5 + 0j)
    except NotImplementedError:
        g_d += 1
    # 3) non-Rectangle inclusion on the pure engine (analytic-walls-only scope)
    disk = Inclusion(shape=Circle(PER / 2.0, PER / 2.0, PER / 4.0), material="ito")
    try:
        pure(_design([Layer("d", 60e-9, "air", inclusions=[disk])], pol="y"),
             None, {}, LAM, 1.0 + 0j, 1.5 + 0j)
    except ValueError:
        g_d += 1
    # 4) non-commensurate Rectangle walls (no uniform N <= pure_max_cells fits)
    odd = Inclusion(shape=Rectangle(PER / 2.0, PER / 2.0, 0.46 * PER, 0.46 * PER),
                    material="ito")
    try:
        pure(_design([Layer("o", 60e-9, "air", inclusions=[odd])], pol="y"),
             None, {}, LAM, 1.0 + 0j, 1.5 + 0j)
    except ValueError:
        g_d += 1
    # 5) laterally-STRUCTURED gridded EpsField on pure (genuinely x-varying values;
    # laterally-uniform grids are in scope -- they slice to uniform slabs, GATE C)
    grid = EpsField(values_zyx=np.broadcast_to(2.0 + 0.1 * np.arange(8.0),
                                               (3, 1, 8)).astype(complex),
                    z_axis_u=np.array([0.0, 50.0, 100.0]),
                    x_axis_u=np.arange(8.0), y_axis_u=np.array([0.0]))
    try:
        pure(_design([Layer("a", 100e-9, "hi")], pol="y"), None, {"a": grid},
             LAM, 1.0 + 0j, 1.5 + 0j)
    except ValueError:
        g_d += 1
    # 6) tensor EpsField (both engines are scalar-only in v1)
    eps_t = np.diag([4.0, 4.0, 4.0]).astype(complex)
    for eng in ("pure", "hybrid"):
        try:
            solv[eng](_design([Layer("a", 100e-9, "hi")], pol="y"),
                      None, {"a": EpsField(tensor=eps_t)}, LAM, 1.0 + 0j, 1.5 + 0j)
        except ValueError:
            g_d += 1
    # 7) bogus engine name
    try:
        make_lumenairy_pmm2d_solver(engine="fmm")
    except ValueError:
        g_d += 1
    g_d = bool(g_d == 8)
    ok = ok and g_d
    print("[lp2b] GATE D: conical / bottom-incidence / circle-on-pure / "
          "non-commensurate / structured-grid-on-pure / tensor x2 / "
          "bad-engine guards ({}/8) -> {}".format(
              8 if g_d else "<8", "PASS" if g_d else "FAIL"), flush=True)

    # ---- GATE E: pure-engine per-layer absorption budget closure (audit C3) ----
    # measured at bring-up (n_modes=6 pure / degree=9 hybrid on the lossy 4x4 patch):
    # pure closes its Gram-flux-vs-far-field budget at 3.7e-15, hybrid at 1.0e-15;
    # A_independent 0.1198 (pure) vs 0.1203 (hybrid) -> cross 4.6e-4 (staggered-modal
    # vs Fourier-projected internal quadrature: two DIFFERENT bases, << the 1e-2 the
    # two-basis total is gated at); lossless twin 5.3e-15; off-switch A_indep None.
    d_abs = _lossy_patch_design(pol="x")
    r_pa = make_lumenairy_pmm2d_solver(engine="pure", n_modes=6, n_orders=3,
                                       absorption=True)(d_abs, None, {}, LAM,
                                                        1.0 + 0j, 1.5 + 0j)
    close_p = abs(r_pa.A_independent - r_pa.A)
    keys_ok = (r_pa.per_region_absorption is not None
               and set(r_pa.per_region_absorption) == {"spacer", "patch"})
    r_ll = make_lumenairy_pmm2d_solver(engine="pure", n_modes=6, n_orders=3,
                                       absorption=True)(_lossless_patch_design(),
                                                        None, {}, LAM,
                                                        1.0 + 0j, 1.5 + 0j)
    r_off = make_lumenairy_pmm2d_solver(engine="pure", n_modes=6, n_orders=3)(
        d_abs, None, {}, LAM, 1.0 + 0j, 1.5 + 0j)
    off_ok = bool(r_off.per_region_absorption is None
                  and r_off.A_independent is None)
    r_ha = make_lumenairy_pmm2d_solver(engine="hybrid", degree=9, n_orders=9,
                                       absorption=True)(d_abs, None, {}, LAM,
                                                        1.0 + 0j, 1.5 + 0j)
    close_h = abs(r_ha.A_independent - r_ha.A)
    cross = abs(r_pa.A_independent - r_ha.A_independent)
    g_e = bool(close_p < 1e-6 and keys_ok and r_pa.A_independent > 1e-3
               and abs(r_ll.A_independent) < 1e-6 and off_ok
               and close_h < 1e-6 and cross < 1e-2)
    ok = ok and g_e
    print("[lp2b] GATE E: pure per-layer absorption: closure |dA| = {:.1e} (< 1e-6), "
          "A_indep = {:.4f} keyed [{}], lossless A_indep = {:.1e} (< 1e-6), "
          "off-switch None {}; Hybrid-vs-Pure |dA_indep| = {:.1e} (< 1e-2, both "
          "close own budget: pure {:.1e} / hybrid {:.1e}) -> {}".format(
              close_p, r_pa.A_independent,
              "+".join(sorted(r_pa.per_region_absorption)) if keys_ok else "BADKEYS",
              r_ll.A_independent, "OK" if off_ok else "BAD", cross, close_p, close_h,
              "PASS" if g_e else "FAIL"), flush=True)

    print("[lp2b] *** LUMENAIRY PMM2D BRIDGE: {} ***".format(
        "PASS" if ok else "FAIL"), flush=True)
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
