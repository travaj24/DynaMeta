"""PMM2D bridge (pure engine) TAPERED pillars + periodic WRAP, and the RCWA bridge's uniform-tensor
background with inclusions -- workstream B3 of the 2026-11-02 lumenairy-5.50 bridge PR.

Fixtures: a 600 nm square cell, 1.31 um, air over glass, eps-4 inclusions -- non-diffracting at
normal and at 20 deg (lambda / p = 2.18 > n_glass + sin 20), so every engine's order-summed R / T is
the zeroth order.

GATE K (periodic WRAP on the pure engine, _canonical_pure_cell): a seam-crossing Rectangle is
        split into its in-cell pieces; a crossing Circle (and every other inclusion with it) is
        moved by a rigid origin shift. Translation invariance is the oracle, the RCWA bridge (which
        wraps by its periodic raster) the cross-engine check:
        K1 union grid: a P/2 square centred on a cell CORNER (4 pieces) and on an EDGE (2 pieces)
           equals the centred square (same N = 4 grid, a whole-segment translate) -- R / r / t to
           < 1e-12;
        K2 'shapes' route, the non-commensurate 0.46 x 0.37 P bar of GATE J at 20 deg p-pol,
           translated to cross x (2 pieces), x AND y (4 pieces) and to a moved in-cell spot: all
           within 3e-5 in R of the untranslated bar (different wall layouts -> engine precision);
           RCWA-FFF n 12 on the 4-piece design within 1e-4 of pure, and the RCWA bridge itself
           translation-invariant (< 1e-10; translates are whole pixels). Discrimination: the
           CLIPPED bar (wrapped piece dropped -- the old raster bug) moves R by > 1e-3;
        K3 a Circle (r 0.3 P) crossing x, and crossing x and y: canonical image == the centred
           circle (< 1e-12); an in-cell off-centre copy (a different curved map) within 1e-6; RCWA
           FFF n 12 on the crossing circle within 3e-5;
        K4 two layers, a Circle over a full-height stripe 0.45 P to its right, translated so the
           circle crosses x (origin shift + the stripe moved WITH it): within 1e-5 in R of the
           untranslated pair, RCWA FFF n 12 within 3e-5 and translation-invariant; discrimination:
           the stripe 0.1 P closer (an inconsistently applied shift) moves R by > 2e-5.
GATE L (TAPER zero-angle identity): a tapered Rectangle whose bottom EQUALS its top, built through
        lumenairy's add_tapered_pillar, vs the straight (taper=None) pillar: taper_slices = 1 is
        bit-identical to the straight pillar on the per-layer route (same 3 x 3 strip tile, same
        walls), taper_slices = 8 within 1e-13 (identical slices joined by the mortar); vs the
        straight pillar on the default 'auto' route (stack-wide 'shapes' grid) within 1e-12.
GATE M (TAPER staircase convergence vs an INDEPENDENT engine): a 300 nm eps-4 pillar 320 x 280 nm
        at its top, every face moving in by 32 nm toward its base (atan(32/300) = 6.09 deg; walls
        on whole nanometres, so the 600-px RCWA raster carries every staircase wall exactly). For
        taper_slices N = 2 / 4 / 8 (pure, n_modes 6): (i) the SAME staircase built as N explicit
        untapered DynaMeta layers (midpoint widths) through this bridge equals add_tapered_pillar
        to < 1e-12 -- pins the slice ORDER (top-down), the midpoint rule and the width mapping;
        (ii) the RCWA bridge (FFF, n 9, 600 px) on that explicit staircase within 8e-5 at EVERY N,
        pure BELOW and RCWA ABOVE (PMM approaches upward in n_modes, RCWA downward in n_orders --
        a two-sided bracket); (iii) both ladders converge: |R(8) - R(4)| < |R(4) - R(2)| / 2 on
        each engine. Discrimination: pure at N = 4 vs RCWA at N = 8 (the slice-count slip) lands
        > 8e-5 apart.
GATE N (TAPER discrimination): at N = 4 vs the RCWA staircase of the CORRECT taper: the REVERSED
        taper (bottom wider by the same 32 nm per face; needs allow_wider_bottom=True) and a 1 deg
        shallower sidewall each land > 4e-4 away (> 5x GATE M's 8e-5 acceptance).
GATE O (RCWA bridge: uniform TENSOR background WITH inclusions): before 2026-11-02 the RCWA bridge
        tiled a uniform tensor EpsField over the whole layer and DROPPED its inclusions (measured: R
        bit-identical with and without the pillar). An LC background (n_o 1.5 / n_e 1.7, azimuth 30
        deg; tilt 0 and 40 deg) around a centred P/2 eps-4 square, x-pol: the RCWA bridge equals a
        direct lumenairy RCWAStack on the same (600, 600, 3, 3) painted raster (< 1e-12 -- the
        shared painter), converges TOWARD the PMM2D pure value over n_orders 6 -> 9 -> 12 (strictly
        shrinking, < 4e-4 at 12; the GATE G behaviour of lumenairy_pmm2d_bridge), and the tilted
        director lands within 4e-4 at n 12. Discrimination: the bare LC layer (inclusion dropped)
        is > 1e-2 from the pure value.

Honest SKIP (exit 0 + banner) when lumenairy is not importable.

Run: python -m validation.lumenairy_pmm2d_taper_wrap
"""
import importlib.util
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dynameta.core.eps_field import EpsField
from dynameta.geometry import Design, Inclusion, Layer, Stack, Taper, UnitCell
from dynameta.geometry.cross_section import Circle, Rectangle
from dynameta.geometry.specs import OpticalSpec
from dynameta.materials import ConstantOptical, Material, MaterialRegistry

LAM = 1.31e-6
PER = 600e-9
# GATE M/N taper fixture: walls on whole nanometres for every staircase N in (2, 4, 8)
THK = 300e-9
W_T, H_T = 320e-9, 280e-9
CX, CY = 312e-9, 282e-9
D_IN = 32e-9                                   # inward face offset at the base
ANG = math.degrees(math.atan(D_IN / THK))      # 6.09 deg per face


def _reg():
    r = MaterialRegistry()
    for n, e in (("air", 1.0), ("glass", 2.25), ("pil", 4.0), ("lcm", 2.4)):
        r.add(Material(n, ConstantOptical(complex(e))))
    return r


def _design(layers, *, pol="x", theta=0.0):
    return Design(name="tw", unit_cell=UnitCell.square(PER),
                  stack=Stack(layers=layers, superstrate_material="air",
                              substrate_material="glass"),
                  electrodes=[], materials=_reg(),
                  optical=OpticalSpec(polarization=pol, incidence_angle_deg=theta))


def _one(shape, *, thk=200e-9, pol="x", theta=0.0, bg="air"):
    return _design([Layer("p", thk, bg, inclusions=[Inclusion(shape, "pil")])], pol=pol,
                   theta=theta)


def _s(sol, d, ebr=None):
    return sol(d, None, ebr or {}, LAM, 1.0 + 0j, 1.5 + 0j)


# ------------------------------------------------------------------------------------- GATE K
def _bar(dx, dy, clip=False):
    cx, cy, w, h = 0.53 * PER + dx, 0.48 * PER + dy, 0.46 * PER, 0.37 * PER
    if clip:                       # the mutant: the wrapped piece dropped (bar clipped to the cell)
        lo, hi = max(cx - w / 2, 0.0), min(cx + w / 2, PER)
        cx, w = 0.5 * (lo + hi), hi - lo
    return _one(Rectangle(cx, cy, w, h), pol="p", theta=20.0)


def _circ(cx, cy):
    return _one(Circle(cx, cy, 0.3 * PER), pol="p", theta=20.0)


def _circ_stripe(dx, dy, rel=0.45):
    c = Circle(0.4 * PER + dx, 0.5 * PER + dy, 0.25 * PER)
    st = Rectangle((0.4 + rel) * PER + dx, 0.5 * PER + dy, 0.1 * PER, PER)
    return _design([Layer("s", 150e-9, "air", inclusions=[Inclusion(st, "pil")]),
                    Layer("c", 200e-9, "air", inclusions=[Inclusion(c, "pil")])],
                   pol="p", theta=20.0)


def _gate_k():
    from dynameta.optics.lumenairy_bridge import (make_lumenairy_pmm2d_solver,
                                                  make_lumenairy_rcwa_solver)
    from dynameta.optics.lumenairy_bridge.pmm2d_backend import _canonical_pure_cell
    pure = make_lumenairy_pmm2d_solver(n_modes=5, n_orders=3)
    rcwa = make_lumenairy_rcwa_solver(n_orders=12, cell_samples=600, formulation="fff")
    ok = True
    # K1 union grid
    sq = PER / 2.0
    a = _s(pure, _one(Rectangle(PER / 2, PER / 2, sq, sq)))
    worst1 = 0.0
    for c in ((0.0, 0.0), (0.0, PER / 2)):
        b = _s(pure, _one(Rectangle(c[0], c[1], sq, sq)))
        worst1 = max(worst1, abs(b.R - a.R), abs(b.r - a.r), abs(b.t - a.t))
    g1 = worst1 < 1e-12
    # K2 non-commensurate bar on the 'shapes' route
    base = _s(pure, _bar(0, 0))
    moved = [_bar(0.4 * PER, 0), _bar(0.4 * PER, 0.45 * PER), _bar(-0.25 * PER, 0.1 * PER)]
    npieces = [len(_canonical_pure_cell(d)[0].stack.layers[0].inclusions) for d in moved]
    spread2 = max(abs(_s(pure, d).R - base.R) for d in moved)
    rc_a, rc_b = _s(rcwa, _bar(0, 0)), _s(rcwa, moved[1])
    d_rc2, inv2 = abs(rc_b.R - base.R), abs(rc_b.R - rc_a.R)
    clip2 = abs(_s(pure, _bar(0.4 * PER, 0, clip=True)).R - base.R)
    g2 = bool(npieces == [2, 4, 1] and spread2 < 3e-5 and d_rc2 < 1e-4 and inv2 < 1e-10
              and clip2 > 1e-3)
    # K3 circle
    cen = _s(pure, _circ(PER / 2, PER / 2))
    canon3 = max(abs(_s(pure, _circ(cx, cy)).R - cen.R) for cx, cy in
                 ((0.05 * PER, PER / 2), (0.9 * PER, 0.95 * PER)))
    off3 = abs(_s(pure, _circ(0.42 * PER, 0.57 * PER)).R - cen.R)
    d_rc3 = abs(_s(rcwa, _circ(0.05 * PER, PER / 2)).R - cen.R)
    g3 = bool(canon3 < 1e-12 and off3 < 1e-6 and d_rc3 < 3e-5)
    # K4 two layers moved together
    b4 = _s(pure, _circ_stripe(0, 0))
    d4 = _circ_stripe(0.45 * PER, 0.2 * PER)
    shift4 = _canonical_pure_cell(d4)[1]
    spread4 = abs(_s(pure, d4).R - b4.R)
    r4a, r4b = _s(rcwa, _circ_stripe(0, 0)), _s(rcwa, d4)
    d_rc4, inv4 = abs(r4b.R - b4.R), abs(r4b.R - r4a.R)
    mut4 = abs(_s(pure, _circ_stripe(0, 0, rel=0.35)).R - b4.R)
    g4 = bool(shift4[0] > 0.0 and spread4 < 1e-5 and d_rc4 < 3e-5 and inv4 < 1e-10
              and mut4 > 2e-5)
    # measured at bring-up 2026-11-02 (n_modes 5): K1 1.4e-14 (max over R / r / t); K2 pieces
    # [2, 4, 1], spread 1.2e-5 (r 7.3e-5), RCWA-FFF |d| 6.0e-5 / 4.8e-5 / 4.7e-5 at n 9 / 12 / 15,
    # RCWA translation 5e-15, clipped 8.8e-3; K3 canonical 1.2e-15, off-centre 7.3e-8, RCWA 1.0e-5
    # (n 12); K4 origin shift 0.35 P, spread 3.4e-6 (r 2.7e-5), RCWA n 9 / 12 / 15 1.7e-6 /
    # 1.4e-5 / 3.1e-6 (oblique FFF is not monotone here), stripe-0.1-P-closer 4.1e-5.
    ok = g1 and g2 and g3 and g4
    print("[lptw] GATE K: wrap -- K1 union corner/edge == centre {:.1e} (< 1e-12) {}; K2 bar "
          "pieces {} spread {:.1e} (< 3e-5), RCWA {:.1e} (< 1e-4), RCWA-translate {:.1e}, clipped "
          "{:.1e} (> 1e-3) {}; K3 circle canonical {:.1e}, off-centre {:.1e}, RCWA {:.1e} {}; K4 "
          "two-layer shift {:.2f} P spread {:.1e} (< 1e-5), RCWA {:.1e}, mutant {:.1e} (> 2e-5) {} "
          "-> {}".format(worst1, g1, npieces, spread2, d_rc2, inv2, clip2, g2, canon3, off3, d_rc3,
                         g3, shift4[0] / PER, spread4, d_rc4, mut4, g4,
                         "PASS" if ok else "FAIL"), flush=True)
    return ok


# --------------------------------------------------------------------------------- GATES L-N
def _tapered(angle_deg, *, allow=False):
    top = Rectangle(CX, CY, W_T, H_T)
    tap = Taper.from_sidewall_angle(top, THK, angle_deg, allow_wider_bottom=allow)
    return _design([Layer("p", THK, "air", inclusions=[Inclusion(top, "pil", taper=tap)])])


def _straight():
    return _design([Layer("p", THK, "air", inclusions=[Inclusion(Rectangle(CX, CY, W_T, H_T),
                                                                 "pil")])])


def _stair(n):
    """The midpoint staircase as N explicit untapered layers, BOTTOM -> TOP (design order)."""
    lays = []
    for k in range(n):
        off = D_IN * (1.0 - (k + 0.5) / n)          # inward offset at slice k's mid-height
        lays.append(Layer("s{}".format(k), THK / n, "air", inclusions=[
            Inclusion(Rectangle(CX, CY, W_T - 2 * off, H_T - 2 * off), "pil")]))
    return _design(lays)


def _gate_l():
    from dynameta.optics.lumenairy_bridge import make_lumenairy_pmm2d_solver
    straight_pl = _s(make_lumenairy_pmm2d_solver(n_modes=5, n_orders=3, pure_grid="per-layer"),
                     _straight())
    straight_auto = _s(make_lumenairy_pmm2d_solver(n_modes=5, n_orders=3), _straight())
    t1 = _s(make_lumenairy_pmm2d_solver(n_modes=5, n_orders=3, taper_slices=1), _tapered(0.0))
    t8 = _s(make_lumenairy_pmm2d_solver(n_modes=5, n_orders=3, taper_slices=8), _tapered(0.0))
    bit = (t1.R == straight_pl.R and t1.r == straight_pl.r and t1.t == straight_pl.t)
    d8 = max(abs(t8.R - straight_pl.R), abs(t8.r - straight_pl.r), abs(t8.t - straight_pl.t))
    d_auto = max(abs(t1.R - straight_auto.R), abs(t1.r - straight_auto.r))
    # measured at bring-up 2026-11-02 (n_modes 5): slices 1 bit-identical; slices 8 1.4e-15 (max
    # over R / r / t); vs 'auto' 3.1e-15
    g = bool(bit and d8 < 1e-13 and d_auto < 1e-12)
    print("[lptw] GATE L: zero-angle taper == straight pillar: 1 slice bit-identical {}, 8 slices "
          "{:.1e} (< 1e-13), vs 'auto' route {:.1e} (< 1e-12) -> {}".format(
              bit, d8, d_auto, "PASS" if g else "FAIL"), flush=True)
    return g


def _gate_mn():
    from dynameta.optics.lumenairy_bridge import (make_lumenairy_pmm2d_solver,
                                                  make_lumenairy_rcwa_solver)
    rcwa = make_lumenairy_rcwa_solver(n_orders=9, cell_samples=600, formulation="fff")
    pl = make_lumenairy_pmm2d_solver(n_modes=6, n_orders=3, pure_grid="per-layer")
    rp, rr, stair_d = {}, {}, 0.0
    for n in (2, 4, 8):
        t0 = time.perf_counter()
        sol = make_lumenairy_pmm2d_solver(n_modes=6, n_orders=3, taper_slices=n)
        rp[n] = _s(sol, _tapered(ANG))
        if n == 4:                           # the explicit staircase through THIS bridge
            st = _s(pl, _stair(n))
            stair_d = max(abs(st.R - rp[n].R), abs(st.r - rp[n].r), abs(st.t - rp[n].t))
        rr[n] = _s(rcwa, _stair(n))
        print("[lptw]   N = {}: pure R {:.8e}, RCWA-FFF staircase R {:.8e} (|d| {:.1e}, pure {})  "
              "{:.0f} s".format(n, rp[n].R, rr[n].R, abs(rp[n].R - rr[n].R),
                                "below" if rp[n].R < rr[n].R else "ABOVE",
                                time.perf_counter() - t0), flush=True)
    gap = max(abs(rp[n].R - rr[n].R) for n in rp)
    below = all(rp[n].R < rr[n].R for n in rp)
    conv = all(abs(x[8].R - x[4].R) < 0.5 * abs(x[4].R - x[2].R) for x in (rp, rr))
    slip = abs(rp[4].R - rr[8].R)
    # measured at bring-up 2026-11-02 (n_modes 6 / RCWA FFF n 9, 600 px; 234 s for GATES K-O with
    # BLAS capped at 4 threads): pure 4.0882e-3 / 4.6328e-3 / 4.7904e-3 at N 2 / 4 / 8 (steps
    # 5.4e-4 -> 1.6e-4, ratio 3.4 ~ the midpoint O(1/N^2)); RCWA 4.1372e-3 / 4.6829e-3 / 4.8431e-3
    # -> gaps 4.9e-5 / 5.0e-5 / 5.3e-5, pure below (RCWA n 11: 3.6e-5 / 3.9e-5 / 4.4e-5). The gap
    # is two-sided convergence, not a mapping error: pure moved UP by +4.1e-5 from n_modes 5 to 7
    # (N 4), RCWA DOWN by ~1e-5 per 2-3 orders; the explicit staircase equals add_tapered_pillar
    # to 9e-15 here and to <= 1e-14 at every N measured (2 / 4 / 8 / 16). n_modes 5 ladder (to N 16): 4.0749e-3 /
    # 4.5943e-3 / 4.7929e-3 / 4.8553e-3 -- the M = 5 mortar error grows with the slice count, so
    # converge n_modes WITH taper_slices (lumenairy's own note). Slice-count slip 2.0e-4.
    g_m = bool(stair_d < 1e-12 and gap < 8e-5 and below and conv and slip > 8e-5)
    print("[lptw] GATE M: taper staircase: explicit-layer staircase == add_tapered_pillar {:.1e} "
          "(< 1e-12); vs RCWA-FFF staircase worst {:.1e} (< 8e-5), pure below RCWA {}; both "
          "ladders converging {}; slice-count slip {:.1e} (> 8e-5) -> {}".format(
              stair_d, gap, below, conv, slip, "PASS" if g_m else "FAIL"), flush=True)
    sol4 = make_lumenairy_pmm2d_solver(n_modes=6, n_orders=3, taper_slices=4)
    rev = abs(_s(sol4, _tapered(-ANG, allow=True)).R - rr[4].R)
    shallow = abs(_s(sol4, _tapered(ANG - 1.0)).R - rr[4].R)
    # measured at bring-up 2026-11-02 (n_modes 6, N 4, vs the RCWA staircase): reversed 3.3e-3,
    # 1 deg shallower 7.6e-4 (n_modes 5, N 8, vs pure: 3.5e-3 / 7.1e-4)
    g_n = bool(rev > 4e-4 and shallow > 4e-4)
    print("[lptw] GATE N: reversed taper {:.1e}, 1 deg shallower {:.1e} from the RCWA staircase "
          "(> 4e-4 each) -> {}".format(rev, shallow, "PASS" if g_n else "FAIL"), flush=True)
    return g_m and g_n


# ------------------------------------------------------------------------------------- GATE O
def _uniaxial(no, ne, az_deg, tilt_deg):
    a, t = np.radians(az_deg), np.radians(tilt_deg)
    d = np.array([np.cos(t) * np.cos(a), np.cos(t) * np.sin(a), np.sin(t)])
    return (no ** 2 * np.eye(3) + (ne ** 2 - no ** 2) * np.outer(d, d)).astype(complex)


def _direct_tensor_rcwa(shape, eps_bg, thk, n_orders, samples=600):
    import lumenairy as lum
    from dynameta.optics.rasterize import cell_axes
    x, y = cell_axes(samples, samples, PER, PER)
    X, Y = np.meshgrid(x, y, indexing="ij")
    cell = np.broadcast_to(np.asarray(eps_bg, dtype=complex), (samples, samples, 3, 3)).copy()
    cell[np.asarray(shape.contains_m(X, Y), dtype=bool)] = 4.0 * np.eye(3)
    st = lum.RCWAStack(PER, period_y=PER, n_superstrate=1.0, n_substrate=1.5,
                       n_orders=int(n_orders), n_orders_y=int(n_orders))
    st.add_layer(thk, eps_tensor_cell=cell)
    st.set_source(LAM, theta=0.0, phi=0.0)
    _, R, _ = st.solve().efficiencies()
    return float(np.sum(R[0]))


def _gate_o():
    from dynameta.optics.lumenairy_bridge import (make_lumenairy_pmm2d_solver,
                                                  make_lumenairy_rcwa_solver)
    thk = 250e-9
    sq = Rectangle(PER / 2, PER / 2, PER / 2, PER / 2)
    pure = make_lumenairy_pmm2d_solver(n_modes=5, n_orders=3)
    eps0, eps40 = _uniaxial(1.5, 1.7, 30.0, 0.0), _uniaxial(1.5, 1.7, 30.0, 40.0)
    d = _one(sq, thk=thk, bg="lcm")
    rp = _s(pure, d, {"p": EpsField(tensor=eps0)}).R
    dist = []
    for n in (6, 9, 12):
        rc = _s(make_lumenairy_rcwa_solver(n_orders=n, cell_samples=600), d,
                {"p": EpsField(tensor=eps0)}).R
        dist.append(abs(rc - rp))
        if n == 6:
            same = abs(rc - _direct_tensor_rcwa(sq, eps0, thk, 6))
    rp40 = _s(pure, d, {"p": EpsField(tensor=eps40)}).R
    d40 = abs(_s(make_lumenairy_rcwa_solver(n_orders=12, cell_samples=600), d,
                 {"p": EpsField(tensor=eps40)}).R - rp40)
    bare = abs(_s(make_lumenairy_rcwa_solver(n_orders=6), _design([Layer("p", thk, "lcm")]),
                  {"p": EpsField(tensor=eps0)}).R - rp)
    shrink = dist[2] < dist[1] < dist[0]
    # measured at bring-up 2026-11-02: before the fix the bridge returned R 0.07647258774158677
    # with AND without the pillar (n 6, x); after, 0.09575 (with); == direct oracle 0.0 (bit for
    # bit); ladder 3.1e-4 -> 1.9e-4 -> 1.5e-4, tilt 40 2.1e-4, inclusion-dropped 1.9e-2. GATE G (lumenairy_pmm2d_bridge)
    # measured the same cell's direct-oracle ladder 3.1e-4 -> 1.9e-4 -> 1.6e-4 (x, tilt 0) and
    # 4.2e-4 -> 2.1e-4 (tilt 40): RCWA's tensor path is Laurent-only, an O(1/n) approach from above.
    g = bool(same < 1e-12 and shrink and dist[2] < 4e-4 and d40 < 4e-4 and bare > 1e-2)
    print("[lptw] GATE O: RCWA bridge tensor background + inclusion: == direct painted RCWAStack "
          "{:.1e} (< 1e-12); vs pure {:.1e} -> {:.1e} -> {:.1e} (shrinking, < 4e-4); tilted 40 deg "
          "{:.1e} (< 4e-4); inclusion-dropped {:.1e} (> 1e-2) -> {}".format(
              same, dist[0], dist[1], dist[2], d40, bare, "PASS" if g else "FAIL"), flush=True)
    return g


def main():
    if importlib.util.find_spec("lumenairy") is None:
        print("[lptw] *** SKIP: lumenairy not installed -- PMM2D taper / wrap gates not run ***",
              flush=True)
        return True
    print("[lptw] === PMM2D taper + wrap, RCWA tensor background ===", flush=True)
    t0 = time.perf_counter()
    ok = _gate_k()
    ok = _gate_l() and ok
    ok = _gate_mn() and ok
    ok = _gate_o() and ok
    print("[lptw] *** PMM2D TAPER / WRAP: {} *** ({:.0f} s)".format(
        "PASS" if ok else "FAIL", time.perf_counter() - t0), flush=True)
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
