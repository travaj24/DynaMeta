"""PMM2D bridge (pure engine) TAPERED pillars + periodic WRAP, and the RCWA bridge's uniform-tensor
background with inclusions (2026-11-02, workstream B3 of the lumenairy-5.50 bridge PR).

Fast gates only (n_modes 4, small n_orders); the convergence ladders vs the RCWA bridge live in
validation/lumenairy_pmm2d_taper_wrap.py GATES K-O. Every physics test carries an oracle AND a
discrimination leg. The first tests are lumenairy-free (pure-Python translation helpers).

Run: python -m pytest -p no:benchmark -q tests/test_lumenairy_pmm2d_taper_wrap.py
"""
import importlib.util
import math

import numpy as np
import pytest

from dynameta.geometry import (Design, Inclusion, Layer, Stack, Taper, UnitCell,
                               UnsupportedGeometryError)
from dynameta.geometry.cross_section import Circle, Ellipse, Rectangle
from dynameta.geometry.specs import OpticalSpec
from dynameta.materials import ConstantOptical, Material, MaterialRegistry

HAVE_LUM = importlib.util.find_spec("lumenairy") is not None
needs_lum = pytest.mark.skipif(not HAVE_LUM, reason="lumenairy not installed")

def _needs(feature):
    """skipif below the bridge's MEASURED per-feature lumenairy floor (_common.FEATURE_FLOORS,
    wave-7 2026-11-02): the CI `floor` leg installs lumenairy 5.22.*, older than every 2026-11-02
    bridge extension, and the bridge refuses those paths there with LumenairyFeatureUnavailable."""
    from dynameta.optics.lumenairy_bridge._common import FEATURE_FLOORS, lumenairy_has
    floor, what = FEATURE_FLOORS[feature]
    return pytest.mark.skipif(not (HAVE_LUM and lumenairy_has(feature)),
                              reason="needs lumenairy >= {} ({})".format(
                                  ".".join(map(str, floor)), what))


LAM = 1.31e-6
PER = 600e-9
THK = 300e-9
W_T, H_T = 320e-9, 280e-9
CX, CY = 312e-9, 282e-9
ANG = math.degrees(math.atan(32e-9 / THK))       # 6.09 deg per face (32 nm in at the base)


def _reg():
    r = MaterialRegistry()
    for n, e in (("air", 1.0), ("glass", 2.25), ("pil", 4.0), ("lcm", 2.4), ("hi", 4.0 + 0.3j)):
        r.add(Material(n, ConstantOptical(complex(e))))
    return r


def _design(layers, *, pol="x", theta=0.0):
    return Design(name="tw", unit_cell=UnitCell.square(PER),
                  stack=Stack(layers=layers, superstrate_material="air", substrate_material="glass"),
                  electrodes=[], materials=_reg(),
                  optical=OpticalSpec(polarization=pol, incidence_angle_deg=theta))


def _one(shape, *, thk=200e-9, bg="air", mat="pil", taper=None):
    return _design([Layer("p", thk, bg, inclusions=[Inclusion(shape, mat, taper=taper)])])


def _tapered(angle, *, allow=False, mat="pil", cx=CX, cy=CY):
    top = Rectangle(cx, cy, W_T, H_T)
    return _one(top, thk=THK, mat=mat,
                taper=Taper.from_sidewall_angle(top, THK, angle, allow_wider_bottom=allow))


def _stair(angle, n):
    d = THK * math.tan(math.radians(angle))
    lays = []
    for k in range(n):                                # BOTTOM -> TOP, midpoint widths
        off = d * (1.0 - (k + 0.5) / n)
        lays.append(Layer("s{}".format(k), THK / n, "air", inclusions=[
            Inclusion(Rectangle(CX, CY, W_T - 2 * off, H_T - 2 * off), "pil")]))
    return _design(lays)


def _s(sol, d, ebr=None):
    return sol(d, None, ebr or {}, LAM, 1.0 + 0j, 1.5 + 0j)


def _uniaxial(no, ne, az_deg, tilt_deg):
    a, t = np.radians(az_deg), np.radians(tilt_deg)
    d = np.array([np.cos(t) * np.cos(a), np.cos(t) * np.sin(a), np.sin(t)])
    return (no ** 2 * np.eye(3) + (ne ** 2 - no ** 2) * np.outer(d, d)).astype(complex)


# ===================================================================== lumenairy-free helpers
def test_canonical_cell_split_shift_and_noop():
    from dynameta.optics.lumenairy_bridge.pmm2d_backend import _canonical_pure_cell, _wrap_intervals
    # an in-cell design is returned AS IS (byte-identical builds for every existing design)
    d = _one(Rectangle(PER / 2, PER / 2, PER / 2, PER / 2))
    dc, sh = _canonical_pure_cell(d)
    assert dc is d and sh == (0.0, 0.0)
    # a square centred on the cell corner -> 4 in-cell pieces covering the same area
    d = _one(Rectangle(0.0, 0.0, PER / 2, PER / 2))
    dc, sh = _canonical_pure_cell(d)
    pieces = dc.stack.layers[0].inclusions
    assert sh == (0.0, 0.0) and len(pieces) == 4
    assert all(p.material == "pil" for p in pieces)
    assert math.isclose(sum(p.shape.width_m * p.shape.height_m for p in pieces), (PER / 2) ** 2,
                        rel_tol=1e-12)
    for p in pieces:
        x0, x1, y0, y1 = p.shape.bbox_m()
        assert -1e-18 <= x0 and x1 <= PER + 1e-18 and -1e-18 <= y0 and y1 <= PER + 1e-18
    assert d.stack.layers[0].inclusions[0].shape.cx_m == 0.0          # input untouched
    # interval reduction: crossing, entirely outside, full period
    assert np.allclose(_wrap_intervals(-0.1 * PER, 0.2 * PER, PER), [(0.9 * PER, PER), (0, 0.2 * PER)])
    assert np.allclose(_wrap_intervals(1.1 * PER, 1.3 * PER, PER), [(0.1 * PER, 0.3 * PER)])
    assert _wrap_intervals(-0.2 * PER, 0.9 * PER, PER) == [(0.0, PER)]
    # a CIRCLE crossing x shifts the WHOLE origin: the circle lands inside, and the rectangle of the
    # other layer moves by the same vector (relative position preserved mod P)
    c = Circle(0.05 * PER, 0.5 * PER, 0.25 * PER)
    r = Rectangle(0.5 * PER, 0.5 * PER, 0.1 * PER, 0.2 * PER)
    d = _design([Layer("r", 1e-7, "air", inclusions=[Inclusion(r, "pil")]),
                 Layer("c", 1e-7, "air", inclusions=[Inclusion(c, "pil")])])
    dc, (sx, sy) = _canonical_pure_cell(d)
    assert sx > 0.0 and sy == 0.0
    cc = dc.stack.layers[1].inclusions[0].shape
    x0, x1, _, _ = cc.bbox_m()
    assert 1e-3 * PER <= x0 and x1 <= PER * (1 - 1e-3)
    # the seam goes to the MIDDLE of the gap the circle leaves, i.e. the circle lands centred --
    # pins the shift direction (a sign slip leaves it at 0.6 P: still inside, so check the spot)
    assert cc.cx_m == pytest.approx(PER / 2, abs=1e-15)
    rr = [p.shape for p in dc.stack.layers[0].inclusions]
    assert len(rr) == 1                                   # the bar stays whole after the shift
    rel = ((rr[0].cx_m - cc.cx_m) - (r.cx_m - c.cx_m)) % PER
    assert min(rel, PER - rel) < 1e-15                    # same relative offset, mod P
    # discrimination of that check: a shift applied to the circle ONLY breaks the offset by sx
    assert min(sx % PER, PER - sx % PER) > 1e-3 * PER
    # a tapered pillar crossing the seam is shifted (never split) with its taper bottom
    top = Rectangle(0.02 * PER, 0.5 * PER, 0.3 * PER, 0.3 * PER)
    d = _one(top, taper=Taper.from_sidewall_angle(top, 2e-7, 2.0))
    dc, (sx, _) = _canonical_pure_cell(d)
    inc = dc.stack.layers[0].inclusions[0]
    assert sx > 0.0 and len(dc.stack.layers[0].inclusions) == 1
    assert inc.taper.bottom.cx_m == pytest.approx(inc.shape.cx_m, abs=1e-15)
    assert 0.0 < inc.shape.bbox_m()[0] and inc.shape.bbox_m()[1] < PER
    # curved features that cover the whole period on an axis: no origin works -> precise refusal
    a, b = Circle(0.25 * PER, 0.5 * PER, 0.26 * PER), Ellipse(0.75 * PER, 0.3 * PER, 0.26 * PER, 0.1 * PER)
    d = _design([Layer("a", 1e-7, "air", inclusions=[Inclusion(a, "pil")]),
                 Layer("b", 1e-7, "air", inclusions=[Inclusion(b, "pil")])])
    with pytest.raises(ValueError, match="no lattice origin"):
        _canonical_pure_cell(d)


def test_taper_route_and_argument_guards():
    from dynameta.optics.lumenairy_bridge.pmm2d_backend import (_check_taper_args, pure_union_grid_n,
                                                                resolve_pure_grid)
    d = _tapered(ANG)
    assert resolve_pure_grid(d) == "per-layer"
    assert resolve_pure_grid(d, "per-layer") == "per-layer"
    for g in ("union", "shapes"):
        with pytest.raises(ValueError, match="TAPERED"):
            resolve_pure_grid(d, g)
    with pytest.raises(ValueError, match="TAPERED"):
        pure_union_grid_n(d)
    assert _check_taper_args(8, "midpoint") == (8, "midpoint")
    for bad in ((0, "midpoint"), (2.5, "midpoint"), (True, "midpoint"), (4, "top")):
        with pytest.raises(ValueError):
            _check_taper_args(*bad)
    # a seam-crossing bar under 'auto' is judged by its wrapped pieces (union grid N = 4 here)
    assert resolve_pure_grid(_one(Rectangle(0.0, 0.0, PER / 2, PER / 2))) == "union"
    assert pure_union_grid_n(_one(Rectangle(0.0, 0.0, PER / 2, PER / 2))) == 4


def test_layer_eps_tensor_cell_is_the_shared_periodic_painter():
    from dynameta.optics.rasterize import cell_axes, layer_eps_cell, layer_eps_tensor_cell
    eps_lc = _uniaxial(1.5, 1.7, 30.0, 40.0)
    lay = Layer("p", 1e-7, "lcm", inclusions=[
        Inclusion(Rectangle(0.0, PER / 2, 0.3 * PER, 0.2 * PER), "pil"),          # crosses x = 0
        Inclusion(Rectangle(0.0, PER / 2, 0.1 * PER, 0.1 * PER), "hi", priority=5)])
    X, Y = np.meshgrid(*cell_axes(60, 60, PER, PER), indexing="ij")
    reg = _reg()
    t = layer_eps_tensor_cell(lay, X, Y, LAM, reg, eps_lc, period_x_m=PER, period_y_m=PER)
    s = layer_eps_cell(lay, X, Y, LAM, reg, {}, period_x_m=PER, period_y_m=PER)
    assert t.shape == (60, 60, 3, 3)
    inc = s != complex(2.4)
    # inclusions: eps * I3 with exactly the scalar painter's values (priority + wrap identical)
    assert np.array_equal(t[inc], s[inc][:, None, None] * np.eye(3))
    assert np.allclose(t[~inc], eps_lc)                                     # background = tensor
    assert inc[0].any() and inc[-1].any()                                   # both wrapped sides
    top = Rectangle(PER / 2, PER / 2, 0.3 * PER, 0.3 * PER)
    lay_t = Layer("p", 1e-7, "air", inclusions=[
        Inclusion(top, "pil", taper=Taper.from_sidewall_angle(top, 1e-7, 2.0))])
    with pytest.raises(UnsupportedGeometryError, match="layer_eps_tensor_cell"):
        layer_eps_tensor_cell(lay_t, X, Y, LAM, reg, eps_lc)


# ============================================================================ lumenairy legs
@needs_lum
@_needs("pmm2d_pure_per_layer")
def test_taper_scope_refusals_and_build():
    from dynameta.core.eps_field import EpsField
    from dynameta.optics.lumenairy_bridge import design_to_pmm2d_stack
    st, names = design_to_pmm2d_stack(_tapered(ANG), LAM, n_modes=4, taper_slices=3)
    assert names == ["p"] * 3 and st.dynameta_origin_shift_m == (0.0, 0.0)
    # crossing the seam: shifted, not refused
    st, _ = design_to_pmm2d_stack(_tapered(ANG, cx=0.05 * PER), LAM, n_modes=4, taper_slices=2)
    assert st.dynameta_origin_shift_m[0] > 0.0
    circ = Circle(PER / 2, PER / 2, 0.2 * PER)
    with pytest.raises(ValueError, match="RECTANGLE only"):
        design_to_pmm2d_stack(_one(circ, taper=Taper.from_sidewall_angle(circ, 2e-7, 2.0)), LAM)
    with pytest.raises(ValueError, match="SCALAR eps_host"):
        design_to_pmm2d_stack(_tapered(ANG), LAM,
                              eps_by_region={"p": EpsField(tensor=_uniaxial(1.5, 1.7, 30, 0))})
    top = Rectangle(CX, CY, W_T, H_T)
    two = _design([Layer("p", THK, "air", inclusions=[
        Inclusion(top, "pil", taper=Taper.from_sidewall_angle(top, THK, ANG)),
        Inclusion(Rectangle(0.9 * PER, 0.9 * PER, 0.05 * PER, 0.05 * PER), "pil")])])
    with pytest.raises(ValueError, match="ONE pillar per layer"):
        design_to_pmm2d_stack(two, LAM)
    with pytest.raises(UnsupportedGeometryError, match="engine='hybrid'"):
        design_to_pmm2d_stack(_tapered(ANG), LAM, engine="hybrid")
    with pytest.raises(ValueError, match="TAPERED"):
        design_to_pmm2d_stack(_tapered(ANG), LAM, pure_grid="shapes")


@needs_lum
@_needs("pmm2d_pure_per_layer")
def test_taper_zero_angle_staircase_and_discrimination():
    # GATES L / M (i) / N (fast copies, n_modes 4): a zero-angle taper IS the straight pillar (1
    # slice: bit for bit); add_tapered_pillar IS the explicit midpoint staircase built from N
    # untapered DynaMeta layers (pins the top-down slice order and the width mapping); the
    # REVERSED taper and a 1 deg shallower one are far outside that agreement.
    from dynameta.optics.lumenairy_bridge import make_lumenairy_pmm2d_solver
    straight = _one(Rectangle(CX, CY, W_T, H_T), thk=THK)
    r0 = _s(make_lumenairy_pmm2d_solver(n_modes=4, n_orders=3, pure_grid="per-layer"), straight)
    t1 = _s(make_lumenairy_pmm2d_solver(n_modes=4, n_orders=3, taper_slices=1), _tapered(0.0))
    assert (t1.R, t1.r, t1.t) == (r0.R, r0.r, r0.t)
    sol2 = make_lumenairy_pmm2d_solver(n_modes=4, n_orders=3, taper_slices=2)
    t2 = _s(sol2, _tapered(ANG))
    st = _s(make_lumenairy_pmm2d_solver(n_modes=4, n_orders=3, pure_grid="per-layer"), _stair(ANG, 2))
    assert max(abs(t2.R - st.R), abs(t2.r - st.r), abs(t2.t - st.t)) < 1e-12
    # the taper is resolved: a 2-slice staircase already moves R off the straight pillar
    assert abs(t2.R - r0.R) > 1e-3
    rev = _s(sol2, _tapered(-ANG, allow=True))
    flipped = _s(make_lumenairy_pmm2d_solver(n_modes=4, n_orders=3, pure_grid="per-layer"),
                 _design(list(reversed(_stair(ANG, 2).stack.layers))))
    assert abs(rev.R - t2.R) > 1e-3
    assert abs(flipped.R - t2.R) > 1e-4               # the upside-down staircase is distinguishable
    assert abs(_s(sol2, _tapered(ANG - 1.0)).R - t2.R) > 1e-4


@needs_lum
@_needs("pmm2d_pure_per_layer")
def test_taper_absorption_keys_by_design_layer():
    from dynameta.optics.lumenairy_bridge import make_lumenairy_pmm2d_solver
    r = _s(make_lumenairy_pmm2d_solver(n_modes=4, n_orders=3, taper_slices=2, absorption=True),
           _tapered(ANG, mat="hi"))
    assert set(r.per_region_absorption) == {"p"}
    assert r.A_independent > 1e-3
    assert abs(r.A_independent - r.A) < 1e-6


@needs_lum
@_needs("pmm2d_pure_shapes")
def test_wrapped_square_equals_centred_square():
    # GATE K1 (fast copy): the union grid N = 4 is invariant under a half-period shift, so a
    # square centred on the corner (4 pieces) / an edge (2 pieces) IS the centred square.
    from dynameta.optics.lumenairy_bridge import make_lumenairy_pmm2d_solver
    sol = make_lumenairy_pmm2d_solver(n_modes=4, n_orders=3)
    a = _s(sol, _one(Rectangle(PER / 2, PER / 2, PER / 2, PER / 2)))
    for c in ((0.0, 0.0), (PER, PER / 2)):
        b = _s(sol, _one(Rectangle(c[0], c[1], PER / 2, PER / 2)))
        assert max(abs(b.R - a.R), abs(b.r - a.r), abs(b.t - a.t)) < 1e-12
    # discrimination: dropping the wrapped piece (the clipped corner quarter) is far off
    q = _s(sol, _one(Rectangle(PER / 8, PER / 8, PER / 4, PER / 4)))
    assert abs(q.R - a.R) > 1e-3


@needs_lum
def test_rcwa_tensor_background_keeps_inclusions():
    # GATE O (fast copy): the RCWA bridge used to tile a uniform tensor EpsField over the WHOLE
    # layer and drop its inclusions (R bit-identical with and without the pillar). Now the pillar is
    # painted over the tensor background (the shared layer_eps_tensor_cell), equal to a direct
    # RCWAStack on the same painted raster; analytic shapes refuse a tensor background.
    import lumenairy as lum
    from dynameta.core.eps_field import EpsField
    from dynameta.optics.lumenairy_bridge import ShapesNotRepresentable, make_lumenairy_rcwa_solver
    from dynameta.optics.rasterize import cell_axes
    eps_lc = _uniaxial(1.5, 1.7, 30.0, 0.0)
    ebr = {"p": EpsField(tensor=eps_lc)}
    sq = Rectangle(PER / 2, PER / 2, PER / 2, PER / 2)
    d = _one(sq, thk=250e-9, bg="lcm")
    sol = make_lumenairy_rcwa_solver(n_orders=4)
    with_p = _s(sol, d, ebr)
    bare = _s(sol, _design([Layer("p", 250e-9, "lcm")]), ebr)
    assert abs(with_p.R - bare.R) > 1e-2
    x, y = cell_axes(128, 128, PER, PER)
    X, Y = np.meshgrid(x, y, indexing="ij")
    cell = np.broadcast_to(eps_lc, (128, 128, 3, 3)).copy()
    cell[np.asarray(sq.contains_m(X, Y), dtype=bool)] = 4.0 * np.eye(3)
    st = lum.RCWAStack(PER, period_y=PER, n_superstrate=1.0, n_substrate=1.5, n_orders=4,
                       n_orders_y=4)
    st.add_layer(250e-9, eps_tensor_cell=cell)
    st.set_source(LAM, theta=0.0, phi=0.0)
    assert abs(float(np.sum(st.solve().efficiencies()[1][0])) - with_p.R) < 1e-12
    assert _s(make_lumenairy_rcwa_solver(n_orders=4, inclusion_model="auto"), d, ebr).R == with_p.R
    from dynameta.optics.lumenairy_bridge._common import LumenairyFeatureUnavailable, lumenairy_has
    if lumenairy_has("rcwa_analytic_shapes"):
        with pytest.raises(ShapesNotRepresentable, match="TENSOR background"):
            _s(make_lumenairy_rcwa_solver(n_orders=4, inclusion_model="analytic"), d, ebr)
    else:                                         # wave-7: refused at construction below 5.31
        with pytest.raises(LumenairyFeatureUnavailable, match=r"needs lumenairy >= 5\.31\.0"):
            make_lumenairy_rcwa_solver(n_orders=4, inclusion_model="analytic")
