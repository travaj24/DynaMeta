"""Workstream C of the 2026-11-02 "lumenairy bridge 5.50 + P3 audit" PR -- gates for:

  (1) RCWA ANALYTIC SHAPES (design_to_rcwa_stack(inclusion_model=...) + LumenairyStackSolver
      `shapes` slabs) vs an INDEPENDENT oracle: the same geometry painted as an AREA-AVERAGED
      eps_cell (16x16 sub-sampling per pixel, built here, not by either code path under test).
      Both are lumenairy's Laurent direct rule, so at fixed n_orders the raster converges to the
      analytic layer as O(S^-2); a 1 % radius or background error is the discrimination leg;
  (2) the PERIODIC raster (boundary-crossing inclusions wrap, as in the FEM mesher);
  (3) PMM inclusion WRAP-AROUND: shifting the cell origin must not change R/T (a clean oracle),
      cross-checked against the RCWA analytic path; clip-instead-of-wrap is the discrimination leg;
  (4) the REVERSE translation of analytic shapes / uniform tensors / uniform cells (round trips);
  (5) the BOR SEM radial basis: FD default byte-identical, SEM vs planar Fresnel per mode, SEM vs a
      converging FD ladder on a ring grating; and audit V-19 (make_bor_spec_solver + the deprecated
      make_lumenairy_bor_solver alias).
"""
import importlib.util
import warnings
from dataclasses import replace

import numpy as np
import pytest

HAVE_LUM = importlib.util.find_spec("lumenairy") is not None
pytestmark = pytest.mark.skipif(not HAVE_LUM, reason="lumenairy not installed")

LAM = 1.31e-6
P = 500e-9


def _reg():
    from dynameta.materials import ConstantOptical, Material, MaterialRegistry
    reg = MaterialRegistry()
    for n, e in (("air", 1.0), ("glass", 2.25), ("hi", complex(6.0, 0.4)), ("lo", 2.1),
                 ("mid", 3.2)):
        reg.add(Material(n, ConstantOptical(complex(e))))
    return reg


def _design(incs, *, bg="lo", theta=11.0, pol="y", px=P, py=P, thk=200e-9):
    from dynameta.geometry import Design, Layer, Stack, UnitCell
    from dynameta.geometry.specs import OpticalSpec
    return Design(name="c", unit_cell=UnitCell(period_x_m=px, period_y_m=py),
                  stack=Stack(layers=[Layer("g", thk, bg, inclusions=incs)],
                              superstrate_material="air", substrate_material="glass"),
                  electrodes=[], materials=_reg(),
                  optical=OpticalSpec(polarization=pol, incidence_angle_deg=theta))


def _solve(d, **kw):
    from dynameta.optics.lumenairy_bridge import make_lumenairy_rcwa_solver
    return make_lumenairy_rcwa_solver(**kw)(d, None, {}, LAM, 1.0 + 0j, 1.5 + 0j)


# ---------------------------------------------------------------------------------------------
# (1) analytic shapes vs the area-averaged raster ladder
# ---------------------------------------------------------------------------------------------

def _area_cell(S, contains, eps_in, eps_bg, k=16):
    xs = (np.arange(S * k) + 0.5) * P / (S * k)
    X, Y = np.meshgrid(xs, xs, indexing="ij")
    f = np.asarray(contains(X, Y), dtype=float).reshape(S, k, S, k).mean(axis=(1, 3))
    return (eps_bg + (eps_in - eps_bg) * f).astype(complex)


def test_analytic_disk_is_the_limit_of_the_area_raster_and_discriminates():
    """MEASURED 2026-11-02 (disk r = 160 nm off-centre in a 500 nm cell, eps 6+0.4i in 2.1, 200 nm,
    n_orders 5x5, theta 11 deg, y-pol): |dR| analytic vs area raster 2.8e-5 (S=128) -> 6.4e-6
    (S=256), ratio ~4.3 = O(S^-2); a 1 % radius error moves R by 1.9e-3, a 1 % background error by
    1.8e-3 -- each > 50x the S=256 agreement. The analytic slab enters through BOTH public routes
    (design_to_rcwa_stack 'analytic' and a LumenairyStackSolver `shapes` slab from
    layer_to_shapes_slab), which must agree to round-off."""
    from dynameta.core.layered import LayeredSlab, LayeredStack
    from dynameta.geometry import Inclusion
    from dynameta.geometry.cross_section import Circle
    from dynameta.optics.lumenairy_bridge import LumenairyStackSolver, layer_to_shapes_slab
    c0, rad, eh, eb = (237e-9, 251e-9), 160e-9, complex(6.0, 0.4), 2.1 + 0j

    def mk(r=rad, bgm="lo"):
        return _design([Inclusion(Circle(c0[0], c0[1], r), "hi")], bg=bgm)

    d = mk()
    ana = _solve(d, n_orders=5, inclusion_model="analytic")
    sol = LumenairyStackSolver(n_orders=5)

    def stk(slab):
        return LayeredStack(1.0 + 0j, 1.5 + 0j, [slab], period_x_m=P, period_y_m=P)
    via_slab = sol.solve(stk(layer_to_shapes_slab(d.stack.layers[0], d, LAM)), LAM, d.optical)
    assert via_slab.R == pytest.approx(ana.R, abs=1e-12)
    assert via_slab.r == pytest.approx(ana.r, abs=1e-12)

    def inside(X, Y, r=rad):
        return (X - c0[0]) ** 2 + (Y - c0[1]) ** 2 <= r ** 2
    errs = []
    for S in (128, 256):
        rr = sol.solve(stk(LayeredSlab(200e-9, eps_cell=_area_cell(S, inside, eh, eb))),
                       LAM, d.optical)
        errs.append(max(abs(rr.R - ana.R), abs(rr.T - ana.T), abs(rr.r - ana.r)))
    assert errs[1] < 2.0e-5, errs                       # measured 9.9e-6 (|dr|)
    assert errs[0] / errs[1] > 3.0, errs                # converging, O(S^-2) (measured ~3.9-4.3)
    # discrimination: the SAME comparison against a wrong radius / wrong background fails by > 50x
    wrong_r = _solve(mk(r=1.01 * rad), n_orders=5, inclusion_model="analytic")
    wrong_bg = sol.solve(stk(LayeredSlab(200e-9, shapes=layer_to_shapes_slab(
        d.stack.layers[0], d, LAM).shapes, eps_background=1.01 * eb)), LAM, d.optical)
    for w in (wrong_r, wrong_bg):
        assert abs(w.R - ana.R) > 50.0 * errs[1], (w.R, ana.R, errs)


def test_inclusion_model_auto_falls_back_and_analytic_refuses():
    """Polygons (no form factor) and OVERLAPPING inclusions (priority painting has no additive
    form-factor equivalent; lumenairy's own validator decides) rasterize under 'auto' and are
    refused under 'analytic' with the reason; 'auto' on a representable layer IS the analytic
    layer; absorption=True + 'analytic' is refused at construction (lumenairy 5.50 has no
    layer_absorption on shapes layers); a typo'd model is refused."""
    from dynameta.geometry import Inclusion
    from dynameta.geometry.cross_section import Circle, Ellipse, Rectangle, RegularPolygon
    from dynameta.optics.lumenairy_bridge import (ShapesNotRepresentable, design_to_rcwa_stack,
                                                  make_lumenairy_rcwa_solver)
    poly = _design([Inclusion(RegularPolygon(P / 2, P / 2, 150e-9, 6), "hi")])
    over = _design([Inclusion(Circle(200e-9, 250e-9, 100e-9), "hi"),
                    Inclusion(Circle(300e-9, 250e-9, 100e-9), "mid", priority=1)], bg="air")
    ok = _design([Inclusion(Rectangle(120e-9, 250e-9, 150e-9, 300e-9), "hi"),
                  Inclusion(Ellipse(360e-9, 250e-9, 80e-9, 160e-9), "mid")])
    for d, why in ((poly, "regular_polygon"), (over, "OVERLAP")):
        stk, _ = design_to_rcwa_stack(d, LAM, n_orders=3, inclusion_model="auto")
        assert [L.kind for L in stk.layers] == ["iso"]
        with pytest.raises(ShapesNotRepresentable, match=why):
            design_to_rcwa_stack(d, LAM, n_orders=3, inclusion_model="analytic")
    stk, _ = design_to_rcwa_stack(ok, LAM, n_orders=3, inclusion_model="auto")
    assert [L.kind for L in stk.layers] == ["shapes"]
    # the DEFAULT is 'auto' since v0.13.0 (user decision 2026-11-02): the representable layer is
    # the analytic one; 'raster' is the explicit opt-out
    stk_d, _ = design_to_rcwa_stack(ok, LAM, n_orders=3)
    assert [L.kind for L in stk_d.layers] == ["shapes"]
    stk_r, _ = design_to_rcwa_stack(ok, LAM, n_orders=3, inclusion_model="raster")
    assert [L.kind for L in stk_r.layers] == ["iso"]
    with pytest.raises(ValueError, match="layer_absorption"):
        make_lumenairy_rcwa_solver(absorption=True, inclusion_model="analytic")
    with pytest.raises(ValueError, match="inclusion_model"):
        design_to_rcwa_stack(ok, LAM, n_orders=3, inclusion_model="exact")
    # absorption=True + 'auto' rasterizes (and so gets its per-layer absorption)
    r = make_lumenairy_rcwa_solver(n_orders=3, absorption=True, inclusion_model="auto")(
        ok, None, {}, LAM, 1.0 + 0j, 1.5 + 0j)
    assert r.A_independent is not None and r.A_independent > 1e-3


def test_auto_is_formulation_aware_and_analytic_refuses_non_laurent():
    """lumenairy's shapes layers are LAURENT BY CONTRACT (RCWAStack.add_layer raises for
    formulation='li' with shapes=). 2026-11-02: under 'auto' a non-Laurent request ('li'/'fff')
    RASTERIZES the inclusion layer and the requested rule reaches lumenairy (it used to be
    silently replaced by Laurent on the shapes path); 'analytic' + 'li'/'fff' raises ValueError,
    in design_to_rcwa_stack AND at solver construction; 'auto' + 'laurent' stays analytic. The
    rasterized layer carries the requested formulation ('fff' is lumenairy's alias of 'li')."""
    from dynameta.geometry import Inclusion
    from dynameta.geometry.cross_section import Circle
    from dynameta.optics.lumenairy_bridge import design_to_rcwa_stack, make_lumenairy_rcwa_solver
    d = _design([Inclusion(Circle(P / 2, P / 2, 150e-9), "hi")])
    for form in ("fff", "li"):
        stk, _ = design_to_rcwa_stack(d, LAM, n_orders=3, inclusion_model="auto",
                                      formulation=form)
        assert [L.kind for L in stk.layers] == ["iso"]
        assert stk.layers[0].formulation == "li"
        with pytest.raises(ValueError, match="requires formulation='laurent'"):
            design_to_rcwa_stack(d, LAM, n_orders=3, inclusion_model="analytic",
                                 formulation=form)
        with pytest.raises(ValueError, match="requires formulation='laurent'"):
            make_lumenairy_rcwa_solver(inclusion_model="analytic", formulation=form)
    stk, _ = design_to_rcwa_stack(d, LAM, n_orders=3, inclusion_model="auto",
                                  formulation="laurent")
    assert [L.kind for L in stk.layers] == ["shapes"]
    # auto + fff is the SAME solve as an explicit raster + fff (bit for bit)
    a = _solve(d, n_orders=3, inclusion_model="auto", formulation="fff")
    r = _solve(d, n_orders=3, inclusion_model="raster", formulation="fff")
    assert (a.R, a.T, a.r) == (r.R, r.T, r.r)


def test_default_raster_lattice_does_not_jump_at_high_n_orders():
    """The default raster lattice is S = max(128, 8 N) (2026-11-02; was max(128, 4 N + 1)).
    Fixture = validation/lumenairy_pmm_bridge GATE B (600 nm period, 50 % metal lines eps -20+2i
    centred at P/2, 120 nm, air | glass 1.5, 1.31 um, normal incidence), TE (E_y along the lines,
    where Laurent is the valid rule), oracle = the 1-D PMM (degree 32: self-converged to 1e-10 in
    degree on this TE cell). MEASURED 2026-11-02 |R - PMM| at N = 16 / 32 / 48:
      new default S (128 / 256 / 384)      4.8e-5 / 1.7e-5 / 8.3e-6   (monotone)
      old lattice (128 / 129 / 193)        4.8e-5 / 4.6e-3 / 3.1e-3   (jumps ~100x at N = 32)
    The old jump is EDGE QUANTIZATION, not aliasing: the odd lattice S = 4 N + 1 puts the edges
    x = P/4, 3P/4 mid-pixel (realised duty 65/129 = 0.5039). cell_samples=128 is the documented way
    to reproduce the old lattice (explicit values keep the old formula); the default is
    byte-identical to it for N <= 16."""
    from dynameta.geometry import Design, Inclusion, Layer, Stack, UnitCell
    from dynameta.geometry.cross_section import Rectangle
    from dynameta.geometry.specs import OpticalSpec
    from dynameta.materials import ConstantOptical, Material, MaterialRegistry
    from dynameta.optics.lumenairy_bridge import make_lumenairy_rcwa_solver
    from dynameta.optics.lumenairy_bridge.pmm_backend import make_lumenairy_pmm_solver
    per, lam = 600e-9, 1.31e-6
    reg = MaterialRegistry()
    for n, e in (("air", 1.0), ("glass", 2.25), ("metal", complex(-20.0, 2.0))):
        reg.add(Material(n, ConstantOptical(complex(e))))
    lines = Inclusion(shape=Rectangle(per / 2, per / 2, 0.5 * per, per), material="metal")
    d = Design(name="gb", unit_cell=UnitCell.square(per),
               stack=Stack(layers=[Layer("grating", 120e-9, "air", inclusions=[lines])],
                           superstrate_material="air", substrate_material="glass"),
               electrodes=[], materials=reg, optical=OpticalSpec(polarization="y"))
    args = (d, None, {}, lam, 1.0 + 0j, 1.5 + 0j)
    ref = make_lumenairy_pmm_solver(degree=32, n_orders=21)(*args).R

    def err(n, **kw):
        return abs(make_lumenairy_rcwa_solver(n_orders=n, inclusion_model="raster", **kw)(
            *args).R - ref)
    new = [err(n) for n in (16, 32, 48)]
    old = [err(n, cell_samples=128) for n in (16, 32, 48)]
    assert new[0] > new[1] > new[2], new                   # measured 4.8e-5 > 1.7e-5 > 8.3e-6
    assert new[2] < 2e-5, new
    assert old[1] > 50.0 * new[1], (old, new)              # the old lattice's jump (4.6e-3)
    assert new[0] == old[0]                                # N <= 16: byte-identical lattices
    # the analytic path (the 'auto' default here) converges monotonically below the raster
    ana = [abs(make_lumenairy_rcwa_solver(n_orders=n)(*args).R - ref) for n in (16, 32, 48)]
    assert ana[0] > ana[1] > ana[2] and ana[2] < 3e-6, ana  # measured 3.5e-5 / 4.5e-6 / 1.3e-6


# ---------------------------------------------------------------------------------------------
# (2) periodic (wrap-aware) raster
# ---------------------------------------------------------------------------------------------

def test_periodic_raster_wraps_and_is_byte_identical_inside_the_cell():
    """A 0.4-period line centred ON x = 0 is the same periodic structure as one centred at P/2.
    MEASURED 2026-11-02 (n_orders 11, p-pol, theta 11 deg): the wrapped raster reproduces the
    centred one EXACTLY (symmetric cell-centred sampling) and the analytic path to its own
    translation invariance (~1e-14), while the OLD clip (= a 0.2-period line) gives R 0.0552 vs
    0.0951 -- the silent error this fixes. Interior inclusions paint byte-identically to the shared
    rasterizer (the FDTD's painter), so no existing result moves."""
    from dynameta.geometry import Inclusion
    from dynameta.geometry.cross_section import Circle, Rectangle
    from dynameta.optics.lumenairy_bridge._shapes import periodic_layer_eps_cell
    from dynameta.optics.rasterize import cell_axes, layer_eps_cell

    def line(cx, w=0.4 * P):
        return _design([Inclusion(Rectangle(cx, P / 2, w, P), "hi")], pol="p")
    centred = _solve(line(P / 2), n_orders=11)
    wrapped = _solve(line(0.0), n_orders=11)
    clipped = _solve(line(0.1 * P, w=0.2 * P), n_orders=11)
    assert wrapped.R == pytest.approx(centred.R, abs=1e-12)
    assert abs(clipped.R - centred.R) > 1e-2                     # what the clip used to solve
    a0 = _solve(line(P / 2), n_orders=11, inclusion_model="analytic")
    a1 = _solve(line(0.0), n_orders=11, inclusion_model="analytic")
    a2 = _solve(line(0.1 * P), n_orders=11, inclusion_model="analytic")
    assert max(abs(a1.R - a0.R), abs(a2.R - a0.R)) < 1e-12
    # byte identity for an interior inclusion (incl. one touching the cell edge)
    d = _design([Inclusion(Circle(0.3 * P, 0.4 * P, 0.2 * P), "hi"),
                 Inclusion(Rectangle(0.9 * P, 0.5 * P, 0.2 * P, 0.3 * P), "mid", priority=2)])
    x, y = cell_axes(64, 48, P, P)
    X, Y = np.meshgrid(x, y, indexing="ij")
    L = d.stack.layers[0]
    assert np.array_equal(periodic_layer_eps_cell(L, X, Y, LAM, d.materials, {}, P, P),
                          layer_eps_cell(L, X, Y, LAM, d.materials, {}))


# ---------------------------------------------------------------------------------------------
# (3) PMM wrap-around
# ---------------------------------------------------------------------------------------------

def _two_lines(xs, pol, px=600e-9):
    from dynameta.geometry import Inclusion
    from dynameta.geometry.cross_section import Rectangle
    return _design([Inclusion(Rectangle(xs[0], px / 2, 0.35 * px, px), "hi"),
                    Inclusion(Rectangle(xs[1], px / 2, 0.2 * px, px), "mid")],
                   bg="air", theta=17.0, pol=pol, px=px, py=px, thk=220e-9)


def test_pmm_wrap_is_origin_shift_invariant_and_matches_rcwa():
    """Shifting every line by the same dx (so one or both cross x = 0 / x = P and WRAP into two
    PMM segments) must not change R/T. MEASURED 2026-11-02 (600 nm cell, lines 0.35P of 6+0.4i and
    0.2P of 3.2, theta 17 deg, degree 16, 21 orders): TE spread 4e-11; TM spread R 8.3e-7 / T
    3.2e-6, which is the PMM DISCRETIZATION error, not a wrap error -- it shrinks with degree
    (24: 2.5e-7 / 1.0e-6; 32: 1.1e-7 / 4.6e-7) while T itself moves 5e-6 from 16 to 32 (the wall
    layout changes with the wrap); the gate asserts that shrinkage. TE agrees with the RCWA analytic
    path (41 orders) to 2.3e-7. Discrimination: clipping the wrapped line at the edge (the
    'rasterizer' reading) moves TM R by 8.5e-3; a line wider than the period is refused."""
    from dynameta.geometry.cross_section import Rectangle
    from dynameta.optics.lumenairy_bridge import make_lumenairy_pmm_solver
    px = 600e-9
    base = (0.3 * px, 0.7 * px)
    pmm = make_lumenairy_pmm_solver(degree=16, n_orders=21)

    def spread(pol, solver):
        Rs = np.asarray([[v.R, v.T] for v in (
            solver(_two_lines(tuple(x + dx for x in base), pol), None, {}, LAM, 1, 1.5)
            for dx in (0.0, -0.3 * px, 0.25 * px, 0.41 * px))])
        return float(np.ptp(Rs, axis=0).max())
    assert spread("y", pmm) < 1e-9
    tm16 = spread("p", pmm)
    tm24 = spread("p", make_lumenairy_pmm_solver(degree=24, n_orders=21))
    assert tm16 < 5e-6 and tm24 < 0.5 * tm16, (tm16, tm24)
    d_te = _two_lines((0.3 * px - 0.3 * px, 0.7 * px - 0.3 * px), "y")      # line 0 on x = 0
    r_p = pmm(d_te, None, {}, LAM, 1, 1.5)
    r_r = _solve(d_te, n_orders=41, inclusion_model="analytic")
    assert abs(r_p.R - r_r.R) < 1e-6 and abs(r_p.T - r_r.T) < 1e-6
    # discrimination: wrap vs clip of a line centred on x = 0
    d = _two_lines((0.3 * px, 0.0), "p")
    wrap = pmm(d, None, {}, LAM, 1, 1.5).R
    d.stack.layers[0].inclusions[1].shape = Rectangle(0.05 * px, px / 2, 0.1 * px, px)
    clip = pmm(d, None, {}, LAM, 1, 1.5).R
    assert abs(wrap - clip) > 1e-3
    d.stack.layers[0].inclusions[1].shape = Rectangle(0.0, px / 2, 1.2 * px, px)
    with pytest.raises(ValueError, match="exceeds the period"):
        pmm(d, None, {}, LAM, 1, 1.5)


# ---------------------------------------------------------------------------------------------
# (4) reverse translation
# ---------------------------------------------------------------------------------------------

def test_reverse_translation_of_shapes_tensor_and_uniform_cells_round_trips():
    """Design -> RCWAStack ('analytic') -> rcwa_stack_to_design must return the SAME CrossSections
    (exact field equality) and the same solve (to round-off); a uniform-tensor layer comes back as a
    tensor EpsField (return_eps_by_region=True) and solves identically; a constant eps_cell comes
    back as a uniform layer. Refusals keep their reasons: a patterned eps_cell (no faithful
    inverse), a tensor without return_eps_by_region (names the flag), a 1-D stack's y-varying shape.
    Discrimination: perturbing the reconstructed radius by 1 % moves R by 9.4e-5 (> 1e8 x the
    round-trip agreement)."""
    import lumenairy as lum
    from dynameta.core.eps_field import EpsField
    from dynameta.geometry import Inclusion, Layer
    from dynameta.geometry.cross_section import Circle, Ellipse, Rectangle
    from dynameta.optics.lumenairy_bridge import design_to_rcwa_stack, rcwa_stack_to_design
    d0 = _design([Inclusion(Circle(150e-9, 200e-9, 110e-9), "hi"),
                  Inclusion(Rectangle(0.0, 400e-9, 120e-9, 140e-9), "mid"),      # wraps in x
                  Inclusion(Ellipse(380e-9, 150e-9, 60e-9, 90e-9), "hi")])
    tens = np.diag([2.0, 3.1, 2.6]).astype(complex)
    d0.stack.layers.append(Layer("lc", 90e-9, "lo"))
    ebr = {"lc": EpsField(tensor=tens)}
    stk, names = design_to_rcwa_stack(d0, LAM, eps_by_region=ebr, n_orders=5,
                                      inclusion_model="analytic")
    assert [L.kind for L in stk.layers] == ["tensor", "shapes"]
    with pytest.raises(NotImplementedError, match="return_eps_by_region"):
        rcwa_stack_to_design(stk, layer_names=names)
    d1, ebr1 = rcwa_stack_to_design(stk, layer_names=names, return_eps_by_region=True)
    g0, g1 = d0.stack.layers[0], [L for L in d1.stack.layers if L.name == "g"][0]
    for a, b in zip(g0.inclusions, g1.inclusions):
        assert type(a.shape) is type(b.shape) and a.shape == b.shape
        assert d1.materials.get(b.material).eps(LAM) == d0.materials.get(a.material).eps(LAM)
    assert d1.materials.get(g1.background_material).eps(LAM) == 2.1
    assert np.array_equal(ebr1["lc"].tensor, tens)
    d1.optical = d0.optical
    kw = dict(n_orders=5, inclusion_model="analytic")
    from dynameta.optics.lumenairy_bridge import make_lumenairy_rcwa_solver
    s = make_lumenairy_rcwa_solver(**kw)
    r0 = s(d0, None, ebr, LAM, 1, 1.5)
    r1 = s(d1, None, ebr1, LAM, 1, 1.5)
    assert abs(r1.R - r0.R) < 1e-13 and abs(r1.r - r0.r) < 1e-13
    g1.inclusions[0].shape = Circle(150e-9, 200e-9, 1.01 * 110e-9)
    assert abs(s(d1, None, ebr1, LAM, 1, 1.5).R - r0.R) > 1e-5      # measured 9.4e-5
    # constant eps_cell -> uniform layer; patterned -> refused with the reason
    rs = lum.RCWAStack(P, period_y=P, n_orders=3)
    rs.add_layer(100e-9, eps_cell=np.full((13, 13), 2.5 + 0.1j))
    dd = rcwa_stack_to_design(rs)
    assert dd.materials.get(dd.stack.layers[0].background_material).eps(LAM) == 2.5 + 0.1j
    cell = np.full((13, 13), 2.0 + 0j)
    cell[:6, :] = 4.0
    rs.add_layer(100e-9, eps_cell=cell)
    with pytest.raises(NotImplementedError, match="no faithful inverse"):
        rcwa_stack_to_design(rs)
    # a 1-D stack's y-varying shape is solved as its y-average -> refused
    r1d = lum.RCWAStack(P, n_orders=5)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")                  # lumenairy's own y-average warning
        r1d.add_layer(100e-9, shapes=[{"shape": "disk", "radius": 100e-9, "eps": 4.0}],
                      eps_background=1.0)
    with pytest.raises(NotImplementedError, match="y-AVERAGE"):
        rcwa_stack_to_design(r1d)


# ---------------------------------------------------------------------------------------------
# (5) BOR SEM basis + audit V-19
# ---------------------------------------------------------------------------------------------

def _bor_spec(**kw):
    from dynameta.optics.lumenairy_bridge import BorLayer, BorStackSpec
    rings = BorLayer(thickness_m=0.4e-6, rings=(2.0e-6, 0.5, 2.0, 1.45), name="rings")
    return BorStackSpec(layers=[rings], azimuthal_order_m=1, r_max_m=8e-6, n_super=1.0,
                        n_sub=1.45, **kw)


def test_bor_fd_default_is_byte_identical_and_sem_is_gated():
    """basis='fd' (the default) forwards NOTHING new to lumenairy: R/T equal a BORStack built the
    pre-2026-11-02 way, bit for bit. 'sem' refuses a bare eps_profile (no walls to mesh on) and an
    unknown basis; a `segments` layer spelling the same annuli as the rings layer reproduces it on
    SEM (same walls -> same answer)."""
    from lumenairy.elements.bor import BORStack
    from dynameta.optics.lumenairy_bridge import BorLayer, solve_bor
    spec = _bor_spec(n_radial=96)
    got = solve_bor(spec, 1.0e-6)
    um = 1.0e6                                  # the bridge's m -> um scale, same float ops
    s = BORStack(Rbig=float(8e-6) * um, m=1, N=96, n_superstrate=1.0 + 0j, n_substrate=1.45 + 0j)
    s.add_layer(float(0.4e-6) * um, rings=(float(2.0e-6) * um, 0.5, 2.0 + 0j, 1.45 + 0j))
    s.set_source(k0=2.0 * np.pi / (float(1.0e-6) * um))
    raw = s.solve()
    order = np.argsort(np.asarray(raw["angles"]))
    assert np.array_equal(got.R, np.asarray(raw["R"], float)[order])
    assert np.array_equal(got.T, np.asarray(raw["T"], float)[order])
    with pytest.raises(ValueError, match="eps_profile"):
        replace(spec, basis="sem", layers=[BorLayer(0.4e-6, eps_profile=lambda r: 2.0 + 0 * r)])
    with pytest.raises(ValueError, match="basis"):
        replace(spec, basis="spectral")
    segs = []
    for k in range(4):
        segs += [((2 * k + 1) * 1.0e-6, 4.0), ((2 * k + 2) * 1.0e-6, 1.45 ** 2)]
    sem_r = solve_bor(replace(spec, basis="sem", degree=8), 1.0e-6)
    sem_s = solve_bor(replace(spec, basis="sem", degree=8,
                              layers=[BorLayer(0.4e-6, segments=segs, name="rings")]), 1.0e-6)
    assert np.allclose(sem_s.R, sem_r.R, rtol=0, atol=1e-12)


def test_bor_sem_vs_fresnel_and_vs_converging_fd_ladder():
    """(a) INDEPENDENT per-mode oracle: a uniform interface has no radial structure, so each
    incident mode reflects as a plane wave at its own polar angle -- SEM R[j] must land on the s-
    OR p-pol Fresnel curve (the validation/lumenairy_bor_bridge.py GATE B bracket). MEASURED
    2026-11-02 (n 1.0 | 1.9, r_max 12 um, degree 8, 47 modes): worst miss 2.3e-8, energy 1.2e-9;
    the SAME bracket against a 1 % wrong substrate index misses by 5.2e-3. (b) ring grating (4
    periods of 2 um, n 2.0 / 1.45, r_max 8 um, m = 1, 31 modes): SEM self-converges (degree 8 vs 10:
    ~2e-6; 10 vs 14: 1.3e-6) and the FD ladder CONVERGES TOWARD it -- worst per-mode |dR| 0.154 /
    0.059 / 0.019 at N = 64 / 128 / 256 (FD angles converge O(N^-2) under it: 0.10 / 0.030 / 0.008
    rad). Discrimination: a 1 % wrong ring index moves SEM by 2.3e-2, > 1e3 x its self-convergence."""
    from dynameta.optics.lumenairy_bridge import BorLayer, BorStackSpec, solve_bor
    n1, n2 = 1.0, 1.9
    uni = BorStackSpec(layers=[BorLayer(0.3e-6, eps=complex(n2 ** 2))], azimuthal_order_m=1,
                       r_max_m=12e-6, n_super=n1, n_sub=n2, basis="sem", degree=8)
    res = solve_bor(uni, 1.0e-6)

    def miss(n_sub):
        th = res.angles_rad
        c1 = np.cos(th)
        c2 = np.sqrt(1.0 - (n1 / n_sub * np.sin(th)) ** 2 + 0j)
        Rs = np.abs((n1 * c1 - n_sub * c2) / (n1 * c1 + n_sub * c2)) ** 2
        Rp = np.abs((n_sub * c1 - n1 * c2) / (n_sub * c1 + n1 * c2)) ** 2
        return float(np.minimum(np.abs(res.R - Rs), np.abs(res.R - Rp)).max())
    assert res.R.size >= 20 and miss(n2) < 1e-6, miss(n2)
    assert miss(1.01 * n2) > 1e-3
    assert np.abs(res.energy - 1.0).max() < 1e-7
    spec = _bor_spec()
    s8 = solve_bor(replace(spec, basis="sem", degree=8), 1.0e-6)
    ref = solve_bor(replace(spec, basis="sem", degree=10), 1.0e-6)
    self_conv = float(np.abs(s8.R - ref.R).max())
    assert 0.0 < self_conv < 1e-5, self_conv              # > 0: degree really reaches lumenairy
    errs = []
    for N in (64, 128, 256):
        fd = solve_bor(replace(spec, n_radial=N), 1.0e-6)
        n = min(fd.R.size, ref.R.size)
        errs.append(float(np.abs(fd.R[:n] - ref.R[:n]).max()))
    assert errs[0] > 2.0 * errs[1] > 4.0 * errs[2], errs
    assert errs[2] > 1e-3, errs        # FD-256 is NOT the reference: basis='sem' really reached it
    wrong = solve_bor(replace(spec, basis="sem", degree=10, layers=[
        BorLayer(0.4e-6, rings=(2.0e-6, 0.5, 2.02, 1.45))]), 1.0e-6)
    assert float(np.abs(wrong.R - ref.R).max()) > 1e3 * self_conv


def test_v19_bor_spec_solver_and_deprecated_alias():
    """audit V-19: make_bor_spec_solver is the honest name; make_lumenairy_bor_solver still works
    with its exact old behaviour (forced n_radial = 256) but warns; handing the returned callable
    the run_pipeline seam's arguments fails with a TypeError that NAMES the contract."""
    from dynameta.optics.lumenairy_bridge import (make_bor_spec_solver,
                                                  make_lumenairy_bor_solver, solve_bor)
    import inspect
    assert inspect.signature(make_lumenairy_bor_solver).parameters["n_radial"].default == 256
    spec = _bor_spec(n_radial=64)
    with pytest.warns(DeprecationWarning, match="V-19"):
        old = make_lumenairy_bor_solver(n_radial=96)
    a = old(spec, 1.0e-6)
    b = make_bor_spec_solver(n_radial=96)(spec, 1.0e-6)
    c = solve_bor(replace(spec, n_radial=96), 1.0e-6)
    assert np.array_equal(a.R, b.R) and np.array_equal(b.R, c.R)
    own = make_bor_spec_solver()(spec, 1.0e-6)              # None = keep the spec's own n_radial
    assert np.array_equal(own.R, solve_bor(spec, 1.0e-6).R)
    with pytest.raises(TypeError, match="BorStackSpec"):
        make_bor_spec_solver()(object(), None, {}, 1.0e-6, 1.0, 1.0)
    with pytest.raises(TypeError, match="NOT a run_pipeline optical_solver"):
        make_bor_spec_solver()(object(), 1.0e-6)
    assert make_bor_spec_solver().seam == "bor_spec"
