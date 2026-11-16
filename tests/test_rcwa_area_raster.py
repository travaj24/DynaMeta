"""v0.16 RCWA bridge: the area-weighted raster on lumenairy's node lattice and coordinate-registered
gridded fields.

ORACLES: closed forms written here (the covered width of a rectangle, the harmonic / arithmetic
means of a two-material pixel, the exact mean of a piecewise-linear function over an interval),
lumenairy's analytic shapes (exact geometry; the S -> infinity limit of a correct raster), and the
same profile on different sample grids (a coordinate-blind reading moves r, a registered one does
not). On v0.15.1 the module dynameta.optics.lumenairy_bridge._raster does not exist and
raster_sampling is an unknown keyword: every gate here fails there; the measured v0.15.1 numbers
(point raster) are quoted in each docstring.
"""
import importlib.util
import pathlib
import re

import numpy as np
import pytest

HAVE_LUM = importlib.util.find_spec("lumenairy") is not None

from _strip_cell_fixture import (LAM, N_BG, P, T_FILM_NM, W, ebr, eps_film,  # noqa: E402
                                 li_companion_floor, strip_design)


def _needs(feature):
    from dynameta.optics.lumenairy_bridge._common import FEATURE_FLOORS, lumenairy_has
    floor, what = FEATURE_FLOORS[feature]
    return pytest.mark.skipif(not (HAVE_LUM and lumenairy_has(feature)),
                              reason="needs lumenairy >= {} ({})".format(
                                  ".".join(map(str, floor)), what))


needs_lum = pytest.mark.skipif(not HAVE_LUM, reason="lumenairy not installed")


def _layer(incs, bg="air", thk=40e-9):
    from dynameta.geometry import Layer
    return Layer("g", thk, bg, inclusions=incs)


def _reg():
    from _strip_cell_fixture import mats
    return mats()


# ---------------------------------------------------------------------------------------------
# exactness of the raster itself (pure numpy)
# ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("S", [128, 400, 2960])
def test_area_raster_realises_every_drawn_width_exactly(S):
    """Sum of the covered fraction x P / S == the drawn width to 1e-12 relative, for strips whose
    edges fall at arbitrary positions inside pixels, on a boundary, and across x = 0 / x = P
    (periodic wrap). The v0.15.1 point raster is off by up to half a pixel per edge: it realises
    the 150 nm strip as 150.495 nm on the non-aligned lattice S = 404 (152 pixels) -- the
    discrimination. TRAP documented here: on every lattice S = 8 N with 5 S / 16 an integer (S = 400,
    the v0.15.1 default at n_orders 50, among them) both edges of this cell fall on pixel
    boundaries and the point raster realises the width EXACTLY (150 pixels), so a point-raster
    error on such a lattice is not a width error."""
    from dynameta.geometry import Inclusion
    from dynameta.geometry.cross_section import Rectangle
    from dynameta.optics.lumenairy_bridge._raster import area_weighted_cell, realised_coverage
    from dynameta.optics.rasterize import cell_axes, layer_eps_cell
    reg = _reg()
    eau = complex(reg.get("metal").eps(LAM))
    rng = np.random.default_rng(S)
    cases = [(P / 2, W), (0.0, 0.3 * P), (P, 0.21 * P), (0.123456789 * P, 0.0987654321 * P)]
    cases += [(float(rng.uniform(0, P)), float(rng.uniform(0.01, 0.9) * P)) for _ in range(6)]
    for cx, w in cases:
        L = _layer([Inclusion(Rectangle(cx, P / 2, w, P), "metal")])
        cell, _ = area_weighted_cell(L, S, 1, LAM, reg, {}, P, P)
        width = realised_coverage(cell[:, 0], 1.0, eau).sum() * P / S
        assert abs(width - w) <= 1e-12 * w, (S, cx, w, width - w)
    if S == 400:
        L = _layer([Inclusion(Rectangle(P / 2, P / 2, W, P), "metal")])
        for s_pt, n_px in ((400, 150), (404, 152)):        # aligned (exact) / non-aligned
            x, y = cell_axes(s_pt, 1, P, P)
            X, Y = np.meshgrid(x, y, indexing="ij")
            pt = layer_eps_cell(L, X, Y, LAM, reg, {}, period_x_m=P, period_y_m=P)
            got = realised_coverage(pt[:, 0], 1.0, eau).sum() * P / s_pt
            assert abs(got - n_px * P / s_pt) < 1e-15, (s_pt, got)


def test_area_raster_2d_rectangles_wrap_and_overlap_are_exact():
    """2-D: the covered AREA of a rectangle crossing both cell edges equals w h to 1e-12; with two
    OVERLAPPING rectangles the later (higher priority) one paints over the earlier one exactly as
    the point painter's priority rule (the visible area of the lower one is its own area minus the
    overlap, computed here by hand)."""
    from dynameta.geometry import Inclusion
    from dynameta.geometry.cross_section import Rectangle
    from dynameta.optics.lumenairy_bridge._raster import area_weighted_cell
    reg = _reg()
    ea, ed = complex(reg.get("metal").eps(LAM)), complex(reg.get("diel").eps(LAM))
    S = 96
    A = Rectangle(0.05 * P, 0.97 * P, 0.31 * P, 0.27 * P)                 # wraps in x and y
    B = Rectangle(0.20 * P, 0.90 * P, 0.30 * P, 0.30 * P)                 # overlaps A
    L = _layer([Inclusion(A, "metal"), Inclusion(B, "diel", priority=1)])
    cell, _ = area_weighted_cell(L, S, S, LAM, reg, {}, P, P)
    # per-pixel fractions: solve cell = 1 + fa (ea - 1) + fd (ed - 1) for (fa, fd)
    M = np.array([[(ea - 1).real, (ed - 1).real], [(ea - 1).imag, (ed - 1).imag]])
    f = np.linalg.solve(M, np.stack([(cell - 1).real.ravel(), (cell - 1).imag.ravel()]))
    area_a, area_d = f[0].sum() * (P / S) ** 2, f[1].sum() * (P / S) ** 2
    ox = max(0.0, min(A.cx_m + A.width_m / 2, B.cx_m + B.width_m / 2)
             - max(A.cx_m - A.width_m / 2, B.cx_m - B.width_m / 2))
    oy = max(0.0, min(A.cy_m + A.height_m / 2, B.cy_m + B.height_m / 2)
             - max(A.cy_m - A.height_m / 2, B.cy_m - B.height_m / 2))
    assert abs(area_d - B.width_m * B.height_m) < 1e-12 * P * P
    assert abs(area_a - (A.width_m * A.height_m - ox * oy)) < 1e-12 * P * P


def test_inverse_rule_companion_is_the_harmonic_mean_across_the_wall():
    """For a pixel a vertical wall crosses at covered fraction f (metal) the companion is
    exx = 1 / (f / e_m + (1 - f) / e_d) (harmonic ACROSS the wall) and eyy = f e_m + (1 - f) e_d
    (arithmetic ALONG it) = eps_cell; for a horizontal wall the two swap. Closed forms, 1e-13."""
    from dynameta.geometry import Inclusion
    from dynameta.geometry.cross_section import Rectangle
    from dynameta.optics.lumenairy_bridge._raster import area_weighted_cell
    reg = _reg()
    em = complex(reg.get("metal").eps(LAM))
    S = 50
    h = P / S
    x0 = 10.3 * h + 0.5 * h            # the left wall 0.3 into pixel 11's cell [10.5 h, 11.5 h]
    R = Rectangle(x0 + 0.2 * P, P / 2, 0.4 * P, P)
    cell, (exx, eyy) = area_weighted_cell(_layer([Inclusion(R, "metal")]), S, 1, LAM, reg, {}, P, P,
                                          companion=True)
    f = 0.7                            # metal covers [x0, 11.5 h]: 0.7 of the pixel
    assert abs(cell[11, 0] - (f * em + (1 - f))) < 1e-13 * abs(em)
    assert abs(exx[11, 0] - 1.0 / (f / em + (1 - f))) < 1e-13
    assert abs(eyy[11, 0] - cell[11, 0]) < 1e-13 * abs(em)
    # 2-D: a horizontal wall -> exx arithmetic, eyy harmonic
    R2 = Rectangle(P / 2, x0 + 0.2 * P, P, 0.4 * P)
    c2, (exx2, eyy2) = area_weighted_cell(_layer([Inclusion(R2, "metal")]), S, S, LAM, reg, {}, P,
                                          P, companion=True)
    assert abs(exx2[7, 11] - (f * em + (1 - f))) < 1e-13 * abs(em)
    assert abs(eyy2[7, 11] - 1.0 / (f / em + (1 - f))) < 1e-13


def test_area_resample_is_the_exact_pixel_mean_of_the_linear_interpolant():
    """area_resample_cell: each node-lattice pixel takes the exact mean of the periodic
    piecewise-linear interpolant over [x_j - h/2, x_j + h/2] (closed-form trapezoids evaluated on
    a 4096-point refinement here) and the companion its exact harmonic mean (scipy quadrature);
    the three usual grid conventions of ONE interpolant (endpoint-inclusive, node, cell-centred)
    resample identically when they carry the same breakpoints."""
    from scipy.integrate import quad
    from dynameta.optics.lumenairy_bridge._raster import area_resample_cell
    xs = np.array([0.0, 50e-9, 120e-9, 200e-9, 300e-9])
    vs = np.array([2.0 + 0.1j, -3.0 + 1.0j, 4.0 + 0.2j, 1.0 + 0.0j, 2.5 + 0.3j])

    def f(x):
        x = np.mod(x, P)
        xx = np.concatenate([xs, [P]])
        vv = np.concatenate([vs, vs[:1]])
        return np.interp(x, xx, vv.real) + 1j * np.interp(x, xx, vv.imag)
    S = 37
    cell, (exx, _) = area_resample_cell(vs[:, None], xs, np.zeros(1), S, 1, P, P,
                                        companion=True)
    h = P / S
    for j in (0, 1, 13, 36):
        a, b = j * h - h / 2, j * h + h / 2
        mean = (quad(lambda x: f(x).real, a, b, limit=200, points=None)[0]
                + 1j * quad(lambda x: f(x).imag, a, b, limit=200)[0]) / h
        inv = (quad(lambda x: (1 / f(x)).real, a, b, limit=200)[0]
               + 1j * quad(lambda x: (1 / f(x)).imag, a, b, limit=200)[0]) / h
        assert abs(cell[j, 0] - mean) < 1e-12 * abs(mean), j
        assert abs(exx[j, 0] - 1 / inv) < 1e-9 * abs(1 / inv), j
    # grid conventions: the inclusive grid (duplicated periodic end point) == the same samples
    xi = np.concatenate([xs, [P]])
    vi = np.concatenate([vs, vs[:1]])
    ci, _ = area_resample_cell(vi[:, None], xi, np.zeros(1), S, 1, P, P)
    assert np.allclose(ci, cell, rtol=1e-14, atol=0)


# ---------------------------------------------------------------------------------------------
# through the bridge
# ---------------------------------------------------------------------------------------------

@needs_lum
@_needs("rcwa_shapes_li")
def test_default_raster_matches_the_analytic_li_layer_on_the_metal_strip():
    """The strip cell in TM with the strip RASTERIZED (inclusion_model='raster') under 'li', n_orders
    50, against lumenairy's analytic 'li' shapes layer (the exact geometry). MEASURED (5.50.2 and
    5.51.0, identical):
      * The area raster's error is a pure O(S^-2) lattice error: err S^2 = 1.0897e4 / 1.0885e4 /
        1.0877e4 / 1.0874e4 / 1.0871e4 / 1.0871e4 at S = 2960 / 4096 / 5920 / 8192 (the default
        lattice of a 1-D stack at this order) / 11840 / 16384 -- constant to 0.25 %. Other lattice
        families sit below that line (5.4e3 .. 6.8e3 at S = 8184 .. 8200).
      * The point raster realises the 150 nm strip either exactly or half a pixel off at each edge
        (the drawn edges are symmetric about the cell centre); where it is off, its error is the
        width error (|dr| / dw ~ 0.13 per nm on this resonant cell): area / point = 0.056 .. 0.061 at
        every non-aligned lattice S = 2956 .. 2988, against 1.0 / 2.0 on the aligned ones. At S ~ 400
        the lattice error itself (3.7e-2 at the aligned S = 400) dominates both rasters (point / area
        1.27 .. 1.70 where non-aligned), so the width comparison is made near S = 2960.
    Asserted: (1) the default lattice is within 2e-4 of the analytic layer -- it reads C / 8192^2 =
    1.620e-4 with the C = 1.087e4 measured above, the largest S^2 coefficient of any lattice measured;
    2e-4 = 1.23 C / 8192^2, a bar that a change of the raster moving C by more than 23 % fails;
    (2) the S^-2 law between S = 2960 and
    the default lattice, err(8192) 8192^2 / (err(2960) 2960^2) within 3 % of 1 (measured 0.998; a
    raster with a width error, or no area weighting, does not follow it); (3) on three non-aligned
    lattices (chosen by the point raster's own realised width, not by r) the area raster is under half
    the point raster's error (measured 0.06)."""
    from dynameta.optics.lumenairy_bridge import make_lumenairy_rcwa_solver as mk
    from dynameta.optics.lumenairy_bridge._raster import realised_coverage
    from dynameta.optics.rasterize import cell_axes, layer_eps_cell
    d = strip_design()
    reg = _reg()
    eau = complex(reg.get("metal").eps(LAM))

    def r_of(**kw):
        return mk(n_orders=50, formulation="li", **kw)(d, None, ebr(), LAM, None, None).r

    def point_width_error_px(S):
        x, y = cell_axes(S, 1, P, P)
        X, Y = np.meshgrid(x, y, indexing="ij")
        pt = layer_eps_cell(d.stack.layers[-1], X, Y, LAM, reg, {}, period_x_m=P, period_y_m=P)
        return (realised_coverage(pt[:, 0], 1.0, eau).sum() * P / S - W) / (P / S)
    ana = r_of()
    area = r_of(inclusion_model="raster")
    assert abs(area - ana) < 2e-4, abs(area - ana)              # measured 1.62e-4
    a2960 = r_of(inclusion_model="raster", cell_samples=2960)
    law = abs(area - ana) * 8192 ** 2 / (abs(a2960 - ana) * 2960 ** 2)
    assert abs(law - 1.0) < 0.03, law                           # measured 0.998
    for S in (2956, 2964, 2972):
        assert abs(abs(point_width_error_px(S)) - 0.5) < 1e-9, S        # half a pixel off
        a = abs(r_of(inclusion_model="raster", cell_samples=S) - ana)
        p = abs(r_of(inclusion_model="raster", raster_sampling="point", cell_samples=S) - ana)
        assert a < 0.5 * p, (S, a, p)                           # measured 0.056 .. 0.061

@needs_lum
@_needs("rcwa_analytic_shapes")
def test_analytic_disk_is_the_limit_of_the_bridge_area_raster():
    """2-D, Laurent (the rule both routes share here: a dielectric disk), the bridge's OWN area
    raster (inclusion_model='raster') against the analytic disk: |dR| falls as O(S^-2) with the
    lattice (the 16 x 16 sub-sampling of curved boundary pixels included). MEASURED (disk r =
    160 nm off-centre in a 500 nm cell, eps 6+0.4i in 2.1, 200 nm, n_orders 5, theta 11 deg,
    y-pol): S = 128 / 256 / 512: 2.7e-5 / 6.4e-6 / 1.7e-6 (ratios 4.3, 3.9). The v0.15.1 point
    raster on the same lattices: 6.0e-6 / 1.0e-4 / 1.1e-5 (Gauss-circle noise, erratic in S)."""
    from dynameta.geometry import Design, Inclusion, Layer, Stack, UnitCell
    from dynameta.geometry.cross_section import Circle
    from dynameta.geometry.specs import OpticalSpec
    from dynameta.materials import ConstantOptical, Material, MaterialRegistry
    from dynameta.optics.lumenairy_bridge import make_lumenairy_rcwa_solver as mk
    reg = MaterialRegistry()
    for n, e in (("air", 1.0), ("glass", 2.25), ("hi", complex(6.0, 0.4)), ("lo", 2.1)):
        reg.add(Material(n, ConstantOptical(complex(e))))
    pc = 500e-9
    d = Design(name="d", unit_cell=UnitCell(period_x_m=pc, period_y_m=pc),
               stack=Stack(layers=[Layer("g", 200e-9, "lo", inclusions=[
                   Inclusion(Circle(237e-9, 251e-9, 160e-9), "hi")])],
                   superstrate_material="air", substrate_material="glass"),
               electrodes=[], materials=reg,
               optical=OpticalSpec(polarization="y", incidence_angle_deg=11.0))
    args = (d, None, {}, 1.31e-6, 1.0, 1.5)
    ana = mk(n_orders=5, inclusion_model="analytic").__call__(*args).R
    errs = [abs(mk(n_orders=5, inclusion_model="raster", cell_samples=S)(*args).R - ana)
            for S in (128, 256, 512)]
    assert errs[2] < 4e-6 and errs[0] / errs[1] > 3.0 and errs[1] / errs[2] > 3.0, errs


def _lamellar_field(x_nm, nz=3):
    """A lateral carrier profile under the strip (raised density, 20 nm linear ramps centred 2.5 nm
    inside the left strip edge and 7.5 nm outside the right one: plateau [137.5, 272.5] nm), sampled
    at x_nm, z-invariant over the 6 nm film."""
    from dynameta.core.eps_field import EpsField
    ramp = np.clip((x_nm - 117.5) / 20.0, 0, 1) * np.clip((292.5 - x_nm) / 20.0, 0, 1)
    vals = np.asarray([eps_film(N_BG * (1.0 + 1.5 * r)) for r in ramp])
    return EpsField(x_axis_u=np.asarray(x_nm, float), y_axis_u=np.array([0.0]),
                    z_axis_u=np.linspace(0.0, T_FILM_NM, nz),
                    values_zyx=np.repeat(vals[None, None, :], nz, axis=0).astype(complex))


@needs_lum
def test_gridded_field_is_registered_by_its_own_coordinates():
    """Coordinate registration: one lamellar profile given on three x grids -- endpoint-inclusive linspace(0, P, 256)
    (what assemble_eps produces), the node lattice j P / 256, and cell centres (j + 1/2) P / 2960 --
    gives the same r at n_orders 50 ('li', strip analytic): MEASURED spread 2.3e-4 (the grids'
    own interpolation differences); v0.15.1 (values read as node samples whatever the axis) 1.9e-3
    on the same three fields (its endpoint-inclusive reading compresses the profile by 255/256 and
    shifts it against the strip).
    At n_orders 100 the 256-sample field no longer raises (v0.15.1: 'the unit-cell sampling (256,
    1) is too coarse for n_orders_x=100'): the field is resampled onto >= max(8 N, 8192) pixels."""
    from dynameta.optics.lumenairy_bridge import make_lumenairy_rcwa_solver as mk
    xp = P * 1e9
    grids = [np.linspace(0.0, xp, 256), np.arange(256) * xp / 256,
             (np.arange(2960) + 0.5) * xp / 2960]
    d = strip_design()
    rs, old = [], []
    for g in grids:
        e = {"film": _lamellar_field(g)}
        with li_companion_floor():                # below lumenairy 5.31 the bridge says the companion is missing
            rs.append(mk(n_orders=50, formulation="li")(d, None, e, LAM, None, None).r)
        old.append(mk(n_orders=50, formulation="li", raster_sampling="point")(
            d, None, e, LAM, None, None).r)
    spread = max(abs(a - b) for a in rs for b in rs)
    spread_old = max(abs(a - b) for a in old for b in old)
    assert spread < 2e-3, (spread, rs)
    assert spread_old > 3.0 * spread, (spread_old, spread)
    with li_companion_floor():
        r100 = mk(n_orders=100)(d, None, {"film": _lamellar_field(grids[0])}, LAM, None, None)
    assert np.isfinite(r100.R)
    with pytest.raises(Exception, match="too coarse"):
        mk(n_orders=100, raster_sampling="point")(d, None, {"film": _lamellar_field(grids[0])},
                                                  LAM, None, None)


@needs_lum
def test_gridded_field_moves_with_the_cell_translation_covariance():
    """Coordinate registration, sign-sensitive with an exact oracle: translating EVERYTHING in a periodic cell (the
    analytic strip AND the gridded film profile under it) by delta leaves r unchanged. The field
    is translated by its COORDINATES (samples rolled by k = 26 grid steps, i.e. delta = 26 P / 256
    = 40.625 nm, an exact translation of the interpolant), the strip by moving its centre. MEASURED
    ('li', n_orders 40): |dr| 8.6e-13 with the registered field; under v0.15.1's coordinate-blind
    reading the field's coordinates do not enter, so the film stays put while the strip moves and
    r changes by 0.077."""
    from dynameta.optics.lumenairy_bridge import make_lumenairy_rcwa_solver as mk
    xp = P * 1e9
    xs = np.arange(256) * xp / 256
    k = 26
    delta = k * P / 256
    f0 = _lamellar_field(xs)
    f1 = _lamellar_field(xs)
    f1.values_zyx = np.roll(f0.values_zyx, k, axis=2)        # f1(x) = f0(x - delta)
    with li_companion_floor():                    # below lumenairy 5.31 the bridge says the companion is missing
        r0 = mk(n_orders=40)(strip_design(), None, {"film": f0}, LAM, None, None).r
        r1 = mk(n_orders=40)(strip_design(cx=P / 2 + delta), None, {"film": f1}, LAM, None, None).r
    assert abs(r1 - r0) < 1e-6, abs(r1 - r0)
    # v0.15.1's reading: the axis is ignored -> the same comparison breaks by the misregistration
    p0 = mk(n_orders=40, raster_sampling="point")(strip_design(), None, {"film": f0}, LAM,
                                                   None, None).r
    pv = mk(n_orders=40, raster_sampling="point")(strip_design(cx=P / 2 + delta), None,
                                                   {"film": f0}, LAM, None, None).r
    assert abs(pv - p0) > 1e3 * max(abs(r1 - r0), 1e-9), abs(pv - p0)


@needs_lum
def test_metre_valued_x_axis_is_refused():
    """The lateral counterpart of the slab-thickness guard: x samples in metres read as nm span ~1e-9 of the
    period and are refused (periodic_samples' coverage guard, shared with the PMM bridge), where
    v0.15.1 ignored the coordinates entirely."""
    from dynameta.optics.lumenairy_bridge import make_lumenairy_rcwa_solver as mk
    f = _lamellar_field(np.linspace(0.0, P * 1e9, 64))
    f.x_axis_u = f.x_axis_u * 1e-9
    with pytest.raises(ValueError, match="METRES"):
        mk(n_orders=10)(strip_design(), None, {"film": f}, LAM, None, None)


def test_one_area_raster_kernel():
    """One implementation per numerical kernel: one implementation of the area raster and of the interval means. The raster
    module imports the interval means from _lateral_partition (identity), and no other module under
    dynameta/ defines the raster's kernels."""
    from dynameta.optics.lumenairy_bridge import _lateral_partition as lp
    from dynameta.optics.lumenairy_bridge import _raster
    assert _raster.interval_means is lp.interval_means
    assert _raster.periodic_samples is lp.periodic_samples
    root = pathlib.Path(__file__).resolve().parents[1] / "dynameta"
    names = ("area_weighted_cell", "area_resample_cell", "area_weighted_tensor_cell",
             "interval_means", "periodic_samples", "_axis_subcells")
    counts = {n: 0 for n in names}
    for f in root.rglob("*.py"):
        txt = f.read_text(encoding="cp1252", errors="replace")
        for n in names:
            counts[n] += len(re.findall("^def " + n + "[(]", txt, flags=re.M))
    assert counts == {n: 1 for n in names}, counts
