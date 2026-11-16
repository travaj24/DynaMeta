"""v0.16 RCWA bridge: the factorization -- formulation='auto'
default, explicit `formulation=` on every lumenairy call, the cache fingerprint, the face-flux
per-layer absorption of an 'li' solve, the v0.15.1 restore, and the slab-thickness guard at this call site.

ORACLES (independent of the code under test):
  * the 2-D TM finite-element solver optics.fem_2d_tm (H_y, corner-graded quadrilateral mesh, no
    Fourier factorization at all) on the SAME metal-strip cell -- the reference for every TM
    accuracy gate (its default setting is 2.8e-5 from its own converged value 0.335270+0.017192i);
  * the v0.15.1 construction rebuilt inline from lumenairy's public API (tests/_strip_cell_fixture
    .v0151_solve) for every bit-identity gate (same process, np.array_equal / ==);
  * Poynting's theorem for the per-layer absorption: the face-flux split must close on the
    far-field 1 - R - T (computed by lumenairy from the reflected / transmitted orders).

On v0.15.1 (the unfixed bridge) every gate that passes a new keyword fails with TypeError, and the
accuracy gate fails by its measured amount (default r 0.169 from the FEM at n_orders 100, 6.9x the
new default; 2.26e-2 at n_orders 200, above the 0.02 bar).
"""
import importlib.util
import warnings

import numpy as np
import pytest

HAVE_LUM = importlib.util.find_spec("lumenairy") is not None
HAVE_NGS = importlib.util.find_spec("ngsolve") is not None
pytestmark = pytest.mark.skipif(not HAVE_LUM, reason="lumenairy not installed")

from _strip_cell_fixture import (LAM, N_BG, P, R_FEM_CONVERGED, ebr, eps_film,  # noqa: E402
                                       fem_strip, graded_film_field, strip_design, v0151_solve)


def _needs(feature):
    from dynameta.optics.lumenairy_bridge._common import FEATURE_FLOORS, lumenairy_has
    floor, what = FEATURE_FLOORS[feature]
    return pytest.mark.skipif(not (HAVE_LUM and lumenairy_has(feature)),
                              reason="needs lumenairy >= {} ({})".format(
                                  ".".join(map(str, floor)), what))


def _solver(**kw):
    from dynameta.optics.lumenairy_bridge import make_lumenairy_rcwa_solver
    return make_lumenairy_rcwa_solver(**kw)


def _legacy(**kw):
    """The v0.15.1 settings, spelled out (formulation='laurent' warns on a TM metal cell)."""
    return _solver(formulation="laurent", raster_sampling="point", **kw)


# ---------------------------------------------------------------------------------------------
# accuracy: the default on the TM metal-strip cell vs the 2-D TM FEM
# ---------------------------------------------------------------------------------------------

@pytest.mark.skipif(not HAVE_NGS, reason="ngsolve (the FEM oracle) not installed")
@_needs("rcwa_shapes_li")
def test_default_converges_on_the_tm_metal_strip_against_the_fem():
    """Metal-strip grating, TM ('x'), normal incidence, default solver settings, against the 2-D TM
    FEM on the same cell. MEASURED (lumenairy 5.50.2 and 5.51.0: analytic shapes under Li's rule;
    FEM default order 4: 0.3352817+0.0171654i, 2.9e-5 from its converged value):
      |r - FEM| at n_orders 50 / 100 / 200:  default 7.04e-2 / 2.46e-2 / 8.91e-3
                                             v0.15.1 default (Laurent) 4.96e-1 / 1.69e-1 / 2.26e-2
    The cell is resonant and width-sensitive (|dr|/dw 0.129 per nm), so every bar below was checked
    over a NEIGHBOURHOOD of strip widths 130 / 140 / 150 / 160 / 170 nm (each against its own FEM):
      default n_orders 200:   2.4e-3 / 5.1e-3 / 8.9e-3 / 7.0e-3 / 3.7e-3   (accuracy bar 0.02)
      default 50 > 100 > 200: holds at every width                          (convergence)
      Laurent / default at n_orders 100: 9.5 / 9.8 / 6.9 / 7.2 / 5.6        (ratio bar 4)
      Laurent at n_orders 100: 0.074 / 0.077 / 0.169 / 0.076 / 0.052         (floor 0.035)
    The 0.02 accuracy bar does not DISCRIMINATE the rule (Laurent at n_orders 200 is under it at
    four of the five widths: 1.0e-2 / 1.7e-2 / 2.3e-2 / 1.4e-2 / 5.2e-3); the ratio at equal
    truncation does, and the floor (1.5x under the smallest neighbour) says the Laurent leg really
    is the slowly converging rule. The default resolves to 'li' and equals the explicit 'li' call bit
    for bit (same route)."""
    d = strip_design()
    r_fem = fem_strip().r
    assert abs(r_fem - R_FEM_CONVERGED) < 2e-4
    err = {}
    for n in (50, 100, 200):
        s = _solver(n_orders=n)
        r = s(d, None, ebr(), LAM, None, None).r
        assert s.last_formulation == "li"
        r_li = _solver(n_orders=n, formulation="li")(d, None, ebr(), LAM, None, None).r
        assert r == r_li
        err[n] = abs(r - r_fem)
    assert err[200] < 0.02, err                          # measured 8.91e-3
    assert err[50] > err[100] > err[200], err            # converging (7.0e-2 -> 2.5e-2 -> 8.9e-3)
    with pytest.warns(Warning, match="formulation='laurent'"):
        old = _legacy(n_orders=100)(d, None, ebr(), LAM, None, None).r
    assert abs(old - r_fem) >= 4.0 * err[100], (abs(old - r_fem), err)   # v0.15.1: 0.169 (6.9x)
    assert abs(old - r_fem) > 0.035, abs(old - r_fem)    # neighbourhood floor (0.052 .. 0.169)

@pytest.mark.skipif(not HAVE_NGS, reason="ngsolve (the FEM oracle) not installed")
@_needs("rcwa_shapes_li")
def test_default_with_a_z_graded_film_against_the_fem():
    """The same cell with a z-GRADED film (21 z samples, accumulation at the top face; a gridded,
    laterally uniform EpsField). The bridge slices it into 20 endpoint-averaged slabs
    (slice_layer_eps_field); the FEM gets the SAME 20 uniform slabs, so the comparison sees only
    the RCWA truncation. MEASURED (FEM -0.029799-0.097421i): |r - FEM| default 3.72e-2 / 1.26e-2
    at n_orders 50 / 100; v0.15.1 default 2.67e-1 / 9.00e-2. Over strip widths 130 / 140 / 150 /
    160 / 170 nm (each against its own FEM): default at n_orders 100 6.8e-3 / 5.6e-3 / 1.3e-2 /
    5.8e-3 / 6.0e-3 (bar 0.02) with 50 > 100 at every width; Laurent / default at n_orders 100 9.3 /
    9.2 / 7.1 / 6.9 / 6.3 (bar 4); Laurent at n_orders 100 0.063 / 0.052 / 0.090 / 0.040 / 0.038
    (floor 0.025, 1.5x under the smallest neighbour)."""
    from dynameta.core.layered import slice_eps_field
    d = strip_design()
    f = graded_film_field()
    slabs = slice_eps_field(f, 1e-9)
    r_fem = fem_strip(film=[(s.thickness_m, complex(s.eps)) for s in slabs]).r
    e = {n: abs(_solver(n_orders=n)(d, None, {"film": f}, LAM, None, None).r - r_fem)
         for n in (50, 100)}
    assert e[100] < 0.02 and e[50] > e[100], e
    with pytest.warns(Warning, match="formulation='laurent'"):
        old = _legacy(n_orders=100)(d, None, {"film": f}, LAM, None, None).r
    assert abs(old - r_fem) >= 4.0 * e[100], (old, r_fem, e)       # measured 7.1x
    assert abs(old - r_fem) > 0.025, (old, r_fem)                  # neighbourhood floor (0.038 .. 0.090)


# ---------------------------------------------------------------------------------------------
# bit-identity: TE, dielectric, uniform, and the explicit v0.15.1 restore
# ---------------------------------------------------------------------------------------------

def _same(a, b):
    return (a.R, a.T, a.r, a.t) == tuple(b)


@_needs("rcwa_analytic_shapes")
@pytest.mark.parametrize("n_orders", [11, 40])
def test_te_and_dielectric_defaults_are_bit_identical_to_v0151(n_orders):
    """TE (E along the strips) on the metal cell, and a low-contrast DIELECTRIC strip in TM and TE:
    'auto' resolves to 'laurent' and the analytic shapes layer is the v0.15.1 layer, so the
    DEFAULT is bit-identical to the v0.15.1 construction (inline rebuild, same process)."""
    for pol, mat in (("y", "metal"), ("x", "diel"), ("y", "diel")):
        d = strip_design(pol=pol, strip_material=mat)
        s = _solver(n_orders=n_orders)
        res = s(d, None, ebr(), LAM, None, None)
        assert s.last_formulation == "laurent"
        assert _same(res, v0151_solve(d, LAM, ebr(), n_orders=n_orders)), (pol, mat)


def test_explicit_laurent_point_restores_v0151_bit_for_bit():
    """formulation='laurent', raster_sampling='point' reproduces v0.15.1 on every dispatch route
    of a 1-D design (inline rebuild, same process): the analytic strip (TM metal -- it warns),
    the point raster (inclusion_model='raster', default and explicit cell_samples=128 lattices --
    the pre-0.13 promise), a z-graded laterally uniform film, and a laterally structured gridded
    film (values passed as they come)."""
    d = strip_design()
    ef = graded_film_field()
    xs = np.arange(64) * (P * 1e9 / 64)
    prof = np.asarray([eps_film(N_BG * (1.0 + 0.5 * np.cos(2 * np.pi * x / (P * 1e9)))) for x in xs])
    from dynameta.core.eps_field import EpsField
    lat = EpsField(x_axis_u=xs, y_axis_u=np.array([0.0]), z_axis_u=np.array([0.0, 3.0, 6.0]),
                   values_zyx=np.repeat(prof[None, None, :], 3, axis=0).astype(complex))
    cases = [(dict(), ebr()), (dict(inclusion_model="raster"), ebr()),
             (dict(inclusion_model="raster", cell_samples=128), ebr()),
             (dict(), {"film": ef}), (dict(), {"film": lat})]
    for kw, e in cases:
        with pytest.warns(Warning, match="formulation='laurent'"):
            res = _legacy(n_orders=12, **kw)(d, None, e, LAM, None, None)
        assert _same(res, v0151_solve(d, LAM, e, n_orders=12, **kw)), kw


def test_uniform_stack_is_bit_identical_and_laurent():
    """No patterned layer: 'auto' -> 'laurent', every layer a uniform one; equal to v0.15.1."""
    from dynameta.geometry import Design, Layer, Stack, UnitCell
    from dynameta.geometry.specs import OpticalSpec
    from _strip_cell_fixture import FILM, mats
    d = Design(name="flat", unit_cell=UnitCell(period_x_m=P, period_y_m=P),
               stack=Stack(layers=[Layer(n, t, m) for n, t, m in FILM],
                           superstrate_material="air", substrate_material="glass"),
               electrodes=[], materials=mats(), optical=OpticalSpec(polarization="x"))
    s = _solver(n_orders=5)
    assert _same(s(d, None, ebr(), LAM, None, None), v0151_solve(d, LAM, ebr(), n_orders=5))
    assert s.last_formulation == "laurent"


# ---------------------------------------------------------------------------------------------
# the rule, the record, the warnings
# ---------------------------------------------------------------------------------------------

def test_auto_rule_table():
    """_auto_formulation, one assertion per row of the rule (module FORMULATION RULE)."""
    from dynameta.optics.lumenairy_bridge.rcwa_backend import _auto_formulation, _is_hard
    assert _auto_formulation([], is_2d=False, pol_row=0, conical=False)[0] == "laurent"
    assert _auto_formulation(["g"], is_2d=False, pol_row=0, conical=False)[0] == "li"
    assert _auto_formulation(["g"], is_2d=False, pol_row=1, conical=False)[0] == "laurent"
    assert _auto_formulation(["g"], is_2d=False, pol_row=1, conical=True)[0] == "li"
    # 2-D stacks keep Laurent's rule (measured: see test_2d_stacks_keep_laurent_...)
    assert _auto_formulation(["g"], is_2d=True, pol_row=0, conical=False)[0] == "laurent"
    assert _auto_formulation(["g"], is_2d=True, pol_row=1, conical=True)[0] == "laurent"
    # hardness: Re(eps) < 0 anywhere, or a contrast above 10 (a zero sample is infinite contrast)
    assert _is_hard([1.0, -0.01 + 1j]) and _is_hard([1.0, 10.5]) and _is_hard([0.0, 1.0])
    assert not _is_hard([1.0, 9.9 + 0.5j]) and not _is_hard([2.25])


@_needs("rcwa_shapes_li")
def test_stack_records_the_rule_and_explicit_laurent_warns_once_per_solver():
    """design_to_rcwa_stack records what ran; an explicit 'laurent' on a stack the rule sends to
    'li' warns once per design_to_rcwa_stack call and ONCE PER SOLVER over a sweep (not per
    wavelength); 'auto' and TE never warn."""
    from dynameta.optics.lumenairy_bridge import design_to_rcwa_stack
    d = strip_design()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        stk, _ = design_to_rcwa_stack(d, LAM, eps_by_region=ebr(), n_orders=8)
        assert (stk.dynameta_formulation, stk.dynameta_formulation_requested) == ("li", "auto")
        assert "grating" in stk.dynameta_formulation_reason
        assert all(L.formulation == "laurent" for L in stk.layers if L.kind == "uniform")
        assert [L.formulation for L in stk.layers if L.kind == "shapes"] == ["li"]
        te, _ = design_to_rcwa_stack(strip_design(pol="y"), LAM, eps_by_region=ebr(), n_orders=8)
        assert te.dynameta_formulation == "laurent"
    with pytest.warns(Warning, match="formulation='laurent'") as rec:
        stk, _ = design_to_rcwa_stack(d, LAM, eps_by_region=ebr(), n_orders=8,
                                      formulation="laurent")
    assert len(rec) == 1 and stk.dynameta_formulation == "laurent"
    s = _solver(n_orders=8, formulation="laurent")
    lams = [1.15e-6, 1.2e-6, 1.25e-6]
    with pytest.warns(Warning) as rec:
        out = s.solve_sweep(d, None, lambda lam: ebr(), lams, None, None)
        s(d, None, ebr(), LAM, None, None)
    assert len(out) == 3 and len(rec) == 1, [str(w.message) for w in rec]
    from dynameta.warning_types import AdvisoryWarning
    assert rec[0].category is AdvisoryWarning


@_needs("rcwa_shapes_li")
def test_2d_stacks_keep_laurent_and_explicit_li_rasterizes_curved_layers():
    """2-D stacks: 'auto' keeps Laurent's rule (bit-identical to v0.15.1 on the analytic layers).
    The plan's draft sent hard 2-D rectangle layers to 'li'; MEASURED first (no reference is run
    here: the 3-D FEM takes 4-25 min per point):
      metal-PATCH array (300 nm squares eps -3+1i, P 600 nm, 60 nm on a 150 nm eps-2.1 spacer,
      glass, 1.31 um, x). References: 3-D FEM (optics.solver, quarter cell) R 0.074403 / 0.074469
      at order 2 / 3 (h 30 nm); 2-D spectral-element PMM (curved-cell engine) R 0.073879 /
      0.074186 / 0.074378 / 0.074485 at n_modes 5 / 6 / 7 / 8.
        n_orders          4        8        12       16       20
        'li' (analytic)   0.07441  0.07428  0.07392  0.07373  0.07365   (drifts away)
        Laurent           0.07987  0.07813  0.07580  0.07479  0.07441   (closes in)
      ('li' on an eps_cell -- area or point raster -- agrees with the analytic 'li' to 5e-5.)
      metal-DISK array (eps -3+1i, r = 0.3 P, P 600 nm, 100 nm, air | glass, 1.31 um, x) vs the
      3-D FEM (order 4, h 20 nm; R 0.135429): |r - FEM| / |R - FEM| at n_orders 6 ... 24
        'li' + area raster  8.1e-3 5.2e-3 5.0e-3 4.5e-3 4.2e-3 3.3e-3 / R 3.3e-3 ... 7.3e-6
        Laurent, exact disk 1.0e-2 9.4e-3 4.8e-3 5.3e-3 4.8e-3 2.4e-3 / R 6.8e-3 ... 1.5e-3
    So lumenairy's 2-D Li rule is not admitted by default. An explicit 'li' still works: a
    rectangle stays an analytic shapes layer under Li's rule, a disk becomes an area raster
    carrying the inverse-rule companion."""
    from dynameta.geometry import Design, Inclusion, Layer, Stack, UnitCell
    from dynameta.geometry.cross_section import Circle, Rectangle
    from dynameta.geometry.specs import OpticalSpec
    from dynameta.materials import ConstantOptical, Material, MaterialRegistry
    from dynameta.optics.lumenairy_bridge import design_to_rcwa_stack
    reg = MaterialRegistry()
    for n, e in (("air", 1.0), ("glass", 2.25), ("metal", -3.0 + 1.0j)):
        reg.add(Material(n, ConstantOptical(complex(e))))
    per = 600e-9

    def design(shape):
        return Design(name="d", unit_cell=UnitCell.square(per),
                      stack=Stack(layers=[Layer("disk", 100e-9, "air",
                                                inclusions=[Inclusion(shape, "metal")])],
                                  superstrate_material="air", substrate_material="glass"),
                      electrodes=[], materials=reg, optical=OpticalSpec(polarization="x"))
    disk = design(Circle(per / 2, per / 2, 0.3 * per))
    rect = design(Rectangle(per / 2, per / 2, 300e-9, 200e-9))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for d in (disk, rect):
            stk, _ = design_to_rcwa_stack(d, 1.31e-6, n_orders=3)
            assert stk.dynameta_formulation == "laurent" and "2-D" in \
                stk.dynameta_formulation_reason
            assert [(L.kind, L.formulation) for L in stk.layers] == [("shapes", "laurent")]
            a = _solver(n_orders=4)(d, None, {}, 1.31e-6, 1.0, 1.5)
            b = _solver(n_orders=4, formulation="laurent", raster_sampling="point")(
                d, None, {}, 1.31e-6, 1.0, 1.5)
            assert (a.R, a.T, a.r, a.t) == (b.R, b.T, b.r, b.t)   # v0.15.1's analytic layer
        stk, _ = design_to_rcwa_stack(disk, 1.31e-6, n_orders=3, formulation="li")
        rec = stk.layers[0]
        assert (rec.kind, rec.formulation) == ("iso", "li")
        if hasattr(rec, "normal_cells"):
            assert rec.normal_cells is not None              # the inverse-rule companion
        stk, _ = design_to_rcwa_stack(rect, 1.31e-6, n_orders=3, formulation="li")
        assert [(L.kind, L.formulation) for L in stk.layers] == [("shapes", "li")]


@_needs("rcwa_analytic_shapes")
def test_every_lumenairy_add_layer_call_names_its_formulation(monkeypatch):
    """lumenairy 5.51 changes add_layer's OWN default to 'auto': no bridge call may rely on the
    default. Spy on RCWAStack.add_layer through every dispatch route of design_to_rcwa_stack
    (uniform, shapes, area and point raster, gridded slabs, uniform tensor, tensor background
    with inclusions) and through LumenairyStackSolver (eps, eps_cell, tensor, shapes slabs)."""
    import lumenairy as lum
    from dynameta.core.eps_field import EpsField
    from dynameta.core.layered import LayeredSlab, LayeredStack
    from dynameta.geometry import Inclusion, Layer
    from dynameta.geometry.cross_section import Circle
    from dynameta.optics.lumenairy_bridge import LumenairyStackSolver, design_to_rcwa_stack
    calls = []
    real = lum.RCWAStack.add_layer

    def spy(self, thickness, **kw):
        calls.append(kw)
        return real(self, thickness, **kw)
    monkeypatch.setattr(lum.RCWAStack, "add_layer", spy)
    d = strip_design()
    d.stack.layers.insert(1, Layer("lc", 20e-9, "oxB"))
    d.stack.layers.insert(1, Layer("pill", 20e-9, "oxB",
                                   inclusions=[Inclusion(Circle(P / 2, P / 2, 60e-9), "metal")]))
    tens = np.diag([2.0, 3.0, 2.5]).astype(complex)
    e = dict(ebr(), lc=EpsField(tensor=tens), pill=EpsField(tensor=tens))
    xs = np.arange(16) * (P * 1e9 / 16)
    e2 = {"film": EpsField(x_axis_u=xs, y_axis_u=np.array([0.0]),
                          z_axis_u=np.array([0.0, 6.0]),
                          values_zyx=np.repeat((2.0 + 0.1 * np.arange(16.0))[None, None, :], 2,
                                               axis=0).astype(complex))}
    for kw in (dict(), dict(inclusion_model="raster"), dict(raster_sampling="point"),
               dict(formulation="li")):
        design_to_rcwa_stack(d, LAM, eps_by_region=e, n_orders=3, **kw)
        design_to_rcwa_stack(strip_design(), LAM, eps_by_region=e2, n_orders=3, **kw)
    sol = LumenairyStackSolver(n_orders=3)
    cell = np.full((13, 13), 2.0 + 0j)
    cell[:6] = -5.0 + 1j
    st = LayeredStack(1.0 + 0j, 1.5 + 0j, [
        LayeredSlab(10e-9, eps=2.0 + 0j), LayeredSlab(10e-9, eps_cell=cell),
        LayeredSlab(10e-9, eps_tensor_cell=np.broadcast_to(tens, (13, 13, 3, 3)).copy()),
        LayeredSlab(10e-9, shapes=[{"shape": "rectangle", "size": (100e-9, 120e-9),
                                    "center": (P / 2, P / 2), "eps": -20 + 1j}],
                    eps_background=1.0 + 0j)], period_x_m=P, period_y_m=P)
    sol.solve(st, LAM, strip_design().optical)
    assert calls and all("formulation" in kw for kw in calls), [
        sorted(kw) for kw in calls if "formulation" not in kw]
    assert {kw["formulation"] for kw in calls} >= {"laurent", "li"}


def test_cache_fingerprint_separates_every_control(tmp_path):
    """make_lumenairy_rcwa_solver carries cache_fingerprint (v0.15.1 had none, so two RCWA solvers
    with different settings shared OpticalSolverCache keys and a cached Laurent result would be
    served for an 'li' request). Every answer-changing control enters it, plus the auto rule's
    version and the lumenairy version; the closure also carries native_carrier_grid."""
    from dynameta.optics.lumenairy_bridge.rcwa_backend import AUTO_RULE_VERSION
    base = _solver().cache_fingerprint
    assert "rule={}".format(AUTO_RULE_VERSION) in base and "lumenairy=" in base
    for kw in (dict(formulation="laurent"), dict(formulation="li"), dict(raster_sampling="point"),
               dict(n_orders=12), dict(n_orders_y=3), dict(n_slices=4), dict(cell_samples=256),
               dict(inclusion_model="raster"), dict(absorption=True), dict(stabilize=True)):
        assert _solver(**kw).cache_fingerprint != base, kw
    assert _solver(n_workers=3).cache_fingerprint == base     # threading changes no answer
    assert _solver().native_carrier_grid is True
    from dynameta.cache import OpticalSolverCache
    a = OpticalSolverCache(_solver(formulation="li"), str(tmp_path / "a.h5"), autosave=False)
    b = OpticalSolverCache(_solver(formulation="laurent"), str(tmp_path / "b.h5"),
                           autosave=False)
    assert a._solver_id != b._solver_id


def test_bad_keywords_are_refused():
    from dynameta.optics.lumenairy_bridge import design_to_rcwa_stack
    with pytest.raises(ValueError, match="formulation"):
        _solver(formulation="direct")
    with pytest.raises(ValueError, match="raster_sampling"):
        _solver(raster_sampling="nearest")
    with pytest.raises(ValueError, match="raster_sampling"):
        design_to_rcwa_stack(strip_design(), LAM, raster_sampling="centre")


def test_metre_axis_field_raises_at_the_rcwa_bridge():
    """The slab-thickness guard at the RCWA call site: a gridded field whose z axis is in metres
    (read as nm: slabs 1e9 times too thin) used to delete the layer silently (v0.15.1, n_orders 5:
    r -0.801514-0.526046i for the 6 nm film vs -0.698758-0.407169i with the nm axis, no error); slice_layer_eps_field
    now refuses it, naming the layer."""
    f = graded_film_field(11)
    f.z_axis_u = f.z_axis_u * 1e-9
    with pytest.raises(ValueError, match="film"):
        _solver(n_orders=5)(strip_design(), None, {"film": f}, LAM, None, None)


# ---------------------------------------------------------------------------------------------
# the sweep resolves once; LumenairyStackSolver
# ---------------------------------------------------------------------------------------------

def test_sweep_resolves_the_formulation_per_wavelength():
    """A dielectric strip whose permittivity is hard (contrast > 10) at ONE wavelength of the
    sweep only. 'auto' is resolved per (design, eps, wavelength)
    point on every path, so a result never depends on which other wavelengths share the sweep or
    are already cached -- solve_sweep runs 'li' only where the strip is hard and gives the
    single-wavelength call's r bit for bit elsewhere. (Before, it resolved once per spectrum:
    ['li', 'li', 'li'].)"""
    from dynameta.materials import Material
    from dynameta.materials.optical_model import OpticalModel

    class _Hard(OpticalModel):                     # eps 12 at 1.20 um, 2.25 elsewhere
        def eps(self, lambda_m, *, n_m3=None):
            return 12.0 + 0.1j if abs(lambda_m - 1.2e-6) < 1e-12 else 2.25 + 0.01j
    d = strip_design(strip_material="hardx", extra=[Material("hardx", _Hard())])
    s = _solver(n_orders=6)
    s(d, None, ebr(), 1.15e-6, None, None)
    assert s.last_formulation == "laurent"
    seen = []
    from dynameta.optics.lumenairy_bridge import rcwa_backend
    real = rcwa_backend._build_rcwa_stack

    def spy(*a, **k):
        stk, names = real(*a, **k)
        seen.append(stk.dynameta_formulation)
        return stk, names
    r_single = complex(s(d, None, ebr(), 1.15e-6, None, None).r)
    rcwa_backend._build_rcwa_stack = spy
    try:
        from _strip_cell_fixture import li_companion_floor
        with li_companion_floor():                 # 1.20 um runs Li (below lumenairy 5.31: said once)
            out = s.solve_sweep(d, None, lambda lam: ebr(), [1.15e-6, 1.2e-6, 1.25e-6], None, None)
    finally:
        rcwa_backend._build_rcwa_stack = real
    assert seen == ["laurent", "li", "laurent"], seen
    assert complex(out[0].r) == r_single


@_needs("rcwa_analytic_shapes")
def test_stack_solver_default_is_the_v0151_solve_and_li_is_explicit():
    """LumenairyStackSolver solves a structured LayeredStack on a 2-D lattice, where 'auto' is
    Laurent's rule: a metal eps_cell slab and a metal shapes slab solve bit for bit as v0.15.1's
    calls did (which named no formulation and got lumenairy 5.50's default, Laurent -- spelled
    out here), with no warning; formulation='li' reaches lumenairy (layer records 'li')."""
    import lumenairy as lum
    from dynameta.core.layered import LayeredSlab, LayeredStack
    from dynameta.optics.lumenairy_bridge import LumenairyStackSolver
    opt = strip_design().optical
    metal = np.ones((41, 41), complex)
    metal[10:30, :] = -60 + 4j
    rect = [{"shape": "rectangle", "size": (150e-9, 150e-9), "center": (P / 2, P / 2),
             "eps": -60 + 4j}]

    def stack(cell=None, shapes=None):
        sl = (LayeredSlab(40e-9, eps_cell=cell) if cell is not None
              else LayeredSlab(40e-9, shapes=shapes, eps_background=1.0 + 0j))
        return LayeredStack(1.0 + 0j, 1.45 + 0j, [sl, LayeredSlab(80e-9, eps=-120 + 20j)],
                            period_x_m=P, period_y_m=P)

    def direct(cell=None, shapes=None, form="laurent"):
        rs = lum.RCWAStack(P, period_y=P, n_superstrate=1.0 + 0j, n_substrate=1.45 + 0j,
                           n_orders=5, n_orders_y=5)
        if cell is not None:
            rs.add_layer(40e-9, eps_cell=cell, formulation=form)
        else:
            rs.add_layer(40e-9, shapes=shapes, eps_background=1.0 + 0j, formulation=form)
        rs.add_layer(80e-9, eps=-120 + 20j)
        rs.set_source(LAM, theta=0.0, phi=0.0)
        return complex(np.asarray(rs.solve().jones_reflection())[0, 0])
    s = LumenairyStackSolver(n_orders=5)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert s.solve(stack(metal), LAM, opt).r == direct(metal)
        assert s.last_formulation == "laurent"
        assert s.solve(stack(shapes=rect), LAM, opt).r == direct(shapes=rect)
        s_li = LumenairyStackSolver(n_orders=5, formulation="li")
        assert s_li.solve(stack(metal), LAM, opt).r == direct(metal, form="li")
        assert s_li.last_formulation == "li"


# ---------------------------------------------------------------------------------------------
# per-layer absorption of an 'li' solve: the face-flux split
# ---------------------------------------------------------------------------------------------

@_needs("rcwa_shapes_absorption")
def test_li_absorption_is_the_face_flux_split_and_closes_independently():
    """With Li's rule lumenairy's Im(eps)|E|^2 volume integral (before 5.51) is not consistent
    with the inverse-rule field; the bridge attributes each layer its Poynting-flux difference.
    The split is an INDEPENDENT measurement: its sum closes on the far-field 1 - R - T to
    round-off (measured 6.6e-15 here), every layer is non-negative (a passive stack), and the
    lossless layers carry ~0. Cross-engine: the 2-D TM FEM's per-region absorption on the same
    cell (mirror 0.3928, strips 0.3366, film 0.1578). MEASURED (TM, lumenairy 5.50.2), face-flux
    split mirror / strips / film and its largest distance from the FEM, n_orders 30 / 50 / 100 /
    150 / 200: 0.230 / 0.093 / 0.034 / 0.020 / 0.012 (at 150: 0.3846 / 0.3561 / 0.1543); the
    bar (0.03) is asserted at n_orders 150, where the split's truncation error (monotone in the
    order) has fallen below it with a 1.5x margin. lumenairy's volume integral on the SAME 'li' solve
    at 150: 0.2911 / 0.4800 / 0.1238 (0.14 off on the strips), so the bar discriminates the two
    splits. The Laurent path keeps lumenairy's own split, bit for bit."""
    d = strip_design()
    res = _solver(n_orders=150, absorption=True)(d, None, ebr(), LAM, None, None)
    pra = res.per_region_absorption
    assert abs(res.A_independent - (1.0 - res.R - res.T)) < 1e-10
    assert min(pra.values()) > -1e-12
    for k in ("lo_oxa", "lo_oxb", "up_oxb", "up_oxa"):
        assert abs(pra[k]) < 1e-9, (k, pra[k])
    assert pra["grating"] > 0.1 and pra["mirror"] > 0.1 and pra["film"] > 0.1
    if HAVE_NGS:
        fem = fem_strip().per_region_absorption
        for mine, theirs in (("mirror", "mirror"), ("grating", "strip"), ("film", "film0")):
            assert abs(pra[mine] - fem[theirs]) < 0.03, (mine, pra[mine], fem[theirs])
    with pytest.warns(Warning, match="formulation='laurent'"):
        lau = _solver(n_orders=12, absorption=True, formulation="laurent")(
            d, None, ebr(), LAM, None, None)
    import lumenairy as lum  # noqa: F401  (lumenairy's own split on the Laurent solve)
    from dynameta.optics.lumenairy_bridge import design_to_rcwa_stack
    with pytest.warns(Warning):
        stk, names = design_to_rcwa_stack(d, LAM, eps_by_region=ebr(), n_orders=12,
                                          formulation="laurent")
    stk.set_source(LAM, theta=0.0, phi=0.0)
    la = np.asarray(stk.solve(retain_internal=True).layer_absorption())[0]
    own = {}
    for nm, v in zip(names, la):
        own[nm] = own.get(nm, 0.0) + float(v)
    assert lau.per_region_absorption == own
