"""Lumenairy 5.51 adoption in the optical bridges: feature detection and the behaviours that change with the
installed lumenairy, tested on BOTH sides (whichever side the installed lumenairy is on; the CI floor leg runs
5.22, the other legs the current release).

* The 5.51 rows of _common.FEATURE_FLOORS ('pmm_geometric_grade', 'pmm_small_element_guard') are decided by
  asking the installed lumenairy for the API (_common.FEATURE_PROBES), not by its version string.
* pmm_kwargs={'grade': 'geometric', ...} reaches PMMStack where it exists and is refused by name where an
  older release would read it as grade=True.
* pmm_kwargs={'allow_small_elements': True} (5.51's way back from its small-element guard) is refused by name
  on older releases.
* At an exact Rayleigh anomaly lumenairy 5.51's PMMStack nudges the wavelength and says so
  (WoodNudgeWarning); older releases solve the ill-posed point silently. The bridge passes the notice on.
* The differentiable PMM forward (rcwa_design) calls PMMStack(far_field_orders=...), never the n_orders alias.
* GAIN MEDIA (Im eps < 0). lumenairy 5.50.x splits the modes of a gain layer in a PMMStack wrongly (R + T of
  1e1 .. 1e4); 5.22.0 and 5.51.0 do it right. The PMM bridge solves a gain layer where the probe 'pmm_gain_media'
  says the installed PMMStack is right and refuses it by name elsewhere; the RCWA and PMM2D bridges solve gain
  layers on every release; a gain half-space is refused by every engine. Reference: the 2-D TM FEM (no lumenairy).
* rcwa_design.rcwa_stack_RT / rcwa_stack_jones default to formulation 'auto' where Li's rule traces under
  jax.grad (probe 'rcwa_li_traced', lumenairy 5.51.0) and to 'laurent' elsewhere; on either side the default's
  jax.grad equals a NumPy central difference of the same default.
"""
import warnings

import numpy as np
import pytest

lum = pytest.importorskip("lumenairy")

from dynameta.optics.lumenairy_bridge import _common as C                       # noqa: E402

P, LAM = 400e-9, 1200e-9


def _api_has_geometric():
    import importlib
    return hasattr(importlib.import_module("lumenairy.elements.pmm"), "GEOMETRIC_DEFAULT_RATIO")


def _strip_design():
    from dynameta.geometry import Design, Inclusion, Layer, Stack, UnitCell
    from dynameta.geometry.cross_section import Rectangle
    from dynameta.geometry.specs import OpticalSpec
    from dynameta.materials import ConstantOptical, Material, MaterialRegistry
    reg = MaterialRegistry()
    for n, e, metal in (("air", 1.0, False), ("glass", 1.45 ** 2, False), ("spacer", 2.0 ** 2, False),
                        ("metal", -60.0 + 4.0j, True), ("mirror_metal", -120.0 + 20.0j, True)):
        reg.add(Material(n, ConstantOptical(complex(e)), is_metal=metal))
    layers = [Layer("mirror", 80e-9, "mirror_metal"), Layer("spacer", 20e-9, "spacer"),
              Layer("grating", 40e-9, "air",
                    inclusions=[Inclusion(Rectangle(P / 2, P / 2, 150e-9, P), "metal")])]
    return Design(name="strip", unit_cell=UnitCell(period_x_m=P, period_y_m=P),
                  stack=Stack(layers=layers, superstrate_material="air", substrate_material="glass"),
                  electrodes=[], materials=reg, optical=OpticalSpec(polarization="x", incidence_angle_deg=0.0))


def test_probe_rows_follow_the_installed_api_not_the_version_string(monkeypatch):
    """The two 5.51 rows agree with the API actually present, and a spoofed __version__ does not change the
    answer (a release candidate or source checkout may carry the API under an older version string)."""
    import inspect
    has_geo = _api_has_geometric()
    has_small = "allow_small_elements" in inspect.signature(lum.PMMStack.__init__).parameters
    assert C.lumenairy_has("pmm_geometric_grade") is has_geo
    assert C.lumenairy_has("pmm_small_element_guard") is has_small
    for spoof in ("5.22.0", "5.99.0"):
        monkeypatch.setattr(lum, "__version__", spoof)
        assert C.lumenairy_has("pmm_geometric_grade") is has_geo
        assert C.lumenairy_has("pmm_small_element_guard") is has_small
    # rows WITHOUT a probe keep the version floor (the existing spoof-based refusal tests rely on it)
    monkeypatch.setattr(lum, "__version__", "5.22.0")
    assert C.lumenairy_has("rcwa_shapes_li") is False
    for name in C.FEATURE_PROBES:
        assert name in C.FEATURE_FLOORS


def test_geometric_grade_reaches_pmmstack_or_is_refused_by_name():
    from dynameta.optics.lumenairy_bridge._common import LumenairyFeatureUnavailable
    from dynameta.optics.lumenairy_bridge.pmm_backend import design_to_pmm_stack
    d = _strip_design()
    kw = {"grade": "geometric", "elements_per_region": 12}
    if not C.lumenairy_has("pmm_geometric_grade"):
        with pytest.raises(LumenairyFeatureUnavailable, match=r"grade='geometric'.*>= 5\.51\.0"):
            design_to_pmm_stack(d, LAM, degree=8, pmm_kwargs=kw)
        return
    stack, _names = design_to_pmm_stack(d, LAM, degree=8, pmm_kwargs=kw)
    assert stack.grade[0] == "geometric"                     # lumenairy stores the validated spec
    # the same stack built directly gives the same Jones matrix bit for bit
    direct = lum.PMMStack(P, n_substrate=1.45, n_superstrate=1.0, degree=8, far_field_orders=21, **kw)
    direct.add_layer(40e-9, segments=[(0.3125, 1.0 + 0j), (0.375, -60.0 + 4.0j), (0.3125, 1.0 + 0j)])
    direct.add_layer(20e-9, eps=4.0 + 0j)
    direct.add_layer(80e-9, eps=-120.0 + 20.0j)
    for st in (stack, direct):
        st.set_source(LAM, theta=0.0)
    j_bridge = np.asarray(stack.solve()[3])
    j_direct = np.asarray(direct.solve()[3])
    assert np.array_equal(j_bridge, j_direct)


def test_allow_small_elements_is_refused_by_name_where_it_does_not_exist():
    from dynameta.optics.lumenairy_bridge._common import LumenairyFeatureUnavailable
    from dynameta.optics.lumenairy_bridge.pmm_backend import _resolve_pmm_kwargs
    if C.lumenairy_has("pmm_small_element_guard"):
        assert _resolve_pmm_kwargs(lum, {"allow_small_elements": True}) == {"allow_small_elements": True}
    else:
        with pytest.raises(LumenairyFeatureUnavailable,
                           match=r"'allow_small_elements' \(needs lumenairy >= 5\.51\.0\)"):
            _resolve_pmm_kwargs(lum, {"allow_small_elements": True})


def test_exact_rayleigh_anomaly_notice_depends_on_the_release():
    """Period == wavelength at normal incidence puts the +-1 orders exactly at cut-off in the superstrate.
    lumenairy >= 5.51 nudges the PMMStack wavelength by ~1e-7 relative and warns (WoodNudgeWarning, a
    UserWarning subclass); older releases solve the ill-posed point without a word. The bridge does not
    filter the notice. (No API exposes this behaviour, so the side is read from the version.)"""
    from dynameta.optics.lumenairy_bridge.pmm_backend import make_lumenairy_pmm_solver
    d = _strip_design()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        res = make_lumenairy_pmm_solver(degree=8)(d, None, {}, P, None, None)
    nudged = [x for x in w if type(x.message).__name__ == "WoodNudgeWarning"]
    if C.lumenairy_version() >= (5, 51, 0):
        assert nudged, [str(x.message)[:80] for x in w]
    else:
        assert not nudged
    assert np.isfinite(res.R) and 0.0 <= res.R <= 1.0 + 1e-9


def test_differentiable_pmm_forward_uses_far_field_orders(monkeypatch):
    from dynameta.optics.lumenairy_bridge import rcwa_design as RD
    seen = {}
    real = lum.PMMStack

    class _Spy(real):
        def __init__(self, *a, **kw):
            seen.update(kw)
            super().__init__(*a, **kw)

    monkeypatch.setattr(lum, "PMMStack", _Spy)
    layers = [([(0.4, 4.0 + 0j), (0.6, 1.0 + 0j)], 100e-9)]
    RD.pmm_stack_RT(layers, 1200e-9, n_substrate=1.45, n_superstrate=1.0, period=610e-9, n_orders=11)
    assert seen.get("far_field_orders") == 11 and "n_orders" not in seen


# ---------------------------------------------------------------------------------------------------------------
# gain media
# ---------------------------------------------------------------------------------------------------------------
# The strip cell of tests/_strip_cell_fixture with the metal strips replaced by a GAIN dielectric, eps 2.25 - 0.05i
# (TM, 1200 nm). Reference: the 2-D TM FEM (optics.fem_2d_tm, no lumenairy), every corner graded
# (corner_mode='all'): r = -0.64131547-0.74579539i at order 4 (6 levels) / 5 / 6 (8 levels) / 7 (10 levels), and
# -0.64131546-0.74579539i at order 7 with half the element size -- converged to ~1e-8.
EPS_GAIN = 2.25 - 0.05j
R_GAIN_FEM = -0.64131547 - 0.74579539j
# The same cell with the passive strip eps 2.25 + 0.01i is 7.8e-4 away (RCWA 'li', n_orders 100): a solver that
# lost the sign of Im eps misses R_GAIN_FEM by that much; one that split the gain modes wrongly by O(1).
GAIN_SHIFT = 7.8e-4


def _gain_strip(pol="x"):
    import _strip_cell_fixture as F
    from dynameta.materials import ConstantOptical, Material
    return F.strip_design(pol=pol, strip_material="gainmat",
                          extra=(Material("gainmat", ConstantOptical(EPS_GAIN)),))


def _solve_gain_strip(solver, design=None):
    import _strip_cell_fixture as F
    return complex(solver(design or _gain_strip(), None, F.ebr(), F.LAM, None, None).r)


def test_gain_strip_fem_reference():
    """R_GAIN_FEM recomputed with the FEM at order 4, corner_mode='all' (named explicitly: the reference must not
    move with the FEM's default corner rule). MEASURED 1e-8 from the order-7 value; bar 1e-7."""
    pytest.importorskip("ngsolve")
    import _strip_cell_fixture as F
    from dynameta.optics.fem_2d_tm import TMBlock, TMLayer, TMMeshSpec, solve_fem_2d_tm
    lays = [TMLayer("mirror", 80e-9, F.EPS_MIRROR), TMLayer("lo_oxa", 2e-9, 1.70 ** 2),
            TMLayer("lo_oxb", 6e-9, 2.00 ** 2), TMLayer("film", 6e-9, F.eps_film()),
            TMLayer("up_oxb", 6e-9, 2.00 ** 2), TMLayer("up_oxa", 2e-9, 1.70 ** 2),
            TMLayer("grating", F.H_STRIP, 1.0 + 0j,
                    (TMBlock(F.P / 2 - F.W / 2, F.P / 2 + F.W / 2, EPS_GAIN, "strip"),))]
    r = solve_fem_2d_tm(F.P, lays, lambda_m=F.LAM, n_super=1.0, n_sub=F.N_GLASS, order=4, num_threads=2,
                        mesh_spec=TMMeshSpec(corner_mode="all")).r
    assert abs(r - R_GAIN_FEM) < 1e-7, r


def test_gain_probe_agrees_with_the_installed_pmmstack_on_the_gain_strip(monkeypatch):
    """The probe solves its own small cell (a gain ridge against RCWAStack). Its verdict must match what the
    installed PMMStack does on a DIFFERENT gain cell: the gain strip built by the bridge with the refusal switched
    off, solved raw, against the FEM. MEASURED: 5.5e-7 from R_GAIN_FEM on 5.22.0 and 5.51.0 (degree 16), R = 6448
    on 5.50.2; the split at 1e-5 is decided by six orders of magnitude either way."""
    import _strip_cell_fixture as F
    from dynameta.optics.lumenairy_bridge.pmm_backend import design_to_pmm_stack
    has = C.lumenairy_has("pmm_gain_media")
    monkeypatch.setitem(C.FEATURE_PROBES, "pmm_gain_media", lambda _lum: True)
    st, _ = design_to_pmm_stack(_gain_strip(), F.LAM, eps_by_region=F.ebr(), degree=16)
    st.set_source(F.LAM, theta=0.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")              # the wrong split also trips lumenairy's energy warning
        r_raw = complex(np.asarray(st.solve()[3])[0, 0])
    assert has is (abs(r_raw - R_GAIN_FEM) < 1e-5), (has, r_raw)


def test_pmm_bridge_solves_a_gain_layer_or_refuses_it_by_name():
    """Where the probe passes, the gain strip through make_lumenairy_pmm_solver against the FEM. MEASURED
    (5.22.0 and 5.51.0): 1.53e-6 / 5.54e-7 / 1.30e-7 at degree 12 / 16 / 24 (the algebraic corner convergence of
    the TM channel); bar 2e-6 at degree 16, 390x under GAIN_SHIFT. Where it fails (5.50.x), the bridge refuses with
    LumenairyFeatureUnavailable naming the layer instead of returning R = 6448."""
    from dynameta.optics.lumenairy_bridge import make_lumenairy_pmm_solver
    if C.lumenairy_has("pmm_gain_media"):
        r = _solve_gain_strip(make_lumenairy_pmm_solver(degree=16))
        assert abs(r - R_GAIN_FEM) < 2e-6, abs(r - R_GAIN_FEM)
    else:
        with pytest.raises(C.LumenairyFeatureUnavailable, match=r"layer 'grating' holds a gain medium.*5\.51\.0"):
            _solve_gain_strip(make_lumenairy_pmm_solver(degree=16))


def test_pmm_gain_refusal_path_and_passive_layers_never_probe(monkeypatch):
    """Both sides on any tree: with the probe answering False the PMM bridge (and the differentiable PMM forward)
    refuse the gain layer; a passive design never runs the probe; the RCWA bridge, which does not depend on it,
    still solves the gain strip."""
    from dynameta.optics.lumenairy_bridge import make_lumenairy_pmm_solver, pmm_stack_RT
    import _strip_cell_fixture as F

    def _no(_lum):
        return False
    monkeypatch.setitem(C.FEATURE_PROBES, "pmm_gain_media", _no)
    with pytest.raises(C.LumenairyFeatureUnavailable, match="gain medium"):
        _solve_gain_strip(make_lumenairy_pmm_solver(degree=8))
    with pytest.raises(C.LumenairyFeatureUnavailable, match="gain medium"):
        pmm_stack_RT([([(0.4, 12.0 - 0.05j), (0.6, 1.0)], 80e-9)], 1000e-9, n_substrate=1.45,
                     n_superstrate=1.0, period=500e-9, degree=8)

    def _boom(_lum):
        raise AssertionError("the gain probe ran for a passive design")
    monkeypatch.setitem(C.FEATURE_PROBES, "pmm_gain_media", _boom)
    res = make_lumenairy_pmm_solver(degree=8)(F.strip_design(), None, F.ebr(), F.LAM, None, None)
    assert np.isfinite(res.R)
    assert C.is_gain_eps(EPS_GAIN) and not C.is_gain_eps(2.25 - 1e-14j) and not C.is_gain_eps(-60 + 4j)
    assert C.is_gain_eps(np.diag([2.0, 2.0 - 0.01j, 2.0])) and not C.is_gain_eps(np.diag([2.0, 2.0 + 0.01j, 2.0]))


def test_rcwa_and_pmm2d_bridges_solve_a_gain_layer_on_every_release():
    """RCWA ('li', n_orders 100) and the PMM2D hybrid engine (degree 11) handle gain layers on 5.22.0, 5.50.2
    and 5.51.0 (measured: no refusal is needed). MEASURED |r - R_GAIN_FEM|: RCWA 3.1e-7 (5.50.2 / 5.51.0; 2.0e-6
    on 5.22.0, whose analytic shapes take Li's rule through a raster without its inverse-rule companion), bar
    5e-6; PMM2D hybrid 2.1e-4 / 2.6e-4 / 3.2e-4 at degree 9 / 11 / 13 (the engine's Fourier floor in this cell,
    not a degree error), bar 5e-4. The PMM2D bar is under GAIN_SHIFT and far under a mis-split answer."""
    from _strip_cell_fixture import li_companion_floor
    from dynameta.optics.lumenairy_bridge import make_lumenairy_pmm2d_solver, make_lumenairy_rcwa_solver
    with li_companion_floor():
        r = _solve_gain_strip(make_lumenairy_rcwa_solver(n_orders=100, formulation="li"))
    assert abs(r - R_GAIN_FEM) < 5e-6, abs(r - R_GAIN_FEM)
    r = _solve_gain_strip(make_lumenairy_pmm2d_solver(engine="hybrid", degree=11))
    assert abs(r - R_GAIN_FEM) < 5e-4, abs(r - R_GAIN_FEM)


def test_differentiable_pmm_forward_gain_layer_against_rcwa():
    """pmm_stack_jones on a gain ridge (eps 12 - 0.05i, 40 % of a 500 nm period, 80 nm, over 100 nm of eps 2.1;
    1000 nm, 0.3 rad) against rcwa_stack_jones ('li', n_orders 40). MEASURED on 5.22.0 / 5.51.0: |dJ| 1.78e-4
    (degree 14: the PMM truncation; RCWA at 40 orders is 1.4e-5 from 80); on 5.50.2 the PMM Jones is 2.8 off, so
    the forward refuses there. Bar 1e-3."""
    from dynameta.optics.lumenairy_bridge import pmm_stack_jones, rcwa_stack_jones
    layers = [([(0.4, 12.0 - 0.05j), (0.6, 1.0 + 0j)], 80e-9), (2.1 + 0j, 100e-9)]
    kw = dict(n_substrate=1.45, n_superstrate=1.0)
    if not C.lumenairy_has("pmm_gain_media"):
        with pytest.raises(C.LumenairyFeatureUnavailable, match="gain medium"):
            pmm_stack_jones(layers, 1000e-9, period=500e-9, angle=0.3, degree=14, **kw)
        return
    with warnings.catch_warnings():
        # R + T = 1.001 / 1.011 (the gain); lumenairy's PMMStack energy notice assumes a passive stack
        warnings.filterwarnings("ignore", message="PMMStack.solve: energy not conserved")
        _o, _R, _T, jp = pmm_stack_jones(layers, 1000e-9, period=500e-9, angle=0.3, degree=14, **kw)
    assert np.all(np.sum(_R, axis=1) + np.sum(_T, axis=1) > 1.0)      # net gain, as for RCWAStack
    s = 1000
    cell = np.where((np.arange(s) + 0.5) / s < 0.4, 12.0 - 0.05j, 1.0 + 0j)
    with warnings.catch_warnings():
        # lumenairy 5.51's RCWA passivity bound (sum over both rows <= 2) does not see the gain of an eps_cell
        warnings.filterwarnings("ignore", message="RCWAStack.solve: passive-structure energy bound")
        _o, _R, _T, jr, _jt = rcwa_stack_jones([(cell, 80e-9), (2.1 + 0j, 100e-9)], 1000e-9, period_x=500e-9,
                                               theta=0.3, n_orders=40, formulation="li", **kw)
    assert np.max(np.abs(np.asarray(jp) - np.asarray(jr))) < 1e-3


def test_gain_half_space_is_refused_by_every_bridge_engine():
    """A gain substrate / superstrate (Im n < 0) is refused before lumenairy is called, on every engine and
    every release (measured without the refusal: PMMStack T = 0 or R + T > 2, PMM2DStackHybrid R = 30,
    RCWAStack an energy error; _common.refuse_gain_half_space)."""
    import _strip_cell_fixture as F
    from dynameta.core.layered import LayeredSlab, LayeredStack
    from dynameta.materials import ConstantOptical, Material
    from dynameta.optics.lumenairy_bridge import (LumenairyStackSolver, make_lumenairy_pmm2d_solver,
                                                  make_lumenairy_pmm_solver, make_lumenairy_rcwa_solver,
                                                  pmm_stack_RT, rcwa_grating_RT, rcwa_stack_RT)
    gsub = Material("gglass", ConstantOptical((1.45 - 0.01j) ** 2))
    for side in ("substrate", "superstrate"):
        d = F.strip_design(extra=(gsub,))
        if side == "substrate":
            d.stack.substrate_material = "gglass"
        else:
            d.stack.superstrate_material = "gglass"
        for mk in (lambda: make_lumenairy_pmm_solver(degree=8), lambda: make_lumenairy_rcwa_solver(n_orders=5),
                   lambda: make_lumenairy_pmm2d_solver(engine="hybrid", degree=5, n_orders=3)):
            with pytest.raises(ValueError, match="gain " + side):
                mk()(d, None, F.ebr(), F.LAM, None, None)
    n_gain = 1.45 - 0.01j
    with pytest.raises(ValueError, match="gain substrate"):
        rcwa_stack_RT([(2.1 + 0j, 100e-9)], 1000e-9, n_substrate=n_gain, n_superstrate=1.0, period_x=500e-9)
    with pytest.raises(ValueError, match="gain superstrate"):
        pmm_stack_RT([(2.1 + 0j, 100e-9)], 1000e-9, n_substrate=1.45, n_superstrate=n_gain, period=500e-9)
    with pytest.raises(ValueError, match="gain substrate"):
        rcwa_grating_RT(500e-9, 4.0 + 0j, 1.0 + 0j, 100e-9, 0.5, 1000e-9, n_substrate=n_gain,
                        n_superstrate=1.0)
    st = LayeredStack(1.0 + 0j, n_gain, [LayeredSlab(100e-9, eps=2.1 + 0j)], period_x_m=F.P, period_y_m=F.P)
    with pytest.raises(ValueError, match="gain substrate"):
        LumenairyStackSolver(n_orders=3).solve(st, F.LAM, F.strip_design().optical)


# ---------------------------------------------------------------------------------------------------------------
# the factorization default of the differentiable RCWA forward
# ---------------------------------------------------------------------------------------------------------------
def _strip_over_mirror_R(e_strip, xp, formulation=None):
    """The TM metal strip over a mirror of rcwa_design's docstring (row 0, n_orders 40)."""
    from dynameta.optics.lumenairy_bridge import rcwa_stack_RT
    x = np.arange(401) / 401
    strip = (x >= 0.3) & (x < 0.7)
    cell = xp.where(xp.asarray(strip), e_strip, 1.0 + 0j)
    kw = {} if formulation is None else {"formulation": formulation}
    R, _T = rcwa_stack_RT([(cell, 40e-9), (4.0 + 0j, 14e-9), (-120 + 20j, 80e-9)], 1200e-9, n_substrate=1.45,
                          n_superstrate=1.0, period_x=400e-9, n_orders=40, row=0, **kw)
    return xp.real(R)


def test_rcwa_stack_default_formulation_follows_the_li_trace_probe():
    """The default is 'auto' where jax.grad traces Li's rule (probe 'rcwa_li_traced') and 'laurent' elsewhere,
    and the probe agrees with what the installed lumenairy does: jax.grad through formulation='li' succeeds where
    it says True and raises where it says False. On either side the default's jax.grad equals a central
    difference of the NumPy call with the same default. MEASURED (R; jax.grad d R / d Re eps_strip; central
    difference h = 1e-4): 5.51.0 default = 'li' 0.898233087113, 3.2243444e-3 vs 3.2243441e-3 (1.1e-7 relative);
    5.22.0 / 5.50.2 default = 'laurent' 0.916054019584, 9.8438786e-4 vs 9.8438518e-4 (2.7e-6 relative, the O(h^2)
    truncation of the difference). Bar 1e-5 relative: an AD gradient of one rule against a difference of the
    other (what a default resolving traced and concrete cells differently gives) is 3.3x off."""
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)
    from dynameta.optics.lumenairy_bridge.rcwa_design import default_stack_formulation
    traces = C.lumenairy_has("rcwa_li_traced")
    assert default_stack_formulation() == ("auto" if traces else "laurent")
    e0, h = -60.0, 1e-4
    r_default = float(_strip_over_mirror_R(e0 + 4j, np))
    r_li = float(_strip_over_mirror_R(e0 + 4j, np, "li"))
    r_laurent = float(_strip_over_mirror_R(e0 + 4j, np, "laurent"))
    assert abs(r_li - r_laurent) > 1e-2                         # the two rules are distinguishable here
    assert r_default == (r_li if traces else r_laurent)

    def grad_of(form):
        return float(jax.grad(lambda er: _strip_over_mirror_R(er + 4j, jnp, form))(jnp.asarray(e0)))
    if traces:
        grad_of("li")
    else:
        with pytest.raises(Exception):
            grad_of("li")
    g = grad_of(None)
    fd = (float(_strip_over_mirror_R(e0 + h + 4j, np)) - float(_strip_over_mirror_R(e0 - h + 4j, np))) / (2 * h)
    assert abs(g - fd) <= 1e-5 * abs(fd), (g, fd)
