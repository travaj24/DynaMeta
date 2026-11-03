"""Gates added with the lumenairy 5.50 re-verification of the bridge (2026-11-02).

(1) The theta = 0 PMM angle gradient. Through 5.49 lumenairy's PMM JAX twin returned a
    WRONG d/dtheta at exactly normal incidence on a mirror-symmetric grating, where symmetry forces
    0: the +/-m half-space orders are an exactly degenerate pair there, and the eig VJP's answer
    depended on the basis LAPACK returned inside the pair. Measured through pmm_stack_RT on THIS
    file's two fixtures (the very fixtures the old ANGLE CAVEAT in rcwa_design.py and the 5.42
    comment in _common.py were measured on): F530 AD dR/dtheta = 4.1e-3 at 5.30.0; F542
    AD d(R+T)/dtheta = -2.987e-6 at 5.31.0 and 5.42.1, bit-identical. lumenairy 5.50.0 routes
    every degenerate eigen-solve of its differentiable solvers through a cluster-adjoint rule
    (CHANGELOG 5.50.0, "JAX gradients at symmetric points"; switch lumenairy.jax_cluster_rule,
    default ON). Re-measured here: |AD| <= 2.0e-12 on both fixtures, both rows, R and R+T.
    The gate below is four-armed so it cannot pass for the wrong reason:
      * premise: the forward is mirror-even (central FD at 0 <= 6.6e-12) -- the truth IS 0;
      * the gate: |AD| at theta = 0 below 1e-9 (3000x under the 5.42 value, 500x above the
        measured 5.50 noise);
      * liveness: off-normal AD == FD to 1e-6 relative and nonzero, so a SEVERED angle gradient
        (which would read 0 at theta = 0 too) fails;
      * discrimination: the SAME gradient with the rule switched OFF exceeds the bar (the
        5.42 defect, reproduced in-process), so the gate demonstrably sees the failure mode.
    Requires lumenairy >= 5.50; on 5.22..5.49 (still inside the bridge floor) the defect is
    live and documented in rcwa_design.py's ANGLE NOTE.

(2) Audit R-32: the lumenairy floor has ONE source of truth (_common.VERSION_FLOOR); the
    pyproject requirement is now gated equal to it instead of kept in sync by hand.
"""
import importlib.util
import pathlib
import re

import numpy as np
import pytest

HAVE_LUM = importlib.util.find_spec("lumenairy") is not None
HAVE_JAX = importlib.util.find_spec("jax") is not None
_REPO = pathlib.Path(__file__).resolve().parents[1]


def _lum_at_least(v):
    if not HAVE_LUM:
        return False
    import lumenairy
    from dynameta.optics.lumenairy_bridge._common import parse_version
    return parse_version(lumenairy.__version__) >= v


needs_550 = pytest.mark.skipif(not (HAVE_JAX and _lum_at_least((5, 50, 0))),
                               reason="needs jax and lumenairy >= 5.50 (the cluster-adjoint rule)")

_LAM, _PER = 1.31e-6, 600e-9
# F530: the 5.30.0 fix-verification fixture; F542: the 5.42.1 differential-probe fixture
_FIX = {
    "F530": ([([(0.25, 1.0 + 0j), (0.5, 6.0 + 0j), (0.25, 1.0 + 0j)], 180e-9)], 10, 9),
    "F542": ([([(0.5, 4.0 + 0j), (0.5, 1.0 + 0j)], 220e-9)], 8, 11),
}
_BAR = 1e-9


def _fom(fix, row, which):
    import jax.numpy as jnp
    from dynameta.optics.lumenairy_bridge import pmm_stack_RT
    layers, degree, n_orders = _FIX[fix]

    def f(a):
        R, T = pmm_stack_RT(layers, _LAM, n_substrate=1.5 + 0j, n_superstrate=1.0 + 0j,
                            period=_PER, angle=a, degree=degree, n_orders=n_orders, row=row)
        return jnp.real(R if which == "R" else R + T)
    return f


@needs_550
@pytest.mark.parametrize("fix,row,which", [
    ("F530", 0, "R"), ("F542", 0, "R+T"),
    # ~40 s of eager cluster-rule grads per case on a loaded box: the second row of the
    # 5.30 fixture is the same mechanism, so it rides the slow leg
    pytest.param("F530", 1, "R", marks=pytest.mark.slow)])
def test_pmm_twin_angle_gradient_vanishes_at_normal_incidence(fix, row, which):
    import jax
    import jax.numpy as jnp
    import lumenairy
    jax.config.update("jax_enable_x64", True)
    f = _fom(fix, row, which)
    # premise: the forward is mirror-even in theta, so the true d/dtheta at 0 is exactly 0
    h = 1e-3
    fd0 = (float(f(jnp.asarray(h))) - float(f(jnp.asarray(-h)))) / (2 * h)
    assert abs(fd0) < 1e-10, "fixture premise broken: R(theta) not even, FD@0 = %.3e" % fd0
    # the gate (default rule ON)
    assert lumenairy.jax_cluster_rule_enabled()
    g0 = float(jax.grad(f)(jnp.asarray(0.0)))
    assert abs(g0) < _BAR, "theta=0 angle gradient regressed: %.3e (5.42.1 read -2.987e-6 on " \
                           "F542, 5.30.0 read 4.1e-3 on F530)" % g0
    # liveness: the angle gradient is NOT severed -- off-normal AD == central FD, nonzero (on R:
    # d(R+T)/dtheta of a lossless grating is 0 at every angle, so it cannot show liveness)
    fr = _fom(fix, row, "R")
    a, hh = 0.2, 1e-5
    ga = float(jax.grad(fr)(jnp.asarray(a)))
    fda = (float(fr(jnp.asarray(a + hh))) - float(fr(jnp.asarray(a - hh)))) / (2 * hh)
    assert abs(ga) > 1e-4, ga
    assert abs(ga - fda) <= 1e-6 * abs(fda), (ga, fda)
    # discrimination: with the rule OFF the pre-5.50 defect is back (measured 2026-11-02:
    # F530 row 0 / row 1 4.126e-3 / -2.073e-2 = the 5.30.0 values; F542 R+T -2.987e-6 = the
    # 5.42.1 value) and the bar catches it
    with lumenairy.jax_cluster_rule(False):
        g_off = float(jax.grad(f)(jnp.asarray(0.0)))
    assert abs(g_off) > 100 * _BAR, \
        "rule-off theta=0 gradient %.3e is under the bar: the gate cannot discriminate" % g_off


def test_pyproject_lumenairy_floor_is_VERSION_FLOOR():
    """audit R-32: pyproject's `lumenairy>=` and _common.VERSION_FLOOR were two hand-synced
    sources of truth. VERSION_FLOOR is THE floor (require_lumenairy enforces it at runtime and
    its error message quotes it); the packaging line must state the same version."""
    from dynameta.optics.lumenairy_bridge._common import VERSION_FLOOR, parse_version
    text = (_REPO / "pyproject.toml").read_text(encoding="cp1252")
    deps = re.search(r"^dependencies\s*=\s*\[(.*?)^\]", text, flags=re.S | re.M)
    assert deps, "pyproject.toml has no [project] dependencies list"
    hits = re.findall(r'"lumenairy\s*>=\s*([0-9][0-9.]*)"', deps.group(1))
    assert len(hits) == 1, "expected exactly one lumenairy floor in dependencies, got %r" % hits
    assert parse_version(hits[0]) == tuple(VERSION_FLOOR), (hits[0], VERSION_FLOOR)


# ---------------------------------------------------------------------------------------------
# (3) wave-7 (2026-11-02): PER-FEATURE lumenairy floors. The CI `floor` leg installs lumenairy
# 5.22.*, older than every 2026-11-02 bridge extension; those paths used to die INSIDE lumenairy
# (TypeError layer_grids, ImportError Circle, a numpy broadcast error on a pure (3, 3) eps) or --
# the RCWA analytic-shape row -- RUN WRONG (5.22 accepts overlapping shapes and adds their form
# factors, so the default inclusion_model='auto' built an additive layer where it should have
# rasterized). The gates below spoof lumenairy.__version__ so they run on EVERY installed version:
# the bridge must refuse BEFORE lumenairy is called (the spoof cannot change lumenairy's own
# behaviour), with the feature, its floor and the installed version in the message. On the
# pre-wave-7 code each of these fails: no LumenairyFeatureUnavailable exists, and with the
# version spoofed below 5.31 'auto' still emitted a 'shapes' layer.
# ---------------------------------------------------------------------------------------------

def _spoof(monkeypatch, v):
    import lumenairy
    monkeypatch.setattr(lumenairy, "__version__", v)


def _feature_design(incs, *, bg="lo", tensor_layer=False):
    from dynameta.geometry import Design, Layer, Stack, UnitCell
    from dynameta.geometry.specs import OpticalSpec
    from dynameta.materials import ConstantOptical, Material, MaterialRegistry
    reg = MaterialRegistry()
    for n, e in (("air", 1.0), ("glass", 2.25), ("hi", 6.0 + 0.4j), ("lo", 2.1), ("mid", 3.2)):
        reg.add(Material(n, ConstantOptical(complex(e))))
    layers = [Layer("g", 200e-9, bg, inclusions=incs)]
    return Design(name="ff", unit_cell=UnitCell(period_x_m=500e-9, period_y_m=500e-9),
                  stack=Stack(layers=layers, superstrate_material="air",
                              substrate_material="glass"),
                  electrodes=[], materials=reg,
                  optical=OpticalSpec(polarization="y", incidence_angle_deg=0.0))


@pytest.mark.skipif(not HAVE_LUM, reason="lumenairy not installed")
def test_rcwa_analytic_shapes_floor_auto_rasterizes_and_analytic_refuses(monkeypatch):
    from dynameta.geometry import Inclusion
    from dynameta.geometry.cross_section import Circle, Ellipse, Rectangle
    from dynameta.optics.lumenairy_bridge import (LumenairyFeatureUnavailable,
                                                  LumenairyStackSolver, design_to_rcwa_stack,
                                                  make_lumenairy_rcwa_solver)
    from dynameta.optics.lumenairy_bridge._shapes import layer_to_shapes_slab
    from dynameta.core.layered import LayeredStack
    ok = _feature_design([Inclusion(Rectangle(120e-9, 250e-9, 150e-9, 300e-9), "hi"),
                          Inclusion(Ellipse(360e-9, 250e-9, 80e-9, 160e-9), "mid")])
    over = _feature_design([Inclusion(Circle(200e-9, 250e-9, 100e-9), "hi"),
                            Inclusion(Circle(300e-9, 250e-9, 100e-9), "mid", priority=1)],
                           bg="air")
    _spoof(monkeypatch, "5.30.0")
    for d in (ok, over):                       # 'auto' (the default) rasterizes below 5.31
        stk, _ = design_to_rcwa_stack(d, 1.31e-6, n_orders=3)
        assert [L.kind for L in stk.layers] == ["iso"]
    with pytest.raises(LumenairyFeatureUnavailable,
                       match=r"needs lumenairy >= 5\.31\.0 \(installed 5\.30\.0\)"):
        design_to_rcwa_stack(ok, 1.31e-6, n_orders=3, inclusion_model="analytic")
    with pytest.raises(LumenairyFeatureUnavailable, match=r">= 5\.31\.0"):
        make_lumenairy_rcwa_solver(inclusion_model="analytic")
    slab = layer_to_shapes_slab(ok.stack.layers[0], ok, 1.31e-6)
    ls = LayeredStack(slabs=[slab], n_super=1.0, n_sub=1.5, period_x_m=500e-9,
                      period_y_m=500e-9)
    with pytest.raises(LumenairyFeatureUnavailable, match="shapes slab"):
        LumenairyStackSolver(n_orders=3).solve(ls, 1.31e-6, ok.optical)
    _spoof(monkeypatch, "5.31.0")              # at the floor: the analytic layer is back
    stk, _ = design_to_rcwa_stack(ok, 1.31e-6, n_orders=3)
    assert [L.kind for L in stk.layers] == ["shapes"]


@pytest.mark.skipif(not HAVE_LUM, reason="lumenairy not installed")
def test_pmm2d_pure_feature_floors_refuse_with_the_version(monkeypatch):
    from dynameta.core.eps_field import EpsField
    from dynameta.geometry import Inclusion, Taper
    from dynameta.geometry.cross_section import Circle, Polygon, Rectangle
    from dynameta.optics.lumenairy_bridge import (LumenairyFeatureUnavailable,
                                                  design_to_pmm2d_stack)
    circ = _feature_design([Inclusion(Circle(250e-9, 250e-9, 120e-9), "hi")])
    sq = Rectangle(250e-9, 250e-9, 250e-9, 250e-9)
    tap = _feature_design([Inclusion(sq, "hi", taper=Taper(bottom=Rectangle(
        250e-9, 250e-9, 200e-9, 200e-9)))])
    tens = _feature_design([Inclusion(sq, "hi")])
    ebr = {"g": EpsField(tensor=(2.0 * np.eye(3) + 0.1 * np.eye(3)[[1, 0, 2]]).astype(complex))}
    _spoof(monkeypatch, "5.49.0")
    with pytest.raises(LumenairyFeatureUnavailable,
                       match=r"'auto' -> 'shapes'.*needs lumenairy >= 5\.50\.0 \(installed 5\.49\.0\)"):
        design_to_pmm2d_stack(circ, 1.31e-6, n_modes=4)
    # a user error (no exact pure image) still outranks the version refusal
    tri = _feature_design([Inclusion(Polygon([(1e-7, 1e-7), (4e-7, 1.2e-7), (2.5e-7, 4e-7)]),
                                     "hi")])
    with pytest.raises(ValueError, match="no EXACT image"):
        design_to_pmm2d_stack(tri, 1.31e-6, n_modes=4)
    _spoof(monkeypatch, "5.44.0")
    with pytest.raises(LumenairyFeatureUnavailable, match=r"per-layer.*>= 5\.45\.0"):
        design_to_pmm2d_stack(tap, 1.31e-6, n_modes=4)
    with pytest.raises(LumenairyFeatureUnavailable, match=r"per-layer.*>= 5\.45\.0"):
        design_to_pmm2d_stack(tens, 1.31e-6, n_modes=4, pure_grid="per-layer")
    _spoof(monkeypatch, "5.42.1")              # union grid builds on 5.22+, a tensor does not
    design_to_pmm2d_stack(tens, 1.31e-6, n_modes=4)
    with pytest.raises(LumenairyFeatureUnavailable,
                       match=r"TENSOR eps on layer 'g'.*>= 5\.43\.0 \(installed 5\.42\.1\)"):
        design_to_pmm2d_stack(tens, 1.31e-6, n_modes=4, eps_by_region=ebr)
    # the hybrid engine's eps_tensor_cell is 5.22-era: no refusal
    design_to_pmm2d_stack(tens, 1.31e-6, engine="hybrid", degree=5, n_orders=3, eps_by_region=ebr)


@pytest.mark.skipif(not HAVE_LUM, reason="lumenairy not installed")
def test_feature_floor_table_is_sound():
    from dynameta.optics.lumenairy_bridge._common import (FEATURE_FLOORS, VERSION_FLOOR,
                                                          lumenairy_has, lumenairy_version)
    for name, (floor, what) in FEATURE_FLOORS.items():
        assert tuple(floor) > tuple(VERSION_FLOOR), name     # else it is not a feature floor
        assert lumenairy_has(name) == (lumenairy_version() >= floor)
        assert what
