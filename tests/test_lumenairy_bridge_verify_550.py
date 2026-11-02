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
