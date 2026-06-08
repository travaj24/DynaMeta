"""Fast unit test for the multi-objective inverse-design loss combiner (the FDTD forward / full
optimization is covered by validation/fdtd_multiobjective_design.py)."""
import pytest

pytest.importorskip("jax")

import jax
import jax.numpy as jnp

from dynameta.optics.inverse_design import weighted_objective


def test_weighted_objective_combines_max_min_target():
    loss = weighted_objective([
        {"value": lambda p: p[0], "sense": "max", "weight": 2.0},      # reward p0 (subtract 2*p0)
        {"value": lambda p: p[1], "sense": "min", "weight": 1.0},      # penalise p1 (add 1*p1)
        {"value": lambda p: p[2], "target": 0.5, "weight": 3.0}])      # drive p2 -> 0.5 (add 3*(p2-0.5)^2)
    p = jnp.array([0.8, 0.3, 0.7])
    # -2*0.8 + 1*0.3 + 3*(0.2)^2 = -1.6 + 0.3 + 0.12 = -1.18
    assert abs(float(loss(p)) - (-1.18)) < 1e-6
    g = jax.grad(loss)(p)
    assert bool(jnp.all(jnp.isfinite(g)))
    # the 'max' term pushes p0 up (grad < 0), the 'min' term pushes p1 down (grad > 0)
    assert float(g[0]) < 0.0 and float(g[1]) > 0.0
