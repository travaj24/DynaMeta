"""Fast unit tests for the inverse-design Adam driver (analytic losses, no FDTD run)."""
import numpy as np
import pytest

pytest.importorskip("jax")

from dynameta.optics.inverse_design import optimize_fdtd


def test_adam_minimizes_scalar_quadratic():
    popt, hist = optimize_fdtd(lambda p: (p - 3.0) ** 2 + 1.0, 0.0, n_steps=400, lr=0.1)
    assert abs(float(np.asarray(popt)) - 3.0) < 1e-2
    assert hist[-1] < hist[0] and hist[-1] < 1.01            # reaches the minimum value 1.0


def test_adam_minimizes_vector_quadratic_with_clip():
    import jax.numpy as jnp
    target = jnp.array([1.0, -2.0, 3.5])
    popt, hist = optimize_fdtd(lambda p: jnp.sum((p - target) ** 2), jnp.zeros(3),
                               n_steps=500, lr=0.1, clip=(-5.0, 5.0))
    assert np.max(np.abs(np.asarray(popt) - np.asarray(target))) < 1e-2
    assert hist[-1] < hist[0]
