"""Fast unit tests for the topology-optimization primitives (filter / projection / eps interp; no FDTD)."""
import numpy as np
import pytest

pytest.importorskip("jax")

import jax.numpy as jnp

from dynameta.optics.topology_opt import binarization, density_filter, eps_from_density, project


def test_project_endpoints_and_threshold_fixed_point():
    assert abs(float(project(jnp.array(0.0), 8.0))) < 1e-6
    assert abs(float(project(jnp.array(1.0), 8.0)) - 1.0) < 1e-6
    assert abs(float(project(jnp.array(0.5), 8.0)) - 0.5) < 1e-6      # eta=0.5 is the fixed point
    # sharper beta pushes a near-1 value closer to 1
    assert float(project(jnp.array(0.7), 16.0)) > float(project(jnp.array(0.7), 1.0))


def test_filter_preserves_uniform():
    rho = jnp.ones((8, 6)) * 0.3
    assert np.max(np.abs(np.asarray(density_filter(rho, 2.0)) - 0.3)) < 1e-9


def test_filter_smooths_and_is_periodic_in_x():
    spike = jnp.zeros((8, 6)).at[0, 3].set(1.0)
    f = np.asarray(density_filter(spike, 2.0))
    assert 0.0 < f[0, 3] < 1.0 and f[1, 3] > 0.0            # spread to z-neighbour
    assert f[7, 3] > 0.0                                    # wraps to last x row (periodic in x)


def test_filter_clamped_in_z_no_wrap():
    spike = jnp.zeros((8, 6)).at[0, 0].set(1.0)            # at the z=0 edge
    f = np.asarray(density_filter(spike, 2.0))
    assert f[0, 5] == 0.0                                   # does NOT wrap to the far z edge


def test_eps_from_density_endpoints():
    assert float(eps_from_density(jnp.array(0.0), 1.0, 12.0)) == 1.0
    assert float(eps_from_density(jnp.array(1.0), 1.0, 12.0)) == 12.0


def test_binarization_score():
    assert binarization(np.array([0.0, 1.0, 0.02, 0.99])) == 1.0
    assert binarization(np.array([0.5, 0.5])) == 0.0
