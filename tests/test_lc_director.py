"""Unit coverage for the Phase-4 liquid-crystal director driver (carriers.lc_director) -- the
1-constant Frank-elastic Freedericksz transition. Pure numpy/scipy. Run:
python -m pytest tests/test_lc_director.py -q
"""
import numpy as np
import pytest

from dynameta.constants import EPS0
from dynameta.carriers.lc_director import freedericksz_threshold_V, director_profile

K, DEPS, EP, D = 6.5e-12, 11.0, 7.0, 5e-6     # ~5CB nematic cell


def test_threshold_matches_analytic():
    Vth = freedericksz_threshold_V(K, DEPS)
    assert Vth == pytest.approx(np.pi * np.sqrt(K / (EPS0 * DEPS)), rel=1e-12)


def test_threshold_requires_positive_anisotropy():
    with pytest.raises(ValueError):
        freedericksz_threshold_V(K, 0.0)
    with pytest.raises(ValueError):
        freedericksz_threshold_V(K, -2.0)                  # negative anisotropy -> no planar threshold


def test_planar_below_threshold_and_tilt_above():
    Vth = freedericksz_threshold_V(K, DEPS)
    assert director_profile(K, DEPS, EP, D, 0.5 * Vth).theta_max_rad == 0.0      # below -> planar
    assert director_profile(K, DEPS, EP, D, 0.99 * Vth).theta_max_rad == 0.0
    assert director_profile(K, DEPS, EP, D, 1.1 * Vth).theta_max_rad > 0.0       # above -> tilts


def test_tilt_monotonic_in_voltage_and_saturates():
    Vth = freedericksz_threshold_V(K, DEPS)
    tm = [director_profile(K, DEPS, EP, D, r * Vth).theta_max_rad for r in (1.1, 1.5, 2.0, 4.0)]
    assert all(b > a for a, b in zip(tm, tm[1:]))          # monotonic increasing
    assert tm[-1] < 0.5 * np.pi and tm[-1] > np.radians(80)   # saturates toward homeotropic, < 90


def test_threshold_independent_of_thickness():
    # V_th must not depend on cell thickness (classic Freedericksz result)
    Vth = freedericksz_threshold_V(K, DEPS)
    t_thin = director_profile(K, DEPS, EP, 2e-6, 1.5 * Vth).theta_max_rad
    t_thick = director_profile(K, DEPS, EP, 20e-6, 1.5 * Vth).theta_max_rad
    assert t_thin == pytest.approx(t_thick, rel=1e-3)      # same tilt at the same V/V_th


def test_supercritical_sqrt_law_near_threshold():
    Vth = freedericksz_threshold_V(K, DEPS)
    rs = np.array([1.02, 1.05, 1.10])
    tm = np.array([director_profile(K, DEPS, EP, D, r * Vth).theta_max_rad for r in rs])
    ratio = tm ** 2 / (rs - 1.0)                            # theta_max^2 ~ (V/Vth - 1) => ~const
    assert np.all(ratio > 0) and (ratio.max() / ratio.min() < 1.3)


def test_planar_anchoring_boundary_conditions():
    Vth = freedericksz_threshold_V(K, DEPS)
    p = director_profile(K, DEPS, EP, D, 1.6 * Vth, nz=201)
    assert abs(p.theta_rad[0]) < 1e-9 and abs(p.theta_rad[-1]) < 1e-9   # theta=0 at both plates
    assert p.theta_max_rad == pytest.approx(np.max(p.theta_rad))
    assert p.theta_rad.argmax() in (p.theta_rad.size // 2, p.theta_rad.size // 2 - 1,
                                    p.theta_rad.size // 2 + 1)          # peak at the midplane


def test_director_profile_rejects_bad_input():
    with pytest.raises(ValueError):
        director_profile(0.0, DEPS, EP, D, 1.0)            # K <= 0
    with pytest.raises(ValueError):
        director_profile(K, DEPS, EP, D, 1.0, nz=5)        # nz too small
