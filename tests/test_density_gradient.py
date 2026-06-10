"""Tests for the R19 density-gradient post-hoc quantum correction. The Schrodinger-Poisson
dead-layer oracle lives in validation/density_gradient_dead_layer.py."""
import numpy as np
import pytest

from dynameta.constants import HBAR, M_E, Q_E as Q
from dynameta.carriers.density_gradient import (dg_correct_density_1d, dg_length_m,
                                                quantum_potential_V)

MSTAR = 0.35 * M_E


def test_quantum_potential_gaussian_closed_form():
    s = 3e-9
    z = np.linspace(-12e-9, 12e-9, 1601)
    n = 1e26 * np.exp(-z ** 2 / (2.0 * s ** 2))
    lam = quantum_potential_V(z, n, MSTAR)
    b = HBAR ** 2 / (6.0 * MSTAR * Q)
    cf = b * (z ** 2 / (4.0 * s ** 4) - 1.0 / (2.0 * s ** 2))
    inner = np.abs(z) < 8e-9
    assert np.max(np.abs(lam[inner] - cf[inner])) < 1e-3 * np.max(np.abs(cf[inner]))


def test_dg_off_switch_and_dead_layer():
    z = np.linspace(0.0, 15e-9, 901)
    n = np.full_like(z, 2e26)
    assert np.array_equal(dg_correct_density_1d(z, n, MSTAR, gamma=0.0), n)
    n_dg = dg_correct_density_1d(z, n, MSTAR)
    lq = dg_length_m(MSTAR)
    assert n_dg[0] < 1e-3 * 2e26                          # hard wall
    assert abs(n_dg[-1] / 2e26 - 1.0) < 1e-4              # bulk recovered
    assert 0.8e-9 < lq < 1.8e-9                           # the ~1 nm ITO dead-layer scale


def test_dg_guards():
    z = np.linspace(0.0, 10e-9, 301)
    with pytest.raises(ValueError):
        dg_correct_density_1d(z, np.zeros_like(z), MSTAR)
    with pytest.raises(ValueError):
        dg_correct_density_1d(z, np.full_like(z, 1e26), MSTAR, hard_wall="middle")
    with pytest.raises(ValueError):
        quantum_potential_V(z, np.full(300, 1e26), MSTAR)


# ---- R19 follow-on: in-Newton DG-DD (module-level guards; the 4-variable Newton oracle is
# validation/dg_dd_in_newton.py) -----------------------------------------------------------------

def test_dg_b_coefficient_and_guards():
    pytest.importorskip("devsim")
    from dynameta.carriers.physics_density_gradient import dg_b_coefficient, set_dg_gamma
    b = dg_b_coefficient(MSTAR, 1.0)
    assert b == pytest.approx(HBAR ** 2 / (6.0 * MSTAR * Q), rel=1e-14)
    assert dg_b_coefficient(MSTAR, 0.0) == 0.0
    with pytest.raises(ValueError):
        dg_b_coefficient(-1.0)
    with pytest.raises(ValueError):
        set_dg_gamma("d", "r", 1.5)                      # frac outside [0, 1]


def test_dg_contact_guard():
    pytest.importorskip("devsim")
    from dynameta.carriers.physics_density_gradient import setup_contact_dg
    with pytest.raises(ValueError):
        setup_contact_dg("dev", "c", 0.0)
