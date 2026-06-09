"""Unit coverage for the shared trapezoidal integrator (core.numerics.trapz) -- previously used
library-wide but never directly tested (audit xcut-1). Pure numpy."""
import numpy as np
import pytest

from dynameta.core.numerics import trapz


def test_trapz_line():
    x = np.linspace(0.0, 1.0, 101)
    assert trapz(x, x) == pytest.approx(0.5, abs=1e-6)        # integral of y=x over [0,1]


def test_trapz_constant():
    x = np.linspace(0.0, 2.0, 51)
    assert trapz(np.full_like(x, 3.0), x) == pytest.approx(6.0, abs=1e-9)


def test_trapz_empty_and_single_are_zero():
    assert trapz(np.array([]), np.array([])) == 0.0
    assert trapz(np.array([1.0]), np.array([0.0])) == 0.0    # one node -> no interval


def test_trapz_shape_mismatch_raises():
    with pytest.raises(ValueError):                          # numpy broadcast failure (2 vs 4)
        trapz(np.array([1.0, 2.0, 3.0]), np.array([0.0, 1.0, 2.0, 3.0, 4.0]))
