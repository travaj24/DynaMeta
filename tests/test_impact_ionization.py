"""Solver-free tests for the Chynoweth impact-ionization driver (D4). The DEVSIM-backed I_sub
oracles (constant-field bar closed form, reverse-bias junction growth, mesh convergence) live in
validation/impact_ionization_isub.py."""
import math

import numpy as np
import pytest

from dynameta.carriers.impact_ionization import ChynowethParams, SILICON_VANOVERSTRAETEN


def test_alpha_closed_form_and_masking():
    si = SILICON_VANOVERSTRAETEN
    for E in (3.0e7, 6.0e7, 1.2e8):
        assert si.alpha_n(E) == pytest.approx(7.03e7 * math.exp(-1.231e8 / E), rel=1e-12)
        assert si.alpha_p(E) == pytest.approx(1.582e8 * math.exp(-2.036e8 / E), rel=1e-12)
    assert si.alpha_n(0.0) == 0.0
    assert si.alpha_p(-1.0) == 0.0
    arr = si.alpha_n(np.array([0.0, 5.0e7, -2.0]))
    assert arr[0] == 0.0 and arr[2] == 0.0 and arr[1] == si.alpha_n(5.0e7)


def test_si_constants_are_si_not_cgs():
    # the literature cm trap is a factor 100 either way; alpha_n(3e7 V/m) ~ 1.2e6 1/m
    a = SILICON_VANOVERSTRAETEN.alpha_n(3.0e7)
    assert 1.0e5 < a < 1.0e7


def test_alpha_monotone_increasing_in_field():
    si = SILICON_VANOVERSTRAETEN
    E = np.linspace(2.0e7, 2.0e8, 50)
    assert np.all(np.diff(si.alpha_n(E)) > 0.0)
    assert np.all(np.diff(si.alpha_p(E)) > 0.0)


def test_guards():
    with pytest.raises(ValueError):
        ChynowethParams(a_n_per_m=-1.0, b_n_V_per_m=1.0, a_p_per_m=0.0, b_p_V_per_m=1.0)
    pytest.importorskip("devsim")
    from dynameta.carriers.impact_ionization import substrate_current
    with pytest.raises(ValueError):
        substrate_current("dev", "reg", SILICON_VANOVERSTRAETEN, depth_m=0.0)
