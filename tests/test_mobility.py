"""Fast unit tests for the field/density-dependent mobility closure (carriers/mobility.py, roadmap R1).
Pure numpy, no DEVSIM; the DEVSIM J-V oracle lives in validation/dd_field_mobility.py."""
import math

import numpy as np
import pytest

from dynameta.carriers.mobility import (
    MasettiParams, masetti_mu_low, caughey_thomas, drift_velocity, mu_edge_expr_devsim)


def test_caughey_thomas_zero_field_recovers_mu_low():
    mu_low = 0.05
    assert caughey_thomas(mu_low, 0.0, 1e5, beta=2.0) == pytest.approx(mu_low, rel=1e-15)
    # v_sat = inf is the off-switch -> constant mu_low at any field
    assert caughey_thomas(mu_low, 1e8, float("inf"), beta=2.0) == pytest.approx(mu_low, rel=1e-15)


def test_drift_velocity_saturates_to_vsat():
    mu_low, v_sat = 0.05, 1.0e5
    v_hi = drift_velocity(1.0e9, mu_low, v_sat, beta=2.0)        # E >> v_sat/mu_low
    assert v_hi == pytest.approx(v_sat, rel=1e-3)               # v -> v_sat
    # monotone increasing, always below v_sat
    E = np.array([1e4, 1e5, 1e6, 1e7, 1e8])
    v = drift_velocity(E, mu_low, v_sat, beta=2.0)
    assert np.all(np.diff(v) > 0) and np.all(v < v_sat)


def test_canali_beta2_closed_form():
    mu_low, v_sat, E = 0.04, 2.0e5, 3.0e6
    x = mu_low * E / v_sat
    expect = mu_low / math.sqrt(1.0 + x * x)                     # beta=2 -> 1/sqrt(1+x^2)
    assert float(caughey_thomas(mu_low, E, v_sat, beta=2.0)) == pytest.approx(expect, rel=1e-14)


def test_masetti_monotone_and_dilute_limit():
    N = np.array([1e20, 1e22, 1e24, 1e26])
    mu = masetti_mu_low(N)
    assert np.all(np.diff(mu) < 0)                              # decreasing with doping
    assert float(masetti_mu_low(1.0)) == pytest.approx(MasettiParams().mu_max, rel=1e-6)


def test_devsim_string_matches_numpy_smoothed():
    # the emitted DEVSIM edge string must encode EXACTLY the smoothed Caughey-Thomas factor (anti-drift).
    mu_low, v_sat, es, beta = 0.05, 1.5e5, 1.0, 2.0
    expr = mu_edge_expr_devsim(mu_low="mu_n", efield="EF", v_sat="vs", e_smooth="es", beta=beta)
    pyexpr = expr.replace("^", "**")
    for E in (0.0, 1.0e3, 5.0e6, 2.0e8):
        env = {"pow": pow, "mu_n": mu_low, "EF": E, "vs": v_sat, "es": es}
        val = eval(pyexpr, {"__builtins__": {}}, env)           # noqa: S307 (trusted, our own string)
        Eabs = math.sqrt(E * E + es * es)
        ref = mu_low / (1.0 + (mu_low * Eabs / v_sat) ** beta) ** (1.0 / beta)
        assert val == pytest.approx(ref, rel=1e-12)
    # smoothed magnitude -> abs() form (caughey_thomas) for E >> e_smooth
    big = 1.0e6
    sm = mu_low / (1.0 + (mu_low * math.sqrt(big * big + es * es) / v_sat) ** beta) ** (1.0 / beta)
    assert float(caughey_thomas(mu_low, big, v_sat, beta)) == pytest.approx(sm, rel=1e-8)
