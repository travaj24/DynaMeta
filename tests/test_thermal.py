"""Unit coverage for the Phase-2 thermal driver (carriers/thermal) -- the series-thermal-
resistance steady temperature profile that drives the thermo-optic effect. Pure numpy.
Run: python -m pytest tests/test_thermal.py -q
"""
import numpy as np
import pytest

from dynameta.carriers.thermal import steady_layered_temperature, uniform_temperature_rise


def test_series_thermal_resistance_profile():
    k = np.array([1.4, 130.0, 50.0])         # e.g. SiO2 / Si / something [W/(m.K)]
    d = np.array([100e-9, 200e-9, 50e-9])
    q, T0 = 1.0e6, 300.0                       # heat flux [W/m^2], sink temperature [K]
    T = steady_layered_temperature(k, d, q, T_sink_K=T0)
    R = d / k
    R_below = np.concatenate([[0.0], np.cumsum(R)[:-1]])
    assert np.allclose(T, T0 + q * (R_below + 0.5 * R))   # series-resistance mean-T formula
    assert T[0] < T[1] < T[2]                              # warmer away from the sink
    assert np.all(T >= T0)


def test_uniform_temperature_rise_is_q_times_Rtotal():
    T = uniform_temperature_rise(1.0e6, [1.4], [100e-9], T_sink_K=300.0)
    assert T == pytest.approx(300.0 + 1.0e6 * (100e-9 / 1.4))


def test_thermal_rejects_bad_input():
    with pytest.raises(ValueError):
        steady_layered_temperature([1.4, 2.0], [100e-9], 1e6)     # mismatched lengths
    with pytest.raises(ValueError):
        steady_layered_temperature([0.0], [100e-9], 1e6)          # non-positive k
