"""Unit coverage for the Phase-1 electrostatic driver (carriers/electrostatics) -- the
series-capacitor layered static field that drives the EO effects. Pure numpy.
Run: python -m pytest tests/test_electrostatics.py -q
"""
import numpy as np
import pytest

from dynameta.carriers.electrostatics import layered_static_field_z, parallel_plate_field_z


def test_single_layer_is_V_over_gap():
    # one layer: E = -V/d, independent of eps
    E = layered_static_field_z([6.0], [200e-9], 2.0)
    assert E.shape == (1,)
    assert E[0] == pytest.approx(-2.0 / 200e-9)
    assert parallel_plate_field_z(6.0, 200e-9, 2.0) == pytest.approx(-2.0 / 200e-9)


def test_series_capacitor_relations():
    # D = eps_i E_i continuous; V = sum E_i d_i
    eps = np.array([4.0, 18.0, 9.0])          # e.g. oxide / high-k / semiconductor
    d = np.array([50e-9, 10e-9, 200e-9])
    V = 3.0
    E = layered_static_field_z(eps, d, V)
    # (1) displacement D/eps0 = eps_i * E_i is the SAME in every layer
    D = eps * E
    assert np.allclose(D, D[0])
    # (2) the potential drop sums back to -V (E is signed along -z for V>0)
    assert np.sum(E * d) == pytest.approx(-V)
    # (3) the thinnest high-eps layer carries the smallest field (series divider)
    assert abs(E[1]) < abs(E[0]) and abs(E[1]) < abs(E[2])


def test_rejects_bad_input():
    with pytest.raises(ValueError):
        layered_static_field_z([6.0, 4.0], [200e-9], 1.0)      # mismatched lengths
    with pytest.raises(ValueError):
        layered_static_field_z([6.0, -1.0], [200e-9, 50e-9], 1.0)  # non-positive eps
    with pytest.raises(ValueError):
        parallel_plate_field_z(6.0, 0.0, 1.0)                  # zero gap
