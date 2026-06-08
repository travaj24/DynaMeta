"""Smoke + reduces-to-analytic coverage for the NGSolve FEM field drivers (carriers.thermal_fem,
carriers.electrostatics_fem) -- previously exercised only by validation/ (no pytest). Skipped when
ngsolve is absent (the numpy-only CI leg); the rigorous oracles live in validation/thermal_fem.py +
validation/electrostatics_fem.py."""
import numpy as np
import pytest

pytest.importorskip("ngsolve")

from dynameta.carriers.thermal_fem import ThermalLayer, solve_thermal_fem
from dynameta.carriers.electrostatics_fem import ElectrostaticLayer, solve_electrostatics_fem


def test_thermal_fem_single_layer_series_resistance():
    # one conducting layer, bottom sink + top inflow flux, no source: T is linear with
    # dT = flux*L/k across the layer, so the volume-mean rise is flux*L/(2k).
    L, k, flux, Tsink = 100e-9, 1.5, 1.0e8, 300.0
    res = solve_thermal_fem([ThermalLayer("slab", L, k)], period_x_m=60e-9, period_y_m=60e-9,
                            flux_W_m2=flux, T_sink_K=Tsink, maxh_m=30e-9, order=2)
    mean_rise = float(res.mean_T_per_layer()[0]) - Tsink
    assert abs(mean_rise - flux * L / (2.0 * k)) / (flux * L / (2.0 * k)) < 0.05   # ~3.33 K


def test_electrostatics_fem_series_capacitor_field():
    # two dielectric layers in series: the displacement D = eps0 eps_r E_z is continuous, so
    # eps1*E1 == eps2*E2, and the field drops sum to the applied voltage.
    e1, e2, t = 4.0, 16.0, 50e-9
    V = 1.0
    res = solve_electrostatics_fem([ElectrostaticLayer("d1", t, e1), ElectrostaticLayer("d2", t, e2)],
                                   V, period_x_m=60e-9, period_y_m=60e-9, maxh_m=30e-9, order=2)
    Ez = res.mean_Ez_per_layer()
    D1, D2 = e1 * Ez[0], e2 * Ez[1]
    assert abs(D1 - D2) / abs(D1) < 0.03                              # D continuous across the interface
    assert abs(abs(Ez[0]) * t + abs(Ez[1]) * t - V) / V < 0.03        # series fields sum to applied V
