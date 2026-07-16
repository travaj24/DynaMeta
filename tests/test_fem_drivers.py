"""Smoke + reduces-to-analytic coverage for the NGSolve FEM field drivers (carriers.thermal_fem,
carriers.electrostatics_fem) -- previously exercised only by validation/ (no pytest). Skipped when
ngsolve is absent (the numpy-only CI leg); the rigorous oracles live in validation/thermal_fem.py +
validation/electrostatics_fem.py."""
import numpy as np
import pytest

pytest.importorskip("ngsolve")

from dynameta.carriers.thermal_fem import (ThermalLayer, solve_thermal_fem,
                                           solve_thermal_transient_fem)
from dynameta.carriers.electrostatics_fem import ElectrostaticLayer, solve_electrostatics_fem


def test_thermal_fem_single_layer_series_resistance():
    # one conducting layer, bottom sink + top inflow flux, no source: T is linear with
    # dT = flux*L/k across the layer, so the volume-mean rise is flux*L/(2k).
    L, k, flux, Tsink = 100e-9, 1.5, 1.0e8, 300.0
    res = solve_thermal_fem([ThermalLayer("slab", L, k)], period_x_m=60e-9, period_y_m=60e-9,
                            flux_W_m2=flux, T_sink_K=Tsink, maxh_m=30e-9, order=2)
    mean_rise = float(res.mean_T_per_layer()[0]) - Tsink
    assert abs(mean_rise - flux * L / (2.0 * k)) / (flux * L / (2.0 * k)) < 0.05   # ~3.33 K


def test_thermal_transient_reaches_steady():
    # coarse single slab, uniform Joule, uniform IC=T_sink: a few large backward-Euler steps must
    # converge to the steady uniform-Joule profile T_mean = T_sink + Q L^2/(3k) within ~5%.
    L, k, Q, Tsink = 100e-9, 10.0, 5.0e15, 300.0
    slab = [ThermalLayer("slab", L, k, rho_kg_m3=7140.0, Cp_J_kgK=340.0)]   # ITO-like rho*Cp
    tr = solve_thermal_transient_fem(slab, period_x_m=60e-9, period_y_m=60e-9, t_end_s=2e-7,
                                     dt_s=1e-8, flux_W_m2=0.0, T_sink_K=Tsink, joule_W_m3=Q,
                                     maxh_m=30e-9, order=2)
    t_steady = Tsink + Q * L ** 2 / (3.0 * k)
    mean_final = float(tr.mean_T_per_layer()[0])
    assert abs(mean_final - t_steady) / abs(t_steady - Tsink) < 0.05
    # monotone rise (theta=1, no overshoot) + sink pinned at the final field
    assert float(np.min(np.diff(tr.mean_T_per_layer_t[:, 0]))) > -1e-6
    assert abs(tr.T_at(30e-9, 30e-9, 0.0) - Tsink) < 1e-6


def test_thermal_transient_requires_rho_cp():
    # rho/Cp default 0 -> the transient cannot run (would be a singular mass matrix); must raise.
    slab = [ThermalLayer("slab", 100e-9, 10.0)]                              # no rho/Cp
    with pytest.raises(ValueError):
        solve_thermal_transient_fem(slab, period_x_m=60e-9, period_y_m=60e-9, t_end_s=1e-7,
                                    dt_s=1e-8, flux_W_m2=1e8)


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


def test_split_gate_tddb_stress_statistic():
    # audit C4-9: a +/-V split gate has signed layer-mean Ez ~ 0 (the pre-fix TDDB
    # adapter reported ~zero stress -> exponentially overstated t_BD, silently); the
    # sign-robust mean|Ez| statistic must see the ~3 MV/cm field
    ng = pytest.importorskip("ngsolve")
    import warnings
    from types import SimpleNamespace
    from dynameta.carriers.electrostatics_fem import (ElectrostaticLayer,
                                                      solve_electrostatics_fem)
    from dynameta.carriers.fem_mesh import _S
    from dynameta.reliability.tddb import oxide_stress_from_electrothermal
    lay = [ElectrostaticLayer("hfo2", 10e-9, 18.0)]
    P = 300e-9
    vcf = ng.IfPos(ng.x / (_S * P) - 0.5, 3.0, -3.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = solve_electrostatics_fem(lay, 3.0, period_x_m=P, period_y_m=P, top_voltage_cf=vcf)
    am = r.mean_absEz_per_layer()[0]
    assert am > 1e8                                           # ~3 MV/cm seen
    assert abs(r.mean_Ez_per_layer()[0]) < 0.05 * am          # the signed mean cancels
    et = SimpleNamespace(layers=lay, E_result=r, T_per_layer=[300.0])
    ez, T = oxide_stress_from_electrothermal(et, "hfo2")
    assert ez == pytest.approx(am)                            # adapter uses the C4-9 stat
