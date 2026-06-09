"""Smoke + guard tests for the self-consistent electro-thermo-optic Picard driver
(carriers.electrothermal, roadmap R6). Skipped when ngsolve is absent (numpy-only CI leg); the
rigorous oracle lives in validation/electrothermal_picard.py."""
import numpy as np
import pytest

pytest.importorskip("ngsolve")

from dynameta.carriers.electrothermal import (ElectroThermalLayer, solve_electrothermal_picard,
                                              electrothermal_extra_fields)

PERIOD = 400e-9


def _layers(sig):
    return [ElectroThermalLayer("ox", 60e-9, 4.0, 1.4, sigma_S_m=0.0),
            ElectroThermalLayer("cond", 40e-9, 9.0, 5.0, sigma_S_m=sig)]


def test_const_sigma_converges_in_two_iters():
    res = solve_electrothermal_picard(_layers(1e2), 1.0, period_x_m=PERIOD, period_y_m=PERIOD,
                                      max_iter=6, tol_T_K=1e-3, maxh_m=40e-9)
    assert res.converged and res.n_iter == 2          # iter 1 sets T, iter 2 confirms no change
    assert res.residual_history[-1] < 1e-9            # machine no-op on the fixed mesh
    assert res.T_per_layer[1] > 300.0                 # the heated conductive layer is above the sink
    assert np.isfinite(res.total_sink_outflux_W)


def test_single_pass_does_not_raise():
    # max_iter=1 is an explicit weak-coupling single pass: returns (converged may be False), no raise.
    res = solve_electrothermal_picard(_layers(1e2), 1.0, period_x_m=PERIOD, period_y_m=PERIOD,
                                      max_iter=1, maxh_m=40e-9)
    assert res.n_iter == 1


def test_energy_balance():
    res = solve_electrothermal_picard([ElectroThermalLayer("slab", 80e-9, 9.0, 5.0, sigma_S_m=1e2)],
                                      1.0, period_x_m=PERIOD, period_y_m=PERIOD, max_iter=6,
                                      tol_T_K=1e-4, maxh_m=20e-9)
    assert abs(res.total_joule_W - res.total_sink_outflux_W) / abs(res.total_joule_W) < 1e-2


@pytest.mark.parametrize("kw", [
    dict(max_iter=0), dict(relax=0.0), dict(relax=1.5), dict(tol_T_K=0.0), dict(T_sink_K=0.0)])
def test_bad_input_raises(kw):
    base = dict(period_x_m=PERIOD, period_y_m=PERIOD, maxh_m=40e-9)
    base.update(kw)
    with pytest.raises(ValueError):
        solve_electrothermal_picard(_layers(1e2), 1.0, **base)


def test_empty_layers_raises():
    with pytest.raises(ValueError):
        solve_electrothermal_picard([], 1.0, period_x_m=PERIOD, period_y_m=PERIOD)


def test_negative_sigma_raises():
    with pytest.raises(ValueError):
        solve_electrothermal_picard(_layers(lambda T: -1.0), 1.0, period_x_m=PERIOD,
                                    period_y_m=PERIOD, max_iter=3, maxh_m=40e-9)


def test_extra_fields_closure_bad_region():
    with pytest.raises(ValueError):
        electrothermal_extra_fields(_layers(1e2), period_x_m=PERIOD, period_y_m=PERIOD,
                                    voltage_of_bias=lambda bp: 1.0, optical_region="nope")
