"""Fast pure-numpy unit tests for the reliability post-processors (dynameta.reliability,
REL1/REL2/REL4/REL10). The rigorous oracles live in validation/reliability_*.py."""
import numpy as np
import pytest

from dynameta.reliability import (TddbParams, tbd_e_model, tbd_one_over_e, weibull_area_scale,
                                  oxide_stress_from_electrothermal, BtiParams, dvth_power_law,
                                  time_to_dvth, DedopingParams, carrier_decay, enz_wavelength_m,
                                  arrhenius_af, field_af, mttf_use_from_stress, system_mttf,
                                  fit_per_1e9_hours, weibull_earliest_t63)

MV = 1.0e8


# ---- REL1 TDDB ----

def test_tddb_field_and_temp_accelerate():
    p = TddbParams(tau0_s=1.0)
    assert p.tbd_s(8 * MV, 300.0) < p.tbd_s(5 * MV, 300.0)        # more field -> sooner breakdown
    assert p.tbd_s(6 * MV, 400.0) < p.tbd_s(6 * MV, 300.0)        # hotter -> sooner breakdown


def test_tddb_calibration_anchors():
    p = TddbParams.calibrated(E_ox_V_m=7 * MV, T_K=398.0, tbd_s=500.0)
    assert float(p.tbd_s(7 * MV, 398.0)) == pytest.approx(500.0, rel=1e-12)


def test_tddb_guards():
    with pytest.raises(ValueError):
        tbd_e_model(5 * MV, 0.0, tau0_s=1.0, gamma_E_per_MV_cm=3.0, Ea_eV=0.7)   # T <= 0
    with pytest.raises(ValueError):
        tbd_e_model(-1.0, 300.0, tau0_s=1.0, gamma_E_per_MV_cm=3.0, Ea_eV=0.7)   # E < 0
    with pytest.raises(ValueError):
        tbd_one_over_e(0.0, 300.0, tau0_s=1.0, G_MV_cm=350.0)                    # 1/E at E = 0
    with pytest.raises(ValueError):
        weibull_area_scale(1e6, 1e-12, 1e-12, beta=0.0)


def test_tddb_electrothermal_adapter_ducktyped():
    from types import SimpleNamespace
    et = SimpleNamespace(layers=[SimpleNamespace(name="ox"), SimpleNamespace(name="ito")],
                         E_result=SimpleNamespace(mean_Ez_per_layer=lambda: np.array([-5e8, 1e7])),
                         T_per_layer=np.array([330.0, 345.0]))
    E, T = oxide_stress_from_electrothermal(et, "ox")
    assert E == 5e8 and T == 330.0                               # |Ez| magnitude + layer T
    with pytest.raises(ValueError):
        oxide_stress_from_electrothermal(et, "nope")


# ---- REL2 BTI ----

def test_bti_zero_time_zero_drift_and_monotone():
    assert float(dvth_power_law(0.0, 5 * MV, 398.0, A_V=1e-3)) == 0.0
    d1 = float(dvth_power_law(1e5, 5 * MV, 358.0, A_V=1e-3))
    d2 = float(dvth_power_law(1e5, 5 * MV, 398.0, A_V=1e-3))
    assert d2 > d1 > 0.0                                          # hotter degrades FASTER


def test_bti_inversion_round_trip():
    p = dict(A_V=2e-3, n_exp=1 / 6, gamma=0.35, Ea_eV=0.12)
    t = time_to_dvth(0.05, 5 * MV, 398.0, **p)
    assert float(dvth_power_law(t, 5 * MV, 398.0, **p)) == pytest.approx(0.05, rel=1e-12)


def test_bti_guards():
    with pytest.raises(ValueError):
        dvth_power_law(1.0, 5 * MV, 300.0, A_V=1e-3, n_exp=1.5)   # n outside (0,1)
    with pytest.raises(ValueError):
        dvth_power_law(1.0, 5 * MV, 300.0, A_V=1e-3, duty=0.0)    # duty in (0,1]
    with pytest.raises(ValueError):
        BtiParams.calibrated(t_s=0.0, E_ox_V_m=5 * MV, T_K=398.0, dvth_V=0.05)  # zero-drift anchor


# ---- REL4 de-doping ----

def test_dedoping_off_switch_exact_and_decay():
    t = np.linspace(0.0, 1e8, 5)
    assert np.all(carrier_decay(t, 400.0, n0_m3=9e26, params=DedopingParams(lambda0_per_s=0.0)) == 9e26)
    p = DedopingParams(lambda0_per_s=1e22, Ea_eV=2.0, n_min_m3=1e26)
    tau = 1.0 / float(p.rate_per_s(430.0))                        # ~27 s at these params
    n = carrier_decay(np.linspace(0.0, 3.0 * tau, 5), 430.0, n0_m3=9e26, params=p)
    assert np.all(np.diff(n) < 0) and n[-1] >= 1e26               # monotone decay toward n_min


def test_enz_wavelength_matches_exact_crossing():
    from dynameta.materials.optical_model import DrudeOptical
    from dynameta.constants import M_E, Q_E, EPS0, C_LIGHT
    m0, eps_inf, gam, n = 0.3 * M_E, 4.0, 1e13, 9e26
    d = DrudeOptical(eps_inf=eps_inf, m_opt_kg=m0, gamma_rad_s=gam)
    wp2 = n * Q_E ** 2 / (EPS0 * m0)
    lam_exact = 2 * np.pi * C_LIGHT / np.sqrt(wp2 / eps_inf - gam ** 2)
    assert enz_wavelength_m(d, n) == pytest.approx(lam_exact, rel=1e-9)
    with pytest.raises(ValueError):                               # no crossing in a narrow window
        enz_wavelength_m(d, n, lam_lo_m=600e-9, lam_hi_m=700e-9)


# ---- REL10 MTTF umbrella ----

def test_arrhenius_af_parenthesization_and_direction():
    af = arrhenius_af(0.5, 358.0, 398.0)
    assert af == pytest.approx(np.exp((0.5 / 8.617333262e-5) * (1 / 358.0 - 1 / 398.0)), rel=1e-13)
    assert 1.0 < af < 100.0                                       # O(5), NOT an overflow
    assert arrhenius_af(0.5, 398.0, 358.0) < 1.0                  # use hotter than stress -> shorter


def test_system_mttf_competing_risks_and_immortal():
    assert system_mttf([2e5, 5e5, 1e6]) == pytest.approx(125000.0, rel=1e-12)
    assert system_mttf([float("inf"), 2e5]) == pytest.approx(2e5, rel=1e-12)
    assert np.isinf(system_mttf([float("inf")]))
    with pytest.raises(ValueError):
        system_mttf([])
    with pytest.raises(ValueError):
        system_mttf([1e5, -1.0])


def test_weibull_earliest_and_fit():
    assert weibull_earliest_t63(1e6, 64, 2.0) == pytest.approx(125000.0, rel=1e-12)
    assert weibull_earliest_t63(1e6, 1, 2.0) == 1e6               # single element reduces
    assert fit_per_1e9_hours(1e5 * 3600.0) == pytest.approx(1e4, rel=1e-9)
    assert mttf_use_from_stress(500.0, field_af(3.0, 5.0, 8.0)) == pytest.approx(500.0 * np.exp(9.0))
