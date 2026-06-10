"""Unit coverage for the reconfigurable switching drivers (carriers.switching) -- PCM JMAK
crystallization / melt-quench and LC director relaxation. Pure numpy, no solvers.
Run: python -m pytest tests/test_switching.py -q
"""
import numpy as np
import pytest

from dynameta.carriers.switching import PCMSwitching, LCRelaxation

EV = 1.602176634e-19


def _pcm():
    return PCMSwitching(K0_per_s=1e22, E_a_J=2.0 * EV, T_glass_K=425.0, T_melt_K=900.0, avrami_n=3.0)


def test_pcm_integrate_matches_isothermal_avrami():
    pcm = _pcm()
    t = np.linspace(0.0, 2e-7, 300)
    x_int = pcm.integrate(t, np.full_like(t, 700.0))
    x_cf = pcm.fraction_isothermal(t, 700.0)
    assert np.max(np.abs(x_int - x_cf)) < 1e-9               # additivity integrate == closed form
    assert np.all(np.diff(x_int) >= -1e-12) and x_int[0] < 1e-6 and x_int[-1] > 0.99


def test_pcm_melt_quench_and_frozen():
    pcm = _pcm()
    # frozen below the glass onset: x held
    assert np.allclose(pcm.integrate(np.array([0.0, 1.0, 2.0]), np.full(3, 300.0), x0=0.4), 0.4)
    # melt spike (> T_melt) resets to amorphous and stays 0 while molten
    t = np.linspace(0.0, 2e-7, 200)
    T = np.full_like(t, 700.0)
    T[t > 1e-7] = 1000.0
    x = pcm.integrate(t, T)
    assert x[t > 1.05e-7][-1] == 0.0


def test_pcm_guards():
    with pytest.raises(ValueError):
        PCMSwitching(K0_per_s=-1.0, E_a_J=EV, T_glass_K=400.0, T_melt_K=900.0)
    with pytest.raises(ValueError):
        PCMSwitching(K0_per_s=1e22, E_a_J=EV, T_glass_K=900.0, T_melt_K=400.0)   # melt < glass


def test_lc_relaxation_time_and_exponential_decay():
    lc = LCRelaxation(K_elastic_N=1e-11, gamma_visc_Pa_s=0.1, d_m=5e-6)
    assert lc.tau_s() == pytest.approx(0.1 * (5e-6) ** 2 / (1e-11 * np.pi ** 2), rel=1e-12)
    th0 = np.radians(30.0)
    assert lc.relax(np.array([lc.tau_s()]), th0)[0] == pytest.approx(th0 / np.e, rel=1e-9)
    assert lc.relax(np.array([0.0]), th0)[0] == pytest.approx(th0)
    with pytest.raises(ValueError):
        LCRelaxation(K_elastic_N=0.0, gamma_visc_Pa_s=0.1, d_m=5e-6)


# ---- R12: classical nucleation + growth (CNT/KJMA) ----------------------------------------

def _cnt(**kw):
    from dynameta.carriers.switching import PCMClassicalNucleation
    p = dict(I0_per_m3_s=1.0e39, sigma_J_m2=0.075, dHf_J_m3=6.2e8, Omega_m3=2.9e-29,
             u0_m_s=1.0e3, Ea_d_J=2.0 * 1.602176634e-19, Ea_g_J=1.5 * 1.602176634e-19,
             T_glass_K=450.0, T_melt_K=900.0)
    p.update(kw)
    return PCMClassicalNucleation(**p)


def test_cnt_isothermal_matches_avrami4_closed_form():
    pcm = _cnt()
    I = float(pcm.nucleation_rate_I(700.0))
    u = float(pcm.growth_velocity_u(700.0))
    t_c = (3.0 / (np.pi * I * u ** 3)) ** 0.25
    t = np.linspace(0.0, 1.5 * t_c, 4001)
    x = pcm.integrate(t, np.full_like(t, 700.0))
    x_cf = 1.0 - np.exp(-(np.pi / 3.0) * I * u ** 3 * t ** 4)
    assert np.max(np.abs(x - x_cf)) < 1e-4


def test_cnt_growth_only_is_avrami3():
    pcm = _cnt(I0_per_m3_s=0.0, N0_per_m3=1.0e20)
    u = float(pcm.growth_velocity_u(700.0))
    t_c = (3.0 / (4.0 * np.pi * 1.0e20 * u ** 3)) ** (1.0 / 3.0)
    t = np.linspace(0.0, 1.5 * t_c, 4001)
    x = pcm.integrate(t, np.full_like(t, 700.0))
    x_cf = 1.0 - np.exp(-(4.0 * np.pi / 3.0) * 1.0e20 * u ** 3 * t ** 3)
    assert np.max(np.abs(x - x_cf)) < 1e-4


def test_cnt_rates_masked_outside_window_and_nose_interior():
    pcm = _cnt()
    assert float(pcm.nucleation_rate_I(440.0)) == 0.0
    assert float(pcm.nucleation_rate_I(910.0)) == 0.0
    assert float(pcm.growth_velocity_u(910.0)) == 0.0
    Ts = np.linspace(451.0, 899.0, 300)
    Is = np.asarray(pcm.nucleation_rate_I(Ts))
    ipk = int(np.argmax(Is))
    assert 0 < ipk < Ts.size - 1                 # the CNT nose peaks strictly inside (Tg, Tm)
    assert Is[-1] < 1e-20 * Is[ipk]              # barrier divergence kills I at the melt edge


def test_cnt_melt_reset_frozen_hold_and_disabled():
    pcm = _cnt()
    t = np.linspace(0.0, 1.0, 401)
    T = np.full_like(t, 700.0)
    T[200:240] = 950.0
    T[300:] = 400.0
    # rescale time so some crystallization happens before the melt spike
    I = float(pcm.nucleation_rate_I(700.0))
    u = float(pcm.growth_velocity_u(700.0))
    t = t * (3.0 / (np.pi * I * u ** 3)) ** 0.25
    x = pcm.integrate(t, T)
    assert x[199] > 0.0
    assert np.max(x[200:240]) == 0.0             # molten == amorphous exactly
    assert np.max(np.abs(np.diff(x[300:]))) == 0.0   # frozen holds exactly
    off = _cnt(enabled=False).integrate(t, T, x0=0.37)
    assert np.all(off == 0.37)


def test_cnt_guards():
    with pytest.raises(ValueError):
        _cnt(sigma_J_m2=-0.1)
    with pytest.raises(ValueError):
        _cnt(dHf_J_m3=0.0)
    with pytest.raises(ValueError):
        _cnt(Omega_m3=0.0)
    with pytest.raises(ValueError):
        _cnt(T_glass_K=900.0, T_melt_K=450.0)
