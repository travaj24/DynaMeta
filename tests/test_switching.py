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
