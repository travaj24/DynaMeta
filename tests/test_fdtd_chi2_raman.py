"""Solver-light tests for the R15 chi2/Raman FDTD nonlinearities. The quantitative oracles
(coupled-wave SHG closed form, ADE vs solve_ivp, Stokes gain) live in
validation/fdtd_chi2_shg_raman.py."""
import numpy as np
import pytest

from dynameta.optics.fdtd import FDTDLayer
from dynameta.optics.fdtd_nd import solve_fdtd_2d


def _solve(layers, **kw):
    return solve_fdtd_2d(layers, period_x_m=100e-9, lambda_min_m=1.0e-6, lambda_max_m=1.4e-6,
                         resolution=16, backend=kw.pop("backend", "numpy"), **kw)


def test_zero_chi2_raman_byte_identical():
    r0 = _solve([FDTDLayer(150e-9, eps_inf=2.0)])
    r1 = _solve([FDTDLayer(150e-9, eps_inf=2.0, chi2_m_V=0.0, raman_chi3_m2_V2=0.0)])
    assert np.array_equal(r0.R0, r1.R0) and np.array_equal(r0.T0, r1.T0)


def test_chi2_requires_numpy_backend():
    lay = [FDTDLayer(150e-9, eps_inf=2.0, chi2_m_V=1e-12)]
    with pytest.raises(NotImplementedError):
        _solve(lay, backend="numba")


def test_raman_guards():
    with pytest.raises(ValueError):                      # Raman strength without a resonance
        _solve([FDTDLayer(150e-9, eps_inf=2.0, raman_chi3_m2_V2=1e-20)])


def test_chi2_layer_runs_and_stays_linear_at_low_field():
    lay = [FDTDLayer(150e-9, eps_inf=2.0, chi2_m_V=1e-15)]      # negligible chi2*E
    r0 = _solve([FDTDLayer(150e-9, eps_inf=2.0)])
    r1 = _solve(lay)
    m = r0.band
    assert np.max(np.abs(r1.R0[m] - r0.R0[m])) < 1e-9            # linear response unchanged


# ---- R20: clamped-inversion gain line + four-level populations --------------------------------

def test_gain_off_switch_and_guards():
    r0 = _solve([FDTDLayer(150e-9, eps_inf=2.0)])
    r1 = _solve([FDTDLayer(150e-9, eps_inf=2.0, gain_dN_m3=0.0, gain_kappa_C2_kg=0.0)])
    assert np.array_equal(r0.R0, r1.R0)
    with pytest.raises(ValueError):                      # gain strength without w_a/dw
        _solve([FDTDLayer(150e-9, eps_inf=2.0, gain_kappa_C2_kg=1e-8, gain_dN_m3=1e24)])
    with pytest.raises(NotImplementedError):             # numpy-only
        _solve([FDTDLayer(150e-9, eps_inf=2.0, gain_kappa_C2_kg=1e-8, gain_dN_m3=1e24,
                          gain_w_rad_s=1e15, gain_dw_rad_s=1e14)], backend="numba")


def test_four_level_steady_state_and_conservation():
    from dynameta.optics.gain_medium import FourLevelSystem, small_signal_gain_per_m
    s = FourLevelSystem(tau_32_s=4e-7, tau_21_s=2.3e-4, tau_10_s=1e-8, W_p_per_s=50.0,
                        N_total_m3=1e25)
    ss = s.steady_state()
    assert ss.sum() == pytest.approx(1e25, rel=1e-14)
    assert s.inversion_ss_m3() == pytest.approx(50.0 * ss[0] * (2.3e-4 - 1e-8), rel=1e-12)
    N = s.evolve(np.linspace(0.0, 1e-3, 11))
    assert np.max(np.abs(N.sum(axis=1) - 1e25)) < 1e-11 * 1e25
    assert s.inversion_ss_m3() > 0.0                     # tau_21 >> tau_10 -> inverted
    assert small_signal_gain_per_m(1e-8, -1e24, 1.5, 1e14) < 0.0   # dN < 0 -> absorption
    with pytest.raises(ValueError):
        FourLevelSystem(tau_32_s=0.0, tau_21_s=1e-4, tau_10_s=1e-8)
