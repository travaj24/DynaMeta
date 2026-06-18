"""Solver-light tests for the R15 chi2/Raman FDTD nonlinearities. The quantitative oracles
(coupled-wave SHG closed form, ADE vs solve_ivp, Stokes gain) live in
validation/fdtd_chi2_shg_raman.py."""
import numpy as np
import pytest

from dynameta.optics.fdtd import FDTDLayer
from dynameta.optics.fdtd_nd import _HAVE_NUMBA, _have_numba_cuda, solve_fdtd_2d

needs_numba = pytest.mark.skipif(not _HAVE_NUMBA,
                                 reason="numba not installed (CI numpy-only leg)")
needs_gpu = pytest.mark.skipif(not _have_numba_cuda(), reason="no CUDA GPU available")


def _solve(layers, **kw):
    return solve_fdtd_2d(layers, period_x_m=100e-9, lambda_min_m=1.0e-6, lambda_max_m=1.4e-6,
                         resolution=16, backend=kw.pop("backend", "numpy"), **kw)


def test_zero_chi2_raman_byte_identical():
    r0 = _solve([FDTDLayer(150e-9, eps_inf=2.0)])
    r1 = _solve([FDTDLayer(150e-9, eps_inf=2.0, chi2_m_V=0.0, raman_chi3_m2_V2=0.0)])
    assert np.array_equal(r0.R0, r1.R0) and np.array_equal(r0.T0, r1.T0)


@needs_numba
def test_chi2_numba_matches_numpy():
    # numba/jax now carry the nonlinearities (deferred-item completion)
    lay = [FDTDLayer(150e-9, eps_inf=2.0, chi2_m_V=1e-12)]
    r_np = _solve(lay)
    r_nb = _solve(lay, backend="numba")
    m = r_np.band
    assert np.max(np.abs(r_nb.R0[m] - r_np.R0[m])) < 1e-12


@needs_gpu
def test_chi2_gpu_backend_matches_numpy():
    # the cooperative CUDA kernel now carries the cell-local nonlinear recurrences (the
    # rigorous multi-gate equality incl. cupy lives in validation/fdtd_gpu_nonlinear.py)
    lay = [FDTDLayer(150e-9, eps_inf=2.0, chi2_m_V=1e-12)]
    r_np = _solve(lay)
    r_gpu = _solve(lay, backend="numba-cuda")
    m = r_np.band
    assert np.max(np.abs(r_gpu.R0[m] - r_np.R0[m])) < 1e-12


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


@needs_numba
def test_gain_numba_matches_numpy():
    # gain on numba now matches numpy (deferred-item completion)
    lay_g = [FDTDLayer(150e-9, eps_inf=2.0, gain_kappa_C2_kg=1e-8, gain_dN_m3=1e22,
                       gain_w_rad_s=1.5e15, gain_dw_rad_s=1.3e14)]
    r_np = _solve(lay_g)
    r_nb = _solve(lay_g, backend="numba")
    assert np.max(np.abs(r_nb.R0[r_np.band] - r_np.R0[r_np.band])) < 1e-12


def test_four_level_steady_state_and_conservation():
    from dynameta.optics.laser_gain import FourLevelSystem, small_signal_gain_per_m
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


def test_cavity_design_formulas_worked_point():
    # the hand-verified lasing worked point (validation/fdtd_lasing_cavity.py pins):
    # n_c = sqrt(2) / n_out = 10 mirrors, L = 10.6 um, dw = 2pi 2e13, kappa = q^2/m_e
    from dynameta.constants import M_E, Q_E
    from dynameta.optics.laser_gain import (FourLevelSystem, cavity_photon_lifetime_s,
                                             pump_threshold_per_s,
                                             relaxation_oscillation_rad_s,
                                             threshold_inversion_m3)
    n_c, L = np.sqrt(2.0), 10.6e-6
    R = ((n_c - 10.0) / (n_c + 10.0)) ** 2
    kappa, dw = Q_E ** 2 / M_E, 2.0 * np.pi * 2.0e13
    assert R == pytest.approx(0.56582, rel=1e-4)
    tau_p = cavity_photon_lifetime_s(L, n_c, R, R)
    assert tau_p == pytest.approx(87.8e-15, rel=2e-3)
    dN_th = threshold_inversion_m3(kappa, n_c, dw, L, R, R)
    assert dN_th == pytest.approx(8.994e23, rel=2e-3)
    s = FourLevelSystem(tau_32_s=1e-14, tau_21_s=5e-12, tau_10_s=5e-15, N_total_m3=1e25)
    assert pump_threshold_per_s(dN_th, s) == pytest.approx(1.979e10, rel=2e-3)
    w, g = relaxation_oscillation_rad_s(2.0, tau_p, 5e-12)
    assert g == pytest.approx(2.0e11, rel=1e-12)
    assert w == pytest.approx(np.sqrt(1.0 / (tau_p * 5e-12) - g * g), rel=1e-12)


def test_cavity_design_formula_guards():
    from dynameta.optics.laser_gain import (FourLevelSystem, cavity_photon_lifetime_s,
                                             pump_threshold_per_s,
                                             relaxation_oscillation_rad_s,
                                             threshold_inversion_m3)
    with pytest.raises(ValueError):                      # lossless cavity: no finite tau_p
        cavity_photon_lifetime_s(1e-6, 1.5, 1.0, 1.0)
    with pytest.raises(ValueError):                      # Gamma out of range
        threshold_inversion_m3(1e-8, 1.5, 1e14, 1e-6, 0.5, 0.5, Gamma=1.5)
    s = FourLevelSystem(tau_32_s=1e-14, tau_21_s=5e-12, tau_10_s=5e-15, N_total_m3=1e20)
    with pytest.raises(ValueError):                      # threshold unreachable (chain caps)
        pump_threshold_per_s(1e25, s)
    with pytest.raises(ValueError):                      # at/below threshold: no oscillation
        relaxation_oscillation_rad_s(1.0, 1e-13, 5e-12)
