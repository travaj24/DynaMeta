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
