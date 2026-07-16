"""Fast unit tests for the numba-CUDA 2D-TE FDTD backend (a persistent cooperative-groups GPU kernel run
in timestep chunks). Skipped automatically when no CUDA GPU is available. The rigorous gates live in
validation/fdtd_numba_cuda.py."""
import numpy as np
import pytest

from dynameta.optics.fdtd import FDTDLayer
from dynameta.optics.fdtd_nd import solve_fdtd_2d, available_backends, have_numba_cuda

pytestmark = pytest.mark.skipif(not have_numba_cuda(), reason="no CUDA GPU available")

LMIN, LMAX = 1200e-9, 1600e-9


def test_numba_cuda_listed_when_available():
    assert "numba-cuda" in available_backends()


def test_numba_cuda_matches_numpy_uniform():
    # the GPU kernel reproduces the NumPy reference on a uniform dispersive (Drude) slab.
    ol = [FDTDLayer(thickness_m=250e-9, eps_inf=4.0, drude_wp_rad_s=1.5e15, drude_gamma_rad_s=1e14)]
    kw = dict(period_x_m=300e-9, lambda_min_m=LMIN, lambda_max_m=LMAX, resolution=14, nx=8)
    a = solve_fdtd_2d(ol, backend="numpy", **kw)
    g = solve_fdtd_2d(ol, backend="numba-cuda", **kw)
    m = a.band
    assert float(np.max(np.abs(a.R0[m] - g.R0[m]))) < 1e-9      # cooperative GPU kernel == reference
    assert float(np.max(np.abs(a.T0[m] - g.T0[m]))) < 1e-9


def test_numba_cuda_matches_numpy_structured_lorentz():
    # structured grating + Drude + Lorentz pole on the device == the CPU numba kernel.
    def lat(nx, nz, zc, pad, zs):
        e = np.ones((nx, nz)); mm = (zc >= pad) & (zc < pad + zs); e[:nx // 2, mm] = 6.25; return e

    ol = [FDTDLayer(thickness_m=220e-9, eps_inf=3.0, drude_wp_rad_s=1.0e15, drude_gamma_rad_s=8e13,
                    lorentz_delta_eps=1.5, lorentz_w0_rad_s=1.4e15, lorentz_gamma_rad_s=6e13)]
    kw = dict(period_x_m=600e-9, lambda_min_m=LMIN, lambda_max_m=1700e-9, resolution=12, nx=16,
              lateral_eps_inf=lat)
    c = solve_fdtd_2d(ol, backend="numba", **kw)
    g = solve_fdtd_2d(ol, backend="numba-cuda", **kw)
    m = c.band
    assert float(np.max(np.abs(c.R0[m] - g.R0[m]))) < 1e-9
    assert float(np.max(np.abs(c.T0[m] - g.T0[m]))) < 1e-9
