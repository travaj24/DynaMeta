"""Fast unit tests for the magneto-optic 1-D FDTD (fdtd_mo) and the oblique complex-envelope 2D-TE
solver (low resolution -- the rigorous oracle checks live in validation/)."""
import numpy as np
import pytest

from dynameta.optics.fdtd import FDTDLayer
from dynameta.optics.fdtd_mo import MOLayer, solve_fdtd_mo_1d
from dynameta.optics.fdtd_nd import _HAVE_NUMBA, solve_fdtd_2d_oblique

LMIN, LMAX = 1400e-9, 1600e-9


def test_mo_vacuum_is_transparent_and_unrotated():
    r = solve_fdtd_mo_1d([MOLayer(thickness_m=1e-12)], lambda_min_m=LMIN, lambda_max_m=LMAX,
                         resolution=18, pol="y")
    b = r.band
    assert abs(float(np.median(r.T[b])) - 1.0) < 1e-2
    assert float(np.max(np.abs(r.r_co[b]))) < 5e-2                 # no reflection from nothing
    assert float(np.max(np.abs(r.t_cross[b]))) < 1e-6             # no spurious polarization mixing


def test_mo_reduction_wc_zero_no_rotation():
    L = MOLayer(thickness_m=300e-9, eps_xx=2.0, eps_yy=2.0, drude_wp_rad_s=1.2e15,
                drude_gamma_rad_s=2.0e13, cyclotron_wc_rad_s=0.0)
    r = solve_fdtd_mo_1d([L], lambda_min_m=LMIN, lambda_max_m=LMAX, resolution=18, pol="y")
    b = r.band
    assert float(np.max(np.abs(r.faraday_deg[b]))) < 1e-2        # no gyration -> no Faraday rotation
    assert float(np.max(np.abs(r.t_cross[b]))) < 1e-3


def test_mo_gyrotropic_rotates():
    L = MOLayer(thickness_m=400e-9, eps_xx=2.0, eps_yy=2.0, drude_wp_rad_s=1.2e15,
                drude_gamma_rad_s=2.0e13, cyclotron_wc_rad_s=3.0e14)
    r = solve_fdtd_mo_1d([L], lambda_min_m=LMIN, lambda_max_m=LMAX, resolution=18, pol="y")
    b = r.band
    assert abs(float(np.median(r.faraday_deg[b]))) > 1.0          # gyration -> a real rotation
    assert float(np.median(np.abs(r.t_cross[b]))) > 1e-2          # cross-pol appears


def test_mo_eps_circular_reduces_to_drude_at_wc0():
    L = MOLayer(thickness_m=1.0, eps_xx=2.0, eps_yy=2.0, drude_wp_rad_s=1.0e15, drude_gamma_rad_s=1.0e13)
    w = 2.0 * np.pi * 2.0e14
    ep, em = L.eps_circular(w, +1), L.eps_circular(w, -1)
    assert abs(ep - em) < 1e-12                                   # wc=0 -> the two circular modes coincide
    assert ep.imag > 0                                            # passive (loss)


def test_oblique_angle0_reduces_and_conserves_energy():
    r = solve_fdtd_2d_oblique([FDTDLayer(thickness_m=250e-9, eps_inf=4.0)], period_x_m=300e-9,
                              angle_deg=0.0, lambda_min_m=LMIN, lambda_max_m=LMAX, resolution=24)
    b = r.band
    assert np.all(np.abs(r.theta_deg[b]) < 1e-9)                 # angle 0 everywhere
    assert float(np.max(np.abs(r.R0[b] + r.T0[b] - 1.0))) < 2e-2  # lossless energy closes


def test_oblique_angle_is_frequency_dependent():
    r = solve_fdtd_2d_oblique([FDTDLayer(thickness_m=250e-9, eps_inf=4.0)], period_x_m=300e-9,
                              angle_deg=40.0, lambda_min_m=LMIN, lambda_max_m=LMAX, resolution=24)
    b = r.band
    th = r.theta_deg[b]
    assert th.max() > th.min() + 1.0                             # fixed k_par -> theta varies with f
    assert 20.0 < float(np.median(th)) < 60.0                    # near the requested 40 deg


def test_oblique_tm_ppol_energy_and_angle():
    # TM (p-pol: Hy,Ex,Ez) oblique kernel: lossless energy closes + the physical angle varies with f.
    r = solve_fdtd_2d_oblique([FDTDLayer(thickness_m=250e-9, eps_inf=4.0)], period_x_m=300e-9,
                              angle_deg=30.0, lambda_min_m=LMIN, lambda_max_m=LMAX, resolution=20,
                              nx=4, pol="p")
    b = r.band
    assert float(np.max(np.abs(r.R0[b] + r.T0[b] - 1.0))) < 3e-2  # lossless TM energy closes
    th = r.theta_deg[b]
    assert th.max() > th.min() + 1.0                             # fixed k_par -> theta varies with f


@pytest.mark.skipif(not _HAVE_NUMBA, reason="numba not installed")
def test_mo_numba_matches_numpy():
    L = MOLayer(thickness_m=300e-9, eps_xx=4.0, eps_yy=2.25, drude_wp_rad_s=1.6e15,
                drude_gamma_rad_s=8.0e13, cyclotron_wc_rad_s=2.5e14)
    kw = dict(lambda_min_m=LMIN, lambda_max_m=LMAX, resolution=14, pol="y")
    a = solve_fdtd_mo_1d([L], backend="numpy", **kw)
    b = solve_fdtd_mo_1d([L], backend="numba", **kw)
    m = a.band
    assert float(np.max(np.abs(a.R[m] - b.R[m]))) < 1e-10        # JIT loop == reference
    assert float(np.max(np.abs(a.T[m] - b.T[m]))) < 1e-10
    assert float(np.max(np.abs(a.faraday_deg[m] - b.faraday_deg[m]))) < 1e-8


@pytest.mark.skipif(not _HAVE_NUMBA, reason="numba not installed")
def test_oblique_numba_matches_numpy():
    ol = [FDTDLayer(thickness_m=250e-9, eps_inf=4.0)]
    kw = dict(period_x_m=300e-9, angle_deg=30.0, lambda_min_m=LMIN, lambda_max_m=LMAX, resolution=14, nx=4)
    a = solve_fdtd_2d_oblique(ol, backend="numpy", **kw)
    b = solve_fdtd_2d_oblique(ol, backend="numba", **kw)
    m = a.band
    assert float(np.max(np.abs(a.R0[m] - b.R0[m]))) < 1e-10      # complex-envelope JIT loop == reference
    assert float(np.max(np.abs(a.T0[m] - b.T0[m]))) < 1e-10
