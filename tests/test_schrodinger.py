"""Fast, dependency-light (numpy/scipy only) unit tests for the 1D Schrodinger-Poisson
solver: analytic square- and triangular-well eigenvalues + degenerate-filling
consistency. The heavier self-consistent accumulation case lives in
validation/schrodinger_poisson.py."""
import numpy as np
import pytest

from dynameta.carriers.schrodinger_poisson import (
    SchrodingerPoisson1D, HBAR, M_E, Q)

MSTAR = 0.35 * M_E


def test_infinite_square_well():
    L = 10e-9
    z = np.linspace(0.0, L, 801)
    sp = SchrodingerPoisson1D(z, MSTAR, T_K=300.0)
    E, _, _ = sp.solve_schrodinger(np.zeros_like(z), n_states=4)
    n = np.arange(1, 5)
    E_an = n ** 2 * np.pi ** 2 * HBAR ** 2 / (2.0 * MSTAR * L ** 2)
    assert np.allclose(E, E_an, rtol=2e-4)


def test_triangular_well_airy():
    sp_special = pytest.importorskip("scipy.special")
    F, Zmax = 1.0e8, 40e-9
    z = np.linspace(0.0, Zmax, 1601)
    sp = SchrodingerPoisson1D(z, MSTAR, T_K=300.0)
    E, _, _ = sp.solve_schrodinger(Q * F * z, n_states=6)
    a = sp_special.ai_zeros(4)[0]
    pref = (Q * F) ** (2.0 / 3.0) * (HBAR ** 2 / (2.0 * MSTAR)) ** (1.0 / 3.0)
    E_an = np.abs(a) * pref
    assert np.allclose(E[:4], E_an, rtol=2e-3)


def test_degenerate_filling_consistency():
    L = 8e-9
    z = np.linspace(0.0, L, 801)
    sp = SchrodingerPoisson1D(z, MSTAR, T_K=300.0, g_s=2, g_v=1)
    res = sp.density(np.zeros_like(z), 0.15 * Q, n_states=10)
    n_z, zz = res.density_m3, res.z_m
    sheet_nz = float(np.sum(0.5 * (n_z[:-1] + n_z[1:]) * np.diff(zz)))
    sheet_sub = float(np.sum(res.sheet_density_m2))
    assert abs(sheet_nz - sheet_sub) / sheet_sub < 1e-3


def test_uniform_grid_required():
    z = np.concatenate([np.linspace(0, 5e-9, 50), np.linspace(5e-9, 1e-8, 80)])
    with pytest.raises(ValueError):
        SchrodingerPoisson1D(z, MSTAR)


def test_degenerate_slab_recovers_bulk():
    # slab mode (keep all sub-bands) must recover the bulk degenerate density of a
    # heavily-doped semiconductor; the default bound-state rejection (isolated well)
    # collapses it to ~0. Flat-band square slab, E_F from the bulk degenerate relation.
    from dynameta.carriers.schrodinger_poisson import HBAR
    n_bg, t = 4e26, 12e-9
    z = np.linspace(0.0, t, 401)
    sp = SchrodingerPoisson1D(z, MSTAR, T_K=300.0)
    E_F = (HBAR ** 2 / (2.0 * MSTAR)) * (3.0 * np.pi ** 2 * n_bg) ** (2.0 / 3.0)
    n_mid = sp.density(np.zeros_like(z), E_F, n_states=60, bound_tol=1e9).density_m3[200]
    assert 0.85 < n_mid / n_bg < 1.15          # bulk recovered
    n_reject = sp.density(np.zeros_like(z), E_F, n_states=60, bound_tol=1e-3).density_m3[200]
    assert n_reject < 0.8 * n_bg               # rejection under-counts the continuum
