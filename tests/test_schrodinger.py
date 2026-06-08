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


def test_neumann_left_box_eigenvalues_and_nonzero_body():
    # Neumann (z=0) + Dirichlet (z=L) box: E_n = (n-1/2)^2 pi^2 hbar^2 / (2 m L^2), and the
    # ground state is COSINE-like -> NON-zero at the Neumann body (vs the Dirichlet wall's psi=0).
    L = 10e-9
    z = np.linspace(0.0, L, 801)
    sp = SchrodingerPoisson1D(z, MSTAR, T_K=300.0)
    E, V, _ = sp.solve_schrodinger(np.zeros_like(z), n_states=4, neumann_left=True)
    half = (np.arange(1, 5) - 0.5)
    E_an = half ** 2 * np.pi ** 2 * HBAR ** 2 / (2.0 * MSTAR * L ** 2)
    assert np.allclose(E, E_an, rtol=3e-3)
    assert abs(V[0, 0]) > 0.3 * np.abs(V[:, 0]).max()        # ground state non-zero at the Neumann body
    # The Neumann (n-1/2)^2 spectrum is DISTINCT from the Dirichlet n^2 box: the Neumann ground state
    # sits at ~(1/2)^2 = 1/4 of the Dirichlet ground state (a clean signature it is genuinely Neumann).
    Ed, _, _ = sp.solve_schrodinger(np.zeros_like(z), n_states=4)
    assert E[0] < 0.5 * Ed[0]


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


def test_nonparabolic_2d_filling():
    # Kane in-plane nonparabolicity: a single deep-well sub-band's T=0 sheet density is
    # n_s = pref0*(dE + alpha*dE^2); it must exceed the parabolic n_s by (1+alpha*dE).
    from dynameta.carriers.schrodinger_poisson import HBAR, Q
    L = 3e-9
    z = np.linspace(0.0, L, 401)
    sp = SchrodingerPoisson1D(z, 0.30 * M_E, T_K=50.0)
    E, _, _ = sp.solve_schrodinger(np.zeros_like(z), n_states=3)
    E_F = E[0] + 0.16 * Q
    assert E[1] > E_F                                   # only the ground sub-band fills
    ns_np = sp.density(np.zeros_like(z), E_F, n_states=3, alpha_np_per_eV=0.5).sheet_density_m2[0]
    ns_par = sp.density(np.zeros_like(z), E_F, n_states=3, alpha_np_per_eV=0.0).sheet_density_m2[0]
    pref0 = sp.g_s * sp.g_v * (0.30 * M_E) / (2.0 * np.pi * HBAR ** 2)
    dE, a = E_F - E[0], 0.5 / Q
    ns_cf = pref0 * (dE + a * dE ** 2)
    assert abs(ns_np - ns_cf) / ns_cf < 0.03           # matches the T=0 closed form
    assert abs(ns_np / ns_par - (1.0 + a * dE)) < 0.02 # DOS enhancement


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


def _open_body_setup():
    import warnings
    warnings.filterwarnings("ignore")          # the SP non-convergence warning is not under test
    n_bg, t = 4e26, 12e-9
    z = np.linspace(0.0, t, 51)
    sp = SchrodingerPoisson1D(z, MSTAR, T_K=300.0)
    E_F = (HBAR ** 2 / (2.0 * MSTAR)) * (3.0 * np.pi ** 2 * n_bg) ** (2.0 / 3.0)   # bulk degenerate
    kw = dict(eps_r=9.5, doping_m3=np.full(z.size, n_bg), E_F_J=E_F, phi_left_V=0.0, phi_right_V=0.0,
              max_outer=25, tol_V=1e-4, bound_tol=1e9, relax=0.8)
    interior = (z > 5e-9) & (z < 8e-9)
    return sp, z, n_bg, kw, interior


def test_open_body_recovers_bulk_no_pileup():
    # the OPEN body (bulk buffer) recovers n_bg at the body with NO boundary layer, where the Neumann
    # body PILES UP -- the rigorous oracle is validation/sp_open_body.
    sp, z, n_bg, kw, interior = _open_body_setup()
    _, n_open, _ = sp.solve_self_consistent(bulk_buffer_m=8e-9, **kw)
    _, n_neu, _ = sp.solve_self_consistent(neumann_left=True, **kw)
    ni_o = float(np.median(n_open[interior]))
    assert 0.9 < ni_o / n_bg < 1.1                       # interior recovers n_bg
    assert abs(n_open[1] / ni_o - 1.0) < 0.12            # near-body is FLAT (no pile-up / dead layer)
    assert n_neu[1] / float(np.median(n_neu[interior])) > 1.3   # Neumann body PILES UP (the BC it fixes)


def test_open_body_and_neumann_mutually_exclusive():
    sp, z, n_bg, kw, _ = _open_body_setup()
    with pytest.raises(ValueError):
        sp.solve_self_consistent(bulk_buffer_m=8e-9, neumann_left=True, **kw)
