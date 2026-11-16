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
    assert np.allclose(E, E_an, rtol=2e-4, atol=0.0)


def test_neumann_left_box_eigenvalues_and_nonzero_body():
    # Neumann (z=0) + Dirichlet (z=L) box: E_n = (n-1/2)^2 pi^2 hbar^2 / (2 m L^2), and the
    # ground state is COSINE-like -> NON-zero at the Neumann body (vs the Dirichlet wall's psi=0).
    L = 10e-9
    z = np.linspace(0.0, L, 801)
    sp = SchrodingerPoisson1D(z, MSTAR, T_K=300.0)
    E, V, _ = sp.solve_schrodinger(np.zeros_like(z), n_states=4, neumann_left=True)
    half = (np.arange(1, 5) - 0.5)
    E_an = half ** 2 * np.pi ** 2 * HBAR ** 2 / (2.0 * MSTAR * L ** 2)
    assert np.allclose(E, E_an, rtol=3e-3, atol=0.0)
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
    assert np.allclose(E[:4], E_an, rtol=2e-3, atol=0.0)


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


def test_c18_gate_to_psi_s_keeps_vg_over_256_after_bracket_expansion():
    """AUDIT C-18 (2026-11-02): the gate -> psi_s bisection ran a FIXED 8 halvings of [0, hi], i.e.
    bracket/256 -- Vg/256 unexpanded but Vg/4 once the bracket had doubled six times, so a root found
    after expansion came back up to Vg/8 off. A synthetic linear column (residual = k psi_s - Vg, root
    psi* = Vg/k beyond Vg) pins the resolution; pre-fix the error was 0.0125 Vg at k = 1/19.3 and
    0.075 Vg at k = 1/45.7 (bound 0.0039).

    With the sign of the flat-band gate map fixed: (a) the synthetic column is ODD in the surface potential
    (excess = c psi_s); an even column (c |psi|) would encode the sign error of the v0.15.1 residual: the sheet
    excess of a depleting column is negative, and Gauss's law
    Vg = psi_s + q N_exc/C_ox uses SIGNED quantities. (b) The root is found by brentq to 1e-9 V, so the
    resolution bound tightens from Vg/256 to 1e-7 V, and the bracketed case (k > 1) now costs AT MOST
    the old 11 column solves (it was exactly 11: 1 baseline + 2 bracket ends + 8 halvings)."""
    from dynameta.carriers.sp_carrier import SchrodingerPoissonCarrier
    from dynameta.constants import Q_E
    car = SchrodingerPoissonCarrier(oxide_thk_m=2e-9, nz=21, gate_map="flat_band_excess")
    t = car.semi_thk_m
    for k in (1.0 / 19.3, 1.0 / 45.7, 0.37, 3.0):
        c = (k - 1.0) * car._C_ox / (Q_E * t)              # uniform excess density per volt of psi_s
        calls = []

        def column(sp, Nd, psi, c=c):
            calls.append(psi)
            return None, np.full(Nd.shape, car.n_bg_m3) + c * psi   # odd in the surface potential
        car._solve_column = column
        for vg in (0.7, -1.3):
            calls.clear()
            psi = car._gate_to_psi_s(vg)
            root = vg / k
            assert abs(psi - root) <= 1e-7, (k, vg, psi, root)
            if k > 1.0:                                     # bracketed at once
                assert len(calls) <= 11, len(calls)


# ---- v0.16: warm start honours the end potentials ------------------------------------------------
# Film of these tests: the generic gated film (6 nm here, n_bg 3e26 m^-3, m* 0.30 m_e, eps_s 9.0, oxide 8 nm /
# eps 20).

def _q1_setup():
    from dynameta.constants import HBAR as HB, Q_E
    z = np.linspace(0, 6e-9, 241)
    m = 0.30 * M_E
    sp = SchrodingerPoisson1D(z, m)
    EF = HB ** 2 / (2 * m) * (3 * np.pi ** 2 * 3e26) ** (2 / 3)
    kw = dict(eps_r=9.0, doping_m3=np.full_like(z, 3e26), E_F_J=EF, bound_tol=1e9, n_states=80,
              tol_V=1e-6)
    phi1, _, _ = sp.solve_self_consistent(phi_left_V=0.0, phi_right_V=0.5, **kw)
    return sp, kw, -Q_E * phi1


@pytest.mark.parametrize("extra", [{}, {"neumann_left": True}, {"bulk_buffer_m": 2e-9}],
                         ids=["dirichlet", "neumann_body", "bulk_buffer"])
def test_q1_warm_start_honours_end_potentials(extra):
    """A warm start from the 0.5 V solution, re-solved at phi_right = 1.0 V (and phi_left = 0.2 V), must
    reach the boundary values exactly and give the cold-start density. v0.15.1: phi2[-1] = 0.5 (the
    warm start's end), r.converged = True, max|n2 - n3|/max(n3) = 0.34-0.39 -- the old boundary problem
    was re-solved silently. Fixed: 1.8e-7 to 2.9e-7 (the outer tolerance, 1e-6 V)."""
    sp, kw, U1 = _q1_setup()
    for left in (0.0, 0.2):
        phi2, n2, r2 = sp.solve_self_consistent(phi_left_V=left, phi_right_V=1.0, U_init_J=U1,
                                                **kw, **extra)
        phi3, n3, _ = sp.solve_self_consistent(phi_left_V=left, phi_right_V=1.0, **kw, **extra)
        assert r2.converged
        assert phi2[-1] == 1.0
        assert phi2[0] == left or "bulk_buffer_m" in extra   # the buffer strips its far (Dirichlet) end
        assert np.max(np.abs(n2 - n3)) / np.max(n3) < 1e-6


def test_q1_warm_start_with_matching_ends_is_unchanged():
    """The warm-start seed equals the v0.15.1 seed -U/q BIT FOR BIT when its ends already match the
    requested boundary values (the rest of the iteration is unchanged code, so the solve is too), and
    a mismatched seed is corrected by a LINEAR ramp (interior shape kept, ends exact)."""
    from dynameta.carriers.schrodinger_poisson import _warm_start_phi
    from dynameta.constants import Q_E
    rng = np.random.default_rng(7)
    U = -Q_E * (0.3 * rng.standard_normal(57) + np.linspace(0.11, 0.73, 57))
    old = -np.asarray(U) / Q_E
    assert np.array_equal(_warm_start_phi(U, old[0], old[-1]), old)
    new = _warm_start_phi(U, 0.0, 1.0)
    assert new[0] == 0.0 and new[-1] == 1.0
    ramp = new - old
    assert np.allclose(np.diff(ramp, 2), 0.0, atol=1e-14)    # the correction is linear in the index
    sp, kw, U1 = _q1_setup()
    with pytest.raises(ValueError):
        sp.solve_self_consistent(phi_left_V=0.0, phi_right_V=1.0, U_init_J=U1[:-1], **kw)


# ---- v0.16: the bulk Fermi level of the sub-band model ---------------------------------------------

def _fd_norm(j, x):
    """Normalized complete Fermi-Dirac integral F_j(x) = Int t^j/(1+e^(t-x)) dt / Gamma(j+1), by an
    independent scipy quadrature (the oracle; not the module's kernels)."""
    from scipy.integrate import quad
    from scipy.special import gamma
    top = max(x, 0.0) + 80.0
    v, _ = quad(lambda t: t ** j / (1.0 + np.exp(t - x)), 0.0, top, epsabs=0.0, epsrel=1e-13,
                limit=400, points=[x] if x > 0 else None)
    return v / gamma(j + 1.0)


@pytest.mark.parametrize("alpha, ef_eV, ef_T0_eV", [(0.0, 0.543695, 0.544707), (0.4, 0.492361, 0.460049)])
def test_q3_bulk_fermi_level_matches_closed_form(alpha, ef_eV, ef_T0_eV):
    """The model's flat-band bulk is n_sep = (1/pi) Int dk_z n_s(hbar^2 k_z^2/2m); substituting
    u = hbar^2 k_z^2/(2 m kT) and exchanging the integrals gives the closed form
    n_sep = N_c [F_1/2(eta) + 2 alpha kT F_3/2(eta)] (normalized FD integrals, N_c = g_s g_v
    (m kT / 2 pi hbar^2)^(3/2)); at alpha = 0 it is the parabolic N_c F_1/2. bulk_fermi_level must hit
    n_bg on that oracle to 1e-10. v0.15.1 used the T = 0 3D relation (column 3): N_c F_1/2 = 1.00278
    n_bg at alpha = 0 (model level -1.01 meV from it) and 0.89570 n_bg at alpha = 0.4 (+32.31 meV), for
    the generic film 3e26 m^-3, 0.30 m_e, 300 K."""
    from dynameta.constants import KB as kB, Q_E
    from dynameta.carriers.schrodinger_poisson import bulk_fermi_level_3d_T0_J
    m, T, n_bg = 0.30 * M_E, 300.0, 3e26
    sp = SchrodingerPoisson1D(np.linspace(0.0, 1e-8, 11), m, T_K=T)
    E = sp.bulk_fermi_level(n_bg, alpha_np_per_eV=alpha)
    kT = kB * T
    Nc = 2.0 * (m * kT / (2.0 * np.pi * HBAR ** 2)) ** 1.5
    eta = E / kT
    n_cf = Nc * (_fd_norm(0.5, eta) + 2.0 * (alpha / Q_E) * kT * _fd_norm(1.5, eta))
    assert abs(n_cf / n_bg - 1.0) < 1e-10
    assert abs(sp.bulk_density(E, alpha_np_per_eV=alpha) / n_bg - 1.0) < 1e-12
    assert abs(E / Q_E - ef_eV) < 2e-6
    E0 = bulk_fermi_level_3d_T0_J(n_bg, m, alpha_np_per_eV=alpha)
    assert abs(E0 / Q_E - ef_T0_eV) < 2e-6
    n_T0 = Nc * (_fd_norm(0.5, E0 / kT) + 2.0 * (alpha / Q_E) * kT * _fd_norm(1.5, E0 / kT))
    assert abs(n_T0 / n_bg - 1.0) > 2e-3                     # the old level misses the model's bulk


def test_q3_bulk_fermi_level_nondegenerate_and_degeneracy_factors():
    """Boltzmann limit: n_sep -> N_c e^eta, so E_F -> kT ln(n_bg/N_c) for n_bg << N_c; valley
    degeneracy g_v enters N_c linearly. Oracle: the closed form with the independent FD quadrature."""
    from dynameta.constants import KB as kB
    from dynameta.carriers.schrodinger_poisson import sp_bulk_fermi_level_J
    m, T = 0.26 * M_E, 300.0
    kT = kB * T
    for n_bg, gv in ((1e21, 1), (1e23, 2), (3e25, 6)):
        E = sp_bulk_fermi_level_J(n_bg, m, T, g_s=2, g_v=gv)
        Nc = 2.0 * gv * (m * kT / (2.0 * np.pi * HBAR ** 2)) ** 1.5
        assert abs(Nc * _fd_norm(0.5, E / kT) / n_bg - 1.0) < 1e-10
        if n_bg < 1e-3 * Nc:
            assert abs(E - kT * np.log(n_bg / Nc)) < 1e-3 * kT
    with pytest.raises(ValueError):
        sp_bulk_fermi_level_J(0.0, m, T)


def test_q3_flat_band_slab_holds_n_bg():
    """Wide-well limit: a 40 nm flat-band slab (0.025 nm grid, hard walls) filled at the model's E_F
    holds n_bg at its centre: measured 1.000073 (alpha 0) and 1.000044 (alpha 0.4); the residual is the
    finite-difference sub-band energies (O(h^2)). Error budget 1e-3 / 2e-3. v0.15.1's E_F gave 1.002856
    and 0.895757 (fails both). Generic film 3e26 m^-3, 0.30 m_e."""
    L = 40e-9
    nz = int(round(L / 0.025e-9)) + 1
    z = np.linspace(0.0, L, nz)
    sp = SchrodingerPoisson1D(z, 0.30 * M_E, T_K=300.0)
    for alpha, tol in ((0.0, 1e-3), (0.4, 2e-3)):
        E = sp.bulk_fermi_level(3e26, alpha_np_per_eV=alpha)
        n = sp.density(np.zeros(nz), E, n_states=120, bound_tol=1e9, alpha_np_per_eV=alpha).density_m3
        assert abs(n[(nz - 2) // 2] / 3e26 - 1.0) < tol, alpha


@pytest.mark.parametrize("alpha, bound_mV", [(0.0, 0.2), (0.4, 1.0)])
def test_q3_carrier_has_no_spurious_band_bending(alpha, bound_mV):
    """Charge neutrality at flat band: the carrier's self-consistent 0 V solve of a 40 nm slab needs no
    band bending at the centre when E_F is the model's own bulk level. Measured -0.025 / -0.017 mV
    (the O(h^2) grid residual above, (2/3) E_F x 7e-5). v0.15.1: -1.037 mV (alpha 0) and +32.29 mV
    (alpha 0.4), which the solve used to restore n_bg. Generic film 3e26 m^-3, 0.30 m_e, eps 9.0."""
    from dynameta.carriers.sp_carrier import SchrodingerPoissonCarrier
    from dynameta.sweep import BiasPoint
    c = SchrodingerPoissonCarrier(semi_thk_m=40e-9, n_bg_m3=3e26, m_eff_kg=0.30 * M_E, eps_static=9.0, nz=1601,
                                  n_states=120, n_lateral=2, alpha_np_per_eV=alpha)
    assert c.E_F_J == SchrodingerPoisson1D(np.linspace(0, 1e-8, 11), c.m_eff_kg).bulk_fermi_level(
        c.n_bg_m3, alpha_np_per_eV=alpha)
    reg = c.solve(BiasPoint({"gate": 0.0, "body": 0.0}, "0")).regions["semi"]
    phi = reg.grid_fields["potential_V"][0, 0]
    n = reg.grid_fields["electron_density_m3"][0, 0]
    assert abs(phi[800]) * 1e3 < bound_mV
    assert abs(n[800] / 3e26 - 1.0) < 1e-3


def test_q3_legacy_fermi_level_is_bit_identical():
    """fermi_level="bulk3d_T0" restores the v0.15.1 E_F bit for bit (the v0.15.1 expression is
    re-typed here, operation for operation)."""
    from dynameta.carriers.sp_carrier import SchrodingerPoissonCarrier
    from dynameta.constants import HBAR as HB, Q_E
    for alpha, gs, gv, m in ((0.0, 2, 1, 0.30 * M_E), (0.5, 2, 1, 0.25 * M_E), (0.4, 1, 3, 0.2 * M_E)):
        c = SchrodingerPoissonCarrier(n_bg_m3=3e26, m_eff_kg=m, alpha_np_per_eV=alpha, g_s=gs, g_v=gv,
                                      fermi_level="bulk3d_T0")
        gamma_F = (HB ** 2 / (2.0 * float(m))) * (6.0 * np.pi ** 2 * 3e26 / (gs * gv)) ** (2.0 / 3.0)
        if alpha > 0.0:
            a = float(alpha) / Q_E
            old = (-1.0 + np.sqrt(1.0 + 4.0 * a * gamma_F)) / (2.0 * a)
        else:
            old = gamma_F
        assert c.E_F_J == old
    with pytest.raises(ValueError):
        SchrodingerPoissonCarrier(fermi_level="T0")
    with pytest.raises(ValueError):
        SchrodingerPoissonCarrier(gate_map="identity")


# ---- v0.16: Gauss's law at the oxide face --------------------------------------------------------

def test_face_flux_kernel_discrete_gauss_law_and_order():
    """sp_face_flux_C_m2 (a) obeys the discrete Gauss law F_right - F_left = q trapz(n - Nd) EXACTLY for
    any profile satisfying the solver's interior stencil (algebraic identity, round-off level), and
    (b) is a second-order approximation of eps phi' at the ends (oracle: the analytic derivative of a
    smooth phi with n = Nd + eps phi''/q)."""
    from dynameta.carriers.schrodinger_poisson import sp_face_flux_C_m2, EPS0 as E0
    from dynameta.constants import Q_E
    from dynameta.core.numerics import trapz
    eps_r, Nd0 = 9.0, 3e26
    ee = eps_r * E0
    rng = np.random.default_rng(3)
    z = np.linspace(0.0, 5e-9, 41)
    h = z[1] - z[0]
    phi = 0.2 * rng.standard_normal(z.size)
    Nd = np.full(z.size, Nd0)
    n = np.empty(z.size)
    n[1:-1] = Nd[1:-1] + ee * (phi[:-2] - 2 * phi[1:-1] + phi[2:]) / h ** 2 / Q_E
    n[0], n[-1] = 1.3e26, 0.0
    Fr = sp_face_flux_C_m2(phi, n, Nd, h, eps_r, side="right")
    Fl = sp_face_flux_C_m2(phi, n, Nd, h, eps_r, side="left")
    gauss = Q_E * trapz(n - Nd, z)
    assert abs((Fr - Fl) - gauss) < 1e-9 * max(abs(Fr), abs(Fl), abs(gauss))
    errs = []
    for N in (51, 101, 201):
        z = np.linspace(0.0, 5e-9, N)
        h = z[1] - z[0]
        k = 2 * np.pi / 7e-9
        phi = 0.3 * np.sin(k * z + 0.4) + 0.1 * (z / 5e-9) ** 3
        d1 = 0.3 * k * np.cos(k * z + 0.4) + 0.3 * z ** 2 / 5e-9 ** 3
        d2 = -0.3 * k ** 2 * np.sin(k * z + 0.4) + 0.6 * z / 5e-9 ** 3
        nn = Nd0 + ee * d2 / Q_E
        Nd = np.full(N, Nd0)
        errs.append(max(abs(sp_face_flux_C_m2(phi, nn, Nd, h, eps_r) - ee * d1[-1]),
                        abs(sp_face_flux_C_m2(phi, nn, Nd, h, eps_r, side="left") - ee * d1[0])))
    assert errs[0] / errs[1] > 3.6 and errs[1] / errs[2] > 3.6          # second order (ratio -> 4)
    with pytest.raises(ValueError):
        sp_face_flux_C_m2(phi, nn, Nd, h, eps_r, side="top")


def _n4_carrier(gate_map, nz=481):
    from dynameta.carriers.sp_carrier import SchrodingerPoissonCarrier
    return SchrodingerPoissonCarrier(semi_thk_m=6e-9, n_bg_m3=3e26, m_eff_kg=0.30 * M_E, eps_static=9.0,
                                     nz=nz, n_lateral=2, oxide_thk_m=8e-9, eps_oxide=20.0,
                                     gate_map=gate_map)


def _n4_profile(c, vg):
    from dynameta.sweep import BiasPoint
    cf = c.solve(BiasPoint({"gate": vg, "body": 0.0}, "g"))
    reg = cf.regions["semi"]
    return (cf.extras["surface_potential_V"], reg.grid_axes_m["z"],
            reg.grid_fields["potential_V"][0, 0], reg.grid_fields["electron_density_m3"][0, 0])


N4_VGS = (-4.0, -2.0, -1.0, 1.0, 2.0)


def test_n4_flat_band_map_obeys_its_gauss_law_for_both_signs():
    """gate_map="flat_band_excess": Vg = psi_s + q N_exc/C_ox with SIGNED psi_s and N_exc (excess vs the
    psi_s = 0 profile). v0.15.1 dropped the sign of the charge term in depletion: psi_s = -15.20 /
    -13.19 / -12.19 V at -4 / -2 / -1 V with Gauss residuals of -22.39 V, no warning. Measured with the
    fix (nz 481, model E_F): -1.141830 / -0.523020 / -0.250435 / +0.234345 / +0.457603 V; the
    accumulation values differ from v0.15.1's +0.23242 / +0.45703 V by the old Vg/256 bisection
    tolerance (residual -0.0082 / -0.0022 V) and the -1.01 meV E_F change. Pin budget 1e-4 V (outer SP
    tolerance 1e-5 V amplified by the oxide/channel division). Generic film."""
    from dynameta.constants import Q_E
    from dynameta.core.numerics import trapz
    c = _n4_carrier("flat_band_excess")
    _, z, _, n0 = _n4_profile(c, 0.0)
    pins = (-1.141830, -0.523020, -0.250435, 0.234345, 0.457603)
    psis = []
    for vg, pin in zip(N4_VGS, pins):
        psi, _, _, n = _n4_profile(c, vg)
        assert abs(psi + Q_E * trapz(n - n0, z) / c._C_ox - vg) < 1e-6, vg
        assert abs(psi - pin) < 1e-4, (vg, psi)
        psis.append(psi)
    assert np.all(np.sign(psis) == np.sign(N4_VGS)) and np.all(np.diff(psis) > 0.0)


def test_n4_face_field_map_obeys_gauss_law_at_the_oxide_face():
    """gate_map="face_field" (default): continuity of the normal displacement at the oxide face,
    eps_ox (Vg - psi_s)/t_ox = eps_s phi'(face). Checked with the module's conservative flux (root
    accuracy, < 1e-6 V) AND with an independent second-order one-sided difference of the returned
    potential plus Gauss's law written here (measured 4.7e-7 .. 1.7e-5 V; budget 1e-4 V = O(h^2) at
    h = 0.0125 nm). psi_s rises monotonically with Vg and is POSITIVE at Vg = 0 V: the gate-side quantum
    dead layer leaves donor charge whose field ends on the gate. Measured -0.838539 / -0.270639 /
    -0.019089 / +0.217153 (0 V) / +0.442191 / +0.658488 V (the flat-band map, which drops that field,
    gives -1.142 .. +0.458 V). v0.15.1 had no such map: its psi_s at +2 V (0.45703 V) violates this Gauss
    law by -0.93 V. Generic film."""
    from dynameta.carriers.schrodinger_poisson import EPS0 as E0
    c = _n4_carrier("face_field")
    pins = {-4.0: -0.838539, -2.0: -0.270639, -1.0: -0.019089, 0.0: 0.217153, 1.0: 0.442191,
            2.0: 0.658488}
    psis = []
    for vg in sorted(pins):
        psi, z, phi, n = _n4_profile(c, vg)
        h = z[1] - z[0]
        assert abs(psi + c.gate_face_flux(phi, n) / c._C_ox - vg) < 1e-6, vg
        F1 = 9.0 * E0 * (3.0 * phi[-1] - 4.0 * phi[-2] + phi[-3]) / (2.0 * h)
        assert abs(psi + F1 / c._C_ox - vg) < 1e-4, vg
        assert abs(psi - pins[vg]) < 1e-4, (vg, psi)
        assert phi[-1] == psi
        psis.append(psi)
    assert np.all(np.diff(psis) > 0.0)


def test_n4_face_field_is_the_default_and_grid_converged():
    """The default carrier (no gate_map given) uses the face-field map, and its psi_s(+2 V) agrees with
    a brute-force finer-grid solve: nz 241 / 481 / 961 / 1921 measured 0.658447 / 0.658488 / 0.658498 /
    0.658500 V (second-order convergence, steps 41 / 10 / 2 uV). Budget 5e-5 V between nz 481 and
    961 (h = 0.0125 / 0.00625 nm)."""
    from dynameta.carriers.sp_carrier import SchrodingerPoissonCarrier
    c = SchrodingerPoissonCarrier(semi_thk_m=6e-9, n_bg_m3=3e26, m_eff_kg=0.30 * M_E, eps_static=9.0,
                                  nz=481, n_lateral=2, oxide_thk_m=8e-9, eps_oxide=20.0)
    assert c.gate_map == "face_field" and c.fermi_level == "model"
    psi_481 = _n4_profile(c, 2.0)[0]
    psi_961 = _n4_profile(_n4_carrier("face_field", nz=961), 2.0)[0]
    assert abs(psi_481 - psi_961) < 5e-5, (psi_481, psi_961)


def test_one_kernel_census():
    """One implementation per formula: the bulk Fermi level and the oxide-face flux are defined once
    (schrodinger_poisson) and the carrier imports them instead of re-deriving them inline."""
    import pathlib
    import re
    import dynameta
    root = pathlib.Path(dynameta.__file__).parent
    defs = {"sp_bulk_fermi_level_J": 0, "sp_bulk_density_m3": 0, "sp_face_flux_C_m2": 0}
    for p in root.rglob("*.py"):
        src = p.read_text(encoding="utf-8")
        for name in defs:
            defs[name] += len(re.findall(r"^def {}\(".format(name), src, flags=re.M))
    assert all(v == 1 for v in defs.values()), defs
    car = (root / "carriers" / "sp_carrier.py").read_text(encoding="utf-8")
    assert "sp_bulk_fermi_level_J(" in car and "sp_face_flux_C_m2(" in car
    for inline in ("** (2.0 / 3.0)", "phi[-1] - phi[-2]", "phi_V[-1] - phi_V[-2]", "_fermi_log(", "_fd1("):
        assert inline not in car, inline
