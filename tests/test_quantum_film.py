"""Gates for dynameta.optics.quantum_film: the self-consistent (RPA / adiabatic-LDA) out-of-plane response of
a quantum-confined degenerate electron film and its exact planar p-pol element (v0.16).

Every gate here fails on v0.15.1 at `import dynameta.optics.quantum_film` (the module is new); where a
pre-existing model is the comparison, its number is quoted in the gate. Pure numpy/scipy; no solver stack.
Run: python -m pytest tests/test_quantum_film.py -q
"""
import cmath
import functools
import math

import numpy as np
import pytest

from dynameta.constants import C_LIGHT, EPS0, HBAR, KB, M_E, Q_E
from dynameta.optics import nonlocal_tmm as nt
from dynameta.optics.quantum_film import (QuantumFilm, QuantumFilmResponse, film_stack_rt,
                                          film_transfer_matrix, lda_xc_kernel, uniaxial_layer_matrix)

# the generic gated-film fixture
M = 0.30 * M_E
EPS_INF = 4.0
GAM = 1.2e14
N_BG = 3e26
T_FILM = 6e-9
W1200 = 2.0 * math.pi * C_LIGHT / 1200e-9
WPT = math.sqrt(N_BG * Q_E ** 2 / (EPS0 * M) / EPS_INF)      # screened plasma frequency of n_bg
OXB = 2.00 ** 2                                             # the 6 nm cladding oxide
N_GLASS = 1.45


@functools.lru_cache(maxsize=None)
def _flat(t_m, nz):
    return QuantumFilm.flat_band(t_m, N_BG, M, nz=nz)


@functools.lru_cache(maxsize=None)
def _asym(nz=145):
    """6 nm film with an asymmetric (accumulation-like) band profile: no mirror symmetry, so every
    sign and orientation matters."""
    from dynameta.carriers.schrodinger_poisson import sp_bulk_fermi_level_J
    z = np.linspace(0.0, T_FILM, nz)
    phi = 0.35 * np.exp(-(T_FILM - z) / 1.2e-9) - 0.05 * z / T_FILM
    return QuantumFilm.from_potential(z, phi, sp_bulk_fermi_level_J(N_BG, M, 300.0), M)


def _oracle_alpha_perp(film, eps_inf, gamma, omega, f_xc=None):
    """INDEPENDENT oracle (shares no code with the module beyond the inputs): dense numpy eigh of the
    finite-difference Hamiltonian, occupations re-derived (parabolic 2-D Fermi-Dirac), the Lindhard sum
    over ORDERED pairs, the 1-D Coulomb kernel -(e^2/(2 eps0 eps_inf))|z - z'| (instead of the module's
    running-integral form), the external potential energy of a uniform field e (D/(eps0 eps_inf)) z, and
    the dipole p_s = -e int z dn dz. Damping on the velocity: Omega^2 = omega (omega + i gamma)."""
    z = film.z_m
    h = float(z[1] - z[0])
    zi = z[1:-1]
    n = zi.size
    t = HBAR ** 2 / (2.0 * M * h * h)
    H = np.diag(2.0 * t + film.U_J[1:-1]) - t * (np.eye(n, k=1) + np.eye(n, k=-1))
    E, V = np.linalg.eigh(H)
    psi = V / math.sqrt(h)
    kT = KB * film.T_K
    occ = (M * kT / (math.pi * HBAR ** 2)) * np.logaddexp(0.0, (film.fermi_level_J - E) / kT)
    Om = np.sqrt(complex(omega) * (complex(omega) + 1j * gamma))
    chi = np.zeros((n, n), dtype=complex)
    for i in np.flatnonzero(occ > 1e-12 * occ.sum()):
        Z = psi[:, [i]] * psi
        c = occ[i] * (1.0 / (E[i] - E + HBAR * Om) + 1.0 / (E[i] - E - HBAR * Om))
        chi += h * (Z * c[None, :]) @ Z.T
    W = -(Q_E ** 2 / (2.0 * EPS0 * eps_inf)) * np.abs(zi[:, None] - zi[None, :]) * h
    if f_xc is not None:
        W = W + np.diag(f_xc)
    D = 1.0
    dn = np.linalg.solve(np.eye(n) - chi @ W, chi @ (Q_E * D / (EPS0 * eps_inf) * zi))
    return complex(-Q_E * h * np.sum(zi * dn) / D)


# ---- (1) the response itself: independent oracle, cross-route, f-sum ---------------------------------

def test_rpa_alpha_perp_matches_an_independent_oracle_on_both_routes():
    """Asymmetric 6 nm film, 145 nodes, RPA and RPA + LDA xc, at 1200 nm and 0.9 x screened plasma
    frequency: the module's transition-basis ('modes') and Sternheimer ('real_space') routes agree with
    the oracle above to 1e-8 relative (MEASURED <= 9.9e-12 modes, <= 2.3e-12 real space; the discrete
    equations are identical, the algorithms are not)."""
    film = _asym()
    for xc in ("none", "lda"):
        rm = QuantumFilmResponse(film, EPS_INF, GAM, xc=xc, route="modes")
        rr = QuantumFilmResponse(film, EPS_INF, GAM, xc=xc, route="real_space")
        fx = None if xc == "none" else lda_xc_kernel(film.density_m3[1:-1], M, EPS_INF)
        for w in (W1200, 0.9 * WPT):
            ref = _oracle_alpha_perp(film, EPS_INF, GAM, w, f_xc=fx)
            for got in (rm.alpha_perp(w), rr.alpha_perp(w)):
                assert abs(got - ref) < 1e-8 * abs(ref), (xc, w, got, ref)


def test_response_kernels_agree_between_routes_and_with_alpha_perp():
    """The full kernel R (P/eps0 = R d) from the two routes agrees to 1e-8 of its norm (MEASURED 3.3e-11), and its
    uniform-drive sum h sum R equals alpha_perp; the induced charge is neutral and its dipole is
    alpha_perp D (consistency checks)."""
    film = _asym()
    rm = QuantumFilmResponse(film, EPS_INF, GAM, route="modes")
    rr = QuantumFilmResponse(film, EPS_INF, GAM, route="real_space")
    Rm, Rr = rm.response_kernel(W1200), rr.response_kernel(W1200)
    assert np.max(np.abs(Rm - Rr)) < 1e-8 * np.max(np.abs(Rm))
    a = rm.alpha_perp(W1200)
    assert abs(film.h * np.sum(Rr) - a) < 1e-8 * abs(a)
    zi, rho, zh, P = rr.induced_charge(W1200, D_z=1.0)
    assert abs(np.sum(rho) * film.h) < 1e-10 * np.max(np.abs(P))
    assert abs(film.h * np.sum(zi * rho) - a) < 1e-8 * abs(a)          # p_s = int z rho dz


def test_fsum_rule_of_the_full_response():
    """f-sum of the self-consistent response (6 nm flat band, 241 nodes, complete basis). Oracle: the
    finite-difference double-commutator identity C = (e^2/(eps0 eps_inf m)) sum_i n_i kappa_i, kappa_i =
    h sum_k psi_i,k psi_i,k+1, computed here from neighbour products.
      (a) sum of the RPA mode strengths = C to 1e-10 (MEASURED 1.6e-13, also with LDA xc).
      (b) real-space route, beyond every transition: -Omega^2 alpha_perp(omega = 1e21 rad/s) = C to 1e-8
          (MEASURED 3.6e-9, the O(Lambda/omega^2) tail).
      (c) frequency integral (2/pi) int omega Im alpha_perp domega / C = 1 to 1e-4, on a 49-node film with
          gamma 1e15 (all modes resolvable on a log grid), tail beyond the grid added in closed form."""
    film = _flat(T_FILM, 241)
    kappa = film.h * np.sum(film.psi[:-1] * film.psi[1:], axis=0)
    C = Q_E ** 2 / (EPS0 * EPS_INF * M) * float(np.sum(film.sheet_density_m2 * kappa))
    for xc in ("none", "lda"):
        r = QuantumFilmResponse(film, EPS_INF, GAM, xc=xc, route="modes")
        assert abs(np.sum(r.modes()["strength"]) - C) < 1e-10 * C
    rr = QuantumFilmResponse(film, EPS_INF, GAM, route="real_space")
    w = 1e21
    lim = -(w * (w + 1j * GAM)) * rr.alpha_perp(w)
    assert abs(lim - C) < 1e-8 * C, lim / C
    small = _flat(T_FILM, 49)
    rs = QuantumFilmResponse(small, EPS_INF, 1e15, route="modes")
    md = rs.modes()
    ks = small.h * np.sum(small.psi[:-1] * small.psi[1:], axis=0)
    Cs = Q_E ** 2 / (EPS0 * EPS_INF * M) * float(np.sum(small.sheet_density_m2 * ks))
    wg = np.geomspace(1e11, 1e20, 400001)
    O2 = wg * (wg + 1e15j)
    im = np.zeros_like(wg)
    for k in range(0, wg.size, 20000):
        im[k:k + 20000] = np.imag(np.sum(md["strength"][None, :]
                                         / (md["omega2"][None, :] - O2[k:k + 20000, None]), axis=1))
    y = wg * im
    integral = float(np.sum(0.5 * (y[1:] + y[:-1]) * np.diff(wg)))
    tail = Cs * 1e15 / wg[-1]                                          # int_W^inf C gamma / w^2 dw
    assert abs((2.0 / math.pi) * (integral + tail) / Cs - 1.0) < 1e-4


def test_two_level_depolarization_shift_is_the_ando_fowler_stern_formula():
    """Infinite well (5 nm, flat, 401 nodes), ground sub-band occupied, basis cut to the two lowest
    states (max_states=2): the RPA mode is omega~^2 = w21^2 + 2 e^2 dn w21 S/(hbar eps0 eps_inf) with the
    depolarization integral S = int (int_0^z psi1 psi2)^2 dz = 5 L/(9 pi^2) (closed form for sine states;
    Ando-Fowler-Stern 1982 / Allen-Tsui-Vinter 1976). Gate: the shift omega~^2 - w21^2 within 1e-3
    (MEASURED 9.3e-6, the O(h^2) of the grid; E21 = 0.1504 eV, mode 0.2006 eV, shift = 0.78 w21^2). The
    Im alpha_perp peak (gamma 1e12) sits at omega~ to 1e-4.
    The LOCAL model's shift uses z12^2 / L instead of S: ratio 256*9/(81*5 pi^2) = 0.5765 -- the local
    eps_zz (IntersubbandEffect) under-estimates this shift by 42 %, gated here to 1e-3 (MEASURED 5.3e-6). In the COMPLETE
    basis the strongest mode sits at 0.2002 eV with 93 % of the strength (the other transitions mix in). With exchange
    ('x'), omega~^2 = w21^2 (1 + alpha - beta), beta = -(2 dn/(hbar w21)) int zeta^2 f_x with the closed-form
    f_x evaluated here (1e-9)."""
    L = 5e-9
    z = np.linspace(0.0, L, 401)
    film = QuantumFilm.from_potential(z, np.zeros_like(z), 0.08 * Q_E, M)
    r = QuantumFilmResponse(film, EPS_INF, 1e12, max_states=2)
    E = film.energies_J
    w21 = (E[1] - E[0]) / HBAR
    dn = film.sheet_density_m2[0] - film.sheet_density_m2[1]
    S = 5.0 * L / (9.0 * math.pi ** 2)
    shift_afs = 2.0 * Q_E ** 2 * dn * w21 * S / (HBAR * EPS0 * EPS_INF)
    lam = float(r.modes()["omega2"][0])
    assert abs((lam - w21 ** 2) / shift_afs - 1.0) < 1e-3
    wg = np.sqrt(lam) * np.linspace(0.995, 1.005, 20001)
    im = [r.alpha_perp(w).imag for w in wg]
    assert abs(wg[int(np.argmax(im))] / math.sqrt(lam) - 1.0) < 1e-4
    z12 = -16.0 * L / (9.0 * math.pi ** 2)
    assert abs(abs(r.dipoles_m()[0] / z12) - 1.0) < 1e-4            # eigenvector sign is arbitrary
    lam_loc = w21 ** 2 + 2.0 * Q_E ** 2 * dn * w21 * r.dipoles_m()[0] ** 2 / (HBAR * EPS0 * EPS_INF * L)
    assert abs((lam_loc - w21 ** 2) / (lam - w21 ** 2) - 2304.0 / (405.0 * math.pi ** 2)) < 1e-3
    rx = QuantumFilmResponse(film, EPS_INF, 1e12, max_states=2, xc="x")
    n = film.density_m3[1:-1]
    fx = -(1.0 / 3.0) * (3.0 / math.pi) ** (1.0 / 3.0) * n ** (-2.0 / 3.0) * Q_E ** 2 / (4 * math.pi * EPS0 * EPS_INF)
    zeta = film.psi[:, 0] * film.psi[:, 1]
    beta = -(2.0 * dn / (HBAR * w21)) * film.h * np.sum(zeta ** 2 * fx)
    alpha = (lam - w21 ** 2) / w21 ** 2                              # Hartree part, gated above
    assert abs(rx.modes()["omega2"][0] / (w21 ** 2 * (1.0 + alpha - beta)) - 1.0) < 1e-9
    assert beta > 0.0                                                # exchange lowers the mode


def test_local_limit_is_the_part1_effective_eps_zz_and_rpa_differs():
    """The module's uniform-field limit equals IntersubbandEffect ('subbands', Leff = L) to 1e-12
    (consistency); the RPA eps_zz,eq differs from it by 0.17 on the 6 nm flat-band film at 1200 nm
    (MEASURED RPA 2.7673+0.2819i, local 2.8502+0.1396i), i.e. the nonlocal correction is not small.
    """
    from dynameta.carriers.schrodinger_poisson import SubbandResult
    from dynameta.core.effects.electroabsorption import IntersubbandEffect
    film = _flat(T_FILM, 241)
    r = QuantumFilmResponse(film, EPS_INF, GAM)
    res = SubbandResult(energies_J=film.energies_J, psi=film.psi, z_m=film.z_m[1:-1],
                        sheet_density_m2=film.sheet_density_m2)
    e1 = IntersubbandEffect(EPS_INF, M, GAM, GAM, thickness_m=film.thickness_m).eps({"subband": res},
                                                                                     1200e-9)
    exx, ezz, _ = r.effective_layer(W1200)
    assert abs(r.local_eps_zz(W1200) - e1[2, 2]) < 1e-12
    assert abs(exx - e1[0, 0]) < 1e-12
    assert abs(ezz - e1[2, 2]) > 0.08
    assert abs(ezz - (2.7673 + 0.2819j)) < 2e-4


def test_sigma_zz_kernel_reduces_to_the_local_response_for_a_uniform_field():
    """h^2 sum_ab sigma_ab (uniform E) = -i omega eps0 (eps_loc - eps_inf) L (the bare response to a
    uniform field is the local sum, consistency)."""
    film = _asym()
    r = QuantumFilmResponse(film, EPS_INF, GAM)
    s = r.sigma_zz_kernel(W1200)
    lhs = film.h ** 2 * np.sum(s)
    rhs = -1j * W1200 * EPS0 * (r.local_eps_zz(W1200) - EPS_INF) * film.thickness_m
    assert abs(lhs - rhs) < 1e-10 * abs(rhs)


def test_passivity_of_the_response_at_all_frequencies():
    """Im alpha_perp >= 0, Im eps_zz,eq >= 0, Im eps_xx,eq >= 0 from 0.02 to 4 eV (400 frequencies, modes
    route, with and without xc; structural: F_m >= 0 and real Lambda_m > 0) and on 40 of them through
    the real-space route (independent of the eigen-decomposition), on the asymmetric film."""
    film = _asym()
    ws = np.linspace(0.02, 4.0, 400) * Q_E / HBAR
    for xc in ("none", "lda"):
        rm = QuantumFilmResponse(film, EPS_INF, GAM, xc=xc, route="modes")
        for w in ws:
            exx, ezz, _ = rm.effective_layer(w)
            assert rm.alpha_perp(w).imag >= 0.0 and ezz.imag >= 0.0 and exx.imag >= 0.0
    rr = QuantumFilmResponse(film, EPS_INF, GAM, route="real_space")
    for w in ws[::10]:
        a = rr.alpha_perp(w)
        assert a.imag >= -1e-12 * abs(a)


# ---- (2) the planar element -------------------------------------------------------------------------

def test_uniaxial_matrix_and_film_machinery_against_closed_forms():
    """(a) uniaxial_layer_matrix with eps_xx = eps_zz equals nonlocal_tmm's isotropic Abeles matrix to
    1e-13 entry-wise relative; (b) film_transfer_matrix fed a LOCAL graded profile (R = diag(1 - eps_inf/eps_zz(s))) equals the
    cascade of closed-form uniaxial cell matrices to 1e-6 entry-wise (both second order in h), and a
    uniform profile equals one uniaxial matrix to 1e-8 (the staggered scheme's O((k h)^2))."""
    k0 = W1200 / C_LIGHT
    kp = 1.3 * k0 * 0.8
    for e in (2.1 + 0.3j, -3.0 + 0.5j):
        A_ = nt._local_char_matrix(e, 7e-9, k0, kp, "p")
        B_ = uniaxial_layer_matrix(e, e, 7e-9, k0, kp)
        assert np.max(np.abs(A_ - B_) / np.abs(A_)) < 1e-13            # entry-wise relative
    N, L = 401, 5e-9
    h = L / (N - 1)
    s = np.linspace(0.0, L, N)
    sh = 0.5 * (s[:-1] + s[1:])
    exx = 2.5 - 1.8 * np.exp(-s / 1.1e-9) + 0.2j
    ezz = 1.9 - 2.2 * np.exp(-sh / 0.9e-9) + 0.3j
    T = film_transfer_matrix(W1200, kp, h, exx, EPS_INF, np.diag(1.0 - EPS_INF / ezz))
    ref = np.eye(2, dtype=complex)
    for a in range(N - 1):
        ref = uniaxial_layer_matrix(0.5 * (exx[a] + exx[a + 1]), ezz[a], h, k0, kp) @ ref
    assert np.max(np.abs(T - ref) / np.abs(ref)) < 1e-6               # entry-wise relative
    Tu = film_transfer_matrix(W1200, kp, h, np.full(N, 2.2 + 0.1j), EPS_INF,
                              np.diag(np.full(N - 1, 1.0 - EPS_INF / (0.4 + 0.2j))))
    Uu = uniaxial_layer_matrix(2.2 + 0.1j, 0.4 + 0.2j, L, k0, kp)
    assert np.max(np.abs(Tu - Uu) / np.abs(Uu)) < 1e-8                # entry-wise relative


def test_p_reflection_sign_of_film_stack_rt_against_nonlocal_tmm_and_tmm():
    """ONE sign of r_p (solver-free part; the comparison with solve_film_cell and fem_2d_tm is
    tests/test_quantum_film_cell.py::test_p_reflection_sign_agrees_with_solve_film_cell_and_fem_2d_tm):
      * QuantumFilmResponse.stack_rt (model 'equivalent') is film_stack_rt with the equivalent layer, bit for bit;
      * nonlocal_tmm.stack_rt (unchanged since v0.15.1, tangential-E ratio) is EXACTLY the negative of
        film_stack_rt (p-hat convention r = r_H): with the film given as the identity matrix both share every
        operation, so r_film_stack_rt == -r_nonlocal bit for bit;
      * the external anchor: Byrnes `tmm` (the p-hat convention of tmm_reference) on an isotropic stack equals
        film_stack_rt to 1e-13 (MEASURED ~1e-16).
    abs(r) is about 0.1 here, so a flipped sign misses the tmm anchor by about 0.2."""
    lam, th = 1200e-9, 50.0
    w = 2.0 * math.pi * C_LIGHT / lam
    q = QuantumFilmResponse(_flat(T_FILM, 201), EPS_INF, GAM)
    exx, ezz, L = q.effective_layer(w)
    cap, ox = nt.DielectricLayer(4.0, 20e-9), nt.DielectricLayer(1.7 ** 2, 20e-9)
    kw = dict(n_super=1.0, n_sub=1.45, theta_rad=math.radians(th))
    a = q.stack_rt(w, [cap], [ox], model="equivalent", **kw)
    b = film_stack_rt(w, lambda ww, kp: uniaxial_layer_matrix(exx, ezz, L, ww / C_LIGHT, kp), [cap], [ox], **kw)
    assert a.r == b.r
    # the v0.15.1 function: tangential-E ratio, exactly the negative
    eye = np.eye(2, dtype=complex)
    iso = [cap, nt.DielectricLayer(2.2 + 0.3j, L), ox]
    old = nt.stack_rt(w, iso, pol="p", **kw)
    new = film_stack_rt(w, eye, iso, [], **kw)
    assert new.r == -old.r and new.R == old.R and new.T == old.T
    assert abs(new.r) > 0.05                                         # the sign is resolvable here
    tmm = pytest.importorskip("tmm")
    ref = tmm.coh_tmm("p", [1.0, 2.0, cmath.sqrt(2.2 + 0.3j), 1.7, 1.45], [np.inf, 20.0, L * 1e9, 20.0, np.inf],
                      math.radians(th), lam * 1e9)["r"]
    assert abs(new.r - ref) < 1e-13, abs(new.r - ref)


def test_film_orientation_is_reciprocal():
    """Turning the asymmetric film over is the reciprocal inverse: T_bottom = J T_top^-1 J,
    J = diag(1, -1), element by element to 1e-13 of each element (MEASURED <= 2e-15; the elements span
    fourteen decades, so an absolute bar would test only the largest; catches any orientation or sign
    slip of the nonlocal kernel)."""
    r = QuantumFilmResponse(_asym(), EPS_INF, GAM)
    kp = 0.8 * 1.3 * W1200 / C_LIGHT
    Tt = r.transfer_matrix(W1200, kp, superstrate_face="top")
    Tb = r.transfer_matrix(W1200, kp, superstrate_face="bottom")
    J = np.diag([1.0, -1.0])
    assert np.max(np.abs(Tb - J @ np.linalg.inv(Tt) @ J) / np.abs(Tb)) < 1e-13
    assert np.max(np.abs(Tb - Tt)) > 1e-6                            # the film IS asymmetric


def test_energy_conservation_of_the_film_element():
    """(a) the discrete Poynting theorem of the film solve holds to 1e-12 for an arbitrary face state;
    (b) in the stack air | 6 nm oxide | film | 6 nm oxide | glass (60 deg p, lossless claddings),
    A = 1 - R - T equals the film's internal dissipation (sum of Re J.E*/2 over the film / incident
    flux) to 1e-10; (c) gamma = 0: A = 0 to 1e-12 (MEASURED <= 1.0e-15; no spurious gain or loss)."""
    film = _asym()
    r = QuantumFilmResponse(film, EPS_INF, GAM)
    kp = 1.0 * W1200 / C_LIGHT * math.sin(math.radians(60))
    exx = r.eps_xx_nodes(W1200)[::-1]
    R = r.response_kernel(W1200)[::-1, ::-1]
    T, fields = film_transfer_matrix(W1200, kp, film.h, exx, EPS_INF, R, return_solver=True)
    w = np.ones(film.z_m.size)
    w[0] = w[-1] = 0.5

    def check(E0, H0):
        f = fields(E0, H0)
        lhs = (np.conj(H0) * E0 - np.conj(f["Ht_faces"][1]) * f["E_x"][-1]).real
        diss = (np.sum(w * film.h * exx.imag * np.abs(f["E_x"]) ** 2)
                + np.sum(film.h * (f["P_over_eps0"] * np.conj(f["E_z"])).imag))
        return lhs, diss
    lhs, diss = check(0.7 - 0.2j, 0.3 + 1.1j)
    assert abs(lhs - diss) < 1e-12 * abs(diss)
    above = [nt.DielectricLayer(OXB, 6e-9)]
    below = [nt.DielectricLayer(OXB, 6e-9)]
    res = r.stack_rt(W1200, above, below, n_sub=N_GLASS, theta_rad=math.radians(60))
    k0 = W1200 / C_LIGHT
    Ysup = nt._admittance(1.0, nt._kz(1.0, k0, kp), "p")
    # res.r is the p-hat r_H = -E_x(refl)/E_x(inc): the superstrate face carries E = 1 - r, Ht = Y (1 + r)
    top = nt._stack_matrix(W1200, above, k0, kp, "p") @ np.array([1.0 - res.r, Ysup * (1.0 + res.r)])
    _, diss = check(top[0], top[1])
    assert abs(diss / Ysup.real - res.A) < 1e-10
    r0 = QuantumFilmResponse(film, EPS_INF, 0.0)
    for lam in (1150e-9, 1200e-9, 1450e-9):
        w0 = 2.0 * math.pi * C_LIGHT / lam
        assert abs(r0.stack_rt(w0, above, below, n_sub=N_GLASS, theta_rad=math.radians(60)).A) < 1e-12


def test_exact_element_and_equivalent_layer_agree_to_first_order():
    """The thin-film equivalent uniaxial layer (eps_xx,eq, eps_zz,eq, L) reproduces the exact element's r
    on the 6 nm flat-band film at 1200 nm, 60 deg, to 1e-5 (MEASURED 6.7e-7 = O((k0 L)^2) relative)."""
    r = QuantumFilmResponse(_flat(T_FILM, 241), EPS_INF, GAM)
    above = [nt.DielectricLayer(OXB, 6e-9)]
    below = [nt.DielectricLayer(OXB, 6e-9)]
    kw = dict(n_sub=N_GLASS, theta_rad=math.radians(60))
    ex = r.stack_rt(W1200, above, below, **kw)
    eq = r.stack_rt(W1200, above, below, model="equivalent", **kw)
    assert abs(ex.r - eq.r) < 1e-5
    assert abs(ex.r - eq.r) > 1e-9                                   # not trivially the same object


# ---- (3) limits: wide film, hydrodynamics -------------------------------------------------------------

def test_wide_film_limit_approaches_the_local_drude():
    """Flat-band films 10 / 20 / 40 nm (h 0.05 nm) at 1200 nm: abs(eps_zz,eq - eps_D) (eps_D the bulk
    Drude with the same mass) falls strictly, 20 -> 40 nm ratio in [1.5, 2.3] (1/L: 2; MEASURED 0.0922 /
    0.0381 / 0.0211, ratio 1.81), and eps_xx,eq -> eps_D likewise (0.1449 / 0.0728 / 0.0362: the dead
    layer at the walls, a 1/L sheet deficit). The 40 nm bar is 1.3x the measured value, as before.
    """
    wp2 = N_BG * Q_E ** 2 / (EPS0 * M)
    eD = EPS_INF - wp2 / (W1200 * (W1200 + 1j * GAM))
    dz, dx = [], []
    for t, nz in ((10e-9, 201), (20e-9, 401), (40e-9, 801)):
        exx, ezz, _ = QuantumFilmResponse(_flat(t, nz), EPS_INF, GAM, route="real_space").effective_layer(W1200)
        dz.append(abs(ezz - eD))
        dx.append(abs(exx - eD))
    assert dz[0] > dz[1] > dz[2] and 1.5 <= dz[1] / dz[2] <= 2.3, dz
    assert dx[0] > dx[1] > dx[2] and 1.8 <= dx[1] / dx[2] <= 2.2, dx
    assert dz[2] < 0.028


def _hydro_compare(film, w):
    z, n = film.z_m, film.density_m3.copy()
    n[0] = n[-1] = 1e-9 * n.max()                                    # hard-wall face nodes (n = 0)
    hyd = nt.hydro_layers_from_profile(z, n, m_eff_kg=M, eps_inf=EPS_INF, gamma=GAM)
    loc = [nt.DielectricLayer(complex(nt.eps_transverse(w, L_)), L_.thickness_m) for L_ in hyd]
    above = [nt.DielectricLayer(OXB, 6e-9)]
    below = [nt.DielectricLayer(OXB, 6e-9)]
    kw = dict(n_sub=N_GLASS, theta_rad=math.radians(60))
    rr = QuantumFilmResponse(film, EPS_INF, GAM).stack_rt(w, above, below, **kw).r
    # nonlocal_tmm.stack_rt returns the tangential-E r (the negative of the p-hat r of QuantumFilmResponse)
    rl = -nt.stack_rt(w, above + loc + below, **kw).r
    rh = -nt.stack_rt(w, above + hyd + below, **kw).r
    u = nt.hydro_validity(w, [hyd[len(hyd) // 2]]).u[0]
    return rr, rl, rh, u


def test_hydrodynamics_agrees_where_valid_and_not_where_it_is_not():
    """Same density profile (6 nm flat band, 121 nodes) in the hydrodynamic run (nonlocal_tmm, 'high_freq'
    beta, graded, hard walls) and in the RPA element, stack as above. Near the screened plasma
    frequency (omega/w~p = 0.95 / 1.0 / 1.05, centre u = |k_L| v_F/omega = 0.58 / 0.48 / 0.49) the two
    nonlocal corrections r - r_local agree to 10 % (MEASURED 1.4 / 2.6 / 5.9 %). Where hydrodynamics is
    invalid (1.5 w~p, u = 0.96; the 1200 nm regime is u = 1.06) they differ by 66 % of the correction
    (gated > 30 %): Landau damping and the confined particle-hole spectrum, absent from hydrodynamics.
    """
    film = _flat(T_FILM, 121)
    for x in (0.95, 1.0, 1.05):
        rr, rl, rh, u = _hydro_compare(film, x * WPT)
        assert u < 0.6
        assert abs(rr - rh) < 0.10 * abs(rr - rl), (x, abs(rr - rh) / abs(rr - rl))
    rr, rl, rh, u = _hydro_compare(film, 1.5 * WPT)
    assert u > 0.9 and abs(rr - rh) > 0.3 * abs(rr - rl)


# ---- (4) ground state, gate bias, kernels, validity ------------------------------------------------

def test_dual_gate_accessor_and_bias_dependence():
    """QuantumFilm.from_dual_gate recomputes the complete sub-band set from the converged potential: its
    density equals the solver's to 1e-9 n_bg. 6 nm film, 6 nm eps 20 oxides on both faces, 241 nodes:
    accumulation (+2, +2 V) raises the sheet density and depletion (-2, -2 V) lowers it, and the RPA
    eps_zz,eq at 1200 nm moves accordingly (MEASURED 2.9850+0.1455i / 2.5963+0.4622i / 2.3696+0.5655i at
    -2 / 0 / +2 V); A is positive in every case."""
    from dynameta.carriers import DualGateSchrodingerPoisson, GateStack
    s = DualGateSchrodingerPoisson(t_film_m=T_FILM, bottom=GateStack((6e-9, 20.0)),
                                   top=GateStack((6e-9, 20.0)), n_bg_m3=N_BG, m_eff_kg=M,
                                   eps_static=9.0, nz=241)
    ns, ezz = [], []
    above = [nt.DielectricLayer(OXB, 6e-9)]
    below = [nt.DielectricLayer(OXB, 6e-9)]
    for v in (-2.0, 0.0, 2.0):
        sol = s.solve_voltages(v, v)
        film = QuantumFilm.from_dual_gate(s, sol)
        assert np.max(np.abs(film.density_m3 - sol.n_m3)) < 1e-9 * N_BG
        r = QuantumFilmResponse(film, EPS_INF, GAM)
        ns.append(film.sheet_density_total_m2)
        ezz.append(r.effective_layer(W1200)[1])
        assert r.stack_rt(W1200, above, below, n_sub=N_GLASS, theta_rad=math.radians(60)).A > 0.0
    assert ns[0] < ns[1] < ns[2]
    assert ezz[0].real > ezz[1].real > ezz[2].real                  # more electrons, lower Re eps_zz


def test_inplane_drude_weight_parabolic_and_kane():
    """alpha = 0: D_i = e^2 n_i / m exactly. Kane (alpha 0.4 eV^-1): the module's Simpson integral equals
    an independent scipy.quad of (e^2/(2 pi hbar^2)) g int (-df/dE) E (1 + aE)/(1 + 2aE) dE to 1e-8, and
    lies below e^2 n_i / m (the heavier in-plane Fermi-level mass)."""
    from scipy.integrate import quad
    f0 = _flat(T_FILM, 121)
    assert np.allclose(f0.drude_weights(), Q_E ** 2 * f0.sheet_density_m2 / M, rtol=1e-14, atol=0.0)
    fk = QuantumFilm.flat_band(T_FILM, N_BG, M, nz=121, alpha_np_per_eV=0.4)
    a = 0.4 / Q_E
    kT = KB * fk.T_K
    D = fk.drude_weights()
    for i in range(3):
        eta = (fk.fermi_level_J - fk.energies_J[i]) / kT
        g = lambda x: (0.25 / math.cosh(0.5 * (x - eta)) ** 2) * (x * kT) * (1 + a * x * kT) / (1 + 2 * a * x * kT)  # noqa: E731
        val, _ = quad(g, 0.0, max(eta, 0.0) + 60.0, points=[max(eta, 0.0)], epsabs=0.0, epsrel=1e-12, limit=200)
        ref = Q_E ** 2 * 2 / (2 * math.pi * HBAR ** 2) * val
        assert abs(D[i] / ref - 1.0) < 1e-8
        assert D[i] < Q_E ** 2 * fk.sheet_density_m2[i] / M


def test_lda_kernel_exchange_closed_form_and_correlation_derivative():
    """f_x equals the closed form -(1/3)(3/pi)^(1/3) n^(-2/3) e^2/(4 pi eps0 eps) to 1e-12; the PZ81
    correlation part equals a finite-difference second derivative of n eps_c(n) (eps_c written here
    from the PZ81 table) to 1e-6, on both sides of r_s = 1."""
    n = np.array([1e24, 1e25, 3e26, 2e27, 1e28])
    fx = lda_xc_kernel(n, M, EPS_INF, model="x")
    ref = -(1.0 / 3.0) * (3.0 / math.pi) ** (1.0 / 3.0) * n ** (-2.0 / 3.0) * Q_E ** 2 / (4 * math.pi * EPS0 * EPS_INF)
    assert np.max(np.abs(fx / ref - 1.0)) < 1e-12
    a_s = 4 * math.pi * EPS0 * EPS_INF * HBAR ** 2 / (M * Q_E ** 2)
    ha = Q_E ** 2 / (4 * math.pi * EPS0 * EPS_INF * a_s)

    def en_c(nn):
        rs = (3.0 / (4.0 * math.pi * nn)) ** (1.0 / 3.0) / a_s
        ec = np.where(rs >= 1.0, -0.1423 / (1 + 1.0529 * np.sqrt(rs) + 0.3334 * rs),
                      0.0311 * np.log(rs) - 0.048 + 0.0020 * rs * np.log(rs) - 0.0116 * rs)
        return nn * ec * ha
    fc = lda_xc_kernel(n, M, EPS_INF, model="lda") - fx
    for k, nn in enumerate(n):
        dd = 1e-3 * nn
        num = (en_c(nn + dd) - 2 * en_c(nn) + en_c(nn - dd)) / dd ** 2
        assert abs(fc[k] / num - 1.0) < 1e-6, (nn, fc[k], num)
    rs = (3.0 / (4.0 * math.pi * n)) ** (1.0 / 3.0) / a_s
    assert rs.min() < 1.0 < rs.max()
    assert lda_xc_kernel(np.array([0.0]), M, EPS_INF)[0] == 0.0


def test_hamiltonian_and_validity_report():
    """The real-space route's tridiagonal Hamiltonian reproduces the film's sub-band energies (1e-9 of
    E_F, consistency); validity() reports q/k_F and q v_F/omega at 60 deg / 1200 nm well below 0.1, and
    warns when q approaches k_F."""
    from scipy.linalg import eigh_tridiagonal
    from dynameta.warning_types import UnreliableResultWarning
    film = _asym()
    r = QuantumFilmResponse(film, EPS_INF, GAM)
    d, o = r._hamiltonian_bands()
    E = eigh_tridiagonal(d, np.full(d.size - 1, o), eigvals_only=True)
    assert np.max(np.abs(E - film.energies_J)) < 1e-9 * film.fermi_level_J
    v = r.validity(W1200, W1200 / C_LIGHT * math.sin(math.radians(60)))
    assert v.ok and v.q_over_kF < 1e-2 and v.q_vF_over_omega < 1e-2
    with pytest.warns(UnreliableResultWarning, match="q -> 0"):
        assert not r.validity(W1200, 1e9).ok


def test_refusals():
    film = _flat(T_FILM, 49)
    with pytest.raises(ValueError, match="xc"):
        QuantumFilmResponse(film, EPS_INF, GAM, xc="gga")
    with pytest.raises(ValueError, match="route"):
        QuantumFilmResponse(film, EPS_INF, GAM, route="fast")
    with pytest.raises(ValueError, match="eps_inf"):
        QuantumFilmResponse(film, -1.0, GAM)
    with pytest.raises(ValueError, match="complete basis"):
        QuantumFilmResponse(film, EPS_INF, GAM, max_states=3).chi0(W1200)
    with pytest.raises(ValueError, match="superstrate_face"):
        QuantumFilmResponse(film, EPS_INF, GAM).transfer_matrix(W1200, 0.0, superstrate_face="left")
    with pytest.raises(ValueError, match="model"):
        QuantumFilmResponse(film, EPS_INF, GAM).stack_rt(W1200, model="hydro")
    assert film_stack_rt is not None
