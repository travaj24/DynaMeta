"""Gates for the dual-gate Schrodinger-Poisson carrier solver (carriers/sp_dual_gate.py, v0.16).

A thin degenerate film gated through oxide stacks on BOTH faces; the film's Fermi level is the grounded
reference and the face potentials are fixed by Gauss's law at each face. v0.15.1 had no such solver: the
single-gate SchrodingerPoissonCarrier pins phi = 0 at the far face (Dirichlet body), which for the
6 nm / 8 nm-oxide film below violates the far-face Gauss law of a gated face by ~2e-2 C/m^2 at 0 V
(test_far_face_dirichlet_violates_gauss documents the number).

Film: the generic gated film -- 6 nm, n_bg 3e26 m^-3, m* 0.30 m_e, eps_s 9.0, 300 K, h = 0.0125 nm
(481 nodes); symmetric oxides 8 nm / eps 20. Every measured number below is of this film.

Oracles (independent of the solver's code path unless labelled a consistency check):
  * Gauss's law re-derived here at both faces, with the half-cell flux AND an independent one-sided
    second-order difference of the returned potential;
  * a brute-force solver written in this file: dense Hamiltonian + numpy eigh, oxide stacks MESHED
    with the gates as Dirichlet nodes (no capacitance formula, no face-flux kernel, no outer Newton),
    the Fermi level from N_c [F_1/2 + 2 a kT F_3/2] by scipy quadrature, a plain Newton with a dense
    finite-difference Jacobian;
  * the single-gate solver of v0.16 (SchrodingerPoissonCarrier, face-field map) in the limit of a far
    gate behind a very thick oxide;
  * a classical Fermi-Dirac Poisson solve written here (F_1/2 by quadrature) for the wide-film limit.

Run: python -m pytest tests/test_sp_dual_gate.py -q
"""
import numpy as np
import pytest

from dynameta.carriers.sp_dual_gate import (
    DualGateConvergenceError, DualGateSchrodingerPoisson, GateStack)
from dynameta.constants import EPS0, HBAR, KB, M_E, Q_E

MSTAR = 0.30 * M_E
NBG = 3e26
EPS_S = 9.0
T = 300.0


def _f_fd(j, x):
    """Normalized Fermi-Dirac integral F_j(x) by scipy quadrature (t = u^2 substitution for j < 0)."""
    from scipy.integrate import quad
    from scipy.special import gamma
    up = max(x, 0.0) + 60.0
    if j < 0:
        f = lambda u: 2.0 * u ** (2 * j + 1) / (1.0 + np.exp(min(u * u - x, 700.0)))   # noqa: E731
        val = quad(f, 0.0, np.sqrt(up), epsabs=0.0, epsrel=1e-13, limit=400)[0]
    else:
        f = lambda t: t ** j / (1.0 + np.exp(min(t - x, 700.0)))                       # noqa: E731
        pts = [x] if 0.0 < x < up else None
        val = quad(f, 0.0, up, points=pts, epsabs=0.0, epsrel=1e-13, limit=400)[0]
    return val / gamma(j + 1)


def _asym_gates():
    """Asymmetric stacks: two-layer bottom oxide with a work-function offset and negative interface
    charge, single-layer top oxide with the opposite offset and positive charge."""
    bot = GateStack(((4e-9, 3.9), (6e-9, 22.0)), work_function_offset_V=0.3, interface_charge_C_m2=-5e-3)
    top = GateStack(((8e-9, 20.0),), work_function_offset_V=-0.2, interface_charge_C_m2=2e-3)
    return bot, top


def _solver(bot, top, **kw):
    kw.setdefault("t_film_m", 6e-9)
    kw.setdefault("nz", 481)
    return DualGateSchrodingerPoisson(bottom=bot, top=top, n_bg_m3=kw.pop("n_bg_m3", NBG),
                                      m_eff_kg=kw.pop("m_eff_kg", MSTAR), eps_static=EPS_S, T_K=T, **kw)


def _face_laws(s, sol):
    """Gauss's law at both faces, re-derived here: half-cell flux of the end node (z = 0: oxide below,
    z = t: oxide above), the gate charge C (phi_g - phi_face) with C from the layers, the interface charge.
    Returns (R_b, R_t, R_b_onesided, R_t_onesided) in C/m^2."""
    phi, n, z = sol.phi_V, sol.n_m3, sol.z_m
    h = z[1] - z[0]
    ee = EPS_S * EPS0
    Fb = ee * (phi[1] - phi[0]) / h + Q_E * 0.5 * h * (NBG - n[0])
    Ft = ee * (phi[-1] - phi[-2]) / h - Q_E * 0.5 * h * (NBG - n[-1])
    Fb1 = ee * (-3.0 * phi[0] + 4.0 * phi[1] - phi[2]) / (2.0 * h)
    Ft1 = ee * (3.0 * phi[-1] - 4.0 * phi[-2] + phi[-3]) / (2.0 * h)
    out = []
    for F_b, F_t in ((Fb, Ft), (Fb1, Ft1)):
        Rb = -F_b - s.bottom.interface_charge_C_m2 if s.bottom is None else (
            1.0 / sum(t / (EPS0 * e) for t, e in s.bottom.layers)
            * (sol.phi_bottom_V - (sol.V_bottom - s.bottom.work_function_offset_V))
            - F_b - s.bottom.interface_charge_C_m2)
        Rt = (F_t + 1.0 / sum(t / (EPS0 * e) for t, e in s.top.layers)
              * (sol.phi_top_V - (sol.V_top - s.top.work_function_offset_V))
              - s.top.interface_charge_C_m2)
        out += [Rb, Rt]
    return out


# ---------------------------------------------------------------------------------------------------
# Gauss's law, neutrality, signs
# ---------------------------------------------------------------------------------------------------

ASYM_BIASES = ((0.0, 0.0), (2.0, -1.0), (-3.0, 2.0), (-4.0, -4.0), (3.0, 3.0), (1.5, 0.0))


def test_gauss_law_at_both_faces_and_charge_balance():
    """Gauss's law at both faces to the solver tolerance (1e-9 C/m^2) for biases of both signs and
    asymmetric pairs, with asymmetric stacks, work-function offsets and interface charges. Re-derived
    half-cell flux: < 1e-9 C/m^2 (measured <= 9.1e-10). Independent one-sided second-order flux: budget
    2e-6 C/m^2 (O(h^2) at h = 0.0125 nm; measured <= 5.6e-7). Charge balance
    Q_gate_b + Q_gate_t + sigma_b + sigma_t + q trapz(Nd - n) = 0 recomputed here from the returned
    profile: budget 2e-9 C/m^2 (inner self-consistency 1e-11 V; measured <= 4.1e-10)."""
    bot, top = _asym_gates()
    s = _solver(bot, top)
    Cb, Ct = bot.capacitance_F_m2, top.capacitance_F_m2
    for V in ASYM_BIASES:
        sol = s.solve_voltages(*V)
        Rb, Rt, Rb1, Rt1 = _face_laws(s, sol)
        assert max(abs(Rb), abs(Rt)) < 1e-9, (V, Rb, Rt)
        assert max(abs(Rb1), abs(Rt1)) < 2e-6, (V, Rb1, Rt1)
        Qgb = Cb * (V[0] - 0.3 - sol.phi_bottom_V)
        Qgt = Ct * (V[1] + 0.2 - sol.phi_top_V)
        y = NBG - sol.n_m3                                            # trapezoid written out here
        Qf = Q_E * 0.5 * float(np.sum((y[1:] + y[:-1]) * np.diff(sol.z_m)))
        assert abs(Qgb + Qgt - 5e-3 + 2e-3 + Qf) < 2e-9, V
        assert sol.n_m3[0] == 0.0 and sol.n_m3[-1] == 0.0            # hard walls at both faces
        assert sol.phi_V[0] == sol.phi_bottom_V and sol.phi_V[-1] == sol.phi_top_V


def test_sign_pins_gate_charge_and_density_follow_each_gate():
    """Sign-sensitive: raising one gate's voltage raises that face's potential, the electron density at
    that face, the electron sheet excess and that gate's (positive) charge, and moves the other face
    much less; the gate charge has the sign of V - V_wf - phi_face."""
    s = _solver(GateStack((8e-9, 20.0)), GateStack((8e-9, 20.0)))
    base = s.solve_voltages(0.0, 0.0)
    up = s.solve_voltages(0.0, 2.0)
    dn = s.solve_voltages(0.0, -2.0)
    near = slice(-81, None)                                           # top 1 nm
    assert up.phi_top_V > base.phi_top_V > dn.phi_top_V
    assert np.sum(up.n_m3[near]) > np.sum(base.n_m3[near]) > np.sum(dn.n_m3[near])
    assert up.sheet_excess_m2 > base.sheet_excess_m2 > dn.sheet_excess_m2
    assert up.gate_charge_top_C_m2 > 0.0 > dn.gate_charge_top_C_m2
    assert abs(up.phi_bottom_V - base.phi_bottom_V) < 0.5 * abs(up.phi_top_V - base.phi_top_V)
    # bottom face above flat band at 0 V: the dead-layer donor charge's field ends on both gates
    assert base.phi_bottom_V > 0.0 and base.gate_charge_bottom_C_m2 < 0.0


def test_far_face_dirichlet_violates_gauss():
    """What v0.15.1 / the single-gate class does at the far face: phi = 0 pinned there. For the symmetric
    6 nm film with 8 nm / eps 20 oxides on both faces at 0 V, the single-gate column's far-face flux
    violates the gated-face law C (phi_face - phi_g) - F = 0 by -2.03e-2 C/m^2 (its film carries a 0 V
    sheet excess of -1.57e17 m^-2 against the dual-gate -5.97e16 m^-2); the dual-gate solution meets it
    to 1e-9 and is mirror-symmetric instead."""
    from dynameta.carriers.sp_carrier import SchrodingerPoissonCarrier
    from dynameta.carriers.schrodinger_poisson import sp_face_flux_C_m2
    from dynameta.sweep import BiasPoint
    c = SchrodingerPoissonCarrier(semi_thk_m=6e-9, n_bg_m3=NBG, m_eff_kg=MSTAR, eps_static=EPS_S, nz=481,
                                  n_lateral=2, oxide_thk_m=8e-9, eps_oxide=20.0)
    reg = c.solve(BiasPoint({"gate": 0.0, "body": 0.0}, "0")).regions["semi"]
    phi = reg.grid_fields["potential_V"][0, 0]
    n = reg.grid_fields["electron_density_m3"][0, 0]
    Fb = sp_face_flux_C_m2(phi, n, np.full(481, NBG), 6e-9 / 480, EPS_S, side="left")
    C = 20.0 * EPS0 / 8e-9
    viol = C * (phi[0] - 0.0) - Fb
    assert abs(viol) > 1e-2, viol
    s = _solver(GateStack((8e-9, 20.0)), GateStack((8e-9, 20.0)))
    sol = s.solve_voltages(0.0, 0.0)
    assert max(abs(sol.gauss_residual_bottom_C_m2), abs(sol.gauss_residual_top_C_m2)) < 1e-9


# ---------------------------------------------------------------------------------------------------
# independent brute-force oracle
# ---------------------------------------------------------------------------------------------------

def _brute_force(t, nz, bot, top, Vb, Vt, alpha=0.0, n_ox=8):
    """Dual-gate SP written independently: oxide stacks meshed (n_ox intervals per layer), gate metals
    Dirichlet at V - V_wf, sheet charges as interface-node sources, control-volume Poisson through film
    and oxide, dense-eigh Schrodinger in the film (hard walls), Fermi level from N_c [F_1/2 +
    2 a kT F_3/2] = n_bg by quadrature, Newton with a dense finite-difference Jacobian."""
    from scipy.linalg import eigh
    from scipy.optimize import brentq
    kT = KB * T
    a = alpha / Q_E
    Nc = 2.0 * (MSTAR * kT / (2.0 * np.pi * HBAR ** 2)) ** 1.5
    EF = brentq(lambda E: Nc * (_f_fd(0.5, E / kT) + 2.0 * a * kT * _f_fd(1.5, E / kT)) - NBG,
                -1.0 * Q_E, 3.0 * Q_E, xtol=1e-30)
    h = t / (nz - 1)
    zb, eb, db = [0.0], [], []
    pos = 0.0
    for tl, el in bot.layers:
        for _ in range(n_ox):
            pos -= tl / n_ox
            zb.append(pos); eb.append(el); db.append(tl / n_ox)
    zb, eb, db = zb[::-1], eb[::-1], db[::-1]
    zt, et, dt = [], [], []
    pos = t
    for tl, el in top.layers:
        for _ in range(n_ox):
            pos += tl / n_ox
            zt.append(pos); et.append(el); dt.append(tl / n_ox)
    eps_iv = np.array(eb + [EPS_S] * (nz - 1) + et) * EPS0
    dz_iv = np.array(db + [h] * (nz - 1) + dt)
    i0 = len(zb) - 2                                   # unknowns: bottom-oxide interior, film, top interior
    i1 = i0 + nz - 1
    M = (len(zb) - 2) + nz + (len(zt) - 1)
    phi_g = (Vb - bot.work_function_offset_V, Vt - top.work_function_offset_V)
    c = HBAR ** 2 / (2.0 * MSTAR * h ** 2)
    pref = 2.0 * MSTAR / (2.0 * np.pi * HBAR ** 2)
    gl_u, gl_w = np.polynomial.legendre.leggauss(200)        # F_1 by Gauss-Legendre, split at the Fermi edge
    gl_u, gl_w = 0.5 * (gl_u + 1.0), 0.5 * gl_w

    def density(phif):
        H = (np.diag(2.0 * c - Q_E * phif[1:-1]) + np.diag(-c * np.ones(nz - 3), 1)
             + np.diag(-c * np.ones(nz - 3), -1))
        E, V = eigh(H)
        V = V / np.sqrt(np.sum(V ** 2, axis=0) * h)
        x = (EF - E) / kT
        ns = pref * kT * np.logaddexp(0.0, x)
        if a > 0.0:
            edge = np.maximum(x, 0.0)[:, None]
            F1 = np.zeros_like(x)
            for lo, ln in ((0.0 * edge, edge), (edge, 60.0 + 0.0 * edge)):   # [0, edge], [edge, edge + 60]
                tg = lo + ln * gl_u[None, :]
                F1 += np.sum(ln * gl_w[None, :] * tg / (1.0 + np.exp(np.clip(tg - x[:, None], -700, 700))),
                             axis=1)
            ns = ns + pref * 2.0 * a * kT ** 2 * F1
        n = np.zeros(nz)
        n[1:-1] = (V ** 2) @ ns
        return n

    cvw = np.full(nz, h)
    cvw[0] = cvw[-1] = 0.5 * h

    def resid(u):
        phi = np.concatenate([[phi_g[0]], u, [phi_g[1]]])
        flux = eps_iv * np.diff(phi) / dz_iv
        n = density(u[i0:i1 + 1])
        src = np.zeros(M)
        src[i0:i1 + 1] = Q_E * (NBG - n) * cvw
        src[i0] += bot.interface_charge_C_m2
        src[i1] += top.interface_charge_C_m2
        return flux[1:] - flux[:-1] + src, n

    u = np.zeros(M)
    for _ in range(60):
        R, n = resid(u)
        r0 = np.max(np.abs(R))
        if r0 < 1e-13:
            break
        J = np.empty((M, M))
        for j in range(M):
            up_ = u.copy()
            up_[j] += 1e-7
            J[:, j] = (resid(up_)[0] - R) / 1e-7
        du = -np.linalg.solve(J, R)
        lam = 1.0
        while lam > 1e-4 and np.max(np.abs(resid(u + lam * du)[0])) >= r0 * (1.0 - 1e-4 * lam):
            lam *= 0.5
        u = u + lam * du
    R, n = resid(u)
    assert np.max(np.abs(R)) < 1e-12
    return u[i0:i1 + 1], n, EF


@pytest.mark.parametrize("alpha", [0.0, 0.4])
def test_brute_force_meshed_oxide_oracle(alpha):
    """Same film grid (97 nodes over 6 nm, h = 0.0625 nm as before), asymmetric stacks with work-function
    offsets and interface charges, two asymmetric bias pairs. The oracle meshes the oxides and solves one
    global system, so it checks the face-law derivation (capacitance, signs of V_wf and sigma, the
    half-cell flux) and the outer Newton end to end. Same discretization => agreement to the solver
    tolerances: budget 1e-8 V on the potential and 1e-8 n_bg on the density (error bound |R| / (dR/dphi)
    plus the oracle's own 1e-12).
    The outer stop is tightened here to |R| < 1e-11 C/m^2: at the default 1e-9 C/m^2 the bound
    |R| / (dR/dphi) is itself ~1e-8 V, and the generic film's (-2, 3) V point stopped at |R| = 6.3e-10
    with 8.3e-9 V / 1.1e-8 n_bg (over the density budget).
    Measured with the 1e-11 stop: alpha 0: 1.6e-13 / 2.2e-12 V, 4.6e-13 / 1.6e-12 n_bg; alpha 0.4:
    5.4e-13 / 3.6e-12 V, 1.2e-12 / 6.0e-12 n_bg."""
    bot, top = _asym_gates()
    s = _solver(bot, top, nz=97, alpha_np_per_eV=alpha, gauss_tol_C_m2=1e-11)
    for V in ((1.0, -1.5), (-2.0, 3.0)):
        phi_o, n_o, EF = _brute_force(6e-9, 97, bot, top, V[0], V[1], alpha=alpha)
        assert abs(EF / s.E_F_J - 1.0) < 1e-10
        sol = s.solve_voltages(*V)
        assert np.max(np.abs(sol.phi_V - phi_o)) < 1e-8, V
        assert np.max(np.abs(sol.n_m3 - n_o)) < 1e-8 * NBG, V


# ---------------------------------------------------------------------------------------------------
# limits
# ---------------------------------------------------------------------------------------------------

def test_single_gate_limit_far_gate_behind_thick_oxide():
    """Far (bottom) gate at 0 V behind a 10 um vacuum-like oxide (C_b = 8.9e-7 F/m^2), or no gate at all,
    on a 20 nm film: the top face must reproduce the v0.16 single-gate solver (face-field map,
    Neumann body, same top oxide) -- the far face is then charge-neutral and the bulk is flat in the
    middle in both. Measured (20 nm, h = 0.025 nm): top-face potential within 9.6e-6 / 6.3e-6 V of
    the single-gate surface potential at +2 / -2 V (budget 5e-5 V), density over the top 5 nm within
    1.09e-4 / 1.17e-4 n_bg (budget 3e-4). Convergence with film thickness (same comparison): 10 nm
    5.3e-4 V / 6.0e-3 n_bg, 20 nm 9.6e-6 V / 1.09e-4 n_bg, 40 nm 1.0e-7 V / 1.2e-5 n_bg at +2 V: the
    remaining difference is the far wall's influence on the sub-band ladder, not the face law."""
    from dynameta.carriers.sp_carrier import SchrodingerPoissonCarrier
    from dynameta.sweep import BiasPoint
    t, nz = 20e-9, 801
    c = SchrodingerPoissonCarrier(semi_thk_m=t, n_bg_m3=NBG, m_eff_kg=MSTAR, eps_static=EPS_S, nz=nz,
                                  n_lateral=2, oxide_thk_m=8e-9, eps_oxide=20.0, neumann_body=True,
                                  n_states=200)
    thick = _solver(GateStack((1e-5, 1.0)), GateStack((8e-9, 20.0)), t_film_m=t, nz=nz)
    open_ = _solver(None, GateStack((8e-9, 20.0)), t_film_m=t, nz=nz)
    top5 = slice(nz - 201, None)
    for vg in (2.0, -2.0):
        cf = c.solve(BiasPoint({"gate": vg, "body": 0.0}, "g"))
        psi = cf.extras["surface_potential_V"]
        n1 = cf.regions["semi"].grid_fields["electron_density_m3"][0, 0]
        a = thick.solve_voltages(0.0, vg)
        b = open_.solve_voltages(0.0, vg)
        assert abs(a.phi_top_V - b.phi_top_V) < 1e-6, vg                # C_b -> 0 limit reached
        assert abs(a.phi_top_V - psi) < 5e-5, (vg, a.phi_top_V, psi)
        assert np.max(np.abs(a.n_m3[top5] - n1[top5])) < 3e-4 * NBG, vg


def _classical_tools(m):
    """F_1/2 tabulated by quadrature (spline) and a classical Fermi-Dirac Poisson Newton on a uniform
    grid with Dirichlet ends: eps phi'' = -q (N_d - N_c F_1/2((E_F + q phi)/kT))."""
    from scipy.interpolate import CubicSpline
    from scipy.linalg import solve_banded
    kT = KB * T
    Nc = 2.0 * (m * kT / (2.0 * np.pi * HBAR ** 2)) ** 1.5
    eta = np.linspace(-60.0, 80.0, 2801)
    spl = CubicSpline(eta, [_f_fd(0.5, x) for x in eta])
    dspl = spl.derivative()

    def n_cl(phi, EF):
        return Nc * spl((EF + Q_E * phi) / kT)

    def bvp(z, phiL, phiR, nbg, EF):
        ee, hh = EPS_S * EPS0, z[1] - z[0]
        p = np.linspace(phiL, phiR, z.size)
        for _ in range(100):
            x = (EF + Q_E * p[1:-1]) / kT
            R = ee * (p[:-2] - 2.0 * p[1:-1] + p[2:]) / hh ** 2 + Q_E * (nbg - Nc * spl(x))
            ab = np.zeros((3, z.size - 2))
            ab[0, 1:] = ab[2, :-1] = ee / hh ** 2
            ab[1] = -2.0 * ee / hh ** 2 - Q_E * Nc * dspl(x) * Q_E / kT
            dp = solve_banded((1, 1), ab, -R)
            p[1:-1] += dp
            if np.max(np.abs(dp)) < 1e-13:
                break
        return p
    return n_cl, bvp


def test_wide_film_classical_limit_away_from_interfaces():
    """Wide-film limit (30 nm, generic film, gates at +3 / -3 V): away from the walls, beyond the decay of
    the wall-induced density oscillations (thermal length hbar v_F / (2 pi kT) ~ 3 nm), the quantum
    density is the classical Fermi-Dirac density of the local potential, and the potential solves the
    classical Poisson equation there. Oracles written here: n_cl = N_c F_1/2((E_F + q phi)/kT) (F_1/2
    by quadrature) and a classical Poisson Newton on [10 nm, t - 10 nm] with Dirichlet ends from the
    quantum profile. Measured: max|n - n_cl(phi)| = 5.07e-3 n_bg beyond 5 nm, 6.91e-4 beyond 10 nm
    (budget 1e-3, and the decay itself is pinned); classical-Poisson potential within 3.1e-5 V of the
    quantum one on [10, 20] nm (budget 1e-4 V)."""
    t, nz = 30e-9, 1201
    s = _solver(GateStack((8e-9, 20.0)), GateStack((8e-9, 20.0)), t_film_m=t, nz=nz)
    sol = s.solve_voltages(3.0, -3.0)
    n_cl, bvp = _classical_tools(MSTAR)
    z, phi, n = sol.z_m, sol.phi_V, sol.n_m3
    err = {}
    for d in (5e-9, 10e-9):
        w = (z > d) & (z < t - d)
        err[d] = np.max(np.abs(n[w] - n_cl(phi[w], s.E_F_J))) / NBG
    assert err[10e-9] < 1e-3 and err[10e-9] < 0.3 * err[5e-9], err
    i0, i1 = 400, 800                                                 # 10 nm .. 20 nm
    p = bvp(z[i0:i1 + 1], phi[i0], phi[i1], NBG, s.E_F_J)
    assert np.max(np.abs(p - phi[i0:i1 + 1])) < 1e-4


@pytest.mark.slow
def test_semiclassical_limit_quantum_correction_scales_as_hbar2_over_m():
    """Where the potential varies over the film (60 nm, 1e25 m^-3, bottom +2 V / top -3 V: a ~15 nm
    depletion layer at the top), the quantum profile differs from the classical Fermi-Dirac Poisson one
    by the leading quantum (gradient) correction, ~ hbar^2 / (m L^2). Increasing m by 4 must shrink
    the deviation of the classical-Poisson potential (on [10 nm, t - 10 nm], Dirichlet ends from the
    quantum profile) by ~4. Measured m = 0.30 / 1.2 / 4.8 m_e: 1.66e-3 / 5.78e-4 / 1.43e-4 V (ratios 2.9,
    4.0); budget ratio > 2.5 per step. About 40 s."""
    t, nz = 60e-9, 1201
    devs = []
    for mm in (0.30, 1.2, 4.8):
        m = mm * M_E
        s = _solver(GateStack((8e-9, 20.0)), GateStack((8e-9, 20.0)), t_film_m=t, nz=nz, n_bg_m3=1e25,
                    m_eff_kg=m)
        sol = s.solve_voltages(2.0, -3.0)
        _, bvp = _classical_tools(m)
        i0, i1 = 200, 1000
        p = bvp(sol.z_m[i0:i1 + 1], sol.phi_V[i0], sol.phi_V[i1], 1e25, s.E_F_J)
        devs.append(np.max(np.abs(p - sol.phi_V[i0:i1 + 1])))
    assert devs[0] / devs[1] > 2.5 and devs[1] / devs[2] > 2.5, devs
    assert devs[2] < 2e-4, devs


def test_flat_band_centre_of_a_wide_film_holds_n_bg():
    """A 40 nm film between two gates at 0 V: centre n / n_bg within 1e-3 of 1 for alpha 0 and 0.4 (the
    E_F of the model's own bulk; the walls' dead layers are screened within ~1 nm). Measured n/n_bg - 1 =
    -1.4e-5 / -2.9e-6, centre potential -2.4e-5 / -1.8e-5 V."""
    for alpha in (0.0, 0.4):
        s = _solver(GateStack((8e-9, 20.0)), GateStack((8e-9, 20.0)), t_film_m=40e-9, nz=1601,
                    alpha_np_per_eV=alpha)
        sol = s.solve_voltages(0.0, 0.0)
        assert abs(sol.n_m3[800] / NBG - 1.0) < 1e-3, alpha
        assert abs(sol.phi_V[800]) < 1e-3, alpha


# ---------------------------------------------------------------------------------------------------
# symmetry, grid convergence
# ---------------------------------------------------------------------------------------------------

def test_mirror_symmetry_and_gate_exchange():
    """(a) Symmetric stacks and biases give a mirror-symmetric profile; (b) exchanging the two gates
    (stacks, offsets, interface charges and voltages) mirrors the profile, z -> t - z. Budget 1e-9 V and
    1e-9 n_bg (inner tolerance 1e-11 V, outer stop 1e-11 C/m^2). The outer stop is tightened from the
    default 1e-9 C/m^2: a face law met to 1e-9 C/m^2 leaves face potentials free by ~1e-9 / (C + C_s), and
    the generic film's (+2, +2) V point at the default stop is mirror-asymmetric by 1.2e-9 n_bg (6.5e-10 V),
    over the budget. Measured with the 1e-11 stop: 1.6e-10 (symmetric) and 1.1e-12 (exchange). Sign pin: the asymmetric profile is far from its own mirror image (> 0.1 V; measured 0.288 V),
    so the check is not vacuous."""
    g = GateStack((8e-9, 20.0))
    tight = dict(gauss_tol_C_m2=1e-11)
    s = _solver(g, g, **tight)
    for V in ((0.0, 0.0), (2.0, 2.0), (-3.0, -3.0)):
        sol = s.solve_voltages(*V)
        assert np.max(np.abs(sol.phi_V - sol.phi_V[::-1])) < 1e-9, V
        assert np.max(np.abs(sol.n_m3 - sol.n_m3[::-1])) < 1e-9 * NBG, V
    bot, top = _asym_gates()
    s1 = _solver(bot, top, **tight)
    s2 = _solver(top, bot, **tight)
    for V in ((2.0, -1.0), (-3.0, 2.0)):
        a = s1.solve_voltages(*V)
        b = s2.solve_voltages(V[1], V[0])
        assert np.max(np.abs(a.phi_V - b.phi_V[::-1])) < 1e-9, V
        assert np.max(np.abs(a.n_m3 - b.n_m3[::-1])) < 1e-9 * NBG, V
        assert abs(a.gate_charge_bottom_C_m2 - b.gate_charge_top_C_m2) < 1e-10
        assert np.max(np.abs(a.phi_V - a.phi_V[::-1])) > 0.1


def test_grid_convergence_second_order():
    """Face potentials and film charge at an asymmetric bias on nz 241 / 481 / 961 / 1921 (h = 0.025 /
    0.0125 / 0.00625 / 0.003125 nm) converge at second order (successive differences shrink by ~4).
    Measured at (2, -1) V: phi_b 0.33770969 / 0.33773925 / 0.33774664 / 0.33774849 V, phi_t 0.04927634 /
    0.04929096 / 0.04929462 / 0.04929553 V, film charge 1.250902826e-2 / 1.250955348e-2 / 1.250968480e-2 /
    1.250971763e-2 C/m^2: difference ratios 4.00 / 4.00."""
    bot, top = _asym_gates()
    vals = []
    for nz in (241, 481, 961, 1921):
        sol = _solver(bot, top, nz=nz).solve_voltages(2.0, -1.0)
        vals.append((sol.phi_bottom_V, sol.phi_top_V, sol.film_charge_C_m2))
    vals = np.array(vals)
    d = np.abs(np.diff(vals, axis=0))
    for k in range(3):
        assert d[0, k] / d[1, k] > 3.0 and d[1, k] / d[2, k] > 3.0, (k, d[:, k])
    assert np.max(d[2, :2]) < 2e-5


# ---------------------------------------------------------------------------------------------------
# carrier-solver interface, caching, robustness
# ---------------------------------------------------------------------------------------------------

def test_carrier_field_on_the_native_grid_feeds_the_bridge():
    """solve(bias) emits CarrierField(ndim=3) like SchrodingerPoissonCarrier: native uniform z grid,
    laterally uniform, the bias's voltages, extras with the face potentials and charges; the bridge's
    native-3D branch turns it into an eps field whose z profile is the material's eps of n(z)."""
    from dynameta.core import NM, MaterialEpsMap, assemble_eps
    from dynameta.core.alignment import GeometryAlignment, RegionAlignment
    from dynameta.core.carrier_field import ELECTRON_DENSITY, POTENTIAL
    from dynameta.core.lift import IdentityLift
    from dynameta.materials import DrudeOptical, Material, MaterialRegistry
    from dynameta.sweep import BiasPoint
    s = _solver(GateStack((8e-9, 20.0)), GateStack((8e-9, 20.0)), n_lateral=3, semi_material="TCO")
    (rinfo,) = s.regions()
    assert rinfo.ndim == 3 and rinfo.bbox_m[4:] == (0.0, 6e-9)
    cf = s.solve(BiasPoint({"bottom": -1.0, "top": 2.0, "body": 0.0}, "b"))
    assert cf.ndim == 3 and cf.voltages["top"] == 2.0
    reg = cf.regions["semi"]
    z = reg.grid_axes_m["z"]
    assert np.array_equal(z, np.linspace(0.0, 6e-9, 481))
    n3 = reg.grid_fields[ELECTRON_DENSITY]
    assert n3.shape == (3, 3, 481) and np.all(n3 == n3[0, 0])
    sol = s.solve_voltages(-1.0, 2.0)
    assert np.array_equal(n3[0, 0], sol.n_m3)
    assert np.array_equal(reg.grid_fields[POTENTIAL][1, 2], sol.phi_V)
    assert cf.extras["phi_top_V"] == sol.phi_top_V and cf.extras["dual_gate"] is True
    reg_m = MaterialRegistry()
    reg_m.add(Material("TCO", DrudeOptical(eps_inf=4.0, m_opt_kg=0.30 * M_E, gamma_rad_s=1.2e14)))
    align = GeometryAlignment(unit_scale=NM, fixed_eps_regions={}, region_alignments=[
        RegionAlignment("semi", "semi", (0.0, 12e-9, 0.0, 12e-9, 0.0, 6e-9), "z")])
    nmap = MaterialEpsMap(reg_m)
    eps = assemble_eps(cf, align, nmap, IdentityLift(), 1200e-9, mesh_regions=["semi"])["semi"]
    prof = eps.values_zyx[:, 0, 0]
    assert prof.shape == (481,)
    ref = nmap.eps_grid("TCO", {"n": sol.n_m3[None, None, :]}, 1200e-9)[0, 0]
    assert np.allclose(prof, ref, rtol=1e-12, atol=0.0)
    with pytest.raises(ValueError, match="no voltage for gate 'bottom'"):
        s.solve(BiasPoint({"gate": 1.0}, "bad"))


def test_bias_sweep_reuses_the_cache_and_matches_cold_solves():
    """A sweep is warm-started from the nearest cached bias (few inner solves per point); repeating a
    bias returns the cached solution bit for bit; any swept point equals a cold solve of a fresh solver
    to the tolerances (budget 1e-9 V, 1e-9 n_bg). The outer stop is tightened to 1e-11 C/m^2 for this
    comparison: at the default 1e-9 C/m^2 the warm and cold paths stop at different points of the
    tolerance band, up to 8.5e-9 V / 1.1e-8 n_bg apart over this sweep on the generic film (2.2e-9 V /
    3.8e-9 n_bg at the compared point), which the 1e-9 budget does not cover. Measured with the 1e-11 stop:
    worst over all nine points 9.0e-11 V (top), 1.8e-11 V (bottom), 1.7e-10 n_bg. Inner solves per point:
    13 for the first (cold anchor + finite-difference Jacobian), then 4 (about 0.04 s per bias point)."""
    bot, top = _asym_gates()
    s = _solver(bot, top, gauss_tol_C_m2=1e-11)
    sweep = [(0.0, v) for v in np.linspace(-2.0, 2.0, 9)]
    sols = [s.solve_voltages(*V) for V in sweep]
    assert all(max(abs(x.gauss_residual_bottom_C_m2), abs(x.gauss_residual_top_C_m2)) < 1e-9 for x in sols)
    assert max(x.inner_solves for x in sols[1:]) <= 12
    again = s.solve_voltages(*sweep[3])
    assert again.from_cache and np.array_equal(again.n_m3, sols[3].n_m3)
    again.n_m3[:] = 0.0                                               # a copy: the cache is untouched
    assert np.array_equal(s.solve_voltages(*sweep[3]).n_m3, sols[3].n_m3)
    cold = _solver(bot, top, gauss_tol_C_m2=1e-11).solve_voltages(*sweep[6])
    assert abs(cold.phi_top_V - sols[6].phi_top_V) < 1e-9
    assert abs(cold.phi_bottom_V - sols[6].phi_bottom_V) < 1e-9
    assert np.max(np.abs(cold.n_m3 - sols[6].n_m3)) < 1e-9 * NBG


def test_strong_accumulation_and_strong_depletion_converge():
    """Thin high-k top oxide (2 nm / 25): +8 V accumulates a peak > 5 n_bg (measured 12.0 n_bg), -10 V
    depletes the top of the film; reached by cold jumps (continuation from the anchor) with Gauss's law
    met. A 3 nm film between -12 V gates is fully depleted (sheet excess = -n_bg t to 1e-6)."""
    s = _solver(GateStack((10e-9, 3.9)), GateStack((2e-9, 25.0)))
    acc = s.solve_voltages(0.0, 8.0)
    assert acc.n_m3.max() > 5.0 * NBG
    dep = s.solve_voltages(0.0, -10.0)
    assert dep.n_m3[-81:].max() < 0.2 * NBG
    for sol in (acc, dep):
        Rb, Rt, _, _ = _face_laws(s, sol)
        assert max(abs(Rb), abs(Rt)) < 1e-9
    s3 = _solver(GateStack((5e-9, 20.0)), GateStack((5e-9, 20.0)), t_film_m=3e-9, nz=241, n_bg_m3=1e26)
    full = s3.solve_voltages(-12.0, -12.0)
    assert abs(full.sheet_excess_m2 / (-1e26 * 3e-9) - 1.0) < 1e-6


def test_failure_raises_a_clear_error_never_a_silent_answer():
    """A solve that cannot meet Gauss's law within its budgets raises DualGateConvergenceError (here:
    one Newton step and no continuation halving for a cold jump to strong accumulation), and a
    user-fixed n_states that truncates the occupied ladder surfaces the inner solver's
    UnreliableResultWarning."""
    s = _solver(GateStack((10e-9, 3.9)), GateStack((2e-9, 25.0)), max_newton=1, max_halvings=0)
    with pytest.raises(DualGateConvergenceError, match="anchor solve"):
        s.solve_voltages(0.0, 8.0)                     # the cold anchor alone needs more than one step
    s = _solver(GateStack((10e-9, 3.9)), GateStack((2e-9, 25.0)))
    s.solve_voltages(0.0, 0.0)
    s.max_newton, s.max_halvings = 1, 0                # numerical budgets only (not physics)
    with pytest.raises(DualGateConvergenceError, match="no convergence toward"):
        s.solve_voltages(0.0, 8.0)
    assert (0.0, 8.0) not in s._cache
    from dynameta.warning_types import UnreliableResultWarning
    s2 = _solver(GateStack((8e-9, 20.0)), GateStack((8e-9, 20.0)), n_states=2)
    with pytest.warns(UnreliableResultWarning, match="n_states"):
        s2.solve_voltages(0.0, 0.0)


def test_work_function_offset_is_a_gate_voltage_shift_and_interface_charge_sign():
    """Consistency checks: V_wf enters only as V - V_wf (identical solutions to 1e-9 V); a positive
    interface charge at a face attracts electrons there (sign pin)."""
    g = GateStack((8e-9, 20.0))
    a = _solver(g, GateStack((8e-9, 20.0), work_function_offset_V=0.3)).solve_voltages(0.0, 1.3)
    b = _solver(g, g).solve_voltages(0.0, 1.0)
    assert abs(a.phi_top_V - b.phi_top_V) < 1e-9 and abs(a.phi_bottom_V - b.phi_bottom_V) < 1e-9
    c = _solver(g, GateStack((8e-9, 20.0), interface_charge_C_m2=1e-2)).solve_voltages(0.0, 1.0)
    assert c.n_m3[-81:].sum() > b.n_m3[-81:].sum() and c.sheet_excess_m2 > b.sheet_excess_m2


def test_input_validation():
    with pytest.raises(ValueError):
        GateStack(((0.0, 3.9),))
    with pytest.raises(ValueError):
        GateStack(((1e-9, -1.0),))
    with pytest.raises(ValueError):
        GateStack(())
    assert GateStack((2e-9, 4.0)).layers == ((2e-9, 4.0),)
    assert abs(GateStack(((1e-9, 2.0), (3e-9, 6.0))).capacitance_F_m2 - EPS0 / 1e-9) < 1e-18
    with pytest.raises(ValueError):
        _solver(None, None)
    with pytest.raises(ValueError):
        _solver(GateStack((8e-9, 20.0)), None, alpha_np_per_eV=-0.1)
    with pytest.raises(ValueError):
        _solver(GateStack((8e-9, 20.0)), None, bias_keys=("g", "g"))


@pytest.mark.parametrize("eps", [0.0, -9.0, float("nan"), float("inf")])
def test_eps_static_is_validated_under_its_own_name(eps):
    """eps_static must be finite and > 0, checked at construction and reported under its own name with every
    option off; a valid eps_static builds. OLD CODE: 0 / nan raised "xc_eps_r must be finite and > 0" even with xc
    off (the xc permittivity defaults to eps_static), -9 / inf the same, so the match fails."""
    with pytest.raises(ValueError, match="eps_static .*must be finite and > 0"):
        DualGateSchrodingerPoisson(t_film_m=6e-9, bottom=GateStack((8e-9, 20.0)), top=None, n_bg_m3=NBG,
                                   m_eff_kg=MSTAR, eps_static=eps, nz=101)
    assert DualGateSchrodingerPoisson(t_film_m=6e-9, bottom=GateStack((8e-9, 20.0)), top=None, n_bg_m3=NBG,
                                      m_eff_kg=MSTAR, eps_static=EPS_S, nz=101).eps_static == EPS_S


def test_one_kernel_census():
    """The dual-gate module imports the face-flux and bulk-Fermi-level kernels (one implementation per
    formula) and re-derives neither inline."""
    import pathlib
    import dynameta
    src = (pathlib.Path(dynameta.__file__).parent / "carriers" / "sp_dual_gate.py").read_text(encoding="utf-8")
    assert "sp_face_flux_C_m2(" in src and "sp_bulk_fermi_level_J(" in src
    for inline in ("** (2.0 / 3.0)", "phi[-1] - phi[-2]", "phi_V[-1] - phi_V[-2]", "_fermi_log(", "_fd1(",
                   "eigh_tridiagonal"):
        assert inline not in src, inline
    import dynameta.carriers as c
    assert c.DualGateSchrodingerPoisson is DualGateSchrodingerPoisson
    assert "DualGateSchrodingerPoisson" in c.__all__
