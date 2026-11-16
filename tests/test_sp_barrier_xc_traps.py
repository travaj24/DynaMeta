"""Gates for the v0.16 options of the dual-gate Schrodinger-Poisson solver (carriers/sp_dual_gate.py):
finite oxide barriers (wavefunction penetration, BenDaniel-Duke mass step, oxide charge in Poisson),
exchange-correlation in the ground state (LDA, shared functional with optics.quantum_film), and
interface traps (physics_traps.FilmInterfaceTraps, occupancy at the interface electrostatic potential).

v0.15.1 had none of these: its solver puts psi = 0 at both faces, has no exchange-correlation term and no
interface traps. Each gate below states what the hard-wall / no-xc / no-trap model misses by.

Oracles (independent of the solver's code path unless labelled a consistency check):
  * the analytic finite square well with a BenDaniel-Duke mass step (transcendental equation by brentq);
  * a brute-force dual-gate solve written in this file: the WHOLE stack meshed (penetration regions at
    the film spacing, the rest of the oxide coarse) with the gate metals as Dirichlet nodes -- no matching
    plane, no remaining-oxide capacitance, no outer Newton -- a dense BenDaniel-Duke Hamiltonian over the
    film and the penetration regions, numpy eigh, the LDA potential from the published closed forms
    (exchange; Perdew-Zunger 1981 v_c written out, not differentiated), trap charge by scipy quadrature
    of D f(E), and a plain Newton with a dense finite-difference Jacobian (density and potential jointly
    when xc is on);
  * the hard-wall solver in the infinite-barrier limit (rate measured);
  * the small-signal MOS relation d phi_s / d V = C_ox / (C_ox + C_s + q^2 D_it) with C_s from the
    solver's own no-trap response;
  * a classical Fermi-Dirac first-integral MOS relation written here (the classical path's trap model)
    in the semiclassical wide-film limit.

Film: the generic gated film -- 6 nm, n_bg 3e26 m^-3, m* 0.30 m_e, eps_s 9.0, 300 K, h = 0.025 nm (241
nodes); the asymmetric top oxide is 8 nm / eps 20. Every measured number below is of this film.

Run: python -m pytest tests/test_sp_barrier_xc_traps.py -q
"""
import math
import warnings

import numpy as np
import pytest

from dynameta.carriers.physics_traps import FilmInterfaceTraps
from dynameta.carriers.sp_dual_gate import (
    DualGateSchrodingerPoisson, GateStack, OxideBarrier, trap_charge_and_derivative)
from dynameta.constants import EPS0, HBAR, KB, M_E, Q_E

MSTAR = 0.30 * M_E
NBG = 3e26
EPS_S = 9.0
T = 300.0


def _f_fd(j, x):
    """Normalized Fermi-Dirac integral F_j(x) by scipy quadrature."""
    from scipy.integrate import quad
    from scipy.special import gamma
    up = max(x, 0.0) + 60.0
    f = lambda t: t ** j / (1.0 + np.exp(min(t - x, 700.0)))                           # noqa: E731
    pts = [x] if 0.0 < x < up else None
    return quad(f, 0.0, up, points=pts, epsabs=0.0, epsrel=1e-13, limit=400)[0] / gamma(j + 1)


def _solver(bot, top, **kw):
    kw.setdefault("t_film_m", 6e-9)
    kw.setdefault("nz", 241)
    return DualGateSchrodingerPoisson(bottom=bot, top=top, n_bg_m3=kw.pop("n_bg_m3", NBG),
                                      m_eff_kg=kw.pop("m_eff_kg", MSTAR), eps_static=EPS_S, T_K=T, **kw)


# ---- independent LDA (published closed forms; the library differentiates eps_xc instead) -------------
def _vxc_test(n, m, eps_r, model):
    """v_xc [J]: exchange v_x = -(3 n / pi)^(1/3) e^2 / (4 pi eps0 eps_r); Perdew-Zunger (1981) v_c,
    r_s >= 1: gamma (1 + 7/6 b1 sqrt(rs) + 4/3 b2 rs) / (1 + b1 sqrt(rs) + b2 rs)^2,
    r_s < 1: A ln rs + (B - A/3) + 2/3 C rs ln rs + (2 D - C)/3 rs (Hartree*)."""
    n = np.asarray(n, dtype=float)
    out = np.zeros_like(n)
    pos = n > 0.0
    e2 = Q_E ** 2 / (4.0 * np.pi * EPS0 * eps_r)
    out[pos] = -(3.0 * n[pos] / np.pi) ** (1.0 / 3.0) * e2
    if model == "lda":
        a_star = 4.0 * np.pi * EPS0 * eps_r * HBAR ** 2 / (m * Q_E ** 2)
        ha = e2 / a_star
        rs = (3.0 / (4.0 * np.pi * n[pos])) ** (1.0 / 3.0) / a_star
        g, b1, b2 = -0.1423, 1.0529, 0.3334
        A, B, C, D = 0.0311, -0.048, 0.0020, -0.0116
        hi = (g * (1.0 + 7.0 / 6.0 * b1 * np.sqrt(rs) + 4.0 / 3.0 * b2 * rs)
              / (1.0 + b1 * np.sqrt(rs) + b2 * rs) ** 2)
        lo = A * np.log(rs) + (B - A / 3.0) + 2.0 / 3.0 * C * rs * np.log(rs) + (2.0 * D - C) / 3.0 * rs
        out[pos] += np.where(rs >= 1.0, hi, lo) * ha
    return out


def _trap_quad(spec, ef_minus_ec_eV):
    """Trap sheet charge [C/m^2] by scipy quadrature of D f(E) (donor +q (1 - f), acceptor -q f) and the
    discrete levels, E_F - E_c given in eV."""
    from scipy.integrate import quad
    kT = KB * T / Q_E
    f = lambda E: 0.5 * (1.0 - math.tanh(0.5 * (E - ef_minus_ec_eV) / kT))              # noqa: E731
    q = 0.0
    for a, b, kind in spec.bands():
        pts = [ef_minus_ec_eV] if a < ef_minus_ec_eV < b else None
        tol = dict(points=pts, epsabs=1e-15 * (b - a), epsrel=1e-12, limit=500)
        if kind == "acceptor":
            q -= spec.dit_eV_m2 * quad(f, a, b, **tol)[0]
        else:
            g = lambda E: 0.5 * (1.0 + math.tanh(0.5 * (E - ef_minus_ec_eV) / kT))         # noqa: E731
            q += spec.dit_eV_m2 * quad(g, a, b, **tol)[0]
    for e, nn, kind in spec.level_kinds():
        q += -nn * f(e) if kind == "acceptor" else nn * (1.0 - f(e))
    return Q_E * q


# ---------------------------------------------------------------------------------------------------
# options off: the existing code path, bit for bit
# ---------------------------------------------------------------------------------------------------

def test_options_off_is_the_existing_code_path_bit_for_bit():
    """Same-tree A/B: no barrier, a barrier with penetration_m = 0, an empty trap spec (D_it = 0, no
    levels) and xc = None / 'none' all take the hard-wall code path and give np.array_equal results
    (potential, density, face potentials, iteration counts) to a solver built without the options. (The
    same outputs were also checked bit-identical to the code before the options were added, by sha256
    digests.)"""
    def mk(b=None, tr=None, **kw):
        bot = GateStack(((4e-9, 3.9), (6e-9, 22.0)), work_function_offset_V=0.3, interface_charge_C_m2=-5e-3,
                        barrier=b, traps=tr)
        top = GateStack((8e-9, 20.0), work_function_offset_V=-0.2, interface_charge_C_m2=2e-3, barrier=b,
                        traps=tr)
        return _solver(bot, top, **kw)
    ref = mk()
    variants = [mk(OxideBarrier(2.5, 0.3 * M_E, penetration_m=0.0)), mk(tr=FilmInterfaceTraps()),
                mk(tr=FilmInterfaceTraps(levels=((0.1, 0.0),), charge="acceptor")), mk(xc="none"),
                mk(OxideBarrier(2.5, 0.3 * M_E, penetration_m=0.0), FilmInterfaceTraps(), xc=None)]
    for V in ((0.0, 0.0), (2.0, -1.0)):
        a = ref.solve_voltages(*V)
        for s in variants:
            assert not s._general
            b = s.solve_voltages(*V)
            assert np.array_equal(a.phi_V, b.phi_V) and np.array_equal(a.n_m3, b.n_m3), V
            assert (a.phi_bottom_V, a.phi_top_V, a.inner_solves) == (b.phi_bottom_V, b.phi_top_V, b.inner_solves)
            assert b.z_ext_m is None and b.U_ext_J is None


# ---------------------------------------------------------------------------------------------------
# Part 1: finite barriers
# ---------------------------------------------------------------------------------------------------

def _finite_well_levels(L, V0, mw, mb, n_levels):
    """Bound states of the symmetric finite well of width L (mass mw) between barriers V0 (mass mb) with
    the BenDaniel-Duke condition (1/m) psi' continuous: even k tan(k L/2) / mw = kappa / mb, odd
    -k cot(k L/2) / mw = kappa / mb."""
    from scipy.optimize import brentq
    out = []

    def g(E, parity):
        k = math.sqrt(2.0 * mw * E) / HBAR
        kap = math.sqrt(2.0 * mb * (V0 - E)) / HBAR
        x = 0.5 * k * L
        if parity == 0:
            return k * math.sin(x) / mw - kap * math.cos(x) / mb
        return -k * math.cos(x) / mw - kap * math.sin(x) / mb
    Es = np.linspace(1e-6 * V0, V0 * (1 - 1e-9), 20001)
    for par in (0, 1):
        vals = [g(E, par) for E in Es]
        for i in range(len(Es) - 1):
            if vals[i] * vals[i + 1] < 0.0:
                out.append(brentq(g, Es[i], Es[i + 1], args=(par,), xtol=1e-40, rtol=1e-15))
    return sorted(out)[:n_levels]


def test_bendaniel_duke_interface_operator_second_order_and_hermitian():
    """The node-centred finite-volume BenDaniel-Duke operator (cell masses, interface-node potential the
    average of the two sides) against the analytic finite well with a mass step (well 5 nm, m_w 0.30,
    barrier 1.0 eV, m_b 0.2 / 0.8 m_e, 6 nm of barrier each side): the lowest three levels converge at
    second order (error ratio per grid halving 3.977-4.011, measured), error < 2e-5 eV at h = 0.0125 nm
    (measured <= 1.4e-5 eV).
    The matrix assembled from the flux form row by row is symmetric to the last bit, and the solver's
    eigenvectors satisfy it to 1e-10 relative. Failing on v0.15.1: the node-mass form there
    (harmonic mean of the two node masses at each face) cannot represent a step AT a node; given the
    interface node the well mass it converges at FIRST order (ratio 1.82-1.97, error 5.6e-5 to 9.6e-4 eV
    at h = 0.0125 nm, measured)."""
    from dynameta.carriers.schrodinger_poisson import SchrodingerPoisson1D
    L, V0 = 5e-9, 1.0 * Q_E
    for mb in (0.2 * M_E, 0.8 * M_E):
        exact = np.array(_finite_well_levels(L, V0, MSTAR, mb, 3))
        errs, errs_node = [], []
        for nzf in (101, 201, 401):
            h = L / (nzf - 1)
            nb = int(round(6e-9 / h))
            z = h * np.arange(-nb, nzf + nb)
            N = z.size
            inv_m = np.full(N - 1, 1.0 / MSTAR)
            inv_m[:nb] = inv_m[nb + nzf - 1:] = 1.0 / mb
            U = np.zeros(N)
            U[:nb] = U[nb + nzf:] = V0
            U[nb] = U[nb + nzf - 1] = 0.5 * V0
            sp = SchrodingerPoisson1D(z, MSTAR)
            E, psi, _ = sp.solve_schrodinger(U, inv_mass_cell_per_kg=inv_m, n_states=3)
            errs.append(np.abs(E - exact))
            # flux-form assembly, row by row (independent of solve_schrodinger's vectorized bands)
            c = HBAR ** 2 / (2.0 * h * h)
            A = np.zeros((N - 2, N - 2))
            for r in range(N - 2):
                j = r + 1
                A[r, r] = c * (inv_m[j - 1] + inv_m[j]) + U[j]
                if r > 0:
                    A[r, r - 1] = -c * inv_m[j - 1]
                if r < N - 3:
                    A[r, r + 1] = -c * inv_m[j]
            assert np.array_equal(A, A.T)
            assert np.max(np.abs(A @ psi - psi * E)) < 1e-10 * np.max(np.abs(E)) * np.max(np.abs(psi))
            # v0.15.1-style node masses (interface node given the well mass)
            mn = np.full(N, MSTAR)
            mn[:nb] = mn[nb + nzf:] = mb
            En, _, _ = sp.solve_schrodinger(U, m_eff_z_kg=mn, n_states=3)
            errs_node.append(np.abs(En - exact))
        errs, errs_node = np.array(errs), np.array(errs_node)
        r = errs[:-1] / errs[1:]
        assert np.all((r > 3.7) & (r < 4.3)), (mb, r)
        assert np.max(errs[-1]) < 2e-5 * Q_E, errs[-1] / Q_E
        rn = errs_node[:-1, 0] / errs_node[1:, 0]
        assert np.all(rn < 2.6), rn                                    # first order: the old form


def _stacks_barrier(b_bot=None, b_top=None, tr=None, sig=(-5e-3, 2e-3)):
    """Asymmetric stacks: bottom 1 nm low-k (3.9) + 6 nm high-k (22), top 8 nm (20)."""
    bot = GateStack(((1e-9, 3.9), (6e-9, 22.0)), work_function_offset_V=0.3, interface_charge_C_m2=sig[0],
                    barrier=b_bot, traps=tr)
    top = GateStack(((8e-9, 20.0),), work_function_offset_V=-0.2, interface_charge_C_m2=sig[1], barrier=b_top,
                    traps=tr)
    return bot, top


B_BOT = OxideBarrier((3.0, 1.6), (0.4 * M_E, 0.2 * M_E), penetration_m=2e-9)   # spans both bottom layers
B_TOP = OxideBarrier(2.5, 0.3 * M_E, penetration_m=2e-9)
TRAPS = FilmInterfaceTraps(dit_eV_m2=2e17, e_lo_eV=-0.6, e_hi_eV=1.2, neutral_level_eV=0.7,
                           levels=((0.9, 3e16),))


def _brute_force(t, nz, bot, top, Vb, Vt, *, xc=None, n_coarse=8):
    """Whole-stack brute force (module docstring). Penetration regions at the film spacing over each
    barrier's penetration_m; the rest of each stack coarse; gates Dirichlet."""
    from scipy.linalg import eigh
    from scipy.optimize import brentq
    kT = KB * T
    Nc = 2.0 * (MSTAR * kT / (2.0 * np.pi * HBAR ** 2)) ** 1.5
    EF = brentq(lambda E: Nc * _f_fd(0.5, E / kT) - NBG, -1.0 * Q_E, 3.0 * Q_E, xtol=1e-30)
    h = t / (nz - 1)

    def side(g, sign):
        """Mesh beyond one face, outward: list of (dz, eps, mass, offset_J, in_schrodinger)."""
        cells = []
        pen = 0.0 if g.barrier is None else g.barrier.penetration_m
        npen = int(round(pen / h))
        ht = g.barrier.height_eV if g.barrier is not None else None
        ms = g.barrier.mass_kg if g.barrier is not None else None
        depth = 0.0
        for i, (tl, el) in enumerate(g.layers):
            V = (ht[i] if isinstance(ht, tuple) else ht) if ht is not None else 0.0
            m = (ms[i] if isinstance(ms, tuple) else ms) if ms is not None else 1.0
            in_pen = max(0.0, min(tl, npen * h - depth))
            k = int(round(in_pen / h))
            cells += [(h, el, m, V * Q_E, True)] * k
            rest = tl - k * h
            if rest > 1e-15:
                cells += [(rest / n_coarse, el, m, V * Q_E, False)] * n_coarse
            depth += tl
        return cells
    cb, ct = side(bot, -1), side(top, +1)
    cells = cb[::-1] + [(h, EPS_S, MSTAR, 0.0, True)] * (nz - 1) + ct
    dz = np.array([c[0] for c in cells])
    eps_c = np.array([c[1] for c in cells]) * EPS0
    m_c = np.array([c[2] for c in cells])
    U_c = np.array([c[3] for c in cells])
    insch = np.array([c[4] for c in cells])
    Nn = dz.size + 1
    ib, it_ = len(cb), len(cb) + nz - 1                        # interface nodes
    # Schrodinger domain: the node range covered by consecutive in-Schrodinger cells around the film
    s0 = ib - int(np.sum(insch[:ib]))
    s1 = it_ + int(np.sum(insch[it_:]))
    cvw = np.zeros(Nn)                                         # control-volume widths
    cvw[:-1] += 0.5 * dz
    cvw[1:] += 0.5 * dz
    Nd = np.zeros(Nn)                                          # film doping over each control volume
    Nd[ib + 1:it_] = NBG
    Nd[ib] = 0.5 * NBG * h / cvw[ib]
    Nd[it_] = 0.5 * NBG * h / cvw[it_]
    sch_nodes = np.arange(s0 + 1, s1)                          # interior Schrodinger nodes
    ns_ = sch_nodes.size
    c = HBAR ** 2 / (2.0 * h * h)
    im = 1.0 / m_c[s0:s1]                                      # cells between s0..s1
    Unode = np.zeros(Nn)
    for i in range(s0 + 1, s1):
        Unode[i] = 0.5 * (U_c[i - 1] + U_c[i])
    pref = 2.0 * MSTAR / (2.0 * np.pi * HBAR ** 2)
    pg = (Vb - bot.work_function_offset_V, Vt - top.work_function_offset_V)
    vref = _vxc_test(np.array([NBG]), MSTAR, EPS_S, xc)[0] if xc else 0.0

    def fill(phi, n_sch):
        Hd = c * (im[:-1] + im[1:]) + Unode[sch_nodes] - Q_E * phi[sch_nodes]
        if xc:
            Hd = Hd + _vxc_test(n_sch, MSTAR, EPS_S, xc) - vref
        H = np.diag(Hd) + np.diag(-c * im[1:-1], 1) + np.diag(-c * im[1:-1], -1)
        E, Vv = eigh(H)
        Vv = Vv / np.sqrt(np.sum(Vv ** 2, axis=0) * h)
        ns = pref * kT * np.logaddexp(0.0, (EF - E) / kT)
        return (Vv ** 2) @ ns

    def trapq(g, phi_s):
        return 0.0 if g.traps is None else _trap_quad(g.traps, (EF + Q_E * phi_s) / Q_E)

    nphi = Nn - 2

    def resid(u):
        phi = np.concatenate([[pg[0]], u[:nphi], [pg[1]]])
        n_sch = u[nphi:] if xc else None
        n_new = fill(phi, n_sch if xc else None)
        n = np.zeros(Nn)
        n[sch_nodes] = n_sch if xc else n_new
        flux = eps_c * np.diff(phi) / dz
        src = Q_E * (Nd - n) * cvw
        src[ib] += bot.interface_charge_C_m2 + trapq(bot, phi[ib])
        src[it_] += top.interface_charge_C_m2 + trapq(top, phi[it_])
        R = (flux[1:] - flux[:-1] + src[1:-1]) / (Q_E * NBG * h)
        if xc:
            R = np.concatenate([R, (n_sch - n_new) / NBG])
        return R, phi, n

    u = np.zeros(nphi + (ns_ if xc else 0))
    if xc:
        u[nphi:] = NBG * ((sch_nodes > ib) & (sch_nodes < it_))
    for _ in range(60):
        R, phi, n = resid(u)
        r0 = np.max(np.abs(R))
        if r0 < 1e-13:
            break
        J = np.empty((R.size, u.size))
        for j in range(u.size):
            up_ = u.copy()
            d = 1e-7 if j < nphi else 1e-7 * NBG
            up_[j] += d
            J[:, j] = (resid(up_)[0] - R) / d
        du = -np.linalg.solve(J, R)
        lam = 1.0
        while lam > 1e-4 and np.max(np.abs(resid(u + lam * du)[0])) >= r0 * (1.0 - 1e-4 * lam):
            lam *= 0.5
        u = u + lam * du
    R, phi, n = resid(u)
    assert np.max(np.abs(R)) < 1e-11, np.max(np.abs(R))
    return phi[ib:it_ + 1], n[ib:it_ + 1], phi, n, (ib, it_)


@pytest.mark.parametrize("case", ["barrier", "barrier_traps", "barrier_traps_xc", "hard_xc_traps"])
def test_brute_force_whole_stack_oracle(case):
    """Asymmetric stacks with work-function offsets and interface charges; finite barriers on both faces
    (bottom penetration 2 nm spanning a 1 nm 3.0 eV / 0.4 m_e layer and 1 nm of a 1.6 eV / 0.2 m_e
    layer, top 2.5 eV / 0.3 m_e), optionally interface traps (band + level) and LDA xc; 73 film nodes.
    The oracle meshes the whole stack and solves one global system (no matching plane, no remaining-oxide
    capacitance, no outer Newton, xc from the published closed forms, traps by quadrature), so it checks
    the matching-plane derivation, the doping split at the interface node, the sheet sources inside the
    domain, the xc potential and its Newton coupling end to end. Same discretization => budget 1e-8 V and
    1e-8 n_bg (outer stop 1e-9 C/m^2, inner 1e-11 V); measured 3.7e-11 to 7.7e-10 V and 5.4e-11 to
    1.1e-9 n_bg. Failing on v0.15.1 (hard walls): the face potentials differ by 0.05-0.12 V in the
    barrier cases (measured 0.122 / 0.064 V bottom / top without traps)."""
    xc = "lda" if "xc" in case else None
    bb, bt = (None, None) if case.startswith("hard") else (B_BOT, B_TOP)
    tr = TRAPS if "traps" in case else None
    bot, top = _stacks_barrier(bb, bt, tr)
    s = _solver(bot, top, nz=73, xc=xc)
    V = (1.0, -1.5)
    phi_o, n_o, phi_all, n_all, (ib, it_) = _brute_force(6e-9, 73, bot, top, *V, xc=xc)
    sol = s.solve_voltages(*V)
    assert np.max(np.abs(sol.phi_V - phi_o)) < 1e-8, np.max(np.abs(sol.phi_V - phi_o))
    assert np.max(np.abs(sol.n_m3 - n_o)) < 1e-8 * NBG, np.max(np.abs(sol.n_m3 - n_o)) / NBG
    if bb is not None:
        # the oxide parts of the density, node for node, on the penetration regions
        k0 = ib - s._kb
        assert np.max(np.abs(sol.n_ext_m3 - n_all[k0:k0 + sol.n_ext_m3.size])) < 1e-8 * NBG
        hard = _solver(*_stacks_barrier(None, None, tr), nz=73, xc=xc).solve_voltages(*V)
        assert abs(hard.phi_bottom_V - sol.phi_bottom_V) > 0.05


def test_matching_plane_gauss_law_and_exact_charge_balance():
    """Re-derived here from the returned extended profile: Gauss's law at the two matching planes (flux of
    the end cell with that layer's permittivity, the remaining oxide's series capacitance computed here
    from the layers: bottom 5 nm of eps 22, top 6 nm of eps 20), budget 1e-9 C/m^2; the total charge
    balance Q_gate_b + Q_gate_t + sigma_b + sigma_t + Q_it,b + Q_it,t + q trapz_domain(N_d - n) = 0 with
    the trapezoid written out and N_d halved on the interface nodes, budget 2e-9 C/m^2; and the film /
    oxide split adds up (in-film + both penetration regions = domain total, exact to rounding). Biases of
    both signs, traps on, finite barriers. Measured: Gauss <= 3.1e-11, balance <= 4.5e-11 C/m^2."""
    bot, top = _stacks_barrier(B_BOT, B_TOP, TRAPS)
    s = _solver(bot, top)
    Cb = 1.0 / (5e-9 / (EPS0 * 22.0))
    Ct = 1.0 / (6e-9 / (EPS0 * 20.0))
    for V in ((0.0, 0.0), (1.5, -1.0), (-2.0, 1.0)):
        sol = s.solve_voltages(*V)
        z, phi, n = sol.z_ext_m, sol.phi_ext_V, sol.n_ext_m3
        h = z[1] - z[0]
        assert n[0] == 0.0 and n[-1] == 0.0 and sol.n_m3[0] > 0.0 and sol.n_m3[-1] > 0.0
        Fb = 22.0 * EPS0 * (phi[1] - phi[0]) / h                     # wall node: n = N_d = 0
        Ft = 20.0 * EPS0 * (phi[-1] - phi[-2]) / h
        Rb = Cb * (phi[0] - (V[0] - 0.3)) - Fb
        Rt = Ft + Ct * (phi[-1] - (V[1] + 0.2))
        assert max(abs(Rb), abs(Rt)) < 1e-9, (V, Rb, Rt)
        Nd = np.where((z > -1e-15) & (z < 6e-9 + 1e-15), NBG, 0.0)
        Nd[np.argmin(np.abs(z))] *= 0.5
        Nd[np.argmin(np.abs(z - 6e-9))] *= 0.5
        y = Nd - n
        Qdom = Q_E * h * float(np.sum(y))                            # trapz with zero end values
        Qg_b, Qg_t = Cb * (V[0] - 0.3 - phi[0]), Ct * (V[1] + 0.2 - phi[-1])
        tot = Qg_b + Qg_t - 5e-3 + 2e-3 + sol.trap_charge_bottom_C_m2 + sol.trap_charge_top_C_m2 + Qdom
        assert abs(tot) < 2e-9, (V, tot)
        assert abs(sol.neutrality_residual_C_m2) < 2e-9
        split = sol.film_charge_C_m2 + sol.oxide_electron_charge_bottom_C_m2 + sol.oxide_electron_charge_top_C_m2
        assert abs(split - Qdom) < 1e-12 * Q_E * NBG * 6e-9
        assert sol.oxide_electron_charge_bottom_C_m2 < 0.0 and sol.oxide_electron_charge_top_C_m2 < 0.0
        assert abs(sol.gate_charge_bottom_C_m2 - Qg_b) < 1e-12 and abs(sol.gate_charge_top_C_m2 - Qg_t) < 1e-12


def test_infinite_barrier_limit_converges_to_the_hard_wall_solver():
    """Barrier height -> infinity (oxide mass = film mass, 0.5 nm penetration, asymmetric stacks with
    offsets and interface charges, (1, -1) V, 241 nodes): the face potentials, gate charges and density
    converge to the hard-wall solver's. Two regimes, measured: while the tail is resolved (kappa h < 1)
    the difference falls as V_b^-1/2 (the continuum penetration length 1/kappa: 6.09e-2 / 1.85e-2 V at 10 /
    100 eV, ratio 3.30 per decade); once kappa h >> 1 the interface node itself is pushed to zero and the
    discrete operator becomes the hard-wall one, the difference falling as 1/V_b (5.40e-4 / 5.59e-5 /
    5.62e-6 V at 1e4 / 1e5 / 1e6 eV, ratio 9.65-9.96). Budgets: ratio in [2.8, 3.8] per decade below
    100 eV, [8, 12] above 1e4 eV, and 1e-5 V / 1e-4 n_bg at 1e6 eV."""
    bot, top = _stacks_barrier()
    ref = _solver(bot, top).solve_voltages(1.0, -1.0)
    d = {}
    for Vb in (10.0, 100.0, 1e4, 1e5, 1e6):
        b = OxideBarrier(Vb, MSTAR, penetration_m=0.5e-9)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=Warning)   # 10 eV: tail 4e-5 at 0.5 nm (advisory)
            sol = _solver(*_stacks_barrier(b, b)).solve_voltages(1.0, -1.0)
        d[Vb] = (max(abs(sol.phi_bottom_V - ref.phi_bottom_V), abs(sol.phi_top_V - ref.phi_top_V)),
                 np.max(np.abs(sol.n_m3 - ref.n_m3)) / NBG,
                 abs(sol.gate_charge_top_C_m2 - ref.gate_charge_top_C_m2))
    r = d[10.0][0] / d[100.0][0]
    assert 2.8 < r < 3.8, r
    for a, b in ((1e4, 1e5), (1e5, 1e6)):
        for k in range(3):
            assert 8.0 < d[a][k] / d[b][k] < 12.0, (a, k, d[a][k] / d[b][k])
    assert d[1e6][0] < 1e-5 and d[1e6][1] < 1e-4


def test_barrier_mirror_symmetry_and_grid_convergence():
    """(a) Symmetric stacks and biases with finite barriers give a mirror-symmetric extended profile
    (budget 1e-9 V / n_bg; measured 1.0e-12 V). (b) Face potentials and top gate charge on 121 / 241 / 481
    film nodes with 2 nm penetrations (a whole number of cells on every grid; asymmetric two-layer bottom
    barrier, traps off) converge at second order: measured at (1.5, -1.0) V phi_b 0.2539173 / 0.2532526 /
    0.2530852 V, phi_t -0.0301993 / -0.0304738 / -0.0305427 V, Q_gate,t -1.703262e-2 / -1.702643e-2 /
    -1.702487e-2 C/m^2, difference ratios 3.970 / 3.979 / 3.978."""
    g = GateStack((8e-9, 20.0), barrier=OxideBarrier(2.5, 0.3 * M_E, penetration_m=2e-9))
    s = _solver(g, g)
    for V in ((0.0, 0.0), (1.5, 1.5)):
        sol = s.solve_voltages(*V)
        assert np.max(np.abs(sol.phi_ext_V - sol.phi_ext_V[::-1])) < 1e-9
        assert np.max(np.abs(sol.n_ext_m3 - sol.n_ext_m3[::-1])) < 1e-9 * NBG
    bot, top = _stacks_barrier(B_BOT, B_TOP)
    vals = []
    for nz in (121, 241, 481):
        sol = _solver(bot, top, nz=nz).solve_voltages(1.5, -1.0)
        vals.append((sol.phi_bottom_V, sol.phi_top_V, sol.gate_charge_top_C_m2))
    d = np.abs(np.diff(np.array(vals), axis=0))
    for k in range(3):
        assert 3.5 < d[0, k] / d[1, k] < 4.5, (k, d[:, k])


def test_default_penetration_decays_to_round_off_and_regime_warnings():
    """Automatic penetration: int kappa dz = ln(1e16)/2 at E_F + 10 kT (2.5 eV / 0.3 m_e: 5.050 nm on the
    241-node grid; the tail at the far wall is <= 1e-16 of the peak at 0 V and +-2 V, measured 6e-21 /
    6e-17 / 1e-24); capped at the stack (1.5 eV / 0.2 m_e reaches through the 8 nm oxide, the matching
    plane is then the gate). A low barrier under a large gate field puts the oxide band below E_F inside
    the penetration region (tunnelling regime: UnreliableResultWarning), a penetration shorter than the
    tail raises an AdvisoryWarning, and a barrier below E_F + 10 kT is refused."""
    g = GateStack((8e-9, 20.0), barrier=OxideBarrier(2.5, 0.3 * M_E))
    s = _solver(g, g)
    assert abs(s.penetration_m[0] - 5.05e-9) < 1e-15
    for V in ((0.0, 0.0), (2.0, 2.0), (-2.0, -2.0)):
        assert s.solve_voltages(*V).tail_density_ratio < 1e-15
    g2 = GateStack((8e-9, 20.0), barrier=OxideBarrier(1.5, 0.2 * M_E))
    s2 = _solver(g2, g2)
    assert abs(s2.penetration_m[0] - 8e-9) < 1e-15 and s2._gate_at_plane_0
    with pytest.warns(Warning, match="tunnelling regime"):
        s2.solve_voltages(3.0, 3.0)
    g3 = GateStack((8e-9, 20.0), barrier=OxideBarrier(2.5, 0.3 * M_E, penetration_m=0.3e-9))
    with pytest.warns(Warning, match="increase OxideBarrier.penetration_m"):
        _solver(g3, g3).solve_voltages(0.0, 0.0)
    with pytest.raises(ValueError, match="no confinement"):
        _solver(GateStack((8e-9, 20.0), barrier=OxideBarrier(0.5, 0.3 * M_E)), None)
    with pytest.raises(ValueError, match="exceeds the oxide stack"):
        GateStack((8e-9, 20.0), barrier=OxideBarrier(2.5, 0.3 * M_E, penetration_m=9e-9))
    with pytest.raises(ValueError):
        GateStack(((1e-9, 3.9), (6e-9, 22.0)), barrier=OxideBarrier((2.5, 1.0, 1.0), 0.3 * M_E))


# ---------------------------------------------------------------------------------------------------
# Part 2: exchange-correlation
# ---------------------------------------------------------------------------------------------------

def test_lda_potential_closed_forms_kernel_derivative_and_one_implementation():
    """lda_xc_potential against the published closed forms written in this file (exchange; PZ81 v_c,
    which the library obtains by differentiating eps_c instead): 1e-12 relative over r_s 0.05-20; its
    density derivative (central difference) equals lda_xc_kernel to 1e-6 relative; 0 at n <= 0. One
    implementation: _eps_xc_terms is defined once in the package and sp_dual_gate imports the optics
    functions (no exchange constant or PZ81 parameter inline)."""
    import pathlib
    import re
    import dynameta
    from dynameta.optics.quantum_film import lda_xc_kernel, lda_xc_potential
    eps = EPS_S
    a_star = 4.0 * np.pi * EPS0 * eps * HBAR ** 2 / (MSTAR * Q_E ** 2)
    rs = np.geomspace(0.05, 20.0, 41)
    rs = rs[np.abs(rs - 1.0) > 1e-3]           # PZ81 switches branch at r_s = 1 (v_c' jumps there)
    n = 3.0 / (4.0 * np.pi * (rs * a_star) ** 3)
    for model in ("x", "lda"):
        v = lda_xc_potential(n, MSTAR, eps, model=model)
        ref = _vxc_test(n, MSTAR, eps, model)
        assert np.max(np.abs(v / ref - 1.0)) < 1e-12, model
        dn = 1e-5 * n
        fd = (lda_xc_potential(n + dn, MSTAR, eps, model=model)
              - lda_xc_potential(n - dn, MSTAR, eps, model=model)) / (2.0 * dn)
        assert np.max(np.abs(fd / lda_xc_kernel(n, MSTAR, eps, model=model) - 1.0)) < 1e-6, model
    assert lda_xc_potential(np.array([0.0, -1.0]), MSTAR, eps)[0] == 0.0
    root = pathlib.Path(dynameta.__file__).parent
    defs = sum(len(re.findall(r"^def _eps_xc_terms\(", p.read_text(encoding="utf-8"), flags=re.M))
               for p in root.rglob("*.py"))
    assert defs == 1
    src = (root / "carriers" / "sp_dual_gate.py").read_text(encoding="utf-8")
    assert "lda_xc_potential" in src and "lda_xc_kernel" in src
    for inline in ("0.1423", "1.0529", "0.0311", "9.0 / (4.0", "(3.0 / math.pi)"):
        assert inline not in src, inline


def test_xc_bulk_calibration_keeps_the_flat_band_density():
    """The uniform-gas shift is a constant: with V_xc(n) - V_xc(n_bg) in the Hamiltonian the flat-band bulk
    still holds n_bg at the same E_F. A 30 nm film between gates at 0 V with LDA xc: centre n / n_bg - 1
    within 2e-3 (measured +1.47e-4, without xc +1.36e-4: the O(h^2) sub-band error at h = 0.05 nm) and
    centre potential within 1e-3 V of 0 (measured -1.03e-4 V with and without xc). Without the reference
    subtraction the band would sit V_xc(n_bg) = -0.1136 eV (LDA, eps 9.0) lower, i.e. the flat-band density
    would be n_sep(E_F + 0.1136 eV) = 1.33 n_bg (computed here) -- the calibration this gate protects."""
    from dynameta.carriers.schrodinger_poisson import sp_bulk_density_m3
    from dynameta.optics.quantum_film import lda_xc_potential
    g = GateStack((8e-9, 20.0))
    s = _solver(g, g, t_film_m=30e-9, nz=601, xc="lda")
    sol = s.solve_voltages(0.0, 0.0)
    c = 300
    assert abs(sol.n_m3[c] / NBG - 1.0) < 2e-3, sol.n_m3[c] / NBG
    assert abs(sol.phi_V[c]) < 1e-3
    vref = float(lda_xc_potential(np.array([NBG]), MSTAR, EPS_S)[0])
    assert -0.13 * Q_E < vref < -0.10 * Q_E
    shifted = sp_bulk_density_m3(s.E_F_J - vref, MSTAR, T) / NBG
    assert shifted > 1.2
    assert abs(sol.xc_potential_ext_J[c]) < 1e-3 * Q_E


def test_xc_inner_newton_keeps_its_convergence():
    """The xc derivative in the inner Newton (SchrodingerPoisson1D.solve_self_consistent(xc_potential=...)), with
    the library kernel lda_xc_kernel = d lda_xc_potential / dn: face potentials +-0.6 V, tol 1e-11 V, the generic
    film. The test's xc callable records the density the inner Newton passes it at every iteration, so the
    successive changes d_k = max |n_(k+1) - n_k| / n_bg inside each outer iteration are known.
      (a) Quadratic inner tail: every step with 1e-7 < d_k < 1e-2 has d_(k+1) <= 10 d_k^2 + 1e-12 (the last term
          the round-off floor of n / n_bg). MEASURED worst d_(k+1) / d_k^2 = 0.37 (e.g. 2.2e-4 -> 9.5e-9).
      (b) Negative controls: the same solve with the kernel dropped, halved or sign-flipped (the potential still
          updated, so the same fixed point to 1e-9 V) converges only LINEARLY inside and fails (a) (MEASURED
          d_(k+1) / d_k^2 = 135 / 151 / 15 at the first violating step; 1.8 / 1.6 / 2.1x the inner iterations,
          45 / 40 / 53 against 25) -- the gate catches a wrong derivative (a kernel at 0.8x or 2x of the true
          one fails it as well, MEASURED 177 / 24).
      (c) Total inner iterations: dropping the derivative needs > 1.5x as many (45 vs 25 MEASURED).
    The outer iteration count (10 without xc, 13 with, 13 without the derivative) is a property of the film's
    nonlinearity and the xc density fill, not of the inner Newton, and is not gated."""
    from dynameta.carriers.schrodinger_poisson import SchrodingerPoisson1D, sp_bulk_fermi_level_J
    from dynameta.optics.quantum_film import lda_xc_kernel, lda_xc_potential
    z = np.linspace(0.0, 6e-9, 241)
    sp = SchrodingerPoisson1D(z, MSTAR)
    EF = sp_bulk_fermi_level_J(NBG, MSTAR, T)
    vref = float(lda_xc_potential(np.array([NBG]), MSTAR, EPS_S)[0])
    log = {"seq": []}
    dens = sp.density

    def counting_density(*a, **k):                     # one call per outer iteration (+ the final fill)
        log["seq"].append([])
        return dens(*a, **k)
    sp.density = counting_density

    def run(factor):
        log["seq"] = []
        xc = None
        if factor is not None:
            def xc(nn):
                if nn.size == z.size - 2 and log["seq"]:
                    log["seq"][-1].append(np.array(nn))
                return lda_xc_potential(nn, MSTAR, EPS_S) - vref, factor * lda_xc_kernel(nn, MSTAR, EPS_S)
        phi, n, res = sp.solve_self_consistent(eps_r=EPS_S, doping_m3=np.full(241, NBG), E_F_J=EF,
                                               phi_left_V=0.6, phi_right_V=-0.6, n_states=40, bound_tol=1e9,
                                               tol_V=1e-11, relax=1.0, max_outer=100, xc_potential=xc)
        assert res.converged
        steps = []
        for seq in log["seq"]:
            d = [float(np.max(np.abs(b - a))) / NBG for a, b in zip(seq[:-1], seq[1:])]
            steps += [(a, b) for a, b in zip(d[:-1], d[1:]) if 1e-7 < a < 1e-2]
        inner = sum(len(seq) for seq in log["seq"])
        return phi, steps, inner

    def quadratic(steps):
        return bool(steps) and all(b <= 10.0 * a * a + 1e-12 for a, b in steps)
    phi0, _, _ = run(None)
    phi_x, st1, c1 = run(1.0)
    assert np.max(np.abs(phi_x - phi0)) > 1e-3                         # xc is not a no-op
    assert quadratic(st1), st1
    for bad in (0.0, 0.5, -1.0):
        phi_b, st_b, c_b = run(bad)
        assert np.max(np.abs(phi_x - phi_b)) < 1e-9, bad              # same fixed point
        assert not quadratic(st_b), (bad, st_b)
        if bad == 0.0:
            assert c_b > 1.5 * c1, (c1, c_b)


def test_xc_on_the_reference_film_signs():
    """Sign-sensitive consistency of the xc option on the generic film (6 nm, 8 nm / eps 20 oxides, hard
    walls): the exchange-correlation hole makes the gas more compressible, so the gate charge grows in
    magnitude in BOTH directions (measured +0.71 % at +2 V, +0.56 % at -2 V, +0.32 % at 0 V); V_xc - V_xc(n_bg)
    is negative where n > n_bg and positive in the low-density wall layers (measured -0.026 / +0.071 eV at
    +2 V), and the ground sub-band moves down relative to E_F (measured -1.0 meV at +2 V). The response
    module sees the same states: QuantumFilm.from_dual_gate (U including V_xc) reproduces the solver's
    energies (1e-12 eV) and density (1e-9 n_bg)."""
    from dynameta.optics.quantum_film import QuantumFilm
    g = GateStack((8e-9, 20.0))
    s0, s1 = _solver(g, g), _solver(g, g, xc="lda")
    for V in (2.0, -2.0):
        a0, a1 = s0.solve_voltages(V, V), s1.solve_voltages(V, V)
        assert abs(a1.gate_charge_top_C_m2) > abs(a0.gate_charge_top_C_m2) * 1.003, V
    a0, a1 = s0.solve_voltages(2.0, 2.0), s1.solve_voltages(2.0, 2.0)
    assert a1.subbands.energies_J[0] < a0.subbands.energies_J[0] - 5e-4 * Q_E
    vx = a1.xc_potential_ext_J
    ipk = int(np.argmax(a1.n_m3))
    assert a1.n_m3[ipk] > NBG and vx[ipk] < 0.0 and vx[3] > 0.05 * Q_E
    film = QuantumFilm.from_dual_gate(s1, a1)
    k = a1.subbands.energies_J.size
    assert np.max(np.abs(film.energies_J[:k] - a1.subbands.energies_J)) < 1e-12 * Q_E
    assert np.max(np.abs(film.density_m3 - a1.n_ext_m3)) < 1e-9 * NBG


@pytest.mark.parametrize("g_s, g_v", [(2, 2), (1, 1), (1, 2)])
def test_xc_refuses_a_band_the_lda_functional_does_not_describe(g_s, g_v):
    """The LDA functional is that of the spin-unpolarized single-valley gas (g_s = 2, g_v = 1). For g = g_s g_v
    equally populated species exchange is (2/g)^(1/3) of it (2^(-1/3) at g_v = 2) and the correlation is a different
    function, so xc with any other degeneracy is refused with a ValueError naming g_s / g_v -- at construction and
    when xc is switched on afterwards (the options are re-read at the next solve). The same band without xc, and xc
    with (2, 1), build. OLD CODE: accepted silently (g_v = 2 with xc="lda" solved with the single-valley
    functional, no warning)."""
    g = GateStack((8e-9, 20.0))
    for xc in ("x", "lda"):
        with pytest.raises(ValueError, match=r"spin-unpolarized single-valley.*g_s = {}, g_v = {}".format(g_s, g_v)):
            _solver(g, g, xc=xc, g_s=g_s, g_v=g_v)
    s = _solver(g, g, g_s=g_s, g_v=g_v)                               # no xc: builds
    s.xc = "lda"
    with pytest.raises(ValueError, match="spin-unpolarized single-valley"):
        s.solve_voltages(0.0, 0.0)
    assert _solver(g, g, xc="lda").xc == "lda"


def test_xc_barrier_height_convention_adds_the_bulk_shift_in_the_oxide():
    """Convention (module docstring of sp_dual_gate, "Barrier height convention"): potentials are measured from the
    film's renormalized band edge, so in the oxide, where n -> 0, the Hamiltonian carries height_eV - V_xc(n_bg)/q.
    At the far wall of the penetration region (n = 0 exactly) the solver's potential energy is
    U + q phi = q height - V_xc(n_bg) to round-off (1e-12 relative; the cancellation of 2.5 eV against 0.08 eV
    costs ~3e-15), with -V_xc(n_bg) = +113.6 meV for this film (LDA, eps 9; the published exchange term alone is
    105.5 meV, written out here). This is the band-gap renormalization of the degenerate film -- correct for a
    height that is the offset WITHOUT the film's gas; a height measured relative to the degenerate film's own edge
    must be passed as H_measured + V_xc(n_bg)/q, after which the wall sits exactly H_measured above the film's
    renormalized flat-band edge (checked)."""
    from dynameta.optics.quantum_film import lda_xc_potential
    vref = float(lda_xc_potential(np.array([NBG]), MSTAR, EPS_S, model="lda")[0])
    vx = -(3.0 * NBG / np.pi) ** (1.0 / 3.0) * Q_E ** 2 / (4.0 * np.pi * EPS0 * EPS_S)
    assert abs(vx / Q_E + 0.1055) < 1e-4 and -0.12 * Q_E < vref < -0.11 * Q_E
    for H in (2.5, 2.5 + vref / Q_E):                                 # tabulated offset; converted measured one
        g = GateStack((8e-9, 20.0), barrier=OxideBarrier(H, 0.3 * M_E, penetration_m=2e-9))
        sol = _solver(g, g, xc="lda").solve_voltages(0.5, 0.5)
        for k in (0, -1):
            assert sol.n_ext_m3[k] == 0.0 and sol.xc_potential_ext_J[k] == -vref
            wall = sol.U_ext_J[k] + Q_E * sol.phi_ext_V[k]
            assert wall - Q_E * H == pytest.approx(-vref, rel=1e-12, abs=0)
            if H != 2.5:                                              # the converted measured height
                assert wall == pytest.approx(2.5 * Q_E, rel=1e-13, abs=0)   # round-off of q (V_xc, phi) sums


# ---------------------------------------------------------------------------------------------------
# Part 3: interface traps
# ---------------------------------------------------------------------------------------------------

def test_trap_charge_and_exact_derivative_against_quadrature():
    """trap_charge_and_derivative: the charge is FilmInterfaceTraps' closed form (the classical path's
    kernel, by import) -- here against scipy quadrature of D f(E) written in this file, 1e-10 relative --
    and the derivative against a central difference of that charge, 1e-7 relative, for split / acceptor
    / donor bands with discrete levels, E_F - E_c from -0.6 to +0.9 eV. Sign: dQ/dphi_s < 0 (raising
    the surface potential fills acceptor-like states)."""
    EF = 0.5437 * Q_E                     # E_F - E_c of the film
    specs = [FilmInterfaceTraps(1e17, -0.5, 1.0, 0.3, levels=((0.2, 5e16), (0.6, 2e16))),
             FilmInterfaceTraps(5e16, -0.2, 0.8, charge="acceptor", levels=((0.4, 1e16),)),
             FilmInterfaceTraps(5e16, -0.2, 0.8, charge="donor")]
    for spec in specs:
        for phi in (-1.15, -0.6, 0.0, 0.1, 0.33):
            q, dq = trap_charge_and_derivative(spec, phi, EF, T)
            ref = _trap_quad(spec, (EF + Q_E * phi) / Q_E)
            assert abs(q - ref) <= 1e-10 * max(abs(ref), 1e-3 * Q_E * spec.dit_eV_m2), (phi, q, ref)
            dd = 1e-6
            fd = (trap_charge_and_derivative(spec, phi + dd, EF, T)[0]
                  - trap_charge_and_derivative(spec, phi - dd, EF, T)[0]) / (2 * dd)
            assert abs(dq - fd) < 1e-7 * abs(fd) + 1e-10, (phi, dq, fd)      # FD rounding ~5e-13
            assert dq <= 0.0


def test_one_numpy_trap_kernel_shared_by_the_dual_gate_and_classical_paths():
    """One kernel per formula: the numpy charge-and-derivative lives on the spec,
    FilmInterfaceTraps.sheet_charge_and_derivative; sheet_charge_C_m2 and the dual-gate solver both call it. Source
    census: no library module other than physics_traps.py walks a spec's bands() / level_kinds() (the closed form
    needs them), so a second copy (e.g. one inline in sp_dual_gate.py) cannot appear unnoticed."""
    import pathlib
    import re
    pkg = pathlib.Path(__file__).resolve().parents[1] / "dynameta"
    pat = re.compile(r"\.(bands|level_kinds)\(\)")
    hits = ["{}:{}".format(p.relative_to(pkg).as_posix(), i)
            for p in pkg.rglob("*.py") for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1)
            if pat.search(line) and p.name != "physics_traps.py"]
    assert not hits, hits
    spec = FilmInterfaceTraps(1e17, -0.5, 1.0, 0.3, levels=((0.2, 5e16), (0.6, 2e16)))
    EF = 0.5437 * Q_E                     # E_F - E_c of the film
    Vt = KB * T / Q_E
    for phi in (-1.15, 0.0, 0.33):
        eta = (EF + Q_E * phi) / (KB * T)
        q, dq = spec.sheet_charge_and_derivative(eta, V_t=Vt)
        assert trap_charge_and_derivative(spec, phi, EF, T) == (float(q), float(dq) / Vt)
        assert float(spec.sheet_charge_C_m2(eta, V_t=Vt)) == float(q)
    etas = np.linspace(-30.0, 40.0, 7)
    q, dq = spec.sheet_charge_and_derivative(etas, V_t=Vt)
    assert q.shape == dq.shape == etas.shape


def test_zero_trap_density_is_bit_identical_in_every_mode():
    """A trap spec without states adds nothing, bit for bit, also with the other options on (barrier +
    xc), and a nonzero density moves the solution (sign: acceptor-heavy traps charge negative and lower
    the face potential at fixed bias)."""
    b = OxideBarrier(2.5, 0.3 * M_E, penetration_m=2e-9)
    for kw in ({}, {"xc": "lda"}):
        a = _solver(GateStack((8e-9, 20.0), barrier=b), GateStack((8e-9, 20.0), barrier=b), **kw)
        z = _solver(GateStack((8e-9, 20.0), barrier=b, traps=FilmInterfaceTraps()),
                    GateStack((8e-9, 20.0), barrier=b, traps=FilmInterfaceTraps(dit_eV_m2=0.0)), **kw)
        x, y = a.solve_voltages(1.0, -1.0), z.solve_voltages(1.0, -1.0)
        assert np.array_equal(x.phi_ext_V, y.phi_ext_V) and np.array_equal(x.n_ext_m3, y.n_ext_m3)
    acc = FilmInterfaceTraps(1e17, -0.5, 1.5, charge="acceptor")
    w = _solver(GateStack((8e-9, 20.0)), GateStack((8e-9, 20.0), traps=acc)).solve_voltages(0.0, 1.0)
    o = _solver(GateStack((8e-9, 20.0)), GateStack((8e-9, 20.0))).solve_voltages(0.0, 1.0)
    assert w.trap_charge_top_C_m2 < 0.0 and w.phi_top_V < o.phi_top_V


def test_small_signal_gate_efficiency_with_traps():
    """d phi_s / d V_t = C_ox / (C_ox + C_s + C_it) at the top face (bottom gate fixed), hard walls,
    generic film, (0, +1) V. The trap band is centred on the operating point (neutral level = E_F - E_c
    at the no-trap face potential, +-1 eV wide), so the trapped charge vanishes there and the operating
    point is the no-trap one; C_it = -dQ_it/dphi_s = q^2 D_it (1 - 4e-17) and C_s = dF_t/dphi_t from the
    solver's own no-trap response (central differences, dV = 1e-3 V). D_it = 1e16 / 1e17 / 1e18
    m^-2 eV^-1. Budget 1e-6 relative (FD truncation ~1e-7, Gauss tolerance 1e-12 C/m^2 / dV); measured
    1.0e-8 / 1.2e-8 / 4.1e-9. Without traps (v0.15.1 has none) the efficiency stays at the trap-free
    0.22063; with traps it is 0.21716 / 0.19025 / 0.08496 (C_ox 0.02214, C_s 0.07819 F/m^2)."""
    g = GateStack((8e-9, 20.0))
    Ct = g.capacitance_F_m2
    tol = dict(gauss_tol_C_m2=1e-12, inner_tol_V=1e-12)
    s0 = _solver(g, g, **tol)
    V0, dV = 1.0, 1e-3
    p = [s0.solve_voltages(0.0, V0 + k * dV) for k in (-1, 0, 1)]
    phi_t0 = p[1].phi_top_V
    dphi = (p[2].phi_top_V - p[0].phi_top_V) / (2 * dV)
    Cs = Ct * (1.0 - dphi) / dphi                                   # dF_t = C_t (dV - dphi_t)
    e0 = (s0.E_F_J + Q_E * phi_t0) / Q_E
    for D in (1e16, 1e17, 1e18):
        tr = FilmInterfaceTraps(D, e0 - 1.0, e0 + 1.0, e0)
        s = _solver(g, GateStack((8e-9, 20.0), traps=tr), **tol)
        q = [s.solve_voltages(0.0, V0 + k * dV) for k in (-1, 0, 1)]
        assert abs(q[1].phi_top_V - phi_t0) < 1e-10 and abs(q[1].trap_charge_top_C_m2) < 1e-12
        eff = (q[2].phi_top_V - q[0].phi_top_V) / (2 * dV)
        pred = Ct / (Ct + Cs + Q_E * D)                             # q^2 D_it, D_it per J = D / q
        assert abs(eff / pred - 1.0) < 1e-6, (D, eff, pred)


@pytest.mark.parametrize("D", [1e17, 1e18])
def test_outer_jacobian_carries_the_trap_derivative(D):
    """The outer Newton's Jacobian on hard-wall faces with traps (_fd_jacobian: film part by forward difference,
    the trap term with its exact derivative dQ_it/dphi_s) equals a central difference of the COMPLETE face-law
    residual at the converged (0, +1) V point (traps on the top face, C_it = q^2 D_it = 0.016 / 0.16 F/m^2 against
    C_ox + C_s = 0.10): budget 1e-5 relative (the forward difference's truncation at dx = 1e-5 V, MEASURED 8.3e-7 /
    3.9e-7). The derivative only speeds Newton -- the solution is the same without it (the efficiency test above
    passes either way) -- so this is where a dropped trap term shows: MEASURED 1.4e-1 / 6.3e-1 with the derivative
    removed from the Jacobian."""
    tr = FilmInterfaceTraps(D, -0.5, 1.5, 0.54)
    s = _solver(GateStack((8e-9, 20.0)), GateStack((8e-9, 20.0), traps=tr), gauss_tol_C_m2=1e-12,
                inner_tol_V=1e-12)
    s.solve_voltages(0.0, 1.0)
    st = list(s._cache.values())[-1]
    pg = s.gate_potentials_V(0.0, 1.0)
    R = s._evaluate(st.x, st.phi, pg, st.n)[0]
    J = s._fd_jacobian(st.x, R, st.phi, pg, st.n)
    Jc = np.empty((2, 2))
    for j in range(2):
        xp, xm = st.x.copy(), st.x.copy()
        xp[j] += 1e-4
        xm[j] -= 1e-4
        Jc[:, j] = (s._evaluate(xp, st.phi, pg, st.n)[0] - s._evaluate(xm, st.phi, pg, st.n)[0]) / 2e-4
    assert np.max(np.abs(J - Jc)) / np.max(np.abs(Jc)) < 1e-5, (J, Jc)
    assert abs(Jc[1, 1] - Jc[0, 0]) > 0.1 * Q_E * D                 # the trap term is a large part of it


def test_traps_wide_film_agree_with_the_classical_trap_model():
    """Classical-path comparison, wide film (30 nm, n_bg 1e25, m 2 m_e, eps 9.0; top gate 8 nm / eps 20;
    bottom face ungated, so the far side is a neutral bulk and irrelevant), band of traps D_it 5e16
    m^-2 eV^-1 over E_F +- 0.4 eV with the neutral level at E_F. Classical reference written here: the
    Fermi-Dirac MOS relation V = psi_s + (F(psi_s) - Q_it(psi_s)) / C_ox with the first integral
    F^2 = 2 eps_s q int_0^psi (n_cl - N_d) dpsi' (F_1/2 by quadrature) and the classical path's trap kernel
    (FilmInterfaceTraps.sheet_charge_C_m2) at the same E_F.
    (a) Exact part: at EQUAL face potential the trap result is the classical one -- V_traps(psi) -
        V_no_traps(psi) = -Q_it(psi) / C_ox, the no-trap bias found by a secant search on the quantum
        solver: budget 2e-7 V (Gauss 1e-9 C/m^2 / C_ox = 4.5e-8 V per solve, secant to 1e-10 V).
    (b) At equal gate voltage the difference to the classical psi_s is the near-face quantum correction
        of the trap-free film, delta V(psi) = V_q(psi) - V_cl(psi), taken at the trap solution's psi and
        propagated through the classical slope dV/dpsi = 1 + (C_s + C_it) / C_ox: budget 15 % of that
        prediction plus 2e-6 V. In depletion (-0.5 V) the electrons leave the wall and the correction is
        small (measured d psi = 1.1e-4 V); in accumulation (+0.5 V) it is the hard-wall dead layer
        (measured 0.051 V with traps, 0.058 V without) -- a real quantum effect, not a trap-model
        difference."""
    from scipy.integrate import quad
    from scipy.optimize import brentq
    m, nbg, t = 2.0 * M_E, 1e25, 30e-9
    g = GateStack((8e-9, 20.0))
    Cox = g.capacitance_F_m2
    kT = KB * T
    Nc = 2.0 * (m * kT / (2.0 * np.pi * HBAR ** 2)) ** 1.5
    s_nt = _solver(None, g, t_film_m=t, nz=601, n_bg_m3=nbg, m_eff_kg=m)
    EF = s_nt.E_F_J

    def F(psi):                                       # eps_s dphi/dz at the face, along +z (toward the gate)
        val = quad(lambda p: Nc * _f_fd(0.5, (EF + Q_E * p) / kT) - nbg, 0.0, psi, epsabs=0.0, epsrel=1e-11)[0]
        return math.copysign(math.sqrt(max(2.0 * EPS_S * EPS0 * Q_E * val, 0.0)), psi)

    spec = FilmInterfaceTraps(5e16, -0.4, 0.4, EF / Q_E)

    def qit(psi):
        return trap_charge_and_derivative(spec, psi, EF, T)

    def V_cl(psi, traps):
        return psi + (F(psi) - (qit(psi)[0] if traps else 0.0)) / Cox

    s_tr = _solver(None, GateStack((8e-9, 20.0), traps=spec), t_film_m=t, nz=601, n_bg_m3=nbg, m_eff_kg=m)
    for V in (-0.5, 0.5):
        psi1 = s_tr.solve_voltages(0.0, V).phi_top_V
        # (a) the no-trap bias with the same face potential (secant on the quantum solver)
        v0, v1 = V, V - qit(psi1)[0] / Cox
        f0 = s_nt.solve_voltages(0.0, v0).phi_top_V - psi1
        for _ in range(30):
            f1 = s_nt.solve_voltages(0.0, v1).phi_top_V - psi1
            if abs(f1) < 1e-10:
                break
            v0, v1, f0 = v1, v1 - f1 * (v1 - v0) / (f1 - f0), f1
        assert abs(f1) < 1e-10
        assert abs((V - v1) - (-qit(psi1)[0] / Cox)) < 2e-7, (V, V - v1, -qit(psi1)[0] / Cox)
        # (b) equal gate voltage: the trap-free quantum correction at psi1, propagated
        dV = v1 - V_cl(psi1, False)                    # V_q(psi1) - V_cl(psi1), trap-free
        psi_cl = brentq(lambda p: V_cl(p, True) - V, -2.0, 2.0, xtol=1e-14)
        dp = 1e-5
        slope = 1.0 + ((F(psi_cl + dp) - F(psi_cl - dp)) / (2 * dp) - qit(psi_cl)[1]) / Cox
        pred = -dV / slope
        d1 = psi1 - psi_cl
        assert abs(d1 - pred) < 0.15 * abs(pred) + 2e-6, (V, d1, pred)
        if V < 0.0:
            assert abs(d1) < 3e-4, d1


# ---------------------------------------------------------------------------------------------------
# composition: options x asymmetric gates x Kane x cache x QuantumFilm
# ---------------------------------------------------------------------------------------------------

def test_all_options_compose_with_kane_cache_and_quantum_film():
    """Barrier + xc + traps + Kane (alpha 0.4) on asymmetric stacks: Gauss at both matching planes, charge
    balance; a sweep is warm-started from the cache and equals a cold solve (1e-9 V / n_bg); the cache is
    keyed by the options (switching xc on an existing solver rebuilds and uses a separate cache, and
    switching back hits the original entries); QuantumFilm.from_dual_gate builds on the extended domain
    with the solver's potential energy (band offsets, -q phi, V_xc) and per-cell masses, reproduces the
    solver's sub-band energies (1e-12 relative) and density (1e-9 n_bg), and its response is computable
    (real-space and modes routes agree). The CarrierField reports the in-film density in the region and
    the oxide electrons in the extras."""
    from dynameta.optics.quantum_film import QuantumFilm, QuantumFilmResponse
    from dynameta.sweep import BiasPoint
    bot, top = _stacks_barrier(B_BOT, B_TOP, TRAPS)
    s = _solver(bot, top, xc="lda", alpha_np_per_eV=0.4)
    sweep = [(0.0, v) for v in (-1.0, -0.5, 0.0, 0.5, 1.0)]
    sols = [s.solve_voltages(*V) for V in sweep]
    assert all(max(abs(x.gauss_residual_bottom_C_m2), abs(x.gauss_residual_top_C_m2)) < 1e-9 for x in sols)
    assert all(abs(x.neutrality_residual_C_m2) < 2e-9 for x in sols)
    assert max(x.inner_solves for x in sols[1:]) <= 12
    cold = _solver(bot, top, xc="lda", alpha_np_per_eV=0.4).solve_voltages(*sweep[3])
    assert abs(cold.phi_top_V - sols[3].phi_top_V) < 1e-9
    assert np.max(np.abs(cold.n_ext_m3 - sols[3].n_ext_m3)) < 1e-9 * NBG
    n_cached = len(s._cache)
    s.xc = None
    assert len(s._cache) == 0
    alt = s.solve_voltages(*sweep[3])
    assert abs(alt.phi_top_V - sols[3].phi_top_V) > 1e-4
    s.xc = "lda"
    assert len(s._cache) == n_cached and s.solve_voltages(*sweep[3]).from_cache
    sol = sols[4]
    film = QuantumFilm.from_dual_gate(s, sol)
    assert film.z_m.size == sol.z_ext_m.size and film.inv_mass_cell_per_kg is not None
    k = sol.subbands.energies_J.size
    assert np.max(np.abs(film.energies_J[:k] - sol.subbands.energies_J)) < 1e-12 * Q_E
    assert np.max(np.abs(film.density_m3 - sol.n_ext_m3)) < 1e-9 * NBG
    w = 1.57e15                                  # ~1200 nm
    r_m = QuantumFilmResponse(film, eps_inf=4.0, gamma_rad_s=1.2e14, route="modes").alpha_perp(w)
    r_r = QuantumFilmResponse(film, eps_inf=4.0, gamma_rad_s=1.2e14, route="real_space").alpha_perp(w)
    assert abs(r_m - r_r) < 1e-8 * abs(r_m)
    cf = s.solve(BiasPoint({"bottom": 0.0, "top": 1.0}, "b"))
    reg = cf.regions["semi"]
    assert np.array_equal(reg.grid_fields["electron_density_m3"][0, 0], sol.n_m3)
    assert cf.extras["oxide_electron_sheet_top_m2"] > 0.0
    assert np.array_equal(cf.extras["electron_density_ext_m3"], sol.n_ext_m3)
