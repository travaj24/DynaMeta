"""D2d oracle: BIPOLAR density-gradient (electrons AND holes carry a quantum potential) in the coupled
DEVSIM Newton -- physics_density_gradient.setup_bipolar_dg / setup_contact_bipolar_dg /
setup_dg_insulator_wall. Formulation + citations: docs/audit/2026-11-03-devsim-bipolar-density-
gradient.md. Every oracle below is INDEPENDENT of the code under test (closed-form quadrature, the
in-house Schrodinger-Poisson solver, the post-hoc numpy BVP, a classical DEVSIM solve).

Model masses (isotropic, Si-like): m_n = 0.26 m0, m_p = 0.49 m0 -> L_q = sqrt(b/V_t) = 1.375 / 1.001 nm
at gamma = 1 (the Ancona-Iafrate high-temperature value; NOT fitted anywhere below).

GATE A (classical limit): a p-n diode with the bipolar DG ACTIVE at b = 0 (set_dg_gamma(0)) reproduces
        the classical bipolar DD solution (Potential/Electrons/Holes and both terminal currents) to
        solver tolerance, with Lambda_n = Lambda_p = 0.
GATE B (analytic dark space): a flat-band intrinsic bar against an impenetrable wall, potential frozen:
        the DG equations integrate in closed form, z(t) = L_q int_0^t ds/sqrt(F(s)),
        F(t) = t^2 ln t^2 - t^2 + 1, t = sqrt(c/c_bulk), so the dead-layer deficit is
        int (c_bulk - c) dz = I c_bulk L_q with I = int_0^1 (1-t^2)/sqrt(F) dt = 1.187207 -- for
        EACH carrier with its OWN L_q (electrons 1.375 nm, holes 1.001 nm), at gamma = 1 and 1/4.
        Oracle sanity (reported): the Schrodinger-Poisson free gas at the same wall reproduces the
        exact quantum deficit (sqrt(pi)/(2 sqrt 2)) hbar c/sqrt(m kT); DG/QM = 0.7734 is the known
        first-order-gradient shortfall (gamma_eq = 1.67), not a defect.
GATE C (MOS vs Schrodinger-Poisson, the p-channel is the NEW physics): strong ACCUMULATION
        (N = 1e24 m^-3, |psi_s - psi_b| = 0.15 V; peak c/N_c ~ 0.9-1.5) of electrons on n-type AND of
        HOLES on p-type, Fermi-Dirac-consistent DD (N_c, N_v of the SP model), fixed surface potential.
        Against the self-consistent 1-D Schrodinger-Poisson (carriers.schrodinger_poisson; holes by
        the exact electron<->hole mirror) the DG must close >= 75 % of the classical-to-SP gap in BOTH
        the excess sheet density and its centroid (|DG - SP| <= 0.25 |classical - SP|), with the peak
        pushed off the wall to 0.5-1.5x the SP peak depth (classical: AT the wall).
GATE D (fixed point): Boltzmann accumulation (3e23, 0.1 V), both carriers: the converged DG density
        equals the post-hoc DG boundary-value problem (carriers.density_gradient.dg_correct_density_1d,
        scipy solve_bvp) evaluated on the DG solution's OWN potential, to < 2e-3 of the peak.
GATE E (equilibrium, bipolar): INVERSION (p-type 3e23, psi_s - psi_b = 0.60 V: n_s ~ 1e5 p_s), both carriers
        DG at the wall: the modified quasi-Fermi levels phi_n = psi + Lambda_n - V_t ln(n/n_i),
        phi_p = psi - Lambda_p + V_t ln(p/n_i) are flat (|phi| < 1e-9 V), n p = n_i^2 exp((Ln+Lp)/V_t)
        at every node and n p = n_i^2 in the neutral bulk. With quantum_srh=False the classical SRH
        generates in the dark space and the QFLs are NOT flat (reported).
GATE F (transport): a 1-D Si p-n diode (1e23 / 1e22) at +0.4 V and -0.5 V: DG changes the terminal
        current by < 1e-2 relative and by no more than 2 max|Lambda|/V_t (first order), the change
        scales LINEARLY with gamma (ratio gamma=1 / gamma=0.5 within 2 +- 0.2), and the two terminal
        currents conserve to 1e-8 (both carriers' contacts read the DG flux).

Run: python -m validation.bipolar_density_gradient
"""
import contextlib
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import devsim as ds  # noqa: E402

from dynameta.constants import HBAR, KB, M_E, Q_E, T_REF, V_T  # noqa: E402
from dynameta.carriers import eq_registry as _R  # noqa: E402
from dynameta.carriers import physics_bipolar_dd as BP  # noqa: E402
from dynameta.carriers import physics_density_gradient as DG  # noqa: E402
from dynameta.core.numerics import trapz  # noqa: E402

MN, MP = 0.26 * M_E, 0.49 * M_E          # isotropic model masses (electrons, holes)
GVN, GVP = 6, 2                          # valley degeneracies of the SP model -> its N_c, N_v
EPS = 11.7
NI_SI = 1.0e16                           # m^-3
ABS_ERR, REL_ERR, MAX_ITER = 1.0e10, 1.0e-7, 100
W_METRIC = 20e-9                         # sheet/centroid window from the wall


@contextlib.contextmanager
def _quiet():
    sys.stdout.flush()
    saved, devnull = os.dup(1), os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, 1)
        yield
    finally:
        sys.stdout.flush()
        os.dup2(saved, 1)
        os.close(devnull)
        os.close(saved)


def _solve(rel=REL_ERR):
    with _quiet():
        ds.solve(type="dc", absolute_error=ABS_ERR, relative_error=rel, maximum_iterations=MAX_ITER)


def _ramp_param(dev, reg, name, target, n_steps, *, min_frac=1.0 / 512):
    """Continuation of a device parameter to `target` in n_steps equal steps, BISECTING a step that
    fails (the solution is saved before each step and restored on failure). Used for the surface
    potential near the onset of inversion, where a fixed 10 mV step can stall."""
    sols = [v for v in ("Potential", "Electrons", "Holes", "QSqrtN", "QLambda", "QSqrtP", "QLambdaP")
            if v in ds.get_node_model_list(device=dev, region=reg)]
    v0 = float(ds.get_parameter(device=dev, name=name))
    frac, df = 0.0, 1.0 / n_steps
    while frac < 1.0 - 1e-12:
        step = min(df, 1.0 - frac)
        saved = {v: list(ds.get_node_model_values(device=dev, region=reg, name=v)) for v in sols}
        ds.set_parameter(device=dev, name=name, value=v0 + (target - v0) * (frac + step))
        try:
            _solve()
        except ds.error:
            for v, vals in saved.items():
                ds.set_node_values(device=dev, region=reg, name=v, values=vals)
            if step / 2.0 < min_frac / n_steps:
                raise
            df = step / 2.0
            continue
        frac += step
        df = min(1.0 / n_steps, 1.5 * step)


def _drop(dev, mesh):
    with _quiet():
        for d in list(ds.get_device_list()):
            if d == dev:
                ds.delete_device(device=d)
        for m in list(ds.get_mesh_list()):
            if m == mesh:
                ds.delete_mesh(mesh=m)
    _R.clear(dev)


def _get(dev, reg, name):
    return np.asarray(ds.get_node_model_values(device=dev, region=reg, name=name), dtype=np.float64)


def nc_model(m, gv, T=T_REF):
    """Effective DOS 2 g_v (m kT/(2 pi hbar^2))^(3/2) of the isotropic SP model."""
    return 2.0 * gv * (m * KB * T / (2.0 * np.pi * HBAR ** 2)) ** 1.5


def _f12(eta):
    from scipy.integrate import quad
    return 2.0 / np.sqrt(np.pi) * quad(lambda e: np.sqrt(e) / (1.0 + np.exp(e - eta)),
                                        0.0, max(eta, 0.0) + 60.0, limit=200)[0]


def dg_wall_constant():
    """I = int_0^1 (1 - t^2)/sqrt(t^2 ln t^2 - t^2 + 1) dt: the closed-form DG dead-layer deficit in
    units of c_bulk L_q (frozen flat potential, Boltzmann, impenetrable wall)."""
    from scipy.integrate import quad
    return quad(lambda t: (1.0 - t * t) / np.sqrt(t * t * np.log(t * t) - t * t + 1.0)
                if t > 0 else 1.0, 0.0, 1.0, limit=400, epsabs=1e-13, epsrel=1e-12)[0]


def dg_wall_depth(t):
    """z(t)/L_q = int_0^t ds/sqrt(F(s)) of the same closed form (the exact profile, inverted)."""
    from scipy.integrate import quad
    return quad(lambda s: 1.0 / np.sqrt(s * s * np.log(s * s) - s * s + 1.0) if s > 0 else 1.0,
                0.0, t, limit=400, epsabs=1e-13, epsrel=1e-12)[0]


def _bar(dev, mesh, length, *, ps_wall=0.05e-9, net_doping="0.0", n_i=NI_SI, fd=False,
         n_dos=None, n_dos_p=None, quantum_mesh=True):
    _drop(dev, mesh)
    with _quiet():
        ds.create_1d_mesh(mesh=mesh)
        ds.add_1d_mesh_line(mesh=mesh, pos=0.0, ps=ps_wall, tag="wall")
        if quantum_mesh:
            ds.add_1d_mesh_line(mesh=mesh, pos=5e-9, ps=0.1e-9)
            ds.add_1d_mesh_line(mesh=mesh, pos=25e-9, ps=0.5e-9)
        ds.add_1d_mesh_line(mesh=mesh, pos=length, ps=2e-9, tag="sub")
        ds.add_1d_contact(mesh=mesh, name="wall", tag="wall", material="metal")
        ds.add_1d_contact(mesh=mesh, name="sub", tag="sub", material="metal")
        ds.add_1d_region(mesh=mesh, material="Si", region="s", tag1="wall", tag2="sub")
        ds.finalize_mesh(mesh=mesh)
        ds.create_device(mesh=mesh, device=dev)
        ds.node_model(device=dev, region="s", name="NetDoping", equation=net_doping)
        BP.setup_bipolar_region(dev, "s", eps_static=EPS,
                                n_dos_m3=n_dos if n_dos else nc_model(MN, GVN), n_i_m3=n_i,
                                mobility_n_m2Vs=0.1, mobility_p_m2Vs=0.04, tau_n_s=1e-7,
                                tau_p_s=1e-7, fd_enhancement=fd,
                                n_dos_p_m3=n_dos_p if n_dos_p else nc_model(MP, GVP))
        BP.setup_equilibrium_seed_models(dev, "s")
        BP.setup_contact_ohmic_bipolar(dev, "sub")
        for nm, src in (("Electrons", "IntrinsicElectrons"), ("Holes", "IntrinsicHoles")):
            ds.set_node_values(device=dev, region="s", name=nm, init_from=src)
    return "s"


def _dg_ramp(dev, reg, *, gamma_n=1.0, gamma_p=1.0, quantum_srh=True, fracs=(0.1, 0.3, 1.0),
             wall=True, rel=REL_ERR, u_floor_rel=1.0e-6):
    """Install the bipolar DG (+ the wall), seed at the first b fraction, and continue b to its full
    value. A fraction whose solve fails is retried from the last converged state after an inserted
    geometric-mean step (save/restore of every solution variable), down to a 1/64 ratio."""
    with _quiet():
        DG.setup_bipolar_dg(dev, reg, m_eff_n_kg=MN, m_eff_p_kg=MP, gamma_n=gamma_n,
                            gamma_p=gamma_p, quantum_srh=quantum_srh)
        DG.setup_contact_bipolar_dg(dev, "sub")
        if wall:
            DG.setup_dg_insulator_wall(dev, "wall", u_floor_rel=u_floor_rel)
    sols = ("Potential", "Electrons", "Holes", "QSqrtN", "QLambda", "QSqrtP", "QLambdaP")
    todo, done = list(fracs), None
    while todo:
        fr = todo[0]
        saved = ({v: list(ds.get_node_model_values(device=dev, region=reg, name=v)) for v in sols}
                 if done is not None else None)
        DG.set_dg_gamma(dev, reg, fr)
        if done is None:
            DG.seed_bipolar_dg(dev, reg, walls=("wall",) if wall else (), u_floor_rel=u_floor_rel)
        try:
            _solve(rel)
        except ds.error:
            if done is None or fr / done < 1.0 + 1.0 / 64:
                raise
            for v, vals in saved.items():
                ds.set_node_values(device=dev, region=reg, name=v, values=vals)
            todo.insert(0, float(np.sqrt(fr * done)))
            continue
        done = fr
        todo.pop(0)


# ---------------------------------------------------------------------------------------------
# GATE B: closed-form dark space (frozen flat potential)
# ---------------------------------------------------------------------------------------------

def flat_wall(gamma=1.0, ni=1.0e22, length=60e-9, u_floor_rel=1.0e-6):
    """Intrinsic bar (n = p = ni) against the wall, Boltzmann, Potential frozen after the (flat)
    classical solve. Returns {carrier: (deficit/(c L_q), max |z_node - z_exact(t_node)|/L_q)}."""
    from dynameta.carriers.density_gradient import dg_length_m
    dev, mesh = "bdg_flat", "bdg_flat_m"
    try:
        reg = _bar(dev, mesh, length, n_i=ni)
        nn = _get(dev, reg, "x").size
        ds.set_node_values(device=dev, region=reg, name="Potential", values=[0.0] * nn)
        _solve()
        _R.delete_by_name(dev, "PotentialEquation")                  # freeze psi (flat)
        # ramp from the SAME first quantum length whatever gamma (b ~ gamma frac): a first step whose
        # L_q is ~4 mesh cells (0.1 x gamma = 0.025 -> 0.22 nm on the 0.05 nm wall mesh) stalls
        _dg_ramp(dev, reg, gamma_n=gamma, gamma_p=gamma, u_floor_rel=u_floor_rel,
                 fracs=tuple(f for f in (min(1.0, 0.1 / gamma), 0.3 / gamma, 1.0) if f <= 1.0))
        x = _get(dev, reg, "x")
        o = np.argsort(x)
        x = x[o]
        out = {}
        for key, var, m in (("electrons", "Electrons", MN), ("holes", "Holes", MP)):
            c = _get(dev, reg, var)[o]
            lq = dg_length_m(m, gamma=gamma)
            deficit = float(trapz(ni - c, x)) / (ni * lq)
            t = np.sqrt(c / ni)
            sel = (t > 0.05) & (t < 0.95)
            zerr = max(abs(x[i] / lq - dg_wall_depth(t[i])) for i in np.flatnonzero(sel))
            out[key] = (deficit, float(zerr), lq)
        return out
    finally:
        _drop(dev, mesh)


def sp_free_gas_deficit(m=MN):
    """Oracle sanity: the Schrodinger-Poisson free (zero-field, non-degenerate) gas at an infinite
    wall, deficit in units of c_bulk hbar/sqrt(m kT); exact = sqrt(pi)/(2 sqrt 2) = 0.626657."""
    from dynameta.carriers.schrodinger_poisson import SchrodingerPoisson1D
    h, length = 0.05e-9, 60e-9
    z = np.arange(0.0, length + 0.5 * h, h)
    sp = SchrodingerPoisson1D(z, m, T_K=T_REF, g_s=2, g_v=1)
    res = sp.density(np.zeros_like(z), -8.0 * KB * T_REF, n_states=400, bound_tol=1e9,
                     neumann_left=True)
    n = np.zeros_like(z)
    n[:-1] = res.density_m3
    d, n = (length - z)[::-1], n[::-1]
    nb = float(np.interp(30e-9, d, n))
    msk = d < 15e-9
    return float(trapz(nb - n[msk], d[msk])) / (nb * HBAR / np.sqrt(m * KB * T_REF))


# ---------------------------------------------------------------------------------------------
# GATE C / D / E: MOS with a fixed surface potential
# ---------------------------------------------------------------------------------------------

def _metrics(d, c, N):
    msk = d <= W_METRIC
    ex = c[msk] - N
    s = float(trapz(ex, d[msk]))
    return {"sheet": s, "centroid": float(trapz(d[msk] * ex, d[msk])) / s,
            "peak": float(d[msk][int(np.argmax(c[msk]))])}


def sp_accumulation(carrier, N, delta, length=80e-9):
    """Self-consistent Schrodinger-Poisson accumulation layer of the MAJORITY carrier (holes by the
    exact mirror psi -> -psi, N_A -> N_D, m -> m_p), Fermi-Dirac, bulk-buffer body, wall at the far
    end. Returns (distance from wall, density) and the SP metadata."""
    from scipy.optimize import brentq
    from dynameta.carriers.schrodinger_poisson import SchrodingerPoisson1D
    m, gv = (MN, GVN) if carrier == "electrons" else (MP, GVP)
    h = 0.05e-9
    z = np.arange(0.0, length + 0.5 * h, h)
    sp = SchrodingerPoisson1D(z, m, T_K=T_REF, g_s=2, g_v=gv)
    nc = nc_model(m, gv)
    eta = brentq(lambda e: nc * _f12(e) - N, -40.0, 20.0)
    phi, n, res = sp.solve_self_consistent(eps_r=EPS, doping_m3=np.full_like(z, N),
                                           E_F_J=eta * KB * T_REF, phi_left_V=0.0,
                                           phi_right_V=abs(delta), max_outer=200, tol_V=1e-7,
                                           n_states=160, bound_tol=1e9, relax=0.7,
                                           bulk_buffer_m=15e-9)
    return (length - z)[::-1], n[::-1], {"converged": bool(res.converged), "eta_bulk": eta,
                                         "max_c_over_Nc": float(np.max(n) / nc)}


def mos(carrier, N, delta, *, fd=True, length=80e-9, quantum_srh=True, steps=5, dg=True,
        gamma=1.0, rel=REL_ERR):
    """1-D MOS, fixed surface potential psi_s = psi_b + delta at the wall (an impenetrable oxide),
    body ohmic contact. carrier: the MAJORITY ('electrons' = n-type N, 'holes' = p-type N).
    Returns dict(x, psi, n, p, lam_n, lam_p, classical (n, p, psi), psi_b)."""
    dev, mesh = "bdg_mos", "bdg_mos_m"
    sgn = 1.0 if carrier == "electrons" else -1.0
    try:
        reg = _bar(dev, mesh, length, net_doping="{:.17e}".format(sgn * N), fd=fd)
        psi_b = BP.ohmic_built_in_offset_V(sgn * N, n_i_m3=NI_SI, n_dos_m3=nc_model(MN, GVN),
                                           n_dos_p_m3=nc_model(MP, GVP), fd_enhancement=fd)
        ds.set_parameter(device=dev, name="wall_bias", value=psi_b)
        with _quiet():
            ds.contact_node_model(device=dev, contact="wall", name="wall_psi",
                                  equation="Potential - wall_bias")
            ds.contact_node_model(device=dev, contact="wall", name="wall_psi:Potential",
                                  equation="1")
            _R.record_contact_equation(dev, "wall", name="PotentialEquation", node_model="wall_psi")
        nn = _get(dev, reg, "x").size
        ds.set_node_values(device=dev, region=reg, name="Potential", values=[psi_b] * nn)
        _solve()
        _ramp_param(dev, reg, "wall_bias", psi_b + delta, steps)
        x = _get(dev, reg, "x")
        o = np.argsort(x)
        out = {"x": x[o], "psi_b": psi_b,
               "classical": {v: _get(dev, reg, v)[o] for v in ("Potential", "Electrons", "Holes")}}
        if dg:
            _dg_ramp(dev, reg, gamma_n=gamma, gamma_p=gamma, quantum_srh=quantum_srh, rel=rel)
            for v in ("Potential", "Electrons", "Holes", "QLambda", "QLambdaP"):
                out[v] = _get(dev, reg, v)[o]
        return out
    finally:
        _drop(dev, mesh)


def gamma_wall():
    """gamma at which the DG hard-wall deficit (I c L_q, L_q ~ sqrt(gamma)) equals the EXACT quantum
    deficit of a non-degenerate free gas at an infinite wall, (sqrt(pi)/(2 sqrt 2)) hbar c/sqrt(m kT):
    gamma_wall = (3 pi / 4) / I^2 = 1.6717. Fixed a priori by that closed-form limit (GATE B), NOT
    fitted to any MOS solve."""
    return 0.75 * np.pi / dg_wall_constant() ** 2


def mos_vs_sp(carrier, N=1.0e24, delta=0.15, gammas=(1.0,)):
    """GATE C numbers: {'SP'|'classical'|('DG', gamma): metrics} + SP metadata."""
    sgn = 1.0 if carrier == "electrons" else -1.0
    var = "Electrons" if carrier == "electrons" else "Holes"
    d_sp, c_sp, meta = sp_accumulation(carrier, N, delta)
    out = {"SP": _metrics(d_sp, c_sp, N), "meta": meta}
    for g in gammas:
        r = mos(carrier, N, sgn * delta, fd=True, gamma=g)
        out["classical"] = _metrics(r["x"], r["classical"][var], N)
        out[("DG", g)] = _metrics(r["x"], r[var], N)
    return out


def fixed_point(carrier, N=3.0e23, delta=0.1):
    """GATE D: max |c_DG - c_BVP| / max(c_DG), the BVP on the DG solution's own potential."""
    from dynameta.carriers.density_gradient import dg_correct_density_1d
    sgn = 1.0 if carrier == "electrons" else -1.0
    r = mos(carrier, N, sgn * delta, fd=False)
    var, m = ("Electrons", MN) if carrier == "electrons" else ("Holes", MP)
    c_cl = N * np.exp(sgn * (r["Potential"] - r["psi_b"]) / V_T)    # Boltzmann at the DG psi
    c_bvp = dg_correct_density_1d(r["x"], c_cl, m, gamma=1.0, hard_wall="left", tol=1e-8)
    return float(np.max(np.abs(r[var] - c_bvp)) / np.max(r[var]))


def inversion_equilibrium(quantum_srh=True, N_A=3.0e23, delta=0.60, length=150e-9):
    """GATE E: the modified quasi-Fermi levels and the n p product in a strongly inverted MOS."""
    # REL 1e-6: the inversion layer's DG Newton floors at ~1e-6 relative electron updates (16 decades of
    # electron density across the depletion region; the unipolar dg_dd_in_newton note) -- a 1e-7 gate
    # ping-pongs there. phi is then exact to ~V_t x 1e-6.
    r = mos("holes", N_A, delta, fd=False, length=length, quantum_srh=quantum_srh, steps=16,
            rel=1e-6)
    psi, n, p = r["Potential"], r["Electrons"], r["Holes"]
    ln, lp = r["QLambda"], r["QLambdaP"]
    phi_n = psi + ln - V_T * np.log(n / NI_SI)
    phi_p = psi - lp + V_T * np.log(p / NI_SI)
    # log form: |ln(n p / n_i^2) - (Ln + Lp)/V_t| (~ the relative error; no exp under/overflow when a
    # broken reference -- quantum_srh=False -- drives a wall Lambda to thousands of V_t)
    np_q = np.abs(np.log(n / NI_SI) + np.log(p / NI_SI) - (ln + lp) / V_T)
    bulk = r["x"] > 0.8 * length
    return {"max_phi_n": float(np.max(np.abs(phi_n))), "max_phi_p": float(np.max(np.abs(phi_p))),
            "max_phi_interior": float(max(np.max(np.abs(phi_n[1:])), np.max(np.abs(phi_p[1:])))),
            "np_quantum": float(np.max(np_q)),
            "np_bulk": float(np.max(np.abs(n[bulk] * p[bulk] / NI_SI ** 2 - 1.0))),
            "lam_wall_n_Vt": float(ln[0] / V_T), "lam_wall_p_Vt": float(lp[0] / V_T),
            "inversion_peak_nm": float(r["x"][int(np.argmax(n))] * 1e9),
            "classical_peak_nm": float(r["x"][int(np.argmax(r["classical"]["Electrons"]))] * 1e9)}


# ---------------------------------------------------------------------------------------------
# GATE A / F: p-n diode
# ---------------------------------------------------------------------------------------------

DIODE_NA, DIODE_ND, DIODE_LEN, DIODE_XJ = 1.0e23, 1.0e22, 2.0e-6, 1.0e-6


def _diode(dev, mesh):
    _drop(dev, mesh)
    with _quiet():
        ds.create_1d_mesh(mesh=mesh)
        ds.add_1d_mesh_line(mesh=mesh, pos=0.0, ps=10e-9, tag="anode")
        ds.add_1d_mesh_line(mesh=mesh, pos=DIODE_XJ, ps=1e-9)
        ds.add_1d_mesh_line(mesh=mesh, pos=DIODE_LEN, ps=10e-9, tag="cathode")
        ds.add_1d_contact(mesh=mesh, name="anode", tag="anode", material="metal")
        ds.add_1d_contact(mesh=mesh, name="cathode", tag="cathode", material="metal")
        ds.add_1d_region(mesh=mesh, material="Si", region="s", tag1="anode", tag2="cathode")
        ds.finalize_mesh(mesh=mesh)
        ds.create_device(mesh=mesh, device=dev)
        ds.node_model(device=dev, region="s", name="NetDoping",
                      equation="ifelse(x < {:.6e}, {:.6e}, {:.6e})".format(DIODE_XJ, -DIODE_NA,
                                                                          DIODE_ND))
        BP.setup_bipolar_region(dev, "s", eps_static=EPS, n_dos_m3=2.8e25, n_i_m3=NI_SI,
                                mobility_n_m2Vs=0.135, mobility_p_m2Vs=0.048, tau_n_s=1e-7,
                                tau_p_s=1e-7, fd_enhancement=False, n_dos_p_m3=1.04e25)
        BP.setup_equilibrium_seed_models(dev, "s")
        BP.setup_contact_ohmic_bipolar(dev, "anode")
        BP.setup_contact_ohmic_bipolar(dev, "cathode")
        for nm, src in (("Electrons", "IntrinsicElectrons"), ("Holes", "IntrinsicHoles")):
            ds.set_node_values(device=dev, region="s", name=nm, init_from=src)
        ds.node_model(device=dev, region="s", name="_seed_psi",
                      equation=BP.equilibrium_seed_psi_expr(BP.fd_shift_model(dev, "s")))
        ds.set_node_values(device=dev, region="s", name="Potential", init_from="_seed_psi")
    _solve()
    return "s"


def _terminal(dev, contact):
    return sum(float(ds.get_contact_current(device=dev, contact=contact, equation=eq))
               for eq in ("ElectronContinuityEquation", "HoleContinuityEquation"))


def _ramp_bias(dev, v, step=0.05):
    v0 = float(ds.get_parameter(device=dev, name="anode_bias"))
    nstep = int(np.ceil(abs(v - v0) / step))
    for k in range(1, nstep + 1):
        ds.set_parameter(device=dev, name="anode_bias", value=v0 + (v - v0) * k / nstep)
        _solve()


def diode(gamma_fracs=(None,), biases=(0.4, -0.5)):
    """Terminal currents of the diode for each DG fraction (None = classical, no DG installed;
    0.0 = DG installed at b = 0). Returns {frac: {bias: dict(I, I_sum, arrays, max_lam_Vt)}}."""
    out = {}
    for fr in gamma_fracs:
        dev, mesh = "bdg_diode", "bdg_diode_m"
        try:
            reg = _diode(dev, mesh)
            if fr is not None:
                with _quiet():
                    DG.setup_bipolar_dg(dev, reg, m_eff_n_kg=MN, m_eff_p_kg=MP)
                    DG.setup_contact_bipolar_dg(dev, "anode")
                    DG.setup_contact_bipolar_dg(dev, "cathode")
                DG.set_dg_gamma(dev, reg, fr)
                DG.seed_bipolar_dg(dev, reg)
                _solve()
            res = {}
            for v in biases:
                _ramp_bias(dev, 0.0)                     # back to equilibrium in steps
                _ramp_bias(dev, v)
                ia, ic = _terminal(dev, "anode"), _terminal(dev, "cathode")
                # the net-recombination current q int U dx: in steady state it IS the terminal current,
                # but well-conditioned at reverse bias, where the terminal value is a majority-carrier
                # flux ~1e-12 of its drift/diffusion parts (measured jitter ~1e-3 relative, classical
                # AND DG alike, between equivalent converged solves)
                u_tot = _get(dev, reg, "USRH")
                if "RTerm_" + DG.DG_SRH_TERM in ds.get_node_model_list(device=dev, region=reg):
                    u_tot = u_tot + _get(dev, reg, "RTerm_" + DG.DG_SRH_TERM)
                i_rec = Q_E * float(np.sum(u_tot * _get(dev, reg, "NodeVolume")))
                rec = {"I": ia, "I_rec": i_rec, "I_sum": abs(ia + ic) / abs(ia),
                       "arrays": {k: _get(dev, reg, k) for k in ("Potential", "Electrons", "Holes")}}
                if fr is not None:
                    rec["max_lam_Vt"] = max(float(np.max(np.abs(_get(dev, reg, k)))) for k in
                                            ("QLambda", "QLambdaP")) / V_T
                res[v] = rec
            out[fr] = res
        finally:
            _drop(dev, mesh)
    return out


# ---------------------------------------------------------------------------------------------

def main():
    print("[bdg] === D2d bipolar density-gradient ===", flush=True)
    ok = True
    t0 = time.time()

    # GATE A + F (one diode family)
    d = diode(gamma_fracs=(None, 0.0, 0.5, 1.0), biases=(0.4, 0.2, -0.5))
    cl = d[None]
    errs_a = []
    for v in (0.4, 0.2, -0.5):
        for k in ("Potential", "Electrons", "Holes"):
            a, b = cl[v]["arrays"][k], d[0.0][v]["arrays"][k]
            errs_a.append(float(np.max(np.abs(a - b) / np.maximum(np.abs(a), 1e-30))))
        errs_a.append(abs(d[0.0][v]["I"] / cl[v]["I"] - 1.0))
    g_a = bool(max(errs_a) < 1e-9 and d[0.0][0.4]["max_lam_Vt"] < 1e-12)
    ok = ok and g_a
    print("[bdg] GATE A: DG active at b = 0 vs classical bipolar DD: max rel {:.1e} (fields + both "
          "currents), max|Lambda| = {:.1e} V_t -> {}".format(max(errs_a), d[0.0][0.4]["max_lam_Vt"],
                                                             "PASS" if g_a else "FAIL"), flush=True)
    g_f = True
    for v in (0.4, 0.2, -0.5):
        key = "I" if v > 0 else "I_rec"                # reverse: the well-conditioned q int U dx
        r1 = d[1.0][v][key] / cl[v][key] - 1.0
        rh = d[0.5][v][key] / cl[v][key] - 1.0
        lam = d[1.0][v]["max_lam_Vt"]
        ratio = r1 / rh if rh != 0.0 else float("inf")
        cons = d[1.0][v]["I_sum"]
        gv = bool(abs(r1) < 1e-2 and abs(r1) <= 2.0 * lam and 1.8 < ratio < 2.2
                  and (v < 0 or cons < max(1e-6, 10.0 * cl[v]["I_sum"])))
        g_f = g_f and gv
        print("[bdg] GATE F @ {:+.2f} V ({}): classical {:.4e} A/m^2, dI/I(gamma=1) = {:+.3e}, "
              "(gamma=0.5) = {:+.3e}, ratio {:.3f}; max|Lambda| = {:.2e} V_t; |I_a + I_c|/|I| = "
              "{:.1e} (classical {:.1e}) -> {}".format(
                  v, "terminal" if v > 0 else "q int U dx", cl[v][key], r1, rh, ratio, lam, cons,
                  cl[v]["I_sum"], "PASS" if gv else "FAIL"), flush=True)
    ok = ok and g_f

    # GATE B
    ii = dg_wall_constant()
    g_b = True
    for gam in (1.0, 0.25):
        fw = flat_wall(gamma=gam)
        for key, (dfc, zerr, lq) in fw.items():
            gb = bool(abs(dfc / ii - 1.0) < 1e-3 and zerr < 2e-2)
            g_b = g_b and gb
            print("[bdg] GATE B gamma={:.2f} {:9s}: L_q = {:.4f} nm, deficit/(c L_q) = {:.6f} vs "
                  "closed form I = {:.6f} (rel {:+.1e}); profile max |dz|/L_q = {:.1e} -> {}".format(
                      gam, key, lq * 1e9, dfc, ii, dfc / ii - 1.0, zerr, "PASS" if gb else "FAIL"),
                  flush=True)
    qm = sp_free_gas_deficit()
    exact_qm = np.sqrt(np.pi) / (2.0 * np.sqrt(2.0))
    g_bs = abs(qm / exact_qm - 1.0) < 1e-3
    g_b = g_b and g_bs
    print("[bdg] GATE B oracle sanity: SP free-gas wall deficit {:.6f} vs exact {:.6f}; DG(gamma=1)/QM "
          "= {:.4f} (first-order-gradient shortfall, gamma_eq = {:.3f}) -> {}".format(
              qm, exact_qm, ii / np.sqrt(6.0) / exact_qm, (exact_qm * np.sqrt(6.0) / ii) ** 2,
              "PASS" if g_bs else "FAIL"), flush=True)
    ok = ok and g_b

    # GATE C
    g_c = True
    gw = gamma_wall()
    for carrier in ("electrons", "holes"):
        r = mos_vs_sp(carrier, gammas=(1.0, gw))
        sp, cl_ = r["SP"], r["classical"]
        rel = (lambda m, k: m[k] / sp[k] - 1.0)
        print("[bdg] GATE C {:9s}: SP sheet {:.4e} m^-2, centroid {:.3f} nm, peak {:.2f} nm (max c/N_c "
              "{:.2f}); classical sheet {:+.1%}, centroid {:+.1%}, peak AT the wall".format(
                  carrier, sp["sheet"], sp["centroid"] * 1e9, sp["peak"] * 1e9,
                  r["meta"]["max_c_over_Nc"], rel(cl_, "sheet"), rel(cl_, "centroid")), flush=True)
        for g in (1.0, gw):
            dg = r[("DG", g)]
            close = {k: abs(dg[k] - sp[k]) / abs(cl_[k] - sp[k]) for k in ("sheet", "centroid")}
            pk = dg["peak"] / sp["peak"]
            if g == 1.0:     # the Ancona-Iafrate value: must close >= 70 % of the quantum correction
                gc = bool(max(close.values()) <= 0.30 and 0.5 <= pk <= 1.5)
            else:            # gamma_wall (a priori): within 8 % of SP on sheet AND centroid
                gc = bool(abs(rel(dg, "sheet")) <= 0.08 and abs(rel(dg, "centroid")) <= 0.08
                          and 0.75 <= pk <= 1.25)
            gc = gc and r["meta"]["converged"] and cl_["peak"] == 0.0
            g_c = g_c and gc
            print("[bdg]   DG gamma = {:.4f}: sheet {:+.1%}, centroid {:+.1%} vs SP (residual / "
                  "classical gap {:.2f} / {:.2f}); peak {:.2f} nm = {:.2f} x SP -> {}".format(
                      g, rel(dg, "sheet"), rel(dg, "centroid"), close["sheet"], close["centroid"],
                      dg["peak"] * 1e9, pk, "PASS" if gc else "FAIL"), flush=True)
    ok = ok and g_c

    # GATE D
    g_d = True
    for carrier in ("electrons", "holes"):
        e = fixed_point(carrier)
        gd = bool(e < 2e-3)
        g_d = g_d and gd
        print("[bdg] GATE D {:9s}: DG density vs post-hoc DG BVP on its own potential: max {:.1e} "
              "of peak -> {}".format(carrier, e, "PASS" if gd else "FAIL"), flush=True)
    ok = ok and g_d

    # GATE E
    eq = inversion_equilibrium(quantum_srh=True)
    g_e = bool(eq["max_phi_n"] < 1e-8 and eq["max_phi_p"] < 1e-8 and eq["np_quantum"] < 1e-7
               and eq["np_bulk"] < 1e-6 and eq["inversion_peak_nm"] > 0.5
               and eq["classical_peak_nm"] == 0.0)
    eq0 = inversion_equilibrium(quantum_srh=False)
    ok = ok and g_e
    print("[bdg] GATE E: inversion, max|phi_n| = {:.1e} V, max|phi_p| = {:.1e} V, n p / (n_i^2 "
          "e^((Ln+Lp)/V_t)) - 1 <= {:.1e}, bulk n p/n_i^2 - 1 <= {:.1e}; Lambda_wall n/p = {:.1f} / "
          "{:.1f} V_t; inversion peak {:.2f} nm (classical {:.2f}) -> {}".format(
              eq["max_phi_n"], eq["max_phi_p"], eq["np_quantum"], eq["np_bulk"],
              eq["lam_wall_n_Vt"], eq["lam_wall_p_Vt"], eq["inversion_peak_nm"],
              eq["classical_peak_nm"], "PASS" if g_e else "FAIL"), flush=True)
    print("[bdg]   (quantum_srh=False: max|phi_n| = {:.1e} V, max|phi_p| = {:.1e} V, {:.1e} V off the wall "
          "node -- the classical SRH reference breaks detailed balance in the dark space)".format(
              eq0["max_phi_n"], eq0["max_phi_p"], eq0["max_phi_interior"]), flush=True)

    print("[bdg] *** D2d BIPOLAR DENSITY-GRADIENT: {} *** ({:.0f} s)".format(
        "PASS" if ok else "FAIL", time.time() - t0), flush=True)
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
