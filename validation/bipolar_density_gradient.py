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
        phi_p = psi - Lambda_p + V_t ln(p/n_i) are flat (|phi| < 1e-11 V), n p = n_i^2 exp((Ln+Lp)/V_t)
        at every node and n p = n_i^2 in the neutral bulk. With quantum_srh=False the classical SRH
        generates in the dark space and the QFLs are NOT flat (reported).
GATE F (transport): a 1-D Si p-n diode (1e23 / 1e22) at +0.4 V and -0.5 V: DG changes the terminal
        current by < 1e-2 relative and by no more than 2 max|Lambda|/V_t (first order), the change
        scales LINEARLY with gamma (ratio gamma=1 / gamma=0.5 within 2 +- 0.2), and the two terminal
        currents conserve to 1e-8 (both carriers' contacts read the DG flux).
GATE G (v0.15 4.4.3, strong inversion through physics_bipolar_dd.setup_quasi_equilibrium_carrier; its
        code lives here, it RUNS from validation/bipolar_dg_strong_inversion.py -- own nightly script):
        G1-3 every 50 mV step to 1.3 V in <= 12 Newton iterations (N_A 3e23 / 1e24, gamma 0.5 / 1 /
        gamma_wall, double and extended); a NEW Schrodinger-Poisson INVERSION oracle (sp_inversion:
        quantum electrons, classical holes, its own Trellakis Newton; self-checks: free-gas deficit and
        the hbar^2/m -> 0 classical limit vs the FD first integral) against which, with tolerances
        REGISTERED IN ADVANCE (SP_TOL_*), gamma_wall must give N_inv within 10 % and the centroid within
        15 % for N_inv >= 1e15 m^-2, gamma = 1 must close >= 60 % of the classical-SP gap, the classical
        error must be >= 3x the gamma_wall error and the threshold shift must agree within 20 mV (G1-4);
        quasi-static C-V (G1-5); the anisotropic-Si SP run is REPORTED only (decision 51). A miss of a
        registered tolerance is a MODEL RESULT: the gate fails and says so; nothing is retuned.

Run: python -m validation.bipolar_density_gradient
"""
import contextlib
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from dynameta.carriers.devsim_env import import_devsim  # noqa: E402
ds = import_devsim()
from dynameta.constants import HBAR, KB, M_E, Q_E, T_REF, V_T  # noqa: E402
from dynameta.carriers import eq_registry as _R  # noqa: E402
from dynameta.carriers import extended_precision as XP  # noqa: E402
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
    """XP.solve: the extended (mixed-precision) Newton -- every device of this module opts in (_bar,
    _diode: the comments there give the measured double-precision floors)."""
    with _quiet():
        XP.solve(type="dc", absolute_error=ABS_ERR, relative_error=rel, maximum_iterations=MAX_ITER)


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
        # EXTENDED PRECISION for every wall device (2026-11-03). In double the criteria below are
        # finer than the arithmetic: DEVSIM's relative update is |dx|/(|x| + 1e-10) (measured: it
        # reproduces the reported RelError node by node), so where the DG dark-space tail of Lambda
        # passes ~1e-17 V (GATE B, 11-16 nm from the wall) REL_ERR = 1e-7 demands |dLambda| < 1e-17 V,
        # below one ulp of the wall's Lambda = -0.71 V (1.1e-16): GATE B then sat at relative
        # 1.0-4.1e-7 for 100 iterations with every update at 1 ulp. And ABS_ERR = 1e10 m^-3 is 0.6 ulp
        # of GATE C's 7.5e25 / 8.2e25 accumulation peak, while the converged update floor measured
        # 3.9 / 2.1 ulp median, 36 / 8.4 max: the classical surface-potential steps waited 10-18
        # iterations at relative 1e-15 for a lucky update and one step ran out (100) on Elmer's
        # OpenBLAS. Pass/fail was decided by round-off, so it flipped with the BLAS. The mixed-precision
        # Newton (carriers.extended_precision) keeps both criteria as written and makes them
        # attainable: GATE B converges in 5/6/7 iterations, GATE C's classical steps in 6, its metrics
        # move <= 1e-15 against the double run, and GATE C costs 18 s instead of ~8 s.
        XP.enable(dev)
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
        # extended precision here too (2026-11-03): the forward current is a ~3e-9 difference of the
        # contact edges' drift and diffusion terms, so in double the total current density along ONE
        # converged state is constant only to 3.5e-7 and the kahan3 and two-Bernoulli spellings of
        # the same flux differ by 1.8e-8 / 6.2e-9 at the contact edges (measured). GATE A's 1e-9 then
        # held only while the DG at b = 0 repeated the classical arithmetic bit for bit, and GATE F's
        # I(gamma = 0.5) moved 6e-8 with the BLAS. In the extended Newton both are physics again.
        XP.enable(dev)
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
# GATE G (v0.15 item 4.4.3): strong inversion through the QUASI-EQUILIBRIUM closure
# (physics_bipolar_dd.setup_quasi_equilibrium_carrier). Shared by tests/test_quasi_equilibrium.py.
# ---------------------------------------------------------------------------------------------

QE_LEN = 300e-9                          # long enough that the 1.3 V depletion (75 nm at 3e23) is
                                         # followed by > 25 Debye lengths of neutral body


def _nested_mesh(mesh, nseg, beta, length, *, tags=("wall", "sub")):
    """1-D nodes x_k = L (e^{beta k/N} - 1)/(e^beta - 1), k = 0..N (last/first spacing e^beta, first
    spacing ~ L beta/(N (e^beta - 1)), spacing ~ (beta/N)(x + L/(e^beta - 1)) elsewhere): doubling N at
    fixed beta NESTS the meshes (every coarse node is a fine node; DEVSIM's graded mesh lines do not --
    their segment joints cost observed order). Returns the first spacing."""
    xs = length * np.expm1(beta * np.arange(nseg + 1) / nseg) / np.expm1(beta)
    gaps = np.diff(xs)
    for k, xk in enumerate(xs):
        ds.add_1d_mesh_line(mesh=mesh, pos=float(xk), ps=1.01 * float(gaps[min(k, nseg - 1)]),
                            **({"tag": tags[0]} if k == 0 else {"tag": tags[1]} if k == nseg else {}))
    return float(gaps[0])


def _mesh_2d(mesh, mesh_nb, length, lateral, pad=5e-9):
    """The 2-D twin of _nested_mesh (mos_qe(lateral=...)): every node of the nested 1-D x mesh is an x
    line (ps = its larger neighbouring gap, so DEVSIM adds none), `lateral` are the y lines; inert pad
    regions beyond both x ends carry the contacts (DEVSIM's 2-D contacts sit on an inter-region edge --
    the pattern of tests/test_trap_kinetics.py and validation/devsim_noise)."""
    nseg, beta = mesh_nb
    xs = length * np.expm1(beta * np.arange(nseg + 1) / nseg) / np.expm1(beta)
    xs = np.concatenate([[-pad], xs, [length + pad]])
    gaps = np.diff(xs)
    ds.create_2d_mesh(mesh=mesh)
    for k, xk in enumerate(xs):
        g = max(gaps[k - 1] if k > 0 else 0.0, gaps[k] if k < gaps.size else 0.0)
        ds.add_2d_mesh_line(mesh=mesh, dir="x", pos=float(xk), ps=1.01 * float(g))
    yl = np.asarray(lateral, dtype=float)
    yg = np.diff(yl)
    for k, yk in enumerate(yl):
        g = max(yg[k - 1] if k > 0 else 0.0, yg[k] if k < yg.size else 0.0)
        ds.add_2d_mesh_line(mesh=mesh, dir="y", pos=float(yk), ps=1.01 * float(g))
    # The pad boxes stop 0.2 pad SHORT of the semiconductor edge: DEVSIM assigns a triangle to a region box
    # with a tolerance, and a box ending exactly at x = 0 swallowed the first 0.05 nm semiconductor cell
    # (its contact then did not exist -- measured; any box edge >~ 1e-10 m away from that cell is safe).
    ds.add_2d_region(mesh=mesh, material="Si", region="s")
    ds.add_2d_region(mesh=mesh, material="SiO2", region="pad_w", xl=-pad, xh=-0.2 * pad)
    ds.add_2d_region(mesh=mesh, material="SiO2", region="pad_s", xl=length + 0.2 * pad, xh=length + pad)
    ds.add_2d_contact(mesh=mesh, name="wall", region="s", material="metal", xl=0.0, xh=0.0, bloat=1e-13)
    ds.add_2d_contact(mesh=mesh, name="sub", region="s", material="metal", xl=length, xh=length,
                      bloat=1e-13)


def qe_mesh(h0, length=QE_LEN, ratio=40.0):
    """(nseg, beta) of the nested mesh with first spacing ~h0 and last/first spacing `ratio`."""
    beta = float(np.log(ratio))
    return int(round(length * beta / (h0 * np.expm1(beta)))), beta


def mos_qe(N_A, deltas, *, fd=False, dg_gamma=None, mesh_nb=None, length=QE_LEN, ext=False,
           rel=1e-5, close=("electrons",), tau=1e-7, step=0.05, dg_holes=False, tag="q",
           closure=True, quantum_srh=True, probe=None, dg_at_target=False, lateral=None, v_ch=None,
           u_floor_rel=1e-6):
    """1-D p-type MOS (N_A), fixed surface potential psi_s = psi_b + delta at the wall, body ohmic
    'sub'. Classical solve at flat band; optionally the bipolar DG (electrons; holes too with dg_holes)
    at dg_gamma with setup_dg_insulator_wall, gamma-ramped at flat band; then the quasi-equilibrium
    closure on the carriers in `close` (closure=False: the full drift-diffusion rows), and a ramp of
    mesh_nb = (nseg, beta) of _nested_mesh (default qe_mesh(0.05 nm): first spacing 0.05 nm, ratio 40),
    or "p4": the graded mesh lines of the P4 1.1 diagnosis (_bar; with length=150e-9: 268 nodes);
    delta in <= `step` steps through `deltas` (increasing). Returns [dict per delta]: x, psi, n, p,
    lam_n (DG), Q (the semiconductor charge sum q (p - n - N_A) NodeVolume, C/m^2), Q_gauss (the wall
    contact charge, DEVSIM), its (Newton iterations of every ramp step since the previous delta), psi_b,
    and probe(device) (when given: called on the live device at each delta, e.g. a Jacobian SVD).
    dg_at_target (one delta only): ramp the CLASSICAL full model to delta first and install the DG (and
    then the closure) there -- the D2d GATE E order, which the FULL DG model needs (its own psi ramp
    fails on the way to 0.6 V: "Matrix solve failed", measured).
    lateral (v0.15 follow-up): a sequence of y positions [m] -> the SAME device as a 2-D tensor mesh
    (create_2d_mesh: the x nodes of the 1-D mesh times these y lines, right triangles), laterally uniform,
    wall and body contacts on the x = 0 / x = length edges against inert pad regions. Its records carry
    "cols" = {y: dict(x, n, p, psi[, lam_n])} (each lateral line sorted by x; the top-level arrays are
    the y = lateral[0] line) and Q / Q_gauss PER UNIT WIDTH.
    v_ch: the electron closure gets imref_shift = a device parameter "v_ch" of this value [V] (the
    channel-slice imref split phi_n = phi_body + V_ch).
    Raises the solver error of a failed step. ext: the extended-precision Newton (both criteria as
    written); abs_tol: dc_solve.default_abs_tol (plan 4.4.1)."""
    from dynameta.carriers.dc_solve import default_abs_tol
    dev, mesh = "qe_" + tag, "qe_m_" + tag
    _drop(dev, mesh)
    its = []

    def solve():
        with _quiet():
            info = (XP.solve if ext else ds.solve)(type="dc", absolute_error=default_abs_tol(dev),
                                                   relative_error=rel, maximum_iterations=40, info=True)
        its.append(len(info["iterations"]))
        if not info["converged"]:
            raise ds.error("Convergence failure! ({} iterations, mos_qe {})".format(its[-1], tag))

    try:
        with _quiet():
            if lateral is not None:
                _mesh_2d(mesh, mesh_nb or qe_mesh(0.05e-9, length), length, lateral)
            else:
                ds.create_1d_mesh(mesh=mesh)
            if lateral is not None:
                pass
            elif mesh_nb == "p4":            # the P4 1.1 diagnosis mesh (_bar's lines: 268 nodes at 150 nm)
                for pos, ps, tg in ((0.0, 0.05e-9, "wall"), (5e-9, 0.1e-9, None), (25e-9, 0.5e-9, None),
                                    (length, 2e-9, "sub")):
                    ds.add_1d_mesh_line(mesh=mesh, pos=pos, ps=ps, **({"tag": tg} if tg else {}))
            else:
                _nested_mesh(mesh, *(mesh_nb or qe_mesh(0.05e-9, length)), length)
            if lateral is None:
                ds.add_1d_contact(mesh=mesh, name="wall", tag="wall", material="metal")
                ds.add_1d_contact(mesh=mesh, name="sub", tag="sub", material="metal")
                ds.add_1d_region(mesh=mesh, material="Si", region="s", tag1="wall", tag2="sub")
            ds.finalize_mesh(mesh=mesh)
            ds.create_device(mesh=mesh, device=dev)
            if ext:
                XP.enable(dev)
            ds.node_model(device=dev, region="s", name="NetDoping", equation="{:.17e}".format(-N_A))
            BP.setup_bipolar_region(dev, "s", eps_static=EPS, n_dos_m3=nc_model(MN, GVN), n_i_m3=NI_SI,
                                    mobility_n_m2Vs=0.1, mobility_p_m2Vs=0.04, tau_n_s=tau, tau_p_s=tau,
                                    fd_enhancement=fd, n_dos_p_m3=nc_model(MP, GVP))
            BP.setup_equilibrium_seed_models(dev, "s")
            BP.setup_contact_ohmic_bipolar(dev, "sub")
            for nm, src in (("Electrons", "IntrinsicElectrons"), ("Holes", "IntrinsicHoles")):
                ds.set_node_values(device=dev, region="s", name=nm, init_from=src)
            psi_b = BP.ohmic_built_in_offset_V(-N_A, n_i_m3=NI_SI, n_dos_m3=nc_model(MN, GVN),
                                               n_dos_p_m3=nc_model(MP, GVP), fd_enhancement=fd)
            ds.set_parameter(device=dev, name="wall_bias", value=psi_b)
            ds.contact_node_model(device=dev, contact="wall", name="wall_psi",
                                  equation="Potential - wall_bias")
            ds.contact_node_model(device=dev, contact="wall", name="wall_psi:Potential", equation="1")
            _R.record_contact_equation(dev, "wall", name="PotentialEquation", node_model="wall_psi",
                                       edge_charge_model="PotentialEdgeFlux")
            nn = _get(dev, "s", "x").size
            ds.set_node_values(device=dev, region="s", name="Potential", values=[psi_b] * nn)
        solve()
        def install():
            if dg_gamma is not None:
                with _quiet():
                    DG.setup_bipolar_dg(dev, "s", m_eff_n_kg=MN, m_eff_p_kg=MP if dg_holes else None,
                                        gamma_n=dg_gamma, gamma_p=dg_gamma, quantum_srh=quantum_srh)
                    DG.setup_contact_bipolar_dg(dev, "sub")
                    DG.setup_dg_insulator_wall(dev, "wall", u_floor_rel=u_floor_rel)
                for k, fr in enumerate((0.1, 0.3, 1.0)):
                    DG.set_dg_gamma(dev, "s", fr)
                    if k == 0:
                        DG.seed_bipolar_dg(dev, "s", walls=("wall",), u_floor_rel=u_floor_rel)
                    solve()
            if closure:
                if v_ch is not None:
                    ds.set_parameter(device=dev, name="v_ch", value=float(v_ch))
                for c in close:
                    BP.setup_quasi_equilibrium_carrier(
                        dev, "s", c, contact="sub",
                        imref_shift="v_ch" if (v_ch is not None and c == "electrons") else None)
                solve()

        if dg_at_target and len(deltas) != 1:
            raise ValueError("mos_qe: dg_at_target needs a single delta")
        if not dg_at_target:
            install()
        out, d_now = [], 0.0
        for target in deltas:
            its.clear()
            while d_now < target - 1e-12:
                d_now = min(target, d_now + step)
                ds.set_parameter(device=dev, name="wall_bias", value=psi_b + d_now)
                solve()
            if dg_at_target:
                install()
            x = _get(dev, "s", "x")
            width = 1.0
            if lateral is None:
                o = np.argsort(x)
                interior = np.ones(x.size, bool)
                interior[[o[0], o[-1]]] = False
            else:
                y = _get(dev, "s", "y")
                width = float(lateral[-1] - lateral[0])
                cols = {float(yc): np.flatnonzero(np.abs(y - yc) <= 1e-6 * width) for yc in lateral}
                cols = {yc: ix[np.argsort(x[ix])] for yc, ix in cols.items()}
                o = cols[float(lateral[0])]
                interior = (x > x.min() + 1e-6 * length) & (x < x.max() - 1e-6 * length)
            nn_, pp_ = _get(dev, "s", "Electrons"), _get(dev, "s", "Holes")
            rho_v = Q_E * (pp_ - nn_ - N_A) * _get(dev, "s", "NodeVolume")
            n, p = nn_[o], pp_[o]
            rec = {"delta": target, "x": x[o], "psi": _get(dev, "s", "Potential")[o], "n": n, "p": p,
                   "Q": float(np.sum(rho_v)) / width, "Q_interior": float(np.sum(rho_v[interior])) / width,
                   "its": list(its), "psi_b": psi_b,
                   "Q_gauss": {c: float(ds.get_contact_charge(device=dev, contact=c,
                                                              equation="PotentialEquation")) / width
                               for c in ("wall", "sub")}}
            if dg_gamma is not None:
                rec["lam_n"] = _get(dev, "s", "QLambda")[o]
            if lateral is not None:
                fields = [("n", "Electrons"), ("p", "Holes"), ("psi", "Potential")]
                if dg_gamma is not None:
                    fields.append(("lam_n", "QLambda"))
                vals = {k: _get(dev, "s", v) for k, v in fields}
                rec["cols"] = {yc: dict(x=x[ix], **{k: a[ix] for k, a in vals.items()})
                               for yc, ix in cols.items()}
            if probe is not None:
                rec["probe"] = probe(dev)
            out.append(rec)
        return out
    finally:
        _drop(dev, mesh)


def mos_sheet(r):
    """(N_inv [m^-2], centroid [m]) of the electrons of a mos_qe record: the excess over the bulk
    minority density, by the trapezoid rule on the record's own nodes."""
    x, n = r["x"], r["n"] - r["n"][-1]
    s = float(trapz(n, x))
    return s, float(trapz(x * n, x)) / s


def _fd_series(eta, order):
    """Normalized complete Fermi-Dirac integral F_j(eta) = sum_k (-1)^(k+1) e^(k eta) / k^(j+1) (the
    alternating series; d F_j/d eta = F_(j-1)); used only for eta <= -1 (e^eta <= 0.37: 60 terms)."""
    eta = np.asarray(eta, dtype=np.float64)
    if np.any(eta > -1.0):
        raise ValueError("_fd_series: eta <= -1 required (got max {:.3g})".format(float(np.max(eta))))
    k = np.arange(1, 61, dtype=np.float64)
    w = np.exp(eta[..., None] * k)
    return np.sum(np.where(k % 2.0 == 1.0, 1.0, -1.0) * w / k ** (order + 1.0), axis=-1)


def _fd_quad(eta, order):
    """F_j(eta) = (1/Gamma(j+1)) int_0^inf e^j/(1 + exp(e - eta)) de, any eta (scipy quad)."""
    from scipy.integrate import quad
    from scipy.special import gamma as _gam
    return quad(lambda e: e ** order / (1.0 + np.exp(min(e - eta, 700.0))), 0.0,
                max(eta, 0.0) + 60.0, limit=400, epsabs=0.0, epsrel=1e-13)[0] / _gam(order + 1.0)


def sp_inversion(N_A, delta_s, *, valleys=None, length=200e-9, h=0.05e-9, box=30e-9, n_states=240,
                 tol_V=1e-9, max_outer=200, m_scale=1.0, boltzmann=False, fixed_delta=None):
    """NEW independent Schrodinger-Poisson INVERSION oracle (v0.15 4.4.3, plan G1-4): p-type Si (N_A),
    surface band bending delta_s at an infinite wall (z = 0), neutral body at z = length (delta = 0).
    Electrons: subbands of the 1-D effective-mass Schrodinger equation (carriers.schrodinger_poisson
    .SchrodingerPoisson1D.solve_schrodinger, Dirichlet at the wall and at z = box) filled with the 2-D
    Fermi-Dirac sheet density (the body Fermi level), classical F_1/2 outside the box. Holes: CLASSICAL,
    p = N_v F_1/2(eta_p,b - delta/V_t) (exact series). Poisson: its own Trellakis predictor-corrector
    Newton (subband floors shift rigidly with the local potential; banded Jacobian), not the solver's.
    The electron model is the DG's: isotropic m_n = MN, g_v = GVN, so the classical limit has the DG's
    N_c = nc_model(MN, GVN) and the body electron density n_b = n_i^2/p_b of the DD.
    valleys: None (that isotropic model) or [(m_z, m_d, g_v), ...] (REPORTED anisotropic Si runs).
    m_scale: multiply every quantization mass by m_scale and the 2-D prefactor by 1/sqrt(m_scale),
    which keeps N_c and sends hbar^2/m -> 0: the CLASSICAL limit (self-check).
    boltzmann (2026-11-07 audit of the GATE G N_inv miss): Boltzmann statistics everywhere -- subband
    occupation e^x instead of ln(1 + e^x), holes N_v e^eta, classical electrons N_c e^eta -- the oracle
    for a comparison against the Boltzmann (fd=False) DG, which removes every Fermi-Dirac approximation
    from both sides. fixed_delta: a band-bending profile on the oracle's z grid; Poisson is skipped and
    the electrons are filled in that frozen potential (the Airy / triangular-well cross-check).
    Returns dict(z, n, p, delta, N_inv (excess over n_b, m^-2), centroid (m), converged, top_state_kT,
    subbands = [(E_i - E_F [J], sheet density [m^-2]), ...] per valley set)."""
    from scipy.linalg import solve_banded
    from dynameta.carriers.schrodinger_poisson import SchrodingerPoisson1D
    kT = KB * T_REF
    vt = kT / Q_E
    eps = EPS * 8.8541878128e-12
    z = np.arange(0.0, length + 0.5 * h, h)
    nz = z.size
    nb_box = int(round(box / h))
    nc, nv = nc_model(MN, GVN), nc_model(MP, GVP)
    p_b = 1.0 + 0.5 * (N_A + np.sqrt(N_A ** 2 + 4.0 * NI_SI ** 2))     # the DD's CHOLE
    n_b = NI_SI ** 2 / p_b
    from scipy.optimize import brentq
    if boltzmann:
        eta_pb = float(np.log(p_b / nv))
    else:
        eta_pb = brentq(lambda e: nv * _fd_series(np.array([e]), 0.5)[0] - p_b, -60.0, -1.0, xtol=1e-15)
    eta_nb = float(np.log(n_b / nc))                                     # e^-38: F_1/2 = e^eta exactly
    vals = [(MN, MN, GVN)] if valleys is None else list(valleys)
    sps = [(SchrodingerPoisson1D(z[:nb_box + 1], mz * m_scale, T_K=T_REF, g_s=2, g_v=1), md, gv)
           for mz, md, gv in vals]

    def holes(d):
        e = eta_pb - d / vt
        if boltzmann:
            return nv * np.exp(e), -nv / vt * np.exp(e)
        return nv * _fd_series(e, 0.5), -nv / vt * _fd_series(e, -0.5)

    def elec_classical(d):
        e = eta_nb + d / vt
        if boltzmann:
            return nc * np.exp(e), nc / vt * np.exp(e)
        if np.any(e > -1.0):
            f = np.array([_fd_quad(x, 0.5) for x in e])
            fm = np.array([_fd_quad(x, -0.5) for x in e])
        else:
            f, fm = _fd_series(e, 0.5), _fd_series(e, -0.5)
        return nc * f, nc / vt * fm

    def subbands(d):
        U = -kT * eta_nb - Q_E * d[:nb_box + 1]                          # E_c - E_F [J], E_F = 0
        out = []
        top = np.inf
        for sp, md, gv in sps:
            k = int(n_states)
            while True:                    # completeness: the top solved state > 15 kT above E_F
                E, psi, _ = sp.solve_schrodinger(U, n_states=min(k, nb_box - 1))
                if E[-1] / kT > 15.0 or k >= nb_box - 1:
                    break
                k *= 2
            pref = 2.0 * gv * md * kT / (2.0 * np.pi * HBAR ** 2) / np.sqrt(m_scale)
            out.append((E, psi ** 2, pref))
            top = min(top, (E[-1]) / kT)
        return out, top

    def elec(d, sb, d_k):
        n, dn = np.zeros(nz), np.zeros(nz)
        n[nb_box + 1:], dn[nb_box + 1:] = elec_classical(d[nb_box + 1:])   # classical beyond the box
        inner = slice(1, nb_box)
        shift = Q_E * (d[inner] - d_k[inner])[:, None]                  # rigid floor shift (predictor)
        for E, w, pref in sb:
            x = (-E[None, :] + shift) / kT
            if boltzmann:
                occ = fer = np.exp(np.minimum(x, 700.0))
            else:
                occ = np.where(x > 40.0, x, np.log1p(np.exp(np.minimum(x, 40.0))))
                fer = 1.0 / (1.0 + np.exp(-np.clip(x, -700.0, 700.0)))
            n[inner] += np.sum(w * occ, axis=1) * pref
            dn[inner] += np.sum(w * fer, axis=1) * pref * Q_E / kT
        return n, dn

    # initial guess: the depletion approximation
    W = np.sqrt(2.0 * eps * max(delta_s, 0.05) / (Q_E * N_A))
    d = np.where(z < W, delta_s * (1.0 - z / W) ** 2, 0.0)
    d[0], d[-1] = delta_s, 0.0
    converged, top = False, np.nan
    if fixed_delta is not None:
        d = np.asarray(fixed_delta, dtype=np.float64).copy()
        max_outer, converged = 0, True
    for _ in range(max_outer):
        sb, top = subbands(d)
        d_k = d.copy()
        for _inner in range(100):
            n, dn = elec(d, sb, d_k)
            p, dp = holes(d)
            r = eps * (d[:-2] - 2.0 * d[1:-1] + d[2:]) / h ** 2 + Q_E * (p - n - N_A)[1:-1]
            ab = np.zeros((3, nz - 2))
            ab[0, 1:] = eps / h ** 2
            ab[2, :-1] = eps / h ** 2
            ab[1, :] = -2.0 * eps / h ** 2 + Q_E * (dp - dn)[1:-1]
            step = solve_banded((1, 1), ab, -r)
            step = np.clip(step, -2.0 * vt, 2.0 * vt)              # damped Newton: <= 2 V_t per step
            d[1:-1] += step
            if np.max(np.abs(step)) < 1e-13:
                break
        if np.max(np.abs(d - d_k)) < tol_V:
            converged = True
            break
    sb, top = subbands(d)
    n, _ = elec(d, sb, d)
    p, _ = holes(d)
    ex = n - n_b
    s = float(trapz(ex, z))
    sheets = []
    for E, _w, pref in sb:
        x = -E / kT
        occ = np.exp(np.minimum(x, 700.0)) if boltzmann else np.where(
            x > 40.0, x, np.log1p(np.exp(np.minimum(x, 40.0))))
        sheets.append((E, pref * occ))
    return {"z": z, "n": n, "p": p, "delta": d, "N_inv": s, "centroid": float(trapz(z * ex, z)) / s,
            "converged": converged, "top_state_kT": float(top), "subbands": sheets}


def classical_fd_surface_charge(N_A, delta_s):
    """|Q_s| [C/m^2] of the CLASSICAL Fermi-Dirac MOS from Poisson's first integral (the SP oracle's
    classical-limit reference): E_s^2 = (2 q/eps) int_0^delta_s [(n - n_b) - (p - p_b)] dd with
    n = N_c F_1/2(eta_nb + d/V_t), p = N_v F_1/2(eta_pb - d/V_t), integrated in closed form through
    F_3/2 (d F_3/2/d eta = F_1/2)."""
    from scipy.optimize import brentq
    vt = KB * T_REF / Q_E
    eps = EPS * 8.8541878128e-12
    nc, nv = nc_model(MN, GVN), nc_model(MP, GVP)
    p_b = 1.0 + 0.5 * (N_A + np.sqrt(N_A ** 2 + 4.0 * NI_SI ** 2))
    n_b = NI_SI ** 2 / p_b
    eta_pb = brentq(lambda e: nv * _fd_series(np.array([e]), 0.5)[0] - p_b, -60.0, -1.0, xtol=1e-15)
    eta_nb = float(np.log(n_b / nc))
    i_n = vt * nc * (_fd_quad(eta_nb + delta_s / vt, 1.5) - _fd_quad(eta_nb, 1.5)) - n_b * delta_s
    i_p = -vt * nv * (_fd_series(np.array([eta_pb - delta_s / vt]), 1.5)[0]
                      - _fd_series(np.array([eta_pb]), 1.5)[0]) - p_b * delta_s
    return float(np.sqrt(2.0 * Q_E / eps * (i_n - i_p)) * eps)


# REGISTERED IN ADVANCE (plan G1-4, before any measurement; a miss is a MODEL RESULT, never retuned):
SP_TOL_NINV, SP_TOL_CENTROID = 0.10, 0.15           # at gamma_wall, for N_inv >= 1e15 m^-2
SP_GAP_CLOSED_GAMMA1 = 0.60                         # gamma = 1 closes >= 60 % of the classical-SP gap
SP_DVTH_MV = 20.0                                   # threshold-shift agreement DG vs SP [mV]
CV_TOL_CINV, CV_TOL_VTR_MV = 0.05, 20.0             # quasi-static C-V: C_inv/C_ox, transition voltage
GATE_G_DELTAS = tuple(np.round(np.arange(0.85, 1.3001, 0.05), 10))
# MEASURED 2026-11-06 (after registration): the gamma_wall N_inv excess over SP is +10.6..+11.7 % -- the
# registered 10 % is MISSED, a model result (gate_g). The characterization pin (measured range +- 1 point):
# (v0.15.0 default statistics 'fermi_dirac', MEASURED 2026-11-07: +10.5..+11.1 %, inside the pin; the
# 2026-11-06 range above is the 'legacy' statistics. The pin is NOT re-centred: that is a decision.)
G14_NINV_PIN = (0.096, 0.127)


def _vth(deltas, sheets, target=1e16):
    """psi_s - psi_b [V] at which N_inv = target (log-linear interpolation)."""
    return float(np.interp(np.log(target), np.log(np.asarray(sheets)), np.asarray(deltas)))


def strong_inversion_vs_sp(N_A=3.0e23, deltas=GATE_G_DELTAS):
    """GATE G numbers (plan G1-4/G1-5): N_inv, centroid and Q_s of the closed classical, DG gamma = 1
    and DG gamma_wall MOS (Fermi-Dirac DD, electrons DG + closed, holes classical) and of the new SP
    inversion oracle at each band bending."""
    gw = gamma_wall()
    sp = {d: sp_inversion(N_A, d) for d in deltas}
    runs = {"classical": mos_qe(N_A, deltas, fd=True, tag="gg_c"),
            1.0: mos_qe(N_A, deltas, fd=True, dg_gamma=1.0, tag="gg_1"),
            gw: mos_qe(N_A, deltas, fd=True, dg_gamma=gw, tag="gg_w")}
    out = {"deltas": deltas, "gw": gw, "SP": [(sp[d]["N_inv"], sp[d]["centroid"],
                                                Q_E * float(trapz(sp[d]["p"] - sp[d]["n"] - N_A, sp[d]["z"])),
                                                sp[d]["converged"]) for d in deltas]}
    for k, rr in runs.items():
        out[k] = [mos_sheet(r) + (r["Q"],) for r in rr]
    return out


def _cv(deltas, q, t_ox):
    """Quasi-static C-V of a series oxide: V_g = psi_s + |Q_s|/C_ox (1-D Gauss), C = (1/C_ox + 1/C_s)^-1,
    C_s = d|Q_s|/d psi_s by central differences on the delta grid. Returns (V_g, C/C_ox) at the
    interior points."""
    c_ox = 3.9 * 8.8541878128e-12 / t_ox
    d, qa = np.asarray(deltas), np.abs(np.asarray(q))
    cs = (qa[2:] - qa[:-2]) / (d[2:] - d[:-2])
    return d[1:-1] + qa[1:-1] / c_ox, 1.0 / (1.0 + c_ox / cs)


def gate_g_convergence():
    """G1-3: every 50 mV step to 1.3 V in <= 12 Newton iterations, double (REL 1e-5) and extended
    (REL 1e-7), N_A = 3e23 and 1e24, gamma in {0.5, 1, gamma_wall}, bipolar DG on both carriers with
    BOTH closed; returns {(N_A, gamma, prec): (max its, final N_inv)} and the double-vs-extended max
    relative density difference per case."""
    deltas = tuple(np.round(np.arange(0.05, 1.3001, 0.05), 10))
    res, diff = {}, {}
    for N_A in (3.0e23, 1.0e24):
        for g in (0.5, 1.0, gamma_wall()):
            arr = {}
            for ext in (False, True):
                r = mos_qe(N_A, deltas, dg_gamma=g, dg_holes=True, close=("electrons", "holes"),
                           ext=ext, rel=1e-7 if ext else 1e-5, tag="gc{}{}".format(int(ext), len(res)))
                res[(N_A, g, ext)] = (max(max(x["its"]) for x in r), mos_sheet(r[-1])[0])
                arr[ext] = r[-1]
            diff[(N_A, g)] = max(float(np.max(np.abs(arr[True][k] / arr[False][k] - 1.0)))
                                 for k in ("n", "p"))
    return res, diff


def sp_airy_check(F=1e8, delta_s=1.1, h=0.05e-9, N_A=3.0e23, m_q=None):
    """SP oracle cross-check against the TRIANGULAR WELL (2026-11-07 audit of the GATE G N_inv miss):
    a frozen linear band bending delta(z) = max(delta_s - F z, 0) at the infinite wall, Fermi level at the
    body. Exact (infinite triangular well): E_i - E_c(0) = (hbar^2/(2 m))^(1/3) (q F)^(2/3) |a_i|, a_i the
    zeros of Ai (scipy.special.ai_zeros); sheet density sum_i (g_s g_v m kT/(2 pi hbar^2))
    ln(1 + exp(-(E_i - E_F)/kT)) -- the 2-D Fermi-Dirac sum written here from the standard 2-D DOS, not
    from the oracle. m_q: the quantization mass the ORACLE uses (default MN; a refutation knob). Returns
    (max relative error of the three lowest E_i - E_c(0), relative error of the total sheet density)."""
    from scipy.special import ai_zeros
    kT = KB * T_REF
    z = np.arange(0.0, 200e-9 + 0.5 * h, h)
    vals = None if m_q is None else [(m_q, MN, GVN)]
    r = sp_inversion(N_A, delta_s, fixed_delta=np.maximum(delta_s - F * z, 0.0), h=h, valleys=vals)
    E, ns = r["subbands"][0]
    p_b = 1.0 + 0.5 * (N_A + np.sqrt(N_A ** 2 + 4.0 * NI_SI ** 2))
    e0 = -kT * np.log(NI_SI ** 2 / p_b / nc_model(MN, GVN)) - Q_E * delta_s       # E_c(0) - E_F
    ea = e0 + (HBAR ** 2 / (2.0 * MN)) ** (1.0 / 3.0) * (Q_E * F) ** (2.0 / 3.0) * (-ai_zeros(4000)[0])
    x = -ea / kT
    ns_a = float(np.sum(2.0 * GVN * MN * kT / (2.0 * np.pi * HBAR ** 2)
                        * np.where(x > 40.0, x, np.log1p(np.exp(np.minimum(x, 40.0))))))
    return float(np.max(np.abs((E[:3] - ea[:3]) / (ea[:3] - e0)))), float(np.sum(ns)) / ns_a - 1.0


def gate_g():
    """GATE G (v0.15 4.4.3): strong inversion through the quasi-equilibrium closure. Printed and
    gated: G1-3 (convergence matrix), the SP oracle's self-checks, G1-4 (registered fidelity
    tolerances: a miss is reported as a MODEL RESULT and FAILS the gate -- never retuned), G1-5
    (quasi-static C-V vs SP), the anisotropic-Si SP run (reported only, decision 51)."""
    ok = True
    res, diff = gate_g_convergence()
    g3 = all(v[0] <= 12 for v in res.values())
    for (N_A, g, ext), (its, ninv) in sorted(res.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2])):
        print("[bdg] GATE G1-3 N_A {:.0e} gamma {:.3f} {}: max {} Newton its per 50 mV step to 1.3 V, "
              "N_inv(1.3 V) = {:.3e} m^-2".format(N_A, g, "ext" if ext else "dbl", its, ninv), flush=True)
    print("[bdg] GATE G1-3 double vs extended at 1.3 V: max rel density difference {:.1e} -> {}".format(
        max(diff.values()), "PASS" if g3 else "FAIL"), flush=True)
    ok = ok and g3

    # SP oracle self-checks: free gas (existing) + the classical limit (hbar^2/m -> 0, N_c kept)
    qm = sp_free_gas_deficit()
    cl = []
    for d in (0.6, 0.9):
        rc = sp_inversion(3e23, d, m_scale=4e4, h=0.005e-9, box=10e-9, length=150e-9)
        qc = Q_E * float(trapz(rc["p"] - rc["n"] - 3e23, rc["z"]))
        cl.append(abs(qc) / classical_fd_surface_charge(3e23, d) - 1.0)
    g_self = bool(abs(qm / (np.sqrt(np.pi) / (2.0 * np.sqrt(2.0))) - 1.0) < 1e-3 and max(map(abs, cl)) < 1e-3)
    print("[bdg] GATE G SP self-checks: free-gas deficit {:.6f} (exact 0.626657); classical limit "
          "(m x 4e4, N_c kept) vs the FD first integral at 0.6 / 0.9 V: {:+.1e} / {:+.1e} -> {}".format(
              qm, cl[0], cl[1], "PASS" if g_self else "FAIL"), flush=True)
    ok = ok and g_self
    # the triangular-well (Airy) limit, the 2026-11-07 audit: subband energies and the 2-D FD sheet sum.
    # Budget: the 3-point Schrodinger stencil is O(h^2): MEASURED 1.8e-4 / 4.5e-5 (energies, max of three) and
    # +7.8e-4 / +1.9e-4 (sheet) at h = 0.05 / 0.025 nm, F = 1e8 V/m -> 4e-4 / 2e-3 at the oracle's h.
    ea, es = sp_airy_check()
    g_airy = bool(ea <= 4e-4 and abs(es) <= 2e-3)
    print("[bdg] GATE G SP vs the Airy triangular well (F = 1e8 V/m): lowest three subbands {:.1e}, sheet "
          "{:+.1e} -> {}".format(ea, es, "PASS" if g_airy else "FAIL"), flush=True)
    ok = ok and g_airy

    r = strong_inversion_vs_sp()
    gw, dl = r["gw"], r["deltas"]
    g4, miss, excess = True, [], []
    for i, d in enumerate(dl):
        n_sp, c_sp, _, conv = r["SP"][i]
        e = {k: (r[k][i][0] / n_sp - 1.0, r[k][i][1] / c_sp - 1.0) for k in ("classical", 1.0, gw)}
        close = max(abs(e[1.0][j]) / abs(e["classical"][j]) for j in (0, 1))
        cls3 = all(abs(e["classical"][j]) >= 3.0 * abs(e[gw][j]) for j in (0, 1))
        gated = n_sp >= 1e15
        ninv_ok = (not gated) or abs(e[gw][0]) <= SP_TOL_NINV
        if gated:
            excess.append(e[gw][0])
            if not ninv_ok:
                miss.append(d)
        gi = bool(conv and ((not gated) or abs(e[gw][1]) <= SP_TOL_CENTROID)
                  and close <= 1.0 - SP_GAP_CLOSED_GAMMA1 and cls3)
        g4 = g4 and gi
        print("[bdg] GATE G1-4 {:.2f} V: SP N_inv {:.3e} m^-2, centroid {:.3f} nm | gamma_wall {:+.1%} / "
              "{:+.1%} | gamma=1 {:+.1%} / {:+.1%} (residual/gap {:.2f}) | classical {:+.1%} / {:+.1%} -> {}{}"
              .format(d, n_sp, c_sp * 1e9, e[gw][0], e[gw][1], e[1.0][0], e[1.0][1], close,
                      e["classical"][0], e["classical"][1], "PASS" if gi else "FAIL",
                      "" if ninv_ok else " (N_inv beyond the registered {:.0%})".format(SP_TOL_NINV)),
              flush=True)
    vth = {k: _vth(dl, [x[0] for x in r[k]]) for k in ("classical", 1.0, gw)}
    vth_sp = _vth(dl, [x[0] for x in r["SP"]])
    dvth = {k: 1e3 * (vth[k] - vth["classical"]) for k in (1.0, gw)}
    dvth_sp = 1e3 * (vth_sp - vth["classical"])
    g_vth = abs(dvth[gw] - dvth_sp) <= SP_DVTH_MV
    print("[bdg] GATE G1-4 threshold shift (psi_s at N_inv = 1e16 m^-2, vs classical): SP {:+.1f} mV, "
          "DG gamma_wall {:+.1f} mV, gamma=1 {:+.1f} mV -> {}".format(
              dvth_sp, dvth[gw], dvth[1.0], "PASS" if g_vth else "FAIL"), flush=True)
    g4 = g4 and g_vth
    # The registered N_inv tolerance: its verdict is printed as such and is a MODEL RESULT when missed
    # (plan P4 1.8: "a documented model result ... not something to retune"; MEASURED 2026-11-06:
    # +10.6..+11.7 % for N_inv >= 1e15 m^-2, mesh-converged to 0.06 % and SP-converged to 1e-4 -- the
    # first-order DG at the a-priori gamma_wall over-fills the triangular inversion well by ~11 %). It
    # is PINNED as a characterization -- the excess must stay inside G14_NINV_PIN, the measured range
    # +- 1 percentage point (20x the measured discretization) -- so that any change of the model or of
    # the closure that moves it cannot land silently; widening the pin, or tightening it after a model
    # change, is a decision with a new measurement, never a retune.
    pin = bool(excess and all(G14_NINV_PIN[0] <= x <= G14_NINV_PIN[1] for x in excess))
    print("[bdg] GATE G1-4 registered N_inv tolerance ({:.0%} at gamma_wall, N_inv >= 1e15): {} -- {}; "
          "characterization pin {:+.1%}..{:+.1%} -> {}".format(
              SP_TOL_NINV, "MET" if not miss else "MISSED at {}".format(
                  ", ".join("{:.2f} V".format(d) for d in miss)),
              "a MODEL RESULT, reported, not retuned" if miss else "registered expectation held",
              G14_NINV_PIN[0], G14_NINV_PIN[1], "PASS" if pin else "FAIL"), flush=True)
    g4 = g4 and pin
    print("[bdg] GATE G1-4 (centroid within {:.0%}, gap closed >= {:.0%}, classical >= 3x, dV_th within "
          "{:.0f} mV, the N_inv pin): {}".format(SP_TOL_CENTROID, SP_GAP_CLOSED_GAMMA1, SP_DVTH_MV,
                                                  "PASS" if g4 else "FAIL"), flush=True)
    ok = ok and g4

    # REPORTED (2026-11-07 audit of the N_inv miss): the same comparison with BOLTZMANN statistics on both
    # sides (fd=False DG, sp_inversion(boltzmann=True)) -- no Fermi-Dirac approximation anywhere. The
    # excess persists and grows (+12.0 % at 1.0 V .. +16.5 % at 1.3 V, measured), so it is the DG model at
    # the free-gas gamma_wall in a field-confined well, not the FD g-fit (which moves the classical N_inv by
    # only -0.35 .. -0.89 %, the opposite sign; audit note).
    for d in (1.0, 1.3):
        spb = sp_inversion(3e23, d, boltzmann=True)
        dgb = mos_sheet(mos_qe(3e23, (d,), fd=False, dg_gamma=gw, tag="gb")[0])
        print("[bdg] GATE G (reported) Boltzmann on both sides, {:.2f} V: gamma_wall N_inv {:+.1%}, centroid "
              "{:+.1%} vs SP".format(d, dgb[0] / spb["N_inv"] - 1.0, dgb[1] / spb["centroid"] - 1.0),
              flush=True)

    # G1-7 (1-D): mesh convergence of the DG gamma_wall N_inv and centroid at 1.2 V, nested meshes
    sh = [mos_sheet(mos_qe(3e23, (1.2,), fd=True, dg_gamma=gw, mesh_nb=qe_mesh(h), tag="g17_%d" % k)[0])
          for k, h in enumerate((0.1e-9, 0.05e-9, 0.025e-9, 0.0125e-9))]
    orders = [[np.log2(abs(sh[k][j] - sh[k + 1][j]) / abs(sh[k + 1][j] - sh[k + 2][j])) for k in range(2)]
              for j in (0, 1)]
    g7 = bool(min(min(o) for o in orders) >= 1.7)
    print("[bdg] GATE G1-7 mesh convergence (DG gamma_wall, 1.2 V, h0 0.1..0.0125 nm): N_inv orders {:.2f} / "
          "{:.2f}, centroid orders {:.2f} / {:.2f}; finest-mesh change {:.1e} / {:.1e} -> {}".format(
              orders[0][0], orders[0][1], orders[1][0], orders[1][1], abs(sh[3][0] / sh[2][0] - 1.0),
              abs(sh[3][1] / sh[2][1] - 1.0), "PASS" if g7 else "FAIL"), flush=True)
    ok = ok and g7

    g5 = True
    for t_ox in (1.5e-9, 3e-9):
        vg_sp, c_sp_cv = _cv(dl, [x[2] for x in r["SP"]], t_ox)
        vg_dg, c_dg_cv = _cv(dl, [x[2] for x in r[gw]], t_ox)
        cinv = c_dg_cv[-1] / c_sp_cv[-1] - 1.0
        vtr = lambda vg, c: float(np.interp(0.5 * (c.min() + c.max()), c, vg))   # noqa: E731
        dvtr = 1e3 * (vtr(vg_dg, c_dg_cv) - vtr(vg_sp, c_sp_cv))
        gi = bool(abs(cinv) <= CV_TOL_CINV and abs(dvtr) <= CV_TOL_VTR_MV)
        g5 = g5 and gi
        print("[bdg] GATE G1-5 quasi-static C-V, t_ox = {:.1f} nm: C(1.25 V)/C_ox DG gamma_wall {:.4f} vs "
              "SP {:.4f} ({:+.1%}); mid-transition V_g DG - SP = {:+.1f} mV -> {}".format(
                  t_ox * 1e9, c_dg_cv[-1], c_sp_cv[-1], cinv, dvtr, "PASS" if gi else "FAIL"), flush=True)
    ok = ok and g5

    # anisotropic Si (decision 51: REPORTED, not gated): (m_z, m_d, g_v) = 2 x (m_l, m_t) + 4 x (m_t, sqrt(m_l m_t))
    ml, mt = 0.916 * M_E, 0.19 * M_E
    an = sp_inversion(3e23, 1.2, valleys=[(ml, mt, 2), (mt, np.sqrt(ml * mt), 4)])
    iso = r["SP"][list(dl).index(1.2)]
    print("[bdg] GATE G (reported) anisotropic Si SP at 1.2 V: N_inv {:.3e} m^-2 ({:+.1%} vs the isotropic "
          "model SP), centroid {:.3f} nm ({:+.1%}); the isotropic model is what the DG represents".format(
              an["N_inv"], an["N_inv"] / iso[0] - 1.0, an["centroid"] * 1e9, an["centroid"] / iso[1] - 1.0),
          flush=True)
    return ok


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
    g_e = bool(eq["max_phi_n"] < 1e-11 and eq["max_phi_p"] < 1e-11 and eq["np_quantum"] < 1e-10
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

    # GATE G (v0.15 4.4.3, ~8-11 min) runs as its own script: validation/bipolar_dg_strong_inversion.py

    print("[bdg] *** D2d BIPOLAR DENSITY-GRADIENT: {} *** ({:.0f} s)".format(
        "PASS" if ok else "FAIL", time.time() - t0), flush=True)
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
