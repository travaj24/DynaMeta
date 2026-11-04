"""
Validate the opt-in EXTENDED-PRECISION DEVSIM solves (dynameta.carriers.extended_precision, D2b)
against oracles that are INDEPENDENT of the code under test: closed-form diode physics (Shockley
ideal diode with finite neutral widths + the Sah-Noyce-Shockley depletion-region SRH integral on
the depletion-approximation potential) evaluated with scipy.constants and scipy.integrate.quad --
never a number DEVSIM or DynaMeta computed.

GATE A  Wide-bandgap junction (GaN-like, n_i = 1e-4 m^-3 = 1e-10 cm^-3, N_A = N_D = 1e23 m^-3) in
        forward bias 1.0 / 1.5 V: the terminal current (1e-17 ... 1e-13 A/m^2) against the SNS +
        Shockley oracle, and its conservation along the device (Jn + Jp per edge, uniform in DC).
        Double precision: the edge currents are round-off of the ~1e8 A/m^2 majority drift and
        diffusion terms (+-4e-5 A/m^2, both signs). Extended: oracle to < 1e-3.
GATE B  Heavily doped p+n junction (N_A = 1e25, N_D = 1e22, Si-like n_i = 1e16 m^-3, tau = 20 us)
        at zero bias with DynaMeta's DC tolerance (absolute 1e8 m^-3): double cannot converge (the
        p+ continuity update floors at ~1e25 * 1e-16 = 1e9); extended converges and the DC
        small-signal conductance dI/dV (+-0.1 mV) equals the oracle G0 (depletion SRH + diffusion).
        The ssac Re Y(1 Hz) of the same diode is recorded: extended precision does NOT fix it
        (DEVSIM solves the AC system in double -- see the audit note).
GATE C  Default off: a fingerprint (sha256 of every solved array) of the DC / ssac / noise /
        transient paths on the 1-D diode and of the 2-D layered reference metasurface, to compare
        across commits (run with --hashes from a checkout of the base commit and from this one).
GATE D  No leakage: default run -> extended run of the same device -> default run again in ONE
        process; the two default fingerprints are byte-identical and extended_solver is off.
COST    Wall time extended / double on the 1-D diodes and the 2-D layered reference metasurface.

Run: python -m validation.extended_precision [--hashes] [--cost]
"""

from __future__ import annotations

import contextlib
import hashlib
import math
import os
import sys
import time
import warnings

import numpy as np
from scipy.constants import e as QE, k as KB, epsilon_0 as EPS0
from scipy.integrate import quad
from scipy.optimize import brentq

# --root DIR: import dynameta from DIR (a checkout of another commit) -- the default-off fingerprint
# is compared ACROSS commits this way. Otherwise this checkout.
if "--root" in sys.argv:
    sys.path.insert(0, os.path.abspath(sys.argv[sys.argv.index("--root") + 1]))
else:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dynameta.carriers.devsim_env import import_devsim  # noqa: E402
ds = import_devsim()
from dynameta.carriers import eq_registry as _R                           # noqa: E402
from dynameta.carriers import physics_bipolar_dd as BP                    # noqa: E402
from dynameta.carriers.dc_solve import solve_dc                           # noqa: E402

T = 300.0
VT = KB * T / QE


@contextlib.contextmanager
def quiet():
    """Silence DEVSIM's C-level stdout (fd redirect)."""
    sys.stdout.flush()
    saved = os.dup(1)
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, 1)
        yield
    finally:
        sys.stdout.flush()
        os.dup2(saved, 1)
        os.close(devnull)
        os.close(saved)


def fresh() -> None:
    """Drop every device / mesh / circuit node (circuit nodes are global in DEVSIM)."""
    for d in list(ds.get_device_list()):
        ds.delete_device(device=d)
        _R.clear(d)
    for m in list(ds.get_mesh_list()):
        ds.delete_mesh(mesh=m)
    try:
        ds.delete_circuit()
    except Exception:                                           # noqa: BLE001 (no circuit)
        pass


# XP.solve's AdvisoryWarning on an 'ac'/'noise' solve of an opted-in device (matched by text, so
# --root on a pre-D2b checkout needs nothing new).
SMALL_SIGNAL_MARK = "buys nothing -- DEVSIM factorizes"


@contextlib.contextmanager
def small_signal_warnings(log: list, tag: str):
    """Record `tag` for every XP small-signal warning raised inside the block and swallow it;
    re-emit every OTHER warning unchanged (pytest's filterwarnings=error still sees those)."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        yield
    for w in caught:
        if SMALL_SIGNAL_MARK in str(w.message):
            log.append(tag)
        else:
            warnings.warn_explicit(w.message, w.category, w.filename, w.lineno)


def solver_flag():
    """The global extended_solver value, None when DEVSIM never saw it."""
    try:
        return ds.get_parameter(name="extended_solver")
    except ds.error:
        return None


# =====================================================================================================
# oracles (scipy only)
# =====================================================================================================

def _junction(NA, ND, n_i, eps_r, V):
    eps = eps_r * EPS0
    Vbi = VT * math.log(NA * ND / n_i ** 2)
    W = math.sqrt(2 * eps * (Vbi - V - 2 * VT) * (NA + ND) / (QE * NA * ND))
    return eps, Vbi, W, W * ND / (NA + ND), W * NA / (NA + ND)


def diode_current_oracle(V, *, NA, ND, n_i, tau, mu_n, mu_p, eps_r, Wp, Wn):
    """Current density [A/m^2] of an abrupt p-n diode (p side at x < 0, forward V > 0):
    Shockley diffusion with finite neutral widths, J0 = q n_i^2 [Dn/(NA Ln) coth(Wp'/Ln) +
    Dp/(ND Lp) coth(Wn'/Lp)], plus the Sah-Noyce-Shockley depletion SRH current
    q int U dx, U = n_i^2 (e^{V/VT} - 1)/(tau (n + p + 2 n_i)) (n1 = p1 = n_i, flat quasi-Fermi
    levels), on the depletion-approximation potential (parabolic, 2 V_T Debye correction)."""
    eps, Vbi, W, xp, xn = _junction(NA, ND, n_i, eps_r, V)
    Dn, Dp = mu_n * VT, mu_p * VT
    Ln, Lp = math.sqrt(Dn * tau), math.sqrt(Dp * tau)
    J0 = QE * n_i ** 2 * (Dn / (NA * Ln) / math.tanh((Wp - xp) / Ln)
                          + Dp / (ND * Lp) / math.tanh((Wn - xn) / Lp))
    psi_p = V - VT * math.log(NA / n_i)                # p neutral (phi_p = V)
    psi_n = VT * math.log(ND / n_i)                    # n neutral (phi_n = 0)
    a_p, a_n = QE * NA / (2 * eps), QE * ND / (2 * eps)

    def psi(x):
        if x <= 0.0:
            return psi_p + VT + a_p * (x + xp) ** 2
        return psi_n - VT - a_n * (xn - x) ** 2

    em1 = math.expm1(V / VT)

    def U(x):
        u = psi(x)
        return n_i ** 2 * em1 / (tau * (n_i * math.exp(u / VT) + n_i * math.exp((V - u) / VT)
                                        + 2 * n_i))
    xm = brentq(lambda x: psi(x) - V / 2, -xp, xn)    # integrand peak (n = p)
    I_srh = (quad(U, -xp, xm, limit=500, epsabs=0, epsrel=1e-12)[0]
             + quad(U, xm, xn, limit=500, epsabs=0, epsrel=1e-12)[0])
    return {"J": J0 * em1 + QE * I_srh, "J_diff": J0 * em1, "J_srh": QE * I_srh}


def diode_g0_oracle(**kw):
    """Zero-bias conductance [S/m^2] = dJ/dV at 0 by a symmetric difference of the oracle current
    (the integrand is smooth in V; +-1e-6 V reproduces the analytic V -> 0 limit to ~1e-10)."""
    h = 1e-6
    return (diode_current_oracle(h, **kw)["J"] - diode_current_oracle(-h, **kw)["J"]) / (2 * h)


# =====================================================================================================
# 1-D bipolar diode builder (p side at x < xj; 'lo' at x = 0, 'hi' at the far end)
# =====================================================================================================

def diode_1d(dev, lines, net_doping, *, n_i, tau, eps_r, mu_n, mu_p, extended, circuit=False,
             ae=1e8, re=1e-10, presolve=True):
    """Build + equilibrate. extended=True opts the device in right after create_device (the same
    call the builders make). circuit=True drives 'hi' from source V1 on node 'vhi' (ssac/noise/
    transient-ready), else both contacts are bias-parameter ohmic contacts ('lo_bias')."""
    fresh()
    mesh = dev + "_m"
    ds.create_1d_mesh(mesh=mesh)
    for pos, ps, tag in lines:
        ds.add_1d_mesh_line(mesh=mesh, pos=pos, ps=ps, **({"tag": tag} if tag else {}))
    ds.add_1d_contact(mesh=mesh, name="lo", tag="lo", material="metal")
    ds.add_1d_contact(mesh=mesh, name="hi", tag="hi", material="metal")
    ds.add_1d_region(mesh=mesh, material="semi", region="bulk", tag1="lo", tag2="hi")
    ds.finalize_mesh(mesh=mesh)
    ds.create_device(mesh=mesh, device=dev)
    if extended:                                       # imported here: --root on a pre-D2b checkout
        from dynameta.carriers import extended_precision as XP
        XP.enable(dev)
    ds.node_model(device=dev, region="bulk", name="NetDoping", equation=net_doping)
    with quiet():
        BP.setup_bipolar_region(dev, "bulk", eps_static=eps_r, n_dos_m3=2.8e25, n_i_m3=n_i,
                                mobility_n_m2Vs=mu_n, mobility_p_m2Vs=mu_p, tau_n_s=tau,
                                tau_p_s=tau, fd_enhancement=False)
        BP.setup_equilibrium_seed_models(dev, "bulk")
        BP.setup_contact_ohmic_bipolar(dev, "lo")
        if circuit:
            BP.setup_contact_ohmic_bipolar_circuit(dev, "hi", node_name="vhi", source_name="V1")
        else:
            BP.setup_contact_ohmic_bipolar(dev, "hi")
        ds.node_model(device=dev, region="bulk", name="_seed_psi",
                      equation=BP.equilibrium_seed_psi_expr(None))
    _seed(dev, ("Potential", "Electrons", "Holes"))
    if presolve:                                       # frozen-carrier Poisson first (1-D recipe)
        for ceq in ("ElectronContinuityEquation", "HoleContinuityEquation"):
            _R.delete_by_name(dev, ceq)
        _solve(dev, 1e10, 1e-10, 100)
        for ceq in ("ElectronContinuityEquation", "HoleContinuityEquation"):
            _R.reapply_by_name(dev, ceq)
        _seed(dev, ("Electrons", "Holes"))
    _solve(dev, ae, re)
    return dev


def _seed(dev, names):
    src = {"Potential": "_seed_psi", "Electrons": "IntrinsicElectrons", "Holes": "IntrinsicHoles"}
    for nm in names:
        ds.set_node_values(device=dev, region="bulk", name=nm, init_from=src[nm])


def _solve(dev, ae, re=1e-10, mi=80):
    """The DynaMeta DC entry point (dc_solve.solve_dc -> extended_precision.solve)."""
    with quiet():
        solve_dc(dev, method="newton", abs_tol=ae, rel_tol=re, max_iter=mi)


def _nv(dev, name):
    return np.array(ds.get_node_model_values(device=dev, region="bulk", name=name))


def edge_current(dev):
    """Total Jn + Jp on every edge [A/m^2] (uniform in DC 1-D: div J = 0)."""
    jn = np.array(ds.get_edge_model_values(device=dev, region="bulk", name="ElectronCurrent"))
    jp = np.array(ds.get_edge_model_values(device=dev, region="bulk", name="HoleCurrent"))
    return jn + jp


# =====================================================================================================
# GATE A: wide-bandgap forward current
# =====================================================================================================

GAN = dict(NA=1e23, ND=1e23, n_i=1e-4, tau=1e-6, mu_n=0.10, mu_p=0.002, eps_r=9.7,
           Wp=1e-6, Wn=1e-6)
GAN_V = (1.0, 1.5)


def gan_diode(extended):
    g = GAN
    lines = [(0.0, 20e-9, "lo"), (g["Wp"], 1e-9, None), (g["Wp"] + g["Wn"], 20e-9, "hi")]
    dop = "ifelse(x < {}, {}, {})".format(g["Wp"], -g["NA"], g["ND"])
    return diode_1d("gan", lines, dop, n_i=g["n_i"], tau=g["tau"], eps_r=g["eps_r"],
                    mu_n=g["mu_n"], mu_p=g["mu_p"], extended=extended, ae=1e8, re=1e-12,
                    presolve=False)   # the frozen-carrier Poisson stalls at V_bi = 3.2 V (rel 0.8,
    #                                   20 its, both precisions); the coupled Newton from the seed converges


def gate_a_wide_bandgap(extended, biases=GAN_V, step=0.1):
    """{V: (J_mean, J_oracle, conservation spread (max-min)/|mean|, ramp wall time)}."""
    dev = gan_diode(extended)
    oracle_kw = {k: GAN[k] for k in ("NA", "ND", "n_i", "tau", "mu_n", "mu_p", "eps_r", "Wp", "Wn")}
    out, v_now = {}, 0.0
    for V in biases:
        t0 = time.perf_counter()
        n = max(1, int(round((V - v_now) / step)))
        for k in range(1, n + 1):
            ds.set_parameter(device=dev, name="lo_bias", value=v_now + (V - v_now) * k / n)
            _solve(dev, 1e8, 1e-12, 60)
        v_now = V
        J = edge_current(dev)
        out[V] = (float(J.mean()), diode_current_oracle(V, **oracle_kw)["J"],
                  float((J.max() - J.min()) / abs(J.mean())), time.perf_counter() - t0)
    return out


# =====================================================================================================
# GATE B: heavily doped p+n junction at zero bias
# =====================================================================================================

PPN = dict(NA=1e25, ND=1e22, n_i=1e16, tau=2e-5, mu_n=0.135, mu_p=0.048, eps_r=11.7)
PPN["Wp"] = 2e-6
PPN["Wn"] = 5 * math.sqrt(PPN["mu_p"] * VT * PPN["tau"])     # 5 L_p long n side


def ppn_diode(extended, ae=1e8):
    g = PPN
    xj = g["Wp"]
    lines = [(0.0, 50e-9, "lo"), (xj, 1e-9, None), (xj + 1e-6, 10e-9, None),
             (xj + 20e-6, 300e-9, None), (xj + g["Wn"], g["Wn"] / 20, "hi")]
    dop = "ifelse(x < {}, {}, {})".format(xj, -g["NA"], g["ND"])
    return diode_1d("ppn", lines, dop, n_i=g["n_i"], tau=g["tau"], eps_r=g["eps_r"],
                    mu_n=g["mu_n"], mu_p=g["mu_p"], extended=extended, circuit=True, ae=ae,
                    re=1e-5)


def source_current():
    """DC terminal current INTO the device through 'hi' (the n end) [A/m^2]; the p+ end is
    grounded, so a forward bias is a NEGATIVE source value."""
    return -float(ds.get_circuit_node_value(node="V1.I", solution="dcop"))


def dc_conductance(ae, dv=1e-4):
    """G = dI/dV of the diode (forward = p side positive) at zero bias by a central difference of
    solved DC terminal currents; the 'hi' contact is the n side, so V_diode = -V_source."""
    out = []
    for v in (dv, -dv):                                 # source +dv = reverse, -dv = forward
        ds.circuit_alter(name="V1", value=v)
        _solve("ppn", ae, 1e-5)
        out.append(-source_current())                   # diode current (p -> n)
    ds.circuit_alter(name="V1", value=0.0)
    _solve("ppn", ae, 1e-5)
    return (out[1] - out[0]) / (2 * dv)


def gate_b_heavy_doping(extended, ae=1e8):
    """dict: converged, G_dc, G_oracle, ReY(1 Hz) [S/m^2]."""
    oracle = diode_g0_oracle(**{k: PPN[k] for k in ("NA", "ND", "n_i", "tau", "mu_n", "mu_p",
                                                     "eps_r", "Wp", "Wn")})
    try:
        ppn_diode(extended, ae=ae)
    except ds.error as ex:
        return {"converged": False, "error": str(ex).strip(), "G_oracle": oracle}
    from dynameta.carriers.ac_analysis import ssac_admittance
    log = []
    with quiet(), small_signal_warnings(log, "ssac"):
        _, _, G_ac = ssac_admittance([1.0])
    with small_signal_warnings(log, "dc"):
        G_dc = dc_conductance(ae)
    return {"converged": True, "G_dc": G_dc, "G_oracle": oracle, "ReY_1Hz": float(G_ac[0]),
            "small_signal_warnings": log}


# =====================================================================================================
# GATES C / D: default-off fingerprint and leakage
# =====================================================================================================

SI = dict(NA=1e22, ND=1e22, n_i=1e16, tau=1e-6, mu_n=0.135, mu_p=0.048, eps_r=11.7)


def _si_lines():
    Ln = math.sqrt(SI["mu_n"] * VT * SI["tau"])
    Lp = math.sqrt(SI["mu_p"] * VT * SI["tau"])
    xj = 2 * Ln
    return ([(0.0, Ln / 20, "lo"), (xj - 1e-6, 50e-9, None), (xj, 2e-9, None),
             (xj + 1e-6, 50e-9, None), (xj + 2 * Lp, Lp / 20, "hi")],
            "ifelse(x < {}, {}, {})".format(xj, -SI["NA"], SI["ND"]))


def _h(*arrays) -> str:
    m = hashlib.sha256()
    for a in arrays:
        m.update(np.ascontiguousarray(np.asarray(a, dtype=np.float64)).tobytes())
    return m.hexdigest()[:16]


# paths of the LAST fingerprint_1d run that raised XP's small-signal warning (one entry per warning)
LAST_SMALL_SIGNAL_WARNINGS: list = []


def fingerprint_1d(extended=False):
    """sha256 fingerprints of the 1-D Si diode through every DynaMeta solve path: DC (solve_dc) at
    0 and a 0.1 V forward ramp, ssac (ac_analysis), noise (noise_analysis, when present), transient
    (transient_step). Returns {path: hex}; the paths that raised XP's small-signal warning are left
    in LAST_SMALL_SIGNAL_WARNINGS. Tears the device down."""
    from dynameta.carriers.ac_analysis import ssac_admittance
    from dynameta.carriers.transient import transient_step
    log = LAST_SMALL_SIGNAL_WARNINGS
    log.clear()
    lines, dop = _si_lines()
    with small_signal_warnings(log, "dc0"):
        diode_1d("fp", lines, dop, n_i=SI["n_i"], tau=SI["tau"], eps_r=SI["eps_r"],
                 mu_n=SI["mu_n"], mu_p=SI["mu_p"], extended=extended, circuit=True)
    out = {"dc0": _h(_nv("fp", "Potential"), _nv("fp", "Electrons"), _nv("fp", "Holes"))}
    with small_signal_warnings(log, "dc_fwd"):
        for k in range(1, 5):                              # forward 0.1 V: n end at -0.1 V
            ds.circuit_alter(name="V1", value=-0.025 * k)
            _solve("fp", 1e8, 1e-10)
    out["dc_fwd"] = _h(_nv("fp", "Potential"), _nv("fp", "Electrons"), _nv("fp", "Holes"),
                       [source_current()])
    with quiet(), small_signal_warnings(log, "ssac"):
        f, C, G = ssac_admittance([1.0, 1e6])
    out["ssac"] = _h(C, G)
    try:
        from dynameta.carriers import noise_analysis as NA
        with quiet(), small_signal_warnings(log, "noise"):
            out["noise"] = _h(NA.noise_spectrum(device="fp", frequencies=[1.0, 1e6]).S)
    except ImportError:
        out["noise"] = "n/a"
    with quiet(), small_signal_warnings(log, "transient"):
        t, i = transient_step(-0.11, t_end=1e-9, dt0=1e-12, max_steps=200)
    out["transient"] = _h(t, i)
    fresh()
    return out


def fingerprint_2d():
    """sha256 of the 2-D layered reference metasurface (drift-diffusion, 9 regions) at Vg 0, +2 V."""
    from validation._reference_device import build_reference_modulator
    from dynameta.carriers.devsim_layered import LayeredDevsimBuilder
    from dynameta.sweep import BiasPoint
    fresh()
    b = LayeredDevsimBuilder(build_reference_modulator("drift_diffusion"), mesh_name="fp2_m",
                             device_name="fp2_d")
    out = {}
    for vg in (0.0, 2.0):
        with quiet():
            cf = b.solve(BiasPoint({"top_contact": vg}, "g"))
        out["layered_vg{:+.0f}".format(vg)] = _h(*[a for r in sorted(cf.regions)
                                                  for _, a in sorted(cf.regions[r].node_fields.items())])
    b.teardown()
    return out


def gate_d_leakage():
    """(default fingerprint, extended fingerprint, default-again fingerprint, extended_solver after)."""
    h1 = fingerprint_1d(False)
    h2 = fingerprint_1d(True)
    h3 = fingerprint_1d(False)
    return h1, h2, h3, solver_flag()


# =====================================================================================================
# COST
# =====================================================================================================

def cost_1d():
    """Wall time [s] extended vs double: GaN ramp 0 -> 1.5 V (16 Newton solves, 317 nodes) and the
    p+n equilibrium build (presolve + coupled Newton, 2 contacts)."""
    out = {}
    for ext in (False, True):
        t0 = time.perf_counter()
        gate_a_wide_bandgap(ext, biases=(1.5,))
        out[("gan_ramp", ext)] = time.perf_counter() - t0
        t0 = time.perf_counter()
        try:
            ppn_diode(ext, ae=1e8 if ext else 1e13)
        finally:
            out[("ppn_build", ext)] = time.perf_counter() - t0
        fresh()
    return out


def cost_2d(vgs=(0.0, 2.0)):
    """Wall time [s] of LayeredDevsimBuilder.solve on the reference metasurface, double vs extended."""
    from validation._reference_device import build_reference_modulator
    from dynameta.carriers.devsim_layered import LayeredDevsimBuilder
    from dynameta.sweep import BiasPoint
    out = {}
    for ext in (False, True):
        fresh()
        b = LayeredDevsimBuilder(build_reference_modulator("drift_diffusion"), mesh_name="c2_m",
                                 device_name="c2_d", extended_precision=ext)
        for vg in vgs:
            t0 = time.perf_counter()
            with quiet():
                b.solve(BiasPoint({"top_contact": vg}, "g"))
            out[(vg, ext)] = time.perf_counter() - t0
        b.teardown()
    return out


# =====================================================================================================

def main():
    if "--hashes" in sys.argv:
        fp = fingerprint_1d(False)
        fp.update(fingerprint_2d())
        for k, v in fp.items():
            print("{:14s} {}".format(k, v))
        print("extended_solver after:", solver_flag())
        return 0
    if "--cost" in sys.argv:
        for k, v in cost_1d().items():
            print("cost 1-D {:10s} {:8s} {:8.2f} s".format(k[0], "extended" if k[1] else "double", v))
        for k, v in cost_2d().items():
            print("cost 2-D Vg {:+.0f} {:8s} {:8.2f} s".format(k[0], "extended" if k[1] else "double", v))
        return 0
    ok = True
    for ext in (False, True):
        tag = "extended" if ext else "double  "
        for V, (J, Jo, spread, _t) in gate_a_wide_bandgap(ext).items():
            good = abs(J / Jo - 1) < 1e-2 and spread < 1e-3
            ok &= good == ext
            print("A {} V={:.1f}: J={:+.6e} oracle={:.6e} J/oracle-1={:+.2e} spread={:.2e} {}"
                  .format(tag, V, J, Jo, J / Jo - 1, spread, "PASS" if good else "FAIL"))
        b = gate_b_heavy_doping(ext)
        if not b["converged"]:
            print("B {} not converged ({}) -- oracle G0 = {:.6e}".format(tag, b["error"], b["G_oracle"]))
            ok &= not ext
        else:
            good = abs(b["G_dc"] / b["G_oracle"] - 1) < 1e-3
            ok &= good
            print("B {} G_dc={:.6e} oracle={:.6e} ({:+.2e}) ssac ReY(1Hz)={:+.4e}"
                  .format(tag, b["G_dc"], b["G_oracle"], b["G_dc"] / b["G_oracle"] - 1, b["ReY_1Hz"]))
    h1, h2, h3, flag = gate_d_leakage()
    leak_ok = h1 == h3 and not flag
    ok &= leak_ok
    print("D default {}\n  extended {}\n  default again {}\n  extended_solver after: {} -> {}"
          .format(h1, h2, h3, flag, "PASS" if leak_ok else "FAIL"))
    print("ALL PASS" if ok else "FAILURES")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
