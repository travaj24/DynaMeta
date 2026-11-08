"""Child-process scenarios for tests/test_fd_mode_complete.py (v0.15: statistics="fermi_dirac" completed for
local BTBT, nonlocal BTBT, charged bulk traps and interface traps; docs/audit/2026-11-06-v0.15-fermi-dirac-mode-
complete.md).

WHY A CHILD. The nonlocal term is a custom_terms term (process-global, permanent DEVSIM registration), and the
FDT scenarios add a process-global circuit; every scenario therefore runs in a fresh interpreter and reports
plain JSON:

    python tests/_fd_mode_child.py <scenario> <out.json> [options-json]

Devices (all Si-like: N_c = 2.8e25, N_v = 1.04e25 m^-3, no bandgap narrowing unless stated):
  zener  1-D abrupt p+ / n+ junction, N_A = 1e25 (eta_p = -0.0) | N_D = 1e26 m^-3 (eta_n = +2.6), n_i 1e16,
         uniform spacing h over +-6 nm around the junction (an integral number of edges: no odd edge on the
         field peak), plain ohmic contacts at 0 V. The ASYMMETRIC degeneracy is deliberate: the legacy
         statistics' zero-bias defect is ~ exp(Delta_n) - exp(Delta_p), which vanishes on a symmetric junction
         (the 2.7e-4 the nonlocal note measured on the symmetric 5e24 Zener).
  diode  1-D n+ (N_D, degenerate) / p (1e23) diode, 2 um, n_i 1e16, the carrier for the bulk-trap levels.
  mos    1-D n+ Si (N_D) / 10 nm SiO2 with a Potential-only gate: ohmic body, gate at `vg` (an equilibrium for
         ANY vg: no DC path through the oxide), interface traps on si_ox.
  fdt    the FDT diodes: n_i = n_i_fdt (default 3e19) so that the zero-bias conductance is resolvable by DC
         Richardson differences, degenerate n+ side, circuit contact 'hi' (source V1).
"""
from __future__ import annotations

import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for _p in (ROOT, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402

Q = 1.602176634e-19
KB = 1.380649e-23
NC, NV = 2.8e25, 1.04e25
MU_N, MU_P = 0.135, 0.048


def _ds():
    from dynameta.carriers.devsim_env import import_devsim
    return import_devsim()


def _lst(a):
    return [float(v) for v in np.ravel(a)]


def _quiet():
    import validation.devsim_noise as V
    return V.quiet()


def _v(dev, name, region="b"):
    return np.asarray(_ds().get_node_model_values(device=dev, region=region, name=name), dtype=float)


def _ev(dev, name, region="b"):
    return np.asarray(_ds().get_edge_model_values(device=dev, region=region, name=name), dtype=float)


def solve(ae=1e13, re_=1e-12, it=100):
    """DC Newton. With kinetic traps on the device, first in physics_traps.kinetic_dc_form (the DC-form rows,
    as dc_solve.solve_dc does), then in the PHYSICAL form from that root (the state every gate reads)."""
    ds = _ds()
    from dynameta.carriers import physics_traps as PT
    kin = any(PT.kinetic_trap_variables(d, r) for d in ds.get_device_list() for r in ds.get_region_list(device=d))
    with _quiet():
        if kin:
            with PT.kinetic_dc_form():
                info = ds.solve(type="dc", solver_type="direct", absolute_error=ae, relative_error=re_,
                                maximum_iterations=it, info=True)
            if not info["converged"]:
                raise RuntimeError("DC solve (kinetic DC form) did not converge")
        info = ds.solve(type="dc", solver_type="direct", absolute_error=ae, relative_error=re_,
                        maximum_iterations=it, info=True)
    if not info["converged"]:
        raise RuntimeError("DC solve did not converge")
    return len(info["iterations"])


# =====================================================================================================
# mechanisms
# =====================================================================================================

def _btbt(opts):
    from dynameta.carriers.tunneling import BTBTParams, SILICON_HURKX, SILICON_KAO_DIRECT
    m = opts.get("btbt")
    if m == "hurkx":
        return SILICON_HURKX
    if m == "tuple":
        return (SILICON_HURKX, SILICON_KAO_DIRECT)
    if m == "fdt":                                   # strong artificial set (the FDT diode's junction field)
        return BTBTParams(opts.get("btbt_A", 1.0e17), opts.get("btbt_B", 1.0e7), 2.5)
    return None


def _traps(opts):
    from dynameta.carriers import physics_traps as PT
    if not opts.get("traps"):
        return None
    tn, tp = opts.get("trap_tau", (1e-9, 4e-9))
    N = float(opts.get("trap_N", 3e23))
    dyn = bool(opts.get("kinetic"))                 # TrapLevel(dynamic=True): the kinetic occupancy variable
    # a SHALLOW acceptor 0.05 eV below E_c (n1 = N_c exp(-0.05/kT) = 4e24 m^-3: comparable with n on the n+
    # side, where the FD emission density gamma_n n1 matters), a donor 0.06 eV above E_v, and a midgap level:
    # neutral (recombination-only) when quasi-static, ACCEPTOR-like whenever the scenario involves kinetic levels
    # (opts 'kinetic' or 'mid_acceptor': a kinetic level holds charge -- physics_traps.NeutralKineticTrapError,
    # v0.15 neutral-kinetic fix -- and the quasi-static twin of a kinetic device must carry the same charge)
    mid_acc = dyn or bool(opts.get("mid_acceptor"))
    return (PT.TrapLevel(energy_eV=0.05, energy_reference="conduction", density_m3=N, kind="acceptor",
                         tau_n0_s=tn, tau_p0_s=tp, dynamic=dyn),
            PT.TrapLevel(energy_eV=0.06, energy_reference="valence", density_m3=N, kind="donor",
                         tau_n0_s=tp, tau_p0_s=tn, dynamic=dyn),
            PT.TrapLevel(energy_eV=0.0, kind="acceptor" if mid_acc else "neutral", tau_n0_s=tn, tau_p0_s=tp,
                         dynamic=dyn, density_m3=N if mid_acc else 0.0))[:(2 if opts.get("no_neutral") else 3)]


def _region_kw(opts):
    kw = {}
    st = opts.get("stats")
    if st is not None:
        kw["statistics"] = st
    b = _btbt(opts)
    if b is not None:
        kw["btbt"] = b
    t = _traps(opts)
    if t is not None:
        kw["traps"] = t
    if opts.get("bgn"):
        from dynameta.carriers.bipolar_options import BGN_SI_KLAASSEN_1992, BipolarOptions
        kw["options"] = BipolarOptions(bgn=BGN_SI_KLAASSEN_1992)
    return kw


# =====================================================================================================
# devices
# =====================================================================================================

def _setup_region(dev, region, opts, n_i, tau=1e-6):
    from dynameta.carriers import physics_bipolar_dd as BP
    fd = opts.get("stats") != "boltzmann"
    BP.setup_bipolar_region(dev, region, eps_static=11.7, n_dos_m3=NC, n_dos_p_m3=NV, n_i_m3=n_i,
                            mobility_n_m2Vs=MU_N, mobility_p_m2Vs=MU_P, tau_n_s=tau, tau_p_s=tau,
                            fd_enhancement=fd, **_region_kw(opts))
    BP.setup_equilibrium_seed_models(dev, region)


def _seed(dev, region="b"):
    from dynameta.carriers import physics_bipolar_dd as BP
    ds = _ds()
    ds.node_model(device=dev, region=region, name="_seed",
                  equation=BP.equilibrium_seed_psi_expr(BP.fd_shift_model(dev, region)))
    for v, m in (("Potential", "_seed"), ("Electrons", "IntrinsicElectrons"), ("Holes", "IntrinsicHoles")):
        ds.set_node_values(device=dev, region=region, name=v,
                           values=ds.get_node_model_values(device=dev, region=region, name=m))
    _seed_kinetic(dev)


def _seed_kinetic(dev):
    from dynameta.carriers import physics_traps as PT
    if any(PT.kinetic_trap_variables(dev, r) for r in _ds().get_region_list(device=dev)):
        PT.seed_kinetic_traps(dev)


def _presolve(dev, region="b"):
    from dynameta.carriers import eq_registry as _R
    for ceq in ("ElectronContinuityEquation", "HoleContinuityEquation"):
        _R.delete_by_name(dev, ceq)
    solve(1e10, 1e-10, 100)
    for ceq in ("ElectronContinuityEquation", "HoleContinuityEquation"):
        _R.reapply_by_name(dev, ceq)
    _seed_kinetic(dev)


def _lines_1d(ds, mesh, L, xj, h, w, coarse, tags=("pc", "nc")):
    ds.create_1d_mesh(mesh=mesh)
    ds.add_1d_mesh_line(mesh=mesh, pos=0.0, ps=coarse, tag=tags[0])
    ds.add_1d_mesh_line(mesh=mesh, pos=xj - w, ps=h)
    ds.add_1d_mesh_line(mesh=mesh, pos=xj, ps=h)
    ds.add_1d_mesh_line(mesh=mesh, pos=xj + w, ps=h)
    ds.add_1d_mesh_line(mesh=mesh, pos=L, ps=coarse, tag=tags[1])


def build_junction(dev, opts):
    """zener (default) or diode (opts['device'] == 'diode'), plain ohmic contacts pc/nc at 0 V, solved."""
    ds = _ds()
    from dynameta.carriers import eq_registry as _R
    from dynameta.carriers import physics_bipolar_dd as BP
    diode = opts.get("device") == "diode"
    h = float(opts.get("h", 0.2e-9 if not diode else 1e-9))
    if diode:
        L, xj, w, coarse, NA, ND = 2e-6, 1e-6, 40e-9, 10e-9 * h / 1e-9, 1e23, float(opts.get("ND", 1e26))
    else:
        L, xj, w, coarse, NA, ND = 60e-9, 30e-9, 6e-9, 1e-9 * h / 0.2e-9, float(opts.get("NA", 1e25)), \
            float(opts.get("ND", 1e26))
    mesh = dev + "_m"
    _R.clear(dev)
    with _quiet():
        _lines_1d(ds, mesh, L, xj, h, w, coarse)
        ds.add_1d_contact(mesh=mesh, name="pc", tag="pc", material="metal")
        ds.add_1d_contact(mesh=mesh, name="nc", tag="nc", material="metal")
        ds.add_1d_region(mesh=mesh, material="Si", region="b", tag1="pc", tag2="nc")
        ds.finalize_mesh(mesh=mesh)
        ds.create_device(mesh=mesh, device=dev)
        ds.node_model(device=dev, region="b", name="NetDoping",
                      equation="ifelse(x < {!r}, {!r}, {!r})".format(xj, -NA, ND))
        _setup_region(dev, "b", opts, float(opts.get("n_i", 1e16)), tau=float(opts.get("tau", 1e-6)))
        for c in ("pc", "nc"):
            BP.setup_contact_ohmic_bipolar(dev, c)
        _seed(dev)
        _presolve(dev)
    solve()
    return dev


def build_mos(dev, opts):
    """n+ Si / SiO2 / gate, interface traps on si_ox; gate at opts['vg'] relative to the body's pinned psi."""
    ds = _ds()
    from dynameta.carriers import eq_registry as _R
    from dynameta.carriers import physics_bipolar_dd as BP
    from dynameta.carriers import physics_equilibrium as PE
    from dynameta.carriers import physics_traps as PT
    h = float(opts.get("h", 0.1e-9))
    ND = float(opts.get("ND", 1e26))
    mesh = dev + "_m"
    _R.clear(dev)
    with _quiet():
        ds.create_1d_mesh(mesh=mesh)
        ds.add_1d_mesh_line(mesh=mesh, pos=0.0, ps=5.0 * h * 10, tag="body")
        ds.add_1d_mesh_line(mesh=mesh, pos=40e-9, ps=h)
        ds.add_1d_mesh_line(mesh=mesh, pos=50e-9, ps=h, tag="i")
        ds.add_1d_mesh_line(mesh=mesh, pos=60e-9, ps=0.5e-9, tag="gate")
        ds.add_1d_contact(mesh=mesh, name="body", tag="body", material="metal")
        ds.add_1d_contact(mesh=mesh, name="gate", tag="gate", material="metal")
        ds.add_1d_region(mesh=mesh, material="Si", region="b", tag1="body", tag2="i")
        ds.add_1d_region(mesh=mesh, material="SiO2", region="ox", tag1="i", tag2="gate")
        ds.add_1d_interface(mesh=mesh, name="si_ox", tag="i")
        ds.finalize_mesh(mesh=mesh)
        ds.create_device(mesh=mesh, device=dev)
        ds.node_model(device=dev, region="b", name="NetDoping", equation=repr(ND))
        _setup_region(dev, "b", opts, float(opts.get("n_i", 1e16)), tau=float(opts.get("tau", 1e-6)))
        PE.setup_dielectric_region(dev, "ox", 3.9)
        PE.setup_interface(dev, "si_ox")
        BP.setup_contact_ohmic_bipolar(dev, "body")
        PE.setup_contact(dev, "gate")
        it = PT.InterfaceTraps(dit_eV_m2=float(opts.get("dit", 3e16)), band_gap_eV=1.12, charge="split",
                               sigma_n_m2=float(opts.get("sig_n", 1e-19)), sigma_p_m2=1e-20,
                               fixed_charge_C_m2=float(opts.get("qf", 0.0)),
                               **({"dynamic": True, "kinetic_bin_eV": (float(opts["kbin"]) if opts.get("kbin")
                                                                        else None)}
                                  if opts.get("kinetic") else {}))
        info = PT.install_interface_traps(dev, "si_ox", it)
        _seed(dev)
        psib = float(_v(dev, "_seed")[0])
        nox = len(ds.get_node_model_values(device=dev, region="ox", name="x"))
        ds.set_node_values(device=dev, region="ox", name="Potential", values=[psib] * nox)
        ds.set_parameter(device=dev, name="gate_bias", value=psib)
        _presolve(dev)
    solve()
    vg = float(opts.get("vg", 0.0))
    for k in range(1, 11):
        ds.set_parameter(device=dev, name="gate_bias", value=psib + vg * k / 10)
        solve()
    return dev, info


# =====================================================================================================
# measurements
# =====================================================================================================

def _terms(dev, region="b"):
    from dynameta.carriers import physics_bipolar_dd as BP
    return BP.recombination_terms(dev, region)


def mechanism_balance(dev, prefix, region="b"):
    """(net, one_way) = (q Int U, q Int (G + R)/2) of the additive terms whose name starts with `prefix`
    ('btbt' -> edge terms, 'trap' / 'it' -> node terms), with the equations' own weights. At an exact
    equilibrium net = 0; |net|/one_way is the detailed-balance defect."""
    t = _terms(dev, region)
    net = one = 0.0
    for nm, (rm, pm) in t["node"].items():
        if nm.startswith(prefix):
            w = _v(dev, "NodeVolume", region)
            net += float(np.sum(_v(dev, rm, region) * w))
            one += 0.5 * float(np.sum(_v(dev, pm, region) * w))
    for nm, (rm, pm) in t["edge"].items():
        if nm.startswith(prefix):
            w = 2.0 * _ev(dev, "EdgeNodeVolume", region)
            net += float(np.sum(_ev(dev, rm, region) * w))
            one += 0.5 * float(np.sum(_ev(dev, pm, region) * w))
    worst = None
    for key in ("node_electron", "node_hole"):  # kinetic levels / bins: per branch, Int |R| over Int P/2
        bn = bo = 0.0
        for nm, ct in t[key].items():
            if nm.startswith(prefix):
                w = _v(dev, "NodeVolume", region)
                bn += float(np.sum(np.abs(_v(dev, ct.rate, region)) * w))
                bo += 0.5 * float(np.sum(_v(dev, ct.pair, region) * w))
        if bo > 0.0:
            r = bn / bo
            if worst is None or r > worst[0] / worst[1]:
                worst = (bn, bo)
    if worst is not None:                          # the worse branch's (net, one-way)
        net, one = worst
    return Q * net, Q * one


def terminal(dev, contact="nc"):
    ds = _ds()
    return sum(ds.get_contact_current(device=dev, contact=contact, equation=e)
               for e in ("ElectronContinuityEquation", "HoleContinuityEquation"))


def contact_currents(dev, contact):
    ds = _ds()
    return {e[:4]: ds.get_contact_current(device=dev, contact=contact, equation=e)
            for e in ("ElectronContinuityEquation", "HoleContinuityEquation")}


def exact_equilibrium(dev, region="b", iters=8):
    """Set Electrons/Holes to an EXACT discrete equilibrium of the transport's own activities on the current
    psi: L_n = psi/V_t + a (flat E_Fn; a from the solved state at the n contact) and L_p = t - L_n,
    t = ln(n_ie0^2/(N_c N_v)) per node (n_ie0 = NieBgn under BGN, else n_i). Newton on DEVSIM's OWN closed-
    form HJ_L models (setup_log_activity: what the transport, the FD detailed balance and the nonlocal term
    evaluate). Returns max |L_n + L_p - t| after the iteration (the construction's residual)."""
    ds = _ds()
    from dynameta.carriers.physics_heterojunction import setup_log_activity
    fd = float(ds.get_parameter(device=dev, region=region, name="fd_enhancement")) != 0.0
    Ln = setup_log_activity(dev, region, "Electrons", "N_dos", fd=fd)
    Lp = setup_log_activity(dev, region, "Holes", "N_dos_p", fd=fd)
    vt = float(ds.get_parameter(device=dev, region=region, name="V_t"))
    nc = float(ds.get_parameter(device=dev, region=region, name="N_dos"))
    nv = float(ds.get_parameter(device=dev, region=region, name="N_dos_p"))
    nodes = set(ds.get_node_model_list(device=dev, region=region))
    n0 = _v(dev, "NieBgn", region) if "NieBgn" in nodes else \
        float(ds.get_parameter(device=dev, region=region, name="n_i"))
    t = np.log(n0 * n0 / (nc * nv)) * np.ones(len(_v(dev, "x", region)))
    psi = _v(dev, "Potential", region)
    a = float(_v(dev, Ln, region)[-1] - psi[-1] / vt)
    tn = psi / vt + a
    tp = t - tn
    for var, L, tgt in (("Electrons", Ln, tn), ("Holes", Lp, tp)):
        for _ in range(iters):
            c = _v(dev, var, region)
            dL = _v(dev, L + ":" + var, region)            # g/c
            c = c * np.exp((tgt - _v(dev, L, region)) / (c * dL))
            ds.set_node_values(device=dev, region=region, name=var, values=_lst(c))
    return float(np.max(np.abs(_v(dev, Ln, region) + _v(dev, Lp, region) - t)))


# =====================================================================================================
# scenarios
# =====================================================================================================

def sc_zero_bias(opts):
    """Solved zero-bias state of `opts['device']` with the mechanism of opts: the terminal current, the
    mechanism's net and one-way currents (solved state), and the same at the exact discrete equilibrium
    built on the solved psi (exact_equilibrium)."""
    from dynameta.carriers import btbt_nonlocal as NL
    ds = _ds()
    dev = "zb"
    out = {}
    if opts.get("device") == "mos":
        dev, info = build_mos(dev, opts)
        out["n_levels"] = len(info["levels"])
        prefix = "it"
        cc = contact_currents(dev, "body")
        out["I_n_body"], out["I_p_body"] = cc["Elec"], cc["Hole"]
    else:
        build_junction(dev, opts)
        prefix = "btbt" if opts.get("btbt") else "trap"
        out["I_term"] = terminal(dev)
    term = None
    if opts.get("nonlocal"):
        from dynameta.carriers.tunneling import SILICON_HURKX
        term = NL.setup_btbt_nonlocal(dev, "b", NL.NonlocalBTBT(SILICON_HURKX, Eg_tun_eV=opts.get("eg", 0.9)))
        out["its"] = solve()
        out["I_term"] = terminal(dev)

    def balance():
        if term is not None:
            st, res = term.evaluate(dev)
            one = 0.5 * res["R_pre"] * res["pair"] * st["vol"][res["paths"].start]
            net = res["R"] * st["vol"][res["paths"].start]
            return Q * float(np.sum(net)), Q * float(np.sum(one)), float(np.max(np.abs(res["R"]) /
                                                                               (0.5 * res["R_pre"] * res["pair"])))
        net, one = mechanism_balance(dev, prefix)
        return net, one, None
    out["net"], out["one_way"], out["max_path_ratio"] = balance()
    out["residual"] = residual_scale(dev)
    # SRH (the built-in channel) on the same state, for the decomposition of the terminal current
    w = _v(dev, "NodeVolume")
    out["srh_net"] = Q * float(np.sum(_v(dev, "USRH") * w))
    out["x"] = _lst(_v(dev, "x"))
    out["construction_residual"] = exact_equilibrium(dev)
    out["Lmax"] = float(max(np.max(np.abs(_v(dev, "HJ_L_Electrons"))), np.max(np.abs(_v(dev, "HJ_L_Holes")))))
    _seed_kinetic(dev)                                   # w = w_SS at the exact state: its equilibrium odds
    out["eq_net"], out["eq_one_way"], out["eq_max_path_ratio"] = balance()
    # node-wise worst |U| / (one-way) of every local term at the exact state
    t = _terms(dev)
    worst = 0.0
    for nm, (rm, pm) in list(t["node"].items()):
        if nm.startswith(prefix):
            u, pr = _v(dev, rm), _v(dev, pm)
            ok = pr > 0
            worst = max(worst, float(np.max(np.abs(u[ok]) / (0.5 * pr[ok]))) if np.any(ok) else 0.0)
    for nm, (rm, pm) in list(t["edge"].items()):
        if nm.startswith(prefix):
            u, pr = _ev(dev, rm), _ev(dev, pm)
            ok = pr > 0
            worst = max(worst, float(np.max(np.abs(u[ok]) / (0.5 * pr[ok]))) if np.any(ok) else 0.0)
    for key in ("node_electron", "node_hole"):
        for nm, ct in t[key].items():
            if nm.startswith(prefix):
                u, pr = _v(dev, ct.rate), _v(dev, ct.pair)
                ok = pr > 0
                worst = max(worst, float(np.max(np.abs(u[ok]) / (0.5 * pr[ok]))) if np.any(ok) else 0.0)
    usrh = _v(dev, "USRH")
    out["eq_worst_local"] = worst
    out["eq_srh_max"] = float(np.max(np.abs(usrh)))
    ds.delete_device(device=dev)
    return out


def residual_scale(dev, region="b"):
    """max over the continuity rows of |rhs| / (sum_k |J_rk| |x_k|): the solved state's relative residual
    (a false 'converged' -- an update throttled by the positive-variable limiter -- shows here)."""
    import scipy.sparse as sp
    import _custom_terms_child as CC
    ds = _ds()
    av, ai, ap, rhs = CC.matrix()
    J = sp.csc_matrix((av, ai, ap))
    x = np.zeros(J.shape[1])
    for eq in ds.get_equation_list(device=dev, region=region):
        v = ds.get_equation_command(device=dev, region=region, name=eq)["variable_name"]
        c = np.asarray(ds.get_equation_numbers(device=dev, region=region, variable=v))
        x[c[c >= 0]] = _v(dev, v, region)[c >= 0]
    worst = 0.0
    for var in ("Electrons", "Holes"):
        rows = np.asarray(ds.get_equation_numbers(device=dev, region=region, variable=var))[1:-1]
        sc = np.abs(J[rows, :]) @ np.abs(x)
        worst = max(worst, float(np.max(np.abs(rhs[rows]) / sc)))
    return worst


# ---------------------------------------------------------------------------------------------------
# Jacobian: DEVSIM's assembled matrix vs central differences of DEVSIM's own residual
# ---------------------------------------------------------------------------------------------------

def jac_check(dev, region, nodes, variables=("Potential", "Electrons", "Holes"), rel_h=1e-6):
    """For each (var, node): the assembled column J[:, c] against (F(x + h) - F(x - h))/2h and the 2h step
    (get_matrix_and_rhs; rhs = -F), on every row. Returns the worst normalised miss
        max_r |J_rc - fd_rc| / (1e-6 * S_r + 2 |fd_rc - fd2_rc|),
    S_r = sum_k |J_rk x_k| / |x_c| (the row's linearised-term scale per unit of the column variable), i.e.
    a pass is <= 1: the miss is within the central-difference truncation (|fd - fd2| estimates it) plus
    1e-6 of the row's own scale (the round-off of a residual summed from terms of size S_r |x_c|, divided by
    the step h = 1e-6 |x_c|)."""
    import scipy.sparse as sp
    import _custom_terms_child as CC
    ds = _ds()

    def mat():
        av, ai, ap, rhs = CC.matrix()
        return sp.csc_matrix((av, ai, ap)), rhs
    J, _ = mat()
    xall = np.zeros(J.shape[1])
    for eq in ds.get_equation_list(device=dev, region=region):
        v = ds.get_equation_command(device=dev, region=region, name=eq)["variable_name"]
        c = np.asarray(ds.get_equation_numbers(device=dev, region=region, variable=v))
        xall[c[c >= 0]] = _v(dev, v, region)[c >= 0]
    rowsc = np.abs(J) @ np.abs(xall)
    worst, where = 0.0, None
    for var in variables:
        cols = np.asarray(ds.get_equation_numbers(device=dev, region=region, variable=var))
        vals = _v(dev, var, region)
        for i in nodes:
            x0 = float(vals[i])
            h = rel_h * abs(x0) if var != "Potential" else 1e-7
            fds = []
            for step in (h, 2.0 * h):
                r = []
                for s in (1.0, -1.0):
                    ds.set_node_value(device=dev, region=region, name=var, index=int(i), value=x0 + s * step)
                    r.append(mat()[1])
                fds.append(-(r[0] - r[1]) / (2.0 * step))
            ds.set_node_value(device=dev, region=region, name=var, index=int(i), value=x0)
            jc = J[:, int(cols[i])].toarray().ravel()
            scale = 1e-6 * rowsc / max(abs(x0), 1e-300) + 2.0 * np.abs(fds[0] - fds[1]) + 1e-300
            m = np.abs(jc - fds[0]) / scale
            k = int(np.argmax(m))
            if m[k] > worst:
                worst, where = float(m[k]), (var, int(i), k, float(jc[k]), float(fds[0][k]))
    return worst, where


def _knock_out_fd_derivatives(dev, region="b"):
    """ADVERSARIAL: the Jacobian a fixed-n_ie implementation would assemble -- every FD degeneracy-factor
    derivative set to 0 (FDdLnGamma*, and the edge copies the BTBT D and the quasi-field read)."""
    ds = _ds()
    for m in ("FDdLnGammaN", "FDdLnGammaP"):
        ds.node_model(device=dev, region=region, name=m, equation="0")
    edges = set(ds.get_edge_model_list(device=dev, region=region))
    for m in sorted(edges):
        if m.startswith(("FDNiN@", "FDNiP@")) and ":" in m:
            ds.edge_model(device=dev, region=region, name=m, equation="0")
    for v in ("Electrons", "Holes", "Potential"):            # force re-evaluation (DEVSIM model cache)
        ds.set_node_values(device=dev, region=region, name=v, values=_lst(_v(dev, v, region)))


def sc_jacobian(opts):
    """FD-mode Jacobian of the device with the mechanism of opts at a NON-equilibrium state (zener: reverse
    0.5 V; diode: forward 0.3 V; mos: gate -0.3 V with the electrons kicked x1.3), columns = every
    variable at the nodes of a window around the junction / interface."""
    ds = _ds()
    dev = "jz"
    if opts.get("device") == "mos":
        dev, _info = build_mos(dev, opts)
        x = _v(dev, "x")
        ds.set_node_values(device=dev, region="b", name="Electrons", values=_lst(1.3 * _v(dev, "Electrons")))
        nodes = [int(i) for i in np.argsort(-x)[:4]]
    else:
        build_junction(dev, opts)
        if opts.get("nonlocal"):
            from dynameta.carriers import btbt_nonlocal as NL
            from dynameta.carriers.tunneling import SILICON_HURKX
            NL.setup_btbt_nonlocal(dev, "b", NL.NonlocalBTBT(SILICON_HURKX, Eg_tun_eV=opts.get("eg", 0.9)))
            solve()
        c, v = ("pc", -0.5) if opts.get("device") != "diode" else ("nc", -0.3)
        for k in range(1, 6):
            ds.set_parameter(device=dev, name="{}_bias".format(c), value=v * k / 5)
            solve()
        x = _v(dev, "x")
        xj = 30e-9 if opts.get("device") != "diode" else 1e-6
        idx = np.argsort(np.abs(x - xj))[:int(opts.get("ncols", 6))]
        nodes = sorted(int(i) for i in idx if 0 < i < len(x) - 1)
    from dynameta.carriers import physics_traps as PT
    variables = ("Potential", "Electrons", "Holes") + tuple(PT.kinetic_trap_variables(dev, "b"))
    out = {"residual": residual_scale(dev), "variables": list(variables)}
    out["worst"], out["where"] = jac_check(dev, "b", nodes, variables=variables)
    if opts.get("knockout"):
        _knock_out_fd_derivatives(dev)
        out["ko_worst"], out["ko_where"] = jac_check(dev, "b", nodes, variables=variables)
    return out


def sc_dc_kinetic_vs_quasi_static(opts):
    """DC: the device with KINETIC levels (or bins) at a forward / gate operating point equals the same
    device with QUASI-STATIC ones (the K1 identity, now under opts['stats']): potential, carriers, and the
    trap / surface recombination current."""
    ds = _ds()
    out = {}
    for tag, kin in (("qs", False), ("kin", True)):
        o = dict(opts, kinetic=kin, mid_acceptor=1)
        if o.get("device") == "mos":
            dev, _info = build_mos("dk" + tag, o)
            prefix = "it"
        else:
            dev = build_junction("dk" + tag, o)
            for k in range(1, 7):
                ds.set_parameter(device=dev, name="nc_bias", value=-0.05 * k)
                solve()
            prefix = "trap"
        t = _terms(dev)
        w = _v(dev, "NodeVolume")
        if kin:
            cur = sum(float(np.sum(_v(dev, ct.rate) * w)) for nm, ct in t["node_electron"].items()
                      if nm.startswith(prefix))
        else:
            cur = sum(float(np.sum(_v(dev, rm) * w)) for nm, (rm, pm) in t["node"].items() if nm.startswith(prefix))
        one = 0.0
        if kin:                                      # the fast exchange: the round-off scale of R_n vs U
            for key in ("node_electron", "node_hole"):
                one += sum(0.5 * float(np.sum(_v(dev, ct.pair) * w)) for nm, ct in t[key].items()
                           if nm.startswith(prefix))
        out[tag] = {"psi": _v(dev, "Potential"), "n": _v(dev, "Electrons"), "one": Q * one,
                    "p": _v(dev, "Holes"), "I": Q * cur, "residual": residual_scale(dev)}
        ds.delete_device(device=dev)
    q, k = out["qs"], out["kin"]
    return {"dpsi": float(np.max(np.abs(k["psi"] - q["psi"]))),
            "dn": float(np.max(np.abs(k["n"] / q["n"] - 1.0))), "dp": float(np.max(np.abs(k["p"] / q["p"] - 1.0))),
            "I_qs": q["I"], "I_kin": k["I"], "one_way_kin": k["one"],
            "residual": max(q["residual"], k["residual"])}


def sc_jacobian_2d(opts):
    """2-D (element assembly) local BTBT under statistics='fermi_dirac' on a tensor-mesh p+/n+ slab (the
    _fdext_calllog 2-D mesh, doping step along x), Jacobian columns at nodes next to the junction, after a
    -0.3 V reverse ramp. Optional knockout."""
    ds = _ds()
    from dynameta.carriers import eq_registry as _R
    from dynameta.carriers import physics_bipolar_dd as BP
    import _fdext_calllog as CL
    dev, mesh = "j2", "j2_m"
    _R.clear(dev)
    n = int(opts.get("n", 10))
    L = 20e-9
    with _quiet():
        import numpy as _np
        g = _np.linspace(0.0, L, n + 1)
        X, Y = _np.meshgrid(g, _np.linspace(0.0, 4e-9, 3), indexing="ij")
        coords = _np.column_stack([X.ravel(), Y.ravel(), 0.0 * X.ravel()])
        idx = _np.arange(X.size).reshape(X.shape)
        el = []
        for i in range(n):
            for j in range(2):
                a, b, c, d = idx[i, j], idx[i + 1, j], idx[i + 1, j + 1], idx[i, j + 1]
                el += [2, 0, int(a), int(b), int(c), 2, 0, int(a), int(c), int(d)]
        for j in range(2):
            el += [1, 1, int(idx[0, j]), int(idx[0, j + 1])]
            el += [1, 2, int(idx[n, j]), int(idx[n, j + 1])]
        ds.create_gmsh_mesh(mesh=mesh, coordinates=[float(c) for c in coords.ravel()],
                            physical_names=["r", "pc", "nc"], elements=el)
        ds.add_gmsh_region(mesh=mesh, gmsh_name="r", region="b", material="Si")
        ds.add_gmsh_contact(mesh=mesh, gmsh_name="pc", region="b", name="pc", material="metal")
        ds.add_gmsh_contact(mesh=mesh, gmsh_name="nc", region="b", name="nc", material="metal")
        ds.finalize_mesh(mesh=mesh)
        ds.create_device(mesh=mesh, device=dev)
        ds.node_model(device=dev, region="b", name="NetDoping", equation="ifelse(x < 1e-8, -1e25, 1e26)")
        _setup_region(dev, "b", dict(opts, btbt=opts.get("btbt", "hurkx")), 1e16)
        for c in ("pc", "nc"):
            BP.setup_contact_ohmic_bipolar(dev, c)
        _seed(dev)
        _presolve(dev)
    del CL
    solve()
    for k in range(1, 4):
        ds.set_parameter(device=dev, name="pc_bias", value=-0.1 * k)
        solve()
    x = _v(dev, "x")
    nodes = [int(i) for i in np.argsort(np.abs(x - 10e-9))[:4]]
    out = {}
    out["worst"], out["where"] = jac_check(dev, "b", nodes)
    if opts.get("knockout"):
        _knock_out_fd_derivatives(dev)
        out["ko_worst"], out["ko_where"] = jac_check(dev, "b", nodes)
    return out


def sc_jacobian_nonlocal(opts):
    """The nonlocal term's OWN Jacobian share under statistics='fermi_dirac' (tests/_custom_terms_child.
    fd_columns, the shared helper of the nonlocal gates) at a reverse-biased degenerate Zener."""
    import _custom_terms_child as CC
    from dynameta.carriers import btbt_nonlocal as NL
    from dynameta.carriers import custom_terms as CT
    from dynameta.carriers.tunneling import SILICON_HURKX
    ds = _ds()
    dev = "jn"
    build_junction(dev, opts)
    term = NL.setup_btbt_nonlocal(dev, "b", NL.NonlocalBTBT(SILICON_HURKX, Eg_tun_eV=opts.get("eg", 0.9)))
    solve()
    for k in range(1, 6):
        ds.set_parameter(device=dev, name="pc_bias", value=-0.5 * k / 5)
        solve()
    real = term.assemble
    term.active = True
    term.assemble = lambda d, w, t: real(d, w, t) if term.active else ([], [])
    st, res = term.evaluate(dev)
    p, perm = res["paths"], st["perm"]
    k = int(np.argmax(np.abs(res["R"])))
    i, a, b = int(p.start[k]), int(p.a[k]), int(p.b[k])
    rows = np.concatenate([CT.node_rows(dev, "b", "HoleContinuityEquation", perm[[i]]),
                           CT.node_rows(dev, "b", "ElectronContinuityEquation", perm[[a, b]])])
    recs = CC.fd_columns(dev, "b", "Electrons", list(perm[sorted({a, b})]), rows, lambda x: 1e-5 * x, term)
    recs += CC.fd_columns(dev, "b", "Holes", list(perm[[i]]), rows, lambda x: 1e-5 * x, term)
    recs += CC.fd_columns(dev, "b", "Potential", list(perm[[i, (i + a) // 2, a]]), rows, lambda x: 1e-5, term)
    worst = 0.0
    for r in recs:
        J, fd, fd2 = (np.asarray(r[k_]) for k_ in ("J", "fd", "fd2"))
        tol = 1e-6 * r["rhs_scale"] / (abs(r["h"]) / 1e-5 if r["var"] != "Potential" else 1.0) + \
            2.0 * np.abs(fd - fd2) + 1e-300
        worst = max(worst, float(np.max(np.abs(J - fd) / tol)))
    return {"worst": worst, "n": len(recs), "max_J": float(max(np.max(np.abs(r["J"])) for r in recs))}


# ---------------------------------------------------------------------------------------------------
# FDT: S = 4 kT Re Y at equilibrium (bias +-V0 averaged: the circuit's relative update cannot converge at
# exactly zero current, and the odd O(V0) term cancels in the average)
# ---------------------------------------------------------------------------------------------------

def _solve_circuit(ae):
    """DC Newton of a circuit device (relative 1e-9: the circuit current's own round-off); with kinetic traps
    first in their DC form (as solve()), then the physical form."""
    ds = _ds()
    from dynameta.carriers import physics_traps as PT
    kin = any(PT.kinetic_trap_variables(d, r) for d in ds.get_device_list() for r in ds.get_region_list(device=d))
    with _quiet():
        if kin:
            with PT.kinetic_dc_form():
                ds.solve(type="dc", solver_type="direct", absolute_error=ae, relative_error=1e-9,
                         maximum_iterations=60, info=True)
        info = ds.solve(type="dc", solver_type="direct", absolute_error=ae, relative_error=1e-9,
                        maximum_iterations=60, info=True)
    if not info["converged"]:
        raise RuntimeError("DC solve did not converge")


_ADVISED = []        # the quasi-static charged-trap advisories the noise / ssac solves raised (decision 39)


def _advised(fn, *a):
    """Run a noise / ssac call recording the decision-39 AdvisoryWarning (QUASI-STATIC charged traps: the
    small-signal answer is exact only well below the capture rates); every other warning stays an error."""
    import warnings
    from dynameta.warning_types import AdvisoryWarning
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", AdvisoryWarning)
        out = fn(*a)
    for w in caught:
        if issubclass(w.category, AdvisoryWarning) and "QUASI-STATIC CHARGED traps" in str(w.message):
            _ADVISED.append(str(w.message)[:80])
        else:
            warnings.warn_explicit(w.message, w.category, w.filename, w.lineno)
    return out


def fdt_ratio(ae, v0=1e-7, freqs=(1.0, 1e6)):
    """(S/(4 kT Re Y) - 1 at each f, the noise share by mechanism, Re Y) with the +-v0 average."""
    import validation.devsim_noise as V
    ds = _ds()
    S, R, share = 0.0, 0.0, None
    for sg in (1.0, -1.0):
        ds.circuit_alter(name="V1", value=sg * v0)
        _solve_circuit(ae)
        _solve_circuit(ae)
        sn = _advised(V.noise, list(freqs))
        Y = _advised(V.ssac_Y, list(freqs))
        S = S + 0.5 * sn.S
        R = R + 0.5 * Y.real
        if share is None:
            share = {k: float(v[0] / sn.S[0]) for k, v in sn.by_mechanism.items() if float(v[0]) != 0.0}
    return _lst(S / (4.0 * V.KT * R) - 1.0), share, _lst(R)


def build_fdt_diode(dev, opts):
    """1-D n+ (N_D, degenerate) / p (1e22) diode, n_i = 3e19 (Ge-like: a zero-bias conductance ssac
    resolves), 'lo' grounded on p, 'hi' the circuit contact (V1) on n+, with the mechanism of opts."""
    import validation.devsim_noise as V
    ds = _ds()
    from dynameta.carriers import eq_registry as _R
    from dynameta.carriers import physics_bipolar_dd as BP
    V.fresh()
    n_i = float(opts.get("n_i", 3e19))
    ND, NA = float(opts.get("ND", 3e25)), 1e22
    h = float(opts.get("h", 1e-9))
    mesh = dev + "_m"
    _R.clear(dev)
    xj = 3e-6
    with _quiet():
        ds.create_1d_mesh(mesh=mesh)
        ds.add_1d_mesh_line(mesh=mesh, pos=0.0, ps=50e-9, tag="lo")
        ds.add_1d_mesh_line(mesh=mesh, pos=xj - 0.4e-6, ps=h * 5)
        ds.add_1d_mesh_line(mesh=mesh, pos=xj - 0.1e-6, ps=h)
        ds.add_1d_mesh_line(mesh=mesh, pos=xj, ps=h)
        ds.add_1d_mesh_line(mesh=mesh, pos=xj + 0.02e-6, ps=h)
        ds.add_1d_mesh_line(mesh=mesh, pos=xj + 1e-6, ps=50e-9, tag="hi")
        ds.add_1d_contact(mesh=mesh, name="lo", tag="lo", material="metal")
        ds.add_1d_contact(mesh=mesh, name="hi", tag="hi", material="metal")
        ds.add_1d_region(mesh=mesh, material="Si", region="b", tag1="lo", tag2="hi")
        ds.finalize_mesh(mesh=mesh)
        ds.create_device(mesh=mesh, device=dev)
        ds.node_model(device=dev, region="b", name="NetDoping",
                      equation="ifelse(x < {!r}, {!r}, {!r})".format(xj, -NA, ND))
        _setup_region(dev, "b", opts, n_i, tau=float(opts.get("tau", 1e-6)))
        BP.setup_contact_ohmic_bipolar(dev, "lo")
        BP.setup_contact_ohmic_bipolar_circuit(dev, "hi", node_name="vhi", source_name="V1")
        _seed(dev)
        _presolve(dev)
    return dev


def sc_fdt(opts):
    """FDT of the 1-D FDT diode with the mechanism of opts (btbt='fdt' strong artificial local set; nonlocal
    with the same set; traps) under opts['stats']."""
    dev = build_fdt_diode("fd", opts)
    ae = 1e12
    if opts.get("nonlocal"):
        from dynameta.carriers import btbt_nonlocal as NL
        from dynameta.carriers.tunneling import BTBTParams
        NL.setup_btbt_nonlocal(dev, "b", NL.NonlocalBTBT(BTBTParams(opts.get("btbt_A", 1e17), 1e7, 2.5),
                                                         Eg_tun_eV=0.25, max_length_m=3e-7))
    r, share, ReY = fdt_ratio(ae)
    out = {"fdt": r, "share": share, "ReY": ReY, "residual": residual_scale(dev)}
    out["advisories"] = len(_ADVISED)
    if opts.get("dc"):                         # DC conductance by Richardson (+-dv, +-2dv) and the noise at V0
        import validation.devsim_noise as V
        ds = _ds()
        cur = {}
        for v in (1e-4, -1e-4, 2e-4, -2e-4):
            ds.circuit_alter(name="V1", value=v)
            _solve_circuit(ae)
            cur[v] = V.source_current()
        g1, g2 = (cur[1e-4] - cur[-1e-4]) / 2e-4, (cur[2e-4] - cur[-2e-4]) / 4e-4
        G = (4.0 * g1 - g2) / 3.0
        S = 0.0
        for sg in (1.0, -1.0):
            ds.circuit_alter(name="V1", value=sg * 1e-7)
            _solve_circuit(ae)
            _solve_circuit(ae)
            S = S + 0.5 * float(_advised(V.noise, [1.0]).S[0])
        out["fdt_dc"] = S / (4.0 * V.KT * G) - 1.0
        out["G_dc"], out["G_spread"] = G, g1 - g2
    return out


def build_hetero(dev, opts):
    """1-D bipolar n+ (N0) / n (N1) ISOTYPE heterostack joined by a thermionic interface (affinity step
    dchi, Richardson A* = astar x the free-electron value), 'lo' grounded on r0, 'hi' (V1) on r1."""
    import validation.devsim_noise as V
    ds = _ds()
    from dynameta.carriers import eq_registry as _R
    from dynameta.carriers import physics_bipolar_dd as BP
    from dynameta.carriers import physics_equilibrium as PE
    from dynameta.carriers import physics_heterojunction as HJ
    from dynameta.constants import M_E, V_T as VT_CODE
    V.fresh()
    N0, N1 = float(opts.get("N0", 1e26)), float(opts.get("N1", 3e25))
    L, h = 200e-9, float(opts.get("h", 1e-9))
    n_i = 1e16
    fd = opts.get("stats") != "boltzmann"
    mesh = dev + "_m"
    _R.clear(dev)
    with _quiet():
        ds.create_1d_mesh(mesh=mesh)
        ds.add_1d_mesh_line(mesh=mesh, pos=0.0, ps=5e-9, tag="lo")
        ds.add_1d_mesh_line(mesh=mesh, pos=L - 20e-9, ps=h)
        ds.add_1d_mesh_line(mesh=mesh, pos=L, ps=h, tag="mid")
        ds.add_1d_mesh_line(mesh=mesh, pos=L + 20e-9, ps=h)
        ds.add_1d_mesh_line(mesh=mesh, pos=2 * L, ps=5e-9, tag="hi")
        ds.add_1d_contact(mesh=mesh, name="lo", tag="lo", material="metal")
        ds.add_1d_contact(mesh=mesh, name="hi", tag="hi", material="metal")
        ds.add_1d_region(mesh=mesh, material="A", region="r0", tag1="lo", tag2="mid")
        ds.add_1d_region(mesh=mesh, material="B", region="r1", tag1="mid", tag2="hi")
        ds.add_1d_interface(mesh=mesh, name="jn", tag="mid")
        ds.finalize_mesh(mesh=mesh)
        ds.create_device(mesh=mesh, device=dev)
        kw = {} if opts.get("stats") is None else {"statistics": opts["stats"]}
        for r, N in (("r0", N0), ("r1", N1)):
            ds.node_model(device=dev, region=r, name="NetDoping", equation=repr(N))
            BP.setup_bipolar_region(dev, r, eps_static=11.7, n_dos_m3=NC, n_dos_p_m3=NV, n_i_m3=n_i,
                                    mobility_n_m2Vs=MU_N, mobility_p_m2Vs=MU_P, tau_n_s=1e-6, tau_p_s=1e-6,
                                    fd_enhancement=fd, **kw)
            BP.setup_equilibrium_seed_models(dev, r)
        own = {r: float(VT_CODE * math.log(NC / n_i)) for r in ("r0", "r1")}
        fr = HJ.anderson_alignment(own, {"r0": 4.0, "r1": 4.0 - float(opts.get("dchi", 0.1))}, "r0")
        a_star = float(opts.get("astar", 1e-2)) * HJ.richardson_constant_Am2K2(M_E)
        sides = [HJ.HeteroSide(r, fr[r][0], "N_dos", a_star, fd=fd,
                               phi_v_V=fr[r][0] - float(VT_CODE * math.log(NC * NV / n_i ** 2)),
                               p_dos_param="N_dos_p", richardson_p=a_star) for r in ("r0", "r1")]
        PE.setup_interface(dev, "jn")
        HJ.setup_heterointerface_carriers(dev, "jn", sides[0], sides[1], model="thermionic")
        BP.setup_contact_ohmic_bipolar(dev, "lo", psi_offset_V=fr["r0"][1])
        BP.setup_contact_ohmic_bipolar_circuit(dev, "hi", node_name="vhi", source_name="V1",
                                               psi_offset_V=fr["r1"][1])
        for r in ("r0", "r1"):
            ds.node_model(device=dev, region=r, name="_seed", equation="{} + ({!r})".format(
                BP.equilibrium_seed_psi_expr(BP.fd_shift_model(dev, r)), float(fr[r][1])))
            for v, m in (("Potential", "_seed"), ("Electrons", "IntrinsicElectrons"),
                         ("Holes", "IntrinsicHoles")):
                ds.set_node_values(device=dev, region=r, name=v,
                                   values=ds.get_node_model_values(device=dev, region=r, name=m))
    return dev


def build_hetero_unipolar(dev, opts):
    """1-D UNIPOLAR (physics_drift_diffusion) FD isotype heterostack, ITO-like N0 = 3e26 (eta 20) / AZO-like N1 m^-3,
    affinity step dchi, thermionic interface; opts["edge"] "consistent" = statistics="fermi_dirac"; 'lo' the circuit contact (V1) on r0."""
    import validation.devsim_noise as V
    ds = _ds()
    from dynameta.carriers import eq_registry as _R
    from dynameta.carriers import physics_drift_diffusion as DD
    from dynameta.carriers import physics_equilibrium as PE
    from dynameta.carriers import physics_heterojunction as HJ
    from dynameta.carriers.physics_bipolar_dd import fd_potential_shift
    from dynameta.constants import M_E, V_T as VT_CODE
    V.fresh()
    N0, N1 = float(opts.get("N0", 3e26)), float(opts.get("N1", 3e25))
    L, h = 100e-9, float(opts.get("h", 0.5e-9))
    mesh = dev + "_m"
    _R.clear(dev)
    with _quiet():
        ds.create_1d_mesh(mesh=mesh)
        ds.add_1d_mesh_line(mesh=mesh, pos=0.0, ps=2e-9, tag="lo")
        ds.add_1d_mesh_line(mesh=mesh, pos=L - 10e-9, ps=h)
        ds.add_1d_mesh_line(mesh=mesh, pos=L, ps=h, tag="mid")
        ds.add_1d_mesh_line(mesh=mesh, pos=L + 10e-9, ps=h)
        ds.add_1d_mesh_line(mesh=mesh, pos=2 * L, ps=2e-9, tag="hi")
        ds.add_1d_contact(mesh=mesh, name="lo", tag="lo", material="metal")
        ds.add_1d_contact(mesh=mesh, name="hi", tag="hi", material="metal")
        ds.add_1d_region(mesh=mesh, material="A", region="r0", tag1="lo", tag2="mid")
        ds.add_1d_region(mesh=mesh, material="B", region="r1", tag1="mid", tag2="hi")
        ds.add_1d_interface(mesh=mesh, name="jn", tag="mid")
        ds.finalize_mesh(mesh=mesh)
        ds.create_device(mesh=mesh, device=dev)
        own = {}
        kw = {} if opts.get("edge", "average") == "average" else {"statistics": "fermi_dirac"}
        for r, N, m in (("r0", N0, 0.35), ("r1", N1, 0.28)):
            DD.setup_semiconductor_region_dd(dev, r, n_bg_m3=N, eps_static=9.0, dos_mass_kg=m * M_E,
                                             mobility_m2Vs=3e-3, **kw)
            Nc = float(ds.get_parameter(device=dev, region=r, name="N_c"))
            own[r] = -VT_CODE * (math.log(N / Nc) + float(fd_potential_shift(N / Nc)))
        fr = HJ.anderson_alignment(own, {"r0": 4.4, "r1": 4.4 - float(opts.get("dchi", 0.3))}, "r0")
        a_star = float(opts.get("astar", 1e-2)) * HJ.richardson_constant_Am2K2(M_E)
        sides = [HJ.HeteroSide(r, fr[r][0], "N_c", a_star, fd=True) for r in ("r0", "r1")]
        PE.setup_interface(dev, "jn")
        HJ.setup_heterointerface_carriers(dev, "jn", sides[0], sides[1], model="thermionic")
        DD.setup_contact_ohmic_dd_circuit(dev, "lo", node_name="vhi", source_name="V1")
        DD.setup_contact_ohmic_dd(dev, "hi", psi_offset_V=fr["r1"][1])
        for r in ("r0", "r1"):
            nn = len(ds.get_node_model_values(device=dev, region=r, name="x"))
            ds.set_node_values(device=dev, region=r, name="Potential", values=[float(fr[r][1])] * nn)
    return dev


def sc_fdt_hetero(opts):
    """FDT of the thermionic heterostacks (bipolar build_hetero, or unipolar build_hetero_unipolar)."""
    if opts.get("unipolar"):
        build_hetero_unipolar("hu", opts)
        ae = 1e14
    else:
        build_hetero("hb", opts)
        ae = 1e13
    r, share, ReY = fdt_ratio(ae, v0=float(opts.get("v0", 1e-6 if opts.get("unipolar") else 1e-7)))
    return {"fdt": r, "share": share, "ReY": ReY}


def sc_zero_bias_hetero(opts):
    """The heterostack's zero-bias source current (both contacts at 0 V; the battery of the transport) and
    its conductance, so I0/G is the effective zero-bias voltage."""
    import validation.devsim_noise as V
    ds = _ds()
    if opts.get("unipolar"):
        build_hetero_unipolar("hu", opts)
        ae = 1e14
    else:
        build_hetero("hb", opts)
        ae = 1e13
    cur = {}
    for v in (1e-4, -1e-4, 2e-4, -2e-4):
        ds.circuit_alter(name="V1", value=v)
        _solve_circuit(ae)
        cur[v] = V.source_current()
    G = (cur[1e-4] - cur[-1e-4]) / 2e-4
    a1, a2 = 0.5 * (cur[1e-4] + cur[-1e-4]), 0.5 * (cur[2e-4] + cur[-2e-4])
    I0 = (4.0 * a1 - a2) / 3.0                       # I(0): the dv^2 curvature term removed (Richardson)
    return {"G": G, "I0": I0, "V_battery": I0 / G, "I0_plain": a1}


def sc_nondeg(opts):
    """Non-degenerate limit: one forward / reverse operating point per mechanism, for the comparison of the
    statistics at low doping. Returns the solved terminal quantities: zener -> btbt current at -0.6 V
    (q Int G), diode -> trap recombination current at +0.3 V forward, mos -> surface recombination at
    vg (interface)."""
    from dynameta.carriers.tunneling import btbt_generation_current
    ds = _ds()
    out = {}
    if opts.get("device") == "mos":
        dev, _ = build_mos("nd", opts)
        net, one = mechanism_balance(dev, "it")
        out["value"] = net
        out["one_way"] = one
    elif opts.get("device") == "diode":
        dev = build_junction("nd", opts)
        for k in range(1, 7):
            ds.set_parameter(device=dev, name="nc_bias", value=-0.05 * k)
            solve()
        out["value"] = mechanism_balance(dev, "trap")[0]
    else:
        dev = build_junction("nd", opts)
        for k in range(1, 7):
            ds.set_parameter(device=dev, name="pc_bias", value=-0.1 * k)
            solve()
        out["value"] = btbt_generation_current(dev, "b")
    from dynameta.carriers.physics_bipolar_dd import fd_potential_shift
    n = _v(dev, "Electrons")
    p = _v(dev, "Holes")
    out["delta_max"] = float(max(np.max(fd_potential_shift(n / NC)), np.max(fd_potential_shift(p / NV))))
    out["residual"] = residual_scale(dev)
    return out


def sc_nonlocal_refusals(opts):
    """setup_btbt_nonlocal on statistics='fermi_dirac' without BGN (accepted), with BGN (refused), and on the
    legacy statistics with BGN (refused)."""
    from dynameta.carriers import btbt_nonlocal as NL
    from dynameta.carriers.tunneling import SILICON_HURKX
    out = {}
    for tag, o in (("fd_plain", {"stats": "fermi_dirac"}), ("fd_bgn", {"stats": "fermi_dirac", "bgn": 1}),
                   ("legacy_bgn", {"bgn": 1})):
        ds = _ds()
        from dynameta.carriers import eq_registry as _R
        dev, mesh = "nr" + tag, "nr" + tag + "_m"
        _R.clear(dev)
        with _quiet():
            _lines_1d(ds, mesh, 20e-9, 10e-9, 0.5e-9, 2e-9, 1e-9)
            ds.add_1d_contact(mesh=mesh, name="pc", tag="pc", material="metal")
            ds.add_1d_contact(mesh=mesh, name="nc", tag="nc", material="metal")
            ds.add_1d_region(mesh=mesh, material="Si", region="b", tag1="pc", tag2="nc")
            ds.finalize_mesh(mesh=mesh)
            ds.create_device(mesh=mesh, device=dev)
            ds.node_model(device=dev, region="b", name="NetDoping", equation="ifelse(x < 1e-8, -1e24, 1e24)")
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")               # the BGN basis advisory (not under test)
                _setup_region(dev, "b", o, 1e16)
        try:
            NL.setup_btbt_nonlocal(dev, "b", NL.NonlocalBTBT(SILICON_HURKX, Eg_tun_eV=1.12))
            out[tag] = "ok"
        except NotImplementedError as e:
            out[tag] = str(e)
    return out


def _transport_markers(dev, region):
    ds = _ds()
    edges = set(ds.get_edge_model_list(device=dev, region=region))
    g = np.asarray(ds.get_edge_model_values(device=dev, region=region, name="g_enh"))
    return {"g_is_one": bool(np.all(g == 1.0)), "quasi_field": "HJ_Electrons_dDelta" in edges}


def sc_builders(opts):
    """TransportModel.carrier_statistics threaded to the UNIPOLAR drift-diffusion regions of both builders:
    the 2-D LayeredDevsimBuilder (a 40 nm n 3e25 m^-3 bar, solved at 0 V) and the 3-D Devsim3DEquilibrium
    (a small gated cell, built only). Returns, per statistics, the transport markers (g_enh == 1 and the
    quasi-field edge models) of every unipolar region."""
    import warnings
    from dynameta.carriers.devsim_3d import Devsim3DEquilibrium, Stacked3DSpec
    from dynameta.carriers.devsim_layered import LayeredDevsimBuilder
    from dynameta.carriers.physics_equilibrium import M_E
    from dynameta.geometry import Design, Electrode, Layer, Stack, UnitCell
    from dynameta.geometry.specs import Mesh2DSpec
    from dynameta.materials import ConstantOptical, Material, MaterialRegistry
    from dynameta.materials.transport_model import TransportModel
    from dynameta.sweep import BiasPoint
    ds = _ds()
    out = {}
    for tag, st in (("legacy", None), ("fd", "fermi_dirac")):
        tr = TransportModel(n_bg_m3=3e25, eps_static=9.0, dos_mass_kg_of_n_m3=lambda n: 0.3 * M_E,
                            physics="drift_diffusion", mobility_m2Vs_of_n_m3=lambda n: 3e-3,
                            carrier_statistics=st)
        reg = MaterialRegistry()
        reg.add(Material("air", ConstantOptical(1.0 + 0j)))
        reg.add(Material("cmetal", ConstantOptical(-50.0 + 5.0j), is_metal=True))
        reg.add(Material("TCO", ConstantOptical(4.0 + 0j), transport=tr))
        stack = Stack(layers=[Layer("tco", 40e-9, "TCO")], superstrate_material="air", substrate_material="air")
        els = [Electrode("anode", "tco", "x_lo", role="biased"),
               Electrode("cathode", "tco", "x_hi", role="ground", fixed_voltage_V=0.0)]
        d = Design(name="ub" + tag, unit_cell=UnitCell.square(300e-9), stack=stack, electrodes=els, materials=reg,
                   mesh_2d=Mesh2DSpec())
        b = LayeredDevsimBuilder(d, mesh_name="ubm" + tag, device_name="ubd" + tag)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")              # builder advisories (not under test)
            with _quiet():
                b.solve(BiasPoint({"anode": 0.0}, "eq"))
        o = _transport_markers(b.device, "tco")
        ds.delete_device(device=b.device)
        spec = Stacked3DSpec(semi_material="ITO", oxide_material="HfO2", lateral_m=12e-9, semi_thk_m=12e-9,
                             oxide_thk_m=8e-9, n_bg_m3=4e26, eps_semi=9.5, eps_oxide=18.0, dos_mass_kg=0.35 * M_E,
                             physics="drift_diffusion", carrier_statistics=st,
                             chi_semi_eV=4.4, extra_semiconductors=[("xs", "AZO", 5e-9, 9.0, 1e26, 4.3)])
        c = Devsim3DEquilibrium(spec, mesh_name="u3m" + tag, device_name="u3d" + tag)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with _quiet():
                c.build_device()
        o["3d_semi"] = _transport_markers(c.device, "semi")
        o["3d_extra"] = _transport_markers(c.device, "xs")
        out[tag] = o
    return out


def sc_dg_fd(opts):
    """Bipolar DENSITY GRADIENT (both carriers quantum at an insulator wall, quantum_srh) on the strongly
    inverted MOS of validation.bipolar_density_gradient GATE E, with the FD transport (fd=True) under
    opts['stats'] (None = the v0.15.0 default 'fermi_dirac'; 'legacy' injected into setup_bipolar_region).
    Returns the FD modified quasi-Fermi levels phi_n = psi + Lambda_n - V_t (L_n + ln(N_c/n_i)),
    phi_p = psi - Lambda_p + V_t (L_p + ln(N_v/n_i)) (L = ln u + Delta(u), the transport's activity; their
    max |.| -- flat at 0 = the body contact's level at an exact equilibrium), the detailed-balance residual
    |L_n + L_p - ln(n_i^2/(N_c N_v)) - (Lambda_n + Lambda_p)/V_t|, the degeneracy max Delta, and -- with
    opts['jac'] -- the Jacobian check (all equation variables, the 4 nodes nearest the wall, electrons
    kicked x1.2 off the solved state)."""
    import validation.bipolar_density_gradient as V
    from dynameta.carriers import physics_bipolar_dd as BP
    from dynameta.carriers.physics_bipolar_dd import fd_potential_shift
    ds = _ds()
    real = BP.setup_bipolar_region
    if opts.get("stats"):
        def patched(*a, **k):
            k.setdefault("statistics", opts["stats"])
            return real(*a, **k)
        BP.setup_bipolar_region = patched
    keep = {}
    if opts.get("jac"):
        V._drop = lambda dev, mesh: keep.setdefault("dev", dev) and None
    N_A, delta = float(opts.get("N_A", 3e23)), float(opts.get("delta", 0.6))
    r = V.mos("holes", N_A, delta, fd=True, length=150e-9, quantum_srh=True, steps=16, rel=1e-6)
    nc, nv = V.nc_model(V.MN, V.GVN), V.nc_model(V.MP, V.GVP)
    psi, n, p, ln, lp = (r[k] for k in ("Potential", "Electrons", "Holes", "QLambda", "QLambdaP"))
    Ln = np.log(n / nc) + np.asarray(fd_potential_shift(n / nc), float)
    Lp = np.log(p / nv) + np.asarray(fd_potential_shift(p / nv), float)
    vt = V.V_T
    phi_n = psi + ln - vt * (Ln + math.log(nc / V.NI_SI))
    phi_p = psi - lp + vt * (Lp + math.log(nv / V.NI_SI))
    db = np.abs(Ln + Lp - math.log(V.NI_SI ** 2 / (nc * nv)) - (ln + lp) / vt)
    out = {"max_phi_n": float(np.max(np.abs(phi_n))), "max_phi_p": float(np.max(np.abs(phi_p))),
           "db": float(np.max(db)),
           "delta_max": float(max(np.max(fd_potential_shift(n / nc)), np.max(fd_potential_shift(p / nv)))),
           "lam_wall_n_Vt": float(ln[0] / vt)}
    if opts.get("jac"):
        dev = keep["dev"]
        reg = ds.get_region_list(device=dev)[0]
        variables = [ds.get_equation_command(device=dev, region=reg, name=e)["variable_name"]
                     for e in ds.get_equation_list(device=dev, region=reg)]
        ds.set_node_values(device=dev, region=reg, name="Electrons", values=_lst(1.2 * _v(dev, "Electrons", reg)))
        x = _v(dev, "x", reg)
        nodes = [int(i) for i in np.argsort(x)[1:5]]
        out["variables"] = variables
        out["jac_worst"], out["jac_where"] = jac_check(dev, reg, nodes, variables=tuple(variables))
        if opts.get("knockout"):
            _knock_out_fd_derivatives(dev, reg)
            out["ko_worst"], out["ko_where"] = jac_check(dev, reg, nodes, variables=tuple(variables))
    return out


SCENARIOS = {"zero_bias": sc_zero_bias, "jacobian": sc_jacobian, "jacobian_2d": sc_jacobian_2d,
             "jacobian_nonlocal": sc_jacobian_nonlocal, "fdt": sc_fdt, "fdt_hetero": sc_fdt_hetero,
             "zero_bias_hetero": sc_zero_bias_hetero, "nondeg": sc_nondeg,
             "nonlocal_refusals": sc_nonlocal_refusals, "dc_kinetic": sc_dc_kinetic_vs_quasi_static,
             "builders": sc_builders, "dg_fd": sc_dg_fd}


def run_child(scenario, **opts):
    """Run `scenario` in a fresh interpreter on this sys.path; returns its JSON result."""
    import subprocess
    import tempfile
    fd, out = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    try:
        env = dict(os.environ, PYTHONPATH=os.pathsep.join(p for p in sys.path if isinstance(p, str) and p))
        r = subprocess.run([sys.executable, os.path.abspath(__file__), scenario, out, json.dumps(opts)],
                           env=env, capture_output=True, text=True, timeout=1200)
        if r.returncode != 0:
            raise RuntimeError("child {} failed:\n{}".format(scenario, r.stderr[-4000:]))
        with open(out, encoding="utf-8") as f:
            return json.load(f)
    finally:
        os.remove(out)


if __name__ == "__main__":
    import warnings
    warnings.simplefilter("error")                       # warnings are errors, as in the suite
    if not os.environ.get("FDM_NO_HOOK"):                # the suite's DC well-posedness hook (tests/conftest.py)
        import conftest
        _ds()
        conftest._install_dc_well_posedness_hook()
    _opts = json.loads(sys.argv[3]) if len(sys.argv) > 3 else {}
    _res = SCENARIOS[sys.argv[1]](_opts)
    with open(sys.argv[2], "w", encoding="utf-8") as _f:
        json.dump(_res, _f)
