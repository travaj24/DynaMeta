"""Child-process scenarios for tests/test_custom_terms.py (DynaMeta v0.15, plan item 4.0.3).

WHY A CHILD. DEVSIM has no delete_custom_equation: once the custom_terms dispatcher is registered it is
process-global and permanent (MEASURED, devsim 2.11.2; P1 probe), and DEVSIM circuits cannot be removed
in-process either (tests/_bipolar_calllog.py). So every scenario that registers runs here, in a fresh
interpreter on the caller's sys.path, and reports plain JSON (floats round-trip exactly through repr).

    python tests/_custom_terms_child.py <scenario> <out.json> [options]

Toy terms (closed forms re-derived in tests/test_custom_terms.py, never read from this module's results):
  PoissonLink  a NONLOCAL linear coupling of the PotentialEquation rows of two interior nodes a, b of a
               1-D dielectric, r_a += s_a G (psi_a - psi_b), r_b += s_b G (psi_a - psi_b) with
               (s_a, s_b) = (+1, -1): exactly an extra flux link of "capacitance" G [F/m^2] in parallel
               with the a-b segment (DEVSIM's PotentialEdgeFlux adds +eps (psi_0 - psi_1)/L to node 0).
               (+1, +1) is the deliberate wrong sign.
  QFLBridge    an electron transfer between two interior nodes of a 1-D n-type resistor driven by the
               electron quasi-Fermi potential difference, I = G (phi_a - phi_b), phi = psi - V_t ln n,
               r_a += s_a I, r_b += s_b I (removing electrons at a: ElectronGeneration = -q U). Its noise
               is the two-way Poisson stream of the linear conductance at equilibrium, P = 2 kT G / q^2
               (detailed balance), rows (n eq, a, -1), (n eq, b, +1) -- a genuinely two-point source.
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

from dynameta.carriers import custom_terms as CT  # noqa: E402
from dynameta.carriers.custom_terms import CustomTerm  # noqa: E402

EPS_R = 3.9
EPS0 = 8.8541878128e-12
L_OX, H_OX, XA, XB = 100e-9, 2e-9, 30e-9, 70e-9          # toy A: dielectric bar, nodes a, b
L_RES, H_RES, XA_R, XB_R = 1e-6, 2e-8, 0.3e-6, 0.7e-6   # toy B: n-type resistor
ND_RES = 1e22


def _ds():
    from dynameta.carriers.devsim_env import import_devsim
    return import_devsim()


def _node_at(dev, region, x0):
    ds = _ds()
    x = np.asarray(ds.get_node_model_values(device=dev, region=region, name="x"))
    i = int(np.argmin(np.abs(x - x0)))
    assert abs(x[i] - x0) < 1e-15, (x[i], x0)
    return i


# =====================================================================================================
# toy terms
# =====================================================================================================

class PoissonLink(CustomTerm):
    locality = "nonlocal"

    def __init__(self, name, region, a, b, G, signs=(1.0, -1.0), pattern_extra=(), escape=False):
        super().__init__(name)
        self.region, self.a, self.b, self.G, self.signs = region, a, b, G, signs
        self.pattern_extra, self.escape = tuple(pattern_extra), escape

    def equations(self, device):
        return [(self.region, "PotentialEquation")]

    def _rc(self, device):
        rows = CT.node_rows(device, self.region, "PotentialEquation", [self.a, self.b])
        cols = CT.node_columns(device, self.region, "Potential", [self.a, self.b])
        return rows, cols

    def structural_pattern(self, device):
        if not self.pattern_extra and not self.escape:
            return None
        rows, cols = self._rc(device)
        pat = [(int(r), int(c)) for r in rows for c in cols]
        for (i, j) in self.pattern_extra:
            pat.append((int(CT.node_rows(device, self.region, "PotentialEquation", [i])[0]),
                        int(CT.node_columns(device, self.region, "Potential", [j])[0])))
        return pat[:-1] if self.escape else pat              # escape: the last real entry undeclared

    active = True

    def assemble(self, device, what, timemode):
        if timemode != "DC" or not self.active:
            return [], []
        ds = _ds()
        (ra, rb), (ca, cb) = self._rc(device)
        psi = ds.get_node_model_values(device=device, region=self.region, name="Potential")
        flux = self.G * (psi[self.a] - psi[self.b])
        sa, sb = self.signs
        rcv = []
        for r, s in ((ra, sa), (rb, sb)):
            rcv += [int(r), int(ca), s * self.G, int(r), int(cb), -s * self.G]
        return rcv, [int(ra), sa * flux, int(rb), sb * flux]


class QFLBridge(CustomTerm):
    locality = "nonlocal"

    def __init__(self, name, region, a, b, G, signs=(1.0, -1.0), noise=True, mechanism="tunnel_ox",
                 jacobian_exact=True):
        super().__init__(name)
        self.region, self.a, self.b, self.G, self.signs = region, a, b, G, signs
        self.noise, self.mechanism, self.jacobian_exact = noise, mechanism, jacobian_exact

    def equations(self, device):
        return [(self.region, "ElectronContinuityEquation")]

    active = True

    def assemble(self, device, what, timemode):
        if timemode != "DC" or not self.active:
            return [], []
        ds = _ds()
        reg = self.region
        ra, rb = CT.node_rows(device, reg, "ElectronContinuityEquation", [self.a, self.b])
        pa, pb = CT.node_columns(device, reg, "Potential", [self.a, self.b])
        na, nb = CT.node_columns(device, reg, "Electrons", [self.a, self.b])
        psi = ds.get_node_model_values(device=device, region=reg, name="Potential")
        n = ds.get_node_model_values(device=device, region=reg, name="Electrons")
        vt = float(ds.get_parameter(device=device, region=reg, name="V_t"))
        phi_a = psi[self.a] - vt * math.log(n[self.a])
        phi_b = psi[self.b] - vt * math.log(n[self.b])
        cur = self.G * (phi_a - phi_b)
        d = {int(pa): self.G, int(na): -self.G * vt / n[self.a],
             int(pb): -self.G, int(nb): self.G * vt / n[self.b]}
        rcv = []
        for r, s in ((ra, self.signs[0]), (rb, self.signs[1])):
            for c, v in d.items():
                rcv += [int(r), c, s * v]
        return rcv, [int(ra), self.signs[0] * cur, int(rb), self.signs[1] * cur]

    def noise_sources(self, device):
        if not self.noise:
            return None
        from scipy.constants import e as QE, k as KB
        from dynameta.carriers.noise_analysis import SourceRow, TwoPointSource
        ds = _ds()
        T = float(ds.get_parameter(device=device, region=self.region, name="V_t")) * QE / KB
        P = np.array([2.0 * KB * T * self.G / QE ** 2])
        eq = "ElectronContinuityEquation"
        return [TwoPointSource(self.mechanism, self.region, P,
                               (SourceRow(self.region, eq, np.array([-1.0]), np.array([self.a])),
                                SourceRow(self.region, eq, np.array([1.0]), np.array([self.b]))))]


# =====================================================================================================
# devices
# =====================================================================================================

def _quiet():
    import validation.devsim_noise as V
    return V.quiet()


def build_dielectric(dev="tA", region="ox", extended=False):
    """1-D dielectric bar [0, L_OX], uniform h = H_OX: 'lo' grounded, 'hi' on circuit node 'va'
    driven by source 'VA'."""
    ds = _ds()
    from dynameta.carriers import ac_analysis as AC
    from dynameta.carriers import eq_registry as _R
    from dynameta.carriers import extended_precision as XP
    from dynameta.carriers import physics_equilibrium as PE
    mesh = dev + "_m"
    _R.clear(dev)
    with _quiet():
        ds.create_1d_mesh(mesh=mesh)
        ds.add_1d_mesh_line(mesh=mesh, pos=0.0, ps=H_OX, tag="lo")
        ds.add_1d_mesh_line(mesh=mesh, pos=L_OX, ps=H_OX, tag="hi")
        ds.add_1d_contact(mesh=mesh, name="lo", tag="lo", material="metal")
        ds.add_1d_contact(mesh=mesh, name="hi", tag="hi", material="metal")
        ds.add_1d_region(mesh=mesh, material="SiO2", region=region, tag1="lo", tag2="hi")
        ds.finalize_mesh(mesh=mesh)
        ds.create_device(mesh=mesh, device=dev)
        if extended:
            XP.enable(dev)
        PE.setup_dielectric_region(dev, region, EPS_R)
        PE.setup_contact(dev, "lo")
        if "va" not in ds.get_circuit_node_list():
            AC.setup_circuit_contact(dev, "hi", source_name="VA", node_name="va")
        else:                                            # circuit already exists (re-created device)
            cn = "hi_circuit_dirichlet"
            ds.contact_node_model(device=dev, contact="hi", name=cn, equation="Potential - va")
            ds.contact_node_model(device=dev, contact="hi", name=cn + ":Potential", equation="1")
            ds.contact_node_model(device=dev, contact="hi", name=cn + ":va", equation="-1")
            _R.record_contact_equation(dev, "hi", name="PotentialEquation", node_model=cn,
                                       edge_charge_model="PotentialEdgeFlux", circuit_node="va")
    return dev


def delete_device(dev):
    ds = _ds()
    from dynameta.carriers import eq_registry as _R
    ds.delete_device(device=dev)
    ds.delete_mesh(mesh=dev + "_m")
    _R.clear(dev)


def solve_dc(ae=1e-8, re_=1e-13, it=40):
    ds = _ds()
    with _quiet():
        ds.solve(type="dc", absolute_error=ae, relative_error=re_, maximum_iterations=it)


def matrix(dev=None):
    ds = _ds()
    with _quiet():
        m = ds.get_matrix_and_rhs(format="csc")["static"]
    return (np.asarray(m["av"], dtype=float), np.asarray(m["ai"]), np.asarray(m["ap"]),
            np.asarray(m["rhs"], dtype=float))


def _lst(a):
    return [float(x) for x in np.ravel(a)]


def _device_rows():
    """1 + the largest equation number of any device equation of any live device (the circuit rows follow)."""
    ds = _ds()
    top = -1
    for d in ds.get_device_list():
        for r in ds.get_region_list(device=d):
            for eq in ds.get_equation_list(device=d, region=r):
                v = ds.get_equation_command(device=d, region=r, name=eq)["variable_name"]
                c = np.asarray(ds.get_equation_numbers(device=d, region=r, variable=v))
                if c.size:
                    top = max(top, int(c.max()))
    return top + 1


def fd_columns(dev, region, var, nodes, rows, h, term):
    """The custom term's OWN Jacobian share J_on - J_off at `rows` x (var at `nodes`) against central
    differences (steps h and 2h) of its own residual share rhs_on - rhs_off (get_matrix_and_rhs returns
    rhs = -F). Toggling term.active at the same state isolates the term: DEVSIM's own contributions are
    recomputed identically. The round-off scale of a row's residual, R = sum_k |J_on[r, k]| |x_k| (the
    magnitude of the terms the row sums -- each linearised term J x), is returned as rhs_scale."""
    ds = _ds()
    import scipy.sparse as sp

    def both():
        term.active = False
        off = matrix()
        term.active = True
        on = matrix()
        return on, off
    on, off = both()
    Jon = sp.csc_matrix((on[0], on[1], on[2]))
    Jc = Jon - sp.csc_matrix((off[0], off[1], off[2]))
    x = np.zeros(Jon.shape[1])                            # the solution vector, column by column
    # the region's SOLUTION variables, read from its equation commands (not a scan of every node model)
    for eq in ds.get_equation_list(device=dev, region=region):
        v = ds.get_equation_command(device=dev, region=region, name=eq)["variable_name"]
        c = np.asarray(ds.get_equation_numbers(device=dev, region=region, variable=v))
        ok = c >= 0
        x[c[ok]] = np.asarray(ds.get_node_model_values(device=dev, region=region, name=v))[ok]
    # circuit nodes ONLY if the process has a circuit: DEVSIM 2.11.2's get_circuit_node_list() in a process
    # with no circuit trips the FATAL assertion NodeKeeper.cc:266 CIRCUIT_UNEXPECTED, reported by the NEXT
    # DEVSIM call (MEASURED 2026-11-06 on the circuit-free 1-D Zener of tests/_btbt_nonlocal_child.py; the
    # toy devices here all have a circuit, so it never showed). The circuit rows are the matrix rows beyond
    # every device equation.
    if Jon.shape[0] > _device_rows():
        for cn in ds.get_circuit_node_list():
            x[CT.circuit_row(cn)] = float(ds.get_circuit_node_value(node=cn, solution="dcop"))
    R = float(np.max(np.abs(Jon[rows, :]) @ np.abs(x)))
    cols = CT.node_columns(dev, region, var, nodes)
    out = []
    for node, col in zip(nodes, cols):
        x0 = list(ds.get_node_model_values(device=dev, region=region, name=var))[node]
        hh = h(x0)
        rec = {"node": int(node), "var": var, "h": hh, "J": _lst(Jc[rows, int(col)].toarray())}
        for tag, step in (("fd", hh), ("fd2", 2.0 * hh)):
            share = []
            for sgn in (+1.0, -1.0):
                ds.set_node_value(device=dev, region=region, name=var, index=int(node),
                                  value=x0 + sgn * step)
                on, off = both()
                share.append(on[3][rows] - off[3][rows])
            rec[tag] = _lst((share[0] - share[1]) / (2.0 * step))
        ds.set_node_value(device=dev, region=region, name=var, index=int(node), value=x0)
        rec["rhs_scale"] = R
        out.append(rec)
    return out


class _NoPattern(CustomTerm):
    locality = "nonlocal"
    pattern_changes = True

    def assemble(self, device, what, timemode):
        return [], []


class _WithPattern(_NoPattern):
    def structural_pattern(self, device):
        return []


def sc_symbolic(opts):
    """The DEVSIM_NEW_SYMBOLIC advisory: a pattern-changing term that does not declare a structural
    pattern warns under DEVSIM_NEW_SYMBOLIC=0; one that declares it, or the default '1', does not."""
    import warnings
    dev = build_dielectric()
    out = {}
    for tag, env, cls in (("zero_nopattern", "0", _NoPattern), ("zero_pattern", "0", _WithPattern),
                          ("one_nopattern", "1", _NoPattern)):
        os.environ["DEVSIM_NEW_SYMBOLIC"] = env
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            CT.add_custom_term(dev, cls("t_" + tag))
        out[tag] = [(type(x.message).__name__, str(x.message)) for x in w]
    return out


# =====================================================================================================
# scenarios
# =====================================================================================================

def sc_import(opts):
    """Importing every carriers module registers nothing; the FIRST add registers exactly once."""
    import importlib
    import pkgutil
    import dynameta.carriers as C
    ds = _ds()
    calls = []
    real = ds.custom_equation

    def spy(**kw):
        calls.append(kw["name"])
        return real(**kw)
    ds.custom_equation = spy
    mods, skipped = [], []
    for m in pkgutil.iter_modules(C.__path__):
        try:
            importlib.import_module("dynameta.carriers." + m.name)
            mods.append(m.name)
        except ImportError as e:
            skipped.append("{}: {}".format(m.name, e))
    res = {"modules": mods, "skipped": skipped, "after_import": [CT.is_registered(), list(calls)]}
    dev = build_dielectric()
    solve_dc()
    res["after_build_solve"] = [CT.is_registered(), list(calls)]
    a, b = _node_at(dev, "ox", XA), _node_at(dev, "ox", XB)
    CT.add_custom_term(dev, PoissonLink("link", "ox", a, b, 1e-6))
    res["after_first_add"] = [CT.is_registered(), list(calls)]
    CT.add_custom_term(dev, PoissonLink("link2", "ox", a, b, 1e-6))
    res["after_second_add"] = [CT.is_registered(), list(calls)]
    return res


def sc_defaults(opts):
    """The defaults-off fixtures (tests/_sgflux_reference.py) + the BTBT FDT diode noise / ssac,
    optionally with the dispatcher registered FIRST (and no term)."""
    if opts.get("register"):
        CT.register_dispatcher()
    import _sgflux_reference as REF
    res = {"registered": CT.is_registered(), "sgflux": REF.solve_fixtures(statistics="legacy")}   # v0.15.0
    import validation.devsim_noise as V
    from dynameta.carriers.tunneling import BTBTParams
    ff = [1.0, 1e5, 1e7, 1e9]
    V.bipolar_1d("fdt", *V._fdt_lines(), n_i=V.FDT_NI, tau=V.FDT_TAU, ae=1e8,
                 btbt=BTBTParams(A_SI=1.0e12, B_V_per_m=1.0e7, P=2.5))
    sn = V.noise(ff)
    Y = V.ssac_Y(ff)
    res["fdt"] = {"S": _lst(sn.S), "gr": _lst(sn.by_mechanism["gr"]), "ReY": _lst(Y.real),
                  "ImY": _lst(Y.imag),
                  "psi": _lst(V.ds.get_node_model_values(device="fdt", region="bulk", name="Potential"))}
    res["dispatch_calls"] = CT._dispatch_calls
    return res


def _toy_a_state(dev, region="ox"):
    ds = _ds()
    return {"psi": _lst(ds.get_node_model_values(device=dev, region=region, name="Potential")),
            "Q_lo": float(ds.get_contact_charge(device=dev, contact="lo", equation="PotentialEquation")),
            "Q_hi": float(ds.get_contact_charge(device=dev, contact="hi", equation="PotentialEquation"))}


def sc_toy_a(opts):
    """Toy A on the dielectric bar: DC at V, Jacobian FD, ssac C, conservation; wrong sign and the
    extended-precision Newton by option."""
    ds = _ds()
    from dynameta.carriers import ac_analysis as AC
    V0 = 1.0
    G = EPS_R * EPS0 / (XB - XA)                           # doubles the a-b segment's "capacitance"
    signs = tuple(opts.get("signs", (1.0, -1.0)))
    extended = bool(opts.get("extended"))
    dev = build_dielectric(extended=extended)
    a, b = _node_at(dev, "ox", XA), _node_at(dev, "ox", XB)
    ds.circuit_alter(name="VA", value=V0)
    solve_dc()
    res = {"a": a, "b": b, "G": G, "V": V0, "off": _toy_a_state(dev),
           "eps": float(ds.get_parameter(device=dev, region="ox", name="Permittivity")),
           "x": _lst(ds.get_node_model_values(device=dev, region="ox", name="x"))}
    term = CT.add_custom_term(dev, PoissonLink("link", "ox", a, b, G, signs=signs))
    if extended:                                       # the 128-bit Newton with a double-valued term
        from dynameta.carriers import extended_precision as XP
        with _quiet():
            info = XP.solve(type="dc", absolute_error=1e-30, relative_error=float(opts["rel"]),
                            maximum_iterations=60, info=True)
        res["extended_converged"] = bool(info["converged"])
        res["extended_rel"] = [float(it["devices"][0]["relative_error"]) for it in info["iterations"]]
    else:
        solve_dc()
    res["on"] = _toy_a_state(dev)
    if not extended:
        f, C, Gs = AC.ssac_admittance([1e3, 1e6], source_name="VA")
        res["ssac_C"], res["ssac_G"] = _lst(C), _lst(Gs)
        Q = []
        for dv in (-1e-3, 1e-3):
            ds.circuit_alter(name="VA", value=V0 + dv)
            solve_dc()
            Q.append(_toy_a_state(dev)["Q_hi"])
        ds.circuit_alter(name="VA", value=V0)
        solve_dc()
        res["dQdV"] = (Q[1] - Q[0]) / 2e-3
        rows = CT.node_rows(dev, "ox", "PotentialEquation", [a, b])
        res["jac"] = fd_columns(dev, "ox", "Potential", [a, b], rows, lambda x: 1e-3, term)
    return res


def _res_state(dev):
    ds = _ds()
    import validation.devsim_noise as V
    I_lo = sum(float(ds.get_contact_current(device=dev, contact="lo", equation=e))
               for e in ("ElectronContinuityEquation", "HoleContinuityEquation"))
    return {"I_src": V.source_current(), "I_lo": I_lo,
            "psi": _lst(ds.get_node_model_values(device=dev, region="bulk", name="Potential")),
            "n": _lst(ds.get_node_model_values(device=dev, region="bulk", name="Electrons"))}


def build_resistor():
    import validation.devsim_noise as V
    return V.bipolar_1d("tB", [(0.0, H_RES, "lo"), (L_RES, H_RES, "hi")], "%.6e" % ND_RES, n_i=1e16,
                        tau=1e-7, ae=1e8)


def sc_toy_b(opts):
    """Toy B on the n resistor: conductance, Johnson through the two-point source, Jacobian, terminal-
    current conservation at bias, frozen continuity; wrong sign / dropped source / refusals by option."""
    import validation.devsim_noise as V
    from dynameta.carriers import eq_registry as _R
    ds = _ds()
    signs = tuple(opts.get("signs", (1.0, -1.0)))
    dev = build_resistor()
    a, b = _node_at(dev, "bulk", XA_R), _node_at(dev, "bulk", XB_R)
    mu = float(ds.get_parameter(device=dev, region="bulk", name="mu_n"))
    vt = float(ds.get_parameter(device=dev, region="bulk", name="V_t"))
    G = 1.602176634e-19 * mu * ND_RES / (XB_R - XA_R)     # doubles the a-b segment's conductance
    res = {"a": a, "b": b, "G": G, "mu_n": mu, "vt": vt}
    res["G0_off"] = V.dc_conductance(0.0, 1e8)
    term = QFLBridge("bridge", "bulk", a, b, G, signs=signs, noise=opts.get("noise", True),
                     jacobian_exact=opts.get("jacobian_exact", True))
    CT.add_custom_term(dev, term)
    V._solve(1e8, 1e-10)
    res["G0_on"] = V.dc_conductance(0.0, 1e8)
    res["G0_on_2mV"] = V.dc_conductance(0.0, 1e8, dv=2e-3)
    ff = [1.0, 1e6]
    try:
        kw = {"mechanisms": tuple(opts["mechanisms"])} if "mechanisms" in opts else {}
        sn = V.noise(ff, **kw)
        res["S"] = _lst(sn.S)
        res["by_mechanism"] = {k: _lst(v) for k, v in sn.by_mechanism.items()}
    except Exception as e:                                # refusals are asserted by the test
        res["noise_error"] = "{}: {}".format(type(e).__name__, e)
    Y = V.ssac_Y(ff)
    res["ReY"] = _lst(Y.real)
    rows = CT.node_rows(dev, "bulk", "ElectronContinuityEquation", [a, b])
    res["jac"] = (fd_columns(dev, "bulk", "Potential", [a, b], rows, lambda x: 1e-4, term)
                  + fd_columns(dev, "bulk", "Electrons", [a, b], rows, lambda x: 1e-4 * x, term))
    # biased: conservation of the terminal currents, with the converged residual as the budget
    V.set_bias(0.05, 1e8)
    V._solve(1e8, 1e-10)
    res["bias"] = _res_state(dev)
    rhs = matrix()[3]
    nn = len(res["bias"]["n"])
    car = np.concatenate([CT.node_rows(dev, "bulk", e, np.arange(1, nn - 1))
                          for e in ("ElectronContinuityEquation", "HoleContinuityEquation")]
                         + [[CT.circuit_row(c) for c in ds.get_circuit_node_list()]])
    res["bias"]["resid_sum"] = float(np.sum(np.abs(rhs[car])))     # carrier + KCL rows (rhs = -F)
    res["n_edges"] = nn - 1
    psi, n = res["bias"]["psi"], res["bias"]["n"]
    res["bias"]["term_I"] = G * ((psi[a] - vt * math.log(n[a])) - (psi[b] - vt * math.log(n[b])))
    # frozen continuity: Poisson-only solve with the term live -> rows_live False, converges
    _R.delete_by_name(dev, "ElectronContinuityEquation")
    _R.delete_by_name(dev, "HoleContinuityEquation")
    res["rows_live_frozen"] = term.rows_live(dev)
    calls0 = CT._dispatch_calls
    V._solve(1e10, 1e-10, 50)
    res["frozen_solved"] = True
    res["frozen_calls"] = CT._dispatch_calls - calls0
    _R.reapply_by_name(dev, "ElectronContinuityEquation")
    _R.reapply_by_name(dev, "HoleContinuityEquation")
    res["rows_live_after"] = term.rows_live(dev)
    return res


def sc_liveness(opts):
    """Add a term, delete the device, re-create it under the same name: the term must not apply."""
    ds = _ds()
    V0 = 1.0
    dev = build_dielectric()
    a, b = _node_at(dev, "ox", XA), _node_at(dev, "ox", XB)
    ds.circuit_alter(name="VA", value=V0)
    solve_dc()
    ref = _toy_a_state(dev)
    ref_m = [_lst(x) for x in matrix()]
    G = EPS_R * EPS0 / (XB - XA)
    CT.add_custom_term(dev, PoissonLink("link", "ox", a, b, G))
    solve_dc()
    res = {"ref": ref, "on": _toy_a_state(dev), "token": CT._tokens[(dev, "link")]}
    delete_device(dev)
    region = opts.get("region", "ox")
    build_dielectric(region=region)
    if opts.get("forge_marker"):                         # simulate a name-only liveness check
        ds.set_parameter(device=dev, name=CT.MARKER_PREFIX + "link", value=res["token"])
    solve_dc()
    res["recreated"] = {"psi": _lst(ds.get_node_model_values(device=dev, region=region,
                                                             name="Potential"))}
    res["recreated_matrix"] = [_lst(x) for x in matrix()]
    res["ref_matrix"] = ref_m
    res["live_terms"] = [t.name for t in CT.live_terms(dev)]
    res["registry"] = {d: sorted(v) for d, v in CT._terms.items()}
    return res


def sc_structural(opts):
    """Declared structural pattern: nnz grows by the extra candidates, every other entry is bitwise
    unchanged, the DC solution moves by round-off only; an undeclared entry raises out of the solve."""
    ds = _ds()
    dev = build_dielectric()
    a, b = _node_at(dev, "ox", XA), _node_at(dev, "ox", XB)
    c = _node_at(dev, "ox", 90e-9)
    ds.circuit_alter(name="VA", value=1.0)
    G = EPS_R * EPS0 / (XB - XA)
    CT.add_custom_term(dev, PoissonLink("link", "ox", a, b, G))
    solve_dc()
    plain = [_lst(x) for x in matrix()]
    psi_plain = _lst(ds.get_node_model_values(device=dev, region="ox", name="Potential"))
    CT.add_custom_term(dev, PoissonLink("link", "ox", a, b, G, pattern_extra=[(a, c), (b, c)]))
    solve_dc()
    struct = [_lst(x) for x in matrix()]
    psi_struct = _lst(ds.get_node_model_values(device=dev, region="ox", name="Potential"))
    res = {"plain": plain, "struct": struct, "psi_plain": psi_plain, "psi_struct": psi_struct,
           "rows": _lst(CT.node_rows(dev, "ox", "PotentialEquation", [a, b])),
           "col_c": int(CT.node_columns(dev, "ox", "Potential", [c])[0])}
    CT.add_custom_term(dev, PoissonLink("link", "ox", a, b, G, escape=True))
    try:
        solve_dc()
        res["escape_error"] = None
    except Exception as e:
        res["escape_error"] = str(e)
    return res


class _Anon(QFLBridge):
    """A QFLBridge that declares NO rows (unknown footprint) and no noise model."""

    def equations(self, device):
        return ()

    noise_sources = CustomTerm.noise_sources


def _inv(dev, region):
    from dynameta.carriers.physics_bipolar_dd import recombination_terms
    return {n: [rec.term.name, list(rec.regions), rec.noise_model, rec.jacobian_exact]
            for n, rec in recombination_terms(dev, region)["custom"].items()}


def sc_inventory(opts):
    """The "custom" kind of the ONE term inventory (physics_bipolar_dd.recombination_terms, plan 4.0.2):
    which live terms each region lists, with which capability flags, and that noise_analysis refuses
    through it -- a term declaring no rows (listed everywhere) without a noise model; then removal,
    device re-creation."""
    import validation.devsim_noise as V
    res = {}
    dev = build_resistor()
    a, b = _node_at(dev, "bulk", XA_R), _node_at(dev, "bulk", XB_R)
    mu = float(_ds().get_parameter(device=dev, region="bulk", name="mu_n"))
    G = 1.602176634e-19 * mu * ND_RES / (XB_R - XA_R)
    res["before"] = _inv(dev, "bulk")
    CT.add_custom_term(dev, QFLBridge("bridge", "bulk", a, b, G))
    CT.add_custom_term(dev, QFLBridge("inexact", "bulk", a, b, 0.0, jacobian_exact=False))
    CT.add_custom_term(dev, _Anon("anon", "bulk", a, b, 0.0))
    res["with"] = _inv(dev, "bulk")
    V._solve(1e8, 1e-10)

    def noise():
        try:
            V.noise([1.0])
            return None
        except Exception as e:                            # refusals are asserted by the test
            return "{}: {}".format(type(e).__name__, e)
    res["err_all"] = noise()
    CT.remove_custom_term(dev, "anon")
    res["err_inexact"] = noise()
    CT.remove_custom_term(dev, "inexact")
    res["after_remove"] = _inv(dev, "bulk")
    res["err_none"] = noise()
    V.fresh()
    dev = build_resistor()                                 # same device name, re-created
    res["recreated"] = _inv(dev, "bulk")
    return res


def sc_calllog(opts):
    """The D1b call-log digest fixtures recorded with the dispatcher registered FIRST (no term)."""
    if opts.get("register"):
        CT.register_dispatcher()
    import _bipolar_calllog as CL
    log = json.loads(json.dumps(CL._record_in_this_process(canonical=False)))   # same-machine A/B: raw
    return {"registered": CT.is_registered(), "n": len(log), "digest": CL.digest(log),
            "dispatch_calls": CT._dispatch_calls}


SCENARIOS = {"calllog": sc_calllog, "symbolic": sc_symbolic, "import": sc_import,
             "defaults": sc_defaults, "toy_a": sc_toy_a, "toy_b": sc_toy_b, "liveness": sc_liveness,
             "structural": sc_structural, "inventory": sc_inventory}


def run_child(scenario, **opts):
    """Run `scenario` in a fresh interpreter on this sys.path; returns its JSON result."""
    import subprocess
    import tempfile
    fd, out = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    try:
        env = dict(os.environ, PYTHONPATH=os.pathsep.join(p for p in sys.path if isinstance(p, str) and p))
        r = subprocess.run([sys.executable, os.path.abspath(__file__), scenario, out, json.dumps(opts)],
                           env=env, capture_output=True, text=True, timeout=900)
        if r.returncode != 0:
            raise RuntimeError("child {} failed:\n{}".format(scenario, r.stderr[-4000:]))
        with open(out, encoding="utf-8") as f:
            return json.load(f)
    finally:
        os.remove(out)


if __name__ == "__main__":
    import warnings
    warnings.simplefilter("error")                       # warnings are errors, as in the suite
    _opts = json.loads(sys.argv[3]) if len(sys.argv) > 3 else {}
    _res = SCENARIOS[sys.argv[1]](_opts)
    with open(sys.argv[2], "w", encoding="utf-8") as _f:
        json.dump(_res, _f)
