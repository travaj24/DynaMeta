"""Gates for the recombination-term bookkeeping, version 2 (DynaMeta v0.15, plan item 4.0.2; P3 section
1.3.2 / step K0, P1 section 0.3): single-carrier node terms (carrier="electron" / "hole"), derivatives with
respect to EXTRA node solution variables (wrt=), and the unified inventory physics_bipolar_dd.
recombination_terms() that the noise analysis reads (its "custom" kind is gated in tests/test_custom_terms.py,
whose registering cases need a child process).

DEFAULTS OFF (plan section 2 item 5). (i) Same tree, one process: adding the shipped term kinds without the
new keywords and with them at their defaults gives the SAME DEVSIM call log (every model / equation string,
in order) and bit-identical solved arrays, and the composites are the version-1 strings re-typed here. (ii)
The call-log digest (tests/_bipolar_calllog.py, gated in tests/test_bipolar_recombination_bgn_ionization.py)
and (iii) the ee0ad04 numeric cross-version gate (tests/test_devsim_sg_flux.py) are unchanged -- run, not
re-recorded. The cross-VERSION call log of every shipped term path (76e8633 vs this tree) is in the audit note.

ORACLES (closed forms re-derived here; inputs -- mobility, V_t, doping, n_i, the knobs G0 / k -- are read
back as DEVSIM parameters or set here, never results):
  (a) electron-only constant sink G0 in a p slab (and the hole-only twin in an n slab), both contacts at
      0 V, SRH off (tau = 1e3 s): the minority density solves D c'' = G0 with c = c_c (the neutral contact
      value) at both ends, c(x) = c_c - G0 x (L - x) / (2 D). The box scheme's central difference is exact
      on a quadratic, so the nodes must sit ON the parabola. Neglected physics, bounded: drift of the
      minority carrier in the majority's screening field, ~ c_c / N = 1e-12 relative; SRH, Delta c / (G0
      tau) <= L^2 / (8 D tau) = 4e-14. Budget 1e-11 of c_c. Conservation: the term acts on bulk rows only
      (DEVSIM drops a contact node's bulk row), so the two contacts must supply q G0 (L - h) between them --
      an exact discrete identity; budget 64 eps of it. The other carrier carries NO terminal current
      (J = 0 exactly: constant flux, equal quasi-Fermi levels at both contacts), budget = the converged
      residual of its rows (sum |rhs|, as tests/test_custom_terms.py toy B) + N eps T.
  (b) its shot noise through the single-carrier two-point source, S_I = 2 q^2 G0 Sum_i V_i (x_i / L)^2: the
      low-frequency adjoint of a minority carrier removed at x in a short-circuited bar is x / L (the
      diffusive refill splits by distance; y = 1 on the circuit contact's node), V_i from the node
      positions. Budget 1e-11: drift ~1e-12, SRH (L / L_n)^2 = 5e-13, (omega L^2 / D)^2 at 1 Hz ~ 1e-17,
      the adjoint solve's measured round-off 2e-15 (noise_analysis module doc, item 4).
  (c) a term depending on an EXTRA variable TrU (its own equation TrU = n / n_ref, no contact equation):
      the electron sink k n_ref TrU = k n, so c_i = c_c cosh(theta (i - N/2)) / cosh(theta N/2) with
      cosh(theta) = 1 + k h^2 / (2 D) -- the exact solution of the box scheme's recurrence on the uniform
      mesh (budget 1e-11 as (a)); the continuum cosh profile differs by O(h^2): MEASURED 4.5e-5, shown,
      not gated. Jacobian: the TrU column of the electron, hole and Poisson rows against (1) the closed
      forms d/dU [V_i (-q (k_lin n_ref + k_b) U)] etc. (16 eps) and (2) central differences of the
      assembled residual (rhs = -F; the terms are at most quadratic in U, so no truncation: budget
      8 eps R_r / delta with R_r = Sum_k |J[r, k] x_k|).
ADVERSARIAL: carrier="both" breaks (a)'s other-carrier current (the hole rows then get the sink); the wrong
carrier breaks (a)'s profile; a partner row on the hole equation collapses (b); deleting the composite's
TrU derivative (what version 1 builds: Electrons / Holes only) breaks (c)'s Jacobian gate by 100 %.
OLD CODE (76e8633, MEASURED by running this file against `git archive 76e8633`): every gate except the
A/B identity FAILS -- add_recombination_term has no carrier= / wrt= keywords (TypeError) and
recombination_terms has no "node_electron" / "node_hole" / "custom" kinds (KeyError); the A/B identity
fails too, on the same TypeError (the explicit-default call). Runtime ~25 s serial.
"""
import contextlib
import math

import numpy as np
import pytest

devsim = pytest.importorskip("devsim")
V = pytest.importorskip("validation.devsim_noise")
from scipy.constants import e as QE                          # noqa: E402

from dynameta.carriers import physics_bipolar_dd as BP       # noqa: E402

ds = V.ds
EPS = np.finfo(float).eps
L, NSEG = 1e-6, 100
H = L / NSEG
NI, TAU = 1e16, 1e3
EQ_N, EQ_P = "ElectronContinuityEquation", "HoleContinuityEquation"
_LOGGED = ("node_model", "edge_model", "element_model", "equation", "set_parameter", "delete_node_model",
           "delete_edge_model")


@pytest.fixture(autouse=True)
def _clean():
    V.fresh()
    V.set_bias._v = {}
    yield
    V.fresh()


def _vals(dev, name, region="bulk"):
    return np.asarray(ds.get_node_model_values(device=dev, region=region, name=name), dtype=float)


def _slab(dev, doping):
    """Uniform 1-D Si slab [0, L], h = L/100, 'lo' grounded, 'hi' on circuit V1 (validation.devsim_noise)."""
    V.bipolar_1d(dev, [(0.0, H, "lo"), (L, H, "hi")], "%.6e" % doping, n_i=NI, tau=TAU, ae=1e8)
    return dev


def _solve():
    V._solve(1e8, 1e-13, 60)


def _neutral(doping):
    """(majority, minority) neutral contact densities, Boltzmann, complete ionization."""
    a = abs(doping)
    maj = 0.5 * a + math.sqrt(0.25 * a * a + NI * NI)
    return maj, NI * NI / maj


def _param(dev, name):
    return float(ds.get_parameter(device=dev, region="bulk", name=name))


@contextlib.contextmanager
def _call_log():
    """Record every model / equation / parameter-defining DEVSIM call (kwargs repr, device dropped)."""
    log, saved = [], {}

    def wrap(nm, fn):
        def inner(*a, **kw):
            log.append((nm, tuple(sorted((k, repr(v)) for k, v in kw.items() if k != "device"))))
            return fn(*a, **kw)
        return inner
    for nm in _LOGGED:
        saved[nm] = getattr(ds, nm)
        setattr(ds, nm, wrap(nm, saved[nm]))
    try:
        yield log
    finally:
        for nm, fn in saved.items():
            setattr(ds, nm, fn)


def _defined(log, name):
    """The last equation string a log defined for node/edge model `name`."""
    out = [dict(kw)["equation"] for nm, kw in log if nm in ("node_model", "edge_model")
           and dict(kw).get("name") == repr(name)]
    return eval(out[-1])                                    # repr(str) -> str


# =====================================================================================================
# (i) defaults off, same tree
# =====================================================================================================

def _ab_state(explicit):
    dev = _slab("ab", -1e22)
    ds.set_parameter(device=dev, region="bulk", name="G0", value=1e18)
    new = {"carrier": "both", "wrt": None, "partner": None, "mechanism": "gr"} if explicit else {}
    with _call_log() as log:
        BP.add_recombination_term(dev, "bulk", "aa", "1e-30*Electrons*Holes", pair="0", **new)
        BP.add_recombination_term(dev, "bulk", "bb", "G0", pair="G0",
                                  derivs={"Electrons": "0", "Holes": "0"}, **new)
        BP.add_recombination_term(dev, "bulk", "ee", "1e-40*EdgeInverseLength", assembly="edge",
                                  pair="0", **({k: v for k, v in new.items() if k == "carrier"}))
        BP.add_charge_term(dev, "bulk", "qq", "1e10", **({"wrt": None} if explicit else {}))
    _solve()
    arrays = {k: _vals(dev, k).tobytes() for k in ("Potential", "Electrons", "Holes", "ElectronGeneration",
                                                    "HoleGeneration", "PotentialNodeCharge")}
    arrays["I"] = repr([ds.get_contact_current(device=dev, contact=c, equation=e)
                        for c in ("lo", "hi") for e in (EQ_N, EQ_P)])
    inv = BP.recombination_terms(dev, "bulk")
    models = sorted(ds.get_node_model_list(device=dev, region="bulk"))
    V.fresh()
    return log, arrays, inv, models


def test_defaults_off_same_tree_call_log_and_arrays():
    """GATE (i). The shipped term kinds (node with DEVSIM diff, node with closed forms, 1-D edge, charge)
    added without the v0.15 keywords and with them at their defaults: identical DEVSIM call log, identical
    node-model inventory, bit-identical solved arrays and contact currents; the composites are the
    version-1 strings, re-typed; the new inventory kinds are empty."""
    log0, arr0, inv0, mod0 = _ab_state(False)
    log1, arr1, inv1, mod1 = _ab_state(True)
    assert log1 == log0 and len(log0) > 10
    assert mod1 == mod0 and arr1 == arr0
    assert _defined(log0, "ElectronGeneration") == "-ElectronCharge * (USRH + RTerm_aa + RTerm_bb)"
    assert _defined(log0, "HoleGeneration") == "+ElectronCharge * (USRH + RTerm_aa + RTerm_bb)"
    assert _defined(log0, "PotentialNodeCharge") == \
        "-ElectronCharge * (kahan3(Holes, -Electrons, NetDoping) + QTerm_qq)"
    # version 1 differentiated the composites w.r.t. Electrons and Holes only -- so does version 2 here
    assert sorted(dict(kw)["name"] for nm, kw in log0 if nm == "node_model"
                  and dict(kw)["name"].startswith("'ElectronGeneration:")) == \
        sorted(["'ElectronGeneration:Electrons'", "'ElectronGeneration:Holes'"] * 2)
    for inv in (inv0, inv1):
        assert inv["node_electron"] == {} and inv["node_hole"] == {} and inv["custom"] == {}
        assert inv["node"] == {"aa": ("RTerm_aa", "PTerm_aa"), "bb": ("RTerm_bb", "PTerm_bb")}
        assert inv["edge"] == {"ee": ("RTermEdge_ee", "PTermEdge_ee")}


# =====================================================================================================
# (a) single-carrier constant sink vs the continuity closed form; (b) its shot noise
# =====================================================================================================

def _sink(carrier, *, term_carrier=None, doping=None, **kw):
    """Slab with a constant sink G0 of `term_carrier` (default: `carrier`, the minority one), solved."""
    doping = (-1e22 if carrier == "electron" else 1e22) if doping is None else doping
    dev = _slab("sk", doping)
    mu = _param(dev, "mu_n" if carrier == "electron" else "mu_p")
    D = mu * _param(dev, "V_t")
    maj, mino = _neutral(doping)
    G0 = 8.0 * D * 0.5 * mino / L ** 2                          # depletes the centre to c_c / 2
    ds.set_parameter(device=dev, region="bulk", name="G0", value=G0)
    with _call_log() as log:
        BP.add_recombination_term(dev, "bulk", "sink", "G0", pair="G0", carrier=term_carrier or carrier,
                                  **kw)
    _solve()
    return dev, {"D": D, "maj": maj, "mino": mino, "G0": G0, "mu_other": _param(
        dev, "mu_p" if carrier == "electron" else "mu_n"), "log": log}


def _currents(dev):
    return {(c, e): float(ds.get_contact_current(device=dev, contact=c, equation=e))
            for c in ("lo", "hi") for e in (EQ_N, EQ_P)}


def _row_source(dev, eq, flux):
    """(S, budget): the total node source the bulk rows of `eq` carry, S = I_lo + I_hi - Sum rhs over the
    interior rows (rhs = -F) -- the box scheme telescopes every edge flux out of this sum, leaving
    Sum_i V_i x (the rows' node terms), i.e. what the composite routing decides, read from DEVSIM's own
    assembly -- and the round-off budget of that identity: each row's sum rounds by <= 2 eps (|J_k| +
    |J_k-1| + |node term|) with J the SG edge flux `flux` as DEVSIM evaluated it, and the numpy sum of the
    rhs by <= N eps Sum |rhs|."""
    with V.quiet():
        rhs = np.asarray(ds.get_matrix_and_rhs(format="csc")["static"]["rhs"], dtype=float)
    rows = np.asarray(ds.get_equation_numbers(device=dev, region="bulk", equation=eq))[1:-1]
    I = _currents(dev)
    S = I[("lo", eq)] + I[("hi", eq)] - float(np.sum(rhs[rows]))
    J = np.abs(np.asarray(ds.get_edge_model_values(device=dev, region="bulk", name=flux), dtype=float))
    return S, 2.0 * EPS * (2.0 * float(np.sum(J)) + abs(S)) + rows.size * EPS * float(np.sum(np.abs(rhs[rows])))


def _usrh(dev):
    """The base SRH channel's rate from the SOLVED n, p, re-typed (midgap level, tau_n = tau_p = TAU, no
    BGN): U = (n p - n_i^2) / (tau (n + n_i) + tau (p + n_i)) -- it acts on BOTH carriers' rows."""
    n, p = _vals(dev, "Electrons"), _vals(dev, "Holes")
    return (n * p - NI * NI) / (TAU * (n + NI) + TAU * (p + NI))


def _sink_checks(dev, d, carrier):
    """(profile miss / c_c, own-row source miss / budget, other-row source miss / budget). The own rows
    must carry q Sum V_i (G0 + U_i), the other carrier's rows q Sum V_i U_i only (U: the base SRH, ~1e-14
    of G0 here, but resolved by this identity)."""
    x = _vals(dev, "x")
    c = _vals(dev, "Electrons" if carrier == "electron" else "Holes")
    oracle = d["mino"] - d["G0"] * x * (L - x) / (2.0 * d["D"])
    prof = float(np.max(np.abs(c - oracle)) / d["mino"])
    own, other = (EQ_N, EQ_P) if carrier == "electron" else (EQ_P, EQ_N)
    fl = {EQ_N: "ElectronCurrent", EQ_P: "HoleCurrent"}
    vol = np.empty_like(x)
    vol[1:-1] = 0.5 * (x[2:] - x[:-2])
    qU = QE * float(np.sum(vol[1:-1] * _usrh(dev)[1:-1]))
    S_own, b_own = _row_source(dev, own, fl[own])
    S_other, b_other = _row_source(dev, other, fl[other])
    expect_own = QE * d["G0"] * float(np.sum(vol[1:-1])) + qU
    return (prof, abs(abs(S_own) - abs(expect_own)) / (b_own + NSEG * EPS * abs(expect_own)),
            abs(abs(S_other) - abs(qU)) / (b_other + NSEG * EPS * abs(qU)))


@pytest.mark.parametrize("carrier", ["electron", "hole"])
def test_single_carrier_sink_vs_continuity_closed_form(carrier):
    """GATE (a). MEASURED: profile 3.8e-13 (electron) / 4.1e-13 (hole) of c_c; the carrier's own rows carry
    q G0 (L - h) and the other carrier's rows carry nothing, both far inside their round-off budgets
    (audit note). The composites: the term in its own carrier's composite only."""
    dev, d = _sink(carrier)
    prof, own_miss, other_miss = _sink_checks(dev, d, carrier)
    assert prof <= 1e-11, prof
    assert own_miss <= 1.0, own_miss
    assert other_miss <= 1.0, other_miss
    own, other = (("ElectronGeneration", "HoleGeneration") if carrier == "electron"
                  else ("HoleGeneration", "ElectronGeneration"))
    tag = "RTermN_sink" if carrier == "electron" else "RTermP_sink"
    sign = "-" if carrier == "electron" else "+"
    assert _defined(d["log"], own) == "{}ElectronCharge * (USRH + {})".format(sign, tag)
    assert _defined(d["log"], other) == "{}ElectronCharge * USRH".format("+" if sign == "-" else "-")
    inv = BP.recombination_terms(dev, "bulk")
    key = "node_electron" if carrier == "electron" else "node_hole"
    assert inv[key] == {"sink": BP.CarrierTerm(tag, tag.replace("R", "P", 1), None, 0.0, "gr",
                                               EQ_N if carrier == "electron" else EQ_P,
                                               -1.0 if carrier == "electron" else 1.0)}
    assert inv["node"] == {} and inv["node_hole" if key == "node_electron" else "node_electron"] == {}


def test_single_carrier_sink_refuted_by_both_and_by_the_wrong_carrier():
    """ADVERSARIAL for (a). carrier="both" on the same sink: the hole rows then carry q G0 (L - h) too
    (MEASURED: the other-row gate misses by ~1e3 x its budget); carrier="hole" in the p slab: the electron
    profile stays flat (MEASURED 0.50 of c_c off the parabola)."""
    dev, d = _sink("electron", term_carrier="both")
    assert _sink_checks(dev, d, "electron")[2] > 100.0
    V.fresh()
    dev, d = _sink("electron", term_carrier="hole")
    prof = _sink_checks(dev, d, "electron")[0]
    assert prof > 0.4, prof


def _noise_oracle(dev, G0):
    x = _vals(dev, "x")
    vol = np.empty_like(x)                                    # box volumes from the node positions
    vol[1:-1] = 0.5 * (x[2:] - x[:-2])
    vol[0], vol[-1] = 0.5 * (x[1] - x[0]), 0.5 * (x[-1] - x[-2])
    return 2.0 * QE ** 2 * G0 * float(np.sum(vol * (x / L) ** 2))


@pytest.mark.parametrize("carrier", ["electron", "hole"])
def test_single_carrier_sink_shot_noise(carrier):
    """GATE (b). MEASURED 8.4e-14 (electron) / 1.9e-13 (hole) against S_I = 2 q^2 G0 Sum V_i (x_i/L)^2."""
    dev, d = _sink(carrier)
    S = V.noise([1.0], mechanisms=("gr",))
    rel = float(S.S[0]) / _noise_oracle(dev, d["G0"]) - 1.0
    assert abs(rel) <= 1e-11, rel
    assert set(S.by_mechanism) == {"gr"}


def test_single_carrier_noise_refuted_by_a_partner_row():
    """ADVERSARIAL for (b): declaring the hole row as the event's partner (+1: a pair event) makes the
    source |y_p - y_n|^2, which nearly cancels in an ohmic bar (both fields ~ x/L)."""
    dev, d = _sink("electron", partner=(EQ_P, 1.0))
    S = V.noise([1.0], mechanisms=("gr",))
    assert float(S.S[0]) < 0.5 * _noise_oracle(dev, d["G0"])


def test_noise_refuses_single_carrier_terms_without_a_noise_model():
    """The "never silently noiseless" rule for the new kind: no event rate -> refused when its mechanism is
    requested (not otherwise, as for the pair kinds); a partner equation the region lacks -> refused."""
    dev = _slab("rf", -1e22)
    ds.set_parameter(device=dev, region="bulk", name="G0", value=1e15)
    BP.add_recombination_term(dev, "bulk", "s0", "G0", carrier="electron")          # no pair
    _solve()
    assert BP.recombination_terms(dev, "bulk")["node_electron"]["s0"].pair is None
    with pytest.raises(RuntimeError, match="without an event rate"):
        V.noise([1.0])
    V.noise([1.0], mechanisms=("diffusion_n", "diffusion_p"))                       # not requested: runs
    BP.add_recombination_term(dev, "bulk", "s0", "G0", carrier="electron", pair="G0",
                              partner=("TrapEquation_nope", 1.0), mechanism="trap_n")
    assert BP.recombination_terms(dev, "bulk")["node_electron"]["s0"] == BP.CarrierTerm(
        "RTermN_s0", "PTermN_s0", "TrapEquation_nope", 1.0, "trap_n", EQ_N, -1.0)
    with pytest.raises(RuntimeError, match="TrapEquation_nope"):
        V.noise([1.0])
    S = V.noise([1.0], mechanisms=("gr",))                                           # trap_n not asked
    assert set(S.by_mechanism) == {"gr"}


# =====================================================================================================
# (c) extra solution variable: DC closed form and the Jacobian
# =====================================================================================================

def _extra_var_device():
    dev = _slab("xv", -1e22)
    reg = "bulk"
    D = _param(dev, "mu_n") * _param(dev, "V_t")
    _maj, mino = _neutral(-1e22)
    k = 25.0 * D / L ** 2                                       # decay length L / 5
    for nm, v in (("n_ref", mino), ("k_lin", k)):
        ds.set_parameter(device=dev, region=reg, name=nm, value=v)
    ds.node_solution(device=dev, region=reg, name="TrU")
    ds.set_node_values(device=dev, region=reg, name="TrU", values=list(_vals(dev, "Electrons") / mino))
    ds.node_model(device=dev, region=reg, name="TrUNode", equation="TrU - Electrons/n_ref")
    ds.node_model(device=dev, region=reg, name="TrUNode:TrU", equation="1")
    ds.node_model(device=dev, region=reg, name="TrUNode:Electrons", equation="-1/n_ref")
    ds.equation(device=dev, region=reg, name="TrUEquation", variable_name="TrU", node_model="TrUNode",
                variable_update="default")
    BP.add_recombination_term(dev, reg, "lin", "k_lin * n_ref * TrU", carrier="electron", wrt=("TrU",),
                              derivs={"Electrons": "0", "Holes": "0", "TrU": "k_lin * n_ref"},
                              pair="k_lin * n_ref * TrU")
    return dev, D, mino, k


def test_extra_variable_term_dc_closed_form():
    """GATE (c), DC. MEASURED 1.6e-13 of c_c against the box scheme's exact cosh recurrence (the continuum
    cosh profile: 4.5e-5, the O(h^2) discretisation, not gated)."""
    dev, D, mino, k = _extra_var_device()
    _solve()
    n = _vals(dev, "Electrons")
    th = math.acosh(1.0 + k * H * H / (2.0 * D))
    i = np.arange(n.size)
    oracle = mino * np.cosh(th * (i - NSEG / 2)) / np.cosh(th * NSEG / 2)
    assert float(np.max(np.abs(n - oracle)) / mino) <= 1e-11
    assert "ElectronGeneration:TrU" in ds.get_node_model_list(device=dev, region="bulk")
    assert "HoleGeneration:TrU" not in ds.get_node_model_list(device=dev, region="bulk")


def _jac_block(dev):
    """Assembled matrix J = dF/dx (DEVSIM's static matrix), rhs = -F, and the global numbers."""
    import scipy.sparse as sp
    with V.quiet():
        m = ds.get_matrix_and_rhs(format="csc")["static"]
    A = sp.csc_matrix((np.asarray(m["av"], dtype=float), np.asarray(m["ai"]), np.asarray(m["ap"])))
    return A, np.asarray(m["rhs"], dtype=float)


def _numbers(dev, eq=None, var=None):
    kw = {"equation": eq} if eq else {"variable": var}
    return np.asarray(ds.get_equation_numbers(device=dev, region="bulk", **kw), dtype=np.int64)


def _install_more_terms(dev):
    """A hole-only term through a NAMED sub-model (DEVSIM's diff), a both-carrier term and a charge term,
    all depending on TrU."""
    reg = "bulk"
    for nm, v in (("k_nl", 3.0e10), ("k_b", 7.0e18), ("Nt", 5.0e11)):
        ds.set_parameter(device=dev, region=reg, name=nm, value=v)
    ds.node_model(device=dev, region=reg, name="TrUsq", equation="TrU*TrU")
    ds.node_model(device=dev, region=reg, name="TrUsq:TrU", equation="2*TrU")
    BP.add_recombination_term(dev, reg, "nl", "k_nl * n_ref * TrUsq", carrier="hole", wrt=("TrU",),
                              pair="k_nl * n_ref * TrUsq")
    BP.add_recombination_term(dev, reg, "bt", "k_b * TrU", wrt="TrU", pair="k_b * TrU")
    BP.add_charge_term(dev, reg, "occ", "Nt * TrU", wrt=("TrU",))


def _jacobian_gate(dev):
    """Per carrier row family (electron, hole, Poisson), over the TrU column: {"offdiag": max |entry| off
    the node diagonal, "closed": max closed-form miss [eps], "fd": max FD miss / its budget, "fd_rel": max
    FD miss / |FD|}."""
    A, _rhs = _jac_block(dev)
    cols = _numbers(dev, var="TrU")
    x = _vals(dev, "x")
    vol = 0.5 * (x[2:] - x[:-2])                              # interior box volumes, from the positions
    U = _vals(dev, "TrU")
    nref = _param(dev, "n_ref")
    closed = {EQ_N: -QE * (_param(dev, "k_lin") * nref + _param(dev, "k_b")) * vol,
              EQ_P: QE * (2.0 * _param(dev, "k_nl") * nref * U[1:-1] + _param(dev, "k_b")) * vol,
              "PotentialEquation": -QE * _param(dev, "Nt") * vol}
    delta = 1e-4
    rhs = {}
    for s in (1.0, -1.0):
        ds.set_node_values(device=dev, region="bulk", name="TrU", values=list(U + s * delta))
        rhs[s] = _jac_block(dev)[1]
    ds.set_node_values(device=dev, region="bulk", name="TrU", values=list(U))
    fd = -(rhs[1.0] - rhs[-1.0]) / (2.0 * delta)             # rhs = -F; every U moved: the row sums
    xs = np.zeros(A.shape[0])
    for v in ("Potential", "Electrons", "Holes", "TrU"):
        xs[_numbers(dev, var=v)] = _vals(dev, v)
    R = abs(A).dot(np.abs(xs))                                # round-off scale of each row's residual
    out = {}
    for eq, cf in closed.items():
        rows = _numbers(dev, eq=eq)[1:-1]
        blk = A[rows][:, cols].toarray()                      # interior node j+1 <-> column j+1
        j = np.arange(blk.shape[0])
        diag = blk[j, j + 1].copy()
        blk[j, j + 1] = 0.0
        out[eq] = {"offdiag": float(np.max(np.abs(blk))),
                   "closed": float(np.max(np.abs(diag - cf) / np.abs(cf)) / EPS),
                   "fd": float(np.max(np.abs(fd[rows] - diag) / (8.0 * EPS * R[rows] / delta))),
                   "fd_rel": float(np.max(np.abs(fd[rows] - diag) / np.abs(fd[rows])))}
    return out


def test_extra_variable_jacobian_vs_closed_form_and_fd():
    """GATE (c), Jacobian: the TrU column of the electron rows (closed-form term + both-carrier term), the
    hole rows (a term through a NAMED sub-model, DEVSIM's own diff, + the both-carrier term) and the
    Poisson rows (a charge term). MEASURED in the audit note (closed forms within a few eps; FD well
    inside its round-off budget)."""
    dev, _D, _mino, _k = _extra_var_device()
    _solve()
    _install_more_terms(dev)
    g = _jacobian_gate(dev)
    for eq, m in g.items():
        assert m["offdiag"] == 0.0, (eq, m)
        assert m["closed"] <= 16.0, (eq, m)
        assert m["fd"] <= 1.0, (eq, m)
    have = set(ds.get_node_model_list(device=dev, region="bulk"))
    assert {"ElectronGeneration:TrU", "HoleGeneration:TrU", "PotentialNodeCharge:TrU",
            "RTermP_nl:TrU", "RTerm_bt:TrU", "QTerm_occ:TrU"} <= have


def test_extra_variable_jacobian_refuted_without_the_composite_derivative():
    """ADVERSARIAL for (c): version 1 differentiates the composite w.r.t. Electrons / Holes only. Deleting
    ElectronGeneration:TrU (that state) leaves the electron rows' TrU column EMPTY while the residual still
    depends on TrU: the FD gate misses by 100 % (the other rows are untouched)."""
    dev, _D, _mino, _k = _extra_var_device()
    _solve()
    _install_more_terms(dev)
    ds.delete_node_model(device=dev, region="bulk", name="ElectronGeneration:TrU")
    g = _jacobian_gate(dev)
    assert g[EQ_N]["fd_rel"] == 1.0 and g[EQ_N]["fd"] > 1e6, g[EQ_N]
    assert g[EQ_P]["fd"] <= 1.0 and g["PotentialEquation"]["fd"] <= 1.0


# =====================================================================================================
# validation of the new keywords (in-process, cheap)
# =====================================================================================================

def test_new_keywords_are_validated():
    dev = _slab("kw", -1e22)
    r = "bulk"
    with pytest.raises(ValueError, match="carrier must be"):
        BP.add_recombination_term(dev, r, "a", "0", carrier="electrons")
    with pytest.raises(ValueError, match="not a node solution"):
        BP.add_recombination_term(dev, r, "a", "0", carrier="electron", wrt=("NoSuchVar",))
    with pytest.raises(ValueError, match="node terms only"):
        BP.add_recombination_term(dev, r, "a", "0", assembly="edge", carrier="hole")
    with pytest.raises(ValueError, match="SINGLE-carrier"):
        BP.add_recombination_term(dev, r, "a", "0", partner=(EQ_P, 1.0))
    with pytest.raises(ValueError, match="MECHANISMS"):
        BP.add_recombination_term(dev, r, "a", "0", carrier="electron", mechanism="nope")
    with pytest.raises(ValueError, match="partner must be"):
        BP.add_recombination_term(dev, r, "a", "0", carrier="electron", partner=(EQ_P, 0.0))
    ds.node_solution(device=dev, region=r, name="Wv")
    with pytest.raises(ValueError, match="must give exactly"):         # a missing derivative is a 0
        BP.add_recombination_term(dev, r, "a", "Wv", carrier="electron", wrt=("Wv",),
                                  derivs={"Electrons": "0", "Holes": "0"})
    with pytest.raises(ValueError, match="must give exactly"):
        BP.add_charge_term(dev, r, "q", "Wv", derivs={"Electrons": "0", "Holes": "0", "Wv": "1"})
    assert not [m for m in ds.get_node_model_list(device=dev, region=r) if m.startswith(("RTerm", "QTerm"))]
    BP.add_recombination_term(dev, r, "a", "0", pair="0")
    with pytest.raises(ValueError, match="pair term"):                  # a pair AND a branch: refused
        BP.add_recombination_term(dev, r, "a", "0", carrier="hole", pair="0")
    BP.add_recombination_term(dev, r, "k", "0", carrier="electron", pair="0")   # one process's two
    BP.add_recombination_term(dev, r, "k", "0", carrier="hole", pair="0")       # branches share a name
    with pytest.raises(ValueError, match="pair term"):
        BP.add_recombination_term(dev, r, "k", "0", pair="0")
    inv = BP.recombination_terms(dev, r)
    assert list(inv["node_electron"]) == ["k"] and list(inv["node_hole"]) == ["k"]
    # re-adding a term WITHOUT its old extra variable drops the stale derivative and the composite's
    BP.add_recombination_term(dev, r, "w", "Wv", carrier="electron", wrt=("Wv",), pair="0")
    assert "RTermN_w:Wv" in ds.get_node_model_list(device=dev, region=r)
    BP.add_recombination_term(dev, r, "w", "1e3", carrier="electron", pair="0")
    have = ds.get_node_model_list(device=dev, region=r)
    assert "RTermN_w:Wv" not in have and "ElectronGeneration:Wv" in have   # re-differentiated: now 0
    assert float(np.max(np.abs(_vals(dev, "ElectronGeneration:Wv")))) == 0.0
