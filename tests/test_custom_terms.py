"""Gates for Python-assembled custom terms (dynameta.carriers.custom_terms; DynaMeta v0.15, plan item
4.0.3, P1 section 0 gate C0).

EVERY SCENARIO THAT REGISTERS THE DISPATCHER RUNS IN A CHILD PROCESS (tests/_custom_terms_child.py):
DEVSIM has no delete_custom_equation, so a registration would otherwise leak into every later test of the
worker. The in-process tests here only exercise refusals, which validate before touching DEVSIM, and they
assert that nothing got registered.

ORACLES (independent of the code under test; inputs -- permittivity, mobility, doping, the toy coupling
G -- are read back as DEVSIM parameters, never results):
  toy A (PoissonLink, a nonlocal flux link between dielectric nodes a, b): the series network of the
        three segments with G in parallel with the middle one -- closed-form node potentials, contact
        charge Q = C V and C = 1/(x_a/eps + 1/(eps/(x_b - x_a) + G) + (L - x_b)/eps); ssac C(f) must equal
        it; the two contact charges must cancel.
  toy B (QFLBridge, electron transfer a -> b driven by the quasi-Fermi difference in a uniform n bar):
        the same network for the conductance with sigma = q mu n_eq; Johnson-Nyquist S = 4 kT G with the
        bridge's own two-point source (P = 2 kT G/q^2, the detailed-balance event rate of a linear
        conductance); the two terminal currents must cancel.
  Jacobian: central differences of the assembled residual (steps h and 2h) against the assembled matrix.
OLD CODE (9b6937f): dynameta.carriers.custom_terms does not exist -> every gate FAILS with the import
error below (not a skip); and noise_analysis has no TwoPointSource, so the toy-B noise gate fails too.
Runtime: ~45 s serial (each child 1-9 s).
"""
import math
import os
import sys

import numpy as np
import pytest

pytest.importorskip("devsim")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from dynameta.carriers import custom_terms as CT
    import _custom_terms_child as CH
    _IMPORT_ERROR = None
except ImportError as _e:                                   # pragma: no cover - old code only
    CT = CH = None
    _IMPORT_ERROR = _e
finally:
    sys.path.pop(0)

from scipy.constants import e as QE, k as KB             # noqa: E402

EPS = np.finfo(float).eps


@pytest.fixture(autouse=True)
def _need_module():
    if _IMPORT_ERROR is not None:
        pytest.fail("custom_terms unavailable: {!r}".format(_IMPORT_ERROR))


def _child(scenario, **opts):
    return CH.run_child(scenario, **opts)


# =====================================================================================================
# (a) defaults off
# =====================================================================================================

def test_import_registers_nothing_first_add_registers_once():
    """GATE C0(a)-1. Importing every dynameta.carriers module registers no custom equation (a spy on
    ds.custom_equation, in a child); building and solving a device does not either; the FIRST
    add_custom_term registers exactly one dispatcher, a second add none."""
    r = _child("import")
    assert len(r["modules"]) > 30
    assert r["after_import"] == [False, []]
    assert r["after_build_solve"] == [False, []]
    assert r["after_first_add"] == [True, [CT.DISPATCHER_NAME]]
    assert r["after_second_add"] == [True, [CT.DISPATCHER_NAME]]
    assert not CT.is_registered()                         # and not in this (pytest) process either


@pytest.fixture(scope="module")
def defaults_pair():
    return _child("defaults"), _child("defaults", register=True)


def test_registration_alone_changes_nothing(defaults_pair):
    """GATE C0(a)-2 (the plan's refutation). The defaults-off fixtures of tests/_sgflux_reference.py
    (FD diode, Boltzmann diode on a circuit, DG bar: 26 arrays + currents) and the BTBT FDT diode's noise
    S, its GR part and ssac Y at 1 Hz - 1 GHz are BIT-IDENTICAL with the dispatcher registered first (no
    term) and without -- while the registered child's dispatcher really ran (> 100 assembly calls,
    MEASURED 325). The registered arrays also pass the 6595bde cross-version gate."""
    plain, reg = defaults_pair
    assert plain["registered"] is False and reg["registered"] is True
    assert plain["dispatch_calls"] == 0 and reg["dispatch_calls"] > 100
    assert reg["sgflux"] == plain["sgflux"]
    assert reg["fdt"] == plain["fdt"]
    import _sgflux_reference as REF
    rows = REF.compare(REF.load_reference(), reg["sgflux"])
    assert all(r[3] for r in rows) and len(rows) == REF.N_COMPARED


def test_registration_alone_keeps_the_call_log_digest():
    """GATE C0(a)-3. The D1b call-log digest (tests/_bipolar_calllog.py) recorded with the dispatcher
    registered equals the one recorded without (MEASURED: both 3542731548...205f80, 237 calls = the
    digest pinned in tests/test_bipolar_recombination_bgn_ionization.py)."""
    a, b = _child("calllog"), _child("calllog", register=True)
    assert b["registered"] is True
    assert (b["n"], b["digest"]) == (a["n"], a["digest"])


# =====================================================================================================
# refusals (in-process: they validate before touching DEVSIM)
# =====================================================================================================

class _Local(CT.CustomTerm if CT else object):
    def assemble(self, device, what, timemode):
        return [], []


@pytest.mark.parametrize("loc", ["node", "edge", "element"])
def test_local_terms_are_refused(loc):
    t = _Local("x")
    t.locality = loc
    with pytest.raises(ValueError, match="add_recombination_term"):
        CT.add_custom_term("no_such_device", t)
    assert not CT.is_registered()


def test_undeclared_locality_bad_name_and_type_are_refused():
    with pytest.raises(ValueError, match="nonlocal"):
        CT.add_custom_term("no_such_device", _Local("x"))
    t = _Local("bad name")
    t.locality = "nonlocal"
    with pytest.raises(ValueError, match="must match"):
        CT.add_custom_term("no_such_device", t)
    with pytest.raises(TypeError):
        CT.add_custom_term("no_such_device", object())
    t = _Local("ok")
    t.locality = "nonlocal"
    with pytest.raises(ValueError, match="no device"):
        CT.add_custom_term("no_such_device", t)
    assert not CT.is_registered()


def test_new_symbolic_advisory():
    """A pattern-changing term without a structural pattern warns (AdvisoryWarning) only under
    DEVSIM_NEW_SYMBOLIC=0; a declared pattern or the default '1' is silent (child)."""
    r = _child("symbolic")
    assert len(r["zero_nopattern"]) == 1 and r["zero_nopattern"][0][0] == "AdvisoryWarning"
    assert "DEVSIM_NEW_SYMBOLIC" in r["zero_nopattern"][0][1]
    assert r["zero_pattern"] == [] and r["one_nopattern"] == []


# =====================================================================================================
# toy A: nonlocal Poisson link -- closed-form DC, Jacobian, ssac, conservation
# =====================================================================================================

def _network_C(eps, L, xa, xb, G):
    """Series capacitance per area of [0, xa] | ([xa, xb] || G) | [xb, L] (the oracle)."""
    return 1.0 / (xa / eps + 1.0 / (eps / (xb - xa) + G) + (L - xb) / eps)


@pytest.fixture(scope="module")
def toy_a():
    return _child("toy_a")


def _jac_ok(rec):
    """|FD - (-J)| within the FD budget: truncation <= |fd(h) - fd(2h)| (= 3x the O(h^2) error of
    fd(h)), round-off 8 eps R / h with R = sum_k |J[r, k] x_k| the magnitude of the terms the residual
    row sums (rhs_scale, from the child), and 8 eps |J| for the entry itself."""
    fd, fd2, J = (np.asarray(rec[k]) for k in ("fd", "fd2", "J"))
    tol = np.abs(fd - fd2) + 8 * EPS * rec["rhs_scale"] / rec["h"] + 8 * EPS * np.abs(J)
    return np.abs(fd + J) <= tol, np.abs(fd + J), tol


def test_toy_a_dc_matches_the_closed_form_network(toy_a):
    """DC at V = 1 V: every node potential equals the piecewise-linear network solution and Q_hi = C V.
    Budget: round-off of the 51-node Poisson solve, cond ~ N^2 = 2601 times eps -> 5.8e-13 x V (MEASURED
    1.1e-16 V; Q 1.8e-15). The link is load-bearing: C moves 1.25x (3.453e-4 -> 4.316e-4 F/m^2)."""
    r = toy_a
    L, xa, xb = CH.L_OX, CH.XA, CH.XB
    C = _network_C(r["eps"], L, xa, xb, r["G"])
    flux = C * r["V"]
    x = np.asarray(r["x"])
    psi_a, psi_b = flux * xa / r["eps"], r["V"] - flux * (L - xb) / r["eps"]
    psi = np.where(x <= xa, flux * x / r["eps"],
                   np.where(x >= xb, r["V"] - flux * (L - x) / r["eps"],
                            psi_a + (x - xa) * (psi_b - psi_a) / (xb - xa)))
    tol = 2601 * EPS * r["V"]
    assert np.max(np.abs(np.asarray(r["on"]["psi"]) - psi)) <= tol
    assert abs(r["on"]["Q_hi"] / flux - 1) <= 2601 * EPS
    C_off = _network_C(r["eps"], L, xa, xb, 0.0)
    assert abs(r["off"]["Q_hi"] / (C_off * r["V"]) - 1) <= 2601 * EPS
    assert C / C_off > 1.2


def test_toy_a_charge_conservation_and_wrong_sign(toy_a):
    """Q_lo + Q_hi = 0: the contact fluxes differ only by the summed residual, |F_i| ~ eps T per node,
    T = eps V / h the edge-flux scale -> budget N eps T (MEASURED 7.6e-19 C/m^2 = 0.2 eps T). The WRONG
    relative sign (+1, +1) turns the link into a net source: |Q_lo + Q_hi| = 1.6 |Q| (MEASURED), and the
    ssac capacitance collapses (MEASURED -3.9e-18 vs 4.3e-4)."""
    r = toy_a
    T = r["eps"] * r["V"] / CH.H_OX
    assert abs(r["on"]["Q_lo"] + r["on"]["Q_hi"]) <= len(r["x"]) * EPS * T
    bad = _child("toy_a", signs=[1.0, 1.0])
    assert abs(bad["on"]["Q_lo"] + bad["on"]["Q_hi"]) > 0.1 * abs(r["on"]["Q_hi"])
    C = _network_C(r["eps"], CH.L_OX, CH.XA, CH.XB, r["G"])
    assert abs(bad["ssac_C"][0] / C - 1) > 0.1


def test_toy_a_jacobian_and_ssac(toy_a):
    """The term's share of the ASSEMBLED Jacobian (J with the term minus J without, same state) equals the
    central difference of its share of the assembled residual (rows a, b x columns psi_a, psi_b;
    get_matrix_and_rhs returns rhs = -F, MEASURED, hence FD = -J; MEASURED 1.4e-14 relative); the ssac C
    at 1 kHz and 1 MHz equals the closed-form C (the custom static Jacobian enters the small-signal
    matrix; MEASURED 1.1e-15) and the DC dQ/dV (1.1e-13); G_ssac = 0 for a lossless dielectric.
    Budgets: 2601 eps (solve) for C; the DC central difference adds eps |Q| / dv relative (2.2e-13)."""
    r = toy_a
    for rec in r["jac"]:
        ok, err, tol = _jac_ok(rec)
        assert ok.all(), (rec["node"], err, tol)
    C = _network_C(r["eps"], CH.L_OX, CH.XA, CH.XB, r["G"])
    for c in r["ssac_C"]:
        assert abs(c / C - 1) <= 2601 * EPS
    assert all(abs(g) <= 2601 * EPS * 2 * math.pi * 1e3 * C for g in r["ssac_G"])
    assert abs(r["dQdV"] / C - 1) <= 2601 * EPS + 2 * EPS * r["V"] / 1e-3


def test_toy_a_extended_precision_converges_linearly():
    """User decision 8: with a custom term the 128-bit Newton converges LINEARLY (the term is double-
    valued). Gate: it converges to rel 1e-30 within 60 iterations, NOT in the <= 5 the term-free device
    takes (MEASURED 4 without, 34 with: factor ~2.7 per iteration here, where the link equals the a-b
    segment's own coupling), its relative updates decrease monotonically after the first two, and the
    contact charges then cancel exactly and match the closed form to the double rounding of the term
    (budget 16 eps; MEASURED 0 and 1 ulp)."""
    r = _child("toy_a", extended=True, rel=1e-30)
    assert r["extended_converged"]
    rel = r["extended_rel"]
    assert 10 <= len(rel) <= 60
    assert all(b < a for a, b in zip(rel[2:], rel[3:]))
    assert r["on"]["Q_lo"] + r["on"]["Q_hi"] == 0.0
    C = _network_C(r["eps"], CH.L_OX, CH.XA, CH.XB, r["G"])
    assert abs(r["on"]["Q_hi"] / (C * r["V"]) - 1) <= 16 * EPS


# =====================================================================================================
# toy B: quasi-Fermi electron bridge in an n bar -- conductance, Johnson through the two-point source,
# terminal-current conservation, Jacobian, refusals, frozen continuity
# =====================================================================================================

@pytest.fixture(scope="module")
def toy_b():
    return _child("toy_b")


def _network_G(r):
    nd, ni = CH.ND_RES, 1e16
    n_eq = 0.5 * nd + math.sqrt(0.25 * nd * nd + ni * ni)
    sigma = QE * r["mu_n"] * n_eq
    L, xa, xb = CH.L_RES, CH.XA_R, CH.XB_R
    return 1.0 / (xa / sigma + (xb - xa) / (sigma + r["G"] * (xb - xa)) + (L - xb) / sigma), sigma


def test_toy_b_conductance_matches_the_closed_form_network(toy_b):
    """At equilibrium (uniform n = n_eq, zero field) the linear response of the bar is EXACTLY the
    resistor network for the quasi-Fermi potential, so G = 1/(x_a/sigma + (x_b - x_a)/(sigma + G_c
    (x_b - x_a)) + (L - x_b)/sigma). ssac Re Y(1 Hz, 1 MHz) and the DC dI/dV (1 mV central difference,
    Richardson-corrected with the 2 mV one) must equal it to 1e-9 -- the D2a gate-A1 budget for sigma A/L
    on the same kind of bar (its MEASURED floor 1.4e-12). MEASURED here: ssac <= 5.7e-13, DC 2.7e-13 (the
    raw 1 mV difference is 1.3e-8 off: its O(dv^2) term); the term-free bar 3.6e-13 from sigma/L. The
    bridge is load-bearing: G moves 1.25x."""
    r = toy_b
    G, sigma = _network_G(r)
    for y in r["ReY"]:
        assert abs(y / G - 1) < 1e-9
    g_rich = r["G0_on"] + (r["G0_on"] - r["G0_on_2mV"]) / 3.0
    assert abs(g_rich / G - 1) < 1e-9
    assert abs(r["G0_off"] / (sigma / CH.L_RES) - 1) < 1e-9
    assert G / (sigma / CH.L_RES) > 1.2


def test_toy_b_johnson_through_the_two_point_source(toy_b):
    """Johnson-Nyquist S = 4 kT G with G the closed-form network conductance (the FDT oracle): the noise
    sum includes the bridge's TWO-POINT source (mechanism 'tunnel_ox' borrowed for the toy: a carrier
    moved between two remote rows), budget 1e-9 as gate A1 (MEASURED <= 6.1e-13). Refute: leaving the
    bridge's source out (mechanisms without 'tunnel_ox') misses by its share, MEASURED -12.5 %.
    (The bridge source carries G_c |y_a - y_b|^2 4kT -- MEASURED 12.5 % of S, as the network predicts.)"""
    r = toy_b
    G, _ = _network_G(r)
    T = r["vt"] * QE / KB
    S = np.asarray(r["S"])
    assert np.max(np.abs(S / (4 * KB * T * G) - 1)) < 1e-9
    assert r["by_mechanism"]["tunnel_ox"][0] / S[0] > 0.1
    drop = _child("toy_b", mechanisms=["diffusion_n", "diffusion_p", "gr"])
    assert "tunnel_ox" not in drop["by_mechanism"]
    assert abs(drop["S"][0] / (4 * KB * T * G) - 1) > 0.05


def test_toy_b_terminal_current_conservation_and_wrong_sign(toy_b):
    """At 0.05 V the source current and the grounded-contact current cancel: their sum is the summed
    residual of the carrier and KCL rows (rhs, read back) plus the telescoping round-off of N edge fluxes
    of size T = q mu V_t n / h -> budget sum|rhs| + N eps T (MEASURED 2.4e-7 A/m^2 = 1.8e-14 relative).
    The wrong relative sign (+1, +1) is not conserved: MEASURED I_lo + I_src = -2.2e7 vs I 1.4e7."""
    r = toy_b
    b = r["bias"]
    T = QE * r["mu_n"] * r["vt"] * CH.ND_RES / CH.H_RES
    assert abs(b["I_src"] + b["I_lo"]) <= b["resid_sum"] + r["n_edges"] * EPS * T
    assert abs(b["term_I"]) > 0.1 * abs(b["I_src"])                      # the bridge carries current
    bad = _child("toy_b", signs=[1.0, 1.0])["bias"]
    assert abs(bad["I_src"] + bad["I_lo"]) > 0.1 * abs(b["I_src"])


def test_toy_b_jacobian(toy_b):
    """The term's assembled Jacobian share vs the central difference of its residual share at rows a, b,
    columns psi_a, psi_b (h = 0.1 mV) and n_a, n_b (h = 1e-4 n), budget of _jac_ok. MEASURED 1.1e-13
    (psi) and 3.3e-9 (n: exactly the O(h^2) truncation h^2 / (3 n^2) of V_t ln n) relative."""
    for rec in toy_b["jac"]:
        ok, err, tol = _jac_ok(rec)
        assert ok.all(), (rec["node"], rec["var"], err, tol)


def test_noise_refuses_terms_without_noise_model_or_exact_jacobian():
    a = _child("toy_b", noise=False)
    assert "no noise model" in a["noise_error"] and "S" not in a
    b = _child("toy_b", jacobian_exact=False)
    assert "jacobian_exact=False" in b["noise_error"] and "S" not in b


def test_unified_inventory_lists_custom_terms_and_noise_refuses_through_it():
    """Plan 4.0.2 (the "custom" kind; noise_analysis reads the ONE inventory, physics_bipolar_dd.
    recombination_terms, instead of custom_terms.live_terms). A term is listed in the regions of the rows
    it declares -- or, declaring none (unknown footprint), in EVERY region -- with its noise capability
    (overrides noise_sources) and Jacobian exactness, read through custom_terms' DEVSIM-marker liveness.
    The noise refusals fire through it (in name order): the undeclared no-noise term first, then the
    inexact Jacobian; with both removed the noise runs. Re-creating the device under the same name lists
    nothing (the marker parameter died with it). OLD CODE (8bbf36c): recombination_terms has no "custom"
    key -> KeyError in the child (the test fails)."""
    r = _child("inventory")
    assert r["before"] == {}
    assert r["with"] == {"anon": ["anon", [], False, True], "bridge": ["bridge", ["bulk"], True, True],
                         "inexact": ["inexact", ["bulk"], True, False]}
    assert "no noise model" in r["err_all"] and "'anon'" in r["err_all"]
    assert "jacobian_exact=False" in r["err_inexact"] and "'inexact'" in r["err_inexact"]
    assert r["err_none"] is None
    assert r["after_remove"] == {"bridge": ["bridge", ["bulk"], True, True]}
    assert r["recreated"] == {}


def test_frozen_continuity_poisson_presolve_with_a_live_term(toy_b):
    """GATE C0(c). With the continuity equations deleted (eq_registry.delete_by_name, the frozen-carrier
    pre-solve), the live term reports rows_live False, the dispatcher still runs (MEASURED 2 calls) and
    contributes nothing, and the Poisson-only solve converges; after re-adding, rows_live is True."""
    r = toy_b
    assert r["rows_live_frozen"] is False and r["frozen_solved"] is True
    assert r["frozen_calls"] >= 1 and r["rows_live_after"] is True


# =====================================================================================================
# liveness and the fixed pattern
# =====================================================================================================

def test_liveness_after_recreating_the_device():
    """GATE C0(b). Add a term, delete the device and re-create it under the SAME name: the solve and the
    assembled matrix are BIT-IDENTICAL to the term-free device, the stale entry is purged. Refute 1:
    forging the marker parameter (= a name-only liveness check) makes the stale term apply again
    (results differ). Refute 2 (plan): same name, different region name -> no crash, not applied."""
    r = _child("liveness")
    assert r["on"]["psi"] != r["ref"]["psi"]                       # the term did act
    assert r["recreated"]["psi"] == r["ref"]["psi"]
    assert r["recreated_matrix"] == r["ref_matrix"]
    assert r["live_terms"] == [] and r["registry"] == {}
    f = _child("liveness", forge_marker=True)
    assert f["recreated"]["psi"] != f["ref"]["psi"] and f["live_terms"] == ["link"]
    g = _child("liveness", region="ox2")
    assert g["recreated"]["psi"] == g["ref"]["psi"] and g["live_terms"] == []


def test_structural_pattern_reserves_slots_without_changing_values():
    """DEVSIM drops exact zeros (MEASURED), so the declared candidate entries are emitted as +eps / -eps
    pairs (module doc, section 4): nnz grows by exactly the two extra candidates, those slots hold 0.0,
    every other entry is bitwise unchanged, and the DC solution moves only by the round-off of a
    different factorisation pattern (budget 2601 eps V, MEASURED 1 ulp). An entry outside the declared
    pattern aborts the solve with an error naming the term."""
    import scipy.sparse as sp
    r = _child("structural")
    P = sp.csc_matrix(tuple(np.asarray(v) for v in r["plain"][:3]))
    S = sp.csc_matrix(tuple(np.asarray(v) for v in r["struct"][:3]))
    assert S.nnz == P.nnz + 2
    rows = [int(x) for x in r["rows"]]
    for rw in rows:
        assert P[rw, r["col_c"]] == 0.0 and S[rw, r["col_c"]] == 0.0
    coo = S.tocoo()
    pattern = set(zip(coo.row.tolist(), coo.col.tolist()))
    assert all((rw, r["col_c"]) in pattern for rw in rows)
    assert (S - P).count_nonzero() == 0
    assert np.max(np.abs(np.asarray(r["psi_struct"]) - np.asarray(r["psi_plain"]))) <= 2601 * EPS
    assert "outside its declared structural_pattern" in r["escape_error"]
    assert "'link'" in r["escape_error"]
