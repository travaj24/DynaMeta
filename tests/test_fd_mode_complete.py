"""statistics="fermi_dirac" completed (v0.15): local BTBT (incl. tuples), nonlocal BTBT, charged bulk traps,
interface traps and thermionic heterointerfaces under FD detailed balance, with the FD quasi-field transport
(physics_heterojunction.fd_quasi_field_expr). Audit note: docs/audit/2026-11-06-v0.15-fermi-dirac-mode-complete.md.

ORACLES. (1) Thermodynamics: at equilibrium every net rate and every current is zero, and the equilibrium noise
obeys S = 4 kT Re Y (Nyquist / fluctuation-dissipation). Re Y comes from DEVSIM ssac where it is resolvable, or
from a Richardson DC conductance of solved currents (independent of the noise module) where ssac has its
documented ~0.3-1 S/m^2 floor. (2) Central differences of DEVSIM's own assembled residual for the Jacobian.
(3) For the non-degenerate limit, the legacy statistics themselves: the two must agree to within the size of
the FD correction (|Delta| <= delta_max, the largest degeneracy offset on the device).

EVERY GATE IS SHOWN FAILING on the legacy statistics (statistics=None, the unchanged default) with the SAME
mechanism, and on 06b9ade (the pre-change tree) the FD-mode devices do not build at all: setup_bipolar_region
raised NotImplementedError for btbt / charged traps, install_interface_traps and setup_btbt_nonlocal refused
statistics='fermi_dirac' (tests/test_si_bundles_devsim.py G6 on that tree). The adversarial half of each
Jacobian gate knocks the FD degeneracy-factor derivatives out (the Jacobian a fixed-n_ie implementation would
assemble) and must fail.

All DEVSIM work runs in child processes (tests/_fd_mode_child.py: the nonlocal term's registration and the FDT
circuits are process-global). Runtime 4.5 min with -n 2 (measured; 40 tests, none above 42 s).
"""
import os
import sys

import numpy as np
import pytest

pytest.importorskip("devsim")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _fd_mode_child as C  # noqa: E402

EPS = float(np.finfo(float).eps)


def _rb(**kw):
    return C.run_child("zero_bias", **kw)


def _budget(r):
    """Round-off budget of a net rate at an exact equilibrium, relative to the one-way rate: the local
    activities agree to ~ eps |L| (|L| <= Lmax) and the rate expression adds a few ulps: 64 eps (1 + Lmax)."""
    return 64.0 * EPS * (1.0 + r["Lmax"])


# =====================================================================================================
# Z: zero-bias detailed balance on degenerate junctions, each mechanism ON
# =====================================================================================================

# (mechanism options, legacy |net|/one-way MEASURED at h and h/2 -- docstring numbers)
_ZB = {
    "btbt": dict(btbt="hurkx"),
    "btbt_tuple": dict(btbt="tuple"),
    "nonlocal": dict(nonlocal_=1),
    "bulk_traps": dict(traps=1, device="diode"),
    "interface_traps": dict(device="mos", vg=-0.3),
}


def _opts(key, **extra):
    o = dict(_ZB[key])
    if "nonlocal_" in o:
        o["nonlocal"] = o.pop("nonlocal_")
    o.update(extra)
    return o


def _ratio(r):
    return r["max_path_ratio"] if r["max_path_ratio"] is not None else abs(r["net"]) / r["one_way"]


@pytest.mark.parametrize("key", sorted(_ZB))
def test_z_zero_bias_detailed_balance_is_exact(key):
    """GATE Z. The device of each mechanism (zener: p+ 1e25 / n+ 1e26 m^-3, eta_n = +2.6, 0.2 nm; diode: n+ 1e26 /
    p 1e23 with a shallow acceptor 0.05 eV below E_c, a donor 0.06 eV above E_v and a neutral midgap level;
    mos: n+ 1e26 / SiO2 with a split D_it 3e16 m^-2 eV^-1, gate -0.3 V) at zero bias under
    statistics='fermi_dirac': the mechanism's net rate (q Int U, or each nonlocal path's |R|) relative to its
    one-way rate is round-off -- 64 eps (1 + Lmax) -- at h AND at h/2 (exact: no h-dependence), on a converged
    state (relative residual <= 1e-12). The same mechanism at the EXACT discrete equilibrium constructed on the
    solved psi (flat activities, Newton on DEVSIM's own HJ_L models) is round-off node by node.
    MEASURED FD (h): btbt 4.2e-15, tuple 4.2e-15, nonlocal paths 6.0e-15, bulk traps 4.0e-16, interface 1.2e-15.
    SHOWN FAILING on the legacy statistics, same device and mesh (MEASURED |net|/one-way): btbt 2.52e-2,
    nonlocal 2.44e-2, bulk traps 0.121, interface 0.406 -- see test_z_legacy_imbalance for the split into the
    statistics defect and the transport's discretisation defect."""
    for h_scale in (1.0, 0.5):
        o = _opts(key, stats="fermi_dirac")
        if "h" in o or True:
            base = 1e-9 if o.get("device") == "diode" else (1e-10 if o.get("device") == "mos" else 2e-10)
            o["h"] = base * h_scale
        r = _rb(**o)
        assert r["residual"] <= 1e-12, r["residual"]
        b = _budget(r)
        assert _ratio(r) <= b, (key, h_scale, _ratio(r), b)
        assert r["construction_residual"] <= 64.0 * EPS * (1.0 + r["Lmax"])
        eq = r["eq_max_path_ratio"] if r["eq_max_path_ratio"] is not None else r["eq_worst_local"]
        assert eq <= b, (key, eq, b)


@pytest.mark.parametrize("key", sorted(_ZB))
def test_z_legacy_imbalance(key):
    """The LEGACY statistics (statistics=None, the default) with the same mechanism, same devices: the measured
    zero-bias imbalance |net|/one-way at h and h/2 (the numbers of the audit note's decision table), shown
    above the FD budget by >= 1e3. Two defects add up there: (a) the STATISTICS defect (Boltzmann-product
    n_i in D / n1 / the contacts' minority pin), h-independent; (b) the edge-averaged g transport's O(h^2)
    zero-bias battery (removed in FD mode by the quasi-field). Where the legacy imbalance shrinks ~4x on halving
    h it is (b); where it does not, (a). The exact-equilibrium construction isolates (a): the legacy local rates
    at the FD equilibrium (eq_worst_local) are O(Delta)."""
    out = {}
    for h_scale in (1.0, 0.5):
        o = _opts(key, stats="legacy")
        base = 1e-9 if o.get("device") == "diode" else (1e-10 if o.get("device") == "mos" else 2e-10)
        o["h"] = base * h_scale
        r = _rb(**o)
        out[h_scale] = r
        assert _ratio(r) >= 1e3 * _budget(r), (key, h_scale, _ratio(r))
    print(key, {k: (_ratio(v), v.get("eq_worst_local")) for k, v in out.items()})


def test_z_heterostack_battery_is_gone():
    """GATE Z-hetero. The n+ 1e26 / n 3e25 m^-3 thermionic isotype heterostack (dchi 0.1 eV, A* x 1e-2), both
    contacts at 0 V: the zero-bias current (Richardson of +-1e-4 / +-2e-4 V solves) over the conductance -- the
    effective battery -- is round-off under statistics='fermi_dirac' (MEASURED 8.3e-14 V at h = 1 nm and
    8.9e-14 V at 0.5 nm; budget 1e-12 V: the +-dv Richardson leaves O(dv^4) of the I-V curvature, < 1e-14 V
    here, and the currents' round-off ~1e-3 A/m^2 over G = 2e10 S/m^2 is 5e-14 V). SHOWN FAILING on the legacy
    statistics: 1.35e-4 V at h = 1 nm, 3.97e-5 V at 0.5 nm (the averaged-g battery, O(h^1.8))."""
    fd = [C.run_child("zero_bias_hetero", stats="fermi_dirac", h=h) for h in (1e-9, 5e-10)]
    legacy = [C.run_child("zero_bias_hetero", stats="legacy", h=h) for h in (1e-9, 5e-10)]
    for r in fd:
        assert abs(r["V_battery"]) <= 1e-12, r
    assert abs(legacy[0]["V_battery"]) > 1e-5 and abs(legacy[1]["V_battery"]) > 1e-6
    assert 2.5 < legacy[0]["V_battery"] / legacy[1]["V_battery"] < 4.5          # discretisation, not physics


# =====================================================================================================
# J: exact Jacobians (DEVSIM matrix vs central differences of DEVSIM's residual)
# =====================================================================================================

@pytest.mark.parametrize("opts", [dict(btbt="hurkx"), dict(btbt="tuple"), dict(traps=1, device="diode"),
                                  dict(device="mos", vg=-0.3)],
                         ids=["btbt", "btbt_tuple", "bulk_traps", "interface_traps"])
def test_j_jacobian_matches_central_differences(opts):
    """GATE J. Under statistics='fermi_dirac', off equilibrium (zener -0.5 V; diode +0.3 V; mos -0.3 V with the
    electrons x1.3), every column (Potential, Electrons, Holes) at the nodes nearest the junction / interface:
    |J - fd| <= 1e-6 S_r/|x| + 2 |fd - fd2| on every row (S_r the row's term scale; the 2h step estimates the
    truncation) -- normalised miss <= 1. MEASURED 2.7e-4 (btbt), 3.8e-5 (bulk traps), 5.3e-4 (interface).
    ADVERSARIAL: the FD degeneracy-factor derivatives knocked out (FDdLnGamma* = 0 and the BTBT edge copies
    FDNiN@n0:Electrons@n0 ... = 0, i.e. the fixed-n_ie Jacobian) -> MEASURED 24 / 13 / 18: fails."""
    r = C.run_child("jacobian", stats="fermi_dirac", knockout=1, **opts)
    assert r["worst"] <= 1.0, r["where"]
    assert r["ko_worst"] > 1.0, r["ko_where"]


def test_j_jacobian_2d_element_btbt():
    """GATE J-2D. The ELEMENT assembly of local BTBT (2-D tensor mesh, p+ 1e25 / n+ 1e26, -0.3 V) under
    statistics='fermi_dirac': same criterion. MEASURED 6.3e-4 (single set and tuple); knockout 3.7e4."""
    r = C.run_child("jacobian_2d", stats="fermi_dirac", knockout=1)
    assert r["worst"] <= 1.0, r["where"]
    assert r["ko_worst"] > 1.0
    r2 = C.run_child("jacobian_2d", stats="fermi_dirac", btbt="tuple")
    assert r2["worst"] <= 1.0, r2["where"]


def test_j_nonlocal_term_jacobian_under_fermi_dirac():
    """GATE J-NL. The nonlocal term's own Jacobian share (tests/_custom_terms_child.fd_columns, the shared
    helper) at a -0.5 V degenerate Zener under statistics='fermi_dirac': |J - fd| <= 1e-6 R + 2|fd - fd2| on the
    start / end rows, carriers and psi columns. MEASURED 0.31 (normalised)."""
    r = C.run_child("jacobian_nonlocal", stats="fermi_dirac")
    assert r["n"] >= 6 and r["max_J"] > 0.0
    assert r["worst"] <= 1.0


# =====================================================================================================
# F: fluctuation-dissipation, each mechanism, statistics='fermi_dirac'
# =====================================================================================================

def _dc_budget(r):
    """S/(4 kT G_dc) - 1 budget: the Richardson DC conductance's remainder (4/9)((g1 - g2)/G)^2 (the dv^4 term,
    bounded by the dv^2 spread it removed) + 1e-9 (the +-1e-7 V average's (V0/V_t)^2 = 1.5e-11 and the noise
    sum's round-off)."""
    return 0.5 * (r["G_spread"] / r["G_dc"]) ** 2 + 1e-9


@pytest.mark.parametrize("opts", [dict(traps=1, trap_N=1e22), dict(nonlocal_=1), dict()],
                         ids=["bulk_traps", "nonlocal", "srh_only"])
def test_f_fdt_against_dc_conductance(opts):
    """GATE F. The FDT diode (n+ 3e25 m^-3 degenerate / p 1e22, n_i 3e19) under statistics='fermi_dirac':
    S(1 Hz) = 4 kT G_dc with G_dc a Richardson difference of solved DC currents (ssac's ~0.3 S/m^2 floor is
    4e-4 of this device's 703 S/m^2, MEASURED). The mechanism carries >= 15 % of S (bulk traps 70 % gr,
    nonlocal 18 %). MEASURED: bulk traps 6.9e-12, srh-only 1.1e-11 (budget _dc_budget ~1e-9); nonlocal -3.0e-8
    (budget: the nonlocal gates' own FDT tolerance 1e-4, N9 of docs/audit/2026-11-06-v0.15-nonlocal-btbt-1d.md;
    its Boltzmann value there 7e-7). SHOWN FAILING on the legacy statistics: bulk traps -5.3e-3, nonlocal
    -4.8e-3 (MEASURED)."""
    o = dict(opts)
    if "nonlocal_" in o:
        o["nonlocal"] = o.pop("nonlocal_")
    r = C.run_child("fdt", stats="fermi_dirac", dc=1, **o)
    # the decision-39 advisory: quasi-static CHARGED levels in a noise / ssac solve (exact only well below the
    # capture rates -- the 1 Hz identity gated here; the kinetic twin is gate K-FD-F)
    assert (r["advisories"] > 0) == bool(o.get("traps")), r["advisories"]
    tol = 1e-4 if o.get("nonlocal") else _dc_budget(r)
    assert abs(r["fdt_dc"]) <= tol, (r["fdt_dc"], tol)
    if o.get("traps"):
        assert r["share"]["gr"] >= 0.5
    if o.get("nonlocal"):
        assert r["share"]["btbt_nl"] >= 0.15
    if o:
        legacy = C.run_child("fdt", stats="legacy", dc=1, **o)
        assert abs(legacy["fdt_dc"]) > 1e-3


def test_f_local_btbt_fdt_is_the_edge_assembly_discretisation():
    """GATE F-BTBT (and a FINDING). The local BTBT (1-D EDGE assembly: each edge's rate uses the MEAN of its two
    nodes' Hurkx D, delivered to both nodes) has an S = 4kT G_dc miss that is the same under Boltzmann and FD
    statistics and shrinks 4x on halving h: MEASURED -6.35e-5 (FD) / -6.35e-5 (Boltzmann) at h = 1 nm, -1.59e-5
    at 0.5 nm (BTBT gr share 90 %). Its linear response couples the two nodes of an edge asymmetrically (node 0's
    rate depends on node 1's D with node 1's weight), which no node-local GR source reproduces (Onsager symmetry
    is broken at O(h)) -- a pre-existing discretisation property of the 1-D assembly, NOT of the statistics; the
    2-D/3-D element assembly uses each node's own D. Gate: FD within 5 % of Boltzmann, ratio 4 +/- 0.6 on
    halving h, and |miss| <= 1e-4 at 1 nm. The legacy statistics: +3.3e-3 (MEASURED)."""
    fd1 = C.run_child("fdt", stats="fermi_dirac", btbt="fdt", dc=1, h=1e-9)
    fd2 = C.run_child("fdt", stats="fermi_dirac", btbt="fdt", dc=1, h=5e-10)
    bz = C.run_child("fdt", stats="boltzmann", btbt="fdt", dc=1, h=1e-9)
    assert fd1["share"]["gr"] >= 0.5
    assert abs(fd1["fdt_dc"]) <= 1e-4
    assert abs(fd1["fdt_dc"] / bz["fdt_dc"] - 1.0) <= 0.05
    assert 3.4 <= fd1["fdt_dc"] / fd2["fdt_dc"] <= 4.6


@pytest.mark.parametrize("unipolar", [False, True], ids=["bipolar_fermi_dirac", "unipolar_consistent"])
def test_f_thermionic_heterostack_fdt(unipolar):
    """GATE F-TE (the thermionic-interface FDT residual reported by the noise-method work). ROOT CAUSE
    (MEASURED): not the thermionic flux / pair strings -- J = A*T^2 [exp(w0) - exp(w1)] in the FD activities
    is exact at equilibrium and Nyquist-consistent -- but the edge-averaged g of the FD transport: its discrete
    zero-current state is off the activity equilibrium by O(h^2) (a zero-bias battery, gate Z-hetero) and its
    edge conductance is not the noise module's q mu n_eff. With the FD quasi-field (bipolar
    statistics='fermi_dirac', bipolar or unipolar): ssac Re Y vs S at 1 Hz and 1 MHz,
    +-1e-7 V (bipolar) / +-1e-6 V (unipolar) averaged. Budget 1e-9: ssac's ~1 S/m^2 floor over Re Y (1.9e10 /
    1.3e11 S/m^2) is < 1e-10, the even (V0/V_t)^2 term 1.5e-11 / 1.5e-9 x the curvature (resistor-like:
    << 1). MEASURED -1.1e-11 (bipolar, te share 99 %) and 9.1e-12 (unipolar ITO-like 3e26 / AZO-like 3e25,
    dchi 0.3 eV). SHOWN FAILING with the averaged g: bipolar legacy 6.8e-3; unipolar 'average' 2.09e-6 at
    0.5 nm (and 5.37e-7 at 0.25 nm: O(h^2))."""
    kw = dict(unipolar=1, edge="consistent") if unipolar else dict(stats="fermi_dirac")
    r = C.run_child("fdt_hetero", **kw)
    assert max(abs(v) for v in r["fdt"]) <= 1e-9, r["fdt"]
    if not unipolar:
        assert r["share"]["te_n"] >= 0.5
    old = C.run_child("fdt_hetero", **(dict(unipolar=1, edge="average") if unipolar else dict(stats="legacy")))
    assert max(abs(v) for v in old["fdt"]) > 1e-6


# =====================================================================================================
# N: the non-degenerate limit
# =====================================================================================================

@pytest.mark.parametrize("opts", [dict(btbt="hurkx", NA=1e24, ND=1e24), dict(btbt="hurkx", NA=3e24, ND=3e24),
                                  dict(device="diode", traps=1, ND=1e23), dict(device="diode", traps=1, ND=1e24),
                                  dict(device="mos", ND=1e23, vg=-0.2), dict(device="mos", ND=1e24, vg=-0.2)],
                         ids=["btbt_1e24", "btbt_3e24", "traps_1e23", "traps_1e24", "it_1e23", "it_1e24"])
def test_n_non_degenerate_limit_equals_legacy(opts):
    """GATE N. On non-degenerate devices the FD mode and the legacy statistics differ only through the
    degeneracy factors gamma = exp(-Delta), |gamma - 1| <= delta_max (the largest Delta(n/N_c), Delta(p/N_v) on
    the solved device); each mechanism carries at most two of them per carrier (its n1/D densities and the
    contact pin), so the relative difference of its operating-point quantity is <= 4 delta_max. Quantities:
    btbt -- q Int G at -0.6 V (MEASURED rel diff 3.1e-5 / 2.5e-5 at delta_max 0.035 / 0.105); bulk traps -- the
    trap recombination current at +0.3 V (1.8e-5 / 7.0e-6 at 2.6e-3 / 1.2e-2); interface -- the one-way surface
    rate at equilibrium (3.7e-4 / 5.7e-3 at 1.2e-3 / 1.27e-2). The legacy statistics' own zero-bias interface
    imbalance scales WITH delta_max (MEASURED 7.5e-4 / 1.13e-2 of the one-way: 0.61 / 0.89 delta_max) -- it
    vanishes only in the non-degenerate limit, where the two statistics coincide."""
    fd = C.run_child("nondeg", stats="fermi_dirac", **opts)
    lg = C.run_child("nondeg", stats="legacy", **opts)
    assert fd["residual"] <= 1e-12 and lg["residual"] <= 1e-12
    key = "one_way" if opts.get("device") == "mos" else "value"
    rel = abs(fd[key] / lg[key] - 1.0)
    assert rel <= 4.0 * fd["delta_max"], (rel, fd["delta_max"])
    if opts.get("device") == "mos":
        assert abs(fd["value"]) <= 64.0 * EPS * 60.0 * fd["one_way"]       # FD: balanced (Z budget, |L| < 60)
        assert abs(lg["value"]) / lg["one_way"] <= 1.0 * lg["delta_max"]   # legacy: O(delta) only


# =====================================================================================================
# D: defaults off; refusals that remain
# =====================================================================================================

# sha256 of tests/_fdext_calllog.record_call_log() (501 calls: the 1-D legacy-statistics pn region with BGN, a
# shifted built-in SRH level, charged + neutral bulk levels of all three energy references, a BTBT TUPLE, an
# SiO2 region with interface traps; and a 2-D BGN slab with one BTBT set), RECORDED on 06b9ade (the commit this
# work was cut from) before any edit; this tree gives the same log (MEASURED: the JSON logs compare equal).
# RE-RECORDED 2026-11-08 as the CANONICAL log (correctly rounded transcendentals, tests/_cr_libm.py), again on
# an export of 06b9ade: 71a3219d... on Windows AND Linux, = this tree's on both. The pre-2026-11-08 digest
# dcca0b093e85159a6a3e7dfdfce090fba3c6e63162035fa4290e4c5ae7501f49 was of the raw Windows-UCRT log (06b9ade and
# this tree reproduce it there; glibc gives the canonical one raw): docs/audit/2026-11-08-v0.15-linux-ci-exactness.md.
_FDEXT_CALLLOG_DIGEST = "71a3219d99b19df76013665cfe42c7a38e2157464c6a9ab7108526c68616a360"


def test_d_legacy_statistics_definition_is_byte_identical():
    """GATE D (REVISED 2026-11-07 for the v0.15.0 default change). statistics='legacy' -- the v0.14 default,
    kept reachable explicitly -- reproduces the recorded digest of every legacy path (recorded on 06b9ade, 501
    calls) character-for-character; the DEFAULT (statistics unset) now equals statistics='fermi_dirac' and
    differs from it (the intended change: docs/audit/2026-11-06-v0.15.0-release.md, decision (a)).
    REVISED 2026-11-08 (Linux CI): the digest was of repr() of floats some of which pass through a C-library
    transcendental, whose last bit IEEE-754 leaves open. The raw logs of Windows (UCRT) and Linux (glibc 2.39)
    differ in ONE literal, interface-trap bin 14's n1 = n_i exp(-7.384693350077327): 6206809661957.206 (UCRT,
    0.510 ulp from exact) vs ...205 (glibc, 0.490 ulp: correctly rounded), 12 occurrences in 9 calls (MEASURED;
    the skeletons with numbers masked are equal). The log is now recorded in its CANONICAL form
    (tests/_cr_libm.py: math / numpy exp, log, log10, log2, expm1, log1p, power correctly rounded through
    decimal at 60 digits; the other transcendentals refused), a platform-independent function of the code that
    still pins every IEEE operation bit for bit. STILL BITES (MEASURED on Windows and Linux, canonical):
    (a) n1 = n_i (3 exp(e)) / 3 instead of n_i exp(e) -- one extra rounding, mathematically equal: 370 literals
    move by 1-2 ulps, the skeleton is unchanged -- changes the digest (a tolerance on the literals would not
    see it); (b) the FD reference densities in the legacy SRH string (srh_reference_densities fd=True) change
    it. NOT COVERED: the
    C pow() behind the ** operator, which cannot be intercepted; the fixture reaches it in import-time preset
    constants (tunneling.py: 100.0 ** -P, E_J ** 1.5; constants.py C ** 2; bipolar_options.py x ** (1/3)) and
    in fd_potential_shift (physics_bipolar_dd.py: s ** 3), all bitwise equal on UCRT and glibc (MEASURED: the
    raw logs differ only in the exp literal above)."""
    import _fdext_calllog as CL
    import devsim  # noqa: F401
    log = CL.record_call_log(statistics="legacy")
    assert len(log) == 501 and CL.digest(log) == _FDEXT_CALLLOG_DIGEST
    # default vs explicit FD: two logs of ONE process (one libm), compared bitwise -- platform-stable raw
    default = CL.record_call_log(canonical=False)
    assert default == CL.record_call_log(statistics="fermi_dirac", canonical=False)
    assert CL.skeleton(default) != CL.skeleton(log)              # differs from legacy in its TEXT, not by libm


def test_d_kernels_default_strings_unchanged_and_fd_guards():
    """The string kernels: fd=False / n_i_p=None are the legacy strings character-for-character; fd=True needs
    the BGN (n_ie node) path; the FD Hurkx factor is the two-density form."""
    from dynameta.carriers import physics_traps as PT
    from dynameta.carriers.tunneling import hurkx_d_expr
    assert PT.srh_rate_expr() == "(Electrons*Holes - n_i^2)/(taup*(Electrons + n1) + taun*(Holes + p1))"
    assert PT.srh_reference_densities(bgn=True, fd=False) == ("n1*n_ie/n_i", "p1*n_ie/n_i", "n_ie^2")
    assert PT.srh_reference_densities(bgn=True, fd=True) == ("n1*FDNiN/n_i", "p1*FDNiP/n_i", "n_ie^2")
    assert PT.srh_reference_densities(bgn=True, fd=True, reference="conduction") == (
        "n1*FDGammaN", "(n_ie^2/(n1*FDGammaN))", "n_ie^2")
    with pytest.raises(ValueError, match="needs bgn=True"):
        PT.srh_reference_densities(fd=True)
    assert hurkx_d_expr("n", "p") == "((n)*(p) - n_i^2)/(((n) + n_i)*((p) + n_i))"
    assert hurkx_d_expr("n", "p", "A", n_i_p="B") == "((n)*(p) - A*B)/(((n) + A)*((p) + B))"


def test_d_nonlocal_still_refuses_bgn_under_fermi_dirac():
    """setup_btbt_nonlocal accepts statistics='fermi_dirac' WITHOUT narrowing (gated above) and still refuses a
    narrowed region (its gap would vary along the path) in either statistics."""
    r = C.run_child("nonlocal_refusals")
    assert r["fd_plain"] == "ok"
    assert "bandgap narrowing" in r["fd_bgn"] and "bandgap narrowing" in r["legacy_bgn"]


# =====================================================================================================
# K-FD: KINETIC traps (bulk levels and interface bins, v0.15 4.3.2) under statistics='fermi_dirac'
# (integration with the kinetic-traps work: the exact FD emission densities in srh_capture_exprs / odds /
# vacancy / the DC-form occupancy derivatives)
# =====================================================================================================

_KIN = {"bulk": dict(traps=1, device="diode", kinetic=1, h=1e-9),
        "interface": dict(device="mos", vg=-0.3, kinetic=1, h=1e-10, kbin=0.3)}   # 4 kinetic bins (cost)


@pytest.mark.parametrize("key", sorted(_KIN))
def test_kfd_zero_bias_detailed_balance_is_exact(key):
    """GATE K-FD-Z. The Z devices with the levels / bins KINETIC (their occupancy odds a solution variable,
    seeded and solved through the kinetic DC form, then the physical form): per exchange branch,
    Int |R_n| (and |R_p|) over its one-way rate Int P/2 is round-off, 64 eps (1 + Lmax), on the converged
    state; and node by node at an exact FD equilibrium with w re-seeded there. MEASURED FD: bulk 1.5e-16
    (exact state 9.7e-15), interface (4 bins of 0.3 eV) 1.4e-15 (2.8e-15). SHOWN FAILING on the legacy statistics: the exact-state
    worst node 1.10 (bulk) / 0.78 (interface), and the legacy interface device's hole branch 0.41 of
    its exchange; the legacy bulk device carries the quasi-static row's imbalance
    (bulk: 2.249e-4 A/m^2 of trap recombination at zero bias, the same as quasi-static, MEASURED)."""
    r = _rb(stats="fermi_dirac", **_KIN[key])
    assert r["residual"] <= 1e-12
    b = _budget(r)
    assert _ratio(r) <= b and r["eq_worst_local"] <= b, (_ratio(r), r["eq_worst_local"], b)
    legacy = _rb(stats="legacy", **_KIN[key])
    assert legacy["eq_worst_local"] >= 0.5


@pytest.mark.parametrize("key", sorted(_KIN))
def test_kfd_jacobian_matches_central_differences(key):
    """GATE K-FD-J. Off equilibrium (diode +0.3 V; MOS with the electrons x1.3), the full DEVSIM residual's
    central differences against the assembled matrix, columns Potential / Electrons / Holes AND every kinetic
    occupancy variable (TrapW_*, ItW*), criterion of gate J. MEASURED 7.0e-4 (bulk), 5.3e-4 (interface);
    knocked-out FD derivatives 5.8e5 / 3.1e5 (fails)."""
    o = {k: v for k, v in _KIN[key].items() if k != "h"}
    r = C.run_child("jacobian", stats="fermi_dirac", knockout=1, **o)
    assert any(v.startswith(("TrapW_", "ItW")) for v in r["variables"])
    assert r["worst"] <= 1.0, r["where"]
    assert r["ko_worst"] > 1.0


@pytest.mark.parametrize("key", sorted(_KIN))
def test_kfd_dc_equals_quasi_static_fd(key):
    """GATE K-FD-DC (the K1 identity under FD). Same device, kinetic vs quasi-static, statistics='fermi_dirac'
    (diode at +0.3 V; MOS at gate -0.3 V with the kinetic bins at the quasi-static bin width): potential to
    64 eps |psi|max-ish (1e-14 V), carriers to 1e-12 relative (the Newton's last update), and the trap /
    surface recombination current to 64 eps of the kinetic exchange rate (R_n equals U only to the trap row's
    round-off, P eps). MEASURED dpsi 1.1e-16 / 1.1e-16 V, dn 1.8e-15 / 1.1e-15, dp 3.3e-15 / 1.3e-15,
    |dI| 3.0e-9 A/m^2 of a 2.0e8 A/m^2 exchange (bulk) / 1.5e-8 of 1.2e7 (interface, at equilibrium)."""
    o = {k: v for k, v in _KIN[key].items() if k not in ("h", "kinetic", "kbin")}   # bins as the quasi-static
    r = C.run_child("dc_kinetic", stats="fermi_dirac", **o)
    assert r["residual"] <= 1e-12
    assert r["dpsi"] <= 1e-14 and r["dn"] <= 1e-12 and r["dp"] <= 1e-12, r
    assert abs(r["I_kin"] - r["I_qs"]) <= 64.0 * EPS * r["one_way_kin"], r


def test_kfd_trap_noise_fdt():
    """GATE K-FD-F. The FDT diode with the three levels KINETIC under statistics='fermi_dirac': S(1 Hz) =
    4 kT G_dc (Richardson DC conductance), the trap g-r branches (trap_n + trap_p) carrying >= 50 % of S
    (MEASURED 69 %); budget _dc_budget. MEASURED 7.4e-12. SHOWN FAILING on the legacy statistics: -5.3e-3.
    (v0.15 neutral-kinetic fix: the midgap kinetic level is now ACCEPTOR-like -- a kinetic level holds charge --
    re-MEASURED share 69 %, 6.8e-12, legacy -5.4e-3.)
    (FINDING outside this work, MEASURED in every statistics, Boltzmann included: with the NEUTRAL kinetic
    level present, S/(4 kT Re Y) at 1 MHz was off by +0.33 -- a kind='neutral' kinetic level stored N_t f in
    its trap row but not in Poisson, so the AC small-signal was not charge-conserving. FIXED: a kinetic level
    must be acceptor- or donor-like (physics_traps.NeutralKineticTrapError); the kinetic set here is the two
    charged levels -- tests/test_trap_kinetics.py::test_a_kinetic_level_holds_charge_and_conserves_it_in_ssac.)"""
    o = dict(traps=1, trap_N=1e22, kinetic=1, dc=1)
    r = C.run_child("fdt", stats="fermi_dirac", **o)
    assert r["share"].get("trap_n", 0.0) + r["share"].get("trap_p", 0.0) >= 0.5
    assert abs(r["fdt_dc"]) <= _dc_budget(r), (r["fdt_dc"], _dc_budget(r))
    legacy = C.run_child("fdt", stats="legacy", **o)
    assert abs(legacy["fdt_dc"]) > 1e-3


# =====================================================================================================
# B: the unipolar consistent transport threaded through TransportModel and both builders
# =====================================================================================================

def test_b_unipolar_statistics_threaded_through_both_builders():
    """GATE B. TransportModel(physics='drift_diffusion', carrier_statistics='fermi_dirac') gives every unipolar
    region of the 2-D LayeredDevsimBuilder and of the 3-D Devsim3DEquilibrium (the primary and an extra
    heterostack semiconductor, via Stacked3DSpec.from_design / the spec field) the FD quasi-field transport
    (g_enh == 1, the Delta edge models). REVISED 2026-11-07: that is the DEFAULT (None) since v0.15.0;
    'legacy' keeps the averaged g. Its physics: gate F-TE
    (unipolar). 'boltzmann' is refused for drift_diffusion (the unipolar model has no Boltzmann mode)."""
    from dynameta.carriers.physics_equilibrium import M_E
    from dynameta.materials.transport_model import TransportModel
    r = C.run_child("builders")
    for reg in (r["fd"], r["fd"]["3d_semi"], r["fd"]["3d_extra"]):
        assert reg["g_is_one"] and reg["quasi_field"]
    for reg in (r["legacy"], r["legacy"]["3d_semi"], r["legacy"]["3d_extra"]):
        assert not reg["g_is_one"] and not reg["quasi_field"]
    kw = dict(n_bg_m3=3e25, eps_static=9.0, dos_mass_kg_of_n_m3=lambda n: 0.3 * M_E,
              physics="drift_diffusion", mobility_m2Vs_of_n_m3=lambda n: 3e-3)
    assert TransportModel(carrier_statistics="fermi_dirac", **kw).carrier_statistics == "fermi_dirac"
    assert TransportModel(carrier_statistics="legacy", **kw).carrier_statistics == "legacy"
    with pytest.raises(ValueError, match="no Boltzmann mode"):
        TransportModel(carrier_statistics="boltzmann", **kw)


# =====================================================================================================
# DG: density gradient under the v0.15.0 default statistics ('fermi_dirac')
# =====================================================================================================

def test_dg_bipolar_inversion_equilibrium_is_exact_with_fd():
    """GATE DG-FD. The strongly inverted MOS of validation.bipolar_density_gradient GATE E (p 3e23, both
    carriers quantum at the insulator wall, quantum_srh) with the FD transport, under the DEFAULT statistics:
    the FD modified quasi-Fermi levels phi_n = psi + Lambda_n - V_t (L_n + ln(N_c/n_i)) and phi_p are flat
    (= the body contact's 0) and L_n + L_p = ln(n_i^2/(N_c N_v)) + (Lambda_n + Lambda_p)/V_t node by node --
    GATE E's identity with the transport's FD activity L in place of ln(c/N). Budget GATE E's own: 1e-11 V
    and 1e-10 (its 1e-6 relative solve floor; the quantum-SRH detailed balance is exact by construction).
    MEASURED (extended-precision Newton, as GATE E): max |phi_n| 7.8e-16 V, |phi_p| 1.1e-16 V, detailed balance
    2.1e-14 (delta_max 6.1e-3). SHOWN FAILING under statistics='legacy' (the pre-v0.15.0 default: the
    edge-averaged g and the Boltzmann-product SRH and contacts): |phi_n| 1.6e-4 V, |phi_p| 1.0e-7 V, detailed
    balance 6.1e-3 (= delta_max: the Boltzmann product)."""
    r = C.run_child("dg_fd")
    assert r["max_phi_n"] <= 1e-11 and r["max_phi_p"] <= 1e-11, r
    assert r["db"] <= 1e-10, r
    legacy = C.run_child("dg_fd", stats="legacy")
    assert max(legacy["max_phi_n"], legacy["max_phi_p"], legacy["db"]) > 1e-9, legacy


def test_dg_bipolar_jacobian_with_fd():
    """GATE DG-FD-J. The same device in DOUBLE precision (a perturbation does not reach the residual of a
    128-bit device: every central difference read exactly 0, MEASURED), every equation variable (Potential,
    Electrons, Holes, QSqrtN, QLambda, QSqrtP, QLambdaP) at the 4 nodes next to the wall, QSqrtN kicked x1.1
    off the solved state: DEVSIM's matrix vs central differences of its residual (gate J's criterion).
    MEASURED 5.3e-5. (A knock-out of the FD derivatives does not move it -- the inversion layer is barely
    degenerate, delta_max 6.1e-3, so those entries sit below the criterion; the knock-out is gated on the
    degenerate devices of gate J.)"""
    r = C.run_child("dg_fd", jac=1)
    assert any(v.startswith("Q") for v in r["variables"]) and "Electrons" in r["variables"], r["variables"]
    assert r["jac_worst"] <= 1.0, r["jac_where"]
