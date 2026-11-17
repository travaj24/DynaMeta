"""v0.16 gate leakage into a laterally contacted equilibrium film (carriers.gate_leakage; LayeredDevsimBuilder(
gate_leakage=...)). Notes and measurements: docs/audit/2026-11-14-v0.16-gate-leakage-lateral-drop.md.

EVERY DEVICE SCENARIO RUNS IN A CHILD PROCESS (tests/_gate_leakage_child.py): the leakage sink is a custom term whose
DEVSIM dispatcher can never be unregistered. The in-process tests below only exercise refusals that validate before
any DEVSIM call.

Device: the generic gated 6 nm degenerate film of tests/_wiring_fixture.py (n_bg 3e26 m^-3, between 6 nm oxB (eps 20)
+ 2 nm oxA (eps 9) on both faces, metal gates "bot" / "top"), an edge electrode on the film, mobility 3e-3 m^2/Vs;
solved with abs_tol 1e-14 V.

ORACLES (independent of the code under test):
  zero leakage      the equilibrium device without the option, on the identical node set;
  parabola          the distributed resistor V(x) - V_c = J (L x - x^2/2) / sigma_s (one contact) and
                    J x (L - x) / (2 sigma_s) (two contacts), sigma_s = q mu n_bg t, re-derived here;
  ohmic             two contacts at different voltages, no leakage: the linear profile and I = sigma_s dV / L;
  nonlinear         a 1-D boundary-value solve written here (scipy solve_bvp: phi' = -j / sigma(phi), j' = J(V_g - phi))
                    with the same J(V) table and sigma_s(phi) sampled from narrow-cell solves;
  kernel            the Tsu-Esaki/WKB integral re-evaluated here by nested scipy quadrature on the term's own inputs;
  conservation      sum of terminal currents = 0; sum I_k V_k = P_joule + P_tunnel;
  Newton            per-iteration update norms (quadratic tail).
OLD CODE (before this module): carriers.gate_leakage does not exist and LayeredDevsimBuilder has no
gate_leakage keyword, so every child raises (ImportError / TypeError) and every gate here FAILS; physically, the
equilibrium film without it has ONE Fermi level, i.e. a lateral drop of exactly 0 where the gates below measure 0.287 V
(nonlinear case) and 1.11e-6 V (parabola): a 100 % miss.
"""
import os
import sys

import numpy as np
import pytest

pytest.importorskip("devsim")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    import _gate_leakage_child as CH
finally:
    sys.path.pop(0)

Q = 1.602176634e-19
HBAR = 1.054571817e-34
M_E = 9.1093837015e-31
MU = CH.MU
N_BG, T_FILM = 3.0e26, 6e-9
SIGMA_BG = Q * MU * N_BG * T_FILM              # sheet conductance at flat band [S]


def _child(scenario, **opts):
    return CH.run_child(scenario, **opts)


# =====================================================================================================
# zero leakage reproduces the equilibrium solve
# =====================================================================================================
@pytest.fixture(scope="module", params=["plain", "composed"])
def zero(request):
    return request.param, _child("zero", kind=request.param)


def test_zero_leakage_reproduces_the_equilibrium_device(zero):
    """GateLeakage(scale=0) keeps the Fermi field and its equation but no sink: the Fermi field is EXACTLY the contact
    voltage at every node (bitwise 0.0), and every potential / density equals the equilibrium device without the option
    (same node set) to round-off: MEASURED <= 1.1e-14 relative (max |dpsi| 2.2e-16 V) at +2 V, plain and composed
    (Kane band alpha 0.4 /eV, fixed sheet -1e-3 C/m^2 + bulk 1e5 C/m^3, FilmInterfaceTraps D_it 1e16, top-gate work
    function). NOT bit-identical: the extra (decoupled) Fermi block changes UMFPACK's elimination order, so the Newton
    updates round differently; a bit-identical restore is the option off (gate_leakage=None makes no new DEVSIM call;
    tests/test_layered_edge_reference.py's restore digests). Budget 1e-13 relative (a few hundred ulp).
    OLD CODE: no gate_leakage keyword -> the child raises."""
    kind, r = zero
    A, Z = r["eq"], r["zero"]
    assert all(v == 0.0 for v in Z["film|EF_ref_V"])
    keys = [k for k in A if k.endswith("|Potential") or k.endswith("|Electrons")]
    assert len(keys) >= 6
    for k in keys:
        reg = k.split("|")[0]
        assert A[reg + "|x"] == Z[reg + "|x"] and A[reg + "|y"] == Z[reg + "|y"], k   # identical node sets
        a, z = np.array(A[k]), np.array(Z[k])
        assert np.max(np.abs(a - z)) <= 1e-13 * np.max(np.abs(a)), (kind, k)
    rz = r["report_zero"]
    assert rz["max_drop_V"] == 0.0 and all(g["I"] == 0.0 for g in rz["gates"].values())


def test_composed_device_with_leakage_balances(zero):
    """The same devices with the physical leakage on: terminal currents balance to round-off (MEASURED 8e-17 / 5e-16
    relative, plain / composed) and the terminal power equals the film Joule + tunnel dissipation (MEASURED
    identical to the last bit)."""
    _kind, r = zero
    rep = r["report_on"]
    assert abs(rep["balance_rel"]) < 1e-12
    assert rep["P_terminals"] == pytest.approx(rep["P_joule"] + rep["P_tunnel"], rel=1e-10, abs=1e-30)
    assert rep["gates"]["top"]["I"] > 0.0                 # +2 V on the top gate: electrons leave the film


# =====================================================================================================
# the distributed resistor (uniform J into a uniform sheet)
# =====================================================================================================
P_PARA, DX_PARA, J0 = 10e-6, 250e-9, 19.2


@pytest.fixture(scope="module")
def parabola():
    return _child("parabola", P=P_PARA, dx=DX_PARA, J0=J0)


SIGMA_3D = Q * MU * N_BG                       # film conductivity at flat band [S/m]
# The gate modulation of the sheet by the drop itself, C_top dV / (q n_bg t) = 6.5e-8 at the parabola's maximum,
# is the largest effect the analytic profiles below leave out (plus abs_tol / drop = 9e-9): the budget of every
# integral quantity of the uniform-J scenario is 2e-7 relative.
PARA_BUDGET = 2e-7


def _parabola_expectations(r, two):
    """The discrete quantities of the uniform-J solution (J0 into the top face, flat band, uniform n_bg):
    phi_F(x, z) = (column-mean sheet parabola) + J0 z^2 / (2 sigma t) + const (z from the bottom face) away from the
    contact -- the current enters through the top face and spreads over the thickness -- so
      vertical spread  = J0 t / (2 sigma)                   (ptp over a node column)
      max node value   = sheet maximum + J0 t / (3 sigma)    (the top face lies 2/3 of the spread above the mean)
      Joule power      = sum_edges sigma_s (d phi)^2 / dx + J0^2 t^2 L / (3 sigma_s)
    The lateral term is the finite-volume sum on the solve's own nodes (the scheme's nodal values are exact for the
    quadratic, so each edge's dphi / dx is phi' at the edge midpoint and the sum is the midpoint rule of
    Int sigma_s phi'^2 dx, -h^2 / (4 L^2) = -1.6e-4 below J0^2 L^3 / (3 sigma_s) on the uniform 250 nm mesh); the
    second is the vertical current's dissipation, t^2 / L^2 = 3.6e-7 of the total (one contact; 4 t^2 / L^2 with
    two)."""
    x = np.array(r["x_m"])
    if two:
        ana = J0 * x * (P_PARA - x) / (2 * SIGMA_BG)
        top = J0 * P_PARA ** 2 / (8 * SIGMA_BG)
    else:
        ana = J0 * (P_PARA * x - 0.5 * x ** 2) / SIGMA_BG
        top = J0 * P_PARA ** 2 / (2 * SIGMA_BG)
    p_lat = float(np.sum(SIGMA_BG * np.diff(ana) ** 2 / np.diff(x)))
    p_vert = J0 ** 2 * T_FILM ** 2 * P_PARA / (3 * SIGMA_BG)
    return dict(ana=ana, sheet_max=top, max_node=top + J0 * T_FILM / (3 * SIGMA_3D),
                spread=J0 * T_FILM / (2 * SIGMA_3D), p_joule=p_lat + p_vert)


def test_uniform_leakage_gives_the_parabolic_profile_one_contact(parabola):
    """Constant J0 = 19.2 A/m^2 from the top gate into the film (both gates at 0 V: flat band, n = n_bg), contact at
    x = 0, insulating far end at L = 10 um: column-mean phi_F(x) = J0 (L x - x^2/2) / sigma_s, max 1.110e-6 V,
    matched to 6.3e-8 of the maximum (MEASURED; budget PARA_BUDGET); the column-integrated sheet conductance equals
    q mu n_bg t to 8.4e-8 (MEASURED). The vertical structure (_parabola_expectations): spread 3.995e-13 V, MEASURED
    1.1e-7 from J0 t / (2 sigma) (budget 5e-7: the drop modulates n in the top screening layer, width L_s = 0.78 nm,
    by C_top dV / (q n_bg L_s) = 5e-7, which weights the column's conductivity unevenly); max node value MEASURED
    5.6e-8 from the sheet maximum plus the 2/3-spread face excess (without that excess the miss is 3.0e-7, above the
    budget); Joule power MEASURED 4.0e-8 from the discrete lateral sum plus the vertical term (the continuum
    J0^2 L^3 / (3 sigma_s) misses by 1.6e-4, the lateral sum alone by 4.0e-7). Terminal currents exact to round-off
    (MEASURED 1e-16 relative)."""
    r = parabola["one"]
    e = _parabola_expectations(r, two=False)
    phi, ana = np.array(r["phi_F_V"]), e["ana"]
    assert r["sheet"] == pytest.approx(SIGMA_BG, rel=PARA_BUDGET, abs=0)
    assert np.max(np.abs(phi - ana)) <= PARA_BUDGET * ana.max()
    assert r["max_drop_V"] == pytest.approx(e["max_node"], rel=PARA_BUDGET, abs=0)
    assert r["vertical_spread_V"] == pytest.approx(e["spread"], rel=5e-7, abs=0)
    assert r["P_joule"] == pytest.approx(e["p_joule"], rel=PARA_BUDGET, abs=0)
    assert r["gates"]["top"]["I"] == pytest.approx(J0 * P_PARA, rel=1e-12, abs=0)          # A/m (per unit depth)
    assert r["contacts"]["gnd"] == pytest.approx(-J0 * P_PARA, rel=1e-12, abs=0)
    assert abs(r["balance_rel"]) < 1e-12
    # Every terminal sits at 0 V, so P_terminals is exactly 0.0 and P_joule + P_tunnel cancel to round-off: the
    # balance is stated against the magnitude of the two terms, not against their (zero) sum.
    assert abs(r["P_terminals"] - (r["P_joule"] + r["P_tunnel"])) <= 1e-9 * (abs(r["P_joule"]) + abs(r["P_tunnel"]))


def test_uniform_leakage_gives_the_parabolic_profile_two_contacts(parabola):
    """Contacts at both ends (same voltage): column-mean phi_F = J0 x (L - x) / (2 sigma_s), max J0 L^2 / (8 sigma_s);
    MEASURED 4.1e-8 of the maximum; the two contacts carry half the gate current each (to 1.1e-16). The vertical
    terms are 4x larger relative to the quarter-size drop: Joule power MEASURED -2.4e-8 from the discrete lateral
    sum plus the vertical term (that term is 1.4e-6 of the total), max node value 1.4e-8 from the sheet maximum
    plus the face excess (the excess is 9.6e-7 of it)."""
    r = parabola["two"]
    e = _parabola_expectations(r, two=True)
    phi, ana = np.array(r["phi_F_V"]), e["ana"]
    assert np.max(np.abs(phi - ana)) <= PARA_BUDGET * ana.max()
    assert r["max_drop_V"] == pytest.approx(e["max_node"], rel=PARA_BUDGET, abs=0)
    assert r["P_joule"] == pytest.approx(e["p_joule"], rel=PARA_BUDGET, abs=0)
    assert r["contacts"]["gnd"] == pytest.approx(-0.5 * J0 * P_PARA, rel=1e-12, abs=0)
    assert r["contacts"]["gnd2"] == pytest.approx(-0.5 * J0 * P_PARA, rel=1e-12, abs=0)


def test_two_contacts_at_different_voltages_carry_the_ohmic_current(parabola):
    """No leakage, contacts at 0 and 1e-4 V (refused by the equilibrium edge reference: a current must flow): the linear
    profile to 1.3e-6 of dV and I = sigma_s dV / L to 5.0e-6 (MEASURED; budget 2e-5: the sheet modulation by the
    1e-4 V lateral potential, C dV / (q n_bg t) = 5.9e-6, with margin); I V = P_joule (dissipated in the sheet)."""
    r = parabola["linear"]
    dv = 1e-4
    x, phi = np.array(r["x_m"]), np.array(r["phi_F_V"])
    assert np.max(np.abs(phi - dv * x / P_PARA)) <= 2e-5 * dv
    assert r["contacts"]["gnd2"] == pytest.approx(SIGMA_BG * dv / P_PARA, rel=2e-5, abs=0)
    assert r["contacts"]["gnd"] == pytest.approx(-r["contacts"]["gnd2"], rel=1e-8, abs=0)
    assert r["P_joule"] == pytest.approx(r["contacts"]["gnd2"] * dv, rel=1e-8, abs=0)


# =====================================================================================================
# nonlinear leakage vs an independent 1-D solve; lateral mesh convergence
# =====================================================================================================
NL = dict(vt=2.0, J1=7.6e5, V1=0.3, P=2e-6, phis=[0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0],
          dxs=[1e-7, 5e-8, 2.5e-8])


@pytest.fixture(scope="module")
def nonlinear():
    return _child("nonlinear", **NL)


def _bvp_oracle(r):
    from scipy.integrate import solve_bvp
    from scipy.interpolate import PchipInterpolator
    V, J = np.array(r["table_V"]), np.array(r["table_J"])
    yj = PchipInterpolator(V, np.arcsinh(J / 1e-9))          # the LeakageTable's documented interpolant

    def Jf(v):
        return 1e-9 * np.sinh(yj(v))

    sig = PchipInterpolator(np.array(r["phis"]), np.array(r["sigma"]))
    L, s0 = NL["P"], 1e-3

    def f(u, y):                         # u = x / L; y = (phi, j L / (s0 * 1 V)): O(1) unknowns
        return np.vstack([-y[1] * s0 / sig(y[0]), L * L * Jf(NL["vt"] - y[0]) / s0])

    def bc(a, b):
        return np.array([a[0], b[1]])

    u = np.linspace(0.0, 1.0, 4001)
    s = solve_bvp(f, bc, u, np.zeros((2, u.size)), tol=1e-10, max_nodes=400000)
    assert s.status == 0, s.message

    class _Sol:                          # the solution in x [m]
        def sol(self, x):
            return s.sol(np.asarray(x) / L)
    return _Sol()


def test_nonlinear_leakage_matches_an_independent_lateral_solve(nonlinear):
    """J(V) = 7.6e5 sinh(V / 0.3 V) A/m^2 (a LeakageTable), top gate +2 V, L = 2 um: the self-limiting profile (the
    local gate voltage falls as phi_F rises). Oracle: solve_bvp of the sheet equation with sigma_s(phi) from nine
    narrow-cell solves (q mu Int n dz; 9.53e-4 S at phi = 0 down to 7.79e-4 S at 2 V). Finest mesh (25 nm): max
    |phi - oracle| = 2.9e-5 of the 0.2870 V drop; Richardson extrapolation of the three meshes: 5.6e-6 (MEASURED).
    Budget 1e-4 (finest) / 2e-5 (extrapolated): the PCHIP interpolation of sigma between the nine samples and the
    arithmetic edge mean of n. The drop is positive (phi_F rises toward the positive gate; sign-sensitive). The effective
    gate action is 0.9014 (MEASURED). OLD CODE: drop 0 (one Fermi level), a 100 % miss."""
    s = _bvp_oracle(nonlinear)
    fine = nonlinear["runs"][-1]
    x, phi = np.array(fine["x_m"]), np.array(fine["phi_F_V"])
    ana = s.sol(x)[0]
    assert np.all(phi >= 0.0) and ana.max() > 0.25
    assert np.max(np.abs(phi - ana)) <= 1e-4 * ana.max()
    d = [run["max_drop_V"] for run in nonlinear["runs"]]
    rich = d[2] + (d[2] - d[1]) / 3.0
    assert rich == pytest.approx(ana.max(), rel=2e-5, abs=0)
    from dynameta.core.numerics import trapz
    xo = np.linspace(0.0, NL["P"], 20001)
    action = 1.0 - trapz(s.sol(xo)[0], xo) / NL["P"] / NL["vt"]       # mean (V_g - phi_F) / V_g over the face
    assert fine["gates"]["top"]["gate_action"] == pytest.approx(action, abs=1e-4)
    for run in nonlinear["runs"]:
        assert abs(run["balance_rel"]) < 1e-12
        assert run["P_terminals"] == pytest.approx(run["P_joule"] + run["P_tunnel"], rel=1e-10, abs=0)
        assert run["vertical_spread_V"] < 1e-4 * run["max_drop_V"]


def test_lateral_mesh_convergence_is_second_order(nonlinear):
    """Lateral spacing 100 / 50 / 25 nm on the nonlinear case: the max drop converges at observed order 2.00
    (MEASURED 1.998; drops 0.286943, 0.287009, 0.287025 V)."""
    d = [run["max_drop_V"] for run in nonlinear["runs"]]
    order = np.log2((d[0] - d[1]) / (d[1] - d[2]))
    assert 1.8 < order < 2.2
    assert d[0] < d[1] < d[2]


# =====================================================================================================
# the tunnelling current itself
# =====================================================================================================
@pytest.fixture(scope="module")
def kernel():
    return _child("kernel", vt=3.0, vb=-2.0)


def _quad_te(s, i, emax_eV=8.0):
    """Two-sided Tsu-Esaki with WKB through the stack, by nested scipy quadrature on the term's inputs at node i."""
    from scipy.integrate import quad
    kT = s["kT_J"]
    xs, xg, d = s["x_s"][i], s["x_g"][i], s["d"][i]
    layers = s["layers"]

    def theta(E):
        tot, prev = 0.0, 0.0
        for j, (off, t, m) in enumerate(layers):
            ua, ub = off * Q + Q * prev, off * Q + Q * d[j]
            tot += 2.0 * t * quad(lambda u: np.sqrt(max(2.0 * m * M_E * (ua + (ub - ua) * u - E), 0.0)) / HBAR,
                                  0.0, 1.0, limit=200, epsabs=0.0, epsrel=1e-12)[0]
            prev = d[j]
        return tot

    def integrand(E):
        return np.exp(-theta(E)) * kT * (np.logaddexp(0.0, xs - E / kT) - np.logaddexp(0.0, xg - E / kT))

    # break points: the two Fermi levels and every layer's band edge at BOTH its faces (theta(E) has a sqrt kink
    # where E crosses either; the film-side edge of layer j > 0 is off_j + d_(j-1), not off_j)
    starts = [off * Q + Q * (d[j - 1] if j else 0.0) for j, (off, _, _) in enumerate(layers)]
    pts = sorted({p for p in [xs * kT, xg * kT] + [off * Q + Q * dd for (off, _, _), dd in zip(layers, d)]
                  + starts if 0.0 < p < emax_eV * Q})
    pref = Q * s["m_supply"] * M_E / (2.0 * np.pi ** 2 * HBAR ** 3)
    return pref * quad(integrand, 0.0, emax_eV * Q, points=pts, limit=2000, epsabs=0.0, epsrel=1e-9)[0]


@pytest.mark.parametrize("gate", ["top", "bot"])
def test_term_current_matches_an_independent_quadrature(kernel, gate):
    """At the solved device (top +3 V: film -> gate emission, 0.4620 A/m^2; bottom -2 V: gate -> film, -9.07e-6 A/m^2)
    the term's J (reliability.leakage.tsu_esaki_two_sided on the fixed grid) equals the nested-quadrature integral of the
    same stack and inputs to 1.6e-9 (top, a turning-point case) and 5.2e-11 (bottom) (MEASURED). Budget 1e-7 (the fixed grid's documented C^1-kink error, <= 3.8e-9 in
    tests/test_tunneling_kernels.py K6, with margin for the quadrature). The oracle breaks its energy integral at
    every band-edge kink: without the film-side edge of the second layer (E = off_2 + d_1) the top-gate quadrature
    hits scipy's round-off warning on this stack and is itself off by 6e-8. Laterally uniform: the per-node spread
    is <= 7e-12."""
    s = kernel[gate]
    i = len(s["J"]) // 2
    J = _quad_te(s, i)
    assert s["J"][i] == pytest.approx(J, rel=1e-7, abs=0)
    assert np.ptp(s["J"]) <= 1e-8 * abs(s["J"][i])
    assert np.sign(s["J"][i]) == (1.0 if gate == "top" else -1.0)


def test_newton_tail_is_quadratic(kernel):
    """One Newton from a 2 % perturbed top bias at (+3, -2) V with the physical leakage: update norms fall
    quadratically (MEASURED 6.0e-2 -> 5.5e-6 -> 5.0e-12 -> 2.3e-16); e_(k+1) <= 10 e_k^2 on every step above round-off."""
    h = kernel["hist"]
    assert h[-1] < 1e-13 and len(h) <= 6
    for a, b in zip(h[:-1], h[1:]):
        if a > 1e-7:
            assert b <= 10.0 * a * a, h


@pytest.mark.parametrize("wf_top", [None, 5.1])
def test_field_table_is_read_at_the_solved_dielectric_drop(wf_top):
    """LeakageTable.from_field converts a J(E) quoted against the dielectric field
    E = V / t into V = E t; reading that table at V = V_gF - phi_F, the gate-to-film Fermi-potential difference, is
    wrong: that difference also contains the flat-band offset and the film's band bending. MEASURED on the gated 6 nm
    film at +2 V (t = 8 nm): solved drop across the top column 1.726 V (no gate work function) / 0.736 V (5.1 eV),
    against V_gF - phi_F = 2.000 V; the Fermi-difference reading is at E = 2.50e8 V/m instead of 2.16e8 / 9.2e7
    V/m, J +23 % / 3.4x too high. With variable="dielectric" each node reads the table at its column's solved drop,
    derived here from the raw potentials (gate face of up_oxa minus the film's top face, interpolated at the node's x), with exact
    derivatives (quadratic Newton tail) and balanced terminal currents. OLD CODE: the child fails (no V_table)."""
    from dynameta.carriers.gate_leakage import LeakageTable
    r = _child("field_table", vt=2.0, J1=1e2, E1=2e8, t=8e-9, wf_top=wf_top)
    a = {k: np.asarray(v) for k, v in r["arrays"].items()}
    yf, yo = a["film|y"], a["up_oxa|y"]
    top = np.isclose(yf, yf.max(), rtol=0, atol=1e-15)
    gate = np.isclose(yo, yo.max(), rtol=0, atol=1e-15)
    xf, of = a["film|x"][top], np.argsort(a["film|x"][top])
    xg, og = a["up_oxa|x"][gate], np.argsort(a["up_oxa|x"][gate])
    x = np.asarray(r["x_iface"])
    psi_s = np.interp(x, xf[of], a["film|Potential"][top][of])
    v_gate = np.interp(x, xg[og], a["up_oxa|Potential"][gate][og])
    drop = v_gate - psi_s
    assert np.max(np.abs(np.asarray(r["V_table"]) - drop)) <= 1e-12
    tab = LeakageTable(tuple(r["table_V"]), tuple(r["table_J"]))
    J_ref = tab(drop)[0]
    assert np.allclose(r["J"], J_ref, rtol=1e-12, atol=0.0)
    # the Fermi-difference reading exceeds the solved drop by 0.274 / 1.264 V (MEASURED); bars at 0.8 of that
    assert np.min(np.asarray(r["V_local"]) - drop) > (0.22 if wf_top is None else 0.99)   # the Fermi difference is far off
    h = r["hist"]
    assert h[-1] < 1e-13
    for p, q in zip(h[:-1], h[1:]):
        if p > 1e-7:
            assert q <= 10.0 * p * p, h
    assert abs(r["report"]["balance_rel"]) < 1e-12


@pytest.mark.slow
def test_high_bias_converges_with_continuation():
    """+3 V top, +5 V bottom with the child's default barriers (2.5 / 3.4 eV above the film conduction edge, i.e. 1.96 /
    2.86 eV above its Fermi level, masses 0.15 / 0.30): the ramp (adaptive continuation, 0.25 V steps halved on
    failure) converges, the tail is quadratic (MEASURED 6.0e-2 -> 5.5e-6 -> 5.0e-12 -> 7.2e-16) and the currents
    balance. ~50 s."""
    r = _child("newton", vt=3.0, vb=5.0)
    h = r["hist"]
    assert h[-1] < 1e-13
    for a, b in zip(h[:-1], h[1:]):
        if a > 1e-7:
            assert b <= 10.0 * a * a, h
    assert abs(r["report"]["balance_rel"]) < 1e-12


# =====================================================================================================
# refusals
# =====================================================================================================
def test_one_dimensional_device_is_refused():
    r = _child("refuse_1d")
    assert r["error"] is not None and "only 2-D" in r["error"]


def _leak_table():
    from dynameta.carriers.gate_leakage import GateLeakage
    return GateLeakage(model="table", mobility_m2Vs=MU,
                       table=lambda V: (np.zeros(np.shape(V)), np.zeros(np.shape(V))))


def test_builder_refusals_validate_before_devsim():
    import _wiring_fixture as W
    from dynameta.carriers.density_gradient import EquilibriumDensityGradient
    from dynameta.carriers.devsim_layered import LayeredDevsimBuilder
    with pytest.raises(NotImplementedError, match="density_gradient and gate_leakage"):
        LayeredDevsimBuilder(W.film_design(edges=[W.edge()]), gate_leakage=_leak_table(),
                             density_gradient={"film": EquilibriumDensityGradient()})
    with pytest.raises(ValueError, match="no equilibrium film with an x-edge electrode"):
        LayeredDevsimBuilder(W.film_design(), gate_leakage=_leak_table())
    with pytest.raises(NotImplementedError, match="pin_potential"):
        LayeredDevsimBuilder(W.film_design(edges=[W.edge(), W.edge("g2", side="x_hi", pin=True)]),
                             gate_leakage=_leak_table())
    with pytest.raises(NotImplementedError, match="y_lo"):
        LayeredDevsimBuilder(W.film_design(edges=[W.edge(), W.edge("g2", side="y_lo")]), gate_leakage=_leak_table())
    with pytest.raises(TypeError, match="GateLeakage"):
        LayeredDevsimBuilder(W.film_design(edges=[W.edge()]), gate_leakage={"model": "table"})


def test_missing_film_mobility_is_refused_before_devsim():
    """GateLeakage(mobility_m2Vs=None) (documented: the TransportModel's mobility at n_bg) on an
    equilibrium film whose TransportModel has no mobility (equilibrium physics does not need one). OLD CODE: the
    builder accepted it and the first solve failed inside build_device with "TypeError: 'NoneType' object is not
    callable"; now the constructor raises a ValueError naming both mobility inputs. A material WITH a mobility
    still builds with mobility_m2Vs=None."""
    import dataclasses
    import _wiring_fixture as W
    from dynameta.carriers.devsim_layered import LayeredDevsimBuilder
    from dynameta.carriers.gate_leakage import GateLeakage
    nomob = GateLeakage(model="table", table=lambda V: (np.zeros(np.shape(V)), np.zeros(np.shape(V))))
    d = W.film_design(edges=[W.edge()])
    assert d.materials.get("TCO").transport.mobility_m2Vs_of_n_m3 is None
    with pytest.raises(ValueError, match=r"GateLeakage\.mobility_m2Vs is None.*mobility_m2Vs_of_n_m3"):
        LayeredDevsimBuilder(d, gate_leakage=nomob)
    from dynameta.materials import MaterialRegistry
    reg = MaterialRegistry()
    for name in d.materials.names():
        m = d.materials.get(name)
        if name == "TCO":
            m = dataclasses.replace(m, transport=dataclasses.replace(m.transport, mobility_m2Vs_of_n_m3=lambda n: MU))
        reg.add(m)
    b = LayeredDevsimBuilder(W.film_design(edges=[W.edge()], mats=reg), gate_leakage=nomob)   # builds, no DEVSIM call
    assert b._leak_regions == ["film"]
