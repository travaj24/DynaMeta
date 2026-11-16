"""v0.16 gate leakage (carriers.gate_leakage): the parts that run without a live DEVSIM, beyond
tests/test_gate_leakage.py -- the measured-table extrapolation and option refusals, the reduced 1-D lateral
solver's grid / refusal / continuation paths, and the device term's numpy kernels (GateLeakageTerm.state and
.assemble, setup_fermi_field_region's model strings, setup_fermi_contact, setup_gate_leakage's wiring) driven
on hand-made node data through the recording stand-in of tests/_fake_devsim.py.

ORACLES: closed forms re-derived here (y = asinh(J/J0) continued linearly with the end-segment slope; the
uniform-sheet quadratic V = J (L x - x^2/2)/sigma, which the finite-volume scheme reproduces exactly on any
grid); central finite differences of the value function for every hand-written derivative (the term's
dJ/dpsi_s, dJ/dphi_F, dJ/d(face potential), the assembled Jacobian against the assembled residual, the edge
flux derivative models against the emitted flux string); detailed balance (no current at equal Fermi levels);
refusals that name their cause and leave the device untouched. No solve anywhere.
"""
import numpy as np
import pytest

import _fake_devsim as FD
from dynameta.carriers import gate_leakage as GL
from dynameta.carriers import eq_registry as R
from dynameta.carriers import physics_equilibrium as PE
from dynameta.carriers.tunneling import DielectricBarrier
from dynameta.reliability import leakage as LK
from dynameta.warning_types import AdvisoryWarning

Q = 1.602176634e-19
VT = 0.025851999786435535          # kT/q at 300 K
EF = PE.FERMI_REF_PARAM
DEV = "fake_gl_dev"


@pytest.fixture(scope="module")
def eqdg_module():
    """physics_equilibrium_dg (setup_fermi_field_region imports it to refuse a density-gradient film); on a
    leg without devsim it is imported against the stand-in and dropped again afterwards."""
    with FD.modules_imported_with("dynameta.carriers.physics_equilibrium_dg") as (m,):
        yield m


# =====================================================================================================
# LeakageTable / GateLeakage inputs
# =====================================================================================================
def test_table_extrapolates_below_the_table_with_the_first_segment_slope():
    """Below V_0 the table continues y = asinh(J/J0) with the FIRST segment's slope (not PCHIP's end derivative,
    which its limiter may set to 0): J(V) = J0 sinh(y_0 + s (V - V_0)), s = (y_1 - y_0)/(V_1 - V_0). Oracle: that
    closed form re-derived here (1e-12 relative, round-off of sinh/asinh) and the derivative J0 cosh(y) s; the
    flag in_range is False there."""
    V = (-1.0, -0.5, 0.0, 0.5, 1.0)
    J = (-1e-2, -1e-5, 0.0, 1e-5, 1e-2)
    t = GL.LeakageTable(V, J, J0_A_m2=1e-9)
    y = np.arcsinh(np.asarray(J) / 1e-9)
    s = (y[1] - y[0]) / 0.5
    v = np.array([-1.7, -1.2, -1.0 - 1e-9])
    Jt, dJt = t(v)
    ya = y[0] + s * (v + 1.0)
    assert np.allclose(Jt, 1e-9 * np.sinh(ya), rtol=1e-12, atol=0.0)
    assert np.allclose(dJt, 1e-9 * np.cosh(ya) * s, rtol=1e-12, atol=0.0)
    assert Jt[-1] == pytest.approx(J[0], rel=1e-7, abs=0)             # continuous at the table end
    assert not t.in_range(v) and t.in_range([-1.0, 1.0])


def test_table_through_a_sign_change_is_monotone_and_odd():
    """A monotone J(V) that changes sign at V = 0 over 12 decades: the PCHIP of asinh(J/J0) is monotone (a
    shape-preserving interpolant of monotone data, and sinh is increasing), so dJ/dV >= 0 everywhere and J has
    the sign of V; an odd table stays odd (asinh, PCHIP and sinh are all odd-symmetric) to round-off."""
    V = np.linspace(-2.0, 2.0, 17)
    J = 1e-6 * np.sinh(V / 0.08)                     # +-7e4 A/m^2 at the ends, 0 at V = 0
    t = GL.LeakageTable(tuple(V), tuple(J), J0_A_m2=1e-12)
    v = np.linspace(-1.999, 1.999, 2001)
    Jt, dJt = t(v)
    assert np.all(dJt >= 0.0)
    assert np.all(np.sign(Jt[v != 0.0]) == np.sign(v[v != 0.0]))
    # round-off of the polynomial evaluation: MEASURED 6e-15 of max |J| (relative error only at |V| ~ 1e-16)
    assert np.allclose(t(-v)[0], -Jt, rtol=1e-12, atol=1e-12 * np.max(np.abs(J)))


@pytest.mark.parametrize("kw, match", [
    (dict(J0_A_m2=0.0), "J0_A_m2 must be > 0"),
    (dict(J0_A_m2=float("inf")), "J0_A_m2 must be > 0"),
])
def test_table_scale_refusals(kw, match):
    with pytest.raises(ValueError, match=match):
        GL.LeakageTable((0.0, 1.0), (0.0, 1.0), **kw)


def test_from_field_refuses_a_non_positive_thickness():
    with pytest.raises(ValueError, match="thickness_m must be > 0"):
        GL.LeakageTable.from_field((0.0, 1e8), (0.0, 1.0), 0.0)


def test_gate_leakage_option_refusals_name_the_field():
    b = DielectricBarrier(2.5, "FILM", 0.3)
    ok = dict(model="tsu_esaki", barriers=b, gate_fermi_energy_eV=8.0)
    with pytest.raises(ValueError, match="mobility_m2Vs must be > 0"):
        GL.GateLeakage(model="table", table=lambda v: (v, v), mobility_m2Vs=0.0)
    with pytest.raises(ValueError, match="energy_max_eV and energy_step_kT"):
        GL.GateLeakage(energy_max_eV=0.0, **ok)
    with pytest.raises(ValueError, match="energy_max_eV and energy_step_kT"):
        GL.GateLeakage(energy_step_kT=-0.1, **ok)
    with pytest.raises(ValueError, match="m_supply_over_m0 must be > 0"):
        GL.GateLeakage(m_supply_over_m0=0.0, **ok)
    with pytest.raises(ValueError, match="table is for model='table'"):
        GL.GateLeakage(table=lambda v: (v, v), **ok)
    with pytest.raises(ValueError, match="needs barriers"):          # a mapping with a non-barrier value
        GL.GateLeakage(model="tsu_esaki", barriers={"ox": 2.5}, gate_fermi_energy_eV=8.0)
    with pytest.raises(ValueError, match="needs table"):              # a mapping with a non-callable value
        GL.GateLeakage(model="table", table={"top": (0.0, 1.0)})
    with pytest.raises(ValueError, match="scale must be finite"):
        GL.GateLeakage(model="table", table=lambda v: (v, v), scale=float("nan"))


def test_gate_leakage_lookup_helpers():
    """One barrier / one table serves every region / gate; a mapping answers per key and refuses an unknown key
    naming the keys it has; strings for gates / regions become 1-tuples."""
    b = DielectricBarrier(2.5, "FILM", 0.3)
    g = GL.GateLeakage(model="tsu_esaki", barriers=b, gate_fermi_energy_eV=8.0, regions="film")
    assert g.barrier_for("anything") is b and g.regions == ("film",)

    def f(v):
        return v, np.ones_like(v)

    def h(v):
        return 2 * v, 2 * np.ones_like(v)

    assert GL.GateLeakage(model="table", table=f).table_for("any") is f
    gm = GL.GateLeakage(model="table", table={"top": f, "bot": h}, gates=["top", "bot"])
    assert gm.table_for("bot") is h and gm.gates == ("top", "bot")
    with pytest.raises(ValueError, match=r"no entry for gate 'side' \(have \['bot', 'top'\]\)"):
        gm.table_for("side")


# =====================================================================================================
# the reduced 1-D lateral model
# =====================================================================================================
def _const(v):
    return lambda p: (np.full(np.shape(p), v), np.zeros(np.shape(p)))


def test_lateral_1d_on_a_user_grid_reproduces_the_distributed_resistor():
    """A user grid (non-uniform, random cell sizes): the finite-volume scheme with box widths w_i and edge
    conductance is exact for the quadratic of uniform J into a uniform sheet (its truncation error is the
    third derivative, zero), so the result equals V = V_c + J (L x - x^2/2)/sigma to round-off (budget 1e-12 of
    the drop)."""
    rng = np.random.default_rng(3)
    L, sig, J = 2e-4, 5e-4, 80.0
    x = np.concatenate([[0.0], np.cumsum(rng.uniform(0.2, 1.0, 60))])
    x *= L / x[-1]
    r = GL.solve_lateral_1d(L, _const(sig), _const(J), v_contact=-0.1, x_m=x)
    ana = -0.1 + J * (L * x - 0.5 * x ** 2) / sig
    assert np.array_equal(r["x_m"], x)
    assert np.max(np.abs(r["phi_V"] - ana)) <= 1e-12 * (J * L ** 2 / (2 * sig))
    assert r["residual"] <= 1e-12 * J * L


@pytest.mark.parametrize("kw, match", [
    (dict(length_m=0.0), "length_m must be > 0"),
    (dict(contacts="right"), "contacts must be 'left' or 'both'"),
    (dict(x_m=np.array([1e-9, 5e-5, 1e-4])), "x_m must increase from 0 to length_m"),
    (dict(x_m=np.array([0.0, 6e-5, 5e-5, 1e-4])), "x_m must increase from 0 to length_m"),
    (dict(x_m=np.array([0.0, 5e-5, 2e-4])), "x_m must increase from 0 to length_m"),
])
def test_lateral_1d_refusals(kw, match):
    args = dict(length_m=1e-4)
    args.update(kw)
    with pytest.raises(ValueError, match=match):
        GL.solve_lateral_1d(args.pop("length_m"), _const(1e-3), _const(1.0), **args)


def _nonlinear():
    def sig(p):
        return 1e-3 * (1.0 - 0.1 * p), -1e-4 * np.ones_like(p)

    def J(p):
        u = (2.0 - p) / 0.3
        return 2e3 * np.sinh(u), -2e3 * np.cosh(u) / 0.3
    return sig, J


def test_lateral_1d_continuation_reaches_the_same_solution():
    """The homotopy in J: with max_iter = 5 the full step lambda 0 -> 1 cannot converge (the direct solve needs 9
    Newton steps), so lambda is halved and walked back up; the result must be the SAME solution as the direct
    solve (MEASURED 2e-16 V; budget 1e-12 V against the 1e-13 V Newton tolerance) after more total iterations
    (MEASURED 50 vs 9). With max_iter = 2 no step size converges and the solver raises, naming the cause."""
    sig, J = _nonlinear()
    ref = GL.solve_lateral_1d(300e-6, sig, J, n_nodes=201)
    r = GL.solve_lateral_1d(300e-6, sig, J, n_nodes=201, max_iter=5)
    assert r["iterations"] > ref["iterations"] > 1
    assert np.max(np.abs(r["phi_V"] - ref["phi_V"])) <= 1e-12
    with pytest.raises(RuntimeError, match="Newton did not converge"):
        GL.solve_lateral_1d(300e-6, sig, J, n_nodes=41, max_iter=2)


def test_lateral_1d_refuses_a_non_finite_sheet():
    """A NaN sheet conductance is refused, never returned as a NaN profile (the module's own 'non-finite Newton
    update' FloatingPointError; before the input checks, scipy's banded-solve finiteness ValueError escaped
    first -- either refusal is accepted here, the specific message is pinned below)."""
    with np.errstate(invalid="ignore"):
        with pytest.raises((ValueError, FloatingPointError), match="(?i)nan|non-finite"):
            GL.solve_lateral_1d(1e-4, _const(float("nan")), _const(1.0), n_nodes=11)


# =====================================================================================================
# the device term on hand-made node data
# =====================================================================================================
# 4 interface nodes of the film (film node indices 1..4 of 6); two dielectric layers whose far faces carry 5
# nodes each; interface node i reads far nodes (i, i+1) with weights (0.7, 0.3).
IFACE = np.array([1, 2, 3, 4])
PHI_C0 = -5.0 * VT                   # eta = 5 at psi = phi_F = 0: a degenerate film
GATE_OFFSET = 0.2


def _faces():
    W = [(np.array([i, i + 1]), np.array([0.7, 0.3])) for i in range(4)]
    return [("ox1", np.arange(5), W), ("ox2", np.arange(5), W)]


def _fake(V_gate=1.2):
    fk = FD.FakeDevsim(2)
    fk.add_region("film", 6)
    fk.add_region("ox1", 5)
    fk.add_region("ox2", 5)
    fk.params[("film", "Phi_c0")] = PHI_C0
    fk.params[("film", "V_t")] = VT
    fk.params[(None, "top_bias")] = V_gate
    fk.nodes[("film", "Potential")] = np.array([0.0, 0.02, 0.0, -0.01, 0.03, 0.0])
    fk.nodes[("film", EF)] = np.array([0.0, 0.0, 0.01, 0.02, 0.03, 0.0])
    fk.nodes[("ox1", "Potential")] = np.array([0.35, 0.40, 0.42, 0.38, 0.36])
    fk.nodes[("ox2", "Potential")] = np.array([1.00, 1.05, 1.02, 0.98, 1.01])
    return fk


def _term(model, *, scale=1.0, table=None, energy_max_eV=4.0, gate_fermi_eV=8.0, contact=None):
    is_contact = np.zeros(4, dtype=bool) if contact is None else np.asarray(contact, dtype=bool)
    kw = dict(semi="film", gate="top", gate_param="top_bias", gate_offset=GATE_OFFSET, model=model,
              scale=scale, leakage=None, chain=("ox1", "ox2"), iface=IFACE, is_contact=is_contact,
              area=np.array([1e-9, 2e-9, 2e-9, 1e-9]), faces=_faces(), axis=1, interface="film_ox1", lateral=[0])
    if model == "tsu_esaki":
        kw.update(layers=(LK.TunnelLayer(2.0, 1e-9, 0.3), LK.TunnelLayer(1.5, 2e-9, 0.25)),
                  grid=LK.tunnel_energy_grid(Q * VT, energy_max_eV * Q, step_kT=0.25), m_supply=0.3,
                  gate_fermi_J=gate_fermi_eV * Q)
    else:
        kw.update(layers=(), table=table)
    return GL.GateLeakageTerm("gleak_film_top", **kw)


def _J(term):
    return term.state(DEV)["J"].copy()


def _fd(term, fk, key, idx, h):
    base = fk.nodes[key].copy()
    fk.nodes[key] = base.copy()
    fk.nodes[key][idx] += h
    jp = _J(term)
    fk.nodes[key][idx] -= 2 * h
    jm = _J(term)
    fk.nodes[key] = base
    return (jp - jm) / (2 * h)


def _sinh_table():
    V = np.linspace(-3.0, 3.0, 31)
    return GL.LeakageTable(tuple(V), tuple(5.0 * np.sinh(V / 0.25)), J0_A_m2=1e-6)


def test_term_tsu_esaki_derivatives_match_finite_differences(monkeypatch):
    """GateLeakageTerm.state, tsu_esaki: the chain rule from the kernel's exact derivatives to the solution
    variables -- dJ/dpsi_s = (dJ/dx_s + dJ/dx_g)/V_t + sum_k dJ/dd_k (faces held), dJ/dphi_F = -dJ/dx_s / V_t,
    and dJ/d(shift of every node of face k) = -dJ/dd_k (the lateral weights sum to 1). Oracle: central
    differences of J itself, h = 1e-6 V (truncation ~(h/V_t)^2 ~ 1e-9 relative; the kernel's derivatives are
    those of its discrete energy sum, so only that and round-off remain); budget 1e-6 relative to max |dJ|."""
    fk = FD.install(monkeypatch, _fake())
    t = _term("tsu_esaki", scale=1.7)
    st = t.state(DEV)
    assert np.all(st["J"] > 0.0)                     # gate Fermi potential above the film's: film -> gate
    h = 1e-6
    for got, key in ((st["dJ_dpsi"], ("film", "Potential")), (st["dJ_dphi"], ("film", EF))):
        fd = _fd(t, fk, key, IFACE, h)
        assert np.max(np.abs(got - fd)) <= 1e-6 * np.max(np.abs(fd))
    for k, reg in enumerate(("ox1", "ox2")):
        fd = _fd(t, fk, (reg, "Potential"), slice(None), h)
        assert np.max(np.abs(-st["dJ_dd"][:, k] - fd)) <= 1e-6 * np.max(np.abs(fd))
    # x_s, x_g and the column drops as the module header defines them
    psi = fk.nodes[("film", "Potential")][IFACE]
    phi = fk.nodes[("film", EF)][IFACE]
    assert np.allclose(st["x_s"], (psi - phi - PHI_C0) / VT, rtol=1e-14, atol=0.0)
    assert np.allclose(st["x_g"], (psi - PHI_C0 - (1.2 - GATE_OFFSET)) / VT, rtol=1e-14, atol=0.0)
    pf = fk.nodes[("ox2", "Potential")]
    assert np.allclose(st["d"][:, 1], psi - (0.7 * pf[:4] + 0.3 * pf[1:]), rtol=1e-14, atol=0.0)
    assert np.allclose(st["J"], st["J_fwd"] - st["J_bwd"], rtol=1e-12, atol=0.0)
    assert st["V_gF"] == pytest.approx(1.2 - GATE_OFFSET, rel=1e-15, abs=0)


def test_term_tsu_esaki_detailed_balance_and_scale(monkeypatch):
    """At equal Fermi levels (gate Fermi potential = phi_F at a node, any band bending) the two one-way currents
    cancel: x_s = x_g there, so J = 0 to round-off of J_fwd (budget 1e-12 J_fwd). J reverses sign with the
    Fermi-level difference and is linear in scale."""
    fk = FD.install(monkeypatch, _fake())
    fk.nodes[("film", EF)][IFACE] = 0.37
    fk.params[(None, "top_bias")] = 0.37 + GATE_OFFSET
    st = _term("tsu_esaki").state(DEV)
    assert np.all(st["J_fwd"] > 0.0)
    assert np.max(np.abs(st["J"])) <= 1e-12 * np.max(st["J_fwd"])
    fk.params[(None, "top_bias")] = 0.30 + GATE_OFFSET              # gate below the film: gate -> film
    j1 = _term("tsu_esaki").state(DEV)["J"]
    assert np.all(j1 < 0.0)
    assert np.allclose(_term("tsu_esaki", scale=0.25).state(DEV)["J"], 0.25 * j1, rtol=1e-14, atol=0.0)


def test_term_tsu_esaki_refusals_and_tail_warning(monkeypatch):
    """A gate whose band bottom rises above the film conduction edge is refused (electrons below it are outside
    the fixed grid), naming the cause; a grid too short for the supply warns ONCE (AdvisoryWarning naming
    energy_max_eV)."""
    fk = FD.install(monkeypatch, _fake(V_gate=-0.8))        # gate Fermi potential -1 V: its band bottom
    with pytest.raises(RuntimeError, match="band bottom rises above the film conduction edge"):
        _term("tsu_esaki", gate_fermi_eV=0.5).state(DEV)     # (0.5 eV below its E_F) sits above E_c,film
    fk.params[(None, "top_bias")] = 1.2
    t = _term("tsu_esaki", energy_max_eV=0.15)
    with pytest.warns(AdvisoryWarning, match="raise energy_max_eV"):
        st = t.state(DEV)
    assert st["tail"] > 1e-8
    t.state(DEV)                                     # warned once: a second call is silent (error filter)


def test_term_fermi_table_reads_the_fermi_difference(monkeypatch):
    """model='table', variable 'fermi': J = table(V_gF - phi_F) at each node (the LeakageTable evaluated
    directly is the oracle), dJ/dphi_F = -table'(V) (central differences, h = 1e-6 V on a 0.25 V scale), no
    psi_s or face dependence, and the in-range flag of the table."""
    fk = FD.install(monkeypatch, _fake())
    tab = _sinh_table()
    t = _term("table", table=tab, scale=2.0)
    st = t.state(DEV)
    V = (1.2 - GATE_OFFSET) - fk.nodes[("film", EF)][IFACE]
    assert np.allclose(st["J"], 2.0 * tab(V)[0], rtol=1e-14, atol=0.0)
    fd = _fd(t, fk, ("film", EF), IFACE, 1e-6)
    assert np.max(np.abs(st["dJ_dphi"] - fd)) <= 1e-6 * np.max(np.abs(fd))
    assert np.all(st["dJ_dpsi"] == 0.0) and st["dJ_dd"].shape == (4, 0)
    assert st["table_in_range"] is True
    fk.params[(None, "top_bias")] = 5.0
    assert t.state(DEV)["table_in_range"] is False


def test_term_dielectric_table_reads_the_solved_column_drop(monkeypatch):
    """model='table' with a dielectric-drop table (LeakageTable.from_field): V = (gate-face potential at the
    node's lateral position) - psi_s, independent of phi_F and of the inner face; dJ/dpsi_s = -table'(V) and
    dJ/d(gate-face shift) = +table'(V). Oracle: V re-derived from the node data, J from the table directly,
    the derivatives by central differences (h = 1e-6 V)."""
    fk = FD.install(monkeypatch, _fake())
    tab = GL.LeakageTable.from_field(tuple(np.linspace(-3e8, 3e8, 31)),
                                     tuple(5.0 * np.sinh(np.linspace(-3.0, 3.0, 31) / 0.25)), 1e-8)
    t = _term("table", table=tab)
    st = t.state(DEV)
    pf = fk.nodes[("ox2", "Potential")]
    V = 0.7 * pf[:4] + 0.3 * pf[1:] - fk.nodes[("film", "Potential")][IFACE]
    assert np.allclose(st["V_table"], V, rtol=1e-14, atol=0.0)
    assert np.allclose(st["J"], tab(V)[0], rtol=1e-14, atol=0.0)
    fd_psi = _fd(t, fk, ("film", "Potential"), IFACE, 1e-6)
    assert np.max(np.abs(st["dJ_dpsi"] - fd_psi)) <= 1e-6 * np.max(np.abs(fd_psi))
    fd_gate = _fd(t, fk, ("ox2", "Potential"), slice(None), 1e-6)
    assert np.max(np.abs(-st["dJ_dd"][:, 1] - fd_gate)) <= 1e-6 * np.max(np.abs(fd_gate))
    assert np.all(st["dJ_dd"][:, 0] == 0.0) and np.all(st["dJ_dphi"] == 0.0)
    assert np.max(np.abs(_fd(t, fk, ("film", EF), IFACE, 1e-6))) == 0.0
    assert np.max(np.abs(_fd(t, fk, ("ox1", "Potential"), slice(None), 1e-6))) == 0.0


def _numbered(fk):
    """Global equation / column numbers: film rows 100.., film Potential 0.., EF 10.., ox1 20.., ox2 30..; and
    the inverse map column -> (region, variable, node)."""
    perm = np.array([3, 5, 0, 1, 4, 2])
    fk.numbers[("film", "equation", GL.FERMI_FIELD_EQUATION)] = 100 + perm
    cols = {}
    for reg, var, base, n in (("film", "Potential", 0, 6), ("film", EF, 10, 6), ("ox1", "Potential", 20, 5),
                              ("ox2", "Potential", 30, 5)):
        idx = base + np.arange(n)[::-1]
        fk.numbers[(reg, "variable", var)] = idx
        cols.update({int(c): (reg, var, j) for j, c in enumerate(idx)})
    return cols


@pytest.mark.parametrize("model", ["tsu_esaki", "fermi_table", "dielectric_table"])
def test_term_assembled_jacobian_is_the_derivative_of_the_assembled_residual(monkeypatch, model):
    """GateLeakageTerm.assemble on the stand-in's equation numbering: rows only at NON-contact interface nodes
    (node 2 is flagged a contact node here), residual -A_i J_i, and the Jacobian triplets summed per (row,
    column) equal the central difference of the assembled residual with respect to the solution value behind
    that column (h = 1e-6 V; budget 1e-6 of the largest entry), with every entry inside structural_pattern.
    The term is off (no entries) for scale 0 and for any time mode but DC."""
    fk = FD.install(monkeypatch, _fake())
    cols = _numbered(fk)
    contact = [False, True, False, False]
    if model == "tsu_esaki":
        mk = lambda s=1.0: _term("tsu_esaki", scale=s, contact=contact)            # noqa: E731
    elif model == "fermi_table":
        mk = lambda s=1.0: _term("table", table=_sinh_table(), scale=s, contact=contact)   # noqa: E731
    else:
        tab = GL.LeakageTable.from_field(tuple(np.linspace(-3e8, 3e8, 31)),
                                         tuple(5.0 * np.sinh(np.linspace(-3.0, 3.0, 31) / 0.25)), 1e-8)
        mk = lambda s=1.0: _term("table", table=tab, scale=s, contact=contact)      # noqa: E731
    t = mk()
    assert t.equations(DEV) == [("film", GL.FERMI_FIELD_EQUATION)]
    assert np.array_equal(t.rows_nodes, IFACE[[0, 2, 3]])

    def resid():
        _, rv = t.assemble(DEV, None, "DC")
        return dict(zip(rv[0::2], rv[1::2]))

    rcv, rv = t.assemble(DEV, None, "DC")
    rows = rv[0::2]
    perm = fk.numbers[("film", "equation", GL.FERMI_FIELD_EQUATION)]
    assert rows == [int(perm[n]) for n in IFACE[[0, 2, 3]]]
    st = t.state(DEV)
    assert np.allclose(rv[1::2], -(t.area * st["J"])[[0, 2, 3]], rtol=1e-15, atol=0.0)
    jac = {}
    for r, c, v in zip(rcv[0::3], rcv[1::3], rcv[2::3]):
        jac[(r, c)] = jac.get((r, c), 0.0) + v
    assert set(jac) <= set(t.structural_pattern(DEV))
    scale = max(abs(v) for v in jac.values())
    h = 1e-6
    for c, (reg, var, j) in cols.items():
        base = fk.nodes[(reg, var)].copy()
        fk.nodes[(reg, var)][j] = base[j] + h
        rp = resid()
        fk.nodes[(reg, var)][j] = base[j] - h
        rm = resid()
        fk.nodes[(reg, var)] = base
        for r in rows:
            fd = (rp[r] - rm[r]) / (2 * h)
            assert abs(jac.get((r, c), 0.0) - fd) <= 1e-6 * scale, (model, r, reg, var, j)
    assert mk(0.0).assemble(DEV, None, "DC") == ([], [])
    assert t.assemble(DEV, None, "TIME") == ([], [])


# =====================================================================================================
# setup_fermi_field_region / setup_fermi_contact / setup_gate_leakage on the stand-in
# =====================================================================================================
ETA = "((Potential - {} - Phi_c0) / V_t)".format(EF)


def _film_device(monkeypatch, eqdg, *, dimension=2):
    """A 2-D 'device' of an equilibrium film (with a Fermi-level node solution) on an oxide: registry records,
    the region's closure string, node models, and the stand-in routed everywhere (EQDG's module-level ds too)."""
    fk = FD.install(monkeypatch, FD.FakeDevsim(dimension))
    monkeypatch.setattr(eqdg, "ds", fk)
    fk.add_region("film", 4)
    fk.add_region("ox", 4)
    fk.interfaces["film_ox"] = ("film", "ox")
    for nm in ("Potential", EF, "x", "y"):
        fk.nodes[("film", nm)] = np.zeros(4)
    fk.nodes[("ox", "Potential")] = np.zeros(4)
    fk.node_eqs[("film", "Electrons")] = "N_c * exp({})".format(ETA)
    fk.node_eqs[("film", "QTerm_sheet0")] = "1.0"
    monkeypatch.setitem(R._REG, DEV, [
        {"scope": "region", "loc": r, "name": "PotentialEquation", "kwargs": {}} for r in ("film", "ox")])
    monkeypatch.setitem(PE._ETA_BY_REGION, (DEV, "film"), ETA)
    fk.writes = 0
    return fk


def _closure(eta):
    """A smooth stand-in closure n/N_c(eta) and its derivative (any function of eta alone will do)."""
    return np.exp(eta) * (1.0 + 0.1 * eta ** 2), np.exp(eta) * (1.0 + 0.2 * eta + 0.1 * eta ** 2)


def test_fermi_field_edge_current_derivatives_match_finite_differences(monkeypatch, eqdg_module):
    """setup_fermi_field_region writes the film's Fermi-level current j = q mu (n_0 + n_1)/2 (phi_0 - phi_1) / L
    and its four hand-derived derivatives. Oracle: the emitted strings evaluated on edge data from a closure
    n(eta) of eta = (psi - phi_F - Phi_c0)/V_t; each derivative model against the central difference of the
    emitted flux string in psi_0, psi_1, phi_0, phi_1 (h = 1e-7 V, relative budget 1e-7: the flux is smooth on
    the V_t scale, truncation ~(h/V_t)^2). Electrons:EF_ref_V must be -dn/dpsi (an eta-only closure)."""
    fk = _film_device(monkeypatch, eqdg_module)
    GL.setup_fermi_field_region(DEV, "film", mobility_m2Vs=3e-3)
    assert fk.params[("film", GL._MOBILITY)] == 3e-3
    assert fk.node_eqs[("film", "PotentialNodeCharge:" + EF)] == "-PotentialNodeCharge:Potential"
    assert fk.equations[-1]["name"] == GL.FERMI_FIELD_EQUATION
    assert fk.equations[-1]["variable_name"] == EF and fk.equations[-1]["edge_model"] == GL._FLUX
    Nc, eil = 4e25, 1.0 / 2e-9

    def edge_values(psi0, psi1, f0, f1):
        v = {"ElectronCharge": Q, GL._MOBILITY: 3e-3, "EdgeInverseLength": eil}
        for end, psi, f in (("n0", psi0, f0), ("n1", psi1, f1)):
            n, dn = _closure((psi - f - PHI_C0) / VT)
            v["Electrons@" + end] = Nc * n
            v["Electrons:Potential@" + end] = Nc * dn / VT
            v["{}@{}".format(EF, end)] = f
        for end in ("n0", "n1"):
            v["Electrons:{}@{}".format(EF, end)] = FD.evaluate(
                fk.node_eqs[("film", "Electrons:" + EF)], {"Electrons:Potential": v["Electrons:Potential@" + end]})
        return v

    flux = fk.edge_eqs[("film", GL._FLUX)]
    x0 = dict(psi0=0.05, psi1=0.08, f0=0.012, f1=-0.004)
    v0 = edge_values(**x0)
    assert float(FD.evaluate(flux, v0)) == pytest.approx(
        Q * 3e-3 * eil * 0.5 * (v0["Electrons@n0"] + v0["Electrons@n1"]) * (0.012 + 0.004), rel=1e-14)
    n0, dn0 = _closure((0.05 - 0.012 - PHI_C0) / VT)
    assert float(v0["Electrons:{}@n0".format(EF)]) == pytest.approx(-Nc * dn0 / VT, rel=1e-14)
    h = 1e-7
    for key, model in (("psi0", "Potential@n0"), ("psi1", "Potential@n1"), ("f0", EF + "@n0"),
                       ("f1", EF + "@n1")):
        xp, xm = dict(x0), dict(x0)
        xp[key] += h
        xm[key] -= h
        fd = (float(FD.evaluate(flux, edge_values(**xp))) - float(FD.evaluate(flux, edge_values(**xm)))) / (2 * h)
        got = float(FD.evaluate(fk.edge_eqs[("film", "{}:{}".format(GL._FLUX, model))], v0))
        assert got == pytest.approx(fd, rel=1e-7), model


def _refusal_case(fk, monkeypatch, case):
    if case == "dimension":
        fk.dimension = 1
    elif case == "drift_diffusion":
        R._REG[DEV].append({"scope": "region", "loc": "film", "name": "ElectronContinuityEquation", "kwargs": {}})
    elif case == "no_fermi_node":
        monkeypatch.setitem(PE._ETA_BY_REGION, (DEV, "film"), "((Potential - Phi_c0) / V_t)")
    elif case == "fermi_parameter":
        fk.params[("film", EF)] = 0.0
    elif case == "density_gradient":
        fk.params[("film", "eqdg_gamma")] = 1.8
    elif case == "odd_charge":
        fk.node_eqs[("film", "QTerm_custom")] = "1.0"
    elif case == "semiconductor_neighbour":
        fk.node_eqs[("ox", "Electrons")] = "1.0"


@pytest.mark.parametrize("case, exc, match", [
    ("dimension", NotImplementedError, "only 2-D devices"),
    ("drift_diffusion", ValueError, "is not an equilibrium semiconductor region"),
    ("no_fermi_node", ValueError, "has no Fermi-level node solution"),
    ("fermi_parameter", ValueError, "also carries a PARAMETER"),
    ("density_gradient", NotImplementedError, "equilibrium density gradient is not supported"),
    ("odd_charge", NotImplementedError, r"\['QTerm_custom'\]"),
    ("semiconductor_neighbour", NotImplementedError, "Fermi-field interface condition"),
])
def test_fermi_field_refusals_name_the_cause_and_touch_nothing(monkeypatch, eqdg_module, case, exc, match):
    """Every refusal of setup_fermi_field_region fires before the first model is written (the stand-in counts
    writes) and names its cause; the allowed fixed sheet charge (QTerm_sheet0) is NOT refused."""
    fk = _film_device(monkeypatch, eqdg_module)
    _refusal_case(fk, monkeypatch, case)
    fk.writes = 0
    with pytest.raises(exc, match=match):
        GL.setup_fermi_field_region(DEV, "film", mobility_m2Vs=1e-3)
    assert fk.writes == 0


@pytest.mark.parametrize("mu", [0.0, -1e-3, float("nan"), float("inf")])
def test_fermi_field_refuses_a_bad_mobility(monkeypatch, eqdg_module, mu):
    fk = _film_device(monkeypatch, eqdg_module)
    with pytest.raises(ValueError, match="mobility_m2Vs must be finite and > 0"):
        GL.setup_fermi_field_region(DEV, "film", mobility_m2Vs=mu)
    assert fk.writes == 0


def test_fermi_contact_is_a_dirichlet_on_the_fermi_level(monkeypatch):
    """setup_fermi_contact: residual phi_F - V_bias (zero exactly at phi_F = V_bias) with derivative 1 (its
    central difference), recorded on the Fermi-level equation with the film's edge current; the bias
    parameter is created at 0 only when missing (an existing value is kept)."""
    fk = FD.install(monkeypatch, FD.FakeDevsim(2))
    monkeypatch.setitem(R._REG, DEV, [])
    GL.setup_fermi_contact(DEV, "edge")
    assert fk.params[(None, "edge_bias")] == 0.0
    fk.params[(None, "edge_bias")] = 0.3
    GL.setup_fermi_contact(DEV, "edge")
    assert fk.params[(None, "edge_bias")] == 0.3
    res = fk.contact_eqs[("edge", "edge_fermi_dirichlet")]
    der = fk.contact_eqs[("edge", "edge_fermi_dirichlet:" + EF)]
    assert float(FD.evaluate(res, {EF: 0.3, "edge_bias": 0.3})) == 0.0
    fd = (FD.evaluate(res, {EF: 0.3 + 1e-3, "edge_bias": 0.3}) - FD.evaluate(res, {EF: 0.3 - 1e-3, "edge_bias": 0.3}))
    assert float(FD.evaluate(der, {})) == pytest.approx(float(fd) / 2e-3, rel=1e-12, abs=0)
    eq = fk.contact_equations[-1]
    assert eq["name"] == GL.FERMI_FIELD_EQUATION and eq["node_model"] == "edge_fermi_dirichlet"
    assert eq["edge_current_model"] == GL._FLUX
    assert R.recorded(DEV, GL.FERMI_FIELD_EQUATION, scope="contact", loc="edge")


def _leak_device(monkeypatch):
    fk = FD.install(monkeypatch, FD.FakeDevsim(2))
    fk.add_region("film", 4, material="FILM")
    fk.add_region("ox", 4)
    fk.add_region("cap", 4)
    fk.contacts.update({"top": "ox", "edge": "film", "semi_gate": "cap"})
    fk.node_eqs[("cap", "Electrons")] = "1.0"
    fk.region_equations["film"] = [GL.FERMI_FIELD_EQUATION]
    fk.params[("film", "V_t")] = VT
    fk.params[(None, "top_bias")] = 0.0
    return fk


def test_setup_gate_leakage_refusals_before_the_column_search(monkeypatch):
    fk = _leak_device(monkeypatch)
    tab = GL.GateLeakage(model="table", table=_sinh_table())
    with pytest.raises(TypeError, match="leakage must be a GateLeakage"):
        GL.setup_gate_leakage(DEV, "film", "top", {"model": "table"})
    fk.region_equations["film"] = []
    with pytest.raises(ValueError, match=r"setup_fermi_field_region first"):
        GL.setup_gate_leakage(DEV, "film", "top", tab)
    fk.region_equations["film"] = [GL.FERMI_FIELD_EQUATION]
    with pytest.raises(ValueError, match="no contact 'nowhere'"):
        GL.setup_gate_leakage(DEV, "film", "nowhere", tab)
    with pytest.raises(NotImplementedError, match="only metal gates on a dielectric"):
        GL.setup_gate_leakage(DEV, "film", "semi_gate", tab)


def test_setup_gate_leakage_wiring(monkeypatch):
    """With the column search replaced by fixed columns: the term reads the gate's bias parameter less the
    work-function offset, its energy grid is built at the FILM's V_t (kT = q V_t) up to energy_max_eV with
    energy_step_kT panels, the supply mass precedence is argument > GateLeakage field (and none is refused),
    a table model gets the gate's own table, and an undriven gate is refused."""
    fk = _leak_device(monkeypatch)
    cols = dict(chain=("ox",), iface=np.arange(2), is_contact=np.zeros(2, dtype=bool), area=np.ones(2),
                faces=[], layers=(LK.TunnelLayer(2.0, 1e-9, 0.3),), axis=1, interface="film_ox", lateral=[0])
    seen = []
    monkeypatch.setattr(GL, "_gate_columns", lambda *a: dict(cols))
    from dynameta.carriers import custom_terms as CT
    monkeypatch.setattr(CT, "add_custom_term", lambda dev, term: seen.append((dev, term)) or term)
    b = DielectricBarrier(2.0, "FILM", 0.3)
    lk = GL.GateLeakage(model="tsu_esaki", barriers=b, gate_fermi_energy_eV=7.0, m_supply_over_m0=0.4,
                        energy_max_eV=2.0, energy_step_kT=0.5, scale=0.5)
    t = GL.setup_gate_leakage(DEV, "film", "top", lk, gate_offset_V=0.25)
    assert seen == [(DEV, t)] and t.name == "gleak_film_top"
    assert t.gate_param == "top_bias" and t.gate_offset == 0.25 and t.scale == 0.5 and t.m_supply == 0.4
    assert t.gate_fermi_J == pytest.approx(7.0 * Q, rel=1e-15, abs=0)
    assert t.grid.kT_J == pytest.approx(Q * VT, rel=1e-15, abs=0) and t.grid.eps_max_J == pytest.approx(2.0 * Q, rel=1e-15, abs=0)
    assert t.grid.step_J <= 0.5 * Q * VT * (1 + 1e-12)
    fk.params[(None, "top_bias")] = 1.1
    assert t.gate_fermi_potential(DEV) == pytest.approx(0.85, rel=1e-15, abs=0)
    assert GL.setup_gate_leakage(DEV, "film", "top", lk, m_supply_over_m0=0.2, name="g2").m_supply == 0.2
    no_mass = GL.GateLeakage(model="tsu_esaki", barriers=b, gate_fermi_energy_eV=7.0)
    with pytest.raises(ValueError, match="no supply mass"):
        GL.setup_gate_leakage(DEV, "film", "top", no_mass)
    f = _sinh_table()
    tt = GL.setup_gate_leakage(DEV, "film", "top", GL.GateLeakage(model="table", table={"top": f}))
    assert tt.table is f and tt.model == "table"
    del fk.params[(None, "top_bias")]
    with pytest.raises(ValueError, match="not driven by the bias parameter 'top_bias'"):
        GL.setup_gate_leakage(DEV, "film", "top", lk)


# =====================================================================================================
# solve_lateral_1d input validation and Newton failures (pure numpy/scipy; no DEVSIM call; fast legs)
# =====================================================================================================
def _const_fn(v):
    return lambda p: (np.full(np.shape(p), v), np.zeros(np.shape(p)))


@pytest.mark.parametrize("kw, match", [
    (dict(v_contact=float("nan")), "v_contact must be finite"),
    (dict(v_contact=float("inf")), "v_contact must be finite"),
    (dict(length_m=float("inf")), "length_m must be > 0 and finite"),
    (dict(length_m=float("nan")), "length_m must be > 0 and finite"),
    (dict(tol_V=float("nan")), "tol_V must be finite and > 0"),
    (dict(tol_V=0.0), "tol_V must be finite and > 0"),
    (dict(max_iter=0), "max_iter must be >= 1"),
    (dict(n_nodes=1), "n_nodes must be >= 2"),
    (dict(x_m=np.array([0.0, float("nan"), 1e-4])), "x_m must be finite"),
])
def test_lateral_1d_refuses_non_finite_inputs_naming_the_argument(kw, match):
    """Each bad argument is a ValueError naming it, raised before any solve. OLD CODE (MEASURED): a NaN / inf
    v_contact reached scipy's solve_banded and escaped as its ValueError "array must not contain infs or NaNs"; an
    inf length_m produced NaN node positions and the same scipy error; a NaN length_m was refused without naming
    finiteness; tol_V = NaN or 0 ran every Newton step and raised "Newton did not converge (last update 7.87e-28
    V)"; max_iter = 0 raised UnboundLocalError; n_nodes = 1 raised numpy's "zero-size array to reduction
    operation maximum"; a NaN in x_m was reported as "x_m must increase"."""
    from dynameta.carriers import gate_leakage as GL
    args = dict(length_m=1e-4)
    args.update(kw)
    with np.errstate(all="ignore"):
        with pytest.raises(ValueError, match="solve_lateral_1d: " + match):
            GL.solve_lateral_1d(args.pop("length_m"), _const_fn(1e-3), _const_fn(1.0), **args)


def test_lateral_1d_non_finite_callback_reaches_the_module_message():
    """sheet_conductance or injected_current returning NaN: the Newton step fails with the module's own
    FloatingPointError, which the continuation catches (it halves the step) and finally raises. OLD CODE
    (MEASURED): scipy's solve_banded finiteness check raised ValueError("array must not contain infs or NaNs")
    first, which escaped the continuation's except clause, so the "non-finite Newton update" message was
    unreachable."""
    from dynameta.carriers import gate_leakage as GL
    for sig, J in ((_const_fn(float("nan")), _const_fn(1.0)), (_const_fn(1e-3), _const_fn(float("nan")))):
        with np.errstate(all="ignore"):
            with pytest.raises(FloatingPointError, match="solve_lateral_1d: non-finite Newton update"):
                GL.solve_lateral_1d(1e-4, sig, J, n_nodes=11)


def test_lateral_1d_singular_jacobian_reaches_the_module_message():
    """A zero sheet conductance with a current that does not depend on phi makes the Jacobian singular: the module's
    FloatingPointError naming it. OLD CODE (MEASURED): numpy.linalg.LinAlgError("singular matrix") escaped."""
    from dynameta.carriers import gate_leakage as GL
    with pytest.raises(FloatingPointError, match="solve_lateral_1d: singular Jacobian"):
        GL.solve_lateral_1d(1e-4, _const_fn(0.0), _const_fn(1.0), n_nodes=11)
