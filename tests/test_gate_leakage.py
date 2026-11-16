"""v0.16 gate leakage (carriers.gate_leakage): the pure parts -- the measured-leakage table, the input checks, the
reduced 1-D lateral model, and the tunnelling kernel the device term uses (reliability.leakage.tsu_esaki_two_sided)
against the textbook Fowler-Nordheim and direct-tunnelling limits. No DEVSIM (the device gates are in
tests/test_devsim_gate_leakage.py).

ORACLES: closed forms re-derived here -- the distributed resistor V = J (L x - x^2/2) / sigma (and J x (L - x) /
(2 sigma)); scipy solve_bvp of the sheet equation; Fowler-Nordheim J = A E^2 exp(-B/E) with the T = 0 curvature term
(1 - E/B) and the Murphy-Good thermal factor; the T = 0, V -> 0 WKB direct-tunnelling integral through a rectangular
barrier in closed form (Simmons' low-voltage law is its leading term).
OLD CODE (before this module): carriers.gate_leakage does not exist -> every test here fails at import.
"""
import math

import numpy as np
import pytest

from dynameta.carriers import gate_leakage as GL
from dynameta.reliability import leakage as LK

Q = 1.602176634e-19
HBAR = 1.054571817e-34
M_E = 9.1093837015e-31
KB = 1.380649e-23


# =====================================================================================================
# LeakageTable
# =====================================================================================================
def test_table_reproduces_its_points_and_its_derivative():
    """PCHIP of asinh(J/J0): exact at the table points (to 1e-12 relative), C^1, odd tables stay odd, the derivative is
    the derivative of the interpolant (central differences, 1e-6)."""
    V = np.linspace(-3.0, 3.0, 25)
    J = 1e-3 * np.sinh(V / 0.2)
    t = GL.LeakageTable(tuple(V), tuple(J))
    Jt, _ = t(V)
    assert np.allclose(Jt, J, rtol=1e-12, atol=0.0)          # exact 0 at V = 0 (MEASURED <= 1.5e-15)
    v = np.linspace(-2.9, 2.9, 97) + 1e-3
    h = 1e-6
    _, dJ = t(v)
    fd = (t(v + h)[0] - t(v - h)[0]) / (2 * h)
    assert np.allclose(dJ, fd, rtol=1e-6, atol=0.0)           # MEASURED <= 1.0e-9 relative
    assert np.allclose(t(-v)[0], -t(v)[0], rtol=1e-12, atol=0.0)   # MEASURED <= 4.8e-14
    assert t.in_range(v) and not t.in_range([3.5])


def test_table_extrapolates_exponentially_and_from_field_scales():
    """Beyond the table y = asinh(J/J0) continues with the end segment's slope: J is continuous at the end and grows by
    the end segment's ratio per segment width (1e-2 -> 1 over [1, 2] V, so ~100 at 3 V); from_field maps E -> V = E t."""
    t = GL.LeakageTable((0.0, 1.0, 2.0), (0.0, 1e-2, 1.0))
    a, _ = t(2.0)
    b, db = t(2.0 + 1e-9)
    assert b == pytest.approx(a, rel=1e-7) and db > 0.0
    J3, _ = t(3.0)
    assert J3 == pytest.approx(100.0, rel=1e-6)
    f = GL.LeakageTable.from_field((0.0, 1e8, 2e8), (0.0, 1e-2, 1.0), 10e-9)
    assert f.V_V == (0.0, 1.0, 2.0)
    # a J(E) table is read at the solved dielectric drop, a J(V) table at the Fermi-potential difference
    assert f.variable == "dielectric" and t.variable == "fermi"
    with pytest.raises(ValueError, match="variable"):
        GL.LeakageTable((0.0, 1.0), (0.0, 1.0), variable="field")
    with pytest.raises(ValueError, match="do not pass variable"):
        GL.LeakageTable.from_field((0.0, 1e8), (0.0, 1.0), 10e-9, variable="fermi")


@pytest.mark.parametrize("args, match", [
    (((0.0,), (0.0,)), "length >= 2"),
    (((0.0, 0.0), (0.0, 1.0)), "strictly increasing"),
    (((0.0, 1.0), (0.0, float("nan"))), "non-finite"),
])
def test_table_refusals(args, match):
    with pytest.raises(ValueError, match=match):
        GL.LeakageTable(*args)


def test_gate_leakage_input_checks():
    from dynameta.carriers.tunneling import DielectricBarrier
    b = DielectricBarrier(2.5, "FILM", 0.2)
    with pytest.raises(ValueError, match="model"):
        GL.GateLeakage(model="fn")
    with pytest.raises(ValueError, match="needs barriers"):
        GL.GateLeakage(model="tsu_esaki", gate_fermi_energy_eV=10.0)
    with pytest.raises(ValueError, match="gate_fermi_energy_eV"):
        GL.GateLeakage(model="tsu_esaki", barriers=b)
    with pytest.raises(ValueError, match="needs table"):
        GL.GateLeakage(model="table")
    with pytest.raises(ValueError, match="scale"):
        GL.GateLeakage(model="table", table=lambda v: (v, v), scale=-1.0)
    g = GL.GateLeakage(model="tsu_esaki", barriers={"ox": b}, gate_fermi_energy_eV=10.0, gates="top")
    assert g.gates == ("top",) and g.barrier_for("ox") is b
    with pytest.raises(ValueError, match="no entry"):
        g.barrier_for("other")


# =====================================================================================================
# the reduced 1-D lateral model
# =====================================================================================================
def _const(v):
    return lambda p: (np.full(np.shape(p), v), np.zeros(np.shape(p)))


@pytest.mark.parametrize("contacts", ["left", "both"])
def test_lateral_1d_reproduces_the_distributed_resistor(contacts):
    """Uniform J into a uniform sheet: the finite-volume solution on the graded grid is exact for the quadratic
    (MEASURED <= 1.1e-15 relative); budget 1e-12."""
    L, sig, J = 1e-3, 1e-3, 50.0
    r = GL.solve_lateral_1d(L, _const(sig), _const(J), v_contact=0.2, contacts=contacts, n_nodes=301)
    x = r["x_m"]
    ana = 0.2 + (J * (L * x - 0.5 * x ** 2) / sig if contacts == "left" else J * x * (L - x) / (2 * sig))
    assert np.max(np.abs(r["phi_V"] - ana)) <= 1e-12 * np.max(np.abs(ana - 0.2))
    assert r["max_drop_V"] == pytest.approx(J * L ** 2 / (2 * sig) if contacts == "left" else J * L ** 2 / (8 * sig),
                                            rel=1e-12, abs=0)


def test_lateral_1d_nonlinear_matches_solve_bvp():
    """sigma(phi) = 1e-3 (1 - 0.1 phi) S, J(phi) = 2e3 sinh((2 - phi)/0.3) A/m^2, L = 300 um: the self-limiting drop.
    Oracle: scipy solve_bvp (tol 1e-10, scaled) of phi' = -j/sigma, j' = J. MEASURED 4.1e-6 of the 1.236 V drop on the
    default 801-node graded grid, 1.0e-6 on 1601 nodes (second order); budget 2e-5."""
    from scipy.integrate import solve_bvp
    L = 300e-6

    def sig(p):
        return 1e-3 * (1.0 - 0.1 * p), -1e-4 * np.ones_like(p)

    def J(p):
        u = (2.0 - p) / 0.3
        return 2e3 * np.sinh(u), -2e3 * np.cosh(u) / 0.3

    r = GL.solve_lateral_1d(L, sig, J)
    u = np.linspace(0.0, 1.0, 2001)                     # u = x / L, w = j L / (sigma0 V0): O(1) unknowns
    s0 = 1e-3
    s = solve_bvp(lambda u_, y: np.vstack([-y[1] * s0 / sig(y[0])[0], L * L * J(y[0])[0] / s0]),
                  lambda a, b: np.array([a[0], b[1]]), u, np.zeros((2, u.size)), tol=1e-10, max_nodes=500000)
    assert s.status == 0, s.message
    ana = s.sol(r["x_m"] / L)[0]
    assert 0.5 < ana.max() < 2.0
    assert np.max(np.abs(r["phi_V"] - ana)) <= 2e-5 * ana.max()


# =====================================================================================================
# the kernel against the textbook limits
# =====================================================================================================
def _one_layer(offset_eV, t, m, x_s, x_g, d, kT, emax_eV, m_supply, step_kT=0.125):
    grid = LK.tunnel_energy_grid(kT, emax_eV * Q, step_kT=step_kT)
    return LK.tsu_esaki_two_sided(grid, [x_s], [x_g], [[d]], [LK.TunnelLayer(offset_eV, t, m)],
                                  m_supply_over_m0=m_supply).J[0]


def test_kernel_fowler_nordheim_limit():
    """Metal-like supply (E_F = 5 eV above the band bottom, m_supply = m0), barrier phi = 3.1 eV above E_F, m_ox 0.42,
    t = 5 nm, V = 5 and 7.5 V (E = 1.0 / 1.5e9 V/m > phi/t). Textbook: J -> J_FN (1 - E/B) MG(T) with the T = 0
    curvature term 1 - E/B of the linearised exponent and the Murphy-Good factor MG = pi c kT / sin(pi c kT),
    c = 2 sqrt(2 m_ox phi) / (q hbar E) [1/J]. At 30 K (MG = 1.004) the remainder must be O((E/B)^2): MEASURED
    3.41e-3 and 7.35e-3 = 1.99 and 1.91 (E/B)^2, their ratio 2.16 vs (E/B)^2 ratio 2.25 (convergence form; budget
    3 (E/B)^2 and a ratio in [1.8, 2.7]). At 300 K (MG = 1.168) the kernel is within 8.7e-3 of the prediction
    (budget 1.5e-2; the MG factor's own higher-order terms)."""
    EF, phi, m, t = 5.0, 3.1, 0.42, 5e-9
    _a, B = LK.fn_coefficients(m, phi)

    def resid(T, V, step):
        kT = KB * T
        x_s = EF * Q / kT
        J = _one_layer(EF + phi, t, m, x_s, x_s - V * Q / kT, -V, kT, EF + 1.0, 1.0, step_kT=step)
        E = V / t
        c = 2.0 * math.sqrt(2.0 * m * M_E * phi * Q) / (Q * HBAR * E)
        mg = math.pi * c * kT / math.sin(math.pi * c * kT)
        return J / (LK.fowler_nordheim_current(E, m_ox_over_m0=m, phi_b_eV=phi) * (1.0 - E / B) * mg) - 1.0, E / B

    r1, e1 = resid(30.0, 5.0, 0.25)
    r2, e2 = resid(30.0, 7.5, 0.25)
    assert 0.0 < r1 < 3.0 * e1 ** 2 and 0.0 < r2 < 3.0 * e2 ** 2
    assert 1.8 < r2 / r1 < 2.7
    r300, _ = resid(300.0, 5.0, 0.125)
    assert abs(r300) < 1.5e-2


def test_kernel_direct_tunnelling_limit():
    """Rectangular barrier phi = 3 eV above E_F (E_F = 5 eV), t = 2 nm, m_ox = m_supply = 0.5 m0, V = 1 mV: the T = 0,
    V -> 0 WKB integral in closed form, J = (q^2 m V / (2 pi^2 hbar^3)) (2/b^2) [e^(-b sqrt(phi)) (b sqrt(phi) + 1) -
    (same at phi + E_F)], b = 2 t sqrt(2 m) / hbar, times the Murphy-Good factor at 30 K (a 30 K grid keeps the
    thermal correction ~1e-3). MEASURED agreement 6.6e-6; budget 1e-4. Simmons' low-voltage law (the leading term,
    without the +1) is 3.8 % below it here (b sqrt(phi) = 25)."""
    T = 30.0
    kT = KB * T
    EF, phi, m, t, V = 5.0, 3.0, 0.5, 2e-9, 1e-3
    x_s = EF * Q / kT
    J = _one_layer(EF + phi, t, m, x_s, x_s - V * Q / kT, -V, kT, EF + 0.5, m, step_kT=0.25)
    b = 2.0 * t * math.sqrt(2.0 * m * M_E) / HBAR
    p0, p1 = phi * Q, (phi + EF) * Q

    def F(w):
        return math.exp(-b * math.sqrt(w)) * (b * math.sqrt(w) + 1.0)

    pref = Q * Q * m * M_E * V / (2.0 * math.pi ** 2 * HBAR ** 3)
    J0 = pref * (2.0 / b ** 2) * (F(p0) - F(p1))
    c = b / (2.0 * math.sqrt(p0))
    mg = math.pi * c * kT / math.sin(math.pi * c * kT)
    assert J == pytest.approx(J0 * mg, rel=1e-4)
    simmons = pref * (2.0 / b ** 2) * b * math.sqrt(p0) * math.exp(-b * math.sqrt(p0))
    assert 0.9 < simmons / J0 < 1.0
