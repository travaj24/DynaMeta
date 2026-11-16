"""v0.16 equilibrium density gradient (carriers.physics_equilibrium_dg): the parts that are numpy around a few
DEVSIM reads -- the closure inversion, the adaptive continuation walker, the regime numbers and the validity
guard, the DG length, the seed of sqrt(n) and Lambda, and the refusals of setup_equilibrium_dg that fire before
any model is written -- plus the devsim-free equilibrium-closure helpers of physics_equilibrium they build on.
The module imports DEVSIM (through devsim_env) at module import, so on a leg without DEVSIM it is imported against the recording
stand-in of tests/_fake_devsim.py (and dropped again afterwards); the device gates are in
tests/test_equilibrium_density_gradient.py.

ORACLES: forward evaluation of the closure (round trip of the inversion); the continuation contract stated in
the docstring (monotone walk, step <= step, halve on failure and restore the last converged value, double after
two successes, re-raise below the smallest step); the exact Fermi-Dirac ratio F_1/2/F_-1/2 by quadrature for
the DG length (the library uses a rational fit with a documented 1.1 % peak error); k_F = (3 pi^2 n)^(1/3) and
the wall-distance geometry re-derived here; the seed's own defining relations (the closure tie u^2 = N_c
F_1/2(eta + Lambda/V_t) at every node, u = u_floor on the walls, the tanh^2 boundary-layer shape off them);
refusals that name their cause and write nothing.
"""
import math

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.optimize import brentq

import _fake_devsim as FD
from dynameta.carriers import eq_registry as R
from dynameta.carriers import physics_equilibrium as PE
from dynameta.warning_types import UnreliableResultWarning

DEV = "fake_eqdg_dev"
VT = 0.025851999786435535
HBAR = 1.054571817e-34
Q = 1.602176634e-19
M_E = 9.1093837015e-31


@pytest.fixture(scope="module")
def EQDG():
    with FD.modules_imported_with("dynameta.carriers.physics_equilibrium_dg") as (m,):
        yield m


def _fd_integral(j, eta):
    """Normalized Fermi-Dirac integral F_j(eta) = 1/Gamma(j+1) int_0^inf x^j / (1 + e^(x - eta)) dx (quadrature,
    the j = -1/2 case after x = s^2 so the integrand is regular)."""
    if j == -0.5:
        val = quad(lambda s: 2.0 / (1.0 + math.exp(min(s * s - eta, 700.0))), 0.0, math.sqrt(max(eta, 0.0) + 60.0),
                   epsabs=0.0, epsrel=1e-12, limit=400)[0]
    else:
        val = quad(lambda x: x ** j / (1.0 + math.exp(min(x - eta, 700.0))), 0.0, max(eta, 0.0) + 60.0,
                   epsabs=0.0, epsrel=1e-12, limit=400)[0]
    return val / math.gamma(j + 1.0)


# =====================================================================================================
# pure numerics
# =====================================================================================================
def test_invert_closure_round_trips_the_region_closure(EQDG):
    """_invert_closure: F_1/2^-1 by vectorized bisection on [-200, 200] (80 halvings: 3e-22 in eta, below double
    resolution), so F(eta(x)) = x to round-off of the closure (budget 1e-12 relative) across 40 decades, shape
    kept."""
    eta = np.linspace(-80.0, 150.0, 47).reshape(1, 47)
    x = np.vectorize(PE.F12_aymerich_humet)(eta)
    got = EQDG._invert_closure(x)
    assert got.shape == eta.shape
    assert np.allclose(np.vectorize(PE.F12_aymerich_humet)(got), x, rtol=1e-12, atol=0.0)
    assert np.allclose(got, eta, rtol=0.0, atol=1e-9 * 150.0)


class _Walker:
    """A continuation target: solve() fails whenever the applied value is more than `max_jump` from the last
    CONVERGED value, or lies inside `bad` (never solvable); records every apply / solve."""

    def __init__(self, start, max_jump=float("inf"), bad=None):
        self.value, self.conv = start, start
        self.max_jump, self.bad = max_jump, bad
        self.applied, self.solved = [], []

    def apply(self, v):
        self.applied.append(v)
        self.value = v

    def solve(self):
        v = self.value
        if abs(v - self.conv) > self.max_jump + 1e-12 or (self.bad is not None and self.bad(v)):
            raise RuntimeError("no convergence at {}".format(v))
        self.solved.append(v)
        self.conv = v


def test_adaptive_continuation_takes_full_steps_when_they_converge(EQDG):
    w = _Walker(0.0)
    EQDG.adaptive_continuation(w.apply, 0.0, 1.0, w.solve, step=0.25)
    assert w.solved == pytest.approx([0.25, 0.5, 0.75, 1.0], abs=1e-15) and w.solved[-1] == 1.0
    w = _Walker(2.0)                                   # downward, last step shortened to land on the end exactly
    EQDG.adaptive_continuation(w.apply, 2.0, 1.3, w.solve, step=0.25)
    assert w.solved == pytest.approx([1.75, 1.5, 1.3], abs=1e-15) and w.solved[-1] == 1.3
    w = _Walker(0.7)                                   # zero span: one apply + solve at the end value
    EQDG.adaptive_continuation(w.apply, 0.7, 0.7, w.solve, step=0.25)
    assert w.applied == [0.7] and w.solved == [0.7]


def test_adaptive_continuation_halves_restores_and_doubles_back(EQDG):
    """A target that converges only for jumps <= 0.07 from the last converged value, from 0 to 1 in steps of at
    most 0.25: every failed attempt is followed by re-applying the last converged value; accepted steps never
    exceed 0.07 nor 0.25; after two consecutive successes the step doubles (so steps of 0.0625 alternate with
    failed 0.125 attempts); the walk ends exactly at 1."""
    w = _Walker(0.0, max_jump=0.07)
    EQDG.adaptive_continuation(w.apply, 0.0, 1.0, w.solve, step=0.25)
    assert w.solved[-1] == 1.0
    steps = np.diff([0.0] + w.solved)
    assert np.all(steps > 0.0) and np.all(steps <= 0.07 + 1e-12)
    conv = 0.0
    it = iter(w.applied)
    for v in it:
        if abs(v - conv) > 0.07 + 1e-12:              # a failed attempt -> the next apply restores `conv`
            assert next(it) == conv
        else:
            conv = v
    assert w.applied[0] == 0.25                        # the first attempt is the full step
    assert set(steps.tolist()) == {0.0625}            # halved twice (0.25 -> 0.125 -> 0.0625), then held there
    # doubling: after every two accepted steps the next attempt is a 0.125 jump (which fails here)
    jumps = [v - c for v, c in zip(w.applied[1:], w.applied[:-1]) if v > c]
    assert jumps.count(0.125) >= 7


def test_adaptive_continuation_reraises_below_the_smallest_step(EQDG):
    """A target that never converges beyond 0.5: after max_halvings halvings below `step` the last failure is
    re-raised and the last converged value (0.5) is the one applied last."""
    w = _Walker(0.0, bad=lambda v: v > 0.5 + 1e-12)
    with pytest.raises(RuntimeError, match="no convergence"):
        EQDG.adaptive_continuation(w.apply, 0.0, 1.0, w.solve, step=0.25, max_halvings=3)
    assert w.solved[-1] == 0.5 and w.applied[-1] == 0.5
    fails = [v for v in w.applied if v > 0.5 + 1e-12]
    assert fails[-1] - 0.5 == pytest.approx(0.25 / 2 ** 3)


# =====================================================================================================
# the stand-in film: a column of nodes y in [0, t] (x = 0), walls at the ends
# =====================================================================================================
def _film(monkeypatch, EQDG, *, t=10e-9, n=41, n_bg=2e26, N_c=1e25, walls=(True, True), b_frac=1.0,
          gamma=1.8, m=0.30 * M_E, psi=None):
    fk = FD.install(monkeypatch, FD.FakeDevsim(2))
    monkeypatch.setattr(EQDG, "ds", fk)
    fk.add_region("film", n)
    y = np.linspace(0.0, t, n)
    fk.nodes[("film", "x")] = np.zeros(n)
    fk.nodes[("film", "y")] = y
    fk.nodes[("film", "Potential")] = (np.zeros(n) if psi is None else psi(y))
    wall = np.zeros(n)
    wall[0], wall[-1] = float(walls[0]), float(walls[1])
    fk.nodes[("film", EQDG.DG_WALL)] = wall
    b_full = gamma * HBAR ** 2 / (6.0 * m * Q)
    eta_bg = PE.invert_F12(n_bg / N_c)
    for k, v in (("N_c", N_c), ("N_D", n_bg), ("V_t", VT), ("Phi_c0", -eta_bg * VT), ("b_dg_full", b_full),
                 ("b_dg", b_frac * b_full), (EQDG.DG_U_FLOOR, 1e-6 * math.sqrt(n_bg)),
                 (EQDG._EQDG_FLAG, gamma)):
        fk.params[("film", k)] = v
    monkeypatch.setitem(PE._ETA_BY_REGION, (DEV, "film"), "((Potential - Phi_c0) / V_t)")
    fk.writes = 0
    return fk, y


def _exact_g(x):
    eta = brentq(lambda e: _fd_integral(0.5, e) - x, -40.0, 200.0, xtol=1e-13, rtol=1e-13)
    return _fd_integral(0.5, eta) / _fd_integral(-0.5, eta)


@pytest.mark.parametrize("x", [1e-3, 0.5, 5.0, 50.0])
def test_dg_length_against_the_exact_fermi_dirac_ratio(monkeypatch, EQDG, x):
    """equilibrium_dg_length_m = sqrt(b / (g V_t)), g = F_1/2/F_-1/2 at x = n/N_c. Oracle: g from the Fermi-Dirac
    integrals by quadrature (rtol 1e-12) at eta = F_1/2^-1(x); the library's g is a rational fit quoted at <= 1.1 %
    peak error up to eta ~ 32 (x = 50 is eta = 16), i.e. <= 0.56 % in the length: budget 0.6 % (MEASURED 4e-6,
    0.32 %, 0.31 %, 0.20 % at x = 1e-3, 0.5, 5, 50). The length
    scales as sqrt(b), and n_m3 overrides the default n_bg."""
    fk, _ = _film(monkeypatch, EQDG, n_bg=x * 1e25)
    b = fk.params[("film", "b_dg_full")]
    L = EQDG.equilibrium_dg_length_m(DEV, "film")
    assert L == pytest.approx(math.sqrt(b / (_exact_g(x) * VT)), rel=6e-3, abs=0)
    assert EQDG.equilibrium_dg_length_m(DEV, "film", n_m3=x * 1e25) == L
    fk.params[("film", "b_dg_full")] = 4.0 * b
    assert EQDG.equilibrium_dg_length_m(DEV, "film") == pytest.approx(2.0 * L, rel=1e-14, abs=0)


def test_regime_numbers_from_the_geometry(monkeypatch, EQDG):
    """equilibrium_dg_regime: eta_bg inverts the closure at n_bg (round trip N_c F_1/2(eta_bg) = n_bg, rtol 1e-9
    against the inversion's 1e-10 tolerance); t_eff = twice the largest distance to a wall node = t between two
    walls (the middle node is t/2 from both), 2 t over one wall, inf with none; subbands = t_eff k_F / pi with
    k_F = (3 pi^2 n_bg)^(1/3)."""
    fk, _ = _film(monkeypatch, EQDG, t=8e-9, n=33, n_bg=2e26, N_c=1e25)
    reg = EQDG.equilibrium_dg_regime(DEV, "film")
    assert 1e25 * PE.F12_aymerich_humet(reg["eta_bg"]) == pytest.approx(2e26, rel=1e-9)
    kf = (3.0 * math.pi ** 2 * 2e26) ** (1.0 / 3.0)
    assert reg["t_eff_m"] == pytest.approx(8e-9, rel=1e-12, abs=0)
    assert reg["subbands"] == pytest.approx(8e-9 * kf / math.pi, rel=1e-12, abs=0)
    fk.nodes[("film", EQDG.DG_WALL)][-1] = 0.0
    assert EQDG.equilibrium_dg_regime(DEV, "film")["t_eff_m"] == pytest.approx(16e-9, rel=1e-12, abs=0)
    fk.nodes[("film", EQDG.DG_WALL)][:] = 0.0
    reg = EQDG.equilibrium_dg_regime(DEV, "film")
    assert reg["t_eff_m"] == math.inf and reg["subbands"] == math.inf
    assert list(EQDG.equilibrium_dg_wall_nodes(DEV, "film")) == [False] * 33
    assert EQDG.equilibrium_dg_regions(DEV) == ["film"]


def test_validity_guard_is_silent_inside_and_names_each_violation_outside(monkeypatch, EQDG):
    """check_equilibrium_dg_validity: no warning inside the calibrated range (the error filter would fail the
    test); outside it one UnreliableResultWarning naming the violated number(s) with their thresholds -- the
    degeneracy (E_F - E_c)/kT < 7 and/or t k_F/pi < 2 with the wall-to-wall thickness in nm; a non-default
    gamma adds that no constant gamma is validated there, the default gamma does not."""
    _film(monkeypatch, EQDG, t=10e-9, n_bg=2e26, N_c=1e25)          # eta 9.7, t k_F/pi 5.8
    reg = EQDG.check_equilibrium_dg_validity(DEV, "film", gamma=1.5)
    assert reg["eta_bg"] > EQDG.VALID_ETA_MIN and reg["subbands"] > EQDG.VALID_SUBBANDS_MIN
    _film(monkeypatch, EQDG, t=1e-9, n=11, n_bg=2e26, N_c=1e25)      # t k_F/pi 0.58
    with pytest.warns(UnreliableResultWarning, match=r"t k_F/pi = 0\.58 < 2\.0 \(t = 1 nm") as rec:
        EQDG.check_equilibrium_dg_validity(DEV, "film", gamma=EQDG.EQUILIBRIUM_DG_GAMMA)
    assert "flat-band degeneracy" not in str(rec[0].message) and "is not the default" not in str(rec[0].message)
    _film(monkeypatch, EQDG, t=10e-9, n_bg=1e25, N_c=1e25)           # eta ~ 0.6
    with pytest.warns(UnreliableResultWarning, match=r"flat-band degeneracy \(E_F - E_c\)/kT = 0\.\d\d < 7\.0") as rec:
        EQDG.check_equilibrium_dg_validity(DEV, "film", gamma=1.5)
    assert "The given gamma = 1.5 is not the default" in str(rec[0].message)


def test_seed_satisfies_the_tie_and_the_wall_rows(monkeypatch, EQDG):
    """seed_equilibrium_dg on a gated profile (psi rising 0.1 V across the film): at every node the closure tie
    holds, u^2 = N_c F_1/2(eta + Lambda/V_t) (budget 1e-10 relative: the bisection resolves eta to round-off);
    u = u_floor on both wall nodes; off the walls n = max(tanh^2(d/L) n_cl, u_floor^2) with d the distance to the
    nearer wall and L the DG length scaled by sqrt(b/b_full) (b at half its full value here); Lambda < 0 off
    the walls (electrons pushed away from them) and Lambda = 0 where n = n_cl; the temporary eta model is
    deleted."""
    t, n = 10e-9, 41
    fk, y = _film(monkeypatch, EQDG, t=t, n=n, b_frac=0.5, psi=lambda y: 0.1 * y / 10e-9)
    EQDG.seed_equilibrium_dg(DEV, "film")
    u = fk.nodes[("film", "QSqrtN")]
    lam = fk.nodes[("film", "QLambda")]
    eta = (fk.nodes[("film", "Potential")] - fk.params[("film", "Phi_c0")]) / VT
    F = np.vectorize(PE.F12_aymerich_humet)
    assert np.allclose(u ** 2, 1e25 * F(eta + lam / VT), rtol=1e-10, atol=0.0)
    uf = fk.params[("film", EQDG.DG_U_FLOOR)]
    assert u[0] == uf and u[-1] == uf
    L = EQDG.equilibrium_dg_length_m(DEV, "film") * math.sqrt(0.5)
    d = np.minimum(y, t - y)
    n_cl = 1e25 * F(eta)
    expect = np.maximum(np.tanh(d / L) ** 2 * n_cl, uf ** 2)
    assert np.allclose(u[1:-1] ** 2, expect[1:-1], rtol=1e-12)
    assert np.all(lam[1:-1] < 0.0)
    assert "_EqDgSeedEta" in fk.deleted and ("film", "_EqDgSeedEta") not in fk.node_eqs


def test_seed_at_zero_b_is_classical_off_the_walls(monkeypatch, EQDG):
    """At b = 0 (the start of the b ramp) the interior keeps the classical density exactly (Lambda = 0, u =
    sqrt(n_cl) bit for bit) while the wall nodes are still seeded to u_floor with the Lambda that satisfies the
    tie there (the seeded wall avoids the slow Newton walk the docstring describes); with no wall at all every
    node is classical."""
    fk, y = _film(monkeypatch, EQDG, b_frac=0.0)
    EQDG.seed_equilibrium_dg(DEV, "film")
    u, lam = fk.nodes[("film", "QSqrtN")], fk.nodes[("film", "QLambda")]
    eta = (fk.nodes[("film", "Potential")] - fk.params[("film", "Phi_c0")]) / VT
    n_cl = 1e25 * np.vectorize(PE.F12_aymerich_humet)(eta)
    assert np.array_equal(u[1:-1], np.sqrt(n_cl[1:-1])) and np.all(lam[1:-1] == 0.0)
    uf = fk.params[("film", EQDG.DG_U_FLOOR)]
    assert u[0] == uf and lam[0] < -10.0 * VT
    assert 1e25 * PE.F12_aymerich_humet(eta[0] + lam[0] / VT) == pytest.approx(uf ** 2, rel=1e-10)
    fk.nodes[("film", EQDG.DG_WALL)][:] = 0.0
    EQDG.seed_equilibrium_dg(DEV, "film")
    assert np.all(fk.nodes[("film", "QLambda")] == 0.0)


# =====================================================================================================
# setup_equilibrium_dg refusals (before any model is written)
# =====================================================================================================
def _registry(monkeypatch, entries):
    monkeypatch.setitem(R._REG, DEV, [{"scope": "region", "loc": loc, "name": nm, "kwargs": {}} for loc, nm in entries])


@pytest.mark.parametrize("kw, match", [
    (dict(m_eff_kg=0.0), "m_eff_kg must be finite and > 0"),
    (dict(m_eff_kg=float("nan")), "m_eff_kg must be finite and > 0"),
    (dict(gamma=-0.1), "gamma must be finite and >= 0"),
    (dict(gamma=float("inf")), "gamma must be finite and >= 0"),
    (dict(u_floor_rel=0.0), r"u_floor_rel must be in \(0, 1\)"),
    (dict(u_floor_rel=1.0), r"u_floor_rel must be in \(0, 1\)"),
])
def test_setup_option_refusals(monkeypatch, EQDG, kw, match):
    fk, _ = _film(monkeypatch, EQDG)
    _registry(monkeypatch, [("film", "PotentialEquation")])
    args = dict(m_eff_kg=0.3 * M_E)
    args.update(kw)
    with pytest.raises(ValueError, match=match):
        EQDG.setup_equilibrium_dg(DEV, "film", **args)
    assert fk.writes == 0


def test_setup_refuses_regions_it_does_not_model(monkeypatch, EQDG):
    """A region that setup_semiconductor_region did not set up, and a drift-diffusion region, are refused as
    not equilibrium; a Kane band, a second installation and an interface to another semiconductor region are
    refused naming the cause; gamma = 0 installs nothing. None of them writes to the device."""
    fk, _ = _film(monkeypatch, EQDG)
    m = 0.3 * M_E
    _registry(monkeypatch, [("film", "PotentialEquation")])
    with pytest.raises(ValueError, match="is not an equilibrium semiconductor region"):
        EQDG.setup_equilibrium_dg(DEV, "other", m_eff_kg=m)
    R._REG[DEV].append({"scope": "region", "loc": "film", "name": "ElectronContinuityEquation", "kwargs": {}})
    with pytest.raises(ValueError, match="drift-diffusion regions use physics_density_gradient"):
        EQDG.setup_equilibrium_dg(DEV, "film", m_eff_kg=m)
    _registry(monkeypatch, [("film", "PotentialEquation"), ("ox", "PotentialEquation"),
                            ("film2", "PotentialEquation")])
    assert EQDG.setup_equilibrium_dg(DEV, "film", m_eff_kg=m, gamma=0.0) is None
    fk.node_eqs[("film", "KaneDelta")] = "0.0"
    with pytest.raises(NotImplementedError, match="uses a Kane band"):
        EQDG.setup_equilibrium_dg(DEV, "film", m_eff_kg=m)
    del fk.node_eqs[("film", "KaneDelta")]
    fk.nodes[("film", "QSqrtN")] = np.ones(41)
    with pytest.raises(ValueError, match="already carries density-gradient variables"):
        EQDG.setup_equilibrium_dg(DEV, "film", m_eff_kg=m)
    del fk.nodes[("film", "QSqrtN")]
    fk.add_region("ox", 4)
    fk.add_region("film2", 4)
    fk.node_eqs[("film2", "Electrons")] = "1.0"
    fk.interfaces.update({"film_ox": ("film", "ox"), "film_film2": ("film2", "film"), "ox_film2": ("ox", "film2")})
    with pytest.raises(NotImplementedError, match=r"interface\(s\) \['film_film2'\] to a region that is not a "
                                                  r"dielectric"):
        EQDG.setup_equilibrium_dg(DEV, "film", m_eff_kg=m)
    assert EQDG._dielectric_neighbours(DEV, "film") == (["film_ox"], ["film_film2"])
    # a neighbour without recorded physics (e.g. a metal region) is not a dielectric either: also refused
    fk.add_region("metal", 4)
    fk.interfaces = {"film_ox": ("film", "ox"), "film_metal": ("film", "metal")}
    assert EQDG._dielectric_neighbours(DEV, "film") == (["film_ox"], ["film_metal"])
    with pytest.raises(NotImplementedError, match=r"\['film_metal'\]"):
        EQDG.setup_equilibrium_dg(DEV, "film", m_eff_kg=m)
    assert fk.writes == 0


# =====================================================================================================
# the equilibrium-closure helpers the DG builds on (physics_equilibrium, devsim-free)
# =====================================================================================================
def test_effective_dos_matches_the_textbook_value():
    """N_c = 2 (m kT / (2 pi hbar^2))^(3/2): 2.5094e25 m^-3 for the free-electron mass at 300 K (a tabulated
    constant), scaling as m^(3/2) T^(3/2)."""
    n0 = PE.effective_dos_m3(M_E, 300.0)
    assert n0 == pytest.approx(2.5094e25, rel=2e-4)
    assert PE.effective_dos_m3(0.25 * M_E, 1200.0) == pytest.approx(n0, rel=1e-12)


def test_closure_inversion_refusals_name_the_bracket():
    with pytest.raises(ValueError, match="target must be positive"):
        PE.invert_F12(0.0)
    with pytest.raises(ValueError, match="exceeds F_1/2\\(eta_max=80\\)"):
        PE.invert_F12(1e4)
    with pytest.raises(ValueError, match="below F_1/2\\(eta_min=-20\\)"):
        PE.invert_F12(1e-12)
    eta = PE.invert_monotone(lambda e: math.exp(e), 2.0, -5.0, 5.0, 1e-13, name="t", closure="exp")
    assert eta == pytest.approx(math.log(2.0), abs=1e-12)
    # a closure with a jump across the target never meets the tolerance: refused, not the midpoint returned
    with pytest.raises(RuntimeError, match="jump did not converge"):
        PE.invert_monotone(lambda e: 1.0 if e < 0.3 else 3.0, 2.0, -5.0, 5.0, name="jump")


def test_band_selection_and_work_function_offset():
    """kane_band_of: None / 'parabolic' -> parabolic (None), a duck-typed Kane band is returned as is, anything
    else is refused; closure_kwargs builds the matching setup keywords. work_function_offset_V = -(phi_m - W_s),
    W_s = chi + (E_c - E_F): a gate whose work function equals W_s has zero offset; a missing affinity and a
    non-positive work function are refused naming the input."""
    class Kane:
        m0_kg, alpha_per_eV = 0.3 * M_E, 0.4

    class Tr:
        band = Kane()

    k = Kane()
    assert PE.kane_band_of(None) is None and PE.kane_band_of("parabolic") is None and PE.kane_band_of(k) is k
    with pytest.raises(TypeError, match="band must be 'parabolic'"):
        PE.kane_band_of("kane")
    assert PE.closure_kwargs(object(), 1e-31) == {"dos_mass_kg": 1e-31}
    assert PE.closure_kwargs(Tr(), 1e-31) == {"band": Tr.band}
    assert PE.work_function_offset_V(4.7, 4.5, 0.2) == pytest.approx(0.0, abs=1e-15)
    assert PE.work_function_offset_V(5.1, 4.5, -0.3) == pytest.approx(-(5.1 - 4.2), rel=1e-14, abs=0)
    with pytest.raises(ValueError, match="chi_eV is None"):
        PE.work_function_offset_V(5.1, None, 0.0)
    with pytest.raises(ValueError, match="work_function_eV must be > 0"):
        PE.work_function_offset_V(-1.0, 4.5, 0.0)
    with pytest.raises(ValueError, match="is not an equilibrium semiconductor region"):
        PE.equilibrium_eta_expr("no_such_device", "film")
