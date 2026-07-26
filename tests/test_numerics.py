"""Unit coverage for the shared trapezoidal integrator (core.numerics.trapz) -- previously used
library-wide but never directly tested (audit xcut-1). Pure numpy."""
import numpy as np
import pytest

from dynameta.core.numerics import trapz


def test_trapz_line():
    x = np.linspace(0.0, 1.0, 101)
    assert trapz(x, x) == pytest.approx(0.5, abs=1e-6)        # integral of y=x over [0,1]


def test_trapz_constant():
    x = np.linspace(0.0, 2.0, 51)
    assert trapz(np.full_like(x, 3.0), x) == pytest.approx(6.0, abs=1e-9)


def test_trapz_empty_and_single_are_zero():
    assert trapz(np.array([]), np.array([])) == 0.0
    assert trapz(np.array([1.0]), np.array([0.0])) == 0.0    # one node -> no interval


def test_trapz_shape_mismatch_raises():
    with pytest.raises(ValueError):                          # numpy broadcast failure (2 vs 4)
        trapz(np.array([1.0, 2.0, 3.0]), np.array([0.0, 1.0, 2.0, 3.0, 4.0]))


# ------------------------------------------------------------------------------------------------
# AUDIT X-1: the numpy>=1.24 floor. `np.trapezoid` does not exist before numpy 2.0 (the identifier
# is absent from the 1.24 wheel entirely) and `np.trapz` was removed in 2.x, so neither spelling is
# callable across the DECLARED floor -- only core.numerics.trapz is. Eight library sites called
# np.trapezoid directly and raised AttributeError on a 1.24 install; CI (numpy 2.x only) could not
# see it. These gates simulate the 1.24 wheel by DELETING both numpy spellings.
# ------------------------------------------------------------------------------------------------
def _simulate_numpy_124(monkeypatch):
    """Make the environment look like the numpy 1.24 wheel to code that reaches for either
    spelling: `trapezoid` absent (1.24 has only `trapz`) and `trapz` absent (2.x removed it), so
    ANY direct numpy trapezoid call raises AttributeError."""
    monkeypatch.delattr(np, "trapezoid", raising=False)
    monkeypatch.delattr(np, "trapz", raising=False)
    assert not hasattr(np, "trapezoid") and not hasattr(np, "trapz")


def test_shared_trapz_survives_missing_numpy_trapezoid(monkeypatch):
    x = np.linspace(0.0, 2.0, 51)
    ref = trapz(np.sin(x), x)
    _simulate_numpy_124(monkeypatch)
    assert trapz(np.sin(x), x) == ref                        # bit-identical, no numpy name needed


def test_density_gradient_conserve_charge_on_numpy_124(monkeypatch):
    """carriers/density_gradient.py:118 (x2) -- `conserve_charge=True` used np.trapezoid AFTER the
    expensive BVP had already converged (ledger C-12 / S1-1)."""
    from dynameta.constants import M_E
    from dynameta.carriers.density_gradient import dg_correct_density_1d
    z = np.linspace(0.0, 15e-9, 401)
    n_cl = np.full_like(z, 2e26)
    ref = dg_correct_density_1d(z, n_cl, 0.35 * M_E, conserve_charge=True)
    _simulate_numpy_124(monkeypatch)
    got = dg_correct_density_1d(z, n_cl, 0.35 * M_E, conserve_charge=True)
    assert np.array_equal(got, ref)                          # byte-identical, not merely close


def test_nonlinear_limits_on_numpy_124(monkeypatch):
    """fiber_amp/nonlinear_limits.py:203, :286, :402 -- sbs_gain_exponent, srs_gain_exponent and
    double_rayleigh_mpi (7 of the module's 18 public names route through these)."""
    from dynameta.optics.fiber_amp.steady_state import SteadyStateResult
    from dynameta.optics.fiber_amp.waveguide import FiberSpec
    from dynameta.optics.fiber_amp.nonlinear_limits import (double_rayleigh_mpi,
                                                            sbs_gain_exponent, srs_gain_exponent)
    z = np.linspace(0.0, 5.0, 121)
    P = 1.0 * np.exp(1.2 * z / 5.0)                          # a rising amplifier signal profile
    res = SteadyStateResult(z_m=z, power_W=P.reshape(1, -1), lambda_m=np.array([1.064e-6]),
                            u=np.array([1.0]), is_ase=np.array([False]), kind=["signal"],
                            nbar2_z=np.zeros_like(z),
                            signal_gain_dB=np.array([10.0 * np.log10(P[-1] / P[0])]), meta={})
    fib = FiberSpec(10e-6, 0.06, 3e25, 5.0)
    calls = [lambda: sbs_gain_exponent(res, fib, 1.064e-6)["G_B"],
             lambda: srs_gain_exponent(res, fib, 1.064e-6)["G_R"],
             lambda: double_rayleigh_mpi(res, fib, 1.064e-6)]
    ref = [f() for f in calls]
    _simulate_numpy_124(monkeypatch)
    assert [f() for f in calls] == ref                       # byte-identical


def test_eryb_diagnostics_on_numpy_124(monkeypatch):
    """fiber_amp/eryb.py:414, :415, :434 -- the transfer-efficiency and Yb-parasitic-gain
    integrals, both evaluated inside every ErYbAmplifier.solve()."""
    from dynameta.optics.fiber_amp import AseBand, FiberSpec, Pump, Signal, erbium, ytterbium
    from dynameta.optics.fiber_amp.eryb import ErYbAmplifier

    def _solve():
        fib = FiberSpec(3.2e-6, 0.20, 3e25, 4.0, clad_radius_m=125e-6)
        amp = ErYbAmplifier(erbium("aluminosilicate"), ytterbium("phosphosilicate"), fib,
                            [Pump(3.0, 0.976e-6, "fwd", cladding=True)],
                            [Signal(20e-3, 1.550e-6)], AseBand(1.53e-6, 1.565e-6, 4),
                            n_yb_m3=2.5e26, k_tr_m3_s=2e-22)
        r = amp.solve(n_nodes=41)
        return r.meta["eta_transfer"], r.meta["yb_parasitic_gain_dB"]

    ref = _solve()
    assert 0.0 < ref[0] < 1.0
    _simulate_numpy_124(monkeypatch)
    assert _solve() == ref                                   # byte-identical


def test_no_direct_numpy_trapezoid_in_library():
    """The 'ONE home' claim, machine-checked: no library module may reach for a numpy trapezoid
    spelling directly. Sole documented exception: optics/spdc_design.py, whose getattr shim is
    itself floor-correct on BOTH sides (1.x has trapz, 2.x has trapezoid) and which needs the
    2-D `axis=` form core.numerics.trapz does not offer."""
    import ast
    import io
    import pathlib
    root = pathlib.Path(__file__).resolve().parents[1] / "dynameta"
    allowed = {"spdc_design.py"}
    offenders = []
    for f in sorted(root.rglob("*.py")):
        if f.name in allowed:
            continue
        tree = ast.parse(io.open(f, encoding="cp1252", errors="replace").read(), filename=str(f))
        for node in ast.walk(tree):                           # AST, so comments/docstrings are out
            if (isinstance(node, ast.Attribute) and node.attr in ("trapezoid", "trapz")
                    and isinstance(node.value, ast.Name) and node.value.id in ("np", "numpy")):
                offenders.append("{}:{}".format(f, node.lineno))
    assert not offenders, ("direct numpy trapezoid call(s) outside the one home "
                           "(core.numerics.trapz): {}".format(offenders))
