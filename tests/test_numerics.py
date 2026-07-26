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


_TRAPZ_NAMES = ("trapezoid", "trapz")


def _trapezoid_offenders(tree, label):
    """Every way a module can reach a scalar trapezoid other than core.numerics.trapz.

    The first version of this guard matched ONE shape -- `ast.Attribute` whose value is a
    `ast.Name` in a hard-coded {'np', 'numpy'} set -- and was therefore blind to four others
    (audit X-1 follow-on, each verified blind on a synthetic violation):
      * `from numpy import trapezoid` + a bare `trapezoid(...)` call;
      * ANY other alias for the numpy module (`import numpy as onp; onp.trapezoid(...)`) --
        the aliases are now read out of the file's OWN import statements, not guessed
        (dynameta/core/backend.py really does `import numpy as _np`);
      * `getattr(np, "trapezoid")(...)`;
      * the scipy spelling `scipy.integrate.trapezoid`, in either import form -- floor-safe,
        but it is still a second home for the same reduction (it was live in fiber_amp/lma.py).
    `cumulative_trapezoid` is deliberately NOT matched: it is a different reduction with no
    core.numerics home (see the core.numerics.trapz docstring's exception list)."""
    import ast
    np_aliases, bad_from = set(), []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name == "numpy" or a.name.startswith("numpy."):
                    np_aliases.add(a.asname or a.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if mod == "numpy" or mod.startswith("numpy.") or mod.startswith("scipy.integrate"):
                for a in node.names:
                    if a.name in _TRAPZ_NAMES:
                        bad_from.append("{}:{} (from {} import {})".format(label, node.lineno,
                                                                          mod, a.name))
    out = list(bad_from)
    for node in ast.walk(tree):                               # AST, so comments/docstrings are out
        if isinstance(node, ast.Attribute) and node.attr in _TRAPZ_NAMES:
            v = node.value
            if isinstance(v, ast.Name) and v.id in (np_aliases | {"np", "numpy"}):
                out.append("{}:{} ({}.{})".format(label, node.lineno, v.id, node.attr))
            elif isinstance(v, ast.Attribute) and v.attr == "integrate":  # scipy.integrate.trapezoid
                out.append("{}:{} (scipy.integrate.{})".format(label, node.lineno, node.attr))
        elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
              and node.func.id == "getattr" and len(node.args) >= 2
              and isinstance(node.args[0], ast.Name)
              and node.args[0].id in (np_aliases | {"np", "numpy"})
              and isinstance(node.args[1], ast.Constant) and node.args[1].value in _TRAPZ_NAMES):
            out.append("{}:{} (getattr({}, {!r}))".format(label, node.lineno, node.args[0].id,
                                                          node.args[1].value))
    return out


def test_no_direct_numpy_trapezoid_in_library():
    """The 'ONE home' claim, machine-checked over FIVE spellings (see _trapezoid_offenders).
    Documented exception, exhaustive: optics/spdc_design.py, whose getattr shim is itself
    floor-correct on BOTH sides (1.x has trapz, 2.x has trapezoid) and which needs the 2-D
    `axis=` form core.numerics.trapz does not offer. fiber_amp/lma.py used to need a second
    exemption for four bare scipy `trapezoid` calls; they were routed through core.numerics
    instead (byte-identical), so the allow-list stays at one entry."""
    import ast
    import io
    import pathlib
    root = pathlib.Path(__file__).resolve().parents[1] / "dynameta"
    allowed = {"spdc_design.py"}
    offenders, scanned = [], 0
    for f in sorted(root.rglob("*.py")):
        if f.name in allowed:
            continue
        scanned += 1
        tree = ast.parse(io.open(f, encoding="cp1252", errors="replace").read(), filename=str(f))
        offenders += _trapezoid_offenders(tree, str(f))
    assert scanned > 100                                      # the walk really did see the tree
    assert not offenders, ("direct numpy/scipy trapezoid call(s) outside the one home "
                           "(core.numerics.trapz): {}".format(offenders))


def test_trapezoid_guard_catches_every_spelling_it_claims_to():
    """The guard's own gate: each spelling below is a REAL floor hazard (or a second home) and
    must be caught; the two negatives must not fire. Four of these six were silently blind
    before (audit X-1 follow-on) -- a guard that cannot fail is not a guard."""
    import ast
    caught = {
        "np.trapezoid(y, x)": "import numpy as np\nv = np.trapezoid([1,2],[0,1])\n",
        "numpy.trapz(y, x)": "import numpy\nv = numpy.trapz([1,2],[0,1])\n",
        "from numpy import trapezoid": "from numpy import trapezoid\nv = trapezoid([1,2],[0,1])\n",
        "aliased numpy module": "import numpy as onp\nv = onp.trapezoid([1,2],[0,1])\n",
        "getattr(np, 'trapezoid')": "import numpy as np\nv = getattr(np,'trapezoid')([1,2],[0,1])\n",
        "from scipy.integrate import trapezoid":
            "from scipy.integrate import trapezoid\nv = trapezoid([1,2],[0,1])\n",
        "scipy.integrate.trapezoid": "import scipy.integrate\nv = scipy.integrate.trapezoid([1,2],[0,1])\n",
    }
    silent = {
        "docstring mention only": '"""np.trapezoid mention only"""\nv = 1\n',
        "cumulative_trapezoid (documented exception)":
            "from scipy.integrate import cumulative_trapezoid\nv = cumulative_trapezoid([1,2],[0,1])\n",
        "the one home": "from dynameta.core.numerics import trapz\nv = trapz([1,2],[0,1])\n",
    }
    for name, src in caught.items():
        assert _trapezoid_offenders(ast.parse(src), name), "guard is BLIND to: " + name
    for name, src in silent.items():
        assert not _trapezoid_offenders(ast.parse(src), name), "guard FALSE-FIRES on: " + name


def test_trapz_refuses_a_complex_integrand_instead_of_truncating_it():
    """audit X-1: the real-only cast is now explicit. A complex integrand used to be silently
    reduced to its real part (numpy ComplexWarning, easy to filter or miss) where np.trapezoid
    returns a complex value -- so a caller that reached here with a complex array lost half the
    answer with no error. Real input is untouched (byte-identical)."""
    y = np.array([1 + 1j, 2 + 2j, 3 + 3j])
    x = np.array([0.0, 1.0, 2.0])
    with pytest.raises(TypeError, match="REAL integrand"):
        trapz(y, x)
    with pytest.raises(TypeError, match="REAL integrand"):
        trapz([1 + 1j, 2 + 0j], [0.0, 1.0])                   # a python complex list too
    with pytest.raises(TypeError, match="REAL integrand"):
        trapz(x.real, x.astype(complex))                      # complex ABSCISSA as well
    # the documented workaround reproduces what np.trapezoid would have returned
    got = trapz(y.real, x) + 1j * trapz(y.imag, x)
    assert got == complex(np.trapezoid(y, x))
    # and the real path is unchanged
    assert trapz(y.real, x) == float(np.trapezoid(y.real, x))


def test_trapz_is_bit_identical_to_numpy_trapezoid_on_real_data():
    """The docstring's byte-note ('an EXACT power-of-two regrouping'), machine-checked, so the
    X-1 re-routing of eight call sites is provably answer-preserving."""
    rng = np.random.default_rng(7)
    for _ in range(500):
        n = int(rng.integers(2, 40))
        xx = np.sort(rng.normal(0.0, 10.0, n))
        yy = rng.normal(0.0, 1e6, n)
        assert trapz(yy, xx) == float(np.trapezoid(yy, xx))
