"""Unit coverage for the shared numeric helpers in core.numerics: the trapezoidal integrator
(trapz -- used library-wide but never directly tested, audit xcut-1) and ZeroInitBDF, the
BDF subclass that stops scipy reading a row of its own difference array that it never wrote."""
import warnings

import numpy as np
import pytest

from dynameta.core.numerics import trapz

# The ORACLE for the bit-identity claims below -- resolved ONCE, at import, to whichever spelling
# this numpy carries (`trapezoid` on 2.x, `trapz` on the declared 1.24 floor; they are the same
# function under two names). Spelling either one inline would make this module -- the module that
# exists to police exactly that -- the last floor break in tests/, which is what the CI floor leg's
# now-deleted per-test deselect list was hiding (audit X-1).
_NUMPY_TRAPEZOID = getattr(np, "trapezoid", None) or getattr(np, "trapz", None)


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
    import pathlib
    root = pathlib.Path(__file__).resolve().parents[1] / "dynameta"
    allowed = {"spdc_design.py"}
    offenders, scanned = [], 0
    for f in sorted(root.rglob("*.py")):
        if f.name in allowed:
            continue
        scanned += 1
        # audit T-13: was `io.open(...).read()`, which leaked one handle per file (~180 here,
        # ~120 in the sibling scan below) and produced a ResourceWarning + a
        # PytestUnraisableExceptionWarning at GC -- invisible until `filterwarnings = error`
        # landed, then two failures with a traceback that named neither test.
        tree = ast.parse(f.read_text(encoding="cp1252", errors="replace"), filename=str(f))
        offenders += _trapezoid_offenders(tree, str(f))
    assert scanned > 100                                      # the walk really did see the tree
    assert not offenders, ("direct numpy/scipy trapezoid call(s) outside the one home "
                           "(core.numerics.trapz): {}".format(offenders))


def test_no_direct_numpy_trapezoid_in_tests_and_validation():
    """audit X-1, the last leg. The LIBRARY was routed through core.numerics.trapz; tests/ and
    validation/ were not, so 8 tests + 9 validation call sites still raised AttributeError on a
    numpy-1.24 install. CI could not see it either way -- every ubuntu leg resolves numpy 2.x --
    and the floor leg carried a hand-maintained `--deselect` line PER TEST to stay green. All 17
    now route here, the deselect list is deleted from .github/workflows/ci.yml, and this gate is
    what keeps it deleted.

    Allow-list, exhaustive: this module, whose `_NUMPY_TRAPEZOID` is the ORACLE the bit-identity
    claims are checked against and is itself resolved floor-safely (`trapezoid` or `trapz`,
    whichever this numpy has)."""
    import ast
    import pathlib
    root = pathlib.Path(__file__).resolve().parents[1]
    allowed = {"test_numerics.py"}
    offenders, scanned = [], 0
    for sub in ("tests", "validation"):
        for f in sorted((root / sub).rglob("*.py")):
            if f.name in allowed:
                continue
            scanned += 1
            tree = ast.parse(f.read_text(encoding="cp1252", errors="replace"),   # audit T-13
                             filename=str(f))
            offenders += _trapezoid_offenders(tree, str(f))
    assert scanned > 100                                      # both trees really were walked
    assert not offenders, ("direct numpy/scipy trapezoid call(s) in tests/ or validation/ -- these "
                           "break on the declared numpy>=1.24 floor; use core.numerics.trapz: "
                           "{}".format(offenders))


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
    # the documented workaround reproduces what numpy's trapezoid would have returned
    got = trapz(y.real, x) + 1j * trapz(y.imag, x)
    assert got == complex(_NUMPY_TRAPEZOID(y, x))
    # and the real path is unchanged
    assert trapz(y.real, x) == float(_NUMPY_TRAPEZOID(y.real, x))


def test_trapz_is_bit_identical_to_numpy_trapezoid_on_real_data():
    """The docstring's byte-note ('an EXACT power-of-two regrouping'), machine-checked, so the
    X-1 re-routing of eight call sites is provably answer-preserving."""
    rng = np.random.default_rng(7)
    for _ in range(500):
        n = int(rng.integers(2, 40))
        xx = np.sort(rng.normal(0.0, 10.0, n))
        yy = rng.normal(0.0, 1e6, n)
        assert trapz(yy, xx) == float(_NUMPY_TRAPEZOID(yy, xx))


# ---------------------------------------------------------------------------------------------
# ZeroInitBDF: scipy's BDF reads one row of its difference array that it never wrote
# ---------------------------------------------------------------------------------------------
# scipy allocates D = np.empty((MAX_ORDER + 3, n)) and writes only D[0] and D[1]; on the first
# accepted step (order 1) `D[order + 2] = d - D[order + 1]` reads D[2] as it came from the heap.
# A subtract signals IEEE INVALID for a SIGNALLING-NaN operand, so ~1 recycled bit pattern in
# 4096 turns that line into "RuntimeWarning: invalid value encountered in subtract" -- a test
# failure under this repo's filterwarnings=["error"], and a ~9% flake of the numba CI leg
# (job 99382994280, 2026-08-31). These two gates pin the fix AND the hazard it removes.
_SNAN = np.array([0x7FF0000000000001], dtype=np.uint64).view(np.float64)[0]


def _decay_ivp():
    """A trivial stiff-ish linear decay -- enough state for BDF, no physics in the way."""
    k = np.array([1.0, 10.0, 100.0])
    return (lambda t, y: -k * y), 0.0, np.array([1.0, 1.0, 1.0]), 1.0


def test_zero_init_bdf_initialises_the_row_scipy_leaves_to_the_heap():
    from dynameta.core.numerics import ZeroInitBDF
    fun, t0, y0, tb = _decay_ivp()
    s = ZeroInitBDF(fun, t0, y0, tb)
    # also pins scipy's attribute name/shape: a rename upstream fails HERE, loudly, once.
    assert s.D.shape[0] > 2 and np.all(s.D[2:] == 0.0)


def test_uninitialised_bdf_row_signals_invalid_where_zero_init_does_not():
    from scipy.integrate import BDF

    from dynameta.core.numerics import ZeroInitBDF
    with warnings.catch_warnings(record=True) as probe:                 # platform capability
        warnings.simplefilter("always")
        _ = np.array([1.0]) - np.array([_SNAN])
    if not probe:
        pytest.skip("this platform does not signal IEEE INVALID on a signalling-NaN operand")
    fun, t0, y0, tb = _decay_ivp()
    bad = BDF(fun, t0, y0, tb)
    bad.D[2:] = _SNAN                                                   # what the heap can hand it
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        bad.step()
    assert any("invalid value encountered in subtract" in str(w.message) for w in rec), \
        "the pre-fix hazard no longer reproduces -- re-read scipy's bdf.py before deleting this"
    good = ZeroInitBDF(fun, t0, y0, tb)
    with warnings.catch_warnings(record=True) as rec2:
        warnings.simplefilter("always")
        good.step()
    assert not rec2, [str(w.message) for w in rec2]
    # and the fix is answer-preserving: the accepted step is bit-identical to scipy's own
    ref = BDF(fun, t0, y0, tb)
    ref.D[2:] = 0.0
    ref.step()
    assert np.array_equal(good.y, ref.y) and good.t == ref.t


def test_resolve_ivp_method_maps_only_bdf():
    from dynameta.core.numerics import ZeroInitBDF, resolve_ivp_method
    for spelling in ("BDF", "bdf", " Bdf "):
        assert resolve_ivp_method(spelling) is ZeroInitBDF
    for other in ("LSODA", "RK45", "Radau", "DOP853"):
        assert resolve_ivp_method(other) == other
    assert resolve_ivp_method(ZeroInitBDF) is ZeroInitBDF                # already a class
