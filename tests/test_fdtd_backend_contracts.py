"""FDTD 2-D contract gates from the 2026-07-25 audit: D-6 (the lateral_eps_inf pad contract),
D-7 (the backend-equality claims) and D-8 (the CUDA retry's device-state reset).

None of these needs a GPU: D-8's retry NEGOTIATION is a plain function of two callables
(_launch_with_retry), which is exactly why it was split out of the CUDA host driver -- the bug was
in the control flow, not in the kernel. Pure numpy elsewhere.
"""
import numpy as np
import pytest

from dynameta.optics.fdtd_nd import FDTDLayer, solve_fdtd_2d
from dynameta.optics.fdtd_nd.solve2d import _check_lateral_pads

LMIN, LMAX = 1200e-9, 1500e-9


# ---------------------------------------------------------------------------------------------
# D-6: `lateral_eps_inf` REPLACES THE WHOLE (nx, nz) grid -- pads included -- so the painter owns
# the n_super / n_sub pads. The comment used to say "applied in the structure region".
# ---------------------------------------------------------------------------------------------

def _grid(nx, nz, zc, pad, z_struct, e_struct, e_lo, e_hi):
    e = np.empty((nx, nz))
    e[:, zc < pad] = e_lo
    e[:, zc >= pad + z_struct] = e_hi
    band = (zc >= pad) & (zc < pad + z_struct)
    e[:, band] = e_struct
    return e


def test_d6_helper_accepts_correctly_painted_pads_and_rejects_forgotten_ones():
    nx, nz, pad, zs = 4, 60, 100e-9, 60e-9
    dz = (2 * pad + zs) / (nz - 1)
    zc = (np.arange(nz) + 0.5) * dz
    ok = _grid(nx, nz, zc, pad, zs, 4.0, 1.0, 1.5 ** 2)
    _check_lateral_pads("t", ok, zc, pad, zs, 1.0, 1.5)               # painted -> silent
    forgotten = _grid(nx, nz, zc, pad, zs, 4.0, 1.0, 1.0)            # the D-6 trap: ones everywhere
    with pytest.raises(ValueError, match="substrate"):
        _check_lateral_pads("t", forgotten, zc, pad, zs, 1.0, 1.5)
    flooded = np.full((nx, nz), 4.0)                                  # pattern over the WHOLE grid
    with pytest.raises(ValueError, match="superstrate"):
        _check_lateral_pads("t", flooded, zc, pad, zs, 1.0, 1.0)
    # and the default vacuum end media with vacuum pads stays silent (every in-repo painter)
    _check_lateral_pads("t", _grid(nx, nz, zc, pad, zs, 4.0, 1.0, 1.0), zc, pad, zs, 1.0, 1.0)


def test_d6_solve_fdtd_2d_refuses_a_painter_that_drops_the_substrate():
    """End to end through the public entry point. The painter follows the OLD comment ("applied in
    the structure region") and returns ones outside the structure; with n_sub=1.5 that silently
    discarded the substrate while the incident reference, the CPML match and T0's n_sub/n_super
    flux factor all still assumed it -- i.e. a mis-normalized R/T with no diagnostic."""
    def lat(nx, nz, zc, pad, zs):
        e = np.ones((nx, nz))
        e[:, (zc >= pad) & (zc < pad + zs)] = 4.0
        return e

    with pytest.raises(ValueError, match="REPLACES THE WHOLE"):
        solve_fdtd_2d([FDTDLayer(thickness_m=200e-9, eps_inf=4.0)], period_x_m=600e-9, nx=4,
                      lateral_eps_inf=lat, lambda_min_m=LMIN, lambda_max_m=LMAX,
                      resolution=8, n_pad_wave=3.0, n_sub=1.5)


def test_d6_correctly_painted_non_vacuum_pads_still_run_and_normalize():
    """The guard must not fire on the CORRECT painter, and the run it lets through must still be
    energy-consistent: a laterally-UNIFORM pattern painted with the end media is just the plain
    stack, so R+T = 1 on a lossless slab."""
    def lat(nx, nz, zc, pad, zs):
        return _grid(nx, nz, zc, pad, zs, 4.0, 1.0, 1.5 ** 2)

    r = solve_fdtd_2d([FDTDLayer(thickness_m=200e-9, eps_inf=4.0)], period_x_m=600e-9, nx=4,
                      lateral_eps_inf=lat, lambda_min_m=LMIN, lambda_max_m=LMAX,
                      resolution=8, n_pad_wave=3.0, n_sub=1.5)
    m = r.band
    assert np.max(np.abs(r.R0[m] + r.T0[m] - 1.0)) < 5e-2


# ---------------------------------------------------------------------------------------------
# D-7: the backend claims. "byte-for-byte equivalent" / "machine-precision identical" /
# "byte-exact" are arithmetically impossible as written (the numba kernels factor the E-update
# constant differently AND carry fastmath=True); the gates behind them are 1e-9 / 1e-12 / 1e-10.
# ---------------------------------------------------------------------------------------------

_FORBIDDEN = ("byte-for-byte", "byte-exact", "machine-precision identical", "byte identical")


def _doc_sources():
    import dynameta.optics.fdtd_nd as pkg
    from dynameta.optics.fdtd_nd import kernels2d_numba, solve2d
    return {
        "fdtd_nd/__init__ module docstring": pkg.__doc__,
        "solve_fdtd_2d docstring": solve2d.solve_fdtd_2d.__doc__,
        "_te2d_numba docstring": (kernels2d_numba._te2d_numba.__doc__
                                  or kernels2d_numba._te2d_numba.py_func.__doc__),
        "_te2d_cuda docstring": kernels2d_numba._te2d_cuda.__doc__,
    }


def test_d7_no_backend_docstring_claims_byte_identity():
    """A backend claim a user reads as 'bit-for-bit' is a claim the library cannot keep: numpy
    computes EPS0*eps_eff/dt where numba/CUDA compute e0dt*eps_inf, and fastmath=True licenses
    reassociation on top of that. audit D-7."""
    bad = []
    for where, doc in _doc_sources().items():
        low = (doc or "").lower()
        for phrase in _FORBIDDEN:
            if phrase in low:
                bad.append("{}: {!r}".format(where, phrase))
    assert not bad, "byte-identity claim(s) survive in the backend docs: {}".format(bad)


def test_d7_backend_docs_state_the_tolerance_and_cite_the_gate_that_measures_it():
    docs = _doc_sources()
    for where in ("fdtd_nd/__init__ module docstring", "solve_fdtd_2d docstring",
                  "_te2d_numba docstring"):
        doc = docs[where] or ""
        assert "validation/" in doc, "{} cites no validation gate".format(where)
        assert any(t in doc for t in ("1e-9", "1e-10", "1e-12")), \
            "{} states no tolerance".format(where)


def test_d7_numpy_and_numba_agree_to_the_gated_tolerance_not_to_the_last_bit():
    """The positive half of the claim, measured here rather than asserted in prose: on the fixture
    the docstrings point at, the two backends agree well inside 1e-9 -- and (the reason the wording
    changed) they are NOT required to be array-equal."""
    numba = pytest.importorskip("numba")                              # noqa: F841
    from dynameta.optics.fdtd_nd.backends import HAVE_NUMBA
    if not HAVE_NUMBA:
        pytest.skip("numba present but the kernels did not compile here")
    layers = [FDTDLayer(thickness_m=200e-9, eps_inf=4.0)]
    kw = dict(period_x_m=600e-9, nx=4, lambda_min_m=LMIN, lambda_max_m=LMAX, resolution=8,
              n_pad_wave=3.0)
    a = solve_fdtd_2d(layers, backend="numpy", **kw)
    b = solve_fdtd_2d(layers, backend="numba", **kw)
    m = a.band
    assert float(np.max(np.abs(a.R0[m] - b.R0[m]))) < 1e-9
    assert float(np.max(np.abs(a.T0[m] - b.T0[m]))) < 1e-9


# ---------------------------------------------------------------------------------------------
# D-8: the numba-CUDA launch-failure retry re-entered _launch at n0 = 0 without re-zeroing the
# PERSISTENT device field/ADE/CPML state.
# ---------------------------------------------------------------------------------------------

class _FakeDevice:
    """Minimal stand-in for the persistent device state: `march` accumulates steps, `reset` zeroes
    it, and a chosen attempt can fail after a chosen number of chunks (a WDDM TDR timeout on a
    LATER chunk -- the very hazard the chunking exists to avoid)."""

    def __init__(self, nsteps, chunk, fail_plan):
        self.nsteps, self.chunk = nsteps, chunk
        self.fail_plan = list(fail_plan)                  # per attempt: chunks to survive, or None
        self.attempt = -1
        self.steps_taken = 0                              # the "field state"
        self.resets = 0

    def reset(self):
        self.steps_taken = 0
        self.resets += 1

    def launch(self, blocks):
        self.attempt += 1
        survive = (self.fail_plan[self.attempt] if self.attempt < len(self.fail_plan) else None)
        n0, done = 0, 0
        while n0 < self.nsteps:
            if survive is not None and done == survive:
                raise RuntimeError("simulated launch/TDR failure on chunk {}".format(done))
            ns = min(self.chunk, self.nsteps - n0)
            self.steps_taken += ns
            n0 += ns
            done += 1


def test_d8_retry_after_a_late_chunk_failure_does_not_double_march():
    """The bug: a failure on the SECOND chunk left 4 steps already marched into the persistent
    device arrays, and the retry re-entered at n0 = 0, so the successful attempt reported 12 steps
    of evolution for a 8-step run -- silently wrong probes, no error."""
    from dynameta.optics.fdtd_nd.kernels2d_numba import _launch_with_retry
    dev = _FakeDevice(nsteps=8, chunk=4, fail_plan=[1])                # attempt 0 dies on chunk #1
    used = _launch_with_retry(dev.launch, dev.reset, blocks=16, sm=4, need=64)
    assert dev.steps_taken == 8, "device state carried {} steps for an 8-step run".format(
        dev.steps_taken)
    assert dev.attempt == 1 and dev.resets == 2 and used == 8           # halved 16 -> 8


def test_d8_state_is_reset_before_every_attempt_including_the_last_resort():
    """Block-count negotiation can take several attempts (16 -> 8 -> 4 -> the <=#SMs last resort);
    each one must start from a zeroed state, so resets == attempts."""
    from dynameta.optics.fdtd_nd.kernels2d_numba import _launch_with_retry
    dev = _FakeDevice(nsteps=12, chunk=3, fail_plan=[0, 2, 1])          # fail at launch, late, late
    used = _launch_with_retry(dev.launch, dev.reset, blocks=16, sm=4, need=64)
    assert dev.steps_taken == 12
    assert dev.attempt + 1 == dev.resets == 4                           # 16, 8, 4, then <=#SMs
    assert used == 4


def test_d8_first_chunk_failure_still_takes_the_intended_shrink_path():
    """The INTENDED case (a cooperative grid too large to be co-resident fails at launch, before
    any chunk marches) is unchanged: shrink and re-run, exactly one full march of state."""
    from dynameta.optics.fdtd_nd.kernels2d_numba import _launch_with_retry
    dev = _FakeDevice(nsteps=10, chunk=5, fail_plan=[0])
    used = _launch_with_retry(dev.launch, dev.reset, blocks=32, sm=8, need=128)
    assert dev.steps_taken == 10 and used == 16 and dev.attempt == 1


def test_d8_a_persistent_failure_still_propagates():
    """The retry must not swallow a real, repeatable failure: once it is down to the <=#SMs last
    resort the exception surfaces."""
    from dynameta.optics.fdtd_nd.kernels2d_numba import _launch_with_retry
    dev = _FakeDevice(nsteps=6, chunk=3, fail_plan=[0, 0, 0, 0, 0, 0])
    with pytest.raises(RuntimeError, match="simulated launch"):
        _launch_with_retry(dev.launch, dev.reset, blocks=8, sm=4, need=32)


def test_d8_host_driver_routes_through_the_shared_retry_helper():
    """Static pin: _te2d_cuda must use _launch_with_retry (with a reset callable) rather than
    re-rolling the loop -- the code path a box without a CUDA device cannot execute."""
    import ast
    import inspect
    from dynameta.optics.fdtd_nd import kernels2d_numba
    src = inspect.getsource(kernels2d_numba._te2d_cuda)
    tree = ast.parse(src.lstrip())
    calls = [n.func.id for n in ast.walk(tree)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)]
    assert "_launch_with_retry" in calls
    names = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    assert {"_launch", "_reset"} <= names
