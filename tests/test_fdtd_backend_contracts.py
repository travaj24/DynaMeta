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


# =====================================================================================================
# wave-6 (2026-11-02): X-16, V-14, D-15, P-7, D-13
# =====================================================================================================

@pytest.mark.parametrize("name", ["run_2d_te_jax", "run_2d_te_oblique_jax", "run_2d_tm_oblique_jax",
                                  "run_3d_jax", "run_3d_oblique_jax"])
def test_x16_jax_kernels_refuse_a_src_that_disagrees_with_nsteps(name):
    """audit X-16: lax.scan marches len(src) steps and `nsteps` was silently ignored. The guard
    fires before any jax import, so this runs without jax too."""
    import dynameta.optics.fdtd_nd as F
    fn = getattr(F, name)
    nz = 20
    two_d = name.startswith("run_2d")
    shape = (2, nz) if two_d else (2, 2, nz)
    a = np.ones(shape)
    cp = F.cpml_z(nz, 1e-8, 1e-17, 4)
    common = (a, 0 * a, 0 * a) if "oblique" in name else (a, 0 * a, 0 * a, 0 * a)
    grid = (1e-8, 1e-8) if two_d else (1e-8, 1e-8, 1e-8)
    tail = {"run_2d_te_oblique_jax": (0.0,), "run_2d_tm_oblique_jax": (0.0,),
            "run_3d_oblique_jax": (0.0, 0.0, 0.0, 1.0)}.get(name, ())
    with pytest.raises(ValueError, match="X-16"):
        fn(*common, *grid, 1e-17, 7, 5, 6, 14, np.zeros(9), cp, *tail)


def test_v14_importing_the_fdtd_family_does_not_import_numba():
    """audit V-14: fdtd_nd imported numba (+ numba.cuda) eagerly -- measured ~0.7 s of the
    1.4-1.5 s `import dynameta.optics.fdtd_nd` (now 0.66-0.79 s). The kernels JIT lazily, and the
    public HAVE_NUMBA / prange still resolve (on first access)."""
    import subprocess
    import sys
    code = ("import sys, dynameta.optics.fdtd_nd as F, dynameta.optics.fdtd, dynameta.optics.fdtd_mo\n"
            "print('numba' in sys.modules)\n"
            "h = F.HAVE_NUMBA\n"
            "print(h == ('numba' in sys.modules))\n")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    assert out.stdout.split() == ["False", "True"], out.stdout


def test_v14_importing_optics_soa_does_not_import_numba():
    """audit V-14, the soa half (wave-6 H6b, 2026-11-02): soa/qd_gain.py imported numba at module
    load for the opt-in QDGainModel(fast=True) kernels -- `import dynameta.optics.soa` measured
    1.06-1.11 s with numba loaded, 0.74 s without (numba 0.34 s in -X importtime). The two kernels
    now JIT on first call with the same njit options (bit-identical step_slices output vs the
    eager module, ng = 1 and 41), and the private `_HAVE_NUMBA` the tests import still resolves.
    Lives here beside its fdtd_nd sibling above. FAILS on the eager import (first line 'True')."""
    import subprocess
    import sys
    code = ("import sys, dynameta.optics.soa\n"
            "print('numba' in sys.modules)\n"
            "from dynameta.optics.soa import qd_gain as Q\n"
            "print(type(Q._qd_carrier_rk4_numba).__name__, type(Q._qd_carrier_rk4_eh_numba).__name__)\n"
            "m = Q.QDGainModel(Q.QDGainParams())\n"
            "print('numba' in sys.modules)\n"
            "h = Q._HAVE_NUMBA\n"
            "print(h == ('numba' in sys.modules))\n")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    # import AND a default (fast=False) model build stay numba-free; the flag resolves on demand
    assert out.stdout.split() == ["False", "_LazyJit", "_LazyJit", "False", "True"], out.stdout


def test_v14_lazy_njit_kernel_keeps_its_docstring_and_compiles_on_call():
    from dynameta.optics.fdtd_nd import backends, kernels2d_numba
    k = kernels2d_numba._te2d_numba
    assert isinstance(k, backends._LazyJit) and "Numba CPU kernel" in k.__doc__
    assert k.py_func.__name__ == "_te2d_numba"


def test_d15_source_probe_and_grid_blocks_are_single_sourced():
    """audit D-15 census: the source envelope, the pad-fraction probe placement and the z-grid
    sizing now live ONLY in fdtd_nd/grid.py; a re-typed copy in any front end fails here."""
    import pathlib
    import dynameta.optics as O
    root = pathlib.Path(O.__file__).parent
    files = [root / "fdtd.py", root / "fdtd_mo.py"] + sorted((root / "fdtd_nd").glob("*.py"))
    # each group = one block, in every spelling the front ends had drifted into. (The NONUNIFORM
    # 1-D branch places its probes by searchsorted on z_edges -- a different rule, deliberately.)
    groups = (("np.exp(-((tgrid - t0) / tau) ** 2)",),
              ("int(round((0.35 * pad) / dz))", "int(round(0.35 * pad / dz))"),
              ("int(round((0.7 * pad) / dz))", "int(round(0.7 * pad / dz))"),
              ("int(round((pad + z_struct + 0.3 * pad) / dz))",),
              ("Lz = 2.0 * pad + z_struct",), ("/ (resolution * n_max)",),
              ("1.0 / (np.pi * (f_max - f_min))",))
    for grp in groups:
        hits = [f.name for f in files for line in f.read_text(encoding="utf-8").splitlines()
                if any(p in line for p in grp) and not line.lstrip().startswith("#")]
        assert hits == ["grid.py"], (grp, hits)


def _te_oblique_pre_p7(eps_inf, wp, gam, dx, dz, dt, nsteps, k_src, k_pL, k_pR, src, cpml, kx):
    # VERBATIM loop of oblique2d._run_2d_te_oblique before audit P-7 (2026-11-02)
    from dynameta.constants import EPS0, MU0
    nx, nz = eps_inf.shape
    (ke, be, ce), (kh, bh, ch) = cpml
    z = (lambda: np.zeros((nx, nz), dtype=complex))
    Ey, Hx, Hz, Jy, psi_hxz, psi_eyz = z(), z(), z(), z(), z(), z()
    aJ = (1.0 - gam * dt / 2.0) / (1.0 + gam * dt / 2.0)
    bJ = (EPS0 * wp ** 2 * dt / 2.0) / (1.0 + gam * dt / 2.0)
    eyL = np.empty((nsteps, nx), complex); hxL = np.empty((nsteps, nx), complex)
    eyR = np.empty((nsteps, nx), complex); hxR = np.empty((nsteps, nx), complex)
    cmu = dt / MU0; ikx = 1j * kx
    for n in range(nsteps):
        dEy_dz = (Ey[:, 1:] - Ey[:, :-1]) / dz
        psi_hxz[:, :-1] = bh[:-1] * psi_hxz[:, :-1] + ch[:-1] * dEy_dz
        Hx[:, :-1] += cmu * (dEy_dz / kh[:-1] + psi_hxz[:, :-1])
        Hz += -cmu * ((np.roll(Ey, -1, axis=0) - Ey) / dx + ikx * Ey)
        dHx_dz = (Hx[:, 1:] - Hx[:, :-1]) / dz
        psi_eyz[:, 1:] = be[1:] * psi_eyz[:, 1:] + ce[1:] * dHx_dz
        curl = np.zeros((nx, nz), complex)
        curl[:, 1:] += dHx_dz / ke[1:] + psi_eyz[:, 1:]
        curl -= (Hz - np.roll(Hz, 1, axis=0)) / dx + ikx * Hz
        denom = EPS0 * eps_inf / dt + bJ / 2.0
        Eynew = (EPS0 * eps_inf / dt * Ey + curl - 0.5 * (1.0 + aJ) * Jy - 0.5 * bJ * Ey) / denom
        Jy = aJ * Jy + bJ * (Eynew + Ey)
        Eynew[:, k_src] += src[n]
        Eynew[:, 0] = 0.0; Eynew[:, -1] = 0.0
        Ey = Eynew
        eyL[n] = Ey[:, k_pL]; hxL[n] = 0.5 * (Hx[:, k_pL] + Hx[:, k_pL - 1])
        eyR[n] = Ey[:, k_pR]; hxR[n] = 0.5 * (Hx[:, k_pR] + Hx[:, k_pR - 1])
    return eyL, hxL, eyR, hxR


def _tm_oblique_pre_p7(eps_inf, wp, gam, dx, dz, dt, nsteps, k_src, k_pL, k_pR, src, cpml, kx):
    # VERBATIM loop of oblique2d._run_2d_tm_oblique before audit P-7 (2026-11-02)
    from dynameta.constants import EPS0, MU0
    nx, nz = eps_inf.shape
    (ke, be, ce), (kh, bh, ch) = cpml
    z = (lambda: np.zeros((nx, nz), dtype=complex))
    Ex, Ez, Hy, Jx, Jz, psi_hyz, psi_exz = z(), z(), z(), z(), z(), z(), z()
    aJ = (1.0 - gam * dt / 2.0) / (1.0 + gam * dt / 2.0)
    bJ = (EPS0 * wp ** 2 * dt / 2.0) / (1.0 + gam * dt / 2.0)
    exL = np.empty((nsteps, nx), complex); hyL = np.empty((nsteps, nx), complex)
    exR = np.empty((nsteps, nx), complex); hyR = np.empty((nsteps, nx), complex)
    cmu = dt / MU0; ikx = 1j * kx; e0dt = EPS0 / dt
    for n in range(nsteps):
        dEx_dz = (Ex[:, 1:] - Ex[:, :-1]) / dz
        psi_hyz[:, :-1] = bh[:-1] * psi_hyz[:, :-1] + ch[:-1] * dEx_dz
        dEz_dx = (np.roll(Ez, -1, axis=0) - Ez) / dx + ikx * Ez
        Hy[:, :-1] += cmu * (dEz_dx[:, :-1] - (dEx_dz / kh[:-1] + psi_hyz[:, :-1]))
        dHy_dz = (Hy[:, 1:] - Hy[:, :-1]) / dz
        psi_exz[:, 1:] = be[1:] * psi_exz[:, 1:] + ce[1:] * dHy_dz
        curlx = np.zeros((nx, nz), complex)
        curlx[:, 1:] = -(dHy_dz / ke[1:] + psi_exz[:, 1:])
        denom = e0dt * eps_inf + bJ / 2.0
        Exn = (e0dt * eps_inf * Ex + curlx - 0.5 * (1.0 + aJ) * Jx - 0.5 * bJ * Ex) / denom
        Jx = aJ * Jx + bJ * (Exn + Ex)
        Exn[:, k_src] += src[n]
        Exn[:, 0] = 0.0; Exn[:, -1] = 0.0
        Ex = Exn
        curlz = (Hy - np.roll(Hy, 1, axis=0)) / dx + ikx * Hy
        Ezn = (e0dt * eps_inf * Ez + curlz - 0.5 * (1.0 + aJ) * Jz - 0.5 * bJ * Ez) / denom
        Jz = aJ * Jz + bJ * (Ezn + Ez)
        Ez = Ezn
        exL[n] = Ex[:, k_pL]; hyL[n] = 0.5 * (Hy[:, k_pL] + Hy[:, k_pL - 1])
        exR[n] = Ex[:, k_pR]; hyR[n] = 0.5 * (Hy[:, k_pR] + Hy[:, k_pR - 1])
    return exL, hyL, exR, hyR


@pytest.mark.parametrize("drude", [False, True])
def test_p7_oblique_numpy_kernels_bit_identical_to_the_pre_hoist_loop(drude):
    """audit P-7: the hoisted oblique NumPy kernels (TE and TM, passive and Drude) reproduce the
    pre-hoist loop above BIT-FOR-BIT (tobytes)."""
    from dynameta.optics.fdtd_nd import cpml_z
    from dynameta.optics.fdtd_nd import oblique2d as OB
    nx, nz, ns = 4, 60, 300
    eps = np.ones((nx, nz)); eps[:, 25:35] = 3.0
    wp = np.zeros((nx, nz)); gm = np.zeros((nx, nz))
    if drude:
        wp[:, 25:35] = 1.5e15; gm[:, 25:35] = 1e14
    dz, dx = 20e-9, 100e-9
    dt = 0.5 / (2.998e8 * np.sqrt(1 / dx ** 2 + 1 / dz ** 2))
    cp = cpml_z(nz, dz, dt, 8)
    src = np.sin(np.arange(ns) * 0.2) * np.exp(-((np.arange(ns) - 60) / 20.0) ** 2)
    args = (eps, wp, gm, dx, dz, dt, ns, 10, 15, 45, src, cp, 2e6)
    for new, old in ((OB._run_2d_te_oblique, _te_oblique_pre_p7),
                     (OB._run_2d_tm_oblique, _tm_oblique_pre_p7)):
        for a, b in zip(new(*args), old(*args)):
            assert a.tobytes() == b.tobytes()


def test_d13_cpml_is_plain_fdtd_off_the_pml_because_c_is_zero_not_because_b_is_one():
    """audit D-13 (docstring): off the PML alpha = alpha_max, so b != 1 there; c == 0 EXACTLY is
    what keeps the interior plain FDTD. Pins the corrected statement."""
    from dynameta.optics.fdtd_nd import cpml_z
    dz = 20e-9
    (ke, be, ce), (kh, bh, ch) = cpml_z(100, dz, 0.5 * dz / 2.998e8, 12)
    mid = slice(13, -13)
    assert np.all(ce[mid] == 0.0) and np.all(ch[mid] == 0.0) and np.all(ke[mid] == 1.0)
    assert np.all(be[mid] < 1.0) and float(1.0 - be[50]) == pytest.approx(7.5e-7, rel=0.01)


# ---------------------------------------------------------------------------------------------
# S2-11 / S6-12 / S6-11 (wave-7, 2026-11-02): the 2-D reference kernel, the 3-D magneto-optic kernel
# and the 3-D oblique kernel got run_3d's prealloc / roll-free out= rewrite. They must be
# BIT-IDENTICAL to the pre-rewrite kernels (7b9481a). The REAL-valued kernels are pinned by sha256
# digests of their outputs on fixtures built from + - * / only (no exp/sin/pow: `p*p*p`, never
# `p**3`), so the bytes are platform-independent (IEEE-exact ops; numpy never fuses two ufuncs).
# The digests were produced by the 7b9481a kernels and re-produced by the rewritten ones. The
# COMPLEX oblique kernel is compared against its pre-rewrite loop kept verbatim below (the P-7
# pattern): numpy's SIMD complex multiply may use FMA on one CPU and not another, so a digest of
# complex results would not be portable.
# ---------------------------------------------------------------------------------------------

def _sha(arrs):
    import hashlib
    h = hashlib.sha256()
    for a in arrs:
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()[:16]


def _cpml_poly(nz, npml):
    k = np.arange(nz, dtype=float)
    prof = np.zeros(nz)
    u = (npml - k[:npml]) / npml
    prof[:npml] = u * u * u
    v = (k[-npml:] - (nz - 1 - npml)) / npml
    prof[-npml:] = v * v * v
    kap, be, bh = 1.0 + 4.0 * prof, 1.0 - 0.6 * prof, 1.0 - 0.55 * prof
    return (kap, be, 0.9 * (be - 1.0) / kap), (kap, bh, 0.9 * (bh - 1.0) / kap)


def _bump(nsteps, t0, w):
    x = (np.arange(nsteps, dtype=float) - t0) / w
    q = 1.0 - x * x
    return np.where(np.abs(x) < 1.0, x * q * q * q, 0.0)      # zero-mean: no DC charge-up


def _k2d_cases():
    EPS0 = 8.8541878128e-12
    nx, nz, dz, dx = 6, 90, 12e-9, 40e-9
    dt = 0.45 * dz / 3.0e8
    cp = _cpml_poly(nz, 10)
    src = _bump(700, 60.0, 18.0)
    ii, kk = np.meshgrid(np.arange(nx), np.arange(nz), indexing="ij")
    band = (kk >= 40) & (kk < 55)
    win = band.astype(float)
    eps = np.where(band, 2.25 + 0.5 * (ii % 2), 1.0)
    zero = np.zeros((nx, nz))
    wp, gam = np.where(band, 1.2e15, 0.0), np.where(band, 8e13, 0.0)
    chi3 = np.where(band, 0.3, 0.0)
    w0 = 1.6e15
    den = 1.0 + 5e13 * dt / 2.0
    lor = (np.where(band, (2.0 - w0 * w0 * dt * dt) / den, 2.0), np.where(band, (5e13 * dt / 2.0 - 1.0) / den, -1.0),
           np.where(band, EPS0 * 0.8 * w0 * w0 * dt * dt / den, 0.0))
    gain = (lor[0], lor[1], np.where(band, -1e-8 * 1e22 * dt * dt / den, 0.0))
    raman = (lor[0], lor[1], np.where(band, w0 * w0 * dt * dt / den, 0.0), np.where(band, 1e-20, 0.0))
    gdyn = (lor[0], lor[1], 1e-8 * dt * dt / den * win, 2e11 * win,
            np.stack([np.where(band, 6e24, 1e25), win * 1e24, win * 2e24, win * 1e24]),
            4e-7, 2.3e-4, 1e-8, 1.0545718e-34 * 1.6e15, 300)
    Te = np.linspace(300.0, 5000.0, 50)
    tab = (Te, 35.0 * (Te * Te - 300.0 * 300.0), 1.0 - 1e-5 * (Te - 300.0), 1.0 + 2e-4 * (Te - 300.0))
    hot = dict(mask=win, mat_idx=np.where(band, 0, -1), tables=[tab], G=win * 1e17,
               Tl=np.full((nx, nz), 300.0), alpha=win, Te0=np.full((nx, nz), 300.0), n_update=3)
    hot_lat = dict(hot, c_l=np.where(band, 2.5e6, np.inf), g_sub=win * 1e9)
    base, drude = (eps, zero, zero, zero), (eps, wp, gam, zero)
    cases = {"passive": (base, {}), "drude": (drude, {}), "kerr": ((eps, zero, zero, chi3), {}),
             "drude_kerr": ((eps, wp, gam, chi3), {}), "lorentz": (base, dict(lor=lor)),
             "drude_lorentz_gain": (drude, dict(lor=lor, gain=gain)),
             "chi2_raman": (base, dict(chi2=np.where(band, 1e-12, 0.0), raman=raman)),
             "gain_dyn": (base, dict(gain_dyn=gdyn)), "hot": (drude, dict(hot=hot)),
             "hot_lattice": (drude, dict(hot=hot_lat))}
    return cases, (dx, dz, dt, src.size, 12, 20, 75, src, cp)


# sha256[:16] of every output of kernels2d.run_2d_te at 7b9481a (pre-S2-11) on _k2d_cases()
_K2D_PRE_S2_11 = {
    "passive": "7bb880c1aa4889bc", "drude": "f6d244a95b04bd54", "kerr": "9cbd5daef9a910a4", "drude_kerr": "408eb318b2caf306",
    "lorentz": "7033020b33ac4f16", "drude_lorentz_gain": "1455a2c1895fdbed", "chi2_raman": "9f8a99da2106d889",
    "gain_dyn": "3da0ac6d28fb876c", "hot": "723003ecc5fd0f5c", "hot_lattice": "599663423638e796",
}


def _run_k2d(name):
    from dynameta.optics.fdtd_nd.kernels2d import run_2d_te
    cases, (dx, dz, dt, ns, ks, kl, kr, src, cp) = _k2d_cases()
    mats, kw = cases[name]
    kw = dict(kw)
    side = {}
    if "gain_dyn" in kw:
        kw["gain_dyn_out"] = side
    if "hot" in kw:
        kw["hot_out"] = side
    out = list(run_2d_te(*[np.array(m, copy=True) for m in mats], dx, dz, dt, ns, ks, kl, kr, src, cp, **kw))
    return out + [np.asarray(side[k]) for k in sorted(side)]


@pytest.mark.parametrize("name", sorted(_K2D_PRE_S2_11))
def test_s2_11_2d_reference_kernel_prealloc_is_bit_identical(name):
    """audit S2-11 / S6-12: run_2d_te's per-step temporaries became out= chains into preallocated
    scratch + Ey/Jy ping-pong. Every path (passive, Drude, Kerr, Lorentz, gain, chi2 + Raman,
    dynamic four-level gain, hot carrier fixed-bath and finite-lattice) is BIT-IDENTICAL to the
    pre-rewrite kernel. Measured: 1.0x at 64 x 400 (small temporaries are recycled), 2.8-3.4x at
    256 x 1600 and 2.1-2.3x at 1024 x 2000 (fresh page-faulted grids every step before)."""
    out = _run_k2d(name)
    assert all(np.all(np.isfinite(a)) for a in out[:4])
    assert float(np.max(np.abs(out[0]))) < 1.0                 # a sane, decaying fixture
    assert _sha(out) == _K2D_PRE_S2_11[name], name


def _k3d_grid(nx=4, ny=3, nz=60, nsteps=300):
    dz, dx, dy = 15e-9, 50e-9, 60e-9
    dt = 0.4 * dz / 3.0e8
    ii, jj, kk = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij")
    band = (kk >= 26) & (kk < 36)
    return dict(dx=dx, dy=dy, dz=dz, dt=dt, cp=_cpml_poly(nz, 8), src=_bump(nsteps, 50.0, 16.0),
                band=band, ii=ii, jj=jj, nsteps=nsteps)


def _k3d_mo_cases(g):
    b = g["band"]
    exx = np.where(b, 2.25 + 0.25 * (g["ii"] % 2), 1.0)
    eyy = np.where(b, 2.0 + 0.5 * (g["jj"] % 2), 1.0)
    ezz = np.where(b, 2.1, 1.0)
    wp, gam, wc = np.where(b, 1.1e15, 0.0), np.where(b, 9e13, 0.0), np.where(b, 3e14, 0.0)
    z = np.zeros_like(wp)
    return {"mo_gyro_y": (exx, eyy, ezz, wp, gam, wc, "y"), "mo_gyro_x": (exx, eyy, ezz, wp, gam, wc, "x"),
            "mo_aniso_nodrude": (exx, eyy, ezz, z, z, z, "y")}


# sha256[:16] of kernels3d._run_3d_mo's four outputs at 7b9481a (pre-S6-11) on _k3d_mo_cases()
_K3D_MO_PRE_S6_11 = {"mo_gyro_y": "e1e6d30b49e62dc2", "mo_gyro_x": "81fc52a739417fa4", "mo_aniso_nodrude": "810eda4dbaabf415"}


def _oblique_pre_s6_11(eps_inf, wp, gam, dx, dy, dz, dt, nsteps, k_src, k_pL, k_pR, src, cpml, kx, ky, sx, sy):
    """kernels3d._run_3d_oblique as it stood at 7b9481a (before the S6-11 rewrite), verbatim."""
    from dynameta.constants import EPS0, MU0
    nx, ny, nz = eps_inf.shape
    (ke, be, ce), (kh, bh, ch) = cpml
    r = (lambda a: np.asarray(a).reshape(1, 1, nz))
    ke, be, ce = r(ke), r(be), r(ce)
    kh, bh, ch = r(kh), r(bh), r(ch)
    z3 = (lambda: np.zeros((nx, ny, nz), dtype=complex))
    Ex, Ey, Ez = z3(), z3(), z3()
    Hx, Hy, Hz = z3(), z3(), z3()
    Jx, Jy, Jz = z3(), z3(), z3()
    psi_Hx, psi_Hy, psi_Ex, psi_Ey = z3(), z3(), z3(), z3()
    sEy_dz, sEx_dz, sHy_dz, sHx_dz = z3(), z3(), z3(), z3()
    aJ = (1.0 - gam * dt / 2.0) / (1.0 + gam * dt / 2.0)
    bJ = (EPS0 * wp ** 2 * dt / 2.0) / (1.0 + gam * dt / 2.0)
    cmu = dt / MU0; e0dt = EPS0 / dt
    ikx, iky = 1j * kx, 1j * ky
    dxf = (lambda F: (np.roll(F, -1, axis=0) - F) / dx + ikx * F)
    dyf = (lambda F: (np.roll(F, -1, axis=1) - F) / dy + iky * F)
    dxb = (lambda F: (F - np.roll(F, 1, axis=0)) / dx + ikx * F)
    dyb = (lambda F: (F - np.roll(F, 1, axis=1)) / dy + iky * F)
    sh = (nsteps, nx, ny)
    exL, eyL, exR, eyR = np.empty(sh, complex), np.empty(sh, complex), np.empty(sh, complex), np.empty(sh, complex)
    for n in range(nsteps):
        dEy_dz = (Ey[:, :, 1:] - Ey[:, :, :-1]) / dz
        psi_Hx[:, :, :-1] = bh[:, :, :-1] * psi_Hx[:, :, :-1] + ch[:, :, :-1] * dEy_dz
        sEy_dz[:, :, :-1] = dEy_dz / kh[:, :, :-1] + psi_Hx[:, :, :-1]
        Hx -= cmu * (dyf(Ez) - sEy_dz)
        dEx_dz = (Ex[:, :, 1:] - Ex[:, :, :-1]) / dz
        psi_Hy[:, :, :-1] = bh[:, :, :-1] * psi_Hy[:, :, :-1] + ch[:, :, :-1] * dEx_dz
        sEx_dz[:, :, :-1] = dEx_dz / kh[:, :, :-1] + psi_Hy[:, :, :-1]
        Hy -= cmu * (sEx_dz - dxf(Ez))
        Hz -= cmu * (dxf(Ey) - dyf(Ex))
        denom = e0dt * eps_inf + bJ / 2.0
        dHy_dz = (Hy[:, :, 1:] - Hy[:, :, :-1]) / dz
        psi_Ex[:, :, 1:] = be[:, :, 1:] * psi_Ex[:, :, 1:] + ce[:, :, 1:] * dHy_dz
        sHy_dz[:, :, 1:] = dHy_dz / ke[:, :, 1:] + psi_Ex[:, :, 1:]
        curlx = dyb(Hz) - sHy_dz
        Exn = (e0dt * eps_inf * Ex + curlx - 0.5 * (1.0 + aJ) * Jx - 0.5 * bJ * Ex) / denom
        Jx = aJ * Jx + bJ * (Exn + Ex)
        dHx_dz = (Hx[:, :, 1:] - Hx[:, :, :-1]) / dz
        psi_Ey[:, :, 1:] = be[:, :, 1:] * psi_Ey[:, :, 1:] + ce[:, :, 1:] * dHx_dz
        sHx_dz[:, :, 1:] = dHx_dz / ke[:, :, 1:] + psi_Ey[:, :, 1:]
        curly = sHx_dz - dxb(Hz)
        Eyn = (e0dt * eps_inf * Ey + curly - 0.5 * (1.0 + aJ) * Jy - 0.5 * bJ * Ey) / denom
        Jy = aJ * Jy + bJ * (Eyn + Ey)
        curlz = dxb(Hy) - dyb(Hx)
        Ezn = (e0dt * eps_inf * Ez + curlz - 0.5 * (1.0 + aJ) * Jz - 0.5 * bJ * Ez) / denom
        Jz = aJ * Jz + bJ * (Ezn + Ez)
        Exn[:, :, k_src] += sx * src[n]; Eyn[:, :, k_src] += sy * src[n]
        for F in (Exn, Eyn):
            F[:, :, 0] = 0.0; F[:, :, -1] = 0.0
        Ex, Ey, Ez = Exn, Eyn, Ezn
        exL[n] = Ex[:, :, k_pL]; eyL[n] = Ey[:, :, k_pL]
        exR[n] = Ex[:, :, k_pR]; eyR[n] = Ey[:, :, k_pR]
    return exL, eyL, exR, eyR


@pytest.mark.parametrize("name", sorted(_K3D_MO_PRE_S6_11) + ["obl_normal", "obl_kx_drude", "obl_conical"])
def test_s6_11_3d_mo_and_oblique_prealloc_is_bit_identical(name):
    """audit S6-11: _run_3d_mo (the only MO kernel) and the numpy _run_3d_oblique never got run_3d's
    prealloc / roll-free rewrite (~15-35 temporaries + 7-10 np.roll copies per step). Now they have
    it, BIT-FOR-BIT: the MO kernel against its pre-rewrite digests (gyrotropic x / y source,
    anisotropic Drude-free), the oblique kernel against its verbatim pre-rewrite loop (normal,
    oblique Drude, conical)."""
    from dynameta.optics.fdtd_nd import kernels3d as K
    g = _k3d_grid()
    args = (g["dx"], g["dy"], g["dz"], g["dt"], g["nsteps"], 10, 14, 48, g["src"], g["cp"])
    if name.startswith("mo_"):
        exx, eyy, ezz, wp, gam, wc, pol = _k3d_mo_cases(g)[name]
        out = K._run_3d_mo(exx, eyy, ezz, wp, gam, wc, *args, pol)
        assert all(np.all(np.isfinite(a)) for a in out) and float(np.max(np.abs(out[0][-30:]))) < 0.1
        assert _sha(out) == _K3D_MO_PRE_S6_11[name], name
        return
    b = g["band"]
    eps = np.where(b, 2.25 + 0.5 * (g["ii"] % 2), 1.0)
    wp, gam = np.where(b, 1.1e15, 0.0), np.where(b, 9e13, 0.0)
    kx, ky, sx, sy = {"obl_normal": (0.0, 0.0, 0.0, 1.0), "obl_kx_drude": (2.1e6, 0.0, 0.0, 1.0),
                      "obl_conical": (1.3e6, 0.7e6, -0.5, 0.8)}[name]
    if name == "obl_normal":
        wp, gam = 0.0 * wp, 0.0 * gam
    new = K._run_3d_oblique(eps, wp, gam, *args, kx, ky, sx, sy)
    old = _oblique_pre_s6_11(eps, wp, gam, *args, kx, ky, sx, sy)
    for a, o in zip(new, old):
        assert a.tobytes() == o.tobytes()


@pytest.mark.parametrize("kxy", [(0.0, 0.0), (1.3e6, 0.7e6)])
def test_s6_10_numba_oblique_banded_cpml(kxy):
    """audit S6-10 (wave-7, 2026-11-02): _run_3d_oblique_numba stores its four CPML psi grids
    BANDED (one slot per c != 0 z-plane + one shared, forever-zero slot) instead of full-field: 210
    -> 6.6 MB of complex128 at 64x64x800, and 2.78 -> ~2.2 s there (less memory traffic). Under
    fastmath it is not bitwise the full-field kernel (measured 1.9e-16 on |E| ~ 0.94), so it is
    gated where the numba kernels are contracted: against the numpy reference (audit D-7), here
    with cpml_z's REAL profile (c == -0.0 off the slabs) on a non-vacuum substrate (an asymmetric
    pair of slabs). Measured agreement ~1e-15 relative; the gate allows 1e-12."""
    pytest.importorskip("numba")
    from dynameta.optics.fdtd_nd.backends import HAVE_NUMBA
    if not HAVE_NUMBA:
        pytest.skip("numba present but the kernels did not compile here")
    from dynameta.optics.fdtd_nd import cpml_z
    from dynameta.optics.fdtd_nd import kernels3d as K
    from dynameta.optics.fdtd_nd.kernels3d_numba import _run_3d_oblique_numba
    nx, ny, nz, ns = 4, 3, 80, 300
    dz, dx, dy = 15e-9, 50e-9, 60e-9
    dt = 0.4 * dz / 3.0e8
    cp = cpml_z(nz, dz, dt, 10, 1.0, 1.5)
    t = np.arange(ns)
    src = np.sin(0.21 * t) * np.exp(-((t - 60) / 20.0) ** 2)
    kk = np.broadcast_to(np.arange(nz), (nx, ny, nz))
    band = (kk >= 34) & (kk < 46)
    eps = np.where(band, 2.25, 1.0); eps[kk >= 46] = 2.25
    wp, gm = np.where(band, 1.1e15, 0.0), np.where(band, 9e13, 0.0)
    kx, ky = kxy
    ref = K._run_3d_oblique(eps, wp, gm, dx, dy, dz, dt, ns, 10, 16, 64, src, cp, kx, ky, -0.5, 0.8)
    (ke, be, ce), (kh, bh, ch) = cp
    got = _run_3d_oblique_numba(eps, wp, gm, ke, be, ce, kh, bh, ch, dx, dy, dz, dt, ns, 10, 16, 64,
                                src, kx, ky, -0.5, 0.8)
    scale = max(float(np.max(np.abs(a))) for a in ref)
    assert scale > 0.1
    for a, b in zip(ref, got):
        assert float(np.max(np.abs(a - b))) < 1e-12 * scale


def _s2_12_fixture():
    from dynameta.constants import EPS0
    from dynameta.optics.fdtd_nd import cpml_z
    nx, ny, nz, ns = 3, 3, 64, 400
    dz, dx, dy = 15e-9, 60e-9, 60e-9
    dt = 0.4 * dz / 3e8
    cp = cpml_z(nz, dz, dt, 10)
    t = np.arange(ns)
    src = np.sin(0.25 * t) * np.exp(-((t - 70) / 22.0) ** 2)
    kk = np.broadcast_to(np.arange(nz), (nx, ny, nz))
    b = (kk >= 28) & (kk < 38)
    ii = np.broadcast_to(np.arange(nx)[:, None, None], (nx, ny, nz))
    eps = np.where(b, 2.25 + 0.5 * (ii % 2), 1.0)
    z = np.zeros((nx, ny, nz))
    wp, gm = np.where(b, 1e15, 0.0), np.where(b, 1e14, 0.0)
    w0 = 1.6e15
    den = 1 + 5e13 * dt / 2
    lor = (np.where(b, (2 - w0 ** 2 * dt ** 2) / den, 2.0), np.where(b, (5e13 * dt / 2 - 1) / den, -1.0),
           np.where(b, EPS0 * 0.8 * w0 ** 2 * dt ** 2 / den, 0.0))
    terms = {"lorentz": dict(lor=lor), "gain": dict(gain=(lor[0], lor[1], -0.3 * lor[2])),
             "chi2": dict(chi2=np.where(b, 1e-2, 0.0)),
             "raman": dict(raman=(lor[0], lor[1], np.where(b, w0 ** 2 * dt ** 2 / den, 0.0),
                                  np.where(b, 1e-2, 0.0)))}
    terms["all"] = {k: v for d in terms.values() for k, v in d.items()}
    return (eps, wp, gm, z), (dx, dy, dz, dt, ns, 10, 16, 48, src, cp), terms


@pytest.mark.parametrize("term", ["lorentz", "gain", "chi2", "raman", "all"])
def test_s2_12_jax_3d_carries_lorentz_chi2_raman_gain(term):
    """audit S2-12 (wave-7, 2026-11-02): run_3d_jax used to omit the Lorentz / chi2 / Raman / gain
    polarizations its 2-D twin carries, and the dispatcher raised -- 3-D gradient-based design of
    dispersive / active media was impossible. Ported term for term from run_3d. Gate: every term
    (and all four together) matches the numpy run_3d to < 1e-12 relative on all eight probe planes
    (measured 2.2e-15 .. 3.2e-15), where each term moves the probes by 5e-3 .. 5e-2 (so the parity
    is ~12 orders above the noise). With every term off the probes are bitwise the pre-port kernel's
    (measured once against 7b9481a; the carry/trace is unchanged by construction)."""
    pytest.importorskip("jax")
    from dynameta.optics.fdtd_nd.kernels3d import run_3d
    from dynameta.optics.fdtd_nd.kernels3d_jax import run_3d_jax
    mats, grid, terms = _s2_12_fixture()
    kw = terms[term]
    ref0 = run_3d(*mats, *grid[:-1], grid[-1], np)
    ref = run_3d(*mats, *grid[:-1], grid[-1], np, **kw)
    got = [np.asarray(v) for v in run_3d_jax(*mats, *grid, **kw)]
    scale = max(float(np.max(np.abs(a))) for a in ref)
    effect = max(float(np.max(np.abs(a - b))) for a, b in zip(ref, ref0)) / scale
    parity = max(float(np.max(np.abs(a - b))) for a, b in zip(ref, got)) / scale
    assert effect > 1e-3, (term, effect)                     # the term really acts
    assert parity < 1e-12, (term, parity)


def test_s2_12_jax_3d_lorentz_is_differentiable_and_dispatched():
    """The point of S2-12 is jax.grad through a dispersive 3-D march: d(probe energy)/d(Lorentz
    strength) from jax.grad matches a central finite difference, and solve_fdtd_3d(backend='jax')
    accepts a Lorentz layer (it raised NotImplementedError before)."""
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp
    from dynameta.optics.fdtd_nd import solve_fdtd_3d
    from dynameta.optics.fdtd_nd.kernels3d_jax import run_3d_jax
    mats, grid, terms = _s2_12_fixture()
    C1, C2, C3 = terms["lorentz"]["lor"]

    def energy(s):
        out = run_3d_jax(*mats, *grid, lor=(C1, C2, s * jnp.asarray(C3)))
        return jnp.sum(out[4] ** 2 + out[5] ** 2)              # transmitted Ex, Ey plane energy

    g = float(jax.grad(energy)(1.0))
    h = 1e-4
    fd = (float(energy(1.0 + h)) - float(energy(1.0 - h))) / (2.0 * h)
    assert abs(g) > 0.0 and g == pytest.approx(fd, rel=1e-5)
    lay = [FDTDLayer(thickness_m=150e-9, eps_inf=2.0, lorentz_w0_rad_s=1.4e15, lorentz_delta_eps=0.5,
                     lorentz_gamma_rad_s=1e14)]
    kw = dict(period_x_m=200e-9, period_y_m=200e-9, nx=2, ny=2, lambda_min_m=LMIN, lambda_max_m=LMAX,
              resolution=8, n_pad_wave=3.0)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)       # the C3-6 ring-time note
        rj = solve_fdtd_3d(lay, backend="jax", **kw)
        rn = solve_fdtd_3d(lay, backend="numpy", **kw)
    m = rn.band
    assert float(np.max(np.abs(rj.R0[m] - rn.R0[m]))) < 1e-9
