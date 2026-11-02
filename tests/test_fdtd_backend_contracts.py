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
