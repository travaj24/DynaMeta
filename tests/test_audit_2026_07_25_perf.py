"""BYTE-IDENTITY gates for the 2026-07-25 audit's PERFORMANCE remediations (wave 4).

Every item in that wave was a pure speed/memory change with the contract "nothing moves
numerically". These gates pin that contract against FROZEN reference implementations of the code
as it stood BEFORE each hoist -- a plain "does it still pass" suite cannot catch a last-bit drift
in a physics kernel, and a tolerance-based comparison would defeat the whole point.

  P-1  fdtd_nd/kernels2d.run_2d_te   -- per-step loop invariants hoisted; the Kerr / Drude blocks
                                        gated on an EXACT zero test.  Reference: the pre-hoist
                                        inner loop, verbatim.
  P-2  optics/ringdown._hankel       -- fancy-index copy -> sliding_window_view.
  P-3  optics/ringdown.matrix_pencil -- NEW big-N workspace guard (the only behaviour change in
                                        the wave: it RAISES instead of allocating gigabytes).
  P-4  materials/optical_model._maclaurin_re_from_im -- the two loop-invariant parity subsets.
  P-5  optics/spdc_design.jsa        -- np.vectorize -> one array call, with the scalar fallback.
  P-14 optics/resonance.find_poles   -- per-call evaluation memo.

The reference kernels below are DELIBERATE copies of superseded code. They are frozen oracles,
not live physics: do not "fix" them to match a future kernel -- if the kernel changes physics on
purpose, these gates are supposed to fail and be re-derived.
"""
import math

import numpy as np
import pytest

from dynameta.constants import EPS0, MU0
from dynameta.materials.optical_model import _maclaurin_re_from_im
from dynameta.optics import resonance as rz
from dynameta.optics import ringdown as rd
from dynameta.optics.fdtd_nd.cpml import cpml_z
from dynameta.optics.fdtd_nd.kernels2d import run_2d_te
from dynameta.optics.spdc_design import jsa


def _bits(a):
    """Byte-level identity, dtype and shape included (np.array_equal ignores -0.0 vs +0.0)."""
    a = np.asarray(a)
    return (a.dtype.str, a.shape, a.tobytes())


# ==================================================================================== P-1
# Frozen pre-hoist 2-D TE inner loop (kernels2d.py as of v0.9.0, Lorentz path included).
def _run_2d_te_prehoist(eps_inf, wp, gam, chi3, dx, dz, dt, nsteps, k_src, k_pL, k_pR, src,
                        cpml, lor=None):
    xp = np
    nx, nz = eps_inf.shape
    (ke, be, ce), (kh, bh, ch) = cpml
    do_lor = lor is not None
    if do_lor:
        C1, C2, C3 = lor
        PL = xp.zeros((nx, nz)); PLp = xp.zeros((nx, nz))
    Ey = xp.zeros((nx, nz)); Hx = xp.zeros((nx, nz)); Hz = xp.zeros((nx, nz))
    Jy = xp.zeros((nx, nz))
    psi_hxz = xp.zeros((nx, nz)); psi_eyz = xp.zeros((nx, nz))
    aJ = (1.0 - gam * dt / 2.0) / (1.0 + gam * dt / 2.0)
    bJ = (EPS0 * wp ** 2 * dt / 2.0) / (1.0 + gam * dt / 2.0)
    eyL = xp.empty((nsteps, nx)); hxL = xp.empty((nsteps, nx))
    eyR = xp.empty((nsteps, nx)); hxR = xp.empty((nsteps, nx))
    cmu = dt / MU0
    for n in range(nsteps):
        dEy_dz = (Ey[:, 1:] - Ey[:, :-1]) / dz
        psi_hxz[:, :-1] = bh[:-1] * psi_hxz[:, :-1] + ch[:-1] * dEy_dz
        Hx[:, :-1] += cmu * (dEy_dz / kh[:-1] + psi_hxz[:, :-1])
        Hz += -cmu * (xp.roll(Ey, -1, axis=0) - Ey) / dx
        dHx_dz = (Hx[:, 1:] - Hx[:, :-1]) / dz
        psi_eyz[:, 1:] = be[1:] * psi_eyz[:, 1:] + ce[1:] * dHx_dz
        curl = xp.zeros((nx, nz))
        curl[:, 1:] += dHx_dz / ke[1:] + psi_eyz[:, 1:]
        curl -= (Hz - xp.roll(Hz, 1, axis=0)) / dx
        if do_lor:
            PLnew = C1 * PL + C2 * PLp + C3 * Ey
            curl = curl - (PLnew - PL) / dt
            PLp = PL; PL = PLnew
        eps_eff = eps_inf + 3.0 * chi3 * Ey ** 2
        denom = EPS0 * eps_eff / dt + bJ / 2.0
        Eynew = (EPS0 * eps_eff / dt * Ey + curl - 0.5 * (1.0 + aJ) * Jy
                 - 0.5 * bJ * Ey) / denom
        Jynew = aJ * Jy + bJ * (Eynew + Ey)
        Jy = Jynew
        Eynew[:, k_src] += src[n]
        Eynew[:, 0] = 0.0; Eynew[:, -1] = 0.0
        Ey = Eynew
        eyL[n] = Ey[:, k_pL]; hxL[n] = 0.5 * (Hx[:, k_pL] + Hx[:, k_pL - 1])
        eyR[n] = Ey[:, k_pR]; hxR[n] = 0.5 * (Hx[:, k_pR] + Hx[:, k_pR - 1])
    return eyL, hxL, eyR, hxR


_NX, _NZ, _NSTEPS = 16, 160, 260
_DX = _DZ = 5e-9
_DT = 0.5 / (3e8 * math.sqrt(1 / _DX ** 2 + 1 / _DZ ** 2))
_SRC = (np.sin(np.arange(_NSTEPS) * 0.05)
        * np.exp(-((np.arange(_NSTEPS) - 80) / 30.0) ** 2))


def _cell(wp=0.0, gam=0.0, chi3=0.0, eps=2.1, sl=slice(70, 100)):
    ei = np.full((_NX, _NZ), eps)
    w = np.zeros((_NX, _NZ)); g = np.zeros((_NX, _NZ)); c3 = np.zeros((_NX, _NZ))
    w[:, sl] = wp; g[:, sl] = gam; c3[:, sl] = chi3
    return ei, w, g, c3


def _lorentz_arrays(w0=2.0e15, gl=1.0e14, deps=2.0, sl=slice(70, 100)):
    """Physical (stable) Lorentz ADE coefficients, built exactly as solve2d builds them."""
    a = np.zeros((_NX, _NZ)); a[:, sl] = w0
    b = np.zeros((_NX, _NZ)); b[:, sl] = gl
    d = np.zeros((_NX, _NZ)); d[:, sl] = deps
    den = 1.0 + b * _DT / 2.0
    return ((2.0 - a ** 2 * _DT ** 2) / den, (b * _DT / 2.0 - 1.0) / den,
            (EPS0 * d * a ** 2 * _DT ** 2) / den)


@pytest.mark.parametrize("name,kw,lor", [
    ("passive", dict(), False),                       # chi3 == 0 AND wp == 0: both fast paths
    ("passive_damped", dict(gam=1e14), False),        # wp == 0 but gam != 0 -> aJ != 1, bJ == 0
    ("drude", dict(wp=1.2e16, gam=1e14), False),      # Drude ADE live, Kerr hoisted
    ("kerr", dict(chi3=1e-18), False),                # Kerr live, Drude skipped
    ("drude_kerr", dict(wp=1.2e16, gam=1e14, chi3=1e-18), False),
    ("lorentz", dict(), True),                        # the ADE block REBINDS curl each step
])
def test_p1_2d_kernel_is_byte_identical_to_the_prehoist_loop(name, kw, lor):
    """P-1. The hoisted 2-D TE kernel must return BIT-IDENTICAL probes to the pre-hoist loop.

    The two fast paths are gated on an exact `np.any` zero test, so `chi3 == 0` really does make
    `eps_inf + 3*chi3*E^2` equal `eps_inf`, and `wp == 0` really does keep `Jy` at the zeros it
    started at (bJ == 0 exactly, so the two dropped terms are exact zeros). `curl` is refilled in
    a preallocated buffer instead of being reallocated, and the two `np.roll` copies became slice
    writes -- all of which are the same arithmetic on the same operands, not an approximation.
    """
    arrs = _cell(**kw)
    cp = cpml_z(_NZ, _DZ, _DT, 8)
    lo = _lorentz_arrays() if lor else None
    ref = _run_2d_te_prehoist(*arrs, _DX, _DZ, _DT, _NSTEPS, 20, 14, _NZ - 14, _SRC, cp, lor=lo)
    got = run_2d_te(*arrs, _DX, _DZ, _DT, _NSTEPS, 20, 14, _NZ - 14, _SRC, cp, lor=lo)
    for k, (a, b) in enumerate(zip(ref, got)):
        assert np.all(np.isfinite(b)), "{}: probe {} diverged -- the fixture is unstable".format(name, k)
        assert _bits(a) == _bits(b), "{}: probe array {} is not byte-identical".format(name, k)


# ==================================================================================== P-2 / P-3
def _hankel_prehoist(y, L):
    """Frozen pre-P-2 Hankel construction (materialized int64 fancy index)."""
    idx = np.arange(y.size - L)[:, None] + np.arange(L + 1)[None, :]
    return y[idx]


@pytest.mark.parametrize("N,L", [(64, 25), (400, 160), (1201, 480)])
def test_p2_hankel_view_is_byte_identical_to_the_fancy_index_copy(N, L):
    """P-2. `sliding_window_view` must reproduce the fancy-indexed Hankel matrix exactly (it is
    the same elements in the same order, only without the int64 index array -- measured peak
    traced memory 1536 MB -> 0.001 MB at N = 20000)."""
    t = np.arange(N) * 1e-15
    y = np.exp(-t / 3e-13) * np.cos(2 * np.pi * 2e14 * t) + 0.3 * np.cos(2 * np.pi * 2.6e14 * t)
    ref = _hankel_prehoist(y, L)
    got = rd._hankel(y, L)
    assert got.shape == ref.shape
    assert _bits(np.ascontiguousarray(got)) == _bits(ref)
    assert not got.flags.writeable, "the view must stay read-only (a mutating caller must .copy())"


def test_p2_matrix_pencil_modes_unchanged_on_a_two_mode_trace():
    """P-2 end to end: the mode list a caller actually sees is unchanged, exactly."""
    dt = 2e-16
    t = np.arange(1200) * dt
    y = (np.exp(-t / 5e-14) * np.cos(2 * np.pi * 3e14 * t)
         + 0.4 * np.exp(-t / 2e-14) * np.cos(2 * np.pi * 3.6e14 * t))
    modes = rd.matrix_pencil(y, dt)
    assert len(modes) == 2
    # the physics these bits encode (both modes recovered to <0.1%), so a future re-derivation of
    # the frozen reference still has a physical anchor
    f = sorted(m.omega_rad_s / (2 * np.pi) for m in modes)
    assert f[0] == pytest.approx(3.0e14, rel=1e-3)
    assert f[1] == pytest.approx(3.6e14, rel=1e-3)


def test_p3_big_n_guard_fires_above_the_documented_length_without_allocating():
    """P-3. `matrix_pencil` / `ringdown_q` are exported for arbitrary traces and had NO length
    guard: the Hankel SVD is O(N^3) time / O(N^2) memory in the RAW sample count (N = 8000
    measured 13.8 s / ~370 MB; N = 20000 extrapolates to ~4 min / ~2.3 GB). The guard must fire
    BEFORE the allocation -- this test therefore only ever allocates the 160 kB trace itself.
    """
    big = np.zeros(20000)                                   # 0.16 MB in, 2.05 GB refused
    with pytest.raises(ValueError, match="SVD workspace"):
        rd.matrix_pencil(big, 1e-15)
    with pytest.raises(ValueError, match="SVD workspace"):
        rd.ringdown_q(big, 1e-15)                           # the same guard through the wrapper
    # a genuinely complex trace costs 16 B/sample, so its ceiling is ~1/sqrt(2) of the real one
    tc = np.arange(14000) * 1e-15
    with pytest.raises(ValueError, match="SVD workspace"):
        rd.matrix_pencil(np.exp((-1.0 / 3e-13 - 2j * np.pi * 2e14) * tc), 1e-15)
    # the message must name the way out, not just refuse
    try:
        rd.matrix_pencil(big, 1e-15)
    except ValueError as e:
        assert "DECIMATE" in str(e) and "max_bytes" in str(e)


def test_p3_guard_threshold_brackets_the_measured_case_and_clears_shipped_sizes():
    """P-3. Threshold placement, probed through the estimator (no allocation at all): everything
    the repo actually feeds the pencil must pass, and the audit's measured N = 20000 disaster
    case must not.  `fdtd_etalon_ringdown` caps at max_fit_samples = 1200; `hydro_fdtd`'s
    bulk-plasmon ringdown hands over ~5000."""
    def peak(N, itemsize=8):
        L = max(2, min(int(round(0.4 * N)), N - 2))
        return rd._pencil_peak_bytes(N - L, L + 1, itemsize)

    for N in (1200, 5000, 8000, 16000):                     # shipped / plausible
        assert peak(N) <= rd._PENCIL_MAX_BYTES, N
    for N in (20000, 30000):                                # the audit's measured blow-up
        assert peak(N) > rd._PENCIL_MAX_BYTES, N
    assert peak(8000) / 1e6 == pytest.approx(328.0, rel=0.05)   # ~0.33 GB at the 13.8 s case


def test_p3_guard_budget_is_configurable_in_both_directions():
    """P-3. The budget is a knob, not a wall: a caller with the RAM can lift it, and a caller who
    wants a tighter cap gets one. Neither direction may change the returned modes."""
    dt = 2e-16
    t = np.arange(1200) * dt
    y = np.exp(-t / 5e-14) * np.cos(2 * np.pi * 3e14 * t)
    with pytest.raises(ValueError, match="SVD workspace"):
        rd.matrix_pencil(y, dt, max_bytes=1.0)
    a = rd.matrix_pencil(y, dt)
    b = rd.matrix_pencil(y, dt, max_bytes=np.inf)
    assert [(m.omega_rad_s, m.gamma_rad_s, m.amplitude) for m in a] == \
           [(m.omega_rad_s, m.gamma_rad_s, m.amplitude) for m in b]


# ==================================================================================== P-4
def _maclaurin_prehoist(omega, im_chi, h, chunk=0):
    """Frozen pre-P-4 body: a boolean parity mask and two np.where temporaries per row block."""
    omega = np.asarray(omega, dtype=np.float64)
    im_chi = np.asarray(im_chi, dtype=np.float64)
    N = omega.size
    idx = np.arange(N)
    w2 = omega * omega
    p = omega * im_chi
    pref = 4.0 * float(h) / np.pi
    re = np.empty(N, dtype=np.float64)
    step = int(chunk) if int(chunk) > 0 else max(32, min(512, 4_000_000 // max(1, N)))
    for a in range(0, N, step):
        b = min(a + step, N)
        j = idx[a:b]
        mask = (((idx[None, :] - j[:, None]) & 1) == 1)
        den = w2[None, :] - w2[j][:, None]
        re[a:b] = pref * np.where(mask, p[None, :] / np.where(mask, den, 1.0), 0.0).sum(axis=1)
    return re


@pytest.mark.parametrize("N", [33, 256, 1001])
def test_p4_maclaurin_parity_blocks_are_byte_identical(N):
    """P-4. The opposite-parity mask has only two distinct values over the whole sum, so the rows
    are now processed one parity at a time in a single reused buffer. Every row still holds the
    same values (same-parity columns zeroed) and is still reduced by ONE np.sum over its own
    contiguous (N,) block -- i.e. the identical pairwise summation, which is why this is
    byte-identical and why collapsing the two parities into a compacted half-length sum (the
    obvious next 4x) would NOT be."""
    om = np.linspace(1e14, 6e15, N)
    im = 0.4 * np.exp(-((om - 2.5e15) / 5e14) ** 2) + 0.05
    h = float(om[1] - om[0])
    # the frozen reference is the bare Maclaurin sum, i.e. exactly the block this item rewrote;
    # the endpoint-consistency tail (finding Q-5) is downstream of it and untouched.
    ref = _maclaurin_prehoist(om, im, h)
    got = _maclaurin_re_from_im(om, im, h, edge_correct=False)
    assert _bits(ref) == _bits(got)
    # an explicit chunk must not change a single bit either (it is a memory knob, not a method)
    assert _bits(_maclaurin_re_from_im(om, im, h, edge_correct=False, chunk=7)) == _bits(got)
    # and the edge-corrected output is that same sum plus the (unchanged) edge strips
    corr = _maclaurin_re_from_im(om, im, h, edge_correct=True) - got
    assert np.all(np.isfinite(corr)) and np.any(corr != 0.0)


# ==================================================================================== P-5
_WP_SPDC = 2 * np.pi * 3e8 / 775e-9


def _pump(w):
    return np.exp(-((w - _WP_SPDC) / (2 * np.pi * 5e11)) ** 2)


def _dk_elementwise(a, b):
    return 1e-3 * (a - 0.5 * _WP_SPDC) ** 2 / _WP_SPDC ** 2 - 2e-4 * (b - 0.5 * _WP_SPDC) / _WP_SPDC


def test_p5_jsa_array_call_matches_the_vectorize_pass_bitwise():
    """P-5. For an elementwise dk_func the single array call returns the same float64 array as
    the n^2 scalar `np.vectorize` pass -- measured ~139x-441x faster on the dk construction."""
    ws = np.linspace(0.45 * _WP_SPDC, 0.55 * _WP_SPDC, 48)
    WS, WI = np.meshgrid(ws, ws, indexing="ij")
    ref = np.vectorize(lambda a, b: float(_dk_elementwise(a, b)))(WS, WI)
    assert _bits(ref) == _bits(np.asarray(_dk_elementwise(WS, WI), dtype=float))
    # and the JSA built from it
    f = jsa(ws, ws, _pump, _dk_elementwise, 1e-2)
    assert f.shape == (48, 48) and np.isfinite(f).all()


def test_p5_scalar_only_dk_func_still_works_through_the_fallback():
    """P-5. A dk_func that cannot take arrays (it branches on the scalar value, or calls math.*)
    must keep working: the array attempt raises, and the np.vectorize path takes over. A dk_func
    that ignores its arguments and returns a SCALAR also falls back (shape mismatch)."""
    ws = np.linspace(0.45 * _WP_SPDC, 0.55 * _WP_SPDC, 24)

    def dk_scalar_only(a, b):
        d = 1e-3 * (a - 0.5 * _WP_SPDC) ** 2 / _WP_SPDC ** 2
        return math.sqrt(d) if d >= 0 else -math.sqrt(-d)    # ValueError on an array

    with pytest.raises((ValueError, TypeError)):              # the array call really does fail
        dk_scalar_only(np.array([1.0, 2.0]), np.array([1.0, 2.0]))
    f_scalar = jsa(ws, ws, _pump, dk_scalar_only, 1e-2)
    ref = np.vectorize(lambda a, b: float(dk_scalar_only(a, b)))(*np.meshgrid(ws, ws, indexing="ij"))
    from dynameta.optics.twm_reference import phase_matching_sinc
    WS, WI = np.meshgrid(ws, ws, indexing="ij")
    exp = np.asarray(_pump(WS + WI), dtype=complex) * phase_matching_sinc(ref, 1e-2, None)
    exp = exp / np.sqrt(np.sum(np.abs(exp) ** 2))
    assert _bits(f_scalar) == _bits(exp)
    assert np.isfinite(jsa(ws, ws, _pump, lambda a, b: 3.0, 1e-2)).all()   # scalar RETURN


# ==================================================================================== P-14
def test_p14_find_poles_memo_is_per_call_and_never_leaks_between_closures():
    """P-14. The evaluation memo cuts ~57-63% of `find_poles`' function calls (measured 2.2x on a
    39-pole hydrodynamic box). It MUST be per-call: a module-level cache would serve one stack's
    D(omega) to the next one's search over the same box -- which is exactly how `q_budget` drives
    the finder (lossy pass, then lossless pass, same box, different function)."""
    box_c, box_s = complex(2.0, -0.35), complex(1.0, 0.5)
    r1 = rz.find_poles(lambda z: (z - (2.0 - 0.3j)) * (z - (2.5 - 0.1j)), box_c, box_s)
    r2 = rz.find_poles(lambda z: (z - (1.7 - 0.2j)) * (z - (2.4 - 0.4j)), box_c, box_s)
    got1 = sorted((round(z.real, 9), round(z.imag, 9)) for z in r1)
    got2 = sorted((round(z.real, 9), round(z.imag, 9)) for z in r2)
    assert got1 == [(2.0, -0.3), (2.5, -0.1)]
    assert got2 == [(1.7, -0.2), (2.4, -0.4)]

    # a STATEFUL func: the second call must see the new roots, not the first call's cached values
    state = {"a": 2.0 - 0.3j}

    def f(z):
        return z - state["a"]

    a = rz.find_poles(f, box_c, box_s)
    state["a"] = 2.4 - 0.2j
    b = rz.find_poles(f, box_c, box_s)
    assert a and b
    assert abs(a[0] - (2.0 - 0.3j)) < 1e-9 and abs(b[0] - (2.4 - 0.2j)) < 1e-9


def test_p14_find_poles_memo_actually_suppresses_duplicate_evaluations():
    """P-14. The mechanism, not just the result: the finder must ask for the same omega many
    times (linspace(a,b,2n) contains linspace(a,b,n) exactly, and the quad-tree re-walks shared
    box edges) and evaluate it once."""
    seen = []

    def f(z):
        seen.append((z.real, z.imag))
        return (z - (2.0 - 0.3j)) * (z - (2.5 - 0.1j))

    rz.find_poles(f, complex(2.0, -0.35), complex(1.0, 0.5))
    assert len(seen) == len(set(seen)), "a point was evaluated twice: the memo is not working"
    assert len(seen) > 500, "fixture too small to be a meaningful memo test"
