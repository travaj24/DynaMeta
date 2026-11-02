"""Backend probes, the numba njit shim, and string-based backend resolution.

Split from the former monolithic fdtd_nd.py; see the package __init__ docstring
for conventions. Bodies are verbatim from the original module.
"""
from __future__ import annotations

import functools
import threading

# Optional Numba fast CPU kernel (the fused single-pass, prange-threaded backend). Numba JITs the whole
# timestep into one compiled function -> no per-op kernel/launch overhead (the cure for the small-grid
# case where naive vectorized GPU/NumPy is launch-bound), threaded over x. Guarded so the module still
# imports without numba; selected via solve_fdtd_2d(backend='numba').
#
# LAZY, like cupy/jax below (audit V-14, 2026-11-02). This block used to `from numba import njit,
# prange` at import time, three lines above the comment that defers cupy/jax "because importing them
# is slow" -- and numba is slow too: measured 0.32 s for `numba` itself plus 0.39 s for the
# `numba.cuda` import kernels2d_numba made at module level, i.e. ~0.7 s of the 1.4-1.5 s
# `import dynameta.optics.fdtd_nd` cost (and of `import dynameta.optics.fdtd`, which reaches this
# package through spec.py), paid by every numpy-backend user. Now nothing imports numba until a
# numba kernel is CALLED or `have_numba()` / `HAVE_NUMBA` is asked:
#   * `njit(...)` returns a `_LazyJit` that compiles with the SAME numba.njit options on first call
#     (or first attribute access, e.g. `.signatures`), or falls back to the plain Python function
#     when numba is absent (exactly the old no-numba shim);
#   * `prange` here is a placeholder that behaves as `range` from Python and is swapped for
#     `numba.prange` in the kernel module's globals just before that module's first compile -- numba
#     resolves `prange` from the function's globals at compile time, so the compiled kernels are the
#     same IR as before (bit-identical outputs, checked on every numba front end);
#   * `HAVE_NUMBA` is a module attribute computed on first access (PEP 562 `__getattr__`), so
#     `from dynameta.optics.fdtd_nd import HAVE_NUMBA` keeps working; internal code calls
#     `have_numba()` instead, so importing the package never triggers the probe.
_NUMBA_OK = None
_JIT_LOCK = threading.Lock()


def have_numba():
    """True if numba imports (cached; the import is done once, on first ask -- audit V-14)."""
    global _NUMBA_OK
    if _NUMBA_OK is None:
        try:
            import numba                                     # noqa: F401
            _NUMBA_OK = True
        except Exception:                                    # pragma: no cover
            _NUMBA_OK = False
    return _NUMBA_OK


class _PrangePlaceholder:
    """`range` from plain Python; replaced by `numba.prange` in a kernel module's globals right
    before that module's first numba compile (see `_LazyJit._dispatcher`)."""

    def __call__(self, *args):
        return range(*args)

    def __repr__(self):
        return "<fdtd_nd lazy prange placeholder>"


prange = _PrangePlaceholder()


class _LazyJit:
    """A numba.njit-decorated kernel whose decoration (and hence `import numba`) is deferred to the
    first call. Attribute access other than the wrapped function's metadata (`__doc__`, `__name__`,
    `py_func`, ...) is forwarded to the real dispatcher, compiling it first."""

    def __init__(self, py_func, jit_args, jit_kwargs):
        self._disp = None
        self.py_func = py_func
        self._jit_args = jit_args
        self._jit_kwargs = jit_kwargs
        functools.update_wrapper(self, py_func)

    def _dispatcher(self):
        if self._disp is None:
            with _JIT_LOCK:
                if self._disp is None:
                    if have_numba():
                        import numba
                        g = self.py_func.__globals__
                        if isinstance(g.get("prange"), _PrangePlaceholder):
                            g["prange"] = numba.prange
                        self._disp = numba.njit(*self._jit_args, **self._jit_kwargs)(self.py_func)
                    else:                                    # pragma: no cover -- the old no-numba shim
                        self._disp = self.py_func
        return self._disp

    def __call__(self, *args, **kwargs):
        return self._dispatcher()(*args, **kwargs)

    def __getattr__(self, name):
        if name.startswith("__") or name in ("_disp", "py_func", "_jit_args", "_jit_kwargs"):
            raise AttributeError(name)
        return getattr(self._dispatcher(), name)


def njit(*args, **kwargs):
    """Drop-in for `numba.njit` (bare `@njit` or `@njit(**options)`) that defers the numba import
    and compile to the first call; see the module note above (audit V-14)."""
    if len(args) == 1 and callable(args[0]) and not kwargs:
        return _LazyJit(args[0], (), {})

    def _wrap(f):
        return _LazyJit(f, args, kwargs)
    return _wrap


def __getattr__(name):                                       # PEP 562: lazy module attributes
    if name == "HAVE_NUMBA":
        return have_numba()
    raise AttributeError("module {!r} has no attribute {!r}".format(__name__, name))


# --- Optional GPU / autodiff backends, lazily probed (importing cupy/jax is slow, so only on demand). ---
_CUPY_OK = None
_JAX_OK = None


def _have_cupy():
    """True if CuPy imports AND a CUDA device is present -- the vectorized loop runs on it via xp=cupy."""
    global _CUPY_OK
    if _CUPY_OK is None:
        try:
            import cupy as cp
            _CUPY_OK = bool(cp.cuda.runtime.getDeviceCount() > 0)
        except Exception:                                    # pragma: no cover
            _CUPY_OK = False
    return _CUPY_OK


def have_jax():
    """True if a SUPPORTED JAX imports (>= 0.11 -- 0.11 broke several 0.10 APIs and dynameta's
    jax kernels are validated against 0.11 only). An older jax reports unavailable with a
    one-time warning (numpy fallback) rather than crashing deep inside a jitted trace. The
    differentiable XLA lax.scan backend (GPU is WSL2-only on Windows -> CPU)."""
    global _JAX_OK
    if _JAX_OK is None:
        try:
            import jax                                       # noqa: F401
            try:
                from dynameta.core.backend import require_jax_011
                require_jax_011(jax)
            except RuntimeError as e:
                import warnings
                warnings.warn(str(e) + " -- the jax backend is DISABLED for this session",
                              RuntimeWarning, stacklevel=2)
                _JAX_OK = False
                return _JAX_OK
            _JAX_OK = True
        except Exception:                                    # pragma: no cover
            _JAX_OK = False
    return _JAX_OK


_NBCUDA_OK = None


def have_numba_cuda():
    """True iff a numba-CUDA GPU kernel can actually launch here (numba present AND cuda.is_available()).
    Cached; the import + driver probe is done once."""
    global _NBCUDA_OK
    if _NBCUDA_OK is None:
        _NBCUDA_OK = False
        if have_numba():
            try:
                from numba import cuda
                _NBCUDA_OK = bool(cuda.is_available())
            except Exception:
                _NBCUDA_OK = False
    return _NBCUDA_OK


def available_backends():
    """The FDTD backends actually runnable on THIS machine. 'numpy' is always present (the dependency-free
    reference); 'numba' = the fused threaded CPU kernel (fastest for cache-resident unit cells, ~500-1900
    MC/s); 'cupy' = NVIDIA GPU via the vectorized loop (wins on large 3D volumes that fill the device);
    'jax' = the differentiable XLA scan loop (grad-through-FDTD for inverse design); 'numba-cuda' = a fused
    persistent cooperative-groups GPU kernel (2D-TE so far; wins ~3x on SMALL/unit-cell grids, occupancy-
    capped on very large volumes -> use 'cupy' there)."""
    bk = []
    if have_numba():
        bk.append("numba")
    bk.append("numpy")
    if _have_cupy():
        bk.append("cupy")
    if have_jax():
        bk.append("jax")
    if have_numba_cuda():
        bk.append("numba-cuda")
    return bk


def resolve_backend(backend):
    """Map a requested backend -- including 'auto' and the 'cpu'/'gpu' aliases -- to a concrete available
    one, or raise a clear error (available list + install hint) for an unavailable EXPLICIT request. 'auto'
    picks the fastest CPU backend present (numba else numpy) and NEVER silently picks the GPU, because a
    cache-resident metasurface unit cell runs faster on the threaded CPU kernel than on a launch-bound GPU
    (the benchmark: numba 561-1882 MC/s vs cupy 20-120 MC/s on unit-cell grids)."""
    avail = available_backends()
    fast_cpu = "numba" if have_numba() else "numpy"
    name = {"auto": fast_cpu, "cpu": fast_cpu, "gpu": "cupy",
            "cuda": "numba-cuda"}.get(str(backend).lower(), str(backend).lower())
    if name not in avail:
        hint = {"numba": "pip install numba", "cupy": "pip install cupy-cuda12x (and an NVIDIA GPU)",
                "jax": "pip install jax",
                "numba-cuda": "pip install numba-cuda nvidia-cuda-runtime-cu12 (and an NVIDIA GPU)"}.get(name, "")
        raise RuntimeError("FDTD backend '{}' is not available here; available = {}.{}".format(
            backend, avail, (" Try: " + hint) if hint else ""))
    return name


def check_scan_length(entry_point, src, nsteps):
    """audit X-16: the five JAX FDTD kernels (run_2d_te_jax, run_2d_te_oblique_jax,
    run_2d_tm_oblique_jax, run_3d_jax, run_3d_oblique_jax) take `nsteps` but run `lax.scan` over
    `src`, so the loop length is len(src) and `nsteps` was silently ignored. Every in-repo caller
    builds src with exactly nsteps samples, but nothing enforced it: a caller that passed a longer or
    shorter src got a different-length march -- and a mis-sized frequency axis downstream -- instead
    of a shape error. Refuse the mismatch up front (before any jax import), so the NumPy/numba
    kernels (which loop range(nsteps) and index src[n]) and the JAX ones share one contract."""
    n_src = len(src)
    if int(n_src) != int(nsteps):
        raise ValueError(
            "{}: len(src) = {} but nsteps = {} -- the JAX lax.scan marches len(src) steps, so the two "
            "must agree (the NumPy/numba kernels loop range(nsteps)); audit X-16.".format(
                entry_point, int(n_src), int(nsteps)))
