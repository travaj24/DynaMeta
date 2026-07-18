"""Backend probes, the numba njit shim, and string-based backend resolution.

Split from the former monolithic fdtd_nd.py; see the package __init__ docstring
for conventions. Bodies are verbatim from the original module.
"""
from __future__ import annotations

# Optional Numba fast CPU kernel (the fused single-pass, prange-threaded backend). Numba JITs the whole
# timestep into one compiled function -> no per-op kernel/launch overhead (the cure for the small-grid
# case where naive vectorized GPU/NumPy is launch-bound), threaded over x. Guarded so the module still
# imports without numba; selected via solve_fdtd_2d(backend='numba').
try:
    from numba import njit, prange
    HAVE_NUMBA = True
except Exception:                                            # pragma: no cover
    HAVE_NUMBA = False

    def njit(*a, **k):                                       # no-op shim so the def parses without numba
        def _wrap(f):
            return f
        return _wrap if not (len(a) == 1 and callable(a[0])) else a[0]
    prange = range

_HAVE_NUMBA = HAVE_NUMBA                                     # back-compat alias (pre-promotion name)


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
                              RuntimeWarning)
                _JAX_OK = False
                return _JAX_OK
            _JAX_OK = True
        except Exception:                                    # pragma: no cover
            _JAX_OK = False
    return _JAX_OK


_have_jax = have_jax                                         # back-compat alias (pre-promotion name)


_NBCUDA_OK = None


def have_numba_cuda():
    """True iff a numba-CUDA GPU kernel can actually launch here (numba present AND cuda.is_available()).
    Cached; the import + driver probe is done once."""
    global _NBCUDA_OK
    if _NBCUDA_OK is None:
        _NBCUDA_OK = False
        if HAVE_NUMBA:
            try:
                from numba import cuda
                _NBCUDA_OK = bool(cuda.is_available())
            except Exception:
                _NBCUDA_OK = False
    return _NBCUDA_OK


_have_numba_cuda = have_numba_cuda                           # back-compat alias (pre-promotion name)


def available_backends():
    """The FDTD backends actually runnable on THIS machine. 'numpy' is always present (the dependency-free
    reference); 'numba' = the fused threaded CPU kernel (fastest for cache-resident unit cells, ~500-1900
    MC/s); 'cupy' = NVIDIA GPU via the vectorized loop (wins on large 3D volumes that fill the device);
    'jax' = the differentiable XLA scan loop (grad-through-FDTD for inverse design); 'numba-cuda' = a fused
    persistent cooperative-groups GPU kernel (2D-TE so far; wins ~3x on SMALL/unit-cell grids, occupancy-
    capped on very large volumes -> use 'cupy' there)."""
    bk = []
    if HAVE_NUMBA:
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
    fast_cpu = "numba" if HAVE_NUMBA else "numpy"
    name = {"auto": fast_cpu, "cpu": fast_cpu, "gpu": "cupy",
            "cuda": "numba-cuda"}.get(str(backend).lower(), str(backend).lower())
    if name not in avail:
        hint = {"numba": "pip install numba", "cupy": "pip install cupy-cuda12x (and an NVIDIA GPU)",
                "jax": "pip install jax",
                "numba-cuda": "pip install numba-cuda nvidia-cuda-runtime-cu12 (and an NVIDIA GPU)"}.get(name, "")
        raise RuntimeError("FDTD backend '{}' is not available here; available = {}.{}".format(
            backend, avail, (" Try: " + hint) if hint else ""))
    return name


_resolve_backend = resolve_backend                           # back-compat alias (pre-promotion name)
