"""
Array-backend namespace dispatch (Python Array API standard).

A SCOPED adoption of the dispatch pattern from the sibling Lumenairy library
(Free_Space_Optics/Lumenairy/lumenairy/backend/array.py, Andrew Traverso) -- here it is applied
ONLY to the pure-array CONSTITUTIVE material-response maps (the EffectModels in core.effects and the
graphene surface conductivity), NOT to the heavy solvers: NGSolve (FEM) and DEVSIM (carriers) are
C++ and consume host NumPy, and the scipy specializations (eigh_tridiagonal(select), LinearND/
Delaunay, least_squares, the tmm library) have no cupy/jax equivalent, so they stay on NumPy/SciPy.

The point of the seam: a constitutive map eps(fields, lambda) written against `xp = array_namespace
(...)` runs UNMODIFIED on NumPy (the default, bit-identical float64), CuPy, or JAX. When fed JAX
arrays it stays inside the JAX trace -- so eps(E), eps(T), eps(f), eps(theta), sigma(E_F) become
DIFFERENTIABLE, which is what a future RCWA-backed inverse-design loop needs (the gradient flows
design -> fields -> eps through these maps). The backend is pinned by the INPUT array type, not a
global flag; mixing backends in one call raises.

PRECISION: JAX defaults to float32, which WOULD sacrifice precision. This module forces
`jax_enable_x64` on first JAX import so the JAX path is float64 like NumPy. The NumPy default path
is always float64 and bit-identical to the pre-seam code.

Pure dispatch glue: imports only numpy; cupy/jax are detected via find_spec (free) and imported
lazily on first actual use.
"""

from __future__ import annotations

import importlib.util as _importlib_util
from typing import Any, Optional

import numpy as np

CUPY_AVAILABLE = _importlib_util.find_spec("cupy") is not None
JAX_AVAILABLE = _importlib_util.find_spec("jax") is not None

_cp = None
_jnp = None
_jax = None


def _ensure_jax_x64() -> None:
    """Force JAX into float64 mode (it defaults to float32) and FAIL FAST if the switch did not take.
    Called on first JAX use below. CAVEAT: this governs arrays created AFTER the switch -- if user code
    imported jax and built float32 arrays BEFORE touching dynameta, those arrays stay 32-bit; pass
    float64 inputs (or enable jax_enable_x64 yourself at program start). Without x64 the JAX
    constitutive path would silently drop to single precision -- unacceptable for the physics here."""
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as _probe
    import numpy as _np
    if _probe.zeros(1).dtype != _np.float64:    # the update must take effect for NEW arrays
        raise RuntimeError("dynameta: could not enable JAX float64 mode (jax_enable_x64) -- new jax "
                           "arrays are still 32-bit. Enable x64 before importing jax-dependent code "
                           "(e.g. JAX_ENABLE_X64=1 or jax.config.update at program start).")


def _get_cupy() -> Optional[Any]:
    global _cp
    if _cp is None and CUPY_AVAILABLE:
        import cupy as _cp_mod
        _cp = _cp_mod
    return _cp


def _get_jax() -> Optional[Any]:
    global _jax
    if _jax is None and JAX_AVAILABLE:
        _ensure_jax_x64()
        import jax as _jax_mod
        _jax = _jax_mod
    return _jax


def _get_jnp() -> Optional[Any]:
    global _jnp
    if _jnp is None and JAX_AVAILABLE:
        _ensure_jax_x64()
        import jax.numpy as _jnp_mod
        _jnp = _jnp_mod
    return _jnp


def is_numpy_array(x: Any) -> bool:
    """True only for a NumPy ndarray (NumPy 2.x exposes .device too, so duck-typing on .device
    cannot distinguish NumPy from CuPy/JAX -- use isinstance)."""
    return isinstance(x, np.ndarray)


def is_cupy_array(x: Any) -> bool:
    """True for a CuPy ndarray (False if CuPy is absent)."""
    if not CUPY_AVAILABLE:
        return False
    cp = _get_cupy()
    return cp is not None and isinstance(x, cp.ndarray)


def is_jax_array(x: Any) -> bool:
    """True for a concrete OR JIT-traced JAX array (False if JAX is absent)."""
    if not JAX_AVAILABLE:
        return False
    jx = _get_jax()
    if jx is None:
        return False
    if isinstance(x, jx.Array):
        return True
    try:
        return isinstance(x, jx.core.Tracer)
    except AttributeError:
        return False


def array_namespace(*arrays: Any) -> Any:
    """Return the array namespace (numpy / cupy / jax.numpy) for the given arrays, pinned by their
    type. Python scalars / lists / None do not pin a backend; if nothing pins one, returns NumPy.
    Mixing arrays from different backends raises TypeError (convert explicitly at the call site)."""
    # LAZY-IMPORT AVOIDANCE (the NumPy/scalar hot path must never import jax/cupy): is_jax_array /
    # is_cupy_array import jax / cupy to run their isinstance, so we (1) skip None and plain Python
    # scalars/sequences -- which don't pin a backend, and whose numpy-SCALAR forms (np.float64,
    # np.complex128) subclass float/complex so they are caught here too -- and (2) check the cheap
    # import-free is_numpy_array BEFORE the heavy predicates. A genuine non-NumPy array (jax/cupy)
    # is the only thing that reaches -- and thus imports -- its backend.
    saw_jax = saw_cupy = saw_numpy = False
    for a in arrays:
        if a is None or isinstance(a, (bool, int, float, complex, list, tuple)):
            continue
        if is_numpy_array(a):
            saw_numpy = True
        elif is_jax_array(a):
            saw_jax = True
        elif is_cupy_array(a):
            saw_cupy = True
    if sum([saw_jax, saw_cupy, saw_numpy]) > 1:
        raise TypeError("array_namespace: arrays from multiple backends (NumPy/CuPy/JAX); convert "
                        "all inputs to a single backend before calling.")
    if saw_jax:
        return _get_jnp()
    if saw_cupy:
        return _get_cupy()
    return np


def backend_name(xp: Any) -> str:
    """Human-readable name of an xp namespace from array_namespace."""
    if xp is np:
        return "numpy"
    if CUPY_AVAILABLE and xp is _get_cupy():
        return "cupy"
    if JAX_AVAILABLE and xp is _get_jnp():
        return "jax"
    return getattr(xp, "__name__", repr(xp))


def to_numpy(x: Any) -> np.ndarray:
    """Materialise x as a host NumPy ndarray (for I/O, plotting, and the NGSolve/DEVSIM boundary)."""
    if is_numpy_array(x):
        return x
    if is_cupy_array(x):
        cp = _get_cupy()
        return cp.asnumpy(x)               # type: ignore[union-attr]
    return np.asarray(x)                    # JAX arrays + everything else


def to_backend(x: Any, xp: Any) -> Any:
    """Convert x to namespace xp (numpy / cupy / jax.numpy). Cheap if already there; cross-backend
    materialises through the host. Used to lift a model's stored (NumPy) parameters to whatever
    backend the runtime field arrays are on, so array_namespace never sees a mixed-backend call."""
    if xp is np:
        return to_numpy(x)
    if CUPY_AVAILABLE and xp is _get_cupy():
        return x if is_cupy_array(x) else _get_cupy().asarray(to_numpy(x))   # type: ignore[union-attr]
    if JAX_AVAILABLE and xp is _get_jnp():
        return x if is_jax_array(x) else _get_jnp().asarray(to_numpy(x))     # type: ignore[union-attr]
    raise TypeError("to_backend: unrecognised target namespace {!r} (expected numpy/cupy/jax.numpy)"
                    .format(xp))


__all__ = [
    "CUPY_AVAILABLE", "JAX_AVAILABLE",
    "is_numpy_array", "is_cupy_array", "is_jax_array",
    "array_namespace", "backend_name", "to_numpy", "to_backend",
]
