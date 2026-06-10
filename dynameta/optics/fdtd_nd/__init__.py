"""
2D/3D FDTD optical backend (roadmap: extend the validated 1D Yee solver to a periodic 2D/3D engine).

PHASE 0 -- a 2D TE (E_y, H_x, H_z) Yee solver: a plane wave propagating in +z on a unit cell that is
PERIODIC in x (a 1D grating / laterally-structured slab), at NORMAL incidence. Carries the same physics
as the 1D baseline: a semi-implicit Drude ADE per E-component, an instantaneous Kerr chi3, a
modulated-Gaussian soft source, and the TWO-RUN (vacuum reference + structure) broadband R(omega)/
T(omega) extraction. CFS-CPML absorbing layers (+ PEC backing) at the z ends; periodic in x.

Backends (solve_fdtd_2d(backend=...)):
  * 'numba' (FASTEST for unit cells) -- a fused, prange-threaded, JIT-compiled CPU kernel
    (_te2d_numba). The metasurface unit-cell grid is cache-resident, so this runs ~500-1900 MC/s
    (machine-precision identical to the reference; ~10-150x NumPy and FASTER than naive GPU here,
    because a small grid cannot fill a GPU and pays launch/PCIe overhead).
  * 'numpy' (the REFERENCE oracle) -- the vectorized run loop, the correctness baseline every faster
    kernel is validated against, and the dependency-free default.
  * 'cupy' (NVIDIA GPU) -- the vectorized loop on the device; wins only on LARGE grids (big 3D volumes)
    that exceed cache and fill the GPU. A fused CuPy RawKernel / Numba-CUDA kernel is the next step there.
  * 'jax' (DIFFERENTIABLE) -- the same loop as a compiled XLA lax.scan, so jax.grad gives
    d(R,T)/d(geometry/material) for gradient-based inverse design; XLA-fused on CPU (GPU is WSL2-only on
    Windows). Plus the convenience aliases 'auto' (fastest CPU present), 'cpu', 'gpu'.
available_backends() reports what is runnable here; _resolve_backend() maps the request (raising a clear
error with an install hint for an unavailable explicit pick). The hot loop is a swappable kernel boundary,
so a Taichi backend (one-source CPU+CUDA+Vulkan, when a Python-3.14 wheel exists) drops in unchanged.
Convention exp(-i omega t), SI; Im(eps) > 0 = loss. Reduces EXACTLY to the 1D solver + TMM for a
laterally-uniform stack at normal incidence (validation/fdtd_2d_reduces.py).

IMPLEMENTED since this docstring's first draft: CPML, full 3D (solve_fdtd_3d), per-cell diagonal +
magneto-optic tensor eps (solve_fdtd_3d_mo, gyrotropic magnetized-Drude ADE), Drude+Lorentz multipole
ADE, oblique Bloch incidence (2D s/p + 3D, complex-envelope) with BOTH fused numba kernels (2D-TE, 2D-TM,
full-vector 3D -- byte-exact, ~5-8x faster) AND differentiable JAX kernels (2D-TE, 2D-TM, full-vector 3D
-- byte-exact, jax.grad flows through the oblique scan for inverse-design-at-angle), plus a STRUCTURED
(laterally-patterned) 3D diagonal-tensor solve (solve_fdtd_3d_mo `lateral_tensor=`, validated vs grcwa
per-component). STILL DEFERRED: GPU NONLINEAR kernels (the linear numba-cuda path is hardware-validated).
"""

from __future__ import annotations

from dynameta.optics.fdtd import FDTDLayer

from dynameta.optics.fdtd_nd.backends import (_HAVE_NUMBA, _have_cupy, _have_jax,
                                              _have_numba_cuda, _resolve_backend,
                                              available_backends, njit, prange)
from dynameta.optics.fdtd_nd.cpml import _cpml_z
from dynameta.optics.fdtd_nd.results import (FDTD2DObliqueResult, FDTD2DResult,
                                             FDTD3DMOResult, FDTD3DResult, _flux,
                                             _flux3d)
from dynameta.optics.fdtd_nd.kernels2d import _run_2d_te
from dynameta.optics.fdtd_nd.kernels2d_numba import _te2d_cuda, _te2d_numba
from dynameta.optics.fdtd_nd.kernels2d_jax import _run_2d_te_jax
from dynameta.optics.fdtd_nd.oblique2d import (_run_2d_te_oblique,
                                               _run_2d_te_oblique_jax,
                                               _run_2d_tm_oblique,
                                               _run_2d_tm_oblique_jax, _run_oblique,
                                               _te2d_oblique_numba,
                                               _tm2d_oblique_numba)
from dynameta.optics.fdtd_nd.kernels3d import _run_3d, _run_3d_mo, _run_3d_oblique
from dynameta.optics.fdtd_nd.kernels3d_numba import (_run_3d_oblique_numba,
                                                     _te3d_numba)
from dynameta.optics.fdtd_nd.kernels3d_jax import (_run_3d_jax,
                                                   _run_3d_oblique_jax)
from dynameta.optics.fdtd_nd.solve2d import (_dispatch_2d_te, solve_fdtd_2d,
                                             solve_fdtd_2d_oblique)
from dynameta.optics.fdtd_nd.solve3d import (_dispatch_3d, solve_fdtd_3d,
                                             solve_fdtd_3d_mo,
                                             solve_fdtd_3d_oblique)
