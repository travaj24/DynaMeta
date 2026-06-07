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

DEFERRED to later phases: CPML (replaces Mur), full 3D, off-diagonal/magneto-optic tensor eps,
oblique Bloch incidence, and the GPU fast kernel.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from dynameta.constants import C_LIGHT, EPS0
from dynameta.optics.fdtd import FDTDLayer

MU0 = 1.0 / (EPS0 * C_LIGHT ** 2)

# Optional Numba fast CPU kernel (the fused single-pass, prange-threaded backend). Numba JITs the whole
# timestep into one compiled function -> no per-op kernel/launch overhead (the cure for the small-grid
# case where naive vectorized GPU/NumPy is launch-bound), threaded over x. Guarded so the module still
# imports without numba; selected via solve_fdtd_2d(backend='numba').
try:
    from numba import njit, prange
    _HAVE_NUMBA = True
except Exception:                                            # pragma: no cover
    _HAVE_NUMBA = False

    def njit(*a, **k):                                       # no-op shim so the def parses without numba
        def _wrap(f):
            return f
        return _wrap if not (len(a) == 1 and callable(a[0])) else a[0]
    prange = range


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


def _have_jax():
    """True if JAX imports -- the differentiable XLA lax.scan backend (GPU is WSL2-only on Windows -> CPU)."""
    global _JAX_OK
    if _JAX_OK is None:
        try:
            import jax                                       # noqa: F401
            _JAX_OK = True
        except Exception:                                    # pragma: no cover
            _JAX_OK = False
    return _JAX_OK


def available_backends():
    """The FDTD backends actually runnable on THIS machine. 'numpy' is always present (the dependency-free
    reference); 'numba' = the fused threaded CPU kernel (fastest for cache-resident unit cells, ~500-1900
    MC/s); 'cupy' = NVIDIA GPU via the vectorized loop (wins on large 3D volumes that fill the device);
    'jax' = the differentiable XLA scan loop (grad-through-FDTD for inverse design; XLA-fused CPU here).
    Not listed because it needs a CUDA toolkit (numba.cuda.is_available()==False here): 'numba-cuda', a
    fused GPU kernel -- the planned large-3D path once the toolkit is present."""
    bk = []
    if _HAVE_NUMBA:
        bk.append("numba")
    bk.append("numpy")
    if _have_cupy():
        bk.append("cupy")
    if _have_jax():
        bk.append("jax")
    return bk


def _resolve_backend(backend):
    """Map a requested backend -- including 'auto' and the 'cpu'/'gpu' aliases -- to a concrete available
    one, or raise a clear error (available list + install hint) for an unavailable EXPLICIT request. 'auto'
    picks the fastest CPU backend present (numba else numpy) and NEVER silently picks the GPU, because a
    cache-resident metasurface unit cell runs faster on the threaded CPU kernel than on a launch-bound GPU
    (the benchmark: numba 561-1882 MC/s vs cupy 20-120 MC/s on unit-cell grids)."""
    avail = available_backends()
    fast_cpu = "numba" if _HAVE_NUMBA else "numpy"
    name = {"auto": fast_cpu, "cpu": fast_cpu, "gpu": "cupy"}.get(str(backend).lower(), str(backend).lower())
    if name not in avail:
        hint = {"numba": "pip install numba", "cupy": "pip install cupy-cuda12x (and an NVIDIA GPU)",
                "jax": "pip install jax"}.get(name, "")
        raise RuntimeError("FDTD backend '{}' is not available here; available = {}.{}".format(
            backend, avail, (" Try: " + hint) if hint else ""))
    return name


@njit(parallel=True, fastmath=True, cache=True)
def _te2d_numba(eps_inf, wp, gam, chi3, ke, be, ce, kh, bh, ch, dx, dz, dt,
                nsteps, k_src, k_pL, k_pR, src):
    """Fused, prange-threaded 2D TE timestep (the Numba CPU kernel) -- byte-for-byte the same physics as
    _run_2d_te (Yee + semi-implicit Drude ADE + Kerr + CFS-CPML in z + PEC backing, periodic in x), but
    explicit-loop + JIT-compiled so the whole step is ONE compiled pass with no per-op overhead. Returns
    the E_y / co-located H_x probe x-lines at the left/right z-planes."""
    nx, nz = eps_inf.shape
    Ey = np.zeros((nx, nz)); Hx = np.zeros((nx, nz)); Hz = np.zeros((nx, nz))
    Jy = np.zeros((nx, nz)); psi_hxz = np.zeros((nx, nz)); psi_eyz = np.zeros((nx, nz))
    eyL = np.empty((nsteps, nx)); hxL = np.empty((nsteps, nx))
    eyR = np.empty((nsteps, nx)); hxR = np.empty((nsteps, nx))
    cmu = dt / MU0
    e0dt = EPS0 / dt
    for n in range(nsteps):
        # H update (parallel over x)
        for i in prange(nx):
            ip1 = i + 1 if i + 1 < nx else 0
            for k in range(nz - 1):
                d = (Ey[i, k + 1] - Ey[i, k]) / dz
                psi_hxz[i, k] = bh[k] * psi_hxz[i, k] + ch[k] * d
                Hx[i, k] += cmu * (d / kh[k] + psi_hxz[i, k])
            for k in range(nz):
                Hz[i, k] += -cmu * (Ey[ip1, k] - Ey[i, k]) / dx
        # E update (parallel over x; interior z), Drude ADE + Kerr + CPML
        for i in prange(nx):
            im1 = i - 1 if i - 1 >= 0 else nx - 1
            for k in range(1, nz - 1):
                dHxz = (Hx[i, k] - Hx[i, k - 1]) / dz
                psi_eyz[i, k] = be[k] * psi_eyz[i, k] + ce[k] * dHxz
                curl = dHxz / ke[k] + psi_eyz[i, k] - (Hz[i, k] - Hz[im1, k]) / dx
                aJ = (1.0 - gam[i, k] * dt / 2.0) / (1.0 + gam[i, k] * dt / 2.0)
                bJ = (EPS0 * wp[i, k] ** 2 * dt / 2.0) / (1.0 + gam[i, k] * dt / 2.0)
                eps_eff = eps_inf[i, k] + chi3[i, k] * Ey[i, k] ** 2
                denom = e0dt * eps_eff + bJ / 2.0
                eyo = Ey[i, k]
                eyn = (e0dt * eps_eff * eyo + curl - 0.5 * (1.0 + aJ) * Jy[i, k] - 0.5 * bJ * eyo) / denom
                Jy[i, k] = aJ * Jy[i, k] + bJ * (eyn + eyo)
                Ey[i, k] = eyn
        for i in prange(nx):                                 # soft source + PEC backing
            Ey[i, k_src] += src[n]
            Ey[i, 0] = 0.0; Ey[i, nz - 1] = 0.0
        for i in prange(nx):                                 # co-located probes (H_x averaged to E plane)
            eyL[n, i] = Ey[i, k_pL]; hxL[n, i] = 0.5 * (Hx[i, k_pL] + Hx[i, k_pL - 1])
            eyR[n, i] = Ey[i, k_pR]; hxR[n, i] = 0.5 * (Hx[i, k_pR] + Hx[i, k_pR - 1])
    return eyL, hxL, eyR, hxR


@dataclass
class FDTD2DResult:
    freqs_Hz: np.ndarray
    R0: np.ndarray              # 0-order (specular) reflectance from the x-mean field (== 1D/TMM)
    T0: np.ndarray             # 0-order (specular) transmittance from the x-mean field
    R_flux: np.ndarray         # TOTAL reflectance from the Poynting flux (all diffraction orders)
    T_flux: np.ndarray         # TOTAL transmittance from the Poynting flux (all diffraction orders)
    band: np.ndarray            # boolean mask of the well-excited frequency band


def _cpml_z(nz, dz, dt, npml, m=3.0, ma=1.0, kappa_max=5.0, alpha_max=0.2, R0=1.0e-6):
    """CFS-CPML stretched-coordinate coefficients along z (the propagation axis; x is periodic so needs
    no PML). Returns (kappa, b, c) on the E-grid (z=k*dz) and the H-grid (z=(k+1/2)*dz). Roden-Gedney:
    sigma/kappa graded polynomially over the outer `npml` cells each end, alpha (CFS) graded the other
    way; b=exp(-(sigma/kappa+alpha)dt/eps0), c=sigma/(sigma*kappa+kappa^2*alpha)(b-1). Outside the PML
    sigma=alpha=0 -> b=1,c=0 -> plain FDTD."""
    eta0 = np.sqrt(MU0 / EPS0)
    sig_max = -(m + 1.0) * np.log(R0) / (2.0 * eta0 * npml * dz)

    def _coeffs(zpos):                                   # zpos: cell-index position along z (nz,)
        d_lo = np.clip(npml - zpos, 0.0, None)           # depth into the low-z PML (cells)
        d_hi = np.clip(zpos - (nz - 1 - npml), 0.0, None)  # depth into the high-z PML
        rho = np.clip(np.maximum(d_lo, d_hi) / npml, 0.0, 1.0)
        sig = sig_max * rho ** m
        kap = 1.0 + (kappa_max - 1.0) * rho ** m
        alp = alpha_max * (1.0 - rho) ** ma
        b = np.exp(-(sig / kap + alp) * dt / EPS0)
        denom = sig * kap + kap ** 2 * alp
        c = np.where(denom > 0.0, sig / np.where(denom > 0.0, denom, 1.0) * (b - 1.0), 0.0)
        return kap, b, c
    ke, be, ce = _coeffs(np.arange(nz, dtype=float))
    kh, bh, ch = _coeffs(np.arange(nz, dtype=float) + 0.5)
    return (ke, be, ce), (kh, bh, ch)


def _run_2d_te(eps_inf, wp, gam, chi3, dx, dz, dt, nsteps, k_src, k_pL, k_pR, src, cpml, xp=np):
    """One 2D TE pass over a cell-wise (nx,nz) (eps_inf, wp, gamma, chi3) profile. Periodic in x (roll),
    CFS-CPML absorbing layers + PEC backing in z. Records the E_y and H_x x-lines at the left/right
    z-probe planes (for both the x-mean 0-order and the Poynting-flux R/T). Semi-implicit Drude ADE +
    instantaneous Kerr. `cpml` = ((kappa_e,b_e,c_e),(kappa_h,b_h,c_h)) from _cpml_z (z-broadcast)."""
    nx, nz = eps_inf.shape
    (ke, be, ce), (kh, bh, ch) = cpml
    ke = xp.asarray(ke); be = xp.asarray(be); ce = xp.asarray(ce)
    kh = xp.asarray(kh); bh = xp.asarray(bh); ch = xp.asarray(ch)
    Ey = xp.zeros((nx, nz))
    Hx = xp.zeros((nx, nz))                 # Hx[i,k] at (i, k+1/2)
    Hz = xp.zeros((nx, nz))                 # Hz[i,k] at (i+1/2, k)
    Jy = xp.zeros((nx, nz))                 # Drude polarization current (on E_y)
    psi_hxz = xp.zeros((nx, nz))            # CPML convolution memory for dEy/dz (H-grid)
    psi_eyz = xp.zeros((nx, nz))            # CPML convolution memory for dHx/dz (E-grid)
    aJ = (1.0 - gam * dt / 2.0) / (1.0 + gam * dt / 2.0)
    bJ = (EPS0 * wp ** 2 * dt / 2.0) / (1.0 + gam * dt / 2.0)
    eyL = xp.empty((nsteps, nx)); hxL = xp.empty((nsteps, nx))
    eyR = xp.empty((nsteps, nx)); hxR = xp.empty((nsteps, nx))
    cmu = dt / MU0
    for n in range(nsteps):
        # H update: dHx/dt = (1/mu0) (CPML-stretched dEy/dz) ; dHz/dt = -(1/mu0) dEy/dx (periodic x)
        dEy_dz = (Ey[:, 1:] - Ey[:, :-1]) / dz                      # at H positions k=0..nz-2
        psi_hxz[:, :-1] = bh[:-1] * psi_hxz[:, :-1] + ch[:-1] * dEy_dz
        Hx[:, :-1] += cmu * (dEy_dz / kh[:-1] + psi_hxz[:, :-1])
        Hz += -cmu * (xp.roll(Ey, -1, axis=0) - Ey) / dx
        # curl_y(H) = (CPML-stretched dHx/dz) - dHz/dx at the E_y points
        dHx_dz = (Hx[:, 1:] - Hx[:, :-1]) / dz                      # at E positions k=1..nz-1
        psi_eyz[:, 1:] = be[1:] * psi_eyz[:, 1:] + ce[1:] * dHx_dz
        curl = xp.zeros((nx, nz))
        curl[:, 1:] += dHx_dz / ke[1:] + psi_eyz[:, 1:]
        curl -= (Hz - xp.roll(Hz, 1, axis=0)) / dx
        # E update: eps0 eps_eff dEy/dt = curl - J, semi-implicit Drude + instantaneous Kerr
        eps_eff = eps_inf + chi3 * Ey ** 2
        denom = EPS0 * eps_eff / dt + bJ / 2.0
        Eynew = (EPS0 * eps_eff / dt * Ey + curl - 0.5 * (1.0 + aJ) * Jy - 0.5 * bJ * Ey) / denom
        Jy = aJ * Jy + bJ * (Eynew + Ey)
        Eynew[:, k_src] += src[n]            # soft plane source (uniform in x -> normal-incidence plane wave)
        Eynew[:, 0] = 0.0; Eynew[:, -1] = 0.0  # PEC backing the CPML
        Ey = Eynew
        # record E_y and H_x at the probe planes; AVERAGE H_x (at k+/-1/2) onto the E_y plane (at k) so
        # the Poynting flux E_y*H_x co-locates spatially -- else the half-cell z-offset carries a
        # per-diffraction-order phase (each order has a different k_z) that does NOT cancel in the ratio.
        eyL[n] = Ey[:, k_pL]; hxL[n] = 0.5 * (Hx[:, k_pL] + Hx[:, k_pL - 1])
        eyR[n] = Ey[:, k_pR]; hxR[n] = 0.5 * (Hx[:, k_pR] + Hx[:, k_pR - 1])
    return eyL, hxL, eyR, hxR


def _run_2d_te_jax(eps_inf, wp, gam, chi3, dx, dz, dt, nsteps, k_src, k_pL, k_pR, src, cpml):
    """JAX (XLA) backend -- the SAME 2D-TE physics as _run_2d_te, expressed as a single traced, compiled
    lax.scan time loop. Two payoffs: (1) it is DIFFERENTIABLE end-to-end, so a downstream jax.grad gives
    d(R,T)/d(geometry/material) for gradient-based inverse design; (2) XLA fuses the whole step (no
    per-op Python overhead) on CPU and, on a JAX-GPU build (WSL2 on Windows), on the device. Functional
    (immutable .at[]) updates replace the in-place ones; float64 is forced so it matches the reference.
    Returns the four probe x-lines as JAX arrays (the dispatcher converts to NumPy for the FFT/R-T
    extraction; staying in JAX lets a caller jax.grad a scalar objective straight through the time loop,
    the inverse-design path -- see validation/fdtd_2d_autodiff.py). cpml from _cpml_z."""
    import jax
    jax.config.update("jax_enable_x64", True)               # FDTD needs float64 to match the reference
    import jax.numpy as jnp
    from jax import lax
    (ke, be, ce), (kh, bh, ch) = cpml
    ke, be, ce = jnp.asarray(ke), jnp.asarray(be), jnp.asarray(ce)
    kh, bh, ch = jnp.asarray(kh), jnp.asarray(bh), jnp.asarray(ch)
    eps_inf = jnp.asarray(eps_inf); chi3 = jnp.asarray(chi3)
    gam = jnp.asarray(gam); wp = jnp.asarray(wp)
    aJ = (1.0 - gam * dt / 2.0) / (1.0 + gam * dt / 2.0)
    bJ = (EPS0 * wp ** 2 * dt / 2.0) / (1.0 + gam * dt / 2.0)
    nx, nz = eps_inf.shape
    cmu = dt / MU0

    def step(carry, src_n):
        Ey, Hx, Hz, Jy, psi_h, psi_e = carry
        dEy_dz = (Ey[:, 1:] - Ey[:, :-1]) / dz
        psi_h = psi_h.at[:, :-1].set(bh[:-1] * psi_h[:, :-1] + ch[:-1] * dEy_dz)
        Hx = Hx.at[:, :-1].add(cmu * (dEy_dz / kh[:-1] + psi_h[:, :-1]))
        Hz = Hz - cmu * (jnp.roll(Ey, -1, axis=0) - Ey) / dx
        dHx_dz = (Hx[:, 1:] - Hx[:, :-1]) / dz
        psi_e = psi_e.at[:, 1:].set(be[1:] * psi_e[:, 1:] + ce[1:] * dHx_dz)
        curl = jnp.zeros((nx, nz))
        curl = curl.at[:, 1:].add(dHx_dz / ke[1:] + psi_e[:, 1:])
        curl = curl - (Hz - jnp.roll(Hz, 1, axis=0)) / dx
        eps_eff = eps_inf + chi3 * Ey ** 2
        denom = EPS0 * eps_eff / dt + bJ / 2.0
        Eyn = (EPS0 * eps_eff / dt * Ey + curl - 0.5 * (1.0 + aJ) * Jy - 0.5 * bJ * Ey) / denom
        Jy = aJ * Jy + bJ * (Eyn + Ey)
        Eyn = Eyn.at[:, k_src].add(src_n)                   # soft plane source
        Eyn = Eyn.at[:, 0].set(0.0).at[:, nz - 1].set(0.0)  # PEC backing the CPML
        out = (Eyn[:, k_pL], 0.5 * (Hx[:, k_pL] + Hx[:, k_pL - 1]),
               Eyn[:, k_pR], 0.5 * (Hx[:, k_pR] + Hx[:, k_pR - 1]))
        return (Eyn, Hx, Hz, Jy, psi_h, psi_e), out

    z0 = jnp.zeros((nx, nz))
    _, (eyL, hxL, eyR, hxR) = lax.scan(step, (z0, z0, z0, z0, z0, z0), jnp.asarray(src))
    return eyL, hxL, eyR, hxR                               # JAX arrays (differentiable); dispatcher -> NumPy


def _flux(ey, hx):
    """Per-frequency time-averaged +z Poynting power S_z = -Re(E_y H_x*) summed over x, from the rfft
    of the recorded probe x-lines (shape (nsteps, nx)). Half-cell / half-step staggering offsets are
    common to numerator and the incident reference, so they cancel in the R/T ratio."""
    Ey = np.fft.rfft(ey, axis=0)
    Hx = np.fft.rfft(hx, axis=0)
    return -np.sum(np.real(Ey * np.conj(Hx)), axis=1)        # (nfreq,) signed z-power per frequency


def _dispatch_2d_te(name, eps_inf, wp, gam, chi3, dx, dz, dt, nsteps, k_src, k_pL, k_pR, src, cpml, xp=np):
    """Run ONE 2D-TE pass on the named backend and return the four probe x-lines as NumPy arrays, so the
    downstream FFT / R-T extraction stays backend-agnostic. 'numba' = the fused threaded CPU kernel;
    'jax' = the differentiable XLA scan; 'numpy'/'cupy' = the vectorized reference loop on the chosen
    array module (an explicit power-user `xp` is honored even for 'numpy', preserving the old xp=cupy API)."""
    (ke, be, ce), (kh, bh, ch) = cpml
    if name == "numba":
        return _te2d_numba(eps_inf, wp, gam, chi3, ke, be, ce, kh, bh, ch, dx, dz, dt,
                           nsteps, k_src, k_pL, k_pR, src)
    if name == "jax":
        out = _run_2d_te_jax(eps_inf, wp, gam, chi3, dx, dz, dt, nsteps, k_src, k_pL, k_pR, src, cpml)
        return tuple(np.asarray(v) for v in out)            # JAX arrays -> NumPy for the FFT/R-T stage
    if name == "cupy" and xp is np:
        import cupy as xp                                    # backend='cupy' auto-selects the device module
    a = tuple(xp.asarray(v) for v in (eps_inf, wp, gam, chi3))
    out = _run_2d_te(*a, dx, dz, dt, nsteps, k_src, k_pL, k_pR, xp.asarray(src), cpml, xp)
    to_np = (lambda v: np.asarray(v.get()) if hasattr(v, "get") else np.asarray(v))
    return tuple(to_np(v) for v in out)


def solve_fdtd_2d(layers: List[FDTDLayer], *, period_x_m: float, nx: Optional[int] = None,
                  lateral_eps_inf: Optional[np.ndarray] = None,
                  lambda_min_m: float, lambda_max_m: float, resolution: int = 40,
                  courant: float = 0.5, n_pad_wave: float = 6.0, settle: float = 12.0,
                  kerr: bool = False, source_amp: float = 1.0, npml: int = 12,
                  backend: str = "numpy", xp=np) -> FDTD2DResult:
    """Broadband R(f)/T(f) of a periodic (period_x_m) 2D-TE unit cell at NORMAL incidence. `layers`
    is the through-stack (z) profile (vacuum super/substrate); supply `lateral_eps_inf` (shape
    (nx, n_layer_cells) or a callable building the (nx,nz) eps_inf) to make a laterally-structured
    grating, else the stack is laterally UNIFORM (and the result reduces to the 1D solver / TMM).
    Returns both the 0-order (specular, x-mean) and the total-flux (all-diffraction-order) R/T.

    backend selects the compute kernel (see available_backends()): 'auto' (default-fastest CPU present),
    'numpy' (reference), 'numba' (fused threaded CPU -- fastest for unit cells), 'cupy' (NVIDIA GPU),
    'jax' (differentiable XLA), or the 'cpu'/'gpu' aliases. All backends are byte-for-byte equivalent on
    R/T (validation/fdtd_2d_reduces.py GATE D); xp is an advanced override for a custom array module."""
    f_min, f_max = C_LIGHT / lambda_max_m, C_LIGHT / lambda_min_m
    f_c = 0.5 * (f_min + f_max)
    w_min = 2.0 * np.pi * f_min

    def _n_band_max(L):
        eps = complex(L.eps_inf)
        if L.drude_wp_rad_s > 0.0:
            eps = eps - L.drude_wp_rad_s ** 2 / (w_min ** 2 + 1j * L.drude_gamma_rad_s * w_min)
        return abs(np.sqrt(eps))
    n_max = max(1.0, max(_n_band_max(L) for L in layers))
    dz = lambda_min_m / (resolution * n_max)
    if nx is None:
        nx = max(4, int(round(period_x_m / dz)))
    dx = period_x_m / nx
    # 2D CFL: dt <= courant / (c sqrt(1/dx^2 + 1/dz^2))
    dt = courant / (C_LIGHT * np.sqrt(1.0 / dx ** 2 + 1.0 / dz ** 2))

    pad = n_pad_wave * lambda_max_m
    z_struct = float(sum(L.thickness_m for L in layers))
    Lz = 2.0 * pad + z_struct
    nz = int(round(Lz / dz)) + 1

    # z-profile, replicated over nx columns (laterally uniform unless lateral_eps_inf given)
    eps_inf = np.ones((nx, nz)); wp = np.zeros((nx, nz)); gam = np.zeros((nx, nz)); chi3 = np.zeros((nx, nz))
    zc = (np.arange(nz) + 0.5) * dz
    z = pad
    for L in layers:
        m = (zc >= z) & (zc < z + L.thickness_m)
        eps_inf[:, m] = L.eps_inf
        wp[:, m] = L.drude_wp_rad_s
        gam[:, m] = L.drude_gamma_rad_s
        if kerr:
            chi3[:, m] = L.chi3_m2_V2
        z += L.thickness_m
    if lateral_eps_inf is not None:
        # a laterally-structured grating: overwrite eps_inf in the structure band with the (nx, *)
        # lateral pattern (callable(nx,nz)->array, or an (nx,nz) array applied in the structure region)
        lat = lateral_eps_inf(nx, nz, zc, pad, z_struct) if callable(lateral_eps_inf) else np.asarray(lateral_eps_inf)
        eps_inf = np.asarray(lat, dtype=float)

    k_src = max(2, int(round((0.35 * pad) / dz)))
    k_pL = int(round((0.7 * pad) / dz))
    k_pR = int(round((pad + z_struct + 0.3 * pad) / dz))

    tau = 1.0 / (np.pi * (f_max - f_min))
    t0 = settle * tau
    nsteps = int(round((2.0 * t0 + (Lz / C_LIGHT) * 4.0 + 200 * tau) / dt))
    tgrid = np.arange(nsteps) * dt
    src = source_amp * np.exp(-((tgrid - t0) / tau) ** 2) * np.cos(2.0 * np.pi * f_c * (tgrid - t0))

    (ke, be, ce), (kh, bh, ch) = _cpml_z(nz, dz, dt, npml)   # CFS-CPML coeffs in z (material-independent)
    cpml = ((ke, be, ce), (kh, bh, ch))
    name = _resolve_backend(backend)                         # 'auto'/'cpu'/'gpu'/explicit -> concrete backend
    one = np.ones((nx, nz)); zero = np.zeros((nx, nz))

    def run(ei, w, g_, c3):
        return _dispatch_2d_te(name, ei, w, g_, c3, dx, dz, dt, nsteps, k_src, k_pL, k_pR, src, cpml, xp)

    eyL_i, hxL_i, eyR_i, hxR_i = run(one, zero, zero, zero)  # vacuum reference run
    eyL_t, hxL_t, eyR_t, hxR_t = run(eps_inf, wp, gam, chi3)  # structure run

    f = np.fft.rfftfreq(nsteps, dt)
    # ---- 0-order (specular) R/T from the x-MEAN field (== the 1D two-run method) ----
    mL_inc = np.fft.rfft(eyL_i.mean(axis=1)); mR_inc = np.fft.rfft(eyR_i.mean(axis=1))
    mRefl = np.fft.rfft((eyL_t - eyL_i).mean(axis=1)); mTrans = np.fft.rfft(eyR_t.mean(axis=1))
    with np.errstate(divide="ignore", invalid="ignore"):
        R0 = np.abs(mRefl / mL_inc) ** 2
        T0 = np.abs(mTrans / mR_inc) ** 2
    # ---- TOTAL R/T from the Poynting flux (all diffraction orders) ----
    P_inc = _flux(eyL_i, hxL_i)
    P_refl = _flux(eyL_t - eyL_i, hxL_t - hxL_i)
    P_trans = _flux(eyR_t, hxR_t)
    with np.errstate(divide="ignore", invalid="ignore"):
        R_flux = np.abs(P_refl) / np.abs(P_inc)
        T_flux = np.abs(P_trans) / np.abs(P_inc)
    band = (f >= f_min) & (f <= f_max) & (np.abs(mL_inc) > 0.05 * np.max(np.abs(mL_inc)))
    return FDTD2DResult(freqs_Hz=f, R0=R0, T0=T0, R_flux=R_flux, T_flux=T_flux, band=band)
