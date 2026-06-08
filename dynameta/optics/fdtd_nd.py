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
                nsteps, k_src, k_pL, k_pR, src, C1, C2, C3, has_lor):
    """Fused, prange-threaded 2D TE timestep (the Numba CPU kernel) -- byte-for-byte the same physics as
    _run_2d_te (Yee + semi-implicit Drude ADE + Kerr + Lorentz ADE + CFS-CPML in z + PEC backing, periodic
    in x), but explicit-loop + JIT-compiled so the whole step is ONE compiled pass with no per-op overhead.
    C1,C2,C3 = per-cell Lorentz ADE coefficients; has_lor gates the extra pole. Returns the E_y / co-located
    H_x probe x-lines at the left/right z-planes."""
    nx, nz = eps_inf.shape
    Ey = np.zeros((nx, nz)); Hx = np.zeros((nx, nz)); Hz = np.zeros((nx, nz))
    Jy = np.zeros((nx, nz)); psi_hxz = np.zeros((nx, nz)); psi_eyz = np.zeros((nx, nz))
    PL = np.zeros((nx, nz)); PLp = np.zeros((nx, nz))           # Lorentz polarization (now / previous)
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
        # E update (parallel over x; interior z), Drude ADE + Kerr + Lorentz ADE + CPML
        for i in prange(nx):
            im1 = i - 1 if i - 1 >= 0 else nx - 1
            for k in range(1, nz - 1):
                dHxz = (Hx[i, k] - Hx[i, k - 1]) / dz
                psi_eyz[i, k] = be[k] * psi_eyz[i, k] + ce[k] * dHxz
                curl = dHxz / ke[k] + psi_eyz[i, k] - (Hz[i, k] - Hz[im1, k]) / dx
                eyo = Ey[i, k]
                if has_lor:                                    # Lorentz ADE: dPL/dt enters the E-update
                    pln = C1[i, k] * PL[i, k] + C2[i, k] * PLp[i, k] + C3[i, k] * eyo
                    curl = curl - (pln - PL[i, k]) / dt
                    PLp[i, k] = PL[i, k]; PL[i, k] = pln
                aJ = (1.0 - gam[i, k] * dt / 2.0) / (1.0 + gam[i, k] * dt / 2.0)
                bJ = (EPS0 * wp[i, k] ** 2 * dt / 2.0) / (1.0 + gam[i, k] * dt / 2.0)
                eps_eff = eps_inf[i, k] + chi3[i, k] * Ey[i, k] ** 2
                denom = e0dt * eps_eff + bJ / 2.0
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
    r0: Optional[np.ndarray] = None   # COMPLEX 0-order reflection coeff, phase de-embedded to the front face
    t0: Optional[np.ndarray] = None   # COMPLEX 0-order transmission coeff, de-embedded across the structure


def _cpml_z(nz, dz, dt, npml, n_super=1.0, n_sub=1.0, m=3.0, ma=1.0, kappa_max=5.0, alpha_max=0.2, R0=1.0e-6):
    """CFS-CPML stretched-coordinate coefficients along z (the propagation axis; x is periodic so needs
    no PML). Returns (kappa, b, c) on the E-grid (z=k*dz) and the H-grid (z=(k+1/2)*dz). Roden-Gedney:
    sigma/kappa graded polynomially over the outer `npml` cells each end, alpha (CFS) graded the other
    way; b=exp(-(sigma/kappa+alpha)dt/eps0), c=sigma/(sigma*kappa+kappa^2*alpha)(b-1). Outside the PML
    sigma=alpha=0 -> b=1,c=0 -> plain FDTD.

    n_super / n_sub (default 1 = vacuum) impedance-match the conductivity to the END MEDIUM each PML
    terminates: the matched conductivity for a medium of refractive index n is sigma ~ n / eta0 (wave
    impedance eta0/n), so the low-z PML scales sig_max by n_super and the high-z PML by n_sub. With the
    defaults the per-cell scale is 1.0 everywhere -> coefficients are byte-identical to the vacuum case."""
    eta0 = np.sqrt(MU0 / EPS0)
    sig_max = -(m + 1.0) * np.log(R0) / (2.0 * eta0 * npml * dz)

    def _coeffs(zpos):                                   # zpos: cell-index position along z (nz,)
        d_lo = np.clip(npml - zpos, 0.0, None)           # depth into the low-z PML (cells)
        d_hi = np.clip(zpos - (nz - 1 - npml), 0.0, None)  # depth into the high-z PML
        rho = np.clip(np.maximum(d_lo, d_hi) / npml, 0.0, 1.0)
        nfac = np.where(d_lo >= d_hi, n_super, n_sub)    # which end-medium this PML terminates (low->super)
        sig = sig_max * nfac * rho ** m
        kap = 1.0 + (kappa_max - 1.0) * rho ** m
        alp = alpha_max * (1.0 - rho) ** ma
        b = np.exp(-(sig / kap + alp) * dt / EPS0)
        denom = sig * kap + kap ** 2 * alp
        c = np.where(denom > 0.0, sig / np.where(denom > 0.0, denom, 1.0) * (b - 1.0), 0.0)
        return kap, b, c
    ke, be, ce = _coeffs(np.arange(nz, dtype=float))
    kh, bh, ch = _coeffs(np.arange(nz, dtype=float) + 0.5)
    return (ke, be, ce), (kh, bh, ch)


def _run_2d_te(eps_inf, wp, gam, chi3, dx, dz, dt, nsteps, k_src, k_pL, k_pR, src, cpml, xp=np, lor=None):
    """One 2D TE pass over a cell-wise (nx,nz) (eps_inf, wp, gamma, chi3) profile. Periodic in x (roll),
    CFS-CPML absorbing layers + PEC backing in z. Records the E_y and H_x x-lines at the left/right
    z-probe planes (for both the x-mean 0-order and the Poynting-flux R/T). Semi-implicit Drude ADE +
    instantaneous Kerr. `cpml` = ((kappa_e,b_e,c_e),(kappa_h,b_h,c_h)) from _cpml_z (z-broadcast).
    `lor` = (C1,C2,C3) per-cell Lorentz ADE coefficients (a second polarization PL) or None (no pole)."""
    nx, nz = eps_inf.shape
    (ke, be, ce), (kh, bh, ch) = cpml
    ke = xp.asarray(ke); be = xp.asarray(be); ce = xp.asarray(ce)
    kh = xp.asarray(kh); bh = xp.asarray(bh); ch = xp.asarray(ch)
    do_lor = lor is not None
    if do_lor:
        C1, C2, C3 = (xp.asarray(lor[0]), xp.asarray(lor[1]), xp.asarray(lor[2]))
        PL = xp.zeros((nx, nz)); PLp = xp.zeros((nx, nz))       # Lorentz polarization (now / previous step)
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
        # Lorentz ADE: PL^{n+1} = C1 PL^n + C2 PL^{n-1} + C3 E^n; its current dPL/dt enters the E-update
        if do_lor:
            PLnew = C1 * PL + C2 * PLp + C3 * Ey
            curl = curl - (PLnew - PL) / dt
            PLp = PL; PL = PLnew
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


def _run_2d_te_jax(eps_inf, wp, gam, chi3, dx, dz, dt, nsteps, k_src, k_pL, k_pR, src, cpml, lor=None):
    """JAX (XLA) backend -- the SAME 2D-TE physics as _run_2d_te, expressed as a single traced, compiled
    lax.scan time loop. Two payoffs: (1) it is DIFFERENTIABLE end-to-end, so a downstream jax.grad gives
    d(R,T)/d(geometry/material) for gradient-based inverse design; (2) XLA fuses the whole step (no
    per-op Python overhead) on CPU and, on a JAX-GPU build (WSL2 on Windows), on the device. Functional
    (immutable .at[]) updates replace the in-place ones; float64 is forced so it matches the reference.
    Returns the four probe x-lines as JAX arrays (the dispatcher converts to NumPy for the FFT/R-T
    extraction; staying in JAX lets a caller jax.grad a scalar objective straight through the time loop,
    the inverse-design path -- see validation/fdtd_2d_autodiff.py). cpml from _cpml_z. `lor`=(C1,C2,C3)
    per-cell Lorentz ADE coefficients (a second polarization PL in the carry) or None (no pole)."""
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
    do_lor = lor is not None
    if do_lor:
        C1, C2, C3 = jnp.asarray(lor[0]), jnp.asarray(lor[1]), jnp.asarray(lor[2])

    def step(carry, src_n):
        Ey, Hx, Hz, Jy, psi_h, psi_e, PL, PLp = carry
        dEy_dz = (Ey[:, 1:] - Ey[:, :-1]) / dz
        psi_h = psi_h.at[:, :-1].set(bh[:-1] * psi_h[:, :-1] + ch[:-1] * dEy_dz)
        Hx = Hx.at[:, :-1].add(cmu * (dEy_dz / kh[:-1] + psi_h[:, :-1]))
        Hz = Hz - cmu * (jnp.roll(Ey, -1, axis=0) - Ey) / dx
        dHx_dz = (Hx[:, 1:] - Hx[:, :-1]) / dz
        psi_e = psi_e.at[:, 1:].set(be[1:] * psi_e[:, 1:] + ce[1:] * dHx_dz)
        curl = jnp.zeros((nx, nz))
        curl = curl.at[:, 1:].add(dHx_dz / ke[1:] + psi_e[:, 1:])
        curl = curl - (Hz - jnp.roll(Hz, 1, axis=0)) / dx
        if do_lor:                                          # Lorentz ADE: dPL/dt enters the E-update
            PLnew = C1 * PL + C2 * PLp + C3 * Ey
            curl = curl - (PLnew - PL) / dt
            PLp, PL = PL, PLnew
        eps_eff = eps_inf + chi3 * Ey ** 2
        denom = EPS0 * eps_eff / dt + bJ / 2.0
        Eyn = (EPS0 * eps_eff / dt * Ey + curl - 0.5 * (1.0 + aJ) * Jy - 0.5 * bJ * Ey) / denom
        Jy = aJ * Jy + bJ * (Eyn + Ey)
        Eyn = Eyn.at[:, k_src].add(src_n)                   # soft plane source
        Eyn = Eyn.at[:, 0].set(0.0).at[:, nz - 1].set(0.0)  # PEC backing the CPML
        out = (Eyn[:, k_pL], 0.5 * (Hx[:, k_pL] + Hx[:, k_pL - 1]),
               Eyn[:, k_pR], 0.5 * (Hx[:, k_pR] + Hx[:, k_pR - 1]))
        return (Eyn, Hx, Hz, Jy, psi_h, psi_e, PL, PLp), out

    z0 = jnp.zeros((nx, nz))
    _, (eyL, hxL, eyR, hxR) = lax.scan(step, tuple(z0 for _ in range(8)), jnp.asarray(src))
    return eyL, hxL, eyR, hxR                               # JAX arrays (differentiable); dispatcher -> NumPy


def _flux(ey, hx):
    """Per-frequency time-averaged +z Poynting power S_z = -Re(E_y H_x*) summed over x, from the rfft
    of the recorded probe x-lines (shape (nsteps, nx)). Half-cell / half-step staggering offsets are
    common to numerator and the incident reference, so they cancel in the R/T ratio."""
    Ey = np.fft.rfft(ey, axis=0)
    Hx = np.fft.rfft(hx, axis=0)
    return -np.sum(np.real(Ey * np.conj(Hx)), axis=1)        # (nfreq,) signed z-power per frequency


def _dispatch_2d_te(name, eps_inf, wp, gam, chi3, dx, dz, dt, nsteps, k_src, k_pL, k_pR, src, cpml, xp=np,
                    lor=None):
    """Run ONE 2D-TE pass on the named backend and return the four probe x-lines as NumPy arrays, so the
    downstream FFT / R-T extraction stays backend-agnostic. 'numba' = the fused threaded CPU kernel;
    'jax' = the differentiable XLA scan; 'numpy'/'cupy' = the vectorized reference loop on the chosen
    array module (an explicit power-user `xp` is honored even for 'numpy', preserving the old xp=cupy API).
    `lor` = (C1,C2,C3) per-cell Lorentz ADE coefficients or None (no Lorentz pole)."""
    (ke, be, ce), (kh, bh, ch) = cpml
    if name == "numba":
        has_lor = lor is not None
        z = np.zeros_like(eps_inf)
        C1, C2, C3 = (lor if has_lor else (z, z, z))
        return _te2d_numba(eps_inf, wp, gam, chi3, ke, be, ce, kh, bh, ch, dx, dz, dt,
                           nsteps, k_src, k_pL, k_pR, src, C1, C2, C3, has_lor)
    if name == "jax":
        out = _run_2d_te_jax(eps_inf, wp, gam, chi3, dx, dz, dt, nsteps, k_src, k_pL, k_pR, src, cpml, lor)
        return tuple(np.asarray(v) for v in out)            # JAX arrays -> NumPy for the FFT/R-T stage
    if name == "cupy" and xp is np:
        import cupy as xp                                    # backend='cupy' auto-selects the device module
    a = tuple(xp.asarray(v) for v in (eps_inf, wp, gam, chi3))
    out = _run_2d_te(*a, dx, dz, dt, nsteps, k_src, k_pL, k_pR, xp.asarray(src), cpml, xp, lor)
    to_np = (lambda v: np.asarray(v.get()) if hasattr(v, "get") else np.asarray(v))
    return tuple(to_np(v) for v in out)


def solve_fdtd_2d(layers: List[FDTDLayer], *, period_x_m: float, nx: Optional[int] = None,
                  lateral_eps_inf: Optional[np.ndarray] = None,
                  lambda_min_m: float, lambda_max_m: float, resolution: int = 40,
                  courant: float = 0.5, n_pad_wave: float = 6.0, settle: float = 12.0,
                  kerr: bool = False, source_amp: float = 1.0, npml: int = 12,
                  n_super: float = 1.0, n_sub: float = 1.0,
                  backend: str = "numpy", xp=np) -> FDTD2DResult:
    """Broadband R(f)/T(f) of a periodic (period_x_m) 2D-TE unit cell at NORMAL incidence. `layers`
    is the through-stack (z) profile; supply `lateral_eps_inf` (shape (nx, n_layer_cells) or a callable
    building the (nx,nz) eps_inf) to make a laterally-structured grating, else the stack is laterally
    UNIFORM (and the result reduces to the 1D solver / TMM). Returns both the 0-order (specular, x-mean)
    and the total-flux (all-diffraction-order) R/T.

    n_super / n_sub (default 1 = vacuum) are the lossless semi-infinite superstrate / substrate indices
    (metasurface-on-glass etc.): the z-pad regions are filled with n_super^2 / n_sub^2, the CPML is
    impedance-matched per end, and the incident reference is a homogeneous-superstrate run so R/T are
    correctly normalized (T carries the n_sub/n_super flux ratio). Reduces byte-identically to vacuum
    at n_super=n_sub=1.

    backend selects the compute kernel (see available_backends()): 'auto' (default-fastest CPU present),
    'numpy' (reference), 'numba' (fused threaded CPU -- fastest for unit cells), 'cupy' (NVIDIA GPU),
    'jax' (differentiable XLA), or the 'cpu'/'gpu' aliases. All backends are byte-for-byte equivalent on
    R/T (validation/fdtd_2d_reduces.py GATE D); xp is an advanced override for a custom array module."""
    if abs(complex(n_super).imag) > 1e-9 or abs(complex(n_sub).imag) > 1e-9:   # mirror the FEM guard
        raise NotImplementedError("solve_fdtd_2d: R/T and the energy budget are defined only for LOSSLESS "
                                  "end media (Im(n)=0); got n_super={}, n_sub={} (use the FEM/TMM solver "
                                  "for an absorbing incidence/exit medium).".format(n_super, n_sub))
    f_min, f_max = C_LIGHT / lambda_max_m, C_LIGHT / lambda_min_m
    f_c = 0.5 * (f_min + f_max)
    w_band = 2.0 * np.pi * np.linspace(f_min, f_max, 9)      # sample the band (a Lorentz peak may be in-band)

    def _n_band_max(L):
        return max(abs(np.sqrt(L.eps_at(w))) for w in w_band)
    n_max = max(1.0, n_super, n_sub, max(_n_band_max(L) for L in layers))
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
    lw0 = np.zeros((nx, nz)); lgam = np.zeros((nx, nz)); ldeps = np.zeros((nx, nz))  # Lorentz pole per cell
    zc = (np.arange(nz) + 0.5) * dz
    # fill the semi-infinite super/substrate pads with the end-media permittivity (so the incident wave is
    # truly in n_super and the structure sees the n_sub backing); vacuum (n=1) leaves this as ones
    eps_inf[:, zc < pad] = n_super ** 2
    eps_inf[:, zc >= pad + z_struct] = n_sub ** 2
    z = pad
    for L in layers:
        m = (zc >= z) & (zc < z + L.thickness_m)
        eps_inf[:, m] = L.eps_inf
        wp[:, m] = L.drude_wp_rad_s
        gam[:, m] = L.drude_gamma_rad_s
        lw0[:, m] = L.lorentz_w0_rad_s
        lgam[:, m] = L.lorentz_gamma_rad_s
        ldeps[:, m] = L.lorentz_delta_eps
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

    # Lorentz ADE coefficients (central difference): PL^{n+1} = C1 PL^n + C2 PL^{n-1} + C3 E^n, where the
    # pole eps += d_eps w0^2/(w0^2 - w^2 - i gl w). lor is applied to the STRUCTURE run only (the reference
    # is the bare superstrate). With d_eps=0 everywhere lor=None -> the path is byte-identical to before.
    lor = None
    if np.any(ldeps != 0.0):
        den = 1.0 + lgam * dt / 2.0
        C1 = (2.0 - lw0 ** 2 * dt ** 2) / den
        C2 = (lgam * dt / 2.0 - 1.0) / den
        C3 = (EPS0 * ldeps * lw0 ** 2 * dt ** 2) / den
        lor = (C1, C2, C3)

    cpml_struct = _cpml_z(nz, dz, dt, npml, n_super, n_sub)  # PML matched to super (low z) + sub (high z)
    cpml_ref = _cpml_z(nz, dz, dt, npml, n_super, n_super)   # homogeneous-superstrate reference -> super both ends
    name = _resolve_backend(backend)                         # 'auto'/'cpu'/'gpu'/explicit -> concrete backend
    one = np.ones((nx, nz)); zero = np.zeros((nx, nz))

    def run(ei, w, g_, c3, cpml, lor=None):
        return _dispatch_2d_te(name, ei, w, g_, c3, dx, dz, dt, nsteps, k_src, k_pL, k_pR, src, cpml, xp, lor)

    # reference = homogeneous superstrate (no structure, no substrate) so the probe sees the pure incident
    # wave in n_super and the reflection subtraction is exact (same incident medium as the structure run)
    eyL_i, hxL_i, eyR_i, hxR_i = run(n_super ** 2 * one, zero, zero, zero, cpml_ref)
    eyL_t, hxL_t, eyR_t, hxR_t = run(eps_inf, wp, gam, chi3, cpml_struct, lor)  # structure run

    f = np.fft.rfftfreq(nsteps, dt)
    # ---- 0-order (specular) R/T from the x-MEAN field (== the 1D two-run method) ----
    mL_inc = np.fft.rfft(eyL_i.mean(axis=1)); mR_inc = np.fft.rfft(eyR_i.mean(axis=1))
    mRefl = np.fft.rfft((eyL_t - eyL_i).mean(axis=1)); mTrans = np.fft.rfft(eyR_t.mean(axis=1))
    k0 = 2.0 * np.pi * f / C_LIGHT
    with np.errstate(divide="ignore", invalid="ignore"):
        R0 = np.abs(mRefl / mL_inc) ** 2
        # power transmittance carries the n_sub/n_super impedance (flux) ratio: the incident reference is
        # measured in n_super, the transmitted field in n_sub (Snell power continuity)
        T0 = np.abs(mTrans / mR_inc) ** 2 * (n_sub / n_super)
        # COMPLEX 0-order coeffs. np.fft.rfft yields exp(+i w t) phasors, but the library convention is
        # exp(-i w t), so conjugate to get the physical complex amplitudes; then de-embed the probe<->face
        # propagation phase. The superstrate phase velocity is c/n_super, so r0c (referenced to the front
        # face z=pad, probe at k_pL) carries n_super in k; t0c keeps the structure traversal phase.
        r0c = np.conj(mRefl / mL_inc) * np.exp(-2j * n_super * k0 * (pad - k_pL * dz))
        t0c = np.conj(mTrans / mR_inc) * np.exp(1j * k0 * z_struct)
    # ---- TOTAL R/T from the Poynting flux (all diffraction orders) ----
    P_inc = _flux(eyL_i, hxL_i)
    P_refl = _flux(eyL_t - eyL_i, hxL_t - hxL_i)
    P_trans = _flux(eyR_t, hxR_t)
    with np.errstate(divide="ignore", invalid="ignore"):
        R_flux = np.abs(P_refl) / np.abs(P_inc)
        T_flux = np.abs(P_trans) / np.abs(P_inc)
    band = (f >= f_min) & (f <= f_max) & (np.abs(mL_inc) > 0.05 * np.max(np.abs(mL_inc)))
    return FDTD2DResult(freqs_Hz=f, R0=R0, T0=T0, R_flux=R_flux, T_flux=T_flux, band=band, r0=r0c, t0=t0c)


# =====================================================================================================
# OBLIQUE incidence (2D-TE / s-pol) via the COMPLEX-ENVELOPE (field-transform) Bloch method.
# The physical field carries a fixed transverse wavevector k_par: E_phys = Psi(x,z,t) exp(i k_par x), so
# the periodic envelope Psi is solved with d/dx -> (d/dx + i k_par) and a ZERO-PHASE periodic roll. Psi is
# complex; at k_par=0 (normal incidence) it reduces to the real solver. A FIXED k_par means the physical
# angle is frequency-dependent, theta(f) = asin(k_par c / (2 pi f)) -- a constant-k_par broadband sweep,
# the natural object for periodic FDTD (a constant-ANGLE sweep needs one run per frequency or a re-scaled
# source). Validated vs s-pol TMM at theta(f). Vacuum ends; uniform or laterally-uniform (the envelope of a
# plane wave has no transverse variation, so nx is small).
# =====================================================================================================
@dataclass
class FDTD2DObliqueResult:
    freqs_Hz: np.ndarray
    theta_deg: np.ndarray       # the frequency-dependent physical angle for the fixed k_par
    R0: np.ndarray              # specular reflectance |r|^2 (s-pol)
    T0: np.ndarray              # specular transmittance |t|^2 (s-pol; vacuum ends -> kz factors cancel)
    band: np.ndarray            # well-excited + below-the-light-line (propagating) frequency mask


def _run_2d_te_oblique(eps_inf, wp, gam, dx, dz, dt, nsteps, k_src, k_pL, k_pR, src, cpml, kx):
    """Complex-envelope oblique 2D-TE (s-pol): Ey,Hx,Hz are the PERIODIC Bloch envelope (physical field =
    envelope * exp(i kx x)), so every x-derivative gains a + i kx term and the periodic roll stays
    zero-phase. Fields are complex; at kx=0 with a real source they stay real (the normal-incidence solver).
    Semi-implicit Drude ADE; CFS-CPML + PEC in z (vacuum ends). Records complex Ey,Hx probe x-lines."""
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
        Hz += -cmu * ((np.roll(Ey, -1, axis=0) - Ey) / dx + ikx * Ey)      # dEy/dx -> d/dx + i kx
        dHx_dz = (Hx[:, 1:] - Hx[:, :-1]) / dz
        psi_eyz[:, 1:] = be[1:] * psi_eyz[:, 1:] + ce[1:] * dHx_dz
        curl = np.zeros((nx, nz), complex)
        curl[:, 1:] += dHx_dz / ke[1:] + psi_eyz[:, 1:]
        curl -= (Hz - np.roll(Hz, 1, axis=0)) / dx + ikx * Hz              # dHz/dx -> d/dx + i kx
        denom = EPS0 * eps_inf / dt + bJ / 2.0
        Eynew = (EPS0 * eps_inf / dt * Ey + curl - 0.5 * (1.0 + aJ) * Jy - 0.5 * bJ * Ey) / denom
        Jy = aJ * Jy + bJ * (Eynew + Ey)
        Eynew[:, k_src] += src[n]
        Eynew[:, 0] = 0.0; Eynew[:, -1] = 0.0
        Ey = Eynew
        eyL[n] = Ey[:, k_pL]; hxL[n] = 0.5 * (Hx[:, k_pL] + Hx[:, k_pL - 1])
        eyR[n] = Ey[:, k_pR]; hxR[n] = 0.5 * (Hx[:, k_pR] + Hx[:, k_pR - 1])
    return eyL, hxL, eyR, hxR


@njit(fastmath=True, cache=True)
def _te2d_oblique_numba(eps_inf, wp, gam, ke, be, ce, kh, bh, ch, dx, dz, dt,
                        nsteps, k_src, k_pL, k_pR, src, kx):
    """Fused, JIT-compiled COMPLEX-ENVELOPE oblique 2D-TE timestep (the Numba CPU kernel) -- the same
    physics as _run_2d_te_oblique (Bloch envelope with d/dx -> d/dx + i kx, semi-implicit Drude ADE, CFS-
    CPML + PEC in z, vacuum ends), but explicit-loop + compiled so the whole step is ONE pass. Fields are
    complex128; kx=0 reduces to the real normal-incidence response. SERIAL (not prange-threaded): the
    oblique envelope is laterally smooth so nx is small (~6-8), and threading that tiny x-extent costs more
    in per-step thread overhead than it saves (measured ~0.6x), whereas the serial JIT is ~5x over NumPy.
    Returns the complex Ey / co-located Hx probe x-lines at the left/right z-planes."""
    nx, nz = eps_inf.shape
    Ey = np.zeros((nx, nz), dtype=np.complex128); Hx = np.zeros((nx, nz), dtype=np.complex128)
    Hz = np.zeros((nx, nz), dtype=np.complex128); Jy = np.zeros((nx, nz), dtype=np.complex128)
    psi_hxz = np.zeros((nx, nz), dtype=np.complex128); psi_eyz = np.zeros((nx, nz), dtype=np.complex128)
    eyL = np.empty((nsteps, nx), dtype=np.complex128); hxL = np.empty((nsteps, nx), dtype=np.complex128)
    eyR = np.empty((nsteps, nx), dtype=np.complex128); hxR = np.empty((nsteps, nx), dtype=np.complex128)
    cmu = dt / MU0
    e0dt = EPS0 / dt
    ikx = 1j * kx
    for n in range(nsteps):
        for i in range(nx):                                    # H update
            ip1 = i + 1 if i + 1 < nx else 0
            for k in range(nz - 1):
                d = (Ey[i, k + 1] - Ey[i, k]) / dz
                psi_hxz[i, k] = bh[k] * psi_hxz[i, k] + ch[k] * d
                Hx[i, k] += cmu * (d / kh[k] + psi_hxz[i, k])
            for k in range(nz):
                Hz[i, k] += -cmu * ((Ey[ip1, k] - Ey[i, k]) / dx + ikx * Ey[i, k])
        for i in range(nx):                                    # E update (Drude ADE + CPML)
            im1 = i - 1 if i - 1 >= 0 else nx - 1
            for k in range(1, nz - 1):
                dHxz = (Hx[i, k] - Hx[i, k - 1]) / dz
                psi_eyz[i, k] = be[k] * psi_eyz[i, k] + ce[k] * dHxz
                curl = dHxz / ke[k] + psi_eyz[i, k] - ((Hz[i, k] - Hz[im1, k]) / dx + ikx * Hz[i, k])
                aJ = (1.0 - gam[i, k] * dt / 2.0) / (1.0 + gam[i, k] * dt / 2.0)
                bJ = (EPS0 * wp[i, k] ** 2 * dt / 2.0) / (1.0 + gam[i, k] * dt / 2.0)
                denom = e0dt * eps_inf[i, k] + bJ / 2.0
                eyo = Ey[i, k]
                eyn = (e0dt * eps_inf[i, k] * eyo + curl - 0.5 * (1.0 + aJ) * Jy[i, k] - 0.5 * bJ * eyo) / denom
                Jy[i, k] = aJ * Jy[i, k] + bJ * (eyn + eyo)
                Ey[i, k] = eyn
        for i in range(nx):                                    # soft source + PEC backing
            Ey[i, k_src] += src[n]
            Ey[i, 0] = 0.0 + 0.0j; Ey[i, nz - 1] = 0.0 + 0.0j
        for i in range(nx):                                    # co-located probes (Hx averaged to E plane)
            eyL[n, i] = Ey[i, k_pL]; hxL[n, i] = 0.5 * (Hx[i, k_pL] + Hx[i, k_pL - 1])
            eyR[n, i] = Ey[i, k_pR]; hxR[n, i] = 0.5 * (Hx[i, k_pR] + Hx[i, k_pR - 1])
    return eyL, hxL, eyR, hxR


def _run_oblique(name, eps_inf, wp, gam, dx, dz, dt, nsteps, k_src, k_pL, k_pR, src, cpml, kx):
    """Run ONE complex-envelope oblique 2D-TE pass on the named backend (only 'numpy' and 'numba' carry the
    complex-envelope path; the JAX/CuPy backends are normal-incidence-only). Returns the four complex probe
    x-lines. 'numba' = the fused threaded kernel; anything else = the vectorized reference loop."""
    if name == "numba":
        (ke, be, ce), (kh, bh, ch) = cpml
        return _te2d_oblique_numba(np.asarray(eps_inf, float), np.asarray(wp, float), np.asarray(gam, float),
                                   ke, be, ce, kh, bh, ch, dx, dz, dt, nsteps, k_src, k_pL, k_pR,
                                   np.asarray(src, float), kx)
    return _run_2d_te_oblique(eps_inf, wp, gam, dx, dz, dt, nsteps, k_src, k_pL, k_pR, src, cpml, kx)


def solve_fdtd_2d_oblique(layers: List[FDTDLayer], *, period_x_m: float, angle_deg: float,
                          lambda_min_m: float, lambda_max_m: float, resolution: int = 40,
                          courant: float = 0.5, n_pad_wave: float = 6.0, settle: float = 12.0,
                          source_amp: float = 1.0, npml: int = 12, nx: int = 8,
                          backend: str = "numpy") -> FDTD2DObliqueResult:
    """Broadband s-pol (TE) reflectance/transmittance of a laterally-uniform stack at OBLIQUE incidence,
    via the complex-envelope Bloch method with a FIXED transverse wavevector k_par = (2 pi / lambda_c)
    sin(angle_deg) (angle_deg the physical angle at the band centre). Because k_par is fixed, the physical
    angle varies with frequency: theta(f) = asin(k_par c/(2 pi f)); the result carries theta_deg(f) and the
    band mask excludes frequencies below the light line (k_par > w/c, evanescent). Vacuum ends. angle_deg=0
    reduces to the normal-incidence solver."""
    if any(L.lorentz_delta_eps != 0.0 for L in layers):     # the oblique kernel carries Drude only
        raise NotImplementedError("solve_fdtd_2d_oblique supports Drude dispersion only (no Lorentz pole "
                                  "yet); use solve_fdtd_2d at normal incidence for a Lorentz material.")
    f_min, f_max = C_LIGHT / lambda_max_m, C_LIGHT / lambda_min_m
    f_c = 0.5 * (f_min + f_max)
    w_band = 2.0 * np.pi * np.linspace(f_min, f_max, 9)

    def _n_band_max(L):
        return max(abs(np.sqrt(L.eps_at(w))) for w in w_band)
    n_max = max(1.0, max(_n_band_max(L) for L in layers))
    dz = lambda_min_m / (resolution * n_max)
    dx = period_x_m / nx
    dt = courant / (C_LIGHT * np.sqrt(1.0 / dx ** 2 + 1.0 / dz ** 2))
    kx = (2.0 * np.pi * f_c / C_LIGHT) * np.sin(np.radians(angle_deg))   # fixed transverse wavevector

    pad = n_pad_wave * lambda_max_m
    z_struct = float(sum(L.thickness_m for L in layers))
    Lz = 2.0 * pad + z_struct
    nz = int(round(Lz / dz)) + 1
    eps_inf = np.ones((nx, nz)); wp = np.zeros((nx, nz)); gam = np.zeros((nx, nz))
    zc = (np.arange(nz) + 0.5) * dz
    z = pad
    for L in layers:
        m = (zc >= z) & (zc < z + L.thickness_m)
        eps_inf[:, m] = L.eps_inf; wp[:, m] = L.drude_wp_rad_s; gam[:, m] = L.drude_gamma_rad_s
        z += L.thickness_m

    k_src = max(2, int(round((0.35 * pad) / dz)))
    k_pL = int(round((0.7 * pad) / dz))
    k_pR = int(round((pad + z_struct + 0.3 * pad) / dz))
    tau = 1.0 / (np.pi * (f_max - f_min))
    t0 = settle * tau
    nsteps = int(round((2.0 * t0 + (Lz / C_LIGHT) * 4.0 + 200 * tau) / dt))
    tgrid = np.arange(nsteps) * dt
    src = source_amp * np.exp(-((tgrid - t0) / tau) ** 2) * np.cos(2.0 * np.pi * f_c * (tgrid - t0))

    cpml = _cpml_z(nz, dz, dt, npml)
    one = np.ones((nx, nz)); zero = np.zeros((nx, nz))
    # 'numba' = the fused threaded complex-envelope kernel; 'auto'/'cpu' pick it when present; everything else
    # falls back to the vectorized NumPy reference (the oblique path is normal-incidence-free of jax/cupy).
    rb = _resolve_backend(backend)
    name = "numba" if (rb == "numba" and _HAVE_NUMBA) else "numpy"
    eyL_i, hxL_i, eyR_i, hxR_i = _run_oblique(name, one, zero, zero, dx, dz, dt, nsteps, k_src, k_pL, k_pR,
                                              src, cpml, kx)
    eyL_t, hxL_t, eyR_t, hxR_t = _run_oblique(name, eps_inf, wp, gam, dx, dz, dt, nsteps, k_src, k_pL, k_pR,
                                              src, cpml, kx)
    # complex envelope -> full FFT; take the positive-frequency half (the forward exp(-iwt) response).
    # rfftfreq gives the monotonic positive-frequency axis matching fft(...)[:nf] within the band.
    nf = nsteps // 2 + 1
    f = np.fft.rfftfreq(nsteps, dt)
    mean = (lambda a: np.fft.fft(a.mean(axis=1))[:nf])
    inc_L = mean(eyL_i); inc_R = mean(eyR_i)
    refl = mean(eyL_t - eyL_i); trans = mean(eyR_t)
    with np.errstate(divide="ignore", invalid="ignore"):
        R0 = np.abs(refl / inc_L) ** 2
        T0 = np.abs(trans / inc_R) ** 2
    sin_t = np.divide(kx * C_LIGHT, 2.0 * np.pi * np.maximum(f, 1e-30))   # sin theta(f) = k_par c / w
    theta = np.degrees(np.arcsin(np.clip(sin_t, -1.0, 1.0)))
    band = (f >= f_min) & (f <= f_max) & (sin_t < 0.999) & (np.abs(inc_L) > 0.05 * np.max(np.abs(inc_L)))
    return FDTD2DObliqueResult(freqs_Hz=f, theta_deg=theta, R0=R0, T0=T0, band=band)


# =====================================================================================================
# 3D: full-vector Yee engine for a 2D-periodic (x AND y) unit cell at normal incidence.
# The 2D-TE engine above is the (d/dy = 0, {Ey,Hx,Hz}) reduction of this; this carries all six field
# components so a genuinely 2D-periodic structure (pillars/holes/crosses) couples into every order.
# =====================================================================================================
@dataclass
class FDTD3DResult:
    """Broadband R(f)/T(f) of a 2D-periodic unit cell (normal incidence, y-polarized source). R0/T0 = the
    specular (0-order) co-pol from the x,y-mean field (== 1D/TMM for a laterally-uniform stack); R_flux/
    T_flux = the total over ALL (kx,ky) diffraction orders from the full Poynting flux S_z = ExHy* - EyHx*."""
    freqs_Hz: np.ndarray
    R0: np.ndarray
    T0: np.ndarray
    R_flux: np.ndarray
    T_flux: np.ndarray
    band: np.ndarray
    r0: Optional[np.ndarray] = None   # COMPLEX co-pol 0-order reflection coeff, de-embedded to the front face
    t0: Optional[np.ndarray] = None   # COMPLEX co-pol 0-order transmission coeff, de-embedded across the cell


def _flux3d(ex, ey, hx, hy):
    """Total time-averaged +z Poynting power per frequency, S_z = Re(Ex Hy* - Ey Hx*) summed over the
    whole (x,y) probe plane (Parseval: the real-space sum over the plane already includes every (kx,ky)
    diffraction order). Each probe array is (nsteps, nx, ny). Reduces to -Re(Ey Hx*) (the 2D _flux) when
    Ex = Hy = 0. The half-cell stagger is common to numerator and incident reference, so it cancels."""
    EX = np.fft.rfft(ex, axis=0); EY = np.fft.rfft(ey, axis=0)
    HX = np.fft.rfft(hx, axis=0); HY = np.fft.rfft(hy, axis=0)
    S = np.real(EX * np.conj(HY) - EY * np.conj(HX))
    return np.sum(S, axis=(1, 2))                            # (nfreq,) signed z-power per frequency


def _run_3d(eps_inf, wp, gam, chi3, dx, dy, dz, dt, nsteps, k_src, k_pL, k_pR, src, cpml, xp=np, lor=None):
    """One full-vector 3D-FDTD pass over a cell-wise (nx,ny,nz) (eps_inf, wp, gamma, chi3) profile.
    Periodic in x and y (roll = Bloch at normal incidence, zero phase), CFS-CPML + PEC backing in z.
    Standard Yee staggering: Ex@(i+1/2,j,k) Ey@(i,j+1/2,k) Ez@(i,j,k+1/2); Hx@(i,j+1/2,k+1/2)
    Hy@(i+1/2,j,k+1/2) Hz@(i+1/2,j+1/2,k). Semi-implicit Drude ADE per E-component + instantaneous Kerr
    (eps_eff = eps_inf + chi3|E|^2) + an optional Lorentz ADE per E-component (`lor`=(C1,C2,C3), a
    polarization PL{x,y,z}). Only the d/dz derivatives are CPML-stretched (x,y are periodic), so
    four psi memories: dEy/dz & dEx/dz (H update), dHx/dz & dHy/dz (E update). Records Ex,Ey,Hx,Hy on the
    left/right z-probe planes (the components that carry S_z). Returns 8 arrays of shape (nsteps,nx,ny)."""
    nx, ny, nz = eps_inf.shape
    (ke, be, ce), (kh, bh, ch) = cpml
    r = (lambda a: xp.asarray(a).reshape(1, 1, nz))          # z-profile -> broadcast over (nx,ny,nz)
    ke, be, ce = r(ke), r(be), r(ce)
    kh, bh, ch = r(kh), r(bh), r(ch)
    z3 = (lambda: xp.zeros((nx, ny, nz)))
    Ex, Ey, Ez = z3(), z3(), z3()
    Hx, Hy, Hz = z3(), z3(), z3()
    Jx, Jy, Jz = z3(), z3(), z3()                            # Drude polarization currents (per E-component)
    psi_Hx, psi_Hy = z3(), z3()                              # CPML memory for dEy/dz, dEx/dz (H-grid)
    psi_Ex, psi_Ey = z3(), z3()                              # CPML memory for dHy/dz, dHx/dz (E-grid)
    do_lor = lor is not None
    if do_lor:                                               # Lorentz ADE: a polarization PL per E-component
        C1, C2, C3 = xp.asarray(lor[0]), xp.asarray(lor[1]), xp.asarray(lor[2])
        PLx, PLy, PLz = z3(), z3(), z3()
        PLpx, PLpy, PLpz = z3(), z3(), z3()
    aJ = (1.0 - gam * dt / 2.0) / (1.0 + gam * dt / 2.0)
    bJ = (EPS0 * wp ** 2 * dt / 2.0) / (1.0 + gam * dt / 2.0)
    cmu = dt / MU0
    sh = (nsteps, nx, ny)
    exL, eyL, hxL, hyL = xp.empty(sh), xp.empty(sh), xp.empty(sh), xp.empty(sh)
    exR, eyR, hxR, hyR = xp.empty(sh), xp.empty(sh), xp.empty(sh), xp.empty(sh)
    for n in range(nsteps):
        # ---------------- H update: dH/dt = -(1/mu) curl E ----------------
        # Hx: -(dEz/dy - dEy/dz) ; dEy/dz is CPML-stretched
        dEy_dz = (Ey[:, :, 1:] - Ey[:, :, :-1]) / dz
        psi_Hx[:, :, :-1] = bh[:, :, :-1] * psi_Hx[:, :, :-1] + ch[:, :, :-1] * dEy_dz
        sEy_dz = z3(); sEy_dz[:, :, :-1] = dEy_dz / kh[:, :, :-1] + psi_Hx[:, :, :-1]
        dEz_dy = (xp.roll(Ez, -1, axis=1) - Ez) / dy
        Hx -= cmu * (dEz_dy - sEy_dz)
        # Hy: -(dEx/dz - dEz/dx) ; dEx/dz is CPML-stretched
        dEx_dz = (Ex[:, :, 1:] - Ex[:, :, :-1]) / dz
        psi_Hy[:, :, :-1] = bh[:, :, :-1] * psi_Hy[:, :, :-1] + ch[:, :, :-1] * dEx_dz
        sEx_dz = z3(); sEx_dz[:, :, :-1] = dEx_dz / kh[:, :, :-1] + psi_Hy[:, :, :-1]
        dEz_dx = (xp.roll(Ez, -1, axis=0) - Ez) / dx
        Hy -= cmu * (sEx_dz - dEz_dx)
        # Hz: -(dEy/dx - dEx/dy) ; both transverse (no CPML)
        dEy_dx = (xp.roll(Ey, -1, axis=0) - Ey) / dx
        dEx_dy = (xp.roll(Ex, -1, axis=1) - Ex) / dy
        Hz -= cmu * (dEy_dx - dEx_dy)
        # ---------------- E update: eps0 eps_eff dE/dt = curl H - J ----------------
        eps_eff = eps_inf + chi3 * (Ex ** 2 + Ey ** 2 + Ez ** 2)
        ce_dt = EPS0 * eps_eff / dt
        denom = ce_dt + bJ / 2.0
        # Ex: (dHz/dy - dHy/dz) ; dHy/dz CPML-stretched
        dHy_dz = (Hy[:, :, 1:] - Hy[:, :, :-1]) / dz
        psi_Ex[:, :, 1:] = be[:, :, 1:] * psi_Ex[:, :, 1:] + ce[:, :, 1:] * dHy_dz
        sHy_dz = z3(); sHy_dz[:, :, 1:] = dHy_dz / ke[:, :, 1:] + psi_Ex[:, :, 1:]
        dHz_dy = (Hz - xp.roll(Hz, 1, axis=1)) / dy
        curlx = dHz_dy - sHy_dz
        if do_lor:                                          # Lorentz dPLx/dt enters the Ex-update
            PLxn = C1 * PLx + C2 * PLpx + C3 * Ex
            curlx = curlx - (PLxn - PLx) / dt
            PLpx, PLx = PLx, PLxn
        Exn = (ce_dt * Ex + curlx - 0.5 * (1.0 + aJ) * Jx - 0.5 * bJ * Ex) / denom
        Jx = aJ * Jx + bJ * (Exn + Ex)
        # Ey: (dHx/dz - dHz/dx) ; dHx/dz CPML-stretched
        dHx_dz = (Hx[:, :, 1:] - Hx[:, :, :-1]) / dz
        psi_Ey[:, :, 1:] = be[:, :, 1:] * psi_Ey[:, :, 1:] + ce[:, :, 1:] * dHx_dz
        sHx_dz = z3(); sHx_dz[:, :, 1:] = dHx_dz / ke[:, :, 1:] + psi_Ey[:, :, 1:]
        dHz_dx = (Hz - xp.roll(Hz, 1, axis=0)) / dx
        curly = sHx_dz - dHz_dx
        if do_lor:
            PLyn = C1 * PLy + C2 * PLpy + C3 * Ey
            curly = curly - (PLyn - PLy) / dt
            PLpy, PLy = PLy, PLyn
        Eyn = (ce_dt * Ey + curly - 0.5 * (1.0 + aJ) * Jy - 0.5 * bJ * Ey) / denom
        Jy = aJ * Jy + bJ * (Eyn + Ey)
        # Ez: (dHy/dx - dHx/dy) ; both transverse (no CPML)
        dHy_dx = (Hy - xp.roll(Hy, 1, axis=0)) / dx
        dHx_dy = (Hx - xp.roll(Hx, 1, axis=1)) / dy
        curlz = dHy_dx - dHx_dy
        if do_lor:
            PLzn = C1 * PLz + C2 * PLpz + C3 * Ez
            curlz = curlz - (PLzn - PLz) / dt
            PLpz, PLz = PLz, PLzn
        Ezn = (ce_dt * Ez + curlz - 0.5 * (1.0 + aJ) * Jz - 0.5 * bJ * Ez) / denom
        Jz = aJ * Jz + bJ * (Ezn + Ez)
        # soft y-polarized plane source (uniform in x,y -> normal incidence), PEC backing the CPML:
        # a z=const PEC plane forces only the TANGENTIAL E (Ex,Ey) to zero; the normal Ez sits half a
        # cell inside and is left to its (purely transverse-curl, in-bounds) update.
        Eyn[:, :, k_src] += src[n]
        for F in (Exn, Eyn):
            F[:, :, 0] = 0.0; F[:, :, -1] = 0.0
        Ex, Ey, Ez = Exn, Eyn, Ezn
        # probe planes: co-locate Hx,Hy (at k+/-1/2) onto the E-plane (k) so S_z co-locates in z
        exL[n] = Ex[:, :, k_pL]; eyL[n] = Ey[:, :, k_pL]
        hxL[n] = 0.5 * (Hx[:, :, k_pL] + Hx[:, :, k_pL - 1]); hyL[n] = 0.5 * (Hy[:, :, k_pL] + Hy[:, :, k_pL - 1])
        exR[n] = Ex[:, :, k_pR]; eyR[n] = Ey[:, :, k_pR]
        hxR[n] = 0.5 * (Hx[:, :, k_pR] + Hx[:, :, k_pR - 1]); hyR[n] = 0.5 * (Hy[:, :, k_pR] + Hy[:, :, k_pR - 1])
    return exL, eyL, hxL, hyL, exR, eyR, hxR, hyR


@njit(parallel=True, fastmath=True, cache=True)
def _te3d_numba(eps_inf, wp, gam, chi3, ke, be, ce, kh, bh, ch, dx, dy, dz, dt,
                nsteps, k_src, k_pL, k_pR, src, C1, C2, C3, has_lor):
    """Fused, prange-threaded full-vector 3D timestep (the Numba CPU kernel) -- byte-near-identical physics
    to _run_3d (six-component Yee + per-component semi-implicit Drude ADE + Kerr + Lorentz ADE + CFS-CPML in
    z + PEC, Bloch-periodic x,y), but explicit-loop + JIT-compiled so the whole step is ONE compiled pass.
    C1,C2,C3 = per-cell Lorentz coefficients, has_lor gates the per-component polarization PL{x,y,z}.
    Parallel-safe over x: the H-phase writes Hx/Hy/Hz[i] (disjoint) reading only E (read-only); the E-phase
    writes Ex/Ey/Ez[i] (disjoint) reading only H. Returns the Ex,Ey,Hx,Hy probe planes (left/right)."""
    nx, ny, nz = eps_inf.shape
    Ex = np.zeros((nx, ny, nz)); Ey = np.zeros((nx, ny, nz)); Ez = np.zeros((nx, ny, nz))
    Hx = np.zeros((nx, ny, nz)); Hy = np.zeros((nx, ny, nz)); Hz = np.zeros((nx, ny, nz))
    Jx = np.zeros((nx, ny, nz)); Jy = np.zeros((nx, ny, nz)); Jz = np.zeros((nx, ny, nz))
    PLx = np.zeros((nx, ny, nz)); PLy = np.zeros((nx, ny, nz)); PLz = np.zeros((nx, ny, nz))
    PLpx = np.zeros((nx, ny, nz)); PLpy = np.zeros((nx, ny, nz)); PLpz = np.zeros((nx, ny, nz))
    psi_Hx = np.zeros((nx, ny, nz)); psi_Hy = np.zeros((nx, ny, nz))
    psi_Ex = np.zeros((nx, ny, nz)); psi_Ey = np.zeros((nx, ny, nz))
    sh = (nsteps, nx, ny)
    exL = np.empty(sh); eyL = np.empty(sh); hxL = np.empty(sh); hyL = np.empty(sh)
    exR = np.empty(sh); eyR = np.empty(sh); hxR = np.empty(sh); hyR = np.empty(sh)
    cmu = dt / MU0
    e0dt = EPS0 / dt
    for n in range(nsteps):
        # ---- H update (dH/dt = -(1/mu) curl E); only d/dz is CPML-stretched ----
        for i in prange(nx):
            ip1 = i + 1 if i + 1 < nx else 0
            for j in range(ny):
                jp1 = j + 1 if j + 1 < ny else 0
                for k in range(nz):
                    dEz_dy = (Ez[i, jp1, k] - Ez[i, j, k]) / dy
                    if k < nz - 1:
                        d = (Ey[i, j, k + 1] - Ey[i, j, k]) / dz
                        psi_Hx[i, j, k] = bh[k] * psi_Hx[i, j, k] + ch[k] * d
                        sEy = d / kh[k] + psi_Hx[i, j, k]
                    else:
                        sEy = 0.0
                    Hx[i, j, k] -= cmu * (dEz_dy - sEy)
                    dEz_dx = (Ez[ip1, j, k] - Ez[i, j, k]) / dx
                    if k < nz - 1:
                        d2 = (Ex[i, j, k + 1] - Ex[i, j, k]) / dz
                        psi_Hy[i, j, k] = bh[k] * psi_Hy[i, j, k] + ch[k] * d2
                        sEx = d2 / kh[k] + psi_Hy[i, j, k]
                    else:
                        sEx = 0.0
                    Hy[i, j, k] -= cmu * (sEx - dEz_dx)
                    Hz[i, j, k] -= cmu * ((Ey[ip1, j, k] - Ey[i, j, k]) / dx - (Ex[i, jp1, k] - Ex[i, j, k]) / dy)
        # ---- E update (eps0 eps_eff dE/dt = curl H - J); per-component Drude ADE + Kerr ----
        for i in prange(nx):
            im1 = i - 1 if i - 1 >= 0 else nx - 1
            for j in range(ny):
                jm1 = j - 1 if j - 1 >= 0 else ny - 1
                for k in range(nz):
                    exo = Ex[i, j, k]; eyo = Ey[i, j, k]; ezo = Ez[i, j, k]
                    g = gam[i, j, k]
                    aJ = (1.0 - g * dt / 2.0) / (1.0 + g * dt / 2.0)
                    bJ = (EPS0 * wp[i, j, k] ** 2 * dt / 2.0) / (1.0 + g * dt / 2.0)
                    eps_eff = eps_inf[i, j, k] + chi3[i, j, k] * (exo * exo + eyo * eyo + ezo * ezo)
                    denom = e0dt * eps_eff + bJ / 2.0
                    coef = 0.5 * (1.0 + aJ)
                    # Ex: curl_x H = dHz/dy - dHy/dz (CPML)
                    dHz_dy = (Hz[i, j, k] - Hz[i, jm1, k]) / dy
                    if k >= 1:
                        dHy_dz = (Hy[i, j, k] - Hy[i, j, k - 1]) / dz
                        psi_Ex[i, j, k] = be[k] * psi_Ex[i, j, k] + ce[k] * dHy_dz
                        sHy = dHy_dz / ke[k] + psi_Ex[i, j, k]
                    else:
                        sHy = 0.0
                    cx = dHz_dy - sHy
                    if has_lor:                             # Lorentz dPLx/dt enters the Ex-update
                        pln = C1[i, j, k] * PLx[i, j, k] + C2[i, j, k] * PLpx[i, j, k] + C3[i, j, k] * exo
                        cx = cx - (pln - PLx[i, j, k]) / dt
                        PLpx[i, j, k] = PLx[i, j, k]; PLx[i, j, k] = pln
                    exn = (e0dt * eps_eff * exo + cx - coef * Jx[i, j, k] - 0.5 * bJ * exo) / denom
                    Jx[i, j, k] = aJ * Jx[i, j, k] + bJ * (exn + exo)
                    # Ey: curl_y H = dHx/dz (CPML) - dHz/dx
                    dHz_dx = (Hz[i, j, k] - Hz[im1, j, k]) / dx
                    if k >= 1:
                        dHx_dz = (Hx[i, j, k] - Hx[i, j, k - 1]) / dz
                        psi_Ey[i, j, k] = be[k] * psi_Ey[i, j, k] + ce[k] * dHx_dz
                        sHx = dHx_dz / ke[k] + psi_Ey[i, j, k]
                    else:
                        sHx = 0.0
                    cy = sHx - dHz_dx
                    if has_lor:
                        pln = C1[i, j, k] * PLy[i, j, k] + C2[i, j, k] * PLpy[i, j, k] + C3[i, j, k] * eyo
                        cy = cy - (pln - PLy[i, j, k]) / dt
                        PLpy[i, j, k] = PLy[i, j, k]; PLy[i, j, k] = pln
                    eyn = (e0dt * eps_eff * eyo + cy - coef * Jy[i, j, k] - 0.5 * bJ * eyo) / denom
                    Jy[i, j, k] = aJ * Jy[i, j, k] + bJ * (eyn + eyo)
                    # Ez: curl_z H = dHy/dx - dHx/dy (transverse, no CPML)
                    cz = (Hy[i, j, k] - Hy[im1, j, k]) / dx - (Hx[i, j, k] - Hx[i, jm1, k]) / dy
                    if has_lor:
                        pln = C1[i, j, k] * PLz[i, j, k] + C2[i, j, k] * PLpz[i, j, k] + C3[i, j, k] * ezo
                        cz = cz - (pln - PLz[i, j, k]) / dt
                        PLpz[i, j, k] = PLz[i, j, k]; PLz[i, j, k] = pln
                    ezn = (e0dt * eps_eff * ezo + cz - coef * Jz[i, j, k] - 0.5 * bJ * ezo) / denom
                    Jz[i, j, k] = aJ * Jz[i, j, k] + bJ * (ezn + ezo)
                    Ex[i, j, k] = exn; Ey[i, j, k] = eyn; Ez[i, j, k] = ezn
        # soft y-pol source + PEC (tangential Ex,Ey only); then co-located probes
        for i in prange(nx):
            for j in range(ny):
                Ey[i, j, k_src] += src[n]
                Ex[i, j, 0] = 0.0; Ex[i, j, nz - 1] = 0.0
                Ey[i, j, 0] = 0.0; Ey[i, j, nz - 1] = 0.0
        for i in prange(nx):
            for j in range(ny):
                exL[n, i, j] = Ex[i, j, k_pL]; eyL[n, i, j] = Ey[i, j, k_pL]
                hxL[n, i, j] = 0.5 * (Hx[i, j, k_pL] + Hx[i, j, k_pL - 1])
                hyL[n, i, j] = 0.5 * (Hy[i, j, k_pL] + Hy[i, j, k_pL - 1])
                exR[n, i, j] = Ex[i, j, k_pR]; eyR[n, i, j] = Ey[i, j, k_pR]
                hxR[n, i, j] = 0.5 * (Hx[i, j, k_pR] + Hx[i, j, k_pR - 1])
                hyR[n, i, j] = 0.5 * (Hy[i, j, k_pR] + Hy[i, j, k_pR - 1])
    return exL, eyL, hxL, hyL, exR, eyR, hxR, hyR


def _run_3d_jax(eps_inf, wp, gam, chi3, dx, dy, dz, dt, nsteps, k_src, k_pL, k_pR, src, cpml):
    """JAX (XLA lax.scan) 3D backend -- the SAME six-component physics as _run_3d, but DIFFERENTIABLE end
    to end: a downstream jax.grad gives d(R,T)/d(geometry/material) for 3D inverse design. Functional
    (immutable .at[]) updates, float64 forced to match the reference. Returns the eight probe planes as JAX
    arrays (the dispatcher converts to NumPy; staying in JAX lets a caller grad straight through the loop)."""
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from jax import lax
    (ke, be, ce), (kh, bh, ch) = cpml
    nx, ny, nz = eps_inf.shape
    rs = (lambda a: jnp.asarray(a).reshape(1, 1, nz))       # z-profile -> broadcast
    ke, be, ce = rs(ke), rs(be), rs(ce)
    kh, bh, ch = rs(kh), rs(bh), rs(ch)
    eps_inf = jnp.asarray(eps_inf); chi3 = jnp.asarray(chi3)
    gam = jnp.asarray(gam); wp = jnp.asarray(wp)
    aJ = (1.0 - gam * dt / 2.0) / (1.0 + gam * dt / 2.0)
    bJ = (EPS0 * wp ** 2 * dt / 2.0) / (1.0 + gam * dt / 2.0)
    cmu = dt / MU0
    zz = jnp.zeros((nx, ny, nz))

    def step(carry, src_n):
        Ex, Ey, Ez, Hx, Hy, Hz, Jx, Jy, Jz, psi_Hx, psi_Hy, psi_Ex, psi_Ey = carry
        # ---- H update ----
        dEy_dz = (Ey[:, :, 1:] - Ey[:, :, :-1]) / dz
        psi_Hx = psi_Hx.at[:, :, :-1].set(bh[:, :, :-1] * psi_Hx[:, :, :-1] + ch[:, :, :-1] * dEy_dz)
        sEy = zz.at[:, :, :-1].set(dEy_dz / kh[:, :, :-1] + psi_Hx[:, :, :-1])
        Hx = Hx - cmu * ((jnp.roll(Ez, -1, axis=1) - Ez) / dy - sEy)
        dEx_dz = (Ex[:, :, 1:] - Ex[:, :, :-1]) / dz
        psi_Hy = psi_Hy.at[:, :, :-1].set(bh[:, :, :-1] * psi_Hy[:, :, :-1] + ch[:, :, :-1] * dEx_dz)
        sEx = zz.at[:, :, :-1].set(dEx_dz / kh[:, :, :-1] + psi_Hy[:, :, :-1])
        Hy = Hy - cmu * (sEx - (jnp.roll(Ez, -1, axis=0) - Ez) / dx)
        Hz = Hz - cmu * ((jnp.roll(Ey, -1, axis=0) - Ey) / dx - (jnp.roll(Ex, -1, axis=1) - Ex) / dy)
        # ---- E update (per-component Drude ADE + Kerr) ----
        eps_eff = eps_inf + chi3 * (Ex ** 2 + Ey ** 2 + Ez ** 2)
        ce_dt = EPS0 * eps_eff / dt
        denom = ce_dt + bJ / 2.0
        coef = 0.5 * (1.0 + aJ)
        dHy_dz = (Hy[:, :, 1:] - Hy[:, :, :-1]) / dz
        psi_Ex = psi_Ex.at[:, :, 1:].set(be[:, :, 1:] * psi_Ex[:, :, 1:] + ce[:, :, 1:] * dHy_dz)
        sHy = zz.at[:, :, 1:].set(dHy_dz / ke[:, :, 1:] + psi_Ex[:, :, 1:])
        Exn = (ce_dt * Ex + ((Hz - jnp.roll(Hz, 1, axis=1)) / dy - sHy) - coef * Jx - 0.5 * bJ * Ex) / denom
        Jx = aJ * Jx + bJ * (Exn + Ex)
        dHx_dz = (Hx[:, :, 1:] - Hx[:, :, :-1]) / dz
        psi_Ey = psi_Ey.at[:, :, 1:].set(be[:, :, 1:] * psi_Ey[:, :, 1:] + ce[:, :, 1:] * dHx_dz)
        sHx = zz.at[:, :, 1:].set(dHx_dz / ke[:, :, 1:] + psi_Ey[:, :, 1:])
        Eyn = (ce_dt * Ey + (sHx - (Hz - jnp.roll(Hz, 1, axis=0)) / dx) - coef * Jy - 0.5 * bJ * Ey) / denom
        Jy = aJ * Jy + bJ * (Eyn + Ey)
        curlz = (Hy - jnp.roll(Hy, 1, axis=0)) / dx - (Hx - jnp.roll(Hx, 1, axis=1)) / dy
        Ezn = (ce_dt * Ez + curlz - coef * Jz - 0.5 * bJ * Ez) / denom
        Jz = aJ * Jz + bJ * (Ezn + Ez)
        Eyn = Eyn.at[:, :, k_src].add(src_n)                # soft y-pol source
        Exn = Exn.at[:, :, 0].set(0.0).at[:, :, nz - 1].set(0.0)   # PEC: tangential Ex,Ey only
        Eyn = Eyn.at[:, :, 0].set(0.0).at[:, :, nz - 1].set(0.0)
        Ex, Ey, Ez = Exn, Eyn, Ezn
        out = (Ex[:, :, k_pL], Ey[:, :, k_pL],
               0.5 * (Hx[:, :, k_pL] + Hx[:, :, k_pL - 1]), 0.5 * (Hy[:, :, k_pL] + Hy[:, :, k_pL - 1]),
               Ex[:, :, k_pR], Ey[:, :, k_pR],
               0.5 * (Hx[:, :, k_pR] + Hx[:, :, k_pR - 1]), 0.5 * (Hy[:, :, k_pR] + Hy[:, :, k_pR - 1]))
        return (Ex, Ey, Ez, Hx, Hy, Hz, Jx, Jy, Jz, psi_Hx, psi_Hy, psi_Ex, psi_Ey), out

    carry0 = tuple(zz for _ in range(13))
    _, outs = lax.scan(step, carry0, jnp.asarray(src))
    return outs                                             # 8-tuple of (nsteps,nx,ny) JAX arrays


def _dispatch_3d(name, eps_inf, wp, gam, chi3, dx, dy, dz, dt, nsteps, k_src, k_pL, k_pR, src, cpml, xp=np,
                 lor=None):
    """Run ONE 3D pass on the named backend, returning the eight probe planes as NumPy arrays (so the
    downstream FFT / R-T extraction is backend-agnostic). 'numba' = the fused threaded CPU kernel (the
    fast 3D path); 'numpy'/'cupy' = the vectorized reference loop. `lor`=(C1,C2,C3) per-cell Lorentz ADE
    coefficients or None. (The jax 3D kernel does not carry the Lorentz ADE yet -> guarded upstream.)"""
    (ke, be, ce), (kh, bh, ch) = cpml
    if name == "numba":
        has_lor = lor is not None
        z = np.zeros_like(eps_inf)
        C1, C2, C3 = (lor if has_lor else (z, z, z))
        return _te3d_numba(eps_inf, wp, gam, chi3, ke, be, ce, kh, bh, ch, dx, dy, dz, dt,
                           nsteps, k_src, k_pL, k_pR, src, C1, C2, C3, has_lor)
    if name == "jax":
        if lor is not None:
            raise NotImplementedError("the jax 3D backend does not carry the Lorentz ADE yet; use "
                                      "backend='numba' or 'numpy' for a 3D Lorentz material.")
        out = _run_3d_jax(eps_inf, wp, gam, chi3, dx, dy, dz, dt, nsteps, k_src, k_pL, k_pR, src, cpml)
        return tuple(np.asarray(v) for v in out)            # JAX arrays -> NumPy for the FFT/R-T stage
    if name == "cupy" and xp is np:
        import cupy as xp
    a = tuple(xp.asarray(v) for v in (eps_inf, wp, gam, chi3))
    out = _run_3d(*a, dx, dy, dz, dt, nsteps, k_src, k_pL, k_pR, xp.asarray(src), cpml, xp, lor)
    to_np = (lambda v: np.asarray(v.get()) if hasattr(v, "get") else np.asarray(v))
    return tuple(to_np(v) for v in out)


def solve_fdtd_3d(layers: List[FDTDLayer], *, period_x_m: float, period_y_m: float,
                  nx: Optional[int] = None, ny: Optional[int] = None,
                  lateral_eps_inf: Optional[np.ndarray] = None,
                  lambda_min_m: float, lambda_max_m: float, resolution: int = 24,
                  courant: float = 0.5, n_pad_wave: float = 4.0, settle: float = 12.0,
                  kerr: bool = False, source_amp: float = 1.0, npml: int = 12,
                  n_super: float = 1.0, n_sub: float = 1.0,
                  backend: str = "numpy", xp=np) -> FDTD3DResult:
    """Broadband R(f)/T(f) of a doubly-periodic (period_x_m x period_y_m) unit cell at NORMAL incidence,
    y-polarized. `layers` = the through-stack (z) profile; supply `lateral_eps_inf` (an (nx,ny,nz) array,
    or a callable(nx,ny,nz,zc,pad,zstruct)->(nx,ny,nz)) to make a 2D-periodic structure, else the stack is
    laterally UNIFORM (and the result reduces to 1D/TMM). Returns both the specular 0-order and the
    total-flux (all (kx,ky) orders) R/T.

    n_super / n_sub (default 1 = vacuum): the lossless semi-infinite superstrate / substrate indices
    (metasurface-on-glass), filling the z-pads with n_super^2 / n_sub^2, with an impedance-matched CPML
    and a homogeneous-superstrate incident reference (T carries the n_sub/n_super flux ratio). Reduces
    byte-identically to vacuum at n_super=n_sub=1.

    backend: 'auto'/'numba' (the fused threaded CPU kernel = the fast 3D path), 'numpy' (reference),
    'jax' (the differentiable XLA scan, for 3D inverse design), or 'cupy'/xp for the GPU. A Lorentz pole
    (lorentz_delta_eps) is carried by the numpy/numba/cupy 3D kernels (per E-component); the jax 3D backend
    does not yet, and raises if a Lorentz layer is run on it."""
    if abs(complex(n_super).imag) > 1e-9 or abs(complex(n_sub).imag) > 1e-9:   # mirror the FEM/2D guard
        raise NotImplementedError("solve_fdtd_3d: R/T and the energy budget are defined only for LOSSLESS "
                                  "end media (Im(n)=0); got n_super={}, n_sub={}.".format(n_super, n_sub))
    f_min, f_max = C_LIGHT / lambda_max_m, C_LIGHT / lambda_min_m
    f_c = 0.5 * (f_min + f_max)
    w_band = 2.0 * np.pi * np.linspace(f_min, f_max, 9)

    def _n_band_max(L):
        return max(abs(np.sqrt(L.eps_at(w))) for w in w_band)
    n_max = max(1.0, n_super, n_sub, max(_n_band_max(L) for L in layers))
    dz = lambda_min_m / (resolution * n_max)
    if nx is None:
        nx = max(4, int(round(period_x_m / dz)))
    if ny is None:
        ny = max(4, int(round(period_y_m / dz)))
    dx = period_x_m / nx
    dy = period_y_m / ny
    # 3D CFL: dt <= courant / (c sqrt(1/dx^2 + 1/dy^2 + 1/dz^2))
    dt = courant / (C_LIGHT * np.sqrt(1.0 / dx ** 2 + 1.0 / dy ** 2 + 1.0 / dz ** 2))

    pad = n_pad_wave * lambda_max_m
    z_struct = float(sum(L.thickness_m for L in layers))
    Lz = 2.0 * pad + z_struct
    nz = int(round(Lz / dz)) + 1

    shape = (nx, ny, nz)
    eps_inf = np.ones(shape); wp = np.zeros(shape); gam = np.zeros(shape); chi3 = np.zeros(shape)
    lw0 = np.zeros(shape); lgam = np.zeros(shape); ldeps = np.zeros(shape)   # Lorentz pole per cell
    zc = (np.arange(nz) + 0.5) * dz
    eps_inf[:, :, zc < pad] = n_super ** 2                   # fill the semi-infinite super/substrate pads
    eps_inf[:, :, zc >= pad + z_struct] = n_sub ** 2
    z = pad
    for L in layers:
        m = (zc >= z) & (zc < z + L.thickness_m)
        eps_inf[:, :, m] = L.eps_inf
        wp[:, :, m] = L.drude_wp_rad_s
        gam[:, :, m] = L.drude_gamma_rad_s
        lw0[:, :, m] = L.lorentz_w0_rad_s
        lgam[:, :, m] = L.lorentz_gamma_rad_s
        ldeps[:, :, m] = L.lorentz_delta_eps
        if kerr:
            chi3[:, :, m] = L.chi3_m2_V2
        z += L.thickness_m
    if lateral_eps_inf is not None:
        lat = lateral_eps_inf(nx, ny, nz, zc, pad, z_struct) if callable(lateral_eps_inf) else np.asarray(lateral_eps_inf)
        eps_inf = np.asarray(lat, dtype=float)

    k_src = max(2, int(round((0.35 * pad) / dz)))
    k_pL = int(round((0.7 * pad) / dz))
    k_pR = int(round((pad + z_struct + 0.3 * pad) / dz))

    tau = 1.0 / (np.pi * (f_max - f_min))
    t0 = settle * tau
    nsteps = int(round((2.0 * t0 + (Lz / C_LIGHT) * 4.0 + 200 * tau) / dt))
    tgrid = np.arange(nsteps) * dt
    src = source_amp * np.exp(-((tgrid - t0) / tau) ** 2) * np.cos(2.0 * np.pi * f_c * (tgrid - t0))

    # Lorentz ADE coefficients (central difference, per E-component); applied to the STRUCTURE run only.
    # d_eps=0 everywhere -> lor=None -> the path is byte-identical to the no-Lorentz solve.
    lor = None
    if np.any(ldeps != 0.0):
        den = 1.0 + lgam * dt / 2.0
        C1 = (2.0 - lw0 ** 2 * dt ** 2) / den
        C2 = (lgam * dt / 2.0 - 1.0) / den
        C3 = (EPS0 * ldeps * lw0 ** 2 * dt ** 2) / den
        lor = (C1, C2, C3)

    cpml_struct = _cpml_z(nz, dz, dt, npml, n_super, n_sub)  # PML matched super (low z) + sub (high z)
    cpml_ref = _cpml_z(nz, dz, dt, npml, n_super, n_super)   # homogeneous-superstrate reference
    name = _resolve_backend(backend)                        # 'auto'/'cpu' -> numba (the fast 3D path)
    one = np.ones(shape); zero = np.zeros(shape)

    def run(ei, w, g_, c3, cpml, lor=None):
        return _dispatch_3d(name, ei, w, g_, c3, dx, dy, dz, dt, nsteps, k_src, k_pL, k_pR, src, cpml, xp, lor)

    exL_i, eyL_i, hxL_i, hyL_i, exR_i, eyR_i, hxR_i, hyR_i = run(
        n_super ** 2 * one, zero, zero, zero, cpml_ref)                  # homogeneous-superstrate reference
    exL_t, eyL_t, hxL_t, hyL_t, exR_t, eyR_t, hxR_t, hyR_t = run(eps_inf, wp, gam, chi3, cpml_struct, lor)  # struct

    f = np.fft.rfftfreq(nsteps, dt)
    # 0-order specular co-pol (E_y) from the x,y-MEAN field (== the 1D two-run method)
    mL_inc = np.fft.rfft(eyL_i.mean(axis=(1, 2))); mR_inc = np.fft.rfft(eyR_i.mean(axis=(1, 2)))
    mRefl = np.fft.rfft((eyL_t - eyL_i).mean(axis=(1, 2))); mTrans = np.fft.rfft(eyR_t.mean(axis=(1, 2)))
    k0 = 2.0 * np.pi * f / C_LIGHT
    with np.errstate(divide="ignore", invalid="ignore"):
        R0 = np.abs(mRefl / mL_inc) ** 2
        T0 = np.abs(mTrans / mR_inc) ** 2 * (n_sub / n_super)   # Snell power-flux ratio (incident in super)
        # COMPLEX co-pol 0-order coeffs: conjugate (rfft is exp(+iwt); convention is exp(-iwt)), then
        # de-embed the propagation phase (superstrate phase velocity c/n_super): r0c to the front face
        # z=pad (probe k_pL); t0c across the cell (the common probe phase cancels in t).
        r0c = np.conj(mRefl / mL_inc) * np.exp(-2j * n_super * k0 * (pad - k_pL * dz))
        t0c = np.conj(mTrans / mR_inc) * np.exp(1j * k0 * z_struct)
    # total R/T from the full Poynting flux (all (kx,ky) diffraction orders)
    P_inc = _flux3d(exL_i, eyL_i, hxL_i, hyL_i)
    P_refl = _flux3d(exL_t - exL_i, eyL_t - eyL_i, hxL_t - hxL_i, hyL_t - hyL_i)
    P_trans = _flux3d(exR_t, eyR_t, hxR_t, hyR_t)
    with np.errstate(divide="ignore", invalid="ignore"):
        R_flux = np.abs(P_refl) / np.abs(P_inc)
        T_flux = np.abs(P_trans) / np.abs(P_inc)
    band = (f >= f_min) & (f <= f_max) & (np.abs(mL_inc) > 0.05 * np.max(np.abs(mL_inc)))
    return FDTD3DResult(freqs_Hz=f, R0=R0, T0=T0, R_flux=R_flux, T_flux=T_flux, band=band, r0=r0c, t0=t0c)
