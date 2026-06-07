"""
2D/3D FDTD optical backend (roadmap: extend the validated 1D Yee solver to a periodic 2D/3D engine).

PHASE 0 -- a 2D TE (E_y, H_x, H_z) Yee solver: a plane wave propagating in +z on a unit cell that is
PERIODIC in x (a 1D grating / laterally-structured slab), at NORMAL incidence. Carries the same physics
as the 1D baseline: a semi-implicit Drude ADE per E-component, an instantaneous Kerr chi3, a
modulated-Gaussian soft source, and the TWO-RUN (vacuum reference + structure) broadband R(omega)/
T(omega) extraction. Mur 1st-order ABC at the z ends; periodic in x.

This is the BACKEND-AGNOSTIC NumPy REFERENCE -- the correctness oracle and CPU/small-grid path that
every faster kernel (Taichi / CuPy RawKernel / JAX) is validated against. The hot run loop takes an
array module `xp` (numpy default) so a drop-in backend swaps with no change to the physics or the
OpticalSolver seam. Convention exp(-i omega t), SI; Im(eps) > 0 = loss. Reduces EXACTLY to the 1D
solver + TMM for a laterally-uniform stack at normal incidence (validation/fdtd_2d_reduces.py).

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


@dataclass
class FDTD2DResult:
    freqs_Hz: np.ndarray
    R0: np.ndarray              # 0-order (specular) reflectance from the x-mean field (== 1D/TMM)
    T0: np.ndarray             # 0-order (specular) transmittance from the x-mean field
    R_flux: np.ndarray         # TOTAL reflectance from the Poynting flux (all diffraction orders)
    T_flux: np.ndarray         # TOTAL transmittance from the Poynting flux (all diffraction orders)
    band: np.ndarray            # boolean mask of the well-excited frequency band


def _run_2d_te(eps_inf, wp, gam, chi3, dx, dz, dt, nsteps, k_src, k_pL, k_pR, src, xp=np):
    """One 2D TE pass over a cell-wise (nx,nz) (eps_inf, wp, gamma, chi3) profile. Periodic in x (roll),
    Mur ABC in z. Records the E_y and H_x x-lines at the left/right z-probe planes (for both the
    x-mean 0-order and the Poynting-flux R/T). Semi-implicit Drude ADE + instantaneous Kerr."""
    nx, nz = eps_inf.shape
    Ey = xp.zeros((nx, nz))
    Hx = xp.zeros((nx, nz))                 # Hx[i,k] at (i, k+1/2)
    Hz = xp.zeros((nx, nz))                 # Hz[i,k] at (i+1/2, k)
    Jy = xp.zeros((nx, nz))                 # Drude polarization current (on E_y)
    aJ = (1.0 - gam * dt / 2.0) / (1.0 + gam * dt / 2.0)
    bJ = (EPS0 * wp ** 2 * dt / 2.0) / (1.0 + gam * dt / 2.0)
    mur = (C_LIGHT * dt - dz) / (C_LIGHT * dt + dz)
    eyL = xp.empty((nsteps, nx)); hxL = xp.empty((nsteps, nx))
    eyR = xp.empty((nsteps, nx)); hxR = xp.empty((nsteps, nx))
    chx = dt / (MU0 * dz)
    chz = dt / (MU0 * dx)
    for n in range(nsteps):
        # old z-edge cells for the Mur ABC (per x-column)
        Ey_L0, Ey_L1 = Ey[:, 0].copy(), Ey[:, 1].copy()
        Ey_R0, Ey_R1 = Ey[:, -1].copy(), Ey[:, -2].copy()
        # H updates: dHx/dt = (1/mu0) dEy/dz ; dHz/dt = -(1/mu0) dEy/dx (periodic in x)
        Hx[:, :-1] += chx * (Ey[:, 1:] - Ey[:, :-1])
        Hz += -chz * (xp.roll(Ey, -1, axis=0) - Ey)
        # curl_y(H) = dHx/dz - dHz/dx at the E_y points (interior in z; periodic in x)
        curl = xp.zeros((nx, nz))
        curl[:, 1:] += (Hx[:, 1:] - Hx[:, :-1]) / dz
        curl -= (Hz - xp.roll(Hz, 1, axis=0)) / dx
        # E update: eps0 eps_eff dEy/dt = curl - J, semi-implicit Drude + instantaneous Kerr
        eps_eff = eps_inf + chi3 * Ey ** 2
        denom = EPS0 * eps_eff / dt + bJ / 2.0
        Eynew = (EPS0 * eps_eff / dt * Ey + curl - 0.5 * (1.0 + aJ) * Jy - 0.5 * bJ * Ey) / denom
        Jy = aJ * Jy + bJ * (Eynew + Ey)
        Eynew[:, k_src] += src[n]            # soft plane source (uniform in x -> normal-incidence plane wave)
        # Mur 1st-order ABC at both z ends
        Eynew[:, 0] = Ey_L1 + mur * (Eynew[:, 1] - Ey_L0)
        Eynew[:, -1] = Ey_R1 + mur * (Eynew[:, -2] - Ey_R0)
        Ey = Eynew
        eyL[n] = Ey[:, k_pL]; hxL[n] = Hx[:, k_pL]
        eyR[n] = Ey[:, k_pR]; hxR[n] = Hx[:, k_pR]
    return eyL, hxL, eyR, hxR


def _flux(ey, hx):
    """Per-frequency time-averaged +z Poynting power S_z = -Re(E_y H_x*) summed over x, from the rfft
    of the recorded probe x-lines (shape (nsteps, nx)). Half-cell / half-step staggering offsets are
    common to numerator and the incident reference, so they cancel in the R/T ratio."""
    Ey = np.fft.rfft(ey, axis=0)
    Hx = np.fft.rfft(hx, axis=0)
    return -np.sum(np.real(Ey * np.conj(Hx)), axis=1)        # (nfreq,) signed z-power per frequency


def solve_fdtd_2d(layers: List[FDTDLayer], *, period_x_m: float, nx: Optional[int] = None,
                  lateral_eps_inf: Optional[np.ndarray] = None,
                  lambda_min_m: float, lambda_max_m: float, resolution: int = 40,
                  courant: float = 0.5, n_pad_wave: float = 6.0, settle: float = 12.0,
                  kerr: bool = False, source_amp: float = 1.0, xp=np) -> FDTD2DResult:
    """Broadband R(f)/T(f) of a periodic (period_x_m) 2D-TE unit cell at NORMAL incidence. `layers`
    is the through-stack (z) profile (vacuum super/substrate); supply `lateral_eps_inf` (shape
    (nx, n_layer_cells) or a callable building the (nx,nz) eps_inf) to make a laterally-structured
    grating, else the stack is laterally UNIFORM (and the result reduces to the 1D solver / TMM).
    Returns both the 0-order (specular, x-mean) and the total-flux (all-diffraction-order) R/T."""
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

    one = np.ones((nx, nz)); zero = np.zeros((nx, nz))
    a = (xp.asarray(eps_inf), xp.asarray(wp), xp.asarray(gam), xp.asarray(chi3))
    eyL_i, hxL_i, eyR_i, hxR_i = _run_2d_te(xp.asarray(one), xp.asarray(zero), xp.asarray(zero),
                                            xp.asarray(zero), dx, dz, dt, nsteps, k_src, k_pL, k_pR,
                                            xp.asarray(src), xp)
    eyL_t, hxL_t, eyR_t, hxR_t = _run_2d_te(*a, dx, dz, dt, nsteps, k_src, k_pL, k_pR, xp.asarray(src), xp)
    # back to numpy for the FFT extraction
    g = (lambda v: np.asarray(v.get()) if hasattr(v, "get") else np.asarray(v))
    eyL_i, hxL_i, eyR_i, hxR_i = map(g, (eyL_i, hxL_i, eyR_i, hxR_i))
    eyL_t, hxL_t, eyR_t, hxR_t = map(g, (eyL_t, hxL_t, eyR_t, hxR_t))

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
