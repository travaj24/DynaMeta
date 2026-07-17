"""3D numba CPU kernels (normal incidence and oblique envelope).

Split from the former monolithic fdtd_nd.py; see the package __init__ docstring
for conventions. Bodies are verbatim from the original module.
"""
from __future__ import annotations

import numpy as np

from dynameta.constants import EPS0, MU0
from dynameta.optics.fdtd_nd.backends import njit, prange

@njit(parallel=True, fastmath=True, cache=True)
def _run_3d_oblique_numba(eps_inf, wp, gam, ke, be, ce, kh, bh, ch, dx, dy, dz, dt,
                          nsteps, k_src, k_pL, k_pR, src, kx, ky, sx, sy):
    """Fused, prange-threaded full-vector COMPLEX-ENVELOPE oblique 3D timestep (the Numba CPU kernel) --
    the same physics as _run_3d_oblique (2D transverse Bloch envelope d/dx->d/dx+i kx, d/dy->d/dy+i ky;
    semi-implicit Drude ADE per E-component; CFS-CPML + PEC in z; s-pol plane source on (sx,sy)), but
    explicit-loop + compiled. complex128; (kx,ky)=0 reduces to the real normal-incidence response. Returns
    the complex Ex,Ey probe planes at the left/right z-planes."""
    nx, ny, nz = eps_inf.shape
    c0 = np.complex128(0.0)
    Ex = np.zeros((nx, ny, nz), dtype=np.complex128); Ey = np.zeros((nx, ny, nz), dtype=np.complex128)
    Ez = np.zeros((nx, ny, nz), dtype=np.complex128)
    Hx = np.zeros((nx, ny, nz), dtype=np.complex128); Hy = np.zeros((nx, ny, nz), dtype=np.complex128)
    Hz = np.zeros((nx, ny, nz), dtype=np.complex128)
    Jx = np.zeros((nx, ny, nz), dtype=np.complex128); Jy = np.zeros((nx, ny, nz), dtype=np.complex128)
    Jz = np.zeros((nx, ny, nz), dtype=np.complex128)
    psi_Hx = np.zeros((nx, ny, nz), dtype=np.complex128); psi_Hy = np.zeros((nx, ny, nz), dtype=np.complex128)
    psi_Ex = np.zeros((nx, ny, nz), dtype=np.complex128); psi_Ey = np.zeros((nx, ny, nz), dtype=np.complex128)
    exL = np.empty((nsteps, nx, ny), dtype=np.complex128); eyL = np.empty((nsteps, nx, ny), dtype=np.complex128)
    exR = np.empty((nsteps, nx, ny), dtype=np.complex128); eyR = np.empty((nsteps, nx, ny), dtype=np.complex128)
    cmu = dt / MU0; e0dt = EPS0 / dt
    ikx = 1j * kx; iky = 1j * ky
    for n in range(nsteps):
        for i in prange(nx):                                  # ---- H update (reads old E) ----
            ip1 = i + 1 if i + 1 < nx else 0
            for j in range(ny):
                jp1 = j + 1 if j + 1 < ny else 0
                for k in range(nz):
                    dyfEz = (Ez[i, jp1, k] - Ez[i, j, k]) / dy + iky * Ez[i, j, k]
                    sEy = c0
                    if k < nz - 1:
                        dEy_dz = (Ey[i, j, k + 1] - Ey[i, j, k]) / dz
                        psi_Hx[i, j, k] = bh[k] * psi_Hx[i, j, k] + ch[k] * dEy_dz
                        sEy = dEy_dz / kh[k] + psi_Hx[i, j, k]
                    Hx[i, j, k] -= cmu * (dyfEz - sEy)
                    dxfEz = (Ez[ip1, j, k] - Ez[i, j, k]) / dx + ikx * Ez[i, j, k]
                    sEx = c0
                    if k < nz - 1:
                        dEx_dz = (Ex[i, j, k + 1] - Ex[i, j, k]) / dz
                        psi_Hy[i, j, k] = bh[k] * psi_Hy[i, j, k] + ch[k] * dEx_dz
                        sEx = dEx_dz / kh[k] + psi_Hy[i, j, k]
                    Hy[i, j, k] -= cmu * (sEx - dxfEz)
                    dxfEy = (Ey[ip1, j, k] - Ey[i, j, k]) / dx + ikx * Ey[i, j, k]
                    dyfEx = (Ex[i, jp1, k] - Ex[i, j, k]) / dy + iky * Ex[i, j, k]
                    Hz[i, j, k] -= cmu * (dxfEy - dyfEx)
        for i in prange(nx):                                  # ---- E update (reads new H, old E/J) ----
            im1 = i - 1 if i - 1 >= 0 else nx - 1
            for j in range(ny):
                jm1 = j - 1 if j - 1 >= 0 else ny - 1
                for k in range(nz):
                    aJ = (1.0 - gam[i, j, k] * dt / 2.0) / (1.0 + gam[i, j, k] * dt / 2.0)
                    bJ = (EPS0 * wp[i, j, k] ** 2 * dt / 2.0) / (1.0 + gam[i, j, k] * dt / 2.0)
                    denom = e0dt * eps_inf[i, j, k] + bJ / 2.0
                    dybHz = (Hz[i, j, k] - Hz[i, jm1, k]) / dy + iky * Hz[i, j, k]
                    sHy = c0
                    if k >= 1:
                        dHy_dz = (Hy[i, j, k] - Hy[i, j, k - 1]) / dz
                        psi_Ex[i, j, k] = be[k] * psi_Ex[i, j, k] + ce[k] * dHy_dz
                        sHy = dHy_dz / ke[k] + psi_Ex[i, j, k]
                    curlx = dybHz - sHy
                    exo = Ex[i, j, k]
                    exn = (e0dt * eps_inf[i, j, k] * exo + curlx - 0.5 * (1.0 + aJ) * Jx[i, j, k] - 0.5 * bJ * exo) / denom
                    Jx[i, j, k] = aJ * Jx[i, j, k] + bJ * (exn + exo)
                    dxbHz = (Hz[i, j, k] - Hz[im1, j, k]) / dx + ikx * Hz[i, j, k]
                    sHx = c0
                    if k >= 1:
                        dHx_dz = (Hx[i, j, k] - Hx[i, j, k - 1]) / dz
                        psi_Ey[i, j, k] = be[k] * psi_Ey[i, j, k] + ce[k] * dHx_dz
                        sHx = dHx_dz / ke[k] + psi_Ey[i, j, k]
                    curly = sHx - dxbHz
                    eyo = Ey[i, j, k]
                    eyn = (e0dt * eps_inf[i, j, k] * eyo + curly - 0.5 * (1.0 + aJ) * Jy[i, j, k] - 0.5 * bJ * eyo) / denom
                    Jy[i, j, k] = aJ * Jy[i, j, k] + bJ * (eyn + eyo)
                    dxbHy = (Hy[i, j, k] - Hy[im1, j, k]) / dx + ikx * Hy[i, j, k]
                    dybHx = (Hx[i, j, k] - Hx[i, jm1, k]) / dy + iky * Hx[i, j, k]
                    curlz = dxbHy - dybHx
                    ezo = Ez[i, j, k]
                    ezn = (e0dt * eps_inf[i, j, k] * ezo + curlz - 0.5 * (1.0 + aJ) * Jz[i, j, k] - 0.5 * bJ * ezo) / denom
                    Jz[i, j, k] = aJ * Jz[i, j, k] + bJ * (ezn + ezo)
                    Ex[i, j, k] = exn; Ey[i, j, k] = eyn; Ez[i, j, k] = ezn
        for i in prange(nx):                                  # source + PEC + probes
            for j in range(ny):
                Ex[i, j, k_src] += sx * src[n]; Ey[i, j, k_src] += sy * src[n]
                Ex[i, j, 0] = c0; Ex[i, j, nz - 1] = c0
                Ey[i, j, 0] = c0; Ey[i, j, nz - 1] = c0
                exL[n, i, j] = Ex[i, j, k_pL]; eyL[n, i, j] = Ey[i, j, k_pL]
                exR[n, i, j] = Ex[i, j, k_pR]; eyR[n, i, j] = Ey[i, j, k_pR]
    return exL, eyL, exR, eyR




@njit(parallel=True, fastmath=True, cache=True)
def _te3d_numba(eps_inf, wp, gam, chi3, ke, be, ce, kh, bh, ch, dx, dy, dz, dt,
                nsteps, k_src, k_pL, k_pR, src, C1, C2, C3, has_lor):
    """Fused, prange-threaded full-vector 3D timestep (the Numba CPU kernel) -- byte-near-identical physics
    to run_3d (six-component Yee + per-component semi-implicit Drude ADE + Kerr + Lorentz ADE + CFS-CPML in
    z + PEC, Bloch-periodic x,y), but explicit-loop + JIT-compiled so the whole step is ONE compiled pass.
    C1,C2,C3 = per-cell Lorentz coefficients, has_lor gates the per-component polarization PL{x,y,z}.
    Parallel-safe over x: the H-phase writes Hx/Hy/Hz[i] (disjoint) reading only E (read-only); the E-phase
    writes Ex/Ey/Ez[i] (disjoint) reading only H. Returns the Ex,Ey,Hx,Hy probe planes (left/right)."""
    nx, ny, nz = eps_inf.shape
    Ex = np.zeros((nx, ny, nz)); Ey = np.zeros((nx, ny, nz)); Ez = np.zeros((nx, ny, nz))
    Hx = np.zeros((nx, ny, nz)); Hy = np.zeros((nx, ny, nz)); Hz = np.zeros((nx, ny, nz))
    Jx = np.zeros((nx, ny, nz)); Jy = np.zeros((nx, ny, nz)); Jz = np.zeros((nx, ny, nz))
    # audit S6-8: the six Lorentz polarization-state grids are only read inside `if has_lor`
    # branches; with the pole off (~157 MB dead at 64x64x800) allocate 1-element stand-ins
    # (never indexed on the dead branch; bit-identical when the pole is on).
    nlx = nx if has_lor else 1
    nly = ny if has_lor else 1
    nlz = nz if has_lor else 1
    PLx = np.zeros((nlx, nly, nlz)); PLy = np.zeros((nlx, nly, nlz)); PLz = np.zeros((nlx, nly, nlz))
    PLpx = np.zeros((nlx, nly, nlz)); PLpy = np.zeros((nlx, nly, nlz)); PLpz = np.zeros((nlx, nly, nlz))
    psi_Hx = np.zeros((nx, ny, nz)); psi_Hy = np.zeros((nx, ny, nz))
    psi_Ex = np.zeros((nx, ny, nz)); psi_Ey = np.zeros((nx, ny, nz))
    sh = (nsteps, nx, ny)
    exL = np.empty(sh); eyL = np.empty(sh); hxL = np.empty(sh); hyL = np.empty(sh)
    exR = np.empty(sh); eyR = np.empty(sh); hxR = np.empty(sh); hyR = np.empty(sh)
    cmu = dt / MU0
    e0dt = EPS0 / dt
    # audit S2-5/S6-6/S6-7: hoist the per-cell time-invariants (Drude aJ/bJ; the Kerr-off
    # E-denominator) out of the time loop -- measured 1.41x on this kernel, bit-identical.
    aJg = np.empty((nx, ny, nz)); bJg = np.empty((nx, ny, nz))
    ee0 = np.empty((nx, ny, nz)); den0 = np.empty((nx, ny, nz))
    has_kerr = False
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                if chi3[i, j, k] != 0.0:
                    has_kerr = True
    for i in prange(nx):
        for j in range(ny):
            for k in range(nz):
                g0 = gam[i, j, k]
                aJg[i, j, k] = (1.0 - g0 * dt / 2.0) / (1.0 + g0 * dt / 2.0)
                bJg[i, j, k] = (EPS0 * wp[i, j, k] ** 2 * dt / 2.0) / (1.0 + g0 * dt / 2.0)
                ee0[i, j, k] = e0dt * eps_inf[i, j, k]
                den0[i, j, k] = ee0[i, j, k] + bJg[i, j, k] / 2.0
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
                    aJ = aJg[i, j, k]
                    bJ = bJg[i, j, k]
                    if has_kerr:
                        eps_eff = eps_inf[i, j, k] + 3.0 * chi3[i, j, k] * (exo * exo + eyo * eyo + ezo * ezo)  # C3-2
                        e0e = e0dt * eps_eff
                        denom = e0e + bJ / 2.0
                    else:
                        e0e = ee0[i, j, k]
                        denom = den0[i, j, k]
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
                    exn = (e0e * exo + cx - coef * Jx[i, j, k] - 0.5 * bJ * exo) / denom
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
                    eyn = (e0e * eyo + cy - coef * Jy[i, j, k] - 0.5 * bJ * eyo) / denom
                    Jy[i, j, k] = aJ * Jy[i, j, k] + bJ * (eyn + eyo)
                    # Ez: curl_z H = dHy/dx - dHx/dy (transverse, no CPML)
                    cz = (Hy[i, j, k] - Hy[im1, j, k]) / dx - (Hx[i, j, k] - Hx[i, jm1, k]) / dy
                    if has_lor:
                        pln = C1[i, j, k] * PLz[i, j, k] + C2[i, j, k] * PLpz[i, j, k] + C3[i, j, k] * ezo
                        cz = cz - (pln - PLz[i, j, k]) / dt
                        PLpz[i, j, k] = PLz[i, j, k]; PLz[i, j, k] = pln
                    ezn = (e0e * ezo + cz - coef * Jz[i, j, k] - 0.5 * bJ * ezo) / denom
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


