"""2D-TE normal-incidence numba CPU kernel and numba-CUDA kernel + host driver.

Split from the former monolithic fdtd_nd.py; see the package __init__ docstring
for conventions. Bodies are verbatim from the original module.
"""
from __future__ import annotations

import numpy as np

from dynameta.constants import EPS0, MU0
from dynameta.optics.fdtd_nd.backends import _HAVE_NUMBA, njit, prange



@njit(parallel=True, fastmath=True, cache=True)
def _te2d_numba(eps_inf, wp, gam, chi3, ke, be, ce, kh, bh, ch, dx, dz, dt,
                nsteps, k_src, k_pL, k_pR, src, C1, C2, C3, has_lor,
                chi2g, has_chi2, R1, R2, R3, chi3R, has_raman, G1, G2, G3, has_gain):
    """Fused, prange-threaded 2D TE timestep (the Numba CPU kernel) -- byte-for-byte the same physics as
    _run_2d_te (Yee + semi-implicit Drude ADE + Kerr + Lorentz ADE + CFS-CPML in z + PEC backing, periodic
    in x), but explicit-loop + JIT-compiled so the whole step is ONE compiled pass with no per-op overhead.
    C1,C2,C3 = per-cell Lorentz ADE coefficients; has_lor gates the extra pole. R15/R20 nonlinearities
    mirror the numpy kernel: chi2 SHG polarization P2 = eps0 chi2 E^2 (lagged dP2/dt), the Raman pair
    (vibrational ADE on E^2 + P_R = eps0 chi3R E Q) and the clamped-inversion gain line (G1,G2,G3
    recursion) -- each gated by its has_* flag (zero-cost branches when off). Returns the E_y /
    co-located H_x probe x-lines at the left/right z-planes."""
    nx, nz = eps_inf.shape
    Ey = np.zeros((nx, nz)); Hx = np.zeros((nx, nz)); Hz = np.zeros((nx, nz))
    Jy = np.zeros((nx, nz)); psi_hxz = np.zeros((nx, nz)); psi_eyz = np.zeros((nx, nz))
    PL = np.zeros((nx, nz)); PLp = np.zeros((nx, nz))           # Lorentz polarization (now / previous)
    P2 = np.zeros((nx, nz))                                     # chi2 SHG polarization
    Q = np.zeros((nx, nz)); Qp = np.zeros((nx, nz)); PR = np.zeros((nx, nz))  # Raman state
    PG = np.zeros((nx, nz)); PGp = np.zeros((nx, nz))           # gain-line polarization
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
                if has_gain:                                   # R20 clamped-inversion gain line
                    pgn = G1[i, k] * PG[i, k] + G2[i, k] * PGp[i, k] + G3[i, k] * eyo
                    curl = curl - (pgn - PG[i, k]) / dt
                    PGp[i, k] = PG[i, k]; PG[i, k] = pgn
                if has_chi2:                                   # R15 chi2 SHG polarization
                    p2n = EPS0 * chi2g[i, k] * eyo * eyo
                    curl = curl - (p2n - P2[i, k]) / dt
                    P2[i, k] = p2n
                if has_raman:                                  # R15 Raman: ADE on E^2 + P_R = eps0 chiR E Q
                    qn = R1[i, k] * Q[i, k] + R2[i, k] * Qp[i, k] + R3[i, k] * eyo * eyo
                    prn = EPS0 * chi3R[i, k] * eyo * qn
                    curl = curl - (prn - PR[i, k]) / dt
                    Qp[i, k] = Q[i, k]; Q[i, k] = qn; PR[i, k] = prn
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


# --- numba-CUDA 2D-TE: a PERSISTENT cooperative-groups kernel. The Yee half-steps are cell-parallel
#     (each phase writes its OWN field and reads only the other), so a CHUNK of timesteps runs in ONE
#     cooperative launch with a grid-wide grid.sync() between the H, E, and source/probe phases -- no
#     per-timestep launch overhead and no host<->device round-trips within a chunk. Grid-stride over cells.
#     Same physics as _te2d_numba; byte-matches to the float64 FMA floor (~1e-12). PERF: cooperative launch
#     caps the grid at the co-resident block count (~#SMs), so this WINS on small / unit-cell grids (~3.4x
#     vs the threaded CPU on a metasurface unit cell @ RTX 4070 Ti) but is occupancy-limited on very large
#     volumes -- prefer the 'cupy' backend (unlimited blocks) there. ---
if _HAVE_NUMBA:
    from numba import cuda as _cuda

    @_cuda.jit(fastmath=True)
    def _te2d_coop_cuda(Ey, Hx, Hz, Jy, psi_hxz, psi_eyz, PL, PLp,
                        P2, Q, Qp, PR, PG, PGp,
                        eps_inf, wp, gam, chi3, C1, C2, C3,
                        chi2g, R1, R2, R3, chi3R, G1, G2, G3,
                        ke, be, ce, kh, bh, ch, d_src,
                        eyL, hxL, eyR, hxR, dx, dz, dt, cmu, e0dt, eps0,
                        n0, ns, k_src, k_pL, k_pR, has_lor, has_chi2, has_raman, has_gain,
                        nx, nz):
        # runs timesteps [n0, n0+ns) -- the host loops over CHUNKS so no single cooperative launch exceeds
        # the WDDM TDR watchdog (~2 s on a display GPU); the field state persists on-device between chunks.
        grid = _cuda.cg.this_grid()
        tid = _cuda.grid(1)
        nthreads = _cuda.gridsize(1)
        ncell = nx * nz
        for nn in range(ns):
            n = n0 + nn
            idx = tid                                          # ---- H phase (reads old Ey) ----
            while idx < ncell:
                i = idx // nz; k = idx % nz
                ip1 = i + 1 if i + 1 < nx else 0
                if k < nz - 1:
                    d = (Ey[i, k + 1] - Ey[i, k]) / dz
                    psi_hxz[i, k] = bh[k] * psi_hxz[i, k] + ch[k] * d
                    Hx[i, k] += cmu * (d / kh[k] + psi_hxz[i, k])
                Hz[i, k] += -cmu * (Ey[ip1, k] - Ey[i, k]) / dx
                idx += nthreads
            grid.sync()
            idx = tid                                          # ---- E phase (reads new H, old Ey/J) ----
            while idx < ncell:
                i = idx // nz; k = idx % nz
                if 1 <= k <= nz - 2:
                    im1 = i - 1 if i - 1 >= 0 else nx - 1
                    dHxz = (Hx[i, k] - Hx[i, k - 1]) / dz
                    psi_eyz[i, k] = be[k] * psi_eyz[i, k] + ce[k] * dHxz
                    curl = dHxz / ke[k] + psi_eyz[i, k] - (Hz[i, k] - Hz[im1, k]) / dx
                    eyo = Ey[i, k]
                    if has_lor:
                        pln = C1[i, k] * PL[i, k] + C2[i, k] * PLp[i, k] + C3[i, k] * eyo
                        curl = curl - (pln - PL[i, k]) / dt
                        PLp[i, k] = PL[i, k]; PL[i, k] = pln
                    if has_gain:                               # R20 clamped-inversion gain line
                        pgn = G1[i, k] * PG[i, k] + G2[i, k] * PGp[i, k] + G3[i, k] * eyo
                        curl = curl - (pgn - PG[i, k]) / dt
                        PGp[i, k] = PG[i, k]; PG[i, k] = pgn
                    if has_chi2:                               # R15 chi2 SHG polarization
                        p2n = eps0 * chi2g[i, k] * eyo * eyo
                        curl = curl - (p2n - P2[i, k]) / dt
                        P2[i, k] = p2n
                    if has_raman:                              # R15 Raman: ADE on E^2 + P_R = eps0 chiR E Q
                        qn = R1[i, k] * Q[i, k] + R2[i, k] * Qp[i, k] + R3[i, k] * eyo * eyo
                        prn = eps0 * chi3R[i, k] * eyo * qn
                        curl = curl - (prn - PR[i, k]) / dt
                        Qp[i, k] = Q[i, k]; Q[i, k] = qn; PR[i, k] = prn
                    aJ = (1.0 - gam[i, k] * dt / 2.0) / (1.0 + gam[i, k] * dt / 2.0)
                    bJ = (eps0 * wp[i, k] ** 2 * dt / 2.0) / (1.0 + gam[i, k] * dt / 2.0)
                    eps_eff = eps_inf[i, k] + chi3[i, k] * eyo * eyo
                    denom = e0dt * eps_eff + bJ / 2.0
                    eyn = (e0dt * eps_eff * eyo + curl - 0.5 * (1.0 + aJ) * Jy[i, k] - 0.5 * bJ * eyo) / denom
                    Jy[i, k] = aJ * Jy[i, k] + bJ * (eyn + eyo)
                    Ey[i, k] = eyn
                idx += nthreads
            grid.sync()
            ii = tid                                           # ---- source + PEC + probe (per column) ----
            while ii < nx:
                Ey[ii, k_src] += d_src[n]
                Ey[ii, 0] = 0.0; Ey[ii, nz - 1] = 0.0
                eyL[n, ii] = Ey[ii, k_pL]; hxL[n, ii] = 0.5 * (Hx[ii, k_pL] + Hx[ii, k_pL - 1])
                eyR[n, ii] = Ey[ii, k_pR]; hxR[n, ii] = 0.5 * (Hx[ii, k_pR] + Hx[ii, k_pR - 1])
                ii += nthreads
            grid.sync()


def _te2d_cuda(eps_inf, wp, gam, chi3, ke, be, ce, kh, bh, ch, dx, dz, dt,
               nsteps, k_src, k_pL, k_pR, src, C1, C2, C3, has_lor,
               chi2g, has_chi2, R1, R2, R3, chi3R, has_raman, G1, G2, G3, has_gain):
    """Host driver for the numba-CUDA 2D-TE FDTD: upload the per-cell profiles + CPML + source to the
    device, run the persistent cooperative-groups kernel in CHUNKS of timesteps (the field state persists
    on-device between chunks; only the probe planes accumulate), copy the probes back. Same result as
    _te2d_numba to the float64 FMA floor (incl. the R15/R20 chi2/Raman/gain nonlinearities, which are
    cell-local recurrences -- no extra grid syncs needed). Chunking keeps each cooperative launch under
    the WDDM TDR watchdog (~2 s on a display GPU) while still amortizing the launch over ~chunk steps.
    Cooperative launch -> blocks must be co-resident: use <= #SMs blocks (1/SM, always safe) with
    grid-stride."""
    from numba import cuda
    nx, nz = eps_inf.shape
    dev = (lambda a: cuda.to_device(np.ascontiguousarray(a, dtype=np.float64)))
    Ey = dev(np.zeros((nx, nz))); Hx = dev(np.zeros((nx, nz))); Hz = dev(np.zeros((nx, nz)))
    Jy = dev(np.zeros((nx, nz))); psi_hxz = dev(np.zeros((nx, nz))); psi_eyz = dev(np.zeros((nx, nz)))
    PL = dev(np.zeros((nx, nz))); PLp = dev(np.zeros((nx, nz)))
    P2 = dev(np.zeros((nx, nz))); Q = dev(np.zeros((nx, nz))); Qp = dev(np.zeros((nx, nz)))
    PR = dev(np.zeros((nx, nz))); PG = dev(np.zeros((nx, nz))); PGp = dev(np.zeros((nx, nz)))
    g_eps, g_wp, g_gam, g_chi3 = dev(eps_inf), dev(wp), dev(gam), dev(chi3)
    g_C1, g_C2, g_C3 = dev(C1), dev(C2), dev(C3)
    g_x2, g_R1, g_R2, g_R3, g_xR = dev(chi2g), dev(R1), dev(R2), dev(R3), dev(chi3R)
    g_G1, g_G2, g_G3 = dev(G1), dev(G2), dev(G3)
    g_ke, g_be, g_ce = dev(ke), dev(be), dev(ce)
    g_kh, g_bh, g_ch = dev(kh), dev(bh), dev(ch)
    g_src = dev(src)
    eyL = cuda.device_array((nsteps, nx)); hxL = cuda.device_array((nsteps, nx))
    eyR = cuda.device_array((nsteps, nx)); hxR = cuda.device_array((nsteps, nx))
    cmu = dt / MU0; e0dt = EPS0 / dt
    tpb = 128
    dev_obj = cuda.get_current_device()
    sm = dev_obj.MULTIPROCESSOR_COUNT
    need = (nx * nz + tpb - 1) // tpb
    # MORE blocks (up to the kernel's co-resident occupancy) -> better GPU utilization on large grids.
    # Size by registers/SM: blocks_per_sm = (regs_per_SM // (regs_per_thread * tpb)), also <= threads/SM.
    try:
        regs = max(1, _te2d_coop_cuda.get_regs_per_thread())
        regs_per_sm = getattr(dev_obj, "MAX_REGISTERS_PER_MULTIPROCESSOR", 65536) or 65536
        thr_per_sm = getattr(dev_obj, "MAX_THREADS_PER_MULTI_PROCESSOR", 1536) or 1536
        bps = max(1, min(regs_per_sm // (regs * tpb), thr_per_sm // tpb))
    except Exception:
        bps = 1
    blocks = max(1, min(sm * bps, need))
    chunk = int(min(nsteps, max(8, 20_000_000 // max(1, nx * nz))))

    def _launch(bk):
        n0 = 0
        while n0 < nsteps:
            ns = min(chunk, nsteps - n0)
            _te2d_coop_cuda[bk, tpb](Ey, Hx, Hz, Jy, psi_hxz, psi_eyz, PL, PLp,
                                     P2, Q, Qp, PR, PG, PGp,
                                     g_eps, g_wp, g_gam, g_chi3, g_C1, g_C2, g_C3,
                                     g_x2, g_R1, g_R2, g_R3, g_xR, g_G1, g_G2, g_G3,
                                     g_ke, g_be, g_ce, g_kh, g_bh, g_ch, g_src,
                                     eyL, hxL, eyR, hxR, dx, dz, dt, cmu, e0dt, EPS0,
                                     n0, ns, k_src, k_pL, k_pR, has_lor, has_chi2,
                                     has_raman, has_gain, nx, nz)
            cuda.synchronize()
            n0 += ns

    while True:                                               # shrink blocks if the cooperative grid is too big
        try:
            _launch(blocks)
            break
        except Exception:
            if blocks <= sm:
                blocks = max(1, min(sm, need))
                _launch(blocks)                               # last resort: <= #SMs (always co-resident)
                break
            blocks = max(sm, blocks // 2)
    return eyL.copy_to_host(), hxL.copy_to_host(), eyR.copy_to_host(), hxR.copy_to_host()
