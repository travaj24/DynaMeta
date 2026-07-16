"""2D oblique (Bloch complex-envelope) kernels, all backends, + the oblique dispatcher.

Split from the former monolithic fdtd_nd.py; see the package __init__ docstring
for conventions. Bodies are verbatim from the original module.
"""
from __future__ import annotations

import numpy as np

from dynameta.constants import EPS0, MU0
from dynameta.optics.fdtd_nd.backends import njit



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


def _run_2d_tm_oblique(eps_inf, wp, gam, dx, dz, dt, nsteps, k_src, k_pL, k_pR, src, cpml, kx):
    """Complex-envelope oblique 2D-TM (p-pol): the in-plane E (Ex, Ez) + the single H_y, all the PERIODIC
    Bloch envelope (physical field = envelope * exp(i kx x)), so every x-derivative gains a + i kx term.
    mu0 dHy/dt = dEz/dx - dEx/dz ; eps dEx/dt = -dHy/dz - Jx ; eps dEz/dt = +dHy/dx - Jz, with the
    semi-implicit Drude ADE on BOTH transverse + longitudinal E-components (Jx, Jz) and CFS-CPML on the z
    derivatives. Source + PEC on the tangential Ex (the dual of the TE Ey). Records the complex tangential
    Ex and co-located Hy probe x-lines (the p-pol R/T come from the Ex up/down ratio, like TE's Ey)."""
    nx, nz = eps_inf.shape
    (ke, be, ce), (kh, bh, ch) = cpml
    z = (lambda: np.zeros((nx, nz), dtype=complex))
    Ex, Ez, Hy, Jx, Jz, psi_hyz, psi_exz = z(), z(), z(), z(), z(), z(), z()
    aJ = (1.0 - gam * dt / 2.0) / (1.0 + gam * dt / 2.0)
    bJ = (EPS0 * wp ** 2 * dt / 2.0) / (1.0 + gam * dt / 2.0)
    exL = np.empty((nsteps, nx), complex); hyL = np.empty((nsteps, nx), complex)
    exR = np.empty((nsteps, nx), complex); hyR = np.empty((nsteps, nx), complex)
    cmu = dt / MU0; ikx = 1j * kx; e0dt = EPS0 / dt
    for n in range(nsteps):
        # H update: mu0 dHy/dt = dEz/dx - dEx/dz  (Hy at k+1/2; CPML-stretched dEx/dz)
        dEx_dz = (Ex[:, 1:] - Ex[:, :-1]) / dz
        psi_hyz[:, :-1] = bh[:-1] * psi_hyz[:, :-1] + ch[:-1] * dEx_dz
        dEz_dx = (np.roll(Ez, -1, axis=0) - Ez) / dx + ikx * Ez            # dEz/dx -> d/dx + i kx
        Hy[:, :-1] += cmu * (dEz_dx[:, :-1] - (dEx_dz / kh[:-1] + psi_hyz[:, :-1]))
        # E update -- Ex: eps dEx/dt = -dHy/dz (CPML-stretched) - Jx
        dHy_dz = (Hy[:, 1:] - Hy[:, :-1]) / dz
        psi_exz[:, 1:] = be[1:] * psi_exz[:, 1:] + ce[1:] * dHy_dz
        curlx = np.zeros((nx, nz), complex)
        curlx[:, 1:] = -(dHy_dz / ke[1:] + psi_exz[:, 1:])
        denom = e0dt * eps_inf + bJ / 2.0
        Exn = (e0dt * eps_inf * Ex + curlx - 0.5 * (1.0 + aJ) * Jx - 0.5 * bJ * Ex) / denom
        Jx = aJ * Jx + bJ * (Exn + Ex)
        Exn[:, k_src] += src[n]
        Exn[:, 0] = 0.0; Exn[:, -1] = 0.0                                  # PEC backing (tangential E)
        Ex = Exn
        # E update -- Ez: eps dEz/dt = +dHy/dx (periodic x) - Jz
        curlz = (Hy - np.roll(Hy, 1, axis=0)) / dx + ikx * Hy             # dHy/dx -> d/dx + i kx
        Ezn = (e0dt * eps_inf * Ez + curlz - 0.5 * (1.0 + aJ) * Jz - 0.5 * bJ * Ez) / denom
        Jz = aJ * Jz + bJ * (Ezn + Ez)
        Ez = Ezn
        exL[n] = Ex[:, k_pL]; hyL[n] = 0.5 * (Hy[:, k_pL] + Hy[:, k_pL - 1])
        exR[n] = Ex[:, k_pR]; hyR[n] = 0.5 * (Hy[:, k_pR] + Hy[:, k_pR - 1])
    return exL, hyL, exR, hyR


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


@njit(fastmath=True, cache=True)
def _tm2d_oblique_numba(eps_inf, wp, gam, ke, be, ce, kh, bh, ch, dx, dz, dt,
                        nsteps, k_src, k_pL, k_pR, src, kx):
    """Fused, JIT-compiled COMPLEX-ENVELOPE oblique 2D-TM (p-pol) timestep -- the same physics as
    _run_2d_tm_oblique (in-plane Ex,Ez + Hy, Bloch envelope d/dx -> d/dx + i kx, semi-implicit Drude ADE on
    BOTH Jx,Jz, CFS-CPML + PEC in z), explicit-loop + compiled so the whole step is ONE pass. Fields
    complex128; kx=0 reduces to the real normal-incidence TM response. SERIAL (nx small for the smooth
    oblique envelope). Returns the complex tangential Ex / co-located Hy probe x-lines (the p-pol R/T come
    from the Ex up/down ratio, the dual of TE's Ey)."""
    nx, nz = eps_inf.shape
    Ex = np.zeros((nx, nz), dtype=np.complex128); Ez = np.zeros((nx, nz), dtype=np.complex128)
    Hy = np.zeros((nx, nz), dtype=np.complex128)
    Jx = np.zeros((nx, nz), dtype=np.complex128); Jz = np.zeros((nx, nz), dtype=np.complex128)
    psi_hyz = np.zeros((nx, nz), dtype=np.complex128); psi_exz = np.zeros((nx, nz), dtype=np.complex128)
    exL = np.empty((nsteps, nx), dtype=np.complex128); hyL = np.empty((nsteps, nx), dtype=np.complex128)
    exR = np.empty((nsteps, nx), dtype=np.complex128); hyR = np.empty((nsteps, nx), dtype=np.complex128)
    cmu = dt / MU0
    e0dt = EPS0 / dt
    ikx = 1j * kx
    for n in range(nsteps):
        for i in range(nx):                                    # H update: mu0 dHy/dt = dEz/dx - dEx/dz
            ip1 = i + 1 if i + 1 < nx else 0
            for k in range(nz - 1):
                dexz = (Ex[i, k + 1] - Ex[i, k]) / dz
                psi_hyz[i, k] = bh[k] * psi_hyz[i, k] + ch[k] * dexz
                dezx = (Ez[ip1, k] - Ez[i, k]) / dx + ikx * Ez[i, k]
                Hy[i, k] += cmu * (dezx - (dexz / kh[k] + psi_hyz[i, k]))
        for i in range(nx):                                    # E update Ex: eps dEx/dt = -dHy/dz - Jx
            for k in range(1, nz - 1):
                dHyz = (Hy[i, k] - Hy[i, k - 1]) / dz
                psi_exz[i, k] = be[k] * psi_exz[i, k] + ce[k] * dHyz
                curlx = -(dHyz / ke[k] + psi_exz[i, k])
                aJ = (1.0 - gam[i, k] * dt / 2.0) / (1.0 + gam[i, k] * dt / 2.0)
                bJ = (EPS0 * wp[i, k] ** 2 * dt / 2.0) / (1.0 + gam[i, k] * dt / 2.0)
                denom = e0dt * eps_inf[i, k] + bJ / 2.0
                exo = Ex[i, k]
                exn = (e0dt * eps_inf[i, k] * exo + curlx - 0.5 * (1.0 + aJ) * Jx[i, k] - 0.5 * bJ * exo) / denom
                Jx[i, k] = aJ * Jx[i, k] + bJ * (exn + exo)
                Ex[i, k] = exn
        for i in range(nx):                                    # soft source + PEC backing (tangential Ex)
            Ex[i, k_src] += src[n]
            Ex[i, 0] = 0.0 + 0.0j; Ex[i, nz - 1] = 0.0 + 0.0j
        for i in range(nx):                                    # E update Ez: eps dEz/dt = +dHy/dx - Jz
            im1 = i - 1 if i - 1 >= 0 else nx - 1
            for k in range(nz):
                curlz = (Hy[i, k] - Hy[im1, k]) / dx + ikx * Hy[i, k]
                aJ = (1.0 - gam[i, k] * dt / 2.0) / (1.0 + gam[i, k] * dt / 2.0)
                bJ = (EPS0 * wp[i, k] ** 2 * dt / 2.0) / (1.0 + gam[i, k] * dt / 2.0)
                denom = e0dt * eps_inf[i, k] + bJ / 2.0
                ezo = Ez[i, k]
                ezn = (e0dt * eps_inf[i, k] * ezo + curlz - 0.5 * (1.0 + aJ) * Jz[i, k] - 0.5 * bJ * ezo) / denom
                Jz[i, k] = aJ * Jz[i, k] + bJ * (ezn + ezo)
                Ez[i, k] = ezn
        for i in range(nx):                                    # co-located probes (Hy averaged to E plane)
            exL[n, i] = Ex[i, k_pL]; hyL[n, i] = 0.5 * (Hy[i, k_pL] + Hy[i, k_pL - 1])
            exR[n, i] = Ex[i, k_pR]; hyR[n, i] = 0.5 * (Hy[i, k_pR] + Hy[i, k_pR - 1])
    return exL, hyL, exR, hyR


def run_2d_te_oblique_jax(eps_inf, wp, gam, dx, dz, dt, nsteps, k_src, k_pL, k_pR, src, cpml, kx):
    """JAX (XLA, lax.scan) twin of _run_2d_te_oblique (s-pol complex-envelope Bloch): the SAME physics
    (d/dx -> d/dx + i kx, semi-implicit Drude ADE, CFS-CPML + PEC in z), as a single traced/compiled
    DIFFERENTIABLE complex128 time loop -- so jax.grad flows d(R,T)/d(geometry/material) straight through
    the oblique scan (the inverse-design path at angle). kx=0 with a real source stays real. Byte-equal to
    the NumPy kernel to ~1e-12. Drude only (no Lorentz/Kerr -- the oblique kernel carries Drude)."""
    import jax
    jax.config.update("jax_enable_x64", True)               # complex128 + match the reference
    import jax.numpy as jnp
    from jax import lax
    (ke, be, ce), (kh, bh, ch) = cpml
    ke, be, ce = jnp.asarray(ke), jnp.asarray(be), jnp.asarray(ce)
    kh, bh, ch = jnp.asarray(kh), jnp.asarray(bh), jnp.asarray(ch)
    eps_inf = jnp.asarray(eps_inf); gam = jnp.asarray(gam); wp = jnp.asarray(wp)
    aJ = (1.0 - gam * dt / 2.0) / (1.0 + gam * dt / 2.0)
    bJ = (EPS0 * wp ** 2 * dt / 2.0) / (1.0 + gam * dt / 2.0)
    nx, nz = eps_inf.shape
    cmu = dt / MU0; ikx = 1j * kx

    def step(carry, src_n):
        Ey, Hx, Hz, Jy, psi_h, psi_e = carry
        dEy_dz = (Ey[:, 1:] - Ey[:, :-1]) / dz
        psi_h = psi_h.at[:, :-1].set(bh[:-1] * psi_h[:, :-1] + ch[:-1] * dEy_dz)
        Hx = Hx.at[:, :-1].add(cmu * (dEy_dz / kh[:-1] + psi_h[:, :-1]))
        Hz = Hz - cmu * ((jnp.roll(Ey, -1, axis=0) - Ey) / dx + ikx * Ey)
        dHx_dz = (Hx[:, 1:] - Hx[:, :-1]) / dz
        psi_e = psi_e.at[:, 1:].set(be[1:] * psi_e[:, 1:] + ce[1:] * dHx_dz)
        curl = jnp.zeros((nx, nz), dtype=jnp.complex128)
        curl = curl.at[:, 1:].add(dHx_dz / ke[1:] + psi_e[:, 1:])
        curl = curl - ((Hz - jnp.roll(Hz, 1, axis=0)) / dx + ikx * Hz)
        denom = EPS0 * eps_inf / dt + bJ / 2.0
        Eyn = (EPS0 * eps_inf / dt * Ey + curl - 0.5 * (1.0 + aJ) * Jy - 0.5 * bJ * Ey) / denom
        Jy = aJ * Jy + bJ * (Eyn + Ey)
        Eyn = Eyn.at[:, k_src].add(src_n)
        Eyn = Eyn.at[:, 0].set(0.0 + 0.0j).at[:, nz - 1].set(0.0 + 0.0j)
        out = (Eyn[:, k_pL], 0.5 * (Hx[:, k_pL] + Hx[:, k_pL - 1]),
               Eyn[:, k_pR], 0.5 * (Hx[:, k_pR] + Hx[:, k_pR - 1]))
        return (Eyn, Hx, Hz, Jy, psi_h, psi_e), out

    z0 = jnp.zeros((nx, nz), dtype=jnp.complex128)
    _, (eyL, hxL, eyR, hxR) = lax.scan(step, tuple(z0 for _ in range(6)),
                                       jnp.asarray(src, dtype=jnp.complex128))
    return eyL, hxL, eyR, hxR


_run_2d_te_oblique_jax = run_2d_te_oblique_jax               # back-compat alias (pre-promotion name)


def run_2d_tm_oblique_jax(eps_inf, wp, gam, dx, dz, dt, nsteps, k_src, k_pL, k_pR, src, cpml, kx):
    """JAX (XLA, lax.scan) twin of _run_2d_tm_oblique (p-pol complex-envelope Bloch): the SAME physics
    (in-plane Ex,Ez + Hy, d/dx -> d/dx + i kx, semi-implicit Drude ADE on Jx,Jz, CFS-CPML + PEC in z) as a
    single traced/compiled DIFFERENTIABLE complex128 time loop -- so jax.grad flows d(R,T)/d(geometry) for
    the TM inverse-design path at angle. Byte-equal to the NumPy TM kernel to ~1e-12. Drude only."""
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from jax import lax
    (ke, be, ce), (kh, bh, ch) = cpml
    ke, be, ce = jnp.asarray(ke), jnp.asarray(be), jnp.asarray(ce)
    kh, bh, ch = jnp.asarray(kh), jnp.asarray(bh), jnp.asarray(ch)
    eps_inf = jnp.asarray(eps_inf); gam = jnp.asarray(gam); wp = jnp.asarray(wp)
    aJ = (1.0 - gam * dt / 2.0) / (1.0 + gam * dt / 2.0)
    bJ = (EPS0 * wp ** 2 * dt / 2.0) / (1.0 + gam * dt / 2.0)
    nx, nz = eps_inf.shape
    cmu = dt / MU0; ikx = 1j * kx; e0dt = EPS0 / dt
    denom = e0dt * eps_inf + bJ / 2.0

    def step(carry, src_n):
        Ex, Ez, Hy, Jx, Jz, psi_hyz, psi_exz = carry
        # H update: mu0 dHy/dt = dEz/dx - dEx/dz (only Hy[:, :-1]; dEz/dx -> d/dx + i kx)
        dEx_dz = (Ex[:, 1:] - Ex[:, :-1]) / dz
        psi_hyz = psi_hyz.at[:, :-1].set(bh[:-1] * psi_hyz[:, :-1] + ch[:-1] * dEx_dz)
        dEz_dx = (jnp.roll(Ez, -1, axis=0) - Ez) / dx + ikx * Ez
        Hy = Hy.at[:, :-1].add(cmu * (dEz_dx[:, :-1] - (dEx_dz / kh[:-1] + psi_hyz[:, :-1])))
        # E update Ex: eps dEx/dt = -dHy/dz - Jx
        dHy_dz = (Hy[:, 1:] - Hy[:, :-1]) / dz
        psi_exz = psi_exz.at[:, 1:].set(be[1:] * psi_exz[:, 1:] + ce[1:] * dHy_dz)
        curlx = jnp.zeros((nx, nz), dtype=jnp.complex128)
        curlx = curlx.at[:, 1:].set(-(dHy_dz / ke[1:] + psi_exz[:, 1:]))
        Exn = (e0dt * eps_inf * Ex + curlx - 0.5 * (1.0 + aJ) * Jx - 0.5 * bJ * Ex) / denom
        Jx = aJ * Jx + bJ * (Exn + Ex)
        Exn = Exn.at[:, k_src].add(src_n)
        Exn = Exn.at[:, 0].set(0.0 + 0.0j).at[:, nz - 1].set(0.0 + 0.0j)
        # E update Ez: eps dEz/dt = +dHy/dx - Jz (dHy/dx -> d/dx + i kx)
        curlz = (Hy - jnp.roll(Hy, 1, axis=0)) / dx + ikx * Hy
        Ezn = (e0dt * eps_inf * Ez + curlz - 0.5 * (1.0 + aJ) * Jz - 0.5 * bJ * Ez) / denom
        Jz = aJ * Jz + bJ * (Ezn + Ez)
        out = (Exn[:, k_pL], 0.5 * (Hy[:, k_pL] + Hy[:, k_pL - 1]),
               Exn[:, k_pR], 0.5 * (Hy[:, k_pR] + Hy[:, k_pR - 1]))
        return (Exn, Ezn, Hy, Jx, Jz, psi_hyz, psi_exz), out

    z0 = jnp.zeros((nx, nz), dtype=jnp.complex128)
    _, (exL, hyL, exR, hyR) = lax.scan(step, tuple(z0 for _ in range(7)),
                                       jnp.asarray(src, dtype=jnp.complex128))
    return exL, hyL, exR, hyR


_run_2d_tm_oblique_jax = run_2d_tm_oblique_jax               # back-compat alias (pre-promotion name)


def _run_oblique(name, eps_inf, wp, gam, dx, dz, dt, nsteps, k_src, k_pL, k_pR, src, cpml, kx, pol="s"):
    """Run ONE complex-envelope oblique 2D pass on the named backend. pol='s' = TE (Ey,Hx,Hz); pol='p' =
    TM (Hy,Ex,Ez). Returns the four complex probe x-lines (tangential E + co-located tangential H).
    'numba' = the fused JIT kernel; 'jax' = the differentiable scan (TE and TM both have one of each);
    anything else runs the vectorized NumPy reference."""
    if pol == "p":
        if name == "numba":
            (ke, be, ce), (kh, bh, ch) = cpml
            return _tm2d_oblique_numba(np.asarray(eps_inf, float), np.asarray(wp, float),
                                       np.asarray(gam, float), ke, be, ce, kh, bh, ch, dx, dz, dt,
                                       nsteps, k_src, k_pL, k_pR, np.asarray(src, float), kx)
        if name == "jax":
            out = run_2d_tm_oblique_jax(eps_inf, wp, gam, dx, dz, dt, nsteps, k_src, k_pL, k_pR, src,
                                        cpml, kx)
            return tuple(np.asarray(v) for v in out)        # JAX -> NumPy for the FFT/R-T stage
        return _run_2d_tm_oblique(eps_inf, wp, gam, dx, dz, dt, nsteps, k_src, k_pL, k_pR, src, cpml, kx)
    if name == "jax":
        out = run_2d_te_oblique_jax(eps_inf, wp, gam, dx, dz, dt, nsteps, k_src, k_pL, k_pR, src, cpml, kx)
        return tuple(np.asarray(v) for v in out)            # JAX -> NumPy for the FFT/R-T stage
    if name == "numba":
        (ke, be, ce), (kh, bh, ch) = cpml
        return _te2d_oblique_numba(np.asarray(eps_inf, float), np.asarray(wp, float), np.asarray(gam, float),
                                   ke, be, ce, kh, bh, ch, dx, dz, dt, nsteps, k_src, k_pL, k_pR,
                                   np.asarray(src, float), kx)
    return _run_2d_te_oblique(eps_inf, wp, gam, dx, dz, dt, nsteps, k_src, k_pL, k_pR, src, cpml, kx)
