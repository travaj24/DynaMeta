"""3D vector kernels (numpy/cupy): normal, oblique-envelope, and magneto-optic.

Split from the former monolithic fdtd_nd.py; see the package __init__ docstring
for conventions. Bodies are verbatim from the original module.
"""
from __future__ import annotations

import numpy as np

from dynameta.constants import EPS0, MU0


def run_3d(eps_inf, wp, gam, chi3, dx, dy, dz, dt, nsteps, k_src, k_pL, k_pR, src, cpml, xp=np, lor=None,
           chi2=None, raman=None, gain=None):
    """One full-vector 3D-FDTD pass over a cell-wise (nx,ny,nz) (eps_inf, wp, gamma, chi3) profile.
    Periodic in x and y (roll = Bloch at normal incidence, zero phase), CFS-CPML + PEC backing in z.
    Standard Yee staggering: Ex@(i+1/2,j,k) Ey@(i,j+1/2,k) Ez@(i,j,k+1/2); Hx@(i,j+1/2,k+1/2)
    Hy@(i+1/2,j,k+1/2) Hz@(i+1/2,j+1/2,k). Semi-implicit Drude ADE per E-component + instantaneous Kerr
    (eps_eff = eps_inf + 3 chi3|E|^2, the standard P = eps0 chi3 E^3 convention, audit C3-2) + an optional Lorentz ADE per E-component (`lor`=(C1,C2,C3), a
    polarization PL{x,y,z}). R15/R20 nonlinearities (None -> byte-identical): chi2 SHG as a DIAGONAL
    tensor model P2_i = eps0 chi2 E_i^2 per component; Raman with ONE isotropic vibrational coordinate
    Q driven by |E|^2 and P_R,i = eps0 chi3R E_i Q (couples components through Q; reduces exactly to
    the 2D scalar model for a single-component field); the clamped-inversion gain line per component
    (the same (G1,G2,G3) recursion as the Lorentz pole). Only the d/dz derivatives are CPML-stretched
    (x,y are periodic), so
    four psi memories: dEy/dz & dEx/dz (H update), dHx/dz & dHy/dz (E update). Records Ex,Ey,Hx,Hy on the
    left/right z-probe planes (the components that carry S_z). Returns 8 arrays of shape (nsteps,nx,ny).

    All per-step temporaries are PREALLOCATED outside the time loop and filled with out= ufuncs
    (audit 6.2 perf): every out= expression reproduces the original arithmetic operand-by-operand
    in the same order, so the result is bit-identical -- only the allocations moved."""
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
    do_chi2 = chi2 is not None
    if do_chi2:
        chi2 = xp.asarray(chi2)
        P2x, P2y, P2z = z3(), z3(), z3()                     # chi2 SHG polarization per component
    do_raman = raman is not None
    if do_raman:
        R1, R2, R3 = xp.asarray(raman[0]), xp.asarray(raman[1]), xp.asarray(raman[2])
        chi3R = xp.asarray(raman[3])
        Q, Qp = z3(), z3()                                   # ONE isotropic vibrational coordinate
        PRx, PRy, PRz = z3(), z3(), z3()
    do_gain = gain is not None
    if do_gain:
        G1, G2, G3 = xp.asarray(gain[0]), xp.asarray(gain[1]), xp.asarray(gain[2])
        PGx, PGy, PGz = z3(), z3(), z3()
        PGpx, PGpy, PGpz = z3(), z3(), z3()
    aJ = (1.0 - gam * dt / 2.0) / (1.0 + gam * dt / 2.0)
    bJ = (EPS0 * wp ** 2 * dt / 2.0) / (1.0 + gam * dt / 2.0)
    cmu = dt / MU0
    # loop-invariant factors hoisted out of the time loop (audit 6.2 perf; identical ops, once)
    chi3_3 = 3.0 * chi3
    bJ_h = bJ / 2.0
    aJ1_h = 0.5 * (1.0 + aJ)
    bJ_half = 0.5 * bJ
    # preallocated per-step scratch (audit 6.2 perf). s*: the CPML edge slice is written once
    # (zeros) and never touched again -- exactly the fresh-z3()-per-step value; the interior is
    # fully rewritten each step. f1/f2/f3 = general full-grid scratch; dzb/tz = (nz-1) z-derivative
    # scratch; Exn/Eyn/Ezn ping-pong with Ex/Ey/Ez (fully overwritten every step).
    sEy_dz, sEx_dz, sHy_dz, sHx_dz = z3(), z3(), z3(), z3()
    f1, f2, f3 = xp.empty((nx, ny, nz)), xp.empty((nx, ny, nz)), xp.empty((nx, ny, nz))
    ce_dt, denom = xp.empty((nx, ny, nz)), xp.empty((nx, ny, nz))
    dzb, tz = xp.empty((nx, ny, nz - 1)), xp.empty((nx, ny, nz - 1))
    Exn, Eyn, Ezn = xp.empty((nx, ny, nz)), xp.empty((nx, ny, nz)), xp.empty((nx, ny, nz))
    kes, bes, ces = ke[:, :, 1:], be[:, :, 1:], ce[:, :, 1:]     # CPML interior views (E-grid)
    khs, bhs, chs = kh[:, :, :-1], bh[:, :, :-1], ch[:, :, :-1]  # CPML interior views (H-grid)

    def _dfwd(F, axis, d, out):
        """(xp.roll(F, -1, axis) - F) / d without the roll/temporary allocations (same values:
        the periodic wrap row is the single [0]-[-1] difference)."""
        if axis == 0:
            xp.subtract(F[1:, :, :], F[:-1, :, :], out=out[:-1, :, :])
            xp.subtract(F[0, :, :], F[-1, :, :], out=out[-1, :, :])
        else:
            xp.subtract(F[:, 1:, :], F[:, :-1, :], out=out[:, :-1, :])
            xp.subtract(F[:, 0, :], F[:, -1, :], out=out[:, -1, :])
        xp.divide(out, d, out=out)
        return out

    def _dbwd(F, axis, d, out):
        """(F - xp.roll(F, 1, axis)) / d without the roll/temporary allocations."""
        if axis == 0:
            xp.subtract(F[1:, :, :], F[:-1, :, :], out=out[1:, :, :])
            xp.subtract(F[0, :, :], F[-1, :, :], out=out[0, :, :])
        else:
            xp.subtract(F[:, 1:, :], F[:, :-1, :], out=out[:, 1:, :])
            xp.subtract(F[:, 0, :], F[:, -1, :], out=out[:, 0, :])
        xp.divide(out, d, out=out)
        return out

    sh = (nsteps, nx, ny)
    exL, eyL, hxL, hyL = xp.empty(sh), xp.empty(sh), xp.empty(sh), xp.empty(sh)
    exR, eyR, hxR, hyR = xp.empty(sh), xp.empty(sh), xp.empty(sh), xp.empty(sh)
    for n in range(nsteps):
        # ---------------- H update: dH/dt = -(1/mu) curl E ----------------
        # Hx: -(dEz/dy - dEy/dz) ; dEy/dz is CPML-stretched
        xp.subtract(Ey[:, :, 1:], Ey[:, :, :-1], out=dzb); xp.divide(dzb, dz, out=dzb)  # dEy/dz
        xp.multiply(chs, dzb, out=tz)
        xp.multiply(bhs, psi_Hx[:, :, :-1], out=psi_Hx[:, :, :-1])
        xp.add(psi_Hx[:, :, :-1], tz, out=psi_Hx[:, :, :-1])
        xp.divide(dzb, khs, out=tz)
        xp.add(tz, psi_Hx[:, :, :-1], out=sEy_dz[:, :, :-1])
        _dfwd(Ez, 1, dy, f1)                                 # dEz/dy
        xp.subtract(f1, sEy_dz, out=f1); xp.multiply(f1, cmu, out=f1)
        xp.subtract(Hx, f1, out=Hx)
        # Hy: -(dEx/dz - dEz/dx) ; dEx/dz is CPML-stretched
        xp.subtract(Ex[:, :, 1:], Ex[:, :, :-1], out=dzb); xp.divide(dzb, dz, out=dzb)  # dEx/dz
        xp.multiply(chs, dzb, out=tz)
        xp.multiply(bhs, psi_Hy[:, :, :-1], out=psi_Hy[:, :, :-1])
        xp.add(psi_Hy[:, :, :-1], tz, out=psi_Hy[:, :, :-1])
        xp.divide(dzb, khs, out=tz)
        xp.add(tz, psi_Hy[:, :, :-1], out=sEx_dz[:, :, :-1])
        _dfwd(Ez, 0, dx, f1)                                 # dEz/dx
        xp.subtract(sEx_dz, f1, out=f1); xp.multiply(f1, cmu, out=f1)
        xp.subtract(Hy, f1, out=Hy)
        # Hz: -(dEy/dx - dEx/dy) ; both transverse (no CPML)
        _dfwd(Ey, 0, dx, f1)
        _dfwd(Ex, 1, dy, f2)
        xp.subtract(f1, f2, out=f1); xp.multiply(f1, cmu, out=f1)
        xp.subtract(Hz, f1, out=Hz)
        # ---------------- E update: eps0 eps_eff dE/dt = curl H - J ----------------
        # the Raman coordinate is shared by all components: advance it ONCE per step on |E|^2
        if do_raman:
            Qnew = R1 * Q + R2 * Qp + R3 * (Ex ** 2 + Ey ** 2 + Ez ** 2)
            Qp = Q; Q = Qnew
        # eps_eff = eps_inf + 3 chi3 |E|^2 (standard chi3, C3-2) -> ce_dt, denom
        xp.multiply(Ex, Ex, out=f1)
        xp.multiply(Ey, Ey, out=f2)
        xp.add(f1, f2, out=f1)
        xp.multiply(Ez, Ez, out=f2)
        xp.add(f1, f2, out=f1)                               # |E|^2
        xp.multiply(chi3_3, f1, out=f1)
        xp.add(eps_inf, f1, out=f1)                          # eps_eff
        xp.multiply(f1, EPS0, out=ce_dt); xp.divide(ce_dt, dt, out=ce_dt)
        xp.add(ce_dt, bJ_h, out=denom)
        # Ex: (dHz/dy - dHy/dz) ; dHy/dz CPML-stretched
        xp.subtract(Hy[:, :, 1:], Hy[:, :, :-1], out=dzb); xp.divide(dzb, dz, out=dzb)  # dHy/dz
        xp.multiply(ces, dzb, out=tz)
        xp.multiply(bes, psi_Ex[:, :, 1:], out=psi_Ex[:, :, 1:])
        xp.add(psi_Ex[:, :, 1:], tz, out=psi_Ex[:, :, 1:])
        xp.divide(dzb, kes, out=tz)
        xp.add(tz, psi_Ex[:, :, 1:], out=sHy_dz[:, :, 1:])
        _dbwd(Hz, 1, dy, f2)                                 # dHz/dy
        xp.subtract(f2, sHy_dz, out=f2)
        curlx = f2
        if do_lor:                                          # Lorentz dPLx/dt enters the Ex-update
            PLxn = C1 * PLx + C2 * PLpx + C3 * Ex
            curlx = curlx - (PLxn - PLx) / dt
            PLpx, PLx = PLx, PLxn
        if do_gain:
            PGxn = G1 * PGx + G2 * PGpx + G3 * Ex
            curlx = curlx - (PGxn - PGx) / dt
            PGpx, PGx = PGx, PGxn
        if do_chi2:
            P2xn = EPS0 * chi2 * Ex ** 2
            curlx = curlx - (P2xn - P2x) / dt
            P2x = P2xn
        if do_raman:
            PRxn = EPS0 * chi3R * Ex * Q
            curlx = curlx - (PRxn - PRx) / dt
            PRx = PRxn
        # Exn = (ce_dt*Ex + curlx - 0.5*(1+aJ)*Jx - 0.5*bJ*Ex) / denom, then the Drude J recursion
        xp.multiply(ce_dt, Ex, out=f1)
        xp.add(f1, curlx, out=f1)
        xp.multiply(aJ1_h, Jx, out=f3)
        xp.subtract(f1, f3, out=f1)
        xp.multiply(bJ_half, Ex, out=f3)
        xp.subtract(f1, f3, out=f1)
        xp.divide(f1, denom, out=Exn)
        xp.add(Exn, Ex, out=f3); xp.multiply(f3, bJ, out=f3)     # bJ * (Exn + Ex)
        xp.multiply(aJ, Jx, out=Jx); xp.add(Jx, f3, out=Jx)      # Jx = aJ*Jx + ...
        # Ey: (dHx/dz - dHz/dx) ; dHx/dz CPML-stretched
        xp.subtract(Hx[:, :, 1:], Hx[:, :, :-1], out=dzb); xp.divide(dzb, dz, out=dzb)  # dHx/dz
        xp.multiply(ces, dzb, out=tz)
        xp.multiply(bes, psi_Ey[:, :, 1:], out=psi_Ey[:, :, 1:])
        xp.add(psi_Ey[:, :, 1:], tz, out=psi_Ey[:, :, 1:])
        xp.divide(dzb, kes, out=tz)
        xp.add(tz, psi_Ey[:, :, 1:], out=sHx_dz[:, :, 1:])
        _dbwd(Hz, 0, dx, f2)                                 # dHz/dx
        xp.subtract(sHx_dz, f2, out=f2)
        curly = f2
        if do_lor:
            PLyn = C1 * PLy + C2 * PLpy + C3 * Ey
            curly = curly - (PLyn - PLy) / dt
            PLpy, PLy = PLy, PLyn
        if do_gain:
            PGyn = G1 * PGy + G2 * PGpy + G3 * Ey
            curly = curly - (PGyn - PGy) / dt
            PGpy, PGy = PGy, PGyn
        if do_chi2:
            P2yn = EPS0 * chi2 * Ey ** 2
            curly = curly - (P2yn - P2y) / dt
            P2y = P2yn
        if do_raman:
            PRyn = EPS0 * chi3R * Ey * Q
            curly = curly - (PRyn - PRy) / dt
            PRy = PRyn
        xp.multiply(ce_dt, Ey, out=f1)
        xp.add(f1, curly, out=f1)
        xp.multiply(aJ1_h, Jy, out=f3)
        xp.subtract(f1, f3, out=f1)
        xp.multiply(bJ_half, Ey, out=f3)
        xp.subtract(f1, f3, out=f1)
        xp.divide(f1, denom, out=Eyn)
        xp.add(Eyn, Ey, out=f3); xp.multiply(f3, bJ, out=f3)
        xp.multiply(aJ, Jy, out=Jy); xp.add(Jy, f3, out=Jy)
        # Ez: (dHy/dx - dHx/dy) ; both transverse (no CPML)
        _dbwd(Hy, 0, dx, f2)                                 # dHy/dx
        _dbwd(Hx, 1, dy, f1)                                 # dHx/dy
        xp.subtract(f2, f1, out=f2)
        curlz = f2
        if do_lor:
            PLzn = C1 * PLz + C2 * PLpz + C3 * Ez
            curlz = curlz - (PLzn - PLz) / dt
            PLpz, PLz = PLz, PLzn
        if do_gain:
            PGzn = G1 * PGz + G2 * PGpz + G3 * Ez
            curlz = curlz - (PGzn - PGz) / dt
            PGpz, PGz = PGz, PGzn
        if do_chi2:
            P2zn = EPS0 * chi2 * Ez ** 2
            curlz = curlz - (P2zn - P2z) / dt
            P2z = P2zn
        if do_raman:
            PRzn = EPS0 * chi3R * Ez * Q
            curlz = curlz - (PRzn - PRz) / dt
            PRz = PRzn
        xp.multiply(ce_dt, Ez, out=f1)
        xp.add(f1, curlz, out=f1)
        xp.multiply(aJ1_h, Jz, out=f3)
        xp.subtract(f1, f3, out=f1)
        xp.multiply(bJ_half, Ez, out=f3)
        xp.subtract(f1, f3, out=f1)
        xp.divide(f1, denom, out=Ezn)
        xp.add(Ezn, Ez, out=f3); xp.multiply(f3, bJ, out=f3)
        xp.multiply(aJ, Jz, out=Jz); xp.add(Jz, f3, out=Jz)
        # soft y-polarized plane source (uniform in x,y -> normal incidence), PEC backing the CPML:
        # a z=const PEC plane forces only the TANGENTIAL E (Ex,Ey) to zero; the normal Ez sits half a
        # cell inside and is left to its (purely transverse-curl, in-bounds) update.
        Eyn[:, :, k_src] += src[n]
        for F in (Exn, Eyn):
            F[:, :, 0] = 0.0; F[:, :, -1] = 0.0
        # ping-pong swap: E^{n+1} becomes current; the retired E^n array is next step's out= target
        Ex, Exn = Exn, Ex
        Ey, Eyn = Eyn, Ey
        Ez, Ezn = Ezn, Ez
        # probe planes: co-locate Hx,Hy (at k+/-1/2) onto the E-plane (k) so S_z co-locates in z
        exL[n] = Ex[:, :, k_pL]; eyL[n] = Ey[:, :, k_pL]
        hxL[n] = 0.5 * (Hx[:, :, k_pL] + Hx[:, :, k_pL - 1]); hyL[n] = 0.5 * (Hy[:, :, k_pL] + Hy[:, :, k_pL - 1])
        exR[n] = Ex[:, :, k_pR]; eyR[n] = Ey[:, :, k_pR]
        hxR[n] = 0.5 * (Hx[:, :, k_pR] + Hx[:, :, k_pR - 1]); hyR[n] = 0.5 * (Hy[:, :, k_pR] + Hy[:, :, k_pR - 1])
    return exL, eyL, hxL, hyL, exR, eyR, hxR, hyR


_run_3d = run_3d                                             # back-compat alias (pre-promotion name)


def _run_3d_oblique(eps_inf, wp, gam, dx, dy, dz, dt, nsteps, k_src, k_pL, k_pR, src, cpml, kx, ky, sx, sy):
    """Full-vector 3D-FDTD at OBLIQUE incidence via the complex-envelope Bloch method with a 2D transverse
    wavevector (kx,ky): the physical field = envelope * exp(i(kx x + ky y)), so EVERY x-derivative gains
    +i kx and every y-derivative +i ky (the x,y rolls stay zero-phase); only d/dz is CFS-CPML-stretched.
    Fields complex; (kx,ky)=0 reduces to the real normal-incidence engine. Semi-implicit Drude ADE per
    E-component; the incident plane wave is injected along the (sx,sy) in-plane E-direction at k_src
    (s-pol = (-sin phi, cos phi)); PEC on the tangential (Ex,Ey). Records the complex tangential Ex,Ey
    probe planes (the s-pol R/T come from their projection onto (sx,sy)). Drude only (no Lorentz)."""
    nx, ny, nz = eps_inf.shape
    (ke, be, ce), (kh, bh, ch) = cpml
    r = (lambda a: np.asarray(a).reshape(1, 1, nz))
    ke, be, ce = r(ke), r(be), r(ce)
    kh, bh, ch = r(kh), r(bh), r(ch)
    z3 = (lambda: np.zeros((nx, ny, nz), dtype=complex))
    Ex, Ey, Ez = z3(), z3(), z3()
    Hx, Hy, Hz = z3(), z3(), z3()
    Jx, Jy, Jz = z3(), z3(), z3()
    psi_Hx, psi_Hy, psi_Ex, psi_Ey = z3(), z3(), z3(), z3()
    # the four stretched-derivative buffers, hoisted out of the time loop (audit 6.2 perf): the
    # CPML edge slice is written once (zeros) and never touched again, the interior is fully
    # rewritten each step -- identical values, no per-step full-grid allocation.
    sEy_dz, sEx_dz, sHy_dz, sHx_dz = z3(), z3(), z3(), z3()
    aJ = (1.0 - gam * dt / 2.0) / (1.0 + gam * dt / 2.0)
    bJ = (EPS0 * wp ** 2 * dt / 2.0) / (1.0 + gam * dt / 2.0)
    cmu = dt / MU0; e0dt = EPS0 / dt
    ikx, iky = 1j * kx, 1j * ky
    dxf = (lambda F: (np.roll(F, -1, axis=0) - F) / dx + ikx * F)   # forward x-deriv + envelope
    dyf = (lambda F: (np.roll(F, -1, axis=1) - F) / dy + iky * F)   # forward y-deriv + envelope
    dxb = (lambda F: (F - np.roll(F, 1, axis=0)) / dx + ikx * F)    # backward x-deriv + envelope
    dyb = (lambda F: (F - np.roll(F, 1, axis=1)) / dy + iky * F)    # backward y-deriv + envelope
    sh = (nsteps, nx, ny)
    exL, eyL, exR, eyR = np.empty(sh, complex), np.empty(sh, complex), np.empty(sh, complex), np.empty(sh, complex)
    for n in range(nsteps):
        # ---- H update: dH/dt = -(1/mu) curl E ; only d/dz CPML-stretched ----
        dEy_dz = (Ey[:, :, 1:] - Ey[:, :, :-1]) / dz
        psi_Hx[:, :, :-1] = bh[:, :, :-1] * psi_Hx[:, :, :-1] + ch[:, :, :-1] * dEy_dz
        sEy_dz[:, :, :-1] = dEy_dz / kh[:, :, :-1] + psi_Hx[:, :, :-1]
        Hx -= cmu * (dyf(Ez) - sEy_dz)
        dEx_dz = (Ex[:, :, 1:] - Ex[:, :, :-1]) / dz
        psi_Hy[:, :, :-1] = bh[:, :, :-1] * psi_Hy[:, :, :-1] + ch[:, :, :-1] * dEx_dz
        sEx_dz[:, :, :-1] = dEx_dz / kh[:, :, :-1] + psi_Hy[:, :, :-1]
        Hy -= cmu * (sEx_dz - dxf(Ez))
        Hz -= cmu * (dxf(Ey) - dyf(Ex))
        # ---- E update: eps0 eps dE/dt = curl H - J ; only d/dz CPML-stretched ----
        denom = e0dt * eps_inf + bJ / 2.0
        dHy_dz = (Hy[:, :, 1:] - Hy[:, :, :-1]) / dz
        psi_Ex[:, :, 1:] = be[:, :, 1:] * psi_Ex[:, :, 1:] + ce[:, :, 1:] * dHy_dz
        sHy_dz[:, :, 1:] = dHy_dz / ke[:, :, 1:] + psi_Ex[:, :, 1:]
        curlx = dyb(Hz) - sHy_dz
        Exn = (e0dt * eps_inf * Ex + curlx - 0.5 * (1.0 + aJ) * Jx - 0.5 * bJ * Ex) / denom
        Jx = aJ * Jx + bJ * (Exn + Ex)
        dHx_dz = (Hx[:, :, 1:] - Hx[:, :, :-1]) / dz
        psi_Ey[:, :, 1:] = be[:, :, 1:] * psi_Ey[:, :, 1:] + ce[:, :, 1:] * dHx_dz
        sHx_dz[:, :, 1:] = dHx_dz / ke[:, :, 1:] + psi_Ey[:, :, 1:]
        curly = sHx_dz - dxb(Hz)
        Eyn = (e0dt * eps_inf * Ey + curly - 0.5 * (1.0 + aJ) * Jy - 0.5 * bJ * Ey) / denom
        Jy = aJ * Jy + bJ * (Eyn + Ey)
        curlz = dxb(Hy) - dyb(Hx)
        Ezn = (e0dt * eps_inf * Ez + curlz - 0.5 * (1.0 + aJ) * Jz - 0.5 * bJ * Ez) / denom
        Jz = aJ * Jz + bJ * (Ezn + Ez)
        Exn[:, :, k_src] += sx * src[n]; Eyn[:, :, k_src] += sy * src[n]    # s-pol plane source
        for F in (Exn, Eyn):
            F[:, :, 0] = 0.0; F[:, :, -1] = 0.0                             # PEC backing
        Ex, Ey, Ez = Exn, Eyn, Ezn
        exL[n] = Ex[:, :, k_pL]; eyL[n] = Ey[:, :, k_pL]
        exR[n] = Ex[:, :, k_pR]; eyR[n] = Ey[:, :, k_pR]
    return exL, eyL, exR, eyR




def _run_3d_mo(exx, eyy, ezz, wp, gam, wc, dx, dy, dz, dt, nsteps, k_src, k_pL, k_pR, src, cpml, pol):
    """Full-vector 3D-FDTD with a per-cell DIAGONAL anisotropy (exx,eyy,ezz) AND a gyrotropic
    MAGNETO-OPTIC response (magnetization along z) via the magnetized-Drude ADE -- the 3D analog of the
    1-D fdtd_mo. The polarization current obeys dJ/dt + gamma J + wc (zhat x J) = eps0 wp^2 E: the
    cyclotron term couples Jx<->Jy (a per-cell 2x2 Crank-Nicolson, semi-implicit with the Ex,Ey update),
    while Jz is a plain scalar Drude. This is the PHYSICALLY-CORRECT time-domain origin of the
    frequency-domain off-diagonal i*g. Normal incidence (Bloch zero-phase rolls), CFS-CPML + PEC in z,
    real fields, soft plane source on `pol` ('x'/'y'). Records the Ex,Ey probe planes (the co/cross-pol
    transmission + Faraday come from them). Reduces to the 1-D fdtd_mo for a laterally-uniform slab."""
    nx, ny, nz = exx.shape
    (ke, be, ce), (kh, bh, ch) = cpml
    r = (lambda a: np.asarray(a).reshape(1, 1, nz))
    ke, be, ce = r(ke), r(be), r(ce)
    kh, bh, ch = r(kh), r(bh), r(ch)
    # per-cell 2x2 magnetized-Drude Crank-Nicolson + E-update matrices, split into scalar components
    # (same algebra as fdtd_mo._run_mo, here per (nx,ny,nz) cell). A J^{n+1}=B J^n + eps0 wp^2(E^{n+1}+E^n)/2
    # with G = gamma I + [[0,-wc],[wc,0]]; A=I/dt+G/2, B=I/dt-G/2.
    a_ = 1.0 / dt + 0.5 * gam; off = 0.5 * wc                    # A=[[a_,-off],[off,a_]]
    detA = a_ ** 2 + off ** 2
    Ai00, Ai01, Ai10, Ai11 = a_ / detA, off / detA, -off / detA, a_ / detA
    b_ = 1.0 / dt - 0.5 * gam                                    # B=[[b_,off],[-off,b_]]
    Ma00 = Ai00 * b_ + Ai01 * (-off); Ma01 = Ai00 * off + Ai01 * b_
    Ma10 = Ai10 * b_ + Ai11 * (-off); Ma11 = Ai10 * off + Ai11 * b_
    s = EPS0 * wp ** 2 * 0.5
    Mb00, Mb01, Mb10, Mb11 = Ai00 * s, Ai01 * s, Ai10 * s, Ai11 * s
    D00, D11 = EPS0 * exx / dt, EPS0 * eyy / dt                  # D = diag(eps0 exx/dt, eps0 eyy/dt)
    M00, M01, M10, M11 = D00 + 0.5 * Mb00, 0.5 * Mb01, 0.5 * Mb10, D11 + 0.5 * Mb11
    detM = M00 * M11 - M01 * M10
    Iv00, Iv01, Iv10, Iv11 = M11 / detM, -M01 / detM, -M10 / detM, M00 / detM   # Einv
    Ep00, Ep01, Ep10, Ep11 = D00 - 0.5 * Mb00, -0.5 * Mb01, -0.5 * Mb10, D11 - 0.5 * Mb11
    Jc00, Jc01, Jc10, Jc11 = 0.5 * (Ma00 + 1.0), 0.5 * Ma01, 0.5 * Ma10, 0.5 * (Ma11 + 1.0)
    aJz = (1.0 - gam * dt / 2.0) / (1.0 + gam * dt / 2.0)        # Ez: plain scalar Drude (no gyro)
    bJz = (EPS0 * wp ** 2 * dt / 2.0) / (1.0 + gam * dt / 2.0)
    z3 = (lambda: np.zeros((nx, ny, nz)))
    Ex, Ey, Ez = z3(), z3(), z3()
    Hx, Hy, Hz = z3(), z3(), z3()
    Jx, Jy, Jz = z3(), z3(), z3()
    psi_Hx, psi_Hy, psi_Ex, psi_Ey = z3(), z3(), z3(), z3()
    # stretched-derivative buffers hoisted out of the time loop (audit 6.2 perf): the CPML edge
    # slice keeps its once-written zeros, the interior is fully rewritten each step
    sEy, sEx, sHy, sHx = z3(), z3(), z3(), z3()
    cmu = dt / MU0
    sh = (nsteps, nx, ny)
    exL, eyL, exR, eyR = np.empty(sh), np.empty(sh), np.empty(sh), np.empty(sh)
    for n in range(nsteps):
        # ---- H update (dH/dt = -(1/mu) curl E); only d/dz CPML-stretched ----
        dEy_dz = (Ey[:, :, 1:] - Ey[:, :, :-1]) / dz
        psi_Hx[:, :, :-1] = bh[:, :, :-1] * psi_Hx[:, :, :-1] + ch[:, :, :-1] * dEy_dz
        sEy[:, :, :-1] = dEy_dz / kh[:, :, :-1] + psi_Hx[:, :, :-1]
        Hx -= cmu * ((np.roll(Ez, -1, axis=1) - Ez) / dy - sEy)
        dEx_dz = (Ex[:, :, 1:] - Ex[:, :, :-1]) / dz
        psi_Hy[:, :, :-1] = bh[:, :, :-1] * psi_Hy[:, :, :-1] + ch[:, :, :-1] * dEx_dz
        sEx[:, :, :-1] = dEx_dz / kh[:, :, :-1] + psi_Hy[:, :, :-1]
        Hy -= cmu * (sEx - (np.roll(Ez, -1, axis=0) - Ez) / dx)
        Hz -= cmu * ((np.roll(Ey, -1, axis=0) - Ey) / dx - (np.roll(Ex, -1, axis=1) - Ex) / dy)
        # ---- E update: curls, then the 2x2 magnetized-Drude CN on (Ex,Ey) + scalar Ez ----
        dHy_dz = (Hy[:, :, 1:] - Hy[:, :, :-1]) / dz
        psi_Ex[:, :, 1:] = be[:, :, 1:] * psi_Ex[:, :, 1:] + ce[:, :, 1:] * dHy_dz
        sHy[:, :, 1:] = dHy_dz / ke[:, :, 1:] + psi_Ex[:, :, 1:]
        curlx = (Hz - np.roll(Hz, 1, axis=1)) / dy - sHy
        dHx_dz = (Hx[:, :, 1:] - Hx[:, :, :-1]) / dz
        psi_Ey[:, :, 1:] = be[:, :, 1:] * psi_Ey[:, :, 1:] + ce[:, :, 1:] * dHx_dz
        sHx[:, :, 1:] = dHx_dz / ke[:, :, 1:] + psi_Ey[:, :, 1:]
        curly = sHx - (Hz - np.roll(Hz, 1, axis=0)) / dx
        curlz = (Hy - np.roll(Hy, 1, axis=0)) / dx - (Hx - np.roll(Hx, 1, axis=1)) / dy
        rhs0 = Ep00 * Ex + Ep01 * Ey + curlx - (Jc00 * Jx + Jc01 * Jy)
        rhs1 = Ep10 * Ex + Ep11 * Ey + curly - (Jc10 * Jx + Jc11 * Jy)
        Exn = Iv00 * rhs0 + Iv01 * rhs1
        Eyn = Iv10 * rhs0 + Iv11 * rhs1
        sx, sy = Exn + Ex, Eyn + Ey                              # J^{n+1} uses OLD J (write to fresh names)
        Jxn = Ma00 * Jx + Ma01 * Jy + Mb00 * sx + Mb01 * sy
        Jyn = Ma10 * Jx + Ma11 * Jy + Mb10 * sx + Mb11 * sy
        Jx, Jy = Jxn, Jyn
        denomz = EPS0 * ezz / dt + bJz / 2.0
        Ezn = (EPS0 * ezz / dt * Ez + curlz - 0.5 * (1.0 + aJz) * Jz - 0.5 * bJz * Ez) / denomz
        Jz = aJz * Jz + bJz * (Ezn + Ez)
        if pol == "y":
            Eyn[:, :, k_src] += src[n]
        else:
            Exn[:, :, k_src] += src[n]
        for F in (Exn, Eyn):
            F[:, :, 0] = 0.0; F[:, :, -1] = 0.0
        Ex, Ey, Ez = Exn, Eyn, Ezn
        exL[n] = Ex[:, :, k_pL]; eyL[n] = Ey[:, :, k_pL]
        exR[n] = Ex[:, :, k_pR]; eyR[n] = Ey[:, :, k_pR]
    return exL, eyL, exR, eyR


