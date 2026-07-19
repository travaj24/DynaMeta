"""2D-TE normal-incidence reference kernel (numpy; cupy via the xp parameter).

Split from the former monolithic fdtd_nd.py; see the package __init__ docstring
for conventions. Bodies are verbatim from the original module.
"""
from __future__ import annotations

import numpy as np

from dynameta.constants import EPS0, MU0



def run_2d_te(eps_inf, wp, gam, chi3, dx, dz, dt, nsteps, k_src, k_pL, k_pR, src, cpml, xp=np, lor=None,
              chi2=None, raman=None, gain=None, gain_dyn=None, gain_dyn_out=None, hot=None, hot_out=None):
    """One 2D TE pass over a cell-wise (nx,nz) (eps_inf, wp, gamma, chi3) profile. Periodic in x (roll),
    CFS-CPML absorbing layers + PEC backing in z. Records the E_y and H_x x-lines at the left/right
    z-probe planes (for both the x-mean 0-order and the Poynting-flux R/T). Semi-implicit Drude ADE +
    instantaneous Kerr. `cpml` = ((kappa_e,b_e,c_e),(kappa_h,b_h,c_h)) from cpml_z (z-broadcast).
    `lor` = (C1,C2,C3) per-cell Lorentz ADE coefficients (a second polarization PL) or None (no pole).

    R15 nonlinear polarizations (None -> byte-identical to the pre-R15 path):
      chi2:  (nx,nz) SHG coefficient [m/V] -- P2 = eps0 chi2 E^2 tracked as a polarization whose
             dP2/dt enters the E-update like the Lorentz dPL/dt (lagged explicit; the E^2 product
             radiates the second harmonic naturally in the time domain). STABILITY: the explicit
             coupling needs the perturbative regime chi2*|E| << 1 (it destabilizes around
             chi2*|E| ~ 0.3, an unphysical 20%-index nonlinearity; real SHG drives are ~1e-4).
      raman: (R1,R2,R3,chi3R) -- the vibrational coordinate ADE Q^{n+1} = R1 Q + R2 Q^{n-1} + R3 E^2
             (Q'' + gam_R Q' + W_R^2 Q = W_R^2 E^2 central-differenced) and the THIRD-order Raman
             polarization P_R = eps0 chi3R E Q (E times the DELAYED E^2 response -- this, not dQ/dt,
             is what produces Stokes gain at w_pump - W_R; coupling dQ/dt directly would radiate at
             0/2w instead).
      gain:  (G1,G2,G3) clamped-inversion gain-line ADE (R20) -- the SAME recursion as the Lorentz
             pole but sourced by -kappa dN E (G3 = -kappa dN dt^2/den), so dN > 0 amplifies and
             dN < 0 is numerically IDENTICAL to a passive pole with delta_eps = kappa|dN|/(eps0 w^2).
      gain_dyn: DYNAMIC four-level gain (R20 follow-on; mutually exclusive with `gain`):
             (G1, G2, kapfac, Wp, Npop0, tau32, tau21, tau10, hw_a, snap_step) where kapfac =
             kappa dt^2/den per cell (so G3(t) = -kapfac (N2 - N1)), Npop0 the (4,nx,nz) initial
             populations, Wp the pump-rate grid [1/s] and hw_a = hbar w_a [J]. Each step couples
             the field to the populations through the STIMULATED rate density S_st =
             -E dPG/dt / (hbar w_a) (the field-polarization work; positive when amplifying ->
             N2 -> N1 transfer, negative in absorption -> N1 -> N2), then advances the four-level
             rate equations by conservative forward Euler (every term appears +/- once, so
             sum(N) drifts only at the per-step rounding floor). gain_dyn_out (a dict) receives
             'dN_snap' = N2 - N1 captured at snap_step and 'Npop_final'.
      hot:   opt-in per-cell HOT-CARRIER two-temperature ADE (roadmap 2.1; NUMPY reference path only).
             A dict {mask,(nx,nz) float 0/1; mat_idx,(nx,nz) int material id (-1 off); tables, a list of
             per-material (Te_grid,U_grid,wp_ratio,gam_ratio) lookup arrays; G,(nx,nz) e-ph coupling
             [W/m^3/K]; Tl,(nx,nz) fixed lattice bath [K]; alpha,(nx,nz) p_abs coupling; Te0,(nx,nz)
             initial electron temperature; n_update, int steps between (wp,gamma) refreshes}. Each step
             integrates dU_e/dt = alpha*p_abs - G(T_e-T_l) with p_abs = J_drude.E (the Drude Joule
             dissipation, which time-averages to the Drude absorption), recovers T_e from U_e by
             interpolating the per-material C_e-integral table, and (every n_update steps) drops the cell's
             (wp,gamma) via the m*(T_e)/gamma(T_e) ratio tables + rebuilds the Drude ADE coefficients.
             None -> byte-identical to the passive path. gain_dyn is incompatible with hot (both re-touch
             the Drude coefficients); the solve front-end never combines them. hot_out (a dict), when
             given, receives the per-step spatially-averaged p_abs/T_e histories + the final T_e / field-
             intensity maps for the uniformity + locality oracles."""
    nx, nz = eps_inf.shape
    (ke, be, ce), (kh, bh, ch) = cpml
    ke = xp.asarray(ke); be = xp.asarray(be); ce = xp.asarray(ce)
    kh = xp.asarray(kh); bh = xp.asarray(bh); ch = xp.asarray(ch)
    do_lor = lor is not None
    if do_lor:
        C1, C2, C3 = (xp.asarray(lor[0]), xp.asarray(lor[1]), xp.asarray(lor[2]))
        PL = xp.zeros((nx, nz)); PLp = xp.zeros((nx, nz))       # Lorentz polarization (now / previous step)
    do_chi2 = chi2 is not None
    if do_chi2:
        chi2 = xp.asarray(chi2)
        P2 = xp.zeros((nx, nz))                                  # SHG polarization eps0 chi2 E^2
    do_raman = raman is not None
    if do_raman:
        R1, R2, R3 = (xp.asarray(raman[0]), xp.asarray(raman[1]), xp.asarray(raman[2]))
        chi3R = xp.asarray(raman[3])
        Q = xp.zeros((nx, nz)); Qp = xp.zeros((nx, nz))          # vibrational coordinate (now/prev)
        PR = xp.zeros((nx, nz))                                  # Raman polarization eps0 chi3R E Q
    do_gain = gain is not None
    if do_gain:
        G1, G2, G3 = (xp.asarray(gain[0]), xp.asarray(gain[1]), xp.asarray(gain[2]))
        PG = xp.zeros((nx, nz)); PGp = xp.zeros((nx, nz))        # gain-line polarization (now/prev)
    do_gdyn = gain_dyn is not None
    if do_gdyn:
        if do_gain:
            raise ValueError("gain and gain_dyn are mutually exclusive")
        (G1, G2, kapfac, Wp, Npop0, tau32, tau21, tau10, hw_a, snap_step) = gain_dyn
        G1 = xp.asarray(G1); G2 = xp.asarray(G2); kapfac = xp.asarray(kapfac)
        Wp = xp.asarray(Wp)
        N0 = xp.asarray(Npop0[0]).copy(); N1 = xp.asarray(Npop0[1]).copy()
        N2 = xp.asarray(Npop0[2]).copy(); N3 = xp.asarray(Npop0[3]).copy()
        PG = xp.zeros((nx, nz)); PGp = xp.zeros((nx, nz))
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
    # --- HOT-CARRIER two-temperature state (roadmap 2.1). NUMPY reference path only: the maps use np.interp
    #     (per-material table inversion), so a non-NumPy xp is refused (the fast/GPU kernels raise upstream
    #     in _dispatch_2d_te). do_hot False -> not one extra op executes -> byte-identical to before. ---
    do_hot = hot is not None
    if do_hot:
        if xp is not np:
            raise ValueError("hot-carrier ADE runs on the NumPy reference kernel only (np.interp table "
                             "inversion); pass backend='numpy' with the default xp.")
        if do_gdyn:
            raise ValueError("hot-carrier and dynamic gain both re-touch the Drude coefficients; "
                             "they are mutually exclusive")
        hmask = np.asarray(hot["mask"], dtype=np.float64)    # (nx,nz) 1 on opted-in cells, 0 elsewhere
        mat_idx = np.asarray(hot["mat_idx"])                 # (nx,nz) material id, -1 off
        htab = hot["tables"]                                 # list of (Te_grid,U_grid,wp_ratio,gam_ratio)
        Ghot = np.asarray(hot["G"], dtype=np.float64)        # (nx,nz) electron-phonon coupling [W/m^3/K]
        Tlhot = np.asarray(hot["Tl"], dtype=np.float64)      # (nx,nz) fixed lattice bath [K]
        alpha_hot = np.asarray(hot["alpha"], dtype=np.float64)  # (nx,nz) p_abs coupling (usually 1)
        n_upd = max(1, int(hot["n_update"]))
        wp0 = np.array(wp, dtype=np.float64)                 # COLD (wp,gamma) anchors; wp/gam mutate below
        gam0 = np.array(gam, dtype=np.float64)
        wp = wp0.copy(); gam = gam0.copy()
        aJ = (1.0 - gam * dt / 2.0) / (1.0 + gam * dt / 2.0)  # rebuilt from the mutable (wp,gam)
        bJ = (EPS0 * wp ** 2 * dt / 2.0) / (1.0 + gam * dt / 2.0)
        U_e = np.zeros((nx, nz))                             # electron energy density above T_e0 [J/m^3]
        Te_hot = np.array(hot["Te0"], dtype=np.float64)      # T_e map (U_e == 0 -> T_e0 per material)
        n_mat = len(htab)
        _rec = hot_out is not None
        if _rec:
            msum = float(hmask.sum()) or 1.0
            pabs_mean = np.empty(nsteps); Te_mean = np.empty(nsteps)
            pabs_int = np.zeros((nx, nz)); E2_int = np.zeros((nx, nz))
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
        # R20 clamped-inversion gain line: the Lorentz recursion with the -kappa dN E source
        if do_gain:
            PGnew = G1 * PG + G2 * PGp + G3 * Ey
            curl = curl - (PGnew - PG) / dt
            PGp = PG; PG = PGnew
        # R20 follow-on DYNAMIC gain: G3(t) from the LOCAL inversion, then the stimulated
        # transfer S_st = -E dPG/dt/(hbar w_a) drives the four-level populations
        if do_gdyn:
            PGnew = G1 * PG + G2 * PGp - kapfac * (N2 - N1) * Ey
            dPG_dt = (PGnew - PG) / dt
            curl = curl - dPG_dt
            PGp = PG; PG = PGnew
            S_st = -(Ey * dPG_dt) / hw_a                          # transitions / (m^3 s); >0 = emission
            f30 = Wp * N0; f32 = N3 / tau32; f21 = N2 / tau21; f10 = N1 / tau10
            N0 = N0 + dt * (f10 - f30)
            N1 = N1 + dt * (f21 + S_st - f10)
            N2 = N2 + dt * (f32 - f21 - S_st)
            N3 = N3 + dt * (f30 - f32)
            if n == snap_step and gain_dyn_out is not None:
                # host conversion must go through .get() on cupy (np.asarray raises there)
                _dn = N2 - N1
                gain_dyn_out["dN_snap"] = (np.asarray(_dn.get()) if hasattr(_dn, "get")
                                           else np.asarray(_dn).copy())
        # R15 chi2 SHG polarization: P2 = eps0 chi2 E^2, lagged-explicit dP2/dt like the Lorentz
        if do_chi2:
            P2new = EPS0 * chi2 * Ey ** 2
            curl = curl - (P2new - P2) / dt
            P2 = P2new
        # R15 Raman: vibrational ADE on E^2, then P_R = eps0 chi3R E Q (the Stokes-gain coupling)
        if do_raman:
            Qnew = R1 * Q + R2 * Qp + R3 * Ey ** 2
            PRnew = EPS0 * chi3R * Ey * Qnew
            curl = curl - (PRnew - PR) / dt
            Qp = Q; Q = Qnew; PR = PRnew
        # E update: eps0 eps_eff dEy/dt = curl - J, semi-implicit Drude + instantaneous Kerr
        eps_eff = eps_inf + 3.0 * chi3 * Ey ** 2       # standard chi3: P = eps0 chi3 E^3 (C3-2)
        denom = EPS0 * eps_eff / dt + bJ / 2.0
        Eynew = (EPS0 * eps_eff / dt * Ey + curl - 0.5 * (1.0 + aJ) * Jy - 0.5 * bJ * Ey) / denom
        Jynew = aJ * Jy + bJ * (Eynew + Ey)            # (== Jy update below; split out so p_abs can read J^n)
        # HOT-CARRIER two-temperature ADE (roadmap 2.1): local Drude Joule dissipation heats the electron
        # gas -> T_e -> (wp,gamma) shift. p_abs = J.E co-locates J and E at the step midpoint (its cycle
        # average is the Drude absorption). Off-mask cells (non-Drude, or not opted in) carry mask 0 and,
        # for a Drude-off cell, J == 0, so p_abs == 0 and T_e stays at T_e0.
        if do_hot:
            p_abs = hmask * (0.5 * (Jy + Jynew)) * (0.5 * (Ey + Eynew))
            U_e += dt * (alpha_hot * p_abs - Ghot * (Te_hot - Tlhot))
            if _rec:
                pabs_mean[n] = float(p_abs.sum()) / msum
                Te_mean[n] = float((Te_hot * hmask).sum()) / msum
                pabs_int += p_abs
                E2_int += Eynew ** 2
            if (n % n_upd) == 0:                       # refresh T_e, (wp,gamma) and the Drude ADE coeffs
                for mi in range(n_mat):
                    sel = (mat_idx == mi)
                    if not sel.any():
                        continue
                    Te_g, U_g, wpr, gmr = htab[mi]
                    te = np.interp(U_e[sel], U_g, Te_g)   # invert U_e -> T_e (C_e integral table)
                    Te_hot[sel] = te
                    wp[sel] = wp0[sel] * np.interp(te, Te_g, wpr)
                    gam[sel] = gam0[sel] * np.interp(te, Te_g, gmr)
                aJ = (1.0 - gam * dt / 2.0) / (1.0 + gam * dt / 2.0)
                bJ = (EPS0 * wp ** 2 * dt / 2.0) / (1.0 + gam * dt / 2.0)
        Jy = Jynew
        Eynew[:, k_src] += src[n]            # soft plane source (uniform in x -> normal-incidence plane wave)
        Eynew[:, 0] = 0.0; Eynew[:, -1] = 0.0  # PEC backing the CPML
        Ey = Eynew
        # record E_y and H_x at the probe planes; AVERAGE H_x (at k+/-1/2) onto the E_y plane (at k) so
        # the Poynting flux E_y*H_x co-locates spatially -- else the half-cell z-offset carries a
        # per-diffraction-order phase (each order has a different k_z) that does NOT cancel in the ratio.
        eyL[n] = Ey[:, k_pL]; hxL[n] = 0.5 * (Hx[:, k_pL] + Hx[:, k_pL - 1])
        eyR[n] = Ey[:, k_pR]; hxR[n] = 0.5 * (Hx[:, k_pR] + Hx[:, k_pR - 1])
    if do_gdyn and gain_dyn_out is not None:
        _h = (lambda a: np.asarray(a.get()) if hasattr(a, "get") else np.asarray(a))
        gain_dyn_out["Npop_final"] = np.stack([_h(N0), _h(N1), _h(N2), _h(N3)])
    if do_hot and _rec:                                      # roadmap 2.1 probes for the 2.1 oracles
        hot_out["t"] = np.arange(nsteps) * dt
        hot_out["p_abs_mean"] = pabs_mean                    # (nsteps,) mask-averaged absorbed power [W/m^3]
        hot_out["Te_mean"] = Te_mean                         # (nsteps,) mask-averaged electron temperature
        hot_out["p_abs_int"] = pabs_int * dt                 # (nx,nz) absorbed energy density [J/m^3]
        hot_out["E2_int"] = E2_int                           # (nx,nz) time-integrated |E_y|^2 (fluence proxy)
        hot_out["Te_final"] = Te_hot.copy()                  # (nx,nz) final electron-temperature map
        hot_out["mask"] = hmask.astype(bool)                 # (nx,nz) opted-in-cell mask
    return eyL, hxL, eyR, hxR


_run_2d_te = run_2d_te                                       # back-compat alias (pre-promotion name)
