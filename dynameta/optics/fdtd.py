"""
1D FDTD (Yee) optical solver -- the time-domain backend (roadmap C9): a single broadband normal-
incidence pulse gives the whole reflection/transmission spectrum R(omega)/T(omega) of a layered
stack, with DISPERSIVE Drude materials (auxiliary-differential-equation, ADE) and an optional
instantaneous KERR (chi3) nonlinearity (the all-optical / self-phase-modulation axis). The
time-domain companion to the frequency-domain FEM/TMM: dispersion and nonlinearity are native here.

Method: staggered Ex/Hy leapfrog along z; vacuum super/substrate with 1st-order Mur ABC at both
ends; a soft modulated-Gaussian plane-wave source. R/T are extracted by the standard TWO-RUN
reference method -- a vacuum reference run gives the incident field at the left/right probes; the
structure run gives total fields, so reflected = total_L - incident_L and transmitted = total_R; the
spectra are the DFT ratios. Lossless => R + T = 1 (checked vs TMM). 1D normal incidence only; a
2D/3D periodic-Bloch FDTD is a future extension. Convention exp(-i omega t), SI; the Drude ADE uses
eps(w) = eps_inf - wp^2/(w^2 + i gamma w).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from dynameta.constants import C_LIGHT, EPS0, MU0  # MU0 single-sourced in constants (was re-derived here)


@dataclass
class FDTDLayer:
    """One layer of the 1D stack. Non-dispersive eps_inf, plus an optional Drude pole
    (eps -= wp^2/(w^2 + i gamma w)), an optional Lorentz pole (eps += d_eps w0^2/(w0^2 - w^2 - i gl w),
    the bound-electron / interband resonance the bare Drude cannot capture) and an instantaneous Kerr
    chi3 in the STANDARD chi^(3) convention: P_NL = eps0 chi3 E^3, i.e. the update uses
    eps_eff = eps_inf + 3 chi3 E(t)^2 and the fundamental-band shift is
    d_eps = (3/4) chi3 |A|^2 (audit C3-2: the update previously used eps_inf + chi3 E^2 --
    an effective chi^(3) three times weaker than the literature value entered, and 4x weaker
    than this docstring then claimed; chi3_m2_V2 now takes literature chi^(3) values
    directly). All optional terms default off, so a plain dielectric is just eps_inf.

    The 2D/3D engine (fdtd_nd) carries the Drude + Kerr; the Lorentz pole is honored by the 2D-TE kernels
    (numpy/numba) via the central-difference ADE (a second polarization state PL). eps(w) at one lambda is
    typically supplied by inverting to a single Drude pole (the seam); fit_drude_lorentz fits BOTH poles
    across a band for an exact broadband metal/interband representation."""
    thickness_m: float
    eps_inf: float = 1.0
    drude_wp_rad_s: float = 0.0
    drude_gamma_rad_s: float = 0.0
    chi3_m2_V2: float = 0.0
    lorentz_w0_rad_s: float = 0.0          # Lorentz resonance frequency (0 = no Lorentz pole)
    lorentz_gamma_rad_s: float = 0.0       # Lorentz damping rate
    lorentz_delta_eps: float = 0.0         # Lorentz static strength (eps(0) gains d_eps)
    # R15 second-order + dispersive third-order nonlinearities (2D-TE numpy kernel; default off):
    chi2_m_V: float = 0.0                  # SHG chi2 [m/V]: P2 = eps0 chi2 E^2 polarization source
    raman_chi3_m2_V2: float = 0.0          # Raman chi3 strength [m^2/V^2]: P_R = eps0 chiR E Q
    raman_w_rad_s: float = 0.0             # Raman vibrational resonance Omega_R [rad/s]
    raman_gamma_rad_s: float = 0.0         # Raman damping [rad/s] (Q'' + g Q' + W^2 Q = W^2 E^2)
    # R20 CLAMPED-INVERSION gain line (2D-TE numpy kernel; default off). The Lorentz-oscillator
    # gain ADE P'' + dw P' + w^2 P = -kappa dN E: inversion dN = N2 - N1 > 0 -> Im(chi) < 0 = GAIN
    # (exp(-i w t)); dN < 0 reduces EXACTLY to a passive Lorentz pole with delta_eps =
    # kappa |dN|/(eps0 w^2). kappa is the classical coupling q^2/m_eff [C^2/kg]; the small-signal
    # intensity gain at line center is g0 = kappa dN / (n c eps0 dw) [1/m]. The inversion is
    # CLAMPED (no rate dynamics in the field loop -- see optics.laser_gain for the four-level
    # populations; dynamic field-population coupling is a documented follow-on).
    gain_w_rad_s: float = 0.0              # transition frequency w_a [rad/s] (0 = no gain line)
    gain_dw_rad_s: float = 0.0             # FWHM linewidth dw_a [rad/s]
    gain_kappa_C2_kg: float = 0.0          # coupling q^2/m_eff [C^2/kg]
    gain_dN_m3: float = 0.0                # clamped inversion N2 - N1 [m^-3] (sign = gain/loss)

    def eps_at(self, w_rad_s):
        """The complex eps(omega) this layer represents (eps_inf - Drude + Lorentz), convention
        exp(-i w t), Im(eps) > 0 = loss. Excludes the intensity-dependent Kerr term."""
        e = complex(self.eps_inf)
        if self.drude_wp_rad_s > 0.0:
            e = e - self.drude_wp_rad_s ** 2 / (w_rad_s ** 2 + 1j * self.drude_gamma_rad_s * w_rad_s)
        if self.lorentz_delta_eps != 0.0 and self.lorentz_w0_rad_s > 0.0:
            w0 = self.lorentz_w0_rad_s
            e = e + self.lorentz_delta_eps * w0 ** 2 / (w0 ** 2 - w_rad_s ** 2 - 1j * self.lorentz_gamma_rad_s * w_rad_s)
        return e


@dataclass
class FDTD1DResult:
    freqs_Hz: np.ndarray
    R: np.ndarray
    T: np.ndarray
    band: np.ndarray            # boolean mask of the trustworthy (well-excited) frequency band


def _run(eps_inf, wp, gam, chi3, dz, dt, nsteps, i_src, i_pL, i_pR, src):
    """One FDTD pass over a prebuilt cell-wise (eps_inf, wp, gamma, chi3) profile. Returns the E
    time series at the left and right probes. Semi-implicit Drude ADE + instantaneous Kerr."""
    nz = eps_inf.size
    Ex = np.zeros(nz)
    Hy = np.zeros(nz - 1)
    J = np.zeros(nz)                                   # Drude polarization current
    aJ = (1.0 - gam * dt / 2.0) / (1.0 + gam * dt / 2.0)
    bJ = (EPS0 * wp ** 2 * dt / 2.0) / (1.0 + gam * dt / 2.0)
    eL = np.empty(nsteps)
    eR = np.empty(nsteps)
    c = C_LIGHT
    mur = (c * dt - dz) / (c * dt + dz)
    for n in range(nsteps):
        # old edge + adjacent cells for the Mur ABC. Position-matched names: L = left end (cells 0,1),
        # R = right end (cells -1,-2); *0 = the boundary cell, *1 = the adjacent interior cell.
        Ex_oldL0, Ex_oldL1 = Ex[0], Ex[1]
        Ex_oldR0, Ex_oldR1 = Ex[-1], Ex[-2]
        # H update (Hy[i] between Ex[i], Ex[i+1])
        Hy += (dt / (MU0 * dz)) * (Ex[1:] - Ex[:-1])
        # E update (interior): eps0 eps_eff dE/dt = -dHy/dz - (J^{n+1}+J^n)/2, with the Kerr eps_eff
        curl = np.zeros(nz)
        curl[1:-1] = (Hy[1:] - Hy[:-1]) / dz
        eps_eff = eps_inf + 3.0 * chi3 * Ex ** 2       # Kerr: d(chi3 E^3)/dt = 3 chi3 E^2 dE/dt (C3-2)
        denom = EPS0 * eps_eff / dt + bJ / 2.0
        Enew = (EPS0 * eps_eff / dt * Ex + curl - 0.5 * (1.0 + aJ) * J - 0.5 * bJ * Ex) / denom
        Jnew = aJ * J + bJ * (Enew + Ex)
        Ex_int = Enew
        Ex_int[i_src] += src[n]                        # soft source
        # 1st-order Mur ABC at both ends (overwrite the edge cells)
        Ex_int[0] = Ex_oldL1 + mur * (Ex_int[1] - Ex_oldL0)
        Ex_int[-1] = Ex_oldR1 + mur * (Ex_int[-2] - Ex_oldR0)
        Ex = Ex_int
        J = Jnew
        eL[n] = Ex[i_pL]
        eR[n] = Ex[i_pR]
    return eL, eR


def solve_fdtd_1d(layers: List[FDTDLayer], *, lambda_min_m: float, lambda_max_m: float,
                  resolution: int = 40, courant: float = 0.5, n_pad_wave: float = 6.0,
                  settle: float = 12.0, kerr: bool = False,
                  source_amp: float = 1.0) -> FDTD1DResult:
    """Broadband R(f)/T(f) of the layered `layers` (vacuum super/substrate) over
    [c/lambda_max, c/lambda_min]. `resolution` = cells per lambda_min in the highest-index medium;
    `courant` the CFL fraction (<= 1, use ~0.5 for Drude); `n_pad_wave` vacuum padding (in lambda_max)
    each side; `settle` the run length in pulse-widths. `kerr=False` zeroes chi3 (linear R/T). Returns
    FDTD1DResult(freqs_Hz, R, T, band) over the well-excited band."""
    # The 1-D engine carries eps_inf + Drude + Kerr(chi3) ONLY. The Lorentz pole, chi2, Raman and the
    # gain line are honored solely by the 2-D-TE kernels (fdtd_nd); _run never reads those arrays, so
    # solving them here would SILENTLY drop the term and the FDTD eps would diverge from the layer's own
    # eps_at(w). Raise loudly (matching the 2D/3D siblings) rather than mis-solve.
    def _unsupported_1d(L):
        bad = []
        if L.lorentz_delta_eps != 0.0 and L.lorentz_w0_rad_s > 0.0:
            bad.append("lorentz")
        if getattr(L, "chi2_m_V", 0.0) != 0.0:
            bad.append("chi2")
        if getattr(L, "raman_chi3_m2_V2", 0.0) != 0.0:
            bad.append("raman")
        if getattr(L, "gain_dN_m3", 0.0) != 0.0 and getattr(L, "gain_w_rad_s", 0.0) > 0.0:
            bad.append("gain")
        return bad

    _bad = sorted({t for L in layers for t in _unsupported_1d(L)})
    if _bad:
        raise NotImplementedError(
            "solve_fdtd_1d supports eps_inf + Drude + Kerr(chi3) only; the layer term(s) {} are "
            "carried ONLY by the 2-D-TE kernels (optics.fdtd_nd), so a 1-D solve would silently drop "
            "them (the FDTD eps would diverge from FDTDLayer.eps_at(w)). Use the 2-D engine, or remove "
            "those terms.".format(_bad))
    f_min, f_max = C_LIGHT / lambda_max_m, C_LIGHT / lambda_min_m
    f_c = 0.5 * (f_min + f_max)
    # Size the grid from the DISPERSIVE |n| over the band, not just sqrt(eps_inf): a below-plasma Drude
    # metal has |eps(w)| >> eps_inf (largest at the band's low-frequency end), so the short skin depth
    # is otherwise silently under-resolved (audit). eps(w) = eps_inf - wp^2/(w^2 + i gamma w).
    w_min = 2.0 * np.pi * f_min

    def _n_band_max(L):
        eps = complex(L.eps_inf)
        if L.drude_wp_rad_s > 0.0:
            eps = eps - L.drude_wp_rad_s ** 2 / (w_min ** 2 + 1j * L.drude_gamma_rad_s * w_min)
        return abs(np.sqrt(eps))                       # |n| (sets both the wavelength and the skin depth)
    n_max = max(1.0, max(_n_band_max(L) for L in layers))
    dz = lambda_min_m / (resolution * n_max)
    dt = courant * dz / C_LIGHT
    pad = n_pad_wave * lambda_max_m
    z_struct = float(sum(L.thickness_m for L in layers))
    Lz = 2.0 * pad + z_struct
    nz = int(round(Lz / dz)) + 1

    # cell-wise material profile (structure centered, vacuum pads each side)
    eps_inf = np.ones(nz)
    wp = np.zeros(nz)
    gam = np.zeros(nz)
    chi3 = np.zeros(nz)
    z0 = pad
    zc = (np.arange(nz) + 0.5) * dz                    # cell centers
    z = z0
    for L in layers:
        m = (zc >= z) & (zc < z + L.thickness_m)
        eps_inf[m] = L.eps_inf
        wp[m] = L.drude_wp_rad_s
        gam[m] = L.drude_gamma_rad_s
        if kerr:
            chi3[m] = L.chi3_m2_V2
        z += L.thickness_m
    i_src = max(2, int(round((0.35 * pad) / dz)))
    i_pL = int(round((0.7 * pad) / dz))                # left probe: between source and structure
    i_pR = int(round((pad + z_struct + 0.3 * pad) / dz))  # right probe: in the sub vacuum

    # modulated-Gaussian source covering [f_min, f_max]
    tau = 1.0 / (np.pi * (f_max - f_min))              # pulse width ~ inverse bandwidth
    t0 = settle * tau
    nsteps = int(round((2.0 * t0 + (Lz / C_LIGHT) * 4.0 + 200 * tau) / dt))
    tgrid = np.arange(nsteps) * dt
    src = source_amp * np.exp(-((tgrid - t0) / tau) ** 2) * np.cos(2.0 * np.pi * f_c * (tgrid - t0))

    # reference (vacuum) run for the incident field, then the structure run
    z1 = np.ones(nz)
    z0v = np.zeros(nz)
    eL_inc, eR_inc = _run(z1, z0v, z0v, z0v, dz, dt, nsteps, i_src, i_pL, i_pR, src)
    eL_tot, eR_tot = _run(eps_inf, wp, gam, chi3, dz, dt, nsteps, i_src, i_pL, i_pR, src)

    f = np.fft.rfftfreq(nsteps, dt)
    Iinc_L = np.fft.rfft(eL_inc)
    Iinc_R = np.fft.rfft(eR_inc)
    Irefl = np.fft.rfft(eL_tot - eL_inc)               # reflected = total - incident at the left
    Itrans = np.fft.rfft(eR_tot)                       # transmitted at the right
    with np.errstate(divide="ignore", invalid="ignore"):
        R = np.abs(Irefl / Iinc_L) ** 2
        T = np.abs(Itrans / Iinc_R) ** 2
    band = (f >= f_min) & (f <= f_max) & (np.abs(Iinc_L) > 0.05 * np.max(np.abs(Iinc_L)))
    return FDTD1DResult(freqs_Hz=f, R=R, T=T, band=band)
