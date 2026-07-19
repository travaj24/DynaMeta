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
from typing import List

import numpy as np

from dynameta.constants import C_LIGHT, EPS0, MU0  # MU0 single-sourced in constants (was re-derived here)


# FDTDLayer moved to the n-D package (fdtd_nd/spec.py, audit 2026-07-05 section 5 ownership
# inversion); re-exported at its old home so `from dynameta.optics.fdtd import FDTDLayer` keeps working.
from dynameta.optics.fdtd_nd.spec import FDTDLayer  # noqa: F401 (re-export)


@dataclass
class FDTD1DResult:
    freqs_Hz: np.ndarray
    R: np.ndarray
    T: np.ndarray
    band: np.ndarray            # boolean mask of the trustworthy (well-excited) frequency band
    # OPT-IN probe (roadmap 1.2, additive; None unless solve_fdtd_1d(return_time_trace=True)).
    # dict of the recorded boundary time series already used for the R/T DFT -- exposed as copies
    # for ringdown harmonic inversion. Keys: dt, t, reflected, transmitted, incident_left,
    # incident_right. Leaving it None keeps R/T/band/freqs_Hz byte-identical to the legacy path.
    time_trace: object = None


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
    # audit S2-17: with Kerr off (the default linear R/T case) eps_eff and the E-denominator are
    # loop-invariant; precompute once. curl is preallocated and refilled in place.
    has_kerr = bool(np.any(chi3 != 0.0))
    e0e_lin = EPS0 * eps_inf / dt
    denom_lin = e0e_lin + bJ / 2.0
    curl = np.zeros(nz)
    for n in range(nsteps):
        # old edge + adjacent cells for the Mur ABC. Position-matched names: L = left end (cells 0,1),
        # R = right end (cells -1,-2); *0 = the boundary cell, *1 = the adjacent interior cell.
        Ex_oldL0, Ex_oldL1 = Ex[0], Ex[1]
        Ex_oldR0, Ex_oldR1 = Ex[-1], Ex[-2]
        # H update (Hy[i] between Ex[i], Ex[i+1])
        Hy += (dt / (MU0 * dz)) * (Ex[1:] - Ex[:-1])
        # E update (interior): eps0 eps_eff dE/dt = -dHy/dz - (J^{n+1}+J^n)/2, with the Kerr eps_eff
        curl[1:-1] = (Hy[1:] - Hy[:-1]) / dz
        if has_kerr:
            eps_eff = eps_inf + 3.0 * chi3 * Ex ** 2   # Kerr: d(chi3 E^3)/dt = 3 chi3 E^2 dE/dt (C3-2)
            e0e = EPS0 * eps_eff / dt
            denom = e0e + bJ / 2.0
        else:                                          # audit S2-17: loop-invariant linear case
            e0e = e0e_lin
            denom = denom_lin
        Enew = (e0e * Ex + curl - 0.5 * (1.0 + aJ) * J - 0.5 * bJ * Ex) / denom
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


def _run_tv(eps_inf, wp, gam, chi3, dz, dt, nsteps, i_src, i_pL, i_pR, src,
            tgrid, mask, eps_inf_of_t, drude_of_t, n_update):
    """Time-VARYING twin of `_run` (roadmap 2.2): the SAME Yee/ADE march, but the material of the
    `mask` cells may change DURING the march via the opt-in hooks, re-derived every `n_update` steps.

    Two physically distinct time boundaries are realized (Morgenthaler, IRE Trans. MTT 6:167 (1958)):

      * eps_inf change (a temporal boundary in the instantaneous permittivity) -- the D-PRESERVING
        update. At a purely temporal boundary D (= eps0 eps_inf E + P) is CONTINUOUS while E JUMPS:
        curl H = dD/dt has no spatial-derivative source at a temporal boundary, so a step in D would
        demand an infinite dD/dt (= finite curl H) -- impossible. Hence D is continuous and, holding
        the stored Drude polarization P (the current state J) fixed, E must rescale
            E_new = E_old * eps_old / eps_new
        (derive E from the conserved D with the NEW eps_inf). FIELD-preserving (leaving E untouched
        when eps jumps) would VIOLATE this boundary condition and inject/destroy energy unphysically.
        On the Yee grid this E rescale with H left untouched (B continuous, mu constant) reproduces
        EXACTLY the Morgenthaler forward+backward time-boundary split (see the test derivation).

      * Drude wp/gamma change -- the polarization CURRENT J is a physical current, CONTINUOUS across
        the boundary; only the J-update coefficients (aJ, bJ) take the new wp/gamma. No field jump.

    COST + APPROXIMATION: re-deriving the affected coefficients is O(nz) per update (negligible vs
    the O(nz) field update); n_update>1 amortizes it. This is the INSTANTANEOUS-PARAMETER
    approximation -- the material is treated as piecewise-constant between updates, using parameters
    frozen at the update instant. A rigorous time-dependent-polarization formulation would convolve
    the constitutive relation with the continuously time-varying response kernel; the instantaneous
    scheme is O(n_update*dt) accurate and exact in the adiabatic (slow) and step (fast) limits, which
    are the two physically meaningful regimes. When the hooks return values equal to the initial ones
    the update block is a strict no-op, so this function is BIT-IDENTICAL to `_run` (gate 1)."""
    eps_inf = np.array(eps_inf, dtype=float, copy=True)     # local copies: hooks mutate in place
    wp = np.array(wp, dtype=float, copy=True)
    gam = np.array(gam, dtype=float, copy=True)
    nz = eps_inf.size
    Ex = np.zeros(nz)
    Hy = np.zeros(nz - 1)
    J = np.zeros(nz)                                   # Drude polarization current
    aJ = (1.0 - gam * dt / 2.0) / (1.0 + gam * dt / 2.0)
    bJ = (EPS0 * wp ** 2 * dt / 2.0) / (1.0 + gam * dt / 2.0)
    eL = np.empty(nsteps)
    eR = np.empty(nsteps)
    c = C_LIGHT
    mur = (c * dt - dz) / (c * dt + dz)                # pads stay vacuum: Mur uses the vacuum speed
    has_kerr = bool(np.any(chi3 != 0.0))
    e0e_lin = EPS0 * eps_inf / dt
    denom_lin = e0e_lin + bJ / 2.0
    curl = np.zeros(nz)
    # scalar snapshots of the (uniform) masked layer's material -- cheap change-detection so a static
    # hook is a strict no-op (byte-identity), and so the D-preserving rescale uses eps_old/eps_new.
    cur_eps = float(eps_inf[mask][0]) if mask.any() else 0.0
    cur_wp = float(wp[mask][0]) if mask.any() else 0.0
    cur_gam = float(gam[mask][0]) if mask.any() else 0.0
    for n in range(nsteps):
        # --- opt-in material update (every n_update steps); no-op when nothing actually changes ---
        if (n % n_update) == 0:
            tt = tgrid[n]
            if eps_inf_of_t is not None:
                new_eps = float(eps_inf_of_t(tt))
                if new_eps != cur_eps:
                    Ex[mask] *= cur_eps / new_eps               # D-preserving: E jumps, D continuous
                    eps_inf[mask] = new_eps
                    cur_eps = new_eps
                    e0e_lin = EPS0 * eps_inf / dt               # eps_inf-dependent coeffs re-derived
                    denom_lin = e0e_lin + bJ / 2.0
            if drude_of_t is not None:
                _wp, _gam = drude_of_t(tt)
                new_wp = float(_wp); new_gam = float(_gam)
                if (new_wp != cur_wp) or (new_gam != cur_gam):
                    wp[mask] = new_wp; gam[mask] = new_gam      # J itself is CONTINUOUS (untouched)
                    aJ[mask] = (1.0 - new_gam * dt / 2.0) / (1.0 + new_gam * dt / 2.0)
                    bJ[mask] = (EPS0 * new_wp ** 2 * dt / 2.0) / (1.0 + new_gam * dt / 2.0)
                    cur_wp = new_wp; cur_gam = new_gam
                    denom_lin = e0e_lin + bJ / 2.0
        # --- the EXACT `_run` inner body (byte-identical when the block above did nothing) ---
        Ex_oldL0, Ex_oldL1 = Ex[0], Ex[1]
        Ex_oldR0, Ex_oldR1 = Ex[-1], Ex[-2]
        Hy += (dt / (MU0 * dz)) * (Ex[1:] - Ex[:-1])
        curl[1:-1] = (Hy[1:] - Hy[:-1]) / dz
        if has_kerr:
            eps_eff = eps_inf + 3.0 * chi3 * Ex ** 2
            e0e = EPS0 * eps_eff / dt
            denom = e0e + bJ / 2.0
        else:
            e0e = e0e_lin
            denom = denom_lin
        Enew = (e0e * Ex + curl - 0.5 * (1.0 + aJ) * J - 0.5 * bJ * Ex) / denom
        Jnew = aJ * J + bJ * (Enew + Ex)
        Ex_int = Enew
        Ex_int[i_src] += src[n]
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
                  source_amp: float = 1.0,
                  return_time_trace: bool = False,
                  eps_inf_of_t=None, drude_of_t=None,
                  time_varying_layer: int = 0, n_update: int = 1) -> FDTD1DResult:
    """Broadband R(f)/T(f) of the layered `layers` (vacuum super/substrate) over
    [c/lambda_max, c/lambda_min]. `resolution` = cells per lambda_min in the highest-index medium;
    `courant` the CFL fraction (<= 1, use ~0.5 for Drude); `n_pad_wave` vacuum padding (in lambda_max)
    each side; `settle` the run length in pulse-widths. `kerr=False` zeroes chi3 (linear R/T). Returns
    FDTD1DResult(freqs_Hz, R, T, band) over the well-excited band.

    OPT-IN TIME VARIATION (roadmap 2.2 -- time refraction / photon acceleration). Supply one or both
    of the hooks to make the designated layer's material change DURING the march (the STRUCTURE run
    only; the vacuum reference run that fixes the incident field is always static):
      * eps_inf_of_t(t) -> scalar eps_inf for layer `time_varying_layer` at simulation time t. A change
        is realized D-PRESERVING (D continuous, E jumps E_new=E_old*eps_old/eps_new -- THE physically
        correct temporal-boundary condition, Morgenthaler IRE Trans. MTT 6:167 (1958); see _run_tv).
      * drude_of_t(t) -> (wp, gamma) for that layer; the polarization current J is CONTINUOUS, only the
        J-update coefficients take the new wp/gamma.
    `n_update` re-derives the affected coefficients every n_update steps (INSTANTANEOUS-PARAMETER
    approximation; see _run_tv for the cost + the rigorous-formulation note). Both hooks None (default)
    keeps the STRUCTURE run on `_run`, so R/T/band/freqs_Hz are BYTE-IDENTICAL to the legacy path; a
    static hook (returning the initial constants) routes through `_run_tv` but is bit-identical by the
    no-op change-detection. Pair with return_time_trace=True + frequency_conversion_diagnostic to read
    the converted output spectrum. NOTE: the grid is sized from the INITIAL layer params, so a large
    eps_inf INCREASE mid-run under-resolves the (shortened) in-medium wavelength -- keep changes modest
    or raise `resolution`."""
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

    # OPT-IN (roadmap 2.2): cell mask of the designated time-varying layer. Built only when a hook is
    # supplied, so the default (no-hook) path below is byte-for-byte the legacy code.
    _hooks_on = (eps_inf_of_t is not None) or (drude_of_t is not None)
    tv_mask = None
    if _hooks_on:
        tv_mask = np.zeros(nz, dtype=bool)
        zt = z0
        for idx, L in enumerate(layers):
            mm = (zc >= zt) & (zc < zt + L.thickness_m)
            if idx == time_varying_layer:
                tv_mask = mm
            zt += L.thickness_m
        if not tv_mask.any():
            raise ValueError("solve_fdtd_1d: time_varying_layer={} designates no grid cells".format(
                time_varying_layer))

    # modulated-Gaussian source covering [f_min, f_max]
    tau = 1.0 / (np.pi * (f_max - f_min))              # pulse width ~ inverse bandwidth
    t0 = settle * tau
    nsteps = int(round((2.0 * t0 + (Lz / C_LIGHT) * 4.0 + 200 * tau) / dt))
    tgrid = np.arange(nsteps) * dt
    src = source_amp * np.exp(-((tgrid - t0) / tau) ** 2) * np.cos(2.0 * np.pi * f_c * (tgrid - t0))

    # reference (vacuum) run for the incident field, then the structure run. The reference is ALWAYS
    # static (it defines the unshifted incident field); only the structure run carries the hooks.
    z1 = np.ones(nz)
    z0v = np.zeros(nz)
    eL_inc, eR_inc = _run(z1, z0v, z0v, z0v, dz, dt, nsteps, i_src, i_pL, i_pR, src)
    if _hooks_on:
        eL_tot, eR_tot = _run_tv(eps_inf, wp, gam, chi3, dz, dt, nsteps, i_src, i_pL, i_pR, src,
                                 tgrid, tv_mask, eps_inf_of_t, drude_of_t, max(1, int(n_update)))
    else:
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
    # OPT-IN (roadmap 1.2): expose the boundary time series already recorded above, as copies.
    # This is purely additive -- R/T/band/freqs_Hz are computed identically whether or not the
    # trace is attached, so return_time_trace=False (default) is byte-identical to the legacy path.
    time_trace = None
    if return_time_trace:
        time_trace = {
            "dt": dt,
            "t": tgrid.copy(),
            "reflected": (eL_tot - eL_inc).copy(),      # reflected = total - incident (left probe)
            "transmitted": eR_tot.copy(),               # transmitted (right probe)
            "incident_left": eL_inc.copy(),
            "incident_right": eR_inc.copy(),
        }
    return FDTD1DResult(freqs_Hz=f, R=R, T=T, band=band, time_trace=time_trace)


def _spectral_centroid(freqs, power, sel):
    """Power-weighted spectral centroid over the selected bins (returns NaN if no power)."""
    w = power[sel]
    tot = float(np.sum(w))
    if tot <= 0.0:
        return float("nan")
    return float(np.sum(freqs[sel] * w) / tot)


def frequency_conversion_diagnostic(trace, *, band_Hz=None, output="transmitted",
                                    reference="incident_right", rel_floor=0.05):
    """Frequency-conversion diagnostic (roadmap 2.2 part b): the output spectrum + centroid vs the
    input, from a recorded FDTD time trace. Pass either an FDTD1DResult (with return_time_trace=True)
    or the time_trace dict itself, or a UniformTimeVaryingResult.

    Compares the power spectrum of the `output` series (default the transmitted trace) against the
    `reference` input series (default the right-probe incident field). The centroid of each is the
    power-weighted mean frequency over `band_Hz` (a (f_lo, f_hi) tuple; default all positive bins),
    restricted to bins above rel_floor*max so a noise floor cannot drag the centroid. Returns a dict:
    freqs_Hz, input_spectrum (|.|^2), output_spectrum, input_centroid_Hz, output_centroid_Hz, and
    ratio = output_centroid/input_centroid (= n_in/n_out for adiabatic time refraction; = the
    Morgenthaler shifted frequency ratio for a fast temporal boundary)."""
    tt = getattr(trace, "time_trace", trace)
    if isinstance(trace, UniformTimeVaryingResult):
        tt = {"dt": trace.dt, output: getattr(trace, output), reference: getattr(trace, reference)}
    if tt is None:
        raise ValueError("frequency_conversion_diagnostic: no time trace (call with "
                         "return_time_trace=True, or pass the trace dict / uniform result)")
    dt = float(tt["dt"])
    out = np.asarray(tt[output], dtype=float)
    ref = np.asarray(tt[reference], dtype=float)
    f = np.fft.rfftfreq(out.size, dt)
    Po = np.abs(np.fft.rfft(out)) ** 2
    fr = np.fft.rfftfreq(ref.size, dt)
    Pr = np.abs(np.fft.rfft(ref)) ** 2
    if band_Hz is not None:
        sel_o = (f >= band_Hz[0]) & (f <= band_Hz[1])
        sel_r = (fr >= band_Hz[0]) & (fr <= band_Hz[1])
    else:
        sel_o = f > 0.0
        sel_r = fr > 0.0
    if sel_o.any():
        sel_o = sel_o & (Po >= rel_floor * np.max(Po[sel_o]))
    if sel_r.any():
        sel_r = sel_r & (Pr >= rel_floor * np.max(Pr[sel_r]))
    fin = _spectral_centroid(fr, Pr, sel_r)
    fout = _spectral_centroid(f, Po, sel_o)
    return {"freqs_Hz": f, "input_spectrum": Pr, "output_spectrum": Po,
            "input_centroid_Hz": fin, "output_centroid_Hz": fout,
            "ratio": (fout / fin) if fin == fin and fin != 0.0 else float("nan")}


# --------------------------------------------------------------------------------------------------
# Uniform-medium time-boundary harness (roadmap 2.2 gates): the CLEAN adiabatic / fast-boundary
# oracle. A finite forward wavepacket is launched by DIRECT initial condition into a uniform medium
# that fills the whole grid -- no spatial interfaces (so no Fresnel factors contaminate the
# amplitudes) and the run is stopped before anything reaches the walls (so the Mur ABC is irrelevant).
# The uniform index changes in time via the SAME D-preserving update as _run_tv (E rescaled by
# eps_old/eps_new, H untouched). This is the transparent physics setting for photon-number
# conservation (adiabatic) and the Morgenthaler forward/backward split (fast step); solve_fdtd_1d is
# the general LAYERED path. Non-dispersive dielectric only (eps = n(t)^2); no Drude/Kerr here.
# --------------------------------------------------------------------------------------------------

@dataclass
class UniformTimeVaryingResult:
    dt: float
    dz: float
    z_m: np.ndarray                 # cell centers
    t_s: np.ndarray                 # time grid
    n_traj: np.ndarray              # index n(t) actually applied at each step
    Ex_final: np.ndarray            # final E snapshot (whole grid)
    Hy_final: np.ndarray            # final H snapshot
    transmitted: np.ndarray         # right-probe (forward-daughter) trace
    reflected: np.ndarray           # left-probe (backward/time-reflected-daughter) trace
    incident_right: np.ndarray      # = transmitted (alias so the diagnostic's default reference works)
    i_pL: int
    i_pR: int
    energy_t: np.ndarray            # total field energy vs step (0.5 eps0 eps Ex^2 + 0.5 mu0 Hy^2)
    k_rad_m: float                  # (conserved) spatial carrier wavenumber
    w_init_rad_s: float             # initial temporal carrier c*k/n_init


def _march_uniform(Ex, Hy, dz, dt, nsteps, index_of_t, n_init, tgrid, i_pL, i_pR, n_update):
    """Leapfrog march of a uniform (spatially constant) but time-varying dielectric. Same Yee update
    signs as `_run` with Drude/Kerr off; the D-preserving E rescale is applied on ALL cells when the
    index changes. Records both probes, the index trajectory, and the total field energy each step."""
    N = Ex.size
    c = C_LIGHT
    cur_n = float(n_init)
    eps = cur_n ** 2
    mur = (c / cur_n * dt - dz) / (c / cur_n * dt + dz)   # ABC tuned to the (current) medium speed
    eL = np.empty(nsteps); eR = np.empty(nsteps)
    energy = np.empty(nsteps); ntraj = np.empty(nsteps)
    curl = np.zeros(N)
    for n in range(nsteps):
        if (n % n_update) == 0:
            new_n = float(index_of_t(tgrid[n]))
            if new_n != cur_n:
                Ex *= (cur_n ** 2) / (new_n ** 2)            # D-preserving temporal boundary
                cur_n = new_n; eps = cur_n ** 2
                mur = (c / cur_n * dt - dz) / (c / cur_n * dt + dz)
        ntraj[n] = cur_n
        Ex_oldL0, Ex_oldL1 = Ex[0], Ex[1]
        Ex_oldR0, Ex_oldR1 = Ex[-1], Ex[-2]
        Hy += (dt / (MU0 * dz)) * (Ex[1:] - Ex[:-1])
        curl[1:-1] = (Hy[1:] - Hy[:-1]) / dz
        Enew = Ex + (dt / (EPS0 * eps)) * curl              # non-dispersive: eps0 eps dE/dt = curl H
        Enew[0] = Ex_oldL1 + mur * (Enew[1] - Ex_oldL0)
        Enew[-1] = Ex_oldR1 + mur * (Enew[-2] - Ex_oldR0)
        Ex = Enew
        eL[n] = Ex[i_pL]; eR[n] = Ex[i_pR]
        energy[n] = 0.5 * EPS0 * eps * float(np.sum(Ex ** 2)) + 0.5 * MU0 * float(np.sum(Hy ** 2))
    return Ex, Hy, eL, eR, energy, ntraj


def run_uniform_time_boundary(*, index_of_t, n_init: float, lambda_med_m: float,
                              domain_wavelengths: float = 120.0, cells_per_wavelength: int = 30,
                              courant: float = 0.5, pulse_fwhm_wavelengths: float = 6.0,
                              run_periods: float = 60.0, n_update: int = 1,
                              probe_offset_wavelengths: float = 20.0, amp: float = 1.0):
    """March a finite FORWARD wavepacket through a spatially-uniform, time-VARYING dielectric medium
    (roadmap 2.2 adiabatic / fast-boundary oracle). The medium index is index_of_t(t); n_init =
    index_of_t(0). The wavepacket is set by a direct forward-wave initial condition (Gaussian envelope
    of FWHM pulse_fwhm_wavelengths, carrier wavelength lambda_med_m IN THE INITIAL MEDIUM), centered in
    a domain of domain_wavelengths, so it is FULLY INSIDE the medium during any index change.

    The conserved quantity is the spatial wavenumber k = 2 pi / lambda_med_m (spatial translation
    symmetry survives a purely temporal change); the temporal frequency tracks omega(t) = c k / n(t),
    so a slow ramp adiabatically translates the frequency (omega_out/omega_in = n_in/n_out) and a fast
    step splits the wave into forward + backward daughters at omega_out = omega_in n_in/n_out with the
    Morgenthaler amplitudes. Returns a UniformTimeVaryingResult. Run length run_periods is in initial
    optical periods; keep it short enough that the daughters do not reach the walls (checked by the
    caller via the snapshots)."""
    k = 2.0 * np.pi / lambda_med_m
    n0 = float(n_init)
    w_init = C_LIGHT * k / n0
    dz = lambda_med_m / cells_per_wavelength
    dt = courant * dz / C_LIGHT                            # stable for any n>=1 (fastest speed <= c)
    N = int(round(domain_wavelengths * cells_per_wavelength)) + 1
    z = np.arange(N) * dz
    z_c = 0.5 * (N - 1) * dz                               # pulse center: room for both daughters
    sigma = (pulse_fwhm_wavelengths * lambda_med_m) / (2.0 * np.sqrt(np.log(2.0)))  # FWHM->1/e^2 half
    v = C_LIGHT / n0
    # forward-wave initial condition on the staggered Yee grid. Ex at integer nodes, time level 0;
    # Hy at half nodes, time level -1/2 (half a step BEHIND) -- the half-step phase/envelope shift makes
    # the launch purely FORWARD to O((k dz)^2). SIGN: this module's Yee update is mu0 dHy/dt = +dEx/dz
    # (Hy is the negative of the textbook Hy), so a +z-propagating wave has Hy = -(n0/(mu0 c)) Ex.
    env = lambda zz: np.exp(-((zz - z_c) / sigma) ** 2)
    Ex0 = amp * env(z) * np.cos(k * z)
    zc_h = (np.arange(N - 1) + 0.5) * dz
    Hy0 = -(n0 / (MU0 * C_LIGHT)) * amp * env(zc_h + 0.5 * v * dt) * np.cos(k * zc_h + w_init * dt / 2.0)
    i_pR = int(round((z_c + probe_offset_wavelengths * lambda_med_m) / dz))
    i_pL = int(round((z_c - probe_offset_wavelengths * lambda_med_m) / dz))
    i_pR = min(i_pR, N - 1); i_pL = max(i_pL, 0)
    T_run = run_periods * (2.0 * np.pi / w_init)
    nsteps = int(round(T_run / dt))
    tgrid = np.arange(nsteps) * dt
    Ex_f, Hy_f, eL, eR, energy, ntraj = _march_uniform(
        Ex0.copy(), Hy0.copy(), dz, dt, nsteps, index_of_t, n0, tgrid, i_pL, i_pR, max(1, int(n_update)))
    return UniformTimeVaryingResult(
        dt=dt, dz=dz, z_m=z, t_s=tgrid, n_traj=ntraj, Ex_final=Ex_f, Hy_final=Hy_f,
        transmitted=eR, reflected=eL, incident_right=eR, i_pL=i_pL, i_pR=i_pR,
        energy_t=energy, k_rad_m=k, w_init_rad_s=w_init)
