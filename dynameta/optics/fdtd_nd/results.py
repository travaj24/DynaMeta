"""Result dataclasses and flux post-processing shared by the solve front-ends.

Split from the former monolithic fdtd_nd.py; see the package __init__ docstring
for conventions. Bodies are verbatim from the original module.

POLARIZATION VOCABULARY (audit V-8): this module speaks {'x', 'y', 'p'} -- the LAB AXIS of the
incident E ('y' = s-pol, 'p' = p-pol, 'x' = E along lab x, transverse only at normal incidence). It
is one of five spellings in the repo -- {'s','p'} is the PLANE-OF-INCIDENCE spelling (tmm_reference,
resonance, nonlocal_tmm, shg_fem's closed forms, the oblique 2-D FDTD), {'te','tm'} the lumenairy
grating bridge's, the integer `row` 0/1 the differentiable Berreman/RCWA/PMM forwards', and
`pol_axis` hydro_fem's 2-D in-plane axis. The map, the `normalize_pol` converter and the
normal-incidence / azimuth caveats live in `dynameta.core.polarization`. The set ACCEPTED here is
UNCHANGED; acceptance unification (b), the V-8 follow-on, widened only the two PLANE-OF-INCIDENCE
families ({'s','p'} and {'te','tm'}), whose aliases name the same physical mode in every geometry
they cover; this vocabulary's crossings depend on the azimuth (or have no image at all), so they
stay STRICT and are made explicitly, through normalize_pol.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np



# --- R_flux / T_flux SIGN CONVENTION on the public result arrays (audit D-9) ---------------------
# `R_flux` / `T_flux` are the SIGNED Poynting ratios R_flux = -P_refl/P_inc, T_flux = P_trans/P_inc
# (solve2d._flux_ratios), not the pre-wave-5 abs(P)/abs(P_inc). Wherever the signs are the physical
# ones the two are BIT-IDENTICAL (IEEE negation is exact), so every IN-BAND value is unchanged --
# verified bitwise on a lossless slab, a Drude metal and an asymmetric-ends stack.
#
# OUT-OF-BAND BINS NOW COME BACK NEGATIVE ABOUT HALF THE TIME, and that is expected, not a defect:
# outside `band` the incident reference carries essentially no power, so these entries are ratios of
# two numerical-noise numbers whose signs are random. Measured on the three fixtures above: 47.3 %,
# 48.3 % and 53.7 % of the out-of-band R_flux bins are negative (46.8-53.1 % for T_flux), against
# 0 % before, when abs() disguised them as small positive numbers. THE NOISE IS THE SAME; only its
# presentation changed, and a negative value is the honest one.
#
# CONSUMERS MUST MASK WITH `band` (they already do): validation/* index `[res.band]`,
# fdtd_seam.run_fdtd_sweep masks with `res.band & (f > 0)`, and fdtd_seam's per-wavelength `_solve`
# interpolates at a target frequency inside the requested [lambda_min, lambda_max]. No in-repo
# consumer reads an unmasked out-of-band bin, so nothing downstream changed; an EXTERNAL consumer
# that plots the raw array without the mask will see sign flips in the noise skirts.
# The in-band sign flip that is NOT noise (a real backward-power result) warns -- see _flux_ratios.


@dataclass
class FDTD2DResult:
    freqs_Hz: np.ndarray
    R0: np.ndarray              # 0-order (specular) reflectance from the x-mean field (== 1D/TMM)
    T0: np.ndarray             # 0-order (specular) transmittance from the x-mean field
    R_flux: np.ndarray         # TOTAL reflectance from the Poynting flux, SIGNED (see the note above)
    T_flux: np.ndarray         # TOTAL transmittance from the Poynting flux, SIGNED (see the note above)
    band: np.ndarray            # boolean mask of the well-excited frequency band
    r0: Optional[np.ndarray] = None   # COMPLEX 0-order reflection coeff, phase de-embedded to the front face
    t0: Optional[np.ndarray] = None   # COMPLEX 0-order transmission coeff, de-embedded across the structure
    # OPT-IN probe (roadmap 3.1 harmonic diagnostics; None unless solve_fdtd_2d(return_time_trace=True),
    # mirroring the 1-D FDTD1DResult.time_trace precedent). dict of the recorded exit/entry-plane E_y + H_x
    # x-lines (nsteps, nx) already used for the R/T extraction -- exposed as copies so optics.harmonics can
    # read the 2w/3w content (which R0/T0/flux, normalized by the ~0 incident reference at n*f0, cannot
    # carry). Keys: dt, t, transmitted(+_hx), reflected(+_hx), incident_left(+_hx), incident_right(+_hx),
    # period_x_m, dx, nx. Leaving it None keeps every other field byte-identical to the legacy path.
    time_trace: Optional[dict] = None




def _half_step_H(nsteps, dt, ndim):
    """exp(+i w dt/2) on the rfft axis, broadcast against an `ndim`-dimensional probe spectrum.

    HALF-TIMESTEP DE-STAGGERING (audit D-2). The Yee kernels record E and H from the SAME loop
    iteration, but the H update runs first: at step n they store E(t = (n+1) dt) and
    H(t = (n+1/2) dt) (kernels2d.py:252-253, kernels3d.py:263-264), so H LAGS E by half a step.
    Its rfft therefore carries a spurious exp(-i w dt/2) relative to E's, and E conj(H) carries
    exp(+i w dt/2). Multiplying the H spectrum by exp(+i w dt/2) removes it exactly.

    Unlike the half-CELL z offset (already removed inside the kernel, which averages H_x onto the
    E_y plane) this does NOT cancel in the R/T ratio. For a PROPAGATING order E H* is nearly real
    and the residue is a common cos(w dt/2) that does cancel; for an EVANESCENT / near-cutoff order
    E H* is nearly pure-imaginary and the uncorrected flux picks up a spurious
    Im(E H*) sin(w dt/2) that survives it.

    SIGN, verified empirically 2026-07-25 (the 2026-07-25 audit's D-2 prescribes the OPPOSITE sign,
    which doubles the error):
      * synthetic evanescent order, true S_z == 0 exactly: uncorrected +7.342e-2 ->
        exp(+i w dt/2) gives -2.6e-16 (machine zero); exp(-i w dt/2) gives +1.460e-1 (1.989x, the
        predicted exactly-2 doubling).
      * real FDTD incident reference (a pure +z plane wave, so E/H must be a REAL -eta ratio):
        max|Im|/max|Re| 1.3873e-2 -> 1.9132e-5 with exp(+i w dt/2) (725x better) and -> 2.7735e-2
        with exp(-i w dt/2) (2x worse).
    IMPACT: on a laterally-uniform slab (propagating 0 order only) R_flux/T_flux move by <= 9.0e-7
    (an O(sin(w dt/2)^2) residue); on a real diffracting grating T_flux moves by up to 2.0e-3
    absolute and the thin-pad (n_pad_wave=2) energy-budget error drops 1.74e-4 -> 1.12e-5 median.
    R0/T0 are amplitude ratios and are NOT affected at all.
    """
    w = 2.0 * np.pi * np.fft.rfftfreq(nsteps, dt)
    return np.exp(0.5j * w * dt).reshape((-1,) + (1,) * (ndim - 1))


def _flux(ey, hx, dt):
    """Per-frequency time-averaged +z Poynting power S_z = -Re(E_y H_x*) summed over x, from the rfft
    of the recorded probe x-lines (shape (nsteps, nx)). The half-CELL z stagger is removed in the
    kernel (H_x averaged onto the E_y plane) and the half-TIMESTEP stagger here, via _half_step_H --
    neither cancels in the R/T ratio for an evanescent/near-cutoff order (audit D-2; the pre-fix
    docstring claimed both did)."""
    Ey = np.fft.rfft(ey, axis=0)
    Hx = np.fft.rfft(hx, axis=0) * _half_step_H(ey.shape[0], dt, 2)
    return -np.sum(np.real(Ey * np.conj(Hx)), axis=1)        # (nfreq,) signed z-power per frequency


@dataclass
class FDTD2DObliqueResult:
    freqs_Hz: np.ndarray
    theta_deg: np.ndarray       # the frequency-dependent physical angle for the fixed k_par
    R0: np.ndarray              # specular reflectance |r|^2 (s-pol)
    T0: np.ndarray              # specular transmittance |t|^2 (s-pol; vacuum ends -> kz factors cancel)
    band: np.ndarray            # well-excited + below-the-light-line (propagating) frequency mask


@dataclass
class FDTD3DResult:
    """Broadband R(f)/T(f) of a 2D-periodic unit cell (normal incidence, y-polarized source). R0/T0 = the
    specular (0-order) co-pol from the x,y-mean field (== 1D/TMM for a laterally-uniform stack); R_flux/
    T_flux = the total over ALL (kx,ky) diffraction orders from the full Poynting flux S_z = ExHy* - EyHx*.

    R_flux/T_flux are SIGNED and their out-of-band bins are noise ratios that flip sign about half the
    time -- see the module-level sign-convention note above (audit D-9). Mask with `band`."""
    freqs_Hz: np.ndarray
    R0: np.ndarray
    T0: np.ndarray
    R_flux: np.ndarray          # SIGNED (see the module note); mask with `band`
    T_flux: np.ndarray          # SIGNED (see the module note); mask with `band`
    band: np.ndarray
    r0: Optional[np.ndarray] = None   # COMPLEX co-pol 0-order reflection coeff, de-embedded to the front face
    t0: Optional[np.ndarray] = None   # COMPLEX co-pol 0-order transmission coeff, de-embedded across the cell


def _flux3d(ex, ey, hx, hy, dt):
    """Total time-averaged +z Poynting power per frequency, S_z = Re(Ex Hy* - Ey Hx*) summed over the
    whole (x,y) probe plane (Parseval: the real-space sum over the plane already includes every (kx,ky)
    diffraction order). Each probe array is (nsteps, nx, ny). Reduces to -Re(Ey Hx*) (the 2D _flux) when
    Ex = Hy = 0. The half-cell stagger is removed in the kernel; the half-TIMESTEP stagger is removed
    here by _half_step_H (audit D-2 -- see its docstring for the empirical sign verification)."""
    ph = _half_step_H(ex.shape[0], dt, 3)
    EX = np.fft.rfft(ex, axis=0); EY = np.fft.rfft(ey, axis=0)
    HX = np.fft.rfft(hx, axis=0) * ph; HY = np.fft.rfft(hy, axis=0) * ph
    S = np.real(EX * np.conj(HY) - EY * np.conj(HX))
    return np.sum(S, axis=(1, 2))                            # (nfreq,) signed z-power per frequency



@dataclass
class FDTD3DMOResult:
    """Broadband co/cross R/T + Faraday rotation of a 3D (laterally-uniform OR patterned) magneto-optic
    stack at normal incidence, input linearly polarized along `pol`."""
    freqs_Hz: np.ndarray
    band: np.ndarray
    t_co: np.ndarray
    t_cross: np.ndarray
    r_co: np.ndarray
    r_cross: np.ndarray
    R: np.ndarray
    T: np.ndarray
    faraday_deg: np.ndarray
