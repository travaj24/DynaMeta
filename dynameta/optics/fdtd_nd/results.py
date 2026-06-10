"""Result dataclasses and flux post-processing shared by the solve front-ends.

Split from the former monolithic fdtd_nd.py; see the package __init__ docstring
for conventions. Bodies are verbatim from the original module.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np



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




def _flux(ey, hx):
    """Per-frequency time-averaged +z Poynting power S_z = -Re(E_y H_x*) summed over x, from the rfft
    of the recorded probe x-lines (shape (nsteps, nx)). Half-cell / half-step staggering offsets are
    common to numerator and the incident reference, so they cancel in the R/T ratio."""
    Ey = np.fft.rfft(ey, axis=0)
    Hx = np.fft.rfft(hx, axis=0)
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
