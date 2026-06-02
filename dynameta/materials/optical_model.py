"""
Optical dispersion models: lambda (and optionally carrier density n) -> complex eps.

These feed Stage 2/3 (the optical side). They are deliberately SEPARATE from
TransportModel (the Stage-1 DC side): in non-parabolic conductors the optical
(conductivity) effective mass differs from the DOS mass, and the static
(Poisson) permittivity differs from the optical eps_inf. Keeping them in
distinct objects stops the two from being conflated (the old DrudeSpec carried
both and strained under an `optical_mass_fn()` hack).

Sign convention: exp(-i*omega*t) (NGSolve's convention). A passive lossy
medium has Im(eps) > 0. Park 2021 Fig. S2 tabulates Re/Im(eps) this way.

Every OpticalModel exposes:
    eps(lambda_m, *, n_m3=None) -> complex | np.ndarray
For density-independent models (metals, dielectrics) n_m3 is ignored.
For free-carrier models (Drude) n_m3 is required and the result has the
shape of n_m3.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Union

import numpy as np


# Physical constants (SI): single source in core/constants.
from dynameta.constants import Q_E, EPS0, C_LIGHT, M_E  # noqa: F401

MassOrFn  = Union[float, Callable[[np.ndarray], np.ndarray]]
GammaOrFn = Union[float, Callable[[np.ndarray], np.ndarray]]


class OpticalModel:
    """Base class. Subclasses implement eps(lambda_m, *, n_m3=None)."""
    requires_density: bool = False

    def eps(self, lambda_m: float, *, n_m3=None):
        raise NotImplementedError


@dataclass
class ConstantOptical(OpticalModel):
    """Wavelength-independent complex eps (metals as fixed eps, simple dielectrics)."""
    value: complex
    requires_density: bool = False

    def eps(self, lambda_m: float, *, n_m3=None):
        v = complex(self.value)
        return v if n_m3 is None else np.full(np.shape(n_m3), v, dtype=np.complex128)


@dataclass
class TabulatedOptical(OpticalModel):
    """Complex eps interpolated from tabulated (lambda, eps) data
    (e.g. measured n,k for a metal or substrate)."""
    lambda_m: np.ndarray
    eps_complex: np.ndarray
    requires_density: bool = False

    def __post_init__(self) -> None:
        order = np.argsort(np.asarray(self.lambda_m, dtype=np.float64))
        self.lambda_m = np.asarray(self.lambda_m, dtype=np.float64)[order]
        self.eps_complex = np.asarray(self.eps_complex, dtype=np.complex128)[order]

    def eps(self, lambda_m: float, *, n_m3=None):
        re = np.interp(lambda_m, self.lambda_m, self.eps_complex.real)
        im = np.interp(lambda_m, self.lambda_m, self.eps_complex.imag)
        v = complex(re, im)
        return v if n_m3 is None else np.full(np.shape(n_m3), v, dtype=np.complex128)


@dataclass
class DrudeOptical(OpticalModel):
    """Free-carrier Drude dispersion:

        eps(n, lambda) = eps_inf - omega_p^2(n) / (omega^2 + i*omega*gamma(n))
        omega_p^2(n)   = n * e^2 / (eps0 * m_opt(n))

    m_opt_kg is the OPTICAL (conductivity) effective mass -- NOT the DOS mass
    used by TransportModel. Both m_opt_kg and gamma_rad_s may be scalars or
    callables of n (per-bias spatial variation).
    """
    eps_inf:     float
    m_opt_kg:    MassOrFn
    gamma_rad_s: GammaOrFn
    requires_density: bool = True

    def eps(self, lambda_m: float, *, n_m3=None):
        if n_m3 is None:
            raise ValueError("DrudeOptical.eps requires n_m3 (carrier density).")
        n = np.asarray(n_m3, dtype=np.float64)
        m = (np.asarray(self.m_opt_kg(n), dtype=np.float64)
              if callable(self.m_opt_kg) else float(self.m_opt_kg))
        g = (np.asarray(self.gamma_rad_s(n), dtype=np.float64)
              if callable(self.gamma_rad_s) else float(self.gamma_rad_s))
        omega = 2.0 * np.pi * C_LIGHT / float(lambda_m)
        omega_p2 = n * Q_E * Q_E / (EPS0 * m)
        return self.eps_inf - omega_p2 / (omega * omega + 1j * omega * g)


def fit_drude_params(*, n_m3, lambda_m, eps_re, eps_im,
                       eps_inf0: float = 3.9,
                       m_eff_ratio0: float = 0.30,
                       gamma0: float = 1.0e14,
                       bounds=None) -> dict:
    """Least-squares fit of CONSTANT optical Drude params (eps_inf, m_opt, gamma)
    to measured / published complex permittivity samples.

    Recovers, e.g. from Park 2021 Fig. S2, eps_inf ~ 4.25, m*/m_e ~ 0.225,
    gamma ~ 1.1e14. The fitted scalars plug straight into a DrudeOptical.
    Sign convention matches DrudeOptical (Im(eps) > 0 for passive loss).

    Returns dict: eps_inf, m_opt_kg, m_eff_ratio, gamma_rad_s, rms_residual, n_points.
    """
    from scipy.optimize import least_squares

    n   = np.asarray(n_m3,     dtype=np.float64).ravel()
    lam = np.asarray(lambda_m, dtype=np.float64).ravel()
    re_t = np.asarray(eps_re,  dtype=np.float64).ravel()
    im_t = np.asarray(eps_im,  dtype=np.float64).ravel()
    if not (len(n) == len(lam) == len(re_t) == len(im_t)):
        raise ValueError("n_m3, lambda_m, eps_re, eps_im must have equal length")

    def residual(p):
        eps_inf, m_ratio, gamma = p
        model = DrudeOptical(eps_inf=eps_inf, m_opt_kg=m_ratio * M_E,
                               gamma_rad_s=gamma)
        out = []
        for ni, li, ri, ii in zip(n, lam, re_t, im_t):
            e = complex(model.eps(li, n_m3=ni))
            out.append(e.real - ri)
            out.append(e.imag - ii)
        return np.asarray(out)

    if bounds is None:
        bounds = ([1.0, 0.05, 1.0e13], [8.0, 1.0, 1.0e15])
    sol = least_squares(residual, [eps_inf0, m_eff_ratio0, gamma0], bounds=bounds)
    eps_inf, m_ratio, gamma = sol.x
    return {
        "eps_inf":      float(eps_inf),
        "m_opt_kg":     float(m_ratio * M_E),
        "m_eff_ratio":  float(m_ratio),
        "gamma_rad_s":  float(gamma),
        "rms_residual": float(np.sqrt(np.mean(sol.fun ** 2))),
        "n_points":     int(len(n)),
    }
