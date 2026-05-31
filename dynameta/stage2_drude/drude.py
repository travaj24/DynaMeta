"""
Drude permittivity for free-carrier semiconductors.

  eps(n, lambda) = eps_inf - omega_p^2(n) / (omega^2 + i*omega*gamma(n))
  omega_p^2(n)   = n * e^2 / (eps0 * m_eff(n))

m_eff_kg and gamma_rad_s may be scalars (uniform) or callables of n
(per-bias spatial variation: Kane non-parabolic, Caughey-Thomas mobility).
"""

from __future__ import annotations

from typing import Callable, Union

import numpy as np


Q_E   = 1.602176634e-19
EPS0  = 8.8541878128e-12
C_LIGHT = 2.99792458e8
M_E   = 9.1093837015e-31


def drude_eps(*, n_m3: np.ndarray,
                lambda_m: float,
                eps_inf: float,
                m_eff_kg: Union[float, Callable[[np.ndarray], np.ndarray]],
                gamma_rad_s: Union[float, Callable[[np.ndarray], np.ndarray]],
                ) -> np.ndarray:
    """Compute complex eps at carrier density n_m3 (any shape) and wavelength
    lambda_m. Returns complex array of same shape as n_m3.
    """
    n_arr = np.asarray(n_m3, dtype=np.float64)
    m_eff = (np.asarray(m_eff_kg(n_arr), dtype=np.float64)
              if callable(m_eff_kg) else float(m_eff_kg))
    gamma = (np.asarray(gamma_rad_s(n_arr), dtype=np.float64)
              if callable(gamma_rad_s) else float(gamma_rad_s))
    omega = 2.0 * np.pi * C_LIGHT / float(lambda_m)
    omega_p2 = n_arr * Q_E * Q_E / (EPS0 * m_eff)
    eps = eps_inf - omega_p2 / (omega * omega + 1j * omega * gamma)
    return eps


def fit_drude_params(*, n_m3, lambda_m, eps_re, eps_im,
                     eps_inf0: float = 3.9,
                     m_eff_ratio0: float = 0.30,
                     gamma0: float = 1.0e14,
                     bounds=None) -> dict:
    """Least-squares fit of CONSTANT Drude params (eps_inf, m_eff, gamma) to
    measured / published complex permittivity samples.

    Use this to derive a `design.DrudeSpec` for a transparent-conducting
    oxide straight from literature eps data -- e.g. Park 2021 Fig. S2 gives
    Re/Im(eps_ITO) at carrier densities n = 3/4/5e20 cm^-3 and lambda = 1.3 um,
    from which this recovers eps_inf ~ 4.25, m*/m_e ~ 0.225, gamma ~ 1.1e14.
    The fitted scalars plug directly into DrudeSpec (constant-mass Drude) or
    serve as the n->0 baseline for a density-dependent model.

    The fit uses this module's own `drude_eps` as the forward model, so the
    sign convention is automatically consistent: Im(eps) > 0 for a passive
    lossy medium (exp(-i w t)), which is how Re/Im are tabulated/plotted in
    Park 2021 Fig. S2 -- pass `eps_im` as those positive values.

    Args (1D array-likes of equal length N; one (density, wavelength) sample
    each):
      n_m3, lambda_m, eps_re, eps_im
      eps_inf0, m_eff_ratio0, gamma0 : initial guesses (m_eff_ratio = m*/m_e)
      bounds : optional ([lo_eps_inf, lo_ratio, lo_gamma],
                          [hi_eps_inf, hi_ratio, hi_gamma]); sensible
                          TCO defaults if None.

    Returns dict: eps_inf, m_eff_kg, m_eff_ratio, gamma_rad_s, rms_residual,
    n_points.
    """
    from scipy.optimize import least_squares

    n = np.asarray(n_m3, dtype=np.float64).ravel()
    lam = np.asarray(lambda_m, dtype=np.float64).ravel()
    re_t = np.asarray(eps_re, dtype=np.float64).ravel()
    im_t = np.asarray(eps_im, dtype=np.float64).ravel()
    if not (len(n) == len(lam) == len(re_t) == len(im_t)):
        raise ValueError("n_m3, lambda_m, eps_re, eps_im must have equal length")

    def residual(p):
        eps_inf, m_ratio, gamma = p
        m_eff = m_ratio * M_E
        out = []
        for ni, li, ri, ii in zip(n, lam, re_t, im_t):
            e = complex(drude_eps(n_m3=ni, lambda_m=li, eps_inf=eps_inf,
                                   m_eff_kg=m_eff, gamma_rad_s=gamma))
            out.append(e.real - ri)
            out.append(e.imag - ii)
        return np.asarray(out)

    if bounds is None:
        bounds = ([1.0, 0.05, 1.0e13], [8.0, 1.0, 1.0e15])
    sol = least_squares(residual, [eps_inf0, m_eff_ratio0, gamma0], bounds=bounds)
    eps_inf, m_ratio, gamma = sol.x
    return {
        "eps_inf":      float(eps_inf),
        "m_eff_kg":     float(m_ratio * M_E),
        "m_eff_ratio":  float(m_ratio),
        "gamma_rad_s":  float(gamma),
        "rms_residual": float(np.sqrt(np.mean(sol.fun ** 2))),
        "n_points":     int(len(n)),
    }
