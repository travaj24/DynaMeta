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

import importlib.util as _importlib_util
import warnings
from dataclasses import dataclass
from typing import Callable, Optional, Union

import numpy as np


# Physical constants (SI): single source in core/constants.
from dynameta.constants import Q_E, EPS0, C_LIGHT, M_E  # noqa: F401

MassOrFn  = Union[float, Callable[[np.ndarray], np.ndarray]]
GammaOrFn = Union[float, Callable[[np.ndarray], np.ndarray]]

# Optional `refractiveindex` package (the refractiveindex.info database) -- detected lazily (free)
# and imported only on first use, mirroring the sibling Lumenairy library's glass.py pattern.
_REFRACTIVEINDEX_AVAILABLE = _importlib_util.find_spec("refractiveindex") is not None


def _get_refractiveindex_material():
    """Return the refractiveindex.RefractiveIndexMaterial class (raises a helpful error if the
    optional package is absent)."""
    if not _REFRACTIVEINDEX_AVAILABLE:
        raise ImportError(
            "RefractiveIndexInfoOptical requires the optional `refractiveindex` package: "
            "`pip install refractiveindex` (it bundles the refractiveindex.info database, so it "
            "works offline). Browse https://refractiveindex.info for the shelf/book/page of a "
            "material; or hand-enter n,k via TabulatedOptical / ConstantOptical instead.")
    from refractiveindex import RefractiveIndexMaterial
    return RefractiveIndexMaterial


class OpticalModel:
    """Base class. Subclasses implement eps(lambda_m, *, n_m3=None)."""

    def eps(self, lambda_m: float, *, n_m3=None):
        raise NotImplementedError


@dataclass
class ConstantOptical(OpticalModel):
    """Wavelength-independent complex eps (metals as fixed eps, simple dielectrics)."""
    value: complex

    def eps(self, lambda_m: float, *, n_m3=None):
        v = complex(self.value)
        return v if n_m3 is None else np.full(np.shape(n_m3), v, dtype=np.complex128)


@dataclass
class TabulatedOptical(OpticalModel):
    """Complex eps interpolated from tabulated (lambda, eps) data
    (e.g. measured n,k for a metal or substrate)."""
    lambda_m: np.ndarray
    eps_complex: np.ndarray

    def __post_init__(self) -> None:
        order = np.argsort(np.asarray(self.lambda_m, dtype=np.float64))
        self.lambda_m = np.asarray(self.lambda_m, dtype=np.float64)[order]
        self.eps_complex = np.asarray(self.eps_complex, dtype=np.complex128)[order]

    def eps(self, lambda_m, *, n_m3=None):
        lam = np.asarray(lambda_m, dtype=np.float64)
        lo, hi = float(self.lambda_m[0]), float(self.lambda_m[-1])
        if float(np.min(lam)) < lo * (1.0 - 1e-9) or float(np.max(lam)) > hi * (1.0 + 1e-9):
            # no SILENT extrapolation (np.interp clamps to the endpoints) -- raise like the
            # RefractiveIndexInfoOptical sibling; the table edge is not a valid eps off-band.
            raise ValueError(
                "TabulatedOptical.eps: wavelength outside the tabulated range [{:.4g}, {:.4g}] m -- "
                "no silent extrapolation (extend the table or use a dispersion model).".format(lo, hi))
        re = np.interp(lam, self.lambda_m, self.eps_complex.real)
        im = np.interp(lam, self.lambda_m, self.eps_complex.imag)
        v = complex(re, im) if lam.ndim == 0 else (re + 1j * im).astype(np.complex128)  # vectorized
        return v if n_m3 is None else np.full(np.shape(n_m3), v, dtype=np.complex128)


@dataclass
class RefractiveIndexInfoOptical(OpticalModel):
    """Complex eps from a refractiveindex.info entry, via the optional `refractiveindex` package
    (ported from the sibling Lumenairy library's glass.py lookup). Identify the entry by
    (shelf, book, page) -- browse https://refractiveindex.info (e.g. 'main','Au','Johnson' or
    'main','SiO2','Malitson'). eps = (n + i k)^2 with k >= 0, so Im(eps) = 2 n k >= 0 -- the
    exp(-i omega t) passive convention used throughout DynaMeta. Density-independent (n_m3 ignored).

    The RefractiveIndexMaterial is constructed once and cached on first eps() call (lazy import).
    An n-only (lossless) entry reports k = 0. Querying outside the entry's tabulated wavelength
    range raises (the index comes back non-finite). Use to_tabulated() to snapshot the entry onto a
    fixed wavelength grid as a portable, offline TabulatedOptical (no runtime dependency once
    frozen)."""
    shelf: str
    book: str
    page: str

    def _material(self):
        m = getattr(self, "_ri_mat", None)
        if m is None:
            m = _get_refractiveindex_material()(shelf=self.shelf, book=self.book, page=self.page)
            object.__setattr__(self, "_ri_mat", m)           # cache (not a dataclass field)
        return m

    def _nk(self, lambda_m: float):
        m = self._material()
        lam_nm = float(lambda_m) * 1e9
        # Enforce the entry's stated wavelength range -- a Sellmeier/formula entry would otherwise
        # SILENTLY EXTRAPOLATE to a physically-invalid value outside its fit range (a tabulated
        # entry returns NaN, caught below). No silent extrapolation (house anti-silent-failure rule).
        try:
            lo_nm, hi_nm = (float(b) for b in m.get_wl_range(unit="nm"))
        except Exception:
            lo_nm, hi_nm = None, None
        if lo_nm is not None and not (lo_nm <= lam_nm <= hi_nm):
            raise ValueError(
                "refractiveindex.info entry {}/{}/{}: wavelength {:.3f} nm is outside the entry's "
                "valid range [{:.1f}, {:.1f}] nm (no silent extrapolation)".format(
                    self.shelf, self.book, self.page, lam_nm, lo_nm, hi_nm))
        n = float(m.get_refractive_index(lam_nm, unit="nm"))
        if not np.isfinite(n):
            raise ValueError(
                "refractiveindex.info entry {}/{}/{} has no data at {:.3f} nm (outside the entry's "
                "tabulated wavelength range)".format(self.shelf, self.book, self.page, lam_nm))
        try:
            k = float(m.get_extinction_coefficient(lam_nm, unit="nm"))
            if not np.isfinite(k):
                k = 0.0
        except Exception:                                    # NoExtinctionCoefficient: lossless entry
            k = 0.0
        return n, k

    def eps(self, lambda_m: float, *, n_m3=None):
        n, k = self._nk(lambda_m)
        v = complex((n + 1j * k) ** 2)                       # Im(eps)=2nk>=0 (exp(-iwt) passive)
        return v if n_m3 is None else np.full(np.shape(n_m3), v, dtype=np.complex128)

    def to_tabulated(self, lambdas_m) -> "TabulatedOptical":
        """Snapshot the entry onto `lambdas_m` as a portable TabulatedOptical (offline; no runtime
        refractiveindex dependency once frozen -- useful for reproducible, shippable material data)."""
        lam = np.atleast_1d(np.asarray(lambdas_m, dtype=np.float64))
        eps = np.array([complex(self.eps(float(L))) for L in lam], dtype=np.complex128)
        return TabulatedOptical(lambda_m=lam, eps_complex=eps)


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
    # anti-silent-failure: least_squares returns its best iterate even on a failed/early-terminated
    # solve, and a parameter pinned at a bound usually means the model cannot fit the data (or the
    # bounds are wrong) -- surface both rather than hand back a quietly-bad Drude fit.
    if not sol.success:
        warnings.warn("fit_drude_params: least_squares did not converge (status={}: {}); the returned "
                      "Drude parameters are the best-so-far iterate and may be unreliable.".format(
                          sol.status, sol.message), RuntimeWarning, stacklevel=2)
    lo, hi = np.asarray(bounds[0], float), np.asarray(bounds[1], float)
    span = np.where((hi - lo) > 0, hi - lo, 1.0)
    pinned = [nm for nm, x, l, h, s in zip(("eps_inf", "m_eff_ratio", "gamma_rad_s"), sol.x, lo, hi, span)
              if min(x - l, h - x) < 1e-3 * s]
    if pinned:
        warnings.warn("fit_drude_params: parameter(s) {} pinned at a fit bound -- the Drude model likely "
                      "cannot represent the data (or the bounds are too tight); widen the bounds or "
                      "check the input n,k.".format(pinned), RuntimeWarning, stacklevel=2)
    eps_inf, m_ratio, gamma = sol.x
    return {
        "eps_inf":      float(eps_inf),
        "m_opt_kg":     float(m_ratio * M_E),
        "m_eff_ratio":  float(m_ratio),
        "gamma_rad_s":  float(gamma),
        "rms_residual": float(np.sqrt(np.mean(sol.fun ** 2))),
        "n_points":     int(len(n)),
    }
