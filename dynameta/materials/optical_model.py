"""
Optical dispersion models: lambda (and optionally carrier density n) -> complex eps.

These feed Stage 2/3 (the optical side). They are deliberately SEPARATE from
TransportModel (the Stage-1 DC side): in non-parabolic conductors the optical
(conductivity) effective mass differs from the DOS mass, and the static
(Poisson) permittivity differs from the optical eps_inf. Keeping them in
distinct objects stops the two from being conflated (the old DrudeSpec carried
both and strained under an `optical_mass_fn()` hack).

Sign convention: exp(-i*omega*t) (NGSolve's convention). A passive lossy
medium has Im(eps) > 0. Near-IR ellipsometry of ITO tabulates Re/Im(eps) this way.

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
from typing import Callable, Union

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
        if np.any(np.asarray(m) <= 0.0) or not np.all(np.isfinite(m)):   # a callable must return m_opt > 0
            raise ValueError("DrudeOptical.eps: m_opt_kg must be finite and > 0 (omega_p^2 ~ 1/m would "
                             "otherwise be inf/NaN); a callable m_opt_kg(n) returned a non-positive value.")
        # gamma = 0 (collisionless/lossless idealization) is legitimate; NEGATIVE gamma flips
        # Im(eps) < 0 = GAIN under exp(-i omega t) -- reject it loudly instead of silently amplifying.
        if np.any(np.asarray(g) < 0.0) or not np.all(np.isfinite(g)):
            raise ValueError("DrudeOptical.eps: gamma_rad_s must be finite and >= 0 (a negative damping "
                             "gives Im(eps) < 0 = gain under exp(-i omega t)); a callable gamma_rad_s(n) "
                             "returned a negative/non-finite value.")
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

    Recovers representative near-IR ITO values eps_inf ~ 4.25, m*/m_e ~ 0.225,
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


# ===========================================================================
# Roadmap 2.3 -- Mermin / extended-Drude damping (frequency-dependent scattering).
#
# Real TCO (ITO) free-carrier scattering is omega-dependent: ionized-impurity
# scattering is LESS effective at optical frequencies than at DC (optical
# mobility > DC mobility -- Fujiwara & Kondo, Phys. Rev. B 71, 075109 (2005)),
# and LO-phonon scattering adds a near-constant floor. The plain DrudeOptical
# above carries a single scalar gamma; the classes here add the gamma(omega)
# tier while staying byte-identical to plain Drude when gamma is constant.
#
# ---------------------------------------------------------------------------
# The k -> 0 (local) Mermin analysis -- LOAD-BEARING, so it is spelled out here.
#
# The number-conserving Mermin dielectric function (N. D. Mermin, Phys. Rev. B
# 1, 2362 (1970), Eq. 8) corrects the relaxation-time (Drude) approximation so
# that LOCAL charge is conserved. With collision frequency gamma and the
# Lindhard (RPA) function eps_L(k, omega):
#
#     eps_M(k,w) = 1 + (1 + i g/w) [eps_L(k, w+i g) - 1]
#                  --------------------------------------------------
#                  1 + (i g/w) [eps_L(k, w+i g) - 1] / [eps_L(k, 0) - 1]
#
# In the long-wavelength (k -> 0, i.e. LOCAL / optical) limit the Lindhard
# function has the two limits
#     eps_L(k->0, w) - 1  ->  -wp^2 / w^2          (FINITE)
#     eps_L(k->0, 0) - 1  ->  +k_TF^2 / k^2 -> +oo (DIVERGES; Thomas-Fermi)
# so the correction ratio [eps_L(k,w+ig)-1]/[eps_L(k,0)-1] -> 0 and the Mermin
# denominator -> 1. The numerator -> (1 + i g/w)(-wp^2/(w+i g)^2), hence
#
#     eps_M(k->0, w) - 1  ->  (1 + i g/w) ( -wp^2 / (w+i g)^2 )
#                          =  -(w+i g)/w * wp^2/(w+i g)^2
#                          =  -wp^2 / ( w (w + i g) )
#                          =  -wp^2 / ( w^2 + i w g ).
#
# That is EXACTLY the plain-Drude free-carrier term (compare DrudeOptical.eps:
# eps_inf - wp^2/(w^2 + i w gamma)). CONCLUSION: the local (k->0) Mermin
# dielectric of a Drude plasma is IDENTICAL to plain Drude with the same gamma.
# The number-conserving correction is a no-op at k = 0. It therefore matters
# ONLY (a) at FINITE k (spatial dispersion / nonlocality -- that is roadmap 2.4,
# the hydrodynamic / nonlocal-TMM tier), or (b) through a NONTRIVIAL gamma(omega).
# tests/test_mermin.py verifies this collapse numerically against a finite-k
# hydrodynamic Lindhard chi (residual ~ (beta k / w)^2 -> 0).
#
# DELIVERABLE (honest scoping): the useful LOCAL knob is a frequency-dependent
# gamma(omega) -- ExtendedDrudeOptical -- plus the representative TCO preset
# gamma_ito_extended and the causality diagnostic check_kk. MerminDrudeOptical
# is provided as a named, literature-citable entry point whose LOCAL (k=0)
# evaluation equals ExtendedDrudeOptical exactly (per the proof above); its
# finite-k number-conserving branch is DEFERRED to roadmap 2.4.
# ===========================================================================

# gamma(omega) spec: a scalar (constant -> plain Drude), a callable omega->gamma
# (rad/s -> rad/s), or a (omega_table, gamma_table) pair interpolated in omega.
GammaOmegaSpec = Union[float, Callable[[np.ndarray], np.ndarray], tuple]


def _resolve_gamma_omega(spec) -> Callable[[np.ndarray], np.ndarray]:
    """Normalize a gamma(omega) spec into a callable omega_rad_s -> gamma_rad_s.

    * callable            -> used directly (arbitrary phenomenological model);
    * (omega_tab, g_tab)  -> linear interpolation in omega, NO silent extrapolation
                             (raises off-table, mirroring TabulatedOptical);
    * scalar              -> a constant callable (the plain-Drude limit).
    """
    if callable(spec):
        return spec
    if isinstance(spec, tuple):
        if len(spec) != 2:
            raise ValueError("ExtendedDrudeOptical gamma_omega tuple must be "
                             "(omega_table, gamma_table).")
        w_tab = np.asarray(spec[0], dtype=np.float64).ravel()
        g_tab = np.asarray(spec[1], dtype=np.float64).ravel()
        if w_tab.size != g_tab.size or w_tab.size < 2:
            raise ValueError("ExtendedDrudeOptical gamma_omega tuple: omega_table and gamma_table "
                             "must have equal length >= 2.")
        order = np.argsort(w_tab)
        w_tab, g_tab = w_tab[order], g_tab[order]

        def _interp(omega):
            om = np.asarray(omega, dtype=np.float64)
            lo, hi = float(w_tab[0]), float(w_tab[-1])
            if float(np.min(om)) < lo * (1.0 - 1e-9) or float(np.max(om)) > hi * (1.0 + 1e-9):
                # no SILENT extrapolation (house anti-silent-failure rule); a KK check on a WIDE
                # grid needs a callable model (e.g. gamma_ito_extended) or a wide-enough table.
                raise ValueError(
                    "ExtendedDrudeOptical gamma table: omega outside the tabulated range "
                    "[{:.4g}, {:.4g}] rad/s -- no silent extrapolation.".format(lo, hi))
            return np.interp(om, w_tab, g_tab)
        return _interp
    g0 = float(spec)
    return lambda omega: g0


@dataclass
class ExtendedDrudeOptical(OpticalModel):
    """Free-carrier Drude dispersion with a FREQUENCY-DEPENDENT damping gamma(omega):

        eps(n, lambda) = eps_inf - omega_p^2(n) / (omega^2 + i*omega*gamma(omega))
        omega_p^2(n)   = n * e^2 / (eps0 * m_opt(n))

    This is the phenomenological extended-Drude tier for real TCOs: ionized-impurity
    scattering falls off toward optical frequencies while an LO-phonon floor remains
    (see gamma_ito_extended). Same sign convention as DrudeOptical (exp(-i omega t),
    Im(eps) >= 0 for passive loss). m_opt_kg is the OPTICAL (conductivity) mass and
    may be a scalar or a callable of n (as in DrudeOptical).

    gamma_omega may be:
      * a scalar          -> a constant gamma; then eps() is BYTE-IDENTICAL to
                             DrudeOptical(eps_inf, m_opt_kg, gamma_rad_s=scalar)
                             (this class delegates the arithmetic to DrudeOptical);
      * a callable omega_rad_s -> gamma_rad_s (e.g. gamma_ito_extended);
      * a (omega_table, gamma_table) pair, interpolated in omega (no extrapolation).

    CAUSALITY: an arbitrary gamma(omega) inserted into the Drude form is NOT guaranteed
    to yield a Kramers-Kronig-causal eps. Use check_kk(model, omega_grid) to get a KK
    residual diagnostic (it reports; it does not enforce). Measured with the finding-Q-5
    quadrature (see check_kk's operating-envelope table): plain Drude sits at rms_norm
    ~2.5e-4 on the shipped grid and FALLS with refinement (causal); the ITO preset holds a
    grid-independent ~1.1e-2 -- comparable to a 2x jump in gamma(omega) -- because the
    omega^1.5 impurity crossover is non-analytic at omega = 0, so it is "nearly causal" and
    NOT in the causal class; a discontinuous gamma(omega) is worse still.
    """
    eps_inf:     float
    m_opt_kg:    MassOrFn
    gamma_omega: GammaOmegaSpec

    def __post_init__(self) -> None:
        self._gamma_fn = _resolve_gamma_omega(self.gamma_omega)   # not a dataclass field

    def gamma_at(self, omega_rad_s):
        """The damping gamma (rad/s) at angular frequency omega_rad_s (scalar or array)."""
        return self._gamma_fn(np.asarray(omega_rad_s, dtype=np.float64))

    def eps(self, lambda_m: float, *, n_m3=None):
        omega = 2.0 * np.pi * C_LIGHT / float(lambda_m)
        g = self._gamma_fn(omega)
        g = float(g) if np.ndim(g) == 0 else np.asarray(g, dtype=np.float64)
        # Delegate to DrudeOptical so the arithmetic (and the m_opt/gamma sign guards) is the
        # SAME code path -- guarantees byte-identity with plain Drude when gamma(omega) is const.
        return DrudeOptical(eps_inf=self.eps_inf, m_opt_kg=self.m_opt_kg,
                            gamma_rad_s=g).eps(lambda_m, n_m3=n_m3)


@dataclass
class MerminDrudeOptical(OpticalModel):
    """Number-conserving Mermin dielectric of a Drude plasma (Mermin, Phys. Rev. B 1,
    2362 (1970)) -- LOCAL (k -> 0) limit.

    As proved analytically in the module header (and verified numerically in
    tests/test_mermin.py against a finite-k hydrodynamic Lindhard chi), the k -> 0
    Mermin dielectric reduces EXACTLY to plain Drude with the same gamma. The
    number-conserving correction is identically zero at k = 0; it only bites at FINITE
    k (spatial dispersion) or through a nontrivial gamma(omega). Consequently this class
    is, at k = 0, EXACTLY ExtendedDrudeOptical(eps_inf, m_opt_kg, gamma_omega) -- it is a
    named, literature-citable entry point, NOT an additional local correction.

    The finite-k number-conserving Mermin branch (k_per_m != 0), which DOES differ from
    Drude, is DEFERRED to roadmap 2.4 (hydrodynamic / nonlocal layered TMM) and raises
    NotImplementedError here rather than silently returning the local (Drude) result.
    """
    eps_inf:     float
    m_opt_kg:    MassOrFn
    gamma_omega: GammaOmegaSpec
    k_per_m:     float = 0.0

    def __post_init__(self) -> None:
        self._local = ExtendedDrudeOptical(self.eps_inf, self.m_opt_kg, self.gamma_omega)

    def eps(self, lambda_m: float, *, n_m3=None):
        if self.k_per_m != 0.0:
            raise NotImplementedError(
                "MerminDrudeOptical: finite-k (k_per_m={!r}) number-conserving Mermin is DEFERRED "
                "to roadmap 2.4 (hydrodynamic / nonlocal TMM); only the LOCAL k=0 limit -- which "
                "equals plain Drude with the supplied gamma(omega) -- is implemented here.".format(
                    self.k_per_m))
        return self._local.eps(lambda_m, n_m3=n_m3)


def gamma_ito_extended(omega_rad_s, *, gamma_dc_rad_s: float = 1.5e14,
                       gamma_inf_rad_s: float = 4.0e13, omega_c_rad_s: float = 1.0e15,
                       p: float = 1.5):
    """Representative ITO extended-Drude damping gamma(omega) [rad/s], a smooth
    ionized-impurity + phonon crossover:

        gamma(omega) = gamma_inf + (gamma_dc - gamma_inf) / (1 + (omega/omega_c)^p)

    The ionized-impurity channel falls off with the standard high-frequency
    omega^(-3/2) trend (p = 3/2; Gerlach/Pisarkiewicz-type optical scattering), so
    gamma DROPS from the DC/low-frequency impurity-dominated value gamma_dc toward the
    LO-phonon floor gamma_inf across the near-IR. Physical consequence pinned in
    tests/test_mermin.py: relative to a plain Drude fixed at the DC damping gamma_dc,
    the extended model has SMALLER gamma in the near-IR/below-plasma window, hence
    REDUCED absorption (Im eps) -- the "optical mobility > DC mobility" effect of the
    TCO extended-Drude literature (Fujiwara & Kondo, Phys. Rev. B 71, 075109 (2005);
    Mendelsberg et al., J. Appl. Phys. 111, 063515 (2012)).

    Defaults are representative near-IR ITO values (gamma_dc ~ 1.5e14, gamma_inf ~ 4e13,
    omega_c ~ 1e15 rad/s). Pass as ExtendedDrudeOptical(gamma_omega=gamma_ito_extended)
    for the defaults, or wrap in a lambda/partial to retune. Callable of omega (scalar or
    array); pure numpy, SI.
    """
    omega = np.asarray(omega_rad_s, dtype=np.float64)
    if omega_c_rad_s <= 0.0:
        raise ValueError("gamma_ito_extended: omega_c_rad_s must be > 0.")
    return gamma_inf_rad_s + (gamma_dc_rad_s - gamma_inf_rad_s) / (1.0 + (omega / omega_c_rad_s) ** p)


def _maclaurin_re_from_im(omega, im_chi, h, *, edge_correct: bool = True, chunk: int = 0):
    """Discrete Kramers-Kronig (Maclaurin method, Ohta & Ishida, Appl. Spectrosc. 42, 952
    (1988)): reconstruct Re(chi)(omega_j) from Im(chi) on a UNIFORM grid by summing only the
    OPPOSITE-parity points (which skips the omega'=omega_j singularity of the principal value):

        Re(chi)(w_j) = (4 h / pi) * sum_{k: (k-j) odd} w_k Im(chi)(w_k) / (w_k^2 - w_j^2).

    Sign is the exp(-i omega t) convention (Im(eps) >= 0 for absorption; Landau-Lifshitz form).

    ENDPOINT CONSISTENCY (``edge_correct``, finding Q-5).  The sum is a MIDPOINT rule with cells of
    width ``2h`` centred on each summed point, so the two parities of ``j`` integrate DIFFERENT
    domains: for odd ``j`` the summed set starts at ``k = 0`` and its first cell reaches down to
    ``w_0 - h``, while for even ``j`` it starts at ``k = 1`` and the strip ``[w_0 - h, w_1 - h]``
    is simply MISSING.  The integrand there is ``p(w) = w Im(chi)(w)``, which for a Drude metal
    tends to the FINITE, LARGE value ``wp^2 / gamma`` as ``w -> 0``, so the omission is not small:
    measured on the shipped grid, the odd-``j`` points reproduce ``Re(eps)`` to ~1e-4 (exact to the
    grid's own truncation) while the even-``j`` points are wrong by ~0.9 -- a pure PARITY ARTEFACT
    that was the whole of the diagnostic's "causal floor" (and why the residual scaled as O(h) and
    why the Drude DC pole was its worst case).  The correction adds each parity's own missing
    low-edge strip ``[0, w_klo - h]`` (and, symmetrically, the missing high-edge strip) as a
    midpoint contribution with ``p`` linearly extrapolated from the first/last two grid points.
    Measured effect on a causal Drude at N = 8000: ``rms_norm`` 1.97e-2 -> 2.48e-4 (80x), and the
    residual then converges at ~O(h^2) instead of O(h).

    ``chunk`` only sets the row-block size of the vectorized evaluation (memory vs speed);
    ``chunk = 0`` (the default) sizes it to keep the working set near 100 MB.
    """
    omega = np.asarray(omega, dtype=np.float64)
    im_chi = np.asarray(im_chi, dtype=np.float64)
    N = omega.size
    h = float(h)
    idx = np.arange(N)
    w2 = omega * omega
    p = omega * im_chi                                  # p(w) = w Im(chi)(w)
    pref = 4.0 * h / np.pi
    re = np.empty(N, dtype=np.float64)
    step = int(chunk) if int(chunk) > 0 else max(32, min(512, 4_000_000 // max(1, N)))
    # PERF (audit P-4): the opposite-parity mask takes only TWO distinct values over the whole
    # sum -- the ODD columns for an even ``j``, the EVEN columns for an odd ``j`` -- so the rows
    # are processed ONE PARITY AT A TIME against a single broadcast (1,N) column mask.  That
    # removes, per block, the int64 ``idx - j`` precursor, the ``& 1``/``== 1`` passes, the
    # boolean mask itself and both ``np.where`` temporaries, leaving one broadcast subtract, one
    # divide, one masked zero-fill and the SAME per-row reduction, all in one reused buffer.
    # BYTE-IDENTICAL by construction: each row holds exactly the values the mask/where pair
    # produced (the same-parity columns are zeroed after the divide instead of being divided by
    # 1.0 and then discarded), and every row is still summed on its own contiguous (N,) block,
    # i.e. through the identical pairwise reduction.  ``errstate`` only silences the 1/0 on the
    # zeroed diagonal, which the pre-P-4 form divided by 1.0 instead.
    zero_cols = ((idx & 1) == 0, (idx & 1) == 1)        # columns to ZERO for j even / j odd
    buf = np.empty((min(step, max(N, 1)), N), dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        for par in (0, 1):
            w2_par = w2[par::2]                         # the j of this parity (strided view)
            re_par = re[par::2]                         # write-through view into `re`
            zc = zero_cols[par][None, :]
            for a in range(0, w2_par.size, step):
                b = min(a + step, w2_par.size)
                blk = buf[:b - a]
                np.subtract(w2[None, :], w2_par[a:b, None], out=blk)        # den
                np.divide(p[None, :], blk, out=blk)                         # p / den
                np.copyto(blk, 0.0, where=zc)                               # same-parity -> 0
                re_par[a:b] = pref * blk.sum(axis=1)
    if not edge_correct or N < 4:
        return re

    # --- both parities must integrate the SAME domain ---------------------------------------
    # a_lo is SIGNED: > 0 means the strip [0, a_lo) is MISSING (add it); < 0 means the first cell
    # reached below omega = 0 and, since p(w) = w Im(w) is EVEN, the strip [0, |a_lo|) was counted
    # TWICE (subtract it).  a_lo == 0 is the exactly-covered case (grids starting at omega_0 = h).
    slope_lo = (p[1] - p[0]) / (omega[1] - omega[0])
    k_lo = np.where((idx & 1) == 1, 0, 1)               # lowest summed index for each j
    a_lo = omega[k_lo] - h
    mid_lo = 0.5 * np.abs(a_lo)
    p_lo = p[0] + slope_lo * (mid_lo - omega[0])        # linear extrapolation of p toward DC
    den_lo = mid_lo * mid_lo - w2
    corr_lo = np.where((a_lo != 0.0) & (den_lo != 0.0),
                       (2.0 / np.pi) * a_lo * p_lo / np.where(den_lo != 0.0, den_lo, 1.0), 0.0)

    last = N - 1
    k_hi = np.where(((last - idx) & 1) == 1, last, last - 1)       # highest summed index
    b_hi = omega[k_hi] + h
    w_top = omega[last] + h
    wid_hi = np.maximum(0.0, w_top - b_hi)              # uncovered strip [b_hi, w_top)
    mid_hi = 0.5 * (b_hi + w_top)
    den_hi = mid_hi * mid_hi - w2
    corr_hi = np.where((wid_hi > 0.0) & (den_hi != 0.0),
                       (2.0 / np.pi) * wid_hi * p[last] / np.where(den_hi != 0.0, den_hi, 1.0),
                       0.0)
    return re + corr_lo + corr_hi


def check_kk(model, omega_grid, *, n_m3=None, metric_band=None, dc_skip: int = 2,
             edge_correct: bool = True, self_calib: bool = True) -> dict:
    """Kramers-Kronig causality DIAGNOSTIC for a frequency-domain OpticalModel (roadmap 2.3).

    An arbitrary gamma(omega) in the Drude form need not be causal. This reconstructs
    Re(eps) from Im(eps) via the Maclaurin discrete KK transform on a WIDE UNIFORM omega
    grid and compares it to the model's OWN Re(eps). It REPORTS a residual; it does NOT
    enforce (a caller decides). The diagnostic discriminates: the residual is SMALL for a
    causal model (plain Drude, a Lorentz oscillator, the smooth ITO extended-Drude preset)
    and LARGE for an acausal eps (a discontinuous gamma(omega); a sign-flipped Im).

    QUADRATURE (finding Q-5).  The bare Maclaurin sum was only FIRST-order accurate here: on a
    causal Drude its normalized residual scaled as O(h), i.e. the old "causal floor" was pure
    quadrature error carrying NO causality information, and the Drude DC pole was its worst case
    (a manifestly causal 2-oscillator Lorentz scored ~330x BELOW the shipped "causal reference"
    on the same grid, so the reference itself destroyed the discrimination). The cause was a
    PARITY ARTEFACT at the low-frequency edge -- see :func:`_maclaurin_re_from_im`, which now makes
    both parities integrate the same domain. Measured on a causal Drude at N = 8000 the residual
    drops 80x and then converges at ~O(h^2), so the causal floor FALLS with grid refinement while
    a genuine acausality does not: the causal/acausal ordering is now grid-independent, and no
    external "reference model" is needed.

    Args:
      model       : any OpticalModel (eps(lambda_m, n_m3=...) contract).
      omega_grid  : 1-D UNIFORM, strictly-positive angular-frequency grid (rad/s). It must
                    be WIDE (extend well above the plasma edge) so the KK integral converges;
                    omega=0 is excluded (it is the Drude DC conductivity pole).
      n_m3        : carrier density passed through to eps (required by Drude-family models).
      metric_band : optional (omega_lo, omega_hi) rad/s over which the residual is summarized;
                    default = auto (the dispersive band around the Re(eps) zero crossing, with
                    the DC-pole edge skipped).
      dc_skip     : grid points skipped at the low-omega edge when auto-picking the band
                    (the DC pole is under-resolved on a uniform grid).
      edge_correct: endpoint-consistent Maclaurin variant (default True). False reproduces the
                    legacy parity-asymmetric sum -- kept ONLY to exhibit the old O(h) floor.
      self_calib  : also run the transform on the step-DOUBLED sub-grid (``omega[::2]``) to get a
                    data-driven estimate of the remaining quadrature error (default True; set
                    False to save ~25% of the runtime when only ``rms_norm`` is wanted).

    Returns a dict with arrays (omega, re_model, re_kk, residual, band_mask), eps_inf_est,
    the normalization scale, and the headline diagnostics max_norm / rms_norm (residual over
    the band, normalized by the dispersive Re swing) plus raw max_abs / rms_abs.

    HOW TO READ THE RESULT (finding Q-5).  ``rms_norm`` is the headline: a causal model's residual
    FALLS with grid refinement (~O(h^2)) while a genuine acausality holds a fixed residual, so
    REFINING the grid can only widen the separation. There is no need for an external "causal
    reference model" (the shipped gate used to compare against plain Drude, the single worst causal
    reference there is).

    RESOLUTION ENVELOPE (fix-verify W1 item 5 -- the claim above holds only INSIDE it).  Grid
    refinement can never invert the ranking, but that is a statement about refining FROM a resolved
    grid; it is NOT a licence to compare residuals on a COARSE one, and the ranking really does
    invert below about N = 6000 on the reference grid.  The Maclaurin quadrature has to RESOLVE the
    sharpest feature of Im(eps): a Lorentz line of FWHM ``gamma`` needs ``gamma / h`` grid points
    across it, and under-resolving it inflates a perfectly CAUSAL model's residual.  Measured on
    the reference grid (``wmax = 80 wp``, band [0.4 wp, 5 wp]) with the 2-oscillator Lorentz whose
    narrowest line is ``gamma = 1e14``:

      N        h          gamma/h   lorentz2 rms_norm   worst acausal separation
      2000     8.53e13    1.17      4.56e-2             NONE -- the causal Lorentz outscores the
                                                        10x gamma jump (4.66e-2)
      4000     4.27e13    2.35      7.79e-3             NONE -- level with the 2x jump (7.93e-3)
      6000     2.84e13    3.52      1.11e-3             7.2x on the 2x jump
      8000     2.13e13    4.69      1.76e-4             39x   (and the drude floor takes over)
      16000    1.07e13    9.38      1.92e-5             150x

    So: keep ``gamma_min / h >= ~5`` for every line in the model.  ``feature_pts`` in the returned
    dict is a MODEL-FREE estimate of that ratio (the sharpest curvature of the sampled Im(eps)
    inside the band, read as a Lorentzian width in grid points), and a ``RuntimeWarning`` fires
    below ``_KK_MIN_FEATURE_PTS``.  A discontinuous ``gamma(omega)`` is of course never resolved
    (``feature_pts`` pinned near 3-5 at every N) -- that is itself the tell, and its ``rms_norm``
    stays flat under refinement while a resolved causal model's falls.

    Secondary keys:
      h, h_rel         : the grid step [rad/s] and h / (omega[-1] - omega[0]).
      quad_floor_norm  : rms over the band of the difference between the step-``h`` and step-
                         ``2h`` reconstructions, normalized by the same scale -- a data-driven
                         estimate of the grid's own quadrature error. ``nan`` when
                         ``self_calib=False``.
      causality_ratio  : ``rms_norm / quad_floor_norm``, i.e. how far the residual sits above the
                         grid's quadrature error. A HINT, not a gate: once the residual is
                         dominated by the finite-grid ``eps_inf_est`` offset rather than by
                         quadrature (which happens quickly for a well-resolved Lorentz), the
                         ratio saturates and can exceed a genuinely acausal model's. Gate on
                         ``rms_norm``.
      rms_norm_per_h   : ``rms_norm / h_rel`` -- the legacy O(h) floor indicator, reported so a
                         caller can see at a glance whether a residual is still quadrature-limited
                         (roughly grid-INVARIANT for an O(h)-limited residual; it FALLS with
                         refinement now, from ~158 to ~2.0 on a Drude at N = 8000).

    OPERATING ENVELOPE (measured; grid ``linspace(wmax/N, wmax, N)``, ``wmax = 80 wp``, band
    [0.4 wp, 5 wp], ITO-like Drude).  ``rms_norm``:

      model                        N = 8000     N = 32000    verdict
      plain Drude                  2.48e-4      2.95e-5      causal (falls ~O(h^2))
      2-oscillator Lorentz         7.75e-4      1.47e-5      causal (falls)
      gamma_ito_extended preset    1.09e-2      1.07e-2      NOT in the causal class: a genuine,
                                                             grid-independent ~1% residual (the
                                                             omega^1.5 crossover is non-analytic
                                                             at omega = 0)
      2x jump in gamma(omega)      9.82e-3      9.57e-3      acausal (flat)
      10x jump in gamma(omega)     2.32e-2      1.72e-2      acausal (flat)
      Lorentz with Im sign flipped 6.01e-1      6.00e-1      acausal (flat)

    So a 2x discontinuity is separated from every causal model by >= 12x at the shipped grid and
    >= 325x at 4x the grid. Discrimination is limited only by how far the causal floor has fallen:
    refine the grid to resolve smaller violations.
    """
    omega = np.asarray(omega_grid, dtype=np.float64).ravel()
    if omega.ndim != 1 or omega.size < 16:
        raise ValueError("check_kk: omega_grid must be a 1-D array of at least 16 points.")
    if np.any(omega <= 0.0):
        raise ValueError("check_kk: omega_grid must be strictly positive (rad/s); omega=0 is the "
                         "Drude DC conductivity pole -- start the grid at a small omega > 0.")
    d = np.diff(omega)
    h = float(np.mean(d))
    if h <= 0.0 or np.max(np.abs(d - h)) > 1e-6 * h:
        raise ValueError("check_kk: the Maclaurin discrete KK requires a UNIFORM, strictly "
                         "increasing omega grid.")
    eps = np.array([complex(model.eps(2.0 * np.pi * C_LIGHT / w, n_m3=n_m3)) for w in omega],
                   dtype=np.complex128)
    re_model = eps.real
    im_chi = eps.imag                                   # Im(eps) = Im(chi); eps_inf is real
    eps_inf_est = float(re_model[-1])                   # chi -> 0 as omega -> inf
    chi = _maclaurin_re_from_im(omega, im_chi, h, edge_correct=edge_correct)
    re_kk = eps_inf_est + chi
    residual = re_model - re_kk
    # step-DOUBLED companion on omega[::2] (same rule, same endpoint treatment) -> the grid's own
    # quadrature-error estimate, evaluated on the shared points.
    quad_err = np.full(omega.size, np.nan, dtype=np.float64)
    if self_calib and omega.size >= 8:
        sub = omega[::2]
        chi2 = _maclaurin_re_from_im(sub, im_chi[::2], float(sub[1] - sub[0]),
                                     edge_correct=edge_correct)
        quad_err[::2] = chi[::2] - chi2
    if metric_band is not None:
        lo, hi = float(metric_band[0]), float(metric_band[1])
        band = (omega >= lo) & (omega <= hi)
    else:
        band = _kk_auto_band(omega, re_model, dc_skip)
    if not np.any(band):
        raise ValueError("check_kk: metric_band selects no grid points.")
    scale = float(np.max(np.abs(re_model[band] - eps_inf_est)))
    if not (scale > 0.0):
        scale = 1.0
    r = residual[band]
    rms_norm = float(np.sqrt(np.mean(r * r)) / scale)
    q = quad_err[band]
    q = q[np.isfinite(q)]
    quad_floor_norm = (float(np.sqrt(np.mean(q * q)) / scale) if q.size else float("nan"))
    span = float(omega[-1] - omega[0])
    h_rel = (h / span) if span > 0.0 else float("nan")
    if np.isfinite(quad_floor_norm) and quad_floor_norm > 0.0:
        causality_ratio = float(rms_norm / quad_floor_norm)
    elif np.isfinite(quad_floor_norm):
        causality_ratio = float("inf")
    else:
        causality_ratio = float("nan")
    feature_pts = _kk_feature_pts(im_chi, band)
    if feature_pts < _KK_MIN_FEATURE_PTS:
        warnings.warn(
            "check_kk: the sharpest feature of Im(eps) inside the metric band spans only about "
            "{:.2f} grid points (h = {:.4g} rad/s, N = {}); the Maclaurin quadrature needs >= {:g} "
            "to resolve a line, and UNDER-RESOLVING a line inflates the residual of a perfectly "
            "CAUSAL model. rms_norm = {:.4e} is therefore NOT comparable against another model's "
            "on this grid -- on the reference grid the ranking inverts below N ~ 6000 (a causal "
            "2-oscillator Lorentz with gamma/h = 2.3 scored 7.8e-3, level with an acausal 2x jump "
            "in gamma). Refine the grid, or narrow the band away from the sharp feature. A "
            "DISCONTINUOUS gamma(omega) never resolves at any N -- that is itself the tell; "
            "confirm by checking whether rms_norm falls under refinement.".format(
                feature_pts, h, omega.size, _KK_MIN_FEATURE_PTS, rms_norm),
            RuntimeWarning, stacklevel=2)
    return {
        "omega": omega, "re_model": re_model, "re_kk": re_kk, "residual": residual,
        "band_mask": band, "eps_inf_est": eps_inf_est, "scale": scale,
        "max_abs": float(np.max(np.abs(r))), "rms_abs": float(np.sqrt(np.mean(r * r))),
        "max_norm": float(np.max(np.abs(r)) / scale),
        "rms_norm": rms_norm,
        "h": h, "h_rel": h_rel,
        "quad_floor_norm": quad_floor_norm,
        "causality_ratio": causality_ratio,
        "rms_norm_per_h": float(rms_norm / h_rel) if h_rel > 0.0 else float("nan"),
        "feature_pts": feature_pts,
        "edge_correct": bool(edge_correct),
    }


# Minimum resolved width, in grid points, of the sharpest Im(eps) feature inside the metric band
# (fix-verify W1 item 5).  Below this the quadrature error of a CAUSAL model swamps the causality
# signal and the causal/acausal ranking can invert; see check_kk's RESOLUTION ENVELOPE table.
_KK_MIN_FEATURE_PTS = 5.0


def _kk_feature_pts(im_chi: np.ndarray, band: np.ndarray) -> float:
    """MODEL-FREE width, in grid points, of the sharpest feature of ``Im(eps)`` inside ``band``.

    A Lorentzian of FWHM ``G`` sampled at step ``h`` has, at its peak, the second difference
    ``y[k-1] - 2 y[k] + y[k+1] ~= -2 A (2h/G)^2``, so the largest band-normalized second difference
    ``rho = max|d2| / max|y|`` inverts to ``G/h ~= sqrt(8/rho)``.  Restricted to the metric band on
    purpose: the low-omega Drude edge (``Im ~ 1/omega^3``) is the sharpest thing on any uniform
    grid, it is never resolved, and check_kk already excludes it from the band."""
    y = np.abs(np.asarray(im_chi, dtype=np.float64))
    if y.size < 3:
        return float("inf")
    d2 = np.abs(y[2:] - 2.0 * y[1:-1] + y[:-2])
    inner = np.asarray(band, dtype=bool)[1:-1]
    if not np.any(inner):
        return float("inf")
    ymax = float(np.max(y[np.asarray(band, dtype=bool)]))
    if not (ymax > 0.0):
        return float("inf")
    rho = float(np.max(d2[inner])) / ymax
    return float(np.sqrt(8.0 / rho)) if rho > 0.0 else float("inf")


def _kk_auto_band(omega, re_model, dc_skip):
    """Auto metric band: the dispersive region around the Re(eps) zero crossing (ENZ /
    screened plasma edge), with the DC-pole edge skipped. Falls back to a mid-grid window."""
    N = omega.size
    lo_floor = float(omega[min(max(int(dc_skip), 1), N - 2)])
    sign = np.sign(re_model)
    cross = np.where(np.diff(sign) != 0)[0]
    if cross.size > 0:
        w_enz = float(omega[cross[0]])
        lo = max(0.3 * w_enz, lo_floor)
        hi = min(6.0 * w_enz, float(omega[-2]))
        if hi <= lo:
            hi = float(omega[-2])
    else:
        lo = float(omega[max(int(0.05 * N), int(dc_skip))])
        hi = float(omega[int(0.6 * N)])
    return (omega >= lo) & (omega <= hi)
