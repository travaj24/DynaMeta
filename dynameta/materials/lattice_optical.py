"""Lattice (polar optical phonon) and oscillator permittivity models.

Phonon-polariton epsilon-near-zero materials (SiC, AlN, SiO2, ...) have no free carriers to speak of: their
infrared permittivity comes from the ions vibrating against each other. Between the transverse (TO) and
longitudinal (LO) optical-phonon frequencies of a mode, Re(eps) < 0 (the Reststrahlen band, where the
crystal reflects almost perfectly); at the LO frequency Re(eps) crosses zero -- the lattice ENZ point.

Models (each an OpticalModel: eps(lambda_m, *, n_m3=None), density-independent):

  PolarLatticeOptical      the factorized TO/LO form (Berreman-Unterwald / Gervais-Piriou / Kukharskii),
                               eps = eps_inf prod_j (w_LO,j^2 - w^2 - i w g_LO,j) / (w_TO,j^2 - w^2 - i w g_TO,j)
                           with separate LO and TO damping (four parameters per mode) -- the form infrared
                           ellipsometry fits use. Passivity is NOT automatic for this form: it is checked at
                           construction and a non-passive damping set is refused.
  LorentzOscillatorOptical the classical sum of Lorentz oscillators (+ an optional Drude term),
                               eps = eps_inf - wp^2 / (w (w + i g_D)) + sum_j d_eps_j w_j^2 / (w_j^2 - w^2 - i w g_j)
                           for multi-oscillator dielectrics and Drude-Lorentz metals (TiN, ZrN). Passive by
                           construction for d_eps_j >= 0, g >= 0.
  GaussianOscillatorOptical  Gaussian-broadened oscillators with their exact Kramers-Kronig partner (Dawson
                           function), the model of Kitamura, Pilon and Jonasz (2007) for fused silica, whose
                           broad amorphous bands a Lorentz sum fits poorly.

ONE Lorentz kernel: lorentz_susceptibility() below is the single implementation of the Lorentz term in the
library; optics.resonance.lorentzian_eps and optics.fdtd_seam.fit_drude_lorentz call it.

Uniaxial crystals (4H/6H-SiC, AlN: E perp c and E par c modes differ) are composed with
materials.uniaxial_optical.UniaxialOptical(ordinary=..., extraordinary=...) -- the tensor goes through the
assembler's existing tensor path (field_response / eps_fields). Doped polar crystals (plasmon-LO-phonon
coupling) are a DrudeOptical whose background is a lattice model: every model here exposes
eps_background(n_m3, lambda_m, eps_inf) (with_free_carriers() builds it), so the free-carrier term is added
to the full lattice response and the coupled L+/L- modes come out of the zeros of the total eps
(coupled_mode_frequencies() gives the undamped closed form for one mode).

Which optical routes consume these models as they are: the scalar models go everywhere a scalar
OpticalModel goes (planar TMM optics.tmm_reference, the RCWA / PMM / Berreman bridges, the FEM); a uniaxial
UniaxialOptical goes to the routes that take a tensor (the FEM, the Berreman, RCWA and PMM bridges for
laterally uniform tensor layers) and is refused by the scalar TMM. The FDTD engines carry one Drude and one
Lorentz pole (FDTDLayer): sample eps(lambda) over the band and fit it with optics.fdtd_seam.
fit_drude_lorentz (exact for a single TO/LO mode with g_TO = g_LO, approximate otherwise); the 1-D engine
refuses the Lorentz pole.

Temperature dependence of the phonon frequencies and damping is NOT included (no clean, verified
parameterization was found for SiC or AlN); the parameters are room-temperature values.

Sign convention exp(-i omega t): Im(eps) >= 0 is loss. Frequencies are angular [rad/s] internally;
from_wavenumbers() constructors take cm^-1 (spectroscopists' units).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from dynameta.constants import C_LIGHT
from dynameta.materials.optical_model import OpticalModel

__all__ = ["lorentz_susceptibility", "PolarLatticeOptical", "LorentzOscillatorOptical", "GaussianOscillatorOptical",
           "with_free_carriers", "coupled_mode_frequencies", "wavenumber_to_rad_s", "rad_s_to_wavenumber",
           "EV_TO_RAD_S"]

_CM_TO_RAD_S = 2.0 * math.pi * C_LIGHT * 100.0      # 1 cm^-1 -> rad/s
EV_TO_RAD_S = 1.602176634e-19 / 1.054571817e-34     # 1 eV / hbar -> rad/s


def wavenumber_to_rad_s(nu_cm):
    """Wavenumber [cm^-1] -> angular frequency [rad/s]."""
    return np.asarray(nu_cm, dtype=np.float64) * _CM_TO_RAD_S if np.ndim(nu_cm) else float(nu_cm) * _CM_TO_RAD_S


def rad_s_to_wavenumber(omega):
    """Angular frequency [rad/s] -> wavenumber [cm^-1]."""
    return np.asarray(omega, dtype=np.float64) / _CM_TO_RAD_S if np.ndim(omega) else float(omega) / _CM_TO_RAD_S


def lorentz_susceptibility(omega_rad_s, delta_eps, omega0, gamma):
    """The Lorentz oscillator term delta_eps omega0^2 / (omega0^2 - omega^2 - i omega gamma) (complex; accepts
    complex omega). THE single Lorentz kernel of the library (optics.resonance.lorentzian_eps and
    optics.fdtd_seam.fit_drude_lorentz use it). Passive (Im >= 0 for real omega > 0) for delta_eps >= 0 and
    gamma >= 0 under exp(-i omega t)."""
    w = np.asarray(omega_rad_s, dtype=np.complex128)
    return delta_eps * omega0 * omega0 / (omega0 * omega0 - w * w - 1j * w * gamma)


def _omega_of(lambda_m):
    return 2.0 * math.pi * C_LIGHT / float(lambda_m)


def _shape_like(v, n_m3):
    return complex(v) if n_m3 is None else np.full(np.shape(n_m3), complex(v), dtype=np.complex128)


def _check_range(model, lambda_m):
    rng = getattr(model, "wavelength_range_m", None)
    if rng is not None:
        lo, hi = rng
        lam = float(lambda_m)
        if not (lo * (1.0 - 1e-12) <= lam <= hi * (1.0 + 1e-12)):
            raise ValueError("{}: wavelength {:.4g} m is outside the model's validity range [{:.4g}, {:.4g}] m "
                             "(the range of the data it was fitted to; no silent extrapolation)".format(
                                 type(model).__name__, lam, lo, hi))


def _check_wavelength_range(rng, where):
    if rng is None:
        return None
    lo, hi = (float(v) for v in rng)
    if not (math.isfinite(lo) and math.isfinite(hi) and 0.0 < lo < hi):
        raise ValueError("{}: wavelength_range_m must be (lo, hi) with 0 < lo < hi [m], got {!r}".format(where, rng))
    return (lo, hi)


class _BackgroundMixin:
    """eps_background(n, lambda, eps_inf) for DrudeOptical(background=...): the full lattice / oscillator
    permittivity as the background of a free-carrier term (plasmon-phonon coupling). The DrudeOptical's own
    eps_inf must equal this model's (refused otherwise: two different high-frequency limits)."""

    def eps_background(self, n_m3, lambda_m, eps_inf):
        if abs(float(eps_inf) - float(self.eps_inf)) > 1e-12 * max(1.0, abs(float(self.eps_inf))):
            raise ValueError("{}.eps_background: the DrudeOptical eps_inf ({!r}) differs from the lattice model's "
                             "eps_inf ({!r}); build the doped model with with_free_carriers() so both are the "
                             "same.".format(type(self).__name__, eps_inf, self.eps_inf))
        v = complex(self.eps(lambda_m))
        return v if n_m3 is None or np.ndim(n_m3) == 0 else np.full(np.shape(n_m3), v, dtype=np.complex128)


@dataclass(frozen=True)
class PolarLatticeOptical(_BackgroundMixin, OpticalModel):
    """Factorized TO/LO lattice permittivity (one or more polar phonon modes):

        eps(w) = eps_inf prod_j (w_LO,j^2 - w^2 - i w g_LO,j) / (w_TO,j^2 - w^2 - i w g_TO,j)

    modes: ((w_TO, w_LO, g_TO, g_LO), ...) in rad/s (use from_wavenumbers for cm^-1), each 0 < w_TO < w_LO
    and g >= 0; for several modes the TO and LO frequencies must interlace (w_TO,1 < w_LO,1 < w_TO,2 <
    w_LO,2 < ...), the condition for a physical (positive) oscillator-strength set.

    Static limit (Lyddane-Sachs-Teller): eps(0) = eps_inf prod (w_LO / w_TO)^2 (static_permittivity());
    check_lst(eps_static) compares it with a measured static value.

    PASSIVITY. Unlike a Lorentz sum, the factorized form can give Im(eps) < 0 (gain) for an unphysical
    damping pair. For one mode Im(eps) has the sign of g_TO w_LO^2 - g_LO w_TO^2 + w^2 (g_LO - g_TO), so
    it is passive at every frequency iff g_TO <= g_LO <= g_TO (w_LO / w_TO)^2. Construction refuses a
    single mode outside that window, and for several modes evaluates Im(eps) on a dense grid (log-spaced
    from 1e-3 w_TO,min to 1e3 w_LO,max plus a fine grid across every mode) and refuses any negative value.

    wavelength_range_m: optional (lo, hi) validity range [m]; eps() raises outside it. Density-independent
    (n_m3 only sets the output shape). Use as DrudeOptical(background=...) for a doped crystal
    (with_free_carriers)."""
    eps_inf: float
    modes: Tuple[Tuple[float, float, float, float], ...]
    wavelength_range_m: Optional[Tuple[float, float]] = None

    def __post_init__(self):
        ei = float(self.eps_inf)
        if not (math.isfinite(ei) and ei > 0.0):
            raise ValueError("PolarLatticeOptical: eps_inf must be finite and > 0, got {!r}".format(self.eps_inf))
        try:
            modes = tuple(tuple(float(v) for v in m) for m in self.modes)
        except (TypeError, ValueError):
            raise ValueError("PolarLatticeOptical: modes must be ((w_TO, w_LO, g_TO, g_LO), ...)") from None
        if not modes or any(len(m) != 4 for m in modes):
            raise ValueError("PolarLatticeOptical: modes must be a non-empty tuple of (w_TO, w_LO, g_TO, g_LO)")
        prev = 0.0
        for wt, wl, gt, gl in modes:
            if not all(math.isfinite(v) for v in (wt, wl, gt, gl)) or gt < 0.0 or gl < 0.0:
                raise ValueError("PolarLatticeOptical: frequencies and damping must be finite, damping >= 0")
            if not (prev < wt < wl):
                raise ValueError("PolarLatticeOptical: each mode needs w_TO < w_LO, and the modes must interlace "
                                 "(w_TO,1 < w_LO,1 < w_TO,2 < ...); got {!r}".format(modes))
            prev = wl
        object.__setattr__(self, "modes", modes)
        object.__setattr__(self, "wavelength_range_m",
                           _check_wavelength_range(self.wavelength_range_m, "PolarLatticeOptical"))
        self._check_passive()

    @classmethod
    def from_wavenumbers(cls, eps_inf, modes_cm, wavelength_range_m=None) -> "PolarLatticeOptical":
        """modes_cm: ((nu_TO, nu_LO, g_TO, g_LO), ...) in cm^-1."""
        return cls(eps_inf=eps_inf, modes=tuple(tuple(wavenumber_to_rad_s(v) for v in m) for m in modes_cm),
                   wavelength_range_m=wavelength_range_m)

    def eps_omega(self, omega):
        """eps at angular frequency omega [rad/s] (array; complex omega accepted)."""
        w = np.asarray(omega, dtype=np.complex128)
        out = np.full(w.shape, complex(self.eps_inf), dtype=np.complex128)
        for wt, wl, gt, gl in self.modes:
            out = out * (wl * wl - w * w - 1j * w * gl) / (wt * wt - w * w - 1j * w * gt)
        return out

    def _check_passive(self):
        if len(self.modes) == 1:
            wt, wl, gt, gl = self.modes[0]
            if not (gt * (1.0 - 1e-12) <= gl <= gt * (wl / wt) ** 2 * (1.0 + 1e-12)):
                raise ValueError(
                    "PolarLatticeOptical: the damping pair g_TO = {:.4g}, g_LO = {:.4g} rad/s gives Im(eps) < 0 "
                    "(gain) somewhere -- the factorized form is passive only for g_TO <= g_LO <= g_TO (w_LO / "
                    "w_TO)^2 = {:.4g} rad/s".format(gt, gl, gt * (wl / wt) ** 2))
            return
        wmin = min(m[0] for m in self.modes)
        wmax = max(m[1] for m in self.modes)
        grids = [np.geomspace(1e-3 * wmin, 1e3 * wmax, 20001)]
        for wt, wl, gt, gl in self.modes:
            g = max(gt, gl, 1e-6 * wt)
            grids.append(np.linspace(max(wt - 50.0 * g, 1e-6 * wt), wl + 50.0 * g, 20001))
        w = np.concatenate(grids)
        e = self.eps_omega(w)
        bad = e.imag < -1e-12 * np.abs(e)
        if np.any(bad):
            raise ValueError("PolarLatticeOptical: Im(eps) < 0 (gain) near {:.4g} cm^-1 for this damping set -- "
                             "not passive; adjust the TO / LO damping (each mode alone must satisfy g_TO <= g_LO "
                             "<= g_TO (w_LO/w_TO)^2, and the product must stay passive)".format(
                                 float(rad_s_to_wavenumber(w[np.argmax(bad)]))))

    def static_permittivity(self) -> float:
        """Lyddane-Sachs-Teller: eps(0) = eps_inf prod (w_LO / w_TO)^2."""
        return float(self.eps_inf * np.prod([(wl / wt) ** 2 for wt, wl, _gt, _gl in self.modes]))

    def check_lst(self, eps_static: float, rtol: float = 0.05) -> float:
        """Relative deviation of the LST static permittivity from a measured eps_static; raises ValueError
        when it exceeds rtol (the TO/LO/eps_inf set is then inconsistent with that static value)."""
        dev = self.static_permittivity() / float(eps_static) - 1.0
        if abs(dev) > float(rtol):
            raise ValueError("PolarLatticeOptical: LST static permittivity {:.4g} deviates {:+.2%} from eps_static "
                             "= {:.4g} (tolerance {:.1%})".format(self.static_permittivity(), dev, eps_static, rtol))
        return dev

    def lo_wavelengths_m(self) -> Tuple[float, ...]:
        """Wavelengths [m] of the LO frequencies (the undamped lattice ENZ points)."""
        return tuple(2.0 * math.pi * C_LIGHT / wl for _wt, wl, _gt, _gl in self.modes)

    def eps(self, lambda_m: float, *, n_m3=None):
        _check_range(self, lambda_m)
        return _shape_like(self.eps_omega(_omega_of(lambda_m))[()], n_m3)


@dataclass(frozen=True)
class LorentzOscillatorOptical(_BackgroundMixin, OpticalModel):
    """Sum of Lorentz oscillators with an optional Drude term:

        eps(w) = eps_inf - wp^2 / (w (w + i g_D)) + sum_j d_eps_j w_j^2 / (w_j^2 - w^2 - i w g_j)

    oscillators: ((d_eps, w0, gamma), ...) with d_eps >= 0 (strength = static contribution), w0 > 0, gamma >= 0
    [rad/s]; drude: None or (wp, g_D) [rad/s] (a metal: TiN, ZrN). Each Lorentz term is
    lorentz_susceptibility (the one kernel). Passive by construction. from_ev() takes energies in eV (the
    units of most Drude-Lorentz tables); from_wavenumbers() cm^-1. static_permittivity() = eps_inf + sum d_eps
    (no Drude term). wavelength_range_m: optional validity range [m]."""
    eps_inf: float
    oscillators: Tuple[Tuple[float, float, float], ...] = ()
    drude: Optional[Tuple[float, float]] = None
    wavelength_range_m: Optional[Tuple[float, float]] = None

    def __post_init__(self):
        ei = float(self.eps_inf)
        if not (math.isfinite(ei) and ei > 0.0):
            raise ValueError("LorentzOscillatorOptical: eps_inf must be finite and > 0")
        osc = tuple(tuple(float(v) for v in o) for o in self.oscillators)
        for o in osc:
            if len(o) != 3 or not all(math.isfinite(v) for v in o) or o[0] < 0.0 or o[1] <= 0.0 or o[2] < 0.0:
                raise ValueError("LorentzOscillatorOptical: each oscillator is (d_eps >= 0, w0 > 0, gamma >= 0); got "
                                 "{!r}".format(o))
        object.__setattr__(self, "oscillators", osc)
        if self.drude is not None:
            d = tuple(float(v) for v in self.drude)
            if len(d) != 2 or not all(math.isfinite(v) for v in d) or d[0] <= 0.0 or d[1] < 0.0:
                raise ValueError("LorentzOscillatorOptical: drude is (wp > 0, gamma >= 0) [rad/s]")
            object.__setattr__(self, "drude", d)
        if not osc and self.drude is None:
            raise ValueError("LorentzOscillatorOptical: give at least one oscillator or a Drude term")
        object.__setattr__(self, "wavelength_range_m",
                           _check_wavelength_range(self.wavelength_range_m, "LorentzOscillatorOptical"))

    @classmethod
    def from_ev(cls, eps_inf, oscillators_ev=(), drude_ev=None, wavelength_range_m=None):
        """oscillators_ev: ((d_eps, E0 [eV], gamma [eV]), ...); drude_ev: (E_p [eV], gamma_D [eV]) or None."""
        osc = tuple((float(d), float(e) * EV_TO_RAD_S, float(g) * EV_TO_RAD_S) for d, e, g in oscillators_ev)
        dr = None if drude_ev is None else (float(drude_ev[0]) * EV_TO_RAD_S, float(drude_ev[1]) * EV_TO_RAD_S)
        return cls(eps_inf=eps_inf, oscillators=osc, drude=dr, wavelength_range_m=wavelength_range_m)

    @classmethod
    def from_wavenumbers(cls, eps_inf, oscillators_cm=(), drude_cm=None, wavelength_range_m=None):
        """oscillators_cm: ((d_eps, nu0, gamma), ...) in cm^-1; drude_cm: (nu_p, gamma_D) in cm^-1 or None."""
        osc = tuple((float(d), wavenumber_to_rad_s(n), wavenumber_to_rad_s(g)) for d, n, g in oscillators_cm)
        dr = None if drude_cm is None else (wavenumber_to_rad_s(drude_cm[0]), wavenumber_to_rad_s(drude_cm[1]))
        return cls(eps_inf=eps_inf, oscillators=osc, drude=dr, wavelength_range_m=wavelength_range_m)

    def eps_omega(self, omega):
        w = np.asarray(omega, dtype=np.complex128)
        out = np.full(w.shape, complex(self.eps_inf), dtype=np.complex128)
        if self.drude is not None:
            wp, gd = self.drude
            out = out - wp * wp / (w * (w + 1j * gd))
        for d, w0, g in self.oscillators:
            out = out + lorentz_susceptibility(w, d, w0, g)
        return out

    def static_permittivity(self) -> float:
        return float(self.eps_inf + sum(d for d, _w, _g in self.oscillators))

    def eps(self, lambda_m: float, *, n_m3=None):
        _check_range(self, lambda_m)
        return _shape_like(self.eps_omega(_omega_of(lambda_m))[()], n_m3)


@dataclass(frozen=True)
class GaussianOscillatorOptical(_BackgroundMixin, OpticalModel):
    """Gaussian-broadened oscillators with their exact Kramers-Kronig partner (A. M. Efimov / D. De Sousa
    Meneses; the form of Kitamura, Pilon and Jonasz, Appl. Opt. 46, 8118 (2007), Eqs. (21)-(24)), for the
    broad bands of amorphous solids:

        eps(nu) = eps_inf + sum_j [g_kk,j(nu) + i g_j(nu)]
        g_j     = a_j [exp(-4 ln2 ((nu - nu_j)/s_j)^2) - exp(-4 ln2 ((nu + nu_j)/s_j)^2)]
        g_kk,j  = (2 a_j / sqrt(pi)) [D(2 sqrt(ln2) (nu + nu_j)/s_j) - D(2 sqrt(ln2) (nu - nu_j)/s_j)]

    with nu the wavenumber [cm^-1], nu_j the band centre, s_j its FWHM [cm^-1], a_j its amplitude and D the
    Dawson function. Im(eps) = sum g_j >= 0 for nu > 0 (passive), odd in nu (causal). oscillators:
    ((a, nu0_cm, fwhm_cm), ...), a >= 0. wavelength_range_m: optional validity range [m]."""
    eps_inf: float
    oscillators: Tuple[Tuple[float, float, float], ...]
    wavelength_range_m: Optional[Tuple[float, float]] = None

    def __post_init__(self):
        ei = float(self.eps_inf)
        if not (math.isfinite(ei) and ei > 0.0):
            raise ValueError("GaussianOscillatorOptical: eps_inf must be finite and > 0")
        osc = tuple(tuple(float(v) for v in o) for o in self.oscillators)
        if not osc:
            raise ValueError("GaussianOscillatorOptical: give at least one oscillator")
        for o in osc:
            if len(o) != 3 or not all(math.isfinite(v) for v in o) or o[0] < 0.0 or o[1] <= 0.0 or o[2] <= 0.0:
                raise ValueError("GaussianOscillatorOptical: each oscillator is (a >= 0, nu0 > 0, fwhm > 0) [cm^-1]")
        object.__setattr__(self, "oscillators", osc)
        object.__setattr__(self, "wavelength_range_m",
                           _check_wavelength_range(self.wavelength_range_m, "GaussianOscillatorOptical"))

    def eps_wavenumber(self, nu_cm):
        """eps at wavenumber nu [cm^-1] (real array)."""
        from scipy.special import dawsn
        nu = np.asarray(nu_cm, dtype=np.float64)
        c = 4.0 * math.log(2.0)
        k = 2.0 * math.sqrt(math.log(2.0))
        re = np.full(nu.shape, float(self.eps_inf))
        im = np.zeros(nu.shape)
        for a, n0, s in self.oscillators:
            im = im + a * (np.exp(-c * ((nu - n0) / s) ** 2) - np.exp(-c * ((nu + n0) / s) ** 2))
            re = re + (2.0 * a / math.sqrt(math.pi)) * (dawsn(k * (nu + n0) / s) - dawsn(k * (nu - n0) / s))
        return re + 1j * im

    def static_permittivity(self) -> float:
        return float(self.eps_wavenumber(0.0).real)

    def eps(self, lambda_m: float, *, n_m3=None):
        _check_range(self, lambda_m)
        return _shape_like(self.eps_wavenumber(0.01 / float(lambda_m))[()], n_m3)


def with_free_carriers(lattice, *, m_opt_kg, gamma_rad_s):
    """A DrudeOptical for a DOPED polar crystal: the free-carrier term on top of the full lattice response
    (plasmon-LO-phonon coupling), eps = eps_lattice(w) - wp^2(n) / (w (w + i gamma)), wp^2 = n e^2 / (eps0
    m_opt). m_opt_kg / gamma_rad_s as DrudeOptical takes them (floats or callables of n). The density n comes
    from the carrier solve (or materials.enz_presets.FixedDensityOptical for a uniformly doped layer)."""
    from dynameta.materials.optical_model import DrudeOptical
    if not hasattr(lattice, "eps_background"):
        raise TypeError("with_free_carriers: lattice must be a PolarLatticeOptical / LorentzOscillatorOptical / "
                        "GaussianOscillatorOptical")
    return DrudeOptical(eps_inf=float(lattice.eps_inf), m_opt_kg=m_opt_kg, gamma_rad_s=gamma_rad_s, background=lattice)


def coupled_mode_frequencies(eps_inf: float, omega_TO: float, omega_LO: float, omega_p: float):
    """Undamped coupled LO-phonon-plasmon frequencies (omega_-, omega_+) [rad/s] of ONE polar mode plus a free-
    carrier plasma (omega_p unscreened, omega_p^2 = n e^2 / (eps0 m)): the zeros of
    eps_inf (w_LO^2 - w^2) / (w_TO^2 - w^2) - omega_p^2 / w^2, i.e. the roots of
    w^4 - w^2 (w_LO^2 + omega_p^2 / eps_inf) + omega_p^2 w_TO^2 / eps_inf = 0. omega_- -> 0 and omega_+ ->
    w_LO as omega_p -> 0; omega_- -> w_TO and omega_+ -> omega_p / sqrt(eps_inf) for large omega_p."""
    s = omega_LO ** 2 + omega_p ** 2 / eps_inf
    p = omega_p ** 2 * omega_TO ** 2 / eps_inf
    disc = math.sqrt(max(s * s - 4.0 * p, 0.0))
    upper2 = 0.5 * (s + disc)
    return math.sqrt(p / upper2), math.sqrt(upper2)            # w_-^2 = p / w_+^2 (no cancellation)
