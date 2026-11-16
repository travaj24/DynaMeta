"""Resolved free-carrier scattering / mass closures for the Drude optical model (roadmap R2):
a Kane nonparabolic optical mass m_opt(n) and a Matthiessen damping Gamma(n; T) = sum of phonon,
ionized-impurity and grain-boundary channels. Both are plain callables-of-n that plug THROUGH the
existing DrudeOptical m_opt_kg / gamma_rad_s callable seam (DrudeOptical itself is unchanged), so the
default constant-Drude behavior is byte-identical when the new knobs are neutral. Pure numpy, SI units;
spell out omega/Gamma/tau. (materials/scattering.py is also the home for the R3 shared ScatteringModel.)

Off-switches:
  KaneOpticalMass(m0_kg=m, alpha_eV=0.0)            -> m_opt(n) == m  exactly (constant).
  MatthiessenGamma(gamma_const_rad_s=g, ...=0)      -> Gamma(n) == g  exactly (constant).

v0.16: the classical SIZE EFFECT of a thin conducting film (BoundaryScattering, fuchs_sondheimer_factor),
an opt-in addition to the bulk rate above (DrudeOptical(boundary=...)). The bulk rate keeps its role (any
float / callable / MatthiessenGamma; a density-dependent ionized-impurity or grain-boundary rate is
already expressible through a callable gamma(n) or MatthiessenGamma's Brooks-Herring channel, so no
separate mobility input is added); the boundary terms are combined with it per permittivity component.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Union

import numpy as np

from dynameta.constants import Q_E, M_E, HBAR
from dynameta.constants import T_REF   # audit X-24: one reference temperature

_N_FLOOR = 1.0e10        # mirror the ITO DOS-mass closure floor (avoid 0**(1/3) edge)


@dataclass(frozen=True)
class KaneOpticalMass:
    """Density-dependent Kane (nonparabolic) optical/conductivity mass.

    For the Kane band E (1 + alpha E) = hbar^2 k^2 / (2 m0), the degenerate optical
    (Drude/plasma) mass is the Fermi-surface value
        m_opt = hbar k_F / v_F = m0 (1 + 2 alpha E_F),
    with k_F = (3 pi^2 n)^(1/3) fixed by counting and E_F the TRUE Kane Fermi level
    solving E_F (1 + alpha E_F) = gamma_F,  gamma_F = hbar^2 k_F^2 / (2 m0).
    The quadratic-inversion identity  1 + 2 alpha E_F = sqrt(1 + 4 alpha gamma_F)
    is what the default computes -- so alpha_eV here is the STANDARD Kane
    nonparabolicity (literature ITO ~0.4 eV^-1), the same alpha the library's
    schrodinger_poisson and carrier_heating closures use.

    audit C7b-1: the previous default evaluated sqrt(1 + 2 alpha gamma_F) -- the Kane
    alpha silently HALVED to leading order (m_opt 8.6-22% low over n = 1e26-2e27 m^-3
    with literature ITO alpha; lambda_ENZ ~4-12% short) -- and its gate/test compared
    against a copy of the same formula. legacy=True keeps the old closure for
    back-compat with alphas CALIBRATED under it (alpha_legacy = 2 alpha_Kane
    reproduces the old numbers exactly, since sqrt(1 + 2*(2a)*g) = sqrt(1 + 4 a g)).

    alpha_eV=0 -> exactly m0 (constant) on both branches. Callable of n -> kg.
    `exponent` is a LEGACY-branch fit knob only (the exact Kane mass has none)."""
    m0_kg: float
    alpha_eV: float = 0.0
    exponent: float = 0.5
    legacy: bool = False

    @classmethod
    def from_band(cls, band) -> "KaneOpticalMass":
        """The optical mass of a materials.KaneBand (v0.16): KaneOpticalMass(m0_kg=band.m0_kg,
        alpha_eV=band.alpha_per_eV) -- the same (m0, alpha) the electrostatics solves with, so the Drude
        mass and the carrier density come from one band (KaneBand.optical_mass returns this)."""
        return cls(m0_kg=float(band.m0_kg), alpha_eV=float(band.alpha_per_eV))

    def __post_init__(self):
        if not self.legacy and self.exponent != 0.5:
            raise ValueError(
                "KaneOpticalMass: `exponent` is a legacy-closure fit knob; the exact Kane "
                "optical mass has no exponent freedom. Pass legacy=True to use the old "
                "(1 + 2 alpha gamma_F)^exponent closure (audit C7b-1).")

    def _gamma_F(self, n_m3):
        """(gamma_F [J], alpha gamma_F) at density n: gamma_F = hbar^2 k_F^2 / (2 m0), k_F = (3 pi^2 n)^(1/3)."""
        n = np.maximum(np.asarray(n_m3, dtype=np.float64), _N_FLOOR)
        kF = np.power(3.0 * np.pi ** 2 * n, 1.0 / 3.0)
        g_F = HBAR ** 2 * kF ** 2 / (2.0 * self.m0_kg)                 # J (parabolic-form energy)
        a_g = self.alpha_eV * g_F / Q_E                                # alpha * gamma_F, dimensionless
        return g_F, a_g

    def __call__(self, n_m3):
        g_F, a_g = self._gamma_F(n_m3)
        if self.legacy:
            return self.m0_kg * np.power(1.0 + 2.0 * a_g, self.exponent)
        return self.m0_kg * np.sqrt(1.0 + 4.0 * a_g)                   # = m0 (1 + 2 alpha E_F)

    def fermi_energy_J(self, n_m3):
        """Degenerate (T = 0) Kane Fermi energy above the band edge [J], the root of E (1 + alpha E) =
        gamma_F: E_F = 2 gamma_F / (1 + sqrt(1 + 4 alpha gamma_F)) (the cancellation-free form; alpha = 0 ->
        gamma_F, the parabolic value). The same root the optical mass m0 (1 + 2 alpha E_F) uses (v0.16:
        the band-filling shift of BandFillingBackground). Not defined for the legacy closure."""
        if self.legacy:
            raise ValueError("KaneOpticalMass.fermi_energy_J: the legacy closure has no Fermi level "
                             "(it is a fit form); use the exact Kane closure (legacy=False).")
        g_F, a_g = self._gamma_F(n_m3)
        return 2.0 * g_F / (1.0 + np.sqrt(1.0 + 4.0 * a_g))


def _bose(x):
    """Bose occupation 1/(exp(x)-1) = 1/expm1(x); x = T_debye/T."""
    return 1.0 / np.expm1(np.asarray(x, dtype=np.float64))


@dataclass(frozen=True)
class MatthiessenGamma:
    """Matthiessen free-carrier damping Gamma(n) = optical_dc_ratio * (1/tau_gb + 1/tau_phonon(T) +
    1/tau_ii(n)), a callable of carrier density n -> rad/s. Temperature is model STATE (T_K), set by the
    caller (e.g. an electro-thermo loop does dataclasses.replace(gamma, T_K=...)); the per-call T/omega
    signature widening is deferred (roadmap R3/R6). All channels are >= 0 so Gamma >= 0 (passive);
    Gamma is exactly 0 only if EVERY channel is left at its zero default (a deliberate lossless
    configuration -- DrudeOptical accepts gamma = 0 but rejects gamma < 0).

      1/tau_gb       = gamma_const_rad_s                         (grain-boundary + any T,n-independent floor)
      1/tau_phonon   = gamma_phonon_300K_rad_s * f_T             (LO/acoustic phonons; f_T below)
      1/tau_ii(n)    = bh_prefactor_rad_s * (n/bh_n_ref_m3)^p * (m_opt(n)/m_ref)      (degenerate Brooks-
                       Herring RATE scaling; audit 7b: the previous (m_ref/m)^2 factor was the
                       MOBILITY mass law mu_ii ~ 1/m*^2, but the damping RATE 1/tau = q/(m* mu)
                       scales ~ m*^{+1} x a screening log -- a Born probe at ITO parameters showed
                       the true rate RISING +28% over m* = 0.30->0.50 me while the old factor FELL
                       to 0.36, a ~3.5x trend distortion. The absolute prefactor stays
                       CALIBRATION-bearing, default 0 = off; re-fit it after this change.)
      f_T            = (T/300)              if debye_T_K <= 0 (linear high-T)
                       bose(Td/T)/bose(Td/300)  otherwise (LO-phonon Bose occupation)
    """
    gamma_const_rad_s: float = 0.0
    gamma_phonon_300K_rad_s: float = 0.0
    T_K: float = T_REF
    debye_T_K: float = 0.0
    bh_prefactor_rad_s: float = 0.0
    bh_n_ref_m3: float = 1.0e27
    bh_exponent: float = 1.0
    m_opt: Optional[Union[float, "KaneOpticalMass"]] = None
    optical_dc_ratio: float = 1.0

    def __post_init__(self):
        # T_K is mutable model state (set via dataclasses.replace by an electro-thermo loop); guard it
        # here so a non-positive temperature fails loudly rather than producing a divide-warning /
        # negative (gain) phonon rate in the Bose branch.
        if not (self.T_K > 0.0):
            raise ValueError("MatthiessenGamma: T_K must be > 0 (K), got {!r}".format(self.T_K))
        # the optical/DC ratio is a positive scale factor; a non-positive value would flip the whole
        # damping to zero/negative (gain under exp(-i omega t)) through an otherwise-valid channel sum.
        if not (self.optical_dc_ratio > 0.0):
            raise ValueError("MatthiessenGamma: optical_dc_ratio must be > 0, got {!r}".format(
                self.optical_dc_ratio))

    def _phonon(self) -> float:
        if self.gamma_phonon_300K_rad_s <= 0.0:
            return 0.0
        if self.debye_T_K <= 0.0:
            return self.gamma_phonon_300K_rad_s * (self.T_K / 300.0)
        return self.gamma_phonon_300K_rad_s * float(_bose(self.debye_T_K / self.T_K) /
                                                    _bose(self.debye_T_K / 300.0))

    def _m_opt(self, n):
        if self.m_opt is None:
            return M_E
        return self.m_opt(n) if callable(self.m_opt) else float(self.m_opt)

    def _ii(self, n):
        if self.bh_prefactor_rad_s <= 0.0:
            return 0.0
        m = self._m_opt(n)
        m_ref = self._m_opt(self.bh_n_ref_m3)
        # audit 7b: RATE ~ m*^{+1} (1/tau = q/(m* mu), mu_ii ~ 1/m*^2), not the old
        # mobility-law (m_ref/m)^2 which moved the channel the WRONG WAY with mass
        return (self.bh_prefactor_rad_s * np.power(n / self.bh_n_ref_m3, self.bh_exponent)
                * (m / m_ref))

    def __call__(self, n_m3):
        n = np.maximum(np.asarray(n_m3, dtype=np.float64), _N_FLOOR)
        return self.optical_dc_ratio * (self.gamma_const_rad_s + self._phonon() + self._ii(n))


@dataclass(frozen=True)
class ScatteringModel:
    """ONE momentum-relaxation law tau(n;T) shared by BOTH the optical Drude damping and the transport
    mobility (roadmap R3), removing the hidden inconsistency of fitting them independently. The shared
    quantity is 1/tau (rad/s), supplied as `one_over_tau` -- typically a MatthiessenGamma (R2) so the
    SAME density/temperature scattering law feeds both sides:

        optical Drude gamma(n) = 1/tau(n)                          (mass enters wp, NOT gamma)
        drift mobility  mu(n)  = hall_factor * q / (m_cond(n) * 1/tau(n))   [m^2/Vs]

    Build the link around tau, never around mu -- writing gamma = q/(m_opt mu) is circular (it just gives
    1/tau back). CAVEATS, encoded as separate inputs so they are not conflated:
      - m_cond is the CONDUCTIVITY (DC) effective mass, DISTINCT from the optical mass m_opt (which sets
        wp) and from the DOS mass; in a nonparabolic conductor all three differ.
      - hall_factor r_H: a measured Hall mobility is r_H * drift mobility; the tau<->mu link uses the
        DRIFT mobility, so set hall_factor only if you want the produced mu to be a Hall mobility
        (default 1.0 = drift). A measured Hall mu fed back to calibrate tau should be divided by r_H.
    Pure numpy; attach to a Material via Material(scattering=...) (opt-in; default unset = byte-identical).
    """
    one_over_tau: Union[float, Callable]
    m_cond_kg: Union[float, Callable] = M_E
    hall_factor: float = 1.0

    def inv_tau(self, n_m3):
        ot = self.one_over_tau
        return np.asarray(ot(n_m3), dtype=np.float64) if callable(ot) else float(ot)

    def tau_s(self, n_m3):
        return 1.0 / self.inv_tau(n_m3)

    def gamma_optical_of_n(self):
        """Optical Drude gamma callable gamma(n) = 1/tau(n) (reuses the MatthiessenGamma directly)."""
        if callable(self.one_over_tau):
            return self.one_over_tau
        return _ConstantRate(float(self.one_over_tau))

    def mobility_of_n(self):
        """Drift mobility callable mu(n) = hall_factor * q / (m_cond(n) * 1/tau_DC(n)) [m^2/Vs].

        Audit S4-5: `one_over_tau` supplies the OPTICAL Drude rate; when it is a MatthiessenGamma
        with optical_dc_ratio != 1 that rate is optical_dc_ratio x the DC momentum-relaxation
        rate, so the DC drift mobility must divide the ratio back OUT -- the old code fed the
        optical rate straight into mu, corrupting the transport mobility (and the DEVSIM solve)
        by exactly that factor. Arbitrary callables without the attribute are taken as DC-equal
        (ratio 1), preserving the shared-law contract.

        The callable is a frozen dataclass (not a closure), so a Material built with scattering= pickles
        (process pools) and has a deterministic repr (cache keys)."""
        ot = self.one_over_tau
        return _DriftMobility(one_over_tau=ot, m_cond_kg=self.m_cond_kg, hall_factor=self.hall_factor,
                              optical_dc_ratio=float(getattr(ot, "optical_dc_ratio", 1.0)))


@dataclass(frozen=True)
class _ConstantRate:
    """A constant rate as a callable of n (picklable, deterministic repr)."""
    value: float

    def __call__(self, n_m3):
        return self.value


@dataclass(frozen=True)
class _DriftMobility:
    """mu(n) = hall_factor * q / (m_cond(n) * 1/tau_DC(n)) with 1/tau_DC = one_over_tau / optical_dc_ratio
    (ScatteringModel.mobility_of_n)."""
    one_over_tau: Union[float, Callable]
    m_cond_kg: Union[float, Callable]
    hall_factor: float
    optical_dc_ratio: float

    def __call__(self, n_m3):
        ot, mc = self.one_over_tau, self.m_cond_kg
        n = np.asarray(n_m3, dtype=np.float64)
        m = np.asarray(mc(n), dtype=np.float64) if callable(mc) else float(mc)
        g = np.asarray(ot(n), dtype=np.float64) if callable(ot) else float(ot)
        g_dc = g / self.optical_dc_ratio                   # undo the DC->optical scaling
        if np.any(np.asarray(m) <= 0.0) or np.any(np.asarray(g_dc) <= 0.0):
            raise ValueError("ScatteringModel.mobility_of_n: m_cond and 1/tau must be > 0 (a "
                             "callable returned a non-positive value -> mu would be inf/NaN/negative).")
        return self.hall_factor * Q_E / (m * g_dc)


# ================================================================================================
# v0.16: boundary (size-effect) scattering of a thin conducting film
# ================================================================================================
#
# IN PLANE (field parallel to the film): the Fuchs-Sondheimer solution of the Boltzmann equation in a
# film of thickness d with a bulk relaxation time tau and a specularity p (the fraction of carriers
# reflected specularly at each face; the rest leave the face with the equilibrium distribution).
# K. Fuchs, Proc. Camb. Phil. Soc. 34, 100 (1938); E. H. Sondheimer, Adv. Phys. 1, 1 (1952):
#
#     sigma / sigma_bulk = F(kappa, p)
#     F = 1 - (3 (1 - p) / (2 kappa)) int_1^inf (t^-3 - t^-5) (1 - exp(-kappa t)) / (1 - p exp(-kappa t)) dt,
#
# kappa = d / l, l = v_F tau the bulk mean free path. At frequency omega (exp(-i omega t)) the Boltzmann
# equation is the d.c. one with 1/tau -> 1/tau - i omega, so for a field uniform across the film (d much
# smaller than the skin depth) the SAME F holds with the complex kappa = d (gamma - i omega) / v_F and
# multiplies the Drude conductivity eps0 wp^2 / (gamma - i omega). Evaluated exactly: expanding
# 1 / (1 - p x) in powers of p x, x = exp(-kappa t), every term is a pair of generalized exponential
# integrals,
#
#     F = 1 - (3 (1 - p) / (2 kappa)) [1/4 - (1 - p) sum_{m >= 1} p^(m-1) (E_3(m kappa) - E_5(m kappa))],
#
# with E_1 from scipy (complex argument), E_2..E_5 by the upward recurrence E_{n+1} = (exp(-z) - z E_n)/n
# for |z| < 50 and the asymptotic series beyond. Limits: p = 1 -> F = 1 exactly (no size effect);
# |kappa| >> 1 -> F = 1 - 3 (1 - p) / (8 kappa), i.e. an ADDITIVE (Matthiessen) rate 3 (1 - p) v_F / (8 d)
# at any frequency; kappa -> 0 (d.c.) -> F ~ (3/4) kappa ln(1/kappa) (p = 0).
#
# OUT OF PLANE (field normal to the film): no comparably standard closed theory exists for a local Drude
# eps_zz; the widely used phenomenology (U. Kreibig and M. Vollmer, Optical Properties of Metal Clusters,
# Springer 1995, sec. 2.2.4; E. A. Coronado and G. C. Schatz, J. Chem. Phys. 119, 3926 (2003) for the
# length L_eff = 4 V / S) adds Gamma_s = A v_F / L_eff to the bulk rate (Matthiessen). A is an ORDER-ONE
# MODEL COEFFICIENT (about 0.1 .. 1.5 in the cluster literature), not a derived number. For a slab
# L_eff = 4 V / S = 2 d; in that convention the in-plane Fuchs-Sondheimer thick-film rate above is
# A = (3/4)(1 - p). The 'local' length (BoundaryScattering.confinement_length_m) is a HEURISTIC for the
# Landau-type damping of carriers confined in an accumulation / depletion layer; calibrate A
# (validation/thin_film_damping.py compares it with the self-consistent quantum response of
# optics.quantum_film, which contains that damping through the sub-band spectrum).
#
# Not modelled: quantization of the motion across the film (needs k_F d >> 1; at k_F d ~ 10 the
# classical theory is only indicative -- use optics.quantum_film), surface-roughness correlation lengths
# (Soffer), grain-boundary reflection (Mayadas-Shatzkes; use a bulk callable rate), anisotropic tau.

_FS_ASYMPTOTIC_ABS = 50.0       # |z| from which E_3, E_5 come from the asymptotic series
_FS_ASYMPTOTIC_TERMS = 45       # 45 terms at |z| = 50: smallest term < 1e-15 relative (order 5)
_FS_MAX_TERMS = 20000           # p-series cap (r = p exp(-Re kappa) near 1 converges slowly)
_FS_TAIL_TOL = 1e-16            # p-series stop: bound on the omitted tail's contribution to F
_PERP_LENGTHS = ("film", "local")


def _expn_3_5(z):
    """(E_3(z), E_5(z)) of complex z with Re z > 0 (elementwise arrays)."""
    from scipy.special import exp1
    z = np.asarray(z, dtype=np.complex128)
    e3 = np.empty_like(z)
    e5 = np.empty_like(z)
    big = np.abs(z) >= _FS_ASYMPTOTIC_ABS
    small = ~big
    if np.any(small):
        zs = z[small]
        ez = np.exp(-zs)
        E = exp1(zs)
        orders = [E]
        for k in range(1, 5):
            E = (ez - zs * E) / k
            orders.append(E)
        e3[small] = orders[2]
        e5[small] = orders[4]
    if np.any(big):
        zb = z[big]
        pref = np.exp(-zb) / zb
        for order, out in ((3, e3), (5, e5)):
            term = np.ones_like(zb)
            acc = np.ones_like(zb)
            for k in range(1, _FS_ASYMPTOTIC_TERMS + 1):
                term = term * (-(order + k - 1)) / zb
                acc = acc + term
                if float(np.max(np.abs(term))) < 1e-17:
                    break
            out[big] = pref * acc
    return e3, e5


def fuchs_sondheimer_factor(kappa, specularity: float = 0.0):
    """Fuchs-Sondheimer in-plane conductivity ratio F(kappa, p) = sigma_film / sigma_bulk (complex array).

    kappa = d / l (d.c.; real) or d (gamma - i omega) / v_F (frequency omega; complex, Re > 0);
    specularity p in [0, 1]. Exact series in generalized exponential integrals (module notes). p = 1
    returns ones (no size effect). Raises on Re(kappa) <= 0, or on a p-series that does not converge in
    _FS_MAX_TERMS terms (p -> 1 together with Re kappa -> 0)."""
    k = np.asarray(kappa, dtype=np.complex128)
    p = float(specularity)
    if not (0.0 <= p <= 1.0):
        raise ValueError("fuchs_sondheimer_factor: specularity must lie in [0, 1], got {!r}".format(p))
    if not np.all(np.isfinite(k)) or np.any(k.real <= 0.0):
        raise ValueError("fuchs_sondheimer_factor: kappa must be finite with Re(kappa) > 0 (d / mean free "
                         "path: a film of positive thickness and a positive bulk rate).")
    if p == 1.0:
        return np.ones_like(k)
    q = 1.0 - p
    e3, e5 = _expn_3_5(k)
    S = e3 - e5
    if p > 0.0:
        a = k.real
        r = p * np.exp(-a)                          # bound on the ratio of successive terms
        scale = 1.5 * q * q / np.abs(k) / (4.0 * (1.0 - r))
        pm = 1.0
        for m in range(2, _FS_MAX_TERMS + 2):
            # omitted tail after m - 1 terms: |sum_{j >= m} p^(j-1) (E3 - E5)(j kappa)|
            #   <= p^(m-1) exp(-m a) / (4 (1 - r))   (|E3 - E5|(z) <= exp(-Re z) / 4)
            if float(np.max(scale * pm * p * np.exp(-m * a))) < _FS_TAIL_TOL:
                break
            pm *= p
            e3, e5 = _expn_3_5(m * k)
            S = S + pm * (e3 - e5)
        else:
            raise ValueError("fuchs_sondheimer_factor: the specular series did not converge in {} terms "
                             "(specularity {} with Re(kappa) down to {:.3g}); use a smaller specularity."
                             .format(_FS_MAX_TERMS, p, float(np.min(k.real))))
    return 1.0 - 1.5 * q / k * (0.25 - q * S)


def fs_effective_rate(omega_rad_s, gamma_bulk_rad_s, fermi_velocity_m_s, thickness_m: float,
                      specularity: float = 0.0):
    """Complex effective in-plane rate of the Fuchs-Sondheimer film: the gamma_eff for which
    eps0 wp^2 / (gamma_eff - i omega) equals F(kappa, p) eps0 wp^2 / (gamma - i omega), kappa =
    d (gamma - i omega) / v_F; gamma_eff = (gamma - i omega) / F + i omega [rad/s]. Re = the damping
    rate (omega = 0: gamma / F, the classical d.c. size effect; |kappa| >> 1: gamma + 3 (1 - p) v_F /
    (8 d)); Im = the accompanying reactive (mass) renormalization (second order in 1 / kappa)."""
    w = np.asarray(omega_rad_s, dtype=np.float64)
    g = np.asarray(gamma_bulk_rad_s, dtype=np.float64)
    v = np.asarray(fermi_velocity_m_s, dtype=np.float64)
    F = fuchs_sondheimer_factor(float(thickness_m) * (g - 1j * w) / v, specularity)
    return (g - 1j * w) / F + 1j * w


_FILM_THICKNESS_RANGE_M = (1e-11, 1e-3)


def _check_film_thickness(d: float, where: str) -> float:
    """d itself, or ValueError when it is outside _FILM_THICKNESS_RANGE_M (a length-unit slip)."""
    lo, hi = _FILM_THICKNESS_RANGE_M
    if not (lo <= d <= hi):
        hint = (" -- it looks like nm passed as metres" if d > hi else
                " -- it looks like metres converted twice")
        raise ValueError("{}: a film thickness of {:.4g} m is outside [{:g}, {:g}] m{}; lengths are in "
                         "metres.".format(where, d, lo, hi, hint))
    return d


@dataclass(frozen=True)
class BoundaryScattering:
    """Opt-in size-effect damping of a thin conducting film, for DrudeOptical(boundary=...) (v0.16).

    The free-carrier response becomes uniaxial (z = the film normal, the assembler's stack axis):
      eps_xx = eps_yy = eps_b - wp^2 F(kappa, p) / (omega (omega + i gamma)),
                        kappa = d (gamma - i omega) / v_F       (Fuchs-Sondheimer, exact; in_plane=True)
      eps_zz          = eps_b - wp^2 / (omega (omega + i (gamma + A v_F / L)))
                        (Kreibig-type Matthiessen rate; perp_coefficient = A > 0)
    gamma = the bulk rate of the DrudeOptical (float, callable of n or MatthiessenGamma), v_F = hbar k_F /
    m_opt(n) at the LOCAL density (nonlocal_tmm.fermi_velocity; with a KaneOpticalMass this is the Kane
    Fermi velocity, since m_opt = hbar k_F / v_F), d = thickness_m or the extent of the carrier region's z
    axis supplied by the assembler. L (perp_length):
      'film'  : L = 4 V / S = 2 d (the Coronado-Schatz geometric length of a slab),
      'local' : 1 / L(z) = 1 / (2 d) + |dn/dz| / n  -- a HEURISTIC confinement length: the film value where
                the density is flat, the density-gradient length inside an accumulation / depletion layer.
                At an extremum of n(z) it returns the film value (the gradient vanishes). No wall-distance
                term: 1 / (distance to a wall) makes the sheet damping log-divergent for a classical
                profile that is finite at the face.
    A is a model coefficient of order one, default 0 (off). CALIBRATION against the self-consistent quantum
    response (validation/thin_film_damping.py; docs/audit/2026-11-14-v0.16-thin-film-damping-and-background.md),
    6 nm generic gated film, 3e26 m^-3: the RPA out-of-plane loss peak is as narrow as the bulk rate at every
    bias (the confined spectrum adds < ~1e13 rad/s), so NO positive A reproduces its linewidth (the local
    graded film is already broader at A = 0); the RPA's 1200 nm p-pol absorption is 1.2-2.0x the local
    film's at A = 0 and is reached within A <= 3 at one bias of five only (A = 2.2 'film' / 0.5 'local') --
    not a constant. Flat-band films 10 / 20 nm: A = 1.7 / 1.3 ('film'). The out-of-plane rate is
    therefore NOT a model of quantum confinement
    for gated few-nm films (use optics.quantum_film); it is an empirical knob. specularity = 1 with
    perp_coefficient = 0 gives back the bulk Drude in every component (still a tensor).

    Validity: classical motion across the film (k_F d >> 1; ~12 at 6 nm, 3e26 m^-3: indicative only),
    field uniform across the film (d << skin depth), isotropic bulk relaxation time, degenerate gas. Do NOT
    combine with a response that already contains this physics: optics.quantum_film (the sub-band
    spectrum carries the out-of-plane Landau-type damping) or a hydrodynamic layer with GNOR diffusion
    (D > 0 is itself a size-dependent damping)."""
    specularity: float = 0.0
    in_plane: bool = True
    perp_coefficient: float = 0.0
    perp_length: str = "film"
    thickness_m: Optional[float] = None

    def __post_init__(self):
        p = float(self.specularity)
        if not (0.0 <= p <= 1.0):
            raise ValueError("BoundaryScattering: specularity must lie in [0, 1], got {!r}".format(p))
        A = float(self.perp_coefficient)
        if not (np.isfinite(A) and A >= 0.0):
            raise ValueError("BoundaryScattering: perp_coefficient (A) must be finite and >= 0 (a negative "
                             "rate is gain), got {!r}".format(self.perp_coefficient))
        if self.perp_length not in _PERP_LENGTHS:
            raise ValueError("BoundaryScattering: perp_length must be one of {}, got {!r}".format(
                _PERP_LENGTHS, self.perp_length))
        if self.thickness_m is not None and not (np.isfinite(self.thickness_m) and self.thickness_m > 0.0):
            raise ValueError("BoundaryScattering: thickness_m must be finite and > 0 (or None to take the "
                             "carrier region's extent), got {!r}".format(self.thickness_m))
        if self.thickness_m is not None:
            _check_film_thickness(float(self.thickness_m), "BoundaryScattering(thickness_m=...)")

    @property
    def needs_position(self) -> bool:
        """True when the model needs the assembler's z axis (no explicit thickness, or the local length)."""
        return self.thickness_m is None or (self.perp_coefficient > 0.0 and self.perp_length == "local")

    def film_thickness(self, z_axis_m=None, film_thickness_m=None) -> float:
        """d [m]: thickness_m, else the film thickness the assembler supplies (film_thickness_m: the whole
        film when it is split into several carrier regions), else the extent of the
        stack-axis coordinates z_axis_m. A thickness outside [1e-11, 1e-3] m is refused (a length unit
        slip: nm read as metres gives ~1 m and the size effect vanishes silently)."""
        if self.thickness_m is not None:
            return float(self.thickness_m)
        if film_thickness_m is not None:
            return _check_film_thickness(float(film_thickness_m), "BoundaryScattering (film thickness)")
        if z_axis_m is None:
            raise ValueError(
                "BoundaryScattering: the film thickness is unknown -- set thickness_m, or evaluate through "
                "the permittivity assembler (MaterialEpsMap / EffectEpsMap supply the carrier region's z "
                "axis), or call DrudeOptical.eps_tensor(..., z_axis_m=...).")
        z = np.asarray(z_axis_m, dtype=np.float64).ravel()
        d = float(abs(z[-1] - z[0])) if z.size >= 2 else 0.0
        if not (d > 0.0):
            raise ValueError("BoundaryScattering: the z axis has no extent (need >= 2 distinct points).")
        return _check_film_thickness(d, "BoundaryScattering (extent of z_axis_m)")

    def in_plane_factor(self, omega_rad_s: float, gamma_bulk_rad_s, fermi_velocity_m_s, thickness_m: float):
        """F(kappa, p) at kappa = d (gamma - i omega) / v_F (arrays broadcast)."""
        g = np.asarray(gamma_bulk_rad_s, dtype=np.float64)
        v = np.asarray(fermi_velocity_m_s, dtype=np.float64)
        return fuchs_sondheimer_factor(float(thickness_m) * (g - 1j * float(omega_rad_s)) / v,
                                       self.specularity)

    def confinement_length_m(self, n_m3, thickness_m: float, z_axis_m=None):
        """L_eff [m] of the out-of-plane rate: 2 d ('film'), or 1 / (1/(2 d) + |dn/dz| / n) along the LAST
        axis of n_m3, whose coordinates are z_axis_m ('local')."""
        L_film = 2.0 * float(thickness_m)
        n = np.asarray(n_m3, dtype=np.float64)
        if self.perp_length == "film":
            return np.full(n.shape, L_film) if n.ndim else L_film
        if z_axis_m is None:
            raise ValueError("BoundaryScattering(perp_length='local'): the density gradient needs the z axis "
                             "of the density grid (supplied by the assembler, or z_axis_m=).")
        z = np.asarray(z_axis_m, dtype=np.float64).ravel()
        if n.ndim == 0 or n.shape[-1] != z.size or z.size < 2:
            raise ValueError("BoundaryScattering(perp_length='local'): the last axis of the density {} must "
                             "match the z axis ({} points, >= 2)".format(n.shape, z.size))
        dn = np.gradient(n, z, axis=-1)
        return 1.0 / (1.0 / L_film + np.abs(dn) / np.maximum(n, _N_FLOOR))

    def perp_rate(self, n_m3, fermi_velocity_m_s, thickness_m: float, z_axis_m=None):
        """Gamma_s = A v_F / L_eff [rad/s] (0.0 when perp_coefficient = 0)."""
        if self.perp_coefficient == 0.0:
            return 0.0
        L = self.confinement_length_m(n_m3, thickness_m, z_axis_m)
        return float(self.perp_coefficient) * np.asarray(fermi_velocity_m_s, dtype=np.float64) / L
