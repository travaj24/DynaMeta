"""Free-carrier scattering kernels of degenerate polycrystalline conducting oxides.

What this module adds to materials.scattering, and why
-------------------------------------------------------
materials.scattering already carries the Kane optical mass (KaneOpticalMass), a Matthiessen damping sum
(MatthiessenGamma) whose ionized-impurity channel is a CALIBRATED power law, and ONE shared tau(n) law for
optics and transport (ScatteringModel). Heavily doped ZnO, CdO and In2O3 films need three things those
cannot express:

  1. IonizedImpurityScattering -- the ionized-impurity-limited mobility of a DEGENERATE, NONPARABOLIC
     band from first principles of the screened Coulomb potential (Brooks-Herring-Dingle with the
     Zawadzki nonparabolic correction in the form of Pisarkiewicz et al.), with no calibration
     prefactor. ResolvedMatthiessenGamma (a MatthiessenGamma subclass) puts it in the Matthiessen sum in
     place of the power-law channel.
  2. MasettiMobility -- the empirical intra-grain ("physical limit") mobility law fitted to the best
     single-crystal / epitaxial films (Masetti form), e.g. the ZnO fit of Ellmer and Mientus.
  3. SetoGrainBarrier + PolycrystallineScattering -- the grain-boundary (Seto) barrier mobility that
     makes the DC (Hall) mobility of a polycrystalline film smaller than the intra-grain mobility seen by
     near-infrared optics, and a ScatteringModel subclass that keeps the two apart: the optical Drude
     damping is the intra-grain rate, the DC drift mobility is the intra-grain mobility in series with the
     grain-boundary mobility.

Plus two small callables used by the presets so that a Material's repr stays deterministic (a lambda
repr carries a memory address): OpticalMobilityDamping (gamma(n) = e / (m_opt(n) mu(n))) and
TabulatedMobility (measured mu(n), log-density interpolation).

Units are SI throughout (m^-3, m^2/(V s), rad/s, J); literature formulae quoted in cm units are converted
at the boundary. Everything is pure numpy and devsim-free.

Frequency dependence (not modelled here, on purpose). Ionized-impurity scattering is weaker at optical
frequencies than at DC (Gerlach-type omega^(-3/2) tail). In doped ZnO films a plain-Drude fit of near-infrared
ellipsometry returns a mobility ~30 % above the DC / far-IR values (Knoops et al., J. Vac. Sci. Technol. A 33,
021509 (2015), Table II, 7.5e20 cm^-3: 38 cm^2/(V s) plain Drude against 28 extended Drude, 29 FTIR, 30 Hall;
the authors prefer the extended Drude model, whose damping falls with frequency). The kernels below are DC laws.
The TCO presets therefore take the optical damping from near-infrared ellipsometry (an optical mobility)
rather than from these DC laws; see materials.tco_presets and docs/enz_presets.md.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional, Tuple, Union

import numpy as np

from dynameta.constants import EPS0, HBAR, KB, Q_E, T_REF
from dynameta.materials.scattering import KaneOpticalMass, MatthiessenGamma, ScatteringModel

__all__ = ["IonizedImpurityScattering", "MasettiMobility", "SetoGrainBarrier", "ResolvedMatthiessenGamma",
           "PolycrystallineScattering", "OpticalMobilityDamping", "TabulatedMobility", "series_mobility"]

_CM3 = 1.0e6            # cm^-3 -> m^-3
_CM2VS = 1.0e-4         # cm^2/(V s) -> m^2/(V s)
_N_FLOOR = 1.0e10       # density floor (as materials.scattering): avoids 0^(1/3) and 1/0 at empty nodes
_H = 2.0 * math.pi * HBAR


def _as_kane_mass(band, where: str) -> KaneOpticalMass:
    """A KaneOpticalMass from a mass [kg] (parabolic), a KaneOpticalMass or a materials.KaneBand."""
    if isinstance(band, KaneOpticalMass):
        if band.legacy:
            raise ValueError("{}: the legacy KaneOpticalMass closure has no Fermi level; use legacy=False".format(
                where))
        return band
    if hasattr(band, "m0_kg") and hasattr(band, "alpha_per_eV"):
        return KaneOpticalMass.from_band(band)
    if isinstance(band, (int, float)) and math.isfinite(float(band)) and float(band) > 0.0:
        return KaneOpticalMass(m0_kg=float(band), alpha_eV=0.0)
    raise TypeError("{}: band must be a mass [kg] > 0, a KaneOpticalMass or a KaneBand, got {!r}".format(
        where, band))


def _positive(v, name: str, where: str) -> float:
    x = float(v)
    if not (math.isfinite(x) and x > 0.0):
        raise ValueError("{}: {} must be finite and > 0, got {!r}".format(where, name, v))
    return x


def _mobility_values(mu, n, where: str):
    """mu(n) as a float array (a float or a callable of n), validated finite and > 0."""
    v = np.asarray(mu(n), dtype=np.float64) if callable(mu) else np.full(np.shape(n), float(mu))
    if not np.all(np.isfinite(v)) or np.any(v <= 0.0):
        raise ValueError("{}: a mobility must be finite and > 0 (m^2/(V s)); got a non-positive or non-finite "
                         "value".format(where))
    return v


def series_mobility(*mobilities):
    """1 / sum_i (1 / mu_i) -- mobilities limited by processes acting in series (Matthiessen's rule for
    rates of the same carriers, or a chain of grains and grain boundaries). Arrays broadcast; each > 0."""
    if not mobilities:
        raise ValueError("series_mobility: give at least one mobility")
    inv = 0.0
    for mu in mobilities:
        m = np.asarray(mu, dtype=np.float64)
        if not np.all(np.isfinite(m)) or np.any(m <= 0.0):
            raise ValueError("series_mobility: every mobility must be finite and > 0")
        inv = inv + 1.0 / m
    out = 1.0 / inv
    return float(out) if np.ndim(out) == 0 else out


# =====================================================================================================
# 1. Ionized-impurity scattering of a degenerate nonparabolic band
# =====================================================================================================

@dataclass(frozen=True)
class IonizedImpurityScattering:
    """Ionized-impurity-limited mobility of a degenerate electron gas in a Kane (nonparabolic) band.

    Formula (SI; Ellmer and Mientus, Thin Solid Films 516, 4620 (2008), Eqs. (1)-(2), who give it for ZnO
    after W. Zawadzki (nonparabolic Brooks-Herring-Dingle) as refined by T. Pisarkiewicz et al.):

        mu_ii   = 3 (eps_r eps0)^2 h^3 n / (Z^2 m*^2 e^3 N_i)  /  F_np(xi_d)
        xi_d    = (3 pi^2)^(1/3) eps_r eps0 h^2 n^(1/3) / (m* e^2)          (= (2 k_F / q_TF)^2)
        F_np    = [1 + (4 xi_np / xi_d)(1 - xi_np / 8)] ln(1 + xi_d) - xi_d / (1 + xi_d)
                  - 2 xi_np (1 - 5 xi_np / 16),          xi_np = 1 - m0* / m*

    with m* = m*(n) = m0* (1 + 2 alpha E_F) the Fermi-surface mass of the band (KaneOpticalMass), h the
    Planck constant, N_i the ionized-impurity density and Z its charge. alpha = 0 gives xi_np = 0 and the
    parabolic Brooks-Herring-Dingle result 24 pi^3 eps^2 hbar^3 n / (e^3 m*^2 Z^2 N_i [ln(1 + xi) -
    xi/(1 + xi)]) exactly (h^3 = 8 pi^3 hbar^3).

    Fields:
      eps_r          : static relative permittivity of the host (screens the impurity potential).
      band           : a mass [kg] (parabolic), KaneOpticalMass or KaneBand -- the SAME band as the optics.
      Z              : impurity charge (1 = singly ionized donor; Bikowski and Ellmer, J. Appl. Phys. 116,
                       143704 (2014) model dopant clustering in ZnO as an effective Z > 1).
      impurity_ratio : N_i / n. None (default) = 1 / Z (fully ionized, uncompensated donors of charge Z, so
                       Z N_i = n). A compensated film has N_i > n / Z.

    Validity: degenerate gas (E_F >> kT; T does not enter), elastic scattering by uncorrelated point
    charges in the Born approximation with Thomas-Fermi screening; a DC (relaxation-time) mobility. It is
    an UPPER bound on the intra-grain mobility of real films (clustering, compensation, neutral impurities
    and phonons lower it further) and it does NOT describe the weaker damping at optical frequencies.
    Callable: __call__(n) is the momentum-relaxation RATE e / (m*(n) mu_ii(n)) [rad/s], so the object plugs
    into DrudeOptical(gamma_rad_s=...), ScatteringModel(one_over_tau=...) or ResolvedMatthiessenGamma."""
    eps_r: float
    band: object
    Z: float = 1.0
    impurity_ratio: Optional[float] = None

    def __post_init__(self):
        _positive(self.eps_r, "eps_r", "IonizedImpurityScattering")
        _positive(self.Z, "Z", "IonizedImpurityScattering")
        if self.impurity_ratio is not None:
            _positive(self.impurity_ratio, "impurity_ratio", "IonizedImpurityScattering")
        _as_kane_mass(self.band, "IonizedImpurityScattering")

    def _mass(self):
        return _as_kane_mass(self.band, "IonizedImpurityScattering")

    def _ratio(self) -> float:
        return (1.0 / float(self.Z)) if self.impurity_ratio is None else float(self.impurity_ratio)

    def screening_parameter(self, n_m3):
        """xi_d(n) = (3 pi^2)^(1/3) eps h^2 n^(1/3) / (m*(n) e^2) (dimensionless)."""
        n = np.maximum(np.asarray(n_m3, dtype=np.float64), _N_FLOOR)
        m = np.asarray(self._mass()(n), dtype=np.float64)
        eps = float(self.eps_r) * EPS0
        return (3.0 * math.pi ** 2) ** (1.0 / 3.0) * eps * _H ** 2 * np.cbrt(n) / (m * Q_E ** 2)

    def screening_function(self, n_m3):
        """F_np(xi_d, xi_np) of the class docstring (the parabolic ln(1 + xi) - xi/(1 + xi) at alpha = 0)."""
        n = np.maximum(np.asarray(n_m3, dtype=np.float64), _N_FLOOR)
        km = self._mass()
        xi = self.screening_parameter(n)
        xnp = 1.0 - km.m0_kg / np.asarray(km(n), dtype=np.float64)
        return ((1.0 + (4.0 * xnp / xi) * (1.0 - xnp / 8.0)) * np.log1p(xi) - xi / (1.0 + xi)
                - 2.0 * xnp * (1.0 - 5.0 * xnp / 16.0))

    def mobility_m2Vs(self, n_m3):
        """mu_ii(n) [m^2/(V s)]."""
        n = np.maximum(np.asarray(n_m3, dtype=np.float64), _N_FLOOR)
        m = np.asarray(self._mass()(n), dtype=np.float64)
        eps = float(self.eps_r) * EPS0
        Ni = self._ratio() * n
        mu = 3.0 * eps ** 2 * _H ** 3 * n / (float(self.Z) ** 2 * m ** 2 * Q_E ** 3 * Ni) / self.screening_function(n)
        return float(mu) if np.ndim(mu) == 0 else mu

    def __call__(self, n_m3):
        n = np.maximum(np.asarray(n_m3, dtype=np.float64), _N_FLOOR)
        out = Q_E / (np.asarray(self._mass()(n), dtype=np.float64) * self.mobility_m2Vs(n))
        return float(out) if np.ndim(out) == 0 else out


# =====================================================================================================
# 2. Empirical intra-grain mobility law (Masetti form)
# =====================================================================================================

@dataclass(frozen=True)
class MasettiMobility:
    """Empirical mobility law of the Masetti form (G. Masetti, M. Severi, S. Solmi, IEEE Trans. Electron
    Devices 30, 764 (1983)), in the parameterization of Ellmer and Mientus (Thin Solid Films 516, 4620
    (2008), Eq. (3)):

        mu(n) = mu_min + (mu_max - mu_min) / (1 + (n / n_ref1)^alpha1) - mu_1 / (1 + (n_ref2 / n)^alpha2)

    Parameters in SI (m^2/(V s), m^-3). Callable: __call__(n) -> mu [m^2/(V s)], scalar or array. Use
    zno_ellmer_mientus_2008() for the ZnO fit (their Table 2): the mobility of the best single-crystal and
    epitaxial ZnO films, i.e. an intra-grain (no grain boundary) DC limit including whatever clustering /
    compensation those films had."""
    mu_max_m2Vs: float
    mu_min_m2Vs: float
    mu_1_m2Vs: float
    n_ref1_m3: float
    alpha1: float
    n_ref2_m3: float
    alpha2: float

    def __post_init__(self):
        for nm in ("mu_max_m2Vs", "mu_min_m2Vs", "n_ref1_m3", "n_ref2_m3", "alpha1", "alpha2"):
            _positive(getattr(self, nm), nm, "MasettiMobility")
        if not (math.isfinite(self.mu_1_m2Vs) and 0.0 <= self.mu_1_m2Vs < self.mu_min_m2Vs):
            raise ValueError("MasettiMobility: mu_1 must lie in [0, mu_min) so that mu(n) stays > 0 "
                             "(mu -> mu_min - mu_1 at high density)")
        if self.mu_max_m2Vs < self.mu_min_m2Vs:
            raise ValueError("MasettiMobility: mu_max must be >= mu_min")

    @classmethod
    def zno_ellmer_mientus_2008(cls) -> "MasettiMobility":
        """ZnO column of Ellmer and Mientus, Thin Solid Films 516, 4620 (2008), Table 2 (accepted manuscript,
        HZB repository): mu_max = 210, mu_min = 55, mu_min - mu_1 = 5 cm^2/(V s), n_ref1 = 4e17 cm^-3,
        alpha1 = 1, n_ref2 = 6e20 cm^-3, alpha2 = 2. Values: 54.3 / 50.3 / 34.6 / 18.3 cm^2/(V s) at
        1e20 / 2e20 / 5e20 / 1e21 cm^-3. CAVEAT (the paper's own): single-crystal ZnO data exist only up to
        ~8e19 cm^-3; the high-density (clustering) branch is set "in analogy to the silicon data", so above
        ~1e20 cm^-3 this law is an extrapolation (best films measured: ~46 cm^2/(V s) Hall at 6e20 cm^-3,
        Cornelius et al., Appl. Phys. Lett. 94, 042103 (2009), against 30 from this law)."""
        return cls(mu_max_m2Vs=210.0 * _CM2VS, mu_min_m2Vs=55.0 * _CM2VS, mu_1_m2Vs=50.0 * _CM2VS,
                   n_ref1_m3=4.0e17 * _CM3, alpha1=1.0, n_ref2_m3=6.0e20 * _CM3, alpha2=2.0)

    @classmethod
    def sno2_ellmer_mientus_2008(cls) -> "MasettiMobility":
        """SnO2 column of the same Table 2: mu_max = 250, mu_min = 50, mu_min - mu_1 = 10 cm^2/(V s), n_ref1 =
        2e18 cm^-3, alpha1 = 1, n_ref2 = 6e20 cm^-3, alpha2 = 2 (fitted "with similar parameters as used for
        zinc oxide" for lack of single-crystal data: a weak constraint)."""
        return cls(mu_max_m2Vs=250.0 * _CM2VS, mu_min_m2Vs=50.0 * _CM2VS, mu_1_m2Vs=40.0 * _CM2VS,
                   n_ref1_m3=2.0e18 * _CM3, alpha1=1.0, n_ref2_m3=6.0e20 * _CM3, alpha2=2.0)

    @classmethod
    def ito_ellmer_mientus_2008(cls) -> "MasettiMobility":
        """ITO column of the same Table 2: mu_max = 210, mu_min = 55, mu_min - mu_1 = 5 cm^2/(V s), n_ref1 =
        1.5e18 cm^-3, alpha1 = 1, n_ref2 = 2e21 cm^-3, alpha2 = 2 (single-crystal In2O3 data are rarer still;
        the same extrapolation caveat applies)."""
        return cls(mu_max_m2Vs=210.0 * _CM2VS, mu_min_m2Vs=55.0 * _CM2VS, mu_1_m2Vs=50.0 * _CM2VS,
                   n_ref1_m3=1.5e18 * _CM3, alpha1=1.0, n_ref2_m3=2.0e21 * _CM3, alpha2=2.0)

    def __call__(self, n_m3):
        n = np.maximum(np.asarray(n_m3, dtype=np.float64), _N_FLOOR)
        mu = (self.mu_min_m2Vs + (self.mu_max_m2Vs - self.mu_min_m2Vs) / (1.0 + (n / self.n_ref1_m3) ** self.alpha1)
              - self.mu_1_m2Vs / (1.0 + (self.n_ref2_m3 / n) ** self.alpha2))
        return float(mu) if np.ndim(mu) == 0 else mu


# =====================================================================================================
# 3. Grain-boundary (Seto) barrier mobility
# =====================================================================================================

@dataclass(frozen=True)
class SetoGrainBarrier:
    """Grain-boundary-limited mobility of a polycrystalline film (J. Y. W. Seto, J. Appl. Phys. 46, 5247
    (1975)), in the form Ellmer and Mientus use for ZnO (Thin Solid Films 516, 4620 (2008), Eqs. (6)-(7c)):

        mu_gb  = mu_0 exp(-E_b / kT),         mu_0 = e L / sqrt(2 pi m kT)        (thermionic emission)
        E_b    = e^2 Q_t^2 / (8 eps_r eps0 n)   for L n > Q_t   (partly depleted grains)
               = e^2 L^2 n / (8 eps_r eps0)     for L n <= Q_t  (fully depleted grains)

    Q_t = trap_density_m2 (charge-trapping states per boundary area), L = grain_size_m, m = mass_kg (a
    float or a callable of n, e.g. KaneOpticalMass), T_K the temperature. The two branches meet at
    n = Q_t / L, where the barrier is largest. Literature Q_t for ZnO films (Ellmer and Mientus, Tables 3-4):
    ~1.3e13 cm^-2 epitaxial ZnO:Al (sputter), ~3e13 cm^-2 polycrystalline ZnO:Al (reactive sputter), 5e12
    cm^-2 epitaxial ZnO:Al,Ga (PLD); grain sizes ~20-50 nm for sputtered films.

    Validity (stated, not hidden): the thermionic, non-degenerate emission over the barrier of Seto's
    model. In degenerately doped films the barrier is thin and low (E_b ~ 0.05 eV at 5e20 cm^-3 for Q_t =
    3e13 cm^-2), carriers at the Fermi level sit E_F - E_c above the band edge (0.3 eV in ZnO at 1e20
    cm^-3, more than E_b ~ 0.23 eV there) and field emission (tunnelling) adds a parallel channel, so this
    mu_gb is a LOWER bound -- strongly so below ~2e20 cm^-3 in ZnO, where Q_t = 3e13 cm^-2 gives mu_gb <
    0.1 cm^2/(V s) against 16 cm^2/(V s) Hall measured at 0.92e20 cm^-3 in ALD AZO with Q_t ~ 2-4e13
    cm^-2 (Romanyuk et al., Acta Phys. Pol. A 129, A-36 (2016), Table II). Trap densities fitted with this
    same formula (the literature values above) partly absorb that; do not mix them with a degenerate
    barrier model."""
    trap_density_m2: float
    grain_size_m: float
    eps_r: float
    mass_kg: object
    T_K: float = T_REF

    def __post_init__(self):
        if not (math.isfinite(self.trap_density_m2) and self.trap_density_m2 >= 0.0):
            raise ValueError("SetoGrainBarrier: trap_density_m2 must be finite and >= 0 (m^-2)")
        g = _positive(self.grain_size_m, "grain_size_m", "SetoGrainBarrier")
        if not (1e-10 <= g <= 1e-4):
            raise ValueError("SetoGrainBarrier: grain_size_m = {:.4g} m is outside [1e-10, 1e-4] m -- lengths are "
                             "in metres".format(g))
        _positive(self.eps_r, "eps_r", "SetoGrainBarrier")
        _positive(self.T_K, "T_K", "SetoGrainBarrier")
        if not callable(self.mass_kg):
            _positive(self.mass_kg, "mass_kg", "SetoGrainBarrier")

    def _m(self, n):
        return np.asarray(self.mass_kg(n), dtype=np.float64) if callable(self.mass_kg) else float(self.mass_kg)

    def barrier_height_J(self, n_m3):
        """E_b(n) [J]."""
        n = np.maximum(np.asarray(n_m3, dtype=np.float64), _N_FLOOR)
        Qt, L = float(self.trap_density_m2), float(self.grain_size_m)
        eps = float(self.eps_r) * EPS0
        partly = Q_E ** 2 * Qt ** 2 / (8.0 * eps * n)
        full = Q_E ** 2 * L ** 2 * n / (8.0 * eps)
        out = np.where(L * n > Qt, partly, full)
        return float(out) if np.ndim(out) == 0 else out

    def mobility_m2Vs(self, n_m3):
        """mu_gb(n) [m^2/(V s)]."""
        n = np.maximum(np.asarray(n_m3, dtype=np.float64), _N_FLOOR)
        kT = KB * float(self.T_K)
        mu0 = Q_E * float(self.grain_size_m) / np.sqrt(2.0 * math.pi * self._m(n) * kT)
        out = mu0 * np.exp(-self.barrier_height_J(n) / kT)
        return float(out) if np.ndim(out) == 0 else out

    __call__ = mobility_m2Vs


# =====================================================================================================
# 4. Matthiessen sum with the resolved ionized-impurity channel
# =====================================================================================================

@dataclass(frozen=True)
class ResolvedMatthiessenGamma(MatthiessenGamma):
    """MatthiessenGamma whose ionized-impurity channel is the resolved IonizedImpurityScattering rate
    instead of the calibrated power law (bh_prefactor_rad_s):

        Gamma(n) = optical_dc_ratio * (gamma_const + gamma_phonon(T) + e / (m*(n) mu_ii(n)))

    Every other channel, the temperature state and optical_dc_ratio are MatthiessenGamma's own (inherited,
    not copied). ionized_impurity=None gives MatthiessenGamma exactly; setting both ionized_impurity and
    bh_prefactor_rad_s > 0 is refused (the impurity channel would be counted twice)."""
    ionized_impurity: Optional[IonizedImpurityScattering] = None

    def __post_init__(self):
        super().__post_init__()
        if self.ionized_impurity is not None:
            if not isinstance(self.ionized_impurity, IonizedImpurityScattering):
                raise TypeError("ResolvedMatthiessenGamma: ionized_impurity must be an IonizedImpurityScattering")
            if self.bh_prefactor_rad_s > 0.0:
                raise ValueError("ResolvedMatthiessenGamma: set ionized_impurity OR bh_prefactor_rad_s, not both "
                                 "(the ionized-impurity channel would be counted twice)")

    def _ii(self, n):
        if self.ionized_impurity is None:
            return super()._ii(n)
        return np.asarray(self.ionized_impurity(n), dtype=np.float64)


# =====================================================================================================
# 5. Optical damping from an optical mobility; measured mobility tables
# =====================================================================================================

@dataclass(frozen=True)
class OpticalMobilityDamping:
    """Drude damping from an OPTICAL mobility: gamma(n) = e / (m_opt(n) mu_opt(n)) [rad/s].

    mu_opt is the mobility that a Drude fit of optical spectra returns (in polycrystalline films the
    intra-grain mobility at that frequency, not the Hall mobility). mobility_m2Vs: a float or a callable of
    n (TabulatedMobility, MasettiMobility, IonizedImpurityScattering.mobility_m2Vs, ...). mass: a mass [kg],
    KaneOpticalMass or KaneBand -- the optical mass of the same band as DrudeOptical.m_opt_kg, so that a
    constant mu_opt gives a damping that follows the mass. Callable of n (scalar or array)."""
    mobility_m2Vs: object
    mass: object

    def __post_init__(self):
        if not callable(self.mobility_m2Vs):
            _positive(self.mobility_m2Vs, "mobility_m2Vs", "OpticalMobilityDamping")
        _as_kane_mass(self.mass, "OpticalMobilityDamping")

    def mobility(self, n_m3):
        n = np.maximum(np.asarray(n_m3, dtype=np.float64), _N_FLOOR)
        return _mobility_values(self.mobility_m2Vs, n, "OpticalMobilityDamping")

    def __call__(self, n_m3):
        n = np.maximum(np.asarray(n_m3, dtype=np.float64), _N_FLOOR)
        m = np.asarray(_as_kane_mass(self.mass, "OpticalMobilityDamping")(n), dtype=np.float64)
        out = Q_E / (m * self.mobility(n))
        return float(out) if np.ndim(out) == 0 else out


@dataclass(frozen=True)
class TabulatedMobility:
    """Measured mobility mu(n) [m^2/(V s)] at densities n_m3 [m^-3] (strictly increasing, > 0), interpolated
    linearly in log10(n). Outside the table: extrapolate='hold' (default -- the end values; a gated film's
    depleted or accumulated nodes leave any measured table, and the preset factories check the
    construction density against the table themselves) or 'raise'. source: free-text provenance (part of
    the repr, so a different table is a different cache key)."""
    n_m3: Tuple[float, ...]
    mobility_m2Vs: Tuple[float, ...]
    extrapolate: str = "hold"
    source: str = ""

    def __post_init__(self):
        n = np.asarray(self.n_m3, dtype=np.float64).ravel()
        mu = np.asarray(self.mobility_m2Vs, dtype=np.float64).ravel()
        if n.size < 2 or n.size != mu.size:
            raise ValueError("TabulatedMobility: n_m3 and mobility_m2Vs need the same length >= 2")
        if n[0] <= 0.0 or not np.all(np.diff(n) > 0.0):
            raise ValueError("TabulatedMobility: n_m3 must be > 0 and strictly increasing")
        if not np.all(np.isfinite(mu)) or np.any(mu <= 0.0):
            raise ValueError("TabulatedMobility: mobilities must be finite and > 0")
        if self.extrapolate not in ("hold", "raise"):
            raise ValueError("TabulatedMobility: extrapolate must be 'hold' or 'raise'")
        object.__setattr__(self, "n_m3", tuple(float(v) for v in n))
        object.__setattr__(self, "mobility_m2Vs", tuple(float(v) for v in mu))

    @property
    def density_range_m3(self) -> Tuple[float, float]:
        return self.n_m3[0], self.n_m3[-1]

    def __call__(self, n_m3):
        n_tab = np.asarray(self.n_m3)
        n = np.asarray(n_m3, dtype=np.float64)
        if self.extrapolate == "raise" and (float(np.min(n)) < n_tab[0] * (1.0 - 1e-12)
                                            or float(np.max(n)) > n_tab[-1] * (1.0 + 1e-12)):
            raise ValueError("TabulatedMobility: density outside the table [{:.4g}, {:.4g}] m^-3".format(
                n_tab[0], n_tab[-1]))
        x = np.log10(np.clip(np.maximum(n, _N_FLOOR), n_tab[0], n_tab[-1]))
        out = np.interp(x, np.log10(n_tab), np.asarray(self.mobility_m2Vs))
        return float(out) if np.ndim(out) == 0 else out


# =====================================================================================================
# 6. Optical (intra-grain) vs DC (Hall) mobility of a polycrystalline film
# =====================================================================================================

@dataclass(frozen=True)
class PolycrystallineScattering(ScatteringModel):
    """ScatteringModel for a polycrystalline film: the optical Drude damping is the intra-grain rate
    (one_over_tau, as in ScatteringModel), the DC drift mobility is the intra-grain DC mobility in series
    with a grain-boundary mobility:

        gamma_optical(n) = one_over_tau(n)                                      (inherited)
        mu_DC(n)         = [1 / mu_ig(n) + 1 / mu_gb(n)]^-1
        mu_ig(n)         = intragrain_mobility(n)  if given,  else ScatteringModel's q / (m_cond 1/tau_DC)

    Why mu_ig may be given separately. ScatteringModel's premise is one tau law for optics and transport.
    In doped oxides the near-infrared Drude damping is WEAKER than the DC intra-grain rate (ionized-impurity
    scattering falls off with frequency; see the module docstring), so a damping calibrated on near-infrared
    spectra and a DC mobility calibrated on Hall data are two measured inputs, not one. intragrain_mobility
    (float or callable, m^2/(V s)) carries the DC one; leave it None to keep ScatteringModel's single law.
    grain_boundary: a SetoGrainBarrier, a float or a callable mu_gb(n), or None (single crystal / epitaxial:
    mu_DC = mu_ig). The produced mobility is a drift mobility times hall_factor (ScatteringModel's
    convention). Attach with Material(scattering=...)."""
    intragrain_mobility: Optional[object] = None
    grain_boundary: Optional[object] = None

    def __post_init__(self):
        if self.intragrain_mobility is not None and not callable(self.intragrain_mobility):
            _positive(self.intragrain_mobility, "intragrain_mobility", "PolycrystallineScattering")
        if self.grain_boundary is not None and not callable(self.grain_boundary):
            _positive(self.grain_boundary, "grain_boundary", "PolycrystallineScattering")

    def mobility_of_n(self):
        """mu_DC(n) as a frozen-dataclass callable (picklable, deterministic repr)."""
        return _PolycrystallineMobility(base=super().mobility_of_n(), intragrain_mobility=self.intragrain_mobility,
                                        grain_boundary=self.grain_boundary, hall_factor=float(self.hall_factor))


@dataclass(frozen=True)
class _PolycrystallineMobility:
    """PolycrystallineScattering.mobility_of_n: r_H [1 / mu_ig(n) + 1 / mu_gb(n)]^-1."""
    base: object
    intragrain_mobility: Optional[object]
    grain_boundary: Optional[object]
    hall_factor: float

    def __call__(self, n_m3):
        n = np.asarray(n_m3, dtype=np.float64)
        rh, gb = self.hall_factor, self.grain_boundary
        if self.intragrain_mobility is None:
            ig = np.asarray(self.base(n), dtype=np.float64) / rh          # drift (base already carries r_H)
        else:
            ig = _mobility_values(self.intragrain_mobility, n, "PolycrystallineScattering.intragrain_mobility")
        if gb is None:
            tot = ig
        else:
            tot = series_mobility(ig, _mobility_values(gb, n, "PolycrystallineScattering.grain_boundary"))
        out = rh * np.asarray(tot, dtype=np.float64)
        return float(out) if np.ndim(out) == 0 else out


#: A mobility argument: a float [m^2/(V s)] or a callable of the density n [m^-3].
MobilityLike = Union[float, Callable[[np.ndarray], np.ndarray]]
