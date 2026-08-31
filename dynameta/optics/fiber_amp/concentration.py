"""Concentration-quenching and photo-degradation effects for heavily doped fibers (docs sec.6),
all OPT-IN: with concentration=None the amplifier is byte-identical to the ideal model.

Three mechanisms, bundled in ConcentrationModel and consumed by the steady_state solver:

  * Homogeneous cooperative UPCONVERSION (coefficient C_up [m^3/s]): two neighbouring excited
    ions interact, one relaxing to ground -- an inversion-dependent loss C_up N2^2 that clamps
    the metastable population. Enters the metastable-fraction balance as the quadratic term
    (already in rare_earth.metastable_fraction); this module supplies C_up.
  * PAIR-INDUCED QUENCHING (Delevaque PTL 1993; Nilsson/Jaskorzynska/Blixt PTL 1993): a fraction
    of ions sit in tightly-spaced pairs whose double-excited state up-converts instantly, so a
    pair can hold at most one excitation. Two conventions (pair_convention):
      "dark" (default): EVERY paired ion is treated as permanently ground-state -- an
        UNBLEACHABLE absorption sigma_a Gamma (pair_fraction n_t) at every wavelength, the
        residual loss surviving infinite pump. This is the PESSIMISTIC BOUND: it overstates the
        canonical Delevaque pair by ~2x at the pump and more at the signal, because the real
        pair keeps one member excited under saturation.
      "delevaque": the literature-standard convention ("one ion per pair completely quenched",
        Nilsson 1993). pair_fraction is then the fraction of ions RESIDING IN PAIRS (2k); half
        of them are dark, the other half return to the active pool. Delevaque-convention
        pair_fraction = 2f reproduces dark-convention pair_fraction = f exactly.
    Measured pair-ion fractions in Al-codoped silica at ~1-2e19 cm^-3: ~4-8% commercial
    (Wagener OL 19, 347 (1994); Kir'yanov JQE 49, 511 (2013)), ~1.6% for selected low-quench
    fiber (Le Gouet JLT 37, 3611 (2019)).
  * Yb PHOTODARKENING: colour-centre formation drives an EQUILIBRIUM excess background loss.
    The equilibrium loss is ~LINEAR in the inversion nbar2 (Jetschke et al., Opt. Express 15,
    14838 (2007); Jauregui/Stihler/Limpert, Adv. Opt. Photon. 12, 429 (2020)) and ~quadratic in
    total Yb density ACROSS fibers (Taccheo et al., Opt. Express 19, 19340 (2011)). The steep
    inversion power laws in the literature -- exponent 4.3 (Jetschke & Roepke, Opt. Lett. 34,
    109 (2009)) or 7 (Koponen et al., Appl. Opt. 47, 1247 (2008)) -- govern the photodarkening
    RATE (time-to-equilibrium), NOT the equilibrium loss; using them for the equilibrium
    under-predicts by 10^2-10^3 at amplifier inversions. Modelled here as an inversion-dependent
    gray loss pd_loss_per_m * nbar2^pd_exponent along the fiber, with pd_exponent = 1 the
    physically correct equilibrium default (changed from the legacy rate-law 7 in the 2026-08-31
    literature audit).

Pure numpy; SI units. docs/fiber_amp_model_spec.md sec.6.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["ConcentrationModel", "erbium_upconversion", "ytterbium_photodarkening"]

_LN10_10 = float(np.log(10.0) / 10.0)          # dB/m -> 1/m


@dataclass(frozen=True)
class ConcentrationModel:
    """Bundle of concentration/degradation parameters (all default to the no-effect value, so an
    all-default model is a no-op). c_up_m3_s = homogeneous upconversion coefficient;
    pair_fraction = quenched-pair parameter, interpreted per pair_convention (see module
    docstring): "dark" (default, pessimistic bound) treats pair_fraction n_t ions as permanently
    dark; "delevaque" (literature-standard) treats pair_fraction as the ions-in-pairs fraction,
    half dark and half active. pd_loss_per_m = photodarkening EQUILIBRIUM excess-loss scale
    [1/m] at full inversion; pd_exponent = the equilibrium inversion law (1 = the measured
    linear law; the legacy 7 is the Koponen RATE exponent and under-predicts equilibrium)."""
    c_up_m3_s: float = 0.0
    pair_fraction: float = 0.0
    pd_loss_per_m: float = 0.0
    pd_exponent: float = 1.0
    pair_convention: str = "dark"

    def __post_init__(self):
        if not (0.0 <= self.pair_fraction < 1.0):
            raise ValueError("ConcentrationModel: pair_fraction must be in [0, 1)")
        if self.pair_convention not in ("dark", "delevaque"):
            raise ValueError("ConcentrationModel: pair_convention must be 'dark' or "
                             "'delevaque', got %r" % (self.pair_convention,))
        for nm, v in (("c_up_m3_s", self.c_up_m3_s), ("pd_loss_per_m", self.pd_loss_per_m),
                      ("pd_exponent", self.pd_exponent)):
            if v < 0.0:
                raise ValueError("ConcentrationModel: {} must be >= 0".format(nm))

    @property
    def is_identity(self) -> bool:
        """True when the model changes nothing (all mechanisms off) -> byte-identical solve."""
        return (self.c_up_m3_s == 0.0 and self.pair_fraction == 0.0
                and self.pd_loss_per_m == 0.0)

    @property
    def _dark_fraction(self) -> float:
        """Fraction of n_t that is permanently dark under the active convention."""
        if self.pair_convention == "delevaque":
            return 0.5 * self.pair_fraction
        return self.pair_fraction

    def active_density(self, n_t_m3: float) -> float:
        """Density of gain-contributing ions = (1 - dark_fraction) n_t. Under "delevaque" the
        excited member of each pair stays in the active pool, so only half the paired ions are
        removed; under "dark" all of them are."""
        return float(n_t_m3 * (1.0 - self._dark_fraction))

    def dark_density(self, n_t_m3: float) -> float:
        """Density of quenched dark ions (always absorbing, at every wavelength)."""
        return float(n_t_m3 * self._dark_fraction)

    def photodarkening_loss_per_m(self, nbar2):
        """Equilibrium photodarkening excess loss [1/m] at local inversion nbar2:
        pd_loss_per_m * nbar2^pd_exponent (broadband/gray). SHAPE-PRESERVING: an array nbar2
        always yields an array (the old scalar-0.0 early return crashed the transient path for
        any ConcentrationModel without photodarkening -- caught by the audit-S3-38 gate)."""
        n = np.clip(np.asarray(nbar2, dtype=np.float64), 0.0, 1.0)
        if self.pd_loss_per_m <= 0.0:
            out = np.zeros_like(n)
        else:
            out = self.pd_loss_per_m * np.power(n, self.pd_exponent)
        return out if out.ndim else float(out)


# ---- literature-anchored factories (docs sec.6) --------------------------------------------

def erbium_upconversion(level: str = "moderate") -> ConcentrationModel:
    """Representative Er homogeneous-upconversion models. C_up rises steeply with Er
    concentration; 'moderate' ~ 3e-24 m^3/s, 'heavy' ~ 1e-23 m^3/s with a few-percent quenched
    pair fraction (order-of-magnitude, calibrate per fiber via Phase 8)."""
    if level == "heavy":
        return ConcentrationModel(c_up_m3_s=1.0e-23, pair_fraction=0.03)
    if level == "light":
        return ConcentrationModel(c_up_m3_s=5.0e-25)
    return ConcentrationModel(c_up_m3_s=3.0e-24)          # moderate (default)


def ytterbium_photodarkening(alpha_eq_dB_per_m: float = 0.58, nbar2_ref: float = 0.46,
                             pd_exponent: float = 1.0) -> ConcentrationModel:
    """Yb photodarkening from a measured EQUILIBRIUM anchor: alpha_eq_dB_per_m is the saturated
    excess loss at the SIGNAL wavelength measured at inversion nbar2_ref (the community
    benchmark is the 976 nm clamp, nbar2 ~ 0.46). The stored scale extrapolates to full
    inversion with the equilibrium law nbar2^pd_exponent (default 1, the measured linear law --
    see the module docstring for why the steep rate exponents must NOT be used here).

    Grade anchors at ~1041 nm, nbar2 = 0.46, Yb ~ 6e19 cm^-3 class (Zhang et al., Front. Phys.
    11, 1124491 (2023); Zhao et al., Opt. Express 25, 18191 (2017)):
      * equimolar Al:P / phosphosilicate:  below detection, <= 0.055 dB/m
      * Yb/Al/Ce kW-class (the DEFAULT):   0.58 dB/m
      * legacy plain Yb/Al:                ~3-5 dB/m (amplifier-relevant; specify the fiber!)
    Transfer between Yb densities with the ~quadratic concentration law (N/N_ref)^2
    (Taccheo 2011). Calibrate per fiber whenever possible."""
    if not (0.0 < nbar2_ref <= 1.0):
        raise ValueError("ytterbium_photodarkening: nbar2_ref must be in (0, 1]")
    scale = alpha_eq_dB_per_m * _LN10_10 / (nbar2_ref ** pd_exponent)
    return ConcentrationModel(pd_loss_per_m=float(scale), pd_exponent=float(pd_exponent))
