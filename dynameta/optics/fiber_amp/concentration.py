"""Concentration-quenching and photo-degradation effects for heavily doped fibers (docs sec.6),
all OPT-IN: with concentration=None the amplifier is byte-identical to the ideal model.

Three mechanisms, bundled in ConcentrationModel and consumed by the steady_state solver:

  * Homogeneous cooperative UPCONVERSION (coefficient C_up [m^3/s]): two neighbouring excited
    ions interact, one relaxing to ground -- an inversion-dependent loss C_up N2^2 that clamps
    the metastable population. Enters the metastable-fraction balance as the quadratic term
    (already in rare_earth.metastable_fraction); this module supplies C_up.
  * Er PAIR-INDUCED QUENCHING (Delevaque JQE 1993): a fraction of ions sit in tightly-spaced
    pairs whose double-excited state up-converts instantly, so those ions never invert. Modelled
    as a "dark" ground-state population of density pair_fraction * n_t that adds an UNBLEACHABLE
    absorption sigma_a Gamma (pair_fraction n_t) at every wavelength -- the residual loss that
    survives even at infinite pump, the observable signature of clustering.
  * Yb PHOTODARKENING (Koponen et al. 2006): colour-centre formation drives an equilibrium excess
    background loss that grows as a steep power of the inversion (~ nbar2^7). Modelled as an
    inversion-dependent gray loss pd_loss_per_m * nbar2^pd_exponent added along the fiber.

Pure numpy; SI units. docs/fiber_amp_model_spec.md sec.6.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["ConcentrationModel", "erbium_upconversion", "ytterbium_photodarkening"]


@dataclass(frozen=True)
class ConcentrationModel:
    """Bundle of concentration/degradation parameters (all default to the no-effect value, so an
    all-default model is a no-op). c_up_m3_s = homogeneous upconversion coefficient;
    pair_fraction = Delevaque dark-pair fraction in [0, 1); pd_loss_per_m = photodarkening
    equilibrium excess-loss scale [1/m]; pd_exponent = its inversion power law."""
    c_up_m3_s: float = 0.0
    pair_fraction: float = 0.0
    pd_loss_per_m: float = 0.0
    pd_exponent: float = 7.0

    def __post_init__(self):
        if not (0.0 <= self.pair_fraction < 1.0):
            raise ValueError("ConcentrationModel: pair_fraction must be in [0, 1)")
        for nm, v in (("c_up_m3_s", self.c_up_m3_s), ("pd_loss_per_m", self.pd_loss_per_m),
                      ("pd_exponent", self.pd_exponent)):
            if v < 0.0:
                raise ValueError("ConcentrationModel: {} must be >= 0".format(nm))

    @property
    def is_identity(self) -> bool:
        """True when the model changes nothing (all mechanisms off) -> byte-identical solve."""
        return (self.c_up_m3_s == 0.0 and self.pair_fraction == 0.0
                and self.pd_loss_per_m == 0.0)

    def active_density(self, n_t_m3: float) -> float:
        """Density of gain-contributing (isolated) ions = (1 - pair_fraction) n_t."""
        return float(n_t_m3 * (1.0 - self.pair_fraction))

    def dark_density(self, n_t_m3: float) -> float:
        """Density of quenched dark-pair ions = pair_fraction n_t (always absorbing)."""
        return float(n_t_m3 * self.pair_fraction)

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


def ytterbium_photodarkening(pd_loss_per_m: float = 0.5, pd_exponent: float = 7.0
                             ) -> ConcentrationModel:
    """Representative Yb photodarkening: an equilibrium excess loss ~ pd_loss_per_m at full
    inversion, falling as nbar2^7 (Koponen) so it is negligible at low inversion. Default
    0.5/m (~2 dB/m) equilibrium scale is an aggressive high-Yb value; calibrate per fiber."""
    return ConcentrationModel(pd_loss_per_m=float(pd_loss_per_m), pd_exponent=float(pd_exponent))
