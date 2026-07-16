"""Doped-fiber waveguide geometry for the amplifier core: the step-index single-mode fiber
parameters, the fundamental-mode field radius (Marcuse), the mode/dopant OVERLAP integral
Gamma(lambda) that turns bulk cross-sections into per-metre coefficients, and the effective /
doped areas. Also the double-clad pump overlap Gamma_p = A_core/A_clad for high-power Yb.

Pure numpy; SI units. Refs: Marcuse (BSTJ 56:703, 1977) for the Gaussian mode-field radius;
Desurvire (EDFA book) for the top-hat-dopant overlap Gamma = 1 - exp(-2 b^2/w^2).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Union

import numpy as np

__all__ = ["FiberSpec", "mode_field_radius_m", "overlap_gamma", "cladding_pump_overlap"]


def mode_field_radius_m(core_radius_m: float, na: float, lambda_m):
    """Gaussian-approximation 1/e field radius w of the LP01 mode (Marcuse):
        w/a = 0.65 + 1.619 V^-1.5 + 2.879 V^-6,   V = 2 pi a NA / lambda.
    Valid ~ 1.2 < V < 2.4 (single-mode). Grows with lambda (weaker guiding) -> larger w ->
    smaller overlap, the physical origin of the wavelength-dependent Gamma."""
    a = float(core_radius_m)
    V = 2.0 * np.pi * a * float(na) / np.asarray(lambda_m, dtype=np.float64)
    V = np.maximum(V, 1e-6)
    return a * (0.65 + 1.619 * V ** -1.5 + 2.879 * V ** -6.0)


@dataclass(frozen=True)
class FiberSpec:
    """A rare-earth-doped single-mode fiber. core_radius_m = core radius a; na = numerical
    aperture; n_t_m3 = dopant ion density; dopant_radius_m = top-hat dopant radius b (defaults
    to the core radius = uniform core doping); length_m = fiber length; background_loss_per_m =
    l(lambda) [1/m] (scalar or callable, the passive fiber attenuation, default 0);
    clad_radius_m = inner-cladding radius for double-clad pumping (None = core-pumped)."""
    core_radius_m: float
    na: float
    n_t_m3: float
    length_m: float
    dopant_radius_m: Optional[float] = None
    background_loss_per_m: Union[float, Callable] = 0.0
    clad_radius_m: Optional[float] = None

    def __post_init__(self):
        for nm, v in (("core_radius_m", self.core_radius_m), ("na", self.na),
                      ("n_t_m3", self.n_t_m3), ("length_m", self.length_m)):
            if not (v > 0.0):
                raise ValueError("FiberSpec: {} must be > 0 (got {!r})".format(nm, v))
        if self.clad_radius_m is not None and not (self.clad_radius_m > self.core_radius_m):
            raise ValueError("FiberSpec: clad_radius_m must exceed core_radius_m")

    @property
    def b_dope_m(self) -> float:
        return float(self.dopant_radius_m if self.dopant_radius_m is not None
                     else self.core_radius_m)

    @property
    def a_dope_m2(self) -> float:
        """Doped cross-sectional area A_dope = pi b^2 (the ion-seen area)."""
        return float(np.pi * self.b_dope_m ** 2)

    def loss_per_m(self, lambda_m):
        loss = self.background_loss_per_m
        return np.asarray(loss(lambda_m) if callable(loss) else np.full_like(
            np.asarray(lambda_m, float), float(loss)), dtype=np.float64)


def overlap_gamma(fiber: FiberSpec, lambda_m):
    """Mode/dopant power-overlap Gamma(lambda) for a top-hat dopant of radius b inside a
    Gaussian LP01 mode of field radius w: Gamma = 1 - exp(-2 b^2 / w^2) (Desurvire). Gamma -> 1
    for tight confinement (short lambda / large core), and falls at long lambda as the mode
    spreads past the dopant -- this wavelength dependence is what makes alpha(lambda) and
    g*(lambda) genuine spectra, not just scaled cross-sections."""
    w = mode_field_radius_m(fiber.core_radius_m, fiber.na, lambda_m)
    b = fiber.b_dope_m
    return 1.0 - np.exp(-2.0 * b ** 2 / w ** 2)


def cladding_pump_overlap(fiber: FiberSpec) -> float:
    """Double-clad pump overlap with the CORE: Gamma_p = A_core / A_clad (a multimode pump
    uniformly fills the inner cladding, so only the core-area fraction overlaps the ions).
    Returns 1.0 for a core-pumped fiber (clad_radius_m is None). This is the single geometry
    factor that makes cladding pumping far weaker per unit length than core pumping -- the
    reason double-clad fibers are metres-to-tens-of-metres long."""
    if fiber.clad_radius_m is None:
        return 1.0
    return float((fiber.core_radius_m / fiber.clad_radius_m) ** 2)
