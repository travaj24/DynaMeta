"""Doped-fiber waveguide geometry for the amplifier core: the step-index single-mode fiber
parameters, the fundamental-mode field radius (Marcuse), the mode/dopant OVERLAP integral
Gamma(lambda) that turns bulk cross-sections into per-metre coefficients, and the effective /
doped areas. Also the double-clad pump overlap for high-power Yb -- which is
Gamma_p = A_DOPE/A_clad, the doped-area fraction, NOT A_core/A_clad (audit S3-9; the two coincide
only for uniform core doping, b_dope = a_core, and using the core radius inflates the clad-pump
absorption and gain by (a_core/b_dope)^2 for a confined dopant -- see cladding_pump_overlap).

Pure numpy; SI units. Refs: Marcuse (BSTJ 56:703, 1977) for the Gaussian mode-field radius;
Desurvire (EDFA book) for the top-hat-dopant overlap Gamma = 1 - exp(-2 b^2/w^2).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Union

import numpy as np

__all__ = ["FiberSpec", "mode_field_radius_m", "overlap_gamma", "cladding_pump_overlap",
           "v_number", "marcuse_validity", "V_MARCUSE_MIN", "V_MARCUSE_MAX"]


V_MARCUSE_MIN = 1.2
V_MARCUSE_MAX = 2.405


def v_number(core_radius_m: float, na: float, lambda_m):
    """Normalized frequency V = 2 pi a NA / lambda (same shape as lambda_m)."""
    return 2.0 * np.pi * float(core_radius_m) * float(na) / np.asarray(lambda_m, dtype=np.float64)


def marcuse_validity(core_radius_m: float, na: float, lambda_m):
    """(all_in_range, V_min, V_max) for the Marcuse Gaussian-mode fit's stated validity window
    V_MARCUSE_MIN < V < V_MARCUSE_MAX (audit 2026-08-04 F-9).

    Why this is a QUERY and not a warning on mode_field_radius_m itself:
      * `lambda_m` is routinely the WHOLE channel-wavelength array (rare_earth.ChannelSet.build ->
        overlap_gamma calls it once per solve with every pump, signal and ASE-bin wavelength), so V
        is an ndarray and any scalar `if` on it raises. This returns reduced scalars instead.
      * a warning on the default path would fire inside the solver hot loop for every large-mode-
        area or cladding-pumped fiber, i.e. on a large fraction of this package's own fixtures --
        and the repo runs pytest with `filterwarnings = ["error"]`, so it would convert working
        configurations into hard failures rather than informing anybody.
    Callers that WANT the diagnostic (a design script, a report) can ask. What is out of range is
    not automatically wrong: at V = 8.2193 (the Smith & Smith LMA fixture at 1032 nm) the
    Gaussian's Gamma is only 1.332% off the exact LP01 value -- 0.979180 against 0.992394,
    RE-MEASURED; this line used to say ~1.1% -- but its SATURATION integral is off by up to 13%
    (0.85 dB/m). Use transverse.ResolvedFiberAmplifier or lma.solve_lp_modes for a multimode core,
    which take the exact LP field.
    """
    V = v_number(core_radius_m, na, lambda_m)
    vmin, vmax = float(np.min(V)), float(np.max(V))
    return bool(vmin > V_MARCUSE_MIN and vmax < V_MARCUSE_MAX), vmin, vmax


def mode_field_radius_m(core_radius_m: float, na: float, lambda_m):
    """Gaussian-approximation 1/e field radius w of the LP01 mode (Marcuse):
        w/a = 0.65 + 1.619 V^-1.5 + 2.879 V^-6,   V = 2 pi a NA / lambda.
    Valid ~ 1.2 < V < 2.4 (single-mode). Grows with lambda (weaker guiding) -> larger w ->
    smaller overlap, the physical origin of the wavelength-dependent Gamma.

    The validity window is NOT enforced here -- `lambda_m` is usually the whole channel array and
    this sits in the solver hot path. Query it explicitly with marcuse_validity() (audit F-9)."""
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
    overlap_override: Union[float, Callable, None] = None

    def __post_init__(self):
        for nm, v in (("core_radius_m", self.core_radius_m), ("na", self.na),
                      ("length_m", self.length_m)):
            if not (v > 0.0):
                raise ValueError("FiberSpec: {} must be > 0 (got {!r})".format(nm, v))
        # n_t_m3 == 0 is LEGAL: an undoped (passive) fiber (audit 2026-08-04 F-10). Every gain and
        # spontaneous-source term is proportional to n_t, so n_t = 0 degrades exactly to a pure-loss
        # channel -- which is the correct physics for a transmission fiber, a pigtail, or the span
        # between two stages of an AmplifierChain. Modelling one previously needed a sentinel
        # (tests/test_fiber_srs.py used n_t_m3=1.0 with the comment "n_t must be > 0").
        # NOTE for callers: a few routines DIVIDE by the active density -- the cooperative-
        # upconversion branch of steady_state._nbar2_c / rare_earth.metastable_fraction (whose
        # quadratic term has n_t in the denominator) and calibration.giles_calibrated_fiber (which
        # forms sigma = alpha / n_t). Those raise or refuse on their own; upconversion with n_t = 0
        # is meaningless by construction, since there are no ions to upconvert.
        if not (self.n_t_m3 >= 0.0):
            raise ValueError("FiberSpec: n_t_m3 must be >= 0 (got {!r}); 0 means an undoped, "
                             "purely passive fiber".format(self.n_t_m3))
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
    g*(lambda) genuine spectra, not just scaled cross-sections. If fiber.overlap_override is set
    (scalar or callable of lambda), it is used verbatim -- the hook for a measured overlap or for
    Giles-parameter calibration that has already folded Gamma into the cross-sections."""
    ov = fiber.overlap_override
    if ov is not None:
        val = ov(lambda_m) if callable(ov) else ov
        return np.broadcast_to(np.asarray(val, np.float64),
                               np.asarray(lambda_m, np.float64).shape).copy()
    w = mode_field_radius_m(fiber.core_radius_m, fiber.na, lambda_m)
    b = fiber.b_dope_m
    return 1.0 - np.exp(-2.0 * b ** 2 / w ** 2)


def cladding_pump_overlap(fiber: FiberSpec) -> float:
    """Double-clad pump overlap with the DOPED region: Gamma_p = A_dope / A_clad (a multimode
    pump uniformly fills the inner cladding, so the fraction of pump power the ions see is the
    doped-area fraction). The solver forms the ion-seen intensity as Gamma_p*P/A_dope, which is
    consistent ONLY if Gamma_p is the power fraction inside the dopant radius b -- so the ratio
    uses b_dope, not the core radius (audit S3-9: using A_core inflates the clad-pump absorption
    and gain by (a_core/b)^2 for confined doping). For uniform core doping (b = a_core, the
    default) the two coincide. Returns 1.0 for a core-pumped fiber (clad_radius_m is None).
    This is the single geometry factor that makes cladding pumping far weaker per unit length
    than core pumping -- the reason double-clad fibers are metres-to-tens-of-metres long."""
    if fiber.clad_radius_m is None:
        return 1.0
    return float((fiber.b_dope_m / fiber.clad_radius_m) ** 2)
