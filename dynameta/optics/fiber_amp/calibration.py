"""Calibrate the fiber-amplifier model to measured data -- the step that turns the
literature-default Gaussian cross-sections into a DEVICE-matched parameter set (mirrors
soa.calibration). docs/fiber_amp_model_spec.md sec.9.

TWO ENTRY POINTS, both feeding the SAME solver:

  * CrossSectionTable / ion_from_cross_sections: plug in MEASURED sigma_a(lambda), sigma_e(lambda)
    tables (e.g. a fiber datasheet or a spectroscopy measurement) through the same RareEarthIon
    interface the literature factories use -- linear interpolation, held flat outside the table.
  * giles_calibrated_fiber: build directly from the manufacturer's GILES PARAMETERS, the
    absorption alpha(lambda) and gain g*(lambda) spectra (in dB/m) plus the mode-doping overlap
    already folded in. These are exactly what vendors publish, so this is usually the calibration
    path. It sets sigma_a_eff = alpha/n_t, sigma_e_eff = g*/n_t and overlap_override = 1, so the
    net gain reproduces g*(lambda) nbar2 - alpha(lambda)(1 - nbar2) by construction.

calibration_report runs a calibrated amplifier at a datasheet operating point and compares gain
and noise figure against the targets. Pure numpy; SI units; ASCII.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from dynameta.optics.fiber_amp.spectroscopy import RareEarthIon
from dynameta.optics.fiber_amp.waveguide import FiberSpec

__all__ = ["CrossSectionTable", "ion_from_cross_sections", "giles_calibrated_fiber",
           "EDFA_CBAND_TARGETS", "calibration_report", "dB_per_m_to_per_m"]

_LN10_OVER_10 = np.log(10.0) / 10.0


def dB_per_m_to_per_m(x_dB_per_m):
    """Convert a power coefficient from dB/m to 1/m (Napierian): x[1/m] = x[dB/m] ln10/10."""
    return np.asarray(x_dB_per_m, float) * _LN10_OVER_10


@dataclass(frozen=True)
class CrossSectionTable:
    """A measured cross-section spectrum sigma(lambda) [m^2] as (lambda_m, sigma_m2) samples,
    linearly interpolated and held flat (clamped to the endpoint) outside the tabulated range.
    Drop-in for spectroscopy.CrossSectionModel: exposes the same .sigma(lambda_m)."""
    lambda_m: np.ndarray
    sigma_m2: np.ndarray

    def __post_init__(self):
        lam = np.asarray(self.lambda_m, float)
        sig = np.asarray(self.sigma_m2, float)
        if lam.ndim != 1 or lam.size < 2 or lam.shape != sig.shape:
            raise ValueError("CrossSectionTable: lambda_m and sigma_m2 must be matching 1-D "
                             "arrays with >= 2 samples")
        if np.any(np.diff(lam) <= 0.0):
            order = np.argsort(lam)
            object.__setattr__(self, "lambda_m", lam[order])
            object.__setattr__(self, "sigma_m2", sig[order])
        else:
            object.__setattr__(self, "lambda_m", lam)
            object.__setattr__(self, "sigma_m2", sig)

    def sigma(self, lambda_m):
        lam = np.asarray(lambda_m, float)
        out = np.interp(lam, self.lambda_m, self.sigma_m2)      # flat-held outside range
        return out if out.ndim else float(out)


def ion_from_cross_sections(name: str, lambda_m, sigma_a_m2, sigma_e_m2, tau_s: float,
                            zero_line_m: float, host: str = "measured") -> RareEarthIon:
    """Build a RareEarthIon from measured absorption/emission cross-section tables. lambda_m is
    the common wavelength grid; sigma_a_m2 / sigma_e_m2 the sampled cross-sections [m^2]."""
    return RareEarthIon(name, CrossSectionTable(lambda_m, sigma_a_m2),
                        CrossSectionTable(lambda_m, sigma_e_m2), tau_s=float(tau_s),
                        zero_line_m=float(zero_line_m), host=host)


def giles_calibrated_fiber(name: str, lambda_m, alpha_dB_per_m, gstar_dB_per_m, *,
                           n_t_m3: float, core_radius_m: float, na: float, length_m: float,
                           tau_s: float, zero_line_m: float, dopant_radius_m: Optional[float] = None,
                           background_loss_per_m=0.0, clad_radius_m: Optional[float] = None,
                           host: str = "giles"):
    """Build (ion, fiber) from vendor GILES PARAMETERS: absorption alpha(lambda) and gain
    g*(lambda) spectra in dB/m (overlap already folded in). Returns effective cross-sections
    sigma_a = alpha/n_t, sigma_e = g*/n_t with overlap_override = 1 so the solver reproduces the
    published spectra. n_t_m3 is the ion density used to define the doped area and the intensity
    scale (the Giles saturation parameter); pick the vendor's value or a standard one."""
    lam = np.asarray(lambda_m, float)
    alpha = dB_per_m_to_per_m(alpha_dB_per_m)
    gstar = dB_per_m_to_per_m(gstar_dB_per_m)
    sa_eff = alpha / n_t_m3
    se_eff = gstar / n_t_m3
    ion = ion_from_cross_sections(name, lam, sa_eff, se_eff, tau_s, zero_line_m, host=host)
    fiber = FiberSpec(core_radius_m=core_radius_m, na=na, n_t_m3=n_t_m3, length_m=length_m,
                      dopant_radius_m=dopant_radius_m, background_loss_per_m=background_loss_per_m,
                      clad_radius_m=clad_radius_m, overlap_override=1.0)
    return ion, fiber


# ---- representative datasheet target (a generic single-mode C-band EDFA gain block) ----------
EDFA_CBAND_TARGETS = {
    "pump_nm": 980.0,
    "signal_nm": 1550.0,
    "pump_power_mW": 100.0,
    "signal_in_dBm": -30.0,
    "small_signal_gain_dB": 30.0,       # typ small-signal gain
    "nf_dB_max": 5.5,                   # typ noise figure
}


@dataclass
class CalibrationReport:
    gain_dB: float
    nf_dB: float
    targets: dict
    gain_ok: bool
    nf_ok: bool

    @property
    def ok(self) -> bool:
        return self.gain_ok and self.nf_ok


def calibration_report(amp, targets: dict = None, *, gain_tol_dB: float = 3.0) -> CalibrationReport:
    """Run a (calibrated) amplifier at the datasheet operating point and compare gain + noise
    figure to the targets. amp must already carry the pump/signal/ASE plan; the signal channel
    nearest targets['signal_nm'] is used. Passes if the gain is within gain_tol_dB of the target
    and the NF is at or below the target ceiling."""
    from dynameta.optics.fiber_amp.noise import analyze_noise
    tg = targets if targets is not None else EDFA_CBAND_TARGETS
    r = amp.solve()
    lam_s = tg["signal_nm"] * 1e-9
    nr = analyze_noise(r, lam_s)
    gain_ok = abs(nr.gain_dB - tg["small_signal_gain_dB"]) <= gain_tol_dB
    nf_ok = nr.nf_dB <= tg["nf_dB_max"] + 1e-9
    return CalibrationReport(nr.gain_dB, nr.nf_dB, dict(tg), gain_ok, nf_ok)
