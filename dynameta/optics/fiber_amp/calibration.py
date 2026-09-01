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
           "ytterbium_melkumov",
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
        below = np.any(lam < self.lambda_m[0] - 1e-12)
        above = np.any(lam > self.lambda_m[-1] + 1e-12)
        if below or above:
            # warn ONLY when the flat-held edge value is a significant fraction of the
            # table peak -- a large extrapolated sigma can manufacture gain from nothing
            # (audit R8), while a negligible tail (e.g. Yb sigma_e held at 0.4% of peak
            # when an Er:Yb chain samples 1550 nm) is the intended behaviour.
            peak = float(np.max(self.sigma_m2))
            edge = max(float(self.sigma_m2[0]) if below else 0.0,
                       float(self.sigma_m2[-1]) if above else 0.0)
            if peak > 0.0 and edge > 0.02 * peak:
                import warnings
                warnings.warn("CrossSectionTable: wavelength outside the measured range "
                              "[%.1f, %.1f] nm -- the edge value (%.1f%% of the table "
                              "peak) is held flat, which can manufacture gain from "
                              "nothing (audit R8)"
                              % (self.lambda_m[0] * 1e9, self.lambda_m[-1] * 1e9,
                                 100.0 * edge / peak), stacklevel=2)
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
    # This path DEFINES the cross-sections as alpha/n_t, so an undoped fiber is not expressible
    # here even though FiberSpec now permits n_t = 0 (audit F-10). Refuse explicitly rather than
    # divide by zero and hand back inf cross-sections.
    if not (float(n_t_m3) > 0.0):
        raise ValueError("giles_calibrated_fiber: n_t_m3 must be > 0 (got {!r}) -- the Giles "
                         "calibration defines sigma = alpha / n_t, so an undoped fiber has no "
                         "cross-sections to calibrate".format(n_t_m3))
    alpha = dB_per_m_to_per_m(alpha_dB_per_m)
    gstar = dB_per_m_to_per_m(gstar_dB_per_m)
    sa_eff = alpha / n_t_m3
    se_eff = gstar / n_t_m3
    ion = ion_from_cross_sections(name, lam, sa_eff, se_eff, tau_s, zero_line_m, host=host)
    fiber = FiberSpec(core_radius_m=core_radius_m, na=na, n_t_m3=n_t_m3, length_m=length_m,
                      dopant_radius_m=dopant_radius_m, background_loss_per_m=background_loss_per_m,
                      clad_radius_m=clad_radius_m, overlap_override=1.0)
    return ion, fiber


# ---- measured Yb3+ aluminosilicate spectrum (Melkumov et al. 2004) ---------------------------
# Melkumov, Bufetov, Kravtsov, Shubin, Dianov, "Absorption and emission cross section of Yb3+
# ions in Al2O3 and P2O5 doped fibers", FORC RAS Preprint No. 5 (2004), arXiv:1502.02885,
# Appendix 2 (p. 56), aluminosilicate ("AC") columns, transcribed verbatim (sigma in pm^2).
# Five cross-checked measurement methods; the deep signal-band absorption tail is the authors'
# McCumber-derived extension, which they validate against direct measurement to <= 10% over
# 950-1030 nm and <= 25% out to 1100 nm. Lifetime tau(2F5/2) = 0.83 ms (their Sec. "lifetimes",
# reproducing Kirchhof & Unger OFC'99 Yb,Al 0.830 ms). Corroborated independently: Paschotta
# et al. 1997 germanosilicate figures give sigma_e(1060) = 3.25e-25 (5% away), and the nLIGHT
# LIEKKI Application Designer spectrum ~2.9e-25. Audit trail:
# docs/audit/2026-08-28-measured-spectra-melkumov-er-pump-band.md
_YB_MELKUMOV_AS_NM = (
    848, 852, 856, 860, 864, 868, 872, 876, 880, 884, 888, 892, 896, 900, 904, 908,
    912, 916, 920, 924, 928, 932, 936, 940, 944, 948, 952, 956, 960, 964, 968, 969,
    970, 971, 972, 973, 974, 975, 976, 977, 978, 979, 980, 981, 982, 983, 984, 985,
    986, 988, 992, 996, 1000, 1004, 1008, 1012, 1016, 1020, 1024, 1028, 1032, 1036,
    1040, 1044, 1048, 1052, 1056, 1060, 1064, 1068, 1072, 1076, 1080, 1084, 1088,
    1092, 1096, 1100, 1104, 1108, 1112, 1116, 1120, 1124, 1128, 1132, 1136, 1140,
    1144, 1148, 1152, 1156, 1160, 1164, 1168, 1172, 1176, 1180)
_YB_MELKUMOV_AS_SIGMA_E_PM2 = (
    2.2e-5, 3.5e-5, 6.3e-5, 1.1e-4, 1.7e-4, 2.7e-4, 4.4e-4, 6.9e-4, 0.0011, 0.0017,
    0.0026, 0.0039, 0.0058, 0.0086, 0.012, 0.017, 0.022, 0.029, 0.034, 0.039, 0.044,
    0.048, 0.050, 0.053, 0.057, 0.062, 0.074, 0.095, 0.13, 0.17, 0.26, 0.34, 0.46,
    0.70, 1.08, 1.58, 2.14, 2.65, 2.97, 2.94, 2.71, 2.28, 1.78, 1.29, 0.91, 0.67,
    0.53, 0.45, 0.41, 0.36, 0.33, 0.33, 0.36, 0.40, 0.46, 0.53, 0.60, 0.65, 0.65,
    0.65, 0.60, 0.55, 0.49, 0.44, 0.39, 0.35, 0.33, 0.31, 0.30, 0.29, 0.27, 0.26,
    0.23, 0.22, 0.21, 0.19, 0.18, 0.16, 0.14, 0.12, 0.11, 0.098, 0.088, 0.076,
    0.071, 0.061, 0.055, 0.047, 0.042, 0.035, 0.031, 0.027, 0.023, 0.021, 0.018,
    0.014, 0.014, 0.012)
_YB_MELKUMOV_AS_SIGMA_A_PM2 = (
    0.033, 0.041, 0.057, 0.075, 0.090, 0.11, 0.14, 0.17, 0.21, 0.26, 0.31, 0.37,
    0.43, 0.50, 0.57, 0.62, 0.65, 0.65, 0.62, 0.57, 0.51, 0.44, 0.38, 0.32, 0.28,
    0.24, 0.23, 0.24, 0.26, 0.28, 0.35, 0.44, 0.57, 0.83, 1.21, 1.68, 2.17, 2.55,
    2.69, 2.53, 2.22, 1.77, 1.32, 0.91, 0.61, 0.43, 0.32, 0.26, 0.23, 0.18, 0.14,
    0.11, 0.099, 0.092, 0.088, 0.084, 0.078, 0.070, 0.059, 0.049, 0.038, 0.029,
    0.022, 0.016, 0.012, 0.0090, 0.0072, 0.0057, 0.0046, 0.0038, 0.0030, 0.0024,
    0.0018, 0.0015, 0.0012, 9.5e-4, 7.3e-4, 5.6e-4, 4.2e-4, 3.2e-4, 2.4e-4, 1.9e-4,
    1.4e-4, 1.1e-4, 8.5e-5, 6.3e-5, 4.9e-5, 3.6e-5, 2.8e-5, 2.0e-5, 1.6e-5, 1.1e-5,
    8.6e-6, 6.8e-6, 4.9e-6, 3.5e-6, 3.1e-6, 2.2e-6)


def ytterbium_melkumov() -> RareEarthIon:
    """Yb3+ built from the MEASURED Melkumov et al. 2004 aluminosilicate spectrum (tabulated
    848-1180 nm, 1 nm resolution around the 976 nm peak) instead of the compact Gaussian-sum
    fit -- THE ion bare spectroscopy.ytterbium() returns since 2026-08-28 (it delegates
    here). Motivation (2026-08-28 audit): the Gaussian fit carries
    1.62x too little oscillator strength for its own 0.83 ms lifetime (Fuchtbauer-Ladenburg),
    and its sigma_e(1060) = 1.95e-25 m^2 sits 1.6x below the measured 3.1e-25 -- the fit's
    three narrow Gaussians lose the spectral wings. This factory is FL-consistent (tau_rad
    within ~15% of tau_s) and carries the real wings; use it wherever the Yb signal-band
    magnitude is load-bearing (holding cost, saturation energy, gain crossovers)."""
    lam = np.asarray(_YB_MELKUMOV_AS_NM, float) * 1e-9
    sa = np.asarray(_YB_MELKUMOV_AS_SIGMA_A_PM2, float) * 1e-24
    se = np.asarray(_YB_MELKUMOV_AS_SIGMA_E_PM2, float) * 1e-24
    # zero line = the TABLE'S OWN sigma_e/sigma_a = 1 crossing (974.26 nm by interpolation;
    # the ratio at 976 nm is 1.104, so declaring 976 skewed every McCumber exponent by
    # ~1.10x -- audit B3 2026-09-01).  976 nm pumping therefore sits 1.7 nm ABOVE the zero
    # line and the inversion clamp is sigma_a/(sigma_a+sigma_e) = 0.475, not 0.500.
    return ion_from_cross_sections("Yb3+(melkumov)", lam, sa, se, tau_s=0.83e-3,
                                   zero_line_m=974.26e-9, host="aluminosilicate/melkumov2004")


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
