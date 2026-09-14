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
           "ytterbium_melkumov", "er_yb_phosphosilicate_reference",
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


# ---- measured Yb3+ PHOSPHOSILICATE spectrum (Melkumov et al. 2004, same table) ---------------
# The P2O5-doped ("FS"/PhS) columns of the SAME Appendix 2 table, on the SAME 98-row wavelength
# grid, transcribed verbatim (sigma in pm^2). Reading convention validated by the AS columns of
# that table, which reproduce the aluminosilicate tuples above ENTRY FOR ENTRY (gated).
#
# WHAT IS DIFFERENT ABOUT THE P HOST, and why it matters for an Er:Yb fiber (every Er:Yb fiber is
# phosphosilicate -- P is the co-dopant that makes the 4I11/2 -> 4I13/2 step fast enough for the
# sensitization to work at all):
#   * the pump line is NARROWER and WEAKER: peak 1.38 pm^2 at 974 nm with a 5.59 nm FWHM in
#     absorption (5.99 nm in emission), against 2.69 pm^2 at 976 nm / 7.66 nm in aluminosilicate.
#     Both widths are computed from these printed rows by linear interpolation at half maximum,
#     not quoted: an earlier project note carried "4.7 nm", which this table does NOT reproduce
#     and which should be treated as unverified.
#   * BECAUSE the line is narrow, the standard 976 nm pump sits well down its flank:
#     sigma_a(976) = 1.01 pm^2, not the 1.38 pm^2 peak. Using the peak at the pump wavelength
#     overstates the pump absorption by 37%.
#   * the zero line is at 972.75 nm (this table's own sigma_e = sigma_a crossing, between the
#     printed 972 and 973 nm rows), 1.5 nm blue of the aluminosilicate 974.23 nm. So 976 nm
#     pumping sits 3.2 nm ABOVE the P-host zero line, sigma_e(976) = 1.18 > sigma_a(976) = 1.01,
#     and the 976 nm Yb inversion clamps at sigma_a/(sigma_a+sigma_e) = 0.461.
#   * tau(2F5/2) = 1.45 ms against 0.83 ms (Melkumov p. 30 and Table 1) -- phosphosilicate Yb
#     lives 1.75x longer, which is why its holding cost is what it is. Independent support:
#     Cheng et al., Materials 15(3), 996 (2022) recover 1.33 ms from an Er-free control, and
#     Kirchhof & Unger OFC'99 WM1 (quoted in Melkumov's own Table 1) 1.276 ms; the spread is
#     real phosphate-composition scatter and 1.45 ms is this table's own fiber.
# Cross-check on the absolute scale: Cheng et al. measure sigma_a(974) = 9.43e-25 m^2 in Er/Yb/P
# core glass, 32% BELOW Melkumov -- a genuine host-to-host spread, not an error in either.
_YB_MELKUMOV_PHS_SIGMA_A_PM2 = (
    0.0011, 0.0015, 0.0035, 0.0057, 0.009, 0.014, 0.017, 0.023, 0.028, 0.033, 0.042, 0.054,
    0.072, 0.1, 0.14, 0.19, 0.23, 0.24, 0.22, 0.21, 0.22, 0.23, 0.23, 0.25, 0.26, 0.26, 0.26,
    0.26, 0.26, 0.27, 0.32, 0.34, 0.39, 0.5, 0.76, 1.15, 1.38, 1.29, 1.01, 0.75, 0.56, 0.43,
    0.35, 0.29, 0.25, 0.22, 0.21, 0.19, 0.18, 0.17, 0.14, 0.12, 0.11, 0.091, 0.075, 0.061,
    0.048, 0.037, 0.029, 0.021, 0.015, 0.011, 0.0076, 0.0057, 0.0044, 0.0036, 0.0029, 0.0022,
    0.0016, 0.0011, 0.00074, 0.00049, 0.00031, 0.00021, 0.00014, 9.7e-05, 6.5e-05, 4.5e-05,
    3.2e-05, 2.3e-05, 1.7e-05, 1.2e-05, 9e-06, 7e-06, 5.5e-06, 4.4e-06, 3.6e-06, 3e-06,
    2.4e-06, 2e-06, 1.7e-06, 1.3e-06, 1.1e-06, 9.1e-07, 7.5e-07, 5.9e-07, 4.6e-07, 3.9e-07)
_YB_MELKUMOV_PHS_SIGMA_E_PM2 = (
    1e-05, 1.9e-05, 2e-05, 2.2e-05, 2.8e-05, 4.6e-05, 7e-05, 0.0001, 0.00016, 0.00024, 0.00038,
    0.00062, 0.0011, 0.0019, 0.0032, 0.0056, 0.0085, 0.012, 0.013, 0.016, 0.021, 0.026, 0.033,
    0.044, 0.057, 0.069, 0.09, 0.11, 0.14, 0.18, 0.25, 0.28, 0.34, 0.45, 0.73, 1.16, 1.46,
    1.43, 1.18, 0.93, 0.73, 0.59, 0.5, 0.43, 0.4, 0.37, 0.36, 0.35, 0.35, 0.41, 0.42, 0.45,
    0.47, 0.49, 0.49, 0.48, 0.45, 0.43, 0.39, 0.35, 0.3, 0.26, 0.21, 0.19, 0.18, 0.17, 0.16,
    0.15, 0.13, 0.11, 0.083, 0.064, 0.049, 0.038, 0.03, 0.025, 0.019, 0.016, 0.013, 0.011,
    0.0093, 0.0078, 0.0068, 0.0062, 0.0057, 0.0052, 0.0051, 0.0049, 0.0044, 0.0042, 0.0041,
    0.0039, 0.0036, 0.0035, 0.0033, 0.003, 0.0027, 0.0026)

_YB_MELKUMOV_HOSTS = {
    # host -> (sigma_a pm^2, sigma_e pm^2, tau_s, zero_line_m, host label)
    "aluminosilicate": (_YB_MELKUMOV_AS_SIGMA_A_PM2, _YB_MELKUMOV_AS_SIGMA_E_PM2,
                        0.83e-3, 974.26e-9, "aluminosilicate/melkumov2004"),
    "phosphosilicate": (_YB_MELKUMOV_PHS_SIGMA_A_PM2, _YB_MELKUMOV_PHS_SIGMA_E_PM2,
                        1.45e-3, 972.75e-9, "phosphosilicate/melkumov2004"),
}


def ytterbium_melkumov(host: str = "aluminosilicate") -> RareEarthIon:
    """Yb3+ built from the MEASURED Melkumov et al. 2004 spectrum (tabulated 848-1180 nm, 1 nm
    resolution around the pump peak) instead of the compact Gaussian-sum fit -- THE ion bare
    spectroscopy.ytterbium() returns since 2026-08-28 for the aluminosilicate host (it delegates
    here). Motivation (2026-08-28 audit): the Gaussian fit carries 1.62x too little oscillator
    strength for its own 0.83 ms lifetime (Fuchtbauer-Ladenburg), and its sigma_e(1060) =
    1.95e-25 m^2 sits 1.6x below the measured 3.1e-25 -- the fit's three narrow Gaussians lose
    the spectral wings. This factory is FL-consistent (tau_rad within ~15% of tau_s) and carries
    the real wings; use it wherever the Yb signal-band magnitude is load-bearing (holding cost,
    saturation energy, gain crossovers).

    host (2026-09-14) selects which pair of columns of the SAME printed table is used:

      * "aluminosilicate" (DEFAULT, unchanged): peak 2.69 / 2.97 pm^2 at 976 nm, 7.66 nm FWHM,
        tau = 0.83 ms, zero line 974.26 nm. Bit-identical to the no-argument call.
      * "phosphosilicate": peak 1.38 / 1.46 pm^2 at 974 nm, 5.59 nm absorption FWHM, tau =
        1.45 ms, zero line 972.75 nm -- the host of every Er:Yb fiber, and NOT what
        spectroscopy.ytterbium("phosphosilicate") returns (that stays the parametric Gaussian
        model, which peaks 1.4e-24 at 974.5 nm with an 8 nm FWHM and therefore reads 1.27e-24 at
        a 976 nm pump, 26% above this table's 1.01e-24). Pass this ion explicitly when the P-host
        pump absorption or the 1-um parasitic band matters -- i.e. in every ErYbAmplifier.

    Anchors this factory reproduces from the printed rows (linear interpolation on the 4 nm grid
    is the convention that reproduces the paper's own reference values exactly, gated):
    PhS 976 nm 1.01e-24 / 1.18e-24, 1030 nm 1.80e-26 / 3.25e-25, 1060 nm 2.20e-27 / 1.50e-25;
    AS 976 nm 2.69e-24 / 2.97e-24, 1030 nm 4.35e-26 / 6.25e-25, 1060 nm 5.70e-27 / 3.10e-25."""
    try:
        sa_pm2, se_pm2, tau, lam0, label = _YB_MELKUMOV_HOSTS[host]
    except KeyError:
        raise ValueError("ytterbium_melkumov: host must be one of %s; got %r"
                         % (sorted(_YB_MELKUMOV_HOSTS), host)) from None
    lam = np.asarray(_YB_MELKUMOV_AS_NM, float) * 1e-9      # ONE grid, both hosts
    sa = np.asarray(sa_pm2, float) * 1e-24
    se = np.asarray(se_pm2, float) * 1e-24
    # zero line = the TABLE'S OWN sigma_e/sigma_a = 1 crossing (AS: 974.26 nm by interpolation;
    # the ratio at 976 nm is 1.104, so declaring 976 skewed every McCumber exponent by
    # ~1.10x -- audit B3 2026-09-01).  976 nm pumping therefore sits 1.7 nm ABOVE the zero
    # line and the inversion clamp is sigma_a/(sigma_a+sigma_e) = 0.475, not 0.500. For the
    # PhS columns the crossing is at 972.75 nm and the 976 nm clamp is 0.461.
    name = "Yb3+(melkumov)" if host == "aluminosilicate" else "Yb3+(melkumov,phs)"
    return ion_from_cross_sections(name, lam, sa, se, tau_s=tau,
                                   zero_line_m=lam0, host=label)


def er_yb_phosphosilicate_reference(population: str = "two"):
    """The REFERENCE core-pumpable Er:Yb phosphosilicate fiber as (er_ion, yb_ion, fiber, kwargs),
    ready for `ErYbAmplifier(er_ion, yb_ion, fiber, pumps, signals, ase, **kwargs)`. Every number
    below is sourced; nothing here is tuned to a device measurement this repo also gates on.

    GEOMETRY AND DOPING -- Rybaltovsky, Lipatov, Lobanov, Abramov, Umnikov, Bazakutsa, Bobkov,
    Butov, Gur'yanov, JOSA B 37(10), 3077 (2020), Table 1, fiber #2: 4 um core (a = 2.0 um),
    delta_n = 0.014 i.e. NA = 0.20, 125 um cladding, 60 dB/m at 1535 nm and 1500 dB/m at 976 nm.
    Back-computing those two absorptions with the Marcuse overlap and the two ions' own
    cross-sections gives N_Er = 4.0e25 m^-3 and N_Yb = 4.0e26 m^-3 (Yb:Er = 10:1). Three
    independent checks pass: the derived MFD (6.37 um) lands inside the Fibercore Er/Yb catalogue
    box (5.3-6.8 um); the composition (0.1 / 0.8 mol% Er2O3 / Yb2O3) gives 4.0e25 and ~3.2e26
    without using the absorptions at all; and the full-inversion gain at 1550 nm, 28.4 dB/m,
    matches Rybaltovsky's measured 0.27 dB/cm. Length 0.8 m (the family is 0.4-1.5 m).
    Background loss 30 dB/km (research-grade; Rybaltovsky, Le Gouet).

    IONS. Er: `spectroscopy.erbium("phosphosilicate")`, the P-host-anchored spectrum (1535 nm
    peak, sigma_a 5.95e-25, tau 9.0 ms). Yb: `ytterbium_melkumov("phosphosilicate")`, the
    measured P2O5 table (tau 1.45 ms, 5.6 nm pump line, zero line 972.75 nm).

    TRANSFER, and the ONE genuinely contested number. Two consistent parameterizations are
    returned; `kwargs` carries the TWO-POPULATION one, because it is the one that carries the
    measured physics:
      * single population: k_tr = 1.0e-21 m^3/s, f = 1. An EFFECTIVE, device-validated
        coefficient: it reproduces the Exail datasheet PCE, the Bai 2015 small-core stage
        (2.63 W / 19.4 dB against a measured 2.6 W / 19.4 dB) and the measured core-pumped
        roll-off. It is NOT a measured microscopic rate and it over-predicts large-core > 20 W
        output by 1.4-2x. Available as `kwargs_single_population`.
      * two populations (`population="two"`, THE DEFAULT): f = 0.9 with a coupled-pool k_tr = 3.0e-21 m^3/s --
        Cheng et al., Materials 15(3), 996 (2022), whose bi-exponential Yb decay in Er/Yb/P glass
        gives a fast-pair rate of 3.07e-21 m^3/s (and an ensemble yield of 1.2e-22 from the same
        measurement, the factor of 26 that the coupled fraction exists to absorb). f = 0.9 is the
        high-ETE commercial-fiber end of the measured range (FORC put Nufern LMA-EYDF-25P/300 at
        ~0.9; Sefler 2004 measured 0.83-0.85 on OFS/INO double-clad fiber).
    K2 = 2.0e-22 m^3/s (Sefler 2004 Table 1: 1.5-4.0e-22 across three fibers; Canat 2006 fits
    1-2e-22). k_back = 0 -- no primary source for a phosphosilicate FIBER value was found, and
    the 2.3e-24 that circulated in this project came from misreading Vysokikh's chi_2, which is a
    Yb stimulated-emission rate; with A_32 = 3e5 and any plausible k_back, phi > 0.999.
    W_mig = 0 by default: no measurement of the migration rate between the pools was located, and
    it is the one parameter here with no source at all.
    C_up = 1.1e-24 m^3/s (Hwang et al., JOSA B 17(5), 833 (2000), bulk phosphate at 5-10x this
    density; the Er:Yb FIBER modelling literature clusters 3-10e-24, so this is the conservative
    end -- see concentration.py). Pair fraction 2k = 1.6% (Le Gouet et al., JLT 37(15), 3611
    (2019), measured on an Er:Yb Al:P fiber), in the Delevaque convention, i.e. 0.8% dark.

    Returns (er_ion, yb_ion, fiber, kwargs) with kwargs containing ONLY ErYbAmplifier
    constructor keywords, so `ErYbAmplifier(er, yb, fiber, pumps, signals, ase, **kwargs)` runs
    as written. The preset is a STARTING POINT for a study, not a
    calibrated device model: k_tr, f, K2 and C_up all sit inside literature ranges that span an
    order of magnitude, and section 4 of the 2026-09-14 audit note reports what each does to the
    predictions."""
    from dynameta.optics.fiber_amp.concentration import ConcentrationModel
    from dynameta.optics.fiber_amp.spectroscopy import erbium

    er = erbium("phosphosilicate")
    yb = ytterbium_melkumov("phosphosilicate")
    fiber = FiberSpec(core_radius_m=2.0e-6, na=0.20, n_t_m3=4.0e25, length_m=0.8,
                      clad_radius_m=62.5e-6,
                      background_loss_per_m=float(dB_per_m_to_per_m(30.0e-3)))
    conc = ConcentrationModel(c_up_m3_s=1.1e-24, pair_fraction=0.016,
                              pair_convention="delevaque")
    if population not in ("two", "single"):
        raise ValueError("er_yb_phosphosilicate_reference: population must be 'two' or "
                         "'single'; got %r" % (population,))
    kwargs = dict(n_yb_m3=4.0e26, k_back_m3_s=0.0, a32_per_s=3.0e5,
                  yb_migration_rate_per_s=0.0, concentration=conc)
    if population == "two":
        kwargs.update(k_tr_m3_s=3.0e-21, yb_coupled_fraction=0.9, k_tr2_m3_s=2.0e-22)
    else:
        kwargs.update(k_tr_m3_s=1.0e-21, yb_coupled_fraction=1.0, k_tr2_m3_s=0.0)
    return er, yb, fiber, kwargs


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
