"""Rare-earth ion spectroscopy for the fiber-amplifier core: absorption / emission
cross-section spectra sigma_a(lambda), sigma_e(lambda) [m^2], the upper-state lifetime, and
the McCumber relation linking the two. Literature-default Er3+ and Yb3+ ions are provided as
factories (aluminosilicate / phosphosilicate hosts); a user calibration plugs in measured
spectra through the SAME CrossSectionModel/RareEarthIon interface (Phase 8).

Cross-sections are parametrized as a small sum of Gaussians ANCHORED to the primary-literature
peak values (docs/fiber_amp_model_spec.md sec.5): the exact spectral shape between anchors is
refinable by calibration, but the peaks / key-wavelength magnitudes and the McCumber link are
correct by construction. Pure numpy; SI units; wavelength in metres unless suffixed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from dynameta.constants import C_LIGHT, H_PLANCK, KB

__all__ = ["CrossSectionModel", "RareEarthIon", "erbium", "ytterbium",
           "at_temperature", "multiphonon_lifetime"]


@dataclass(frozen=True)
class CrossSectionModel:
    """A cross-section spectrum sigma(lambda) [m^2] as a sum of Gaussians in WAVELENGTH:
    sigma(lambda) = SUM_i peak_i * exp(-4 ln2 ((lambda - lambda0_i)/fwhm_i)^2). Each peak is
    (lambda0_m, fwhm_m, sigma_peak_m2). A Gaussian basis keeps sigma >= 0 everywhere and lets
    the literature anchor points be reproduced exactly at their centres."""
    peaks: Tuple[Tuple[float, float, float], ...]

    def __post_init__(self):
        for lam0, fwhm, pk in self.peaks:
            if not (lam0 > 0.0 and fwhm > 0.0 and pk >= 0.0):
                raise ValueError("CrossSectionModel peak (lambda0>0, fwhm>0, sigma>=0); got "
                                 "{!r}".format((lam0, fwhm, pk)))

    def sigma(self, lambda_m):
        lam = np.asarray(lambda_m, dtype=np.float64)
        out = np.zeros_like(lam)
        for lam0, fwhm, pk in self.peaks:
            out = out + pk * np.exp(-4.0 * np.log(2.0) * ((lam - lam0) / fwhm) ** 2)
        return out if out.ndim else float(out)


@dataclass(frozen=True)
class RareEarthIon:
    """One rare-earth dopant: absorption / emission cross-section spectra, the metastable
    upper-state lifetime tau_s, and the McCumber zero-line wavelength (the effective
    manifold-to-manifold energy gap used to derive sigma_e from sigma_a).

    sigma_a, sigma_e are CrossSectionModels [m^2]; tau_s [s]; zero_line_m [m] is the
    zero-phonon-line wavelength (Er 4I13/2<->4I15/2 ~ 1530 nm; Yb 2F5/2<->2F7/2 ~ 975 nm).

    sigma_esa (optional) is the EXCITED-STATE-ABSORPTION cross-section [m^2]: absorption from the
    metastable level to a higher-lying manifold, which (in the fast-relaxation / cycling limit)
    returns the ion to the metastable state, so it is a pure parasitic BEAM LOSS proportional to
    the excited fraction nbar2 -- it robs gain (signal ESA) and pump efficiency (pump ESA) without
    changing the inversion balance. None -> no ESA (the ideal model).

    mccumber_eps_J (optional) is the McCumber excitation chemical potential eps [J] -- the free-
    energy difference between the two thermalized Stark manifolds. None (the default) means
    h c / zero_line_m, i.e. the behaviour this class has always had, so nothing changes unless it
    is set. Setting it to a FITTED value is the knob for a real accuracy limit (audit 2026-08-04
    F-12): eps is properly a manifold-averaged quantity that lies BELOW the absorption peak, and
    taking the peak inflates (eps - h nu) far to the red. Measured on the shipped Yb model with
    eps = h c / 976 nm, sigma_e/sigma_a sits 3-11x below the McCumber prediction over 1030-1080 nm;
    the SAME over-large (eps - h nu) is why thermal.py's pure-McCumber d ln sigma_e/dT comes out
    ~3-5x steeper than measured. One fitted eps fixes both symptoms."""
    name: str
    sigma_a: CrossSectionModel
    sigma_e: CrossSectionModel
    tau_s: float
    zero_line_m: float
    host: str = ""
    sigma_esa: Optional[CrossSectionModel] = None
    mccumber_eps_J: Optional[float] = None

    def __post_init__(self):
        if not (self.tau_s > 0.0 and self.zero_line_m > 0.0):
            raise ValueError("RareEarthIon: tau_s and zero_line_m must be > 0")
        if self.mccumber_eps_J is not None and not (self.mccumber_eps_J > 0.0):
            raise ValueError("RareEarthIon: mccumber_eps_J must be > 0 when given (got {!r})"
                             .format(self.mccumber_eps_J))

    @property
    def eps_J(self) -> float:
        """The McCumber excitation chemical potential actually in force [J]: the fitted
        mccumber_eps_J when set, else the zero-line photon energy h c / zero_line_m."""
        return float(self.mccumber_eps_J if self.mccumber_eps_J is not None
                     else H_PLANCK * C_LIGHT / self.zero_line_m)

    def sigma_esa_of(self, lambda_m):
        """ESA cross-section at lambda_m [m^2]; zeros (same shape) when no ESA model is set."""
        if self.sigma_esa is None:
            lam = np.asarray(lambda_m, dtype=np.float64)
            return np.zeros_like(lam) if lam.ndim else 0.0
        return self.sigma_esa.sigma(lambda_m)

    def sigma_e_mccumber(self, lambda_m, T_K: float = 300.0, eps_J: float = None):
        """Emission cross-section from absorption via McCumber (Phys.Rev.136:A954; Miniscalco-
        Quimby OL16:258): sigma_e(nu) = sigma_a(nu) exp((eps - h nu)/kT). eps = excitation
        chemical potential [J]; defaults to the zero-line photon energy h c / zero_line_m
        (detailed balance crosses over sigma_e = sigma_a exactly at the zero line). This is the
        physical CHECK / derivation of sigma_e -- the model's own sigma_e is an independent
        parametrized fit; a gate asserts the two agree near the zero line."""
        lam = np.asarray(lambda_m, dtype=np.float64)
        nu = C_LIGHT / lam
        if eps_J is None:
            eps_J = self.eps_J          # fitted mccumber_eps_J if set, else the zero line (F-12)
        return self.sigma_a.sigma(lam) * np.exp((eps_J - H_PLANCK * nu) / (KB * float(T_K)))


# ---- literature-default ions (docs/fiber_amp_model_spec.md sec.5) --------------------------

class _McCumberEmission:
    """sigma_e derived from sigma_a by the McCumber relation at a fixed reference temperature:
        sigma_e(lambda) = sigma_a(lambda) exp[(eps - h c/lambda) / (k T)].
    Duck-types CrossSectionModel (only `.sigma()` is required of it, exactly as CrossSectionTable
    does). Used by erbium(cband_refit=True) so the two spectra CANNOT disagree -- see that factory.
    """

    __slots__ = ("sigma_a", "eps_J", "T_K")

    def __init__(self, sigma_a, eps_J: float, T_K: float = 300.0):
        self.sigma_a = sigma_a
        self.eps_J = float(eps_J)
        self.T_K = float(T_K)

    def sigma(self, lambda_m):
        lam = np.asarray(lambda_m, dtype=np.float64)
        nu = C_LIGHT / lam
        return self.sigma_a.sigma(lam) * np.exp((self.eps_J - H_PLANCK * nu) / (KB * self.T_K))


# The extra C-band sigma_a Gaussian that erbium(cband_refit=True) adds. FITTED, not measured: the
# criteria were (i) sigma_a strictly DECREASING over 1533-1570 nm, (ii) the trusted 1530 and 1560 nm
# anchor values perturbed by < 2%, (iii) no perturbation of the 980 / 1480 nm pump bands. A grid
# search over (peak, centre, FWHM) returned this as the minimum-perturbation single Gaussian:
# 1530 nm moves +1.54%, 1560 nm +0.99%, and the pump bands move by < 1e-49 m^2 (i.e. not at all).
_ER_CBAND_REFIT_PEAK = (1.543e-6, 0.014e-6, 1.0e-25)


def erbium(host: str = "aluminosilicate", *, esa: bool = False,
           cband_refit: bool = True) -> RareEarthIon:
    """Er3+ in an aluminosilicate EDF (Strohhofer-Polman / standard EDF anchors).

    PUMP BANDS: 980 nm (4I11/2) AND 1480 nm in-band (the 4I13/2 upper edge, sigma_a 0.8e-25 m^2) --
    the latter is what a high-power booster uses, because its quantum defect is far smaller
    (1480/1550 = 0.955 against 980/1550 = 0.632). SIGNAL: the 1530-1565 nm C band.
    Peaks: sigma_a 5.7e-25 m^2 at 1530 nm, 1.69e-25 at 1560 nm, 1.7e-25 at 980 nm, 0.8e-25 at
    1480 nm; sigma_e 5.7e-25 at 1532 nm, 3.04e-25 at 1560 nm. tau(4I13/2) = 10 ms.

    esa=True adds the 980 nm pump excited-state absorption (4I11/2->4F7/2, ~0.4e-25 m^2) that
    limits 980-pumped efficiency; the C-band signal ESA is negligible in silica so none is added.

    cband_refit=True IS NOW THE DEFAULT (audit 2026-08-04 F-1, made default 2026-08-05). Pass
    cband_refit=False for the pre-2026-08 spectra, which are retained because the repo's older
    pinned gate numbers and docs were measured against them. The two-spectrum Gaussian model it
    replaces has a real accuracy defect IN THE MIDDLE OF THE C BAND: it anchors sigma_a at 980 /
    1480 / 1530 / 1560 nm and sigma_e independently at 1532 / 1560 nm, and between the C-band
    anchors two things go wrong:

      * the model CONTRADICTS ITS OWN McCumber relation. sigma_e is 1.27x the value
        sigma_e_mccumber() predicts from the model's own sigma_a at 1550 nm, and 2.06x at 1540 nm,
        while at the 1530 and 1560 nm anchors the two agree to within 2%. McCumber is an exact
        thermodynamic constraint between the two spectra, so at least one of them is wrong across
        1534-1556 nm. This needs no external dataset to establish -- it is an internal
        contradiction between two things this module already asserts.
      * sigma_a has a LOCAL MINIMUM at 1543 nm -- the interpolation trough between the 1530 and
        1560 Gaussians, deeper than the anchors on either side warrant, so the interpolated SHAPE
        carries structure the anchors do not.

    SCOPE OF THE CLAIM, stated carefully. A real EDFA's GAIN spectrum genuinely does have a C-band
    valley near 1540 nm -- that is why gain-flattening filters exist -- so a non-monotone gain is
    NOT by itself evidence of a modelling defect, and the refit does not make the gain monotone.
    What the refit fixes is the part that is unambiguous: the internal thermodynamic contradiction
    (exactly 1.000x after the refit, against 2.058x worst-case before) and the excess depth of the
    sigma_a trough. Measured on a saturated 1.4 um-core EDFA at 300 mW of 976 nm pump, the
    power-conversion step from 1540 to 1545 nm goes from -6.8 percentage points to +0.9.

    The refit does two things. It adds ONE C-band sigma_a Gaussian (_ER_CBAND_REFIT_PEAK, fitted to
    restore monotonicity while moving the trusted 1530 / 1560 nm anchors by 1.5% / 1.0% and the
    pump bands not at all), and it then DERIVES sigma_e from sigma_a by McCumber, which makes the
    two spectra consistent BY CONSTRUCTION. That derivation is validated rather than assumed:
    McCumber-from-sigma_a reproduces the model's own independent emission anchors to 0.43% at
    1532 nm and 1.58% at 1560 nm, so the sigma_e Gaussian sum was very nearly redundant already.

    It also improves the 1480 nm pump: the default sigma_e(1480) is ~0 (4.6e-30 m^2), whereas
    McCumber gives 2.77e-26 m^2, so the in-band pump is correctly partially bleached and cannot
    invert past sigma_a/(sigma_a+sigma_e) = 0.74. Real 1480-pumped Er cannot be fully inverted.

    KNOWN CONSEQUENCE OF THE RED-SIDE BEHAVIOUR, measured, and the one thing to watch now that this
    is the default: the GAIN-TILT PEAK at LOW inversion moves out of the C band. Peak wavelength of
    Gamma n_t (sigma_e nbar2 - sigma_a (1 - nbar2)) scanned over 1520-1620 nm:

        nbar2      0.95    0.90    0.70    0.60    0.50    0.45    0.35
        legacy    1532.5  1532.5  1533.8  1535.0  1536.8  1537.5  1539.8  nm
        refit     1530.8  1530.8  1531.0  1531.8  1567.8  1569.2  1576.0  nm

    The two agree wherever an EDFA actually operates (nbar2 >~ 0.6). Below about 0.5 the refit puts
    the peak in the L band, which is qualitatively right -- L-band EDFAs really do run at low
    inversion -- but the POSITION is not trustworthy because sigma_e there rests on the extrapolated
    tail. If you need gain tilt at low inversion, or the L band at all, use cband_refit=False or
    supply measured spectra.

    VALIDITY of the refit: 1520-1570 nm, and it is NOT an L-band model. Outside that window the
    derived sigma_e is the product of two things that nearly cancel -- sigma_a's GAUSSIAN TAIL,
    which decays faster than real Er emission does, and the McCumber factor exp((eps - h nu)/kT),
    which grows to the red -- so it is unreliable in BOTH directions, not merely optimistic.
    Measured against the default model: +7% at 1550 nm, +17% at 1570, +21% at 1580, then -6% at
    1600 and -93% at 1650. Neither model is truth out there; the point is that the refit's
    agreement is a C-band statement. For the L band, fit RareEarthIon.mccumber_eps_J and supply a
    tabulated sigma_a that carries the real long-wavelength tail (audit F-12).

    RE-BASELINING. Because this is now the default, C-band numbers move by roughly 0.1-0.5 dB of
    gain and a few percent of PCE relative to the pre-2026-08 model; the affected gates and the
    achieved-number block of docs/fiber_amp_model_spec.md were re-measured when the default flipped.
    cband_refit=False reproduces the old ion EXACTLY for anyone reproducing an old result.
    """
    a_peaks = [
        (0.980e-6, 0.013e-6, 1.7e-25),                # 4I11/2 (980 nm pump)
        (1.480e-6, 0.040e-6, 0.8e-25),                # 1480 nm in-band pump (4I13/2 upper edge)
        (1.530e-6, 0.011e-6, 5.7e-25),                # C-band absorption peak
        (1.560e-6, 0.035e-6, 1.69e-25),               # C-band shoulder anchor
    ]
    if cband_refit:
        a_peaks.append(_ER_CBAND_REFIT_PEAK)          # fills the spurious 1543 nm trough
    sigma_a = CrossSectionModel(tuple(a_peaks))
    if cband_refit:
        sigma_e = _McCumberEmission(sigma_a, H_PLANCK * C_LIGHT / 1.530e-6, 300.0)
    else:
        sigma_e = CrossSectionModel((
            (1.532e-6, 0.012e-6, 5.7e-25),            # emission peak (near the abs peak)
            (1.560e-6, 0.040e-6, 3.04e-25),           # C-band emission shoulder anchor
        ))
    sigma_esa = CrossSectionModel((
        (0.980e-6, 0.016e-6, 0.4e-25),                # 4I11/2 -> 4F7/2 pump ESA at 980 nm
    )) if esa else None
    return RareEarthIon("Er3+" + ("(cband_refit)" if cband_refit else ""), sigma_a, sigma_e,
                        tau_s=10.0e-3, zero_line_m=1.530e-6, host=host, sigma_esa=sigma_esa)


def ytterbium(host: str = "aluminosilicate") -> RareEarthIon:
    """Yb3+ (2F5/2<->2F7/2): broad 850-1000 nm absorption (peak 976 nm), 1000-1100 nm emission,
    strong signal-band ground-state reabsorption (the quasi-three-level signature). Host peaks:
    sigma_a,peak = 2.7e-24 m^2 at 976 nm (aluminosilicate) / 1.4e-24 at 974.5 nm
    (phosphosilicate); tau(2F5/2) = 0.83 ms (alumino) / 1.45 ms (phospho). Yb is intrinsically
    ESA-FREE (2F5/2 is the only excited 4f manifold, so no higher level is reachable) -- the
    electronic-structure reason Yb reaches near-quantum-defect efficiency; sigma_esa is left
    None."""
    if host.startswith("phospho"):
        pk_a, lam_a, tau = 1.4e-24, 0.9745e-6, 1.45e-3
    else:                                              # aluminosilicate (default)
        pk_a, lam_a, tau = 2.7e-24, 0.976e-6, 0.83e-3
    sigma_a = CrossSectionModel((
        (0.915e-6, 0.035e-6, 0.30 * pk_a),            # broad 915 nm shoulder (pump option)
        (lam_a, 0.008e-6, pk_a),                       # 976 nm absorption peak
        (1.030e-6, 0.050e-6, 0.030 * pk_a),           # signal-band reabsorption tail (3-level)
    ))
    sigma_e = CrossSectionModel((
        (lam_a, 0.010e-6, 0.98 * pk_a),               # 976 nm emission peak (~ sigma_a peak)
        (1.030e-6, 0.045e-6, 0.11 * pk_a),            # 1030 nm emission
        (1.060e-6, 0.035e-6, 0.040 * pk_a),           # 1060 nm emission tail
    ))
    return RareEarthIon("Yb3+", sigma_a, sigma_e, tau_s=tau, zero_line_m=lam_a, host=host)


# ---- temperature dependence (docs sec.10) --------------------------------------------------

@dataclass(frozen=True)
class _McCumberScaledEmission:
    """Emission cross-section re-scaled from a reference temperature to T by the McCumber factor
    ratio: sigma_e(nu, T) = sigma_e(nu, T_ref) exp[(eps - h nu)(1/kT - 1/kT_ref)]. At T = T_ref
    the factor is exactly 1 (the reference spectrum is returned unchanged), and the detailed-
    balance crossover stays pinned at the zero line (eps = h nu) for every T. Duck-types the
    CrossSectionModel .sigma() interface."""
    base: object
    eps_J: float
    T_K: float
    T_ref_K: float

    def sigma(self, lambda_m):
        lam = np.asarray(lambda_m, dtype=np.float64)
        nu = C_LIGHT / lam
        expo = (self.eps_J - H_PLANCK * nu) * (1.0 / (KB * self.T_K) - 1.0 / (KB * self.T_ref_K))
        out = np.asarray(self.base.sigma(lam), np.float64) * np.exp(expo)
        return out if out.ndim else float(out)


def at_temperature(ion: RareEarthIon, T_K: float, *, T_ref_K: float = 300.0,
                   eps_J: float = None, tau_s: float = None) -> RareEarthIon:
    """Return a copy of ion at operating temperature T_K (docs sec.10). The emission cross-section
    is McCumber-scaled from T_ref_K to T (sigma_a and the zero line are held -- sigma_a's own
    thermal-broadening is second order); at T = T_ref_K the ion is byte-identical. Pass tau_s to
    override the lifetime (e.g. from multiphonon_lifetime); otherwise the reference tau is kept.
    eps_J defaults to the ion's own McCumber eps (its fitted mccumber_eps_J if it carries one, else
    the zero-line photon energy h c / zero_line_m -- audit F-12).

    NOTE ON COMPOSITION. This scales the ion's EXISTING sigma_e by a T-RATIO
    exp[(eps - h nu)(1/kT - 1/kT_ref)]; it does not re-derive sigma_e from sigma_a. So it composes
    correctly even when sigma_e is itself McCumber-derived (as in erbium(cband_refit=True)): the
    ratio is 1 at T_ref and the T_ref spectrum is whatever the ion already had. What it must NOT do
    is drop the fitted eps on the rebuilt ion, which would silently revert the scaling factor of
    every subsequent call -- hence mccumber_eps_J is carried through below."""
    if eps_J is None:
        eps_J = ion.eps_J
    if float(T_K) == float(T_ref_K) and tau_s is None:
        return ion                                      # exact no-op at the reference temperature
    se_T = _McCumberScaledEmission(ion.sigma_e, eps_J, float(T_K), float(T_ref_K))
    return RareEarthIon(ion.name, ion.sigma_a, se_T, ion.tau_s if tau_s is None else float(tau_s),
                        ion.zero_line_m, ion.host, sigma_esa=ion.sigma_esa,
                        mccumber_eps_J=ion.mccumber_eps_J)


def multiphonon_lifetime(tau_radiative_s: float, T_K: float, *, gap_cm: float,
                         phonon_cm: float = 1100.0, coupling_per_s: float = 0.0,
                         alpha_per_cm: float = 4.5e-3) -> float:
    """Metastable lifetime at T from multiphonon nonradiative decay (docs sec.10), the
    Miyakawa-Dexter energy-gap law with the Bose stimulated-phonon temperature factor:
        1/tau(T) = 1/tau_radiative + W_nr(T),
        W_nr(T) = coupling * exp(-alpha_per_cm * gap_cm) * (nbar + 1)^p,
        nbar = 1/(exp(h c phonon_cm / kT) - 1),  p = gap_cm / phonon_cm  (phonons to bridge gap).
    The exp(-alpha*gap) ENERGY-GAP LAW is the dominant gap dependence -- a LARGER gap is
    exponentially LESS quenched, which is why Er (4I13/2 ~6500 cm^-1) is nearly radiative /
    T-independent while a small-gap transition quenches strongly. The (nbar+1)^p factor makes
    W_nr rise with T. coupling=0 -> purely radiative (tau_radiative, T-independent).

    UNITS of `alpha_per_cm` (audit A-16): the NAME says 1/cm; the quantity is in CENTIMETRES.
    It multiplies `gap_cm` (a wavenumber, cm^-1) in the exponent, so `alpha * gap` is
    dimensionless only if alpha carries cm -- i.e. alpha is the Miyakawa-Dexter gap-law slope
    per unit WAVENUMBER, conventionally quoted as ~4.5e-3 cm (silica). Read the default as
    "4.5e-3 per cm^-1", not "4.5e-3 cm^-1". The name is kept for back-compat; the value and the
    physics are unchanged."""
    if coupling_per_s <= 0.0:
        return float(tau_radiative_s)
    nu_ph = C_LIGHT * (phonon_cm * 100.0)               # phonon frequency [Hz] (cm^-1 -> m^-1 -> Hz)
    nbar = 1.0 / (np.expm1(H_PLANCK * nu_ph / (KB * float(T_K))))
    p = gap_cm / phonon_cm
    w0 = coupling_per_s * np.exp(-alpha_per_cm * gap_cm)     # energy-gap law (T->0 rate)
    w_nr = w0 * (nbar + 1.0) ** p
    return float(1.0 / (1.0 / tau_radiative_s + w_nr))
