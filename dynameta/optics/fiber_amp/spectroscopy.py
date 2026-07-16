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
    changing the inversion balance. None -> no ESA (the ideal model)."""
    name: str
    sigma_a: CrossSectionModel
    sigma_e: CrossSectionModel
    tau_s: float
    zero_line_m: float
    host: str = ""
    sigma_esa: Optional[CrossSectionModel] = None

    def __post_init__(self):
        if not (self.tau_s > 0.0 and self.zero_line_m > 0.0):
            raise ValueError("RareEarthIon: tau_s and zero_line_m must be > 0")

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
            eps_J = H_PLANCK * C_LIGHT / self.zero_line_m
        return self.sigma_a.sigma(lam) * np.exp((eps_J - H_PLANCK * nu) / (KB * float(T_K)))


# ---- literature-default ions (docs/fiber_amp_model_spec.md sec.5) --------------------------

def erbium(host: str = "aluminosilicate", *, esa: bool = False) -> RareEarthIon:
    """Er3+ in an aluminosilicate EDF (Strohhofer-Polman / standard EDF anchors): 980 nm and
    1480 nm pump bands, 1530-1565 nm C-band signal. Peaks: sigma_a 5.7e-25 m^2 at 1530 nm,
    1.69e-25 at 1560 nm, 1.7e-25 at 980 nm; sigma_e 5.7e-25 at 1532 nm, 3.04e-25 at 1560 nm.
    tau(4I13/2) = 10 ms. esa=True adds the 980 nm pump excited-state absorption (4I11/2->4F7/2,
    ~0.4e-25 m^2) that limits 980-pumped efficiency; the C-band signal ESA is negligible in
    silica so none is added there."""
    sigma_a = CrossSectionModel((
        (0.980e-6, 0.013e-6, 1.7e-25),                # 4I11/2 (980 nm pump)
        (1.480e-6, 0.040e-6, 0.8e-25),                # 1480 nm in-band pump (4I13/2 upper edge)
        (1.530e-6, 0.011e-6, 5.7e-25),                # C-band absorption peak
        (1.560e-6, 0.035e-6, 1.69e-25),               # C-band shoulder anchor
    ))
    sigma_e = CrossSectionModel((
        (1.532e-6, 0.012e-6, 5.7e-25),                # emission peak (near the abs peak)
        (1.560e-6, 0.040e-6, 3.04e-25),               # C-band emission shoulder anchor
    ))
    sigma_esa = CrossSectionModel((
        (0.980e-6, 0.016e-6, 0.4e-25),                # 4I11/2 -> 4F7/2 pump ESA at 980 nm
    )) if esa else None
    return RareEarthIon("Er3+", sigma_a, sigma_e, tau_s=10.0e-3, zero_line_m=1.530e-6,
                        host=host, sigma_esa=sigma_esa)


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
    eps_J defaults to the zero-line photon energy h c / zero_line_m."""
    if eps_J is None:
        eps_J = H_PLANCK * C_LIGHT / ion.zero_line_m
    if float(T_K) == float(T_ref_K) and tau_s is None:
        return ion                                      # exact no-op at the reference temperature
    se_T = _McCumberScaledEmission(ion.sigma_e, eps_J, float(T_K), float(T_ref_K))
    return RareEarthIon(ion.name, ion.sigma_a, se_T, ion.tau_s if tau_s is None else float(tau_s),
                        ion.zero_line_m, ion.host, sigma_esa=ion.sigma_esa)


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
    W_nr rise with T. coupling=0 -> purely radiative (tau_radiative, T-independent)."""
    if coupling_per_s <= 0.0:
        return float(tau_radiative_s)
    nu_ph = C_LIGHT * (phonon_cm * 100.0)               # phonon frequency [Hz] (cm^-1 -> m^-1 -> Hz)
    nbar = 1.0 / (np.expm1(H_PLANCK * nu_ph / (KB * float(T_K))))
    p = gap_cm / phonon_cm
    w0 = coupling_per_s * np.exp(-alpha_per_cm * gap_cm)     # energy-gap law (T->0 rate)
    w_nr = w0 * (nbar + 1.0) ** p
    return float(1.0 / (1.0 / tau_radiative_s + w_nr))
