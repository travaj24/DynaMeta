"""Temperature model for the QD-SOA (roadmap SOA generality; dossier Topic 3).

Two temperature effects, kept distinct:

1. BANDGAP (Varshni) shift of the gain-peak wavelength. The semiconductor bandgap narrows with
   temperature as Eg(T) = Eg(0) - a T^2/(T + b) (Vurgaftman, Meyer & Ram-Mohan, JAP 89, 5815
   (2001)), so the whole QD gain comb red-shifts by dEg between the reference and target T. This
   is the dominant, well-characterized T effect on the PEAK LOCATION (measured ~0.2-0.4 nm/K for
   InAs QDs near 1300 nm; ~0.5-0.6 nm/K for 1550 nm InGaAsP wells).

2. Detailed-balance carrier REDISTRIBUTION of the peak-gain MAGNITUDE. As T rises, thermal escape
   (ES->WL) and back-transfer (GS->ES), both slaved to capture by detailed balance at the current
   T (qd_gain.with_detailed_balance_taus / with_full_detailed_balance), pull carriers OUT of the
   ground state and depress the gain. For deep confinement (large dE_ES_GS / dE_WL_ES) this
   redistribution is weak, which is exactly the QD temperature-insensitivity (high T0) advantage
   over bulk/QW gain -- the accepted mechanism (Sugawara; p-doped QD 'infinite' T0).

An OPTIONAL homogeneous-broadening growth (LO-phonon dephasing, Bose-occupied) is also provided
(default OFF): Gamma_hom(T) = Gamma_hom_ref + b_LO (n_LO(T) - n_LO(T_ref)), n_LO = 1/(exp(E_LO/kT)
- 1) (Borri et al., PRL 87, 157401 (2001): 300 K ensemble homogeneous FWHM 10-20 meV, near-zero
dephasing at low T).

Pure numpy; SI (energies returned in eV where named _ev, wavelengths in nm where named _nm);
ASCII only. Back-compatible: qd_params_at_temperature at T = T_ref is a no-op on an already
detailed-balanced parameter set.
"""

from __future__ import annotations

from typing import Tuple, Union

import numpy as np

from dynameta.constants import C_LIGHT, H_PLANCK, KB, Q_E
from dynameta.optics.soa.qd_gain import QDGainParams

__all__ = ["VARSHNI_PARAMS", "B_LO_10MEV_HZ", "varshni_eg_ev", "d_eg_dT_ev_per_K",
           "gain_peak_drift_nm_per_K", "fwhm_hom_at_temperature", "qd_params_at_temperature"]


# Varshni (Eg0 [eV], alpha [eV/K], beta [K]) -- Vurgaftman JAP 89, 5815 (2001).
VARSHNI_PARAMS = {
    "GaAs":       (1.519, 5.405e-4, 204.0),
    "InAs":       (0.417, 2.76e-4, 93.0),
    "InP":        (1.4236, 3.63e-4, 162.0),
    "InGaAs_LM":  (0.816, 2.9e-4, 193.0),   # In0.53Ga0.47As lattice-matched to InP
}

# b_LO [Hz] that yields ~10 meV homogeneous-FWHM growth from 0 -> 300 K at E_LO = 36 meV
# (n_LO(300 K) = 1/(exp(36/25.85) - 1) = 0.3306; 10 meV / 0.3306 = 30.25 meV -> Hz). Per Borri 2001.
# This is a documented reference magnitude; the qd_params_at_temperature default is 0 (OFF).
B_LO_10MEV_HZ = 30.25e-3 * Q_E / H_PLANCK   # ~7.31e12 Hz


def _varshni_abc(material: Union[str, Tuple[float, float, float]]) -> Tuple[float, float, float]:
    if isinstance(material, str):
        if material not in VARSHNI_PARAMS:
            raise ValueError("varshni: unknown material '{}' (known: {}); pass a custom "
                             "(Eg0_eV, alpha_eV_K, beta_K) tuple instead".format(
                                 material, sorted(VARSHNI_PARAMS)))
        return VARSHNI_PARAMS[material]
    abc = tuple(float(x) for x in material)
    if len(abc) != 3:
        raise ValueError("varshni: custom material must be (Eg0_eV, alpha_eV_K, beta_K)")
    return abc  # type: ignore[return-value]


def varshni_eg_ev(T_K: float, material: Union[str, Tuple[float, float, float]] = "InAs") -> float:
    """Varshni bandgap Eg(T) = Eg0 - alpha T^2/(T + beta) [eV]. material is a preset name
    (VARSHNI_PARAMS) or a custom (Eg0_eV, alpha_eV_K, beta_K) tuple."""
    Eg0, a, b = _varshni_abc(material)
    T = float(T_K)
    return float(Eg0 - a * T * T / (T + b))


def d_eg_dT_ev_per_K(T_K: float, material: Union[str, Tuple[float, float, float]] = "InAs") -> float:
    """Varshni bandgap temperature slope dEg/dT = -alpha T (T + 2 beta)/(T + beta)^2 [eV/K]
    (<= 0; the gap narrows as T rises). Hand-derived from d/dT[-a T^2/(T+b)]."""
    _Eg0, a, b = _varshni_abc(material)
    T = float(T_K)
    return float(-a * T * (T + 2.0 * b) / (T + b) ** 2)


def gain_peak_drift_nm_per_K(lambda_nm: float, material: Union[str, Tuple[float, float, float]] = "InAs",
                             T_K: float = 300.0) -> float:
    """Gain-peak wavelength drift |dlambda/dT| [nm/K] at emission wavelength lambda_nm, from the
    Varshni gap slope: dlambda/dT = (lambda^2/(h c)) |dEg/dT|. (E = h c/lambda -> dlambda =
    -(lambda^2/hc) dE; the QD peak tracks the gap even though the QD emission energy > Eg by the
    confinement energy.)"""
    lam_m = float(lambda_nm) * 1.0e-9
    dEg_J = abs(d_eg_dT_ev_per_K(T_K, material)) * Q_E
    return float(lam_m * lam_m / (H_PLANCK * C_LIGHT) * dEg_J * 1.0e9)


def fwhm_hom_at_temperature(fwhm_ref_hz: float, T_K: float, T_ref_K: float, *,
                            b_LO_hz: float = 0.0, E_LO_meV: float = 36.0) -> float:
    """Homogeneous FWHM at T with LO-phonon (Bose) dephasing growth referenced to T_ref:
    Gamma(T) = Gamma_ref + b_LO (n_LO(T) - n_LO(T_ref)), n_LO = 1/(exp(E_LO/kT) - 1). b_LO_hz = 0
    (default) -> returns fwhm_ref_hz UNCHANGED (byte-safe / OFF). B_LO_10MEV_HZ gives ~10 meV growth
    over 0 -> 300 K."""
    if b_LO_hz == 0.0:
        return float(fwhm_ref_hz)
    if not (E_LO_meV > 0.0):
        raise ValueError("fwhm_hom_at_temperature: E_LO_meV must be > 0")
    elo = E_LO_meV * 1.0e-3 * Q_E
    nT = 1.0 / np.expm1(elo / (KB * float(T_K)))
    n0 = 1.0 / np.expm1(elo / (KB * float(T_ref_K)))
    return float(fwhm_ref_hz + b_LO_hz * (nT - n0))


def qd_params_at_temperature(params: QDGainParams, T_K: float, *, material: Union[str, Tuple] = "InAs",
                             T_ref_K: float = None, full_detailed_balance: bool = False,
                             b_LO_hz: float = 0.0, E_LO_meV: float = 36.0) -> QDGainParams:
    """Return a copy of params retargeted to temperature T_K:
      (a) T_K set on the copy,
      (b) nu0_Hz red-shifted by the Varshni gap change between T_ref and T (dnu0 = dEg/h; dEg < 0
          for T > T_ref -> lower nu0, longer wavelength),
      (c) the detailed-balance escape times RE-DERIVED at T (with_detailed_balance_taus, or
          with_full_detailed_balance if full_detailed_balance=True) so the escape/back-transfer track
          the new thermal equilibrium -- the temperature-insensitivity mechanism,
      (d) optionally the homogeneous FWHM grown by LO-phonon dephasing (b_LO_hz > 0; default OFF).

    T_ref_K defaults to params.T_K. NO-OP CONTRACT: on an already detailed-balanced params, calling
    with T_K == T_ref returns a field-identical copy (dEg = 0, dnu0 = 0, no broadening delta, and the
    re-derivation reproduces the same tau). SI; ASCII."""
    from dataclasses import replace
    T_ref = params.T_K if T_ref_K is None else float(T_ref_K)
    dEg_eV = varshni_eg_ev(T_K, material) - varshni_eg_ev(T_ref, material)   # <= 0 for T > T_ref
    dnu0_Hz = dEg_eV * Q_E / H_PLANCK                                        # red-shift (< 0)
    fwhm_new = fwhm_hom_at_temperature(params.fwhm_hom_Hz, T_K, T_ref, b_LO_hz=b_LO_hz,
                                       E_LO_meV=E_LO_meV)
    p2 = replace(params, T_K=float(T_K), nu0_Hz=float(params.nu0_Hz + dnu0_Hz),
                 fwhm_hom_Hz=float(fwhm_new))
    return p2.with_full_detailed_balance() if full_detailed_balance else p2.with_detailed_balance_taus()
