"""Nonlinear power limits of the rare-earth fiber amplifier (docs sec.8): the estimators and
post-processors that decide how much power a given fiber can carry before a parasitic nonlinear
process -- stimulated Brillouin scattering (SBS), stimulated Raman scattering (SRS), transverse
mode instability (TMI), or double-Rayleigh multi-path interference (MPI) -- steals it, distorts
it, or destabilizes the beam. These are DESIGN-SIDE limits, not part of the coupled steady-state
solve: SBS/SRS thresholds gate the single-frequency / narrow-band output; TMI gates the
average-power beam quality of a high-power double-clad Yb amplifier; double-Rayleigh MPI sets a
noise floor on a high-gain link. The COUPLED SRS Stokes channel (a second propagating field
inside FiberAmplifier.solve) is a separate, heavier model; everything here is either a
closed-form threshold or a post-processor of a converged SteadyStateResult.

Two forms of each scattering limit are provided, mirroring the two regimes in the literature:

  * PASSIVE (Smith criterion): the classic long-passive-fiber threshold
        P_th = C K A_eff / (g L_eff),   L_eff = (1 - exp(-alpha L)) / alpha,
    where C = ln(P_out_Stokes / P_seed) ~ 21 for SBS (18-21; NOT fundamental -- it is the log of
    the seed-to-output ratio at "threshold") and ~16 (forward) / ~20 (backward) for SRS. K is the
    polarization factor (1 for co-polarized PM, 3/2 for standard randomly-birefringent SMF, 2 for
    a scrambled state). [Smith, Appl. Opt. 11, 2489 (1972); Kobyakov, Sauer & Chowdhury, Adv.
    Opt. Photon. 2, 1 (2010); van Deventer & Boot, JLT 12, 585 (1994) for K.]

  * ACTIVE (position-dependent gain exponent): in an amplifier the local signal power P(z)
    already carries the pump gain and background loss, so the accumulated Brillouin/Raman gain is
        G = (g / A_eff) integral_0^L P(z) dz,      threshold at G ~ C,
    integrated directly over the SteadyStateResult signal profile (do NOT also fold in L_eff --
    that double-counts the gain). [Kobyakov 2010; Gray et al., Opt. Express 15, 17044 (2007), who
    reach a 500 W single-frequency LMA output at this SBS limit.]

Material / accuracy notes carried as defaults:
  g_B: 5e-11 m/W is the ideal pure-silica peak; the EFFECTIVE value in real GeO2-doped SMF is
    ~2e-11 m/W (acousto-optic overlap, dopant broadening) -- the default here, for design. g_B is
    nearly wavelength-independent because g_B ~ 1/(lambda^2 dnu_B) and dnu_B ~ 1/lambda^2.
  Brillouin shift nu_B = 2 n v_a / lambda ~ 11 GHz at 1550 nm, ~16 GHz at 1060 nm. Brillouin
    gain linewidth dnu_B ~ 30 MHz at 1550 nm (literature spread 20-50 MHz) scaling as 1/lambda^2,
    so ~60-75 MHz at 1060 nm.
  SBS thermal seed: the backward Stokes wave is seeded by thermal phonons, occupation
    n_th = 1/(exp(h nu_B / kT) - 1) ~ 560 at 11 GHz / 300 K (kT >> h nu_B, so NOT one photon per
    mode) -- P_seed ~ h nu_s dnu_B n_th. [Boyd, Rzazewski & Narum, PRA 42, 5514 (1990).]
  g_R: 1.0e-13 m/W co-polarized peak at a 1.0 um pump [Stolen & Ippen, APL 22, 276 (1973)],
    scaling ~1/lambda_pump (so ~0.6e-13 at 1550 nm); Stokes shift 13.2 THz (~440 cm^-1).
    [Agrawal, Nonlinear Fiber Optics.]

TMI is a heat-driven beam-quality limit, not a scattering threshold: the LP01xLP11 beat note plus
quantum-defect heating writes a moving thermal (STRS) grating that couples power out of the
fundamental mode above a threshold average power [Smith & Smith, Opt. Express 19, 10180 (2011);
Jauregui, Stihler & Limpert, Adv. Opt. Photon. 12, 429 (2020)]. The implementable estimator
    P_th ~ C0 (lambda_s / d_core)^2 kappa / (eta_heat (dn/dT) Gamma_ov)
captures the ROBUST scaling (P_th falls with core diameter and with heat fraction, rises with
1/(dn/dT) and 1/Gamma_ov); C0 is a single dimensionless constant that MUST be pinned to one
measured point. Here C0 is calibrated so a 20-um-core, 1.06-um, eta_heat = 0.09, Gamma_ov = 0.5
fiber sits at 1.0 kW (a representative 20/400 co-pumped Yb value). ABSOLUTE ACCURACY IS 2-3x
ONLY. In particular the 85-um rod point (~250-300 W measured; Eidam, Opt. Express 19, 13218
(2011)) is NOT recovered at the calibration's eta_heat / Gamma_ov: at identical eta_heat = 0.09,
Gamma_ov = 0.5 the pure (lambda_s / d_core)^2 scaling gives only 1000 (20/85)^2 = 55 W. The best
PHYSICALLY-admissible rod parameters (eta_heat = 0.052, the Yb 976->1030 quantum-defect floor;
Gamma_ov = 0.3; lambda_s = 1.03 um) raise this to ~150 W -- within the documented 2-3x band of
the measurement, but not on the nose. This is the inherent parameter-sensitivity of a
one-constant TMI estimator, not a bug; treat the number as order-of-magnitude and trust only the
scaling trends.

Double-Rayleigh backscatter (DRB): distributed Rayleigh scattering has a backward-captured tail
alpha_R S (S = the guided-mode recapture fraction); light scattered twice (forward at z1 <
backward at z2, back to forward) co-propagates with the signal and beats with it as a
multi-path-interference (MPI) noise floor. The key simplification is that the net gain between
the two scatter points is exactly the signal power ratio, G(z1, z2) = P(z2)/P(z1) (common-path
gain cancels), so
    R_MPI = (alpha_R S)^2 integral_0^L dz2 integral_0^{z2} dz1 (P(z2)/P(z1))^2.
[Brinkmeyer, Appl. Opt. 19, 3574 (1980) and Nakazawa, JOSA 73, 1175 (1983) for S; Bromage, JLT
22, 79 (2004) and Fludger & Mears, JLT 19, 536 (2001) for the DRB/MPI integral; Gimlett & Cheung,
JLT 7, 888 (1989) for the MPI RIN spectrum.] MPI below -40 dB is negligible (< 0.5 dB penalty).

Pure numpy; SI units; ASCII-only. docs/fiber_amp_model_spec.md sec.8.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from dynameta.constants import C_LIGHT, H_PLANCK, KB
from dynameta.optics.fiber_amp.steady_state import SteadyStateResult
from dynameta.optics.fiber_amp.waveguide import FiberSpec, mode_field_radius_m

__all__ = [
    # Brillouin spectroscopy
    "brillouin_shift_hz", "brillouin_linewidth_hz", "brillouin_phonon_number",
    # SBS
    "effective_length_m", "sbs_threshold_W", "sbs_gain_exponent",
    # SRS
    "srs_threshold_W", "srs_stokes_wavelength_m", "raman_gain_coefficient", "srs_gain_exponent",
    # TMI
    "tmi_threshold_W", "TMI_C0_DEFAULT",
    # Rayleigh / double-Rayleigh MPI
    "rayleigh_alpha_per_m", "capture_fraction", "double_rayleigh_mpi",
    "mpi_beat_variance_ratio", "mpi_rin_per_hz", "mpi_power_penalty_dB",
]

# Silica defaults (docs sec.8 header). n = refractive index; v_a = longitudinal acoustic
# velocity [m/s]; the effective (not ideal-peak) Brillouin gain; the Raman peak gain and shift.
N_SILICA = 1.45
V_ACOUSTIC_SILICA = 5960.0
G_B_EFFECTIVE = 2.0e-11          # m/W, real GeO2-SMF effective peak (ideal pure silica ~5e-11)
DNU_B_REF_HZ = 30.0e6            # Brillouin gain FWHM at LAMBDA_B_REF (20-50 MHz literature spread)
LAMBDA_B_REF_M = 1.55e-6
G_R_PEAK = 1.0e-13              # m/W, co-polarized Raman peak gain at LAMBDA_R_REF
LAMBDA_R_REF_M = 1.0e-6
RAMAN_SHIFT_HZ = 13.2e12        # peak Stokes shift (~440 cm^-1)


# ============================ Brillouin spectroscopy ======================================

def brillouin_shift_hz(lambda_m, n: float = N_SILICA, v_a: float = V_ACOUSTIC_SILICA):
    """Brillouin frequency shift nu_B = 2 n v_a / lambda [Hz] -- the frequency of the acoustic
    phonon (and the down-shift of the backscattered Stokes light) for backscattering off the
    forward-travelling density grating. ~11.1 GHz at 1550 nm, ~16.2 GHz at 1060 nm for fused
    silica (n = 1.45, v_a = 5960 m/s). [Kobyakov, Sauer & Chowdhury, Adv. Opt. Photon. 2, 1
    (2010).]"""
    return 2.0 * float(n) * float(v_a) / np.asarray(lambda_m, dtype=np.float64)


def brillouin_linewidth_hz(lambda_m, dnu_ref_hz: float = DNU_B_REF_HZ,
                           lambda_ref_m: float = LAMBDA_B_REF_M):
    """Brillouin gain FWHM dnu_B [Hz] = dnu_ref (lambda_ref / lambda)^2 -- the Lorentzian
    linewidth of the Brillouin gain, set by the acoustic-phonon lifetime, scaling as 1/lambda^2.
    Anchored at ~30 MHz for 1550 nm (the literature spread is a broad 20-50 MHz depending on
    core composition and draw); this gives ~64 MHz at 1060 nm. This is the linewidth that both
    sets the thermal-seed bandwidth and enters the source-linewidth SBS-suppression factor
    (1 + dnu_source / dnu_B)."""
    return float(dnu_ref_hz) * (float(lambda_ref_m) / np.asarray(lambda_m, dtype=np.float64)) ** 2


def brillouin_phonon_number(lambda_m, T_K: float = 300.0, n: float = N_SILICA,
                            v_a: float = V_ACOUSTIC_SILICA) -> float:
    """Thermal-equilibrium acoustic-phonon occupation n_th = 1 / (exp(h nu_B / kT) - 1) at the
    Brillouin shift nu_B(lambda). Because h nu_B << kT at GHz shifts (h nu_B / kT ~ 1.8e-3 at
    11 GHz / 300 K), n_th ~ kT / (h nu_B) ~ 560 -- the SBS Stokes wave is seeded by HUNDREDS of
    thermal phonons per mode, not the single spontaneous photon of a high-frequency process.
    Uses expm1 for accuracy in that small-argument regime. [Boyd, Rzazewski & Narum, PRA 42,
    5514 (1990).]"""
    nu_b = brillouin_shift_hz(lambda_m, n, v_a)
    return float(1.0 / np.expm1(H_PLANCK * nu_b / (KB * float(T_K))))


# ============================ SBS ==========================================================

def effective_length_m(length_m: float, alpha_per_m: float = 0.0) -> float:
    """Nonlinear effective length L_eff = (1 - exp(-alpha L)) / alpha [m], the loss-weighted
    interaction length that appears in every passive nonlinear threshold. Reduces to the physical
    length L in the loss-free limit (alpha -> 0, handled analytically) and saturates at 1/alpha
    for a long lossy fiber. [Agrawal, Nonlinear Fiber Optics.]"""
    L = float(length_m)
    a = float(alpha_per_m)
    if a <= 0.0:
        return L
    return (1.0 - np.exp(-a * L)) / a


def sbs_threshold_W(a_eff_m2: float, *, g_b: float = G_B_EFFECTIVE, length_m: float,
                    alpha_per_m: float = 0.0, C: float = 21.0, K: float = 1.5,
                    dnu_source_hz: float = 0.0, dnu_b_hz: Optional[float] = None,
                    lambda_m: float = 1.55e-6) -> float:
    """Passive SBS power threshold (Smith criterion) [W]:

        P_th = C K A_eff / (g_B L_eff) * (1 + dnu_source / dnu_B),
        L_eff = (1 - exp(-alpha L)) / alpha.

    C = ln(P_out_Stokes / P_seed) ~ 21 (range 18-21; NOT a fundamental constant -- it is the log
    of the seed-to-output ratio deemed "threshold", so use 19 for a conservative limiter). K is
    the polarization factor: 1 (co-polarized PM), 3/2 (standard randomly-birefringent SMF,
    the default), 2 (fully scrambled). g_B defaults to the effective 2e-11 m/W of real doped SMF.

    The bracket is the source-broadening SUPPRESSION factor: a source with Lorentzian FWHM
    dnu_source (>> the Brillouin interaction bandwidth but << nu_B) raises the threshold by
    1 + dnu_source / dnu_B, because only the fraction dnu_B / dnu_source of the pump spectrum
    overlaps the narrow Brillouin gain. dnu_b_hz defaults to brillouin_linewidth_hz(lambda_m);
    valid ~10-20% for a smooth Lorentzian source, NOT for combs or discrete phase-modulation
    tones. [Smith, Appl. Opt. 11, 2489 (1972); Kobyakov 2010; van Deventer & Boot, JLT 12, 585
    (1994).]"""
    L_eff = effective_length_m(length_m, alpha_per_m)
    if dnu_b_hz is None:
        dnu_b_hz = brillouin_linewidth_hz(lambda_m)
    broaden = 1.0 + float(dnu_source_hz) / float(dnu_b_hz)
    return float(C * K * float(a_eff_m2) / (float(g_b) * L_eff) * broaden)


def sbs_gain_exponent(result: SteadyStateResult, fiber: FiberSpec, signal_lambda_m: float, *,
                      g_b: float = G_B_EFFECTIVE, C: float = 21.0, T_K: float = 300.0) -> dict:
    """Accumulated SBS gain exponent of an ACTIVE amplifier from a converged SteadyStateResult:

        G_B = (g_B / A_eff) integral_0^L P_signal(z) dz     (np.trapezoid over the signal profile)

    P_signal(z) already carries the amplifier gain and loss, so the L_eff of the passive form is
    NOT used (that would double-count the gain). A_eff = pi w^2 from the Gaussian mode-field
    radius (Marcuse) at the signal wavelength. Backward Stokes seeding is thermal:
    P_seed = h nu_s dnu_B n_th with n_th = 1/(exp(h nu_B / kT) - 1); the projected Stokes output
    is P_seed exp(G_B). The single-frequency amplifier is SBS-limited when G_B ~ C (~21), i.e.
    threshold_margin = G_B / C ~ 1. [Kobyakov 2010; Gray et al., Opt. Express 15, 17044 (2007).]

    Returns a dict: G_B, threshold_margin (G_B/C), a_eff_m2, integral_P_dz_Wm, nu_B_hz, dnu_B_hz,
    n_th, P_seed_W, P_stokes_out_W."""
    z, P = _signal_power_profile(result, signal_lambda_m)
    integral_P_dz = float(np.trapezoid(P, z))
    w = float(mode_field_radius_m(fiber.core_radius_m, fiber.na, signal_lambda_m))
    a_eff = float(np.pi * w * w)
    G_B = float(g_b) * integral_P_dz / a_eff

    nu_s = C_LIGHT / float(signal_lambda_m)
    nu_b = float(brillouin_shift_hz(signal_lambda_m))
    dnu_b = float(brillouin_linewidth_hz(signal_lambda_m))
    n_th = brillouin_phonon_number(signal_lambda_m, T_K)
    P_seed = H_PLANCK * nu_s * dnu_b * n_th
    return {
        "G_B": G_B,
        "threshold_margin": G_B / float(C),
        "a_eff_m2": a_eff,
        "integral_P_dz_Wm": integral_P_dz,
        "nu_B_hz": nu_b,
        "dnu_B_hz": dnu_b,
        "n_th": n_th,
        "P_seed_W": float(P_seed),
        "P_stokes_out_W": float(P_seed * np.exp(G_B)),
    }


# ============================ SRS ==========================================================

def srs_threshold_W(a_eff_m2: float, *, g_r: float, length_m: float, alpha_per_m: float = 0.0,
                    direction: str = "forward", C_forward: float = 16.0,
                    C_backward: float = 20.0) -> float:
    """Passive SRS power threshold (Smith criterion) [W]:

        P_th = C_R A_eff / (g_R L_eff),   L_eff = (1 - exp(-alpha L)) / alpha.

    C_R ~ 16 for the forward (co-propagating) Stokes wave and ~20 for the backward Stokes wave;
    the extra critical gain of the backward geometry lifts its threshold. SRS carries no source-
    linewidth factor (the ~5-6 THz Raman gain FWHM dwarfs any practical source linewidth) and no
    polarization K prefactor is folded in here (use the orthogonal g_R ~ g_R/3 if the signal is
    depolarized). [Smith, Appl. Opt. 11, 2489 (1972); Agrawal, Nonlinear Fiber Optics.]"""
    L_eff = effective_length_m(length_m, alpha_per_m)
    d = direction.lower()
    if d.startswith("f"):
        C_R = C_forward
    elif d.startswith("b"):
        C_R = C_backward
    else:
        raise ValueError("srs_threshold_W: direction must be 'forward' or 'backward'")
    return float(C_R * float(a_eff_m2) / (float(g_r) * L_eff))


def srs_stokes_wavelength_m(pump_lambda_m: float, shift_hz: float = RAMAN_SHIFT_HZ) -> float:
    """First Stokes wavelength for a Raman pump at pump_lambda_m: nu_S = nu_pump - shift,
    lambda_S = c / nu_S [m]. The default 13.2 THz is the silica Raman gain peak; the Stokes light
    is red-shifted (longer wavelength) from the pump (1064 nm -> ~1116 nm; 1550 nm -> ~1660 nm).
    [Agrawal, Nonlinear Fiber Optics.]"""
    nu_pump = C_LIGHT / float(pump_lambda_m)
    nu_stokes = nu_pump - float(shift_hz)
    if nu_stokes <= 0.0:
        raise ValueError("srs_stokes_wavelength_m: shift exceeds pump frequency")
    return float(C_LIGHT / nu_stokes)


def raman_gain_coefficient(pump_lambda_m: float, g_r_peak: float = G_R_PEAK,
                           lambda_ref_m: float = LAMBDA_R_REF_M) -> float:
    """Peak Raman gain coefficient g_R [m/W] scaled from the 1.0-um reference by the inverse-pump-
    wavelength law g_R(lambda) = g_R_peak (lambda_ref / lambda_pump): ~1.0e-13 at 1.0 um, ~0.94e-13
    at 1.064 um, ~0.65e-13 at 1.55 um. [Stolen & Ippen, APL 22, 276 (1973).]"""
    return float(g_r_peak) * float(lambda_ref_m) / float(pump_lambda_m)


def srs_gain_exponent(result: SteadyStateResult, fiber: FiberSpec, signal_lambda_m: float, *,
                      g_r: Optional[float] = None, C: float = 16.0) -> dict:
    """Accumulated SRS gain exponent of an ACTIVE amplifier from a converged SteadyStateResult
    (the signal acts as the Raman pump for its own first Stokes band):

        G_R = (g_R / A_eff) integral_0^L P_signal(z) dz,     threshold at G_R ~ C (~16).

    g_R defaults to the wavelength-scaled peak raman_gain_coefficient(signal_lambda_m). A_eff =
    pi w^2 (Marcuse) at the signal wavelength; the co-propagating-Stokes A_eff_R (two-mode
    overlap) is approximated by this single-mode A_eff here. The Stokes band sits 13.2 THz to the
    red. [Kobyakov 2010; Agrawal, Nonlinear Fiber Optics.]

    Returns a dict: G_R, threshold_margin (G_R/C), g_R, a_eff_m2, integral_P_dz_Wm,
    stokes_lambda_m."""
    z, P = _signal_power_profile(result, signal_lambda_m)
    integral_P_dz = float(np.trapezoid(P, z))
    w = float(mode_field_radius_m(fiber.core_radius_m, fiber.na, signal_lambda_m))
    a_eff = float(np.pi * w * w)
    if g_r is None:
        g_r = raman_gain_coefficient(signal_lambda_m)
    G_R = float(g_r) * integral_P_dz / a_eff
    return {
        "G_R": G_R,
        "threshold_margin": G_R / float(C),
        "g_R": float(g_r),
        "a_eff_m2": a_eff,
        "integral_P_dz_Wm": integral_P_dz,
        "stokes_lambda_m": srs_stokes_wavelength_m(signal_lambda_m),
    }


# ============================ TMI ==========================================================

def _calibrate_tmi_c0() -> float:
    """Pin the TMI estimator constant C0 to a single measured point: a 20-um core (d_core =
    20 um), lambda_s = 1.06 um, eta_heat = 0.09, Gamma_ov = 0.5, kappa = 1.38 W/m/K, dn/dT =
    1.2e-5 /K sits at 1.0 kW (a representative 20/400 co-pumped Yb value). C0 =
    P_th eta_heat (dn/dT) Gamma_ov / ((lambda_s / d_core)^2 kappa)."""
    P_th, d_core, lam, eta, gov, kappa, dndt = 1000.0, 20e-6, 1.06e-6, 0.09, 0.5, 1.38, 1.2e-5
    return P_th * eta * dndt * gov / ((lam / d_core) ** 2 * kappa)


TMI_C0_DEFAULT = _calibrate_tmi_c0()   # ~0.1393 (dimensionless), calibrated at 1 kW / 20-400 Yb


def tmi_threshold_W(core_diameter_m: float, lambda_s_m: float, eta_heat: float, *,
                    gamma_ov: float = 0.5, kappa: float = 1.38, dndt: float = 1.2e-5,
                    C0: Optional[float] = None) -> float:
    """Transverse-mode-instability average-power threshold estimator [W]:

        P_th = C0 (lambda_s / d_core)^2 kappa / (eta_heat (dn/dT) Gamma_ov).

    The moving thermal (STRS) grating written by the LP01xLP11 beat plus quantum-defect heat
    couples power out of the fundamental mode above P_th. eta_heat is the fraction of signal power
    dumped as heat (>= the quantum defect; ~0.09 for a real 976->1030 Yb amplifier once background
    loss and out-of-band ASE are added); Gamma_ov ~ 0.3-0.7 is the LP01-LP11 thermo-optic overlap;
    kappa (silica ~1.38 W/m/K) and dn/dT (~1.2e-5 /K) are the host thermal / thermo-optic
    constants. C0 defaults to TMI_C0_DEFAULT (pinned to 1 kW at a 20/400 Yb; expose C0 to
    re-pin to another datum).

    ABSOLUTE ACCURACY IS 2-3x ONLY -- trust the SCALING (P_th falls as core diameter grows, falls
    with heat fraction, rises with 1/(dn/dT) and 1/Gamma_ov), not the number. See the module
    docstring for why the 85-um rod point (~250-300 W) is not reproduced on the nose by the
    calibration parameters. [Smith & Smith, Opt. Express 19, 10180 (2011); Jauregui, Stihler &
    Limpert, Adv. Opt. Photon. 12, 429 (2020); Eidam, Opt. Express 19, 13218 (2011).]"""
    if C0 is None:
        C0 = TMI_C0_DEFAULT
    return float(C0 * (float(lambda_s_m) / float(core_diameter_m)) ** 2 * float(kappa)
                 / (float(eta_heat) * float(dndt) * float(gamma_ov)))


# ============================ Rayleigh / double-Rayleigh MPI ================================

def rayleigh_alpha_per_m(lambda_m, A_R_dB_km_um4: float = 0.9):
    """Rayleigh-scattering attenuation coefficient alpha_R [1/m] from the empirical law
    alpha_R[dB/km] = A_R / lambda_um^4, converted to natural units (x ln(10)/1e4). A_R ~ 0.7-1.0
    dB/km/um^4 (default 0.9). This is the Rayleigh PART of the fiber loss (~0.12-0.15 of the
    0.18-0.20 dB/km SMF total at 1550 nm), i.e. ~3.6e-5 /m at 1550 nm and ~1.6e-4 /m at 1060 nm.
    The backward-captured fraction of it (alpha_R S) is what drives double-Rayleigh MPI."""
    lam_um = np.asarray(lambda_m, dtype=np.float64) * 1e6
    alpha_db_km = float(A_R_dB_km_um4) / lam_um ** 4
    return alpha_db_km * np.log(10.0) / 1.0e4


def capture_fraction(lambda_m, mode_field_radius_m_val, n: float = N_SILICA):
    """Guided-mode Rayleigh recapture fraction S = (3/2) (lambda / (2 pi n w0))^2 -- the fraction
    of Rayleigh-scattered power recaptured into the BACKWARD guided mode, from the dipole-
    radiation overlap with the Gaussian mode of field radius w0. Typically 1e-3 to 1.6e-3
    (-28 to -33 dB); the prefactor scatters in the literature, so S is exposed as an override in
    double_rayleigh_mpi. [Brinkmeyer, Appl. Opt. 19, 3574 (1980); Nakazawa, JOSA 73, 1175
    (1983).]"""
    lam = np.asarray(lambda_m, dtype=np.float64)
    w0 = np.asarray(mode_field_radius_m_val, dtype=np.float64)
    return 1.5 * (lam / (2.0 * np.pi * float(n) * w0)) ** 2


def double_rayleigh_mpi(result: SteadyStateResult, fiber: FiberSpec, signal_lambda_m: float, *,
                        S: Optional[float] = None, alpha_R: Optional[float] = None) -> float:
    """Double-Rayleigh multi-path-interference ratio R_MPI (crosstalk power / signal power) from a
    converged SteadyStateResult:

        R_MPI = (alpha_R S)^2 integral_0^L dz2 integral_0^{z2} dz1 (P(z2) / P(z1))^2,

    using the identity that the net field gain between the forward scatter point z1 and the deeper
    backward scatter point z2 is exactly the signal power ratio G(z1, z2) = P(z2)/P(z1) (the
    common-path gain to and from the scatter pair cancels). The double integral over the triangle
    z1 < z2 is evaluated on the SteadyStateResult mesh with an outer P(z2)/P(z1) ratio matrix,
    a triangular (lower) mask, and trapezoid weights (M ~ 201, so the (M, M) matrix is cheap).

    S defaults to the guided-mode capture_fraction at the Marcuse mode radius; alpha_R defaults to
    rayleigh_alpha_per_m(signal_lambda_m). For a passive uniform fiber (flat P) this reduces to
    the closed form (alpha_R S)^2 L^2 / 2. MPI below -40 dB (R_MPI < 1e-4) is negligible.
    [Bromage, JLT 22, 79 (2004); Fludger & Mears, JLT 19, 536 (2001).]"""
    z, P = _signal_power_profile(result, signal_lambda_m)
    if alpha_R is None:
        alpha_R = float(rayleigh_alpha_per_m(signal_lambda_m))
    if S is None:
        w0 = float(mode_field_radius_m(fiber.core_radius_m, fiber.na, signal_lambda_m))
        S = float(capture_fraction(signal_lambda_m, w0))

    M = z.size
    # ratio[i, j] = P(z_j) / P(z_i)  (i indexes the forward scatter z1, j the deeper z2)
    ratio = P[np.newaxis, :] / P[:, np.newaxis]
    integrand = ratio * ratio                                   # (P(z2)/P(z1))^2
    dz = np.diff(z)
    # inner integral over z1 in [0, z2] for every column j: cumulative trapezoid down axis 0
    col_incr = 0.5 * (integrand[:-1, :] + integrand[1:, :]) * dz[:, np.newaxis]  # (M-1, M)
    cum = np.empty((M, M))
    cum[0, :] = 0.0
    cum[1:, :] = np.cumsum(col_incr, axis=0)                    # cum[i, j] = int_0^{z_i} ... dz1
    inner = np.diagonal(cum).copy()                            # triangular mask: z1 <= z2 = z_j
    R_MPI = (float(alpha_R) * float(S)) ** 2 * float(np.trapezoid(inner, z))
    return R_MPI


def mpi_beat_variance_ratio(R_MPI: float, B_e: float, dnu_drb: float, k_pol: float = 1.0) -> float:
    """Developed-regime MPI beat-noise variance-to-signal ratio sigma^2 / I^2 =
    k_pol R_MPI (2 B_e / dnu_DRB): the double-Rayleigh field beats with the signal within the
    electrical bandwidth B_e, and only the fraction 2 B_e / dnu_DRB of the DRB bandwidth dnu_DRB
    lands in-band. k_pol = 1 is the worst (co-polarized) case; ~1/2-2/3 for a scrambled state (the
    doubly-scattered light retains a DOP ~ 1/9). [Bromage, JLT 22, 79 (2004).]"""
    return float(k_pol) * float(R_MPI) * (2.0 * float(B_e) / float(dnu_drb))


def mpi_rin_per_hz(R_MPI: float, f_hz, dnu_source_hz: float):
    """MPI relative-intensity-noise spectral density [1/Hz]:
    RIN_MPI(f) = (4 R_MPI / (pi dnu_s)) / (1 + (f / dnu_s)^2) -- a Lorentzian of half-width equal
    to the source linewidth dnu_s (the beat spectrum of two copies of a Lorentzian-linewidth
    source), area 2 R_MPI. [Gimlett & Cheung, JLT 7, 888 (1989).]"""
    f = np.asarray(f_hz, dtype=np.float64)
    dnu = float(dnu_source_hz)
    return (4.0 * float(R_MPI) / (np.pi * dnu)) / (1.0 + (f / dnu) ** 2)


def mpi_power_penalty_dB(sigma2_over_I2: float, Q: float = 6.0) -> float:
    """Receiver power penalty [dB] from MPI beat noise: PP = -5 log10(1 - Q^2 sigma^2 / I^2),
    with Q = 6 at a 1e-9 error rate. Diverges as Q^2 sigma^2/I^2 -> 1 (an MPI noise floor the
    receiver cannot out-power); < 0.5 dB for the -40 dB-MPI acceptance threshold. [Gimlett &
    Cheung, JLT 7, 888 (1989).]"""
    arg = 1.0 - float(Q) ** 2 * float(sigma2_over_I2)
    if arg <= 0.0:
        return float("inf")
    return float(-5.0 * np.log10(arg))


# ============================ internal helpers =============================================

def _signal_power_profile(result: SteadyStateResult, signal_lambda_m: float):
    """(z_m, P_signal(z)) for the 'signal' channel of the SteadyStateResult nearest
    signal_lambda_m. Powers are floored at a tiny positive value so the P(z2)/P(z1) ratios (SBS,
    SRS, DRB) never divide by zero."""
    kind = result.kind
    sig = [k for k in range(len(kind)) if kind[k] == "signal"]
    if not sig:
        raise ValueError("_signal_power_profile: SteadyStateResult has no 'signal' channel")
    lam = np.asarray(result.lambda_m, dtype=np.float64)
    idx = sig[int(np.argmin(np.abs(lam[sig] - float(signal_lambda_m))))]
    P = np.maximum(np.asarray(result.power_W[idx], dtype=np.float64), 1e-300)
    return np.asarray(result.z_m, dtype=np.float64), P
