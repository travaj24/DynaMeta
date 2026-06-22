"""Calibrate the QD-SOA model to a measured datasheet -- the step that turns the generic-parameter gain
core into a DEVICE-matched parameter set. The first target is the Innolume BOA1310060CC600MXXXX (a 1310 nm
QD Booster Optical Amplifier on carrier), the very 'Innolume set' QDGainParams' docstring notes does not
yet exist.

WHAT THIS FITS (the STATIC / CW load-bearing axes the datasheet constrains): peak wavelength (read-off),
gain bandwidth (fwhm_inhom), small-signal gain magnitude (sigma_pk, the effective free factor of the
degenerate product Gamma*N_q*mu_GS*sigma_pk*L -- N_q is FIXED at a standard QD value, recorded as the
convention), the absolute saturation output power P_sat (A_mode), and the GS/ES band split (dE_ES_GS,
enabling the two-band ASE). The fit + the validation use only STEADY-STATE physics (the gain core's
saturation_curve gives the local saturated gain g_QD(P); the device output is the z-integral
dP/dz = (Gamma g_QD(P) - alpha_i) P) -- the time-domain marcher OVERFLOWS at this device's 35 dB
single-pass gain, and the steady-state path is exact for the CW datasheet numbers anyway.

WHAT THIS DOES NOT CALIBRATE (no datasheet data -- left at flagged defaults, UNCALIBRATED): the linewidth-
enhancement factor alpha_lef (no chirp/FWM data), the carrier kinetic times tau_cap/tau_esc/tau_ES_GS
(no pump-probe / modulation-bandwidth), RIN / linewidth, NF(lambda)/NF(G), TPA/FCA, and the thermal
slopes. Pin those with a pump-probe gain-recovery trace, an FWM / chirp-asymmetry measurement, an RF-RIN
measurement, a spectral-NF measurement, and gain-vs-temperature data respectively.

SI; ASCII; exp(-i omega t).
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from dynameta.constants import C_LIGHT, Q_E   # single-source CODATA (was re-declared here)
from dynameta.optics.soa.qd_gain import QDGainModel, QDGainParams

H_PLANCK = 6.62607015e-34                     # exact CODATA h (constants.py carries only HBAR)

# Innolume BOA1310060CC600MXXXX datasheet (99-S01-273-01), 25 C, 2000 mA operating point.
INNOLUME_BOA1310_TARGETS = {
    "peak_nm": 1310.0,            # gain peak wavelength
    "G0_dB": 35.0,               # small-signal (Pin=-20 dBm) chip gain, typ
    "bandwidth_nm": 60.0,        # -3 dB gain bandwidth, typ
    "Psat_out_dBm": 23.2,        # -3 dB saturation output power @ 2000 mA (20.3/24.7 at 1/3 A)
    "NF_dB_max": 5.0,            # noise figure @ Pin=-20 dBm, excluding input coupling
    "ase_es_nm": 1210.0,         # ES/WL ASE band (blue of the GS signal band)
    "drive_A": 2.0, "L_m": 8.0e-3, "T_K": 298.0, "facet_R": 1.0e-5,
}


@dataclass(frozen=True)
class CalibratedDevice:
    """A device-matched QD-SOA: the fitted gain params + the device-level constants (length, internal
    loss, operating drive, signal frequency) that the datasheet pins but QDGainParams does not hold."""
    params: QDGainParams
    length_m: float
    alpha_i_per_m: float
    drive_A: float
    nu0_Hz: float
    name: str
    report: dict


def _net_gain_spectrum(model, drive, nu, alpha_i, L):
    """Unsaturated NET chip gain [dB] over the length L at frequencies nu (S_conf -> 0)."""
    y = model.steady_state(drive, S_conf_m3=0.0)
    g = model.material_gain_per_m(model.rho_GS(y), np.atleast_1d(nu))
    return (10.0 / np.log(10.0)) * (model.gamma_confinement * g - alpha_i) * L


def _g0_dB(model, drive, nu0, alpha_i, L):
    return float(_net_gain_spectrum(model, drive, np.array([nu0]), alpha_i, L)[0])


def _bandwidth_nm(model, drive, nu0, alpha_i, L):
    """-3 dB bandwidth [nm] of the MATERIAL gain coefficient g(nu) (half-max of g) -- the INTRINSIC
    gain width (~ the inhomogeneous FWHM), gain-level-independent and consistent with the datasheet's
    visible ~120 nm gain-spectrum span. NB the NET amplifier-gain -3 dB at the 35 dB small-signal peak
    is much NARROWER (high-gain spectral narrowing makes a 3 dB drop a tiny fractional g drop), so
    matching the datasheet 60 nm there would force an unphysically wide distribution -- the intrinsic
    interpretation is the physical one."""
    nu = nu0 + np.linspace(-30e12, 30e12, 1201)
    y = model.steady_state(drive, S_conf_m3=0.0)
    g = model.material_gain_per_m(model.rho_GS(y), nu)
    above = nu[g >= 0.5 * g.max()]
    if above.size < 2:
        return 0.0
    lam = C_LIGHT / nu0
    return float((above.max() - above.min()) * lam * lam / C_LIGHT * 1.0e9)


def device_saturation_curve(model, drive, nu, alpha_i, L, P_in_W, nz=2000):
    """Absolute CW saturation curve P_out(P_in) [W] by STEADY-STATE z-integration (robust at high gain
    where the time-domain marcher overflows). g_QD(P) = the gain core's local saturated material gain at
    local power P (QDGainModel.saturation_curve, steady state at each photon density); the device output
    integrates dP/dz = (Gamma g_QD(P) - alpha_i) P over L by RK4. Returns (P_in_W, P_out_W)."""
    P_in = np.atleast_1d(np.asarray(P_in_W, dtype=np.float64))
    # local saturated gain on a wide power grid bracketing P_in .. P_out
    P_grid = np.logspace(np.log10(P_in.min()) - 1.0, np.log10(P_in.max() * 1.0e4) + 1.0, 400)
    g_loc, _S = model.saturation_curve(drive, P_grid, nu_s_Hz=nu)
    gam = model.gamma_confinement
    gQD = lambda P: np.interp(P, P_grid, g_loc)
    dz = L / int(nz)
    P_out = np.empty(P_in.size)
    for i, P0 in enumerate(P_in):
        P = float(P0)
        for _ in range(int(nz)):
            k1 = (gam * gQD(P) - alpha_i) * P
            k2 = (gam * gQD(P + 0.5 * dz * k1) - alpha_i) * (P + 0.5 * dz * k1)
            k3 = (gam * gQD(P + 0.5 * dz * k2) - alpha_i) * (P + 0.5 * dz * k2)
            k4 = (gam * gQD(P + dz * k3) - alpha_i) * (P + dz * k3)
            P = P + dz / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        P_out[i] = P
    return P_in, P_out


def _psat_out_dBm(model, drive, nu, alpha_i, L):
    """Output -3 dB saturation power [dBm] from the steady-state device saturation curve."""
    P_in = np.logspace(-9.0, 0.0, 40)                       # 1 nW .. 1 W input
    P_in, P_out = device_saturation_curve(model, drive, nu, alpha_i, L, P_in)
    G_dB = 10.0 * np.log10(P_out / P_in)
    G0 = G_dB[0]
    target = G0 - 3.0
    if np.nanmin(G_dB) > target:
        return np.nan, G0
    # output power at the -3 dB compression point (interp on the monotone-decreasing gain)
    logPout = np.log10(P_out)
    log_pout_sat = float(np.interp(target, G_dB[::-1], logPout[::-1]))
    return float(10.0 * np.log10((10.0 ** log_pout_sat) / 1.0e-3)), float(G0)


def calibrate_innolume_boa1310(N_q_m3=5.0e22, alpha_i_per_m=300.0, n_groups=41, verbose=False):
    """Fit the QD-SOA model to the Innolume BOA1310060 datasheet (static/CW axes). N_q_m3 is FIXED at a
    standard QD value (the degenerate product Gamma*N_q*mu_GS*sigma_pk is broken by holding N_q, Gamma,
    mu_GS and fitting sigma_pk -- the EFFECTIVE-PRODUCT convention). alpha_i_per_m is the internal loss
    (only loosely bounded by the NF <= 5 dB; default 300 /m = 3 /cm). Returns a CalibratedDevice.
    Staged fit (each step a 1-param solve; gain magnitude<->bandwidth iterated):
      0 read-offs: nu0=1310 nm, L=8 mm, drive=2 A, T=298 K, ES band ON (dE from the 1210/1310 split);
      1 fwhm_inhom -> 60 nm -3 dB bandwidth;  2 sigma_pk -> 35 dB small-signal gain (iterate 1<->2);
      3 A_mode -> 23.2 dBm output saturation power."""
    t = INNOLUME_BOA1310_TARGETS
    nu0 = C_LIGHT / (t["peak_nm"] * 1.0e-9)
    L, drive = t["L_m"], t["drive_A"]
    dE = H_PLANCK * (C_LIGHT / (t["ase_es_nm"] * 1.0e-9) - nu0) / Q_E   # GS/ES split [eV] (ES blue of GS)

    def mk(fwhm_inh, sig, A_mode, sig_es):
        p = QDGainParams(n_groups=n_groups, nu0_Hz=nu0, fwhm_inhom_Hz=fwhm_inh, sigma_pk_m2=sig,
                         A_mode_m2=A_mode, T_K=t["T_K"], sigma_pk_ES_m2=sig_es, dE_ES_GS_eV=dE,
                         N_q_m3=N_q_m3)
        return QDGainModel(p.with_detailed_balance_taus())

    fwhm, sig, A_mode = 13.0e12, 2.0e-18, 0.4e-12
    sig_es = 0.0                                            # GS-only while fitting the GS leg
    # steps 1<->2: bandwidth (fwhm) and magnitude (sigma_pk, ~linear above the alpha_i offset)
    for _ in range(6):
        m = mk(fwhm, sig, A_mode, sig_es)
        G0 = _g0_dB(m, drive, nu0, alpha_i_per_m, L)
        sig = sig * (t["G0_dB"] + (10.0 / np.log(10.0)) * alpha_i_per_m * L) / (
            G0 + (10.0 / np.log(10.0)) * alpha_i_per_m * L)
        m = mk(fwhm, sig, A_mode, sig_es)
        bw = _bandwidth_nm(m, drive, nu0, alpha_i_per_m, L)
        fwhm = fwhm * (t["bandwidth_nm"] / bw) ** 0.7        # damped update (bw grows ~ with fwhm)
    # step 3: A_mode for P_sat (P_sat ~ linear in A_mode; photon density = P/(A_mode v_g h nu))
    for _ in range(8):
        m = mk(fwhm, sig, A_mode, sig_es)
        ps, _ = _psat_out_dBm(m, drive, nu0, alpha_i_per_m, L)
        if not np.isfinite(ps):
            A_mode *= 0.5                                   # not reaching -3 dB -> smaller A_mode
            continue
        A_mode = A_mode * 10.0 ** ((t["Psat_out_dBm"] - ps) / 10.0)
    # enable the ES band: a plausible strength (ASE curves are 'reference only', not an absolute target)
    sig_es = 0.3 * sig
    m = mk(fwhm, sig, A_mode, sig_es)
    G0 = _g0_dB(m, drive, nu0, alpha_i_per_m, L)
    bw = _bandwidth_nm(m, drive, nu0, alpha_i_per_m, L)
    ps, _ = _psat_out_dBm(m, drive, nu0, alpha_i_per_m, L)
    report = {"G0_dB": G0, "bandwidth_nm": bw, "Psat_out_dBm": ps, "sigma_pk_m2": sig,
              "fwhm_inhom_Hz": fwhm, "A_mode_m2": A_mode, "dE_ES_GS_eV": dE, "N_q_m3": N_q_m3,
              "alpha_i_per_m": alpha_i_per_m}
    if verbose:
        for k, v in report.items():
            print("  {:16s} {}".format(k, v))
    return CalibratedDevice(params=m.p, length_m=L, alpha_i_per_m=alpha_i_per_m, drive_A=drive,
                            nu0_Hz=nu0, name="Innolume BOA1310060 (static/CW fit)", report=report)


@dataclass(frozen=True)
class InferredDynamic:
    """A DYNAMIC parameter INFERRED from the static/CW calibration -- NOT measured. value/unit, the
    confidence (HIGH/MEDIUM/LOW with the reason), and the method/physics used. These are physically-
    motivated ESTIMATES to seed the dynamic phases (18-33) instead of the generic defaults; every one
    must be flagged inferred-not-measured and refined by the named dynamic measurement before any
    dynamic/coherent prediction is trusted."""
    value: float
    unit: str
    confidence: str
    method: str


def infer_dynamics_from_cw(device, *, slow_div_deg=6.0, fast_div_deg=27.0):
    """Infer DYNAMIC parameters from the calibrated static/CW set (device = a CalibratedDevice). Returns a
    dict of InferredDynamic. EACH IS AN ESTIMATE, NOT A MEASUREMENT -- the CW datasheet does not contain
    dynamic data; these exploit physical LINKS between CW observables and dynamic quantities, and carry an
    explicit per-parameter confidence. Inputs beyond the device: the datasheet far-field beam divergence
    (slow 6 deg / fast 27 deg FWHM) used to pin the mode area.

    The chain rests on the textbook SOA relation P_sat = h nu A_eff / (Gamma a tau_eff): the CW P_sat (and
    the divergence-derived A_eff and the model differential gain a) jointly pin tau_eff. The Phase-34 fit
    is left UNTOUCHED (it reproduces every CW number via an EFFECTIVE A_mode that absorbed the a*tau
    degeneracy); these inferred values are the PHYSICAL interpretation of that same saturation."""
    import numpy as _np
    m = QDGainModel(device.params)
    nu0, drive, L = device.nu0_Hz, device.drive_A, device.length_m
    lam = C_LIGHT / nu0
    out = {}

    # --- (1) effective mode area A_eff from the far-field divergence -------------------------------------
    # Gaussian far-field: a 1/e^2-intensity near-field radius w gives a 1/e^2 half-angle theta = lam/(pi w).
    # The datasheet quotes the FWHM full-angle; for a Gaussian, FWHM_full = 1.18 * theta_(1/e2,half), so
    # w = lam / (pi * theta_FWHM_full / 1.18). A_eff (1/e^2 intensity) = pi * w_slow * w_fast.
    # CONFIDENCE: MEDIUM. The geometric mode size from divergence is well-defined (HIGH), but mapping it to
    # the model's effective SATURATION area carries a ~2x convention factor (effective-area vs 1/e^2 area)
    # and assumes a single Gaussian transverse mode (a real ridge mode is only approximately Gaussian).
    th_s = _np.deg2rad(slow_div_deg) / 1.18
    th_f = _np.deg2rad(fast_div_deg) / 1.18
    w_s, w_f = lam / (_np.pi * th_s), lam / (_np.pi * th_f)
    A_eff = float(_np.pi * w_s * w_f)
    out["A_eff_m2"] = InferredDynamic(
        A_eff, "m^2", "MEDIUM (geometric mode size HIGH; ~2x effective-area convention + Gaussian-mode "
        "assumption)", "A_eff = pi*w_s*w_f, w = lambda/(pi*theta_FWHM/1.18) from the 6deg/27deg far-field")

    # --- (2) GS differential gain dg/dN -- DIAGNOSTIC: it reveals GAIN CLAMPING -------------------------
    # Finite-difference the GS material gain vs a small injection step at the operating point; N is the
    # total confined+WL carrier density. At 2 A the GS is ~fully inverted (2 rho_GS - 1 ~ 0.998), so the
    # extra carriers go into the WL/ES RESERVOIR, not the clamped GS -> dg/dN ~ 0. This is NOT a usable
    # differential gain (it would blow the textbook P_sat = h nu A/(Gamma a tau) up); it is the SIGNATURE
    # that the QD gain is CLAMPED and the saturation is RESERVOIR-limited, which is why tau_eff below uses
    # the stimulated CROSS-SECTION form (sigma_pk) instead of dg/dN. CONFIDENCE: HIGH as a clamping
    # diagnostic (dg/dN ~ 0 at the operating point); NOT to be used as a differential-gain value.
    y1 = m.steady_state(drive, S_conf_m3=0.0)
    y2 = m.steady_state(drive * 1.02, S_conf_m3=0.0)
    g1 = float(m.material_gain_per_m(m.rho_GS(y1), nu0))
    g2 = float(m.material_gain_per_m(m.rho_GS(y2), nu0))
    N1, N2 = m.total_carrier_density(y1), m.total_carrier_density(y2)
    a_diff = (g2 - g1) / (N2 - N1) if N2 != N1 else float("nan")
    out["dg_dN_diagnostic_m2"] = InferredDynamic(
        float(a_diff), "m^2 (dg/dN, ~0 = CLAMPED)", "HIGH as a clamping diagnostic (GS inverted ~0.998 at "
        "2 A -> dg/dN ~ 0, reservoir-limited); NOT a usable differential gain",
        "finite-difference dg/dN at the operating point (clamped)")

    # --- (3) effective gain-recovery / saturation time tau_eff (cross-section form) --------------------
    # For a CLAMPED QD gain the saturation is set by the stimulated CROSS-SECTION, not dg/dN: a dot's
    # stimulated rate is sigma_pk * v_g * S_conf, the confined photon density is S_conf = Gamma P /
    # (A_eff v_g h nu), and saturation occurs when that rate balances the recovery 1/tau_eff. Setting
    # sigma_pk * Gamma * P_sat / (A_eff h nu) = 1/tau_eff gives
    #     tau_eff = A_eff h nu / (Gamma sigma_pk P_sat),
    # using the PHYSICAL A_eff (1), the FITTED sigma_pk, and the calibrated P_sat. For this device this is
    # ~100 ps -- the QD reservoir-refill gain-recovery time that governs pattern effects / XGM, NOT the slow
    # ns carrier lifetime and NOT the sub-ps SHB/carrier-heating (both are separate timescales absent from
    # a CW P_sat). CONFIDENCE: MEDIUM for the order of magnitude -- it inherits the A_eff ~2x convention and
    # the Gamma/sigma_pk degeneracy from the gain fit, so trust the ~10^2 ps SCALE, not the digits; pin it
    # with a pump-probe gain-recovery trace.
    Psat_W = 10.0 ** (device.report["Psat_out_dBm"] / 10.0) * 1.0e-3
    gam, sig = m.gamma_confinement, float(device.params.sigma_pk_m2)
    tau_eff = (H_PLANCK * nu0 * A_eff) / (gam * sig * Psat_W) if (sig and Psat_W) else float("nan")
    out["tau_eff_s"] = InferredDynamic(
        float(tau_eff), "s", "MEDIUM (order of magnitude ~100 ps; inherits A_eff ~2x + the Gamma/sigma_pk "
        "degeneracy; the fast reservoir-refill recovery, NOT the slow lifetime or the sub-ps SHB/CH)",
        "tau_eff = A_eff h nu / (Gamma sigma_pk P_sat), cross-section saturation form (QD gain is clamped)")

    # --- (4) small-signal modulation / gain-recovery 3 dB frequency (order of magnitude) ---------------
    # The CW-saturation time sets the low-frequency recovery: f_3dB ~ 1/(2 pi tau_eff). CONFIDENCE: LOW --
    # this is only the SLOW envelope; the true high-speed response is dominated by the ps reservoir dynamics
    # (unmeasured), so the real modulation bandwidth is HIGHER than this estimate.
    f3 = 1.0 / (2.0 * _np.pi * tau_eff) if tau_eff and _np.isfinite(tau_eff) else float("nan")
    out["f_3dB_slow_Hz"] = InferredDynamic(
        float(f3), "Hz", "LOW (slow-envelope only; the ps reservoir dynamics raise the true bandwidth)",
        "f ~ 1/(2 pi tau_eff)")

    # --- (5) linewidth enhancement factor alpha (NOT reliably inferable) --------------------------------
    # alpha = -dn'/dn'' is the carrier derivative of the KK-paired index/gain. Inferring it from the CW gain
    # would need the gain ASYMMETRY (the carrier-induced index slope), but the fitted GS gain is a SYMMETRIC
    # Gaussian comb -> its Kramers-Kronig index change is ANTISYMMETRIC and crosses ZERO at the 1310 nm peak
    # -> the KK estimate of alpha at the operating wavelength is ~0, a trivial (useless) lower bound. The
    # real alpha (1-3 for QD near the GS) comes from the asymmetric WL/ES background the symmetric fit omits.
    # CONFIDENCE: LOW / effectively NOT inferable from this calibration. alpha stays at the flagged default
    # (alpha_lef=2.0); pin it with an FWM up/down-asymmetry or AM/PM-chirp measurement.
    out["alpha_lef"] = InferredDynamic(
        float(device.params.alpha_lef), "-", "LOW / NOT inferable (KK of the symmetric fitted gain gives "
        "~0 at the peak; the real alpha needs the gain asymmetry / a measurement)",
        "KK of the CW gain is antisymmetric -> ~0 at peak; default 2.0 retained as a placeholder")

    # --- NOTE on the kinetic rate RATIOS (HIGH, already applied) -----------------------------------------
    # The forward/backward kinetic-rate RATIOS (tau_esc/tau_cap and tau_GS_ES/tau_ES_GS) are pinned by
    # DETAILED BALANCE given the GS/ES energy separation dE_ES_GS = 0.078 eV, which WAS calibrated from the
    # 1210/1310 nm two-band ASE split and is ALREADY APPLIED in the fitted params (.with_detailed_balance_
    # taus()). CONFIDENCE: HIGH for the ratios (exact detailed balance); the ABSOLUTE kinetic times remain
    # uncalibrated (only their ratios + the aggregate tau_eff above are constrained).
    out["dE_ES_GS_eV"] = InferredDynamic(
        float(device.params.dE_ES_GS_eV), "eV", "HIGH for the detailed-balance rate RATIOS it fixes (from "
        "the CW two-band ASE); absolute kinetic times still uncalibrated",
        "dE from the 1210/1310 nm ASE split -> detailed-balance forward/backward tau ratios")
    return out
