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

from dynameta.optics.soa.qd_gain import QDGainModel, QDGainParams

C_LIGHT = 2.99792458e8
Q_E = 1.602176634e-19
H_PLANCK = 6.62607015e-34

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
