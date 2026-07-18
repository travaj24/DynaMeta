"""Two-tone third-order intermodulation distortion (IMD3), OIP3 and SFDR for the QD-SOA
(roadmap SOA analog-link metrics; dossier Topic 4).

An analog RF-over-fibre / microwave-photonic link driven through a saturating SOA generates
third-order intermodulation products from the carrier-density pulsation the beating tones
imprint on the gain. Two equal-amplitude optical tones at f1, f2 (spacing Omega = 2 pi
(f2 - f1)) beat the gain at Omega; that gain modulation casts sidebands onto each tone at
f1 - Omega = 2 f1 - f2 and f2 + Omega = 2 f2 - f1 -- the IM3 products that fall IN BAND and
set the spurious-free dynamic range.

CLOSED FORM (Agrawal, Electron. Lett. 23, 1175 (1987); IEEE JQE 1988; Mecozzi & Mork,
JOSA B 14, 761 (1997)):

    IM3/C = [ (G - 1)/4 * (P_out/P_sat) * H(Omega) ]^2 * (1 + alpha^2)
    H(Omega) = 1 / sqrt(1 + (Omega tau_eff)^2),   tau_eff = tau_s / (1 + P_out/P_sat)

IM3/C is the ratio of the IM3 sideband POWER to the carrier (fundamental) POWER. The three
LOAD-BEARING scalings (the physics this module pins):
  - grows as (P_out/P_sat)^2  (third-order: the sideband FIELD is third order in the tones,
    so its power is (P_out/P_sat)^2 -> a 2:1 slope of IM3/C vs drive in log-log, below saturation),
  - rolls off past the effective carrier knee 1/tau_eff at 6 dB/octave in the sideband FIELD
    (H ~ 1/Omega), hence H^2 ~ 1/Omega^2 -> IM3/C drops by a factor 4 (12 dB, 20 log10 of the
    squared ratio) per octave once Omega tau_eff >> 1,
  - the linewidth-enhancement factor alpha turns the pure amplitude (gain) pulsation into a
    coupled amplitude+phase pulsation, enhancing the products by (1 + alpha^2).

PREFACTOR CAVEAT (dossier CAVEAT + Topic 4): the (G - 1)/4 prefactor is the WEAK-COMPRESSION /
small-gain form. It matches the exact saturating IMD3 (and the numeric Agrawal-Olsson oracle
below) only for modest gain G ~ 2-3; at high gain the true IMD3 SATURATES (the carrier pulsation
depth caps at ~P_out/P_sat/(1+P_out/P_sat), independent of the bare (G-1)), so the literal
(G-1)/4 OVER-estimates -- the numeric oracle exhibits exactly this cap. The SCALING LAWS above are
the load-bearing physics (they are gain-independent); the exact prefactor is per-reference (per-tone
vs total-power convention) and would need a Mecozzi-Mork digit-exact pin. imd3_ratio implements the
literal closed form as specified; imd3_numeric_agrawal_olsson is the independent oracle.

Spurious-free dynamic range from the output third-order intercept OIP3 and the noise floor
(the analog-link figure of merit):

    OIP3 [dBm] = P_out [dBm] - (1/2) * 10 log10(IM3/C)         (3:1 IM3 vs 1:1 carrier intercept)
    SFDR [dB.Hz^(2/3)] = (2/3) (OIP3 - N_floor)                (third-order limited; Guzzon &
                                                                Coldren IEEE JQE 2012)

Pure numpy; SI units (powers in W unless a name ends _dbm/_dbm_hz); ASCII only.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from dynameta.optics.soa.traveling_wave import agrawal_olsson_output

__all__ = ["imd3_ratio", "tau_eff_s", "oip3_dbm", "two_tone_oip3_dbm", "sfdr_db_hz23",
           "imd3_numeric_agrawal_olsson"]


def tau_eff_s(tau_s: float, P_out_W: float, P_sat_W: float) -> float:
    """Saturation-shortened effective carrier lifetime tau_eff = tau_s/(1 + P_out/P_sat) [s].
    The knee of the IMD3 rolloff sits at 1/tau_eff (the fastest rate the inversion can follow the
    beat); it moves UP in frequency as the amplifier saturates (P_out/P_sat rises)."""
    if not (tau_s > 0.0 and P_sat_W > 0.0):
        raise ValueError("tau_eff_s: tau_s and P_sat_W must be > 0")
    return float(tau_s / (1.0 + P_out_W / P_sat_W))


def imd3_ratio(G: float, P_out_W: float, P_sat_W: float, omega_beat_rad_s: float,
               tau_s: float, alpha_lef: float) -> float:
    """Closed-form third-order intermodulation-to-carrier POWER ratio IM3/C (linear, dimensionless):

        IM3/C = [ (G-1)/4 * (P_out/P_sat) * H ]^2 * (1 + alpha^2),
        H = 1/sqrt(1 + (Omega tau_eff)^2),  tau_eff = tau_s/(1 + P_out/P_sat).

    G the amplifier power gain (linear), P_out the output signal power [W], P_sat the saturation
    power [W], omega_beat the tone-spacing angular frequency [rad/s], tau_s the carrier lifetime [s],
    alpha_lef the linewidth-enhancement factor. See the module docstring for the (G-1)/4 weak-gain
    caveat -- the (P_out/P_sat)^2, 6 dB/octave and (1+alpha^2) SCALINGS are the load-bearing physics
    and are gain-independent. Convert to dBc with 10 log10(IM3/C)."""
    if not (P_sat_W > 0.0 and tau_s > 0.0):
        raise ValueError("imd3_ratio: P_sat_W and tau_s must be > 0")
    teff = tau_eff_s(tau_s, P_out_W, P_sat_W)
    H = 1.0 / np.sqrt(1.0 + (omega_beat_rad_s * teff) ** 2)
    field = 0.25 * (G - 1.0) * (P_out_W / P_sat_W) * H
    return float(field * field * (1.0 + alpha_lef * alpha_lef))


def oip3_dbm(P_out_W: float, imd3_ratio_lin: float) -> float:
    """Output third-order intercept point OIP3 [dBm] from the carrier output power and the
    IM3/C power ratio at that operating point. The carrier rises 1:1 and the IM3 products 3:1
    with drive, so they meet (extrapolated) at

        OIP3 = P_out + (1/2)(C - IM3) = P_out[dBm] - (1/2) * 10 log10(IM3/C).

    Below saturation OIP3 is drive-INDEPENDENT (IM3/C ~ P_out^2 cancels the P_out term) -- the
    intrinsic distortion figure of the amplifier."""
    if not (P_out_W > 0.0 and imd3_ratio_lin > 0.0):
        raise ValueError("oip3_dbm: P_out_W and imd3_ratio_lin must be > 0")
    p_out_dbm = 10.0 * np.log10(P_out_W / 1.0e-3)
    return float(p_out_dbm - 5.0 * np.log10(imd3_ratio_lin))


def two_tone_oip3_dbm(G: float, P_out_grid_W, P_sat_W: float, omega_beat_rad_s: float,
                      tau_s: float, alpha_lef: float) -> float:
    """OIP3 [dBm] from a two-tone drive SWEEP: evaluate the closed-form IM3/C over the output-power
    grid P_out_grid_W (kept below saturation), take the OIP3 at each point and return the median
    (below saturation every point yields the same intercept, so the median is a robust estimate that
    ignores any near-saturation outliers). This is the OIP3 the datasheet-style SFDR quotes."""
    P = np.atleast_1d(np.asarray(P_out_grid_W, dtype=np.float64))
    ois = np.array([oip3_dbm(float(Pi), imd3_ratio(G, float(Pi), P_sat_W, omega_beat_rad_s,
                                                    tau_s, alpha_lef)) for Pi in P])
    return float(np.median(ois))


def sfdr_db_hz23(oip3_dbm_val: float, noise_floor_dbm_hz: float) -> float:
    """Third-order spurious-free dynamic range SFDR [dB.Hz^(2/3)] = (2/3)(OIP3 - N_floor), with
    OIP3 and the noise floor both in the dBm scale (N_floor in dBm/Hz). The 2/3 exponent is the
    third-order signature: for each dB the noise floor drops, both the usable top (compression) and
    the spur floor move, and the distortion-free window opens as the 2/3 power of the intercept-to-
    noise span (Guzzon & Coldren IEEE JQE 2012)."""
    return float((2.0 / 3.0) * (oip3_dbm_val - noise_floor_dbm_hz))


def imd3_numeric_agrawal_olsson(G_target: float, P_out_over_Psat: float, omega_beat_rad_s: float,
                                L_m: float, tau_c_s: float, E_sat_J: float, *,
                                n_settle_beats: int = 80, n_meas_beats: int = 80,
                                npb: int = 1024) -> Tuple[float, float, float]:
    """Independent NUMERIC oracle for IM3/C via the exact Agrawal-Olsson lumped saturable-gain model
    (traveling_wave.agrawal_olsson_output) -- no closed-form assumption. Drives the amplifier with a
    genuine two equal-tone field A(t) = sqrt(P1)(exp(i w1 t) + exp(i w2 t)); the Agrawal-Olsson model
    is power-in/power-out, so the input POWER is |A|^2 = 2 P1 (1 + cos(Omega t)), a 100%-depth beat at
    the tone spacing Omega. The gain saturation folds this into harmonics of Omega in the output power;
    the third-order intermodulation product 2 f1 - f2 shows up, after photodetection (|.|^2), as the
    2 Omega component beating against the fundamental Omega component. Hence

        IM3/C (power) = ( |FFT[P_out](2 Omega)| / |FFT[P_out](Omega)| )^2

    (the electrical amplitude ratio at 2 Omega vs Omega is the IM3-field/carrier-field ratio, squared
    to a power ratio). The alpha enhancement is NOT present (a pure power model carries no phase), so
    this validates the closed form at alpha = 0. RETURNS (IM3/C, G_measured, P_out_mean_W).

    The bias tone power P1 is chosen from the unsaturated gain so the saturated mean output lands near
    P_out_over_Psat * P_sat (P_sat = E_sat/tau_c). Keeps dt = (2 pi/Omega)/npb < tau_c/2 for RK4
    stability (raise npb, or Omega, if this trips). Integer measured beats -> the Omega / 2 Omega FFT
    bins are exact (bin = n_meas_beats)."""
    if not (G_target > 1.0 and P_out_over_Psat > 0.0 and omega_beat_rad_s > 0.0):
        raise ValueError("imd3_numeric_agrawal_olsson: need G_target>1, P_out_over_Psat>0, Omega>0")
    P_sat = E_sat_J / tau_c_s
    g0 = np.log(G_target) / L_m                              # unsaturated modal gain [1/m]
    # bias tone power: mean input P_in_bar = 2 P1; target saturated output ~ frac P_sat. Use the
    # unsaturated gain as a first estimate (the measured G is returned so the caller reads the actual
    # operating point rather than relying on this seed).
    P1 = 0.5 * (P_out_over_Psat * P_sat) / G_target
    T = 2.0 * np.pi / omega_beat_rad_s
    dt = T / npb
    if dt >= 0.5 * tau_c_s:
        raise ValueError("imd3_numeric_agrawal_olsson: dt={:.2e}s >= tau_c/2; raise npb or Omega "
                         "(RK4 of the carrier equation is stiff for dt >~ tau_c)".format(dt))
    nt = (n_settle_beats + n_meas_beats) * npb
    t = np.arange(nt) * dt
    P_in = 2.0 * P1 * (1.0 + np.cos(omega_beat_rad_s * t))
    P_out = agrawal_olsson_output(t, P_in, g0, L_m, tau_c_s, E_sat_J)
    tail = P_out[n_settle_beats * npb:]                     # integer meas beats -> clean bins
    F = np.fft.rfft(tail - tail.mean())
    k = n_meas_beats                                        # Omega bin (that many cycles in the tail)
    a1 = np.abs(F[k])
    a2 = np.abs(F[2 * k])
    im3_c = float((a2 / a1) ** 2)
    G_meas = float(tail.mean() / P_in[n_settle_beats * npb:].mean())
    return im3_c, G_meas, float(tail.mean())
