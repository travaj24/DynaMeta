"""Analog-channel metrics for the QD-SOA gain leg (roadmap SOA Phase 4): turn the gain
transfer curve + the ASE noise floor into SFDR / ENOB and the optimal drive power.

The incoherent OVMM encodes the value in INTENSITY, so the figure of merit is the usable
analog window: distortion (gain compression -> harmonics) sets the ceiling, the ASE/detector
beat noise sets the floor, and there is an OPTIMAL drive between them (back off from
saturation) -- the spec's "window, not a wall." This module is solver-agnostic: it consumes
a sampled transfer curve P_out(P_in) and a noise-variance callable, so it pairs with the
QDGainModel saturation curve (static) or a traveling-wave transfer curve (dynamic).

Harmonic distortion from the transfer-curve curvature: for P_in = P0 (1 + m sin wt), a local
cubic fit T(P) about P0 gives, with A = P0 m,
    fundamental  c1 = T' A + (1/8) T''' A^3
    2nd harmonic c2 = (1/4) T'' A^2
    3rd harmonic c3 = (1/24) T''' A^3
The detected (photocurrent) signal and distortion variances are R^2/2 times c1^2 and
(c2^2 + c3^2); SNDR = signal/(noise + distortion); ENOB = (SNDR_dB - 1.76)/6.02.

Pure numpy; SI units.
"""

from __future__ import annotations

import numpy as np

__all__ = ["transfer_derivatives", "harmonic_amplitudes", "sndr_db", "enob",
           "sndr_vs_drive", "optimal_drive_power", "predistort", "pattern_penalty_dB",
           "sfdr_dB", "thermal_drift_budget_K", "facet_gain_ripple_dB", "ripple_enob_ceiling"]


def transfer_derivatives(P_in_grid, P_out_grid, P0):
    """Local (T', T'', T''') of the transfer curve P_out(P_in) at P0 via a cubic fit to the
    nearest samples -- the curvature that produces harmonic distortion."""
    P_in = np.asarray(P_in_grid, dtype=np.float64)
    P_out = np.asarray(P_out_grid, dtype=np.float64)
    order = np.argsort(P_in)
    P_in, P_out = P_in[order], P_out[order]
    i = int(np.clip(np.searchsorted(P_in, P0), 3, P_in.size - 4))
    sl = slice(i - 3, i + 4)
    c = np.polyfit(P_in[sl] - P0, P_out[sl], 3)               # c[0] x^3 + c[1] x^2 + c[2] x + c[3]
    return float(c[2]), float(2.0 * c[1]), float(6.0 * c[0])  # T', T'', T'''


def harmonic_amplitudes(Tp, Tpp, Tppp, A):
    """Fundamental / 2nd / 3rd harmonic AMPLITUDES (magnitudes; the cos2wt/sin3wt signs are
    dropped, which is immaterial since SNDR uses them squared) in P_out for a drive amplitude
    A about the bias (A = P0 * modulation_index)."""
    c1 = Tp * A + (1.0 / 8.0) * Tppp * A ** 3
    c2 = 0.25 * Tpp * A ** 2
    c3 = (1.0 / 24.0) * Tppp * A ** 3
    return c1, c2, c3


def sndr_db(signal_var, noise_var, distortion_var):
    denom = max(noise_var + distortion_var, 1e-300)
    return float(10.0 * np.log10(max(signal_var, 1e-300) / denom))


def enob(sndr_decibels):
    """Effective number of bits from SNDR [dB]: ENOB = (SNDR - 1.76)/6.02."""
    return float((sndr_decibels - 1.76) / 6.02)


def sndr_vs_drive(P_in_grid, P_out_grid, noise_var_of_Pout, P0_array, *, mod_index=0.3,
                  R_A_W=1.0):
    """SNDR [dB] vs bias drive power P0 for a sinusoidally intensity-modulated signal.
    `noise_var_of_Pout(P_out)` returns the detector noise variance [A^2] at the detected mean
    output power. Returns (sndr_dB[P0], enob[P0], idx_opt) -- the interior maximum is the
    optimal analog operating point (noise-limited below, distortion-limited above)."""
    P0s = np.atleast_1d(np.asarray(P0_array, dtype=np.float64))
    sndr = np.empty(P0s.size)
    for k, P0 in enumerate(P0s):
        Tp, Tpp, Tppp = transfer_derivatives(P_in_grid, P_out_grid, float(P0))
        A = P0 * mod_index
        c1, c2, c3 = harmonic_amplitudes(Tp, Tpp, Tppp, A)
        sig = 0.5 * (R_A_W * c1) ** 2
        dist = 0.5 * (R_A_W * c2) ** 2 + 0.5 * (R_A_W * c3) ** 2
        P_out0 = float(np.interp(P0, np.sort(P_in_grid),
                                 np.asarray(P_out_grid)[np.argsort(P_in_grid)]))
        noise = float(noise_var_of_Pout(P_out0))
        sndr[k] = sndr_db(sig, noise, dist)
    return sndr, np.array([enob(s) for s in sndr]), int(np.argmax(sndr))


def optimal_drive_power(P_in_grid, P_out_grid, noise_var_of_Pout, P0_array, *,
                        mod_index=0.3, R_A_W=1.0):
    """The drive power that maximizes SNDR (the analog operating point) and its (SNDR, ENOB).
    The interior optimum trades the ASE-noise floor (dominant at low drive) against gain-
    compression distortion (dominant at high drive)."""
    sndr, eno, iopt = sndr_vs_drive(P_in_grid, P_out_grid, noise_var_of_Pout, P0_array,
                                    mod_index=mod_index, R_A_W=R_A_W)
    P0s = np.atleast_1d(np.asarray(P0_array, dtype=np.float64))
    return float(P0s[iopt]), float(sndr[iopt]), float(eno[iopt])


def predistort(P_in_grid, P_out_grid, P_out_target):
    """Inverse transfer: the input power(s) that, through the compression curve P_out(P_in),
    yield the requested output P_out_target -- i.e. invert gain compression to linearize the
    channel. Operates on the MONOTONE-rising branch (sorted by P_in): if the curve rolls over
    in deep saturation it is truncated at the peak with a RuntimeWarning (the conjugate
    descending branch is not invertible), so a flattened curve cannot silently blend the two
    branches. Targets outside the invertible range clip to its ends."""
    P_in = np.asarray(P_in_grid, dtype=np.float64)
    P_out = np.asarray(P_out_grid, dtype=np.float64)
    o = np.argsort(P_in)                                       # sort by the independent variable
    P_in, P_out = P_in[o], P_out[o]
    imax = int(np.argmax(P_out))
    if imax < P_out.size - 1:                                  # rollover: keep the rising branch
        import warnings
        warnings.warn("predistort: P_out(P_in) rolls over (deep saturation); inverting only "
                      "the monotone-rising branch up to the peak", RuntimeWarning, stacklevel=2)
        P_in, P_out = P_in[:imax + 1], P_out[:imax + 1]
    return np.interp(np.asarray(P_out_target, dtype=np.float64), P_out, P_in)


def pattern_penalty_dB(mark_peaks):
    """Pattern penalty [dB] of an amplified mark sequence: the peak-to-peak spread of the
    per-'1' output peaks (gain-recovery memory imprints a pattern-dependent amplitude). 0 dB
    is pattern-free."""
    pk = np.asarray(mark_peaks, dtype=np.float64)
    if pk.size == 0 or pk.min() <= 0.0:
        raise ValueError("pattern_penalty_dB: need positive mark peaks")
    return float(10.0 * np.log10(pk.max() / pk.min()))


def sfdr_dB(P_in_grid, P_out_grid, P0, noise_var, *, mod_index=0.3, R_A_W=1.0):
    """Spurious-free dynamic range [dB] at drive P0: the ratio (in dB) of the fundamental to
    the larger of the 2nd/3rd compression harmonics, floored at the noise level -- the usable
    distortion-free window above the noise."""
    Tp, Tpp, Tppp = transfer_derivatives(P_in_grid, P_out_grid, float(P0))
    c1, c2, c3 = harmonic_amplitudes(Tp, Tpp, Tppp, P0 * mod_index)
    sig = 0.5 * (R_A_W * c1) ** 2
    spur = max(0.5 * (R_A_W * c2) ** 2, 0.5 * (R_A_W * c3) ** 2, float(noise_var))
    return float(10.0 * np.log10(max(sig, 1e-300) / max(spur, 1e-300)))


def thermal_drift_budget_K(n_bits, dGdT_dB_per_K):
    """Max junction-temperature drift [K] for which a STATIC predistortion calibration stays
    good to half an LSB at n_bits: the fractional gain drift (1/G) dG/dT = (ln10/10) dG/dT[dB]
    must stay below 2^-(n_bits+1), so dT_max = 2^-(n_bits+1) / ((ln10/10) |dG/dT[dB/K]|).
    Tighter (smaller) as resolution grows -- why self-heating gates predistortion ENOB."""
    s = (np.log(10.0) / 10.0) * abs(float(dGdT_dB_per_K))    # fractional gain sensitivity /K
    if s <= 0.0:
        return float("inf")
    return float(2.0 ** (-(n_bits + 1)) / s)


def facet_gain_ripple_dB(G, R1, R2=None):
    """Fabry-Perot residual-facet gain ripple [dB] for a single-pass POWER gain G and facet power
    reflectivities R1, R2 (R2 defaults to R1). A real (not perfectly AR-coated) SOA has residual
    facet reflectivity; the round-trip cavity makes the transmitted gain oscillate with the
    round-trip phase between G_max ~ G/(1 - G sqrt(R1 R2))^2 and G_min ~ G/(1 + G sqrt(R1 R2))^2,
    so the peak-to-valley ripple is
        ripple_dB = 20 log10[(1 + G sqrt(R1 R2)) / (1 - G sqrt(R1 R2))]   (Saitoh & Mukai 1990).
    R = 0 -> 0 ripple (the ideal traveling-wave limit the single-pass model assumes); G sqrt(R1 R2)
    -> 1 is the lasing threshold (ripple -> inf) and raises. This gain-flatness budget caps the
    analog ENOB of the amplifier (see ripple_enob_ceiling)."""
    G = float(G)
    R1 = float(R1)
    R2 = float(R1 if R2 is None else R2)
    if not (G > 0.0 and R1 >= 0.0 and R2 >= 0.0):
        raise ValueError("facet_gain_ripple_dB: need G > 0 and R1, R2 >= 0")
    gr = G * np.sqrt(R1 * R2)
    if gr >= 1.0:
        raise ValueError("facet_gain_ripple_dB: G sqrt(R1 R2) = {:.4f} >= 1 -- at/above the lasing "
                         "threshold; the device is a Fabry-Perot laser, not an amplifier".format(gr))
    return float(20.0 * np.log10((1.0 + gr) / (1.0 - gr)))


def ripple_enob_ceiling(ripple_dB):
    """ENOB ceiling imposed by a peak-to-valley gain ripple [dB]: the fractional POWER spread
    m = 10^(ripple_dB/10) - 1 is an irreducible full-scale gain error on the intensity-encoded
    photocurrent channel this subpackage models (direct detection, i ~ R P_out), so the
    resolution cannot exceed n = -log2(m) bits (m must fall below one LSB = 2^-n). Returns +inf
    for zero ripple. E.g. 0.17 dB -> ~4.6 bits, 1.7 dB -> ~1.2 bits (Saitoh & Mukai facet-ripple
    regime). Audit S3-5: the old /20 (field-amplitude) conversion overstated the ceiling by ~1
    bit; facet_gain_ripple_dB returns a POWER-ratio dB (20 log10[(1+gr)/(1-gr)] = 10 log10 of
    the peak/valley power-gain ratio), so the intensity channel takes the /10 form."""
    m = 10.0 ** (abs(float(ripple_dB)) / 10.0) - 1.0
    if m <= 0.0:
        return float("inf")
    return float(-np.log2(m))
