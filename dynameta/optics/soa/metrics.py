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
           "sndr_vs_drive"]


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
