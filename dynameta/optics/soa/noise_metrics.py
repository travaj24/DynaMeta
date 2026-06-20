"""QD-SOA noise OBSERVABLES: the relative-intensity-noise spectrum RIN(f), the field-autocorrelation
linewidth, and the Schawlow-Townes-Henry (1+alpha^2) linewidth formula. These are post-processing
READOUTS on a time-domain trace (e.g. the Langevin output of TravelingWaveSOA.amplify_coherent(
langevin=True)) -- they add NO physics to the marcher; they turn its stochastic output into the
standard analog-link noise figures. Pure numpy; SI; ASCII.
"""
from __future__ import annotations

import numpy as np


def rin_spectrum(P_t, dt_s):
    """One-sided relative-intensity-noise PSD RIN(f) [1/Hz] of a power trace P(t) sampled at dt:

        RIN(f) = S_dP(f) / <P>^2,   dP = P - <P>,

    with S_dP the one-sided periodogram of the power fluctuation. Parseval: integral_0^fNyq RIN df =
    var(P)/<P>^2 (the total fractional intensity-noise variance). Returns (f_Hz, rin_per_Hz). A
    sinusoidal intensity modulation of depth m shows a line at its frequency whose integral is m^2/2."""
    P = np.asarray(P_t, dtype=np.float64)
    N = P.size
    if N < 4:
        raise ValueError("rin_spectrum: need >= 4 samples")
    Pbar = float(P.mean())
    if Pbar <= 0.0:
        raise ValueError("rin_spectrum: mean power must be > 0")
    X = np.fft.rfft(P - Pbar)
    psd = (float(dt_s) / N) * np.abs(X) ** 2          # two-sided periodogram on the rfft bins
    psd[1:] *= 2.0                                     # one-sided: double every bin except DC
    if N % 2 == 0:
        psd[-1] /= 2.0                                 # the Nyquist bin is not doubled (even N)
    f = np.fft.rfftfreq(N, d=float(dt_s))
    return f, psd / (Pbar * Pbar)


def linewidth_from_field(A_t, dt_s, *, n_fit: int = 20):
    """Lorentzian FWHM linewidth Delta_nu [Hz] from the first-order field autocorrelation g1(tau) =
    <A*(t) A(t+tau)> / <|A|^2>. For a phase-diffusing (Lorentzian) line |g1(tau)| = exp(-pi Delta_nu
    |tau|), so Delta_nu = -slope/pi of ln|g1| over the first n_fit lags. A pure tone gives Delta_nu ->
    0; a Wiener phase walk with per-step variance v over dt gives Delta_nu = v/(2 pi dt). Returns
    Delta_nu [Hz]. (g1 via the Wiener-Khinchin FFT autocorrelation, unbiased by the overlap count.)"""
    A = np.asarray(A_t, dtype=np.complex128)
    N = A.size
    if N < n_fit + 2:
        raise ValueError("linewidth_from_field: need > n_fit+1 samples")
    F = np.fft.fft(A, 2 * N)
    ac = np.fft.ifft(np.abs(F) ** 2)[:N] / np.arange(N, 0, -1)   # unbiased autocorrelation
    if np.abs(ac[0]) == 0.0:
        raise ValueError("linewidth_from_field: zero-power field (cannot normalize g1)")
    g1 = np.abs(ac) / np.abs(ac[0])
    lags = np.arange(1, n_fit + 1)
    g1f = np.clip(g1[1:n_fit + 1], 1e-300, None)
    slope = np.polyfit(lags * float(dt_s), np.log(g1f), 1)[0]    # = -pi Delta_nu
    return max(-slope / np.pi, 0.0)


def henry_factor(alpha: float) -> float:
    """The Henry linewidth-enhancement factor 1 + alpha^2 (amplitude-phase coupling: the linewidth is
    that many times the Schawlow-Townes value)."""
    return 1.0 + float(alpha) ** 2


def schawlow_townes_henry_linewidth(R_sp_per_s, n_photons, alpha):
    """Schawlow-Townes-Henry laser linewidth [Hz]:

        Delta_nu = (R_sp / (4 pi N_ph)) (1 + alpha^2),

    R_sp = spontaneous-emission rate into the lasing mode [1/s], N_ph = intracavity photon number. The
    (1+alpha^2) is the Henry amplitude-phase coupling enhancement over the bare Schawlow-Townes value
    R_sp/(4 pi N_ph). This is the LASER-CAVITY (clamped-gain) linewidth REFERENCE -- a single-pass SOA
    does not lase, so this is the closed-form oracle for the (1+alpha^2) factor, not a property of the
    amplifier marcher (whose Langevin output captures the amplitude-phase coupling DIRECTION via
    linewidth_from_field, but not the gain-clamped cavity narrowing)."""
    if n_photons <= 0.0:
        raise ValueError("schawlow_townes_henry_linewidth: n_photons must be > 0")
    return (float(R_sp_per_s) / (4.0 * np.pi * float(n_photons))) * (1.0 + float(alpha) ** 2)
