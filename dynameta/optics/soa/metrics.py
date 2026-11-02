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

__all__ = ["transfer_derivatives", "transfer_derivatives_fd", "TransferFitWarning",
           "harmonic_amplitudes", "sndr_db", "enob",
           "sndr_vs_drive", "optimal_drive_power", "predistort", "pattern_penalty_dB",
           "sfdr_dB", "thermal_drift_budget_K", "facet_gain_ripple_dB", "ripple_enob_ceiling"]


class TransferFitWarning(RuntimeWarning):
    """The sampled transfer-curve derivatives are not resolved by the grid (audit A-23)."""


def _cubic_derivs(P_in, P_out, P0, k):
    """(T', T'', T''') from a cubic fit to the k samples nearest P0 (P_in sorted)."""
    h = k // 2
    i = int(np.clip(np.searchsorted(P_in, P0), h, P_in.size - h - 1))
    sl = slice(i - h, i + h + 1)
    c = np.polyfit(P_in[sl] - P0, P_out[sl], 3)               # c[0] x^3 + c[1] x^2 + c[2] x + c[3]
    return float(c[2]), float(2.0 * c[1]), float(6.0 * c[0])  # T', T'', T'''


def _sampled_derivs(P_in_grid, P_out_grid, P0):
    """(T', T'', T''', err): the 7-sample cubic (the historical estimator, bit-identical) plus
    its accuracy indicator err = max over T'' and T''' of |fit9 - fit7| / |fit7| -- how much the
    curvature moves when the stencil widens to 9 samples (nan with fewer than 9 samples)."""
    P_in = np.asarray(P_in_grid, dtype=np.float64)
    P_out = np.asarray(P_out_grid, dtype=np.float64)
    order = np.argsort(P_in)
    P_in, P_out = P_in[order], P_out[order]
    d7 = _cubic_derivs(P_in, P_out, P0, 7)
    err = float("nan")
    if P_in.size >= 9:
        d9 = _cubic_derivs(P_in, P_out, P0, 9)
        err = max(abs(d9[j] - d7[j]) / max(abs(d7[j]), 1e-300) for j in (1, 2))
    return d7 + (err,)


def _warn_unresolved(where, err, P0, warn_rtol, fd=False):
    if warn_rtol is not None and np.isfinite(err) and err > warn_rtol:
        import warnings
        if fd:
            warnings.warn(
                "{}: the finite-difference curvature of transfer= at P0 = {:.4g} W is uncertain by "
                "~{:.1%} (the h and h/2 stencils disagree): the callable is too noisy for rel_step, or "
                "the curve varies on a scale below rel_step * P0 -- adjust rel_step (audit A-23)."
                .format(where, float(P0), err), TransferFitWarning, stacklevel=3)
            return
        warnings.warn(
            "{}: the transfer-curve curvature at P0 = {:.4g} W is not resolved by the sampled grid "
            "(T''/T''' move by {:.1%} when the fit stencil widens from 7 to 9 samples; the true error "
            "is typically 1-3x that). Harmonic distortion / SNDR / SFDR inherit it. Sample the curve "
            "more densely around P0, or pass transfer= (a callable P_out(P_in)) for finite-difference "
            "derivatives (audit A-23).".format(where, float(P0), err),
            TransferFitWarning, stacklevel=3)


def transfer_derivatives(P_in_grid, P_out_grid, P0, *, return_error=False, warn_rtol=0.05):
    """Local (T', T'', T''') of the transfer curve P_out(P_in) at P0 via a cubic fit to the
    nearest samples -- the curvature that produces harmonic distortion.

    ACCURACY (audit A-23, measured 2026-11-02). The 7 nearest samples of a LOG-spaced sweep span a
    wide power range (5.6x on 41 points over 5 decades), so the fit is not local: on a
    homogeneously saturating T = G0 P/(1 + P/Psat), T''' comes out 11% / 35% / 47% low at
    P0 = 0.1 / 1 / 3 Psat (SFDR 0.02 / 0.50 / 1.69 dB optimistic at 0.3 modulation); 3% / 11% /
    25% on 81 points. No fixed 7-sample formula is uniformly better, so the estimator is unchanged
    and instead REPORTS itself: err = the relative change of T''/T''' when the stencil widens to 9
    samples (on those cases 10% / 30% / 34% and 2.5% / 9% / 10%; on a tanh transfer within ~1-3x
    of the true error -- up to
    3.4x BELOW it, at most 1.2x above), insensitive to 1e-6 solver noise. A
    TransferFitWarning is emitted when err > warn_rtol (None disables). return_error=True returns
    (T', T'', T''', err); err is nan with fewer than 9 samples. For accurate derivatives pass a
    callable to transfer_derivatives_fd (or transfer= to sndr_vs_drive / sfdr_dB)."""
    Tp, Tpp, Tppp, err = _sampled_derivs(P_in_grid, P_out_grid, P0)
    _warn_unresolved("transfer_derivatives", err, P0, warn_rtol)
    return (Tp, Tpp, Tppp, err) if return_error else (Tp, Tpp, Tppp)


def transfer_derivatives_fd(transfer, P0, *, rel_step=0.15, return_error=False):
    """(T', T'', T''') of a CALLABLE transfer curve P_out = transfer(P_in) at P0 by central
    differences with one Richardson step (audit A-23): every stencil is O(h^2) at step h and h/2,
    extrapolated to O(h^4). err combines the two error sources, max over the three derivatives:
        truncation  |D(h/2) - D(h)| / 3                (Richardson's error of the h/2 stencil --
                                                        conservative for the extrapolated value)
        noise       sigma * ||w||                      (sigma = rms residual of a degree-5 fit to
                                                        the 7 samples, w = the result's exact linear
                                                        weights on them)
    combined in quadrature and divided by min(|result|, |D(h)|, |D(h/2)|), so noise that blows the
    result up cannot shrink its own relative error (normalizing by the result alone hid a 16x-wrong
    T''' behind 0.23). Measured on G0 P/(1+P/Psat): clean err 2e-4 / 7e-3 / 2e-2 at P0 = 0.1 / 1 /
    3 Psat (true 2e-7 / 2e-4 / 9e-4; tanh at 3 Psat 5e-2 vs 3e-3); with 1e-6 / 1e-5 / 1e-4 relative
    noise median err 9e-3 / 5.5e-2 / 6.8e-1 vs true 4e-3 / 4e-2 / 4e-1 (err >= true/2 in 97 / 79 /
    85% of 120 draws). h = rel_step * P0; the stencil reaches P0 +- 2h
    (rel_step must be < 0.5 so it stays at positive power). Seven calls to `transfer`, one scalar
    at a time (a solver wrapper need not be vectorized). The default step balances the two error
    sources, measured 2026-11-02 on G0 P/(1 + P/Psat) and G0 Psat tanh(P/Psat), P0 = 0.1 / 1 /
    3 Psat:
        rel_step  truncation in T'''        1e-6 relative noise on the calls (median / worst of 60)
        0.05      <= 1e-5                    1.1e-1 / 4.8e-1
        0.15      <= 3e-3 (tanh at 3 Psat)   4.3e-3 / 1.8e-2
        0.30      <= 4e-2                    3.1e-3 / 4.6e-3
    (noise enters as eps / rel_step^3, truncation as rel_step^4). Check err; raise rel_step for a
    noisier solver."""
    P0 = float(P0)
    rel_step = float(rel_step)
    if not (P0 > 0.0 and 0.0 < rel_step < 0.5):
        raise ValueError("transfer_derivatives_fd: need P0 > 0 and 0 < rel_step < 0.5 "
                         "(got P0={!r}, rel_step={!r})".format(P0, rel_step))
    h = rel_step * P0
    ks = (-4, -2, -1, 0, 1, 2, 4)                     # sample offsets in units of h/2
    y = np.array([float(transfer(P0 + k * h / 2.0)) for k in ks])

    def stencils(vals, m):                 # m = 2 -> step h, m = 1 -> step h/2
        f = dict(zip(ks, vals))
        s = m * h / 2.0
        d1 = (f[m] - f[-m]) / (2.0 * s)
        d2 = (f[m] - 2.0 * f[0] + f[-m]) / s ** 2
        d3 = (f[2 * m] - 2.0 * f[m] + 2.0 * f[-m] - f[-2 * m]) / (2.0 * s ** 3)
        return np.array([d1, d2, d3])

    def richardson(vals):
        fine, coarse = stencils(vals, 1), stencils(vals, 2)
        return fine + (fine - coarse) / 3.0, fine, coarse
    rich, fine, coarse = richardson(y)
    # noise: rms residual of a degree-5 fit (1 dof) propagated through the result's linear weights
    t = np.asarray(ks, dtype=np.float64) / 4.0
    V = np.vander(t, 6)
    resid = y - V @ np.linalg.lstsq(V, y, rcond=None)[0]
    sigma = float(np.sqrt(np.sum(resid ** 2)))
    w = np.array([richardson(row)[0] for row in np.eye(len(ks))])      # (7, 3) linear weights
    noise = sigma * np.sqrt(np.sum(w ** 2, axis=0))
    trunc = np.abs(fine - coarse) / 3.0
    scale = np.minimum(np.abs(rich), np.minimum(np.abs(fine), np.abs(coarse)))
    err = float(np.max(np.hypot(trunc, noise) / np.maximum(scale, 1e-300)))
    out = (float(rich[0]), float(rich[1]), float(rich[2]))
    return out + (err,) if return_error else out


def _derivs_and_level(P_in_grid, P_out_grid, P0, transfer, rel_step):
    """(T', T'', T''', err, P_out(P0)) from the callable when given, else from the samples."""
    if transfer is not None:
        Tp, Tpp, Tppp, err = transfer_derivatives_fd(transfer, P0, rel_step=rel_step,
                                                     return_error=True)
        return Tp, Tpp, Tppp, err, float(transfer(float(P0)))
    Tp, Tpp, Tppp, err = _sampled_derivs(P_in_grid, P_out_grid, P0)
    P_out0 = float(np.interp(P0, np.sort(P_in_grid),
                             np.asarray(P_out_grid)[np.argsort(P_in_grid)]))
    return Tp, Tpp, Tppp, err, P_out0


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
                  R_A_W=1.0, transfer=None, rel_step=0.15, warn_rtol=0.05):
    """SNDR [dB] vs bias drive power P0 for a sinusoidally intensity-modulated signal.
    `noise_var_of_Pout(P_out)` returns the detector noise variance [A^2] at the detected mean
    output power. Returns (sndr_dB[P0], enob[P0], idx_opt) -- the interior maximum is the
    optimal analog operating point (noise-limited below, distortion-limited above).

    The curvature comes from the sampled (P_in_grid, P_out_grid) via transfer_derivatives (one
    TransferFitWarning per call, naming the worst P0, when the grid does not resolve it -- audit
    A-23), or, with transfer= (a callable P_out(P_in)), from transfer_derivatives_fd at
    rel_step; the grids are then unused and may be None."""
    P0s = np.atleast_1d(np.asarray(P0_array, dtype=np.float64))
    sndr = np.empty(P0s.size)
    worst = (-1.0, None)
    for k, P0 in enumerate(P0s):
        Tp, Tpp, Tppp, err, P_out0 = _derivs_and_level(P_in_grid, P_out_grid, float(P0),
                                                       transfer, rel_step)
        if np.isfinite(err) and err > worst[0]:
            worst = (err, float(P0))
        A = P0 * mod_index
        c1, c2, c3 = harmonic_amplitudes(Tp, Tpp, Tppp, A)
        sig = 0.5 * (R_A_W * c1) ** 2
        dist = 0.5 * (R_A_W * c2) ** 2 + 0.5 * (R_A_W * c3) ** 2
        noise = float(noise_var_of_Pout(P_out0))
        sndr[k] = sndr_db(sig, noise, dist)
    if worst[1] is not None:
        _warn_unresolved("sndr_vs_drive", worst[0], worst[1], warn_rtol, fd=transfer is not None)
    return sndr, np.array([enob(s) for s in sndr]), int(np.argmax(sndr))


def optimal_drive_power(P_in_grid, P_out_grid, noise_var_of_Pout, P0_array, *,
                        mod_index=0.3, R_A_W=1.0, transfer=None, rel_step=0.15, warn_rtol=0.05):
    """The drive power that maximizes SNDR (the analog operating point) and its (SNDR, ENOB).
    The interior optimum trades the ASE-noise floor (dominant at low drive) against gain-
    compression distortion (dominant at high drive). transfer= / rel_step / warn_rtol as in
    sndr_vs_drive."""
    sndr, eno, iopt = sndr_vs_drive(P_in_grid, P_out_grid, noise_var_of_Pout, P0_array,
                                    mod_index=mod_index, R_A_W=R_A_W, transfer=transfer,
                                    rel_step=rel_step, warn_rtol=warn_rtol)
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


def sfdr_dB(P_in_grid, P_out_grid, P0, noise_var, *, mod_index=0.3, R_A_W=1.0, transfer=None,
            rel_step=0.15, warn_rtol=0.05):
    """Spurious-free dynamic range [dB] at drive P0: the ratio (in dB) of the fundamental to
    the larger of the 2nd/3rd compression harmonics, floored at the noise level -- the usable
    distortion-free window above the noise. transfer= / rel_step / warn_rtol as in
    sndr_vs_drive (audit A-23: the sampled path can be ~1.7 dB optimistic near saturation on a
    41-point log grid)."""
    Tp, Tpp, Tppp, err, _ = _derivs_and_level(P_in_grid, P_out_grid, float(P0), transfer, rel_step)
    _warn_unresolved("sfdr_dB", err, P0, warn_rtol, fd=transfer is not None)
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
