"""
Ringdown harmonic inversion (roadmap 1.2): extract resonance poles (omega_0, gamma, Q)
from a time trace of DAMPED EXPONENTIALS -- e.g. an FDTD field after a pulse has passed --
by the matrix-pencil method (Hua & Sarkar, IEEE Trans. Antennas Propag. 38:814 (1990)).

Why not just FFT: the FFT resolves two modes only if they are separated by more than the
inverse record length (1 / (N dt)); a matrix pencil resolves damped complex exponentials far
below that limit (it fits poles, not bins), so a short ringdown that shows a SINGLE FFT peak
can still be split into its constituent modes. This is the geometry-general pole route: it
works on anything the FDTD (or any solver) can march.

Method (SVD-truncated / total-least-squares matrix pencil): from the uniformly sampled trace
y[n] = sum_k R_k z_k^n, z_k = exp(s_k dt), build the Hankel data matrix, take its SVD, keep the
signal subspace (model-order selection off the singular-value gap), and read the z_k off a
small generalized eigenproblem in the truncated right-singular vectors. Residues R_k follow
from a Vandermonde least squares. s_k = ln(z_k)/dt.

CONVENTIONS (SI, exp(-i omega t) library convention -- decaying modes have Im(omega_t) < 0):
  * A physical field trace is REAL. A real signal is fit with complex poles that come out in
    exact conjugate pairs; each physical mode is reported ONCE with omega_rad_s > 0.
  * A pole s = -gamma/2 - i omega_0 corresponds to the complex resonance frequency
    omega_t = i s = omega_0 - i gamma/2 (Re omega_t = omega_0 = -Im s, Im omega_t = -gamma/2).
  * AMPLITUDE-vs-ENERGY DECAY CONVENTION (documented and load-bearing):
        the FIELD amplitude decays as exp(-gamma t / 2),
        so the ENERGY (|field|^2) decays as exp(-gamma t).
    Hence gamma_rad_s is the ENERGY decay rate; the energy 1/e time is tau_E = 1/gamma and the
    energy half-life is t_1/2 = ln 2 / gamma.
  * QUALITY FACTOR:  q = omega_0 / gamma  (an ENERGY Q). This is identical to the pole-finder
    convention Q = Re(omega_t) / (2 |Im(omega_t)|) = omega_0 / (2 * gamma/2) = omega_0 / gamma
    for the SAME resonance, so ringdown Q and complex-omega-pole Q agree by construction.
  * The reported complex `amplitude` reconstructs the trace as
        REAL input:     y(t) ~ sum_modes Re( amplitude * exp(-i omega_t t) )
        COMPLEX input:  y(t) ~ sum_modes     amplitude * exp(-i omega_t t)
    so for a real signal a pure cosine A cos(omega t + phi) exp(-gamma t/2) returns
    amplitude = A exp(-i phi) (|amplitude| = A is the t=0 peak amplitude).

Opt-in FDTD probe: solve_fdtd_1d(..., return_time_trace=True) attaches the recorded reflected/
transmitted boundary time series (already computed for the R/T DFT) without changing any
existing output; fdtd_etalon_ringdown() drives it and inverts the post-pulse tail.

ONE ringdown Q (audit Q-22): ringdown_q and fdtd_etalon_ringdown extract modes through the same
_pencil_modes path (matrix pencil + the real-trace VARPRO refinement + dominant-by-|amplitude|),
so the same samples give the same (f0, Q) from both; matrix_pencil alone is the raw pencil.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np

from dynameta.constants import C_LIGHT
# audit V-17 leftover (wave-7, 2026-11-02): the four unreliable-result warns below adopt
# UnreliableResultWarning, still a RuntimeWarning subclass, so pytest.warns(RuntimeWarning) holds.
from dynameta.warning_types import UnreliableResultWarning

__all__ = [
    "Mode",
    "Drift",
    "matrix_pencil",
    "ringdown_q",
    "EtalonRingdown",
    "fdtd_etalon_ringdown",
]


@dataclass
class Mode:
    """One damped-exponential mode extracted from a ringdown trace.

    Fields (SI, exp(-i omega t) convention):
      omega_rad_s : resonance angular frequency omega_0 > 0 [rad/s]
      gamma_rad_s : ENERGY decay rate gamma > 0 [rad/s]; field ~ exp(-gamma t/2),
                    energy ~ exp(-gamma t)
      q           : quality factor omega_0 / gamma (energy Q; matches the pole-finder's
                    Re(omega_t)/(2|Im omega_t|))
      amplitude   : complex phasor amplitude (see module docstring reconstruction convention)
      snr_est     : |amplitude| / RMS(fit residual); np.inf for a perfect (noise-free) fit
    """

    omega_rad_s: float
    gamma_rad_s: float
    q: float
    amplitude: complex
    snr_est: float

    @property
    def f_hz(self) -> float:
        """Ordinary frequency omega_0 / (2 pi) [Hz]."""
        return self.omega_rad_s / (2.0 * np.pi)


@dataclass
class Drift:
    """A NON-RESONANT slow term of a fitted trace (2026-11-03): a pencil pole whose frequency is
    BELOW THE WINDOW'S FREQUENCY RESOLUTION, |omega| < omega_res = 2 pi / (N dt) with N the number
    of fitted samples -- it completes less than one cycle in the window, so it is a drift / offset,
    not a resonance. It is kept in every least-squares fit (it must absorb the slow component) but
    never reported as a Mode, never enters (f0, Q), and is exempt from the growing-pole warning: a
    slowly GROWING drift (e.g. a first-order Mur wall's unabsorbed static residue in an FDTD trace)
    is not an amplifying resonance.

    Fields: gamma_rad_s -- the rate in the Mode convention, term ~ exp(-gamma t / 2) (NEGATIVE =
    growing); omega_rad_s -- its |omega| (< threshold_rad_s; 0 for a real pole); amplitude -- the
    complex phasor amplitude (the real part for omega = 0); threshold_rad_s -- omega_res of the fit."""

    gamma_rad_s: float
    omega_rad_s: float
    amplitude: complex
    threshold_rad_s: float


def _drift_reconstruction(n: int, dt: float, drift: Sequence["Drift"]) -> np.ndarray:
    """Real reconstruction of drift terms, SIGNED rate (a growing drift grows)."""
    t = np.arange(int(n)) * float(dt)
    out = np.zeros(int(n), dtype=np.float64)
    for d in drift:
        e = np.exp(-0.5 * d.gamma_rad_s * t)
        if d.omega_rad_s * dt > _OMEGA_DC_TOL:
            out += e * (d.amplitude.real * np.cos(d.omega_rad_s * t) + d.amplitude.imag * np.sin(d.omega_rad_s * t))
        else:
            out += e * d.amplitude.real
    return out


# |z| in (1, 1 + _GAIN_TOL] is the numerical image of a LOSSLESS pole (gamma = 0), not gain:
# snapped onto the unit circle. Above it the pole is genuinely growing -- see matrix_pencil's
# DISCRETE-POLE DOMAIN note (finding Q-9).
_GAIN_TOL = 1e-9


def _hankel(y: np.ndarray, L: int) -> np.ndarray:
    """(N-L) x (L+1) Hankel data matrix H[i, j] = y[i + j] (the matrix-pencil data matrix).

    Returned as a READ-ONLY STRIDED VIEW of ``y`` (audit P-2).  The previous form built an
    ``(N-L) x (L+1)`` int64 index array purely to fancy-index ``y``, i.e. an integer array
    exactly as large as the result, discarded immediately: measured peak traced memory 1536 MB
    at N = 20000 (result 768 MB) plus 0.35 s of pure copy time, versus 0.001 MB / 20 us for the
    view.  The only consumer is :func:`numpy.linalg.svd`, which makes its own contiguous LAPACK
    copy either way, so the values reaching LAPACK -- and therefore every number downstream --
    are bit-for-bit what they were.  The view is NON-WRITEABLE and non-contiguous: any future
    caller that wants to mutate ``H`` (or needs C order) must ``.copy()`` it first.
    """
    rows = y.size - L
    return np.lib.stride_tricks.sliding_window_view(y, L + 1)[:rows]


# --- big-N guard on the public pencil entry points (audit P-3) --------------------------------
# `matrix_pencil` is O(N^3) in time and O(N^2) in memory in the RAW trace length: with the
# default pencil_frac = 0.4 the Hankel matrix is 0.6N x 0.4N and `np.linalg.svd` allocates its
# own copy of it plus U (0.6N x 0.4N, computed by LAPACK and never read here) and Vh
# (0.4N x 0.4N).  Measured: N = 8000 -> 13.8 s and ~370 MB; N = 20000 -> ~4 min and ~2.3 GB.
# The repo's own front end `fdtd_etalon_ringdown` decimates and keeps its fit under
# _FIT_SAMPLES_BUDGET = 6000 samples (coarsening the stride first), but `matrix_pencil` /
# `ringdown_q` are exported for arbitrary traces and
# `solve_fdtd_1d(..., return_time_trace=True)` hands out traces of exactly the length that blows
# this up.  So the workspace estimate below is checked BEFORE anything large is allocated and a
# too-long trace RAISES (it never silently decimates -- decimation is an accuracy change and must
# be the caller's explicit choice, as in `fdtd_etalon_ringdown`).
_PENCIL_MAX_BYTES = 2_000_000_000                       # 2 GB SVD workspace budget


def _pencil_peak_bytes(rows: int, cols: int, itemsize: int) -> float:
    """Estimated peak bytes of the pencil SVD step: the contiguous LAPACK copy of the Hankel
    data, U (rows x k) and Vh (k x cols) with k = min(rows, cols).  The strided Hankel view
    itself costs nothing (P-2), so this counts ONE copy of the matrix, not two.  With
    pencil_frac = 0.4 it is ~0.64 * itemsize * N^2, so the 2 GB budget is reached at
    N ~ 19760 samples (real trace) / N ~ 13970 (complex): the audit's measured N = 20000 case
    (~4 min of SVD, 2.3 GB peak before P-2) raises, while its N = 8000 case (13.8 s, ~370 MB)
    still runs."""
    k = min(int(rows), int(cols))
    return float(itemsize) * (float(rows) * cols + float(rows) * k + float(k) * cols)


def _select_order(sv: np.ndarray, svd_tol: float, max_modes: Optional[int]) -> int:
    """Model order from the singular-value spectrum: the largest LOG gap (relative drop)
    between consecutive singular values, searched only over values above the absolute floor
    `svd_tol` (relative to sv[0]) and below `max_modes`. Robust to both the noise-free case
    (a machine-eps cliff after the true order) and a noise floor (the signal->noise gap)."""
    P = sv.size
    if P == 0:
        return 0
    norm = sv / sv[0]
    n_above = int(np.count_nonzero(norm >= svd_tol))
    n_above = max(1, n_above)
    cap = n_above if max_modes is None else min(n_above, int(max_modes))
    cap = max(1, min(cap, P - 1))          # need sv[cap] to exist to see the gap after `cap`
    logs = np.log(norm[:cap + 1] + 1e-300)
    drops = logs[:-1] - logs[1:]           # drops[i] = log(norm[i]) - log(norm[i+1]) >= 0
    if drops.size == 0:
        return 1
    return int(np.argmax(drops)) + 1


def matrix_pencil(signal: Sequence[complex], dt: float, *, pencil_frac: float = 0.4,
                  svd_tol: float = 1e-9, max_modes: Optional[int] = None,
                  t_start: float = 0.0, amp_floor: float = 1e-3,
                  real_signal: Optional[bool] = None,
                  omega_zero_tol: float = 1e-6,
                  allow_gain: bool = False,
                  max_bytes: Optional[float] = None,
                  drift_out: Optional[list] = None) -> List[Mode]:
    """Extract damped-exponential modes from a uniformly sampled trace by the matrix-pencil
    method (Hua & Sarkar 1990).

    Args:
      signal      : uniformly sampled trace y[n] (real physical field, or complex).
      dt          : sample spacing [s].
      pencil_frac : pencil parameter L = round(pencil_frac * N_fit); the paper's efficient/
                    low-variance window is L in [N/3, N/2] (default 0.4).
      svd_tol     : singular values below svd_tol * sv[0] are treated as the noise/eps floor
                    and never enter the signal subspace.
      max_modes   : optional hard cap on the number of poles taken from the SVD.
      t_start     : skip the driven transient -- fit only samples with t >= t_start (t = n dt).
      amp_floor   : residual prune -- drop modes whose |amplitude| < amp_floor * max|amplitude|
                    (relative amplitude floor; removes noise-driven spurious modes).
      real_signal : force the real (conjugate-pair) collapse on/off; None auto-detects from the
                    input dtype / imaginary content.
      omega_zero_tol : |omega dt| below this (radians) is treated as a zero-frequency (pure
                    decay) mode rather than half of an oscillatory pair.
      allow_gain  : keep GROWING poles (|z| > 1, i.e. gamma < 0 and q < 0). Default False --
                    the `Mode` contract documents gamma > 0, and a growing pole additionally
                    OVERFLOWS the residue Vandermonde at large N (finding Q-9). See the
                    DISCRETE-POLE DOMAIN note below.
      drift_out   : optional list that RECEIVES the drift terms (Drift objects): poles with
                    |omega| < omega_res = 2 pi / (N dt) (N = fitted samples), i.e. below the
                    window's frequency resolution. They stay in the residue fit, are exempt from
                    the growing-pole warning, and are NEVER returned as Modes (2026-11-03; see
                    Drift). Pass a list to see them; they are dropped from the output either way.
      max_bytes   : SVD workspace budget [bytes]; None uses `_PENCIL_MAX_BYTES` (2 GB). A trace
                    long enough to exceed it RAISES instead of allocating (audit P-3) -- see the
                    COST note below. `np.inf` disables the guard.

    Returns:
      list[Mode] sorted by descending |amplitude| (dominant first).

    DISCRETE-POLE DOMAIN (findings Q-8 / Q-9). The z-plane poles are mapped to continuous poles
    by ``s = log(z)/dt``, which needs the COMPLEX branch: ``np.linalg.eigvals`` returns a REAL
    array when every eigenvalue happens to be real, and ``np.log`` of a NEGATIVE real then gives
    ``nan`` -- a Nyquist-frequency pole (``z < 0``, ``omega = pi/dt``) used to escape the public
    API as ``omega_rad_s = 0`` with ``gamma_rad_s = nan``.  ``z`` is therefore cast to
    ``complex128`` before the log, and the self-conjugate Nyquist pole is reported ONCE (no
    residue doubling) rather than being dropped by the conjugate collapse.
    Growing poles (``|z| > 1``) are dropped with an ``UnreliableResultWarning`` (a
    ``RuntimeWarning`` subclass) unless ``allow_gain=True``:
    they contradict the documented ``gamma > 0``, and the residue Vandermonde ``z**n`` overflows
    to ``inf`` once ``|z|**(N-1)`` passes the float64 range (for ``|z| = 1.5`` that is ``N ~
    1750`` -- the audit note's ``1.5**1200`` is actually 2.04e211, finite; the threshold is
    higher than quoted but the mechanism is real), which poisons the residue least-squares and
    the amplitude sort for EVERY mode, not just the growing one.  On the ``allow_gain=True`` path
    the Vandermonde is therefore built in log space and column-normalized, so no ``N`` overflows.

    COST -- THIS SCALES WITH THE RAW TRACE LENGTH (audit P-3). The Hankel matrix is
    ``(1-pencil_frac)N x pencil_frac*N``, so the SVD is ``O(N^3)`` in time and ``O(N^2)`` in
    memory in the number of SAMPLES, independent of how many modes are wanted: measured 0.23 s at
    N = 2000, 13.8 s at N = 8000, and ~4 min / ~2.3 GB extrapolated at N = 20000.  A trace whose
    estimated workspace exceeds ``max_bytes`` (2 GB by default, i.e. about 19700 real samples)
    RAISES with the decimation recipe rather than allocating.  MORE SAMPLES ARE NOT MORE
    INFORMATION here: the pencil fits poles, so a trace decimated to ~8-10 samples per period of
    the highest mode of interest carries the same poles at a fraction of the cost -- which is
    exactly what :func:`fdtd_etalon_ringdown` does (it decimates to
    ``target_samples_per_period`` and, by default, fits the whole data-driven window under a
    6000-sample budget -- see ``_FIT_SAMPLES_BUDGET``).
    """
    y_full = np.asarray(signal)
    dt = float(dt)
    if dt <= 0.0:
        raise ValueError("dt must be > 0")
    # windowing: drop the driven transient before t_start
    n0 = int(np.ceil(t_start / dt)) if t_start > 0.0 else 0
    n0 = max(0, min(n0, y_full.size - 1))
    y = y_full[n0:]
    N = y.size
    if N < 4:
        raise ValueError("need at least 4 samples after t_start for a matrix-pencil fit")

    if real_signal is None:
        real_signal = not (np.iscomplexobj(y_full) and
                           np.max(np.abs(y.imag)) > 1e-12 * (np.max(np.abs(y)) + 1e-300))
    if real_signal:
        y = y.real.astype(np.float64)

    L = int(round(pencil_frac * N))
    L = max(2, min(L, N - 2))              # keep both Hankel dimensions >= 2

    # BIG-N GUARD (audit P-3): refuse a trace whose SVD workspace blows past `max_bytes` BEFORE
    # allocating anything large. The cost is O(N^3) time / O(N^2) memory in the RAW trace length,
    # and nothing upstream caps it: raise with the decimation recipe rather than spend four
    # minutes and a few GB. Never auto-decimates -- that would be a silent accuracy change.
    budget = float(_PENCIL_MAX_BYTES if max_bytes is None else max_bytes)
    need = _pencil_peak_bytes(N - L, L + 1, y.dtype.itemsize)
    if need > budget:
        n_fit = int((budget / max(need, 1.0)) ** 0.5 * N)
        raise ValueError(
            "matrix_pencil: a {}-sample trace needs an estimated {:.2f} GB of SVD workspace "
            "(Hankel {} x {}, {} bytes/sample), over the {:.2f} GB budget -- and the fit is "
            "O(N^3) in time (N = 8000 measures 13.8 s, N = 20000 extrapolates to ~4 min). "
            "DECIMATE the trace first (keep >= ~8 samples per period of the highest mode, e.g. "
            "`signal[::k]` with `dt*k`, which is what fdtd_etalon_ringdown does), shorten it "
            "with `t_start`, or -- if you really do have the memory -- pass `max_bytes` above "
            "{:.2f}e9. About {} samples fit the current budget."
            .format(N, need / 1e9, N - L, L + 1, y.dtype.itemsize, budget / 1e9,
                    budget / 1e9, n_fit))

    H = _hankel(y, L)
    # SVD of the Hankel data matrix; the right singular vectors carry the shift structure.
    U, sv, Vh = np.linalg.svd(H, full_matrices=False)
    V = Vh.conj().T                        # (L+1) x r

    M = _select_order(sv, svd_tol, max_modes)
    M = max(1, min(M, V.shape[1] - 1, L))  # need V1/V2 (one row removed) to keep >= M columns

    Vs = V[:, :M]                          # (L+1) x M signal subspace
    V1 = Vs[:-1, :]                        # remove last row  -> L x M
    V2 = Vs[1:, :]                         # remove first row -> L x M
    # z_k are the eigenvalues of the M x M matrix pinv(V1) @ V2 (TLS matrix pencil).
    Zmat = np.linalg.pinv(V1) @ V2
    # CAST TO COMPLEX (finding Q-8): np.linalg.eigvals returns a REAL array when every eigenvalue
    # of the real Zmat happens to be real, and np.log of a NEGATIVE real float64 is nan -- a pole
    # at exactly Nyquist (z < 0) then leaked out as omega = 0 with gamma = nan. The complex log
    # takes the correct branch ln|z| + i pi.
    z = np.asarray(np.linalg.eigvals(Zmat), dtype=np.complex128)

    # discard non-physical / numerically dead poles: |z| ~ 0 (infinite damping) is meaningless.
    z = z[np.abs(z) > 1e-12]
    if z.size == 0:
        return []
    # GROWING-POLE POLICY (finding Q-9). |z| > 1 is gamma < 0 and q < 0, which the Mode contract
    # forbids. Poles a hair outside the unit circle are the numerical image of a LOSSLESS mode:
    # snap those to |z| = 1 exactly (gamma = 0, q = inf) instead of reporting a large negative
    # gamma manufactured by a 1e-12 modulus excess. Genuinely growing poles are dropped with a
    # warning unless the caller opted in.
    absz = np.abs(z)
    marginal = (absz > 1.0) & (absz <= 1.0 + _GAIN_TOL)
    if np.any(marginal):
        z = np.where(marginal, z / np.where(absz > 0.0, absz, 1.0), z)
        absz = np.abs(z)
    # DRIFT CLASSIFICATION (2026-11-03): a pole with |omega| below the window's frequency
    # resolution omega_res = 2 pi / (N dt) completes < 1 cycle in the window -- non-resonant drift.
    # Such poles are exempt from the growing-pole policy (kept in the fit, reported via drift_out).
    omega_res = 2.0 * np.pi / (N * dt)
    is_drift = np.abs(np.log(z).imag) / dt < omega_res
    growing = (absz > 1.0 + _GAIN_TOL) & ~is_drift
    if np.any(growing) and not allow_gain:
        warnings.warn(
            "matrix_pencil: dropped {} GROWING pole(s) (|z| up to {:.6g} > 1, i.e. gamma < 0 and "
            "q < 0), which contradict the documented Mode contract (gamma > 0) and can overflow "
            "the residue Vandermonde on a long trace. Pass allow_gain=True to keep them (an "
            "amplifying medium / an unstable simulation), or shorten the fit window.".format(
                int(np.count_nonzero(growing)), float(absz[growing].max())),
            UnreliableResultWarning, stacklevel=2)
        z = z[~growing]
        absz = absz[~growing]
        is_drift = is_drift[~growing]
        if z.size == 0:
            return []
    s = np.log(z) / dt                     # continuous poles s_k = -gamma/2 -/+ i omega_0

    # residues by Vandermonde least squares: y[n] = sum_k R_k z_k^n
    n_idx = np.arange(N)
    if absz.max() <= 1.0 + _GAIN_TOL:
        # decaying (or unit-circle) poles: z**n cannot overflow -- (1+1e-9)**N is ~1 even at
        # N = 1e5 -- so keep the original power form and stay BYTE-IDENTICAL on every physical
        # ringdown, including one whose marginal pole was just snapped onto the unit circle.
        Zvand = z[None, :] ** n_idx[:, None]                 # N x M
        col_scale = np.ones(z.size, dtype=np.complex128)
    else:
        # allow_gain path only: build the Vandermonde in LOG space with each column divided by
        # its own largest magnitude, so |z| > 1 cannot reach inf (1.5**1200 = inf poisons the
        # lstsq and the |amplitude| sort for EVERY mode). Solving the scaled system and undoing
        # the scale afterwards recovers the same residues.
        logz = np.log(z)
        c = np.maximum(0.0, (N - 1) * np.real(logz))
        Zvand = np.exp(n_idx[:, None] * logz[None, :] - c[None, :])
        col_scale = np.exp(-c).astype(np.complex128)
    R_scaled, *_ = np.linalg.lstsq(Zvand, y, rcond=None)

    # full reconstruction -> residual RMS for the SNR estimate
    recon = Zvand @ R_scaled
    R = R_scaled * col_scale
    resid = y - (recon.real if real_signal else recon)
    res_rms = float(np.sqrt(np.mean(np.abs(resid) ** 2)))

    modes = []                                          # (omega, gamma, amp, is_drift)
    if real_signal:
        # Real signal: poles are real (Im s = 0) or exact conjugate pairs. Report each physical
        # mode once by taking the Im(s) < 0 representative (the exp(-i omega_0 t), i.e. positive
        # physical-frequency member) and doubling its residue; report a real pole (omega ~ 0) once.
        w_tol = omega_zero_tol / dt
        w_nyq = np.pi / dt
        for si, Ri, dr in zip(s, R, is_drift):
            omega = -si.imag
            gamma = -2.0 * si.real
            if abs(abs(si.imag) - w_nyq) <= w_tol:      # SELF-CONJUGATE Nyquist pole (finding Q-8)
                # z is real NEGATIVE: y[n] ~ Re(A) (-1)^n. It is its own conjugate partner, so it
                # is reported ONCE with an UNDOUBLED real residue -- the plain conjugate collapse
                # below would send it to the "already counted" branch and drop it silently.
                omega = w_nyq
                amp = complex(Ri.real, 0.0)
            elif si.imag < -w_tol:                      # positive-frequency representative
                amp = 2.0 * Ri
            elif abs(si.imag) <= w_tol:                 # zero-frequency (pure decay) mode
                omega = 0.0
                amp = complex(Ri.real, 0.0)
            else:                                       # conjugate partner already counted
                continue
            modes.append((omega, gamma, amp, bool(dr)))
    else:
        for si, Ri, dr in zip(s, R, is_drift):
            modes.append((-si.imag, -2.0 * si.real, Ri, bool(dr)))

    if not modes:
        return []

    amax = max(abs(m[2]) for m in modes)
    out: List[Mode] = []
    drift: List[Drift] = []
    for omega, gamma, amp, dr in modes:
        if dr:                                          # non-resonant drift: reported separately
            drift.append(Drift(gamma_rad_s=float(gamma), omega_rad_s=float(abs(omega)),
                               amplitude=complex(amp), threshold_rad_s=float(omega_res)))
            continue
        if amax > 0.0 and abs(amp) < amp_floor * amax:
            continue                                    # residual/amplitude prune
        q = float(omega / gamma) if gamma != 0.0 else np.inf
        snr = float(abs(amp) / res_rms) if res_rms > 0.0 else np.inf
        out.append(Mode(omega_rad_s=float(omega), gamma_rad_s=float(gamma), q=q,
                        amplitude=complex(amp), snr_est=snr))
    out.sort(key=lambda m: abs(m.amplitude), reverse=True)
    if drift_out is not None:
        drift.sort(key=lambda d: abs(d.amplitude), reverse=True)
        drift_out.extend(drift)
    return out


_OMEGA_DC_TOL = 1e-9          # |omega * dt| below this is a pure-decay (DC / drift) column


def _real_reconstruction(n: int, dt: float, modes: Sequence[Mode]) -> np.ndarray:
    """Reconstruct a REAL trace from a mode list using the module's documented convention
    y(t) ~ sum_k Re(A_k exp(-i omega_k t)) exp(-gamma_k t / 2)  (an omega ~ 0 mode is the pure
    decay Re(A) exp(-gamma t / 2)).  Used to score a refinement against its own seed."""
    t = np.arange(int(n)) * float(dt)
    out = np.zeros(int(n), dtype=np.float64)
    for m in modes:
        e = np.exp(-0.5 * abs(m.gamma_rad_s) * t)
        if m.omega_rad_s * dt > _OMEGA_DC_TOL:
            out += e * (m.amplitude.real * np.cos(m.omega_rad_s * t)
                        + m.amplitude.imag * np.sin(m.omega_rad_s * t))
        else:
            out += e * m.amplitude.real
    return out


# ---- ABSOLUTE fit-quality tells (fix-verify W1 kill 2) --------------------------------------
# _nls_refine_real's gate is RELATIVE: it only asks whether the refit beats its own pencil seed.
# When the window is source-contaminated BOTH fits are garbage and the comparison passes, so the
# public path returned a wrong Q with refine_note = '' and no warning at all. These two absolute
# tells separate the shipped configurations from the contaminated ones by ~an order of magnitude
# each (measured on the n_slab = 3.5..15 FDTD etalons vs a start_frac sweep that cuts into the
# driven pulse):
#   residual   rms(reconstruction - data) / ptp(data):  1.8e-3 .. 2.4e-3 good,  8.4e-3 .. 1.4e-1 bad
#   amplitude  sum_k |A_k| / max|data|:                 1.2 .. 2.1     good,  10 .. 5.7e3     bad
# The second catches near-cancelling amplitude blow-up (a rank-deficient design matrix reproduces
# the data from huge opposed columns); the first catches a fit that simply does not describe it.
_FIT_RESID_TOL = 5.0e-3
_FIT_AMP_SANITY = 20.0
# A window must span a few amplitude e-foldings for gamma (hence Q) to be observable rather than
# extrapolated; see fdtd_etalon_ringdown's SCOPE note.
_MIN_WINDOW_NEPERS = 3.0
# DEFAULT FIT LENGTH (2026-11-02, wave-6 H6b). fdtd_etalon_ringdown used to TRUNCATE the decimated
# window at a fixed max_fit_samples = 1200, which at n_slab = 20 fit only the first 0.42 true
# e-foldings of a 3176-sample window and returned Q = 252.8 against Fabry-Perot 439.5 (-44.5%).
# The default is now the WHOLE data-driven window. This budget only keeps the O(N^3) pencil in
# check on an unusually long record; measured on the n_slab = 20 trace (pencil + VARPRO, all
# modes): 1200 samples 1.1 s, 3176 1.3 s, 4730 1.9 s, 6352 4.5 s, 7940 9.6 s, and the shipped
# windows at 20 samples/period are 1845 / 2500 / 4838 / 4937 / 4891 / 3176 samples for
# n_slab = 3.5 / 5 / 7 / 10 / 15 / 20 -- all under it, so it never binds on a shipped config.
# Over budget the stride is COARSENED first (down to _MIN_SAMPLES_PER_PERIOD at the band centre),
# which keeps the decay span -- the thing Q needs -- instead of cutting it: on the same traces
# 8 samples/period over the whole window gave Q within 0.03-0.25 percentage points of 20
# samples/period (n_slab 3.5/7/10/15/20: -1.25 / +1.13 / +0.74 / +0.66 / +0.69% vs FP, against
# -1.01 / +1.09 / +0.70 / +0.63 / +0.63% at 20 samples/period). Only past that is the window
# truncated (and the span guard says so).
_FIT_SAMPLES_BUDGET = 6000
_MIN_SAMPLES_PER_PERIOD = 8
_LEGACY_FIT_SAMPLES = 1200                  # the start_frac (fixed-fraction window) path only


def _fit_quality(y: np.ndarray, dt: float, modes: Sequence[Mode], drift: Sequence[Drift] = ()) -> dict:
    """ABSOLUTE goodness-of-fit of ``modes`` (+ the fit's non-resonant ``drift`` terms, which are
    part of the model) on the windowed trace ``y``: the reconstruction RMS relative to the data's
    peak-to-peak, and the summed |amplitude| relative to the data's peak. Both are scale-free and
    neither is a comparison against another (possibly equally bad) fit."""
    y = np.asarray(y, dtype=np.float64)
    ptp = float(np.ptp(y))
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if modes or drift:
        rec = _real_reconstruction(y.size, dt, modes)
        if drift:
            rec = rec + _drift_reconstruction(y.size, dt, drift)
        rms = float(np.sqrt(np.mean((rec - y) ** 2)))
    else:
        rms = float(np.sqrt(np.mean(y ** 2)))
    amp = float(sum(abs(m.amplitude) for m in modes)) + float(sum(abs(d.amplitude) for d in drift))
    return {"rms": rms, "ptp": ptp, "peak": peak, "amp_sum": amp,
            "rms_rel": rms / ptp if ptp > 0.0 else float("inf"),
            "amp_rel": amp / peak if peak > 0.0 else float("inf")}


def _nls_refine_real(y: np.ndarray, dt: float, modes: List[Mode], *,
                     max_refine: int = 6, fixed_extra: Sequence[Drift] = (),
                     fixed_out: Optional[list] = None) -> tuple:
    """VARPRO refinement of pencil modes on a REAL trace: hold the mode COUNT fixed, optimize
    the nonlinear parameters (omega_k, gamma_k) by least squares with the linear (cos/sin)
    amplitudes solved exactly at each step. The pencil poles come out of an SVD subspace whose
    last digits are LAPACK-build-dependent -- near a marginal model order, two correct BLAS
    stacks can return dominant-mode Q values differing by tens of percent (observed: Windows
    dev box vs CI linux wheels straddling a 12% gate on the SAME deterministic FDTD trace).
    The NLS optimum is a property of the DATA, so refined modes are platform-stable.

    EVERY pencil mode contributes columns to the design matrix (finding Q-2).  Only the
    ``max_refine`` DOMINANT OSCILLATORY modes have their nonlinear parameters (omega, gamma)
    varied; every other mode keeps its seed (omega, gamma) and contributes FIXED columns -- two
    (cos/sin) for an oscillatory mode, ONE real exponential ``exp(-gamma t / 2)`` for a
    non-oscillatory (omega ~ 0, pure-decay / DC-drift) mode.  Only the LINEAR amplitudes of the
    'rest' modes are solved.  The pre-fix code sliced ``modes[:max_refine]`` and dropped every
    remaining mode from the design matrix entirely, so their energy aliased into the refined
    oscillators -- an excluded DC component dragged a refined mode to omega -> 0 with a 31x
    spurious amplitude, and even with no DC anywhere the refined reconstruction RMS came out
    1.3-2.7x WORSE than the pencil seed at n_slab = 4..6.

    SAFETY GATE (also finding Q-2): the refined reconstruction RMS is compared against the
    pencil seed's own reconstruction RMS, and a refinement that does not improve on the seed --
    or that sends a refined mode outside [0.5, 2] x its seed omega, to omega ~ 0, or to a
    non-positive/non-finite (omega, gamma) -- is REJECTED and the seed returned unchanged.

    NONDIMENSIONAL PROBLEM (2026-11-02, the wave-1 Q-22 finding "the refinement is INERT").  The
    optimizer works in SAMPLE time ``n = t / dt`` on the PEAK-NORMALIZED trace ``y / max|y|``, so
    its unknowns are ``omega dt`` and ``gamma dt / 2`` (O(0.01..3) for any sampled ringdown) and
    its residual is O(1).  The pre-fix SI parametrization (omega ~ 1e15, gamma/2 ~ 1e13 rad/s on
    an optical trace, field amplitudes of whatever units the solver used) made the raw gradient
    J^T r ~ 1e-15 -- under ``gtol = 1e-14`` before the first step -- so ``least_squares`` stopped
    at nfev = 1 (status 1, dx = 0) on EVERY measured optical trace (FDTD n_slab 3.5/7/10 and the
    synthetic ones): the documented refinement never moved a pole, and the platform-stability
    claim above held only because the pencil itself happened to agree across builds.  Only the
    LINEAR amplitudes were re-solved.  The scaled problem converges in ~4-10 evaluations; the
    returned (omega, gamma, amplitude) are mapped back to SI exactly (divide by dt, multiply by
    the peak), and every gate below runs in SI, unchanged.

    Modes are re-sorted by refined |amplitude|.  Returns ``(modes, note)`` where ``note`` is a
    short debug string ('' when the refinement was accepted).  Falls back to the input modes
    unchanged if scipy is unavailable or the fit fails."""
    y = np.asarray(y, dtype=np.float64)
    seed = list(modes)
    if not seed:
        return seed, "no modes to refine"
    dt = float(dt)
    t = np.arange(y.size) * dt
    n_t = np.arange(y.size, dtype=np.float64)           # SAMPLE time t / dt (nondimensional)
    y_pk = float(np.max(np.abs(y))) if y.size else 0.0
    y_pk = y_pk if (np.isfinite(y_pk) and y_pk > 0.0) else 1.0
    y_n = y / y_pk                                      # peak-normalized data, O(1) residual
    is_osc = [bool(m.omega_rad_s * dt > _OMEGA_DC_TOL) for m in seed]
    # index-based split (finding Q-21: the old `m not in osc` used dataclass VALUE equality, so
    # two degenerate modes both landed in `osc` and `rest` silently lost one).
    ref_idx = [i for i, o in enumerate(is_osc) if o][:int(max_refine)]
    fix_idx = [i for i in range(len(seed)) if i not in set(ref_idx)]
    if not ref_idx:
        return seed, "no oscillatory mode to refine"
    try:
        from scipy.optimize import least_squares
    except Exception:                                   # pragma: no cover - scipy is a core dep
        return seed, "scipy unavailable: refinement skipped"

    K = len(ref_idx)
    w0 = np.array([seed[i].omega_rad_s for i in ref_idx], dtype=np.float64)
    h0 = np.array([0.5 * seed[i].gamma_rad_s for i in ref_idx], dtype=np.float64)

    # FIXED columns: seed (omega, gamma) held, linear amplitude still solved.
    fixed_cols: List[np.ndarray] = []
    fixed_map: List[tuple] = []                         # (mode index, n_cols)
    for i in fix_idx:
        m = seed[i]
        e = np.exp(-0.5 * abs(m.gamma_rad_s) * t)
        if is_osc[i]:
            fixed_cols.append(e * np.cos(m.omega_rad_s * t))
            fixed_cols.append(e * np.sin(m.omega_rad_s * t))
            fixed_map.append((i, 2))
        else:
            fixed_cols.append(e)                        # pure decay / DC-drift column
            fixed_map.append((i, 1))
    # 2026-11-03: the pencil's non-resonant DRIFT terms (``fixed_extra``) are part of the model:
    # FIXED columns (SIGNED rate -- a growing drift grows), amplitudes re-solved, never refined and
    # never returned as modes (re-solved copies go to ``fixed_out``). Empty -> bit-identical.
    for dft in fixed_extra:
        e = np.exp(-0.5 * dft.gamma_rad_s * t)
        if dft.omega_rad_s * dt > _OMEGA_DC_TOL:
            fixed_cols.append(e * np.cos(dft.omega_rad_s * t))
            fixed_cols.append(e * np.sin(dft.omega_rad_s * t))
            fixed_map.append((dft, 2))
        else:
            fixed_cols.append(e)
            fixed_map.append((dft, 1))
    F = (np.column_stack(fixed_cols) if fixed_cols
         else np.empty((y.size, 0), dtype=np.float64))

    def _design(wn, hn):
        # wn = omega dt, hn = gamma dt / 2: the columns exp(-hn n) (cos, sin)(wn n) are the SI
        # exp(-gamma t / 2) (cos, sin)(omega t) up to rounding of the product.
        cols = np.empty((y.size, 2 * K + F.shape[1]), dtype=np.float64)
        for k in range(K):
            e = np.exp(-np.abs(hn[k]) * n_t)
            cols[:, 2 * k] = e * np.cos(wn[k] * n_t)
            cols[:, 2 * k + 1] = e * np.sin(wn[k] * n_t)
        if F.shape[1]:
            cols[:, 2 * K:] = F
        return cols

    def _resid(p):
        A = _design(p[:K], p[K:])
        c, *_ = np.linalg.lstsq(A, y_n, rcond=None)
        return A @ c - y_n

    try:
        sol = least_squares(_resid, np.concatenate([w0 * dt, h0 * dt]), x_scale="jac",
                            ftol=1e-14, xtol=1e-14, gtol=1e-14, max_nfev=400)
        wn, hn = sol.x[:K], np.abs(sol.x[K:])
        w, h = wn / dt, hn / dt                         # back to SI [rad/s]
        A = _design(wn, hn)
        c_n, *_ = np.linalg.lstsq(A, y_n, rcond=None)
        c = c_n * y_pk                                  # amplitudes back in the data's units
        rms = float(np.sqrt(np.mean((A @ c_n - y_n) ** 2))) * y_pk
    except Exception:                                   # pragma: no cover - defensive
        return seed, "least_squares failed: seed kept"

    # --- per-mode sanity: reject an escaped / invented refit outright ---------------------
    for k in range(K):
        gam = 2.0 * h[k]
        ws = w0[k]
        if not (np.isfinite(w[k]) and np.isfinite(gam)) or gam <= 0.0 or w[k] <= 0.0:
            return seed, "refit degenerate (mode {}): seed kept".format(ref_idx[k])
        if w[k] * dt <= _OMEGA_DC_TOL:
            return seed, "refit collapsed a mode to omega ~ 0: seed kept"
        if not (0.5 * ws <= w[k] <= 2.0 * ws):
            return seed, "refit left the seed omega basin (mode {}): seed kept".format(ref_idx[k])

    # --- the cross-cutting safety gate: never accept a refit worse than its own seed -------
    rec_seed = _real_reconstruction(y.size, dt, seed)
    if fixed_extra:
        rec_seed = rec_seed + _drift_reconstruction(y.size, dt, fixed_extra)
    rms_seed = float(np.sqrt(np.mean((rec_seed - y) ** 2)))
    if not np.isfinite(rms) or rms > rms_seed:
        return seed, ("refined RMS {:.6e} > seed RMS {:.6e}: seed kept".format(rms, rms_seed))

    denom = rms + 1e-300
    out: List[Mode] = []
    for k in range(K):
        amp = complex(c[2 * k], c[2 * k + 1])           # y = Re(A e^{-i w t}) e^{-h t}
        gam = 2.0 * h[k]
        out.append(Mode(omega_rad_s=float(w[k]), gamma_rad_s=float(gam),
                        q=float(w[k] / gam), amplitude=amp,
                        snr_est=float(abs(amp) / denom)))
    j = 2 * K
    for i, ncol in fixed_map:                           # seed (omega, gamma), re-solved amplitude
        amp = complex(c[j], c[j + 1]) if ncol == 2 else complex(c[j], 0.0)
        j += ncol
        if isinstance(i, Drift):                        # a drift column: never a mode
            if fixed_out is not None:
                fixed_out.append(Drift(gamma_rad_s=i.gamma_rad_s, omega_rad_s=i.omega_rad_s,
                                       amplitude=amp, threshold_rad_s=i.threshold_rad_s))
            continue
        m = seed[i]
        out.append(Mode(omega_rad_s=m.omega_rad_s, gamma_rad_s=m.gamma_rad_s, q=m.q,
                        amplitude=amp, snr_est=float(abs(amp) / denom)))
    out.sort(key=lambda m: abs(m.amplitude), reverse=True)
    return out, ""


def _pencil_modes(signal, dt: float, *, refine: bool, max_refine: int, **pencil_kw) -> tuple:
    """THE ringdown-mode extraction shared by ringdown_q and fdtd_etalon_ringdown (audit Q-22):
    matrix_pencil, then -- on a REAL trace, when ``refine`` -- the VARPRO refinement
    _nls_refine_real on the same samples the pencil fitted (after its ``t_start`` cut). Returns
    ``(modes, note)``; ``note`` as in _nls_refine_real ('' when the refinement was accepted).
    A COMPLEX trace has no real-trace refinement, so it returns the pencil modes with a note."""
    drift: List[Drift] = []
    modes = matrix_pencil(signal, dt, drift_out=drift, **pencil_kw)
    if not (refine and modes):
        return modes, "", drift
    y = np.asarray(signal)
    dt = float(dt)
    t_start = float(pencil_kw.get("t_start", 0.0))
    n0 = int(np.ceil(t_start / dt)) if t_start > 0.0 else 0     # matrix_pencil's own cut
    y = y[max(0, min(n0, y.size - 1)):]
    real_signal = pencil_kw.get("real_signal")
    if real_signal is None:                             # matrix_pencil's own auto-detect
        real_signal = not (np.iscomplexobj(y) and
                           np.max(np.abs(y.imag)) > 1e-12 * (np.max(np.abs(y)) + 1e-300))
    if not real_signal:
        return modes, "complex trace: refinement not applicable, pencil modes returned", drift
    drift_ref: List[Drift] = []
    out, note = _nls_refine_real(np.real(y), dt, modes, max_refine=max_refine, fixed_extra=drift,
                                 fixed_out=drift_ref)
    return out, note, (drift_ref if drift_ref else drift)


def ringdown_q(signal: Sequence[complex], dt: float, *, refine: bool = True,
               max_refine: int = 6, **kwargs) -> tuple:
    """Convenience: dominant-mode (f0_Hz, Q) of a ringdown trace. Extra kwargs pass through to
    matrix_pencil. Raises ValueError if no mode is found.

    "The ringdown Q" is ONE definition in this module (audit Q-22): matrix_pencil followed, on a
    real trace, by the same VARPRO refinement and dominant-mode choice fdtd_etalon_ringdown applies
    (both go through _pencil_modes). So ``ringdown_q(er.signal_used, er.dt_used, <the pencil
    kwargs fdtd_etalon_ringdown used>, real_signal=True)`` returns exactly ``(er.f0_Hz, er.q)``.
    Before 2026-11-02 this wrapper returned the RAW pencil dominant mode, and the two disagreed on
    the same trace whenever the refinement's re-solved amplitudes re-ranked the modes -- measured
    on two equal-amplitude modes (Q 40 and 25) plus 5% noise: Q = 24.85 here vs 40.18 there.
    ``refine=False`` gives the raw pencil dominant mode (``matrix_pencil(...)[0]``).

    Inherits :func:`matrix_pencil`'s O(N^3)/O(N^2) cost in the RAW trace length and its
    ``max_bytes`` big-N guard (audit P-3): a multi-thousand-sample FDTD trace should be decimated
    (or handed to :func:`fdtd_etalon_ringdown`, which decimates for you) before it gets here."""
    modes, _, drift = _pencil_modes(signal, dt, refine=refine, max_refine=max_refine, **kwargs)
    if not modes:
        raise ValueError("matrix_pencil found no modes in the trace" + (
            " (only {} non-resonant drift term(s) below the window's frequency resolution)".format(
                len(drift)) if drift else ""))
    m = modes[0]                                        # already sorted dominant-first
    return m.f_hz, m.q


# --------------------------------------------------------------------------------------------
# FDTD etalon ringdown helper
# --------------------------------------------------------------------------------------------

def _ringdown_window(sig: np.ndarray, dt: float, f_c: float, *,
                     drop_db_start: float = 50.0, floor_margin: float = 1e3,
                     decades_span: float = 10.0, min_blocks: int = 8,
                     plateau_rel: float = 1e-6, min_drop_db: float = 20.0) -> tuple:
    """Data-driven fit window (i0, i1) for a ringdown trace.

    START after the driven pulse's envelope has fallen ``drop_db_start`` dB below its peak (past
    the fast source tail, onto the exponential cavity leak).  A FIXED-fraction window is a
    precision cliff: on a long record the coherent mode can be at ~1e-13 of peak by the window
    start, so the fit reads the platform-dependent float64 noise floor (observed: two correct
    numpy builds returning Q values 28% apart from bit-identical physics).

    END at whichever comes FIRST of (a) ``decades_span`` decades of ENVELOPE DECAY below the
    window start and (b) ``floor_margin`` x the measured late-time numeric floor -- and only if
    that floor is a genuine PLATEAU (``floor < plateau_rel * env[start]``).  If neither is
    reached the FULL REMAINING RECORD is used.

    FINDING Q-1 (the pre-fix defect).  The end threshold was ``max(floor*floor_margin,
    peak*1e-13)`` with ``floor = median(env[last 10%])`` and no plateau test, so on a trace that
    is still ringing at the end of the record the "floor" is the LIVE SIGNAL: the end threshold
    landed ABOVE the envelope at the window start, the first-crossing search returned index 0,
    and the window collapsed to the hardcoded 8-sample minimum -- which then died downstream in
    ``matrix_pencil`` with "need at least 4 samples". It broke ``fdtd_etalon_ringdown`` from
    ``n_slab ~ 7`` (Q ~ 50) upward, and a LONGER record did not rescue it (lengthening the record
    stretches the decay proportionally, so the median tracks it). The 8-sample window was also the
    terminal state of every malformed input -- pure noise, a badly mis-specified ``f_c`` -- i.e.
    silent failure. The window now (i) drives the END off the envelope decay itself, (ii) never
    collapses (a sub-``min_blocks`` window falls back to the full remaining record), (iii) adapts
    the START threshold when ``drop_db_start`` is not reachable, and (iv) RAISES a named
    diagnostic when the trace genuinely carries no ringdown or is too short.

    The envelope is a block max over ~2 carrier periods at ``f_c``.  ``f_c`` is a caller-supplied
    band centre, so the block width is GUARDED against a mis-specified ``f_c`` by the dominant
    frequency measured from the trace itself (a 100x-too-high ``f_c`` gave sub-period blocks whose
    "envelope" dives into every zero crossing; a 100x-too-low one gave a handful of blocks and a
    silently wrong window).

    Parameters
    ----------
    drop_db_start : dB below the envelope peak at which the fit window starts. If the record never
        gets that far down, HALF the deepest reachable drop is used instead (so a 40 dB-total
        trace starts at 20 dB, still past the source tail); the shipped FDTD configs all reach
        150-300 dB, so the effective start is the full 50 dB and is unchanged by this fallback.
    min_drop_db : if the envelope does not fall even this far below its peak anywhere in the
        usable record, the trace carries no ringdown and a ValueError is raised. Pure white noise
        block-maxima span only ~7.7 dB, well under the 20 dB default.
    decades_span : decades of ENVELOPE amplitude decay below the window start at which the window
        ends. The default 10 (200 dB) is essentially the float64 dynamic range of the march: it is
        a safety net, and it is NOT binding on any shipped config (where the plateau floor is
        reached first).
    floor_margin, plateau_rel : the late-time floor ``median(env[last 10%])`` is used as an end
        threshold (times ``floor_margin``) ONLY when it is a genuine plateau, i.e. below
        ``plateau_rel`` times the envelope at the window start.
    min_blocks : the minimum number of envelope blocks a window must span. A shorter window is
        widened to the full remaining record rather than truncated.

    Raises
    ------
    ValueError
        If the record holds too few envelope blocks after the pulse peak, or if the envelope
        never falls ``min_drop_db`` below its peak (a noise-like trace with no ringdown).
    """
    a = np.abs(np.asarray(sig, dtype=np.float64))
    n_samp = int(a.size)
    dt = float(dt)
    min_blocks = max(2, int(min_blocks))
    if n_samp < 16:
        raise ValueError("_ringdown_window: trace has {} samples; need at least 16.".format(n_samp))

    # ---- block width ~2 carrier periods, guarded against a mis-specified f_c ---------------
    w = max(1, int(round(2.0 / (max(float(f_c), 1e-300) * dt))))
    ac = np.asarray(sig, dtype=np.float64)
    spec = np.abs(np.fft.rfft(ac - float(np.mean(ac))))
    if spec.size > 2:
        k = int(np.argmax(spec[1:])) + 1
        f_meas = float(np.fft.rfftfreq(n_samp, dt)[k])
        if f_meas > 0.0:
            w_meas = max(1, int(round(2.0 / (f_meas * dt))))
            # keep w within [1, 4] carrier periods of the MEASURED carrier: a no-op whenever the
            # supplied f_c is right (all shipped configs sit at 0.96-1.08 x f_meas).
            w = int(min(max(w, max(1, w_meas // 2)), 2 * w_meas))
    w = max(1, min(w, max(1, n_samp // (4 * min_blocks))))   # always >= 4*min_blocks blocks

    nb = max(1, int(np.ceil(n_samp / w)))
    pad = np.pad(a, (0, nb * w - n_samp), constant_values=0.0)
    env = pad.reshape(nb, w).max(axis=1)               # block envelope, one point per w samples
    pk = int(np.argmax(env))
    peak = float(env[pk])
    if peak <= 0.0:
        return 0, n_samp

    # ---- START: deepest reachable drop below the peak, capped at drop_db_start -------------
    last_start = nb - min_blocks                       # a window must still hold min_blocks blocks
    if last_start <= pk:
        raise ValueError(
            "_ringdown_window: record too short -- only {} envelope blocks ({} samples each) "
            "follow the pulse peak, need at least {}. Lengthen the record (larger `settle`) or "
            "lower `min_blocks`.".format(nb - pk, w, min_blocks))
    seg = env[pk:last_start + 1]
    e_min = float(np.min(seg))
    deepest_db = (20.0 * np.log10(peak / e_min)) if e_min > 0.0 else float("inf")
    if deepest_db < float(min_drop_db):
        raise ValueError(
            "_ringdown_window: no ringdown decay detected -- the block envelope falls only "
            "{:.1f} dB below its peak over the usable record (need {:.1f} dB). The trace is "
            "noise-like, or the record ends before the cavity leaks; check the excitation band "
            "and `settle`.".format(deepest_db, float(min_drop_db)))
    drop_eff = min(float(drop_db_start), 0.5 * deepest_db)
    th_start = peak * 10.0 ** (-drop_eff / 20.0)
    below = np.nonzero(seg < th_start)[0]
    b0 = pk + (int(below[0]) if below.size else 1)
    b0 = int(min(max(b0, 0), last_start))

    # ---- END: envelope decay OR a genuine numeric plateau, whichever comes FIRST -----------
    env_start = float(env[b0])
    floor = float(np.median(env[int(0.9 * nb):])) if nb >= 10 else 0.0
    plateau = (floor > 0.0) and (floor < float(plateau_rel) * env_start)
    th_floor = floor * float(floor_margin) if plateau else 0.0
    th_dec = env_start * 10.0 ** (-float(decades_span))
    th_end = max(th_floor, th_dec, peak * 1e-13)
    ends = np.nonzero(env[b0:] < th_end)[0]
    b1 = b0 + (int(ends[0]) if ends.size else nb - b0)
    if b1 - b0 < min_blocks:
        b1 = nb                                        # never collapse: full remaining record

    i0 = int(max(0, min(b0 * w, n_samp - 1)))
    i1 = int(max(i0, min(b1 * w, n_samp)))
    if i1 - i0 < 16:                                   # pragma: no cover - unreachable guard
        raise ValueError(
            "_ringdown_window: the fit window collapsed to {} samples (block width {}, blocks "
            "{}..{} of {}); the trace carries no usable ringdown tail.".format(
                i1 - i0, w, b0, b1, nb))
    return i0, i1


@dataclass
class EtalonRingdown:
    """Result of fdtd_etalon_ringdown: the extracted modes plus the windowed/decimated trace
    that was inverted, and the dominant-mode (f0_Hz, Q).

    ``refine_note`` is a debug string: empty when the NLS (VARPRO) refinement was accepted, and
    otherwise the reason the PENCIL SEED was returned instead (finding Q-2's safety gate --
    e.g. "refined RMS ... > seed RMS ...").  It ALSO carries an ``UNRELIABLE FIT: ...`` prefix
    (alongside an ``UnreliableResultWarning``, a ``RuntimeWarning`` subclass) whenever the
    returned modes fail the ABSOLUTE goodness-of-fit
    tells in ``_fit_quality`` -- the refinement gate is only RELATIVE to its own seed, so a
    source-contaminated window used to come back with an empty note and a 73-93%-wrong Q.
    ``window`` is the (i0, i1) sample window that was cut out of the raw FDTD trace."""

    modes: List[Mode]
    f0_Hz: float
    q: float
    dt_used: float
    t_used: np.ndarray
    signal_used: np.ndarray
    refine_note: str = ""
    window: tuple = (0, 0)
    drift: list = field(default_factory=list)       # non-resonant drift terms (Drift), 2026-11-03


def fdtd_etalon_ringdown(n_slab: float, thickness_m: float, *, lambda_min_m: float,
                         lambda_max_m: float, resolution: int = 30, courant: float = 0.5,
                         settle: float = 12.0, use: str = "reflected",
                         start_frac: Optional[float] = None,
                         target_samples_per_period: int = 20,
                         max_fit_samples: Optional[int] = None, pencil_frac: float = 0.4,
                         svd_tol: float = 1e-6, max_modes: Optional[int] = None,
                         amp_floor: float = 5e-2, refine: bool = True,
                         max_refine: int = 6) -> EtalonRingdown:
    """Drive solve_fdtd_1d on a high-index dielectric slab (a leaky Fabry-Perot etalon), window
    out the driven pulse, decimate the ringdown tail, and matrix-pencil-invert it.

    The etalon rings down as the trapped pulse leaks through the two interfaces; the reflected
    (or transmitted) boundary trace is a damped sinusoid at an etalon mode omega_m = m pi c /
    (n L). Returns the extracted modes and the dominant (f0_Hz, Q).

    Args:
      n_slab, thickness_m : slab index and thickness (vacuum super/substrate).
      lambda_min_m, lambda_max_m : source band; center it on the mode order you want excited.
      use          : "reflected" (default) or "transmitted" tail to invert.
      start_frac   : None (default) = DATA-DRIVEN window via _ringdown_window (start when the
                     envelope is 50 dB below peak, end a margin above the numeric floor --
                     platform-stable, fits the clean exponential decades). A float in (0, 1) gives
                     the legacy fixed-fraction window start. That path is FRAGILE at BOTH ends: too
                     small and the window still holds the driven pulse (start_frac = 0.02 measured
                     Q 73% low at n_slab = 3.5 and 93% low at n_slab = 7, because the envelope
                     peaks at ~6.7% of the record); too large and the mode is already at the
                     float64 floor. It is validated (0 < start_frac < 1), it WARNS when the window
                     begins at or before the envelope peak, and whatever it produces is scored by
                     the absolute fit-quality tells below.
      target_samples_per_period : decimation target (>= ~8 to stay above Nyquist).
      max_fit_samples : None (default) = fit the WHOLE data-driven window; only past the
                     ``_FIT_SAMPLES_BUDGET`` (6000-sample) pencil budget is the stride coarsened
                     (to >= 8 samples/period) and, failing that, the window cut. An int is a hard
                     TRUNCATION of the decimated window (the pre-2026-11-02 default was 1200, which
                     cut the n_slab = 20 fit to 0.42 true e-foldings: Q -44.5%) -- see below. On
                     the legacy ``start_frac`` path None keeps that 1200 cut: that window has no
                     data-driven end and would otherwise run into the float64 floor.
      refine       : NLS (VARPRO) refinement of the pencil modes on the windowed data
                     (default True) -- makes the reported (f0, Q) a platform-stable property
                     of the trace rather than of the LAPACK build (see _nls_refine_real).
                     The refinement is REJECTED (seed returned, reason in
                     ``EtalonRingdown.refine_note``) if it does not beat the seed's own
                     reconstruction RMS.
      max_refine   : how many DOMINANT OSCILLATORY modes have their (omega, gamma) varied by the
                     refinement. Every other mode still contributes FIXED columns to the design
                     matrix, so no mode's energy can alias into a refined one (finding Q-2).
      the remaining kwargs pass through to matrix_pencil.

    OPERATING ENVELOPE (fix-verify W1 items 2 and 9). Two absolute checks run on the result and
    each raises an ``UnreliableResultWarning`` (a ``RuntimeWarning`` subclass) AND annotates
    ``EtalonRingdown.refine_note``:

      * GOODNESS OF FIT -- ``_fit_quality``: reconstruction RMS vs the window's peak-to-peak, and
        summed |amplitude| vs the data peak. The refinement's own gate is RELATIVE (refit vs its
        pencil seed), so when the window is source-contaminated both fits are garbage and the
        comparison passes silently; these two are absolute. Measured: 1.8e-3..2.4e-3 and 1.2..2.1
        on the shipped n_slab = 3.5..15 etalons, versus 8.4e-3..1.4e-1 and 10..5.7e3 on windows
        that cut into the driven pulse.
      * WINDOW SPAN -- the fitted window should cover at least ~3 amplitude e-foldings of the
        dominant mode. Until 2026-11-02 the default ``max_fit_samples = 1200`` TRUNCATED the fit to
        60 carrier periods at 20 samples/period, i.e. ``60 pi / Q`` nepers (13.9 at n_slab = 3.5,
        3.5 at 7, 1.7 at 10, 0.75 at 15), and at n_slab = 20 that is where it broke: 0.42 TRUE
        e-foldings, ~29 near-equal etalon modes unseparated, Q = 252.8 against the Fabry-Perot
        455.2 (-44.5%), RMS tell 3.5e-2. Raising ``settle`` did NOT help at that cap (the window
        starts later and the cut is the same length: settle 24 gave Q = 5408). The default now
        fits the whole data-driven window (1845..4937 decimated samples on the shipped configs,
        0.25-2.4 s of pencil next to a 3-53 s FDTD march), measured vs Fabry-Perot at n_slab
        3.5/5/7/10/15/20: -1.01/+0.31/+1.09/+0.70/+0.63/+0.63%, spans 21.3/14.5/14.0/6.9/3.1/1.10
        nepers. n_slab = 20 still WARNS (1.10 < 3): the window ends at the march's numeric floor
        (settle 24 gives the same 3176-sample, 1.11-neper window and Q 441.29, so `settle` is not a
        lever either), and the flag is CONSERVATIVE -- that Q is +0.6%. An explicit int
        ``max_fit_samples`` that truncates the window says so in the warning.
    """
    from dynameta.optics.fdtd import solve_fdtd_1d, FDTDLayer

    res = solve_fdtd_1d([FDTDLayer(thickness_m=thickness_m, eps_inf=n_slab ** 2)],
                        lambda_min_m=lambda_min_m, lambda_max_m=lambda_max_m,
                        resolution=resolution, courant=courant, settle=settle,
                        return_time_trace=True)
    tr = res.time_trace
    if tr is None:                                      # defensive: the opt-in probe must fire
        raise RuntimeError("solve_fdtd_1d did not attach a time_trace")
    dt = float(tr["dt"])
    sig_full = np.asarray(tr[use], dtype=np.float64)
    N = sig_full.size

    # window: skip the driven transient, keep the CLEAN part of the ringdown tail
    f_c = 0.5 * (C_LIGHT / lambda_max_m + C_LIGHT / lambda_min_m)
    if start_frac is None:
        i0, i1 = _ringdown_window(sig_full, dt, f_c)
    else:                                               # legacy fixed-fraction window
        start_frac = float(start_frac)
        if not (np.isfinite(start_frac) and 0.0 < start_frac < 1.0):
            raise ValueError(
                "fdtd_etalon_ringdown: start_frac must satisfy 0 < start_frac < 1 (it is the "
                "FRACTION of the record at which the legacy fixed window starts); got {!r}. Pass "
                "start_frac=None for the data-driven window.".format(start_frac))
        i0 = max(0, min(int(round(start_frac * N)), N - 8))
        i1 = N
        # The driven pulse peaks well inside the record (~6.7% of it for the shipped configs), so a
        # small start_frac puts the fit ON the source, not on the cavity leak.  That is exactly how
        # start_frac=0.02 produced a Q that was 73-93% low.
        i_pk = int(np.argmax(np.abs(sig_full)))
        if i0 <= i_pk:
            warnings.warn(
                "fdtd_etalon_ringdown: start_frac={:g} starts the fit at sample {} but the trace "
                "envelope still PEAKS at sample {} (fraction {:.4f}) -- the window contains the "
                "driven transient, not the ringdown, and the extracted Q will be wrong. Use "
                "start_frac=None (the data-driven window, which starts {:.0f} dB below the peak) "
                "or a start_frac above {:.3f}.".format(
                    start_frac, i0, i_pk, i_pk / float(N), 50.0, i_pk / float(N)),
                UnreliableResultWarning, stacklevel=2)
    tail = sig_full[i0:i1]

    # decimate to ~target_samples_per_period at the band center (keeps the pencil small + fast)
    stride = max(1, int(round((1.0 / f_c) / (target_samples_per_period * dt))))
    truncated = False
    whole_window = max_fit_samples is None and start_frac is None
    if whole_window:
        # DEFAULT: fit the WHOLE data-driven window (see _FIT_SAMPLES_BUDGET). Over the pencil
        # budget, coarsen the stride before cutting the window: Q lives in the decay SPAN, not in
        # the samples per period.
        n_cap = _FIT_SAMPLES_BUDGET
        if -(-tail.size // stride) > n_cap:
            stride_max = max(stride, int((1.0 / f_c) / (_MIN_SAMPLES_PER_PERIOD * dt)))
            stride = min(stride_max, -(-tail.size // n_cap))
    else:
        # An explicit cap -- or the LEGACY start_frac path, whose window has no data-driven END
        # (it runs to the end of the record, into the float64 floor), so it keeps its pre-2026-11-02
        # 1200-sample cut and its numbers unchanged. (Uncut at start_frac = 0.2, n_slab = 3.5, it
        # fits 4393 samples down into the floor: Q 13.2975 -> 13.2950 plus a GROWING-pole advisory.)
        n_cap = _LEGACY_FIT_SAMPLES if max_fit_samples is None else int(max_fit_samples)
        if n_cap < 16:
            raise ValueError("fdtd_etalon_ringdown: max_fit_samples must be None (the whole "
                             "data-driven window) or an int >= 16; got {!r}"
                             .format(max_fit_samples))
    tail_d = tail[::stride]
    if tail_d.size > n_cap:
        tail_d = tail_d[:n_cap]
        truncated = True
    dt_d = dt * stride
    tail_d = tail_d - np.mean(tail_d)                   # remove any DC / slow ABC drift

    # NLS (VARPRO) refinement: the pencil seed's dominant-mode Q is sensitive to the LAPACK
    # build at marginal model orders (two correct BLAS stacks straddled a 12% gate on the same
    # trace); the refined optimum is a property of the data and platform-stable. Pencil +
    # refinement go through _pencil_modes, the SAME path ringdown_q uses (audit Q-22).
    modes, note, drift = _pencil_modes(tail_d, dt_d, refine=refine, max_refine=max_refine,
                                pencil_frac=pencil_frac, svd_tol=svd_tol, max_modes=max_modes,
                                amp_floor=amp_floor, real_signal=True)
    t_used = np.arange(tail_d.size) * dt_d
    if modes:
        f0, q = modes[0].f_hz, modes[0].q
    else:
        f0, q = float("nan"), float("nan")

    # ---- ABSOLUTE fit-quality tells (fix-verify W1 kill 2) ---------------------------------
    # _nls_refine_real only compares the refit against its own seed, so a window that is still
    # inside the driven pulse produced garbage from BOTH and reported refine_note = '' silently.
    qual = _fit_quality(tail_d, dt_d, modes, drift)
    bad = []
    if not np.isfinite(qual["rms_rel"]) or qual["rms_rel"] > _FIT_RESID_TOL:
        bad.append("reconstruction RMS is {:.3e} of the window's peak-to-peak (limit {:.1e})"
                   .format(qual["rms_rel"], _FIT_RESID_TOL))
    if not np.isfinite(qual["amp_rel"]) or qual["amp_rel"] > _FIT_AMP_SANITY:
        bad.append("summed |amplitude| is {:.3g}x the data peak (limit {:g}: near-cancelling "
                   "modes)".format(qual["amp_rel"], _FIT_AMP_SANITY))
    if bad:
        msg = ("fdtd_etalon_ringdown: the fit does NOT describe the window -- "
               + "; ".join(bad)
               + ". The window is probably source-contaminated (it still holds the driven pulse) "
                 "or too short; (f0, Q) = ({:.4e} Hz, {:.4f}) is not trustworthy. Use "
                 "start_frac=None for the data-driven window.".format(f0, q))
        note = (note + " | " if note else "") + "UNRELIABLE FIT: " + "; ".join(bad)
        warnings.warn(msg, UnreliableResultWarning, stacklevel=2)

    # ---- window-span scope guard (fix-verify W1 item 9) ------------------------------------
    if modes and np.isfinite(q) and t_used.size > 1:
        nepers = 0.5 * modes[0].gamma_rad_s * float(t_used[-1])     # FIELD e-foldings spanned
        if nepers < _MIN_WINDOW_NEPERS:
            if truncated and whole_window:
                why = ("The whole data-driven window ({} samples at {} samples/period) exceeds the "
                       "{}-sample pencil budget even at the coarsest stride, so the fit was cut "
                       "to its first {}. Pass an explicit larger max_fit_samples (the pencil is "
                       "O(N^3) in it) or shorten the record.".format(
                           -(-tail.size // stride), _MIN_SAMPLES_PER_PERIOD,
                           _FIT_SAMPLES_BUDGET, n_cap))
            elif truncated:
                why = ("The fit was TRUNCATED to max_fit_samples = {} decimated samples of the {} "
                       "the window holds -- that cap, not the record, limits the span (at "
                       "n_slab = 20 the old default cap 1200 gave Q = 252.8 vs Fabry-Perot 455.2, "
                       "-44.5%). Pass max_fit_samples=None with start_frac=None to fit the whole "
                       "data-driven window.".format(n_cap, -(-tail.size // stride)))
            else:
                why = ("This is the WHOLE data-driven window ({} decimated samples at stride {}), "
                       "which ends where the envelope reaches the march's numeric floor: neither "
                       "max_fit_samples nor a longer record lengthens it (measured at n_slab = 20: "
                       "settle 24 gave the same 3176-sample, 1.11-neper window, Q 441.29). It is a "
                       "CONSERVATIVE flag -- that whole-window fit at n_slab = 20 (1.10 nepers) "
                       "measured Q = 442.24 vs Fabry-Perot 439.46 (+0.6%).".format(
                           tail_d.size, stride))
            warnings.warn(
                "fdtd_etalon_ringdown: the fitted window spans only {:.2f} amplitude e-foldings "
                "(nepers) of the dominant mode, under the ~{:g} below which the decay rate -- "
                "hence Q = {:.4g} -- is weakly constrained (the fit interpolates a decay it barely "
                "sees). {}".format(nepers, _MIN_WINDOW_NEPERS, q, why),
                UnreliableResultWarning, stacklevel=2)

    return EtalonRingdown(modes=modes, f0_Hz=f0, q=q, dt_used=dt_d,
                          t_used=t_used, signal_used=tail_d, refine_note=note,
                          window=(int(i0), int(i1)), drift=list(drift))
