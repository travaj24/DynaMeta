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
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

from dynameta.constants import C_LIGHT

__all__ = [
    "Mode",
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


def _hankel(y: np.ndarray, L: int) -> np.ndarray:
    """(N-L) x (L+1) Hankel data matrix H[i, j] = y[i + j] (the matrix-pencil data matrix)."""
    N = y.size
    rows = N - L
    # strided view materialized to a contiguous array (residue solve etc. want a real copy)
    idx = np.arange(rows)[:, None] + np.arange(L + 1)[None, :]
    return y[idx]


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
                  omega_zero_tol: float = 1e-6) -> List[Mode]:
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

    Returns:
      list[Mode] sorted by descending |amplitude| (dominant first).
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
    z = np.linalg.eigvals(Zmat)

    # discard non-physical / numerically dead poles: |z| ~ 0 (infinite damping) is meaningless.
    z = z[np.abs(z) > 1e-12]
    if z.size == 0:
        return []
    s = np.log(z) / dt                     # continuous poles s_k = -gamma/2 -/+ i omega_0

    # residues by Vandermonde least squares: y[n] = sum_k R_k z_k^n
    n_idx = np.arange(N)
    Zvand = z[None, :] ** n_idx[:, None]   # N x M
    R, *_ = np.linalg.lstsq(Zvand, y, rcond=None)

    # full reconstruction -> residual RMS for the SNR estimate
    recon = Zvand @ R
    resid = y - (recon.real if real_signal else recon)
    res_rms = float(np.sqrt(np.mean(np.abs(resid) ** 2)))

    modes: List[Mode] = []
    if real_signal:
        # Real signal: poles are real (Im s = 0) or exact conjugate pairs. Report each physical
        # mode once by taking the Im(s) < 0 representative (the exp(-i omega_0 t), i.e. positive
        # physical-frequency member) and doubling its residue; report a real pole (omega ~ 0) once.
        w_tol = omega_zero_tol / dt
        for si, Ri in zip(s, R):
            omega = -si.imag
            gamma = -2.0 * si.real
            if si.imag < -w_tol:                        # positive-frequency representative
                amp = 2.0 * Ri
            elif abs(si.imag) <= w_tol:                 # zero-frequency (pure decay) mode
                omega = 0.0
                amp = complex(Ri.real, 0.0)
            else:                                       # conjugate partner already counted
                continue
            modes.append((omega, gamma, amp))
    else:
        for si, Ri in zip(s, R):
            modes.append((-si.imag, -2.0 * si.real, Ri))

    if not modes:
        return []

    amax = max(abs(a) for _, _, a in modes)
    out: List[Mode] = []
    for omega, gamma, amp in modes:
        if amax > 0.0 and abs(amp) < amp_floor * amax:
            continue                                    # residual/amplitude prune
        q = float(omega / gamma) if gamma != 0.0 else np.inf
        snr = float(abs(amp) / res_rms) if res_rms > 0.0 else np.inf
        out.append(Mode(omega_rad_s=float(omega), gamma_rad_s=float(gamma), q=q,
                        amplitude=complex(amp), snr_est=snr))
    out.sort(key=lambda m: abs(m.amplitude), reverse=True)
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


def _fit_quality(y: np.ndarray, dt: float, modes: Sequence[Mode]) -> dict:
    """ABSOLUTE goodness-of-fit of ``modes`` on the windowed trace ``y``: the reconstruction RMS
    relative to the data's peak-to-peak, and the summed |amplitude| relative to the data's peak.
    Both are scale-free and neither is a comparison against another (possibly equally bad) fit."""
    y = np.asarray(y, dtype=np.float64)
    ptp = float(np.ptp(y))
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    rms = float(np.sqrt(np.mean((_real_reconstruction(y.size, dt, modes) - y) ** 2))) if modes \
        else float(np.sqrt(np.mean(y ** 2)))
    amp = float(sum(abs(m.amplitude) for m in modes))
    return {"rms": rms, "ptp": ptp, "peak": peak, "amp_sum": amp,
            "rms_rel": rms / ptp if ptp > 0.0 else float("inf"),
            "amp_rel": amp / peak if peak > 0.0 else float("inf")}


def _nls_refine_real(y: np.ndarray, dt: float, modes: List[Mode], *,
                     max_refine: int = 6) -> tuple:
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

    Modes are re-sorted by refined |amplitude|.  Returns ``(modes, note)`` where ``note`` is a
    short debug string ('' when the refinement was accepted).  Falls back to the input modes
    unchanged if scipy is unavailable or the fit fails."""
    y = np.asarray(y, dtype=np.float64)
    seed = list(modes)
    if not seed:
        return seed, "no modes to refine"
    dt = float(dt)
    t = np.arange(y.size) * dt
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
    F = (np.column_stack(fixed_cols) if fixed_cols
         else np.empty((y.size, 0), dtype=np.float64))

    def _design(w, h):
        cols = np.empty((y.size, 2 * K + F.shape[1]), dtype=np.float64)
        for k in range(K):
            e = np.exp(-np.abs(h[k]) * t)
            cols[:, 2 * k] = e * np.cos(w[k] * t)
            cols[:, 2 * k + 1] = e * np.sin(w[k] * t)
        if F.shape[1]:
            cols[:, 2 * K:] = F
        return cols

    def _resid(p):
        A = _design(p[:K], p[K:])
        c, *_ = np.linalg.lstsq(A, y, rcond=None)
        return A @ c - y

    try:
        sol = least_squares(_resid, np.concatenate([w0, h0]), x_scale="jac",
                            ftol=1e-14, xtol=1e-14, gtol=1e-14, max_nfev=400)
        w, h = sol.x[:K], np.abs(sol.x[K:])
        A = _design(w, h)
        c, *_ = np.linalg.lstsq(A, y, rcond=None)
        rms = float(np.sqrt(np.mean((A @ c - y) ** 2)))
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
    rms_seed = float(np.sqrt(np.mean((_real_reconstruction(y.size, dt, seed) - y) ** 2)))
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
        m = seed[i]
        amp = complex(c[j], c[j + 1]) if ncol == 2 else complex(c[j], 0.0)
        j += ncol
        out.append(Mode(omega_rad_s=m.omega_rad_s, gamma_rad_s=m.gamma_rad_s, q=m.q,
                        amplitude=amp, snr_est=float(abs(amp) / denom)))
    out.sort(key=lambda m: abs(m.amplitude), reverse=True)
    return out, ""


def ringdown_q(signal: Sequence[complex], dt: float, **kwargs) -> tuple:
    """Convenience: dominant-mode (f0_Hz, Q) of a ringdown trace. Extra kwargs pass through to
    matrix_pencil. Raises ValueError if no mode is found."""
    modes = matrix_pencil(signal, dt, **kwargs)
    if not modes:
        raise ValueError("matrix_pencil found no modes in the trace")
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
    (alongside a ``RuntimeWarning``) whenever the returned modes fail the ABSOLUTE goodness-of-fit
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


def fdtd_etalon_ringdown(n_slab: float, thickness_m: float, *, lambda_min_m: float,
                         lambda_max_m: float, resolution: int = 30, courant: float = 0.5,
                         settle: float = 12.0, use: str = "reflected",
                         start_frac: Optional[float] = None,
                         target_samples_per_period: int = 20,
                         max_fit_samples: int = 1200, pencil_frac: float = 0.4,
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
      max_fit_samples : cap on the number of decimated samples fed to the pencil (speed).
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
    each raises a ``RuntimeWarning`` AND annotates ``EtalonRingdown.refine_note``:

      * GOODNESS OF FIT -- ``_fit_quality``: reconstruction RMS vs the window's peak-to-peak, and
        summed |amplitude| vs the data peak. The refinement's own gate is RELATIVE (refit vs its
        pencil seed), so when the window is source-contaminated both fits are garbage and the
        comparison passes silently; these two are absolute. Measured: 1.8e-3..2.4e-3 and 1.2..2.1
        on the shipped n_slab = 3.5..15 etalons, versus 8.4e-3..1.4e-1 and 10..5.7e3 on windows
        that cut into the driven pulse.
      * WINDOW SPAN -- the fitted window should cover at least ~3 amplitude e-foldings of the
        dominant mode. ``max_fit_samples = 1200`` at ``target_samples_per_period = 20`` is 60
        carrier periods, i.e. only ``60 pi / Q`` nepers: 13.7 at n_slab = 3.5, 3.5 at 7, 1.7 at 10,
        0.75 at 15. THAT is the shipped scope limit. It is a CONSERVATIVE flag -- on these clean
        traces the extraction still holds to +0.7% at n_slab = 10 and 15 -- but n_slab = 20 is
        where it breaks: Q = 169.9 against the Fabry-Perot 455.2 (-62.7%), and the RMS tell flags
        that one too (3.7e-2). Raise ``max_fit_samples`` (and ``settle`` for the extra record) to
        go higher; the pencil is O(N^3) in the fitted length, so this is a deliberate speed/range
        trade, not a bound of the method.
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
                RuntimeWarning, stacklevel=2)
    tail = sig_full[i0:i1]

    # decimate to ~target_samples_per_period at the band center (keeps the pencil small + fast)
    stride = max(1, int(round((1.0 / f_c) / (target_samples_per_period * dt))))
    tail_d = tail[::stride]
    if tail_d.size > max_fit_samples:
        tail_d = tail_d[:max_fit_samples]
    dt_d = dt * stride
    tail_d = tail_d - np.mean(tail_d)                   # remove any DC / slow ABC drift

    modes = matrix_pencil(tail_d, dt_d, pencil_frac=pencil_frac, svd_tol=svd_tol,
                          max_modes=max_modes, amp_floor=amp_floor, real_signal=True)
    # NLS (VARPRO) refinement: the pencil seed's dominant-mode Q is sensitive to the LAPACK
    # build at marginal model orders (two correct BLAS stacks straddled a 12% gate on the same
    # trace); the refined optimum is a property of the data and platform-stable.
    note = ""
    if refine and modes:
        modes, note = _nls_refine_real(tail_d, dt_d, modes, max_refine=max_refine)
    t_used = np.arange(tail_d.size) * dt_d
    if modes:
        f0, q = modes[0].f_hz, modes[0].q
    else:
        f0, q = float("nan"), float("nan")

    # ---- ABSOLUTE fit-quality tells (fix-verify W1 kill 2) ---------------------------------
    # _nls_refine_real only compares the refit against its own seed, so a window that is still
    # inside the driven pulse produced garbage from BOTH and reported refine_note = '' silently.
    qual = _fit_quality(tail_d, dt_d, modes)
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
        warnings.warn(msg, RuntimeWarning, stacklevel=2)

    # ---- window-span scope guard (fix-verify W1 item 9) ------------------------------------
    if modes and np.isfinite(q) and t_used.size > 1:
        nepers = 0.5 * modes[0].gamma_rad_s * float(t_used[-1])     # FIELD e-foldings spanned
        if nepers < _MIN_WINDOW_NEPERS:
            warnings.warn(
                "fdtd_etalon_ringdown: the fitted window spans only {:.2f} amplitude e-foldings "
                "(nepers) of the dominant mode, under the ~{:g} below which the decay rate -- "
                "hence Q = {:.4g} -- is weakly constrained (the fit interpolates a decay it barely "
                "sees). The span is {} decimated samples at {} samples/period, i.e. pi * "
                "n_periods / Q nepers, so it shrinks as 1/Q: this is the max_fit_samples cap "
                "meeting a high-Q cavity, NOT a limit of the method. Raise max_fit_samples (and "
                "`settle` for the extra record) or lower target_samples_per_period. This is a "
                "CONSERVATIVE flag -- on clean traces the extraction held to +0.7% at n_slab = 10 "
                "(1.7 nepers) and 15 (0.75) -- but it is where it breaks: n_slab = 20 returns "
                "Q = 169.9 against a Fabry-Perot 455.2 (-62.7%).".format(
                    nepers, _MIN_WINDOW_NEPERS, q, int(min(max_fit_samples, tail_d.size)),
                    int(target_samples_per_period)),
                RuntimeWarning, stacklevel=2)

    return EtalonRingdown(modes=modes, f0_Hz=f0, q=q, dt_used=dt_d,
                          t_used=t_used, signal_used=tail_d, refine_note=note,
                          window=(int(i0), int(i1)))
