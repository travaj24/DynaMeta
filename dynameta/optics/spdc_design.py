"""SPDC DESIGN tier -- spontaneous parametric down-conversion via the quantum-classical
(stimulated <-> spontaneous) correspondence (roadmap item 4.4).

No quantum state is simulated. This module estimates the *design-level* observables of a
chi2 photon-pair source -- pair generation rate, joint spectral amplitude (JSA), Schmidt
number, heralded bandwidths -- from the CLASSICAL three-wave-mixing physics of
``twm_reference`` (item 4.1). The bridge is the Helt-Liscidini-Sipe correspondence.

--------------------------------------------------------------------------------------------
THE HELT-LISCIDINI-SIPE CORRESPONDENCE  (the "reversed SFG" / stimulated-emission relation)
--------------------------------------------------------------------------------------------
Helt, Liscidini & Sipe, JOSA B 29, 2199 (2012); Liscidini & Sipe, PRL 111, 193602 (2013).

The spontaneous process is the STIMULATED (seeded) process with the seed replaced by the
vacuum -- "one photon per mode". Take the classical difference-frequency partner of SPDC:
a strong CW pump at omega_p and a bright CW seed at the signal frequency omega_s generate an
idler at omega_i = omega_p - omega_s. Let

    G(omega_s) := (idler photons generated) / (seed signal photons in)

be the classical, dimensionless, undepleted-pump photon-number DFG efficiency for that
signal frequency. Each generated idler photon is emitted together with an added signal
photon -- a PAIR. Replacing the seed with vacuum injects exactly one photon per temporal
mode; a signal bandwidth d(omega_s) carries d(omega_s) / (2 pi) modes per unit time, so

    PREFACTOR STATEMENT:   dR_pairs / d(omega_s) = G(omega_s) / (2 pi)          [pairs / s per
                                                                                 unit omega_s]
    R_pairs = (1 / 2 pi) integral d(omega_s) G(omega_s)                          [pairs / s]

The universal prefactor multiplying the classical per-photon efficiency is exactly 1/(2 pi)
(the temporal-mode density). Because DFG and SFG share the same |kappa|^2 coupling, G is
equivalently the classical SFG per-photon efficiency for combining omega_s + omega_i ->
omega_p; that is the "reversed SFG" form of the correspondence.

Using ``twm_reference``'s undepleted low-gain result G = kappa_s kappa_i |A_p|^2 L^2
|sinc(dk L/2)|^2 (dk = k_p - k_s - k_i, kappa_j = omega_j d_eff/(n_j c), |A_p|^2 = 2 P_p /
(A n_p eps0 c) for a beam of area A and power P_p) this gives the textbook CW bulk-crystal
spectral pair rate

    dR / d(omega_s) = (omega_s omega_i d_eff^2 L^2 P_p) / (pi n_s n_i n_p eps0 c^3 A)
                      |sinc(dk L / 2)|^2 ,

which is dimensionless (R in pairs/s, R/P_p in pairs/s/W). Integrated over the sinc^2
phase-matching bandwidth (~1/L) it scales as R ~ L, the standard bulk result.

ASSUMPTIONS (all inherited from ``twm_reference``'s undepleted CWEs): undepleted CW pump;
lossless, collinear, single spatial mode of effective area A; undepleted signal/idler
(spontaneous / low-gain limit, so cosh/sinh -> the leading (gL)); slowly varying
(Delta omega << omega). Multimode/entangled-pump structure enters only through the JSA below.

--------------------------------------------------------------------------------------------
JOINT SPECTRAL AMPLITUDE
--------------------------------------------------------------------------------------------
For a pump of (possibly pulsed) spectral envelope alpha(omega), energy conservation ties the
pair to the pump at omega_s + omega_i, and momentum conservation to the phase-matching
function Phi (the sinc of ``twm_reference``, incl. QPM). The JSA factorizes as

    f(omega_s, omega_i) = alpha(omega_s + omega_i) * Phi(dk(omega_s, omega_i), L, Lambda),

returned here normalized (Frobenius norm 1). Its Schmidt decomposition (SVD) gives the
Schmidt number K = 1 / sum_k lambda_k^2 (lambda_k = normalized squared singular values); K = 1
is a spectrally pure (separable) heralded single photon, K >> 1 a highly entangled pair.
"""

from __future__ import annotations

import math
import warnings
from typing import Callable, Optional, Tuple

import numpy as np

from dynameta.constants import C_LIGHT, EPS0
from dynameta.optics.twm_reference import phase_matching_sinc

__all__ = [
    "pair_rate_from_sfg",
    "spectral_pair_rate_closed_form",
    "jsa",
    "jsi",
    "schmidt_number",
    "heralded_bandwidths",
    "HELT_SIPE_PREFACTOR",
]

# The universal Helt-Liscidini-Sipe prefactor: pairs/s per unit signal angular frequency
# equal 1/(2 pi) times the classical per-photon DFG/SFG efficiency (temporal-mode density).
HELT_SIPE_PREFACTOR = 1.0 / (2.0 * math.pi)

# numpy >=2.0 renamed trapz -> trapezoid; support both.
_trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))


def _pump_amp_sq(pump_power_W: float, area_m2: float, n_p: float) -> float:
    """|A_p|^2 (V/m)^2 from CW pump power and beam area, real-peak convention I = (1/2) n eps0
    c |A|^2 => |A_p|^2 = 2 P_p / (A n_p eps0 c)."""
    return 2.0 * float(pump_power_W) / (float(area_m2) * n_p * EPS0 * C_LIGHT)


def pair_rate_from_sfg(omega_s_grid: np.ndarray, omega_p: float, d_eff: float, length_m: float,
                       *, n_s=1.0, n_i=1.0, n_p=1.0, pump_power_W: float, area_m2: float,
                       dk_func: Optional[Callable[[float, float], float]] = None,
                       qpm_period: Optional[float] = None) -> dict:
    """SPDC pair rate for a CW pump via the Helt-Liscidini-Sipe correspondence.

    For every signal frequency in ``omega_s_grid`` the idler is fixed by energy conservation
    omega_i = omega_p - omega_s. The classical per-photon efficiency
        G(omega_s) = kappa_s kappa_i |A_p|^2 L^2 |Phi(dk)|^2
    is evaluated from ``twm_reference`` (kappa_j = omega_j d_eff/(n_j c)); the pair spectral
    density is G / (2 pi) and the total rate integrates it over omega_s.

    ``dk_func(omega_s, omega_i) -> dk`` supplies the phase mismatch (default: perfect phase
    matching, dk = 0, the flat uniform-crystal limit). ``qpm_period`` engages first-order QPM.
    Indices ``n_s/n_i/n_p`` may be floats or callables n(omega).

    DEGENERACY CONVENTION (audit Q-14). G is built here directly from ``kappa_s kappa_i`` --
    ``twm_reference``'s NONDEGENERATE (``distinct_modes=True``) coupling, with no 1/2 degeneracy
    factor -- and it never routes through ``sfg_undepleted``, so that function's degenerate-spec
    refusal does not reach this path. The convention therefore holds unchanged AT the
    frequency-degenerate point omega_s = omega_i = omega_p/2: signal and idler are treated as
    DISTINGUISHABLE modes (the type-II / orthogonal-polarization case, where the nondegenerate
    coupling is the correct one), and ``rate_pairs_per_s`` integrates dR/domega_s over the whole
    supplied ``omega_s_grid``, counting each (omega_s, omega_i) assignment once. For a type-I
    setup in which signal and idler are the SAME mode, (omega_s, omega_i) and (omega_i, omega_s)
    are the same physical pair, so a grid symmetric about omega_p/2 counts every pair twice and
    the integrated rate must be halved; the single-mode degenerate coupling additionally carries
    the ``shg_undepleted`` 1/2 factor. Neither correction is applied here.

    Returns the spectral density dR/domega_s (array, dimensionless), the integrated rate R
    (pairs/s), R per watt (pairs/s/W), and G(omega_s)."""
    omega_s = np.asarray(omega_s_grid, dtype=float)
    omega_i = float(omega_p) - omega_s

    def _n(n, w):
        return np.array([float(n(x)) for x in np.atleast_1d(w)]) if callable(n) else \
            np.full(np.atleast_1d(w).shape, float(n))

    ns = _n(n_s, omega_s)
    ni = _n(n_i, omega_i)
    npu = float(n_p(omega_p)) if callable(n_p) else float(n_p)

    kappa_s = omega_s * d_eff / (ns * C_LIGHT)
    kappa_i = omega_i * d_eff / (ni * C_LIGHT)
    Ap2 = _pump_amp_sq(pump_power_W, area_m2, npu)

    if dk_func is None:
        dk = np.zeros_like(omega_s)
    else:
        dk = np.array([float(dk_func(ws, wi)) for ws, wi in zip(omega_s, omega_i)])
    phi = phase_matching_sinc(dk, length_m, qpm_period)
    G = kappa_s * kappa_i * Ap2 * length_m ** 2 * np.abs(phi) ** 2

    spectral = HELT_SIPE_PREFACTOR * G
    R = float(_trapz(spectral, omega_s)) if omega_s.size > 1 else float(spectral[0])
    return {
        "spectral_density": spectral,           # dR/domega_s  (dimensionless)
        "rate_pairs_per_s": R,
        "rate_per_watt": R / float(pump_power_W) if pump_power_W else float("nan"),
        "G": G,
        "omega_i": omega_i,
    }


def spectral_pair_rate_closed_form(omega_s, omega_p, d_eff, length_m, *, n_s=1.0, n_i=1.0,
                                   n_p=1.0, pump_power_W, area_m2, dk=0.0,
                                   qpm_period: Optional[float] = None):
    """The bulk-crystal spectral pair rate written out directly from the correspondence:

        dR/domega_s = (omega_s omega_i d_eff^2 L^2 P_p)/(pi n_s n_i n_p eps0 c^3 A) |sinc(dk L/2)|^2.

    Provided as an independent, in-closed-form cross-check of ``pair_rate_from_sfg`` (they
    must agree in the uniform limit). ``omega_s`` may be scalar or array; ``omega_i`` = omega_p
    - omega_s."""
    omega_s = np.asarray(omega_s, dtype=float)
    omega_i = float(omega_p) - omega_s
    ns = np.array([float(n_s(x)) for x in np.atleast_1d(omega_s)]) if callable(n_s) else \
        np.full(np.atleast_1d(omega_s).shape, float(n_s))
    ni = np.array([float(n_i(x)) for x in np.atleast_1d(omega_i)]) if callable(n_i) else \
        np.full(np.atleast_1d(omega_i).shape, float(n_i))
    npu = float(n_p(omega_p)) if callable(n_p) else float(n_p)
    phi = phase_matching_sinc(np.asarray(dk, dtype=float) * np.ones_like(omega_s),
                              length_m, qpm_period)
    pref = (omega_s * omega_i * d_eff ** 2 * length_m ** 2 * float(pump_power_W)) / \
           (math.pi * ns * ni * npu * EPS0 * C_LIGHT ** 3 * float(area_m2))
    return pref * np.abs(phi) ** 2


def jsa(omega_s_grid: np.ndarray, omega_i_grid: np.ndarray,
        pump_envelope: Callable[[np.ndarray], np.ndarray],
        dk_func: Callable[[float, float], float], length_m: float,
        *, qpm_period: Optional[float] = None, normalize: bool = True) -> np.ndarray:
    """Joint spectral amplitude f(omega_s, omega_i) = alpha(omega_s + omega_i) *
    Phi(dk(omega_s, omega_i), L, Lambda), on the outer grid of ``omega_s_grid`` (rows) x
    ``omega_i_grid`` (cols).

    ``pump_envelope(omega)`` returns the (complex) pump spectral amplitude at omega =
    omega_s + omega_i. ``dk_func(omega_s, omega_i)`` returns the phase mismatch; the
    phase-matching function is ``twm_reference.phase_matching_sinc`` (sinc, incl. QPM).
    Normalized to unit Frobenius norm by default.

    ``dk_func`` IS CALLED WITH ARRAYS FIRST (audit P-5).  It is tried once on the whole
    ``(n_s, n_i)`` grid; if that raises, or does not return an array of the grid's shape, the
    call falls back to the per-point ``np.vectorize`` pass, so a scalar-only ``dk_func`` (one
    that branches on ``if dk < 0`` or calls ``math.*``) keeps working exactly as before.  For an
    elementwise (numpy-written) ``dk_func`` the two produce bit-identical ``dk`` while the array
    call is ~441x faster on a 220 x 220 grid -- the vectorize pass is one Python call plus a
    ``float()`` round trip per grid point, ~1.6 s of pure dispatch on a 1024 x 1024 JSA.  A
    ``dk_func`` with SIDE EFFECTS (a counter, a log) will therefore see one extra whole-grid call
    before the fallback."""
    ws = np.asarray(omega_s_grid, dtype=float)
    wi = np.asarray(omega_i_grid, dtype=float)
    WS, WI = np.meshgrid(ws, wi, indexing="ij")
    alpha = np.asarray(pump_envelope(WS + WI), dtype=complex)
    dk = None
    try:                                        # elementwise dk_func: one vectorized call
        trial = np.asarray(dk_func(WS, WI), dtype=float)
        if trial.shape == WS.shape:
            dk = trial
    except Exception:                           # scalar-only dk_func (branches, math.*, ...)
        dk = None
    if dk is None:                              # unchanged legacy path, incl. a scalar return
        dk = np.vectorize(lambda a, b: float(dk_func(a, b)))(WS, WI)
    phi = phase_matching_sinc(dk, length_m, qpm_period)
    f = alpha * phi
    if normalize:
        nrm = np.sqrt(np.sum(np.abs(f) ** 2))
        if nrm > 0:
            f = f / nrm
    return f


def jsi(jsa_matrix: np.ndarray) -> np.ndarray:
    """Joint spectral INTENSITY |f|^2 (the directly measurable coincidence spectrum)."""
    return np.abs(np.asarray(jsa_matrix)) ** 2


def schmidt_number(jsa_matrix: np.ndarray) -> dict:
    """Schmidt number K = 1 / sum_k lambda_k^2 of the JSA, from its singular values s_k
    (lambda_k = s_k^2 / sum s^2, the normalized Schmidt coefficients). K = 1 is a separable /
    spectrally pure state (heralds a pure single photon); K >> 1 is a spectrally entangled
    pair. Returns K, the purity 1/K, and the normalized Schmidt spectrum."""
    f = np.asarray(jsa_matrix, dtype=complex)
    s = np.linalg.svd(f, compute_uv=False)
    lam = s ** 2
    tot = float(np.sum(lam))
    if tot <= 0:
        raise ValueError("schmidt_number: JSA has zero norm.")
    lam = lam / tot
    K = 1.0 / float(np.sum(lam ** 2))
    return {"schmidt_number": K, "purity": 1.0 / K, "schmidt_spectrum": lam}


def _fwhm_resolved(x: np.ndarray, y: np.ndarray) -> Tuple[float, bool]:
    """Full width at half maximum of a single-peaked profile y(x) by linear interpolation of the
    half-max crossings, plus whether that width is RESOLVED by the sampled window.

    ``resolved`` is False when the profile is still at or above half-max at the first or the last
    sample: there is no crossing to interpolate on that side, so the returned number is a LOWER
    BOUND set by the window, not a width of the physics (audit Q-13). This is an exact test, not a
    tolerance -- it is precisely the condition under which the ``x[lo]`` / ``x[hi]`` edge fallbacks
    below fire. Returns (0.0, False) for a degenerate/empty profile."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if y.size < 2 or np.max(y) <= 0:
        return 0.0, False
    half = 0.5 * np.max(y)
    above = y >= half
    idx = np.where(above)[0]
    if idx.size == 0:
        return 0.0, False
    lo, hi = idx[0], idx[-1]

    def _cross(i0, i1):
        if i0 == i1:
            return x[i0]
        x0, x1, y0, y1 = x[i0], x[i1], y[i0], y[i1]
        if y1 == y0:
            return x0
        return x0 + (half - y0) * (x1 - x0) / (y1 - y0)

    xl = _cross(lo - 1, lo) if lo > 0 else x[lo]
    xr = _cross(hi, hi + 1) if hi < x.size - 1 else x[hi]
    return abs(xr - xl), bool(lo > 0 and hi < x.size - 1)


def _fwhm(x: np.ndarray, y: np.ndarray) -> float:
    """Full width at half maximum of a single-peaked profile y(x) -- the width alone; see
    :func:`_fwhm_resolved` for the width plus its window-resolution flag."""
    return _fwhm_resolved(x, y)[0]


def _cell_weights(ws: np.ndarray, wi: np.ndarray) -> np.ndarray:
    """Quadrature weight of every (omega_s, omega_i) grid CELL: the outer product of the 1-D
    trapezoid weights, i.e. ``dw_s[i] * dw_i[j]`` on a uniform grid. This is the frequency-grid
    Jacobian the rotated marginals need (audit Q-13): a raw ``np.histogram(..., weights=P)`` sums
    bare samples, which is proportional to the integral only when every cell has the same area.
    On a NON-UNIFORM grid it is not, and the reported widths are wrong by tens of percent
    (measured: 62 % on a power-1.6 stretched grid, vs 2.5 % with these weights). Matches the
    ``trapezoid`` rule already used for the signal / idler marginals."""
    def _w(a):
        a = np.asarray(a, dtype=float)
        if a.size == 1:
            return np.ones(1)
        w = np.empty_like(a)
        w[1:-1] = 0.5 * (a[2:] - a[:-2])
        w[0] = 0.5 * (a[1] - a[0])
        w[-1] = 0.5 * (a[-1] - a[-2])
        return w
    return np.outer(_w(ws), _w(wi))


# Largest max/min spread of the per-bin quadrature measure (`den` below) at which the rotated
# marginals are believed. See `heralded_bandwidths`' GRID-UNIFORMITY GUARD section for the
# calibration; 2.1 sits ~6 % above the worst measured value on a grid that is UNIFORM in each
# axis (1.988, the worst over 224 uniform grids of unequal N and unequal span) and ~6 % below the
# least offending measured failure (2.233, a log-spaced 1601x1601 grid whose phase-matching width
# is 31 % low and gets WORSE with refinement).
_ROTATED_MEASURE_MAX_RATIO = 2.1


# Smallest fraction of the requested bins that may survive before the rotated coordinate is
# declared DESTROYED rather than merely coarse. On any grid whose omega values resolve their own
# spacing, essentially every bin is occupied (measured >= 0.98 over uniform, power-law, log and
# random families). It collapses only when the grid SPAN is at the floating-point resolution of
# the CARRIER -- omega ~ 1.2e15 sampled over a span of 5 rad/s puts the spacing below the ulp of
# omega_s + omega_i, so 400 distinct frequencies alias onto 11 distinct sums (measured
# kept/nbin = 0.014-0.11, reported pump width -100 %). 0.5 sits ~4.5x above that and ~2x below
# every healthy case.
_ROTATED_MIN_BIN_FRACTION = 0.5

# Relative tolerance for treating two values of the rotated coordinate as the SAME comb line (see
# `_rotated_marginal`). Scaled by the coordinate's own range, so it is a statement about the comb,
# not about the carrier: it must swallow the ~1e-16-relative jitter of `omega_s + omega_i` while
# staying far below the comb spacing (1/nbin of the range).
_ROTATED_COMB_REL_TOL = 1e-9


def _measure_ratio(den: np.ndarray) -> float:
    """max/min of the per-bin quadrature measure over the KEPT bins -- 1.0 when every retained
    u-bin integrates the same area of the (omega_s, omega_i) plane, which is the condition under
    which the un-normalised per-bin sum IS the marginal. ``inf`` for an empty/degenerate profile
    (which the caller then treats as unusable)."""
    d = np.asarray(den, dtype=float)
    if d.size == 0 or not np.all(np.isfinite(d)) or float(np.min(d)) <= 0.0:
        return float("inf")
    return float(np.max(d) / np.min(d))


def _rotated_marginal(P: np.ndarray, u_vals: np.ndarray, weights: np.ndarray,
                      mask: np.ndarray, nbin: int
                      ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Quadrature-project the JSI ``P`` onto a rotated coordinate ``u_vals`` over the cells
    selected by ``mask``: returns (abscissa, marginal, measure).

    ``marginal`` is the AREA-WEIGHTED sum ``sum P * dw_s dw_i`` per bin -- a genuine
    ``integral |f|^2 dv`` up to the constant rotation Jacobian, not a bare sample count.
    ``measure`` is the same sum with ``P`` replaced by 1, i.e. the quadrature measure of the bin;
    over the constant-v-extent window of :func:`heralded_bandwidths` it is the SAME for every bin
    when the grid is uniform, so its spread is the estimator's own self-check (see
    :func:`_measure_ratio`). Bins with no cell in them are dropped (on a uniform grid the rotated
    coordinate is quantized, so a finer binning leaves a comb of empty bins that corrupts the
    half-max interpolation).

    ABSCISSA = MEASURE CENTROID, NOT BIN CENTRE (audit Q-13, the even-N parity comb). The rotated
    coordinate on a uniform grid is a COMB: ``u = omega_s + omega_i`` takes one value per
    ``i + j``. Binning a comb on a ``linspace`` of ``nbin`` bins is an aliasing problem, and the
    geometric bin centre is not where the mass in that bin actually sits:

      * the centres span only ``(1 - 1/nbin)`` of the data range, so EVERY width measured on them
        is systematically low by that factor (the "-0.246 % residual that halves with the grid"
        this module used to report at N = 401 is exactly 1/401 = 0.249 %, not discretisation);
      * when the comb spacing and the bin width are incommensurate the value-to-centre
        displacement is a saw of up to half a bin, whose phase depends on the PARITY of N.
        Measured on the shipped 401-point gate configuration: +0.83 % at N = 400 against
        -0.25 % at N = 401, a 1.07 % parity step that grew to ~10 % at N = 50.

    Reporting each bin at the measure-weighted centroid of the coordinate inside it removes both:
    a bin holding exactly one comb line is reported AT that line (exactly), and a bin that
    aliases two is reported at their measure centroid, which is where their summed mass is. The
    two parities then agree to ~1e-5 relative (measured +0.0031 % at N = 400 vs +0.0034 % at
    N = 401, from +0.83 % / -0.25 %), and the residual falls ~100x on every uniform grid tested.
    The abscissa is weighted by the MEASURE, not by ``P``: the location of a bin is a property of
    the grid, so it must not move with the data.

    BIN EDGES FOLLOW THE COMB WHEN THERE IS ONE. A ``linspace`` of ``nbin`` bins does not divide
    the comb, so a comb line can land on a bin EDGE and have its mass split in two. That is the
    same aliasing as above but in amplitude rather than position, and the split does not need an
    incommensurate spacing to happen -- floating-point jitter suffices. On a grid symmetric about
    zero, ``omega_s + omega_i`` cancels to ~1e-16 relative, which is enough to scatter each comb
    line across an edge: measured on ``linspace(-5, 5, 400)`` the per-bin measure spread was 3.85
    (it is 1.005 on the same grid offset by any carrier). So when the coordinate IS a comb -- the
    distinct values, clustered at ``_ROTATED_COMB_REL_TOL`` of the range, number no more than
    ``nbin`` -- the edges are placed at the midpoints BETWEEN comb lines instead, giving exactly
    one line per bin. On a genuinely non-uniform grid every value is distinct, the cluster count
    exceeds ``nbin``, and the plain ``linspace`` is used unchanged.

    Measured effect: the zero-centred grids above go from a measure spread of 3.13-4.35 (which
    the caller's uniformity guard rightly refuses) to 1.002-1.020, with the width error improving
    from +0.63 % / +0.039 % to +0.053 % / +0.0033 %. Every grid carried at a physical optical
    frequency is bit-for-bit unchanged (14 families checked): there the comb is already exact and
    the two binnings agree."""
    u = np.asarray(u_vals)[mask]
    if u.size == 0:
        return np.array([0.0]), np.array([0.0]), np.array([0.0])
    umin, umax = float(np.min(u)), float(np.max(u))
    w = np.asarray(weights)[mask]
    if umax <= umin:
        return (np.array([umin]), np.array([float(np.sum(np.asarray(P)[mask]))]),
                np.array([float(np.sum(w))]))
    edges = np.linspace(umin, umax, int(nbin) + 1)
    uniq = np.unique(u)
    breaks = np.flatnonzero(np.diff(uniq) > _ROTATED_COMB_REL_TOL * (umax - umin))
    if 1 <= breaks.size <= int(nbin) - 1:               # a comb of (breaks + 1) <= nbin lines
        lo = np.concatenate(([0], breaks + 1))
        hi = np.concatenate((breaks + 1, [uniq.size]))
        lines = 0.5 * (uniq[lo] + uniq[hi - 1])         # centre of each cluster
        mids = 0.5 * (lines[:-1] + lines[1:])
        edges = np.concatenate(([2.0 * lines[0] - mids[0]], mids,
                                [2.0 * lines[-1] - mids[-1]]))
    num, _ = np.histogram(u, bins=edges, weights=(np.asarray(P) * weights)[mask])
    den, _ = np.histogram(u, bins=edges, weights=w)
    moment, _ = np.histogram(u, bins=edges, weights=w * u)
    keep = den > 0.0
    return moment[keep] / den[keep], num[keep], den[keep]


def heralded_bandwidths(jsa_matrix: np.ndarray, omega_s_grid: np.ndarray,
                        omega_i_grid: np.ndarray) -> dict:
    """Marginal / rotated bandwidths of the pair. Integrating |f|^2 over the idler gives the
    signal single-count spectrum (its FWHM is the heralded signal bandwidth), and vice versa.

    The two physically meaningful widths are along the ROTATED coordinates:
      * the ANTI-DIAGONAL coordinate omega_s + omega_i, on which the pump envelope
        alpha(omega_s + omega_i) lives -> its FWHM is the PUMP bandwidth;
      * the DIAGONAL coordinate omega_s - omega_i, on which the phase-matching sinc varies
        -> its FWHM is the PHASE-MATCHING bandwidth.

    Any width that the sampled window does not RESOLVE -- the marginal never falls to half its
    maximum on one side, so the "width" is the window's, not the physics' -- is returned as
    ``nan`` with a ``RuntimeWarning``, and its ``*_resolved`` companion key is False. Each
    resolved width also comes with its ``*_resolved`` flag set True.

    RETURN CONTRACT. The four ``*_bandwidth`` keys and their two geometric aliases
    (``antidiagonal_bandwidth`` = pump, ``diagonal_bandwidth`` = phase matching) are FLOATS, and
    are ``nan`` exactly when the corresponding ``*_resolved`` flag (present for the four base
    keys only) is False -- there is no other nan and no other sentinel. ``signal_marginal`` and
    ``idler_marginal`` are 1-D arrays on ``omega_s_grid`` / ``omega_i_grid``. The two ROTATED
    marginals are different: ``pump_marginal`` and ``phase_matching_marginal`` are TUPLES
    ``(abscissa, marginal)`` of two equal-length 1-D arrays, because the rotated coordinate has
    no pre-existing axis to hang on -- the abscissa is the binned u (respectively v) axis, and
    it is NOT ``omega_s_grid``, is generally shorter than it, and is unequally spaced when bins
    are dropped. Both marginal tuples are returned in full even when the corresponding width is
    nan (they are what a caller inspects to see WHY), so a nan width never implies a missing or
    empty marginal.

    ---------------------------------------------------------------------------------------
    HOW THE ROTATED MARGINALS ARE TAKEN (audit Q-13)
    ---------------------------------------------------------------------------------------
    Two things were wrong with the plain ``np.histogram(omega_s +/- omega_i, weights=|f|^2)``
    this used to do, and both are now fixed:

    (i) NO JACOBIAN. A bare histogram sums SAMPLES, which is proportional to
        ``integral |f|^2 dv`` only if every grid cell has the same area. On a non-uniform
        (omega_s, omega_i) grid it is not: measured on a power-1.6 stretched grid the reported
        pump bandwidth was 62 % low. Every cell now carries its trapezoid weight
        ``dw_s dw_i`` (:func:`_cell_weights`), the same quadrature the signal / idler marginals
        already used, which brings that case to 2.5 %.

    (ii) A TENT-SHAPED DOMAIN. On a rectangular (omega_s, omega_i) grid the strip of constant
        ``u = omega_s + omega_i`` has a LENGTH that varies with u (a tent), so the integral over
        it mixes the physics with the shape of the box. With ``Phi == 1`` -- no phase-matching
        structure at all -- the old code reported a phase-matching bandwidth of 19.1 on a
        ``[-5, 5]^2`` grid: the tent of the grid, nothing else, and a 3.5 %-low pump bandwidth
        from the same cause. Dividing by the per-bin cell count (a strip AVERAGE) does not fix
        it -- it swaps the tent for its reciprocal and is WORSE on physical inputs (measured
        +10.7 % and +34 % where the plain sum was +1.6 % and -0.5 %). Instead both marginals are
        taken over the largest window in which EVERY u-strip has the SAME v-extent: the
        axis-aligned rectangle of half-extent ``H = min(half-span omega_s, half-span omega_i)``
        in both u and v, which is inscribed in the rotated domain (its corners satisfy
        ``|u'| + |v'| <= 2H <= 2 R_s, 2 R_i``). The tent then cancels identically, and the
        remaining bias is pure binning. Measured against the closed-form widths of separable
        Gaussian test JSAs: <= 0.9 % on the ``Phi == 1``, narrow, wide and physically-offset
        cases and 2.5 % on the stretched grid, vs up to 62 % before.

    Binning: one bin per ``max(n_s, n_i)`` over that inscribed window, which on a uniform grid is
    one bin per distinct value of the (quantized) rotated coordinate. Empty bins are dropped, and
    every kept bin is reported at the quadrature-measure CENTROID of the coordinate inside it
    rather than at the geometric bin centre -- see :func:`_rotated_marginal` for why the centre
    carried a systematic ``-1/nbin`` compression plus an N-parity saw (+0.83 % at N = 400 against
    -0.25 % at N = 401), and the ~100x improvement the centroid buys.

    ---------------------------------------------------------------------------------------
    GRID-UNIFORMITY GUARD (audit Q-13 residual)
    ---------------------------------------------------------------------------------------
    The per-cell trapezoid weights make the marginal an integral, but they do NOT make the
    estimator convergent on an arbitrary grid, and the failure was silent: on a log-spaced grid
    the phase-matching width came back 30 % low and got WORSE under refinement (-29.8 % at
    N = 401 -> -30.2 % at 801 -> -30.7 % at 1601), on a randomly-spaced grid up to -82 %, with
    every ``*_resolved`` flag True.

    The detector is the estimator's own quadrature measure. Over the inscribed constant-v-extent
    window every u-bin covers the same area of the (omega_s, omega_i) plane, so the per-bin
    measure ``den`` (:func:`_rotated_marginal`'s third return) is CONSTANT exactly when the
    un-normalised per-bin sum is the marginal. When ``max(den)/min(den)`` on either rotated axis
    exceeds ``_ROTATED_MEASURE_MAX_RATIO`` -- OR the occupied bins fall below
    ``_ROTATED_MIN_BIN_FRACTION`` of the requested ones, the separate signature of a rotated
    coordinate that floating point has aliased away -- both rotated widths are returned as
    ``nan`` with a ``RuntimeWarning`` and their ``*_resolved`` flags are False. Calibration
    (separable Gaussian JSA whose two rotated widths are known in closed form; |error| is the
    worse leg):

        grid family                                    max(den)/min(den)   |error|
        uniform, equal spans, N = 200..801                 1.003-1.010     < 0.02 %
        uniform, equal spans, N = 50..101                  1.003-1.042     < 0.42 %
        uniform, unequal N (61x61 .. 512x512, 56 pairs)    1.010-1.980     < 0.2 %, except
                                                                           137x201 at -5.5 %
        uniform, unequal spans r = 0.25..1 (186 grids)     1.003-1.988     < 1 % on 176 of them,
                                                                           worst -8.4 % (N = 101)
        power-1.6 stretch, 401                             1.208           +1.0 % / -6.3 %
        power-2.2 stretch, 401                             1.485           -1.0 % / -4.9 %
        ------------------------------- threshold 2.1 -----------------------------------------
        log-spaced, N = 201..1601                          2.233-2.293     up to -32 %
        log-spaced, harsher                               13.02            -74 %
        random-spaced, 401, 8 seeds                        2.274-3.408     -33 % to -98 %

    and the bin-collapse leg, which is a different failure and needs its own statistic (the few
    bins that survive can have a perfectly uniform measure -- 1.39 in the case below):

        grid family                                   occupied / requested bins   |error|
        every family in the table above                    0.98 - 1.00           (as above)
        span 5 rad/s on a 1.2e15 rad/s carrier,
          N = 100 / 400 / 801                              0.110 / 0.028 / 0.014  47 % - 100 %

    The guard is applied to BOTH rotated widths when EITHER axis trips: the two marginals come
    from one cell partition and one window, so the defect is a property of the grid, not of a
    leg. The signal / idler marginals are unaffected -- they are ordinary trapezoid integrals on
    the supplied axes and were never the subject of Q-13.

    THIS IS A NECESSARY SCREEN, NOT AN ACCURACY CERTIFICATE, and the table above already shows
    both leaks. It is calibrated on ONE measured failure class -- the non-convergent error of
    strongly irregular spacing -- and passing it does not mean the width is good:

      * a smooth ``sinh`` stretch measures 1.24-1.44, BELOW power-2.2's 1.485, and is 3-30 % low
        at every refinement level tested (401 / 801 / 1601), i.e. non-convergent and undetected;
      * a perfectly UNIFORM 137x201 grid is -5.5 % low at a measure ratio of 1.078, and a uniform
        N = 101 grid with span ratio 0.925 is -8.4 % low at 1.068. Those are not grid-uniformity
        failures at all: they are the inscribed window landing incommensurately on two axes whose
        spacings do not divide, and no statistic of ``den`` sees them.

    The requirement this module states is a grid UNIFORM in each of omega_s and omega_i, with
    enough points (N >= 200 measured < 0.02 %) and comparable spans. The guard catches the loud
    violations of it; the power-law band is the tested tolerance, not a licence for arbitrary
    spacing; and a width that matters should be checked by refining the grid.

    PRECONDITION on the DIAGONAL / ANTI-DIAGONAL LABELLING. Calling ``omega_s - omega_i`` "the"
    phase-matching coordinate assumes the JSA's phase-matching lobe is aligned with the
    anti-diagonal, i.e. SYMMETRIC GROUP VELOCITIES: ``dk`` linear in (omega_s, omega_i) with
    equal and opposite slopes, ``d(dk)/d omega_s = -d(dk)/d omega_i``. For general dispersion the
    JSA is tilted at some other angle and the two marginals MIX -- both reported widths are then
    projections of a tilted ellipse, not the pump and phase-matching widths. The Schmidt number
    (:func:`schmidt_number`) is the rotation-independent statement and should be preferred when
    the group velocities are not symmetric."""
    ws = np.asarray(omega_s_grid, dtype=float)
    wi = np.asarray(omega_i_grid, dtype=float)
    P = jsi(jsa_matrix)
    sig = _trapz(P, wi, axis=1)
    idl = _trapz(P, ws, axis=0)
    WS, WI = np.meshgrid(ws, wi, indexing="ij")
    U, V = WS + WI, WS - WI

    # Inscribed constant-v-extent window (see (ii) above). Half-extent H in BOTH rotated axes.
    s_c, s_r = 0.5 * (ws.min() + ws.max()), 0.5 * (ws.max() - ws.min())
    i_c, i_r = 0.5 * (wi.min() + wi.max()), 0.5 * (wi.max() - wi.min())
    H = min(s_r, i_r)
    tol = 1.0 + 1e-12                                   # keep the boundary cells, not fp noise
    mask = (np.abs(U - (s_c + i_c)) <= H * tol) & (np.abs(V - (s_c - i_c)) <= H * tol)

    weights = _cell_weights(ws, wi)
    nbin = max(4, max(ws.size, wi.size))
    u_ax, u_marg, u_den = _rotated_marginal(P, U, weights, mask, nbin)   # pump coordinate
    v_ax, v_marg, v_den = _rotated_marginal(P, V, weights, mask, nbin)   # phase-matching coord.

    # GRID-UNIFORMITY GUARD (see the docstring). One number for both rotated legs: they share the
    # cell partition and the window, so a non-constant quadrature measure on either axis condemns
    # both.
    measure_ratio = max(_measure_ratio(u_den), _measure_ratio(v_den))
    # ... and the second failure it cannot see: a rotated coordinate whose bins have COLLAPSED
    # because omega_s + omega_i cannot resolve the grid spacing in floating point. There the
    # measure can look perfectly uniform across the few bins that survive.
    bin_fraction = min(u_den.size, v_den.size) / float(nbin)
    grid_ok = bool(measure_ratio <= _ROTATED_MEASURE_MAX_RATIO
                   and bin_fraction >= _ROTATED_MIN_BIN_FRACTION)

    widths = {
        "signal_bandwidth": _fwhm_resolved(ws, sig),
        "idler_bandwidth": _fwhm_resolved(wi, idl),
        "pump_bandwidth": _fwhm_resolved(u_ax, u_marg),
        "phase_matching_bandwidth": _fwhm_resolved(v_ax, v_marg),
    }
    _ROTATED = ("pump_bandwidth", "phase_matching_bandwidth")
    out = {}
    for key, (val, resolved) in widths.items():
        # The grid defect is reported FIRST when both apply: an unresolved window is a statement
        # about how far the profile was sampled, but a non-constant quadrature measure means the
        # profile itself is not the marginal, which is the more fundamental cause to act on.
        if key in _ROTATED and not grid_ok:
            resolved = False
            warnings.warn(
                "heralded_bandwidths: '{}' is NOT trustworthy on this grid -- the rotated "
                "marginals require a grid on which every bin of omega_s +/- omega_i carries the "
                "SAME quadrature measure, i.e. a grid UNIFORMLY spaced in each of omega_s and "
                "omega_i, and spanning enough of the carrier that the sum resolves that spacing "
                "in floating point. Measured here: per-bin measure spread {:.3f} (threshold "
                "{:g}), and {:.3f} of the requested bins occupied (threshold {:g}). The first "
                "means the reported profile is modulated by the sampling density rather than by "
                "the joint spectrum -- on log-spaced and randomly-spaced grids a 30 % to 82 % "
                "error that does NOT shrink under refinement; the second means the rotated "
                "coordinate has aliased onto a handful of values and the profile is not the "
                "physics at all. Returning nan (the window-limited number was {:.6g}). Resample "
                "the JSA onto uniform omega_s / omega_i axes; the signal / idler widths are "
                "unaffected. See audit Q-13.".format(
                    key, measure_ratio, _ROTATED_MEASURE_MAX_RATIO, bin_fraction,
                    _ROTATED_MIN_BIN_FRACTION, val),
                RuntimeWarning, stacklevel=2)
        elif not resolved:
            warnings.warn(
                "heralded_bandwidths: '{}' is NOT resolved by this grid -- the marginal is still "
                "at or above half its maximum at the edge of the sampled window, so its FWHM is "
                "a property of the window ({:.6g} wide), not of the pair. Returning nan. Widen "
                "the omega grids (or, for the rotated widths, make them span enough that the "
                "joint spectrum decays inside the inscribed window). See audit "
                "Q-13.".format(key, val), RuntimeWarning, stacklevel=2)
        out[key] = float(val) if resolved else float("nan")
        out[key + "_resolved"] = resolved
    # Historical aliases: the rotated widths under their geometric names.
    out["antidiagonal_bandwidth"] = out["pump_bandwidth"]
    out["diagonal_bandwidth"] = out["phase_matching_bandwidth"]
    out["signal_marginal"] = sig
    out["idler_marginal"] = idl
    out["pump_marginal"] = (u_ax, u_marg)
    out["phase_matching_marginal"] = (v_ax, v_marg)
    return out
