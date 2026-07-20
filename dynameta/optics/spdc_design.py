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
    Normalized to unit Frobenius norm by default."""
    ws = np.asarray(omega_s_grid, dtype=float)
    wi = np.asarray(omega_i_grid, dtype=float)
    WS, WI = np.meshgrid(ws, wi, indexing="ij")
    alpha = np.asarray(pump_envelope(WS + WI), dtype=complex)
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


def _fwhm(x: np.ndarray, y: np.ndarray) -> float:
    """Full width at half maximum of a single-peaked profile y(x) by linear interpolation of
    the half-max crossings. Returns 0 for a degenerate/empty profile."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if y.size < 2 or np.max(y) <= 0:
        return 0.0
    half = 0.5 * np.max(y)
    above = y >= half
    idx = np.where(above)[0]
    if idx.size == 0:
        return 0.0
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
    return abs(xr - xl)


def _rotated_marginal(P: np.ndarray, u_vals: np.ndarray, nbin: int) -> Tuple[np.ndarray, np.ndarray]:
    """Histogram-project the JSI weights ``P`` onto a rotated coordinate ``u_vals`` (both
    flattened to the same shape), returning (bin_centres, marginal). Bin count ``nbin`` set to
    resolve the finer of the two grid axes."""
    umin, umax = float(np.min(u_vals)), float(np.max(u_vals))
    if umax <= umin:
        return np.array([umin]), np.array([float(np.sum(P))])
    edges = np.linspace(umin, umax, nbin + 1)
    hist, _ = np.histogram(u_vals.ravel(), bins=edges, weights=P.ravel())
    centres = 0.5 * (edges[:-1] + edges[1:])
    return centres, hist


def heralded_bandwidths(jsa_matrix: np.ndarray, omega_s_grid: np.ndarray,
                        omega_i_grid: np.ndarray) -> dict:
    """Marginal / rotated bandwidths of the pair. Integrating |f|^2 over the idler gives the
    signal single-count spectrum (its FWHM is the heralded signal bandwidth), and vice versa.

    The two physically meaningful widths are along the ROTATED coordinates:
      * the ANTI-DIAGONAL coordinate omega_s + omega_i, on which the pump envelope
        alpha(omega_s + omega_i) lives -> its FWHM is the PUMP bandwidth;
      * the DIAGONAL coordinate omega_s - omega_i, on which the phase-matching sinc varies
        -> its FWHM is the PHASE-MATCHING bandwidth.
    Computed by histogram-projecting |f|^2 onto omega_s +/- omega_i (robust to unequal grids)."""
    ws = np.asarray(omega_s_grid, dtype=float)
    wi = np.asarray(omega_i_grid, dtype=float)
    P = jsi(jsa_matrix)
    sig = _trapz(P, wi, axis=1)
    idl = _trapz(P, ws, axis=0)
    WS, WI = np.meshgrid(ws, wi, indexing="ij")
    nbin = 2 * max(ws.size, wi.size)
    u_ax, u_marg = _rotated_marginal(P, WS + WI, nbin)     # pump coordinate
    v_ax, v_marg = _rotated_marginal(P, WS - WI, nbin)     # phase-matching coordinate
    return {
        "signal_bandwidth": _fwhm(ws, sig),
        "idler_bandwidth": _fwhm(wi, idl),
        "antidiagonal_bandwidth": _fwhm(u_ax, u_marg),     # == pump bandwidth
        "diagonal_bandwidth": _fwhm(v_ax, v_marg),         # == phase-matching bandwidth
        "pump_bandwidth": _fwhm(u_ax, u_marg),
        "phase_matching_bandwidth": _fwhm(v_ax, v_marg),
        "signal_marginal": sig,
        "idler_marginal": idl,
    }
