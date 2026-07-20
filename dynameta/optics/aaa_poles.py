"""Rational (AAA) pole extraction of resonances from REAL-frequency response samples.

Roadmap item 5.5.  Instead of forcing a complex frequency through an external solver, this
module fits an ``AAA`` barycentric rational approximant (Nakatsukasa, Sete & Trefethen, SIAM J.
Sci. Comput. 40, A1494 (2018)) to REAL-frequency response samples produced by ANY solver -- the
lumenairy RCWA bridge, the FEM, the TMM reference, or even a MEASURED spectrum -- and reads the
resonance poles / residues / zeros straight off the approximant.  This is the established
quasi-normal-mode (QNM) extraction route, with NO upstream changes and no complex-frequency
capability required from the data source.

Contents
--------
* :func:`aaa` -- the AAA algorithm, implemented faithfully after Nakatsukasa-Sete-Trefethen:
  greedy support-point selection (residual argmax), Loewner-matrix SVD for the barycentric
  weights, and pole / residue / zero computation from the barycentric form via the arrowhead
  generalized eigenvalue problem.  Returns an :class:`AAAResult` (support points, values, weights,
  poles, residues, zeros; callable for evaluation).
* :func:`q_from_pole` -- quality factor of a complex pole, IDENTICAL to
  :func:`dynameta.optics.resonance.pole_q` (``Q = |Re(w)| / (2|Im(w)|)``).
* :func:`find_resonances` -- physical resonance extraction from a (real-omega, response) sweep
  with SPURIOUS-POLE (Froissart-doublet) filtering: residue-magnitude threshold, stability under
  sample decimation, and the lower-half-plane physicality filter.  Returns
  ``[Resonance(omega_tilde, Q, residue), ...]``.
* :func:`sweep_and_extract` -- convenience driver: adaptively samples a callable
  ``solver(omega_real) -> response`` across a band (refining where the response has structure)
  and extracts the resonances.

-------------------------------------------------------------------------------------------------
PHYSICALITY / FILTERING RULE (why the poles land where they do)
-------------------------------------------------------------------------------------------------
Under this library's ``exp(-i*omega*t)`` time convention a causal, passive scattering response
``f(omega)`` is ANALYTIC in the UPPER half of the complex-omega plane; all of its singularities --
the resonances / QNMs -- lie in the LOWER half plane at ``omega_tilde = omega_0 - i*gamma/2``
(``Im < 0``, a DECAYING mode), with ``Q = omega_0 / (2|Im(omega_tilde)|)`` (matching
``optics.resonance``).

A barycentric rational approximant is a RATIONAL function, and a generic rational has poles in
BOTH half planes.  Which of AAA's poles are physical depends on the data:

  * COMPLEX analytic data (e.g. the complex transmission amplitude ``t(omega)`` sampled on the
    real axis).  The true function is analytic in the upper half plane, so when the data is clean
    AAA places the physical poles directly in the LOWER half plane and puts nothing physical in
    the upper half plane.  (Verified empirically -- see the tests: the Fabry-Perot poles come out
    at ``Im < 0`` with no upper-half counterpart.)

  * REAL-VALUED data (e.g. a transmittance / reflectance ``T(omega) = |t|^2``, or a measured
    intensity).  Real samples force real barycentric weights, hence an approximant with
    conjugate-symmetric poles: every physical pole ``omega_0 - i*gamma/2`` appears TOGETHER with
    its unphysical mirror ``omega_0 + i*gamma/2``.  Real-axis data ALONE cannot distinguish a pole
    from its conjugate mirror; the ``exp(-i*omega*t)`` causality of the underlying passive system
    is the external fact that selects ``Im < 0`` as physical.

The SELECTION / FILTERING rule is therefore the SAME in both cases: keep poles with
``Im(omega_tilde) < 0`` (decaying); for real data this automatically discards the upper-half
conjugate mirror, and for complex analytic data it is a no-op that simply confirms the clean data
placed every physical pole correctly.  On top of physicality, :func:`find_resonances` rejects
spurious FROISSART DOUBLETS (a pole with a near-coincident zero, the signature of over-fitting or
noise) by (i) a residue-magnitude floor -- a Froissart pole nearly cancels against its partner
zero, so its residue is tiny -- and (ii) a stability test: a genuine resonance pole barely moves
when the sample set is decimated, whereas a Froissart / noise-driven pole jumps.

References
----------
* Y. Nakatsukasa, O. Sete, L. N. Trefethen, "The AAA algorithm for rational approximation",
  SIAM J. Sci. Comput. 40, A1494 (2018).  (Greedy support selection, Loewner-SVD weights,
  arrowhead generalized-eigenvalue poles / zeros, Froissart-doublet discussion.)
* P. Lalanne, W. Yan, K. Vynck, C. Sauvan, J.-P. Hugonin, "Light Interaction with Photonic and
  Plasmonic Resonances", Laser Photonics Rev. 12, 1700113 (2018).  (QNM poles, ``Q``, the
  ``exp(-i*omega*t)`` lower-half-plane sign.)
* J. Gilles, ... (Froissart, "Approximation de Pade", 1969 -- the doublet phenomenon named here).

Conventions: SI units, ``exp(-i*omega*t)`` (physical decaying poles have ``Im < 0``), pure
numpy/scipy, ASCII-only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, List, NamedTuple, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "AAAResult",
    "Resonance",
    "SweepResult",
    "aaa",
    "q_from_pole",
    "find_resonances",
    "sweep_and_extract",
]


# ------------------------------------------------------------------------------------------------
# Quality factor (identical to optics.resonance.pole_q)
# ------------------------------------------------------------------------------------------------
def q_from_pole(omega_tilde) -> float:
    """Quality factor of a complex pole ``omega_tilde = omega_0 - i*gamma/2``:

        Q = |Re(omega_tilde)| / (2 |Im(omega_tilde)|).

    Returns ``+inf`` for a real (lossless / undamped) pole.  This is byte-identical to
    :func:`dynameta.optics.resonance.pole_q` -- the single Q convention across the resonance
    tooling."""
    w = complex(omega_tilde)
    im = abs(w.imag)
    if im == 0.0:
        return float("inf")
    return abs(w.real) / (2.0 * im)


# ------------------------------------------------------------------------------------------------
# AAA result container
# ------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class AAAResult:
    """A barycentric-rational approximant produced by :func:`aaa`, callable for evaluation.

    The rational is
        ``r(z) = [sum_j w_j f_j / (z - z_j)] / [sum_j w_j / (z - z_j)]``
    over the support points ``z_j`` (``support_points``) with values ``f_j`` (``support_values``)
    and weights ``w_j`` (``weights``).  ``poles`` / ``zeros`` are the finite eigenvalues of the
    arrowhead generalized eigenproblem of the denominator / numerator; ``residues[k]`` is the
    residue of ``r`` at ``poles[k]``.  ``errors`` is the greedy convergence history (max
    ``|f - r|`` over the sample set at each degree)."""

    support_points: np.ndarray
    support_values: np.ndarray
    weights: np.ndarray
    poles: np.ndarray
    residues: np.ndarray
    zeros: np.ndarray
    errors: np.ndarray
    max_error: float
    real_data: bool = False

    @property
    def degree(self) -> int:
        """Degree of the approximant = (number of support points) - 1."""
        return int(self.support_points.size) - 1

    def __call__(self, z):
        """Evaluate the barycentric rational at ``z`` (scalar or array).  At a support point the
        value is the exact interpolated ``f_j`` (the 0/0 barycentric limit is handled)."""
        return _bary_eval(np.asarray(z, dtype=np.complex128), self.support_points,
                          self.support_values, self.weights)


class Resonance(NamedTuple):
    """One extracted resonance.  ``omega_tilde`` is the complex pole (``Im < 0``, decaying under
    ``exp(-i*omega*t)``), ``Q`` its quality factor (:func:`q_from_pole`), ``residue`` the
    approximant's residue there (a coupling-strength proxy)."""

    omega_tilde: complex
    Q: float
    residue: complex


@dataclass(frozen=True)
class SweepResult:
    """Result of :func:`sweep_and_extract`: the extracted ``resonances`` plus the adaptive sample
    grid actually used (``omega`` and the corresponding ``response`` values, sorted)."""

    resonances: List[Resonance]
    omega: np.ndarray
    response: np.ndarray
    approximant: AAAResult = field(repr=False)


# ------------------------------------------------------------------------------------------------
# Barycentric evaluation
# ------------------------------------------------------------------------------------------------
def _bary_eval(z: np.ndarray, zj: np.ndarray, fj: np.ndarray, wj: np.ndarray) -> np.ndarray:
    """Barycentric rational ``r(z)`` with exact handling of ``z`` hitting a support point.
    Vectorized over ``z``.  Returns a complex array shaped like ``z`` (scalar in -> 0-d array)."""
    z = np.asarray(z, dtype=np.complex128)
    shape = z.shape
    zf = z.ravel()
    with np.errstate(divide="ignore", invalid="ignore"):
        diff = zf[:, None] - zj[None, :]                       # (Nz, m)
        C = 1.0 / diff                                         # Cauchy matrix (inf where coincident)
        num = C @ (wj * fj)
        den = C @ wj
        out = num / den
    # Fix up any z exactly on a support point (0/0 -> the interpolated value f_j).
    hit_rows, hit_cols = np.nonzero(~np.isfinite(diff) | (diff == 0.0))
    for r_i, c_j in zip(hit_rows, hit_cols):
        out[r_i] = fj[c_j]
    return out.reshape(shape)


# ------------------------------------------------------------------------------------------------
# Poles / residues / zeros from the barycentric form (arrowhead generalized eigenproblem)
# ------------------------------------------------------------------------------------------------
def _poles(zj: np.ndarray, wj: np.ndarray) -> np.ndarray:
    """Finite poles of the barycentric rational = zeros of the denominator ``sum_j w_j/(z-z_j)``,
    as the finite eigenvalues of the (m+1) arrowhead generalized eigenproblem (Nakatsukasa et al.
    2018, eq. for poles): ``E x = lam B x`` with

        E = [[0, w^T], [1, diag(z_j)]],   B = diag(0, 1, ..., 1).

    Two of the ``m+1`` eigenvalues are infinite (dropped); the remaining ``m-1`` are the poles."""
    from scipy.linalg import eig
    m = zj.size
    if m < 2:
        return np.array([], dtype=np.complex128)
    B = np.eye(m + 1, dtype=np.complex128)
    B[0, 0] = 0.0
    E = np.zeros((m + 1, m + 1), dtype=np.complex128)
    E[0, 1:] = wj
    E[1:, 0] = 1.0
    E[1:, 1:] = np.diag(zj)
    lam = eig(E, B, right=False)
    return lam[np.isfinite(lam)]


def _zeros(zj: np.ndarray, fj: np.ndarray, wj: np.ndarray) -> np.ndarray:
    """Finite zeros of the barycentric rational = zeros of the numerator ``sum_j w_j f_j/(z-z_j)``,
    the same arrowhead eigenproblem with ``w_j f_j`` in the top row (Nakatsukasa et al. 2018)."""
    from scipy.linalg import eig
    m = zj.size
    if m < 2:
        return np.array([], dtype=np.complex128)
    B = np.eye(m + 1, dtype=np.complex128)
    B[0, 0] = 0.0
    E = np.zeros((m + 1, m + 1), dtype=np.complex128)
    E[0, 1:] = wj * fj
    E[1:, 0] = 1.0
    E[1:, 1:] = np.diag(zj)
    lam = eig(E, B, right=False)
    return lam[np.isfinite(lam)]


def _residues(poles: np.ndarray, zj: np.ndarray, fj: np.ndarray, wj: np.ndarray) -> np.ndarray:
    """Residue of ``r = N/D`` at each (simple) pole ``p``: ``res = N(p) / D'(p)`` with
    ``N(z) = sum_j w_j f_j/(z-z_j)``, ``D(z) = sum_j w_j/(z-z_j)``, ``D'(z) = -sum_j w_j/(z-z_j)^2``.
    """
    res = np.empty(poles.size, dtype=np.complex128)
    for k, p in enumerate(poles):
        d = p - zj
        with np.errstate(divide="ignore", invalid="ignore"):
            N = np.sum(wj * fj / d)
            Dp = -np.sum(wj / (d * d))
        res[k] = N / Dp if (np.isfinite(Dp) and Dp != 0.0) else np.nan
    return res


# ------------------------------------------------------------------------------------------------
# The AAA algorithm
# ------------------------------------------------------------------------------------------------
def aaa(z_samples: Sequence[complex], f_samples: Sequence[complex], *, tol: float = 1e-13,
        max_degree: int = 100) -> AAAResult:
    """AAA rational approximation of ``f`` sampled at ``z`` (Nakatsukasa, Sete & Trefethen 2018).

    Greedy loop: start from the mean of ``f``; at each step add the sample where the current
    approximant has the LARGEST residual as a new support point; solve the Loewner-matrix
    least-squares problem (SVD, smallest right singular vector) for the barycentric weights; stop
    when ``max|f - r| <= tol * max|f|`` or ``max_degree`` support points have been added.

    Parameters
    ----------
    z_samples : sequence of complex
        Sample abscissae (for a real-frequency sweep these are the real ``omega`` values).
    f_samples : sequence of complex or real
        Response values at ``z_samples``.  Real input is detected and flagged on the result
        (``AAAResult.real_data``) -- its poles are conjugate-symmetric (see the module docstring).
    tol : float
        Relative convergence tolerance on ``max|f|``.  Pass ``0.0`` to force fitting all the way to
        ``max_degree`` (used to DEMONSTRATE Froissart doublets on clean data).
    max_degree : int
        Maximum number of support points minus one; capped internally at ``len(z)-1`` (and at
        roughly ``len(z)//2`` so the Loewner system stays overdetermined).

    Returns
    -------
    AAAResult
    """
    z = np.asarray(z_samples, dtype=np.complex128).ravel()
    f_in = np.asarray(f_samples).ravel()
    if z.shape != f_in.shape:
        raise ValueError("z_samples and f_samples must have equal length")
    M = z.size
    if M < 3:
        raise ValueError("need >= 3 samples for AAA")
    finite = np.isfinite(z) & np.isfinite(f_in.astype(np.complex128))
    z, f_in = z[finite], f_in[finite]
    M = z.size
    if M < 3:
        raise ValueError("need >= 3 finite samples for AAA")

    real_data = not np.iscomplexobj(f_in) or (
        float(np.max(np.abs(f_in.imag))) <= 1e-12 * (float(np.max(np.abs(f_in))) + 1e-300))
    f = f_in.astype(np.complex128)

    # Normalize the abscissa to O(1) for a well-conditioned pole eigenproblem.  The barycentric
    # weights are invariant under this affine reparameterization (the shift/scale cancels between
    # numerator and denominator), so the SAME weights serve the original-coordinate barycentric
    # form; only the poles/zeros/residues are mapped back at the end.
    z0 = complex(np.mean(z))
    scale = 0.5 * (float(np.max(z.real)) - float(np.min(z.real)))
    if scale <= 0.0:
        scale = float(np.max(np.abs(z - z0))) or 1.0
    u = (z - z0) / scale

    fmax = float(np.max(np.abs(f))) or 1.0
    hard_cap = min(int(max_degree), M - 1, max(1, M // 2))

    J = np.ones(M, dtype=bool)                                 # mask of non-support samples
    zj_idx: List[int] = []
    R = np.full(M, np.mean(f), dtype=np.complex128)            # current approximant on the grid
    errors: List[float] = []
    wj = np.array([1.0], dtype=np.complex128)

    for _ in range(hard_cap):
        # Greedy: new support point = sample of largest residual among the remaining ones.
        resid = np.abs(f - R)
        resid_masked = np.where(J, resid, -np.inf)
        j = int(np.argmax(resid_masked))
        J[j] = False
        zj_idx.append(j)

        sup = np.array(zj_idx, dtype=int)
        uj = u[sup]
        fj = f[sup]
        rest = np.nonzero(J)[0]
        if rest.size == 0:
            wj = np.ones(sup.size, dtype=np.complex128) / math.sqrt(sup.size)
            R = f.copy()
            errors.append(0.0)
            break

        # Loewner matrix A[i,k] = (f_rest_i - f_sup_k) / (u_rest_i - u_sup_k).
        with np.errstate(divide="ignore", invalid="ignore"):
            Cauchy = 1.0 / (u[rest][:, None] - uj[None, :])
        A = (f[rest][:, None] - fj[None, :]) * Cauchy
        # Weights = right singular vector of the smallest singular value.
        _, _, Vh = np.linalg.svd(A, full_matrices=False)
        wj = np.conj(Vh[-1, :])

        # Evaluate r on the whole grid: support points interpolate exactly; the rest via bary.
        num = Cauchy @ (wj * fj)
        den = Cauchy @ wj
        R = f.copy()
        R[rest] = num / den

        err = float(np.max(np.abs(f - R)))
        errors.append(err)
        if err <= tol * fmax:
            break

    sup = np.array(zj_idx, dtype=int)
    uj = u[sup]
    fj = f[sup]
    zpts = z[sup]

    # Poles / zeros / residues in normalized coords, mapped back to the original abscissa.
    poles_u = _poles(uj, wj)
    zeros_u = _zeros(uj, fj, wj)
    res_u = _residues(poles_u, uj, fj, wj)
    poles = z0 + scale * poles_u
    zeros = z0 + scale * zeros_u
    residues = scale * res_u

    return AAAResult(
        support_points=zpts,
        support_values=fj,
        weights=wj,
        poles=poles,
        residues=residues,
        zeros=zeros,
        errors=np.asarray(errors, dtype=np.float64),
        max_error=float(errors[-1]) if errors else float("nan"),
        real_data=bool(real_data),
    )


# ------------------------------------------------------------------------------------------------
# Physical resonance extraction with Froissart-doublet / spurious-pole filtering
# ------------------------------------------------------------------------------------------------
def _in_band(re: float, lo: float, hi: float, margin: float) -> bool:
    span = hi - lo
    return (lo - margin * span) <= re <= (hi + margin * span)


def _physical_candidates(result: AAAResult, lo: float, hi: float, band_margin: float,
                         im_atol: float) -> List[Tuple[complex, complex]]:
    """(pole, residue) pairs that pass the physicality + in-band gate: ``Im < 0`` (decaying,
    ``exp(-i*omega*t)``), ``Re > 0``, and ``Re`` inside the swept band (+/- a margin).  Deduplicated
    (conjugate mirrors of real-data poles are auto-dropped by the ``Im < 0`` cut)."""
    out: List[Tuple[complex, complex]] = []
    for p, r in zip(result.poles, result.residues):
        p = complex(p)
        if not (np.isfinite(p.real) and np.isfinite(p.imag)):
            continue
        if p.imag >= -im_atol:                                 # keep strictly lower half (decaying)
            continue
        if p.real <= 0.0:
            continue
        if not _in_band(p.real, lo, hi, band_margin):
            continue
        if not np.isfinite(r):
            continue
        # dedup
        if any(abs(p - q) <= 1e-6 * max(abs(p), 1.0) for q, _ in out):
            continue
        out.append((p, complex(r)))
    return out


def find_resonances(omega_real: Sequence[float], response, *, residue_floor: Optional[float] = None,
                    stability_check: bool = True, tol: float = 1e-11, max_degree: int = 100,
                    band: Optional[Tuple[float, float]] = None, band_margin: float = 0.02,
                    stability_rtol: float = 5e-3, residue_rel_floor: float = 1e-3,
                    im_atol_rel: float = 1e-9) -> List[Resonance]:
    """Extract physical resonances from a real-frequency sweep with spurious-pole filtering.

    Fits :func:`aaa` to ``(omega_real, response)``, then keeps only poles that pass ALL of:

      (i)   PHYSICALITY -- ``Im(pole) < 0`` (decaying under ``exp(-i*omega*t)``) and ``Re > 0``,
            inside the swept band.  For real-valued ``response`` this discards the unphysical
            upper-half conjugate mirror of each pole (see the module docstring); for complex
            analytic ``response`` it merely confirms the clean placement.
      (ii)  RESIDUE FLOOR -- ``|residue| >= residue_floor`` (absolute) or, if ``residue_floor`` is
            ``None``, ``|residue| >= residue_rel_floor * max|residue|`` over the candidates.  A
            Froissart doublet's pole nearly cancels its partner zero, so its residue is tiny.
      (iii) STABILITY (if ``stability_check``) -- the pole must persist, to within
            ``stability_rtol * |pole|``, when the sample set is DECIMATED (every other sample).  A
            genuine resonance barely moves; a Froissart / noise-driven pole jumps.

    Parameters
    ----------
    omega_real : sequence of float
        Real angular-frequency samples [rad/s] (need not be sorted / uniform).
    response : sequence of complex or real
        The measured / computed response at each ``omega`` (complex amplitude such as ``t``, or a
        real intensity such as transmittance).
    residue_floor : float, optional
        Absolute residue-magnitude threshold.  ``None`` -> relative (``residue_rel_floor``).
    stability_check : bool
        Enable the decimation-stability filter (default on).
    tol, max_degree : float, int
        Passed to :func:`aaa`.  A high ``max_degree`` with ``tol=0`` over-fits and MANUFACTURES
        Froissart doublets -- which the filter then removes (the Froissart demo gate).
    band : (float, float), optional
        Physical band ``(omega_lo, omega_hi)``; defaults to the sampled range.

    Returns
    -------
    list of Resonance
        Sorted by ascending ``Re(omega_tilde)``.
    """
    w = np.asarray(omega_real, dtype=np.float64).ravel()
    f = np.asarray(response).ravel()
    if w.shape != f.shape:
        raise ValueError("omega_real and response must have equal length")
    order = np.argsort(w)
    w, f = w[order], f[order]

    lo = float(w.min()) if band is None else float(band[0])
    hi = float(w.max()) if band is None else float(band[1])
    im_atol = im_atol_rel * max(abs(lo), abs(hi), 1.0)

    res = aaa(w.astype(np.complex128), f, tol=tol, max_degree=max_degree)
    cand = _physical_candidates(res, lo, hi, band_margin, im_atol)
    if not cand:
        return []

    # (ii) residue-magnitude floor.
    res_mags = np.array([abs(r) for _, r in cand])
    if residue_floor is None:
        floor = residue_rel_floor * float(np.max(res_mags))
    else:
        floor = float(residue_floor)
    cand = [(p, r) for (p, r), rm in zip(cand, res_mags) if rm >= floor]
    if not cand:
        return []

    # (iii) stability under decimation.
    if stability_check and w.size >= 8:
        res_dec = aaa(w[::2].astype(np.complex128), f[::2], tol=tol,
                      max_degree=max_degree)
        dec = _physical_candidates(res_dec, lo, hi, band_margin, im_atol)
        dec_poles = np.array([p for p, _ in dec], dtype=np.complex128)
        stable = []
        for p, r in cand:
            if dec_poles.size == 0:
                continue
            d = float(np.min(np.abs(dec_poles - p)))
            if d <= stability_rtol * abs(p):
                stable.append((p, r))
        cand = stable

    out = [Resonance(omega_tilde=p, Q=q_from_pole(p), residue=r) for p, r in cand]
    out.sort(key=lambda rr: rr.omega_tilde.real)
    return out


# ------------------------------------------------------------------------------------------------
# Adaptive sweep + extract convenience
# ------------------------------------------------------------------------------------------------
def sweep_and_extract(solver: Callable[[float], complex], omega_min: float, omega_max: float, *,
                      n_initial: int = 65, max_samples: int = 513, n_refine_rounds: int = 5,
                      refine_frac: float = 0.3, tol: float = 1e-11, max_degree: int = 100,
                      residue_floor: Optional[float] = None, stability_check: bool = True,
                      **find_kwargs) -> SweepResult:
    """Adaptively sample ``solver(omega) -> response`` across ``[omega_min, omega_max]`` -- refining
    where the response has the most structure -- and extract the resonances with
    :func:`find_resonances`.

    Refinement: start from a uniform grid of ``n_initial`` points; each round inserts midpoints
    into the intervals whose neighbouring-response jump ``|f_{i+1} - f_i|`` is in the top
    ``refine_frac`` fraction (so points concentrate around sharp resonances), until
    ``n_refine_rounds`` rounds pass or ``max_samples`` is reached.

    Parameters
    ----------
    solver : callable
        ``solver(omega_real) -> complex or real response``.  Called once per sample.
    omega_min, omega_max : float
        Band edges [rad/s].
    n_initial, max_samples, n_refine_rounds, refine_frac : sampling controls.
    tol, max_degree, residue_floor, stability_check, **find_kwargs :
        Forwarded to :func:`find_resonances`.

    Returns
    -------
    SweepResult
    """
    if not (omega_max > omega_min):
        raise ValueError("omega_max must exceed omega_min")
    w = np.linspace(float(omega_min), float(omega_max), int(n_initial))
    f = np.array([complex(solver(float(wi))) for wi in w], dtype=np.complex128)

    for _ in range(int(n_refine_rounds)):
        if w.size >= max_samples:
            break
        jumps = np.abs(np.diff(f))
        if jumps.size == 0 or float(np.max(jumps)) == 0.0:
            break
        thresh = np.quantile(jumps, 1.0 - refine_frac)
        pick = np.nonzero(jumps >= max(thresh, 1e-300))[0]
        budget = max_samples - w.size
        if pick.size > budget:
            pick = pick[np.argsort(jumps[pick])[::-1][:budget]]
        if pick.size == 0:
            break
        mids = 0.5 * (w[pick] + w[pick + 1])
        fm = np.array([complex(solver(float(m))) for m in mids], dtype=np.complex128)
        w = np.concatenate([w, mids])
        f = np.concatenate([f, fm])
        srt = np.argsort(w)
        w, f = w[srt], f[srt]

    # Collapse a purely real response to a real array so find_resonances sees the real-data path.
    if float(np.max(np.abs(f.imag))) <= 1e-12 * (float(np.max(np.abs(f))) + 1e-300):
        f_use = f.real
    else:
        f_use = f
    resonances = find_resonances(w, f_use, residue_floor=residue_floor,
                                 stability_check=stability_check, tol=tol,
                                 max_degree=max_degree, band=(omega_min, omega_max),
                                 **find_kwargs)
    approx = aaa(w.astype(np.complex128), f_use, tol=tol, max_degree=max_degree)
    return SweepResult(resonances=resonances, omega=w, response=f_use, approximant=approx)
