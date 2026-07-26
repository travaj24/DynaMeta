"""Three-wave-mixing (chi2) coupled-wave REFERENCE solver -- SFG / DFG / OPA / SHG.

This is the classical, plane-wave, slowly-varying-envelope oracle for every second-order
three-wave process (roadmap item 4.1). It provides:

  * undepleted closed forms -- ``sfg_undepleted`` (sinc^2 vs phase mismatch), ``opa_gain``
    (signal cosh^2 / idler sinh^2 with the below/above-threshold trig<->hyperbolic
    crossover), and ``shg_undepleted`` (the degenerate limit);
  * ``twm_propagate`` -- the DEPLETED three-envelope RK/DOP853 integrator (complex A1, A2,
    A3 vs z) with EXACT Manley-Rowe photon-flux diagnostics and total-power conservation;
  * quasi-phase-matching (QPM): a sign-flipping d_eff(z) of period Lambda and the first-order
    effective (2/pi) d_eff law + ``phase_matching_sinc`` (consumed by ``spdc_design.jsa``).
    The poling square wave carries BOTH first grating orders m = +-1, so one positive period
    Lambda = 2 pi / |dk| phase-matches EITHER sign of dk (``qpm_period_for``'s SIGN
    CONVENTION note); ``phase_matching_sinc`` picks the matching order.

--------------------------------------------------------------------------------------------
SIGN / AMPLITUDE CONVENTION  (DERIVED here for the library-wide exp(-i omega t) convention;
                              the i's are the OPPOSITE of the exp(+i omega t) textbooks)
--------------------------------------------------------------------------------------------
Real field, REAL-PEAK-AMPLITUDE convention (each A_j is the physical peak field, V/m):

    E_j(z, t) = (1/2) A_j(z) exp[ i (k_j z - omega_j t) ] + c.c.  =>  E_j = |A_j| cos(...)

so a pump written E = A0 cos(omega t) has envelope A1 = A0 (NOT A0/2). Second-order
polarization taken SCALAR and instantaneous with the SAME chi2 as the FDTD kernels:

    P_NL(t) = eps0 chi2 E(t)^2 ,     d_eff := chi2 / 2 .

Inserting E = sum_j E_j into the driven wave equation
    d^2 E/dz^2 - (n^2/c^2) d^2 E/dt^2 = mu0 d^2 P_NL/dt^2
and matching the exp(-i omega_3 t) component under the SVEA (drop d^2 A/dz^2) gives, for
omega_3 = omega_1 + omega_2 and the phase mismatch

    dk := k_3 - k_1 - k_2 ,     k_j = n_j omega_j / c ,

the NONDEGENERATE coupled-wave equations

    dA1/dz = i (omega1 d_eff / (n1 c)) A3 A2* exp(+ i dk z)
    dA2/dz = i (omega2 d_eff / (n2 c)) A3 A1* exp(+ i dk z)
    dA3/dz = i (omega3 d_eff / (n3 c)) A1 A2  exp(- i dk z)

NOTE the exponent sign: the A3 (sum-frequency) equation carries exp(-i dk z) for exp(-i
omega t). Boyd (Nonlinear Optics 3rd ed., Eqs. 2.2.10-2.2.12) writes exp(+i DeltaK z) with
DeltaK = k1 + k2 - k3 = -dk, i.e. the SAME physics; the roadmap's literal "exp(+i dk z)"
with dk = k3 - k1 - k2 is the conjugate and is corrected here by the derivation. Boyd's
prefactor 2 i omega^2 d_eff/(k c^2) = 2 i omega d_eff/(n c) carries an extra factor 2 that is
absorbed by Boyd's positive-frequency amplitude being half of this real-peak A_j.

DEGENERATE SHG (omega1 = omega2 = omega, omega3 = 2 omega) is NOT the nondegenerate set with
A1 = A2: squaring a SINGLE field gives half the cross-term of two distinct fields, so a 1/2
degeneracy factor appears on the up-conversion source:

    dA_f/dz = i (omega  d_eff / (n_f c)) A_s A_f* exp(+ i dk z)          (fundamental)
    dA_s/dz = i (2 omega d_eff / (n_s c)) (A_f^2 / 2) exp(- i dk z)      (second harmonic)
    dk = k_s - 2 k_f .

Undepleted phase-matched SHG then gives second-harmonic peak field
    |A_s(L)| = (omega chi2 / (2 n c)) A0^2 L        (chi2 = 2 d_eff, n_s = n_f = n)
which is byte-for-byte the degenerate coupled-wave oracle of
``validation/fdtd_chi2_shg_raman.py`` (its GATE B closed form). twm_reference USES the
d_eff = chi2/2 convention; pass ``d_eff = chi2/2`` to reproduce an FDTD chi2.

--------------------------------------------------------------------------------------------
DEGENERATE SPECS ARE REFUSED, NOT GUESSED  (audit Q-14)  ** behaviour change, v0.9.1 **
--------------------------------------------------------------------------------------------
A ``TWMSpec`` with ``omega1 == omega2`` used to run the NONDEGENERATE equations silently: for
A1 = A2 = A that is exactly 2x the SHG field (4x the intensity), and Manley-Rowe stays clean
(residual 3.6e-16) so no diagnostic fires. The 1/2 above is derived here and applied in
``shg_undepleted`` / ``twm_propagate(degenerate=True)``, but nothing routed a degenerate spec
there.

The frequencies alone CANNOT decide which physics is right, which is why this is refused rather
than auto-routed:

  * omega1 = omega2 and the two waves are the SAME MODE (collinear, same polarization, same
    index) -> SHG, the 1/2 factor applies. Use ``shg_undepleted`` / ``degenerate=True``.
  * omega1 = omega2 and the two waves are DISTINCT MODES -- type-II degenerate SFG/SPDC, where
    signal and idler are orthogonal polarizations with n1 != n2, the most common degenerate
    geometry in practice -> the NONDEGENERATE equations are correct as written, and there is no
    1/2. Say so with ``distinct_modes=True`` / ``degenerate=False``.

Auto-routing would silently pick the first for every caller, and in ``twm_propagate`` it would
also silently DISCARD ``amp2`` and ``n2``. So the nondegenerate entry points
(``sfg_undepleted``, ``twm_propagate``) raise ``ValueError`` on a degenerate spec unless the
caller states which case it is. ``opa_gain`` is NOT gated (degenerate OPA -- a single
signal = idler mode -- carries the same factor; see its docstring), and neither is ``TWMSpec``
itself, so building a degenerate spec for ``shg_undepleted`` stays a one-liner.

The refusal is TWO-SIDED. ``twm_propagate(spec, ..., degenerate=True)`` on a spec whose
``is_degenerate()`` is False also raises: that branch propagates ONE fundamental mode and
silently discards ``amp2``, ``n2`` and ``omega2``, which is the identical silent-discard failure
the ``degenerate=None`` refusal exists to prevent, reached through the other door. Measured
before the fix: ``omega2 = 1.31 omega1``, ``amp2 = 2e7`` V/m ignored, ``|A3(L)| = 2.79e7``
reported with ``degenerate=True`` and no diagnostic.

``TWMSpec`` additionally rejects a NON-FINITE ``omega1`` / ``omega2`` at construction, because
the degeneracy predicate is a comparison and nan compares False against everything: a nan spec
answered "not degenerate" and ran the nondegenerate equations, and ``omega1 = omega2 = inf``
answered "not degenerate" for a spec that is degenerate by inspection.

Intensity / photon flux (real-peak convention):
    I_j = (1/2) n_j eps0 c |A_j|^2 ,   photon flux Phi_j = I_j / (hbar omega_j) .
Manley-Rowe: with the equations above dN1/dz = dN2/dz = -dN3/dz where N_j := n_j |A_j|^2 /
omega_j, so N1 + N3 and N2 + N3 are conserved (one omega3 photon per omega1 + omega2 pair),
and total power sum_j I_j is conserved exactly (lossless).

References:
  R. W. Boyd, Nonlinear Optics, 3rd ed., Ch. 2 (CWEs, Manley-Rowe, QPM), Academic Press 2008.
  M. M. Fejer et al., IEEE J. Quantum Electron. 28, 2631 (1992) (first-order QPM, (2/pi) d_eff).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional, Union

import numpy as np
from scipy.integrate import solve_ivp

from dynameta.constants import C_LIGHT, EPS0, HBAR

__all__ = [
    "DEGENERACY_REL_TOL",
    "TWMSpec",
    "TWMResult",
    "sfg_undepleted",
    "shg_undepleted",
    "opa_gain",
    "twm_propagate",
    "phase_matching_sinc",
    "qpm_period_for",
    "effective_deff_qpm",
]

_IndexLike = Union[float, Callable[[float], float]]

# Relative separation below which omega1 and omega2 are the SAME frequency for the purpose of the
# degeneracy factor (audit Q-14). Loose enough to catch a spec built from two rounded floats or
# from omega_p/2 arithmetic, far tighter than any physical detuning: at omega ~ 1e15 rad/s this is
# ~1e6 rad/s = 0.16 MHz, orders of magnitude below the linewidth of anything this module models.
DEGENERACY_REL_TOL = 1e-9

_DEGENERATE_MSG = (
    "{where}: this TWMSpec is DEGENERATE (omega1 = {w1:.12g}, omega2 = {w2:.12g}, relative "
    "separation {sep:.3g} <= {tol:g}) and the NONDEGENERATE coupled-wave equations do not apply "
    "to it unaltered -- for A1 = A2 they give exactly 2x the second-harmonic field (4x the "
    "intensity), and Manley-Rowe stays conserved so nothing else flags it (audit Q-14). The "
    "frequencies alone cannot say which physics you mean, so state it:\n"
    "  * ONE mode driving its own second harmonic (collinear, same polarization/index) -> SHG: "
    "use {shg}, which carries the 1/2 degeneracy factor.\n"
    "  * TWO distinct modes that happen to share a frequency (type-II degenerate SFG/SPDC: "
    "orthogonal polarizations, n1 != n2) -> the nondegenerate equations ARE correct: pass "
    "{distinct} to run them.")


def _sinc(x: np.ndarray | float) -> np.ndarray | float:
    """sin(x)/x (the UNNORMALIZED sinc; numpy.sinc is sin(pi x)/(pi x))."""
    return np.sinc(np.asarray(x, dtype=float) / np.pi)


def _index(n: _IndexLike, omega: float) -> float:
    """Refractive index at omega: a constant float, or a callable n(omega)."""
    return float(n(omega)) if callable(n) else float(n)


@dataclass
class TWMSpec:
    """A collinear three-wave chi2 interaction omega3 = omega1 + omega2.

    Provide the two lower frequencies (omega3 is implied). Indices are either constants or
    callables n(omega) (dispersion). ``d_eff`` is the effective nonlinear coefficient in
    m/V with the P_NL = 2 eps0 d_eff E^2 (d_eff = chi2/2) convention documented in the module
    docstring. ``length`` is the interaction length (m). ``dk_override`` forces the phase
    mismatch dk = k3 - k1 - k2 (rad/m); otherwise it is computed from the indices.
    ``qpm_period`` (m), if set, engages first-order quasi-phase-matching (sign-flipping
    d_eff of that period).

    A spec with ``omega1 == omega2`` is DEGENERATE (see ``is_degenerate``). Building one is
    fine -- it is what ``shg_undepleted`` takes -- but the NONDEGENERATE entry points refuse it
    unless you say which physics you mean; see the module docstring (audit Q-14).
    """

    omega1: float
    omega2: float
    d_eff: float
    length: float
    n1: _IndexLike = 1.0
    n2: _IndexLike = 1.0
    n3: _IndexLike = 1.0
    dk_override: Optional[float] = None
    qpm_period: Optional[float] = None

    def __post_init__(self) -> None:
        """Validate that the two frequencies are FINITE (audit Q-14 residual).

        ``is_degenerate`` is a comparison, and every comparison against a NaN is False -- so a
        spec built with ``omega1 = nan`` (or ``inf``) reported ``is_degenerate() -> False`` and
        sailed straight through the degeneracy refusal into the nondegenerate equations, which
        then returned nan/inf amplitudes with no diagnostic naming the cause. ``inf`` is worse
        than nan: ``TWMSpec(omega1=inf, omega2=inf)`` is degenerate by inspection yet
        ``abs(inf - inf) = nan <= rel_tol * inf`` is False, so the guard's own predicate reports
        the OPPOSITE of the truth. Neither is a frequency, so both are rejected at construction
        and the degeneracy predicate downstream only ever sees finite numbers."""
        for name in ("omega1", "omega2"):
            val = getattr(self, name)
            try:
                fval = float(val)
            except (TypeError, ValueError):
                raise ValueError(
                    "TWMSpec.{}: expected a real frequency in rad/s, got {!r}".format(name, val)
                ) from None
            if not math.isfinite(fval):
                raise ValueError(
                    "TWMSpec.{}: must be a FINITE frequency in rad/s, got {!r}. A non-finite "
                    "omega defeats the degeneracy test itself -- `is_degenerate` is a comparison "
                    "and every comparison against nan is False, so a nan/inf spec reported "
                    "NONdegenerate and ran through the nondegenerate coupled-wave equations "
                    "(audit Q-14).".format(name, val))

    @property
    def omega3(self) -> float:
        return float(self.omega1) + float(self.omega2)

    def is_degenerate(self, rel_tol: float = DEGENERACY_REL_TOL) -> bool:
        """True when omega1 and omega2 coincide to ``rel_tol`` -- i.e. omega3 = 2 omega1 and the
        interaction is (or may be) second-harmonic generation rather than sum-frequency. See the
        module docstring's DEGENERATE SPECS section for why this is a REFUSAL, not a routing
        decision (audit Q-14).

        Both frequencies are guaranteed FINITE by ``__post_init__``, so this predicate never has
        to answer for a nan (which would compare False and report a nan spec as nondegenerate)."""
        w1, w2 = float(self.omega1), float(self.omega2)
        scale = max(abs(w1), abs(w2))
        if scale == 0.0:
            return True
        return abs(w1 - w2) <= rel_tol * scale

    def n(self, which: int) -> float:
        omega = (self.omega1, self.omega2, self.omega3)[which - 1]
        n = (self.n1, self.n2, self.n3)[which - 1]
        return _index(n, omega)

    def k(self, which: int) -> float:
        omega = (self.omega1, self.omega2, self.omega3)[which - 1]
        return self.n(which) * omega / C_LIGHT

    @property
    def dk(self) -> float:
        """Phase mismatch dk = k3 - k1 - k2 (rad/m)."""
        if self.dk_override is not None:
            return float(self.dk_override)
        return self.k(3) - self.k(1) - self.k(2)

    def kappa(self, which: int) -> float:
        """Coupling omega_j d_eff / (n_j c)  (units 1/V; kappa * A * length is dimensionless)."""
        omega = (self.omega1, self.omega2, self.omega3)[which - 1]
        return omega * self.d_eff / (self.n(which) * C_LIGHT)

    def intensity(self, which: int, amp: complex) -> float:
        """Time-averaged intensity I = (1/2) n eps0 c |A|^2 (W/m^2) for envelope ``amp``."""
        return 0.5 * self.n(which) * EPS0 * C_LIGHT * abs(amp) ** 2

    def photon_flux(self, which: int, amp: complex) -> float:
        """Photon flux Phi = I / (hbar omega) (photons / s / m^2)."""
        omega = (self.omega1, self.omega2, self.omega3)[which - 1]
        return self.intensity(which, amp) / (HBAR * omega)


# --------------------------------------------------------------------------------------------
# Quasi-phase-matching helpers
# --------------------------------------------------------------------------------------------

def qpm_period_for(dk: float, order: int = 1) -> float:
    """m-th order QPM period that phase-matches a residual mismatch dk = k3 - k1 - k2:

        Lambda = 2 pi m / |dk|      (a POSITIVE, physical poling period, m = ``order``).

    SIGN CONVENTION (the whole module is consistent with this; see ``phase_matching_sinc``).
    The poling square wave is an EVEN function of Lambda and carries BOTH SIGNS of every odd
    grating order, supplying momentum -/+ 2 pi m / Lambda. A given physical grating therefore
    phase-matches dk = +2 pi m/Lambda AND dk = -2 pi m/Lambda; only the ORDER SIGN differs.
    The returned period is the positive one -- what a poling mask is actually written to --
    for either sign of dk, and ``phase_matching_sinc`` selects the matching order sign. The
    signed form Lambda = 2 pi m / dk describes the SAME physical grating and is accepted
    everywhere a period is taken (``phase_matching_sinc`` picks the matching order either way,
    and ``twm_propagate`` bounds its step with |Lambda|), so the two spellings give identical
    physics end to end. Prefer the positive one: it is a length, and callers do arithmetic on
    it (crystal length = N * Lambda).

    Raises for dk == 0 (already phase matched -- no poling needed)."""
    if dk == 0.0:
        raise ValueError("qpm_period_for: dk == 0 is already phase matched; no QPM needed.")
    return 2.0 * math.pi * int(order) / abs(float(dk))


def effective_deff_qpm(d_eff: float, order: int = 1) -> float:
    """First-order QPM effective coefficient d_Q = (2 / (m pi)) d_eff (Fejer 1992). The square
    of the ratio, (2 / pi)^2 for m = 1, is the efficiency penalty vs true phase matching."""
    return 2.0 * float(d_eff) / (int(order) * math.pi)


def _qpm_sign(z: np.ndarray | float, period: float) -> np.ndarray | float:
    """Square-wave poling sign(cos(2 pi z / Lambda)) in {-1, +1}; the sign of d_eff(z).

    EVEN in ``period``: cos is even, so a negative Lambda is the SAME physical grating (the
    signed-period convention of ``qpm_period_for``)."""
    ph = np.cos(2.0 * math.pi * np.asarray(z, dtype=float) / period)
    return np.where(ph >= 0.0, 1.0, -1.0)


def phase_matching_sinc(dk: np.ndarray | float, length: float,
                        qpm_period: Optional[float] = None) -> np.ndarray | complex:
    """Complex phase-matching function Phi = (1/L) integral_0^L d_eff(z)/d_eff exp(-i dk z) dz.

    Uniform crystal: Phi = sinc(dk L / 2) exp(-i dk L / 2)  (magnitude |sinc(dk L/2)|).

    With QPM (period Lambda) the poling square wave has the Fourier series
        sign(cos(2 pi z/Lambda)) = sum_{m odd} c_m exp(i 2 pi m z / Lambda),
        c_m = (2/(pi |m|)) (-1)^((|m|-1)/2)     ==>  c_{+1} = c_{-1} = 2/pi,
    i.e. it carries EVERY ODD order m = +-1, +-3, +-5, ..., supplying momentum -/+ 2 pi m/Lambda.
    Term m contributes (1/L) integral_0^L exp(-i (dk - 2 pi m/Lambda) z) dz, so
        Phi_QPM = (2/(pi |m|)) sinc(dk_eff L / 2) exp(-i dk_eff L / 2),
        dk_eff  = dk - 2 pi m / Lambda,
    with m the NEAREST ODD order to dk Lambda / (2 pi), chosen ELEMENTWISE and carrying its own
    (2/(pi |m|)) weight. Ties (|dk Lambda / 2 pi| an even integer, including dk == 0) resolve to
    the SMALLER |m|, which is the one with the larger weight; dk == 0 therefore gives m = +1.
    Both signs of an order carry the same weight, so a grating of period |Lambda| matches
    dk = +2 pi m/Lambda and dk = -2 pi m/Lambda equally well -- which is why ``qpm_period_for``
    can return a positive period for either sign of dk (see its SIGN CONVENTION note). The
    negative-m branch is what makes dk < 0 (reachable whenever n3 < n1, n2) phase-match, and it
    also makes a SIGNED (negative) Lambda behave identically to its positive twin (m flips with
    Lambda, so m * 2 pi / Lambda -- and hence Phi -- does not; bitwise, at every dk except the
    dk == 0 tie, where the m = +1 convention is taken in the SIGN OF Lambda's frame and the two
    spellings come out complex-conjugate -- of a Phi that is real, so they agree to rounding).

    WHY ALL ODD ORDERS, not just m = +-1 (fix-verify W1 kill 3). Keeping only the first order made
    ``qpm_period_for(dk, order=3)`` / ``effective_deff_qpm(d, 3)`` -- which exist, and which return
    the m-th-order period and its 2/(3 pi) coefficient -- inconsistent with this function by
    fifteen orders of magnitude: at dk = +-3 * 2 pi/Lambda the exact square-wave response is
    2/(3 pi) = 0.2122, and the truncated form returned 2.4e-16 (100% error, silently). Every
    HIGHER-order QPM design (a coarser, easier-to-pole mask driven on m = 3 or 5) landed in that
    hole. Only the NEAREST odd order is summed -- the neighbouring orders are down by both their
    1/|m| weight and their sinc, and the residual is the ~1% closed-form-vs-integrator tolerance
    the QPM gates carry (measured against a brute-force piecewise-exact quadrature: <= 1.1e-2
    relative wherever |Phi| >= 0.5, i.e. inside the main lobe of whichever order is matched).
    Used by ``spdc_design.jsa`` (the sinc(delta k L/2) phase-matching envelope) and by the
    undepleted closed forms below. ``dk`` may be an array."""
    dk = np.asarray(dk, dtype=float)
    if qpm_period is None:
        arg = dk * length / 2.0
        return _sinc(arg) * np.exp(-1j * arg)
    g = 2.0 * math.pi / float(qpm_period)
    u = dk / g                                          # dk in units of the grating vector
    # nearest ODD integer to u, ties (|u| even) to the SMALLER |m| = the larger Fourier weight.
    # Written via |u| and re-signed so the selection is exactly antisymmetric in u -- np.round's
    # banker's rounding is not, and would break Phi(-dk) == conj(Phi(dk)) at the tie points.
    m_abs = (2.0 * np.maximum(np.ceil(0.5 * np.abs(u)), 1.0) - 1.0)[()]
    m = np.where(u < 0.0, -m_abs, m_abs)[()]
    dk_eff = dk - m * g
    # c_m = (2/(pi |m|)) (-1)^((|m|-1)/2): +1 for |m| = 1, 5, 9, ... and -1 for |m| = 3, 7, 11, ...
    # (|m| mod 4). Dropping it would leave |Phi| right and its PHASE wrong on those orders.
    c_m = (2.0 / (math.pi * m_abs)) * np.where(np.mod(m_abs, 4.0) == 1.0, 1.0, -1.0)[()]
    arg = dk_eff * length / 2.0
    return c_m * _sinc(arg) * np.exp(-1j * arg)


# --------------------------------------------------------------------------------------------
# Undepleted closed forms
# --------------------------------------------------------------------------------------------

def _reject_degenerate(spec: TWMSpec, where: str, shg: str, distinct: str) -> None:
    """Raise ``ValueError`` if ``spec`` is degenerate -- the audit Q-14 guard on the entry points
    that implement the NONDEGENERATE coupled-wave equations. No-op otherwise (so every
    nondegenerate call, which is every previously-correct call, is bit-for-bit untouched)."""
    if not spec.is_degenerate():
        return
    w1, w2 = float(spec.omega1), float(spec.omega2)
    scale = max(abs(w1), abs(w2)) or 1.0
    raise ValueError(_DEGENERATE_MSG.format(where=where, w1=w1, w2=w2,
                                            sep=abs(w1 - w2) / scale,
                                            tol=DEGENERACY_REL_TOL, shg=shg, distinct=distinct))


_NONDEGENERATE_MSG = (
    "{where}: degenerate=True was asked for, but this TWMSpec is NONDEGENERATE (omega1 = "
    "{w1:.12g}, omega2 = {w2:.12g}, relative separation {sep:.3g} > {tol:g}). The degenerate SHG "
    "branch propagates a SINGLE fundamental mode: it reads amp1/n1/omega1 and SILENTLY DISCARDS "
    "amp2, n2 and omega2, so the answer is the second harmonic of omega1 alone -- not the "
    "sum-frequency of your two waves, and every Manley-Rowe diagnostic stays clean while it is "
    "wrong. This is the same silent-discard failure the degeneracy guard exists to prevent, "
    "reached through the other door (audit Q-14). Either pass a spec whose omega1 and omega2 "
    "coincide (is_degenerate() True), or use shg_undepleted(spec, amp_fund) for the undepleted "
    "closed form; for two genuinely distinct waves drop degenerate=True and let the "
    "nondegenerate equations run.")


def _reject_nondegenerate(spec: TWMSpec, where: str) -> None:
    """Raise ``ValueError`` if ``spec`` is NOT degenerate -- the mirror of
    :func:`_reject_degenerate`, guarding the explicit ``degenerate=True`` opt-in (audit Q-14
    residual). No-op on a degenerate spec, so every correct degenerate call is bit-for-bit
    untouched."""
    if spec.is_degenerate():
        return
    w1, w2 = float(spec.omega1), float(spec.omega2)
    scale = max(abs(w1), abs(w2)) or 1.0
    raise ValueError(_NONDEGENERATE_MSG.format(where=where, w1=w1, w2=w2,
                                               sep=abs(w1 - w2) / scale,
                                               tol=DEGENERACY_REL_TOL))


def sfg_undepleted(spec: TWMSpec, amp1: complex, amp2: complex, *,
                   distinct_modes: bool = False) -> dict:
    """Undepleted sum-frequency A3(L) for constant inputs A1, A2 (A3(0) = 0):

        A3(L) = i kappa3 A1 A2 L Phi(dk),   Phi = sinc(dk L/2) exp(-i dk L/2)  (+ QPM),

    so the SFG intensity is sinc^2 in the phase mismatch. Returns the complex A3(L), its
    intensity, the pump-referenced conversion efficiency I3(L)/I1(0), and |sinc|^2.

    Raises ``ValueError`` on a DEGENERATE spec (omega1 == omega2) -- these are the nondegenerate
    equations and they overshoot the second harmonic by exactly 2x in field. Use
    ``shg_undepleted`` for one mode driving its own harmonic, or ``distinct_modes=True`` for two
    genuinely distinct co-frequency modes (type-II). See the module docstring (audit Q-14).
    ``distinct_modes=True`` reproduces the pre-fix numbers bit-for-bit."""
    if not distinct_modes:
        _reject_degenerate(spec, "sfg_undepleted", "shg_undepleted(spec, amp_fund)",
                           "distinct_modes=True")
    L = spec.length
    # A3(L) = i kappa3 A1 A2 integral_0^L (d_eff(z)/d_eff) exp(-i dk z) dz = i kappa3 A1 A2 L Phi;
    # phase_matching_sinc carries the (2/pi) first-order factor and the QPM peak shift, so
    # kappa3 here uses the FULL d_eff (no double counting).
    phi = phase_matching_sinc(spec.dk, L, spec.qpm_period)
    kappa3 = spec.kappa(3)
    a3 = 1j * kappa3 * amp1 * amp2 * L * phi
    I3 = spec.intensity(3, a3)
    I1 = spec.intensity(1, amp1)
    return {
        "A3_L": complex(a3),
        "I3_L": float(I3),
        "efficiency": float(I3 / I1) if I1 > 0 else float("nan"),
        "sinc2": float(abs(phi) ** 2),
    }


def shg_undepleted(spec: TWMSpec, amp_fund: complex) -> dict:
    """Undepleted degenerate second harmonic for a constant fundamental A_f (A_s(0) = 0):

        A_s(L) = i kappa_s (A_f^2 / 2) L Phi(dk),   kappa_s = omega3 d_eff / (n3 c),

    with the 1/2 degeneracy factor (module docstring). For omega1 = omega2 = omega, dk = 0,
    n_s = n_f, and d_eff = chi2/2 the peak SH field is (omega chi2 / (2 n c)) |A_f|^2 L -- the
    ``fdtd_chi2_shg_raman`` GATE B oracle. Returns A_s(L), its intensity, and eta = I_s/I_f."""
    L = spec.length
    # phase_matching_sinc carries the (2/pi) QPM factor + peak shift; kappa_s uses full d_eff.
    phi = phase_matching_sinc(spec.dk, L, spec.qpm_period)
    kappa_s = spec.kappa(3)
    a_s = 1j * kappa_s * (amp_fund ** 2 / 2.0) * L * phi
    Is = spec.intensity(3, a_s)
    If = spec.intensity(1, amp_fund)
    return {
        "A_s_L": complex(a_s),
        "I_s_L": float(Is),
        "efficiency": float(Is / If) if If > 0 else float("nan"),
        "sinc2": float(abs(phi) ** 2),
    }


def opa_gain(spec: TWMSpec, amp_pump: complex, amp_signal: complex,
             amp_idler: complex = 0.0) -> dict:
    """Optical parametric amplification: strong UNDEPLETED pump A3 = ``amp_pump`` at omega3
    down-converting to signal A1 (omega1) and idler A2 (omega2). Closed-form solution of

        dA1/dz = i kappa1 A3 A2* exp(+i dk z),   dA2/dz = i kappa2 A3 A1* exp(+i dk z)

    with the gain coefficient

        g = sqrt( kappa1 kappa2 |A3|^2 - (dk/2)^2 ) .

    g REAL (above threshold) -> exponential signal cosh^2 / idler sinh^2 growth; g imaginary
    (below threshold, dk too large) -> oscillatory cos^2/sin^2 (cosh/sinh of an imaginary
    argument). Handled in one branch via complex g. For dk = 0, signal-only input:
    |A1(L)|^2 = |A1(0)|^2 cosh^2(gL), |A2(L)|^2 = (kappa2/kappa1)|A1(0)|^2 sinh^2(gL).

    Returns the complex output signal/idler amplitudes, their power gains, |g| or the
    oscillation rate, and the Manley-Rowe check (idler photons gained == signal photons
    gained).

    DEGENERATE OPA (omega1 == omega2, i.e. omega3 = 2 omega1) is NOT gated here the way
    ``sfg_undepleted`` / ``twm_propagate`` are (audit Q-14), because these two equations are also
    the correct nondegenerate physics for a type-II degenerate parametric amplifier (signal and
    idler orthogonal, n1 != n2), which is the common case. If signal and idler are the SAME mode
    -- true single-mode degenerate parametric amplification / squeezing -- the same 1/2 factor
    the module docstring derives applies to the down-conversion source and these gains overshoot;
    that regime is out of scope for this three-envelope reference."""
    L = spec.length
    dk = spec.dk
    k1, k2 = spec.kappa(1), spec.kappa(2)
    Ap = complex(amp_pump)
    g2 = k1 * k2 * abs(Ap) ** 2 - (dk / 2.0) ** 2
    g = np.sqrt(complex(g2))                     # complex: real -> hyperbolic, imag -> trig
    a10, a20 = complex(amp_signal), complex(amp_idler)
    # d/dz [B1; B2*] = M [B1; B2*] with B_j = A_j exp(-i dk z/2); solve the 2x2 exactly.
    ch = np.cosh(g * L)
    sh_over_g = (np.sinh(g * L) / g) if abs(g) > 0 else complex(L)   # sinh(gL)/g -> L as g->0
    B1 = a10 * (ch - 1j * (dk / 2.0) * sh_over_g) + 1j * k1 * Ap * np.conj(a20) * sh_over_g
    B2c = np.conj(a20) * (ch + 1j * (dk / 2.0) * sh_over_g) - 1j * k2 * np.conj(Ap) * a10 * sh_over_g
    a1L = B1 * np.exp(1j * dk * L / 2.0)
    a2L = np.conj(B2c) * np.exp(1j * dk * L / 2.0)
    dN1 = spec.photon_flux(1, a1L) - spec.photon_flux(1, a10)
    dN2 = spec.photon_flux(2, a2L) - spec.photon_flux(2, a20)
    return {
        "A1_L": complex(a1L),
        "A2_L": complex(a2L),
        "signal_gain": float(abs(a1L) ** 2 / abs(a10) ** 2) if a10 != 0 else float("nan"),
        "idler_gain": float(abs(a2L) ** 2 / abs(a20) ** 2) if a20 != 0 else float("nan"),
        "g": complex(g),
        "gL": complex(g * L),
        "above_threshold": bool(g2 > 0.0),
        "manley_rowe_residual": float(abs(dN1 - dN2) / (abs(dN1) + abs(dN2) + 1e-300)),
    }


# --------------------------------------------------------------------------------------------
# Depleted three-envelope integrator
# --------------------------------------------------------------------------------------------

@dataclass
class TWMResult:
    """Output of ``twm_propagate``. ``z`` is the sampled grid (m); A1/A2/A3 the complex
    envelopes; power the per-wave and total intensities (W/m^2); photon flux per wave;
    Manley-Rowe invariants N1+N3, N2+N3 and their max relative drift; total-power drift."""

    z: np.ndarray
    A1: np.ndarray
    A2: np.ndarray
    A3: np.ndarray
    I1: np.ndarray
    I2: np.ndarray
    I3: np.ndarray
    total_power: np.ndarray
    N1: np.ndarray
    N2: np.ndarray
    N3: np.ndarray
    manley_rowe_13: np.ndarray
    manley_rowe_23: np.ndarray
    mr13_residual: float
    mr23_residual: float
    power_residual: float
    degenerate: bool


def twm_propagate(spec: TWMSpec, amp1: complex, amp2: complex, amp3: complex,
                  *, n_out: int = 129, degenerate: Optional[bool] = None,
                  rtol: float = 3e-14, atol: float = 1e-15,
                  max_step: Optional[float] = None) -> TWMResult:
    """Integrate the DEPLETED coupled-wave equations from z = 0 to spec.length (DOP853).

    Nondegenerate: state (A1, A2, A3) = (``amp1``, ``amp2``, ``amp3``). Degenerate SHG
    (``degenerate=True``): fundamental A_f = ``amp1`` at omega1, second harmonic A_s =
    ``amp3`` at omega3 = 2 omega1 (``amp2`` ignored); uses the 1/2 degeneracy source. QPM
    (spec.qpm_period set) flips d_eff(z) as a square wave of that period.

    ``degenerate`` is TRI-STATE (audit Q-14; it used to default to ``False``):
      * ``None`` (default) -- infer from the spec, and REFUSE a degenerate one. A spec with
        omega1 == omega2 raises ``ValueError``, because running it nondegenerate silently gives
        2x the second-harmonic field while every diagnostic stays clean, and auto-routing it to
        the SHG branch would silently discard ``amp2`` and ``n2``.
      * ``True``  -- degenerate SHG branch. RAISES ``ValueError`` on a spec whose
        ``is_degenerate()`` is False: that branch reads amp1/n1/omega1 and silently discards
        amp2/n2/omega2, which is the very failure the ``None`` refusal exists to prevent,
        reached through the other door. On a genuinely degenerate spec it is unchanged.
      * ``False`` -- nondegenerate branch, no check: the explicit "these are two distinct modes
        that happen to share a frequency" (type-II) opt-in, bit-for-bit the pre-fix behaviour.

    Amplitudes are internally rescaled to O(1) before integration (the coupling then carries
    the scale) so the DOP853 tolerances resolve the invariants to ~1e-12; the result is
    un-scaled back to V/m. Returns a ``TWMResult`` with exact Manley-Rowe / power diagnostics.
    """
    if degenerate is None:
        _reject_degenerate(spec, "twm_propagate", "degenerate=True", "degenerate=False")
        degenerate = False
    elif degenerate:
        _reject_nondegenerate(spec, "twm_propagate")
    L = float(spec.length)
    dk = spec.dk
    period = spec.qpm_period

    scale = max(abs(amp1), abs(amp2), abs(amp3), 1e-300)

    def deff_sign(z: float) -> float:
        return 1.0 if period is None else float(_qpm_sign(z, period))

    if degenerate:
        kf = spec.kappa(1)          # omega1 d_eff/(n1 c)
        ks = spec.kappa(3)          # omega3 d_eff/(n3 c), omega3 = 2 omega1
        b0 = np.array([amp1 / scale, amp3 / scale], dtype=complex)

        def rhs(z, y):
            s = deff_sign(z)
            af = y[0] + 1j * y[1]
            ash = y[2] + 1j * y[3]
            daf = 1j * (kf * scale * s) * ash * np.conj(af) * np.exp(1j * dk * z)
            das = 1j * (ks * scale * s) * (af ** 2 / 2.0) * np.exp(-1j * dk * z)
            return [daf.real, daf.imag, das.real, das.imag]

        y0 = [b0[0].real, b0[0].imag, b0[1].real, b0[1].imag]
    else:
        k1, k2, k3 = spec.kappa(1), spec.kappa(2), spec.kappa(3)
        b0 = np.array([amp1 / scale, amp2 / scale, amp3 / scale], dtype=complex)

        def rhs(z, y):
            s = deff_sign(z)
            a1 = y[0] + 1j * y[1]
            a2 = y[2] + 1j * y[3]
            a3 = y[4] + 1j * y[5]
            d1 = 1j * (k1 * scale * s) * a3 * np.conj(a2) * np.exp(1j * dk * z)
            d2 = 1j * (k2 * scale * s) * a3 * np.conj(a1) * np.exp(1j * dk * z)
            d3 = 1j * (k3 * scale * s) * a1 * a2 * np.exp(-1j * dk * z)
            return [d1.real, d1.imag, d2.real, d2.imag, d3.real, d3.imag]

        y0 = [b0[0].real, b0[0].imag, b0[1].real, b0[1].imag, b0[2].real, b0[2].imag]

    z_eval = np.linspace(0.0, L, int(n_out))
    if max_step is None:
        # resolve the QPM domains (and any fast beat) if poling is on.  |period|: the poling
        # square wave is even in Lambda, so a signed (negative) period from the
        # Lambda = 2 pi m / dk convention is the same grating and must not produce a negative
        # step bound (see qpm_period_for's SIGN CONVENTION note).
        max_step = (abs(period) / 20.0) if period is not None else np.inf
    sol = solve_ivp(rhs, (0.0, L), y0, method="DOP853", t_eval=z_eval,
                    rtol=rtol, atol=atol, max_step=max_step, dense_output=False)
    if not sol.success:
        raise RuntimeError("twm_propagate: DOP853 failed: {}".format(sol.message))

    z = sol.t
    if degenerate:
        A1 = (sol.y[0] + 1j * sol.y[1]) * scale
        A3 = (sol.y[2] + 1j * sol.y[3]) * scale
        A2 = np.zeros_like(A1)
        I1 = spec.intensity(1, A1)
        I3 = spec.intensity(3, A3)
        I2 = np.zeros_like(I1)
        N1 = np.array([spec.photon_flux(1, a) for a in A1])
        N3 = np.array([spec.photon_flux(3, a) for a in A3])
        N2 = np.zeros_like(N1)
        # 2 fundamental photons per SH photon: N1 + 2 N3 conserved
        mr13 = N1 + 2.0 * N3
        mr23 = mr13.copy()
    else:
        A1 = (sol.y[0] + 1j * sol.y[1]) * scale
        A2 = (sol.y[2] + 1j * sol.y[3]) * scale
        A3 = (sol.y[4] + 1j * sol.y[5]) * scale
        I1 = spec.intensity(1, A1)
        I2 = spec.intensity(2, A2)
        I3 = spec.intensity(3, A3)
        N1 = np.array([spec.photon_flux(1, a) for a in A1])
        N2 = np.array([spec.photon_flux(2, a) for a in A2])
        N3 = np.array([spec.photon_flux(3, a) for a in A3])
        mr13 = N1 + N3
        mr23 = N2 + N3

    total = I1 + I2 + I3
    ref13 = abs(mr13[0]) + 1e-300
    ref23 = abs(mr23[0]) + 1e-300
    refP = abs(total[0]) + 1e-300
    return TWMResult(
        z=z, A1=A1, A2=A2, A3=A3, I1=I1, I2=I2, I3=I3, total_power=total,
        N1=N1, N2=N2, N3=N3, manley_rowe_13=mr13, manley_rowe_23=mr23,
        mr13_residual=float(np.max(np.abs(mr13 - mr13[0])) / ref13),
        mr23_residual=float(np.max(np.abs(mr23 - mr23[0])) / ref23),
        power_residual=float(np.max(np.abs(total - total[0])) / refP),
        degenerate=bool(degenerate),
    )
