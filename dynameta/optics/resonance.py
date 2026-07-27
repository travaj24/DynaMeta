"""Complex-omega pole finder for planar layered stacks -- resonances / quasi-normal modes (QNMs).

A resonance of a scattering system is a POLE of its scattering response at a complex frequency
``omega_tilde = omega_0 - i*gamma/2``.  Under this library's ``exp(-i*omega*t)`` time convention a
DECAYING mode sits in the LOWER half plane (``Im(omega_tilde) < 0``) and its quality factor is

    Q = omega_0 / (2 |Im(omega_tilde)|) = Re(omega_tilde) / (2 |Im(omega_tilde)|).

This module provides:

  * ``layered_smatrix_complex`` -- a self-contained transfer-matrix evaluator of the 2x2 stack
    scattering matrix, ANALYTIC in complex ``omega`` (no external ``tmm`` call, so it can be
    continued off the real axis).  On the real axis its reflectance/transmittance reproduce
    ``dynameta.optics.tmm_reference`` to machine precision.
  * ``drude_eps`` / ``lorentzian_eps`` -- closed-form, analytic-in-omega material models.
  * ``find_poles`` -- argument-principle (winding-number) contour counting on a rectangle in the
    complex-omega plane to LOCATE poles, followed by Newton refinement.  Subdivides recursively
    until each sub-rectangle isolates a single pole (robust to nearby poles/zeros).
  * ``pole_q`` -- the Q of a complex pole.
  * ``track_pole`` -- continuation tracking of one pole along a swept parameter (warm-started
    Newton with step halving on jump detection).
  * ``q_budget`` -- the radiative/absorptive Q split via the lossless/lossy two-pass
    (re-find the pole with the material losses removed: ``Q_rad``; ``1/Q_abs = 1/Q - 1/Q_rad``).
  * ``berreman_enz_pole`` -- convenience finder for the ENZ / Berreman mode of a thin Drude film.

-------------------------------------------------------------------------------------------------
BRANCH CHOICE (this is where naive implementations die)
-------------------------------------------------------------------------------------------------
In every medium the out-of-plane wavevector is ``kz = sqrt(eps*(omega/c)^2 - k_par^2)`` where the
in-plane wavevector ``k_par`` is held FIXED during the complex-omega continuation (the physically
correct QNM convention -- the mode is labelled by its conserved transverse momentum, not by a
fixed real angle).  ``kz`` is evaluated on numpy's PRINCIPAL square-root branch,
``np.sqrt(... + 0j)`` (the branch with ``Re(kz) >= 0``, and ``Im(kz) > 0`` on the negative-real
cut).  Why this is the correct OUTGOING branch under ``exp(-i*omega*t)`` with a forward wave
``exp(+i*kz*z)``:

  * On the real-omega axis with a lossless propagating channel the principal branch gives
    ``Re(kz) > 0``: the transmitted/reflected wave carries power AWAY from the stack (outgoing).
  * For an evanescent channel (``k_par > n*omega/c``) the argument is real-negative and the
    principal branch gives ``Im(kz) > 0``: the wave DECAYS away from the stack (outgoing/bound).
  * Continuing ``omega`` into the lower half plane at fixed real ``eps``, the principal branch
    stays continuous (the argument ``eps*(omega/c)^2 - k_par^2`` is generically complex and does
    not cross the negative-real cut inside a search box around a resonance), so it remains the
    analytic continuation of the real-axis OUTGOING wave.  A genuine QNM therefore GROWS spatially
    at infinity (``Im(kz) < 0`` in the propagating end media) -- the hallmark of a leaky resonance
    -- which the principal branch reproduces automatically.

Inside a FINITE layer the sign of ``kz`` only relabels which internal amplitude is "forward"; the
scattering matrix (hence the pole locations) is invariant under ``kz -> -kz`` there, so only the
two semi-infinite end media fix the physics -- and for them the principal branch is exactly the
outgoing choice.  The finder assumes the search rectangle does not straddle a branch point
``kz = 0`` (``eps*(omega/c)^2 = k_par^2``) of an end medium; for a well-separated resonance this is
comfortably satisfied.

References
----------
* S. J. Byrnes, "Multilayer optical calculations", arXiv:1603.02720 (the ``tmm`` conventions the
  real-axis evaluator matches).
* P. Lalanne, W. Yan, K. Vynck, C. Sauvan, J.-P. Hugonin, "Light Interaction with Photonic and
  Plasmonic Resonances", Laser Photonics Rev. 12, 1700113 (2018) (QNM definitions, Q, the
  ``exp(-i*omega*t)`` sign of ``omega_tilde``).
* L. M. Delves, J. N. Lyness, "A numerical method for locating the zeros of an analytic function",
  Math. Comp. 21, 543 (1967) (argument-principle root counting).
* S. Vassant, J.-P. Hugonin, F. Marquier, J.-J. Greffet, "Berreman mode and epsilon near zero
  mode", Opt. Express 20, 23971 (2012) (the thin-film ENZ/Berreman mode found by
  ``berreman_enz_pole``).

Conventions: SI units, ``exp(-i*omega*t)`` (a passive/absorbing medium has ``Im(eps) > 0``), pure
numpy/scipy, ASCII-only.

POLARIZATION VOCABULARY (audit V-8): this module speaks {'s', 'p'} -- E relative to the PLANE OF
INCIDENCE. It is one of five spellings in the repo -- {'x','y','p'} is OpticalSpec's LAB AXIS,
{'te','tm'} the lumenairy grating bridge's, the integer `row` 0/1 the differentiable
Berreman/RCWA/PMM forwards', and `pol_axis` hydro_fem's 2-D in-plane axis. The map, the
`normalize_pol` converter and the normal-incidence / azimuth caveats live in
`dynameta.core.polarization`. ACCEPTANCE UNIFICATION (b) -- the V-8 follow-on -- widened the
ACCEPTED set here by exactly the UNCONDITIONAL aliases: this module's entry points also take
`'te'`/`'tm'` and mixed case, normalized to `'s'`/`'p'` at the door, because in a planar stack TE is s
and TM is p by definition of the plane of incidence, at every angle and in every material. No valid
call changed by a bit -- `'s'`/`'p'` never touch the guard. The geometry-DEPENDENT spellings
(OpticalSpec's lab `'x'`/`'y'`, the integer `row`) are still REFUSED: convert those YOURSELF with
`normalize_pol`, which demands the azimuth and refuses rather than guess.
"""

from __future__ import annotations

import cmath
import math
import warnings
from typing import Callable, List, NamedTuple, Sequence, Tuple, Union

import numpy as np

from dynameta.constants import C_LIGHT

__all__ = [
    "drude_eps",
    "lorentzian_eps",
    "SMatrix",
    "layered_smatrix_complex",
    "k_par_from_angle",
    "smatrix_pole_func",
    "newton_refine",
    "pole_q",
    "find_poles",
    "track_pole",
    "q_budget",
    "berreman_enz_pole",
]

# A layer is (eps, thickness_m); eps is a complex constant OR a callable eps(omega_rad_s).
EpsSpec = Union[complex, float, Callable[[complex], complex]]
Layer = Tuple[EpsSpec, float]


# ------------------------------------------------------------------------------------------------
# Analytic material models (closed form => analytic in complex omega)
# ------------------------------------------------------------------------------------------------
def drude_eps(omega_rad_s, eps_inf, wp, gamma):
    """Free-carrier Drude permittivity, analytic in ``omega``:

        eps(omega) = eps_inf - wp**2 / (omega**2 + i*omega*gamma)

    ``wp`` is the (unscreened) plasma frequency ``sqrt(n e^2 / (eps0 m*))`` [rad/s] and ``gamma``
    the collision rate [rad/s].  Under ``exp(-i*omega*t)`` a real ``omega > 0`` with ``gamma >= 0``
    gives ``Im(eps) >= 0`` (passive), matching ``materials.DrudeOptical``.  The ENZ (epsilon near
    zero) crossing of ``Re(eps)`` is at ``omega ~ wp / sqrt(eps_inf)``.  Accepts complex ``omega``
    (the whole point -- it is continued off the real axis by the pole finder)."""
    w = np.asarray(omega_rad_s, dtype=np.complex128)
    return eps_inf - wp * wp / (w * w + 1j * w * gamma)


def lorentzian_eps(omega_rad_s, eps_inf, delta_eps, omega0, gamma):
    """Single-oscillator Lorentz permittivity, analytic in ``omega``:

        eps(omega) = eps_inf + delta_eps * omega0**2 / (omega0**2 - omega**2 - i*omega*gamma)

    ``delta_eps`` is the oscillator strength (static contribution ``eps(0)-eps_inf``), ``omega0``
    the resonance frequency [rad/s], ``gamma`` the linewidth [rad/s].  Under ``exp(-i*omega*t)`` a
    real ``omega > 0`` with ``gamma >= 0`` gives ``Im(eps) >= 0`` (passive).  Accepts complex
    ``omega``."""
    w = np.asarray(omega_rad_s, dtype=np.complex128)
    return eps_inf + delta_eps * omega0 * omega0 / (omega0 * omega0 - w * w - 1j * w * gamma)


def _eval_eps(spec: EpsSpec, omega: complex) -> complex:
    """Evaluate a layer's eps spec (constant or callable eps(omega)) at complex omega."""
    return complex(spec(omega)) if callable(spec) else complex(spec)


# ------------------------------------------------------------------------------------------------
# Complex-omega transfer-matrix S-matrix evaluator
# ------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------
# SCALAR 2x2 ALGEBRA (audit P-6)
#
# Everything below this line -- _kz, _interface, _propagate, the layered_smatrix_complex cascade
# and _stack_denominator -- is SCALAR physics: ~30 complex flops per call on 2x2 matrices whose
# entries are complex NUMBERS, not arrays.  It used to be written in array numpy
# (`np.array([[a, b], [c, d]])` + `@`), where the measured cost was essentially all dispatch:
# a 2x2 array build 9.3 us and a 2x2 `A @ B` 6.4-10.9 us, against arithmetic that takes tens of
# nanoseconds.  Written with plain Python complex scalars (SAME operation order, same formulas)
# it is 2.6x-3.6x on `_stack_denominator`, 4.9x on `layered_smatrix_complex` and 1.7-1.8x on
# `find_poles` end to end (124.1 -> 72.6 ms on the 3-pole Fabry-Perot box on an idle machine;
# 232.2 -> 129.2 ms on a loaded one), on top of the P-14 memo.
#
# THIS IS NOT BIT-IDENTICAL, and it is the only change in the module that is not.  numpy's 2x2
# `@` routes through BLAS `zgemm`, which accumulates differently from `a*e + b*g`, and numpy's
# complex division differs from CPython's.  Measured drift, and why it is accepted:
#   * D(omega) itself: max 8.7e-16 (Fabry-Perot), 6.4e-16 (Drude film), 2.2e-14 (3-layer lossy)
#     relative, over 4000 random complex omega each -- 1-2 ulp.
#   * POLE POSITIONS on every shipped gate configuration (Fabry-Perot m = 3..6, the 3-pole box,
#     three Berreman/ENZ films through berreman_enz_pole's own cleared function and default box):
#     worst 2.1e-16 relative, worst Q drift 4.0e-16.
#   * HOW MANY poles come back BIT-identical is CONFIGURATION-DEPENDENT and is a minority on the
#     broad suite: measured 58/253 (23 %) over 24 randomized 4-layer stacks plus near-degenerate
#     and extreme-Q boxes, and 16/24 (67 %) over the extreme-Q / near-degenerate / Berreman
#     families alone.  The earlier "most poles are bit-identical" was read off the second, smaller
#     family only.  The number that carries the acceptance is the DRIFT bound above, not the
#     identity fraction.
#   * That drift is four decades below the 1e-12 acceptance bound this change was gated on, and
#     below `find_poles`' own `refine_tol = 1e-11` Newton tolerance -- the finder cannot resolve
#     it.
#
# WHAT IS SAFE TO SUBSTITUTE.  Measured on this platform over 20 000-60 000 samples per class,
# for the argument classes this module can actually produce AND for classes it cannot (so the
# exceptions are on record):
#   * `cmath.sqrt` == `np.sqrt` BIT-for-bit (0 mismatches) on generic, negative-real-cut
#     (including exact -0.0 imaginary parts), tiny-imaginary, 1e30-magnitude, and the `eps k0^2 -
#     kpar^2` shape `_kz` actually feeds it.  EXCEPTION: DENORMAL arguments (|z| ~ 1e-310) differ
#     in essentially every sample (39706 mismatching words / 20 000 values).  `_kz` can only reach
#     that by an exact cancellation of `eps k0^2` against `kpar^2` down to 1e-310, which no
#     physical stack does; if a caller ever contrives one, the two spellings disagree there.
#   * `cmath.exp` == `np.exp` BIT-for-bit (0/40 000) on the argument shape `_propagate` produces,
#     `1j*kz*d` with |arg| ~ 30, and on the negative-real-cut / tiny-imaginary / huge / kz^2
#     classes.  EXCEPTIONS at LARGE |arg|: 7/38428 mismatches at |arg| ~ 400 and 30/45432 at
#     |arg| ~ 1e3 (and cmath.exp raises OverflowError where np.exp returns inf, which is why the
#     measurement excludes the overflowing samples).  A layer thick enough in units of the decay
#     length to put |kz d| past ~400 has `exp` at ~1e173 and is numerically meaningless anyway.
#   * `cmath.cos` / `cmath.sin` are NOT bit-identical to numpy's -- 12362/40000 mismatches on the
#     `kz d` argument shape itself, 882/31362 on generic arguments.  `_stack_denominator`
#     therefore KEEPS `np.cos` / `np.sin` and converts the np.complex128 result to a Python
#     complex with `complex(...)`, which is an exact round trip (0/400 000).
#   * `drude_eps` / `lorentzian_eps` are deliberately NOT rewritten: they are public, accept
#     ARRAYS, and their `wp^2 / (w^2 + i w gamma)` would change value under CPython's division
#     algorithm.  The remaining per-call dispatch there lives in the caller's eps closure.
# ------------------------------------------------------------------------------------------------
def _kz(eps: complex, k0: complex, k_par: complex) -> complex:
    """Out-of-plane wavevector on the PRINCIPAL (outgoing) branch -- see the module branch note.
    ``cmath.sqrt`` is bit-identical to the ``np.sqrt(... + 0j)`` this replaced on every argument
    class reachable here; it is NOT on DENORMAL arguments, which need an exact 1e-310-level
    cancellation of ``eps k0^2`` against ``kpar^2`` to reach (audit P-6)."""
    return cmath.sqrt(eps * k0 * k0 - k_par * k_par)


# The ONE polarization vocabulary of this module (audit V-3/V-8): the 'sp' family of
# dynameta.core.polarization, matching layered_smatrix_complex's long-standing check.  ACCEPTANCE
# UNIFICATION (b): 'te'/'tm' and mixed case are ACCEPTED and normalized to 's'/'p' at the door --
# TE is s and TM is p by definition of the plane of incidence, so that widening is unconditional
# and lossless.  What stays REFUSED is the geometry-DEPENDENT crossing: OpticalSpec's lab 'x'/'y'
# (the correspondence needs the azimuth) and the integer `row`.  No valid call moved: 's'/'p'
# never reach the guard.
_POL_NO_DEFAULT = ("There is no default here: 'x' -- and every other off-vocabulary label -- used "
                   "to return the p-polarized function silently (audit V-3). 'TE'/'TM' left that "
                   "class in the other direction: they are now ACCEPTED as unconditional aliases "
                   "of 's'/'p' and give the mode their name already meant.")


def _reject_pol(pol, where: str):
    """Raise the shared V-8 vocabulary error for a ``pol`` the 'sp' vocabulary does not accept.

    The import is LAZY, inside the failure path only, so the valid path pays nothing and this
    module gains no import edge (dynameta.core.polarization is stdlib-only documentation +
    validation)."""
    from dynameta.core.polarization import pol_vocabulary_error
    raise pol_vocabulary_error(pol, "sp", where=where, param="pol", extra=_POL_NO_DEFAULT)


def _accept_pol(pol, where: str, param: str = "pol") -> str:
    """ACCEPTANCE UNIFICATION (b): normalize an sp-family label to the canonical 's'/'p', accepting
    the UNCONDITIONAL 'te'/'tm' and mixed-case spellings (in a planar stack TE is s and TM is p by
    definition of the plane of incidence, at every angle), or raise the shared V-8 error.

    Reached ONLY when the label is not already 's'/'p' -- every caller keeps that cheap `not in`
    test inline -- so a valid call runs bit-identically and this module still gains no import edge
    on the happy path (the import is lazy, inside the widening/failure branch)."""
    from dynameta.core.polarization import accept_pol
    return accept_pol(pol, "sp", where=where, param=param, extra=_POL_NO_DEFAULT)


def _admittance(eps: complex, kz: complex, pol: str) -> complex:
    """Reduced optical admittance ``Y = H_tan / E_tan`` (common factors dropped):
    s-pol (TE) ``Y ~ kz``; p-pol (TM) ``Y ~ eps / kz``.  Reflectance ``|r|^2`` and transmittance
    ``|t|^2 * Re(Y_sub)/Re(Y_super)`` built from these reproduce the exact Fresnel power
    coefficients for both polarizations (validated against ``tmm`` to ~1e-14).

    AUDIT V-3/V-8: RAISES on any other ``pol``.  This used to fall through to p-pol (``return kz
    if pol == "s" else eps/kz``) while its verbatim twin ``nonlocal_tmm._admittance`` fell through
    to s-pol -- mirror-image silent defaults, so the two modules DISAGREED on every off-vocabulary
    string (``"TE"``, ``"TM"``, ``"x"``, ``"S"``).  Both twins now raise, the stricter of the two
    behaviours and the one the public entry points (``layered_smatrix_complex``, ``stack_rt``,
    ``pole_function``) already had."""
    if pol not in ("s", "p"):
        pol = _accept_pol(pol, "_admittance")
    return kz if pol == "s" else eps / kz


def _interface(Ya: complex, Yb: complex) -> Tuple[complex, complex, complex, complex]:
    """2x2 interface matrix mapping the (forward, backward) tangential-E amplitudes referenced on
    the ``b`` side to those on the ``a`` side (continuity of E_tan and H_tan)::

        [A_a; B_a] = 0.5 * [[1+rho, 1-rho], [1-rho, 1+rho]] [A_b; B_b],   rho = Y_b / Y_a

    Returned ROW-MAJOR as ``(m00, m01, m10, m11)``, not as a 2x2 array (audit P-6)."""
    rho = Yb / Ya
    p = 0.5 * (1.0 + rho)
    m = 0.5 * (1.0 - rho)
    return p, m, m, p


def _propagate(kz: complex, d: float) -> Tuple[complex, complex]:
    """2x2 propagation matrix advancing the reference plane by ``d`` toward the substrate.
    Forward wave ~ ``exp(+i*kz*z)`` (``exp(-i*omega*t)`` convention) => ``diag(e^{-i kz d},
    e^{+i kz d})``.  This sign is what places DECAYING poles in the lower half plane (verified
    against the Fabry-Perot closed form).

    Returned as the DIAGONAL pair ``(m00, m11)`` -- it has no off-diagonal (audit P-6)."""
    e = cmath.exp(1j * kz * d)
    return 1.0 / e, e


def _mat2_mul(A: Tuple[complex, complex, complex, complex],
              B: Tuple[complex, complex, complex, complex]
              ) -> Tuple[complex, complex, complex, complex]:
    """Row-major 2x2 product ``A @ B`` on plain complex scalars (audit P-6)."""
    a00, a01, a10, a11 = A
    b00, b01, b10, b11 = B
    return (a00 * b00 + a01 * b10, a00 * b01 + a01 * b11,
            a10 * b00 + a11 * b10, a10 * b01 + a11 * b11)


class SMatrix(NamedTuple):
    """Result of :func:`layered_smatrix_complex`.

    Attributes
    ----------
    r, t : complex
        Top-side amplitude reflection / transmission coefficients (tangential-E ratios).
    R, T : float
        Power reflectance ``|r|^2`` and transmittance ``|t|^2 * Re(Y_sub)/Re(Y_super)`` (physical,
        angle/index corrected; matches ``tmm`` on the real axis).  ``T`` is real only for lossless
        end media -- ``float(...)`` of a tiny imaginary residue.
    M11 : complex
        The (0,0) entry of the total transfer matrix (``t = 1/M11``).  Its zeros ARE the scattering
        poles -- but do NOT feed it to :func:`find_poles` (audit Q-12).  ``M11`` is built from the
        amplitude-transfer matrices, whose entries carry an EXPLICIT ``exp(+-i kz d)`` in each
        finite layer, so it inherits that layer's ``kz = sqrt(...)`` BRANCH CUT; when a search-box
        edge crosses the cut ``arg(M11)`` jumps by ~pi and the argument-principle winding
        miscounts.  Use :func:`smatrix_pole_func` (the Abeles CHARACTERISTIC-matrix denominator
        ``D(omega)``, even in every layer ``kz`` and therefore branch-cut-free -- see
        :func:`_stack_denominator`) or :func:`nonlocal_tmm.pole_function` as the pole function.
        ``M11`` is exported for amplitude bookkeeping and real-axis diagnostics.
    S : np.ndarray
        The full 2x2 scattering matrix ``[[r_top, t_bottom], [t_top, r_bottom]]`` (all four
        entries share the ``M11 = 0`` pole).
    """

    r: complex
    t: complex
    R: float
    T: float
    M11: complex
    S: np.ndarray


def layered_smatrix_complex(omega_rad_s, layers: Sequence[Layer], *, theta_rad: float = 0.0,
                            pol: str = "s", n_super=1.0, n_sub=1.0, k_par_m=None) -> SMatrix:
    """Complex-omega 2x2 scattering matrix of ``super | layers | sub``.

    Parameters
    ----------
    omega_rad_s : complex or float
        Angular frequency [rad/s], possibly complex (the evaluator is analytic in it).
    layers : sequence of (eps, thickness_m)
        Ordered from the superstrate side to the substrate side.  ``eps`` is a complex constant or
        a callable ``eps(omega)`` (e.g. ``lambda w: drude_eps(w, ...)``).
    theta_rad : float
        Incidence angle in the superstrate.  Sets ``k_par = n_super * (omega/c) * sin(theta)`` at
        the ``omega`` PASSED IN -- correct for a single real-axis evaluation.  For complex-omega
        pole tracking pass an explicit ``k_par_m`` instead so ``k_par`` stays FIXED (the QNM
        convention); mixing ``theta_rad`` with complex ``omega`` would let ``k_par`` drift.
    pol : {'s', 'p'}
        Polarization (TE / TM).
    n_super, n_sub : complex
        Semi-infinite superstrate/substrate refractive INDICES (eps = n**2).
    k_par_m : float or complex, optional
        Explicit in-plane wavevector [1/m]; overrides ``theta_rad``.  Held fixed => the physically
        correct continuation for QNM/pole work.

    Returns
    -------
    SMatrix
    """
    if pol not in ("s", "p"):
        pol = _accept_pol(pol, "layered_smatrix_complex")    # audit V-3/V-8: one vocabulary, one message
    omega = complex(omega_rad_s)
    k0 = omega / C_LIGHT
    eps_super = complex(n_super) ** 2
    eps_sub = complex(n_sub) ** 2
    if k_par_m is None:
        k_par = complex(n_super) * (omega / C_LIGHT) * math.sin(theta_rad)
    else:
        k_par = complex(k_par_m)

    kz_super = _kz(eps_super, k0, k_par)
    kz_sub = _kz(eps_sub, k0, k_par)
    Y_super = _admittance(eps_super, kz_super, pol)
    Y_sub = _admittance(eps_sub, kz_sub, pol)

    eps_layers = [_eval_eps(e, omega) for e, _ in layers]
    kz_layers = [_kz(e, k0, k_par) for e in eps_layers]
    Y_layers = [_admittance(e, kz, pol) for e, kz in zip(eps_layers, kz_layers)]

    Ylist = [Y_super] + Y_layers + [Y_sub]
    N = len(layers)

    # Scalar 2x2 cascade, same association as the numpy `M @ P @ I` it replaces: `(M @ P) @ I`
    # per layer (audit P-6).  `P` is diagonal, so `M @ P` is a column scaling.
    M11, M12, M21, M22 = _interface(Ylist[0], Ylist[1])
    for j in range(N):
        p00, p11 = _propagate(kz_layers[j], float(layers[j][1]))
        M11, M12, M21, M22 = _mat2_mul((M11 * p00, M12 * p11, M21 * p00, M22 * p11),
                                       _interface(Ylist[j + 1], Ylist[j + 2]))

    detM = M11 * M22 - M12 * M21

    r = M21 / M11
    t = 1.0 / M11
    R = float(abs(r) ** 2)
    # Power transmittance: |t|^2 times the outgoing/incoming admittance ratio (real parts).
    denom = Y_super.real
    T = float(abs(t) ** 2 * (Y_sub.real / denom)) if denom != 0.0 else float("nan")
    S = np.array([[M21 / M11, detM / M11],
                  [1.0 / M11, -M12 / M11]], dtype=np.complex128)
    return SMatrix(r=r, t=t, R=R, T=T, M11=M11, S=S)


def k_par_from_angle(n_super, omega_ref_rad_s, theta_rad) -> float:
    """Fixed in-plane wavevector ``k_par = Re(n_super) * (omega_ref/c) * sin(theta)`` [1/m] to hold
    during complex-omega continuation.  Evaluate ONCE at the real carrier frequency ``omega_ref``
    (e.g. the resonance's real part), then pass it as ``k_par_m`` everywhere -- the QNM convention.
    ``n_super`` must be a lossless (real) incidence index for ``theta`` to be a real angle."""
    return float(np.real(n_super) * (omega_ref_rad_s / C_LIGHT) * math.sin(theta_rad))


def _stack_denominator(omega: complex, layers: Sequence[Layer], pol: str, n_super, n_sub,
                       k_par) -> complex:
    """Analytic scattering-pole function ``D(omega)`` whose zeros are the scattering poles -- the
    function fed to the finder / Newton.

    Built from the Abeles CHARACTERISTIC (field-transfer) matrix.  Each layer contributes

        m_j = [[cos(phi_j),         -i sin(phi_j) / Y_j],
               [-i Y_j sin(phi_j),   cos(phi_j)        ]],   phi_j = kz_j * d_j

    (the ``-i`` signs are the ``exp(-i*omega*t)`` / forward ``exp(+i*kz*z)`` convention -- the
    opposite Macleod ``+i`` matrix would place poles in the UPPER half plane, growing modes).  With
    the total ``M_c = m_1 ... m_N`` and the outgoing end-media admittances ``Y_super``, ``Y_sub``,
    the reflection is ``r = (Y_super*B - C)/(Y_super*B + C)`` where ``B = M_c[0,0] + M_c[0,1]*Y_sub``,
    ``C = M_c[1,0] + M_c[1,1]*Y_sub``; the scattering pole is

        D(omega) = Y_super * B + C = 0.

    Why the characteristic matrix and NOT ``M11 = 1/t`` or the Airy cascade: every ``m_j`` entry is
    ``cos(kz d)``, ``sin(kz d)/Y`` or ``Y sin(kz d)`` -- ALL EVEN in the layer ``kz`` (``sin`` and
    ``Y ~ kz`` are both odd), hence functions of ``kz^2 = eps*k0^2 - k_par^2``, which is a
    polynomial in ``omega`` with NO square-root branch cut.  A pole function carrying an explicit
    ``kz`` (``M11``, or ``exp(2 i kz d)`` in the Airy cascade) inherits the finite-layer branch cut;
    when a search-box edge crosses it ``arg(D)`` jumps by ~pi and the argument-principle winding
    miscounts.  ``D`` here is analytic in ``omega`` except for the branch points of the SEMI-INFINITE
    end media (``Y_super``, ``Y_sub`` at their light lines ``kz_end = 0``) -- which are physical and
    kept outside a well-placed box -- and, for p-polarization, a SIMPLE POLE at a layer ENZ
    crossing ``eps_j = 0`` (from ``1/Y_p = kz/eps``); either position p-pol boxes to exclude that
    point, or clear it by multiplying ``D`` by ``eps_j(omega)`` (the same trick
    ``nonlocal_tmm.pole_function`` uses for its csc poles -- see :func:`berreman_enz_pole`,
    where the genuine Berreman zero sits right next to the ENZ point and the cleared form is
    essential)."""
    k0 = omega / C_LIGHT
    kpar = complex(k_par)
    eps_super = complex(n_super) ** 2
    eps_sub = complex(n_sub) ** 2
    Y_super = _admittance(eps_super, _kz(eps_super, k0, kpar), pol)
    Y_sub = _admittance(eps_sub, _kz(eps_sub, k0, kpar), pol)

    # Scalar 2x2 cascade (audit P-6), same order as the numpy `Mc = Mc @ m` it replaces.
    # np.cos / np.sin are KEPT -- cmath's are not bit-identical to them -- and their
    # np.complex128 results are round-tripped exactly through complex() so the rest of the
    # product runs in CPython scalar arithmetic.
    a, b, c_, d_ = 1.0 + 0j, 0.0 + 0j, 0.0 + 0j, 1.0 + 0j
    for e_spec, d in layers:
        eps = _eval_eps(e_spec, omega)
        kz = _kz(eps, k0, kpar)
        Y = _admittance(eps, kz, pol)
        phi = kz * float(d)
        c = complex(np.cos(phi))
        s = complex(np.sin(phi))
        a, b, c_, d_ = _mat2_mul((a, b, c_, d_), (c, -1j * s / Y, -1j * Y * s, c))

    B = a + b * Y_sub
    C = c_ + d_ * Y_sub
    return complex(Y_super * B + C)


def smatrix_pole_func(layers: Sequence[Layer], *, pol: str = "s", n_super=1.0, n_sub=1.0,
                      k_par_m=0.0) -> Callable[[complex], complex]:
    """Return the analytic scattering-pole function ``D(omega)`` (a closure holding ``k_par``
    FIXED), whose zeros are the scattering poles of the stack.  Feed this to :func:`find_poles` /
    :func:`newton_refine`.  See :func:`_stack_denominator` for why the characteristic-matrix form
    is used (branch-cut-free in the layer wavevectors, correct decaying-pole sign).

    ``pol`` is validated EAGERLY against the same ``{'s', 'p'}`` set
    :func:`layered_smatrix_complex` accepts (audit V-3: this entry point -- the module's primary
    public pole finder -- used to accept anything and silently return the P-POL function, so a
    ``pol='TE'`` pole hunt returned p-pol poles that look entirely plausible; validating in the
    factory, not in the closure, surfaces the typo at the call site rather than inside
    ``find_poles``)."""
    if pol not in ("s", "p"):
        pol = _accept_pol(pol, "smatrix_pole_func")

    def D(omega):
        return _stack_denominator(omega, layers, pol, n_super, n_sub, k_par_m)
    return D


# ------------------------------------------------------------------------------------------------
# Newton refinement and Q
# ------------------------------------------------------------------------------------------------
def pole_q(omega_tilde) -> float:
    """Quality factor of a complex pole ``omega_tilde = omega_0 - i*gamma/2``:

        Q = |Re(omega_tilde)| / (2 |Im(omega_tilde)|) .

    Returns ``+inf`` for a real (lossless, undamped) pole.

    THE single Q convention across the resonance tooling: ``optics.aaa_poles.q_from_pole`` is a
    thin alias of THIS function, not a second implementation (audit X-5 -- the two used to be
    byte-identical copies whose docstrings had already drifted, this one omitting the ``|.|`` the
    code has always applied to ``Re``).  The absolute value matters: a pole finder working on the
    lower half-plane can hand back the ``-conj`` partner, and Q is positive for both."""
    w = complex(omega_tilde)
    im = abs(w.imag)
    if im == 0.0:
        return float("inf")
    return abs(w.real) / (2.0 * im)


def newton_refine(func: Callable[[complex], complex], z0, *, tol: float = 1e-11,
                  maxiter: int = 100, h_rel: float = 1e-7,
                  require_convergence: bool = False) -> complex:
    """Newton's method on an analytic ``func`` with a central-difference derivative (the material
    models are analytic but not necessarily cheap to differentiate in closed form).  ``tol`` is the
    RELATIVE step-size stopping criterion on ``omega``.

    Returns the LAST iterate.  NOTE (finding Q-4): that is *not* a converged root -- there is no
    residual or basin test here, and for a ``func`` that never vanishes the last iterate is
    meaningless (a ``func`` identically 1.0 returns ~1e15 - 1e13j).  Callers that need a *validated*
    root must test the result themselves (see :func:`q_budget`'s proximity + residual gate) or pass
    ``require_convergence=True``, which raises ``ValueError`` when the relative step-size criterion
    was never met within ``maxiter``."""
    z = complex(z0)
    converged = False
    for _ in range(maxiter):
        f = func(z)
        if f == 0.0:
            converged = True
            break
        h = h_rel * max(abs(z), 1.0)
        fp = (func(z + h) - func(z - h)) / (2.0 * h)
        if fp == 0.0 or not np.isfinite(fp):
            break
        dz = f / fp
        z = z - dz
        if abs(dz) <= tol * max(abs(z), 1.0):
            converged = True
            break
    if require_convergence and not converged:
        raise ValueError(
            "newton_refine: no convergence to a relative step of {:g} in {} iterations from z0 = "
            "{!r} (last iterate {!r}); the seed is likely outside the root's basin.".format(
                tol, maxiter, complex(z0), z))
    return z


# ------------------------------------------------------------------------------------------------
# Argument-principle pole finder
# ------------------------------------------------------------------------------------------------
def _rect_boundary_points(rect: Tuple[float, float, float, float], n: int) -> List[complex]:
    """Corner-to-corner boundary of a rectangle (re0, re1, im0, im1), ``n`` points per edge,
    traversed counter-clockwise (closed loop, no duplicated corners)."""
    re0, re1, im0, im1 = rect
    re = np.linspace(re0, re1, n, endpoint=False)
    im = np.linspace(im0, im1, n, endpoint=False)
    pts = []
    pts += [complex(x, im0) for x in re]                       # bottom, left->right
    pts += [complex(re1, y) for y in im]                       # right, bottom->top
    pts += [complex(x, im1) for x in re[::-1]]                 # top, right->left
    pts += [complex(re0, y) for y in im[::-1]]                 # left, top->bottom
    return pts


def _winding(func: Callable[[complex], complex], rect: Tuple[float, float, float, float],
             n: int) -> Tuple[float, float]:
    """Winding number (1/2pi) * closed-contour change of arg(func) around ``rect``, plus the max
    single-step |delta arg| (an under-sampling diagnostic).  For an analytic ``func`` with no poles
    of its own this equals the number of ZEROS enclosed (argument principle)."""
    pts = _rect_boundary_points(rect, n)
    vals = np.array([func(p) for p in pts], dtype=np.complex128)
    acc = 0.0
    maxstep = 0.0
    m = len(vals)
    for k in range(m):
        a = vals[k]
        b = vals[(k + 1) % m]
        d = math.atan2((b / a).imag, (b / a).real) if a != 0.0 else 0.0
        acc += d
        maxstep = max(maxstep, abs(d))
    return acc / (2.0 * math.pi), maxstep


# Winding-trust thresholds, ONE definition (they used to be bare literals repeated at three call
# sites).  `_MAXSTEP_TRUST`: the largest single-step |delta arg| a boundary sampling may show and
# still be believed -- past ~1.2 rad the contour is either under-sampled or straddling a zero, and
# a ~pi jump aliases as its own negative.  `_WINDING_INT_TOL`: how far the raw winding may sit from
# the nearest integer before the count is called non-integer (the argument principle returns an
# exact integer for a well-sampled contour with no boundary zero).
_MAXSTEP_TRUST = 1.2
_WINDING_INT_TOL = 0.25


def _winding_densified(func: Callable[[complex], complex],
                       rect: Tuple[float, float, float, float],
                       n_grid: int) -> Tuple[float, float]:
    """:func:`_winding` with adaptive boundary densification (doubling up to 16x while any
    single phase step exceeds ``_MAXSTEP_TRUST`` ~ 1.2 rad).  Returns ``(w, maxstep)`` at the final
    density.  A residual ``maxstep > _MAXSTEP_TRUST`` after densification flags an UNTRUSTWORTHY
    count -- typically a zero lying on (or hugging) the contour, whose ~pi phase jump no sampling
    density removes.  :func:`find_poles` checks this at EVERY box, root included (audit Q-11)."""
    w, maxstep = _winding(func, rect, n_grid)
    ng = n_grid
    while maxstep > _MAXSTEP_TRUST and ng < n_grid * 16:
        ng *= 2
        w, maxstep = _winding(func, rect, ng)
    return w, maxstep


# Quad-tree split fractions tried in order.  0.5 first (the natural bisection); the other two are
# irrational offsets used when the parent-vs-children count-consistency check fails -- a pole
# sitting ON a dividing line corrupts both children's windings, and shifting the line by an
# irrational fraction of the box is guaranteed to move it off any such pole.
_SPLIT_FRACS = (0.5,
                0.5 + 0.5 * (math.sqrt(5.0) - 2.0),      # ~0.618 (golden section)
                0.5 - 0.25 * (math.sqrt(2.0) - 1.0))     # ~0.396

# find_poles' per-call evaluation memo (audit P-14).  Entry cap: a (float, float) key plus a
# complex value costs ~200 B, so 200k entries bound the memo at ~40 MB even if a caller drives an
# extreme n_grid / max_depth.  Past the cap the finder simply stops caching -- the values it
# returns do not change, only the hit rate.  `_MEMO_MISS` is a private sentinel so that a
# `func_of_omega` legitimately returning None is not mistaken for a cache miss.
_POLE_MEMO_MAX = 200_000
_MEMO_MISS = object()


def _interior_seed(func: Callable[[complex], complex],
                   rect: Tuple[float, float, float, float], n: int) -> complex:
    """Seed Newton at the interior grid point of least |func| (a coarse basin locator)."""
    re0, re1, im0, im1 = rect
    re = np.linspace(re0, re1, n + 2)[1:-1]
    im = np.linspace(im0, im1, n + 2)[1:-1]
    best = complex(0.5 * (re0 + re1), 0.5 * (im0 + im1))
    best_val = float("inf")
    for x in re:
        for y in im:
            z = complex(x, y)
            v = abs(func(z))
            if v < best_val:
                best_val = v
                best = z
    return best


def _inside(z: complex, rect: Tuple[float, float, float, float], pad: float = 0.5) -> bool:
    re0, re1, im0, im1 = rect
    wr = (re1 - re0) * pad
    wi = (im1 - im0) * pad
    return (re0 - wr) <= z.real <= (re1 + wr) and (im0 - wi) <= z.imag <= (im1 + wi)


_ON_UNTRUSTED = ("warn", "raise", "ignore")


def _check_on_untrusted(on_untrusted, where: str) -> str:
    """Validate the shared ``on_untrusted`` vocabulary, ONE definition for every entry point that
    has a trust signal to report (audit Q-11)."""
    if on_untrusted not in _ON_UNTRUSTED:
        raise ValueError("{}: on_untrusted must be one of {}; got {!r}".format(
            where, _ON_UNTRUSTED, on_untrusted))
    return on_untrusted


def _report_untrusted(msg: str, on_untrusted: str, stacklevel: int = 3) -> None:
    """Emit ``msg`` under the caller's ``on_untrusted`` policy: warn / raise / ignore."""
    if on_untrusted == "ignore":
        return
    if on_untrusted == "raise":
        raise RuntimeError(msg)
    warnings.warn(msg, RuntimeWarning, stacklevel=stacklevel)


# How far the root box is shrunk / grown for the boundary re-check (audit Q-11 residual): a
# relative perturbation of each half-extent.  A zero closer than this to the boundary is treated
# as STRADDLING it -- the count it contributes is ambiguous, which is exactly the failure the
# diagnostic exists for.  Small enough that no interior pole changes side.
_BOX_RECHECK_REL = 1e-6


def _scaled_rect(rect: Tuple[float, float, float, float],
                 factor: float) -> Tuple[float, float, float, float]:
    """``rect`` scaled about its own centre by ``factor`` in both axes."""
    re0, re1, im0, im1 = rect
    cr, ci = 0.5 * (re0 + re1), 0.5 * (im0 + im1)
    hr, hi = 0.5 * (re1 - re0) * factor, 0.5 * (im1 - im0) * factor
    return (cr - hr, cr + hr, ci - hi, ci + hi)


def _boundary_is_clear(func: Callable[[complex], complex],
                       rect: Tuple[float, float, float, float],
                       n_grid: int, count: int) -> bool:
    """Does the root box's zero COUNT survive a hair's shrink and a hair's growth?

    This is the second, independent signal the Q-11 corroboration was missing.  A large residual
    ``maxstep`` has two completely different causes: a zero sitting ON (or hugging) the contour,
    whose ~pi phase jump aliases and makes the count meaningless -- and a contour that legitimately
    sweeps a lot of phase between samples with every zero comfortably inside.  Matching the refined
    pole count cannot tell them apart, because a boundary zero is usually FOUND by the quad-tree
    too, so the counts agree by coincidence (measured: 7 of 12 boundary-straddling geometries were
    silenced this way, including a single zero on the right edge -- winding 1.0, maxstep 3.14, one
    pole found, no warning).

    Shrinking and growing the box by ``_BOX_RECHECK_REL`` separates them: a zero within that
    distance of the boundary lands on OPPOSITE sides of the two contours, so the two counts differ
    and the answer is declared unclear.  A zero well inside does not move side, and the two counts
    agree however wild ``maxstep`` is.  ``maxstep`` is deliberately NOT re-tested here -- the
    legitimate high-maxstep contour (the nonlocal_tmm bulk-plasmon box runs at ~pi) must pass."""
    for factor in (1.0 - _BOX_RECHECK_REL, 1.0 + _BOX_RECHECK_REL):
        w, _ms = _winding_densified(func, _scaled_rect(rect, factor), n_grid)
        c = int(round(w))
        if c != count or abs(w - c) > _WINDING_INT_TOL:
            return False
    return True


# Message for an untrustworthy ROOT-box winding (audit Q-11).  Formatted with the rectangle, the
# raw winding and the residual max phase step so the caller can act on it rather than guess.
_UNTRUSTED_ROOT_MSG = (
    "find_poles: the winding count on the SEARCH BOX ITSELF is untrustworthy and the {n} pole(s) "
    "actually located do not corroborate it (raw winding {w:.6f}, residual max phase step "
    "{ms:.3f} rad, trust threshold {thr}) on rect re=[{r0:.6g}, {r1:.6g}] "
    "im=[{i0:.6g}, {i1:.6g}]. A zero lying ON or hugging the box boundary makes its ~pi phase "
    "jump alias, so the count -- and hence this result, INCLUDING an empty list -- may be wrong: "
    "an empty list here does NOT mean 'no poles'. Move or grow the box (an irrational fraction "
    "of its own span is guaranteed to shift the boundary off the offending zero), or raise "
    "n_grid. Pass on_untrusted='raise' to make this fatal, or 'ignore' to silence it.")

# A zero STRADDLES the search-box boundary: the winding was untrustworthy, the refined pole count
# happened to match it, but shrinking and growing the box by a hair gives two DIFFERENT counts --
# so the match was a coincidence and the boundary zero's membership is undecided (audit Q-11
# residual).
_STRADDLE_ROOT_MSG = (
    "find_poles: a zero STRADDLES the search-box boundary. The winding count on the box itself is "
    "untrustworthy (raw winding {w:.6f}, residual max phase step {ms:.3f} rad, trust threshold "
    "{thr}) and although the {n} pole(s) located match it, that agreement does not corroborate "
    "anything here: shrinking and growing the box by {rel:g} of its own half-extents gives "
    "DIFFERENT counts, which only a zero on (or within that distance of) the contour can do. "
    "Rect re=[{r0:.6g}, {r1:.6g}] im=[{i0:.6g}, {i1:.6g}]. Whether that zero is inside the box is "
    "undecided, so both the count and the returned list may be off by it. Move or grow the box "
    "(an irrational fraction of its own span is guaranteed to shift the boundary off the "
    "offending zero). Pass on_untrusted='raise' to make this fatal, or 'ignore' to silence it.")

# The winding was TRUSTWORTHY and integral, and it counts MORE zeros than the quad-tree refined
# (audit Q-11 residual).  The old check only looked at untrustworthy windings, so this -- the
# argument principle and the search disagreeing while both are believable -- was silent.
_MISSING_POLES_MSG = (
    "find_poles: the argument principle counts {w} zero(s) in the search box (winding {wraw:.6f}, "
    "residual max phase step {ms:.3f} rad -- a TRUSTWORTHY count) but only {n} distinct pole(s) "
    "were refined out of it, on rect re=[{r0:.6g}, {r1:.6g}] im=[{i0:.6g}, {i1:.6g}]. The "
    "returned list is INCOMPLETE unless the missing zeros are degenerate or closer together than "
    "dedup_rel={dd:g} (the winding counts with MULTIPLICITY; this finder returns distinct "
    "positions). Otherwise raise n_grid or max_depth, or split the box. Pass on_untrusted='raise' "
    "to make this fatal, or 'ignore' to silence it.")


def find_poles(func_of_omega: Callable[[complex], complex], omega_center, omega_span, *,
               n_grid: int = 40, refine_tol: float = 1e-11, max_depth: int = 8,
               dedup_rel: float = 1e-6, on_untrusted: str = "warn") -> List[complex]:
    """Locate the poles of a scattering response (the zeros of ``func_of_omega``, e.g. ``D(omega)``
    from :func:`smatrix_pole_func`) inside a rectangle of the complex-omega plane, via the argument
    principle + Newton refinement.

    Parameters
    ----------
    func_of_omega : callable
        Analytic function whose ZEROS are the sought poles.  Build it with
        :func:`smatrix_pole_func` or :func:`nonlocal_tmm.pole_function`.

        It must be ANALYTIC (branch-cut-free) inside and on the search box -- the argument
        principle counts nothing else.  In particular do NOT pass ``SMatrix.M11`` or a
        ``1/S``/Airy-cascade form (audit Q-12): those carry an explicit ``exp(+-i kz d)`` per
        finite layer and so inherit that layer's ``kz`` branch cut, and a box edge crossing the
        cut makes ``arg`` jump by ~pi and the winding miscount.  The characteristic-matrix
        denominator :func:`smatrix_pole_func` returns is even in every layer ``kz`` and has the
        same zeros with no such cut -- see :func:`_stack_denominator` for the derivation and for
        the two non-analyticities that DO survive (the semi-infinite end-media light lines, and
        the p-polarized simple pole at a layer ENZ crossing).

        It must also be DETERMINISTIC in ``omega``: repeated evaluations at the same point are
        served from a per-call memo (see below).
    omega_center : complex
        Centre of the search rectangle.  Give it a NEGATIVE imaginary part (or a tall enough span)
        to bracket decaying poles (``Im < 0``).
    omega_span : complex or float
        Half-extents of the rectangle: ``Re`` half-width = ``|Re(omega_span)|``, ``Im`` half-width
        = ``|Im(omega_span)|``.  A real scalar makes a square box.
    n_grid : int
        Boundary samples per edge for the winding integral, and interior seed-grid resolution.
        Doubled adaptively when the boundary is under-sampled.
    refine_tol : float
        Relative Newton tolerance.
    max_depth : int
        Max quad-tree subdivision depth (guards pathological non-isolation).
    dedup_rel : float
        Relative tolerance for merging duplicate poles found in adjacent sub-boxes.
    on_untrusted : {'warn', 'raise', 'ignore'}
        What to do when the ROOT (caller-supplied) box's zero count cannot be trusted, in EITHER
        direction -- see the ROOT-BOX DIAGNOSTIC note below.  ``'warn'`` (default) emits a
        ``RuntimeWarning`` and returns whatever was found; ``'raise'`` raises ``RuntimeError``;
        ``'ignore'`` restores the pre-Q-11 silence.  The returned poles are IDENTICAL in all three
        modes.  The same keyword, vocabulary and default are accepted by :func:`track_pole`,
        :func:`q_budget` and :func:`berreman_enz_pole`.

    Returns
    -------
    list of complex
        Refined pole positions (unordered), each a zero of ``func_of_omega`` inside the box.

    Notes
    -----
    ROOT-BOX DIAGNOSTIC (audit Q-11).  :func:`_winding_densified` returns its own
    "is this count believable" signal -- the residual max single-step ``|delta arg|`` after
    densification -- and the quad-tree has always checked it on every CHILD box before accepting a
    split.  It was DROPPED on the root box, the one the caller actually asked about.  A pole
    sitting on the search-box boundary makes the raw winding come out ~0.0 with a residual step of
    ~pi/2 or more, and ``count <= 0`` returned an empty list: indistinguishable from the honest
    "the box is empty" answer, with nothing for the caller to test.  The root count is now
    validated on both signals (``maxstep <= _MAXSTEP_TRUST`` and a winding within
    ``_WINDING_INT_TOL`` of an integer), CORROBORATED against the number of poles the quad-tree
    actually refined, and reported through ``on_untrusted``.  The corroboration matters: a
    contour can legitimately run at ``maxstep ~ pi`` with an exactly integral winding, and when
    the refined pole count matches it the two independent signals agree.

    The check is TWO-SIDED (Q-11 residual).  Matching counts is not by itself corroboration, and
    a trustworthy winding is not by itself a complete answer:

      * UNTRUSTWORTHY winding, count MATCHED.  A zero on the boundary is usually FOUND by the
        quad-tree as well, so the counts agree by coincidence and the diagnostic went silent --
        measured on 12 boundary geometries, 7 were silenced this way (a single zero on the right
        edge: winding 1.0, maxstep 3.14, one pole found, no warning; "2 inside + 1 on the top
        edge": winding 2.0, maxstep 3.14, two found, no warning).  The count is now RE-CHECKED by
        shrinking and growing the box by ``_BOX_RECHECK_REL`` of its own half-extents
        (:func:`_boundary_is_clear`): a zero within that distance of the contour lands on opposite
        sides of the two and the counts differ, while every interior zero keeps its side however
        large ``maxstep`` is.  Only a CLEAR re-check silences the report.
      * TRUSTWORTHY, integral winding, count SHORT.  The argument principle says N, the quad-tree
        refines M < N: the returned list is incomplete and nothing said so (measured: a box with
        8 simple zeros, winding exactly 8.0 at maxstep 0.49, returned 6).  This now reports too.
        The winding counts with MULTIPLICITY while this finder returns DISTINCT positions, so a
        genuinely degenerate or sub-``dedup_rel`` cluster reports here as well -- correctly: the
        list is short of the count either way, and the message says so.

    This is a pure diagnostic: the search, the recursion and the returned poles are bit-for-bit
    unchanged, and the two extra winding evaluations of the re-check are only ever paid on the
    ``bad and corroborated`` branch.

    PER-CALL EVALUATION MEMO (audit P-14).  About 63 % of the evaluations this finder requests are
    at points it has ALREADY evaluated: ``np.linspace(a, b, 2n, endpoint=False)`` CONTAINS
    ``np.linspace(a, b, n, endpoint=False)`` exactly (the step halves, so every coarse point is an
    even-indexed fine point), so each densification level inside :func:`_winding_densified`
    re-walks the previous one bit-for-bit, and the quad-tree re-walks corner/edge points shared
    between a parent box, its candidate splits and its children.  Since ``func_of_omega`` is a
    deterministic function of ``omega``, a dict memo on the exact float pair returns literally the
    same bits: measured **1.90x** with an exactly identical pole set (121771 requests, 76644 hits,
    45127 unique evaluations) on a pole-dense hydrodynamic box.  Two properties are load-bearing:
    the memo lives for ONE call (a module-level cache would serve values across different
    ``layers`` / ``k_par`` / ``loss_scale`` closures), and it keys on the EXACT ``(real, imag)``
    pair -- never a rounded or tolerance key.  It stops growing past ``_POLE_MEMO_MAX`` entries
    (correctness is unaffected; only the hit rate drops), and a point with a zero or NaN component
    bypasses it entirely (``0.0`` and ``-0.0`` are the same dict key but need not be the same side
    of a branch cut; NaN never matches itself).
    """
    _check_on_untrusted(on_untrusted, "find_poles")
    span = complex(omega_span)
    sr = abs(span.real) if span.real != 0.0 else abs(span.imag)
    si = abs(span.imag) if span.imag != 0.0 else abs(span.real)
    c = complex(omega_center)
    root_rect = (c.real - sr, c.real + sr, c.imag - si, c.imag + si)

    found: List[complex] = []
    memo: dict = {}
    miss = _MEMO_MISS
    root_diag: List[Tuple[float, float]] = []

    def func(z):
        """`func_of_omega` memoized on the exact (real, imag) pair -- see the Notes above."""
        zr, zi = z.real, z.imag
        if zr == 0.0 or zi == 0.0 or zr != zr or zi != zi:
            return func_of_omega(z)                # +-0.0 alias / NaN: never keyed
        key = (zr, zi)
        v = memo.get(key, miss)
        if v is miss:
            v = func_of_omega(z)
            if len(memo) < _POLE_MEMO_MAX:
                memo[key] = v
        return v

    def newton_in(rect):
        seed = _interior_seed(func, rect, max(6, n_grid // 4))
        z = newton_refine(func, seed, tol=refine_tol)
        if _inside(z, rect) and np.isfinite(z):
            found.append(z)

    def recurse(rect, depth, count=None):
        if count is None:
            w, ms = _winding_densified(func, rect, n_grid)
            if depth == 0:                         # the caller's own box: keep the diagnostic
                root_diag.append((w, ms))
            count = int(round(w))
        if count <= 0:
            return
        re0, re1, im0, im1 = rect
        tiny = (re1 - re0) < dedup_rel * max(abs(re0), abs(re1), 1.0)
        if count == 1 or depth >= max_depth or tiny:
            newton_in(rect)
            return
        # Subdivide into 4 quadrants -- with a VALIDATED split.  A pole lying ON a dividing line
        # (e.g. a box centred exactly on a pole, the natural user call) corrupts both adjacent
        # children's winding integrals: the ~pi phase step across the boundary zero survives any
        # sampling density, and the pole is silently dropped.  So each candidate split must have
        # (i) every child boundary well-sampled after densification (maxstep <= 1.2), (ii) every
        # child winding a clean integer, and (iii) the children counts SUMMING to the parent
        # count.  On failure the dividing lines move to an irrational fraction of the box
        # (guaranteed off the offending pole) and the check repeats.
        for frac in _SPLIT_FRACS:
            rm = re0 + frac * (re1 - re0)
            imm = im0 + frac * (im1 - im0)
            subs = ((re0, rm, im0, imm), (rm, re1, im0, imm),
                    (re0, rm, imm, im1), (rm, re1, imm, im1))
            child_counts = []
            ok = True
            for sub in subs:
                ws, ms = _winding_densified(func, sub, n_grid)
                cs = int(round(ws))
                if ms > _MAXSTEP_TRUST or abs(ws - cs) > _WINDING_INT_TOL:
                    ok = False                     # a zero sits on / hugs this child boundary
                    break
                child_counts.append(cs)
            if ok and sum(child_counts) == count:
                for sub, cs in zip(subs, child_counts):
                    recurse(sub, depth + 1, count=cs)
                return
        # No split offset yielded a fully-validated partition -- a pole hugs every candidate
        # dividing line (pole-DENSE box, e.g. a bulk-plasmon comb). Fall back to plain bisection
        # with per-child re-counting: each child's own boundaries move again as it subdivides,
        # so deeper recursion recovers isolated poles best-effort (the pre-fix behaviour) --
        # far better than collapsing the whole box onto a single Newton seed.
        rm = 0.5 * (re0 + re1)
        imm = 0.5 * (im0 + im1)
        for sub in ((re0, rm, im0, imm), (rm, re1, im0, imm),
                    (re0, rm, imm, im1), (rm, re1, imm, im1)):
            recurse(sub, depth + 1)

    recurse(root_rect, 0)

    # Deduplicate (adjacent boxes can each converge to a shared boundary pole).
    uniq: List[complex] = []
    for z in found:
        if all(abs(z - u) > dedup_rel * max(abs(z), 1.0) for u in uniq):
            uniq.append(z)

    # Q-11: report the root-box winding diagnostic the finder has always computed and discarded.
    # AFTER the search, so the poles are assembled identically whatever `on_untrusted` says, and
    # so the located poles can CORROBORATE the count: if the argument principle said N and the
    # quad-tree then refined N distinct roots inside the box, the two independent signals agree
    # and a large residual phase step is a property of the contour, not a lost pole (measured: the
    # nonlocal_tmm bulk-plasmon gate runs at maxstep ~ pi with an exactly integral winding of 2
    # and returns exactly 2 poles).  Without corroboration -- and in particular for the EMPTY
    # result, the outcome the finding is about -- the count is reported as untrustworthy.
    if root_diag and on_untrusted != "ignore":
        w_root, ms_root = root_diag[0]
        count_root = int(round(w_root))
        bad = ms_root > _MAXSTEP_TRUST or abs(w_root - count_root) > _WINDING_INT_TOL
        corroborated = count_root > 0 and len(uniq) == count_root
        rect_fmt = dict(r0=root_rect[0], r1=root_rect[1], i0=root_rect[2], i1=root_rect[3])
        if bad and not corroborated:
            _report_untrusted(_UNTRUSTED_ROOT_MSG.format(
                w=w_root, ms=ms_root, thr=_MAXSTEP_TRUST, n=len(uniq), **rect_fmt), on_untrusted)
        elif bad:
            # Corroborated -- but corroboration is only worth something once the OTHER cause of a
            # bad signal is excluded. Re-check the boundary (Q-11 residual).
            if not _boundary_is_clear(func, root_rect, n_grid, count_root):
                _report_untrusted(_STRADDLE_ROOT_MSG.format(
                    w=w_root, ms=ms_root, thr=_MAXSTEP_TRUST, n=len(uniq),
                    rel=_BOX_RECHECK_REL, **rect_fmt), on_untrusted)
        elif count_root > len(uniq):
            # The winding IS trustworthy and it counts more zeros than were refined. Silent
            # before: the check only ever looked at UNtrustworthy windings.
            _report_untrusted(_MISSING_POLES_MSG.format(
                w=count_root, wraw=w_root, ms=ms_root, n=len(uniq), dd=dedup_rel, **rect_fmt),
                on_untrusted)
    return uniq


# ------------------------------------------------------------------------------------------------
# Parameter tracking (continuation)
# ------------------------------------------------------------------------------------------------
def track_pole(solver: Callable[[float], Callable[[complex], complex]], pole0, param_values,
               *, refine_tol: float = 1e-11, jump_rel: float = 0.25,
               max_subdiv: int = 40, on_untrusted: str = "warn") -> List[complex]:
    """Track a single pole across a swept parameter by warm-started Newton continuation.

    Parameters
    ----------
    solver : callable
        ``solver(param_value) -> D`` where ``D(omega)`` is the pole function (zeros = poles) at that
        parameter value (build it with :func:`smatrix_pole_func`).
    pole0 : complex
        Pole at the first parameter value (a good initial guess; it is re-refined).
    param_values : sequence of float
        The parameter samples, in order.
    jump_rel : float
        If a Newton step from one sample to the next moves the pole by more than this RELATIVE
        amount, the parameter interval is bisected (step halving) and re-tracked -- guards against
        Newton jumping to a neighbouring pole across too-coarse a step.
    max_subdiv : int
        Recursion cap on the bisection.
    on_untrusted : {'warn', 'raise', 'ignore'}
        What to do when the continuation accepts a step it could not validate: the ``jump_rel``
        guard exists precisely because Newton can hop to a NEIGHBOURING pole across a coarse
        parameter step, and at ``depth >= max_subdiv`` the bisection gives up and returns the
        jumped value anyway.  That acceptance used to be silent, so a track that changed branch
        looked exactly like one that did not (audit Q-11: same keyword, vocabulary and default as
        :func:`find_poles`).  ``'warn'`` (default) emits a ``RuntimeWarning`` naming the
        parameter and the size of the jump; ``'raise'`` raises ``RuntimeError``; ``'ignore'``
        restores the earlier silence.  The returned track is IDENTICAL in all three modes.

    Returns
    -------
    list of complex
        The tracked pole at each parameter value (same length as ``param_values``).
    """
    _check_on_untrusted(on_untrusted, "track_pole")
    params = [float(p) for p in param_values]
    if not params:
        return []

    def step(p_from, z_from, p_to, depth):
        z = newton_refine(solver(p_to), z_from, tol=refine_tol)
        jump = abs(z - z_from)
        if jump <= jump_rel * max(abs(z_from), 1.0):
            return z
        if depth >= max_subdiv:
            _report_untrusted(
                "track_pole: the bisection hit max_subdiv={} and the step is STILL a jump -- the "
                "pole moved by {:.6e} ({:.3g} of |pole|, allowed jump_rel={:g}) between parameter "
                "{:.6g} and {:.6g}. Newton may have landed on a NEIGHBOURING pole, in which case "
                "the rest of this track follows the wrong branch and no later value flags it. "
                "Sample the parameter more finely, raise max_subdiv, or re-seed with find_poles. "
                "Pass on_untrusted='raise' to make this fatal, or 'ignore' to silence "
                "it.".format(max_subdiv, jump, jump / max(abs(z_from), 1.0), jump_rel,
                             p_from, p_to), on_untrusted, stacklevel=4)
            return z
        pm = 0.5 * (p_from + p_to)
        zm = step(p_from, z_from, pm, depth + 1)
        return step(pm, zm, p_to, depth + 1)

    out = [newton_refine(solver(params[0]), complex(pole0), tol=refine_tol)]
    for p in params[1:]:
        out.append(step(params[len(out) - 1], out[-1], p, 0))
    return out


# ------------------------------------------------------------------------------------------------
# Radiative / absorptive Q split
# ------------------------------------------------------------------------------------------------
def q_budget(make_pole_func: Callable[[float], Callable[[complex], complex]], pole0, *,
             refine_tol: float = 1e-11, loss_scale: float = 1.0,
             rad_proximity_linewidths: float = 5.0, rad_proximity_rel_floor: float = 1e-3,
             residual_rel: float = 1e-6, degenerate_rel: float = 1e-9,
             on_untrusted: str = "warn") -> dict:
    """Split the total Q of a pole into radiative and absorptive parts by the lossless/lossy
    two-pass (Lalanne et al. 2018).

    ``make_pole_func(loss_scale) -> D`` must return the pole function ``D(omega)`` (zeros = poles)
    with the material LOSSES scaled by ``loss_scale`` (so ``0.0`` is lossless, ``1.0`` the physical
    stack, ``2.0`` double loss).  Build such a factory by scaling the imaginary part of each layer's
    eps, or the Drude ``gamma``.  Keeping it a MATERIAL-level knob preserves analyticity in omega
    (scaling ``Im(eps(omega))`` pointwise for complex omega would not be analytic).

    The lossless pass re-finds the pole with ``loss_scale = 0`` (warm-started from the lossy pole):
    ``Q_rad``.  Then ``1/Q_abs = 1/Q_total - 1/Q_rad``.

    DISPERSIVE-MEDIUM BIAS (finding Q-15).  The two passes are evaluated at two DIFFERENT
    resonance frequencies whenever the loss knob is dispersive.  For a Drude layer
    ``eps = eps_inf - wp^2/(w^2 + i w g)`` the real part is ``eps_inf - wp^2/(w^2 + g^2)``, so
    setting ``g = 0`` moves ``Re(eps)`` by ``wp^2 g^2 / (w^2 (w^2 + g^2))`` and therefore moves
    the pole: the lossless linewidth ``2|Im pole_rad|`` is the radiative linewidth AT
    ``Re(pole_rad)``, not at ``Re(pole_total)``.  Measured on a thin Drude film at
    ``gamma/omega_0 ~ 0.1``: ``Re(pole)`` shifts -0.0605% and the implied ``gamma_abs`` comes out
    2.2% below the true collision rate.  The bias GROWS with ``gamma/omega_0``, so read the
    returned ``omega0_shift_rel`` (the fractional ``Re`` shift between the two passes) as the
    split's own error bar -- it is small exactly when the split is trustworthy.  For a
    NON-dispersive loss knob (scaling a constant ``Im(eps)``) the shift is second order and the
    split is unbiased.

    ENZ / p-POLARIZATION PRECONDITION.  A p-pol pole function over a material with an
    epsilon-near-zero crossing carries a SPURIOUS SIMPLE POLE of ``D`` at ``eps_layer(omega) = 0``
    (from the ``1/Y_p = kz/eps`` admittance entry).  Feed ``q_budget`` the ENZ-CLEARED function
    ``D_c = D * eps_layer(omega)`` -- exactly what :func:`berreman_enz_pole` builds, and what
    :func:`_stack_denominator` documents.  Without the clearing, the lossless Newton pass falls off
    the spurious pole into a far-plane stray: measured ``Re(pole) shift = +2256.83%`` to a root
    that is a genuine zero of ``D`` (``|D| = 1.55e-23`` vs an off-pole reference ``5.28e-07``) and
    is CORRECTLY SIGNED (``Re > 0``, ``Im < 0``), so no sign test can catch it (finding Q-4).

    VALIDITY GATE (finding Q-4).  ``pole_rad`` is accepted only if it lies within
    ``rad_proximity_linewidths`` total linewidths (``2 |Im pole_total|``, floored at
    ``rad_proximity_rel_floor * |pole_total|``) of ``pole_total`` AND ``|D_lossless(pole_rad)|`` is
    below ``residual_rel`` times an off-pole reference magnitude.  On failure ``Q_rad``, ``Q_abs``
    and ``inv_Q_abs`` are ``nan`` (NEVER ``+inf``), ``pole_rad_ok`` is ``False``, ``warning``
    carries the reason, and the report goes out under ``on_untrusted``.

    ``on_untrusted`` : {'warn', 'raise', 'ignore'} -- the policy for BOTH reports this function
    can make (the rejected lossless root above, and a negative ``1/Q_abs``).  Same keyword,
    vocabulary and default as :func:`find_poles` (audit Q-11): ``'warn'`` emits the
    ``RuntimeWarning`` exactly as before, ``'raise'`` makes it a ``RuntimeError``, ``'ignore'``
    silences it.  The returned dict -- ``warning`` string and ``pole_rad_ok`` flag included -- is
    IDENTICAL in all three modes, so ``'ignore'`` suppresses the noise without hiding the finding.

    Returns
    -------
    dict
        ``pole_total``, ``pole_rad``, ``Q_total``, ``Q_rad``, ``Q_abs``, ``inv_Q_abs``,
        ``pole_rad_ok`` (bool), ``pole_rad_shift_rel``, ``pole_rad_residual_rel``,
        ``omega0_shift_rel`` (the dispersive-bias diagnostic above), ``warning``.
        ``Q_abs = +inf`` flags a genuinely lossless / degenerate split (``|1/Q_abs|`` below
        ``degenerate_rel * 1/Q_total``); ``Q_abs = nan`` flags an INVALID split -- either a rejected
        lossless root or an unphysical negative ``1/Q_abs``.
    """
    _check_on_untrusted(on_untrusted, "q_budget")
    d_lossy = make_pole_func(float(loss_scale))
    d_rad = make_pole_func(0.0)
    pole_total = newton_refine(d_lossy, complex(pole0), tol=refine_tol)
    pole_rad = newton_refine(d_rad, pole_total, tol=refine_tol)
    q_total = pole_q(pole_total)

    # --- validate the lossless root: PROXIMITY first (the sign test is provably insufficient --
    # the observed escape has Re > 0 and Im < 0 and is a genuine zero of D). -------------------
    scale = abs(pole_total) if abs(pole_total) > 0.0 else 1.0
    linewidth = 2.0 * abs(pole_total.imag)
    bound = max(float(rad_proximity_linewidths) * linewidth,
                float(rad_proximity_rel_floor) * scale)
    shift = abs(pole_rad - pole_total)
    shift_rel = shift / scale
    # Q-15 diagnostic: the FRACTIONAL Re shift between the lossy and lossless passes. It is the
    # split's own error bar for a dispersive loss knob (zeroing a Drude gamma also moves
    # Re(eps), so the two linewidths are measured at two different omega_0).
    omega0_shift_rel = ((pole_rad.real - pole_total.real) / pole_total.real
                        if pole_total.real != 0.0 else float("nan"))
    # off-pole reference magnitude of the LOSSLESS function, half a Re-unit into the lower plane
    ref_pt = complex(pole_total.real, -0.5 * abs(pole_total.real) - abs(pole_total.imag))
    try:
        ref = abs(complex(d_rad(ref_pt)))
    except Exception:                                             # pragma: no cover - defensive
        ref = 0.0
    try:
        res_abs = abs(complex(d_rad(pole_rad)))
    except Exception:                                             # pragma: no cover - defensive
        res_abs = float("inf")
    residual_rel_meas = res_abs / ref if ref > 0.0 else float("inf")

    warning = ""
    ok = True
    if not np.isfinite(shift) or shift > bound:
        ok = False
        warning = (
            "q_budget: the lossless (loss_scale = 0) Newton pass ESCAPED the pole -- "
            "|pole_rad - pole_total| = {:.4e} rad/s ({:.2f}% of |pole_total|) exceeds the allowed "
            "{:.4e} ({:g} linewidths). Q_rad/Q_abs returned as NaN. If this is a p-polarized pole "
            "function over an ENZ material, feed the ENZ-CLEARED denominator D_c(omega) = "
            "D(omega) * eps_layer(omega) (as berreman_enz_pole does); otherwise reseed closer to "
            "the pole or widen rad_proximity_linewidths.".format(
                shift, 100.0 * shift_rel, bound, float(rad_proximity_linewidths)))
    elif not (residual_rel_meas <= float(residual_rel)):
        ok = False
        warning = (
            "q_budget: the lossless Newton pass did NOT converge to a zero -- "
            "|D_rad(pole_rad)| is {:.4e} of the off-pole reference (allowed {:g}). Q_rad/Q_abs "
            "returned as NaN.".format(residual_rel_meas, float(residual_rel)))

    if not ok:
        _report_untrusted(warning, on_untrusted)
        q_rad = float("nan")
        inv_q_abs = float("nan")
        q_abs = float("nan")
    else:
        q_rad = pole_q(pole_rad)
        inv_q_abs = (1.0 / q_total) - (1.0 / q_rad if math.isfinite(q_rad) else 0.0)
        # RELATIVE degeneracy threshold (the old absolute 1e-15 silently mapped a NEGATIVE
        # 1/Q_abs -- the Q-4 failure signature -- to +inf).
        eps_deg = float(degenerate_rel) * abs(1.0 / q_total) if q_total > 0.0 else 0.0
        if inv_q_abs > eps_deg:
            q_abs = 1.0 / inv_q_abs
        elif abs(inv_q_abs) <= eps_deg:
            q_abs = float("inf")                                  # lossless / degenerate split
        else:
            q_abs = float("nan")                                  # unphysical: Q_total > Q_rad
            warning = (
                "q_budget: 1/Q_abs = {:.4e} is NEGATIVE (Q_total = {:.6g} EXCEEDS Q_rad = {:.6g}), "
                "which is unphysical for an absorbing stack; the two passes are measuring "
                "different modes or different omega_0 (a dispersive lossless pass also shifts "
                "Re(eps)). Q_abs returned as NaN.".format(inv_q_abs, q_total, q_rad))
            _report_untrusted(warning, on_untrusted)
    return {
        "pole_total": pole_total,
        "pole_rad": pole_rad,
        "Q_total": q_total,
        "Q_rad": q_rad,
        "Q_abs": q_abs,
        "inv_Q_abs": inv_q_abs,
        "pole_rad_ok": bool(ok),
        "pole_rad_shift_rel": float(shift_rel),
        "pole_rad_residual_rel": float(residual_rel_meas),
        "omega0_shift_rel": float(omega0_shift_rel),
        "warning": warning,
    }


# ------------------------------------------------------------------------------------------------
# ENZ / Berreman thin-film mode
# ------------------------------------------------------------------------------------------------
def berreman_enz_pole(*, eps_inf: float, wp: float, gamma: float, thickness_m: float,
                      theta_rad: float, n_super=1.0, n_sub=1.0, omega_center=None,
                      omega_span=None, n_grid: int = 48, refine_tol: float = 1e-11,
                      on_untrusted: str = "warn") -> dict:
    """Find the ENZ / Berreman mode pole of a single thin Drude film (p-polarization, oblique).

    A subwavelength Drude film supports a leaky p-polarized mode near its epsilon-near-zero
    crossing ``omega ~ wp / sqrt(eps_inf)`` (Re(eps) = 0), the "Berreman mode" (Vassant et al.,
    Opt. Express 20, 23971 (2012)).  ``k_par`` is FIXED at ``omega_p`` for the oblique angle
    (QNM convention), then the scattering pole is located near ``omega_p``.

    Parameters
    ----------
    eps_inf, wp, gamma : float
        Drude parameters of the film (``wp``, ``gamma`` in rad/s).
    thickness_m : float
        Film thickness [m].  Thinner films push the mode TOWARD ``omega_p``.
    theta_rad : float
        Incidence angle (p-pol) in the superstrate.
    n_super, n_sub : complex
        Semi-infinite end indices (default vacuum both sides).
    omega_center, omega_span : complex, optional
        Search box; defaults bracket a region around ``omega_p`` in the lower half plane.
    on_untrusted : {'warn', 'raise', 'ignore'}
        Forwarded verbatim to :func:`find_poles` for the search this performs -- same vocabulary,
        same default (audit Q-11).  It governs the ROOT-BOX diagnostic on the default (or
        supplied) box, which is exactly the box a caller of this convenience wrapper never sees.

    Returns
    -------
    dict
        ``omega`` (complex pole), ``Q``, ``omega_p``, ``k_par``, and ``poles`` (all found in the
        box).  Raises ``ValueError`` if no decaying pole is found.
    """
    _check_on_untrusted(on_untrusted, "berreman_enz_pole")
    omega_p = wp / math.sqrt(eps_inf)
    k_par = k_par_from_angle(n_super, omega_p, theta_rad)

    def eps_film(w):
        return drude_eps(w, eps_inf, wp, gamma)

    film = (eps_film, float(thickness_m))
    func = smatrix_pole_func([film], pol="p", n_super=n_super, n_sub=n_sub, k_par_m=k_par)

    # The p-pol pole function D(omega) carries a SPURIOUS SIMPLE POLE at the film's ENZ crossing
    # eps_film(omega) = 0 (through the 1/Y_p = kz/eps admittance entry).  The genuine Berreman
    # zero sits right next to that point -- for eps_inf > 1 (every real TCO/ITO film) practically
    # ON TOP of it -- so the argument principle over any box containing both nets
    # (zeros - poles) ~ 0 and the mode is MISSED, while naive Newton seeds fall off to far-plane
    # strays (the pre-2026-07-19 failure: spurious poles at Re ~ 0 or ~10*omega_p returned
    # silently).  Clear the admittance pole the same way nonlocal_tmm.pole_function clears its
    # csc poles: D_c = D * eps_film is analytic at the ENZ point (simple pole times simple zero
    # -> finite NON-zero), keeps every scattering zero, and introduces no new one.
    def func_cleared(w):
        return func(w) * complex(eps_film(w))

    # Default box: bracket omega_p in Re and hug the real axis from below -- the high-Q Berreman
    # poles (small gamma and/or eps_inf > 1) sit at Im ~ -gamma, far shallower than the old
    # deeper default box reached.
    if omega_center is None:
        omega_center = complex(1.02 * omega_p, -0.10 * omega_p)
    if omega_span is None:
        omega_span = complex(0.14 * omega_p, 0.099 * omega_p)

    poles = find_poles(func_cleared, omega_center, omega_span, n_grid=n_grid,
                       refine_tol=refine_tol, on_untrusted=on_untrusted)
    # Backstop, independent of the winding machinery: Newton seeded at the coarse-grid minimum
    # of |D_c| over the box (the pole-cleared surface has its global minimum in the zero's
    # basin; verified against the driven-absorptance oracle in the tests).
    oc = complex(omega_center)
    osp = complex(omega_span)
    rect = (oc.real - abs(osp.real), oc.real + abs(osp.real),
            oc.imag - abs(osp.imag), oc.imag + abs(osp.imag))
    seed = _interior_seed(func_cleared, rect, max(16, n_grid // 2))
    poles.append(newton_refine(func_cleared, seed, tol=refine_tol))

    # Keep only genuine decaying zeros, VERIFIED ON THE ORIGINAL D (|D| negligible vs the
    # off-pole scale -- this also rejects any stray at the ENZ point itself, where |D| blows
    # up), inside/near the search box, deduplicated.
    scale = abs(func(complex(omega_p, -0.5 * omega_p)))       # reference magnitude of D off-pole
    genuine = []
    for p in poles:
        if (p.imag < 0.0 and p.real > 0.0 and _inside(p, rect)
                and abs(func(p)) < 1e-6 * max(scale, 1e-300)):
            if all(abs(p - g) > 1e-6 * max(abs(p), 1.0) for g in genuine):
                genuine.append(p)
    if not genuine:
        raise ValueError(
            "berreman_enz_pole: no decaying pole (Im < 0) found near omega_p = {:.4e} rad/s; widen "
            "omega_span/omega_center or check the Drude parameters.".format(omega_p))
    # The Berreman mode is the decaying pole closest to omega_p in Re.
    pole = min(genuine, key=lambda p: abs(p.real - omega_p))
    return {
        "omega": pole,
        "Q": pole_q(pole),
        "omega_p": omega_p,
        "k_par": k_par,
        "poles": genuine,
    }
