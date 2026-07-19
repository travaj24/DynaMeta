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
"""

from __future__ import annotations

import math
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
def _kz(eps: complex, k0: complex, k_par: complex) -> complex:
    """Out-of-plane wavevector on the PRINCIPAL (outgoing) branch -- see the module branch note."""
    return np.sqrt(eps * k0 * k0 - k_par * k_par + 0j)


def _admittance(eps: complex, kz: complex, pol: str) -> complex:
    """Reduced optical admittance ``Y = H_tan / E_tan`` (common factors dropped):
    s-pol (TE) ``Y ~ kz``; p-pol (TM) ``Y ~ eps / kz``.  Reflectance ``|r|^2`` and transmittance
    ``|t|^2 * Re(Y_sub)/Re(Y_super)`` built from these reproduce the exact Fresnel power
    coefficients for both polarizations (validated against ``tmm`` to ~1e-14)."""
    return kz if pol == "s" else eps / kz


def _interface(Ya: complex, Yb: complex) -> np.ndarray:
    """2x2 interface matrix mapping the (forward, backward) tangential-E amplitudes referenced on
    the ``b`` side to those on the ``a`` side (continuity of E_tan and H_tan)::

        [A_a; B_a] = 0.5 * [[1+rho, 1-rho], [1-rho, 1+rho]] [A_b; B_b],   rho = Y_b / Y_a
    """
    rho = Yb / Ya
    return 0.5 * np.array([[1.0 + rho, 1.0 - rho],
                           [1.0 - rho, 1.0 + rho]], dtype=np.complex128)


def _propagate(kz: complex, d: float) -> np.ndarray:
    """2x2 propagation matrix advancing the reference plane by ``d`` toward the substrate.
    Forward wave ~ ``exp(+i*kz*z)`` (``exp(-i*omega*t)`` convention) => ``diag(e^{-i kz d},
    e^{+i kz d})``.  This sign is what places DECAYING poles in the lower half plane (verified
    against the Fabry-Perot closed form)."""
    e = np.exp(1j * kz * d)
    return np.array([[1.0 / e, 0.0], [0.0, e]], dtype=np.complex128)


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
        The (0,0) entry of the total transfer matrix.  ``t = 1/M11``; the SCATTERING POLES are its
        zeros, so this is the analytic function fed to :func:`find_poles`.
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
        raise ValueError("pol must be 's' or 'p'")
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

    M = _interface(Ylist[0], Ylist[1])
    for j in range(N):
        M = M @ _propagate(kz_layers[j], float(layers[j][1])) @ _interface(Ylist[j + 1], Ylist[j + 2])

    M11 = M[0, 0]
    M12 = M[0, 1]
    M21 = M[1, 0]
    M22 = M[1, 1]
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

    Mc = np.eye(2, dtype=np.complex128)
    for e_spec, d in layers:
        eps = _eval_eps(e_spec, omega)
        kz = _kz(eps, k0, kpar)
        Y = _admittance(eps, kz, pol)
        phi = kz * float(d)
        c = np.cos(phi)
        s = np.sin(phi)
        m = np.array([[c, -1j * s / Y], [-1j * Y * s, c]], dtype=np.complex128)
        Mc = Mc @ m

    B = Mc[0, 0] + Mc[0, 1] * Y_sub
    C = Mc[1, 0] + Mc[1, 1] * Y_sub
    return complex(Y_super * B + C)


def smatrix_pole_func(layers: Sequence[Layer], *, pol: str = "s", n_super=1.0, n_sub=1.0,
                      k_par_m=0.0) -> Callable[[complex], complex]:
    """Return the analytic scattering-pole function ``D(omega)`` (a closure holding ``k_par``
    FIXED), whose zeros are the scattering poles of the stack.  Feed this to :func:`find_poles` /
    :func:`newton_refine`.  See :func:`_stack_denominator` for why the characteristic-matrix form
    is used (branch-cut-free in the layer wavevectors, correct decaying-pole sign)."""
    def D(omega):
        return _stack_denominator(omega, layers, pol, n_super, n_sub, k_par_m)
    return D


# ------------------------------------------------------------------------------------------------
# Newton refinement and Q
# ------------------------------------------------------------------------------------------------
def pole_q(omega_tilde) -> float:
    """Quality factor of a complex pole: ``Q = Re(omega_tilde) / (2 |Im(omega_tilde)|)``.  Returns
    ``+inf`` for a real (lossless, undamped) pole."""
    w = complex(omega_tilde)
    im = abs(w.imag)
    if im == 0.0:
        return float("inf")
    return abs(w.real) / (2.0 * im)


def newton_refine(func: Callable[[complex], complex], z0, *, tol: float = 1e-11,
                  maxiter: int = 100, h_rel: float = 1e-7) -> complex:
    """Newton's method on an analytic ``func`` with a central-difference derivative (the material
    models are analytic but not necessarily cheap to differentiate in closed form).  ``tol`` is the
    RELATIVE step-size stopping criterion on ``omega``.  Returns the best iterate."""
    z = complex(z0)
    for _ in range(maxiter):
        f = func(z)
        if f == 0.0:
            return z
        h = h_rel * max(abs(z), 1.0)
        fp = (func(z + h) - func(z - h)) / (2.0 * h)
        if fp == 0.0 or not np.isfinite(fp):
            break
        dz = f / fp
        z = z - dz
        if abs(dz) <= tol * max(abs(z), 1.0):
            return z
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


def _winding_densified(func: Callable[[complex], complex],
                       rect: Tuple[float, float, float, float],
                       n_grid: int) -> Tuple[float, float]:
    """:func:`_winding` with adaptive boundary densification (doubling up to 16x while any
    single phase step exceeds ~1.2 rad).  Returns ``(w, maxstep)`` at the final density.  A
    residual ``maxstep > 1.2`` after densification flags an UNTRUSTWORTHY count -- typically a
    zero lying on (or hugging) the contour, whose ~pi phase jump no sampling density removes."""
    w, maxstep = _winding(func, rect, n_grid)
    ng = n_grid
    while maxstep > 1.2 and ng < n_grid * 16:
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


def find_poles(func_of_omega: Callable[[complex], complex], omega_center, omega_span, *,
               n_grid: int = 40, refine_tol: float = 1e-11, max_depth: int = 8,
               dedup_rel: float = 1e-6) -> List[complex]:
    """Locate the poles of a scattering response (the zeros of ``func_of_omega``, e.g. ``M11`` from
    :func:`smatrix_pole_func`) inside a rectangle of the complex-omega plane, via the argument
    principle + Newton refinement.

    Parameters
    ----------
    func_of_omega : callable
        Analytic function whose ZEROS are the sought poles (``1/S``, ``M11``, or ``det`` of the
        inverse scattering matrix).
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

    Returns
    -------
    list of complex
        Refined pole positions (unordered), each a zero of ``func_of_omega`` inside the box.
    """
    span = complex(omega_span)
    sr = abs(span.real) if span.real != 0.0 else abs(span.imag)
    si = abs(span.imag) if span.imag != 0.0 else abs(span.real)
    c = complex(omega_center)
    root_rect = (c.real - sr, c.real + sr, c.imag - si, c.imag + si)

    found: List[complex] = []

    def newton_in(rect):
        seed = _interior_seed(func_of_omega, rect, max(6, n_grid // 4))
        z = newton_refine(func_of_omega, seed, tol=refine_tol)
        if _inside(z, rect) and np.isfinite(z):
            found.append(z)

    def recurse(rect, depth, count=None):
        if count is None:
            w, _ = _winding_densified(func_of_omega, rect, n_grid)
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
                ws, ms = _winding_densified(func_of_omega, sub, n_grid)
                cs = int(round(ws))
                if ms > 1.2 or abs(ws - cs) > 0.25:
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
    return uniq


# ------------------------------------------------------------------------------------------------
# Parameter tracking (continuation)
# ------------------------------------------------------------------------------------------------
def track_pole(solver: Callable[[float], Callable[[complex], complex]], pole0, param_values,
               *, refine_tol: float = 1e-11, jump_rel: float = 0.25,
               max_subdiv: int = 40) -> List[complex]:
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

    Returns
    -------
    list of complex
        The tracked pole at each parameter value (same length as ``param_values``).
    """
    params = [float(p) for p in param_values]
    if not params:
        return []

    def step(p_from, z_from, p_to, depth):
        z = newton_refine(solver(p_to), z_from, tol=refine_tol)
        if abs(z - z_from) <= jump_rel * max(abs(z_from), 1.0) or depth >= max_subdiv:
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
             refine_tol: float = 1e-11, loss_scale: float = 1.0) -> dict:
    """Split the total Q of a pole into radiative and absorptive parts by the lossless/lossy
    two-pass (Lalanne et al. 2018).

    ``make_pole_func(loss_scale) -> D`` must return the pole function ``D(omega)`` (zeros = poles)
    with the material LOSSES scaled by ``loss_scale`` (so ``0.0`` is lossless, ``1.0`` the physical
    stack, ``2.0`` double loss).  Build such a factory by scaling the imaginary part of each layer's
    eps, or the Drude ``gamma``.  Keeping it a MATERIAL-level knob preserves analyticity in omega
    (scaling ``Im(eps(omega))`` pointwise for complex omega would not be analytic).

    The lossless pass re-finds the pole with ``loss_scale = 0`` (warm-started from the lossy pole):
    ``Q_rad``.  Then ``1/Q_abs = 1/Q_total - 1/Q_rad``.

    Returns
    -------
    dict
        ``pole_total``, ``pole_rad``, ``Q_total``, ``Q_rad``, ``Q_abs``, ``inv_Q_abs``.
        ``Q_abs = +inf`` (``inv_Q_abs <= 0``) flags a lossless / numerically-degenerate case.
    """
    pole_total = newton_refine(make_pole_func(float(loss_scale)), complex(pole0), tol=refine_tol)
    pole_rad = newton_refine(make_pole_func(0.0), pole_total, tol=refine_tol)
    q_total = pole_q(pole_total)
    q_rad = pole_q(pole_rad)
    inv_q_abs = (1.0 / q_total) - (1.0 / q_rad if math.isfinite(q_rad) else 0.0)
    q_abs = (1.0 / inv_q_abs) if inv_q_abs > 1e-15 else float("inf")
    return {
        "pole_total": pole_total,
        "pole_rad": pole_rad,
        "Q_total": q_total,
        "Q_rad": q_rad,
        "Q_abs": q_abs,
        "inv_Q_abs": inv_q_abs,
    }


# ------------------------------------------------------------------------------------------------
# ENZ / Berreman thin-film mode
# ------------------------------------------------------------------------------------------------
def berreman_enz_pole(*, eps_inf: float, wp: float, gamma: float, thickness_m: float,
                      theta_rad: float, n_super=1.0, n_sub=1.0, omega_center=None,
                      omega_span=None, n_grid: int = 48, refine_tol: float = 1e-11) -> dict:
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

    Returns
    -------
    dict
        ``omega`` (complex pole), ``Q``, ``omega_p``, ``k_par``, and ``poles`` (all found in the
        box).  Raises ``ValueError`` if no decaying pole is found.
    """
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
                       refine_tol=refine_tol)
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
