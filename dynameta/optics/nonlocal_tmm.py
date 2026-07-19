"""Hydrodynamic (nonlocal) Drude LAYERED transfer matrix -- roadmap item 2.4.

The local Drude model treats a metal's free-electron response as purely LOCAL: the
current at a point depends only on the field at that same point.  The hydrodynamic model
(HDM) restores the electron-gas PRESSURE (a ``beta**2 grad(div J)`` term, ``beta`` the
hydrodynamic velocity ~ Fermi velocity).  Physically this adds a LONGITUDINAL
(compressional / bulk-plasmon) wave on top of the usual transverse electromagnetic wave,
and it makes the metal's response NONLOCAL (spatially dispersive):

  * thin metal films acquire thickness-dependent ENZ / Berreman shifts (a BLUESHIFT that
    scales like 1/thickness);
  * a film supports discrete bulk-plasmon standing-wave resonances ABOVE the plasma
    frequency ``omega_p`` at ``k_L(omega) d ~ m*pi`` (m = 1, 2, 3, ...), seen as extra
    absorption peaks that have NO counterpart in the local model.

This module implements the p-polarized (TM) FOUR-WAVE layered transfer matrix (2 transverse
+ 2 longitudinal amplitudes inside each hydrodynamic metal layer) with the ADDITIONAL
BOUNDARY CONDITION (ABC) ``J_normal = 0`` (the normal free-electron current -- equivalently
the normal electron velocity -- vanishes at every metal surface: the abrupt "hard-wall"
model, no electron spill-out).  s-polarization (TE) has NO longitudinal coupling (the
pressure term acts only on the compressional / curl-free part of J, which does not couple to
a TE field), so s-pol reduces to the ordinary LOCAL transfer matrix and is returned as such.

-------------------------------------------------------------------------------------------------
THE LONGITUDINAL WAVENUMBER k_L (derived in THIS library's exp(-i*omega*t) convention)
-------------------------------------------------------------------------------------------------
Linearized hydrodynamic (Euler) equation of motion for the free-electron current density J,
with pressure term ``beta**2``, collision rate ``gamma``, plasma frequency ``omega_p``
(``omega_p**2 = n0 e**2 / (eps0 m*)``), under exp(-i*omega*t) (so d/dt -> -i*omega):

    beta**2 grad(div J) + omega*(omega + i*gamma) J = i*omega*eps0*omega_p**2 * E .   (HDM)

(Sign check -- the LOCAL limit ``beta -> 0`` with J = -i*omega*P gives
``eps(omega) = eps_inf - omega_p**2 / (omega**2 + i*omega*gamma)``, exactly this library's
``resonance.drude_eps`` and ``materials.DrudeOptical``: a passive absorber has Im(eps) > 0.)

Decompose J into a TRANSVERSE part (``div J = 0``) and a LONGITUDINAL part (``curl J = 0``,
J parallel to its wavevector k):

  * TRANSVERSE: the pressure term ``grad(div J)`` vanishes, so the transverse response is
    the ORDINARY LOCAL Drude -- NO spatial dispersion:
        eps_T(omega) = eps_inf - omega_p**2 / (omega**2 + i*omega*gamma).
    The transverse wave has ``k_z_T = sqrt(eps_T*(omega/c)**2 - k_par**2)`` (the usual TMM
    wavevector); at ``beta -> 0`` the whole model collapses onto this local result.

  * LONGITUDINAL: for a plane wave ``J_L ~ exp(i k_L.r)`` with ``J_L || k_L`` one has
    ``grad(div J_L) = -k_L**2 J_L``, and the mode lives at the ZERO of the LONGITUDINAL
    dielectric function
        eps_L(omega, k) = eps_inf - omega_p**2 / (omega**2 + i*omega*gamma - beta**2 * k**2).
    Setting eps_L = 0 (a longitudinal wave needs ``D_L = eps0 eps_L E_L = 0`` with E_L != 0)
    and solving for k gives the LONGITUDINAL WAVENUMBER

        k_L**2 = (omega**2 + i*gamma*omega - omega_p**2 / eps_inf) / beta**2 .    (k_L)

    In each layer the out-of-plane longitudinal wavevector is ``k_z_L = sqrt(k_L**2 -
    k_par**2)`` (``k_par`` the conserved in-plane wavevector, shared by all waves).

BRANCH / SIGN CHOICE.  Both ``k_z_T`` and ``k_z_L`` are taken on numpy's PRINCIPAL square-root
branch, ``sqrt(... + 0j)``.  Inside a FINITE layer both signs (+/-) of each are present
(forward + backward wave), and the layer's characteristic matrix is EVEN in ``k_z_T`` and in
``k_z_L`` (every entry is ``cos``, ``sin/(k_z)`` or ``(k_z)*sin`` -- all even functions of the
wavevector, hence functions of ``k_z**2`` with NO square-root branch cut), so the sign is
immaterial and the pole function is branch-cut-free in the LAYER wavevectors -- exactly the
property ``optics.resonance`` relies on for its argument-principle pole finder.  The forward
wave is ``exp(+i k_z z)`` under exp(-i*omega*t) (the same outgoing convention as
``optics.resonance``): a DECAYING resonance therefore sits in the LOWER half omega-plane
(Im(omega) < 0), so ``pole_function`` here is directly compatible with
``resonance.find_poles`` (which expects an analytic ``omega -> D(omega)`` whose zeros are the
scattering poles).

GNOR (generalized nonlocal optical response, Mortensen et al. 2014): induced-charge DIFFUSION
(constant ``D``, units m**2/s) is folded into the convection ``beta**2`` as a single COMPLEX
nonlocal parameter

    beta**2  ->  beta_eff**2 = beta**2 + D*(gamma - i*omega) ,

used verbatim in (k_L).  For real omega this adds a NEGATIVE imaginary part to ``beta_eff**2``,
which broadens the bulk-plasmon resonances (extra size-dependent damping) without a large
shift of their centre -- the GNOR signature.

ABC (additional boundary condition).  The extra longitudinal wave needs one extra boundary
condition beyond (E_tangential, H_tangential) continuity.  Here it is the HARD-WALL condition
that the normal free-electron current vanishes at each metal face, ``J_z = 0`` (Melnyk &
Harris 1970; Sipe; Raza et al. 2015 review).  Imposing ``J_z = 0`` at BOTH faces of a metal
layer eliminates its two longitudinal amplitudes in terms of its two transverse amplitudes,
collapsing the hydrodynamic layer to an EFFECTIVE 2x2 characteristic matrix in
(E_tangential, H_tangential) -- so the whole stack is a standard 2x2 Abeles cascade whose
metal layers merely carry a modified matrix.  The longitudinal contribution is proportional
to ``k_par`` and so VANISHES at normal incidence (no E_z to drive the compressional wave); it
resonates when ``sin(k_z_L d) = 0``, i.e. ``k_z_L d = m*pi`` -- the bulk-plasmon standing wave.

References
----------
* K. F. Melnyk, M. J. Harris, Phys. Rev. B 2, 835 (1970) -- the ABC layered formalism.
* N. A. Mortensen, S. Raza, M. Wubs, T. Sondergaard, S. I. Bozhevolnyi, "A generalized
  non-local optical response theory for plasmonic nanostructures", Nat. Commun. 5, 3809
  (2014) -- GNOR (the ``beta**2 -> beta**2 + D(gamma - i*omega)`` convection+diffusion knob).
* S. Raza, S. I. Bozhevolnyi, M. Wubs, N. A. Mortensen, "Nonlocal optical response in metallic
  nanostructures", J. Phys.: Condens. Matter 27, 183204 (2015) -- review; the slab closed
  forms and the bulk-plasmon standing waves above omega_p.
* G. Barton, "Some surface effects in the hydrodynamic model of metals", Rep. Prog. Phys. 42,
  963 (1979) -- beta**2 = (3/5) v_F**2 (high-frequency) vs (1/3) v_F**2 (Thomas-Fermi).

Conventions: SI units, exp(-i*omega*t) (a passive/absorbing medium has Im(eps) > 0), pure
numpy/scipy, ASCII-only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, NamedTuple, Sequence, Union

import numpy as np

from dynameta.constants import C_LIGHT, EPS0

__all__ = [
    "beta_from_vf",
    "HydroLayer",
    "DielectricLayer",
    "eps_transverse",
    "beta_eff_squared",
    "kL_squared",
    "RTA",
    "stack_rt",
    "rta",
    "k_par_from_angle",
    "pole_function",
]


# ------------------------------------------------------------------------------------------------
# beta from the Fermi velocity
# ------------------------------------------------------------------------------------------------
def beta_from_vf(v_f: float, convention: str = "high_freq") -> float:
    """Hydrodynamic velocity ``beta`` [m/s] from the Fermi velocity ``v_f`` [m/s].

    ``convention``:
      * ``"high_freq"`` (default): ``beta = sqrt(3/5) * v_f`` -- the high-frequency plasma limit
        ``omega >> gamma`` (the standard optical choice; Barton 1979).
      * ``"thomas_fermi"``: ``beta = sqrt(1/3) * v_f`` -- the low-frequency / static
        Thomas-Fermi limit.

    ``beta**2`` is what enters the longitudinal wavenumber (k_L)."""
    v = float(v_f)
    if convention == "high_freq":
        return math.sqrt(3.0 / 5.0) * v
    if convention == "thomas_fermi":
        return math.sqrt(1.0 / 3.0) * v
    raise ValueError("convention must be 'high_freq' or 'thomas_fermi'; got {!r}".format(convention))


# ------------------------------------------------------------------------------------------------
# Layer specifications
# ------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class HydroLayer:
    """A hydrodynamic-(nonlocal-)Drude metal layer.

    Parameters
    ----------
    eps_inf : float
        Bound-background (core/interband) permittivity.
    wp : float
        Plasma frequency ``omega_p`` [rad/s].
    gamma : float
        Collision / damping rate [rad/s] (>= 0; passive under exp(-i*omega*t)).
    beta : float
        Hydrodynamic velocity [m/s] (the pressure term ``beta**2``).  Build from a Fermi
        velocity with :func:`beta_from_vf`.  ``beta -> 0`` recovers the LOCAL Drude metal.
    thickness_m : float
        Layer thickness [m].
    D : float
        GNOR induced-charge diffusion constant [m**2/s] (default 0 -> pure hydrodynamic).  Enters
        as ``beta_eff**2 = beta**2 + D*(gamma - i*omega)``.
    """

    eps_inf: float
    wp: float
    gamma: float
    beta: float
    thickness_m: float
    D: float = 0.0


@dataclass(frozen=True)
class DielectricLayer:
    """A plain (local) dielectric layer: constant complex ``eps`` and thickness."""

    eps: complex
    thickness_m: float


Layer = Union[HydroLayer, DielectricLayer]


# ------------------------------------------------------------------------------------------------
# Material response
# ------------------------------------------------------------------------------------------------
def eps_transverse(omega, layer: HydroLayer) -> complex:
    """Transverse (ordinary, LOCAL) Drude permittivity of a hydrodynamic layer:

        eps_T(omega) = eps_inf - omega_p**2 / (omega**2 + i*omega*gamma).

    Independent of ``k`` -- the hydrodynamic pressure does not affect the transverse response
    (see the module derivation).  Accepts complex omega (analytic)."""
    w = complex(omega)
    return layer.eps_inf - layer.wp * layer.wp / (w * w + 1j * w * layer.gamma)


def beta_eff_squared(omega, layer: HydroLayer) -> complex:
    """GNOR effective nonlocal parameter ``beta_eff**2 = beta**2 + D*(gamma - i*omega)`` [m**2/s**2].
    With ``D = 0`` this is just ``beta**2`` (pure hydrodynamic convection)."""
    w = complex(omega)
    return layer.beta * layer.beta + layer.D * (layer.gamma - 1j * w)


def kL_squared(omega, layer: HydroLayer) -> complex:
    """Longitudinal (bulk-plasmon) wavenumber squared, in this library's exp(-i*omega*t)
    convention (see the module header for the derivation from eps_L(omega, k) = 0):

        k_L**2 = (omega**2 + i*gamma*omega - omega_p**2 / eps_inf) / beta_eff**2 .

    ``beta_eff**2`` carries the GNOR diffusion knob.  For a real omega above ``omega_p /
    sqrt(eps_inf)`` the real part is positive (a propagating bulk plasmon); below it the
    longitudinal wave is evanescent."""
    w = complex(omega)
    num = w * w + 1j * layer.gamma * w - layer.wp * layer.wp / layer.eps_inf
    return num / beta_eff_squared(w, layer)


# ------------------------------------------------------------------------------------------------
# Numerically stable cot / csc (large |Im| longitudinal phases at small beta)
# ------------------------------------------------------------------------------------------------
def _cot(p: complex) -> complex:
    """cot(p) = cos(p)/sin(p), numerically stable for large |Im(p)|.

    For a tiny ``beta`` the longitudinal phase ``p = k_z_L d`` is enormous and ``sin``/``cos``
    overflow; ``cot`` and ``csc`` however tend to FINITE limits (cot -> -+i, csc -> 0).  These
    are computed by factoring out the dominant exponential so nothing overflows."""
    z = complex(p)
    if z.imag >= 0.0:
        m = np.exp(2j * z)             # |m| = exp(-2*Im) <= 1
        return complex(1j * (m + 1.0) / (m - 1.0))
    m = np.exp(-2j * z)                # |m| = exp(2*Im) <= 1
    return complex(1j * (1.0 + m) / (1.0 - m))


def _csc(p: complex) -> complex:
    """csc(p) = 1/sin(p), numerically stable for large |Im(p)| (tends to 0)."""
    z = complex(p)
    if z.imag >= 0.0:
        t = np.exp(1j * z)             # |t| = exp(-Im) <= 1
        return complex(2j * t / (t * t - 1.0))
    u = np.exp(-1j * z)                # |u| = exp(Im) <= 1
    return complex(2j * u / (1.0 - u * u))


# ------------------------------------------------------------------------------------------------
# Per-medium out-of-plane wavevector and admittance
# ------------------------------------------------------------------------------------------------
def _kz(eps: complex, k0: complex, k_par: complex) -> complex:
    """Out-of-plane wavevector on the principal (outgoing) branch -- see module branch note."""
    return np.sqrt(eps * k0 * k0 - k_par * k_par + 0j)


def _admittance(eps: complex, kz: complex, pol: str) -> complex:
    """Physical optical admittance ``Y = H_tan / E_tan`` for a forward wave (H_tan is H_y for
    p-pol, and -H_x for s-pol, so both polarizations share the same sign pattern):
    p-pol (TM) ``Y = omega*eps0*eps / kz``; s-pol (TE) ``Y = kz / (omega*mu0)``.  The
    reflectance / transmittance built from these reproduce the exact Fresnel power coefficients
    (validated against ``tmm`` in the tests)."""
    if pol == "p":
        return eps / kz          # reduced by the common omega*eps0 (cancels in every ratio used)
    return kz                    # reduced by the common 1/(omega*mu0)


# ------------------------------------------------------------------------------------------------
# Per-layer 2x2 characteristic (Abeles) matrix in the (E_tan, H_tan) basis
# ------------------------------------------------------------------------------------------------
def _local_char_matrix(eps: complex, thickness_m: float, k0: complex, k_par: complex,
                       pol: str) -> np.ndarray:
    """Ordinary LOCAL characteristic matrix mapping (E_tan, H_tan) from the top face to the
    bottom face of a scalar slab: ``[[cos q, i sin q / Y], [i Y sin q, cos q]]`` with
    ``q = kz*d`` and ``Y`` the polarization admittance.  This is the standard Abeles matrix; it
    reproduces ``tmm`` for both polarizations (gate 1)."""
    kz = _kz(eps, k0, k_par)
    Y = _admittance(eps, kz, pol)
    q = kz * float(thickness_m)
    c = np.cos(q)
    s = np.sin(q)
    return np.array([[c, 1j * s / Y],
                     [1j * Y * s, c]], dtype=np.complex128)


def _hydro_char_matrix_p(omega: complex, layer: HydroLayer, k0: complex,
                         k_par: complex) -> np.ndarray:
    """p-pol EFFECTIVE 2x2 characteristic matrix of a hydrodynamic metal layer, in the
    (E_x, H_y) basis, AFTER eliminating the two longitudinal amplitudes via the ABC
    ``J_z = 0`` at both faces (see the module header).

    Inside the layer the general p-pol field is 2 transverse waves (amplitudes h_+, h_-;
    ``H_y ~ +-exp(i k_z_T z)``) plus 2 longitudinal waves (amplitudes a_+, a_-;
    ``exp(i k_z_L z)``).  Field pieces (forward-wave conventions of the module header):

        transverse:   E_x = C_T*(h_+ e^{iq z'} - h_- e^{-iq z'}),  H_y = h_+ e^{iq z'} + h_- ...,
                      with C_T = k_z_T/(omega eps0 eps_T);  and J_z^T = alpha * H_y,
                      alpha = i k_par (eps_T - eps_inf)/eps_T.
        longitudinal: E_x^L = k_par (a_+ + a_-) e^{i p z'} ...,  E_z^L = k_z_L (a_+ - a_-) ...,
                      H = 0;  and J_z^L = i omega eps0 eps_inf k_z_L (a_+ - a_-) ...  (using
                      eps_L = 0 => D_L = 0 => P_free,L = -eps0 eps_inf E_L).

    Imposing ``J_z(top) = J_z(bottom) = 0`` solves (a_+, a_-) in terms of (h_+, h_-); the
    resulting top and bottom (E_x, H_y) are then linear in (h_+, h_-), giving the 2x2 map
    below.  The longitudinal terms are all proportional to ``L1 = k_par**2 (eps_T - eps_inf)/
    (omega eps0 eps_inf eps_T k_z_L)`` and appear only through ``cot p`` / ``csc p`` (p = k_z_L
    d); ``L1`` ~ 1/k_z_L ~ beta and ``csc p -> 0`` as ``beta -> 0``, so the matrix collapses to
    the LOCAL one (gate 1).  ``cot p``, ``csc p`` diverge at ``sin p = 0`` (``k_z_L d = m*pi``):
    the bulk-plasmon standing-wave resonances (gate 2)."""
    eps_T = eps_transverse(omega, layer)
    kz_T = _kz(eps_T, k0, k_par)
    kz_L = np.sqrt(kL_squared(omega, layer) - k_par * k_par + 0j)
    d = float(layer.thickness_m)

    C_T = kz_T / (omega * EPS0 * eps_T)           # = 1 / Y_p (E/H for a forward transverse wave)
    q = kz_T * d
    p = kz_L * d
    cq = np.cos(q)
    sq = np.sin(q)
    cotp = _cot(p)
    cscp = _csc(p)

    # Longitudinal coupling strength (proportional to k_par**2; zero at normal incidence).
    L1 = (k_par * k_par * (eps_T - layer.eps_inf)
          / (omega * EPS0 * layer.eps_inf * eps_T * kz_L))

    # Top-face state [E_x(0); H_y(0)] = P0 @ (v0, u0), bottom = Pd @ (v0, u0), with
    # v0 = h_+ - h_-, u0 = h_+ + h_-.  P0 is upper triangular (H_y(0) = u0).
    d00 = C_T - L1 * sq * cscp                     # P0[0,0]
    e01 = -1j * L1 * (cotp - cq * cscp)            # P0[0,1]
    Pd00 = C_T * cq - L1 * sq * cotp
    Pd01 = 1j * C_T * sq - 1j * L1 * (cscp - cq * cotp)
    Pd10 = 1j * sq
    Pd11 = cq

    # M = Pd @ inv(P0);  inv(P0) = [[1/d00, -e01/d00], [0, 1]].
    M00 = Pd00 / d00
    M01 = -Pd00 * e01 / d00 + Pd01
    M10 = Pd10 / d00
    M11 = -Pd10 * e01 / d00 + Pd11

    # The above is built in PHYSICAL fields (H_y physical, C_T = kz/(omega eps0 eps)); the rest
    # of the cascade uses the REDUCED admittance Y_p = eps/kz (the common omega*eps0 dropped, as
    # it cancels in every ratio).  Convert with the similarity S = diag(1, omega*eps0) so this
    # matrix acts on the reduced state [E_x, H_y/(omega eps0)] like the local layers:
    #   M_red[0,1] = M_phys[0,1]*(omega eps0),  M_red[1,0] = M_phys[1,0]/(omega eps0).
    wep = omega * EPS0
    return np.array([[M00, M01 * wep], [M10 / wep, M11]], dtype=np.complex128)


def _layer_matrix(omega: complex, layer: Layer, k0: complex, k_par: complex,
                  pol: str) -> np.ndarray:
    """Characteristic matrix for one layer.  Dielectric -> local; hydrodynamic metal in s-pol
    -> local with eps = eps_T (no longitudinal coupling in TE); hydrodynamic metal in p-pol ->
    the effective ABC matrix."""
    if isinstance(layer, DielectricLayer):
        return _local_char_matrix(complex(layer.eps), layer.thickness_m, k0, k_par, pol)
    if isinstance(layer, HydroLayer):
        if pol == "p":
            return _hydro_char_matrix_p(omega, layer, k0, k_par)
        return _local_char_matrix(eps_transverse(omega, layer), layer.thickness_m,
                                  k0, k_par, pol)
    raise TypeError("layer must be a HydroLayer or DielectricLayer; got {!r}".format(type(layer)))


def _stack_matrix(omega: complex, layers: Sequence[Layer], k0: complex, k_par: complex,
                  pol: str) -> np.ndarray:
    """Total characteristic matrix of the stack, mapping (E_tan, H_tan) from the SUPERSTRATE
    face to the SUBSTRATE face.  ``layers`` are ordered superstrate-side first, so
    ``M_total = M_{N} @ ... @ M_1`` (each layer post-multiplies onto the running product from
    the left)."""
    M = np.eye(2, dtype=np.complex128)
    for layer in layers:
        M = _layer_matrix(omega, layer, k0, k_par, pol) @ M
    return M


# ------------------------------------------------------------------------------------------------
# Reflectance / transmittance / absorptance
# ------------------------------------------------------------------------------------------------
class RTA(NamedTuple):
    """Result of :func:`stack_rt`.

    Attributes
    ----------
    r, t : complex
        Amplitude reflection / transmission coefficients (tangential-E ratios).
    R, T, A : float
        Power reflectance ``|r|**2``, transmittance ``|t|**2 Re(Y_sub)/Re(Y_super)`` (angle /
        index corrected), and absorptance ``A = 1 - R - T``.
    D_pole : complex
        The reflection denominator ``Q - P``; its ZEROS are the scattering poles (fed to
        :func:`pole_function`).
    """

    r: complex
    t: complex
    R: float
    T: float
    A: float
    D_pole: complex


def k_par_from_angle(n_super, omega, theta_rad) -> float:
    """In-plane wavevector ``k_par = Re(n_super) (omega/c) sin(theta)`` [1/m].  ``n_super`` must
    be a real (lossless) incidence index for ``theta`` to be a real angle.  Hold this FIXED
    when continuing to complex omega (the QNM/pole convention)."""
    return float(np.real(n_super) * (omega / C_LIGHT) * math.sin(theta_rad))


def _rt_from_matrix(M: np.ndarray, Y_super: complex, Y_sub: complex):
    """Amplitude r, t and the pole denominator from the total characteristic matrix and the
    end-media admittances.  Derived by matching (E_tan, H_tan) of incident+reflected (super) to
    transmitted (sub) across ``M`` (see module):

        P = Y_sub*M00 - M10,   Q = Y_super*(Y_sub*M01 - M11),
        r = (P + Q)/(Q - P),   t = M00*(1 + r) + M01*Y_super*(1 - r),
        D_pole = Q - P  (zero => scattering pole)."""
    M00, M01 = M[0, 0], M[0, 1]
    M10, M11 = M[1, 0], M[1, 1]
    P = Y_sub * M00 - M10
    Q = Y_super * (Y_sub * M01 - M11)
    D_pole = Q - P
    r = (P + Q) / D_pole
    t = M00 * (1.0 + r) + M01 * Y_super * (1.0 - r)
    return r, t, D_pole


def stack_rt(omega, layers: Sequence[Layer], *, pol: str = "p", n_super=1.0, n_sub=1.0,
             theta_rad: float = 0.0, k_par_m=None) -> RTA:
    """Reflectance / transmittance / absorptance of ``super | layers | sub`` at ``omega``.

    Parameters
    ----------
    omega : float or complex
        Angular frequency [rad/s].
    layers : sequence of HydroLayer / DielectricLayer
        Ordered from the superstrate side to the substrate side.
    pol : {'p', 's'}
        Polarization.  Only p-pol carries the longitudinal (nonlocal) physics; s-pol is the
        local result.
    n_super, n_sub : complex
        Semi-infinite superstrate / substrate refractive indices (eps = n**2).
    theta_rad : float
        Incidence angle in the superstrate (used if ``k_par_m`` is None).
    k_par_m : float or complex, optional
        Explicit in-plane wavevector [1/m]; overrides ``theta_rad`` (pass this, held fixed, for
        complex-omega pole work).

    Returns
    -------
    RTA
    """
    if pol not in ("p", "s"):
        raise ValueError("pol must be 'p' or 's'")
    w = complex(omega)
    k0 = w / C_LIGHT
    eps_super = complex(n_super) ** 2
    eps_sub = complex(n_sub) ** 2
    if k_par_m is None:
        k_par = complex(n_super) * k0 * math.sin(theta_rad)
    else:
        k_par = complex(k_par_m)

    M = _stack_matrix(w, layers, k0, k_par, pol)
    Y_super = _admittance(eps_super, _kz(eps_super, k0, k_par), pol)
    Y_sub = _admittance(eps_sub, _kz(eps_sub, k0, k_par), pol)

    r, t, D_pole = _rt_from_matrix(M, Y_super, Y_sub)
    R = float(abs(r) ** 2)
    denom = Y_super.real
    T = float(abs(t) ** 2 * (Y_sub.real / denom)) if denom != 0.0 else float("nan")
    A = 1.0 - R - T
    return RTA(r=r, t=t, R=R, T=T, A=A, D_pole=D_pole)


def rta(omega, layers: Sequence[Layer], **kwargs):
    """Convenience wrapper returning just ``(R, T, A)`` from :func:`stack_rt`."""
    res = stack_rt(omega, layers, **kwargs)
    return res.R, res.T, res.A


# ------------------------------------------------------------------------------------------------
# Pole function (compatible with optics.resonance.find_poles)
# ------------------------------------------------------------------------------------------------
def pole_function(layers: Sequence[Layer], *, pol: str = "p", n_super=1.0, n_sub=1.0,
                  k_par_m=0.0) -> Callable[[complex], complex]:
    """Return the analytic scattering-pole function (a closure holding ``k_par`` FIXED), whose
    ZEROS are the scattering poles of the hydrodynamic stack.  Feed it to
    ``dynameta.optics.resonance.find_poles`` / ``newton_refine``.

    The nonlocal bulk-plasmon and ENZ / Berreman modes appear as complex-omega zeros with
    Im(omega) < 0 (decaying, exp(-i*omega*t) -- the SAME outgoing/decaying convention as
    ``optics.resonance``; the forward wave is exp(+i*k_z*z)).

    IMPORTANT (why the returned function is ``(Q - P) * prod_j sin(k_z_L,j d_j)`` and not the
    bare reflection denominator ``Q - P``): a p-pol hydrodynamic layer's effective matrix carries
    the ABC ``1/sin(k_z_L d)`` (``csc``/``cot``) terms, so ``Q - P`` has its OWN POLES at the
    standing-wave points ``sin(k_z_L d) = 0``.  The argument principle counts (zeros - poles), so
    a search box enclosing a scattering zero AND such a csc-pole would net ~0 and the pole would
    be MISSED.  ``Q - P`` has a SIMPLE pole there, so multiplying by ``sin(k_z_L d)`` cancels it
    (infinity*0 -> a finite NON-zero value, introducing NO spurious zero) and leaves an analytic
    function whose only zeros are the physical scattering poles.  Branch-cut-free in the layer
    wavevectors (even in ``k_z_T`` / ``k_z_L``); the only omega-plane non-analyticities are the
    end-media light lines and, for p-pol, the ``eps_T = 0`` admittance point -- place the search
    box to exclude both (as ``resonance.find_poles`` already requires for local stacks).  Intended
    for GENUINE nonlocal stacks (finite ``beta``); in the ``beta -> 0`` local limit there is no
    bulk-plasmon pole and ``sin(k_z_L d)`` overflows (bulk plasmons do not exist there)."""
    if pol not in ("p", "s"):
        raise ValueError("pol must be 'p' or 's'")
    kpar = complex(k_par_m)
    eps_super = complex(n_super) ** 2
    eps_sub = complex(n_sub) ** 2
    # p-pol hydrodynamic layers whose csc pole must be cleared from the pole function.
    hydro_p = [layer for layer in layers if pol == "p" and isinstance(layer, HydroLayer)]

    def D(omega):
        w = complex(omega)
        k0 = w / C_LIGHT
        M = _stack_matrix(w, layers, k0, kpar, pol)
        Y_super = _admittance(eps_super, _kz(eps_super, k0, kpar), pol)
        Y_sub = _admittance(eps_sub, _kz(eps_sub, k0, kpar), pol)
        M00, M01 = M[0, 0], M[0, 1]
        M10, M11 = M[1, 0], M[1, 1]
        P = Y_sub * M00 - M10
        Q = Y_super * (Y_sub * M01 - M11)
        val = Q - P
        for layer in hydro_p:                      # clear the ABC 1/sin(k_z_L d) poles
            kz_L = np.sqrt(kL_squared(w, layer) - kpar * kpar + 0j)
            val = val * np.sin(kz_L * float(layer.thickness_m))
        return complex(val)

    return D
