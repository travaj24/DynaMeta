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
that the normal free-electron current vanishes at each conductor face, ``J_z = 0`` (Melnyk &
Harris 1970; Sipe; Raza et al. 2015 review).  Imposing ``J_z = 0`` at BOTH faces of a metal
layer eliminates its two longitudinal amplitudes in terms of its two transverse amplitudes,
collapsing the hydrodynamic layer to an EFFECTIVE 2x2 characteristic matrix in
(E_tangential, H_tangential) -- so the whole stack is a standard 2x2 Abeles cascade whose
metal layers merely carry a modified matrix.  The longitudinal contribution is proportional
to ``k_par`` and so VANISHES at normal incidence (no E_z to drive the compressional wave); it
resonates when ``sin(k_z_L d) = 0``, i.e. ``k_z_L d = m*pi`` -- the bulk-plasmon standing wave.

-------------------------------------------------------------------------------------------------
RUNS: SUB-LAYERS OF ONE CONDUCTOR, AND GRADED DENSITY (v0.16)
-------------------------------------------------------------------------------------------------
The hard wall belongs at a CONDUCTOR BOUNDARY (conductor/dielectric, or the contact of two
conductors the user declares distinct), never between sub-layers of one electron gas. Adjacent
``HydroLayer`` objects sharing a non-None ``conductor_id`` form a RUN: one connected electron gas,
each sub-layer with its own equilibrium density (its own ``wp``, ``beta``, ``gamma``, ``eps_inf``),
J_z = 0 imposed only on the run's two outer faces. A z-graded free-carrier profile (an accumulation
or depletion layer) is modelled as a run of thin uniform sub-layers; ``hydro_layers_from_profile``
builds that run from n(z) on a grid, and the staircase converges to the continuous profile at
second order (gated against an independent boundary-value solve).

PRESSURE MODEL. Barotropic electron gas, pressure p(n), chemical potential mu(n) with
dp = n dmu. The per-particle Euler equation m(dv/dt + gamma v) = -e E - grad mu(n), linearized
about an equilibrium n0(z) that is in force balance under the SAME pressure (-e E0 = grad mu(n0),
the zeroth order cancels exactly), with J = -e n0 v and rho1 = -e n1:

    (d/dt + gamma) J = eps0 wp(z)**2 (E + grad psi),   psi = delta mu / e = mu'(n0) n1 / e,

    psi = -beta(z)**2 rho1 / (eps0 wp(z)**2),   beta**2 = n0 mu'(n0) / m   (LOCAL in n0),

with wp**2 = n0 e**2/(eps0 m). For a uniform layer this is exactly the (HDM) equation above. For a
degenerate parabolic band mu'(n0) n0 = (2/3) E_F, so beta**2 = v_F(n0)**2 / 3 (Thomas-Fermi); the
optical 'high_freq' choice (3/5) v_F**2 rescales the pressure by 9/5, uniformly, so the same form
holds. beta therefore varies with the local density (v_F ~ n0**(1/3)); ``hydro_layers_from_profile``
sets it per sub-layer. GNOR: beta**2 -> beta_eff**2 = beta**2 + D(gamma - i omega) inside psi
(diffusion driven by the same electrochemical gradient, Einstein relation).

INTERFACE CONDITIONS inside one conductor. Divide the momentum equation by wp**2 and integrate the
normal component across an interface (or across a thin smooth transition): J_z and E_z stay bounded
and wp**2 stays away from zero, so the integral of d(psi)/dz vanishes -- psi is CONTINUOUS. Charge
conservation with a bounded rho1 (no sheet charge inside the gas) makes J_z CONTINUOUS. With
Maxwell's E_x and H_y continuity the run carries the four-vector

    (E_x, H_y, J_z, psi)   continuous at every internal interface;  J_z = 0 at the run's faces.

psi continuity is continuity of the linearized electrochemical potential -- the static limit of
the condition is "equal Fermi levels across the junction". It is NOT continuity of the pressure
term beta**2 n1 (= n0 psi up to constants): that condition (a) belongs to a different continuum
equation, one that drops the -e n1 E0 term and so does not keep a graded equilibrium stationary,
(b) breaks energy conservation -- the conserved flux is S_z - (1/2) Re(psi* J_z) (Poynting plus
the pressure work), continuous only if psi and J_z both are; a lossless graded run under
pressure continuity shows spurious gain, A = -2e-6 to -3.5e-6 measured -- and (c) converges, as the
staircase is refined, to an answer 3e-4 away in r from the boundary-value oracle of the graded
film. For one effective mass across the run, psi continuity is the same statement as continuity
of beta**2 div(J)/wp**2. The kernel (``hydro_conductivity``, ``hydro_psi_per_charge``,
``hydro_state_generator``, ``hydro_modes``, ``HYDRO_INTERFACE_FIELDS``) is importable on its own so
that other discretizations use the same equations: in a finite-element setting psi is the
natural H1 (continuous) unknown and J the H(div) one, and these conditions hold without being
imposed.

SOLVER. Each sub-layer is solved exactly (two transverse + two longitudinal waves, closed form).
The run is a banded global linear system in the wave amplitudes, each amplitude referenced at
the face its wave decays away from (every exponential evaluated has modulus <= 1), with row and
column equilibration; it is reduced to the 2x2 (E_x, H_y) matrix the rest of the cascade uses. No
4x4 transfer matrix (with its exp(+|k_z,L| d) growth) is ever formed: a uniform film split into 1000
sub-layers reproduces the single-layer result to 4e-14 in r. A run of ONE layer uses the closed-form
ABC matrix unchanged (bit-identical to v0.15.1). s-polarization has no longitudinal coupling and
is the local cascade, run or not; a run whose members are all in the exact local limit (beta = D =
0) is the local graded-layer cascade.

VALIDITY. Hydrodynamics is the long-wavelength expansion of the electron gas response. With the
'high_freq' beta it is valid while u = |k_L| v_F / |omega| << 1 for the longitudinal wave of every
sub-layer; as u approaches 1 the wave enters the particle-hole continuum (Landau damping, absent
from the model) and the next term of the expansion is no longer small. With the 'thomas_fermi'
beta it is the static-screening limit, valid for u >> 1 (the neglected Landau damping is ~
pi/(2u) relative). The model also needs a degenerate gas (E_F >> k_B T): in a depletion layer
where E_F falls to a few k_B T the degenerate pressure is wrong. ``hydro_validity`` reports u
per layer (and E_F / k_B T when the mass and temperature are given) and flags a solve outside
these limits; ``stack_rt(check_validity=True)`` turns a flag into an UnreliableResultWarning.
Note that a degenerate oxide film in the near infrared typically sits near u ~ 1: the solver then
returns the hydrodynamic answer, which is itself outside its validity.

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

THE HOME OF THE HYDRODYNAMIC MATERIAL MODEL (audit X-4/X-5). ``beta_from_vf``, ``eps_transverse``,
``beta_eff_squared`` and ``kL_squared`` are defined HERE and nowhere else; the FEM tier
(``optics.hydro_fem.HydroParams``) delegates to them through ``HydroParams.as_layer()`` and the
FDTD tier (``optics.hydro_fdtd``) re-exports ``beta_from_vf`` verbatim. A change to any of the four
is a change to every hydrodynamic solver in the library, which is the point: the cross-solver
bulk-plasmon gates compare solvers, not two transcriptions of one formula. The contract is gated by
tests/test_hydro_fem.py::test_hydroparams_match_nonlocal_tmm, which covers every shared function
over metals x GNOR settings x real and complex frequencies, EXACTLY.

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

import math
import warnings
from dataclasses import dataclass
from typing import Callable, Hashable, NamedTuple, Optional, Sequence, Union

import numpy as np

from dynameta.constants import C_LIGHT, EPS0, HBAR, KB, Q_E
from dynameta.optics.tmm_reference import _check_energy_budget   # ONE energy-budget rule (Q-23)

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
    # v0.16: runs of sub-layers of one conductor, graded density
    "HYDRO_INTERFACE_FIELDS",
    "fermi_velocity",
    "beta_from_density",
    "hydro_conductivity",
    "hydro_psi_per_charge",
    "hydro_state_generator",
    "hydro_modes",
    "hydro_layers_from_profile",
    "HydroValidity",
    "hydro_validity",
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
    conductor_id : hashable, optional
        Which conductor this layer belongs to (v0.16).  ADJACENT HydroLayers with the
        same non-None id form a RUN -- one connected electron gas, (E_x, H_y, J_z, psi) continuous
        between its sub-layers, J_z = 0 only on its two outer faces (see the module header).
        Adjacent HydroLayers with different non-None ids are two conductors in contact: a wall
        between them.  ``None`` (default) keeps the v0.15.1 meaning (the layer is walled on both
        faces), but a ``None`` layer directly adjacent to another HydroLayer is refused by
        ``stack_rt`` / ``pole_function`` unless ``allow_walls=True`` (the explicit opt-in to the old
        physics), because a wall inside one conductor is almost never what was meant.
    """

    eps_inf: float
    wp: float
    gamma: float
    beta: float
    thickness_m: float
    D: float = 0.0
    conductor_id: Optional[Hashable] = None


@dataclass(frozen=True)
class DielectricLayer:
    """A plain (local) dielectric layer: constant complex ``eps`` and thickness."""

    eps: complex
    thickness_m: float


NonlocalLayer = Union[HydroLayer, DielectricLayer]


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
    longitudinal wave is evanescent.  At ``beta = D = 0`` it is infinite and this raises
    ``ZeroDivisionError`` -- there is no longitudinal wave; the stack routines treat that layer
    as the local Drude metal (audit Q-18)."""
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
# AUDIT Q-19 (2026-11-02): `_kz`, `_admittance` and `k_par_from_angle` were VERBATIM copies of
# optics.resonance's (and the `_admittance` twins had already diverged once -- V-3/V-8).  ONE
# implementation now lives in optics.resonance and this module uses it: `_kz` (principal branch,
# cmath.sqrt -- bit-identical to the np.sqrt(... + 0j) this module had on every reachable argument
# class, audit P-6) and `_admittance` (reduced Y: s ~ kz, p ~ eps/kz; raises off-vocabulary)
# are imported below; `k_par_from_angle` keeps this module's public signature as a one-line
# delegate.  The resonance module imports nothing from here, so there is no cycle.  NOT
# bit-identical, by a stated tolerance: the square root itself is (0 of 2400 values differ when
# the shared kz is re-wrapped as np.complex128), but kz is now a Python complex, so downstream
# divisions such as eps/kz run CPython's complex division instead of numpy's.  Measured on 2400
# stack_rt / pole_function values (p and s, oblique, complex omega, GNOR): 647 differ, max
# relative change 2.4e-14.  Gate: test_nonlocal_tmm.py::test_q19_one_kz_admittance_kpar.
from dynameta.optics.resonance import _admittance, _kz  # noqa: E402
from dynameta.optics.resonance import k_par_from_angle as _res_k_par_from_angle  # noqa: E402


# The ONE polarization vocabulary of this module (audit V-3/V-8): the 'sp' family of
# dynameta.core.polarization, matching the stack_rt / pole_function checks -- and, since ACCEPTANCE
# UNIFICATION (b), its unconditional 'te'/'tm' + mixed-case aliases, normalized at the door exactly
# as in the resonance twin (the two must accept and reject the SAME set, or the V-3 divergence
# comes back through the front door).  The geometry-DEPENDENT lab 'x'/'y' and the integer `row`
# stay REFUSED.  No valid call moved: 's'/'p' never reach the guard.
_POL_NO_DEFAULT = ("There is no default here: 'x' -- and every other off-vocabulary label -- used "
                   "to return the s-polarized admittance silently, the MIRROR of the p-pol "
                   "fallthrough in optics.resonance (audit V-3). 'TE'/'TM' left that class in the "
                   "other direction: they are now ACCEPTED as unconditional aliases of 's'/'p'.")


def _reject_pol(pol, where: str):
    """Raise the shared V-8 vocabulary error for a ``pol`` the 'sp' vocabulary does not accept.

    LAZY import, failure path only: the valid path pays nothing and this module gains no import
    edge (dynameta.core.polarization is stdlib-only documentation + validation)."""
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


def _is_local_limit(layer: HydroLayer) -> bool:
    """True when ``beta_eff**2 = beta**2 + D*(gamma - i*omega)`` vanishes IDENTICALLY, i.e.
    ``beta == 0`` and ``D == 0``: no pressure, no diffusion, no longitudinal wave (k_L is
    infinite).  AUDIT Q-18: such a layer IS the local Drude metal the ``HydroLayer`` docstring
    promises -- the ABC matrix's longitudinal terms are all ``~ L1 ~ 1/k_z_L -> 0`` (with ``csc p
    -> 0``, ``cot p`` bounded), so its exact limit is the local Abeles matrix at ``eps_T`` -- but
    evaluated as written ``kL_squared`` divides by ``beta_eff**2 = 0`` and every p-pol call raised
    ``ZeroDivisionError``."""
    return layer.beta == 0 and layer.D == 0


def _layer_matrix(omega: complex, layer: NonlocalLayer, k0: complex, k_par: complex,
                  pol: str) -> np.ndarray:
    """Characteristic matrix for one layer.  Dielectric -> local; hydrodynamic metal in s-pol
    -> local with eps = eps_T (no longitudinal coupling in TE); hydrodynamic metal in p-pol ->
    the effective ABC matrix, or -- in the exact local limit ``beta = D = 0`` (audit Q-18) -- the
    local matrix at ``eps_T``, which is that matrix's beta -> 0 limit."""
    if isinstance(layer, DielectricLayer):
        return _local_char_matrix(complex(layer.eps), layer.thickness_m, k0, k_par, pol)
    if isinstance(layer, HydroLayer):
        if pol == "p" and not _is_local_limit(layer):
            return _hydro_char_matrix_p(omega, layer, k0, k_par)
        return _local_char_matrix(eps_transverse(omega, layer), layer.thickness_m,
                                  k0, k_par, pol)
    raise TypeError("layer must be a HydroLayer or DielectricLayer; got {!r}".format(type(layer)))


# ------------------------------------------------------------------------------------------------
# THE KERNEL: pressure model + interface conditions for graded hydrodynamic conductors (v0.16)
# ------------------------------------------------------------------------------------------------
#: The state that is CONTINUOUS at every internal interface of one conductor (p-pol, planar):
#: tangential E, tangential H, normal free current, and psi = delta(mu)/e (the linearized
#: chemical-potential perturbation in volts).  J_z = 0 only at a conductor boundary.
HYDRO_INTERFACE_FIELDS = ("E_x", "H_y", "J_z", "psi")

_BETA2_OVER_VF2 = {"high_freq": 3.0 / 5.0, "thomas_fermi": 1.0 / 3.0}


def fermi_velocity(n_m3, m_eff_kg: float):
    """Parabolic-band Fermi velocity ``v_F = hbar (3 pi**2 n)**(1/3) / m`` [m/s] of a degenerate
    electron gas of density ``n_m3`` [m**-3] (scalar or array) and effective mass ``m_eff_kg``."""
    n = np.asarray(n_m3, dtype=float)
    v = HBAR * np.cbrt(3.0 * math.pi ** 2 * n) / float(m_eff_kg)
    return float(v) if v.ndim == 0 else v


def beta_from_density(n_m3, m_eff_kg: float, convention: str = "high_freq"):
    """LOCAL hydrodynamic velocity ``beta(n0)`` [m/s]: :func:`beta_from_vf` of the local Fermi
    velocity :func:`fermi_velocity` (``beta**2 = n0 mu'(n0)/m``: v_F**2/3 for 'thomas_fermi', the
    9/5-rescaled (3/5) v_F**2 for 'high_freq').  Scalar or array."""
    v = np.asarray(fermi_velocity(n_m3, m_eff_kg), dtype=float)
    b = np.array([beta_from_vf(x, convention) for x in v.ravel()]).reshape(v.shape)
    return float(b) if b.ndim == 0 else b


def hydro_conductivity(omega, layer: HydroLayer) -> complex:
    """Drude conductivity ``sigma = eps0 wp**2 / (gamma - i omega)`` [S/m] of a hydrodynamic layer:
    the constitutive law of the kernel is ``J = sigma (E + grad psi)``.  ``sigma = -i omega eps0
    (eps_T - eps_inf)``."""
    w = complex(omega)
    return EPS0 * layer.wp * layer.wp / (layer.gamma - 1j * w)


def hydro_psi_per_charge(omega, layer: HydroLayer) -> complex:
    """``c`` in ``psi = c * rho1``: ``c = -beta_eff**2 / (eps0 wp**2)`` [V m**3 / C].

    ``psi = delta(mu)/e`` is the linearized chemical-potential perturbation in volts, the quantity
    that is continuous between sub-layers of one conductor.  With ``rho1 = div(J)/(i omega)`` this
    is ``psi = -beta_eff**2 div(J) / (i omega eps0 wp**2)``; for one effective mass across a run,
    continuity of psi is continuity of ``beta**2 div(J) / wp**2``."""
    if layer.wp == 0:
        raise ValueError("hydro_psi_per_charge: wp = 0 -- no free carriers, psi is undefined")
    return -beta_eff_squared(omega, layer) / (EPS0 * layer.wp * layer.wp)


def _delta_eps(w: complex, layer: HydroLayer) -> complex:
    """``eps_T - eps_inf = -wp**2/(omega**2 + i omega gamma)``, evaluated with the same operations as
    :func:`eps_transverse` so that ``eps_inf + _delta_eps`` IS ``eps_transverse`` bit for bit."""
    return -(layer.wp * layer.wp / (w * w + 1j * w * layer.gamma))


def _kz_decaying(x: complex) -> complex:
    """The square root of ``x`` with ``Im >= 0`` (the wave exp(+i kz z) decays, or propagates, toward
    +z).  Both signs are present in every layer, so the choice only fixes which face a wave is
    referenced to."""
    k = complex(np.sqrt(complex(x) + 0j))
    return -k if k.imag < 0.0 else k


def hydro_state_generator(omega, layer: HydroLayer, k_par) -> np.ndarray:
    """The 4x4 generator ``K`` of the p-pol hydrodynamic field equations in a uniform layer,
    ``d u / dz = K u``, for the CONTINUOUS state ``u = (E_x, H_y/(omega eps0), J_z/(omega eps0), psi)``
    (the reduced H and J share the factor omega*eps0, so continuity is unaffected).  Fields vary as
    exp(i k_par x - i omega t); z points from the superstrate into the stack.

    Derived from curl E = i omega mu0 H, curl H = -i omega eps0 eps_inf E + J, div J = i omega rho1,
    J = sigma (E + grad psi), psi = -beta_eff**2 rho1/(eps0 wp**2); with Delta = eps_T - eps_inf
    (so sigma = -i omega eps0 Delta) and k0 = omega/c::

        dE_x/dz = i (k0**2 - k_par**2/eps_inf) H~ + (k_par/eps_inf) J~
        dH~/dz  = i eps_T E_x - k_par Delta psi
        dJ~/dz  = -k_par Delta E_x - i (k_par**2 Delta + wp**2/beta_eff**2) psi
        dpsi/dz = (k_par/eps_inf) H~ + i (1/Delta + 1/eps_inf) J~

    Its eigenvalues are +-i k_z,T and +-i k_z,L (:func:`hydro_modes`).  For a graded conductor the
    same equations hold POINTWISE with wp(z), beta(z): this generator, evaluated at each z, is the
    ODE of the continuous problem, and (E_x, H_y, J_z, psi) is continuous across any interface
    inside the conductor."""
    if _is_local_limit(layer):
        raise ValueError("hydro_state_generator: beta = D = 0 is the local limit (no longitudinal "
                         "wave, psi is identically 0); use the local Abeles matrix")
    w = complex(omega)
    kp = complex(k_par)
    k0 = w / C_LIGHT
    dl = _delta_eps(w, layer)
    eps_t = layer.eps_inf + dl
    ei = layer.eps_inf
    g = layer.wp * layer.wp / beta_eff_squared(w, layer)
    return np.array([[0.0, 1j * (k0 * k0 - kp * kp / ei), kp / ei, 0.0],
                     [1j * eps_t, 0.0, 0.0, -kp * dl],
                     [-kp * dl, 0.0, 0.0, -1j * (kp * kp * dl + g)],
                     [0.0, kp / ei, 1j * (1.0 / dl + 1.0 / ei), 0.0]], dtype=np.complex128)


def _hydro_modes(w: complex, layer: HydroLayer, k0: complex, kp: complex):
    """Closed-form eigenvectors (columns) and z-wavevectors of :func:`hydro_state_generator`, order
    (T+, T-, L+, L-); Im(k_z) >= 0 for the '+' waves.  Transverse: H~ = 1, E_x = +-k_z,T/eps_T,
    J~ = i k_par Delta/eps_T, psi = 0.  Longitudinal (amplitude a = 1, E = (k_par, +-k_z,L) a,
    H = 0): J~ = +-i eps_inf k_z,L, psi = i eps_T/Delta."""
    dl = _delta_eps(w, layer)
    eps_t = eps_transverse(w, layer)
    kzt = _kz_decaying(eps_t * k0 * k0 - kp * kp)
    kzl = _kz_decaying(kL_squared(w, layer) - kp * kp)
    yinv = kzt / eps_t
    alpha = 1j * kp * dl / eps_t
    jl = 1j * layer.eps_inf * kzl
    psil = 1j * eps_t / dl
    phi = np.array([[yinv, -yinv, kp, kp],
                    [1.0, 1.0, 0.0, 0.0],
                    [alpha, alpha, jl, -jl],
                    [0.0, 0.0, psil, psil]], dtype=np.complex128)
    return phi, np.array([kzt, -kzt, kzl, -kzl], dtype=np.complex128)


def hydro_modes(omega, layer: HydroLayer, k_par):
    """``(Phi, kz)``: the four plane-wave solutions of a uniform hydrodynamic layer, columns of
    ``Phi`` in the state of :func:`hydro_state_generator`, ordered (T+, T-, L+, L-), with
    ``K @ Phi == Phi @ diag(1j * kz)``.  ``Im(kz) >= 0`` for the '+' waves."""
    if _is_local_limit(layer):
        raise ValueError("hydro_modes: beta = D = 0 is the local limit (no longitudinal wave)")
    w = complex(omega)
    return _hydro_modes(w, layer, w / C_LIGHT, complex(k_par))


# ------------------------------------------------------------------------------------------------
# Runs: grouping, validation, and the stable banded run solve
# ------------------------------------------------------------------------------------------------
def _hydro_blocks(layers: Sequence[NonlocalLayer], allow_walls: bool):
    """Group ``layers`` into blocks: a RUN (list of >= 2 adjacent HydroLayers with one non-None
    conductor_id) or a single layer (list of 1).  Refuses an ambiguous wall (an unlabelled
    HydroLayer directly against another HydroLayer) unless ``allow_walls``, a run mixing local-limit
    and nonlocal members, and a nonlocal run member without carriers."""
    blocks = []
    for i, layer in enumerate(layers):
        if isinstance(layer, HydroLayer) and blocks and isinstance(blocks[-1][-1], HydroLayer):
            a, b = blocks[-1][-1].conductor_id, layer.conductor_id
            if a is not None and a == b:
                blocks[-1].append(layer)
                continue
            if (a is None or b is None) and not allow_walls:
                raise ValueError(
                    "layers {} and {} are adjacent HydroLayers and at least one has conductor_id="
                    "None: a hard wall J_z = 0 would sit between them. Sub-layers of ONE conductor "
                    "(a uniform film cut in pieces, a graded density profile) must share a "
                    "conductor_id so that they form a connected run; two different conductors in "
                    "contact take different non-None ids. Pass allow_walls=True to keep the v0.15.1 "
                    "physics (a wall at every HydroLayer face).".format(i - 1, i))
        blocks.append([layer])
    for blk in blocks:
        if len(blk) > 1:
            local = [_is_local_limit(x) for x in blk]
            if any(local) and not all(local):
                raise ValueError(
                    "run conductor_id={!r} mixes local-limit members (beta = D = 0) with nonlocal "
                    "ones: psi would be forced to 0 on one side and free on the other (ill-posed). "
                    "Give every member a nonzero beta (or D), or make all of them local."
                    .format(blk[0].conductor_id))
            if not any(local) and any(not (x.wp > 0) for x in blk):
                raise ValueError("run conductor_id={!r}: every member of a nonlocal run needs "
                                 "wp > 0 (free carriers)".format(blk[0].conductor_id))
    return blocks


def _is_nonlocal_run(block) -> bool:
    return len(block) > 1 and not _is_local_limit(block[0])


def _hydro_run_matrix_p(omega: complex, run: Sequence[HydroLayer], k0: complex, k_par: complex,
                        interface: str = "electrochemical") -> np.ndarray:
    """p-pol 2x2 characteristic matrix (reduced (E_x, H_y/(omega eps0)) basis, top face -> bottom
    face) of a RUN of hydrodynamic sub-layers of one conductor.

    Unknowns: the four wave amplitudes of every sub-layer, each referenced at the face its wave
    decays away from ('+' waves at the top face, '-' waves at the bottom face), so every exponential
    in the system is exp(i kz d) with Im(kz) >= 0 -- nothing grows, however thin the sub-layers or
    however evanescent the longitudinal waves.  Equations (4N for N sub-layers): J_z = 0 on the
    top face, E_x and H~ prescribed on the top face (the two right-hand sides), (E_x, H~, J~, psi)
    continuous at each internal interface, J_z = 0 on the bottom face.  Banded (6 below, 4 above
    the diagonal), row- and column-equilibrated, solved with partial pivoting; the bottom-face
    (E_x, H~) of the two solutions are the matrix columns.

    ``interface='pressure'`` is a PRIVATE test hook that replaces psi continuity by continuity of
    the pressure term beta**2 n1 (proportional to wp**2 psi for one effective mass): the rejected
    alternative, kept only so that the gates can show they discriminate."""
    from scipy.linalg import solve_banded

    w = complex(omega)
    n_sub = len(run)
    n = 4 * n_sub
    tops, bots, wts = [], [], []
    for lay in run:
        phi, kz = _hydro_modes(w, lay, k0, complex(k_par))
        d = float(lay.thickness_m)
        et = np.exp(1j * kz[0] * d)
        el = np.exp(1j * kz[2] * d)
        tops.append(phi * np.array([1.0, et, 1.0, el])[None, :])
        bots.append(phi * np.array([et, 1.0, el, 1.0])[None, :])
        if interface == "electrochemical":
            wts.append(1.0)
        elif interface == "pressure":
            wts.append(lay.wp * lay.wp)
        else:
            raise ValueError("interface must be 'electrochemical' (or the test hook 'pressure')")

    # rows as (row index, first column, values)
    rows = [(0, 0, tops[0][2]), (1, 0, tops[0][0]), (2, 0, tops[0][1])]
    for k in range(n_sub - 1):
        for i in range(4):
            a, b = bots[k][i], tops[k + 1][i]
            if i == 3:
                a, b = a * wts[k], b * wts[k + 1]
            rows.append((3 + 4 * k + i, 4 * k, np.concatenate([a, -b])))
    rows.append((n - 1, n - 4, bots[-1][2]))
    rhs = np.zeros((n, 2), dtype=np.complex128)
    rhs[1, 0] = 1.0
    rhs[2, 1] = 1.0

    lo, up = 6, 4
    ab = np.zeros((lo + up + 1, n), dtype=np.complex128)
    for r, c0, vals in rows:
        s = np.max(np.abs(vals))
        s = 1.0 / s if s > 0.0 else 1.0
        rhs[r] *= s
        cols = np.arange(c0, c0 + vals.size)
        ab[up + r - cols, cols] = vals * s
    cs = np.max(np.abs(ab), axis=0)
    cs = np.where(cs > 0.0, 1.0 / np.where(cs > 0.0, cs, 1.0), 1.0)
    ab *= cs[None, :]
    x = solve_banded((lo, up), ab, rhs, check_finite=False) * cs[:, None]
    return bots[-1][0:2] @ x[n - 4:n]


def _stack_matrix(omega: complex, layers: Sequence[NonlocalLayer], k0: complex, k_par: complex,
                  pol: str, allow_walls: bool = False) -> np.ndarray:
    """Total characteristic matrix of the stack, mapping (E_tan, H_tan) from the SUPERSTRATE
    face to the SUBSTRATE face.  ``layers`` are ordered superstrate-side first, so
    ``M_total = M_{N} @ ... @ M_1`` (each layer post-multiplies onto the running product from
    the left).  A p-pol nonlocal RUN (>= 2 sub-layers of one conductor) enters as one block,
    :func:`_hydro_run_matrix_p`; every other layer enters exactly as in v0.15.1."""
    M = np.eye(2, dtype=np.complex128)
    for block in _hydro_blocks(layers, allow_walls):
        if pol == "p" and _is_nonlocal_run(block):
            M = _hydro_run_matrix_p(omega, block, k0, k_par) @ M
            continue
        for layer in block:
            M = _layer_matrix(omega, layer, k0, k_par, pol) @ M
    return M


# ------------------------------------------------------------------------------------------------
# Graded profile -> run, and the validity diagnostic
# ------------------------------------------------------------------------------------------------
def hydro_layers_from_profile(z_m, n_m3, *, m_eff_kg: float, eps_inf: float, gamma: float,
                              n_layers: Optional[int] = None, convention: str = "high_freq",
                              D: float = 0.0, conductor_id: Hashable = "graded") -> list:
    """A RUN of uniform hydrodynamic sub-layers representing the density profile ``n_m3`` sampled
    at ``z_m`` (``z_m[0]`` is the face nearest the superstrate, ``z_m`` strictly increasing).

    Each sub-layer carries the EXACT cell mean of the piecewise-linear interpolant of n(z) (so the
    sheet density of every cell, hence the integral of wp**2, is conserved), and from it
    ``wp = sqrt(n e**2/(eps0 m))`` and the LOCAL ``beta = beta_from_density(n, m, convention)``.
    ``n_layers=None`` makes one sub-layer per grid cell ``[z_i, z_i+1]``; an integer resamples the
    profile onto that many equal cells over ``[z_m[0], z_m[-1]]`` (the refinement knob: the
    staircase converges to the continuous profile at second order in the cell size).
    Parabolic band, one effective mass, scalar ``eps_inf`` / ``gamma`` / ``D``; build HydroLayers by
    hand (with one shared ``conductor_id``) for anything else.  Every sub-layer gets
    ``conductor_id``.  Densities must be positive; see :func:`hydro_validity` for where the
    model applies (a depletion layer that is no longer degenerate is outside it)."""
    z = np.asarray(z_m, dtype=float)
    nn = np.asarray(n_m3, dtype=float)
    if z.ndim != 1 or nn.shape != z.shape or z.size < 2:
        raise ValueError("z_m and n_m3 must be 1-D arrays of the same length >= 2")
    if not (np.all(np.isfinite(z)) and np.all(np.isfinite(nn))):
        raise ValueError("z_m and n_m3 must be finite")
    if np.any(np.diff(z) <= 0.0):
        raise ValueError("z_m must be strictly increasing")
    if np.any(nn <= 0.0):
        raise ValueError("n_m3 must be > 0 everywhere (a run member needs free carriers)")
    if conductor_id is None:
        raise ValueError("conductor_id must not be None: the sub-layers must form one run")
    if n_layers is None:
        edges = z
        nav = 0.5 * (nn[:-1] + nn[1:])
    else:
        if int(n_layers) != n_layers or int(n_layers) < 1:
            raise ValueError("n_layers must be a positive integer")
        edges = np.linspace(z[0], z[-1], int(n_layers) + 1)
        cum = np.concatenate([[0.0], np.cumsum(0.5 * (nn[:-1] + nn[1:]) * np.diff(z))])
        seg = np.clip(np.searchsorted(z, edges, side="right") - 1, 0, z.size - 2)
        F = cum[seg] + 0.5 * (edges - z[seg]) * (nn[seg] + np.interp(edges, z, nn))
        nav = np.diff(F) / np.diff(edges)
    m = float(m_eff_kg)
    out = []
    for nk, dk in zip(nav, np.diff(edges)):
        wp = math.sqrt(nk * Q_E * Q_E / (EPS0 * m))
        out.append(HydroLayer(float(eps_inf), wp, float(gamma),
                              beta_from_density(float(nk), m, convention), float(dk),
                              D=float(D), conductor_id=conductor_id))
    return out


class HydroValidity(NamedTuple):
    """Result of :func:`hydro_validity`.

    Attributes
    ----------
    u : tuple of float
        Per input layer, ``|k_L| v_F / |omega|`` of its longitudinal wave (``nan`` for dielectric and
        local-limit layers), with ``v_F = beta / sqrt(c)`` for the stated beta convention.
    degeneracy : tuple of float
        Per input layer, ``E_F / (k_B T)`` (``nan`` unless ``m_eff_kg`` and ``T_K`` were given).
    ok : bool
        True when every hydrodynamic layer is inside the limits.
    message : str
        Empty when ``ok``; otherwise names the worst layer and the limit it breaks.
    """

    u: tuple
    degeneracy: tuple
    ok: bool
    message: str


def hydro_validity(omega, layers: Sequence[NonlocalLayer], *, convention: str = "high_freq",
                   u_limit: Optional[float] = None, m_eff_kg: Optional[float] = None,
                   T_K: Optional[float] = None, degeneracy_min: float = 10.0) -> HydroValidity:
    """Report whether a hydrodynamic solve at ``omega`` is inside the model's validity.

    For every nonlocal HydroLayer, ``u = |k_L| v_F / |omega|`` (``k_L`` the full longitudinal
    wavevector, :func:`kL_squared`; ``v_F = beta/sqrt(3/5)`` or ``beta/sqrt(1/3)`` for ``convention``
    'high_freq' / 'thomas_fermi', i.e. the convention the betas were built with).

    * 'high_freq' (the dynamic, omega >> k v_F expansion): needs ``u <= u_limit`` (default 0.5).
      At u = 0.5 the first term the model misses, (3/7 - 9/25) u**4 of the Lindhard expansion, is
      0.4 %; at u -> 1 the wave reaches the particle-hole continuum (Landau damping, absent from
      the model) and the expansion fails.
    * 'thomas_fermi' (the static-screening limit): needs ``u >= u_limit`` (default 10), where the
      neglected Landau damping, ~ pi/(2u) relative, is below ~16 %.

    With ``m_eff_kg`` and ``T_K`` also given, ``E_F / k_B T`` is reported per layer (``n`` from
    ``wp`` and the mass) and must be >= ``degeneracy_min`` (default 10: the Sommerfeld correction to
    the degenerate pressure, (5 pi**2/12)(k_B T/E_F)**2, is then 4 %).  Pure diagnostic: nothing
    is changed; ``stack_rt(check_validity=True)`` warns on a failed report."""
    if convention not in _BETA2_OVER_VF2:
        raise ValueError("convention must be 'high_freq' or 'thomas_fermi'; got {!r}"
                         .format(convention))
    hf = convention == "high_freq"
    lim = float(u_limit) if u_limit is not None else (0.5 if hf else 10.0)
    w = complex(omega)
    us, degs, bad = [], [], []
    for i, layer in enumerate(layers):
        if not isinstance(layer, HydroLayer) or _is_local_limit(layer):
            us.append(float("nan"))
            degs.append(float("nan"))
            continue
        vf = layer.beta / math.sqrt(_BETA2_OVER_VF2[convention])
        u = math.sqrt(abs(kL_squared(w, layer))) * vf / abs(w)
        us.append(u)
        if (hf and u > lim) or (not hf and u < lim):
            bad.append((abs(math.log(u / lim)) if u > 0 else float("inf"),
                        "layer {}: u = |k_L| v_F/omega = {:.3g} {} {:.3g} ({} beta)".format(
                            i, u, ">" if hf else "<", lim, convention)))
        if m_eff_kg is not None and T_K is not None:
            nk = EPS0 * float(m_eff_kg) * layer.wp * layer.wp / (Q_E * Q_E)
            ef = HBAR * HBAR * (3.0 * math.pi ** 2 * nk) ** (2.0 / 3.0) / (2.0 * float(m_eff_kg))
            dg = ef / (KB * float(T_K))
            degs.append(dg)
            if dg < degeneracy_min:
                bad.append((math.log(degeneracy_min / dg) if dg > 0 else float("inf"),
                            "layer {}: E_F/k_B T = {:.3g} < {:.3g} (not degenerate)".format(
                                i, dg, degeneracy_min)))
        else:
            degs.append(float("nan"))
    if not bad:
        return HydroValidity(tuple(us), tuple(degs), True, "")
    bad.sort(key=lambda t: -t[0])
    msg = ("hydrodynamic solve outside the model's validity in {} check(s); worst: {}. The "
           "hydrodynamic (long-wavelength) electron-gas response does not hold there (Landau "
           "damping / non-degenerate gas); treat the result as qualitative.".format(
               len(bad), bad[0][1]))
    return HydroValidity(tuple(us), tuple(degs), False, msg)


# ------------------------------------------------------------------------------------------------
# Reflectance / transmittance / absorptance
# ------------------------------------------------------------------------------------------------
class RTA(NamedTuple):
    """Result of :func:`stack_rt`.

    Attributes
    ----------
    r, t : complex
        Amplitude reflection / transmission coefficients (tangential-E ratios; for p-polarization
        r is the NEGATIVE of the library's p-hat r_p = r_H, see :func:`stack_rt`). The same
        tuple returned by ``optics.quantum_film.film_stack_rt`` carries the p-hat r = r_H instead
        (its docstring).
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
    when continuing to complex omega (the QNM/pole convention).  Delegates to the ONE
    implementation, :func:`dynameta.optics.resonance.k_par_from_angle` (audit Q-19); kept as a
    wrapper only so this module's ``omega`` keyword keeps working."""
    return _res_k_par_from_angle(n_super, omega, theta_rad)


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


def stack_rt(omega, layers: Sequence[NonlocalLayer], *, pol: str = "p", n_super=1.0, n_sub=1.0,
             theta_rad: float = 0.0, k_par_m=None, check_energy: bool = True,
             allow_walls: bool = False, check_validity: bool = False) -> RTA:
    """Reflectance / transmittance / absorptance of ``super | layers | sub`` at ``omega``.

    SIGN CONVENTION OF r FOR p-POLARIZATION (unchanged since v0.15.1): ``r`` and ``t`` are
    TANGENTIAL-E ratios, r = E_x(refl) / E_x(inc). For p-polarization this is the NEGATIVE of the
    library's p-hat convention r_p = r_H = H_y(refl) / H_y(inc) used by ``optics.fem_2d_tm``
    (polarization 'p'), the lumenairy bridges, ``optics.tmm_reference``, ``optics.quantum_film_cell``
    and ``optics.quantum_film.film_stack_rt`` / ``QuantumFilmResponse.stack_rt``: the same planar
    p-pol stack gives ``nonlocal_tmm.stack_rt(...).r == -fem_2d_tm(...).r`` (to the routes'
    discretization; exactly ``-film_stack_rt(...).r`` for the same layers). Negate ``r`` before
    comparing with those routes. s-polarization (E_y ratio) is the same in every route. R, T, A and
    ``D_pole`` do not depend on the sign.

    Parameters
    ----------
    omega : float or complex
        Angular frequency [rad/s].
    layers : sequence of HydroLayer / DielectricLayer
        Ordered from the superstrate side to the substrate side.  Adjacent HydroLayers with one
        non-None ``conductor_id`` form a RUN (one connected electron gas; see the module header).
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
    check_energy : bool
        ENERGY-BUDGET GUARD (audit Q-23, 2026-11-02): the ``tmm_reference`` policy, through its
        ``_check_energy_budget`` -- on the REAL axis with a propagating lossless incidence, a
        non-finite or < -1e-6 R, T or A raises ValueError instead of returning a negative
        "absorptance" (an interior gain layer, ``gamma < 0`` or ``Im(eps) < 0``, used to pass
        silently here while the TMM oracle raised).  Complex ``omega``, evanescent incidence and
        a lossy superstrate are not checked.  ``False`` for a deliberate gain medium.
    allow_walls : bool
        Opt-in to the v0.15.1 physics for unlabelled adjacent HydroLayers (a hard wall J_z = 0
        between them).  Without it such a pair raises ValueError (v0.16: walls only at conductor
        boundaries).
    check_validity : bool
        Run :func:`hydro_validity` (default limits, 'high_freq' convention) for p-pol and emit an
        UnreliableResultWarning when the solve is outside the hydrodynamic model's validity.

    Returns
    -------
    RTA
    """
    if pol not in ("p", "s"):
        pol = _accept_pol(pol, "stack_rt")               # audit V-3/V-8: one vocabulary, one message
    w = complex(omega)
    k0 = w / C_LIGHT
    eps_super = complex(n_super) ** 2
    eps_sub = complex(n_sub) ** 2
    if k_par_m is None:
        k_par = complex(n_super) * k0 * math.sin(theta_rad)
    else:
        k_par = complex(k_par_m)

    M = _stack_matrix(w, layers, k0, k_par, pol, allow_walls=allow_walls)
    if check_validity and pol == "p":
        report = hydro_validity(w, layers)
        if not report.ok:
            from dynameta.warning_types import UnreliableResultWarning
            warnings.warn(report.message, UnreliableResultWarning, stacklevel=2)
    Y_super = _admittance(eps_super, _kz(eps_super, k0, k_par), pol)
    Y_sub = _admittance(eps_sub, _kz(eps_sub, k0, k_par), pol)

    r, t, D_pole = _rt_from_matrix(M, Y_super, Y_sub)
    R = float(abs(r) ** 2)
    denom = Y_super.real
    T = float(abs(t) ** 2 * (Y_sub.real / denom)) if denom != 0.0 else float("nan")
    if check_energy and w.imag == 0.0 and denom > 0.0 and eps_super.imag == 0.0:
        _check_energy_budget(R, T, where="nonlocal_tmm.stack_rt")            # audit Q-23
    A = 1.0 - R - T
    return RTA(r=r, t=t, R=R, T=T, A=A, D_pole=D_pole)


def rta(omega, layers: Sequence[NonlocalLayer], **kwargs):
    """Convenience wrapper returning just ``(R, T, A)`` from :func:`stack_rt`."""
    res = stack_rt(omega, layers, **kwargs)
    return res.R, res.T, res.A


# ------------------------------------------------------------------------------------------------
# Pole function (compatible with optics.resonance.find_poles)
# ------------------------------------------------------------------------------------------------
def pole_function(layers: Sequence[NonlocalLayer], *, pol: str = "p", n_super=1.0, n_sub=1.0,
                  k_par_m=0.0, allow_walls: bool = False) -> Callable[[complex], complex]:
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
    wavevectors (even in ``k_z_T`` / ``k_z_L``); the omega-plane non-analyticities are the
    end-media light lines, for p-pol the ``eps_T = 0`` admittance point, AND (audit Q-26,
    2026-11-02 -- this list used to stop at the first two) the ZEROS of each hydrodynamic layer's
    ``d00 = C_T - L1 sin(q) csc(p)`` (``_hydro_char_matrix_p``: every entry of the layer matrix
    carries ``1/d00``).  Those are NOT removed by the ``sin(k_z_L d)`` clearing: they sit just off
    the standing-wave points, where ``sin p = L1 sin(q) / C_T`` is small, and can be simple poles
    of the returned function.  Measured on a 2 nm Drude film (eps_inf 1, wp 1.37e16, gamma 1e14,
    beta 1.07e6, k_par 2e7): at m = 1 the d00 zero is at p/pi = 1.00016 and ``|D * d00|`` is
    constant to 2e-4 for offsets 1e-6..1e-8 (a simple pole of D, inside any box drawn around the
    m = 1 mode); at m = 2 (p/pi = 1.99992) the residue vanishes and D stays finite.  Place the
    search box to exclude all of them (as ``resonance.find_poles`` already requires for the light
    lines of local stacks), or check the winding count against a refinement.  Intended
    for GENUINE nonlocal stacks (finite ``beta``); in the ``beta -> 0`` local limit there is no
    bulk-plasmon pole and ``sin(k_z_L d)`` overflows (bulk plasmons do not exist there).  The EXACT
    local limit ``beta = D = 0`` is handled (audit Q-18): such a layer is the local Drude layer
    and contributes no clearing factor.

    RUNS (v0.16): the adjacency rules of :func:`stack_rt` apply (``allow_walls``).  A p-pol
    NONLOCAL run of >= 2 sub-layers is REFUSED (NotImplementedError): its matrix has its own
    standing-wave poles, which have no analytic clearing factor analogous to sin(k_z,L d) here
    (the natural candidate, the determinant of the run's global system, depends on which face each
    wave is referenced to and is not analytic in omega).  Locate such modes on the real axis with
    :func:`stack_rt`.  Single-layer runs, local-limit runs and s-pol are unaffected."""
    if pol not in ("p", "s"):
        pol = _accept_pol(pol, "pole_function")          # audit V-3/V-8: one vocabulary, one message
    blocks = _hydro_blocks(layers, allow_walls)
    if pol == "p" and any(_is_nonlocal_run(b) for b in blocks):
        raise NotImplementedError(
            "pole_function does not support a p-pol run of >= 2 nonlocal sub-layers of one "
            "conductor (no analytic clearing of the run's standing-wave poles); scan stack_rt on "
            "the real axis instead")
    kpar = complex(k_par_m)
    eps_super = complex(n_super) ** 2
    eps_sub = complex(n_sub) ** 2
    # p-pol hydrodynamic layers whose csc pole must be cleared from the pole function.
    # A layer in the exact local limit (beta = D = 0, audit Q-18) has no ABC and no csc pole.
    hydro_p = [layer for layer in layers if pol == "p" and isinstance(layer, HydroLayer)
               and not _is_local_limit(layer)]

    def D(omega):
        w = complex(omega)
        k0 = w / C_LIGHT
        M = _stack_matrix(w, layers, k0, kpar, pol, allow_walls=allow_walls)
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


def __getattr__(name):
    # audit V-11 (2026-11-02): `Layer` was this module's name for the HydroLayer | DielectricLayer UNION type, while
    # `dynameta.Layer` / `dynameta.geometry.Layer` is the Design dataclass -- one name bound to three
    # KINDS of object (the dataclass, resonance's (eps, d) tuple alias, nonlocal_tmm's union alias),
    # two of them in public signatures. The alias is now `NonlocalLayer`; the old spelling still resolves,
    # with a DeprecationWarning (it was never in __all__ and had zero readers outside this module).
    if name == "Layer":
        import warnings
        from dynameta.warning_types import DynaMetaDeprecationWarning
        warnings.warn("dynameta.optics.nonlocal_tmm.Layer is deprecated (audit V-11): it is not dynameta.Layer; use "
                      "dynameta.optics.nonlocal_tmm.NonlocalLayer", DynaMetaDeprecationWarning, stacklevel=2)
        return NonlocalLayer
    raise AttributeError("module {!r} has no attribute {!r}".format(__name__, name))
