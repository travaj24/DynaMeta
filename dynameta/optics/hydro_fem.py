"""Hydrodynamic (nonlocal) Drude FINITE-ELEMENT tier -- roadmap items 3.3 (HDM -> GNOR -> QCM)
and 5.4 (2-D coupled-HDM STABILIZATION).

Item 2.4 (``optics.nonlocal_tmm``) solved the LAYERED hydrodynamic problem with a 4-wave
transfer matrix.  This module carries the SAME physics into a FINITE-ELEMENT weak form -- the
coupled ``(E, J)`` system the generalizes beyond planar stacks -- plus the quantum-corrected
(QCM) sub-nanometre gap material.  It reuses ``nonlocal_tmm``'s conventions EXACTLY (same
``beta**2 = (3/5) v_F**2``, same GNOR ``beta_eff**2 = beta**2 + D(gamma - i*omega)``, same
``exp(-i*omega*t)`` so a passive absorber has ``Im(eps) > 0``, the ABC ``J_normal = 0``).

The 2-D scatterer solver (:func:`scattering_2d`, ``local=False``) is the STABILIZED reformulation
of roadmap item 5.4 -- the scalar-longitudinal-potential form derived below.  It is stable at
plasmon resonances and in sub-5-nm gaps (the original vector-``J`` form was indefinite and blew
up there); it delivers the CYLINDER-BLUESHIFT and GAP-SATURATION gates quantitatively.

------------------------------------------------------------------------------------------------
THE COUPLED (E, J) WEAK FORM (Toscano et al., Opt. Express 20, 4176 (2012))
------------------------------------------------------------------------------------------------
In a hydrodynamic metal the free-electron current ``J`` is an independent field obeying its own
equation of motion (the electron-gas pressure ``beta**2`` adds a longitudinal / compressional
wave).  The frequency-domain strong form, SI, ``exp(-i*omega*t)``:

    curl curl E - (omega/c)**2 eps_inf E = i*omega*mu0 J                  in the metal,   (M)
    beta**2 grad(div J) + omega(omega + i*gamma) J = i*omega*eps0*wp**2 E in the metal,   (H)
    curl curl E - (omega/c)**2 eps_b E = 0                                elsewhere,

with the ADDITIONAL BOUNDARY CONDITION (ABC) ``J.n = 0`` on every metal surface (the hard-wall
Melnyk-Harris / Sipe model; no electron spill-out).  ``eps_inf`` is the bound-background
permittivity of the metal; the free response lives entirely in ``J``.  The LOCAL Drude limit is
recovered as ``beta -> 0``: (H) gives ``J = i*eps0*wp**2 E/(omega + i*gamma)`` pointwise, and
substituting into (M) yields ``eps(omega) = eps_inf - wp**2/(omega**2 + i*omega*gamma)`` -- this
library's ``resonance.drude_eps`` / ``nonlocal_tmm.eps_transverse`` (passive: ``Im(eps) > 0``).

Galerkin weak form (test ``v`` for E in H(curl), test ``w`` for J with the ABC ``w.n = 0``
ESSENTIAL, so the boundary term ``surf-int (div J)(w.n)`` from integrating ``grad(div J)`` by
parts VANISHES):

    int[ curl E . curl v - k0**2 eps E.v ] dV  -  i*omega*mu0 int_metal J.v dV  = src_E(v),
    -beta_eff**2 int_metal (div J)(div w) dV + omega(omega+i*gamma) int_metal J.w dV
        - i*omega*eps0*wp**2 int_metal E.w dV  =  src_J(w),

with ``beta_eff**2 = beta**2 + D(gamma - i*omega)`` the GNOR knob (Mortensen et al. 2014).  A
SCATTERED-field formulation is used: ``E`` is the scattered field, ``E_inc`` the analytic
incident plane wave, and the metal carries the sources ``src_E = k0**2 (eps_inf - eps_b) E_inc``,
``src_J = i*omega*eps0*wp**2 E_inc``.

------------------------------------------------------------------------------------------------
THE 1-D LAYERED SOLVER (robust; item 3.3)
------------------------------------------------------------------------------------------------
``hydro_layered_1d`` -- a 1-D-in-z coupled ``(E, J)`` FEM for a LAYERED stack.  The screening layer
is trivially resolved (fine z-mesh), so this solver is ROBUST.  At NORMAL incidence it reproduces
``nonlocal_tmm`` R/T/A to ~1e-8 relative (mesh/order-limited, NOT machine precision: measured worst
5.5e-8 over 0.4-1.3 wp at an off-gate parameter set, ~1e-10 well below wp; there is no longitudinal
coupling at normal incidence -- it validates the local reduction, the ABC bookkeeping, the units and
the transparent boundary condition).  At OBLIQUE incidence it reproduces the BULK-PLASMON
standing-wave absorption peaks at ``k_L d = m*pi`` to < 1% (the core nonlocal physics -- the pressure
term and ABC); its ABSOLUTE absorption error grows with angle because a single scalar impedance BC
cannot fully absorb the oblique VECTOR p-pol wave in a nodal discretization: measured ~0.5-3% at
30 deg and up to ~20% at 60 deg near 0.7 wp (documented, not gated -- the peak POSITIONS validate
the physics).

------------------------------------------------------------------------------------------------
THE 2-D SCATTERER SOLVER (item 5.4): why the vector-J form failed, and the cure that fixed it
------------------------------------------------------------------------------------------------
THE ORIGINAL OBSTRUCTION (the indefinite vector-J block -- the spectral analysis).  The shipped
3.3 form carried ``J`` as a full vector in ``H(div)`` and the metal J-block bilinear form was
``omega(omega+i*gamma)|J|**2 - beta_eff**2 |div J|**2``.  Helmholtz-decompose ``J = J_T + grad
phi`` (``div J_T = 0``): the LONGITUDINAL part ``grad phi`` sees ``-beta_eff**2 |div J|**2`` (a
stiff, high-wavenumber operator whose screening length ``delta_L ~ beta/sqrt(wp**2/eps_inf -
omega**2) ~ 0.1-0.2 nm`` is FAR below the geometric scale), while the TRANSVERSE part ``J_T`` sees
ONLY the mass term ``omega(omega+i*gamma)|J_T|**2`` -- there is NOTHING to make the two parts
comparable, so the block's spectrum straddles zero (INDEFINITE) and its essential spectrum has an
accumulation point where ``omega(omega+i*gamma) - beta_eff**2 k**2 = 0`` for the discrete
longitudinal wavenumbers ``k`` the mesh supports.  Driving at a generic ``omega`` lands ARBITRARILY
CLOSE to one of those discrete longitudinal eigenvalues -> a near-null internal mode -> ``||E_scat||
/||E_inc||`` of 1e19-1e53 at plasmon resonances and in sub-5-nm gaps.  Refining the surface mesh
DENSIFIES the discrete longitudinal spectrum (more near-degeneracies), so it does not cure the
problem -- it can worsen it.  (This is why 3.3 gated ``scattering_2d`` to RAISE.)

CURE (a) -- THE SCALAR-LONGITUDINAL-POTENTIAL REFORMULATION (the one that worked).  Investigated in
the roadmap's order; (a) succeeded on the first try, so (b) grad-div augmentation, (c) static
condensation / complex-shifted factorization, and (d) first-order least squares were NOT needed
(their trade-offs are noted at the end).  The idea: eliminate the transverse ``J`` ANALYTICALLY and
keep only a SCALAR longitudinal unknown, turning the indefinite vector block into a well-understood
scalar Helmholtz.

  In a homogeneous metal the longitudinal content of (H) is a scalar Helmholtz.  Take ``div`` of
  (H) and set ``psi := div J`` (``grad div = lap`` on gradients):

      beta_eff**2 lap(psi) + Omega psi = i*omega*eps0*wp**2 div E ,   Omega := omega(omega+i*gamma).
                                                                                                 (P)

  Solve (H) pointwise for ``J`` and substitute into (M).  ``J = (i*omega*eps0*wp**2/Omega) E -
  (beta_eff**2/Omega) grad psi``; feeding this into ``curl curl E - k0**2 eps_inf E = i*omega*mu0 J``
  and using ``omega**2 mu0 eps0 = k0**2`` collapses the transverse free response into the LOCAL
  Drude permittivity ``eps_T = eps_inf - wp**2/Omega``:

      curl curl E - k0**2 eps_T(omega) E = -(i*omega*mu0 beta_eff**2/Omega) grad psi .            (M')

  Self-consistency check: take ``div`` of (M') (``div curl curl == 0``) to get ``div E`` in terms of
  ``lap psi`` and back-substitute into (P); the coupled system's longitudinal wavenumber comes out
  EXACTLY ``k_L**2 = (Omega - wp**2/eps_inf)/beta_eff**2`` -- IDENTICAL to ``nonlocal_tmm.kL_squared``.

  THE COUPLED WEAK FORM.  ``E`` in ``H(curl)`` (whole domain, PEC 'outer', PML on 'pml'); ``psi`` in
  ``H1`` on the METAL only.  Test ``v`` (H(curl)) and ``w`` (H1).  Integrate the ``lap psi`` term in
  (P) by parts AND the ``div E`` source by parts (mandatory: ``H(curl)`` ``E`` has no ``L2``
  divergence); the two resulting metal-surface integrals CANCEL once the ABC is applied, leaving:

      int_all [curl E . curl v - k0**2 eps(x) E.v] dV
          + (i*omega*mu0 beta_eff**2/Omega) int_metal grad(psi) . v dV                = f_E(v),
      int_metal [-beta_eff**2 grad(psi).grad(w) + Omega psi w] dV
          + i*omega*eps0*wp**2 int_metal E . grad(w) dV                               = f_psi(w),

  with ``eps(x) = eps_T`` in metal, ``eps_host`` outside.  Scattered-field sources (``E`` scattered,
  ``E_inc`` the analytic plane wave): ``f_E = k0**2 int_metal (eps_T - eps_host) E_inc . v`` and
  ``f_psi = -i*omega*eps0*wp**2 int_metal E_inc . grad(w)``.  (Implementation applies a symmetric
  numerical rescaling ``psi = S psi_hat`` + equation scale so the assembled matrix is
  complex-SYMMETRIC and every entry is O(1e-3..1) -- a well-conditioned direct ``umfpack`` solve.)

  THE ABC AS A NATURAL CONDITION.  The hard-wall ABC ``J.n = 0`` is, via ``J`` above,
  ``grad(psi).n = (i*omega*eps0*wp**2/beta_eff**2) E.n`` on the metal surface.  Integrating (P)'s
  ``lap psi`` by parts produces ``+int_surf beta_eff**2 (grad psi.n) w`` and integrating the ``div E``
  source by parts produces ``-int_surf i*omega*eps0*wp**2 (E.n) w``; substituting the ABC makes them
  EQUAL AND OPPOSITE, so they cancel and ``psi`` needs NO essential (dirichlet) constraint -- the ABC
  is enforced NATURALLY by the clean weak form (verify: the natural BC of the surface-free form is
  exactly the ABC).  This is the crux of the stabilization: the transverse block is gone (folded into
  ``eps_T``), the ``E`` block is now the ROBUST local-Drude curl-curl operator, and the only extra
  unknown ``psi`` obeys a scalar dissipative Helmholtz (``Im Omega = omega*gamma > 0`` -> invertible,
  not indefinite).  ``beta -> 0`` smoothly decouples ``psi`` (coupling ``~ beta_eff**2 -> 0``) and
  recovers the local solver EXACTLY (machine-precision local limit, unconditionally stable).

WHY (b)-(d) WERE NOT NEEDED (documented alternatives).  (b) grad-div augmentation
``+ s (div J)(div w)`` stabilizes the transverse null space but leaves the stiff longitudinal
operator on the full vector field and needs a swept ``s`` per regime; (c) a complex-shifted direct
factorization moves off the discrete-eigenvalue accumulation but re-solves the same large
vector system every frequency; (d) a first-order least-squares form is unconditionally coercive but
squares the condition number and doubles the unknowns.  (a) is strictly better here: it removes the
bad block ANALYTICALLY, shrinks the extra unknown from a vector to a scalar, and yields a symmetric
positive-definite-in-structure Helmholtz that any direct solver handles.

VALIDATED GATES (item 5.4 success criteria, all ngsolve-gated in tests/test_hydro_fem.py):
  * CYLINDER BLUESHIFT -- the coupled dipole-SP peak MINUS the local peak (same mesh) matches the
    quasistatic Raza closed form :func:`cylinder_blueshift_raza` to ~3-10% over R = 2.5-4 nm (well
    inside the roadmap's 15%); positive (blue) and ~1/R.
  * GAP SATURATION -- a metal-metal dimer swept ~10 -> 2 nm: the local/hydro gap-centre enhancement
    RATIO grows MONOTONICALLY as the gap shrinks (the local field diverges ~1/gap while the HDM caps
    it), and the coupled solve stays BOUNDED at 2 nm (the old blow-up regime).
  * LOCAL LIMIT -- ``beta -> 0`` reproduces the 2-D local-Drude solve (machine-close).
  * BOUNDED NORMS / ENERGY -- ``||E_scat||/||E_inc||`` stays O(1) at resonance and in tight gaps;
    ``P_abs, P_scat >= 0`` and the total-field flux balances ``-P_abs``.

The instability guard (:class:`HydroFEMUnstable`) is KEPT as a safety net for meshes that cannot
resolve the screening length at all, or for an aggressively-set ``unstable_ratio``.

------------------------------------------------------------------------------------------------
QCM -- the quantum-corrected gap material (Esteban et al., Nat. Commun. 3, 825 (2012))
------------------------------------------------------------------------------------------------
Below ~1 nm the two metal surfaces of a gap exchange electrons by TUNNELING; the classical (local
or hydrodynamic) field enhancement, which keeps growing as the gap closes, is instead SHORTED
OUT.  The QCM replaces the vacuum gap by an effective medium of permittivity ``eps_QCM(omega,
gap)`` that (i) is vacuum for large gaps (no tunneling) and (ii) turns metallic (conducting) as
the gap closes, so the enhancement is NON-MONOTONIC in gap size and PEAKS near ~1 nm -- the Esteban
signature.  See :class:`QCMGapMaterial`.  Usable in BOTH this solver and any standard local FEM.

Conventions: SI units, ``exp(-i*omega*t)`` (passive absorber ``Im(eps) > 0``), pure numpy/scipy
cores, NGSolve imported LAZILY (only the FEM solvers need it), ASCII-only.

References
----------
* C. Toscano, J. Straubel, A. Kwiatkowski, C. Rockstuhl, F. Evers, H. Xu, N. A. Mortensen,
  M. Wubs, "Resonance shifts and spill-out effects in self-consistent hydrodynamic nanoplasmonics"
  / G. Toscano et al., "Modified field enhancement and extinction by plasmonic nanowire dimers due
  to nonlocal response", Opt. Express 20, 4176 (2012) -- the coupled (E, J) FEM weak form.
* S. Raza, S. I. Bozhevolnyi, M. Wubs, N. A. Mortensen, "Nonlocal optical response in metallic
  nanostructures", J. Phys.: Condens. Matter 27, 183204 (2015) -- slab/cylinder closed forms.
* N. A. Mortensen, S. Raza, M. Wubs, T. Sondergaard, S. I. Bozhevolnyi, Nat. Commun. 5, 3809
  (2014) -- GNOR (the ``beta**2 -> beta**2 + D(gamma - i*omega)`` diffusion knob).
* R. Esteban, A. G. Borisov, P. Nordlander, J. Aizpurua, "Bridging quantum and classical
  plasmonics with a quantum-corrected model", Nat. Commun. 3, 825 (2012) -- the QCM gap material.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import NamedTuple, Optional

import numpy as np

from dynameta.constants import C_LIGHT, EPS0, MU0

__all__ = [
    "beta_from_vf",
    "HydroParams",
    "drude_eps",
    "QCMGapMaterial",
    "HydroFEMUnstable",
    "LayeredResult",
    "hydro_layered_1d",
    "bulk_plasmon_omega",
    "cylinder_sp_omega",
    "cylinder_blueshift_raza",
    "ScatterResult",
    "cylinder_mesh",
    "dimer_mesh",
    "dimer_gap_mesh",
    "gap_enhancement_2d",
    "scattering_2d",
    "sp_resonance_omega",
]

Z0 = math.sqrt(MU0 / EPS0)          # free-space wave impedance [ohm]
_L0 = 1.0e-9                        # length unit: mesh coordinates are in nm


# ================================================================================================
# Material physics (mirrors nonlocal_tmm conventions EXACTLY)
# ================================================================================================
def beta_from_vf(v_f: float, convention: str = "high_freq") -> float:
    """Hydrodynamic velocity ``beta`` [m/s] from the Fermi velocity (see nonlocal_tmm):
    ``"high_freq"`` -> ``sqrt(3/5) v_f`` (the optical choice); ``"thomas_fermi"`` -> ``sqrt(1/3) v_f``."""
    v = float(v_f)
    if convention == "high_freq":
        return math.sqrt(3.0 / 5.0) * v
    if convention == "thomas_fermi":
        return math.sqrt(1.0 / 3.0) * v
    raise ValueError("convention must be 'high_freq' or 'thomas_fermi'; got {!r}".format(convention))


@dataclass(frozen=True)
class HydroParams:
    """Hydrodynamic-Drude metal parameters (SI, rad/s).

    Attributes
    ----------
    eps_inf : float
        Bound-background permittivity.
    wp : float
        Plasma frequency ``omega_p`` [rad/s].
    gamma : float
        Collision rate [rad/s] (>= 0; passive under ``exp(-i*omega*t)``).
    beta : float
        Hydrodynamic velocity [m/s] (build with :func:`beta_from_vf`).  ``beta -> 0`` is local Drude.
    D : float
        GNOR diffusion constant [m**2/s] (default 0).  Enters as ``beta_eff**2 = beta**2 +
        D(gamma - i*omega)`` -- IDENTICAL to ``nonlocal_tmm.beta_eff_squared``.
    """

    eps_inf: float
    wp: float
    gamma: float
    beta: float
    D: float = 0.0

    def eps_transverse(self, omega) -> complex:
        """Transverse (ordinary, LOCAL) Drude permittivity ``eps_inf - wp**2/(omega**2 +
        i*omega*gamma)`` -- identical to ``nonlocal_tmm.eps_transverse``."""
        w = complex(omega)
        return self.eps_inf - self.wp * self.wp / (w * w + 1j * w * self.gamma)

    def beta_eff_squared(self, omega) -> complex:
        """GNOR effective nonlocal parameter ``beta**2 + D(gamma - i*omega)`` [m**2/s**2]
        (identical to ``nonlocal_tmm.beta_eff_squared``; ``Im < 0`` for real omega -> broadening)."""
        w = complex(omega)
        return self.beta * self.beta + self.D * (self.gamma - 1j * w)

    def kL_squared(self, omega) -> complex:
        """Longitudinal (bulk-plasmon) wavenumber squared ``(omega**2 + i*gamma*omega -
        wp**2/eps_inf)/beta_eff**2`` -- identical to ``nonlocal_tmm.kL_squared``."""
        w = complex(omega)
        num = w * w + 1j * self.gamma * w - self.wp * self.wp / self.eps_inf
        return num / self.beta_eff_squared(w)


def drude_eps(omega, params: HydroParams) -> complex:
    """Local Drude permittivity of ``params`` (the ``beta -> 0`` limit); alias of
    :meth:`HydroParams.eps_transverse` (passive: ``Im(eps) > 0`` under ``exp(-i*omega*t)``)."""
    return params.eps_transverse(omega)


def bulk_plasmon_omega(m: int, params: HydroParams, d_nm: float) -> float:
    """Undamped bulk-plasmon standing-wave frequency of a film of thickness ``d_nm`` from the
    quantization ``k_L d = m*pi`` (``gamma -> 0``, ``D -> 0``):

        omega_m = sqrt( wp**2/eps_inf + beta**2 (m*pi/d)**2 ) .

    Inverting ``nonlocal_tmm.kL_squared = 0`` gives ``beta**2 k_L**2 = omega**2 - wp**2/eps_inf``;
    at ``k_L = m*pi/d`` this is the closed form above.  The ``1/d`` term is the nonlocal ENZ /
    bulk-plasmon BLUESHIFT (a thinner film -> higher omega).  Matches ``nonlocal_tmm``'s gate-2
    ``_bulk_plasmon_omega`` (odd ``m`` couple for a symmetric film -- the ABC selection rule)."""
    d = d_nm * _L0
    return math.sqrt(params.wp ** 2 / params.eps_inf + params.beta ** 2 * (m * math.pi / d) ** 2)


def cylinder_sp_omega(params: HydroParams, eps_b: float = 1.0) -> float:
    """LOCAL (quasistatic Frohlich) dipole surface-plasmon frequency of a 2-D metal cylinder in a
    host ``eps_b``.  The 2-D dipole has depolarisation factor 1/2, so the polarisability
    ``alpha ~ (eps - eps_b)/(eps + eps_b)`` resonates at ``eps(omega) = -eps_b`` -- for a Drude
    metal ``omega_sp = wp / sqrt(eps_inf + eps_b)``."""
    return params.wp / math.sqrt(params.eps_inf + float(eps_b))


def cylinder_blueshift_raza(params: HydroParams, R_nm: float, eps_b: float = 1.0) -> float:
    """Nonlocal (HDM) dipole-SP RELATIVE blueshift ``delta_omega/omega_sp`` of a 2-D metal cylinder
    of radius ``R_nm`` in a host ``eps_b``, from the QUASISTATIC hydrodynamic boundary-value
    problem in the thin-screening (large ``k_L R``) limit (Raza et al. 2015):

        delta_omega/omega_sp = (beta / (2 wp R)) sqrt( eps_b (eps_inf + eps_b) / eps_inf ) .

    DERIVATION (the closed-form oracle for the cylinder gate; full algebra in the module header).
    Quasistatic dipole mode m=1: outside ``phi = (-E0 r + p/r) cos(theta)``; inside a harmonic part
    ``a r cos(theta)`` plus an induced free charge ``rho = rho0 I_1(kappa r) cos(theta)`` obeying
    ``(lap + k_L^2) rho = 0`` (so ``kappa = sqrt(-k_L^2)`` below ``wp/sqrt(eps_inf)``), with the
    particular potential ``-rho/(eps0 eps_inf kappa^2)``.  Three boundary conditions -- ``phi``
    continuous, background normal-D continuous ``eps_inf E_n,in = eps_b E_n,out`` (no free surface
    charge because the ABC makes ``P_free,n(R) = 0``), and the ABC ``J_n = 0`` i.e.
    ``eps0 wp^2 E_n = beta^2 d rho/dn`` -- close the system.  Eliminating the amplitudes gives the
    dipole resonance condition

        eps_T(omega) + eps_b [ 1 - eta I_1(kappa R)/(R kappa I_1'(kappa R)) ] = 0 ,
        eta = wp^2/(Omega eps_inf),   Omega = omega(omega + i gamma) ,

    which reduces to the LOCAL Frohlich ``eps_T = -eps_b`` as ``beta -> 0`` (``I_1/(R kappa I_1')
    -> 1/(kappa R) -> 0``); the leading ``1/(kappa R)`` surface term is the blueshift above.  A
    POSITIVE (blue) shift scaling as ``1/R`` -- the nonlocal inverse-size signature.  (Exact vs this
    asymptotic: ~2-4% over R = 2.5-5 nm, kappa R = 18-30.)"""
    R_m = float(R_nm) * _L0
    return (params.beta / (2.0 * params.wp * R_m)) * \
        math.sqrt(float(eps_b) * (params.eps_inf + float(eps_b)) / params.eps_inf)


# ================================================================================================
# QCM -- quantum-corrected gap material (Esteban et al. 2012)
# ================================================================================================
@dataclass(frozen=True)
class QCMGapMaterial:
    """Quantum-corrected effective-conductivity GAP material (Esteban et al., Nat. Commun. 3,
    825 (2012)).  Models the sub-nanometre tunnelling that SHORTS a plasmonic gap.

    The gap between two metals is replaced by an effective medium of a DRUDE-like permittivity
    whose free-electron response is scaled by a tunnelling FILLING factor ``T(gap)`` that switches
    the medium from vacuum (large gap, no tunnelling) to metallic (sub-nm gap, tunnelling shorts
    the gap):

        eps_QCM(omega, gap) = eps_bg + T(gap) * [ eps_metal(omega) - eps_bg ] ,
        T(gap)              = exp( -(gap - gap0)/l_t )  clamped to [0, 1] ,

    with ``eps_metal(omega) = eps_inf - wp**2/(omega**2 + i*omega*gamma_g)`` the metal Drude
    response (the SAME free-electron parameters as the surrounding metal, with an optionally
    enhanced tunnelling damping ``gamma_g``), ``l_t`` the tunnelling decay length and ``gap0`` the
    contact offset.  For ``gap >> l_t`` -> ``T -> 0`` -> ``eps_QCM -> eps_bg`` (a passive vacuum /
    host gap); for ``gap -> 0`` -> ``T -> 1`` -> ``eps_QCM -> eps_metal`` (a conductive short).

    PARAMETERIZATION + VALIDITY WINDOW.  ``l_t`` is the single physical knob: it is the exponential
    tunnelling-current decay length, set in Esteban 2012 by the metal work function (``l_t ~ 0.03
    -0.05 nm`` for gold; the effective onset of a strong tunnelling short is a few ``l_t`` ~ 0.3-1
    nm).  The default here (``l_t = 0.4 nm``, ``gap0 = 0``) places the NON-MONOTONIC enhancement
    peak near ~1 nm, the experimentally observed crossover.  The model is a CLASSICAL surrogate for
    a quantum effect and is quantitative only in the tunnelling regime ``gap in ~[0.1, 1.5] nm``;
    above ~1.5 nm it is (correctly) inert, and below ~0.1 nm (true contact) the local-conductance
    surrogate breaks down and a full quantum treatment is required.  Usable in ANY local FEM /
    transfer-matrix solve (just an ``eps(omega)`` for the gap region), as well as here.
    """

    eps_inf: float = 1.0
    wp: float = 1.37e16                 # gold-like plasma frequency [rad/s]
    gamma_g: float = 1.0e14             # tunnelling-region damping [rad/s]
    l_t_nm: float = 0.4                 # tunnelling decay length [nm]
    gap0_nm: float = 0.0                # contact offset [nm]
    eps_bg: complex = 1.0               # host/background filling the gap when no tunnelling

    def filling(self, gap_nm: float) -> float:
        """Tunnelling filling factor ``T(gap) = exp(-(gap - gap0)/l_t)`` clamped to ``[0, 1]``."""
        t = math.exp(-(float(gap_nm) - self.gap0_nm) / self.l_t_nm)
        return min(1.0, max(0.0, t))

    def eps_metal(self, omega) -> complex:
        """The fully-shorted (contact) metal Drude permittivity ``eps_inf - wp**2/(omega**2 +
        i*omega*gamma_g)`` (passive: ``Im > 0``)."""
        w = complex(omega)
        return self.eps_inf - self.wp * self.wp / (w * w + 1j * w * self.gamma_g)

    def eps(self, omega, gap_nm: float) -> complex:
        """Effective gap permittivity ``eps_bg + T(gap) (eps_metal - eps_bg)`` (Esteban 2012).
        Interpolates the gap medium from ``eps_bg`` (large gap) to metallic (contact)."""
        T = self.filling(gap_nm)
        return complex(self.eps_bg) + T * (self.eps_metal(omega) - complex(self.eps_bg))


# ================================================================================================
# 1-D-in-z coupled (E, J) hydrodynamic FEM  (the robust, oracle-backed core)
# ================================================================================================
class LayeredResult(NamedTuple):
    """Result of :func:`hydro_layered_1d`.

    Attributes
    ----------
    R, T, A : float
        Power reflectance / transmittance (0-order) and absorptance ``A = 1 - R - T``.  R/T are
        extracted by matching the outgoing p-pol plane wave in each vacuum buffer.
    A_volume : float
        The INDEPENDENT volumetric absorptance ``P_abs / P_inc``, ``P_abs = (1/2) int Re(E.J*) dV``.
        At normal incidence ``A_volume == A == nonlocal_tmm`` to ~1e-8 relative (mesh/order-
        limited); at oblique it is the PREFERRED absorptance (the R/T plane-wave fit carries the
        vector-BC error and can go unphysical), but it still carries the scalar-impedance-BC
        error itself: measured ~0.5-3% vs nonlocal_tmm at 30 deg, up to ~20% at 60 deg near
        0.7 wp.  Peak POSITIONS remain < 1%-accurate; treat oblique ABSOLUTE values as
        approximate.
    """

    R: float
    T: float
    A: float
    A_volume: float


def _build_1d_mesh(layers, hmap, iface_z):
    """Build a graded 1-D netgen mesh.  ``layers``: list of ``(name, z0, z1)`` (nm).  ``hmap``:
    name -> max element size (nm).  ``iface_z``: z values (nm) tagged as boundary 'metal_iface'
    (the ABC faces).  Boundary ids: 'zlo' (bottom), 'zhi' (top), 'metal_iface'."""
    import ngsolve as ng
    from netgen.meshing import Mesh as NgMesh, MeshPoint, Element1D, Element0D, Pnt

    zpts = []
    for (nm, z0, z1) in layers:
        h = hmap[nm]
        n = max(2, int(math.ceil((z1 - z0) / h)))
        zpts.append(np.linspace(z0, z1, n + 1))
    allz = np.unique(np.concatenate(zpts))
    m = NgMesh(dim=1)
    pids = [m.Add(MeshPoint(Pnt(float(z), 0, 0))) for z in allz]
    ridx = {}
    for (nm, z0, z1) in layers:
        if nm not in ridx:
            ridx[nm] = m.AddRegion(nm, 1)
    for i in range(len(allz) - 1):
        zc = 0.5 * (allz[i] + allz[i + 1])
        nm = next(ln for (ln, z0, z1) in layers if z0 - 1e-9 <= zc <= z1 + 1e-9)
        m.Add(Element1D([pids[i], pids[i + 1]], index=ridx[nm]))
    m.Add(Element0D(pids[0], index=1)); m.SetBCName(0, "zlo")
    m.Add(Element0D(pids[-1], index=2)); m.SetBCName(1, "zhi")
    m.SetBCName(2, "metal_iface")
    for zi in iface_z:
        k = int(np.argmin(np.abs(allz - zi)))
        m.Add(Element0D(pids[k], index=3))
    return ng.Mesh(m), allz


def hydro_layered_1d(omega, params: HydroParams, d_nm: float, *, theta_rad: float = 0.0,
                     hydro: bool = True, buffer_nm: float = 150.0, order: int = 3,
                     metal_cells: int = 80) -> LayeredResult:
    """Coupled (E, J) hydrodynamic FEM for a single metal film ``vacuum | metal(d_nm) | vacuum``,
    p-polarised plane-wave incidence at ``theta_rad``.  Solves the weak form documented in the
    module header on a 1-D-in-z mesh (fields ``~ f(z) exp(i*kx*x)``, ``kx = k0 sin(theta)``).

    Robust and oracle-backed: at NORMAL incidence R/T/A match ``nonlocal_tmm`` to ~1e-8
    relative (mesh/order-limited); at OBLIQUE incidence the bulk-plasmon absorption peaks land
    at ``k_L d = m*pi`` (:func:`bulk_plasmon_omega`) to < 1%, while the ABSOLUTE absorptance
    degrades with angle (up to ~20% at 60 deg) -- see the module header for the exact scope.

    Parameters
    ----------
    omega : float
        Angular frequency [rad/s].
    params : HydroParams
        Metal parameters (the GNOR ``D`` knob lives here).
    d_nm : float
        Film thickness [nm].
    theta_rad : float
        Incidence angle (0 = normal).  Vacuum superstrate/substrate.
    hydro : bool
        ``True`` -> the full coupled HDM (pressure term + ABC).  ``False`` -> the LOCAL Drude
        reduction (no pressure term, no ABC) on the same mesh, for the local-limit gate.
    buffer_nm, order, metal_cells : mesh / discretisation controls.

    Returns
    -------
    LayeredResult
    """
    import ngsolve as ng

    eps_inf, wp, gamma = params.eps_inf, params.wp, params.gamma
    k0 = omega / C_LIGHT
    kx = k0 * math.sin(theta_rad)
    kz = k0 * math.cos(theta_rad)
    k0p, kxp, kzp = k0 * _L0, kx * _L0, kz * _L0
    if kzp == 0.0:
        raise ValueError("grazing incidence (kz = 0) is singular")

    z = 0.0
    layers = [("sub", z, z + buffer_nm)]; z += buffer_nm
    zm0 = z; layers.append(("metal", z, z + d_nm)); z += d_nm; zm1 = z
    layers.append(("sup", z, z + buffer_nm)); z += buffer_nm
    hmap = {"sub": 15.0, "metal": max(0.04, d_nm / metal_cells), "sup": 15.0}
    mesh, allz = _build_1d_mesh(layers, hmap, iface_z=(zm0, zm1))

    eps_b = 1.0
    zc = ng.x
    ph = ng.exp(1j * kz * zc * _L0)                 # incident phase exp(i kz z), z = zc*L0
    Einc_x, Einc_z = (kz / k0) * ph, (-kx / k0) * ph  # p-pol, |E_inc| = 1

    Vx = ng.H1(mesh, order=order, complex=True)     # Ex (transparent BC, no dirichlet)
    Vz = ng.H1(mesh, order=order, complex=True)     # Ez (natural)
    jz_dir = "metal_iface" if hydro else ""         # ABC J.n = 0 only in the hydro model
    Vjx = ng.H1(mesh, order=order, complex=True, definedon=mesh.Materials("metal"))
    Vjz = ng.H1(mesh, order=order, complex=True, definedon=mesh.Materials("metal"), dirichlet=jz_dir)
    fes = ng.FESpace([Vx, Vz, Vjx, Vjz])
    (Ex, Ez, Jx, Jz), (vx, vz, wx, wz) = fes.TnT()
    dm = ng.dx(definedon=mesh.Materials("metal"))

    def curl(ux, uz):
        return ng.grad(ux)[0] - 1j * kxp * uz       # reduced 2-D TM curl (scaled to nm coords)

    cE, cv = curl(Ex, Ez), curl(vx, vz)
    a = ng.BilinearForm(fes, symmetric=False)
    a += (cE * cv - k0p ** 2 * eps_inf * (Ex * vx + Ez * vz)) * ng.dx("metal")
    a += (cE * cv - k0p ** 2 * eps_b * (Ex * vx + Ez * vz)) * ng.dx("sub|sup")
    a += (-1j * omega * MU0 * _L0 ** 2 * (Jx * vx + Jz * vz)) * dm
    # transparent (outgoing p-pol impedance) BC on Ex: cE = i(k0**2/kz)Ex -> Robin -i(k0'^2/kz')Ex
    a += (-1j * (k0p ** 2 / kzp) * Ex.Trace() * vx.Trace()) * ng.ds("zlo|zhi")
    if hydro:
        be2 = params.beta_eff_squared(omega)        # GNOR complex beta_eff**2
        divJ = 1j * kxp * Jx + ng.grad(Jz)[0]
        divw = 1j * kxp * wx + ng.grad(wz)[0]
        a += (-(be2 / _L0 ** 2) * divJ * divw
              + omega * (omega + 1j * gamma) * (Jx * wx + Jz * wz)
              - 1j * omega * EPS0 * wp ** 2 * (Ex * wx + Ez * wz)) * dm
    else:
        a += (omega * (omega + 1j * gamma) * (Jx * wx + Jz * wz)
              - 1j * omega * EPS0 * wp ** 2 * (Ex * wx + Ez * wz)) * dm
    f = ng.LinearForm(fes)
    f += (k0p ** 2 * (eps_inf - eps_b) * (Einc_x * vx + Einc_z * vz)) * dm
    f += (1j * omega * EPS0 * wp ** 2 * (Einc_x * wx + Einc_z * wz)) * dm

    gfu = ng.GridFunction(fes)
    with ng.TaskManager():
        a.Assemble(); f.Assemble()
        gfu.vec.data = a.mat.Inverse(freedofs=fes.FreeDofs(), inverse="umfpack") * f.vec
    gEx, gEz, gJx, gJz = gfu.components
    Etx, Etz = Einc_x + gEx, Einc_z + gEz

    # volumetric absorptance (independent, angle-robust)
    Pabs = _L0 * ng.Integrate(0.5 * ((Etx * ng.Conj(gJx) + Etz * ng.Conj(gJz)).real),
                              mesh, definedon=mesh.Materials("metal"))
    Pinc = kz / (2.0 * k0 * Z0)
    A_vol = float(Pabs / Pinc)

    # R / T by matching the outgoing plane wave in each vacuum buffer (0-order)
    def ev(cf, zz):
        return complex(cf(mesh(float(zz))))
    zr, zt = allz[3], allz[-4]
    r_ex = ev(gEx, zr) / np.exp(-1j * kzp * zr) / (kz / k0)     # reflected Ex amplitude / incident
    t_ex = ev(Etx, zt) / np.exp(1j * kzp * zt) / (kz / k0)      # transmitted Ex amplitude / incident
    R = float(abs(r_ex) ** 2)
    T = float(abs(t_ex) ** 2)
    A = float(1.0 - R - T)
    return LayeredResult(R=R, T=T, A=A, A_volume=A_vol)


# ================================================================================================
# 2-D scattering coupled (E, J) FEM  (cylinder / nanowire dimer)
# ================================================================================================
class HydroFEMUnstable(RuntimeError):
    """Safety-net exception for the 2-D coupled HDM solve (:func:`scattering_2d`, ``local=False``).

    The item-5.4 scalar-longitudinal-potential reformulation is STABLE in the validated regimes
    (cylinder blueshift, gap saturation down to 2 nm, local limit) -- the old indefinite-vector-J
    blow-up (field norm 1e19-1e53) is gone.  This exception is RETAINED as a guard: if a mesh cannot
    resolve the ~0.1 nm longitudinal screening length at all (so the scalar-Helmholtz sector is
    under-resolved), or ``unstable_ratio`` is set aggressively, ``scattering_2d`` raises this rather
    than returning a suspect field.  Refine the metal-surface mesh (``surf_nm``/``h_surf``), raise
    ``order_psi``, or fall back to ``hydro_layered_1d`` / the QCM material."""


class ScatterResult(NamedTuple):
    """Result of :func:`scattering_2d`.  Powers are SI, per unit out-of-plane length [W/m].

    Attributes
    ----------
    enhancement : float
        Near-field enhancement ``|E_total| / |E_inc|`` at the probe point (the gap centre for a
        dimer, a surface point for a single scatterer).
    P_abs : float
        Absorbed power ``(1/2) int Re(E.J*) dV`` in the metal (hydro), or ``(1/2) omega eps0
        Im(eps) int |E|**2 dV`` (local).  Always >= 0 for a passive medium.
    P_scat : float
        Scattered power: the outward Poynting flux of the SCATTERED field on a host contour
        (radius-independent in a lossless host -- a conserved flux).
    P_ext : float
        Extinction ``P_abs + P_scat``.
    energy_residual : float
        Energy-conservation check ``(P_tot_flux + P_abs) / max(P_abs, P_scat)``: the net outward
        flux of the TOTAL field on the same contour must equal ``-P_abs`` (all absorbed power flows
        inward), so this is ~0 for a converged, energy-conserving solve.
    """

    enhancement: float
    P_abs: float
    P_scat: float
    P_ext: float
    energy_residual: float


def cylinder_mesh(R_nm: float, *, host_nm: float = 90.0, pml_nm: float = 70.0,
                  h_host: float = 16.0, h_metal: float = 1.0, h_surf: Optional[float] = None,
                  surf_nm: float = 0.0):
    """A single metal cylinder (radius ``R_nm``) centred at the origin, in a circular host region,
    with a radial PML ring.  Regions 'metal'/'host'/'pml'; boundaries 'metal_iface' (ABC) and
    'outer' (PEC).  Optionally a refined surface shell of thickness ``surf_nm`` at mesh size
    ``h_surf`` (to resolve the longitudinal screening layer for the HDM path)."""
    import ngsolve as ng
    from netgen.occ import WorkPlane, OCCGeometry, Glue, Axes

    Rp, Rpml = R_nm + host_nm, R_nm + host_nm + pml_nm
    wp = WorkPlane(Axes((0, 0, 0), n=(0, 0, 1), h=(1, 0, 0)))
    if h_surf is not None and 0.0 < surf_nm < R_nm:
        core = wp.Circle(0, 0, R_nm - surf_nm).Face(); core.name = "metal"; core.maxh = h_metal
        shell = wp.Circle(0, 0, R_nm).Face()
        ann = shell - core; ann.name = "metal"; ann.maxh = h_surf
        metal = Glue([core, ann]); pieces = [core, ann]; outer_metal = shell
    else:
        met = wp.Circle(0, 0, R_nm).Face(); met.name = "metal"; met.maxh = h_metal
        metal = met; pieces = [met]; outer_metal = met
    for e in metal.edges:
        e.name = "metal_iface"
    host = wp.Circle(0, 0, Rp).Face(); host.name = "host"
    pml = wp.Circle(0, 0, Rpml).Face(); pml.name = "pml"; pml.edges.name = "outer"
    hr = host - outer_metal; hr.name = "host"
    pr = pml - host; pr.name = "pml"
    shape = Glue(pieces + [hr, pr])
    mesh = ng.Mesh(OCCGeometry(shape, dim=2).GenerateMesh(maxh=h_host)); mesh.Curve(3)
    mesh.SetPML(ng.pml.Radial(origin=(0, 0), rad=Rp, alpha=1j), "pml")
    return mesh, Rp


def dimer_mesh(R_nm: float, gap_nm: float, *, host_nm: float = 70.0, pml_nm: float = 60.0,
               h_host: float = 14.0, h_metal: float = 4.0, h_surf: Optional[float] = None,
               surf_nm: float = 0.0):
    """Two identical metal cylinders (radius ``R_nm``) separated by ``gap_nm`` along x, centred on
    the origin, in a circular host + radial PML.  Same region/boundary names as :func:`cylinder_mesh`.
    The gap centre is the origin ``(0, 0)`` (the near-field probe point)."""
    import ngsolve as ng
    from netgen.occ import WorkPlane, OCCGeometry, Glue, Axes

    cx = gap_nm / 2.0 + R_nm
    Rp, Rpml = 2 * R_nm + gap_nm + host_nm, 2 * R_nm + gap_nm + host_nm + pml_nm
    wp = WorkPlane(Axes((0, 0, 0), n=(0, 0, 1), h=(1, 0, 0)))
    pieces, shells = [], []
    for xc in (-cx, cx):
        if h_surf is not None and 0.0 < surf_nm < R_nm:
            core = wp.Circle(xc, 0, R_nm - surf_nm).Face(); core.name = "metal"; core.maxh = h_metal
            shell = wp.Circle(xc, 0, R_nm).Face()
            ann = shell - core; ann.name = "metal"; ann.maxh = h_surf
            pieces += [core, ann]; shells.append(shell)
        else:
            met = wp.Circle(xc, 0, R_nm).Face(); met.name = "metal"; met.maxh = h_metal
            pieces.append(met); shells.append(met)
    metal = Glue(pieces)
    for e in metal.edges:
        e.name = "metal_iface"
    host = wp.Circle(0, 0, Rp).Face(); host.name = "host"
    pml = wp.Circle(0, 0, Rpml).Face(); pml.name = "pml"; pml.edges.name = "outer"
    hr = host
    for s in shells:
        hr = hr - s
    hr.name = "host"
    pr = pml - host; pr.name = "pml"
    shape = Glue(pieces + [hr, pr])
    mesh = ng.Mesh(OCCGeometry(shape, dim=2).GenerateMesh(maxh=h_host)); mesh.Curve(3)
    mesh.SetPML(ng.pml.Radial(origin=(0, 0), rad=Rp, alpha=1j), "pml")
    return mesh, Rp


def dimer_gap_mesh(R_nm: float, gap_nm: float, *, host_nm: float = 60.0, pml_nm: float = 50.0,
                   h_host: float = 12.0, h_metal: float = 4.0, gap_h: Optional[float] = None):
    """A nanowire dimer (radius ``R_nm``, separation ``gap_nm``) whose GAP is a separate meshed
    region named 'gap' (a rectangular sliver between the cylinders, clipped to them).  Assign the
    'gap' region an arbitrary permittivity (e.g. the QCM :meth:`QCMGapMaterial.eps`) in
    :func:`gap_enhancement_2d`.  Regions 'metal'/'gap'/'host'/'pml'; boundary 'outer' (PEC).  The
    gap centre is the origin.  Used for the QCM gate: a LOCAL solve, robust at sub-nm gaps."""
    import ngsolve as ng
    from netgen.occ import WorkPlane, OCCGeometry, Glue, Axes

    cx = gap_nm / 2.0 + R_nm
    Rp, Rpml = 2 * R_nm + gap_nm + host_nm, 2 * R_nm + gap_nm + host_nm + pml_nm
    gh = gap_h if gap_h is not None else max(0.08, gap_nm / 4.0)
    wp = WorkPlane(Axes((0, 0, 0), n=(0, 0, 1), h=(1, 0, 0)))
    c1 = wp.Circle(-cx, 0, R_nm).Face(); c1.name = "metal"; c1.maxh = h_metal
    c2 = wp.Circle(cx, 0, R_nm).Face(); c2.name = "metal"; c2.maxh = h_metal
    gapbox = wp.MoveTo(-gap_nm / 2.0, -0.7 * R_nm).Rectangle(gap_nm, 1.4 * R_nm).Face()
    gapbox.name = "gap"; gapbox.maxh = gh
    gapbox = gapbox - c1 - c2
    host = wp.Circle(0, 0, Rp).Face(); host.name = "host"
    pml = wp.Circle(0, 0, Rpml).Face(); pml.name = "pml"; pml.edges.name = "outer"
    hr = host - c1 - c2 - gapbox; hr.name = "host"
    pr = pml - host; pr.name = "pml"
    shape = Glue([c1, c2, gapbox, hr, pr])
    mesh = ng.Mesh(OCCGeometry(shape, dim=2).GenerateMesh(maxh=h_host)); mesh.Curve(3)
    mesh.SetPML(ng.pml.Radial(origin=(0, 0), rad=Rp, alpha=1j), "pml")
    return mesh, Rp


def gap_enhancement_2d(mesh, omega, eps_metal: complex, eps_gap: complex, *,
                       pol_axis: str = "x", order: int = 2) -> float:
    """Gap-centre near-field enhancement ``|E_total|/|E_inc|`` for a LOCAL 2-D dimer solve where the
    'metal' region has permittivity ``eps_metal`` and the 'gap' region ``eps_gap`` (the rest is
    vacuum).  A rock-solid local curl-curl solve on a :func:`dimer_gap_mesh` -- the vehicle for the
    QCM gate: sweep the gap, set ``eps_gap = QCMGapMaterial.eps(omega, gap)``, and the enhancement
    is NON-MONOTONIC (peaks near ~1 nm, then the tunnelling short DROPS it -- Esteban 2012)."""
    import ngsolve as ng

    k0 = (omega / C_LIGHT) * _L0
    prop = ng.y if pol_axis == "x" else ng.x
    comp = 0 if pol_axis == "x" else 1
    Einc = ng.CoefficientFunction((ng.exp(1j * k0 * prop), 0) if pol_axis == "x"
                                  else (0, ng.exp(1j * k0 * prop)))
    eps_cf = ng.CoefficientFunction([eps_metal if mm == "metal" else
                                     (eps_gap if mm == "gap" else 1.0) for mm in mesh.GetMaterials()])
    fes = ng.HCurl(mesh, order=order, complex=True, dirichlet="outer")
    E, v = fes.TnT()
    a = ng.BilinearForm(fes, symmetric=True)
    a += (ng.curl(E) * ng.curl(v) - k0 ** 2 * eps_cf * (E * v)) * ng.dx
    f = ng.LinearForm(fes)
    f += (k0 ** 2 * (eps_cf - 1.0) * (Einc * v)) * ng.dx(definedon=mesh.Materials("metal|gap"))
    gfu = ng.GridFunction(fes)
    with ng.TaskManager():
        a.Assemble(); f.Assemble()
        gfu.vec.data = a.mat.Inverse(freedofs=fes.FreeDofs(), inverse="umfpack") * f.vec
    Etot = Einc + gfu
    return float(abs(complex(Etot[comp](mesh(0.0, 0.0)))))


def scattering_2d(mesh, omega, params: HydroParams, *, local: bool = False,
                  eps_host: complex = 1.0, pol_axis: str = "x",
                  probe=(0.0, 0.0), flux_radius: Optional[float] = None,
                  order: int = 2, order_psi: Optional[int] = None,
                  unstable_ratio: float = 1e3) -> ScatterResult:
    """2-D TM scattering off the metal region of ``mesh`` under a p-polarised plane wave.

    ``local=False`` solves the STABILISED coupled HDM weak form -- the scalar-longitudinal-potential
    reformulation of roadmap item 5.4 (the ``(E in H(curl), psi in H1)`` system with ``psi = div J``;
    see the module header): the transverse free response is folded analytically into the LOCAL
    ``eps_T(omega)``, and the nonlocal longitudinal physics lives in a scalar Helmholtz for ``psi``
    coupled to ``E`` through ``grad(psi)``, with the ABC ``J.n = 0`` as a NATURAL boundary condition.
    This replaces the original indefinite vector-``J`` block and is STABLE at plasmon resonances and
    in sub-5-nm gaps.  ``local=True`` solves the ordinary local-Drude curl-curl (``eps = drude_eps``)
    on the same mesh.  Incident wave: unit-amplitude p-pol, E along ``pol_axis`` ('x' or 'y'),
    propagating along the other axis.  PML on the 'pml' region, PEC on 'outer'.  Scattered-field
    formulation (host permittivity ``eps_host``).

    ``order`` is the ``H(curl)`` order for ``E``; ``order_psi`` (default ``order + 1``) is the
    ``H1`` order for the longitudinal potential ``psi`` (one order higher resolves the thin
    ~0.1 nm surface screening layer without needing an extreme surface mesh).

    The instability guard (:class:`HydroFEMUnstable`) is KEPT as a safety net: if a mesh cannot
    resolve the longitudinal screening length at all, or ``unstable_ratio`` is set aggressively,
    the solve can still be flagged rather than returned silently.  In the validated regimes
    (cylinder blueshift, gap saturation, local limit) the reformulation stays bounded.
    """
    import ngsolve as ng

    k0 = (omega / C_LIGHT) * _L0                    # nm^-1
    prop = ng.y if pol_axis == "x" else ng.x
    comp = 0 if pol_axis == "x" else 1
    Einc = ng.CoefficientFunction((ng.exp(1j * k0 * prop), 0) if pol_axis == "x"
                                  else (0, ng.exp(1j * k0 * prop)))
    dm = ng.dx(definedon=mesh.Materials("metal"))

    if local:
        eps_m = drude_eps(omega, params)
        eps_cf = ng.CoefficientFunction([eps_m if mm == "metal" else eps_host
                                         for mm in mesh.GetMaterials()])
        fes = ng.HCurl(mesh, order=order, complex=True, dirichlet="outer")
        E, v = fes.TnT()
        a = ng.BilinearForm(fes, symmetric=True)
        a += (ng.curl(E) * ng.curl(v) - k0 ** 2 * eps_cf * (E * v)) * ng.dx
        f = ng.LinearForm(fes)
        f += (k0 ** 2 * (eps_cf - eps_host) * (Einc * v)) * dm
        gfu = ng.GridFunction(fes)
        with ng.TaskManager():
            a.Assemble(); f.Assemble()
            gfu.vec.data = a.mat.Inverse(freedofs=fes.FreeDofs(), inverse="umfpack") * f.vec
        Escat = gfu
        Etot = Einc + gfu
        Pabs = _L0 ** 2 * float(ng.Integrate(0.5 * omega * EPS0 * complex(eps_m).imag
                                * (Etot * ng.Conj(Etot)).real, mesh, definedon=mesh.Materials("metal")))
    else:
        # ---- scalar-longitudinal-potential reformulation (roadmap 5.4 cure (a)) ------------------
        # Unknowns: E in H(curl) over the whole domain, psi := div J in H1 over the metal.  The
        # transverse (local) free response is folded into eps_T; psi carries only the nonlocal
        # longitudinal correction.  Strong form (metal):
        #   curl curl E - k0^2 eps_T E = -(i w mu0 be2/Om) grad psi
        #   be2 lap psi + Om psi        =  i w eps0 wp^2 div E
        # ABC J.n = 0  <=>  grad(psi).n = (i w eps0 wp^2/be2) E.n : a NATURAL condition (the two
        # surface terms from integrating be2 lap psi and i w eps0 wp^2 div E by parts CANCEL), so
        # psi carries NO essential (dirichlet) constraint.
        Om = omega * (omega + 1j * params.gamma)            # Omega = w(w + i gamma)
        be2 = params.beta_eff_squared(omega)                # beta_eff^2 (GNOR knob in params.D)
        wp2 = params.wp ** 2
        eps_T = drude_eps(omega, params)                    # LOCAL transverse Drude permittivity
        # Block coefficients in nm-mesh coordinates, with a symmetric numerical rescaling of psi
        # (psi = S * psi_hat, equation scaled by Rc) so the two off-diagonal blocks are EQUAL
        # (complex-symmetric) and every entry is O(1e-3..1) -- a well-conditioned direct solve.
        C_Ep = 1j * omega * MU0 * be2 / Om                  # E-eq  <- grad psi
        C_pE = 1j * omega * EPS0 * wp2                       # psi-eq <- div E
        a_raw = C_Ep * _L0
        b_raw = C_pE * _L0
        c_raw = -be2                                        # psi stiffness (grad.grad)
        d_raw = Om * _L0 ** 2                                # psi mass
        beta_s = max(params.beta, 1.0e3)                    # scale floor (safe as beta -> 0)
        S = params.wp / (Z0 * _L0 * beta_s)                 # real trial scale
        Rc = a_raw * S / b_raw                              # -> complex-symmetric (a_raw S = Rc b_raw)
        coup = a_raw * S                                    # both off-diagonal blocks
        stiff = Rc * c_raw * S
        mass = Rc * d_raw * S
        op = order + 1 if order_psi is None else int(order_psi)
        eps_cf = ng.CoefficientFunction([eps_T if mm == "metal" else eps_host
                                         for mm in mesh.GetMaterials()])
        fes = (ng.HCurl(mesh, order=order, complex=True, dirichlet="outer")
               * ng.H1(mesh, order=op, complex=True, definedon=mesh.Materials("metal")))
        (E, psi), (v, w) = fes.TnT()
        a = ng.BilinearForm(fes, symmetric=True)
        a += (ng.curl(E) * ng.curl(v) - k0 ** 2 * eps_cf * (E * v)) * ng.dx
        a += (coup * (ng.grad(psi) * v)) * dm
        a += (stiff * (ng.grad(psi) * ng.grad(w)) + mass * (psi * w)
              + coup * (E * ng.grad(w))) * dm
        f = ng.LinearForm(fes)
        f += (k0 ** 2 * (eps_T - eps_host) * (Einc * v)) * dm
        f += (-coup * (Einc * ng.grad(w))) * dm
        gfu = ng.GridFunction(fes)
        with ng.TaskManager():
            a.Assemble(); f.Assemble()
            gfu.vec.data = a.mat.Inverse(freedofs=fes.FreeDofs(), inverse="umfpack") * f.vec
        gE, gpsi = gfu.components
        Escat, Etot = gE, Einc + gE
        # J = (i w eps0 wp^2/Om) E_tot - (be2/Om) grad_x psi ;  psi = S psi_hat, grad_x = grad_xi/_L0
        Jcf = (1j * omega * EPS0 * wp2 / Om) * Etot \
            - (be2 / Om) * (S / _L0) * ng.grad(gpsi)
        Pabs = _L0 ** 2 * float(ng.Integrate(0.5 * (Etot * ng.Conj(Jcf)).real, mesh,
                                definedon=mesh.Materials("metal")))

    # instability guard: scattered-field L2 norm vs incident (host region)
    sc_norm = float(ng.Integrate((Escat * ng.Conj(Escat)).real, mesh,
                                 definedon=mesh.Materials("host")).real) ** 0.5
    inc_norm = float(ng.Integrate((Einc * ng.Conj(Einc)).real, mesh,
                                  definedon=mesh.Materials("host")).real) ** 0.5
    if inc_norm > 0 and sc_norm / inc_norm > unstable_ratio and not local:
        raise HydroFEMUnstable(
            "2-D coupled HDM scattered-field norm exceeded the guard (||E_scat||/||E_inc|| = "
            "{:.2e} > {:.0e}); the longitudinal screening length is likely unresolved on this "
            "mesh. Refine the metal-surface mesh (surf_nm/h_surf), raise order_psi, move off "
            "resonance, or use hydro_layered_1d / the QCM material.".format(
                sc_norm / inc_norm, unstable_ratio))

    enh = abs(complex(Etot[comp](mesh(float(probe[0]), float(probe[1])))))

    # curls: scattered from the HCurl field; incident analytically (curl of a plane wave).
    curl_scat = ng.curl(Escat)
    if pol_axis == "x":                                     # E_inc = (exp(i k0 y), 0)
        curl_inc = -1j * k0 * ng.exp(1j * k0 * prop)        # curl' = -i k0 exp(i k0 y)
    else:                                                   # E_inc = (0, exp(i k0 x))
        curl_inc = 1j * k0 * ng.exp(1j * k0 * prop)
    P_scat = _flux_on_contour(mesh, Escat, curl_scat, omega, flux_radius)
    # energy conservation: TOTAL-field outward flux must equal -P_abs (absorbed power flows in)
    if flux_radius is not None:
        P_tot_flux = _flux_on_contour(mesh, Etot, curl_scat + curl_inc, omega, flux_radius)
        scale = max(abs(Pabs), abs(P_scat), 1e-300)
        e_res = (P_tot_flux + Pabs) / scale
    else:
        e_res = float("nan")
    return ScatterResult(enhancement=float(enh), P_abs=Pabs, P_scat=P_scat,
                         P_ext=Pabs + P_scat, energy_residual=float(e_res))


def sp_resonance_omega(mesh, params: HydroParams, omegas, *, local: bool,
                       order: int = 2, order_psi: Optional[int] = None,
                       flux_radius: Optional[float] = None) -> float:
    """Locate a dipole surface-plasmon resonance as the parabola-refined peak of ``P_abs(omega)``
    over the scan ``omegas`` (a 3-point vertex refine).  ``local`` picks the local-Drude or the
    coupled HDM (:func:`scattering_2d`) solve.  The CYLINDER-BLUESHIFT gate takes the coupled peak
    MINUS the local peak on the SAME mesh, so the mesh's absolute-position error cancels and the
    residue is the physical nonlocal blueshift (compare :func:`cylinder_blueshift_raza`)."""
    ws = np.asarray(omegas, dtype=float)
    P = np.array([scattering_2d(mesh, float(w), params, local=local, order=order,
                                order_psi=order_psi, flux_radius=flux_radius).P_abs for w in ws])
    i = int(np.argmax(P))
    if 0 < i < ws.size - 1:
        y0, y1, y2 = P[i - 1], P[i], P[i + 1]
        denom = y0 - 2.0 * y1 + y2
        off = 0.5 * (y0 - y2) / denom if denom != 0.0 else 0.0
        return float(ws[i] + off * (ws[1] - ws[0]))
    return float(ws[i])


def _flux_on_contour(mesh, E_cf, curl_cf, omega, flux_radius) -> float:
    """Time-averaged outward Poynting flux (per unit out-of-plane length) of a 2-D TM field on a
    circle of radius ``flux_radius`` (nm):

        P = (1/2) contour-int Re(E x H*).n ds ,   H_z = curl(E)/(i*omega*mu0)

    ``E_cf`` is the (2-vector) field and ``curl_cf`` its scalar out-of-plane curl (both evaluable
    pointwise).  Uniform point sampling on the contour (robust, geometry-agnostic).  Returns NaN
    if the radius is not supplied or every sample misses the mesh."""
    if flux_radius is None:
        return float("nan")
    N = 240
    ang = np.linspace(0.0, 2.0 * math.pi, N, endpoint=False)
    ssum = 0.0
    cnt = 0
    for th in ang:
        x, y = flux_radius * math.cos(th), flux_radius * math.sin(th)
        try:
            E = E_cf(mesh(x, y))
            cval = curl_cf(mesh(x, y))
            cl = complex(cval[0] if isinstance(cval, (tuple, list)) else cval)
        except Exception:
            continue
        Ex, Ey = complex(E[0]), complex(E[1])
        Hz = cl / _L0 / (1j * omega * MU0)                   # physical H_z (1/L0 restores 1/m)
        # S = (1/2) Re(E x H*), H = Hz zhat -> S_r = (1/2)Re(Ey Hz* cos - Ex Hz* sin)
        Sr = 0.5 * (Ey * np.conj(Hz) * math.cos(th) - Ex * np.conj(Hz) * math.sin(th)).real
        ssum += Sr
        cnt += 1
    if cnt == 0:
        return float("nan")
    return float((ssum / cnt) * 2.0 * math.pi * flux_radius * _L0)
