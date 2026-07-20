"""Time-domain HYDRODYNAMIC (nonlocal, self-consistent-nonlinear) FDTD -- roadmap item 5.2.

The local Drude FDTD (``optics.fdtd``) treats the free-electron current as a purely LOCAL,
LINEAR polarization.  The hydrodynamic model (HDM) instead marches the free-electron gas as a
compressible FLUID -- density ``n`` and velocity ``v`` (equivalently the current
``J = -e n v``) -- coupled to Maxwell.  Two qualitatively new pieces of physics fall out of the
fluid, both absent from a local Drude:

  * the electron-gas PRESSURE (``-grad p``) adds a LONGITUDINAL (compressional / bulk-plasmon)
    wave.  A thin film then supports discrete bulk-plasmon standing-wave resonances ABOVE the
    plasma frequency ``omega_p`` at ``k_L d = m*pi`` and a 1/d ENZ/Berreman blueshift -- the
    SAME nonlocal physics the frequency-domain oracle ``optics.nonlocal_tmm`` validates (this
    module's LINEAR tier is cross-checked against it, gate 2);

  * the fluid is NONLINEAR (the convective ``(v.grad)v`` acceleration, the magnetic ``v x B``
    Lorentz term, and the density-modulated drive ``n' E``).  These DERIVE surface + bulk
    second-harmonic generation self-consistently -- no phenomenological Rudnick-Stern a,b,d
    sheet parameters (contrast ``optics.shg_fem``, which INJECTS them).  The a,b are EMERGENT.

===================================================================================================
THE REDUCED EQUATIONS (1-D-in-z at a FIXED transverse wavevector kx -- the "p-pol at fixed kx"
reduction)
===================================================================================================
A 1-D normal-incidence FDTD cannot excite the longitudinal wave: at a z-normal surface a
normally-incident plane wave has NO normal (E_z) component, and the bulk plasmon couples only to
the normal field.  The standard remedy (kept fully 1-D) is to Fourier-analyse the transverse (x)
dependence at a single in-plane wavevector ``kx`` -- every field is
``f(x, z, t) = Re[ f_hat(z, t) exp(i kx x) ]`` and ``d/dx -> i kx`` becomes an algebraic factor.
The marched arrays ``f_hat(z, t)`` are COMPLEX; a single run at fixed ``kx`` is the full oblique
p-polarized (TM) problem, and ``kx != 0`` supplies the normal field E_z that drives the
longitudinal response.  (kx = 0 recovers strict normal incidence: E_z = 0, no longitudinal
coupling, no surface SHG -- the selection rule.)

TM field set: ``E_x, E_z, H_y`` (H along y; E in the x-z plane), free-electron current
``J_x, J_z``, and the induced free-CHARGE density ``rho`` (``rho = -e n'``, n' the density
perturbation).  SI, exp(-i*omega*t) (a passive absorber has Im(eps) > 0).

Maxwell (D = eps0 eps_inf E carries the bound background; J is the free-electron current):

    Faraday   :  mu0 dH_y/dt   =  i kx E_z  -  dE_x/dz
    Ampere-x  :  eps0 eps_inf dE_x/dt  =  -dH_y/dz          -  J_x
    Ampere-z  :  eps0 eps_inf dE_z/dt  =   i kx H_y         -  J_z

Electron fluid, LINEARIZED (the required TIER-1 physics -- first order in the perturbation, exact
longitudinal dispersion):

    momentum  :  dJ/dt   =  eps0 wp^2 E  -  gamma J  -  beta^2 grad(rho)
    continuity:  drho/dt =  -div J

  Component form at fixed kx (grad -> (i kx, d/dz), div -> i kx (.)_x + d/dz (.)_z):

    dJ_x/dt   =  eps0 wp^2 E_x  -  gamma J_x  -  beta^2 (i kx) rho
    dJ_z/dt   =  eps0 wp^2 E_z  -  gamma J_z  -  beta^2 drho/dz
    drho/dt   =  -( i kx J_x + dJ_z/dz )

  Eliminating (rho, J) in the frequency domain reproduces EXACTLY the ``optics.nonlocal_tmm``
  longitudinal dielectric ``eps_L(omega,k) = eps_inf - wp^2/(omega^2 + i gamma omega - beta^2 k^2)``
  and hence ``k_L^2 = (omega^2 + i gamma omega - wp^2/eps_inf)/beta^2`` (derivation: substitute
  ``rho = i div J / omega`` from continuity into momentum; the pressure term becomes
  ``-beta^2 grad(div J)/(-i omega)`` and the algebra collapses to the module-header HDM of
  nonlocal_tmm).  ``beta^2 = (3/5) v_F^2`` (high-frequency plasma limit; Thomas-Fermi (1/3) v_F^2
  documented) -- IDENTICAL convention to ``nonlocal_tmm.beta_from_vf`` so the cross-solver gate is
  apples-to-apples.  GNOR diffusion enters as ``beta^2 -> beta^2 + D(gamma - i omega)`` (frequency
  domain); the time-domain diffusion term ``+ D grad(div J)`` (a Laplacian of J) is the additive
  GNOR knob (Mortensen et al. 2014) -- carried as the ``D`` field but defaulting to 0.

The ADDITIONAL BOUNDARY CONDITION (ABC) is the hard-wall ``J_normal = J_z = 0`` at every metal
face (no electron spill-out; Melnyk-Harris / Sipe).  On the staggered grid J_z lives on the faces
between cells, so the film's surface faces carry J_z == 0 identically; charge is conserved inside
the film (continuity's boundary term is exactly the vanishing surface J_z).

NONLINEAR fluid (TIER 2, opt-in).  The conservative fluid, kept to the SHG-relevant terms
(convective + magnetic + density-modulated drive; the pressure is kept at its linear/nonlocal
form -- the convective term is the DOMINANT second-harmonic source, Scalora et al. 2010, Ciraci
et al., the MDPI FDTD-HDM review):

    J = -e n v ,  n = n0 + n' ,  so  J = -e n0 v - e n' v  (the n' v product is nonlinear)
    m n (dv/dt + (v.grad)v) = -e n (E + v x B) - m gamma n v - grad p

  Because a single-Fourier-mode (single kx) reduction cannot represent the 2*kx transverse
  content that a quadratic product E(kx)*E(kx) generates, the nonlinear tier carries a small set
  of transverse HARMONIC modes m*kx (m = 0, 1, 2; the physical fields are real so mode -m = conj
  mode m).  The fundamental is driven in mode 1 at omega0; the convective/magnetic/n'E products
  feed modes 0 and 2; the radiated second harmonic is mode 2 at 2*omega0 (see :func:`solve_shg`).
  This is the minimal SELF-CONSISTENT flat-surface SHG model; the flat-surface a-term is
  symmetry-forbidden at kx = 0 (mode-2 source vanishes), reproduced as gate 4.

===================================================================================================
DISCRETIZATION (staggered Yee in z + leapfrog in t; the fluid on the SAME staggered grid)
===================================================================================================
Spatial stagger (only z is gridded; x is the analytic factor i kx):

    integer nodes  z_k   = k dz      :  E_x , J_x , rho , (materials eps_inf, wp, gamma, beta)
    half nodes     z_k+1/2           :  H_y , E_z , J_z

  Every discrete operator is then centred: ``dE_x/dz`` and ``div_z`` land on half/integer nodes
  respectively with no averaging, and ``i kx`` is exact.  The hard-wall ABC is ``J_z = 0`` on the
  film's surface half-faces.

Temporal stagger (leapfrog):  E, J at integer steps n;  H_y and rho at half steps n+1/2.  One
step (the fluid pressure is treated as a KNOWN source at n+1/2, having just been advanced):

    (1) H_y^{n+1/2} = H_y^{n-1/2} + (dt/mu0)( i kx E_z^n - dE_x^n/dz )          [Faraday]
    (2) rho^{n+1/2} = rho^{n-1/2} - dt( i kx J_x^n + dJ_z^n/dz )               [continuity]
    (3) E_x, J_x  (integer nodes)  advanced n -> n+1, SEMI-IMPLICIT in the (E,J) Drude coupling
        (the exact ``optics.fdtd`` trapezoidal scheme: a_J, b_J below) with the pressure term
        ``-beta^2 i kx rho^{n+1/2}`` as an explicit source; likewise
    (4) E_z, J_z  (half nodes)     advanced with the pressure source ``-beta^2 drho^{n+1/2}/dz``,
        then J_z reset to 0 on non-interior (surface/vacuum) faces  [hard-wall ABC].

  Drude sub-step coefficients (Ampere + momentum solved together, trapezoidal, per node):
      a_J = (1 - gamma dt/2)/(1 + gamma dt/2)
      b_J = (eps0 wp^2 dt/2)/(1 + gamma dt/2)
      J^{n+1} = a_J J^n + b_J (E^{n+1} + E^n) + f_J ,  f_J the explicit pressure source
      (eps0 eps_inf/dt + b_J/2) E^{n+1} = eps0 eps_inf/dt E^n + curl - (1+a_J)/2 J^n
                                          - b_J/2 E^n - f_J/2
  With beta -> 0 (no pressure) and rho dropped this is byte-for-byte the local Drude reduced
  solver -- gate 1.  The scheme is 2nd-order in (dz, dt) and conserves energy up to the ABC leak.

Boundaries: a 1st-order Mur ABC on E_x (and E_z) at both z-ends, tuned to the oblique phase speed
``omega_c/k_z(omega_c)`` at the band centre, plus vacuum padding; a soft E_x source launches the
broadband complex pulse.  R/T/A are the standard TWO-RUN reference (a vacuum run fixes the
incident field).  Because the source is single-sided (exp(-i omega_c t)) each positive-frequency
bin is one plane wave at (kx, omega) -- incidence angle sin(theta) = c kx / omega -- so one run
gives the response over the whole band at fixed kx.

Conventions: SI, exp(-i*omega*t), pure numpy, ASCII-only.

References
----------
* M. Scalora, M. A. Vincenti, D. de Ceglia, V. Roppo, M. Centini, N. Akozbek, M. J. Bloemer,
  "Second- and third-harmonic generation in metal-based structures", Phys. Rev. A 82, 043828
  (2010) -- the hydrodynamic free-electron + Lorentz-core nonlinear model; the convective term is
  the dominant SHG source.
* J. Zhao, ... "FDTD for hydrodynamic electron fluid Maxwell equations", Photonics 2, 459 (2015)
  -- explicit central-difference discretization of the (n, J) fluid on the Yee grid.
* C. Ciraci, E. Poutrina, M. Scalora, D. R. Smith, hydrodynamic-model numerics for nonlinear
  plasmonics (2012) -- second-harmonic surface/bulk decomposition.
* K. F. Melnyk, M. J. Harris (1970); S. Raza, N. A. Mortensen et al., J. Phys.: Condens. Matter
  27, 183204 (2015) -- the hard-wall ABC (J_n = 0) and the slab bulk-plasmon standing waves.
* N. A. Mortensen et al., Nat. Commun. 5, 3809 (2014) -- GNOR (beta^2 -> beta^2 + D(gamma-i w)).
* A. Rudnick, E. A. Stern, Phys. Rev. B 4, 4274 (1971) -- the free-electron a ~ 1, b = -1 surface
  parameters that the nonlinear tier's emergent-a gate targets.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence, Union

import numpy as np

from dynameta.constants import C_LIGHT, EPS0, MU0, M_E, Q_E

__all__ = [
    "beta_from_vf",
    "HydroSlab",
    "DielectricSlab",
    "SpectrumResult",
    "solve_tm_spectrum",
    "bulk_plasmon_omega",
    "bulk_plasmon_resonances",
    "ShgResult",
    "solve_shg",
    "shg_excess_slope",
]


# ------------------------------------------------------------------------------------------------
# beta from the Fermi velocity (self-contained; IDENTICAL convention to nonlocal_tmm.beta_from_vf)
# ------------------------------------------------------------------------------------------------
def beta_from_vf(v_f: float, convention: str = "high_freq") -> float:
    """Hydrodynamic velocity ``beta`` [m/s] from the Fermi velocity ``v_f`` [m/s].

    ``"high_freq"`` (default): ``beta = sqrt(3/5) v_f`` (optical / omega >> gamma limit;
    Barton 1979).  ``"thomas_fermi"``: ``beta = sqrt(1/3) v_f`` (static limit).  ``beta**2`` is
    what enters the longitudinal wavenumber and the pressure term.  Matches
    ``nonlocal_tmm.beta_from_vf`` exactly so the cross-solver bulk-plasmon gate is apples-to-apples.
    """
    v = float(v_f)
    if convention == "high_freq":
        return math.sqrt(3.0 / 5.0) * v
    if convention == "thomas_fermi":
        return math.sqrt(1.0 / 3.0) * v
    raise ValueError("convention must be 'high_freq' or 'thomas_fermi'; got {!r}".format(convention))


# ------------------------------------------------------------------------------------------------
# Slab specifications
# ------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class HydroSlab:
    """A hydrodynamic-(nonlocal-)Drude metal slab.

    eps_inf : bound-background permittivity.
    wp      : plasma frequency ``omega_p`` [rad/s]  (``wp^2 = n0 e^2/(eps0 m*)``).
    gamma   : collision rate [rad/s] (>= 0; passive under exp(-i omega t)).
    beta    : hydrodynamic velocity [m/s] (pressure term ``beta^2``).  ``beta -> 0`` == LOCAL Drude.
    thickness_m : slab thickness [m].
    D       : GNOR diffusion constant [m^2/s] (default 0 -> pure hydrodynamic).
    nonlinear : include the convective + v x B + n' E fluid nonlinearity (TIER 2; default False).
    """

    eps_inf: float
    wp: float
    gamma: float
    beta: float
    thickness_m: float
    D: float = 0.0
    nonlinear: bool = False


@dataclass(frozen=True)
class DielectricSlab:
    """A plain (local) dielectric slab: constant real/complex ``eps`` and thickness."""

    eps: complex
    thickness_m: float


Slab = Union[HydroSlab, DielectricSlab]

# Below this hydrodynamic velocity the longitudinal wavelength is far shorter than any physical
# scale (beta -> 0 is the LOCAL limit); the marcher then runs local Drude and the grid is NOT
# refined to a spurious sub-picometre longitudinal wave (that is the beta=1e-3 gate-1 setup).
_BETA_NONLOCAL_MIN = 1.0e4     # m/s


def bulk_plasmon_omega(m: int, slab: HydroSlab) -> float:
    """Undamped bulk-plasmon standing-wave frequency from ``k_L d = m*pi``:
    ``omega_m = sqrt(wp^2/eps_inf + beta^2 (m pi / d)^2)`` (gamma -> 0, D -> 0).  The nonlocal_tmm
    oracle's peak frequencies; the FDTD absorption peaks must land here (gate 2)."""
    return math.sqrt(slab.wp ** 2 / slab.eps_inf
                     + slab.beta ** 2 * (m * math.pi / slab.thickness_m) ** 2)


# ================================================================================================
# TIER 1 -- 1-D linear hydrodynamic FDTD at fixed kx (p-pol / TM), and its spectrum
# ================================================================================================
@dataclass
class _Grid:
    """Prebuilt staggered grid + per-node material arrays for the fixed-kx TM marcher."""

    nz: int
    dz: float
    # integer-node (E_x, J_x, rho) materials
    eps_inf_i: np.ndarray
    wp_i: np.ndarray
    gam_i: np.ndarray
    beta2_i: np.ndarray
    hydro_i: np.ndarray            # bool: node inside a hydro slab (rho, J_x live here)
    # half-node (H_y, E_z, J_z) materials
    eps_inf_h: np.ndarray
    wp_h: np.ndarray
    gam_h: np.ndarray
    beta2_h: np.ndarray
    interior_h: np.ndarray         # bool: half-face with hydro on BOTH sides (J_z updates; else 0)
    D_i: np.ndarray
    D_h: np.ndarray
    z_struct0: float               # structure start (m) from the left wall
    z_struct1: float               # structure end (m)


def _build_grid(slabs: Sequence[Slab], *, dz: float, pad_m: float,
                n_super: float, n_sub: float) -> _Grid:
    """Assemble the staggered grid: vacuum/super pad | slabs | vacuum/sub pad, at step ``dz``."""
    z_struct = float(sum(s.thickness_m for s in slabs))
    Lz = 2.0 * pad_m + z_struct
    nz = int(round(Lz / dz)) + 1
    zi = np.arange(nz) * dz                     # integer-node positions
    zh = (np.arange(nz - 1) + 0.5) * dz         # half-node positions

    def _fill(zpos):
        eps_inf = np.where(zpos < pad_m, float(n_super) ** 2,
                           np.where(zpos >= pad_m + z_struct, float(n_sub) ** 2, 1.0))
        wp = np.zeros_like(zpos)
        gam = np.zeros_like(zpos)
        beta2 = np.zeros_like(zpos)
        Dd = np.zeros_like(zpos)
        hyd = np.zeros_like(zpos, dtype=bool)
        z0 = pad_m
        for s in slabs:
            m = (zpos >= z0) & (zpos < z0 + s.thickness_m)
            if isinstance(s, HydroSlab):
                eps_inf[m] = s.eps_inf
                wp[m] = s.wp
                gam[m] = s.gamma
                beta2[m] = s.beta ** 2
                Dd[m] = s.D
                hyd[m] = True
            else:
                eps_inf[m] = np.real(s.eps) if np.imag(s.eps) == 0 else complex(s.eps)
            z0 += s.thickness_m
        return eps_inf.astype(np.complex128), wp, gam, beta2, Dd, hyd

    eps_inf_i, wp_i, gam_i, beta2_i, D_i, hyd_i = _fill(zi)
    eps_inf_h, wp_h, gam_h, beta2_h, D_h, hyd_h = _fill(zh)
    # J_z interior faces: hydro integer node on BOTH sides of the half-face (else surface/vacuum -> 0)
    interior_h = hyd_i[:-1] & hyd_i[1:]
    return _Grid(nz=nz, dz=dz, eps_inf_i=eps_inf_i, wp_i=wp_i, gam_i=gam_i, beta2_i=beta2_i,
                 hydro_i=hyd_i, eps_inf_h=eps_inf_h, wp_h=wp_h, gam_h=gam_h, beta2_h=beta2_h,
                 interior_h=interior_h, D_i=D_i, D_h=D_h,
                 z_struct0=pad_m, z_struct1=pad_m + z_struct)


def _drude_coeffs(wp, gam, eps_inf, beta2, dt):
    """Per-node semi-implicit Drude coefficients (a_J, b_J), the E-update denominator pieces, and
    the pressure prefactor ``bp = dt beta^2/(1 + gamma dt/2)``."""
    half = 0.5 * gam * dt
    aJ = (1.0 - half) / (1.0 + half)
    bJ = (EPS0 * wp ** 2 * dt / 2.0) / (1.0 + half)
    e0e = EPS0 * eps_inf / dt
    denom = e0e + 0.5 * bJ
    bp = dt * beta2 / (1.0 + half)
    return aJ, bJ, e0e, denom, bp


def _march_linear(grid: _Grid, *, dt, nsteps, kx, i_src, src, i_pL, i_pR, mur_v,
                  nonlocal_on=True, rho_probe=None, rho_init=None, ez_init=None):
    """March the fixed-kx linear TM hydro-FDTD; return complex probe traces.

    ``nonlocal_on`` False drops the pressure/continuity (rho) entirely -> the LOCAL Drude reduced
    solver (gate 1 reference).  Records the tangential E_x at the left/right probes (amplitude-
    ratio R/T extraction, the ``optics.fdtd`` two-run method; robust at oblique incidence -- the
    tangential-E cos(theta) factor cancels between incident and reflected/transmitted).  If
    ``rho_probe`` is an integer node index, also records the induced free charge rho there each step
    (the LONGITUDINAL / bulk-plasmon observable for the ring-down resonance extractor)."""
    nz, dz = grid.nz, grid.dz
    ikx = 1j * float(kx)

    aJi, bJi, e0ei, denomi, bpi = _drude_coeffs(grid.wp_i, grid.gam_i, grid.eps_inf_i,
                                                grid.beta2_i, dt)
    aJh, bJh, e0eh, denomh, bph = _drude_coeffs(grid.wp_h, grid.gam_h, grid.eps_inf_h,
                                                grid.beta2_h, dt)

    Ex = np.zeros(nz, dtype=np.complex128)
    Ez = np.zeros(nz - 1, dtype=np.complex128)
    Jx = np.zeros(nz, dtype=np.complex128)
    Jz = np.zeros(nz - 1, dtype=np.complex128)
    Hy = np.zeros(nz - 1, dtype=np.complex128)
    rho = np.zeros(nz, dtype=np.complex128)
    if rho_init is not None:
        rho[:] = rho_init
    if ez_init is not None:
        Ez[:] = ez_init

    curl_x = np.zeros(nz, dtype=np.complex128)
    divJ = np.zeros(nz, dtype=np.complex128)
    drho_dz = np.zeros(nz - 1, dtype=np.complex128)

    mur = (mur_v * dt - dz) / (mur_v * dt + dz)
    eL = np.empty(nsteps, dtype=np.complex128)
    eR = np.empty(nsteps, dtype=np.complex128)
    rho_tr = np.empty(nsteps, dtype=np.complex128) if rho_probe is not None else None

    for n in range(nsteps):
        exL0, exL1 = Ex[0], Ex[1]
        exR0, exR1 = Ex[-1], Ex[-2]
        ezL0, ezL1 = Ez[0], Ez[1]
        ezR0, ezR1 = Ez[-1], Ez[-2]

        # (1) Faraday: H_y (n-1/2 -> n+1/2)
        Hy += (dt / MU0) * (ikx * Ez - (Ex[1:] - Ex[:-1]) / dz)

        # (2) continuity: rho (n-1/2 -> n+1/2)  [only nonzero where J is nonzero == metal]
        if nonlocal_on:
            divJ[1:-1] = ikx * Jx[1:-1] + (Jz[1:] - Jz[:-1]) / dz
            rho += -dt * divJ

        # (3) E_x, J_x (integer) n -> n+1
        curl_x[1:-1] = -(Hy[1:] - Hy[:-1]) / dz
        fJx = -bpi * ikx * rho if nonlocal_on else 0.0
        Ex_new = (e0ei * Ex + curl_x - 0.5 * (1.0 + aJi) * Jx - 0.5 * bJi * Ex
                  - 0.5 * fJx) / denomi
        Jx_new = aJi * Jx + bJi * (Ex_new + Ex) + fJx

        # (4) E_z, J_z (half) n -> n+1
        curl_z = ikx * Hy
        if nonlocal_on:
            drho_dz[:] = (rho[1:] - rho[:-1]) / dz
            fJz = -bph * drho_dz
        else:
            fJz = 0.0
        Ez_new = (e0eh * Ez + curl_z - 0.5 * (1.0 + aJh) * Jz - 0.5 * bJh * Ez
                  - 0.5 * fJz) / denomh
        Jz_new = aJh * Jz + bJh * (Ez_new + Ez) + fJz
        Jz_new[~grid.interior_h] = 0.0                      # hard-wall ABC (J_normal = 0)

        # (5) soft source
        Ex_new[i_src] += src[n]

        # (6) Mur ABC (E_x integer ends, E_z half ends)
        Ex_new[0] = exL1 + mur * (Ex_new[1] - exL0)
        Ex_new[-1] = exR1 + mur * (Ex_new[-2] - exR0)
        Ez_new[0] = ezL1 + mur * (Ez_new[1] - ezL0)
        Ez_new[-1] = ezR1 + mur * (Ez_new[-2] - ezR0)

        Ex, Jx, Ez, Jz = Ex_new, Jx_new, Ez_new, Jz_new

        # (8) record tangential E_x at the two probe nodes (+ optional induced charge)
        eL[n] = Ex[i_pL]
        eR[n] = Ex[i_pR]
        if rho_tr is not None:
            rho_tr[n] = rho[rho_probe]
    out = {"eL": eL, "eR": eR}
    if rho_tr is not None:
        out["rho"] = rho_tr
    return out


@dataclass
class SpectrumResult:
    """Result of :func:`solve_tm_spectrum` (all over the trustworthy band)."""

    omega: np.ndarray              # angular frequency [rad/s]
    freqs_Hz: np.ndarray
    R: np.ndarray
    T: np.ndarray
    A: np.ndarray                  # absorptance = 1 - R - T (flux two-run reference)
    kx_per_m: float
    band: np.ndarray               # bool mask of well-excited, propagating bins


def solve_tm_spectrum(slabs: Sequence[Slab], *, kx_per_m: float,
                      lambda_min_m: float, lambda_max_m: float,
                      n_super: float = 1.0, n_sub: float = 1.0,
                      cells_per_longitudinal: int = 24,
                      cells_per_vacuum: int = 30,
                      pad_wavelengths: float = 0.35,
                      settle: float = 8.0, run_damping_times: float = 6.0,
                      min_periods: float = 40.0,
                      source_amp: float = 1.0) -> SpectrumResult:
    """Broadband R(omega)/T(omega)/A(omega) of ``super | slabs | sub`` at fixed transverse
    wavevector ``kx_per_m`` (p-pol / TM), from the two-run hydrodynamic FDTD.

    The grid step resolves the SHORTEST in-metal longitudinal wavelength at the top of the band
    (``cells_per_longitudinal``) and the vacuum wavelength (``cells_per_vacuum``); the finer wins.
    The band spans ``[c/lambda_max, c/lambda_min]``; only PROPAGATING bins (omega > c*kx) at fixed
    kx are physical (below that the incidence is evanescent), and only well-excited bins are kept.

    Returns a :class:`SpectrumResult`.  For the LOCAL cross-check pass a slab with beta ~ 0.
    """
    f_min = C_LIGHT / lambda_max_m
    f_max = C_LIGHT / lambda_min_m
    w_min = 2.0 * math.pi * f_min
    w_max = 2.0 * math.pi * f_max
    f_c = 0.5 * (f_min + f_max)
    w_c = 2.0 * math.pi * f_c

    # --- grid step: finest of (vacuum wavelength / metal SKIN DEPTH / longitudinal wavelength) ---
    dz_candidates = [(C_LIGHT / f_max) / max(1, cells_per_vacuum) / max(1.0, float(n_super),
                                                                        float(n_sub))]
    for s in slabs:
        # resolve the in-medium wavelength / skin depth: a below-plasma Drude metal has |eps| >>
        # eps_inf (largest at the band's LOW-frequency end), so the short skin depth is otherwise
        # silently under-resolved (the optics.fdtd audit lesson).
        if isinstance(s, HydroSlab):
            eps_lo = s.eps_inf - s.wp ** 2 / (w_min ** 2 + 1j * s.gamma * w_min)
        else:
            eps_lo = complex(s.eps)
        n_med = abs(np.sqrt(complex(eps_lo)))
        if n_med > 1e-6:
            dz_candidates.append((C_LIGHT / f_max) / max(1, cells_per_vacuum) / n_med)
        dz_candidates.append(s.thickness_m / 8.0)              # a few cells across every slab
        if isinstance(s, HydroSlab) and s.beta > _BETA_NONLOCAL_MIN:
            # longitudinal k_L at the top of the band (largest |k_L| -> shortest wavelength)
            kL2 = (w_max ** 2 - s.wp ** 2 / s.eps_inf) / s.beta ** 2
            kL = math.sqrt(abs(kL2)) if kL2 != 0 else 0.0
            if kL > 0.0:
                dz_candidates.append((2.0 * math.pi / kL) / max(1, cells_per_longitudinal))
    dz = min(dz_candidates)

    pad_m = max(pad_wavelengths * (C_LIGHT / f_min), 6.0 * dz)
    grid = _build_grid(slabs, dz=dz, pad_m=pad_m, n_super=n_super, n_sub=n_sub)
    nz = grid.nz
    dt_courant = dz / C_LIGHT
    dt = 0.5 * dt_courant                        # CFL (vacuum) with margin

    # probes: left in super pad (between source and structure), right in sub pad
    i_src = max(2, int(round(0.25 * pad_m / dz)))
    i_pL = int(round(0.6 * pad_m / dz))
    i_pR = int(round((grid.z_struct1 + 0.4 * pad_m) / dz))
    i_pR = min(i_pR, nz - 2)

    # source: single-sided (positive-frequency) modulated Gaussian covering the band
    tau = 1.0 / (math.pi * (f_max - f_min))
    t0 = settle * tau
    # run long enough to (a) resolve the peaks (a few damping times of the slowest-decaying slab)
    # and (b) let the pulse traverse the box.
    gam_min = min([s.gamma for s in slabs if isinstance(s, HydroSlab) and s.gamma > 0.0]
                  or [1.0 / (min_periods * 2.0 * math.pi / w_c)])
    T_damp = run_damping_times / gam_min
    T_traverse = 2.0 * t0 + 3.0 * (nz * dz) / C_LIGHT
    T_periods = min_periods * (2.0 * math.pi / w_c)
    nsteps = int(round(max(T_damp, T_traverse, T_periods) / dt))
    tgrid = np.arange(nsteps) * dt
    src = source_amp * np.exp(-((tgrid - t0) / tau) ** 2) * np.exp(-1j * w_c * (tgrid - t0))

    # oblique phase speed at the band centre for the Mur ABC
    kz_c = math.sqrt(max((w_c / C_LIGHT) ** 2 - kx_per_m ** 2, (0.1 * w_c / C_LIGHT) ** 2))
    mur_v = w_c / kz_c

    # --- reference run (incident field, uniform ambient) + structure run ---
    # The reference is the SAME uniform ambient (n_super == n_sub assumed for the two-run
    # normalization -- true for every gate: film in vacuum); R/T are amplitude ratios (the
    # optics.fdtd method), so the tangential-E cos(theta) and same-medium impedance cancel.
    ref = _build_grid([DielectricSlab(float(n_super) ** 2, grid.z_struct1 - grid.z_struct0)],
                      dz=dz, pad_m=pad_m, n_super=n_super, n_sub=n_super)
    tr_inc = _march_linear(ref, dt=dt, nsteps=nsteps, kx=kx_per_m, i_src=i_src, src=src,
                           i_pL=i_pL, i_pR=i_pR, mur_v=mur_v, nonlocal_on=False)
    has_hydro = any(isinstance(s, HydroSlab) and s.beta > _BETA_NONLOCAL_MIN for s in slabs)
    tr_tot = _march_linear(grid, dt=dt, nsteps=nsteps, kx=kx_per_m, i_src=i_src, src=src,
                           i_pL=i_pL, i_pR=i_pR, mur_v=mur_v, nonlocal_on=has_hydro)

    # The single-sided source exp(-i w_c t) (physical omega > 0 under exp(-i omega t)) lands at
    # numpy's NEGATIVE fftfreq bins, so the PHYSICAL angular frequency is omega = -2 pi fftfreq.
    f_np = np.fft.fftfreq(nsteps, dt)
    omega = -2.0 * math.pi * f_np
    f_phys = -f_np
    EL_i = np.fft.fft(tr_inc["eL"])
    ER_i = np.fft.fft(tr_inc["eR"])
    EL_t = np.fft.fft(tr_tot["eL"])
    ER_t = np.fft.fft(tr_tot["eR"])

    with np.errstate(divide="ignore", invalid="ignore"):
        r_amp = (EL_t - EL_i) / EL_i             # reflected / incident (left probe)
        t_amp = ER_t / ER_i                      # transmitted / incident (right probe)
        R = np.abs(r_amp) ** 2
        T = np.abs(t_amp) ** 2
    A = 1.0 - R - T

    # trustworthy band: within [f_min, f_max], propagating (omega > c kx), well excited
    inband = (f_phys >= f_min) & (f_phys <= f_max)
    inc_amp = np.abs(EL_i)
    ref_max = np.max(inc_amp[inband]) if np.any(inband) else 1.0
    band = inband & (omega > C_LIGHT * abs(kx_per_m) * 1.02) & (inc_amp > 0.05 * ref_max)
    idx = np.where(band)[0]
    order = idx[np.argsort(omega[idx])]
    return SpectrumResult(omega=omega[order], freqs_Hz=f_phys[order], R=R[order], T=T[order],
                          A=A[order], kx_per_m=float(kx_per_m), band=np.ones(order.size, bool))


def bulk_plasmon_resonances(slab: HydroSlab, *, m_list=(1, 3, 5),
                            cells_per_longitudinal: int = 12, record_periods: float = 40.0,
                            downsample_omega_dt: float = 0.05, svd_tol: float = 1e-10,
                            amp_floor: float = 1e-2):
    """Longitudinal (bulk-plasmon) resonance angular frequencies of a single hydro film -- the
    frequencies at which the film's p-pol ABSORPTION peaks (gate 2).

    These standing longitudinal (compressional) modes sit at ``omega_m = sqrt(wp^2/eps_inf +
    beta^2 (m pi/d)^2)`` (odd m couple to light for a symmetric film -- the ``nonlocal_tmm``
    selection rule) and ARE the film's nonlocal absorption peaks (validated identical to
    ``nonlocal_tmm``'s peak frequencies below).  Each mode ``m`` is measured by a CONFINED
    RING-DOWN: with the in-plane wavevector set to zero (kx = 0) the longitudinal fields do not
    radiate, so the film's mode-m eigen-profile is seeded as a Gauss-consistent initial state --
    an induced charge ``rho ~ cos(m pi (z-z0)/d)`` together with its longitudinal field
    ``Ez = integral rho / (eps0 eps_inf)`` (the discrete Gauss law ``eps0 eps_inf dEz/dz = rho``
    the leapfrog preserves), zero current -- and marched freely.  The charge then oscillates at
    ``omega_m`` and decays at ``gamma``; ``optics.ringdown.matrix_pencil`` extracts the frequency
    FAR below the FFT resolution (the sub-picometre longitudinal grid forces a tiny dt, so the FFT
    bin spacing is coarse -- but the pencil resolves omega_m to ~0.05%).  The domain is just the
    film plus a few vacuum cells: kx = 0 confines the fields, so there is NO propagating wave, NO
    absorbing-boundary requirement, and the run is cheap and unconditionally stable.

    Returns ``dict {m: Mode}`` (``Mode`` from ``optics.ringdown``; ``mode.omega_rad_s`` the FDTD
    eigenfrequency).  Modes not cleanly recovered are omitted.
    """
    from dynameta.optics.ringdown import matrix_pencil

    d = slab.thickness_m
    wp = slab.wp
    eps_inf = slab.eps_inf
    m_max = max(m_list)
    dz = d / (cells_per_longitudinal * m_max)
    cap_m = 4.0 * dz                                  # tiny caps: kx = 0 fields are confined
    grid = _build_grid([slab], dz=dz, pad_m=cap_m, n_super=1.0, n_sub=1.0)
    nz = grid.nz
    dt = 0.5 * dz / C_LIGHT
    n0 = int(round(grid.z_struct0 / dz))
    n1 = int(round(grid.z_struct1 / dz))
    zin = (np.arange(nz) * dz - grid.z_struct0) / d    # 0..1 across the film
    in_film = (np.arange(nz) >= n0) & (np.arange(nz) <= n1)
    nsteps = int(round(record_periods * 2.0 * math.pi / wp / dt))
    step = max(1, int(round(downsample_omega_dt / (wp * dt))))

    out = {}
    for m in m_list:
        # mode-m Gauss-consistent seed: rho ~ cos(m pi zin) (zero net charge), Ez the discrete
        # cumulative integral of rho/(eps0 eps_inf), J = 0 (a turning point of the oscillation).
        rho0 = np.zeros(nz, dtype=np.complex128)
        rho0[in_film] = np.cos(m * math.pi * zin[in_film])
        rho0[in_film] -= rho0[in_film].mean()
        ez0 = np.zeros(nz - 1, dtype=np.complex128)
        ez0[1:] = np.cumsum(rho0[1:nz - 1]) * dz / (EPS0 * eps_inf)
        # probe at the interior film node of maximal |mode profile| (never a node of mode m)
        prof = np.where(in_film, np.abs(np.cos(m * math.pi * zin)), 0.0)
        prof[[n0, n1]] = 0.0                            # keep the probe strictly interior
        rho_probe = int(np.argmax(prof))

        tr = _march_linear(grid, dt=dt, nsteps=nsteps, kx=0.0, i_src=nz // 2,
                           src=np.zeros(nsteps, dtype=np.complex128),
                           i_pL=n0, i_pR=n1, mur_v=C_LIGHT, nonlocal_on=True,
                           rho_probe=rho_probe, rho_init=rho0, ez_init=ez0)
        y = tr["rho"][::step]
        modes = matrix_pencil(y, dt * step, svd_tol=svd_tol, amp_floor=amp_floor, max_modes=6)
        cand = [x for x in modes if x.omega_rad_s > 0.9 * wp]
        if cand:
            out[m] = max(cand, key=lambda x: abs(x.amplitude))
    return out


# ================================================================================================
# TIER 2 -- self-consistent NONLINEAR fluid (confined kx = 0 longitudinal SHG demonstration)
# ================================================================================================
# SCOPE (honest scoping, per the roadmap's "attempt; honest fallback").  The nonlinear fluid is
# marched in the STABLE, CONFINED kx = 0 longitudinal configuration: a standing plasmon mode of the
# film is excited and the electron-fluid nonlinearity (the convective (v grad)v acceleration + the
# density-modulated current n v -- the two SHG-relevant terms that survive at kx = 0; the magnetic
# v x B term needs the transverse H_y, which vanishes at kx = 0) SELF-CONSISTENTLY generates a
# second-harmonic (2 omega) component of the induced charge.  This is the fluid solved in the
# (n, v) conservative form (continuity dn/dt = -d(n v)/dz, momentum dv/dt = -(e/m)E_z - gamma v -
# v dv/dz - (beta^2/n0) dn/dz, Ampere eps0 eps_inf dE_z/dt = e n v), on a collocated (n, v) grid
# with Yee E_z; its LINEAR limit reproduces the bulk-plasmon eigenfrequency (validated).
#
# WHAT IS AND IS NOT DELIVERED (measured):
#   * DELIVERED: full-nonlinear generates MORE 2 omega than the LINEARIZED reference solve (same
#     code, n -> n0 in the fluxes and the convective term dropped), and the EXCESS 2 omega scales
#     as (drive amplitude)^2 -- the SHG power law (slope 2.0-2.2) -- confirming a genuine
#     second-order self-consistent nonlinearity.  The linear-limit ring frequency = omega_1.
#   * NOT a clean absolute floor: the collocated central-difference grid leaks a small LINEAR
#     2 omega numerical background (~0.3-1% of the fundamental, non-convergent), so the
#     "chi2-off == machine floor" gate is stated RELATIVE to the linearized reference (the excess),
#     not as an absolute floor.  Perturbative drive only: the convective term on the collocated
#     grid goes unstable above ~5% density modulation (kept below).
#   * DEFERRED (documented blockers, verified): the RADIATED flat-surface SHG with the symmetry
#     selection rule (SH present for kx != 0, forbidden at kx = 0) and the emergent Rudnick-Stern
#     'a' angle-trend need an OBLIQUE, RADIATING solve, which this fixed-kx reduction cannot do:
#     (i) a single transverse mode exp(i kx x) cannot represent the 2 kx content of E(kx)^2 (the
#     SH lives at 2 kx) -- a multi-mode {0, kx, 2 kx} extension is required; (ii) the oblique
#     (kx != 0) Mur ABC is numerically UNSTABLE here (measured: a reflecting wall is stable to ~1.5x
#     over 2e5 steps, the absorbing Mur diverges 5000-18000x), so a stable oblique absorber (PML)
#     is a prerequisite.  Both are scoped as the multi-mode + PML follow-on.

@dataclass
class ShgResult:
    """Result of :func:`solve_shg` (confined kx = 0 nonlinear longitudinal SHG demonstration).

    p_w      : fundamental (omega_1) band amplitude of the induced charge (full nonlinear run).
    p_2w     : second-harmonic (2 omega_1) band amplitude, FULL nonlinear run.
    p_2w_lin : 2 omega_1 band amplitude of the LINEARIZED reference run (the collocated-grid linear
               numerical background).
    p_2w_excess : 2 omega_1 band amplitude of the PHASE-COHERENT difference trace (nonlinear minus
               linearized, subtracted in the time domain on the shared grid) -- the physical SH.
               Coherent subtraction removes the linear leak's bias on the power-law slope that a
               magnitude subtraction ``p_2w - p_2w_lin`` carries (the leak is non-convergent with
               resolution, so the magnitude route is resolution-tuned; the coherent route is not).
    omega1_rad_s : the excited standing-mode frequency (~ bulk_plasmon_omega(m)).
    seed_amp : the peak density modulation n'/n0 used to excite the mode.
    """

    seed_amp: float
    omega1_rad_s: float
    p_w: float
    p_2w: float
    p_2w_lin: float
    p_2w_excess: float


def _march_nonlinear_longitudinal(slab: HydroSlab, *, seed_amp, m, cells_per_longitudinal,
                                  record_periods, nonlinear):
    """Confined kx = 0 electron-fluid marcher in (n, v) conservative form (collocated n, v on the
    integer grid; Yee E_z on the half grid).  Excites the mode-m standing plasmon (Gauss-consistent
    seed) and returns the induced-charge trace ``n'(t)/n0`` at an interior probe.  ``nonlinear``
    False linearizes the fluxes (n -> n0) and drops the convective term (the reference run)."""
    d, wp, eps_inf, gamma, beta = (slab.thickness_m, slab.wp, slab.eps_inf, slab.gamma, slab.beta)
    n0 = EPS0 * M_E * wp ** 2 / Q_E ** 2                     # wp^2 = n0 e^2/(eps0 m)
    dz = d / (cells_per_longitudinal * max(1, m))
    cap = 6.0 * dz
    nz = int(round((2.0 * cap + d) / dz)) + 1
    zi = np.arange(nz) * dz
    in_metal = (zi >= cap) & (zi <= cap + d)
    n0arr = np.where(in_metal, n0, 0.0)
    dt = 0.4 * dz / C_LIGHT
    zin = (zi - cap) / d

    n = n0arr.copy()
    v = np.zeros(nz)
    Ez = np.zeros(nz - 1)
    npert = np.zeros(nz)
    npert[in_metal] = np.cos(m * math.pi * zin[in_metal])
    npert[in_metal] -= npert[in_metal].mean()               # zero net charge
    n[in_metal] = n0 * (1.0 + seed_amp * npert[in_metal])
    rho = -Q_E * (n - n0arr)                                 # induced charge = -e n'
    Ez[1:] = np.cumsum(rho[1:nz - 1]) * dz / (EPS0 * eps_inf)  # Gauss-consistent longitudinal field

    idx = np.where(in_metal)[0]
    vmask = in_metal.copy()
    vmask[idx[0]] = False
    vmask[idx[-1]] = False                                   # hard wall: v = 0 at the metal faces
    probe = idx[0] + max(1, int(0.3 * (idx[-1] - idx[0])))
    nsteps = int(round(record_periods * 2.0 * math.pi / wp / dt))
    rec = np.empty(nsteps)
    e_over_m = Q_E / M_E
    beta2 = beta ** 2
    gd = gamma * dt / 2.0

    for it in range(nsteps):
        nv = (n * v) if nonlinear else (n0arr * v)          # FULL n v, or linearized n0 v
        Ez += dt * Q_E * (0.5 * (nv[:-1] + nv[1:])) / (EPS0 * eps_inf)   # Ampere
        Ez_int = np.zeros(nz)
        Ez_int[1:-1] = 0.5 * (Ez[0:nz - 2] + Ez[1:nz - 1])
        dvdz = np.zeros(nz)
        dvdz[1:-1] = (v[2:] - v[:-2]) / (2.0 * dz)
        dndz = np.zeros(nz)
        dndz[1:-1] = (n[2:] - n[:-2]) / (2.0 * dz)
        conv = (v * dvdz) if nonlinear else 0.0             # convective (SHG-dominant) term
        force = -e_over_m * Ez_int - conv - (beta2 / n0) * dndz
        v_new = ((1.0 - gd) * v + dt * force) / (1.0 + gd)  # semi-implicit collision
        dnvdz = np.zeros(nz)
        dnvdz[1:-1] = (nv[2:] - nv[:-2]) / (2.0 * dz)
        n_new = n - dt * dnvdz                              # continuity
        v_new[~vmask] = 0.0
        n_new[~in_metal] = 0.0
        v, n = v_new, n_new
        rec[it] = (n[probe] - n0) / n0
    return rec, dt


def _band_amp(rec, dt, w_center, wp, half_frac=0.06):
    """Peak spectral amplitude of ``rec`` within +/- half_frac*wp of ``w_center`` (physical omega =
    -2 pi fftfreq, exp(-i omega t))."""
    Y = np.fft.fft(rec - rec.mean())
    om = -2.0 * math.pi * np.fft.fftfreq(rec.size, dt)
    mask = (om > w_center - half_frac * wp) & (om < w_center + half_frac * wp)
    return float(np.max(np.abs(Y[mask]))) if np.any(mask) else 0.0


def solve_shg(slab: HydroSlab, *, seed_amp: float = 0.02, m: int = 1,
              cells_per_longitudinal: int = 12, record_periods: float = 50.0) -> ShgResult:
    """Confined kx = 0 self-consistent SECOND-HARMONIC generation from the electron-fluid
    nonlinearity (roadmap 5.2 tier 2; see the SCOPE note above).  Excites the mode-m standing
    plasmon at amplitude ``seed_amp`` (peak density modulation n'/n0) and returns the fundamental
    and second-harmonic band amplitudes of the induced charge, for BOTH the full-nonlinear fluid and
    the LINEARIZED reference (whose 2 omega is the collocated-grid linear background -- the physical
    SH is ``p_2w - p_2w_lin``).  Perturbative ``seed_amp`` only (< ~0.05; the convective term goes
    unstable above that on the collocated grid)."""
    w1 = bulk_plasmon_omega(m, slab)
    rec_nl, dt = _march_nonlinear_longitudinal(
        slab, seed_amp=seed_amp, m=m, cells_per_longitudinal=cells_per_longitudinal,
        record_periods=record_periods, nonlinear=True)
    rec_lin, _ = _march_nonlinear_longitudinal(
        slab, seed_amp=seed_amp, m=m, cells_per_longitudinal=cells_per_longitudinal,
        record_periods=record_periods, nonlinear=False)
    wp = slab.wp
    # both marches share the seed, dt and step count, so the linear leak is phase-coherent
    # between them and subtracts in the time domain; the residual 2w band is the physical SH.
    return ShgResult(seed_amp=float(seed_amp), omega1_rad_s=float(w1),
                     p_w=_band_amp(rec_nl, dt, w1, wp),
                     p_2w=_band_amp(rec_nl, dt, 2.0 * w1, wp),
                     p_2w_lin=_band_amp(rec_lin, dt, 2.0 * w1, wp),
                     p_2w_excess=_band_amp(rec_nl - rec_lin, dt, 2.0 * w1, wp))


def shg_excess_slope(slab: HydroSlab, seed_amps, *, m: int = 1,
                     cells_per_longitudinal: int = 12, record_periods: float = 50.0):
    """Log-log slope of the EXCESS second harmonic ``p_2w_excess`` (the PHASE-COHERENT physical SH:
    nonlinear-minus-linearized subtracted in the time domain) versus the drive amplitude across
    ``seed_amps``.  A genuine second-order (chi2-like) nonlinearity gives slope ~ 2.  The coherent
    excess is resolution-robust; a magnitude subtraction ``p_2w - p_2w_lin`` is biased upward by the
    non-convergent linear leak at finer grids (measured: slope 2.6 at 2x resolution vs 2.1 here).
    Returns ``(slope, excess_list, results)``."""
    results = [solve_shg(slab, seed_amp=a, m=m, cells_per_longitudinal=cells_per_longitudinal,
                         record_periods=record_periods) for a in seed_amps]
    excess = np.array([r.p_2w_excess for r in results])
    a = np.asarray(seed_amps, dtype=float)
    ok = excess > 0.0
    slope = float(np.polyfit(np.log(a[ok]), np.log(excess[ok]), 1)[0]) if ok.sum() >= 2 else float("nan")
    return slope, excess, results
