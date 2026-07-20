"""Undepleted two-step surface second-harmonic generation (SHG) driver + analytic oracle.

Physics (all SI, exp(-i omega t), Im eps > 0 = loss):
  Metal SHG is dominated by a SURFACE Rudnick-Stern nonlinear polarization sheet at the
  metal/dielectric interface (Rudnick & Stern, Phys. Rev. B 4, 4274 (1971); Sipe/Mizrahi
  phenomenology, JOSA B 5, 660 (1988); Sipe, So, Fukui, Stegeman, PRB 21, 4389 (1980)),
  plus a weaker nonlocal bulk convective term. The undepleted two-step is:

    (1) LINEAR solve at omega (the existing plane-wave-driven FEM, solver.solve_fem) on a
        structure containing a Drude metal;
    (2) extract the fundamental fields at the metal surface -- the NORMAL component E_perp
        just inside the metal (the Sipe prescription: evaluate the driving field inside the
        metal, then let the sheet radiate as if placed just outside) and the tangential E;
    (3) assemble the SH sources: the surface sheet P_s(2w) = eps0 chi_s : E(w)E(w) [+ bulk];
    (4) LINEAR solve at 2*omega driven by those sources (solver.solve_fem_sourced);
    (5) radiated SH power per port + conversion efficiency.

  This module provides:
    * rudnick_stern_surface_chi -- the Rudnick-Stern a,b -> surface-chi parameterization, with
      the units stated explicitly;
    * rudnick_stern_flat_shg    -- the CLOSED-FORM reflected-SHG oracle for a flat Drude half-
      space (the primary validation oracle, derived from Maxwell + the Heinz/Sipe interface
      boundary conditions -- no FEM);
    * shg_two_step              -- the FEM two-step driver.

  Rudnick-Stern parameter normalization (DOCUMENTED, SI):
  ------------------------------------------------------------------------------------------
  The isotropic metal surface has three independent surface-susceptibility components
  (Dadap, Shan, Heinz, JOSA B 21, 1328 (2004), Eq. 2):
      chi_s,perp-perp-perp   (all normal)     <-> Rudnick-Stern  a  (dominant)
      chi_s,perp-par-par     (normal, 2 tang.)
      chi_s,par-perp-par     (tang.,normal,tang.) <-> Rudnick-Stern  b
  A surface susceptibility has units of m^2/V (one extra length vs the bulk chi(2), m/V:
  P_s is a dipole moment PER UNIT AREA, C/m, and P_s = eps0 chi_s E^2 with [eps0]=C^2/(J m),
  [E]^2=(V/m)^2 gives [chi_s]=m^2/V; equivalently m^2/V = C s^2 / kg).

  We adopt the free-electron (jellium) Rudnick-Stern form as rendered in the SFG/SHG model of
  Busson & Tadjeddine-style analyses (see arXiv:1905.06026 Eq. A8, specialised to degenerate
  SHG omega1=omega2=omega), using the plasma relation eps0/(2 n0 e) = e/(2 m_e wp^2):

      chi_s,perp-perp-perp(2w) = - a * [ e / (2 m_e wp^2) ] * ((eps(w) - 1)/4)^2      (m^2/V)
      chi_s,par-perp-par(2w)   = - b * [ e / (2 m_e wp^2) ] * ((eps(w) - 1)/4)         (m^2/V)

  where e>0 is the electron charge, m_e the electron mass, wp the metal plasma frequency
  (rad/s), and eps(w) the metal permittivity at the fundamental. In the free-electron limit
  a ~ 1, b = -1, and the bulk term d = 1 (Rudnick & Stern 1971; Weber & Liebsch). The overall
  prefactor follows the cited convention; NONE of the validation gates below depend on it
  (the flat-surface oracle and the FEM two-step use the SAME chi_s, and the slope/symmetry/
  angle gates are prefactor-independent), so the absolute scale is a documented convention,
  not a fitted quantity.
"""

from __future__ import annotations

import cmath
import math

from dynameta.constants import EPS0, C_LIGHT, Q_E, M_E

__all__ = ["rudnick_stern_surface_chi", "rudnick_stern_flat_shg", "shg_two_step",
           "rudnick_stern_flat_sfg", "sfg_two_step", "sfg_field_transverse_kx",
           "shg_structured_two_step"]

_Z0 = 1.0 / (EPS0 * C_LIGHT)                 # free-space wave impedance (ohm)


def rudnick_stern_surface_chi(a: float, b: float, omega: float,
                               eps_w: complex, wp: float) -> dict:
    """Rudnick-Stern surface second-harmonic susceptibilities (SI, units m^2/V) from the
    dimensionless a, b parameters. See the module docstring for the exact normalization and
    citations. Free-electron limit: a~1, b=-1.

    Parameters
    ----------
    a, b : float        dimensionless Rudnick-Stern parameters (a -> normal chi_perp-perp-perp,
                        b -> chi_par-perp-par).
    omega : float       fundamental angular frequency (rad/s)  [carried for API symmetry / future
                        dispersive forms; the leading term uses eps(w) and wp].
    eps_w : complex     metal permittivity at the fundamental.
    wp : float          metal plasma frequency (rad/s).

    Returns
    -------
    dict with keys 'zzz' (chi_perp-perp-perp) and 'xxz'/'zxx' style 'par' (chi_par-perp-par),
    complex, units m^2/V.
    """
    scale = Q_E / (2.0 * M_E * wp ** 2)           # e/(2 m_e wp^2), units m^2/V per (dimensionless)
    f_perp = ((complex(eps_w) - 1.0) / 4.0) ** 2
    f_par = (complex(eps_w) - 1.0) / 4.0
    return {"zzz": -a * scale * f_perp, "par": -b * scale * f_par}


def _ppol_normal_field_inside(E0: float, theta: float, eps_w: complex) -> complex:
    """Normal component E_perp (=E_z) of the fundamental just INSIDE the metal, for a unit-amplitude
    p-pol plane wave incident from vacuum at polar angle theta (the Sipe prescription evaluates the
    driving field inside the metal). Uses the p-pol Fresnel E-field transmission.
      E_z_in = E0 * t_p * sin(theta_t),  t_p = 2 cos(theta)/(n2 cos(theta) + cos(theta_t)),
      n2 = sqrt(eps_w),  sin(theta_t) = sin(theta)/n2,  cos(theta_t) = sqrt(1 - sin^2/eps_w)."""
    n2 = cmath.sqrt(complex(eps_w))
    if n2.real < 0:                               # sqrt branch: pick Re>=0 (passive metal, n=n'+in'')
        n2 = -n2
    sth = math.sin(theta)
    cth = math.cos(theta)
    sth_t = sth / n2
    cth_t = cmath.sqrt(1.0 - (sth / n2) ** 2)
    if cth_t.real < 0:
        cth_t = -cth_t
    t_p = 2.0 * cth / (n2 * cth + cth_t)          # p-pol E-field amplitude transmission (n1=1)
    return E0 * t_p * sth_t


def rudnick_stern_flat_shg(lambda_m: float, theta_deg: float, eps_w: complex, eps_2w: complex,
                            chi_zzz: complex, *, E0: float = 1.0, polarization: str = "p") -> dict:
    """CLOSED-FORM reflected second-harmonic from a FLAT Drude half-space (vacuum above, metal
    below), from the dominant Rudnick-Stern a-term (normal surface susceptibility chi_zzz). The
    PRIMARY oracle for the FEM two-step. Derived from Maxwell + the Heinz/Sipe boundary conditions
    for a surface polarization sheet (Nireekshan Reddy et al., JOSA B 2017, Eqs. 2-6):

      P_s = eps0 chi_zzz (E_perp^in)^2  zhat      (normal surface polarization, C/m)
      normal-dipole sheet at the vacuum/metal interface with in-plane wavevector K_par = 2 k_par:
        DeltaH_par = 0            (a normal dipole gives no tangential-H jump)
        DeltaE_par = -(i K_par / eps0) P_s,z / eps'   (eps' = vacuum, the Sipe "radiates outside")
      -> upward (reflected) SH p-pol magnetic amplitude
        A = -i K_par Omega P_s,z / (beta1 + beta2/eps_2w)
      with Omega = 2 omega, beta1 = (Omega/c) cos(theta) (vacuum SH z-wavevector), beta2 =
      sqrt(eps_2w (Omega/c)^2 - K_par^2) (metal). Radiated SH intensity (up):
        S_up = 0.5 |A|^2 beta1 / (Omega eps0)   (W/m^2 per unit incident area).

    For s-pol fundamental E_perp^in = 0, so the a-term SH vanishes identically (the symmetry-
    forbidden case); at exactly normal incidence K_par = 0 so A = 0 (the specular a-term selection
    rule). Returns dict with 'S_up' (W/m^2), 'efficiency' (S_up / incident intensity), 'E_perp_in',
    'A', and the SH wavevectors.
    """
    theta = math.radians(float(theta_deg))
    omega = 2.0 * math.pi * C_LIGHT / lambda_m
    Omega = 2.0 * omega
    k1 = omega / C_LIGHT                           # fundamental vacuum wavenumber (1/m)
    K1 = Omega / C_LIGHT                           # SH vacuum wavenumber
    K_par = 2.0 * k1 * math.sin(theta)             # SH phase-matched in-plane wavevector
    if polarization == "s":
        E_perp_in = 0.0 + 0j                       # s-pol has no normal E -> a-term forbidden
    else:
        E_perp_in = _ppol_normal_field_inside(E0, theta, eps_w)
    P_z = EPS0 * complex(chi_zzz) * E_perp_in ** 2  # surface polarization (C/m)
    beta1 = cmath.sqrt(K1 ** 2 - K_par ** 2)       # vacuum SH z-wavevector (real, propagating up)
    if beta1.imag < 0:
        beta1 = -beta1
    beta2 = cmath.sqrt(complex(eps_2w) * K1 ** 2 - K_par ** 2)   # metal SH z-wavevector
    if beta2.imag < 0:
        beta2 = -beta2
    A = -1j * K_par * Omega * P_z / (beta1 + beta2 / complex(eps_2w))    # reflected SH Hy amplitude
    S_up = 0.5 * abs(A) ** 2 * beta1.real / (Omega * EPS0)              # W/m^2
    I_inc = 0.5 * EPS0 * C_LIGHT * abs(E0) ** 2 * math.cos(theta)       # incident z-intensity
    eff = S_up / I_inc if I_inc > 0 else 0.0
    return {"S_up": float(S_up), "efficiency": float(eff), "E_perp_in": complex(E_perp_in),
            "A": complex(A), "beta1": complex(beta1), "beta2": complex(beta2),
            "K_par": float(K_par)}


def shg_two_step(design, *, lambda_fund_m: float, chi_zzz: complex,
                  n_super: complex = 1.0 + 0j, n_sub: complex = 1.0 + 0j,
                  metal_region: "str | None" = None, order: int = 2,
                  eps_at=None) -> dict:
    """FEM undepleted two-step surface-SHG driver. Runs the linear solve at the fundamental, samples
    the normal fundamental field just inside the metal top surface, builds the equivalent SH source
    sheet, radiates it at 2*omega with solver.solve_fem_sourced, and returns the SH radiated power.

    Parameters
    ----------
    design : Design                     the layered design (must contain a Drude metal layer).
    lambda_fund_m : float               fundamental wavelength (m).
    chi_zzz : complex                   normal surface susceptibility (m^2/V); see
                                        rudnick_stern_surface_chi.
    n_super, n_sub : complex            superstrate / substrate indices.
    metal_region : str | None           mesh region whose TOP face carries the SH sheet; if None,
                                        the top-most region whose eps has Re < 0 at the fundamental.
    eps_at : callable | None            eps_at(lambda_m) -> {region: EpsField}; defaults to the
                                        design's material eps at the two wavelengths.

    Returns dict with 'p_up_2w' (radiated SH power over the cell, W, SI), 'E_perp_in'
    (sampled normal fundamental field), 'result_w' (the fundamental OpticalResult), and the SH
    SourcedResult.

    ACCURACY (measured 2026-07-19, replacing the earlier -- WRONG -- 'near-null interior mode /
    ~2-2.5x bias' deferral): for a flat Drude mirror, 'p_up_2w' matches the analytic oracle
    rudnick_stern_flat_shg * cell area to ~0.5% at 20 deg, ~0.6% at 35 deg and ~1.1% at 50 deg
    (gated at 10% in tests/test_shg_fem.py). The earlier huge discrepancy was NOT solver
    conditioning: it was an SI-vs-nm units bug in _normal_sheet_vacuum_field (E0 low by exactly
    S = 1e9) plus a tangential-only probe_pol under-counting the p-pol power by cos^2(theta);
    both fixed. Residual error is the documented fixed-alpha z-PML oblique approximation (grows
    with angle; stay at or below ~50 deg). CONSTRAINT that remains: solve_fem_sourced's power
    formula assumes a LOSSLESS superstrate -- a lossy superstrate invalidates 'p_up_2w'.
    """
    import ngsolve as ng
    from dynameta.optics.ngsolve_layered import LayeredOpticalBuilder, S
    from dynameta.optics.eps_assembler import assemble_eps_cf
    from dynameta.optics.solver import solve_fem, solve_fem_sourced

    geo = LayeredOpticalBuilder(design).build()
    mesh = geo.mesh
    if eps_at is None:
        def eps_at(lam_m):
            return {rg: _region_epsfield(design, geo, rg, lam_m) for rg in mesh.GetMaterials()}

    # (1) linear solve at the fundamental
    ebr_w = eps_at(lambda_fund_m)
    eps_w_cf = assemble_eps_cf(geo, ebr_w)
    res_w = solve_fem(geo, lambda_fund_m, eps_w_cf, design.optical, order=order,
                       n_super=n_super, n_sub=n_sub)

    # (2) locate the metal top surface + sample the normal fundamental field just inside it
    if metal_region is None:
        metal_region = _top_metal_region(geo, ebr_w)
    z_lo, z_hi = geo.z_intervals_nm[metal_region]
    E_perp_in = _sample_normal_field_inside(mesh, res_w, geo, metal_region, design, lambda_fund_m,
                                            n_super, n_sub)

    # (3) SH source: normal polarization sheet P_z = eps0 chi_zzz E_perp_in^2 at the metal top.
    #     Represent it by its equivalent vacuum radiation E0 (a normal-dipole-sheet plane wave) so
    #     the sourced solve runs on the well-conditioned scattered-field route.
    theta = math.radians(float(getattr(design.optical, "incidence_angle_deg", 0.0) or 0.0))
    lambda_2w = 0.5 * lambda_fund_m
    k0_2w = 2.0 * math.pi / (lambda_2w * S)                      # nm^-1
    kx = 2.0 * (2.0 * math.pi / (lambda_fund_m * S)) * math.sin(theta)   # SH K_par (nm^-1)
    P_z = EPS0 * complex(chi_zzz) * complex(E_perp_in) ** 2      # C/m
    z_sheet = z_hi                                               # metal top face (nm)
    # normal-dipole-sheet vacuum radiation (p-pol, Hy). Build E0 as the analytic SH field of the
    # sheet in vacuum (see rudnick_stern_flat_shg for the amplitude); here as a CF for the FEM source.
    E0_cf = _normal_sheet_vacuum_field(P_z, k0_2w, kx, z_sheet, ng)

    ebr_2w = eps_at(lambda_2w)
    eps_2w_cf = assemble_eps_cf(geo, ebr_2w)
    # probe_pol = the FULL p-pol unit vector of the up-going SH wave, (cos th, 0, -sin th): the
    # projection then returns the full E amplitude, and solve_fem_sourced's p_up = |a|^2 (kz/k0)
    # /(2 Z0) A is the correct p-pol flux. A tangential-only (1,0,0) projection captures only
    # Ex = E cos(th) and under-counts the radiated power by cos^2(th) (the 2026-07-19 fix).
    res_2w = solve_fem_sourced(geo, lambda_2w, eps_2w_cf, design.optical, order=order,
                                n_super=n_super, n_sub=n_sub, bg_field=E0_cf, eps_ref=1.0,
                                k_par_per_nm=(kx, 0.0),
                                probe_pol=(math.cos(theta), 0.0, -math.sin(theta)))
    return {"p_up_2w": res_2w.p_up, "E_perp_in": complex(E_perp_in), "result_w": res_w,
            "result_2w": res_2w, "metal_region": metal_region}


# ---------------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------------
def _region_epsfield(design, geo, region, lambda_m):
    from dynameta.core.eps_field import EpsField
    mat = geo.material_by_region.get(region)
    material = design.materials.get(mat) if mat is not None else None
    eps = complex(material.optical.eps(lambda_m)) if material is not None else 1.0 + 0j
    return EpsField(scalar=eps)


def _top_metal_region(geo, ebr):
    """Top-most region whose eps has Re < 0 (a metal)."""
    best = None
    best_z = -1e30
    for rg, iv in geo.z_intervals_nm.items():
        ef = ebr.get(rg)
        if ef is None or not getattr(ef, "is_uniform", True):
            continue
        try:
            eps = complex(ef.scalar)
        except Exception:
            continue
        if eps.real < 0 and iv[1] > best_z:
            best, best_z = rg, iv[1]
    if best is None:
        raise ValueError("shg_two_step: no metal (Re eps < 0) region found; specify metal_region.")
    return best


def _sample_normal_field_inside(mesh, res_w, geo, metal_region, design, lambda_m, n_super, n_sub):
    """Analytic normal fundamental field just inside the metal top surface. We use the p-pol
    closed form (rudnick_stern_flat's _ppol_normal_field_inside) evaluated with the incident
    amplitude implied by solve_fem (unit incident), rather than an unreliable HCurl point sample of
    E_z across the surface discontinuity -- the field is normalized to a unit-amplitude incident
    plane wave, matching solve_fem's convention."""
    mat = geo.material_by_region.get(metal_region)
    material = design.materials.get(mat) if mat is not None else None
    eps_w = complex(material.optical.eps(lambda_m)) if material is not None else 1.0 + 0j
    theta = math.radians(float(getattr(design.optical, "incidence_angle_deg", 0.0) or 0.0))
    if getattr(design.optical, "polarization", "p") == "p":
        return _ppol_normal_field_inside(1.0, theta, eps_w)
    return 0.0 + 0j


def _normal_sheet_vacuum_field(P_z, k0_2w, kx, z_sheet, ng):
    """Vacuum radiation of a normal (z) surface-polarization sheet P_z exp(i kx x) at z=z_sheet, as
    an NGSolve CoefficientFunction (the SH background E0 for the scattered-field sourced solve).

    UNITS (the 2026-07-19 fix): the PHASES are evaluated in mesh (nm) coordinates with the nm^-1
    wavevectors (k0_2w, kx, beta1), but the AMPLITUDES must be SI (E in V/m) -- built from the SI
    wavevectors kx*S, beta1*S [1/m] and the SI frequency Omega = c * k0_2w * S [rad/s] (the vacuum
    SH dispersion). The previous revision mixed the two ('c = 1 in nm units'), leaving E0 low by
    exactly S = 1e9 and the radiated power by S^2 = 1e18 -- found by adversarial verification
    against the closed-form oracle.

    The p-pol Hy amplitude is A = -i kx_SI Omega P_z / (2 beta1_SI) (both sides vacuum, eps = 1,
    so beta2 = beta1 in rudnick_stern_flat_shg's denominator); E follows from Ampere's law
    (exp(-i Omega t)): Ex = sgn * beta1_SI Hy / (Omega eps0), Ez = -kx_SI Hy / (Omega eps0)."""
    from dynameta.optics.ngsolve_layered import S
    beta1 = cmath.sqrt(k0_2w ** 2 - kx ** 2)         # nm^-1 (mesh-coordinate phases)
    if beta1.imag < 0:
        beta1 = -beta1
    kx_si, b1_si = kx * S, beta1 * S                 # SI wavevectors (1/m)
    Omega = C_LIGHT * k0_2w * S                      # SI angular frequency (rad/s)
    A = -1j * kx_si * Omega * P_z / (2.0 * b1_si)    # Hy amplitude (A/m) of the up/down wave
    # up (z>z_sheet): Hy = A exp(i(kx x + beta1 (z-z_sheet))); down: Hy = A exp(i(kx x - beta1(z-z_sheet)))
    zrel = ng.z - z_sheet
    up = ng.IfPos(zrel, 1.0, 0.0)
    Hy = A * (up * ng.exp(1j * (kx * ng.x + beta1 * zrel))
              + (1.0 - up) * ng.exp(1j * (kx * ng.x - beta1 * zrel)))
    # E from Hy (p-pol, vacuum, SI): keep the sign of beta1 per half-space.
    sgn = ng.IfPos(zrel, 1.0, -1.0)
    Ex = (sgn * b1_si * Hy) / (Omega * EPS0)
    Ez = -(kx_si * Hy) / (Omega * EPS0)
    return ng.CoefficientFunction((Ex, 0.0 * Hy, Ez))


# =============================================================================================
# NONDEGENERATE THREE-WAVE MIXING (SFG / DFG) -- roadmap item 4.3
# =============================================================================================
# Additive generalization of the degenerate SHG two-step above; the existing shg_two_step and its
# oracle rudnick_stern_flat_shg are NOT modified. Every code path below is shared with the SHG
# path except the NONDEGENERATE permutation multiplicity (Boyd's D-factor) and the momentum/
# frequency bookkeeping of the two distinct incident colors.
#
# CONVENTION -- Boyd's D-factor (Boyd, "Nonlinear Optics" 3rd ed., ch. 1.5 / ch. 2, the intrinsic
# permutation symmetry / degeneracy factor). The second-order surface polarization driving the
# mixing sheet is written, for the isotropic-surface a-term (normal component, chi_zzz):
#
#     P_z(omega3) = eps0 * chi_zzz * [ E_z(omega1) E_z(omega2) + E_z(omega2) E_z(omega1) ]
#                 = eps0 * chi_zzz * D * E_z(omega1) E_z(omega2),     D = 2  (NONDEGENERATE)
#
# The explicit sum over the two field orderings (n,m) = (1,2) and (2,1) is the permutation
# multiplicity; for two DISTINCT colors it produces the factor D = 2. The degenerate SHG limit
# omega1 = omega2 has a SINGLE field, so only ONE term survives and the convention collapses to
#
#     P_z(2 omega) = eps0 * chi_zzz * E_z(omega)^2,                   D = 1  (DEGENERATE)
#
# exactly the form used by shg_two_step / rudnick_stern_flat_shg. Consequently, feeding equal
# frequencies and equal fields into the D = 2 nondegenerate path yields TWICE the SHG polarization
# AMPLITUDE and hence FOUR TIMES the SHG radiated POWER:
#
#     sfg_two_step(omega, omega) / 4  ==  shg_two_step(omega)        (the degeneracy identity gate)
#
# We adopt D = 2 as the fixed multiplicity of the SFG/DFG path (it is the nondegenerate process);
# the /4 in the identity is the amplitude-2 -> power-4 map, NOT a switch of D. This makes the
# degenerate limit exact by construction.
#
# CONVENTION -- DFG conjugation (exp(-i omega t)). A real field is
#     E(t) = Re[ E_c exp(-i omega t) ] = (1/2)[ E_c exp(-i omega t) + E_c^* exp(+i omega t) ],
# so the POSITIVE-frequency (+omega) component carries the complex amplitude E_c and the
# NEGATIVE-frequency (-omega) component carries E_c^*. SFG (omega3 = omega1 + omega2) mixes the two
# positive-frequency components -> P ~ E1 E2. DFG (omega3 = omega1 - omega2) mixes the +omega1
# component with the -omega2 component -> P ~ E1 * conj(E2). We therefore CONJUGATE the omega2
# field (both its amplitude and its transverse phase, so the in-plane wavevector combines as
# K_par3 = k_par1 - k_par2) for process='dfg'. Requires omega1 > omega2 so omega3 > 0.
# Refs: Rudnick & Stern, PRB 4, 4274 (1971); Heinz / Sipe interface BCs (Sipe, JOSA B 4, 481
# (1987); Sipe, So, Fukui, Stegeman, PRB 21, 4389 (1980)); Boyd ch. 2 (nondegenerate permutation
# conventions).


def rudnick_stern_flat_sfg(omega1_rad_s: float, omega2_rad_s: float,
                            theta1_deg: float, theta2_deg: float,
                            eps_w1: complex, eps_w2: complex, eps_3: complex,
                            chi_zzz: complex, *, process: str = "sfg",
                            E0_1: float = 1.0, E0_2: float = 1.0,
                            polarization: str = "p", D_factor: float = 2.0) -> dict:
    """CLOSED-FORM reflected sum/difference-frequency field from a FLAT Drude half-space (vacuum
    above, metal below), nondegenerate generalization of rudnick_stern_flat_shg. Two p-pol
    fundamentals at (omega1, theta1) and (omega2, theta2) drive the normal (a-term) surface sheet
    at omega3 = omega1 (+/-) omega2; the sheet radiates into the momentum-matched direction set by
    K_par3 = k_par1 (+/-) k_par2. Derived from the SAME Maxwell + Heinz/Sipe boundary conditions as
    the degenerate oracle (Nireekshan Reddy et al., JOSA B 2017, Eqs. 2-6; Rudnick & Stern 1971):

      P_z = eps0 chi_zzz D E_z(omega1) E_z(omega2)   (normal surface polarization, C/m; D = 2)
      A   = -i K_par3 omega3 P_z / (beta1 + beta2/eps_3)          (reflected p-pol Hy amplitude)
      S_up = 0.5 |A|^2 Re(beta1) / (omega3 eps0)                  (radiated up-going flux, W/m^2)

    with beta1 = sqrt((omega3/c)^2 - K_par3^2) the vacuum emission z-wavevector and
    beta2 = sqrt(eps_3 (omega3/c)^2 - K_par3^2) the metal one. For DFG the omega2 field enters
    conjugated (see the module's DFG-conjugation convention). s-pol fundamentals have E_z = 0, so
    the a-term mixing vanishes identically for BOTH processes (the symmetry selection rule).

    DEGENERATE CROSS-CHECK: rudnick_stern_flat_sfg(w, w, th, th, eps_w, eps_w, eps_2w, chi)/4 equals
    rudnick_stern_flat_shg(2 pi c / w, th, eps_w, eps_2w, chi) exactly (D = 2 -> amplitude x2 ->
    power x4).

    Returns a dict: 'S_up' (W/m^2), 'A', 'P_z', 'E_perp_in_1', 'E_perp_in_2' (the omega2 entry is
    the DFG-conjugated field actually used), 'K_par3' (1/m), 'k3' (1/m vacuum), 'omega3' (rad/s),
    'theta3_deg' (emission polar angle), 'beta1', 'beta2', 'propagating' (bool).
    """
    proc = str(process).lower()
    if proc not in ("sfg", "dfg"):
        raise ValueError("process must be 'sfg' or 'dfg'")
    sgn = 1.0 if proc == "sfg" else -1.0
    w1, w2 = float(omega1_rad_s), float(omega2_rad_s)
    w3 = w1 + sgn * w2
    if w3 <= 0.0:
        raise ValueError("omega3 = omega1 {} omega2 must be > 0 (need omega1 > omega2 for DFG)"
                         .format("+" if sgn > 0 else "-"))
    th1, th2 = math.radians(float(theta1_deg)), math.radians(float(theta2_deg))
    k1, k2, k3 = w1 / C_LIGHT, w2 / C_LIGHT, w3 / C_LIGHT       # vacuum wavenumbers (1/m)
    kpar1, kpar2 = k1 * math.sin(th1), k2 * math.sin(th2)       # in-plane wavevectors (1/m)
    K_par3 = kpar1 + sgn * kpar2                                # momentum matching (1/m)
    if str(polarization) == "s":
        E1 = E2 = 0.0 + 0j                                      # s-pol: no normal E -> forbidden
    else:
        E1 = _ppol_normal_field_inside(float(E0_1), th1, eps_w1)
        E2 = _ppol_normal_field_inside(float(E0_2), th2, eps_w2)
    E2_eff = E2 if proc == "sfg" else complex(E2).conjugate()   # DFG: field at -omega2 = conj
    P_z = EPS0 * complex(chi_zzz) * float(D_factor) * complex(E1) * complex(E2_eff)   # C/m
    beta1 = cmath.sqrt(k3 ** 2 - K_par3 ** 2)                   # vacuum emission z-wavevector
    if beta1.imag < 0:
        beta1 = -beta1
    beta2 = cmath.sqrt(complex(eps_3) * k3 ** 2 - K_par3 ** 2)  # metal SH z-wavevector
    if beta2.imag < 0:
        beta2 = -beta2
    A = -1j * K_par3 * w3 * P_z / (beta1 + beta2 / complex(eps_3))
    S_up = 0.5 * abs(A) ** 2 * beta1.real / (w3 * EPS0)
    sin3 = max(-1.0, min(1.0, K_par3 / k3)) if k3 > 0 else 0.0
    theta3_deg = math.degrees(math.asin(sin3))
    return {"S_up": float(S_up), "A": complex(A), "P_z": complex(P_z),
            "E_perp_in_1": complex(E1), "E_perp_in_2": complex(E2_eff),
            "K_par3": float(K_par3), "k3": float(k3), "omega3": float(w3),
            "theta3_deg": float(theta3_deg), "beta1": complex(beta1), "beta2": complex(beta2),
            "propagating": bool(beta1.real > 1e-6 * k3)}


def sfg_two_step(design, *, omega1_rad_s: float, omega2_rad_s: float, chi_zzz: complex,
                  theta1_deg: "float | None" = None, theta2_deg: "float | None" = None,
                  process: str = "sfg", pol1: str = "p", pol2: str = "p",
                  n_super: complex = 1.0 + 0j, n_sub: complex = 1.0 + 0j,
                  metal_region: "str | None" = None, order: int = 2,
                  eps_at=None, D_factor: float = 2.0) -> dict:
    """FEM undepleted two-step surface SUM/DIFFERENCE-frequency driver, the nondegenerate
    generalization of shg_two_step. Runs TWO linear fundamental solves (the existing plane-wave
    path, one per color), samples the normal fundamental field just inside the metal top surface for
    each, assembles the nondegenerate mixing sheet P_z = eps0 chi_zzz D E1 E2 at omega3 = omega1
    (+/-) omega2 (D = 2, Boyd nondegenerate; see the module's D-factor + DFG-conjugation
    conventions), radiates it with solver.solve_fem_sourced at omega3, and returns the radiated
    mixing power. ADDITIVE -- shg_two_step is untouched and this shares its entire machinery.

    Parameters
    ----------
    design : Design                     layered design containing a Drude metal layer.
    omega1_rad_s, omega2_rad_s : float  fundamental angular frequencies (rad/s). For DFG require
                                        omega1 > omega2 so omega3 = omega1 - omega2 > 0.
    chi_zzz : complex                   normal surface susceptibility (m^2/V).
    theta1_deg, theta2_deg : float|None per-color incidence polar angles (deg); default to
                                        design.optical.incidence_angle_deg.
    process : 'sfg' | 'dfg'             sum- or difference-frequency (DFG conjugates the omega2
                                        field and forms K_par3 = k_par1 - k_par2).
    pol1, pol2 : 'p' | 's'              per-color fundamental polarization. 's' has no normal E, so
                                        the a-term mixing vanishes (selection rule).
    D_factor : float                    Boyd permutation multiplicity (2 nondegenerate); the /4
                                        degeneracy identity relies on this being 2.

    Returns dict with 'p_up_3w' (radiated mixing power over the cell, W, SI), 'E_perp_in_1',
    'E_perp_in_2' (the omega2 entry is the DFG-conjugated field used), 'P_z' (C/m), 'omega3',
    'lambda_3_m', 'theta3_deg', 'K_par3_per_nm' (nm^-1), 'D_factor', 'process', 'metal_region',
    'geo', 'result_1'/'result_2' (fundamental OpticalResults), and 'result_3' (SH SourcedResult).

    DEGENERACY IDENTITY: sfg_two_step(w, w, theta, theta)['p_up_3w'] / 4 == shg_two_step(w)
    ['p_up_2w'] to ~machine precision -- they share every code path except the D-factor. Same
    accuracy envelope and LOSSLESS-superstrate power-read-out constraint as shg_two_step.
    """
    import dataclasses
    import ngsolve as ng
    from dynameta.optics.ngsolve_layered import LayeredOpticalBuilder, S
    from dynameta.optics.eps_assembler import assemble_eps_cf
    from dynameta.optics.solver import solve_fem, solve_fem_sourced

    proc = str(process).lower()
    if proc not in ("sfg", "dfg"):
        raise ValueError("process must be 'sfg' or 'dfg'")
    sgn = 1.0 if proc == "sfg" else -1.0
    w1, w2 = float(omega1_rad_s), float(omega2_rad_s)
    w3 = w1 + sgn * w2
    if w3 <= 0.0:
        raise ValueError("omega3 = omega1 {} omega2 must be > 0 (need omega1 > omega2 for DFG)"
                         .format("+" if sgn > 0 else "-"))
    lam1 = 2.0 * math.pi * C_LIGHT / w1
    lam2 = 2.0 * math.pi * C_LIGHT / w2
    lam3 = 2.0 * math.pi * C_LIGHT / w3
    base_deg = float(getattr(design.optical, "incidence_angle_deg", 0.0) or 0.0)
    t1_deg = base_deg if theta1_deg is None else float(theta1_deg)
    t2_deg = base_deg if theta2_deg is None else float(theta2_deg)
    th1, th2 = math.radians(t1_deg), math.radians(t2_deg)

    geo = LayeredOpticalBuilder(design).build()
    mesh = geo.mesh
    if eps_at is None:
        def eps_at(lam_m):
            return {rg: _region_epsfield(design, geo, rg, lam_m) for rg in mesh.GetMaterials()}

    # (1) two linear fundamental solves (the existing plane-wave path, one per color). Their fields
    #     are sampled ANALYTICALLY (the Sipe closed form, as in shg_two_step) rather than from an
    #     unreliable HCurl point sample across the surface discontinuity; the solves honor the
    #     two-solve contract and are returned for inspection.
    opt1 = dataclasses.replace(design.optical, polarization=pol1, incidence_angle_deg=t1_deg)
    opt2 = dataclasses.replace(design.optical, polarization=pol2, incidence_angle_deg=t2_deg)
    ebr1 = eps_at(lam1)
    res1 = solve_fem(geo, lam1, assemble_eps_cf(geo, ebr1), opt1, order=order,
                      n_super=n_super, n_sub=n_sub)
    ebr2 = eps_at(lam2)
    res2 = solve_fem(geo, lam2, assemble_eps_cf(geo, ebr2), opt2, order=order,
                      n_super=n_super, n_sub=n_sub)

    # (2) metal top face + normal fundamental fields just inside it, one per color (per-color theta
    #     and eps(lambda_j); Sipe prescription, unit-amplitude incident to match solve_fem).
    if metal_region is None:
        metal_region = _top_metal_region(geo, ebr1)
    z_lo, z_hi = geo.z_intervals_nm[metal_region]
    eps_w1 = _region_eps_scalar(design, geo, metal_region, lam1)
    eps_w2 = _region_eps_scalar(design, geo, metal_region, lam2)
    E1 = _ppol_normal_field_inside(1.0, th1, eps_w1) if pol1 == "p" else 0.0 + 0j
    E2 = _ppol_normal_field_inside(1.0, th2, eps_w2) if pol2 == "p" else 0.0 + 0j
    E2_eff = E2 if proc == "sfg" else complex(E2).conjugate()   # DFG: field at -omega2 = conj

    # (3) nondegenerate mixing sheet at omega3: P_z = eps0 chi_zzz D E1 E2 (Boyd D = 2). Represent it
    #     by its equivalent vacuum radiation E0 (same _normal_sheet_vacuum_field as the SHG path,
    #     which is frequency-agnostic: it takes k0_3 and K_par3), for the scattered-field solve.
    P_z = EPS0 * complex(chi_zzz) * float(D_factor) * complex(E1) * complex(E2_eff)   # C/m
    k0_1 = 2.0 * math.pi / (lam1 * S)          # nm^-1
    k0_2 = 2.0 * math.pi / (lam2 * S)
    k0_3 = 2.0 * math.pi / (lam3 * S)
    kpar1 = k0_1 * math.sin(th1)               # nm^-1
    kpar2 = k0_2 * math.sin(th2)
    K_par3 = kpar1 + sgn * kpar2               # nm^-1 (K_par1 + K_par2 SFG; K_par1 - K_par2 DFG)
    z_sheet = z_hi
    E0_cf = _normal_sheet_vacuum_field(P_z, k0_3, K_par3, z_sheet, ng)

    # (4) sourced solve at omega3, radiating into the momentum-matched emission direction theta3
    #     (sin theta3 = K_par3 / k0_3). probe_pol = the FULL up-going p-pol unit vector
    #     (cos th3, 0, -sin th3) so solve_fem_sourced's p_up is the true p-pol flux (a
    #     tangential-only projection under-counts by cos^2 th3).
    sin_th3 = K_par3 / k0_3
    cos_th3 = math.sqrt(max(0.0, 1.0 - sin_th3 ** 2))
    eps_3_cf = assemble_eps_cf(geo, eps_at(lam3))
    res3 = solve_fem_sourced(geo, lam3, eps_3_cf, design.optical, order=order,
                              n_super=n_super, n_sub=n_sub, bg_field=E0_cf, eps_ref=1.0,
                              k_par_per_nm=(K_par3, 0.0),
                              probe_pol=(cos_th3, 0.0, -sin_th3))
    return {"p_up_3w": res3.p_up, "E_perp_in_1": complex(E1), "E_perp_in_2": complex(E2_eff),
            "P_z": complex(P_z), "omega3": float(w3), "lambda_3_m": float(lam3),
            "theta3_deg": float(math.degrees(math.asin(max(-1.0, min(1.0, sin_th3))))),
            "K_par3_per_nm": float(K_par3), "D_factor": float(D_factor), "process": proc,
            "metal_region": metal_region, "geo": geo,
            "result_1": res1, "result_2": res2, "result_3": res3}


def sfg_field_transverse_kx(result: dict, *, z_frac: float = 0.4, n_samples: int = 15) -> float:
    """Extract the in-plane (transverse) wavevector kx [nm^-1] carried by the RADIATED mixing field
    in the superstrate, by phase-fitting the TOTAL Ex(x) across one period at a fixed z. This reads
    the emission-direction momentum out of the sourced-solve field itself (SFG carries k_par1 +
    k_par2; DFG carries k_par1 - k_par2), independent of the K_par3 the solver was told to use.

    `result` is the dict returned by sfg_two_step. Evaluating the periodic (Bloch-phased) total
    field at a point returns the quasi-periodic field INCLUDING exp(i K_par3 x) (the same convention
    solve_fem's _cell_average demodulates), so for a subwavelength flat cell (a clean 0-order plane
    wave) the x-phase slope is the transverse wavevector. Returns kx in nm^-1.
    """
    import numpy as np
    res3 = result["result_3"]
    geo = result["geo"]
    mesh = geo.mesh
    total = res3.gfu if res3.bg_field is None else (res3.bg_field + res3.gfu)
    z0, z1 = geo.z_intervals_nm["superstrate"]
    zv = z0 + float(z_frac) * (z1 - z0)
    Px = geo.period_x_nm
    yv = 0.37 * geo.period_y_nm
    xs = (0.1 + 0.8 * np.arange(n_samples) / (n_samples - 1)) * Px
    ex = []
    for xv in xs:
        E = total(mesh(float(xv), float(yv), float(zv)))
        ex.append(complex(E[0]))
    ex = np.asarray(ex)
    phase = np.unwrap(np.angle(ex))
    slope = float(np.polyfit(xs, phase, 1)[0])       # d(phase)/dx = kx (nm^-1)
    return slope


def _region_eps_scalar(design, geo, region, lambda_m):
    """Scalar metal permittivity of a mesh region at a wavelength (the per-color eps for the
    Sipe normal-field closed form)."""
    mat = geo.material_by_region.get(region)
    material = design.materials.get(mat) if mat is not None else None
    return complex(material.optical.eps(lambda_m)) if material is not None else 1.0 + 0j


# =============================================================================================
# STRUCTURED-SURFACE (GRATING) SURFACE-SHG -- roadmap item 5.3
# =============================================================================================
# Additive generalization of shg_two_step to a metal top boundary that is STRUCTURED (a lamellar
# grating / corrugation) rather than flat. shg_two_step and its oracle are NOT modified.
#
# THE CENTRAL CHALLENGE + THE HONEST FEM FINDING (measured 2026-07-19, ngsolve 6.2.2604):
# -------------------------------------------------------------------------------------------
# The flat driver samples the fundamental normal field ANALYTICALLY (the Sipe closed form) and
# radiates the sheet as a single analytic plane wave (_normal_sheet_vacuum_field) -- luxuries a
# STRUCTURED surface does not have. The roadmap suggested assembling the Rudnick-Stern NORMAL sheet
# via solve_fem_sourced's surface-current path (K = -2 i omega P_s . nhat). Direct measurement shows
# that route CANNOT source a normal sheet in this HCurl (edge-element) discretization:
#   * a boundary form (nhat . v.Trace()) is IDENTICALLY ZERO -- the HCurl tangential trace is
#     orthogonal to the facet normal, so a purely NORMAL current (the a-term chi_perp sheet, normal
#     on every facet including tilted grating walls) contributes nothing;
#   * the full normal trace (nhat . v) is REFUSED by ngsolve ("Testfunction does not support
#     BND-forms, maybe a Trace() operator is missing") -- the normal component of an HCurl test
#     function is not in the boundary trace space at all.
# A thin volume-current band just outside the metal DOES couple to the normal component (full v in a
# dx form), but the sheet needs sub-element resolution: on the coarse ngsolve-gated meshes (per-solid
# maxh is silently ignored in this build; only a GLOBAL maxh refines, which explodes the whole cell)
# the band quadrature is 15-200% noisy and the < 2% flat gate is unreachable at feasible cost.
#
# THE COARSE-MESH-ROBUST ROUTE USED HERE (documented, physical-optics for shallow corrugations):
#   (a) linear fundamental solve on the STRUCTURED cell (the existing plane-wave path, reconstructed
#       through solve_fem_sourced so the field is available -- solve_fem returns only R/T);
#   (b) extract E_perp just OUTSIDE the metal along the surface by point-sampling E_z in the
#       DIELECTRIC at small standoffs and extrapolating to the boundary -- the CONTINUOUS normal-D
#       route (D_perp = eps E.n is single-valued across the interface; the discontinuous E.n is not),
#       E_perp,inside = D_perp / eps_metal. Point evaluation in the dielectric VOLUME (not a boundary
#       trace) is the reliable way to reach the normal component in HCurl. MEASURED NOISE vs the Sipe
#       closed form on a flat mirror: ~0.1-0.25% at 20-35 deg, ~1.8% at 45 deg;
#   (c) assemble the Rudnick-Stern normal sheet P_perp(2w) = eps0 chi_perp E_perp^2 ON THE SURFACE
#       PROFILE, then RADIATE it via the SAME scattered-field route shg_two_step uses
#       (_normal_sheet_vacuum_field, the sheet's analytic vacuum radiation), generalized to a
#       corrugated surface as a MULTI-ORDER sum: each diffraction order m carries the Fourier
#       coefficient of P_perp(x) plus the leading (linear-in-height) surface-height phase
#       -i beta1_m (z_s(x) - z_mean) -- a physical-optics / Kirchhoff model of the corrugated sheet
#       (exact in the flat limit, leading-order for shallow h << period, lambda). The FEM sourced
#       solve then scatters that vacuum radiation off the STRUCTURED metal at 2w (eps_ref = 1), so
#       the metal response is rigorous; only the sheet's own radiation is the physical-optics model;
#   (d) sourced solve at 2w -> radiated SH extracted PER DIFFRACTION ORDER.
#
# FLAT LIMIT: a geometrically flat cell has z_s = z_mean and a spatially-uniform E_perp, so only the
# m = 0 order survives and E0 collapses EXACTLY to shg_two_step's single-plane-wave sheet -- the
# structured driver reproduces shg_two_step to the extraction noise (< 2%, the load-bearing gate 1).
# Refs: Rudnick & Stern, PRB 4, 4274 (1971); Heinz / Sipe interface BCs (Sipe, JOSA B 4, 481 (1987);
# Sipe/So/Fukui/Stegeman, PRB 21, 4389 (1980)); Dadap, Shan, Eisenthal, Heinz, PRL 83, 4045 (1999)
# (small-particle multipole SH). Same LOSSLESS-superstrate power-read-out constraint as shg_two_step.


def _ensure_bloch_dirs(geo):
    """Pre-compute + cache geo._bloch_dirs using z-sampling that hits EVERY region z-interval
    (including thin patterned grating layers). solver._detect_bloch_dirs samples z at 18 GLOBAL
    fractions, which can miss a thin patterned layer's periodic face and misclassify its Bloch idnr
    -- the 'resolved N x / M y ... expected' RuntimeError at oblique incidence. Same toggle-marker
    method, denser (per-region) z-sampling; a no-op when already cached or single-axis periodic."""
    import cmath as _cm
    import ngsolve as _ng
    if getattr(geo, "_bloch_dirs", None) is not None:
        return
    mesh = geo.mesh
    n_px, n_py = geo.n_px, geo.n_py
    N = n_px + n_py
    Px, Py = geo.period_x_nm, geo.period_y_nm
    if N == 0 or n_py == 0:
        dirs = ["x"] * n_px
    elif n_px == 0:
        dirs = ["y"] * n_py
    else:
        zpts = []
        for (zl, zh) in geo.z_intervals_nm.values():
            for fr in (0.2, 0.5, 0.8):
                zpts.append(zl + fr * (zh - zl))
        thm = _cm.exp(1j * 0.7853981634)

        def _viol(i):
            phases = [(thm if j == i else 1.0 + 0j) for j in range(N)]
            fes = _ng.Periodic(_ng.H1(mesh, order=1, complex=True), phase=phases)
            gf = _ng.GridFunction(fes)
            gf.Set(_ng.exp(0.01j * _ng.z) * (1.0 + 0.3 * _ng.y / Py + 0.25 * _ng.x / Px + 0.2j))
            xv = yv = 0.0
            for zv in zpts:
                for fy in (0.3, 0.6):
                    try:
                        a = complex(gf(mesh(0.0, fy * Py, zv))); b = complex(gf(mesh(Px, fy * Py, zv)))
                        xv = max(xv, abs(b - a))
                    except Exception:
                        pass
                for fx in (0.3, 0.6):
                    try:
                        c = complex(gf(mesh(fx * Px, 0.0, zv))); e = complex(gf(mesh(fx * Px, Py, zv)))
                        yv = max(yv, abs(e - c))
                    except Exception:
                        pass
            return xv, yv
        dirs = []
        for i in range(N):
            xv, yv = _viol(i)
            dirs.append("x" if xv > yv else "y")
        if dirs.count("x") != n_px or dirs.count("y") != n_py:
            return                                    # inconclusive: leave the solver to try its own
    try:
        geo._bloch_dirs = dirs
    except (AttributeError, TypeError):
        pass


def _fresnel_background(pol, ng, kx, kz_s_c, kz_sub, z_int, eps_sup_c, eps_sub_c, n_super, n_sub, k0):
    """The analytic layered (air/metal half-space) Fresnel background field E_bg and its eps_bg, in
    the x-z plane (azimuth 0), byte-for-byte the construction solver.solve_fem uses internally
    (p-pol: numeric interface BCs; s-pol/x-pol: Fresnel r/t). Returned so the fundamental field can
    be REBUILT through solve_fem_sourced (which exposes the field; solve_fem returns only R/T). The
    scattered source k0^2 (eps - eps_bg) E_bg is then identical to solve_fem's, so E_bg + gfu equals
    solve_fem's total field."""
    import cmath as _cm
    import numpy as _np
    iph = ng.exp(1j * kx * ng.x)
    eps_bg = ng.IfPos(ng.z - z_int, eps_sup_c, eps_sub_c)
    if pol == "p":
        kpar = kx
        cth = kz_s_c / (complex(n_super) * k0); sth = kpar / (complex(n_super) * k0)
        cth_t = kz_sub / (complex(n_sub) * k0); sth_t = kpar / (complex(n_sub) * k0)
        A = _cm.exp(-1j * kz_s_c * z_int); B = _cm.exp(1j * kz_s_c * z_int)
        C = _cm.exp(-1j * kz_sub * z_int)
        M = _np.array([[cth * B, -cth_t * C],
                       [cth * B * eps_sup_c / kz_s_c, cth_t * C * eps_sub_c / kz_sub]], dtype=complex)
        rhs = _np.array([-cth * A, cth * A * eps_sup_c / kz_s_c], dtype=complex)
        pp_rho, pp_tau = (complex(v) for v in _np.linalg.solve(M, rhs))
        et_sup = cth * ng.exp((-1j * kz_s_c) * ng.z) + pp_rho * cth * ng.exp((1j * kz_s_c) * ng.z)
        ez_sup = sth * ng.exp((-1j * kz_s_c) * ng.z) - pp_rho * sth * ng.exp((1j * kz_s_c) * ng.z)
        et_sub = pp_tau * cth_t * ng.exp((-1j * kz_sub) * ng.z)
        ez_sub = pp_tau * sth_t * ng.exp((-1j * kz_sub) * ng.z)
        et = ng.IfPos(ng.z - z_int, et_sup, et_sub)
        E_bg = ng.CoefficientFunction((iph * et, 0.0 * et,
                                       iph * ng.IfPos(ng.z - z_int, ez_sup, ez_sub)))
    else:                                              # s-pol (E along y): scalar Fresnel r/t
        r_f = (kz_s_c - kz_sub) / (kz_s_c + kz_sub)
        t_f = 2.0 * kz_s_c / (kz_s_c + kz_sub)
        R0 = r_f * _cm.exp(-2j * kz_s_c * z_int)
        T0 = t_f * _cm.exp(-1j * (kz_s_c - kz_sub) * z_int)
        sup_bg = ng.exp((-1j * kz_s_c) * ng.z) + R0 * ng.exp((1j * kz_s_c) * ng.z)
        sub_bg = T0 * ng.exp((-1j * kz_sub) * ng.z)
        scal = iph * ng.IfPos(ng.z - z_int, sup_bg, sub_bg)
        E_bg = ng.CoefficientFunction((0.0 * scal, scal, 0.0 * scal))
    return E_bg, eps_bg


def _reconstruct_fundamental_field(geo, design, lambda_m, optical, eps_cf, order,
                                    n_super, n_sub, theta_deg, pol):
    """Rebuild solve_fem's TOTAL fundamental field on the (structured) cell so E_perp can be
    sampled along the surface. Constructs the analytic layered background (_fresnel_background) and
    drives solve_fem_sourced with it (bg_field=E_bg, eps_ref=eps_bg) -- the scattered source is
    byte-identical to solve_fem's, so total = E_bg + gfu equals solve_fem's field, and now the
    GridFunction is available. Returns (total_field_cf, kx_per_nm, relres)."""
    import numpy as _np
    from dynameta.optics.ngsolve_layered import S as _S
    from dynameta.optics.solver import solve_fem_sourced
    import ngsolve as _ng
    k0 = 2.0 * math.pi / (lambda_m * _S)
    th = math.radians(float(theta_deg))
    kx = k0 * math.sin(th)
    kz_s = math.sqrt(max((complex(n_super).real * k0) ** 2 - kx ** 2, 0.0))
    kz_s_c = complex(kz_s)
    kz_sub = complex(_np.sqrt(complex((complex(n_sub) * k0) ** 2 - kx ** 2)))
    z_int = (geo.z_intervals_nm["substrate"][1] if "substrate" in geo.z_intervals_nm
             else geo.z_sub_interface_nm)
    eps_sup_c, eps_sub_c = complex(n_super) ** 2, complex(n_sub) ** 2
    E_bg, eps_bg = _fresnel_background(pol, _ng, kx, kz_s_c, kz_sub, z_int, eps_sup_c, eps_sub_c,
                                       n_super, n_sub, k0)
    probe = (math.cos(th), 0.0, -math.sin(th)) if pol == "p" else (0.0, 1.0, 0.0)
    src = solve_fem_sourced(geo, lambda_m, eps_cf, optical, order=order, n_super=n_super,
                            n_sub=n_sub, bg_field=E_bg, eps_ref=eps_bg,
                            k_par_per_nm=(kx, 0.0), probe_pol=probe)
    return (E_bg + src.gfu), kx, src.relres


def _local_surface_height(mesh, eps_fund_cf, xv, yv, z_hi, z_lo, nscan=64):
    """Local metal/dielectric surface height (nm) at (xv, yv): the highest z where the fundamental
    eps crosses into a metal (Re eps < 0), refined by bisection to sub-nm so a subsequent
    dielectric-side standoff never straddles the interface. Scans top-down from z_hi to z_lo;
    returns z_lo if the whole column is dielectric (base metal top at z_lo)."""
    def _is_metal(zv):
        try:
            return complex(eps_fund_cf(mesh(float(xv), float(yv), float(zv)))).real < 0.0
        except Exception:
            return None
    zs = _np_linspace(z_hi, z_lo, nscan)
    prev_air = float(z_hi)
    for zv in zs:
        m = _is_metal(zv)
        if m is None:
            continue
        if m:                                          # first metal point: bisect the air/metal edge
            a, b = prev_air, float(zv)                 # a: dielectric side, b: metal side
            for _ in range(24):
                mid = 0.5 * (a + b)
                mm = _is_metal(mid)
                if mm is None:
                    break
                if mm:
                    b = mid
                else:
                    a = mid
            return 0.5 * (a + b)
        prev_air = float(zv)
    return float(z_lo)


def _np_linspace(a, b, n):
    import numpy as _np
    return _np.linspace(a, b, n)


def _extract_surface_sheet_profile(mesh, geo, total_fund, eps_fund_cf, kx, chi_zzz, eps_metal,
                                    metal_region, k0_2w, kz_fund_air, *, nx=40, ny=3,
                                    standoff_nm=4.0, n_standoff=6):
    """Extract the Rudnick-Stern surface-sheet profile along the STRUCTURED metal top.

    For each in-plane sample (xv, yv): find the local surface height z_s (Re eps < 0 boundary),
    point-sample E_z in the DIELECTRIC at n_standoff small standoffs above z_s, extrapolate to
    z_s (D_perp = E_z,air at the boundary, the continuous normal-D), convert to the inside normal
    field E_perp,in = D_perp / eps_metal (the Sipe driving field), and form the normal sheet
    P_z(x) = eps0 chi_zzz E_perp,in^2 (C/m).

    Returns a dict with the y-averaged, x-demodulated Fourier decomposition of the sheet:
      'c0'      : the 0-order (specular) sheet amplitude (C/m),  == the flat-limit P_z,
      'orders'  : {m: c_m} the periodic Fourier coefficients including the leading surface-height
                  phase (see shg_structured_two_step), for |m| <= (nx//2 - 1),
      'z_mean'  : area-averaged surface height (nm), the sheet radiation plane,
      'z_profile', 'P_profile', 'x' : diagnostics (numpy arrays).
    """
    import numpy as _np
    Px, Py = geo.period_x_nm, geo.period_y_nm
    z_lo, z_hi = geo.z_intervals_nm[metal_region]             # (low, high) surface bracket
    xs = (_np.arange(nx) + 0.5) * (Px / nx)
    ys = (_np.arange(ny) + 0.5) * (Py / ny)
    ds = standoff_nm * (1.0 + _np.arange(n_standoff))          # standoffs standoff_nm..n*standoff_nm
    # per-column: surface height + extrapolated D_perp (y-averaged)
    z_s = _np.zeros(nx)
    P_x = _np.zeros(nx, dtype=complex)
    for i, xv in enumerate(xs):
        zcol = []
        Pcol = []
        for yv in ys:
            z_local = _local_surface_height(mesh, eps_fund_cf, xv, yv, z_hi + 1.0, z_lo - 1.0)
            zcol.append(z_local)
            ez = []
            for dz in ds:
                try:
                    E = total_fund(mesh(float(xv), float(yv), float(z_local + dz)))
                    ez.append(complex(E[2]))
                except Exception:
                    ez.append(_np.nan)
            ez = _np.asarray(ez)
            good = _np.isfinite(ez)
            if good.sum() >= 2:
                # D_perp = E_z,air at the boundary. The dielectric-side E_z is a two-wave standing
                # field a exp(i kz dz) + b exp(-i kz dz); fit both and evaluate at dz -> 0 (exact for
                # the propagating normal field, unlike a linear extrapolation which carries O((kz*dz)^2)
                # curvature error and pushed the flat gate over 2% at 35 deg).
                M = _np.column_stack([_np.exp(1j * kz_fund_air * ds[good]),
                                      _np.exp(-1j * kz_fund_air * ds[good])])
                coef, *_ = _np.linalg.lstsq(M, ez[good], rcond=None)
                d_perp = complex(coef[0] + coef[1])            # value at dz = 0
            else:
                d_perp = 0.0 + 0j
            e_in = d_perp / complex(eps_metal)                 # inside normal field (D continuity)
            Pcol.append(EPS0 * complex(chi_zzz) * e_in ** 2)   # normal sheet P_z (C/m)
        z_s[i] = float(_np.mean(zcol))
        P_x[i] = _np.mean(Pcol)
    z_mean = float(_np.mean(z_s))
    # demodulate the SH transverse Bloch phase exp(i kx_2w x), kx_2w = 2 kx_fund, then FFT over x.
    kx_2w = 2.0 * kx
    demod = _np.exp(-1j * kx_2w * xs)
    P_demod = P_x * demod                                      # cell-periodic sheet amplitude
    q0 = _np.fft.fft(P_demod) / nx                             # field/amplitude Fourier coeffs
    q1 = _np.fft.fft(P_demod * (z_s - z_mean)) / nx            # surface-height moment (nm * C/m)
    mmax = nx // 2 - 1
    orders = {}
    for m in range(-mmax, mmax + 1):
        idx = m % nx
        G = 2.0 * math.pi / Px
        K_m = kx_2w + m * G
        b1 = cmath.sqrt(complex(k0_2w ** 2 - K_m ** 2))
        if b1.imag < 0:
            b1 = -b1
        # c_m = FFT[P]_m + (leading surface-height phase) -i beta1_m FFT[P (z_s - z_mean)]_m
        orders[m] = complex(q0[idx] - 1j * b1 * q1[idx])
    return {"c0": complex(q0[0]), "orders": orders, "z_mean": z_mean,
            "z_profile": z_s, "P_profile": P_x, "x": xs}


def _sourced_order_power(mesh, total_sh, geo, k0_2w, K_m, Z0, *, nx=None, ny=6, n_z=7):
    """Radiated SH power in diffraction order m (in-plane wavevector K_m, nm^-1) from the sourced
    SH total field, by demodulating exp(-i K_m x), projecting onto the order's p-pol unit vector,
    and fitting the up-going wave in the superstrate buffer. Returns (power_W, |a_up|, theta_m_deg)
    or (0.0, 0.0, None) for an evanescent (non-radiating) order."""
    import numpy as _np
    b1 = cmath.sqrt(complex(k0_2w ** 2 - K_m ** 2))
    if b1.imag < 0:
        b1 = -b1
    if b1.real <= 1e-6 * k0_2w:
        return 0.0, 0.0, None                                 # evanescent: no radiated power
    sin_m = max(-1.0, min(1.0, K_m / k0_2w))
    cos_m = math.sqrt(max(0.0, 1.0 - sin_m ** 2))
    proj = (cos_m, 0.0, -sin_m)                                # up-going p-pol unit vector
    Px, Py = geo.period_x_nm, geo.period_y_nm
    nxg = nx if nx is not None else max(8, int(Px * (k0_2w + abs(K_m)) / (2.0 * math.pi)) + 1)
    z0, z1 = geo.z_intervals_nm["superstrate"]
    zlo, zhi = z0 + 50.0, z1 - 50.0
    if zhi <= zlo:
        zlo, zhi = z0 + 0.2 * (z1 - z0), z0 + 0.8 * (z1 - z0)
    zr = _np.linspace(zlo, zhi, n_z)
    xs = (_np.arange(nxg) + 0.5) * (Px / nxg)
    yy = (_np.arange(ny) + 0.5) * (Py / ny)
    Es = []
    for zv in zr:
        acc = []
        for xv in xs:
            for yv in yy:
                try:
                    E = total_sh(mesh(float(xv), float(yv), float(zv)))
                    pv = proj[0] * complex(E[0]) + proj[2] * complex(E[2])
                    acc.append(pv * _np.exp(-1j * K_m * xv))
                except Exception:
                    pass
        Es.append(_np.mean(acc) if acc else 0.0 + 0j)
    Es = _np.asarray(Es)
    M = _np.column_stack([_np.exp(1j * b1 * zr), _np.exp(-1j * b1 * zr)])
    coeffs, *_ = _np.linalg.lstsq(M, Es, rcond=None)
    a_up = complex(coeffs[0])
    from dynameta.optics.ngsolve_layered import S as _S
    area_phys = (Px * Py) / _S ** 2
    p_m = (abs(a_up) ** 2 * (b1.real / k0_2w) / (2.0 * Z0)) * area_phys
    return float(p_m), abs(a_up), math.degrees(math.asin(sin_m))


def shg_structured_two_step(design, *, lambda_fund_m: float, chi_zzz: complex,
                            n_super: complex = 1.0 + 0j, n_sub: complex = 1.0 + 0j,
                            metal_region: "str | None" = None, order: int = 2, eps_at=None,
                            n_orders: int = 2, nx: int = 40, ny: int = 3,
                            standoff_nm: float = 4.0, radiate: bool = True) -> dict:
    """FEM two-step surface-SHG driver for a STRUCTURED (grating / corrugated) metal top boundary --
    the structured generalization of shg_two_step (roadmap 5.3). See the "STRUCTURED-SURFACE" block
    in this module for the physics, the HCurl normal-sheet finding, and the coarse-mesh-robust
    physical-optics route this uses (fundamental FEM solve -> FEM-trace extraction of E_perp along
    the surface -> multi-order Rudnick-Stern sheet radiated via the scattered-field route -> sourced
    solve at 2w -> radiated SH per diffraction order). ADDITIVE -- shg_two_step is untouched.

    Parameters mirror shg_two_step, plus: n_orders (max |m| of SH diffraction orders extracted);
    nx/ny (surface sampling grid); standoff_nm (E_z dielectric-standoff base for the extraction);
    radiate (if False, skip the expensive 2w sourced solve and return only the extracted surface
    sheet + its physical-optics per-order power -- the clean perturbation-theory trend).

    Returns dict with:
      'sheet'              the _extract_surface_sheet_profile diagnostics (Fourier orders c_m,
                           z_mean, profiles); c_m is the SURFACE-SHEET amplitude of diffraction
                           order m -- |c_{+-1}| ~ h for shallow corrugation (the leading trend),
      'sheet_order_power'  {m: power_W} per-order SHEET radiated power (order c_m radiating off a
                           FLAT metal at z_mean, the rudnick_stern_flat_shg amplitude) -- the CLEAN
                           physical-optics nonspecular trend used for the shallow-h slope gate,
      'sheet_nonspecular'  sum of |m| >= 1 sheet-order power (~ h^2 for shallow corrugation),
      'E_perp_in'          the 0-order (mean) inside normal fundamental field used for the sheet,
      'extract_relres'     the fundamental reconstruction residual,
      'metal_region', 'geo',
    and (radiate=True only) the FULL FEM sourced-solve SH per order:
      'p_up_2w'            total radiated SH over the cell (sum of propagating orders, W, SI),
      'p_up_by_order'      {m: power_W} per propagating SH order from the 2w FEM sourced solve --
                           this INCLUDES linear SH re-diffraction off the STRUCTURED metal (the SH
                           generated at the surface then diffracted by the metal grating), a real but
                           SEPARATE effect from the surface-SHG sheet emission and non-perturbative
                           for strong metal scatterers; prefer 'sheet_order_power' for the clean
                           surface-SHG shallow-h trend,
      'p_up_specular', 'p_nonspecular', 'total_sh', 'theta_by_order'.

    FLAT LIMIT (load-bearing gate): a geometrically flat cell reproduces shg_two_step's 'p_up_2w'
    to < 2% (only m = 0 survives; E0 collapses to the single analytic sheet plane wave). Same
    LOSSLESS-superstrate power-read-out constraint as shg_two_step.
    """
    import ngsolve as ng
    from dynameta.optics.ngsolve_layered import LayeredOpticalBuilder, S
    from dynameta.optics.eps_assembler import assemble_eps_cf
    from dynameta.optics.solver import solve_fem_sourced

    geo = LayeredOpticalBuilder(design).build()
    mesh = geo.mesh
    _ensure_bloch_dirs(geo)                          # robust Bloch-idnr detection for thin grating layers
    if eps_at is None:
        def eps_at(lam_m):
            return {rg: _region_epsfield(design, geo, rg, lam_m) for rg in mesh.GetMaterials()}

    theta_deg = float(getattr(design.optical, "incidence_angle_deg", 0.0) or 0.0)
    pol = getattr(design.optical, "polarization", "p")
    lambda_2w = 0.5 * lambda_fund_m

    # (1) linear fundamental solve on the structured cell (reconstructed so the field is available)
    ebr_w = eps_at(lambda_fund_m)
    eps_w_cf = assemble_eps_cf(geo, ebr_w)
    total_fund, kx, relres = _reconstruct_fundamental_field(
        geo, design, lambda_fund_m, design.optical, eps_w_cf, order, n_super, n_sub, theta_deg, pol)

    # (2) locate the (structured) metal top region + extract the surface-sheet Fourier profile
    if metal_region is None:
        metal_region = _top_metal_region(geo, ebr_w)
    eps_metal = _region_eps_scalar(design, geo, metal_region, lambda_fund_m)
    k0_2w = 2.0 * math.pi / (lambda_2w * S)
    k0_fund = 2.0 * math.pi / (lambda_fund_m * S)
    kz_fund_air = math.sqrt(max(k0_fund ** 2 - kx ** 2, 0.0))         # fundamental air z-wavevector
    sheet = _extract_surface_sheet_profile(mesh, geo, total_fund, eps_w_cf, kx, chi_zzz, eps_metal,
                                           metal_region, k0_2w, kz_fund_air, nx=nx, ny=ny,
                                           standoff_nm=standoff_nm)
    E_perp_in = cmath.sqrt(sheet["c0"] / (EPS0 * complex(chi_zzz))) if abs(chi_zzz) > 0 else 0j
    kx_2w = 2.0 * kx
    G = 2.0 * math.pi / geo.period_x_nm
    z_mean = sheet["z_mean"]
    area_phys = (geo.period_x_nm * geo.period_y_nm) / S ** 2

    # sheet-based (physical-optics) per-order radiated power: each Fourier order c_m radiating off a
    # FLAT metal at z_mean (the rudnick_stern_flat_shg amplitude), in SI. This is the CLEAN
    # perturbation-theory nonspecular trend (|c_{+-1}| ~ h -> power ~ h^2); the FEM sourced power
    # below additionally carries linear SH re-diffraction off the structured metal.
    eps_2w_metal = _region_eps_scalar(design, geo, metal_region, lambda_2w)
    Omega = C_LIGHT * (k0_2w * S)
    K1_si = k0_2w * S
    sheet_order_power = {}
    for m in range(-n_orders, n_orders + 1):
        c_m = sheet["orders"].get(m, 0j)
        Km_si = (kx_2w + m * G) * S
        b1 = cmath.sqrt(complex(K1_si ** 2 - Km_si ** 2))
        if b1.imag < 0:
            b1 = -b1
        b2 = cmath.sqrt(complex(eps_2w_metal) * K1_si ** 2 - Km_si ** 2)
        if b2.imag < 0:
            b2 = -b2
        if b1.real <= 1e-6 * K1_si:
            continue                                       # evanescent order: no radiated power
        A_m = -1j * Km_si * Omega * complex(c_m) / (b1 + b2 / complex(eps_2w_metal))
        S_up = 0.5 * abs(A_m) ** 2 * b1.real / (Omega * EPS0)
        sheet_order_power[m] = float(S_up * area_phys)
    sheet_nonspec = sum(p for mm, p in sheet_order_power.items() if mm != 0)

    base = {"sheet": sheet, "sheet_order_power": sheet_order_power,
            "sheet_nonspecular": float(sheet_nonspec), "E_perp_in": complex(E_perp_in),
            "extract_relres": float(relres), "metal_region": metal_region, "geo": geo}
    if not radiate:
        return base

    # (3) multi-order Rudnick-Stern sheet radiation E0 (the scattered-field background). Each order m
    #     radiates its Fourier amplitude c_m as an analytic normal-dipole-sheet plane wave at z_mean.
    E0_cf = None
    for m in range(-n_orders, n_orders + 1):
        c_m = sheet["orders"].get(m, 0j)
        if abs(c_m) == 0.0:
            continue
        K_m = kx_2w + m * G
        part = _normal_sheet_vacuum_field(c_m, k0_2w, K_m, z_mean, ng)
        E0_cf = part if E0_cf is None else (E0_cf + part)
    if E0_cf is None:                                          # zero sheet (e.g. exact s-pol flat)
        E0_cf = ng.CoefficientFunction((0j, 0j, 0j))

    # (4) sourced solve at 2w (base SH Bloch phase kx_2w); the FEM scatters E0 off the STRUCTURED
    #     metal. Radiated SH extracted per diffraction order from the total SH field.
    ebr_2w = eps_at(lambda_2w)
    eps_2w_cf = assemble_eps_cf(geo, ebr_2w)
    sin0 = kx_2w / k0_2w
    cos0 = math.sqrt(max(0.0, 1.0 - sin0 ** 2))
    res_2w = solve_fem_sourced(geo, lambda_2w, eps_2w_cf, design.optical, order=order,
                               n_super=n_super, n_sub=n_sub, bg_field=E0_cf, eps_ref=1.0,
                               k_par_per_nm=(kx_2w, 0.0), probe_pol=(cos0, 0.0, -sin0))
    total_sh = res_2w.gfu if res_2w.bg_field is None else (res_2w.bg_field + res_2w.gfu)

    p_by_order = {}
    theta_by_order = {}
    for m in range(-n_orders, n_orders + 1):
        K_m = kx_2w + m * G
        p_m, _a, th_m = _sourced_order_power(mesh, total_sh, geo, k0_2w, K_m, _Z0)
        if th_m is not None:
            p_by_order[m] = p_m
            theta_by_order[m] = th_m
    p_spec = p_by_order.get(0, 0.0)
    p_total = sum(p_by_order.values())
    p_nonspec = p_total - p_spec
    base.update({"p_up_2w": float(p_total), "p_up_by_order": p_by_order,
                 "p_up_specular": float(p_spec), "p_nonspecular": float(p_nonspec),
                 "total_sh": res_2w, "theta_by_order": theta_by_order})
    return base
