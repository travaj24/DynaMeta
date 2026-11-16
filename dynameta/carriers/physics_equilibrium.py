"""
Equilibrium (single-variable nonlinear Poisson) DEVSIM physics, ported from the
proven this-session formulation. Potential is the ONLY solution variable;
electron density is a DERIVED node_model n = N_c * F_1/2((V - Phi_c0)/V_t).
NO continuity equation, NO currents (that is Phase 4 drift_diffusion).

Self-contained: Aymerich-Humet F_1/2 (DEVSIM's built-in Fermi is broken at the
degenerate eta ITO operates at), Phi_c0 calibration (n = n_bg at V = 0), and the
region/contact/interface setup helpers. Contact node-models use UNIQUE per-contact
names (two grounds on one region must not collide -> the empty-row singularity
this session root-caused).

v0.16 primitives (all opt-in; a device that uses none keeps every model string unchanged):
  * setup_semiconductor_region(fermi_ref_param=...): the region's Fermi level becomes -q <param>, the
    exact equilibrium effect of an edge contact at that voltage, without a Dirichlet column;
  * install_sheet_charge / install_bulk_charge / add_potential_charge_term: fixed (or, through the
    last, Potential-dependent) charge in any region's Poisson row (sign convention at the definitions);
  * work_function_offset_V: the gate-contact offset -(phi_m - W_s)/q of a metal work function;
  * setup_semiconductor_region(band=KaneBand(...)): the Kane (nonparabolic) closure (carriers.kane_band),
    with Phi_c0 calibrated on it; closure_kwargs picks dos_mass_kg / band for a TransportModel.
"""

from __future__ import annotations

import math
import re
from typing import Optional

# audit S1-2: devsim is imported LAZILY (module __getattr__) so the pure-math Fermi machinery
# (F12_aymerich_humet / invert_F12 / require_positive -- the Phi_c0 calibration whose wrong root
# would silently corrupt every downstream carrier solve) is importable and TESTABLE without the
# heavy solver stack. The ds.* call sites resolve through this shim on first devsim use.

# audit X-23 (2026-11-02): the shim class was a character-for-character copy of eq_registry's; this
# module already imports eq_registry, so it now shares that ONE lazy `ds` object.
from dynameta.carriers import eq_registry as _R
from dynameta.carriers.eq_registry import _DevsimShim, ds  # noqa: F401  (re-exported spelling)

# Physical constants (SI): single source in core/constants. Re-exported here because several
# carrier modules historically import Q_E / V_T / EPS0 / M_E from physics_equilibrium.
from dynameta.constants import Q_E, EPS0, KB, HBAR, M_E, T_REF, V_T  # noqa: E402,F401

# v0.16: the region parameter name the builders use for an equilibrium region's Fermi-level
# reference (setup_semiconductor_region(fermi_ref_param=FERMI_REF_PARAM)); its derivative is precomputed
# in _SIMPLIFIED, another name works but simplifies live (slow, correct).
FERMI_REF_PARAM = "EF_ref_V"
# The equilibrium (and unipolar drift-diffusion) Poisson node charge -rho = q (n - N_D) [C/m^3]: ONE
# spelling, read by setup_semiconductor_region and by the charge-term composite (add_potential_charge_term),
# which checks it against the live model before extending it.
_NET_CHARGE = "ElectronCharge * (Electrons - N_D)"
_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_RESERVED_SYMBOLS = frozenset(("Potential", "Electrons", "Holes", "Phi_c0", "V_t", "N_c", "N_D",
                               "ElectronCharge", "Permittivity", "NodeVolume", "x", "y", "z", "KaneDelta"))
# (device, region) -> the eta expression of the region's closure (equilibrium_eta_expr); written by every
# setup_semiconductor_region call, so it can never be stale for a live equilibrium region.
_ETA_BY_REGION: dict = {}

_AH_C = 3.0 * math.sqrt(math.pi) / 4.0
# Aymerich-Humet F_1/2 fit coefficients: ONE source for the python evaluator AND the
# DEVSIM expression twin below (each previously re-typed them; repr() keeps the DEVSIM
# string byte-identical).
_AH_P0, _AH_P1, _AH_B0, _AH_B1 = 50.0, 33.6, 0.68, 0.17


def require_positive(**named_values) -> None:
    """Anti-silent-failure guard for the physics-setup boundary: RAISE ValueError if any named
    value is not finite and strictly positive. A NaN / negative / zero material parameter (eps,
    DOS mass, mobility, lifetime, density) otherwise sails straight into the DEVSIM solve and
    produces a silently-wrong or NaN result (e.g. dos_mass=0 -> N_c=0 -> n_bg/N_c=inf; tau=0 ->
    SRH 0/0 NaN; negative mobility -> wrong-sign current). Shared by all carrier physics setups."""
    for name, v in named_values.items():
        fv = float(v)
        if not math.isfinite(fv) or fv <= 0.0:
            raise ValueError("{} must be finite and > 0, got {!r}".format(name, v))


def F12_aymerich_humet(eta: float) -> float:
    a = eta**4 + _AH_P0 + _AH_P1 * eta * (1.0 - _AH_B0 * math.exp(-_AH_B1 * (eta + 1.0)**2))
    A = (3.0 * math.sqrt(math.pi) / 4.0) / a**(3.0 / 8.0)
    B = math.exp(-eta)
    return 1.0 / (A + B)


def F12_aymerich_humet_expr(eta_expr: str = "eta") -> str:
    """DEVSIM expression for F_1/2(eta). No sqrt() (parser lacks it); leading
    constant precomputed. Unclamped exp(-eta) -- log_damp handles overshoot."""
    e = "({})".format(eta_expr)
    a_poly = ("({{e}}*{{e}}*{{e}}*{{e}} + {p0!r} + {p1!r} * {{e}} * "
                "(1.0 - {b0!r} * exp(-{b1!r} * ({{e}} + 1.0) * ({{e}} + 1.0))))").format(
                    p0=_AH_P0, p1=_AH_P1, b0=_AH_B0, b1=_AH_B1).format(e=e)
    A = "({c} * pow({a}, -0.375))".format(c=_AH_C, a=a_poly)
    B = "exp(-{})".format(e)
    return "(1.0 / ({} + {}))".format(A, B)


# D0 (2026-11-03): PRECOMPUTED DEVSIM simplify() results, keyed by the exact input string.
# setup_semiconductor_region's ONE expensive call is the Jacobian of the Aymerich-Humet electron
# density, `simplify(diff(N_c * F_1/2[(Potential - Phi_c0)/V_t], Potential))`: symdiff's simplify()
# of that expression took 152 s of the 152 s tests/test_devsim_smoke.py::test_devsim_equilibrium_smoke
# (devsim 2.11.2; 253-273 s under load; ~375 s per region and 310-358 s for that test on 2.10), while
# diff() alone takes < 1 ms. The result contains parameter NAMES only (N_c, Phi_c0, V_t), so it is a
# constant: the string below is DEVSIM 2.11.2's own output for that input, and handing it to
# node_model replays it in 0.5 ms. MEASURED bit-identical: on 4001 nodes spanning eta = -40..100 the
# replayed model equals the live simplify(diff()) model bit for bit, and it agrees with the
# UNSIMPLIFIED diff() to 9.3e-16 relative. A changed fit coefficient or eta string changes the key,
# so the lookup misses and DEVSIM simplifies live (correct, only slow) -- a stale derivative can never
# be reused. Gates: tests/test_equilibrium_simplify_cache.py (the @slow one re-runs the live
# simplify and requires the identical string and bit-identical values).
_SIMPLIFIED = {
    ('simplify(diff(N_c * (1.0 / ((1.329340388179137 * pow((((Potential - Phi_c0) / V_t)*((Potenti'
     'al - Phi_c0) / V_t)*((Potential - Phi_c0) / V_t)*((Potential - Phi_c0) / V_t) + 50.0 + 33.6 '
     '* ((Potential - Phi_c0) / V_t) * (1.0 - 0.68 * exp(-0.17 * (((Potential - Phi_c0) / V_t) + 1'
     '.0) * (((Potential - Phi_c0) / V_t) + 1.0)))), -0.375)) + exp(-((Potential - Phi_c0) / V_t))'
     ')), Potential))'):
    ('(-N_c * (-(4.985026455671764e-01 * ((3.360000000000000e+01 * (1 - (6.800000000000000e-01 * e'
     'xp(((-1.700000000000000e-01) * pow((1 + ((-Phi_c0 + Potential) * pow(V_t,(-1)))),2))))) * po'
     'w(V_t,(-1))) + (4 * pow((-Phi_c0 + Potential),3) * pow(V_t,(-4))) + (7.768320000000002e+00 *'
     ' (-Phi_c0 + Potential) * (1 + ((-Phi_c0 + Potential) * pow(V_t,(-1)))) * exp(((-1.7000000000'
     '00000e-01) * pow((1 + ((-Phi_c0 + Potential) * pow(V_t,(-1)))),2))) * pow(V_t,(-2)))) * pow('
     '(50 + (3.360000000000000e+01 * (-Phi_c0 + Potential) * (1 - (6.800000000000000e-01 * exp(((-'
     '1.700000000000000e-01) * pow((1 + ((-Phi_c0 + Potential) * pow(V_t,(-1)))),2))))) * pow(V_t,'
     '(-1))) + (pow((-Phi_c0 + Potential),4) * pow(V_t,(-4)))),(-1.375000000000000e+00))) - (exp(('
     '-(-Phi_c0 + Potential) * pow(V_t,(-1)))) * pow(V_t,(-1)))) * pow(((1.329340388179137e+00 * p'
     'ow((50 + (3.360000000000000e+01 * (-Phi_c0 + Potential) * (1 - (6.800000000000000e-01 * exp('
     '((-1.700000000000000e-01) * pow((1 + ((-Phi_c0 + Potential) * pow(V_t,(-1)))),2))))) * pow(V'
     '_t,(-1))) + (pow((-Phi_c0 + Potential),4) * pow(V_t,(-4)))),(-3.750000000000000e-01))) + exp'
     '((-(-Phi_c0 + Potential) * pow(V_t,(-1))))),(-2)))'),
    # v0.16: the same derivative with the Fermi-level reference parameter FERMI_REF_PARAM in eta,
    # (Potential - EF_ref_V - Phi_c0)/V_t -- DEVSIM 2.11.2's own simplify() output (298.7 s live, MEASURED
    # 2026-11-14); it is the entry above with (-Phi_c0 + Potential) -> (-EF_ref_V - Phi_c0 + Potential).
    ('simplify(diff(N_c * (1.0 / ((1.329340388179137 * pow((((Potential - EF_ref_V - Phi_c0) / V_t)*('
     '(Potential - EF_ref_V - Phi_c0) / V_t)*((Potential - EF_ref_V - Phi_c0) / V_t)*((Potential - EF'
     '_ref_V - Phi_c0) / V_t) + 50.0 + 33.6 * ((Potential - EF_ref_V - Phi_c0) / V_t) * (1.0 - 0.68 *'
     ' exp(-0.17 * (((Potential - EF_ref_V - Phi_c0) / V_t) + 1.0) * (((Potential - EF_ref_V - Phi_c0'
     ') / V_t) + 1.0)))), -0.375)) + exp(-((Potential - EF_ref_V - Phi_c0) / V_t)))), Potential))'):
    ('(-N_c * (-(4.985026455671764e-01 * ((3.360000000000000e+01 * (1 - (6.800000000000000e-01 * exp('
     '((-1.700000000000000e-01) * pow((1 + ((-EF_ref_V - Phi_c0 + Potential) * pow(V_t,(-1)))),2)))))'
     ' * pow(V_t,(-1))) + (4 * pow((-EF_ref_V - Phi_c0 + Potential),3) * pow(V_t,(-4))) + (7.76832000'
     '0000002e+00 * (-EF_ref_V - Phi_c0 + Potential) * (1 + ((-EF_ref_V - Phi_c0 + Potential) * pow(V'
     '_t,(-1)))) * exp(((-1.700000000000000e-01) * pow((1 + ((-EF_ref_V - Phi_c0 + Potential) * pow(V'
     '_t,(-1)))),2))) * pow(V_t,(-2)))) * pow((50 + (3.360000000000000e+01 * (-EF_ref_V - Phi_c0 + Po'
     'tential) * (1 - (6.800000000000000e-01 * exp(((-1.700000000000000e-01) * pow((1 + ((-EF_ref_V -'
     ' Phi_c0 + Potential) * pow(V_t,(-1)))),2))))) * pow(V_t,(-1))) + (pow((-EF_ref_V - Phi_c0 + Pot'
     'ential),4) * pow(V_t,(-4)))),(-1.375000000000000e+00))) - (exp((-(-EF_ref_V - Phi_c0 + Potentia'
     'l) * pow(V_t,(-1)))) * pow(V_t,(-1)))) * pow(((1.329340388179137e+00 * pow((50 + (3.36000000000'
     '0000e+01 * (-EF_ref_V - Phi_c0 + Potential) * (1 - (6.800000000000000e-01 * exp(((-1.7000000000'
     '00000e-01) * pow((1 + ((-EF_ref_V - Phi_c0 + Potential) * pow(V_t,(-1)))),2))))) * pow(V_t,(-1)'
     ')) + (pow((-EF_ref_V - Phi_c0 + Potential),4) * pow(V_t,(-4)))),(-3.750000000000000e-01))) + ex'
     'p((-(-EF_ref_V - Phi_c0 + Potential) * pow(V_t,(-1))))),(-2)))'),
}


def simplified(expr: str) -> str:
    """The DEVSIM equation for a `simplify(...)` expression: its precomputed DEVSIM result when
    `expr` is a key of _SIMPLIFIED (bit-identical, see above), else `expr` unchanged (DEVSIM
    simplifies it live)."""
    return _SIMPLIFIED.get(expr, expr)


def effective_dos_m3(m_dos_kg, T_K: float = T_REF):
    """Effective band density of states N = 2 (m* kB T / (2 pi hbar^2))^(3/2) [m^-3] from a DOS
    mass -- the N_c (or N_v) of n = N F_1/2(eta), the x = n/N_c of the generalized-Einstein factor.
    THE single implementation (audit X-15, 2026-11-02): it had four -- devsim_layered and devsim_3d
    each carried a byte-identical private `_effective_dos_m3` with a T_K knob no caller ever set,
    setup_phi_c0 re-typed it inline hard-pinned to T_REF, and optics.soa.qw_gain._band_dos a fourth
    in m0 units. Every caller now evaluates this one body, with the same operand order, so all four
    are bit-identical to what they computed before. Array-broadcasting in m_dos_kg / T_K."""
    return 2.0 * (m_dos_kg * KB * T_K / (2.0 * math.pi * HBAR ** 2)) ** 1.5


def invert_monotone(F, target: float, eta_min: float = -20.0, eta_max: float = 80.0,
                    tol: float = 1e-10, *, name: str = "invert_F12", closure: str = "F_1/2") -> float:
    """eta with F(eta) = target for an increasing closure F (n/N_c of a band), by bisection to rtol `tol`:
    THE inversion kernel of invert_F12 (the parabolic closure) and kane_band.invert_kane_closure (v0.16;
    the loop and the checks are the former body of invert_F12, so invert_F12 is unchanged bit for bit).
    `name` / `closure` only label the messages."""
    if target <= 0:
        raise ValueError("target must be positive, got {}".format(target))
    # Anti-silent-failure: bisection assumes the root lies in [eta_min, eta_max]. If the target
    # is outside [F_1/2(eta_min), F_1/2(eta_max)] the loop drives the bracket to an endpoint and
    # would otherwise return that endpoint with NO signal -- a silently-miscalibrated Phi_c0 (e.g.
    # n_bg/N_c > F_1/2(80) ~ 538 returns eta~80, ~50% wrong). Check the bracket up front, and
    # RAISE on non-convergence rather than returning the midpoint.
    f_lo, f_hi = F(eta_min), F(eta_max)
    if target > f_hi:
        raise ValueError(
            "{}: target n/N_c={:.4e} exceeds {}(eta_max={:.0f})={:.4e}; the root is "
            "outside the solver bracket -- the material is more degenerate than the eta range. "
            "Raise eta_max (and verify n_bg/N_c is physical).".format(name, target, closure, eta_max, f_hi))
    if target < f_lo:
        raise ValueError(
            "{}: target n/N_c={:.4e} is below {}(eta_min={:.0f})={:.4e}; the root is "
            "below the solver bracket (extreme non-degeneracy). Lower eta_min.".format(
                name, target, closure, eta_min, f_lo))
    for _ in range(200):
        eta_mid = 0.5 * (eta_min + eta_max)
        f_mid = F(eta_mid)
        if abs(f_mid / target - 1.0) < tol:
            return eta_mid
        if f_mid < target:
            eta_min = eta_mid
        else:
            eta_max = eta_mid
    raise RuntimeError(
        "{} did not converge to rtol={:.1e} in 200 bisection steps (target={:.4e}, "
        "final bracket [{:.4f}, {:.4f}]).".format(name, tol, target, eta_min, eta_max))


def invert_F12(target: float, eta_min: float = -20.0, eta_max: float = 80.0,
                tol: float = 1e-10) -> float:
    """eta with F12_aymerich_humet(eta) = target (bisection, rtol `tol`; invert_monotone)."""
    return invert_monotone(F12_aymerich_humet, target, eta_min, eta_max, tol)


def kane_band_of(band):
    """None for the parabolic closure (band None or "parabolic"), else the Kane band object (anything with
    m0_kg and alpha_per_eV: materials.KaneBand; duck-typed so this module does not import materials)."""
    if band is None or (isinstance(band, str) and band == "parabolic"):
        return None
    if hasattr(band, "m0_kg") and hasattr(band, "alpha_per_eV"):
        return band
    raise TypeError("band must be 'parabolic' (or None) or a materials.KaneBand, got {!r}".format(band))


def closure_kwargs(transport, dos_mass_kg: float) -> dict:
    """The band keywords of setup_semiconductor_region for a TransportModel: {"dos_mass_kg": dos_mass_kg}
    for the parabolic closure (every v0.15.1 call), {"band": transport.band} for a Kane band (the band
    sets N_c from its band-edge mass and the closure). Shared by the 2-D and 3-D builders."""
    kb = kane_band_of(getattr(transport, "band", None))
    return {"dos_mass_kg": dos_mass_kg} if kb is None else {"band": kb}


def setup_phi_c0(device: str, region: str, n_bg_m3: float,
                  dos_mass_kg: float, band=None) -> float:
    """Set Phi_c0/N_c/N_D so n(V=0) = n_bg. Phi_c0 = -eta_bg * V_t. v0.16: with a Kane `band` (alpha > 0)
    N_c is the band-edge mass's and eta_bg inverts the Kane closure (kane_band.invert_kane_closure);
    dos_mass_kg must then be the band's m0_kg."""
    kb = kane_band_of(band)
    N_c = effective_dos_m3(dos_mass_kg, T_REF)            # audit X-15: the single N_c
    if kb is not None and float(kb.alpha_per_eV) != 0.0:
        from dynameta.carriers import kane_band as _KB
        eta_bg = _KB.invert_kane_closure(n_bg_m3 / N_c, _KB.kane_b(kb.alpha_per_eV, T_REF))
    else:
        eta_bg = invert_F12(n_bg_m3 / N_c)
    Phi_c0 = -eta_bg * V_T
    ds.set_parameter(device=device, region=region, name="Phi_c0", value=Phi_c0)
    ds.set_parameter(device=device, region=region, name="N_c", value=N_c)
    ds.set_parameter(device=device, region=region, name="N_D", value=n_bg_m3)
    return Phi_c0


def _poisson_edge_models(device: str, region: str) -> None:
    ds.edge_from_node_model(device=device, region=region, node_model="Potential")
    e_ef = "(Potential@n0 - Potential@n1) * EdgeInverseLength"
    ds.edge_model(device=device, region=region, name="ElectricField", equation=e_ef)
    ds.edge_model(device=device, region=region, name="ElectricField:Potential@n0",
                    equation="simplify(diff({}, Potential@n0))".format(e_ef))
    ds.edge_model(device=device, region=region, name="ElectricField:Potential@n1",
                    equation="simplify(diff({}, Potential@n1))".format(e_ef))
    e_flux = "Permittivity * ElectricField"
    ds.edge_model(device=device, region=region, name="PotentialEdgeFlux", equation=e_flux)
    ds.edge_model(device=device, region=region, name="PotentialEdgeFlux:Potential@n0",
                    equation="simplify(diff({}, Potential@n0))".format(e_flux))
    ds.edge_model(device=device, region=region, name="PotentialEdgeFlux:Potential@n1",
                    equation="simplify(diff({}, Potential@n1))".format(e_flux))


def setup_dielectric_region(device: str, region: str, eps_static: float) -> None:
    require_positive(eps_static=eps_static)
    _ETA_BY_REGION.pop((device, region), None)      # a rebuilt device may reuse a region name
    ds.set_parameter(device=device, region=region, name="Permittivity",
                       value=eps_static * EPS0)
    ds.set_parameter(device=device, region=region, name="ElectronCharge", value=Q_E)
    ds.node_solution(device=device, region=region, name="Potential")
    n_nodes = len(ds.get_node_model_values(device=device, region=region, name="x"))
    ds.set_node_values(device=device, region=region, name="Potential",
                          values=[0.0] * n_nodes)
    _poisson_edge_models(device, region)
    _R.record_region_equation(device, region, name="PotentialEquation",
                  variable_name="Potential", edge_model="PotentialEdgeFlux",
                  variable_update="default")


def _check_param_name(name: str, owner: str) -> str:
    """A DEVSIM parameter name usable inside a model expression: an identifier that is not one of the
    region's own symbols (a clash would silently change the meaning of the expression)."""
    if not isinstance(name, str) or not _IDENT.match(name) or name in _RESERVED_SYMBOLS:
        raise ValueError("{}: {!r} is not a usable parameter name (an identifier other than {})".format(
            owner, name, sorted(_RESERVED_SYMBOLS)))
    return name


def equilibrium_eta_expr(device: str, region: str) -> str:
    """The DEVSIM expression of eta = (E_F - E_c)/kT that the equilibrium region's electron closure
    n = N_c F_1/2(eta) uses -- "(Potential - Phi_c0) / V_t", or with a Fermi-level reference parameter
    "(Potential - <param> - Phi_c0) / V_t" (setup_semiconductor_region). Consumers that need the same
    Fermi level (the equilibrium interface traps, physics_traps.FilmInterfaceTraps) build on this string,
    so the two can never disagree. Raises for a region that setup_semiconductor_region did not set up."""
    try:
        return _ETA_BY_REGION[(device, region)]
    except KeyError:
        raise ValueError("region {!r} of device {!r} is not an equilibrium semiconductor region "
                         "(physics_equilibrium.setup_semiconductor_region)".format(region, device)) from None


def setup_semiconductor_region(device: str, region: str, *,
                                 n_bg_m3: float, eps_static: float,
                                 dos_mass_kg: Optional[float] = None, fermi_ref_param: Optional[str] = None,
                                 band=None, fermi_ref_node: bool = False) -> None:
    """Single-variable nonlinear Poisson with Fermi-Dirac electron density.

    Band (v0.16): give EXACTLY ONE of
      dos_mass_kg -- the parabolic closure n = N_c F_1/2(eta), N_c from this DOS mass (every v0.15.1 call);
      band        -- a materials.KaneBand(m0_kg, alpha_per_eV): the Kane closure n = N_c(m0) [F_1/2(eta) +
                     Delta(eta, alpha kT)] (carriers.kane_band), with Phi_c0 calibrated on it (n = n_bg at
                     psi = 0). The increment is the node model KaneDelta (+ KaneDelta:Potential, DEVSIM's
                     unsimplified diff -- exact and fast); Electrons = N_c * (F_1/2 + KaneDelta) and
                     Electrons:Potential = the precomputed F_1/2 derivative + N_c * KaneDelta:Potential.
                     alpha = 0 makes every model string and value character-for-character the parabolic
                     closure with dos_mass_kg = m0_kg ("parabolic" / None means no band).

    fermi_ref_param: None (default) -- the region's Fermi level is the potential reference 0, and every
      model string is character-for-character the v0.15.1 one. A parameter NAME (FERMI_REF_PARAM =
      "EF_ref_V" is the builders' spelling) -- the region parameter of that name is created at 0.0 and
      the Fermi level becomes E_F = -q <param>: eta = (Potential - <param> - Phi_c0)/V_t. Setting the
      parameter to V is exactly what an equilibrium (no-current) contact at voltage V does to the whole
      region, without pinning the potential anywhere: the solution at (V, every Dirichlet potential + V)
      is the V = 0 solution with every potential shifted by V. (Derivation: eta = (E_F - E_c)/kT with
      E_c = q (Phi_c0 - psi) and E_F = -q V, so eta = (psi - V - Phi_c0)/V_t.) Phi_c0 still calibrates
      n = n_bg at psi = V.

    fermi_ref_node (v0.16, gate leakage; needs fermi_ref_param): False (default) -- the reference is the region
      PARAMETER above. True -- the same name is created as a NODE SOLUTION (initialised to 0.0) instead, so the
      Fermi level can vary from node to node, E_F(r) = -q <name>(r); no parameter of that name is created. Every
      model string is the parameter case's character for character (the closure, its precomputed derivative,
      equilibrium_eta_expr), so a node solution equal everywhere to the parameter value gives bit-identical
      Electrons and Electrons:Potential. The node solution needs its own equation and the Poisson derivative
      with respect to it: carriers.gate_leakage.setup_fermi_field_region installs both (a node solution
      without an equation is not solved)."""
    kb = kane_band_of(band)
    if (kb is None) == (dos_mass_kg is None):
        raise ValueError("setup_semiconductor_region: give exactly one of dos_mass_kg (parabolic closure) and "
                         "band (a KaneBand); got dos_mass_kg={!r}, band={!r}".format(dos_mass_kg, band))
    if kb is not None:
        dos_mass_kg = kb.m0_kg
        if not (math.isfinite(float(kb.alpha_per_eV)) and float(kb.alpha_per_eV) >= 0.0):
            raise ValueError("KaneBand.alpha_per_eV must be finite and >= 0, got {!r}".format(kb.alpha_per_eV))
        if float(kb.alpha_per_eV) == 0.0:
            kb = None                                         # alpha = 0: the parabolic closure, verbatim
    require_positive(n_bg_m3=n_bg_m3, eps_static=eps_static, dos_mass_kg=dos_mass_kg)
    if fermi_ref_param is not None:
        _check_param_name(fermi_ref_param, "setup_semiconductor_region(fermi_ref_param=)")
    elif fermi_ref_node:
        raise ValueError("setup_semiconductor_region: fermi_ref_node=True needs fermi_ref_param (the name of the "
                         "Fermi-level node solution)")
    setup_phi_c0(device, region, n_bg_m3, dos_mass_kg, band=kb)
    ds.set_parameter(device=device, region=region, name="Permittivity",
                       value=eps_static * EPS0)
    ds.set_parameter(device=device, region=region, name="ElectronCharge", value=Q_E)
    ds.set_parameter(device=device, region=region, name="V_t", value=V_T)
    if fermi_ref_param is not None and not fermi_ref_node:
        ds.set_parameter(device=device, region=region, name=fermi_ref_param, value=0.0)
    ds.node_solution(device=device, region=region, name="Potential")
    ds.edge_from_node_model(device=device, region=region, node_model="Potential")
    n_nodes = len(ds.get_node_model_values(device=device, region=region, name="x"))
    ds.set_node_values(device=device, region=region, name="Potential",
                          values=[0.0] * n_nodes)
    if fermi_ref_node:                                        # v0.16 (gate leakage): E_F as a node solution
        ds.node_solution(device=device, region=region, name=fermi_ref_param)
        ds.set_node_values(device=device, region=region, name=fermi_ref_param, values=[0.0] * n_nodes)
    eta = ("(Potential - Phi_c0) / V_t" if fermi_ref_param is None
           else "(Potential - {} - Phi_c0) / V_t".format(fermi_ref_param))
    _ETA_BY_REGION[(device, region)] = eta
    elec_eq = "N_c * " + F12_aymerich_humet_expr(eta)
    if kb is None:
        ds.node_model(device=device, region=region, name="Electrons", equation=elec_eq)
        ds.node_model(device=device, region=region, name="Electrons:Potential",      # D0: precomputed
                        equation=simplified("simplify(diff({}, Potential))".format(elec_eq)))
    else:                                                     # v0.16: the Kane closure
        from dynameta.carriers import kane_band as _KB
        dm = _KB.KANE_INCREMENT_MODEL
        d_eq = _KB.kane_increment_expr(eta, _KB.kane_b(kb.alpha_per_eV, T_REF))
        ds.node_model(device=device, region=region, name=dm, equation=d_eq)
        ds.node_model(device=device, region=region, name=dm + ":Potential",
                        equation="diff({}, Potential)".format(d_eq))
        ds.node_model(device=device, region=region, name="Electrons",
                        equation="N_c * ({} + {})".format(F12_aymerich_humet_expr(eta), dm))
        ds.node_model(device=device, region=region, name="Electrons:Potential",
                        equation="({}) + N_c * {}:Potential".format(
                            simplified("simplify(diff({}, Potential))".format(elec_eq)), dm))
    net_eq = _NET_CHARGE
    ds.node_model(device=device, region=region, name="PotentialNodeCharge", equation=net_eq)
    ds.node_model(device=device, region=region, name="PotentialNodeCharge:Potential",
                    equation="simplify(diff({}, Potential))".format(net_eq))
    _poisson_edge_models(device, region)
    _R.record_region_equation(device, region, name="PotentialEquation",
                  variable_name="Potential", node_model="PotentialNodeCharge",
                  edge_model="PotentialEdgeFlux", variable_update="log_damp")


def setup_interface(device: str, interface: str) -> None:
    """Potential continuity across a region-region interface."""
    ds.interface_model(device=device, interface=interface, name="continuousPotential",
                          equation="Potential@r0 - Potential@r1")
    ds.interface_model(device=device, interface=interface,
                          name="continuousPotential:Potential@r0", equation="1.0")
    ds.interface_model(device=device, interface=interface,
                          name="continuousPotential:Potential@r1", equation="-1.0")
    _R.record_interface_equation(device, interface, name="PotentialEquation",
                              interface_model="continuousPotential", type="continuous")


def psi_offset_term(psi_offset_V: float) -> str:
    """The contact-Dirichlet term for a heterostack frame shift s(r) (physics_heterojunction): an ohmic
    contact on region r pins Potential = bias + s(r). Returns "" for s == 0 so every contact string of a
    single-semiconductor / same-frame device is CHARACTER-FOR-CHARACTER what it was (v0.14.0, D1a)."""
    v = float(psi_offset_V)
    if v != v or v in (float("inf"), float("-inf")):
        raise ValueError("psi_offset_V must be finite, got {!r}".format(psi_offset_V))
    return "" if v == 0.0 else " - ({!r})".format(v)


def setup_contact(device: str, contact: str, *, psi_offset_V: float = 0.0) -> None:
    """Dirichlet Potential = <contact>_bias (+ psi_offset_V, the heterostack frame shift of the contact's
    region), with UNIQUE per-contact node-model name + edge_charge_model (closes the displacement flux
    at contact/interface corner nodes -- the empty-row fix)."""
    ds.set_parameter(device=device, name="{}_bias".format(contact), value=0.0)
    cn = "{}_potential_dirichlet".format(contact)
    ds.contact_node_model(device=device, contact=contact, name=cn,
                              equation="Potential - {}_bias".format(contact) + psi_offset_term(psi_offset_V))
    ds.contact_node_model(device=device, contact=contact, name="{}:Potential".format(cn),
                              equation="1")
    _R.record_contact_equation(device, contact, name="PotentialEquation",
                            node_model=cn, edge_charge_model="PotentialEdgeFlux")


# ------------------------------------------------------------------------------------------------
# v0.16: fixed charges and gate work functions for equilibrium devices
# ------------------------------------------------------------------------------------------------
# SIGN CONVENTION. Every region's Poisson row is  sum_edges PotentialEdgeFlux + NodeVolume *
# PotentialNodeCharge = 0  with PotentialEdgeFlux = eps (psi_0 - psi_1)/L the OUTWARD displacement flux,
# so the box form of Gauss's law (outward flux = enclosed charge) makes PotentialNodeCharge = -rho:
# q (n - N_D) here, -q (p - n + N) in the bipolar model. A charge term is a POSITIVE charge density in
# units of +q [m^-3] (the bipolar add_charge_term convention), so the composite is
#     PotentialNodeCharge = <base> - ElectronCharge * (QTerm_a + QTerm_b + ...),
# base = q (n - N_D) on an equilibrium or unipolar drift-diffusion region and absent (0) on a dielectric.
# A sheet charge Q [C/m^2] at an interface is the term  w Q/q  with w = A_node/NodeVolume [1/m] the
# interface area each node owns over its control volume (physics_traps.interface_node_area -- 1 in 1-D,
# half of each interface edge in 2-D), so the node residual gains exactly -A_node Q: the box-method
# Gauss law of a surface charge. The interface's `continuous` Potential equation sums the two region
# rows at every interface node, so the result does not depend on which side carries the term.
_QTERM = "QTerm_"


def _poisson_region_kind(device: str, region: str) -> str:
    """'dielectric' | 'equilibrium' | 'drift_diffusion' | 'bipolar' for a region with a recorded
    PotentialEquation; raises for anything else (a metal, an unset region)."""
    rec = _R.recorded(device, "PotentialEquation", scope="region", loc=region)
    if len(rec) != 1:
        raise ValueError("region {!r} of device {!r} has no (or more than one) recorded PotentialEquation: "
                         "a charge can be added only to a dielectric or semiconductor region whose physics "
                         "is set up".format(region, device))
    if _R.recorded(device, "HoleContinuityEquation", scope="region", loc=region):
        return "bipolar"
    if _R.recorded(device, "ElectronContinuityEquation", scope="region", loc=region):
        return "drift_diffusion"
    if "Electrons" in ds.get_node_model_list(device=device, region=region):
        return "equilibrium"
    return "dielectric"


def _check_base_charge(device: str, region: str, kind: str) -> None:
    """Before the FIRST charge term replaces PotentialNodeCharge, confirm the live model is the one this
    module knows (values bit-identical to _NET_CHARGE; a dielectric must have none) -- a composite built
    on a different base would silently drop physics another module installed."""
    models = ds.get_node_model_list(device=device, region=region)
    if kind == "dielectric":
        rec = _R.recorded(device, "PotentialEquation", scope="region", loc=region)[0]["kwargs"]
        if rec.get("node_model") or "PotentialNodeCharge" in models:
            raise RuntimeError("dielectric region {!r} already carries a Poisson node charge this module did "
                               "not build; refusing to replace it".format(region))
        return
    probe = "_PotentialNodeChargeBaseCheck"
    ds.node_model(device=device, region=region, name=probe, equation=_NET_CHARGE)
    try:
        same = (list(ds.get_node_model_values(device=device, region=region, name=probe))
                == list(ds.get_node_model_values(device=device, region=region, name="PotentialNodeCharge")))
    finally:
        ds.delete_node_model(device=device, region=region, name=probe)
    if not same:
        raise RuntimeError("region {!r}: its PotentialNodeCharge is not {!r} (another physics module "
                           "extended it); refusing to rebuild it with charge terms".format(region, _NET_CHARGE))


def _rebuild_potential_node_charge(device: str, region: str, kind: str) -> None:
    terms = sorted(m for m in ds.get_node_model_list(device=device, region=region)
                   if m.startswith(_QTERM) and ":" not in m)
    s = " + ".join(terms)
    d_s = " + ".join(t + ":Potential" for t in terms)
    if kind == "dielectric":
        pnc = "-ElectronCharge * ({})".format(s)
        deriv = {"Potential": "-ElectronCharge * ({})".format(d_s)}
    elif kind == "equilibrium":
        pnc = "{} - ElectronCharge * ({})".format(_NET_CHARGE, s)
        deriv = {"Potential": "ElectronCharge * Electrons:Potential - ElectronCharge * ({})".format(d_s)}
    else:                                                    # unipolar drift-diffusion: n is a variable
        pnc = "{} - ElectronCharge * ({})".format(_NET_CHARGE, s)
        deriv = {"Electrons": "ElectronCharge", "Potential": "-ElectronCharge * ({})".format(d_s)}
    ds.node_model(device=device, region=region, name="PotentialNodeCharge", equation=pnc)
    for var, eq in deriv.items():
        ds.node_model(device=device, region=region, name="PotentialNodeCharge:" + var, equation=eq)
    rec = _R.recorded(device, "PotentialEquation", scope="region", loc=region)[0]["kwargs"]
    if not rec.get("node_model"):                            # a dielectric's first charge: wire the slot
        _R.amend_region_equation(device, region, "PotentialEquation", node_model="PotentialNodeCharge")


def add_potential_charge_term(device: str, region: str, name: str, density_expr: str, *,
                              d_density_d_potential: str = "0") -> str:
    """Add a positive space-charge density `density_expr` [m^-3, units of +q; a DEVSIM node expression]
    to `region`'s Poisson equation as node model QTerm_<name> (+ QTerm_<name>:Potential =
    `d_density_d_potential`, which MUST be the exact derivative -- DEVSIM never differentiates it here, so
    a Potential-dependent term keeps Newton quadratic only through it). Works on a dielectric, an
    equilibrium or a unipolar drift-diffusion region (PotentialNodeCharge rebuilt as described above;
    a dielectric's PotentialEquation gains the node model), and on a bipolar region through
    physics_bipolar_dd.add_charge_term (Potential-independent terms only there). A device that never
    calls this keeps every model string unchanged. Re-adding a name replaces that term. Returns the
    model name."""
    if not isinstance(name, str) or not _IDENT.match(name):
        raise ValueError("add_potential_charge_term: name must be an identifier, got {!r}".format(name))
    kind = _poisson_region_kind(device, region)
    if kind == "bipolar":
        if d_density_d_potential.strip() != "0":
            raise NotImplementedError(
                "add_potential_charge_term: a Potential-dependent charge on the bipolar region {!r}; its "
                "Poisson composite is differentiated with respect to Electrons/Holes only (use "
                "physics_traps.InterfaceTraps for bias-dependent interface charge there)".format(region))
        from dynameta.carriers import physics_bipolar_dd as _BP   # lazy: BP imports this module
        return _BP.add_charge_term(device, region, name, density_expr,
                                   derivs={"Electrons": "0", "Holes": "0"})
    models = ds.get_node_model_list(device=device, region=region)
    if not any(m.startswith(_QTERM) and ":" not in m for m in models):
        _check_base_charge(device, region, kind)
    model = _QTERM + name
    ds.node_model(device=device, region=region, name=model, equation=density_expr)
    ds.node_model(device=device, region=region, name=model + ":Potential", equation=d_density_d_potential)
    if kind == "drift_diffusion":
        ds.node_model(device=device, region=region, name=model + ":Electrons", equation="0")
    _rebuild_potential_node_charge(device, region, kind)
    return model


def _finite(value, what: str) -> float:
    v = float(value)
    if not math.isfinite(v):
        raise ValueError("{} must be finite, got {!r}".format(what, value))
    return v


def install_sheet_charge(device: str, interface: str, region: str, Q_C_m2: float) -> str:
    """A fixed sheet charge Q_C_m2 [C/m^2, signed] at `interface`, carried by `region` (one of the two
    regions of the interface; any dielectric or semiconductor region -- the result does not depend on the
    side, see the sign-convention block above). Independent of carriers: no trap occupancy, no bipolar
    requirement. The interface weight is stored as node solution SheetArea<k> (A_node/NodeVolume, 0 off
    the interface) and the term is QTerm_sheet<k> = SheetArea<k> * Q/q. Returns the term's model name.

    Effect (1-D MOS, oxide capacitance C_ox): at fixed semiconductor charge the gate voltage moves by
    -Q/C_ox (tests/test_equilibrium_charges.py)."""
    q = _finite(Q_C_m2, "install_sheet_charge: Q_C_m2")
    if interface not in list(ds.get_interface_list(device=device)):
        raise ValueError("install_sheet_charge: device {!r} has no interface {!r}".format(device, interface))
    regions = list(ds.get_region_list(device=device, interface=interface))
    if region not in regions:
        raise ValueError("install_sheet_charge: region {!r} is not on interface {!r} ({})".format(
            region, interface, regions))
    _poisson_region_kind(device, region)                     # refuse a metal / unset region up front
    weight, _area = interface_weight(device, region, interface)
    k = sum(1 for m in ds.get_node_model_list(device=device, region=region) if m.startswith("SheetArea"))
    a_name = "SheetArea{}".format(k)
    ds.node_solution(device=device, region=region, name=a_name)
    ds.set_node_values(device=device, region=region, name=a_name, values=weight)
    return add_potential_charge_term(device, region, "sheet{}".format(k),
                                     "{} * ({!r})".format(a_name, q / Q_E))


def interface_weight(device: str, region: str, interface: str):
    """(w, A): w = A_node/NodeVolume [1/m in 3-D, 1/m in 2-D per unit depth, 1/m in 1-D] per node of
    `region` (0 off `interface`) and A the interface area each node owns
    (physics_traps.interface_node_area). The sheet-quantity weight shared by the sheet charge and the
    equilibrium interface traps (the same expression install_interface_traps uses for ItArea)."""
    from dynameta.carriers.physics_traps import interface_node_area   # lazy (no import cycle at load)
    import numpy as _np
    area = interface_node_area(device, region, interface)
    vol = _np.asarray(ds.get_node_model_values(device=device, region=region, name="NodeVolume"))
    weight = _np.where(area > 0.0, area / _np.where(vol > 0.0, vol, 1.0), 0.0)
    return [float(v) for v in weight], area


def install_bulk_charge(device: str, region: str, rho_C_m3: float) -> str:
    """A uniform fixed space charge rho_C_m3 [C/m^3, signed] in `region` (typically oxide bulk charge),
    as the constant term QTerm_bulk<k> = rho/q. Returns the term's model name.

    Effect (1-D MOS, uniform rho in an oxide of thickness t_ox): the gate voltage at fixed semiconductor
    charge moves by -rho t_ox^2 / (2 eps_ox) (the charge -- per area rho t_ox -- acts at its centroid,
    t_ox/2 from the gate)."""
    r = _finite(rho_C_m3, "install_bulk_charge: rho_C_m3")
    _poisson_region_kind(device, region)
    k = sum(1 for m in ds.get_node_model_list(device=device, region=region)
            if m.startswith(_QTERM + "bulk") and ":" not in m)
    return add_potential_charge_term(device, region, "bulk{}".format(k), "({!r})".format(r / Q_E))


def work_function_offset_V(work_function_eV: float, chi_eV: Optional[float], ec_minus_ef_eV: float) -> float:
    """The gate-contact potential offset of a metal of work function phi_m: -(phi_m - W_s)/q [V], with
    W_s = chi + (E_c - E_F) [eV] the semiconductor work function of the device's frame reference (its
    electron affinity and its conduction-band edge above the Fermi level at psi = 0 -- Phi_c0 of an
    equilibrium region, Phi_c of the builders' Anderson frame; negative for a degenerate film).

    Derivation: the vacuum level is continuous, E_vac = W_s - q psi everywhere in the device's frame
    (psi = 0, E_F = 0 at the reference: E_vac - E_F = chi + E_c - E_F). At the gate, E_vac = E_F,metal +
    phi_m = -q V_bias + phi_m, so psi_gate = V_bias - (phi_m - W_s)/q. The gate contact therefore pins
    Potential = bias + this offset (+ any frame offset of its region). A larger phi_m acts like a more
    negative bias (electrons leave the semiconductor); the flat-band voltage is (phi_m - W_s)/q. The
    relation does not involve the semiconductor's Fermi level, so it holds unchanged with a Fermi-level
    reference parameter (fermi_ref_param) at any value.

    For a result bit-identical to "the same device biased by V - (phi_m - W_s)/q", add the offset to
    the bias VALUE (bias + offset in Python) rather than writing it into the contact expression: x + (-d)
    and x - d are the same IEEE operation, while a separate term in the DEVSIM string rounds differently
    (agreement to the solver tolerance only)."""
    if chi_eV is None:
        raise ValueError("work_function_offset_V: the semiconductor's electron affinity chi_eV is None. A gate "
                         "work function needs W_s = chi + (E_c - E_F); set TransportModel.chi_eV (eV) -- it "
                         "has no default so a band alignment is never set by a default.")
    phi_m = _finite(work_function_eV, "work_function_eV")
    if not phi_m > 0.0:
        raise ValueError("work_function_eV must be > 0, got {!r}".format(work_function_eV))
    w_s = _finite(chi_eV, "chi_eV") + _finite(ec_minus_ef_eV, "ec_minus_ef_eV")
    return -(phi_m - w_s)
