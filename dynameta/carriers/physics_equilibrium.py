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
"""

from __future__ import annotations

import math

import devsim as ds

from dynameta.carriers import eq_registry as _R

# Physical constants (SI): single source in core/constants. Re-exported here because several
# carrier modules historically import Q_E / V_T / EPS0 / M_E from physics_equilibrium.
from dynameta.constants import Q_E, EPS0, KB, HBAR, M_E, T_REF, V_T  # noqa: E402,F401

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


def invert_F12(target: float, eta_min: float = -20.0, eta_max: float = 80.0,
                tol: float = 1e-10) -> float:
    if target <= 0:
        raise ValueError("target must be positive, got {}".format(target))
    # Anti-silent-failure: bisection assumes the root lies in [eta_min, eta_max]. If the target
    # is outside [F_1/2(eta_min), F_1/2(eta_max)] the loop drives the bracket to an endpoint and
    # would otherwise return that endpoint with NO signal -- a silently-miscalibrated Phi_c0 (e.g.
    # n_bg/N_c > F_1/2(80) ~ 538 returns eta~80, ~50% wrong). Check the bracket up front, and
    # RAISE on non-convergence rather than returning the midpoint.
    f_lo, f_hi = F12_aymerich_humet(eta_min), F12_aymerich_humet(eta_max)
    if target > f_hi:
        raise ValueError(
            "invert_F12: target n/N_c={:.4e} exceeds F_1/2(eta_max={:.0f})={:.4e}; the root is "
            "outside the solver bracket -- the material is more degenerate than the eta range. "
            "Raise eta_max (and verify n_bg/N_c is physical).".format(target, eta_max, f_hi))
    if target < f_lo:
        raise ValueError(
            "invert_F12: target n/N_c={:.4e} is below F_1/2(eta_min={:.0f})={:.4e}; the root is "
            "below the solver bracket (extreme non-degeneracy). Lower eta_min.".format(
                target, eta_min, f_lo))
    for _ in range(200):
        eta_mid = 0.5 * (eta_min + eta_max)
        f_mid = F12_aymerich_humet(eta_mid)
        if abs(f_mid / target - 1.0) < tol:
            return eta_mid
        if f_mid < target:
            eta_min = eta_mid
        else:
            eta_max = eta_mid
    raise RuntimeError(
        "invert_F12 did not converge to rtol={:.1e} in 200 bisection steps (target={:.4e}, "
        "final bracket [{:.4f}, {:.4f}]).".format(tol, target, eta_min, eta_max))


def setup_phi_c0(device: str, region: str, n_bg_m3: float,
                  dos_mass_kg: float) -> float:
    """Set Phi_c0/N_c/N_D so n(V=0) = n_bg. Phi_c0 = -eta_bg * V_t."""
    N_c = 2.0 * (dos_mass_kg * KB * T_REF / (2.0 * math.pi * HBAR**2))**1.5
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


def setup_semiconductor_region(device: str, region: str, *,
                                 n_bg_m3: float, eps_static: float,
                                 dos_mass_kg: float) -> None:
    """Single-variable nonlinear Poisson with Fermi-Dirac electron density."""
    require_positive(n_bg_m3=n_bg_m3, eps_static=eps_static, dos_mass_kg=dos_mass_kg)
    setup_phi_c0(device, region, n_bg_m3, dos_mass_kg)
    ds.set_parameter(device=device, region=region, name="Permittivity",
                       value=eps_static * EPS0)
    ds.set_parameter(device=device, region=region, name="ElectronCharge", value=Q_E)
    ds.set_parameter(device=device, region=region, name="V_t", value=V_T)
    ds.node_solution(device=device, region=region, name="Potential")
    ds.edge_from_node_model(device=device, region=region, node_model="Potential")
    n_nodes = len(ds.get_node_model_values(device=device, region=region, name="x"))
    ds.set_node_values(device=device, region=region, name="Potential",
                          values=[0.0] * n_nodes)
    eta = "(Potential - Phi_c0) / V_t"
    elec_eq = "N_c * " + F12_aymerich_humet_expr(eta)
    ds.node_model(device=device, region=region, name="Electrons", equation=elec_eq)
    ds.node_model(device=device, region=region, name="Electrons:Potential",
                    equation="simplify(diff({}, Potential))".format(elec_eq))
    net_eq = "ElectronCharge * (Electrons - N_D)"
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


def setup_contact(device: str, contact: str) -> None:
    """Dirichlet Potential = <contact>_bias, with UNIQUE per-contact node-model
    name + edge_charge_model (closes the displacement flux at contact/interface
    corner nodes -- the empty-row fix)."""
    ds.set_parameter(device=device, name="{}_bias".format(contact), value=0.0)
    cn = "{}_potential_dirichlet".format(contact)
    ds.contact_node_model(device=device, contact=contact, name=cn,
                              equation="Potential - {}_bias".format(contact))
    ds.contact_node_model(device=device, contact=contact, name="{}:Potential".format(cn),
                              equation="1")
    _R.record_contact_equation(device, contact, name="PotentialEquation",
                            node_model=cn, edge_charge_model="PotentialEdgeFlux")
