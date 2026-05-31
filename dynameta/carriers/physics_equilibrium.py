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

# Constants (SI)
Q_E   = 1.602176634e-19
EPS0  = 8.8541878128e-12
KB    = 1.380649e-23
HBAR  = 1.054571817e-34
M_E   = 9.1093837015e-31
T_REF = 300.0
V_T   = KB * T_REF / Q_E

_AH_C = 3.0 * math.sqrt(math.pi) / 4.0


def F12_aymerich_humet(eta: float) -> float:
    a = eta**4 + 50.0 + 33.6 * eta * (1.0 - 0.68 * math.exp(-0.17 * (eta + 1.0)**2))
    A = (3.0 * math.sqrt(math.pi) / 4.0) / a**(3.0 / 8.0)
    B = math.exp(-eta)
    return 1.0 / (A + B)


def F12_aymerich_humet_expr(eta_expr: str = "eta") -> str:
    """DEVSIM expression for F_1/2(eta). No sqrt() (parser lacks it); leading
    constant precomputed. Unclamped exp(-eta) -- log_damp handles overshoot."""
    e = "({})".format(eta_expr)
    a_poly = ("({e}*{e}*{e}*{e} + 50.0 + 33.6 * {e} * "
                "(1.0 - 0.68 * exp(-0.17 * ({e} + 1.0) * ({e} + 1.0))))").format(e=e)
    A = "({c} * pow({a}, -0.375))".format(c=_AH_C, a=a_poly)
    B = "exp(-{})".format(e)
    return "(1.0 / ({} + {}))".format(A, B)


def invert_F12(target: float, eta_min: float = -20.0, eta_max: float = 80.0,
                tol: float = 1e-10) -> float:
    if target <= 0:
        raise ValueError("target must be positive, got {}".format(target))
    for _ in range(200):
        eta_mid = 0.5 * (eta_min + eta_max)
        f_mid = F12_aymerich_humet(eta_mid)
        if abs(f_mid / target - 1.0) < tol:
            return eta_mid
        if f_mid < target:
            eta_min = eta_mid
        else:
            eta_max = eta_mid
    return 0.5 * (eta_min + eta_max)


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
