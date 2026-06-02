"""
Single-carrier (electron) drift-diffusion, DEVSIM-native Scharfetter-Gummel with
a FERMI-DIRAC diffusion-enhancement for degenerate semiconductors (ITO).

Electrons is a SOLUTION VARIABLE (the only formulation DEVSIM differentiates
natively -- putting the carrier density / F_1/2 expression inside an edge model
hangs DEVSIM's expr system, whether via a node-model reference or inlined).

  Poisson:  div(eps grad psi) = q (n - N_D)
  SG (Boltzmann):  Jn = q mu EdgeInvLength V_t [n@n1 B(dpsi/V_t) - n@n0 B(-dpsi/V_t)]
  FD enhancement:  replace dpsi/V_t by (dpsi/V_t)/g and scale the current by g,
                   g(n) = generalized-Einstein ratio F_1/2(eta)/F_-1/2(eta).
  Continuity: div(Jn) = 0 (DC, no recombination).

The exact g needs eta = invF_1/2(n/N_c), which is not expressible in DEVSIM. We use a
RATIONAL FIT of g as a function of x = n/N_c in u = x^(1/3):
  g(x) = 1 + (a x + c x^(4/3)) / (1 + b x^(1/3) + d x^(2/3))
with (a,b,c,d) a least-squares fit accurate to ~1.1% PEAK (near eta~1-2) and <0.5% across
ITO's regime (eta>=10; 0.33% at eta~22, x up to ~80). The Boltzmann VALUE limit is exact
(g(0)=1); the degenerate leading coefficient is NOT anchored (the fit -> (c/d) x^(2/3) =
0.69 x^(2/3) vs the true (2/3)(c1)^(2/3) x^(2/3) = 0.81 x^(2/3), ~15% low), so the fit is
valid to ~eta=32 and undershoots beyond. This REPLACES the old degenerate-asymptote
1+(2/3)(c1 x)^(2/3), which was 6.6% high at ITO's operating point and 25-35% high through
the moderate-degeneracy transition (audit F1; accuracy bounds re-measured audit DD-1/DD-2). g still enters only as pow()s of the
SOLUTION variable Electrons, so the SG derivatives differentiate cleanly (putting the full
F_1/2 in an edge model instead hangs DEVSIM). At equilibrium (Jn = 0) the enhanced SG
gives n ~ exp(psi/(g V_t)), the Fermi-Dirac softening.

VALIDATED REGIME: transport / current-flow and zero bias -- the clean
[metal/ITO/metal] slab converges, and Park zero-bias gives n = n_bg. This is the
regime DD is for (currents, J-V, dynamics).

KNOWN LIMITATION: a GATED CAPACITOR (e.g. Park under gate bias) still does NOT
converge. With no DC current path the continuity equation must propagate the
carrier level across the gate-field region from only the weak 2-node ITO-edge
grounds -- ill-conditioned regardless of Boltzmann-vs-Fermi-Dirac (the g
enhancement and a relaxation regularization were both tried; neither fixed it).
For DC gate accumulation use the equilibrium physics mode -- it is the physically
AND numerically correct tool there (n is local, no continuity equation; no DC
current flows through the gate oxide). Closing gated-DD would need full-edge
(not 2-node) ITO ohmic contacts or a Gummel-style outer iteration -- a scoped
future effort.
"""

from __future__ import annotations

import devsim as ds

from dynameta.carriers.physics_equilibrium import (
    Q_E, EPS0, V_T, setup_phi_c0, _poisson_edge_models)
from dynameta.carriers import eq_registry as _R
from dynameta.carriers.einstein import GA as _GA, GB as _GB, GC as _GC, GD as _GD

# Generalized-Einstein degeneracy factor g(x)=F_1/2(eta)/F_-1/2(eta) as a rational fit in
# u=x^(1/3), x=n/N_c (~1.1% peak, <0.5% over ITO's eta>=10; exact VALUE at the Boltzmann
# limit g(0)=1; valid to eta~32). Coefficients live in carriers/einstein.py (single source,
# shared with physics_bipolar_dd, unit-tested in tests/test_carriers_gfactor.py).
_P13, _P23, _P43 = 1.0 / 3.0, 2.0 / 3.0, 4.0 / 3.0


def _g_expr(s: str) -> str:
    """g(Electrons{s}/N_c) as a DEVSIM edge expression (pow()s of the solution variable)."""
    X = "(Electrons{}/N_c)".format(s)
    return ("(1.0 + ({a}*{X} + {c}*pow({X},{p43}))/(1.0 + {b}*pow({X},{p13}) + "
            "{d}*pow({X},{p23})))").format(a=_GA, b=_GB, c=_GC, d=_GD, X=X,
                                            p13=_P13, p23=_P23, p43=_P43)


def _edge_with_derivs(device, region, name, eq, wrt):
    ds.edge_model(device=device, region=region, name=name, equation=eq)
    for w in wrt:
        for nd in ("n0", "n1"):
            ds.edge_model(device=device, region=region,
                            name="{}:{}@{}".format(name, w, nd),
                            equation="simplify(diff({}, {}@{}))".format(eq, w, nd))


def setup_semiconductor_region_dd(device: str, region: str, *,
                                    n_bg_m3: float, eps_static: float,
                                    dos_mass_kg: float, mobility_m2Vs: float) -> None:
    setup_phi_c0(device, region, n_bg_m3, dos_mass_kg)   # Phi_c0, N_c, N_D
    ds.set_parameter(device=device, region=region, name="Permittivity", value=eps_static * EPS0)
    ds.set_parameter(device=device, region=region, name="ElectronCharge", value=Q_E)
    ds.set_parameter(device=device, region=region, name="V_t", value=V_T)
    ds.set_parameter(device=device, region=region, name="mu_n", value=mobility_m2Vs)

    ds.node_solution(device=device, region=region, name="Potential")
    ds.node_solution(device=device, region=region, name="Electrons")
    ds.edge_from_node_model(device=device, region=region, node_model="Potential")
    ds.edge_from_node_model(device=device, region=region, node_model="Electrons")
    n_nodes = len(ds.get_node_model_values(device=device, region=region, name="x"))
    ds.set_node_values(device=device, region=region, name="Potential", values=[0.0] * n_nodes)
    ds.set_node_values(device=device, region=region, name="Electrons", values=[n_bg_m3] * n_nodes)

    # Poisson
    net = "ElectronCharge * (Electrons - N_D)"
    ds.node_model(device=device, region=region, name="PotentialNodeCharge", equation=net)
    ds.node_model(device=device, region=region, name="PotentialNodeCharge:Electrons",
                    equation="ElectronCharge")
    ds.node_model(device=device, region=region, name="PotentialNodeCharge:Potential",
                    equation="0")
    _poisson_edge_models(device, region)
    _R.record_region_equation(device, region, name="PotentialEquation",
                  variable_name="Potential", node_model="PotentialNodeCharge",
                  edge_model="PotentialEdgeFlux", variable_update="log_damp")

    # --- FD diffusion-enhancement g(n) (accurate rational fit; see module docstring) ---
    # edge value g_enh = average of g(Electrons/N_c) over the edge's two nodes.
    g_avg = "(0.5*({} + {}))".format(_g_expr("@n0"), _g_expr("@n1"))
    _edge_with_derivs(device, region, "g_enh", g_avg, ("Electrons",))

    # vdiff and the FD-scaled argument vdiff_g = vdiff / g_enh
    _edge_with_derivs(device, region, "vdiff",
                        "(Potential@n0 - Potential@n1)/V_t", ("Potential",))
    _edge_with_derivs(device, region, "vdiff_g", "vdiff / g_enh",
                        ("Potential", "Electrons"))
    _edge_with_derivs(device, region, "Bern_g", "B(vdiff_g)",
                        ("Potential", "Electrons"))

    # Enhanced SG current
    jn = ("ElectronCharge*mu_n*EdgeInverseLength*V_t*g_enh*"
           "kahan3(Electrons@n1*Bern_g, Electrons@n1*vdiff_g, -Electrons@n0*Bern_g)")
    _edge_with_derivs(device, region, "ElectronCurrent", jn, ("Electrons", "Potential"))

    _R.record_region_equation(device, region, name="ElectronContinuityEquation",
                  variable_name="Electrons", edge_model="ElectronCurrent",
                  variable_update="positive")


def setup_contact_ohmic_dd(device: str, contact: str) -> None:
    """Ohmic contact: pin Potential = bias, Electrons = N_D (charge-neutral n-type)."""
    ds.set_parameter(device=device, name="{}_bias".format(contact), value=0.0)
    bias = "{}_bias".format(contact)
    cp = "{}_potential_dirichlet".format(contact)
    ds.contact_node_model(device=device, contact=contact, name=cp,
                              equation="Potential - {}".format(bias))
    ds.contact_node_model(device=device, contact=contact, name="{}:Potential".format(cp),
                              equation="1")
    _R.record_contact_equation(device, contact, name="PotentialEquation",
                            node_model=cp, edge_charge_model="PotentialEdgeFlux")
    ce = "{}_electrons_dirichlet".format(contact)
    ds.contact_node_model(device=device, contact=contact, name=ce, equation="Electrons - N_D")
    ds.contact_node_model(device=device, contact=contact, name="{}:Electrons".format(ce),
                              equation="1")
    # edge_current_model is REQUIRED for get_contact_current to return the terminal
    # current (else it reads 0). The Dirichlet node_model still pins n at the contact;
    # adding the current model only enables current extraction (it does not change the
    # boundary condition). Omitting it was harmless for the MOS-cap (no current queried)
    # but made the unipolar terminal current unreadable -- exposed by a 3D resistor test.
    _R.record_contact_equation(device, contact, name="ElectronContinuityEquation",
                            node_model=ce, edge_current_model="ElectronCurrent")
