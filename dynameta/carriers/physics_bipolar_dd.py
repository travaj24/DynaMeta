"""
Bipolar drift-diffusion (electrons + holes + Shockley-Read-Hall recombination),
DEVSIM-native, in SI units. Opt-in extension of the electrons-only
physics_drift_diffusion.py -- mirrors its SI conventions and the FERMI-DIRAC
diffusion-enhancement (FD g-factor), and reimplements the holes / SRH / bipolar
Poisson / charge-neutral contacts in SI (the bundled simple_physics.py is CGS).

Solution variables: Potential, Electrons, Holes (three coupled fields).

DEVSIM sign convention (residual = q*divergence form), per docs/implementation_notes.md:
  Jn = +q mu_n EdgeInvLength V_t [ Electrons@n1 B(vdiff) + Electrons@n1 vdiff
                                   - Electrons@n0 B(vdiff) ]            (electrons)
  Jp = -q mu_p EdgeInvLength V_t [ Holes@n1 B(vdiff) - Holes@n0 B(vdiff)
                                   - Holes@n0 vdiff ]                   (holes; q->-q
                                   AND the vdiff drift term on the @n0 node)
  Poisson node charge : PotentialNodeCharge = -q (Holes - Electrons + NetDoping)
  SRH source          : USRH = (n p - n_i^2)/(taup (n + n1) + taun (p + p1))
                        ElectronGeneration = -q USRH  (into ElectronContinuityEquation)
                        HoleGeneration     = +q USRH  (into HoleContinuityEquation)

FD enhancement (accurate rational fit of the generalized-Einstein ratio g=F_1/2/F_-1/2
in pow()s of the solution variable -- the only form DEVSIM differentiates without hanging;
see physics_drift_diffusion.py for the fit + coefficients):
  g(x) = 1 + (a x + c x^(4/3))/(1 + b x^(1/3) + d x^(2/3)),  x = c/N_dos
  replace vdiff/V_t by (vdiff/V_t)/g and scale the current by g.
For a non-degenerate Si diode (c/N_dos << 1) g -> 1 and the current reduces EXACTLY to
standard Boltzmann Scharfetter-Gummel; for ITO it is the FD softening (~1.1% peak, <0.5%
over eta>=10, valid to eta~32; vs the old degenerate-asymptote form's 6-35% error; audit F1,
bounds re-measured DD-1/DD-2). NOTE: holes use this SAME N_dos (~N_c); a degenerate p-type
material would need a separate valence-band N_v for the hole g-factor (audit DD-5).

Charge-neutral ohmic contact (recipe): pin
  n0 = 1/2 ( NetDoping + sqrt(NetDoping^2 + 4 n_i^2) )   (majority on n-side),
  p0 = n_i^2 / n0,  with the symmetric swap on the p-side, and the Potential
contact carries the built-in offset + ifelse(N>0, -V_t log(n0/n_i), V_t log(p0/n_i)).

Staged solve: see solve_bipolar_diode in validation/bipolar_diode.py:
  (1) potential-only pre-solve (freeze carriers); (2) seed Electrons/Holes from the
  Boltzmann equilibrium node models; (3) coupled 3-variable Newton; (4) bias ramp.
  variable_update: Potential "log_damp", Electrons/Holes "positive".
"""

from __future__ import annotations

import devsim as ds

from dynameta.carriers.physics_equilibrium import (
    Q_E, EPS0, V_T, _poisson_edge_models)
from dynameta.carriers import eq_registry as _R
from dynameta.carriers.eq_registry import (edge_with_derivs as _edge_with_derivs,
                                           node_with_derivs as _node_with_derivs)
from dynameta.carriers.einstein import g_expr_devsim


def _g_expr(var: str, s: str) -> str:
    """g(var{s}/N_dos) as a DEVSIM edge expression -- the rational fit + its coefficients live
    in carriers/einstein.g_expr_devsim (the single source, shared with physics_drift_diffusion;
    same coefficients for electrons and holes)."""
    return g_expr_devsim(var, "N_dos", s)


def setup_bipolar_region(device: str, region: str, *,
                         eps_static: float, n_dos_m3: float, n_i_m3: float,
                         mobility_n_m2Vs: float, mobility_p_m2Vs: float,
                         tau_n_s: float, tau_p_s: float,
                         fd_enhancement: bool = True) -> None:
    """Attach bipolar drift-diffusion physics to a semiconductor region.

    Parameters (all SI):
      eps_static       : static relative permittivity (dimensionless)
      n_dos_m3         : effective DOS used by the FD g-factor (~N_c, m^-3). For a
                         non-degenerate diode the exact value is irrelevant (g~1).
      n_i_m3           : intrinsic carrier density (m^-3)
      mobility_n_m2Vs  : electron mobility (m^2/(V s))
      mobility_p_m2Vs  : hole mobility (m^2/(V s))
      tau_n_s, tau_p_s : SRH lifetimes (s)
      fd_enhancement   : if True, apply the degenerate FD g-factor to BOTH currents;
                         if False, plain Boltzmann Scharfetter-Gummel (g==1).

    The region must already carry a "NetDoping" node model (signed: +Nd donor,
    -Na acceptor). Build it before calling (the diode mesh sets it per-region).
    """
    ds.set_parameter(device=device, region=region, name="Permittivity",
                     value=eps_static * EPS0)
    ds.set_parameter(device=device, region=region, name="ElectronCharge", value=Q_E)
    ds.set_parameter(device=device, region=region, name="V_t", value=V_T)
    ds.set_parameter(device=device, region=region, name="mu_n", value=mobility_n_m2Vs)
    ds.set_parameter(device=device, region=region, name="mu_p", value=mobility_p_m2Vs)
    ds.set_parameter(device=device, region=region, name="N_dos", value=n_dos_m3)
    ds.set_parameter(device=device, region=region, name="n_i", value=n_i_m3)
    ds.set_parameter(device=device, region=region, name="taun", value=tau_n_s)
    ds.set_parameter(device=device, region=region, name="taup", value=tau_p_s)
    ds.set_parameter(device=device, region=region, name="n1", value=n_i_m3)  # SRH trap
    ds.set_parameter(device=device, region=region, name="p1", value=n_i_m3)

    # --- solution variables ---
    ds.node_solution(device=device, region=region, name="Potential")
    ds.node_solution(device=device, region=region, name="Electrons")
    ds.node_solution(device=device, region=region, name="Holes")
    ds.edge_from_node_model(device=device, region=region, node_model="Potential")
    ds.edge_from_node_model(device=device, region=region, node_model="Electrons")
    ds.edge_from_node_model(device=device, region=region, node_model="Holes")

    # --- bipolar Poisson node charge: -q (p - n + NetDoping) ---
    pne = "-ElectronCharge * kahan3(Holes, -Electrons, NetDoping)"
    _node_with_derivs(device, region, "PotentialNodeCharge", pne, ("Electrons", "Holes"))
    _poisson_edge_models(device, region)
    _R.record_region_equation(device, region, name="PotentialEquation",
                              variable_name="Potential",
                              node_model="PotentialNodeCharge",
                              edge_model="PotentialEdgeFlux",
                              variable_update="log_damp")

    # --- shared Bernoulli argument vdiff = (psi@n0 - psi@n1)/V_t ---
    _edge_with_derivs(device, region, "vdiff",
                      "(Potential@n0 - Potential@n1)/V_t", ("Potential",))

    # --- FD diffusion enhancement g(c) (accurate rational fit; see module docstring) ---
    # edge value = average of g(c/N_dos) over the edge's two nodes.
    if fd_enhancement:
        g_n = "(0.5*({} + {}))".format(_g_expr("Electrons", "@n0"), _g_expr("Electrons", "@n1"))
        g_p = "(0.5*({} + {}))".format(_g_expr("Holes", "@n0"), _g_expr("Holes", "@n1"))
        _edge_with_derivs(device, region, "g_enh", g_n, ("Electrons",))
        _edge_with_derivs(device, region, "g_enh_p", g_p, ("Holes",))
    else:
        ds.edge_model(device=device, region=region, name="g_enh", equation="1.0")
        ds.edge_model(device=device, region=region, name="g_enh_p", equation="1.0")

    # --- electron current (FD-scaled Scharfetter-Gummel) ---
    # vdiff_g = vdiff/g_enh ; Bern_g = B(vdiff_g) ; Jn scaled by g_enh.
    _edge_with_derivs(device, region, "vdiff_g", "vdiff / g_enh",
                      ("Potential", "Electrons"))
    _edge_with_derivs(device, region, "Bern_g", "B(vdiff_g)",
                      ("Potential", "Electrons"))
    jn = ("ElectronCharge*mu_n*EdgeInverseLength*V_t*g_enh*"
          "kahan3(Electrons@n1*Bern_g, Electrons@n1*vdiff_g, -Electrons@n0*Bern_g)")
    _edge_with_derivs(device, region, "ElectronCurrent", jn,
                      ("Electrons", "Potential", "Holes"))

    # --- hole current (q -> -q; vdiff drift term moved to @n0) ---
    # vdiff_gp = vdiff/g_enh_p ; Bern_gp = B(vdiff_gp) ; Jp scaled by g_enh_p.
    _edge_with_derivs(device, region, "vdiff_gp", "vdiff / g_enh_p",
                      ("Potential", "Holes"))
    _edge_with_derivs(device, region, "Bern_gp", "B(vdiff_gp)",
                      ("Potential", "Holes"))
    jp = ("-ElectronCharge*mu_p*EdgeInverseLength*V_t*g_enh_p*"
          "kahan3(Holes@n1*Bern_gp, -Holes@n0*Bern_gp, -Holes@n0*vdiff_gp)")
    _edge_with_derivs(device, region, "HoleCurrent", jp,
                      ("Holes", "Potential", "Electrons"))

    # --- SRH recombination: one node model into BOTH continuity equations ---
    usrh = "(Electrons*Holes - n_i^2)/(taup*(Electrons + n1) + taun*(Holes + p1))"
    _node_with_derivs(device, region, "USRH", usrh, ("Electrons", "Holes"))
    gn = "-ElectronCharge * USRH"
    gp = "+ElectronCharge * USRH"
    _node_with_derivs(device, region, "ElectronGeneration", gn, ("Electrons", "Holes"))
    _node_with_derivs(device, region, "HoleGeneration", gp, ("Electrons", "Holes"))

    # --- continuity equations (DC: no time_node_model) ---
    _R.record_region_equation(device, region, name="ElectronContinuityEquation",
                              variable_name="Electrons", edge_model="ElectronCurrent",
                              node_model="ElectronGeneration",
                              variable_update="positive")
    _R.record_region_equation(device, region, name="HoleContinuityEquation",
                              variable_name="Holes", edge_model="HoleCurrent",
                              node_model="HoleGeneration",
                              variable_update="positive")


# Charge-neutral equilibrium carrier node models (SI). Used both to seed the
# coupled solve and inside the contact pinning expression. The +1.0 floor keeps
# the sqrt/abs well-conditioned in fully-depleted or undoped cells.
CELEC = "(1.0 + 0.5*abs(NetDoping + (NetDoping^2 + 4 * n_i^2)^(0.5)))"
CHOLE = "(1.0 + 0.5*abs(-NetDoping + (NetDoping^2 + 4 * n_i^2)^(0.5)))"


def setup_equilibrium_seed_models(device: str, region: str) -> None:
    """Create IntrinsicElectrons/IntrinsicHoles node models (charge-neutral
    equilibrium) so the staged solve can seed Electrons/Holes before the coupled
    Newton. n0 = majority on n-side, p0 = n_i^2/n0 (symmetric swap on p-side)."""
    n0 = "ifelse(NetDoping > 0, {ce}, n_i^2/{ch})".format(ce=CELEC, ch=CHOLE)
    p0 = "ifelse(NetDoping < 0, {ch}, n_i^2/{ce})".format(ce=CELEC, ch=CHOLE)
    ds.node_model(device=device, region=region, name="IntrinsicElectrons", equation=n0)
    ds.node_model(device=device, region=region, name="IntrinsicHoles", equation=p0)


def setup_contact_ohmic_bipolar(device: str, contact: str) -> None:
    """Bipolar ohmic contact: pin Potential (with the built-in offset) and pin
    Electrons/Holes to their charge-neutral equilibrium values."""
    ds.set_parameter(device=device, name="{}_bias".format(contact), value=0.0)
    bias = "{}_bias".format(contact)

    # Potential: Dirichlet + built-in potential offset (Boltzmann reference n_i).
    cp = "{}_potential_dirichlet".format(contact)
    pot_eq = ("Potential - {b} + ifelse(NetDoping > 0, "
              "-V_t*log({ce}/n_i), V_t*log({ch}/n_i))").format(
                  b=bias, ce=CELEC, ch=CHOLE)
    ds.contact_node_model(device=device, contact=contact, name=cp, equation=pot_eq)
    ds.contact_node_model(device=device, contact=contact,
                          name="{}:Potential".format(cp), equation="1")
    _R.record_contact_equation(device, contact, name="PotentialEquation",
                               node_model=cp, edge_charge_model="PotentialEdgeFlux")

    # Electrons pinned to n0 (n-side) or n_i^2/p0 (p-side); derivative literal "1".
    ce = "{}_electrons_dirichlet".format(contact)
    elec_eq = "Electrons - ifelse(NetDoping > 0, {ce}, n_i^2/{ch})".format(
        ce=CELEC, ch=CHOLE)
    ds.contact_node_model(device=device, contact=contact, name=ce, equation=elec_eq)
    ds.contact_node_model(device=device, contact=contact,
                          name="{}:Electrons".format(ce), equation="1")
    _R.record_contact_equation(device, contact, name="ElectronContinuityEquation",
                               node_model=ce, edge_current_model="ElectronCurrent")

    # Holes pinned to p0 (p-side) or n_i^2/n0 (n-side); derivative literal "1".
    ch = "{}_holes_dirichlet".format(contact)
    hole_eq = "Holes - ifelse(NetDoping < 0, {ch}, n_i^2/{ce})".format(
        ce=CELEC, ch=CHOLE)
    ds.contact_node_model(device=device, contact=contact, name=ch, equation=hole_eq)
    ds.contact_node_model(device=device, contact=contact,
                          name="{}:Holes".format(ch), equation="1")
    _R.record_contact_equation(device, contact, name="HoleContinuityEquation",
                               node_model=ch, edge_current_model="HoleCurrent")
