"""
Single-carrier (electron) drift-diffusion, DEVSIM-native Scharfetter-Gummel with
a FERMI-DIRAC diffusion-enhancement for degenerate semiconductors (ITO).

Electrons is a SOLUTION VARIABLE (the only formulation DEVSIM differentiates
natively -- putting the carrier density / F_1/2 expression inside an edge model
hangs DEVSIM's expr system, whether via a node-model reference or inlined).

  Poisson:  div(eps grad psi) = q (n - N_D)
  SG (Boltzmann):  with vdiff = (psi@n0 - psi@n1)/V_t (the library's edge sign convention),
                   Jn = q mu EdgeInvLength V_t [n@n1 B(-vdiff) - n@n0 B(vdiff)], built by the ONE
                   SG kernel (sg_flux, 2026-11-04): B(+-vdiff) both, sign-selected so the identity
                   B(x) + x = B(-x) is only used for x >= 0 (the former kahan3 spelling used it for
                   every x and cancelled for vdiff << 0: 1.9e-7 at -21, 100 % at -40).
  FD enhancement:  replace vdiff by vdiff/g and scale the current by g,
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
[metal/ITO/metal] slab converges, and reference zero-bias gives n = n_bg. This is the
regime DD is for (currents, J-V, dynamics).

GATED CAPACITOR (no DC current path): CONVERGES in 1D and reduces to the equilibrium Fermi-Dirac
profile in the zero-current limit (validation/gated_dd.py: DD profile == equilibrium to ~1e-4, the
gate charge Q(Vg) / C-V matches to ~1e-4). The recipe: (1) a FULL-BOUNDARY ohmic contact pinning
Electrons = N_D (in 1D the ITO ohmic end IS the full boundary, so the continuity equation is well
anchored); (2) a relaxed RELATIVE tolerance (at the trivial bias Potential ~ 0 makes ||du||/||u||
floor at the numerical-precision ~1e-6, so a 1e-10 rel target never converges -- the ABSOLUTE error
is the real gate); (3) a staged solve -- a potential-only Poisson pre-solve (continuity frozen via
eq_registry) then coupled Newton.

2D METASURFACE (validated): the ITO ohmic ground there was a weak ~2-node edge contact (DEVSIM
captures only the box corners of a lateral contact at a domain boundary), which could not anchor
the continuity equation. It was promoted to a FULL-EDGE ohmic ground via a thin adjacent edge-metal
region (so the ground is a region-region interface with full-line node capture, like a
horizontal-face contact) -- shipped and validated in 2D (8106849, validation/gated_dd_2d.py) and
wired into the layered builder for the reference metasurface (77f92ab, validation/gated_dd_builder.py;
devsim_layered.py _dd_full_edge_grounds). 1D + 2D + builder are validated; 3D DD on the gated
stack remains the open extension.
"""

from __future__ import annotations

import math
import warnings
from typing import Optional

from dynameta.carriers.devsim_env import import_devsim

ds = import_devsim()      # D0: devsim via its ONE import home (BLAS pinned before the first import)

from dynameta.carriers.physics_equilibrium import (
    Q_E, EPS0, V_T, setup_phi_c0, _poisson_edge_models, require_positive, invert_F12,
    psi_offset_term)
from dynameta.carriers import eq_registry as _R
from dynameta.carriers import sg_flux as _SG
from dynameta.carriers.eq_registry import (edge_with_derivs as _edge_with_derivs,
                                           node_with_derivs as _node_with_derivs)
from dynameta.carriers.einstein import g_expr_devsim
from dynameta.carriers.mobility import mu_edge_expr_devsim
from dynameta.warning_types import UnreliableResultWarning


def _g_expr(s: str) -> str:
    """g(Electrons{s}/N_c) as a DEVSIM edge expression -- the rational fit + its coefficients
    live in carriers/einstein.g_expr_devsim (the single source, shared with physics_bipolar_dd
    and unit-tested in tests/test_carriers_gfactor.py)."""
    return g_expr_devsim("Electrons", "N_c", s)


def setup_semiconductor_region_dd(device: str, region: str, *,
                                    n_bg_m3: float, eps_static: float,
                                    dos_mass_kg: float, mobility_m2Vs: float,
                                    field_mobility: bool = False,
                                    v_sat_ms: Optional[float] = None,
                                    ct_beta: float = 2.0, e_smooth_Vm: float = 1.0,
                                    statistics: Optional[str] = None) -> None:
    """Unipolar (electron) drift-diffusion region with the FD g-factor transport.

    statistics (v0.15; TransportModel.carrier_statistics): which FD SG flux.
      None / "fermi_dirac" (the DEFAULT since v0.15.0, user decision 2026-11-07): the FD quasi-field (g = 1,
          Delta1 - Delta0 in the SG argument; physics_heterojunction.fd_quasi_field_expr) -- the same transport
          as the bipolar default: exact discrete equilibrium and an exactly Nyquist-consistent diffusion noise
          (MEASURED 9e-12 on an ITO-like 3e26 / AZO-like 3e25 m^-3 thermionic stack).
          docs/audit/2026-11-06-v0.15-fermi-dirac-mode-complete.md.
      "legacy" (the v0.14 default, byte-identical; NOT recommended -- reproducing v0.14 numbers only): the
          edge-averaged g, 0.5 (g(n0) + g(n1)). Its discrete zero-current state is NOT the activity
          equilibrium the contacts and a thermionic heterointerface use: an O(h^2) zero-bias battery next to a
          degenerate density gradient, and an S = 4kT Re Y residual of the same order (MEASURED: 2.1e-6 on that
          stack at h = 0.5 nm, 5.4e-7 at 0.25 nm).
      ("boltzmann" does not exist here: the unipolar model is FD by construction -- refused.)"""
    if statistics not in (None, "fermi_dirac", "legacy"):
        raise ValueError("setup_semiconductor_region_dd: statistics must be None, 'fermi_dirac' or 'legacy' (the "
                         "unipolar model is Fermi-Dirac by construction), got {!r}".format(statistics))
    require_positive(n_bg_m3=n_bg_m3, eps_static=eps_static, dos_mass_kg=dos_mass_kg,
                     mobility_m2Vs=mobility_m2Vs)
    setup_phi_c0(device, region, n_bg_m3, dos_mass_kg)   # Phi_c0, N_c, N_D
    # FD g-factor honesty guard: the rational fit is accurate to ~eta=32 and undershoots beyond
    # (module docstring). Gate ACCUMULATION drives the local eta above the background eta_bg, so a
    # high eta_bg means the solve operates near/over the fit's validity edge -- warn at setup.
    N_c = float(ds.get_parameter(device=device, region=region, name="N_c"))
    eta_bg = invert_F12(n_bg_m3 / N_c)
    if eta_bg > 28.0:
        warnings.warn(
            "setup_semiconductor_region_dd: background degeneracy eta_bg={:.1f} is near the FD "
            "g-factor fit's validity edge (~eta=32); gate accumulation pushes the local eta "
            "higher, where g undershoots (the diffusion/Einstein term grows ~1-2% inaccurate). "
            "Treat strong-accumulation DD results as approximate.".format(eta_bg), UnreliableResultWarning, stacklevel=2)
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
    fdq = ""
    if statistics != "legacy":                     # g = 1 + the FD quasi-field (fd_quasi_field_expr)
        from dynameta.carriers.physics_heterojunction import fd_quasi_field_expr
        fdq = " + " + fd_quasi_field_expr(device, region, "Electrons", "N_c")
        ds.edge_model(device=device, region=region, name="g_enh", equation="1.0")
    else:
        g_avg = "(0.5*({} + {}))".format(_g_expr("@n0"), _g_expr("@n1"))
        _edge_with_derivs(device, region, "g_enh", g_avg, ("Electrons",))

    # vdiff and the FD-scaled argument vdiff_g = vdiff / g_enh
    _edge_with_derivs(device, region, "vdiff",
                        "(Potential@n0 - Potential@n1)/V_t", ("Potential",))
    _edge_with_derivs(device, region, "vdiff_g", "vdiff / g_enh" + fdq,
                        ("Potential", "Electrons"))

    # Enhanced SG current. FIELD-DEPENDENT MOBILITY (R1): when field_mobility is on, the scalar mu_n is
    # replaced by a Caughey-Thomas velocity-saturated edge model mu_n_edge(|E_par|) (mu_n stays the
    # low-field value); when off (default) the jn string keeps the scalar `mu_n` literal so the
    # residual + Jacobian assembly are character-for-character today's constant-mobility solve.
    mu_sym = "mu_n"
    if field_mobility:
        if v_sat_ms is None or not math.isfinite(float(v_sat_ms)) or float(v_sat_ms) <= 0.0:
            raise ValueError("field_mobility=True requires a positive finite v_sat_ms (m/s)")
        ds.set_parameter(device=device, region=region, name="v_sat", value=float(v_sat_ms))
        ds.set_parameter(device=device, region=region, name="E_smooth", value=float(e_smooth_Vm))
        mu_expr = mu_edge_expr_devsim(mu_low="mu_n", efield="ElectricField", v_sat="v_sat",
                                      e_smooth="E_smooth", beta=ct_beta)
        _edge_with_derivs(device, region, "mu_n_edge", mu_expr, ("Potential",))
        mu_sym = "mu_n_edge"
    # Bern_g = B(vdiff_g), Bern_g_m = B(-vdiff_g): the ONE SG kernel (sg_flux; sign-selected two-
    # Bernoulli bracket since 2026-11-04 -- the kahan3 B(v) + v spelling lost 1.9e-7 at v = -21 and
    # 100 % at -40).
    _SG.install_sg_current(device, region, carrier="electrons", density="Electrons",
                           vdiff="vdiff_g", bern="Bern_g", bern_m="Bern_g_m",
                           current="ElectronCurrent",
                           prefactor="ElectronCharge*" + mu_sym + "*EdgeInverseLength*V_t*g_enh",
                           wrt=("Potential", "Electrons"), current_wrt=("Electrons", "Potential"))

    # transient/AC charge: the d(q n)/dt time term (NCharge = -q n, the electron charge density).
    # Defined ALWAYS -- a steady-state DC solve ignores time_node_model, while a transient
    # (solve type="transient_*") or small-signal ssac solve USES it. It is what makes the
    # carrier-charge (gate-accumulation) capacitance appear in the ssac admittance; without it ssac
    # on this unipolar DD device would see only the resistive part (mirrors the bipolar
    # NCharge/PCharge). No recombination in this model -> no node_model on the continuity equation.
    _node_with_derivs(device, region, "NCharge", "-ElectronCharge * Electrons", ("Electrons",))
    _R.record_region_equation(device, region, name="ElectronContinuityEquation",
                  variable_name="Electrons", edge_model="ElectronCurrent",
                  time_node_model="NCharge", variable_update="positive")


def setup_contact_ohmic_dd(device: str, contact: str, *, psi_offset_V: float = 0.0) -> None:
    """Ohmic contact: pin Potential = bias (+ psi_offset_V, the heterostack frame shift s(r) of the
    contact's region -- physics_heterojunction), Electrons = N_D (charge-neutral n-type)."""
    ds.set_parameter(device=device, name="{}_bias".format(contact), value=0.0)
    bias = "{}_bias".format(contact)
    cp = "{}_potential_dirichlet".format(contact)
    ds.contact_node_model(device=device, contact=contact, name=cp,
                              equation="Potential - {}".format(bias) + psi_offset_term(psi_offset_V))
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


def setup_contact_ohmic_dd_circuit(device: str, contact: str, *, node_name: str = "vac",
                                   source_name: str = "V1", v_ac: float = 1.0,
                                   drive: str = "voltage") -> tuple:
    """Circuit-driven twin of setup_contact_ohmic_dd (v0.14.0, for ac_analysis / noise_analysis on a
    UNIPOLAR DD device): Potential tied to circuit node `node_name` (no bias parameter), Electrons
    pinned to N_D, and BOTH the displacement (PotentialEdgeFlux) and the electron current
    (ElectronCurrent) coupled into that node, so the source current '<source_name>.I' is the total
    small-signal terminal current. drive="voltage" (source 'V...') or "current" (source 'I...', the
    open-circuit terminal for S_V); see physics_bipolar_dd.setup_contact_ohmic_bipolar_circuit.
    setup_contact_ohmic_dd itself is unchanged. Returns (source_name, node_name)."""
    from dynameta.carriers.physics_bipolar_dd import _check_drive
    source_name = _check_drive(drive, source_name, "setup_contact_ohmic_dd_circuit")
    if not (float(v_ac) > 0.0):
        raise ValueError("setup_contact_ohmic_dd_circuit: v_ac must be > 0; got {!r}".format(v_ac))
    ds.add_circuit_node(name=node_name, variable_update="default")
    if drive == "current":                 # DEVSIM refuses an AC part on a current source
        ds.circuit_element(name=source_name, n1=node_name, n2="0", value=0.0)
    else:
        ds.circuit_element(name=source_name, n1=node_name, n2="0", value=0.0,
                           acreal=float(v_ac), acimag=0.0)
    cp = "{}_potential_circuit".format(contact)
    ds.contact_node_model(device=device, contact=contact, name=cp,
                          equation="Potential - {}".format(node_name))
    ds.contact_node_model(device=device, contact=contact, name="{}:Potential".format(cp),
                          equation="1")
    ds.contact_node_model(device=device, contact=contact, name="{}:{}".format(cp, node_name),
                          equation="-1")
    _R.record_contact_equation(device, contact, name="PotentialEquation", node_model=cp,
                               edge_charge_model="PotentialEdgeFlux", circuit_node=node_name)
    ce = "{}_electrons_circuit".format(contact)
    ds.contact_node_model(device=device, contact=contact, name=ce, equation="Electrons - N_D")
    ds.contact_node_model(device=device, contact=contact, name="{}:Electrons".format(ce),
                          equation="1")
    _R.record_contact_equation(device, contact, name="ElectronContinuityEquation", node_model=ce,
                               edge_current_model="ElectronCurrent", circuit_node=node_name)
    return source_name, node_name
