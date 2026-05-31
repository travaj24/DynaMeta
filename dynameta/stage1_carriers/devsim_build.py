"""
Build a 2D DEVSIM device from a dynameta.Design.

The DEVSIM Cartesian 2D mesh has:
  x in [0, P]              (lateral, FULL period -- no symmetry)
  y in [0, sum_layer_thk]  (vertical stack, matches layer order)

Layers with lateral_extent='patch_footprint' get a limited x-range
[(P-L)/2, (P+L)/2]. Layers with lateral_extent='full_period' span all x.

Electrodes are translated to DEVSIM contacts: each contact attaches to
a NEIGHBOURING region with the metal layer's bbox (per DEVSIM 2D pattern).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import devsim as ds

from dynameta.design import Design, Layer, Electrode
from dynameta.stage1_carriers import physics as P


# ---------------------------------------------------------------------------
# Mesh + region construction
# ---------------------------------------------------------------------------

@dataclass
class Stage1BuildResult:
    """Returned by build_devsim_device. Holds DEVSIM handles + layer/electrode
    metadata so downstream code (solver, dumper) doesn't have to re-derive
    things from the Design."""
    device:                    str
    mesh:                      str
    z_intervals:               Dict[str, Tuple[float, float]]
    semiconductor_layer_names: List[str]
    actual_interfaces:         List[str]
    actual_contacts:           List[str]


def build_devsim_device(design: Design,
                          *, mesh_name: str = "ms_mesh",
                          device_name: str = "ms_device"
                          ) -> Stage1BuildResult:
    """Build the 2D DEVSIM device + register parameters / equations for
    the drift-diffusion + Poisson solve.

    Returns a Stage1BuildResult with the handles + metadata needed by
    the rest of Stage 1.
    """
    spec = design.mesh_2d
    P_m  = design.period_m
    L_m  = design.patch_side_m
    z_intervals = design.layer_z_intervals()

    # Metals are treated as EQUIPOTENTIAL CONTACTS, not meshed Poisson regions.
    # Meshing a metal as a free Laplace field (eps_r=1) couples a large
    # unconstrained potential field into the ITO's exponential F_1/2, producing
    # a positive-feedback Newton divergence. So we mesh only the non-metal
    # (dielectric + semiconductor) cavity layers, and each metal electrode
    # becomes a Dirichlet contact on the adjacent meshed boundary.
    meshed_layers = [L for L in design.layers if L.role != "metal"]
    if not meshed_layers:
        raise ValueError("Design has no non-metal layers to mesh.")
    layer_index = {L.name: i for i, L in enumerate(design.layers)}

    def metal_contact_target(metal_layer: Layer) -> Tuple[str, float]:
        """For a metal electrode, return (meshed_region_name, contact_z): the
        nearest meshed layer and the z where the metal touches it."""
        idx = layer_index[metal_layer.name]
        for j in range(idx + 1, len(design.layers)):       # search upward
            if design.layers[j].role != "metal":
                mr = design.layers[j]
                return mr.name, z_intervals[mr.name][0]     # metal below -> region's bottom
        for j in range(idx - 1, -1, -1):                    # search downward
            if design.layers[j].role != "metal":
                mr = design.layers[j]
                return mr.name, z_intervals[mr.name][1]     # metal above -> region's top
        raise ValueError("metal '{}' has no meshed neighbour".format(metal_layer.name))

    ds.create_2d_mesh(mesh=mesh_name)

    # ---------- x mesh lines ----------
    px_lo = (P_m - L_m) / 2.0
    px_hi = (P_m + L_m) / 2.0
    ds.add_2d_mesh_line(mesh=mesh_name, dir="x", pos=0.0,
                          ns=spec.x_spacing_sym_m, ps=spec.x_spacing_sym_m)
    ds.add_2d_mesh_line(mesh=mesh_name, dir="x", pos=px_lo,
                          ns=spec.x_spacing_patch_edge_m,
                          ps=spec.x_spacing_patch_edge_m)
    ds.add_2d_mesh_line(mesh=mesh_name, dir="x", pos=P_m / 2.0,
                          ns=spec.x_spacing_patch_mid_m,
                          ps=spec.x_spacing_patch_mid_m)
    ds.add_2d_mesh_line(mesh=mesh_name, dir="x", pos=px_hi,
                          ns=spec.x_spacing_patch_edge_m,
                          ps=spec.x_spacing_patch_edge_m)
    ds.add_2d_mesh_line(mesh=mesh_name, dir="x", pos=P_m,
                          ns=spec.x_spacing_sym_m, ps=spec.x_spacing_sym_m)

    # ---------- y mesh lines (ALL layers meshed: a metal must be MESHED for its
    # Dirichlet contact to realize -- the contact box has to overlap real mesh
    # cells to capture the adjacent region's boundary edges. Metals get NO
    # equation (inert, no node_solution -> 0 matrix rows), exactly like the
    # working Modulator's patch_metal. The interior ITO/oxide layers get the
    # Modulator's interface-zone refinement so the ITO column has interior
    # nodes (the ground contact then captures adjacent nodes WITH edges, not
    # just the 2 interface-corner nodes). ----------
    izone = spec.interface_zone_m
    for L in design.layers:
        z_lo, z_hi = z_intervals[L.name]
        thk = z_hi - z_lo
        # Fine line at the lower boundary; fine if this layer or its neighbour
        # is the semiconductor, else coarse.
        ds.add_2d_mesh_line(mesh=mesh_name, dir="y", pos=z_lo,
                              ns=spec.interface_ps_m, ps=spec.interface_ps_m)
        # Interior interface-zone lines (Modulator mesh_lines_for_layer): bracket
        # each boundary with a coarse line at +/- zone so the boundary refinement
        # grades out into the bulk and the layer gets MULTIPLE interior nodes.
        zone = min(izone, 0.45 * thk)
        if zone > 0:
            ds.add_2d_mesh_line(mesh=mesh_name, dir="y", pos=z_lo + zone,
                                  ns=spec.coarse_ps_m, ps=spec.coarse_ps_m)
            ds.add_2d_mesh_line(mesh=mesh_name, dir="y", pos=z_hi - zone,
                                  ns=spec.coarse_ps_m, ps=spec.coarse_ps_m)
        if thk > 4 * izone:
            ds.add_2d_mesh_line(mesh=mesh_name, dir="y", pos=0.5 * (z_lo + z_hi),
                                  ns=spec.coarse_ps_m, ps=spec.coarse_ps_m)
    # Top of the stack
    top_z = z_intervals[design.layers[-1].name][1]
    ds.add_2d_mesh_line(mesh=mesh_name, dir="y", pos=top_z,
                          ns=spec.interface_ps_m, ps=spec.interface_ps_m)

    # ---------- regions (ALL layers meshed; metals are inert -- see _setup) ----------
    for L in design.layers:
        z_lo, z_hi = z_intervals[L.name]
        if L.lateral_extent.kind == "patch_footprint":
            xl, xh = px_lo, px_hi
        elif L.lateral_extent.kind == "full_period":
            xl, xh = 0.0, P_m
        else:
            xl, xh, _, _ = L.lateral_extent.bbox(P_m, L_m)[:4]
        ds.add_2d_region(mesh=mesh_name, region=L.name,
                          material=L.material,
                          xl=xl, xh=xh, yl=z_lo, yh=z_hi, bloat=1e-12)

    # ---------- interfaces (layer i -> layer i+1) ----------
    # Only add the interface where BOTH layers exist at that x range. For
    # patch_footprint layers, the interface is only over the patch.
    for i in range(len(meshed_layers) - 1):
        L0 = meshed_layers[i]
        L1 = meshed_layers[i + 1]
        # only z-adjacent meshed layers share an interface (skip if a metal
        # gap sits between them)
        if abs(z_intervals[L0.name][1] - z_intervals[L1.name][0]) > 1e-15:
            continue
        z_iface = z_intervals[L0.name][1]
        # x-overlap of the two layers' extents
        x0_lo, x0_hi = _layer_x_bounds(L0, P_m, L_m, px_lo, px_hi)
        x1_lo, x1_hi = _layer_x_bounds(L1, P_m, L_m, px_lo, px_hi)
        xl_if = max(x0_lo, x1_lo)
        xh_if = min(x0_hi, x1_hi)
        if xh_if <= xl_if:
            continue
        ds.add_2d_interface(mesh=mesh_name,
                              name="{}__{}".format(L0.name, L1.name),
                              region0=L0.name, region1=L1.name,
                              xl=xl_if, xh=xh_if,
                              yl=z_iface, yh=z_iface,
                              bloat=1e-12)

    # ---------- electrodes -> DEVSIM contacts ----------
    # Each Electrode resolves to a bounding box on the attached layer.
    # DEVSIM's pattern: attach the contact to a NEIGHBOURING region with
    # bbox = the metal layer's extent. We do that here by looking up the
    # layer ABOVE the attached layer (or the layer BELOW for the bottom-
    # most metal). Edge-located ITO grounds attach to the semiconductor
    # itself.
    contact_to_region: Dict[str, str] = {}
    for E in design.electrodes:
        attached_L = design.find_layer(E.attached_layer)
        z_iv = z_intervals[E.attached_layer]
        xlo, xhi, ylo, yhi, zlo, zhi = E.resolve_bbox(P_m, L_m, z_iv)
        if E.location in ("x_lo_edge", "x_hi_edge", "y_lo_edge", "y_hi_edge"):
            # Edge ground (e.g. ITO at x=0 / x=P): FULL z-range so the contact
            # realizes (an inset z-range fails to form a boundary contact). The
            # corner conflict with the continuous interfaces is removed by
            # insetting those INTERFACES off x=0/P (below) -- so the corner is a
            # PURE contact node, which the edge_charge_model contact equation
            # then assembles correctly.
            region = attached_L.name
            yl_c, yh_c = z_iv[0], z_iv[1]
        elif attached_L.role == "metal":
            # Metal electrode: NOT meshed -> Dirichlet contact on the adjacent
            # meshed region. Use a SLAB spanning from that region's touching
            # boundary through the metal's z-extent (NOT a zero-thickness line):
            # DEVSIM needs the slab to overlap the boundary so it captures the
            # boundary EDGES (a line catches nodes but no edges -> contact is
            # silently dropped). min/max also bridges any inert gap (e.g. the
            # adhesion layer between upper_al2o3 and the patch). Matches the
            # working Modulator mos_cap_2d contact pattern.
            region, ybnd = metal_contact_target(attached_L)
            yl_c, yh_c = min(ybnd, zlo), max(ybnd, zhi)
        else:
            region = attached_L.name
            yl_c = yh_c = z_iv[1]
        ds.add_2d_contact(mesh=mesh_name, name=E.name,
                            material="metal", region=region,
                            xl=xlo, xh=xhi, yl=yl_c, yh=yh_c, bloat=1e-10)
        contact_to_region[E.name] = region

    ds.finalize_mesh(mesh=mesh_name)
    ds.create_device(mesh=mesh_name, device=device_name)

    # ---------- parameters + equations per region (meshed only) ----------
    for L in meshed_layers:
        _setup_region(design, device_name, L)

    # ---------- interfaces ----------
    actual_interfaces = list(ds.get_interface_list(device=device_name))
    for iface in actual_interfaces:
        _setup_interface(device_name, iface)

    # ---------- contacts (contact_to_region built during contact creation) ----------
    actual_contacts = list(ds.get_contact_list(device=device_name))
    for c in actual_contacts:
        region = contact_to_region.get(c)
        if region is None:
            continue
        _setup_contact(device_name, c, region, design)

    semi_names = [L.name for L in design.layers if L.role == "semiconductor"]
    return Stage1BuildResult(
        device=device_name, mesh=mesh_name,
        z_intervals=z_intervals,
        semiconductor_layer_names=semi_names,
        actual_interfaces=actual_interfaces,
        actual_contacts=actual_contacts,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _layer_x_bounds(L: Layer, P_m: float, L_patch_m: float,
                     px_lo: float, px_hi: float) -> Tuple[float, float]:
    if L.lateral_extent.kind == "patch_footprint":
        return (px_lo, px_hi)
    if L.lateral_extent.kind == "full_period":
        return (0.0, P_m)
    xlo, xhi, _, _ = L.lateral_extent.bbox(P_m, L_patch_m)[:4]
    return (xlo, xhi)


def _resolve_contact_regions(design: Design,
                                layer_index: Dict[str, int]) -> Dict[str, str]:
    """Map each Electrode name to the DEVSIM region the contact attached to.

    Contacts now always attach to their OWN attached layer (metals at the
    outer face, edge grounds at the edge), so the mapping is direct.
    """
    return {E.name: E.attached_layer for E in design.electrodes}


def _setup_region(design: Design, device: str, L: Layer) -> None:
    """Register the appropriate Poisson/continuity equations on this region."""
    if L.role == "semiconductor":
        _setup_semiconductor(design, device, L)
    elif L.role == "dielectric":
        _setup_dielectric(design, device, L)
    elif L.role == "metal":
        _setup_metal(design, device, L)


def _setup_metal(design: Design, device: str, L: Layer) -> None:
    """Metal region: Poisson with eps_r = 1 + Dirichlet contact BC.
    Effectively an equipotential at the contact bias."""
    mat = design.materials.get(L.material)
    # Metal -- give it an arbitrary eps_r so Poisson is well-defined;
    # contact BCs will pin the potential.
    ds.set_parameter(device=device, region=L.name, name="Permittivity",
                       value=1.0 * P.EPS0)
    ds.set_parameter(device=device, region=L.name, name="ElectronCharge",
                       value=P.Q_E)
    ds.node_solution(device=device, region=L.name, name="Potential")
    ds.edge_from_node_model(device=device, region=L.name, node_model="Potential")
    ds.edge_model(device=device, region=L.name,
                    name="ElectricField",
                    equation="(Potential@n0 - Potential@n1) * EdgeInverseLength")
    ds.edge_model(device=device, region=L.name,
                    name="ElectricField:Potential@n0",
                    equation="EdgeInverseLength")
    ds.edge_model(device=device, region=L.name,
                    name="ElectricField:Potential@n1",
                    equation="-EdgeInverseLength")
    ds.edge_model(device=device, region=L.name,
                    name="PotentialEdgeFlux",
                    equation="Permittivity * ElectricField")
    ds.edge_model(device=device, region=L.name,
                    name="PotentialEdgeFlux:Potential@n0",
                    equation="Permittivity * EdgeInverseLength")
    ds.edge_model(device=device, region=L.name,
                    name="PotentialEdgeFlux:Potential@n1",
                    equation="-Permittivity * EdgeInverseLength")
    ds.equation(device=device, region=L.name, name="PotentialEquation",
                  variable_name="Potential",
                  edge_model="PotentialEdgeFlux",
                  variable_update="default")


def _setup_dielectric(design: Design, device: str, L: Layer) -> None:
    """Dielectric region: pure Poisson (no carriers)."""
    mat = design.materials.get(L.material)
    eps_r = L.eps_static_override
    if eps_r is None:
        if mat.drude is not None:
            eps_r = mat.drude.eps_static
        else:
            # Treat dielectric optical eps_inf as a stand-in if no static given
            eps_r = mat.optical_eps(1300e-9).real
    ds.set_parameter(device=device, region=L.name, name="Permittivity",
                       value=eps_r * P.EPS0)
    ds.set_parameter(device=device, region=L.name, name="ElectronCharge",
                       value=P.Q_E)
    ds.node_solution(device=device, region=L.name, name="Potential")
    ds.edge_from_node_model(device=device, region=L.name, node_model="Potential")
    ds.edge_model(device=device, region=L.name,
                    name="ElectricField",
                    equation="(Potential@n0 - Potential@n1) * EdgeInverseLength")
    ds.edge_model(device=device, region=L.name,
                    name="ElectricField:Potential@n0",
                    equation="EdgeInverseLength")
    ds.edge_model(device=device, region=L.name,
                    name="ElectricField:Potential@n1",
                    equation="-EdgeInverseLength")
    ds.edge_model(device=device, region=L.name,
                    name="PotentialEdgeFlux",
                    equation="Permittivity * ElectricField")
    ds.edge_model(device=device, region=L.name,
                    name="PotentialEdgeFlux:Potential@n0",
                    equation="Permittivity * EdgeInverseLength")
    ds.edge_model(device=device, region=L.name,
                    name="PotentialEdgeFlux:Potential@n1",
                    equation="-Permittivity * EdgeInverseLength")
    ds.equation(device=device, region=L.name, name="PotentialEquation",
                  variable_name="Potential",
                  edge_model="PotentialEdgeFlux",
                  variable_update="default")


def _setup_semiconductor(design: Design, device: str, L: Layer) -> None:
    """Drift-diffusion + Poisson for a semiconductor (e.g. ITO)."""
    mat = design.materials.get(L.material)
    if mat.drude is None:
        raise ValueError("Semiconductor layer '{}' material '{}' has no "
                          "DrudeSpec".format(L.name, L.material))
    drude = mat.drude
    # Phi_c0 and N_c (uses density-dependent m_eff at n_bg)
    m_eff_at_bg = float(drude.m_eff_kg_of_n_m3(drude.n_bg_m3))
    Phi_c0 = P.setup_phi_c0(device, L.name, drude.band_gap_eV, drude.chi_eV,
                              drude.n_bg_m3, m_eff_at_bg)
    eps_r = L.eps_static_override if L.eps_static_override is not None else drude.eps_static
    ds.set_parameter(device=device, region=L.name, name="Permittivity",
                       value=eps_r * P.EPS0)
    ds.set_parameter(device=device, region=L.name, name="ElectronCharge",
                       value=P.Q_E)
    ds.set_parameter(device=device, region=L.name, name="V_t", value=P.V_T)
    # Potential is the ONLY solution variable. Electrons is a DERIVED node_model
    # n = N_c * F_1/2((Potential - Phi_c0)/V_t); there is NO separate Electrons
    # solution variable and NO continuity equation. This single-variable
    # nonlinear Poisson is the working Modulator mos_cap formulation
    # (setup_ito_region). A second Electrons solution variable is what produced
    # the contact/interface corner singularities (empty matrix rows) -- so it is
    # removed entirely. variable_update="log_damp" damps the Newton Potential
    # step so the exponential F_1/2 cannot overflow.
    ds.node_solution(device=device, region=L.name, name="Potential")
    ds.edge_from_node_model(device=device, region=L.name, node_model="Potential")
    n_nodes = len(ds.get_node_model_values(device=device, region=L.name, name="x"))
    ds.set_node_values(device=device, region=L.name, name="Potential",
                          values=[0.0] * n_nodes)
    # Electrons as a function of Potential (+ its derivative via DEVSIM diff).
    eta = "(Potential - Phi_c0) / V_t"
    elec_eq = "N_c * " + P.F12_aymerich_humet_expr(eta)
    ds.node_model(device=device, region=L.name, name="Electrons", equation=elec_eq)
    ds.node_model(device=device, region=L.name, name="Electrons:Potential",
                    equation="simplify(diff({}, Potential))".format(elec_eq))
    # Poisson net charge: q*(n - N_D), n = Electrons(Potential).
    net_eq = "ElectronCharge * (Electrons - N_D)"
    ds.node_model(device=device, region=L.name, name="PotentialNodeCharge",
                    equation=net_eq)
    ds.node_model(device=device, region=L.name, name="PotentialNodeCharge:Potential",
                    equation="simplify(diff({}, Potential))".format(net_eq))
    # Displacement-flux edge model (derivatives via DEVSIM diff).
    e_ef = "(Potential@n0 - Potential@n1) * EdgeInverseLength"
    ds.edge_model(device=device, region=L.name, name="ElectricField", equation=e_ef)
    ds.edge_model(device=device, region=L.name, name="ElectricField:Potential@n0",
                    equation="simplify(diff({}, Potential@n0))".format(e_ef))
    ds.edge_model(device=device, region=L.name, name="ElectricField:Potential@n1",
                    equation="simplify(diff({}, Potential@n1))".format(e_ef))
    e_flux = "Permittivity * ElectricField"
    ds.edge_model(device=device, region=L.name, name="PotentialEdgeFlux", equation=e_flux)
    ds.edge_model(device=device, region=L.name, name="PotentialEdgeFlux:Potential@n0",
                    equation="simplify(diff({}, Potential@n0))".format(e_flux))
    ds.edge_model(device=device, region=L.name, name="PotentialEdgeFlux:Potential@n1",
                    equation="simplify(diff({}, Potential@n1))".format(e_flux))
    ds.equation(device=device, region=L.name, name="PotentialEquation",
                  variable_name="Potential",
                  node_model="PotentialNodeCharge",
                  edge_model="PotentialEdgeFlux",
                  variable_update="log_damp")


def _setup_interface(device: str, iface: str) -> None:
    """Continuity of Potential across an interface."""
    ds.interface_model(device=device, interface=iface,
                          name="continuousPotential",
                          equation="Potential@r0 - Potential@r1")
    ds.interface_model(device=device, interface=iface,
                          name="continuousPotential:Potential@r0",
                          equation="1.0")
    ds.interface_model(device=device, interface=iface,
                          name="continuousPotential:Potential@r1",
                          equation="-1.0")
    ds.interface_equation(device=device, interface=iface,
                              name="PotentialEquation",
                              interface_model="continuousPotential",
                              type="continuous")


def _setup_contact(device: str, contact: str, region: str,
                    design: Design) -> None:
    """Dirichlet BC at a contact: Potential = bias parameter."""
    ds.set_parameter(device=device, name="{}_bias".format(contact), value=0.0)
    # UNIQUE node-model name per contact. Two contacts on the SAME region
    # (e.g. ito_gnd_left and ito_gnd_right both on the ITO) would otherwise
    # share the name "contactPotential" in that region's model namespace, and
    # the second setup REPLACES the first ("Replacing Node Model contactPotential
    # in region ito") -- leaving the first contact's nodes with no Dirichlet
    # equation (empty matrix rows -> singular). The Modulator uses
    # "{contact}_potential_dirichlet" for exactly this reason.
    contact_node = "{}_potential_dirichlet".format(contact)
    ds.contact_node_model(device=device, contact=contact,
                              name=contact_node,
                              equation="Potential - {}_bias".format(contact))
    ds.contact_node_model(device=device, contact=contact,
                              name="{}:Potential".format(contact_node),
                              equation="1")
    # Pin ONLY Potential, WITH edge_charge_model to close the displacement flux
    # at the contact. This matches the working Modulator mos_cap setup_contact:
    # the edge_charge_model is what lets a contact node that ALSO lies on a
    # continuous interface (the ITO ground/interface corners) assemble a proper
    # equation -- without it those shared corner nodes get empty matrix rows
    # (-> singular). Electrons at the contact follow the bulk ElectronEquation
    # given the pinned Potential, so no separate Electron contact equation is
    # needed (and adding one re-introduces the empty-corner problem).
    ds.contact_equation(device=device, contact=contact,
                            name="PotentialEquation",
                            node_model=contact_node,
                            edge_charge_model="PotentialEdgeFlux")
