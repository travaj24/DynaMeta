"""Tunneling on DEVSIM devices (D1d, v0.14.0): band-to-band tunneling (BTBT) GENERATION coupled into the
bipolar drift-diffusion Newton, and the gate-oxide tunneling CURRENT post-processed from the solved
oxide field. Both are OPT-IN; nothing here runs unless asked, so every existing device is unchanged.

BAND-TO-BAND TUNNELING (local Kane / Hurkx model)
-------------------------------------------------
Net BTBT recombination rate [1/(m^3 s)], added to SRH in BOTH continuity equations:

    U_BTBT = A |E|^P exp(-B/|E|) * D(n, p),       D = (n p - n_i^2) / ((n + n_i)(p + n_i))

  * the field prefactor A |E|^P exp(-B/|E|) is Kane's tunneling rate (E. O. Kane, J. Appl. Phys. 32,
    83 (1961)): P = 2 for a DIRECT gap, P = 2.5 for the phonon-assisted (indirect) transition;
  * D is the Hurkx factor (G. A. M. Hurkx, D. B. M. Klaassen, M. P. G. Knuvers, IEEE Trans. Electron
    Devices 39, 331 (1992)): D = 0 at equilibrium (n p = n_i^2) so U vanishes EXACTLY there, D -> -1
    in a depleted reverse-biased junction (n, p << n_i: net GENERATION, U < 0), D > 0 under forward
    bias (tunneling-assisted recombination). Without D the Kane rate generates at zero bias -- a
    current with no driving force (the gate in tests/test_devsim_tunneling.py shows the magnitude).

Si parameters (Hurkx 1992; the same numbers are the Hurkx-model defaults of Synopsys Sentaurus
Device): A = 4e14 cm^-1/2 V^-5/2 s^-1, B = 1.9e7 V/cm, P = 2.5. In SI (BTBTParams.from_cgs; with E
in V/cm the cgs rate is A E^P cm^-3 s^-1): A_SI = A_cgs 1e6 100^-P = 4.0e15 m^-3 s^-1 (V/m)^-2.5,
B_SI = 1.9e9 V/m. No other material set is shipped: build BTBTParams from a cited source.

FIELD: LOCAL |E| (the nonlocal path-integrated model -- tunneling over the band-bending distance,
Hurkx sec. IV / Schenk -- is a documented follow-up). Discretisation, by dimension:
  * 2-D / 3-D: the ELEMENT field. DEVSIM's element_from_edge_model gives the exact P1 gradient of the
    potential in each triangle/tetrahedron (MEASURED: a uniform 1e8 V/m field at 30 deg is reproduced
    to 2e-16 on a 2-D tensor mesh), so |E| is ISOTROPIC -- unlike the edge-projected field the
    impact-ionization post-processor uses (impact_ionization.py: 1.8-3.1x low on 3-D tets). U is
    assembled per element edge into each end node's share of that element's volume
    (volume_node0_model / volume_node1_model, weight ElementNodeVolume) with D evaluated at THAT node --
    so a uniform field integrates exactly on any mesh whose element volumes are exact (2 sum ENV = the
    element volume). DEVSIM 2.11 breaks that identity on elements that are not well centred (it sums the
    magnitudes of the circumcentric pieces: an obtuse triangle 11x, the old gmsh tets 1.2-1.5x), so
    setup_btbt runs the shared element_geometry.check_element_geometry first and REFUSES such a mesh
    (InexactElementGeometryError). The library meshes (2-D rectilinear tensor, 3-D Kuhn tensor) pass.
  * 1-D: DEVSIM has no element models in 1-D (element_from_edge_model raises), and there the edge
    field IS the field: U on each edge with D = the mean of its two end-node D values, assembled via
    edge_volume_model (EdgeNodeVolume = L/2 to each node).
  |E| carries a 1 V/m floor, |E| = sqrt(E.E + 1 (V/m)^2), so its derivative is finite at E = 0 (and
  exp(-B/1 V/m) = 0 there); the bias it adds is < 1e-14 relative above 1e7 V/m.

COUPLING: SELF-CONSISTENT (inside the coupled Newton): the generation feeds the continuity equations
with full analytic Jacobians (the derivative models are DEVSIM symbolic diffs; checked against finite
differences in the tests). Temperature: fixed T_REF (no Eg(T)^1.5 scaling of B). On a region that runs
at another temperature (BipolarOptions.T_K) setup_btbt still installs the T_REF parameters and emits an
AdvisoryWarning saying so (v0.14 row 3.2-9); the E_g(T) scaling of B is planned for v0.15. Until then,
for a T_K device supply A, B fitted AT that temperature (BTBTParams.from_cgs) -- the warning still fires,
because the code cannot tell which temperature a parameter set belongs to.

GATE-OXIDE TUNNELING CURRENT (post-processed)
---------------------------------------------
oxide_tunneling_current() reads the CONVERGED potential in a dielectric region, takes the oxide-side
field normal to its (planar) interface with a semiconductor region, and evaluates one of the shared
kernels of dynameta.reliability.leakage at each interface node:
  "fn"           Fowler-Nordheim, J = A E^2 exp(-B/E) (Lenzlinger & Snow, J. Appl. Phys. 40, 278 (1969));
  "schuegraf_hu" the full Schuegraf-Hu trapezoid closed form (== FN above the barrier);
  "tsu_esaki"    the Tsu-Esaki/WKB quadrature (both regimes; needs the supply E_F and mass).
At each interface node the oxide drop V_ox is the COLUMN drop: the potential on the opposite (gate-side)
face of the oxide, interpolated to the node's lateral position, minus the node's potential; E_ox =
V_ox/t_ox with t_ox the oxide thickness along the normal. Both are exact for the linear potential of a
charge-free planar oxide on any mesh (a uniform-gate capacitor is exact); under a patch-gate edge they
are the straight-column approximation. The current is integrated over the interface with DEVSIM's
SurfaceArea node weights.

WHY POST-PROCESSED, and its limit: the tunneling current is NOT fed back into the semiconductor
continuity equation, so the carrier profile is that of an ideal (non-leaky) capacitor. That is exact to
first order while the leakage is small against the transport currents (the static-power / hold-bias
question it answers). It is WRONG where the leakage itself sets the carrier state -- a floating or
deep-depleted body whose minority supply is the tunneling current, or a gate stack in heavy FN
injection. For those, use the self-consistent coupling below (v0.15); the post-processor stays the right
tool for the static leakage of a gate that does not perturb the carriers.

SIGN: J > 0 when the gate side of the oxide is at the HIGHER potential (electrons tunnel from the
semiconductor into the gate; conventional current flows gate -> semiconductor). The SAME barrier/supply
parameters serve both polarities (an asymmetric gate/semiconductor stack needs two calls).

UNITS follow the D1 contact-current convention: a 1-D device is per unit area (I in A/m^2); the 2-D
layered mesh is per unit depth (pass depth_m = period_y_m to get A); the 3-D gmsh mesh is in metres
(I already in A).

SELF-CONSISTENT OXIDE TUNNELLING (v0.15, plan item 4.1.1; docs/audit/2026-11-06-v0.15-oxide-tunnelling.md)
------------------------------------------------------------------------------------------------------
setup_oxide_tunneling(device, semi, gate, OxideTunnelingParams(model="tsu_esaki",
coupling="self_consistent", barriers=SIO2_ON_SI, gate_fermi_energy_eV=...)) puts the oxide current INTO
the Newton, as a nonlocal custom term (carriers.custom_terms):
  * CURRENT: the two-sided Tsu-Esaki integral with WKB transmission through the dielectric STACK
    (reliability.leakage.tsu_esaki_two_sided; per layer the closed linear-barrier exponent, Anderson-rule
    band offsets), on a FIXED energy grid so that J is a smooth function of the state and the analytic
    Jacobian is the exact derivative of the discrete sum (FD-checked; jacobian_exact=True).
  * SUPPLY: the semiconductor side is the LOCAL electron quasi-Fermi level of each interface node,
    (E_Fs - E_c)/kT = ln(n/N_dos) + Delta(n/N_dos) -- the bulk's own closed-form Fermi-Dirac log-activity
    (physics_heterojunction.setup_log_activity; Delta = 0 for Boltzmann). So J vanishes POINTWISE at
    equilibrium (detailed balance by construction) and J ~ n_s in the non-degenerate limit -- the supply
    limitation of a deep-depleted body appears by itself, with no fitted factor. The gate side is the
    metal Fermi level -q V_gate; its barrier is READ from the gate contact's potential offset (user
    decision 12): phi_Bg = q (phi_c - offset) + offset_last_layer, so electrostatics and tunnelling cannot
    disagree about the gate work function (gate_offset_for_work_function converts a work function).
  * EQUATIONS: -A_i J_i on the ElectronContinuityEquation row of every semiconductor interface node i
    (A_i = its SurfaceArea), +Sum_i A_i J_i on the gate's KCL row when the gate is CIRCUIT-driven -- then
    ssac sees the tunnel conductance and terminal currents balance in the circuit. A bias-parameter gate
    gets no equation entry: its DC current is the module-reported I_gate (oxide_tunneling_gate_currents;
    contact_current.extract_contact_currents adds it). No circuit is ever created (decision 11).
  * NOISE: the shot noise of the tunnel current as a two-point source (noise_analysis, mechanism
    "tunnel_ox"): electron row of node i (-1) to the gate circuit row (+1), rate A S/(2 q^2) with
    S = 2 q J coth(qV/2kT), qV = E_Fs - E_Fg -- the exact T << 1 partition, = 4 kT G at equilibrium.
  REFUSED, with the reason: FN / Schuegraf-Hu as self-consistent supplies (one-sided: an equilibrium
  current and an FDT violation; they stay post-processors), poly-Si (semiconductor) gates, patch gates,
  non-planar columns, bandgap narrowing / density gradient on the region, a gate band bottom above E_c,s.
  NOT MODELLED (stated): hole and valence-band tunnelling (electrons only, decision 14: the report warns
  where p_s > n_s under a strong field); image-force lowering (decision 15: the 3.1 eV Lenzlinger-Snow
  barrier is an FN fit that already absorbs it -- adding it would double-count ~0.26 V at 1 MV/cm);
  2-D quantisation of an inversion / accumulation supply; oxide trapping. No measured high-k stack J-V
  is in hand (decision 16), so stacks are validated only against the kernel's quadrature oracles.
  ACCURACY NOTES (MEASURED, audit note): with the Fermi-Dirac g-factor the DISCRETE bulk equilibrium is
  off the closed-form activity by O(h^2) in the degenerate surface layer (zero-bias |J| / J_one-way =
  1.6e-3 at h = 0.1 nm, 4.0e-4 at 0.05 nm; Boltzmann: 1e-14) -- mesh the surface finely, or use Boltzmann
  where degeneracy is irrelevant. On a non-well-centred (gmsh) 3-D mesh the lateral oxide-voltage spread is
  amplified ~b = B/E ~ 25x into the current: setup_oxide_tunneling warns there.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np

from dynameta.carriers.custom_terms import CustomTerm
from dynameta.constants import Q_E, T_REF, V_T
from dynameta.warning_types import AdvisoryWarning

__all__ = ["BTBTParams", "SILICON_HURKX", "btbt_rate_expr", "hurkx_d_expr", "setup_btbt", "btbt_generation_current",
           "OxideTunnelingParams", "oxide_tunneling_current", "TUNNEL_MODELS", "TUNNEL_COUPLINGS",
           "DielectricBarrier", "SIO2_ON_SI", "setup_oxide_tunneling", "OxideTunnelTerm",
           "oxide_tunneling_terms", "oxide_tunneling_gate_currents", "gate_offset_for_work_function",
           "dielectric_column"]


# ------------------------------------------------------------------------------------------------
# BTBT
# ------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class BTBTParams:
    """Kane/Hurkx BTBT coefficients in SI: rate A |E|^P exp(-B/|E|) [1/(m^3 s)] with |E| in V/m, so
    A_SI is in m^-3 s^-1 (V/m)^-P and B in V/m. The Hurkx D factor is ALWAYS applied (it is what makes
    the rate vanish at equilibrium); see the module header."""
    A_SI: float
    B_V_per_m: float
    P: float = 2.5

    def __post_init__(self):
        if not (self.A_SI >= 0.0 and self.B_V_per_m > 0.0 and self.P >= 0.0):
            raise ValueError("BTBTParams: need A_SI >= 0, B_V_per_m > 0, P >= 0")

    @classmethod
    def from_cgs(cls, A_cgs: float, B_V_per_cm: float, P: float) -> "BTBTParams":
        """From the literature's cgs form, rate = A_cgs (E[V/cm])^P [cm^-3 s^-1]:
        A_SI = A_cgs * 1e6 * 100^-P, B_SI = 100 * B_V_per_cm."""
        return cls(A_SI=float(A_cgs) * 1.0e6 * 100.0 ** (-float(P)),
                   B_V_per_m=100.0 * float(B_V_per_cm), P=float(P))


# Hurkx, Klaassen & Knuvers, IEEE TED 39, 331 (1992): Si, phonon-assisted. A_SI = 4.0e15, B = 1.9e9 V/m.
SILICON_HURKX = BTBTParams.from_cgs(4.0e14, 1.9e7, 2.5)


def hurkx_d_expr(n: str, p: str, n_i: str = "n_i", *, pair: bool = False) -> str:
    """The Hurkx factor D = (n p - n_i^2)/((n + n_i)(p + n_i)) as a DEVSIM expression. `n_i` is the
    intrinsic-density symbol (a region parameter by default; pass the node-suffixed effective density,
    e.g. "n_ie@n0", under bandgap narrowing). pair=True gives the PAIR factor (n p + n_i^2)/(...): the
    detailed-balance split R ~ n p, G ~ n_i^2 of the net rate, whose sum G + R is the noise GR source
    (noise_analysis) -- the same split as the SRH kernel's pair rate."""
    return "(({n})*({p}) {s} {ni}^2)/((({n}) + {ni})*(({p}) + {ni}))".format(
        n=n, p=p, ni=n_i, s="+" if pair else "-")


def btbt_rate_expr(params: BTBTParams, *, emag: str, d: str) -> str:
    """THE shared DEVSIM expression for the net BTBT recombination rate U [1/(m^3 s)]:
    A exp(P ln|E| - B/|E|) * D. `emag` names a model/expression for |E| (> 0 -- floored by the callers);
    `d` is the D-factor expression for the node(s) the term is assembled into (hurkx_d_expr at one
    node in 2-D/3-D, the mean of the two end nodes in 1-D). exp(P ln E - B/E) instead of
    E^P exp(-B/E): one exponential, no overflow of E^P and no 0*inf at the floor.

    Coefficients are written with "%.17e" (always a FLOAT literal). v0.14 integration fix: the former
    "%.17g" printed an integral value as an INTEGER literal ("10000000000000000"), and DEVSIM 2.11.2's
    parser CLAMPS integer literals above 2^31 - 1 to 2147483647 (MEASURED: "10000000000000000" ->
    2147483647.0, "1e16" -> 1e16) -- so any A >= 2^31 or B >= 2^31 V/m that happens to be integral
    was silently replaced. The shipped SILICON_HURKX (A = 4000000000000000.5, B = 1.9e9 < 2^31) was not
    affected; the parsed doubles are identical, so its results are unchanged."""
    return "{A:.17e}*exp({P:.17e}*log({E}) - {B:.17e}/({E}))*({D})".format(
        A=float(params.A_SI), P=float(params.P), B=float(params.B_V_per_m), E=emag, D=d)


def _require_bipolar(device: str, region: str):
    from dynameta.carriers import eq_registry as _R
    for eq in ("ElectronContinuityEquation", "HoleContinuityEquation"):
        if not _R.recorded(device, eq, scope="region", loc=region):
            raise ValueError("setup_btbt: region {!r} has no {} -- BTBT generation needs a bipolar "
                             "drift-diffusion region (physics_bipolar_dd.setup_bipolar_region)".format(
                                 region, eq))


def setup_btbt(device: str, region: str, params: BTBTParams, *, n_i: str = "n_i") -> None:
    """Install the BTBT term (module header) into an EXISTING bipolar region: build the field-magnitude
    and rate models (the diagnostics BTBT_Emag, BTBT_U / BTBT_U0, BTBT_U1) and add the rate through the
    ONE additive term mechanism, physics_bipolar_dd.add_recombination_term (assembly "edge" in 1-D,
    "element" in 2-D/3-D), with its noise pair rate. That mechanism wires the continuity equations'
    volume slots once (eq_registry.amend_region_equation, recorded, so a Gummel/staged
    delete_by_name -> reapply_by_name keeps them) and SUMS every term sharing a slot.
    Called by setup_bipolar_region(btbt=...).

    `n_i`: the intrinsic density in the Hurkx D -- a region PARAMETER name ("n_i", default) or a NODE
    MODEL name ("n_ie" under bandgap narrowing: setup_bipolar_region passes it, so D vanishes at the
    narrowed equilibrium n p = n_ie^2); a node model is evaluated at each D node (@n0/@n1 or
    @en0/@en1)."""
    from dynameta.carriers.devsim_env import import_devsim
    ds = import_devsim()
    from dynameta.carriers import physics_bipolar_dd as _BP
    from dynameta.carriers.eq_registry import edge_with_derivs

    if not isinstance(params, BTBTParams):
        raise TypeError("setup_btbt: params must be a BTBTParams (got {!r})".format(type(params)))
    _require_bipolar(device, region)
    # v0.14 row 3.2-9 (decision D-H4): the Kane/Hurkx A, B are T_REF numbers and are NOT scaled with the
    # region's temperature (B ~ E_g^(3/2) moves with E_g(T)); say so instead of running silently.
    vt = float(ds.get_parameter(device=device, region=region, name="V_t"))
    if vt != V_T:
        from dynameta.carriers.physics_heterojunction import temperature_of_V_t
        warnings.warn(
            "setup_btbt: region {!r} runs at T = {:.6g} K (BipolarOptions.T_K) but the BTBT parameters "
            "(A, B, P) are used as given, i.e. as at T_REF = {} K: B ~ E_g^(3/2) is not rescaled with "
            "E_g(T) (planned for v0.15). Supply A, B fitted at this temperature, or treat the BTBT "
            "current as a T_REF estimate.".format(region, temperature_of_V_t(vt), T_REF),
            AdvisoryWarning, stacklevel=3)
    ni_node = n_i in ds.get_node_model_list(device=device, region=region)

    def ni_at(sfx):
        return n_i + sfx if ni_node else n_i

    dim = int(ds.get_dimension(device=device))
    if dim == 1:
        # the edge field IS the field in 1-D; D = mean of the two end-node values
        if ni_node:
            ds.edge_from_node_model(device=device, region=region, node_model=n_i)
        edge_with_derivs(device, region, "BTBT_Emag", "pow(ElectricField^2 + 1.0, 0.5)", ("Potential",))

        def d_avg(pair):
            return "0.5*({} + {})".format(
                hurkx_d_expr("Electrons@n0", "Holes@n0", ni_at("@n0"), pair=pair),
                hurkx_d_expr("Electrons@n1", "Holes@n1", ni_at("@n1"), pair=pair))
        wrt = ("Potential", "Electrons", "Holes")
        edge_with_derivs(device, region, "BTBT_U", btbt_rate_expr(params, emag="BTBT_Emag",
                                                                  d=d_avg(False)), wrt)
        _BP.add_recombination_term(device, region, "btbt", "BTBT_U", assembly="edge",
                                   pair=btbt_rate_expr(params, emag="BTBT_Emag", d=d_avg(True)))
        return
    if dim not in (2, 3):                                     # pragma: no cover - devsim dims are 1..3
        raise NotImplementedError("setup_btbt: dimension {}".format(dim))
    # the element assembly weights each element edge's rate by ElementNodeVolume: exact only on a
    # well-centred mesh -- refuse any other (row 3.2-13, the shared check), BEFORE any model is created
    from dynameta.carriers.element_geometry import check_element_geometry
    check_element_geometry(device, region, caller="setup_btbt ({}-D element assembly)".format(dim))
    from dynameta.carriers.eq_registry import element_with_derivs
    ens = ["@en{}".format(k) for k in range(dim + 1)]           # the element's nodes (en0/en1 = the edge)
    comps = ["ElectricField_{}".format(c) for c in "xyz"[:dim]]
    ds.element_from_edge_model(device=device, region=region, edge_model="ElectricField")
    ds.element_from_edge_model(device=device, region=region, edge_model="ElectricField",
                               derivative="Potential")
    if ni_node:
        ds.element_from_node_model(device=device, region=region, node_model=n_i)
    pot = ["Potential" + e for e in ens]
    element_with_derivs(device, region, "BTBT_Emag",
                        "pow({} + 1.0, 0.5)".format(" + ".join(c + "^2" for c in comps)), pot)
    # DEVSIM's element assembler demands the derivative w.r.t. EVERY element node of EVERY solution
    # variable of the equation (MEASURED: "Model: ...:Electrons@en1 MISSING" is FATAL), so the carrier
    # derivatives are created on all ens, most of them simplifying to the literal 0.
    wrt = pot + ["Electrons" + e for e in ens] + ["Holes" + e for e in ens]
    shares, pairs = [], []
    for k in (0, 1):                                          # node0 / node1 share of each element edge
        nd = "@en{}".format(k)
        u = btbt_rate_expr(params, emag="BTBT_Emag",
                           d=hurkx_d_expr("Electrons" + nd, "Holes" + nd, ni_at(nd)))
        element_with_derivs(device, region, "BTBT_U{}".format(k), u, wrt)
        shares.append("BTBT_U{}".format(k))
        pairs.append(btbt_rate_expr(params, emag="BTBT_Emag",
                                    d=hurkx_d_expr("Electrons" + nd, "Holes" + nd, ni_at(nd), pair=True)))
    _BP.add_recombination_term(device, region, "btbt", tuple(shares), assembly="element",
                               pair=tuple(pairs))


def btbt_generation_current(device: str, region: str, *, depth_m: Optional[float] = None) -> float:
    """q * Int G_BTBT dV over the solved region [A] (G = -U: positive = net GENERATION), with the SAME
    quadrature weights the equations assemble (so in steady state it equals the BTBT share of the
    terminal current). 1-D: per unit area [A/m^2]; 2-D layered: per unit depth unless depth_m is given.
    Requires setup_btbt on the region."""
    from dynameta.carriers.devsim_env import import_devsim
    ds = import_devsim()
    if depth_m is not None and not (depth_m > 0.0):
        raise ValueError("depth_m must be > 0 (or None to keep the mesh's native units)")
    dim = int(ds.get_dimension(device=device))
    if dim == 1:
        u = np.asarray(ds.get_edge_model_values(device=device, region=region, name="BTBT_U"))
        w = 2.0 * np.asarray(ds.get_edge_model_values(device=device, region=region,
                                                      name="EdgeNodeVolume"))
        integral = float(np.sum(u * w))
    else:
        u0 = np.asarray(ds.get_element_model_values(device=device, region=region, name="BTBT_U0"))
        u1 = np.asarray(ds.get_element_model_values(device=device, region=region, name="BTBT_U1"))
        w = np.asarray(ds.get_element_model_values(device=device, region=region,
                                                   name="ElementNodeVolume"))
        integral = float(np.sum((u0 + u1) * w))
    return -Q_E * integral * (float(depth_m) if depth_m is not None else 1.0)


# ------------------------------------------------------------------------------------------------
# gate-oxide tunneling current (post-processed)
# ------------------------------------------------------------------------------------------------

TUNNEL_MODELS = ("fn", "schuegraf_hu", "tsu_esaki")
TUNNEL_COUPLINGS = ("post", "self_consistent")


@dataclass(frozen=True)
class DielectricBarrier:
    """One dielectric of a self-consistent tunnelling stack (v0.15, user decision 13): its FITTED
    conduction-band barrier height `barrier_eV` against the NAMED semiconductor material `against`, and
    its tunnelling mass. On a device whose semiconductor IS `against` the barrier is used as given (it is
    a fitted number, never re-derived from affinities -- so a user chi of 4.17 eV for Si cannot silently
    turn the 3.1 eV fit into 3.22 eV). On any OTHER semiconductor the barrier is carried over by Anderson's
    affinity rule, chi_d = chi_against - barrier_eV and offset = chi_s - chi_d, which needs BOTH affinities:
    a None chi refuses (plan 5.4) -- there is no neutral default for a band offset."""
    barrier_eV: float
    against: str
    m_tunnel_over_m0: float

    def __post_init__(self):
        if not (self.barrier_eV > 0.0 and self.m_tunnel_over_m0 > 0.0):
            raise ValueError("DielectricBarrier: barrier_eV and m_tunnel_over_m0 must be > 0")
        if not (isinstance(self.against, str) and self.against):
            raise ValueError("DielectricBarrier.against must name the semiconductor material the barrier "
                             "was fitted against (e.g. 'Si')")

    def offset_eV(self, semiconductor: str, chi_eV: Optional[Mapping[str, Optional[float]]] = None) -> float:
        """The conduction-band offset of this dielectric above `semiconductor`'s conduction edge [eV]."""
        if semiconductor == self.against:
            return float(self.barrier_eV)
        chi = dict(chi_eV or {})
        c_ref, c_s = chi.get(self.against), chi.get(semiconductor)
        if c_ref is None or c_s is None:
            raise ValueError(
                "DielectricBarrier: the barrier ({} eV) is quoted against {!r}, but the tunnelling "
                "semiconductor is {!r}; carrying it over needs both electron affinities (Anderson rule, "
                "chi_dielectric = chi_{} - barrier) and chi[{!r}] = {!r}, chi[{!r}] = {!r}. There is no "
                "neutral default for a band offset: give both chi_eV, or quote the barrier against {!r}."
                .format(self.barrier_eV, self.against, semiconductor, self.against, self.against, c_ref,
                        semiconductor, c_s, semiconductor))
        return float(c_s) - (float(c_ref) - float(self.barrier_eV))


# SiO2 on Si, the Lenzlinger-Snow FN fit (J. Appl. Phys. 40, 278 (1969)): phi_b = 3.1 eV, m_ox = 0.42 m0.
# It ABSORBS image-force lowering (user decision 15: no image force is added -- it would double-count
# ~0.26 V at 1 MV/cm, sqrt(q E / (4 pi eps_opt eps0)) with eps_opt = 2.13).
SIO2_ON_SI = DielectricBarrier(barrier_eV=3.1, against="Si", m_tunnel_over_m0=0.42)


@dataclass(frozen=True)
class OxideTunnelingParams:
    """Gate-oxide tunneling settings (module header). SiO2 defaults: phi_b = 3.1 eV, m_ox = 0.42 m0 (the
    Lenzlinger-Snow fit). model in TUNNEL_MODELS; "tsu_esaki" REQUIRES E_F_eV (the cathode Fermi energy
    above its supply band bottom -- no default, it is a property of the stack) and takes
    m_supply_over_m0 / T_K. "fn" is valid only for V_ox >= phi_b; below it the call warns (use
    "tsu_esaki" for a thin oxide in direct tunneling).

    WHICH COUPLING (v0.15, plan item 4.1.1):
      coupling="post" (default): the POST-PROCESSOR (oxide_tunneling_current on the solved ideal-
        capacitor state; one-sided FN / S-H / TE). Right for the static leakage of a gate whose current is
        small against the transport currents. It never removes electrons from the body, gives a nonzero
        current at equilibrium, and puts no current on the gate terminal.
      coupling="self_consistent": the current enters the Newton (setup_oxide_tunneling): a sink on the
        electron row of every semiconductor interface node and, when the gate is circuit-driven, the
        source on its KCL row -- two-sided Tsu-Esaki with WKB through the dielectric STACK, supplied by
        the local electron quasi-Fermi level. Use it for deep depletion / floating bodies whose minority
        supply IS the tunnelling, heavy injection, terminal-current balance, ssac and noise. Requires
        model="tsu_esaki" -- FN and Schuegraf-Hu are one-sided limits that violate detailed balance
        (a nonzero equilibrium current, and an FDT violation in noise), so they are REFUSED here (user
        decision 10) and stay available as post-processors only.
    Self-consistent fields (ignored by "post"):
      barriers   a DielectricBarrier for every dielectric of the column, or ONE applied to all (required);
                 e.g. SIO2_ON_SI.
      gate_fermi_energy_eV   the metal gate's Fermi energy above its band bottom [eV] (required, no
                 default; ~5-11 eV for common metals). Only used to check that the gate band bottom stays
                 below the semiconductor conduction edge (else refused: electrons only, from E_c,s up).
      gate_work_function_eV  optional: the gate work function [eV] as a CHECK against the work function
                 the gate contact's potential offset implies (the barrier is always READ from the contact,
                 user decision 12); needs the semiconductor chi. See gate_offset_for_work_function.
      energy_max_eV / energy_step_kT   the FIXED energy grid (reliability.leakage.tunnel_energy_grid):
                 eps in [0, energy_max_eV] above E_c,s in panels of energy_step_kT * kT.
      E_F_eV must stay None (the supply is the solved n_s) and T_K at its default (the region V_t is
      used); phi_b_eV / m_ox_over_m0 are the post-processor's and are not read."""
    model: str = "fn"
    phi_b_eV: float = 3.1
    m_ox_over_m0: float = 0.42
    E_F_eV: Optional[float] = None
    m_supply_over_m0: float = 1.0
    T_K: float = T_REF
    coupling: str = "post"
    barriers: Optional[object] = None
    gate_fermi_energy_eV: Optional[float] = None
    gate_work_function_eV: Optional[float] = None
    energy_max_eV: float = 4.0
    energy_step_kT: float = 0.125

    def __post_init__(self):
        if self.model not in TUNNEL_MODELS:
            raise ValueError("OxideTunnelingParams.model must be one of {} (got {!r})".format(
                TUNNEL_MODELS, self.model))
        if self.coupling not in TUNNEL_COUPLINGS:
            raise ValueError("OxideTunnelingParams.coupling must be one of {} (got {!r})".format(
                TUNNEL_COUPLINGS, self.coupling))
        if not (self.phi_b_eV > 0.0 and self.m_ox_over_m0 > 0.0 and self.m_supply_over_m0 > 0.0
                and self.T_K >= 0.0):
            raise ValueError("OxideTunnelingParams: phi_b_eV, m_ox_over_m0, m_supply_over_m0 must be "
                             "> 0 and T_K >= 0")
        if self.coupling == "self_consistent":
            self._check_self_consistent()
            return
        if self.model == "tsu_esaki" and not (self.E_F_eV is not None and self.E_F_eV > 0.0):
            raise ValueError("OxideTunnelingParams(model='tsu_esaki') needs E_F_eV > 0 (the cathode "
                             "supply Fermi energy; a metal gate ~3-11 eV, an accumulated Si surface "
                             "~0.05-0.2 eV) -- there is no safe default")

    def _check_self_consistent(self):
        if self.model != "tsu_esaki":
            raise ValueError(
                "OxideTunnelingParams(coupling='self_consistent'): model={!r} is REFUSED -- Fowler-Nordheim "
                "and Schuegraf-Hu are ONE-SIDED (cathode-only) limits: fed into the Newton they drive a "
                "current at equilibrium (no detailed balance) and their noise violates the fluctuation-"
                "dissipation theorem. Use model='tsu_esaki' (two-sided, zero at equilibrium by "
                "construction); FN / S-H remain available as post-processors (coupling='post')."
                .format(self.model))
        if self.E_F_eV is not None:
            raise ValueError("OxideTunnelingParams(coupling='self_consistent'): E_F_eV must be None -- the "
                             "semiconductor supply is the SOLVED interface electron density (its quasi-Fermi "
                             "level), not a fixed cathode Fermi energy")
        if self.T_K != T_REF:
            raise ValueError("OxideTunnelingParams(coupling='self_consistent'): T_K must stay at its default; "
                             "the temperature is the semiconductor region's V_t (BipolarOptions.T_K)")
        b = self.barriers
        ok = isinstance(b, DielectricBarrier) or (
            isinstance(b, Mapping) and len(b) > 0 and all(isinstance(v, DielectricBarrier) for v in b.values()))
        if not ok:
            raise ValueError("OxideTunnelingParams(coupling='self_consistent') needs barriers: a "
                             "DielectricBarrier (e.g. SIO2_ON_SI) or a {dielectric region: DielectricBarrier} "
                             "mapping (user decision 13: a fitted barrier against a NAMED semiconductor)")
        if not (self.gate_fermi_energy_eV is not None and self.gate_fermi_energy_eV > 0.0):
            raise ValueError("OxideTunnelingParams(coupling='self_consistent') needs gate_fermi_energy_eV > 0 "
                             "(the metal gate's Fermi energy above its band bottom, ~5-11 eV; no default)")
        if not (self.energy_max_eV > 0.0 and self.energy_step_kT > 0.0):
            raise ValueError("OxideTunnelingParams: energy_max_eV and energy_step_kT must be > 0")

    def barrier_for(self, region: str) -> DielectricBarrier:
        """The DielectricBarrier of dielectric `region` (self-consistent mode)."""
        if isinstance(self.barriers, DielectricBarrier):
            return self.barriers
        if region not in self.barriers:
            raise ValueError("OxideTunnelingParams.barriers has no entry for dielectric region {!r} (have {})"
                             .format(region, sorted(self.barriers)))
        return self.barriers[region]

    def current_density(self, V_ox_V, t_ox_m: float):
        """|J| [A/m^2] for oxide drop(s) V_ox_V >= 0 across thickness t_ox_m (dispatch to the shared
        leakage kernels)."""
        from dynameta.reliability import leakage as LK
        V = np.asarray(V_ox_V, dtype=np.float64)
        if self.model == "fn":
            return LK.fowler_nordheim_current(V / float(t_ox_m), m_ox_over_m0=self.m_ox_over_m0,
                                              phi_b_eV=self.phi_b_eV)
        if self.model == "schuegraf_hu":
            return LK.schuegraf_hu_current(V, t_ox_m, m_ox_over_m0=self.m_ox_over_m0,
                                           phi_b_eV=self.phi_b_eV)
        return LK.tsu_esaki_current(V, t_ox_m, phi_b_eV=self.phi_b_eV, E_F_eV=float(self.E_F_eV),
                                    m_ox_over_m0=self.m_ox_over_m0,
                                    m_supply_over_m0=self.m_supply_over_m0, T_K=self.T_K)


def _interface_between(device: str, region_a: str, region_b: str) -> str:
    from dynameta.carriers.devsim_env import import_devsim
    ds = import_devsim()
    hits = [i for i in ds.get_interface_list(device=device)
            if set(ds.get_region_list(device=device, interface=i)) == {region_a, region_b}]
    if len(hits) != 1:
        raise ValueError("oxide_tunneling_current: expected exactly one interface between {!r} and "
                         "{!r}, found {}".format(region_a, region_b, hits))
    return hits[0]


def oxide_tunneling_current(device: str, oxide_region: str, semiconductor_region: str,
                            params: OxideTunnelingParams, *,
                            depth_m: Optional[float] = None) -> Dict[str, object]:
    """Post-process the gate-oxide tunneling current through `oxide_region` from the solved potential
    (module header for the method, sign and units). Returns a dict:
      I              total current through the oxide/semiconductor interface (sign: module header)
      J_A_m2         signed current density at each interface node [A/m^2]
      V_ox_V         |oxide drop| through the oxide column above each interface node [V]
      E_ox_V_m       V_ox / t_ox [V/m] (the uniform field of a charge-free planar oxide)
      area           the SurfaceArea weight of each node (1-D: 1; 2-D: m; 3-D: m^2)
      t_ox_m, normal_axis, interface, model.
    Requires a PLANAR interface normal to a coordinate axis (both builders' stacks are).
    With params.coupling == "self_consistent" (v0.15) this returns the LIVE term's own kernel call instead
    (the same keys plus coupling, I_gate, J_fwd/J_bwd, S, n_s, gate offset / barrier / work function), and
    checks params.gate_work_function_eV against the contact offset when given."""
    from dynameta.carriers.devsim_env import import_devsim
    ds = import_devsim()
    if not isinstance(params, OxideTunnelingParams):
        raise TypeError("oxide_tunneling_current: params must be an OxideTunnelingParams")
    if depth_m is not None and not (depth_m > 0.0):
        raise ValueError("depth_m must be > 0 (or None to keep the mesh's native units)")
    if params.coupling == "self_consistent":                 # v0.15: the live term's own kernel call
        return _self_consistent_report(device, oxide_region, semiconductor_region, params, depth_m)
    dim = int(ds.get_dimension(device=device))
    itf = _interface_between(device, oxide_region, semiconductor_region)
    iface = _interface_node_indices(device, itf, oxide_region)

    def node(name):
        return np.asarray(ds.get_node_model_values(device=device, region=oxide_region, name=name),
                          dtype=np.float64)

    X = np.column_stack([node(c) for c in "xyz"[:dim]])
    psi = node("Potential")
    # planar interface: exactly one axis along which every interface node sits at the same coordinate
    span = np.ptp(X, axis=0)
    ispan = np.ptp(X[iface], axis=0)
    flat = [0] if dim == 1 else [k for k in range(dim)
                                 if ispan[k] <= 1e-9 * max(float(span.max()), 1e-30)]
    if len(flat) != 1:
        raise NotImplementedError("oxide_tunneling_current: the {}/{} interface is not planar and "
                                  "normal to a coordinate axis (per-axis spread {}); only layered "
                                  "stacks are supported".format(oxide_region, semiconductor_region,
                                                                ispan.tolist()))
    ax = flat[0]
    z_if = float(np.mean(X[iface, ax]))
    t_ox = float(span[ax])
    far_z = float(X[:, ax].max()) if float(np.mean(X[:, ax])) > z_if else float(X[:, ax].min())
    far = np.nonzero(np.abs(X[:, ax] - far_z) <= 1e-9 * t_ox)[0]  # the opposite (gate-side) face
    # COLUMN DROP: V_col = psi(opposite face, same lateral position) - psi(interface node), with the
    # opposite face interpolated piecewise-linearly in the lateral coordinate(s). Exact for any linear
    # potential (a charge-free planar oxide) on any mesh, and -- unlike the nodal gradient at the
    # interface -- it does not amplify the FV solution's lateral noise on an unstructured 3-D mesh:
    # MEASURED on a laterally-uniform 3-D Si/SiO2 stack (4 nm cell, mesh_min 0.5 nm): the per-node
    # interface gradient varied 17.7 % across the interface, the column drop 1.5 %.
    lat = [k for k in range(dim) if k != ax]
    if not lat:
        psi_far = np.full(iface.size, float(np.mean(psi[far])))
    elif len(lat) == 1:
        o = np.argsort(X[far, lat[0]])
        psi_far = np.interp(X[iface, lat[0]], X[far, lat[0]][o], psi[far][o])
    else:
        from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
        pts = X[far][:, lat]
        psi_far = LinearNDInterpolator(pts, psi[far])(X[iface][:, lat])
        miss = ~np.isfinite(psi_far)                         # hull round-off at the cell boundary
        if np.any(miss):
            psi_far[miss] = NearestNDInterpolator(pts, psi[far])(X[iface][miss][:, lat])
    G = psi_far - psi[iface]                                   # > 0: the gate side is at the higher potential
    V_ox = np.abs(G)
    E_ox = V_ox / t_ox
    if params.model == "fn" and bool(np.any((V_ox > 0.0) & (V_ox < params.phi_b_eV))):
        warnings.warn(
            "oxide_tunneling_current(model='fn'): {} of {} interface nodes have V_ox below the barrier "
            "(max V_ox = {:.3g} V < phi_b = {:.3g} eV) -- Fowler-Nordheim is the TRIANGULAR-barrier limit "
            "and under-counts direct tunneling there (by orders of magnitude for a few-nm oxide). Use "
            "model='tsu_esaki' (or 'schuegraf_hu' at moderate V_ox).".format(
                int(np.count_nonzero((V_ox > 0.0) & (V_ox < params.phi_b_eV))), V_ox.size,
                float(V_ox.max()), params.phi_b_eV), AdvisoryWarning, stacklevel=2)
    J = np.sign(G) * np.asarray(params.current_density(V_ox, t_ox), dtype=np.float64)
    area = node("SurfaceArea")[iface]
    total = float(np.sum(J * area)) * (float(depth_m) if depth_m is not None else 1.0)
    return {"I": total, "J_A_m2": J, "E_ox_V_m": E_ox, "V_ox_V": V_ox, "area": area,
            "t_ox_m": t_ox, "normal_axis": ax, "interface": itf, "model": params.model}


# ------------------------------------------------------------------------------------------------
# v0.15: SELF-CONSISTENT gate-oxide tunnelling (plan item 4.1.1; module header section
# "SELF-CONSISTENT OXIDE TUNNELLING"; audit note docs/audit/2026-11-06-v0.15-oxide-tunnelling.md)
# ------------------------------------------------------------------------------------------------

_EQ_N = "ElectronContinuityEquation"
_HOLE_WARN_FIELD_V_M = 1e8          # 1 MV/cm: above it hole / valence-band tunnelling can matter (decision 14)
_TAIL_WARN = 1e-8                   # integrand at the grid top / one-way sum: the grid is too short
_REPORT_TAIL_WARN = 1e-12


def _interface_node_indices(device: str, itf: str, region: str) -> np.ndarray:
    """Node indices (in `region`) of the nodes of interface `itf` (the post-processor's lookup)."""
    from dynameta.carriers.devsim_env import import_devsim
    ds = import_devsim()
    side = "r0" if ds.get_region_list(device=device, interface=itf)[0] == region else "r1"
    nm = "_tun_oxide_idx_" + side
    if nm not in ds.get_interface_model_list(device=device, interface=itf):
        ds.interface_model(device=device, interface=itf, name=nm, equation="node_index@" + side)
    return np.asarray(ds.get_interface_model_values(device=device, interface=itf, name=nm)).astype(int)


def _coords(device: str, region: str) -> np.ndarray:
    from dynameta.carriers.devsim_env import import_devsim
    ds = import_devsim()
    dim = int(ds.get_dimension(device=device))
    return np.column_stack([np.asarray(ds.get_node_model_values(device=device, region=region, name=c),
                                       dtype=np.float64) for c in "xyz"[:dim]])


def _has_carriers(device: str, region: str) -> bool:
    from dynameta.carriers.devsim_env import import_devsim
    ds = import_devsim()
    return "Electrons" in ds.get_node_model_list(device=device, region=region)


def _lateral_weights(face_lat: np.ndarray, query_lat: np.ndarray, tol: float):
    """Piecewise-linear interpolation weights of face nodes at the query points' lateral positions:
    a list (per query) of (face node indices, weights). A coincident node (within tol) takes weight 1
    exactly -- the tensor-mesh case, where every interface node has its column partner."""
    n = query_lat.shape[0]
    if face_lat.shape[1] == 0:                               # 1-D: one face node
        if face_lat.shape[0] != 1:
            raise ValueError("oxide tunnelling: a 1-D face must have exactly one node")
        return [(np.array([0]), np.array([1.0])) for _ in range(n)]
    out = []
    from scipy.spatial import cKDTree
    dnear, near = cKDTree(face_lat).query(query_lat)
    if face_lat.shape[1] == 1:
        o = np.argsort(face_lat[:, 0])
        xs = face_lat[o, 0]
        for i in range(n):
            if dnear[i] <= tol:
                out.append((np.array([near[i]]), np.array([1.0])))
                continue
            q = query_lat[i, 0]
            j = int(np.clip(np.searchsorted(xs, q) - 1, 0, xs.size - 2))
            t = (q - xs[j]) / (xs[j + 1] - xs[j])
            if not (-1e-12 <= t <= 1.0 + 1e-12):
                raise ValueError("oxide tunnelling: an interface node lies outside the lateral extent of "
                                 "the far face (patch gate?) -- not supported")
            out.append((np.array([o[j], o[j + 1]]), np.array([1.0 - t, t])))
        return out
    from scipy.spatial import Delaunay
    tri = None
    for i in range(n):
        if dnear[i] <= tol:
            out.append((np.array([near[i]]), np.array([1.0])))
            continue
        if tri is None:
            tri = Delaunay(face_lat)
        s = int(tri.find_simplex(query_lat[i]))
        if s < 0:
            raise ValueError("oxide tunnelling: an interface node lies outside the lateral hull of the far "
                             "face (patch gate?) -- not supported")
        T = tri.transform[s]
        b = T[:2].dot(query_lat[i] - T[2])
        w = np.array([b[0], b[1], 1.0 - b[0] - b[1]])
        out.append((tri.simplices[s].copy(), w))
    return out


def gate_offset_for_work_function(work_function_eV: float, chi_s_eV: float, phi_c_V: float) -> float:
    """The gate-contact potential offset [V] (physics_equilibrium.setup_contact(psi_offset_V=...)) that
    gives a metal gate the work function W_m, on a semiconductor of affinity chi_s whose conduction edge
    sits at E_c = q (phi_c - psi) in the device frame: offset = chi_s + phi_c - W_m. Derivation: the gate
    Fermi level is -q V_g, the gate vacuum level q(phi_c - psi_g) + chi_s with psi_g = V_g + offset."""
    return float(chi_s_eV) + float(phi_c_V) - float(work_function_eV)


class OxideTunnelTerm(CustomTerm):
    """The self-consistent gate-oxide tunnelling term (setup_oxide_tunneling builds it). Nonlocal: the
    current at interface node i depends on the potential of the far faces of every dielectric of its
    column. Residual: -A_i J_i on the electron row of node i; +Sum_i A_i J_i on the gate's KCL row
    when the gate is circuit-driven. Jacobian: exact derivative of the discrete sum (fixed grid)."""

    locality = "nonlocal"
    jacobian_exact = True
    pattern_changes = False

    def __init__(self, name, **kw):
        super().__init__(name)
        self.__dict__.update(kw)
        self.active = True
        self._tail_warned = False
        self.last = None

    @property
    def gate_node(self):
        """The gate's circuit node, read LIVE from the equation registry (None for a bias-parameter gate):
        a gate repointed circuit-driven after installation (LayeredDevsimBuilder.set_ssac_gate) is picked
        up at the next assembly."""
        from dynameta.carriers import eq_registry as _R
        return _R.circuit_driven_contacts(self.device).get(self.gate)

    # ---- protocol ----
    def equations(self, device):
        eqs = [(self.semi, _EQ_N)]
        if self.gate_node is not None:
            eqs.append((None, self.gate_node))
        return eqs

    def _columns(self, device):
        from dynameta.carriers import custom_terms as CT
        rows = CT.node_rows(device, self.semi, _EQ_N, self.iface)
        cP = CT.node_columns(device, self.semi, "Potential", self.iface)
        cN = CT.node_columns(device, self.semi, "Electrons", self.iface)
        fcols = [CT.node_columns(device, reg, "Potential", nodes) for reg, nodes, _ in self.faces]
        g = CT.circuit_row(self.gate_node) if self.gate_node is not None else None
        return rows, cP, cN, fcols, g

    def structural_pattern(self, device):
        rows, cP, cN, fcols, g = self._columns(device)
        pat = set()
        for i, r in enumerate(rows):
            cols = [cP[i], cN[i]] + [fc[j] for (_, _, W), fc in zip(self.faces, fcols)
                                     for j in W[i][0]] + ([g] if g is not None else [])
            for c in cols:
                pat.add((int(r), int(c)))
                if g is not None:
                    pat.add((int(g), int(c)))
        return sorted(pat)

    def state(self, device):
        """The kernel call at the CURRENT solution (also the report's: one evaluation path)."""
        from dynameta.carriers.devsim_env import import_devsim
        from dynameta.reliability.leakage import tsu_esaki_two_sided
        ds = import_devsim()

        def nv(reg, name):
            return np.asarray(ds.get_node_model_values(device=device, region=reg, name=name),
                              dtype=np.float64)
        psi_s = nv(self.semi, "Potential")[self.iface]
        n_s = nv(self.semi, "Electrons")[self.iface]
        L = nv(self.semi, self.activity)[self.iface]
        dL = nv(self.semi, self.activity + ":Electrons")[self.iface]
        d = np.empty((self.iface.size, len(self.faces)))
        for k, (reg, nodes, W) in enumerate(self.faces):
            pf = nv(reg, "Potential")[nodes]
            d[:, k] = psi_s - np.array([float(np.dot(w, pf[j])) for j, w in W])
        if self.gate_node is not None:
            Vg = float(ds.get_circuit_node_value(node=self.gate_node, solution="dcop"))
        else:
            Vg = float(ds.get_parameter(device=device, name=self.gate_param))
        vt = self.V_t
        x_s = L
        x_g = (psi_s - self.phi_c - Vg) / vt
        bottom = x_g * self.grid.kT_J - self.gate_fermi_J        # gate band bottom - E_c,s [J]
        if np.any(bottom >= 0.0):
            raise RuntimeError(
                "oxide tunnelling {!r}: the gate's band bottom (E_Fg - {:.4g} eV) rises above the "
                "semiconductor conduction edge at {} interface node(s) (max {:.4g} eV above): electrons "
                "below the gate band bottom cannot tunnel, which the fixed grid starting at E_c,s does "
                "not model -- refused (gate_fermi_energy_eV too small or an extreme gate bias)".format(
                    self.name, self.gate_fermi_J / Q_E, int(np.count_nonzero(bottom >= 0.0)),
                    float(np.max(bottom)) / Q_E))
        r = tsu_esaki_two_sided(self.grid, x_s, x_g, d, self.layers, m_supply_over_m0=self.m_supply)
        if not self._tail_warned and float(np.max(r.tail)) > _TAIL_WARN:
            self._tail_warned = True
            warnings.warn(
                "oxide tunnelling {!r}: the energy grid [0, {:.3g} eV] truncates the supply integral "
                "(integrand at its top = {:.2e} of the one-way sum); raise energy_max_eV".format(
                    self.name, self.grid.eps_max_J / Q_E, float(np.max(r.tail))),
                AdvisoryWarning, stacklevel=2)
        st = {"psi_s": psi_s, "n_s": n_s, "dL": dL, "d": d, "V_g": Vg, "x_s": x_s, "x_g": x_g, "te": r}
        self.last = st
        return st

    def assemble(self, device, what, timemode):
        if timemode != "DC" or not self.active:
            return [], []
        st = self.state(device)
        r = st["te"]
        rows, cP, cN, fcols, g = self._columns(device)
        A = self.area
        vt = self.V_t
        dpsi = r.dJ_dxg / vt + np.sum(r.dJ_dd, axis=1)
        dn = r.dJ_dxs * st["dL"]
        R, C, V = [], [], []
        for i in range(rows.size):
            ents = [(int(cP[i]), dpsi[i]), (int(cN[i]), dn[i])]
            for k, ((_, _, W), fc) in enumerate(zip(self.faces, fcols)):
                for j, w in zip(*W[i]):
                    ents.append((int(fc[j]), -r.dJ_dd[i, k] * float(w)))
            if g is not None:
                ents.append((int(g), -r.dJ_dxg[i] / vt))
            for c, dJ in ents:
                R.append(int(rows[i]))
                C.append(c)
                V.append(-A[i] * float(dJ))
                if g is not None:
                    R.append(int(g))
                    C.append(c)
                    V.append(A[i] * float(dJ))
        rcv = [x for t in zip(R, C, V) for x in t]
        rv = [x for t in zip([int(x) for x in rows], (-A * r.J).tolist()) for x in t]
        if g is not None:
            rv += [int(g), float(np.sum(A * r.J))]
        return rcv, rv

    def noise_sources(self, device):
        """The tunnel shot noise as ONE two-point source per interface node (mechanism tunnel_ox):
        an event moves one electron from node i's electron row (c = -1) to the gate's circuit row
        (c = +1; omitted for a bias-parameter gate, whose potential is fixed). Rate P = A S / (2 q^2),
        S = 2 q J coth(Delta/2) (reliability.leakage.tsu_esaki_two_sided): the exact T << 1 partition,
        the Nyquist value at equilibrium."""
        from dynameta.carriers.noise_analysis import SourceRow, TwoPointSource
        st = self.state(device)
        P = self.area * st["te"].S / (2.0 * Q_E * Q_E)
        n = self.iface.size
        rows = [SourceRow(self.semi, _EQ_N, -np.ones(n), self.iface.copy())]
        if self.gate_node is not None:
            rows.append(SourceRow(None, self.gate_node, np.ones(n), None))
        return [TwoPointSource("tunnel_ox", self.semi, P, tuple(rows), self.iface.copy())]


def dielectric_column(device: str, semiconductor_region: str, gate_contact: str) -> Optional[List[str]]:
    """The dielectric regions from `semiconductor_region` to the region of `gate_contact`, in order (the
    shortest chain of carrier-free regions joined by interfaces), or None when there is none. Used by
    setup_oxide_tunneling and the builders to find which semiconductor a gate tunnels to."""
    from dynameta.carriers.devsim_env import import_devsim
    ds = import_devsim()
    if gate_contact not in ds.get_contact_list(device=device):
        return None
    target = ds.get_region_list(device=device, contact=gate_contact)[0]
    if _has_carriers(device, target):
        return None
    adj: Dict[str, List[str]] = {}
    for i in ds.get_interface_list(device=device):
        rr = list(ds.get_region_list(device=device, interface=i))
        for r in rr:
            adj.setdefault(r, []).extend(x for x in rr if x != r)
    prev = {semiconductor_region: None}
    queue = [semiconductor_region]
    while queue:
        cur = queue.pop(0)
        for r in sorted(adj.get(cur, ())):
            if r in prev or _has_carriers(device, r):
                continue
            prev[r] = cur
            if r == target:
                out = [r]
                while prev[out[-1]] != semiconductor_region:
                    out.append(prev[out[-1]])
                return out[::-1]
            queue.append(r)
    return None


def setup_oxide_tunneling(device: str, semiconductor_region: str, gate_contact: str,
                          params: OxideTunnelingParams, *, dielectric_regions: Optional[Sequence[str]] = None,
                          semiconductor_material: Optional[str] = None,
                          chi_eV: Optional[Mapping[str, Optional[float]]] = None,
                          phi_c_V: Optional[float] = None, name: Optional[str] = None):
    """Install SELF-CONSISTENT gate-oxide tunnelling (module header; plan item 4.1.1) from
    `semiconductor_region` through its dielectric column to the metal `gate_contact`, as a custom term
    (carriers.custom_terms). Call after the regions, interfaces and contacts exist (before or after the
    first solve). Returns the term (its .state(device) is the kernel call at the current solution).

    params      OxideTunnelingParams(model="tsu_esaki", coupling="self_consistent", barriers=...,
                gate_fermi_energy_eV=...). m_supply_over_m0 is the semiconductor's transverse mass.
    dielectric_regions  the dielectric regions from the semiconductor to the gate, in order; None walks
                the interfaces (refused if ambiguous).
    semiconductor_material / chi_eV   the material name compared with each DielectricBarrier.against
                (default: DEVSIM's material of the region) and {material: chi_eV} for the Anderson
                carry-over (only needed when they differ, user decision 13).
    phi_c_V     the semiconductor conduction edge E_c = q (phi_c - psi) in the device frame; default the
                bipolar own frame V_t ln(N_dos/n_i) (a heterostack builder passes its Anderson frame).

    REQUIRES a bipolar drift-diffusion semiconductor (physics_bipolar_dd: the sink needs an electron
    continuity row; the equilibrium / unipolar physics are refused), a planar column normal to a
    coordinate axis, and a METAL gate covering the far face (a semiconductor -- poly-Si -- gate is
    REFUSED, user decision 17). Refused also: bandgap narrowing and density gradient on the region (the
    band edge the oxide offset hangs on would move; v0.15 scope). Electrons only (decision 14): the report
    warns where holes would matter. The gate is circuit-driven (ac_analysis.setup_circuit_contact) -> the
    current also enters its KCL row and ssac / noise see it; a bias-parameter gate gets NO equation entry
    (DEVSIM's gate contact current stays the displacement-only 0 in DC) and the gate current is the
    module-reported I_gate (oxide_tunneling_gate_currents; contact_current.extract_contact_currents adds
    it). This never CREATES a circuit (DEVSIM circuits cannot be removed, decision 11)."""
    from dynameta.carriers import custom_terms as CT
    from dynameta.carriers import eq_registry as _R
    from dynameta.carriers.devsim_env import import_devsim
    from dynameta.carriers.noise_analysis import _contact_nodes
    from dynameta.carriers.physics_heterojunction import setup_log_activity
    from dynameta.reliability.leakage import TunnelLayer, tunnel_energy_grid
    ds = import_devsim()
    if not isinstance(params, OxideTunnelingParams) or params.coupling != "self_consistent":
        raise ValueError("setup_oxide_tunneling: needs OxideTunnelingParams(coupling='self_consistent', "
                         "model='tsu_esaki', ...)")
    semi = semiconductor_region
    if semi not in ds.get_region_list(device=device):
        raise ValueError("setup_oxide_tunneling: no region {!r}".format(semi))
    for eq in (_EQ_N, "HoleContinuityEquation"):
        if not _R.recorded(device, eq, scope="region", loc=semi):
            raise ValueError("setup_oxide_tunneling: region {!r} has no {} -- the self-consistent sink needs "
                             "a bipolar drift-diffusion semiconductor (physics_bipolar_dd; physics="
                             "'bipolar_dd'); the equilibrium / unipolar physics are refused".format(semi, eq))
    models = set(ds.get_node_model_list(device=device, region=semi))
    if "BgnDc" in models:
        raise NotImplementedError("setup_oxide_tunneling: region {!r} carries bandgap narrowing; the oxide "
                                  "offset is quoted against the unnarrowed conduction edge -- not modelled in "
                                  "v0.15".format(semi))
    if "ElectronCurrentDG" in set(ds.get_edge_model_list(device=device, region=semi)):
        raise NotImplementedError("setup_oxide_tunneling: region {!r} uses the density-gradient model; its "
                                  "quantum potential shifts the supply -- not modelled in v0.15".format(semi))
    if gate_contact not in ds.get_contact_list(device=device):
        raise ValueError("setup_oxide_tunneling: no contact {!r}".format(gate_contact))
    gate_region = ds.get_region_list(device=device, contact=gate_contact)[0]
    if _has_carriers(device, gate_region):
        raise NotImplementedError(
            "setup_oxide_tunneling: the gate contact {!r} sits on region {!r}, which carries electrons: a "
            "semiconductor (poly-Si) gate is REFUSED (user decision 17) -- its depletion and its own "
            "continuity row are not modelled; use a metal gate on the dielectric".format(gate_contact,
                                                                                         gate_region))
    # ---- the dielectric column, semiconductor side first ----
    if dielectric_regions is None:
        chain = dielectric_column(device, semi, gate_contact)
        if chain is None:
            raise ValueError("setup_oxide_tunneling: no column of dielectric regions joins {!r} to the gate {!r} "
                             "(pass dielectric_regions)".format(semi, gate_contact))
    else:
        chain = list(dielectric_regions)
        if not chain or chain[-1] != gate_region:
            raise ValueError("setup_oxide_tunneling: dielectric_regions must end on the gate's region {!r}"
                             .format(gate_region))
        for r in chain:
            if _has_carriers(device, r):
                raise NotImplementedError("setup_oxide_tunneling: {!r} carries electrons (a semiconductor gate "
                                          "or layer is REFUSED, user decision 17)".format(r))
    # ---- geometry: planar faces normal to one axis ----
    dim = int(ds.get_dimension(device=device))
    Xs = _coords(device, semi)
    itf0 = _interface_between(device, semi, chain[0])
    iface_all = _interface_node_indices(device, itf0, semi)
    span_all = float(np.max(np.ptp(Xs, axis=0))) if dim > 1 else float(np.ptp(Xs[:, 0]))
    tol = 1e-9 * max(span_all, 1e-30)
    if dim == 1:
        ax = 0
    else:
        isp = np.ptp(Xs[iface_all], axis=0)
        flat = [k for k in range(dim) if isp[k] <= tol]
        if len(flat) != 1:
            raise NotImplementedError("setup_oxide_tunneling: the {}/{} interface is not planar and normal to a "
                                      "coordinate axis (spread {}); only layered stacks".format(
                                          semi, chain[0], isp.tolist()))
        ax = flat[0]
    lat = [k for k in range(dim) if k != ax]
    contact_nodes = set()
    for c in ds.get_contact_list(device=device):
        contact_nodes.update(int(i) for i in _contact_nodes(device, semi, c))
    keep = np.array([i not in contact_nodes for i in iface_all], dtype=bool)
    iface = iface_all[keep]
    if iface.size == 0:
        raise ValueError("setup_oxide_tunneling: every {}/{} interface node is a contact node".format(
            semi, chain[0]))
    area_all = np.asarray(ds.get_node_model_values(device=device, region=semi, name="SurfaceArea"))[iface_all]
    if not np.all(keep):
        warnings.warn(
            "setup_oxide_tunneling: {} of {} interface nodes of {}/{} are contact nodes (DEVSIM drops their "
            "electron rows); their {:.3g} of the interface area carries no tunnelling current".format(
                int(np.count_nonzero(~keep)), iface_all.size, semi, chain[0],
                float(np.sum(area_all[~keep]) / np.sum(area_all))), AdvisoryWarning, stacklevel=2)
    area = area_all[keep].astype(np.float64)
    z = float(np.mean(Xs[iface_all, ax]))
    faces, layers = [], []
    sem_mat = semiconductor_material if semiconductor_material is not None else ds.get_material(
        device=device, region=semi)
    prev = semi
    for j, reg in enumerate(chain):
        Xr = _coords(device, reg)
        near = _interface_node_indices(device, _interface_between(device, prev, reg), reg)
        if j + 1 < len(chain):
            far = _interface_node_indices(device, _interface_between(device, reg, chain[j + 1]), reg)
        else:
            far = _contact_nodes(device, reg, gate_contact)
        if dim > 1 and (np.ptp(Xr[near, ax]) > tol or np.ptp(Xr[far, ax]) > tol):
            raise NotImplementedError("setup_oxide_tunneling: dielectric {!r} has a non-planar face".format(reg))
        z_near, z_far = float(np.mean(Xr[near, ax])), float(np.mean(Xr[far, ax]))
        if abs(z_near - z) > tol or not (abs(z_far - z) > abs(z_near - z) + tol):
            raise ValueError("setup_oxide_tunneling: dielectric {!r} is not stacked along axis {} away from the "
                             "interface".format(reg, ax))
        if j + 1 == len(chain):                           # the gate must cover the far face
            onface = np.nonzero(np.abs(Xr[:, ax] - z_far) <= tol)[0]
            if not set(onface.tolist()) <= set(np.asarray(far).tolist()):
                raise NotImplementedError(
                    "setup_oxide_tunneling: the gate {!r} covers only part of the far face of {!r} (a patch "
                    "gate): the straight-column self-consistent term is not supported there".format(
                        gate_contact, reg))
        W = _lateral_weights(Xr[far][:, lat], Xs[iface][:, lat], tol)
        faces.append((reg, np.asarray(far, dtype=np.int64), W))
        b = params.barrier_for(reg)
        layers.append(TunnelLayer(b.offset_eV(sem_mat, chi_eV), abs(z_far - z_near), b.m_tunnel_over_m0))
        z = z_far
        prev = reg
    # ---- mesh quality (plan 4.1.1: the column drop on an unstructured mesh; MEASURED, audit note 5) ----
    aligned = all(len(w[0]) == 1 for (_, _, W) in faces for w in W)
    if dim > 1 and not aligned:
        from dynameta.carriers.devsim_geometry import exact_geometry_applied
        from dynameta.carriers.element_geometry import check_element_geometry
        from dynameta.warning_types import UnreliableResultWarning
        inexact = [r for r in [semi] + list(chain)
                   if check_element_geometry(device, r, raise_error=False)["n_inexact"] > 0
                   and not exact_geometry_applied(device, r)]
        if inexact:
            why = ("DEVSIM's box geometry is also inexact on {} (non-well-centred elements, "
                   "devsim_geometry.apply_exact_geometry not applied): MEASURED on the gmsh Si / 2 nm SiO2 "
                   "stack (Stacked3DSpec(mesher='gmsh'), 1.2 V) the lateral oxide-voltage spread over the "
                   "interface was 5.1 % and the per-node current spread 92 %".format(inexact))
        else:
            why = ("MEASURED on the gmsh Si / 2 nm SiO2 stack WITH devsim_geometry.apply_exact_geometry: a 0.6 % "
                   "lateral oxide-voltage spread (8.6x less than without it) still gives a 10 % per-node current "
                   "spread")
        warnings.warn(
            "setup_oxide_tunneling: the mesh is not column-aligned (an interface node has no partner node on a "
            "dielectric face above it), so the far-face potential is interpolated and the unstructured finite-"
            "volume solution's lateral noise is amplified by the tunnel exponent (MEASURED |d ln J / d ln V_ox| ~ 15-18 over the nodes); "
            "{}. Use a tensor mesh (Stacked3DSpec(mesher='tensor'), the 2-D layered builder) for quantitative "
            "currents.".format(why), UnreliableResultWarning, stacklevel=2)
    # ---- supply, frame, gate ----
    fd = float(ds.get_parameter(device=device, region=semi, name="fd_enhancement")) > 0.0
    act = setup_log_activity(device, semi, "Electrons", "N_dos", fd=fd)
    vt = float(ds.get_parameter(device=device, region=semi, name="V_t"))
    if phi_c_V is None:
        phi_c_V = vt * math.log(float(ds.get_parameter(device=device, region=semi, name="N_dos"))
                                / float(ds.get_parameter(device=device, region=semi, name="n_i")))
    gate_node = _R.circuit_driven_contacts(device).get(gate_contact)       # (re-read live by the term)
    gate_param = "{}_bias".format(gate_contact)
    if gate_node is None and gate_param not in ds.get_parameter_list(device=device):
        raise ValueError("setup_oxide_tunneling: gate {!r} is neither circuit-driven nor driven by the bias "
                         "parameter {!r} (physics_equilibrium.setup_contact)".format(gate_contact, gate_param))
    grid = tunnel_energy_grid(Q_E * vt, params.energy_max_eV * Q_E, step_kT=params.energy_step_kT)
    term = OxideTunnelTerm(name or "oxtun_{}_{}".format(semi, gate_contact),
               semi=semi, chain=tuple(chain), iface=iface, area=area, faces=faces, layers=tuple(layers),
               grid=grid, phi_c=float(phi_c_V), V_t=vt, gate=gate_contact, device=device,
               gate_param=gate_param, gate_fermi_J=float(params.gate_fermi_energy_eV) * Q_E,
               m_supply=float(params.m_supply_over_m0), activity=act, axis=ax, interface=itf0,
               params=params, semiconductor_material=sem_mat,
               chi_s=(dict(chi_eV or {}).get(sem_mat)), dim=dim)
    CT.add_custom_term(device, term)
    return term


def oxide_tunneling_terms(device: str) -> list:
    """The LIVE self-consistent oxide-tunnelling terms on `device` (custom_terms liveness: a deleted or
    re-created device has none)."""
    from dynameta.carriers import custom_terms as CT
    return [t for t in CT.live_terms(device) if isinstance(t, OxideTunnelTerm)]


def oxide_tunneling_gate_currents(device: str, *, depth_m: Optional[float] = None) -> Dict[str, float]:
    """{gate contact: I_gate} of every live self-consistent oxide-tunnelling term: the DC terminal current
    INTO the device through the gate (DEVSIM's contact-current sign), Sum_i A_i J_i. DEVSIM's own
    get_contact_current on a dielectric gate is the displacement charge only -- 0 in DC -- so every DC
    terminal-current consumer must add this (contact_current.extract_contact_currents does). Units as the
    mesh (1-D A/m^2, 2-D A/m unless depth_m, 3-D A)."""
    out: Dict[str, float] = {}
    for t in oxide_tunneling_terms(device):
        st = t.state(device)
        I = float(np.sum(t.area * st["te"].J)) * (float(depth_m) if depth_m is not None else 1.0)
        out[t.gate] = out.get(t.gate, 0.0) + I
    return out


def _self_consistent_report(device: str, oxide_region: str, semiconductor_region: str,
                            params: OxideTunnelingParams, depth_m: Optional[float]) -> Dict[str, object]:
    from dynameta.carriers.devsim_env import import_devsim
    ds = import_devsim()
    terms = [t for t in oxide_tunneling_terms(device) if t.semi == semiconductor_region and oxide_region in t.chain]
    if len(terms) != 1:
        raise ValueError("oxide_tunneling_current(coupling='self_consistent'): {} live self-consistent term(s) "
                         "from {!r} through {!r} (install it with setup_oxide_tunneling)".format(
                             len(terms), semiconductor_region, oxide_region))
    t = terms[0]
    st = t.state(device)
    te = st["te"]
    scale = float(depth_m) if depth_m is not None else 1.0
    V_stack = -st["d"][:, -1]                                 # psi_gate - psi_s (> 0: gate higher)
    t_tot = float(sum(l.thickness_m for l in t.layers))
    E_ox = np.abs(V_stack) / t_tot
    gate_offset = float(np.mean(np.asarray(ds.get_node_model_values(
        device=device, region=t.faces[-1][0], name="Potential"))[t.faces[-1][1]])) - st["V_g"]
    phi_bg = t.phi_c - gate_offset + t.layers[-1].offset_eV       # gate barrier above E_Fg [eV]
    out = {"I": float(np.sum(t.area * te.J)) * scale, "J_A_m2": te.J.copy(), "E_ox_V_m": E_ox,
           "V_ox_V": np.abs(V_stack), "area": t.area.copy(), "t_ox_m": t_tot, "normal_axis": t.axis,
           "interface": t.interface, "model": "tsu_esaki", "coupling": "self_consistent",
           "I_gate": float(np.sum(t.area * te.J)) * scale, "J_fwd_A_m2": te.J_fwd.copy(),
           "J_bwd_A_m2": te.J_bwd.copy(), "S_A2_Hz_m2": te.S.copy(), "n_s_m3": st["n_s"].copy(),
           "gate_offset_V": gate_offset, "gate_barrier_eV": phi_bg, "layers": t.layers,
           "gate_node": t.gate_node, "max_tail": float(np.max(te.tail))}
    if t.chi_s is not None:
        out["gate_work_function_eV"] = float(t.chi_s) + t.phi_c - gate_offset
    W = params.gate_work_function_eV
    if W is not None:
        if t.chi_s is None:
            raise ValueError("oxide_tunneling_current: gate_work_function_eV is a check against the contact "
                             "offset and needs the semiconductor's chi_eV (setup_oxide_tunneling(chi_eV=...))")
        if abs(out["gate_work_function_eV"] - float(W)) > 1e-6:
            raise ValueError(
                "oxide_tunneling_current: the gate contact's offset {:.6g} V implies W_m = {:.6g} eV, not the "
                "stated gate_work_function_eV = {:.6g} eV; set the gate contact's psi_offset_V to {:.6g} V "
                "(gate_offset_for_work_function) -- the barrier is READ from the contact (decision 12)".format(
                    gate_offset, out["gate_work_function_eV"], float(W),
                    gate_offset_for_work_function(W, t.chi_s, t.phi_c)))
    if out["max_tail"] > _REPORT_TAIL_WARN:
        warnings.warn("oxide_tunneling_current: the energy grid truncates the supply integral ({:.2e} of the "
                      "one-way sum at its top); raise energy_max_eV".format(out["max_tail"]),
                      AdvisoryWarning, stacklevel=3)
    p_s = np.asarray(ds.get_node_model_values(device=device, region=t.semi, name="Holes"))[t.iface]
    bad = (p_s > st["n_s"]) & (E_ox > _HOLE_WARN_FIELD_V_M)
    if np.any(bad):
        warnings.warn(
            "oxide_tunneling_current: at {} of {} interface nodes holes outnumber electrons (p_s > n_s) under "
            "|E_ox| > 1 MV/cm: hole and valence-band electron tunnelling are NOT modelled (electrons only, "
            "user decision 14) -- the gate current there may be underestimated".format(
                int(np.count_nonzero(bad)), bad.size), AdvisoryWarning, stacklevel=3)
    return out
