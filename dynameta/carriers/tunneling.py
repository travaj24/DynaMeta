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
    (volume_node0_model / volume_node1_model, weight ElementNodeVolume: sum 2*ENV = region volume
    EXACTLY) with D evaluated at THAT node -- so a uniform field integrates exactly on any mesh.
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
injection. A self-consistent interface sink is the follow-up (the decision is the user's -- see
docs/audit/2026-11-03-devsim-tunneling.md).

SIGN: J > 0 when the gate side of the oxide is at the HIGHER potential (electrons tunnel from the
semiconductor into the gate; conventional current flows gate -> semiconductor). The SAME barrier/supply
parameters serve both polarities (an asymmetric gate/semiconductor stack needs two calls).

UNITS follow the D1 contact-current convention: a 1-D device is per unit area (I in A/m^2); the 2-D
layered mesh is per unit depth (pass depth_m = period_y_m to get A); the 3-D gmsh mesh is in metres
(I already in A).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from dynameta.constants import Q_E, T_REF, V_T
from dynameta.warning_types import AdvisoryWarning

__all__ = ["BTBTParams", "SILICON_HURKX", "btbt_rate_expr", "hurkx_d_expr", "setup_btbt", "btbt_generation_current",
           "OxideTunnelingParams", "oxide_tunneling_current", "TUNNEL_MODELS"]


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


@dataclass(frozen=True)
class OxideTunnelingParams:
    """Gate-oxide tunneling post-processor settings (module header). SiO2 defaults: phi_b = 3.1 eV,
    m_ox = 0.42 m0 (the Lenzlinger-Snow fit). model in TUNNEL_MODELS; "tsu_esaki" REQUIRES E_F_eV (the
    cathode Fermi energy above its supply band bottom -- no default, it is a property of the stack)
    and takes m_supply_over_m0 / T_K. "fn" is valid only for V_ox >= phi_b; below it the call warns
    (use "tsu_esaki" for a thin oxide in direct tunneling)."""
    model: str = "fn"
    phi_b_eV: float = 3.1
    m_ox_over_m0: float = 0.42
    E_F_eV: Optional[float] = None
    m_supply_over_m0: float = 1.0
    T_K: float = T_REF

    def __post_init__(self):
        if self.model not in TUNNEL_MODELS:
            raise ValueError("OxideTunnelingParams.model must be one of {} (got {!r})".format(
                TUNNEL_MODELS, self.model))
        if not (self.phi_b_eV > 0.0 and self.m_ox_over_m0 > 0.0 and self.m_supply_over_m0 > 0.0
                and self.T_K >= 0.0):
            raise ValueError("OxideTunnelingParams: phi_b_eV, m_ox_over_m0, m_supply_over_m0 must be "
                             "> 0 and T_K >= 0")
        if self.model == "tsu_esaki" and not (self.E_F_eV is not None and self.E_F_eV > 0.0):
            raise ValueError("OxideTunnelingParams(model='tsu_esaki') needs E_F_eV > 0 (the cathode "
                             "supply Fermi energy; a metal gate ~3-11 eV, an accumulated Si surface "
                             "~0.05-0.2 eV) -- there is no safe default")

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
    Requires a PLANAR interface normal to a coordinate axis (both builders' stacks are)."""
    from dynameta.carriers.devsim_env import import_devsim
    ds = import_devsim()
    if not isinstance(params, OxideTunnelingParams):
        raise TypeError("oxide_tunneling_current: params must be an OxideTunnelingParams")
    if depth_m is not None and not (depth_m > 0.0):
        raise ValueError("depth_m must be > 0 (or None to keep the mesh's native units)")
    dim = int(ds.get_dimension(device=device))
    itf = _interface_between(device, oxide_region, semiconductor_region)
    side = "r0" if ds.get_region_list(device=device, interface=itf)[0] == oxide_region else "r1"
    nm = "_tun_oxide_idx_" + side
    if nm not in ds.get_interface_model_list(device=device, interface=itf):
        ds.interface_model(device=device, interface=itf, name=nm, equation="node_index@" + side)
    iface = np.asarray(ds.get_interface_model_values(device=device, interface=itf, name=nm)).astype(int)

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
