"""Gate leakage into a laterally contacted equilibrium film, and the lateral Fermi-level drop it causes
(DynaMeta v0.16, opt-in; notes: docs/audit/2026-11-14-v0.16-gate-leakage-lateral-drop.md).

WHY. A gated degenerate film a few nanometres thick, contacted only at an edge, is solved by the equilibrium
physics (physics_equilibrium) with ONE Fermi level for the whole region, set by the edge electrode
(fermi_ref_param). That is exact only while no current flows. A gate current through the thin dielectrics must
leave the film laterally, through a resistive sheet, to the edge contact; the film's Fermi level then varies
along the film and the gate-to-film voltage that sets the accumulation is smaller than the applied one away
from the contact.

FORMULATION (quasi-equilibrium film with a laterally varying Fermi level). The film keeps the equilibrium
closure n = N_c F(eta), eta = (psi - phi_F - Phi_c0)/V_t (parabolic or Kane), but phi_F becomes a NODE SOLUTION
(E_F = -q phi_F; physics_equilibrium.setup_semiconductor_region(fermi_ref_node=True), the same name as the
region parameter of the edge reference, so every closure string is unchanged) governed by current continuity:

    div j = 0 in the film,   j = -sigma grad phi_F,   sigma = q mu n        (FermiLevelEquation)
    sigma dphi_F/dn = J_g on a face under gate g                            (the tunnelling sink, below)
    phi_F = V_c at the edge electrode,  zero normal current elsewhere.

j is the electron current in quasi-Fermi form (drift + diffusion with the Fermi-Dirac Einstein relation).
Integrated over the film thickness this is the sheet equation  -d/dx (sigma_s dphi_F/dx) = J_top + J_bot  with
sigma_s = q mu Int n dz -- the distributed resistor. phi_F is not flat across the film: the gate current enters
through one face and spreads, so phi_F(z) is a parabola in z with peak-to-peak J t / (2 q mu n) (uniform n;
reported as `vertical_spread_V`), the face the current crosses lying 2/3 of that above the column mean, and
the vertical current adds J^2 t / (3 q mu n) per unit length to the Joule power. Relative to the lateral
drop of a film contacted at one end, J L^2 / (2 sigma_s), the spread is t^2 / L^2 (4 t^2 / L^2 with contacts
at both ends): negligible for films much longer than thick (3.6e-7 for a 6 nm film over 10 um, MEASURED equal
to the formula in tests/test_devsim_gate_leakage.py) but not in a short cell (2.1 % MEASURED for a 6 nm film
in a 40 nm cell). `max_drop_V` is a node maximum and includes the face excess. The finite-volume flux on an
edge is q mu (n_0 + n_1)/2 (phi_F0 - phi_F1)/L (an arithmetic edge average of n, second order on a smooth
profile; not Scharfetter-Gummel). mu is a constant (the TransportModel's mobility at n_bg in the builder).

GATE CURRENT. J_g at an interface node of the film is the net conventional current density from gate g into
the film (J > 0: electrons tunnel film -> gate, the TE kernel's sign) as a function of the local state. Two
models:
  "tsu_esaki" -- THE library kernel, reliability.leakage.tsu_esaki_two_sided (two-sided Tsu-Esaki supply with
      WKB transmission through the dielectric STACK, fixed energy grid, exact derivatives; validated in
      tests/test_tunneling_kernels.py), fed exactly as tunneling.OxideTunnelTerm feeds it, with the film supply
      x_s = eta at the interface node (the equilibrium closure's own Fermi level), x_g = (psi_s - Phi_c0 -
      V_gF)/V_t (V_gF = the gate's Fermi potential, its bias), and the column drops psi_s - psi(face k) read
      from the solution at the same lateral position. It covers direct tunnelling and Fowler-Nordheim
      tunnelling (one formula, both regimes) of conduction-band electrons, in both directions. NOT covered:
      trap-assisted tunnelling, Poole-Frenkel and hopping conduction, hot carriers, hole / valence-band
      tunnelling, image-force lowering, dielectric breakdown. Real high-k films at several MV/cm are often
      dominated by trap-assisted conduction, which this model UNDER-ESTIMATES (often by orders of magnitude):
      use the "table" model with measured data for those.
  "table" -- a user J(V) in the local gate-to-film Fermi-potential difference V = V_gF - phi_F (what a J-V
      measurement on a capacitor gives): a LeakageTable (measured points) or a callable V -> (J, dJ/dV); one
      for all gates or {gate: ...}. A J(E) quoted against the dielectric field E = V / t
      (LeakageTable.from_field, variable="dielectric") is read at the SOLVED drop across the gate's dielectric
      column at each node instead (exact derivatives in the film and gate-face potentials).
GateLeakage.scale multiplies J (sensitivity studies; 0 switches the sink off while keeping the field).

COUPLING. The sink enters the FermiLevelEquation row of every non-contact interface node as -A_i J_i (A_i the
interface area the node owns, physics_traps.interface_node_area) through a NONLOCAL custom term
(custom_terms; it reads the potential of the dielectric faces of its column), with the exact derivatives of the
discrete kernel; Poisson gains PotentialNodeCharge:EF_ref_V = -PotentialNodeCharge:Potential (every charge
term of an equilibrium film is a function of eta or a constant -- checked at setup: closure, Kane increment,
fixed sheet / bulk charges, FilmInterfaceTraps). The gate terminals are bias parameters: the gate currents are
reported, not assembled into a circuit row. Contact nodes carry no sink row (DEVSIM replaces their rows); their
tunnelling current is counted in the report as entering the contact directly, so the terminal currents
balance.

SCOPE AND REFUSALS. 2-D only (refused in 1-D and 3-D: the 3-D builder has no lateral Fermi field). Equilibrium
semiconductor regions only; the density-gradient option is REFUSED on the same region (its quantum potential
shifts the supply and the Lambda equation would need the Fermi field). A film with an interface to another
semiconductor region is refused (no Fermi-field interface condition). Noise / small-signal analyses are not
modelled (the term has no noise model, so noise_analysis refuses such a device).

PUBLIC API: GateLeakage, LeakageTable, setup_fermi_field_region, setup_fermi_contact, setup_gate_leakage,
GateLeakageTerm, gate_leakage_terms, gate_leakage_report, solve_lateral_1d, FERMI_FIELD_EQUATION. The 2-D
builder: LayeredDevsimBuilder(gate_leakage=GateLeakage(...)).
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Callable, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from dynameta.carriers import eq_registry as _R
from dynameta.carriers.custom_terms import CustomTerm
from dynameta.constants import Q_E
from dynameta.warning_types import AdvisoryWarning

__all__ = ["GateLeakage", "LeakageTable", "setup_fermi_field_region", "setup_fermi_contact",
           "setup_gate_leakage", "GateLeakageTerm", "gate_leakage_terms", "gate_leakage_report",
           "solve_lateral_1d", "FERMI_FIELD_EQUATION", "LEAKAGE_MODELS"]

LEAKAGE_MODELS = ("tsu_esaki", "table")
#: what a model="table" curve's V is (LeakageTable.variable): the gate-to-film Fermi-potential difference, or the
#: solved potential drop across the gate's dielectric column
TABLE_VARIABLES = ("fermi", "dielectric")
#: the film's Fermi-potential continuity equation (variable: physics_equilibrium.FERMI_REF_PARAM as a node solution)
FERMI_FIELD_EQUATION = "FermiLevelEquation"
_FLUX = "FermiLevelEdgeCurrent"
_MOBILITY = "FermiLevelMobility"
_TAIL_WARN = 1e-8
# charge terms of an equilibrium region whose Potential dependence is through eta only (or none): the
# fixed sheet / bulk charges (physics_equilibrium) and the film interface traps (physics_traps)
_ETA_ONLY_QTERMS = ("QTerm_sheet", "QTerm_bulk", "QTerm_fit")


def _fermi_name() -> str:
    from dynameta.carriers import physics_equilibrium as PE
    return PE.FERMI_REF_PARAM


# =====================================================================================================
# user inputs
# =====================================================================================================

@dataclass(frozen=True)
class LeakageTable:
    """A measured leakage curve J(V) [A/m^2] versus the gate-to-film Fermi-potential difference V [V]
    (V = V_gate - phi_F; J > 0 = conventional current from the gate into the film, i.e. electrons leaving the
    film, which is the sign at V > 0). Interpolated as a monotone C^1 cubic (PCHIP) of y = asinh(J / J0), so
    decades of J are followed smoothly and J may change sign; outside the table y is extended linearly with
    the slope of the end segment (exponential in J; continuous, its derivative may jump at the table end) and the
    report flags it. J(0) should be 0 (no current at equal Fermi
    levels); a table is not required to say so. J0_A_m2 is the asinh scale (J below it is interpolated
    linearly).

    variable: what V is at each interface node. "fermi" (default): the gate-to-film Fermi-potential difference
    V_gF - phi_F (a J-V measurement against the applied voltage). "dielectric": the potential drop across the
    gate's whole dielectric column, gate face minus film face, read from the solution at the node's lateral
    position (what a J(E) curve quoted against the dielectric field E = V / t means; from_field builds it). The
    two differ by the flat-band offset (work functions, affinities) and the film's band bending and degeneracy:
    MEASURED on the generic gated 6 nm film of tests/test_devsim_gate_leakage.py at +2 V, a dielectric drop of
    1.726 V (no gate work function) / 0.736 V (gate work function 5.1 eV) against V_gF - phi_F = 2 V."""
    V_V: Tuple[float, ...]
    J_A_m2: Tuple[float, ...]
    J0_A_m2: float = 1e-9
    variable: str = "fermi"

    def __post_init__(self):
        if self.variable not in TABLE_VARIABLES:
            raise ValueError("LeakageTable.variable must be one of {} (got {!r})".format(TABLE_VARIABLES,
                                                                                    self.variable))
        v = np.asarray(self.V_V, dtype=np.float64)
        j = np.asarray(self.J_A_m2, dtype=np.float64)
        if v.ndim != 1 or v.shape != j.shape or v.size < 2:
            raise ValueError("LeakageTable: V_V and J_A_m2 must be 1-D of equal length >= 2")
        if not (np.all(np.isfinite(v)) and np.all(np.isfinite(j))):
            raise ValueError("LeakageTable: non-finite entries")
        if not np.all(np.diff(v) > 0.0):
            raise ValueError("LeakageTable: V_V must be strictly increasing")
        if not (math.isfinite(self.J0_A_m2) and self.J0_A_m2 > 0.0):
            raise ValueError("LeakageTable: J0_A_m2 must be > 0")
        object.__setattr__(self, "V_V", tuple(float(x) for x in v))
        object.__setattr__(self, "J_A_m2", tuple(float(x) for x in j))

    @classmethod
    def from_field(cls, E_V_m, J_A_m2, thickness_m: float, **kw) -> "LeakageTable":
        """A J(E) curve quoted against the dielectric field E = V / thickness_m (the convention of the measurement:
        physical or equivalent oxide thickness) -> the table in V = E t with variable="dielectric", so it is read
        at the SOLVED drop across the gate's dielectric column, not at the gate-to-film Fermi-potential
        difference. (The two differ: on a gated 6 nm film at +2 V, t = 8 nm, V_gF - phi_F corresponds to E =
        2.50e8 V/m where the solved field is 2.16e8 V/m, J +23 % for a sinh(E / 2e8 V/m) curve; 3.4x with a 5.1 eV
        gate; tests/test_devsim_gate_leakage.py.)"""
        if not (thickness_m > 0.0):
            raise ValueError("LeakageTable.from_field: thickness_m must be > 0")
        if "variable" in kw:
            raise ValueError("LeakageTable.from_field: the variable is the dielectric drop (do not pass variable=)")
        return cls(tuple(float(e) * float(thickness_m) for e in E_V_m), tuple(J_A_m2), variable="dielectric", **kw)

    def _pchip(self):
        from scipy.interpolate import PchipInterpolator
        y = np.arcsinh(np.asarray(self.J_A_m2) / self.J0_A_m2)
        return PchipInterpolator(np.asarray(self.V_V), y, extrapolate=False)

    def __call__(self, V):
        """(J [A/m^2], dJ/dV [A/m^2/V]) at V (array)."""
        V = np.asarray(V, dtype=np.float64)
        p = self._pchip()
        dp = p.derivative()
        v0, v1 = self.V_V[0], self.V_V[-1]
        inside = (V >= v0) & (V <= v1)
        y = np.empty_like(V)
        dy = np.empty_like(V)
        y[inside] = p(V[inside])
        dy[inside] = dp(V[inside])
        lo, hi = V < v0, V > v1
        ys = np.arcsinh(np.asarray(self.J_A_m2) / self.J0_A_m2)
        vv = np.asarray(self.V_V)
        if np.any(lo):                                        # the end SEGMENT's slope (PCHIP's end derivative
            s = float((ys[1] - ys[0]) / (vv[1] - vv[0]))      # can be 0 by its shape-preserving limiter)
            y[lo] = ys[0] + s * (V[lo] - v0)
            dy[lo] = s
        if np.any(hi):
            s = float((ys[-1] - ys[-2]) / (vv[-1] - vv[-2]))
            y[hi] = ys[-1] + s * (V[hi] - v1)
            dy[hi] = s
        J = self.J0_A_m2 * np.sinh(y)
        return J, self.J0_A_m2 * np.cosh(y) * dy

    def in_range(self, V) -> bool:
        V = np.asarray(V, dtype=np.float64)
        return bool(np.all((V >= self.V_V[0]) & (V <= self.V_V[-1])))


@dataclass(frozen=True)
class GateLeakage:
    """Gate leakage into an equilibrium film with a lateral Fermi-level field (module header).

    model        (required) "tsu_esaki" (physical, the library's two-sided Tsu-Esaki/WKB kernel) or "table" (user J(V)).
    barriers     tsu_esaki: a tunneling.DielectricBarrier for every dielectric region of a gate column, or one
                 applied to all; `against` names the FILM material (its conduction-band barrier) or uses the
                 Anderson carry-over with the materials' chi_eV (DielectricBarrier docstring).
    gate_fermi_energy_eV  tsu_esaki: the metal gate's Fermi energy above its band bottom [eV] (required; only
                 checks that the gate band bottom stays below the film conduction edge).
    m_supply_over_m0  tsu_esaki: transverse (supply) mass of the Tsu-Esaki prefactor; None = the film's DOS mass
                 (the builder passes it; a Kane band uses its band-edge mass -- a parabolic supply).
    energy_max_eV / energy_step_kT  tsu_esaki: the fixed energy grid (reliability.leakage.tunnel_energy_grid).
    table        model "table": a LeakageTable or a callable V -> (J, dJ/dV), or {gate electrode: one of them}.
    mobility_m2Vs  the film mobility of sigma = q mu n; None = the TransportModel's mobility at n_bg (builder;
                 the builder refuses None when the film's TransportModel has no mobility_m2Vs_of_n_m3).
    scale        multiplies every gate current (>= 0; 0 = no sink, the field kept).
    gates        gate electrodes that leak (None = every gate whose dielectric column reaches the film).
    regions      film regions (None = every equilibrium semiconductor layer with an unpinned x-edge electrode)."""
    model: str
    barriers: object = None
    gate_fermi_energy_eV: Optional[float] = None
    m_supply_over_m0: Optional[float] = None
    energy_max_eV: float = 4.0
    energy_step_kT: float = 0.125
    table: object = None
    mobility_m2Vs: Optional[float] = None
    scale: float = 1.0
    gates: Optional[Tuple[str, ...]] = None
    regions: Optional[Tuple[str, ...]] = None

    def __post_init__(self):
        from dynameta.carriers.tunneling import DielectricBarrier
        if self.model not in LEAKAGE_MODELS:
            raise ValueError("GateLeakage.model must be one of {} (got {!r})".format(LEAKAGE_MODELS, self.model))
        if not (math.isfinite(self.scale) and self.scale >= 0.0):
            raise ValueError("GateLeakage.scale must be finite and >= 0")
        if self.mobility_m2Vs is not None and not (self.mobility_m2Vs > 0.0):
            raise ValueError("GateLeakage.mobility_m2Vs must be > 0")
        for nm in ("gates", "regions"):
            v = getattr(self, nm)
            if v is not None:
                if isinstance(v, str):
                    v = (v,)
                object.__setattr__(self, nm, tuple(v))
        if self.model == "tsu_esaki":
            b = self.barriers
            ok = isinstance(b, DielectricBarrier) or (
                isinstance(b, Mapping) and len(b) > 0 and all(isinstance(x, DielectricBarrier) for x in b.values()))
            if not ok:
                raise ValueError("GateLeakage(model='tsu_esaki') needs barriers: a tunneling.DielectricBarrier or a "
                                 "{dielectric region: DielectricBarrier} mapping")
            if not (self.gate_fermi_energy_eV is not None and self.gate_fermi_energy_eV > 0.0):
                raise ValueError("GateLeakage(model='tsu_esaki') needs gate_fermi_energy_eV > 0 (the metal gate's Fermi "
                                 "energy above its band bottom, ~5-11 eV; no default)")
            if not (self.energy_max_eV > 0.0 and self.energy_step_kT > 0.0):
                raise ValueError("GateLeakage: energy_max_eV and energy_step_kT must be > 0")
            if self.m_supply_over_m0 is not None and not (self.m_supply_over_m0 > 0.0):
                raise ValueError("GateLeakage.m_supply_over_m0 must be > 0")
            if self.table is not None:
                raise ValueError("GateLeakage(model='tsu_esaki'): table is for model='table'")
        else:
            t = self.table
            ok = callable(t) or (isinstance(t, Mapping) and len(t) > 0 and all(callable(x) for x in t.values()))
            if not ok:
                raise ValueError("GateLeakage(model='table') needs table: a LeakageTable or a callable V -> (J, dJ/dV), "
                                 "or {gate electrode: one of them}")

    def barrier_for(self, region: str):
        from dynameta.carriers.tunneling import DielectricBarrier
        if isinstance(self.barriers, DielectricBarrier):
            return self.barriers
        if region not in self.barriers:
            raise ValueError("GateLeakage.barriers has no entry for dielectric region {!r} (have {})".format(
                region, sorted(self.barriers)))
        return self.barriers[region]

    def table_for(self, gate: str):
        if callable(self.table):
            return self.table
        if gate not in self.table:
            raise ValueError("GateLeakage.table has no entry for gate {!r} (have {})".format(gate, sorted(self.table)))
        return self.table[gate]


# =====================================================================================================
# the Fermi-potential field of an equilibrium film
# =====================================================================================================

def _node(ds, device, region, name):
    return np.asarray(ds.get_node_model_values(device=device, region=region, name=name), dtype=np.float64)


def setup_fermi_field_region(device: str, region: str, *, mobility_m2Vs: float) -> None:
    """Install the film's Fermi-potential continuity equation (module header) on an equilibrium region set up
    with setup_semiconductor_region(fermi_ref_param=FERMI_REF_PARAM, fermi_ref_node=True). Call AFTER every
    charge term of the region (fixed charges, film interface traps) and before the first solve. Refuses a
    density-gradient region, a non-equilibrium region, a charge term whose Potential dependence is not through
    eta, and any dimension other than 2."""
    from dynameta.carriers.devsim_env import import_devsim
    from dynameta.carriers import physics_equilibrium as PE
    ds = import_devsim()
    ef = _fermi_name()
    if not (math.isfinite(float(mobility_m2Vs)) and mobility_m2Vs > 0.0):
        raise ValueError("setup_fermi_field_region: mobility_m2Vs must be finite and > 0")
    if int(ds.get_dimension(device=device)) != 2:
        raise NotImplementedError(
            "gate leakage / lateral Fermi field: only 2-D devices are supported (LayeredDevsimBuilder); the 1-D "
            "device has no lateral direction and the 3-D builder has no lateral Fermi field (not validated)")
    if PE._poisson_region_kind(device, region) != "equilibrium":
        raise ValueError("setup_fermi_field_region: region {!r} is not an equilibrium semiconductor region".format(region))
    eta = PE.equilibrium_eta_expr(device, region)
    if ef not in ds.get_node_model_list(device=device, region=region) or ef not in eta:
        raise ValueError("setup_fermi_field_region: region {!r} has no Fermi-level node solution {!r} in its closure "
                         "(setup_semiconductor_region(fermi_ref_param={!r}, fermi_ref_node=True))".format(region, ef, ef))
    if ef in ds.get_parameter_list(device=device, region=region):
        raise ValueError("setup_fermi_field_region: region {!r} also carries a PARAMETER {!r}; the Fermi level must be "
                         "the node solution only".format(region, ef))
    from dynameta.carriers import physics_equilibrium_dg as EQDG
    if region in EQDG.equilibrium_dg_regions(device):
        raise NotImplementedError(
            "gate leakage / lateral Fermi field on region {!r}: the equilibrium density gradient is not supported on "
            "the same region (its quantum potential shifts the tunnelling supply and its Lambda equation does not "
            "follow a varying Fermi level); solve the leakage without density_gradient".format(region))
    models = ds.get_node_model_list(device=device, region=region)
    odd = sorted(m for m in models if m.startswith(PE._QTERM) and ":" not in m
                 and not m.startswith(_ETA_ONLY_QTERMS))
    if odd:
        raise NotImplementedError(
            "setup_fermi_field_region: region {!r} carries charge term(s) {} whose dependence on the Fermi level is "
            "unknown (only the fixed sheet / bulk charges and the film interface traps are supported)".format(region, odd))
    for itf in ds.get_interface_list(device=device):
        rr = list(ds.get_region_list(device=device, interface=itf))
        if region in rr:
            other = [r for r in rr if r != region][0]
            if "Electrons" in ds.get_node_model_list(device=device, region=other):
                raise NotImplementedError(
                    "setup_fermi_field_region: region {!r} has an interface to the semiconductor region {!r}; a "
                    "Fermi-field interface condition is not modelled".format(region, other))
    # Poisson: every charge term is a function of eta = (psi - EF - Phi_c0)/V_t (or constant), so d/dEF = -d/dpsi
    ds.node_model(device=device, region=region, name="Electrons:" + ef, equation="-Electrons:Potential")
    ds.node_model(device=device, region=region, name="PotentialNodeCharge:" + ef,
                  equation="-PotentialNodeCharge:Potential")
    for m in ("Electrons", "Electrons:Potential", "Electrons:" + ef, ef):
        ds.edge_from_node_model(device=device, region=region, node_model=m)
    ds.set_parameter(device=device, region=region, name=_MOBILITY, value=float(mobility_m2Vs))
    g = "ElectronCharge * {} * EdgeInverseLength".format(_MOBILITY)
    nbar = "0.5 * (Electrons@n0 + Electrons@n1)"
    dphi = "({0}@n0 - {0}@n1)".format(ef)
    ds.edge_model(device=device, region=region, name=_FLUX, equation="{} * {} * {}".format(g, nbar, dphi))
    for k in ("n0", "n1"):
        ds.edge_model(device=device, region=region, name="{}:Potential@{}".format(_FLUX, k),
                      equation="{} * 0.5 * Electrons:Potential@{} * {}".format(g, k, dphi))
    ds.edge_model(device=device, region=region, name="{}:{}@n0".format(_FLUX, ef),
                  equation="{} * (0.5 * Electrons:{}@n0 * {} + {})".format(g, ef, dphi, nbar))
    ds.edge_model(device=device, region=region, name="{}:{}@n1".format(_FLUX, ef),
                  equation="{} * (0.5 * Electrons:{}@n1 * {} - {})".format(g, ef, dphi, nbar))
    _R.record_region_equation(device, region, name=FERMI_FIELD_EQUATION, variable_name=ef,
                              edge_model=_FLUX, variable_update="default")


def setup_fermi_contact(device: str, contact: str) -> None:
    """The edge electrode `contact` of a Fermi-field film: phi_F = <contact>_bias on its nodes (a Dirichlet on the
    FermiLevelEquation only -- the Potential floats there, as for the equilibrium edge reference). The contact
    must capture the whole edge column (the builder's full-edge strip does)."""
    from dynameta.carriers.devsim_env import import_devsim
    ds = import_devsim()
    ef = _fermi_name()
    p = "{}_bias".format(contact)
    if p not in ds.get_parameter_list(device=device):
        ds.set_parameter(device=device, name=p, value=0.0)
    cn = "{}_fermi_dirichlet".format(contact)
    ds.contact_node_model(device=device, contact=contact, name=cn, equation="{} - {}".format(ef, p))
    ds.contact_node_model(device=device, contact=contact, name="{}:{}".format(cn, ef), equation="1")
    _R.record_contact_equation(device, contact, name=FERMI_FIELD_EQUATION, node_model=cn, edge_current_model=_FLUX)


# =====================================================================================================
# the tunnelling sink
# =====================================================================================================

def _gate_columns(ds, device, semi, gate_contact, leakage, sem_mat, chi_eV):
    """Interface nodes of `semi` toward gate_contact (ALL of them, contact nodes flagged), the dielectric chain,
    each layer's far face with lateral weights, and the TunnelLayer list (tsu_esaki)."""
    from dynameta.carriers import tunneling as TN
    from dynameta.carriers.noise_analysis import _contact_nodes
    from dynameta.carriers.physics_traps import interface_node_area
    from dynameta.reliability.leakage import TunnelLayer
    chain = TN.dielectric_column(device, semi, gate_contact)
    if chain is None:
        raise ValueError("gate leakage: no column of dielectric regions joins {!r} to the gate {!r}".format(
            semi, gate_contact))
    gate_region = ds.get_region_list(device=device, contact=gate_contact)[0]
    if gate_region != chain[-1]:
        raise ValueError("gate leakage: the gate {!r} does not sit on the last dielectric of its column".format(
            gate_contact))
    Xs = TN._coords(device, semi)
    itf0 = TN._interface_between(device, semi, chain[0])
    iface = TN._interface_node_indices(device, itf0, semi)
    span = float(np.max(np.ptp(Xs, axis=0)))
    tol = 1e-9 * max(span, 1e-30)
    isp = np.ptp(Xs[iface], axis=0)
    flat = [k for k in range(Xs.shape[1]) if isp[k] <= tol]
    if len(flat) != 1:
        raise NotImplementedError("gate leakage: the {}/{} interface is not planar and normal to a coordinate axis "
                                  "(spread {})".format(semi, chain[0], isp.tolist()))
    ax = flat[0]
    lat = [k for k in range(Xs.shape[1]) if k != ax]
    cnodes = set()
    for c in ds.get_contact_list(device=device):
        cnodes.update(int(i) for i in _contact_nodes(device, semi, c))
    is_contact = np.array([int(i) in cnodes for i in iface], dtype=bool)
    area = interface_node_area(device, semi, itf0)[iface].astype(np.float64)
    z = float(np.mean(Xs[iface, ax]))
    faces, layers, prev = [], [], semi
    for j, reg in enumerate(chain):
        Xr = TN._coords(device, reg)
        near = TN._interface_node_indices(device, TN._interface_between(device, prev, reg), reg)
        far = (TN._interface_node_indices(device, TN._interface_between(device, reg, chain[j + 1]), reg)
               if j + 1 < len(chain) else _contact_nodes(device, reg, gate_contact))
        if np.ptp(Xr[near, ax]) > tol or np.ptp(Xr[far, ax]) > tol:
            raise NotImplementedError("gate leakage: dielectric {!r} has a non-planar face".format(reg))
        z_near, z_far = float(np.mean(Xr[near, ax])), float(np.mean(Xr[far, ax]))
        if abs(z_near - z) > tol or not (abs(z_far - z) > abs(z_near - z) + tol):
            raise ValueError("gate leakage: dielectric {!r} is not stacked away from the film".format(reg))
        if j + 1 == len(chain):
            onface = np.nonzero(np.abs(Xr[:, ax] - z_far) <= tol)[0]
            if not set(onface.tolist()) <= set(np.asarray(far).tolist()):
                raise NotImplementedError("gate leakage: the gate {!r} covers only part of the far face of {!r} (a "
                                          "patch gate) -- not supported".format(gate_contact, reg))
        W = TN._lateral_weights(Xr[far][:, lat], Xs[iface][:, lat], tol)
        faces.append((reg, np.asarray(far, dtype=np.int64), W))
        if leakage.model == "tsu_esaki":
            b = leakage.barrier_for(reg)
            layers.append(TunnelLayer(b.offset_eV(sem_mat, chi_eV), abs(z_far - z_near), b.m_tunnel_over_m0))
        z, prev = z_far, reg
    return dict(chain=tuple(chain), iface=np.asarray(iface, dtype=np.int64), is_contact=is_contact, area=area,
                faces=faces, layers=tuple(layers), axis=ax, interface=itf0, lateral=lat)


class GateLeakageTerm(CustomTerm):
    """The tunnelling sink of one gate into a Fermi-field film (setup_gate_leakage builds it). Residual: -A_i s J_i
    on the FermiLevelEquation row of each non-contact interface node i; Jacobian: exact derivatives (psi_s,
    phi_F at the node; the dielectric face potentials of its column, tsu_esaki only)."""

    locality = "nonlocal"
    jacobian_exact = True
    pattern_changes = False

    def __init__(self, name, **kw):
        super().__init__(name)
        self.__dict__.update(kw)
        self._tail_warned = False
        self.last = None
        # a dielectric-drop table (LeakageTable.from_field) reads the column's face potentials, like tsu_esaki
        self._dielectric_table = (self.model == "table"
                                  and getattr(getattr(self, "table", None), "variable", "fermi") == "dielectric")
        self._reads_faces = self.model == "tsu_esaki" or self._dielectric_table

    def equations(self, device):
        return [(self.semi, FERMI_FIELD_EQUATION)]

    @property
    def rows_nodes(self):
        return self.iface[~self.is_contact]

    def _columns(self, device):
        from dynameta.carriers import custom_terms as CT
        ef = _fermi_name()
        nodes = self.rows_nodes
        rows = CT.node_rows(device, self.semi, FERMI_FIELD_EQUATION, nodes)
        cP = CT.node_columns(device, self.semi, "Potential", nodes)
        cF = CT.node_columns(device, self.semi, ef, nodes)
        keep = np.nonzero(~self.is_contact)[0]
        fcols = []
        if self._reads_faces:
            for reg, far, W in self.faces:
                cols = CT.node_columns(device, reg, "Potential", far)
                fcols.append([(cols[W[i][0]], W[i][1]) for i in keep])
        return rows, cP, cF, fcols

    def structural_pattern(self, device):
        rows, cP, cF, fcols = self._columns(device)
        pat = set()
        for i, r in enumerate(rows):
            for c in [cP[i], cF[i]] + [int(cc) for fc in fcols for cc in fc[i][0]]:
                pat.add((int(r), int(c)))
        return sorted(pat)

    def gate_fermi_potential(self, device) -> float:
        """The gate's Fermi potential V_gF = its bias (the <gate>_bias parameter minus a work-function offset)."""
        from dynameta.carriers.devsim_env import import_devsim
        ds = import_devsim()
        return float(ds.get_parameter(device=device, name=self.gate_param)) - self.gate_offset

    def state(self, device):
        """J [A/m^2] (scale applied) and its derivatives at EVERY interface node of the column (contact nodes too:
        the report counts their current), at the current solution."""
        from dynameta.carriers.devsim_env import import_devsim
        ds = import_devsim()
        ef = _fermi_name()
        psi_s = _node(ds, device, self.semi, "Potential")[self.iface]
        phi = _node(ds, device, self.semi, ef)[self.iface]
        VgF = self.gate_fermi_potential(device)
        s = self.scale
        n = self.iface.size
        if self._reads_faces:                                 # column drops psi_s - psi(far face of layer k)
            d = np.empty((n, len(self.faces)))
            for k, (reg, far, W) in enumerate(self.faces):
                pf = _node(ds, device, reg, "Potential")[far]
                d[:, k] = psi_s - np.array([float(np.dot(w, pf[j])) for j, w in W])
        if self._dielectric_table:
            # V = the drop across the whole column, gate face minus film face = -d_last: dV/dpsi_s = -1 and
            # dV/d(gate-face potential) = +1, i.e. dJ/dd_last = -dJ/dV (the assembly's face-column convention)
            V = -d[:, -1]
            J, dJ = self.table(V)
            J = np.asarray(J, dtype=np.float64) * np.ones(n)
            dJ = np.asarray(dJ, dtype=np.float64) * np.ones(n)
            dJ_dd = np.zeros((n, len(self.faces)))
            dJ_dd[:, -1] = -s * dJ
            st = {"J": s * J, "dJ_dpsi": -s * dJ, "dJ_dphi": np.zeros(n), "dJ_dd": dJ_dd,
                  "V_local": VgF - phi, "V_table": V, "d": d, "psi_s": psi_s, "phi": phi, "V_gF": VgF, "tail": 0.0}
            if hasattr(self.table, "in_range"):
                st["table_in_range"] = bool(self.table.in_range(V))
            self.last = st
            return st
        if self.model == "table":
            V = VgF - phi
            J, dJ = self.table(V)
            J = np.asarray(J, dtype=np.float64) * np.ones(n)
            dJ = np.asarray(dJ, dtype=np.float64) * np.ones(n)
            st = {"J": s * J, "dJ_dpsi": np.zeros(n), "dJ_dphi": -s * dJ, "dJ_dd": np.zeros((n, 0)),
                  "V_local": V, "psi_s": psi_s, "phi": phi, "V_gF": VgF, "tail": 0.0}
            if hasattr(self.table, "in_range"):
                st["table_in_range"] = bool(self.table.in_range(V))
            self.last = st
            return st
        from dynameta.reliability.leakage import tsu_esaki_two_sided
        Phi_c0 = float(ds.get_parameter(device=device, region=self.semi, name="Phi_c0"))
        vt = float(ds.get_parameter(device=device, region=self.semi, name="V_t"))
        x_s = (psi_s - phi - Phi_c0) / vt
        x_g = (psi_s - Phi_c0 - VgF) / vt
        bottom = x_g * self.grid.kT_J - self.gate_fermi_J
        if np.any(bottom >= 0.0):
            raise RuntimeError(
                "gate leakage {!r}: the gate's band bottom rises above the film conduction edge at {} interface "
                "node(s) (max {:.4g} eV above); electrons below it cannot tunnel, which the fixed grid does not model "
                "(gate_fermi_energy_eV too small or an extreme bias)".format(
                    self.name, int(np.count_nonzero(bottom >= 0.0)), float(np.max(bottom)) / Q_E))
        r = tsu_esaki_two_sided(self.grid, x_s, x_g, d, self.layers, m_supply_over_m0=self.m_supply)
        if not self._tail_warned and float(np.max(r.tail)) > _TAIL_WARN:
            self._tail_warned = True
            warnings.warn("gate leakage {!r}: the energy grid [0, {:.3g} eV] truncates the supply integral ({:.2e} of "
                          "the one-way sum at its top); raise energy_max_eV".format(
                              self.name, self.grid.eps_max_J / Q_E, float(np.max(r.tail))),
                          AdvisoryWarning, stacklevel=2)
        dpsi = (r.dJ_dxs + r.dJ_dxg) / vt + np.sum(r.dJ_dd, axis=1)
        st = {"J": s * r.J, "dJ_dpsi": s * dpsi, "dJ_dphi": -s * r.dJ_dxs / vt, "dJ_dd": s * r.dJ_dd,
              "V_local": VgF - phi, "psi_s": psi_s, "phi": phi, "V_gF": VgF, "d": d, "x_s": x_s, "x_g": x_g,
              "J_fwd": s * r.J_fwd, "J_bwd": s * r.J_bwd, "tail": float(np.max(r.tail))}
        self.last = st
        return st

    def assemble(self, device, what, timemode):
        if timemode != "DC" or self.scale == 0.0:
            return [], []
        st = self.state(device)
        rows, cP, cF, fcols = self._columns(device)
        keep = np.nonzero(~self.is_contact)[0]
        A = self.area[keep]
        J = st["J"][keep]
        R, C, V = [], [], []
        for i in range(rows.size):
            k = keep[i]
            ents = [(int(cP[i]), st["dJ_dpsi"][k]), (int(cF[i]), st["dJ_dphi"][k])]
            for f, fc in enumerate(fcols):
                for c, w in zip(*fc[i]):
                    ents.append((int(c), -st["dJ_dd"][k, f] * float(w)))
            for c, dJ in ents:
                R.append(int(rows[i]))
                C.append(c)
                V.append(-A[i] * float(dJ))
        rcv = [x for t in zip(R, C, V) for x in t]
        rv = [x for t in zip([int(x) for x in rows], (-A * J).tolist()) for x in t]
        return rcv, rv


def setup_gate_leakage(device: str, region: str, gate_contact: str, leakage: GateLeakage, *,
                       gate_offset_V: float = 0.0, semiconductor_material: Optional[str] = None,
                       chi_eV: Optional[Mapping[str, Optional[float]]] = None,
                       m_supply_over_m0: Optional[float] = None, name: Optional[str] = None) -> GateLeakageTerm:
    """Install the tunnelling sink of the metal gate `gate_contact` into the Fermi-field film `region`
    (setup_fermi_field_region first). gate_offset_V: the gate contact's potential offset beyond its bias (a work
    function, physics_equilibrium.work_function_offset_V): the gate's Fermi potential is <gate>_bias - offset.
    semiconductor_material / chi_eV: compared with each DielectricBarrier.against (Anderson carry-over).
    m_supply_over_m0 overrides GateLeakage.m_supply_over_m0 (required for tsu_esaki when that is None)."""
    from dynameta.carriers import custom_terms as CT
    from dynameta.carriers.devsim_env import import_devsim
    from dynameta.reliability.leakage import tunnel_energy_grid
    ds = import_devsim()
    if not isinstance(leakage, GateLeakage):
        raise TypeError("setup_gate_leakage: leakage must be a GateLeakage")
    if FERMI_FIELD_EQUATION not in ds.get_equation_list(device=device, region=region):
        raise ValueError("setup_gate_leakage: region {!r} has no {} (setup_fermi_field_region first)".format(
            region, FERMI_FIELD_EQUATION))
    if gate_contact not in ds.get_contact_list(device=device):
        raise ValueError("setup_gate_leakage: no contact {!r}".format(gate_contact))
    gate_region = ds.get_region_list(device=device, contact=gate_contact)[0]
    if "Electrons" in ds.get_node_model_list(device=device, region=gate_region):
        raise NotImplementedError("setup_gate_leakage: the gate {!r} sits on a semiconductor region; only metal gates "
                                  "on a dielectric are modelled".format(gate_contact))
    sem_mat = semiconductor_material if semiconductor_material is not None else ds.get_material(
        device=device, region=region)
    cols = _gate_columns(ds, device, region, gate_contact, leakage, sem_mat, chi_eV)
    kw = dict(semi=region, gate=gate_contact, gate_param="{}_bias".format(gate_contact),
              gate_offset=float(gate_offset_V), model=leakage.model, scale=float(leakage.scale), leakage=leakage,
              **cols)
    if gate_contact + "_bias" not in ds.get_parameter_list(device=device):
        raise ValueError("setup_gate_leakage: gate {!r} is not driven by the bias parameter {!r}".format(
            gate_contact, gate_contact + "_bias"))
    if leakage.model == "tsu_esaki":
        m = m_supply_over_m0 if m_supply_over_m0 is not None else leakage.m_supply_over_m0
        if m is None or not (m > 0.0):
            raise ValueError("setup_gate_leakage(model='tsu_esaki'): no supply mass (GateLeakage.m_supply_over_m0 or "
                             "m_supply_over_m0=)")
        vt = float(ds.get_parameter(device=device, region=region, name="V_t"))
        kw.update(grid=tunnel_energy_grid(Q_E * vt, leakage.energy_max_eV * Q_E, step_kT=leakage.energy_step_kT),
                  m_supply=float(m), gate_fermi_J=float(leakage.gate_fermi_energy_eV) * Q_E)
    else:
        kw.update(table=leakage.table_for(gate_contact))
    term = GateLeakageTerm(name or "gleak_{}_{}".format(region, gate_contact), **kw)
    CT.add_custom_term(device, term)
    return term


def gate_leakage_terms(device: str, region: Optional[str] = None) -> list:
    """The live gate-leakage terms of `device` (of one film region when given)."""
    from dynameta.carriers import custom_terms as CT
    return [t for t in CT.live_terms(device) if isinstance(t, GateLeakageTerm) and (region is None or t.semi == region)]


# =====================================================================================================
# diagnostics
# =====================================================================================================

def _edge_arrays(ds, device, region):
    if "_gl_idx@n0" not in ds.get_edge_model_list(device=device, region=region):
        ds.node_model(device=device, region=region, name="_gl_idx", equation="node_index")
        ds.edge_from_node_model(device=device, region=region, node_model="_gl_idx")
    i0 = np.rint(ds.get_edge_model_values(device=device, region=region, name="_gl_idx@n0")).astype(int)
    i1 = np.rint(ds.get_edge_model_values(device=device, region=region, name="_gl_idx@n1")).astype(int)
    F = np.asarray(ds.get_edge_model_values(device=device, region=region, name=_FLUX))
    couple = np.asarray(ds.get_edge_model_values(device=device, region=region, name="EdgeCouple"))
    return i0, i1, F * couple


def gate_leakage_report(device: str, region: str, *, contacts: Sequence[str] = (),
                        depth_m: Optional[float] = None) -> Dict[str, object]:
    """Per-bias diagnostics of a Fermi-field film (2-D: currents per unit depth [A/m], powers [W/m], unless
    depth_m scales them to A / W). Every terminal current is the current INTO the device through it.
      gates          {gate: {"I": current, "V_fermi": its Fermi potential, "J_max": signed J at the node of largest
                     |J|, "J_mean": area-weighted mean J, "gate_action": area-weighted mean of (V_gF - phi_F) /
                     (V_gF - V_c) over the film face (1 = nominal; NaN when V_gF == V_c), "table_in_range"}}
      contacts       {contact: I} (the lateral current delivered through the edge contact, including the
                     tunnelling current of its own nodes)
      balance        sum of every terminal current (0 to the solver tolerance), and balance_rel = balance /
                     (sum |I|)
      x_m, phi_F_V   the column-mean Fermi potential along the film (ascending lateral coordinate) and
                     phi_F at each face row ({interface: array})
      max_drop_V     max over the film nodes of |phi_F - V_c| (single contact voltage), ef_span_V = ptp(phi_F)
      vertical_spread_V  max over node columns of ptp(phi_F) (the quasi-equilibrium check)
      P_joule, P_tunnel, P_terminals  dissipated power in the film sheet, in the tunnel junctions (sum A J (V_gF -
                     phi_F)), and sum_k I_k V_k over the terminals (= P_joule + P_tunnel)."""
    from dynameta.carriers.devsim_env import import_devsim
    from dynameta.carriers.noise_analysis import _contact_nodes
    ds = import_devsim()
    ef = _fermi_name()
    sc = float(depth_m) if depth_m is not None else 1.0
    phi = _node(ds, device, region, ef)
    X = np.column_stack([_node(ds, device, region, c) for c in "xy"])
    i0, i1, I_e = _edge_arrays(ds, device, region)          # I_e: current n0 -> n1 on each edge [A/m]
    out_nodes = np.zeros(phi.size)
    np.add.at(out_nodes, i0, I_e)
    np.add.at(out_nodes, i1, -I_e)
    terms = gate_leakage_terms(device, region)
    gates, inj = {}, np.zeros(phi.size)
    P_tun = 0.0
    contacts = list(contacts) or [c for c in ds.get_contact_list(device=device)
                                  if _contact_nodes(device, region, c).size
                                  and FERMI_FIELD_EQUATION in ds.get_contact_equation_list(device=device, contact=c)]
    Vc = {c: float(ds.get_parameter(device=device, name=c + "_bias")) for c in contacts}
    vc_ref = float(np.mean(list(Vc.values()))) if Vc else 0.0
    interfaces = {}
    for t in terms:
        st = t.state(device)
        I_nodes = t.area * st["J"]
        np.add.at(inj, t.iface, I_nodes)
        P_tun += float(np.sum(I_nodes * (st["V_gF"] - st["phi"])))
        k = int(np.argmax(np.abs(st["J"])))
        den = st["V_gF"] - vc_ref
        action = (float(np.sum(t.area * (st["V_gF"] - st["phi"])) / np.sum(t.area) / den)
                  if den != 0.0 else float("nan"))
        g = {"I": float(np.sum(I_nodes)) * sc, "V_fermi": st["V_gF"], "J_max": float(st["J"][k]),
             "J_mean": float(np.sum(I_nodes) / np.sum(t.area)), "gate_action": action, "interface": t.interface}
        if "table_in_range" in st:
            g["table_in_range"] = st["table_in_range"]
            if not st["table_in_range"]:
                warnings.warn("gate leakage {!r}: the local gate-to-film voltage left the LeakageTable range "
                              "(extrapolated)".format(t.name), AdvisoryWarning, stacklevel=2)
        gates[t.gate] = g
        o = np.argsort(X[t.iface, 0])
        interfaces[t.interface] = (X[t.iface, 0][o], phi[t.iface][o])
    cur = {}
    for c in contacts:
        cn = _contact_nodes(device, region, c)
        # current from the contact into the film = net outflow of its nodes minus the tunnelling injected there
        cur[c] = float(np.sum(out_nodes[cn]) - np.sum(inj[cn])) * sc
    total = sum(g["I"] for g in gates.values()) + sum(cur.values())
    mag = sum(abs(g["I"]) for g in gates.values()) + sum(abs(v) for v in cur.values())
    P_joule = float(np.sum(I_e * (phi[i0] - phi[i1]))) * sc
    P_term = (sum(g["I"] * g["V_fermi"] for g in gates.values()) + sum(cur[c] * Vc[c] for c in contacts))
    kx = np.round(X[:, 0] / 1e-15).astype(np.int64)
    ux = np.unique(kx)
    xm = ux * 1e-15
    col_mean = np.array([float(np.mean(phi[kx == k])) for k in ux])
    vspread = max(float(np.ptp(phi[kx == k])) for k in ux)
    single = len(set(Vc.values())) <= 1
    return {"gates": gates, "contacts": cur, "balance": total, "balance_rel": (total / mag if mag > 0 else 0.0),
            "x_m": xm, "phi_F_V": col_mean, "phi_F_faces": interfaces,
            "max_drop_V": (float(np.max(np.abs(phi - vc_ref))) if single else float("nan")),
            "ef_span_V": float(np.ptp(phi)), "vertical_spread_V": vspread,
            "P_joule": P_joule, "P_tunnel": P_tun * sc, "P_terminals": P_term, "V_contacts": Vc}


# =====================================================================================================
# the reduced 1-D lateral model (sheet equation), for films far longer than a 2-D mesh can hold
# =====================================================================================================

def solve_lateral_1d(length_m: float, sheet_conductance: Callable, injected_current: Callable, *,
                     v_contact: float = 0.0, contacts: str = "left", x_m: Optional[np.ndarray] = None,
                     n_nodes: int = 801, tol_V: float = 1e-13, max_iter: int = 80) -> Dict[str, np.ndarray]:
    """The sheet equation of the module header in one lateral dimension:

        -d/dx (sigma_s(phi) dphi/dx) = J(phi),   phi = v_contact at x = 0 (contacts="left", zero current at x = L)
                                                 or at both ends (contacts="both")

    sheet_conductance(phi) -> (sigma_s [S], dsigma_s/dphi) and injected_current(phi) -> (J [A/m^2], dJ/dphi), J the
    NET conventional current density injected into the film by all gates (> 0 into the film) at the local
    Fermi potential phi (both close over the gate voltages). Finite volumes on x_m (default: n_nodes points graded
    geometrically toward the contact(s), ratio 1e-4 between the first and the largest cell), edge conductance
    the mean of the two nodes', Newton with the exact Jacobian and a homotopy in J (lambda 0 -> 1, halved on a
    failed step). Returns x, phi, max_drop (max |phi - v_contact|), iterations and the residual.

    Refusals: a non-finite or non-positive length_m / tol_V, a non-finite v_contact or x_m, max_iter < 1 or
    n_nodes < 2 raise ValueError naming the argument before any solve. A residual or Jacobian that turns NaN /
    inf (sheet_conductance or injected_current returned a non-finite value) or a singular Jacobian (e.g. a zero
    sheet conductance) fails the Newton step; the continuation halves its step and, when no step size is left,
    raises FloatingPointError naming the cause."""
    if not (math.isfinite(float(length_m)) and length_m > 0.0):
        raise ValueError("solve_lateral_1d: length_m must be > 0 and finite, got {!r}".format(length_m))
    if contacts not in ("left", "both"):
        raise ValueError("solve_lateral_1d: contacts must be 'left' or 'both'")
    if not math.isfinite(float(v_contact)):
        raise ValueError("solve_lateral_1d: v_contact must be finite, got {!r}".format(v_contact))
    if not (math.isfinite(float(tol_V)) and tol_V > 0.0):
        raise ValueError("solve_lateral_1d: tol_V must be finite and > 0, got {!r}".format(tol_V))
    if int(max_iter) < 1:
        raise ValueError("solve_lateral_1d: max_iter must be >= 1, got {!r}".format(max_iter))
    if x_m is None:
        n = int(n_nodes)
        if n < 2:
            raise ValueError("solve_lateral_1d: n_nodes must be >= 2, got {!r}".format(n_nodes))
        if contacts == "left":
            x = _graded(n, length_m)
        else:
            half = _graded(n // 2 + 1, 0.5 * length_m)
            x = np.concatenate([half, length_m - half[-2::-1]])
    else:
        x = np.asarray(x_m, dtype=np.float64)
        if not np.all(np.isfinite(x)):
            raise ValueError("solve_lateral_1d: x_m must be finite")
        if x[0] != 0.0 or not np.isclose(x[-1], length_m) or not np.all(np.diff(x) > 0.0):
            raise ValueError("solve_lateral_1d: x_m must increase from 0 to length_m")
    h = np.diff(x)
    w = np.zeros(x.size)                                  # box widths
    w[:-1] += 0.5 * h
    w[1:] += 0.5 * h
    fixed = np.zeros(x.size, dtype=bool)
    fixed[0] = True
    if contacts == "both":
        fixed[-1] = True
    free = np.nonzero(~fixed)[0]
    phi = np.full(x.size, float(v_contact))

    def residual(p, lam):
        s, ds_ = sheet_conductance(p)
        J, dJ = injected_current(p)
        s = np.asarray(s, dtype=np.float64) * np.ones_like(p)
        ds_ = np.asarray(ds_, dtype=np.float64) * np.ones_like(p)
        J = np.asarray(J, dtype=np.float64) * np.ones_like(p)
        dJ = np.asarray(dJ, dtype=np.float64) * np.ones_like(p)
        se = 0.5 * (s[:-1] + s[1:])
        flux = se * (p[:-1] - p[1:]) / h                    # current i -> i+1 [A/m]
        r = np.zeros(p.size)
        r[:-1] += flux
        r[1:] -= flux
        r -= lam * w * J
        n_ = p.size
        main = np.zeros(n_)
        up = np.zeros(n_ - 1)
        lo = np.zeros(n_ - 1)
        g = se / h
        dp = (p[:-1] - p[1:]) / h
        # d flux_i / d p_i = g + 0.5 ds_i dp ; d flux_i / d p_{i+1} = -g + 0.5 ds_{i+1} dp
        a = g + 0.5 * ds_[:-1] * dp
        b = -g + 0.5 * ds_[1:] * dp
        main[:-1] += a
        up += b
        lo -= a
        main[1:] -= b
        main -= lam * w * dJ
        return r, main, up, lo

    def newton(p, lam):
        from scipy.linalg import solve_banded
        for it in range(max_iter):
            r, main, up, lo = residual(p, lam)
            m = main[free]
            u_ = up[free[:-1]] if free.size > 1 else np.zeros(0)
            l_ = lo[free[1:] - 1] if free.size > 1 else np.zeros(0)
            ab = np.zeros((3, free.size))
            ab[0, 1:] = u_
            ab[1] = m
            ab[2, :-1] = l_
            if not (np.all(np.isfinite(ab)) and np.all(np.isfinite(r[free]))):
                # checked here: solve_banded's own finiteness check would raise a ValueError the continuation
                # below does not catch
                raise FloatingPointError(
                    "solve_lateral_1d: non-finite Newton update (the residual or Jacobian is NaN / inf: "
                    "sheet_conductance or injected_current returned a non-finite value)")
            try:
                dx = solve_banded((1, 1), ab, -r[free], check_finite=False)
            except np.linalg.LinAlgError as e:
                raise FloatingPointError("solve_lateral_1d: singular Jacobian ({}; e.g. a zero sheet "
                                         "conductance)".format(e)) from e
            if not np.all(np.isfinite(dx)):
                raise FloatingPointError("solve_lateral_1d: non-finite Newton update")
            step = np.max(np.abs(dx))
            dx *= min(1.0, 0.5 / step) if step > 0.5 else 1.0     # damp a large first step (0.5 V cap)
            p = p.copy()
            p[free] += dx
            if step < tol_V:
                return p, it + 1
        raise RuntimeError("solve_lateral_1d: Newton did not converge (last update {:.3g} V)".format(step))

    lam, dlam, its = 0.0, 1.0, 0
    while lam < 1.0:
        nxt = min(1.0, lam + dlam)
        try:
            phi, k = newton(phi, nxt)
            its += k
            lam = nxt
            dlam = min(1.0, 2.0 * dlam)
        except (RuntimeError, FloatingPointError):
            dlam *= 0.5
            if dlam < 1e-6:
                raise
    r = residual(phi, 1.0)[0]
    return {"x_m": x, "phi_V": phi, "max_drop_V": float(np.max(np.abs(phi - v_contact))), "iterations": its,
            "residual": float(np.max(np.abs(r[free]))) if free.size else 0.0}


def _graded(n: int, length_m: float) -> np.ndarray:
    """n points on [0, length_m], cells growing geometrically away from 0 (first / largest = 1e-4)."""
    u = np.geomspace(1e-4, 1.0, n - 1)
    x = np.concatenate([[0.0], np.cumsum(u)])
    return x * (length_m / x[-1])
