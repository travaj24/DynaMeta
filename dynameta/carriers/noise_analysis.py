"""
Small-signal NOISE of a DEVSIM drift-diffusion device by the impedance-field method -- the noise
companion of ac_analysis (DynaMeta v0.14.0, workstream D2a). Opt-in: nothing here runs unless it
is called, and it only READS the device (its DC operating point and models are left unchanged).

=====================================================================================================
1. THE IMPEDANCE-FIELD METHOD (Shockley, Copeland & James, "The impedance field method of noise
   calculation in active semiconductor devices", in Quantum Theory of Atoms, Molecules and the
   Solid State, Academic Press 1966, pp. 537-563; Bonani & Ghione, "Noise in Semiconductor
   Devices: Modeling and Simulation", Springer 2001, ch. 3 + 4).

   Linearise the device about its DC point. A microscopic fluctuation s(r, omega) (a current-
   density or a generation-rate fluctuation) perturbs the right-hand side of the small-signal
   system; the terminal response is a LINEAR functional of it,
        delta_out(omega) = integral G(r, omega) s(r, omega) d^3r,
   G being the scalar (for rate sources) or vector (for current-density sources) Green's
   function -- the "impedance field" when the output is an open-circuit voltage, the "admittance
   field" when it is a short-circuit current. For spatially uncorrelated sources with one-sided
   local spectral density K(r) [ <s(r) s*(r')> = K(r) delta(r - r') ] the output spectral density
   is the incoherent sum
        S_out(f) = integral |G(r, omega)|^2 K(r) d^3r.                       (Bonani-Ghione 3.x)
   All densities here are ONE-SIDED (unilateral) in f, the convention of the SOA noise code
   (optics/soa/noise_metrics.py) and of Bonani & Ghione's K values.

   G for EVERY source location at once costs one linear solve per frequency: G is a row of
   J(omega)^-1, i.e. the solution of the ADJOINT system J(omega)^T y = e_out, with
   J(omega) = J_static + i omega J_dynamic the ssac matrix (the Green's-function / adjoint technique
   of Bonani, Ghione, Pinto, Smith, IEEE TED 45, 261 (1998); DEVSIM manual sec. 9.2.3 "Noise and
   sensitivity analysis" documents the native solve and its "<out>_<Equation>_real/_imag" fields).

2. LOCAL NOISE SOURCES (Bonani & Ghione 2001, ch. 3; van der Ziel, "Noise in Solid State Devices
   and Circuits", Wiley 1986).
   * Diffusion (velocity-fluctuation) noise, per carrier, uncorrelated between carriers:
        K_n = 4 q^2 D_n n,   K_p = 4 q^2 D_p p     [one-sided, per Cartesian component]
     with the Einstein relation D = mu kT/q, i.e. K = 4 kT q mu n = 4 kT sigma_n -- the
     Nyquist form, which is what makes the equilibrium output obey S = 4 kT Re(Y) (section 4).
     Degenerate regions (FD g-factor on): the NOISE diffusivity stays mu kT/q (Price 1965; Bonani
     & Ghione sec. 3.4: the Nyquist-consistent velocity-fluctuation source), NOT the generalized
     g mu kT/q of the transport model -- so the FDT gate is exact only for g == 1 (Boltzmann).
   * Generation-recombination (GR) noise, one Poissonian net process per mechanism:
        K_gr = 2 (G + R)    [(m^-3 s^-1)^2 m^3 / Hz],
     entering the electron and hole continuity equations FULLY CORRELATED (one event removes or
     creates an electron AND a hole: delta gamma_n = delta gamma_p). For SRH this is the standard
     reduction for a trap level whose occupancy follows the carriers instantaneously (the DEVSIM
     SRH model has no trap-occupancy variable): with U = (np - n_ie^2)/D the split is
     R = np/D, G = n_ie^2/D, so G + R = (np + n_ie^2)/D. The same split for Auger
     ((C_n n + C_p p)(np +- n_ie^2)) and radiative (B (np +- n_ie^2)). At equilibrium
     2(G + R) = 4 R_eq = 4 kT x (the linear-response GR conductance), so the GR source is exactly
     Nyquist-consistent too. An external generation G_ext (BipolarOptions.generation_m3s) is
     counted as a Poisson process (photogeneration shot noise) unless external_generation_shot=
     False (default True, kept by user decision 2026-11-04; see noise_spectrum's docstring for
     when to switch it off).
     v0.14 integration: the sources come from the ONE additive term mechanism of the region
     (physics_bipolar_dd.recombination_terms): the base SRH channel (physics_traps.srh_rate_expr(
     pair=True)) plus the pair rate PTerm_* every term carries -- Auger, radiative, external G, each
     bulk TrapLevel (any E_t), interface-trap surface SRH, and BTBT (edge/element assembled, its
     Hurkx split (np +- n_ie^2)/((n + n_ie)(p + n_ie))). Each reaches a node's residual with the SAME
     quadrature weights the continuity equations use (NodeVolume / EdgeNodeVolume / ElementNodeVolume).
     A term without a pair rate makes noise_spectrum REFUSE the region (never silently noiseless).
   * Optional 1/f (Hooge 1969, Phys. Lett. A 29, 139; Hooge, IEEE TED 41, 1926 (1994)), OFF unless
     hooge_alpha is given. It is a fluctuation of the local CONDUCTIVITY, not of the carrier velocity:
     delta j(r) = xi(r) J(r) with xi = delta sigma / sigma one scalar per point,
     <xi(r) xi(r')> = alpha_H / (f n) delta(r - r'), so the source tensor is RANK ONE along J,
        K_H = alpha_H J J^T / (n f)   per carrier,     S = Int alpha_H (J . grad y)^2 / (n f) dV
     (derived here from Hooge's empirical law; for an ohmic conductor the adjoint field is the
     normalised potential, giving the familiar non-uniform-field form S_R/R^2 = alpha_H
     Int |E|^4 / (n f) / (Int |E|^2)^2), NOT isotropic like the diffusion source. A uniform resistor gives S_I/I^2 = alpha_H/(N f) at ANY orientation.

3. DISCRETIZATION (the box / Scharfetter-Gummel scheme DEVSIM assembles).
   * A current-density source on edge e (nodes 0 -> 1, Voronoi couple c_e, length L_e) is the
     edge-projected fluctuation averaged over the edge volume c_e L_e, so <|dj_e|^2> = K_e/(c_e L_e);
     it enters the residual of node 0 as +c_e dj_e and of node 1 as -c_e dj_e. Its Green's function
     is c_e (y_0 - y_1) and its contribution is
            S_e = (c_e / L_e) |y_0 - y_1|^2 K_e.
     Sum_e c_e L_e (dy/L_e)^2 is the box scheme's discrete Dirichlet energy, so for a uniform bar the
     edge sum reproduces 4 kT G EXACTLY in the discrete sense (measured |S/4kTG - 1| <= 8.9e-16 on
     non-uniform 1-D, 2-D and unstructured 3-D tetra meshes; validation/devsim_noise.py gate A).
   * THE EDGE CARRIER DENSITY. Inside an SG edge (linear psi, constant J) a distributed source
     dj(x) reaches the nodes weighted by w(x) = exp(-(psi - psi_0)/V_t), and n(x) is the SG
     interpolant; integrating K(x) w^2 / (int w)^2 over the edge gives, in closed form,
            n_eff = 1/2 [ n_0 B(v) + n_1 B(-v) ],      v = vdiff_g (the edge's own SG argument),
            p_eff = 1/2 [ p_0 B(-v_p) + p_1 B(v_p) ],
     each node density times the Bernoulli factor it carries in the flux. At equilibrium the two
     halves are equal (J = 0) and q mu n_eff is EXACTLY the SG edge conductance with respect to the
     quasi-Fermi potential, so the discrete network is Nyquist-exact (a plain arithmetic average of
     n_0, n_1 is not: it misses the junction's exponential profile). Off equilibrium it is the exact
     SG-segment integral, so no extra assumption enters under bias.
   * HOOGE IS ASSEMBLED PER ELEMENT, not per edge (v0.14.0 blocker fix, 2026-11-04). The edge form
     above treats every edge as an independent source -- right for the isotropic diffusion source,
     whose K is a multiple of the identity, and WRONG for the rank-one Hooge tensor: one element's
     edge currents share ONE xi, so the edge form drops their cross terms. For a uniform field at angle
     theta to the mesh axes the edge sum is (cos^4 + sin^4) of the truth -- MEASURED on 6595bde (uniform
     2-D DD resistor on a right-isosceles mesh, exact DEVSIM volumes): S f N/(alpha I^2) = 1.0000 at
     0 deg, 0.5000 at 45 deg; 0.688 on the gmsh-tet 3-D bar of gate A. Now, with c_eT the element edge
     couple (ElementEdgeCouple), a_e = J_e / sqrt(n_eff) (the edge-projected current and the SG
     density above) and V_T = Sum_e c_eT L_e / d (the element volume; d = dimension):
            S_T = (alpha_H / f) |Sum_{e in T} c_eT a_e (y_0 - y_1)|^2 / V_T .
     Sum_e c_eT L_e t_e t_e^T = V_T I on a well-centred element (the box-scheme identity), so for a
     uniform J and n this is exactly V_T alpha_H (J . grad y)^2 / (n f) at any orientation. In 1-D the
     element IS the edge (V = c L) and S_T is the old edge term identically; on a rectilinear 2-D
     right-triangle mesh (and Kuhn tets in 3-D) every element holds one edge per axis, so an
     axis-aligned current reproduces the edge sum too. MEASURED after the fix: 45 deg 1.0000 (to
     3.6e-10, the 1 mV drift residual the aligned case also shows); 1-D gate F and the aligned 2-D
     cases unchanged to round-off. Exact only on well-centred elements, like the volumes themselves:
     the gmsh-tet 3-D bar of gate A (sum NodeVolume = 1.225 x its true volume) reads 1.055 (was 0.688).
   * GR is a node source: residual perturbation -q V_i dU (electrons), +q V_i dU (holes), so
            S_i = q^2 V_i |y_p,i - y_n,i|^2 2 (G + R)_i .
   * The edge/node volumes are DEVSIM's own EdgeCouple / EdgeLength / NodeVolume, so the same code
     serves 1-D (per unit area: A^2/Hz/m^2, like ssac's S/m^2), 2-D (per unit depth: A^2/Hz/m) and
     3-D (absolute A^2/Hz).

4. NATIVE vs CUSTOM -- what DEVSIM does and what this module adds.
   DEVSIM 2.11's solve(type="noise", frequency=f, output_node=out) IS the adjoint solve of item 1:
   it factors J(omega)^T, solves with the unit vector on `out`'s circuit equation, and stores y per
   region equation as node models "<out>_<Equation>_real/_imag" (+ element gradients "_gradx/y/z",
   which we do NOT use: they are element-interpolated and are zeroed on contact nodes). Verified
   here against an independent scipy solve of (J_static + i omega J_dynamic)^T y = e_out built from
   ds.get_matrix_and_rhs (tests/test_devsim_noise.py: 1.6e-15 electron field, 2.2e-15 minority-hole
   field once the scipy reference is row/column equilibrated -- unequilibrated scipy is 2e-4 off on
   the minority rows, UMFPACK scales internally). DEVSIM ships NO local noise
   sources -- those (items 2-3) are this module. One fix-up is required: a contact node's bulk row
   is dropped from the system (permutation row -1), so DEVSIM writes y = 0 there. For a GROUNDED
   contact that is the right Green's function (its current is not measured); for a CIRCUIT contact
   the edge current of a contact edge is assembled into the contact's circuit node instead, so the
   correct y there is the adjoint value of that circuit node (read from the circuit solution
   "noise_<out>_real/_imag"; +1 in the resistor, which the Johnson gate pins -- a wrong sign would
   be off by a factor ~(1.9/0.1)^2 on the last edge).

5. OUTPUTS. noise_spectrum(device, f, output_node="V1.I") -> S(f) at any circuit node: a voltage
   source's branch current ("<V>.I", short-circuit current noise S_I [A^2/Hz]) or a node voltage of
   a current-driven contact (open-circuit S_V [V^2/Hz]). terminal_noise() returns both S_I and the
   Norton-to-Thevenin S_V = S_I / |Y|^2 of a voltage-driven two-terminal (Y from ac_analysis's
   ssac, the same operating point). Per-mechanism, per-region totals always; per-node contribution
   maps (|G|^2 K, edge halves split to their two nodes) with maps=True.

MEASURED (validation/devsim_noise.py; tests/test_devsim_noise.py):
  A Johnson-Nyquist S = 4kT G_dc: <= 8.9e-16 (bipolar 1-D, unipolar FD 1-D, bipolar 2-D, 3-D tetra).
  B FDT on a long diode, 1 Hz - 10 GHz: S_I = 4kT Re Y to 4.5e-6, open-circuit S_V = 4kT Re(1/Y) to
    1.0e-5; two regions + carrier-continuous interface to 5.8e-6; with Auger + radiative + BGN 2.3e-6.
  C ideal long Si diode: S = 2qI coth(qV/2kT) to 2.0e-4 for 0.05-0.55 V (full shot 2qI forward);
    van der Ziel S(f) = 4kT Re Y(f) - 2qI to 0.9 % up to omega tau = 126.
  D SRH GR slab: GR part == ambipolar Lorentzian (with contact shorting) to 3e-5, f/f_c 1e-3 .. 100.
  E biased resistor S = 4kT G (1 + 6e-6) vs forward diode S = 2qI = 0.502 x 4kT G_d.
  F Hooge S = alpha I^2/(N f) to 7.6e-6 (1-D); 2-D at 0 and 45 deg to the mesh to 3.6e-10
    (tests/test_devsim_noise_hooge.py; the pre-fix edge sum gave 0.5 at 45 deg).
  G 2-D layered metasurface (FD ITO): FDT to 1.3e-11 at flat band, 3.7e-5 at +2 V accumulation (the
    FD g-factor SG is Nyquist-consistent only to O(h^2)).

LIMITS: the DD source set only (no hot-carrier noise temperature, no trap-occupancy dynamics -- the
fast-trap SRH reduction above; CHARGED trap levels are quasi-static, so the ssac response -- and with
it the FDT -- holds only in the quasi-static regime, MEASURED 5.6e-3 at 100 kHz); a D1a
THERMIONIC-EMISSION heterointerface (fluxterm) carries no noise source -- its shot noise
2q(J_0->1 + J_1->0) is missing, so the FDT misses wherever the interface resistance matters (MEASURED
-0.69 at 1 Hz with A* x 1e-6); the continuous-QFL heterointerface is covered (FDT 5.7e-6); cylindrical volume
models and element-assembled currents are refused (the term mechanism's ElectronGenerationEdge
edge_volume_model is a generation term and allowed); the FD-degenerate
diffusion source uses mu kT/q (gate G: 3.7e-5 FDT residual in accumulation). Multi-region carrier
interfaces work through DEVSIM's permuted rows (gate B3). A biased DD resistor is 4kT G only up to a
physical drift/Debye-screening term (1.1e-3 at 1 um / 0.1 V, ~V^2 L^-2.7). Requires DEVSIM.

LOW-FREQUENCY CONDUCTANCE FOR THE FDT (D2b, docs/audit/2026-11-03-devsim-extended-precision.md). The
noise sum (all positive terms) stays exact where the ssac Re Y does not: a Si-like p+n diode has
G0 = 2.58e-5 S/m^2 but Re Y(1 Hz) = -0.27 S/m^2 driven at the p+ end (~+-6e-5 at the n end). The
noise solve is one double factorization like ssac, and extended precision does NOT lift that floor
(an AdvisoryWarning says so on an opted-in device). Compare 4kT G against the DC dI/dV (+-0.1 mV
central difference of solved DC terminal currents: G0 to 1e-6), and drive the LIGHTLY doped contact.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, Optional, Sequence

import numpy as np

from dynameta.carriers import extended_precision as _XP
from dynameta.carriers.eq_registry import ds
from dynameta.constants import KB, Q_E

__all__ = ["NoiseSpectrum", "TerminalNoise", "noise_spectrum", "terminal_noise", "bernoulli",
           "MECHANISMS"]

#: the source mechanisms this module knows; "hooge" is evaluated only when hooge_alpha is given.
MECHANISMS = ("diffusion_n", "diffusion_p", "gr", "hooge_n", "hooge_p")

_EQ_N, _EQ_P = "ElectronContinuityEquation", "HoleContinuityEquation"
# the additive-term composites physics_bipolar_dd wires into the edge volume slot (1-D BTBT): a
# generation term, not a cylindrical volume -- allowed
_EDGE_GEN = {_EQ_N: "ElectronGenerationEdge", _EQ_P: "HoleGenerationEdge"}


def bernoulli(x):
    """B(x) = x / (exp(x) - 1), B(0) = 1, evaluated without cancellation (expm1; series near 0)."""
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    small = np.abs(x) < 1e-8
    out[small] = 1.0 - 0.5 * x[small]
    xs = x[~small]
    out[~small] = xs / np.expm1(xs)
    return out


@dataclass(frozen=True)
class NoiseSpectrum:
    """One-sided output noise spectral density at circuit node `output_node`.

    S [A^2/Hz for a '<V>.I' branch current, V^2/Hz for a node voltage; per m^2 in 1-D, per m of
    depth in 2-D]; by_mechanism / by_region / by_region_mechanism split S exactly (they sum to it).
    maps (only with maps=True): {region: {"x","y","z": node coords, mechanism: (n_f, n_nodes)
    per-node contributions, edge contributions split half to each end node, Hooge element
    contributions split equally among the element's nodes}}."""
    frequencies: np.ndarray
    S: np.ndarray
    output_node: str
    kind: str                                            # "current" | "voltage"
    by_mechanism: Dict[str, np.ndarray]
    by_region: Dict[str, np.ndarray]
    by_region_mechanism: Dict[str, Dict[str, np.ndarray]]
    temperature_K: Dict[str, float]
    maps: Optional[Dict[str, Dict[str, np.ndarray]]] = field(default=None, repr=False)


@dataclass(frozen=True)
class TerminalNoise:
    """S_I (short-circuit current noise at the driven terminal) with the ssac admittance Y at the
    same operating point, and S_V = S_I / |Y|^2 (open-circuit voltage noise, Norton -> Thevenin)."""
    frequencies: np.ndarray
    S_I: np.ndarray
    S_V: np.ndarray
    Y: np.ndarray
    current: NoiseSpectrum


# =====================================================================================================
# device introspection
# =====================================================================================================

def _nv(device, region, name):
    return np.asarray(ds.get_node_model_values(device=device, region=region, name=name),
                      dtype=np.float64)


def _ev(device, region, name):
    return np.asarray(ds.get_edge_model_values(device=device, region=region, name=name),
                      dtype=np.float64)


def _elv(device, region, name):
    return np.asarray(ds.get_element_model_values(device=device, region=region, name=name),
                      dtype=np.float64)


def _param(device, region, name):
    return float(ds.get_parameter(device=device, region=region, name=name))


def _region_equations(device: str, region: str):
    """The continuity equations of a carrier region ({} for Poisson-only / metal regions)."""
    eqs = set(ds.get_equation_list(device=device, region=region))
    out = {}
    for eq, cur in ((_EQ_N, "ElectronCurrent"), (_EQ_P, "HoleCurrent")):
        if eq not in eqs:
            continue
        cmd = ds.get_equation_command(device=device, region=region, name=eq)
        if cmd.get("edge_model") != cur or cmd.get("element_model"):
            raise NotImplementedError(
                "noise_analysis: region {!r} equation {} assembles its current as edge_model={!r} "
                "element_model={!r}; the noise sources are defined only for the edge-assembled "
                "Scharfetter-Gummel current {!r}.".format(region, eq, cmd.get("edge_model"),
                                                          cmd.get("element_model"), cur))
        if cmd.get("edge_volume_model") not in (None, "", _EDGE_GEN[eq]) or cmd.get("volume_model"):
            raise NotImplementedError(
                "noise_analysis: region {!r} equation {} uses a custom volume model (cylindrical "
                "coordinates?); the edge/node source volumes would be inconsistent -- not "
                "supported.".format(region, eq))
        out[eq] = cmd
    return out


def _edge_nodes(device: str, region: str):
    """(n0, n1) node indices of every edge, from DEVSIM's built-in node_index model."""
    have = set(ds.get_edge_model_list(device=device, region=region))
    made = "node_index@n0" not in have
    if made:
        ds.edge_from_node_model(device=device, region=region, node_model="node_index")
    i0 = _ev(device, region, "node_index@n0").astype(np.int64)
    i1 = _ev(device, region, "node_index@n1").astype(np.int64)
    if made:
        ds.delete_edge_model(device=device, region=region, name="node_index@n0")
        ds.delete_edge_model(device=device, region=region, name="node_index@n1")
    return i0, i1


def _pair_term_models(device: str, region: str, external_generation_shot: bool) -> Dict[str, list]:
    """The PAIR-rate (G + R) models of every additive recombination term the region carries, by
    assembly ({"node": [...], "edge": [...], "element": [(pair0, pair1), ...]}), read from the ONE term
    mechanism (physics_bipolar_dd.recombination_terms) -- Auger, radiative, external G, every bulk
    TrapLevel, interface-trap surface SRH, BTBT. A term WITHOUT a pair rate makes the noise refuse the
    region (it would otherwise be silently noiseless and break the FDT)."""
    from dynameta.carriers.bipolar_options import GEXT_TERM
    from dynameta.carriers.physics_bipolar_dd import recombination_terms
    terms = recombination_terms(device, region)
    out: Dict[str, list] = {"node": [], "edge": [], "element": []}
    for kind, entries in terms.items():
        for name, (_rate, pm) in entries.items():
            if pm is None:
                raise RuntimeError(
                    "noise_analysis: region {!r} carries the {} recombination term {!r} without a pair "
                    "rate G + R (physics_bipolar_dd.add_recombination_term(..., pair=...)); its GR "
                    "noise would be silently missing -- refusing.".format(region, kind, name))
            if name == GEXT_TERM and not external_generation_shot:
                continue                                        # deterministic external generation
            out[kind].append(pm)
    return out


def _gr_pair_per_node(device: str, region: str, external_generation_shot: bool, n_nodes: int,
                      i0, i1) -> Optional[np.ndarray]:
    """Volume-integrated pair rate P_i = Int (G + R) [1/s] each node's residual receives, summed over
    every recombination mechanism of the region, assembled with the SAME weights the continuity
    equations use: NodeVolume for node terms (the base SRH channel -- physics_traps.srh_rate_expr(
    pair=True), the one SRH kernel with the region's BGN substitution -- and every PTerm_*),
    EdgeNodeVolume to both end nodes for edge terms, ElementNodeVolume to the element edge's
    node0/node1 for element terms. None for a region without SRH."""
    nodes = set(ds.get_node_model_list(device=device, region=region))
    if "USRH" not in nodes:
        return None
    from dynameta.carriers.physics_traps import srh_rate_expr
    pm = _pair_term_models(device, region, external_generation_shot)
    ex = " + ".join(["({})".format(srh_rate_expr(bgn="n_ie" in nodes, pair=True))] + pm["node"])
    nm = "_NoiseGRPairRate"
    ds.node_model(device=device, region=region, name=nm, equation=ex)
    P = _nv(device, region, nm) * _nv(device, region, "NodeVolume")
    ds.delete_node_model(device=device, region=region, name=nm)
    if pm["edge"]:
        w = sum(_ev(device, region, m) for m in pm["edge"]) * _ev(device, region, "EdgeNodeVolume")
        np.add.at(P, i0, w)
        np.add.at(P, i1, w)
    if pm["element"]:
        have = set(ds.get_element_model_list(device=device, region=region))
        made = "node_index@en0" not in have
        if made:
            ds.element_from_node_model(device=device, region=region, node_model="node_index")
        env = _elv(device, region, "ElementNodeVolume")
        for k in (0, 1):
            idx = _elv(device, region, "node_index@en{}".format(k)).astype(np.int64)
            np.add.at(P, idx, sum(_elv(device, region, pair[k]) for pair in pm["element"]) * env)
        if made:
            for m in ds.get_element_model_list(device=device, region=region):
                if m.startswith("node_index@en"):
                    ds.delete_element_model(device=device, region=region, name=m)
    if P.size != n_nodes:                                       # pragma: no cover - DEVSIM invariant
        raise RuntimeError("noise_analysis: pair-rate size mismatch on region {!r}".format(region))
    return P


def _circuit_contacts(device: str):
    """{contact: {equation: circuit_node}} for every contact equation coupled to a circuit node."""
    out: Dict[str, Dict[str, str]] = {}
    for c in ds.get_contact_list(device=device):
        for eq in ds.get_contact_equation_list(device=device, contact=c):
            cmd = ds.get_contact_equation_command(device=device, contact=c, name=eq)
            if cmd.get("circuit_node"):
                out.setdefault(c, {})[eq] = cmd["circuit_node"]
    return out


def _contact_nodes(device: str, region: str, contact: str) -> np.ndarray:
    try:
        el = ds.get_element_node_list(device=device, region=region, contact=contact)
    except Exception:                                           # contact not on this region
        return np.zeros(0, dtype=np.int64)
    return np.unique(np.asarray([i for e in el for i in e], dtype=np.int64))


# =====================================================================================================
# core
# =====================================================================================================

def _hooge_elements(device: str, region: str, i0, i1, couple, length):
    """Per-element view of the edges for the rank-one Hooge source (module doc, item 3):
    (E, C, V, nodes) with E[T, k] the global edge index of element T's k-th edge, C[T, k] its element
    couple c_eT, V[T] = Sum_k C L / d the element volume, nodes[T] the element's nodes. In 1-D DEVSIM
    has no element models and the element IS the edge (C = EdgeCouple, V = c L)."""
    dim = int(ds.get_dimension(device=device))
    if dim == 1:
        return (np.arange(i0.size)[:, None], couple[:, None], couple * length,
                np.stack([i0, i1], axis=1))
    have = set(ds.get_element_model_list(device=device, region=region))
    made = "node_index@en0" not in have
    if made:
        ds.element_from_node_model(device=device, region=region, node_model="node_index")
    e0 = _elv(device, region, "node_index@en0").astype(np.int64)
    e1 = _elv(device, region, "node_index@en1").astype(np.int64)
    cT = _elv(device, region, "ElementEdgeCouple")
    if made:
        for m in ds.get_element_model_list(device=device, region=region):
            if m.startswith("node_index@en"):
                ds.delete_element_model(device=device, region=region, name=m)
    nodes = np.asarray(ds.get_element_node_list(device=device, region=region), dtype=np.int64)
    k = dim * (dim + 1) // 2                                   # edges per triangle / tetrahedron
    n_el = nodes.shape[0]
    M = int(max(i0.max(), i1.max())) + 1
    gkey = np.minimum(i0, i1) * M + np.maximum(i0, i1)
    order = np.argsort(gkey)
    ekey = np.minimum(e0, e1) * M + np.maximum(e0, e1)
    pos = np.minimum(np.searchsorted(gkey[order], ekey), gkey.size - 1)
    # DEVSIM layout assumption, CHECKED rather than trusted: element-edge values come in blocks of k per
    # element in get_element_node_list order, and every element edge is a region edge
    ok = (nodes.ndim == 2 and nodes.shape[1] == dim + 1 and e0.size == n_el * k
          and bool(np.all(gkey[order][pos] == ekey))
          and bool(np.all((nodes[:, :, None] == e0.reshape(n_el, k)[:, None, :]).any(axis=1)))
          and bool(np.all((nodes[:, :, None] == e1.reshape(n_el, k)[:, None, :]).any(axis=1))))
    if not ok:                                                  # pragma: no cover - DEVSIM invariant
        raise RuntimeError("noise_analysis: DEVSIM element-edge layout on region {!r} is not the "
                           "documented one -- the Hooge element assembly refuses to guess."
                           .format(region))
    E = order[pos].reshape(n_el, k)
    C = cT.reshape(n_el, k)
    return E, C, (C * length[E]).sum(axis=1) / dim, nodes


class _RegionData:
    """Frequency-independent source data of one carrier region (the DC operating point)."""

    def __init__(self, device, region, eqs, *, mechanisms, hooge_alpha, external_generation_shot):
        self.region = region
        self.eqs = eqs
        self.i0, self.i1 = _edge_nodes(device, region)
        couple = _ev(device, region, "EdgeCouple")
        length = _ev(device, region, "EdgeLength")
        self.geo_edge = couple / length                         # c_e / L_e
        self.vol = _nv(device, region, "NodeVolume")
        self.n_nodes = self.vol.size
        V_t = _param(device, region, "V_t")
        self.T_K = V_t * Q_E / KB
        self.edge_K: Dict[str, np.ndarray] = {}                 # mechanism -> K_e (per edge)
        self.edge_eq: Dict[str, str] = {}
        self.hooge: Dict[str, np.ndarray] = {}                  # mechanism -> a_e, a_e^2 = K_e f
        edges = set(ds.get_edge_model_list(device=device, region=region))
        if _EQ_N in eqs:
            mu = _ev(device, region, "mu_n_edge") if "mu_n_edge" in edges else _param(
                device, region, "mu_n")
            v = _ev(device, region, "vdiff_g")
            n_eff = 0.5 * (_ev(device, region, "Electrons@n0") * bernoulli(v)
                           + _ev(device, region, "Electrons@n1") * bernoulli(-v))
            if "diffusion_n" in mechanisms:
                self.edge_K["diffusion_n"] = 4.0 * Q_E * Q_E * V_t * mu * n_eff
                self.edge_eq["diffusion_n"] = _EQ_N
            if hooge_alpha is not None and "hooge_n" in mechanisms:
                J = _ev(device, region, "ElectronCurrent")
                self.hooge["hooge_n"] = math.sqrt(float(hooge_alpha)) * J / np.sqrt(n_eff)
                self.edge_eq["hooge_n"] = _EQ_N
        if _EQ_P in eqs:
            mu = _ev(device, region, "mu_p_edge") if "mu_p_edge" in edges else _param(
                device, region, "mu_p")
            v = _ev(device, region, "vdiff_gp")
            p_eff = 0.5 * (_ev(device, region, "Holes@n0") * bernoulli(-v)
                           + _ev(device, region, "Holes@n1") * bernoulli(v))
            if "diffusion_p" in mechanisms:
                self.edge_K["diffusion_p"] = 4.0 * Q_E * Q_E * V_t * mu * p_eff
                self.edge_eq["diffusion_p"] = _EQ_P
            if hooge_alpha is not None and "hooge_p" in mechanisms:
                J = _ev(device, region, "HoleCurrent")
                self.hooge["hooge_p"] = math.sqrt(float(hooge_alpha)) * J / np.sqrt(p_eff)
                self.edge_eq["hooge_p"] = _EQ_P
        # the Hooge source's element structure (module doc, item 3): one xi per element
        self.hooge_el = (_hooge_elements(device, region, self.i0, self.i1, couple, length)
                         if self.hooge else None)
        self.gr = None                          # 2 Int (G + R) per node [1/s] (every mechanism)
        if "gr" in mechanisms and _EQ_N in eqs and _EQ_P in eqs:
            P = _gr_pair_per_node(device, region, external_generation_shot, self.n_nodes,
                                  self.i0, self.i1)
            if P is not None:
                self.gr = 2.0 * P


def _green(device, region, out, eq, n_nodes, fixups):
    """Complex adjoint field y of equation `eq` in `region` from DEVSIM's noise solve, with the
    circuit-contact fix-up (see module doc, item 4)."""
    base = "{}_{}".format(out, eq)
    y = _nv(device, region, base + "_real") + 1j * _nv(device, region, base + "_imag")
    for nodes, val in fixups.get((region, eq), ()):
        if np.any(y[nodes] != 0.0):
            raise RuntimeError(
                "noise_analysis: DEVSIM wrote a nonzero impedance field on contact nodes of region "
                "{!r} ({}); the contact fix-up assumes the documented zero (DEVSIM semantics "
                "changed?) -- refusing to guess.".format(region, eq))
        y[nodes] = val
    return y


def _cleanup_noise_models(device, regions, out):
    for r in regions:
        for nm in ds.get_node_model_list(device=device, region=r):
            if nm.startswith(out + "_"):
                ds.delete_node_model(device=device, region=r, name=nm)


def noise_spectrum(device: str, frequencies, *, output_node: str = "V1.I",
                   regions: Optional[Iterable[str]] = None,
                   mechanisms: Sequence[str] = ("diffusion_n", "diffusion_p", "gr"),
                   hooge_alpha: Optional[float] = None, external_generation_shot: bool = True,
                   maps: bool = False) -> NoiseSpectrum:
    """One-sided noise spectral density at circuit node `output_node` of `device`, at its CURRENT
    DC operating point (solve DC first, exactly as for ac_analysis.ssac_admittance), by the
    impedance-field method (module doc). `output_node` is a voltage source's branch current
    ("V1.I": short-circuit terminal current noise S_I) or a circuit node voltage (open-circuit S_V
    of a CURRENT-driven contact). `regions` defaults to every region carrying continuity equations.
    `mechanisms` picks the local sources (MECHANISMS); "hooge_n"/"hooge_p" are added automatically
    when hooge_alpha (the dimensionless Hooge parameter) is given. The DC state is not modified.

    external_generation_shot (default True; kept by user decision 2026-11-04). Applies only to a
    region carrying an external generation G_ext (BipolarOptions.generation_m3s, the additive term
    bipolar_options.GEXT_TERM) and only through the "gr" mechanism.
      True  -- G_ext is a POISSON process: every generated pair is an independent random event (the
               photons of a coherent or thermal beam absorbed one at a time), so it contributes the
               GR source K = 2 G_ext [(m^-3 s^-1)^2 m^3/Hz], fully correlated between the electron
               and hole equations like every pair process. A photodiode that collects every pair then
               shows the photocurrent shot noise S_I = 2 q I_ph on top of its dark noise. This is the
               physical choice whenever G_ext stands for real optical absorption.
      False -- G_ext is DETERMINISTIC: it still sets the DC operating point (n, p, and through them
               every other source), but adds no noise of its own. Set it when G_ext is a modelling
               device rather than a photon stream -- a prescribed drive or injection rate, a ramp /
               continuation aid, a stand-in for a noiseless pump -- or when the source statistics are
               accounted for elsewhere (sub-Poissonian light, a separate RIN term), so they are not
               counted twice.
    Not validated: no oracle in tests/ or validation/ exercises G_ext noise (the 2 q I_ph limit is the
    textbook expectation, not a measured gate; docs/audit/2026-11-03-devsim-noise.md, open question
    1)."""
    freqs = np.atleast_1d(np.asarray(frequencies, dtype=np.float64))
    if freqs.size == 0 or np.any(~np.isfinite(freqs)) or np.any(freqs <= 0.0):
        raise ValueError("noise_spectrum: frequencies must be finite and > 0 Hz")
    mech = set(mechanisms)
    unknown = mech - set(MECHANISMS)
    if unknown:
        raise ValueError("noise_spectrum: unknown mechanism(s) {}; known: {}".format(
            sorted(unknown), MECHANISMS))
    if hooge_alpha is not None:
        if not (math.isfinite(float(hooge_alpha)) and float(hooge_alpha) > 0.0):
            raise ValueError("noise_spectrum: hooge_alpha must be finite and > 0")
        mech |= {"hooge_n", "hooge_p"}
    if output_node not in ds.get_circuit_node_list():
        raise ValueError("noise_spectrum: {!r} is not a circuit node; have {} (attach a circuit "
                         "contact first: ac_analysis.setup_circuit_contact / physics_bipolar_dd."
                         "setup_contact_ohmic_bipolar_circuit / physics_drift_diffusion."
                         "setup_contact_ohmic_dd_circuit).".format(
                             output_node, list(ds.get_circuit_node_list())))
    kind = "current" if output_node.endswith(".I") else "voltage"

    all_regions = list(ds.get_region_list(device=device))
    if regions is None:
        regions = [r for r in all_regions if _region_equations(device, r)]
    regions = list(regions)
    data = []
    for r in regions:
        if r not in all_regions:
            raise ValueError("noise_spectrum: unknown region {!r}".format(r))
        eqs = _region_equations(device, r)
        if not eqs:
            raise ValueError("noise_spectrum: region {!r} has no continuity equation (no carrier "
                             "noise sources)".format(r))
        data.append(_RegionData(device, r, eqs, mechanisms=mech, hooge_alpha=hooge_alpha,
                                external_generation_shot=external_generation_shot))

    circ = _circuit_contacts(device)
    contact_nodes = {}
    for d in data:
        for c, eqmap in circ.items():
            nodes = _contact_nodes(device, d.region, c)
            if nodes.size:
                for eq, cn in eqmap.items():
                    if eq in d.eqs:
                        contact_nodes.setdefault((d.region, eq), []).append((nodes, cn))

    present = sorted({m for d in data for m in list(d.edge_K) + list(d.hooge)}
                     | ({"gr"} if any(d.gr is not None for d in data) else set()))
    nf = freqs.size
    by_rm = {d.region: {m: np.zeros(nf) for m in present} for d in data}
    map_out = None
    if maps:
        map_out = {}
        for d in data:
            mm = {ax: _nv(device, d.region, ax) for ax in ("x", "y", "z")}
            for m in present:
                mm[m] = np.zeros((nf, d.n_nodes))
            map_out[d.region] = mm

    sol_re, sol_im = "noise_{}_real".format(output_node), "noise_{}_imag".format(output_node)
    try:
        for k, f in enumerate(freqs):
            _XP.solve(type="noise", frequency=float(f), output_node=output_node)  # 128-bit iff opted in
            fix = {}
            for key, lst in contact_nodes.items():
                fix[key] = [(nodes, complex(
                    float(ds.get_circuit_node_value(node=cn, solution=sol_re)),
                    float(ds.get_circuit_node_value(node=cn, solution=sol_im))))
                    for nodes, cn in lst]
            for d in data:
                ys = {eq: _green(device, d.region, output_node, eq, d.n_nodes, fix)
                      for eq in d.eqs}
                for m in present:
                    if m in d.hooge:                    # rank-one source: one xi per element
                        y = ys[d.edge_eq[m]]
                        E, C, VT, el_nodes = d.hooge_el
                        A = np.sum(C * (d.hooge[m] * (y[d.i0] - y[d.i1]))[E], axis=1)
                        s_T = np.zeros(VT.size)
                        live = VT > 0.0                 # an all-zero-couple element carries nothing
                        s_T[live] = np.abs(A[live]) ** 2 / VT[live] / float(f)
                        by_rm[d.region][m][k] = float(np.sum(s_T))
                        if maps:
                            nm = np.zeros(d.n_nodes)
                            np.add.at(nm, el_nodes, (s_T / el_nodes.shape[1])[:, None])
                            map_out[d.region][m][k] = nm
                    elif m in d.edge_K:
                        y = ys[d.edge_eq[m]]
                        K = d.edge_K[m]
                        s_e = d.geo_edge * np.abs(y[d.i0] - y[d.i1]) ** 2 * K
                        by_rm[d.region][m][k] = float(np.sum(s_e))
                        if maps:
                            nm = np.zeros(d.n_nodes)
                            np.add.at(nm, d.i0, 0.5 * s_e)
                            np.add.at(nm, d.i1, 0.5 * s_e)
                            map_out[d.region][m][k] = nm
                    elif m == "gr" and d.gr is not None:
                        s_i = Q_E * Q_E * np.abs(ys[_EQ_P] - ys[_EQ_N]) ** 2 * d.gr
                        by_rm[d.region][m][k] = float(np.sum(s_i))
                        if maps:
                            map_out[d.region][m][k] = s_i
    finally:
        _cleanup_noise_models(device, [d.region for d in data], output_node)

    by_m = {m: sum(by_rm[r][m] for r in by_rm) for m in present}
    by_r = {r: sum(by_rm[r][m] for m in present) for r in by_rm}
    S = sum(by_m.values()) if by_m else np.zeros(nf)
    return NoiseSpectrum(frequencies=freqs, S=np.asarray(S, dtype=np.float64),
                         output_node=output_node, kind=kind, by_mechanism=by_m, by_region=by_r,
                         by_region_mechanism=by_rm,
                         temperature_K={d.region: d.T_K for d in data}, maps=map_out)


def terminal_noise(device: str, frequencies, *, source_name: str = "V1", v_ac: float = 1.0,
                   **kw) -> TerminalNoise:
    """S_I(f) at the voltage-driven terminal `source_name` (noise_spectrum on '<source>.I'), the
    ssac admittance Y(f) = G + i omega C at the same operating point (ac_analysis.ssac_admittance),
    and the open-circuit S_V = S_I / |Y|^2 (a linear two-terminal's Norton -> Thevenin). Extra
    keywords go to noise_spectrum -- including external_generation_shot (default True: an external
    generation G_ext is Poisson photogeneration and adds its shot noise 2 G_ext to the GR source;
    False: G_ext is a deterministic drive that only sets the operating point -- noise_spectrum's
    docstring says when to use which)."""
    from dynameta.carriers.ac_analysis import ssac_admittance
    sn = noise_spectrum(device, frequencies, output_node="{}.I".format(source_name), **kw)
    f, C, G = ssac_admittance(sn.frequencies, source_name=source_name, v_ac=v_ac)
    Y = G + 1j * 2.0 * np.pi * f * C
    return TerminalNoise(frequencies=sn.frequencies, S_I=sn.S, S_V=sn.S / np.abs(Y) ** 2, Y=Y,
                         current=sn)
