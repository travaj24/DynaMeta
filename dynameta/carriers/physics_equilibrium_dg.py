"""In-Newton density gradient for EQUILIBRIUM (Potential-only) semiconductor regions.

An equilibrium region (physics_equilibrium.setup_semiconductor_region) has the potential psi as its only
unknown and the electron density as a closed function of it, n = N_c F_1/2(eta), eta = (psi - [EF_ref] -
Phi_c0)/V_t. The density-gradient (DG) model adds the quantum potential Lambda to the electrons' effective
band edge,

    n = N_c F_1/2(eta + Lambda/V_t),        Lambda = b lap(sqrt n) / sqrt n,
    b = gamma hbar^2 / (6 m q)  [V m^2]     (physics_density_gradient.dg_b_coefficient).

Prefactor convention: Lambda is in volts and acts like a potential (E_c,eff = E_c - q Lambda), so the
Bohm-type term reads Lambda = -(Q / q) with Q = -(gamma hbar^2 / (6 m)) lap(sqrt n)/sqrt n, i.e. the
"r = 3" form of hbar^2/(2 m r) with the fit factor gamma in front (gamma = 1: the Wigner / Ancona value).
Near an insulating wall lap(sqrt n)/sqrt n < 0, so Lambda < 0 and electrons are pushed off the wall.
Since the density is a closed function of (psi, Lambda), there is no continuity equation; the DG enters as
two extra nodal unknowns, u = sqrt(n) (QSqrtN) and Lambda (QLambda), with

    Poisson     div(eps grad psi) = q (n(psi, Lambda) - N_D) - (fixed / trap charge terms)  [unchanged row]
    u-row       u^2 - n(psi, Lambda) = 0                                  (the closure tie)
    Lambda-row  b lap(u) - Lambda u = 0                                   (the shared DG kernel)

The Lambda-row is physics_density_gradient._install_dg_variables (the kernel of the drift-diffusion and
bipolar DG, edge flux b grad(u) + node term Lambda u, converged Lambda = +b u''/u), with the tie replaced
by the equilibrium closure. Every Jacobian entry is analytic: the closure's derivatives are DEVSIM's own
diff() of the Aymerich-Humet expression (the "Electrons" node model is redefined on eta + Lambda/V_t, with
Electrons:Potential and Electrons:QLambda), Poisson gains PotentialNodeCharge:QLambda, and DEVSIM's diff()
carries those names into the tie's derivatives. gamma = 0 is never installed (the classical region is
left untouched, so it is bit-identical); a b ramp (physics_density_gradient.set_dg_gamma) passes through
b = 0, where Lambda = 0 off the walls.

INSULATING WALLS. At every node of the region on an interface with a DIELECTRIC region the carrier
wavefunction vanishes (infinite barrier, no penetration): u = u_floor (a tiny positive floor, u_floor_rel
sqrt(n_bg), because u's update is 'positive'), the tie keeps setting Lambda there (Lambda_wall ~ V_t
ln(u_floor^2/n): the finite stand-in for the log divergence), and the Lambda-row is replaced. DEVSIM cannot
place a contact on an interface edge (a contact there deletes the interface, MEASURED), so the row is
replaced through a node mask: the node term at a wall node is DgWall * DgWallK * (u - u_floor) instead of
Lambda u, with DgWallK = 1e10 b_full (A_node/NodeVolume)^2, which outweighs the wall node's edge fluxes by
2e10 (b_full/b): the wall Dirichlet then holds to 1e-10 of the neighbouring u, far below the floor.

OHMIC CONTACTS. A DEVSIM contact on the region (the 3-D builder's body contact, a potential-pinned edge
electrode) is a carrier reservoir: Lambda = 0 and u^2 = n there (contact rows; the Potential row is the
contact's own).

WHAT IS NOT SUPPORTED (refused): a Kane band (the quantization mass and the calibration below are those of
a parabolic band) and a region with an interface to any region that is not a dielectric -- another semiconductor
region, or a region without recorded Poisson physics such as a metal (u and Lambda would need interface
conditions there).

CALIBRATION (gamma for degenerate films): EQUILIBRIUM_DG_GAMMA = 1.8 with the hard wall, fitted to
carriers.sp_dual_gate.DualGateSchrodingerPoisson (validation/equilibrium_density_gradient.py); inside its validity
range ((E_F - E_c)/kT >= 7 and t k_F/pi >= 2, check_equilibrium_dg_validity, which warns at setup otherwise) the
sheet charge is within 1 % and the profile L1 within 0.12 of the Schrodinger-Poisson film's carrier sheet. Notes:
docs/audit/2026-11-14-v0.16-equilibrium-density-gradient.md.
"""

from __future__ import annotations

import math
import warnings
from typing import Optional

import numpy as np

from dynameta.carriers import eq_registry as _R
from dynameta.carriers import physics_density_gradient as PDG
from dynameta.carriers import physics_equilibrium as PE
from dynameta.carriers.devsim_env import import_devsim
from dynameta.warning_types import UnreliableResultWarning

ds = import_devsim()

__all__ = ["setup_equilibrium_dg", "seed_equilibrium_dg", "set_equilibrium_dg_fraction", "adaptive_continuation",
           "equilibrium_dg_regions", "equilibrium_dg_wall_nodes", "equilibrium_dg_length_m", "equilibrium_dg_regime",
           "check_equilibrium_dg_validity", "EQUILIBRIUM_DG_GAMMA", "VALID_ETA_MIN", "VALID_SUBBANDS_MIN",
           "BUDGET_DSHEET", "BUDGET_L1", "DG_RAMP", "ETA_Q_FLOOR", "DG_WALL_PENALTY", "DG_WALL", "DG_WALL_K",
           "DG_U_FLOOR"]

DG_WALL = "DgWall"            # node solution: 1 at an insulating-wall node, 0 elsewhere
DG_WALL_K = "DgWallK"         # node solution: the wall-row penalty K = DG_WALL_PENALTY b_full (A/V)^2
DG_U_FLOOR = "dg_u_floor"     # region parameter: u at a wall node
DG_WALL_PENALTY = 1.0e10      # K V / (b A / h) = 2 DG_WALL_PENALTY at b = b_full (module header)
_EQDG_FLAG = "eqdg_gamma"     # region parameter: marks an equilibrium DG region (value = gamma)

# EQUILIBRIUM_DG_GAMMA -- the default gamma of the equilibrium DG, a CALIBRATION (not derived), together with the
# hard insulating wall (u = 0 at the dielectric interface, no penetration) as the wall condition.
#   FITTED TO: carriers.sp_dual_gate.DualGateSchrodingerPoisson (parabolic band, hard walls, 300 K, all sub-bands;
#     h = 0.0125 nm) on the same film and gate stacks, the DG on the 2-D builder's default (screening-length) mesh.
#   MATRIX: validation/equilibrium_density_gradient.py --full (fixture validation/_eqdg_fixture.py): t = 3/6/10/20/40
#     nm x n_bg = 1e26/3e26/1e27 m^-3 x m = 0.2/0.30 m_e on a two-layer stack (6 nm eps 20 + 2 nm eps 9, C = 1.70e-2
#     F/m^2) on both faces, 8 bias pairs from full depletion (a 3 nm 1e26 film) to strong accumulation (one gate
#     or both); plus a 2 nm eps 25 stack (t = 3/6/20 nm, 1e26/1e27, -1.5 .. +3 V): 276 points, every
#     one converged; flat-band (E_F - E_c)/kT 10.0 .. 70.5, t k_F/pi 1.37 .. 39.4
#     (docs/audit/2026-11-14-v0.16-equilibrium-density-gradient.md has the per-film table).
#   HOW: at each point the gamma* at which the DG sheet excess equals Schrodinger-Poisson's (linear interpolation
#     of the sheet error on the scan gamma = 1, 1.5, 1.7, 1.8, 1.9, 2.1, 2.4). Over the 236 points inside the
#     validity range with a sign change on the scan: median 1.79 (10-90 %: 1.73 .. 1.89; extremes 1.64 / 2.08 at
#     points where the sheet hardly depends on gamma); outside the range (35 points) 1.35 .. 2.15. 1.8 is that
#     median and the minimax of the worst in-range sheet error on the scan (0.44 % at 1.8 against 0.73 % / 0.68 % at
#     1.7 / 1.9). ONE value holds across bias, thickness, density and mass: no trend of gamma* with any of them
#     beyond the 10-90 % band.
#   MEASURED at 1.8, inside the validity range (VALID_ETA_MIN, VALID_SUBBANDS_MIN): sheet-excess error <= 0.44 % of
#     max(n_bg t, the SP carrier sheet) (the worst points are the thin films on the 2 nm eps 25 stack and the 6 nm
#     1e26 films: the highest fields on the fewest sub-bands), profile L1 <= 0.110 of the same (Schrodinger-
#     Poisson's Friedel oscillations, which no local model has, set most of it); the classical model on the same
#     points: up to 19.1 % and 0.51. Outside the range (3 nm films, t k_F/pi < 2): up to 2.97 % and 0.18.
#   PHYSICS: with the Fermi-Dirac closure the DG dark space at a hard wall in a degenerate gas (frozen potential,
#     T -> 0: n ~ (E_F + q Lambda)^(3/2)) integrates in closed form: deficit = sqrt(gamma/3) I_TF / k_F with
#     I_TF = int_0^1 (1 - w^2)/sqrt(3 w^(10/3)/5 - w^2 + 2/5) dw = 1.64378; equating it to the exact free-gas
#     (Friedel) deficit 3 pi/(8 k_F) gives gamma_TF = 3 (3 pi/(8 I_TF))^2 = 1.541 -- a priori, the same order as the
#     Boltzmann wall value GAMMA_WALL (1.67), and NOT the "statistics-consistent" gamma ~ 1/g a BOLTZMANN closure
#     needs (density_gradient module header): the FD closure carries the degeneracy itself. The SP-fitted 1.8 is
#     17 % above gamma_TF: in a gated film the occupied sub-bands and the self-consistent field push the
#     first-order coefficient up from the free-gas wall value (as GAMMA_SI100_SP_CALIBRATED does for Si
#     inversion layers). Gated: tests/test_equilibrium_density_gradient.py (I_TF, gamma_TF, and the DEVSIM rows
#     against the finite-T first integral, which tends to the T = 0 value: 0.35722 vs 0.35810 at eta 21, 0.23965 vs
#     0.23973 at eta 47, in n_b sqrt(b/V_t); m = 0.30 m_e at 3e26 / 1e27 m^-3).
EQUILIBRIUM_DG_GAMMA = 1.8

# the validity range of the default gamma (check_equilibrium_dg_validity: flat-band (E_F - E_c)/kT and t k_F/pi,
# the matrix above covered 10.0 .. 70.5 and 1.37 .. 39.4) and the error budget inside it (validation/
# equilibrium_density_gradient.py GATE A; both relative to max(n_bg t, the Schrodinger-Poisson carrier sheet))
VALID_ETA_MIN = 7.0
VALID_SUBBANDS_MIN = 2.0
BUDGET_DSHEET = 0.015
BUDGET_L1 = 0.15

# the floor of the closure argument eta + Lambda/V_t (a Newton-transient overflow guard; setup_equilibrium_dg)
ETA_Q_FLOOR = -600.0

# the b ramp of a first (zero-bias) solve, after the b = 0 classical solve (LayeredDevsimBuilder)
DG_RAMP = (0.25, 0.5, 1.0)


def _is_equilibrium_region(device: str, region: str) -> bool:
    try:
        PE.equilibrium_eta_expr(device, region)
    except ValueError:
        return False
    return not _R.recorded(device, "ElectronContinuityEquation", scope="region", loc=region)


def equilibrium_dg_regions(device: str) -> list:
    """The regions of `device` carrying the equilibrium density gradient (setup_equilibrium_dg)."""
    return [r for r in ds.get_region_list(device=device)
            if _EQDG_FLAG in ds.get_parameter_list(device=device, region=r)]


def _dielectric_neighbours(device: str, region: str):
    """(interfaces to dielectric regions, interfaces to any other kind of region) of `region`."""
    diel, other = [], []
    for itf in ds.get_interface_list(device=device):
        regs = list(ds.get_region_list(device=device, interface=itf))
        if region not in regs:
            continue
        nb = regs[1] if regs[0] == region else regs[0]
        try:
            kind = PE._poisson_region_kind(device, nb)
        except ValueError:
            kind = "unknown"
        (diel if kind == "dielectric" else other).append(itf)
    return diel, other


def _neighbour_description(device: str, region: str, itf: str) -> str:
    """"<interface> -> region <name> (<what it is>)" for a refusal message: the neighbour's Poisson kind, or no
    recorded physics (a metal or an unset region)."""
    regs = list(ds.get_region_list(device=device, interface=itf))
    nb = regs[1] if regs[0] == region else regs[0]
    try:
        kind = PE._poisson_region_kind(device, nb)
    except ValueError:
        return "{!r} -> region {!r}: no recorded Poisson physics (a metal or an unset region)".format(itf, nb)
    what = {"equilibrium": "an equilibrium semiconductor", "drift_diffusion": "a drift-diffusion semiconductor",
            "bipolar": "a bipolar semiconductor"}.get(kind, kind)
    return "{!r} -> region {!r}: {}".format(itf, nb, what)


def equilibrium_dg_wall_nodes(device: str, region: str) -> np.ndarray:
    """Boolean mask of the region's insulating-wall nodes (DgWall == 1)."""
    return np.asarray(ds.get_node_model_values(device=device, region=region, name=DG_WALL)) > 0.5


def setup_equilibrium_dg(device: str, region: str, *, m_eff_kg: float, gamma: float = EQUILIBRIUM_DG_GAMMA,
                         u_floor_rel: float = 1.0e-6) -> None:
    """Install the in-Newton density gradient on the equilibrium region `region` (module header).

    Call AFTER physics_equilibrium.setup_semiconductor_region on the region and AFTER the device's
    interfaces exist (the insulating walls are read from them); fixed charges, work functions, the
    Fermi-level reference and film interface traps may be installed before or after (the traps keep the
    classical eta: a localized state's occupancy depends on E_F - E_c, not on the carriers' quantum
    potential). m_eff_kg is the QUANTIZATION mass (the effective mass normal to the walls); gamma the DG
    fit factor (EQUILIBRIUM_DG_GAMMA: the calibrated default). gamma == 0 installs nothing.

    Seed with seed_equilibrium_dg after setting the bias / before the first solve."""
    PE.require_positive(m_eff_kg=m_eff_kg)
    if not (math.isfinite(float(gamma)) and gamma >= 0.0):
        raise ValueError("setup_equilibrium_dg: gamma must be finite and >= 0, got {!r}".format(gamma))
    if not (0.0 < u_floor_rel < 1.0):
        raise ValueError("setup_equilibrium_dg: u_floor_rel must be in (0, 1)")
    if not _is_equilibrium_region(device, region):
        raise ValueError("setup_equilibrium_dg: region {!r} is not an equilibrium semiconductor region "
                         "(physics_equilibrium.setup_semiconductor_region); drift-diffusion regions use "
                         "physics_density_gradient".format(region))
    if gamma == 0.0:
        return
    models = ds.get_node_model_list(device=device, region=region)
    if "KaneDelta" in models:
        raise NotImplementedError(
            "setup_equilibrium_dg: region {!r} uses a Kane band. The equilibrium density gradient supports the "
            "parabolic closure only: its quantization mass and its calibration against Schrodinger-Poisson are "
            "those of a parabolic band.".format(region))
    if "QSqrtN" in models:
        raise ValueError("setup_equilibrium_dg: region {!r} already carries density-gradient variables".format(
            region))
    contacts = [c for c in ds.get_contact_list(device=device)
                if region in ds.get_region_list(device=device, contact=c)]
    diel, semi = _dielectric_neighbours(device, region)
    if semi:
        raise NotImplementedError(
            "setup_equilibrium_dg: region {!r} has interface(s) {} to a region that is not a dielectric ({}); "
            "the density gradient would need interface conditions for sqrt(n) and Lambda there (not "
            "implemented). Model the film as one region with dielectric neighbours.".format(
                region, semi, "; ".join(_neighbour_description(device, region, itf) for itf in semi)))

    b_full = PDG.dg_b_coefficient(m_eff_kg, gamma)
    eta = PE.equilibrium_eta_expr(device, region)
    n_bg = float(ds.get_parameter(device=device, region=region, name="N_D"))
    ds.set_parameter(device=device, region=region, name=_EQDG_FLAG, value=float(gamma))
    ds.set_parameter(device=device, region=region, name="eqdg_m_eff_kg", value=float(m_eff_kg))
    ds.set_parameter(device=device, region=region, name=DG_U_FLOOR, value=float(u_floor_rel * math.sqrt(n_bg)))

    # the variables first (a DEVSIM model binds its references when it is created)
    PDG._declare_dg_variables(device, region, "QSqrtN", "QLambda")
    nn = len(ds.get_node_model_values(device=device, region=region, name="x"))
    ds.set_node_values(device=device, region=region, name="QSqrtN", values=[math.sqrt(n_bg)] * nn)
    ds.set_node_values(device=device, region=region, name="QLambda", values=[0.0] * nn)

    # the closure on the quantum-shifted eta; analytic derivatives (DEVSIM diff, unsimplified: exact, fast)
    # the argument is clamped below at ETA_Q_FLOOR (a Newton-transient guard: exp(-eta) of the closure overflows
    # near eta = -709, which DEVSIM traps as FATAL; a converged state sits far above, n/N_c ~ 1e-261 there)
    eta_q = "max({} + QLambda / V_t, {!r})".format(eta, ETA_Q_FLOOR)
    elec = "N_c * " + PE.F12_aymerich_humet_expr(eta_q)
    ds.node_model(device=device, region=region, name="Electrons", equation=elec)
    ds.node_model(device=device, region=region, name="Electrons:Potential",
                  equation="diff({}, Potential)".format(elec))
    ds.node_model(device=device, region=region, name="Electrons:QLambda",
                  equation="diff({}, QLambda)".format(elec))
    ds.node_model(device=device, region=region, name="PotentialNodeCharge:QLambda",
                  equation="ElectronCharge * Electrons:QLambda")

    # the shared DG kernel with the equilibrium tie u^2 - n(psi, Lambda)
    PDG._install_dg_variables(device, region, u="QSqrtN", lam="QLambda", carrier="Electrons", b="b_dg",
                              b_full=b_full, tie=("QSqrtN*QSqrtN - Electrons", ("QSqrtN", "Potential", "QLambda")),
                              declare=False)

    # Lambda is a potential [V]: damp its Newton updates like the electrostatic potential's (log_damp), so a
    # transient cannot swing it by tens of volts in one step (MEASURED: DEVSIM FATAL overflows otherwise on
    # depleted 3 nm films)
    rec = _R.recorded(device, "QLambdaEquation", scope="region", loc=region)
    kw = dict(rec[-1]["kwargs"])
    kw["variable_update"] = "log_damp"
    ds.delete_equation(device=device, region=region, name="QLambdaEquation")
    _R.forget(device, "QLambdaEquation", loc=region, scope="region")
    _R.record_region_equation(device, region, **kw)

    # insulating walls: the node mask and the penalty (module header)
    wall = np.zeros(nn)
    wk = np.zeros(nn)
    for itf in diel:
        w, area = PE.interface_weight(device, region, itf)
        w = np.asarray(w)
        on = np.asarray(area) > 0.0
        wall[on] = 1.0
        wk[on] = np.maximum(wk[on], DG_WALL_PENALTY * b_full * w[on] ** 2)
    for nm, vals in ((DG_WALL, wall), (DG_WALL_K, wk)):
        ds.node_solution(device=device, region=region, name=nm)
        ds.set_node_values(device=device, region=region, name=nm, values=[float(v) for v in vals])
    if np.any(wall > 0.5):
        node = "QLambda*QSqrtN*(1 - {w}) + {w}*{k}*(QSqrtN - {f})".format(w=DG_WALL, k=DG_WALL_K, f=DG_U_FLOOR)
        ds.node_model(device=device, region=region, name="QLambdaNode", equation=node)
        ds.node_model(device=device, region=region, name="QLambdaNode:QLambda",
                      equation="QSqrtN*(1 - {w})".format(w=DG_WALL))
        ds.node_model(device=device, region=region, name="QLambdaNode:QSqrtN",
                      equation="QLambda*(1 - {w}) + {w}*{k}".format(w=DG_WALL, k=DG_WALL_K))
    for c in contacts:
        _install_ohmic_rows(device, c)
    # the validity range is that of the model, not of one gamma: outside it the per-point matching gamma spreads
    # over 1.36 .. 2.40 (the calibration notes), so no constant gamma -- the default or a user's -- is validated
    # there. (It once ran only at the default gamma: a user gamma of 1.7 on the 3 nm / 1e26 m^-3 film, t k_F/pi =
    # 1.37, built silently.)
    check_equilibrium_dg_validity(device, region, stacklevel=3, gamma=float(gamma))


def _install_ohmic_rows(device: str, contact: str) -> None:
    """An ohmic (metal) contact on an equilibrium DG region: Lambda = 0 and the tie u^2 = n(psi, 0) at its
    nodes -- the bulk-reservoir condition of the density-gradient literature (the contact holds the
    classical density). The contact's Potential row is the caller's (physics_equilibrium.setup_contact)."""
    rows = (("qsqrtn_ohmic", "QSqrtNEquation", "QSqrtN*QSqrtN - Electrons", ("QSqrtN", "Potential", "QLambda")),
            ("qlambda_ohmic", "QLambdaEquation", "QLambda", ("QLambda",)))
    for suffix, eq, expr, wrt in rows:
        nm = "{}_{}".format(contact, suffix)
        ds.contact_node_model(device=device, contact=contact, name=nm, equation=expr)
        for w in wrt:
            ds.contact_node_model(device=device, contact=contact, name="{}:{}".format(nm, w),
                                  equation="simplify(diff({}, {}))".format(expr, w))
        _R.record_contact_equation(device, contact, name=eq, node_model=nm)


def equilibrium_dg_length_m(device: str, region: str, n_m3: "float | None" = None) -> float:
    """The Fermi-Dirac-consistent DG length sqrt(b/(g V_t)) of the region at density n_m3 (default n_bg),
    g = F_1/2/F_-1/2 (carriers.einstein.g_einstein): the dead-layer scale of a degenerate film."""
    from dynameta.carriers.einstein import g_einstein
    b = float(ds.get_parameter(device=device, region=region, name="b_dg_full"))
    v_t = float(ds.get_parameter(device=device, region=region, name="V_t"))
    n_c = float(ds.get_parameter(device=device, region=region, name="N_c"))
    n = float(ds.get_parameter(device=device, region=region, name="N_D")) if n_m3 is None else float(n_m3)
    return math.sqrt(b / (float(g_einstein(n / n_c)) * v_t))


def _invert_closure(x: np.ndarray) -> np.ndarray:
    """eta with F_1/2(eta) = x elementwise (the region closure's Aymerich-Humet F_1/2, vectorized bisection on
    the ONE evaluator physics_equilibrium.F12_aymerich_humet)."""
    f = np.vectorize(PE.F12_aymerich_humet)
    lo = np.full(x.shape, -200.0)
    hi = np.full(x.shape, 200.0)
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        below = f(mid) < x
        lo = np.where(below, mid, lo)
        hi = np.where(below, hi, mid)
    return 0.5 * (lo + hi)


def seed_equilibrium_dg(device: str, region: str) -> None:
    """Seed QSqrtN / QLambda of an equilibrium DG region from the CURRENT potential so that the tie holds at
    every node and the wall rows hold at every wall node: n_cl = N_c F_1/2(eta) of the current potential; off
    the walls the frozen-potential boundary-layer shape n = tanh^2(d / L) n_cl (d the distance to the nearest
    wall node, L = equilibrium_dg_length_m at n_bg scaled to the CURRENT b; n = n_cl when b = 0); at a wall
    node n = u_floor^2; u = sqrt(n), Lambda = V_t [F_1/2^-1(n / N_c) - eta] (0 where n = n_cl). Seeding the
    wall nodes even at b = 0 matters: from Lambda = 0 the tie's Newton walks Lambda_wall down by about V_t per
    iteration in the non-degenerate tail (tens of iterations, MEASURED as convergence failures)."""
    v_t = float(ds.get_parameter(device=device, region=region, name="V_t"))
    n_c = float(ds.get_parameter(device=device, region=region, name="N_c"))
    eta_m = "_EqDgSeedEta"
    ds.node_model(device=device, region=region, name=eta_m, equation=PE.equilibrium_eta_expr(device, region))
    eta = np.asarray(ds.get_node_model_values(device=device, region=region, name=eta_m))
    ds.delete_node_model(device=device, region=region, name=eta_m)
    n_cl = n_c * np.vectorize(PE.F12_aymerich_humet)(eta)
    wall = equilibrium_dg_wall_nodes(device, region)
    b = float(ds.get_parameter(device=device, region=region, name="b_dg"))
    n = n_cl.copy()
    if np.any(wall):
        u_floor = float(ds.get_parameter(device=device, region=region, name=DG_U_FLOOR))
        if b > 0.0:
            from scipy.spatial import cKDTree
            nodes = set(ds.get_node_model_list(device=device, region=region))
            xyz = np.stack([np.asarray(ds.get_node_model_values(device=device, region=region, name=a))
                            for a in ("x", "y", "z") if a in nodes], axis=1)
            d = cKDTree(xyz[wall]).query(xyz)[0]
            lq = equilibrium_dg_length_m(device, region) * math.sqrt(b / float(
                ds.get_parameter(device=device, region=region, name="b_dg_full")))
            n = np.maximum(np.tanh(d / lq) ** 2 * n_cl, u_floor ** 2)
        n[wall] = u_floor ** 2
    moved = n != n_cl
    lam = np.zeros_like(n)
    if np.any(moved):
        lam[moved] = v_t * (_invert_closure(n[moved] / n_c) - eta[moved])
    ds.set_node_values(device=device, region=region, name="QSqrtN", values=[float(v) for v in np.sqrt(n)])
    ds.set_node_values(device=device, region=region, name="QLambda", values=[float(v) for v in lam])


def adaptive_continuation(apply, start: float, end: float, solve, *, step: float, max_halvings: int = 8) -> None:
    """Walk a scalar continuation parameter from `start` to `end` (apply(value) sets it, solve() runs the Newton
    and raises on failure, leaving the last converged state in place as DEVSIM does): steps of at most `step`,
    halved on a failure (up to max_halvings times below `step`) and doubled back after two successes. Raises the
    last failure when even the smallest step fails. The DG device's continuation (bias ramps, the b ramp):
    a deeply depleted thin film needs ~0.02 V gate steps near full depletion (MEASURED: 3 nm, 1e26 m^-3, -4 V
    fails in 0.25 V steps and converges in 0.02 V steps), while accumulation takes the full step."""
    span = float(end) - float(start)
    if span == 0.0:
        apply(float(end))
        solve()
        return
    h_max = abs(float(step))
    h_min = h_max / 2.0 ** max_halvings
    pos, h, ok_run = float(start), min(h_max, abs(span)), 0
    sign = 1.0 if span > 0.0 else -1.0
    while sign * (float(end) - pos) > 0.0:
        nxt = pos + sign * min(h, sign * (float(end) - pos))
        if abs(float(end) - nxt) < 1e-12 * max(1.0, abs(span)):
            nxt = float(end)
        apply(nxt)
        try:
            solve()
        except Exception:                                   # noqa: BLE001 (retried with a smaller step)
            apply(pos)
            if h / 2.0 < h_min:
                raise
            h, ok_run = h / 2.0, 0
            continue
        pos, ok_run = nxt, ok_run + 1
        if ok_run >= 2 and h < h_max:
            h, ok_run = min(2.0 * h, h_max), 0


def set_equilibrium_dg_fraction(device: str, region: str, frac: float) -> None:
    """Scale the region's b to `frac` of its full value (the b ramp; physics_density_gradient.set_dg_gamma)."""
    PDG.set_dg_gamma(device, region, frac)


def equilibrium_dg_regime(device: str, region: str) -> dict:
    """The dimensionless numbers the calibration is stated in, for an equilibrium DG region:
      eta_bg   : (E_F - E_c)/kT of the flat-band bulk (the region closure at n_bg);
      t_eff_m  : twice the largest node distance to the nearest insulating wall (the film thickness between
                 two walls; 2 x the thickness for one wall over a reservoir);
      subbands : t_eff k_F / pi with k_F = (3 pi^2 n_bg)^(1/3) -- about the number of occupied sub-bands of the
                 flat-band film."""
    n_c = float(ds.get_parameter(device=device, region=region, name="N_c"))
    n_bg = float(ds.get_parameter(device=device, region=region, name="N_D"))
    eta_bg = PE.invert_F12(n_bg / n_c)
    wall = equilibrium_dg_wall_nodes(device, region)
    t_eff = float("inf")
    if np.any(wall):
        from scipy.spatial import cKDTree
        nodes = set(ds.get_node_model_list(device=device, region=region))
        xyz = np.stack([np.asarray(ds.get_node_model_values(device=device, region=region, name=a))
                        for a in ("x", "y", "z") if a in nodes], axis=1)
        t_eff = 2.0 * float(np.max(cKDTree(xyz[wall]).query(xyz)[0]))
    k_f = (3.0 * math.pi ** 2 * n_bg) ** (1.0 / 3.0)
    return {"eta_bg": float(eta_bg), "t_eff_m": t_eff, "subbands": t_eff * k_f / math.pi}


def check_equilibrium_dg_validity(device: str, region: str, *, stacklevel: int = 2,
                                  gamma: Optional[float] = None) -> dict:
    """Warn (UnreliableResultWarning) when the region is outside the range the density gradient was calibrated
    and measured in against Schrodinger-Poisson (EQUILIBRIUM_DG_GAMMA): flat-band degeneracy eta_bg <
    VALID_ETA_MIN or fewer than VALID_SUBBANDS_MIN occupied sub-bands (equilibrium_dg_regime). The range does not
    depend on gamma (setup_equilibrium_dg runs this for every gamma > 0); gamma only names a non-default value in
    the message. Returns the regime numbers."""
    reg = equilibrium_dg_regime(device, region)
    bad = []
    if reg["eta_bg"] < VALID_ETA_MIN:
        bad.append("flat-band degeneracy (E_F - E_c)/kT = {:.2f} < {:.1f}".format(reg["eta_bg"], VALID_ETA_MIN))
    if reg["subbands"] < VALID_SUBBANDS_MIN:
        bad.append("t k_F/pi = {:.2f} < {:.1f} (t = {:.3g} nm between insulating walls)".format(
            reg["subbands"], VALID_SUBBANDS_MIN, reg["t_eff_m"] * 1e9))
    if bad:
        user = ("" if gamma is None or gamma == EQUILIBRIUM_DG_GAMMA else
                " The given gamma = {:g} is not the default, but outside this range the gamma that matches "
                "Schrodinger-Poisson varies from point to point, so no constant gamma is validated there.".format(
                    gamma))
        warnings.warn(
            "equilibrium density gradient on region {!r}: outside the calibrated range ({}). The default gamma "
            "was measured against Schrodinger-Poisson for degenerate films only (docs/audit/2026-11-14-v0.16-"
            "equilibrium-density-gradient.md); expect larger errors in the sheet charge and the dead layer -- "
            "check against carriers.sp_dual_gate.DualGateSchrodingerPoisson.{}".format(region, "; ".join(bad), user),
            UnreliableResultWarning, stacklevel=stacklevel)
    return reg
