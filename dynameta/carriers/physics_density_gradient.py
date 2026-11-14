"""IN-NEWTON density-gradient drift-diffusion (R19 follow-on): the quantum potential as a
SOLUTION VARIABLE of the coupled DEVSIM Newton, not a post-processor.

Two new solution variables extend the unipolar (Potential, Electrons) system to a 4-variable
Newton:

    u      = QSqrtN   with the algebraic constraint   u^2 - n = 0,
    Lambda = QLambda  with the PDE                    b lap(u) - Lambda u = 0,   b = gamma hbar^2/(6 m q),

and the Scharfetter-Gummel current is rebuilt on the QUANTUM-SHIFTED potential psi + Lambda
(vdiff_dg = (psi@n0 + Lambda@n0 - psi@n1 - Lambda@n1)/V_t), which is the density-gradient model:
the electron gas feels the Bohm/quantum force -grad(Lambda) on top of the electrostatic one.
The Lambda-equation is assembled DEVSIM-natively the same way Poisson is (an edge flux model
b grad(u) + a node model) -- the discrete Laplacian lives in the EQUATION assembly, which is the
piece a node model alone cannot express (node models cannot reference neighbor nodes; the
original R19 recipe foundered exactly there).

Usage: AFTER physics_drift_diffusion.setup_semiconductor_region_dd has built the classical
system (ideally after a CONVERGED classical solve -- the natural initial guess), call
setup_dg_quantum_correction(...) on the region and setup_contact_dg(...) on every contact, seed
with seed_dg_from_solution(...), then gamma-RAMP: solve at b_dg fractions [0.25, 0.5, ...] of
the full value (set_dg_gamma). b_dg = 0 with the DG equations active reduces to the classical
solution (Lambda-eq collapses to Lambda u = 0 -> Lambda = 0, the current to the classical SG).

SCOPE (v1, validated): 1D/2D unipolar regions with ohmic contacts (u pinned to sqrt(n_contact),
Lambda to 0 -- bulk contacts carry no quantum correction). The oxide-interface wall (u -> 0, the
MOS dead layer) is setup_dg_insulator_wall (below; zero normal flux, unipolar and bipolar regions),
seeded with seed_dg_from_solution(..., walls=(contact,)): on a unipolar region it reproduces the
closed-form DG dark space (I = 1.187207 n0 L_q) to 2e-5 (tests/test_density_gradient.py, v0.15
item 4.4.4). The former unipolar setup_dg_hard_wall rows (a pinned carrier row and a pinned
Lambda: a carrier SINK that converged to a spurious wide-depletion branch) are retired: the name is
a deprecated shim onto setup_dg_insulator_wall in v0.15 and is removed in v0.16
(docs/audit/2026-11-06-v0.15-dg-wall-strong-inversion.md).

BIPOLAR (D2d, v0.14.0; end of this file): setup_bipolar_dg gives electrons AND holes their own quantum
potential on a physics_bipolar_dd region (same kernel, _install_dg_variables), with the ohmic rows
setup_contact_bipolar_dg, the zero-flux impenetrable-oxide wall setup_dg_insulator_wall (validated
against the closed-form DG dark space and Schrodinger-Poisson) and seed_bipolar_dg. Doc:
docs/audit/2026-11-03-devsim-bipolar-density-gradient.md.
"""

from __future__ import annotations

import numpy as np

from dynameta.carriers.devsim_env import import_devsim

ds = import_devsim()      # D0: devsim via its ONE import home (BLAS pinned before the first import)

from dynameta.carriers import eq_registry as _R
from dynameta.carriers import sg_flux as _SG
from dynameta.carriers.eq_registry import edge_with_derivs as _edge_with_derivs
from dynameta.carriers.eq_registry import node_with_derivs as _node_with_derivs
from dynameta.carriers.density_gradient import GAMMA_WALL   # the hard-wall gamma (2026-11-04)
from dynameta.carriers.density_gradient import (GAMMA_SI100_SP_CALIBRATED, SI100_ELECTRON_VALLEYS,  # noqa: F401
                                                ValleySet)    # v0.15.1: the valley-resolved electron DG (data)
from dynameta.constants import HBAR, Q_E

__all__ = ["setup_dg_quantum_correction", "setup_contact_dg", "setup_dg_hard_wall",
           "seed_dg_from_solution", "set_dg_gamma", "dg_b_coefficient", "GAMMA_WALL",
           # D2d: bipolar density-gradient
           "setup_bipolar_dg", "setup_contact_bipolar_dg", "setup_dg_insulator_wall",
           "seed_bipolar_dg", "bipolar_dg_carriers", "DG_SRH_TERM",
           # v0.15.1: the valley-resolved electron DG (setup_bipolar_dg(electron_valleys=...))
           "ValleySet", "SI100_ELECTRON_VALLEYS", "GAMMA_SI100_SP_CALIBRATED", "electron_valley_sets",
           "valley_densities"]


def dg_b_coefficient(m_eff_kg: float, gamma: float = 1.0) -> float:
    """b = gamma hbar^2 / (6 m q) [V m^2].

    Choosing gamma (default 1): b, and with it the quantum length L_q = sqrt(b/V_t), scale as gamma
    and sqrt(gamma). gamma = 1 is the Wigner / Ancona value for smooth potentials (bulk problems,
    literature comparisons); GAMMA_WALL (= (3 pi/4)/I^2 = 1.6717) makes the dark space at a hard
    insulator wall equal the exact free-gas one (oxide-bounded accumulation / inversion layers).
    Full guidance and the measured Schrodinger-Poisson comparison: setup_bipolar_dg."""
    if not (m_eff_kg > 0.0 and gamma >= 0.0):
        raise ValueError("density-gradient: m_eff_kg > 0 and gamma >= 0 required")
    return gamma * HBAR ** 2 / (6.0 * m_eff_kg * Q_E)


def _install_dg_variables(device: str, region: str, *, u: str, lam: str, carrier: str,
                          b: str, b_full: float, tie=None, declare: bool = True) -> None:
    """THE density-gradient kernel, shared by the unipolar electron DG and both carriers of the
    bipolar DG (D2d): the parameter pair `b` / `b`_full, the solution variables u = sqrt(carrier)
    and Lambda, the algebraic tie u^2 - c = 0 and the Poisson-style Lambda-equation
    b lap(u) - Lambda u = 0 (edge flux b grad(u) + node term; the node-term SIGN pairs with DEVSIM's
    flux orientation so the converged Lambda equals +b u''/u -- pinned by the independent-stencil
    oracle of validation/dg_dd_in_newton.py GATE B). With (u, lam, carrier, b) = (QSqrtN, QLambda,
    Electrons, b_dg) every DEVSIM call is the pre-D2d unipolar one, in order, character-for-character
    (tests/_dg_calllog.py digest). tie (v0.15.1, the valley-resolved electrons): (expression, wrt) of the
    u-row instead of u^2 - c (the valley sets' sum / intervalley rows, _valley_tie); None = the
    unchanged single-population tie. declare=False: u and lam already exist (the valley sets declare
    all their variables first, because a DEVSIM model binds its references when it is created)."""
    ds.set_parameter(device=device, region=region, name=b, value=b_full)
    ds.set_parameter(device=device, region=region, name=b + "_full", value=b_full)

    if declare:
        _declare_dg_variables(device, region, u, lam)

    # u-equation (variable u): the algebraic tie u^2 = c
    if tie is None:
        _node_with_derivs(device, region, u + "Constraint", "{u}*{u} - {c}".format(u=u, c=carrier),
                          (u, carrier))
    else:
        _node_with_derivs(device, region, u + "Constraint", tie[0], tie[1])
    _R.record_region_equation(device, region, name=u + "Equation", variable_name=u,
                              node_model=u + "Constraint", variable_update="positive")

    # Lambda-equation (variable lam): b lap(u) - Lambda u = 0, assembled Poisson-style
    _edge_with_derivs(device, region, u + "GradFlux",
                      "{b}*({u}@n0 - {u}@n1)*EdgeInverseLength".format(b=b, u=u), (u,))
    _node_with_derivs(device, region, lam + "Node", "{l}*{u}".format(l=lam, u=u), (lam, u))
    _R.record_region_equation(device, region, name=lam + "Equation", variable_name=lam,
                              edge_model=u + "GradFlux", node_model=lam + "Node",
                              variable_update="default")


def _declare_dg_variables(device: str, region: str, u: str, lam: str) -> None:
    """The DG solution variables u and lam (+ their edge copies) of _install_dg_variables."""
    for nm in (u, lam):
        ds.node_solution(device=device, region=region, name=nm)
        ds.edge_from_node_model(device=device, region=region, node_model=nm)


# The quantum-shifted SG fluxes go through the ONE SG kernel (sg_flux.install_sg_current) with
# upper_clamp=True: B(min(v, 200)), a value no-op kept for the log-singular Lambda of a hard wall in
# Newton transients (DEVSIM 2.11.2's B / dBdx are finite at any argument -- sg_flux header). The
# unipolar DG's former TWO-SIDED clamp B(clamp(v, -200, 200)) is gone with its kahan3 flux: under
# a4f00d9's plain two-Bernoulli bracket it froze BOTH Bernoullis once |v| > 200 and zeroed a wall
# node's Jacobian column (MEASURED 2026-11-03 on the bipolar DG: "Convergence failure!" / "umfpack
# numeric failed"); the kernel's sign-selected bracket keeps the linear c v term there anyway.


def setup_dg_quantum_correction(device: str, region: str, *, m_eff_kg: float,
                                gamma: float = 1.0) -> None:
    """Add the (QSqrtN, QLambda) variables + equations and re-point the electron continuity at
    the quantum-shifted Scharfetter-Gummel current (module header). Requires the classical
    unipolar setup on the region first. (Bipolar regions: setup_bipolar_dg.)

    Choosing gamma (the default stays gamma = 1; user decision 2026-11-04):
      gamma scales b = gamma hbar^2/(6 m q) -- the quantum length L_q = hbar sqrt(gamma/(6 m kT)) over
      which electrons are pushed off a wall (dark-space width ~ sqrt(gamma)); gamma = 0 with the DG
      equations active reduces to the classical solution.
      * gamma = 1 -- the Wigner / Ancona high-temperature value for SMOOTH potentials. Use it for
        bulk / smooth-potential problems (the ohmic-contact bars of validation/dg_dd_in_newton.py)
        and to compare with the literature.
      * GAMMA_WALL (= 1.6717) -- matches the exact free-gas dark space at a hard insulator wall
        (gamma = 1 gives 0.7734 of it). Use it for an accumulation / inversion layer against a gate
        oxide. The Schrodinger-Poisson comparison that supports it (setup_bipolar_dg, GATE C: excess
        sheet +5.6 % vs +21.3 %, centroid -5.1 % vs -17.5 % for electrons) was MEASURED with the
        bipolar DG and setup_dg_insulator_wall. This unipolar kernel is the same
        (_install_dg_variables) and, with setup_dg_insulator_wall, gives the same solution as the
        bipolar DG on the same bar (gated, tests/test_density_gradient.py), but GATE C itself was
        not rerun on a unipolar region. For a wall use setup_dg_insulator_wall (the old
        setup_dg_hard_wall is a deprecated shim onto it).
      * Not validated: strong inversion, degenerate Fermi-Dirac beyond GATE C's regime, 2-D/3-D
        walls. Details: setup_bipolar_dg."""
    b_full = dg_b_coefficient(m_eff_kg, gamma)
    _install_dg_variables(device, region, u="QSqrtN", lam="QLambda", carrier="Electrons",
                          b="b_dg", b_full=b_full)

    # quantum-shifted SG current: psi_eff = psi + Lambda
    _edge_with_derivs(device, region, "vdiff_dg",
                      "(Potential@n0 + QLambda@n0 - Potential@n1 - QLambda@n1)/V_t",
                      ("Potential", "QLambda"))
    # the region's FD transport: with the consistent quasi-field (statistics default since v0.15.0) the
    # degeneracy is the Delta1 - Delta0 term of the classical argument, carried over here; the legacy averaged g
    # is the division (unchanged string)
    fdq = ""
    if "HJ_Electrons_dDelta" in ds.get_edge_model_list(device=device, region=region):
        from dynameta.carriers.physics_heterojunction import fd_quasi_field_expr
        fdq = " + " + fd_quasi_field_expr(device, region, "Electrons", "N_c")
    _edge_with_derivs(device, region, "vdiff_g_dg", "vdiff_dg / g_enh" + fdq,
                      ("Potential", "QLambda", "Electrons"))
    _SG.install_sg_current(device, region, carrier="electrons", density="Electrons",
                           vdiff="vdiff_g_dg", bern="Bern_g_dg", bern_m="Bern_g_dg_m",
                           current="ElectronCurrentDG", prefactor=_PREF["electrons"],
                           wrt=("Potential", "QLambda", "Electrons"),
                           current_wrt=("Electrons", "Potential", "QLambda"), upper_clamp=True)
    ds.delete_equation(device=device, region=region, name="ElectronContinuityEquation")
    _R.forget(device, "ElectronContinuityEquation", loc=region)
    _R.record_region_equation(device, region, name="ElectronContinuityEquation",
                              variable_name="Electrons", edge_model="ElectronCurrentDG",
                              time_node_model="NCharge", variable_update="positive")


# The quantum-shifted Scharfetter-Gummel fluxes: the classical prefactors (physics_bipolar_dd) on the
# DG Bernoulli argument, through the ONE SG kernel (sg_flux) --
#     Jn ~ n@n1 B(-v) - n@n0 B(v) ,   Jp ~ -( p@n1 B(v) - p@n0 B(-v) ) ,
# both Bernoullis evaluated, combined sign-selected (sg_flux header).
# HISTORY. The bipolar DG switched first (a4f00d9, 2026-11-03, to the plain bracket above): the
# kahan3 spelling B(-v) = B(v) + v cancels for v << 0 (relative error ~ eps |v| e^|v|: 1.9e-7 at
# v = -21, 6.5e-5 at -27.6, 100 % at -40), and an insulator-wall edge sits at |v| = ln(c@n1/c@n0) ~ 20-28 BY CONSTRUCTION (the wall
# node holds c = u_floor^2 = 1e-12 c_cl); GATE C (validation/bipolar_density_gradient.py) then sat in
# a period-3 Newton cycle at relative update 1.9-3.8e-7 against REL_ERR 1e-7 and passed or failed
# on the BLAS's round-off (docs/audit/2026-11-03-devsim-bipolar-density-gradient.md sec. 1.6). The
# unipolar DG and every classical flux followed on 2026-11-04 (user decision), all through sg_flux,
# whose sign-selected bracket replaced the plain one too (same large-|v| accuracy, 4x less ssac
# round-off near equilibrium; docs/audit/2026-11-04-sg-flux-two-bernoulli.md).
_PREF = {"electrons": "ElectronCharge*mu_n*EdgeInverseLength*V_t*g_enh",
         "holes": "-ElectronCharge*mu_p*EdgeInverseLength*V_t*g_enh_p"}


def setup_contact_dg(device: str, contact: str, n_contact_m3: float) -> None:
    """Bulk-ohmic DG contact: u = sqrt(n_contact), Lambda = 0 (no quantum correction in the
    contact reservoir).

    TERMINAL CURRENT (audit S1-5, 2026-07-17; fixed wave-7 2026-11-02). setup_dg_quantum_correction
    re-points the REGION continuity at ElectronCurrentDG, but the ohmic CONTACT equation that
    physics_drift_diffusion.setup_contact_ohmic_dd recorded still carried
    edge_current_model='ElectronCurrent' -- the classical SG flux on psi alone. DEVSIM integrates the
    contact's edge_current_model to report get_contact_current, so a DG device's terminal current
    was read from the pre-quantum flux on the contact edges while the solve itself (node_model pin,
    unchanged) was right. The two differ by the q mu n grad(Lambda) Bohm term on the contact edge:
    measured on a 40 nm bar with a 4e26 -> 4e25 m^-3 step 3 nm from the right contact at 10 mV
    (max |Lambda| 10 mV), the two terminal currents failed to conserve by |I_L + I_R|/|I_L| =
    2.95e-2 before (the right one read the classical flux; 6.0e-3 with the step 6 nm away) and
    4.4e-16 after, both now equal to the interior DG edge current to 2e-16. The contact's ElectronContinuityEquation is now
    re-bound here with edge_current_model='ElectronCurrentDG' (same node_model, so the pinned
    boundary condition -- and every converged solution -- is unchanged). Requires
    setup_dg_quantum_correction on the contact's region FIRST (the documented order); raises if
    ElectronCurrentDG does not exist there yet."""
    if not (n_contact_m3 > 0.0):
        raise ValueError("setup_contact_dg: n_contact_m3 must be > 0")
    _repoint_contact_current_to_dg(device, contact)
    cu = "{}_qsqrtn_dirichlet".format(contact)
    ds.contact_node_model(device=device, contact=contact, name=cu,
                          equation="QSqrtN - {:.16e}".format(float(np.sqrt(n_contact_m3))))
    ds.contact_node_model(device=device, contact=contact, name="{}:QSqrtN".format(cu),
                          equation="1")
    _R.record_contact_equation(device, contact, name="QSqrtNEquation", node_model=cu)
    cl = "{}_qlambda_dirichlet".format(contact)
    ds.contact_node_model(device=device, contact=contact, name=cl, equation="QLambda")
    ds.contact_node_model(device=device, contact=contact, name="{}:QLambda".format(cl),
                          equation="1")
    _R.record_contact_equation(device, contact, name="QLambdaEquation", node_model=cl)


def _repoint_contact_current_to_dg(device: str, contact: str, *,
                                   eq: str = "ElectronContinuityEquation",
                                   cur: str = "ElectronCurrent",
                                   cur_dg: str = "ElectronCurrentDG",
                                   who: str = "setup_contact_dg",
                                   installer: str = "setup_dg_quantum_correction") -> None:
    """audit S1-5: re-bind an ohmic contact's continuity equation `eq` so its edge_current_model is
    the DG current `cur_dg` instead of the classical `cur` (see setup_contact_dg). A contact with no
    recorded classical current (e.g. a bare potential contact) is left untouched. Every other kwarg
    of the recorded contact equation (node_model, circuit_node, ...) is kept, so the boundary
    condition itself is unchanged. The defaults are the unipolar electron call; the bipolar DG
    (setup_contact_bipolar_dg, D2d) passes the hole names too."""
    entries = [e for e in _R.recorded(device, eq, scope="contact", loc=contact)
               if e["kwargs"].get("edge_current_model") == cur]
    if not entries:
        return
    region = ds.get_region_list(device=device, contact=contact)[0]
    if cur_dg not in ds.get_edge_model_list(device=device, region=region):
        raise ValueError(
            "{}('{}'): region '{}' has no {} -- call {} on it BEFORE {}, so the contact's terminal "
            "current is read from the density-gradient flux (audit S1-5)".format(
                who, contact, region, cur_dg, installer, who))
    kw = dict(entries[-1]["kwargs"])
    kw["edge_current_model"] = cur_dg
    ds.delete_contact_equation(device=device, contact=contact, name=eq)
    _R.forget(device, eq, loc=contact, scope="contact")
    _R.record_contact_equation(device, contact, **kw)


# v0.15 (plan 4.4.4, decision 53): what the deprecation warning states -- the MEASURED change for a
# device that used the old rows (docs/audit/2026-11-06-v0.15-dg-wall-strong-inversion.md, sec. 1).
HARD_WALL_DEPRECATION = (
    "setup_dg_hard_wall is deprecated and will be removed in v0.16: it now installs "
    "setup_dg_insulator_wall(device, contact), the zero-flux wall (lambda_pin_factor is ignored). "
    "Results of a device built with it CHANGE: the old rows pinned the carrier row (n = u^2) and "
    "Lambda (-lambda_pin_factor V_t), so the wall node was a carrier sink (measured wall-edge current "
    "6e-3 to 1e-1 of q mu n0 V_t/L_q) and the solve converged to a spurious wide-depletion branch "
    "(dead-layer deficit 23-28 n0 L_q); the insulator wall conserves carriers (wall current <= 1e-14 "
    "of that scale) and reproduces the closed-form DG dark space, deficit 1.187207 n0 L_q, to 2e-5. "
    "Call setup_dg_insulator_wall directly and seed with seed_dg_from_solution(device, region, "
    "walls=(contact,)).")


def setup_dg_hard_wall(device: str, contact: str, *, lambda_pin_factor: float = 15.0) -> None:
    """DEPRECATED (v0.15; removed in v0.16): a shim that installs setup_dg_insulator_wall(device,
    contact) and issues a DeprecationWarning (HARD_WALL_DEPRECATION) stating the measured change.
    Use setup_dg_insulator_wall directly, and seed_dg_from_solution(device, region, walls=(contact,))
    for the wall-aware seed.

    `lambda_pin_factor` is still VALIDATED (> 0, ValueError otherwise, before any warning) so an old
    script with a bad value still fails loudly, but it is otherwise IGNORED: the insulator wall pins
    no Lambda (its Lambda is whatever makes the wall-edge flux vanish) and creates no
    wall_lambda_pin parameter.

    What the old rows were, and why they were retired (plan 4.4.4, decision 53): u = 1e-6 sqrt(N_D),
    Lambda = -lambda_pin_factor V_t (a ramped device parameter) and the CARRIER row n = u^2 at the
    wall node. The pinned carrier row made the wall node a carrier SINK whose SG current was driven
    by the pinned Lambda jump, and the converged profile was a spurious wide-depletion branch.
    MEASURED on this module's unipolar bar (frozen flat psi, gamma ramp to 1; tests/
    test_density_gradient.py, docs/audit/2026-11-06-v0.15-dg-wall-strong-inversion.md):
        n0 = 1e21 m^-3 : old deficit 22.85 n0 L_q, wall current 6.3e-3 q mu n0 V_t/L_q;
                         insulator wall 1.18723 (closed form 1.187207, +1.7e-5), current 3e-15
        n0 = 4e26 (ITO): old deficit 28.0 n0 L_q, current 9.6e-2; insulator wall 0.346 (the FD
                         g-factor shortens the dark space ~sqrt(g); gated against the FD first
                         integral)
    The old rows survive only as a test fixture (tests/_dg_legacy_wall.py) for the cross-version
    flux gate and the negative controls."""
    import warnings
    if not (lambda_pin_factor > 0.0):
        raise ValueError("setup_dg_hard_wall: lambda_pin_factor must be > 0")
    warnings.warn(HARD_WALL_DEPRECATION, DeprecationWarning, stacklevel=2)
    setup_dg_insulator_wall(device, contact)


def seed_dg_from_solution(device: str, region: str, *, walls=(),
                          u_floor_rel: float = 1.0e-6) -> None:
    """Seed the unipolar DG variables from the CURRENT (converged classical) solution -- call after
    it, before the gamma ramp. The ONE seed kernel (_seed_dg, shared with seed_bipolar_dg):
    QSqrtN = sqrt(Electrons), QLambda = 0; and, next to each insulator-wall contact in `walls`
    (setup_dg_insulator_wall), the wall boundary-layer shape t(d) = max(tanh(d/L_q), u_floor_rel):
    u = t sqrt(n), n = t^2 n, Lambda = V_t ln(t^2) (seed_bipolar_dg's docstring).

    Which to use: walls=() (default) for a wall-free device (ohmic bars, junctions) -- exactly the
    pre-v0.15 seed (same node values, gated); walls=(contact,) whenever the region has an insulator
    wall: the flat seed against the wall converges too, but slower (measured on the 4e26 ITO bar's
    first gamma step: 27 Newton iterations flat vs 4-23 seeded, both to the same solution)."""
    _seed_dg(device, region, walls=walls, u_floor_rel=u_floor_rel, who="seed_dg_from_solution")


def set_dg_gamma(device: str, region: str, frac: float) -> None:
    """Scale b_dg (and a bipolar region's b_dg_p) to `frac` of its full value (the gamma-ramp
    convergence aid; frac in [0,1]). LEGACY (removed with the setup_dg_hard_wall shim in v0.16): a
    device that still carries the old hard wall's `wall_lambda_pin_full` device parameter (a script
    that built the old rows itself) gets that pin co-ramped with the same fraction; no library code
    creates the parameter any more, so for every wall this is a no-op."""
    if not (0.0 <= frac <= 1.0):
        raise ValueError("set_dg_gamma: frac must be in [0, 1]")
    # D2d: a bipolar DG region (setup_bipolar_dg) may carry the electron b_dg, the hole b_dg_p, or
    # both; the hole twin co-ramps with the SAME fraction. A unipolar region always has b_dg_full and
    # never b_dg_p_full, so its DEVSIM call sequence is the pre-D2d one (tests/_dg_calllog.py digest).
    rparams = ds.get_parameter_list(device=device, region=region)
    if _VDG_SETS in rparams:                # v0.15.1: valley-resolved electrons, every set with the same fraction
        for k in range(electron_valley_sets(device, region)):
            b_full = float(ds.get_parameter(device=device, region=region, name=_vk("b_dg", k) + "_full"))
            ds.set_parameter(device=device, region=region, name=_vk("b_dg", k), value=frac * b_full)
    elif "b_dg_full" in rparams or "b_dg_p_full" not in rparams:
        b_full = float(ds.get_parameter(device=device, region=region, name="b_dg_full"))
        ds.set_parameter(device=device, region=region, name="b_dg", value=frac * b_full)
    if "b_dg_p_full" in rparams:
        bp_full = float(ds.get_parameter(device=device, region=region, name="b_dg_p_full"))
        ds.set_parameter(device=device, region=region, name="b_dg_p", value=frac * bp_full)
    # audit C-17: ask whether the hard wall exists instead of reading ANY DEVSIM error as "no hard
    # wall" -- the bare `except Exception` would also swallow a real failure (bad device, a non-numeric
    # pin) and leave the Lambda pin stale at its previous fraction while b_dg moved on.
    if "wall_lambda_pin_full" not in ds.get_parameter_list(device=device):
        return                                          # no hard wall on this device
    pin_full = float(ds.get_parameter(device=device, name="wall_lambda_pin_full"))
    ds.set_parameter(device=device, name="wall_lambda_pin", value=frac * pin_full)


# =================================================================================================
# D2d (v0.14.0): BIPOLAR density-gradient -- electrons AND holes each carry a quantum potential.
# Formulation, citations and the measured oracles: docs/audit/2026-11-03-devsim-bipolar-density-
# gradient.md and validation/bipolar_density_gradient.py. In short (potentials in volts):
#
#   Lambda_n = b_n lap(sqrt n)/sqrt n ,  b_n = gamma_n hbar^2/(6 m_n q)     (electrons; as above)
#   Lambda_p = b_p lap(sqrt p)/sqrt p ,  b_p = gamma_p hbar^2/(6 m_p q)     (holes: own mass/gamma)
#
# (Ancona & Tiersten, PRB 35, 7959 (1987); Ancona & Iafrate, PRB 39, 9536 (1989); Wettstein, Schenk
# & Fichtner, IEEE TED 48, 279 (2001): one such potential per carrier type, each with its own
# effective mass and fit factor.) Both act as BAND-EDGE SHIFTS:
#
#   E_c,eff = E_c - q Lambda_n ,   E_v,eff = E_v + q Lambda_p
#   n = n_i exp((psi + Lambda_n - phi_n)/V_t) ,   p = n_i exp((phi_p - psi + Lambda_p)/V_t)
#
# i.e. electrons drift in psi + Lambda_n and holes in psi - Lambda_p -- EXACTLY the Lundstrom-
# Schuelke quasi-field slot the bandgap-narrowing D_c / D_v already use (bipolar_options), so the
# DG fluxes are built ON TOP of whatever vdiff_g / vdiff_gp the region carries (BGN included):
#   vdiff_g_dg  = vdiff_g  + (Lambda_n@n0 - Lambda_n@n1)/(V_t g_enh)
#   vdiff_gp_dg = vdiff_gp - (Lambda_p@n0 - Lambda_p@n1)/(V_t g_enh_p)
# Near a wall lap(sqrt c)/sqrt c < 0, so Lambda_n, Lambda_p < 0 and BOTH carriers are repelled.
# Equilibrium: n p = n_i^2 exp((Lambda_n + Lambda_p)/V_t) -- n_i^2 only where Lambda -> 0.
# =================================================================================================

_BIPOLAR_DG = {
    "electrons": dict(u="QSqrtN", lam="QLambda", c="Electrons", b="b_dg",
                      eq="ElectronContinuityEquation", cur="ElectronCurrent",
                      cur_dg="ElectronCurrentDG", vd="vdiff_g_dg", bern="Bern_g_dg",
                      bern_m="Bern_g_dg_m",
                      vd_expr="vdiff_g + (QLambda@n0 - QLambda@n1)/(V_t*g_enh)"),
    "holes": dict(u="QSqrtP", lam="QLambdaP", c="Holes", b="b_dg_p",
                  eq="HoleContinuityEquation", cur="HoleCurrent",
                  cur_dg="HoleCurrentDG", vd="vdiff_gp_dg", bern="Bern_gp_dg",
                  bern_m="Bern_gp_dg_m",
                  vd_expr="vdiff_gp - (QLambdaP@n0 - QLambdaP@n1)/(V_t*g_enh_p)"),
}
DG_SRH_TERM = "dgsrh"            # RTerm_dgsrh: the quantum detailed-balance correction to USRH


def bipolar_dg_carriers(device: str, region: str) -> tuple:
    """The carriers ('electrons', 'holes') that carry a density-gradient quantum potential on a
    BIPOLAR region (read back from DEVSIM: the region has the carrier's u variable and its DG
    current). () for a classical region."""
    nodes = set(ds.get_node_model_list(device=device, region=region))
    edges = set(ds.get_edge_model_list(device=device, region=region))
    return tuple(k for k, sp in _BIPOLAR_DG.items()
                 if (sp["u"] in nodes or (k == "electrons" and "QSqrtN_v0" in nodes))   # v0.15.1: valley sets
                 and sp["cur_dg"] in edges)


def _repoint_region_current(device: str, region: str, eq: str, cur_dg: str) -> None:
    """Re-bind a recorded REGION continuity equation to the DG flux, keeping every other kwarg the
    bipolar setup (and the additive term mechanism: node_model / edge_volume_model / element
    slots) recorded -- the generation composites, the time term and any BTBT slot survive."""
    rec = _R.recorded(device, eq, scope="region", loc=region)
    if len(rec) != 1:
        raise ValueError("setup_bipolar_dg: region {!r} has no recorded {} -- a bipolar drift-"
                         "diffusion region (physics_bipolar_dd.setup_bipolar_region) is required"
                         .format(region, eq))
    if rec[0]["kwargs"].get("node_model") in ("QE_n", "QE_p"):
        # v0.15 (plan 4.4.3): re-pointing would bolt the DG flux onto the algebraic closure row
        raise ValueError("setup_bipolar_dg: {} on region {!r} is closed by physics_bipolar_dd."
                         "setup_quasi_equilibrium_carrier -- call setup_bipolar_dg BEFORE the closure "
                         "(it then includes the carrier's Lambda)".format(eq, region))
    kw = dict(rec[0]["kwargs"])
    kw["edge_model"] = cur_dg
    ds.delete_equation(device=device, region=region, name=eq)
    _R.forget(device, eq, loc=region, scope="region")
    _R.record_region_equation(device, region, **kw)


def setup_bipolar_dg(device: str, region: str, *, m_eff_n_kg: "float | None" = None,
                     m_eff_p_kg: "float | None" = None, gamma_n: float = 1.0,
                     gamma_p: float = 1.0, quantum_srh: bool = True,
                     electron_valleys=None) -> tuple:
    """Opt-in BIPOLAR density-gradient on a region built by physics_bipolar_dd.setup_bipolar_region
    (call AFTER it, with any BipolarOptions / traps / BTBT already installed, and ideally after a
    converged classical solve -- the seed). A carrier gets a quantum potential iff its mass is given:
    m_eff_n_kg -> (QSqrtN, QLambda, b_dg), m_eff_p_kg -> (QSqrtP, QLambdaP, b_dg_p); the other
    carrier stays classical. b = gamma hbar^2/(6 m q) per carrier (dg_b_coefficient).

    Choosing gamma (gamma_n, gamma_p; the default stays 1 -- user decision 2026-11-04):
      What it does: gamma scales b = gamma hbar^2/(6 m q), i.e. the quantum length
      L_q = hbar sqrt(gamma/(6 m kT)) over which a carrier is pushed off a wall; the dark-space width
      and the missing sheet charge grow as sqrt(gamma). gamma = 0 is classical drift-diffusion.
      * gamma = 1 (default) -- the Wigner / Ancona high-temperature value, derived for SMOOTH
        potentials. Use it for bulk / smooth-potential problems (junctions, graded doping, no hard
        wall) and whenever you compare with the literature, which uses 1. At a hard wall it
        under-sizes the dark space: its deficit is 0.7734 of the exact quantum free gas (GATE B).
      * GAMMA_WALL = (3 pi/4)/I^2 = 1.6717 (density_gradient.GAMMA_WALL, re-exported here) -- the
        gamma whose DG dark space at an impenetrable wall equals the EXACT free-gas one. It is fixed a
        priori by that limit, not fitted. Use it for accumulation / inversion layers against a gate
        oxide (setup_dg_insulator_wall). MEASURED against self-consistent Schrodinger-Poisson (GATE C,
        N = 1e24 m^-3, |psi_s - psi_b| = 0.15 V accumulation, FD-consistent DD; electrons / holes):
            excess sheet density   gamma = 1: +21.3 % / +17.3 %    GAMMA_WALL: +5.6 % / +4.0 %
            sheet centroid         gamma = 1: -17.5 % / -14.7 %    GAMMA_WALL: -5.1 % / -3.7 %
            peak depth (x SP)      gamma = 1: 0.67 / 0.68          GAMMA_WALL: 0.84 / 0.82
            (classical DD: sheet +177 % / +125 %, centroid -64 % / -56 %, peak AT the wall)
        and at N = 1e23, 0.075 V GAMMA_WALL is within 1.5 % for both carriers.
      * STRONG INVERSION (v0.15, plan 4.4.3): the full coupled Newton converges only up to weak
        inversion in 1-D (0.60 V on 3e23 m^-3 p-type; 0.70 V fails: a near-null minority mode). For a
        MOS capacitor close the carriers with physics_bipolar_dd.setup_quasi_equilibrium_carrier
        (AFTER this call; both carriers when both carry DG) -- it then reaches 1.3 V. Against the
        Schrodinger-Poisson inversion oracle (validation/bipolar_density_gradient.py GATE G) GAMMA_WALL
        is the better gamma there too; the measured fidelity is in that gate's output and the audit
        note docs/audit/2026-11-06-v0.15-dg-wall-strong-inversion.md. Not validated: DEGENERATE Fermi-Dirac beyond
        GATE C's regime (with fd_enhancement gamma also absorbs the degeneracy calibration; GATE C
        peaks at c/N_c = 0.37-0.59 and lands within 5.6 %, a regime result, not a derivation);
        GAMMA_WALL in a wall-free smooth-potential problem; 2-D / 3-D (only 1-D oracles).
      Each carrier takes its own gamma; for one gate oxide pass GAMMA_WALL to both.
      Doc: docs/audit/2026-11-03-devsim-bipolar-density-gradient.md (GATE B, GATE C, Limits).

    electron_valleys (v0.15.1, opt-in; default None = the isotropic electron DG above, unchanged):
      a table of density_gradient.ValleySet (e.g. SI100_ELECTRON_VALLEYS) -> VALLEY-RESOLVED electrons: every set
      gets its own quantum potential with its own quantization mass and share of N_c, all sets share one quasi-
      Fermi level, the electron current is the single continuity row on the total density (module block
      "VALLEY-RESOLVED"). Pass it INSTEAD of m_eff_n_kg (both: ValueError); gamma_n applies to every set. A
      one-set table IS the isotropic model (same rows, bit-identical solve). Needs statistics 'fermi_dirac' or
      'boltzmann' ('legacy' is refused). Read back: electron_valley_sets / valley_densities; QLambda is then the
      effective Lambda_eff (a model), the per-set variables are QSqrtN_v<k> / QLambda_v<k>.
      WHY: a real Si(100) inversion layer moves into the two valleys with the heavy confinement mass m_l; one
      isotropic quantum potential cannot represent that repopulation. MEASURED against the DOS-consistent
      anisotropic Schrodinger-Poisson oracle (p-Si 1e23 .. 1e24 m^-3, 1.00 .. 1.30 V, FD, GAMMA_WALL, closure;
      validation/valley_density_gradient.py GATE VA): N_inv +2.9 .. +7.9 % for N_inv >= 1e16 m^-2 (up to +14.5 %
      near threshold), centroid -1.2 .. -5.2 %, 2-fold share within 0.02 .. 0.08 of the SP's; the isotropic model
      against the same oracle: -8.7 .. -39 %. Cost: +2 unknowns per node per extra set; MEASURED per Newton
      iteration 1.9x the isotropic DG's in 1-D with electron DG only, but 0.1-0.8x in 1-D with the hole DG and in 2-D
      (there the isotropic system factorizes slowly; audit note sec. 6); Newton iterations equal or fewer.
      gamma_n = GAMMA_SI100_SP_CALIBRATED (= 1.88, a FIT to that oracle; its comment in density_gradient states
      fit range and residuals: N_inv -1.3 .. +0.9 %, centroid -2.8 .. +1.5 %) is the opt-in calibration; the
      default stays gamma = 1 and GAMMA_WALL stays the a-priori wall value.
      Limits (not measured): other orientations (other tables), holes (single population), valley-dependent
      mobility (one mu_n), oxide penetration, temperatures other than 300 K, 2-D/3-D confinement.
      Doc: docs/audit/2026-11-10-v0.15.1-valley-resolved-dg.md.

    Each carrier gets the shared kernel (_install_dg_variables) and its continuity equation is
    re-pointed at the quantum-shifted SG flux (ElectronCurrentDG / HoleCurrentDG) built on the
    region's own quasi-field (vdiff_g / vdiff_gp: FD g-factor, BGN band edges and all -- header
    block above). The generation composites, time terms and BTBT slots of the equation are kept.

    quantum_srh (default True; kept by user decision 2026-11-04):
      What it does: the built-in SRH channel's reference follows the quantum band edges. DG
      equilibrium is n p = n_i^2 exp((Lambda_n + Lambda_p)/V_t), not n_i^2, so the SRH kernel is
      re-referenced n_i'^2 -> n_i'^2 exp((Lambda_n + Lambda_p)/V_t) (n1', p1' by the square root;
      physics_traps.srh_reference_densities(scale=...)), entered through the ONE additive mechanism as
      RTerm_dgsrh = U_SRH(quantum) - USRH (+ its pair rate, so the noise inventory sees it). U_SRH
      then vanishes in equilibrium everywhere: DG equilibrium is detailed-balance-consistent.
      MEASURED (GATE E, inverted MOS, p-type 3e23 m^-3, psi_s - psi_b = 0.60 V, Lambda = -27.6 V_t at
      the wall): max |phi_n| = 3.3e-16 V, max |phi_p| = 5.6e-17 V with it. With quantum_srh=False the
      classical reference generates pairs in the dark space in equilibrium and the quasi-Fermi levels
      are no longer flat: max |phi_n| = 6 mV off the wall node, max |phi_p| = 62 V at the wall node
      (the doc's GATE E table; the classical reference is meaningless there).
      When to turn it off: only to reproduce a code / paper that keeps the classical n_i^2 reference
      under DG (a like-for-like comparison); never for physics.
      Only the BUILT-IN channel is corrected: Auger / radiative / trap levels / interface traps /
      BTBT keep the classical n_i reference (not in detailed balance inside the dark space; not
      gated -- doc: Limits).

    Returns the tuple of DG carriers installed."""
    valleys = None
    if electron_valleys is not None:
        from dynameta.carriers.density_gradient import check_valley_table
        valleys = check_valley_table(electron_valleys)
        if m_eff_n_kg is not None:
            raise ValueError("setup_bipolar_dg: give m_eff_n_kg OR electron_valleys (the sets carry their own "
                             "quantization masses), not both")
        if len(valleys) == 1:            # one set of share 1 IS the isotropic model: its rows, call for call
            m_eff_n_kg, valleys = valleys[0].m_z_kg, None
    if m_eff_n_kg is None and m_eff_p_kg is None and valleys is None:
        raise ValueError("setup_bipolar_dg: give m_eff_n_kg and/or m_eff_p_kg (the carriers that get "
                         "a quantum potential)")
    if "Holes" not in ds.get_node_model_list(device=device, region=region):
        raise ValueError("setup_bipolar_dg: region {!r} is not a bipolar region (no Holes); use "
                         "setup_dg_quantum_correction for a unipolar one".format(region))
    for key, m in (("electrons", m_eff_n_kg if valleys is None else 1.0), ("holes", m_eff_p_kg)):
        if m is not None:                    # refuse BEFORE installing anything (v0.15 plan 4.4.3)
            rec = _R.recorded(device, _BIPOLAR_DG[key]["eq"], scope="region", loc=region)
            if rec and rec[-1]["kwargs"].get("node_model") in ("QE_n", "QE_p"):
                _repoint_region_current(device, region, _BIPOLAR_DG[key]["eq"],
                                        _BIPOLAR_DG[key]["cur_dg"])      # raises the closure message
    if valleys is not None:
        _valley_fd(device, region)           # refuses statistics='legacy' before installing anything
    done = []
    for key, m, g in (("electrons", m_eff_n_kg, gamma_n), ("holes", m_eff_p_kg, gamma_p)):
        sp = _BIPOLAR_DG[key]
        if key == "electrons" and valleys is not None:
            # v0.15.1: valley-resolved electrons -- the same flux on Lambda_eff (the QLambda model)
            wrt = _install_valley_electrons(device, region, valleys, g)
        elif m is None:
            continue
        else:
            _install_dg_variables(device, region, u=sp["u"], lam=sp["lam"], carrier=sp["c"],
                                  b=sp["b"], b_full=dg_b_coefficient(m, g))
            wrt = ("Potential", sp["lam"], sp["c"])
        _edge_with_derivs(device, region, sp["vd"], sp["vd_expr"], wrt)
        _SG.install_sg_current(device, region, carrier=key, density=sp["c"], vdiff=sp["vd"],
                               bern=sp["bern"], bern_m=sp["bern_m"], current=sp["cur_dg"],
                               prefactor=_PREF[key], wrt=wrt, upper_clamp=True)
        _repoint_region_current(device, region, sp["eq"], sp["cur_dg"])
        done.append(key)
    if quantum_srh:
        _install_quantum_srh(device, region, tuple(done))
    return tuple(done)


def _install_quantum_srh(device: str, region: str, carriers: tuple) -> None:
    """RTerm_dgsrh (setup_bipolar_dg, quantum_srh=True): the built-in SRH channel re-referenced to
    the quantum band edges, as the additive correction U_SRH(n_i' s) - USRH with
    s = DgNieFactor = exp((Lambda_n + Lambda_p)/(2 V_t)). The exponent is clamped at +-200 (Newton
    transients only; it never binds at convergence). This is an exp() guard, not the Bernoulli one:
    since the SG-kernel merge (2026-11-04) the DG fluxes clamp the Bernoulli argument above only. The term's Lambda
    derivatives (which the additive mechanism, written for Electrons/Holes, does not create) are
    added here and wired into the generation composites' Jacobian -- RTerm_dgsrh is the only
    Lambda-dependent term, so those composite derivatives stay valid whatever terms are added later."""
    from dynameta.carriers import physics_bipolar_dd as _BP
    from dynameta.carriers.physics_traps import srh_rate_expr
    lams = [_BIPOLAR_DG[k]["lam"] for k in carriers]
    arg = "(0.5*({})/V_t)".format(" + ".join(lams))
    # v0.15.1: valley-resolved electrons -- QLambda is the Lambda_eff MODEL (the total density's band-edge
    # shift, so DG equilibrium is still n p = n_i'^2 exp((Lambda_eff + Lambda_p)/V_t)); the derivatives go to
    # its variables (every u_k and Lambda_0) through the chain rule
    wrt = tuple(lams)
    if "electrons" in carriers and electron_valley_sets(device, region):
        wrt = _eff_wrt(electron_valley_sets(device, region)) + tuple(x for x in lams if x != "QLambda")
        lams = list(wrt)
    _node_with_derivs(device, region, "DgNieFactor",
                      "exp(ifelse({a} > 200, 200, ifelse({a} < -200, -200, {a})))".format(a=arg),
                      wrt)
    bgn = "n_ie" in ds.get_node_model_list(device=device, region=region)
    fkw = {"fd": True} if "FDGammaN" in ds.get_node_model_list(device=device, region=region) else {}   # USRH's
    rate = "({}) - USRH".format(srh_rate_expr(bgn=bgn, scale="DgNieFactor", **fkw))
    pair = "({}) - ({})".format(srh_rate_expr(bgn=bgn, scale="DgNieFactor", pair=True, **fkw),
                                srh_rate_expr(bgn=bgn, pair=True, **fkw))
    model = _BP.add_recombination_term(device, region, DG_SRH_TERM, rate, pair=pair)
    for lam in lams:
        ds.node_model(device=device, region=region, name="{}:{}".format(model, lam),
                      equation="simplify(diff({}, {}))".format(rate, lam))
        ds.node_model(device=device, region=region, name="ElectronGeneration:{}".format(lam),
                      equation="-ElectronCharge * {}:{}".format(model, lam))
        ds.node_model(device=device, region=region, name="HoleGeneration:{}".format(lam),
                      equation="ElectronCharge * {}:{}".format(model, lam))


def setup_contact_bipolar_dg(device: str, contact: str) -> None:
    """Ohmic contact rows for the bipolar DG carriers of the contact's region (call AFTER
    physics_bipolar_dd.setup_contact_ohmic_bipolar[_circuit] and setup_bipolar_dg). For each DG
    carrier c: Lambda_c = 0 and u_c^2 = c (the SAME tie as the bulk, so u follows whatever the one
    ohmic contact rule pins for c -- the charge-neutral value, BGN / ionization / traps included),
    and the contact's continuity equation reports its terminal current from the DG flux (audit S1-5
    for both carriers; circuit contacts keep their circuit_node).

    Why Lambda = 0: the ohmic contact pins c to the CLASSICAL charge-neutral density at the contact's
    psi and phi_c = bias; the modified quasi-Fermi relation c = c_cl exp(Lambda_c/V_t) then forces
    Lambda_c = 0 there -- a consequence of the classical contact rule, not an extra assumption (the
    bulk-reservoir condition of the DG literature; see the D2d audit doc, sec. 1.4)."""
    region = ds.get_region_list(device=device, contact=contact)[0]
    carriers = bipolar_dg_carriers(device, region)
    if not carriers:
        raise ValueError("setup_contact_bipolar_dg('{}'): region '{}' carries no bipolar DG carrier -- "
                         "call setup_bipolar_dg on it first".format(contact, region))
    for key in carriers:
        sp = _BIPOLAR_DG[key]
        _repoint_contact_current_to_dg(device, contact, eq=sp["eq"], cur=sp["cur"],
                                       cur_dg=sp["cur_dg"], who="setup_contact_bipolar_dg",
                                       installer="setup_bipolar_dg")
        if key == "electrons" and electron_valley_sets(device, region):
            # v0.15.1: each valley set gets Lambda_k = 0 and the bulk u-rows (the classical split n_k = s_k n)
            _valley_contact_rows(device, contact, region, "ohmic", None)
            continue
        u, lam, c = sp["u"], sp["lam"], sp["c"]
        cu = "{}_{}_ohmic".format(contact, u.lower())
        ds.contact_node_model(device=device, contact=contact, name=cu,
                              equation="{u}*{u} - {c}".format(u=u, c=c))
        ds.contact_node_model(device=device, contact=contact, name="{}:{}".format(cu, u),
                              equation="2*{}".format(u))
        ds.contact_node_model(device=device, contact=contact, name="{}:{}".format(cu, c),
                              equation="-1")
        _R.record_contact_equation(device, contact, name=u + "Equation", node_model=cu)
        cl = "{}_{}_dirichlet".format(contact, lam.lower())
        ds.contact_node_model(device=device, contact=contact, name=cl, equation=lam)
        ds.contact_node_model(device=device, contact=contact, name="{}:{}".format(cl, lam),
                              equation="1")
        _R.record_contact_equation(device, contact, name=lam + "Equation", node_model=cl)


def _contact_nodes(device: str, region: str, contact: str) -> np.ndarray:
    el = ds.get_element_node_list(device=device, region=region, contact=contact)
    return np.unique(np.asarray([i for e in el for i in e], dtype=np.int64))


# ---- walls: THE density-gradient insulator wall, for unipolar AND bipolar regions (D2d; since v0.15
# also the unipolar wall -- setup_dg_hard_wall is a deprecated shim onto it; the name is kept,
# decision 53 / D2-c) ---------------------------------------------------------------------------------

def setup_dg_insulator_wall(device: str, contact: str, *, u_floor_rel: float = 1.0e-6) -> None:
    """IMPENETRABLE semiconductor/insulator wall for every DG carrier of the contact's region
    (unipolar setup_dg_quantum_correction or bipolar setup_bipolar_dg) -- the infinite-barrier limit
    of a gate oxide, where the confined carrier's
    wavefunction (and so u = sqrt(c)) vanishes. `contact` is the DEVSIM contact tagging the
    semiconductor's boundary nodes there; its Potential row is the caller's (a surface-potential
    Dirichlet, a gate Robin row, or none = zero field), and NO current crosses it.

    The rows at each wall node, per DG carrier c:
      * u row      : u = u_floor  (u_floor_rel x sqrt(c) at the wall from the CURRENT solution --
                     the classical one; a tiny POSITIVE floor because u's variable_update='positive'
                     forbids an exact 0; c(wall) = u_floor^2 = 1e-12 c_cl by default);
      * Lambda row : c - u^2 = 0  (the bulk tie, moved to the Lambda row);
      * c row      : the NATURAL continuity row -- zero normal flux through the insulator.
    Lambda(wall) is therefore not pinned at all: it is the value that makes the wall edge's SG flux
    vanish with c(wall) = u_floor^2, i.e. Lambda_wall ~ V_t ln(u_floor^2 / c_cl) (the finite,
    floor-dependent stand-in for the physical log divergence: the frozen-potential first integral
    (u')^2 = (V_t/b)(u^2 ln(u^2/n0) - u^2 + n0) keeps u'(wall) finite while Lambda = V_t ln(u^2/n0)
    diverges, u Lambda -> 0). The floor only fixes the wall node's own (dead) density; the profile
    beyond the wall node does not depend on it (gated in tests/test_bipolar_density_gradient.py and,
    unipolar, tests/test_density_gradient.py). Seed with seed_bipolar_dg / seed_dg_from_solution
    (walls=(contact,)).

    Why not the retired setup_dg_hard_wall rows: there the CARRIER row was pinned (c = u^2) and Lambda
    to a regularization depth, so the wall node was a carrier SINK whose SG flux was driven by the
    pinned Lambda jump -- a spurious wide-depletion branch (deficit 23-28 n0 L_q against the closed
    form's 1.187). Here the continuity row stays natural, so the wall conserves carriers by
    construction (wall-edge current <= 1e-14 of q mu n0 V_t/L_q, measured). On a carrier closed by
    physics_bipolar_dd.setup_quasi_equilibrium_carrier the natural row IS the quasi-equilibrium row."""
    if not (0.0 < u_floor_rel < 1.0):
        raise ValueError("setup_dg_insulator_wall: u_floor_rel must be in (0, 1)")
    region = ds.get_region_list(device=device, contact=contact)[0]
    vsets = electron_valley_sets(device, region)
    keys = tuple(k for k, sp in _BIPOLAR_DG.items()
                 if sp["u"] in ds.get_node_model_list(device=device, region=region)
                 or (k == "electrons" and vsets))
    if not keys:
        raise ValueError("setup_dg_insulator_wall('{}'): region '{}' carries no DG carrier".format(
            contact, region))
    idx = _contact_nodes(device, region, contact)
    for key in keys:
        sp = _BIPOLAR_DG[key]
        u, lam, c = sp["u"], sp["lam"], sp["c"]
        cv = np.asarray(ds.get_node_model_values(device=device, region=region, name=c))[idx]
        if not (cv.size and np.all(np.isfinite(cv)) and np.max(cv) > 0.0):
            raise ValueError("setup_dg_insulator_wall('{}'): {} at the wall is not positive -- solve "
                             "the classical device first (the floor is relative to it)".format(
                                 contact, c))
        if key == "electrons" and vsets:
            # v0.15.1: u_k = u_floor_rel sqrt(s_k c) for every valley set; the u-rows move to the Lambda rows
            _valley_contact_rows(device, contact, region, "wall", u_floor_rel * np.sqrt(np.max(cv)))
            continue
        u_floor = float(u_floor_rel * np.sqrt(np.max(cv)))
        cu = "{}_{}_wall".format(contact, u.lower())
        ds.contact_node_model(device=device, contact=contact, name=cu,
                              equation="{} - {:.16e}".format(u, u_floor))
        ds.contact_node_model(device=device, contact=contact, name="{}:{}".format(cu, u),
                              equation="1")
        _R.record_contact_equation(device, contact, name=u + "Equation", node_model=cu)
        cl = "{}_{}_wall".format(contact, lam.lower())
        ds.contact_node_model(device=device, contact=contact, name=cl,
                              equation="{c} - {u}*{u}".format(c=c, u=u))
        ds.contact_node_model(device=device, contact=contact, name="{}:{}".format(cl, c),
                              equation="1")
        ds.contact_node_model(device=device, contact=contact, name="{}:{}".format(cl, u),
                              equation="-2*{}".format(u))
        _R.record_contact_equation(device, contact, name=lam + "Equation", node_model=cl)


def seed_bipolar_dg(device: str, region: str, *, walls=(), u_floor_rel: float = 1.0e-6) -> None:
    """Seed every DG carrier of a bipolar region from the CURRENT (converged classical) solution:
    u = sqrt(c), Lambda = 0 -- and, next to each insulator wall contact in `walls`, the hard-wall
    boundary-layer shape t(d) = max(tanh(d/L_q), u_floor_rel) at distance d from the wall
    (L_q = sqrt(b/V_t) of the carrier, at the CURRENT b): u = t sqrt(c_cl), c = t^2 c_cl and
    Lambda = V_t ln(t^2), the frozen-potential Boltzmann relation, so the seed's wall-edge fluxes
    already vanish. Seeding the flat classical profile against a wall overflows the first Newton
    transients (the retired unipolar hard wall's experience). The kernel (_seed_dg) is shared with
    the unipolar seed_dg_from_solution."""
    _seed_dg(device, region, walls=walls, u_floor_rel=u_floor_rel, who="seed_bipolar_dg")


def _seed_dg(device: str, region: str, *, walls, u_floor_rel: float, who: str) -> None:
    """THE density-gradient seed kernel (seed_bipolar_dg, seed_dg_from_solution): every DG carrier
    of the region (bipolar_dg_carriers -- a unipolar region's QSqrtN + ElectronCurrentDG count as
    'electrons'). walls=() writes u = 1.0 * sqrt(max(c, 0)) and Lambda = V_t ln(1) = 0.0: bitwise the
    pre-v0.15 unipolar seed (gated)."""
    keys = bipolar_dg_carriers(device, region)
    if not keys:
        raise ValueError("{}: region {!r} carries no density-gradient carrier".format(who, region))
    v_t = float(ds.get_parameter(device=device, region=region, name="V_t"))
    nodes = set(ds.get_node_model_list(device=device, region=region))
    xyz = np.stack([np.asarray(ds.get_node_model_values(device=device, region=region, name=a))
                    for a in ("x", "y", "z") if a in nodes], axis=1)
    d = None
    if walls:
        from scipy.spatial import cKDTree        # nearest-wall distance, O(N log W) memory-safe
        wall_pts = np.concatenate([xyz[_contact_nodes(device, region, w)] for w in walls])
        d = cKDTree(wall_pts).query(xyz)[0]
    for key in keys:
        sp = _BIPOLAR_DG[key]
        if key == "electrons" and electron_valley_sets(device, region):
            _seed_valleys(device, region, d, v_t, u_floor_rel)          # v0.15.1
            continue
        c = np.maximum(np.asarray(ds.get_node_model_values(device=device, region=region,
                                                           name=sp["c"])), 0.0)
        t = np.ones_like(c)
        b = float(ds.get_parameter(device=device, region=region, name=sp["b"]))
        if d is not None and b > 0.0:
            t = np.maximum(np.tanh(d / np.sqrt(b / v_t)), u_floor_rel)
        u = t * np.sqrt(c)
        ds.set_node_values(device=device, region=region, name=sp["u"], values=list(u))
        if d is not None and b > 0.0:
            ds.set_node_values(device=device, region=region, name=sp["c"],
                               values=list(np.maximum(u * u, 1e-300)))
        ds.set_node_values(device=device, region=region, name=sp["lam"],
                           values=list(v_t * np.log(t * t)))


# =================================================================================================
# v0.15.1: VALLEY-RESOLVED electron density gradient (opt-in: setup_bipolar_dg(electron_valleys=...)).
# Doc: docs/audit/2026-11-10-v0.15.1-valley-resolved-dg.md. Valley set k of the table
# (density_gradient.ValleySet; set 0 is the REFERENCE set), N_k = dos_share_k N_c:
#
#   variables   u_k = QSqrtN_v<k> = sqrt(n_k),   Lambda_k = QLambda_v<k>,   b_k = gamma hbar^2/(6 m_z,k q)
#   Lambda-row  b_k lap(u_k) - Lambda_k u_k = 0          (the shared kernel _install_dg_variables, per set)
#   u-rows      k = 0 :  sum_j u_j^2 - n = 0             (Poisson, SRH and continuity see the TOTAL n)
#               k >= 1:  V_t [A(n_k/N_k) - A(n_0/N_0)] - (Lambda_k - Lambda_0) = 0
#   A(x) = ln x + Delta(x): the region's activity (Boltzmann ln x; statistics="fermi_dirac": the FD built-in
#   correction of physics_heterojunction.setup_log_activity -- reused, not re-implemented).
#
# Why the second row: n_k = N_k F(eta + (psi + Lambda_k)/V_t) with ONE eta for all sets (one electron
# quasi-Fermi level: intervalley equilibrium) means A(n_k/N_k) - Lambda_k/V_t = eta + psi/V_t is the same
# for every set. With Lambda_k = 0, n_k/N_k = n/N_c: the classical split is n_k = dos_share_k n exactly.
#
# THE CURRENT (one continuity row for the total n). A common quasi-Fermi level phi_n and one mobility give
# J = q mu n grad(phi_n) for the TOTAL density, and the total obeys
#     n = sum_k N_k F(eta + (psi + Lambda_k)/V_t)  =:  N_c F(eta + (psi + Lambda_eff)/V_t),
# which defines Lambda_eff uniquely (F is monotone). The total is therefore a SINGLE population in the
# effective band edge psi + Lambda_eff, and the single-population DG Scharfetter-Gummel flux applies with
# Lambda -> Lambda_eff, unchanged (same kernel, same vd_expr, BGN / FD quasi-fields included). At a node,
# using the k >= 1 rows (any set gives the same value; set 0 is used),
#     Lambda_eff = Lambda_0 + V_t [A(U/N_c) - A(n_0/N_0)],   U = sum_k u_k^2      (= n by the k = 0 row)
# (Boltzmann: V_t ln sum_k s_k exp(Lambda_k/V_t)). U, not the Electrons variable: with A(n/N_c) the flux
# carries ln(n) of its own density -- the quasi-Fermi form of SG, exponential in n along a Newton step --
# and a p-n diode's Newton DIVERGED on the 0.40 -> 0.35 V step back (MEASURED: relative update 1e3 at
# iteration 0, 1e116 residual at 3); with U the flux depends on n only as the single-population one does
# (prefactor, FD quasi-field) and on the sets' own (u_k, Lambda_0), and the same ramp converges.
# QLambda is that node MODEL here (not a variable); its exact derivatives w.r.t. u_0..u_{V-1} and Lambda_0
# are models, and DEVSIM's diff() picks them up by name, so every consumer of QLambda (the flux, the quantum
# SRH) gets the exact Jacobian through the chain rule. Equilibrium: the flux vanishes iff the edge's two nodes
# share eta, i.e. phi_n flat -- exactly the closure's state (setup_quasi_equilibrium_carrier).
# Per-set SG fluxes summed would instead hold each set's current constant along an edge, i.e. NO
# intervalley transfer inside an edge, contradicting the shared quasi-Fermi level; the single flux on the
# total keeps the physics stated above.
# NOT modelled: valley-dependent mobility (the conductivity-mass anisotropy of the sets: one mu_n for all),
# intervalley non-equilibrium, holes (single population), oxide penetration (infinite barrier).
# =================================================================================================

_VDG_SETS = "vdg_n_sets"          # region parameter: number of electron valley sets (absent: not valley-resolved)


def _vk(name: str, k: int) -> str:
    return "{}_v{}".format(name, k)


def electron_valley_sets(device: str, region: str) -> int:
    """The number of electron valley sets of a region set up with setup_bipolar_dg(electron_valleys=...)
    (>= 2), or 0 when the electrons are not valley-resolved (a one-set table IS the isotropic model and
    installs its rows, so it reads 0 too)."""
    if _VDG_SETS not in ds.get_parameter_list(device=device, region=region):
        return 0
    return int(round(float(ds.get_parameter(device=device, region=region, name=_VDG_SETS))))


def _valley_fd(device: str, region: str) -> bool:
    """True for a statistics='fermi_dirac' region, False for 'boltzmann'; 'legacy' is refused (its averaged-g
    transport has no activity equilibrium for the sets to share)."""
    rp = ds.get_parameter_list(device=device, region=region)
    fd = float(ds.get_parameter(device=device, region=region, name="fd_enhancement")) > 0.0
    if fd and "fd_detailed_balance" not in rp:
        raise ValueError("setup_bipolar_dg(electron_valleys=...): region {!r} uses statistics='legacy' (the v0.14 "
                         "averaged-g transport); the valley-resolved DG needs statistics='fermi_dirac' or "
                         "'boltzmann'".format(region))
    return fd


def _valley_tie(k: int, nsets: int, fd: bool) -> tuple:
    """(expression, wrt) of valley set k's u-row (module block above): k = 0 the total-density sum, k >= 1 the
    shared quasi-Fermi level against set 0, in volts. THE one copy of these rows: the bulk, the ohmic contact
    and the insulator wall all use it."""
    if k == 0:
        u = tuple(_vk("QSqrtN", j) for j in range(nsets))
        return " + ".join("{0}*{0}".format(x) for x in u) + " - Electrons", u + ("Electrons",)
    dlt = " + HJ_VDGn_v{k}_Delta - HJ_VDGn_v0_Delta".format(k=k) if fd else ""
    expr = ("V_t*(2*log(QSqrtN_v{k}/QSqrtN_v0) + log(N_dos_v0/N_dos_v{k}){d}) - QLambda_v{k} + QLambda_v0"
            .format(k=k, d=dlt))
    return expr, ("QSqrtN_v0", _vk("QSqrtN", k), "QLambda_v0", _vk("QLambda", k))



def _eff_wrt(nsets: int) -> tuple:
    """The variables Lambda_eff depends on: every set's u_k and the reference set's Lambda_0."""
    return tuple(_vk("QSqrtN", k) for k in range(nsets)) + ("QLambda_v0",)


def _install_valley_electrons(device: str, region: str, valleys: tuple, gamma: float) -> tuple:
    """The valley-resolved electron DG rows (module block above) on a bipolar region; returns the flux wrt."""
    from dynameta.carriers.physics_heterojunction import fd_delta_derivative, setup_log_activity
    fd = _valley_fd(device, region)
    nsets = len(valleys)
    n_c = float(ds.get_parameter(device=device, region=region, name="N_dos"))
    ds.set_parameter(device=device, region=region, name=_VDG_SETS, value=float(nsets))
    for k, v in enumerate(valleys):
        ds.set_parameter(device=device, region=region, name=_vk("vdg_share", k), value=float(v.dos_share))
        ds.set_parameter(device=device, region=region, name=_vk("N_dos", k), value=float(v.dos_share) * n_c)
        _declare_dg_variables(device, region, _vk("QSqrtN", k), _vk("QLambda", k))
    # the set densities n_k = u_k^2 and U = sum_k u_k^2 (the sets' total, = n by the k = 0 row), with the
    # derivative chain to the u's that DEVSIM's diff() looks up by name (<model>:<variable>): the FD correction
    # Delta of each (the intervalley rows, Lambda_eff) and the reference set's activity (the closure row)
    usum = " + ".join("{0}*{0}".format(_vk("QSqrtN", k)) for k in range(nsets))
    dens = [(_vk("VDGn", k), "{0}*{0}".format(_vk("QSqrtN", k)), _vk("N_dos", k), (k,)) for k in range(nsets)]
    for nk, expr, dos, ks in dens + [("VDGn", usum, "N_dos", tuple(range(nsets)))]:
        ds.node_model(device=device, region=region, name=nk, equation=expr)
        dd = fd_delta_derivative(device, region, nk, dos) if fd else None     # HJ_<nk>_Delta:<nk>
        for k in ks:
            u = _vk("QSqrtN", k)
            ds.node_model(device=device, region=region, name="{}:{}".format(nk, u), equation="2*{}".format(u))
            if fd:
                ds.node_model(device=device, region=region, name="HJ_{}_Delta:{}".format(nk, u),
                              equation="{d}*{n}:{u}".format(d=dd, n=nk, u=u))
    act = setup_log_activity(device, region, "VDGn_v0", "N_dos_v0", fd=fd)
    ds.node_model(device=device, region=region, name=act + ":QSqrtN_v0",
                  equation="{}:VDGn_v0*VDGn_v0:QSqrtN_v0".format(act))
    for k, v in enumerate(valleys):
        _install_dg_variables(device, region, u=_vk("QSqrtN", k), lam=_vk("QLambda", k), carrier="Electrons",
                              b=_vk("b_dg", k), b_full=dg_b_coefficient(v.m_z_kg, gamma),
                              tie=_valley_tie(k, nsets, fd), declare=False)
    # Lambda_eff: the effective band-edge shift of the total density (node model + exact derivatives, carried
    # to the edges under the names diff() looks up: QLambda@n0:<var>@n0)
    dlt = " + HJ_VDGn_Delta - HJ_VDGn_v0_Delta" if fd else ""
    eff_wrt = _eff_wrt(nsets)
    _node_with_derivs(device, region, "QLambda",
                      "QLambda_v0 + V_t*(log((VDGn*N_dos_v0)/(N_dos*QSqrtN_v0*QSqrtN_v0)){})".format(dlt),
                      eff_wrt)
    ds.edge_from_node_model(device=device, region=region, node_model="QLambda")
    for w in eff_wrt:
        ds.edge_from_node_model(device=device, region=region, node_model="QLambda:" + w)
        for nd in ("n0", "n1"):
            ds.edge_model(device=device, region=region, name="QLambda@{nd}:{w}@{nd}".format(nd=nd, w=w),
                          equation="QLambda:{w}@{nd}".format(w=w, nd=nd))
    return ("Potential", "Electrons") + eff_wrt


def _valley_closure_terms(device: str, region: str) -> "dict | None":
    """For physics_bipolar_dd.setup_quasi_equilibrium_carrier on valley-resolved electrons: the closure row
    written on the reference set, R_n = phi_ref - [psi + D_c + Lambda_0 - V_t (A(n_0/N_0) + ln(N_c/n_i))]
    (= the single-population row with Lambda -> Lambda_eff; A(n/N_c) cancels exactly). None otherwise."""
    if not electron_valley_sets(device, region):
        return None
    return {"lam": "QLambda_v0", "act": "HJ_L_VDGn_v0", "var": "QSqrtN_v0"}


def valley_densities(device: str, region: str) -> "list | None":
    """[n_k node values (m^-3), one array per valley set, table order] of a valley-resolved region, else None."""
    nsets = electron_valley_sets(device, region)
    if not nsets:
        return None
    return [np.asarray(ds.get_node_model_values(device=device, region=region, name=_vk("VDGn", k)))
            for k in range(nsets)]


def _valley_contact_rows(device: str, contact: str, region: str, kind: str, floor_scale) -> None:
    """Contact rows of the valley sets (setup_contact_bipolar_dg, setup_dg_insulator_wall), from the ONE copy of
    the u-rows (_valley_tie). kind="ohmic": u-row = the bulk row, Lambda_k = 0 -- the classical split at the
    contact (as the single population's u^2 = c, Lambda = 0). kind="wall": u_k = floor_scale sqrt(s_k) (the
    floor u_floor_rel sqrt(n_k) of the classical split) and the bulk u-row moves to the Lambda row, so each
    set's Lambda at the wall is whatever makes the rows hold (zero normal flux, the single-population wall)."""
    nsets, fd = electron_valley_sets(device, region), _valley_fd(device, region)

    def row(name, eq_name, expr, wrt):
        ds.contact_node_model(device=device, contact=contact, name=name, equation=expr)
        for w in wrt:
            ds.contact_node_model(device=device, contact=contact, name="{}:{}".format(name, w),
                                  equation="simplify(diff({}, {}))".format(expr, w))
        _R.record_contact_equation(device, contact, name=eq_name, node_model=name)

    for k in range(nsets):
        u, lam = _vk("QSqrtN", k), _vk("QLambda", k)
        tie, wrt = _valley_tie(k, nsets, fd)
        if kind == "ohmic":
            row("{}_{}_ohmic".format(contact, u.lower()), u + "Equation", tie, wrt)
            row("{}_{}_dirichlet".format(contact, lam.lower()), lam + "Equation", lam, (lam,))
        else:
            share = float(ds.get_parameter(device=device, region=region, name=_vk("vdg_share", k)))
            row("{}_{}_wall".format(contact, u.lower()), u + "Equation",
                "{} - {:.16e}".format(u, float(floor_scale * np.sqrt(share))), (u,))
            row("{}_{}_wall".format(contact, lam.lower()), lam + "Equation", tie, wrt)


def _seed_valleys(device: str, region: str, d, v_t: float, u_floor_rel: float) -> None:
    """_seed_dg for valley-resolved electrons: the classical split n_k = s_k n, and next to a wall (d: the
    distance to it, None = no wall) each set's own boundary-layer shape t_k = max(tanh(d/L_q,k), u_floor_rel):
    u_k = t_k sqrt(s_k n), Lambda_k = V_t ln(t_k^2), and n = sum_k u_k^2 (the Boltzmann frozen-potential
    relation per set, which also satisfies the sets' shared quasi-Fermi level)."""
    n = np.maximum(np.asarray(ds.get_node_model_values(device=device, region=region, name="Electrons")), 0.0)
    tot = np.zeros_like(n)
    for k in range(electron_valley_sets(device, region)):
        share = float(ds.get_parameter(device=device, region=region, name=_vk("vdg_share", k)))
        b = float(ds.get_parameter(device=device, region=region, name=_vk("b_dg", k)))
        t = np.ones_like(n)
        if d is not None and b > 0.0:
            t = np.maximum(np.tanh(d / np.sqrt(b / v_t)), u_floor_rel)
        u = t * np.sqrt(share * n)
        tot += u * u
        ds.set_node_values(device=device, region=region, name=_vk("QSqrtN", k), values=list(u))
        ds.set_node_values(device=device, region=region, name=_vk("QLambda", k),
                           values=list(v_t * np.log(t * t)))
    if d is not None:
        ds.set_node_values(device=device, region=region, name="Electrons", values=list(np.maximum(tot, 1e-300)))
