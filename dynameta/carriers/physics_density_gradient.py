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
Lambda to 0 -- bulk contacts carry no quantum correction). The oxide-interface hard wall
(u -> 0, the MOS dead layer) now has an EXPERIMENTAL implementation, setup_dg_hard_wall
(contact-row pins; the pins are gated in tests/test_density_gradient.py, the resulting dead-layer
PROFILE is not validated -- see its docstring); the validated post-hoc closure
(carriers.density_gradient) remains the dead-layer tool.

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
from dynameta.carriers.eq_registry import edge_with_derivs as _edge_with_derivs
from dynameta.carriers.eq_registry import node_with_derivs as _node_with_derivs
from dynameta.constants import HBAR, Q_E

__all__ = ["setup_dg_quantum_correction", "setup_contact_dg", "setup_dg_hard_wall",
           "seed_dg_from_solution", "set_dg_gamma", "dg_b_coefficient",
           # D2d: bipolar density-gradient
           "setup_bipolar_dg", "setup_contact_bipolar_dg", "setup_dg_insulator_wall",
           "seed_bipolar_dg", "bipolar_dg_carriers", "DG_SRH_TERM"]


def dg_b_coefficient(m_eff_kg: float, gamma: float = 1.0) -> float:
    """b = gamma hbar^2 / (6 m q) [V m^2]."""
    if not (m_eff_kg > 0.0 and gamma >= 0.0):
        raise ValueError("density-gradient: m_eff_kg > 0 and gamma >= 0 required")
    return gamma * HBAR ** 2 / (6.0 * m_eff_kg * Q_E)


def _install_dg_variables(device: str, region: str, *, u: str, lam: str, carrier: str,
                          b: str, b_full: float) -> None:
    """THE density-gradient kernel, shared by the unipolar electron DG and both carriers of the
    bipolar DG (D2d): the parameter pair `b` / `b`_full, the solution variables u = sqrt(carrier)
    and Lambda, the algebraic tie u^2 - c = 0 and the Poisson-style Lambda-equation
    b lap(u) - Lambda u = 0 (edge flux b grad(u) + node term; the node-term SIGN pairs with DEVSIM's
    flux orientation so the converged Lambda equals +b u''/u -- pinned by the independent-stencil
    oracle of validation/dg_dd_in_newton.py GATE B). With (u, lam, carrier, b) = (QSqrtN, QLambda,
    Electrons, b_dg) every DEVSIM call is the pre-D2d unipolar one, in order, character-for-character
    (tests/_dg_calllog.py digest)."""
    ds.set_parameter(device=device, region=region, name=b, value=b_full)
    ds.set_parameter(device=device, region=region, name=b + "_full", value=b_full)

    for nm in (u, lam):
        ds.node_solution(device=device, region=region, name=nm)
        ds.edge_from_node_model(device=device, region=region, node_model=nm)

    # u-equation (variable u): the algebraic tie u^2 = c
    _node_with_derivs(device, region, u + "Constraint", "{u}*{u} - {c}".format(u=u, c=carrier),
                      (u, carrier))
    _R.record_region_equation(device, region, name=u + "Equation", variable_name=u,
                              node_model=u + "Constraint", variable_update="positive")

    # Lambda-equation (variable lam): b lap(u) - Lambda u = 0, assembled Poisson-style
    _edge_with_derivs(device, region, u + "GradFlux",
                      "{b}*({u}@n0 - {u}@n1)*EdgeInverseLength".format(b=b, u=u), (u,))
    _node_with_derivs(device, region, lam + "Node", "{l}*{u}".format(l=lam, u=u), (lam, u))
    _R.record_region_equation(device, region, name=lam + "Equation", variable_name=lam,
                              edge_model=u + "GradFlux", node_model=lam + "Node",
                              variable_update="default")


def _clamped_bernoulli(arg: str) -> str:
    """B(arg) with the argument CLAMPED at +-200 (e^200 ~ 7e86, far from overflow): near a hard wall
    the log-singular Lambda makes Newton TRANSIENTS overflow B()'s exp; the clamp never binds at
    convergence (|vdiff| <~ 30 there), and ifelse evaluates the unclamped branch with the EXACT same
    arithmetic, so converged results are bit-identical (re-verified against the dg_dd_in_newton
    gates)."""
    return "B(ifelse({a} > 200, 200, ifelse({a} < -200, -200, {a})))".format(a=arg)


def _upper_clamped_bernoulli(arg: str) -> str:
    """B(arg) with the argument clamped at +200 ONLY -- the Bernoulli pair of the bipolar DG's two-
    Bernoulli flux (_JN_DG_BIPOLAR). e^arg can overflow only for arg -> +inf, where B -> 0 anyway; for
    arg -> -inf B(arg) -> -arg, finite, with slope -1. Clamping that side too (as _clamped_bernoulli
    does) makes BOTH Bernoullis of the pair constant once |v| > 200, so a wall edge's flux -- the only
    equation that sees the wall node's Lambda -- loses its whole Lambda dependence in a Newton
    transient and UMFPACK meets an exact zero column (MEASURED 2026-11-03: GATE B flat_wall
    "Convergence failure!" and inversion_equilibrium(quantum_srh=False) "umfpack numeric failed" under
    both OpenBLAS builds with the two-sided clamp; the kahan3 form never noticed because its unclamped
    c@n1 v term kept a slope). DEVSIM 2.11.2's own B / dBdx are overflow-safe at any argument
    (measured B(1e5) = 0, B(-1e5) = 1e5, dBdx = -0 / -1, no inf/NaN), so the clamp is belt-and-braces;
    it never binds at convergence."""
    return "B(ifelse({a} > 200, 200, {a}))".format(a=arg)


def setup_dg_quantum_correction(device: str, region: str, *, m_eff_kg: float,
                                gamma: float = 1.0) -> None:
    """Add the (QSqrtN, QLambda) variables + equations and re-point the electron continuity at
    the quantum-shifted Scharfetter-Gummel current (module header). Requires the classical
    unipolar setup on the region first. (Bipolar regions: setup_bipolar_dg.)"""
    b_full = dg_b_coefficient(m_eff_kg, gamma)
    _install_dg_variables(device, region, u="QSqrtN", lam="QLambda", carrier="Electrons",
                          b="b_dg", b_full=b_full)

    # quantum-shifted SG current: psi_eff = psi + Lambda
    _edge_with_derivs(device, region, "vdiff_dg",
                      "(Potential@n0 + QLambda@n0 - Potential@n1 - QLambda@n1)/V_t",
                      ("Potential", "QLambda"))
    _edge_with_derivs(device, region, "vdiff_g_dg", "vdiff_dg / g_enh",
                      ("Potential", "QLambda", "Electrons"))
    _edge_with_derivs(device, region, "Bern_g_dg", _clamped_bernoulli("vdiff_g_dg"),
                      ("Potential", "QLambda", "Electrons"))
    _edge_with_derivs(device, region, "ElectronCurrentDG", _JN_DG,
                      ("Electrons", "Potential", "QLambda"))
    ds.delete_equation(device=device, region=region, name="ElectronContinuityEquation")
    _R.forget(device, "ElectronContinuityEquation", loc=region)
    _R.record_region_equation(device, region, name="ElectronContinuityEquation",
                              variable_name="Electrons", edge_model="ElectronCurrentDG",
                              time_node_model="NCharge", variable_update="positive")


# the quantum-shifted Scharfetter-Gummel fluxes -- the classical bipolar strings
# (physics_bipolar_dd) with vdiff_g -> vdiff_g_dg and Bern_g -> Bern_g_dg (holes: _gp -> _gp_dg)
_JN_DG = ("ElectronCharge*mu_n*EdgeInverseLength*V_t*g_enh*"
          "kahan3(Electrons@n1*Bern_g_dg, Electrons@n1*vdiff_g_dg, -Electrons@n0*Bern_g_dg)")
_JP_DG = ("-ElectronCharge*mu_p*EdgeInverseLength*V_t*g_enh_p*"
          "kahan3(Holes@n1*Bern_gp_dg, -Holes@n0*Bern_gp_dg, -Holes@n0*vdiff_gp_dg)")

# The BIPOLAR DG fluxes (setup_bipolar_dg) in the TWO-BERNOULLI form
#     Jn ~ n@n1 B(-v) - n@n0 B(v) ,   Jp ~ -( p@n1 B(v) - p@n0 B(-v) ) ,
# NOT the kahan3 form above, which spells B(-v) as B(v) + v. That identity cancels catastrophically
# for v << 0: B(v) ~ |v| and v ~ -|v| leave B(-v) ~ |v| e^-|v| as the difference of two O(|v|)
# numbers, so its relative error is ~ eps |v| e^|v| -- MEASURED against mpmath on DEVSIM 2.11.2
# (2026-11-03): 1.9e-7 at v = -21, 6.5e-5 at -27.6, 100 % at -40 (and dBdx(v) + 1, the Jacobian,
# 3.8e-9 / 1.5e-6 / 34 %), while DEVSIM's own B(-v) and -dBdx(-v) stay <= 2.1e-16 at every v. A
# classical DD edge never sees |v| ~ 20 (the mesh resolves the Debye length), but an insulator-wall
# edge does BY CONSTRUCTION: the wall node holds c = u_floor^2 = 1e-12 c_cl, so its zero-flux edge
# sits at |v| = ln(c@n1/c@n0) ~ 20-28 -- with v < 0 for ELECTRONS when the wall is the edge's n0
# (for holes the kahan3 form multiplies the cancelling B(v) + v by the depleted p@n0, so the same
# loss hits them on the opposite edge orientation, i.e. on any 2-D/3-D wall). Measured in GATE C
# (validation/bipolar_density_gradient.py, p-type MOS, 1e24, 0.15 V): Lambda_n at the wall node --
# fixed ONLY by that edge's flux (setup_dg_insulator_wall) -- converged linearly and then sat in a
# period-3 Newton cycle at relative update 1.9-3.8e-7 for 100 iterations, against REL_ERR 1e-7, so
# the solve passed or failed on the BLAS's round-off (fails with scipy's OpenBLAS, passed with
# Elmer's in 23 iterations still dithering at 4.6-8.0e-8). The two-Bernoulli form has no
# cancellation (its subtraction is the physical diffusion-drift balance, error eps |terms| as in
# kahan3) and is algebraically the same flux. The UNIPOLAR DG keeps _JN_DG: its definition is pinned
# character-for-character by tests/_dg_calllog.py and no validated unipolar recipe has a wall, but a
# unipolar region given setup_dg_insulator_wall / setup_dg_hard_wall would carry the same loss (open:
# docs/audit/2026-11-03-devsim-bipolar-density-gradient.md, sec. 1.6).
_JN_DG_BIPOLAR = ("ElectronCharge*mu_n*EdgeInverseLength*V_t*g_enh*"
                  "(Electrons@n1*Bern_g_dg_m - Electrons@n0*Bern_g_dg)")
_JP_DG_BIPOLAR = ("-ElectronCharge*mu_p*EdgeInverseLength*V_t*g_enh_p*"
                  "(Holes@n1*Bern_gp_dg - Holes@n0*Bern_gp_dg_m)")


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


def setup_dg_hard_wall(device: str, contact: str, *, lambda_pin_factor: float = 15.0) -> None:
    """EXPERIMENTAL / NOT YET VALIDATED (gated by tests/test_density_gradient.py::
    test_dg_hard_wall_pins_the_wall_rows): the Newton landscape near the log-singular wall is so
    flat that ramped solves stall on spurious wide-depletion states at practical tolerances -- the
    discrete mechanics below are pinned and probed (that test asserts every one of them), but the
    dead-layer PROFILE has NOT been validated against the post-hoc BVP: re-measured 2026-07-26,
    the frozen-psi in-Newton profile differs from dg_correct_density_1d by max |dn|/N0 = 0.80 and
    the self-consistent leg raises a DEVSIM convergence failure. The post-hoc closure
    (carriers.density_gradient.dg_correct_density_1d) remains THE validated dead-layer tool.

    Oxide/insulator HARD WALL at `contact` for the in-Newton DG system (the MOS dead-layer
    boundary): u -> 0 (a tiny positive floor) Dirichlet, plus a DEEP REGULARIZATION pin on
    Lambda.

    Why a regularization and not a value: the equilibrium DG closure's first integral gives
    (u')^2 = (V_t/b)(u^2 ln(u^2/n0) - u^2 + n0), so u'(wall) = sqrt(V_t n0 / b) is FINITE but
    Lambda = V_t ln(u^2/n0) DIVERGES logarithmically at the wall -- there is no finite
    physical wall value to pin (the tanh profile's finite b u''/u = -V_t belongs to a
    DIFFERENT closure and does not solve this system). The wall node's Lambda row is
    therefore pinned to the deep value -lambda_pin_factor * V_t: any pin a few V_t below the
    first interior node's Lambda drives the wall-edge Scharfetter-Gummel density to ~ 0
    (relative error e^(dLambda/V_t)). Insensitivity to the factor is a claim about the CONVERGED
    profile and so is not gated while the profile itself is open (above); what IS gated is that
    the pin is installed at zero depth and co-ramps with set_dg_gamma.

    The ELECTRON row at the wall node is pinned to the SAME constraint as the bulk,
    n = u^2 (the pinned u-floor then makes n(wall) = floor^2 ~ 0, the dead-layer endpoint).
    The bulk (natural) continuity row at a bare contact node was measured NOT to behave as
    the zero-flux row (n(wall) floated pin-dependently), hence the explicit pin. A Boltzmann
    quasi-equilibrium pin n = N_D exp((Potential + QLambda)/V_t) is WRONG here: it would
    evaluate the REGULARIZATION Lambda-pin as if it were the physical (log-divergent) wall
    Lambda and return a finite density; n = u^2 references no Lambda and needs no
    exponential. Potential keeps its natural bulk row (probed on DEVSIM 2.10.0: equations
    without a contact_equation retain the bulk assembly -- no empty-row singularity).

    Continuation plan (from cef01d9): tighter-tolerance damped Newton solves or a u_floor
    continuation on the wall row; then the bipolar (hole) DG twin with
    psi_eff,p = psi - Lambda_p. The four physics gates that plan has to clear are written out in
    test_dg_hard_wall_pins_the_wall_rows' docstring (they used to live in a parked, never-executed
    validation/_dg_hard_wall_wip.py -- audit X-6)."""
    if not (lambda_pin_factor > 0.0):
        raise ValueError("setup_dg_hard_wall: lambda_pin_factor must be > 0")
    from dynameta.constants import V_T
    v_t = V_T   # single-source thermal voltage (audit S1-7)
    # the pin is a DEVICE parameter so the gamma ramp can co-ramp it (a full-depth pin at the
    # first small-gamma step makes the wall-edge Bernoulli overflow during Newton transients)
    ds.set_parameter(device=device, name="wall_lambda_pin", value=0.0)
    ds.set_parameter(device=device, name="wall_lambda_pin_full",
                     value=float(lambda_pin_factor) * v_t)
    # u is pinned to a TINY POSITIVE floor, not literal 0: the QSqrtN variable uses DEVSIM's
    # variable_update='positive' (the validated unipolar-DG choice), which FORBIDS an exact
    # zero ('Solution Variable has negative or zero value'). u_floor = 1e-6 sqrt(N_D) makes
    # n(wall) ~ 1e-12 N_D -- physically indistinguishable from a hard zero.
    region = ds.get_region_list(device=device, contact=contact)[0]
    n_d = float(ds.get_parameter(device=device, region=region, name="N_D"))
    u_floor = 1.0e-6 * np.sqrt(n_d)
    cu = "{}_qsqrtn_wall".format(contact)
    ds.contact_node_model(device=device, contact=contact, name=cu,
                          equation="QSqrtN - {:.16e}".format(u_floor))
    ds.contact_node_model(device=device, contact=contact, name="{}:QSqrtN".format(cu),
                          equation="1")
    _R.record_contact_equation(device, contact, name="QSqrtNEquation", node_model=cu)
    cl = "{}_qlambda_wall".format(contact)
    ds.contact_node_model(device=device, contact=contact, name=cl,
                          equation="QLambda + wall_lambda_pin")
    ds.contact_node_model(device=device, contact=contact, name="{}:QLambda".format(cl),
                          equation="1")
    _R.record_contact_equation(device, contact, name="QLambdaEquation", node_model=cl)
    ce = "{}_electrons_wall".format(contact)
    # the wall node's ELECTRON row is the SAME constraint as the bulk, n = u^2 (so the pinned
    # u-floor makes n(wall) = floor^2 ~ 0, the dead-layer endpoint). The natural bulk
    # continuity row at a bare contact node was measured to FLOAT (n(wall) pin-dependent),
    # and a Boltzmann pin n = N_D exp((psi+Lambda)/V_t) is WRONG here: it would evaluate the
    # REGULARIZATION Lambda-pin as if it were the physical (log-divergent) wall Lambda and
    # return a finite density. n = u^2 references no Lambda and needs no exponential.
    ds.contact_node_model(device=device, contact=contact, name=ce,
                          equation="Electrons - QSqrtN*QSqrtN")
    ds.contact_node_model(device=device, contact=contact, name="{}:Electrons".format(ce),
                          equation="1")
    ds.contact_node_model(device=device, contact=contact, name="{}:QSqrtN".format(ce),
                          equation="-2*QSqrtN")
    _R.record_contact_equation(device, contact, name="ElectronContinuityEquation",
                               node_model=ce)


def seed_dg_from_solution(device: str, region: str) -> None:
    """Seed QSqrtN = sqrt(Electrons) (current solution) and QLambda = 0 -- call after the
    CONVERGED classical solve, before the gamma ramp."""
    n = np.asarray(ds.get_node_model_values(device=device, region=region, name="Electrons"))
    ds.set_node_values(device=device, region=region, name="QSqrtN",
                       values=list(np.sqrt(np.maximum(n, 0.0))))
    ds.set_node_values(device=device, region=region, name="QLambda",
                       values=[0.0] * n.size)


def set_dg_gamma(device: str, region: str, frac: float) -> None:
    """Scale b_dg to `frac` of its full value (the gamma-ramp convergence aid; frac in [0,1]).
    When a hard wall is present (setup_dg_hard_wall), its Lambda pin co-ramps with the same
    fraction -- a full-depth pin against a small-gamma bulk overflows the wall-edge
    Bernoulli during Newton transients."""
    if not (0.0 <= frac <= 1.0):
        raise ValueError("set_dg_gamma: frac must be in [0, 1]")
    # D2d: a bipolar DG region (setup_bipolar_dg) may carry the electron b_dg, the hole b_dg_p, or
    # both; the hole twin co-ramps with the SAME fraction. A unipolar region always has b_dg_full and
    # never b_dg_p_full, so its DEVSIM call sequence is the pre-D2d one (tests/_dg_calllog.py digest).
    rparams = ds.get_parameter_list(device=device, region=region)
    if "b_dg_full" in rparams or "b_dg_p_full" not in rparams:
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
                      vd_expr="vdiff_g + (QLambda@n0 - QLambda@n1)/(V_t*g_enh)",
                      jexpr=_JN_DG_BIPOLAR),
    "holes": dict(u="QSqrtP", lam="QLambdaP", c="Holes", b="b_dg_p",
                  eq="HoleContinuityEquation", cur="HoleCurrent",
                  cur_dg="HoleCurrentDG", vd="vdiff_gp_dg", bern="Bern_gp_dg",
                  bern_m="Bern_gp_dg_m",
                  vd_expr="vdiff_gp - (QLambdaP@n0 - QLambdaP@n1)/(V_t*g_enh_p)",
                  jexpr=_JP_DG_BIPOLAR),
}
DG_SRH_TERM = "dgsrh"            # RTerm_dgsrh: the quantum detailed-balance correction to USRH


def bipolar_dg_carriers(device: str, region: str) -> tuple:
    """The carriers ('electrons', 'holes') that carry a density-gradient quantum potential on a
    BIPOLAR region (read back from DEVSIM: the region has the carrier's u variable and its DG
    current). () for a classical region."""
    nodes = set(ds.get_node_model_list(device=device, region=region))
    edges = set(ds.get_edge_model_list(device=device, region=region))
    return tuple(k for k, sp in _BIPOLAR_DG.items() if sp["u"] in nodes and sp["cur_dg"] in edges)


def _repoint_region_current(device: str, region: str, eq: str, cur_dg: str) -> None:
    """Re-bind a recorded REGION continuity equation to the DG flux, keeping every other kwarg the
    bipolar setup (and the additive term mechanism: node_model / edge_volume_model / element
    slots) recorded -- the generation composites, the time term and any BTBT slot survive."""
    rec = _R.recorded(device, eq, scope="region", loc=region)
    if len(rec) != 1:
        raise ValueError("setup_bipolar_dg: region {!r} has no recorded {} -- a bipolar drift-"
                         "diffusion region (physics_bipolar_dd.setup_bipolar_region) is required"
                         .format(region, eq))
    kw = dict(rec[0]["kwargs"])
    kw["edge_model"] = cur_dg
    ds.delete_equation(device=device, region=region, name=eq)
    _R.forget(device, eq, loc=region, scope="region")
    _R.record_region_equation(device, region, **kw)


def setup_bipolar_dg(device: str, region: str, *, m_eff_n_kg: "float | None" = None,
                     m_eff_p_kg: "float | None" = None, gamma_n: float = 1.0,
                     gamma_p: float = 1.0, quantum_srh: bool = True) -> tuple:
    """Opt-in BIPOLAR density-gradient on a region built by physics_bipolar_dd.setup_bipolar_region
    (call AFTER it, with any BipolarOptions / traps / BTBT already installed, and ideally after a
    converged classical solve -- the seed). A carrier gets a quantum potential iff its mass is given:
    m_eff_n_kg -> (QSqrtN, QLambda, b_dg), m_eff_p_kg -> (QSqrtP, QLambdaP, b_dg_p); the other
    carrier stays classical. b = gamma hbar^2/(6 m q) per carrier (dg_b_coefficient).

    Each carrier gets the shared kernel (_install_dg_variables) and its continuity equation is
    re-pointed at the quantum-shifted SG flux (ElectronCurrentDG / HoleCurrentDG) built on the
    region's own quasi-field (vdiff_g / vdiff_gp: FD g-factor, BGN band edges and all -- header
    block above). The generation composites, time terms and BTBT slots of the equation are kept.

    quantum_srh (default True): the built-in SRH channel's intrinsic level follows the quantum band
    edges, n_i'^2 -> n_i'^2 exp((Lambda_n + Lambda_p)/V_t) (n1', p1' by the square root;
    physics_traps.srh_reference_densities(scale=...)), entered through the ONE additive mechanism as
    RTerm_dgsrh = U_SRH(quantum) - USRH (+ its pair rate). Without it U_SRH != 0 in EQUILIBRIUM
    wherever Lambda != 0 (n p = n_i^2 e^((Ln+Lp)/V_t) there), i.e. detailed balance fails in the
    quantum dark space. Only the BUILT-IN channel is corrected: Auger / radiative / trap levels /
    interface traps / BTBT keep the classical n_i reference (doc: Limits).

    Returns the tuple of DG carriers installed."""
    if m_eff_n_kg is None and m_eff_p_kg is None:
        raise ValueError("setup_bipolar_dg: give m_eff_n_kg and/or m_eff_p_kg (the carriers that get "
                         "a quantum potential)")
    if "Holes" not in ds.get_node_model_list(device=device, region=region):
        raise ValueError("setup_bipolar_dg: region {!r} is not a bipolar region (no Holes); use "
                         "setup_dg_quantum_correction for a unipolar one".format(region))
    done = []
    for key, m, g in (("electrons", m_eff_n_kg, gamma_n), ("holes", m_eff_p_kg, gamma_p)):
        if m is None:
            continue
        sp = _BIPOLAR_DG[key]
        _install_dg_variables(device, region, u=sp["u"], lam=sp["lam"], carrier=sp["c"],
                              b=sp["b"], b_full=dg_b_coefficient(m, g))
        wrt = ("Potential", sp["lam"], sp["c"])
        _edge_with_derivs(device, region, sp["vd"], sp["vd_expr"], wrt)
        _edge_with_derivs(device, region, sp["bern"], _upper_clamped_bernoulli(sp["vd"]), wrt)
        _edge_with_derivs(device, region, sp["bern_m"],                # B(-v): see _JN_DG_BIPOLAR
                          _upper_clamped_bernoulli("(-{})".format(sp["vd"])), wrt)
        _edge_with_derivs(device, region, sp["cur_dg"], sp["jexpr"], wrt)
        _repoint_region_current(device, region, sp["eq"], sp["cur_dg"])
        done.append(key)
    if quantum_srh:
        _install_quantum_srh(device, region, tuple(done))
    return tuple(done)


def _install_quantum_srh(device: str, region: str, carriers: tuple) -> None:
    """RTerm_dgsrh (setup_bipolar_dg, quantum_srh=True): the built-in SRH channel re-referenced to
    the quantum band edges, as the additive correction U_SRH(n_i' s) - USRH with
    s = DgNieFactor = exp((Lambda_n + Lambda_p)/(2 V_t)). The exponent is clamped at +-200 like the
    Bernoulli argument (Newton transients only; it never binds at convergence). The term's Lambda
    derivatives (which the additive mechanism, written for Electrons/Holes, does not create) are
    added here and wired into the generation composites' Jacobian -- RTerm_dgsrh is the only
    Lambda-dependent term, so those composite derivatives stay valid whatever terms are added later."""
    from dynameta.carriers import physics_bipolar_dd as _BP
    from dynameta.carriers.physics_traps import srh_rate_expr
    lams = [_BIPOLAR_DG[k]["lam"] for k in carriers]
    arg = "(0.5*({})/V_t)".format(" + ".join(lams))
    _node_with_derivs(device, region, "DgNieFactor",
                      "exp(ifelse({a} > 200, 200, ifelse({a} < -200, -200, {a})))".format(a=arg),
                      tuple(lams))
    bgn = "n_ie" in ds.get_node_model_list(device=device, region=region)
    rate = "({}) - USRH".format(srh_rate_expr(bgn=bgn, scale="DgNieFactor"))
    pair = "({}) - ({})".format(srh_rate_expr(bgn=bgn, scale="DgNieFactor", pair=True),
                                srh_rate_expr(bgn=bgn, pair=True))
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


def setup_dg_insulator_wall(device: str, contact: str, *, u_floor_rel: float = 1.0e-6) -> None:
    """IMPENETRABLE semiconductor/insulator wall for every DG carrier of the contact's region
    (bipolar DG; D2d) -- the infinite-barrier limit of a gate oxide, where the confined carrier's
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
    floor-dependent stand-in for the physical log divergence -- u Lambda -> 0 at the wall; see
    setup_dg_hard_wall's docstring). The floor only fixes the wall node's own (dead) density; the
    profile beyond the wall node does not depend on it (gated in tests/test_bipolar_density_gradient.py).

    Why not setup_dg_hard_wall's rows: there the CARRIER row is pinned (c = u^2) and Lambda is pinned
    to a regularization depth, so the wall node is a carrier SINK whose SG flux is driven by the pinned
    Lambda jump -- the spurious wide-depletion branch it documents. Here the continuity row stays
    natural, so the wall conserves carriers by construction. (setup_dg_hard_wall itself is untouched --
    unipolar byte-identity.) Works on any region carrying DG carriers (unipolar QSqrtN included)."""
    if not (0.0 < u_floor_rel < 1.0):
        raise ValueError("setup_dg_insulator_wall: u_floor_rel must be in (0, 1)")
    region = ds.get_region_list(device=device, contact=contact)[0]
    keys = tuple(k for k, sp in _BIPOLAR_DG.items()
                 if sp["u"] in ds.get_node_model_list(device=device, region=region))
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
    transients (the unipolar setup_dg_hard_wall experience)."""
    keys = bipolar_dg_carriers(device, region)
    if not keys:
        raise ValueError("seed_bipolar_dg: region {!r} carries no bipolar DG carrier".format(region))
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
