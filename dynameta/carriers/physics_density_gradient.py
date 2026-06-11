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
(contact-row pins, unvalidated -- see its docstring); the validated post-hoc closure
(carriers.density_gradient) remains the dead-layer tool. Bipolar adds the hole twin
symmetrically (follow-on).
"""

from __future__ import annotations

import numpy as np

import devsim as ds

from dynameta.carriers import eq_registry as _R
from dynameta.carriers.eq_registry import edge_with_derivs as _edge_with_derivs
from dynameta.carriers.eq_registry import node_with_derivs as _node_with_derivs
from dynameta.constants import HBAR, Q_E

__all__ = ["setup_dg_quantum_correction", "setup_contact_dg", "setup_dg_hard_wall",
           "seed_dg_from_solution", "set_dg_gamma", "dg_b_coefficient"]


def dg_b_coefficient(m_eff_kg: float, gamma: float = 1.0) -> float:
    """b = gamma hbar^2 / (6 m q) [V m^2]."""
    if not (m_eff_kg > 0.0 and gamma >= 0.0):
        raise ValueError("density-gradient: m_eff_kg > 0 and gamma >= 0 required")
    return gamma * HBAR ** 2 / (6.0 * m_eff_kg * Q_E)


def setup_dg_quantum_correction(device: str, region: str, *, m_eff_kg: float,
                                gamma: float = 1.0) -> None:
    """Add the (QSqrtN, QLambda) variables + equations and re-point the electron continuity at
    the quantum-shifted Scharfetter-Gummel current (module header). Requires the classical
    unipolar setup on the region first."""
    b_full = dg_b_coefficient(m_eff_kg, gamma)
    ds.set_parameter(device=device, region=region, name="b_dg", value=b_full)
    ds.set_parameter(device=device, region=region, name="b_dg_full", value=b_full)

    for nm in ("QSqrtN", "QLambda"):
        ds.node_solution(device=device, region=region, name=nm)
        ds.edge_from_node_model(device=device, region=region, node_model=nm)

    # u-equation (variable QSqrtN): the algebraic tie u^2 = n
    _node_with_derivs(device, region, "QSqrtNConstraint", "QSqrtN*QSqrtN - Electrons",
                      ("QSqrtN", "Electrons"))
    _R.record_region_equation(device, region, name="QSqrtNEquation", variable_name="QSqrtN",
                              node_model="QSqrtNConstraint", variable_update="positive")

    # Lambda-equation (variable QLambda): b lap(u) - Lambda u = 0, assembled Poisson-style
    # (edge flux b grad(u) + node term; the node-term SIGN pairs with DEVSIM's flux orientation
    # so the converged Lambda equals +b u''/u -- pinned by the independent-stencil oracle).
    _edge_with_derivs(device, region, "QSqrtNGradFlux",
                      "b_dg*(QSqrtN@n0 - QSqrtN@n1)*EdgeInverseLength", ("QSqrtN",))
    _node_with_derivs(device, region, "QLambdaNode", "QLambda*QSqrtN", ("QLambda", "QSqrtN"))
    _R.record_region_equation(device, region, name="QLambdaEquation", variable_name="QLambda",
                              edge_model="QSqrtNGradFlux", node_model="QLambdaNode",
                              variable_update="default")

    # quantum-shifted SG current: psi_eff = psi + Lambda
    _edge_with_derivs(device, region, "vdiff_dg",
                      "(Potential@n0 + QLambda@n0 - Potential@n1 - QLambda@n1)/V_t",
                      ("Potential", "QLambda"))
    _edge_with_derivs(device, region, "vdiff_g_dg", "vdiff_dg / g_enh",
                      ("Potential", "QLambda", "Electrons"))
    # Bernoulli argument CLAMPED at +-200 (e^200 ~ 7e86, far from overflow): near a hard
    # wall the log-singular Lambda makes Newton TRANSIENTS overflow B()'s exp; the clamp
    # never binds at convergence (|vdiff| <~ 15 there), and ifelse evaluates the unclamped
    # branch with the EXACT same arithmetic, so converged results are bit-identical
    # (re-verified against the dg_dd_in_newton gates).
    _edge_with_derivs(device, region, "Bern_g_dg",
                      "B(ifelse(vdiff_g_dg > 200, 200, "
                      "ifelse(vdiff_g_dg < -200, -200, vdiff_g_dg)))",
                      ("Potential", "QLambda", "Electrons"))
    jn = ("ElectronCharge*mu_n*EdgeInverseLength*V_t*g_enh*"
          "kahan3(Electrons@n1*Bern_g_dg, Electrons@n1*vdiff_g_dg, -Electrons@n0*Bern_g_dg)")
    _edge_with_derivs(device, region, "ElectronCurrentDG", jn,
                      ("Electrons", "Potential", "QLambda"))
    ds.delete_equation(device=device, region=region, name="ElectronContinuityEquation")
    _R.forget(device, "ElectronContinuityEquation", loc=region)
    _R.record_region_equation(device, region, name="ElectronContinuityEquation",
                              variable_name="Electrons", edge_model="ElectronCurrentDG",
                              time_node_model="NCharge", variable_update="positive")


def setup_contact_dg(device: str, contact: str, n_contact_m3: float) -> None:
    """Bulk-ohmic DG contact: u = sqrt(n_contact), Lambda = 0 (no quantum correction in the
    contact reservoir)."""
    if not (n_contact_m3 > 0.0):
        raise ValueError("setup_contact_dg: n_contact_m3 must be > 0")
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


def setup_dg_hard_wall(device: str, contact: str, *, lambda_pin_factor: float = 15.0) -> None:
    """EXPERIMENTAL / NOT YET VALIDATED (see validation/_dg_hard_wall_wip.py): the Newton
    landscape near the log-singular wall is so flat that ramped solves stall on spurious
    wide-depletion states at practical tolerances -- the discrete mechanics below are pinned
    and probed, but the dead-layer profile has NOT yet been validated against the post-hoc
    BVP. The post-hoc closure (carriers.density_gradient.dg_correct_density_1d) remains THE
    validated dead-layer tool.

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
    (relative error e^(dLambda/V_t)), and the validation gates INSENSITIVITY to the factor
    (validation/_dg_hard_wall_wip.py GATE C).

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
    psi_eff,p = psi - Lambda_p."""
    if not (lambda_pin_factor > 0.0):
        raise ValueError("setup_dg_hard_wall: lambda_pin_factor must be > 0")
    from dynameta.constants import KB, Q_E as _QE, T_REF
    v_t = KB * T_REF / _QE
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
    b_full = float(ds.get_parameter(device=device, region=region, name="b_dg_full"))
    ds.set_parameter(device=device, region=region, name="b_dg", value=frac * b_full)
    try:
        pin_full = float(ds.get_parameter(device=device, name="wall_lambda_pin_full"))
    except Exception:
        return                                          # no hard wall on this device
    ds.set_parameter(device=device, name="wall_lambda_pin", value=frac * pin_full)
