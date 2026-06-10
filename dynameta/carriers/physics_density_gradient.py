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
(u -> 0, the MOS dead layer) needs an interface equation on the semiconductor-oxide boundary --
a follow-on; the validated post-hoc closure (carriers.density_gradient) remains the dead-layer
tool. Bipolar adds the hole twin symmetrically (follow-on).
"""

from __future__ import annotations

import numpy as np

import devsim as ds

from dynameta.carriers import eq_registry as _R
from dynameta.carriers.eq_registry import edge_with_derivs as _edge_with_derivs
from dynameta.carriers.eq_registry import node_with_derivs as _node_with_derivs
from dynameta.constants import HBAR, Q_E

__all__ = ["setup_dg_quantum_correction", "setup_contact_dg", "seed_dg_from_solution",
           "set_dg_gamma", "dg_b_coefficient"]


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
    _edge_with_derivs(device, region, "Bern_g_dg", "B(vdiff_g_dg)",
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


def seed_dg_from_solution(device: str, region: str) -> None:
    """Seed QSqrtN = sqrt(Electrons) (current solution) and QLambda = 0 -- call after the
    CONVERGED classical solve, before the gamma ramp."""
    n = np.asarray(ds.get_node_model_values(device=device, region=region, name="Electrons"))
    ds.set_node_values(device=device, region=region, name="QSqrtN",
                       values=list(np.sqrt(np.maximum(n, 0.0))))
    ds.set_node_values(device=device, region=region, name="QLambda",
                       values=[0.0] * n.size)


def set_dg_gamma(device: str, region: str, frac: float) -> None:
    """Scale b_dg to `frac` of its full value (the gamma-ramp convergence aid; frac in [0,1])."""
    if not (0.0 <= frac <= 1.0):
        raise ValueError("set_dg_gamma: frac must be in [0, 1]")
    b_full = float(ds.get_parameter(device=device, region=region, name="b_dg_full"))
    ds.set_parameter(device=device, region=region, name="b_dg", value=frac * b_full)
