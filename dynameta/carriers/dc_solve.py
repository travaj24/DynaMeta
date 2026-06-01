"""
Pluggable Stage-1 DC solve methods.

  "newton" : coupled Newton (DEVSIM's native ds.solve). Fast and quadratic when
             it converges -- the default; good for equilibrium and for
             well-conditioned transport.
  "gummel" : decoupled (Gummel) outer iteration -- alternately solve the Poisson
             equations with the carrier densities frozen, then the carrier
             continuity equations with the potential frozen, until both stop
             changing. Intended to be robust to the stiff Poisson<->continuity
             coupling that makes coupled Newton diverge on degenerate gated
             accumulation. Variables are frozen by DELETING their equations
             (via eq_registry) for one sub-solve and re-adding them after.
             EXPERIMENTAL: it has no validation coverage and is NOT yet known to
             converge the gated-accumulation case it was built for (see
             physics_drift_diffusion KNOWN LIMITATION); "newton" is the default.

Add new methods by extending solve_dc's dispatch.
"""

from __future__ import annotations

import warnings
from typing import Sequence

import numpy as np
import devsim as ds

from dynameta.carriers import eq_registry as R

POTENTIAL_EQ = "PotentialEquation"
CARRIER_EQS = ("ElectronContinuityEquation",)   # extensible (e.g. holes)
# Solution variables tracked for the Gummel convergence snapshot. (No ElectronQFL: there
# is no quasi-Fermi-level formulation -- Electrons/Holes are the solution variables; F7.)
_TRACK_VARS = ("Potential", "Electrons", "Holes")


def solve_dc(device: str, *, method: str = "newton",
              abs_tol: float = 1e10, rel_tol: float = 1e-5, max_iter: int = 60,
              gummel_outer_max: int = 120, gummel_tol: float = 1e-5,
              gummel_inner_iter: int = 30,
              semiconductor_regions: Sequence[str] = (),
              verbose: bool = False) -> None:
    if method == "newton":
        ds.solve(type="dc", solver_type="direct", absolute_error=abs_tol,
                  relative_error=rel_tol, maximum_iterations=max_iter)
        return
    if method == "gummel":
        warnings.warn(
            "solve_dc(method='gummel') is EXPERIMENTAL: no validation coverage and not yet "
            "known to converge the gated-accumulation case it targets. Use method='newton' "
            "(default) or the equilibrium physics mode.", stacklevel=2)
        # No carrier equations present (e.g. equilibrium mode) -> Gummel is just
        # a Poisson solve; fall back to Newton.
        if not (set(CARRIER_EQS) & R.equation_names(device)):
            ds.solve(type="dc", solver_type="direct", absolute_error=abs_tol,
                      relative_error=rel_tol, maximum_iterations=max_iter)
            return
        _gummel(device, abs_tol, rel_tol, gummel_inner_iter, gummel_outer_max,
                gummel_tol, semiconductor_regions, verbose)
        return
    raise ValueError("unknown dc solve method: {!r}".format(method))


def _snapshot(device, regions):
    out = {}
    for r in regions:
        for v in _TRACK_VARS:
            try:
                out[(r, v)] = np.array(
                    ds.get_node_model_values(device=device, region=r, name=v))
            except Exception:
                pass   # variable not present in this region
    return out


def _max_rel_change(after, before):
    m = 0.0
    for k, va in after.items():
        vb = before.get(k)
        if vb is None:
            continue
        m = max(m, float(np.max(np.abs(va - vb) / np.maximum(np.abs(vb), 1e-30))))
    return m


def _gummel(device, abs_tol, rel_tol, inner_iter, outer_max, outer_tol,
             semi_regions, verbose):
    for outer in range(outer_max):
        before = _snapshot(device, semi_regions)
        # (1) Poisson sub-solve: freeze carriers
        for ceq in CARRIER_EQS:
            R.delete_by_name(device, ceq)
        ds.solve(type="dc", solver_type="direct", absolute_error=abs_tol,
                  relative_error=rel_tol, maximum_iterations=inner_iter)
        for ceq in CARRIER_EQS:
            R.reapply_by_name(device, ceq)
        # (2) Continuity sub-solve: freeze Potential
        R.delete_by_name(device, POTENTIAL_EQ)
        ds.solve(type="dc", solver_type="direct", absolute_error=abs_tol,
                  relative_error=rel_tol, maximum_iterations=inner_iter)
        R.reapply_by_name(device, POTENTIAL_EQ)

        delta = _max_rel_change(_snapshot(device, semi_regions), before)
        if verbose:
            print("[gummel]   outer {:3d}: max rel change = {:.3e}".format(outer, delta),
                   flush=True)
        if delta < outer_tol:
            return
    raise RuntimeError(
        "Gummel did not converge in {} outer iterations".format(outer_max))
