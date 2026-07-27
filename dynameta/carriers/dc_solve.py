"""
Pluggable Stage-1 DC solve methods.

  "newton" : coupled Newton (DEVSIM's native ds.solve). Fast and quadratic when
             it converges -- the default; good for equilibrium and for
             well-conditioned transport.
  "gummel" : decoupled (Gummel) outer iteration -- alternately solve the Poisson
             equations with the carrier densities frozen, then the carrier
             continuity equations with the potential frozen, until both stop
             changing. Intended to be robust to the stiff Poisson<->continuity
             coupling on degenerate gated accumulation; coupled Newton no longer
             diverges there with the staged Poisson-presolve recipe
             (validation/gated_dd*.py), so Gummel remains an alternative outer
             iteration. Variables are frozen by DELETING their equations
             (via eq_registry) for one sub-solve and re-adding them after.
             VALIDATED on unipolar ohmic transport: validation/gummel_vs_newton.py
             shows the Gummel fixed point equals the coupled-Newton solution to
             machine precision (fields and terminal current) and matches the
             analytic ohmic limit. The GUMMEL ROUTE is STILL EXPERIMENTAL for
             the degenerate gated-ACCUMULATION case it was originally built
             for -- untested there; the PROVEN route for gated accumulation is
             the staged Poisson-presolve + coupled Newton recipe:
             validation/gated_dd.py (1D), gated_dd_2d.py (2D full-edge ohmic
             ground), gated_dd_builder.py (the reference metasurface). (The
             physics_drift_diffusion KNOWN LIMITATION marker this caveat once
             cited was removed 7e77f1c, 2026-06-02.) "newton" stays the default.

Add new methods by extending solve_dc's dispatch.
"""

from __future__ import annotations

import warnings
from contextlib import contextmanager
from typing import Sequence

import numpy as np
import devsim as ds

from dynameta.carriers import eq_registry as R
# audit C10-d: the Fermi-Dirac SOLUTION-level ceiling test. solve_dc is the one wrapper that owns a
# CONVERGED state (both its methods return only after the solve succeeded), which is what the check
# needs -- the doping-level test at contact setup runs before any solve and cannot see accumulation.
from dynameta.carriers.physics_bipolar_dd import warn_if_solution_exceeds_fd_ceiling

POTENTIAL_EQ = "PotentialEquation"
# The Gummel path is UNIPOLAR (electron-only): it freezes exactly these continuity equations during
# the Poisson sub-solve. A bipolar device would also need HoleContinuityEquation frozen -- use the
# Newton method (method="newton") for bipolar, which solves all equations together.
CARRIER_EQS = ("ElectronContinuityEquation",)
# Solution variables tracked for the Gummel convergence snapshot (electron-only -> no 'Holes'; there
# is also no ElectronQFL -- Electrons is the solution variable, not a quasi-Fermi level; F7).
_TRACK_VARS = ("Potential", "Electrons")


def solve_dc(device: str, *, method: str = "newton",
              abs_tol: float = 1e10, rel_tol: float = 1e-5, max_iter: int = 60,
              gummel_outer_max: int = 120, gummel_tol: float = 1e-5,
              gummel_inner_iter: int = 30,
              semiconductor_regions: Sequence[str] = (),
              verbose: bool = False) -> None:
    if method == "newton":
        ds.solve(type="dc", solver_type="direct", absolute_error=abs_tol,
                  relative_error=rel_tol, maximum_iterations=max_iter)
        warn_if_solution_exceeds_fd_ceiling(device)             # audit C10-d (converged state)
        return
    if method == "gummel":
        warnings.warn(
            "solve_dc(method='gummel'): validated against Newton on unipolar ohmic transport "
            "(validation/gummel_vs_newton.py), but EXPERIMENTAL for the degenerate "
            "gated-accumulation case it targets (convergence unproven there; the proven "
            "route is the staged Poisson-presolve + coupled Newton recipe, "
            "validation/gated_dd*.py). "
            "method='newton' (default) remains the supported general route.", stacklevel=2)
        # No carrier equations present (e.g. equilibrium mode) -> Gummel is just
        # a Poisson solve; fall back to Newton.
        if not (set(CARRIER_EQS) & R.equation_names(device)):
            ds.solve(type="dc", solver_type="direct", absolute_error=abs_tol,
                      relative_error=rel_tol, maximum_iterations=max_iter)
            warn_if_solution_exceeds_fd_ceiling(device)         # audit C10-d (converged state)
            return
        # audit C-6: the signature default semiconductor_regions=() makes the Gummel convergence
        # test VACUOUS -- _snapshot returns {}, _max_rel_change({}, {}) == 0.0 < gummel_tol, and the
        # solver reports success after ONE outer pass having never checked that anything converged.
        # Every in-repo caller passes the argument, so this only ever fires on the public-API trap.
        if not tuple(semiconductor_regions):
            raise ValueError(
                "solve_dc(method='gummel'): semiconductor_regions is EMPTY. The Gummel outer "
                "iteration measures convergence from a snapshot of {} on those regions, so with no "
                "regions the test is vacuous (max relative change over an empty set = 0.0 < "
                "gummel_tol) and the solve returns SUCCESS after one outer pass without ever "
                "converging (audit C-6). Pass the semiconductor region name(s), e.g. "
                "semiconductor_regions=['semi'] -- devsim.get_region_list(device={!r}) lists "
                "them.".format(list(_TRACK_VARS), device))
        _gummel(device, abs_tol, rel_tol, gummel_inner_iter, gummel_outer_max,
                gummel_tol, semiconductor_regions, verbose)
        # ONE check per solve_dc call, not one per outer iteration: _gummel raises unless the outer
        # loop actually converged, so this line is reached only with a converged state in hand.
        warn_if_solution_exceeds_fd_ceiling(device)             # audit C10-d
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
    # Symmetric relative change with a sane floor: denominator = max(|after|,|before|,1e-6).
    # The old 1e-30 floor made a variable that legitimately passes through 0 (Potential
    # starts near 0) report an enormous spurious relative change (gummel-only; the path is
    # experimental, but the convergence proxy should still be meaningful).
    m = 0.0
    for k, va in after.items():
        vb = before.get(k)
        if vb is None:
            continue
        denom = np.maximum(np.maximum(np.abs(va), np.abs(vb)), 1e-6)
        m = max(m, float(np.max(np.abs(va - vb) / denom)))
    return m


@contextmanager
def _frozen(device, eq_names):
    """Freeze the variables of `eq_names` (DEVSIM has no equation-subset flag, so "freeze" means
    DELETE the equations) for the duration of the block, and ALWAYS re-add them.

    audit C-5: the freeze/solve/thaw sequence used to be three bare statements, so a Newton
    failure inside the sub-solve (`ds.solve` raises `devsim.error` on non-convergence -- the
    ROUTINE outcome the Gummel outer loop exists to tolerate) propagated out with the equations
    still deleted. The live DEVSIM device was then permanently missing an equation while
    eq_registry still recorded it, so every later solve on that device -- including a caller's
    fall-back to method='newton' -- silently solved a DIFFERENT, under-determined problem.
    Probed on DEVSIM 2.10.0. `reapply_by_name` re-creates from the recorded kwargs and the node/
    edge models persist across the delete, so re-adding is safe on any exit path."""
    for name in eq_names:
        R.delete_by_name(device, name)
    try:
        yield
    finally:
        for name in eq_names:
            R.reapply_by_name(device, name)


def _gummel(device, abs_tol, rel_tol, inner_iter, outer_max, outer_tol,
             semi_regions, verbose):
    for outer in range(outer_max):
        before = _snapshot(device, semi_regions)
        if not before:
            # audit C-6 (second mode): non-empty regions that carry NONE of the tracked variables
            # (a misspelled region name -- _snapshot swallows the per-variable lookup error) leaves
            # the convergence test just as vacuous as an empty tuple.
            raise ValueError(
                "solve_dc(method='gummel'): none of the tracked solution variables {} were "
                "readable on semiconductor_regions={} -- the convergence test would be vacuous "
                "(audit C-6). Check the region names against "
                "devsim.get_region_list(device={!r}).".format(list(_TRACK_VARS),
                                                              list(semi_regions), device))
        # (1) Poisson sub-solve: freeze carriers
        with _frozen(device, CARRIER_EQS):
            ds.solve(type="dc", solver_type="direct", absolute_error=abs_tol,
                      relative_error=rel_tol, maximum_iterations=inner_iter)
        # (2) Continuity sub-solve: freeze Potential
        with _frozen(device, (POTENTIAL_EQ,)):
            ds.solve(type="dc", solver_type="direct", absolute_error=abs_tol,
                      relative_error=rel_tol, maximum_iterations=inner_iter)

        delta = _max_rel_change(_snapshot(device, semi_regions), before)
        if verbose:
            print("[gummel]   outer {:3d}: max rel change = {:.3e}".format(outer, delta),
                   flush=True)
        if delta < outer_tol:
            return
    raise RuntimeError(
        "Gummel did not converge in {} outer iterations".format(outer_max))
