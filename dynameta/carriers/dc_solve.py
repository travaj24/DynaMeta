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
             UNIPOLAR ONLY: a device carrying HoleContinuityEquation is REFUSED with a
             ValueError naming method="newton" (audit C-15).

Add new methods by extending solve_dc's dispatch.
"""

from __future__ import annotations

import warnings
from contextlib import contextmanager
from typing import Sequence

import numpy as np
from dynameta.carriers.devsim_env import import_devsim

ds = import_devsim()      # D0: devsim via its ONE import home (BLAS pinned before the first import)

from dynameta.carriers import eq_registry as R
from dynameta.carriers import extended_precision as XP
# audit C10-d: the Fermi-Dirac SOLUTION-level ceiling test. solve_dc is the one wrapper that owns a
# CONVERGED state (both its methods return only after the solve succeeded), which is what the check
# needs -- the doping-level test at contact setup runs before any solve and cannot see accumulation.
from dynameta.carriers.physics_bipolar_dd import warn_if_solution_exceeds_fd_ceiling
from dynameta.warning_types import UnreliableResultWarning

POTENTIAL_EQ = "PotentialEquation"
# The Gummel path is UNIPOLAR (electron-only): it freezes exactly these continuity equations during
# the Poisson sub-solve. A bipolar device is REFUSED (audit C-15, below) -- use method="newton".
CARRIER_EQS = ("ElectronContinuityEquation",)
# Continuity equations the Gummel splitting does NOT know how to freeze. Their presence on the device
# means the "Poisson sub-solve" would really be a coupled Poisson+hole Newton -- not Gummel at all.
_UNSUPPORTED_CARRIER_EQS = ("HoleContinuityEquation",)
# Solution variables tracked for the Gummel convergence snapshot (electron-only -> no 'Holes'; there
# is also no ElectronQFL -- Electrons is the solution variable, not a quasi-Fermi level; F7).
_TRACK_VARS = ("Potential", "Electrons")


# ---- v0.15 (plan 4.4.1): the density-scaled default absolute tolerance -------------------------
# DEVSIM's Newton convergence test (devsim/devsim main, read 2026-11-04: Equation.cc DefaultUpdate,
# Region.cc / Device.cc Update, math/Newton.cc):
#   per equation   abs = max_nodes |dx|          rel = max_nodes |dx| / (|x| + 1e-10)
#   per region     the SUM over its equations (both)
#   per device     abs = MAX over regions,       rel = SUM over regions
#   converged      rel < relative_error  AND  abs < absolute_error, for every device (and the circuit)
# So the absolute error of a drift-diffusion region is in practice its largest DENSITY update (m^-3;
# a Poisson-only region's is in volts), and because the two tests are ANDed, raising absolute_error
# can never accept a state the relative test rejects -- it can only remove a test that cannot be met.
# Round-off floors max|dx| at kappa * u * S (S = solution_size below; kappa measured 2-4 ulp median,
# <= 36 ulp max on the GATE C accumulation, tests/test_dc_solve_abs_tol.py), so a fixed 1e10 m^-3 is
# below that floor once S > ~1e24 m^-3 and Newton then stalls at relative ~1e-15 "waiting for a lucky
# update". The default is therefore the larger of the historical 1e10 and ABS_TOL_ULPS unit round-offs
# of S: 1024 u = 512 ulp, 14x the worst measured floor.
ABS_TOL_FLOOR = 1.0e10
ABS_TOL_ULPS = 1024
UNIT_ROUNDOFF_DOUBLE = 2.0 ** -53
# boost cpp_bin_float_quad / __float128: 113-bit significand (the extended Newton's RESIDUAL precision;
# its relative updates fall to ~1e-33, extended_precision module docstring)
UNIT_ROUNDOFF_EXTENDED = 2.0 ** -113


def solution_size(device: "str | None" = None) -> float:
    """DEVSIM's aggregate solution size S: the max over regions of the SUM over that region's
    equations of max over nodes |x| (x = the equation's variable), evaluated on the CURRENT node
    values -- the same aggregation DEVSIM applies to the Newton updates (module comment above), so
    `ABS_TOL_ULPS * u * S` is the absolute error round-off alone produces. `device=None` takes the max
    over every live device: DEVSIM solves them all in ONE Newton and tests each against the same
    absolute_error. Units are mixed exactly as DEVSIM mixes them (m^-3 + V + ...)."""
    devices = list(ds.get_device_list()) if device is None else [device]
    size = 0.0
    for d in devices:
        for r in ds.get_region_list(device=d):
            s_r = 0.0
            for eq in ds.get_equation_list(device=d, region=r):
                var = ds.get_equation_command(device=d, region=r, name=eq)["variable_name"]
                vals = np.abs(np.asarray(ds.get_node_model_values(device=d, region=r, name=var),
                                         dtype=np.float64))
                if vals.size:
                    s_r += float(vals.max())
            size = max(size, s_r)
    return size


def default_abs_tol(device: "str | None" = None) -> float:
    """The default DC absolute tolerance `max(ABS_TOL_FLOOR, ABS_TOL_ULPS * u * S)` (plan 4.4.1,
    decision 58) -- the ONE implementation; `solve_dc(abs_tol=None)` resolves through it (with its
    `device`), on the PRE-solve state.

    u = 2^-53, or 2^-113 when any live device opted into extended precision (DEVSIM then runs the
    extended Newton for every live device). S = `solution_size(device)` (None: all live devices).

    WHEN TO USE WHICH:
      * leave abs_tol=None (this default) for any DC solve -- it equals the historical 1e10 exactly
        (the same float reaches DEVSIM, so results are byte-identical) whenever
        S < 1e10 / (1024 * 2^-53) = 8.8e22 m^-3, and on EVERY extended-precision device (up to
        S ~ 1e41); above that it is the round-off-attainable criterion, 512 ulp of S.
      * pass an explicit float to reproduce a historical run or to demand an absolute density
        criterion on purpose; it is passed to DEVSIM unchanged.
    What the default does NOT do: it never loosens the delivered accuracy -- the relative criterion
    (a MAX over nodes of |dx|/(|x| + 1e-10)) still binds every node, minority densities included
    (a 1e8 m^-3 hole density next to a 7.5e25 electron peak is still converged to rel_tol: gate
    tests/test_dc_solve_abs_tol.py). The relative criterion's own floor for variables near 0
    (|dx|/(|x| + 1e-10) with x ~ 0: density-gradient Lambda tails) is a separate unattainability that
    this does not address -- that one is what extended precision is for."""
    u = UNIT_ROUNDOFF_EXTENDED if XP.any_enabled() else UNIT_ROUNDOFF_DOUBLE
    return max(ABS_TOL_FLOOR, ABS_TOL_ULPS * u * solution_size(device))


def solve_dc(device: str, *, method: str = "newton",
              abs_tol: "float | None" = None, rel_tol: float = 1e-5, max_iter: int = 60,
              gummel_outer_max: int = 120, gummel_tol: float = 1e-5,
              gummel_inner_iter: int = 30,
              semiconductor_regions: Sequence[str] = (),
              verbose: bool = False) -> None:
    """One DC operating-point solve of every live DEVSIM device (DEVSIM's Newton is global).

    abs_tol: None (default, v0.15) resolves through `default_abs_tol(device)` on the pre-solve state
        -- identical to the old fixed 1e10 for S < 8.8e22 m^-3 and on extended-precision devices, the
        round-off-attainable 512 ulp of S above (see default_abs_tol for when to override). It sizes
        `device` only, so a light device keeps exactly 1e10 whatever else is live; when SEVERAL live
        devices are solved together on purpose, pass abs_tol=default_abs_tol() (all live devices). A
        float is passed to DEVSIM unchanged. The Gummel sub-solves use the same resolved value.
    rel_tol, max_iter: DEVSIM's relative_error / maximum_iterations, unchanged.
    method: "newton" (coupled, the default) or "gummel" (unipolar only; module docstring)."""
    if method == "newton":
        if abs_tol is None:
            abs_tol = default_abs_tol(device)
        XP.solve(type="dc", solver_type="direct", absolute_error=abs_tol,
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
            "method='newton' (default) remains the supported general route.", UnreliableResultWarning, stacklevel=2)
        if abs_tol is None:
            abs_tol = default_abs_tol(device)
        # No carrier equations present (e.g. equilibrium mode) -> Gummel is just
        # a Poisson solve; fall back to Newton.
        if not (set(CARRIER_EQS) & R.equation_names(device)):
            XP.solve(type="dc", solver_type="direct", absolute_error=abs_tol,
                      relative_error=rel_tol, maximum_iterations=max_iter)
            warn_if_solution_exceeds_fd_ceiling(device)         # audit C10-d (converged state)
            return
        # audit C-15 (2026-11-02): REFUSE a bipolar device instead of silently running a non-Gummel
        # splitting on it. CARRIER_EQS holds electrons only, so on a device that also carries
        # HoleContinuityEquation step (1) solved Poisson COUPLED to the hole equation (electrons
        # frozen), step (2) solved electrons+holes, and the convergence snapshot never looked at
        # Holes. Measured on a 200 nm abrupt Si pn diode (+-1e23 m^-3, Boltzmann, DEVSIM 2.10):
        # from the converged equilibrium, a 20 mV (and a 300 mV) forward step ran all 120 outer
        # passes without converging and left Potential/Electrons/Holes 31x/9.6x/0.90x (max rel)
        # away from the coupled-Newton answer. Implementing bipolar Gummel by ALSO freezing the hole
        # density was tried and is worse: the density-frozen (linear) Poisson step diverges on the
        # same diode at both biases (devsim "Convergence failure!", Holes off by 1.9e6x / 9.7e9x),
        # the classic failure that makes textbook Gummel freeze the QUASI-FERMI potentials and solve
        # a NONLINEAR Poisson (Gummel 1964; Selberherr, "Analysis and Simulation of Semiconductor
        # Devices", Sec. 7.2). That is a different solver (Slotboom/QFL variables) with no
        # validation in this repo, and the bipolar builders (devsim_3d, devsim_layered
        # _solve_bipolar) already hard-route to Newton, so the honest option is to refuse loudly.
        bipolar = sorted(set(_UNSUPPORTED_CARRIER_EQS) & R.equation_names(device))
        if bipolar:
            raise ValueError(
                "solve_dc(method='gummel'): device {!r} carries {} -- it is BIPOLAR, and the Gummel "
                "route is electron-only (it freezes {} only, so the 'Poisson' sub-solve would be a "
                "coupled Poisson+hole Newton and the convergence test would never see Holes; audit "
                "C-15). Use method='newton' (the coupled solve the bipolar builders already use)."
                .format(device, bipolar, list(CARRIER_EQS)))
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
            XP.solve(type="dc", solver_type="direct", absolute_error=abs_tol,
                      relative_error=rel_tol, maximum_iterations=inner_iter)
        # (2) Continuity sub-solve: freeze Potential
        with _frozen(device, (POTENTIAL_EQ,)):
            XP.solve(type="dc", solver_type="direct", absolute_error=abs_tol,
                      relative_error=rel_tol, maximum_iterations=inner_iter)

        delta = _max_rel_change(_snapshot(device, semi_regions), before)
        if verbose:
            print("[gummel]   outer {:3d}: max rel change = {:.3e}".format(outer, delta),
                   flush=True)
        if delta < outer_tol:
            return
    raise RuntimeError(
        "Gummel did not converge in {} outer iterations".format(outer_max))
