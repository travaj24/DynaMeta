"""Self-consistent (ASE-coupled) time step for the fiber-amplifier transient march
(dynamics.simulate_transient / simulate_transient_eryb, ase_mode="self_consistent").

WHAT THE QUASI-STATIC STEP DOES, AND WHERE IT BREAKS. The march's default step freezes the
populations, propagates every channel through the resulting gain exactly (dynamics._propagate_
fixed is the integrating-factor solution of dP/dz = u(g P + s) for a g that does not depend on
P), and then advances the populations with those powers. At the march's FIXED POINT that is
exact. Away from it the step is an EXPLICIT one in the ASE feedback: the ASE generated inside a
step never depletes the inversion that made it, so a population error is amplified by
exp(INT g dz) before it comes back into the update. Once that exponent is large the iteration is
unstable and the march settles somewhere other than the amplifier's own solve() -- the audit-A-7
regime, which dynamics measures every step and reports on meta['quasi_static_valid'].

WHY SUB-STEPPING AND A LITERAL "FROZEN-POPULATION BVP" BOTH FAIL. At genuinely fixed populations
the power problem is LINEAR and fully decoupled between directions: g(z) and s(z) are functions
of z alone, so _propagate_fixed is already its exact solution and a relaxation over it converges
in one sweep by construction. There is nothing to make self-consistent at frozen populations,
and a finer z or t grid returns the same over-amplified ASE. The coupling that is missing is the
one between the powers and the population CHANGE across the step.

WHAT THIS MODULE ADDS. Make the population change implicit and solve the resulting nonlinear
power problem to the steady solver's own tolerance. Write y for the population state at the
start of a step (the single-ion metastable fraction nbar2, or the co-doped (f2, b2) pair /
(f2, b2c, b2nc) triple), dt for the step, and

    advance(P)  = the EXISTING population update (the exponential integrator / exponential
                  Rosenbrock step) taken from y over dt with the rates read off the power
                  profile P -- unchanged algebra, called with a different P,
    gain_source = the EXISTING per-z gain and spontaneous-source assembly,
    propagate   = the EXISTING exact frozen-gain sweep.

The step solves the coupled pair

    P = propagate(gain_source(y_end)),        y_end = advance(P)                         (*)

for (P, y_end) simultaneously, by the same damped fixed-point iteration the steady solver uses
and against the same two residuals (steady_state._relaxation_residuals: endpoint AND interior
profile, each channel normalised by its own peak). The populations are then advanced with the
converged powers, which is exactly the existing step evaluated at a converged P.

WHY (*) IS STABLE AND HAS THE RIGHT LIMITS.

  * dt -> 0. advance(P) -> y for every P, so (*) collapses to one call of propagate at the
    frozen populations: the quasi-static step, recovered exactly rather than approximately.
  * dt -> infinity. The exponential integrator returns the root of its local balance (the
    single-ion update returns n2_ss = R_a/B exactly; the co-doped Rosenbrock step returns a
    Newton step onto the local root), so (*) becomes "the powers propagate through the gain the
    populations they sustain" -- the steady-state relaxation problem itself.
  * FIXED POINT. If y is the amplifier's steady state and P its steady profile, then advance
    returns y for ANY dt (the increment carries the rate right-hand side as a factor), so (*) is
    satisfied by the pair unchanged. The march's fixed point is the amplifier's own solve(), not
    a nearby one -- the property the existing marches already have, preserved.
  * STABILITY. Inside a step the gain now DEPLETES as the ASE grows: a population excursion that
    would raise INT g dz raises the ASE, the ASE raises the stimulated-emission rate in
    advance(), and y_end comes back down. That negative feedback is what the explicit step is
    missing, and it is why the iteration converges where the explicit march diverges.

WHAT IS NOT CHANGED. The REPORTED powers at time t_i remain the instantaneous frozen-population
propagation at the reported populations nbar2_zt[i] -- those two really are the exact
quasi-static pair at that instant, and keeping them means the result arrays, the resolved ASE
and frame_as_steady mean exactly what they always did. Only the population UPDATE reads the
self-consistent powers. So the two modes differ only through the population trajectory, and in
the regime where the quasi-static flag stays True they agree to the size of the ASE perturbation
itself (gated at < 1e-3 dB and < 1%), not to O(dt).

Pure numpy; no new physics constants. docs/audit/2026-09-15-march-self-consistent-ase.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Tuple

import numpy as np

# The steady solver's convergence test, reused verbatim rather than re-spelled: "the same
# tolerance as the steady solver" has to mean the same RESIDUALS as the steady solver, or the
# two tolerances are not comparable numbers (audit X-3's single-home rule).
from dynameta.optics.fiber_amp.steady_state import _relaxation_residuals

# The PUBLIC surface of this module -- and therefore (audit X-10's facade contract) what
# dynameta.optics.fiber_amp re-exports. The three names below are the ones a caller of
# simulate_transient can use: the valid ase_mode strings, and the two dataclasses that carry the
# inner solve's settings and its per-step report. Everything else here -- check_ase_mode,
# auto_switch, predicted_step_ase_error, needs_ase_probe, solve_self_consistent_step,
# residual_views -- is the march's internal seam, called by dynamics through the module object
# and gated directly in tests/test_march_self_consistent_ase.py; putting it in the package
# facade would advertise a stepping API that only dynamics is in a position to drive.
__all__ = ["ASE_MODES", "SelfConsistentControl", "StepReport"]

ASE_MODES = ("quasi_static", "self_consistent", "auto")

# Inner-iteration defaults. tol is the steady solver's own default (FiberAmplifier.solve(tol=
# 1e-6)); max_iter is below its 200 because a MARCH warm-starts every step from the previous
# one, so a converged march spends 2-4 iterations per step and only the first step after a
# switch costs more.
_DEFAULT_TOL = 1e-6
_DEFAULT_MAX_ITER = 120

# The damping ladder, and the reason it is the steady solver's (1.0, 0.5, 0.25) rather than a
# fixed number: undamped is 3-10x cheaper when it works and does not work on a strongly
# counter-coupled amplifier. Unlike the steady solver, the rung is CARRIED ACROSS STEPS -- a
# march takes thousands of steps on ONE amplifier, so re-discovering the rung from the top every
# step would pay the failed-rung cost every step.
_RELAX_LADDER = (1.0, 0.5, 0.25)

# "auto": switch BEFORE the quasi-static step leaves its regime, not after. The margins are the
# audit-A-7 ones (dynamics._ASE_TO_LAUNCHED_LIMIT / _GAIN_INTEGRAL_LIMIT); this is the fraction
# of them at which a step is handed to the self-consistent solver. Half the limit is one full
# e-fold of headroom on the gain-integral criterion (10 vs 20), and the A-7 sweep's valid and
# broken populations are still four decades apart there.
_AUTO_SWITCH_FRACTION = 0.5

# The PRIMARY, and the only PREDICTIVE, switch criterion: how much the step's OWN ASE would move
# if the gain were the one the step ends at rather than the one it starts at. ASE leaves the
# fiber as exp(INT g dz), so a step over which the ASE gain integral drifts by D mis-states its
# own ASE by the factor e^D - 1; multiplied by the ASE's share of the launched power that is the
# fraction of the LAUNCHED power the frozen step gets wrong. Switch above 1% of launched.
#
# Why this and not the two absolute margins alone: the absolute margins describe a step that is
# ALREADY in the runaway, and on a cold start every step looks healthy right up to the one that
# inverts the fiber -- measured on the reference co-doped amplifier at a uniform 1 ms grid, the
# absolute margins switch exactly ONE step and the march still lands 0.24 dB off, where this
# criterion switches the eight steps that matter and lands on 0.001 dB. The absolute margins are
# kept as a backstop for a march that starts inside the bad regime.
_AUTO_STEP_ASE_ERROR = 1.0e-2
_AUTO_DRIFT_CLIP = 50.0          # e^50 ~ 5e21; past this the answer is "switch" either way

# When the cheap e^D estimate above is not usable -- because the step STARTS with no ASE at all
# (a cold fiber emits nothing, so ase_ratio = 0 and the product is 0 whatever the gain does) --
# "auto" spends ONE extra frozen-gain propagation at the projected populations and compares the
# two ASE ratios exactly. The probe runs only when the projected gain integral is above this
# floor (some ASE channel is net-amplifying, so there is a runaway to have) and the integral
# actually MOVED across the step; on a march sitting at its operating point neither holds and
# the probe never runs.
_AUTO_PROBE_GI_FLOOR = 0.0
_AUTO_PROBE_DRIFT = 1.0e-3

# STICKINESS. The quasi-static instability is a property of the ITERATION, not of one step: the
# error a frozen step injects is amplified by the NEXT few steps, and by the time the monitor can
# see it on local quantities the march is already ringing. So a trip arms a window of steps that
# are taken self-consistently whatever the monitor says, re-armed on every trip. MEASURED on the
# reference co-doped amplifier at a uniform 1 ms grid: with no window the march still rings at a
# GROWING +-0.3 dB and auto lands 0.011 dB off the fully self-consistent answer; with this window
# it tracks it to below 1e-3 dB. A march at its operating point never trips, so the window costs
# nothing there.
_AUTO_STICKY_STEPS = 8


@dataclass
class SelfConsistentControl:
    """Inner-solve settings, and the mutable state that makes the inner solve cheap.

    tol / max_iter are the inner fixed-point iteration's convergence threshold and cap.
    relax_ladder is walked only when a rung fails; `rung` is the entry that last WORKED and is
    where the next step starts, so the ladder is climbed at most once per march rather than once
    per step. `seed` is the previous step's converged power profile -- the warm start, which is
    what turns a 10-30 iteration cold solve into a 2-4 iteration one.
    """
    tol: float = _DEFAULT_TOL
    max_iter: int = _DEFAULT_MAX_ITER
    relax_ladder: Tuple[float, ...] = _RELAX_LADDER
    switch_fraction: float = _AUTO_SWITCH_FRACTION
    step_error: float = _AUTO_STEP_ASE_ERROR
    sticky_steps: int = _AUTO_STICKY_STEPS
    rung: int = 0
    seed: Optional[np.ndarray] = field(default=None, repr=False)

    # ---- running diagnostics, reported on TransientResult.meta ----
    sticky_left: int = 0
    n_steps: int = 0                 # steps taken self-consistently
    n_probes: int = 0                # extra propagations spent measuring the projected ASE
    n_iter_total: int = 0            # inner iterations summed over those steps
    max_iter_used: int = 0
    max_profile_residual: float = 0.0
    n_nonconverged: int = 0

    def arm(self):
        """A genuine trip: this step AND the next `sticky_steps` are taken self-consistently."""
        self.sticky_left = int(self.sticky_steps)

    def sticky(self) -> bool:
        """Consume one step of an armed window. False once the window is spent."""
        if self.sticky_left > 0:
            self.sticky_left -= 1
            return True
        return False

    def record(self, rep: "StepReport"):
        self.n_steps += 1
        self.n_iter_total += rep.iterations
        self.max_iter_used = max(self.max_iter_used, rep.iterations)
        if np.isfinite(rep.profile_residual):
            self.max_profile_residual = max(self.max_profile_residual, rep.profile_residual)
        if not rep.converged:
            self.n_nonconverged += 1


@dataclass
class StepReport:
    """What one self-consistent step did."""
    converged: bool
    iterations: int
    endpoint_residual: float
    profile_residual: float
    relax: float


def check_ase_mode(mode) -> str:
    """Validate and return the ase_mode string. Refuses by NAME rather than letting a typo take
    the default path silently -- a mis-spelled "selfconsistent" that quietly ran the quasi-static
    march is exactly the silent-wrong failure this option exists to remove."""
    if mode not in ASE_MODES:
        raise ValueError("simulate_transient: ase_mode must be one of %s; got %r"
                         % (", ".join(repr(m) for m in ASE_MODES), mode))
    return str(mode)


def residual_views(P, fwd, bwd):
    """(endpoint vector, full profile) in the steady solver's channel ordering [fwd..., bwd...],
    each channel read at ITS OWN output end. _relaxation_residuals requires that ordering so the
    per-channel peak lines up with the per-channel endpoint."""
    if bwd.size:
        out = np.concatenate([P[fwd, -1], P[bwd, 0]])
        prof = np.concatenate([P[fwd], P[bwd]], axis=0)
    else:
        out = P[fwd, -1].copy()
        prof = P[fwd].copy()
    return out, prof


def solve_self_consistent_step(propagate: Callable, advance: Callable, gain_source: Callable,
                               fwd: np.ndarray, bwd: np.ndarray,
                               ctrl: SelfConsistentControl,
                               p_frozen: np.ndarray) -> Tuple[np.ndarray, StepReport]:
    """Solve  P = propagate(gain_source(advance(P)))  for ONE time step, and return (P, report).

    propagate(g, s) -> (K, Nz) powers for a per-z gain g and spontaneous source s, with the
        step's boundary powers and directions already bound (dynamics._propagate_fixed).
    advance(P)      -> the population state at the END of the step, from the powers P. This is
        the march's OWN update (exponential integrator / exponential Rosenbrock), so nothing
        about the population algebra is duplicated here.
    gain_source(y)  -> (g, s) for a population state y; the march's own assembly.
    fwd / bwd       -> channel index arrays, for the residual's channel ordering.
    p_frozen        -> the frozen-population propagation of this step, used as the seed when no
        warm start is available (the first switched step) and as the fallback if the iteration
        produces non-finite powers at every rung.

    The caller advances the populations itself, with the returned P, so that the population
    update and its diagnostics stay in one place in dynamics.
    """
    seed = ctrl.seed
    if seed is None or seed.shape != p_frozen.shape or not np.all(np.isfinite(seed)):
        seed = p_frozen
    best_P = None
    best_rep = None
    for rung in range(int(ctrl.rung), len(ctrl.relax_ladder)):
        relax = float(ctrl.relax_ladder[rung])
        P = seed
        last_out = None
        last_prof = None
        end_r = float("inf")
        prof_r = float("inf")
        converged = False
        used = 0
        for it in range(int(ctrl.max_iter)):
            y = advance(P)
            g, s = gain_source(y)
            P_new = propagate(g, s)
            if not np.all(np.isfinite(P_new)):
                break
            # relax == 1.0 skips the blend entirely (the steady solver's rule): a
            # "1.0 * new + 0.0 * old" blend is NOT a no-op, because 0.0 * inf is NaN.
            P = P_new if relax == 1.0 else relax * P_new + (1.0 - relax) * P
            used = it + 1
            out, prof = residual_views(P, fwd, bwd)
            if last_out is not None:
                end_r, prof_r = _relaxation_residuals(out, last_out, prof, last_prof)
                if end_r < ctrl.tol and prof_r < ctrl.tol:
                    converged = True
                    break
            last_out = out
            last_prof = prof
        rep = StepReport(converged, used, float(end_r), float(prof_r), relax)
        if converged:
            # LATCH the rung that worked: the next step starts here rather than re-paying the
            # failed rungs. Downward only -- a march that needed damping once needs it again,
            # and climbing back up would re-pay the failure on every step that tried.
            ctrl.rung = rung
            ctrl.seed = P
            return P, rep
        better = best_rep is None or (np.isfinite(prof_r)
                                      and not (np.isfinite(best_rep.profile_residual)
                                               and best_rep.profile_residual <= prof_r))
        if better:
            best_P = P
            best_rep = rep
    if best_P is None or not np.all(np.isfinite(best_P)):
        best_P = p_frozen
        best_rep = StepReport(False, 0, float("inf"), float("inf"),
                              float(ctrl.relax_ladder[min(int(ctrl.rung),
                                                          len(ctrl.relax_ladder) - 1)]))
    ctrl.rung = len(ctrl.relax_ladder) - 1
    ctrl.seed = best_P
    return best_P, best_rep


def predicted_step_ase_error(ase_ratio: float, gain_integral: float,
                             projected_gain_integral: float) -> float:
    """The fraction of the LAUNCHED power this step's frozen-gain ASE is expected to mis-state:
    the ASE's own share of the launched power times |e^D - 1|, D the drift of the ASE gain
    integral across the step. Exactly 0 for a step that does not move the gain (D = 0) or that
    carries no ASE, which is what makes it safe to compare against an absolute threshold."""
    if not (np.isfinite(ase_ratio) and np.isfinite(gain_integral)
            and np.isfinite(projected_gain_integral)):
        return float("inf")
    drift = float(projected_gain_integral) - float(gain_integral)
    drift = min(max(drift, -_AUTO_DRIFT_CLIP), _AUTO_DRIFT_CLIP)
    return max(float(ase_ratio), 0.0) * abs(float(np.expm1(drift)))


def needs_ase_probe(gain_integral: float, projected_gain_integral: float) -> bool:
    """Is it worth ONE extra frozen-gain propagation to measure this step's ASE drift exactly
    instead of estimating it? Only when some ASE channel is net-amplifying at the projected
    populations and the gain integral actually moved -- i.e. never on a march sitting still, and
    always on the step that first inverts a cold fiber (where the cheap estimate is identically
    zero because the fiber was not emitting yet)."""
    if not (np.isfinite(gain_integral) and np.isfinite(projected_gain_integral)):
        return True
    return (projected_gain_integral > _AUTO_PROBE_GI_FLOOR
            and abs(projected_gain_integral - gain_integral) > _AUTO_PROBE_DRIFT)


def auto_switch(ase_ratio: float, gain_integral: float, projected_gain_integral: float,
                nonfinite: bool, limits: dict, fraction: float,
                step_error: float = _AUTO_STEP_ASE_ERROR,
                probed_ase_ratio: Optional[float] = None) -> bool:
    """Should THIS step be taken self-consistently?

    Three independent tests, any of which switches the step:
      * PREDICTIVE (estimate) -- predicted_step_ase_error above, against `step_error` (default 1%
        of the launched power).
      * PREDICTIVE (probe) -- if the caller measured the ASE ratio at the projected populations
        (needs_ase_probe said it was worth a propagation), the EXACT change against the same
        threshold. This is the one that fires on the step that inverts a cold fiber.
      * BACKSTOP -- the two audit-A-7 margins, each against `fraction` of its documented limit,
        evaluated both at the step's own populations and at the ones the explicit update would
        reach. This covers a march that STARTS inside the bad regime, where there is no drift to
        detect because nothing is changing.
    """
    if nonfinite:
        return True
    if not (np.isfinite(ase_ratio) and np.isfinite(gain_integral)
            and np.isfinite(projected_gain_integral)):
        return True
    if predicted_step_ase_error(ase_ratio, gain_integral, projected_gain_integral) > step_error:
        return True
    if probed_ase_ratio is not None:
        if not np.isfinite(probed_ase_ratio):
            return True
        if abs(float(probed_ase_ratio) - float(ase_ratio)) > step_error:
            return True
    if ase_ratio > fraction * limits["ase_to_launched"]:
        return True
    gi = max(float(gain_integral), float(projected_gain_integral))
    return gi > fraction * limits["ase_gain_integral"]
