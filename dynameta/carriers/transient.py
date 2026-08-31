"""
Large-signal TRANSIENT (time-domain) carrier dynamics via DEVSIM's BDF time integration -- the
companion to the small-signal ac_analysis. After a DC operating point, change a circuit-driven
contact's bias and integrate the device forward in time, recording the terminal current I(t) (the
modulator turn-on / turn-off waveform, reverse-recovery, charge storage, ...).

Prerequisites (same as ssac): a transient-ready region -- the d(q n)/dt charge time-node models
that physics_bipolar_dd.setup_bipolar_region defines (NCharge/PCharge) -- and a circuit-driven
contact (physics_bipolar_dd.setup_contact_ohmic_bipolar_circuit). Requires DEVSIM.

Adaptive stepping: a LARGE charge_error (accept the step; DEVSIM's tight LTE control would reject
the first sub-step of a discontinuous bias change) + a robust user-side controller -- grow dt on a
successful step, and on a Newton convergence failure retry with the step size moved in whichever
direction that failure's MECHANISM calls for (the two classes below). dt0 must NOT be
<< the device's dielectric/RC time: a tiny dt makes the charge/dt term dominate and ill-conditions
the Jacobian (a too-small dt0 stalls the solve). The trajectory ACCURACY is bounded only by the dt
cap here; the integrator's correctness is validated by the transient RELAXING to the independent DC
solution at the final bias (validation/transient_diode.py settles to ~1e-5).

TWO failure classes, TWO step responses (nightly run 33347138729, 2026-08-31, found the hard way).
A Newton failure here is NOT always "the step is too large":

  (1) STEP TOO LARGE. The device equations themselves are far from converged -- the bias moved
      further in one dt than Newton can follow. Halving dt is the cure, as always.

  (2) TERMINAL-CURRENT ROUND-OFF STALL. The device equations converge to machine precision in ONE
      iteration, and the ONLY unmet test is the CIRCUIT (terminal-current) relative error. This is
      what a decaying transient does to a circuit-driven contact: the terminal current is the near
      cancellation of the carrier and displacement fluxes at the contact, so once I(t) has decayed
      toward the settled DC level its RELATIVE round-off floor is ~ eps_mach*Q_contact/(dt*|I|) --
      it scales as 1/dt. On the validation diode's reverse step that floor passes ABOVE the 1e-6
      exit tolerance while dt is still ~1e-11 s, Newton then random-walks on round-off noise until a
      fluctuation happens to dip under tolerance, and whether it does so inside maximum_iterations
      is a coin flip decided by the last bits of the arithmetic (hence: passes on one CI runner,
      fails on the next, with NO code change). Halving dt makes this class STRICTLY WORSE -- the
      floor goes as 1/dt -- so the plain halve-and-retry loop cascades all the way to the dt floor
      and kills the run mid-transient. The cure is the opposite move: GROW dt until the terminal
      current's round-off floor drops back under the exit tolerance.

DEVSIM's solve(info=True) returns the per-equation errors instead of raising, which is what lets the
two classes be told apart here rather than guessed at.
"""

from __future__ import annotations

import numpy as np
import devsim as ds


def _device_is_converged(info: dict, relative_error: float, absolute_error: float) -> bool:
    """True if, at the LAST Newton iteration of a failed solve, every DEVICE equation met both
    tolerances -- i.e. the semiconductor physics is solved and only the circuit (terminal-current)
    equation held the solve back. That is the round-off-stall signature of class (2) above; a
    genuinely too-large step (class (1)) leaves the device equations far from converged."""
    its = info.get("iterations") or ()
    if not its:
        return False                                  # no per-iteration data -> assume class (1)
    for dev in its[-1].get("devices", ()):
        for reg in dev.get("regions", ()):
            for eq in reg.get("equations", ()):
                # DEVSIM's per-equation test is AND, not OR (verified against a forced failure:
                # abs=6.6e8 < absolute_error=1e18 but rel=7.6e-3 > 1e-6 is reported NOT converged).
                if not (eq["relative_error"] < relative_error
                        and eq["absolute_error"] < absolute_error):
                    return False
    return True


def transient_step(v_to: float, *, t_end: float, dt0: float = 1.0e-14,
                   dt_growth: float = 1.3, dt_cap_frac: float = 20.0,
                   source_name: str = "V1", charge_error: float = 1.0e30,
                   max_steps: int = 2000, absolute_error: float = 1.0e18,
                   relative_error: float = 1.0e-6, maximum_iterations: int = 40,
                   dt_stall_growth: float = 4.0, max_stall_growths: int = 8):
    """Change circuit source `source_name` to `v_to` and integrate the device forward to `t_end`
    (s) with adaptive BDF1. The device must already be at a DC operating point (solved) and
    transient-ready. Returns (t_s, I): arrays of time (s) and terminal current I = `source_name`.I
    (A; A/m^2 in 1-D) at each accepted step.

    dt grows by `dt_growth` per accepted step (capped at t_end/`dt_cap_frac`). On a Newton
    convergence failure the response depends on WHICH equations failed (see the module docstring):
    a failure with the device equations still unconverged HALVES dt (floor 1e-19 s -> RuntimeError);
    a failure in which only the CIRCUIT (terminal-current) relative test is unmet is a round-off
    stall whose floor scales as 1/dt, so dt is GROWN by `dt_stall_growth` (up to `max_stall_growths`
    times per step, never past the dt cap) instead -- halving that class provably makes it worse.
    `charge_error` is left large so the discontinuous bias change is accepted; pass a finite value
    for DEVSIM's native LTE step control (then ramp the bias over a finite rise time rather than
    stepping it, or the first sub-step is rejected). Set the prior DC operating point with
    ds.circuit_alter(name=source_name, value=...) + a dc solve before calling."""
    if t_end <= 0.0 or dt0 <= 0.0:
        raise ValueError("t_end and dt0 must be > 0")
    if dt_growth <= 1.0 or dt_cap_frac <= 0.0:
        raise ValueError("dt_growth must be > 1 and dt_cap_frac > 0 (a non-growing or "
                         "negative cap gives a stuck or backward-in-time step); got "
                         "dt_growth={}, dt_cap_frac={}".format(dt_growth, dt_cap_frac))
    if dt_stall_growth <= 1.0 or max_stall_growths < 0:
        raise ValueError("dt_stall_growth must be > 1 and max_stall_growths >= 0 (the round-off "
                         "stall is escaped by GROWING dt); got dt_stall_growth={}, "
                         "max_stall_growths={}".format(dt_stall_growth, max_stall_growths))
    # establish the transient initial condition at the current DC state
    ds.solve(type="transient_dc", absolute_error=absolute_error, relative_error=relative_error,
             maximum_iterations=maximum_iterations)
    ds.circuit_alter(name=source_name, value=float(v_to))            # the bias change
    src_i = "{}.I".format(source_name)
    ts, Is = [], []
    t, dt = 0.0, float(dt0)
    cap = float(t_end) / float(dt_cap_frac)
    steps = 0
    grown = 0                       # stall-escape growths spent on the CURRENT step (reset on accept)
    while t < t_end and steps < max_steps:
        dt = min(dt, t_end - t)         # clamp the final step so it lands ON t_end (audit: was ~5% over)
        stalled = False
        try:
            # info=True returns the per-equation errors and reports non-convergence as a RETURN
            # value; only hard solver errors (singular factorization, exp overflow) still raise.
            info = ds.solve(type="transient_bdf1", tdelta=dt, charge_error=charge_error,
                            absolute_error=absolute_error, relative_error=relative_error,
                            maximum_iterations=maximum_iterations, info=True)
            if isinstance(info, dict) and not info.get("converged", True):
                # Class (2) -- the device physics IS solved and only the circuit's terminal-current
                # relative test is unmet: a round-off floor ~ eps_mach*Q_contact/(dt*|I|) that
                # HALVING dt would only push further above tolerance. Grow out of it instead. Once
                # dt is large enough the floor stays under tolerance for the rest of the run, so
                # this fires a couple of times at the crossover and never again.
                # `dt_try > dt` keeps the last step honest: once dt is clamped to (t_end - t) a
                # "growth" would be clamped straight back and re-run the identical failing solve.
                dt_try = min(dt * float(dt_stall_growth), cap, t_end - t)
                if (grown < int(max_stall_growths) and dt_try > dt
                        and _device_is_converged(info, relative_error, absolute_error)):
                    dt = dt_try
                    grown += 1
                    continue
                stalled = True                                       # fall through to halving
        except ds.error as msg:                                      # DEVSIM solver failure
            # Halve + retry on the RECOVERABLE, step-size-related failures: Newton
            # non-convergence, a singular/ill-conditioned factorization, or an exp overflow --
            # all eased by a smaller tdelta. Re-raise anything else (a genuine setup error must
            # not be silently retried into the dt floor).
            s = str(msg).lower()                                     # case-robust + full words (was the
            # over-broad fragments "onvergence"/"teration"/"verflow" that could match unrelated text)
            if not any(k in s for k in ("convergence", "factoriz", "overflow", "iteration")):
                raise
            stalled = True
        if stalled:
            # Class (1) -- the device equations are genuinely unconverged (or the stall survived
            # every growth): the step really is too large, so halve as before. Spend the rest of
            # this step's growth budget so a halved retry cannot bounce straight back up.
            grown = int(max_stall_growths)
            dt *= 0.5
            if dt < 1.0e-19:
                raise RuntimeError(
                    "transient_step: stalled at t={:.3e} s (dt floor) -- either the bias change is "
                    "too abrupt / the operating point too stiff (try a smaller initial step or ramp "
                    "the bias over a finite rise time), or the terminal current has decayed so far "
                    "that its round-off floor sits above relative_error={:.1e} and even "
                    "{} growths of dt could not clear it (raise max_stall_growths / "
                    "dt_cap_frac, or relax relative_error).".format(
                        t, relative_error, max_stall_growths))
            continue
        t += dt
        steps += 1
        grown = 0
        ts.append(t)
        Is.append(float(ds.get_circuit_node_value(node=src_i, solution="dcop")))
        dt = min(dt * float(dt_growth), cap)
    if t < t_end:                                                    # anti-silent-failure
        raise RuntimeError(
            "transient_step: hit max_steps={} at t={:.3e} s < t_end={:.3e} s -- the returned "
            "waveform is INCOMPLETE (it never reached t_end). Raise max_steps, increase dt0 / "
            "dt_cap_frac, or soften the bias step.".format(max_steps, t, t_end))
    return np.asarray(ts, dtype=np.float64), np.asarray(Is, dtype=np.float64)
