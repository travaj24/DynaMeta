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
successful step, HALVE and retry on a Newton convergence failure (down to a floor). dt0 must NOT be
<< the device's dielectric/RC time: a tiny dt makes the charge/dt term dominate and ill-conditions
the Jacobian (a too-small dt0 stalls the solve). The trajectory ACCURACY is bounded only by the dt
cap here; the integrator's correctness is validated by the transient RELAXING to the independent DC
solution at the final bias (validation/transient_diode.py settles to ~1e-6).
"""

from __future__ import annotations

import numpy as np
import devsim as ds


def transient_step(v_to: float, *, t_end: float, dt0: float = 1.0e-14,
                   dt_growth: float = 1.3, dt_cap_frac: float = 20.0,
                   source_name: str = "V1", charge_error: float = 1.0e30,
                   max_steps: int = 2000, absolute_error: float = 1.0e18,
                   relative_error: float = 1.0e-6, maximum_iterations: int = 40):
    """Change circuit source `source_name` to `v_to` and integrate the device forward to `t_end`
    (s) with adaptive BDF1. The device must already be at a DC operating point (solved) and
    transient-ready. Returns (t_s, I): arrays of time (s) and terminal current I = `source_name`.I
    (A; A/m^2 in 1-D) at each accepted step.

    dt grows by `dt_growth` per accepted step (capped at t_end/`dt_cap_frac`) and HALVES on a Newton
    convergence failure (floor 1e-19 s -> RuntimeError). `charge_error` is left large so the
    discontinuous bias change is accepted; pass a finite value for DEVSIM's native LTE step control
    (then ramp the bias over a finite rise time rather than stepping it, or the first sub-step is
    rejected). Set the prior DC operating point with ds.circuit_alter(name=source_name, value=...)
    + a dc solve before calling."""
    if t_end <= 0.0 or dt0 <= 0.0:
        raise ValueError("t_end and dt0 must be > 0")
    # establish the transient initial condition at the current DC state
    ds.solve(type="transient_dc", absolute_error=absolute_error, relative_error=relative_error,
             maximum_iterations=maximum_iterations)
    ds.circuit_alter(name=source_name, value=float(v_to))            # the bias change
    src_i = "{}.I".format(source_name)
    ts, Is = [], []
    t, dt = 0.0, float(dt0)
    cap = float(t_end) / float(dt_cap_frac)
    steps = 0
    while t < t_end and steps < max_steps:
        try:
            ds.solve(type="transient_bdf1", tdelta=dt, charge_error=charge_error,
                     absolute_error=absolute_error, relative_error=relative_error,
                     maximum_iterations=maximum_iterations)
        except ds.error as msg:                                      # DEVSIM convergence failure
            if "onvergence" not in str(msg):
                raise
            dt *= 0.5
            if dt < 1.0e-19:
                raise RuntimeError(
                    "transient_step: stalled at t={:.3e} s (dt floor) -- the bias change may be too "
                    "abrupt or the operating point too stiff. Try a smaller initial step or ramp "
                    "the bias over a finite rise time.".format(t))
            continue
        t += dt
        steps += 1
        ts.append(t)
        Is.append(float(ds.get_circuit_node_value(node=src_i, solution="dcop")))
        dt = min(dt * float(dt_growth), cap)
    return np.asarray(ts, dtype=np.float64), np.asarray(Is, dtype=np.float64)
