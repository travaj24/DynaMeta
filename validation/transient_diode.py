"""
Validate large-signal TRANSIENT carrier dynamics (carriers.transient.transient_step) on the same
drift-diffusion p-n diode as validation/ac_diode. After a DC operating point, apply a bias STEP and
integrate the device forward in time; the rigorous correctness check is that the transient RELAXES
to the INDEPENDENT DC solution at the final bias (the time integrator must be consistent with the
steady-state solver) -- this is tight and not limited by spike-integration error.

GATE A (settling): a reverse step (-0.5 -> -1.0 V) and a forward step (0.0 -> +0.5 V) each relax to
        the independently DC-solved terminal current at the final bias, to < 1e-3 relative.
GATE B (a real transient happened): the reverse step shows a decaying CAPACITIVE current spike
        (peak |I| during the transient >> the settled |I|), i.e. the response is dynamic, not
        instantaneous.
INFO   : the reverse step's integrated charge INT I dt vs the analytic depletion-charge change dQ
         (a physical cross-check; ~20% here -- trapezoid over the stiff turn-on spike, NOT physics:
         the ssac junction C matches the same depletion model to ~1%).

Reuses the diode build/solve from validation.ac_diode. Run: python -m validation.transient_diode
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dynameta.carriers import transient as TR
from validation.ac_diode import (build, staged_equilibrium_solve, ramp_to, depletion_cap,
                                  _solve, VBI)

REV_FROM, REV_TO = -0.5, -1.0
FWD_FROM, FWD_TO = 0.0, 0.5
T_END = 1.0e-7
SETTLE_RTOL = 1.0e-3


def _terminal_current():
    import devsim as ds
    return float(ds.get_circuit_node_value(node="V1.I", solution="dcop"))


def _settle_test(v_from, v_to, label):
    """Ramp to v_from (DC), step to v_to (transient), then compare the settled transient current to
    the independent DC solve at v_to. Returns (settle_rel, t_s, I)."""
    import devsim as ds
    ramp_to(v_from)
    _solve()
    t_s, I = TR.transient_step(v_to, t_end=T_END, source_name="V1")
    i_trans_end = float(I[-1])
    ds.circuit_alter(name="V1", value=v_to)        # independent DC at the final bias
    _solve()
    i_dc = _terminal_current()
    settle_rel = abs(i_trans_end - i_dc) / max(abs(i_dc), 1e-30)
    print("[t] {}: {} steps  I_trans(end)={:+.4e}  I_DC={:+.4e}  settle_rel={:.2e}".format(
        label, t_s.size, i_trans_end, i_dc, settle_rel), flush=True)
    return settle_rel, t_s, I


def main():
    print("[t] === Large-signal TRANSIENT on a drift-diffusion p-n diode (settles to DC) ===",
          flush=True)
    print("[t] Vbi={:.4f} V; p-contact circuit-driven (source V1); BDF1 adaptive stepping".format(
        VBI), flush=True)
    build()
    staged_equilibrium_solve()

    rev_rel, t_rev, I_rev = _settle_test(REV_FROM, REV_TO, "reverse step -0.5->-1.0")
    # a real (decaying capacitive) transient: the peak current dwarfs the settled value
    peak_over_settled = float(np.max(np.abs(I_rev)) / max(abs(I_rev[-1]), 1e-30))
    transient_happened = bool(peak_over_settled > 100.0)
    # INFO: charge conservation vs the analytic depletion-charge change (integration-limited)
    dQ_num = float(np.sum(0.5 * (np.abs(I_rev[1:]) + np.abs(I_rev[:-1])) * np.diff(t_rev)))
    Vgrid = np.linspace(REV_FROM, REV_TO, 200)
    dQ_anal = abs(float(np.sum(0.5 * (np.array([depletion_cap(v) for v in Vgrid])[1:]
                                      + np.array([depletion_cap(v) for v in Vgrid])[:-1])
                               * np.diff(Vgrid))))
    print("[t]   reverse transient: peak|I|/settled={:.2e} (decaying spike); INT|I|dt={:.3e} vs "
          "depletion dQ={:.3e} C/m^2 (ratio {:.2f}, integration-limited)".format(
              peak_over_settled, dQ_num, dQ_anal, dQ_num / dQ_anal), flush=True)

    fwd_rel, t_fwd, I_fwd = _settle_test(FWD_FROM, FWD_TO, "forward step  0.0->+0.5")

    settles = bool(rev_rel < SETTLE_RTOL and fwd_rel < SETTLE_RTOL)
    gate_a = settles
    gate_b = transient_happened
    overall = gate_a and gate_b
    print("[t]", flush=True)
    print("[t] GATE A (reverse & forward transients settle to DC, rel<{:.0e}): {}".format(
        SETTLE_RTOL, "PASS" if gate_a else "FAIL"), flush=True)
    print("[t] GATE B (a real decaying capacitive transient occurred): {}".format(
        "PASS" if gate_b else "FAIL"), flush=True)
    print("[t] *** TRANSIENT DRIFT-DIFFUSION DIODE: {} ***".format("PASS" if overall else "FAIL"),
          flush=True)
    return overall


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
