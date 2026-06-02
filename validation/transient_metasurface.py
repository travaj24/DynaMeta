"""
Large-signal TRANSIENT (time-domain) gate response of the FULL Park metasurface (2D drift-diffusion)
-- the large-signal companion to validation/resolved_bandwidth_metasurface.py (the small-signal ssac
C(omega)), completing the carrier-dynamics arc on the real geometry.

Builds the Park DD metasurface (LayeredDevsimBuilder), records the independent DC ITO charge at the
target gate bias and at the operating point (via the builder's proven bias ramp), repoints the gate
(top_contact) CIRCUIT-DRIVEN via the first-class builder API (LayeredDevsimBuilder.set_ssac_gate),
then STEPS the gate bias and integrates the device forward in time (transient.transient_step,
adaptive BDF1), recording the terminal-current waveform I(t).

ORACLE (settling -- integration-noise-free, the same one validation/transient_diode.py uses): a
correct time integrator RELAXES to the steady state regardless of path. So the ITO accumulated charge
at the END of the transient (launched from Vg0) must equal the INDEPENDENTLY-computed DC value at the
target bias Vg1, where Q = q*sum(NodeVolume*(n - n_bg)) over the ITO (2D, C/m per unit y). The
independent DC target is computed by the robust builder bias ramp BEFORE the gate is repointed
circuit-driven -- a bias-Dirichlet and a circuit-Dirichlet impose the SAME Potential=Vg, so the DC
steady states coincide.

GATE A: the waveform completes (reaches t_end) and I(t) is finite -- transient_step RAISES otherwise,
        so reaching this point already proves it; we also assert I is finite and the device moved.
GATE B: the end-of-transient ITO charge == the independent DC charge at Vg1, within SETTLE_RTOL of
        the full Vg0->Vg1 charge swing (the transient settled onto the DC target).
INFO  : the terminal current decays from its turn-on transient toward the DC gate leakage (~0 for a
        dielectric gate); and INT I dt vs the gate charge swing (charge conservation -- a trapezoid
        over the stiff turn-on spike, informational only).

Run: python -m validation.transient_metasurface
"""

import contextlib
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import devsim as ds

from examples.park_2021 import build_park_design
from dynameta.carriers.devsim_layered import LayeredDevsimBuilder
from dynameta.carriers import transient as TR
from dynameta.sweep import BiasPoint
from dynameta.core.numerics import trapz as _trapz
from dynameta.constants import Q_E

GATE = "top_contact"
VG0, VG1 = 0.0, 1.0
# Integration horizon: the gate charges through the ITO sheet resistance laterally from the edge
# grounds -- a distributed RC line with settling time tau ~ rho_s * C_area * L^2 ~ 0.3 ps (L ~ half
# the period). t_end = 1e-11 s (~37 tau) settles the ITO charge to ~1e-8 of the swing. Do NOT
# over-extend t_end: once the device is fully settled, the adaptive dt grows to its cap and BDF1
# stalls trying to step a dead-flat state (the relative-update floor on a trivial step) -- pick
# t_end a few tens of tau, not orders beyond it.
DT0, T_END = 1.0e-14, 1.0e-11        # adaptive-BDF1 initial step / integration horizon (s)
SETTLE_RTOL = 2.0e-3                  # |q_end - q_dc(Vg1)| must be < this * |q_dc(Vg1) - q_dc(Vg0)|


@contextlib.contextmanager
def _quiet():
    sys.stdout.flush(); saved = os.dup(1); devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, 1); yield
    finally:
        sys.stdout.flush(); os.dup2(saved, 1); os.close(devnull); os.close(saved)


def _ito_charge_per_y(dev, n_bg):
    """Q = q * sum(NodeVolume * (n - n_bg)) over the ITO region [C/m per unit y] (2D)."""
    nv = np.array(ds.get_node_model_values(device=dev, region="ito", name="NodeVolume"),
                  dtype=np.float64)
    n = np.array(ds.get_node_model_values(device=dev, region="ito", name="Electrons"),
                 dtype=np.float64)
    return Q_E * float(np.sum(nv * (n - n_bg)))


def main():
    print("[t] === Large-signal transient gate step on the Park metasurface (2D DD) ===", flush=True)
    d = build_park_design("drift_diffusion")
    b = LayeredDevsimBuilder(d, mesh_name="tms_m", device_name="tms_d")
    n_bg = float(d.materials.get("ITO").transport.n_bg_m3)

    # independent DC targets via the builder's proven bias ramp (before repointing the gate)
    with _quiet():
        b.solve(BiasPoint({GATE: VG1}, "g{:+.0f}".format(VG1)))
    q_dc1 = _ito_charge_per_y(b.device, n_bg)
    with _quiet():
        b.solve(BiasPoint({GATE: VG0}, "g{:+.0f}".format(VG0)))   # operating point
    q_dc0 = _ito_charge_per_y(b.device, n_bg)
    swing = q_dc1 - q_dc0
    print("[t] DC ITO charge: q(Vg={:.0f})={:.5e}  q(Vg={:.0f})={:.5e} C/m  swing={:.5e}".format(
        VG0, q_dc0, VG1, q_dc1, swing), flush=True)

    # repoint the gate circuit-driven (first-class API) and integrate the Vg0 -> Vg1 step
    b.set_ssac_gate(GATE, source_name="V1")
    with _quiet():
        t, I = TR.transient_step(VG1, t_end=T_END, dt0=DT0, source_name="V1")
    q_end = _ito_charge_per_y(b.device, n_bg)

    settle = abs(q_end - q_dc1) / max(abs(swing), 1e-30)
    int_I = float(_trapz(I, t))                                   # INT I dt over the waveform
    cc = abs(int_I) / max(abs(swing), 1e-30)                      # charge-conservation ratio (info)
    i_finite = bool(np.all(np.isfinite(I))) and bool(np.isfinite(q_end))
    moved = bool(abs(q_end - q_dc0) > 0.1 * abs(swing))           # the transient actually charged

    print("[t] transient: {} steps to t_end={:.1e} s ; I0={:+.3e} -> I_end={:+.3e} A/m".format(
        t.size, T_END, I[0], I[-1]), flush=True)
    print("[t]   end-of-transient ITO charge q_end={:.5e} C/m ; settle (|q_end-q_dc1|/|swing|)="
          "{:.3e}".format(q_end, settle), flush=True)
    print("[t]   INT I dt={:.3e} C/m ; charge-conservation INT/swing={:.3f} (info; trapezoid over "
          "the turn-on spike)".format(int_I, cc), flush=True)

    gate_a = i_finite and moved
    gate_b = bool(settle < SETTLE_RTOL)
    overall = gate_a and gate_b
    print("[t]", flush=True)
    print("[t] GATE A (waveform completes, I finite, device charged): {}".format(
        "PASS" if gate_a else "FAIL"), flush=True)
    print("[t] GATE B (transient settles to the independent DC at Vg1, within {:.2%} of swing): "
          "{}".format(SETTLE_RTOL, "PASS" if gate_b else "FAIL"), flush=True)
    print("[t] *** LARGE-SIGNAL TRANSIENT (Park metasurface): {} ***".format(
        "PASS" if overall else "FAIL"), flush=True)
    return overall


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
