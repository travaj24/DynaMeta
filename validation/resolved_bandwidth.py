"""
Validate the FREQUENCY-RESOLVED gate capacitance of a gated UNIPOLAR drift-diffusion device via
ssac, enabled by the new unipolar charge time-node model (physics_drift_diffusion: NCharge = -q n
on the continuity equation). A 1D gate | oxide | ITO | ohmic cap with the gate CIRCUIT-DRIVEN:
ac_analysis.ssac_admittance gives C(omega), G(omega) of the gate.

Two checks tie it down:
  * ORACLE: the ssac low-frequency C must equal the QUASI-STATIC gate capacitance dQ/dVg (Q = the
    accumulated electron sheet charge q INT (n - n_bg) dz), computed on the SAME device by a small
    bias difference. ssac (a rigorous small-signal linearization) and dQ/dV (a finite difference)
    must agree -- this is what proves the new NCharge model captures the carrier-charge capacitance
    (without it ssac would return only the resistive part).
  * RESOLVED: C(omega) is frequency-FLAT across kHz..THz -- the thin-ITO intrinsic carrier response
    has no roll-off in the modulation band, so the device bandwidth is set by the ACCESS RC (the
    lumped f_3dB), not by carrier transport. G(omega) ~ omega^2 (the ITO series resistance) confirms
    the gate is a near-ideal capacitor with a small series-R loss.

GATE A: ssac C(omega) converges, C > 0, and is frequency-flat (spread < 1e-3).
GATE B: ssac low-f C == quasi-static dQ/dVg within 5% (the oracle).
INFO  : the access-RC f_3dB from lumped_rc_bandwidth (the real device bottleneck).

Run: python -m validation.resolved_bandwidth
"""

import contextlib
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import devsim as ds

from dynameta.carriers import physics_equilibrium as EQ
from dynameta.carriers import physics_drift_diffusion as DD
from dynameta.carriers import ac_analysis as AC
from dynameta.carriers import eq_registry as _R
from dynameta.core.numerics import trapz as _trapz
from dynameta.analysis import sheet_resistance_ohm_sq, lumped_rc_bandwidth
from dynameta.constants import M_E, EPS0, Q_E

T_OX, T_ITO = 10e-9, 12e-9
EPS_OX, EPS_ITO, N_BG, DOS_MASS, MU = 9.0, 9.5, 4e26, 0.35 * M_E, 30e-4
VG = 1.0
DV = 0.25                                  # +/- bias for the quasi-static dQ/dVg
FREQS = [1.0e3, 1.0e6, 1.0e9, 1.0e12]
FLAT_RTOL, ORACLE_RTOL = 1.0e-3, 5.0e-2


@contextlib.contextmanager
def _quiet():
    sys.stdout.flush(); saved = os.dup(1); devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, 1); yield
    finally:
        sys.stdout.flush(); os.dup2(saved, 1); os.close(devnull); os.close(saved)


def _build():
    ds.create_1d_mesh(mesh="rb")
    ds.add_1d_mesh_line(mesh="rb", pos=0.0, ps=1e-9, tag="gate")
    ds.add_1d_mesh_line(mesh="rb", pos=T_OX, ps=2e-10, tag="mid")
    ds.add_1d_mesh_line(mesh="rb", pos=T_OX + T_ITO, ps=1e-9, tag="back")
    ds.add_1d_contact(mesh="rb", name="gate", tag="gate", material="metal")
    ds.add_1d_contact(mesh="rb", name="back", tag="back", material="metal")
    ds.add_1d_region(mesh="rb", material="ox", region="ox", tag1="gate", tag2="mid")
    ds.add_1d_region(mesh="rb", material="ito", region="ito", tag1="mid", tag2="back")
    ds.add_1d_interface(mesh="rb", name="ox_ito", tag="mid")
    ds.finalize_mesh(mesh="rb"); ds.create_device(mesh="rb", device="rb")
    EQ.setup_dielectric_region("rb", "ox", eps_static=EPS_OX)
    DD.setup_semiconductor_region_dd("rb", "ito", n_bg_m3=N_BG, eps_static=EPS_ITO,
                                     dos_mass_kg=DOS_MASS, mobility_m2Vs=MU)
    EQ.setup_interface("rb", "ox_ito")
    AC.setup_circuit_contact("rb", "gate", node_name="vg", source_name="V1")   # circuit-driven gate
    DD.setup_contact_ohmic_dd("rb", "back")                                    # ITO ohmic ground
    for reg in ("ox", "ito"):
        nx = len(ds.get_node_model_values(device="rb", region=reg, name="x"))
        ds.set_node_values(device="rb", region=reg, name="Potential", values=[0.0] * nx)
    ds.set_node_values(device="rb", region="ito", name="Electrons",
                       values=[N_BG] * len(ds.get_node_model_values(device="rb", region="ito", name="x")))
    ds.set_parameter(device="rb", name="back_bias", value=0.0)


def _dcsolve(ae=1e18, re=1e-5, mi=80):
    with _quiet():
        ds.solve(type="dc", solver_type="direct", absolute_error=ae, relative_error=re,
                 maximum_iterations=mi)


def _ramp_gate(v_to, step=0.25):
    v = float(ds.get_circuit_node_value(node="vg", solution="dcop"))
    while abs(v - v_to) > 1e-9:
        v = max(v_to, v - step) if v_to < v else min(v_to, v + step)
        ds.circuit_alter(name="V1", value=v); _dcsolve()


def _gate_sheet_charge():
    z = np.array(ds.get_node_model_values(device="rb", region="ito", name="x"), dtype=np.float64)
    n = np.array(ds.get_node_model_values(device="rb", region="ito", name="Electrons"), dtype=np.float64)
    return Q_E * float(_trapz(n - N_BG, z))            # accumulated electron sheet charge [C/m^2]


def main():
    print("[t] === Frequency-resolved gate C(omega) on a unipolar DD gated cap (ssac) ===",
          flush=True)
    _build()
    # staged DC solve to VG (gate circuit-driven)
    _R.delete_by_name("rb", "ElectronContinuityEquation")
    with _quiet():
        ds.solve(type="dc", solver_type="direct", absolute_error=1e10, relative_error=1e-5,
                 maximum_iterations=60)
    _R.reapply_by_name("rb", "ElectronContinuityEquation")
    _dcsolve()
    _ramp_gate(VG)

    freqs, C, G = AC.ssac_admittance(FREQS, source_name="V1")
    c_mean = float(np.mean(C))
    spread = float((C.max() - C.min()) / c_mean) if c_mean != 0 else float("inf")
    c_flat = bool(spread < FLAT_RTOL)
    c_pos = bool(np.all(C > 0))

    # quasi-static dQ/dVg on the SAME device (small symmetric bias difference about VG)
    _ramp_gate(VG + DV); q_hi = _gate_sheet_charge()
    _ramp_gate(VG - DV); q_lo = _gate_sheet_charge()
    c_qs = (q_hi - q_lo) / (2.0 * DV)
    oracle_rel = abs(c_mean - c_qs) / max(abs(c_qs), 1e-30)
    oracle_ok = bool(oracle_rel < ORACLE_RTOL)

    # INFO: the access-RC f_3dB (the real device bottleneck, since C is intrinsically flat)
    rho_s = sheet_resistance_ohm_sq(N_BG, MU, T_ITO)
    R, C_cell, f3db = lumped_rc_bandwidth(c_mean, rho_s, path_length_m=5e-6, pad_width_m=1e-6,
                                          cell_area_m2=(370e-9) ** 2)

    for f, c, g in zip(FREQS, C, G):
        print("[t]   f={:9.1e} Hz  C={:.4e} F/m^2  G={:+.3e} S/m^2".format(f, c, g), flush=True)
    print("[t]   ssac C={:.4e} ; quasi-static dQ/dVg={:.4e} F/m^2 ; oracle rel-diff={:.2e}".format(
        c_mean, c_qs, oracle_rel), flush=True)
    print("[t]   C frequency-flat (spread {:.1e}); access-RC f_3dB={:.1f} GHz (the real "
          "bottleneck; intrinsic C is flat to THz)".format(spread, float(f3db) * 1e-9), flush=True)

    gate_a = c_flat and c_pos
    gate_b = oracle_ok
    overall = gate_a and gate_b
    print("[t]", flush=True)
    print("[t] GATE A (ssac C(omega) converges, C>0, frequency-flat): {}".format(
        "PASS" if gate_a else "FAIL"), flush=True)
    print("[t] GATE B (ssac low-f C == quasi-static dQ/dVg within {:.0%}): {}".format(
        ORACLE_RTOL, "PASS" if gate_b else "FAIL"), flush=True)
    print("[t] *** RESOLVED-BANDWIDTH ssac (unipolar DD gate): {} ***".format(
        "PASS" if overall else "FAIL"), flush=True)
    return overall


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
