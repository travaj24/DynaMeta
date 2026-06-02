"""
Frequency-resolved gate capacitance of the FULL Park metasurface (2D drift-diffusion) via ssac --
the resolved-bandwidth capability on the real geometry, completing the gated-DD arc.

Builds the Park DD metasurface (LayeredDevsimBuilder, ITO = drift_diffusion with the full-edge
ground), DC-solves to a gate operating point, then reconfigures the gate (top_contact) as a
CIRCUIT-DRIVEN contact (delete its bias Dirichlet, re-add via ac_analysis.setup_circuit_contact) and
runs ssac -> C(omega), G(omega) of the gate. (A clean builder API for an ssac gate is a future
polish; this demonstrates the capability via a post-solve reconfiguration.)

ORACLE: the ssac gate capacitance must equal the QUASI-STATIC dQ/dVg, where Q is the ITO
accumulated electron charge q*sum(NodeVolume*(n - n_bg)) over the ITO region -- both in 2D per-unit-y
units (F/m), computed on the SAME device by a small bias difference. ssac (small-signal
linearization) and dQ/dV (finite difference) must agree, confirming the unipolar NCharge model
carries the carrier-charge capacitance through the full 2D metasurface.

GATE A: ssac C(omega) converges, C > 0, frequency-flat (spread < 1e-3) -- the intrinsic gate
        response has no carrier roll-off in the modulation band.
GATE B: ssac C == quasi-static dQ/dVg within 10% (the oracle).
INFO  : the per-area gate C (C_2D / period_x) and the access-RC f_3dB (the real bottleneck).

Run: python -m validation.resolved_bandwidth_metasurface
"""

import contextlib
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import devsim as ds

from examples.park_2021 import build_park_design
from dynameta.carriers.devsim_layered import LayeredDevsimBuilder
from dynameta.carriers import ac_analysis as AC
from dynameta.sweep import BiasPoint
from dynameta.analysis import sheet_resistance_ohm_sq, lumped_rc_bandwidth
from dynameta.constants import Q_E

VG = 1.0
DV = 0.25
FREQS = [1.0e3, 1.0e6, 1.0e9]
GATE = "top_contact"
FLAT_RTOL, ORACLE_RTOL = 1.0e-3, 1.0e-1
MU, T_ITO, PERIOD = 30e-4, 5e-9, 370e-9


@contextlib.contextmanager
def _quiet():
    sys.stdout.flush(); saved = os.dup(1); devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, 1); yield
    finally:
        sys.stdout.flush(); os.dup2(saved, 1); os.close(devnull); os.close(saved)


def _ito_excess_charge_per_y(dev, n_bg):
    """Q = q * sum(NodeVolume * (n - n_bg)) over the ITO region [C/m per unit y] (2D)."""
    nv = np.array(ds.get_node_model_values(device=dev, region="ito", name="NodeVolume"), dtype=np.float64)
    n = np.array(ds.get_node_model_values(device=dev, region="ito", name="Electrons"), dtype=np.float64)
    return Q_E * float(np.sum(nv * (n - n_bg)))


def _resolve_gate(dev, v):
    ds.circuit_alter(name="V1", value=float(v))
    with _quiet():
        ds.solve(type="dc", solver_type="direct", absolute_error=1e18, relative_error=1e-5,
                 maximum_iterations=60)


def main():
    print("[t] === Frequency-resolved gate C(omega) on the Park metasurface (2D DD, ssac) ===",
          flush=True)
    d = build_park_design("drift_diffusion")
    b = LayeredDevsimBuilder(d, mesh_name="rbm_m", device_name="rbm_d")
    n_bg = float(d.materials.get("ITO").transport.n_bg_m3)
    with _quiet():
        b.solve(BiasPoint({GATE: VG}, "g{:+.0f}".format(VG)))     # DC operating point
    print("[t] DD Park solved at gate +{:.1f} V; gate region = {}".format(
        VG, b._contact_region.get(GATE)), flush=True)

    # reconfigure the gate circuit-driven (delete bias Dirichlet -> circuit contact) and re-settle
    ds.delete_contact_equation(device=b.device, contact=GATE, name="PotentialEquation")
    AC.setup_circuit_contact(b.device, GATE, node_name="vg", source_name="V1")
    _resolve_gate(b.device, VG)

    freqs, C, G = AC.ssac_admittance(FREQS, source_name="V1")     # F/m (2D per unit y)
    c_mean = float(np.mean(C))
    spread = float((C.max() - C.min()) / c_mean) if c_mean != 0 else float("inf")
    c_flat = bool(spread < FLAT_RTOL)
    c_pos = bool(np.all(C > 0))

    # quasi-static dQ/dVg on the SAME device (both ssac C and this are F/m per unit y)
    _resolve_gate(b.device, VG + DV); q_hi = _ito_excess_charge_per_y(b.device, n_bg)
    _resolve_gate(b.device, VG - DV); q_lo = _ito_excess_charge_per_y(b.device, n_bg)
    c_qs = abs(q_hi - q_lo) / (2.0 * DV)
    oracle_rel = abs(c_mean - c_qs) / max(abs(c_qs), 1e-30)
    oracle_ok = bool(oracle_rel < ORACLE_RTOL)

    c_perarea = c_mean / PERIOD                                   # F/m^2 (divide out the cell x-extent)
    rho_s = sheet_resistance_ohm_sq(n_bg, MU, T_ITO)
    R, C_cell, f3db = lumped_rc_bandwidth(c_perarea, rho_s, path_length_m=5e-6, pad_width_m=1e-6,
                                          cell_area_m2=PERIOD ** 2)

    for f, c, g in zip(FREQS, C, G):
        print("[t]   f={:9.1e} Hz  C={:.4e} F/m  G={:+.3e} S/m".format(f, c, g), flush=True)
    print("[t]   ssac C={:.4e} ; quasi-static dQ/dVg={:.4e} F/m ; oracle rel-diff={:.2e}".format(
        c_mean, c_qs, oracle_rel), flush=True)
    print("[t]   per-area gate C = {:.4e} F/m^2 ; access-RC f_3dB = {:.1f} GHz (intrinsic C flat -> "
          "RC-limited)".format(c_perarea, float(f3db) * 1e-9), flush=True)

    gate_a = c_flat and c_pos
    gate_b = oracle_ok
    overall = gate_a and gate_b
    print("[t]", flush=True)
    print("[t] GATE A (metasurface ssac C(omega) converges, C>0, frequency-flat): {}".format(
        "PASS" if gate_a else "FAIL"), flush=True)
    print("[t] GATE B (ssac C == quasi-static dQ/dVg within {:.0%}): {}".format(
        ORACLE_RTOL, "PASS" if gate_b else "FAIL"), flush=True)
    print("[t] *** RESOLVED-BANDWIDTH ssac (Park metasurface): {} ***".format(
        "PASS" if overall else "FAIL"), flush=True)
    return overall


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
