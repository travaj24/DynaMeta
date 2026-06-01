"""Validate the ported Stage-4 bandwidth CHAIN end to end: a real carrier-solve voltage
sweep -> C(V) -> intrinsic RC f_3dB (analysis.gate_cv + lumped_rc_bandwidth). This is the
cheapest dynamic figure-of-merit a DC modulator model can produce (no AC/transient solver),
ported from Metasurface_Modulator stage4_system/access_R_f3dB.py.

Runs a gated SchrodingerPoisson sweep (1D, no devsim/ngsolve) WITH an oxide so the calibrated
oxide-series-capacitance psi_s map runs (a PHYSICAL C, not the no-oxide identity-map
over-estimate), integrates the gate charge per bias, differentiates to C(V), and combines with
the ITO sheet resistance + an access geometry. Checks the CHAIN is self-consistent: gate
charge MONOTONIC (accumulation), C > 0 and in a physical mF/m^2 range, and f_3dB in a physical
band. NOTE: this validates that dynameta's own charge->C->bandwidth path produces a sane,
self-consistent f_3dB -- NOT that it reproduces the Metasurface_Modulator's specific ~15.4 GHz
figure (that fixed-C RC-formula check lives in tests/test_bridge.py; audit AN-2).
Run: python -m validation.bandwidth_cv
"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dynameta.carriers.sp_carrier import SchrodingerPoissonCarrier
from dynameta.sweep import BiasPoint
from dynameta.analysis import (gate_cv, sheet_resistance_ohm_sq, lumped_rc_bandwidth,
                               switching_energy_per_area)

N_BG, T_SEMI, PERIOD, MU = 4e26, 5e-9, 370e-9, 30e-4    # ITO film; mu ~ 30 cm^2/Vs (Park t=5nm)
OXIDE_T, EPS_OX = 5e-9, 9.0                             # gate oxide -> calibrated psi_s map
BIASES = [0.0, 0.5, 1.0, 1.5, 2.0]
PATH_M, PAD_M = 5e-6, 1e-6                               # medium access geometry


def main():
    carrier = SchrodingerPoissonCarrier(semi_thk_m=T_SEMI, n_bg_m3=N_BG, nz=301, n_states=60,
                                        oxide_thk_m=OXIDE_T, eps_oxide=EPS_OX)
    fields = [carrier.solve(BiasPoint({"gate": vg, "body": 0.0}, "vg%.2f" % vg)) for vg in BIASES]
    Vg, Q, Vmid, C = gate_cv(fields, "semi", voltage_key="gate")
    rho_s = sheet_resistance_ohm_sq(N_BG, MU, T_SEMI)
    R, C_cell, f3db = lumped_rc_bandwidth(C, rho_s, path_length_m=PATH_M, pad_width_m=PAD_M,
                                          cell_area_m2=PERIOD ** 2)
    E_area = switching_energy_per_area(C, voltage_swing_V=(BIASES[-1] - BIASES[0]))
    print("[t] BANDWIDTH chain: ITO n_bg={:.1e} mu={:.0f}cm2/Vs t={:.0f}nm cell={:.0f}nm".format(
        N_BG, MU * 1e4, T_SEMI * 1e9, PERIOD * 1e9), flush=True)
    print("[t]   sheet R = {:.0f} Ohm/sq;  access R = {:.0f} Ohm ({:.1f}um/{:.1f}um)".format(
        rho_s, R, PATH_M * 1e6, PAD_M * 1e6), flush=True)
    print("[t]   Q(Vg) [C/m^2] = " + " ".join("{:+.3e}".format(q) for q in Q), flush=True)
    for vm, c, f in zip(Vmid, C, f3db):
        print("[t]   Vg~{:+.2f}V  C={:.3e} F/m^2 ({:.2f} mF/m^2)  C_cell={:.3f} fF  "
              "f_3dB={:.2f} GHz".format(vm, c, c * 1e3, c * PERIOD ** 2 * 1e15, f * 1e-9), flush=True)
    q_monotonic = bool(np.all(np.diff(Q) > -1e-6 * abs(Q).max()))   # accumulation rises with Vg
    c_positive = bool(np.all(C > 0))
    # 1-50 mF/m^2: a physical gated-ITO accumulation capacitance. This also catches a
    # regression to the no-oxide identity-map psi_s, which over-estimates C to
    # ~86-113 mF/m^2 (the degenerate-pass the original [0.1,1000]GHz band missed; audit AN-2).
    c_physical = bool(np.all(C > 1e-3) and np.all(C < 50e-3))
    f_ghz_band = bool(np.all(np.isfinite(f3db)) and np.all(f3db > 2e9) and np.all(f3db < 100e9))
    e_positive = bool(np.all(E_area > 0))
    ok = q_monotonic and c_positive and c_physical and f_ghz_band and e_positive
    print("[t]   gate-charge monotonic={}  C>0={}  C in [1,50]mF/m^2={}  f_3dB in [2,100]GHz={}  "
          "E>0={}".format(q_monotonic, c_positive, c_physical, f_ghz_band, e_positive), flush=True)
    print("[t] *** LUMPED-RC BANDWIDTH (C-V -> intrinsic f_3dB): {} ***".format(
        "PASS" if ok else "FAIL"), flush=True)
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
