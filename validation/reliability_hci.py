"""REL8 hot-carrier-injection oracle (lucky-electron, honestly marginal for the vertical MOS-cap).

GATE A (reduces-to-closed-form): I_sub = 0 -> t_HCI = inf (no hot carriers); the Takeda power-law
        ratio (I2/I1)^(-m) with m = 2/3 to machine; the trap-generation rate carries the elementary
        charge correctly (rate * q * W * L == A_it * I_sub -- the audit-corrected dimension).
GATE B (the sign quirk): with the PHYSICAL negative activation (Ea ~ -0.1 eV) HCI is WORSE (shorter
        t) at LOW temperature; a positive Ea reverses the ordering -- both directions checked.

Run: python -m validation.reliability_hci
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dynameta.reliability.hci import (trap_generation_rate_per_m2_s, hci_time_to_failure_s, Q_E)


def main():
    print("[rh] === REL8 hot-carrier injection (lucky electron) ===", flush=True)
    ok = True

    # ---- GATE A ----
    inf_ok = np.isinf(hci_time_to_failure_s(0.0, 300.0, C_s=1e6, width_m=1e-6))
    r = float(hci_time_to_failure_s(2e-6, 300.0, C_s=1e6, width_m=1e-6)
              / hci_time_to_failure_s(1e-6, 300.0, C_s=1e6, width_m=1e-6))
    r_an = 2.0 ** (-2.0 / 3.0)
    rate = trap_generation_rate_per_m2_s(5e-6, 1e-6, 2e-6, A_it=1e-3)
    q_dim = rate * Q_E * 1e-6 * 2e-6                        # must equal A_it * I_sub
    g_a = bool(inf_ok and abs(r / r_an - 1) < 1e-12 and abs(q_dim - 1e-3 * 5e-6) / (1e-3 * 5e-6) < 1e-12)
    ok = ok and g_a
    print("[rh] GATE A: I_sub=0 -> inf; Takeda ratio {:.4f} == 2^(-2/3); trap rate q-dimension exact "
          "-> {}".format(r, "PASS" if g_a else "FAIL"), flush=True)

    # ---- GATE B: the negative-Ea direction ----
    cold_neg = float(hci_time_to_failure_s(1e-6, 250.0, C_s=1e6, width_m=1e-6, Ea_eV=-0.1))
    hot_neg = float(hci_time_to_failure_s(1e-6, 350.0, C_s=1e6, width_m=1e-6, Ea_eV=-0.1))
    cold_pos = float(hci_time_to_failure_s(1e-6, 250.0, C_s=1e6, width_m=1e-6, Ea_eV=+0.1))
    hot_pos = float(hci_time_to_failure_s(1e-6, 350.0, C_s=1e6, width_m=1e-6, Ea_eV=+0.1))
    g_b = bool(cold_neg < hot_neg and cold_pos > hot_pos)
    ok = ok and g_b
    print("[rh] GATE B: Ea=-0.1 eV -> WORSE cold (t(250K) {:.3g} < t(350K) {:.3g}); Ea=+0.1 reverses "
          "-> {}".format(cold_neg, hot_neg, "PASS" if g_b else "FAIL"), flush=True)

    print("[rh] *** REL8 HCI: {} ***".format("PASS" if ok else "FAIL"), flush=True)
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
