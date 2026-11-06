"""v0.15 item 4.4.3 oracle: STRONG-INVERSION density gradient through the quasi-equilibrium closure
(physics_bipolar_dd.setup_quasi_equilibrium_carrier) -- GATE G, split out of
validation/bipolar_density_gradient.py (2026-11-06) so that file is back to its ~45 s D2d gates and this
~8-11 min nightly gate runs as its own script (its own run_all shard slot and RAM watchdog). Nothing was
dropped or coarsened: the gate, the harness (mos_qe), the Schrodinger-Poisson inversion oracle
(sp_inversion) and their tolerances are the ones in bipolar_density_gradient.py, called from here.

GATE G (see bipolar_density_gradient.gate_g and docs/audit/2026-11-06-v0.15-dg-wall-strong-inversion.md):
  G1-3  every 50 mV step to 1.3 V in <= 12 Newton iterations: N_A 3e23 / 1e24, gamma 0.5 / 1 / gamma_wall,
        double (REL 1e-5) and extended (REL 1e-7), both carriers DG and closed;
  SP self-checks (free-gas deficit; the hbar^2/m -> 0 classical limit vs the FD first integral);
  G1-4  DG vs the SP inversion oracle with tolerances REGISTERED IN ADVANCE; the registered N_inv 10 % is
        MISSED (+10.6..+11.7 %): a documented model result, reported and pinned as a characterization;
  G1-7  1-D mesh convergence of N_inv and the centroid; G1-5 the quasi-static C-V vs SP;
  the anisotropic-Si SP run, reported only (decision 51).

Run: python -m validation.bipolar_dg_strong_inversion
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import validation.bipolar_density_gradient as V  # noqa: E402


def main():
    print("[bdg] === v0.15 4.4.3 strong-inversion DG (quasi-equilibrium closure) ===", flush=True)
    t0 = time.time()
    ok = V.gate_g()
    print("[bdg] *** STRONG-INVERSION DG (GATE G): {} *** ({:.0f} s)".format(
        "PASS" if ok else "FAIL", time.time() - t0), flush=True)
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
