"""Validate native 3D DRIFT-DIFFUSION carriers (Stacked3DSpec physics='drift_diffusion')
in its EQUILIBRIUM LIMIT. A gated MOS-cap is a zero-current device at DC, so this is a
convergence + sign + statistics check, NOT a transport test (it is mobility-independent;
the genuine transport gate is validation/carriers_3d_resistor.py -- audit F2). 3D DD must
converge and reduce to the 3D equilibrium accumulation: with the accurate generalized-
Einstein g-factor the DD zero-current limit n~exp(psi/(g V_t)) now reproduces the
equilibrium Aymerich-Humet F_1/2 accumulation to ~0 (the old degenerate-asymptote g needed
a 25% tolerance here; audit F1/F2). Run:  python -m validation.carriers_3d_dd
"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dynameta.carriers.devsim_3d import Devsim3DEquilibrium, Stacked3DSpec
from dynameta.core.carrier_field import ELECTRON_DENSITY
from dynameta.sweep import BiasPoint


def gate_ratio(cf, n_bg):
    v = cf.regions["semi"].grid_fields[ELECTRON_DENSITY]   # (nx, ny, nz)
    top = float(np.mean(v[:, :, -1])) / n_bg               # gate/oxide side
    bot = float(np.mean(v[:, :, 0])) / n_bg                # body side
    return top, bot


def solve(physics, vg, n_bg):
    spec = Stacked3DSpec(lateral_m=12e-9, semi_thk_m=12e-9, oxide_thk_m=8e-9,
                          n_bg_m3=n_bg, physics=physics)
    s = Devsim3DEquilibrium(spec)
    cf = s.solve(BiasPoint({"gate": vg, "body": 0.0}, "{}@{:+.0f}V".format(physics, vg)))
    top, bot = gate_ratio(cf, n_bg)
    s.teardown()
    return top, bot


def main():
    n_bg = 4e26
    print("[t] 3D carriers: equilibrium vs drift-diffusion (gated MOS-cap, zero-current)", flush=True)
    eq_top, eq_bot = solve("equilibrium", 1.0, n_bg)
    print("[t] equilibrium  +1V: n_top/n_bg={:.3f}  n_body/n_bg={:.3f}".format(eq_top, eq_bot), flush=True)
    dd_top, dd_bot = solve("drift_diffusion", 1.0, n_bg)
    print("[t] drift-diff   +1V: n_top/n_bg={:.3f}  n_body/n_bg={:.3f}".format(dd_top, dd_bot), flush=True)
    dd_dep, _ = solve("drift_diffusion", -1.0, n_bg)
    print("[t] drift-diff   -1V: n_top/n_bg={:.3f}".format(dd_dep), flush=True)

    converged = np.isfinite([dd_top, dd_bot, dd_dep]).all()
    sign_ok = dd_top > 1.02 and dd_dep < 0.98               # +1V accumulates, -1V depletes
    # DD reduces to the equilibrium accumulation in the zero-current limit. With the
    # accurate generalized-Einstein g the agreement is now near-exact, so the tolerance is
    # tightened from the old 25% (which was sized to absorb the degenerate-asymptote
    # g-error) to 5% (audit F2).
    rel = abs(dd_top - eq_top) / max(eq_top, 1e-9)
    reduces = rel < 0.05
    print("[t] DD vs equilibrium n_top rel-diff = {:.3f}".format(rel), flush=True)
    ok = converged and sign_ok and reduces
    print("[t] *** 3D DRIFT-DIFFUSION (equilibrium-limit + sign): converged={} sign_correct={} "
          "reduces_to_equilibrium={} -> {} ***".format(
        bool(converged), bool(sign_ok), bool(reduces), "PASS" if ok else "FAIL"), flush=True)
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
