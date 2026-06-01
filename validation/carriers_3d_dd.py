"""Validate native 3D DRIFT-DIFFUSION carriers (Stacked3DSpec physics='drift_diffusion').
A gated MOS-cap is a zero-current device at DC, so 3D DD must converge and REDUCE to the
3D equilibrium accumulation (the roadmap gate). Solve the same stack at +1V both ways
and compare the gate-side n/n_bg; also check DD is sign-correct (+1V accumulates,
-1V depletes). Run:  python -m validation.carriers_3d_dd
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
    # DD should reduce to equilibrium accumulation (statistics models differ slightly:
    # equilibrium Aymerich-Humet F_1/2 vs DD degenerate-limit FD g-factor -> allow 25%)
    rel = abs(dd_top - eq_top) / max(eq_top, 1e-9)
    reduces = rel < 0.25
    print("[t] DD vs equilibrium n_top rel-diff = {:.3f}".format(rel), flush=True)
    ok = converged and sign_ok and reduces
    print("[t] *** 3D DRIFT-DIFFUSION: converged={} sign_correct={} reduces_to_equilibrium={} -> {} ***".format(
        bool(converged), bool(sign_ok), bool(reduces), "PASS" if ok else "FAIL"), flush=True)


if __name__ == "__main__":
    main()
