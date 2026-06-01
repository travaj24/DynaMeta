"""Validate that ITO band NONPARABOLICITY is now reachable THROUGH the SchrodingerPoisson
CarrierSolver (audit SP-3): SchrodingerPoissonCarrier(alpha_np_per_eV>0) applies a post-hoc
Kane nonparabolic 2D fill on the converged potential, so the emitted accumulation density is
DOS-enhanced vs the parabolic carrier. (The self-consistent potential + bulk E_F stay
parabolic, documented.) Run: python -m validation.sp_carrier_nonparabolic
"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dynameta.carriers.sp_carrier import SchrodingerPoissonCarrier
from dynameta.core.carrier_field import ELECTRON_DENSITY
from dynameta.sweep import BiasPoint

N_BG = 4e26


def _gate_density(alpha):
    c = SchrodingerPoissonCarrier(semi_thk_m=12e-9, n_bg_m3=N_BG, nz=301, n_states=60,
                                   alpha_np_per_eV=alpha)
    cf = c.solve(BiasPoint({"gate": 0.5, "body": 0.0}, "a%.2f" % alpha))
    n = cf.regions["semi"].grid_fields[ELECTRON_DENSITY][0, 0, :]
    c.teardown()
    return np.asarray(n, float)


def main():
    n_par = _gate_density(0.0)
    n_np = _gate_density(0.5)
    finite = bool(np.isfinite(n_np).all() and np.all(n_np >= 0.0))
    enhanced = float(n_np.max()) > float(n_par.max())      # nonparabolic DOS -> more carriers
    ratio = float(n_np.max()) / max(float(n_par.max()), 1.0)
    print("[t] SP carrier nonparabolicity (alpha=0.5/eV vs parabolic), Vg=+0.5V:", flush=True)
    print("[t]   peak n: parabolic={:.3e}  nonparabolic={:.3e}  ratio={:.3f}".format(
        n_par.max(), n_np.max(), ratio), flush=True)
    ok = finite and enhanced and (1.0 < ratio < 3.0)       # enhanced but physically bounded
    print("[t] *** SP CARRIER NONPARABOLIC (alpha reachable via the carrier): {} ***".format(
        "PASS" if ok else "FAIL"), flush=True)
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
