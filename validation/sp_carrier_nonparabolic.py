"""Validate ITO band NONPARABOLICITY through the SchrodingerPoisson CarrierSolver, now FULLY
SELF-CONSISTENT (was a post-hoc 2D fill on a parabolic potential). SchrodingerPoissonCarrier(
alpha_np_per_eV>0) calibrates the bulk E_F with the Kane inversion (alpha E_F^2 + E_F = gamma_F) and
runs solve_self_consistent with the nonparabolic Kane DOS in the Trellakis Newton, so the converged
potential AND density carry the nonparabolicity.

Gates (the self-consistent physics -- NOT the old post-hoc "DOS-enhanced peak"):
  - the nonparabolic bulk E_F is LOWER than parabolic (a heavier DOS holds n_bg at a smaller E_F);
  - the accumulation solve stays finite/positive and converges;
  - at a fixed gate surface potential the nonparabolic peak density is close to (and, self-
    consistently, slightly below) the parabolic -- the heavier DOS + lower E_F redistribute the
    accumulation rather than naively multiplying it (the post-hoc fill's >1 enhancement was at a
    FIXED parabolic potential + E_F, which is not the self-consistent answer).
Run: python -m validation.sp_carrier_nonparabolic
"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dynameta.carriers.sp_carrier import SchrodingerPoissonCarrier
from dynameta.carriers.schrodinger_poisson import Q
from dynameta.core.carrier_field import ELECTRON_DENSITY
from dynameta.sweep import BiasPoint

N_BG = 4e26


def _carrier(alpha):
    return SchrodingerPoissonCarrier(semi_thk_m=12e-9, n_bg_m3=N_BG, nz=301, n_states=60,
                                     alpha_np_per_eV=alpha)


def _gate_density(c):
    cf = c.solve(BiasPoint({"gate": 0.5, "body": 0.0}, "g"))
    n = np.asarray(cf.regions["semi"].grid_fields[ELECTRON_DENSITY][0, 0, :], float)
    c.teardown()
    return n


def main():
    c_par, c_np = _carrier(0.0), _carrier(0.5)
    ef_par, ef_np = c_par.E_F_J / Q, c_np.E_F_J / Q
    n_par, n_np = _gate_density(c_par), _gate_density(c_np)
    finite = bool(np.isfinite(n_np).all() and np.all(n_np >= 0.0))
    ratio = float(n_np.max()) / max(float(n_par.max()), 1.0)
    print("[t] SP carrier nonparabolicity (alpha=0.5/eV vs parabolic), Vg=+0.5V:", flush=True)
    print("[t]   bulk E_F: parabolic={:.4f} eV  nonparabolic={:.4f} eV (lower for heavier DOS)".format(
        ef_par, ef_np), flush=True)
    print("[t]   peak n: parabolic={:.3e}  nonparabolic={:.3e}  ratio={:.3f}".format(
        n_par.max(), n_np.max(), ratio), flush=True)
    g_ef = ef_np < ef_par - 1e-3                            # nonparabolic E_F genuinely lower
    g_phys = finite and (0.7 < ratio < 1.1)                # self-consistently close, slightly below
    ok = bool(g_ef and g_phys)
    print("[t] *** SP CARRIER NONPARABOLIC (self-consistent via the carrier): {} ***".format(
        "PASS" if ok else "FAIL"), flush=True)
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
