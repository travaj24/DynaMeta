"""Validate ITO band NONPARABOLICITY through the SchrodingerPoisson CarrierSolver, now FULLY
SELF-CONSISTENT (was a post-hoc 2D fill on a parabolic potential). SchrodingerPoissonCarrier(
alpha_np_per_eV>0) calibrates the bulk E_F on the SP model's own flat-band bulk (parabolic z, in-plane
Kane DOS, finite T; SchrodingerPoisson1D.bulk_fermi_level) and runs solve_self_consistent with the
nonparabolic Kane DOS in the Trellakis Newton, so the converged potential AND density carry the
nonparabolicity.

Gates (the self-consistent physics -- NOT the old post-hoc "DOS-enhanced peak"):
  - the nonparabolic bulk E_F is LOWER than parabolic (a heavier DOS holds n_bg at a smaller E_F);
  - the accumulation solve stays finite/positive and converges;
  - at a fixed gate surface potential the nonparabolic peak density is close to the parabolic one
    (ratio in 0.7-1.1) -- the heavier DOS + lower E_F redistribute the accumulation rather than
    naively multiplying it (the post-hoc fill's >1 enhancement was at a FIXED parabolic potential +
    E_F, which is not the self-consistent answer).
Measured (model E_F; m0 0.25 m_e, alpha 0.4/eV, n_bg 3e26 m^-3): E_F 0.6528 / 0.5818 eV, peak ratio 1.014.
Run: python -m validation.sp_carrier_nonparabolic
"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dynameta.carriers.sp_carrier import SchrodingerPoissonCarrier
from dynameta.carriers.schrodinger_poisson import Q, M_E
from dynameta.core.carrier_field import ELECTRON_DENSITY
from dynameta.sweep import BiasPoint

N_BG = 3e26
M0 = 0.25 * M_E                 # Kane band-edge mass (both carriers)
ALPHA = 0.4                     # eV^-1


def _carrier(alpha):
    return SchrodingerPoissonCarrier(semi_thk_m=12e-9, n_bg_m3=N_BG, m_eff_kg=M0, eps_static=9.0, nz=301,
                                     n_states=60,
                                     alpha_np_per_eV=alpha)


def _gate_density(c):
    cf = c.solve(BiasPoint({"gate": 0.5, "body": 0.0}, "g"))
    n = np.asarray(cf.regions["semi"].grid_fields[ELECTRON_DENSITY][0, 0, :], float)
    c.teardown()
    return n


def main():
    c_par, c_np = _carrier(0.0), _carrier(ALPHA)
    ef_par, ef_np = c_par.E_F_J / Q, c_np.E_F_J / Q
    n_par, n_np = _gate_density(c_par), _gate_density(c_np)
    finite = bool(np.isfinite(n_np).all() and np.all(n_np >= 0.0))
    ratio = float(n_np.max()) / max(float(n_par.max()), 1.0)
    print("[t] SP carrier nonparabolicity (alpha={}/eV vs parabolic), Vg=+0.5V:".format(ALPHA), flush=True)
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
