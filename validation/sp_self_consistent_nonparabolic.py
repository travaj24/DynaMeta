"""Validate the FULLY SELF-CONSISTENT nonparabolic Schrodinger-Poisson solve (the open S-P piece).
Previously Kane nonparabolicity was reachable only as a POST-HOC 2D fill on a converged PARABOLIC
potential; now solve_self_consistent(alpha_np_per_eV>0) folds the Kane DOS m*(eps)=m*0(1+2 alpha eps)
into the Trellakis inner Newton's a-priori density AND its Jacobian (the same closed form as density(),
n_s = pref0(kT F0 + 2a kT^2 F1)), and the device carrier uses a nonparabolic BULK E_F (Kane inversion
alpha E_F^2 + E_F = gamma_F). Gates:

  A BULK CONSISTENCY: a flat-band (unbiased) thick slab recovers n_bg in the middle for BOTH the
    parabolic and the nonparabolic (3D-calibrated E_F + 2D nonparabolic fill) -- proving the 3D bulk
    E_F calibration is consistent with the 2D sub-band nonparabolic fill.
  B BULK E_F: the nonparabolic E_F is LOWER than the parabolic (heavier DOS holds n_bg at a smaller
    E_F) and satisfies the Kane inversion exactly.
  C GENUINELY SELF-CONSISTENT: under a gate bias the self-consistent nonparabolic density differs
    measurably from a POST-HOC nonparabolic fill on the parabolic potential -- i.e. the nonparabolic
    DOS actually changed the converged potential, not just the final fill.
  D REDUCES + CONVERGES: alpha=0 is byte-identical to the parabolic solve; the nonparabolic solve
    converges and stays finite/positive.

Run: python -m validation.sp_self_consistent_nonparabolic
"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dynameta.carriers.schrodinger_poisson import SchrodingerPoisson1D, Q, HBAR, M_E

N_BG = 4e26
M = 0.35 * M_E
GS, GV = 2, 1
ALPHA = 0.5            # eV^-1 (ITO-like)
EPS_R = 9.5


def _bulk_EF(alpha_eV):
    gammaF = (HBAR ** 2 / (2.0 * M)) * (6.0 * np.pi ** 2 * N_BG / (GS * GV)) ** (2.0 / 3.0)
    if alpha_eV <= 0.0:
        return gammaF, gammaF
    a = alpha_eV / Q
    return (-1.0 + np.sqrt(1.0 + 4.0 * a * gammaF)) / (2.0 * a), gammaF


def main():
    print("[t] === Self-consistent nonparabolic Schrodinger-Poisson ===", flush=True)
    ok = True

    # ---- A/B: flat-band thick slab recovers n_bg for parabolic AND nonparabolic ----
    L, nz = 60e-9, 601
    z = np.linspace(0.0, L, nz)
    sp = SchrodingerPoisson1D(z, M, T_K=300.0, g_s=GS, g_v=GV)
    mids = {}
    EFs = {}
    for a in (0.0, ALPHA):
        ef, gammaF = _bulk_EF(a)
        EFs[a] = ef
        phi, nfull, res = sp.solve_self_consistent(
            eps_r=EPS_R, doping_m3=np.full(nz, N_BG), E_F_J=ef, phi_left_V=0.0, phi_right_V=0.0,
            n_states=120, bound_tol=1e9, max_outer=80, tol_V=1e-5, alpha_np_per_eV=a)
        mids[a] = float(nfull[nz // 2 - 30:nz // 2 + 30].mean())
        print("[t] A flat-band alpha={:.1f}: E_F={:.4f} eV  n_mid/n_bg={:.3f}  converged={}".format(
            a, ef / Q, mids[a] / N_BG, res.converged), flush=True)
        ok = ok and bool(res.converged) and abs(mids[a] / N_BG - 1.0) < 0.03
    # B: Kane inversion exact + E_F lowered
    ef_np, gammaF = _bulk_EF(ALPHA)
    a_J = ALPHA / Q
    kane_resid = abs(a_J * ef_np ** 2 + ef_np - gammaF) / gammaF
    lowered = ef_np < EFs[0.0]
    print("[t] B nonparabolic E_F {:.4f} < parabolic {:.4f} eV : {} ; Kane-inversion residual={:.1e}".format(
        ef_np / Q, EFs[0.0] / Q, lowered, kane_resid), flush=True)
    ok = ok and lowered and kane_resid < 1e-9

    # ---- C: self-consistent != post-hoc (under a gate bias) ----
    Lt, nzt = 14e-9, 281
    zt = np.linspace(0.0, Lt, nzt)
    spt = SchrodingerPoisson1D(zt, M, T_K=300.0, g_s=GS, g_v=GV)
    Nd = np.full(nzt, N_BG)
    ef_par, _ = _bulk_EF(0.0)
    phi_par, n_par, _ = spt.solve_self_consistent(eps_r=EPS_R, doping_m3=Nd, E_F_J=ef_par,
        phi_left_V=0.0, phi_right_V=0.5, n_states=80, bound_tol=1e9, max_outer=80, tol_V=1e-5)
    # post-hoc nonparabolic fill on the PARABOLIC potential + PARABOLIC E_F (the old behavior)
    n_posthoc = np.zeros(nzt)
    n_posthoc[1:-1] = spt.density(-Q * phi_par, ef_par, n_states=80, bound_tol=1e9,
                                  alpha_np_per_eV=ALPHA).density_m3
    # fully self-consistent nonparabolic (nonparabolic potential + nonparabolic E_F)
    phi_np, n_np, res_np = spt.solve_self_consistent(eps_r=EPS_R, doping_m3=Nd, E_F_J=ef_np,
        phi_left_V=0.0, phi_right_V=0.5, n_states=80, bound_tol=1e9, max_outer=80, tol_V=1e-5,
        alpha_np_per_eV=ALPHA)
    peak = max(float(n_np.max()), float(n_posthoc.max()))
    rel_diff = float(np.max(np.abs(n_np - n_posthoc))) / peak
    dphi = float(np.max(np.abs(phi_np - phi_par)))
    print("[t] C self-consistent vs post-hoc: max|dn|/peak={:.3f}  max|dphi|={:.4f} V (must be > noise)".format(
        rel_diff, dphi), flush=True)
    ok = ok and (rel_diff > 0.02) and (dphi > 1e-3)

    # ---- D: finite/positive + converged + the DOCSTRING-ADVERTISED alpha=0 byte-identity
    # (audit 7.3: promised, previously absent) -- alpha_np_per_eV=0.0 must be EXACTLY the
    # parabolic code path, not merely close ----
    finite = bool(np.isfinite(n_np).all() and np.all(n_np >= -1e10) and res_np.converged)
    phi_a0, n_a0, _ = spt.solve_self_consistent(eps_r=EPS_R, doping_m3=Nd, E_F_J=ef_par,
        phi_left_V=0.0, phi_right_V=0.5, n_states=80, bound_tol=1e9, max_outer=80, tol_V=1e-5,
        alpha_np_per_eV=0.0)
    byteid = bool(np.array_equal(phi_a0, phi_par) and np.array_equal(n_a0, n_par))
    print("[t] D nonparabolic biased solve converged={} finite_nonneg={}; alpha=0 byte-identical "
          "to parabolic={}".format(res_np.converged, finite, byteid), flush=True)
    ok = ok and finite and byteid

    print("[t] *** SELF-CONSISTENT NONPARABOLIC S-P: {} ***".format("PASS" if ok else "FAIL"), flush=True)
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
