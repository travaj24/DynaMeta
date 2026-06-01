"""Validate the Kane in-plane nonparabolicity in the SP 2D filling. For a single 2D
sub-band with the energy-dependent mass m*(eps)=m*0(1+2 alpha eps), the T=0 sheet density
is the closed form n_s = (g_s g_v m*0/2 pi hbar^2)[dE + alpha*dE^2], dE = E_F - E_1. Use a
DEEP NARROW well (only the ground sub-band below E_F) at low T (near-degenerate), and
check: (1) the numerical nonparabolic n_s matches the closed form; (2) it EXCEEDS the
parabolic n_s by ~(1+alpha*dE) (the DOS enhancement); (3) the effective mass at the Fermi
level is m*0(1+2 alpha dE). Run:  python -m validation.sp_nonparabolic
"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dynameta.carriers.schrodinger_poisson import SchrodingerPoisson1D, HBAR, M_E, Q

MSTAR0 = 0.30 * M_E
ALPHA = 0.5            # eV^-1 (ITO-like Kane nonparabolicity)
DE_EV = 0.16           # in-plane fill above the ground sub-band
TOL = 0.03


def main():
    L = 3e-9                                   # narrow well -> E_2 well above E_F
    z = np.linspace(0.0, L, 601)
    sp = SchrodingerPoisson1D(z, MSTAR0, T_K=50.0, g_s=2, g_v=1)   # low T ~ degenerate
    E, _, _ = sp.solve_schrodinger(np.zeros_like(z), n_states=4)
    E1, E2 = E[0], E[1]
    E_F = E1 + DE_EV * Q
    assert E2 > E_F, "E_2 must be above E_F so only the ground sub-band fills"

    res_np = sp.density(np.zeros_like(z), E_F, n_states=4, alpha_np_per_eV=ALPHA)
    res_par = sp.density(np.zeros_like(z), E_F, n_states=4, alpha_np_per_eV=0.0)
    ns_np, ns_par = res_np.sheet_density_m2[0], res_par.sheet_density_m2[0]

    pref0 = sp.g_s * sp.g_v * MSTAR0 / (2.0 * np.pi * HBAR ** 2)   # m^-2 J^-1
    dE = E_F - E1
    a = ALPHA / Q
    ns_cf = pref0 * (dE + a * dE ** 2)                            # T=0 nonparabolic closed form
    m_ratio = 1.0 + 2.0 * a * dE                                  # m*(E_F)/m*0

    print("[t] E1={:.4f} eV  E2={:.4f} eV  E_F={:.4f} eV  dE={:.4f} eV".format(
        E1 / Q, E2 / Q, E_F / Q, dE / Q), flush=True)
    print("[t] n_s nonparabolic={:.4e}  closed-form(T=0)={:.4e}  rel={:.3f}".format(
        ns_np, ns_cf, abs(ns_np - ns_cf) / ns_cf), flush=True)
    print("[t] n_s parabolic={:.4e}  nonpar/par={:.3f}  expected(1+a*dE)={:.3f}".format(
        ns_par, ns_np / ns_par, 1.0 + a * dE), flush=True)
    print("[t] effective-mass enhancement m*(E_F)/m*0 = {:.3f}".format(m_ratio), flush=True)

    matches_cf = abs(ns_np - ns_cf) / ns_cf < TOL
    enhanced = abs(ns_np / ns_par - (1.0 + a * dE)) < 0.02
    ok = matches_cf and enhanced and (ns_np > ns_par)
    print("[t] *** SP NONPARABOLICITY: matches_closed_form={} DOS_enhanced={} -> {} ***".format(
        bool(matches_cf), bool(enhanced and ns_np > ns_par), "PASS" if ok else "FAIL"), flush=True)


if __name__ == "__main__":
    main()
