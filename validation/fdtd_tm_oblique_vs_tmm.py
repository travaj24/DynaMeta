"""Validate the p-pol (TM) OBLIQUE 2D FDTD (the complex-envelope Bloch method, Hy/Ex/Ez) against p-pol
coherent TMM at the frequency-dependent angle theta(f)=asin(k_par c/w). The TE (s-pol) oblique path was
already validated; this adds the dual TM kernel (Drude ADE on both in-plane E-components). A dielectric
slab air/slab(n=2,250nm)/air at a band-centre 30deg: across the propagating band the FDTD R0/T0 track
tmm('p', theta(f)) to the thin-slab discretization floor, the physical angle varies with frequency, and
energy closes (vacuum ends). A TE cross-check is run for parity. Run: python -m validation.fdtd_tm_oblique_vs_tmm
"""
import os
import sys
import math

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tmm
from dynameta.optics.fdtd_nd import FDTDLayer, solve_fdtd_2d_oblique
from dynameta.constants import C_LIGHT

N_SLAB, D_NM, THETA = 2.0, 250.0, 30.0
TOL = 4.0e-2            # the thin-slab discretization floor (the s-pol oblique path documents ~2%)


def _check(pol):
    r = solve_fdtd_2d_oblique([FDTDLayer(thickness_m=D_NM * 1e-9, eps_inf=N_SLAB ** 2)], period_x_m=400e-9,
                              angle_deg=THETA, lambda_min_m=1.2e-6, lambda_max_m=1.8e-6, resolution=28,
                              nx=6, pol=pol)
    b = r.band
    f, th, R, T = r.freqs_Hz[b], r.theta_deg[b], r.R0[b], r.T0[b]
    dR = dT = 0.0
    for i in np.linspace(0, len(f) - 1, 6).astype(int):
        lam_nm = C_LIGHT / f[i] * 1e9
        res = tmm.coh_tmm(pol, [1.0, complex(N_SLAB), 1.0], [np.inf, D_NM, np.inf],
                          math.radians(th[i]), lam_nm)
        dR = max(dR, abs(R[i] - res["R"])); dT = max(dT, abs(T[i] - res["T"]))
    energy = float(np.max(np.abs(R + T - 1.0)))
    angle_span = float(th.max() - th.min())
    return dR, dT, energy, angle_span, float(th.min()), float(th.max())


def main():
    print("[t] === p-pol (TM) + s-pol oblique 2D FDTD vs tmm(theta(f)) ===", flush=True)
    ok = True
    for pol, tag in (("p", "TM (p-pol, NEW)"), ("s", "TE (s-pol, parity)")):
        dR, dT, energy, span, th0, th1 = _check(pol)
        good = (dR < TOL) and (dT < TOL) and (energy < 3e-2) and (span > 1.0)
        ok = ok and good
        print("[t] {:<20s}: theta {:.1f}-{:.1f}deg, max|dR0|={:.2e} max|dT0|={:.2e} (tol {:.0e}); "
              "|R+T-1|={:.2e}; angle-effect>>error -> {}".format(
                  tag, th0, th1, dR, dT, TOL, energy, "OK" if good else "FAIL"), flush=True)
    print("[t] *** TM/TE OBLIQUE FDTD vs TMM: {} ***".format("PASS" if ok else "FAIL"), flush=True)
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
