"""OBLIQUE incidence (2D-TE / s-pol) FDTD via the complex-envelope Bloch method, vs coherent TMM at angle.
The solver fixes the transverse wavevector k_par = (2pi/lambda_c) sin(angle), so the physical angle is
frequency-dependent theta(f) = asin(k_par c / w); the result is compared to s-pol TMM AT theta(f) (an exact
angle-dependent oracle). A laterally-uniform slab must reduce to TMM.

GATES (lossless slab, vacuum ends):
  0  REDUCTION: angle=0 reproduces normal-incidence TMM.
  1  OBLIQUE: at angle in {30,45} deg the FDTD R0/T0 track s-pol TMM(theta(f)) across the band, the
     response is genuinely ANGLE-dependent (differs from the normal-incidence TMM), and energy R0+T0 ~ 1.

Run: python -m validation.fdtd_2d_oblique_vs_tmm
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dynameta.constants import C_LIGHT
from dynameta.optics.fdtd import FDTDLayer
from dynameta.optics.fdtd_nd import solve_fdtd_2d_oblique

LMIN, LMAX, RES = 1300e-9, 1700e-9, 60
N_SLAB, D = 1.9, 280e-9
S = 1.0e6


def _tmm_s(theta_deg, fHz):
    import tmm
    res = tmm.coh_tmm("s", [1.0, N_SLAB, 1.0], [np.inf, D * S, np.inf],
                      np.radians(theta_deg), (C_LIGHT / fHz) * S)
    return float(res["R"]), float(res["T"])


def _gate(angle, tol):
    r = solve_fdtd_2d_oblique([FDTDLayer(thickness_m=D, eps_inf=N_SLAB ** 2)], period_x_m=300e-9,
                              angle_deg=angle, lambda_min_m=LMIN, lambda_max_m=LMAX, resolution=RES)
    b = r.band
    fb, th = r.freqs_Hz[b], r.theta_deg[b]
    Rt = np.array([_tmm_s(t, f)[0] for t, f in zip(th, fb)])
    Tt = np.array([_tmm_s(t, f)[1] for t, f in zip(th, fb)])
    R0n = np.array([_tmm_s(0.0, f)[0] for f in fb])         # normal-incidence TMM (the "angle does nothing" null)
    dR = float(np.max(np.abs(r.R0[b] - Rt))); dT = float(np.max(np.abs(r.T0[b] - Tt)))
    en = float(np.max(np.abs(r.R0[b] + r.T0[b] - 1.0)))
    angle_effect = float(np.max(np.abs(Rt - R0n)))          # how far oblique TMM is from normal TMM
    ok = (dR < tol) and (dT < tol) and (en < tol)
    print("[ob] angle={:4.1f} deg (theta(f) {:.1f}-{:.1f}): max|dR0|={:.2e} max|dT0|={:.2e} max|R+T-1|={:.2e}"
          " | angle-effect={:.3f}  {}".format(angle, th.min(), th.max(), dR, dT, en, angle_effect,
                                              "PASS" if ok else "FAIL"), flush=True)
    return ok, angle_effect


def main():
    print("[ob] === Oblique 2D-TE (s-pol) complex-envelope FDTD vs TMM at angle ===", flush=True)
    g0, _ = _gate(0.0, 2.0e-2)
    g30, e30 = _gate(30.0, 2.5e-2)
    g45, e45 = _gate(45.0, 3.0e-2)
    real_angle = (e30 > 0.02) and (e45 > e30)               # oblique genuinely differs from normal, more so at 45
    ok = g0 and g30 and g45 and real_angle
    print("[ob] angle-dependence is real (oblique TMM differs from normal, growing with angle): {}".format(
        "PASS" if real_angle else "FAIL"), flush=True)
    print("[ob] *** OBLIQUE BLOCH FDTD (s-pol, reduction + TMM(theta(f)) + energy + angle-effect): {} ***".format(
        "PASS" if ok else "FAIL"), flush=True)
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
