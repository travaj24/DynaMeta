"""Validate the FULL-VECTOR 3D magneto-optic / anisotropic FDTD (solve_fdtd_3d_mo, the per-cell 3x3
diagonal eps + magnetized-Drude cyclotron ADE in the six-component 3D Yee engine) against the validated
1-D magneto-optic solver (fdtd_mo, itself checked vs the circular-eigenmode Jones-TMM). For a laterally-
uniform stack at normal incidence the 3D gyrotropic engine must REDUCE to the 1-D Faraday rotation.

GATE A (Faraday reduces to 1-D): the SAME gyrotropic slab (magnetized-Drude wc != 0) through the 3D
        engine matches solve_fdtd_mo_1d's Faraday rotation to < 0.05 deg and its R/T.
GATE B (birefringence + energy): a diagonal-anisotropic slab (wc=0, eps_xx != eps_yy) develops NO
        cross-pol (no gyrotropy -> no polarization mixing) and conserves energy (R+T = 1).

Run: python -m validation.fdtd_3d_mo_vs_1d
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dynameta.optics.fdtd_mo import MOLayer, solve_fdtd_mo_1d
from dynameta.optics.fdtd_nd import solve_fdtd_3d_mo

LMIN, LMAX = 1.2e-6, 1.8e-6


def main():
    print("[m3] === 3D magneto-optic FDTD vs the validated 1-D MO solver ===", flush=True)

    # GATE A: gyrotropic slab -- 3D Faraday must equal 1-D Faraday
    gyro = [MOLayer(thickness_m=300e-9, eps_xx=4.0, eps_yy=4.0, drude_wp_rad_s=2.0e15,
                    drude_gamma_rad_s=1.0e14, cyclotron_wc_rad_s=3.0e14)]
    r1 = solve_fdtd_mo_1d(gyro, lambda_min_m=LMIN, lambda_max_m=LMAX, resolution=40, pol="y")
    r3 = solve_fdtd_3d_mo(gyro, period_x_m=300e-9, period_y_m=300e-9, lambda_min_m=LMIN, lambda_max_m=LMAX,
                          resolution=40, pol="y", nx=4, ny=4)
    f1 = r1.freqs_Hz[r1.band]
    far1 = float(np.median(r1.faraday_deg[r1.band]))
    far3 = float(np.median(np.interp(f1, r3.freqs_Hz[r3.band], r3.faraday_deg[r3.band])))
    R1 = float(np.median(r1.R[r1.band])); R3 = float(np.median(np.interp(f1, r3.freqs_Hz[r3.band], r3.R[r3.band])))
    T1 = float(np.median(r1.T[r1.band])); T3 = float(np.median(np.interp(f1, r3.freqs_Hz[r3.band], r3.T[r3.band])))
    dfar = abs(far1 - far3)
    g_a = (dfar < 0.05) and (abs(R1 - R3) < 5e-3) and (abs(T1 - T3) < 5e-3)
    print("[m3] A gyro Faraday: 1D={:.3f}deg 3D={:.3f}deg |d|={:.4f} ; R 1D/3D {:.4f}/{:.4f} ; "
          "T {:.4f}/{:.4f} -> {}".format(far1, far3, dfar, R1, R3, T1, T3, "OK" if g_a else "FAIL"), flush=True)

    # GATE B: diagonal-anisotropic (no gyro) -- no cross-pol, energy conserves
    bire = [MOLayer(thickness_m=300e-9, eps_xx=4.0, eps_yy=2.25, drude_wp_rad_s=0.0,
                    drude_gamma_rad_s=0.0, cyclotron_wc_rad_s=0.0)]
    rb = solve_fdtd_3d_mo(bire, period_x_m=300e-9, period_y_m=300e-9, lambda_min_m=LMIN, lambda_max_m=LMAX,
                          resolution=40, pol="y", nx=4, ny=4)
    bb = rb.band
    cross = float(np.median(np.abs(rb.t_cross[bb])))
    energy = float(np.max(np.abs(rb.R[bb] + rb.T[bb] - 1.0)))
    g_b = (cross < 1e-6) and (energy < 2e-2)
    print("[m3] B birefringent (no gyro): cross-pol |t_cross|={:.2e} (->0); |R+T-1|={:.2e} -> {}".format(
        cross, energy, "OK" if g_b else "FAIL"), flush=True)

    ok = g_a and g_b
    print("[m3] *** 3D MAGNETO-OPTIC / TENSOR FDTD: {} ***".format("PASS" if ok else "FAIL"), flush=True)
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
