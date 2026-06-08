"""Validate OBLIQUE incidence in the FULL-VECTOR 3D FDTD engine (the complex-envelope Bloch method with
a 2D transverse wavevector (kx,ky)). The 2D oblique paths use a single kx; this exercises the genuine
2D transverse Bloch (kx AND ky) in the six-component 3D Yee engine. For a laterally-UNIFORM stack the
result must (a) match coherent TMM(theta(f),'s') and (b) be AZIMUTH-INVARIANT (an isotropic layered
stack is rotationally symmetric about z, so a conical az=30deg gives the same R/T as az=0).

GATE A: 3D oblique (conical, az=30deg) R0/T0 track tmm('s', theta(f)) to the thin-slab floor, energy
        closes (vacuum ends).
GATE B: azimuth-invariance -- R0(az=0) == R0(az=30) to ~machine precision (the 2D Bloch envelope kx,ky
        is exactly symmetric for a uniform stack).

Run: python -m validation.fdtd_3d_oblique_vs_tmm
"""
import os
import sys
import math

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tmm
from dynameta.optics.fdtd_nd import FDTDLayer, solve_fdtd_3d_oblique
from dynameta.constants import C_LIGHT

N_SLAB, D_NM = 2.0, 250.0
TOL = 5.0e-2           # thin-slab discretization floor at this resolution


def _run(az):
    return solve_fdtd_3d_oblique([FDTDLayer(thickness_m=D_NM * 1e-9, eps_inf=N_SLAB ** 2)],
                                 period_x_m=400e-9, period_y_m=400e-9, angle_deg=30.0, azimuth_deg=az,
                                 lambda_min_m=1.2e-6, lambda_max_m=1.8e-6, resolution=20, nx=6, ny=6)


def main():
    print("[o3] === 3D oblique FDTD (2D transverse Bloch) vs tmm + azimuth-invariance ===", flush=True)
    r0, r30 = _run(0.0), _run(30.0)
    b = r30.band
    f, th, R, T = r30.freqs_Hz[b], r30.theta_deg[b], r30.R0[b], r30.T0[b]
    dR = dT = 0.0
    for i in np.linspace(0, len(f) - 1, 5).astype(int):
        res = tmm.coh_tmm("s", [1.0, complex(N_SLAB), 1.0], [np.inf, D_NM, np.inf],
                          math.radians(th[i]), C_LIGHT / f[i] * 1e9)
        dR = max(dR, abs(R[i] - res["R"])); dT = max(dT, abs(T[i] - res["T"]))
    energy = float(np.max(np.abs(R + T - 1.0)))
    g_a = (dR < TOL) and (dT < TOL) and (energy < 3e-2)
    print("[o3] A conical(az=30): theta {:.1f}-{:.1f}deg, max|dR0|={:.2e} max|dT0|={:.2e} (tol {:.0e}); "
          "|R+T-1|={:.2e} -> {}".format(th.min(), th.max(), dR, dT, TOL, energy, "OK" if g_a else "FAIL"),
          flush=True)
    bb = r0.band & r30.band
    az_inv = float(np.max(np.abs(r0.R0[bb] - r30.R0[bb])))
    g_b = az_inv < 1e-9
    print("[o3] B azimuth-invariance: max|R0(az0)-R0(az30)|={:.2e} (uniform stack -> ~0) -> {}".format(
        az_inv, "OK" if g_b else "FAIL"), flush=True)
    ok = g_a and g_b
    print("[o3] *** 3D OBLIQUE FDTD vs TMM: {} ***".format("PASS" if ok else "FAIL"), flush=True)
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
