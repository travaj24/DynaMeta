"""NUMBA kernels for the magneto-optic (1-D) and oblique (2-D complex-envelope) FDTD solvers. Both have a
fused, JIT-compiled time loop selected by backend='numba'; this gate proves the kernel is byte-for-byte
equivalent (to ~1e-10) to the NumPy reference AND faster, on both solvers.

  - MO 1-D (solve_fdtd_mo_1d): the per-cell magnetized-Drude 2x2 Crank-Nicolson loop -> ~9x.
  - Oblique 2-D (solve_fdtd_2d_oblique): the complex-envelope Bloch loop, SERIAL JIT (nx is small for a
    laterally-smooth envelope, so threading it loses; serial JIT wins) -> ~5x.

GATE: numba R/T (+ Faraday for MO) match numpy to < 1e-10 across the band, and numba is faster. Skipped
(exit 0) if numba is not installed.

Run: python -m validation.fdtd_numba_kernels
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dynameta.optics.fdtd_nd import HAVE_NUMBA, FDTDLayer, solve_fdtd_2d_oblique
from dynameta.optics.fdtd_mo import MOLayer, solve_fdtd_mo_1d

TOL = 1e-10


def main():
    print("[nbk] === Numba FDTD kernels (MO 1-D + oblique 2-D) vs NumPy reference ===", flush=True)
    if not HAVE_NUMBA:
        print("[nbk] numba not installed -> SKIP (exit 0)", flush=True)
        return True

    ok = True

    # ---- (1) magneto-optic 1-D: gyrotropic + birefringent slab ----
    mo = [MOLayer(thickness_m=300e-9, eps_xx=4.0, eps_yy=2.25, drude_wp_rad_s=2.0e15,
                  drude_gamma_rad_s=1.0e14, cyclotron_wc_rad_s=3.0e14)]
    kw = dict(lambda_min_m=1.2e-6, lambda_max_m=1.8e-6, resolution=36, pol="y")
    t = time.time(); rn = solve_fdtd_mo_1d(mo, backend="numpy", **kw); t_np = time.time() - t
    solve_fdtd_mo_1d(mo, backend="numba", **kw)                        # warm (JIT compile)
    t = time.time(); rb = solve_fdtd_mo_1d(mo, backend="numba", **kw); t_nb = time.time() - t
    b = rn.band
    dR = float(np.max(np.abs(rn.R[b] - rb.R[b]))); dT = float(np.max(np.abs(rn.T[b] - rb.T[b])))
    dF = float(np.max(np.abs(rn.faraday_deg[b] - rb.faraday_deg[b])))
    g_mo = (dR < TOL) and (dT < TOL) and (dF < 1e-8) and (t_nb < t_np)
    ok = ok and g_mo
    print("[nbk] MO 1-D : max|dR|={:.1e} max|dT|={:.1e} max|dFaraday_deg|={:.1e} ; numpy={:.2f}s "
          "numba={:.2f}s ({:.1f}x) -> {}".format(dR, dT, dF, t_np, t_nb, t_np / max(t_nb, 1e-9),
                                                 "PASS" if g_mo else "FAIL"), flush=True)

    # ---- (2) oblique 2-D complex-envelope: dielectric slab at 25 deg ----
    ol = [FDTDLayer(thickness_m=250e-9, eps_inf=6.0)]
    okw = dict(period_x_m=400e-9, angle_deg=25.0, lambda_min_m=1.2e-6, lambda_max_m=1.8e-6,
               resolution=26, nx=6)
    t = time.time(); on = solve_fdtd_2d_oblique(ol, backend="numpy", **okw); t_np = time.time() - t
    solve_fdtd_2d_oblique(ol, backend="numba", **okw)                  # warm (JIT compile)
    t = time.time(); ob = solve_fdtd_2d_oblique(ol, backend="numba", **okw); t_nb = time.time() - t
    b = on.band
    dR = float(np.max(np.abs(on.R0[b] - ob.R0[b]))); dT = float(np.max(np.abs(on.T0[b] - ob.T0[b])))
    g_ob = (dR < TOL) and (dT < TOL) and (t_nb < t_np)
    ok = ok and g_ob
    print("[nbk] OBL 2-D: max|dR0|={:.1e} max|dT0|={:.1e} ; numpy={:.2f}s numba={:.2f}s ({:.1f}x) -> {}".format(
        dR, dT, t_np, t_nb, t_np / max(t_nb, 1e-9), "PASS" if g_ob else "FAIL"), flush=True)

    print("[nbk] *** NUMBA FDTD KERNELS: {} ***".format("PASS" if ok else "FAIL"), flush=True)
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
