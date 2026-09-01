"""3D full-vector FDTD: genuine 2D-periodic diffraction + all-order energy closure.

Split out of validation/fdtd_3d_reduces.py (2026-09-01) UNCHANGED -- same pillar, same period,
same resolution, same tolerances. Only the file it lives in is different.

WHY IT WAS SPLIT. fdtd_3d_reduces measured 2133 s end to end on the dev box, and 88% of that sat
in two gates: this one at 1191 s and the cross-polarization L-pillar at 690 s. Both are true
2D-periodic cells carrying a full lateral grid, where the reduce-to-1D/2D gates run on a 4x4
lateral grid and cost tens of seconds. That single script then set run_all's PER_SCRIPT_TIMEOUT_S
for all ~215 scripts: at the ~1.9x runner variance measured across the four nightly shards, a
2133 s script needs a ~4000 s cap, and a 4000 s cap cannot detect a hang in anything else. It had
already timed out twice (1800 s, then 2700 s on nightly 33445577932) while PASSING at 1935 s in
between -- the signature of a cap sitting inside the noise, not of a broken gate.

Splitting the two expensive gates into their own scripts leaves every piece far enough under the
cap to be decided by the physics rather than by which runner it landed on, lets run_all's
round-robin sharding balance them independently, and keeps the cap meaningful for everything else.
No gate was weakened to fit a budget -- the point of GATE D's resolution=60, and of this gate's
full lateral grid, is that they cost what they cost.

GATE C (genuine 3D diffraction + energy): a true 2D-periodic dielectric pillar -- the all-order
        flux conserves energy (median |R+T-1| small; the max spikes only at grazing
        diffraction-order emergence, the npml-independent PML limit) WHILE the specular 0-order
        R0+T0 dips below 1 (energy correctly diffracted into the (kx,ky) orders that only a
        2D-periodic cell supports). A 1-D or 2-D engine cannot produce this at all.

Run: python -m validation.fdtd_3d_pillar_diffraction
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dynameta.optics.fdtd import FDTDLayer                              # noqa: E402
from dynameta.optics.fdtd_nd import solve_fdtd_3d                       # noqa: E402
from validation._harness import Harness                                 # noqa: E402

LMIN, LMAX = 1200e-9, 1800e-9


def main():
    H = Harness("f3c", "3D full-vector FDTD: 2D-periodic pillar diffraction")

    def pillar(nx, ny, nz, zc, pad, zs):
        e = np.ones((nx, ny, nz)); inb = (zc >= pad) & (zc < pad + zs)
        qx, qy = nx // 4, ny // 4
        blk = np.zeros((nx, ny), dtype=bool); blk[qx:nx - qx, qy:ny - qy] = True
        for k in np.where(inb)[0]:
            e[:, :, k] = np.where(blk, 6.25, 1.0)            # n=2.5 pillar in vacuum
        return e

    def gate_c():
        rC = solve_fdtd_3d([FDTDLayer(thickness_m=500e-9, eps_inf=6.25)],
                           period_x_m=900e-9, period_y_m=900e-9, lateral_eps_inf=pillar,
                           lambda_min_m=LMIN, lambda_max_m=LMAX, resolution=14, n_pad_wave=4.0)
        mC = rC.band
        e_abs = np.abs(rC.R_flux[mC] + rC.T_flux[mC] - 1.0)
        en_med, en_max = float(np.median(e_abs)), float(np.max(e_abs))
        spec_min = float((rC.R0[mC] + rC.T0[mC]).min())
        ok = bool(en_med < 1e-2 and spec_min < 0.95)
        return ok, ("2D-periodic pillar: flux |R+T-1| median={:.2e} max={:.2e} (grazing) ; "
                    "0-order min(R0+T0)={:.3f} (<1 = diffracted)".format(en_med, en_max, spec_min))

    H.run("C", gate_c)
    return H.summary("3D FULL-VECTOR FDTD (2D-periodic pillar: all-order energy + diffraction)")


if __name__ == "__main__":
    raise SystemExit(main())
