"""3D full-vector FDTD: cross-polarization from an asymmetric 2D-periodic cell.

Split out of validation/fdtd_3d_reduces.py (2026-09-01) UNCHANGED -- same L-pillar, same period,
same resolution, same tolerances. Only the file it lives in is different; see
validation/fdtd_3d_pillar_diffraction.py for why the split was made (that script carried 1191 s of
the original 2133 s, this one 690 s, and together they set a global timeout cap that no other
script needed).

GATE E (cross-polarization): an ASYMMETRIC (L-shaped) 2D-periodic pillar has no mirror symmetry,
        so a y-polarized input generates x-pol (the full Ex/Ez/Hy coupling + the Ex Hy* cross term
        in S_z). The all-order flux STILL conserves energy -- a wrong cross-term sign or a missing
        component would break energy badly here (a dedicated probe measured cross-pol |Ex|/|Ey|
        0-order = 0.87 for this cell). This is the gate that would catch a component the symmetric
        fixtures cannot see, because a mirror-symmetric cell converts nothing and would pass with
        the cross term entirely absent.

Run: python -m validation.fdtd_3d_crosspol
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
    H = Harness("f3e", "3D full-vector FDTD: asymmetric-cell cross-polarization")

    def lshape(nx, ny, nz, zc, pad, zs):
        e = np.ones((nx, ny, nz)); inb = (zc >= pad) & (zc < pad + zs)
        blk = np.zeros((nx, ny), dtype=bool)
        blk[nx // 4:nx - nx // 4, ny // 4:ny // 2] = True
        blk[nx // 4:nx // 2, ny // 4:ny - ny // 4] = True    # L (no mirror symmetry -> y->x conversion)
        for k in np.where(inb)[0]:
            e[:, :, k] = np.where(blk, 6.25, 1.0)
        return e

    def gate_e():
        rE = solve_fdtd_3d([FDTDLayer(thickness_m=450e-9, eps_inf=6.25)],
                           period_x_m=950e-9, period_y_m=950e-9, lateral_eps_inf=lshape,
                           lambda_min_m=LMIN, lambda_max_m=LMAX, resolution=12, n_pad_wave=4.0)
        mE = rE.band
        eE = np.abs(rE.R_flux[mE] + rE.T_flux[mE] - 1.0)
        eE_med = float(np.median(eE))
        spec_minE = float((rE.R0[mE] + rE.T0[mE]).min())
        ok = bool(eE_med < 1e-2 and spec_minE < 0.9)
        return ok, ("cross-pol (asymmetric L-pillar): flux |R+T-1| median={:.2e} ; co-pol 0-order "
                    "min(R0+T0)={:.3f} (strong cross-pol + diffraction, energy still closes)"
                    .format(eE_med, spec_minE))

    H.run("E", gate_e)
    return H.summary("3D FULL-VECTOR FDTD (asymmetric cell: cross-polarization + energy)")


if __name__ == "__main__":
    raise SystemExit(main())
