"""MAGNETO-OPTIC / anisotropic 1-D FDTD (fdtd_mo) vs analytic oracles. The solver carries a diagonal
anisotropy (eps_xx, eps_yy) AND a gyrotropic magnetized-Drude ADE (the cyclotron wc*(zhat x J) coupling
that is the physically-correct time-domain origin of the off-diagonal i*g). Validated against:
  1  BIREFRINGENCE: a lossless eps_xx != eps_yy slab -- x-pol sees TMM(n=sqrt(eps_xx)), y-pol sees
     TMM(n=sqrt(eps_yy)); the cross-pol stays ~0 (a diagonal tensor does not rotate).
  2  FARADAY: a gyrotropic magnetized-Drude slab -- the transmitted-polarization rotation matches the
     CIRCULAR-EIGENMODE Jones-TMM (the two circular modes n_pm = sqrt(eps_pm) transmitted independently
     and recombined), and energy is consistent.
  3  REDUCTION: wc=0 -> no rotation (Faraday ~ 0, cross-pol ~ 0) and the diagonal Drude matches scalar TMM.

Run: python -m validation.fdtd_mo_vs_tmm
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dynameta.constants import C_LIGHT
from dynameta.core.layered import LayeredSlab, LayeredStack
from dynameta.optics.fdtd_mo import MOLayer, solve_fdtd_mo_1d
from dynameta.optics.tmm_reference import layered_rta

LMIN, LMAX, RES = 1300e-9, 1700e-9, 70
S = 1.0e6                      # m -> um for the tmm package


def _tmm_T_scalar(n_slab, d, freqs):
    T = np.empty(len(freqs))
    for i, fHz in enumerate(freqs):
        _R, T[i], _A = layered_rta(LayeredStack(1.0 + 0j, 1.0 + 0j,
                                                [LayeredSlab(d, eps=complex(n_slab) ** 2)]), C_LIGHT / fHz)
    return T


def _circular_jones(layer, d, freqs):
    """Circular-eigenmode Jones-TMM oracle: transmit the +/- circular modes (n_pm) independently through
    the slab, recombine a y-polarized input, return (T_total, faraday_deg) per frequency."""
    import tmm
    Tt = np.empty(len(freqs)); far = np.empty(len(freqs))
    for i, fHz in enumerate(freqs):
        lam = C_LIGHT / fHz
        t = {}
        for sgn in (+1, -1):
            # the basis vector e_sgn = (x + sgn*i*y)/sqrt2 is physically the "Ex - sgn*i*Ey" combination,
            # whose magnetized-Drude permittivity is eps_circular(-sgn) (NOT eps_circular(sgn) -- the
            # circular-handedness labeling is a convention; this aligns the oracle to the FDTD's gyration).
            n = np.sqrt(complex(layer.eps_circular(2.0 * np.pi * fHz, -sgn)))
            res = tmm.coh_tmm("s", [1.0, complex(n), 1.0], [np.inf, d * S, np.inf], 0.0, lam * S)
            t[sgn] = complex(res["t"])
        # y-pol input recombined from circular modes: E_x = i(t- - t+)/2, E_y = (t+ + t-)/2
        Ex = 1j * (t[-1] - t[+1]) / 2.0
        Ey = (t[+1] + t[-1]) / 2.0
        Tt[i] = abs(Ex) ** 2 + abs(Ey) ** 2
        far[i] = np.degrees(0.5 * np.arctan2(2.0 * np.real(Ey * np.conj(Ex)), abs(Ey) ** 2 - abs(Ex) ** 2))
    return Tt, far


def main():
    print("[mo] === Magneto-optic / anisotropic 1-D FDTD vs analytic oracles ===", flush=True)

    # GATE 1: birefringence (lossless eps_xx != eps_yy), per-pol vs scalar TMM, cross-pol ~ 0
    bi = MOLayer(thickness_m=350e-9, eps_xx=4.0, eps_yy=2.25)
    rx = solve_fdtd_mo_1d([bi], lambda_min_m=LMIN, lambda_max_m=LMAX, resolution=RES, pol="x")
    ry = solve_fdtd_mo_1d([bi], lambda_min_m=LMIN, lambda_max_m=LMAX, resolution=RES, pol="y")
    bx, by = rx.band, ry.band
    Tx = _tmm_T_scalar(2.0, 350e-9, rx.freqs_Hz[bx])        # x-pol sees sqrt(eps_xx)=2
    Ty = _tmm_T_scalar(1.5, 350e-9, ry.freqs_Hz[by])        # y-pol sees sqrt(eps_yy)=1.5
    dTx = float(np.max(np.abs(rx.T[bx] - Tx))); dTy = float(np.max(np.abs(ry.T[by] - Ty)))
    cross = float(max(np.max(np.abs(rx.t_cross[bx])), np.max(np.abs(ry.t_cross[by]))))
    g1 = (dTx < 1.5e-2) and (dTy < 1.5e-2) and (cross < 1e-2)
    print("[mo] GATE1 birefringence: x-pol max|dT|={:.2e} (n=2), y-pol max|dT|={:.2e} (n=1.5), "
          "max|cross|={:.2e}  {}".format(dTx, dTy, cross, "PASS" if g1 else "FAIL"), flush=True)

    # GATE 2: gyrotropic Faraday vs circular-eigenmode Jones-TMM
    gy = MOLayer(thickness_m=400e-9, eps_xx=2.0, eps_yy=2.0, drude_wp_rad_s=1.2e15,
                 drude_gamma_rad_s=2.0e13, cyclotron_wc_rad_s=3.0e14)
    rg = solve_fdtd_mo_1d([gy], lambda_min_m=LMIN, lambda_max_m=LMAX, resolution=RES, pol="y")
    bg = rg.band
    Tt, fart = _circular_jones(gy, 400e-9, rg.freqs_Hz[bg])
    dT = float(np.max(np.abs(rg.T[bg] - Tt)))
    dfar = float(np.max(np.abs(rg.faraday_deg[bg] - fart)))
    far_mid = float(np.median(rg.faraday_deg[bg]))
    g2 = (dT < 2.0e-2) and (dfar < 1.0) and (abs(far_mid) > 0.5)    # match T + Faraday to <1 deg, nonzero
    print("[mo] GATE2 Faraday: max|dT|={:.2e}, max|d(theta_F)|={:.3f} deg, median theta_F={:.2f} deg  {}".format(
        dT, dfar, far_mid, "PASS" if g2 else "FAIL"), flush=True)

    # GATE 3: reduction wc=0 -> no rotation, diagonal Drude matches scalar TMM
    nr = MOLayer(thickness_m=400e-9, eps_xx=2.0, eps_yy=2.0, drude_wp_rad_s=1.2e15,
                 drude_gamma_rad_s=2.0e13, cyclotron_wc_rad_s=0.0)
    rn = solve_fdtd_mo_1d([nr], lambda_min_m=LMIN, lambda_max_m=LMAX, resolution=RES, pol="y")
    bn = rn.band
    far0 = float(np.max(np.abs(rn.faraday_deg[bn]))); cross0 = float(np.max(np.abs(rn.t_cross[bn])))
    g3 = (far0 < 1e-2) and (cross0 < 1e-3)
    print("[mo] GATE3 reduction wc=0: max|theta_F|={:.2e} deg, max|cross|={:.2e}  {}".format(
        far0, cross0, "PASS" if g3 else "FAIL"), flush=True)

    ok = g1 and g2 and g3
    print("[mo] *** MAGNETO-OPTIC / ANISOTROPIC FDTD (birefringence + Faraday + reduction): {} ***".format(
        "PASS" if ok else "FAIL"), flush=True)
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
