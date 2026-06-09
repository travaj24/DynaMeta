"""Validate the 2-D nematic director theta(x, z) solver in dynameta/carriers/lc_director_2d.py for
laterally patterned (pixelated / fringing-field) LC cells. Independent oracle = the 1-D director
(lc_director.director_profile_bvp at K11=K33) plus analytic field limits.

GATE A (reduce to 1-D): a laterally UNIFORM top voltage with field='uniform_columns' gives an
        x-INDEPENDENT theta(x,z) (each column decoupled) that equals director_profile_bvp(K11=K33=K) to
        the finite-difference discretization floor (x-variation ~1e-5 deg; vs the adaptive 1-D solve
        <0.05 deg at nz=41, shrinking with nz).
GATE B (2-D Laplace field): with a uniform top voltage the 2-D potential solve gives Ez = -V/d uniform,
        a negligible lateral Ex, and the SAME director as the uniform-column limit.
GATE C (lateral patterning + fringing): a two-pixel (low / high voltage) top electrode -> the deep
        interior of each WIDE pixel recovers its own 1-D director (within the elastic-bleed tolerance),
        the optical n_eff(x) varies laterally (clear ON/OFF contrast), and the lateral fringing field Ex
        is strong at the pixel boundary but small in the pixel interior -- the 2-D effect the 1-D model
        cannot represent.

Run: python -m validation.lc_director_2d
"""
import os
import sys
import math

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dynameta.carriers.lc_director_2d import director_profile_2d
from dynameta.carriers.lc_director import director_profile_bvp

THB = math.radians(89.9)
KW = dict(K=17e-12, eps_para=18.7, eps_perp=4.0, d_planar=2e-6, theta_b_rad=THB, n_o=1.52, n_e=1.74)


def _bvp(V, nz):
    return director_profile_bvp(V_app=V, K11=KW["K"], K33=KW["K"], eps_para=KW["eps_para"],
                                eps_perp=KW["eps_perp"], d_planar=KW["d_planar"], nz=nz,
                                theta_b_rad=THB, field_model="uniform", n_o=KW["n_o"], n_e=KW["n_e"])


def main():
    print("[2d] === LC 2-D lateral director theta(x,z) ===", flush=True)

    # GATE A: reduce to 1-D (uniform columns)
    nz = 41
    rA = director_profile_2d(V_top=2.0, field="uniform_columns", Lx_m=8e-6, nx=17, nz=nz, **KW)
    st = _bvp(2.0, nz)
    xvar = math.degrees(float(np.max(np.abs(rA.theta_field_rad - rA.theta_field_rad[0][None, :]))))
    d1d = math.degrees(float(np.max(np.abs(rA.theta_field_rad - st.theta_field_rad[None, :]))))
    g_a = xvar < 1e-3 and d1d < 0.05 and rA.success
    print("[2d] A reduce-1D: x-variation {:.2e} deg, max|theta-director_profile_bvp| {:.4f} deg (FD floor),"
          " iters {} -> {}".format(xvar, d1d, rA.iters, "OK" if g_a else "FAIL"), flush=True)

    # GATE B: 2-D Laplace field, uniform top voltage
    rB = director_profile_2d(V_top=2.0, field="laplace", Lx_m=8e-6, nx=17, nz=nz, **KW)
    Ez_mag = float(np.mean(np.abs(rB.Ez))); Ex_rel = float(np.max(np.abs(rB.Ex))) / Ez_mag
    dvs_col = math.degrees(float(np.max(np.abs(rB.theta_field_rad - rA.theta_field_rad))))
    g_b = abs(Ez_mag - 2.0 / KW["d_planar"]) / (2.0 / KW["d_planar"]) < 1e-3 and Ex_rel < 1e-3 \
        and dvs_col < 1e-2
    print("[2d] B laplace uniform: |Ez| {:.4e} vs V/d {:.4e}, |Ex|/|Ez| {:.2e}, max|theta-uniformcol| "
          "{:.2e} deg -> {}".format(Ez_mag, 2.0 / KW["d_planar"], Ex_rel, dvs_col, "OK" if g_b else "FAIL"),
          flush=True)

    # GATE C: two WIDE pixels (low 0.5 V / high 3.0 V), period 24 um (pixel half-width = 6 d)
    Lx = 24e-6

    def Vstep(xa):
        return np.where(xa < Lx / 2, 0.5, 3.0)

    rC = director_profile_2d(V_top=Vstep, field="laplace", x_boundary="periodic", Lx_m=Lx, nx=49, nz=nz, **KW)
    lo, hi = _bvp(0.5, nz), _bvp(3.0, nz)
    ilo, ihi = rC.x_m.size // 4, 3 * rC.x_m.size // 4          # deep pixel interiors
    mlo = math.degrees(rC.theta_field_rad[ilo, nz // 2]); mhi = math.degrees(rC.theta_field_rad[ihi, nz // 2])
    mlo1, mhi1 = math.degrees(lo.theta_field_rad[nz // 2]), math.degrees(hi.theta_field_rad[nz // 2])
    # fringing: Ex large near the pixel boundary, small deep in a pixel
    ib = rC.x_m.size // 2                                       # boundary column
    Ex_edge = float(np.max(np.abs(rC.Ex[ib - 1:ib + 2, :]))); Ex_int = float(np.max(np.abs(rC.Ex[ilo, :])))
    neff_contrast = abs(rC.n_eff_of_x[ihi] - rC.n_eff_of_x[ilo])
    g_c = (abs(mlo - mlo1) < 1.0 and abs(mhi - mhi1) < 1.0 and neff_contrast > 0.1
           and Ex_edge > 1e6 and Ex_int < 0.1 * Ex_edge and rC.success)
    print("[2d] C patterned: interior midplane lo {:.2f}(1D {:.2f}) hi {:.2f}(1D {:.2f}) deg; n_eff(x) "
          "lo {:.4f} hi {:.4f} (contrast {:.3f}); Ex edge {:.2e} interior {:.2e} V/m -> {}".format(
              mlo, mlo1, mhi, mhi1, rC.n_eff_of_x[ilo], rC.n_eff_of_x[ihi], neff_contrast,
              Ex_edge, Ex_int, "OK" if g_c else "FAIL"), flush=True)

    ok = g_a and g_b and g_c
    print("[2d] *** LC 2-D LATERAL DIRECTOR: {} ***".format("PASS" if ok else "FAIL"), flush=True)
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
