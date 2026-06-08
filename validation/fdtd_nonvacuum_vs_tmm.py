"""NON-VACUUM semi-infinite end media (metasurface-on-substrate) in the 2D-TE FDTD. The engine now fills
the z-pads with the superstrate / substrate permittivity n_super^2 / n_sub^2, impedance-matches the CPML
per end, and uses a homogeneous-superstrate incident reference so R/T are correctly normalized (T carries
the n_sub/n_super flux ratio). This validates the new physics against coherent TMM (the exact three-medium
Airy oracle) and a lossless energy budget.

GATES (laterally-uniform slab, so the FDTD MUST reduce to TMM):
  0  BACKWARD COMPAT: n_super=n_sub=1 (vacuum) slab matches TMM (the pre-existing path is unchanged).
  1  THREE-MEDIUM AIRY: glass superstrate n_super=1.5, slab n=2.0, substrate n_sub=1.3 -> FDTD R0/T0
     match TMM R/T across the band.
  2  ASYMMETRIC ENERGY: a lossless slab between n_super=1.5 and n_sub=1.8 conserves R_flux + T_flux = 1
     (Poynting flux already carries the n through H, so no extra factor).

Run: python -m validation.fdtd_2d_nonvacuum_vs_tmm
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dynameta.constants import C_LIGHT
from dynameta.core.layered import LayeredSlab, LayeredStack
from dynameta.optics.fdtd import FDTDLayer
from dynameta.optics.fdtd_nd import available_backends, solve_fdtd_2d, solve_fdtd_3d
from dynameta.optics.tmm_reference import layered_rta

LMIN, LMAX, RES = 1200e-9, 1800e-9, 44


def _tmm_band(n_super, n_slab, d, n_sub, freqs):
    R = np.empty(len(freqs)); T = np.empty(len(freqs))
    for i, fHz in enumerate(freqs):
        lam = C_LIGHT / fHz
        stack = LayeredStack(complex(n_super), complex(n_sub), [LayeredSlab(d, eps=complex(n_slab) ** 2)])
        R[i], T[i], _ = layered_rta(stack, lam)
    return R, T


def _gate(tag, n_super, n_slab, d, n_sub, tol_rt, tol_en):
    res = solve_fdtd_2d([FDTDLayer(thickness_m=d, eps_inf=n_slab ** 2)], period_x_m=300e-9,
                        lambda_min_m=LMIN, lambda_max_m=LMAX, resolution=RES,
                        n_super=n_super, n_sub=n_sub, backend="numpy")
    b = res.band
    fb = res.freqs_Hz[b]
    Rt, Tt = _tmm_band(n_super, n_slab, d, n_sub, fb)
    dR = float(np.max(np.abs(res.R0[b] - Rt)))
    dT = float(np.max(np.abs(res.T0[b] - Tt)))
    den = float(np.max(np.abs(res.R_flux[b] + res.T_flux[b] - 1.0)))
    ok = (dR < tol_rt) and (dT < tol_rt) and (den < tol_en)
    print("[nv] {}: n_super={:.2f} n_slab={:.2f} n_sub={:.2f} -> max|dR0|={:.2e} max|dT0|={:.2e} "
          "max|R+T-1|={:.2e}  {}".format(tag, n_super, n_slab, n_sub, dR, dT, den,
                                          "PASS" if ok else "FAIL"), flush=True)
    return ok


def _gate3d(tag, n_super, n_slab, d, n_sub, tol_rt, tol_en):
    """The same three-medium Airy gate through the FULL-VECTOR 3D engine (tiny lateral grid since the slab
    is laterally uniform -> reduces to 1D/TMM), confirming the 3D n_super/n_sub mirror of the 2D path."""
    bk = "numba" if "numba" in available_backends() else "numpy"
    res = solve_fdtd_3d([FDTDLayer(thickness_m=d, eps_inf=n_slab ** 2)], period_x_m=300e-9, period_y_m=300e-9,
                        nx=4, ny=4, lambda_min_m=LMIN, lambda_max_m=LMAX, resolution=22,
                        n_super=n_super, n_sub=n_sub, backend=bk)
    b = res.band
    Rt, Tt = _tmm_band(n_super, n_slab, d, n_sub, res.freqs_Hz[b])
    dR = float(np.max(np.abs(res.R0[b] - Rt))); dT = float(np.max(np.abs(res.T0[b] - Tt)))
    den = float(np.max(np.abs(res.R_flux[b] + res.T_flux[b] - 1.0)))
    ok = (dR < tol_rt) and (dT < tol_rt) and (den < tol_en)
    print("[nv] {}: n_super={:.2f} n_slab={:.2f} n_sub={:.2f} -> max|dR0|={:.2e} max|dT0|={:.2e} "
          "max|R+T-1|={:.2e}  {}  (backend={})".format(tag, n_super, n_slab, n_sub, dR, dT, den,
                                                        "PASS" if ok else "FAIL", bk), flush=True)
    return ok


def main():
    print("[nv] === Non-vacuum FDTD end media vs coherent TMM (three-medium Airy + energy) ===", flush=True)
    g0 = _gate("GATE0 vacuum    ", 1.0, 2.0, 300e-9, 1.0, 1.5e-2, 1.0e-2)
    g1 = _gate("GATE1 Airy 3-med", 1.5, 2.0, 300e-9, 1.3, 1.5e-2, 1.0e-2)
    g2 = _gate("GATE2 asym energ", 1.5, 3.0, 100e-9, 1.8, 2.0e-2, 1.0e-2)
    g3 = _gate3d("GATE3 3D Airy   ", 1.5, 2.0, 300e-9, 1.3, 2.0e-2, 1.5e-2)
    ok = g0 and g1 and g2 and g3
    print("[nv] *** NON-VACUUM END MEDIA (Airy three-medium 2D+3D + lossless energy): {} ***".format(
        "PASS" if ok else "FAIL"), flush=True)
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
