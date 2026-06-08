"""DRUDE + LORENTZ dispersion in the full-vector 3D FDTD (the Lorentz ADE is now carried per E-component in
the numpy/numba/cupy 3D kernels; the 2D-TE-only guard is removed). A laterally-uniform slab must reduce to
coherent TMM evaluated with the SAME eps(w) at each wavelength -- the independent oracle.

GATES (uniform slab, tiny lateral grid so it reduces to 1D/TMM):
  1  DRUDE+LORENTZ slab (resonance in-band) -> 3D FDTD R0/T0 match dispersive TMM.
  2  PURE LORENTZ resonant dielectric (no Drude) -> 3D FDTD R0/T0 match dispersive TMM (isolates the ADE).
  3  BACKEND: numba == numpy on the 3D Lorentz slab (the new per-component ADE is identical across kernels).

Run: python -m validation.fdtd_3d_lorentz_vs_tmm
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dynameta.constants import C_LIGHT
from dynameta.core.layered import LayeredSlab, LayeredStack
from dynameta.optics.fdtd import FDTDLayer
from dynameta.optics.fdtd_nd import available_backends, solve_fdtd_3d
from dynameta.optics.tmm_reference import layered_rta

LMIN, LMAX, RES = 1200e-9, 1800e-9, 28
RES_BK = 10                            # the numba==numpy cross-check runs at low res (numpy 3D is slow)


def _tmm_rt(layer, d, freqs):
    R = np.empty(len(freqs)); T = np.empty(len(freqs))
    for i, fHz in enumerate(freqs):
        eps = layer.eps_at(2.0 * np.pi * fHz)
        R[i], T[i], _ = layered_rta(LayeredStack(1.0 + 0j, 1.0 + 0j, [LayeredSlab(d, eps=eps)]), C_LIGHT / fHz)
    return R, T


def _fdtd3d(layer, d, backend, res=RES):
    return solve_fdtd_3d([FDTDLayer(thickness_m=d, eps_inf=layer.eps_inf, drude_wp_rad_s=layer.drude_wp_rad_s,
                                    drude_gamma_rad_s=layer.drude_gamma_rad_s, lorentz_w0_rad_s=layer.lorentz_w0_rad_s,
                                    lorentz_gamma_rad_s=layer.lorentz_gamma_rad_s,
                                    lorentz_delta_eps=layer.lorentz_delta_eps)],
                         period_x_m=300e-9, period_y_m=300e-9, nx=4, ny=4, lambda_min_m=LMIN,
                         lambda_max_m=LMAX, resolution=res, backend=backend)


def _gate(tag, layer, d, tol, backend):
    res = _fdtd3d(layer, d, backend)
    b = res.band
    Rt, Tt = _tmm_rt(layer, d, res.freqs_Hz[b])
    dR = float(np.max(np.abs(res.R0[b] - Rt))); dT = float(np.max(np.abs(res.T0[b] - Tt)))
    ok = (dR < tol) and (dT < tol)
    print("[3l] {}: max|dR0|={:.2e} max|dT0|={:.2e} (tol {:.0e}, backend={})  {}".format(
        tag, dR, dT, tol, backend, "PASS" if ok else "FAIL"), flush=True)
    return ok


def main():
    print("[3l] === Drude+Lorentz 3D FDTD vs dispersive coherent TMM ===", flush=True)
    bk = "numba" if "numba" in available_backends() else "numpy"
    dl = FDTDLayer(thickness_m=200e-9, eps_inf=2.0, drude_wp_rad_s=1.4e15, drude_gamma_rad_s=5.0e13,
                   lorentz_w0_rad_s=1.30e15, lorentz_gamma_rad_s=1.2e14, lorentz_delta_eps=1.0)
    g1 = _gate("GATE1 Drude+Lorentz", dl, 200e-9, 2.5e-2, bk)
    lo = FDTDLayer(thickness_m=250e-9, eps_inf=2.25, lorentz_w0_rad_s=1.30e15,
                   lorentz_gamma_rad_s=1.2e14, lorentz_delta_eps=1.5)
    g2 = _gate("GATE2 pure Lorentz ", lo, 250e-9, 2.5e-2, bk)

    g3 = True
    if "numba" in available_backends():
        rn = _fdtd3d(dl, 200e-9, "numba", RES_BK); ru = _fdtd3d(dl, 200e-9, "numpy", RES_BK)
        b = rn.band
        dmax = float(np.max(np.abs(rn.R0[b] - ru.R0[b])) + np.max(np.abs(rn.T0[b] - ru.T0[b])))
        g3 = dmax < 1.0e-9
        print("[3l] GATE3 backend numba==numpy (res={}): max|dR0|+|dT0|={:.2e}  {}".format(
            RES_BK, dmax, "PASS" if g3 else "FAIL"), flush=True)

    ok = g1 and g2 and g3
    print("[3l] *** DRUDE+LORENTZ 3D FDTD (vs dispersive TMM + backend consistency): {} ***".format(
        "PASS" if ok else "FAIL"), flush=True)
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
