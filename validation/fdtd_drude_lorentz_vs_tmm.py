"""DRUDE + LORENTZ dispersion in the 2D-TE FDTD. The single-Drude engine cannot represent a bound-electron
/ interband resonance (a Lorentz pole); this adds the central-difference Lorentz ADE (a second polarization
PL) so eps(w) = eps_inf - wp^2/(w^2 + i w gd) + d_eps w0^2/(w0^2 - w^2 - i w gl) runs natively across the
band. Validated against coherent TMM evaluated with the SAME eps(w) at each wavelength (the independent
oracle), plus the fit and cross-backend consistency.

GATES (laterally-uniform slab -> FDTD MUST reduce to TMM):
  1  DRUDE+LORENTZ slab (resonance in-band) -> FDTD R0/T0 match dispersive TMM.
  2  PURE LORENTZ resonant dielectric (no Drude) -> FDTD R0/T0 match dispersive TMM (isolates the new ADE).
  3  FIT: fit_drude_lorentz recovers a known Drude+Lorentz eps(w) (small RMS) and the fitted layer's FDTD
     R0/T0 match TMM.
  4  BACKEND: numba == numpy on the Lorentz slab (the new ADE is identical across kernels).

Run: python -m validation.fdtd_drude_lorentz_vs_tmm
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dynameta.constants import C_LIGHT
from dynameta.core.layered import LayeredSlab, LayeredStack
from dynameta.optics.fdtd import FDTDLayer
from dynameta.optics.fdtd_nd import available_backends, solve_fdtd_2d
from dynameta.optics.fdtd_seam import fit_drude_lorentz
from dynameta.optics.tmm_reference import layered_rta

LMIN, LMAX, RES = 1200e-9, 1800e-9, 40
BK = "numba" if "numba" in available_backends() else "numpy"   # the gates run on the fast kernel


def _tmm_rt(layer, d, freqs):
    R = np.empty(len(freqs)); T = np.empty(len(freqs))
    for i, fHz in enumerate(freqs):
        lam = C_LIGHT / fHz
        eps = layer.eps_at(2.0 * np.pi * fHz)
        R[i], T[i], _ = layered_rta(LayeredStack(1.0 + 0j, 1.0 + 0j, [LayeredSlab(d, eps=eps)]), lam)
    return R, T


def _fdtd_rt(layer, d, backend):
    res = solve_fdtd_2d([FDTDLayer(thickness_m=d, eps_inf=layer.eps_inf,
                                   drude_wp_rad_s=layer.drude_wp_rad_s, drude_gamma_rad_s=layer.drude_gamma_rad_s,
                                   lorentz_w0_rad_s=layer.lorentz_w0_rad_s,
                                   lorentz_gamma_rad_s=layer.lorentz_gamma_rad_s,
                                   lorentz_delta_eps=layer.lorentz_delta_eps)],
                        period_x_m=300e-9, lambda_min_m=LMIN, lambda_max_m=LMAX, resolution=RES, backend=backend)
    return res


def _gate(tag, layer, d, tol, backend="numpy"):
    res = _fdtd_rt(layer, d, backend)
    b = res.band
    Rt, Tt = _tmm_rt(layer, d, res.freqs_Hz[b])
    dR = float(np.max(np.abs(res.R0[b] - Rt))); dT = float(np.max(np.abs(res.T0[b] - Tt)))
    ok = (dR < tol) and (dT < tol)
    print("[dl] {}: max|dR0|={:.2e} max|dT0|={:.2e} (tol {:.0e}, backend={})  {}".format(
        tag, dR, dT, tol, backend, "PASS" if ok else "FAIL"), flush=True)
    return ok


def main():
    print("[dl] === Drude+Lorentz FDTD vs dispersive coherent TMM ===", flush=True)
    # a metal-with-interband: Drude free-carrier tail + a Lorentz resonance inside the 1.2-1.8 um band
    dl = FDTDLayer(thickness_m=200e-9, eps_inf=2.0, drude_wp_rad_s=1.4e15, drude_gamma_rad_s=5.0e13,
                   lorentz_w0_rad_s=1.30e15, lorentz_gamma_rad_s=1.2e14, lorentz_delta_eps=1.0)
    g1 = _gate("GATE1 Drude+Lorentz", dl, 200e-9, 2.0e-2, BK)

    lo = FDTDLayer(thickness_m=250e-9, eps_inf=2.25, lorentz_w0_rad_s=1.30e15,
                   lorentz_gamma_rad_s=1.2e14, lorentz_delta_eps=1.5)        # pure Lorentz (no Drude)
    g2 = _gate("GATE2 pure Lorentz ", lo, 250e-9, 2.0e-2, BK)

    # GATE 3: fit a known Drude+Lorentz eps(w), recover it, run the fitted layer
    lams = np.linspace(LMIN, LMAX, 13)
    w = 2.0 * np.pi * C_LIGHT / lams
    eps_true = np.array([dl.eps_at(wi) for wi in w])
    fit = fit_drude_lorentz(lams, eps_true)
    model = np.array([FDTDLayer(thickness_m=1.0, **fit).eps_at(wi) for wi in w])
    rms = float(np.sqrt(np.mean(np.abs(model - eps_true) ** 2)) / np.mean(np.abs(eps_true)))
    fitted = FDTDLayer(thickness_m=200e-9, **fit)
    g3 = _gate("GATE3 fitted layer ", fitted, 200e-9, 2.0e-2, BK)
    g3 = g3 and (rms < 1.0e-2)
    print("[dl]   fit RMS(eps) rel = {:.2e} (recovers the Drude+Lorentz dispersion)".format(rms), flush=True)

    # GATE 4: cross-backend consistency on the Lorentz slab
    g4 = True
    if "numba" in available_backends():
        rn = _fdtd_rt(dl, 200e-9, "numba"); ru = _fdtd_rt(dl, 200e-9, "numpy")
        b = rn.band
        dmax = float(np.max(np.abs(rn.R0[b] - ru.R0[b])) + np.max(np.abs(rn.T0[b] - ru.T0[b])))
        g4 = dmax < 1.0e-9
        print("[dl] GATE4 backend numba==numpy: max|dR0|+|dT0|={:.2e}  {}".format(
            dmax, "PASS" if g4 else "FAIL"), flush=True)

    ok = g1 and g2 and g3 and g4
    print("[dl] *** DRUDE+LORENTZ ADE (vs dispersive TMM + fit + backend consistency): {} ***".format(
        "PASS" if ok else "FAIL"), flush=True)
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
