"""Validate the numba-CUDA 2D-TE FDTD backend (a persistent cooperative-groups GPU kernel run in
timestep CHUNKS to stay under the WDDM TDR watchdog). Skipped (exit 0) if no CUDA GPU is available.

GATE A (byte-match): backend='numba-cuda' reproduces the CPU reference (backend='numpy') R0/T0/R_flux to
        the float64 FMA floor (< 1e-9) -- the GPU kernel is the SAME physics, just parallelized.
GATE B (vacuum transparency): a vacuum slab (eps=1, no Drude) on the GPU backend gives R ~ 0, T ~ 1
        across the band (an end-to-end physical sanity check with no external reference).
GATE C (dispersive + Lorentz + structured): a laterally-structured Drude+Lorentz grating on the GPU ==
        the CPU numba kernel (exercises the per-cell Drude ADE, the Lorentz ADE pole, and the lateral
        pattern on the device).

Run: python -m validation.fdtd_numba_cuda
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dynameta.optics.fdtd import FDTDLayer
from dynameta.optics.fdtd_nd import solve_fdtd_2d, _have_numba_cuda


def main():
    print("[cu] === numba-CUDA 2D-TE FDTD backend ===", flush=True)
    if not _have_numba_cuda():
        print("[cu] no CUDA GPU available -> SKIP (exit 0)", flush=True)
        return True

    # GATE A: byte-match vs the NumPy reference (uniform Drude slab)
    ol = [FDTDLayer(thickness_m=250e-9, eps_inf=4.0, drude_wp_rad_s=1.5e15, drude_gamma_rad_s=1e14)]
    kw = dict(period_x_m=300e-9, lambda_min_m=1200e-9, lambda_max_m=1600e-9, resolution=16, nx=8)
    a = solve_fdtd_2d(ol, backend="numpy", **kw)
    g = solve_fdtd_2d(ol, backend="numba-cuda", **kw)
    m = a.band
    dR = float(np.max(np.abs(a.R0[m] - g.R0[m]))); dT = float(np.max(np.abs(a.T0[m] - g.T0[m])))
    dF = float(np.max(np.abs(a.R_flux[m] - g.R_flux[m])))
    g_a = dR < 1e-9 and dT < 1e-9 and dF < 1e-9
    print("[cu] A GPU==numpy: max|dR0|={:.2e} max|dT0|={:.2e} max|dRflux|={:.2e} -> {}".format(
        dR, dT, dF, "OK" if g_a else "FAIL"), flush=True)

    # GATE B: vacuum slab on the GPU path -> transparent (R~0, T~1)
    vac = [FDTDLayer(thickness_m=250e-9, eps_inf=1.0)]
    gv = solve_fdtd_2d(vac, backend="numba-cuda", **kw)
    mv = gv.band
    Rv = float(np.max(np.abs(gv.R0[mv]))); Tv = float(np.min(gv.T0[mv]))
    g_b = Rv < 2e-2 and Tv > 0.98
    print("[cu] B GPU vacuum slab: max R0={:.2e} (~0), min T0={:.4f} (~1) -> {}".format(
        Rv, Tv, "OK" if g_b else "FAIL"), flush=True)

    # GATE C: structured Drude+Lorentz grating, GPU == CPU numba (per-cell ADE + lateral pattern + Lorentz)
    def lat(nx, nz, zc, pad, zs):
        e = np.ones((nx, nz)); mm = (zc >= pad) & (zc < pad + zs); e[:nx // 2, mm] = 6.25; return e

    olc = [FDTDLayer(thickness_m=220e-9, eps_inf=3.0, drude_wp_rad_s=1.0e15, drude_gamma_rad_s=8e13,
                     lorentz_delta_eps=1.5, lorentz_w0_rad_s=1.4e15, lorentz_gamma_rad_s=6e13)]
    kwc = dict(period_x_m=600e-9, lambda_min_m=1200e-9, lambda_max_m=1700e-9, resolution=14, nx=24,
               lateral_eps_inf=lat)
    cpu = solve_fdtd_2d(olc, backend="numba", **kwc)
    gpu = solve_fdtd_2d(olc, backend="numba-cuda", **kwc)
    mc = cpu.band
    dc = max(float(np.max(np.abs(cpu.R0[mc] - gpu.R0[mc]))), float(np.max(np.abs(cpu.T0[mc] - gpu.T0[mc]))))
    g_c = dc < 1e-9
    print("[cu] C structured Drude+Lorentz GPU==CPU: max|d|={:.2e} -> {}".format(dc, "OK" if g_c else "FAIL"),
          flush=True)

    ok = g_a and g_b and g_c
    print("[cu] *** numba-CUDA 2D-TE FDTD: {} ***".format("PASS" if ok else "FAIL"), flush=True)
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
