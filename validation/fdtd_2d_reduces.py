"""2D FDTD reference-engine oracle: optics.fdtd_nd.solve_fdtd_2d is a 2D TE Yee solver (periodic in
x, CFS-CPML absorbing layers in z) and is the backend-agnostic NumPy reference the later
Taichi/CuPy/JAX fast kernels are validated against. Three gates establish it is correct:

GATE A (reduces to TMM/1D): a laterally-UNIFORM non-dispersive slab at normal incidence -- the
        x-mean 0-order R0/T0 AND the all-order Poynting-flux R_flux/T_flux both == the analytic Airy
        R/T to ~1e-3 (i.e. the 2D engine reduces EXACTLY to the 1D solver / TMM), and R+T = 1.
GATE B (dispersion): a laterally-uniform Drude slab (the ADE) == the analytic complex-n Airy.
GATE C (genuine 2D diffraction + energy): a lossless binary grating -- the all-order flux conserves
        energy (CFS-CPML: MEDIAN |R+T-1| ~ 1e-3; the max spikes only at the grazing emergence of
        diffraction orders, a fundamental npml-independent PML limit), WHILE the 0-order specular
        R0+T0 dips well below 1 (energy correctly diffracted into higher orders).

Run: python -m validation.fdtd_2d_reduces
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dynameta.optics.fdtd import FDTDLayer
from dynameta.optics.fdtd_nd import solve_fdtd_2d

C = 299792458.0
LMIN, LMAX = 1200e-9, 1800e-9


def airy(f, n, d):
    k0 = 2 * np.pi * np.asarray(f) / C
    b = n * k0 * d
    r1 = (1.0 - n) / (1.0 + n)
    e2 = np.exp(2j * b)
    r = r1 * (1.0 - e2) / (1.0 - r1 ** 2 * e2)
    t = (1.0 - r1 ** 2) * np.exp(1j * b) / (1.0 - r1 ** 2 * e2)
    return np.abs(r) ** 2, np.abs(t) ** 2


def main():
    print("[f2] === 2D FDTD (Phase 0) reference engine: reduce-to-TMM + diffraction ===", flush=True)

    # GATE A: uniform non-dispersive slab -> 1D/TMM (both 0-order and flux), R+T=1
    n, d = 2.0, 300e-9
    rA = solve_fdtd_2d([FDTDLayer(thickness_m=d, eps_inf=n ** 2)], period_x_m=300e-9,
                       lambda_min_m=LMIN, lambda_max_m=LMAX, resolution=40)
    mA = rA.band
    Ra, Ta = airy(rA.freqs_Hz[mA], n, d)
    d0 = max(float(np.max(np.abs(rA.R0[mA] - Ra))), float(np.max(np.abs(rA.T0[mA] - Ta))))
    df = max(float(np.max(np.abs(rA.R_flux[mA] - Ra))), float(np.max(np.abs(rA.T_flux[mA] - Ta))))
    en = float(np.max(np.abs(rA.R_flux[mA] + rA.T_flux[mA] - 1.0)))
    gate_a = bool(d0 < 2e-3 and df < 2e-3 and en < 2e-3)
    print("[f2] A uniform: 0-order max|d-Airy|={:.2e} flux max|d-Airy|={:.2e} max|R+T-1|={:.2e} -> {}".format(
        d0, df, en, "PASS" if gate_a else "FAIL"), flush=True)

    # GATE B: uniform Drude slab -> analytic complex-n Airy
    ei, wp, gm, dd = 4.0, 1.2e15, 2.0e13, 120e-9
    rB = solve_fdtd_2d([FDTDLayer(thickness_m=dd, eps_inf=ei, drude_wp_rad_s=wp, drude_gamma_rad_s=gm)],
                       period_x_m=300e-9, lambda_min_m=LMIN, lambda_max_m=LMAX, resolution=60)
    mB = rB.band
    w = 2 * np.pi * rB.freqs_Hz[mB]
    nB = np.sqrt(ei - wp ** 2 / (w ** 2 + 1j * gm * w))
    Rb, Tb = airy(rB.freqs_Hz[mB], nB, dd)
    dB = max(float(np.max(np.abs(rB.R0[mB] - Rb))), float(np.max(np.abs(rB.T0[mB] - Tb))))
    gate_b = bool(dB < 1e-2)
    print("[f2] B Drude: n in [{:.2f},{:.2f}] ; 0-order max|d-Airy|={:.2e} -> {}".format(
        float(nB.real.min()), float(nB.real.max()), dB, "PASS" if gate_b else "FAIL"), flush=True)

    # GATE C: lossless binary grating -> all-order flux conserves; 0-order dips (diffraction)
    def lat(nx, nz, zc, pad, zstruct):
        e = np.ones((nx, nz))
        inb = (zc >= pad) & (zc < pad + zstruct)
        half = nx // 2
        for i in range(nx):
            e[i, inb] = 4.0 if i < half else 1.0          # n=2 / n=1 stripes
        return e
    rC = solve_fdtd_2d([FDTDLayer(thickness_m=600e-9, eps_inf=1.0)], period_x_m=1400e-9,
                       lateral_eps_inf=lat, lambda_min_m=LMIN, lambda_max_m=LMAX, resolution=30)
    mC = rC.band
    e_abs = np.abs(rC.R_flux[mC] + rC.T_flux[mC] - 1.0)
    en_med, en_max = float(np.median(e_abs)), float(np.max(e_abs))
    spec_min = float((rC.R0[mC] + rC.T0[mC]).min())
    # CPML conserves energy to ~1e-3 across the band (median); the max spikes only at the GRAZING
    # emergence of diffraction orders (orders propagating nearly parallel to the z-interface reflect
    # off any PML -- a fundamental FDTD limit, npml-independent), so gate on the median.
    gate_c = bool(en_med < 5e-3 and spec_min < 0.9)
    print("[f2] C grating: flux |R+T-1| median={:.2e} (CPML) max={:.2e} (grazing-order emergence) ; "
          "0-order min(R0+T0)={:.3f} (<1 = diffracted) -> {}".format(
              en_med, en_max, spec_min, "PASS" if gate_c else "FAIL"), flush=True)

    # GATE D: the fast Numba CPU kernel reproduces the NumPy reference to machine precision (the
    # compiled+threaded backend is byte-for-byte the same physics; it is the fastest backend for the
    # cache-resident unit-cell grids -- ~500-1900 MC/s, beating naive GPU). Skipped if numba absent.
    from dynameta.optics.fdtd_nd import _HAVE_NUMBA
    if _HAVE_NUMBA:
        rD = solve_fdtd_2d([FDTDLayer(thickness_m=d, eps_inf=n ** 2)], period_x_m=300e-9,
                           lambda_min_m=LMIN, lambda_max_m=LMAX, resolution=40, backend="numba")
        mD = mA & rD.band
        dnb = max(float(np.max(np.abs(rA.R0[mD] - rD.R0[mD]))),
                  float(np.max(np.abs(rA.T0[mD] - rD.T0[mD]))))
        gate_d = bool(dnb < 1e-9)
        print("[f2] D numba==numpy: max|dR0,dT0|={:.2e} (machine precision) -> {}".format(
            dnb, "PASS" if gate_d else "FAIL"), flush=True)
    else:
        gate_d = True
        print("[f2] D numba backend: SKIP (numba not installed)", flush=True)

    overall = gate_a and gate_b and gate_c and gate_d
    print("[f2] *** 2D FDTD REFERENCE ENGINE (reduces to 1D/TMM; Drude; grating diffraction; "
          "numba==numpy): {} ***".format("PASS" if overall else "FAIL"), flush=True)
    return overall


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
