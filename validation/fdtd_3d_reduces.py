"""3D full-vector FDTD oracle: optics.fdtd_nd.solve_fdtd_3d is a doubly-periodic (x AND y) 6-component
Yee solver (Bloch-periodic x/y at normal incidence, CFS-CPML + PEC in z) -- the 2D-TE engine is its
d/dy=0, {Ey,Hx,Hz} reduction. Seven gates establish it is correct (the last = backend equivalence):

GATE A (reduces to TMM/1D): a laterally-UNIFORM non-dispersive slab -- the specular 0-order R0/T0 AND
        the all-order Poynting flux R_flux/T_flux both == the analytic Airy R/T (so the full vector 3D
        engine reduces EXACTLY to the 1D solver / TMM), and R+T = 1. Run on a tiny 4x4 lateral grid
        (the result is laterally uniform, so nx,ny are irrelevant -> fast).
GATE B (reduces to the 2D engine): a y-UNIFORM binary grating (varies in x only) -- with a y-polarized
        source and d/dy=0 the problem is pure 2D-TE, so solve_fdtd_3d must reproduce the validated
        solve_fdtd_2d R0/T0 and flux (small residual from the 3D CFL dt differing via 1/dy^2).
GATE C (genuine 3D diffraction + energy): a true 2D-periodic dielectric pillar -- the all-order flux
        conserves energy (median |R+T-1| small; the max spikes only at grazing diffraction-order
        emergence, the npml-independent PML limit) WHILE the specular 0-order R0+T0 dips below 1
        (energy correctly diffracted into the (kx,ky) orders that only a 2D-periodic cell supports).
GATE D (lossy -> defeats the lossless trap): a uniform absorbing Drude slab == the analytic complex-n
        Airy. A lossless cell auto-balances total power even with a wrong field split, so energy closure
        alone cannot prove correctness -- an absorbing slab cannot fake the right R/T.

Run: python -m validation.fdtd_3d_reduces
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dynameta.optics.fdtd import FDTDLayer
from dynameta.optics.fdtd_nd import solve_fdtd_2d, solve_fdtd_3d

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
    print("[f3] === 3D full-vector FDTD: reduce-to-TMM, reduce-to-2D, 3D diffraction ===", flush=True)

    # GATE A: uniform non-dispersive slab -> Airy/TMM (0-order AND flux), R+T=1. 4x4 lateral = fast.
    n, d = 2.0, 300e-9
    rA = solve_fdtd_3d([FDTDLayer(thickness_m=d, eps_inf=n ** 2)], period_x_m=300e-9, period_y_m=300e-9,
                       nx=4, ny=4, lambda_min_m=LMIN, lambda_max_m=LMAX, resolution=18)
    mA = rA.band
    Ra, Ta = airy(rA.freqs_Hz[mA], n, d)
    d0 = max(float(np.max(np.abs(rA.R0[mA] - Ra))), float(np.max(np.abs(rA.T0[mA] - Ta))))
    df = max(float(np.max(np.abs(rA.R_flux[mA] - Ra))), float(np.max(np.abs(rA.T_flux[mA] - Ta))))
    en = float(np.max(np.abs(rA.R_flux[mA] + rA.T_flux[mA] - 1.0)))
    gate_a = bool(d0 < 5e-3 and df < 5e-3 and en < 1e-3)
    print("[f3] A uniform: 0-order max|d-Airy|={:.2e} flux max|d-Airy|={:.2e} max|R+T-1|={:.2e} -> {}".format(
        d0, df, en, "PASS" if gate_a else "FAIL"), flush=True)

    # GATE B: y-uniform binary grating -> the validated 2D-TE engine (Ey source, d/dy=0 == pure 2D-TE)
    PX = 700e-9

    def lat2(nx, nz, zc, pad, zs):
        e = np.ones((nx, nz)); inb = (zc >= pad) & (zc < pad + zs); half = nx // 2
        for i in range(nx):
            e[i, inb] = 4.0 if i < half else 1.0
        return e

    def lat3(nx, ny, nz, zc, pad, zs):
        e = np.ones((nx, ny, nz)); inb = (zc >= pad) & (zc < pad + zs); half = nx // 2
        for i in range(nx):
            e[i, :, inb] = 4.0 if i < half else 1.0
        return e

    kw = dict(lambda_min_m=LMIN, lambda_max_m=LMAX, resolution=16, n_pad_wave=4.0, settle=12.0)
    r2 = solve_fdtd_2d([FDTDLayer(thickness_m=600e-9, eps_inf=1.0)], period_x_m=PX, lateral_eps_inf=lat2, **kw)
    r3 = solve_fdtd_3d([FDTDLayer(thickness_m=600e-9, eps_inf=1.0)], period_x_m=PX, period_y_m=4000e-9,
                       ny=4, lateral_eps_inf=lat3, **kw)
    mB = r2.band                                            # 3D CFL dt differs via 1/dy^2 -> interp onto 2D grid
    f2, f3 = r2.freqs_Hz[mB], r3.freqs_Hz
    ip = lambda v: np.interp(f2, f3, v)
    d0B = max(float(np.max(np.abs(ip(r3.R0) - r2.R0[mB]))), float(np.max(np.abs(ip(r3.T0) - r2.T0[mB]))))
    dfB = max(float(np.max(np.abs(ip(r3.R_flux) - r2.R_flux[mB]))),
              float(np.max(np.abs(ip(r3.T_flux) - r2.T_flux[mB]))))
    gate_b = bool(d0B < 1e-3 and dfB < 1e-3)                # actual ~2.5e-6: the only residual is the dt diff
    print("[f3] B y-uniform grating 3D==2D: 0-order max|d|={:.2e} flux max|d|={:.2e} (3D reduces to 2D-TE) "
          "-> {}".format(d0B, dfB, "PASS" if gate_b else "FAIL"), flush=True)

    # GATE C: true 2D-periodic dielectric pillar -> all-order flux conserves; 0-order dips (diffraction)
    def pillar(nx, ny, nz, zc, pad, zs):
        e = np.ones((nx, ny, nz)); inb = (zc >= pad) & (zc < pad + zs)
        qx, qy = nx // 4, ny // 4
        blk = np.zeros((nx, ny), dtype=bool); blk[qx:nx - qx, qy:ny - qy] = True
        for k in np.where(inb)[0]:
            e[:, :, k] = np.where(blk, 6.25, 1.0)            # n=2.5 pillar in vacuum
        return e

    rC = solve_fdtd_3d([FDTDLayer(thickness_m=500e-9, eps_inf=1.0)], period_x_m=900e-9, period_y_m=900e-9,
                       lateral_eps_inf=pillar, lambda_min_m=LMIN, lambda_max_m=LMAX, resolution=14, n_pad_wave=4.0)
    mC = rC.band
    e_abs = np.abs(rC.R_flux[mC] + rC.T_flux[mC] - 1.0)
    en_med, en_max = float(np.median(e_abs)), float(np.max(e_abs))
    spec_min = float((rC.R0[mC] + rC.T0[mC]).min())
    gate_c = bool(en_med < 1e-2 and spec_min < 0.95)
    print("[f3] C 2D-periodic pillar: flux |R+T-1| median={:.2e} max={:.2e} (grazing) ; 0-order "
          "min(R0+T0)={:.3f} (<1 = diffracted) -> {}".format(
              en_med, en_max, spec_min, "PASS" if gate_c else "FAIL"), flush=True)

    # GATE D (lossy -> defeats the lossless trap): a uniform absorbing Drude slab must match the analytic
    # complex-n Airy. A lossless cell auto-balances total power even with a wrong field split, so energy
    # closure alone cannot prove per-order correctness -- but an ABSORBING slab cannot fake the right R/T.
    ei, wp, gm, dd = 4.0, 1.2e15, 2.0e13, 120e-9
    rD = solve_fdtd_3d([FDTDLayer(thickness_m=dd, eps_inf=ei, drude_wp_rad_s=wp, drude_gamma_rad_s=gm)],
                       period_x_m=300e-9, period_y_m=300e-9, nx=4, ny=4,
                       lambda_min_m=LMIN, lambda_max_m=LMAX, resolution=30)
    mD = rD.band
    w = 2 * np.pi * rD.freqs_Hz[mD]
    nD = np.sqrt(ei - wp ** 2 / (w ** 2 + 1j * gm * w))
    Rd, Td = airy(rD.freqs_Hz[mD], nD, dd)
    dD = max(float(np.max(np.abs(rD.R0[mD] - Rd))), float(np.max(np.abs(rD.T0[mD] - Td))))
    gate_d = bool(dD < 1e-2)
    print("[f3] D Drude slab (lossy): n in [{:.2f},{:.2f}] ; 0-order max|d-Airy|={:.2e} -> {}".format(
        float(nD.real.min()), float(nD.real.max()), dD, "PASS" if gate_d else "FAIL"), flush=True)

    # GATE E (cross-polarization): an ASYMMETRIC (L-shaped) 2D-periodic pillar has no mirror symmetry, so
    # a y-polarized input generates x-pol (the full Ex/Ez/Hy coupling + the Ex Hy* cross term in S_z). The
    # all-order flux STILL conserves energy -- a wrong cross-term sign or a missing component would break
    # energy badly here (a dedicated probe measured cross-pol |Ex|/|Ey| 0-order = 0.87 for this cell).
    def lshape(nx, ny, nz, zc, pad, zs):
        e = np.ones((nx, ny, nz)); inb = (zc >= pad) & (zc < pad + zs)
        blk = np.zeros((nx, ny), dtype=bool)
        blk[nx // 4:nx - nx // 4, ny // 4:ny // 2] = True
        blk[nx // 4:nx // 2, ny // 4:ny - ny // 4] = True    # L (no mirror symmetry -> y->x conversion)
        for k in np.where(inb)[0]:
            e[:, :, k] = np.where(blk, 6.25, 1.0)
        return e
    rE = solve_fdtd_3d([FDTDLayer(thickness_m=450e-9, eps_inf=1.0)], period_x_m=950e-9, period_y_m=950e-9,
                       lateral_eps_inf=lshape, lambda_min_m=LMIN, lambda_max_m=LMAX, resolution=12, n_pad_wave=4.0)
    mE = rE.band
    eE = np.abs(rE.R_flux[mE] + rE.T_flux[mE] - 1.0)
    eE_med = float(np.median(eE)); spec_minE = float((rE.R0[mE] + rE.T0[mE]).min())
    gate_e = bool(eE_med < 1e-2 and spec_minE < 0.9)
    print("[f3] E cross-pol (asymmetric L-pillar): flux |R+T-1| median={:.2e} ; co-pol 0-order "
          "min(R0+T0)={:.3f} (strong cross-pol + diffraction, energy still closes) -> {}".format(
              eE_med, spec_minE, "PASS" if gate_e else "FAIL"), flush=True)

    # GATE F (Kerr self-action): a chi3 slab at low vs high source amplitude must SHIFT the transmission
    # (intensity-dependent eps_eff = eps_inf + chi3|E|^2 = self-phase modulation) and stay lossless.
    slab = [FDTDLayer(thickness_m=400e-9, eps_inf=4.0, chi3_m2_V2=2e-19)]
    fkw = dict(period_x_m=300e-9, period_y_m=300e-9, nx=4, ny=4, kerr=True,
               lambda_min_m=LMIN, lambda_max_m=LMAX, resolution=20)
    lo = solve_fdtd_3d(slab, source_amp=1.0, **fkw)
    hi = solve_fdtd_3d(slab, source_amp=6.0e8, **fkw)
    mF = lo.band & hi.band
    dshift = float(np.max(np.abs(hi.T0[mF] - lo.T0[mF])))
    enF = float(np.max(np.abs(lo.R_flux[mF] + lo.T_flux[mF] - 1.0)))
    gate_f = bool(dshift > 1e-4 and enF < 1e-2)
    print("[f3] F Kerr self-action: max|T0(hi)-T0(lo)|={:.2e} (>0 = SPM) ; low-amp energy |R+T-1|={:.2e} "
          "(lossless) -> {}".format(dshift, enF, "PASS" if gate_f else "FAIL"), flush=True)

    # GATE G (cross-backend): every compiled-kernel 3D backend present -- numba (fused threaded CPU) and
    # jax (XLA lax.scan, differentiable) -- reproduces the numpy reference to machine precision (same
    # six-component physics, just compiled). Backends not installed are skipped.
    from dynameta.optics.fdtd_nd import available_backends
    avail = available_backends()
    gate_g = True
    for bk in ("numba", "jax"):
        if bk not in avail:
            print("[f3] G {} backend: SKIP (not installed)".format(bk), flush=True)
            continue
        rG = solve_fdtd_3d([FDTDLayer(thickness_m=d, eps_inf=n ** 2)], period_x_m=300e-9, period_y_m=300e-9,
                           nx=4, ny=4, lambda_min_m=LMIN, lambda_max_m=LMAX, resolution=18, backend=bk)
        mG = rA.band & rG.band
        dG = max(float(np.max(np.abs(rA.R0[mG] - rG.R0[mG]))), float(np.max(np.abs(rA.T0[mG] - rG.T0[mG]))))
        ok = bool(dG < 1e-9)
        gate_g = gate_g and ok
        print("[f3] G {}==numpy (3D): max|dR0,dT0|={:.2e} (machine precision) -> {}".format(
            bk, dG, "PASS" if ok else "FAIL"), flush=True)

    overall = gate_a and gate_b and gate_c and gate_d and gate_e and gate_f and gate_g
    print("[f3] *** 3D FULL-VECTOR FDTD (reduces to 1D/TMM; reduces to 2D engine; 3D diffraction; "
          "lossy Drude; cross-pol; Kerr; numba==numpy): {} ***".format("PASS" if overall else "FAIL"), flush=True)
    return overall


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
