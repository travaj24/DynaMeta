"""3D full-vector FDTD oracle: optics.fdtd_nd.solve_fdtd_3d is a doubly-periodic (x AND y) 6-component
Yee solver (Bloch-periodic x/y at normal incidence, CFS-CPML + PEC in z) -- the 2D-TE engine is its
d/dy=0, {Ey,Hx,Hz} reduction. Five gates here establish it is correct (the last = backend
equivalence); the two genuinely 2D-periodic gates live in their own scripts (see the note below).

GATE A (reduces to TMM/1D): a laterally-UNIFORM non-dispersive slab -- the specular 0-order R0/T0 AND
        the all-order Poynting flux R_flux/T_flux both == the analytic Airy R/T (so the full vector 3D
        engine reduces EXACTLY to the 1D solver / TMM), and R+T = 1. Run on a tiny 4x4 lateral grid
        (the result is laterally uniform, so nx,ny are irrelevant -> fast).
GATE B (reduces to the 2D engine): a y-UNIFORM binary grating (varies in x only) -- with a y-polarized
        source and d/dy=0 the problem is pure 2D-TE, so solve_fdtd_3d must reproduce the validated
        solve_fdtd_2d R0/T0 and flux (small residual from the 3D CFL dt differing via 1/dy^2).
GATE D (lossy -> defeats the lossless trap): a uniform absorbing Drude slab == the analytic complex-n
        Airy. A lossless cell auto-balances total power even with a wrong field split, so energy closure
        alone cannot prove correctness -- an absorbing slab cannot fake the right R/T.

SPLIT 2026-09-01. GATE C (2D-periodic pillar diffraction) and GATE E (asymmetric-cell
cross-polarization) moved UNCHANGED to validation/fdtd_3d_pillar_diffraction.py and
validation/fdtd_3d_crosspol.py. They are true 2D-periodic cells carrying a full lateral grid and
cost 1191 s and 690 s of this script's original 2133 s, where the gates that remain here run on a
4x4 lateral grid in tens of seconds. Because run_all's PER_SCRIPT_TIMEOUT_S is GLOBAL, one
2133 s script set the cap for all ~215: at the ~1.9x runner variance measured across the nightly
shards it needs ~4000 s, and a 4000 s cap cannot detect a hang anywhere else. This script had
already timed out at 1800 s and again at 2700 s (nightly 33445577932) while PASSING at 1935 s in
between -- a cap sitting inside the noise, not a broken gate. No gate was weakened to fit a
budget; they were separated so each is decided by its physics.

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

# GATE A convergence-gate constant, PER GEOMETRY (2026-11-03 decision): the worst second-order
# constant of THIS fixture's own stack (300 nm n = 2 slab, band LMIN-LMAX, res 18, nx = ny = 4) over
# the 8 sub-cell offsets u = k/8, k = 0..7 (incl. u = 0 and 1/2) of its front face (fdtd_nd.grid.subcell_offset_pads, base
# n_pad_wave 4) AND over the band: C = max of max(|dR0|,|dT0|,|dR_flux|,|dT_flux|) / X(f)^2 with
# X = n_max k0(f) dz = 2 pi LMIN f / (res c) (fdtd_nd.grid.second_order_constant). No safety factor.
# Regenerate: python -m validation.fdtd_3d_reduces --regen-cgeom   (measured 2026-11-03, numpy = the fixture config).
# Part (a) below is a REGRESSION bound; the physics gate is part (b), the observed order.
C_GEOM_3D = 0.08735397979210943   # 2026-11-03, u = k/8: 0.087354 0.079760 0.059720 0.035424 0.044972 0.035414 0.059719 0.079699
# The fixture IS the u = 0 member of its own sweep, so (a) sits at ratio 1 by construction; allow
# cross-platform floating-point round-off only (a physics regression moves the ratio by percent).
FP_ROUNDOFF = 1e-6


def _gate_a_err(res, n_pad_wave=4.0, nx=4, backend="numpy", z_subcell="average"):
    """(err(f), X(f), R+T-1 max, 0-order max err, flux max err, result) of the GATE A slab vs Airy."""
    n, d = 2.0, 300e-9
    r = solve_fdtd_3d([FDTDLayer(thickness_m=d, eps_inf=n ** 2)], period_x_m=300e-9, period_y_m=300e-9,
                      nx=nx, ny=nx, lambda_min_m=LMIN, lambda_max_m=LMAX, resolution=res, n_pad_wave=n_pad_wave,
                      backend=backend, z_subcell=z_subcell)
    m = r.band
    Ra, Ta = airy(r.freqs_Hz[m], n, d)
    e0 = np.maximum(np.abs(r.R0[m] - Ra), np.abs(r.T0[m] - Ta))
    ef = np.maximum(np.abs(r.R_flux[m] - Ra), np.abs(r.T_flux[m] - Ta))
    X = 2.0 * np.pi * LMIN * r.freqs_Hz[m] / (res * C)
    en = float(np.max(np.abs(r.R_flux[m] + r.T_flux[m] - 1.0)))
    return np.maximum(e0, ef), X, en, float(e0.max()), float(ef.max()), r


def regen_c_geom(res=18):
    """The GATE A constant from the registration sweep of THIS geometry (see C_GEOM_3D)."""
    from dynameta.optics.fdtd_nd.grid import layer_band_index_max, second_order_constant, subcell_offset_pads
    L = [FDTDLayer(thickness_m=300e-9, eps_inf=4.0)]
    n_max = layer_band_index_max(L, 2 * np.pi * np.linspace(C / LMAX, C / LMIN, 9), (1.0, 1.0))
    dz = LMIN / (res * n_max)
    cs = []
    for npw in subcell_offset_pads(dz, LMAX, 8, 4.0):
        err, X, _en, _a, _b, _r = _gate_a_err(res, npw, nx=4, backend="numpy")   # = the fixture config
        cs.append(second_order_constant(err, X))
    return max(cs), cs


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
    # 2026-11-03: the single-resolution pin (d0, df < 5e-3) is a CONVERGENCE gate against the
    # true-geometry Airy. (a) REGRESSION bound: at THIS res, per frequency, error <= C_GEOM_3D * X(f)^2
    # with this geometry's own registration-sweep constant (see C_GEOM_3D). (b) PHYSICS: observed
    # order >= 1.7 between res 18 and 36, on a cheaper probe of the same engine (nx = ny = 2, numba;
    # laterally uniform, so nx is irrelevant) -- the 2x-res run of THIS config costs ~16x its ~21 s.
    n, d = 2.0, 300e-9
    err_f, X, en, d0, df, rA = _gate_a_err(18)            # rA: also the GATE G numpy reference
    ratio_a = float(np.max(err_f / (C_GEOM_3D * X ** 2)))
    e_lo = float(np.max(_gate_a_err(18, nx=2, backend="numba")[0]))
    e_hi = float(np.max(_gate_a_err(36, nx=2, backend="numba")[0]))
    order_a = float(np.log2(e_lo / e_hi))
    gate_a = bool(ratio_a <= 1.0 + FP_ROUNDOFF and order_a >= 1.7 and en < 1e-3)
    print("[f3] A uniform: 0-order max|d-Airy|={:.2e} flux max|d-Airy|={:.2e} max|R+T-1|={:.2e}; (a) max "
          "err/(C X^2)={:.3f} (C_geom={}) ; (b) order {:.2f} ({:.2e} -> {:.2e}, nx=ny=2 probe) -> {}".format(
              d0, df, en, ratio_a, C_GEOM_3D, order_a, e_lo, e_hi, "PASS" if gate_a else "FAIL"), flush=True)

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
    # layer eps_inf sized to the LATERAL pattern max (4.0): the painter overwrites the eps
    # everywhere, but dz/n_max derive from `layers` -- the C3-3 guard now (correctly)
    # refuses the old eps_inf=1.0 fixture, whose dz was silently 2x under-resolved
    r2 = solve_fdtd_2d([FDTDLayer(thickness_m=600e-9, eps_inf=4.0)], period_x_m=PX, lateral_eps_inf=lat2, **kw)
    r3 = solve_fdtd_3d([FDTDLayer(thickness_m=600e-9, eps_inf=4.0)], period_x_m=PX, period_y_m=4000e-9,
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

    # GATE D (lossy -> defeats the lossless trap): a uniform absorbing Drude slab must match the analytic
    # complex-n Airy. A lossless cell auto-balances total power even with a wrong field split, so energy
    # closure alone cannot prove per-order correctness -- but an ABSORBING slab cannot fake the right R/T.
    # resolution=60 == the validated 2D GATE B grid for this same slab: any coarser and the 120nm slab
    # quantizes badly in dz (band-max n_max sizing -> dz=21.6nm at res 30 -> 5 cells = 108nm effective),
    # so the gate would measure the staircase thickness error (~2e-2), not the lossy-Drude physics
    # (the engine matches Airy at the discretized thickness to ~3e-3 even on that coarse grid).
    ei, wp, gm, dd = 4.0, 1.2e15, 2.0e13, 120e-9
    rD = solve_fdtd_3d([FDTDLayer(thickness_m=dd, eps_inf=ei, drude_wp_rad_s=wp, drude_gamma_rad_s=gm)],
                       period_x_m=300e-9, period_y_m=300e-9, nx=4, ny=4,
                       lambda_min_m=LMIN, lambda_max_m=LMAX, resolution=60)
    mD = rD.band
    w = 2 * np.pi * rD.freqs_Hz[mD]
    nD = np.sqrt(ei - wp ** 2 / (w ** 2 + 1j * gm * w))
    Rd, Td = airy(rD.freqs_Hz[mD], nD, dd)
    dD = max(float(np.max(np.abs(rD.R0[mD] - Rd))), float(np.max(np.abs(rD.T0[mD] - Td))))
    gate_d = bool(dD < 1e-2)
    print("[f3] D Drude slab (lossy): n in [{:.2f},{:.2f}] ; 0-order max|d-Airy|={:.2e} -> {}".format(
        float(nD.real.min()), float(nD.real.max()), dD, "PASS" if gate_d else "FAIL"), flush=True)

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

    overall = gate_a and gate_b and gate_d and gate_f and gate_g
    print("[f3] *** 3D FULL-VECTOR FDTD (reduces to 1D/TMM; reduces to 2D engine; lossy Drude; "
          "Kerr; numba==numpy): {} ***".format("PASS" if overall else "FAIL"), flush=True)
    return overall


if __name__ == "__main__":
    if "--regen-cgeom" in sys.argv[1:]:
        c, per = regen_c_geom()
        print("[f3] C_geom (GATE A, res 18, 8 offsets k/8 x band) = {!r}; per offset {}".format(
            c, ", ".join("{:.6f}".format(v) for v in per)))
        raise SystemExit(0)
    raise SystemExit(0 if main() else 1)
