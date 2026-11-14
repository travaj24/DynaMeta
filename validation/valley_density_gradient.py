"""v0.15.1 oracle: the VALLEY-RESOLVED electron density gradient (physics_density_gradient.setup_bipolar_dg(
electron_valleys=density_gradient.SI100_ELECTRON_VALLEYS)) against the anisotropic Schrodinger-Poisson
inversion oracle, the Si(100) calibration of gamma, and the convergence matrix. Audit note:
docs/audit/2026-11-10-v0.15.1-valley-resolved-dg.md. Harness and oracle: validation/bipolar_density_gradient.py
(mos_qe(valleys=...), sp_inversion(valleys=sp_valleys(...))). Every number here is printed; the gates are:

GATE VA (the closure path; p-Si N_A = 1e23 / 3e23 / 1e24 m^-3, psi_s - psi_b = 1.00 .. 1.30 V in 50 mV steps,
        Fermi-Dirac, GAMMA_WALL, electrons closed; the SP with the SAME Si(100) valleys, DOS-consistent):
        VA-1 the v0.15 GATE G tolerances, registered in advance for the DG-vs-SP comparison (N_inv 10 %,
             centroid 15 %), over N_inv >= 1e16 m^-2 (the range the research note stated its band for);
        VA-2 characterization pin (measured 2026-11-10, +- 1 point, ~40x the default mesh's discretization
             error): N_inv error in [+1.86, +8.87] % and centroid in [-5.23, -0.20] % for N_inv >= 1e16;
        VA-3 the valley-resolved error is smaller than the isotropic model's (vs the same oracle) at EVERY
             point of the matrix (the isotropic model is the only option of e0fa0bb);
        reported: the 2-fold share (DG vs SP), the isotropic model against its own isotropic SP, gamma = 1.
GATE VC (the calibration, density_gradient.GAMMA_SI100_SP_CALIBRATED): direct runs at the constant over the same
        matrix; its comment's stated residuals (N_inv -1.33 .. +0.93 %, centroid -2.75 .. +1.51 %, all 21 points;
        pinned rounded outward to 0.05 point) must hold; gamma*(N_inv) per point is reported (cubic spline over
        gamma = 1.75 .. 2.05).
GATE VG (G1-3 with the valley sets): every 50 mV step to 1.3 V in <= 12 Newton iterations, N_A 3e23 / 1e24,
        gamma 0.5 / 1 / GAMMA_WALL, hole DG + electron sets, both carriers closed, double (REL 1e-5); the
        isotropic model's iteration counts are printed next to it.

Run: python -m validation.valley_density_gradient
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import validation.bipolar_density_gradient as V  # noqa: E402
from dynameta.carriers.density_gradient import (GAMMA_SI100_SP_CALIBRATED,  # noqa: E402
                                                SI100_ELECTRON_VALLEYS as SI)

NAS = (1.0e23, 3.0e23, 1.0e24)
DELTAS = tuple(np.round(np.arange(1.0, 1.3001, 0.05), 10))
VA1_NINV_MIN = 1.0e16                       # m^-2
VA2_PIN_NINV = (0.0186, 0.0887)             # measured +2.86 .. +7.87 %, +- 1 point
VA2_PIN_CENTROID = (-0.0523, -0.0020)       # measured -4.23 .. -1.20 %, +- 1 point
# GAMMA_SI100_SP_CALIBRATED's stated residuals (measured -1.33 .. +0.93 % / -2.75 .. +1.51 %, rounded outward to
# 0.05 point -- a characterization pin of the fit, not a tolerance on physics)
VC_NINV, VC_CENTROID = (-0.0135, 0.0095), (-0.0280, 0.0155)
VG_MAX_ITS = 12


def _sp(na, d):
    s = V.sp_inversion(na, d, valleys=V.sp_valleys(SI))
    sb = [float(np.sum(x[1])) for x in s["subbands"]]
    return s["N_inv"], s["centroid"], sb[0] / sum(sb), s["converged"]


def gate_va():
    ok = True
    gw = V.gamma_wall()
    rows = []
    t_v = t_i = 0.0
    for na in NAS:
        sp = {d: _sp(na, d) for d in DELTAS}
        spi = {d: V.sp_inversion(na, d) for d in DELTAS}
        t = time.time()
        vw = V.mos_qe(na, DELTAS, fd=True, dg_gamma=gw, valleys=SI, tag="va_w")
        t_v += time.time() - t
        v1 = V.mos_qe(na, DELTAS, fd=True, dg_gamma=1.0, valleys=SI, tag="va_1")
        t = time.time()
        iw = V.mos_qe(na, DELTAS, fd=True, dg_gamma=gw, tag="va_i")
        t_i += time.time() - t
        for d, a, b, c in zip(DELTAS, vw, v1, iw):
            n_sp, c_sp, s_sp, conv = sp[d]
            ok &= bool(conv) and bool(spi[d]["converged"])
            e = lambda r, nn=n_sp, cc=c_sp: (V.mos_sheet(r)[0] / nn - 1.0, V.mos_sheet(r)[1] / cc - 1.0)  # noqa: E731
            ii = (V.mos_sheet(c)[0] / spi[d]["N_inv"] - 1.0, V.mos_sheet(c)[1] / spi[d]["centroid"] - 1.0)
            rows.append(dict(na=na, d=d, n_sp=n_sp, c_sp=c_sp, s_sp=s_sp, w=e(a), one=e(b), iso=e(c), iso_iso=ii,
                             share=V.valley_shares(a)[0], its=max(a["its"]), its_iso=max(c["its"])))
    print("[vdg] GATE VA: valley-resolved DG (FD, closure) vs the DOS-consistent anisotropic SP", flush=True)
    print("[vdg]   N_A    dpsi  SP N_inv  SP c[nm] SP 2f | GAMMA_WALL N/c (2f)       | gamma=1 N/c     | "
          "isotropic@GW vs same SP | iso@GW vs iso SP | its v/iso", flush=True)
    for r in rows:
        print("[vdg]  {:.0e} {:.2f}  {:.3e}  {:.3f}  {:.3f} | {:+6.2%}/{:+6.2%} ({:.3f}) | {:+6.1%}/{:+6.1%} | "
              "{:+6.1%}/{:+6.1%}         | {:+6.1%}/{:+6.1%} | {}/{}".format(
                  r["na"], r["d"], r["n_sp"], r["c_sp"] * 1e9, r["s_sp"], *r["w"], r["share"], *r["one"], *r["iso"],
                  *r["iso_iso"], r["its"], r["its_iso"]), flush=True)
    sel = [r for r in rows if r["n_sp"] >= VA1_NINV_MIN]
    en = [r["w"][0] for r in sel]
    ec = [r["w"][1] for r in sel]
    va1 = max(abs(x) for x in en) <= V.SP_TOL_NINV and max(abs(x) for x in ec) <= V.SP_TOL_CENTROID
    va2 = (VA2_PIN_NINV[0] <= min(en) and max(en) <= VA2_PIN_NINV[1]
           and VA2_PIN_CENTROID[0] <= min(ec) and max(ec) <= VA2_PIN_CENTROID[1])
    va3 = all(abs(r["w"][0]) < abs(r["iso"][0]) for r in rows)
    print("[vdg]   N_inv >= 1e16 m^-2 ({} points): N_inv error {:+.2%} .. {:+.2%}, centroid {:+.2%} .. {:+.2%}; "
          "2-fold share DG - SP {:+.3f} .. {:+.3f}".format(
              len(sel), min(en), max(en), min(ec), max(ec), min(r["share"] - r["s_sp"] for r in sel),
              max(r["share"] - r["s_sp"] for r in sel)), flush=True)
    print("[vdg]   all points: N_inv error {:+.2%} .. {:+.2%}; isotropic model vs the same SP {:+.1%} .. {:+.1%}".format(
        min(r["w"][0] for r in rows), max(r["w"][0] for r in rows), min(r["iso"][0] for r in rows),
        max(r["iso"][0] for r in rows)), flush=True)
    print("[vdg]   ramp time (7 biases, 1-D, per doping, sum of 3): valley {:.1f} s, isotropic {:.1f} s".format(t_v, t_i),
          flush=True)
    print("[vdg]   VA-1 registered GATE G tolerances (10 % / 15 %): {}; VA-2 pin: {}; VA-3 valley < isotropic "
          "everywhere: {}".format(*("PASS" if x else "FAIL" for x in (va1, va2, va3))), flush=True)
    return ok and va1 and va2 and va3


def gate_vc():
    from scipy.interpolate import CubicSpline
    grid = (1.75, 1.80, 1.85, 1.90, 1.95, 2.00, 2.05)
    g = GAMMA_SI100_SP_CALIBRATED
    print("[vdg] GATE VC: GAMMA_SI100_SP_CALIBRATED = {!r}".format(g), flush=True)
    res, gst = [], []
    for na in NAS:
        sp = {d: _sp(na, d) for d in DELTAS}
        rc = V.mos_qe(na, DELTAS, fd=True, dg_gamma=g, valleys=SI, tag="vc")
        scan = [V.mos_qe(na, DELTAS, fd=True, dg_gamma=gg, valleys=SI, tag="vcs") for gg in grid]
        for i, d in enumerate(DELTAS):
            n_sp, c_sp = sp[d][:2]
            N, c = V.mos_sheet(rc[i])
            res.append((N / n_sp - 1.0, c / c_sp - 1.0))
            f = CubicSpline(grid, [V.mos_sheet(s[i])[0] / n_sp - 1.0 for s in scan])
            xs = np.linspace(grid[0], grid[-1], 3001)
            fx = f(xs)
            gst.append(float(xs[np.argmin(np.abs(fx))]) if fx.min() < 0.0 < fx.max() else float("nan"))
            print("[vdg]   {:.0e} {:.2f} V  SP N_inv {:.3e}: at the constant N_inv {:+.2%} centroid {:+.2%}; "
                  "gamma*(N_inv) {:.3f}".format(na, d, n_sp, res[-1][0], res[-1][1], gst[-1]), flush=True)
    en, ec = [r[0] for r in res], [r[1] for r in res]
    ok = VC_NINV[0] <= min(en) and max(en) <= VC_NINV[1] and VC_CENTROID[0] <= min(ec) and max(ec) <= VC_CENTROID[1]
    print("[vdg]   N_inv {:+.2%} .. {:+.2%}, centroid {:+.2%} .. {:+.2%} (stated {:+.2%} .. {:+.2%} / {:+.2%} .. "
          "{:+.2%}); gamma*(N_inv) {:.3f} .. {:.3f}: {}".format(min(en), max(en), min(ec), max(ec), *VC_NINV,
                                                               *VC_CENTROID, np.nanmin(gst), np.nanmax(gst),
                                                               "PASS" if ok else "FAIL"), flush=True)
    return ok


def gate_vg():
    deltas = tuple(np.round(np.arange(0.05, 1.3001, 0.05), 10))
    ok = True
    for na in (3.0e23, 1.0e24):
        for g in (0.5, 1.0, V.gamma_wall()):
            its = {}
            for kind, vv in (("valley", SI), ("iso", None)):
                t = time.time()
                r = V.mos_qe(na, deltas, dg_gamma=g, dg_holes=True, close=("electrons", "holes"), valleys=vv,
                             tag="vg" + kind)
                its[kind] = (max(max(x["its"]) for x in r), time.time() - t)
            ok &= its["valley"][0] <= VG_MAX_ITS
            print("[vdg] GATE VG  N_A {:.0e} gamma {:.3f}: max Newton iterations per 50 mV step valley {} ({:.1f} s), "
                  "isotropic {} ({:.1f} s)".format(na, g, its["valley"][0], its["valley"][1], its["iso"][0],
                                                   its["iso"][1]), flush=True)
    return ok


def main():
    print("[vdg] === v0.15.1 valley-resolved electron density gradient (Si(100)) ===", flush=True)
    t0 = time.time()
    ok = True
    for name, fn in (("VA", gate_va), ("VC", gate_vc), ("VG", gate_vg)):
        t = time.time()
        r = fn()
        print("[vdg] GATE {}: {} ({:.0f} s)".format(name, "PASS" if r else "FAIL", time.time() - t), flush=True)
        ok &= r
    print("[vdg] *** VALLEY-RESOLVED DG: {} *** ({:.0f} s)".format("PASS" if ok else "FAIL", time.time() - t0),
          flush=True)
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
