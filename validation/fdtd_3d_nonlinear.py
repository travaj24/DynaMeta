"""3D chi2/Raman/gain FDTD oracle (deferred-item completion).

GATE A (reduces to 2D, machine): a laterally-uniform stack driven y-polarized at the SAME dt --
        every x,y-uniform 3D column evolves exactly like the 2D-TE kernel (Ex = Ez = 0, the
        Raman |E|^2 drive and per-component chi2/gain reduce to the scalar model), so the 3D
        exit probe equals the 2D one to the float64 floor with chi2 + Raman + gain ALL active.
        The 2D kernel itself carries the R15/R20 closed-form oracles, so equality transfers them.
GATE B (public plumbing): solve_fdtd_3d with each nonlinearity active runs and CHANGES the
        spectrum vs the zero run (the layer fields reach the kernel); zero fields ARRAY-EQUAL.
GATE C (guards): backend='numba' raises with an active 3D nonlinearity; Raman/gain without
        their resonance parameters raise.

Run: python -m validation.fdtd_3d_nonlinear
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dynameta.constants import C_LIGHT, M_E, Q_E
from dynameta.optics.fdtd import FDTDLayer
from dynameta.optics.fdtd_nd import _cpml_z, _run_2d_te, _run_3d, solve_fdtd_3d

N_MED = np.sqrt(2.0)
W0 = 2.0 * np.pi * 2.5e14


def main():
    print("[n3] === 3D chi2/Raman/gain FDTD ===", flush=True)
    ok = True

    # ---- GATE A: kernel-level 3D == 2D with all nonlinearities active ----
    dz = 12e-9
    n_pad, n_str = 50, 30
    nz = 2 * n_pad + n_str
    dx = 4.0 * dz
    dt = 0.4 / (C_LIGHT * np.sqrt(1.0 / dx ** 2 + 1.0 / dx ** 2 + 1.0 / dz ** 2))  # 3D CFL, shared
    tau = 50e-15
    t0 = 6.0 * tau
    nsteps = int(round((2.0 * t0 + 150e-15) / dt))
    t = np.arange(nsteps) * dt
    src = 3.0e8 * np.exp(-((t - t0) / tau) ** 2) * np.cos(W0 * (t - t0))
    cpml = _cpml_z(nz, dz, dt, 12, N_MED, N_MED)
    k_src, k_pL, k_pR = 14, 18, nz - 14

    def win2(val):
        a = np.zeros((4, nz)); a[:, n_pad:n_pad + n_str] = val
        return a

    def win3(val):
        a = np.zeros((4, 4, nz)); a[:, :, n_pad:n_pad + n_str] = val
        return a

    W_R, g_R = 2.0 * np.pi * 1.0e13, 2.0 * np.pi * 1.0e12
    dwg = 2.0 * np.pi * 2.0e13
    kdn = (Q_E ** 2 / M_E) * 2.0e23

    def coeffs(win):
        den_r = 1.0 + g_R * dt / 2.0
        raman = (win(1.0) * 0 + (2.0 - W_R ** 2 * dt ** 2) / den_r,
                 win(1.0) * 0 + (g_R * dt / 2.0 - 1.0) / den_r,
                 win((W_R ** 2 * dt ** 2) / den_r) * 0 + win(W_R ** 2 * dt ** 2 / den_r),
                 win(1.0e-22))
        den_g = 1.0 + dwg * dt / 2.0
        gain = (win(1.0) * 0 + (2.0 - W0 ** 2 * dt ** 2) / den_g,
                win(1.0) * 0 + (dwg * dt / 2.0 - 1.0) / den_g,
                win(-kdn * dt ** 2 / den_g))
        return win(2.0e-12), raman, gain                      # chi2, raman, gain

    chi2_2, ram_2, gn_2 = coeffs(win2)
    chi2_3, ram_3, gn_3 = coeffs(win3)
    eps2 = np.full((4, nz), N_MED ** 2)
    eps3 = np.full((4, 4, nz), N_MED ** 2)
    z2 = np.zeros((4, nz)); z3 = np.zeros((4, 4, nz))

    _, _, ey2, _ = _run_2d_te(eps2, z2, z2, z2, dx, dz, dt, nsteps, k_src, k_pL, k_pR, src,
                              cpml, np, None, chi2=chi2_2, raman=ram_2, gain=gn_2)
    out3 = _run_3d(eps3, z3, z3, z3, dx, dx, dz, dt, nsteps, k_src, k_pL, k_pR, src, cpml,
                   np, None, chi2=chi2_3, raman=ram_3, gain=gn_3)
    ey3 = out3[5].mean(axis=(1, 2))                           # eyR plane -> x,y mean
    dA = float(np.max(np.abs(ey3 - ey2.mean(axis=1)))) / float(np.max(np.abs(ey2)))
    g_a = bool(dA < 1e-12)
    ok = ok and g_a
    print("[n3] GATE A: 3D exit probe == 2D (chi2+Raman+gain all active), rel {:.1e} -> {}"
          .format(dA, "PASS" if g_a else "FAIL"), flush=True)

    # ---- GATE B: public plumbing ----
    # source_amp 1e8 V/m: the field-QUADRATIC chi2/Raman effects are ~1e-21 at the default
    # 1 V/m drive (the first draft of this gate measured exactly 0.0 for that reason); the gain
    # line is linear in the field and registers at any amplitude.
    kw = dict(period_x_m=80e-9, period_y_m=80e-9, lambda_min_m=1.0e-6, lambda_max_m=1.4e-6,
              resolution=10, backend="numpy", source_amp=1.0e8)
    base = solve_fdtd_3d([FDTDLayer(150e-9, eps_inf=2.0)], **kw)
    zero = solve_fdtd_3d([FDTDLayer(150e-9, eps_inf=2.0, chi2_m_V=0.0, gain_dN_m3=0.0)], **kw)
    effects = {}
    for nm, f in (("chi2", dict(chi2_m_V=5e-12)),
                  ("raman", dict(raman_chi3_m2_V2=1e-21, raman_w_rad_s=2e13 * 2 * np.pi,
                                 raman_gamma_rad_s=2e12 * 2 * np.pi)),
                  ("gain", dict(gain_w_rad_s=W0, gain_dw_rad_s=dwg,
                                gain_kappa_C2_kg=Q_E ** 2 / M_E, gain_dN_m3=5e24))):
        r = solve_fdtd_3d([FDTDLayer(150e-9, eps_inf=2.0, **f)], **kw)
        m = r.band & base.band
        effects[nm] = float(np.max(np.abs(r.R0[m] - base.R0[m])))
    g_b = bool(np.array_equal(base.R0, zero.R0)
               and all(np.isfinite(v) and v > 1e-12 for v in effects.values()))
    ok = ok and g_b
    print("[n3] GATE B: zero fields ARRAY-EQUAL; active fields change R0 ({}) -> {}".format(
        ", ".join("{} {:.1e}".format(k, v) for k, v in effects.items()),
        "PASS" if g_b else "FAIL"), flush=True)

    # ---- GATE C: guards ----
    guards = 0
    try:
        solve_fdtd_3d([FDTDLayer(150e-9, eps_inf=2.0, chi2_m_V=1e-12)],
                      **{**kw, "backend": "numba"})
    except NotImplementedError:
        guards += 1
    try:
        solve_fdtd_3d([FDTDLayer(150e-9, eps_inf=2.0, raman_chi3_m2_V2=1e-21)], **kw)
    except ValueError:
        guards += 1
    g_c = bool(guards == 2)
    ok = ok and g_c
    print("[n3] GATE C: numba-with-nonlinearity + missing-resonance guards raise -> {}".format(
        "PASS" if g_c else "FAIL"), flush=True)

    print("[n3] *** 3D NONLINEAR FDTD: {} ***".format("PASS" if ok else "FAIL"), flush=True)
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
