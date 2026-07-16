"""GPU (cupy + numba-cuda) chi2/Raman/gain FDTD kernels vs the CPU reference (the last
guarded backend gap, unblocked 2026-06-10 when CUDA 13.1 + RTX hardware was verified present).

The cupy path is the SAME xp-parameterized reference loop as numpy (equality should be exact
or FMA-floor); the numba-cuda path is the persistent cooperative-groups kernel extended with
the cell-local chi2/Raman/gain recurrences (same physics as _te2d_numba; float64 FMA floor).

GATE A: chi2-only      -- cupy == numpy and numba-cuda == numpy on R0/T0 in-band (< 1e-12).
GATE B: Raman-only     -- same equality.
GATE C: clamped gain   -- same equality.
GATE D: ALL active     -- chi2 + Raman + gain simultaneously, same equality.
GATE E: DYNAMIC gain (gain_dyn, raw kernel path) on cupy == numpy, including the host-side
        output contract (dN_snap / Npop_final come back as NumPy arrays; < 1e-12).

Honest SKIP (exit 0 with a SKIP banner) when no CUDA GPU / cupy is importable -- mirrors
fdtd_numba_cuda.py so a CPU-only box stays green without faking a PASS.

Run: python -m validation.fdtd_gpu_nonlinear
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dynameta.optics.fdtd import FDTDLayer
from dynameta.optics.fdtd_nd import have_numba_cuda, solve_fdtd_2d

KW = dict(period_x_m=100e-9, lambda_min_m=1.0e-6, lambda_max_m=1.4e-6, resolution=16, nx=4)
TOL = 1e-12


def _have_cupy_device():
    try:
        import cupy as cp
        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


def _equal(layers, backends, tag, ok):
    r_ref = solve_fdtd_2d(layers, backend="numpy", **KW)
    m = r_ref.band
    worst = 0.0
    for bk in backends:
        r = solve_fdtd_2d(layers, backend=bk, **KW)
        worst = max(worst, float(np.max(np.abs(r.R0[m] - r_ref.R0[m]))),
                    float(np.max(np.abs(r.T0[m] - r_ref.T0[m]))))
    g = bool(worst < TOL)
    print("[gnl] GATE {}: {} on {}: worst |d| = {:.2e} -> {}".format(
        tag, layers[0].__class__.__name__ and tag, "+".join(backends), worst,
        "PASS" if g else "FAIL"), flush=True)
    return ok and g


def main():
    print("[gnl] === GPU chi2/Raman/gain kernels vs CPU reference ===", flush=True)
    backends = []
    if _have_cupy_device():
        backends.append("cupy")
    if have_numba_cuda():
        backends.append("numba-cuda")
    if not backends:
        print("[gnl] *** SKIP: no CUDA GPU / cupy available -- GPU nonlinear gates not run ***",
              flush=True)
        return True

    ok = True
    ok = _equal([FDTDLayer(150e-9, eps_inf=2.0, chi2_m_V=1e-12)], backends, "A chi2", ok)
    ok = _equal([FDTDLayer(150e-9, eps_inf=2.0, raman_chi3_m2_V2=1e-22,
                           raman_w_rad_s=1.0e14, raman_gamma_rad_s=2.0e13)],
                backends, "B Raman", ok)
    ok = _equal([FDTDLayer(150e-9, eps_inf=2.0, gain_kappa_C2_kg=1e-8, gain_dN_m3=1e22,
                           gain_w_rad_s=1.5e15, gain_dw_rad_s=1.3e14)],
                backends, "C gain", ok)
    ok = _equal([FDTDLayer(150e-9, eps_inf=2.0, chi2_m_V=1e-12, raman_chi3_m2_V2=1e-22,
                           raman_w_rad_s=1.0e14, raman_gamma_rad_s=2.0e13,
                           gain_kappa_C2_kg=1e-8, gain_dN_m3=1e22,
                           gain_w_rad_s=1.5e15, gain_dw_rad_s=1.3e14)],
                backends, "D all-active", ok)

    if "cupy" in backends:
        import cupy as cp
        from dynameta.constants import C_LIGHT, HBAR
        from dynameta.optics.fdtd_nd import cpml_z, run_2d_te
        nx, n_pad, n_str = 4, 40, 24
        nz = 2 * n_pad + n_str
        dz = 10e-9; dx = 4.0 * dz
        dt = 0.5 / (C_LIGHT * np.sqrt(1.0 / dx ** 2 + 1.0 / dz ** 2))
        cpml = cpml_z(nz, dz, dt, 12, 1.0, 1.0)
        w_a, dw = 2.0 * np.pi * 2.5e14, 2.0 * np.pi * 2.0e13
        den = 1.0 + dw * dt / 2.0
        G1 = np.full((nx, nz), (2.0 - w_a ** 2 * dt ** 2) / den)
        G2 = np.full((nx, nz), (dw * dt / 2.0 - 1.0) / den)
        win = np.zeros((nx, nz)); win[:, n_pad:n_pad + n_str] = 1.0
        kapfac = (1.0e-8) * dt ** 2 / den * win
        Wp = 2.0e11 * win
        Npop0 = np.stack([np.where(win > 0, 0.7e25, 1e25), 0.0 * win, 0.29e25 * win,
                          0.01e25 * win])
        nsteps = 600
        t = np.arange(nsteps) * dt
        src = 1.0e3 * np.exp(-((t - 60 * dt) / (20 * dt)) ** 2) * np.cos(w_a * t)
        eps = np.full((nx, nz), 2.0); zeros = np.zeros((nx, nz))
        snap = 400
        out_np, out_cp = {}, {}
        gd_np = (G1, G2, kapfac, Wp, Npop0, 1e-14, 1e-13, 5e-15, HBAR * w_a, snap)
        _, _, eyR_np, _ = run_2d_te(eps, zeros, zeros, zeros, dx, dz, dt, nsteps, 16, 20,
                                     nz - 16, src, cpml, np, None, gain_dyn=gd_np,
                                     gain_dyn_out=out_np)
        gd_cp = tuple(cp.asarray(v) if isinstance(v, np.ndarray) else v for v in gd_np)
        _, _, eyR_cp, _ = run_2d_te(cp.asarray(eps), cp.asarray(zeros), cp.asarray(zeros),
                                     cp.asarray(zeros), dx, dz, dt, nsteps, 16, 20, nz - 16,
                                     cp.asarray(src), cpml, cp, None, gain_dyn=gd_cp,
                                     gain_dyn_out=out_cp)
        d_field = float(np.max(np.abs(cp.asnumpy(eyR_cp) - eyR_np)))
        d_snap = float(np.max(np.abs(out_cp["dN_snap"] - out_np["dN_snap"])))
        d_pop = float(np.max(np.abs(out_cp["Npop_final"] - out_np["Npop_final"])))
        host_ok = isinstance(out_cp["dN_snap"], np.ndarray) and isinstance(
            out_cp["Npop_final"], np.ndarray)
        rel_pop = d_pop / 1e25
        g_e = bool(d_field < TOL and d_snap / 1e25 < TOL and rel_pop < TOL and host_ok)
        ok = ok and g_e
        print("[gnl] GATE E: dynamic gain cupy == numpy: field {:.2e}, dN_snap rel {:.2e}, "
              "Npop rel {:.2e}, host-numpy outputs {} -> {}".format(
                  d_field, d_snap / 1e25, rel_pop, host_ok, "PASS" if g_e else "FAIL"),
              flush=True)

    print("[gnl] *** GPU NONLINEAR KERNELS ({}): {} ***".format(
        "+".join(backends), "PASS" if ok else "FAIL"), flush=True)
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
