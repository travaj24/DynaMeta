"""Validate the JAX backend for the OBLIQUE 2D FDTD (complex-envelope Bloch, s-pol): a single traced
lax.scan time loop that is (1) byte-for-byte equal to the NumPy oblique kernel on R/T and (2)
DIFFERENTIABLE end-to-end, so jax.grad gives d(objective)/d(geometry) straight through the oblique scan
-- gradient-based inverse design AT AN ANGLE (the normal-incidence jax backend already powered the
multi-objective designer; this extends it to oblique).

GATE A: backend='jax' reproduces backend='numpy' R0/T0 to < 1e-10 across the band (2D-TE / s-pol).
GATE B: jax.grad of a transmission-energy objective wrt the slab eps, through run_2d_te_oblique_jax,
        is finite and non-zero (the adjoint flows through the complex-envelope time loop).
GATE C: the 2D-TM (p-pol) JAX kernel == NumPy across angles {0,20,40,55} deg (dispersive Drude) < 1e-9.
GATE D: the full-vector 3D oblique JAX kernel == NumPy across conical (angle,azimuth)
        {(0,0),(20,0),(20,45),(30,90)} deg (exercises kx AND ky) < 1e-9.
GATE E: jax.grad through BOTH the 2D-TM and the 3D oblique kernels is finite + nonzero
        (differentiable inverse design at angle for p-pol and the full-vector 3D path).

Skipped (exit 42 = the run_all SKIP category, audit C6-6) if JAX is not installed. Run: python -m validation.fdtd_oblique_jax
"""
import os
import sys
import math

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dynameta.optics.fdtd_nd import (FDTDLayer, solve_fdtd_2d_oblique, solve_fdtd_3d_oblique,
                                     have_jax, cpml_z)
from dynameta.constants import C_LIGHT


def main():
    print("[t] === JAX oblique 2D FDTD: jax==numpy + differentiable ===", flush=True)
    if not have_jax():
        print("[t] JAX not installed -> SKIP (exit 42; run_all counts it separately, audit C6-6)", flush=True)
        raise SystemExit(42)
    L = [FDTDLayer(thickness_m=250e-9, eps_inf=4.0)]
    kw = dict(period_x_m=320e-9, angle_deg=25.0, lambda_min_m=1.2e-6, lambda_max_m=1.8e-6,
              resolution=22, nx=6)
    rn = solve_fdtd_2d_oblique(L, backend="numpy", **kw)
    rj = solve_fdtd_2d_oblique(L, backend="jax", **kw)
    b = rn.band
    dR = float(np.max(np.abs(rn.R0[b] - rj.R0[b]))); dT = float(np.max(np.abs(rn.T0[b] - rj.T0[b])))
    g_a = (dR < 1e-10) and (dT < 1e-10)
    print("[t] A jax vs numpy: max|dR0|={:.2e} max|dT0|={:.2e} -> {}".format(
        dR, dT, "OK" if g_a else "FAIL"), flush=True)

    import jax, jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)
    from dynameta.optics.fdtd_nd import run_2d_te_oblique_jax
    nx, nz = 6, 400
    dz = 1.2e-6 / (22 * 2.0); dx = 320e-9 / nx
    dt = 0.5 / (C_LIGHT * np.sqrt(1.0 / dx ** 2 + 1.0 / dz ** 2))
    kx = (2.0 * np.pi * 2e14 / C_LIGHT) * np.sin(np.radians(25.0))
    nsteps = 2500
    tg = np.arange(nsteps) * dt
    src = np.exp(-((tg - 300 * dt) / (120 * dt)) ** 2) * np.cos(0.2 * np.arange(nsteps))
    cpml = cpml_z(nz, dz, dt, 12)
    wp = jnp.zeros((nx, nz)); gam = jnp.zeros((nx, nz))

    def loss(eps_slab):
        eps = jnp.ones((nx, nz)).at[:, 150:250].set(eps_slab)
        _eyL, _hxL, eyR, _hxR = run_2d_te_oblique_jax(eps, wp, gam, dx, dz, dt, nsteps, 100, 200, 350,
                                                       jnp.asarray(src), cpml, kx)
        return jnp.sum(jnp.abs(jnp.fft.fft(eyR.mean(axis=1))) ** 2)

    grad = float(jax.grad(loss)(4.0))
    g_b = bool(np.isfinite(grad) and abs(grad) > 0.0)
    print("[t] B d(loss)/d(eps_slab) through the oblique scan = {:.3e} (finite + nonzero) -> {}".format(
        grad, "OK" if g_b else "FAIL"), flush=True)

    # GATE C: 2D-TM (p-pol) jax == numpy across angles
    old = [FDTDLayer(thickness_m=250e-9, eps_inf=4.0, drude_wp_rad_s=1.5e15, drude_gamma_rad_s=1.0e14)]
    dC = 0.0
    for ang in (0.0, 20.0, 40.0, 55.0):
        kwc = dict(period_x_m=300e-9, angle_deg=ang, lambda_min_m=1.4e-6, lambda_max_m=1.6e-6,
                   resolution=12, nx=6, pol="p")
        a = solve_fdtd_2d_oblique(old, backend="numpy", **kwc)
        j = solve_fdtd_2d_oblique(old, backend="jax", **kwc)
        mm = a.band
        dC = max(dC, float(np.max(np.abs(a.R0[mm] - j.R0[mm]))), float(np.max(np.abs(a.T0[mm] - j.T0[mm]))))
    g_c = dC < 1e-9
    print("[t] C 2D-TM jax==numpy over angles 0/20/40/55: max|d|={:.2e} -> {}".format(
        dC, "OK" if g_c else "FAIL"), flush=True)

    # GATE D: 3D oblique jax == numpy across conical (angle, azimuth)
    dD = 0.0
    for ang, az in ((0.0, 0.0), (20.0, 0.0), (20.0, 45.0), (30.0, 90.0)):
        kwd = dict(period_x_m=300e-9, period_y_m=300e-9, angle_deg=ang, azimuth_deg=az,
                   lambda_min_m=1.4e-6, lambda_max_m=1.6e-6, resolution=9, nx=5, ny=5, settle=8.0,
                   n_pad_wave=2.5)
        a = solve_fdtd_3d_oblique(old, backend="numpy", **kwd)
        j = solve_fdtd_3d_oblique(old, backend="jax", **kwd)
        mm = a.band
        dD = max(dD, float(np.max(np.abs(a.R0[mm] - j.R0[mm]))), float(np.max(np.abs(a.T0[mm] - j.T0[mm]))))
    g_d = dD < 1e-9
    print("[t] D 3D oblique jax==numpy over (0,0)/(20,0)/(20,45)/(30,90): max|d|={:.2e} -> {}".format(
        dD, "OK" if g_d else "FAIL"), flush=True)

    # GATE E: jax.grad through the 2D-TM and 3D oblique kernels (finite + nonzero)
    from dynameta.optics.fdtd_nd import run_2d_tm_oblique_jax, run_3d_oblique_jax
    nx2, nz2 = 4, 70
    dz2 = 30e-9; dx2 = 300e-9 / nx2
    dt2 = 0.4 / (C_LIGHT * math.sqrt(1.0 / dx2 ** 2 + 1.0 / dz2 ** 2))
    band2 = np.zeros((nx2, nz2)); band2[:, 28:42] = 1.0
    ns2 = 500; tg2 = np.arange(ns2) * dt2
    src2 = np.exp(-((tg2 - 120 * dt2) / (60 * dt2)) ** 2) * np.cos(2 * np.pi * (C_LIGHT / 1.5e-6) * (tg2 - 120 * dt2))
    cp2 = cpml_z(nz2, dz2, dt2, 10)
    kx2 = (2 * np.pi / 1.5e-6) * math.sin(math.radians(30.0))

    def lossTM(s):
        eps = 1.0 + s * jnp.asarray(band2) * 3.0
        out = run_2d_tm_oblique_jax(eps, jnp.zeros((nx2, nz2)), jnp.zeros((nx2, nz2)), dx2, dz2, dt2,
                                     ns2, 12, 20, 55, src2, cp2, kx2)
        return jnp.sum(jnp.abs(out[2]) ** 2)
    gTM = float(jax.grad(lossTM)(1.0))

    nx3 = ny3 = 3; nz3 = 46
    dz3 = 34e-9; dx3 = dy3 = 300e-9 / nx3
    dt3 = 0.4 / (C_LIGHT * math.sqrt(1.0 / dx3 ** 2 + 1.0 / dy3 ** 2 + 1.0 / dz3 ** 2))
    band3 = np.zeros((nx3, ny3, nz3)); band3[:, :, 18:28] = 1.0
    ns3 = 360; tg3 = np.arange(ns3) * dt3
    src3 = np.exp(-((tg3 - 90 * dt3) / (45 * dt3)) ** 2) * np.cos(2 * np.pi * (C_LIGHT / 1.5e-6) * (tg3 - 90 * dt3))
    cp3 = cpml_z(nz3, dz3, dt3, 8)
    kx3 = (2 * np.pi / 1.5e-6) * math.sin(math.radians(20.0))

    def loss3D(s):
        eps = 1.0 + s * jnp.asarray(band3) * 3.0
        out = run_3d_oblique_jax(eps, jnp.zeros((nx3, ny3, nz3)), jnp.zeros((nx3, ny3, nz3)), dx3, dy3,
                                  dz3, dt3, ns3, 10, 16, 38, src3, cp3, kx3, 0.0, 0.0, 1.0)
        return jnp.sum(jnp.abs(out[3]) ** 2)                        # eyR (co-pol at azimuth 0)
    g3D = float(jax.grad(loss3D)(1.0))
    g_e = bool(np.isfinite(gTM) and abs(gTM) > 0 and np.isfinite(g3D) and abs(g3D) > 0)
    print("[t] E jax.grad: d/ds(2D-TM)={:.3e} d/ds(3D)={:.3e} (finite+nonzero) -> {}".format(
        gTM, g3D, "OK" if g_e else "FAIL"), flush=True)

    ok = g_a and g_b and g_c and g_d and g_e
    print("[t] *** JAX OBLIQUE FDTD (s/p 2D + 3D + grad): {} ***".format("PASS" if ok else "FAIL"), flush=True)
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
