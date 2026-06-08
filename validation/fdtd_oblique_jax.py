"""Validate the JAX backend for the OBLIQUE 2D FDTD (complex-envelope Bloch, s-pol): a single traced
lax.scan time loop that is (1) byte-for-byte equal to the NumPy oblique kernel on R/T and (2)
DIFFERENTIABLE end-to-end, so jax.grad gives d(objective)/d(geometry) straight through the oblique scan
-- gradient-based inverse design AT AN ANGLE (the normal-incidence jax backend already powered the
multi-objective designer; this extends it to oblique).

GATE A: backend='jax' reproduces backend='numpy' R0/T0 to < 1e-10 across the band.
GATE B: jax.grad of a transmission-energy objective wrt the slab eps, through _run_2d_te_oblique_jax,
        is finite and non-zero (the adjoint flows through the complex-envelope time loop).

Skipped (exit 0) if JAX is not installed. Run: python -m validation.fdtd_oblique_jax
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dynameta.optics.fdtd_nd import FDTDLayer, solve_fdtd_2d_oblique, _have_jax, _cpml_z
from dynameta.constants import C_LIGHT


def main():
    print("[t] === JAX oblique 2D FDTD: jax==numpy + differentiable ===", flush=True)
    if not _have_jax():
        print("[t] JAX not installed -> SKIP (exit 0)", flush=True)
        return True
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
    from dynameta.optics.fdtd_nd import _run_2d_te_oblique_jax
    nx, nz = 6, 400
    dz = 1.2e-6 / (22 * 2.0); dx = 320e-9 / nx
    dt = 0.5 / (C_LIGHT * np.sqrt(1.0 / dx ** 2 + 1.0 / dz ** 2))
    kx = (2.0 * np.pi * 2e14 / C_LIGHT) * np.sin(np.radians(25.0))
    nsteps = 2500
    tg = np.arange(nsteps) * dt
    src = np.exp(-((tg - 300 * dt) / (120 * dt)) ** 2) * np.cos(0.2 * np.arange(nsteps))
    cpml = _cpml_z(nz, dz, dt, 12)
    wp = jnp.zeros((nx, nz)); gam = jnp.zeros((nx, nz))

    def loss(eps_slab):
        eps = jnp.ones((nx, nz)).at[:, 150:250].set(eps_slab)
        _eyL, _hxL, eyR, _hxR = _run_2d_te_oblique_jax(eps, wp, gam, dx, dz, dt, nsteps, 100, 200, 350,
                                                       jnp.asarray(src), cpml, kx)
        return jnp.sum(jnp.abs(jnp.fft.fft(eyR.mean(axis=1))) ** 2)

    grad = float(jax.grad(loss)(4.0))
    g_b = bool(np.isfinite(grad) and abs(grad) > 0.0)
    print("[t] B d(loss)/d(eps_slab) through the oblique scan = {:.3e} (finite + nonzero) -> {}".format(
        grad, "OK" if g_b else "FAIL"), flush=True)
    ok = g_a and g_b
    print("[t] *** JAX OBLIQUE FDTD: {} ***".format("PASS" if ok else "FAIL"), flush=True)
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
