"""3D FDTD autodiff oracle: the 'jax' 3D backend (optics.fdtd_nd.run_3d_jax) is DIFFERENTIABLE end to
end -- the full six-component vector FDTD runs inside a compiled XLA lax.scan, so a scalar objective built
from its output fields can be differentiated with jax.grad straight THROUGH the 3D time loop. That is the
gradient a 3D inverse-design / topology-optimization outer loop needs (d(figure of merit)/d(geometry or
material), no finite-difference re-solves).

GATE (gradient correctness): a transmitted-energy objective J(s) = sum |E_y(right probe)|^2 for a slab
whose permittivity is scaled by s. The reverse-mode jax.grad dJ/ds must match a central finite-difference
to ~1e-3 relative AND be non-trivially nonzero. Grid kept deliberately TINY (reverse-mode stores all 13
carried fields per step). Skipped (exit 42 = the run_all SKIP category, audit C6-6) if JAX is not installed.

Run: python -m validation.fdtd_3d_autodiff
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dynameta.optics.fdtd_nd import cpml_z, have_jax

C = 299792458.0


def _build_small_slab():
    """A deliberately TINY uniform-slab 3D-FDTD setup so reverse-mode through the 13-field scan is cheap."""
    lam, n, d, res = 1500e-9, 2.0, 300e-9, 10
    dz = lam / (res * n)
    nx = ny = 4
    dx = dy = dz
    dt = 0.5 / (C * np.sqrt(1.0 / dx ** 2 + 1.0 / dy ** 2 + 1.0 / dz ** 2))
    pad = 1.0 * lam
    Lz = 2.0 * pad + d
    nz = int(round(Lz / dz)) + 1
    zc = (np.arange(nz) + 0.5) * dz
    slab = (zc >= pad) & (zc < pad + d)
    eps_base = np.ones((nx, ny, nz)); eps_base[:, :, slab] = n ** 2
    mask = np.zeros((nx, ny, nz)); mask[:, :, slab] = 1.0
    k_src = max(2, int(round(0.4 * pad / dz)))
    k_pL = int(round(0.7 * pad / dz))
    k_pR = int(round((pad + d + 0.3 * pad) / dz))
    fc = C / lam
    tau = 4.0 / fc
    t0 = 3.0 * tau
    nsteps = int(round((2.0 * t0 + 2.0 * (Lz / C) + 10.0 * tau) / dt))
    tg = np.arange(nsteps) * dt
    src = np.exp(-((tg - t0) / tau) ** 2) * np.cos(2.0 * np.pi * fc * (tg - t0))
    cpml = cpml_z(nz, dz, dt, npml=8)
    return dict(eps_base=eps_base, mask=mask, nx=nx, ny=ny, nz=nz, dx=dx, dy=dy, dz=dz, dt=dt,
                nsteps=nsteps, k_src=k_src, k_pL=k_pL, k_pR=k_pR, src=src, cpml=cpml)


def main():
    print("[f3d] === 3D FDTD autodiff: jax.grad straight through the 3D FDTD time loop ===", flush=True)
    if not have_jax():
        print("[f3d] JAX not installed -> SKIP (exit 42; run_all counts it separately, audit C6-6)", flush=True)
        raise SystemExit(42)

    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from dynameta.optics.fdtd_nd import run_3d_jax

    g = _build_small_slab()
    eps_base = jnp.asarray(g["eps_base"]); mask = jnp.asarray(g["mask"])
    z = jnp.zeros((g["nx"], g["ny"], g["nz"]))              # zero Drude/Kerr fields

    def objective(s):
        """Transmitted-energy figure of merit for a slab whose eps is scaled by s (s=1 -> base)."""
        eps = eps_base * (1.0 + mask * (s - 1.0))
        out = run_3d_jax(eps, z, z, z, g["dx"], g["dy"], g["dz"], g["dt"], g["nsteps"],
                          g["k_src"], g["k_pL"], g["k_pR"], g["src"], g["cpml"])
        eyR = out[5]                                        # (exL,eyL,hxL,hyL, exR,eyR,hxR,hyR)
        return jnp.sum(eyR ** 2)

    s0 = 1.0
    J0 = float(objective(s0))
    grad_ad = float(jax.grad(objective)(s0))               # reverse-mode AD through the 3D scan
    h = 1.0e-3
    grad_fd = (float(objective(s0 + h)) - float(objective(s0 - h))) / (2.0 * h)

    rel = abs(grad_ad - grad_fd) / max(abs(grad_fd), 1e-30)
    nonzero = abs(grad_ad) > 1e-6 * max(abs(J0), 1e-30)
    gate = bool(rel < 1e-3 and nonzero)
    print("[f3d] J(1)={:.6e}  dJ/ds: autodiff={:.6e}  finite-diff={:.6e}".format(J0, grad_ad, grad_fd),
          flush=True)
    print("[f3d] grad rel-err(AD vs FD)={:.2e} (<1e-3) ; nonzero={} -> {}".format(
        rel, nonzero, "PASS" if gate else "FAIL"), flush=True)
    print("[f3d] *** 3D FDTD jax.grad correct through the time loop (3D inverse-design ready): {} ***".format(
        "PASS" if gate else "FAIL"), flush=True)
    return gate


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
