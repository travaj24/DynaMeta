"""2D FDTD autodiff oracle: the 'jax' backend is DIFFERENTIABLE end-to-end. optics.fdtd_nd._run_2d_te_jax
runs the same 2D-TE physics as the NumPy reference inside a compiled XLA lax.scan, so a scalar objective
built from its output fields can be differentiated with jax.grad straight THROUGH the FDTD time loop --
the gradient an inverse-design / topology-optimization outer loop needs (d(figure of merit)/d(material
or geometry), with no finite-difference re-solves).

GATE (gradient correctness): a transmitted-energy objective J(s) = sum |E_y(right probe)|^2 for a slab
whose permittivity is scaled by s. The reverse-mode jax.grad dJ/ds must match a central finite-difference
(two extra forward solves) to ~1e-4 relative, AND be non-trivially nonzero (a zero gradient would pass a
loose FD check vacuously). Establishes the backend is genuinely usable for gradient-based design, not just
a faster forward solver. Skipped (exit 42 = the run_all SKIP category, audit C6-6) if JAX is not installed.

Run: python -m validation.fdtd_2d_autodiff
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dynameta.optics.fdtd_nd import _cpml_z, _have_jax

C = 299792458.0
EPS0 = 8.8541878128e-12
MU0 = 1.0 / (EPS0 * C ** 2)


def _build_small_slab():
    """A deliberately SMALL uniform-slab 2D-TE setup (so reverse-mode through the scan is cheap): returns
    the grid arrays + a boolean slab mask whose eps the objective scales. Physics need only be smooth in
    the scale parameter, not fully converged -- this is a gradient-correctness test, not an accuracy one."""
    lam, n, d, res = 1500e-9, 2.0, 300e-9, 20
    dz = lam / (res * n)
    nx = 4
    dx = dz
    dt = 0.5 / (C * np.sqrt(1.0 / dx ** 2 + 1.0 / dz ** 2))
    pad = 2.5 * lam
    Lz = 2.0 * pad + d
    nz = int(round(Lz / dz)) + 1
    zc = (np.arange(nz) + 0.5) * dz
    slab = (zc >= pad) & (zc < pad + d)
    eps_base = np.ones((nx, nz))
    eps_base[:, slab] = n ** 2
    mask = np.zeros((nx, nz)); mask[:, slab] = 1.0
    k_src = max(2, int(round((0.4 * pad) / dz)))
    k_pL = int(round((0.7 * pad) / dz))
    k_pR = int(round((pad + d + 0.3 * pad) / dz))
    fc = C / lam
    tau = 6.0 / fc
    t0 = 4.0 * tau
    nsteps = int(round((2.0 * t0 + 3.0 * (Lz / C) + 40.0 * tau) / dt))
    tg = np.arange(nsteps) * dt
    src = np.exp(-((tg - t0) / tau) ** 2) * np.cos(2.0 * np.pi * fc * (tg - t0))
    cpml = _cpml_z(nz, dz, dt, npml=10)
    return dict(eps_base=eps_base, mask=mask, nx=nx, nz=nz, dx=dx, dz=dz, dt=dt, nsteps=nsteps,
                k_src=k_src, k_pL=k_pL, k_pR=k_pR, src=src, cpml=cpml)


def main():
    print("[fad] === 2D FDTD autodiff: jax.grad straight through the FDTD time loop ===", flush=True)
    if not _have_jax():
        print("[fad] JAX not installed -> SKIP (exit 42; run_all counts it separately, audit C6-6)", flush=True)
        raise SystemExit(42)

    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from dynameta.optics.fdtd_nd import _run_2d_te_jax

    g = _build_small_slab()
    eps_base = jnp.asarray(g["eps_base"]); mask = jnp.asarray(g["mask"])
    wp = jnp.zeros((g["nx"], g["nz"])); gam = jnp.zeros((g["nx"], g["nz"])); chi3 = jnp.zeros((g["nx"], g["nz"]))

    def objective(s):
        """Transmitted-energy figure of merit for a slab whose eps is scaled by s (s=1 -> base)."""
        eps = eps_base * (1.0 + mask * (s - 1.0))           # scale ONLY the slab cells, smoothly in s
        _, _, eyR, _ = _run_2d_te_jax(eps, wp, gam, chi3, g["dx"], g["dz"], g["dt"], g["nsteps"],
                                      g["k_src"], g["k_pL"], g["k_pR"], g["src"], g["cpml"])
        return jnp.sum(eyR ** 2)

    s0 = 1.0
    J0 = float(objective(s0))
    grad_ad = float(jax.grad(objective)(s0))                # reverse-mode AD through the scan
    h = 1.0e-3
    grad_fd = (float(objective(s0 + h)) - float(objective(s0 - h))) / (2.0 * h)  # central difference

    rel = abs(grad_ad - grad_fd) / max(abs(grad_fd), 1e-30)
    nonzero = abs(grad_ad) > 1e-6 * max(abs(J0), 1e-30)
    gate = bool(rel < 1e-3 and nonzero)
    print("[fad] J(1)={:.6e}  dJ/ds: autodiff={:.6e}  finite-diff={:.6e}".format(J0, grad_ad, grad_fd),
          flush=True)
    print("[fad] grad rel-err(AD vs FD)={:.2e} (<1e-3) ; nonzero={} -> {}".format(
        rel, nonzero, "PASS" if gate else "FAIL"), flush=True)
    print("[fad] *** FDTD jax.grad is correct through the time loop (inverse-design ready): {} ***".format(
        "PASS" if gate else "FAIL"), flush=True)
    return gate


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
