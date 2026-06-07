"""Gradient-based INVERSE DESIGN over the differentiable JAX-FDTD backend -- the capability the
differentiable kernels (optics.fdtd_nd._run_2d_te_jax / _run_3d_jax) were built for. The user writes a
JAX-differentiable loss(params) -> scalar that builds an eps grid from `params`, runs the JAX FDTD, and
returns a figure of merit (e.g. reflectance at a wavelength, or a spectral mismatch); `optimize_fdtd`
then Adam-steps the params downhill using jax.grad straight through the time loop -- no finite-difference
re-solves. Convention exp(-i omega t), SI.

Pattern:
    from dynameta.optics.fdtd_nd import _run_2d_te_jax
    def loss(eps_slab):                       # eps_slab is a JAX scalar/array being optimised
        eps = eps_base.at[:, slab].set(eps_slab)
        eyL_t, *_ = _run_2d_te_jax(eps, wp0, gam0, chi3_0, dx, dz, dt, nsteps, k_src, k_pL, k_pR, src, cpml)
        mRefl = jnp.fft.rfft((eyL_t - eyL_vac).mean(axis=1))   # eyL_vac precomputed (eps-independent)
        return jnp.abs(mRefl[ix] / mL_inc[ix]) ** 2           # reflectance at the target bin
    eps_opt, history = optimize_fdtd(loss, 2.5, n_steps=50, lr=0.1, clip=(1.0, 9.0))
"""
from __future__ import annotations

import numpy as np


def optimize_fdtd(loss_fn, params0, *, n_steps: int = 60, lr: float = 0.05, b1: float = 0.9,
                  b2: float = 0.999, eps: float = 1e-8, clip=None, callback=None):
    """Adam optimiser over a DIFFERENTIABLE JAX-FDTD loss. `loss_fn(params) -> scalar` must be built from
    the JAX FDTD backend so jax.grad flows through the time loop. `params0` is a scalar or array. `clip`
    = (lo, hi) bounds applied after each step (e.g. keep eps physical). `callback(step, loss, params)`
    runs each step. Returns (params_opt, history) with history = loss per step. value_and_grad is JIT-
    compiled once, so steps after the first are fast. Requires JAX (x64 enabled here for FDTD accuracy)."""
    import jax
    import jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)
    vg = jax.jit(jax.value_and_grad(loss_fn))
    p = jnp.asarray(params0, dtype=jnp.float64)
    m = jnp.zeros_like(p)
    v = jnp.zeros_like(p)
    history = []
    for t in range(1, int(n_steps) + 1):
        loss_val, grad = vg(p)
        m = b1 * m + (1.0 - b1) * grad
        v = b2 * v + (1.0 - b2) * grad ** 2
        m_hat = m / (1.0 - b1 ** t)
        v_hat = v / (1.0 - b2 ** t)
        p = p - lr * m_hat / (jnp.sqrt(v_hat) + eps)
        if clip is not None:
            p = jnp.clip(p, clip[0], clip[1])
        history.append(float(loss_val))
        if callback is not None:
            callback(t, float(loss_val), np.asarray(p))
    return np.asarray(p), history
