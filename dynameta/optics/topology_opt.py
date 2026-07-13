"""Density-based TOPOLOGY OPTIMIZATION over the differentiable JAX-FDTD backend -- the design-tool leap:
go from forward characterization to AUTOMATED metasurface design. A continuous density field rho in
[0,1] over the patterned region is mapped to a manufacturable eps grid through the standard
inverse-design pipeline, all differentiable so jax.grad gives the adjoint sensitivity straight through
the FDTD time loop:

    rho --(spatial FILTER, min feature size)--> rho_f --(tanh PROJECTION, beta -> binary)--> rho_p
        --(linear eps interpolation)--> eps(x,z) --(JAX FDTD)--> figure of merit

topology_optimize() Adam-steps rho downhill while ANNEALING the projection sharpness beta (gray -> binary),
the established density-TO recipe (Sigmund/Wang filter+projection; Molesky et al. adjoint photonics).
Convention exp(-i omega t), SI. Pairs with optics.inverse_design.optimize_fdtd (the Adam driver).

The forward (rho_p -> scalar loss) is supplied by the caller -- it builds the FDTD eps from the projected
density (eps_from_density) into its grid, runs the JAX FDTD, and returns the objective.
"""
from __future__ import annotations

import numpy as np


def _conic_offsets(radius):
    """(dx, dz, weight) of a conic (linear-decay) filter kernel within `radius` cells."""
    r = int(np.ceil(radius))
    offs = []
    for dx in range(-r, r + 1):
        for dz in range(-r, r + 1):
            d = (dx * dx + dz * dz) ** 0.5
            if d <= radius:
                offs.append((dx, dz, 1.0 - d / radius))
    return offs


def _shift_clamp(arr, shift, axis):
    """Shift along `axis` by `shift` with EDGE-clamp padding (the filter does not wrap on this axis)."""
    if shift == 0:
        return arr
    import jax.numpy as jnp
    n = arr.shape[axis]
    lead = [slice(None)] * arr.ndim
    if shift > 0:
        lead[axis] = slice(0, 1)
        pad = jnp.repeat(arr[tuple(lead)], shift, axis=axis)
        keep = [slice(None)] * arr.ndim; keep[axis] = slice(0, n - shift)
        return jnp.concatenate([pad, arr[tuple(keep)]], axis=axis)
    k = -shift
    lead[axis] = slice(n - 1, n)
    pad = jnp.repeat(arr[tuple(lead)], k, axis=axis)
    keep = [slice(None)] * arr.ndim; keep[axis] = slice(k, n)
    return jnp.concatenate([arr[tuple(keep)], pad], axis=axis)


def density_filter(rho, radius, periodic_axes=(0,)):
    """Conic spatial filter on a 2D density region -- imposes a minimum feature size (kills pixel-scale /
    checkerboard designs). Axes in `periodic_axes` wrap (jnp.roll); the rest are EDGE-clamped. Defaults
    to (0,) = periodic axis-0 (x), clamped axis-1 (z) -- the 2D-FDTD pattern. For a 3D-FDTD LATERAL
    pattern (x AND y periodic) pass periodic_axes=(0, 1). Differentiable; radius <= 0 -> identity."""
    if radius is None or radius <= 0:
        return rho
    import jax.numpy as jnp
    acc = jnp.zeros_like(rho)
    wsum = 0.0
    for da, db, w in _conic_offsets(radius):
        s = jnp.roll(rho, da, axis=0) if 0 in periodic_axes else _shift_clamp(rho, da, 0)
        s = jnp.roll(s, db, axis=1) if 1 in periodic_axes else _shift_clamp(s, db, 1)
        acc = acc + w * s
        wsum += w
    return acc / wsum


def project(rho, beta, eta=0.5):
    """Smooth threshold projection toward BINARY (0/1) at threshold eta, sharpness beta (Wang 2011). As
    beta -> inf, rho_p -> a hard 0/1 design; beta is annealed UP across optimisation stages. Differentiable."""
    import jax.numpy as jnp
    num = jnp.tanh(beta * eta) + jnp.tanh(beta * (rho - eta))
    den = jnp.tanh(beta * eta) + jnp.tanh(beta * (1.0 - eta))
    return num / den


def eps_from_density(rho, eps_lo, eps_hi):
    """Linear material interpolation eps = eps_lo + rho*(eps_hi - eps_lo): rho=0 -> eps_lo (void/low index),
    rho=1 -> eps_hi (solid/high index). Differentiable in rho."""
    return eps_lo + rho * (eps_hi - eps_lo)


def binarization(rho_p, tol=0.05):
    """Fraction of the (projected) design that is within `tol` of 0 or 1 -- a manufacturability score
    (1.0 = fully binary). Plain numpy."""
    r = np.asarray(rho_p)
    return float(np.mean((r < tol) | (r > 1.0 - tol)))


def topology_optimize(forward_loss, rho0, *, filter_radius, periodic_axes=(0,),
                      betas=(1.0, 2.0, 4.0, 8.0, 16.0), steps_per_beta=20, lr=0.05, eta=0.5,
                      callback=None):
    """Density topology optimisation. `forward_loss(rho_projected) -> scalar` builds the FDTD eps from the
    PROJECTED density (eps_from_density into the caller's grid), runs the JAX FDTD, and returns the
    objective to MINIMISE. This driver applies the filter + tanh projection inside the loss and Adam-steps
    rho while annealing beta (gray -> binary) over the `betas` schedule. Returns (rho_opt, rho_projected,
    history). rho is clipped to [0,1]. Reuses optics.inverse_design.optimize_fdtd (Adam + jax.grad)."""
    import jax
    import jax.numpy as jnp
    from dynameta.optics.inverse_design import optimize_fdtd
    jax.config.update("jax_enable_x64", True)
    rho = jnp.asarray(np.asarray(rho0, dtype=float))
    history = []

    # ONE jitted value_and_grad with beta as a TRACED argument (audit 6.2 perf): baking each beta
    # into the loss as a python constant forced a fresh XLA compile per continuation stage (5
    # compiles for 5 betas); tracing beta lets a single compile serve the whole annealing schedule.
    # A traced float is results-identical to the same float constant.
    def _loss(r, beta):
        return forward_loss(project(density_filter(r, filter_radius, periodic_axes), beta, eta))
    vg = jax.jit(jax.value_and_grad(_loss))
    for beta in betas:
        b = jnp.asarray(float(beta))
        rho_np, h = optimize_fdtd(None, rho, n_steps=steps_per_beta, lr=lr, clip=(0.0, 1.0),
                                  value_and_grad=lambda p, _b=b: vg(p, _b))
        rho = jnp.asarray(rho_np)
        history.extend(h)
        if callback is not None:
            callback(float(beta), h)
    rho_p = np.asarray(project(density_filter(rho, filter_radius, periodic_axes), betas[-1], eta))
    return np.asarray(rho), rho_p, history
