"""Gradient-based INVERSE DESIGN over the differentiable JAX-FDTD backend -- the capability the
differentiable kernels (optics.fdtd_nd.run_2d_te_jax / run_3d_jax) were built for. The user writes a
JAX-differentiable loss(params) -> scalar that builds an eps grid from `params`, runs the JAX FDTD, and
returns a figure of merit (e.g. reflectance at a wavelength, or a spectral mismatch); `optimize_fdtd`
then Adam-steps the params downhill using jax.grad straight through the time loop -- no finite-difference
re-solves. Convention exp(-i omega t), SI.

Pattern:
    from dynameta.optics.fdtd_nd import run_2d_te_jax
    def loss(eps_slab):                       # eps_slab is a JAX scalar/array being optimised
        eps = eps_base.at[:, slab].set(eps_slab)
        eyL_t, *_ = run_2d_te_jax(eps, wp0, gam0, chi3_0, dx, dz, dt, nsteps, k_src, k_pL, k_pR, src, cpml)
        mRefl = jnp.fft.rfft((eyL_t - eyL_vac).mean(axis=1))   # eyL_vac precomputed (eps-independent)
        return jnp.abs(mRefl[ix] / mL_inc[ix]) ** 2           # reflectance at the target bin
    eps_opt, history = optimize_fdtd(loss, 2.5, n_steps=50, lr=0.1, clip=(1.0, 9.0))
"""
from __future__ import annotations

import numpy as np

from dynameta.constants import C_LIGHT, EPS0


def weighted_objective(terms):
    """Combine MULTIPLE objective terms into ONE differentiable scalar loss to MINIMISE (for
    topology_optimize / optimize_fdtd). Each `term` is a dict:
        {"value": fn(p) -> jax scalar, "weight": float (default 1.0), and ONE of:
            "sense": "max"  -> reward a large value (subtract weight*value),
            "sense": "min"  -> penalise a large value (add weight*value),
            "target": float -> drive value toward target (add weight*(value-target)^2)}
    So a wavelength-selective reflector is [{value: R_at_lambda1, sense:'max'},
    {value: R_at_lambda2, sense:'min'}]; a broadband flat target is many {value, target} terms. Returns
    loss(p). This is the multi-objective / multi-wavelength front-end -- the per-wavelength values come from
    a differentiable forward (Fdtd2dDesignProblem.spectrum)."""
    def loss(p):
        import jax.numpy as jnp
        total = jnp.asarray(0.0)
        for t in terms:
            if "target" in t and "sense" in t:           # the docstring says ONE of; enforce it
                raise ValueError("weighted_objective term has BOTH 'target' and 'sense'; supply exactly "
                                 "one ('sense' would be silently ignored when 'target' is present).")
            v = t["value"](p)
            w = float(t.get("weight", 1.0))
            if "target" in t:
                # squared-MAGNITUDE so the loss is real even if value() returns a complex amplitude
                # (e.g. a target on r/t); for a real R/T this is identical to (v - target)^2.
                total = total + w * jnp.abs(v - t["target"]) ** 2
            elif t.get("sense", "max") == "min":
                total = total + w * v
            else:
                total = total - w * v
        return total
    return loss


class Fdtd2dDesignProblem:
    """A DIFFERENTIABLE 2-D-TE FDTD forward over a designable density slab, for multi-wavelength /
    multi-objective inverse design. Builds the grid + a vacuum reference run ONCE; `spectrum(rho_p)` maps a
    projected density (nx, n_des) -> eps in the design slab -> a SINGLE JAX FDTD -> (R, T) at EACH target
    wavelength (the whole spectrum from one differentiable solve). Pair `.R`/`.T`/`.spectrum` with
    weighted_objective() + topology_optimize. Requires JAX (x64). The design slab sits between vacuum pads
    (the canonical free-standing patterned-slab inverse-design problem); eps_lo/eps_hi bound the material
    interpolation. The patterned region is x-periodic (a 1-D grating in the 2-D-TE plane)."""

    def __init__(self, *, period_x_m, lambdas_m, slab_thickness_m, n_des=4, eps_lo=1.0, eps_hi=12.0,
                 resolution=11, n_pad_wave=1.1, courant=0.5, settle=None, nx=None):
        import jax
        jax.config.update("jax_enable_x64", True)
        import jax.numpy as jnp
        from dynameta.optics.fdtd_nd import cpml_z, run_2d_te_jax
        self._run = run_2d_te_jax
        lams = np.asarray(lambdas_m, dtype=float).ravel()
        lo, hi = float(lams.min()) * 0.85, float(lams.max()) * 1.15
        n_max = np.sqrt(max(eps_hi, 1.0))
        dz = lo / (resolution * n_max); dx = dz
        dt = courant / (C_LIGHT * np.sqrt(1.0 / dx ** 2 + 1.0 / dz ** 2))
        if nx is None:
            nx = max(4, int(round(period_x_m / dz)))
        dx = period_x_m / nx
        dt = courant / (C_LIGHT * np.sqrt(1.0 / dx ** 2 + 1.0 / dz ** 2))
        pad = n_pad_wave * hi
        nz = int(round((2.0 * pad + slab_thickness_m) / dz)) + 1
        des_lo = int(round(pad / dz)); des_z = np.arange(des_lo, des_lo + n_des)
        k_src = max(2, int(round(0.4 * pad / dz)))
        k_pL = int(round(0.7 * pad / dz))
        k_pR = int(round((pad + n_des * dz + 0.3 * pad) / dz))
        f_c = C_LIGHT / (0.5 * (lo + hi))
        tau = 1.0 / (np.pi * (C_LIGHT / lo - C_LIGHT / hi))
        t0 = (settle if settle is not None else 12.0) * tau
        nsteps = int(round((2.0 * t0 + 4.0 * nz * dz / C_LIGHT + 200 * tau) / dt))
        tg = np.arange(nsteps) * dt
        src = np.exp(-((tg - t0) / tau) ** 2) * np.cos(2.0 * np.pi * f_c * (tg - t0))
        cpml = cpml_z(nz, dz, dt, 8)
        self.eps_lo, self.eps_hi, self.n_des = eps_lo, eps_hi, n_des
        self.des_z = jnp.asarray(des_z)
        self.eps_base = jnp.ones((nx, nz))
        self.wp0 = jnp.zeros((nx, nz)); self.gam0 = jnp.zeros((nx, nz)); self.chi3_0 = jnp.zeros((nx, nz))
        self.args = (dx, dz, dt, nsteps, k_src, k_pL, k_pR, jnp.asarray(src), cpml)
        # frequency bins for each target wavelength + the vacuum incident reference (eps-independent)
        f = np.fft.rfftfreq(nsteps, dt)
        self.bins = jnp.asarray([int(np.argmin(np.abs(f - C_LIGHT / L))) for L in lams])
        eyL_v, _, eyR_v, _ = self._run(self.eps_base, self.wp0, self.gam0, self.chi3_0, *self.args)
        self.eyL_vac = eyL_v
        self.mL_inc = jnp.fft.rfft(eyL_v.mean(axis=1))[self.bins]
        self.mR_inc = jnp.fft.rfft(eyR_v.mean(axis=1))[self.bins]
        # the vacuum reference amplitudes normalize R/T (spectrum: R=|mRefl/mL_inc|^2); a ~0 reference at
        # a target bin would make R/T NaN and silently poison the optimizer -- fail loudly here instead.
        if float(jnp.min(jnp.abs(self.mL_inc))) <= 0.0 or float(jnp.min(jnp.abs(self.mR_inc))) <= 0.0:
            raise ValueError("Fdtd2dDesignProblem: the vacuum reference field is ~0 at a target "
                             "wavelength bin (R/T would be NaN); check lambdas_m vs the FDTD frequency "
                             "resolution / settle time.")
        self.lambdas_m = lams

    def spectrum(self, rho_p):
        """(R, T) at the target wavelengths from ONE differentiable FDTD solve. rho_p: (nx, n_des) in [0,1].
        R/T are JAX arrays of length len(lambdas_m)."""
        import jax.numpy as jnp
        from dynameta.optics.topology_opt import eps_from_density
        eps_d = eps_from_density(rho_p, self.eps_lo, self.eps_hi)
        eps_inf = self.eps_base.at[:, self.des_z].set(eps_d)
        eyL_t, _hxL, eyR_t, _hxR = self._run(eps_inf, self.wp0, self.gam0, self.chi3_0, *self.args)
        mRefl = jnp.fft.rfft((eyL_t - self.eyL_vac).mean(axis=1))[self.bins]
        mTrans = jnp.fft.rfft(eyR_t.mean(axis=1))[self.bins]
        R = jnp.abs(mRefl / self.mL_inc) ** 2
        T = jnp.abs(mTrans / self.mR_inc) ** 2
        return R, T

    def R(self, rho_p):
        return self.spectrum(rho_p)[0]

    def T(self, rho_p):
        return self.spectrum(rho_p)[1]


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
        # NON-FINITE guard: if the loss/grad blew up (exploded eps, bad lr), stop and return the last
        # finite params (p is still the previous iterate here) + record nan -- do NOT silently keep
        # stepping with NaN params (audit gap).
        if not (bool(jnp.isfinite(loss_val)) and bool(jnp.all(jnp.isfinite(grad)))):
            history.append(float("nan"))
            if callback is not None:
                callback(t, float("nan"), np.asarray(p))
            break
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
