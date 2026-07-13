"""FDTD TOPOLOGY-OPTIMIZATION oracle: optics.topology_opt does density-based inverse design over the
differentiable JAX FDTD (filter -> tanh projection -> eps interpolation -> FDTD -> figure of merit), so a
metasurface's GEOMETRY can be optimized by jax.grad straight through the time loop. Two checks:

GATE 1 (adjoint gradient correctness): jax.grad of the FULL pipeline (density_filter -> project ->
        eps_from_density -> JAX FDTD -> reflectance) w.r.t. the density rho matches a central
        finite-difference on sampled design cells -- the adjoint sensitivity is correct end to end.
GATE 2 (optimization works): a topology-optimization run (design a patterned layer to MAXIMISE
        reflectance at lambda) starting from a gray field strongly INCREASES R and BINARISES the design
        (a manufacturable 0/1 pattern), via the filter+projection+beta-annealing recipe.

Deliberately small grid (reverse-mode stores the FDTD carry per step). Skipped (exit 42 = the run_all SKIP category, audit C6-6) if JAX absent.

Run: python -m validation.fdtd_topology_opt
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dynameta.optics.fdtd_nd import _cpml_z, _have_jax

C = 299792458.0
LAM = 1500e-9
EPS_HI = 12.0          # high-index solid (a-Si-like); eps_lo = 1 (void)
RAD = 2.0              # filter radius (min feature size, cells)
NX, NLAYER = 20, 6     # design region: 20 (periodic x) x 6 (z) cells


def _grid():
    res, n_max = 12, np.sqrt(EPS_HI)
    dz = LAM / (res * n_max)
    dx = dz
    dt = 0.5 / (C * np.sqrt(1.0 / dx ** 2 + 1.0 / dz ** 2))
    pad = 1.2 * LAM
    d_layer = NLAYER * dz
    Lz = 2.0 * pad + d_layer
    nz = int(round(Lz / dz)) + 1
    zc = (np.arange(nz) + 0.5) * dz
    k0 = int(round(pad / dz))
    layer = np.zeros(nz, dtype=bool); layer[k0:k0 + NLAYER] = True
    k_src = max(2, int(round(0.4 * pad / dz)))
    k_pL = int(round(0.7 * pad / dz))
    k_pR = int(round((pad + d_layer + 0.3 * pad) / dz))
    fc = C / LAM
    df = C / (LAM * 0.87) - C / (LAM * 1.13)
    tau = 1.0 / (np.pi * df)
    t0 = 3.0 * tau
    nsteps = int(round((2.0 * t0 + 2.0 * (Lz / C) + 12.0 * tau) / dt))
    tg = np.arange(nsteps) * dt
    src = np.exp(-((tg - t0) / tau) ** 2) * np.cos(2.0 * np.pi * fc * (tg - t0))
    cpml = _cpml_z(nz, dz, dt, npml=8)
    ix = int(np.argmin(np.abs(np.fft.rfftfreq(nsteps, dt) - fc)))
    return dict(nx=NX, nz=nz, dx=dx, dz=dz, dt=dt, nsteps=nsteps, layer=layer, k_src=k_src,
                k_pL=k_pL, k_pR=k_pR, src=src, cpml=cpml, ix=ix)


def main():
    print("[fto] === FDTD topology optimization: adjoint gradient + optimize a patterned layer ===", flush=True)
    if not _have_jax():
        print("[fto] JAX not installed -> SKIP (exit 42; run_all counts it separately, audit C6-6)", flush=True)
        raise SystemExit(42)

    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from dynameta.optics.fdtd_nd import _run_2d_te_jax
    from dynameta.optics.topology_opt import (binarization, density_filter, eps_from_density,
                                              project, topology_optimize)

    g = _grid()
    nx, nz = g["nx"], g["nz"]
    layer = jnp.asarray(g["layer"])
    z2 = jnp.zeros((nx, nz))
    args = (g["dx"], g["dz"], g["dt"], g["nsteps"], g["k_src"], g["k_pL"], g["k_pR"],
            jnp.asarray(g["src"]), g["cpml"])
    eps_base = jnp.ones((nx, nz))
    eyL_vac = _run_2d_te_jax(eps_base, z2, z2, z2, *args)[0]
    mL_inc = jnp.fft.rfft(eyL_vac.mean(axis=1))[g["ix"]]

    def forward_loss(rho_p):                                 # rho_p: (nx, NLAYER) projected density
        eps_layer = eps_from_density(rho_p, 1.0, EPS_HI)
        eps = eps_base.at[:, layer].set(eps_layer)
        eyL_t = _run_2d_te_jax(eps, z2, z2, z2, *args)[0]
        mRefl = jnp.fft.rfft((eyL_t - eyL_vac).mean(axis=1))[g["ix"]]
        return -jnp.abs(mRefl / mL_inc) ** 2                # maximise R == minimise -R

    # GATE 1: adjoint gradient vs finite-difference (full filter->project->eps->FDTD pipeline)
    rng = np.random.RandomState(0)
    rho0 = jnp.asarray(0.4 + 0.2 * rng.rand(nx, NLAYER))    # a non-degenerate gray start
    full = lambda r: forward_loss(project(density_filter(r, RAD), 4.0))
    grad = np.asarray(jax.grad(full)(rho0))
    h = 1e-3
    cells = [(3, 2), (10, 4), (17, 1)]
    rel = 0.0
    for (i, j) in cells:
        rp = rho0.at[i, j].add(h); rm = rho0.at[i, j].add(-h)
        gfd = (float(full(rp)) - float(full(rm))) / (2.0 * h)
        rel = max(rel, abs(grad[i, j] - gfd) / max(abs(gfd), 1e-9))
    gate1 = bool(rel < 5e-3)
    print("[fto] 1 adjoint grad vs finite-diff ({} cells): max rel-err={:.2e} (<5e-3) -> {}".format(
        len(cells), rel, "PASS" if gate1 else "FAIL"), flush=True)

    # GATE 2: optimise a patterned layer to maximise R; expect R up + a binarised design
    rho_opt, rho_p, hist = topology_optimize(forward_loss, 0.5 * np.ones((nx, NLAYER)),
                                             filter_radius=RAD, betas=(2.0, 8.0, 16.0),
                                             steps_per_beta=12, lr=0.12)
    R_init, R_final = -hist[0], -float(min(hist))
    binar = binarization(rho_p)
    gate2 = bool(R_final > R_init + 0.1 and R_final > 0.5 and binar > 0.7)
    print("[fto] 2 optimize R: R {:.3f} -> {:.3f} (+{:.3f}) ; binarization={:.2f} (manufacturable) -> "
          "{}".format(R_init, R_final, R_final - R_init, binar, "PASS" if gate2 else "FAIL"), flush=True)

    overall = gate1 and gate2
    print("[fto] *** FDTD TOPOLOGY OPTIMIZATION (adjoint grad correct; design optimised + binarised): "
          "{} ***".format("PASS" if overall else "FAIL"), flush=True)
    return overall


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
