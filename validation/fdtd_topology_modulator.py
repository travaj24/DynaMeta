"""DEVICE-FOM topology optimization: design the metasurface GEOMETRY to maximise the MODULATION CONTRAST
(the device figure of merit), not just a single-state reflectance. A patterned dielectric layer (the
design variable rho) sits with a tunable layer whose eps SWITCHES between two bias states; the optimiser
shapes rho so the bias switch maximally changes the reflection -- automated modulator design over the
differentiable FDTD. Plus a 3D check, since real metasurface cells are 2D-periodic.

GATE 1 (2D device-FOM optimization): topology-optimize a patterned layer to maximise |R_on - R_off| at
        lambda; the contrast strongly INCREASES from a gray start and the design BINARISES (manufacturable).
GATE 2 (3D adjoint gradient): jax.grad of the full lateral-pattern pipeline (density_filter[x,y periodic]
        -> project -> eps -> 3D JAX FDTD -> R) matches a central finite-difference -- 3D topology
        optimization is differentiable end to end (the design-tool works on the real 2D-periodic cell).

Small grids (reverse-mode stores the FDTD carry per step; the 2D run does TWO FDTDs/eval). Skip if no JAX.

Run: python -m validation.fdtd_topology_modulator
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dynameta.optics.fdtd_nd import _cpml_z, _have_jax

C = 299792458.0
LAM = 1500e-9
EPS_HI = 12.0
ITO_OFF, ITO_ON = 3.5, 1.5      # tunable-layer eps in the two bias states (real eps switch, FP-clean)


def _grid_2d():
    res, n_max = 12, np.sqrt(EPS_HI)
    dz = LAM / (res * n_max); dx = dz
    dt = 0.5 / (C * np.sqrt(1.0 / dx ** 2 + 1.0 / dz ** 2))
    pad = 1.2 * LAM
    n_des, n_ito = 4, 3
    Lz = 2.0 * pad + (n_des + n_ito) * dz
    nz = int(round(Lz / dz)) + 1
    k0 = int(round(pad / dz))
    des = np.zeros(nz, bool); des[k0:k0 + n_des] = True
    ito = np.zeros(nz, bool); ito[k0 + n_des:k0 + n_des + n_ito] = True
    k_src = max(2, int(round(0.4 * pad / dz))); k_pL = int(round(0.7 * pad / dz))
    k_pR = int(round((pad + (n_des + n_ito) * dz + 0.3 * pad) / dz))
    fc = C / LAM; df = C / (LAM * 0.87) - C / (LAM * 1.13); tau = 1.0 / (np.pi * df); t0 = 3.0 * tau
    nsteps = int(round((2.0 * t0 + 2.0 * (Lz / C) + 12.0 * tau) / dt))
    tg = np.arange(nsteps) * dt
    src = np.exp(-((tg - t0) / tau) ** 2) * np.cos(2.0 * np.pi * fc * (tg - t0))
    ix = int(np.argmin(np.abs(np.fft.rfftfreq(nsteps, dt) - fc)))
    return dict(nx=16, nz=nz, n_des=n_des, dx=dx, dz=dz, dt=dt, nsteps=nsteps, des=des, ito=ito,
                k_src=k_src, k_pL=k_pL, k_pR=k_pR, src=src, cpml=_cpml_z(nz, dz, dt, 8), ix=ix)


def _grid_3d():
    res, n_max = 8, np.sqrt(EPS_HI)
    dz = LAM / (res * n_max); dx = dy = dz
    dt = 0.5 / (C * np.sqrt(1.0 / dx ** 2 + 1.0 / dy ** 2 + 1.0 / dz ** 2))
    pad = 0.9 * LAM
    n_des = 3
    Lz = 2.0 * pad + n_des * dz
    nz = int(round(Lz / dz)) + 1
    k0 = int(round(pad / dz))
    des = np.zeros(nz, bool); des[k0:k0 + n_des] = True
    k_src = max(2, int(round(0.4 * pad / dz))); k_pL = int(round(0.7 * pad / dz))
    k_pR = int(round((pad + n_des * dz + 0.3 * pad) / dz))
    fc = C / LAM; tau = 4.0 / fc; t0 = 3.0 * tau
    nsteps = int(round((2.0 * t0 + 2.0 * (Lz / C) + 10.0 * tau) / dt))
    tg = np.arange(nsteps) * dt
    src = np.exp(-((tg - t0) / tau) ** 2) * np.cos(2.0 * np.pi * fc * (tg - t0))
    ix = int(np.argmin(np.abs(np.fft.rfftfreq(nsteps, dt) - fc)))
    return dict(nx=4, ny=4, nz=nz, dx=dx, dy=dy, dz=dz, dt=dt, nsteps=nsteps, des=des,
                k_src=k_src, k_pL=k_pL, k_pR=k_pR, src=src, cpml=_cpml_z(nz, dz, dt, 6), ix=ix)


def main():
    print("[ftm] === Device-FOM topology optimization: maximise modulation contrast + 3D grad ===", flush=True)
    if not _have_jax():
        print("[ftm] JAX not installed -> SKIP (exit 0)", flush=True)
        return True
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from dynameta.optics.fdtd_nd import _run_2d_te_jax, _run_3d_jax
    from dynameta.optics.topology_opt import (binarization, density_filter, eps_from_density,
                                              project, topology_optimize)

    # ---- GATE 1: 2D, maximise |R_on - R_off| ----
    g = _grid_2d(); nx, nz = g["nx"], g["nz"]
    des, ito = jnp.asarray(g["des"]), jnp.asarray(g["ito"])
    z2 = jnp.zeros((nx, nz))
    a2 = (g["dx"], g["dz"], g["dt"], g["nsteps"], g["k_src"], g["k_pL"], g["k_pR"], jnp.asarray(g["src"]), g["cpml"])
    base2 = jnp.ones((nx, nz))
    vac2 = _run_2d_te_jax(base2, z2, z2, z2, *a2)[0]
    mL2 = jnp.fft.rfft(vac2.mean(axis=1))[g["ix"]]

    def _R2(eps):
        eyL = _run_2d_te_jax(eps, z2, z2, z2, *a2)[0]
        return jnp.abs(jnp.fft.rfft((eyL - vac2).mean(axis=1))[g["ix"]] / mL2) ** 2

    def contrast_loss(rho_p):                               # rho_p: (nx, n_des)
        eps_d = eps_from_density(rho_p, 1.0, EPS_HI)
        e = base2.at[:, des].set(eps_d)
        R_off = _R2(e.at[:, ito].set(ITO_OFF))
        R_on = _R2(e.at[:, ito].set(ITO_ON))
        return -(R_on - R_off) ** 2                         # maximise the SQUARED contrast (sign-agnostic)

    rho0 = 0.5 * np.ones((nx, g["n_des"]))
    dR_init = float(np.sqrt(-contrast_loss(jnp.asarray(rho0))))
    rho, rho_p, hist = topology_optimize(contrast_loss, rho0, filter_radius=2.0, periodic_axes=(0,),
                                         betas=(2.0, 8.0, 16.0), steps_per_beta=10, lr=0.12)
    dR_final = float(np.sqrt(-min(hist)))
    binar = binarization(rho_p)
    gate1 = bool(dR_final > 1.3 * dR_init and binar > 0.7)   # the optimiser substantially grows the contrast
    print("[ftm] 1 maximise contrast: |dR| {:.3f} -> {:.3f} ({:.2f}x, +{:.0f}%) ; binarization={:.2f} -> "
          "{}".format(dR_init, dR_final, dR_final / max(dR_init, 1e-9),
                      100.0 * (dR_final / max(dR_init, 1e-9) - 1.0), binar, "PASS" if gate1 else "FAIL"),
          flush=True)

    # ---- GATE 2: 3D adjoint gradient of the lateral-pattern pipeline ----
    h = _grid_3d(); hx, hy, hz = h["nx"], h["ny"], h["nz"]
    des3 = jnp.asarray(h["des"]); z3 = jnp.zeros((hx, hy, hz))
    a3 = (h["dx"], h["dy"], h["dz"], h["dt"], h["nsteps"], h["k_src"], h["k_pL"], h["k_pR"], jnp.asarray(h["src"]), h["cpml"])
    base3 = jnp.ones((hx, hy, hz))
    vac3 = _run_3d_jax(base3, z3, z3, z3, *a3)[1]           # eyL plane (out idx 1)
    mL3 = jnp.fft.rfft(vac3.mean(axis=(1, 2)))[h["ix"]]

    def loss3(rho):                                         # rho: (hx, hy) lateral pattern, x&y periodic
        rp = project(density_filter(rho, 1.5, periodic_axes=(0, 1)), 6.0)
        eps_lat = eps_from_density(rp, 1.0, EPS_HI)         # (hx, hy)
        eps = base3.at[:, :, des3].set(eps_lat[:, :, None])
        eyL = _run_3d_jax(eps, z3, z3, z3, *a3)[1]
        return jnp.abs(jnp.fft.rfft((eyL - vac3).mean(axis=(1, 2)))[h["ix"]] / mL3) ** 2

    r0 = jnp.asarray(0.4 + 0.2 * np.random.RandomState(0).rand(hx, hy))
    grad3 = np.asarray(jax.grad(loss3)(r0))
    hh = 1e-3; rel3 = 0.0
    for (i, j) in [(1, 2), (3, 0)]:
        gp = (float(loss3(r0.at[i, j].add(hh))) - float(loss3(r0.at[i, j].add(-hh)))) / (2.0 * hh)
        rel3 = max(rel3, abs(grad3[i, j] - gp) / max(abs(gp), 1e-9))
    gate2 = bool(rel3 < 5e-3)
    print("[ftm] 2 3D adjoint grad vs finite-diff (lateral x,y-periodic): max rel-err={:.2e} -> {}".format(
        rel3, "PASS" if gate2 else "FAIL"), flush=True)

    overall = gate1 and gate2
    print("[ftm] *** DEVICE-FOM TOPOLOGY OPTIMIZATION (2D contrast maximised; 3D grad correct): {} ***".format(
        "PASS" if overall else "FAIL"), flush=True)
    return overall


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
