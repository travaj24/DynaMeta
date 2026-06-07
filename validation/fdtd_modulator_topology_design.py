"""CONVERGENCE: design a REALISTIC ENZ modulator's geometry by topology optimization -- the design-tool
(differentiable FDTD topology opt) applied to the ENZ-graded carrier->optics coupling. A graded ITO layer
(a depth-resolved free-carrier Drude profile that SWITCHES between two gate-bias states -- background vs
accumulation, the latter crossing ENZ) sits with a designable dielectric resonator layer (the density
field rho). The optimizer shapes rho so the bias switch maximally changes the reflection -- i.e. it
designs the metasurface geometry to MAXIMISE the modulation contrast of the actual ENZ device.

This fuses everything: carrier-density -> graded-ENZ eps (the quantitative coupling) -> differentiable
FDTD -> jax.grad -> topology optimization of the geometry for the device figure of merit.

GATE: starting from a gray design, the contrast |R_on - R_off| (each a TWO-state FDTD with the graded ITO
      Drude) strongly INCREASES and the design BINARISES (a manufacturable 0/1 pattern). Skip if no JAX.

Run: python -m validation.fdtd_modulator_topology_design
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dynameta.materials import DrudeOptical, M_E
from dynameta.optics.fdtd_nd import _cpml_z, _have_jax
from dynameta.optics.fdtd_seam import _eps_to_fdtd_layer

C = 299792458.0
LAM = 1500e-9
EPS_HI = 12.0
ITO_DRUDE = DrudeOptical(eps_inf=3.9, m_opt_kg=0.35 * M_E, gamma_rad_s=1.0e14)
N_ITO = 3                      # graded ITO sublayers (z-cells)
# carrier density (m^-3) per ITO sublayer (front..back): OFF = background; ON = accumulation crossing ENZ
N_OFF = np.array([4.0e26, 4.0e26, 4.0e26])
N_ON = np.array([9.0e26, 6.0e26, 4.5e26])


def _ito_drude_per_cell(n_m3):
    """(eps_inf, wp, gamma) per ITO sublayer from its carrier density, via the one-Drude inversion."""
    layers = [_eps_to_fdtd_layer(1e-9, complex(ITO_DRUDE.eps(LAM, n_m3=float(ni))), LAM) for ni in n_m3]
    return (np.array([L.eps_inf for L in layers]), np.array([L.drude_wp_rad_s for L in layers]),
            np.array([L.drude_gamma_rad_s for L in layers]))


def _grid():
    res, n_max = 11, np.sqrt(EPS_HI)
    dz = LAM / (res * n_max); dx = dz
    dt = 0.5 / (C * np.sqrt(1.0 / dx ** 2 + 1.0 / dz ** 2))
    pad = 1.1 * LAM
    n_des = 4
    Lz = 2.0 * pad + (n_des + N_ITO) * dz
    nz = int(round(Lz / dz)) + 1
    k0 = int(round(pad / dz))
    des = np.zeros(nz, bool); des[k0:k0 + n_des] = True
    itoz = np.arange(k0 + n_des, k0 + n_des + N_ITO)
    k_src = max(2, int(round(0.4 * pad / dz))); k_pL = int(round(0.7 * pad / dz))
    k_pR = int(round((pad + (n_des + N_ITO) * dz + 0.3 * pad) / dz))
    fc = C / LAM; tau = 4.0 / fc; t0 = 3.0 * tau
    nsteps = int(round((2.0 * t0 + 2.0 * (Lz / C) + 10.0 * tau) / dt))
    tg = np.arange(nsteps) * dt
    src = np.exp(-((tg - t0) / tau) ** 2) * np.cos(2.0 * np.pi * fc * (tg - t0))
    ix = int(np.argmin(np.abs(np.fft.rfftfreq(nsteps, dt) - fc)))
    return dict(nx=14, nz=nz, n_des=n_des, dx=dx, dz=dz, dt=dt, nsteps=nsteps, des=des, itoz=itoz,
                k_src=k_src, k_pL=k_pL, k_pR=k_pR, src=src, cpml=_cpml_z(nz, dz, dt, 8), ix=ix)


def main():
    print("[fmt] === ENZ modulator GEOMETRY design by topology optimization (graded ITO, 2 states) ===", flush=True)
    if not _have_jax():
        print("[fmt] JAX not installed -> SKIP (exit 0)", flush=True)
        return True
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from dynameta.optics.fdtd_nd import _run_2d_te_jax
    from dynameta.optics.topology_opt import binarization, eps_from_density, topology_optimize

    g = _grid(); nx, nz = g["nx"], g["nz"]
    des = jnp.asarray(g["des"]); itoz = g["itoz"]
    z2 = jnp.zeros((nx, nz))
    a2 = (g["dx"], g["dz"], g["dt"], g["nsteps"], g["k_src"], g["k_pL"], g["k_pR"], jnp.asarray(g["src"]), g["cpml"])

    # constant per-state grids: ITO eps_inf / wp / gamma set on the ITO z-cells (design region filled later)
    def _state_grids(n_prof):
        einf, wp, gam = _ito_drude_per_cell(n_prof)
        E = np.ones((nx, nz)); W = np.zeros((nx, nz)); G = np.zeros((nx, nz))
        E[:, itoz] = einf[None, :]; W[:, itoz] = wp[None, :]; G[:, itoz] = gam[None, :]
        return jnp.asarray(E), jnp.asarray(W), jnp.asarray(G)
    Eoff, Woff, Goff = _state_grids(N_OFF)
    Eon, Won, Gon = _state_grids(N_ON)

    vac = _run_2d_te_jax(jnp.ones((nx, nz)), z2, z2, z2, *a2)[0]
    mL = jnp.fft.rfft(vac.mean(axis=1))[g["ix"]]

    def _R(Ec, Wc, Gc, eps_d):
        eps_inf = Ec.at[:, des].set(eps_d)
        eyL = _run_2d_te_jax(eps_inf, Wc, Gc, z2, *a2)[0]
        return jnp.abs(jnp.fft.rfft((eyL - vac).mean(axis=1))[g["ix"]] / mL) ** 2

    def contrast_loss(rho_p):                               # rho_p: (nx, n_des) design density (projected)
        eps_d = eps_from_density(rho_p, 1.0, EPS_HI)
        R_off = _R(Eoff, Woff, Goff, eps_d)
        R_on = _R(Eon, Won, Gon, eps_d)
        return -(R_on - R_off) ** 2

    rho0 = 0.5 * np.ones((nx, g["n_des"]))
    dR0 = float(np.sqrt(-contrast_loss(jnp.asarray(rho0))))
    rho, rho_p, hist = topology_optimize(contrast_loss, rho0, filter_radius=2.0, periodic_axes=(0,),
                                         betas=(2.0, 8.0, 16.0), steps_per_beta=10, lr=0.12)
    dRf = float(np.sqrt(-min(hist)))
    binar = binarization(rho_p)
    gate = bool(dRf > 1.3 * dR0 and binar > 0.7)
    print("[fmt] graded ITO eps(front) OFF={:.3f} ON={:.3f} (ENZ-crossing on accumulation)".format(
        complex(ITO_DRUDE.eps(LAM, n_m3=float(N_OFF[0]))), complex(ITO_DRUDE.eps(LAM, n_m3=float(N_ON[0])))),
        flush=True)
    print("[fmt] design contrast |dR| {:.3f} -> {:.3f} ({:.2f}x, +{:.0f}%) ; binarization={:.2f} -> {}".format(
        dR0, dRf, dRf / max(dR0, 1e-9), 100.0 * (dRf / max(dR0, 1e-9) - 1.0), binar,
        "PASS" if gate else "FAIL"), flush=True)
    print("[fmt] *** ENZ MODULATOR GEOMETRY DESIGN (graded-ENZ optics + topology opt of the cell): {} ***".format(
        "PASS" if gate else "FAIL"), flush=True)
    return gate


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
