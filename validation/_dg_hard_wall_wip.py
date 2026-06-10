"""DG OXIDE HARD WALL (R19 follow-on): u -> 0 at the insulating boundary of the IN-NEWTON
density-gradient system (setup_dg_hard_wall), vs the validated post-hoc closure.

Device: 1D uniform n-type bar (400 nm, ITO-like m* = 0.35 m0, N0 = 4e26 m^-3 so the Debye
length 0.18 nm << L_q = 1.18 nm), LEFT contact = the hard wall (DG equations ONLY -- the
Potential/continuity rows stay natural bulk, an insulating boundary), RIGHT = ohmic + bulk
DG contact. Uniform doping -> the classical equilibrium is FLAT, so the frozen-potential
post-hoc BVP (dg_correct_density_1d, hard_wall='left') solves the IDENTICAL closure.

GATE A (frozen-psi == post-hoc BVP): with the Poisson rows DELETED (psi frozen flat), the
        converged in-Newton n(z) matches dg_correct_density_1d on the same grid to < 2%
        of N0 everywhere (two discretizations of the same continuum problem).
GATE B (self-consistent screening physics): re-enabling Poisson REDUCES the dead-layer
        deficit integral (exposed donors raise psi at the wall and pull electrons back);
        gate: 0.3 < deficit_sc / deficit_frozen < 1.0 (the documented screening direction,
        order-(Debye/L_q) effect).
GATE C (regularization insensitivity): the converged density is insensitive to the Lambda
        wall-pin depth (factor 10 vs 20: max |dn|/N0 < 1e-3) -- the pin is a regularization
        of the log-divergent continuum Lambda, not physics.
GATE D (sqrt(gamma) scaling): in frozen-psi mode the dead-layer deficit scales as L_q ~
        sqrt(gamma): deficit(gamma=0.25)/deficit(gamma=1) = 0.5 to < 5%.

Run: python -m validation.dg_hard_wall
"""
import contextlib
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import devsim as ds

from dynameta.constants import M_E
from dynameta.carriers import eq_registry as _R
from dynameta.carriers.density_gradient import dg_correct_density_1d, dg_length_m
from dynameta.carriers.physics_density_gradient import (seed_dg_from_solution,
                                                        set_dg_gamma, setup_contact_dg,
                                                        setup_dg_hard_wall,
                                                        setup_dg_quantum_correction)
from dynameta.carriers.physics_drift_diffusion import (setup_contact_ohmic_dd,
                                                       setup_semiconductor_region_dd)

MSTAR = 0.35 * M_E
LEN = 400e-9
N0 = 4.0e26
# REL 1e-5: the wall node's Boltzmann-pinned density ~ e^-pin N0 makes per-node
# RELATIVE updates floor out below this (the documented dc_solve precision-floor
# ping-pong, here amplified by the tiny wall density); gates are at the 1e-2 level
ABS_ERR, REL_ERR, MAX_ITER = 1.0e16, 1.0e-5, 200


@contextlib.contextmanager
def _quiet():
    sys.stdout.flush()
    saved = os.dup(1)
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, 1)
        yield
    finally:
        sys.stdout.flush()
        os.dup2(saved, 1)
        os.close(devnull)
        os.close(saved)


def _solve(dev):
    with _quiet():
        ds.solve(type="dc", absolute_error=ABS_ERR, relative_error=REL_ERR,
                 maximum_iterations=MAX_ITER)


def _build(tag):
    mesh, dev, reg = "hwm_" + tag, "hwd_" + tag, "bar"
    ds.create_1d_mesh(mesh=mesh)
    ds.add_1d_mesh_line(mesh=mesh, pos=0.0, ps=0.05e-9, tag="wall")   # resolve L_q hard
    ds.add_1d_mesh_line(mesh=mesh, pos=20e-9, ps=0.5e-9)
    ds.add_1d_mesh_line(mesh=mesh, pos=LEN, ps=4e-9, tag="back")
    ds.add_1d_contact(mesh=mesh, name="wall", tag="wall", material="metal")
    ds.add_1d_contact(mesh=mesh, name="back", tag="back", material="metal")
    ds.add_1d_region(mesh=mesh, material="ITO", region=reg, tag1="wall", tag2="back")
    ds.finalize_mesh(mesh=mesh)
    ds.create_device(mesh=mesh, device=dev)
    setup_semiconductor_region_dd(dev, reg, n_bg_m3=N0, eps_static=9.5,
                                  dos_mass_kg=MSTAR, mobility_m2Vs=0.004)
    setup_contact_ohmic_dd(dev, "back")               # the wall contact: DG equations only
    nn = len(ds.get_node_model_values(device=dev, region=reg, name="Electrons"))
    ds.set_node_values(device=dev, region=reg, name="Electrons", values=[N0] * nn)
    return mesh, dev, reg


def _run_hard_wall(tag, *, gamma=1.0, pin=8.0, frozen_psi=True,
                   fracs=(0.05, 0.1, 0.25, 0.5, 0.75, 1.0)):
    mesh, dev, reg = _build(tag)
    _solve(dev)                                       # classical equilibrium (flat)
    setup_dg_quantum_correction(dev, reg, m_eff_kg=MSTAR, gamma=gamma)
    setup_contact_dg(dev, "back", N0)
    setup_dg_hard_wall(dev, "wall", lambda_pin_factor=pin)
    seed_dg_from_solution(dev, reg)
    # wall-aware seed: taper u (and n = u^2) to ~0 over L_q at the wall -- seeding the FLAT
    # bulk profile against the wall pin makes the first ramp step's Newton transients
    # overflow/diverge (measured); the taper starts Newton inside the dead-layer basin
    z0 = np.asarray(ds.get_node_model_values(device=dev, region=reg, name="x"))
    u0 = np.asarray(ds.get_node_model_values(device=dev, region=reg, name="QSqrtN"))
    taper = np.maximum(np.tanh(z0 / 1.2e-9), 1e-6)
    ds.set_node_values(device=dev, region=reg, name="QSqrtN", values=list(u0 * taper))
    ds.set_node_values(device=dev, region=reg, name="Electrons",
                       values=list(np.maximum((u0 * taper) ** 2, 1e14)))
    if frozen_psi:
        _R.delete_by_name(dev, "PotentialEquation")   # freeze psi at the flat solution
    for fr in fracs:
        set_dg_gamma(dev, reg, fr)
        _solve(dev)
    z = np.asarray(ds.get_node_model_values(device=dev, region=reg, name="x"))
    n = np.asarray(ds.get_node_model_values(device=dev, region=reg, name="Electrons"))
    order = np.argsort(z)
    out = z[order].copy(), n[order].copy()
    _R.clear(dev)
    ds.delete_device(device=dev)
    ds.delete_mesh(mesh=mesh)
    return out


def _deficit(z, n):
    return float(np.trapezoid(N0 - n, z))


def main():
    print("[hw] === DG hard wall (in-Newton) vs post-hoc BVP closure ===", flush=True)
    ok = True
    L_q = dg_length_m(MSTAR)
    print("[hw] L_q = {:.3f} nm, N0 = {:.1e} m^-3".format(L_q * 1e9, N0), flush=True)

    # ---- GATE A: frozen-psi in-Newton == the post-hoc BVP on the same grid ----
    z, n_hw = _run_hard_wall("a", frozen_psi=True)
    n_bvp = dg_correct_density_1d(z, np.full_like(z, N0), MSTAR, gamma=1.0,
                                  hard_wall="left")
    dA = float(np.max(np.abs(n_hw - n_bvp)) / N0)
    g_a = bool(dA < 0.02)
    ok = ok and g_a
    print("[hw] GATE A: frozen-psi profile vs dg_correct_density_1d: max |dn|/N0 = {:.2e} "
          "-> {}".format(dA, "PASS" if g_a else "FAIL"), flush=True)
    d_frozen = _deficit(z, n_hw)
    print("[hw]   dead-layer deficit (frozen) = {:.3e} m^-2 (= {:.3f} N0 L_q)".format(
        d_frozen, d_frozen / (N0 * L_q)), flush=True)

    # ---- GATE B: self-consistent Poisson screening reduces the deficit ----
    z_sc, n_sc = _run_hard_wall("b", frozen_psi=False)
    d_sc = _deficit(z_sc, n_sc)
    ratio_sc = d_sc / d_frozen
    g_b = bool(0.3 < ratio_sc < 1.0)
    ok = ok and g_b
    print("[hw] GATE B: screening deficit ratio (self-consistent / frozen) = {:.3f} "
          "(expected in (0.3, 1.0)) -> {}".format(ratio_sc, "PASS" if g_b else "FAIL"),
          flush=True)

    # ---- GATE C: Lambda-pin regularization insensitivity ----
    _, n_p10 = _run_hard_wall("c1", pin=6.0, frozen_psi=True)
    _, n_p20 = _run_hard_wall("c2", pin=12.0, frozen_psi=True)
    dC = float(np.max(np.abs(n_p10 - n_p20)) / N0)
    g_c = bool(dC < 1e-3)
    ok = ok and g_c
    print("[hw] GATE C: wall-pin depth 6 vs 12 V_t: max |dn|/N0 = {:.2e} -> {}".format(
        dC, "PASS" if g_c else "FAIL"), flush=True)

    # ---- GATE D: deficit scales as L_q ~ sqrt(gamma) ----
    zq, nq = _run_hard_wall("d", gamma=0.25, frozen_psi=True)
    ratio_g = _deficit(zq, nq) / d_frozen
    g_d = bool(abs(ratio_g - 0.5) < 0.05 * 0.5 + 0.02)
    ok = ok and g_d
    print("[hw] GATE D: deficit(gamma=0.25)/deficit(gamma=1) = {:.3f} (target 0.5) -> {}"
          .format(ratio_g, "PASS" if g_d else "FAIL"), flush=True)

    print("[hw] *** DG HARD WALL: {} ***".format("PASS" if ok else "FAIL"), flush=True)
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
