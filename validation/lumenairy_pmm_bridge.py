"""Lumenairy PMM backend bridge (roadmap v0.5 A2) vs TMM / RCWA-referee / analytic oracles.

GATE A (unstructured == TMM): a 2-layer ASYMMETRIC lossy stack at normal + 30 deg y/p --
        the PMM seam's R/T/r match tmm_reference (< 1e-8; also pins the stack layer ORDER
        and the lab-basis -> Byrnes p-pol r conversion shared with the RCWA bridge).
GATE B (the REFEREE role): a lossy metal lamellar grating in TM (E_x across the lines,
        the slow-RCWA case): PMM (spectral, no Fourier floor, high degree) is the
        reference; the RCWA bridge (1-D lamellar fast path) error |R - R_pmm| DECREASES
        monotonically with n_orders and converges to < 2e-3 (measured 7.3e-4 at n = 32) --
        two unrelated methods agreeing on a lossy metal TM cell; PMM adjudicating RCWA
        truncation is the designed synergy.
GATE C (OOP tensor specialist vs ANALYTIC): a uniform GYROTROPIC slab (eps_xy = +i g, the
        MagnetoOpticModel convention) through the PMM seam as a (3,3) tensor layer -- the
        zeroth-order reflection Jones matches the hand-derived circular-eigenmode oracle
        J = [[s, -i d], [i d, s]], s = (r+ + r-)/2, d = (r+ - r-)/2, with r+- the scalar
        Airy reflections at n+- = sqrt(eps_r -+ g) (eps.(x +- i y) = (eps_r -+ g)(x +- i y)
        -- hand-derived eigenpair; < 1e-3, the documented PMM OOP floor).
GATE D (scope guards): partial-y rectangle, structured-grid EpsField, and conical azimuth
        all raise.

Honest SKIP (exit 0 + banner) when lumenairy is not importable.

Run: python -m validation.lumenairy_pmm_bridge
"""
import importlib.util
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dynameta.core.eps_field import EpsField
from dynameta.geometry import Design, Inclusion, Layer, Stack, UnitCell
from dynameta.geometry.cross_section import Rectangle
from dynameta.geometry.specs import OpticalSpec
from dynameta.materials import ConstantOptical, Material, MaterialRegistry

LAM = 1.31e-6
PER = 600e-9


def _design(layers, *, pol="y", theta=0.0, extra=()):
    reg = MaterialRegistry()
    reg.add(Material("air", ConstantOptical(1.0 + 0j)))
    reg.add(Material("glass", ConstantOptical(complex(1.5 ** 2))))
    reg.add(Material("hi", ConstantOptical(complex(4.0, 0.3))))
    reg.add(Material("lo", ConstantOptical(complex(2.1))))
    reg.add(Material("metal", ConstantOptical(complex(-20.0, 2.0))))
    for nm, eps in extra:
        reg.add(Material(nm, ConstantOptical(eps)))
    return Design(name="pmm", unit_cell=UnitCell.square(PER),
                  stack=Stack(layers=layers, superstrate_material="air",
                              substrate_material="glass"),
                  electrodes=[], materials=reg,
                  optical=OpticalSpec(polarization=pol, incidence_angle_deg=theta))


def _airy_r(n1, n2, n3, d, lam):
    """Two-interface (Airy) reflection coefficient at normal incidence."""
    r12 = (n1 - n2) / (n1 + n2)
    r23 = (n2 - n3) / (n2 + n3)
    ph = np.exp(2j * np.pi / lam * n2 * d * 2.0)
    return (r12 + r23 * ph) / (1.0 + r12 * r23 * ph)


def main():
    if importlib.util.find_spec("lumenairy") is None:
        print("[lpb] *** SKIP: lumenairy not installed -- PMM gates not run ***", flush=True)
        return True
    from dynameta.optics.lumenairy_bridge import make_lumenairy_rcwa_solver
    from dynameta.optics.lumenairy_bridge.pmm_backend import (design_to_pmm_stack,
                                                              make_lumenairy_pmm_solver)
    from dynameta.optics.tmm_reference import make_layered_tmm_solver

    print("[lpb] === Lumenairy PMM bridge vs TMM / RCWA-referee / analytic ===", flush=True)
    ok = True
    tmm = make_layered_tmm_solver()
    pmm = make_lumenairy_pmm_solver(degree=14, n_orders=15)

    # ---- GATE A: unstructured asymmetric stack vs TMM ----
    lays = [Layer("a", 120e-9, "hi"), Layer("b", 200e-9, "lo")]
    worstA = 0.0
    for pol, th in (("y", 0.0), ("y", 30.0), ("p", 30.0)):
        d = _design(lays, pol=pol, theta=th)
        r_t = tmm(d, None, {}, LAM, 1.0 + 0j, 1.5 + 0j)
        r_p = pmm(d, None, {}, LAM, 1.0 + 0j, 1.5 + 0j)
        worstA = max(worstA, abs(r_p.R - r_t.R), abs(r_p.T - r_t.T), abs(r_p.r - r_t.r))
    g_a = bool(worstA < 1e-8)
    ok = ok and g_a
    print("[lpb] GATE A: unstructured vs TMM (normal + 30deg y/p): worst |d| = {:.2e} -> {}"
          .format(worstA, "PASS" if g_a else "FAIL"), flush=True)

    # ---- GATE B: PMM as the convergence referee for RCWA (metal TM grating) ----
    lines = Inclusion(shape=Rectangle(PER / 2.0, PER / 2.0, 0.5 * PER, PER), material="metal")
    g_lays = [Layer("grating", 120e-9, "air", inclusions=[lines])]
    d = _design(g_lays, pol="x")                          # E_x across the lines = TM
    ref = make_lumenairy_pmm_solver(degree=24, n_orders=21)(d, None, {}, LAM,
                                                            1.0 + 0j, 1.5 + 0j)
    errs = []
    for n in (8, 16, 32):
        rc = make_lumenairy_rcwa_solver(n_orders=n)(d, None, {}, LAM, 1.0 + 0j, 1.5 + 0j)
        errs.append(abs(rc.R - ref.R))
        print("[lpb]   RCWA n_orders {:2d}: R = {:.5f} (PMM ref {:.5f}, |d| = {:.1e})"
              .format(n, rc.R, ref.R, errs[-1]), flush=True)
    r_li = make_lumenairy_rcwa_solver(n_orders=32, formulation="li")(
        d, None, {}, LAM, 1.0 + 0j, 1.5 + 0j)
    print("[lpb]   (formulation='li' n_orders 32 for contrast: |d| = {:.1e})".format(
        abs(r_li.R - ref.R)), flush=True)
    # measured: the 1-D lamellar fast path converges to the spectral PMM reference at
    # 7.3e-4 by n_orders = 32 -- two UNRELATED methods agreeing on a lossy metal TM cell
    g_b = bool(errs[-1] < 2e-3 and errs[-1] < errs[1] < errs[0])
    ok = ok and g_b
    print("[lpb] GATE B: RCWA error vs PMM reference decreases ({:.1e} -> {:.1e}) and "
          "converges < 2e-3 -> {}".format(errs[0], errs[-1], "PASS" if g_b else "FAIL"),
          flush=True)

    # ---- GATE C: OOP gyrotropic tensor vs the analytic circular-eigenmode Jones ----
    eps_r, g = 5.0, 0.05
    d_mo = 1.0e-6
    eps_t = np.array([[eps_r, 1j * g, 0.0], [-1j * g, eps_r, 0.0],
                      [0.0, 0.0, eps_r]], dtype=complex)
    d = _design([Layer("mo", d_mo, "hi")], pol="x")
    ef = {"mo": EpsField(tensor=eps_t)}
    stack, _ = design_to_pmm_stack(d, LAM, eps_by_region=ef, degree=14, n_orders=15)
    stack.set_source(LAM, theta=0.0)
    orders, R2, T2, jones = stack.solve()
    n_p = np.sqrt(eps_r - g)                              # eps.(x + i y) = (eps_r - g)(x + i y)
    n_m = np.sqrt(eps_r + g)
    rp = _airy_r(1.0, n_p, 1.5, d_mo, LAM)
    rm = _airy_r(1.0, n_m, 1.5, d_mo, LAM)
    s, dd = (rp + rm) / 2.0, (rp - rm) / 2.0
    J_an = np.array([[s, -1j * dd], [1j * dd, s]])
    dC = float(np.max(np.abs(np.asarray(jones) - J_an)))
    g_c = bool(dC < 1e-3)
    ok = ok and g_c
    print("[lpb] GATE C: gyrotropic OOP-capable tensor slab -- PMM reflection Jones vs "
          "circular-eigenmode analytic: max |dJ| = {:.2e} -> {}".format(
              dC, "PASS" if g_c else "FAIL"), flush=True)

    # ---- GATE D: scope guards ----
    g_d = 0
    half_y = Inclusion(shape=Rectangle(PER / 2.0, PER / 4.0, 0.3 * PER, 0.5 * PER),
                       material="hi")
    try:
        pmm(_design([Layer("bad", 100e-9, "air", inclusions=[half_y])], pol="y"),
            None, {}, LAM, 1.0 + 0j, 1.5 + 0j)
    except ValueError:
        g_d += 1
    grid = EpsField(values_zyx=np.full((3, 1, 8), 2.0 + 0j),
                    z_axis_u=np.array([0.0, 50.0, 100.0]),
                    x_axis_u=np.arange(8.0), y_axis_u=np.array([0.0]))
    try:
        pmm(_design([Layer("a", 100e-9, "hi")], pol="y"), None, {"a": grid}, LAM,
            1.0 + 0j, 1.5 + 0j)
    except ValueError:
        g_d += 1
    try:
        d_con = _design(lays, pol="y")
        object.__setattr__(d_con.optical, "azimuth_deg", 20.0)
        pmm(d_con, None, {}, LAM, 1.0 + 0j, 1.5 + 0j)
    except (NotImplementedError, Exception) as exc:
        g_d += 1 if isinstance(exc, NotImplementedError) else 0
    g_d = bool(g_d == 3)
    ok = ok and g_d
    print("[lpb] GATE D: partial-y / gridded-structured / conical guards -> {}".format(
        "PASS" if g_d else "FAIL"), flush=True)

    print("[lpb] *** LUMENAIRY PMM BRIDGE: {} ***".format("PASS" if ok else "FAIL"),
          flush=True)
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
