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
GATE D (scope guards): partial-y rectangle, structured-grid EpsField, and incidence_side=
        'bottom' all raise (conical is NO LONGER a guard -- GATE F solves it).
GATE E (per-layer absorption): PMM at RCWA-bridge parity -- budget closure + the RCWA split
        converging toward the PMM reference on a lossy metal-grating / lossy-spacer stack.
GATE F (CONICAL s/p synthesis, audit 8.1-1 / consumer-gap B): the bridge synthesizes the
        rotated s/p totals + co-pol r/t from lumenairy 5.22 per_order_amplitudes (the native
        lab rows are s/p mixtures at phi != 0). Three oracles: (1) a UNIFORM slab at conical
        vs analytic oblique Airy -- machine-exact, incl. |r|^2 == R and R + T = 1, pinning the
        p-pol sec^2-theta normalization; (2) phi -> 0 reduction on a patterned grating
        reproduces the validated in-plane R AND co-pol r (magnitude + phase); (3) a lossless
        conical grating conserves energy and a metal grating's R genuinely MOVES with phi.

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


def _design(layers, *, pol="y", theta=0.0, phi=0.0, extra=(), sub="glass"):
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
                              substrate_material=sub),
                  electrodes=[], materials=reg,
                  optical=OpticalSpec(polarization=pol, incidence_angle_deg=theta,
                                      azimuth_deg=phi))


def _airy_oblique_R(n0, n1, n2, d, lam, theta0_rad, pol):
    """Reflectance of a single slab n0|n1|n2 at oblique incidence (s|p), exact -- the
    machine-precision oracle for the conical synthesis on a UNIFORM (unpatterned) cell."""
    k0 = 2.0 * np.pi / lam
    s0 = n0 * np.sin(theta0_rad)
    kz = lambda n: np.sqrt((n * k0) ** 2 - (k0 * s0) ** 2 + 0j)
    kz0, kz1, kz2 = kz(n0), kz(n1), kz(n2)
    if pol == "s":
        rij = lambda ki, kj: (ki - kj) / (ki + kj)
        r01, r12 = rij(kz0, kz1), rij(kz1, kz2)
    else:
        rij = lambda ni, nj, ki, kj: (nj ** 2 * ki - ni ** 2 * kj) / (nj ** 2 * ki + ni ** 2 * kj)
        r01, r12 = rij(n0, n1, kz0, kz1), rij(n1, n2, kz1, kz2)
    ph = np.exp(1j * kz1 * d)
    r = (r01 + r12 * ph ** 2) / (1.0 + r01 * r12 * ph ** 2)
    return float(abs(r) ** 2)


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

    # ---- GATE D: scope guards (conical is NO LONGER a guard -- it solves, see GATE F) ----
    g_d = 0
    half_y = Inclusion(shape=Rectangle(PER / 2.0, PER / 4.0, 0.3 * PER, 0.5 * PER),
                       material="hi")
    try:
        pmm(_design([Layer("bad", 100e-9, "air", inclusions=[half_y])], pol="y"),
            None, {}, LAM, 1.0 + 0j, 1.5 + 0j)
    except ValueError:
        g_d += 1
    # genuinely x-VARYING values: laterally-uniform grids are in scope (they slice to
    # uniform slabs); only true lateral structure must raise
    grid = EpsField(values_zyx=np.broadcast_to(2.0 + 0.1 * np.arange(8.0),
                                               (3, 1, 8)).astype(complex),
                    z_axis_u=np.array([0.0, 50.0, 100.0]),
                    x_axis_u=np.arange(8.0), y_axis_u=np.array([0.0]))
    try:
        pmm(_design([Layer("a", 100e-9, "hi")], pol="y"), None, {"a": grid}, LAM,
            1.0 + 0j, 1.5 + 0j)
    except ValueError:
        g_d += 1
    # incidence_side='bottom' still raises (the bridge solves TOP incidence only)
    try:
        d_bot = _design(lays, pol="y")
        object.__setattr__(d_bot.optical, "incidence_side", "bottom")
        pmm(d_bot, None, {}, LAM, 1.0 + 0j, 1.5 + 0j)
    except NotImplementedError:
        g_d += 1
    g_d = bool(g_d == 3)
    ok = ok and g_d
    print("[lpb] GATE D: partial-y / gridded-structured / conical guards -> {}".format(
        "PASS" if g_d else "FAIL"), flush=True)

    # ---- GATE E: per-layer absorption parity (audit 8.1-3) -- CROSS-ENGINE oracle:
    # the same lossy 2-layer device (metal TM grating over a lossy spacer) through
    # PMM absorption=True and RCWA absorption=True. The engines share no code below
    # the bridge (spectral-element walls vs Fourier factorization), so agreeing
    # per-LAYER absorbed fractions are an independent check that the PMM internal
    # z-flux map is keyed and normalized right -- and each engine must close its own
    # budget sum(A_i) == 1 - R - T (PMM's flux map vs Rayleigh far field). ----
    lossy_lays = [Layer("grating", 120e-9, "air", inclusions=[lines]),
                  Layer("spacer", 200e-9, "lossy")]
    d_abs = _design(lossy_lays, pol="x", extra=(("lossy", complex(6.0, 0.8)),))
    r_p = make_lumenairy_pmm_solver(degree=24, n_orders=21, absorption=True)(
        d_abs, None, {}, LAM, 1.0 + 0j, 1.5 + 0j)
    r_p2 = make_lumenairy_pmm_solver(degree=32, n_orders=31, absorption=True)(
        d_abs, None, {}, LAM, 1.0 + 0j, 1.5 + 0j)
    keys_ok = (r_p.per_region_absorption is not None
               and set(r_p.per_region_absorption) == {"grating", "spacer"})
    close_p = abs(r_p.A_independent - r_p.A) if r_p.A_independent is not None else 1.0
    self_conv = max(abs(r_p.per_region_absorption[k] - r_p2.per_region_absorption[k])
                    for k in ("grating", "spacer")) if keys_ok else 1.0
    # RCWA's per-layer split on a metal TM cell is Gibbs-limited at low orders (measured:
    # grating 0.124 -> 0.068 -> 0.056 over n_orders 32/64/96 toward PMM's 0.0457, which is
    # itself degree-stable at 2e-5) -- so the cross-engine leg demands CONVERGENCE TOWARD
    # the PMM split (strictly shrinking |d|, direction-sensitive: a key-swap or
    # mis-normalized PMM map would not be approached) plus agreement at n_orders=96.
    dists = []
    for n in (32, 64, 96):
        r_r = make_lumenairy_rcwa_solver(n_orders=n, absorption=True)(
            d_abs, None, {}, LAM, 1.0 + 0j, 1.5 + 0j)
        dists.append(max(abs(r_p.per_region_absorption[k] - r_r.per_region_absorption[k])
                         for k in ("grating", "spacer")) if keys_ok else 1.0)
    g_e = bool(keys_ok and close_p < 1e-6 and self_conv < 1e-3
               and dists[2] < dists[1] < dists[0] and dists[2] < 1.5e-2)
    ok = ok and g_e
    print("[lpb] GATE E: PMM per-layer absorption -- budget closure {:.1e}; PMM degree-"
          "stability {:.1e}; RCWA split converges toward PMM ({:.3f} -> {:.3f} -> {:.3f}, "
          "final < 1.5e-2) [PMM grating/spacer = {:.4f}/{:.4f}] -> {}".format(
              close_p, self_conv, dists[0], dists[1], dists[2],
              r_p.per_region_absorption.get("grating", float("nan")) if keys_ok else float("nan"),
              r_p.per_region_absorption.get("spacer", float("nan")) if keys_ok else float("nan"),
              "PASS" if g_e else "FAIL"), flush=True)

    # ---- GATE F: CONICAL s/p synthesis (audit 8.1-1 / consumer-gap B) ----
    # The bridge synthesizes the rotated s/p totals + co-pol r/t from the per-order complex
    # amplitudes (native lab rows are s/p mixtures at phi != 0). THREE independent oracles:
    #  (1) a UNIFORM slab at conical vs analytic oblique Airy -- machine-exact, pins the p-pol
    #      normalization (|E_inc,p|^2 = sec^2 theta) and the flux weights the totals hinge on;
    #  (2) phi -> 0 reduction on a PATTERNED metal grating: conical(1e-4 deg) must reproduce the
    #      validated in-plane R AND the co-pol r (magnitude + PHASE) -- pins the convention;
    #  (3) genuine conical (phi=30) on a lossless dielectric grating: energy R+T ~ 1 AND phi
    #      MATTERS (conical R differs from the in-plane R -- a phi-ignoring synthesis would not).
    pmm_c = make_lumenairy_pmm_solver(degree=20, n_orders=15)
    uslab = [Layer("u", 180e-9, "uni")]
    worst_uni = 0.0
    for pol, pt in (("y", "s"), ("p", "p")):
        d = _design(uslab, pol=pol, theta=30.0, phi=37.0,
                    extra=(("uni", complex(2.2 ** 2)),), sub="glass")
        r = pmm_c(d, None, {}, LAM, 1.0 + 0j, 1.5 + 0j)
        Ra = _airy_oblique_R(1.0, 2.2, 1.5, 180e-9, LAM, np.radians(30.0), pt)
        worst_uni = max(worst_uni, abs(r.R - Ra), abs(abs(r.r) ** 2 - r.R),
                        abs(r.R + r.T - 1.0))
    # reduction on a lossless dielectric grating (energy must close), phi-matters on a strongly
    # form-birefringent METAL grating (a subwavelength dielectric grating's phi-dependence is a
    # tiny form-birefringence effect; a metal grating's is large -- the anti-triviality guard).
    diel = [Layer("grat", 120e-9, "air",
                  inclusions=[Inclusion(shape=Rectangle(PER / 2.0, PER / 2.0, 0.5 * PER, PER),
                                        material="lossless")])]
    metal = [Layer("grat", 120e-9, "air",
                   inclusions=[Inclusion(shape=Rectangle(PER / 2.0, PER / 2.0, 0.5 * PER, PER),
                                         material="metal")])]
    red = 0.0
    energy_c = 0.0
    phi_moves = 0.0
    pmm_g = make_lumenairy_pmm_solver(degree=20, n_orders=21)
    ex = (("lossless", complex(3.3 ** 2, 0.0)),)
    for pol in ("y", "p"):
        r0 = pmm_g(_design(diel, pol=pol, theta=20.0, phi=0.0, extra=ex), None, {},
                   LAM, 1.0 + 0j, 1.5 + 0j)
        rc = pmm_g(_design(diel, pol=pol, theta=20.0, phi=1e-4, extra=ex), None, {},
                   LAM, 1.0 + 0j, 1.5 + 0j)
        rg = pmm_g(_design(diel, pol=pol, theta=20.0, phi=30.0, extra=ex), None, {},
                   LAM, 1.0 + 0j, 1.5 + 0j)
        red = max(red, abs(r0.R - rc.R), abs(r0.r - rc.r))          # phi->0 reduction (incl. phase)
        energy_c = max(energy_c, abs(rg.R + rg.T - 1.0))            # lossless conical closes
        # metal grating: conical(45deg) vs in-plane R must differ substantially (phi is used)
        m0 = pmm_g(_design(metal, pol=pol, theta=20.0, phi=0.0), None, {}, LAM, 1.0 + 0j, 1.5 + 0j)
        m45 = pmm_g(_design(metal, pol=pol, theta=20.0, phi=45.0), None, {}, LAM, 1.0 + 0j, 1.5 + 0j)
        phi_moves = max(phi_moves, abs(m45.R - m0.R))
    g_f = bool(worst_uni < 1e-6 and red < 1e-8 and phi_moves > 1e-3 and energy_c < 5e-3)
    ok = ok and g_f
    print("[lpb] GATE F: conical s/p -- uniform vs analytic oblique Airy (incl |r|^2==R, R+T=1) "
          "{:.1e}; phi->0 reduction (R + co-pol r phase) {:.1e}; phi-matters {:.3f}; lossless "
          "conical energy |R+T-1| {:.1e} -> {}".format(
              worst_uni, red, phi_moves, energy_c, "PASS" if g_f else "FAIL"), flush=True)

    print("[lpb] *** LUMENAIRY PMM BRIDGE: {} ***".format("PASS" if ok else "FAIL"),
          flush=True)
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
