"""Lumenairy PMM backend bridge (roadmap v0.5 A2) vs TMM / RCWA-referee / analytic oracles.

GATE A (unstructured == TMM): a 2-layer ASYMMETRIC lossy stack at normal + 30 deg y/p --
        the PMM seam's R/T/r match tmm_reference (< 1e-8; also pins the stack layer ORDER
        and the lab-basis -> Byrnes p-pol r conversion shared with the RCWA bridge).
GATE B (the REFEREE role, rewritten 2026-11-02): a lossy metal lamellar grating (eps -20+2i,
        50 % duty, 120 nm). PMM is the reference, SELF-CONVERGED first: TE in degree to 1e-10;
        TM only ALGEBRAICALLY (a lossy-metal right-angle corner, Li & Granet JOSA A 28:738
        (2011) Type I, Re tau = 0.534) -- the gate checks |R(96) - R(128)| < 2e-5 and uses
        degree 128 (1.3e-5 below the extrapolated 0.508768). TE (E_y along the lines, Laurent
        valid): RCWA-analytic (the 'auto' default) AND RCWA-raster (default lattice) both
        decrease monotonically over n_orders 16/32/48 to < 3e-6 / < 2e-5. TM (E_x across the
        lines): Li's inverse rule ('li', Li 1996 JOSA A 13:1870; 'auto' rasterizes it) decreases
        monotonically over 32/64/128 to < 1.5e-3, at an algebraic rate (> 0.7; measured ~1.3,
        ~2 Re tau); the LAURENT stall is printed, not gated as convergence, and the
        discrimination legs are li >= 5x closer than Laurent at n = 128 and a 1 % duty change
        moving the TM reference by > 10x the li error.
GATE C (OOP tensor specialist vs ANALYTIC): a uniform GYROTROPIC slab (eps_xy = +i g, the
        MagnetoOpticModel convention) through the PMM seam as a (3,3) tensor layer -- the
        zeroth-order reflection Jones matches the hand-derived circular-eigenmode oracle
        J = [[s, -i d], [i d, s]], s = (r+ + r-)/2, d = (r+ - r-)/2, with r+- the scalar
        Airy reflections at n+- = sqrt(eps_r -+ g) (eps.(x +- i y) = (eps_r -+ g)(x +- i y)
        -- hand-derived eigenpair; < 1e-3, the documented PMM OOP floor).
GATE D (scope guards): partial-y rectangle, structured-grid EpsField, and incidence_side=
        'bottom' all raise (conical is NO LONGER a guard -- GATE F solves it).
GATE E (per-layer absorption): PMM at RCWA-bridge parity -- budget closure + the RCWA split
        agreeing with the PMM reference on a lossy metal-grating / lossy-spacer stack (TE tight;
        TM at n_orders 256, past Laurent's pre-asymptotic overshoot -- rewritten 2026-11-02).
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

    # ---- GATE B: PMM as the convergence referee for RCWA (metal lamellar grating) ----
    # REWRITTEN 2026-11-02. The old gate ("RCWA/laurent TM converges to the PMM ref, 7.3e-4 at
    # n = 32") passed by COINCIDENCE: at n_orders >= 32 the old raster lattice max(128, 4 N + 1)
    # went ODD (S = 129), putting the line edges mid-pixel -- the realised duty was 65/129 =
    # 0.50388, and that edge-quantization error (dR/d(duty) = 2.34 on TM here) happened to cancel
    # most of Laurent's TM error. Properly sampled raster and analytic Laurent both sit at ~3e-2 at
    # n = 32 (and 6.5e-3 at n = 128): Laurent's rule is the wrong factorization for E_x across a
    # high-contrast wall (Li 1996), it converges only slowly and non-monotonically. Measured
    # 2026-11-02, all vs the converged references below:
    #   TE  n_orders      16       32       48     (Laurent valid)
    #       analytic    3.5e-5   4.5e-6   1.3e-6
    #       raster new  4.8e-5   1.7e-5   8.3e-6   (S = 128 / 256 / 384, max(128, 8 N))
    #       raster old  4.8e-5   4.6e-3   3.1e-3   (S = 128 / 129 / 193 -- duty 0.5039 at 129:
    #                                               dR/d(duty)_TE 1.19 x 0.00388 = 4.6e-3, exact)
    #   TM  n_orders      32       64       128
    #       li, raster  4.7e-3   1.7e-3   8.0e-4   (algebraic, ~N^-1.15 fitted over 256..768)
    #       analytic L  3.0e-2   9.1e-3   6.5e-3   (overshoots: R 0.478 -> 0.518 -> 0.515 ... 0.509
    #                                               at n = 768, from ABOVE; li comes from below)
    # TM REFERENCE (is the ~4.6e-3 'li' "floor" the PMM ref or the RCWA? -- neither is buggy): the
    # PMM is itself only algebraically convergent on this lossy-metal corner (Li & Granet 2011,
    # Type I, Re tau 0.534 from lumenairy's classify_from_grating): bridge PMM degree 24/32/48/64/
    # 96/128 = 0.508291/0.508511/0.508661/0.508710/0.508744/0.508755, fitted R_inf - c d^-2.17 ->
    # 0.508768 (lumenairy pmm_efficiency_1d, 1 and 6 elements/region, agree to 1.5e-7); RCWA 'li'
    # n_orders 256/384/512/768 = 0.508391/0.508524/0.508587/0.508647, fitted N^-1.15 -> 0.508747;
    # Laurent closes in from above (0.50945 at 768). So the old degree-24 reference was 4.8e-4
    # LOW, and 'li' at n = 32 is 5.0e-3 low simply because it converges as ~N^-1.15 (not a floor).
    lines = Inclusion(shape=Rectangle(PER / 2.0, PER / 2.0, 0.5 * PER, PER), material="metal")
    g_lays = [Layer("grating", 120e-9, "air", inclusions=[lines])]
    args_te = (_design(g_lays, pol="y"), None, {}, LAM, 1.0 + 0j, 1.5 + 0j)
    args_tm = (_design(g_lays, pol="x"), None, {}, LAM, 1.0 + 0j, 1.5 + 0j)
    pmm_te = [make_lumenairy_pmm_solver(degree=dg, n_orders=21)(*args_te).R for dg in (32, 48)]
    pmm_tm = [make_lumenairy_pmm_solver(degree=dg, n_orders=21)(*args_tm).R for dg in (96, 128)]
    ref_te, ref_tm = pmm_te[-1], pmm_tm[-1]
    self_te, self_tm = abs(pmm_te[1] - pmm_te[0]), abs(pmm_tm[1] - pmm_tm[0])
    print("[lpb]   PMM refs: TE R = {:.10f} (degree 32 -> 48: {:.1e}); TM R = {:.7f} (degree 96 "
          "-> 128: {:.1e}, algebraic corner convergence)".format(ref_te, self_te, ref_tm, self_tm),
          flush=True)
    te_n = (16, 32, 48)
    e_ana = [abs(make_lumenairy_rcwa_solver(n_orders=n)(*args_te).R - ref_te) for n in te_n]
    e_ras = [abs(make_lumenairy_rcwa_solver(n_orders=n, inclusion_model="raster")(*args_te).R
                 - ref_te) for n in te_n]
    print("[lpb]   TE |R - PMM| n_orders 16/32/48: analytic {} | raster (default S) {}".format(
        " ".join("{:.1e}".format(e) for e in e_ana), " ".join("{:.1e}".format(e) for e in e_ras)),
        flush=True)
    tm_n = (32, 64, 128)
    e_li = [abs(make_lumenairy_rcwa_solver(n_orders=n, formulation="li")(*args_tm).R - ref_tm)
            for n in tm_n]
    e_lau = [abs(make_lumenairy_rcwa_solver(n_orders=n)(*args_tm).R - ref_tm) for n in tm_n]
    rate = float(np.log(e_li[0] / e_li[-1]) / np.log(tm_n[-1] / tm_n[0]))
    print("[lpb]   TM |R - PMM| n_orders 32/64/128: li (raster) {} (rate N^-{:.2f}) | Laurent "
          "(analytic, NOT gated as convergence) {}".format(
              " ".join("{:.1e}".format(e) for e in e_li), rate,
              " ".join("{:.1e}".format(e) for e in e_lau)), flush=True)
    # discrimination: a 1 % duty change moves the TM reference far more than the li error
    wide = Inclusion(shape=Rectangle(PER / 2.0, PER / 2.0, 0.51 * PER, PER), material="metal")
    d_duty = abs(make_lumenairy_pmm_solver(degree=128, n_orders=21)(
        _design([Layer("grating", 120e-9, "air", inclusions=[wide])], pol="x"), None, {}, LAM,
        1.0 + 0j, 1.5 + 0j).R - ref_tm)
    g_b = bool(self_te < 1e-9 and self_tm < 2e-5
               and e_ana[0] > e_ana[1] > e_ana[2] and e_ana[2] < 3e-6
               and e_ras[0] > e_ras[1] > e_ras[2] and e_ras[2] < 2e-5
               and e_li[0] > e_li[1] > e_li[2] and e_li[2] < 1.5e-3 and rate > 0.7
               and e_lau[-1] > 5.0 * e_li[-1] and d_duty > 10.0 * e_li[-1])
    ok = ok and g_b
    print("[lpb] GATE B: PMM refs self-converged; TE analytic + raster monotone to {:.1e} / {:.1e};"
          " TM li monotone to {:.1e}; Laurent/li at 128 = {:.1f}x; 1 % duty moves TM R {:.1e} "
          "-> {}".format(e_ana[2], e_ras[2], e_li[2], e_lau[-1] / e_li[-1], d_duty,
                         "PASS" if g_b else "FAIL"), flush=True)

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
    # RCWA's per-layer split (absorption=True rasterizes under 'auto'; lumenairy integrates
    # Im(eps)|E|^2 on a 64-point x grid, 8 z-midpoints per layer, normalised to 1 - R - T).
    # REWRITTEN 2026-11-02. The old leg ("strictly shrinking over n_orders 32/64/96: grating
    # 0.124 -> 0.068 -> 0.056 toward PMM 0.0457") rode the same ODD-lattice artifact as GATE B:
    # S = 129/257/385 realised a duty of 0.5039/0.5019/0.5013, a drift that shrinks with N. On
    # the 2026-11-02 lattice (S = 8 N: the geometry exact) the Laurent TM split is
    # NON-MONOTONE -- distance to the degree-24 PMM split used here 0.020 / 0.049 / 0.033 at
    # 32/64/96 and 5.0e-4 at 256; to the degree-128 split 0.011 / 3.2e-3 / 9.8e-4 / 1.7e-3 at
    # 128/192/256/320 (Laurent's TM overshoot, cf. GATE B; 'li' is no cure here:
    # its split wanders 0.049-0.060 on the grating, the E_x quadrature across the walls). PMM
    # split, degree 24 -> 128: grating 0.045665 -> 0.045756, spacer 0.356931 -> 0.356455. So the
    # TM leg is AGREEMENT past the transient (n_orders 256, < 3e-3) with the n = 64 overshoot
    # printed, and a TE leg (Laurent valid) is the tight cross-engine check: |d| 5e-5 at n = 32
    # (RCWA's split is quadrature-limited near 1e-4 in TE: 9e-5 at n = 96). A key swap or a
    # mis-normalized map is off by >= 0.02 (TM) / 0.24 (TE) -- direction-sensitive either way.
    def _split_dist(r_a, r_b):
        return max(abs(r_a.per_region_absorption[k] - r_b.per_region_absorption[k])
                   for k in ("grating", "spacer"))
    dists = []
    for n in (64, 256):
        r_r = make_lumenairy_rcwa_solver(n_orders=n, absorption=True)(
            d_abs, None, {}, LAM, 1.0 + 0j, 1.5 + 0j)
        dists.append(_split_dist(r_p, r_r) if keys_ok else 1.0)
    d_te = _design(lossy_lays, pol="y", extra=(("lossy", complex(6.0, 0.8)),))
    r_pte = make_lumenairy_pmm_solver(degree=24, n_orders=21, absorption=True)(
        d_te, None, {}, LAM, 1.0 + 0j, 1.5 + 0j)
    r_rte = make_lumenairy_rcwa_solver(n_orders=32, absorption=True)(
        d_te, None, {}, LAM, 1.0 + 0j, 1.5 + 0j)
    dist_te = _split_dist(r_pte, r_rte) if keys_ok else 1.0
    g_e = bool(keys_ok and close_p < 1e-6 and self_conv < 1e-3
               and dists[1] < 3e-3 and dist_te < 3e-4)
    ok = ok and g_e
    print("[lpb] GATE E: PMM per-layer absorption -- budget closure {:.1e}; PMM degree-"
          "stability {:.1e}; RCWA split vs PMM: TM n_orders 64 {:.3f} (Laurent overshoot, "
          "printed) -> 256 {:.1e} (< 3e-3); TE n_orders 32 {:.1e} (< 3e-4) [PMM TM grating/"
          "spacer = {:.4f}/{:.4f}] -> {}".format(
              close_p, self_conv, dists[0], dists[1], dist_te,
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
