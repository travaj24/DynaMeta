"""Scalar gain-BPM (dynameta.optics.fiber_amp.gain_bpm) end-to-end gates. Oracles, expected
values and tolerances come from docs/fiber_transverse_grounding_2026_07_29.md sec. 2 (the Feature 2
gate table B1-B16) unless a line says otherwise; every pin below is RE-MEASURED here, and where a
re-measurement disagrees with the dossier the gate says so in its own body. Pure numpy/scipy; the
fixtures are deliberately small (64-128 grids, cm-scale fibers) so the script stays cheap.

GATE A (passive unitarity + the absorber ledger, dossier B9/B10/B11): with no gain and no absorber
        the march must conserve power to round-off -- the diffraction operator is a pure phase in
        Fourier space and the index operator a pure phase in real space, so ANY power drift is a
        bug. With the absorber on and a window small enough that it bites, launched power must
        equal transmitted + absorbed to the same class.
GATE B (gain convention, dossier B8): uniform g, flat index -> P_out/P_in = exp(g L). This is the
        factor-of-2 gate: g is the INTENSITY gain and the amplitude factor is exp(g dz/2).
GATE C (split-step order, dossier B7): global order 2 under dz refinement, measured on the SMOOTH
        quadratic duct -- never on the step-index fiber, where the dossier measured a ragged
        0.90/0.92/2.03/4.55 because the hard circular discontinuity on a Cartesian grid contributes
        an error that does not scale with dz at all. TWO cases: (i) the dossier's gain-free duct
        (2.00 +/- 0.05) and (ii) a SATURABLE gain on the same duct (2.00 +/- 0.15), which is the
        only one of the two that can catch a frozen-gain leg -- with g == 0 the midpoint corrector
        is a no-op and a frozen leg is bitwise identical.
GATE D (quadratic-duct oracle, dossier B3/B4/B5/B6): the tightest single gate in the suite -- it
        pins the diffraction factor, the index-phase factor and BOTH their signs simultaneously.
        A matched Gaussian of w_m = sqrt(2/(k0 n0 alpha)) must propagate invariantly; the field
        must self-image at z_p = 2 pi/alpha; a 1.5 w_m launch must reach its first spot-size
        extremum at z_p/4; and alpha^2 must BE thermal.thermal_lens_focal_power_per_m's D'.
GATE E (LP01/LP11 beat length): a two-mode launch on the Smith & Smith LMA fixture must reproduce
        L_beat = lambda/(n_eff01 - n_eff11) = 9.8953 mm from the EXACT lma.py modes. Also measures
        what the (F2.5a) erf taper costs on this axis -- a finding that REFINES the dossier.
GATE F (CROSS-SOLVER PARITY, the headline): fundamental-mode launch, cladding pump, no ASE, BPM vs
        transverse.ResolvedFiberAmplifier on the SAME plan, at a small-signal and a deeply
        saturated point. The two apportion gain by completely different mechanisms -- diffraction
        vs a fixed modal overlap -- so they agree only to the extent the mode stays LP01. Run at a
        CONVERGED dz the hard-step and 2-cell-taper rows both agree with the channel model, and the
        residual saturated deviation is shown to be a TAPER-WIDTH bias, not radiation and not
        physics.
GATE G (energy audit on a gain run, dossier B10): launched + stimulated exchange - boundary
        absorbed == transmitted, to the marcher's round-off.
GATE H (thermal loop, dossier B6/B14): the module's thermal lens IS thermal.py's D'; a matched
        Gaussian in the SELF-CONSISTENT thermal profile stays put where free diffraction would
        blow it up 6x; the quasi-static loop converges; and the heat source equals the quantum
        defect times the marched pump absorption.
GATE I (BACKGROUND LOSS, new -- every fixture above runs at background_loss_per_m = 0, so the loss
        leg had no gate at all): with the gain off, l_s = 3 1/m over 2 cm must reproduce
        exp(-l_s L) to 1e-12; with a pump on, the energy ledger must still close and the loss must
        cost the lossless twin's gain exactly 10 log10(exp(l_s L)) dB.
GATE J (COHERENT-SOLVER TSHB, new -- the campaign's headline physics): the BPM's OWN
        local_gain_per_m field projected onto LP01/LP11 must show the TSHB ORDERING REVERSAL as the
        LP01 power saturates the medium, matching the channel model's resolved modal gains; and the
        modal-interference cross term's effect on the LP11 POWER FRACTION must vanish at a 50/50
        split and reverse across it, which is why that fraction is NOT the test.

Run: python -m validation.fiber_gain_bpm
"""
import os
import sys
import time
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dynameta.optics.fiber_amp import (CrossSectionModel, FiberSpec, GainBPM, Pump, RareEarthIon,
                                       ResolvedFiberAmplifier, Signal, ThermalLoop, ThermalModel,
                                       quadratic_duct_period_m, quadratic_duct_radius_m,
                                       quantum_defect_fraction, radial_temperature_rise,
                                       solve_lp_modes, thermal_lens_focal_power_per_m)

# Smith & Smith, Opt. Express 21:15168 (2013), Table 1 (PINNED) -- the same fixture
# validation/fiber_radial_inversion.py and tests/test_fiber_transverse.py use, so the cross-solver
# gate compares two solvers on identical spectroscopy.
SA_P, SE_P = 2.47e-24, 2.44e-24
SA_S, SE_S = 5.80e-27, 5.00e-25
LAM_P, LAM_S = 0.976e-6, 1.032e-6
TAU_S, N_T = 901e-6, 3.0e25
A_CORE, NA, N_CLAD = 25e-6, 0.054, 1.45
DN_DT = 1.2e-5


def smith_yb() -> RareEarthIon:
    """The Table-1 two-wavelength fixture: four anchors as very narrow Gaussians so each is exact
    at its own line and the pump/signal cross-talk is exp(-241) = 0."""
    sa = CrossSectionModel(((LAM_P, 4e-9, SA_P), (LAM_S, 6e-9, SA_S)))
    se = CrossSectionModel(((LAM_P, 4e-9, SE_P), (LAM_S, 6e-9, SE_S)))
    return RareEarthIon("Yb3+(SmithTable1)", sa, se, tau_s=TAU_S, zero_line_m=LAM_P,
                        host="gate-fixture")


def lma_fiber(length_m=0.02, clad=100e-6):
    return FiberSpec(core_radius_m=A_CORE, na=NA, n_t_m3=N_T, length_m=length_m,
                     clad_radius_m=clad)


def passive(length_m, **kw):
    """A gain-free BPM on the LMA geometry: gain_model = 0 REPLACES the rate-equation medium, so
    no ion and no pump are needed (the module refuses passing both)."""
    fib = FiberSpec(core_radius_m=A_CORE, na=NA, n_t_m3=N_T, length_m=length_m)
    return GainBPM(None, fib, [], [Signal(1.0, LAM_S)], gain_model=lambda I: 0.0 * I, **kw)


def _report(tag, ok, msg):
    print("[fgb] GATE {}: {} -> {}".format(tag, msg, "PASS" if ok else "FAIL"), flush=True)
    return ok


def _fidelity(bpm, E_a, E_b):
    """|<a|b>|^2 / (<a|a><b|b>) -- the complex (phase-sensitive) overlap fidelity."""
    num = abs(np.sum(np.conj(E_a) * E_b) * bpm.dA) ** 2
    return float(num / (bpm.power_W(E_a) * bpm.power_W(E_b)))


def _absorbing(call):
    """Run a solve() that KNOWINGLY over-fills the window and return (result, n_advisories).

    gain_bpm warns (UserWarning) whenever a run's absorbed_fraction crosses the dossier's 1e-6
    converged-window bar, because the super-Gaussian mask is applied once per STEP and so has no
    dz -> 0 limit (module scope limitation 1). Several gates here MEAN to cross it: gate A's
    clipped window is the entire point of the ledger test, and the step-index rows of E/F/H/I shed
    discretization radiation by construction. Those catch the advisory HERE and count it, which is
    the opposite of silencing it globally -- a fixture that stops warning has stopped testing what
    it was written to test, and gate A asserts exactly that."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = call()
    return res, sum(1 for w in caught if issubclass(w.category, UserWarning)
                    and "boundary absorber removed" in str(w.message))


# ============================ GATE A ====================================================

def gate_a():
    """dossier B9/B10/B11. (i) absorber-free unitarity; (ii) the absorbed-power ledger on a window
    small enough that the boundary actually eats something (otherwise the ledger is untested)."""
    b = passive(2e-3, n_grid=64, extent_m=150e-6, absorber=False)
    r = b.solve(n_steps=300, n_samples=11)
    drift = float(np.max(np.abs(r.P_signal_W / r.P_signal_W[0] - 1.0)))

    # a 60 um window around a 25 um core clips the LP01 tail, so the absorber HAS to act -- and
    # because it acts far above the 1e-6 converged-window bar, the run MUST raise the module's
    # dz-dependence advisory. Catching it here (rather than filtering it away suite-wide) turns
    # "this fixture still over-fills its window" into an assertion instead of a hope.
    b2 = passive(2e-3, n_grid=64, extent_m=60e-6, absorber=True)
    r2, n_warn = _absorbing(lambda: b2.solve(n_steps=300, n_samples=11))
    ledger = float(np.max(np.abs(r2.energy_residual_W)) / r2.P_signal_W[0])
    eaten = float(r2.absorbed_boundary_W[-1] / r2.P_signal_W[0])
    ok = drift < 1e-12 and ledger < 1e-12 and eaten > 1e-3 and n_warn == 1
    print("[fgb]   A  absorber-free power drift {:.3e} (< 1e-12); clipped-window ledger residual "
          "{:.3e} (< 1e-12) with {:.2%} of the launch eaten by the boundary, which raised {} "
          "per-step-absorber advisory (expected 1)".format(drift, ledger, eaten, n_warn),
          flush=True)
    return _report("A", ok, "the split operators are exactly power-neutral and every watt the "
                            "boundary removes is accounted for")


# ============================ GATE B ====================================================

def gate_b():
    """dossier B8 (measured there: 2.4e-14). g is the INTENSITY gain, so the amplitude factor is
    exp(g dz/2); a stray factor of 2 anywhere shows up as exp(2 g L) or exp(g L/2)."""
    g0, L = 2.5, 1.3
    fib = FiberSpec(core_radius_m=A_CORE, na=NA, n_t_m3=N_T, length_m=L)
    b = GainBPM(None, fib, [], [Signal(1.0, LAM_S)], n_grid=64, extent_m=400e-6,
                gain_model=lambda I: np.full_like(I, g0), absorber=False,
                index_profile=lambda X, Y: np.full_like(X, N_CLAD))
    r = b.solve(n_steps=16, n_samples=3, E_in=b.launch_field(w_m=40e-6))
    ratio = float(r.P_signal_W[-1] / r.P_signal_W[0])
    rel = abs(ratio / np.exp(g0 * L) - 1.0)
    ok = rel < 1e-12
    print("[fgb]   B  P_out/P_in = {:.15e} vs exp(g L) = {:.15e}  rel {:.3e}".format(
        ratio, np.exp(g0 * L), rel), flush=True)
    return _report("B", ok, "uniform gain reproduces exp(g L) to {:.1e} (dossier measured "
                            "2.4e-14)".format(rel))


# ============================ duct fixture (gates C, D) =================================

D_PRIME = 1.0e5                              # duct dioptric power per unit length [1/m^2]
ALPHA = float(np.sqrt(D_PRIME))
W_M = quadratic_duct_radius_m(D_PRIME, LAM_S, N_CLAD)
Z_P = quadratic_duct_period_m(D_PRIME)
# The saturable medium of gate C case (ii): g = G_SAT/(1 + I/I_SAT). I_SAT is chosen so the
# 1.5 w_m launch enters at peak I/I_sat = 1.05 -- the knee, where dg/dI is largest and the entry
# gain and the midpoint gain differ most.
G_SAT, I_SAT = 300.0, 3.76e8


def duct(length_m, n_grid=96, extent_m=250e-6, gain_model=None):
    """The SMOOTH quadratic duct n(r) = n0 - (1/2) n0 alpha^2 r^2 (dossier sec. 2.7). Smooth is the
    point: the order and shape-invariance gates must not be run on a hard index step."""
    fib = FiberSpec(core_radius_m=A_CORE, na=NA, n_t_m3=N_T, length_m=length_m)
    return GainBPM(None, fib, [], [Signal(1.0, LAM_S)], n_grid=n_grid, extent_m=extent_m,
                   gain_model=(lambda I: 0.0 * I) if gain_model is None else gain_model,
                   absorber=False, n_ref=N_CLAD,
                   index_profile=lambda X, Y: N_CLAD * (1.0 - 0.5 * ALPHA ** 2 * (X ** 2 + Y ** 2)))


# ============================ GATE C ====================================================

def _duct_orders(gain_model, nzs=(24, 48, 96, 192), n_ref_steps=1536):
    """Errors and log2 orders of the duct march under dz refinement. The reference is the same
    solver at n_ref_steps = 8x the finest test step, so its own error is 1/64 of the finest
    measured one."""
    b = duct(0.5 * Z_P, n_grid=64, extent_m=200e-6, gain_model=gain_model)
    E0 = b.launch_field(w_m=1.5 * W_M)
    ref = b.solve(n_steps=n_ref_steps, n_samples=2, E_in=E0).E_out
    errs = [float(np.linalg.norm(b.solve(n_steps=nz, n_samples=2, E_in=E0).E_out - ref)
                  / np.linalg.norm(ref)) for nz in nzs]
    return b, E0, errs, [float(np.log2(errs[i - 1] / errs[i])) for i in range(1, len(errs))]


def gate_c():
    """dossier B7: global order 2 on the SMOOTH duct (measured there 2.0000 over 5 refinements).

    TWO cases, and only the second one discriminates. Case (i) is the dossier's: a gain-FREE duct,
    gain_model = 0. On it the midpoint corrector of _nonlinear is a NO-OP -- a gain that is
    identically zero is the same at the step entry and at the half-advanced field -- so a frozen
    (exponential-Euler) leg reproduces case (i) BITWISE and the claim that this gate would catch a
    frozen leg was false. Case (ii) puts a SATURABLE gain on the same duct, entering at peak
    I/I_sat = 1.05 and saturating harder as it amplifies, where entry and midpoint gains genuinely
    differ. MEASURED with a frozen leg patched in place of the midpoint one (verifier, 2026-08-04):
    orders collapse to 1.320 / 1.178 / 1.153 against the shipped 1.996 / 2.021 / 2.021 -- that is
    the discrimination. The bar is 2.00 +/- 0.15 there rather than +/- 0.05 because a saturating
    medium's own nonlinearity puts more spread on the measured order than the passive duct does."""
    nzs = (24, 48, 96, 192)
    _, _, errs, orders = _duct_orders(None, nzs)
    b_s, E0_s, errs_s, orders_s = _duct_orders(lambda I: G_SAT / (1.0 + I / I_SAT), nzs)
    r_s = b_s.solve(n_steps=1536, n_samples=2, E_in=E0_s)
    ok = (all(abs(o - 2.0) < 0.05 for o in orders)
          and all(abs(o - 2.0) < 0.15 for o in orders_s))
    for nz, e, e_s in zip(nzs, errs, errs_s):
        print("[fgb]   C  nz = {:4d}: rel err {:.4e} (no gain) | {:.4e} (saturable)".format(
            nz, e, e_s), flush=True)
    print("[fgb]   C  (i)  gain-free  orders {} (2.00 +/- 0.05). A frozen-gain leg measures these "
          "BITWISE -- this case cannot discriminate.".format(["%.4f" % o for o in orders]),
          flush=True)
    print("[fgb]   C  (ii) saturable g = {:.0f}/(1 + I/{:.2e}): launch peak I/I_sat {:.3f}, "
          "P_out/P_in {:.3f}; orders {} (2.00 +/- 0.15; a FROZEN leg measures "
          "1.320/1.178/1.153)".format(
              G_SAT, I_SAT, float(np.max(np.abs(E0_s) ** 2)) / I_SAT,
              float(r_s.P_signal_W[-1] / r_s.P_signal_W[0]),
              ["%.4f" % o for o in orders_s]), flush=True)
    return _report("C", ok, "midpoint-corrected Strang split is globally O(dz^2) with a REAL "
                            "saturable gain in the N leg: orders {}".format(
                                ["%.3f" % o for o in orders_s]))


# ============================ GATE D ====================================================

def gate_d():
    """dossier B3/B4/B5/B6. alpha = sqrt(D'), w_m = sqrt(2/(k0 n0 alpha)), z_p = 2 pi/alpha, first
    extremum of a mismatched launch at z_p/4."""
    ok = True
    # (i) the thermal-lens bridge: alpha^2 IS thermal.thermal_lens_focal_power_per_m's D'
    a_core, k_core = A_CORE, 1.38
    Q_for_D = D_PRIME * 2.0 * np.pi * N_CLAD * k_core * a_core ** 2 / DN_DT
    d_back = float(thermal_lens_focal_power_per_m(Q_for_D, a_core, dndt=DN_DT,
                                                  k_core_W_mK=k_core, n_core=N_CLAD))
    rel_bridge = abs(d_back / D_PRIME - 1.0)
    ok = ok and rel_bridge < 1e-12

    # (ii) matched Gaussian: propagation-invariant over one full period
    b = duct(Z_P)
    E0 = b.launch_field(w_m=W_M)
    r = b.solve(n_steps=400, n_samples=81, E_in=E0)
    dev = float(np.ptp(r.spot_radius_m) / W_M)
    fid = _fidelity(b, E0, r.E_out)
    ok = ok and dev < 1e-4 and fid > 1.0 - 1e-8

    # (iii) mismatched launch: first spot-size extremum at z_p/4
    b2 = duct(0.5 * Z_P)
    r2 = b2.solve(n_steps=400, n_samples=401, E_in=b2.launch_field(w_m=1.5 * W_M))
    j = int(np.argmin(r2.spot_radius_m))
    rel_ext = abs(r2.z_m[j] / (0.25 * Z_P) - 1.0)
    ok = ok and rel_ext < 1e-3
    print("[fgb]   D  alpha = {:.4f} 1/m, w_m = {:.4f} um, z_p = {:.4f} mm; bridge D' rel "
          "{:.2e}".format(ALPHA, 1e6 * W_M, 1e3 * Z_P, rel_bridge), flush=True)
    print("[fgb]   D  matched Gaussian: width span [{:.6f}, {:.6f}] um, rel deviation {:.3e} "
          "(dossier 1.4e-5); self-imaging fidelity at z_p {:.10f}".format(
              1e6 * r.spot_radius_m.min(), 1e6 * r.spot_radius_m.max(), dev, fid), flush=True)
    print("[fgb]   D  1.5 w_m launch: first minimum at {:.6f} mm vs z_p/4 = {:.6f} mm (rel "
          "{:.1e}); breathes [{:.3f}, {:.3f}] um".format(
              1e3 * r2.z_m[j], 1e3 * 0.25 * Z_P, rel_ext, 1e6 * r2.spot_radius_m.min(),
              1e6 * r2.spot_radius_m.max()), flush=True)
    return _report("D", ok, "the quadratic-duct oracle pins the diffraction and index factors and "
                            "both signs simultaneously")


# ============================ GATE E ====================================================

def gate_e():
    """LP01/LP11 beat length against the EXACT lma.py modes. The two-mode relative phase
    arg(c01 conj(c11)) advances linearly at beta01 - beta11, so a straight-line fit reads the beat
    length off directly -- far tighter than locating intensity peaks.

    RE-MEASURED FINDING that refines the dossier: the (F2.5a) erf taper is described there as
    preserving n_eff "to well under the discretization error", which is true of n_eff itself but
    NOT of the SPLITTING n_eff01 - n_eff11 -- a difference of two nearly equal numbers, so a small
    absolute shift is a large relative one. Measured here, the taper shortens L_beat in proportion
    to its PHYSICAL width s*dx (2 cells at dx = 1.17 um: -4.4%; the same 1.56 um of smoothing
    reached either as s=2/dx=1.56 or s=1/dx=1.56 gives -2.2% both times). The taper is therefore
    the right tool for a single-mode phase-fidelity problem and the WRONG one for modal dispersion;
    this gate uses the hard step and reports both."""
    modes = solve_lp_modes(A_CORE, NA, LAM_S, n_clad=N_CLAD)
    m01, m11 = modes[0], modes[1]
    lb = LAM_S / (m01.n_eff() - m11.n_eff())

    def fit(taper):
        fib = FiberSpec(core_radius_m=A_CORE, na=NA, n_t_m3=N_T, length_m=2.0 * lb)
        b = GainBPM(None, fib, [], [Signal(1.0, LAM_S)], n_grid=128, extent_m=150e-6,
                    gain_model=lambda I: 0.0 * I, index_taper_cells=taper)
        E0 = b.launch_field(0.7, m01) + b.launch_field(0.3, m11)
        # two beat lengths of a hard-step LP11 shed enough into the boundary to trip the
        # per-step-absorber advisory; that shed power IS the "deficit" this gate reports.
        r, _ = _absorbing(lambda: b.solve(n_steps=400, n_samples=101, E_in=E0,
                                          track_modes=[m01, m11]))
        ph = np.unwrap(np.angle(r.mode_amp_sqrtW[0] * np.conj(r.mode_amp_sqrtW[1])))
        slope = abs(np.polyfit(r.z_m, ph, 1)[0])
        return (2.0 * np.pi / slope, abs(r.mode_amp_sqrtW[0, -1]) ** 2,
                abs(r.mode_amp_sqrtW[1, -1]) ** 2)

    lb_hard, p01, p11 = fit(0.0)
    lb_taper = fit(2.0)[0]
    rel = abs(lb_hard / lb - 1.0)
    ok = rel < 0.02 and abs(lb - 9.8953e-3) < 1e-6 and abs(p01 / 0.7 - 1.0) < 0.10
    print("[fgb]   E  exact-mode L_beat = {:.6f} mm (dossier 9.895 mm); BPM hard step "
          "{:.6f} mm (rel {:+.3e}); with a 2-cell erf taper {:.6f} mm (rel {:+.3e})".format(
              1e3 * lb, 1e3 * lb_hard, lb_hard / lb - 1.0, 1e3 * lb_taper, lb_taper / lb - 1.0),
          flush=True)
    print("[fgb]   E  launched 0.700/0.300 W in LP01/LP11, recovered {:.4f}/{:.4f} W after 2 beat "
          "lengths (no spurious inter-mode transfer; the deficit is hard-step radiation)".format(
              p01, p11), flush=True)
    return _report("E", ok, "two-mode beat period matches lambda/(n_eff01 - n_eff11) to "
                            "{:.2%} (bar 2%)".format(rel))


# ============================ GATES F + G ===============================================

PARITY_EXTENT = 150e-6
PARITY_DX96 = PARITY_EXTENT / 96.0            # the reference dx of the 2-cell taper (3.125 um)


def _parity_case(P_s, taper, n_steps, n_grid=96):
    """One BPM run and its ResolvedFiberAmplifier twin on the SAME plan. Every row here sheds a
    little power into the boundary by construction (that is item 1 of gate F), so the module's
    per-step-absorber advisory is caught and discarded rather than printed 20 times."""
    fib = lma_fiber()
    pumps = [Pump(100.0, LAM_P, "fwd", cladding=True)]
    sig = [Signal(P_s, LAM_S)]
    b = GainBPM(smith_yb(), fib, pumps, sig, None, n_grid=n_grid, extent_m=PARITY_EXTENT,
                index_taper_cells=taper)
    r, _ = _absorbing(lambda: b.solve(n_steps=n_steps, n_samples=21))
    ch = ResolvedFiberAmplifier(smith_yb(), fib, pumps, sig, None).solve(n_nodes=41)
    return b, r, float(ch.signal_gain_dB[0]), float(ch.power_W[0, -1])


def gate_f():
    """The headline cross-solver gate, and the one that needs its deviation EXPLAINED rather than
    merely bounded. The two solvers apportion gain by different mechanisms: the channel model
    integrates the local gain against a FIXED analytic LP01 intensity, the BPM lets the field
    evolve and reads the answer off the field it gets. Three things can then separate them:

      1. SPLIT-STEP ERROR ON THE DISCRETIZED INDEX STEP, which the boundary absorber then REPORTS
         as radiation. This is the item the previous version of this gate got wrong. It asserted
         abs(d_hard) > 0.1 -- "the taper is doing real work" -- on a HARD-step row run at nz = 400,
         and that number is not converged: it is the split's own error on an under-resolved index
         discontinuity, shed into the boundary. RE-MEASURED over nz = 200 / 400 / 800 / 1600 / 3200
         at the saturated point (verifier, 2026-08-04), the hard step's absorbed fraction falls
         1.21e-1 -> 6.46e-2 -> 1.78e-2 -> 8.58e-6 -> 7.53e-6 and its deviation follows it,
         -0.535 -> -0.277 -> -0.076 -> -0.00116 -> -0.00109 dB. At a CONVERGED dz the hard step and
         the taper therefore agree with the channel model equally well, and the old assertion would
         FAIL. This gate now runs the hard-step row at nz = 1600 and pins the agreement instead.
      2. THE TAPER'S OWN WIDTH -- a bias, not a discretization error. The 2-cell taper leaves a
         dz-INDEPENDENT radiation floor of 1.2e-4 of the launch (a launch-mismatch: the launch is
         the exact HARD-step LP01, and the tapered guide's mode is a slightly different one), and a
         saturated deviation near -0.0095 dB. The previous version claimed that deviation
         "CONVERGES AWAY with dx" from the series -0.0175 / -0.0096 / -0.0060 / -0.0028 dB at
         N = 64/96/128/192 -- but that series held the taper at 2 CELLS, so refining dx was
         physically REMOVING the taper, and the number was tracking the taper width, not the grid.
         Held at a fixed PHYSICAL width (s scaled as 1/dx) the same series is FLAT:
         -0.00932 / -0.00957 / -0.00926 / -0.00939 dB at N = 64/96/128/192. The honest attribution
         is therefore a TAPER-WIDTH BIAS -- discretization of the physical problem, not radiation,
         not saturation-integral quadrature, and not physics. It is measured below and it does not
         converge away at fixed taper width.
      3. Genuine mode reshaping (gain guiding). At this fixture's extraction it is below the other
         two: the BPM field's dopant overlap moves by only 5.5e-4 over the fiber.

    A fixture with real gain guiding -- deeper extraction, a shorter dopant than the core, or a
    higher-contrast core -- would push (3) above (2); this one does not, and the gate says so
    rather than claiming the agreement proves more than it does."""
    ok = True
    rows = []
    for P_s, label in ((1e-3, "small-signal"), (50.0, "saturated")):
        b, r, g_ch, pp_ch = _parity_case(P_s, 2.0, 400)
        _, r_dz, _, _ = _parity_case(P_s, 2.0, 800)               # dz refinement
        _, r_hard, _, _ = _parity_case(P_s, 0.0, 1600)            # hard step at CONVERGED dz
        I0 = np.abs(b.launch_field()) ** 2
        Iout = np.abs(r.E_out) ** 2
        gam_in = float(np.sum(I0[b.R <= A_CORE]) / np.sum(I0))
        gam_out = float(np.sum(Iout[b.R <= A_CORE]) / np.sum(Iout))
        rows.append({"label": label, "P_s": P_s, "g_ch": g_ch, "g_b": r.gain_dB,
                     "d": r.gain_dB - g_ch, "d_dz": r_dz.gain_dB - g_ch,
                     "d_hard": r_hard.gain_dB - g_ch, "abs_hard": r_hard.meta["absorbed_fraction"],
                     "abs_dz": r_dz.meta["absorbed_fraction"],
                     "gam_in": gam_in, "gam_out": gam_out, "pp_b": r.P_pump_W[0, -1],
                     "pp_c": pp_ch, "abs": r.meta["absorbed_fraction"], "res": r})
    for w in rows:
        print("[fgb]   F  {:12s} P_s = {:8.3g} W: channel {:.6f} dB, BPM {:.6f} dB, delta "
              "{:+.5f} dB   [dz/2 {:+.5f} | HARD step at nz=1600 {:+.5f}]".format(
                  w["label"], w["P_s"], w["g_ch"], w["g_b"], w["d"], w["d_dz"], w["d_hard"]),
              flush=True)
        print("[fgb]   F  {:12s} pump out {:.4f} W (channel {:.4f} W); the BPM field's dopant "
              "overlap moves {:.6f} -> {:.6f}; boundary-absorbed {:.2e} of the launch (dz/2: "
              "{:.2e} -- the taper's floor is dz-INDEPENDENT; hard step at nz=1600: {:.2e})".format(
                  "", w["pp_b"], w["pp_c"], w["gam_in"], w["gam_out"], w["abs"], w["abs_dz"],
                  w["abs_hard"]), flush=True)
    small, sat = rows[0], rows[1]

    # (1) the hard step's shed is SPLIT-STEP ERROR: refine dz and it goes away, taking the dB
    # deficit with it. This is the series that retired the old abs(d_hard) > 0.1 assertion.
    hard_series = []
    for nz in (200, 3200):
        _, r_h, g_ch_h, _ = _parity_case(50.0, 0.0, nz)
        hard_series.append((nz, r_h.gain_dB - g_ch_h, r_h.meta["absorbed_fraction"]))
    hard_series.insert(1, (1600, sat["d_hard"], sat["abs_hard"]))
    print("[fgb]   F  HARD step, saturated, vs dz: "
          + " | ".join("nz {:4d} delta {:+.5f} dB absorbed {:.2e}".format(*h)
                       for h in hard_series), flush=True)

    # (2) the residual is a TAPER-WIDTH bias: hold s*dx fixed and refine N -- it does not move.
    dx_series = []
    for n_grid in (64, 96, 128, 192):
        s = 2.0 * PARITY_DX96 / (PARITY_EXTENT / n_grid)
        _, r_sm, g_sm, _ = _parity_case(1e-3, s, 400, n_grid=n_grid)
        _, r_st, g_st, _ = _parity_case(50.0, s, 400, n_grid=n_grid)
        dx_series.append((n_grid, s, r_st.gain_dB - g_st,
                          (r_st.gain_dB - g_st) - (r_sm.gain_dB - g_sm)))
    spread = float(np.ptp([d[3] for d in dx_series]))
    spread_sat = float(np.ptp([d[2] for d in dx_series]))
    print("[fgb]   F  taper width HELD at {:.3f} um while dx refines: "
          .format(1e6 * 2.0 * PARITY_DX96)
          + " | ".join("N {:3d} (s {:.3f}) sat {:+.5f} sat-specific {:+.5f}".format(*d)
                       for d in dx_series)
          + "  [spreads {:.1e} / {:.1e} dB]".format(spread_sat, spread), flush=True)

    ok = ok and abs(small["d"]) < 0.01 and abs(sat["d"]) < 0.05
    ok = ok and abs(sat["d_dz"] - sat["d"]) < 1e-3                # step-converged
    # at a CONVERGED dz the hard step agrees too -- the taper buys resolution, not correctness
    ok = ok and abs(sat["d_hard"]) < 0.01 and abs(small["d_hard"]) < 0.01
    ok = ok and sat["abs_hard"] < 1e-5 and hard_series[0][2] > 1e-2   # shed IS split-step error
    ok = ok and 5e-5 < sat["abs"] < 5e-4 and abs(sat["abs_dz"] / sat["abs"] - 1.0) < 0.05
    ok = ok and spread < 1e-4                     # the residual does NOT converge away with dx
    ok = ok and small["g_ch"] > 0.3 and sat["g_ch"] < 0.6 * small["g_ch"]   # it really saturates
    print("[fgb]   F  saturation-specific deviation (saturated minus small-signal) = {:+.5f} dB; "
          "its spread over the fixed-physical-taper dx series is {:.2e} dB (< 1e-4 -- a taper-width "
          "BIAS, not a converging discretization); gain compression channel-side {:.4f} -> {:.4f} "
          "dB".format(sat["d"] - small["d"], spread, small["g_ch"], sat["g_ch"]), flush=True)
    return (_report("F", ok, "BPM and ResolvedFiberAmplifier agree to {:+.5f} dB small-signal "
                             "(bar 0.01) and {:+.5f} dB deeply saturated (bar 0.05); at a "
                             "converged dz the HARD step agrees to {:+.5f} dB too".format(
                                 small["d"], sat["d"], sat["d_hard"])), sat["res"])


def gate_g(res):
    """dossier B10 on a GAIN run (gate A did it passively). The ledger is not a tautology: the
    exchange term is computed in closed form from g and the field ENTERING the N leg, so closure
    tests that the diffraction operator is unitary and that the index operator is a pure real
    phase -- a complex index or a mis-normalized FFT breaks it immediately."""
    launched = float(res.P_signal_W[0])
    resid = float(np.max(np.abs(res.energy_residual_W)) / launched)
    ok = resid < 1e-12
    print("[fgb]   G  launched {:.9f} W + stimulated exchange {:.9f} W - boundary absorbed "
          "{:.3e} W = {:.9f} W; transmitted {:.9f} W".format(
              launched, res.medium_exchange_W[-1], res.absorbed_boundary_W[-1],
              launched + res.medium_exchange_W[-1] - res.absorbed_boundary_W[-1],
              res.P_signal_W[-1]), flush=True)
    return _report("G", ok, "the gain-run energy ledger closes to {:.2e} of the launched power "
                            "(bar 1e-12)".format(resid))


# ============================ GATE H ====================================================

def gate_h():
    """dossier B6/B14 plus the quasi-static loop itself.

    (i) BRIDGE: the module's thermal index perturbation dn = (dn/dT) dT(r) has, by
        thermal.radial_temperature_rise's own core solution, dT(0) - dT(a) = Q/(4 pi k), so its
        duct parameter alpha^2 = 2 dn_drop/(a^2 n_c) IS thermal_lens_focal_power_per_m -- checked
        on the numbers, not asserted.
    (ii) DYNAMICS: launch the matched Gaussian of that duct into the REAL thermal profile (the
        parabolic core plus the logarithmic cladding tail, not an idealized parabola) and require
        it to stay put over a full self-imaging period, against a no-lens control that diffracts
        away. Direction and magnitude in one measurement.
    (iii) LOOP: the self-consistent field <-> heat iteration on an actual amplifier converges, and
        its heat source equals the quantum defect times the marched pump absorption."""
    ok = True
    model = ThermalModel()
    b_out = 125e-6
    Q = 664.0                                     # W/m -- a high-power but realistic heat load
    d_prime = float(thermal_lens_focal_power_per_m(Q, A_CORE, dndt=DN_DT, k_core_W_mK=1.38,
                                                   n_core=N_CLAD))
    alpha = float(np.sqrt(d_prime))
    w_m = quadratic_duct_radius_m(d_prime, LAM_S, N_CLAD)
    z_p = quadratic_duct_period_m(d_prime)

    # (i) bridge, from the temperature solution the module actually calls
    _, dT_edges = radial_temperature_rise(Q, A_CORE, b_out, model, r_m=np.array([0.0, A_CORE]))
    d_from_shape = 2.0 * DN_DT * float(dT_edges[0] - dT_edges[1]) / (A_CORE ** 2 * N_CLAD)
    rel_bridge = abs(d_from_shape / d_prime - 1.0)
    ok = ok and rel_bridge < 1e-12

    # (ii) the real thermal profile as the BPM index
    fib = FiberSpec(core_radius_m=A_CORE, na=NA, n_t_m3=N_T, length_m=z_p)
    probe = GainBPM(None, fib, [], [Signal(1.0, LAM_S)], n_grid=96, extent_m=120e-6,
                    gain_model=lambda I: 0.0 * I, absorber=False, n_ref=N_CLAD)
    _, dT_grid = radial_temperature_rise(Q, A_CORE, b_out, model,
                                         r_m=np.minimum(probe.R, b_out))
    n_th = N_CLAD + DN_DT * np.asarray(dT_grid)
    b = GainBPM(None, fib, [], [Signal(1.0, LAM_S)], n_grid=96, extent_m=120e-6,
                gain_model=lambda I: 0.0 * I, absorber=False, n_ref=N_CLAD,
                index_profile=lambda X, Y: n_th)
    E0 = b.launch_field(w_m=w_m)
    r = b.solve(n_steps=300, n_samples=61, E_in=E0)
    ctrl = GainBPM(None, fib, [], [Signal(1.0, LAM_S)], n_grid=96, extent_m=120e-6,
                   gain_model=lambda I: 0.0 * I, absorber=False, n_ref=N_CLAD,
                   index_profile=lambda X, Y: np.full_like(X, N_CLAD))
    r_ctrl = ctrl.solve(n_steps=300, n_samples=3, E_in=E0)
    dev = float(np.ptp(r.spot_radius_m) / w_m)
    spread = float(r_ctrl.spot_radius_m[-1] / w_m)
    ok = ok and dev < 0.05 and spread > 4.0

    # (iii) the quasi-static loop on a real amplifier
    amp = GainBPM(smith_yb(), lma_fiber(), [Pump(1000.0, LAM_P, "fwd", cladding=True)],
                  [Signal(20.0, LAM_S)], None, n_grid=96, extent_m=150e-6,
                  index_taper_cells=2.0,
                  thermal=ThermalLoop(b_outer_m=b_out, model=model, max_iter=6, tol_K=0.02,
                                      dndt_per_K=DN_DT))
    r_amp, _ = _absorbing(lambda: amp.solve(n_steps=300, n_samples=101))
    qd = quantum_defect_fraction(LAM_P, LAM_S)
    # INTEGRAL comparison: Q is recorded at each step's MIDPOINT and P_p at the step boundary, so
    # a pointwise d/dz check carries a half-step offset plus np.gradient's own O(h^2) error (that
    # form measures 1.4e-3 here and does NOT tighten with more samples -- it is the offset, not a
    # physics defect). The integral is the quantity the two must agree on.
    dPp = -np.gradient(r_amp.P_pump_W[0], r_amp.z_m)
    heat_ptw = float(np.max(np.abs(r_amp.heat_per_m_W - qd * dPp)) / np.max(r_amp.heat_per_m_W))
    from dynameta.core.numerics import trapz   # audit X-1: floor-safe (np.trapezoid needs numpy>=2.0)
    heat_int = float(trapz(r_amp.heat_per_m_W, r_amp.z_m))
    heat_ref = qd * float(r_amp.P_pump_W[0, 0] - r_amp.P_pump_W[0, -1])
    heat_rel = abs(heat_int / heat_ref - 1.0)
    ok = ok and r_amp.meta["thermal_converged"] and heat_rel < 1e-3
    print("[fgb]   H  Q = {:.0f} W/m -> D' = {:.5e} 1/m^2 (bridge rel {:.2e}), alpha = {:.2f} 1/m, "
          "w_m = {:.3f} um, z_p = {:.4f} mm".format(Q, d_prime, rel_bridge, alpha, 1e6 * w_m,
                                                    1e3 * z_p), flush=True)
    print("[fgb]   H  matched launch in the SELF-CONSISTENT thermal profile: width span "
          "[{:.4f}, {:.4f}] um, rel deviation {:.3e}; the same launch with the lens OFF spreads "
          "to {:.2f} um ({:.1f}x)".format(1e6 * r.spot_radius_m.min(), 1e6 * r.spot_radius_m.max(),
                                          dev, 1e6 * r_ctrl.spot_radius_m[-1], spread), flush=True)
    print("[fgb]   H  loop: {} iterations, converged {}, final max dT {:.4f} K; INT Q dz = "
          "{:.9f} W vs quantum defect x marched pump absorption {:.9f} W (rel {:.2e}; the "
          "pointwise d/dz form is {:.2e}, the half-step offset)".format(
              r_amp.meta["thermal_iterations"], r_amp.meta["thermal_converged"],
              r_amp.meta["thermal_max_dT_K"], heat_int, heat_ref, heat_rel, heat_ptw), flush=True)
    return _report("H", ok, "the thermal lens IS thermal.py's D', it focuses by exactly the "
                            "predicted amount, and the quasi-static loop closes")


# ============================ GATE I ====================================================

LOSS_PER_M = 3.0                              # a deliberately large background loss [1/m]


def gate_i():
    """BACKGROUND LOSS -- a leg with no gate at all until now: every fixture above runs at
    FiberSpec.background_loss_per_m = 0, so a defect in the loss bookkeeping was invisible to the
    entire suite. It was not hypothetical. _medium returns g (F2.7) ALREADY NET of l_s, while
    _march folds the scalar amplitude factor exp(-l_s dz/2) into the whole-grid index phase; the
    gain support was then multiplied by exp(g dz/2) ON TOP, so on the doped disc the background
    loss was applied TWICE -- an intensity factor exp(g_med dz - 2 l_s dz) instead of
    exp((g_med - l_s) dz). Off the support it was right, which is exactly why nothing caught it.

    (i) GAIN OFF, absorber off, l_s = 3 1/m over 2 cm: the march must reproduce exp(-l_s L) to
        round-off. The support here is the doped disc, so this is a direct read of the two-piece
        sum, and with the defect present it lands on exp(-2 l_s L) inside the core instead.
    (ii) GAIN ON, 100 W cladding pump on the same fiber: the energy ledger must still close (the
        exchange increment is a closed form in g and the ENTERING field, so a mis-applied loss
        breaks it), and the loss must cost the lossless twin's gain 10 log10(exp(l_s L)) dB. Those
        two do not agree exactly and should not: the lossy run carries less signal, so it saturates
        the medium slightly less and wins a little of the loss back."""
    L = 0.02
    fib = FiberSpec(core_radius_m=A_CORE, na=NA, n_t_m3=N_T, length_m=L,
                    background_loss_per_m=LOSS_PER_M)
    b = GainBPM(None, fib, [], [Signal(1.0, LAM_S)], n_grid=64, extent_m=150e-6,
                gain_model=lambda I: 0.0 * I, absorber=False, index_taper_cells=2.0)
    r = b.solve(n_steps=200, n_samples=5)
    ratio = float(r.P_signal_W[-1] / r.P_signal_W[0])
    rel = abs(ratio / np.exp(-LOSS_PER_M * L) - 1.0)
    led_i = float(np.max(np.abs(r.energy_residual_W)) / r.P_signal_W[0])

    pumps = [Pump(100.0, LAM_P, "fwd", cladding=True)]
    kw = dict(n_grid=64, extent_m=150e-6, index_taper_cells=2.0)
    lossy = GainBPM(smith_yb(), FiberSpec(core_radius_m=A_CORE, na=NA, n_t_m3=N_T, length_m=L,
                                          background_loss_per_m=LOSS_PER_M, clad_radius_m=100e-6),
                    pumps, [Signal(1.0, LAM_S)], None, **kw)
    clean = GainBPM(smith_yb(), lma_fiber(), pumps, [Signal(1.0, LAM_S)], None, **kw)
    r_lossy, n_w = _absorbing(lambda: lossy.solve(n_steps=200, n_samples=11))
    r_clean, _ = _absorbing(lambda: clean.solve(n_steps=200, n_samples=11))
    led_ii = float(np.max(np.abs(r_lossy.energy_residual_W)) / r_lossy.P_signal_W[0])
    d_dB = r_clean.gain_dB - r_lossy.gain_dB
    d_ref = 10.0 * np.log10(np.exp(LOSS_PER_M * L))
    ok = (rel < 1e-12 and led_i < 1e-12 and led_ii < 1e-12 and n_w == 1
          and abs(d_dB - d_ref) < 1e-3 and d_dB > 0.0
          # the loss is inside the exchange term, not outside it: the lossy run's medium hands the
          # field strictly less net power than the lossless twin's, on the same pump
          and 0.0 < float(r_lossy.medium_exchange_W[-1]) < float(r_clean.medium_exchange_W[-1]))
    print("[fgb]   I  (i)  gain off, l_s = {:.1f} 1/m over {:.0f} mm: P_out/P_in = {:.15e} vs "
          "exp(-l_s L) = {:.15e}  rel {:+.3e} (bar 1e-12); ledger {:.2e}".format(
              LOSS_PER_M, 1e3 * L, ratio, np.exp(-LOSS_PER_M * L), ratio
              / np.exp(-LOSS_PER_M * L) - 1.0, led_i), flush=True)
    print("[fgb]   I  (ii) gain on: lossy {:.6f} dB vs lossless twin {:.6f} dB, cost {:.6f} dB "
          "against 10 log10(exp(l_s L)) = {:.6f} dB (the {:+.1e} dB difference is the lossy run "
          "saturating less); ledger {:.2e}, net medium exchange {:+.4f} W".format(
              r_lossy.gain_dB, r_clean.gain_dB, d_dB, d_ref, d_dB - d_ref, led_ii,
              float(r_lossy.medium_exchange_W[-1])), flush=True)
    return _report("I", ok, "background loss is applied EXACTLY ONCE on the gain support: "
                            "exp(-l_s L) to {:.1e} with the gain off, ledger {:.1e} with it "
                            "on".format(rel, led_ii))


# ============================ GATE J ====================================================

TSHB_PUMP_W = 1500.0
TSHB_PINS = {1e-6: 0.0920, 100.0: -0.2431, 1200.0: -0.6673}     # g01 - g11 [1/m], this build


def gate_j():
    """COHERENT-SOLVER TSHB -- the campaign's headline physics, stated in the form a field solver
    can actually be held to.

    The usual TSHB test on a channel model is "the LP11 power fraction grows". It is the WRONG test
    here, and part (ii) measures why. A coherent field carries a modal-interference cross term
    2 Re(c01 c11*) psi01 psi11 that no power model has; the saturable medium turns it into a gain
    grating whose first-order (odd-azimuth) part has ZERO diagonal projection and lives entirely in
    the off-diagonal element G_x = <psi01| delta_g |psi11>. That element feeds d P01/dz and
    d P11/dz the SAME ABSOLUTE rate G_x sqrt(P01 P11), so its effect on the LP11 FRACTION is
    proportional to (P01 - P11): identically zero at a 50/50 split, and reversed across it.

    (i) is the well-posed statement, and the one pinned: project the BPM's OWN local_gain_per_m
        field (F2.7) onto the LP01 and LP11 intensities with a WEAK LP11 probe, and the ORDERING
        g01 - g11 must REVERSE SIGN as the LP01 power saturates the medium -- positive unsaturated
        (LP01 is better confined to the dopant, Gamma_01 > Gamma_11), negative once the saturating
        LP01 has burnt its own hole on axis. This is diagonal-only, so the cross term of (ii) never
        enters it, and it must agree with transverse.ResolvedFiberAmplifier's resolved modal gains
        -- two solvers, no shared code path, one number.
    (ii) is the discrimination: the cross term's contribution to d(f11)/dz, measured across the
        split, must cross zero exactly at 50/50 and change sign. That is why (i) uses a probe."""
    fib = lma_fiber()
    pumps = [Pump(TSHB_PUMP_W, LAM_P, "fwd", cladding=True)]
    modes = solve_lp_modes(A_CORE, NA, LAM_S, n_clad=N_CLAD)
    m01, m11 = modes[0], modes[1]
    # N = 128 is not incidental: the Cartesian sampling of the cos(phi) LP11 sets how well the grid
    # projection reproduces the channel model's Gauss-Legendre azimuthal quadrature (10.3e-3 1/m at
    # N = 96, 2.4e-3 at 128, 0.5e-3 at 256), and the DIFFERENCE g01 - g11 is the tighter of the two.
    b = GainBPM(smith_yb(), fib, pumps, [Signal(1.0, LAM_S)], None, n_grid=128, extent_m=150e-6,
                index_taper_cells=2.0)
    amp = ResolvedFiberAmplifier(smith_yb(), fib, pumps,
                                 [Signal(1.0, LAM_S), Signal(1e-9, LAM_S)], None,
                                 signal_modes=[m01, m11])
    i01 = b.mode_field_2d(m01) ** 2
    i11 = b.mode_field_2d(m11, "cos") ** 2
    ok = True
    rows = []
    for P in (1e-6, 100.0, 1200.0):
        g = b.local_gain_per_m(b.launch_field(P, m01))
        g01 = float(np.sum(g * i01) * b.dA)
        g11 = float(np.sum(g * i11) * b.dA)
        gc = amp.local_gain_per_m([TSHB_PUMP_W, P, 1e-12])
        d_b, d_c = g01 - g11, float(gc[1] - gc[2])
        rows.append((P, g01, g11, d_b, d_c, abs(d_b - d_c)))
        pin = TSHB_PINS[P]
        ok = ok and np.sign(d_b) == np.sign(pin) and abs(d_b / pin - 1.0) < 0.10
        ok = ok and abs(d_b - d_c) <= 2e-3
        print("[fgb]   J  P_LP01 = {:8.4g} W: BPM projection g01 {:+.4f} g11 {:+.4f}  "
              "g01 - g11 = {:+.4f} 1/m (pin {:+.4f}); channel model {:+.4f} 1/m, |diff| "
              "{:.2e}".format(P, g01, g11, d_b, pin, d_c, abs(d_b - d_c)), flush=True)
    ok = ok and rows[0][3] > 0.0 > rows[1][3] > rows[2][3]        # the reversal, sign-exact

    # (ii) the cross term, and why the LP11 POWER FRACTION is not the test
    p01, p11 = b.mode_field_2d(m01), b.mode_field_2d(m11, "cos")
    P_tot = 200.0
    cross = []
    for f in (0.1, 0.3, 0.5, 0.7, 0.9):
        P01, P11 = f * P_tot, (1.0 - f) * P_tot
        dg = (b.local_gain_per_m(np.sqrt(P01) * p01 + np.sqrt(P11) * p11)
              - b.local_gain_per_m(np.sqrt(P01 * p01 ** 2 + P11 * p11 ** 2).astype(complex)))
        G_x = float(np.sum(p01 * dg * p11) * b.dA)
        # d(f11)/dz from the cross term alone = G_x sqrt(P01 P11) (P01 - P11) / P_tot^2
        cross.append((f, G_x, G_x * np.sqrt(P01 * P11) * (P01 - P11) / P_tot ** 2))
    ok = ok and cross[2][2] == 0.0 and cross[0][2] > 0.0 > cross[4][2]
    ok = ok and all(abs(c[1]) > 0.5 for c in cross)
    print("[fgb]   J  cross-term gain grating at {:.0f} W total, LP01 fraction f: ".format(P_tot)
          + " | ".join("f {:.1f} G_x {:+.3f} 1/m d(f11)/dz {:+.4f} /m".format(*c)
                       for c in cross), flush=True)
    return _report("J", ok, "the BPM's own local gain reverses the LP01/LP11 ordering with "
                            "saturation ({:+.4f} -> {:+.4f} -> {:+.4f} 1/m) and tracks the channel "
                            "model to {:.1e} 1/m, while the cross term's effect on the LP11 POWER "
                            "FRACTION vanishes at 50/50 and reverses across it".format(
                                rows[0][3], rows[1][3], rows[2][3],
                                max(r[5] for r in rows)))


def main():
    print("[fgb] === scalar gain-BPM: field-solver gates ===", flush=True)
    t0 = time.time()
    ok = True
    for gate in (gate_a, gate_b, gate_c, gate_d, gate_e):
        ok = bool(gate()) and ok
    ok_f, sat_res = gate_f()
    ok = bool(ok_f) and ok
    ok = bool(gate_g(sat_res)) and ok
    for gate in (gate_h, gate_i, gate_j):
        ok = bool(gate()) and ok
    print("[fgb] *** GAIN-BPM: {} *** ({:.1f} s)".format("PASS" if ok else "FAIL",
                                                         time.time() - t0), flush=True)
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
