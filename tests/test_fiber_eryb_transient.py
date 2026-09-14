"""Gates for the Er:Yb co-doped amplifier's ENERGY CLOSURE and TRANSIENT MARCH
(dynameta.optics.fiber_amp.eryb + efficiency._dissipated_power_W + dynamics.simulate_transient_
eryb), the two things a downstream study needed before ErYbAmplifier could stand in for
FiberAmplifier everywhere.

WHAT IS BEING DISCRIMINATED, and by what independent oracle:

  * The CLOSURE (efficiency.energy_balance_residual_W). Oracle: the endpoint flux difference.
    `launched - exiting` is pure bookkeeping on the returned array; the dissipation is recomputed
    from the amplifier's own right-hand side on that same array, so the residual is zero only if
    the returned P(z) actually solves the ODEs. It is NOT an identity (thermal.total_heat_W would
    make it one -- see the efficiency module docstring), and it falls as O(dz^2), which is what
    gate (d) pins.

  * The TERM SPLIT (ErYbAmplifier.energy_terms). Oracle: itself, twice. q_opt is assembled from
    the PROPAGATION coefficients; dU/dt + D_loss + D_Er + D_Yb + D_tr is assembled from the
    POPULATION rate equations and the two ions' zero-line energies. They are the same balance only
    if the Yb -> Er transfer defect is charged exactly once and with the right sign, which is not
    a tautology: test_transfer_defect_is_positive_sized_and_charged_exactly_once charges it twice
    by hand and shows the identity then breaks by the defect's own size.

  * The MARCH (simulate_transient on an ErYbAmplifier). Three independent oracles: the SINGLE-ION
    march in the k_tr -> 0 limit (completely separate code -- a scalar exponential integrator on a
    one-reservoir balance), the steady RELAXATION solve as the long-time limit, and the energy
    balance step by step.

Every gate here is falsifiable and the sizes are pinned, not just the signs. The pathology gates
(the coarse-grid runaway, the broken-defect identity) are PREMISE-GATED: the healthy case is
asserted first in the same test, so a change that makes everything trip cannot pass them.
"""

import warnings

import numpy as np
import pytest

from dynameta.core.numerics import trapz          # audit X-1: floor-safe (not np.trapezoid)
from dynameta.optics.fiber_amp import (AseBand, FiberAmplifier, FiberSpec, Pump, PumpSource,
                                       Signal, amplifier_saturation_energy, erbium,
                                       saturation_energy, simulate_transient,
                                       simulate_transient_eryb, wall_plug_efficiency, ytterbium)
from dynameta.optics.fiber_amp.eryb import ErYbAmplifier

ER = erbium("aluminosilicate")
YB = ytterbium("phosphosilicate")            # phospho Yb: tau_Yb = 1.45 ms (the k_tr host)
_CLOSURE_SOURCE = PumpSource(wallplug_efficiency=1.0)     # the balance is optical; no electrics


def _core_amp(length_m=2.0):
    """The CORE-PUMPED reference point of the closure work: a = 2.8 um, NA 0.20, N_Er 2e25,
    N_Yb 2e26, k_tr 2e-22, a 200 mW co-propagating 976 nm pump, a 0.5 mW 1550 nm signal, the
    C band in 24 bins and a 1000-1100 nm Yb parasitic band. Deliberately the HARD mesh case:
    gamma N_Yb sigma_a(976) ~ 200 /m, so the pump is absorbed in ~5 mm and the z mesh has to
    resolve that before the power balance can close."""
    return ErYbAmplifier(ER, YB, FiberSpec(2.8e-6, 0.20, 2.0e25, length_m),
                         [Pump(0.200, 0.976e-6, "fwd")], [Signal(0.5e-3, 1.550e-6)],
                         AseBand(1.520e-6, 1.570e-6, 24), n_yb_m3=2.0e26, k_tr_m3_s=2.0e-22,
                         yb_ase=AseBand(1.000e-6, 1.100e-6, 8))


def _clad_amp(p_sig_W=1.0e-3, bins=8, length_m=4.0):
    """The CLADDING-PUMPED point the transient gates use -- the configuration an EYDFA actually
    ships in. The pump overlap is A_dope/A_clad ~ 2e-3, so the 976 nm absorption is ~1 /m over a
    4 m fiber and 161 nodes resolve BOTH the pump and the 1550 nm signal. (The core-pumped point
    above needs ~1600 nodes for the same statement; that is a mesh property of the frozen-
    population propagator, not of the integrator -- see the audit note.)"""
    return ErYbAmplifier(ER, YB,
                         FiberSpec(2.8e-6, 0.20, 2.0e25, length_m, clad_radius_m=62.5e-6),
                         [Pump(1.0, 0.976e-6, "fwd", cladding=True)],
                         [Signal(p_sig_W, 1.550e-6)], AseBand(1.520e-6, 1.570e-6, bins),
                         n_yb_m3=2.0e26, k_tr_m3_s=2.0e-22)


def _closure_rel(amp, res):
    """|energy_balance_residual_W| / launched pump -- exactly what the downstream study's
    solve_closed() refines the mesh against."""
    b = wall_plug_efficiency(amp, res, _CLOSURE_SOURCE)
    return abs(float(b.energy_balance_residual_W)) / max(float(b.p_pump_launched_W), 1e-30)


def _exiting_W(P, direction):
    """Total power leaving the fiber: every channel read at ITS OWN output end."""
    return float(np.sum([P[k, -1] if direction[k] > 0 else P[k, 0] for k in range(P.shape[0])]))


# ==================== gate (d): the steady closure exists and is mesh-convergent ============

def test_closure_is_finite_and_falls_to_1e_5_with_mesh():
    # BEFORE this work efficiency._dissipated_power_W returned NaN for anything without
    # _dP_full_c, so energy_balance_residual_W was NaN for EVERY co-doped result and the study's
    # solve_closed() had nothing to refine against. Two claims: the number is finite, and it is
    # the O(dz^2) discretization of the rate integral -- i.e. it FALLS, monotonically, with the
    # mesh, and reaches the study's 1e-5 requirement.
    amp = _core_amp()
    rel = []
    for n in (81, 161, 321, 641):
        with warnings.catch_warnings():
            # the coarsest mesh legitimately trips efficiency's own 1e-3 balance note
            warnings.simplefilter("ignore", RuntimeWarning)
            r = amp.solve(n_nodes=n)
            rel.append(_closure_rel(amp, r))
        assert r.meta["converged"]
    assert np.all(np.isfinite(rel)), rel
    assert np.all(np.diff(rel) < 0.0), rel                    # monotone fall, no plateau
    # MEASURED 2026-09-13: 2.12e-3 / 1.16e-4 / 9.97e-6 / 2.14e-6 -- a factor 4+ per doubling.
    assert rel[0] > 1e-4, rel                                 # the coarse mesh really is bad
    assert rel[-1] < 1e-5, rel                                # and the fine one really closes
    assert rel[-2] / rel[-1] > 3.0, rel                       # second-order tail, not a floor


def test_closure_note_fires_only_when_the_balance_is_open():
    # premise: the fine mesh closes and says nothing. Only then is the coarse mesh's complaint
    # evidence of anything.
    amp = _core_amp()
    fine = wall_plug_efficiency(amp, amp.solve(n_nodes=641), _CLOSURE_SOURCE)
    assert [n for n in fine.notes if "power balance" in n] == []
    assert np.isfinite(fine.energy_balance_residual_W)
    coarse_res = amp.solve(n_nodes=81)
    coarse = wall_plug_efficiency(amp, coarse_res, _CLOSURE_SOURCE)
    assert any("power balance closes to only" in n for n in coarse.notes), coarse.notes


def test_closure_detects_a_corrupted_profile():
    # The whole point of closing against the RATE BALANCE rather than the endpoint flux: the
    # latter is X - X and stays at machine zero on a corrupted solve. Perturb the signal output
    # endpoint by 1% and the residual must pick up EXACTLY that mis-stated watt -- the dissipation
    # side barely moves, because one node of one channel is half a trapezoid cell of it.
    amp = _core_amp()
    r = amp.solve(n_nodes=641)
    good = _closure_rel(amp, r)
    sig = [i for i, k in enumerate(r.kind) if k == "signal"][0]
    injected = 0.01 * float(r.power_W[sig, -1]) / float(amp.pumps[0].power_W)
    r.power_W[sig, -1] *= 1.01
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        bad = _closure_rel(amp, r)
    # MEASURED 2026-09-13: 2.14e-6 -> 1.75e-4, a factor 82, and 1.75e-4 IS the injected error
    assert bad > 50.0 * good, (good, bad)
    assert abs(bad - injected) < 0.2 * injected, (bad, injected)


# ==================== the term split: q_opt == dU/dt + dissipation ==========================

def test_energy_terms_identity_holds_to_round_off():
    amp = _core_amp()
    r = amp.solve(n_nodes=321)
    t = amp.energy_terms(r.power_W, r.nbar2_z, r.meta["beta_yb_z"])
    scale = float(np.max(np.abs(t["q_optical"])))
    err = float(np.max(np.abs(t["q_optical"] - t["d_stored_dt"] - t["dissipation"]))) / scale
    assert err < 1e-10, err                                   # MEASURED 4.1e-14
    # at the STEADY state the stored energy is stationary, so q_opt is pure dissipation
    assert float(np.max(np.abs(t["d_stored_dt"]))) / scale < 1e-10
    # and the aggregate the efficiency layer uses is the same number
    agg = float(trapz(t["dissipation"], r.z_m))
    assert abs(agg - amp._rate_balance_dissipation_W(r)) < 1e-9 * abs(agg)


def test_transfer_defect_is_positive_sized_and_charged_exactly_once():
    # PREMISE (proved first): the split closes. Only then does breaking it prove anything.
    amp = _core_amp()
    r = amp.solve(n_nodes=321)
    t = amp.energy_terms(r.power_W, r.nbar2_z, r.meta["beta_yb_z"])
    total = float(trapz(t["dissipation"], r.z_m))
    d_tr = float(trapz(t["transfer_defect"], r.z_m))
    assert np.all(t["transfer_defect"] >= 0.0)                # Yb quantum > Er quantum, always
    # the defect is (eps_Yb - eps_Er) per transfer: 975 nm -> 1530 nm is 0.46 eV, 36% of the
    # pump photon. Its SHARE of the dissipation is set by how much of the pump reaches Er.
    assert 0.01 < d_tr / total < 0.9, (d_tr, total)
    per_event = d_tr / float(trapz(t["transfer_rate_per_m"], r.z_m))
    assert abs(per_event - (YB.eps_J - ER.eps_J)) < 1e-12 * YB.eps_J
    assert 0.40 < per_event / 1.602176634e-19 < 0.52          # ~0.46 eV
    # DISCRIMINATION: charge the defect twice and the identity breaks by its own size.
    broken = t["dissipation"] + t["transfer_defect"]
    scale = float(np.max(np.abs(t["q_optical"])))
    err = float(np.max(np.abs(t["q_optical"] - t["d_stored_dt"] - broken))) / scale
    assert err > 1e-3, err


# ==================== the phi_1 kernel against an INDEPENDENT oracle =========================

def test_phi1_kernel_matches_scipy_expm_including_coalesced_eigenvalues():
    # dt phi_1(dt J) is the whole integrator. Oracle: scipy.linalg.expm on the AUGMENTED matrix,
    #     expm([[dt J, dt F], [0, 0]])[:2, 2] == dt phi_1(dt J) F,
    # which is a completely different algorithm (Pade scaling-and-squaring on a 3x3) reaching the
    # same quantity. Swept over 400 random stiff 2x2s with the sign structure _fb_jacobian
    # guarantees (J11, J22 < 0; J12, J21 >= 0; det > 0) and dt over six decades.
    from scipy.linalg import expm

    from dynameta.optics.fiber_amp.dynamics import _phi1_dt_2x2
    rng = np.random.default_rng(20260913)
    worst = 0.0
    for _ in range(400):
        scale = 10.0 ** rng.uniform(0.0, 6.0)
        a, d = -scale * rng.uniform(0.1, 1.0), -scale * rng.uniform(0.1, 1.0)
        b, c = scale * rng.uniform(0.0, 1.0), scale * rng.uniform(0.0, 1.0)
        if a * d - b * c <= 0.0:
            b, c = 0.1 * b, 0.1 * c
        dt = 10.0 ** rng.uniform(-9.0, -3.0)
        F = rng.normal(size=2) * scale * 1e-3
        M = np.array(_phi1_dt_2x2(*[np.array([v]) for v in (a, b, c, d)], dt)).reshape(2, 2)
        A = np.zeros((3, 3))
        A[:2, :2] = dt * np.array([[a, b], [c, d]])
        A[:2, 2] = dt * F
        ref = expm(A)[:2, 2]
        worst = max(worst, float(np.max(np.abs(M @ F - ref)) / max(np.max(np.abs(ref)), 1e-300)))
    assert worst < 1e-12, worst                              # MEASURED 1.0e-14

    # the case the scaling-and-squaring form exists for: EXACTLY coalesced eigenvalues, where the
    # divided-difference (eigen-decomposition) route is 0/0. The Er and Yb blocks cross routinely.
    F = np.array([1.0, -2.0])
    for eps in (0.0, 1e-14, 1e-8):
        a = d = -1.0e5
        M = np.array(_phi1_dt_2x2(*[np.array([v]) for v in (a, eps, eps, d)], 1e-6)).reshape(2, 2)
        A = np.zeros((3, 3))
        A[:2, :2] = 1e-6 * np.array([[a, eps], [eps, d]])
        A[:2, 2] = 1e-6 * F
        err = float(np.max(np.abs(M @ F - expm(A)[:2, 2])) / np.max(np.abs(expm(A)[:2, 2])))
        assert err < 1e-13, (eps, err)                       # MEASURED 1.1e-16 at all three


# ==================== gate (a): the Er-only limit reproduces the single-ion march ===========

def test_er_only_limit_reproduces_the_single_ion_march():
    # k_tr = 0 and a vanishing Yb density collapse the co-doped pair to the two-level EDFA
    # algebra, and the exponential-Rosenbrock update collapses (exactly, not approximately) to the
    # scalar exponential integrator: with the coupling off the Jacobian is diagonal and
    # dt phi_1(dt J) F reduces to n2_ss + (n2 - n2_ss) exp(-B dt) term for term. The two marches
    # are completely separate code, so this is a real cross-check of the 2x2 phi_1.
    fib = FiberSpec(2.0e-6, 0.20, 2.0e25, 6.0)
    pumps = [Pump(120e-3, 0.980e-6, "fwd")]
    signals = [Signal(50e-6, 1.560e-6)]
    t = np.linspace(0.0, 40e-3, 200)
    single = simulate_transient(FiberAmplifier(ER, fib, pumps, signals, None), t,
                                n_nodes=31, nbar2_0=0.10)
    co = simulate_transient(ErYbAmplifier(ER, YB, fib, pumps, signals, None, n_yb_m3=1e10,
                                          k_tr_m3_s=0.0), t, n_nodes=31, nbar2_0=(0.10, 0.0))
    dg = float(np.max(np.abs(co.signal_gain_dB - single.signal_gain_dB)))
    assert dg <= 1e-3, dg                                     # the stated gate
    assert dg < 1e-9, dg                                      # MEASURED 3.6e-14 dB
    assert float(np.max(np.abs(co.nbar2_zt - single.nbar2_zt))) < 1e-12
    # the Yb reservoir is present but inert at N_Yb = 1e10 / k_tr = 0
    assert float(np.max(co.meta["beta_yb"])) < 1.0


# ==================== gate (b): steady consistency ==========================================

def test_constant_drive_transient_sits_on_the_steady_solve():
    amp = _clad_amp()
    ss = amp.solve(n_nodes=161)
    tr = simulate_transient(amp, np.linspace(0.0, 30e-3, 120), n_nodes=161)
    assert tr.meta["quasi_static_valid"]
    dev = float(np.max(np.abs(tr.signal_gain_dB[:, 0] - ss.signal_gain_dB[0])))
    assert dev < 1e-3, dev                                    # MEASURED 4.2e-4 dB at 161 nodes
    # the seed is the solve itself, so the FIRST frame is the tightest statement of all
    assert abs(float(tr.signal_gain_dB[0, 0]) - float(ss.signal_gain_dB[0])) < 5e-4
    # and the Yb reservoir is carried, not silently dropped
    assert tr.meta["beta_yb"].shape == tr.nbar2_zt.shape
    assert np.allclose(tr.meta["beta_yb"][0], ss.meta["beta_yb_z"], atol=1e-9)


def test_perturbed_start_settles_onto_the_steady_gain():
    amp = _clad_amp()
    ss = amp.solve(n_nodes=161)
    t = np.concatenate([[0.0], np.logspace(-6, np.log10(60e-3), 200)])
    tr = simulate_transient(amp, t, n_nodes=161, nbar2_0=(0.02, 0.02))
    assert tr.meta["quasi_static_valid"]
    assert float(tr.signal_gain_dB[0, 0]) < -20.0             # really started cold
    end = float(tr.signal_gain_dB[-1, 0])
    assert abs(end - float(ss.signal_gain_dB[0])) < 1e-3, end  # MEASURED 2.9e-4 dB
    assert np.all(np.diff(tr.signal_gain_dB[:, 0]) > -1e-6)   # monotone charge-up, no ringing


def test_f2_only_seed_starts_the_yb_reservoir_at_its_quasi_equilibrium():
    # premise: with BOTH reservoirs supplied the march reaches the steady gain (above). A bare f2
    # has to invent b2; seeding it at 0 would inject a spurious millisecond of Yb charging, so it
    # is seeded from the Yb closed form at that f2 and the first drive.
    amp = _clad_amp()
    ss = amp.solve(n_nodes=161)
    t = np.concatenate([[0.0], np.logspace(-6, np.log10(60e-3), 200)])
    tr = simulate_transient(amp, t, n_nodes=161, nbar2_0=0.02)
    assert tr.meta["beta_yb_seed_residual"] < 1e-6            # the seed fixed point converged
    assert float(np.mean(tr.meta["beta_yb"][0])) > 1e-3       # NOT zero
    cold = simulate_transient(amp, t, n_nodes=161, nbar2_0=(0.02, 0.0))
    assert float(np.mean(cold.meta["beta_yb"][0])) == 0.0     # the explicit pair IS honoured
    # both reach the same operating point -- the seed changes the path, not the fixed point
    assert abs(float(tr.signal_gain_dB[-1, 0]) - float(ss.signal_gain_dB[0])) < 1e-3
    assert abs(float(cold.signal_gain_dB[-1, 0]) - float(ss.signal_gain_dB[0])) < 1e-3


# ==================== gate (c): step-drive energy conservation ==============================

def _step_closure(dt, *, n_nodes=161, t_end=3.0e-5, t_ref=2.0e-5, idle_dB=20.0):
    """Idle -> data step seeded from the IDLE steady state (so the drive is constant across the
    whole march and no step straddles a discontinuity). Returns the step-averaged residuals of
    the power balance for the step STARTING at t_ref:

        resid = |0.5 [(in - out - D)_i + (in - out - D)_{i+1}] - (U_{i+1} - U_i)/dt| / launched

    'flux' reads (in - out) off the propagated endpoints; 'rate' reads it as INT q_opt dz from the
    rate balance instead, so their DIFFERENCE is the z-mesh error alone and their common part is
    the time-integration error alone."""
    amp = _clad_amp()
    p_data = float(amp.signals[0].power_W)
    idle = amp.with_signals([Signal(p_data * 10 ** (-idle_dB / 10.0), 1.550e-6)])
    r0 = idle.solve(n_nodes=n_nodes)
    seed = (np.asarray(r0.nbar2_z, float), np.asarray(r0.meta["beta_yb_z"], float))
    nt = int(round(t_end / dt)) + 1
    tr = simulate_transient(amp, np.arange(nt) * dt, n_nodes=n_nodes, nbar2_0=seed,
                            store_profiles=True)
    assert tr.meta["quasi_static_valid"]
    launched = float(amp.pumps[0].power_W) + p_data
    i0 = int(round(t_ref / dt))
    U, net, qnet = {}, {}, {}
    for i in (i0, i0 + 1):
        P = tr.power_zt[i]
        te = amp.energy_terms(P, tr.nbar2_zt[i], tr.meta["beta_yb"][i])
        D = float(trapz(te["dissipation"], tr.z_m))
        U[i] = amp.stored_energy_J(tr.nbar2_zt[i], tr.meta["beta_yb"][i], tr.z_m)
        net[i] = (launched - _exiting_W(P, tr.plan.direction)) - D
        qnet[i] = float(trapz(te["q_optical"], tr.z_m)) - D
    dU = (U[i0 + 1] - U[i0]) / dt
    return {"flux": abs(0.5 * (net[i0] + net[i0 + 1]) - dU) / launched,
            "rate": abs(0.5 * (qnet[i0] + qnet[i0 + 1]) - dU) / launched,
            "mesh": abs(net[i0] - qnet[i0]) / launched,
            "dU_rel": abs(dU) / launched}


def test_step_drive_energy_balance_closes_and_is_first_order_in_dt():
    # launched - exiting - dissipated - d(stored)/dt, step by step, on an idle -> data step.
    # `stored` is BOTH reservoirs' excitation energy. The march freezes the powers across a step,
    # so the balance closes to O(dt) -- that is the quasi-static approximation itself, not a
    # defect of the integrator, and the gate is the ORDER as much as the size.
    out = [_step_closure(dt) for dt in (1.0e-6, 5.0e-7, 2.5e-7)]
    dU = [o["dU_rel"] for o in out]
    assert min(dU) > 0.1, dU              # premise: the reservoirs really are moving fast here
    r = [o["rate"] for o in out]
    # MEASURED 2026-09-13: 1.77e-3 / 9.50e-4 / 4.92e-4 -- ratios 1.86, 1.93 -> first order
    assert r[0] < 5e-3 and r[-1] < 1e-3, r
    assert r[0] / r[1] > 1.6 and r[1] / r[2] > 1.6, r
    # and the end-to-end (flux-endpoint) version tracks it
    assert all(abs(o["flux"] - o["rate"]) < 0.3 * o["rate"] + 1e-4 for o in out)


def test_step_drive_mesh_error_is_second_order_in_dz():
    # the part of the step-by-step residual that is the z mesh rather than the time step: it is
    # the gap between the propagated endpoints and the integral of the amplifier's own RHS.
    mesh = [_step_closure(5.0e-7, n_nodes=n)["mesh"] for n in (81, 161, 321)]
    # MEASURED: 2.33e-4 / 5.81e-5 / 1.45e-5 -- exactly 4x per doubling
    assert mesh[0] / mesh[1] > 3.5 and mesh[1] / mesh[2] > 3.5, mesh
    assert mesh[-1] < 5e-5, mesh


# ==================== gate (e): stiff-regime stability ======================================

def test_stiff_pair_is_stable_over_four_decades_of_dt():
    # A SHORT fiber (1 m) so a deliberately far-from-equilibrium seed cannot also trip the
    # separate audit-A-7 ASE monitor: the claim under test is the INTEGRATOR's, and mixing the two
    # failure modes into one gate would make neither of them falsifiable. (The ASE monitor has its
    # own gate below.)
    amp = _clad_amp(length_m=1.0)
    stiffness = amp._k_tr * amp._n_er * YB.tau_s
    assert 4.0 < stiffness < 8.0, stiffness          # premise: k_tr N_Er tau_Yb ~ 6, as specified
    seen = []
    for dt in np.logspace(-8, -4, 9):
        tr = simulate_transient(amp, np.arange(30) * dt, n_nodes=41, nbar2_0=(0.4, 0.3))
        f2, b2 = tr.nbar2_zt, tr.meta["beta_yb"]
        assert np.all(np.isfinite(f2)) and np.all(np.isfinite(b2)), dt
        assert f2.min() >= 0.0 and f2.max() <= 1.0, (dt, f2.min(), f2.max())
        assert b2.min() >= 0.0 and b2.max() <= 1.0, (dt, b2.min(), b2.max())
        seen.append(tr.meta["max_dt_times_rate"])
    # PREMISE that the sweep really did span the stiff regime: ||dt J||_inf crosses 1 inside it,
    # so the last decades take steps LONGER than the fastest local time constant -- exactly the
    # regime an explicit or semi-implicit-split update cannot survive.
    # MEASURED 2026-09-13: 3.63e-4 at dt = 1e-8 up to 3.63 at dt = 1e-4.
    assert min(seen) < 0.05, seen
    assert max(seen) > 3.0, seen
    assert max(seen) / min(seen) > 1e3, seen


def test_coarse_grid_from_a_cold_start_trips_the_quasi_static_flag():
    # PREMISE: a well-resolved grid from the same cold start is valid and lands on the steady gain
    # (asserted in test_perturbed_start_settles_onto_the_steady_gain). A grid coarse enough that a
    # single step carries the inversion across the ASE-clamped operating point does NOT: the
    # frozen-population step then propagates a gain the ASE it generates never gets to deplete.
    amp = _clad_amp()
    with pytest.warns(RuntimeWarning, match="OUT OF ITS VALID REGIME"):
        tr = simulate_transient(amp, np.linspace(0.0, 60e-3, 200), n_nodes=161,
                                nbar2_0=(0.02, 0.02))
    assert tr.meta["quasi_static_valid"] is False
    assert tr.meta["max_ase_to_launched"] > 1.0
    assert np.all(np.isfinite(tr.nbar2_zt))          # still returned, still bounded
    assert tr.nbar2_zt.max() <= 1.0 and tr.meta["beta_yb"].max() <= 1.0


# ==================== the public contract the downstream study calls ========================

def test_simulate_transient_dispatches_and_keeps_the_result_contract():
    amp = _clad_amp(bins=4)
    t = np.linspace(0.0, 2e-3, 10)
    tr = simulate_transient(amp, t, n_nodes=41, store_profiles=True)
    direct = simulate_transient_eryb(amp, t, n_nodes=41, store_profiles=True)
    assert np.array_equal(tr.nbar2_zt, direct.nbar2_zt)       # the dispatch IS the function
    # everything the study reads off a TransientResult
    assert tr.signal_gain_dB.shape == (10, 1)
    assert tr.pump_out_W.shape == (10, 1)
    assert tr.ase_fwd_W.shape == (10, 4) and tr.ase_bwd_W.shape == (10, 4)
    assert np.all(np.diff(tr.ase_lambda_m) > 0.0)             # sorted by wavelength
    assert tr.ase_psd_1pol_W_Hz("fwd").shape == (10, 4)
    assert tr.plan is not None and tr.plan.channels is None   # co-doped plan: two ions per channel
    assert tr.power_zt.shape == (10, tr.plan.lambda_m.size, 41)
    fr = tr.frame_as_steady(5)
    assert fr.signal_gain_dB.shape == (1,)
    assert np.array_equal(fr.nbar2_z, tr.nbar2_zt[5])
    # the co-doped frame must carry the spectroscopy AND the Yb inversion (ChannelPlan.channels is
    # None here, so without the meta fallback the frame would reach the noise layer bare)
    for key in ("sigma_a", "sigma_e", "sigma_a_yb", "sigma_e_yb", "beta_yb_z"):
        assert key in fr.meta, key
    assert np.array_equal(fr.meta["beta_yb_z"], tr.meta["beta_yb"][5])


def test_simulate_transient_still_refuses_classes_it_cannot_march():
    # the dispatch must not become "anything with a _plan()": the radially-resolved class has a
    # per-node inversion FIELD, not a mean-field reservoir, and there is no march for it here.
    from dynameta.optics.fiber_amp.transverse import ResolvedFiberAmplifier
    amp = ResolvedFiberAmplifier(ER, FiberSpec(2.0e-6, 0.20, 2.0e25, 4.0),
                                 [Pump(50e-3, 0.980e-6, "fwd")], [Signal(1e-4, 1.55e-6)], None)
    with pytest.raises(TypeError, match="FiberAmplifier and ErYbAmplifier only"):
        simulate_transient(amp, np.linspace(0.0, 1e-3, 4))


def test_nbar2_0_tuple_must_be_a_pair():
    amp = _clad_amp(bins=2)
    with pytest.raises(ValueError, match="must be the pair"):
        simulate_transient(amp, np.linspace(0.0, 1e-4, 3), n_nodes=21, nbar2_0=(0.1, 0.2, 0.3))


def test_solve_accepts_relax_auto_without_changing_its_default():
    # substitutability, not a new default: a caller written against FiberAmplifier passes
    # relax="auto" straight through (the downstream solve_closed does exactly that), and this
    # class used to answer with a ValueError. It now walks the same (1.0, 0.5, 0.25) ladder --
    # while relax=1.0 STAYS the default here and still takes exactly one attempt.
    amp = _clad_amp(bins=4)
    pinned = amp.solve(n_nodes=81)
    assert pinned.meta["relax"] == 1.0
    assert pinned.meta["relax_attempts"] == (1.0,)
    auto = amp.solve(n_nodes=81, relax="auto")
    assert auto.meta["relax_attempts"][0] == 1.0
    # this amplifier converges on the first rung, so "auto" must return the SAME numbers as the
    # default -- the ladder is a path, not a different fixed point
    assert np.array_equal(auto.power_W, pinned.power_W)
    assert np.array_equal(auto.nbar2_z, pinned.nbar2_z)
    with pytest.raises(ValueError, match='relax must be a number'):
        amp.solve(n_nodes=41, relax="fast")


def test_amplifier_saturation_energy_picks_the_right_ion():
    amp = _clad_amp(bins=2)
    lam = 1.550e-6
    assert amplifier_saturation_energy(amp, lam) == saturation_energy(ER, amp.fiber, lam)
    plain = FiberAmplifier(ER, amp.fiber, list(amp.pumps), list(amp.signals), None)
    assert amplifier_saturation_energy(plain, lam) == saturation_energy(ER, amp.fiber, lam)


# ==================== gate (f): the single-ion paths are BYTE-IDENTICAL =====================

# Recorded on main @ 3496a99 (pre-change) and re-read on this branch: the dispatch, the
# frame_as_steady fallback and the efficiency hook must not move the FiberAmplifier path.
#
# WHY TOLERANCES AND NOT `==`. The first version of this gate compared float64 hex with `==`.
# That was WRONG, and this branch's first CI run falsified it twice, in two different ways, while
# every other test on both legs passed:
#   * floor leg (numpy 1.24 / scipy 1.10 / py3.10): all ten MARCH values matched bit for bit, but
#     the wall-plug residual came out 1.0193e-4 W against the 1.0187e-4 W recorded here -- 6.2e-4
#     relative. `amp.solve()` drives scipy's ADAPTIVE LSODA, whose step sequence differs between
#     scipy versions, so it settles on a slightly different point of the same fixed point.
#   * py3.10 leg (newest numpy): `gain_last` differed by ONE ULP (...d80 vs ...d7f) -- a reduction
#     order / BLAS difference inside the march itself.
# Bitwise reproducibility is therefore a property of the BUILD, not of this change, and asserting
# it is a FALSE GATE. What is pinned below are tolerances 7+ orders tighter than any regression
# this change could cause -- a mis-routed march or a mis-fired efficiency hook moves gains by dB
# or returns NaN, not by 1e-9 -- and the environment-INDEPENDENT half of the claim (that the three
# new branches cannot be entered AT ALL on a FiberAmplifier) is the structural test below it.
_MARCH_RTOL = 1e-9             # 1 ULP is 2e-16 relative, so this carries ~7 orders of headroom
_MAIN_REF_MARCH = {            # march-derived
    "gain_first": float.fromhex("-0x1.134743697490ep+5"),
    "gain_last": float.fromhex("0x1.d896223627d7fp+4"),
    "nbar2_sum": float.fromhex("0x1.623cd0a5dfcd6p+7"),
    "sig_out_last": float.fromhex("0x1.702683058903cp-5"),
    "pmp_out_last": float.fromhex("0x1.757e645fc9bfdp-19"),
    "ase_fwd_sum": float.fromhex("0x1.944fbc1fd8aa7p-8"),
    "ase_bwd_sum": float.fromhex("0x1.0c5a8e008ca22p-2"),
    "power_sum": float.fromhex("0x1.4151a15d47f9cp+4"),
    "frame7_gain": float.fromhex("0x1.d8994926e9221p+4"),
    "frame7_nbar2": float.fromhex("0x1.802b6c8f4d9bep+3"),
}
# solve()-derived: LSODA-path dependent across scipy versions, so a looser (still tiny) bound.
_SOLVE_RTOL = 1e-5
_MAIN_REF_SOLVE = {
    "wpe_eta": float.fromhex("0x1.35c525dfa543dp-3"),
    "wpe_heat": float.fromhex("0x1.c579edf2f9106p-5"),
    "ss_gain": float.fromhex("0x1.d878d5553d0c6p+4"),
}
# the power-balance residual, held in the unit it MEANS: a fraction of the launched pump.
_REF_PUMP_W = 0.120
_MAIN_REF_RESID_OVER_PUMP = float.fromhex("0x1.ab429dfd45c00p-14") / _REF_PUMP_W
_RESID_TOL_OVER_PUMP = 1e-5    # 20x the 5.2e-7-of-launched shift the floor leg measured


def _single_ion_reference_run():
    amp = FiberAmplifier(erbium(), FiberSpec(2.0e-6, 0.20, 2.0e25, 6.0),
                         [Pump(120e-3, 0.980e-6, "fwd")], [Signal(50e-6, 1.560e-6)],
                         AseBand(1.53e-6, 1.565e-6, 4))
    tr = simulate_transient(amp, np.linspace(0.0, 20e-3, 17), n_nodes=21, nbar2_0=0.10,
                            store_profiles=True)
    fr = tr.frame_as_steady(7)
    ss = amp.solve(n_nodes=81)
    b = wall_plug_efficiency(amp, ss, PumpSource(wallplug_efficiency=0.45,
                                                coupling_efficiency=0.9))
    return {
        "gain_first": float(tr.signal_gain_dB[0, 0]),
        "gain_last": float(tr.signal_gain_dB[-1, 0]),
        "nbar2_sum": float(np.sum(tr.nbar2_zt)),
        "sig_out_last": float(tr.signal_out_W[-1, 0]),
        "pmp_out_last": float(tr.pump_out_W[-1, 0]),
        "ase_fwd_sum": float(np.sum(tr.ase_fwd_W)),
        "ase_bwd_sum": float(np.sum(tr.ase_bwd_W)),
        "power_sum": float(np.sum(tr.power_zt)),
        "frame7_gain": float(fr.signal_gain_dB[0]),
        "frame7_nbar2": float(np.sum(fr.nbar2_z)),
        "wpe_resid": float(b.energy_balance_residual_W),
        "wpe_eta": float(b.eta_wallplug),
        "wpe_heat": float(b.heat_W),
        "ss_gain": float(ss.signal_gain_dB[0]),
    }, fr, amp


def test_single_ion_march_and_efficiency_are_unchanged_from_main():
    got, fr, amp = _single_ion_reference_run()
    for key, want in _MAIN_REF_MARCH.items():
        assert abs(got[key] - want) <= _MARCH_RTOL * abs(want), (key, got[key].hex(), want.hex())
    for key, want in _MAIN_REF_SOLVE.items():
        assert abs(got[key] - want) <= _SOLVE_RTOL * abs(want), (key, got[key], want)
    assert abs(got["wpe_resid"] / _REF_PUMP_W - _MAIN_REF_RESID_OVER_PUMP) < _RESID_TOL_OVER_PUMP
    # the frame's meta must not have GROWN co-doped keys on a single-ion amplifier either
    assert set(fr.meta) == {"converged", "iterations", "dnu_hz", "gamma", "m_modes", "mcc",
                            "transient_frame", "t_s", "quasi_static_valid",
                            "sigma_a", "sigma_e", "sigma_esa"}


def test_the_new_branches_are_structurally_unreachable_for_a_single_ion_amplifier():
    """The environment-INDEPENDENT half of gate (f). The numeric pins above say the ANSWERS did
    not move; these say the three new branches cannot even be ENTERED on a FiberAmplifier, which
    is the actual claim and does not depend on a numpy/scipy build.

      * efficiency._dissipated_power_W tries `_rate_balance_dissipation_W` FIRST -- FiberAmplifier
        must not define it, or the single-ion closure would silently change algorithm.
      * simulate_transient dispatches only on isinstance(amp, ErYbAmplifier).
      * frame_as_steady's co-doped branch runs only when plan.channels is None.
    """
    _, fr, amp = _single_ion_reference_run()
    assert not hasattr(amp, "_rate_balance_dissipation_W")
    assert hasattr(amp, "_dP_full_c")                       # so it takes the ORIGINAL path
    assert isinstance(amp, FiberAmplifier) and not isinstance(amp, ErYbAmplifier)
    assert amp.channel_plan().channels is not None          # so the co-doped frame branch is dead
    assert "beta_yb_z" not in fr.meta and "sigma_a_yb" not in fr.meta
