"""Discrimination-proven gates for the Er:Yb TWO-POPULATION physics (2026-09-14):
the coupled/uncoupled ytterbium split (Dong 2020), the secondary transfer K2 (Sefler 2004),
the Yb-Yb migration exchange, the ConcentrationModel on the co-doped class, distributed
temperature for both ions, and the phosphosilicate calibration factories.

Every mechanism is OPT-IN and every gate here is falsifiable: the byte-identity block pins the
v0.11.1 answers to tolerances seven orders tighter than any regression this change could cause,
and the physics blocks are monotonicity and limit statements that a wrong sign or a mis-weighted
pool breaks immediately. Bidirectional-adversarial: the claimed successes (transfer rises with
the coupled fraction, migration equalizes the pools, K2 drains the ytterbium) are proven out, and
the claimed failure mode (a single-population model cannot separate the 1-um parasitic threshold
from the output power) is what gate 5 measures rather than asserts.

Audit trail: docs/audit/2026-09-14-eryb-two-population-physics.md.
"""

import numpy as np
import pytest

from dynameta.core.numerics import trapz
from dynameta.optics.fiber_amp import (
    erbium, ytterbium, FiberSpec, Pump, Signal, AseBand, FiberAmplifier, ConcentrationModel,
    ThermalModel, solve_with_thermal_feedback, at_temperature, simulate_transient,
    wall_plug_efficiency, PumpSource, ytterbium_melkumov, er_yb_phosphosilicate_reference,
)
from dynameta.optics.fiber_amp.calibration import (
    _YB_MELKUMOV_AS_NM, _YB_MELKUMOV_AS_SIGMA_A_PM2, _YB_MELKUMOV_AS_SIGMA_E_PM2,
)
from dynameta.optics.fiber_amp.eryb import ErYbAmplifier

ER = erbium("aluminosilicate")
YB = ytterbium("phosphosilicate")
_BG = 30e-3 * np.log(10.0) / 10.0                      # 30 dB/km, the research-grade figure
REF_PUMP_W = 1.0


def ref_amp(**kw):
    """THE reference co-doped fixture of this file: a 6 um-core cladding-pumped Er:Yb at the
    study's own densities (N_Er 4e25, N_Yb 4e26, 10:1), 1 W of 976 nm into a 125 um cladding,
    20 mW of 1550 nm seed, 3 m. It amplifies (~13.8 dB), its transfer efficiency is 0.76 -- well
    inside the measured 0.6-0.9 band -- and its 1-um parasitic gain is -25 dB, i.e. it sits in
    the regime the model is meant for rather than on a numerical edge."""
    fib = FiberSpec(3.0e-6, 0.20, 4.0e25, 3.0, clad_radius_m=62.5e-6, background_loss_per_m=_BG)
    base = dict(n_yb_m3=4.0e26, k_tr_m3_s=2.0e-22, yb_ase=AseBand(1.00e-6, 1.10e-6, 4),
                upconversion_C_up=1.1e-24)
    base.update(kw)
    return ErYbAmplifier(ER, YB, fib, [Pump(REF_PUMP_W, 0.976e-6, "fwd", cladding=True)],
                         [Signal(20e-3, 1.550e-6)], AseBand(1.53e-6, 1.565e-6, 6), **base)


def _all_on(**kw):
    """The same fixture with EVERY new mechanism active at literature-range values."""
    conc = ConcentrationModel(c_up_m3_s=1.1e-24, pair_fraction=0.03, pair_convention="delevaque",
                              pd_loss_per_m=0.02, pd_exponent=1.0)
    base = dict(yb_coupled_fraction=0.85, k_tr_m3_s=3.0e-21, k_tr2_m3_s=2.0e-22,
                yb_migration_rate_per_s=5.0e3, concentration=conc, upconversion_C_up=0.0)
    base.update(kw)
    return ref_amp(**base)


# ================== gate 1: the DEFAULTS are v0.11.1, to 1e-5 relative =======================
# Recorded on main @ a2ad5f1 (v0.11.1, pre-change) with the fixture above, re-read on this
# branch, where all 29 recorded values came back BIT-IDENTICAL. That bit-identity is NOT what is
# asserted here, and the reason is the lesson the 2026-09-13 gate (f) recorded the hard way: CI
# falsified a bitwise pin twice, once through scipy's adaptive LSODA step sequence (6.2e-4
# relative on the floor leg) and once through a 1-ULP BLAS reduction difference. Bitwise
# reproducibility is a property of the BUILD, not of this change.
#
# EVERY value pinned below is LSODA-path dependent -- including the transient ones, because this
# fixture seeds the march from `amp.solve()` (nbar2_0 = None) rather than from an explicit
# number, so the march inherits the steady solve's path. The 2026-09-13 gate could hold its march
# block to 1e-9 precisely because it seeded explicitly; that distinction is the whole reason this
# block uses one looser tolerance instead of two. 1e-5 still carries four orders of headroom over
# the largest environment shift that note measured, and eleven orders over any regression this
# change could cause: a mis-routed pool or a leaked K2 term moves gains by dB.
#
# The two halves of the claim that a build CANNOT move are separate gates:
#   * test_explicit_defaults_are_bitwise_identical_to_omitting_them -- same process, same
#     arithmetic, exact equality, so it is reproducible on every runner;
#   * test_the_new_branches_are_structurally_unreachable_with_the_defaults -- the branches cannot
#     be entered at all.
_V0111_RTOL = 1e-5
_V0111_STEADY = {
    "ss_gain_dB": 13.821288424131597,
    "ss_nbar2_sum": 57.21842385984466,
    "ss_beta_yb_sum": 4.426301986364895,
    "ss_power_sum": 80.20120067422589,
    "ss_eta_tr": 0.7556463976787159,
    "ss_yb_par_dB": -24.87318007637147,
}
_V0111_CLOSURE = {
    "et_q_optical": 20.29268117688902,
    "et_stored_J_per_m": 0.018606243199211947,
    "et_background_loss": 0.5540102673380897,
    "et_er_dissipation": 1.4092514018140272,
    "et_yb_dissipation": 6.975811359430976,
    "et_transfer_defect": 11.353608148308616,
    "et_dissipation": 20.29268117689171,
    "et_transfer_rate_per_m": 1.5340742485769965e+20,
    "stored_J": 0.00045618774481413673,
}
_V0111_TRANSIENT = {
    "tr_gain_first": 13.887291060354897,
    "tr_gain_last": 9.998040530973174,
    "tr_nbar2_sum": 166.81194690864368,
    "tr_beta_yb_sum": 8.544504825715151,
    "tr_sig_out_last": 0.1999097834702496,
    "tr_power_sum": 137.6724195193193,
    "tr_ase_fwd_sum": 0.0017458625517523224,
    "tr_ase_bwd_sum": 0.006288938837448401,
    "tr_pmp_out_last": 0.012322195521123153,
}
_V0111_SOLVE_RTOL = 1e-5
_V0111_WPE = {"wpe_eta": 0.18716025911380932, "wpe_heat_W": 0.5008411925823796}
_V0111_RESID_OVER_PUMP = -7.726057247747775e-06
_RESID_TOL_OVER_PUMP = 1e-5


def _pump_drive(t):
    return np.array([REF_PUMP_W if t < 1e-3 else 0.4 * REF_PUMP_W])


def test_defaults_reproduce_v0_11_1_steady_closure_and_transient():
    amp = ref_amp()
    r = amp.solve(n_nodes=121)
    got = {"ss_gain_dB": float(r.signal_gain_dB[0]),
           "ss_nbar2_sum": float(np.sum(r.nbar2_z)),
           "ss_beta_yb_sum": float(np.sum(r.meta["beta_yb_z"])),
           "ss_power_sum": float(np.sum(r.power_W)),
           "ss_eta_tr": float(r.meta["eta_transfer"]),
           "ss_yb_par_dB": float(r.meta["yb_parasitic_gain_dB"])}
    for k, want in _V0111_STEADY.items():
        assert abs(got[k] - want) <= _V0111_RTOL * abs(want), (k, got[k], want)

    et = amp.energy_terms(r.power_W, r.nbar2_z, r.meta["beta_yb_z"])
    got_c = {"et_" + k: float(np.sum(et[k])) for k in
             ("q_optical", "stored_J_per_m", "background_loss", "er_dissipation",
              "yb_dissipation", "transfer_defect", "dissipation", "transfer_rate_per_m")}
    got_c["stored_J"] = float(amp.stored_energy_J(r.nbar2_z, r.meta["beta_yb_z"], r.z_m))
    for k, want in _V0111_CLOSURE.items():
        assert abs(got_c[k] - want) <= _V0111_RTOL * abs(want), (k, got_c[k], want)
    # the NEW term must be identically zero, not merely small, with k_tr2 = 0
    assert np.all(et["k2_defect"] == 0.0) and np.all(et["k2_rate_per_m"] == 0.0)

    b = wall_plug_efficiency(amp, r, PumpSource(wallplug_efficiency=0.45, coupling_efficiency=0.9))
    for k, want in _V0111_WPE.items():
        v = float(b.eta_wallplug if k == "wpe_eta" else b.heat_W)
        assert abs(v - want) <= _V0111_SOLVE_RTOL * abs(want), (k, v, want)
    assert abs(float(b.energy_balance_residual_W) / REF_PUMP_W
               - _V0111_RESID_OVER_PUMP) < _RESID_TOL_OVER_PUMP

    tr = simulate_transient(amp, np.linspace(0.0, 4e-3, 17), n_nodes=21,
                            pump_drive=_pump_drive, store_profiles=True)
    assert tr.meta["quasi_static_valid"]
    got_t = {"tr_gain_first": float(tr.signal_gain_dB[0, 0]),
             "tr_gain_last": float(tr.signal_gain_dB[-1, 0]),
             "tr_nbar2_sum": float(np.sum(tr.nbar2_zt)),
             "tr_beta_yb_sum": float(np.sum(tr.meta["beta_yb"])),
             "tr_sig_out_last": float(tr.signal_out_W[-1, 0]),
             "tr_power_sum": float(np.sum(tr.power_zt)),
             "tr_ase_fwd_sum": float(np.sum(tr.ase_fwd_W)),
             "tr_ase_bwd_sum": float(np.sum(tr.ase_bwd_W)),
             "tr_pmp_out_last": float(tr.pump_out_W[-1, 0])}
    for k, want in _V0111_TRANSIENT.items():
        assert abs(got_t[k] - want) <= _V0111_RTOL * abs(want), (k, got_t[k], want)
    assert tr.meta["beta_yb_coupled"] is None and tr.meta["beta_yb_uncoupled"] is None


def test_explicit_defaults_are_bitwise_identical_to_omitting_them():
    """The build-INDEPENDENT numeric half of gate 1. Spelling every new parameter out at its
    default value -- including an all-default (identity) ConcentrationModel -- must give the
    SAME FLOATS as omitting them, to the last bit. This runs both amplifiers in ONE process with
    ONE numpy and ONE scipy, so unlike a recorded pin it is exact on every runner, and it is the
    assertion that actually says "the new physics is off by default"."""
    bare = ref_amp()
    spelled = ref_amp(yb_coupled_fraction=1.0, k_tr2_m3_s=0.0, yb_migration_rate_per_s=0.0,
                      concentration=ConcentrationModel())
    assert spelled._two_pop is False and spelled.concentration is None
    r0, r1 = bare.solve(n_nodes=81), spelled.solve(n_nodes=81)
    for name in ("power_W", "nbar2_z", "signal_gain_dB"):
        assert np.array_equal(getattr(r0, name), getattr(r1, name)), name
    for key in ("beta_yb_z", "eta_transfer", "yb_parasitic_gain_dB"):
        assert np.array_equal(np.asarray(r0.meta[key]), np.asarray(r1.meta[key])), key
    e0 = bare.energy_terms(r0.power_W, r0.nbar2_z, r0.meta["beta_yb_z"])
    e1 = spelled.energy_terms(r1.power_W, r1.nbar2_z, r1.meta["beta_yb_z"])
    assert set(e0) == set(e1)
    for key in e0:
        assert np.array_equal(np.asarray(e0[key]), np.asarray(e1[key])), key
    t0 = simulate_transient(bare, np.linspace(0.0, 2e-3, 9), n_nodes=21,
                            pump_drive=_pump_drive)
    t1 = simulate_transient(spelled, np.linspace(0.0, 2e-3, 9), n_nodes=21,
                            pump_drive=_pump_drive)
    assert t0.meta["quasi_static_valid"] and t1.meta["quasi_static_valid"]
    for name in ("nbar2_zt", "signal_gain_dB", "signal_out_W", "pump_out_W"):
        assert np.array_equal(getattr(t0, name), getattr(t1, name)), name
    assert np.array_equal(t0.meta["beta_yb"], t1.meta["beta_yb"])


def test_the_new_branches_are_structurally_unreachable_with_the_defaults():
    """The build-INDEPENDENT half of gate 1: with every new parameter at its default the
    two-population routine, the concentration arithmetic and the McCumber scaling cannot be
    ENTERED, so the pinned numbers above cannot drift for a reason this change introduced."""
    amp = ref_amp()
    assert amp._two_pop is False and amp._fc == 1.0 and amp._k_tr2 == 0.0 and amp._w_mig == 0.0
    assert amp.concentration is None and amp._n_er_dark == 0.0
    assert amp._n_er == amp.fiber.n_t_m3 == amp._n_er_total
    assert amp._Tz is None
    pl = amp._plan()
    assert amp._mcc_matrices(pl, np.linspace(0.0, 3.0, 5)) is None
    # _fbb must hand back the ONE-pool answer, and the same object for the weighted mean
    f2, b2c, b2nc, bbar = amp._fbb(1e3, 1e2, 1e4, 1e3)
    assert b2nc is None and bbar is b2c
    r = amp.solve(n_nodes=41)
    assert r.meta["beta_yb_uncoupled_z"] is None
    assert r.meta["beta_yb_coupled_z"] is r.meta["beta_yb_z"]


# ================ gate 2: f -> 0 is a pure Yb-loss fiber and the Er-only limit ================

def test_zero_coupled_fraction_is_the_er_only_limit_with_the_same_total_yb():
    """With f = 0 NO ytterbium transfers, so at the SAME total N_Yb the amplifier must reduce to
    the k_tr = 0 fiber -- an erbium amplifier that additionally carries a full ytterbium pool
    soaking up pump and radiating it away. This is the sharpest single test of the pool
    bookkeeping: if any transfer term kept N_Yb instead of f N_Yb, or if the uncoupled pool were
    left out of the propagation, the two gains would not coincide."""
    a_f0 = ref_amp(yb_coupled_fraction=0.0, k_tr_m3_s=3.0e-21)
    a_k0 = ref_amp(k_tr_m3_s=0.0)
    r0, rk = a_f0.solve(n_nodes=81), a_k0.solve(n_nodes=81)
    assert r0.meta["converged"] and rk.meta["converged"]
    # premise: this fixture is actually pumped and actually transfers when f > 0, so the
    # agreement below is a statement about the coupling and not about a dead amplifier
    assert ref_amp(k_tr_m3_s=3.0e-21).solve(n_nodes=81).signal_gain_dB[0] > \
        float(r0.signal_gain_dB[0]) + 5.0
    assert abs(float(r0.signal_gain_dB[0]) - float(rk.signal_gain_dB[0])) < 1e-5
    assert r0.meta["eta_transfer"] == 0.0                      # nothing transfers, exactly
    # the ytterbium is still there and still absorbing: the uncoupled pool carries the inversion
    assert np.max(r0.meta["beta_yb_uncoupled_z"]) > 0.05
    assert np.allclose(r0.meta["beta_yb_z"], r0.meta["beta_yb_uncoupled_z"], rtol=0, atol=0)


# =========== gate 3: the coupled fraction, K2 and migration each move the right way ==========

def test_transfer_and_parasitic_threshold_track_the_coupled_fraction():
    """Dong et al. 2020's qualitative claim, as a MONOTONIC gate and not a number: at a FIXED
    f * k_tr (the product a single-population fit sees) a larger coupled fraction gives more
    transfer and a LOWER 1-um parasitic gain -- i.e. the parasitic threshold RISES with f. That
    is the whole content of their two-population argument: the 1-um behaviour and the output
    power are separate observables, which is exactly what a one-pool model cannot express (it
    needs two mutually inconsistent k_tr values, 1.1e-21 and 2.63e-21, for the same device)."""
    fk = 0.9 * 3.0e-21
    fs = (0.3, 0.5, 0.7, 0.9)
    gains, etas, par = [], [], []
    for f in fs:
        r = ref_amp(yb_coupled_fraction=f, k_tr_m3_s=fk / f).solve(n_nodes=81)
        assert r.meta["converged"]
        gains.append(float(r.signal_gain_dB[0]))
        etas.append(float(r.meta["eta_transfer"]))
        par.append(float(r.meta["yb_parasitic_gain_dB"]))
    # premise: the sweep must actually MOVE something, or monotonicity is vacuous
    assert max(par) - min(par) > 20.0, par
    assert np.all(np.diff(etas) > 0.0), etas
    assert np.all(np.diff(gains) > 0.0), gains
    assert np.all(np.diff(par) < 0.0), par                  # threshold rises as parasitic falls


def test_secondary_transfer_drains_the_coupled_yb_without_changing_the_er_balance():
    """K2 takes an excited Yb and an Er that is ALREADY in 4I13/2 and returns the Er to 4I13/2,
    so it must (i) lower the transfer efficiency and the gain monotonically, (ii) leave the Er
    rate equation algebraically untouched -- the Er RHS at a fixed state must not move at all --
    and (iii) show up as its own heat term, charged at the FULL Yb quantum."""
    amp0 = ref_amp(yb_coupled_fraction=0.9, k_tr_m3_s=3.0e-21, k_tr2_m3_s=0.0)
    gains, etas = [], []
    for k2 in (0.0, 1.5e-22, 4.0e-22):
        a = ref_amp(yb_coupled_fraction=0.9, k_tr_m3_s=3.0e-21, k_tr2_m3_s=k2)
        r = a.solve(n_nodes=81)
        gains.append(float(r.signal_gain_dB[0]))
        etas.append(float(r.meta["eta_transfer"]))
        # (ii) the Er equation is untouched at a fixed state
        df0 = amp0._fb_rhs3(1e3, 1e2, 1e4, 1e3, 0.5, 0.2, 0.3)[0]
        df = a._fb_rhs3(1e3, 1e2, 1e4, 1e3, 0.5, 0.2, 0.3)[0]
        assert df == df0
    assert gains[0] - gains[-1] > 0.3, gains                  # premise: K2 actually bites
    assert np.all(np.diff(gains) < 0.0), gains
    assert np.all(np.diff(etas) < 0.0), etas
    # (iii) the K2 heat is the FULL Yb quantum times the K2 rate
    a = ref_amp(yb_coupled_fraction=0.9, k_tr_m3_s=3.0e-21, k_tr2_m3_s=4.0e-22)
    r = a.solve(n_nodes=81)
    et = a.energy_terms(r.power_W, r.nbar2_z, r.meta["beta_yb_coupled_z"],
                        r.meta["beta_yb_uncoupled_z"])
    assert np.all(et["k2_defect"] >= 0.0) and float(np.max(et["k2_defect"])) > 0.0
    assert np.allclose(et["k2_defect"], float(a.yb_ion.eps_J) * et["k2_rate_per_m"],
                       rtol=1e-12, atol=0.0)


def test_migration_conserves_excitation_and_equalizes_the_two_pools():
    """The detailed-balance derivation, gated on its two defining properties. (1) The exchange
    conserves the TOTAL ytterbium excitation exactly: the population-weighted sum of the two
    migration terms is zero at every state, for every W. (2) Its only equilibrium is EQUAL
    EXCITATION FRACTIONS (never equal densities), so raising W drives |b2c - b2nc| to zero
    monotonically -- which is the ensemble-versus-pair reconciliation: a fast-pair k_tr on a
    small coupled pool then delivers the ensemble transfer yield of a large one."""
    a = ref_amp(yb_coupled_fraction=0.5, k_tr_m3_s=3.0e-21, yb_migration_rate_per_s=1.0e4)
    fc = a._fc
    rng = np.random.default_rng(11)
    for _ in range(50):
        y = rng.uniform(0.0, 1.0, 3)
        R = 10.0 ** rng.uniform(-1.0, 5.0, 4)
        a0 = ref_amp(yb_coupled_fraction=0.5, k_tr_m3_s=3.0e-21, yb_migration_rate_per_s=0.0)
        d_with = a._fb_rhs3(R[0], R[1], R[2], R[3], y[0], y[1], y[2])
        d_none = a0._fb_rhs3(R[0], R[1], R[2], R[3], y[0], y[1], y[2])
        net = (fc * (d_with[1] - d_none[1]) + (1.0 - fc) * (d_with[2] - d_none[2]))
        scale = max(abs(d_with[1]), abs(d_with[2]), 1.0)
        assert abs(float(net)) <= 1e-12 * scale, (net, scale)
    # EQUAL FRACTIONS, not equal densities, is the exchange's only equilibrium: at b2c = b2nc
    # the migration term vanishes identically in BOTH pools, whatever the pool sizes are. (The
    # two RHS values are NOT equal there -- the coupled pool still carries the transfer and K2
    # drains -- which is exactly why the test differences the W > 0 and W = 0 amplifiers.)
    a0 = ref_amp(yb_coupled_fraction=0.5, k_tr_m3_s=3.0e-21, yb_migration_rate_per_s=0.0)
    eq_w = a._fb_rhs3(1e3, 1e2, 1e4, 1e3, 0.4, 0.37, 0.37)
    eq_0 = a0._fb_rhs3(1e3, 1e2, 1e4, 1e3, 0.4, 0.37, 0.37)
    assert eq_w == eq_0
    # ... and it does NOT vanish when the fractions differ, so the check above is not vacuous
    ne_w = a._fb_rhs3(1e3, 1e2, 1e4, 1e3, 0.4, 0.20, 0.50)
    ne_0 = a0._fb_rhs3(1e3, 1e2, 1e4, 1e3, 0.4, 0.20, 0.50)
    assert ne_w[1] != ne_0[1] and ne_w[2] != ne_0[2]
    gaps, etas = [], []
    for w in (0.0, 1.0e2, 1.0e4, 1.0e6):
        r = ref_amp(yb_coupled_fraction=0.5, k_tr_m3_s=3.0e-21,
                    yb_migration_rate_per_s=w).solve(n_nodes=61)
        gaps.append(float(np.max(np.abs(r.meta["beta_yb_coupled_z"]
                                        - r.meta["beta_yb_uncoupled_z"]))))
        etas.append(float(r.meta["eta_transfer"]))
    assert gaps[0] > 0.1 and gaps[-1] < 1e-2, gaps           # premise + the limit
    assert np.all(np.diff(gaps) < 0.0), gaps
    assert np.all(np.diff(etas) > 0.0), etas


# ================= gate 4: the two algebras agree, and the 3x3 stays stable ==================

def test_three_unknown_algebra_reduces_to_the_one_pool_closed_form():
    """`_solve_fbb` at f = 1, K2 = 0, W = 0 must return `_solve_fb`'s root. They are separate
    code paths on purpose -- the one-pool form is retained VERBATIM so the default model is
    bit-identical -- so this is the gate that stops them drifting apart."""
    a1 = ref_amp()
    a2 = ref_amp()
    a2._two_pop = True                     # force the general routine on identical parameters
    rng = np.random.default_rng(3)
    worst = 0.0
    for _ in range(300):
        R = 10.0 ** rng.uniform(-2.0, 6.0, 4)
        f1, b1 = a1._solve_fb(*R)
        f2, b2, b3 = a2._solve_fbb(*R)
        # b3 (the uncoupled pool) is NOT compared: with W = 0 it carries no transfer drain, so
        # it sits ABOVE b2 by construction and is a different physical quantity.
        worst = max(worst, abs(f1 - f2), abs(b1 - b2))
    assert worst < 1e-11, worst


def test_three_state_jacobian_eigenvalues_stay_in_the_left_half_plane():
    """What licenses the exponential Rosenbrock step at n = 3. No theorem is claimed once K2
    breaks the cross-term cancellation, so the claim is gated numerically over a wide sweep of
    operating points: every eigenvalue keeps a strictly negative real part."""
    a = ref_amp(yb_coupled_fraction=0.6, k_tr_m3_s=3.0e-21, k_tr2_m3_s=4.0e-22,
                yb_migration_rate_per_s=1.0e4)
    rng = np.random.default_rng(5)
    worst_re = -np.inf
    for _ in range(300):
        R = 10.0 ** rng.uniform(-2.0, 6.0, 4)
        y = rng.uniform(0.0, 1.0, 3)
        J = a._fb_jacobian3(R[0], R[1], R[2], R[3], y[0], y[1], y[2])
        M = np.array([[float(J[i][j]) for j in range(3)] for i in range(3)])
        worst_re = max(worst_re, float(np.max(np.linalg.eigvals(M).real)))
    assert worst_re < 0.0, worst_re


# ================== gate 5: closure and the transient with EVERY mechanism on ================

def test_closure_holds_to_1e_5_with_every_mechanism_on():
    """Two independent statements. (1) The per-z identity q_opt = dU/dt + D_loss + D_Er + D_Yb +
    D_tr + D_K2 is an algebraic rearrangement of one rate balance, so it must close to round-off
    -- a missing pool weight or a double-charged defect breaks it at the percent level. (2) The
    wall-plug residual, an INDEPENDENT closure that re-feeds the returned profile through the
    amplifier's own right-hand side, falls below 1e-5 of the launched pump and converges with the
    mesh."""
    amp = _all_on()
    resid = []
    for n in (161, 321, 641):
        r = amp.solve(n_nodes=n)
        assert r.meta["converged"]
        et = amp.energy_terms(r.power_W, r.nbar2_z, r.meta["beta_yb_coupled_z"],
                              r.meta["beta_yb_uncoupled_z"])
        scale = float(np.max(np.abs(et["q_optical"])))
        assert scale > 0.1                                  # premise: there IS energy flowing
        ident = float(np.max(np.abs(et["q_optical"] - et["d_stored_dt"] - et["dissipation"])))
        assert ident < 1e-10 * scale, (n, ident, scale)
        b = wall_plug_efficiency(amp, r, PumpSource(wallplug_efficiency=1.0))
        resid.append(abs(float(b.energy_balance_residual_W)) / REF_PUMP_W)
    assert resid[-1] < 1e-5, resid
    assert resid[-1] < resid[0], resid                      # and it CONVERGES with the mesh
    # every new dissipation channel is present and positive, not silently zero
    r = amp.solve(n_nodes=161)
    et = amp.energy_terms(r.power_W, r.nbar2_z, r.meta["beta_yb_coupled_z"],
                          r.meta["beta_yb_uncoupled_z"])
    for key in ("transfer_defect", "k2_defect", "background_loss"):
        assert float(trapz(et[key], r.z_m)) > 0.0, key


def test_transient_settles_onto_the_steady_solve_with_every_mechanism_on():
    """The three-reservoir march must converge to the SAME fixed point the three-unknown steady
    solve finds -- that is the property the exponential Rosenbrock step was chosen for, and it is
    what a mis-signed migration term or a mis-ordered state vector destroys."""
    amp = _all_on()
    r = amp.solve(n_nodes=81)
    t = np.concatenate([[0.0], np.logspace(-7.0, np.log10(0.08), 70)])
    tr = simulate_transient(amp, t, n_nodes=81, nbar2_0=(0.02, 0.02, 0.02))
    assert tr.meta["quasi_static_valid"], tr.meta["validity_warning"]
    assert tr.meta["beta_yb_coupled"] is not None and tr.meta["beta_yb_uncoupled"] is not None
    # the reported beta_yb is the POPULATION-WEIGHTED mean of the two pools
    fc = amp._fc
    assert np.allclose(tr.meta["beta_yb"],
                       fc * tr.meta["beta_yb_coupled"] + (1.0 - fc) * tr.meta["beta_yb_uncoupled"],
                       rtol=0.0, atol=0.0)
    # premise: the march really did start far away and travel
    assert abs(float(tr.signal_gain_dB[0, 0]) - float(r.signal_gain_dB[0])) > 5.0
    assert abs(float(tr.signal_gain_dB[-1, 0]) - float(r.signal_gain_dB[0])) < 0.05


def test_a_pair_seed_seeds_the_uncoupled_pool_at_its_quasi_equilibrium():
    """A 2-tuple seed on a two-population amplifier leaves the UNCOUPLED pool unstated, and
    starting it at zero would inject a millisecond of Yb charging the caller did not ask for --
    the same argument that put the coupled pool at its quasi-equilibrium in 2026-09-13."""
    amp = _all_on()
    tr = simulate_transient(amp, np.linspace(0.0, 2e-5, 5), n_nodes=41, nbar2_0=(0.2, 0.1))
    assert float(np.max(tr.meta["beta_yb_uncoupled"][0])) > 1e-3
    assert tr.meta["beta_yb_seed_residual"] < 1e-3


def test_a_triple_seed_is_refused_on_a_one_pool_amplifier():
    with pytest.raises(ValueError, match="must be the pair"):
        simulate_transient(ref_amp(), np.linspace(0.0, 1e-5, 3), n_nodes=21,
                           nbar2_0=(0.1, 0.2, 0.3))


def test_energy_terms_refuses_a_missing_or_spurious_uncoupled_pool():
    """Silently defaulting the uncoupled inversion would mis-state the stored energy and the Yb
    fluorescence, so it is refused in BOTH directions."""
    amp = _all_on()
    r = amp.solve(n_nodes=41)
    with pytest.raises(ValueError, match="UNCOUPLED ytterbium pool"):
        amp.energy_terms(r.power_W, r.nbar2_z, r.meta["beta_yb_coupled_z"])
    one = ref_amp()
    r1 = one.solve(n_nodes=41)
    with pytest.raises(ValueError, match="ONE ytterbium pool"):
        one.energy_terms(r1.power_W, r1.nbar2_z, r1.meta["beta_yb_z"], r1.meta["beta_yb_z"])


# ==================== gate 6: the ConcentrationModel on the co-doped class ===================

def test_concentration_reproduces_the_single_ion_pair_penalty_in_the_er_only_limit():
    """With the ytterbium switched off the co-doped class must charge pair-induced quenching and
    upconversion EXACTLY as FiberAmplifier does -- one convention, two classes. Both conventions
    are checked, and the Delevaque 2f == dark f identity with them."""
    fib = FiberSpec(1.4e-6, 0.24, 1.0e25, 6.0)
    pumps, sigs = [Pump(100e-3, 0.980e-6, "fwd")], [Signal(1e-6, 1.560e-6)]
    ase = AseBand(1.52e-6, 1.575e-6, 8)

    def pair(cm):
        g1 = FiberAmplifier(ER, fib, pumps, sigs, ase, concentration=cm).solve(n_nodes=81)
        g2 = ErYbAmplifier(ER, YB, fib, pumps, sigs, ase, n_yb_m3=1e10, k_tr_m3_s=0.0,
                           concentration=cm).solve(n_nodes=81)
        return float(g1.signal_gain_dB[0]), float(g2.signal_gain_dB[0])

    clean = pair(ConcentrationModel(c_up_m3_s=2e-24))
    dark = pair(ConcentrationModel(c_up_m3_s=2e-24, pair_fraction=0.025))
    delev = pair(ConcentrationModel(c_up_m3_s=2e-24, pair_fraction=0.05,
                                    pair_convention="delevaque"))
    assert clean[0] - dark[0] > 0.4, (clean, dark)       # premise: the pairs actually cost gain
    for a, b in (clean, dark, delev):
        assert abs(a - b) < 1e-4, (a, b)
    assert abs(dark[1] - delev[1]) < 1e-9                # 2f delevaque == f dark, on this class


def test_photodarkening_reads_the_population_weighted_yb_inversion():
    """The Yb gray loss is driven by bbar, not by the coupled pool alone. An uncoupled pool sits
    at a HIGHER inversion than a coupled one (nothing drains it), so reading b2c would understate
    the loss -- this gate holds the model to the weighted mean by construction and checks the
    loss actually bites."""
    pd = ConcentrationModel(pd_loss_per_m=0.4, pd_exponent=1.0)
    base = ref_amp(yb_coupled_fraction=0.5, k_tr_m3_s=3.0e-21)
    dark = ref_amp(yb_coupled_fraction=0.5, k_tr_m3_s=3.0e-21, concentration=pd,
                   upconversion_C_up=0.0)
    r0, r1 = base.solve(n_nodes=81), dark.solve(n_nodes=81)
    assert float(r0.signal_gain_dB[0]) - float(r1.signal_gain_dB[0]) > 0.2
    # the uncoupled pool really is the more inverted one, which is why the weighting matters
    assert np.max(r1.meta["beta_yb_uncoupled_z"]) > np.max(r1.meta["beta_yb_coupled_z"])


def test_the_c_up_entry_points_are_reconciled_to_one():
    """ONE entry point: a ConcentrationModel wins over a raw upconversion_C_up and says so."""
    cm = ConcentrationModel(c_up_m3_s=7e-24)
    a = ref_amp(concentration=cm, upconversion_C_up=0.0)
    assert a.upconversion_C_up == 7e-24
    with pytest.warns(UserWarning, match="OVERRIDDEN"):
        b = ref_amp(concentration=cm, upconversion_C_up=1e-24)
    assert b.upconversion_C_up == 7e-24
    # an identity model collapses to the None path, so it is byte-identical to no model at all
    c = ref_amp(concentration=ConcentrationModel())
    assert c.concentration is None and c._n_er_dark == 0.0


def test_the_clone_protocol_carries_every_new_optin():
    amp = _all_on()
    amp.set_temperature_profile(np.linspace(0.0, 3.0, 4), np.full(4, 340.0))
    for clone in (amp.with_signals(list(amp.signals)), amp.with_pumps(list(amp.pumps)),
                  amp.without_ase()):
        assert clone._fc == amp._fc and clone._k_tr2 == amp._k_tr2
        assert clone._w_mig == amp._w_mig and clone._two_pop == amp._two_pop
        assert clone.concentration is amp.concentration
        assert clone._n_er == amp._n_er and clone._n_er_dark == amp._n_er_dark
        assert clone._Tz is amp._Tz
        assert isinstance(clone, ErYbAmplifier)
    amp.clear_temperature_profile()


# ============================ gate 7: distributed temperature ===============================

def test_uniform_temperature_profile_equals_the_global_at_temperature_solve():
    """The per-z McCumber scaling of BOTH ions, each about ITS OWN zero line, must reproduce
    spectroscopy.at_temperature applied to both ions for a uniform profile. A shared eps would
    fail this by orders of magnitude in the pump band, where the Yb line sits."""
    T = 380.0
    amp = ref_amp(yb_coupled_fraction=0.9, k_tr_m3_s=3.0e-21, k_tr2_m3_s=2.0e-22)
    cold = float(amp.solve(n_nodes=101).signal_gain_dB[0])
    amp.set_temperature_profile(np.linspace(0.0, amp.fiber.length_m, 9), np.full(9, T),
                                T_ref_K=300.0)
    hot = float(amp.solve(n_nodes=101).signal_gain_dB[0])
    amp.clear_temperature_profile()
    glob = ErYbAmplifier(at_temperature(ER, T), at_temperature(YB, T), amp.fiber,
                         list(amp.pumps), list(amp.signals), amp.ase, n_yb_m3=4.0e26,
                         k_tr_m3_s=3.0e-21, yb_ase=amp.yb_ase, upconversion_C_up=1.1e-24,
                         yb_coupled_fraction=0.9, k_tr2_m3_s=2.0e-22)
    ref = float(glob.solve(n_nodes=101).signal_gain_dB[0])
    assert abs(hot - cold) > 0.01, (hot, cold)           # premise: 80 K actually does something
    assert abs(hot - ref) < 1e-6, (hot, ref)


def test_hot_cladding_pumped_feedback_converges_on_the_rate_balance_heat():
    """The self-consistent Q(z) -> T(z) loop on a co-doped amplifier, driven by the amplifier's
    own rate balance (quantum defect + transfer defect + K2 defect + both pools' fluorescence +
    background loss) rather than by a flux gradient. Residual is STATED, not assumed."""
    amp = _all_on()
    res, Tz, info = solve_with_thermal_feedback(
        amp, ThermalModel(h_conv_W_m2K=200.0, T_coolant_K=320.0), b_outer_m=62.5e-6, n_nodes=81)
    amp.clear_temperature_profile()
    assert info["heat_source"] == "rate_balance"
    assert info["converged_T"], info
    assert info["max_dT_K"] < 0.2, info["max_dT_K"]
    assert float(np.max(Tz)) > 320.0 and float(np.max(Tz)) < 600.0
    assert np.all(np.isfinite(res.signal_gain_dB))
    # a FiberAmplifier must keep the flux-gradient path -- no silent algorithm change
    plain = FiberAmplifier(ER, FiberSpec(1.4e-6, 0.24, 1.0e25, 6.0),
                           [Pump(100e-3, 0.980e-6, "fwd")], [Signal(1e-4, 1.560e-6)])
    assert not hasattr(plain, "_heat_profile_W_per_m")
    _r, _T, info2 = solve_with_thermal_feedback(plain, ThermalModel(), b_outer_m=62.5e-6,
                                                n_nodes=41, max_iter=2)
    assert info2["heat_source"] == "flux_gradient"


def test_the_march_refuses_a_temperature_profile_rather_than_ignoring_it():
    amp = ref_amp()
    amp.set_temperature_profile(np.linspace(0.0, 3.0, 4), np.full(4, 340.0))
    with pytest.raises(NotImplementedError, match="temperature profile"):
        simulate_transient(amp, np.linspace(0.0, 1e-5, 3), n_nodes=21)
    amp.clear_temperature_profile()


# ============================== gate 8: the calibration anchors =============================

def test_melkumov_aluminosilicate_branch_is_untouched():
    """The default call and the explicit aluminosilicate host must be the SAME ion, bit for bit,
    and spectroscopy.ytterbium() must still delegate to it."""
    lam = np.linspace(0.85e-6, 1.18e-6, 4001)
    a, b = ytterbium_melkumov(), ytterbium_melkumov("aluminosilicate")
    for ion in (b, ytterbium("aluminosilicate")):
        assert np.array_equal(a.sigma_a.sigma(lam), ion.sigma_a.sigma(lam))
        assert np.array_equal(a.sigma_e.sigma(lam), ion.sigma_e.sigma(lam))
        assert a.tau_s == ion.tau_s and a.zero_line_m == ion.zero_line_m
    # and the table itself reproduces the printed AS rows exactly, entry for entry
    nm = np.asarray(_YB_MELKUMOV_AS_NM, float) * 1e-9
    assert np.array_equal(a.sigma_a.sigma(nm),
                          np.asarray(_YB_MELKUMOV_AS_SIGMA_A_PM2, float) * 1e-24)
    assert np.array_equal(a.sigma_e.sigma(nm),
                          np.asarray(_YB_MELKUMOV_AS_SIGMA_E_PM2, float) * 1e-24)
    with pytest.raises(ValueError, match="host must be"):
        ytterbium_melkumov("tellurite")


def test_melkumov_phosphosilicate_anchors():
    """The P2O5 columns of the same table, at the three wavelengths that matter: the 976 nm pump,
    the 1030 nm parasitic peak and 1060 nm. Also the two structural facts a narrow P-host line
    forces -- the pump sits BELOW the 974 nm peak, and 976 nm is on the bleaching side."""
    p = ytterbium_melkumov("phosphosilicate")
    for nm, sa_t, se_t in ((976e-9, 1.01e-24, 1.18e-24), (1030e-9, 1.80e-26, 3.25e-25),
                           (1060e-9, 2.20e-27, 1.50e-25)):
        sa, se = float(p.sigma_a.sigma(nm)), float(p.sigma_e.sigma(nm))
        assert abs(sa / sa_t - 1.0) < 0.05, (nm, sa, sa_t)
        assert abs(se / se_t - 1.0) < 0.05, (nm, se, se_t)
    assert abs(p.tau_s - 1.45e-3) < 1e-9
    assert 972.0e-9 < p.zero_line_m < 973.5e-9              # the table's own crossing
    assert float(p.sigma_a.sigma(974e-9)) > float(p.sigma_a.sigma(976e-9))   # pump off-peak
    assert float(p.sigma_e.sigma(976e-9)) > float(p.sigma_a.sigma(976e-9))   # bleaching side
    # the 5.6 nm absorption FWHM of the P-host pump line, computed from the shipped table
    grid = np.arange(960.0, 990.0, 0.01) * 1e-9
    sa = p.sigma_a.sigma(grid)
    half = grid[sa >= 0.5 * sa.max()]
    fwhm_nm = 1e9 * (half[-1] - half[0])
    assert 5.3 < fwhm_nm < 5.9, fwhm_nm
    # ... against the aluminosilicate line, which is markedly broader
    sa_as = ytterbium_melkumov().sigma_a.sigma(grid)
    half_as = grid[sa_as >= 0.5 * sa_as.max()]
    assert 1e9 * (half_as[-1] - half_as[0]) > 7.0


def test_phosphosilicate_erbium_carries_the_measured_p_host_anchors():
    """erbium("phosphosilicate") stops being a label. Anchored at the three measured Le Gouet
    2019 wavelengths, red-shifted peak, shorter lifetime, McCumber-derived emission -- and the
    default aluminosilicate ion untouched."""
    p = erbium("phosphosilicate")
    legacy = erbium("phosphosilicate", p_host_spectra=False)
    lam = np.linspace(0.9e-6, 1.7e-6, 8001)
    assert np.array_equal(legacy.sigma_a.sigma(lam), ER.sigma_a.sigma(lam))
    assert np.array_equal(legacy.sigma_e.sigma(lam), ER.sigma_e.sigma(lam))
    assert legacy.tau_s == ER.tau_s and legacy.zero_line_m == ER.zero_line_m
    for nm, target in ((1535e-9, 5.948e-25), (1550e-9, 2.141e-25), (1560e-9, 1.313e-25)):
        assert abs(float(p.sigma_a.sigma(nm)) / target - 1.0) < 0.01, nm
    grid = np.arange(1500.0, 1600.0, 0.1) * 1e-9
    peak_nm = 1e9 * grid[int(np.argmax(p.sigma_a.sigma(grid)))]
    assert 1534.0 < peak_nm < 1536.0, peak_nm
    assert abs(p.tau_s - 9.0e-3) < 1e-9
    # the red wing: measured P-host sigma_e at 1560 nm is ~40% below the aluminosilicate fit
    assert 0.55 < float(p.sigma_e.sigma(1560e-9)) / float(ER.sigma_e.sigma(1560e-9)) < 0.70
    # emission is McCumber-derived from the SAME sigma_a about the fitted eps
    eps = p.eps_J
    from dynameta.constants import C_LIGHT, H_PLANCK, KB
    for nm in (1530e-9, 1545e-9, 1565e-9):
        want = float(p.sigma_a.sigma(nm)) * np.exp((eps - H_PLANCK * C_LIGHT / nm) / (KB * 300.0))
        assert abs(float(p.sigma_e.sigma(nm)) / want - 1.0) < 1e-12
    # the pump bands are DELIBERATELY the aluminosilicate ones (1-2% of an EYDFA's absorption)
    assert float(p.sigma_a.sigma(976e-9)) == float(ER.sigma_a.sigma(976e-9))


def test_the_reference_preset_builds_a_working_amplifier():
    """er_yb_phosphosilicate_reference returns constructor-ready kwargs for both
    parameterizations, and the two-population one really is two-population."""
    import dataclasses
    er, yb, fib, kw = er_yb_phosphosilicate_reference()
    assert fib.core_radius_m == 2.0e-6 and fib.na == 0.20 and fib.n_t_m3 == 4.0e25
    assert fib.length_m == 0.8 and fib.clad_radius_m == 62.5e-6
    assert kw["n_yb_m3"] == 4.0e26 and kw["yb_coupled_fraction"] == 0.9
    assert kw["k_tr_m3_s"] == 3.0e-21 and kw["k_tr2_m3_s"] == 2.0e-22
    # The preset's own 0.8 m is the length/gain crossing the reference document derives at
    # k_tr = 7.5e-23; at the pair-rate coefficient carried here the ytterbium drains ~7x faster,
    # so the pump-absorption length collapses to ~0.2 m at 1 W and the length has to be
    # re-optimised per coefficient. That is exactly what this gate demonstrates, and why the
    # preset is a starting point rather than a calibrated device.
    short = dataclasses.replace(fib, length_m=0.2)
    amp = ErYbAmplifier(er, yb, short, [Pump(1.0, 0.976e-6, "fwd")], [Signal(1e-4, 1.550e-6)],
                        AseBand(1.53e-6, 1.565e-6, 4), **kw)
    assert amp._two_pop and amp._n_er_dark > 0.0
    r = amp.solve(n_nodes=81)
    assert r.meta["converged"] and float(r.signal_gain_dB[0]) > 3.0
    assert r.meta["beta_yb_uncoupled_z"] is not None
    # at 976 nm the P-host ytterbium bleaches to its own clamp sigma_a/(sigma_a+sigma_e) = 0.461
    assert 0.40 < float(np.max(r.meta["beta_yb_uncoupled_z"])) < 0.47
    long_amp = ErYbAmplifier(er, yb, fib, list(amp.pumps), list(amp.signals), amp.ase, **kw)
    assert float(long_amp.solve(n_nodes=81).signal_gain_dB[0]) < float(r.signal_gain_dB[0])

    _e2, _y2, _f2, kw1 = er_yb_phosphosilicate_reference("single")
    assert kw1["yb_coupled_fraction"] == 1.0 and kw1["k_tr2_m3_s"] == 0.0
    assert kw1["k_tr_m3_s"] == 1.0e-21
    amp1 = ErYbAmplifier(er, yb, fib, list(amp.pumps), list(amp.signals), amp.ase, **kw1)
    assert amp1._two_pop is False
    with pytest.raises(ValueError, match="population must be"):
        er_yb_phosphosilicate_reference("three")


def test_parameter_validation_refuses_unphysical_two_population_inputs():
    for bad in (dict(yb_coupled_fraction=-0.1), dict(yb_coupled_fraction=1.5),
                dict(k_tr2_m3_s=-1e-22), dict(yb_migration_rate_per_s=-1.0)):
        with pytest.raises(ValueError):
            ref_amp(**bad)
