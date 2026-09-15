"""Discrimination-proven gates for the two 2026-09-15 Er:Yb upgrades: the EXPLICIT Er 4I11/2
level (`tau32_s`, with the explicit back-transfer and the optional Dong-2020 upconversion
routing) and the Yb EMISSION CROSS-SECTION SCALE (`yb_sigma_e_scale` and the calibration helper
`eryb_fit_yb_sigma_e_scale`).

Both are OPT-IN and both defaults are asserted bitwise rather than argued. The physics gates are
limit statements and monotonicity claims that a wrong sign, a mis-split channel mask or a
double-charged energy term breaks immediately, and they are bidirectional: the claimed successes
(the adiabatic limit is recovered, the closure total does not move, the fit recovers a synthetic
scale) are proven out, and the claimed failure modes (a long tau_32 costs gain through
back-transfer; an unscaled Yb emission over-predicts the 1-um ASE by orders of magnitude) are
MEASURED rather than asserted.

Audit trail: docs/audit/2026-09-15-eryb-4i11-2-and-yb-sigma-scale.md.
"""

import numpy as np
import pytest

from dynameta.optics.fiber_amp import (
    RATE_ARRHENIUS_30PCT_300_480K, RATE_ARRHENIUS_CHENG_2022, YB_STARK_976_CANAT_DUSSARDIER,
    AseBand, ConcentrationModel, FiberSpec, Pump, PumpSource, RateTemperatureLaw, Signal, erbium,
    simulate_transient, wall_plug_efficiency, ytterbium,
)
from dynameta.optics.fiber_amp.eryb import (
    ErYbAmplifier, YbSigmaEScaleFit, _eryb_1um_and_signal, eryb_fit_yb_sigma_e_scale,
)

ER = erbium("aluminosilicate")
YB = ytterbium("phosphosilicate")
_BG = 30e-3 * np.log(10.0) / 10.0                      # 30 dB/km, the research-grade figure
REF_PUMP_W = 1.0
TAU32_SEFLER = 7.0e-6                                  # Sefler 2004, the measured 4I11/2 lifetime
TAU32_SLOW = 5.0e-5                                    # the aluminosilicate / low-phosphorus end


def clad_amp(**kw):
    """THE reference co-doped fixture, shared with tests/test_fiber_eryb_physics.py: a 6 um-core
    CLADDING-pumped Er:Yb at the study's densities (N_Er 4e25, N_Yb 4e26), 1 W of 976 nm into a
    125 um cladding, 20 mW of 1550 nm seed, 3 m. It amplifies ~13.8 dB and sits in the regime the
    model is meant for rather than on a numerical edge."""
    fib = FiberSpec(3.0e-6, 0.20, 4.0e25, 3.0, clad_radius_m=62.5e-6, background_loss_per_m=_BG)
    base = dict(n_yb_m3=4.0e26, k_tr_m3_s=2.0e-22, yb_ase=AseBand(1.00e-6, 1.10e-6, 4),
                upconversion_C_up=1.1e-24)
    base.update(kw)
    return ErYbAmplifier(ER, YB, fib, [Pump(REF_PUMP_W, 0.976e-6, "fwd", cladding=True)],
                         [Signal(20e-3, 1.550e-6)], AseBand(1.53e-6, 1.565e-6, 6), **base)


def core_amp(**kw):
    """The CORE-pumped reference point of the 2026-09-13 closure note -- a 2.8 um core, NA 0.20,
    N_Er 2e25, N_Yb 2e26, k_tr 2e-22, 200 mW of 976 nm co-pump, 0.5 mW of 1550 nm, L 2 m. This is
    the point the brief names for the tau_32 sensitivity, and it is the hard case: the pump is
    absorbed in millimetres and the DIRECT erbium 976 nm absorption is not negligible there."""
    fib = FiberSpec(2.8e-6, 0.20, 2.0e25, 2.0)
    base = dict(n_yb_m3=2.0e26, k_tr_m3_s=2.0e-22, yb_ase=AseBand(1.00e-6, 1.10e-6, 8))
    base.update(kw)
    return ErYbAmplifier(ER, YB, fib, [Pump(0.2, 0.976e-6, "fwd")], [Signal(0.5e-3, 1.550e-6)],
                         AseBand(1.52e-6, 1.57e-6, 24), **base)


def _pump_drive(t):
    return np.array([REF_PUMP_W if t < 1e-3 else 0.4 * REF_PUMP_W])


# ============ gate 1: the defaults are the previous model, bitwise, in ONE process ============

def test_explicit_defaults_are_bitwise_identical_to_omitting_them():
    """The build-INDEPENDENT half of the no-change claim. Spelling every new parameter out at its
    default -- tau32_s=None, er_4i11_2_zero_line_m at its default, yb_sigma_e_scale=1.0 -- must
    give the SAME FLOATS as omitting them, to the last bit, across the steady solve, the energy
    split and the march. This runs both amplifiers in one process with one numpy and one scipy,
    so unlike a recorded pin it is exact on every runner, and it is the assertion that actually
    says "the new physics is off by default"."""
    bare = clad_amp()
    spelled = clad_amp(tau32_s=None, er_4i11_2_zero_line_m=977.0e-9, yb_sigma_e_scale=1.0)
    r0, r1 = bare.solve(n_nodes=81), spelled.solve(n_nodes=81)
    for name in ("power_W", "nbar2_z", "signal_gain_dB"):
        assert np.array_equal(getattr(r0, name), getattr(r1, name)), name
    for key in ("beta_yb_z", "eta_transfer", "yb_parasitic_gain_dB", "sigma_e_yb"):
        assert np.array_equal(np.asarray(r0.meta[key]), np.asarray(r1.meta[key])), key
    e0 = bare.energy_terms(r0.power_W, r0.nbar2_z, r0.meta["beta_yb_z"])
    e1 = spelled.energy_terms(r1.power_W, r1.nbar2_z, r1.meta["beta_yb_z"])
    assert set(e0) == set(e1)
    for key in e0:
        assert np.array_equal(np.asarray(e0[key]), np.asarray(e1[key])), key
    t0 = simulate_transient(bare, np.linspace(0.0, 2e-3, 9), n_nodes=21, pump_drive=_pump_drive)
    t1 = simulate_transient(spelled, np.linspace(0.0, 2e-3, 9), n_nodes=21,
                            pump_drive=_pump_drive)
    for name in ("nbar2_zt", "signal_gain_dB", "signal_out_W", "pump_out_W"):
        assert np.array_equal(getattr(t0, name), getattr(t1, name)), name
    assert np.array_equal(t0.meta["beta_yb"], t1.meta["beta_yb"])


def test_the_new_branches_are_structurally_unreachable_with_the_defaults():
    """The other build-independent half: with tau32_s unset and the scale at 1.0 the
    explicit-4I11/2 algebra, the level-2/3 channel masks and the emission scaling cannot be
    ENTERED at all, so no pinned number anywhere in the repo can drift for a reason this change
    introduced."""
    amp = clad_amp()
    assert amp._n3 is False and amp._tau32 is None and amp._cup_via_n3 is False
    assert amp._yb_se_scale == 1.0
    c = amp._coeffs(amp._plan())
    for key in ("w3", "flux_a_er3", "flux_e_er3", "g_e_er2", "g_e_er3", "s_er2", "s_er3"):
        assert key not in c, key
    # the phi refinement is still the adiabatic one, i.e. NOT forced to 1 by the n3 guard
    a_phi = clad_amp(k_back_m3_s=1e-22)
    assert float(a_phi._phi(np.array([0.3]))[0]) < 1.0
    r = amp.solve(n_nodes=41)
    assert r.meta["er_4i11_2_z"] is None and r.meta["tau32_s"] is None
    assert r.meta["back_transfer_ratio"] is None and r.meta["yb_sigma_e_scale"] == 1.0
    with pytest.raises(ValueError, match="eliminates the Er 4I11/2 level"):
        amp.energy_terms(r.power_W, r.nbar2_z, r.meta["beta_yb_z"], f3=np.zeros_like(r.z_m))
    tr = simulate_transient(amp, np.linspace(0.0, 1e-3, 5), n_nodes=11)
    assert tr.meta["er_4i11_2"] is None and tr.meta["tau32_s"] is None


# ================ gate 2: tau_32 -> 0 IS the adiabatic model, to 1e-6 dB =====================

def test_tau32_to_zero_reproduces_the_adiabatic_model():
    """The limit the whole upgrade has to satisfy. With tau_32 = 1 ps the 4I11/2 population is
    2.7e-9 of N_Er and the gain must be the adiabatic one; the gate is 1e-6 dB and the measured
    deviation is ~4e-9 dB.

    DISCRIMINATION (this is what makes the gate falsifiable rather than vacuous): the sweep below
    spans four decades of tau_32, so a mis-derived n3 balance -- a dropped ground-state factor,
    a transfer landing on the wrong level, a sign -- would show up as a residual PROPORTIONAL to
    tau_32, which is exactly what the 1 us and 7 us rows measure (1.7e-3 and 1.2e-2 dB)."""
    g0 = float(clad_amp().solve(n_nodes=161).signal_gain_dB[0])
    r_fast = clad_amp(tau32_s=1e-12).solve(n_nodes=161)
    assert r_fast.meta["converged"]
    assert float(r_fast.meta["er_4i11_2_z"].max()) < 1e-8
    assert abs(float(r_fast.signal_gain_dB[0]) - g0) < 1e-6
    # premise: the SAME machinery moves the gain measurably at a physical tau_32, so the
    # agreement above is a statement about the limit and not about a dead code path
    d1 = abs(float(clad_amp(tau32_s=1e-6).solve(n_nodes=161).signal_gain_dB[0]) - g0)
    d7 = abs(float(clad_amp(tau32_s=TAU32_SEFLER).solve(n_nodes=161).signal_gain_dB[0]) - g0)
    assert 1e-4 < d1 < d7 and d7 > 1e-2


def test_the_level_split_conserves_the_total_erbium_rates():
    """W12 + W13 must be the SAME number the adiabatic path calls R_a_Er, and W21 + W31 the same
    R_e_Er -- the split is computed as a difference precisely so no channel can be dropped or
    double-counted. Also pins WHICH channels are level-3: a 976 nm pump and a 1030 nm Yb ASE bin
    terminate on 4I11/2, a 1550 nm signal and a 1530 nm ASE bin on 4I13/2, with the crossover at
    1192 nm for the shipped ions."""
    amp = clad_amp(tau32_s=TAU32_SEFLER)
    pl = amp._plan()
    c = amp._coeffs(pl)
    lam = pl["lam"]
    w3 = c["w3"]
    assert np.all(w3[lam < 1.15e-6] == 1.0) and np.all(w3[lam > 1.25e-6] == 0.0)
    lam_x = 2.0 * 6.62607015e-34 * 299792458.0 / (float(ER.eps_J) + amp._eps3)
    assert 1.18e-6 < lam_x < 1.20e-6
    rng = np.random.default_rng(3)
    P = 10.0 ** rng.uniform(-6, 0, size=lam.size)
    ra, re, ra_y, re_y = amp._rates_profile(c, P[:, None])
    W12, W21, W13, W31, ry_a, ry_e = amp._rates_profile_n3(c, P[:, None])
    assert float(W12[0] + W13[0]) == float(ra[0]) or abs(W12[0] + W13[0] - ra[0]) <= 1e-15 * ra[0]
    assert abs(float(W21[0] + W31[0]) - float(re[0])) <= 1e-15 * abs(float(re[0]) + 1e-300)
    assert float(ry_a[0]) == float(ra_y[0]) and float(ry_e[0]) == float(re_y[0])
    assert float(W13[0]) > 0.0                      # the 976 nm pump really is on level 3


# =============== gate 3: the energy closure, and what the split does to it ===================

def test_the_closure_total_is_unchanged_in_the_adiabatic_limit_and_only_the_naming_moves():
    """The brief's closure requirement, and the sharpest single test of the energy bookkeeping.
    Making the 4I11/2 explicit re-labels heat: the transfer now lands on 4I11/2, so its defect is
    (eps_Yb - eps_3) and the remaining (eps_3 - eps_Er) is charged to the relaxation. The TOTAL
    dissipation must not move (measured 3.6e-10 relative), and the reallocation must be EXACT:
    whatever (D_tr + D_32) gains over the adiabatic D_tr, D_Er must lose, because the direct
    976 nm erbium absorption was charged at eps_Er before and is charged at eps_3 now."""
    a_ad = clad_amp()
    r_ad = a_ad.solve(n_nodes=161)
    e_ad = a_ad.energy_terms(r_ad.power_W, r_ad.nbar2_z, r_ad.meta["beta_yb_z"])
    a3 = clad_amp(tau32_s=1e-12)
    r3 = a3.solve(n_nodes=161)
    e3 = a3.energy_terms(r3.power_W, r3.nbar2_z, r3.meta["beta_yb_z"],
                         f3=r3.meta["er_4i11_2_z"])
    tot_ad, tot_3 = float(np.sum(e_ad["dissipation"])), float(np.sum(e3["dissipation"]))
    assert abs(tot_3 - tot_ad) <= 1e-8 * abs(tot_ad)
    assert abs(float(np.sum(e3["q_optical"])) - float(np.sum(e_ad["q_optical"]))) \
        <= 1e-8 * abs(float(np.sum(e_ad["q_optical"])))
    moved = (float(np.sum(e3["transfer_defect"])) + float(np.sum(e3["relaxation_32_defect"]))
             - float(np.sum(e_ad["transfer_defect"])))
    lost = float(np.sum(e_ad["er_dissipation"])) - float(np.sum(e3["er_dissipation"]))
    # The reallocation is exact only IN the limit: both sides carry an O(f3) error on the TOTAL
    # dissipation, which is the scale they live on (the moved piece is 0.7% of it), so the
    # tolerance is written against the total and not against the difference. Measured 1.5e-10 of
    # the total here, and 2.4e-9 when both splits are evaluated on the SAME profile below --
    # which is the comparison that removes the two independent LSODA paths and leaves only the
    # bookkeeping.
    assert abs(moved - lost) <= 1e-7 * abs(tot_ad)
    e_same = clad_amp().energy_terms(r3.power_W, r3.nbar2_z, r3.meta["beta_yb_z"])
    moved_s = (float(np.sum(e3["transfer_defect"])) + float(np.sum(e3["relaxation_32_defect"]))
               - float(np.sum(e_same["transfer_defect"])))
    lost_s = float(np.sum(e_same["er_dissipation"])) - float(np.sum(e3["er_dissipation"]))
    assert abs(moved_s - lost_s) <= 1e-7 * abs(tot_ad)
    assert abs(float(np.sum(e_same["dissipation"])) - float(np.sum(e3["dissipation"]))) \
        <= 1e-7 * abs(tot_ad)
    # premise: the reallocation is a real, sized effect (1.2% of the transfer defect here), so
    # the equality above is not two zeros agreeing
    assert abs(moved) > 1e-2 * abs(float(np.sum(e_ad["transfer_defect"])))
    # and the relaxation is now the DOMINANT named term, the transfer defect the small one:
    # eps_Yb - eps_3 is a 2.6 meV near-resonance, eps_3 - eps_Er the 0.46 eV multiphonon drop
    assert float(np.sum(e3["relaxation_32_defect"])) > \
        50.0 * float(np.sum(e3["transfer_defect"]))


def test_closure_holds_per_z_and_in_the_wall_plug_budget_with_the_level_on():
    """The identity q_opt = dU/dt + D_loss + D_Er + D_Yb + D_tr + D_K2 + D_32 + D_bk, with every
    new mechanism on (explicit level, explicit back-transfer, routed upconversion, two Yb pools,
    K2, migration, concentration). Per-z it is an algebraic rearrangement of one rate balance, so
    anything above round-off is an implementation defect; the INDEPENDENT check is the wall-plug
    residual, which comes from -INT dF/dz of the amplifier's own RHS."""
    conc = ConcentrationModel(c_up_m3_s=1.1e-24, pair_fraction=0.03, pair_convention="delevaque",
                              pd_loss_per_m=0.02, pd_exponent=1.0)
    a = clad_amp(tau32_s=TAU32_SEFLER, k_back_m3_s=2.0e-22, upconversion_via_4i11_2=True,
                 yb_coupled_fraction=0.85, k_tr_m3_s=3.0e-21, k_tr2_m3_s=2.0e-22,
                 yb_migration_rate_per_s=5.0e3, concentration=conc, upconversion_C_up=0.0)
    resid = []
    for n in (321, 641):
        r = a.solve(n_nodes=n)
        assert r.meta["converged"]
        et = a.energy_terms(r.power_W, r.nbar2_z, r.meta["beta_yb_coupled_z"],
                            r.meta["beta_yb_uncoupled_z"], f3=r.meta["er_4i11_2_z"])
        rel = np.max(np.abs(et["q_optical"] - et["d_stored_dt"] - et["dissipation"])
                     / np.maximum(np.abs(et["q_optical"]), 1e-30))
        assert float(rel) < 1e-11, float(rel)
        b = wall_plug_efficiency(a, r, PumpSource(wallplug_efficiency=1.0))
        resid.append(abs(float(b.energy_balance_residual_W)) / REF_PUMP_W)
    # the residual is MESH-limited (the trapezoid on the rate integral, O(dz^2)), and at the
    # fast-transfer coefficient this configuration carries the pump is gone in centimetres:
    # 5.4e-5 / 1.4e-5 / 3.4e-6 at 161 / 321 / 641, i.e. 4x per doubling exactly
    assert resid[1] < 1e-5 and resid[0] / resid[1] > 3.0


def test_energy_terms_and_stored_energy_refuse_a_missing_or_spurious_f3():
    """Guessing f3 would silently mis-state the stored energy and the ground fraction, so both
    entry points refuse it in BOTH directions -- missing on an explicit-level amplifier, and
    supplied on an adiabatic one."""
    a = clad_amp(tau32_s=TAU32_SEFLER)
    r = a.solve(n_nodes=41)
    with pytest.raises(ValueError, match="EXPLICIT Er 4I11/2"):
        a.energy_terms(r.power_W, r.nbar2_z, r.meta["beta_yb_z"])
    with pytest.raises(ValueError, match="EXPLICIT Er 4I11/2"):
        a.stored_energy_J(r.nbar2_z, r.meta["beta_yb_z"], r.z_m)
    u_no3 = clad_amp().stored_energy_J(r.nbar2_z, r.meta["beta_yb_z"], r.z_m)
    u_3 = a.stored_energy_J(r.nbar2_z, r.meta["beta_yb_z"], r.z_m, f3=r.meta["er_4i11_2_z"])
    assert u_3 > u_no3                                  # the 4I11/2 stores a 977 nm quantum each


# ============ gate 4: the physics -- the bottleneck, the back-transfer, the routing ==========

def test_tau32_on_the_core_pumped_reference_point():
    """The brief's sensitivity point. At Sefler's measured tau_32 = 7 us the core-pumped
    reference point loses 0.018 dB of gain and the 4I11/2 population stays under 1% of N_Er --
    i.e. the adiabatic model is a good approximation THERE, which is the honest reading. The
    sweep also has to be monotone in the right directions: a longer tau_32 holds more population
    in 4I11/2 (less erbium ground available to accept transfer) and therefore less gain."""
    # 161 nodes: MEASURED 2026-09-14 against 241 and 321. The quantity the gate bounds,
    # g0 - g(7 us), is 0.018382 / 0.018371 / 0.018368 dB at 161 / 241 / 321 -- mesh-converged to
    # 1.4e-5 dB, three orders inside the 0.03 bound -- and the three f3 maxima agree to every
    # printed digit (4.2e-4 / 2.90e-3 / 2.010e-2), so both 1% thresholds keep their margin.
    g0 = float(core_amp().solve(n_nodes=161).signal_gain_dB[0])
    out = []
    for t32 in (1e-6, TAU32_SEFLER, TAU32_SLOW):
        r = core_amp(tau32_s=t32).solve(n_nodes=161)
        assert r.meta["converged"]
        out.append((float(r.signal_gain_dB[0]), float(r.meta["er_4i11_2_z"].max())))
    gains = [g for g, _ in out]
    f3s = [f for _, f in out]
    assert gains[0] > gains[1] > gains[2]               # longer tau_32 -> less gain
    assert f3s[0] < f3s[1] < f3s[2]                     # longer tau_32 -> more 4I11/2
    assert 0.0 < g0 - gains[1] < 0.03                   # 7 us costs 0.018 dB here
    assert f3s[1] < 0.01                                # n3/N_Er stays under 1% at 7 us
    assert f3s[2] > 0.01                                # and passes it at 50 us


def test_back_transfer_loss_is_monotone_in_tau32_at_fixed_k_back():
    """The loss channel the explicit level exists to expose. Back-transfer is k_back n3 n5c, and
    n3 grows in proportion to tau_32, so at a FIXED k_back the returned fraction and the gain
    penalty must both be monotone in tau_32 -- and at the aluminosilicate end (50 us) the penalty
    has to be measurable, not a rounding effect. k_back = k_tr is Dong 2020's own assumption
    (C36 = C63)."""
    k = 2.0e-22
    ratios, gains = [], []
    # 81 nodes: MEASURED 2026-09-14 against 101 and 121 (the previous setting, itself measured
    # against 161). The returned fractions are ratios of z-integrals, not endpoint-sensitive
    # gains, and reproduce to SIX decimals across the three meshes (0.070714 / 0.344765-0.344763
    # / 0.784101-0.784099); the gains move by at most 4.9e-4 dB, against monotone steps of 0.29
    # and 1.76 dB and a back-transfer loss of 2.002 dB gated at > 1.0. This sweep was the most
    # expensive in the file -- a k_back = k_tr back-transfer couples the two ions both ways at
    # once -- which is why it is the one that pays for the finer mesh and does not need it.
    for t32 in (1e-6, TAU32_SEFLER, TAU32_SLOW):
        r = clad_amp(tau32_s=t32, k_back_m3_s=k).solve(n_nodes=81)
        assert r.meta["converged"]
        ratios.append(float(r.meta["back_transfer_ratio"]))
        gains.append(float(r.signal_gain_dB[0]))
    assert ratios[0] < ratios[1] < ratios[2] and ratios[0] > 0.0
    assert gains[0] > gains[1] > gains[2]
    g_no_back = float(clad_amp(tau32_s=TAU32_SLOW).solve(n_nodes=81).signal_gain_dB[0])
    assert g_no_back - gains[2] > 1.0                  # ~2.0 dB of back-transfer loss at 50 us
    # k_back = 0 must leave NOTHING behind: the ratio is exactly zero, not merely small
    assert clad_amp(tau32_s=TAU32_SEFLER).solve(n_nodes=41).meta["back_transfer_ratio"] == 0.0


def test_the_upconversion_routing_is_a_no_op_in_the_adiabatic_limit_and_refused_without_it():
    """Dong 2020 Eq. (2) routes cooperative upconversion through 4I11/2: -2 C_up n2^2 out of
    4I13/2 and +C_up n2^2 into 4I11/2. Once n3 relaxes instantly that is the single -C_up N_Er
    f2^2 this module has always used, so at tau_32 = 1 ps the two routings must agree -- and at a
    physical tau_32 they must NOT, or the option would be decorative. Because the routings are
    identical in the adiabatic limit, the flag is refused outright when tau32_s is None rather
    than silently accepted."""
    with pytest.raises(ValueError, match="upconversion_via_4i11_2"):
        clad_amp(upconversion_via_4i11_2=True)
    kw = dict(upconversion_C_up=3.0e-24, k_tr_m3_s=3.0e-21)
    g_a = float(clad_amp(tau32_s=1e-12, **kw).solve(n_nodes=81).signal_gain_dB[0])
    g_b = float(clad_amp(tau32_s=1e-12, upconversion_via_4i11_2=True,
                         **kw).solve(n_nodes=81).signal_gain_dB[0])
    assert abs(g_a - g_b) < 1e-6
    g_c = float(clad_amp(tau32_s=TAU32_SLOW, **kw).solve(n_nodes=81).signal_gain_dB[0])
    g_d = float(clad_amp(tau32_s=TAU32_SLOW, upconversion_via_4i11_2=True,
                         **kw).solve(n_nodes=81).signal_gain_dB[0])
    assert abs(g_c - g_d) > 1e-4


# ================= gate 5: the four-reservoir march (stability and fixed point) ==============

def test_the_steady_state_is_an_exact_fixed_point_of_the_four_state_rhs():
    """`_solve_f3fb` returns the state at which all four balances vanish, so the march built on
    `_fb_rhs_n3` has the STEADY SOLVE's own fixed point rather than a nearby one. Measured as the
    RHS evaluated on the solved profile, normalised by the largest rate in the same balance."""
    a = clad_amp(tau32_s=TAU32_SEFLER, k_back_m3_s=2.0e-22)
    r = a.solve(n_nodes=81)
    c = a._coeffs(a._plan())
    rt = a._rates_profile_n3(c, r.power_W)
    f3 = np.asarray(r.meta["er_4i11_2_z"], float)
    f2 = np.asarray(r.nbar2_z, float)
    b2 = np.asarray(r.meta["beta_yb_z"], float)
    df2, df3, dbc, _dbn = a._fb_rhs_n3(rt[0], rt[1], rt[2], rt[3], rt[4], rt[5], f2, f3, b2, b2)
    scale2 = rt[0] + rt[1] + 1.0 / a._tau_er
    scale3 = rt[2] + rt[3] + 1.0 / a._tau32
    scaleb = rt[4] + rt[5] + 1.0 / a._tau_yb
    assert float(np.max(np.abs(df2) / scale2)) < 1e-12
    assert float(np.max(np.abs(df3) / scale3)) < 1e-12
    assert float(np.max(np.abs(dbc) / scaleb)) < 1e-12


def test_the_four_state_jacobian_keeps_every_eigenvalue_in_the_left_half_plane():
    """What the exponential Rosenbrock step actually needs. No determinant theorem is claimed for
    the 4x4 -- J[0][1] = A_32 - W12 is positive and J[2][0] changes sign with K2 -- so the
    property is GATED over 300 random operating points spanning five decades of rate. The
    eigenvalues are NOT all real here (unlike the 2x2 and 3x3 cases), which is exactly why phi_1
    is evaluated on the matrix by scaling-and-squaring rather than through an eigen-decomposition
    whose divided difference would have to handle a complex pair."""
    rng = np.random.default_rng(7)
    a = clad_amp(tau32_s=TAU32_SEFLER, k_back_m3_s=2e-22, yb_coupled_fraction=0.6,
                 k_tr2_m3_s=4e-22, yb_migration_rate_per_s=1e4)
    worst, worst_im = -np.inf, 0.0
    for _ in range(300):
        W12, W21, W13, W31 = 10.0 ** rng.uniform(0, 5, 4)
        ray, rey = 10.0 ** rng.uniform(1, 6, 2)
        f2, f3 = rng.uniform(0, 0.7), rng.uniform(0, 0.2)
        b, bn = rng.uniform(0, 1), rng.uniform(0, 1)
        J = a._fb_jacobian_n3(W12, W21, W13, W31, ray, rey, f2, f3, b, bn)
        ev = np.linalg.eigvals(np.array([[float(x) for x in row] for row in J]))
        worst = max(worst, float(np.max(ev.real)))
        worst_im = max(worst_im, float(np.max(np.abs(ev.imag))))
    assert worst < 0.0, worst
    assert worst_im > 0.0                               # complex pairs DO occur: see the docstring


def test_the_march_settles_onto_solve_with_four_reservoirs_and_survives_a_stiff_dt_sweep():
    """(i) A constant-drive march started from solve() sits still, and one started 88 dB away
    settles back onto it. (ii) The step stays finite with the populations inside the simplex
    f2, f3 >= 0, f2 + f3 <= 1 for dt from 1e-8 to 1e-3 s -- four decades either side of
    tau_32 = 7 us, which is the whole point of integrating the stiffest row exponentially."""
    a = clad_amp(tau32_s=TAU32_SEFLER)
    rs = float(a.solve(n_nodes=161).signal_gain_dB[0])
    tr = simulate_transient(a, np.linspace(0.0, 4e-2, 41), n_nodes=161)
    assert tr.meta["quasi_static_valid"]
    assert float(np.max(np.abs(tr.signal_gain_dB[:, 0] - rs))) < 5e-3
    assert tr.meta["er_4i11_2"].shape == tr.nbar2_zt.shape
    tr2 = simulate_transient(a, np.logspace(-8, -1, 60), n_nodes=161, nbar2_0=(0.02, 0.0, 0.02))
    assert float(tr2.signal_gain_dB[0, 0]) < 0.0        # starts far below the steady state
    assert abs(float(tr2.signal_gain_dB[-1, 0]) - rs) < 5e-3
    for dt in (1e-8, 1e-6, 1e-4, 1e-3):
        tt = np.arange(0.0, 21.0 * dt, dt)
        trx = simulate_transient(a, tt, n_nodes=41,
                                 pump_drive=lambda t, d=dt: np.array([1.0 if t < 5 * d else 0.3]))
        f3t = np.asarray(trx.meta["er_4i11_2"], float)
        assert np.all(np.isfinite(trx.nbar2_zt)) and np.all(np.isfinite(f3t))
        assert np.all(f3t >= 0.0) and np.all(trx.nbar2_zt + f3t <= 1.0 + 1e-12)
        assert np.all(np.isfinite(trx.signal_out_W))


def test_the_march_seed_tuple_reads_four_reservoirs_and_refuses_the_wrong_shape():
    """With an explicit 4I11/2 the seed tuple means (f2, b2), (f2, f3, b2) or
    (f2, f3, b2c, b2nc) -- a deliberate re-reading of the 3-tuple, resolved by the AMPLIFIER and
    never by sniffing shapes, so the refusals are part of the contract."""
    a = clad_amp(tau32_s=TAU32_SEFLER)
    seed = (0.02, 1e-3, 0.05)          # low inversion: the frozen-step ASE stays a perturbation
    tr = simulate_transient(a, np.linspace(0.0, 1e-4, 5), n_nodes=21, nbar2_0=seed)
    assert tr.meta["quasi_static_valid"]
    assert abs(float(tr.meta["er_4i11_2"][0].max()) - seed[1]) < 1e-12
    assert abs(float(tr.nbar2_zt[0].max()) - seed[0]) < 1e-12
    with pytest.raises(ValueError, match="4-tuple seed"):
        simulate_transient(a, np.linspace(0.0, 1e-4, 5), n_nodes=21,
                           nbar2_0=(0.02, 1e-3, 0.05, 0.05))
    with pytest.raises(ValueError, match="explicit 4I11/2"):
        simulate_transient(a, np.linspace(0.0, 1e-4, 5), n_nodes=21,
                           nbar2_0=(0.02, 1e-3, 0.05, 0.05, 0.05))
    # a 2-tuple still seeds f3 from the closed-form quasi-equilibrium rather than from zero
    tr2 = simulate_transient(a, np.linspace(0.0, 1e-4, 5), n_nodes=21, nbar2_0=(0.02, 0.05))
    assert float(tr2.meta["er_4i11_2"][0].max()) > 0.0


# ==================== gate 6: the Yb emission scale and its calibration ======================

def test_yb_sigma_e_scale_of_one_is_bitwise_identical_and_the_scale_reaches_every_consumer():
    """Default identity first, then the claim that ONE multiplication in `_plan` really does
    scale the whole model: the plan spectrum, the modal gain, the ASE source and the parasitic
    diagnostic all move together, while sigma_a_Yb does not move at all."""
    a0, a1 = clad_amp(), clad_amp(yb_sigma_e_scale=1.0)
    r0, r1 = a0.solve(n_nodes=81), a1.solve(n_nodes=81)
    assert np.array_equal(r0.power_W, r1.power_W)
    assert r0.meta["yb_parasitic_gain_dB"] == r1.meta["yb_parasitic_gain_dB"]
    s = 0.4
    a2 = clad_amp(yb_sigma_e_scale=s)
    p0, p2 = a0._plan(), a2._plan()
    assert np.allclose(p2["se_yb"], s * p0["se_yb"], rtol=0, atol=0)
    assert np.array_equal(p2["sa_yb"], p0["sa_yb"])         # absorption is NOT touched
    assert np.array_equal(p2["se_er"], p0["se_er"])         # nor is the erbium
    c0, c2 = a0._coeffs(p0), a2._coeffs(p2)
    # the derived coefficients are the SAME product chain with the scale inside it, so they
    # agree to the association of a float multiply, not bit for bit
    assert np.allclose(c2["g_e_yb"], s * c0["g_e_yb"], rtol=1e-15, atol=0)
    assert np.allclose(c2["s_yb"], s * c0["s_yb"], rtol=1e-15, atol=0)
    assert np.allclose(c2["flux_e_yb"], s * c0["flux_e_yb"], rtol=1e-15, atol=0)
    assert np.array_equal(c2["g_a_yb"], c0["g_a_yb"])
    r2 = a2.solve(n_nodes=81)
    assert r2.meta["yb_parasitic_gain_dB"] < r0.meta["yb_parasitic_gain_dB"]
    assert r2.meta["yb_sigma_e_scale"] == s
    # and it survives the re-seed protocol, which is what the fit walks its bracket with
    assert a0.with_yb_sigma_e_scale(s)._yb_se_scale == s
    assert a2.with_signals(list(a2.signals))._yb_se_scale == s


def test_the_scale_moves_the_1um_band_by_orders_and_the_signal_by_hundredths_of_a_dB():
    """Morasse's observation, as a gate: the 1-um ASE is exponential in the Yb gain integral and
    the C-band signal is not, so one scale can move the first by orders of magnitude while the
    second barely notices. That asymmetry is the entire justification for calibrating sigma_e_Yb
    against a 1-um measurement, and if it failed the fit would be refitting the amplifier."""
    ase, sig = [], []
    for s in (0.3, 0.6, 1.0):
        p_ase, p_sig, conv = _eryb_1um_and_signal(clad_amp(), s, "bwd", 81, {})
        assert conv
        ase.append(p_ase)
        sig.append(p_sig)
    assert ase[0] < ase[1] < ase[2]                                  # monotone in the scale
    assert ase[2] / ase[0] > 10.0                                    # measured 14.6x
    assert abs(10.0 * np.log10(sig[2] / sig[0])) < 0.1               # measured 0.032 dB


def test_the_fit_recovers_a_synthetic_scale():
    """The falsifiable version of the calibration claim: generate the 1-um ASE of an amplifier
    whose scale is KNOWN to be 0.4, hand that number back to the fit as if it were a
    measurement, and require 0.40 +- 0.02. This tests the root find, the bracket expansion, the
    band selection and the re-seed protocol at once."""
    truth = 0.4
    target, sig_true, conv = _eryb_1um_and_signal(clad_amp(), truth, "bwd", 81, {})
    assert conv and target > 0.0
    fit = eryb_fit_yb_sigma_e_scale(clad_amp(), target, "bwd", n_nodes=81)
    assert isinstance(fit, YbSigmaEScaleFit) and fit.converged
    assert abs(fit.scale - truth) < 0.02, fit
    assert abs(fit.ase_1um_W / target - 1.0) < 1e-2
    assert abs(fit.signal_out_W / sig_true - 1.0) < 1e-4
    assert fit.ase_1um_unit_scale_W > 5.0 * target       # the over-prediction it removed
    assert fit.solves < 20 and "scale=" in str(fit)
    # the forward direction is a different measurement and must be accepted as one
    t_f, _s_f, _c = _eryb_1um_and_signal(clad_amp(), truth, "fwd", 81, {})
    fit_f = eryb_fit_yb_sigma_e_scale(clad_amp(), t_f, "fwd", n_nodes=81)
    assert abs(fit_f.scale - truth) < 0.02, fit_f


def test_the_fit_refuses_what_it_cannot_do():
    """Every refusal is a case where returning a number would be worse than raising: no Yb band
    to read, a direction that is not a direction, overlapping bands that make "the 1-um output"
    ill-defined, and a target no scale in the search range can reach."""
    with pytest.raises(ValueError, match="no yb_ase band"):
        eryb_fit_yb_sigma_e_scale(clad_amp(yb_ase=None), 1e-5, "bwd", n_nodes=41)
    with pytest.raises(ValueError, match="direction"):
        eryb_fit_yb_sigma_e_scale(clad_amp(), 1e-5, "backward", n_nodes=41)
    with pytest.raises(ValueError, match="must be > 0"):
        eryb_fit_yb_sigma_e_scale(clad_amp(), 0.0, "bwd", n_nodes=41)
    overlapping = clad_amp(yb_ase=AseBand(1.40e-6, 1.60e-6, 4))
    with pytest.raises(ValueError, match="overlaps"):
        eryb_fit_yb_sigma_e_scale(overlapping, 1e-5, "bwd", n_nodes=41)
    with pytest.raises(RuntimeError, match="no Yb emission scale"):
        eryb_fit_yb_sigma_e_scale(clad_amp(), 1.0, "bwd", n_nodes=41, scale_max=1.5,
                                  bracket=(0.5, 1.0))


def test_the_two_upgrades_compose():
    """They are independent options and have to remain so: an amplifier carrying BOTH an explicit
    4I11/2 and a scaled Yb emission must solve, close its energy balance and march."""
    a = clad_amp(tau32_s=TAU32_SEFLER, k_back_m3_s=2e-22, yb_sigma_e_scale=0.4)
    r = a.solve(n_nodes=321)      # mesh-limited: 1.1e-5 at 161, 2.9e-6 here
    assert r.meta["converged"]
    assert r.meta["tau32_s"] == TAU32_SEFLER and r.meta["yb_sigma_e_scale"] == 0.4
    et = a.energy_terms(r.power_W, r.nbar2_z, r.meta["beta_yb_z"], f3=r.meta["er_4i11_2_z"])
    rel = np.max(np.abs(et["q_optical"] - et["d_stored_dt"] - et["dissipation"])
                 / np.maximum(np.abs(et["q_optical"]), 1e-30))
    assert float(rel) < 1e-11
    b = wall_plug_efficiency(a, r, PumpSource(wallplug_efficiency=1.0))
    assert abs(float(b.energy_balance_residual_W)) / REF_PUMP_W < 1e-5
    tr = simulate_transient(a, np.linspace(0.0, 2e-3, 9), n_nodes=41, pump_drive=_pump_drive)
    assert tr.meta["quasi_static_valid"] and np.all(np.isfinite(tr.signal_gain_dB))
    fr = a.with_signals(list(a.signals))
    assert fr._tau32 == TAU32_SEFLER and fr._yb_se_scale == 0.4       # both ride the clone


# ============ gate 7: the two 2026-09-15 builds are orthogonal (the merge gate) ==============

def test_this_build_and_the_migration_thermal_build_are_orthogonal():
    """THE MERGE GATE. This branch and `feat/eryb-migration-thermal-fit` (v0.11.3) both rewrote
    the same five methods -- `_mcc_matrices`, `_dP`, `_fb_profile`, `_rates_profile`,
    `energy_terms` -- one to carry an explicit 4I11/2 and a scaled Yb emission, the other to
    carry an Arrhenius rate law and a Stark-band cross-section scale. After the merge each pair
    must still be INERT when the other pair is on, and all four must compose.

    The four assertions, in the order that localises a mistake fastest:
      (a) with a temperature profile set and BOTH of the v0.11.3 options OFF, this build's
          numbers are what they were before the merge -- so their 4-slot temperature bundle did
          not change the isothermal-rate path;
      (b) with BOTH of this build's options OFF, the v0.11.3 options move the gain by a sized
          amount -- the premise that makes (c) meaningful;
      (c) turning this build's options on TOP of theirs changes the gain again, and the energy
          identity still closes to round-off with all four active;
      (d) the Arrhenius law actually reaches the explicit-4I11/2 algebra: at the same profile,
          scaling k_tr with temperature must move the 4I11/2 population, which it can only do if
          `rs` is threaded into `_n3_coeffs`."""
    z_prof = np.linspace(0.0, 3.0, 21)
    T_prof = 300.0 + 60.0 * np.exp(-z_prof / 0.5)

    def hot(**kw):
        a = clad_amp(**kw)
        a.set_temperature_profile(z_prof, T_prof, T_ref_K=300.0)
        return a

    # (a) their bundle is inert on this build's path when their options are off
    g_iso = float(clad_amp(tau32_s=TAU32_SEFLER).solve(n_nodes=81).signal_gain_dB[0])
    r_t = hot(tau32_s=TAU32_SEFLER).solve(n_nodes=81)
    g_t = float(r_t.signal_gain_dB[0])
    assert abs(g_t - g_iso) > 1e-3                      # the profile itself does something
    r_t2 = hot(tau32_s=TAU32_SEFLER, rate_temperature=RateTemperatureLaw(),
               yb_stark_thermal=None).solve(n_nodes=81)
    assert float(r_t2.signal_gain_dB[0]) == g_t         # an identity law is exactly the None path

    # (b) their options move the gain with MY options off -- the premise for (c)
    g_plain = float(hot().solve(n_nodes=81).signal_gain_dB[0])
    g_theirs = float(hot(rate_temperature=RATE_ARRHENIUS_CHENG_2022,
                         yb_stark_thermal=YB_STARK_976_CANAT_DUSSARDIER
                         ).solve(n_nodes=81).signal_gain_dB[0])
    assert abs(g_theirs - g_plain) > 1e-3, (g_theirs, g_plain)

    # (c) all four on: it solves, it moves, and the closure still holds to round-off
    a_all = hot(tau32_s=TAU32_SEFLER, k_back_m3_s=2e-22, yb_sigma_e_scale=0.4,
                rate_temperature=RATE_ARRHENIUS_CHENG_2022,
                yb_stark_thermal=YB_STARK_976_CANAT_DUSSARDIER)
    # 81 nodes, the SAME mesh as g_theirs above, so (c)'s gain comparison is no longer made
    # across two meshes. The closure here is a per-z round-off identity, not a mesh-convergence
    # statement: MEASURED 3.125e-14 / 3.775e-14 / 3.723e-14 at 81 / 121 / 161, i.e. flat in dz.
    r_all = a_all.solve(n_nodes=81)
    assert r_all.meta["converged"]
    assert abs(float(r_all.signal_gain_dB[0]) - g_theirs) > 1e-3
    et = a_all.energy_terms(r_all.power_W, r_all.nbar2_z, r_all.meta["beta_yb_z"],
                            z_m=r_all.z_m, f3=r_all.meta["er_4i11_2_z"])
    rel = np.max(np.abs(et["q_optical"] - et["d_stored_dt"] - et["dissipation"])
                 / np.maximum(np.abs(et["q_optical"]), 1e-30))
    assert float(rel) < 1e-11, float(rel)

    # (d) the Arrhenius scale reaches the 4I11/2 algebra itself, not just the two-level one.
    # r_t IS the no-law solve -- same constructor, same nodes -- so (a)'s result is reused here
    # rather than recomputed (2026-09-14: one 81-node hot solve, ~12 s of the py3.12 leg).
    r_no_law = r_t
    r_law = hot(tau32_s=TAU32_SEFLER,
                rate_temperature=RATE_ARRHENIUS_30PCT_300_480K).solve(n_nodes=81)
    assert not np.array_equal(r_no_law.meta["er_4i11_2_z"], r_law.meta["er_4i11_2_z"])
    assert float(np.max(np.abs(r_law.meta["er_4i11_2_z"]
                               - r_no_law.meta["er_4i11_2_z"]))) > 1e-6
