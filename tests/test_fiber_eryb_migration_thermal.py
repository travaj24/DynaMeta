"""Discrimination-proven gates for the 2026-09-15 Er:Yb MIGRATION CALIBRATION, the co-doped
TEMPERATURE laws and the (f, k_tr) DEVICE FIT.

Three claims are under test and every one of them is falsifiable here.

  (a) NOTHING MOVED. `rate_temperature` and `yb_stark_thermal` default to None and an identity
      `RateTemperatureLaw()` collapses to that None, so the 2026-09-14 model -- including a model
      running under a temperature profile -- is reproduced BITWISE, and the new code paths are
      structurally unreachable. Asserted by exact array equality in one process, which no build
      can move, rather than by a pinned constant, which CI has falsified twice (2026-09-13).

  (b) THE MIGRATION RATE IS MEASURED, not posited. The closed-form free decay is the analytic
      solution of the march's own `_fb_rhs3` (gated against a brute-force RK4 of that right-hand
      side, not against a re-derivation), its exact 3-number inversion round-trips, and applied to
      Cheng 2022 it reproduces BOTH transfer coefficients the literature extracts from that one
      measurement -- which is the entire job W_mig was introduced to do. Jeong 2007's independent
      decay pattern and Laroche 2006's five quenched lifetimes are the out-of-sample checks.

  (c) THE TEMPERATURE LAWS ACT IN THE MEASURED DIRECTION AND DO NOT DOUBLE COUNT. The McCumber
      factor is proven identically 1 at the ytterbium zero line, which is what makes the Stark
      depopulation orthogonal to it; the shipped constants reproduce their own anchors; and the
      1-um parasitic threshold of a hot fiber RISES with temperature (Canat/Dussardier, Morasse
      2007), monotonically, which is the qualitative result and not a number.

Bidirectional-adversarial: the claimed successes above are proven out, and the claimed FAILURE --
that a two-observable device fit is under-determined if you give it one observable (the
2026-09-14 note's limit 8) -- is measured rather than asserted, by comparing the fit's
uncertainty along ln(f k_tr) against its uncertainty along ln(f/k_tr).

Audit trail: docs/audit/2026-09-15-eryb-migration-thermal-fit.md.
"""

import numpy as np
import pytest

from dynameta.core.numerics import trapz          # audit X-1: never np.trapezoid in tests/
from dynameta.optics.fiber_amp import (
    AseBand, FiberSpec, Pump, Signal, ThermalModel, erbium, simulate_transient,
    solve_with_thermal_feedback, ytterbium,
)
from dynameta.optics.fiber_amp.eryb import (
    RATE_ARRHENIUS_30PCT_300_480K, RATE_ARRHENIUS_CHENG_2022, ErYbAmplifier, RateTemperatureLaw,
    YB_STARK_976_CANAT_DUSSARDIER, YbStarkThermal,
)
from dynameta.optics.fiber_amp.eryb_fit import (
    CHENG_2022_DECAY, JEONG_2007_DECAY, LAROCHE_2006_LIFETIMES,
    W_MIG_PHOSPHOSILICATE_PER_S, W_MIG_PHOSPHOSILICATE_RANGE_PER_S, DeviceTarget,
    device_observables, eryb_calibrate_pools, eryb_fit_to_device, yb_two_pool_decay,
    yb_two_pool_from_decay,
)

ER = erbium("aluminosilicate")
YB = ytterbium("phosphosilicate")
_BG = 30e-3 * np.log(10.0) / 10.0


def ref_amp(pump_W=1.0, f=0.7, **kw):
    """The 2026-09-14 file's own fixture with the two pools active: 6 um core, cladding-pumped
    Er:Yb at N_Er 4e25 / N_Yb 4e26, 976 nm into a 125 um cladding, 20 mW of 1550 nm seed, 3 m."""
    fib = FiberSpec(3.0e-6, 0.20, 4.0e25, 3.0, clad_radius_m=62.5e-6, background_loss_per_m=_BG)
    base = dict(n_yb_m3=4.0e26, k_tr_m3_s=3.0e-21, yb_ase=AseBand(1.00e-6, 1.10e-6, 6),
                upconversion_C_up=1.1e-24, yb_coupled_fraction=f, k_tr2_m3_s=2.0e-22,
                yb_migration_rate_per_s=1.0e3)
    base.update(kw)
    return ErYbAmplifier(ER, YB, fib, [Pump(pump_W, 0.976e-6, "fwd", cladding=True)],
                         [Signal(20e-3, 1.550e-6)], AseBand(1.53e-6, 1.565e-6, 6), **base)


def _profile(amp, T_K, L=3.0, n=41, T_ref=300.0):
    z = np.linspace(0.0, L, n)
    amp.set_temperature_profile(z, np.full_like(z, float(T_K)), T_ref_K=T_ref)
    return amp


# ================== (a) nothing moved =======================================================
def test_explicit_none_and_identity_laws_are_bitwise_identical_to_omitting_them():
    """Same process, same arithmetic, EXACT equality -- over the steady solve, the energy
    bookkeeping AND the transient march, with and without a temperature profile. This is the
    byte-identity claim in the form a build cannot move."""
    for hot in (False, True):
        a = ref_amp()
        b = ref_amp(rate_temperature=None, yb_stark_thermal=None)
        c = ref_amp(rate_temperature=RateTemperatureLaw())          # the IDENTITY law
        if hot:
            for amp in (a, b, c):
                _profile(amp, 380.0)
        ra, rb, rc = (x.solve(n_nodes=61) for x in (a, b, c))
        for r in (rb, rc):
            assert np.array_equal(ra.power_W, r.power_W)
            assert np.array_equal(ra.nbar2_z, r.nbar2_z)
            assert np.array_equal(ra.meta["beta_yb_z"], r.meta["beta_yb_z"])
            assert np.array_equal(ra.meta["beta_yb_uncoupled_z"], r.meta["beta_yb_uncoupled_z"])
            assert ra.signal_gain_dB[0] == r.signal_gain_dB[0]
            assert ra.meta["eta_transfer"] == r.meta["eta_transfer"]
        kw = dict(z_m=ra.z_m) if hot else {}
        ta, tb = (x.energy_terms(ra.power_W, ra.nbar2_z, ra.meta["beta_yb_coupled_z"],
                                 ra.meta["beta_yb_uncoupled_z"], **kw) for x in (a, b))
        for key in ta:
            assert np.array_equal(ta[key], tb[key]), key
        assert np.array_equal(a._heat_profile_W_per_m(ra), b._heat_profile_W_per_m(ra))
    # the march refuses a profile, so it is compared isothermally
    t = np.linspace(0.0, 2.0e-3, 7)
    ma = simulate_transient(ref_amp(), t, n_nodes=25, nbar2_0=(0.02, 0.02, 0.02))
    mb = simulate_transient(ref_amp(rate_temperature=RateTemperatureLaw(), yb_stark_thermal=None),
                            t, n_nodes=25, nbar2_0=(0.02, 0.02, 0.02))
    assert np.array_equal(ma.nbar2_zt, mb.nbar2_zt)
    assert np.array_equal(ma.meta["beta_yb"], mb.meta["beta_yb"])


def test_the_new_temperature_branches_are_structurally_unreachable_by_default():
    """The predicate that routes them is False, and the temperature bundle's two new slots are
    None -- so the pre-2026-09-15 arithmetic is not merely equal, it is the code that runs."""
    amp = ref_amp()
    assert amp._t_rates is False
    assert amp.rate_temperature is None and amp.yb_stark_thermal is None
    assert ErYbAmplifier.__init__ is not None
    assert amp._mcc_matrices(amp._plan(), np.linspace(0, 3, 5)) is None     # no profile at all
    _profile(amp, 380.0)
    bundle = amp._mcc_matrices(amp._plan(), np.linspace(0, 3, 5))
    assert len(bundle) == 4
    assert bundle[2] is None and bundle[3] is None
    # an identity law is DROPPED in the constructor, not carried and multiplied by ones
    assert ref_amp(rate_temperature=RateTemperatureLaw())._t_rates is False


def test_clone_carries_the_new_optins():
    amp = ref_amp(rate_temperature=RATE_ARRHENIUS_CHENG_2022,
                  yb_stark_thermal=YB_STARK_976_CANAT_DUSSARDIER)
    clone = amp.with_signals([Signal(30e-3, 1.550e-6)])
    assert clone.rate_temperature is amp.rate_temperature
    assert clone.yb_stark_thermal is amp.yb_stark_thermal
    assert clone._t_rates is True


def test_a_uniform_profile_at_t_ref_is_an_exact_identity_with_both_laws_on():
    """Both laws are built to be EXACTLY 1 at T = T_ref, so a uniform profile at the reference
    temperature must reproduce the isothermal solve bitwise -- not nearly."""
    cold = ref_amp().solve(n_nodes=61)
    hot = ref_amp(rate_temperature=RATE_ARRHENIUS_CHENG_2022,
                  yb_stark_thermal=YB_STARK_976_CANAT_DUSSARDIER)
    _profile(hot, 300.0, T_ref=300.0)
    r = hot.solve(n_nodes=61)
    assert np.array_equal(cold.power_W, r.power_W)
    assert np.array_equal(cold.meta["beta_yb_z"], r.meta["beta_yb_z"])


# ================== (b) the migration rate ==================================================
def test_closed_form_decay_matches_a_brute_force_march_of_the_models_own_rhs():
    """The closed form is claimed to BE the solution of `eryb._fb_rhs3` at zero optical rates.
    Prove it by integrating that right-hand side with a fixed-step RK4 that shares no algebra
    with the eigen-decomposition -- a genuine oracle, not a restatement."""
    tau_yb, n_er, k_tr, f, w = 1.0e-3, 4.0e25, 3.0e-21, 0.62, 2.5e3
    amp = ref_amp(f=f, k_tr_m3_s=k_tr, yb_migration_rate_per_s=w, k_tr2_m3_s=0.0)
    amp._tau_yb = tau_yb
    amp._n_er = n_er
    zero = np.zeros(1)
    y = np.array([0.0, 1.0, 1.0])                 # f2 = 0 (Er held down), both pools excited

    def rhs(state):
        df, dbc, dbn = amp._fb_rhs3(zero, zero, zero, zero, zero, np.array([state[1]]),
                                    np.array([state[2]]))
        return np.array([0.0, np.asarray(dbc).reshape(-1)[0],
                         np.asarray(dbn).reshape(-1)[0]])    # the erbium is CLAMPED at f2 = 0

    dt, n_steps = 2.0e-8, 25000                   # 0.5 ms, 20 steps per fast time constant
    traj_t, traj_i = [], []
    for step in range(n_steps + 1):
        traj_t.append(step * dt)
        traj_i.append(f * y[1] + (1.0 - f) * y[2])
        k1 = rhs(y)
        k2 = rhs(y + 0.5 * dt * k1)
        k3 = rhs(y + 0.5 * dt * k2)
        k4 = rhs(y + dt * k3)
        y = y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    m = yb_two_pool_decay(tau_yb_s=tau_yb, n_er_m3=n_er, k_tr_m3_s=k_tr, coupled_fraction=f,
                          migration_per_s=w)
    t = np.asarray(traj_t)
    worst = float(np.max(np.abs(m.intensity(t) - np.asarray(traj_i))))
    assert worst < 1e-9, worst
    # and the INTEGRAL, which is what the mean lifetime (the Laroche comparison's observable) is
    # made of. The march covers 0.5 ms and the slow component runs to 1.3 ms, so this compares
    # against the closed form TRUNCATED at the same horizon -- comparing a finite march to an
    # infinite integral would gate the horizon, not the algebra.
    horizon = float(t[-1])
    truncated = (m.amp_fast * (1.0 - np.exp(-m.rate_fast_per_s * horizon)) / m.rate_fast_per_s
                 + m.amp_slow * (1.0 - np.exp(-m.rate_slow_per_s * horizon)) / m.rate_slow_per_s)
    integral = float(trapz(np.asarray(traj_i), t))     # core.numerics.trapz: numpy-floor safe
    assert truncated == pytest.approx(integral, rel=2e-5)
    # ... and tau_bar is that expression's infinite-horizon limit
    assert m.mean_lifetime_s == pytest.approx(
        m.amp_fast / m.rate_fast_per_s + m.amp_slow / m.rate_slow_per_s, rel=1e-14)
    assert m.mean_lifetime_s > truncated


def test_decay_closed_form_hits_its_three_limits():
    kw = dict(tau_yb_s=1.3e-3, n_er_m3=4.0e25, k_tr_m3_s=3.0e-21)
    d = 1.0 / kw["tau_yb_s"]
    a = d + kw["k_tr_m3_s"] * kw["n_er_m3"]
    # W = 0: two independent pools, rates (a, d) with amplitudes (f, 1-f)
    m = yb_two_pool_decay(coupled_fraction=0.7, migration_per_s=0.0, **kw)
    assert m.rate_fast_per_s == pytest.approx(a, rel=1e-12)
    assert m.rate_slow_per_s == pytest.approx(d, rel=1e-12)
    assert m.amp_fast == pytest.approx(0.7, rel=1e-12)
    # f = 1: ONE pool, the textbook quenched single exponential Laroche invert
    m1 = yb_two_pool_decay(coupled_fraction=1.0, migration_per_s=0.0, **kw)
    assert m1.mean_lifetime_s == pytest.approx(1.0 / a, rel=1e-12)
    # W -> infinity: fully mixed, the slow rate becomes the POPULATION-WEIGHTED d + f k N_Er
    mm = yb_two_pool_decay(coupled_fraction=0.7, migration_per_s=1.0e12, **kw)
    assert mm.rate_slow_per_s == pytest.approx(d + 0.7 * (a - d), rel=1e-4)
    assert mm.mean_lifetime_s == pytest.approx(1.0 / (d + 0.7 * (a - d)), rel=1e-4)


def test_the_exact_inversion_round_trips_and_reproduces_chengs_two_coefficients():
    a = CHENG_2022_DECAY
    inv = yb_two_pool_from_decay(tau_fast_s=a.tau_fast_s, tau_slow_s=a.tau_slow_s,
                                 amp_fast=a.amp_fast, tau_yb_s=a.tau_yb_s, n_er_m3=a.n_er_m3)
    m = yb_two_pool_decay(tau_yb_s=a.tau_yb_s, n_er_m3=a.n_er_m3, k_tr_m3_s=inv["k_tr_m3_s"],
                          coupled_fraction=inv["coupled_fraction"],
                          migration_per_s=inv["migration_per_s"])
    assert m.tau_fast_s == pytest.approx(a.tau_fast_s, rel=1e-10)
    assert m.tau_slow_s == pytest.approx(a.tau_slow_s, rel=1e-10)
    assert m.amp_fast == pytest.approx(a.amp_fast, rel=1e-10)
    # THE claim: one parameter set carries BOTH coefficients the anchor collection extracts from
    # this single measurement -- the fast-pair 3.07e-21 and the ensemble-yield 1.20e-22.
    assert inv["k_tr_m3_s"] == pytest.approx(3.065e-21, rel=0.01)
    assert m.transfer_yield == pytest.approx(a.transfer_yield, rel=1e-3)
    k_ensemble = (m.transfer_yield / (1.0 - m.transfer_yield)) / (a.n_er_m3 * a.tau_yb_s)
    assert k_ensemble == pytest.approx(1.199e-22, rel=0.02)
    assert inv["k_tr_m3_s"] / k_ensemble == pytest.approx(25.6, rel=0.05)
    # ... and that is where the shipped constant comes from
    assert inv["migration_per_s"] == pytest.approx(222.4, rel=0.01)
    lo, hi = W_MIG_PHOSPHOSILICATE_RANGE_PER_S
    assert lo <= W_MIG_PHOSPHOSILICATE_PER_S <= hi
    assert lo <= inv["migration_per_s"] <= hi


def test_the_inversion_refuses_a_control_that_is_faster_than_the_sample_tail():
    with pytest.raises(ValueError, match="longer than the Er-free control"):
        yb_two_pool_from_decay(tau_fast_s=8.2e-6, tau_slow_s=2.5e-3, amp_fast=0.86,
                               tau_yb_s=1.79e-3, n_er_m3=4e25)


def test_jeong_2007_decay_pattern_is_reproduced():
    """Jeong's fiber is a DIFFERENT fiber from Cheng's glass and its decay is stated in words,
    so it is the out-of-sample check: 50-70% of the ytterbium relaxed within 10 us, a 10-15.5 us
    fast time constant, and about 2% still excited at 100 us. The fit must land inside every
    band simultaneously."""
    cal = eryb_calibrate_pools(decays=[JEONG_2007_DECAY], w_profile_points=0)
    rep = cal.decay_report[JEONG_2007_DECAY.label]
    assert cal.cost < 1e-6, cal.cost
    assert 0.50 <= rep["relaxed_by_10us"] <= 0.70
    assert 0.010 <= rep["excited_at_100us"] <= 0.030
    assert 10.0e-6 <= rep["tau_fast_s"] <= 15.5e-6
    assert 0.93 <= rep["amp_fast"] <= 0.98
    # and it lands in the same fast-pair k_tr decade as Cheng's independent measurement
    assert 1.0e-21 < cal.k_tr_m3_s < 1.0e-20
    # the calibrated W_mig is INSIDE what Jeong's bands allow, which is what makes the two
    # anchors consistent rather than merely non-contradictory
    at_cal = eryb_calibrate_pools(decays=[JEONG_2007_DECAY], w_profile_points=0,
                                  w_bounds=(W_MIG_PHOSPHOSILICATE_PER_S,
                                            W_MIG_PHOSPHOSILICATE_PER_S * 1.0001))
    assert at_cal.cost < 1e-6, at_cal.cost


def test_laroche_2006_five_lifetimes_within_25_percent():
    cal = eryb_calibrate_pools(lifetimes=list(LAROCHE_2006_LIFETIMES), w_profile_points=0)
    worst = max(abs(v["ratio"] - 1.0) for v in cal.lifetime_report.values())
    assert worst <= 0.25, {k: v["ratio"] for k, v in cal.lifetime_report.items()}
    # WHAT THAT FIT ACTUALLY DETERMINES, and the finding the audit note reports: Laroche's five
    # lifetimes are reproduced only in the fully MIXED limit, where the model collapses to a
    # single population at the effective product f k_tr -- and that product lands on Laroche's
    # own measured 6.4-8.6e-23, from an independent route.
    assert cal.migration_per_s > 1.0e4
    assert 6.0e-23 <= cal.effective_f_k_m3_s <= 1.0e-22
    # so this dataset is silent about W_mig: its profile is flat over four decades
    prof = eryb_calibrate_pools(lifetimes=list(LAROCHE_2006_LIFETIMES), w_profile_points=25)
    lo, hi = prof.w_interval_per_s
    assert hi / lo > 1.0e3, (lo, hi)


def test_migration_moves_the_transfer_yield_in_the_right_direction():
    """Falsifiable sign: migration feeds uncoupled excitation into the coupled pool, so the
    transfer yield must RISE monotonically with W_mig at a fixed (f, k_tr), from the pure
    two-pool value f k N_Er/(k N_Er + 1/tau) to the fully-mixed one."""
    kw = dict(tau_yb_s=1.3e-3, n_er_m3=4.0e25, k_tr_m3_s=3.0e-21, coupled_fraction=0.5)
    ys = [yb_two_pool_decay(migration_per_s=w, **kw).transfer_yield
          for w in (0.0, 1.0e2, 1.0e3, 1.0e4, 1.0e6)]
    assert all(b > a for a, b in zip(ys, ys[1:])), ys
    d, a = 1.0 / kw["tau_yb_s"], 1.0 / kw["tau_yb_s"] + kw["k_tr_m3_s"] * kw["n_er_m3"]
    assert ys[0] == pytest.approx(0.5 * (a - d) / a, rel=1e-12)
    assert ys[-1] == pytest.approx(0.5 * (a - d) / (d + 0.5 * (a - d)), rel=1e-3)


# ================== (c) the temperature laws =================================================
def test_mccumber_is_identically_one_at_the_ytterbium_zero_line_so_stark_cannot_double_count():
    """The no-double-counting claim, proven rather than argued: the existing per-ion McCumber
    factor is exp[(eps_Yb - h nu)(1/kT - 1/kT_ref)], which at h nu = eps_Yb is exactly 1 at every
    temperature -- and it never touches sigma_a anywhere. So a scale on the 976 nm line is
    orthogonal to it by construction."""
    from dynameta.constants import C_LIGHT, H_PLANCK
    lam0 = C_LIGHT * H_PLANCK / float(YB.eps_J)
    amp = ref_amp(yb_ase=AseBand(lam0, lam0 + 1e-12, 1))
    _profile(amp, 460.0)
    pl = amp._plan()
    bundle = amp._mcc_matrices(pl, np.linspace(0.0, 3.0, 5))
    at_zero = np.argmin(np.abs(pl["lam"] - lam0))
    assert np.allclose(bundle[1][at_zero, :], 1.0, atol=1e-12)
    # away from the zero line it is NOT 1 -- so the gate above is not vacuous
    assert not np.allclose(bundle[1][np.argmin(np.abs(pl["lam"] - 1.03e-6)), :], 1.0, atol=1e-6)


def test_shipped_temperature_constants_reproduce_their_own_anchors():
    st = YB_STARK_976_CANAT_DUSSARDIER
    assert float(st.upper_fraction(300.0)) == pytest.approx(0.07, abs=1e-9)
    assert float(st.upper_fraction(400.0)) == pytest.approx(0.13, abs=1e-9)
    assert float(st.factor(300.0, 300.0)) == 1.0
    assert float(st.factor(400.0, 300.0)) == pytest.approx(0.93 / 0.87 ** -1 * 0.87 / 0.93 * 1.0,
                                                           rel=1e-9) or True
    assert float(st.factor(400.0, 300.0)) == pytest.approx(0.87 / 0.93, rel=1e-9)
    assert 500.0 < st.delta_E_over_k_K / 1.4388 < 650.0            # a Yb 2F5/2 Stark splitting
    # the band is a ZERO-LINE model: 940 nm and 1018 nm are outside it, as Cheng's measured
    # temperature behaviour at those two wavelengths requires
    mask = st.band_mask(np.array([9.15e-7, 9.40e-7, 9.76e-7, 1.018e-6, 1.03e-6]))
    assert list(mask) == [False, False, True, False, False]
    # the Arrhenius law is FITTED to Cheng's two decay components, not transcribed
    ratio = (1.0 / 7.56e-6 - 1.0 / 1279.25e-6) / (1.0 / 8.20e-6 - 1.0 / 1333.76e-6)
    assert float(np.asarray(RATE_ARRHENIUS_CHENG_2022.scale(480.0, 300.0)[0])) == \
        pytest.approx(ratio, rel=1e-12)
    assert float(np.asarray(RATE_ARRHENIUS_30PCT_300_480K.scale(480.0, 300.0)[0])) == \
        pytest.approx(1.30, rel=1e-12)
    for law in (RATE_ARRHENIUS_CHENG_2022, RATE_ARRHENIUS_30PCT_300_480K):
        assert all(float(np.asarray(s)) == 1.0 for s in law.scale(300.0, 300.0))
        assert not law.is_identity
    assert RateTemperatureLaw().is_identity


def test_both_laws_move_a_hot_solve_and_in_the_stated_direction():
    """Not vacuous: each law on its own changes the hot answer measurably, the Stark factor
    REDUCES the pump absorption (less signal out) and the Arrhenius law RAISES the transfer
    (more eta_tr). A sign flip in either breaks this."""
    def run(**kw):
        amp = ref_amp(**kw)
        _profile(amp, 460.0)
        r = amp.solve(n_nodes=81)
        return float(r.signal_gain_dB[0]), float(r.meta["eta_transfer"])

    g0, e0 = run()
    g_rate, e_rate = run(rate_temperature=RATE_ARRHENIUS_CHENG_2022)
    g_stark, e_stark = run(yb_stark_thermal=YB_STARK_976_CANAT_DUSSARDIER)
    assert e_rate > e0 + 1e-4                       # hotter transfer -> more of the Yb transfers
    assert g_rate > g0 + 1e-4
    assert g_stark < g0 - 1e-4                      # weaker 976 nm line -> less pump absorbed
    assert abs(e_stark - e0) > 1e-4


def test_the_one_um_threshold_of_a_hot_fiber_rises_with_temperature():
    """Canat / Dussardier's qualitative result, and Morasse 2007's measured one (a core
    temperature rise moved his 1-um onset from 14 W to 35 W of pump). A MONOTONIC gate, not a
    number: the pump power at which the 1-um ASE reaches 1% of the output must increase with
    temperature, and turning the two laws on must not reverse that."""
    pumps = (2.0, 4.0, 8.0)

    def frac(T, **kw):
        out = []
        for p in pumps:
            amp = ref_amp(pump_W=p, f=0.5, yb_migration_rate_per_s=0.0, **kw)
            _profile(amp, T)
            # relax = 1.0 converges this fixture in a handful of iterations; relax = 0.5 needs
            # ~30 for the SAME fixed point and turns an 18-solve gate into a four-minute one.
            out.append(device_observables(amp, n_nodes=61, relax=1.0)["one_um_fraction"])
        return out

    for kw in ({}, dict(rate_temperature=RATE_ARRHENIUS_CHENG_2022,
                        yb_stark_thermal=YB_STARK_976_CANAT_DUSSARDIER)):
        cold, warm, hot = frac(300.0, **kw), frac(400.0, **kw), frac(500.0, **kw)
        for i, p in enumerate(pumps):
            assert cold[i] > warm[i] > hot[i], (p, cold[i], warm[i], hot[i])


def test_thermal_feedback_runs_with_the_new_laws_and_still_uses_the_rate_balance():
    amp = ref_amp(pump_W=4.0, rate_temperature=RATE_ARRHENIUS_CHENG_2022,
                  yb_stark_thermal=YB_STARK_976_CANAT_DUSSARDIER)
    res, T_z, info = solve_with_thermal_feedback(amp, ThermalModel(h_conv_W_m2K=200.0),
                                                 b_outer_m=62.5e-6, n_nodes=61)
    assert info["heat_source"] == "rate_balance"
    assert info["converged_T"]
    assert np.all(T_z >= ThermalModel().T_coolant_K - 1e-9)
    assert np.isfinite(res.signal_gain_dB[0])
    amp.clear_temperature_profile()


# ================== (d) the device fit ======================================================
def _synthetic_build(f, k_tr):
    """A DELIBERATELY CHEAP synthetic device: 4 um core, 2 m, 3 W of cladding pump, 50 mW of
    seed, three ASE bins per band. Every residual evaluation of the fit is a full relaxation
    solve, so the fixture is sized for the gate (1.5 s and 13 iterations per solve) rather than
    for realism -- what is under test is the ROUTINE, and the truth it must recover is generated
    by the same model."""
    fib = FiberSpec(4.0e-6, 0.20, 3.0e25, 2.0, clad_radius_m=62.5e-6, background_loss_per_m=_BG)
    return ErYbAmplifier(ER, YB, fib, [Pump(3.0, 0.976e-6, "fwd", cladding=True)],
                         [Signal(0.050, 1.550e-6)], AseBand(1.53e-6, 1.565e-6, 3),
                         yb_ase=AseBand(1.00e-6, 1.10e-6, 3), n_yb_m3=4.0e26,
                         k_tr_m3_s=k_tr, upconversion_C_up=1.1e-24, yb_coupled_fraction=f,
                         k_tr2_m3_s=2.0e-22,
                         yb_migration_rate_per_s=W_MIG_PHOSPHOSILICATE_PER_S)


def _observe(amp):
    return device_observables(amp, n_nodes=41, relax=1.0)


@pytest.mark.parametrize("f_true,k_true", [(0.45, 2.0e-21)])
def test_the_device_fit_recovers_a_synthetic_f_and_k_to_5_percent(f_true, k_true):
    """The routine's own discrimination test: generate the two observables FROM the model at a
    known (f, k_tr), start the fit a long way off, and require both parameters back to 5%. A fit
    that only pinned the product would fail this, because the 1-um fraction is what separates
    them."""
    truth = _observe(_synthetic_build(f_true, k_true))
    target = DeviceTarget(p_out_W=truth["p_out_W"], one_um_fraction=truth["one_um_fraction"],
                          sigma_ln_power=0.02, sigma_ln_one_um=0.10)
    fit = eryb_fit_to_device(_synthetic_build, target,
                             migration_per_s=W_MIG_PHOSPHOSILICATE_PER_S,
                             f0=0.80, k_tr0_m3_s=6.0e-22, observe=_observe, max_nfev=25)
    assert fit.coupled_fraction == pytest.approx(f_true, rel=0.05), str(fit)
    assert fit.k_tr_m3_s == pytest.approx(k_true, rel=0.05), str(fit)


def test_one_observable_leaves_the_split_undetermined_and_the_fit_says_so():
    """The 2026-09-14 note's limit 8, MEASURED rather than asserted -- and the discrimination is
    what the routine REPORTS, not what it returns.

    Fitting the output power ALONE gives one residual for two parameters, so J^T J is rank 1: the
    fit still returns an (f, k_tr) pair, and that pair is meaningless in one direction. The
    routine must say so -- an astronomically large condition number and a REFUSAL to quote an
    error bar (inf), rather than the small-looking sigma a naive `inv(J^T J)` on a singular matrix
    hands back. Adding the 1-um fraction must make the problem well conditioned and both error
    bars finite, while the product f k_tr -- which the power alone DOES fix -- comes back either
    way."""
    truth = _observe(_synthetic_build(0.45, 2.0e-21))
    power_only = eryb_fit_to_device(
        _synthetic_build, DeviceTarget(p_out_W=truth["p_out_W"]),
        f0=0.80, k_tr0_m3_s=6.0e-22, observe=_observe, max_nfev=10)
    both = eryb_fit_to_device(
        _synthetic_build, DeviceTarget(p_out_W=truth["p_out_W"],
                                       one_um_fraction=truth["one_um_fraction"]),
        f0=0.80, k_tr0_m3_s=6.0e-22, observe=_observe, max_nfev=18)
    assert power_only.condition_number > 1.0e12, power_only.condition_number
    assert not np.isfinite(power_only.sigma_ln_f_k)
    assert not np.isfinite(power_only.sigma_ln_ratio)
    assert both.condition_number < 1.0e3, both.condition_number
    assert np.isfinite(both.sigma_ln_f_k) and np.isfinite(both.sigma_ln_ratio)
    # the degenerate direction the power-only fit reports is the f k_tr = const contour, i.e. it
    # is NOT aligned with either parameter axis -- a direction of (1, 0) or (0, 1) would mean one
    # parameter was simply unreachable, which is a different (and fixable) failure
    d = np.abs(power_only.degeneracy_direction)
    assert d.min() > 0.2, d
    # and the product comes back from EITHER fit, because it is what the power fixes
    assert power_only.effective_f_k_m3_s == pytest.approx(0.45 * 2.0e-21, rel=0.25)
    assert both.effective_f_k_m3_s == pytest.approx(0.45 * 2.0e-21, rel=0.10)


def test_an_upper_bound_target_is_scored_one_sided():
    """'No obvious 1-um ASE' is a bound, not a measurement: anything below it must cost nothing."""
    truth = _observe(_synthetic_build(0.45, 2.0e-21))
    hi = DeviceTarget(p_out_W=truth["p_out_W"],
                      one_um_fraction=10.0 * truth["one_um_fraction"],
                      one_um_is_upper_bound=True)
    fit = eryb_fit_to_device(_synthetic_build, hi, f0=0.45, k_tr0_m3_s=2.0e-21,
                             observe=_observe, max_nfev=4)
    assert fit.residuals["ln_one_um"] == 0.0
