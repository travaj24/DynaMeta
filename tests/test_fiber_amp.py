"""Discrimination-proven physics gates for the rare-earth fiber-amplifier package
(dynameta.optics.fiber_amp), mirroring the depth of test_soa.py. Pure numpy/scipy; each test is
a falsifiable gate, kept small so the suite runs in CI. Grouped by build phase."""

import numpy as np

from dynameta.constants import C_LIGHT, H_PLANCK
from dynameta.optics.fiber_amp import (
    erbium, ytterbium, FiberSpec, overlap_gamma, cladding_pump_overlap,
    ChannelSet, metastable_fraction, gain_coeff_per_m,
    RareEarthIon, CrossSectionModel, at_temperature, multiphonon_lifetime,
    Pump, Signal, AseBand, FiberAmplifier,
    analyze_noise, noise_figure,
    gain_compression_curve, slope_efficiency, power_conversion_efficiency, stokes_limit,
    ConcentrationModel,
    ThermalModel, quantum_defect_fraction, total_heat_W, heat_load_per_m,
    peak_temperature_rise, radial_temperature_rise,
    simulate_transient, saturation_energy, frantz_nodvik_output_energy, frantz_nodvik_pulse,
    CrossSectionTable, giles_calibrated_fiber, dB_per_m_to_per_m,
    detection_noise,
)

ER = erbium("aluminosilicate")
YB = ytterbium("aluminosilicate")


def _edf(length_m=6.0, n_t=1.0e25):
    return FiberSpec(core_radius_m=1.4e-6, na=0.24, n_t_m3=n_t, length_m=length_m)


def _amp(length_m=6.0, pump_mW=100.0, sig_uW=1.0, n_bins=10, sig_nm=1560.0):
    ase = AseBand(1.52e-6, 1.575e-6, n_bins=n_bins) if n_bins else None
    return FiberAmplifier(ER, _edf(length_m), [Pump(pump_mW * 1e-3, 0.980e-6, "fwd")],
                          [Signal(sig_uW * 1e-6, sig_nm * 1e-9)], ase)


# ============================ Phase 1: spectroscopy + rate core ============================

def test_mccumber_crossover_at_zero_line():
    lam0 = ER.zero_line_m
    ratio = ER.sigma_e_mccumber(lam0) / ER.sigma_a.sigma(lam0)
    assert abs(ratio - 1.0) < 1e-9          # detailed balance: sigma_e = sigma_a at the zero line


def test_cross_sections_nonnegative_and_peaked():
    lam = np.linspace(1.45e-6, 1.62e-6, 500)
    assert np.all(ER.sigma_a.sigma(lam) >= 0.0) and np.all(ER.sigma_e.sigma(lam) >= 0.0)
    # absorption peak near 1530 nm
    assert abs(lam[np.argmax(ER.sigma_a.sigma(lam))] - 1.530e-6) < 3e-9


def test_overlap_bounds_and_dispersion():
    f = _edf()
    g530 = float(overlap_gamma(f, 1.530e-6))
    g1560 = float(overlap_gamma(f, 1.560e-6))
    assert 0.0 < g1560 < g530 < 1.0         # in (0,1) and falls with wavelength


def test_overlap_override_respected():
    f = FiberSpec(1.4e-6, 0.24, 1e25, 1.0, overlap_override=0.42)
    assert abs(float(overlap_gamma(f, 1.55e-6)) - 0.42) < 1e-12


def test_metastable_fraction_bounds_and_saturation():
    f = _edf()
    ch = ChannelSet.build(ER, f, np.array([0.980e-6]), np.array([1.0]))
    n_lo = metastable_fraction(ch, np.array([1e-4]), f)
    n_hi = metastable_fraction(ch, np.array([10.0]), f)
    assert 0.0 <= n_lo < n_hi <= 1.0 and n_hi > 0.95    # strong pump -> near full inversion


def test_gain_core_beer_lambert_and_full_inversion():
    f = _edf()
    ch = ChannelSet.build(ER, f, np.array([1.560e-6]), np.array([1.0]))
    g0 = gain_coeff_per_m(ch, 0.0, f)[0]        # unpumped -> -alpha
    g1 = gain_coeff_per_m(ch, 1.0, f)[0]        # full inversion -> +g*
    alpha = float(ch.gamma[0] * f.n_t_m3 * ch.sigma_a[0])
    gstar = float(ch.gamma[0] * f.n_t_m3 * ch.sigma_e[0])
    assert abs(g0 + alpha) < 1e-12 and abs(g1 - gstar) < 1e-12


# ============================ Phase 2: steady state ============================

def test_steady_beer_lambert_matches_analytic():
    amp = FiberAmplifier(ER, _edf(6.0), [], [Signal(1e-6, 1.560e-6)], None)
    r = amp.solve()
    f = _edf(6.0)
    alpha = float(overlap_gamma(f, 1.560e-6)) * f.n_t_m3 * float(ER.sigma_a.sigma(1.560e-6))
    analytic_dB = -10.0 * np.log10(np.e) * alpha * f.length_m
    assert abs(float(r.signal_gain_dB[0]) - analytic_dB) < 0.05


def test_steady_pumped_converges_and_physical():
    r = _amp(pump_mW=100.0).solve()
    assert r.meta["converged"]
    assert 0.0 <= r.nbar2_z.min() and r.nbar2_z.max() <= 1.0
    assert float(r.signal_gain_dB[0]) > 10.0


def test_photon_number_not_exceeding_pump():
    r = _amp(pump_mW=100.0).solve()
    ip, is_ = r.kind.index("pump"), r.kind.index("signal")

    def ph(P, lam):
        return P / (H_PLANCK * C_LIGHT / lam)
    pump_loss = ph(r.power_W[ip, 0], r.lambda_m[ip]) - ph(r.power_W[ip, -1], r.lambda_m[ip])
    gained = ph(r.power_W[is_, -1], r.lambda_m[is_]) - ph(r.power_W[is_, 0], r.lambda_m[is_])
    gained += sum(ph(r.power_W[k, -1 if r.u[k] > 0 else 0], r.lambda_m[k])
                  for k in np.where(r.is_ase)[0])
    assert 0.0 < gained / pump_loss <= 1.02


def test_gain_saturation_and_ase_quenching():
    weak = _amp(pump_mW=100.0, sig_uW=1.0).solve()
    strong = _amp(pump_mW=100.0, sig_uW=2000.0).solve()   # 2 mW input
    g_weak = float(weak.signal_gain_dB[0])
    g_strong = float(strong.signal_gain_dB[0])
    ase_weak = float(np.sum(weak.power_W[(weak.u > 0) & weak.is_ase, -1]))
    ase_strong = float(np.sum(strong.power_W[(strong.u > 0) & strong.is_ase, -1]))
    assert g_strong < g_weak - 3.0 and ase_strong < ase_weak


# ============================ Phase 3: ASE + noise figure ============================

def test_local_nsp_quantum_floor():
    nr = analyze_noise(_amp(pump_mW=100.0).solve(), 1.560e-6)
    assert nr.n_sp_local_min >= 1.0 - 1e-9


def test_noise_figure_self_consistency():
    nf_lin, G, n_sp = noise_figure(_amp(pump_mW=100.0).solve(), 1.560e-6)
    assert abs(nf_lin - (2.0 * n_sp * (G - 1.0) + 1.0) / G) / nf_lin < 1e-6


def test_high_gain_preamp_approaches_3dB():
    amp = FiberAmplifier(ER, _edf(1.5, n_t=2.5e25), [Pump(1.5, 0.980e-6, "fwd")],
                         [Signal(1e-6, 1.560e-6)], AseBand(1.52e-6, 1.575e-6, 30))
    nr = analyze_noise(amp.solve(), 1.560e-6)
    assert nr.gain_dB > 12.0 and abs(nr.nf_dB - 10.0 * np.log10(2.0)) < 0.4


# ============================ Phase 4: saturation + metrics ============================

def test_slope_efficiency_below_stokes():
    amp = _amp(pump_mW=150.0, n_bins=8)
    se = slope_efficiency(amp, np.linspace(20e-3, 400e-3, 6), saturating_signal_W=5e-3)
    ceil = stokes_limit(0.980e-6, 1.560e-6)
    assert 0.0 < se.slope <= ceil * 1.02


def test_gain_compression_monotonic_and_psat():
    cc = gain_compression_curve(_amp(pump_mW=150.0, n_bins=8), np.geomspace(1e-7, 3e-2, 8))
    assert np.all(np.diff(cc.gain_dB) <= 1e-6)
    assert np.isfinite(cc.p_sat_out_W) and cc.p_sat_out_W > 0.0


def test_pce_below_stokes():
    amp = FiberAmplifier(ER, _edf(6.0), [Pump(300e-3, 0.980e-6, "fwd")],
                         [Signal(5e-3, 1.560e-6)], AseBand(1.52e-6, 1.575e-6, 8))
    pce = power_conversion_efficiency(amp, amp.solve())
    assert 0.0 < pce < stokes_limit(0.980e-6, 1.560e-6)


def test_gain_tilt_peak_migrates_red_with_lower_inversion():
    f = _edf()
    lam = np.linspace(1.525e-6, 1.565e-6, 81)
    gam, sa, se = overlap_gamma(f, lam), ER.sigma_a.sigma(lam), ER.sigma_e.sigma(lam)

    def peak(n2):
        return lam[int(np.argmax(gam * f.n_t_m3 * (se * n2 - sa * (1 - n2))))]
    assert peak(0.45) > peak(0.90) + 1e-9


# ============================ Phase 5: concentration / degradation ============================

def test_concentration_opt_in_byte_identical():
    ase = AseBand(1.52e-6, 1.575e-6, 8)
    r0 = FiberAmplifier(ER, _edf(6.0), [Pump(100e-3, 0.980e-6, "fwd")],
                        [Signal(1e-6, 1.560e-6)], ase).solve()
    r1 = FiberAmplifier(ER, _edf(6.0), [Pump(100e-3, 0.980e-6, "fwd")],
                        [Signal(1e-6, 1.560e-6)], ase,
                        concentration=ConcentrationModel()).solve()     # all-default identity
    assert np.array_equal(r0.power_W, r1.power_W) and np.array_equal(r0.nbar2_z, r1.nbar2_z)


def test_upconversion_clamps_inversion_and_gain():
    def run(cup):
        conc = ConcentrationModel(c_up_m3_s=cup) if cup else None
        return FiberAmplifier(ER, _edf(6.0), [Pump(100e-3, 0.980e-6, "fwd")],
                              [Signal(1e-6, 1.560e-6)], AseBand(1.52e-6, 1.575e-6, 8),
                              concentration=conc).solve()
    r0, rc = run(0.0), run(3e-23)
    assert rc.nbar2_z.max() < r0.nbar2_z.max() and float(rc.signal_gain_dB[0]) < float(
        r0.signal_gain_dB[0])


def test_piq_unbleachable_residual_gain_penalty():
    base = FiberAmplifier(ER, _edf(6.0), [Pump(800e-3, 0.980e-6, "fwd")],
                          [Signal(1e-6, 1.560e-6)], AseBand(1.52e-6, 1.575e-6, 8))
    piq = FiberAmplifier(ER, _edf(6.0), [Pump(800e-3, 0.980e-6, "fwd")],
                         [Signal(1e-6, 1.560e-6)], AseBand(1.52e-6, 1.575e-6, 8),
                         concentration=ConcentrationModel(pair_fraction=0.10))
    assert float(piq.solve().signal_gain_dB[0]) < float(base.solve().signal_gain_dB[0]) - 0.5


def test_photodarkening_worse_at_high_inversion():
    pd = ConcentrationModel(pd_loss_per_m=2.0, pd_exponent=7.0)

    def yb(conc, pump_mW, length, pump_lam):
        fb = FiberSpec(3.0e-6, 0.10, 5.0e25, length)
        return FiberAmplifier(YB, fb, [Pump(pump_mW * 1e-3, pump_lam, "fwd")],
                              [Signal(1e-6, 1.060e-6)], None, concentration=conc)
    n2_hi = float(yb(None, 3000, 0.6, 0.915e-6).solve().nbar2_z.max())
    pen_hi = (float(yb(None, 3000, 0.6, 0.915e-6).solve().signal_gain_dB[0])
              - float(yb(pd, 3000, 0.6, 0.915e-6).solve().signal_gain_dB[0]))
    pen_lo = (float(yb(None, 150, 3.0, 0.976e-6).solve().signal_gain_dB[0])
              - float(yb(pd, 150, 3.0, 0.976e-6).solve().signal_gain_dB[0]))
    assert n2_hi > 0.9 and pen_hi > 0.2 and pen_hi > pen_lo


def test_yb_zero_line_pump_caps_inversion():
    # pumping Yb at the 976 nm zero line (sigma_a ~ sigma_e) cannot invert past ~0.5
    fb = FiberSpec(3.0e-6, 0.10, 5.0e25, 1.0)
    r = FiberAmplifier(YB, fb, [Pump(3.0, 0.976e-6, "fwd")], [Signal(1e-6, 1.030e-6)],
                       None).solve()
    assert r.nbar2_z.max() < 0.55


# ============================ Phase 6: cladding + thermal ============================

def test_cladding_pump_overlap_ratio():
    core = FiberSpec(2.0e-6, 0.20, 2.0e25, 0.5)
    dc = FiberSpec(2.0e-6, 0.20, 2.0e25, 0.5, clad_radius_m=25.0e-6)
    assert abs(cladding_pump_overlap(dc) - (2.0 / 25.0) ** 2) < 1e-12

    def alpha(fiber, clad):
        r = FiberAmplifier(ER, fiber, [Pump(1e-6, 0.980e-6, "fwd", cladding=clad)],
                           [Signal(1e-9, 1.560e-6)], None).solve()
        ip = r.kind.index("pump")
        return -np.log(r.power_W[ip, -1] / r.power_W[ip, 0]) / fiber.length_m
    ratio = alpha(dc, True) / alpha(core, False)
    expect = cladding_pump_overlap(dc) / float(overlap_gamma(core, 0.980e-6))
    assert abs(ratio - expect) / expect < 1e-3


def test_heat_energy_balance():
    amp = FiberAmplifier(ER, FiberSpec(2.0e-6, 0.20, 2.0e25, 6.0),
                         [Pump(300e-3, 0.980e-6, "fwd")], [Signal(5e-3, 1.560e-6)],
                         AseBand(1.52e-6, 1.575e-6, 8))
    r = amp.solve()
    ip, is_ = r.kind.index("pump"), r.kind.index("signal")
    pump_abs = float(r.power_W[ip, 0] - r.power_W[ip, -1])
    sig_add = float(r.power_W[is_, -1] - r.power_W[is_, 0])
    ase_out = float(np.sum(r.power_W[(r.u > 0) & r.is_ase, -1])
                    + np.sum(r.power_W[(r.u < 0) & r.is_ase, 0]))
    heat_bal = pump_abs - sig_add - ase_out
    assert abs(total_heat_W(r) - heat_bal) < 1e-9 * max(1.0, abs(heat_bal)) + 1e-12
    assert abs(np.trapezoid(heat_load_per_m(r), r.z_m) - total_heat_W(r)) / abs(
        total_heat_W(r)) < 5e-3


def test_quantum_defect_contrast_yb_below_er():
    assert quantum_defect_fraction(0.976e-6, 1.030e-6) < 0.10 < quantum_defect_fraction(
        0.980e-6, 1.560e-6)


def test_brown_hoffman_matches_fd():
    tm = ThermalModel(1.38, 1.38, 1000.0, 300.0)
    Q, a, b = 20.0, 3.0e-6, 62.5e-6

    def fd(N=1500):
        r = np.linspace(0.0, b, N)
        dr = r[1] - r[0]
        qv = np.where(r <= a, Q / (np.pi * a ** 2), 0.0)
        k = np.where(r <= a, tm.core_k_W_mK, tm.clad_k_W_mK)
        A = np.zeros((N, N))
        rhs = np.zeros(N)
        for i in range(1, N - 1):
            rm, rp = 0.5 * (r[i - 1] + r[i]), 0.5 * (r[i] + r[i + 1])
            km, kp = 0.5 * (k[i - 1] + k[i]), 0.5 * (k[i] + k[i + 1])
            A[i, i - 1], A[i, i + 1] = rm * km / dr ** 2, rp * kp / dr ** 2
            A[i, i] = -(rm * km + rp * kp) / dr ** 2
            rhs[i] = -qv[i] * r[i]
        A[0, 0], A[0, 1] = 1.0, -1.0
        A[-1, -1], A[-1, -2] = k[-1] / dr + tm.h_conv_W_m2K, -k[-1] / dr
        return float(np.linalg.solve(A, rhs)[0])
    dT = peak_temperature_rise(Q, a, b, tm)
    assert abs(dT - fd()) / fd() < 0.03      # FD converges slowly through the r=0 singularity
    _, prof = radial_temperature_rise(Q, a, b, tm)
    assert np.all(np.diff(prof) <= 1e-9)                              # monotonic
    assert abs(peak_temperature_rise(2 * Q, a, b, tm) / dT - 2.0) < 1e-9   # linear in Q


# ============================ Phase 7: dynamics + Frantz-Nodvik ============================

def test_transient_relaxes_to_steady():
    amp = FiberAmplifier(ER, FiberSpec(2.0e-6, 0.20, 2.0e25, 6.0),
                         [Pump(120e-3, 0.980e-6, "fwd")], [Signal(50e-6, 1.560e-6)], None)
    G_ss = float(amp.solve().signal_gain_dB[0])
    tr = simulate_transient(amp, np.linspace(0.0, 40e-3, 1000), n_nodes=31, nbar2_0=0.10)
    assert abs(float(tr.signal_gain_dB[-1, 0]) - G_ss) < 0.1


def test_gain_recovery_faster_than_lifetime():
    amp = FiberAmplifier(ER, FiberSpec(2.0e-6, 0.20, 2.0e25, 6.0),
                         [Pump(120e-3, 0.980e-6, "fwd")], [Signal(50e-6, 1.560e-6)], None)
    t = np.linspace(0.0, 40e-3, 2000)
    tr = simulate_transient(amp, t, n_nodes=41, nbar2_0=0.10)
    g = tr.signal_gain_dB[:, 0]
    target = g[0] + 0.632 * (g[-1] - g[0])
    tau_eff = t[np.argmax(g >= target)]
    assert 0.0 < tau_eff < ER.tau_s


def test_add_drop_cross_gain_modulation():
    amp = FiberAmplifier(ER, FiberSpec(2.0e-6, 0.20, 2.0e25, 6.0),
                         [Pump(150e-3, 0.980e-6, "fwd")],
                         [Signal(2e-3, 1.560e-6), Signal(10e-6, 1.545e-6)], None)
    t = np.linspace(0.0, 20e-3, 1000)
    td = 8e-3
    tr = simulate_transient(amp, t, n_nodes=41,
                            signal_drive=lambda tt: np.array([2e-3 if tt < td else 1e-9, 10e-6]))
    probe = tr.signal_gain_dB[:, 1]
    before = float(probe[np.argmax(t >= td) - 1])
    assert float(probe[-1]) > before + 0.5


def test_frantz_nodvik_energy_limits():
    Esat = saturation_energy(ER, _edf(), 1.560e-6)
    G0 = np.exp(3.0)
    small = float(frantz_nodvik_output_energy(1e-4 * Esat, G0, Esat))
    large = float(frantz_nodvik_output_energy(50.0 * Esat, G0, Esat))
    assert abs(small / (G0 * 1e-4 * Esat) - 1.0) < 1e-3          # linear at low input
    assert abs((large - 50.0 * Esat) - Esat * np.log(G0)) / (Esat * np.log(G0)) < 1e-2


def test_frantz_nodvik_temporal_reshaping():
    Esat = saturation_energy(ER, _edf(), 1.560e-6)
    G0 = np.exp(3.0)
    t = np.linspace(-5e-9, 5e-9, 1500)
    p_in = np.exp(-(t / 1.5e-9) ** 2)
    p_in *= (4.0 * Esat) / np.trapezoid(p_in, t)
    p_out = frantz_nodvik_pulse(t, p_in, G0, Esat)
    E_out = np.trapezoid(p_out, t)
    E_ana = float(frantz_nodvik_output_energy(np.trapezoid(p_in, t), G0, Esat))
    assert abs(E_out - E_ana) / E_ana < 1e-2
    assert abs(p_out[0] / p_in[0] - G0) / G0 < 0.02 and p_out[-1] / p_in[-1] < 1.1


# ============================ Phase 8: calibration ============================

def test_cross_section_table_interpolates_and_holds_flat():
    lam = np.linspace(1.50e-6, 1.60e-6, 20)
    tab = CrossSectionTable(lam, ER.sigma_a.sigma(lam))
    assert np.max(np.abs(tab.sigma(lam) - ER.sigma_a.sigma(lam))) < 1e-30
    assert abs(tab.sigma(1.40e-6) - tab.sigma_m2[0]) < 1e-30     # clamped below range


def test_giles_calibration_roundtrip():
    n_t = 1.0e25
    lam = np.linspace(1.50e-6, 1.60e-6, 40)
    gam = overlap_gamma(FiberSpec(1.4e-6, 0.24, n_t, 1.0), lam)
    alpha_dBm = ER.sigma_a.sigma(lam) * gam * n_t / dB_per_m_to_per_m(1.0)
    gstar_dBm = ER.sigma_e.sigma(lam) * gam * n_t / dB_per_m_to_per_m(1.0)
    ion, fib = giles_calibrated_fiber("Er", lam, alpha_dBm, gstar_dBm, n_t_m3=n_t,
                                      core_radius_m=1.4e-6, na=0.24, length_m=3.0,
                                      tau_s=10e-3, zero_line_m=1.53e-6)
    r = FiberAmplifier(ion, fib, [Pump(1e-7, 1.530e-6, "fwd")],
                       [Signal(1e-10, 1.560e-6)], None).solve()
    ip = r.kind.index("pump")
    a_meas = -np.log(r.power_W[ip, -1] / r.power_W[ip, 0]) / fib.length_m
    a_true = float(np.interp(1.530e-6, lam, dB_per_m_to_per_m(alpha_dBm)))
    assert abs(a_meas - a_true) / a_true < 1e-3


def test_calibration_report_flags_match_and_miss():
    from dynameta.optics.fiber_amp import calibration_report
    amp = FiberAmplifier(ER, _edf(4.0, n_t=1.5e25), [Pump(150e-3, 0.980e-6, "fwd")],
                         [Signal(1e-6, 1.550e-6)], AseBand(1.53e-6, 1.565e-6, 8))
    achieved = calibration_report(amp).gain_dB     # what this config actually delivers
    # a target at the achieved gain (with a realistic NF ceiling) must be flagged as a match
    good = calibration_report(amp, {"signal_nm": 1550.0, "small_signal_gain_dB": achieved,
                                    "nf_dB_max": 6.0}, gain_tol_dB=2.0)
    assert good.ok and good.nf_dB >= 10.0 * np.log10(2.0) - 0.3   # NF at/above the quantum limit
    # an unreachable target must be flagged as a miss
    bad = calibration_report(amp, {"signal_nm": 1550.0, "small_signal_gain_dB": 60.0,
                                   "nf_dB_max": 6.0}, gain_tol_dB=1.0)
    assert not bad.gain_ok


# ============================ Phase 9: excited-state absorption (ESA) ============================

def _er_esa(peak):
    base = erbium()
    esa = CrossSectionModel(((0.980e-6, 0.016e-6, peak),))
    return RareEarthIon(base.name, base.sigma_a, base.sigma_e, base.tau_s, base.zero_line_m,
                        base.host, sigma_esa=esa)


def _esa_amp(ion):
    return FiberAmplifier(ion, _edf(6.0), [Pump(100e-3, 0.980e-6, "fwd")],
                          [Signal(1e-6, 1.560e-6)], AseBand(1.52e-6, 1.575e-6, 8))


def test_esa_opt_in_byte_identical():
    r_none = _esa_amp(erbium(esa=False)).solve()     # sigma_esa=None
    r_zero = _esa_amp(_er_esa(0.0)).solve()           # zero-magnitude ESA model
    assert np.array_equal(r_none.power_W, r_zero.power_W)


def test_esa_reduces_gain_and_raises_heat():
    r_off = _esa_amp(erbium(esa=False)).solve()
    r_on = _esa_amp(erbium(esa=True)).solve()
    ip = r_off.kind.index("pump")
    hf_off = total_heat_W(r_off) / float(r_off.power_W[ip, 0] - r_off.power_W[ip, -1])
    hf_on = total_heat_W(r_on) / float(r_on.power_W[ip, 0] - r_on.power_W[ip, -1])
    assert float(r_on.signal_gain_dB[0]) < float(r_off.signal_gain_dB[0]) - 0.3
    assert hf_on > hf_off                              # ESA is a parasitic pump->heat channel


def test_esa_monotonic_and_localized_at_pump():
    gains = [float(_esa_amp(_er_esa(pk)).solve().signal_gain_dB[0])
             for pk in (0.0, 0.4e-25, 1.6e-25)]
    assert gains[0] > gains[1] > gains[2]              # more ESA -> less gain
    er = erbium(esa=True)                              # Er ESA sits on the 980 nm pump only
    assert float(er.sigma_esa_of(0.980e-6)) > 1e-26
    assert float(er.sigma_esa_of(1.560e-6)) < 0.01 * float(er.sigma_esa_of(0.980e-6))


def test_ytterbium_is_esa_free():
    assert ytterbium().sigma_esa is None              # Yb: only one excited 4f manifold


# ============================ Phase 10: temperature dependence ============================

def test_temperature_ref_byte_identical():
    r0 = _esa_amp(erbium()).solve()
    r_ref = _esa_amp(at_temperature(erbium(), 300.0, T_ref_K=300.0)).solve()
    assert np.array_equal(r0.power_W, r_ref.power_W)   # T = T_ref is a no-op


def test_mccumber_crossover_T_invariant():
    lam0 = ER.zero_line_m
    ratios = [float(at_temperature(ER, T).sigma_e.sigma(lam0))
              / float(at_temperature(ER, T).sigma_a.sigma(lam0)) for T in (250.0, 350.0, 450.0)]
    assert max(ratios) - min(ratios) < 1e-12           # crossover is T-pinned at the zero line


def test_gain_tilt_with_temperature():
    # McCumber-T lowers the red-side (1560) emission relative to the blue (1530) as T rises
    def se_ratio(T):
        ion = at_temperature(ER, T)
        return float(ion.sigma_e.sigma(1.560e-6)) / float(ion.sigma_e.sigma(1.530e-6))
    assert se_ratio(360.0) < se_ratio(280.0)
    g_cold = float(_esa_amp(at_temperature(ER, 280.0)).solve().signal_gain_dB[0])
    g_hot = float(_esa_amp(at_temperature(ER, 360.0)).solve().signal_gain_dB[0])
    assert g_hot < g_cold                              # red-side amplifier gain drops with T


def test_multiphonon_energy_gap_law():
    tau = 10.0e-3
    assert multiphonon_lifetime(tau, 500.0, gap_cm=6500.0) == tau          # coupling 0 -> radiative
    big = multiphonon_lifetime(tau, 300.0, gap_cm=6500.0, coupling_per_s=1e8)
    small = multiphonon_lifetime(tau, 300.0, gap_cm=3000.0, coupling_per_s=1e8)
    small_hot = multiphonon_lifetime(tau, 450.0, gap_cm=3000.0, coupling_per_s=1e8)
    assert abs(big - tau) / tau < 0.01                 # large gap ~ radiative (energy-gap law)
    assert small < 0.9 * tau                           # small gap quenched
    assert small_hot < small and big > small           # T lowers tau; larger gap less quenched


# ============================ Phase 11: detector beat-noise spectra ============================

def _bn_amp(sig_W):
    return FiberAmplifier(ER, _edf(6.0), [Pump(120e-3, 0.980e-6, "fwd")],
                          [Signal(sig_W, 1.560e-6)], AseBand(1.52e-6, 1.575e-6, 20))


def test_beat_noise_nf_reduces_to_optical_nf():
    r = _bn_amp(1e-3).solve()                           # strong signal -> signal-spont dominated
    nf_lin, _, _ = noise_figure(r, 1.560e-6)
    bn = detection_noise(r, 1.560e-6, optical_bw_Hz=50e9, electrical_bw_Hz=10e9,
                         quantum_efficiency=1.0)
    assert bn.dominant_term == "sig-sp"
    assert abs(bn.nf_beat_dB - 10.0 * np.log10(nf_lin)) < 0.05


def test_beat_noise_term_crossover():
    lo = detection_noise(_bn_amp(1e-9).solve(), 1.560e-6, optical_bw_Hz=50e9, electrical_bw_Hz=10e9)
    hi = detection_noise(_bn_amp(1e-3).solve(), 1.560e-6, optical_bw_Hz=50e9, electrical_bw_Hz=10e9)
    assert lo.dominant_term == "sp-sp" and hi.dominant_term == "sig-sp"


def test_optical_filter_cuts_spont_spont():
    r = _bn_amp(1e-9).solve()
    wide = detection_noise(r, 1.560e-6, optical_bw_Hz=200e9, electrical_bw_Hz=10e9)
    narrow = detection_noise(r, 1.560e-6, optical_bw_Hz=25e9, electrical_bw_Hz=10e9)
    assert narrow.var_sp_sp < wide.var_sp_sp and narrow.snr_elec_dB > wide.snr_elec_dB


def test_electrical_snr_rises_with_signal():
    snr = [detection_noise(_bn_amp(p).solve(), 1.560e-6, optical_bw_Hz=50e9,
                           electrical_bw_Hz=10e9).snr_elec_dB for p in (1e-6, 1e-5, 1e-4)]
    assert snr[0] < snr[1] < snr[2]
