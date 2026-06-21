"""Fast unit tests for the QD-SOA package (the rigorous multi-gate physics lives in
validation/qd_soa_gain_core.py and validation/qd_soa_traveling_wave.py)."""
import numpy as np
import pytest

from dynameta.optics.soa import (QDGainModel, QDGainParams, TravelingWaveSOA,
                                 TwoLevelSaturableGain, UltrafastCompression,
                                 agrawal_olsson_output)


def test_params_validation():
    with pytest.raises(ValueError):
        QDGainParams(N_q_m3=-1.0)
    with pytest.raises(ValueError):
        QDGainParams(Gamma=1.5)


def test_transparency_sign():
    # gain is negative (absorbing) below transparency, positive above; single group so
    # g0 = 0 exactly at rho_GS = 1/2
    m = QDGainModel(QDGainParams(n_groups=1).with_detailed_balance_taus())
    g_lo = m.small_signal_gain_per_m(0.5e-3)
    g_hi = m.small_signal_gain_per_m(20.0e-3)
    assert g_lo < 0.0 < g_hi


def test_conservation_closure():
    # the rhs_fields internal transitions cancel: d(n_tot)/dt == injection - recomb - stim
    m = QDGainModel(QDGainParams(n_groups=5).with_detailed_balance_taus())
    ng = m.ng
    rng = np.random.default_rng(1)
    N_w = np.array([6.0e23])
    rES = (0.2 + 0.5 * rng.random(ng))[None, :]
    rGS = (0.1 + 0.7 * rng.random(ng))[None, :]
    I, S, nu0 = 2.0e-2, 2.0e21, m.p.nu0_Hz
    dNw, dES, dGS = m.rhs_fields(N_w, rES, rGS, I, S, nu0)
    dn = dNw[0] + m.p.N_q_m3 * np.sum(m.w_j * (m.p.mu_ES * dES[0] + m.p.mu_GS * dGS[0]))
    L = m._lorentzian(nu0 - m.nu_j)
    stim = m.p.N_q_m3 * np.sum(m.w_j * m.p.mu_GS * m.p.v_g_m_s * m.p.sigma_pk_m2 * L
                              * (2.0 * rGS[0] - 1.0) * S)
    spont = m.p.N_q_m3 * np.sum(m.w_j * (m.p.mu_ES * rES[0] ** 2 + m.p.mu_GS * rGS[0] ** 2)) \
        / m.p.tau_sp_s
    recomb = m.p.B_wl_m3_s * N_w[0] ** 2 + m.p.C_wl_m6_s * N_w[0] ** 3
    expected = I / (1.602176634e-19 * m.p.V_a_m3) - recomb - spont - stim
    assert abs(dn - expected) / abs(expected) < 1e-9


def test_traveling_wave_matches_agrawal_olsson():
    # the distributed two-level engine reduces to the analytic lumped saturable-gain law
    g0, L, tau_c, E_sat, vg = 2300.0, 0.5e-3, 200e-12, 2.0e-12, 8.5e7
    tl = TwoLevelSaturableGain(g0_per_m=g0, tau_c_s=tau_c, E_sat_J=E_sat, v_g_m_s=vg)
    soa = TravelingWaveSOA(tl, L, 60, nu_s_Hz=1.934e14)
    dt = soa.dt
    nt = int(400e-12 / dt)
    t = np.arange(nt) * dt
    P_in = 0.05 * np.exp(-0.5 * ((t - 180e-12) / 25e-12) ** 2)
    r = soa.amplify(P_in, drive=None)
    ref = agrawal_olsson_output(t, P_in, g0, L, tau_c, E_sat)
    assert abs(r["P_out"].max() - ref.max()) / ref.max() < 5e-3


def test_zero_input_no_spurious_gain():
    m = QDGainModel(QDGainParams(n_groups=1).with_detailed_balance_taus())
    soa = TravelingWaveSOA(m, 0.5e-3, 40, nu_s_Hz=m.p.nu0_Hz)
    r = soa.amplify(np.zeros(300), drive=30.0e-3)
    assert r["P_out"][-1] < 1e-12


def test_coherent_reduces_to_power():
    # alpha=0, single real CW tone: |A_out|^2 from the coherent march == power P_out
    m = QDGainModel(QDGainParams(n_groups=1).with_detailed_balance_taus())
    soa = TravelingWaveSOA(m, 0.5e-3, 40, nu_s_Hz=m.p.nu0_Hz)
    n = 800
    co = soa.amplify_coherent(np.full(n, np.sqrt(1e-3)), drive=40e-3, alpha_lef=0.0)
    pw = soa.amplify(np.full(n, 1e-3), drive=40e-3)
    assert abs(co["P_out"][-1] - pw["P_out"][-1]) / pw["P_out"][-1] < 1e-12


def test_nsp_and_noise_figure_limits():
    from dynameta.optics.soa import inversion_factor_nsp, noise_figure
    assert abs(inversion_factor_nsp(1.0) - 1.0) < 1e-12        # full inversion
    assert inversion_factor_nsp(0.5001) > 1e3                  # near transparency
    nf = noise_figure(1.0e3, 1.0)                              # high gain, full inversion
    assert abs(10.0 * np.log10(nf) - 3.01) < 0.05             # 3 dB quantum limit
    assert noise_figure(100.0, 1.5, Gamma_g_per_m=300.0, alpha_i_per_m=60.0) > \
        noise_figure(100.0, 1.5)                               # internal loss degrades NF


def test_ase_reduces_to_analytic():
    from dynameta.optics.soa import ase_output_psd, inversion_factor_nsp
    HNU = 2.0 * np.pi * 1.054571817e-34 * 1.934e14
    rho, Nz, L, Gamma, g = 0.95, 300, 1e-3, 0.06, 6000.0
    S = ase_output_psd(np.full(Nz, g), np.full(Nz, rho), L / Nz, 1.934e14, Gamma, m_pol=1)
    G = np.exp(Gamma * g * L)
    assert abs(S - inversion_factor_nsp(rho) * HNU * (G - 1.0)) / S < 2e-2


def test_sndr_has_interior_optimum():
    # synthetic compressing transfer curve + a fixed noise floor -> SNDR peaks at an interior
    # drive (noise-limited below, distortion-limited above)
    from dynameta.optics.soa import sndr_vs_drive
    P_in = np.logspace(-5, -1, 200)
    P_out = P_in * 1000.0 / (1.0 + P_in / 5e-3)               # gain compresses above ~5 mW
    # a realistic (non-negligible) noise floor so the low-drive end is noise-limited
    sndr, eno, iopt = sndr_vs_drive(P_in, P_out, lambda Po: 1e-6,
                                    np.logspace(-4.5, -2.3, 24), mod_index=0.3)
    assert 0 < iopt < sndr.size - 1 and sndr[iopt] > sndr[0] and sndr[iopt] > sndr[-1]


def test_ultrafast_off_switch_and_compression():
    m = QDGainModel(QDGainParams(n_groups=1).with_detailed_balance_taus())
    soa = TravelingWaveSOA(m, 0.6e-3, 40, nu_s_Hz=m.p.nu0_Hz)
    P = np.full(1500, 5e-3)
    base = soa.amplify(P, 40e-3)
    off = soa.amplify(P, 40e-3, ultrafast=UltrafastCompression())          # eps=0
    assert np.array_equal(base["P_out"], off["P_out"])                     # byte-identical
    uf = UltrafastCompression(eps_shb_m3=8e-23, eps_ch_m3=1.2e-22)
    on = soa.amplify(P, 40e-3, ultrafast=uf)
    assert on["P_out"][-1] < base["P_out"][-1]                             # extra compression


def test_predistortion_linearizes():
    from dynameta.optics.soa import predistort
    P_in = np.logspace(-5, -1, 300)
    P_out = P_in * 1000.0 / (1.0 + P_in / 5e-3)                            # compressing curve
    targets = np.linspace(P_out[20], P_out[-20], 9)
    req = predistort(P_in, P_out, targets)
    achieved = req * 1000.0 / (1.0 + req / 5e-3)
    assert np.max(np.abs(achieved - targets)) / (targets[-1] - targets[0]) < 1e-2


def test_thermal_budget_and_sfdr():
    from dynameta.optics.soa import pattern_penalty_dB, sfdr_dB, thermal_drift_budget_K
    assert thermal_drift_budget_K(8, 0.02) < thermal_drift_budget_K(4, 0.02)   # tighter w/ bits
    assert pattern_penalty_dB([1.0, 0.9]) > 0.0
    P_in = np.logspace(-5, -1, 300)
    P_out = P_in * 1000.0 / (1.0 + P_in / 5e-3)
    assert np.isfinite(sfdr_dB(P_in, P_out, 1e-3, 1e-9))


def test_line_filter_off_is_flat_engine():
    # line_filter=False (default) is the flat-gain branch verbatim -> reduces to the power engine
    m = QDGainModel(QDGainParams(n_groups=21).with_detailed_balance_taus())
    soa = TravelingWaveSOA(m, 0.5e-3, 40, nu_s_Hz=m.p.nu0_Hz)
    n = 800
    co = soa.amplify_coherent(np.full(n, np.sqrt(1e-3)), drive=40e-3, alpha_lef=0.0,
                              line_filter=False)
    pw = soa.amplify(np.full(n, 1e-3), drive=40e-3)
    assert abs(co["P_out"][-1] - pw["P_out"][-1]) / pw["P_out"][-1] < 1e-12


def test_line_filter_spectral_gain_matches_analytic():
    # a weak detuned CW probe sees per-tone gain == the analytic Lorentzian ensemble (the flat
    # engine would give g(nu_s) for every tone)
    m = QDGainModel(QDGainParams(n_groups=41).with_detailed_balance_taus())
    nu0, L, Nz = m.p.nu0_Hz, 0.6e-3, 60
    soa = TravelingWaveSOA(m, L, Nz, nu_s_Hz=nu0)
    rho = m.rho_GS(m.steady_state(40e-3))
    f = 150e9
    t = np.arange(int(1.2e-9 / soa.dt)) * soa.dt
    a = soa.amplify_coherent(1e-4 * np.exp(-1j * 2 * np.pi * f * t), drive=40e-3, alpha_lef=0.0,
                             line_filter=True)["A_out"][int(0.9 * t.size):]
    G_num = 20.0 * np.log10(np.abs(a).mean() / 1e-4)
    G_an = 10.0 * np.log10(np.exp(m.p.Gamma * m.material_gain_per_m(rho, nu0 + f) * L))
    assert abs(G_num - G_an) < 0.02


def test_line_filter_requires_spectral_model():
    tl = TwoLevelSaturableGain(g0_per_m=2000.0, tau_c_s=200e-12, E_sat_J=2e-12)
    soa = TravelingWaveSOA(tl, 0.5e-3, 40, nu_s_Hz=1.934e14)
    with pytest.raises(ValueError):
        soa.amplify_coherent(np.full(50, 1e-3) + 0j, drive=None, line_filter=True)


def test_gvd_off_is_byte_identical():
    # beta2=0 (default) leaves the coherent field branch unchanged
    tl = TwoLevelSaturableGain(g0_per_m=1500.0, tau_c_s=200e-12, E_sat_J=2e-12)
    soa = TravelingWaveSOA(tl, 0.5e-3, 64, nu_s_Hz=1.934e14)
    t = np.arange(400) * soa.dt
    A = (np.exp(-((t - t.mean()) ** 2) / (2 * (8 * soa.dt) ** 2)) + 0j)
    base = soa.amplify_coherent(A, drive=None, alpha_lef=0.0)["A_out"]
    z0 = soa.amplify_coherent(A, drive=None, alpha_lef=0.0, beta2_s2_per_m=0.0)["A_out"]
    assert np.array_equal(base, z0)


def test_gvd_broadens_gaussian_per_nlse():
    # gain-free Gaussian broadens to T0 sqrt(1+(L/L_D)^2), L_D = T0^2/|beta2| (Agrawal ch.3)
    tl = TwoLevelSaturableGain(g0_per_m=0.0, tau_c_s=1e-9, E_sat_J=1e-12)
    L, nz = 1.0e-3, 256
    soa = TravelingWaveSOA(tl, L, nz)
    nt = 4 * nz
    t = np.arange(nt) * soa.dt
    W = nz * soa.dt
    T0 = W / 16.0
    beta2 = T0 * T0 / L                                    # L_D = L
    A = np.exp(-((t - (nt // 2) * soa.dt) ** 2) / (2 * T0 * T0)) + 0j
    out = soa.amplify_coherent(A, drive=None, beta2_s2_per_m=beta2)["A_out"]

    def rms(w):
        m1 = (t * w).sum() / w.sum()
        return np.sqrt((t * t * w).sum() / w.sum() - m1 * m1)
    ratio = rms(np.abs(out) ** 2) / rms(np.abs(A) ** 2)
    assert abs(ratio - np.sqrt(2.0)) / np.sqrt(2.0) < 1e-2     # L/L_D = 1 -> sqrt(2)
    # unitary: gain-free dispersion conserves the pulse energy
    assert abs((np.abs(out) ** 2).sum() - (np.abs(A) ** 2).sum()) / (np.abs(A) ** 2).sum() < 1e-9


def test_gvd_segments_reduction_and_guard():
    # gvd_segments=1 == the single device-scale split; gain-free is S-invariant; bad S raises
    tl = TwoLevelSaturableGain(g0_per_m=0.0, tau_c_s=1e-9, E_sat_J=1e-12)
    L, nz = 1.0e-3, 128
    soa = TravelingWaveSOA(tl, L, nz)
    nt = 3 * nz
    t = np.arange(nt) * soa.dt
    T0 = nz * soa.dt / 16.0
    beta2 = T0 * T0 / L
    A = np.exp(-((t - (nt // 2) * soa.dt) ** 2) / (2 * T0 * T0)) + 0j
    base = soa.amplify_coherent(A, drive=None, beta2_s2_per_m=beta2)["A_out"]
    s1 = soa.amplify_coherent(A, drive=None, beta2_s2_per_m=beta2, gvd_segments=1)["A_out"]
    s4 = soa.amplify_coherent(A, drive=None, beta2_s2_per_m=beta2, gvd_segments=4)["A_out"]
    assert np.array_equal(base, s1)                           # S=1 == single split
    # gain-free: dispersion commutes with the delay -> S-invariant up to the causal device-fill
    # truncation at segment boundaries (machine in a well-contained window; see the validation)
    assert np.max(np.abs(s4 - s1)) < 1e-6
    with pytest.raises(ValueError):
        soa.amplify_coherent(A, drive=None, beta2_s2_per_m=beta2, gvd_segments=7)   # 7 does not divide 128


def test_alpha_density_dependence():
    # slope=0 -> scalar alpha, byte-identical coherent engine; slope!=0 -> per-slice array that
    # shifts alpha away from alpha_lef with inversion
    m0 = QDGainModel(QDGainParams(n_groups=21).with_detailed_balance_taus())
    assert m0.alpha_lef_slices(m0.init_slices(20, 40e-3)) == m0.p.alpha_lef    # scalar
    soa0 = TravelingWaveSOA(m0, 0.5e-3, 40, nu_s_Hz=m0.p.nu0_Hz)
    A = np.full(600, 1e-4) + 0j
    assert np.array_equal(soa0.amplify_coherent(A, 40e-3)["A_out"],
                          soa0.amplify_coherent(A, 40e-3, alpha_lef=2.0)["A_out"])
    md = QDGainModel(QDGainParams(n_groups=21, alpha_lef_density_slope=4.0).with_detailed_balance_taus())
    al = md.alpha_lef_slices(md.init_slices(20, 40e-3))
    assert np.ndim(al) == 1 and np.all(al > md.p.alpha_lef)    # alpha rises with inversion (slope>0)


def test_pdg_reduction_and_ratio():
    # pdg_ratio=1 with one pol dark reduces to single-pol; small-signal PDG = (1-r) Gamma g L
    m = QDGainModel(QDGainParams(n_groups=21).with_detailed_balance_taus())
    L = 1.0e-3
    soa = TravelingWaveSOA(m, L, 80, nu_s_Hz=m.p.nu0_Hz)
    nt, eps = 2000, 1e-4
    Aw = np.full(nt, eps) + 0j
    single = soa.amplify_coherent(Aw, 60e-3, alpha_lef=0.0)["A_out"]
    dz = soa.amplify_coherent_dualpol(Aw, np.zeros(nt) + 0j, 60e-3, alpha_lef=0.0, pdg_ratio=1.0)
    assert np.max(np.abs(dz["A_te_out"] - single)) < 1e-15    # TE-only, r=1 == single-pol
    r = 0.5
    d2 = soa.amplify_coherent_dualpol(Aw, Aw, 60e-3, alpha_lef=0.0, pdg_ratio=r)
    pdg = 10 * np.log10((np.abs(d2["A_te_out"][-1]) / np.abs(d2["A_tm_out"][-1])) ** 2)
    g = float(m.material_gain_per_m(m.rho_GS(m.steady_state(60e-3)), m.p.nu0_Hz))
    assert abs(pdg - (1 - r) * m.p.Gamma * g * L * 10 / np.log(10)) / pdg < 0.05


def test_fabry_perot_reduction_and_ripple():
    # R=0 reduces to single-pass; the ripple over a phase sweep matches the Saitoh-Mukai metric
    from dynameta.optics.soa.metrics import facet_gain_ripple_dB
    m = QDGainModel(QDGainParams(n_groups=21).with_detailed_balance_taus())
    soa = TravelingWaveSOA(m, 1.5e-3, 100, nu_s_Hz=m.p.nu0_Hz)
    nt, eps = 3000, 1e-5
    A = np.full(nt, eps) + 0j
    sp = soa.amplify_coherent(A, 80e-3, alpha_lef=0.0)["A_out"]
    fp0 = soa.amplify_fabry_perot(A, 80e-3, R1=0.0, R2=0.0, alpha_lef=0.0)["A_out"]
    assert np.max(np.abs(fp0 - sp)) < 1e-15                    # R=0 == single-pass
    Gsp = np.abs(sp[-1]) ** 2 / eps ** 2
    R = 3e-3
    gains = [10 * np.log10(np.abs(soa.amplify_fabry_perot(A, 80e-3, R1=R, R2=R, alpha_lef=0.0,
             roundtrip_phase=p)["A_out"][-1]) ** 2 / eps ** 2) for p in np.linspace(0, 2 * np.pi, 13)]
    assert abs((max(gains) - min(gains)) - facet_gain_ripple_dB(Gsp, R, R)) < 0.02
    with pytest.raises(ValueError):
        soa.amplify_fabry_perot(A, 80e-3, R1=1.0, R2=0.0)      # R>=1 invalid


def test_ase_zresolved_reduction_and_profile():
    # OFF == frozen transport; ON gives a non-uniform S_ase(z) anti-correlated with the local gain
    from dynameta.optics.soa import (ase_self_consistent_zresolved, ase_spectrum_bidirectional)
    # (small grid here; the full convergence/correlation rigor lives in validation/qd_soa_ase_zresolved)
    m = QDGainModel(QDGainParams(n_groups=9).with_detailed_balance_taus())
    nu0 = m.p.nu0_Hz
    nu = np.linspace(nu0 - 3e12, nu0 + 3e12, 9)
    dnu = np.gradient(nu)
    L, I, nz = 1.5e-3, 80e-3, 8
    off = ase_self_consistent_zresolved(m, I, 0.0, nu0, nu, dnu, L, n_slices=nz,
                                        ase_saturation=False, m_pol=2)
    y = m.steady_state(I, S_conf_m3=0.0, nu_s_Hz=nu0)
    g = m.material_gain_per_m(m.rho_GS(y), nu)
    gsp = m.emission_gain_per_m(m.rho_GS(y), nu)
    ref = ase_spectrum_bidirectional(np.tile(g, (nz, 1)), np.tile(gsp, (nz, 1)), L / nz, nu, dnu,
                                     m.p.Gamma, m_pol=2)
    assert np.max(np.abs(off["S_f"] - ref["S_f"])) < 1e-280 + 1e-13 * np.max(ref["S_f"])
    on = ase_self_consistent_zresolved(m, I, 0.0, nu0, nu, dnu, L, n_slices=nz, ase_saturation=True,
                                       ase_strength=300.0, m_pol=2, beta=0.6, max_iter=60)
    Sz = on["S_ase_z"]
    gpk = on["g_sat_z"][:, np.argmin(np.abs(nu - nu0))]
    assert on["converged"] and (Sz.max() - Sz.min()) / Sz.mean() > 0.1   # real z-profile
    assert np.corrcoef(Sz, gpk)[0, 1] < -0.99                            # gain low where ASE high


def test_many_body_gain_reduction_and_bgr():
    from dynameta.optics.soa import ManyBody
    m0 = QDGainModel(QDGainParams(n_groups=41).with_detailed_balance_taus())
    nu0 = m0.p.nu0_Hz
    nu = np.linspace(nu0 - 8e12, nu0 + 8e12, 400)
    rho = np.full(41, 0.9)
    # enabled with zero corrections == free-carrier
    mr = QDGainModel(QDGainParams(n_groups=41).with_detailed_balance_taus(),
                     many_body=ManyBody(enabled=True, bgr_coeff=0.0, gamma_eid_Hz=0.0,
                                        coulomb_enh=0.0))
    g_mb, gi = mr.material_gain_index_mb(rho, nu, 1e24)
    assert np.max(np.abs(g_mb - m0.material_gain_per_m(rho, nu))) < 1e-9 * np.max(np.abs(g_mb)) + 1e-9
    # BGR red-shifts the peak by the analytic amount
    nc = np.linspace(nu0 - 2e13, nu0 + 5e12, 4000)
    mB = QDGainModel(QDGainParams(n_groups=41).with_detailed_balance_taus(),
                     many_body=ManyBody(enabled=True, exciton_rydberg_meV=12.0, exciton_bohr_nm=12.0,
                                        bgr_coeff=1.9))
    gN, _ = mB.material_gain_index_mb(rho, nc, 1e24)
    meas = nc[np.argmax(gN)] - nu0
    assert abs(meas - mB._mb_bgr_shift_Hz(1e24)) / abs(mB._mb_bgr_shift_Hz(1e24)) < 0.05
    with pytest.raises(ValueError):
        ManyBody(enabled=True, exciton_rydberg_meV=-1.0)              # guard


def test_langevin_reduction_and_statistics():
    m = QDGainModel(QDGainParams(n_groups=15).with_detailed_balance_taus())
    soa = TravelingWaveSOA(m, 1.0e-3, 60, nu_s_Hz=m.p.nu0_Hz)
    A0 = np.zeros(8000) + 0j
    det = soa.amplify_coherent(A0, 60e-3, alpha_lef=0.0)["A_out"]
    off = soa.amplify_coherent(A0, 60e-3, alpha_lef=0.0, langevin=False)["A_out"]
    assert np.array_equal(det, off)                                  # off == deterministic
    r1 = soa.amplify_coherent(A0, 60e-3, alpha_lef=0.0, langevin=True, seed=3)["A_out"]
    r2 = soa.amplify_coherent(A0, 60e-3, alpha_lef=0.0, langevin=True, seed=3)["A_out"]
    assert np.array_equal(r1, r2)                                    # seed reproducible
    Ii = np.abs(r1[60 + 200:]) ** 2
    assert abs(np.mean(Ii ** 2) / np.mean(Ii) ** 2 - 2.0) < 0.15     # complex-Gaussian ASE intensity


def test_transport_reduction_and_profile():
    m = QDGainModel(QDGainParams(n_groups=15).with_detailed_balance_taus())
    soa = TravelingWaveSOA(m, 0.5e-3, 40, nu_s_Hz=m.p.nu0_Hz)
    nt = 3000
    P = np.full(nt, 1e-5)
    base = soa.amplify(P, 40e-3)["P_out"]
    assert np.array_equal(soa.amplify(P, 40e-3, transport_tau_s=0.0)["P_out"], base)   # tau=0 == lumped
    assert np.array_equal(soa.amplify(P, np.full(40, 40e-3))["P_out"], base)           # uniform prof == scalar
    # transport leaves DC gain invariant
    g0 = soa.amplify(P, 40e-3, transport_tau_s=0.0, return_traces=True)["g_zt"][-1, 20]
    gt = soa.amplify(P, 40e-3, transport_tau_s=300e-12, return_traces=True)["g_zt"][-1, 20]
    assert abs(gt - g0) / abs(g0) < 1e-6
    # DD injection profile -> non-uniform gain
    st = soa.amplify(np.full(1500, 1e-5), np.linspace(20e-3, 60e-3, 40))["state"]
    gz = m.gain_per_m_slices(st, m.p.nu0_Hz)
    assert np.all(np.diff(gz) > 0)                                    # follows the ramp


def test_thermal_profile_reduction_and_coupling():
    from dynameta.optics.soa import (SelfHeating, dome_analytic, sample_T_along_axis,
                                     thermal_profile_steady_1d)
    nz, L, T0, Rp, kA = 40, 1e-3, 300.0, 5e-4, 2e-5
    dz = L / nz
    q = np.full(nz, 2e4)
    # kappaA=0, insulated -> lumped T = T0 + q Rth' per slice
    Tl = thermal_profile_steady_1d(q, dz, 0.0, Rp, T0, ends="insulated")
    assert np.max(np.abs(Tl - (T0 + q * Rp))) < 1e-9
    # sunk-facet dome == analytic cosh (node grid)
    dzn = L / (nz - 1)
    zn = np.arange(nz) * dzn
    Tn = thermal_profile_steady_1d(q, dzn, kA, Rp, T0, ends="sunk")
    assert np.max(np.abs(Tn - dome_analytic(q[0], L, kA, Rp, T0, zn))) / (Tn.max() - T0) < 1e-2
    # per-slice gain coupling: T0 -> exact reduction; hot T(z) -> lower local gain
    sh = SelfHeating(Rth_K_W=50.0, dnu0_dT_Hz_K=20e9, dg_dT_frac_per_K=-0.01, T0_K=T0)
    m = QDGainModel(QDGainParams(n_groups=15).with_detailed_balance_taus(), self_heating=sh)
    st = m.init_slices(nz, 40e-3)
    nu0 = m.p.nu0_Hz
    assert np.array_equal(m.gain_per_m_thermal(st, nu0, np.full(nz, T0)),
                          m.gain_per_m_slices(st, nu0))
    Tdome = thermal_profile_steady_1d(np.full(nz, 3e4), dz, kA, Rp, T0, ends="sunk")
    g = m.gain_per_m_thermal(st, nu0, Tdome)
    assert g[nz // 2] < g[0]
    # uniform-hot T(z) reduces to the lumped set_temperature gain
    g_hot = m.gain_per_m_thermal(st, nu0, np.full(nz, 330.0))
    m.set_temperature(330.0)
    assert np.max(np.abs(g_hot - m.gain_per_m_slices(st, nu0))) < 1e-9
    m.set_temperature(T0)
    # external-FEM sampling seam
    Tcb = sample_T_along_axis(lambda x, y, z: T0 + 10.0 * z / L, (np.arange(nz) + 0.5) * dz, axis="z")
    assert np.all(np.isfinite(m.gain_per_m_thermal(st, nu0, Tcb)))
    # ES-band thermal coupling: with the ES band active the thermal gain INCLUDES it and still
    # reduces exactly to gain_per_m_slices at T0 (and to set_temperature for uniform-hot)
    mes = QDGainModel(QDGainParams(n_groups=15, sigma_pk_ES_m2=1e-19).with_detailed_balance_taus(),
                      self_heating=sh)
    ste = mes.init_slices(nz, 40e-3)
    assert np.array_equal(mes.gain_per_m_thermal(ste, nu0, np.full(nz, T0)),
                          mes.gain_per_m_slices(ste, nu0))
    ghe = mes.gain_per_m_thermal(ste, nu0, np.full(nz, 330.0))
    mes.set_temperature(330.0)
    assert np.max(np.abs(ghe - mes.gain_per_m_slices(ste, nu0))) < 1e-9
    # transient 1-D thermal: t->inf == steady; lumped RC charge-up == analytic exp
    from dynameta.optics.soa import thermal_profile_transient_1d
    Cl, qa, Rpr, kAr = 1e-3, np.full(nz, 2e4), 5e-4, 2e-5
    tau = Cl * Rpr
    Ttr = thermal_profile_transient_1d(qa, dz, kAr, Rpr, T0, Cl, tau / 50, 4000, ends="sunk")
    assert np.max(np.abs(Ttr - thermal_profile_steady_1d(qa, dz, kAr, Rpr, T0, ends="sunk"))) < 1e-6
    Hh = thermal_profile_transient_1d(qa, dz, 0.0, Rpr, T0, Cl, tau / 200, 400, ends="insulated",
                                      return_history=True)
    th = np.arange(Hh.shape[0]) * (tau / 200)
    assert np.max(np.abs(Hh[:, nz // 2] - (T0 + qa[0] * Rpr * (1 - np.exp(-th / tau))))) / (qa[0] * Rpr) < 1e-2


def test_nonlinear_loss_tpa_fca():
    from dynameta.optics.soa import NonlinearLoss
    m = QDGainModel(QDGainParams(n_groups=15).with_detailed_balance_taus())
    nu = m.p.nu0_Hz
    gam = m.gamma_confinement
    A_eff = m.p.A_mode_m2
    L, nz, I = 1.0e-3, 200, 40e-3
    soa = TravelingWaveSOA(m, L, nz, nu_s_Hz=nu, alpha_i_per_m=300.0)
    nt = 3 * nz
    Pin = np.full(nt, 1e-7)
    base = soa.amplify(Pin, I)["P_out"]
    # byte-identical default
    assert np.array_equal(base, soa.amplify(Pin, I, nl_loss=None)["P_out"])
    assert np.array_equal(base, soa.amplify(Pin, I, nl_loss=NonlinearLoss(0.0, 0.0, 0.0))["P_out"])
    # FCA exact: ratio == exp(-sigma Nw L)
    Nw = m.wl_density_slices(m.init_slices(nz, I))[0]
    sig = 5e-21
    ratio = soa.amplify(Pin, I, nl_loss=NonlinearLoss(0.0, sig, 0.0))["P_out"][-1] / base[-1]
    assert abs(ratio - np.exp(-sig * Nw * L)) / np.exp(-sig * Nw * L) < 1e-6
    # TPA passive Bernoulli at transparency
    def g_of_I(Ix):
        return m.gain_per_m_slices(m.init_slices(2, Ix), nu)[0]
    lo, hi = 1e-4, 40e-3
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if g_of_I(mid) < 0 else (lo, mid)
    I_tr = 0.5 * (lo + hi)
    alpha_i, beta, P0 = 100.0, 8e-11, 0.5
    s = TravelingWaveSOA(m, L, nz, nu_s_Hz=nu, alpha_i_per_m=alpha_i)
    tpa = s.amplify(np.full(nt, P0), I_tr, nl_loss=NonlinearLoss(beta, 0.0, A_eff))["P_out"][-1]
    a, b = -alpha_i, beta / A_eff
    P_an = a * P0 * np.exp(a * L) / (a + b * P0 * (np.exp(a * L) - 1.0))
    assert abs(tpa - P_an) / P_an < 1e-3
    # passivity: nonlinear loss never increases output
    assert tpa <= s.amplify(np.full(nt, P0), I_tr)["P_out"][-1]


def test_carrier_leakage():
    from dynameta.optics.soa import Leakage
    from dynameta.optics.soa.qd_gain import KB, Q_E
    pf = lambda: QDGainParams(n_groups=15).with_detailed_balance_taus()
    nu = pf().nu0_Hz
    I = 40e-3
    lk = Leakage(tau_leak0_s=5e-12, E_barrier_eV=0.10)
    m0 = QDGainModel(pf())
    m = QDGainModel(pf(), leakage=lk)
    # disabled Leakage(0) byte-identical to None
    md = QDGainModel(pf(), leakage=Leakage(tau_leak0_s=0.0))
    assert np.array_equal(m0.steady_state(I), md.steady_state(I))
    # Arrhenius rate exact
    assert abs(lk.rate_at(300.0) - np.exp(-0.10 * Q_E / (KB * 300.0)) / 5e-12) / lk.rate_at(300.0) < 1e-12
    assert abs(lk.rate_at(340.0) / lk.rate_at(300.0)
               - np.exp(-0.10 * Q_E / KB * (1 / 340. - 1 / 300.))) < 1e-12
    # term-level exactness: dN_w(leak) - dN_w(no) == -leak_rate N_w
    st = m0.init_slices(3, I)
    d_no = m0.rhs_fields(st[0], st[1], st[2], I, 0.0, nu)[0]
    d_lk = m.rhs_fields(st[0], st[1], st[2], I, 0.0, nu)[0]
    assert np.max(np.abs((d_lk - d_no) - (-m._leak_rate() * st[0]))) < 1e-3
    # gain suppressed, more so with faster leakage
    g0 = m0.gain_per_m_slices(m0.init_slices(2, I), nu)[0]
    gL = m.gain_per_m_slices(m.init_slices(2, I), nu)[0]
    mf = QDGainModel(pf(), leakage=Leakage(tau_leak0_s=2e-12, E_barrier_eV=0.10))  # faster leak
    gf = mf.gain_per_m_slices(mf.init_slices(2, I), nu)[0]                          # its OWN state
    assert gf < gL < g0


def test_electrical_rc():
    m = QDGainModel(QDGainParams(n_groups=11).with_detailed_balance_taus())
    soa = TravelingWaveSOA(m, 0.5e-3, 40, nu_s_Hz=m.p.nu0_Hz)
    nt = 4000
    P = np.full(nt, 1e-5)
    base = soa.amplify(P, 40e-3, return_traces=True)["g_zt"]
    # rc=0 and constant-drive rc>0 both byte-identical
    assert np.array_equal(base, soa.amplify(P, 40e-3, rc_tau_s=0.0, return_traces=True)["g_zt"])
    assert np.array_equal(base, soa.amplify(P, 40e-3, rc_tau_s=100e-12, return_traces=True)["g_zt"])
    # RC filter pole exact: |H(fRC)| == 1/sqrt(2)
    tau, dtf, N = 100e-12, 1e-13, 100000
    tf = np.arange(N) * dtf
    fRC = 1 / (2 * np.pi * tau)
    sig = np.sin(2 * np.pi * fRC * tf)
    rc, a, out = sig[0], dtf / tau, np.empty(N)
    for n in range(N):
        rc = rc + a * (sig[n] - rc); out[n] = rc
    k = int(np.argmin(np.abs(np.fft.rfftfreq(N, dtf) - fRC)))
    assert abs(np.abs(np.fft.rfft(out)[k]) / np.abs(np.fft.rfft(sig)[k]) - 1 / np.sqrt(2)) < 1e-2
    # step delay: RC delays the gain rise
    Istep = np.where(np.arange(nt) < nt // 4, 30e-3, 55e-3)
    def t50(rc_tau):
        g = soa.amplify(P, Istep, rc_tau_s=rc_tau, return_traces=True)["g_zt"][soa.nz:, soa.nz // 2]
        return np.argmax(g >= 0.5 * (g[0] + g[-1]))
    assert t50(100e-12) > t50(0.0)


def test_hammer():
    """Thin wrapper -- run the exhaustive cross-cutting hammer validation in the pytest suite."""
    import importlib
    ham = importlib.import_module("validation.qd_soa_hammer")
    assert ham.main() is True


def test_saturation_power():
    """Thin wrapper -- absolute P_sat (mW/dBm) + detuning + eta_in fiber-to-fiber NF validation."""
    import importlib
    sat = importlib.import_module("validation.qd_soa_saturation_power")
    assert sat.main() is True


def test_wdm():
    """Thin wrapper -- WDM multi-channel wavelength-resolved cross-gain saturation validation."""
    import importlib
    wdm = importlib.import_module("validation.qd_soa_wdm")
    assert wdm.main() is True


def test_filament_qd():
    """Thin wrapper -- QD-coupled transverse filamentation (real QD gain in the 2-D BPM) validation."""
    import importlib
    fil = importlib.import_module("validation.qd_soa_filament_qd")
    assert fil.main() is True


def test_maxwell_bloch():
    """Thin wrapper -- coherent Maxwell-Bloch (Rabi, photon echo, pulse-area) validation."""
    import importlib
    mb = importlib.import_module("validation.qd_soa_maxwell_bloch")
    assert mb.main() is True


def test_calibration_innolume():
    """Thin wrapper -- static/CW calibration to the Innolume BOA1310060 datasheet."""
    import importlib
    cal = importlib.import_module("validation.qd_soa_calibration_innolume")
    assert cal.main() is True


def test_inferred_dynamics():
    """Thin wrapper -- dynamic parameters inferred from the CW calibration (flagged estimates)."""
    import importlib
    inf = importlib.import_module("validation.qd_soa_inferred_dynamics")
    assert inf.main() is True


def test_nonmarkovian_lineshape():
    from dynameta.optics.soa.lineshape import (biexp_memory_kernel, lorentzian_area,
                                               nonmarkovian_lineshape)
    g1, g2, w1 = 20e9, 80e9, 0.6
    N, dt = 16384, 2e-9 / 16384
    t = (np.arange(N) - N // 2) * dt
    M = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(biexp_memory_kernel(t, g1, g2, w1)))) * dt
    f = np.fft.fftshift(np.fft.fftfreq(N, dt))
    Lan = nonmarkovian_lineshape(f, g1, g2, w1)
    sel = np.abs(f) < 4 * g2
    assert np.max(np.abs(M.real[sel] - Lan[sel])) / Lan.max() < 1e-3      # Wiener-Khinchin
    assert np.allclose(nonmarkovian_lineshape(f, g1, g2, 1.0), lorentzian_area(f, g1))  # Markovian
    # model gain reduces to gain_per_m_slices (GS) at w1=1
    m = QDGainModel(QDGainParams(n_groups=15).with_detailed_balance_taus())
    st = m.init_slices(4, 40e-3)
    nu0 = m.p.nu0_Hz
    assert np.max(np.abs(m.gain_per_m_nonmarkovian(st, nu0, gamma2_factor=3.0, w1=1.0)
                         - m.gain_per_m_slices(st, nu0))) < 1e-12


def test_reduced_sbe():
    from dynameta.optics.soa.sbe import reduced_sbe_susceptibility, sbe_gain_per_m
    hw = np.linspace(0.90, 1.12, 200)
    _, chi0 = reduced_sbe_susceptibility(hw, coulomb_V0=0.0, nk=160)
    _, chiC = reduced_sbe_susceptibility(hw, coulomb_V0=3e-29, nk=160)
    g0 = sbe_gain_per_m(hw, chi0)
    gC = sbe_gain_per_m(hw, chiC)
    assert gC.max() > g0.max()                                            # Coulomb enhancement
    assert g0.min() < 0.0 < g0.max()                                      # gain + absorption regions
    # unpumped only absorbs
    _, chiL = reduced_sbe_susceptibility(hw, coulomb_V0=0.0, N_2d_m2=1e12, nk=160)
    assert sbe_gain_per_m(hw, chiL).max() <= 1e-30


def test_vectorial_pdg():
    m = QDGainModel(QDGainParams(n_groups=21).with_detailed_balance_taus())
    soa = TravelingWaveSOA(m, 0.5e-3, 40, nu_s_Hz=m.p.nu0_Hz)
    nu0, fwhm = m.p.nu0_Hz, m.p.fwhm_inhom_Hz
    nt = 2000
    te = np.full(nt, np.sqrt(1e-6) + 0j)
    tm = np.full(nt, np.sqrt(1e-6) + 0j)
    # shift=0 byte-identical to the scalar-pdg dualpol
    a = soa.amplify_coherent_dualpol(te, tm, 40e-3, pdg_ratio=0.7)
    b = soa.amplify_coherent_dualpol(te, tm, 40e-3, pdg_ratio=0.7, tm_peak_shift_Hz=0.0)
    assert np.array_equal(a["A_tm_out"], b["A_tm_out"])
    # ratio=1, shift=0 -> degenerate
    d = soa.amplify_coherent_dualpol(te, tm, 40e-3, pdg_ratio=1.0, tm_peak_shift_Hz=0.0)
    assert abs(d["P_te_out"][-1] - d["P_tm_out"][-1]) / d["P_te_out"][-1] < 1e-12
    # frequency-dependent PDG reverses sign across the split
    def pdg(nu):
        o = soa.amplify_coherent_dualpol(te, tm, 40e-3, nu_s_Hz=nu, pdg_ratio=1.0,
                                         tm_peak_shift_Hz=0.5 * fwhm)
        return 10 * np.log10(o["P_te_out"][-1] / o["P_tm_out"][-1])
    assert pdg(nu0 - 0.5 * fwhm) > 0.05 > -0.05 > pdg(nu0 + 0.5 * fwhm)


def test_rin_and_linewidth():
    from dynameta.optics.soa import (henry_factor, linewidth_from_field, rin_spectrum,
                                     schawlow_townes_henry_linewidth)
    rng = np.random.default_rng(1)
    N, dt = 100000, 1e-12
    # RIN Parseval
    P = 1e-3 * (1 + 0.05 * rng.standard_normal(N))
    f, rin = rin_spectrum(P, dt)
    assert abs(np.trapezoid(rin, f) / (np.var(P) / P.mean() ** 2) - 1.0) < 1e-3
    # RIN sinusoid integral == m^2/2
    t = np.arange(N) * dt
    f2, rin2 = rin_spectrum(1e-3 * (1 + 0.1 * np.cos(2 * np.pi * 2e9 * t)), dt)
    assert abs(np.trapezoid(rin2, f2) - 0.005) / 0.005 < 1e-3
    # linewidth recovery + pure tone
    v = 1e-3
    dnu = linewidth_from_field(np.exp(1j * np.cumsum(np.sqrt(v) * rng.standard_normal(N))), dt)
    assert abs(dnu - v / (2 * np.pi * dt)) / (v / (2 * np.pi * dt)) < 0.05
    assert linewidth_from_field(np.exp(1j * 2 * np.pi * 1e9 * t), dt) < 1.0
    # Schawlow-Townes-Henry (1+alpha^2)
    a = schawlow_townes_henry_linewidth(1e12, 1e4, 3.0) / schawlow_townes_henry_linewidth(1e12, 1e4, 0.0)
    assert abs(a - henry_factor(3.0)) < 1e-12


def test_transverse_bpm():
    from dynameta.optics.soa import TransverseBPM
    lam, n0 = 1.3e-6, 3.4
    # pure diffraction == Gaussian w(z) law + unitary energy
    bd = TransverseBPM(400e-6, 2048, lam, n0, g0_per_m=0.0)
    w0 = 12e-6
    A = np.exp(-(bd.x / w0) ** 2) + 0j
    zR = np.pi * n0 * w0 ** 2 / lam
    od = bd.propagate(A, zR, 120)
    assert abs(bd.rms_width(od["I_out"]) - (w0 / 2) * np.sqrt(2)) / ((w0 / 2) * np.sqrt(2)) < 1e-3
    assert abs(od["I_out"].sum() / (np.abs(A) ** 2).sum() - 1.0) < 1e-9
    # uniform beam stays flat (lateral lumping limit)
    bu = TransverseBPM(100e-6, 128, lam, n0, g0_per_m=1500.0, Isat_W=5e-3, alpha_i_per_m=200.0)
    ou = bu.propagate(np.full(128, np.sqrt(1e-3) + 0j), 0.4e-3, 200)
    assert (ou["I_out"].max() - ou["I_out"].min()) / ou["I_out"].mean() < 1e-12
    # self-focusing: alpha>0 narrower than alpha=0
    def w_alpha(a):
        bc = TransverseBPM(300e-6, 1024, lam, n0, g0_per_m=1500.0, Isat_W=1e-3, alpha_lef=a)
        return bc.rms_width(bc.propagate(np.sqrt(3e-3) * np.exp(-(bc.x / 15e-6) ** 2) + 0j,
                                         0.4e-3, 300)["I_out"])
    assert w_alpha(3.0) < w_alpha(0.0)
    # lateral diffusion smooths the SHB gain hole
    def contrast(Ld):
        be = TransverseBPM(120e-6, 256, lam, n0, g0_per_m=1500.0, Isat_W=1e-3, L_diff_m=Ld)
        g = be.carrier_gain(np.sqrt(2e-3) * np.exp(-(be.x / 8e-6) ** 2) + 0j)
        return (g.max() - g.min()) / g.mean()
    assert contrast(8e-6) < contrast(0.0)
    # thermal lensing: dndt=0 byte-id; linear ramp steers exactly; hot centre focuses
    bl = TransverseBPM(400e-6, 2048, lam, n0, g0_per_m=0.0)
    Al = np.exp(-(bl.x / 20e-6) ** 2) + 0j
    assert np.array_equal(bl.propagate(Al, 0.3e-3, 150)["A_out"],
                          bl.propagate(Al, 0.3e-3, 150, T_profile_x=np.ones(2048), dndt_per_K=0.0)["A_out"])
    oL = bl.propagate(Al, 0.3e-3, 150, T_profile_x=1e5 * bl.x, dndt_per_K=3e-4)
    Fk = np.abs(np.fft.fft(oL["A_out"])) ** 2
    kxc = np.sum(bl.kx * Fk) / np.sum(Fk)
    assert abs(kxc - bl.k0 * 3e-4 * 1e5 * 0.3e-3) / abs(bl.k0 * 3e-4 * 1e5 * 0.3e-3) < 1e-9
    Tq = -0.5 * 5e12 * bl.x ** 2
    wn = bl.rms_width(bl.propagate(Al, 0.6e-3, 300)["I_out"])
    wf = bl.rms_width(bl.propagate(Al, 0.6e-3, 300, T_profile_x=Tq, dndt_per_K=2e-3)["I_out"])
    assert wf < wn


def test_carrier_leakage_numba_parity():
    from dynameta.optics.soa import Leakage
    from dynameta.optics.soa.qd_gain import _HAVE_NUMBA
    if not _HAVE_NUMBA:
        pytest.skip("numba not installed")
    lk = Leakage(tau_leak0_s=5e-12, E_barrier_eV=0.10)
    pf = lambda: QDGainParams(n_groups=21).with_detailed_balance_taus()
    mn = QDGainModel(pf(), leakage=lk, fast=True)
    mc = QDGainModel(pf(), leakage=lk)
    nu = pf().nu0_Hz
    a = mn.step_slices(mn.init_slices(6, 40e-3), 1e-4, 1e-13, nu, 40e-3)
    b = mc.step_slices(mc.init_slices(6, 40e-3), 1e-4, 1e-13, nu, 40e-3)
    assert np.max(np.abs(a[0] - b[0])) / np.max(np.abs(b[0])) < 1e-14


def test_numba_carrier_step_parity():
    from dynameta.optics.soa.qd_gain import _HAVE_NUMBA
    if not _HAVE_NUMBA:
        pytest.skip("numba not installed")
    m0 = QDGainModel(QDGainParams(n_groups=41).with_detailed_balance_taus())
    m1 = QDGainModel(QDGainParams(n_groups=41).with_detailed_balance_taus(), fast=True)
    st = m0.init_slices(40, 40e-3)
    P = np.full(40, 5e-3)
    a = m0.step_slices(st, P, 1.4e-13, m0.p.nu0_Hz, 40e-3)
    b = m1.step_slices(st, P, 1.4e-13, m0.p.nu0_Hz, 40e-3)
    for x, y in zip(a, b):
        assert np.max(np.abs(x - y)) / max(float(np.max(np.abs(x))), 1e-300) < 1e-12


def test_fast_without_numba_raises_or_works():
    # fast=True must either work (numba present) or raise a clear error (never silently slow)
    from dynameta.optics.soa.qd_gain import _HAVE_NUMBA
    if _HAVE_NUMBA:
        assert QDGainModel(QDGainParams(n_groups=1), fast=True)._use_numba is True
    else:
        with pytest.raises(RuntimeError):
            QDGainModel(QDGainParams(n_groups=1), fast=True)


def test_eh_split_reduces_to_excitonic():
    # symmetric e/h (all hole times defaulted) reproduces the excitonic steady state + gain
    exc = QDGainModel(QDGainParams(n_groups=11).with_detailed_balance_taus())
    eh = QDGainModel(QDGainParams(n_groups=11, eh_split=True).with_detailed_balance_taus())
    ye, yh = exc.steady_state(30e-3), eh.steady_state(30e-3)
    assert np.max(np.abs(exc.rho_GS(ye) - eh.f_c_GS(yh))) < 1e-9
    assert np.max(np.abs(eh.f_c_GS(yh) - eh.f_v_GS(yh))) < 1e-11    # stays on f_c=f_v manifold
    se, sh = exc.init_slices(3, 30e-3), eh.init_slices(3, 30e-3)
    g_e = exc.gain_per_m_slices(se, exc.p.nu0_Hz)
    g_h = eh.gain_per_m_slices(sh, eh.p.nu0_Hz)
    assert np.max(np.abs(g_e - g_h)) / np.max(np.abs(g_e)) < 1e-9


def test_eh_gain_nsp_forms():
    from dynameta.optics.soa import inversion_factor_nsp, inversion_factor_nsp_eh
    # n_sp reduces to the excitonic form at f_c=f_v and is +inf at transparency f_c+f_v=1
    assert abs(inversion_factor_nsp_eh(0.85, 0.85) - inversion_factor_nsp(0.85)) < 1e-12
    assert not np.isfinite(inversion_factor_nsp_eh(0.6, 0.4))       # f_c+f_v = 1
    eh = QDGainModel(QDGainParams(n_groups=7, eh_split=True))
    ng = eh.ng
    fcG = np.full((1, ng), 0.8)
    fvG = np.full((1, ng), 0.7)
    st = (np.zeros(1), np.zeros(1), np.zeros((1, ng)), np.zeros((1, ng)), fcG, fvG)
    wl = eh.w_j * eh._lorentzian(eh.p.nu0_Hz - eh.nu_j)
    g_ref = eh._gain_pref * np.sum((fcG[0] + fvG[0] - 1.0) * wl)
    assert abs(eh.gain_per_m_slices(st, eh.p.nu0_Hz)[0] - g_ref) / abs(g_ref) < 1e-12


def test_selfheating_reduction_redshift_fixedpoint():
    from dynameta.optics.soa import SelfHeating
    P = QDGainParams(n_groups=11).with_detailed_balance_taus()
    m_iso = QDGainModel(P)
    y = m_iso.steady_state(40e-3)
    nu = np.linspace(P.nu0_Hz - 4e12, P.nu0_Hz + 4e12, 1001)
    g_iso = m_iso.material_gain_per_m(m_iso.rho_GS(y), nu)
    # Rth=0 with nonzero coefficients -> isothermal byte-identical
    m0 = QDGainModel(P, self_heating=SelfHeating(Rth_K_W=0.0, dnu0_dT_Hz_K=6e10,
                                                 dg_dT_frac_per_K=-2e-3))
    assert np.array_equal(g_iso, m0.material_gain_per_m(m_iso.rho_GS(y), nu))
    # set_temperature red-shifts the peak by ~dnu0_dT*dT (within a group spacing)
    sh = SelfHeating(Rth_K_W=300.0, dnu0_dT_Hz_K=6.2e10, dg_dT_frac_per_K=-1.5e-3)
    m = QDGainModel(P, self_heating=sh)
    rGS = m.rho_GS(m.steady_state(40e-3))
    m.set_temperature(300.0)
    p0 = nu[np.argmax(m.material_gain_per_m(rGS, nu))]
    m.set_temperature(330.0)
    pT = nu[np.argmax(m.material_gain_per_m(rGS, nu))]
    assert abs((p0 - pT) - sh.dnu0_dT_Hz_K * 30.0) < (m._nu_j0[1] - m._nu_j0[0])
    # self-consistent fixed point heats above ambient and the ENOB budget is finite/sane
    g_ss, T_star, G_dB = m.steady_gain_self_consistent(50e-3, 1e-4, 0.6e-3)
    assert T_star > sh.T0_K
    from dynameta.optics.soa import thermal_drift_budget_K
    assert thermal_drift_budget_K(8, m.dGdT_dB_per_K(50e-3, 1e-4, 0.6e-3, T_star)) > 0.0


def test_selfheating_eh_es_numba_parity():
    # the adversarially-found combination: fast=True + eh_split + ES + active self-heating must
    # keep numpy/numba parity (gain emission AND stim depletion both scaled by gain_scale)
    from dynameta.optics.soa.qd_gain import _HAVE_NUMBA
    from dynameta.optics.soa import SelfHeating
    if not _HAVE_NUMBA:
        pytest.skip("numba not installed")
    P = QDGainParams(n_groups=21, eh_split=True, sigma_pk_ES_m2=3e-19).with_detailed_balance_taus()
    sh = SelfHeating(Rth_K_W=300.0, dnu0_dT_Hz_K=6e10, dg_dT_frac_per_K=-2e-3)
    a = QDGainModel(P, self_heating=sh)
    b = QDGainModel(P, self_heating=sh, fast=True)
    a.set_temperature(335.0)
    b.set_temperature(335.0)                                  # gain_scale != 1
    assert abs(a._gain_scale - 1.0) > 1e-3
    st = a.init_slices(20, 40e-3)
    Pl = np.full(20, 6e-3)
    ra = a.step_slices(st, Pl, 1.4e-13, a.p.nu0_Hz, 40e-3)
    rb = b.step_slices(st, Pl, 1.4e-13, a.p.nu0_Hz, 40e-3)
    for x, y in zip(ra, rb):
        assert np.max(np.abs(x - y)) / max(float(np.max(np.abs(x))), 1e-300) < 1e-12


def test_bidir_ase_reduces_and_symmetry():
    from dynameta.optics.soa import (ase_output_psd, ase_spectrum_bidirectional,
                                     noise_figure)
    m = QDGainModel(QDGainParams(n_groups=1).with_detailed_balance_taus())
    nu0, Gamma, N, dz = m.p.nu0_Hz, m.p.Gamma, 50, 1e-5
    rho = np.full(1, 0.85)
    g = float(m.material_gain_per_m(rho, nu0))
    gsp = float(m.emission_gain_per_m(rho, nu0))
    anchor = ase_output_psd(np.full(N, g), np.full(N, 0.85), dz, nu0, Gamma, 0.0, m_pol=2)
    got = ase_spectrum_bidirectional(np.full((N, 1), g), np.full((N, 1), gsp), dz, np.array([nu0]),
                                     np.array([1e10]), Gamma, m_pol=2, direction="forward")
    assert abs(got["S_f_out"][0] - anchor) / abs(anchor) < 1e-13      # reduction
    both = ase_spectrum_bidirectional(np.full((N, 1), g), np.full((N, 1), gsp), dz, np.array([nu0]),
                                      np.array([1e10]), Gamma, m_pol=2)
    assert abs(both["S_f"][0] - both["S_b"][0]) / both["S_f"][0] < 1e-13   # uniform symmetry
    # spectral NF at this single band == noise_figure
    nsp = float(gsp / g)
    assert abs(both["NF"][0] - noise_figure(float(both["G"][0]), nsp)) / both["NF"][0] < 1e-10


def test_bidir_ase_saturation_clamps():
    from dynameta.optics.soa import ase_self_consistent
    m = QDGainModel(QDGainParams(n_groups=15).with_detailed_balance_taus())
    nu0 = m.p.nu0_Hz
    nu = np.linspace(nu0 - 3e12, nu0 + 3e12, 21)
    dnu = np.gradient(nu)
    g_unsat = m.material_gain_per_m(m.rho_GS(m.steady_state(40e-3)), nu)
    off = ase_self_consistent(m, 40e-3, 0.0, nu0, nu, dnu, 0.6e-3, n_slices=40, ase_saturation=False)
    on = ase_self_consistent(m, 40e-3, 0.0, nu0, nu, dnu, 0.6e-3, n_slices=40, ase_saturation=True,
                             ase_strength=20.0)
    assert np.array_equal(off["g_sat"], g_unsat)                      # OFF == unsaturated
    assert np.max(off["g_sat"] - on["g_sat"]) > 0.0 and on["converged"]   # ASE clamps the gain


def test_facet_gain_ripple_and_ceiling():
    from dynameta.optics.soa import facet_gain_ripple_dB, ripple_enob_ceiling
    # Saitoh-Mukai facet ripple: G=20 dB, R=1e-4 -> ~0.17 dB ripple -> ~5.6-bit ENOB ceiling
    r = facet_gain_ripple_dB(100.0, 1e-4)
    assert abs(r - 0.174) < 5e-3
    assert abs(ripple_enob_ceiling(r) - 5.63) < 0.1
    assert facet_gain_ripple_dB(100.0, 0.0) == 0.0            # AR-coated traveling-wave limit
    assert not np.isfinite(ripple_enob_ceiling(0.0))          # zero ripple -> no ENOB ceiling
    with pytest.raises(ValueError):                           # above lasing threshold
        facet_gain_ripple_dB(1e4, 1e-3)


def test_es_band_reduction_and_gain():
    # sigma_pk_ES=0 -> GS-only byte-identical; sigma_pk_ES>0 -> ES gain matches analytic ensemble
    gs = QDGainModel(QDGainParams(n_groups=11).with_detailed_balance_taus())
    y = gs.steady_state(30e-3)
    nu = np.linspace(gs.p.nu0_Hz - 5e12, gs.p.nu0_Hz + 2e13, 30)
    assert np.array_equal(gs.material_gain_per_m(gs.rho_GS(y), nu),
                          gs.total_material_gain(gs.rho_ES(y), gs.rho_GS(y), nu))
    es = QDGainModel(QDGainParams(n_groups=11, sigma_pk_ES_m2=3e-19).with_detailed_balance_taus())
    from dynameta.constants import HBAR
    nu_ES = es.p.nu0_Hz + es.p.dE_ES_GS_eV * 1.602176634e-19 / (2 * np.pi * HBAR)
    ye = es.steady_state(40e-3)
    rES = es.rho_ES(ye)
    hwE = es._hw_ES
    g_num = es.total_material_gain(rES, np.full(11, 0.5), nu_ES)
    g_ref = float(np.sum(es.p.N_q_m3 * es.w_j * es.p.mu_ES * es.p.sigma_pk_ES_m2
                         * hwE**2 / ((nu_ES - es.nu_ES_j)**2 + hwE**2) * (2 * rES - 1.0)))
    assert abs(g_num - g_ref) / abs(g_ref) < 1e-12


def test_es_numba_parity():
    from dynameta.optics.soa.qd_gain import _HAVE_NUMBA
    if not _HAVE_NUMBA:
        pytest.skip("numba not installed")
    for kw in ({}, {"eh_split": True}):
        a = QDGainModel(QDGainParams(n_groups=15, sigma_pk_ES_m2=3e-19, **kw)
                        .with_detailed_balance_taus())
        b = QDGainModel(QDGainParams(n_groups=15, sigma_pk_ES_m2=3e-19, **kw)
                        .with_detailed_balance_taus(), fast=True)
        nu_ES = a.nu_ES_j[a.ng // 2]
        st = a.init_slices(20, 40e-3)
        P = np.full(20, 5e-3)
        ra = a.step_slices(st, P, 1.4e-13, nu_ES, 40e-3)
        rb = b.step_slices(st, P, 1.4e-13, nu_ES, 40e-3)
        for x, y in zip(ra, rb):
            assert np.max(np.abs(x - y)) / max(float(np.max(np.abs(x))), 1e-300) < 1e-12


def test_eh_numba_parity():
    from dynameta.optics.soa.qd_gain import _HAVE_NUMBA
    if not _HAVE_NUMBA:
        pytest.skip("numba not installed")
    a = QDGainModel(QDGainParams(n_groups=21, eh_split=True, tau_cap_h_s=0.25e-12)
                    .with_detailed_balance_taus())
    b = QDGainModel(QDGainParams(n_groups=21, eh_split=True, tau_cap_h_s=0.25e-12)
                    .with_detailed_balance_taus(), fast=True)
    st = a.init_slices(30, 30e-3)
    P = np.full(30, 5e-3)
    ra = a.step_slices(st, P, 1.4e-13, a.p.nu0_Hz, 30e-3)
    rb = b.step_slices(st, P, 1.4e-13, a.p.nu0_Hz, 30e-3)
    for x, y in zip(ra, rb):
        assert np.max(np.abs(x - y)) / max(float(np.max(np.abs(x))), 1e-300) < 1e-12
