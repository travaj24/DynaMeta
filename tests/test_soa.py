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
