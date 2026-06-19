"""Fast unit tests for the QD-SOA package (the rigorous multi-gate physics lives in
validation/qd_soa_gain_core.py and validation/qd_soa_traveling_wave.py)."""
import numpy as np
import pytest

from dynameta.optics.soa import (QDGainModel, QDGainParams, TravelingWaveSOA,
                                 TwoLevelSaturableGain, agrawal_olsson_output)


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
