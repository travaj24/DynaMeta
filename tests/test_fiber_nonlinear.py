"""Discrimination-proven physics gates for the fiber-amplifier nonlinear power limits
(dynameta.optics.fiber_amp.nonlinear_limits): SBS / SRS thresholds (passive Smith + active
gain-exponent), the TMI average-power estimator, and double-Rayleigh MPI. Each test is a
falsifiable numeric gate drawn from the formulation dossier; the whole file is closed-form /
post-processing only (no steady-state solves), so it runs in well under a second.

Pure numpy/scipy; SI units; ASCII-only.
"""

import numpy as np

from dynameta.optics.fiber_amp.steady_state import SteadyStateResult
from dynameta.optics.fiber_amp.waveguide import FiberSpec, mode_field_radius_m
from dynameta.optics.fiber_amp.nonlinear_limits import (
    brillouin_shift_hz, brillouin_linewidth_hz, brillouin_phonon_number,
    effective_length_m, sbs_threshold_W, sbs_gain_exponent,
    srs_threshold_W, srs_stokes_wavelength_m, raman_gain_coefficient, srs_gain_exponent,
    tmi_threshold_W, TMI_C0_DEFAULT,
    rayleigh_alpha_per_m, capture_fraction, double_rayleigh_mpi,
    mpi_beat_variance_ratio, mpi_rin_per_hz, mpi_power_penalty_dB,
)

ALPHA_02_DB_KM = 0.2 * np.log(10.0) / 1.0e4     # 0.2 dB/km in 1/m = 4.6052e-5
LEN_50KM = 50.0e3


def _synth_result(z, P, lam=1.064e-6):
    """A minimal converged-solve stand-in: one forward 'signal' channel carrying the profile P(z)
    on mesh z. Only the fields the nonlinear post-processors read need to be physical."""
    z = np.asarray(z, dtype=np.float64)
    P = np.asarray(P, dtype=np.float64).reshape(1, -1)
    return SteadyStateResult(
        z_m=z, power_W=P, lambda_m=np.array([lam]),
        u=np.array([1.0]), is_ase=np.array([False]), kind=["signal"],
        nbar2_z=np.zeros_like(z),
        signal_gain_dB=np.array([10.0 * np.log10(P[0, -1] / P[0, 0])]), meta={})


# ============================ Brillouin spectroscopy sanity ================================

def test_brillouin_shift_and_linewidth_scaling():
    assert abs(brillouin_shift_hz(1.55e-6) - 11.15e9) < 0.2e9        # ~11 GHz at 1550
    assert abs(brillouin_shift_hz(1.06e-6) - 16.3e9) < 0.3e9         # ~16 GHz at 1060
    # linewidth ~30 MHz at 1550, 1/lambda^2 scaling -> ~64 MHz at 1060
    assert abs(brillouin_linewidth_hz(1.55e-6) - 30.0e6) < 1.0e6
    assert abs(brillouin_linewidth_hz(1.06e-6) - 64.0e6) < 2.0e6


# ============================ GATE 1: SBS passive (SMF) ====================================

def test_sbs_passive_smf_threshold_gate():
    P_th = sbs_threshold_W(80e-12, g_b=2e-11, length_m=LEN_50KM, alpha_per_m=ALPHA_02_DB_KM,
                           C=21.0, K=1.0)
    assert 3.0e-3 <= P_th <= 7.0e-3            # ~4.3 mW, lit 5-6 mW band


def test_sbs_linewidth_factor_doubles_threshold():
    base = sbs_threshold_W(80e-12, g_b=2e-11, length_m=LEN_50KM, alpha_per_m=ALPHA_02_DB_KM,
                           C=21.0, K=1.0, dnu_b_hz=30e6, dnu_source_hz=0.0)
    broad = sbs_threshold_W(80e-12, g_b=2e-11, length_m=LEN_50KM, alpha_per_m=ALPHA_02_DB_KM,
                            C=21.0, K=1.0, dnu_b_hz=30e6, dnu_source_hz=30e6)
    assert np.isclose(broad / base, 2.0, rtol=1e-12)     # (1 + dnu_source/dnu_B) = 2


def test_sbs_threshold_proportional_to_aeff():
    p1 = sbs_threshold_W(80e-12, g_b=2e-11, length_m=LEN_50KM, alpha_per_m=ALPHA_02_DB_KM, K=1.0)
    p2 = sbs_threshold_W(160e-12, g_b=2e-11, length_m=LEN_50KM, alpha_per_m=ALPHA_02_DB_KM, K=1.0)
    assert np.isclose(p2 / p1, 2.0, rtol=1e-12)


def test_sbs_threshold_inverse_effective_length():
    la, lb = 20.0e3, 50.0e3
    pa = sbs_threshold_W(80e-12, g_b=2e-11, length_m=la, alpha_per_m=ALPHA_02_DB_KM, K=1.0)
    pb = sbs_threshold_W(80e-12, g_b=2e-11, length_m=lb, alpha_per_m=ALPHA_02_DB_KM, K=1.0)
    ratio_expected = (effective_length_m(lb, ALPHA_02_DB_KM)
                      / effective_length_m(la, ALPHA_02_DB_KM))
    assert np.isclose(pa / pb, ratio_expected, rtol=1e-12)    # P_th ~ 1/L_eff exactly


# ============================ GATE 2: SBS active amplifier =================================

def test_sbs_active_gain_exponent_gate():
    # LMA fiber whose Gaussian A_eff lands near 700 um^2 at 1064 nm (25-um core, low NA).
    fiber = FiberSpec(core_radius_m=12.5e-6, na=0.029, n_t_m3=1.0e25, length_m=1.5)
    w = float(mode_field_radius_m(fiber.core_radius_m, fiber.na, 1.064e-6))
    a_eff = np.pi * w * w
    assert 650e-12 <= a_eff <= 750e-12          # A_eff verified ~703 um^2

    # fabricate an exponential P(z) rescaled so integral P dz = 735 W.m exactly (trapezoid)
    z = np.linspace(0.0, 1.5, 201)
    P = np.exp(2.0 * z / 1.5)
    P *= 735.0 / np.trapezoid(P, z)
    res = _synth_result(z, P, lam=1.064e-6)

    out = sbs_gain_exponent(res, fiber, 1.064e-6, g_b=2e-11)
    assert np.isclose(out["integral_P_dz_Wm"], 735.0, rtol=1e-9)
    assert 21.0 * 0.85 <= out["G_B"] <= 21.0 * 1.15       # G_B within 15% of 21
    assert out["P_stokes_out_W"] > out["P_seed_W"]        # net Brillouin gain

    # thermal seed occupation at 1550 nm / 300 K
    n_th = brillouin_phonon_number(1.55e-6, 300.0)
    assert 500.0 <= n_th <= 650.0                          # ~560 phonons per mode


# ============================ GATE 3: SRS passive =========================================

def test_srs_passive_threshold_gate():
    p_srs = srs_threshold_W(80e-12, g_r=0.6e-13, length_m=LEN_50KM, alpha_per_m=ALPHA_02_DB_KM,
                            direction="forward")
    assert 0.6 <= p_srs <= 1.5                             # ~1.09 W

    p_sbs = sbs_threshold_W(80e-12, g_b=2e-11, length_m=LEN_50KM, alpha_per_m=ALPHA_02_DB_KM,
                            C=21.0, K=1.0)
    assert p_srs / p_sbs > 100.0                           # ~254x the SBS threshold


def test_srs_backward_threshold_higher_than_forward():
    pf = srs_threshold_W(80e-12, g_r=0.6e-13, length_m=LEN_50KM, direction="forward")
    pb = srs_threshold_W(80e-12, g_r=0.6e-13, length_m=LEN_50KM, direction="backward")
    assert np.isclose(pb / pf, 20.0 / 16.0, rtol=1e-12)


def test_srs_stokes_redshift_and_gain_exponent():
    # Stokes band 13.2 THz to the red of a 1064-nm signal
    assert srs_stokes_wavelength_m(1.064e-6) > 1.064e-6
    assert abs(srs_stokes_wavelength_m(1.064e-6) - 1.1156e-6) < 5e-9

    fiber = FiberSpec(core_radius_m=12.5e-6, na=0.029, n_t_m3=1.0e25, length_m=1.5)
    z = np.linspace(0.0, 1.5, 201)
    P = np.exp(2.0 * z / 1.5)
    P *= 735.0 / np.trapezoid(P, z)
    out = srs_gain_exponent(_synth_result(z, P, 1.064e-6), fiber, 1.064e-6)
    # default g_R scales as 1e-13*(1e-6/lambda)
    assert np.isclose(out["g_R"], raman_gain_coefficient(1.064e-6), rtol=1e-12)
    assert out["G_R"] > 0.0 and out["stokes_lambda_m"] > 1.064e-6


# ============================ GATE 4: TMI estimator =======================================

def test_tmi_calibration_point_exact():
    p = tmi_threshold_W(20e-6, 1.06e-6, 0.09, gamma_ov=0.5, kappa=1.38, dndt=1.2e-5)
    assert np.isclose(p, 1000.0, rtol=1e-9)               # default C0 pins 1 kW here
    assert TMI_C0_DEFAULT > 0.0


def test_tmi_scaling_exact_lambda_over_dcore_squared():
    # at IDENTICAL eta_heat / Gamma_ov the 85-um core is the pure (lambda/d_core)^2 scaling:
    # 1000 * (20/85)^2 = 55.36 W (NOT the ~250-300 W rod datum; see module docstring caveat).
    p85 = tmi_threshold_W(85e-6, 1.06e-6, 0.09, gamma_ov=0.5)
    assert np.isclose(p85, 1000.0 * (20.0 / 85.0) ** 2, rtol=1e-9)
    assert np.isclose(p85, 55.36, rtol=2e-3)


def test_tmi_robust_monotonic_trends():
    p_lo = tmi_threshold_W(20e-6, 1.06e-6, 0.09, gamma_ov=0.5)
    p_hot = tmi_threshold_W(20e-6, 1.06e-6, 0.18, gamma_ov=0.5)      # eta_heat doubled
    assert np.isclose(p_hot, 0.5 * p_lo, rtol=1e-9)                  # P_th halves exactly
    p_lowov = tmi_threshold_W(20e-6, 1.06e-6, 0.09, gamma_ov=0.3)    # overlap reduced
    assert p_lowov > p_lo                                            # P_th rises


def test_tmi_rod_datapoint_within_2to3x_band():
    # honest recovery attempt: best physically-admissible 85-um rod parameters
    # (eta_heat = 0.052 Yb 976->1030 quantum-defect floor, Gamma_ov = 0.3, lambda_s = 1.03 um)
    # give ~151 W, within the documented 2-3x band of the measured 250-300 W (Eidam 2011).
    p_rod = tmi_threshold_W(85e-6, 1.03e-6, 0.052, gamma_ov=0.3)
    assert 100.0 <= p_rod <= 300.0
    assert 275.0 / p_rod <= 3.0                                       # within a factor of 3


# ============================ GATE 5: double-Rayleigh MPI =================================

def _dummy_fiber():
    return FiberSpec(core_radius_m=5.0e-6, na=0.14, n_t_m3=1.0e25, length_m=1.0)


def test_drb_passive_uniform_closed_form():
    L = 50.0e3
    z = np.linspace(0.0, L, 401)
    P = np.full_like(z, 5.0)                    # flat power, no gain -> G(z1,z2) = 1
    S, aR = 1.0e-3, 3.2e-5
    R = double_rayleigh_mpi(_synth_result(z, P), _dummy_fiber(), 1.55e-6, S=S, alpha_R=aR)
    closed = (aR * S) ** 2 * L ** 2 / 2.0
    assert np.isclose(R, closed, rtol=1e-2)     # matches (alpha_R S)^2 L^2/2 to 1%


def test_drb_exponential_gain_closed_form():
    L, g = 2.0, np.log(100.0) / 2.0             # 20 dB (x100) over 2 m
    z = np.linspace(0.0, L, 401)
    P = np.exp(g * z)
    S, aR = 1.0e-3, 3.2e-5
    R = double_rayleigh_mpi(_synth_result(z, P), _dummy_fiber(), 1.06e-6, S=S, alpha_R=aR)
    # independent analytic oracle: for P=P0 e^{gz}, (P(z2)/P(z1))^2 = e^{2g(z2-z1)},
    # double integral over z1<z2 = (e^{2gL} - 1 - 2gL) / (4 g^2)
    oracle = (aR * S) ** 2 * (np.exp(2 * g * L) - 1.0 - 2 * g * L) / (4.0 * g * g)
    assert np.isclose(R, oracle, rtol=5e-3)     # numerical integral matches oracle to 0.5%
    assert 10.0 * np.log10(R) < -40.0           # well below the -40 dB acceptance floor


def test_drb_monotonic_in_gain():
    L = 2.0
    z = np.linspace(0.0, L, 401)
    S, aR = 1.0e-3, 3.2e-5
    prev = -1.0
    for gain_dB in (0.0, 5.0, 10.0, 15.0, 20.0, 25.0):
        g = gain_dB * np.log(10.0) / 10.0 / L
        P = np.exp(g * z)
        R = double_rayleigh_mpi(_synth_result(z, P), _dummy_fiber(), 1.06e-6, S=S, alpha_R=aR)
        assert R > prev                          # strictly increasing with gain
        prev = R


def test_rayleigh_alpha_and_capture_fraction():
    # Rayleigh loss part: ~3.6e-5 /m at 1550 (A_R=0.9), and 1/lambda^4 -> much larger at 1060
    a1550 = float(rayleigh_alpha_per_m(1.55e-6))
    a1060 = float(rayleigh_alpha_per_m(1.06e-6))
    assert 2.5e-5 <= a1550 <= 5.0e-5
    assert np.isclose(a1060 / a1550, (1.55 / 1.06) ** 4, rtol=1e-9)     # 1/lambda^4 scaling
    # capture fraction ~1e-3..1.6e-3 for a ~5 um SMF mode radius at 1550
    S = float(capture_fraction(1.55e-6, 5.2e-6))
    assert 1.0e-3 <= S <= 1.7e-3


def test_mpi_postprocessors():
    R = 1.0e-4                                   # -40 dB
    # beat variance: k_pol R (2 B_e / dnu_DRB)
    s2 = mpi_beat_variance_ratio(R, B_e=10.0e9, dnu_drb=1.0e11, k_pol=1.0)
    assert np.isclose(s2, R * 2.0 * 10.0e9 / 1.0e11, rtol=1e-12)
    # RIN Lorentzian: peak at f=0 is 4R/(pi dnu_s), half at f=dnu_s
    dnu_s = 1.0e6
    rin0 = mpi_rin_per_hz(R, 0.0, dnu_s)
    rin1 = mpi_rin_per_hz(R, dnu_s, dnu_s)
    assert np.isclose(rin0, 4.0 * R / (np.pi * dnu_s), rtol=1e-12)
    assert np.isclose(rin1 / rin0, 0.5, rtol=1e-12)
    # penalty small for a -40 dB, in-band-limited case; grows and diverges as Q^2 s2 -> 1
    assert mpi_power_penalty_dB(1.0e-5, Q=6.0) < 0.5
    assert mpi_power_penalty_dB(1.0 / 36.0 + 1e-9, Q=6.0) == float("inf")


# ============================ GATE 6: import contract =====================================

def test_base_import_contract_intact():
    import dynameta                                # base package imports cleanly
    from dynameta.optics.fiber_amp import nonlinear_limits
    for name in ("sbs_threshold_W", "srs_threshold_W", "tmi_threshold_W",
                 "double_rayleigh_mpi"):
        assert hasattr(nonlinear_limits, name)
    assert isinstance(dynameta, type(np))          # it is a module
