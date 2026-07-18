"""Physics gates for the Connelly bulk / bulk-QW SOA gain core (dynameta.optics.soa.qw_gain).

These are the load-bearing sign / magnitude / device gates from the formulation dossier (Topics
6, 7, 8) and the Thorlabs BOA1004P anchor (1550 nm, 27 dB, 85 nm, 15 dBm, 7.5 dB). The
transparency gate is the sign-trap killer: with a flipped valence occupation the g = 0 crossing
would NOT satisfy h nu = quasi-Fermi separation, which gate 1 pins to < 2 meV.

Numbers reached (informational, printed under -s): see the module docstring and the per-gate
asserts below.
"""
import numpy as np
import pytest
from dataclasses import replace

from dynameta.constants import C_LIGHT, H_PLANCK, Q_E
from dynameta.optics.soa import qw_gain as qw
from dynameta.optics.soa.qw_gain import BulkGainParams


# Reference bulk InGaAsP device (dossier Connelly gate): 600 um chip, internal loss ~3000 /m,
# 130 mA drive; the default BulkGainParams geometry is sized for this operating point.
L_M = 600.0e-6
ALPHA_I = 3000.0
DRIVE_A = 130.0e-3


def _ref_params():
    return BulkGainParams()


def _peak_nu_and_Ntr(p):
    """Peak-gain frequency at a well-inverted reference density, and the transparency density at
    that frequency."""
    nu_ref, _ = qw.gain_peak(2.0e24, p)
    N_tr = qw.transparency_density(nu_ref, p)
    return nu_ref, N_tr


# --------------------------------------------------------------------------------------------------
# gate 1 -- TRANSPARENCY (the sign trap)
# --------------------------------------------------------------------------------------------------
def test_gate1_transparency_and_fermi_identity():
    p = _ref_params()
    nu_ref, N_tr = _peak_nu_and_Ntr(p)

    # N_tr in the dossier band
    assert 0.8e24 < N_tr < 2.0e24, N_tr

    # gain sign flips across transparency at the peak frequency
    g_hi = qw.material_gain_per_m(nu_ref, 2.0 * N_tr, p)
    g_lo = qw.material_gain_per_m(nu_ref, 0.5 * N_tr, p)
    assert g_hi > 0.0, g_hi
    assert g_lo < 0.0, g_lo

    # the classic identity: at transparency the photon energy EQUALS the quasi-Fermi separation.
    # A wrong f_v sign breaks this even though a g=0 crossing still exists.
    hnu_eV = H_PLANCK * nu_ref / Q_E
    dF_eV = qw.quasi_fermi_separation_eV(N_tr, p)
    assert abs(dF_eV - hnu_eV) < 2.0e-3, (dF_eV, hnu_eV)
    print("\n[gate1] N_tr={:.3e} m^-3  hnu={:.5f} eV  dF={:.5f} eV  diff={:.2e} meV"
          .format(N_tr, hnu_eV, dF_eV, (dF_eV - hnu_eV) * 1e3))


# --------------------------------------------------------------------------------------------------
# gate 2 -- LOGARITHMIC GAIN LAW  g_pk = g0 ln(N/N_tr)
# --------------------------------------------------------------------------------------------------
def test_gate2_log_gain_fit():
    p = _ref_params()
    _nu_ref, N_tr = _peak_nu_and_Ntr(p)
    N = np.linspace(1.2, 3.0, 15) * N_tr
    g_pk = np.array([qw.gain_peak(Ni, p)[1] for Ni in N])
    x = np.log(N / N_tr)
    g0 = float(np.sum(x * g_pk) / np.sum(x * x))          # through-origin least squares
    r2 = 1.0 - np.sum((g_pk - g0 * x) ** 2) / np.sum((g_pk - g_pk.mean()) ** 2)
    assert 1.0e5 < g0 < 3.0e5, g0
    assert r2 > 0.98, r2
    print("\n[gate2] g0={:.3e} /m  R2={:.4f}".format(g0, r2))


# --------------------------------------------------------------------------------------------------
# gate 3 -- CONNELLY DEVICE GATE  (bulk InGaAsP, 130 mA, 600 um)
# --------------------------------------------------------------------------------------------------
def test_gate3_connelly_device_gain():
    p = _ref_params()
    N0 = qw.steady_state_N(DRIVE_A, p, P_W=0.0)
    nu_pk, g_pk = qw.gain_peak(N0, p)
    lam_pk_nm = C_LIGHT / nu_pk * 1e9

    G_dB = qw.device_gain_dB(DRIVE_A, nu_pk, L_M, ALPHA_I, P_in_W=1.0e-9, params=p, nz=200)

    assert 20.0 <= G_dB <= 33.0, G_dB
    assert 1500.0 <= lam_pk_nm <= 1600.0, lam_pk_nm
    print("\n[gate3] G={:.2f} dB  peak={:.1f} nm  N0={:.3e}  g_pk={:.3e} /m"
          .format(G_dB, lam_pk_nm, N0, g_pk))


# --------------------------------------------------------------------------------------------------
# gate 4 -- SATURATION  (output-referred -3 dB power; sub-linear in deep saturation)
# --------------------------------------------------------------------------------------------------
def test_gate4_saturation():
    p = _ref_params()
    N0 = qw.steady_state_N(DRIVE_A, p, P_W=0.0)
    nu_pk, _ = qw.gain_peak(N0, p)

    psat_dbm, G0 = qw.saturation_output_power_dbm(DRIVE_A, nu_pk, L_M, ALPHA_I, p, nz=150)
    assert 5.0 <= psat_dbm <= 20.0, psat_dbm

    # deep saturation: output grows SUB-linearly in input (gain compresses)
    _, Pout = qw.device_output_power_W(DRIVE_A, nu_pk, L_M, ALPHA_I,
                                       np.array([1.0e-2, 2.0e-2]), p, nz=150)
    growth = Pout[1] / Pout[0]
    assert growth < 2.0, growth                            # < input ratio of 2
    print("\n[gate4] Psat_out={:.2f} dBm  G0={:.2f} dB  deep-sat output ratio={:.3f} (<2)"
          .format(psat_dbm, G0, growth))


# --------------------------------------------------------------------------------------------------
# gate 5 -- SUB-TRANSPARENCY ASE  (emission source finite/positive where net gain is absorbing)
# --------------------------------------------------------------------------------------------------
def test_gate5_sub_transparency_ase():
    p = _ref_params()
    nu_ref, N_tr = _peak_nu_and_Ntr(p)

    g_e = qw.emission_gain_per_m(nu_ref, 0.9 * N_tr, p)
    g_m = qw.material_gain_per_m(nu_ref, 0.9 * N_tr, p)
    assert g_m < 0.0, g_m                                  # net absorbing below transparency
    assert g_e > 0.0 and np.isfinite(g_e), g_e             # emission source still positive/finite

    # continuity through N_tr: g_e smooth and monotone, no 0*inf blow-up
    Ns = np.array([0.9, 1.0, 1.1]) * N_tr
    ge = np.array([qw.emission_gain_per_m(nu_ref, Ni, p) for Ni in Ns])
    assert np.all(np.diff(ge) > 0.0)                       # increasing, finite
    assert np.all(np.isfinite(ge))
    print("\n[gate5] g_e(0.9,1.0,1.1 N_tr)={} /m  g_m(0.9 N_tr)={:.3e} /m"
          .format(np.array2string(ge, precision=3), g_m))


# --------------------------------------------------------------------------------------------------
# gate 6 -- TEMPERATURE  (>10% peak-gain drop + Varshni red-shift, 298 -> 338 K at fixed I)
# --------------------------------------------------------------------------------------------------
def test_gate6_temperature_sensitivity():
    p298 = replace(_ref_params(), T_K=298.0)
    p338 = replace(_ref_params(), T_K=338.0)
    N298 = qw.steady_state_N(DRIVE_A, p298)
    N338 = qw.steady_state_N(DRIVE_A, p338)
    nu298, g298 = qw.gain_peak(N298, p298)
    nu338, g338 = qw.gain_peak(N338, p338)

    drop = 1.0 - g338 / g298
    redshift_nm = C_LIGHT / nu338 * 1e9 - C_LIGHT / nu298 * 1e9
    assert drop > 0.10, drop                               # QW/bulk T-sensitivity (QD contrast)
    assert redshift_nm > 0.0, redshift_nm                  # Varshni red-shift
    print("\n[gate6] peak-gain drop={:.1f}%  red-shift={:.1f} nm".format(drop * 100, redshift_nm))


# --------------------------------------------------------------------------------------------------
# gate 7 -- NOISE FIGURE  (in [4, 9] dB at the operating point, alpha_i > 0)
# --------------------------------------------------------------------------------------------------
def test_gate7_noise_figure():
    p = _ref_params()
    N0 = qw.steady_state_N(DRIVE_A, p)
    nu_pk, _ = qw.gain_peak(N0, p)
    nf_db = qw.noise_figure_db(DRIVE_A, nu_pk, L_M, ALPHA_I, p, nz=150)
    assert 4.0 <= nf_db <= 9.0, nf_db
    print("\n[gate7] NF={:.2f} dB".format(nf_db))


# --------------------------------------------------------------------------------------------------
# gate 8 -- BOA1004P ORIENTATION (soft): G ~ 27 dB AND net -3 dB BW > 40 nm simultaneously
# --------------------------------------------------------------------------------------------------
def test_gate8_boa1004p_orientation():
    # Documented tweaks (NOT a full calibration): the reference bulk core at the default 130 mA /
    # 600 um operating point with the internal loss nudged to 2500 /m (within the dossier 2500-4000
    # /m band) lands the small-signal chip gain on the BOA's 27 dB and, because the net gain is
    # moderate (little high-gain narrowing), keeps a broad net band.
    p = _ref_params()
    alpha_i = 2500.0
    N0 = qw.steady_state_N(DRIVE_A, p)
    nu_pk, _ = qw.gain_peak(N0, p)

    G_dB = qw.device_gain_dB(DRIVE_A, nu_pk, L_M, alpha_i, P_in_W=1.0e-9, params=p, nz=200)

    lam = np.linspace(1460e-9, 1660e-9, 400)
    nu = C_LIGHT / lam
    g_mat = qw.small_signal_gain_spectrum(DRIVE_A, nu, p)
    net_dB = (10.0 / np.log(10.0)) * (p.Gamma * g_mat - alpha_i) * L_M
    peak = float(net_dB.max())
    in_band = lam[net_dB >= peak - 3.0]
    bw_nm = float((in_band.max() - in_band.min()) * 1e9)

    # honest reporting: G near 27 dB (booster class) and a >40 nm net band together
    assert 24.0 <= G_dB <= 30.0, G_dB
    assert bw_nm > 40.0, bw_nm
    print("\n[gate8] BOA-orientation: G={:.2f} dB (target 27), net -3dB BW={:.1f} nm (target >40; "
          "datasheet 85), peak={:.1f} nm".format(G_dB, bw_nm, C_LIGHT / nu_pk * 1e9))


# --------------------------------------------------------------------------------------------------
# housekeeping
# --------------------------------------------------------------------------------------------------
def test_params_frozen_and_exports():
    p = BulkGainParams()
    with pytest.raises(Exception):
        p.Eg0_eV = 0.9                                     # frozen dataclass
    for name in ("BulkGainParams", "material_gain_per_m", "emission_gain_per_m",
                 "steady_state_N", "device_gain_dB", "noise_figure_db",
                 "saturation_output_power_dbm"):
        assert name in qw.__all__
    assert p.v_g_m_s == pytest.approx(C_LIGHT / p.n_group)


def test_gain_scalar_vs_array_consistency():
    p = BulkGainParams()
    nu0 = C_LIGHT / 1.55e-6
    g_scalar = qw.material_gain_per_m(nu0, 2.0e24, p)
    g_arr = qw.material_gain_per_m(np.array([nu0, nu0]), 2.0e24, p)
    assert np.isscalar(g_scalar) or np.ndim(g_scalar) == 0
    assert np.allclose(g_arr, g_scalar)
