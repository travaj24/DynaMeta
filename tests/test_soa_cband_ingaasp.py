"""v0.15.1 (user decision 2026-11-08: "I would like the option to use the SOA for C-band"): the opt-in C-band
active material of optics.soa -- InP-lattice-matched Ga(x)In(1-x)As(z)P(1-z), x = 0.47 z, z = 0.937
("InGaAsP_LM_C", BulkGainParams.c_band()) -- against Vurgaftman, Meyer & Ram-Mohan, J. Appl. Phys. 89, 5815
(2001), read from the full text (docs/audit/2026-11-08-v0.15-soa-cband.md).

THE ORACLE is the paper's numbers RE-TYPED below and the paper's equations evaluated here, independently of
the module (no dynameta call inside _oracle_*):
  GaAs Table I (1.519 eV, 0.5405 meV/K, 204 K; a_lc 5.65325 A), InAs Table III (0.417 eV, 0.276 meV/K, 93 K;
  6.0583 A), InP Table VI (1.4236 eV, 0.363 meV/K, 162 K; 5.8697 A), GaP Table IV (a_lc 5.4505 A);
  GaInAs Table XIII bowing 0.477 eV at x(In) = 0.53 (Eq. (4.1));
  GaInAsP on InP, Sec. V.B.1: (InP)(1-z)(Ga0.47In0.53As)(z) with Eq. (5.4)
      G(z) = (1-z) G_InP + z G_GaInAs - z (1-z) C,  C = 0.13 eV "at all temperatures",
  the temperature dependence from the binary Varshni laws (the paper's recommendation).
The paper's printed closed forms (1.4236(1-z) + 0.816z + 0.13z^2 at 0 K, 1.353(1-z) + 0.737z + 0.13z^2 at
300 K) drop the -0.13 z term of -z(1-z)C: they give 0.946 / 0.867 eV at z = 1, not the paper's own GaInAs end
points (0.816 / 0.737 eV). test_printed_closed_form_is_inconsistent_with_eq_5_4 pins that, and the oracle
follows Eq. (5.4) as the text instructs.

ERROR BUDGETS. Kernel vs oracle: the same closed form in a different operation order, <= ~30 double
operations on O(1 eV) terms -> <= 30 * 2.2e-16 * 1.5 eV ~ 1e-14 eV; tolerance 1e-13 eV. Quoted paper values
rounded to 1 meV -> +-0.5 meV. Gain-peak target: z is the brentq root z* = 0.93737 rounded to 0.001, so the peak
can sit up to half a step (5e-4 x |dEg/dz| = 5e-4 x 0.502 eV = 0.251 meV -> 0.487 nm at 1550 nm; the peak
tracks the gap one-for-one, test_soa_ingaas_gap_vurgaftman) plus one gain_peak grid step (0.25 eV / 4000 =
62.5 ueV -> 0.121 nm) from 1550.0 nm: tolerance 0.61 nm.

FAILS ON THE OLD CODE (0554410, MEASURED): "InGaAsP_LM_C", ingaasp_lm_gap_law and BulkGainParams.c_band do not
exist (ValueError "unknown material" / ImportError / AttributeError), and the nested quaternary spec raises
"not a binary". The C-band device the v0.14 notes told users to build (tests/test_soa_qw.py CBAND_GAP, an
uncited triple) misses the quaternary oracle by -30.72 / -26.71 / -5.51 / +5.09 meV at 0 / 77 / 298 / 400 K and
peaks at 1560.4 nm, 10.4 nm off the 1550 nm target (> the 0.61 nm budget); the default BulkGainParams() peaks
at 1612.5 nm.
"""
import numpy as np
import pytest
from dataclasses import fields, replace

from dynameta.constants import C_LIGHT, H_PLANCK, Q_E
from dynameta.materials import band_temperature as BT
from dynameta.optics.soa import qw_gain as qw
from dynameta.optics.soa import temperature as tm
from dynameta.optics.soa.qw_gain import BulkGainParams

# --- Vurgaftman 2001, re-typed (eV, eV/K, K; Angstrom) -------------------------------------------------------
GAAS = (1.519, 0.5405e-3, 204.0)      # Table I
INAS = (0.417, 0.276e-3, 93.0)        # Table III
INP = (1.4236, 0.363e-3, 162.0)       # Table VI
C_GAINAS, X_IN = 0.477, 0.53          # Table XIII; Ga0.47In0.53As
C_Q = 0.13                            # Sec. V.B.1, Eq. (5.4) bowing for (InP)(1-z)(Ga0.47In0.53As)(z)
A_GAAS, A_INAS, A_INP, A_GAP = 5.65325, 6.0583, 5.8697, 5.4505      # Tables I, III, VI, IV (300 K)
Z_C = 0.937                           # the preset composition (solved; module comment / audit note)

OLD_CBAND_TRIPLE = (0.816, 2.9e-4, 193.0)   # tests/test_soa_qw.py CBAND_GAP: the v0.14 "pass it explicitly" advice
TOL_EV = 1.0e-13
TOL_QUOTED_EV = 0.5e-3
TOL_PEAK_NM = 0.61
L_M, DRIVE_A = 600.0e-6, 130.0e-3


def _varshni(T, e0, a, b):
    return e0 - a * T ** 2 / (T + b)


def _oracle_gainas(T):
    return (1.0 - X_IN) * _varshni(T, *GAAS) + X_IN * _varshni(T, *INAS) - X_IN * (1.0 - X_IN) * C_GAINAS


def _oracle_q(T, z=Z_C, C=C_Q):
    """Eq. (5.4) between InP and Ga0.47In0.53As."""
    return (1.0 - z) * _varshni(T, *INP) + z * _oracle_gainas(T) - z * (1.0 - z) * C


def _peak_nm(p):
    N0 = qw.steady_state_N(DRIVE_A, p)
    nu, g = qw.gain_peak(N0, p)
    return C_LIGHT / nu * 1e9, g, N0


# --- the material law ------------------------------------------------------------------------------------------
@pytest.mark.parametrize("T", [0.0, 77.0, 298.0, 300.0, 400.0])
def test_cband_quaternary_gap_matches_retyped_vurgaftman_eq_5_4(T):
    """Kernel vs the re-typed Eq. (5.4) oracle, 1e-13 eV. The old uncited C-band triple misses the same oracle by
    -30.72 / -26.71 / -5.51 / -5.31 / +5.09 meV (MEASURED, asserted as the discrimination check)."""
    ref = _oracle_q(T)
    assert abs(tm.varshni_eg_ev(T, "InGaAsP_LM_C") - ref) < TOL_EV
    assert abs(tm.varshni_eg_ev(T, tm.ingaasp_lm_gap_law(Z_C)) - ref) < TOL_EV
    for z in (0.0, 0.3, 0.6, 1.0):                      # the whole lattice-matched family
        assert abs(tm.varshni_eg_ev(T, tm.ingaasp_lm_gap_law(z)) - _oracle_q(T, z)) < TOL_EV, z
    miss_old = _varshni(T, *OLD_CBAND_TRIPLE) - ref
    expected = {0.0: -30.72e-3, 77.0: -26.71e-3, 298.0: -5.51e-3, 300.0: -5.31e-3, 400.0: 5.09e-3}[T]
    assert abs(miss_old - expected) < 0.01e-3, (T, miss_old)


def test_quaternary_end_points_are_the_values_the_paper_quotes():
    """Sec. V.B.1 quotes the 300 K end points of the law, 1.353 eV (InP) and 0.737 eV (Ga0.47In0.53As)."""
    assert abs(tm.varshni_eg_ev(300.0, tm.ingaasp_lm_gap_law(0.0)) - 1.353) < TOL_QUOTED_EV
    assert abs(tm.varshni_eg_ev(300.0, tm.ingaasp_lm_gap_law(1.0)) - 0.737) < TOL_QUOTED_EV
    assert abs(tm.varshni_eg_ev(0.0, tm.ingaasp_lm_gap_law(1.0)) - 0.816) < TOL_QUOTED_EV


def test_printed_closed_form_is_inconsistent_with_eq_5_4():
    """Why the code follows Eq. (5.4) and not the printed polynomial: at z = 1 the printed forms give
    0.946 / 0.867 eV, 130 meV off the paper's own GaInAs end points; at z = 0.937 the printed 300 K form is
    121.6 meV wider than Eq. (5.4) (= 0.13 z, MEASURED)."""
    printed_0K = lambda z: 1.4236 * (1 - z) + 0.816 * z + 0.13 * z ** 2
    printed_300 = lambda z: 1.353 * (1 - z) + 0.737 * z + 0.13 * z ** 2
    assert abs(printed_0K(1.0) - 0.816) > 0.12 and abs(printed_300(1.0) - 0.737) > 0.12
    assert printed_300(Z_C) - tm.varshni_eg_ev(300.0, "InGaAsP_LM_C") == pytest.approx(0.13 * Z_C, abs=2e-3)


def test_cband_slope_against_the_oracle_central_difference():
    """dEg/dT vs a central difference of the INDEPENDENT oracle (h = 1e-3 K; rounding ~eps Eg / h ~ 2e-13 eV/K on
    3.5e-4 eV/K -> rel 1e-8 is > 1000x margin)."""
    h = 1.0e-3
    for T in (77.0, 298.0, 400.0):
        fd = (_oracle_q(T + h) - _oracle_q(T - h)) / (2.0 * h)
        assert tm.d_eg_dT_ev_per_K(T, "InGaAsP_LM_C") == pytest.approx(fd, rel=1e-8)
    # the gap-only gain-peak drift at 1550 nm, from the oracle slope: 0.675 nm/K (MEASURED)
    hc_eVnm = H_PLANCK * C_LIGHT / Q_E * 1e9
    fd300 = (_oracle_q(300.0 + h) - _oracle_q(300.0 - h)) / (2.0 * h)
    assert tm.gain_peak_drift_nm_per_K(1550.0, "InGaAsP_LM_C", 300.0) == \
        pytest.approx(1550.0 ** 2 / hc_eVnm * abs(fd300), rel=1e-8)


def test_preset_composition_is_lattice_matched_by_vegard_on_the_binary_constants():
    """Vegard over the four binaries (Tables I, III, IV, VI): the paper's scheme x = 0.47 z at z = 0.937 is
    -3.50e-4 off InP (MEASURED), inside LATTICE_MATCH_RTOL = 1e-3; the exact Vegard match there is
    x = 0.4353 (vs 0.4404). The module helper and the kernel's table equal the re-typed values."""
    z, x = Z_C, 0.47 * Z_C
    a_q = x * z * A_GAAS + x * (1 - z) * A_GAP + (1 - x) * z * A_INAS + (1 - x) * (1 - z) * A_INP
    mis = (a_q - A_INP) / A_INP
    assert BT.ingaasp_lattice_mismatch(z) == pytest.approx(mis, rel=1e-12)
    assert abs(mis) < BT.LATTICE_MATCH_RTOL
    assert mis == pytest.approx(-3.50e-4, abs=0.01e-4)
    x_exact = z * (A_INAS - A_INP) / (z * (A_INAS - A_GAAS) + (1 - z) * (A_INP - A_GAP))
    assert BT.ingaasp_lattice_mismatch(z, x_exact) == pytest.approx(0.0, abs=1e-15)
    assert x_exact == pytest.approx(0.4353, abs=1e-4)
    assert BT.LATTICE_CONSTANT_300K_A == {"GaAs": A_GAAS, "AlAs": 5.6611, "GaP": A_GAP, "InAs": A_INAS,
                                          "InP": A_INP}
    assert BT.ALLOY_GAP_PARAMS["InGaAsP_LM_C"] == ("InP", "InGaAs_LM", Z_C, C_Q) == BT.ingaasp_lm_gap_law(Z_C)
    assert BT.INGAASP_LM_C_Z == Z_C and BT.INGAASP_LM_BOWING_EV == C_Q


def test_only_lattice_matched_binary_or_ternary_end_points_are_accepted():
    """Eq. (5.4) combines two LATTICE-MATCHED binary or ternary end points: a mismatched pair, a quaternary end
    point and an end point without a known lattice constant are refused (the v0.14 'not a binary' refusal of
    ("InGaAs_LM", "InAs", ...) is kept, now with the mismatch quoted)."""
    with pytest.raises(ValueError, match="not a binary.*-3.14 %"):
        tm.varshni_eg_ev(300.0, ("InGaAs_LM", "InAs", 0.5, 0.0))
    with pytest.raises(ValueError, match="ternary of two binaries"):
        tm.varshni_eg_ev(300.0, ("InP", "InGaAsP_LM_C", 0.5, 0.0))
    with pytest.raises(ValueError, match="known lattice constant"):
        tm.varshni_eg_ev(300.0, ((1.4236, 0.363e-3, 162.0), "InGaAs_LM", 0.5, 0.13))
    with pytest.raises(ValueError, match="z \\(the As fraction\\) must be in"):
        tm.ingaasp_lm_gap_law(1.2)


def test_ternary_path_and_default_soa_are_bit_identical_to_v0_14():
    """Defaults-off rule, the in-suite third of the proof (the other two: the 1182-value SHA-256 census on
    0554410 vs this tree, and the unchanged SOA suites; audit note). The ternary terms are structurally the v0.14
    ones, and the pure-arithmetic gap values are pinned to their 0554410 bits (no transcendental call, so the
    bits are platform-independent IEEE arithmetic)."""
    terms, off = BT._gap_terms("InGaAs_LM")
    assert terms == ((1.0 - 0.53, BT.VARSHNI_PARAMS["GaAs"]), (0.53, BT.VARSHNI_PARAMS["InAs"]))
    assert off == 0.53 * (1.0 - 0.53) * 0.477
    pins = {77.0: ("0x1.9c7f2487b264fp-1", "0x1.30c86a10073d8p-63", "-0x1.d2edfef390f01p-13"),
            298.0: ("0x1.79d588d366e4fp-1", "0x1.172bf09889d30p-63", "-0x1.6f180b02295fep-12"),
            400.0: ("0x1.67177b23aeff1p-1", "0x1.0952c8f551446p-63", "-0x1.7fea7c5c9c845p-12")}
    p = BulkGainParams()
    for T, (eg, eg_J, slope) in pins.items():
        assert tm.varshni_eg_ev(T, "InGaAs_LM").hex() == eg
        assert qw.band_edge_Eg_J(0.0, replace(p, T_K=T)).hex() == eg_J
        assert tm.d_eg_dT_ev_per_K(T, "InGaAs_LM").hex() == slope
    v014_defaults = {"Eg0_eV": None, "varshni_alpha_eV_K": None, "varshni_beta_K": None, "bgs_coeff_eV_m": 0.0,
                     "m_e": 0.045, "m_dh": 0.46, "n_index": 3.5, "n_group": 3.7, "tau_rad_s": 1e-09,
                     "tau_intraband_s": 1e-13, "A_srh_per_s": 2.5e8, "B_rad_m3_s": 1e-16, "C_aug_m6_s": 5e-41,
                     "Gamma": 0.3, "V_active_m3": 8.5e-16, "A_xsec_m2": 1.4167e-12, "T_K": 298.0,
                     "gap_material": "InGaAs_LM"}
    assert {f.name: f.default for f in fields(BulkGainParams)} == v014_defaults


# --- the SOA consumers -----------------------------------------------------------------------------------------
def test_c_band_preset_is_the_quaternary_and_only_the_gap_differs():
    p = BulkGainParams.c_band()
    assert p.gap_law == "InGaAsP_LM_C"
    assert p == replace(BulkGainParams(), gap_material="InGaAsP_LM_C")
    assert BulkGainParams.c_band(T_K=320.0, Gamma=0.25) == replace(p, T_K=320.0, Gamma=0.25)
    for T in (77.0, 298.0, 400.0):
        assert abs(qw.band_edge_Eg_J(0.0, replace(p, T_K=T)) / Q_E - _oracle_q(T)) < TOL_EV
    for bad in ("gap_material", "Eg0_eV", "varshni_alpha_eV_K", "varshni_beta_K"):
        with pytest.raises(TypeError, match="would replace the C-band gap"):
            BulkGainParams.c_band(**{bad: 0.8})
    # the same material key works in the other gap-law consumers (QD retargeting, carriers' BandTemperature)
    assert BT.BandTemperature("InGaAsP_LM_C", 1.0).gap_change_eV(350.0, 300.0) == \
        pytest.approx(_oracle_q(350.0) - _oracle_q(300.0), abs=TOL_EV)


def test_c_band_gain_peak_lands_on_1550_nm():
    """The gate the preset exists for: material-gain peak at the reference operating point (130 mA, 298 K,
    default geometry) within TOL_PEAK_NM (module docstring) of 1550.0 nm, and inside the ITU C-band
    (1530-1565 nm). MEASURED 1549.68 nm. Old code: no preset; the v0.14 explicit C-band triple peaks at
    1560.43 nm (10.4 nm off) and the default at 1612.50 nm."""
    lam, g, N0 = _peak_nm(BulkGainParams.c_band())
    assert abs(lam - 1550.0) <= TOL_PEAK_NM, lam
    assert 1530.0 <= lam <= 1565.0
    lam_old, _, _ = _peak_nm(BulkGainParams(Eg0_eV=OLD_CBAND_TRIPLE[0], varshni_alpha_eV_K=OLD_CBAND_TRIPLE[1],
                                            varshni_beta_K=OLD_CBAND_TRIPLE[2]))
    assert abs(lam_old - 1550.0) > 10.0 * TOL_PEAK_NM, lam_old
    assert _peak_nm(BulkGainParams())[0] > 1565.0
    print("\n[cband] peak {:.2f} nm  g_pk {:.4e} /m  N0 {:.4e} m^-3 (old triple {:.2f} nm)".format(lam, g, N0, lam_old))


def test_adversarial_perturbations_break_the_gates():
    """Each must miss its gate: Pearsall's bowing 0.149 eV (quoted in Sec. V.B.1) moves the gap 1.12 meV
    (>> 1e-13 eV); z one grower step off (0.936) puts the peak 1.34 nm from 1550 nm (> 0.61 nm); the 'Q1.55'
    composition (1.55 um GAP, z = 0.875) peaks at ~1491 nm, outside the C-band (all MEASURED)."""
    pearsall = tm.varshni_eg_ev(300.0, ("InP", "InGaAs_LM", Z_C, 0.149))
    assert abs(pearsall - _oracle_q(300.0)) > 1.0e-3
    lam_step, _, _ = _peak_nm(BulkGainParams(gap_material=tm.ingaasp_lm_gap_law(0.936)))
    assert abs(lam_step - 1550.0) > TOL_PEAK_NM, lam_step
    z_q155 = 0.875
    assert abs(_oracle_q(300.0, z_q155) - H_PLANCK * C_LIGHT / 1.55e-6 / Q_E) < 1.0e-3   # it IS the Q1.55 gap
    lam_q, _, _ = _peak_nm(BulkGainParams(gap_material=tm.ingaasp_lm_gap_law(z_q155)))
    assert lam_q < 1530.0, lam_q


def test_c_band_vs_old_triple_moves_only_through_the_gap():
    """Rule-7 cross-check (the identity of test_soa_ingaas_gap_vurgaftman): at equal N and T,
    g(Eg + x)(Eg + x)^2 is the same function of the excess energy x for both gaps (rel 1e-12), and the MEASURED
    peak shift equals the gap shift within 0.3 meV (grid step + 1/nu^2 tilt)."""
    new = BulkGainParams.c_band()
    old = BulkGainParams(Eg0_eV=OLD_CBAND_TRIPLE[0], varshni_alpha_eV_K=OLD_CBAND_TRIPLE[1],
                         varshni_beta_K=OLD_CBAND_TRIPLE[2])
    N = qw.steady_state_N(DRIVE_A, new)
    assert N == qw.steady_state_N(DRIVE_A, old)          # the reservoir does not see the gap
    x_J = np.linspace(1.0e-3, 0.12, 60) * Q_E
    eg_n, eg_o = qw.band_edge_Eg_J(N, new), qw.band_edge_Eg_J(N, old)
    g_n = qw.material_gain_per_m((eg_n + x_J) / H_PLANCK, N, new) * (eg_n + x_J) ** 2
    g_o = qw.material_gain_per_m((eg_o + x_J) / H_PLANCK, N, old) * (eg_o + x_J) ** 2
    np.testing.assert_allclose(g_n, g_o, rtol=1e-12, atol=0.0)
    d_pk = H_PLANCK * (qw.gain_peak(N, new)[0] - qw.gain_peak(N, old)[0]) / Q_E
    assert abs(d_pk - (eg_n - eg_o) / Q_E) < 0.3e-3


def test_c_band_device_numbers_and_temperature_drift_report():
    """Device-level numbers of the preset next to the old triple (MEASURED, printed under -s; the audit note
    table). Asserted only as physics windows: the BOA1004P-class gain window of test_soa_qw gate 8 (24-30 dB at
    alpha_i = 2500 /m) and a red drift. Fixed-current peak drift 288 -> 308 K: c_band 1.007 nm/K, old triple
    0.813 nm/K (gain_peak grid; 1.001 / 0.812 on a 10x finer grid); gap-only drift at 1550 nm 0.675 vs 0.476 nm/K;
    peak 1549.68 vs 1560.43 nm; G 27.52 vs 28.00 dB."""
    rows = {}
    for name, p in (("c_band", BulkGainParams.c_band()),
                    ("old_triple", BulkGainParams(Eg0_eV=OLD_CBAND_TRIPLE[0],
                                                  varshni_alpha_eV_K=OLD_CBAND_TRIPLE[1],
                                                  varshni_beta_K=OLD_CBAND_TRIPLE[2]))):
        lam, _g, N0 = _peak_nm(p)
        nu = C_LIGHT / (lam * 1e-9)
        G = qw.device_gain_dB(DRIVE_A, nu, L_M, 2500.0, params=p)
        pk = {T: _peak_nm(replace(p, T_K=T))[0] for T in (288.0, 308.0)}
        drift = (pk[308.0] - pk[288.0]) / 20.0
        rows[name] = (lam, G, drift)
        assert 24.0 <= G <= 30.0, (name, G)
        assert drift > 0.0
    print("\n[cband] c_band: peak {:.2f} nm  G(2500/m) {:.2f} dB  drift {:.3f} nm/K | old triple: {:.2f} nm  "
          "{:.2f} dB  {:.3f} nm/K".format(*rows["c_band"], *rows["old_triple"]))
