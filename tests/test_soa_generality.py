"""Fast unit tests for the QD-SOA generality batch (dossier Topics 1-6 + Topic 3 temperature):
confined-state Auger (loss vs capture), WL<->ES full detailed balance, the e/h-split ES OPTICAL
split, the area-conserving sech lineshape, sub-transparency ASE, two-tone IMD3/OIP3/SFDR, and the
Varshni temperature model. Each new physics is OPT-IN; the bit-identity tests protect the whole
existing suite.

DEGENERACY-CONVENTION NOTE (detailed balance): the dossier's Topic-2 gate quotes the ES<->GS ratio
as 2 exp(dE/kT) = 30 (using g_upper/g_lower = mu_ES/mu_GS = 2). That is inconsistent with (a) the
dossier's own Topic-2 line-19 form 1/tau_esc = (1/tau_cap)(g_ES/g_GS) exp(-dE/kT), which gives
tau_GS_ES/tau_ES_GS = (mu_GS/mu_ES) exp(dE/kT) = 7.50, and (b) the actual per-STATE-occupation rate
structure of qd_gain (the exchange balance mu_ES*fwd = mu_GS*bwd with Fermi occupancy gives
(mu_GS/mu_ES) exp(dE/kT), hand-derived). The existing with_detailed_balance_taus (which MUST stay
byte-stable) uses (mu_GS/mu_ES) = 0.5 -> 7.50, the physically-correct value for this model, so the
tests pin 7.50 (NOT 30) and the low-drive occupation ratio to exp(-dE/kT) (NOT 2 exp(-dE/kT))."""
import numpy as np
import pytest

from dynameta.constants import C_LIGHT, H_PLANCK, KB, Q_E
from dynameta.optics.soa.qd_gain import QDGainModel, QDGainParams


# ---------------------------------------------------------------------------
# Golden values captured from the pre-edit engine (bit-identity protection).
# ---------------------------------------------------------------------------
GOLD_Y5 = [1.066515588039224e+24, 0.9768301765751776, 0.9768301765751776, 0.9768301765751777,
           0.9768301765751777, 0.9768301765751777, 0.9969152392213361, 0.9969152392213361,
           0.9969152392213361, 0.9969152392213361, 0.9969152392213361]
GOLD_G5 = 30196.61596352387
GOLD_G_DEF = 12211.053892818682
GOLD_SPEC5 = [266.68203359207394, 850.9884699547493, 5738.899515970181, 30196.61596352387,
              5738.899515970182, 850.9884699547493, 266.682033592074]


# =========================== ITEM 1: AUGER =================================

def test_auger_capture_effective_time():
    # 1a: with C_W the capture rate is 1/tau_cap_eff = 1/tau_cap + C_W N_w. Probe the rhs
    # capture-in flux (rES=rGS=0 -> only the capture-in term survives) and back out tau_cap_eff.
    tau_cap, C_W, Nw = 1.0e-12, 1.0e-14, 1.0e24
    p = QDGainParams(n_groups=1, tau_cap_s=tau_cap, auger_capture_Cw_m3_s=C_W)
    m = QDGainModel(p)
    _, dES, _ = m.rhs_fields(np.array([Nw]), np.array([[0.0]]), np.array([[0.0]]), 0.0, 0.0, p.nu0_Hz)
    cap_den = tau_cap * p.mu_ES * p.N_q_m3
    expected = Nw / cap_den * (1.0 + tau_cap * C_W * Nw)
    assert np.isclose(dES[0, 0], expected, rtol=1e-12)
    tau_cap_eff = tau_cap / (1.0 + tau_cap * C_W * Nw)          # dossier gate form
    assert np.isclose(tau_cap_eff, tau_cap / (1.0 + tau_cap * C_W * 1e24), rtol=1e-12)


def test_auger_loss_lowers_Nw_growing_with_drive():
    # 1b: loss Auger C_A N_w^3 drops steady-state N_w vs C_A=0, and the drop GROWS with drive.
    drops = []
    for drive in (10e-3, 40e-3, 100e-3):
        m0 = QDGainModel(QDGainParams(n_groups=1))
        mA = QDGainModel(QDGainParams(n_groups=1, auger_wl_C_m6_s=4e-41))
        Nw0 = m0.steady_state(drive)[0]
        NwA = mA.steady_state(drive)[0]
        assert NwA < Nw0                                       # loss Auger lowers N_w
        drops.append((Nw0 - NwA) / Nw0)
    assert drops[0] < drops[1] < drops[2]                      # drop grows with drive


def test_auger_bit_identity_defaults():
    # 1c: all-default (Auger off) results bit-identical to the pre-edit golden values.
    m = QDGainModel(QDGainParams(n_groups=5).with_detailed_balance_taus())
    y = m.steady_state(20.0e-3)
    assert all(float(a) == b for a, b in zip(y, GOLD_Y5))
    assert float(m.material_gain_per_m(m.rho_GS(y), m.p.nu0_Hz)) == GOLD_G5
    nus = m.p.nu0_Hz + np.linspace(-8e12, 8e12, 7)
    assert all(float(a) == b for a, b in zip(m.material_gain_per_m(m.rho_GS(y), nus), GOLD_SPEC5))
    m2 = QDGainModel(QDGainParams())
    y2 = m2.steady_state(15.0e-3)
    assert float(m2.material_gain_per_m(m2.rho_GS(y2), m2.p.nu0_Hz)) == GOLD_G_DEF


def test_auger_numba_fast_guard():
    # the numba fast path cannot carry the density-dependent Auger terms -> reject up front.
    pytest.importorskip("numba")                               # skip if numba absent
    with pytest.raises(ValueError):
        QDGainModel(QDGainParams(n_groups=1, auger_wl_C_m6_s=1e-41), fast=True)
    with pytest.raises(ValueError):
        QDGainModel(QDGainParams(n_groups=1, auger_capture_Cw_m3_s=1e-14), fast=True)
    # a fast model with NO Auger is still fine
    assert QDGainModel(QDGainParams(n_groups=1), fast=True) is not None


def test_auger_eh_split_conserves_and_reduces():
    # eh-split Auger: symmetric reduction (Nwe=Nwh, equal e/h times) matches the excitonic Auger rhs.
    kw = dict(n_groups=3, auger_wl_C_m6_s=4e-41, auger_capture_Cw_m3_s=1e-14, auger_capture_Ce_m3_s=1e-12)
    pe = QDGainParams(eh_split=True, **kw).with_detailed_balance_taus()
    px = QDGainParams(**kw).with_detailed_balance_taus()
    me, mx = QDGainModel(pe), QDGainModel(px)
    ng = mx.ng
    Nw = 8.0e23
    rES = np.full((1, ng), 0.6)
    rGS = np.full((1, ng), 0.8)
    dNwx, dESx, dGSx = mx.rhs_fields(np.array([Nw]), rES, rGS, 3e-2, 1e21, px.nu0_Hz)
    de = me.rhs_fields_eh(np.array([Nw]), np.array([Nw]), rES, rES, rGS, rGS, 3e-2, 1e21, pe.nu0_Hz)
    assert np.allclose(de[0], dNwx, rtol=1e-12) and np.allclose(de[1], dNwx, rtol=1e-12)
    assert np.allclose(de[2], dESx, rtol=1e-12) and np.allclose(de[4], dGSx, rtol=1e-12)


# =================== ITEM 2: WL<->ES DETAILED BALANCE ======================

def test_es_gs_detailed_balance_ratio():
    # tau_GS_ES/tau_ES_GS = (mu_GS/mu_ES) exp(dE/kT) = 7.50 at dE=70meV, T=300 (the physically
    # correct value for this rate structure; see module docstring re the dossier's 30).
    p = QDGainParams(dE_ES_GS_eV=0.070)
    pdb = p.with_detailed_balance_taus()
    ratio = pdb.tau_GS_ES_s / pdb.tau_ES_GS_s
    expect = (p.mu_GS / p.mu_ES) * np.exp(0.070 * Q_E / (KB * 300.0))
    assert np.isclose(ratio, expect, rtol=1e-2)
    assert np.isclose(ratio, 7.4975, rtol=1e-3)


def test_wl_es_full_detailed_balance_T_trend():
    # full detailed balance slaves tau_esc (ES->WL); tau_esc rises as T falls (exp dominates the
    # 1/T sheet-DOS prefactor) -> the ratio tau_esc/tau_cap at 250K > 300K > 350K.
    ratios = []
    for T in (250.0, 300.0, 350.0):
        pf = QDGainParams(dE_ES_GS_eV=0.070, T_K=T).with_full_detailed_balance()
        ratios.append(pf.tau_esc_s / pf.tau_cap_s)
    assert ratios[0] > ratios[1] > ratios[2]
    # the base ES<->GS branch stays byte-stable under with_full_detailed_balance
    p = QDGainParams(dE_ES_GS_eV=0.070)
    assert p.with_full_detailed_balance().tau_GS_ES_s == p.with_detailed_balance_taus().tau_GS_ES_s


def test_low_drive_fermi_occupation_ratio():
    # Equilibrium contract, TIGHT form: the rate equations carry (1 - rho) Pauli-blocking
    # factors, so the exact detailed-balance statement is the FERMI ratio
    #     rho_ES (1 - rho_GS) / [rho_GS (1 - rho_ES)] = exp(-dE/kT)   (per-state),
    # NOT the bare Boltzmann occupancy ratio (which it approaches only as rho -> 0; the
    # residual of the bare form ~ rho_GS, measured 6.5% at rho_GS = 0.064 -- probe-refuted
    # as a physics error and identified as the blocking correction). The Fermi form holds to
    # < 0.5% at ANY sub-degenerate drive; assert it at two operating points plus the bare
    # Boltzmann band as the legacy sanity check.
    boltz = np.exp(-0.070 * Q_E / (KB * 300.0))
    bare_low = None
    for drive_A in (0.02e-3, 0.2e-3):
        p = QDGainParams(n_groups=1, dE_ES_GS_eV=0.070).with_full_detailed_balance()
        m = QDGainModel(p)
        y = m.steady_state(drive_A)
        rgs, res = float(m.rho_GS(y)[0]), float(m.rho_ES(y)[0])
        fermi = res * (1.0 - rgs) / (rgs * (1.0 - res))
        assert abs(fermi - boltz) / boltz < 5e-3, (drive_A, fermi, boltz)
        if bare_low is None:
            bare_low = res / rgs                      # bare ratio at the LOW-occupancy point
    assert abs(bare_low - boltz) / boltz < 0.20       # bare form: loose dilute-limit band only


def test_dark_relaxation_reaches_boltzmann_equilibrium():
    # THE independent oracle for BOTH detailed-balance ladders (adversarial-verifier method,
    # which caught and now pins the corrected WL<->ES state-count ratio): an internal-only dark
    # system (no injection, photons, recombination, or spontaneous decay) conserves carrier
    # number and MUST relax to per-state Boltzmann occupations at T. rho_GS/rho_ES ->
    # exp(dE_ES_GS/kT) and rho_ES/f_WL -> exp(dE_WL_ES/kT) with f_WL the per-state WL occupation
    # (N_w/N_q)/(rho_WL_eff A_dot).
    from scipy.integrate import solve_ivp
    from dynameta.constants import HBAR
    p = QDGainParams(n_groups=1, dE_ES_GS_eV=0.060, dE_WL_ES_eV=0.080, T_K=300.0,
                     B_wl_m3_s=0.0, C_wl_m6_s=0.0, tau_sp_s=1.0e30)
    pf = p.with_full_detailed_balance()
    m = QDGainModel(pf)
    y0 = np.zeros(1 + 2 * m.ng)
    y0[0] = 1.0e21
    y0[1:] = 1.0e-3
    sol = solve_ivp(lambda t, y: m.rhs(y, 0.0, 0.0, pf.nu0_Hz), (0.0, 5.0e-8), y0,
                    method="BDF", rtol=1e-11, atol=1e-16, t_eval=[5.0e-8])
    Nw, rES, rGS = sol.y[0, -1], sol.y[1, -1], sol.y[2, -1]
    kT_eV = KB * 300.0 / Q_E
    assert abs(rGS / rES / np.exp(0.060 / kT_eV) - 1.0) < 0.02
    rho_wl_eff = pf.m_wl_eff_kg * KB * 300.0 / (np.pi * HBAR * HBAR)
    f_WL = (Nw / pf.N_q_m3) / (rho_wl_eff * (1.0 / (pf.N_q_m3 * pf.t_qd_m)))
    assert abs(rES / f_WL / np.exp(0.080 / kT_eV) - 1.0) < 0.05


def test_full_detailed_balance_stays_byte_stable_basic():
    # with_full_detailed_balance must NOT perturb with_detailed_balance_taus (superset contract).
    p = QDGainParams(n_groups=5)
    assert p.with_detailed_balance_taus().tau_GS_ES_s == p.with_full_detailed_balance().tau_GS_ES_s


# ===================== ITEM 3: ES OPTICAL SPLIT ============================

def test_es_optical_split_moves_es_peak_only():
    e_over_h = Q_E / H_PLANCK
    base = QDGainParams(n_groups=21, sigma_pk_ES_m2=5e-19)
    opt = QDGainParams(n_groups=21, sigma_pk_ES_m2=5e-19, dE_ES_GS_optical_eV=0.10)
    mb, mo = QDGainModel(base), QDGainModel(opt)
    # ES comb sits at the OPTICAL offset; GS comb untouched
    assert np.isclose((mo.nu_ES_j.mean() - mo.p.nu0_Hz) / e_over_h, 0.10, rtol=1e-6)
    assert np.isclose((mb.nu_ES_j.mean() - mb.p.nu0_Hz) / e_over_h, 0.060, rtol=1e-6)
    assert np.array_equal(mb.nu_j, mo.nu_j)
    # gain at nu0 + 0.10 e/h is much higher for the optically-shifted model
    yo, yb = mo.steady_state(40e-3), mb.steady_state(40e-3)
    nu_probe = mo.p.nu0_Hz + 0.10 * e_over_h
    g_opt = mo.total_material_gain(mo.rho_ES(yo), mo.rho_GS(yo), nu_probe)
    g_base = mb.total_material_gain(mb.rho_ES(yb), mb.rho_GS(yb), nu_probe)
    assert g_opt > 5.0 * g_base


def test_es_optical_split_default_bit_identity():
    # None (default) reuses dE_ES_GS_eV -> byte-identical ES comb + gain.
    d = QDGainParams(n_groups=11, sigma_pk_ES_m2=5e-19)
    e = QDGainParams(n_groups=11, sigma_pk_ES_m2=5e-19, dE_ES_GS_optical_eV=None)
    md, me = QDGainModel(d), QDGainModel(e)
    assert np.array_equal(md.nu_ES_j, me.nu_ES_j)


# ===================== ITEM 4: SECH LINESHAPE ==============================

def _lorentzian_area_analytic(hw):
    return np.pi * hw                                          # integral of hw^2/(dnu^2+hw^2)


def test_sech_area_conserving():
    # 4a: the sech area equals the peak-normalized Lorentzian's analytic area pi*hw to 1e-6.
    from scipy.integrate import quad
    hw = 0.5e12
    mS = QDGainModel(QDGainParams(fwhm_hom_Hz=2 * hw, lineshape="sech"))
    area = quad(lambda d: float(mS._lorentzian(np.array([d]))[0]), -400 * hw, 400 * hw, limit=800)[0]
    assert abs(area - _lorentzian_area_analytic(hw)) / _lorentzian_area_analytic(hw) < 1e-6


def test_sech_wings_kill_subgap_tail():
    # 4b: at 5*FWHM detuning the sech value is < 1e-3 of the Lorentzian value (exponential wings).
    hw = 0.5e12
    mL = QDGainModel(QDGainParams(fwhm_hom_Hz=2 * hw, lineshape="lorentzian"))
    mS = QDGainModel(QDGainParams(fwhm_hom_Hz=2 * hw, lineshape="sech"))
    d5 = 5.0 * (2.0 * hw)
    vL = float(mL._lorentzian(np.array([d5]))[0])
    vS = float(mS._lorentzian(np.array([d5]))[0])
    assert vS / vL < 1e-3


def test_sech_integrated_gain_conserved():
    # 4c: integrated material gain over nu conserved between lineshapes to 1% (same params otherwise).
    mL = QDGainModel(QDGainParams(lineshape="lorentzian"))
    mS = QDGainModel(QDGainParams(lineshape="sech"))
    nus = mL.p.nu0_Hz + np.linspace(-60e12, 60e12, 6001)
    yL = mL.steady_state(40e-3)
    yS = mS.steady_state(40e-3)
    IL = np.trapezoid(mL.material_gain_per_m(mL.rho_GS(yL), nus), nus)
    IS = np.trapezoid(mS.material_gain_per_m(mS.rho_GS(yS), nus), nus)
    assert abs(IL - IS) / abs(IL) < 1e-2


def test_lorentzian_default_bit_identity():
    # 4d: the default (lorentzian) is byte-identical to the pre-edit golden lineshape values.
    m = QDGainModel(QDGainParams(n_groups=5, lineshape="lorentzian").with_detailed_balance_taus())
    y = m.steady_state(20.0e-3)
    assert float(m.material_gain_per_m(m.rho_GS(y), m.p.nu0_Hz)) == GOLD_G5


# ================ ITEM 5: SUB-TRANSPARENCY ASE ============================

def test_ase_emission_source_continuous_through_transparency():
    from dynameta.optics.soa.ase_noise import ase_output_psd
    m = QDGainModel(QDGainParams(n_groups=1))
    nu0, Gamma, dz = m.p.nu0_Hz, m.p.Gamma, 1.0e-5
    rhos = np.linspace(0.3, 0.9, 25)
    S = []
    for rho in rhos:
        rr = np.array([rho])
        g = float(m.material_gain_per_m(rr, nu0))
        gsp = float(m.emission_gain_per_m(rr, nu0))
        S.append(ase_output_psd(np.array([g]), np.array([rho]), dz, nu0, Gamma,
                                gsp_slices=np.array([gsp]), per_pol=True))
    S = np.array(S)
    assert np.all(np.isfinite(S)) and np.all(S > 0.0)          # finite, positive everywhere
    jumps = np.abs(np.diff(S)) / S[:-1]
    assert jumps.max() < 0.5                                   # smooth, no dropout at rho=0.5


def test_ase_emission_source_quarter_at_transparency():
    from dynameta.optics.soa.ase_noise import ase_output_psd
    m = QDGainModel(QDGainParams(n_groups=1))
    nu0, Gamma, dz = m.p.nu0_Hz, m.p.Gamma, 1.0e-5
    g5 = float(m.material_gain_per_m(np.array([0.5]), nu0))
    gsp5 = float(m.emission_gain_per_m(np.array([0.5]), nu0))
    assert abs(g5) < 1e-6                                      # net gain ~ 0 at rho=1/2
    pref = m.p.N_q_m3 * m.p.mu_GS * m.p.sigma_pk_m2            # single group, L=1, w=1 -> gsp = pref*rho^2
    assert np.isclose(gsp5 / pref, 0.25, rtol=1e-9)           # source proportional to f_e f_h = 1/4
    S = ase_output_psd(np.array([g5]), np.array([0.5]), dz, nu0, Gamma,
                       gsp_slices=np.array([gsp5]), per_pol=True)
    assert np.isclose(S, Gamma * gsp5 * H_PLANCK * nu0 * dz, rtol=1e-9) and S > 0.0
    # legacy (no gsp) DROPS the source at transparency (n_sp = inf -> 0*inf guarded to 0)
    with np.errstate(invalid="ignore"):
        S_legacy = ase_output_psd(np.array([g5]), np.array([0.5]), dz, nu0, Gamma, per_pol=True)
    assert S_legacy == 0.0


def test_ase_emission_agrees_above_transparency():
    # emission-only and legacy n_sp sources coincide above transparency (g_sp = g n_sp there).
    from dynameta.optics.soa.ase_noise import ase_output_psd
    m = QDGainModel(QDGainParams(n_groups=1))
    nu0, Gamma, dz = m.p.nu0_Hz, m.p.Gamma, 1.0e-5
    rr = np.array([0.9])
    g = float(m.material_gain_per_m(rr, nu0))
    gsp = float(m.emission_gain_per_m(rr, nu0))
    Se = ase_output_psd(np.array([g]), rr, dz, nu0, Gamma, gsp_slices=np.array([gsp]), per_pol=True)
    Sl = ase_output_psd(np.array([g]), rr, dz, nu0, Gamma, per_pol=True)
    assert abs(Se - Sl) / Sl < 1e-12


def test_bidirectional_ase_emits_below_transparency():
    # pin: ase_spectrum_bidirectional uses g_sp (no n_sp division) -> a sub-transparency (g<0,
    # g_sp>0) slice still emits (finite, positive forward PSD).
    from dynameta.optics.soa.ase_noise import ase_spectrum_bidirectional
    m = QDGainModel(QDGainParams(n_groups=1))
    nu = np.array([m.p.nu0_Hz])
    rr = np.array([0.4])                                       # below transparency: g < 0
    g = m.material_gain_per_m(rr, nu)[None, :]
    gsp = m.emission_gain_per_m(rr, nu)[None, :]
    assert g[0, 0] < 0.0 and gsp[0, 0] > 0.0
    res = ase_spectrum_bidirectional(np.tile(g, (20, 1)), np.tile(gsp, (20, 1)), 1e-5, nu,
                                     np.array([1e12]), m.p.Gamma)
    assert np.isfinite(res["S_f"][0]) and res["S_f"][0] > 0.0


# ========================= ITEM 6: IMD3 ===================================

_IMD_DEV = dict(L_m=0.5e-3, tau_c_s=200e-12, E_sat_J=1.0e-12)
_PSAT = _IMD_DEV["E_sat_J"] / _IMD_DEV["tau_c_s"]              # 5 mW


def test_imd3_closed_form_scalings():
    from dynameta.optics.soa.imd import imd3_ratio
    # (P_out/P_sat)^2 scaling (Omega=0 -> H=1 exactly, isolates the pure square)
    r1 = imd3_ratio(5.0, 0.5e-3, _PSAT, 0.0, 200e-12, 0.0)
    r2 = imd3_ratio(5.0, 1.0e-3, _PSAT, 0.0, 200e-12, 0.0)
    assert np.isclose(r2 / r1, 4.0, rtol=1e-9)                 # doubling P_out -> 4x IM3/C
    # (1+alpha^2) enhancement
    r0 = imd3_ratio(5.0, 1e-3, _PSAT, 0.0, 200e-12, 0.0)
    ra = imd3_ratio(5.0, 1e-3, _PSAT, 0.0, 200e-12, 5.0)
    assert np.isclose(ra / r0, 1.0 + 25.0, rtol=1e-9)
    # 6 dB/octave field rolloff (H^2 -> factor 4 per octave far above the knee)
    tau = 200e-12
    teff = tau / (1.0 + 1e-3 / _PSAT)
    knee = 1.0 / teff
    ra2 = imd3_ratio(5.0, 1e-3, _PSAT, 8 * knee, tau, 0.0)
    rb2 = imd3_ratio(5.0, 1e-3, _PSAT, 16 * knee, tau, 0.0)
    assert 9.0 < 20.0 * np.log10(ra2 / rb2) < 15.0


def test_imd3_numeric_oracle_magnitude():
    # numeric Agrawal-Olsson oracle vs closed form within a factor of 3 at P_out/Psat ~ 0.3
    # (modest gain where the (G-1)/4 weak-compression prefactor is valid).
    from dynameta.optics.soa.imd import imd3_ratio, imd3_numeric_agrawal_olsson
    Om = 2 * np.pi * 2e7
    im3, Gmeas, Pout = imd3_numeric_agrawal_olsson(2.6, 0.30, Om, npb=1024,
                                                   n_settle_beats=60, n_meas_beats=60, **_IMD_DEV)
    cf = imd3_ratio(Gmeas, Pout, _PSAT, Om, _IMD_DEV["tau_c_s"], 0.0)
    assert 1.0 / 3.0 < cf / im3 < 3.0


def test_imd3_numeric_power_slope():
    # numeric oracle: IM3/C ~ (P_out/Psat)^2 -> log-log slope 2 (+-30%) deep below saturation.
    from dynameta.optics.soa.imd import imd3_numeric_agrawal_olsson
    Om = 2 * np.pi * 2e7
    pts = []
    for frac in (0.03, 0.06, 0.12):
        im3, _G, Pout = imd3_numeric_agrawal_olsson(2.6, frac, Om, npb=1024, n_settle_beats=60,
                                                    n_meas_beats=60, **_IMD_DEV)
        pts.append((Pout / _PSAT, im3))
    slope = (np.log10(pts[-1][1]) - np.log10(pts[0][1])) / (np.log10(pts[-1][0]) - np.log10(pts[0][0]))
    assert 1.4 < slope < 2.6


def test_imd3_numeric_rolloff():
    # numeric oracle: 6 dB/oct field rolloff -> 12 dB/oct in the squared IM3/C ratio (20 log10),
    # measured at two beat frequencies an octave apart above the knee.
    from dynameta.optics.soa.imd import imd3_numeric_agrawal_olsson, tau_eff_s
    frac = 0.30
    _, _G, Pout = imd3_numeric_agrawal_olsson(2.6, frac, 2 * np.pi * 2e7, npb=1024,
                                              n_settle_beats=40, n_meas_beats=40, **_IMD_DEV)
    knee = 1.0 / tau_eff_s(_IMD_DEV["tau_c_s"], Pout, _PSAT)
    fa, fb = 4.0 * knee, 8.0 * knee                            # both above the knee, one octave apart
    ia = imd3_numeric_agrawal_olsson(2.6, frac, fa, npb=1024, n_settle_beats=50,
                                     n_meas_beats=50, **_IMD_DEV)[0]
    ib = imd3_numeric_agrawal_olsson(2.6, frac, fb, npb=1024, n_settle_beats=50,
                                     n_meas_beats=50, **_IMD_DEV)[0]
    assert 9.0 < 20.0 * np.log10(ia / ib) < 15.0


def test_sfdr_lands_in_range():
    from dynameta.optics.soa.imd import two_tone_oip3_dbm, sfdr_db_hz23
    grid = _PSAT * np.array([0.10, 0.15, 0.20, 0.25])          # below-saturation drive sweep
    oip3 = two_tone_oip3_dbm(3.0, grid, _PSAT, 2 * np.pi * 1e7, 200e-12, 2.0)
    sfdr = sfdr_db_hz23(oip3, -160.0)
    assert 80.0 <= sfdr <= 115.0


def test_oip3_drive_independent_below_saturation():
    from dynameta.optics.soa.imd import imd3_ratio, oip3_dbm
    o = [oip3_dbm(P, imd3_ratio(4.0, P, _PSAT, 2 * np.pi * 1e7, 200e-12, 3.0))
         for P in (0.2e-3, 0.4e-3, 0.8e-3)]
    assert max(o) - min(o) < 0.1                               # constant intercept below saturation


# ===================== ITEM 7: TEMPERATURE ================================

def test_varshni_slope_and_drift():
    from dynameta.optics.soa.temperature import d_eg_dT_ev_per_K, gain_peak_drift_nm_per_K
    a, b, T = 2.76e-4, 93.0, 300.0
    manual = -a * T * (T + 2.0 * b) / (T + b) ** 2
    assert np.isclose(d_eg_dT_ev_per_K(300.0, "InAs"), manual, rtol=1e-12)
    assert 0.33 <= gain_peak_drift_nm_per_K(1300.0, "InAs", 300.0) <= 0.40


def test_qd_params_at_temperature_noop():
    from dataclasses import asdict
    from dynameta.optics.soa.temperature import qd_params_at_temperature
    p0 = QDGainParams(n_groups=5, dE_ES_GS_eV=0.10).with_detailed_balance_taus()
    p_same = qd_params_at_temperature(p0, p0.T_K, material="InAs")
    assert asdict(p0) == asdict(p_same)


def test_qd_temperature_insensitivity_and_varshni_tracking():
    from dynameta.optics.soa.temperature import qd_params_at_temperature, gain_peak_drift_nm_per_K
    base = QDGainParams(n_groups=21, dE_ES_GS_eV=0.10, T_K=293.0).with_detailed_balance_taus()
    peaks, lams = [], []
    for T in (293.0, 343.0):
        pT = qd_params_at_temperature(base, T, material="InAs", T_ref_K=293.0)
        mT = QDGainModel(pT)
        yT = mT.steady_state(40e-3)
        nus = pT.nu0_Hz + np.linspace(-30e12, 30e12, 1201)
        g = mT.material_gain_per_m(mT.rho_GS(yT), nus)
        i = int(np.argmax(g))
        peaks.append(g[i])
        lams.append(C_LIGHT / nus[i] * 1e9)
    assert abs(peaks[1] - peaks[0]) / peaks[0] < 0.10          # deep dE -> peak gain T-insensitive
    varshni_50K = gain_peak_drift_nm_per_K(lams[0], "InAs") * 50.0
    assert np.isclose(lams[1] - lams[0], varshni_50K, rtol=0.10)   # peak wavelength tracks Varshni
