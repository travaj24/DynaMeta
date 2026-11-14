"""Devsim-free gates of the v0.15 Si bundles (consolidated plan item 4.2.1, P2 items 1-2: P1-P5, Q1-Q4;
user decisions 26-29): dynameta.materials.semiconductor_presets and the paired BGN models / pairing
check of dynameta.carriers.bipolar_options. DEVSIM gates: tests/test_si_bundles_devsim.py. Audit note:
docs/audit/2026-11-06-v0.15-si-bundles.md.

ORACLES: Yan & Cuevas, J. Appl. Phys. 114, 044508 (2013) constants re-typed here in CGS with their
equation / table numbers (read in the primary text); F_1/2 by scipy quadrature written here; k/q from
scipy.constants. FAILS ON OLD CODE (4b6ca37): the module and the 'power_log' kind / pairing fields do not
exist -- the imports below raise at collection, which fails the file (no importorskip on purpose).
"""
import dataclasses
import hashlib
import math

import numpy as np
import pytest
from scipy.constants import e as QE, k as KB
from scipy.integrate import quad
from scipy.optimize import brentq

from dynameta.carriers import bipolar_options as BO
from dynameta.materials import semiconductor_presets as SP

VT300 = KB * 300.0 / QE
CM3 = 1e6
# --- Yan & Cuevas 2013, re-typed --------------------------------------------------------------------
A_FD, NREF_FD, B_FD = 0.042e-3, 1e14, 3       # Table I "Fermi-Dirac based BGN (this work)", Eq. (3), C = 0
A_FD_SD = 0.003e-3                             # its quoted +/- (one standard deviation, Sec. IV)
A_BZ, NREF_BZ, B_BZ = 12.96e-3, 1e17, 1        # Table I "Boltzmann based apparent BGN (this work)", C = 0
A_BZ_SD = 0.70e-3
NI_YC, NC_YC, NV_YC = 9.65e9, 2.86e19, 3.1e19  # Sec. III.B (n_i, Ref. 32); Sec. I (N_c, N_v, Ref. 3)


def _F12(eta):
    v, _ = quad(lambda x: math.sqrt(x) / (1.0 + math.exp(min(x - eta, 700.0))), 0.0, max(eta, 0.0) + 60.0,
                points=[eta] if eta > 0 else None, epsabs=0.0, epsrel=1e-13, limit=400)
    return 2.0 / math.sqrt(math.pi) * v


def _F12inv(u):
    return brentq(lambda t: _F12(t) - u, -60.0, 200.0, xtol=1e-15, rtol=1e-15)


def _power_log(N, A, nref, b):
    return A * math.log(N / nref) ** b if N > nref else 0.0


def test_p1_yan_cuevas_constants_are_the_papers():
    """P1 value pins: every constant of the two house sets equals Table I / Sec. I / Sec. III.B, exactly.
    Adversarial: a frozen sha256 of the preset reprs catches a silent edit under an unchanged name (a
    value change must get a NEW preset name)."""
    fd, bz = BO.BGN_SI_YAN_CUEVAS_2013_FD, BO.BGN_SI_YAN_CUEVAS_2013_BOLTZMANN
    assert (fd.kind, fd.E_ref_eV, fd.N_ref_m3, fd.exponent_b, fd.offset_eV) == ("power_log", A_FD,
                                                                                NREF_FD * CM3, B_FD, 0.0)
    assert (bz.kind, bz.E_ref_eV, bz.N_ref_m3, bz.exponent_b, bz.offset_eV) == ("power_log", A_BZ,
                                                                                NREF_BZ * CM3, B_BZ, 0.0)
    for b in (fd, bz):
        assert b.doping_basis == "donors" and b.outside_basis == "refuse" and b.verification == "primary"
        assert b.paired_n_i_m3 == NI_YC * CM3 and b.paired_T_K == 300.0 and "Table I" in b.locator
    assert fd.paired_statistics == "fermi_dirac" and bz.paired_statistics == "boltzmann"
    assert fd.paired_N_c_m3 == NC_YC * CM3 and fd.paired_N_v_m3 == NV_YC * CM3
    p = SP.SI_PAIRING_YC2013_FD
    assert (p.n_i_m3, p.N_c_m3, p.N_v_m3, p.T_K) == (NI_YC * CM3, NC_YC * CM3, NV_YC * CM3, 300.0)
    assert p.bgn_n is fd and p.bgn_p is None and SP.SI_PAIRING_YC2013_BOLTZMANN.bgn_n is bz
    frozen = repr((fd, bz, SP.SI_PAIRING_YC2013_FD, SP.SI_PAIRING_YC2013_BOLTZMANN))
    assert hashlib.sha256(frozen.encode()).hexdigest() == \
        "f86ba7494157bdfcc06ffab7de7c633cfeebe4b5cc1406258d8bc3529c2a699d"


def test_p2_power_log_formula_hand_values_and_clamp():
    """P2: Eq. (3) evaluated by hand (re-typed constants) at 1e18 / 1e19 / 1e20 cm^-3 -- FD 32.815 /
    64.092 / 110.752 meV, Boltzmann 29.842 / 59.683 / 89.525 meV -- to 1e-13 (one log and one pow). The
    clamp below N_ref (our choice, stated in the module doc) gives 0 there; N <= 0 gives 0."""
    for N in (1e18, 1e19, 1e20):
        assert float(BO.BGN_SI_YAN_CUEVAS_2013_FD.delta_Eg_eV(N * CM3)) == pytest.approx(
            _power_log(N, A_FD, NREF_FD, B_FD), rel=1e-13)
        assert float(BO.BGN_SI_YAN_CUEVAS_2013_BOLTZMANN.delta_Eg_eV(N * CM3)) == pytest.approx(
            _power_log(N, A_BZ, NREF_BZ, B_BZ), rel=1e-13)
    assert _power_log(1e19, A_FD, NREF_FD, B_FD) == pytest.approx(64.092e-3, abs=1e-6)
    assert _power_log(1e20, A_BZ, NREF_BZ, B_BZ) == pytest.approx(89.525e-3, abs=1e-6)
    for N in (0.0, -1e20, 1e13, 1e14):
        assert float(BO.BGN_SI_YAN_CUEVAS_2013_FD.delta_Eg_eV(N * CM3)) == 0.0
    assert float(BO.BGN_SI_YAN_CUEVAS_2013_BOLTZMANN.delta_Eg_eV(5e16 * CM3)) == 0.0
    with pytest.raises(ValueError, match="power_log"):
        BO.BGNModel(kind="slotboom", E_ref_eV=9e-3, N_ref_m3=1e23, exponent_b=3.0)


def test_p3_fd_real_and_boltzmann_apparent_sets_agree_on_n0p0():
    """P3 (adversarial against transcription; plan 4.2.1 gate 1). Yan & Cuevas state that both their
    parameterisations reproduce the same n0 p0 over the diffused range (Fig. 3). Independently: (i) FD
    statistics with the SHIPPED real BGN, n0 p0 = n_i^2 (n0/N_c) exp(-F_1/2^-1(n0/N_c)) exp(dEg/kT)
    (Eq. (1), scipy F_1/2); (ii) Boltzmann n_i^2 exp(dEg_app/kT) with the SHIPPED apparent set. Over
    N_D = 1e18 .. 5e19 cm^-3 |ratio - 1| < 0.3 (the plan's bound). Its derivation, RE-CHECKED here: the
    quoted one-sigma +/-0.003 meV and +/-0.70 meV on A, at ln(N/N_ref) = 9.2..13.1 (FD, cubed) and
    2.3..6.2 (Boltzmann), move n_ie^2 by 9.5..30 % and 6.4..18 %; combined in quadrature (independent
    fits) 11.5 % at 1e18, 23.5 % at 1e19 (the plan's "about 25 %"), 35 % at 5e19 -- so the flat 0.3 is
    ~1.3 sigma at 1e19 and STRICTER than one sigma at the top of the range (the plan's "+/-19 %, +/-13 %"
    stopped at 1e19). Each point must also lie within its own one sigma. MEASURED ratios 1.108 / 1.039 /
    1.049 / 1.034 / 0.953 (max 0.108 at 1e18, inside that point's 11.5 %). A constant mistyped by 10x
    misses by e^(several): asserted."""
    def ratio(fd_set, bz_set, N):
        u = N / NC_YC
        fd_np = NI_YC ** 2 * u * math.exp(-_F12inv(u)) * math.exp(float(fd_set.delta_Eg_eV(N * CM3)) / VT300)
        bz_np = NI_YC ** 2 * math.exp(float(bz_set.delta_Eg_eV(N * CM3)) / VT300)
        return fd_np / bz_np
    grid = (1e18, 3e18, 1e19, 3e19, 5e19)
    def sigma(N):           # one-sigma n_ie^2 spread of the two fits, in quadrature
        return math.hypot(math.expm1(A_FD_SD * math.log(N / NREF_FD) ** 3 / VT300),
                          math.expm1(A_BZ_SD * math.log(N / NREF_BZ) / VT300))
    assert sigma(1e19) == pytest.approx(0.235, abs=5e-3) and 0.3 < 1.4 * sigma(1e19)
    for N in grid:
        r = ratio(BO.BGN_SI_YAN_CUEVAS_2013_FD, BO.BGN_SI_YAN_CUEVAS_2013_BOLTZMANN, N)
        assert abs(r - 1.0) < 0.3 and abs(r - 1.0) < sigma(N)
    bad = dataclasses.replace(BO.BGN_SI_YAN_CUEVAS_2013_FD, E_ref_eV=10 * A_FD)
    assert abs(ratio(bad, BO.BGN_SI_YAN_CUEVAS_2013_BOLTZMANN, 1e19) - 1.0) > 10.0


def test_p4_statistical_gap_and_dos_masses():
    """P4: band_gap_eV = kT ln(N_c N_v / n_i^2) (scipy constants) = 1.12973 eV to 1e-12 for n_i 9.65e9, and
    the DOS masses reproduce N_c, N_v through the library's effective_dos_m3 (consistency, 1e-13) and
    through the textbook form 2 (2 pi m kT/h^2)^1.5 (oracle; 1.8e-9 budget = 3x the library's
    CODATA-rounded HBAR literal). The optical 1.124 eV would be refused by the 2-D heterojunction check
    (_HJ_EG_TOL_EV = 5 meV): it is 5.73 meV off -- why the bundles carry the statistical gap."""
    from dynameta.constants import HBAR
    from dynameta.carriers.physics_equilibrium import effective_dos_m3
    for p in SP.SI_PAIRINGS:
        eg = VT300 * math.log(p.N_c_m3 * p.N_v_m3 / p.n_i_m3 ** 2)
        assert p.band_gap_eV == pytest.approx(eg, rel=1e-12)
        assert eg == pytest.approx(1.12973, abs=5e-6)
        tol = 3.0 * abs(HBAR * 2.0 * math.pi / 6.62607015e-34 - 1.0) + 1e-13
        for N in (p.N_c_m3, p.N_v_m3):
            m = p.dos_mass_kg(N)
            assert effective_dos_m3(m) == pytest.approx(N, rel=1e-13)
            assert 2.0 * (2.0 * math.pi * m * KB * 300.0 / 6.62607015e-34 ** 2) ** 1.5 == pytest.approx(N, rel=tol)
    assert abs(SP.SI_PAIRING_YC2013_FD.band_gap_eV - 1.124) > 0.005
    assert SP.SI_PAIRING_YC2013_FD.dos_mass_kg(NC_YC * CM3) / 9.1093837015e-31 == pytest.approx(1.0911, abs=1e-4)
    assert SP.SI_PAIRING_YC2013_FD.dos_mass_kg(NV_YC * CM3) / 9.1093837015e-31 == pytest.approx(1.1513, abs=1e-4)


def test_p5_opt_in_identity_and_the_transport_factory():
    """P5: nothing is applied automatically -- a TransportModel built without presets has
    bipolar_options None and carrier_statistics None; MaterialRegistry starts empty. carrier_statistics
    validates (value; bipolar_dd only). silicon_transport sets the pairing's n_i, statistical gap,
    statistics, chi 4.05 eV at 300 K, and REQUIRES the mobilities (no default)."""
    from dynameta.constants import M_E
    from dynameta.materials import MaterialRegistry, TransportModel
    const = lambda v: (lambda n: np.full_like(np.asarray(n, float), v))          # noqa: E731
    base = dict(n_bg_m3=1e24, eps_static=11.7, dos_mass_kg_of_n_m3=const(1.08 * M_E),
                mobility_m2Vs_of_n_m3=const(0.135))
    tr = TransportModel(physics="bipolar_dd", n_i_m3=1e16, tau_srh_s=1e-7,
                        hole_mobility_m2Vs_of_n_m3=const(0.048), **base)
    assert tr.bipolar_options is None and tr.carrier_statistics is None
    assert MaterialRegistry().names() == [] and len(MaterialRegistry()) == 0
    with pytest.raises(ValueError, match="carrier_statistics must be"):
        dataclasses.replace(tr, carrier_statistics="fd")
    with pytest.raises(ValueError, match="bipolar_dd"):
        TransportModel(physics="drift_diffusion", carrier_statistics="boltzmann", **base)
    t = SP.silicon_transport(n_bg_m3=1e25, mobility_n_m2Vs=0.1, mobility_p_m2Vs=0.04, tau_srh_s=1e-6)
    assert (t.n_i_m3, t.carrier_statistics, t.chi_eV, t.chi_reference_T_K, t.physics) == (
        NI_YC * CM3, "fermi_dirac", 4.05, 300.0, "bipolar_dd")
    assert t.band_gap_eV == SP.SI_PAIRING_YC2013_FD.band_gap_eV
    assert t.bipolar_options.bgn is BO.BGN_SI_YAN_CUEVAS_2013_FD
    assert float(t.mobility_m2Vs_of_n_m3(1e25)) == 0.1 and float(t.hole_mobility_m2Vs_of_n_m3(1e25)) == 0.04
    with pytest.raises(TypeError):
        SP.silicon_transport(n_bg_m3=1e25, tau_srh_s=1e-6)                         # mobilities required


# --- the pairing check (Q1-Q4), pure -----------------------------------------------------------------

def _check(**kw):
    args = dict(n_i_m3=NI_YC * CM3, N_c_m3=NC_YC * CM3, N_v_m3=NV_YC * CM3, statistics="fermi_dirac",
                T_K=None)
    args.update(kw)
    bgn = args.pop("bgn", BO.BGN_SI_YAN_CUEVAS_2013_FD)
    return BO.check_bgn_pairing(bgn, **args)


def test_q2_n_i_mismatch_is_refused_with_its_gap_equivalent():
    """Q2: n_i = 1.0e10 against the 9.65e9 pairing raises BGNPairingError whose message carries the gap
    equivalent kT ln((1e10/9.65e9)^2) = 1.84 meV (scipy constants, here) and n_ie^2's +7.4 %; 9.65e9
    passes; the 1e-3 tolerance edge (n_i quoted to 3 figures; 0.05 meV) is exact: 0.9e-3 off passes,
    1.1e-3 off raises. The N_c / N_v of an FD pairing are checked the same way."""
    gap_meV = 1e3 * VT300 * math.log((1e10 / NI_YC) ** 2)
    assert round(gap_meV, 2) == 1.84
    with pytest.raises(BO.BGNPairingError) as err:
        _check(n_i_m3=1e16)
    assert "{:+.2f} meV".format(gap_meV) in str(err.value) and "+7.4 %" in str(err.value)
    assert _check() is None
    assert _check(n_i_m3=NI_YC * CM3 * (1 + 0.9e-3)) is None
    with pytest.raises(BO.BGNPairingError):
        _check(n_i_m3=NI_YC * CM3 * (1 + 1.1e-3))
    with pytest.raises(BO.BGNPairingError, match="N_c"):
        _check(N_c_m3=2.8e25)
    assert _check(bgn=BO.BGN_SI_YAN_CUEVAS_2013_BOLTZMANN, N_c_m3=2.8e25, statistics="boltzmann") is None
    assert _check(bgn=BO.BGN_SI_DEL_ALAMO_1985, n_i_m3=1e16) is None             # legacy: never checked


def test_q3_statistics_mismatch_is_refused():
    """Q3: an FD-paired set on the legacy statistics ('fermi_dirac_legacy': FD transport with the
    Boltzmann-product detailed balance) or on Boltzmann raises; a Boltzmann-paired (apparent) set on any
    FD statistics raises (the double count); matching statistics pass. resolve_statistics maps the
    setup_bipolar_region pair and refuses a contradiction. REVISED 2026-11-07 (v0.15.0 default): the unset
    statistics with the FD g-factor resolve to 'fermi_dirac'; 'legacy' resolves to 'fermi_dirac_legacy' (and
    the label is accepted back)."""
    for stats in ("fermi_dirac_legacy", "boltzmann"):
        with pytest.raises(BO.BGNPairingError, match="fermi_dirac"):
            _check(statistics=stats)
    for stats in ("fermi_dirac_legacy", "fermi_dirac"):
        with pytest.raises(BO.BGNPairingError, match="twice"):
            _check(bgn=BO.BGN_SI_YAN_CUEVAS_2013_BOLTZMANN, statistics=stats)
    assert _check(bgn=BO.BGN_SI_YAN_CUEVAS_2013_BOLTZMANN, statistics="boltzmann") is None
    assert BO.resolve_statistics(None, True) == "fermi_dirac"
    assert BO.resolve_statistics("legacy", True) == "fermi_dirac_legacy"
    assert BO.resolve_statistics("fermi_dirac_legacy", True) == "fermi_dirac_legacy"
    assert BO.resolve_statistics(None, False) == "boltzmann"
    assert BO.resolve_statistics("fermi_dirac", True) == "fermi_dirac"
    for bad in (("boltzmann", True), ("fermi_dirac", False), ("legacy", False)):
        with pytest.raises(ValueError, match="contradicts"):
            BO.resolve_statistics(*bad)


def test_q4_temperature_refusal_and_the_override():
    """Q4 (decision 29): T_K = 350 K with a paired preset raises; within 0.5 K of 300 K passes; with
    allow=True (BipolarOptions.allow_unpaired_bgn) the message is RETURNED for the caller's one warning."""
    with pytest.raises(BO.BGNPairingError, match="350 K"):
        _check(T_K=350.0)
    assert _check(T_K=300.4) is None
    msg = _check(T_K=350.0, allow=True)
    assert isinstance(msg, str) and "allow_unpaired_bgn" in msg


def test_q1_every_bundle_is_internally_paired_and_a_mixed_bundle_is_caught():
    """Q1: every shipped bundle satisfies the gap identity and carries BGN sets paired with its own n_i,
    statistics, T (and N_c, N_v for FD). Adversarial: a bundle assembled from mismatched sets (the
    Boltzmann set in the FD bundle; an FD set re-paired to another n_i) is refused at construction."""
    for p in SP.SI_PAIRINGS:
        for b in (p.bgn_n, p.bgn_p):
            if b is not None:
                assert _check(bgn=b, n_i_m3=p.n_i_m3, N_c_m3=p.N_c_m3, N_v_m3=p.N_v_m3,
                              statistics=p.statistics) is None
    with pytest.raises(ValueError, match="not paired"):
        dataclasses.replace(SP.SI_PAIRING_YC2013_FD, bgn_n=BO.BGN_SI_YAN_CUEVAS_2013_BOLTZMANN)
    with pytest.raises(ValueError, match="not paired"):
        dataclasses.replace(SP.SI_PAIRING_YC2013_FD,
                            bgn_n=dataclasses.replace(BO.BGN_SI_YAN_CUEVAS_2013_FD, paired_n_i_m3=1e16))
    with pytest.raises(ValueError, match="BOTH"):
        BO.BGNModel(kind="power_log", E_ref_eV=1e-3, N_ref_m3=1e20, paired_n_i_m3=1e16)
    with pytest.raises(ValueError, match="paired_N_c_m3"):
        BO.BGNModel(kind="power_log", E_ref_eV=1e-3, N_ref_m3=1e20, paired_n_i_m3=1e16,
                    paired_statistics="fermi_dirac")


# --- p-type refusal (the 2014 constants are pending) --------------------------------------------------

def test_p_type_nodes_are_refused_until_the_2014_constants_are_read():
    """Decision 26 / plan 1.4: the house sets are n-type only; a p-type node above N_ref raises
    BGNFitPendingError naming Yan & Cuevas JAP 116, 194505 (2014) -- in equilibrium_fields (every setup
    path) and at construction in silicon_transport(acceptor=True). n-type and compensated-n nodes are not
    refused; bgn=False builds a p-type region knowingly. (Boltzmann pairing here: the FD root would need
    the DEVSIM module's Delta.)"""
    o = BO.BipolarOptions(bgn=BO.BGN_SI_YAN_CUEVAS_2013_BOLTZMANN)
    kw = dict(n_i_m3=NI_YC * CM3, N_c_m3=NC_YC * CM3, N_v_m3=NV_YC * CM3, options=o, fd_enhancement=False)
    with pytest.raises(BO.BGNFitPendingError, match="194505"):
        BO.equilibrium_fields(np.array([1e25, -1e24]), **kw)          # N_A 1e18 > N_ref 1e17 cm^-3
    f = BO.equilibrium_fields(np.array([1e25, 1e25, -1e23]), donors_m3=np.array([1e25, 1e25, 0.0]),
                              acceptors_m3=np.array([0.0, 1e21, 1e23]), **kw)  # n, compensated n, p AT N_ref
    assert np.all(f["dEg_eV"][:2] > 0.05) and f["dEg_eV"][2] == 0.0
    with pytest.raises(BO.BGNFitPendingError, match="194505"):
        SP.silicon_transport(n_bg_m3=1e23, acceptor=True, mobility_n_m2Vs=0.1, mobility_p_m2Vs=0.04,
                             tau_srh_s=1e-6)
    t = SP.silicon_transport(n_bg_m3=1e23, acceptor=True, mobility_n_m2Vs=0.1, mobility_p_m2Vs=0.04,
                             tau_srh_s=1e-6, options=SP.silicon_bipolar_options(bgn=False))
    assert t.bipolar_options.bgn is None


def test_legacy_equilibrium_fields_are_unchanged_by_the_statistics_keyword():
    """Defaults off (pure half): equilibrium_fields with statistics=None is BITWISE the call without the
    keyword for the legacy presets (Boltzmann frame; the FD frame runs in the DEVSIM file), and returns
    pairing_advisory None for unpaired sets."""
    for bgn in (BO.BGN_SI_SLOTBOOM_1977, BO.BGN_SI_KLAASSEN_1992, BO.BGN_SI_DEL_ALAMO_1985):
        o = BO.BipolarOptions(bgn=bgn, ionization=BO.IONIZATION_SI_P_B)
        N = np.array([1e25, -3e24, 1e22, 1e26])
        a = BO.equilibrium_fields(N, n_i_m3=1e16, N_c_m3=2.8e25, N_v_m3=1.04e25, options=o, fd_enhancement=False)
        b = BO.equilibrium_fields(N, n_i_m3=1e16, N_c_m3=2.8e25, N_v_m3=1.04e25, options=o, fd_enhancement=False,
                                  statistics=None)
        for k in ("dEg_eV", "n_ie", "n0", "p0", "psi_eq", "seed_shift"):
            assert a[k].tobytes() == b[k].tobytes()
        assert a["pairing_advisory"] is None and a["n_ie0"].tobytes() == a["n_ie"].tobytes()
