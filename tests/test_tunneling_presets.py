"""v0.15 plan item 4.1.2 -- BTBT parameter sets, the Kane direct-set function and the opt-in temperature
scaling of B (carriers.tunneling), devsim-free gates. The DEVSIM gates (per-set Zener wiring, the
tuple-of-sets sum, the FDT with two BTBT terms, defaults off in the device, the builder hook) are in
tests/test_devsim_btbt_presets.py. Audit note: docs/audit/2026-11-06-v0.15-btbt-presets.md.

INDEPENDENT ORACLES (nothing below is imported from the code under test except the objects tested):
  * the printed tables RE-TYPED here with their units: Kao et al., IEEE TED 59, 292 (2012) Table III as
    reprinted in Jiao et al., J. Semicond. 34, 092002 (2013) Table 1 (page image read; the Kao primary was
    NOT readable -- closed access, imec copy embargoed), A in cm^-3 s^-1 with F in V/cm, B in MV/cm; Hurkx
    1992 (via Sentaurus UG Table 90) A in cm^-1/2 V^-5/2 s^-1, B in V/cm. The SI values are HAND-converted
    here and written as literals (A_SI = A_cgs x 1e6 cm^3/m^3 x (1e-2 m/cm)^P; B_SI = 1e8 x B[MV/cm]).
  * the Kane uniform-field rate re-coded from scipy.constants in Kane's hbar form,
    q^2 F^2 m_r^(1/2) / (18 pi hbar^2 E_g^(1/2)) exp(-pi m_r^(1/2) E_g^(3/2) / (2 q hbar F)) (g = 2), and
    the h form of Sentaurus UG Eqs. 495-496 with g. m0 is the CODATA-2018 literal 9.1093837015e-31 kg (the
    package's constant): scipy >= 1.15 ships CODATA-2022 m_e = 9.1093837139e-31, 1.36e-8 relative higher,
    which would move A and B by 6.8e-9 -- a constants-vintage difference, not a formula error, so the
    vintage is pinned here and stated.
  * Vurgaftman, Meyer & Ram-Mohan, JAP 89, 5815 (2001), RE-TYPED (re-read from refs/vurgaftman2001.txt):
    Eq. (2.13) Varshni; Table I GaAs (1.519 eV, 0.5405 meV/K, 204 K; m_e 0.067; gamma1 6.98, gamma2 2.06);
    Table III InAs (0.417 eV, 0.276 meV/K, 93 K; m_e 0.026; gamma1 20.0, gamma2 8.5); Table XIII GaInAs
    bowings (E_g 0.477 eV, m_e 0.0091, m_lh(001) 0.0202); Eq. (2.16) m0/m_lh[001] = gamma1 + 2 gamma2;
    Eq. (4.1) (1-x) A + x B - x(1-x) C at x(In) = 0.53.

FAILS ON THE OLD CODE (e7c469d; MEASURED by running this file there: 21 failed, the strict xfail fails
too) -- SILICON_KAO_*, GERMANIUM_KAO*, IN053GA047AS_KANE, kane_direct_params, btbt_at_temperature and
btbt_for_material do not exist (AttributeError); BTBTParams has no T_K / scale_B_with_gap / source / process /
verification fields (TypeError). The strict xfail (Kao consistency) is the plan's P3.
"""
import dataclasses
import math
import sys

import pytest
from scipy import constants as SC

from dynameta.carriers import tunneling as TN

EPS = sys.float_info.epsilon
M0_CODATA2018 = 9.1093837015e-31          # kg (dynameta.constants.M_E; see the module docstring)
Q, H, HBAR = SC.e, SC.h, SC.hbar          # exact-SI e and h (hbar = h / 2 pi)

# ---- the printed sets, RE-TYPED with their printed units, and their HAND-converted SI values -----------
# (name, A printed, A unit, B printed, B unit, P, A_SI literal [m^-3 s^-1 (V/m)^-P], B_SI literal [V/m], process)
PRINTED = [
    # Hurkx 1992: 4e14 cm^-1/2 V^-5/2 s^-1 = cm^-3 s^-1 (V/cm)^-2.5; x 1e6 x 1e-5 = 4e15
    ("SILICON_HURKX", 4.0e14, "cm^-1/2 V^-5/2 s^-1", 1.9e7, "V/cm", 2.5, 4.0e15, 1.9e9, "phonon_assisted"),
    # Kao Table III (Jiao Table 1), Si: direct 1.35e20 cm^-3 s^-1, 101 MV/cm, P 2 -> x 1e6 x 1e-4
    ("SILICON_KAO_DIRECT", 1.35e20, "cm^-3 s^-1 (F/1 V/cm)^2", 101.0, "MV/cm", 2.0, 1.35e22, 1.01e10, "direct"),
    # Si indirect 3.29e15 cm^-3 s^-1, 23.8 MV/cm, P 2.5 -> x 1e6 x 1e-5
    ("SILICON_KAO_INDIRECT", 3.29e15, "cm^-3 s^-1 (F/1 V/cm)^2.5", 23.8, "MV/cm", 2.5, 3.29e16, 2.38e9,
     "phonon_assisted"),
    # Ge direct 1.46e20, 6.04 MV/cm, P 2
    ("GERMANIUM_KAO_DIRECT", 1.46e20, "cm^-3 s^-1 (F/1 V/cm)^2", 6.04, "MV/cm", 2.0, 1.46e22, 6.04e8, "direct"),
    # Ge indirect 1.67e15, 6.55 MV/cm, P 2.5
    ("GERMANIUM_KAO_INDIRECT", 1.67e15, "cm^-3 s^-1 (F/1 V/cm)^2.5", 6.55, "MV/cm", 2.5, 1.67e16, 6.55e8,
     "phonon_assisted"),
]
# P1 tolerance: from_cgs does A x 1e6 x 100^-P (<= 3 roundings, |err| <= 3 eps/2 ~ 3.3e-16) against a correctly
# rounded literal (eps/2): 1e-15 relative covers it; a unit error (cm^-1 for cm^-3 = 1e4) misses by 1e19.
TOL_PIN = 1e-15


@pytest.mark.parametrize("row", PRINTED, ids=[r[0] for r in PRINTED])
def test_p1_each_set_reproduces_the_printed_table(row):
    """GATE P1: A_SI, B, P equal the printed values hand-converted above; provenance recorded."""
    name, _a, _au, _b, _bu, P, A_si, B_si, process = row
    p = getattr(TN, name)
    assert p.P == P
    assert abs(p.A_SI / A_si - 1.0) <= TOL_PIN, (p.A_SI, A_si)
    assert abs(p.B_V_per_m / B_si - 1.0) <= TOL_PIN, (p.B_V_per_m, B_si)
    assert p.process == process
    assert p.verification == "secondary"           # no primary read for any of these (module docstring)
    assert p.T_K is None and p.scale_B_with_gap is False       # T_REF numbers, scaling off by default
    if name.startswith(("SILICON_KAO", "GERMANIUM_KAO")):
        assert "Table III" in p.source and "IEEE TED 59, 292 (2012)" in p.source and "Jiao" in p.source
    else:
        assert "IEEE TED 39, 331 (1992)" in p.source
    # ADVERSARIAL: the same printed number read with a cm^-1 volume unit (x 1e2 instead of x 1e6) is off 1e4
    wrong = _a * 1e2 * 100.0 ** (-P)
    assert abs(math.log10(wrong / p.A_SI) + 4.0) < 1e-12 and abs(wrong / p.A_SI - 1.0) > TOL_PIN


def test_p1_germanium_is_the_tuple_of_both_processes():
    assert TN.GERMANIUM_KAO == (TN.GERMANIUM_KAO_DIRECT, TN.GERMANIUM_KAO_INDIRECT)
    assert [p.P for p in TN.GERMANIUM_KAO] == [2.0, 2.5]


def test_p1_hurkx_set_is_bit_identical_to_v014():
    """Defaults off (P7, devsim-free half): SILICON_HURKX keeps the v0.14 bits (float.hex re-typed from e7c469d:
    A_SI = 4000000000000000.5, B = 1.9e9, P = 2.5) and equals a bare BTBTParams of those numbers (provenance
    is not compared)."""
    p = TN.SILICON_HURKX
    assert (p.A_SI.hex(), p.B_V_per_m.hex(), p.P) == ("0x1.c6bf526340001p+51", (1.9e9).hex(), 2.5)
    assert p == TN.BTBTParams(4000000000000000.5, 1.9e9, 2.5)
    assert hash(p) == hash(TN.BTBTParams(4000000000000000.5, 1.9e9, 2.5))
    # the positional v0.14 constructor and from_cgs without keywords are unchanged
    q = TN.BTBTParams(1.0e12, 1.0e7, 2.5)
    assert (q.T_K, q.scale_B_with_gap, q.source, q.process, q.verification) == (None, False, "", None, None)
    assert TN.BTBTParams.from_cgs(4.0e14, 1.9e7, 2.5) == p


# ---- P2: the Kane function ---------------------------------------------------------------------------------
def _kane_h(Eg_eV, mr, g=2.0):
    """Sentaurus UG Eqs. 495-496, h form, F0 = 1 V/m, re-coded here."""
    E = Eg_eV * Q
    m = mr * M0_CODATA2018
    return (g * math.pi * math.sqrt(m) * Q ** 2 / (9.0 * H ** 2 * math.sqrt(E)),
            math.pi ** 2 * math.sqrt(m) * E ** 1.5 / (Q * H))


def _kane_rate_hbar(Eg_eV, mr, F):
    """Kane 1961 uniform-field direct rate, hbar form, g = 2 (re-coded here)."""
    E = Eg_eV * Q
    m = mr * M0_CODATA2018
    return (Q ** 2 * F ** 2 * math.sqrt(m) / (18.0 * math.pi * HBAR ** 2 * math.sqrt(E))
            * math.exp(-math.pi * math.sqrt(m) * E ** 1.5 / (2.0 * Q * HBAR * F)))


# P2 tolerance: ~10 roundings on each side (<= 1.1e-15) amplified in the rate by B/F <= 60 (exp) -> < 1e-13.
TOL_KANE = 1e-13
KANE_CASES = [(0.80, 0.022), (0.7372566705, 0.0234627), (3.4, 0.07), (0.36, 0.0123)]


@pytest.mark.parametrize("Eg,mr", KANE_CASES)
def test_p2_kane_function_matches_an_independent_recoding(Eg, mr):
    p = TN.kane_direct_params(Eg, mr, source="test")
    A, B = _kane_h(Eg, mr)
    assert p.P == 2.0 and p.process == "direct" and p.verification == "computed"
    assert abs(p.A_SI / A - 1.0) <= TOL_KANE and abs(p.B_V_per_m / B - 1.0) <= TOL_KANE
    for F in (1e8, 3e8, 1e9):
        rate = p.A_SI * F ** 2 * math.exp(-p.B_V_per_m / F)
        assert abs(rate / _kane_rate_hbar(Eg, mr, F) - 1.0) <= TOL_KANE
    # ADVERSARIAL: g = 1 is EXACTLY half of A (a power-of-two scaling), B unchanged
    p1 = TN.kane_direct_params(Eg, mr, 1.0, source="test")
    assert 2.0 * p1.A_SI == p.A_SI and p1.B_V_per_m == p.B_V_per_m


def test_p2_kane_function_refuses_bad_inputs():
    for args in ((0.0, 0.02), (0.8, -0.02), (0.8, 0.02, 0.0)):
        with pytest.raises(ValueError):
            TN.kane_direct_params(*args, source="x")


# ---- the computed InGaAs set: Kane-formula consistency with the re-typed Vurgaftman inputs ------------------
GAAS = (1.519, 0.5405e-3, 204.0)     # Table I
INAS = (0.417, 0.276e-3, 93.0)       # Table III
X_IN = 0.53


def _eq41(a, b, x, c):
    return (1.0 - x) * a + x * b - x * (1.0 - x) * c


def _varshni(T, e0, a, b):
    return e0 - a * T * T / (T + b)


def _ingaas_inputs(T=300.0):
    eg = _eq41(_varshni(T, *GAAS), _varshni(T, *INAS), X_IN, 0.477)
    m_e = _eq41(0.067, 0.026, X_IN, 0.0091)
    m_lh = _eq41(1.0 / (6.98 + 2 * 2.06), 1.0 / (20.0 + 2 * 8.5), X_IN, 0.0202)
    return eg, m_e, m_lh, 1.0 / (1.0 / m_e + 1.0 / m_lh)


def test_ingaas_set_is_the_kane_formula_at_the_vurgaftman_inputs():
    """GATE (brief: 'Kane-formula consistency for the computed InGaAs set'). Inputs re-derived here from the
    re-typed tables; the set must equal the Kane h form at them to TOL_KANE (the gap re-derivation differs
    from the package's by operation order only, <= 1e-15 eV, x 1.5/E_g in B: < 3e-15)."""
    eg, m_e, m_lh, mr = _ingaas_inputs()
    assert abs(eg - 0.737) < 5e-4                     # the paper's quoted 300 K value (Sec. V.B.1)
    assert abs(m_e - 0.043) < 5e-5                    # Sec. IV.A.2 "0.043 m0"
    assert abs(m_lh - 1.0 / (11.01 + 2 * 4.18)) < 5e-5   # Alavi's gamma1, gamma2 quoted in Sec. IV.A.2
    p = TN.IN053GA047AS_KANE
    A, B = _kane_h(eg, mr)
    assert abs(p.A_SI / A - 1.0) <= TOL_KANE and abs(p.B_V_per_m / B - 1.0) <= TOL_KANE
    assert p.verification == "computed" and p.process == "direct" and p.P == 2.0
    assert "COMPUTED" in p.source and "Vurgaftman" in p.source
    # ADVERSARIAL: the repo's pre-v0.14 InGaAs gap (0.763 eV, the old triple) moves B by +5.4 %
    _A_old, B_old = _kane_h(0.763059, mr)
    assert abs(B_old / p.B_V_per_m - 1.0) > 0.05


@pytest.mark.xfail(strict=True, reason=(
    "Kao 2012 primary NOT read (closed access; imec copy embargoed): its E_g and m_r inputs are unknown. With "
    "the plan's inference E_g(Gamma) = 0.80 eV, m_r = 0.022 the Kane formula reproduces the Ge-direct B (6.039 "
    "vs 6.04 MV/cm) but gives A = 1.614e20, +10.5 % over the printed 1.46e20 (no (E_g, m_r) with g = 2 matches "
    "both: 0.841 eV / 0.0189 would). Shipped as printed (user decision 19); this becomes a gate once the "
    "primary is read."))
def test_p3_kao_ge_direct_is_consistent_with_the_kane_formula():
    """PLAN P3 (strict xfail): B to 1 %, A to Kao's printed precision (3 significant figures: 0.5/146)."""
    A, B = _kane_h(0.80, 0.022)
    p = TN.GERMANIUM_KAO_DIRECT
    assert abs(p.B_V_per_m / B - 1.0) <= 0.01
    assert abs(p.A_SI / A - 1.0) <= 0.5 / 146.0


# ---- P6: ordering sanity (coarse; labelled as sanity, not a gate) --------------------------------------------
def _kane_field(p, F):
    return p.A_SI * F ** p.P * math.exp(-p.B_V_per_m / F)


def test_p6_ordering_sanity():
    """Coarse sanity (plan P6). The plan's 'the Ge direct share grows with field' is FALSE for these numbers
    and was replaced: direct/indirect = (A_d/A_i) F^-0.5 exp((B_i - B_d)/F) = 8.7e5 F^-0.5 exp(5.1e7/F) is
    DECREASING in F (P = 2.5 > 2) yet stays > 1 up to F ~ 7.6e11 V/m -- i.e. direct dominates at every
    practical field, which is Kao's own uniform-field conclusion ('direct tunneling always dominates over
    indirect tunneling in Ge', abstract), an independent check of the re-typed digits."""
    F = 3e8
    assert _kane_field(TN.GERMANIUM_KAO_DIRECT, F) > 1e3 * _kane_field(TN.SILICON_KAO_INDIRECT, F)
    fields = (1e7, 1e8, 3e8, 1e9, 1e10)
    share = [_kane_field(TN.GERMANIUM_KAO_DIRECT, f) / sum(_kane_field(p, f) for p in TN.GERMANIUM_KAO)
             for f in fields]
    assert all(s > 0.5 for s in share) and all(b < a for a, b in zip(share, share[1:]))
    sets = [TN.SILICON_HURKX, TN.SILICON_KAO_DIRECT, TN.SILICON_KAO_INDIRECT, *TN.GERMANIUM_KAO,
            TN.IN053GA047AS_KANE]
    for p in sets:
        r = [_kane_field(p, f) for f in (1e7, 1e8, 3e8, 1e9, 3e9)]
        assert all(b > a for a, b in zip(r, r[1:]))


# ---- P8: temperature scaling of B (user decision 20) -----------------------------------------------------------
def _tm(T_K=None, band_model=True, btbt=None, gap=None):
    from dynameta.carriers.bipolar_options import BipolarOptions
    from dynameta.materials import BandTemperature, TransportModel
    eg300 = _ingaas_inputs(300.0)[0] if gap is None else gap
    kw = dict(n_bg_m3=1e23, eps_static=13.9, dos_mass_kg_of_n_m3=lambda n: 0.043 * M0_CODATA2018,
              physics="bipolar_dd", n_i_m3=6.3e17, tau_srh_s=1e-9, band_gap_eV=eg300, btbt=btbt,
              hole_mobility_m2Vs_of_n_m3=lambda n: 0.03, mobility_m2Vs_of_n_m3=lambda n: 1.0,
              bipolar_options=None if T_K is None else BipolarOptions(T_K=T_K))
    if band_model:
        kw["band_temperature"] = BandTemperature("InGaAs_LM", affinity_fraction=1.0)
    return TransportModel(**kw)


def _ratio_oracle(T, T0=300.0):
    """E_g(T)/E_g(T0) of the anchored model: band_gap_eV (the 300 K value) + V(T) - V(300), V re-coded."""
    eg300 = _ingaas_inputs(300.0)[0]
    v = lambda t: _ingaas_inputs(t)[0]                                    # noqa: E731
    return (eg300 + v(T) - v(300.0)) / (eg300 + v(T0) - v(300.0))


# P8 tolerance: the ratio is ~6 roundings on each side (gap differences of O(0.05 eV) on 0.7 eV: ~1e-15),
# x 1.5 through the power, plus the product: < 5e-15 relative; 1e-14 is the plan's figure.
TOL_T = 1e-14


@pytest.mark.parametrize("T", [250.0, 350.0, 400.0])
def test_p8_B_scales_as_the_gap_to_the_three_halves(T):
    base = dataclasses.replace(TN.IN053GA047AS_KANE, scale_B_with_gap=True)
    out = TN.btbt_for_material(base, _tm(T_K=T, btbt=base))
    want = base.B_V_per_m * _ratio_oracle(T) ** 1.5
    assert abs(out.B_V_per_m / want - 1.0) <= TOL_T
    assert out.A_SI == base.A_SI and out.P == base.P          # REFUTE: A must NOT change (bit for bit)
    assert out.T_K == T and out.scale_B_with_gap
    # ADVERSARIAL: a linear law (exponent 1) or the inverted ratio misses by far more than TOL_T
    assert abs(base.B_V_per_m * _ratio_oracle(T) / want - 1.0) > 1e3 * TOL_T
    assert abs(base.B_V_per_m * _ratio_oracle(T) ** -1.5 / want - 1.0) > 1e3 * TOL_T
    # the public kernel with an explicit ratio gives the same bits
    from dynameta.materials import band_gap_ratio
    r = band_gap_ratio(_tm(T_K=T), T)
    assert TN.btbt_at_temperature(base, T, r) == out


def test_p8_identity_at_300K_is_bitwise():
    base = dataclasses.replace(TN.IN053GA047AS_KANE, scale_B_with_gap=True)
    for tr in (_tm(T_K=None, btbt=base), _tm(T_K=300.0, btbt=base)):
        out = TN.btbt_for_material(base, tr)
        assert out.B_V_per_m.hex() == base.B_V_per_m.hex() and out.A_SI.hex() == base.A_SI.hex()
        assert out.quoted_T_K == 300.0


def test_p8_scaling_is_opt_in_and_needs_a_band_model():
    """Defaults off: a set WITHOUT the flag passes through as the same object, model or not; a flagged set
    without a band-temperature model passes unscaled (setup_btbt then warns -- devsim test); None -> None."""
    plain = TN.IN053GA047AS_KANE
    assert TN.btbt_for_material(plain, _tm(T_K=350.0, btbt=plain)) is plain
    assert TN.btbt_for_material(None, _tm(T_K=350.0)) is None
    flagged = dataclasses.replace(plain, scale_B_with_gap=True)
    assert TN.btbt_for_material(flagged, _tm(T_K=350.0, band_model=False, btbt=flagged)) is flagged
    # a tuple: only the flagged set is scaled, order kept
    tup = (plain, flagged)
    out = TN.btbt_for_material(tup, _tm(T_K=350.0, btbt=tup))
    assert out[0] is plain and out[1].B_V_per_m != flagged.B_V_per_m and isinstance(out, tuple)
    assert TN.btbt_for_material(TN.GERMANIUM_KAO, _tm(T_K=350.0)) is TN.GERMANIUM_KAO


def test_p8_set_quoted_at_another_temperature_scales_from_there():
    """A set fitted at 250 K (T_K=250) and run at 350 K scales by E_g(350)/E_g(250)."""
    base = dataclasses.replace(TN.IN053GA047AS_KANE, scale_B_with_gap=True, T_K=250.0)
    out = TN.btbt_for_material(base, _tm(T_K=350.0, btbt=base))
    assert abs(out.B_V_per_m / (base.B_V_per_m * _ratio_oracle(350.0, 250.0) ** 1.5) - 1.0) <= TOL_T
    with pytest.raises(ValueError, match="different temperatures"):
        TN.btbt_at_temperature((base, dataclasses.replace(base, T_K=None)), 350.0, 1.01)


def test_params_validation():
    with pytest.raises(ValueError):
        TN.BTBTParams(1.0, 1.0, 2.0, T_K=0.0)
    with pytest.raises(TypeError):
        TN.BTBTParams(1.0, 1.0, 2.0, scale_B_with_gap=1)
    with pytest.raises(ValueError):
        TN.BTBTParams(1.0, 1.0, 2.0, process="phonon")
    with pytest.raises(ValueError):
        TN.BTBTParams(1.0, 1.0, 2.0, verification="trusted")
    for bad in ((), [TN.SILICON_HURKX, 3], "x"):
        with pytest.raises(TypeError):
            TN.btbt_at_temperature(bad, 300.0, 1.0)
    with pytest.raises(ValueError):
        TN.btbt_at_temperature(TN.SILICON_HURKX, 300.0, -1.0)
