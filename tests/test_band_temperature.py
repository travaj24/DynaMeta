"""v0.15 plan 4.3.3 -- the band-temperature model (materials.band_temperature), devsim-free gates.

Design + measured numbers: docs/audit/2026-11-06-v0.15-band-temperature.md. The DEVSIM gates (V_bi(T),
Delta E_c(T), the thermionic conductance with an independent barrier, the builder A/B identity) are in
tests/test_devsim_band_temperature.py.

INDEPENDENT ORACLE: Vurgaftman, Meyer & Ram-Mohan, J. Appl. Phys. 89, 5815 (2001), re-read from the paper
and RE-TYPED here (not imported): Eq. (2.13) E_g(T) = E_g(0) - alpha T^2/(T + beta); Table I GaAs
(1.519 eV, 0.5405 meV/K, 204 K); Table II AlAs Gamma (3.099 eV, 0.885 meV/K, 530 K); Eq. (4.1) with the
Table XII AlGaAs Gamma bowing C = -0.127 + 1.310 x, temperature-independent (Sec. IV.A.1). Every formula
below is written out in the test.

FAILS ON THE OLD CODE (cc0bdf0, MEASURED by running this file there: 10 of 10 fail): materials.band_temperature
does not exist (ImportError inside every test), physics_heterojunction has no affinity_temperature_tags, and
TransportModel has no band_temperature field (KeyError in test_defaults_off); optics.soa.temperature had no
AlAs row (varshni_eg_ev(T, "AlAs") raised ValueError).
"""
import dataclasses
import math
import sys

import pytest

from dynameta.constants import KB, M_E, Q_E, T_REF

EPS = sys.float_info.epsilon

# ---- the re-typed tables (meV/K -> eV/K written as 1e-3 * the printed value) --------------------------
GAAS_T1 = (1.519, 0.5405 * 1e-3, 204.0)          # Table I
ALAS_T2 = (3.099, 0.885 * 1e-3, 530.0)           # Table II, Gamma valley


def varshni(T, e0, a, b):
    """Eq. (2.13)."""
    return e0 - a * T * T / (T + b)


def bowing_algaas(x):
    """Table XII, Gamma gap: C = -0.127 + 1.310 x [eV]."""
    return -0.127 + 1.310 * x


def gap_algaas(T, x):
    """Eq. (4.1) with A = GaAs, B = AlAs, x = Al fraction, T-independent bowing."""
    return (1.0 - x) * varshni(T, *GAAS_T1) + x * varshni(T, *ALAS_T2) - x * (1.0 - x) * bowing_algaas(x)


# Varshni gate tolerance: the plan's 1e-12 eV. Round-off of either side is <= ~15 double operations on
# O(3 eV) terms, ~7e-15 eV (MEASURED max |diff| 0.0: the same operation order), so 1e-12 leaves > 100x headroom while a
# mis-typed parameter (>= 1 in the last printed digit: >= 1e-4 eV at 300 K for alpha) misses it by 1e8.
TOL_EG = 1e-12
TEMPS = (0.0, 77.0, 200.0, 250.0, 300.0, 350.0, 400.0)


def _tm(**kw):
    from dynameta.materials import TransportModel
    base = dict(n_bg_m3=1e23, eps_static=12.9, dos_mass_kg_of_n_m3=lambda n: 0.067 * M_E)
    base.update(kw)
    return TransportModel(**base)


def _hex(v):
    return None if v is None else float(v).hex()


# =====================================================================================================
# H2-a: the Varshni numbers
# =====================================================================================================
def test_varshni_values_match_the_retyped_vurgaftman_tables():
    """GATE H2-a. GaAs, AlAs and Al(x)Ga(1-x)As (x = 0.1, 0.3, 0.35) at 0-400 K from the ONE package kernel
    vs the test's own Eq. (2.13) / (4.1) with the re-typed Table I / II / XII values, to 1e-12 eV; the plan's
    own arithmetic (GaAs 1.422482 eV and AlAs Gamma 3.003036 eV at 300 K, rounded to 1e-6) to 5e-7 eV; and
    the bowing is temperature-independent (the alloy minus its linear interpolation is the same at every T).
    Old code: no AlAs row (ValueError) and no algaas_gap_law (ImportError)."""
    from dynameta.materials.band_temperature import VARSHNI_PARAMS, algaas_gap_law, varshni_eg_ev
    assert VARSHNI_PARAMS["GaAs"] == pytest.approx(GAAS_T1, rel=1e-15)
    assert VARSHNI_PARAMS["AlAs"] == pytest.approx(ALAS_T2, rel=1e-15)
    for T in TEMPS:
        assert abs(varshni_eg_ev(T, "GaAs") - varshni(T, *GAAS_T1)) < TOL_EG
        assert abs(varshni_eg_ev(T, "AlAs") - varshni(T, *ALAS_T2)) < TOL_EG
        for x in (0.1, 0.3, 0.35):
            assert abs(varshni_eg_ev(T, algaas_gap_law(x)) - gap_algaas(T, x)) < TOL_EG, (T, x)
    assert abs(varshni_eg_ev(300.0, "GaAs") - 1.422482) < 5e-7          # P3 plan section 3.1
    assert abs(varshni_eg_ev(300.0, "AlAs") - 3.003036) < 5e-7
    lin = [varshni_eg_ev(T, algaas_gap_law(0.3)) - (0.7 * varshni_eg_ev(T, "GaAs") + 0.3 * varshni_eg_ev(T, "AlAs"))
           for T in TEMPS]
    assert max(lin) - min(lin) < TOL_EG                                  # -x(1-x)C at every T
    assert lin[0] == pytest.approx(-0.3 * 0.7 * bowing_algaas(0.3), abs=TOL_EG)
    # adversarial: a swapped composition and a Table XII constant off by one in its last printed digit
    # (1.311 x) each break the 1e-12 gate by orders of magnitude
    assert abs(varshni_eg_ev(300.0, ("GaAs", "AlAs", 0.7, bowing_algaas(0.3))) - gap_algaas(300.0, 0.3)) > 0.1
    assert abs(varshni_eg_ev(300.0, ("GaAs", "AlAs", 0.3, -0.127 + 1.311 * 0.3)) - gap_algaas(300.0, 0.3)) > 1e-5


def test_one_gap_kernel_for_the_whole_package():
    """Plan 4.0.4: ONE Varshni function and ONE table. optics.soa re-exports the materials-level objects
    (the same function and dict objects, so its public names keep working), and no second Varshni law is
    defined anywhere in the package. SOA results are bit-identical: MEASURED -- 178 SOA values (every gap
    law x 8 temperatures, slopes, peak drifts, qd_params_at_temperature, the BulkGainParams band edge, gain
    peak and a 64-point gain spectrum) hash to the same SHA-256 on cc0bdf0 and on this tree (audit note)."""
    import ast
    import os
    import dynameta
    from dynameta.materials import band_temperature as BT
    from dynameta.optics.soa import temperature as soa_t
    import dynameta.optics.soa as soa
    for nm in ("varshni_eg_ev", "d_eg_dT_ev_per_K", "VARSHNI_PARAMS", "ALLOY_GAP_PARAMS"):
        assert getattr(soa_t, nm) is getattr(BT, nm), nm
        assert getattr(soa, nm) is getattr(BT, nm), nm
    root = os.path.dirname(dynameta.__file__)
    defs = []
    for dp, _dn, fns in os.walk(root):
        for fn in fns:
            if fn.endswith(".py"):
                with open(os.path.join(dp, fn), encoding="utf-8") as fh:
                    tree = ast.parse(fh.read())
                defs += [(fn, n.name) for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                         and "varshni" in n.name.lower()]
    assert sorted(defs) == [("band_temperature.py", "_varshni_abc"), ("band_temperature.py", "varshni_eg_ev")]


# =====================================================================================================
# H2-b (unit half) and H2-f: band data at T
# =====================================================================================================
def test_band_data_is_the_input_bit_for_bit_at_the_reference_T():
    """GATE H2-b (unit half; decision 32). With a band-temperature model, at T = T_ref the returned E_g, chi
    and n_i are the INPUT floats bit for bit -- anchored gap, every affinity fraction, derived n_i on and off,
    for T_ref = 300 K (unstated), 250 K and 77 K, binary and ternary laws; the absolute anchor returns the
    law's gap there and the input chi / n_i bit for bit. Without a model the inputs come back as the SAME
    objects at any T (the v0.14 path). Adversarial: the absolute form at T_ref = 300 K is NOT the input gap
    (it is 1.6 meV off for the validation GaAs 1.424 eV) -- the reason for anchoring."""
    from dynameta.materials.band_temperature import BandTemperature, algaas_gap_law, band_data_at_T
    for t_ref in (None, 250.0, 77.0):
        for law in ("GaAs", algaas_gap_law(0.3)):
            for gamma in (0.0, 0.5, 1.0):
                for derive in (False, True):
                    tr = _tm(band_gap_eV=1.424, chi_eV=4.07, n_i_m3=2.1e12, chi_reference_T_K=t_ref,
                             band_temperature=BandTemperature(law, gamma, derive_n_i=derive))
                    T = T_REF if t_ref is None else t_ref
                    bd = band_data_at_T(tr, T)
                    assert (_hex(bd.E_g_eV), _hex(bd.chi_eV), _hex(bd.n_i_m3)) == \
                           (_hex(1.424), _hex(4.07), _hex(2.1e12)), (t_ref, law, gamma, derive)
    from dynameta.materials.band_temperature import varshni_eg_ev
    tr = _tm(chi_eV=4.07, n_i_m3=2.1e12, band_temperature=BandTemperature("GaAs", 1.0, anchor="absolute",
                                                                           derive_n_i=True))
    bd = band_data_at_T(tr, T_REF)
    assert (_hex(bd.E_g_eV), _hex(bd.chi_eV), _hex(bd.n_i_m3)) == (_hex(varshni_eg_ev(T_REF, "GaAs")),
                                                                  _hex(4.07), _hex(2.1e12))
    assert abs(bd.E_g_eV - 1.424) > 1e-3                              # adversarial: absolute != the input
    plain = _tm(band_gap_eV=1.424, chi_eV=4.07, n_i_m3=2.1e12)
    for T in (77.0, 300.0, 400.0):
        bd = band_data_at_T(plain, T)
        assert bd.E_g_eV is plain.band_gap_eV and bd.chi_eV is plain.chi_eV and bd.n_i_m3 is plain.n_i_m3


def _nc_oracle(m_rel, T):
    """Effective DOS 2 (2 pi m k T / h^2)^(3/2) with the EXACT SI h (oracle side; the library's HBAR is the
    CODATA-rounded literal -- irrelevant here: n_i(T)/n_i(T_ref) depends only on the T^(3/2) law)."""
    h = 6.62607015e-34
    return 2.0 * (2.0 * math.pi * m_rel * M_E * KB * T / h ** 2) ** 1.5


@pytest.mark.parametrize("anchor", ["reference", "absolute"])
def test_band_data_at_T_against_independent_formulas(anchor):
    """GATE H2-a/H2-f. GaAs data quoted at T_ref = 300 K (anchored) or at 0 K (absolute; the law's gap),
    evaluated at 77-400 K, against formulas written here: E_g(T) = E_g,ref + V(T) - V(T_ref) (or V(T)),
    chi(T) = chi_ref - gamma [V(T) - V(T_ref)] for gamma = 1 and 1/2, and with derive_n_i the derived
    n_i(T) = sqrt(N_c(T) N_v(T)) exp(-E_g(T)/2kT) from the masses, n_i_ref = the same expression at T_ref.
    Tolerances: 1e-12 eV for E_g and chi (as H2-a); n_i relative 64 eps max|E_g/2kT| (the exp of an
    O(50) argument carries its absolute rounding as a relative error; MEASURED <= 6.9e-15 rel); the material's
    own statistical gap kT ln(N_c N_v / n_i(T)^2) equals E_g(T) to 1e-12 eV -- the band-gap check then
    passes by construction. Adversarial: swapping gamma (1 <-> 1/2) moves chi by >= 10 meV at 77 K."""
    from dynameta.materials.band_temperature import BandTemperature, band_data_at_T
    me, mh = 0.067, 0.50
    t_ref = 300.0 if anchor == "reference" else 0.0
    eg_ref = 1.424 if anchor == "reference" else GAAS_T1[0]
    kt = lambda T: KB * T / Q_E                                                     # noqa: E731
    ni_ref = (math.sqrt(_nc_oracle(me, 300.0) * _nc_oracle(mh, 300.0))
              * math.exp(-(eg_ref if anchor == "reference" else varshni(300.0, *GAAS_T1)) / (2 * kt(300.0))))
    for gamma in (1.0, 0.5):
        tr = _tm(band_gap_eV=(1.424 if anchor == "reference" else None), chi_eV=4.07,
                 n_i_m3=ni_ref, chi_reference_T_K=(None if anchor == "reference" else t_ref),
                 band_temperature=BandTemperature("GaAs", gamma, anchor=anchor, derive_n_i=False))
        for T in (77.0, 200.0, 250.0, 350.0, 400.0):
            d_v = varshni(T, *GAAS_T1) - varshni(t_ref, *GAAS_T1)
            bd = band_data_at_T(tr, T)
            eg = (eg_ref + d_v) if anchor == "reference" else varshni(T, *GAAS_T1)
            assert abs(bd.E_g_eV - eg) < TOL_EG
            assert abs(bd.chi_eV - (4.07 - gamma * d_v)) < TOL_EG
            assert bd.n_i_m3 == ni_ref                                   # derive_n_i=False: as given
            other = 4.07 - (1.5 - gamma) * d_v                          # adversarial: the other convention
            if T == 77.0:
                assert abs(other - bd.chi_eV) > 1e-2 or anchor == "absolute"
    if anchor == "absolute":                                            # n_i cannot be anchored at 0 K
        return
    tr = _tm(band_gap_eV=1.424, chi_eV=4.07, n_i_m3=ni_ref, dos_mass_p_kg=mh * M_E,
             band_temperature=BandTemperature("GaAs", 1.0, derive_n_i=True))
    for T in (77.0, 200.0, 250.0, 300.0, 350.0, 400.0):
        eg = 1.424 + varshni(T, *GAAS_T1) - varshni(300.0, *GAAS_T1)
        x = eg / (2 * kt(T))
        ni = math.sqrt(_nc_oracle(me, T) * _nc_oracle(mh, T)) * math.exp(-x)
        bd = band_data_at_T(tr, T)
        assert bd.n_i_m3 == pytest.approx(ni, rel=64 * EPS * max(x, 1.424 / (2 * kt(300.0)))), T
        eg_stat = kt(T) * math.log(_nc_oracle(me, T) * _nc_oracle(mh, T) / bd.n_i_m3 ** 2)
        assert abs(eg_stat - eg) < 1e-12, T


def test_anchoring_keeps_a_paired_300K_gap_and_n_i():
    """Decision 32 / plan 5.3, the reason for the anchored form. P3's absolute Si Varshni triple (1.17 eV,
    4.73e-4 eV/K, 636 K; primary attribution UNVERIFIED -- used here only as the plan's arithmetic) gives
    1.1245 eV at 300 K, 5.2 meV below the 1.12973 eV a paired Si bundle implies (kT ln(N_c N_v/n_i^2) with
    n_i = 9.65e9 cm^-3); the anchored model returns the bundle's own gap and n_i bit for bit at 300 K and
    moves them only by the law's change elsewhere."""
    from dynameta.materials.band_temperature import BandTemperature, band_data_at_T, varshni_eg_ev
    si = (1.17, 4.73e-4, 636.0)
    eg_abs = varshni_eg_ev(300.0, si)
    assert abs(eg_abs - 1.1245) < 5e-5 and abs(1.12973 - eg_abs - 5.2e-3) < 1e-4     # plan 5.3 table
    tr = _tm(band_gap_eV=1.12973, chi_eV=4.05, n_i_m3=9.65e15,
             band_temperature=BandTemperature(si, 0.5, derive_n_i=True))
    bd = band_data_at_T(tr, 300.0)
    assert (_hex(bd.E_g_eV), _hex(bd.n_i_m3)) == (_hex(1.12973), _hex(9.65e15))
    bd = band_data_at_T(tr, 350.0)
    assert abs(bd.E_g_eV - (1.12973 + varshni(350.0, *si) - varshni(300.0, *si))) < TOL_EG


# =====================================================================================================
# API: required affinity fraction, validation, defaults off, the BTBT accessor
# =====================================================================================================
def test_band_temperature_inputs_are_validated():
    """Decision 46: the affinity fraction is REQUIRED (no default) and in [0, 1]; the anchor's inputs must
    exist and nothing given may be silently unused (absolute + band_gap_eV, derive_n_i without n_i or at
    T_ref = 0 K); chi_reference_T_K = 0 K (data quoted at 0 K) only with a model."""
    from dynameta.materials.band_temperature import BandTemperature
    with pytest.raises(TypeError):
        BandTemperature("GaAs")                                          # no default gamma_chi
    for bad in (-0.1, 1.5, float("nan"), True, "1"):
        with pytest.raises(ValueError, match="affinity_fraction"):
            BandTemperature("GaAs", bad)
    with pytest.raises(ValueError, match="anchor"):
        BandTemperature("GaAs", 1.0, anchor="0K")
    with pytest.raises(ValueError, match="varshni"):
        BandTemperature("GaSb", 1.0)                                    # no such row
    with pytest.raises(TypeError, match="derive_n_i"):
        BandTemperature("GaAs", 1.0, derive_n_i=1)
    assert BandTemperature(["GaAs", "AlAs", 0.3, 0.266], 1).gap_law == ("GaAs", "AlAs", 0.3, 0.266)
    bt = BandTemperature("GaAs", 0.5)
    with pytest.raises(ValueError, match="band_gap_eV"):
        _tm(chi_eV=4.07, band_temperature=bt)                           # anchored: needs the anchor gap
    with pytest.raises(ValueError, match="leave TransportModel.band_gap_eV None"):
        _tm(band_gap_eV=1.424, band_temperature=BandTemperature("GaAs", 0.5, anchor="absolute"))
    with pytest.raises(ValueError, match="n_i_m3"):
        _tm(band_gap_eV=1.424, band_temperature=BandTemperature("GaAs", 0.5, derive_n_i=True))
    with pytest.raises(ValueError, match="0 K"):
        _tm(n_i_m3=1e12, chi_reference_T_K=0.0,
            band_temperature=BandTemperature("GaAs", 0.5, anchor="absolute", derive_n_i=True))
    with pytest.raises(ValueError, match="chi_reference_T_K"):
        _tm(band_gap_eV=1.424, chi_reference_T_K=0.0)                   # 0 K tag without a model
    _tm(chi_eV=4.07, chi_reference_T_K=0.0, band_temperature=BandTemperature("GaAs", 1.0, anchor="absolute"))
    with pytest.raises(TypeError, match="BandTemperature"):
        _tm(band_gap_eV=1.424, band_temperature=("GaAs", 1.0))


def test_defaults_off():
    """The field defaults to None and a model-free material is untouched (the v0.14 behaviour); the A/B
    builder identity and the call-log / cross-version guards are in tests/test_devsim_band_temperature.py
    and the existing digest gates (audit note)."""
    from dynameta.materials import TransportModel
    assert {f.name: f.default for f in dataclasses.fields(TransportModel)}["band_temperature"] is None
    # appended after carrier_statistics: positional order kept (v0.15 4.2.4 appends hooge_alpha_n/_p after it)
    names = [f.name for f in dataclasses.fields(TransportModel)]
    assert names[names.index("carrier_statistics") + 1] == "band_temperature"
    assert names[names.index("band_temperature") + 1:] == ["hooge_alpha_n", "hooge_alpha_p"]
    assert _tm().band_temperature is None


def test_gap_ratio_accessor_for_the_btbt_scaling():
    """Plan 5.2 item 2: BTBT reads E_g(T)/E_g(300) from the region's band-temperature model when present
    (the scaling itself is item 4.1.2). None without a model; else the ratio of the anchored gaps,
    independently: (E_ref + V(T) - V(T_ref)) / (E_ref + V(300) - V(T_ref))."""
    from dynameta.materials.band_temperature import BandTemperature, band_gap_ratio
    assert band_gap_ratio(_tm(band_gap_eV=1.424), 250.0) is None
    tr = _tm(band_gap_eV=1.43, chi_reference_T_K=77.0, band_temperature=BandTemperature("GaAs", 1.0))
    for T in (200.0, 300.0, 400.0):
        ref = (1.43 + varshni(T, *GAAS_T1) - varshni(77.0, *GAAS_T1)) / \
              (1.43 + varshni(300.0, *GAAS_T1) - varshni(77.0, *GAAS_T1))
        assert band_gap_ratio(tr, T) == pytest.approx(ref, rel=1e-14)
    assert band_gap_ratio(tr, 300.0) == 1.0


# =====================================================================================================
# integration with the v0.14 chi(T) guard
# =====================================================================================================
def test_a_band_model_satisfies_the_affinity_temperature_guard():
    """A material with a band-temperature model has chi evaluated at T, so the v0.14 guard counts it as
    consistent under every on_chi_temperature setting; a material without one is checked exactly as before
    (an untagged one in the same stack still refuses, named alone); without models the tags are the v0.14
    tags (chi_reference_T_K) unchanged."""
    from dynameta.carriers.physics_heterojunction import (AffinityTemperatureError, affinity_temperature_tags,
                                                          check_affinity_temperature)
    from dynameta.materials.band_temperature import BandTemperature
    bt = BandTemperature("GaAs", 1.0)
    modelled = _tm(band_gap_eV=1.424, chi_eV=4.07, band_temperature=bt)
    untagged = _tm(band_gap_eV=1.798, chi_eV=3.83)
    tagged = _tm(band_gap_eV=1.798, chi_eV=3.83, chi_reference_T_K=250.0)
    tags = affinity_temperature_tags({"GaAs": modelled, "AlGaAs": tagged}, 250.0)
    assert tags == {"GaAs": 250.0, "AlGaAs": 250.0}
    check_affinity_temperature(tags, 250.0)                              # "refuse" default: consistent
    with pytest.raises(AffinityTemperatureError) as ei:
        check_affinity_temperature(affinity_temperature_tags({"GaAs": modelled, "AlGaAs": untagged}, 250.0), 250.0)
    msg = str(ei.value)
    assert "AlGaAs (chi_eV quoted at 300 K" in msg
    assert "GaAs (chi_eV" not in msg.replace("AlGaAs (chi_eV", "")       # the modelled one is not named
    assert "band_temperature" in msg                                    # the physics remedy is named
    plain = {"A": untagged, "B": tagged}
    assert affinity_temperature_tags(plain, 250.0) == {m: t.chi_reference_T_K for m, t in plain.items()}
