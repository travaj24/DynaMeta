"""Devsim-free gates of the v0.15 III-V recombination presets (consolidated plan item 4.2.2, P2 item 3:
R1, R2, R4, R5-pure; user decision 30): dynameta.materials.semiconductor_presets and the internal /
effective labelling of carriers.bipolar_options.RadiativeModel. DEVSIM gates (R3, R4 advisory, R5 bits):
tests/test_iii_v_presets_devsim.py. Audit note: docs/audit/2026-11-06-v0.15-extended-precision-design-iii-v.md.

ORACLES are the papers' numbers RE-TYPED here in the units they quote, with their locators:
  Niemeyer et al., AIP Adv. 9, 045034 (2019): abstract / Sec. VI B_rad = (3.3 +/- 0.6)e-10 cm^3/s
    (p < 1e18 cm^-3), Eq. (1) B_eff = (1 - f) B_rad with f = 0.76 +/- 0.04, and the DIRECTLY MEASURED
    B_eff = (0.78 +/- 0.07)e-10 cm^3/s of the 2 um DH (abstract, Sec. VII) -- read in the primary.
  Strauss, Ruhle & Kohler, APL 62, 55 (1993): C_eff = (7 +/- 4)e-30 cm^6/s (abstract record only).
  Semyonov et al., arXiv:1003.6095, Eq. (6): B_R = 1.2e-10 (1 - 0.092 n/1e18) cm^3/s -- read in the primary.
FAILS ON OLD CODE (40b1ede): the presets and RadiativeModel(kind=, photon_recycling_factor=) do not exist --
the import below raises at collection (ImportError), which fails the file (no importorskip on purpose).
"""
import dataclasses
import hashlib
import math

import pytest

from dynameta.carriers import bipolar_options as BO
from dynameta.materials import semiconductor_presets as SP
from dynameta.materials.semiconductor_presets import (AUGER_GAAS_STRAUSS_1993,
                                                      RADIATIVE_GAAS_NIEMEYER_2019_INTERNAL,
                                                      RADIATIVE_INP_SEMYONOV_2010_INTERNAL)

CM3S, CM6S = 1e-6, 1e-12
# --- re-typed literature values (CGS, as quoted) --------------------------------------------------
NIEMEYER_B_INT, NIEMEYER_B_INT_SD = 3.3e-10, 0.6e-10    # abstract; Sec. VI, p < 1e18 cm^-3
NIEMEYER_F, NIEMEYER_F_SD = 0.76, 0.04                  # abstract: photon recycling factor, 2 um DH
NIEMEYER_B_EFF, NIEMEYER_B_EFF_SD = 0.78e-10, 0.07e-10  # abstract / Sec. VII: measured, 2 um DH
STRAUSS_C_EFF = 7e-30                                   # abstract: (7 +/- 4)e-30 cm^6/s
SEMYONOV_B0 = 1.2e-10                                   # Eq. (6), n -> 0 intercept


def test_r1_values_labels_and_locators_are_the_papers():
    """GATE R1. Every shipped constant equals the re-typed value exactly (SI), each B is labelled
    kind='internal' with no factor, and the verification labels say which were read in the primary
    (Niemeyer, Semyonov) and which only via a secondary record (Strauss). Adversarial: a frozen sha256 of
    the three reprs catches a silent value / label edit under an unchanged name."""
    g, c, i = RADIATIVE_GAAS_NIEMEYER_2019_INTERNAL, AUGER_GAAS_STRAUSS_1993, RADIATIVE_INP_SEMYONOV_2010_INTERNAL
    assert g.B_m3s == NIEMEYER_B_INT * CM3S and i.B_m3s == SEMYONOV_B0 * CM3S
    assert c.C_n_m6s == c.C_p_m6s == 3.5e-30 * CM6S and not c.coulomb_enhanced
    assert (g.kind, i.kind) == ("internal", "internal")
    assert g.photon_recycling_factor is None and i.photon_recycling_factor is None
    assert (g.verification, i.verification, c.verification) == ("primary", "primary", "secondary")
    assert "INTERNAL" in g.source and "INTERNAL" in i.source and "split EQUALLY" in c.source
    assert "Eq. (1)" in g.locator and "Eq. (6)" in i.locator and "abstract only" in c.locator
    assert SP.III_V_RECOMBINATION_PRESETS == (g, c, i)
    frozen = repr(SP.III_V_RECOMBINATION_PRESETS)
    assert hashlib.sha256(frozen.encode()).hexdigest() == \
        "4807bf022708742ea0d2cadfc8edad3e8dcec5bcf1cfc2ded208a3df903c5710"


def test_r1_auger_equal_split_sums_to_the_ambipolar_coefficient():
    """GATE R1 (Auger). The equal split is the AMBIPOLAR reading: at n = p the kernel's U_A = (C_n n + C_p p)
    (n p - n_ie^2) -> (C_n + C_p) n^3, so C_n + C_p must equal Strauss's C_eff = 7e-30 cm^6/s exactly.
    Adversarial: the per-carrier reading of a secondary user (7e-30 EACH) would give 2x -- refused here."""
    c = AUGER_GAAS_STRAUSS_1993
    assert (c.C_n_m6s + c.C_p_m6s) / CM6S == pytest.approx(STRAUSS_C_EFF, rel=1e-15, abs=0.0)
    per_carrier_sum = 2 * STRAUSS_C_EFF * CM6S                      # C_n = C_p = 7e-30 each
    assert per_carrier_sum / (c.C_n_m6s + c.C_p_m6s) == pytest.approx(2.0, rel=1e-15, abs=0.0)


def test_r2_internal_times_recycling_matches_the_measured_effective_B():
    """GATE R2 (decision 30). B_int (1 - P_r) through the code path (preset.with_photon_recycling(0.76).
    net_B_m3s) = 3.3e-10 x 0.24 = 0.792e-10 cm^3/s must match Niemeyer's INDEPENDENTLY MEASURED B_eff of the
    same 2 um DH, 0.78 +/- 0.07e-10, within his stated uncertainty (budget: the measurement's own one
    sigma, 0.07e-10; the propagated sigma of B_int (1 - f), sqrt((0.6 x 0.24)^2 + (3.3 x 0.04)^2) =
    0.195e-10, is wider still). Measured: 0.792 - 0.78 = +0.012e-10 = 0.17 sigma.
    ADVERSARIAL (swapped labels must fail by ~4.2x):
      * 3.3e-10 mislabelled EFFECTIVE (used as the net loss): 3.3 / 0.78 = 4.23x, 36 sigma off;
      * 0.78e-10 mislabelled INTERNAL and recycled again: 0.78 x 0.24 = 0.187e-10, 4.17x low, 8.5 sigma."""
    net = RADIATIVE_GAAS_NIEMEYER_2019_INTERNAL.with_photon_recycling(NIEMEYER_F).net_B_m3s / CM3S
    assert net == pytest.approx(NIEMEYER_B_INT * (1 - NIEMEYER_F), rel=1e-15, abs=0.0)        # 0.792e-10, ONE factor
    assert net == pytest.approx(0.792e-10, rel=1e-12, abs=0.0)
    assert abs(net - NIEMEYER_B_EFF) <= NIEMEYER_B_EFF_SD                             # the gate
    sigma_prop = math.hypot(NIEMEYER_B_INT_SD * (1 - NIEMEYER_F), NIEMEYER_B_INT * NIEMEYER_F_SD)
    assert abs(net - NIEMEYER_B_EFF) <= sigma_prop
    # swapped label 1: the internal B used as the effective one (no recycling)
    as_effective = dataclasses.replace(RADIATIVE_GAAS_NIEMEYER_2019_INTERNAL, kind="effective").net_B_m3s / CM3S
    assert as_effective / NIEMEYER_B_EFF == pytest.approx(4.23, abs=0.01)
    assert abs(as_effective - NIEMEYER_B_EFF) > 30 * NIEMEYER_B_EFF_SD
    # swapped label 2: the measured effective B labelled internal and recycled again
    double = BO.RadiativeModel(B_m3s=NIEMEYER_B_EFF * CM3S, kind="internal",
                               photon_recycling_factor=NIEMEYER_F).net_B_m3s / CM3S
    assert NIEMEYER_B_EFF / double == pytest.approx(4.17, abs=0.01)
    assert abs(double - NIEMEYER_B_EFF) > 8 * NIEMEYER_B_EFF_SD


def test_r4_refusals():
    """GATE R4 (construction half; the advisory is DEVSIM-side). Effective + a factor raises (recycling
    twice); an UNLABELLED B + a factor raises (which B is it?); P_r outside [0, 1) raises; a bad kind or
    verification raises. with_photon_recycling refuses on an effective B."""
    with pytest.raises(ValueError, match="TWICE"):
        BO.RadiativeModel(B_m3s=1e-16, kind="effective", photon_recycling_factor=0.5)
    with pytest.raises(ValueError, match="needs kind='internal'"):
        BO.RadiativeModel(B_m3s=1e-16, photon_recycling_factor=0.5)
    for bad in (1.0, -0.1, float("nan")):
        with pytest.raises(ValueError, match=r"\[0, 1\)"):
            RADIATIVE_GAAS_NIEMEYER_2019_INTERNAL.with_photon_recycling(bad)
    with pytest.raises(ValueError, match="kind must be"):
        BO.RadiativeModel(B_m3s=1e-16, kind="intrinsic")
    with pytest.raises(ValueError, match="verification"):
        BO.RadiativeModel(B_m3s=1e-16, verification="checked")
    with pytest.raises(ValueError, match="verification"):
        BO.AugerModel(C_n_m6s=1e-42, C_p_m6s=1e-42, verification="checked")
    eff = BO.RadiativeModel(B_m3s=0.78e-16, kind="effective")
    with pytest.raises(ValueError, match="TWICE"):
        eff.with_photon_recycling(0.76)
    assert RADIATIVE_GAAS_NIEMEYER_2019_INTERNAL.with_photon_recycling(0.0).net_B_m3s == \
        RADIATIVE_GAAS_NIEMEYER_2019_INTERNAL.B_m3s                       # P_r = 0 is allowed


def test_r5_legacy_models_are_unchanged():
    """GATE R5 (pure half). A legacy RadiativeModel(B_m3s=...) / AugerModel keeps its fields' meaning:
    kind None, verification 'unverified', and net_B_m3s IS the given float object (so the B_rad
    parameter write is bit-identical; the DEVSIM half reads the parameter back). The Si presets keep their
    values."""
    for b in (9.5e-21, 1e-16, 3.3e-16, 0.1 + 0.2):
        m = BO.RadiativeModel(B_m3s=b)
        assert m.kind is None and m.photon_recycling_factor is None and m.verification == "unverified"
        assert m.net_B_m3s is m.B_m3s
    assert BO.RADIATIVE_SI_TRUPKE_2003.B_m3s == 9.5e-15 * CM3S
    assert BO.RADIATIVE_SI_TRUPKE_2003.net_B_m3s is BO.RADIATIVE_SI_TRUPKE_2003.B_m3s
    assert BO.AUGER_SI_DZIEWIOR_SCHMID_1977.verification == "unverified"


def test_deferred_materials_are_not_shipped():
    """Decision 30: InGaAs, GaN and Ge are deferred and Varshni's calculated 7.2e-10 is never shipped as a
    measured B -- no preset of those names / that value exists."""
    names = [n for n in dir(SP) if n.isupper()]
    assert not [n for n in names if any(k in n for k in ("INGAAS", "GAN", "_GE_", "GE_", "VARSHNI"))]
    for n in names:
        v = getattr(SP, n)
        if isinstance(v, BO.RadiativeModel):
            assert v.B_m3s != 7.2e-10 * CM3S
