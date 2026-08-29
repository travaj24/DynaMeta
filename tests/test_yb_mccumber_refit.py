"""Gates for ytterbium(mccumber_refit=True) -- the 2026-08-23 signal-band sigma_a refit
(docs/audit/2026-08-23-yb-signal-band-mccumber-refit.md). Mirrors the erbium(cband_refit)
gate style: every test is falsifiable, and the legacy variant is exercised alongside the
refit so each gate PROVES it discriminates (the legacy must FAIL the physics condition the
refit satisfies).

Since 2026-08-28 the DEFAULT ytterbium() is the measured Melkumov table
(tests/test_measured_spectra_2026_08_28.py); the gates here pin the PARAMETRIC
Gaussian-sum variants, so they request them explicitly via mccumber_refit=True/False
(an explicit mccumber_refit always selects the parametric ion)."""

import math

import numpy as np

from dynameta.constants import C_LIGHT, H_PLANCK
from dynameta.optics.fiber_amp import (
    CrossSectionModel, FiberSpec, at_temperature, overlap_gamma, ytterbium,
)

KB = 1.380649e-23
SIGNAL_NM = np.array([1010.0, 1030.0, 1060.0, 1080.0, 1100.0])
# pump grid ends just BELOW the 1000 nm crossover: 1000.0 * 1e-9 rounds one ulp above the
# 1.000e-6 crossover literal in IEEE754 and would (deterministically) take the derived
# branch; the seam gate covers the boundary itself.
PUMP_NM = np.linspace(850.0, 999.5, 300)


def _mcc_ratio(lam_m, zero_line_m, T_K=300.0):
    """Detailed-balance sigma_a/sigma_e at 300 K against the ion's zero line."""
    nu = C_LIGHT / np.asarray(lam_m, float)
    eps = H_PLANCK * C_LIGHT / zero_line_m
    return np.exp((H_PLANCK * nu - eps) / (KB * T_K))


def test_refit_signal_band_obeys_own_mccumber():
    """By construction: sigma_a = sigma_e * exp((h nu - eps)/kT) in the signal band, to
    machine precision.  Trips if the hybrid's derivation, eps, or crossover drifts."""
    for host in ("aluminosilicate", "phosphosilicate"):
        ion = ytterbium(host, mccumber_refit=True)
        lam = SIGNAL_NM * 1e-9
        want = ion.sigma_e.sigma(lam) * _mcc_ratio(lam, ion.zero_line_m)
        got = ion.sigma_a.sigma(lam)
        assert np.allclose(got, want, rtol=1e-12), host


def test_legacy_violates_mccumber_proving_discrimination():
    """The legacy tail violates the same relation by >3x at 1030 nm and >7x at 1060 nm --
    the bug this refit removes.  If THIS gate ever fails, the legacy escape hatch no longer
    reproduces the defect and the bit-exactness contract is broken."""
    ion = ytterbium(mccumber_refit=False)
    for lam_nm, floor in ((1030.0, 3.0), (1060.0, 7.0)):
        lam = lam_nm * 1e-9
        want = float(ion.sigma_e.sigma(lam)) * float(_mcc_ratio(lam, ion.zero_line_m))
        got = float(ion.sigma_a.sigma(lam))
        assert got / want > floor, (lam_nm, got / want)


def test_escape_hatch_is_bit_exact_legacy():
    """mccumber_refit=False reproduces the pre-2026-08-23 construction EXACTLY: the frozen
    legacy peak list, re-evaluated here, must match bit-for-bit on a dense grid."""
    pk_a, lam_a = 2.7e-24, 0.976e-6
    legacy = CrossSectionModel((
        (0.915e-6, 0.035e-6, 0.30 * pk_a),
        (lam_a, 0.008e-6, pk_a),
        (1.030e-6, 0.050e-6, 0.030 * pk_a),
    ))
    ion = ytterbium(mccumber_refit=False)
    lam = np.linspace(0.85e-6, 1.12e-6, 1001)
    assert np.array_equal(ion.sigma_a.sigma(lam), legacy.sigma(lam))
    assert ion.name == "Yb3+"


def test_pump_band_identical_between_variants():
    """The refit must not touch the pump band: sigma_a below the 1000 nm crossover is the
    tabulated spectrum in BOTH variants, difference exactly zero."""
    for host in ("aluminosilicate", "phosphosilicate"):
        new = ytterbium(host, mccumber_refit=True)
        old = ytterbium(host, mccumber_refit=False)
        lam = PUMP_NM * 1e-9
        assert np.array_equal(new.sigma_a.sigma(lam), old.sigma_a.sigma(lam)), host
        assert np.array_equal(new.sigma_e.sigma(lam), old.sigma_e.sigma(lam)), host


def test_transparency_inversions_hit_textbook_anchors():
    """The independent physical anchor that would have caught the bug: Yb transparency
    inversion sigma_a/(sigma_a+sigma_e) is ~0.05 at 1030 nm and ~0.01-0.02 at 1064 nm in
    every published Yb-silica dataset.  Refit must land in those windows; legacy must land
    OUTSIDE both (it gave 0.21 / 0.13)."""
    new = ytterbium(mccumber_refit=True)
    old = ytterbium(mccumber_refit=False)

    def transp(ion, lam):
        sa, se = float(ion.sigma_a.sigma(lam)), float(ion.sigma_e.sigma(lam))
        return sa / (sa + se)

    assert 0.03 < transp(new, 1.030e-6) < 0.12
    assert 0.005 < transp(new, 1.060e-6) < 0.05
    assert transp(old, 1.030e-6) > 0.15
    assert transp(old, 1.060e-6) > 0.10


def test_unpumped_signal_loss_matches_real_fiber_scale():
    """Fiber-level consequence: on a ~556 dB/m-at-976 core-pumped Yb fiber the unpumped
    1060 nm loss must be sub-1 dB/m (real-fiber spectra: ~0.1-0.2% of the 976 peak), where
    the legacy gave 5.8 dB/m."""
    fib = FiberSpec(core_radius_m=3.0e-6, na=0.12, n_t_m3=6.0e25, length_m=1.0)
    lam = 1.060e-6
    gam = float(overlap_gamma(fib, lam))
    to_dB = 10.0 / math.log(10.0)
    loss_new = gam * fib.n_t_m3 * float(
        ytterbium(mccumber_refit=True).sigma_a.sigma(lam)) * to_dB
    loss_old = gam * fib.n_t_m3 * float(
        ytterbium(mccumber_refit=False).sigma_a.sigma(lam)) * to_dB
    assert 0.3 < loss_new < 1.5, loss_new
    assert loss_old > 4.0, loss_old
    assert loss_old / loss_new > 5.0


def test_seam_at_crossover_is_benign():
    """The 1000 nm crossover sits in the pump/signal dead zone where the two branches nearly
    agree on the shipped model; the step across the seam must stay under 20% and both sides
    must be small against the pump band (< 2% of the 976 nm peak)."""
    ion = ytterbium(mccumber_refit=True)
    lo = float(ion.sigma_a.sigma(0.9995e-6))
    hi = float(ion.sigma_a.sigma(1.0005e-6))
    assert abs(hi - lo) / lo < 0.20, (lo, hi)
    assert max(lo, hi) < 0.02 * float(ion.sigma_a.sigma(0.976e-6))


def test_hybrid_sigma_is_finite_positive_and_composes():
    """The hybrid duck-type must behave like a CrossSectionModel everywhere the package
    samples it: finite, non-negative, scalar-in scalar-out, and at_temperature must compose
    (it rebuilds the ion around the existing sigma_a)."""
    ion = ytterbium(mccumber_refit=True)
    lam = np.linspace(0.85e-6, 1.12e-6, 2001)
    s = ion.sigma_a.sigma(lam)
    assert np.all(np.isfinite(s)) and np.all(s >= 0.0)
    assert isinstance(ion.sigma_a.sigma(1.06e-6), float)
    hot = at_temperature(ion, 320.0)
    assert np.isfinite(float(hot.sigma_a.sigma(1.06e-6)))
    assert np.isfinite(float(hot.sigma_e.sigma(1.06e-6)))
