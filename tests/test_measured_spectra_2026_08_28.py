"""Gates for the 2026-08-28 measured-spectra corrections (audit
docs/audit/2026-08-28-measured-spectra-melkumov-er-pump-band.md):

  * calibration.ytterbium_melkumov(): Yb3+ built from the measured Melkumov 2004
    aluminosilicate table instead of the Gaussian-sum fit -- THE DEFAULT ytterbium()
    ion since 2026-08-28 (escape hatches: melkumov_tables=False, or an EXPLICIT
    mccumber_refit=True/False, select the parametric variants bit-exactly);
  * spectroscopy.erbium(pump_band_refit=True): corrected 4I11/2 pump band
    (978 nm peak, ~21 nm FWHM, 2.55e-25 peak) -- THE DEFAULT since 2026-08-28
    (pump_band_refit=False reproduces the legacy band bit-exactly).

The Fuchtbauer-Ladenburg gate is deliberately Yb-ONLY: Yb's 2F5/2 is essentially fully
radiative, so 1/tau_rad = 8 pi n^2 c INT sigma_e / lambda^4 dlambda is a hard constraint.
Barnes et al. (IEEE JQE 27, 1004 (1991)) showed FL extraction is UNRELIABLE for Er (40%
method disagreement, sigma_e/sigma_a ordering flips at the 1530 peak), so no Er FL gate.
Discrimination: the legacy Gaussian-sum Yb spectrum FAILS the FL gate (that is the defect
it documents); the Melkumov table passes.
"""
import numpy as np

from dynameta.core.numerics import trapz  # floor-safe (np.trapezoid needs numpy>=2.0)
from dynameta.optics.fiber_amp import erbium, ytterbium
from dynameta.optics.fiber_amp.calibration import ytterbium_melkumov

C_LIGHT = 299792458.0
N_SILICA = 1.45


def _fl_tau_rad_s(ion, lam_lo=850e-9, lam_hi=1180e-9, n=1.45):
    lam = np.linspace(lam_lo, lam_hi, 4001)
    se = np.array([ion.sigma_e.sigma(x) for x in lam])
    inv_tau = 8.0 * np.pi * n * n * C_LIGHT * trapz(se / lam ** 4, lam)
    return 1.0 / inv_tau


# ---------------------------------------------------------------------------- Yb: FL gate
def test_yb_melkumov_fl_consistent():
    """The measured spectrum's radiative lifetime must agree with the asserted tau_s
    (Yb is ~fully radiative)."""
    ion = ytterbium_melkumov()
    ratio = _fl_tau_rad_s(ion) / ion.tau_s
    assert 0.70 < ratio < 1.35, ratio


def test_yb_legacy_gaussian_fails_fl():
    """DISCRIMINATION: the Gaussian-sum fit carries ~1.6x too little oscillator strength;
    reverting to it must trip this gate."""
    legacy = ytterbium(melkumov_tables=False)
    ratio = _fl_tau_rad_s(legacy) / legacy.tau_s
    assert ratio > 1.45, ratio


# --------------------------------------------------------------------- the 2026-08-28 defaults
def test_yb_default_is_melkumov():
    """Bare ytterbium() must BE the measured-table ion (and therefore FL-consistent)."""
    ion = ytterbium()
    assert ion.name == "Yb3+(melkumov)", ion.name
    assert abs(ion.sigma_e.sigma(1060e-9) / 3.1e-25 - 1) < 0.02
    ratio = _fl_tau_rad_s(ion) / ion.tau_s
    assert 0.70 < ratio < 1.35, ratio


def test_yb_explicit_mccumber_refit_still_parametric():
    """Explicitly requesting a parametric variant must keep meaning what it said before the
    default flip: mccumber_refit=True selects the Gaussian-sum ion with the derived signal
    band (sigma_e(1060) ~ 1.95e-25, the fit's low value), bit-identical to
    melkumov_tables=False."""
    a = ytterbium(mccumber_refit=True)
    b = ytterbium(melkumov_tables=False)
    assert a.name == b.name == "Yb3+(mccumber_refit)"
    for lam in (915e-9, 976e-9, 1030e-9, 1060e-9, 1090e-9):
        assert a.sigma_a.sigma(lam) == b.sigma_a.sigma(lam)
        assert a.sigma_e.sigma(lam) == b.sigma_e.sigma(lam)
    assert abs(a.sigma_e.sigma(1060e-9) / 1.95e-25 - 1) < 0.03


def test_yb_phospho_stays_parametric():
    """No measured table for the phosphosilicate host -- it must keep the parametric model."""
    ion = ytterbium("phosphosilicate")
    assert ion.name == "Yb3+(mccumber_refit)", ion.name
    assert abs(ion.tau_s - 1.45e-3) < 1e-6


def test_er_default_is_pump_band_refit():
    """Bare erbium() must carry the measured 4I11/2 band (sigma_a(976) ~ 2.4e-25)."""
    s976 = erbium().sigma_a.sigma(976e-9)
    assert 2.3e-25 < s976 < 2.6e-25, s976


def test_default_yb_fixture_recovers_physical_branch():
    """The Melkumov default's stronger ASE widens the relax=1.0 oscillation region: on the
    F-13/F-14 audit fixture (2 W, fully absorbed pump) plain Gauss-Seidel now falls onto the
    spurious near-unpumped branch. The SHIPPED relax='auto' ladder must walk to 0.5 and land
    on the physical branch: converged, >30 dB of gain, and a genuinely inverted front end
    (the spurious branch sits at nbar2 ~ 0.04 everywhere)."""
    import numpy as np
    from dynameta.optics.fiber_amp import AseBand, FiberAmplifier, FiberSpec, Pump, Signal
    fib = FiberSpec(3.0e-6, 0.12, 6.0e25, 4.0)
    amp = FiberAmplifier(ytterbium(), fib, [Pump(2.0, 0.976e-6)],
                         [Signal(0.243e-3, 1.060e-6)], AseBand(1.02e-6, 1.10e-6, 24))
    r = amp.solve(n_nodes=201, max_iter=400, relax="auto")
    assert r.meta["converged"] and r.meta["relax_attempts"][-1] == 0.5
    assert float(r.signal_gain_dB[0]) > 30.0
    assert float(np.max(r.nbar2_z)) > 0.4


# ------------------------------------------------------------------- Yb: anchors + McCumber
def test_yb_melkumov_anchor_values():
    ion = ytterbium_melkumov()
    assert abs(ion.sigma_e.sigma(1060e-9) / 3.1e-25 - 1) < 0.02
    assert abs(ion.sigma_a.sigma(1060e-9) / 5.7e-27 - 1) < 0.02
    assert abs(ion.sigma_a.sigma(976e-9) / 2.69e-24 - 1) < 0.02
    assert abs(ion.sigma_e.sigma(976e-9) / 2.97e-24 - 1) < 0.02
    assert abs(ion.sigma_e.sigma(1030e-9) / 6.3e-25 - 1) < 0.03   # interp of 1028/1032
    assert abs(ion.tau_s - 0.83e-3) < 1e-6


def test_yb_melkumov_mccumber_zero_line():
    """Back-solving eps from sigma_e/sigma_a across the signal band must give a stable
    zero line near 976 nm (the table is detailed-balance consistent)."""
    ion = ytterbium_melkumov()
    h, k, c, T = 6.62607015e-34, 1.380649e-23, C_LIGHT, 300.0
    zl = []
    for lam_nm in (1030, 1040, 1056, 1060, 1064, 1080):
        lam = lam_nm * 1e-9
        r = ion.sigma_e.sigma(lam) / ion.sigma_a.sigma(lam)
        eps = h * c / lam + k * T * np.log(r)
        zl.append(h * c / eps * 1e9)
    zl = np.array(zl)
    assert np.all(np.abs(zl - 975.8) < 4.0), zl


def test_yb_melkumov_unpumped_loss_realistic():
    """Unpumped 1060 nm loss on the SAT56-class reference geometry (a=3 um, NA 0.12,
    Gamma~0.75) at commercial-class doping 3e25 m^-3 must land near the measured
    0.5-1 dB/m for real Yb fiber."""
    ion = ytterbium_melkumov()
    loss_dB_m = 0.75 * 3e25 * ion.sigma_a.sigma(1060e-9) * 4.342944819
    assert 0.3 < loss_dB_m < 1.2, loss_dB_m


# ------------------------------------------------------------------------- Er: pump band
def test_er_pump_band_refit_values():
    ion = erbium(pump_band_refit=True)
    s976 = ion.sigma_a.sigma(976e-9)
    assert 2.3e-25 < s976 < 2.6e-25, s976
    lam = np.linspace(960e-9, 1000e-9, 2001)
    sa = np.array([ion.sigma_a.sigma(x) for x in lam])
    lam_pk = lam[int(np.argmax(sa))] * 1e9
    assert 976.5 < lam_pk < 979.5, lam_pk
    # emission stays ~0 across the pump band (true three-level pump)
    assert ion.sigma_e.sigma(976e-9) < 1e-27


def test_er_pump_band_legacy_reproduced_exactly():
    """pump_band_refit=False must reproduce the pre-2026-08-28 band bit-exactly: the legacy
    Gaussian (980 nm / 13 nm FWHM / 1.7e-25 peak) evaluated directly."""
    legacy = erbium(pump_band_refit=False)
    assert abs(legacy.sigma_a.sigma(976e-9) / 1.31e-25 - 1) < 0.01
    assert abs(legacy.sigma_a.sigma(980e-9) / 1.7e-25 - 1) < 0.01


def test_er_signal_band_untouched_by_pump_refit():
    a = erbium(pump_band_refit=True)
    b = erbium(pump_band_refit=False)
    for lam in (1480e-9, 1530e-9, 1550e-9, 1560e-9):
        assert a.sigma_a.sigma(lam) == b.sigma_a.sigma(lam)
        assert a.sigma_e.sigma(lam) == b.sigma_e.sigma(lam)
