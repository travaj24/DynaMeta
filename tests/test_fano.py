"""Gates for the Fano / Lorentzian lineshape fitting + quasi-BIC scaling tooling
(analysis.fano_fit / lorentzian_fit / quasi_bic_scaling), roadmap item 1.3.

Physics references:
  * Fano lineshape T = a_bg + b_bg (q + eps_r)^2 / (1 + eps_r^2), eps_r = 2(x-x0)/gamma
    -- Fano, Phys. Rev. 124:1866 (1961).
  * Symmetry-protected quasi-BIC radiative Q ~ delta^-2 in the asymmetry parameter delta
    -- Koshelev et al., PRL 121:193903 (2018).
  * Fabry-Perot etalon pole (symmetric slab) Q = -m pi / (2 ln|r12|) -- derived in-test.

Pure numpy/scipy; the only DynaMeta dependencies are analysis.py (under test) and the
tmm_reference oracle for the etalon cross-gate.
"""
import numpy as np
import pytest

from dynameta.analysis import (
    FanoFit,
    LorentzianFit,
    fano_fit,
    lorentzian_fit,
    quasi_bic_scaling,
    resonance_dip,
    resonance_shift,
)


# ---------------------------------------------------------------------------
# GATE 1 -- synthetic Fano round-trips across the whole q range with 0.5% noise.
#
# Tolerances (documented per regime): x0 rtol < 1e-4 (position is the best-determined
# parameter); gamma rtol < 2%; q within 2% rtol + 0.03 atol -- the additive floor covers
# the symmetric-dip q=0 regime (where a pure rtol is meaningless) and the strong-asymmetry
# |q|=10 regime (where the antiresonance sits far in one wing); both edges must fit WITHOUT
# divergence (finite q).
# ---------------------------------------------------------------------------
def test_fano_roundtrip_all_q_regimes():
    rng = np.random.default_rng(1234)
    x0_true = 1.935e14           # Hz (a ~1.55 um optical resonance)
    gamma_true = 2.0e12          # Hz  (Q ~ 97)
    a_bg, b_bg = 0.15, 0.60
    eps = np.linspace(-14.0, 14.0, 561)          # wide enough to capture eps=-q dip at |q|=10
    x = x0_true + 0.5 * gamma_true * eps

    for q in (-5.0, -1.0, -0.2, 0.0, 0.3, 2.0, 10.0):
        T = a_bg + b_bg * (q + eps) ** 2 / (1.0 + eps ** 2)
        T_noisy = T + 0.005 * T.max() * rng.standard_normal(T.size)   # 0.5% noise
        fit = fano_fit(x, T_noisy, x_kind="freq")

        assert isinstance(fit, FanoFit)
        assert np.isfinite(fit.q), "q diverged at q={} (must fit without divergence)".format(q)
        assert abs(fit.omega0 - x0_true) / x0_true < 1e-4, "x0 off at q={}".format(q)
        assert abs(fit.gamma_fwhm - gamma_true) / gamma_true < 2e-2, "gamma off at q={}".format(q)
        assert abs(fit.q - q) <= 0.02 * abs(q) + 0.03, "q={} recovered as {}".format(q, fit.q)
        # Q = |x0| / gamma consistency
        assert fit.Q == pytest.approx(abs(fit.omega0) / fit.gamma_fwhm, rel=1e-12)
        # background / amplitude recovered to a few percent
        assert fit.a_bg == pytest.approx(a_bg, abs=0.03)
        assert fit.b_bg == pytest.approx(b_bg, rel=0.05)


# ---------------------------------------------------------------------------
# GATE 2 -- on a PURE Lorentzian, lorentzian_fit and fano_fit return the SAME (x0, gamma)
# (they share the VARPRO core; the Fano dispersive column just collapses to ~0).
# ---------------------------------------------------------------------------
def test_lorentzian_limit_matches_fano():
    x0 = 1.0e14
    g = 1.5e12
    x = np.linspace(x0 - 12 * g, x0 + 12 * g, 801)
    T = 0.2 + 0.7 / (1.0 + (2.0 * (x - x0) / g) ** 2)          # pure symmetric Lorentzian peak

    lf = lorentzian_fit(x, T)
    ff = fano_fit(x, T, x_kind="freq")
    assert isinstance(lf, LorentzianFit)

    assert abs(lf.x0 - ff.omega0) / x0 < 1e-6
    assert abs(lf.fwhm - ff.gamma_fwhm) / g < 1e-6
    # lorentzian_fit itself recovers the truth
    assert lf.x0 == pytest.approx(x0, rel=1e-6)
    assert lf.fwhm == pytest.approx(g, rel=1e-6)
    assert lf.amplitude > 0.0                                  # a peak
    assert lf.baseline == pytest.approx(0.2, abs=1e-6)


# ---------------------------------------------------------------------------
# GATE 3 -- TMM cross-gate. Build a driven transmission spectrum of a symmetric n=3.5 slab
# in vacuum with tmm_reference around one Fabry-Perot resonance (m=4) and check the fitted
# Q against the CLOSED-FORM etalon pole Q.
#
# Derivation (symmetric lossless slab, index n, thickness L, vacuum both sides). The
# transmission denominator carries the round-trip factor 1 - r12^2 exp(2 i delta),
# delta = n omega L / c, r12 = (n-1)/(n+1). Its complex-omega pole (round-trip gain unity)
# is at delta = m pi - i ln|r12|, i.e. omega_t = (m pi c)/(n L) - i (c/(n L)) ln|r12|.
# With omega_0 = m pi c/(n L) and FWHM gamma = 2|Im omega_t| = -2 (c/(nL)) ln|r12| (ln<0),
#     Q = omega_0 / gamma = -m pi / (2 ln|r12|).
# A narrow-window fit of |T(omega)|^2 recovers this pole width (the near-peak lineshape is
# Lorentzian in omega with the pole HWHM), so fit-Q agrees with Q_pole to ~1%.
# ---------------------------------------------------------------------------
def _etalon_transmission(freqs_hz, n, L_m):
    from dynameta.optics.tmm_reference import stack_rta
    c = 299_792_458.0
    return np.array([stack_rta(1.0, [(n, L_m)], 1.0, c / f, pol="s")[1] for f in freqs_hz])


def test_tmm_etalon_pole_q_cross_gate():
    c = 299_792_458.0
    n = 3.5
    L = 1.0e-6
    m = 4
    f_m = m * c / (2.0 * n * L)                        # resonance frequency (Hz)
    r12 = (n - 1.0) / (n + 1.0)
    Q_pole = -m * np.pi / (2.0 * np.log(abs(r12)))     # closed-form etalon pole Q

    f = np.linspace(f_m * 0.97, f_m * 1.03, 1201)      # +-3% window: near-peak Lorentzian core
    T = _etalon_transmission(f, n, L)
    assert T.max() == pytest.approx(1.0, abs=1e-6)     # lossless slab -> unit peak transmission

    lf = lorentzian_fit(f, T)
    ff = fano_fit(f, T, x_kind="freq")

    assert lf.x0 == pytest.approx(f_m, rel=1e-4)
    assert lf.Q == pytest.approx(Q_pole, rel=0.02), "lorentzian Q {} vs pole {}".format(lf.Q, Q_pole)
    assert ff.Q == pytest.approx(Q_pole, rel=0.02), "fano Q {} vs pole {}".format(ff.Q, Q_pole)


def test_tmm_etalon_vs_resonance_pole_finder_optional():
    """OPTIONAL cross-check against optics/resonance.py (item 1.1, written concurrently by
    another agent). Skips gracefully if the module is absent or its API differs -- this gate
    must NOT depend on it."""
    try:
        from dynameta.optics import resonance as _res
    except ImportError:
        pytest.skip("optics/resonance.py not present yet (item 1.1); optional cross-check")

    c = 299_792_458.0
    n = 3.5
    L = 1.0e-6
    m = 4
    f_m = m * c / (2.0 * n * L)
    f = np.linspace(f_m * 0.97, f_m * 1.03, 1201)
    T = _etalon_transmission(f, n, L)
    fit_Q = lorentzian_fit(f, T).Q

    # Probe a few plausible entry points; skip if none matches (do not fail the suite).
    pole_Q = None
    for name in ("etalon_pole", "etalon_poles", "find_poles", "pole_q", "resonance_q"):
        fn = getattr(_res, name, None)
        if fn is None:
            continue
        try:
            out = fn(n=n, thickness_m=L, m=m)
            pole_Q = float(getattr(out, "Q", out))
            break
        except Exception:
            continue
    if pole_Q is None:
        pytest.skip("optics/resonance.py present but no recognized pole-Q entry point; "
                    "cross-check skipped (item 1.1 API not finalized)")
    assert fit_Q == pytest.approx(pole_Q, rel=0.03)


# ---------------------------------------------------------------------------
# GATE 4 -- quasi-BIC Q ~ 1/delta^2 scaling law. Clean synthetic data (5% scatter) must
# recover exponent -2.00 +/- 0.05 with r2 > 0.99; a dataset whose two highest-Q points are
# saturated by an absorption floor (Q_abs) breaks the power law and the r2 collapses.
# ---------------------------------------------------------------------------
def test_quasi_bic_scaling_minus_two():
    rng = np.random.default_rng(0)
    delta = np.array([0.02, 0.03, 0.045, 0.07, 0.10, 0.15, 0.20])
    Q_clean = 1.0 / delta ** 2                                  # canonical Q = C/delta^2, C=1
    Q_noisy = Q_clean * (1.0 + 0.05 * rng.standard_normal(delta.size))

    exponent, prefactor, r2 = quasi_bic_scaling(delta, Q_noisy)
    assert exponent == pytest.approx(-2.0, abs=0.05)
    assert r2 > 0.99
    assert prefactor == pytest.approx(1.0, rel=0.25)           # C recovered to O(scatter)

    # Contaminate: the two smallest-delta (highest-Q) modes are absorption-limited at Q_abs.
    Q_contam = Q_noisy.copy()
    Q_contam[0] = 300.0
    Q_contam[1] = 300.0
    exp_c, _, r2_c = quasi_bic_scaling(delta, Q_contam)
    assert r2_c < 0.95, "saturated Q should collapse the power-law r2 (got {})".format(r2_c)
    assert r2_c < r2 - 0.1                                      # a clear, flagged drop


# ---------------------------------------------------------------------------
# GATE 5 -- byte-stability of the pre-existing resonance_dip / resonance_shift. Goldens were
# captured by running the CURRENT code on this fixed synthetic spectrum BEFORE the item-1.3
# additions; the additive edit must not perturb them (rtol 5e-12).
# ---------------------------------------------------------------------------
def test_resonance_dip_shift_byte_stable():
    lam = np.linspace(1200.0, 1400.0, 41)
    ref = 1.0 - 0.90 / (1.0 + ((lam - 1305.7) / 8.0) ** 2)
    test = 1.0 - 0.85 / (1.0 + ((lam - 1312.3) / 9.5) ** 2)

    dip_nm, dip_val = resonance_dip(lam, ref)
    shift = resonance_shift(lam, ref, test)

    assert dip_nm == pytest.approx(1305.5157431738487, rel=5e-12)
    assert dip_val == pytest.approx(0.10422568467402016, rel=5e-12)
    assert shift == pytest.approx(6.695385825972835, rel=5e-12)

    # edge-fallback branch (discrete minimum at an array edge -> no parabola)
    edge_nm, edge_val = resonance_dip(lam[:9], np.linspace(0.1, 0.9, 9))
    assert edge_nm == pytest.approx(1200.0, rel=5e-12)
    assert edge_val == pytest.approx(0.1, rel=5e-12)
