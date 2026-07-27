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
# GATE 2b (audit Q-3) -- SMALL-AMPLITUDE spectra must fit, not silently return the SEED.
#
# scipy's least_squares `gtol` is an ABSOLUTE bound on the scaled gradient norm, so any finite
# value is really a bound on the spectrum AMPLITUDE: with the shipped gtol = 1e-15 the optimizer
# terminated on its first iteration for small features and _fano_varpro returned the seed
# (gamma = 0.15 * span) -- a 200%-wrong FWHM and a 3x-wrong Q, with NO exception and no flag.
# The controlling quantity is the FEATURE amplitude, not |y|: a tiny feature on an O(1) baseline
# failed identically to a uniformly tiny spectrum.  Both regimes are pinned here, along with
# residual_rms as the tell (it separates a recovered fit from a seed fallback by ~5 decades).
# ---------------------------------------------------------------------------
def test_small_amplitude_spectra_recover_q_not_the_seed():
    x0, g = 1.0e14, 2.0e12
    Q_true = x0 / g                                            # = 50
    x = np.linspace(x0 - 12 * g, x0 + 12 * g, 801)
    lor = 1.0 / (1.0 + (2.0 * (x - x0) / g) ** 2)
    gamma_seed = 0.15 * (x.max() - x.min())                    # the seed the old code returned
    assert abs(gamma_seed / g - 3.6) < 1e-9                    # seed FWHM is 3.6x the truth

    # (a) a 1e-8 FEATURE on a 1.0 baseline (|y| ~ 1: the amplitude, not the scale, is what bites)
    for amp in (1.0, 1e-4, 1e-8):
        lf = lorentzian_fit(x, 1.0 + amp * lor)
        ff = fano_fit(x, 1.0 + amp * lor, x_kind="freq")
        assert lf.Q == pytest.approx(Q_true, rel=0.03), "lorentzian Q at feature {}".format(amp)
        assert ff.Q == pytest.approx(Q_true, rel=0.03), "fano Q at feature {}".format(amp)
        assert lf.fwhm == pytest.approx(g, rel=0.03)
        assert abs(lf.fwhm - gamma_seed) > 0.5 * gamma_seed    # NOT the seed
        assert lf.residual_rms < 1e-4 * amp                    # the tell: fit, not fallback

    # (b) a uniformly SMALL spectrum (magnitude 1e-7 and far below)
    for scale in (1e-7, 1e-9, 1e-12):
        y = scale * (0.2 + lor)
        lf = lorentzian_fit(x, y)
        ff = fano_fit(x, y, x_kind="freq")
        assert lf.Q == pytest.approx(Q_true, rel=0.03), "lorentzian Q at scale {}".format(scale)
        assert ff.Q == pytest.approx(Q_true, rel=0.03), "fano Q at scale {}".format(scale)
        assert lf.amplitude == pytest.approx(scale, rel=0.03)
        assert lf.baseline == pytest.approx(0.2 * scale, rel=0.03)
        assert lf.residual_rms < 1e-6 * (y.max() - y.min())

    # (c) the fit is EXACT (not merely within a few %) once the feature is resolvable
    lf = lorentzian_fit(x, 1e-12 * (0.2 + lor))
    assert lf.fwhm == pytest.approx(g, rel=1e-6)
    assert lf.x0 == pytest.approx(x0, rel=1e-9)


# ---------------------------------------------------------------------------
# GATE 2c (audit Q-10) -- the q -> +-inf cut-off must fire on DEGENERACY, not on size.
#
# fano_fit recovers (a_bg, b_bg, q) from the fitted symmetric+dispersive coefficients
# (C0, C_L, C_D) with C_L = b_bg(q^2-1), C_D = 2 b_bg q. The cut-off used to test
# b_bg <= 1e-12 (|C0| + Rq); since Rq -> b_bg q^2 that reads 1 <= 1e-12 q^2 and fired from
# |q| ~ 1e6 -- on data the fit had reproduced to 1e-16 -- and then returned q = inf ALONGSIDE a
# finite b_bg, a triple that reconstructs to inf. The controlling quantity is the DISPERSIVE
# weight: C_D / (|C0| + Rq) is exactly 2/q, so it is a conditioning test on q itself.
#
# Pinned here: (a) large-but-recoverable q comes back as a number and its triple RECONSTRUCTS
# the data; (b) the genuine peak limit returns a SELF-CONSISTENT triple (b_bg exactly 0.0,
# a_bg the true background, peak_height the true peak); (c) the q -> 0 dip branch is untouched
# by the cut-off; (d) every ordinary fit is BIT-IDENTICAL to the old recovery.
# ---------------------------------------------------------------------------
def _fano_curve(x, x0, gamma, a_bg, b_bg, q):
    eps = 2.0 * (x - x0) / gamma
    return a_bg + b_bg * (q + eps) ** 2 / (1.0 + eps ** 2)


def _reconstruct(fit, x):
    """T(x) from the RETURNED FanoFit record -- through the finite-q form when q is finite, and
    through the documented q -> +-inf limit T = a_bg + peak_height/(1+eps^2) when it is not."""
    eps = 2.0 * (x - fit.omega0) / fit.gamma_fwhm
    if np.isfinite(fit.q):
        return fit.a_bg + fit.b_bg * (fit.q + eps) ** 2 / (1.0 + eps ** 2)
    return fit.a_bg + fit.peak_height / (1.0 + eps ** 2)


def test_large_q_is_recovered_and_the_returned_triple_reconstructs():
    x0, gamma, a_bg, b_bg = 1.935e14, 2.0e12, 0.1, 0.7
    eps = np.linspace(-30.0, 30.0, 2001)
    x = x0 + 0.5 * gamma * eps

    # (a) LARGE but exactly recoverable q -- the regime the old cut-off swallowed
    for q in (1.0e5, 1.0e6, 1.0e8, 1.0e10, 1.0e12):
        T = _fano_curve(x, x0, gamma, a_bg, b_bg, q)
        fit = fano_fit(x, T, x_kind="freq")
        assert np.isfinite(fit.q), "q={} was discarded as inf".format(q)
        assert fit.q == pytest.approx(q, rel=1e-4), (q, fit.q)
        assert fit.b_bg == pytest.approx(b_bg, rel=1e-3)
        assert fit.peak_height == pytest.approx(b_bg * q ** 2, rel=1e-3)
        # the RETURNED record reproduces the data it was fitted to
        assert np.max(np.abs(_reconstruct(fit, x) - T)) / np.max(np.abs(T)) < 1e-10, q
        assert fit.residual_rms < 1e-12 * np.max(np.abs(T))

    # (b) the GENUINE peak limit (a pure Lorentzian: no dispersive component at all) -> the
    #     self-consistent inf triple.  Pre-fix this returned q = inf WITH b_bg = 0.7.
    peak = 0.7
    T = a_bg + peak / (1.0 + eps ** 2)
    fit = fano_fit(x, T, x_kind="freq")
    # The SIGN of the reported inf follows the residual dispersive coefficient, which on an
    # exactly symmetric peak is roundoff (|C_D| ~ 5e-18 of the model scale) -- +inf and -inf are
    # the SAME lineshape in this limit, so only the magnitude is pinned.
    assert abs(fit.q) == float("inf")
    assert fit.b_bg == 0.0                                  # EXACTLY zero, not 1e-25
    assert fit.a_bg == pytest.approx(a_bg, abs=1e-12)       # the FULL background
    assert fit.peak_height == pytest.approx(peak, rel=1e-9)
    assert np.max(np.abs(_reconstruct(fit, x) - T)) / np.max(np.abs(T)) < 1e-12

    # (c) the DIP branch (C_L <= 0) is untouched: C_D -> 0 there means q -> 0, NOT q -> inf.
    for q in (0.0, 1.0e-13):
        T = _fano_curve(x, x0, gamma, a_bg, b_bg, q)
        fit = fano_fit(x, T, x_kind="freq")
        assert np.isfinite(fit.q) and abs(fit.q) < 1e-6, (q, fit.q)
        assert fit.b_bg == pytest.approx(b_bg, rel=1e-6)


def test_q_reliable_flags_a_magnitude_the_noise_cannot_support():
    """AUDIT Q-10 residual. The conditioning cut-off asks whether the dispersive weight C_D clears
    the LINEAR SOLVE's roundoff. On measured data the binding floor is the NOISE, ~13 decades
    higher -- and because C_D relative to the model scale is exactly 2/q, a large q is measured by
    an ever-fainter asymmetry. So the fit reproduces noisy data to its noise level and returns a q
    that is orders of magnitude wrong, with nothing in the record to say so: measured 22x wrong at
    1e-3 fractional noise with q_true = 1e6.

    ``FanoFit.q_reliable`` is that missing statement. It compares |C_D| with residual_rms; the
    calibration (6 noise levels x 12 decades of q) maps that ratio to the worst-case error in q as
    ~0.2/ratio, and the threshold 20 means "q good to ~1 %". This gate pins BOTH anchors of the
    calibration and the fact that no returned VALUE changed."""
    x0, gamma, a_bg, b_bg = 1.935e14, 2.0e12, 0.1, 0.7
    eps = np.linspace(-30.0, 30.0, 2001)
    x = x0 + 0.5 * gamma * eps

    # (a) CLEAN fits stay reliable at every magnitude -- 12 decades of q, measured SNR >= 1.5e4
    for q in (1.0, 1.0e2, 1.0e4, 1.0e6, 1.0e9, 1.0e12):
        fit = fano_fit(x, _fano_curve(x, x0, gamma, a_bg, b_bg, q), x_kind="freq")
        assert fit.q_reliable is True, (q, fit.q, fit.residual_rms)
        assert fit.q == pytest.approx(q, rel=1e-4)

    # (b) 1e-3 noise with q >= 1e6: FLAGGED, every trial (measured SNR <= 0.58, a 35x margin),
    #     and the flag is right -- the returned q really is order-of-magnitude wrong
    rng = np.random.default_rng(11)
    worst_err = 0.0
    for q in (1.0e6, 1.0e9, 1.0e12):
        for _ in range(6):
            T = _fano_curve(x, x0, gamma, a_bg, b_bg, q)
            T = T + 1e-3 * np.ptp(T) * rng.standard_normal(T.size)
            fit = fano_fit(x, T, x_kind="freq")
            assert fit.q_reliable is False, (q, fit.q, fit.residual_rms)
            if np.isfinite(fit.q):
                worst_err = max(worst_err, abs(fit.q / q - 1))
    assert worst_err > 0.5, worst_err        # the flagged fits ARE badly wrong (measured ~1)

    # (c) noise that the asymmetry easily clears is NOT flagged (small q, same noise)
    for q in (0.5, 2.0, 10.0):
        T = _fano_curve(x, x0, gamma, a_bg, b_bg, q)
        T = T + 1e-3 * np.ptp(T) * rng.standard_normal(T.size)
        fit = fano_fit(x, T, x_kind="freq")
        assert fit.q_reliable is True, (q, fit.q, fit.residual_rms)
        assert fit.q == pytest.approx(q, rel=0.05)

    # (d) the +-inf branch is a DEGENERACY FLAG, not a measurement -- never "reliable"
    fit = fano_fit(x, a_bg + 0.7 / (1.0 + eps ** 2), x_kind="freq")
    assert abs(fit.q) == float("inf") and fit.q_reliable is False
    assert fit.peak_height == pytest.approx(0.7, rel=1e-9)      # ... the usable number is finite

    # (e) NOTHING ELSE MOVED. Every other field is bit-for-bit what the same fit gave before the
    #     flag existed -- reproduced here by rebuilding the record from _fano_varpro directly.
    from dynameta.analysis import _fano_varpro
    for q in (0.3, 7.0, 1.0e5):
        T = _fano_curve(x, x0, gamma, a_bg, b_bg, q)
        fit = fano_fit(x, T, x_kind="freq")
        x0f, gf, coef, rms = _fano_varpro(x, T, ncols=3)
        assert fit.omega0 == x0f and fit.gamma_fwhm == gf and fit.residual_rms == rms
        C_D = float(coef[2])
        assert fit.q_reliable == bool(np.isfinite(fit.q)
                                      and (rms <= 0.0 or abs(C_D) >= 20.0 * rms))


def test_fano_recovery_is_bit_identical_outside_the_q10_band():
    """Byte-identity gate for Q-10. Everything the fix does NOT intend to change must come back
    bit-for-bit: the OLD recovery is re-implemented here from the pre-fix source and compared to
    the shipped one on the SAME varpro coefficients, so only the recovery algebra is under test.
    The intended differences are exactly two: the peak branch (C_L > 0) inside the old cut-off but
    outside the new one, and b_bg/a_bg in the genuine q -> inf limit."""
    from dynameta.analysis import _fano_varpro

    def old_recovery(coef):
        C0, C_L, C_D = float(coef[0]), float(coef[1]), float(coef[2])
        Rq = float(np.hypot(C_L, C_D))
        if C_L > 0.0:
            b = (C_D * C_D) / (2.0 * (Rq + C_L)) if (Rq + C_L) > 0.0 else 0.0
        else:
            b = 0.5 * (-C_L + Rq)
        b_scale = abs(C0) + Rq + 1e-300
        if b <= 1e-12 * b_scale:
            return (float(np.inf) if C_D >= 0.0 else float(-np.inf)), C0 - b, b
        return C_D / (2.0 * b), C0 - b, b

    rng = np.random.default_rng(7)
    x0, gamma = 1.935e14, 2.0e12
    eps = np.linspace(-14.0, 14.0, 561)
    x = x0 + 0.5 * gamma * eps
    checked = 0
    for q in (-10.0, -5.0, -1.0, -0.2, 0.0, 0.3, 2.0, 10.0, 100.0, 1.0e4):
        for a_bg, b_bg in ((0.15, 0.60), (0.0, 1.0), (-0.4, 0.05)):
            T = _fano_curve(x, x0, gamma, a_bg, b_bg, q)
            T = T + 0.003 * np.ptp(T) * rng.standard_normal(T.size)      # noisy, like real data
            fit = fano_fit(x, T, x_kind="freq")
            q_old, a_old, b_old = old_recovery(_fano_varpro(x, T, 3)[2])
            assert fit.q == q_old, (q, a_bg, b_bg, fit.q, q_old)         # BIT-identical
            assert fit.a_bg == a_old
            assert fit.b_bg == b_old
            checked += 1
    assert checked == 30


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


def test_tmm_etalon_fit_q_matches_resonance_pole_finder():
    """Cross-INSTRUMENT gate: the Q fitted from a driven `tmm`-package transmission spectrum must
    equal the Q of the complex-omega pole that optics.resonance locates numerically for the SAME
    etalon. Two independent routes to one number -- a real-axis lineshape fit of a third-party TMM
    spectrum, and an argument-principle pole hunt on dynameta's own characteristic-matrix
    denominator -- with the closed form above as the shared anchor.

    AUDIT X-2/T-7/T-24: this test had NEVER EXECUTED. It probed five guessed entry points with a
    `fn(n=..., thickness_m=..., m=...)` signature that matches none of them, swallowed the
    resulting TypeError in `except Exception: continue`, and skipped with an "API not finalized"
    excuse -- in the very commit that created the API. It is now written against the real
    signatures (smatrix_pole_func -> find_poles -> pole_q) with no try/except and no skip.
    tests/test_resonance_crossgate.py runs the sibling three-instrument gate, but its spectrum
    comes from `layered_smatrix_complex`; this one is the only place the `tmm` package's driven
    spectrum is tied to the pole finder.
    """
    from dynameta.optics.resonance import find_poles, pole_q, smatrix_pole_func

    c = 299_792_458.0
    n = 3.5
    L = 1.0e-6
    m = 4
    f_m = m * c / (2.0 * n * L)
    f = np.linspace(f_m * 0.97, f_m * 1.03, 1201)
    T = _etalon_transmission(f, n, L)
    fit_Q = lorentzian_fit(f, T).Q

    om_m = 2.0 * np.pi * f_m
    r12 = (n - 1.0) / (n + 1.0)
    Q_pole_cf = -m * np.pi / (2.0 * np.log(abs(r12)))          # closed form, for the search box
    half_width = om_m / (2.0 * Q_pole_cf)                      # |Im omega| of the pole
    D = smatrix_pole_func([(complex(n) ** 2, L)])              # (eps, thickness) layer spec
    poles = find_poles(D, om_m - 0.6j * half_width, 0.06 * om_m + 1.5j * half_width, n_grid=48)
    assert poles, "pole finder found no pole near the m={} etalon resonance".format(m)
    pole = min(poles, key=lambda p: abs(p.real - om_m))
    pole_Q = pole_q(pole)

    assert pole.real == pytest.approx(om_m, rel=1e-8)          # the pole finder found THIS mode
    assert pole_Q == pytest.approx(Q_pole_cf, rel=1e-6)        # ... at the closed-form width
    assert fit_Q == pytest.approx(pole_Q, rel=0.03)            # THE cross-instrument statement


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
# GATE 5 -- stability of the pre-existing resonance_dip / resonance_shift. Goldens were
# captured by running the CURRENT code on this fixed synthetic spectrum BEFORE the item-1.3
# additions; the additive edit must not perturb them. The parabola-refined quantities go
# through a BLAS-backed fit whose last digits are platform-dependent (measured: CI numpy
# builds drift ~1.4e-10 relative, each leg differently), so those goldens pin at rel=1e-8
# (100x the observed drift, still far below any behavioral change); the fit-free edge
# fallbacks stay at 5e-12.
# ---------------------------------------------------------------------------
def test_resonance_dip_shift_byte_stable():
    lam = np.linspace(1200.0, 1400.0, 41)
    ref = 1.0 - 0.90 / (1.0 + ((lam - 1305.7) / 8.0) ** 2)
    test = 1.0 - 0.85 / (1.0 + ((lam - 1312.3) / 9.5) ** 2)

    dip_nm, dip_val = resonance_dip(lam, ref)
    shift = resonance_shift(lam, ref, test)

    assert dip_nm == pytest.approx(1305.5157431738487, rel=1e-8)
    assert dip_val == pytest.approx(0.10422568467402016, rel=1e-8)
    assert shift == pytest.approx(6.695385825972835, rel=1e-8)

    # edge-fallback branch (discrete minimum at an array edge -> no parabola)
    edge_nm, edge_val = resonance_dip(lam[:9], np.linspace(0.1, 0.9, 9))
    assert edge_nm == pytest.approx(1200.0, rel=5e-12)
    assert edge_val == pytest.approx(0.1, rel=5e-12)
