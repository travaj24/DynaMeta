"""Gates for the SPDC design tier (roadmap 4.4) -- the Helt-Liscidini-Sipe quantum-classical
correspondence built on twm_reference. Pure numpy/scipy; runs in CI. Gates:
  (6) textbook CW bulk-crystal pair rate recovered in the uniform limit -- reconstructed
      in-test from the correspondence itself + dimensional analysis (pairs/s per W);
  (7) JSA anti-diagonal width == pump bandwidth, diagonal width == phase-matching bandwidth
      on a constructed separable case (20%);
  (8) Schmidt number == 1 for a separable/matched case, >> 1 (>5) for a long-crystal CW case;
  (9) QPM shifts the JSA centre to the twm-predicted (omega_s, omega_i) (1%).
"""
import warnings

import numpy as np
import pytest

from dynameta.constants import C_LIGHT, EPS0
from dynameta.optics.twm_reference import qpm_period_for
from dynameta.optics.spdc_design import (
    pair_rate_from_sfg, spectral_pair_rate_closed_form, jsa, jsi, schmidt_number,
    heralded_bandwidths, HELT_SIPE_PREFACTOR,
)

WP = 2.0 * np.pi * 3.75e14        # pump ~ 375 THz
WS0 = WP / 2.0                    # degenerate signal/idler
WI0 = WP - WS0


# ------------------------------------------------------------------ gate 6: CW bulk pair rate
def test_cw_bulk_pair_rate_and_dimensions():
    d_eff, L, P, A = 1.0e-12, 1.0e-3, 0.1, 1.0e-8
    n_s = n_i = n_p = 2.0
    ws = np.linspace(WS0 - 2.0e13, WS0 + 2.0e13, 600)

    out = pair_rate_from_sfg(ws, WP, d_eff, L, n_s=n_s, n_i=n_i, n_p=n_p,
                             pump_power_W=P, area_m2=A, dk_func=None)

    # (a) recover the closed form CONSTRUCTED FROM THE CORRESPONDENCE ITSELF (independent of
    #     the code path): dR/domega_s = (omega_s omega_i d_eff^2 L^2 P) /
    #     (pi n_s n_i n_p eps0 c^3 A) sinc^2(dk L/2). Written here from SI constants; matching
    #     the code confirms both the 1/(2pi) prefactor and the units.
    omega_i = WP - ws
    closed = (ws * omega_i * d_eff ** 2 * L ** 2 * P) / \
             (np.pi * n_s * n_i * n_p * EPS0 * C_LIGHT ** 3 * A)   # dk = 0 -> sinc^2 = 1
    assert np.max(np.abs(out["spectral_density"] - closed) / np.max(closed)) < 1e-9
    # the module's own closed-form helper agrees too
    cf = spectral_pair_rate_closed_form(ws, WP, d_eff, L, n_s=n_s, n_i=n_i, n_p=n_p,
                                        pump_power_W=P, area_m2=A, dk=0.0)
    assert np.max(np.abs(out["spectral_density"] - cf) / np.max(cf)) < 1e-9

    # (b) the correspondence prefactor is exactly 1/(2 pi)
    assert abs(HELT_SIPE_PREFACTOR - 1.0 / (2.0 * np.pi)) < 1e-15

    # (c) DIMENSIONAL ANALYSIS -> pairs/s per W: R is a finite positive rate, LINEAR in pump
    #     power (pairs/s/W), quadratic in length (dR/domega_s ~ L^2), inverse in beam area.
    assert out["rate_pairs_per_s"] > 0 and np.isfinite(out["rate_pairs_per_s"])
    out2 = pair_rate_from_sfg(ws, WP, d_eff, L, n_s=n_s, n_i=n_i, n_p=n_p,
                              pump_power_W=2.0 * P, area_m2=A)
    assert abs(out2["rate_pairs_per_s"] / out["rate_pairs_per_s"] - 2.0) < 1e-9    # linear in P
    assert abs(out["rate_per_watt"] - out["rate_pairs_per_s"] / P) < 1e-6 * out["rate_per_watt"]
    outL = pair_rate_from_sfg(ws, WP, d_eff, 2.0 * L, n_s=n_s, n_i=n_i, n_p=n_p,
                              pump_power_W=P, area_m2=A)
    assert abs(np.max(outL["spectral_density"]) / np.max(out["spectral_density"]) - 4.0) < 1e-9
    outA = pair_rate_from_sfg(ws, WP, d_eff, L, n_s=n_s, n_i=n_i, n_p=n_p,
                              pump_power_W=P, area_m2=A / 2.0)
    assert abs(outA["rate_pairs_per_s"] / out["rate_pairs_per_s"] - 2.0) < 1e-9    # ~ 1/A


# ------------------------------------------------------------------ gate 7: JSA widths
def _pump(sigma):
    return lambda u: np.exp(-((u - WP) / (2.0 * sigma)) ** 2)


def test_jsa_widths_pump_and_phase_matching():
    # separable-in-(u=ws+wi, v=ws-wi) construction: pump on u, phase matching on v, with a
    # PHYSICAL group-velocity-scale slope (~ n/c) so both widths resolve on one grid.
    sigma_p = 8.0e11
    L = 1.0e-3
    slope = 5.0e-9                    # d(dk)/d(ws-wi), group-velocity-mismatch scale
    dkf = lambda a, b: slope * ((a - b) - (WS0 - WI0))
    span, N = 5.0e12, 401
    wsg = np.linspace(WS0 - span, WS0 + span, N)
    wig = np.linspace(WI0 - span, WI0 + span, N)

    F = jsa(wsg, wig, _pump(sigma_p), dkf, L)
    hb = heralded_bandwidths(F, wsg, wig)

    pump_fwhm = 2.0 * np.sqrt(2.0 * np.log(2.0)) * sigma_p         # FWHM of |alpha(u)|^2
    pm_fwhm = 2.0 * 1.39156 * 2.0 / (slope * L)                   # FWHM of |sinc(slope v L/2)|^2
    # AUDIT Q-13 re-pin (twice). The rotated marginals carry the grid Jacobian and are taken over
    # the inscribed constant-v-extent window, which removes the domain tent; and each kept bin is
    # now reported at its quadrature-measure CENTROID rather than at the geometric bin centre,
    # which removes the -1/nbin abscissa compression the old "-0.246 % discretisation residual"
    # actually was (1/401 = 0.249 %). This gate's 20 % band was sized for the un-Jacobianed
    # histogram; it is tightened to 0.1 %.
    #   pump : 1.90186619e12 (+0.956 %) -> 1.87921289e12 (-0.246 %) -> 1.88391092e12 (+0.0029 %)
    #   pm   : 1.13544577e12 (+1.994 %) -> 1.11050811e12 (-0.246 %) -> 1.11328438e12 (+0.0033 %)
    assert abs(hb["antidiagonal_bandwidth"] - pump_fwhm) / pump_fwhm < 1e-3
    assert abs(hb["diagonal_bandwidth"] - pm_fwhm) / pm_fwhm < 1e-3
    assert hb["antidiagonal_bandwidth"] == pytest.approx(1.883910921e12, rel=1e-6)
    assert hb["diagonal_bandwidth"] == pytest.approx(1.113284383e12, rel=1e-6)
    # aliases carry the physical names
    assert hb["pump_bandwidth"] == hb["antidiagonal_bandwidth"]
    assert hb["phase_matching_bandwidth"] == hb["diagonal_bandwidth"]
    # ... and this configuration resolves everything it reports
    assert all(hb[k] for k in hb if k.endswith("_resolved"))


def test_rotated_widths_do_not_depend_on_the_PARITY_of_the_grid(recwarn):
    """AUDIT Q-13 residual: the even-N binning comb. The rotated coordinate on a uniform grid is
    a COMB (one value per i+j), and binning it on a linspace of nbin bins put the reported mass
    at the geometric bin CENTRE -- which is displaced from the comb line by a saw of up to half a
    bin whose phase flips with the parity of N. Measured on this configuration: +0.827 % at
    N = 400 against -0.246 % at N = 401 (a 1.07 % parity step), growing to ~+10 % at N = 50.
    Reporting each bin at the measure centroid of the coordinate inside it removes it."""
    sigma_p, L, slope = 8.0e11, 1.0e-3, 5.0e-9
    dkf = lambda a, b: slope * ((a - b) - (WS0 - WI0))
    span = 5.0e12
    pump_fwhm = 2.0 * np.sqrt(2.0 * np.log(2.0)) * sigma_p
    pm_fwhm = 2.0 * 1.39156 * 2.0 / (slope * L)

    def err(N):
        wsg = np.linspace(WS0 - span, WS0 + span, N)
        wig = np.linspace(WI0 - span, WI0 + span, N)
        hb = heralded_bandwidths(jsa(wsg, wig, _pump(sigma_p), dkf, L), wsg, wig)
        assert hb["pump_bandwidth_resolved"] and hb["phase_matching_bandwidth_resolved"], N
        return (abs(hb["pump_bandwidth"] / pump_fwhm - 1),
                abs(hb["phase_matching_bandwidth"] / pm_fwhm - 1))

    e400, e401 = err(400), err(401)
    # the shipped configuration: even and odd N agree to better than 2x on BOTH legs (measured
    # 1.13x on the pump leg, 1.16x on the phase-matching leg; it was 3.4x / 4.5x before)
    for a, b in zip(e400, e401):
        assert max(a, b) <= 2.0 * min(a, b), (e400, e401)
        assert max(a, b) < 1e-4, (e400, e401)               # and both are < 0.01 % of the truth
    # ... and the parity step stays bounded where it used to be worst (N ~ 50: +10 % vs -0.25 %)
    e50, e51 = err(50), err(51)
    for a, b in zip(e50, e51):
        assert max(a, b) < 3e-3, (e50, e51)                 # < 0.3 %, was 10 %
    assert not [w for w in recwarn if issubclass(w.category, RuntimeWarning)]


# ------------------------------------------- gate 7b: rotated marginals vs quadrature (Q-13)
_SIG_U, _SIG_V = 8.0e11, 6.0e11        # Gaussian widths of the separable test JSA, in u and v


def _sep_jsa(ws, wi):
    """f = exp(-u^2/(4 sig_u^2)) exp(-v^2/(4 sig_v^2)) with u = ws+wi-WP, v = (ws-wi)-(WS0-WI0).
    |f|^2 is then Gaussian with std sig_u in u and sig_v in v, so both rotated FWHMs are known in
    closed form: 2 sqrt(2 ln 2) sig."""
    u = ws + wi - WP
    v = (ws - wi) - (WS0 - WI0)
    return np.exp(-u ** 2 / (4.0 * _SIG_U ** 2)) * np.exp(-v ** 2 / (4.0 * _SIG_V ** 2))


def _rotated_quadrature_oracle(H, n=2001):
    """INDEPENDENT instrument for the two rotated widths: sample the ANALYTIC JSA on a regular
    (u, v) grid -- no rotation of a rectangular (omega_s, omega_i) grid, no histogram anywhere --
    and integrate |f|^2 along the other coordinate with the trapezoid rule over the same
    +-H window, then take the FWHM of the resulting 1-D profile."""
    from dynameta.optics.spdc_design import _fwhm
    out = []
    for rotate in (False, True):
        t = np.linspace(-H, H, n)
        s = np.linspace(-H, H, n)
        T, S = np.meshgrid(t, s, indexing="ij")
        U, V = (T, S) if not rotate else (S, T)
        P = np.abs(_sep_jsa(WS0 + 0.5 * (U + V), WI0 + 0.5 * (U - V))) ** 2
        # trapezoid along the second axis, written out (no numpy/scipy trapezoid spelling works
        # across the declared numpy floor -- tests/test_numerics.py machine-checks that, and an
        # oracle should not borrow the shim from the module it is auditing anyway)
        ds = s[1] - s[0]
        out.append(_fwhm(t, ds * (P.sum(axis=1) - 0.5 * P[:, 0] - 0.5 * P[:, -1])))
    return out                                          # (pump width, phase-matching width)


def test_rotated_marginals_match_a_direct_quadrature_oracle():
    """AUDIT Q-13. ``heralded_bandwidths``' rotated marginals used to be un-Jacobianed histograms
    of a rectangular grid: ``np.histogram(omega_s +/- omega_i, weights=|f|^2)`` sums bare SAMPLES,
    which is an integral only when every cell has the same area, and the strip of constant
    omega_s + omega_i has a length that varies with it (a tent of the grid, not of the physics).

    Gated here against two independent references: the closed-form width of a separable Gaussian
    JSA, and a direct quadrature of the ANALYTIC JSA on a regular (u, v) grid (no rotation of a
    rectangular grid, no histogram) -- see ``_rotated_quadrature_oracle``."""
    span, N = 5.0e12, 401
    wsg = np.linspace(WS0 - span, WS0 + span, N)
    wig = np.linspace(WI0 - span, WI0 + span, N)
    WS, WI = np.meshgrid(wsg, wig, indexing="ij")
    hb = heralded_bandwidths(_sep_jsa(WS, WI), wsg, wig)

    closed_u = 2.0 * np.sqrt(2.0 * np.log(2.0)) * _SIG_U
    closed_v = 2.0 * np.sqrt(2.0 * np.log(2.0)) * _SIG_V
    orac_u, orac_v = _rotated_quadrature_oracle(span)
    # the oracle itself reproduces the closed form -- it is a valid instrument
    assert orac_u == pytest.approx(closed_u, rel=1e-3)
    assert orac_v == pytest.approx(closed_v, rel=1e-3)
    # ... and the shipped estimator tracks it. The band is 1e-4, not the 5e-3 the bin-centre
    # abscissa needed: with the measure-centroid abscissa the estimator sits 2.8e-5 from this
    # oracle, which itself sits 1.3e-6 from the closed form at n = 2001 (and 6.6e-9 at 16001), so
    # the oracle is a 20x sharper instrument than the quantity it is measuring.
    assert hb["pump_bandwidth"] == pytest.approx(orac_u, rel=1e-4)
    assert hb["phase_matching_bandwidth"] == pytest.approx(orac_v, rel=1e-4)

    # convergence: halving the grid roughly doubles the (binning-limited) residual, and it stays
    # a residual -- this is discretisation, not a systematic tent. Measured 2.8e-5 -> 6.7e-5.
    half = heralded_bandwidths(_sep_jsa(*np.meshgrid(wsg[::2], wig[::2], indexing="ij")),
                               wsg[::2], wig[::2])
    assert abs(half["pump_bandwidth"] / orac_u - 1) > abs(hb["pump_bandwidth"] / orac_u - 1)
    assert abs(half["pump_bandwidth"] / orac_u - 1) < 1e-4


def test_rotated_marginals_are_correct_on_a_NON_UNIFORM_grid():
    """AUDIT Q-13, the Jacobian statement. The same physical JSA sampled on a STRETCHED
    (omega_s, omega_i) grid must give the same widths. A bare sample histogram does not: measured
    -63.6 % (pump) / -53.6 % (phase matching) on a power-1.6 stretch and -88.4 % / -83.9 % on
    power 2.2. With the per-cell trapezoid weights it is a few percent."""
    span, N = 5.0e12, 401
    g = np.linspace(-1.0, 1.0, N)
    closed_u = 2.0 * np.sqrt(2.0 * np.log(2.0)) * _SIG_U
    closed_v = 2.0 * np.sqrt(2.0 * np.log(2.0)) * _SIG_V
    for power in (1.0, 1.6, 2.2):
        d = np.sign(g) * np.abs(g) ** power * span
        wsg, wig = WS0 + d, WI0 + d
        WS, WI = np.meshgrid(wsg, wig, indexing="ij")
        hb = heralded_bandwidths(_sep_jsa(WS, WI), wsg, wig)
        assert hb["pump_bandwidth"] == pytest.approx(closed_u, rel=0.03), power
        assert hb["phase_matching_bandwidth"] == pytest.approx(closed_v, rel=0.07), power


def test_strongly_nonuniform_grids_are_REFUSED_not_silently_wrong():
    """AUDIT Q-13 residual: the SILENT non-convergent failure. The per-cell trapezoid weights make
    the rotated marginal an integral, but they do not make it convergent on an arbitrary grid --
    and nothing said so. Measured before the guard, all four ``*_resolved`` flags True:

        log-spaced grid   phase-matching width -29.8 % at N=401, -30.2 % at 801, -30.7 % at 1601
                          (WORSE under refinement -- not a discretisation residual)
        random spacing    -82 % (seed 7), down to -98 % on other seeds

    The detector is the estimator's own quadrature measure: over the inscribed constant-v-extent
    window every u-bin covers the same area, so ``max(den)/min(den)`` is 1 exactly when the
    un-normalised per-bin sum IS the marginal. Calibrated so the power-law stretches (which ARE
    accurate to a few percent) pass and the log / random grids trip -- the separation measured on
    the JSA of ``_sep_jsa`` is 1.485 (power-2.2, worst passing STRETCH), 1.988 (worst passing
    grid of any kind, over 224 uniform grids of unequal N and unequal span) vs 2.233 (log 1601)
    and 2.274 (worst random seed).

    A SECOND statistic is needed for a second failure, which the ratio cannot see: when the grid
    SPAN sits at the floating-point resolution of the CARRIER the rotated coordinate aliases onto
    a handful of values, and the measure across those few bins is perfectly uniform (1.39). The
    occupied-bin FRACTION separates that one -- 0.98-1.00 on every healthy family, 0.014-0.11
    there. Part (d) below.

    NOT attempted: dividing by ``den``. That is the strip AVERAGE, and it is worse on the grids
    that work -- measured -8.8 % / -22.6 % on the power-1.6 / power-2.2 stretches where the plain
    sum is +1.0 % / -1.0 %."""
    span, N = 5.0e12, 401
    closed_u = 2.0 * np.sqrt(2.0 * np.log(2.0)) * _SIG_U
    closed_v = 2.0 * np.sqrt(2.0 * np.log(2.0)) * _SIG_V

    def widths(wsg, wig):
        WS, WI = np.meshgrid(wsg, wig, indexing="ij")
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            hb = heralded_bandwidths(_sep_jsa(WS, WI), wsg, wig)
        return hb, [str(w.message) for w in rec if issubclass(w.category, RuntimeWarning)]

    def _log(n):
        d = np.geomspace(1.0, 11.0, n) - 6.0
        return d / d.max() * span

    def _random(n, seed=7):
        d = np.sort(np.random.default_rng(seed).uniform(-span, span, n))
        d[0], d[-1] = -span, span
        return d

    # (a) the loud failures are refused, on BOTH rotated legs, and they say why
    for tag, d in (("log-401", _log(401)), ("log-801", _log(801)),
                   ("random-401", _random(N))):
        hb, flagged = widths(WS0 + d, WI0 + d)
        assert np.isnan(hb["pump_bandwidth"]), tag
        assert np.isnan(hb["phase_matching_bandwidth"]), tag
        assert hb["pump_bandwidth_resolved"] is False, tag
        assert hb["phase_matching_bandwidth_resolved"] is False, tag
        assert np.isnan(hb["antidiagonal_bandwidth"]) and np.isnan(hb["diagonal_bandwidth"]), tag
        # ... and the message names the grid requirement, not the window
        assert any("UNIFORMLY spaced" in m and "quadrature measure" in m for m in flagged), tag
        # the signal / idler legs are NOT condemned by a rotated-axis defect
        assert hb["signal_bandwidth_resolved"] is True, tag
        assert np.isfinite(hb["signal_bandwidth"]), tag
        # both rotated marginals still come back in full for inspection (the nan contract)
        for key in ("pump_marginal", "phase_matching_marginal"):
            ax, mg = hb[key]
            assert ax.shape == mg.shape and ax.size > 1, (tag, key)

    # (b) NO false fire on the grids that work: uniform (equal and unequal N and spans) and the
    #     power-law stretches this module's Jacobian statement was gated on
    g = np.linspace(-1.0, 1.0, N)
    ok = [("power-1.6", *(2 * [np.sign(g) * np.abs(g) ** 1.6 * span])),
          ("power-2.2", *(2 * [np.sign(g) * np.abs(g) ** 2.2 * span])),
          ("uniform 401", np.linspace(-span, span, 401), np.linspace(-span, span, 401)),
          ("uniform 400", np.linspace(-span, span, 400), np.linspace(-span, span, 400)),
          ("uniform 301x401", np.linspace(-span, span, 301), np.linspace(-span, span, 401)),
          ("unequal span 0.4", np.linspace(-span, span, 401),
           np.linspace(-0.4 * span, 0.4 * span, 401))]
    for tag, ds, di in ok:
        hb, flagged = widths(WS0 + ds, WI0 + di)
        assert hb["pump_bandwidth_resolved"] is True, (tag, flagged)
        assert hb["phase_matching_bandwidth_resolved"] is True, (tag, flagged)
        assert not flagged, (tag, flagged)
        assert hb["pump_bandwidth"] == pytest.approx(closed_u, rel=0.03), tag
        assert hb["phase_matching_bandwidth"] == pytest.approx(closed_v, rel=0.07), tag

    # (c) NO false fire on a grid centred at ZERO either. `omega_s + omega_i` cancels there to
    #     ~1e-16 relative, which scatters each comb line across a linspace bin edge and made the
    #     measure spread read 3.13-4.35 on perfectly uniform grids whose width error is < 0.7 %.
    #     The comb-aware bin edges remove it (measured spread 1.002-1.020).
    for N in (100, 200, 400, 401, 800):
        z = np.linspace(-5.0, 5.0, N)
        ZS, ZI = np.meshgrid(z, z, indexing="ij")
        u, v = ZS + ZI, ZS - ZI
        F = np.exp(-u ** 2 / (4.0 * 0.8 ** 2)) * np.exp(-v ** 2 / (4.0 * 0.6 ** 2))
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            hb = heralded_bandwidths(F, z, z)
        assert hb["pump_bandwidth_resolved"] is True, N
        assert hb["phase_matching_bandwidth_resolved"] is True, N
        assert not [w for w in rec if issubclass(w.category, RuntimeWarning)], N
        assert hb["pump_bandwidth"] == pytest.approx(2.0 * np.sqrt(2.0 * np.log(2.0)) * 0.8,
                                                     rel=1e-3), N

    # (d) the OTHER refusal: a rotated coordinate floating point has aliased away. A 5 rad/s span
    #     on a 1.2e15 rad/s carrier puts the grid spacing below the ulp of omega_s + omega_i, so
    #     400 distinct frequencies collapse onto 11 distinct sums. The measure across those 11 is
    #     uniform (1.39), so the ratio test cannot see it -- the occupied-bin fraction can.
    #     Measured before the guard: pump width reported as -100 % / +112 % of the truth.
    for N in (100, 400, 801):
        c = 1.178e15
        z = np.linspace(c - 5.0, c + 5.0, N)
        ZS, ZI = np.meshgrid(z, z, indexing="ij")
        u, v = (ZS - c) + (ZI - c), (ZS - c) - (ZI - c)
        F = np.exp(-u ** 2 / (4.0 * 0.8 ** 2)) * np.exp(-v ** 2 / (4.0 * 0.6 ** 2))
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            hb = heralded_bandwidths(F, z, z)
        assert np.isnan(hb["pump_bandwidth"]) and np.isnan(hb["phase_matching_bandwidth"]), N
        assert hb["pump_bandwidth_resolved"] is False, N
        flagged = [str(w.message) for w in rec if issubclass(w.category, RuntimeWarning)]
        assert any("bins occupied" in m for m in flagged), (N, flagged)

    # (e) the THRESHOLD's own margins, measured on the statistic itself rather than inferred from
    #     the pass/fail outcome -- these are what would silently erode if the binning changed
    from dynameta.optics.spdc_design import (_ROTATED_MEASURE_MAX_RATIO, _cell_weights,
                                             _measure_ratio, _rotated_marginal)

    def ratio(wsg, wig):
        WS, WI = np.meshgrid(wsg, wig, indexing="ij")
        P = jsi(_sep_jsa(WS, WI))
        U, V = WS + WI, WS - WI
        s_c, s_r = 0.5 * (wsg.min() + wsg.max()), 0.5 * (wsg.max() - wsg.min())
        i_c, i_r = 0.5 * (wig.min() + wig.max()), 0.5 * (wig.max() - wig.min())
        H = min(s_r, i_r)
        mask = ((np.abs(U - (s_c + i_c)) <= H * (1 + 1e-12))
                & (np.abs(V - (s_c - i_c)) <= H * (1 + 1e-12)))
        wts = _cell_weights(wsg, wig)
        nb = max(4, max(wsg.size, wig.size))
        return max(_measure_ratio(_rotated_marginal(P, U, wts, mask, nb)[2]),
                   _measure_ratio(_rotated_marginal(P, V, wts, mask, nb)[2]))

    # worst measured UNIFORM grid (201x301) -- must stay clear of the threshold from below
    worst_pass = ratio(np.linspace(WS0 - span, WS0 + span, 201),
                       np.linspace(WI0 - span, WI0 + span, 301))
    assert worst_pass == pytest.approx(1.9804, rel=1e-3)
    assert worst_pass < _ROTATED_MEASURE_MAX_RATIO
    # least-offending measured FAILURE (log-spaced, the finest grid: the error does not shrink)
    d = np.geomspace(1.0, 11.0, 1601) - 6.0
    d = d / d.max() * span
    least_trip = ratio(WS0 + d, WI0 + d)
    assert least_trip == pytest.approx(2.2328, rel=1e-3)
    assert least_trip > _ROTATED_MEASURE_MAX_RATIO
    # ... and the threshold sits between them with >= 5 % of headroom on BOTH sides
    assert _ROTATED_MEASURE_MAX_RATIO / worst_pass > 1.05
    assert least_trip / _ROTATED_MEASURE_MAX_RATIO > 1.05


def test_unstructured_phase_matching_is_reported_nan_not_a_grid_artifact():
    """AUDIT Q-13, the auditor's own reproduction. With Phi == 1 identically -- no phase-matching
    structure AT ALL -- the reported ``phase_matching_bandwidth`` was 19.1 on a [-5, 5]^2 grid
    whose v range is [-10, 10]: the geometric tent of the grid, reported as physics. It is now
    nan with a RuntimeWarning and a False ``_resolved`` flag, because the marginal never falls to
    half its maximum inside the window. The pump width, measured on the same JSI, improves from
    -3.49 % to -0.89 % of the truth."""
    N = 201
    ws = np.linspace(-5.0, 5.0, N)
    wi = np.linspace(-5.0, 5.0, N)
    WS, WI = np.meshgrid(ws, wi, indexing="ij")
    F = np.exp(-((WS + WI) ** 2) / 2.0)              # alpha(u) = exp(-u^2/2), Phi == 1

    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        hb = heralded_bandwidths(F, ws, wi)
    assert np.isnan(hb["phase_matching_bandwidth"])
    assert hb["phase_matching_bandwidth_resolved"] is False
    assert np.isnan(hb["diagonal_bandwidth"])                    # the alias follows
    flagged = [str(w.message) for w in rec if issubclass(w.category, RuntimeWarning)]
    assert any("phase_matching_bandwidth" in m and "NOT resolved" in m for m in flagged)

    # the pump width IS resolved on the same data, and is now right to < 1 %
    assert hb["pump_bandwidth_resolved"] is True
    assert hb["pump_bandwidth"] == pytest.approx(2.0 * np.sqrt(np.log(2.0)), rel=0.01)


def test_signal_and_idler_marginals_are_untouched_by_the_q13_fix():
    """The signal / idler marginals were ALREADY Jacobian-correct (trapezoid over the other axis);
    Q-13 is about the rotated ones. Pin that they come back bit-for-bit from the same quadrature
    the module has always used."""
    from dynameta.optics.spdc_design import _fwhm, _trapz
    span, N = 5.0e12, 401
    wsg = np.linspace(WS0 - span, WS0 + span, N)
    wig = np.linspace(WI0 - span, WI0 + span, N)
    WS, WI = np.meshgrid(wsg, wig, indexing="ij")
    F = _sep_jsa(WS, WI)
    P = jsi(F)
    hb = heralded_bandwidths(F, wsg, wig)
    assert hb["signal_bandwidth"] == _fwhm(wsg, _trapz(P, wig, axis=1))
    assert hb["idler_bandwidth"] == _fwhm(wig, _trapz(P, wsg, axis=0))
    assert np.array_equal(hb["signal_marginal"], _trapz(P, wig, axis=1))
    assert np.array_equal(hb["idler_marginal"], _trapz(P, wsg, axis=0))


# ------------------------------------------------------------------ gate 8: Schmidt number
def test_schmidt_separable_matched_and_long_crystal():
    span, N = 5.0e12, 401
    wsg = np.linspace(WS0 - span, WS0 + span, N)
    wig = np.linspace(WI0 - span, WI0 + span, N)

    # (a) exactly separable JSA (outer product of two 1-D Gaussians) -> K == 1 (estimator proof)
    g1 = np.exp(-((wsg - WS0) / 1.0e12) ** 2)
    g2 = np.exp(-((wig - WI0) / 1.0e12) ** 2)
    assert abs(schmidt_number(np.outer(g1, g2))["schmidt_number"] - 1.0) < 1e-9

    # (b) physically matched sinc source (pump width tuned to the phase-matching main lobe) ->
    #     K ~ 1 (the sinc-limited purity floor ~0.82, i.e. K ~ 1.2)
    sigma_p, L = 8.0e11, 1.0e-3
    pump_fwhm = 2.0 * np.sqrt(2.0 * np.log(2.0)) * sigma_p
    slope_m = 2.0 * 1.39156 * 2.0 / (pump_fwhm * L)              # phase-match FWHM == pump FWHM
    dkf_m = lambda a, b: slope_m * ((a - b) - (WS0 - WI0))
    Km = schmidt_number(jsa(wsg, wig, _pump(sigma_p), dkf_m, L))["schmidt_number"]
    assert Km < 1.5

    # (c) long-crystal CW: broad (near-CW) pump + long crystal (narrow sinc) -> K >> 1
    dkf = lambda a, b: 5.0e-9 * ((a - b) - (WS0 - WI0))
    Kl = schmidt_number(jsa(wsg, wig, _pump(3.0e13), dkf, 2.0e-2))["schmidt_number"]
    assert Kl > 5.0


# ------------------------------------------------------------------ gate 9: QPM centre shift
def test_qpm_shifts_jsa_centre():
    # dk(ws) = beta (ws - ws0): degenerate phase matching at ws0 with NO poling. First-order
    # QPM of period Lambda moves the phase-matched point to dk = 2 pi / Lambda, i.e.
    # ws = ws0 + (2 pi / Lambda) / beta -- the twm_reference prediction.
    #
    # AUDIT Q-7: the poling square wave carries BOTH first orders m = +-1 with the same 2/pi
    # weight, so a linear dk(ws) is phase-matched at ws0 +- delta -- two bands of EQUAL height,
    # not one (verified against a raw quadrature of the poling profile in
    # tests/test_twm.py::test_qpm_both_grating_orders_vs_brute_force_square_wave).  The m = +1
    # band is the design target and is checked at its predicted position; the m = -1 mirror is
    # pinned as physics, not tolerated as an artefact.
    sigma_p, L, beta = 6.0e11, 5.0e-4, 1.0e-8
    dkf = lambda a, b: beta * (a - WS0)
    span, N = 6.0e12, 801
    wsg = np.linspace(WS0 - span, WS0 + span, N)
    wig = np.linspace(WI0 - span, WI0 + span, N)

    # no poling -> peak at degeneracy
    F0 = jsi(jsa(wsg, wig, _pump(sigma_p), dkf, L))
    i0, j0 = np.unravel_index(int(np.argmax(F0)), F0.shape)
    assert abs(wsg[i0] - WS0) < (wsg[1] - wsg[0])

    # pole to shift the phase-matched signal by delta
    delta = 2.0e12
    dk_target = beta * delta
    Lam = qpm_period_for(dk_target)                              # 2 pi / dk_target
    Fq = jsi(jsa(wsg, wig, _pump(sigma_p), dkf, L, qpm_period=Lam))
    ws_pred = WS0 + dk_target / beta                            # == WS0 + delta  (m = +1 band)
    wi_pred = WP - ws_pred
    # m = +1 band: restrict to the dk > 0 half plane (ws > WS0), where it is the only band.
    up = wsg > WS0
    Fup = np.where(up[:, None], Fq, -np.inf)
    ip, jp = np.unravel_index(int(np.argmax(Fup)), Fq.shape)
    assert abs(wsg[ip] - ws_pred) / (ws_pred - WS0) < 1e-2
    assert abs(wig[jp] - wi_pred) / abs(wi_pred - WI0) < 1e-2

    # m = -1 mirror band at ws = WS0 - delta, SAME height (equal 2/pi Fourier weight).
    Fdn = np.where((wsg < WS0)[:, None], Fq, -np.inf)
    im, jm = np.unravel_index(int(np.argmax(Fdn)), Fq.shape)
    assert abs(wsg[im] - (WS0 - dk_target / beta)) / (ws_pred - WS0) < 1e-2
    assert abs(wig[jm] - (WP - wsg[im])) <= 2.0 * (wig[1] - wig[0])
    assert abs(Fq[im, jm] - Fq[ip, jp]) <= 1e-9 * Fq[ip, jp]
