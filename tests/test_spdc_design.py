"""Gates for the SPDC design tier (roadmap 4.4) -- the Helt-Liscidini-Sipe quantum-classical
correspondence built on twm_reference. Pure numpy/scipy; runs in CI. Gates:
  (6) textbook CW bulk-crystal pair rate recovered in the uniform limit -- reconstructed
      in-test from the correspondence itself + dimensional analysis (pairs/s per W);
  (7) JSA anti-diagonal width == pump bandwidth, diagonal width == phase-matching bandwidth
      on a constructed separable case (20%);
  (8) Schmidt number == 1 for a separable/matched case, >> 1 (>5) for a long-crystal CW case;
  (9) QPM shifts the JSA centre to the twm-predicted (omega_s, omega_i) (1%).
"""
import numpy as np

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
    assert abs(hb["antidiagonal_bandwidth"] - pump_fwhm) / pump_fwhm < 0.20
    assert abs(hb["diagonal_bandwidth"] - pm_fwhm) / pm_fwhm < 0.20
    # aliases carry the physical names
    assert hb["pump_bandwidth"] == hb["antidiagonal_bandwidth"]
    assert hb["phase_matching_bandwidth"] == hb["diagonal_bandwidth"]


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
    ip, jp = np.unravel_index(int(np.argmax(Fq)), Fq.shape)
    ws_pred = WS0 + dk_target / beta                            # == WS0 + delta
    wi_pred = WP - ws_pred
    assert abs(wsg[ip] - ws_pred) / (ws_pred - WS0) < 1e-2
    assert abs(wig[jp] - wi_pred) / abs(wi_pred - WI0) < 1e-2
