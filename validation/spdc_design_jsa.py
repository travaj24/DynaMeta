"""Solver-free gates for the SPDC DESIGN tier in ``optics.spdc_design`` (JSA, Schmidt number,
heralded bandwidths, Helt-Liscidini-Sipe pair rate) -- a module the audit (X-19) flagged as having
no validation script of its own, so its only oracle was the test written beside it in the same
commit.

Pure numpy/scipy, ~2 s wall (measured 2.1 s cold, 2026-11-02).  Every expected value is a closed form derived in this file (or a JSA
built here from the textbook definition with ``np.sinc``), never a number produced by the module.

GATE A  SCHMIDT NUMBER, GAUSSIAN CLOSED FORM.  For f(x, y) = exp(-a x^2 - b y^2 - 2 c x y) the
        reduced density is Gaussian and Tr(rho^2)/Tr(rho)^2 = sqrt(1 - c^2/(a b)), so

            K = 1 / sqrt(1 - c^2 / (a b)),   lambda_k = (1 - mu) mu^k,   mu = (K - 1)/(K + 1)

        (the thermal Schmidt spectrum).  ``schmidt_number`` must reproduce K, purity = 1/K and the
        first five lambda_k for separable, correlated, anti-correlated and strongly correlated
        cases (with a linear spectral phase that must NOT change K).
GATE B  JSA FACTORIZATION + ROTATED WIDTHS.  ``jsa`` must equal alpha(ws + wi) * sinc(dk L/2) *
        exp(-i dk L/2) built here from (1/L) int_0^L exp(-i dk z) dz, unit Frobenius norm, and the
        scalar-only dk_func fallback must be bit-identical to the array path.  For a Gaussian pump
        and symmetric group velocities (dk = kappa (ws - wi)), ``heralded_bandwidths`` must return
        the pump FWHM 2 sqrt(2 ln 2) sigma_p and the phase-matching FWHM 4 x_h/(kappa L), where
        sinc^2(x_h) = 1/2 (x_h = 1.39155737825...).  On in-file Gaussian JSAs the signal / idler /
        u / v FWHMs must match 2 sqrt(ln2 / (2(a - c^2/a))), 2 sqrt(ln2/(a + c)), 2 sqrt(ln2/(a - c)).
GATE C  TRUST WARNINGS.  On every resolved uniform grid above: ZERO warnings.  A window too narrow
        to reach half maximum -> nan + RuntimeWarning + ``*_resolved`` False for all four widths;
        a log-spaced grid -> the grid-uniformity guard turns ONLY the two rotated widths into nan
        (signal / idler stay finite and resolved); a span below the carrier's float resolution ->
        the bin-collapse guard does the same.  nan <=> not resolved, exactly.
GATE D  PAIR RATE.  ``pair_rate_from_sfg`` and ``spectral_pair_rate_closed_form`` must equal the
        in-file G/(2 pi), G = kappa_s kappa_i |A_p|^2 L^2 sinc^2, kappa = w d/(n c),
        |A_p|^2 = 2 P/(A n_p eps0 c) pointwise; HELT_SIPE_PREFACTOR = 1/(2 pi).  With n(w) = n0 w/w_r
        (so kappa is constant) and dk = beta (ws - w0) the integrated rate has the EXACT closed form

            R = (kappa^2 |A_p|^2 L^2 / 2 pi) (4 / (beta L)) [Si(2X) - sin^2(X)/X],  X = beta L D / 2,

        which the trapezoid total must hit with second-order convergence; R / P_p must be the
        reported ``rate_per_watt``.

Run: python -m validation.spdc_design_jsa
"""
import math
import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

C = 299792458.0
EPS0 = 8.8541878128e-12           # CODATA 2018
X_HALF_SINC2 = 1.3915573782515103  # sinc^2(x) = 1/2, sinc(x) = sin(x)/x


def _quiet(fn, *a, **k):
    """Call fn, returning (result, [caught warnings]); the passing path is asserted warning-free."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        out = fn(*a, **k)
    return out, list(w)


def _gauss_jsa(x, y, a, b, c, x0=0.0, y0=0.0):
    X, Y = np.meshgrid(np.asarray(x) - x0, np.asarray(y) - y0, indexing="ij")
    return np.exp(-a * X ** 2 - b * Y ** 2 - 2.0 * c * X * Y)


def main():
    from scipy.special import sici
    from dynameta.optics import spdc_design as sd

    ok = True

    # ---------------------------------------------------------------- GATE A: Schmidt closed form
    print("[spdc] === GATE A: Schmidt number of Gaussian JSAs vs the closed form ===", flush=True)
    x = np.linspace(-12.0, 12.0, 401)          # box convergence measured: +-8 -> 2e-5, +-12 -> 5e-11
    a_ok, worst_k, worst_l = True, 0.0, 0.0
    for a, b, c in ((1.0, 1.0, 0.0), (1.0, 1.0, 0.6), (1.0, 1.0, -0.8), (0.5, 2.0, 0.9),
                    (1.0, 1.0, 0.96)):
        X = np.meshgrid(x, x, indexing="ij")[0]
        f = _gauss_jsa(x, x, a, b, c) * np.exp(1j * 0.7 * X)      # local phase: K invariant
        r, w = _quiet(sd.schmidt_number, f)
        K = 1.0 / math.sqrt(1.0 - c * c / (a * b))
        mu = (K - 1.0) / (K + 1.0)
        lam = (1.0 - mu) * mu ** np.arange(5)
        e_k = abs(r["schmidt_number"] / K - 1.0)
        e_p = abs(r["purity"] * K - 1.0)
        e_l = float(np.max(np.abs(r["schmidt_spectrum"][:5] - lam)))
        e_n = abs(float(np.sum(r["schmidt_spectrum"])) - 1.0)
        worst_k = max(worst_k, e_k, e_p, e_n)
        worst_l = max(worst_l, e_l)
        a_ok = a_ok and not w
        print("[spdc]   a={} b={} c={:+.2f}: K = {:.12f} vs {:.12f} (rel {:.1e}); purity rel {:.1e};"
              " lambda_0..4 max abs {:.1e}".format(a, b, c, r["schmidt_number"], K, e_k, e_p, e_l),
              flush=True)
    a_ok = a_ok and worst_k < 1e-9 and worst_l < 1e-9          # measured 5.4e-11 / 1.2e-11
    print("[spdc] GATE A (worst K/purity {:.1e} < 1e-9, worst lambda_k {:.1e} < 1e-9) -> {}".format(
        worst_k, worst_l, "PASS" if a_ok else "FAIL"), flush=True)
    ok = ok and a_ok

    # ---------------------------------------------------------------- GATE B: JSA + widths
    print("[spdc] === GATE B: JSA factorization and closed-form heralded widths ===", flush=True)
    w0, span, n = 1.2e15, 2.0e13, 401
    ws = w0 + np.linspace(-span, span, n)
    wi = w0 + np.linspace(-span, span, n)
    sig_p, L, kap = 4.0e12, 5.0e-3, 2.0e-10          # pump |alpha|^2 std [rad/s]; GVM slope [s/m]

    def alpha(u):
        return np.exp(-(u - 2.0 * w0) ** 2 / (4.0 * sig_p ** 2))   # |alpha|^2: std sig_p

    def dk_arr(a_, b_):
        return kap * (a_ - b_)

    def dk_scalar(a_, b_):                       # scalar-only: math.fsum raises on arrays
        return math.fsum([kap * a_, -kap * b_])

    f, w1 = _quiet(sd.jsa, ws, wi, alpha, dk_arr, L)
    f_s, w2 = _quiet(sd.jsa, ws, wi, alpha, dk_scalar, L)
    WS, WI = np.meshgrid(ws, wi, indexing="ij")
    xx = kap * (WS - WI) * L / 2.0
    ref = alpha(WS + WI) * np.sinc(xx / np.pi) * np.exp(-1j * xx)  # (1/L) int_0^L e^{-i dk z} dz
    ref = ref / np.sqrt(np.sum(np.abs(ref) ** 2))
    e_f = float(np.max(np.abs(f - ref)))
    e_nrm = abs(float(np.sum(np.abs(f) ** 2)) - 1.0)
    e_sc = float(np.max(np.abs(f_s - f)))
    jsi_ok = bool(np.array_equal(sd.jsi(f), np.abs(f) ** 2))
    hb, w3 = _quiet(sd.heralded_bandwidths, f, ws, wi)
    pump_th = 2.0 * math.sqrt(2.0 * math.log(2.0)) * sig_p
    pm_th = 4.0 * X_HALF_SINC2 / (kap * L)
    e_pu = abs(hb["pump_bandwidth"] / pump_th - 1.0)
    e_pm = abs(hb["phase_matching_bandwidth"] / pm_th - 1.0)
    print("[spdc]   jsa vs in-file alpha*sinc*phase: max |diff| {:.1e}; |f|_F^2 - 1 = {:.1e}; "
          "scalar-dk fallback vs array path {:.1e}; jsi == |f|^2: {}".format(
              e_f, e_nrm, e_sc, jsi_ok), flush=True)
    print("[spdc]   Gaussian pump x sinc PM: pump FWHM {:.6e} vs {:.6e} (rel {:.1e}); PM FWHM "
          "{:.6e} vs 4 x_h/(kappa L) {:.6e} (rel {:.1e})".format(
              hb["pump_bandwidth"], pump_th, e_pu, hb["phase_matching_bandwidth"], pm_th, e_pm),
          flush=True)
    b_ok = (not (w1 or w2 or w3) and e_f < 1e-15 and e_nrm < 1e-12 and e_sc == 0.0 and jsi_ok
            and e_pu < 2e-4 and e_pm < 1e-4)                 # measured 6.1e-5 / 1.3e-5
    worst_g = 0.0
    ln2 = math.log(2.0)
    for a, c, half in ((1.0, 0.6, 8.0), (1.0, -0.5, 8.0), (1.0, 0.0, 6.0)):
        g = 50.0 + np.linspace(-half, half, 401)               # offset carrier, uniform grid
        fg = _gauss_jsa(g, g, a, a, c, 50.0, 50.0)
        hg, w = _quiet(sd.heralded_bandwidths, fg, g, g)
        b_ok = b_ok and not w
        s_th = 2.0 * math.sqrt(ln2 / (2.0 * (a - c * c / a)))
        u_th = 2.0 * math.sqrt(ln2 / (a + c))
        v_th = 2.0 * math.sqrt(ln2 / (a - c))
        errs = (hg["signal_bandwidth"] / s_th - 1.0, hg["idler_bandwidth"] / s_th - 1.0,
                hg["pump_bandwidth"] / u_th - 1.0, hg["phase_matching_bandwidth"] / v_th - 1.0)
        worst_g = max(worst_g, max(abs(e_) for e_ in errs))
        print("[spdc]   Gaussian a={} c={:+.1f}: signal/idler/u/v FWHM rel err {:+.1e} {:+.1e} "
              "{:+.1e} {:+.1e}".format(a, c, *errs), flush=True)
    b_ok = b_ok and worst_g < 4e-4                              # measured worst 1.8e-4
    print("[spdc] GATE B (worst Gaussian-width rel {:.1e} < 4e-4) -> {}".format(
        worst_g, "PASS" if b_ok else "FAIL"), flush=True)
    ok = ok and b_ok

    # ---------------------------------------------------------------- GATE C: trust warnings
    print("[spdc] === GATE C: heralded-bandwidth trust warnings fire only on bad grids ===",
          flush=True)
    keys = ("signal_bandwidth", "idler_bandwidth", "pump_bandwidth", "phase_matching_bandwidth")

    def _nan_iff_unresolved(h):
        return all(math.isnan(h[k]) == (not h[k + "_resolved"]) for k in keys)

    c_ok = True
    # (1) window narrower than the half-maximum: all four widths are window-limited
    g = 50.0 + np.linspace(-1.0, 1.0, 201)
    h1, w = _quiet(sd.heralded_bandwidths, _gauss_jsa(g, g, 0.2, 0.2, 0.0, 50.0, 50.0), g, g)
    n_rt = sum(1 for q in w if issubclass(q.category, RuntimeWarning))
    ok1 = (n_rt == 4 and len(w) == 4 and not any(h1[k + "_resolved"] for k in keys)
           and _nan_iff_unresolved(h1))
    print("[spdc]   under-resolved window: {} RuntimeWarning(s), resolved flags {} -> {}".format(
        n_rt, [h1[k + "_resolved"] for k in keys], "ok" if ok1 else "WRONG"), flush=True)
    # (2) log-spaced axes: the rotated legs are refused, the trapezoid marginals are not
    g = 50.0 + np.geomspace(1.0, 17.0, 401) - 9.0
    h2, w = _quiet(sd.heralded_bandwidths, _gauss_jsa(g, g, 1.0, 1.0, 0.6, 50.0, 50.0), g, g)
    s_th = 2.0 * math.sqrt(ln2 / (2.0 * (1.0 - 0.36)))
    e_sig_log = abs(h2["signal_bandwidth"] / s_th - 1.0)
    ok2 = (len(w) == 2 and all("NOT trustworthy" in str(q.message) for q in w)
           and math.isnan(h2["pump_bandwidth"]) and math.isnan(h2["phase_matching_bandwidth"])
           and h2["signal_bandwidth_resolved"] and h2["idler_bandwidth_resolved"]
           and _nan_iff_unresolved(h2) and e_sig_log < 2e-3)    # measured 4.3e-4 (trapezoid)
    print("[spdc]   log-spaced grid: {} grid-guard warning(s); rotated widths nan: {}; signal "
          "width still {:.4f} vs closed form {:.4f} (rel {:.1e}) -> {}".format(
              len(w), math.isnan(h2["pump_bandwidth"]), h2["signal_bandwidth"], s_th, e_sig_log,
              "ok" if ok2 else "WRONG"), flush=True)
    # (3) span below the carrier's float resolution: omega_s + omega_i aliases onto a few values
    g = 1.2e15 + np.linspace(-2.5, 2.5, 400)
    h3, w = _quiet(sd.heralded_bandwidths, _gauss_jsa(g, g, 1.0, 1.0, 0.0, 1.2e15, 1.2e15), g, g)
    ok3 = (len(w) == 2 and all("NOT trustworthy" in str(q.message) for q in w)
           and math.isnan(h3["pump_bandwidth"]) and math.isnan(h3["phase_matching_bandwidth"])
           and _nan_iff_unresolved(h3))
    print("[spdc]   5 rad/s span on a 1.2e15 carrier ({} distinct axis values of 400): {} "
          "warning(s), rotated widths nan -> {}".format(np.unique(g).size, len(w),
                                                        "ok" if ok3 else "WRONG"), flush=True)
    # (4) the resolved uniform grids of GATE B emitted nothing (asserted there); restate it here
    ok4 = not w3 and hb["pump_bandwidth_resolved"] and hb["phase_matching_bandwidth_resolved"] \
        and _nan_iff_unresolved(hb)
    print("[spdc]   resolved uniform grids (GATE B): 0 warnings, all flags True -> {}".format(
        "ok" if ok4 else "WRONG"), flush=True)
    c_ok = ok1 and ok2 and ok3 and ok4
    print("[spdc] GATE C -> {}".format("PASS" if c_ok else "FAIL"), flush=True)
    ok = ok and c_ok

    # ---------------------------------------------------------------- GATE D: pair rate
    print("[spdc] === GATE D: Helt-Liscidini-Sipe pair rate vs closed forms ===", flush=True)
    wr, n0, d_eff, area, p_w, beta = 1.2e15, 2.2, 14e-12, (10e-6) ** 2, 0.05, 1.0e-9
    wp = 2.0 * wr

    def n_lin(w_):                               # n ~ w makes kappa = w d/(n c) frequency-flat
        return n0 * w_ / wr

    def dk_lin(a_, b_):
        return beta * (a_ - b_) / 2.0            # = beta (ws - w0), w0 = wp/2

    d_ok = abs(sd.HELT_SIPE_PREFACTOR * 2.0 * math.pi - 1.0) < 1e-15
    kap_c = wr * d_eff / (n0 * C)
    ap2 = 2.0 * p_w / (area * n0 * EPS0 * C)
    D = 2.0e13
    rates, worst_pt = {}, 0.0
    for L_ in (1e-2, 2e-2):
        X_ = beta * L_ * D / 2.0
        R_th = (kap_c ** 2 * ap2 * L_ ** 2 / (2.0 * math.pi)) * (4.0 / (beta * L_)) * (
            sici(2.0 * X_)[0] - math.sin(X_) ** 2 / X_)
        for npts in (4001, 8001):
            om = wr + np.linspace(-D, D, npts)
            r, w = _quiet(sd.pair_rate_from_sfg, om, wp, d_eff, L_, n_s=n_lin, n_i=n_lin,
                          n_p=n0, pump_power_W=p_w, area_m2=area, dk_func=dk_lin)
            oi = wp - om
            xx = dk_lin(om, oi) * L_ / 2.0
            pw = ((om * d_eff / (n_lin(om) * C)) * (oi * d_eff / (n_lin(oi) * C)) * ap2 * L_ ** 2
                  * np.sinc(xx / np.pi) ** 2 / (2.0 * math.pi))
            cf, w_cf = _quiet(sd.spectral_pair_rate_closed_form, om, wp, d_eff, L_, n_s=n_lin,
                              n_i=n_lin, n_p=n0, pump_power_W=p_w, area_m2=area,
                              dk=dk_lin(om, oi))
            e_pt = max(float(np.max(np.abs(r["spectral_density"] / pw - 1.0))),
                       float(np.max(np.abs(cf / pw - 1.0))))
            worst_pt = max(worst_pt, e_pt)
            rates[(L_, npts)] = (r["rate_pairs_per_s"] / R_th - 1.0)
            d_ok = (d_ok and not w and not w_cf
                    and abs(r["rate_per_watt"] * p_w / r["rate_pairs_per_s"] - 1.0) < 1e-15)
            print("[spdc]   L={:.0f} mm N={}: R = {:.10e} vs Si closed form {:.10e} (rel {:+.1e}); "
                  "pointwise dR/dw vs in-file G/2pi {:.1e}".format(
                      L_ * 1e3, npts, r["rate_pairs_per_s"], R_th, rates[(L_, npts)], e_pt),
                  flush=True)
    conv = [rates[(L_, 4001)] / rates[(L_, 8001)] for L_ in (1e-2, 2e-2)]
    d_ok = (d_ok and worst_pt < 1e-13 and all(abs(v) < 2e-8 for v in rates.values())
            and all(3.5 < q < 4.5 for q in conv))      # measured 1.2e-8 / 2.9e-9; ratio 4.0
    print("[spdc]   trapezoid error halving the step: x{:.2f} / x{:.2f} (second order -> 4)".format(
        *conv), flush=True)
    print("[spdc] GATE D (pointwise {:.1e} < 1e-13; integrated worst {:.1e} < 2e-8) -> {}".format(
        worst_pt, max(abs(v) for v in rates.values()), "PASS" if d_ok else "FAIL"), flush=True)
    ok = ok and d_ok

    print("[spdc] *** SPDC DESIGN (JSA / SCHMIDT / HERALDED WIDTHS / PAIR RATE): {} ***".format(
        "PASS" if ok else "FAIL"), flush=True)
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
