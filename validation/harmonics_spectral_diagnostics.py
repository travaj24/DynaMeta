"""Solver-free gates for the SH/TH spectral DIAGNOSTICS in ``optics.harmonics`` -- a module the audit
(X-19) flagged as having no validation script of its own, so its only oracle was the test written
beside it in the same commit.

Pure numpy, ~1 s wall (measured 0.96 s cold, 2026-11-02).  No FDTD run is involved: every input is a SYNTHETIC exit-plane time trace built
here from Gaussian-envelope cosines whose band energies are known in closed form, so nothing is
checked against the implementation that produced it.  The closed forms used throughout:

  * a component A g(t) cos(2 pi f t + phi), g = exp(-(t - tc)^2 / (2 tau^2)), carries the energy
    integral (A g cos)^2 dt = A^2 tau sqrt(pi) / 2 (the 2f term is negligible for f tau >> 1);
  * discrete Parseval on N samples at step dt: a narrowband positive-frequency lobe holds
    sum_band |rfft|^2 = (N / 2) sum_j x_j^2 = N A^2 tau sqrt(pi) / (4 dt);
  * its power spectrum is a Gaussian of std sigma_f = 1 / (2 sqrt(2) pi tau), so the fraction of
    it inside [f - hw, f + hw] is erf(hw / (sqrt(2) sigma_f)).

GATE A  BAND RECOVERY.  ``harmonic_spectrum`` on a w + 2w + 3w trace (amplitudes 1 / 0.3 / 0.05,
        envelope tau/sqrt(n) as the n-fold product g^n gives) must return each band power at the
        Parseval closed form and P_nw/P_w = (A_n/A_1)^2 / sqrt(n); ``mixing_spectrum`` on an
        eight-line two-colour trace (f1, f2, f1+f2, |f1-f2|, 2f1, 2f2, 2f1-f2, 2f2-f1) must return
        every band at its closed form, with no degeneracy and no leakage flag.
GATE B  POYNTING PATH + NORMALIZATIONS.  With an H_x probe recorded HALF A STEP EARLY (the Yee
        stagger, H = -E(t - dt/2)/eta) the Poynting band powers must equal the |E|^2 ones / eta
        (the D-2 half-step correction cancels the stagger exactly); with H CO-LOCATED the residual
        factor must be the closed form cos(pi f dt) -- which pins both the sign and the size of
        the correction.  ``conversion_efficiency`` (incident and transmitted normalizations) and
        ``undepleted_validity`` must reproduce the closed-form ratios of the chosen amplitudes.
GATE C  ORDERS + SLOPES.  On a periodic x-line, a physical exp(+i G x) wave (exp(-i w t)
        convention) must be filed under order +1 (audit D-4), a 4:1 blazed mix must report
        P(+1)/P(-1) = 16, and the order sum must equal the x-summed Parseval energy.
        ``power_slope`` over a pump sweep with A_2 ~ A_1^2, A_3 ~ A_1^3 must return slopes 2 / 3.
GATE D  PUMP-BANDWIDTH GUARD.  The measured pump std must match sigma_f = 1/(2 sqrt(2) pi tau);
        a pure-fundamental pulse's PHANTOM P_2w/P_w must match the Gaussian-tail closed form
        [Phi(1.3/s) - Phi(0.7/s)] / [2 Phi(0.15/s) - 1] (s = sigma_f/f0, bw = 0.15) across five
        bandwidths; the UserWarning + ``pump_broadband`` flag must fire exactly when s exceeds the
        documented bound 0.21 (1 - 2 bw) = 0.147, and stay silent on the narrowband traces.

Run: python -m validation.harmonics_spectral_diagnostics
"""
import math
import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

F0 = 2.0e14                       # pump carrier [Hz]
DT = 1.0 / (40.0 * F0)            # 40 samples per pump cycle -> Nyquist 20 f0
NS = 2 ** 15                      # record 4.1 ps
T = np.arange(NS) * DT
TC = 0.5 * NS * DT
TAU = 50e-15                      # sigma_f / f0 = 0.011: deep inside the narrowband window
ETA0 = 376.730313668              # vacuum impedance [ohm]


# ------------------------------------------------------------------ in-file oracle (no dynameta)
def _line(amp, f_hz, tau, phase=0.0, shift=0.0):
    """A g(t - shift) cos(2 pi f (t - shift - tc) + phase): one Gaussian-envelope spectral line."""
    tt = T - shift
    return amp * np.exp(-(tt - TC) ** 2 / (2.0 * tau * tau)) * np.cos(
        2.0 * np.pi * f_hz * (tt - TC) + phase)


def _parseval_band(amp, tau, f_hz=None, half_width=None):
    """Closed-form one-sided band sum of |rfft|^2 for one line: N A^2 tau sqrt(pi) / (4 dt), times
    the Gaussian capture fraction erf(hw / (sqrt(2) sigma_f)) when a band half-width is given."""
    p = NS * amp * amp * tau * math.sqrt(math.pi) / (4.0 * DT)
    if half_width is not None:
        sig_f = 1.0 / (2.0 * math.sqrt(2.0) * math.pi * tau)
        p *= math.erf(half_width / (math.sqrt(2.0) * sig_f))
    return p


def _phi(x):
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


def _quiet(fn, *a, **k):
    """Call fn, returning (result, [caught warnings]) -- the passing path must be warning-clean
    under a warnings-as-errors wrapper, so every call is recorded and the count asserted."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        out = fn(*a, **k)
    return out, list(w)


def main():
    from dynameta.optics import harmonics as hm

    ok = True

    # ---------------------------------------------------------------- GATE A: band recovery
    print("[hsd] === GATE A: band powers vs the Parseval closed form ===", flush=True)
    amps = {1: 1.0, 2: 0.3, 3: 0.05}
    e = sum(_line(a, n * F0, TAU / math.sqrt(n), 0.3 * n) for n, a in amps.items())
    hs, w = _quiet(hm.harmonic_spectrum, e, F0, dt=DT)
    a_ok = not w and not hs["pump_broadband"] and hs["power_type"] == "e2_density"
    worst = 0.0
    for n, a in amps.items():
        tau_n = TAU / math.sqrt(n)
        e_abs = abs(hs["power"][n] / _parseval_band(a, tau_n) - 1.0)
        e_rat = abs((hs["power"][n] / hs["P_w"]) / ((a / amps[1]) ** 2 / math.sqrt(n)) - 1.0)
        worst = max(worst, e_abs, e_rat)
        print("[hsd]   n={}: P = {:.9e} vs closed form {:.9e} (rel {:.1e}); P_n/P_1 rel err "
              "{:.1e}".format(n, hs["power"][n], _parseval_band(a, tau_n), e_abs, e_rat), flush=True)
    a_ok = a_ok and worst < 1e-12                       # measured 3.2e-14 (3w, smallest band)
    # two-colour mixing: eight distinct lines, longer envelope so every bw = 0.05 band captures
    # its line to < 1e-9 (the |f1 - f2| band is only +-3.5 THz wide); the closed form still carries
    # the erf capture fraction so the oracle is exact, not "approximately one".
    f1, f2, tau_m = 2.0e14, 2.7e14, 200e-15
    mix_amp = {(1, 0): 1.0, (0, 1): 0.7, (1, 1): 0.05, (1, -1): 0.03, (2, 0): 0.02,
               (0, 2): 0.015, (2, -1): 0.01, (-1, 2): 0.008}
    em = sum(_line(a, abs(p * f1 + q * f2), tau_m, 0.1 * p) for (p, q), a in mix_amp.items())
    ms, w = _quiet(hm.mixing_spectrum, em, f1, f2, dt=DT)
    m_ok = not w and not ms["pump_broadband"] and not ms["degenerate_groups"]
    worst_m = 0.0
    for lab, (p, q) in ms["coeffs"].items():
        fc = abs(p * f1 + q * f2)
        want = _parseval_band(mix_amp[(p, q)], tau_m, fc, ms["bandwidth_frac"] * fc)
        err = abs(ms["power"][lab] / want - 1.0)
        worst_m = max(worst_m, err)
        print("[hsd]   mixing {:>7s} @ {:.2f} THz: P/closed form - 1 = {:+.1e}".format(
            lab, fc / 1e12, ms["power"][lab] / want - 1.0), flush=True)
    m_ok = m_ok and worst_m < 1e-9                      # measured worst 1.9e-10 (f1 - f2 band)
    print("[hsd] GATE A (harmonics worst {:.1e} < 1e-12; mixing worst {:.1e} < 1e-9; warnings "
          "{}) -> {}".format(worst, worst_m, len(w), "PASS" if a_ok and m_ok else "FAIL"),
          flush=True)
    ok = ok and a_ok and m_ok

    # ---------------------------------------------------------------- GATE B: Poynting + norms
    print("[hsd] === GATE B: Poynting path (Yee half-step) and efficiency normalizations ===",
          flush=True)
    h_yee = -sum(_line(a, n * F0, TAU / math.sqrt(n), 0.3 * n, shift=0.5 * DT)
                 for n, a in amps.items()) / ETA0
    hs_s, w1 = _quiet(hm.harmonic_spectrum, (e, h_yee, DT), F0)
    hs_c, w2 = _quiet(hm.harmonic_spectrum, (e, -e / ETA0, DT), F0)
    b_ok = not w1 and not w2 and hs_s["power_type"] == "poynting_flux"
    worst_s, worst_c = 0.0, 0.0
    for n in amps:
        r_s = hs_s["power"][n] * ETA0 / hs["power"][n]
        r_c = hs_c["power"][n] * ETA0 / hs["power"][n]
        cos_n = math.cos(math.pi * n * F0 * DT)
        worst_s = max(worst_s, abs(r_s - 1.0))
        worst_c = max(worst_c, abs(r_c / cos_n - 1.0))
        print("[hsd]   n={}: staggered-H S*eta/|E|^2 = {:.15f}; co-located-H ratio {:.7f} vs "
              "cos(pi n f0 dt) = {:.7f}".format(n, r_s, r_c, cos_n), flush=True)
    # measured: staggered 9.1e-15; co-located vs cos(pi f dt) 1.2e-6 (the cosine varies across the
    # band, the closed form is its value at the line centre). A WRONG-sign correction would show
    # cos(2 pi f dt) on the staggered probe instead, a 1.1e-1 miss at 3w (f dt = 0.075).
    b_ok = b_ok and worst_s < 1e-12 and worst_c < 1e-5
    inc_amp, tr = 1.5, {1: 1.0, 2: 0.2, 3: 0.05}
    trace = {"dt": DT,
             "transmitted": sum(_line(a, n * F0, TAU / math.sqrt(n)) for n, a in tr.items()),
             "incident_right": _line(inc_amp, F0, TAU)}
    ce, w3 = _quiet(hm.conversion_efficiency, trace, F0)
    ct, w4 = _quiet(hm.conversion_efficiency, trace, F0, normalization="transmitted")
    uv, w5 = _quiet(hm.undepleted_validity, trace, F0)
    e_inc = inc_amp ** 2 * TAU                          # energy ~ A^2 tau (common factors cancel)
    shg = tr[2] ** 2 * TAU / math.sqrt(2.0)
    thg = tr[3] ** 2 * TAU / math.sqrt(3.0)
    e_tr = tr[1] ** 2 * TAU
    errs = [abs(ce["eta_shg"] / (shg / e_inc) - 1.0), abs(ce["eta_thg"] / (thg / e_inc) - 1.0),
            abs(ct["eta_shg"] / (shg / e_tr) - 1.0),
            abs(uv["eta_total"] / ((shg + thg) / e_inc) - 1.0),
            abs(uv["depletion_measured"] - (1.0 - e_tr / e_inc))]
    b_ok = (b_ok and not (w3 or w4 or w5) and ce["normalization"] == "incident"
            and ct["normalization"] == "transmitted" and uv["undepleted"] and max(errs) < 1e-12)
    print("[hsd]   eta_shg {:.6e} / eta_thg {:.6e} (incident), eta_shg {:.6e} (transmitted), "
          "eta_total {:.6e}, depletion_measured {:.6f} vs 1 - (1/1.5)^2 = {:.6f}; worst rel "
          "{:.1e}".format(ce["eta_shg"], ce["eta_thg"], ct["eta_shg"], uv["eta_total"],
                          uv["depletion_measured"], 1.0 - e_tr / e_inc, max(errs)), flush=True)
    print("[hsd] GATE B (staggered {:.1e} < 1e-12, co-located-vs-cos {:.1e} < 1e-5, efficiencies "
          "{:.1e} < 1e-12) -> {}".format(worst_s, worst_c, max(errs), "PASS" if b_ok else "FAIL"),
          flush=True)
    ok = ok and b_ok

    # ---------------------------------------------------------------- GATE C: orders + slopes
    print("[hsd] === GATE C: diffraction-order labels and log-log power slopes ===", flush=True)
    nx = 8
    x = np.arange(nx) / nx                               # one period, units of Lambda
    env = np.exp(-(T - TC) ** 2 / (2.0 * TAU * TAU))[:, None]
    ph = 2.0 * np.pi * F0 * (T - TC)[:, None]
    # Re[a exp(i(+G x) - i w t)] = a cos(w t - G x): physical order +1 under exp(-i w t)
    eline = env * (4.0 * np.cos(ph - 2.0 * np.pi * x[None, :])
                   + 1.0 * np.cos(ph + 2.0 * np.pi * x[None, :]))
    ho, w = _quiet(hm.harmonic_spectrum, eline, F0, dt=DT)
    by = dict(zip(ho["orders"].tolist(), ho["power_by_order"][1].tolist()))
    p_plus, p_minus = by[1], by[-1]
    rest = sum(v for m, v in by.items() if m not in (1, -1))
    blaze = p_plus / p_minus
    e_plus = abs(p_plus / (nx * _parseval_band(4.0, TAU)) - 1.0)
    e_sum = abs(sum(by.values()) / (nx * (_parseval_band(4.0, TAU) + _parseval_band(1.0, TAU)))
                - 1.0)
    c_ok = (not w and abs(blaze / 16.0 - 1.0) < 1e-12 and e_plus < 1e-12 and e_sum < 1e-12
            and rest < 1e-12 * p_plus)
    print("[hsd]   blazed 4:1 mix: P(+1)/P(-1) = {:.12f} (closed form 16); P(+1) rel {:.1e}; "
          "order sum vs x-summed Parseval rel {:.1e}; other orders {:.1e} of P(+1)".format(
              blaze, e_plus, e_sum, rest / p_plus), flush=True)
    p1, p2, p3 = [], [], []
    for a1 in (0.25, 0.5, 1.0, 2.0, 4.0):
        ee = (_line(a1, F0, TAU) + _line(0.01 * a1 ** 2, 2 * F0, TAU / math.sqrt(2.0))
              + _line(1e-3 * a1 ** 3, 3 * F0, TAU / math.sqrt(3.0)))
        r, w = _quiet(hm.harmonic_spectrum, ee, F0, dt=DT)
        c_ok = c_ok and not w
        p1.append(r["P_w"])
        p2.append(r["P_2w"])
        p3.append(r["P_3w"])
    s2 = hm.power_slope(p1, p2)
    s3 = hm.power_slope(p1, p3)
    c_ok = (c_ok and abs(s2["slope"] - 2.0) < 1e-9 and abs(s3["slope"] - 3.0) < 1e-9
            and s2["r2"] > 1.0 - 1e-12 and s3["r2"] > 1.0 - 1e-12)
    print("[hsd]   pump sweep x16: slope(P_2w) = {:.12f} (r2 {:.15f}), slope(P_3w) = {:.12f} "
          "(r2 {:.15f})".format(s2["slope"], s2["r2"], s3["slope"], s3["r2"]), flush=True)
    print("[hsd] GATE C -> {}".format("PASS" if c_ok else "FAIL"), flush=True)
    ok = ok and c_ok

    # ---------------------------------------------------------------- GATE D: pump guard
    print("[hsd] === GATE D: pump-bandwidth guard and the phantom-2w closed form ===", flush=True)
    d_ok = True
    bound = 0.21 * (1.0 - 2.0 * 0.15)                    # documented bound, 0.147 f0 at bw = 0.15
    sig_th = 1.0 / (2.0 * math.sqrt(2.0) * math.pi * TAU)
    r0, w = _quiet(hm.harmonic_spectrum, _line(1.0, F0, TAU), F0, dt=DT)
    e_sig = abs(r0["pump_sigma_hz"] / sig_th - 1.0)
    d_ok = d_ok and not w and not r0["pump_broadband"] and e_sig < 1e-9 and \
        r0["P_2w"] / r0["P_w"] < 1e-20
    print("[hsd]   tau = 50 fs: pump_sigma {:.6e} Hz vs 1/(2 sqrt2 pi tau) {:.6e} (rel {:.1e}); "
          "phantom P_2w/P_w {:.1e}; warnings {}".format(r0["pump_sigma_hz"], sig_th, e_sig,
                                                        r0["P_2w"] / r0["P_w"], len(w)), flush=True)
    worst_ph = 0.0
    for s in (0.106, 0.140, 0.159, 0.208, 0.25):
        tau_b = 1.0 / (2.0 * math.sqrt(2.0) * math.pi * s * F0)
        r, w = _quiet(hm.harmonic_spectrum, _line(1.0, F0, tau_b), F0, dt=DT)
        ph_th = (_phi(1.3 / s) - _phi(0.7 / s)) / (2.0 * _phi(0.15 / s) - 1.0)
        ph = r["P_2w"] / r["P_w"]
        err = abs(ph / ph_th - 1.0)
        worst_ph = max(worst_ph, err)
        # the guard measures sigma over [0.5, 1.5] f0, which truncates a broad Gaussian (0.220
        # measured at s = 0.25), so the flag is checked against the MEASURED sigma it reports AND
        # against the true s on cases far from the bound
        n_warn = sum(1 for q in w if issubclass(q.category, UserWarning)
                     and "BROADBAND" in str(q.message))
        flag_ok = (r["pump_broadband"] == (r["pump_sigma_hz"] > bound * F0)
                   and r["pump_broadband"] == (s > bound) and n_warn == int(r["pump_broadband"])
                   and len(w) == n_warn)
        d_ok = d_ok and flag_ok and err < 2.5e-2
        print("[hsd]   s = {:.3f}: phantom P_2w/P_w {:.4e} vs Gaussian-tail closed form {:.4e} "
              "(rel {:.1e}); measured sigma {:.4f} f0, broadband={} warnings={} -> {}".format(
                  s, ph, ph_th, err, r["pump_sigma_hz"] / F0, r["pump_broadband"], n_warn,
                  "ok" if flag_ok else "WRONG"), flush=True)
    # measured worst 1.0e-2 (s = 0.106, where the 2w band sits 6.6 sigma out and the bin-edge
    # quantization df = 1.2e-3 f0 moves the tail integral by ~1 %); 2.5e-2 is that x 2.5
    print("[hsd] GATE D (sigma rel {:.1e} < 1e-9; phantom worst rel {:.1e} < 2.5e-2; flag + "
          "warning exactly when s > {:.3f}) -> {}".format(e_sig, worst_ph, bound,
                                                         "PASS" if d_ok else "FAIL"), flush=True)
    ok = ok and d_ok

    print("[hsd] *** HARMONIC SPECTRAL DIAGNOSTICS: {} ***".format("PASS" if ok else "FAIL"),
          flush=True)
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
