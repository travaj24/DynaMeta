"""QD-SOA RIN spectrum + field-autocorrelation linewidth observables vs analytic oracles
(optics.soa.noise_metrics). These are post-processing READOUTS on a time-domain trace -- e.g. the
Langevin output of amplify_coherent(langevin=True) -- adding NO physics to the marcher; they turn its
stochastic output into the standard analog-link noise figures.

GATE A (RIN Parseval): integral_0^fNyq RIN(f) df == var(P)/<P>^2 for a noisy power trace (the
        one-sided PSD normalization is exact).
GATE B (RIN sinusoid): a power modulation P = <P>(1 + m cos 2pi fm t) shows a RIN line at fm whose
        total integral equals m^2/2 (the fractional-intensity variance of a sinusoid).
GATE C (linewidth recovery): a synthetic Wiener phase walk (per-step variance v over dt) is a
        Lorentzian line of width Delta_nu = v/(2 pi dt) -- linewidth_from_field recovers it; a pure
        tone gives Delta_nu -> 0.
GATE D (Schawlow-Townes-Henry): schawlow_townes_henry_linewidth scales EXACTLY as (1 + alpha^2) over
        the bare R_sp/(4 pi N_ph), and reduces to Schawlow-Townes at alpha = 0.
GATE E (marcher Henry DIRECTION): the SOA's own Langevin output is broader in linewidth with a nonzero
        alpha than with alpha = 0 -- the amplitude->phase coupling the fs-step marcher captures (the
        direction of the Henry effect; the gain-clamped cavity narrowing is the laser oracle of Gate D,
        not the single-pass amplifier). Finite, non-negative.

Run: python -m validation.qd_soa_rin_linewidth
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dynameta.optics.soa import (QDGainModel, QDGainParams, TravelingWaveSOA, henry_factor,
                                 linewidth_from_field, rin_spectrum,
                                 schawlow_townes_henry_linewidth)


def main():
    print("[rl] === QD-SOA RIN spectrum + linewidth observables vs oracles ===", flush=True)
    ok = True
    rng = np.random.default_rng(0)
    N, dt = 200000, 1.0e-12

    # ---- GATE A: RIN Parseval ----
    P = 1.0e-3 * (1.0 + 0.05 * rng.standard_normal(N))
    f, rin = rin_spectrum(P, dt)
    # NB the residual here is np.trapezoid endpoint quadrature (~1e-2 at small N), NOT a normalization
    # error -- a rectangle sum np.sum(rin)*(f[1]-f[0]) is Parseval-exact to machine precision.
    relA = abs(np.trapezoid(rin, f) / (np.var(P) / P.mean() ** 2) - 1.0)
    g_a = bool(relA < 1e-3)
    ok = ok and g_a
    print("[rl] GATE A: integral RIN df == var(P)/<P>^2 (rel {:.1e}) -> {}".format(
        relA, "PASS" if g_a else "FAIL"), flush=True)

    # ---- GATE B: RIN sinusoid ----
    fm, m = 2.0e9, 0.10
    t = np.arange(N) * dt
    Ps = 1.0e-3 * (1.0 + m * np.cos(2.0 * np.pi * fm * t))
    f2, rin2 = rin_spectrum(Ps, dt)
    peak_f = f2[np.argmax(rin2)]
    tot = np.trapezoid(rin2, f2)
    g_b = bool(abs(peak_f - fm) < 2.0 * (f2[1] - f2[0])
               and abs(tot - m * m / 2.0) / (m * m / 2.0) < 1e-3)
    ok = ok and g_b
    print("[rl] GATE B: sinusoid RIN line at {:.2f} GHz, integral {:.4e} == m^2/2 {:.4e} -> {}".format(
        peak_f / 1e9, tot, m * m / 2.0, "PASS" if g_b else "FAIL"), flush=True)

    # ---- GATE C: linewidth recovery ----
    v = 1.0e-3
    phi = np.cumsum(np.sqrt(v) * rng.standard_normal(N))
    dnu_meas = linewidth_from_field(np.exp(1j * phi), dt)
    dnu_true = v / (2.0 * np.pi * dt)
    relC = abs(dnu_meas - dnu_true) / dnu_true
    dnu_tone = linewidth_from_field(np.exp(1j * 2.0 * np.pi * 1e9 * t), dt)
    g_c = bool(relC < 0.05 and dnu_tone / dnu_true < 1e-6)
    ok = ok and g_c
    print("[rl] GATE C: phase-walk linewidth {:.3e} == v/(2pi dt) {:.3e} (rel {:.3f}); tone -> {:.1e} "
          "-> {}".format(dnu_meas, dnu_true, relC, dnu_tone, "PASS" if g_c else "FAIL"), flush=True)

    # ---- GATE D: Schawlow-Townes-Henry (1+alpha^2) ----
    R_sp, Nph = 1.0e12, 1.0e4
    st0 = schawlow_townes_henry_linewidth(R_sp, Nph, 0.0)
    st3 = schawlow_townes_henry_linewidth(R_sp, Nph, 3.0)
    bare = R_sp / (4.0 * np.pi * Nph)
    relD = max(abs(st0 - bare) / bare, abs(st3 / st0 - henry_factor(3.0)) / henry_factor(3.0))
    g_d = bool(relD < 1e-12)
    ok = ok and g_d
    print("[rl] GATE D: ST-H == (R_sp/4pi N)(1+alpha^2) (ST0 {:.3e}, ratio(alpha=3) {:.1f} == 1+9, rel "
          "{:.1e}) -> {}".format(st0, st3 / st0, relD, "PASS" if g_d else "FAIL"), flush=True)

    # ---- GATE E: marcher Henry direction (alpha>0 broadens the amplified linewidth) ----
    m_mod = QDGainModel(QDGainParams(n_groups=11).with_detailed_balance_taus())
    soa = TravelingWaveSOA(m_mod, 0.6e-3, 30, nu_s_Hz=m_mod.p.nu0_Hz)
    nt = 4096
    A_in = np.full(nt, np.sqrt(1e-5) + 0j)              # CW coherent input
    dlw0, dlw4 = [], []
    for s in range(4):                                 # average over seeds (the source is stochastic)
        o0 = soa.amplify_coherent(A_in, 40e-3, alpha_lef=0.0, langevin=True, seed=s)
        o4 = soa.amplify_coherent(A_in, 40e-3, alpha_lef=4.0, langevin=True, seed=s)
        dlw0.append(linewidth_from_field(o0["A_out"][soa.nz:], soa.dt))
        dlw4.append(linewidth_from_field(o4["A_out"][soa.nz:], soa.dt))
    lw0, lw4 = float(np.mean(dlw0)), float(np.mean(dlw4))
    g_e = bool(lw4 > lw0 and np.isfinite(lw4) and lw0 >= 0.0)
    ok = ok and g_e
    print("[rl] GATE E: SOA Langevin linewidth broadens with alpha (alpha=0 {:.3e} -> alpha=4 {:.3e} "
          "Hz) -> {}".format(lw0, lw4, "PASS" if g_e else "FAIL"), flush=True)

    print("[rl] *** QD-SOA RIN + LINEWIDTH: {} ***".format("PASS" if ok else "FAIL"), flush=True)
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
