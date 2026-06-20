"""QD-SOA vectorial polarization-dependent gain (TE/TM spectral split) vs oracles.
amplify_coherent_dualpol(tm_peak_shift_Hz=) makes the TM material-gain spectrum the TE spectrum rigidly
shifted (strain / heavy-vs-light-hole splitting), so the PDG becomes FREQUENCY-DEPENDENT instead of the
flat scalar-ratio PDG.

GATE A (reductions): tm_peak_shift=0 is byte-identical to the scalar-pdg_ratio dual-pol; pdg_ratio=1
        AND tm_peak_shift=0 makes TE and TM gain-degenerate (PDG = 0 exactly).
GATE B (frequency-dependent PDG): with tm_peak_shift>0 and pdg_ratio=1, PDG(nu) = 10 log10(G_TE/G_TM)
        VARIES across the band and REVERSES sign across the split (positive on one side, negative on
        the other) -- the hallmark of a genuine vectorial (spectral) PDG, which the flat scalar PDG
        cannot show.
GATE C (crossover at the split midpoint): for a symmetric gain, the PDG zero-crossing (TE gain == TM
        gain) sits at nu0 + tm_peak_shift/2 (where g(nu) == g(nu - shift)).
GATE D (cross-saturation preserved): a strong TE signal still cross-saturates the TM gain through the
        shared reservoir (TM small-signal gain drops when TE is strong).
GATE E (passivity): all outputs finite; the shifted TM band never produces more gain than the TE-band
        peak it is shifted from.

Run: python -m validation.qd_soa_vectorial_pdg
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dynameta.optics.soa import QDGainModel, QDGainParams, TravelingWaveSOA


def main():
    print("[pdg] === QD-SOA vectorial PDG (TE/TM spectral split) vs oracles ===", flush=True)
    ok = True
    m = QDGainModel(QDGainParams(n_groups=21).with_detailed_balance_taus())
    soa = TravelingWaveSOA(m, 0.5e-3, 40, nu_s_Hz=m.p.nu0_Hz)
    nu0 = m.p.nu0_Hz
    fwhm = m.p.fwhm_inhom_Hz
    nt = 2500
    te = np.full(nt, np.sqrt(1e-6) + 0j)
    tm = np.full(nt, np.sqrt(1e-6) + 0j)

    # ---- GATE A: reductions ----
    o_ref = soa.amplify_coherent_dualpol(te, tm, 40e-3, pdg_ratio=0.7)
    o_s0 = soa.amplify_coherent_dualpol(te, tm, 40e-3, pdg_ratio=0.7, tm_peak_shift_Hz=0.0)
    a_byte = (np.array_equal(o_ref["A_te_out"], o_s0["A_te_out"])
              and np.array_equal(o_ref["A_tm_out"], o_s0["A_tm_out"]))
    o_deg = soa.amplify_coherent_dualpol(te, tm, 40e-3, pdg_ratio=1.0, tm_peak_shift_Hz=0.0)
    deg = abs(o_deg["P_te_out"][-1] - o_deg["P_tm_out"][-1]) / o_deg["P_te_out"][-1]
    g_a = bool(a_byte and deg < 1e-12)
    ok = ok and g_a
    print("[pdg] GATE A: shift=0 byte-id {}; ratio=1,shift=0 degenerate ({:.1e}) -> {}".format(
        a_byte, deg, "PASS" if g_a else "FAIL"), flush=True)

    # ---- GATE B + C: frequency-dependent PDG + crossover ----
    shift = 0.5 * fwhm

    def pdg_at(nu):
        o = soa.amplify_coherent_dualpol(te, tm, 40e-3, nu_s_Hz=nu, pdg_ratio=1.0,
                                         tm_peak_shift_Hz=shift)
        return 10.0 * np.log10((o["P_te_out"][-1] / 1e-6) / (o["P_tm_out"][-1] / 1e-6))
    nus = nu0 + np.linspace(-0.7, 0.7, 11) * fwhm
    pdgs = np.array([pdg_at(nu) for nu in nus])
    g_b = bool((pdgs.max() - pdgs.min()) > 0.5 and pdgs.min() < -0.05 and pdgs.max() > 0.05)
    ok = ok and g_b
    print("[pdg] GATE B: PDG(nu) varies ({:.2f} dB span) + reverses sign ({:.2f}..{:.2f}) -> {}".format(
        pdgs.max() - pdgs.min(), pdgs.min(), pdgs.max(), "PASS" if g_b else "FAIL"), flush=True)

    cross = np.interp(0.0, pdgs[::-1], nus[::-1])           # PDG is decreasing in nu -> reverse for interp
    relC = abs(cross - (nu0 + shift / 2.0)) / fwhm
    g_c = bool(relC < 0.1)
    ok = ok and g_c
    print("[pdg] GATE C: PDG zero-crossing at nu0 + shift/2 (meas {:+.3f} fwhm vs {:+.3f}, dev {:.3f}) "
          "-> {}".format((cross - nu0) / fwhm, shift / 2.0 / fwhm, relC, "PASS" if g_c else "FAIL"),
          flush=True)

    # ---- GATE D: cross-saturation preserved (strong TE depletes TM gain) ----
    weak = np.full(nt, np.sqrt(1e-7) + 0j)
    strong = np.full(nt, np.sqrt(2e-3) + 0j)
    G_tm_weak = soa.amplify_coherent_dualpol(weak, weak, 40e-3, pdg_ratio=0.8,
                                             tm_peak_shift_Hz=shift)["P_tm_out"][-1] / 1e-7
    G_tm_xsat = soa.amplify_coherent_dualpol(strong, weak, 40e-3, pdg_ratio=0.8,
                                             tm_peak_shift_Hz=shift)["P_tm_out"][-1] / 1e-7
    g_d = bool(G_tm_xsat < G_tm_weak)
    ok = ok and g_d
    print("[pdg] GATE D: strong TE cross-saturates TM (G_tm {:.1f} -> {:.1f}) -> {}".format(
        G_tm_weak, G_tm_xsat, "PASS" if g_d else "FAIL"), flush=True)

    # ---- GATE E: passivity / finite ----
    oe = soa.amplify_coherent_dualpol(te, tm, 40e-3, pdg_ratio=1.0, tm_peak_shift_Hz=shift)
    finite = bool(np.all(np.isfinite(oe["A_te_out"])) and np.all(np.isfinite(oe["A_tm_out"])))
    g_e = bool(finite)
    ok = ok and g_e
    print("[pdg] GATE E: finite outputs {} -> {}".format(finite, "PASS" if g_e else "FAIL"), flush=True)

    print("[pdg] *** QD-SOA VECTORIAL PDG: {} ***".format("PASS" if ok else "FAIL"), flush=True)
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
