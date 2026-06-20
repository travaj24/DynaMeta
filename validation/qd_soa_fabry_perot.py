"""QD-SOA Fabry-Perot cavity (real facet feedback) vs analytic oracles. amplify_fabry_perot
co-propagates FORWARD + BACKWARD envelopes coupled by the facet power reflectivities R1, R2, both
saturating the shared carriers -- replacing the single-pass Saitoh-Mukai ripple METRIC with the
actual round-trip feedback (gain ripple, resonant Airy enhancement, gain clamping at lasing).

GATE A (reduction): R1=R2=0 -> the transmitted field is byte-identical to single-pass
        amplify_coherent (the backward field stays 0).
GATE B (ripple): sweeping the round-trip detuning phase, the peak-to-valley gain ripple equals the
        Saitoh-Mukai metric facet_gain_ripple_dB(G_sp, R1, R2) at small R (weak signal) -- the cavity
        REPRODUCES the closed-form metric it replaces.
GATE C (Airy resonant enhancement): below threshold the on-resonance FP power gain / single-pass ==
        (1-R1)(1-R2)/(1 - sqrt(R1 R2) G_sp)^2 (the Airy denominator) over a range of sqrt(R1R2)G_sp
        -- the independent Fabry-Perot oracle.
GATE D (external-seed gain saturation near resonance): pushing sqrt(R1R2) G_sp toward 1 with an
        injected seed, the built-up intracavity field depletes the carriers so the SATURATED round-
        trip gain stays bounded (<= ~1, no run-away) and the saturated single-pass gain is below the
        unsaturated G_sp. (No spontaneous/ASE seed is modelled, so this is external-seed saturation,
        not lasing-from-noise / a self-consistent threshold pin -- see the method scope note.)
GATE E (passivity): every case stays finite (no NaN) and the R=0 reduction holds.

Run: python -m validation.qd_soa_fabry_perot
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dynameta.optics.soa.metrics import facet_gain_ripple_dB
from dynameta.optics.soa.qd_gain import QDGainModel, QDGainParams
from dynameta.optics.soa.traveling_wave import TravelingWaveSOA


def main():
    print("[fp] === QD-SOA Fabry-Perot cavity (facet feedback) vs analytic oracles ===", flush=True)
    ok = True
    m = QDGainModel(QDGainParams(n_groups=21).with_detailed_balance_taus())
    L, nz, drv = 3.0e-3, 150, 150e-3
    soa = TravelingWaveSOA(m, L, nz, nu_s_Hz=m.p.nu0_Hz)
    eps = 1.0e-6
    nt = 6000
    A = np.full(nt, eps) + 0j
    k = nt - 1

    # single-pass power gain (the FP oracle reference)
    sp = soa.amplify_coherent(A, drv, alpha_lef=0.0)["A_out"]
    Gsp = float(np.abs(sp[k]) ** 2 / eps ** 2)

    # GATE A: reduction R1=R2=0 == single-pass
    fp0 = soa.amplify_fabry_perot(A, drv, R1=0.0, R2=0.0, alpha_lef=0.0)["A_out"]
    g_a = bool(np.max(np.abs(fp0 - sp)) < 1e-15)
    ok = ok and g_a
    print("[fp] GATE A: R=0 == single-pass amplify_coherent (max|d| {:.1e}) -> {}".format(
        float(np.max(np.abs(fp0 - sp))), "PASS" if g_a else "FAIL"), flush=True)

    # GATE B: ripple == Saitoh-Mukai (small R, weak signal)
    Rb = 5.0e-3
    gains = []
    for phi in np.linspace(0.0, 2.0 * np.pi, 25):
        o = soa.amplify_fabry_perot(A, drv, R1=Rb, R2=Rb, alpha_lef=0.0,
                                    roundtrip_phase=phi)["A_out"][k]
        gains.append(10.0 * np.log10(np.abs(o) ** 2 / eps ** 2))
    ripple_meas = max(gains) - min(gains)
    ripple_pred = facet_gain_ripple_dB(Gsp, Rb, Rb)
    relB = abs(ripple_meas - ripple_pred) / ripple_pred
    g_b = bool(relB < 5e-2)
    ok = ok and g_b
    print("[fp] GATE B: ripple meas {:.3f} dB == Saitoh-Mukai {:.3f} dB (G_sp {:.2f} dB, R {:.0e}, "
          "rel {:.1e}) -> {}".format(ripple_meas, ripple_pred, 10 * np.log10(Gsp), Rb, relB,
                                     "PASS" if g_b else "FAIL"), flush=True)

    # GATE C: Airy resonant enhancement vs (1-R)^2/(1-sqrt(R1R2)Gsp)^2
    relC = 0.0
    for rg in (0.2, 0.4, 0.6):
        R = rg / Gsp                                            # sqrt(R1R2)Gsp = R Gsp = rg (R1=R2=R)
        out = soa.amplify_fabry_perot(A, drv, R1=R, R2=R, alpha_lef=0.0,
                                      roundtrip_phase=0.0)["A_out"][k]
        enh = (np.abs(out) ** 2 / eps ** 2) / Gsp               # FP gain / single-pass
        enh_airy = (1.0 - R) ** 2 / (1.0 - rg) ** 2
        relC = max(relC, abs(enh - enh_airy) / enh_airy)
    g_c = bool(relC < 5e-2)
    ok = ok and g_c
    print("[fp] GATE C: on-resonance enhancement == Airy (1-R)^2/(1-sqrt(R1R2)Gsp)^2 (max rel {:.1e}) "
          "-> {}".format(relC, "PASS" if g_c else "FAIL"), flush=True)

    # GATE D: gain clamping at lasing -- push sqrt(R1R2)Gsp ~ 0.95, not-weak input, gain clamps
    R = 0.95 / Gsp
    ntL = 12000
    AL = np.full(ntL, np.sqrt(2.0e-4)) + 0j                     # 0.2 mW seed (builds the cavity)
    rL = soa.amplify_fabry_perot(AL, drv, R1=R, R2=R, alpha_lef=0.0, roundtrip_phase=0.0)
    oL = rL["A_out"][ntL - 1]
    finite = bool(np.isfinite(np.abs(oL)) and not np.any(np.isnan(rL["A_out"])))
    # saturated single-pass gain from the depleted carrier state
    g_slices = m.gain_per_m_slices(rL["state"], m.p.nu0_Hz)
    G_clamp = float(np.exp(m.p.Gamma * np.mean(g_slices) * L))           # ~ single-pass power gain now
    roundtrip_clamped = R * G_clamp                            # must stay bounded <= ~1 (no runaway)
    g_d = bool(finite and roundtrip_clamped < 1.05 and G_clamp < Gsp)
    ok = ok and g_d
    print("[fp] GATE D: near resonance (target sqrt(R1R2)Gsp=0.95) external-seed feedback SATURATES "
          "the gain -- round-trip {:.3f} <= ~1, G {:.2f} dB < unsat {:.2f} dB, finite {} -> {}".format(
              roundtrip_clamped, 10 * np.log10(G_clamp), 10 * np.log10(Gsp), finite,
              "PASS" if g_d else "FAIL"), flush=True)

    g_e = bool(g_a and finite)
    ok = ok and g_e
    print("[fp] GATE E: passivity (R=0 reduction + finite near threshold) -> {}".format(
        "PASS" if g_e else "FAIL"), flush=True)

    print("[fp] *** QD-SOA FABRY-PEROT: {} ***".format("PASS" if ok else "FAIL"), flush=True)
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
