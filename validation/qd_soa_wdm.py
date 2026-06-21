"""QD-SOA WDM multi-channel cross-gain saturation -- the wavelength-resolved carrier back-reaction the
single-scalar-at-nu_s marcher lumped together (spec 8 THz comb treated as monochromatic). Each channel
saturates the QD groups IT overlaps via its OWN homogeneous lineshape: the per-group stimulated drive is
sum_k L(nu_k) S_conf(P_k, nu_k) (model.step_slices_wdm / rhs_fields ls_gs), so widely-spaced channels
bleach DIFFERENT groups and barely cross-saturate -- the inhomogeneous-broadening / spectral-hole-burning
low-crosstalk advantage of a QD-SOA. (Research-grade gap from the 2026-06-20 physics-gap audit.)

GATE A (single-channel reduction): TravelingWaveSOA.amplify_wdm with ONE channel reproduces
        amplify_coherent (flat-gain, same alpha) -- the multi-channel engine collapses to the validated
        single-channel marcher.
GATE B (co-located lumping): TWO channels at the SAME frequency nu0 -- their total output power equals a
        SINGLE channel carrying the summed input power (the shared reservoir cannot distinguish
        same-frequency channels; ls_gs = L(nu0)(S1+S2)).
GATE C (wavelength-resolved cross-saturation): the cross-gain saturation a strong pump imposes on a weak
        probe DECREASES monotonically as the pump-probe separation grows (probe-alone vs probe-with-pump,
        both at the SAME probe frequency) -- each bleaches its own groups, so the overlap (hence the
        crosstalk) falls with detuning.
GATE D (passivity): at low drive (unpumped) every channel sees net loss (gain_dB < 0) -- an absorber.
GATE E (crosstalk suppression vs the lumped model): the far-detuned cross-saturation is much smaller than
        the CO-LOCATED cross-saturation -- and the single-scalar-at-nu0 lump would predict the co-located
        value for ALL separations, so this quantifies the resolved-physics advantage.

Run: python -m validation.qd_soa_wdm
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dynameta.optics.soa import QDGainModel, QDGainParams, TravelingWaveSOA


def _probe_gain_dB(soa, drive, pump, probe, nu_pump, nu_probe, nt):
    """probe output gain [dB] WITH a co-propagating pump (channel 1 is the probe)."""
    r = soa.amplify_wdm([(nu_pump, pump), (nu_probe, probe)], drive, alpha_lef=0.0)
    return 10.0 * np.log10(r["channels"][1]["P_out"][-1] / (np.abs(probe[0]) ** 2))


def _alone_gain_dB(soa, drive, probe, nu_probe):
    r = soa.amplify_wdm([(nu_probe, probe)], drive, alpha_lef=0.0)
    return 10.0 * np.log10(r["channels"][0]["P_out"][-1] / (np.abs(probe[0]) ** 2))


def main():
    print("[wdm] === QD-SOA WDM multi-channel cross-gain saturation ===", flush=True)
    ok = True
    m = QDGainModel(QDGainParams(n_groups=21).with_detailed_balance_taus())
    nu0 = m.p.nu0_Hz
    fwhm_hom = float(m.p.fwhm_hom_Hz)
    soa = TravelingWaveSOA(m, 2.0e-3, 40, nu_s_Hz=nu0, alpha_i_per_m=150.0)
    drive, nt = 120e-3, 4000
    A1 = np.full(nt, np.sqrt(1.0e-4) + 0j)                   # 0.1 mW
    A2 = np.full(nt, np.sqrt(3.0e-4) + 0j)                   # 0.3 mW

    # ---- GATE A: single-channel reduction to amplify_coherent ----
    rc = soa.amplify_coherent(A1, drive, alpha_lef=0.0)["P_out"][-1]
    rw = soa.amplify_wdm([(nu0, A1)], drive, alpha_lef=0.0)["channels"][0]["P_out"][-1]
    relA = abs(rw - rc) / rc
    g_a = bool(relA < 1e-9)
    ok = ok and g_a
    print("[wdm] GATE A: single-channel == amplify_coherent (rel {:.1e}) -> {}".format(
        relA, "PASS" if g_a else "FAIL"), flush=True)

    # ---- GATE B: two co-located channels == single channel at summed power ----
    two = soa.amplify_wdm([(nu0, A1), (nu0, A2)], drive, alpha_lef=0.0)["channels"]
    tot_two = two[0]["P_out"][-1] + two[1]["P_out"][-1]
    Acomb = np.sqrt(np.abs(A1) ** 2 + np.abs(A2) ** 2) + 0j
    one = soa.amplify_wdm([(nu0, Acomb)], drive, alpha_lef=0.0)["channels"][0]["P_out"][-1]
    relB = abs(tot_two - one) / one
    g_b = bool(relB < 1e-9)
    ok = ok and g_b
    print("[wdm] GATE B: co-located 2ch total == single @ summed power (rel {:.1e}) -> {}".format(
        relB, "PASS" if g_b else "FAIL"), flush=True)

    # ---- GATE C: cross-saturation decreases with channel separation ----
    pump = np.full(nt, np.sqrt(5.0e-2) + 0j)                 # 50 mW pump at nu0
    probe = np.full(nt, np.sqrt(1.0e-6) + 0j)               # 1 uW probe at nu0 + sep
    seps = np.array([0.3, 1.0, 2.0, 4.0]) * fwhm_hom
    xsat = []
    for sep in seps:
        nu_p = nu0 + sep
        g_alone = _alone_gain_dB(soa, drive, probe, nu_p)
        g_pump = _probe_gain_dB(soa, drive, pump, probe, nu0, nu_p, nt)
        xsat.append(g_alone - g_pump)                        # cross-saturation [dB]
    xsat = np.array(xsat)
    g_c = bool(np.all(np.diff(xsat) < 0.0) and xsat[0] > 2.0 * xsat[-1])  # strictly decreasing
    ok = ok and g_c
    print("[wdm] GATE C: XSAT(sep/hom={}) = {} dB monotone-decreasing -> {}".format(
        [float(round(s / fwhm_hom, 1)) for s in seps], [float(x) for x in np.round(xsat, 4)],
        "PASS" if g_c else "FAIL"), flush=True)

    # ---- GATE D: passivity (unpumped absorbs every channel) ----
    soa_lo = TravelingWaveSOA(m, 2.0e-3, 40, nu_s_Hz=nu0, alpha_i_per_m=150.0)
    rlo = soa_lo.amplify_wdm([(nu0, probe), (nu0 + fwhm_hom, probe)], 1.0e-4, alpha_lef=0.0)["channels"]
    gains_lo = [10.0 * np.log10(c["P_out"][-1] / 1.0e-6) for c in rlo]
    g_d = bool(all(g < 0.0 for g in gains_lo))
    ok = ok and g_d
    print("[wdm] GATE D: unpumped channels absorb (gains {} dB < 0) -> {}".format(
        [float(round(g, 2)) for g in gains_lo], "PASS" if g_d else "FAIL"), flush=True)

    # ---- GATE E: crosstalk suppression vs the co-located (lumped) value ----
    g_alone0 = _alone_gain_dB(soa, drive, probe, nu0)
    g_pump0 = _probe_gain_dB(soa, drive, pump, probe, nu0, nu0, nt)
    xsat_coloc = g_alone0 - g_pump0                          # what the lump-at-nu0 model predicts
    supp = xsat[-1] / xsat_coloc                             # far-detuned / co-located
    g_e = bool(xsat_coloc > 0.0 and supp < 0.3)
    ok = ok and g_e
    print("[wdm] GATE E: XSAT far {:.4f} dB vs co-located/lumped {:.4f} dB (ratio {:.3f} < 0.3) -> "
          "{}".format(xsat[-1], xsat_coloc, supp, "PASS" if g_e else "FAIL"), flush=True)

    print("[wdm] *** QD-SOA WDM MULTI-CHANNEL CROSS-SATURATION: {} ***".format("PASS" if ok else "FAIL"),
          flush=True)
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
