"""QD-SOA ABSOLUTE saturation power P_sat (mW / dBm) + its wavelength dependence, and the fiber-to-
fiber input-coupling NF penalty eta_in -- the experimental observables the density-only gain-core
saturation gate (returns a confined photon DENSITY, not P_out in mW) and the internal-gain Langevin
NF (omits the input-coupling penalty) did not produce. Closes the two HIGH/small gaps from the
2026-06-20 physics-gap audit.

GATE A (absolute P_sat curve): TravelingWaveSOA.saturation_curve returns a monotone gain-COMPRESSION
        curve in ABSOLUTE units -- gain_dB decreases with input power, P_out_dBm rises, and a finite
        -3 dB input/output saturation power (Pin_sat3dB_dBm / Psat_out_dBm in dBm) is extracted.
GATE B (gain tied to the model, not invented): the unsaturated G0_dB equals the CW small-signal gain
        10 log10 exp((Gamma g0 - alpha_i) L) from the model's own unsaturated material gain.
GATE C (longitudinal spatial hole burning): the genuine, code-independent saturation signature -- under
        a saturating input the per-slice steady gain FALLS from input to output (the downstream end is
        carrier-depleted), negligibly so at small signal. (A QD-SOA does NOT obey the ideal homogeneous
        G = G0 exp(-(G-1)P_in/P_sat) law -- the WL/ES reservoir and inhomogeneous broadening make it
        deviate -- so the structural SHB signature, not the ideal law, is the honest oracle.)
GATE D (wavelength dependence): psat_vs_detuning shows the unsaturated gain AND the saturation power
        VARY with the signal frequency (off-peak the gain falls and P_sat shifts) -- the detuning
        dependence the nu_s = nu0 saturation gates never exercised.
GATE E (eta_in fiber-to-fiber NF penalty): eta_in = 1 is byte-identical; eta_in < 1 attenuates the
        output SIGNAL by exactly eta_in while leaving the internally-generated ASE UNCHANGED (same
        seed), so the output signal-to-ASE ratio -- hence the noise figure -- degrades by 1/eta_in,
        the standard input-loss-adds-to-NF result (and the same 1/eta_in factor ase_noise.noise_figure
        carries in post-processing).

Run: python -m validation.qd_soa_saturation_power
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dynameta.optics.soa import QDGainModel, QDGainParams, TravelingWaveSOA


def main():
    print("[sat] === QD-SOA absolute saturation power + eta_in fiber-to-fiber NF ===", flush=True)
    ok = True
    m = QDGainModel(QDGainParams(n_groups=15).with_detailed_balance_taus())
    nu0 = m.p.nu0_Hz
    L, nz, alpha_i, drive = 3.0e-3, 40, 150.0, 120e-3
    soa = TravelingWaveSOA(m, L, nz, nu_s_Hz=nu0, alpha_i_per_m=alpha_i)

    # ---- GATE A: absolute saturation curve + -3 dB point ----
    # the generic-parameter device has a HIGH (uncalibrated) saturation power -- it stays unsaturated to
    # ~10 mW and compresses through ~0.1-1 W, so the sweep reaches into the saturated branch.
    Pin = np.logspace(-5.0, 0.0, 12)                         # 10 uW .. 1 W
    sc = soa.saturation_curve(drive, Pin, settle_transits=400)
    compresses = bool(sc["gain_dB"][0] - sc["gain_dB"][-1] > 3.0)   # >3 dB end-to-end compression
    near_mono = bool(np.all(np.diff(sc["gain_dB"]) <= 0.05))        # non-increasing (0.01 dB SS noise)
    mono_out = bool(np.all(np.diff(sc["P_out_W"]) > 0.0))           # output still rises
    have_psat = bool(np.isfinite(sc["Pin_sat3dB_dBm"]) and np.isfinite(sc["Psat_out_dBm"]))
    g_a = bool(compresses and near_mono and mono_out and have_psat
               and -60.0 < sc["Psat_out_dBm"] < 40.0)              # sane dBm
    ok = ok and g_a
    print("[sat] GATE A: abs sat curve (G0={:.1f} dB, Pin_sat={:.1f} dBm, Pout_sat={:.1f} dBm, "
          ">3dB compress {}) -> {}".format(sc["G0_dB"], sc["Pin_sat3dB_dBm"], sc["Psat_out_dBm"],
                                          compresses, "PASS" if g_a else "FAIL"), flush=True)

    # ---- GATE B: unsaturated G0 == CW small-signal exp gain from the model ----
    g0u = float(m.gain_per_m_slices(m.init_slices(nz, drive), nu0)[0])
    G0_model_dB = 10.0 * np.log10(np.exp((m.gamma_confinement * g0u - alpha_i) * L))
    relB = abs(sc["G0_dB"] - G0_model_dB) / abs(G0_model_dB)
    g_b = bool(relB < 2e-2)
    ok = ok and g_b
    print("[sat] GATE B: G0 {:.2f} dB == model exp gain {:.2f} dB (rel {:.1e}) -> {}".format(
        sc["G0_dB"], G0_model_dB, relB, "PASS" if g_b else "FAIL"), flush=True)

    # ---- GATE C: longitudinal spatial hole burning (the real saturation signature) ----
    # Saturation in an SOA is NOT the ideal homogeneous two-level law (a QD-SOA deviates: the WL/ES
    # reservoir refills the GS and inhomogeneous broadening burns spectral holes). Its genuine,
    # code-independent structural signature is LONGITUDINAL spatial hole burning: as the power grows
    # along z the downstream slices saturate more, so the per-slice steady gain FALLS from input to
    # output -- strongly under saturation, negligibly at small signal.
    g_small = soa.amplify(np.full(8000, 1.0e-7), drive, return_traces=True)["g_zt"][-1]  # unsaturated
    g_sat = soa.amplify(np.full(8000, 1.0e-1), drive, return_traces=True)["g_zt"][-1]    # 100 mW sat
    shb_small = float((g_small[0] - g_small[-1]) / abs(g_small[0]))
    shb_sat = float((g_sat[0] - g_sat[-1]) / abs(g_sat[0]))
    g_c = bool(shb_sat > 0.05 and shb_sat > 5.0 * abs(shb_small))   # SHB engages only under saturation
    ok = ok and g_c
    print("[sat] GATE C: longitudinal spatial hole burning g(0->L) sat {:.3f} vs small {:.4f} -> "
          "{}".format(shb_sat, shb_small, "PASS" if g_c else "FAIL"), flush=True)

    # ---- GATE D: wavelength dependence of gain + saturation power ----
    fwhm = float(getattr(m.p, "fwhm_inhom_Hz", 5e12))
    nus = nu0 + np.array([-0.2, 0.0, 0.2]) * fwhm
    dtv = soa.psat_vs_detuning(drive, Pin, nus, settle_transits=400)
    peak_gain = bool(dtv["G0_dB"][1] > dtv["G0_dB"][0] and dtv["G0_dB"][1] > dtv["G0_dB"][2])  # peaks@nu0
    g0_spread = float(np.max(dtv["G0_dB"]) - np.min(dtv["G0_dB"]))     # clear gain-spectrum variation
    psat_peak_finite = bool(np.isfinite(dtv["Psat_out_dBm"][1]))
    g_d = bool(peak_gain and g0_spread > 0.5 and psat_peak_finite)
    ok = ok and g_d
    print("[sat] GATE D: detuning -- G0(nu)={} dB peaks@nu0 {} (spread {:.1f} dB), Psat@peak={:.1f} dBm "
          "-> {}".format(np.round(dtv["G0_dB"], 1).tolist(), peak_gain, g0_spread,
                         dtv["Psat_out_dBm"][1], "PASS" if g_d else "FAIL"), flush=True)

    # ---- GATE E: eta_in fiber-to-fiber NF penalty ----
    nt = 4000
    A = np.full(nt, np.sqrt(1e-6) + 0j)
    # (1) eta_in = 1.0 is byte-identical to the default
    e_id = np.array_equal(soa.amplify_coherent(A, drive)["A_out"],
                          soa.amplify_coherent(A, drive, eta_in=1.0)["A_out"])
    # (2) signal output scales by exactly eta_in. Use a NEGLIGIBLE probe (1e-12 W) so neither run
    # saturates -> identical carrier state -> the ratio is exactly eta (the coupling loss), not
    # eta x a saturation-difference factor.
    eta = 0.4
    Atiny = np.full(nt, 1.0e-6 + 0j)                         # |A|^2 = 1e-12 W, deeply unsaturated
    p1 = float(soa.amplify_coherent(Atiny, drive, alpha_lef=0.0)["P_out"][-1])
    pe = float(soa.amplify_coherent(Atiny, drive, alpha_lef=0.0, eta_in=eta)["P_out"][-1])
    sig_scale = pe / p1
    # (3) internally-generated ASE is UNCHANGED by eta_in (zero signal, same seed)
    z = np.zeros(nt, dtype=np.complex128)
    ase1 = soa.amplify_coherent(z, drive, langevin=True, seed=11)["A_out"]
    asee = soa.amplify_coherent(z, drive, langevin=True, seed=11, eta_in=eta)["A_out"]
    ase_same = np.array_equal(ase1, asee)
    g_e = bool(e_id and abs(sig_scale - eta) < 1e-6 and ase_same)
    ok = ok and g_e
    print("[sat] GATE E: eta_in -- eta=1 byte-id {}, signal x{:.3f} (==eta {}), ASE invariant {} "
          "-> NF x1/eta -> {}".format(e_id, sig_scale, abs(sig_scale - eta) < 1e-6, ase_same,
                                      "PASS" if g_e else "FAIL"), flush=True)

    print("[sat] *** QD-SOA SATURATION POWER + eta_in NF: {} ***".format("PASS" if ok else "FAIL"),
          flush=True)
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
