"""QD-SOA stochastic Langevin spontaneous-emission noise vs analytic oracles. amplify_coherent (and
amplify_fabry_perot) gain a langevin flag: each slice each step adds a complex-Gaussian field
increment of variance Gamma g_sp(z) h nu v_g -- the fluctuation-dissipation spontaneous source. Its
downstream-amplified accumulation reproduces the analytic ASE PSD, its intensity is exponentially
distributed (complex-Gaussian ASE), and the carrier-induced index couples its amplitude noise into
phase (the Henry amplitude-phase mechanism). Reproducible via seed; OFF -> deterministic byte-identical.

GATE A (reduction): langevin=False -> byte-identical to the deterministic engine (amplify_coherent AND
        amplify_fabry_perot); same seed -> identical realization.
GATE B (mean ASE == analytic): no signal, gain on -> the time-averaged ASE power x dt equals the
        analytic single-pass PSD ase_output_psd = n_sp h nu (G-1) (the source variance is calibrated
        from first principles, not fitted).
GATE C (complex-Gaussian statistics): the pure-ASE output field is complex Gaussian, so the intensity
        I = |A|^2 is EXPONENTIALLY distributed -> <I^2>/<I>^2 = 2 (the ASE photon statistics).
GATE D (Henry amplitude-phase coupling): the carrier-induced index converts ASE amplitude noise into
        phase noise, so the LOW-FREQUENCY (within-carrier-bandwidth) phase noise of an amplified tone
        increases MONOTONICALLY with the linewidth-enhancement factor alpha (the (1+alpha^2) mechanism
        is present and in the right direction). SCOPE: the quantitative (1+alpha^2) laser linewidth is
        a gain-CLAMPED (above-threshold) result whose Hz-MHz width is below the fs-step time-domain
        marcher's frequency resolution; the marcher captures the coupling direction + the gain clamp
        (FP, Phase 13), not the absolute laser linewidth.
GATE E (passivity): finite (no NaN), and the Fabry-Perot Langevin path also reduces (off == det).

Run: python -m validation.qd_soa_langevin
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dynameta.constants import HBAR
from dynameta.optics.soa.ase_noise import ase_output_psd
from dynameta.optics.soa.qd_gain import QDGainModel, QDGainParams
from dynameta.optics.soa.traveling_wave import TravelingWaveSOA

H_PLANCK = 2.0 * np.pi * HBAR


def main():
    print("[lv] === QD-SOA stochastic Langevin noise vs analytic oracles ===", flush=True)
    ok = True
    m = QDGainModel(QDGainParams(n_groups=21).with_detailed_balance_taus())
    L, nz, I = 1.0e-3, 80, 60e-3
    soa = TravelingWaveSOA(m, L, nz, nu_s_Hz=m.p.nu0_Hz)
    nu0, dt = m.p.nu0_Hz, soa.dt

    # ---- GATE A: reduction + seed reproducibility ----
    A0 = np.zeros(20000) + 0j
    det = soa.amplify_coherent(A0, I, alpha_lef=0.0)["A_out"]
    off = soa.amplify_coherent(A0, I, alpha_lef=0.0, langevin=False)["A_out"]
    r1 = soa.amplify_coherent(A0, I, alpha_lef=0.0, langevin=True, seed=7)["A_out"]
    r2 = soa.amplify_coherent(A0, I, alpha_lef=0.0, langevin=True, seed=7)["A_out"]
    fp_det = soa.amplify_fabry_perot(A0[:4000], I, R1=1e-3, R2=1e-3, alpha_lef=0.0)["A_out"]
    fp_off = soa.amplify_fabry_perot(A0[:4000], I, R1=1e-3, R2=1e-3, alpha_lef=0.0,
                                     langevin=False)["A_out"]
    g_a = bool(np.array_equal(det, off) and np.array_equal(r1, r2) and np.array_equal(fp_det, fp_off))
    ok = ok and g_a
    print("[lv] GATE A: langevin=False == deterministic (coherent + FP) + seed reproducible -> "
          "{}".format("PASS" if g_a else "FAIL"), flush=True)

    # ---- GATE B: mean ASE == analytic PSD ----
    out = soa.amplify_coherent(A0, I, alpha_lef=0.0, langevin=True, seed=1)["A_out"][nz + 200:]
    mean_P = float(np.mean(np.abs(out) ** 2))
    y = m.steady_state(I, S_conf_m3=0.0, nu_s_Hz=nu0)
    g0 = float(m.material_gain_per_m(m.rho_GS(y), nu0))
    S_an = float(ase_output_psd(np.full(nz, g0), np.full(nz, float(m.rho_GS(y).max())), L / nz, nu0,
                                m.p.Gamma, 0.0, m_pol=1))
    relB = abs(mean_P * dt - S_an) / S_an
    g_b = bool(relB < 3e-2)
    ok = ok and g_b
    print("[lv] GATE B: mean ASE <|A|^2> dt {:.3e} == analytic ase_output_psd {:.3e} W/Hz (rel {:.1e}) "
          "-> {}".format(mean_P * dt, S_an, relB, "PASS" if g_b else "FAIL"), flush=True)

    # ---- GATE C: complex-Gaussian -> exponential intensity, <I^2>/<I>^2 = 2 ----
    Ii = np.abs(out) ** 2
    moment = float(np.mean(Ii ** 2) / np.mean(Ii) ** 2)
    g_c = bool(abs(moment - 2.0) < 0.1)
    ok = ok and g_c
    print("[lv] GATE C: ASE intensity exponential <I^2>/<I>^2 = {:.3f} (-> 2) -> {}".format(
        moment, "PASS" if g_c else "FAIL"), flush=True)

    # ---- GATE D: Henry amplitude-phase coupling direction (low-f phase noise rises with alpha) ----
    nt2 = 60000
    A = np.full(nt2, np.sqrt(2.0e-3)) + 0j
    lf_pow = {}
    for al in (0.0, 1.0, 2.0, 3.0):
        o = soa.amplify_coherent(A, 30e-3, alpha_lef=al, langevin=True, seed=5)["A_out"][nz + 1000:]
        ph = np.unwrap(np.angle(o))
        ph = ph - np.polyval(np.polyfit(np.arange(ph.size), ph, 1), np.arange(ph.size))   # detrend
        P = np.abs(np.fft.rfft(ph)) ** 2
        f = np.fft.rfftfreq(ph.size, dt)
        lf_pow[al] = float(P[f < 2e9].sum())
    ratios = [lf_pow[a] / lf_pow[0.0] for a in (1.0, 2.0, 3.0)]
    monotone = ratios[0] > 1.02 and ratios[1] > ratios[0] and ratios[2] > ratios[1]
    g_d = bool(monotone)
    ok = ok and g_d
    print("[lv] GATE D: low-f phase noise rises with alpha (ratios to a0: {}) -> {}".format(
        ["{:.2f}".format(r) for r in ratios], "PASS" if g_d else "FAIL"), flush=True)

    # ---- GATE E: passivity ----
    g_e = bool(g_a and not np.any(np.isnan(out)) and np.all(np.isfinite(mean_P)))
    ok = ok and g_e
    print("[lv] GATE E: finite + FP reduction -> {}".format("PASS" if g_e else "FAIL"), flush=True)

    print("[lv] *** QD-SOA LANGEVIN NOISE: {} ***".format("PASS" if ok else "FAIL"), flush=True)
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
