"""Coherent multi-tone QD-SOA nonlinearities (roadmap SOA Phase 3): cross-gain modulation
(XGM) and four-wave mixing (FWM) from the complex-envelope traveling-wave engine
(TravelingWaveSOA.amplify_coherent), vs analytic / known-limit oracles.

GATE A (reduces to the verified power engine): with alpha = 0 and a single real CW tone the
        coherent solve's |A_out|^2 equals the (Phase-2-validated) power-propagation P_out to
        machine precision -- the complex march is consistent with the real one.
GATE B (FWM sidebands appear): two CW tones (pump at envelope 0, probe at +f_d within the
        carrier cutoff) generate the conjugate FWM sideband at -f_d (the carrier-density-
        pulsation product 2 nu_pump - nu_probe) with measurable conversion efficiency.
GATE C (FWM efficiency rolls off with detuning): the conjugate conversion efficiency
        decreases monotonically as the tone spacing rises past the carrier cutoff ~1/(2 pi
        tau) -- carriers can no longer follow the beat.
GATE D (the (1 + alpha^2) carrier-density-pulsation law): the FWM conjugate efficiency scales
        as 1 + alpha^2 with the linewidth enhancement factor -- the index grating adds
        alpha^2 on top of the gain grating's 1 (Agrawal-Olsson / Mecozzi CDP-FWM). Verified
        to < 5% across alpha = 0..5.
GATE E (cross-gain modulation): a strong CW pump saturates the medium and suppresses the gain
        a co-propagating (far-detuned, beyond FWM) probe sees -- the XGM that crosstalks WDM
        channels. The suppression grows monotonically with pump power and exceeds 1 dB once
        the pump saturates the amplifier.

Run: python -m validation.qd_soa_fwm_xgm
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dynameta.optics.soa.qd_gain import QDGainModel, QDGainParams
from dynameta.optics.soa.traveling_wave import TravelingWaveSOA


def _conj_efficiency(soa, dt, f_d_Hz, alpha, *, Pp=4.0e-3, Ps=1.0e-4, drive=40.0e-3,
                     n_beats=40):
    """FWM conjugate (at -f_d) conversion efficiency = P(-f_d)/P(probe at +f_d), from the
    settled output spectrum of pump (envelope 0) + weak probe (+f_d)."""
    nt = int(n_beats / f_d_Hz / dt)
    t = np.arange(nt) * dt
    A_in = np.sqrt(Pp) + np.sqrt(Ps) * np.exp(1j * 2.0 * np.pi * f_d_Hz * t)
    r = soa.amplify_coherent(A_in, drive=drive, alpha_lef=alpha)
    y = r["A_out"][nt // 2:]
    Y = np.abs(np.fft.fft(y * np.hanning(y.size))) ** 2
    f = np.fft.fftfreq(y.size, dt)

    def binpow(ft):
        return Y[int(np.argmin(np.abs(f - ft)))]
    return binpow(-f_d_Hz) / binpow(f_d_Hz), binpow(2.0 * f_d_Hz) / binpow(f_d_Hz)


def main():
    print("[fx] === QD-SOA coherent multi-tone: XGM + FWM vs analytic oracles ===", flush=True)
    ok = True
    qd = QDGainModel(QDGainParams(n_groups=1).with_detailed_balance_taus())
    nu0 = qd.p.nu0_Hz
    L, Nz = 0.6e-3, 50
    soa = TravelingWaveSOA(qd, L, Nz, nu_s_Hz=nu0)
    dt = soa.dt

    # ---- GATE A: coherent (alpha=0, single tone) == power engine ----
    nt = int(3.0e-9 / dt)
    Pcw = 2.0e-3
    a_co = soa.amplify_coherent(np.full(nt, np.sqrt(Pcw)), drive=40.0e-3, alpha_lef=0.0)
    a_pw = soa.amplify(np.full(nt, Pcw), drive=40.0e-3)
    relA = abs(a_co["P_out"][-1] - a_pw["P_out"][-1]) / a_pw["P_out"][-1]
    g_a = bool(relA < 1e-12)
    ok = ok and g_a
    print("[fx] GATE A: coherent(alpha=0,1 tone) |A|^2 == power P_out (rel {:.1e}) -> {}".format(
        relA, "PASS" if g_a else "FAIL"), flush=True)

    # ---- GATE B: FWM conjugate sideband present ----
    eta_conj, eta_up = _conj_efficiency(soa, dt, 20e9, 2.0)
    g_b = bool(eta_conj > 1e-5 and np.isfinite(eta_conj))
    ok = ok and g_b
    print("[fx] GATE B: FWM conjugate sideband at -f_d present (eta_conj/probe = {:.2e}, "
          "up-product {:.2e}) -> {}".format(eta_conj, eta_up, "PASS" if g_b else "FAIL"),
          flush=True)

    # ---- GATE C: FWM efficiency rolls off with detuning ----
    etas = [(_conj_efficiency(soa, dt, fd * 1e9, 2.0)[0]) for fd in (10.0, 40.0, 120.0)]
    g_c = bool(etas[0] > etas[1] > etas[2] > 0.0)
    ok = ok and g_c
    print("[fx] GATE C: FWM eta rolls off with detuning [{:.2e}, {:.2e}, {:.2e}] over "
          "[10, 40, 120] GHz -> {}".format(etas[0], etas[1], etas[2],
                                           "PASS" if g_c else "FAIL"), flush=True)

    # ---- GATE D: (1 + alpha^2) carrier-density-pulsation law ----
    e0 = _conj_efficiency(soa, dt, 20e9, 0.0)[0]
    worst = 0.0
    rows = []
    for a in (1.0, 2.0, 5.0):
        ratio = _conj_efficiency(soa, dt, 20e9, a)[0] / e0
        worst = max(worst, abs(ratio - (1.0 + a * a)) / (1.0 + a * a))
        rows.append((a, ratio, 1.0 + a * a))
    g_d = bool(worst < 0.05)
    ok = ok and g_d
    print("[fx] GATE D: FWM eta(alpha)/eta(0) == 1 + alpha^2 (worst rel {:.3f}); "
          "{} -> {}".format(worst, ", ".join("a={:.0f}:{:.2f}vs{:.0f}".format(a, r, e)
                                              for a, r, e in rows),
                            "PASS" if g_d else "FAIL"), flush=True)

    # ---- GATE E: cross-gain modulation (settled, far-detuned probe) ----
    fd, Ps = 200e9, 1.0e-6
    ntE = int(5.0e-9 / dt)                                    # settle >> tau_sp
    tE = np.arange(ntE) * dt
    fE = np.fft.fftfreq(int(ntE - int(0.7 * ntE)), dt)

    def probe_gain_ratio(Pp):
        sl = slice(int(0.7 * ntE), ntE)
        a1 = soa.amplify_coherent(np.sqrt(Pp) + np.sqrt(Ps) * np.exp(1j * 2 * np.pi * fd * tE),
                                  drive=40.0e-3, alpha_lef=2.0)["A_out"][sl]
        a0 = soa.amplify_coherent(np.sqrt(Ps) * np.exp(1j * 2 * np.pi * fd * tE),
                                  drive=40.0e-3, alpha_lef=2.0)["A_out"][sl]

        def b(arr):
            Y = np.abs(np.fft.fft(arr * np.hanning(arr.size))) ** 2
            return Y[int(np.argmin(np.abs(fE - fd)))]
        return b(a1) / b(a0)
    xgm_lo = 10.0 * np.log10(probe_gain_ratio(8.0e-3))
    xgm_hi = 10.0 * np.log10(probe_gain_ratio(30.0e-3))
    g_e = bool(xgm_hi < xgm_lo < 0.1 and xgm_hi < -1.0)
    ok = ok and g_e
    print("[fx] GATE E: XGM probe-gain suppression {:.2f} dB (8 mW) -> {:.2f} dB (30 mW pump), "
          "grows with pump -> {}".format(xgm_lo, xgm_hi, "PASS" if g_e else "FAIL"), flush=True)

    # ---- GATE F: FWM up/down-conversion asymmetry grows with alpha ----
    # probe ABOVE vs BELOW the pump gives the SAME conjugate efficiency at alpha = 0 (the gain
    # grating is symmetric) but DIFFERS once the index grating (alpha) breaks the symmetry --
    # the carrier-density-pulsation up/down asymmetry.
    def asym(alpha, fd=20e9):
        e_up, _ = _conj_efficiency(soa, dt, fd, alpha)        # probe at +fd -> conj at -fd
        # probe at -fd -> conj at +fd
        nt = int(40 / fd / dt)
        t = np.arange(nt) * dt
        A = np.sqrt(4.0e-3) + np.sqrt(1.0e-4) * np.exp(-1j * 2.0 * np.pi * fd * t)
        r = soa.amplify_coherent(A, drive=40.0e-3, alpha_lef=alpha)
        y = r["A_out"][nt // 2:]
        Y = np.abs(np.fft.fft(y * np.hanning(y.size))) ** 2
        f = np.fft.fftfreq(y.size, dt)
        e_dn = Y[int(np.argmin(np.abs(f - fd)))] / Y[int(np.argmin(np.abs(f + fd)))]
        return abs(e_up - e_dn) / (e_up + e_dn)
    asym0, asym_hi = asym(0.0), asym(4.0)
    g_f = bool(asym0 < 1e-3 and asym_hi > 3e-3 and asym_hi > asym0)
    ok = ok and g_f
    print("[fx] GATE F: FWM up/down asymmetry alpha=0 {:.2e} (symmetric) -> alpha=4 {:.2e} "
          "(index grating breaks symmetry) -> {}".format(asym0, asym_hi,
                                                         "PASS" if g_f else "FAIL"), flush=True)

    print("[fx] *** QD-SOA FWM/XGM: {} ***".format("PASS" if ok else "FAIL"), flush=True)
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
