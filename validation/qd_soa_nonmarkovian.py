"""QD-SOA two-timescale (heterogeneous-rate) dephasing lineshape vs oracles. A two-channel dipole-
correlation m(t) = w1 exp(-2 pi g1|t|) + (1-w1) exp(-2 pi g2|t|) Fourier-transforms to a two-component
(SUPER-Lorentzian: sharper core, heavier wing) homogeneous line -- the multi-rate generalization of the
single-rate single-Lorentzian line the gain model assumes (each exp channel is itself single-rate /
Markovian; a sum of two is heterogeneous, not genuine bath-memory non-Markovianity).

GATE A (Wiener-Khinchin): the FFT of biexp_memory_kernel equals nonmarkovian_lineshape (the line is
        the Fourier transform of the dipole-correlation memory).
GATE B (Markovian limit): w1 = 1 (and g1 = g2) recover the single Lorentzian exactly.
GATE C (model gain reduction): QDGainModel.gain_per_m_nonmarkovian reduces EXACTLY to gain_per_m_slices
        (GS band) when w1 = 1 or gamma2_factor = 1 (the single-Lorentzian gain).
GATE D (genuinely non-Lorentzian): a single Lorentzian has L(2 HWHM)/L(0) = 1/5 exactly; the biexp line
        (narrow core + broad wing) is SUPER-Lorentzian -- its wing at 2 x its own HWHM EXCEEDS 1/5
        (heavier wing), so it cannot be any single Lorentzian.
GATE E (area + passivity): the line is area-normalized (analytic integral = 1; numeric -> 1 on a wide
        grid) and non-negative everywhere.

Run: python -m validation.qd_soa_nonmarkovian
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dynameta.optics.soa import QDGainModel, QDGainParams
from dynameta.optics.soa.lineshape import (biexp_memory_kernel, lorentzian_area,
                                           nonmarkovian_lineshape)


def main():
    print("[nm] === QD-SOA two-timescale (heterogeneous) dephasing lineshape vs oracles ===",
          flush=True)
    ok = True
    g1, g2, w1 = 20e9, 80e9, 0.6

    # ---- GATE A: Wiener-Khinchin (FFT of the memory kernel == the lineshape) ----
    N, dt = 16384, 2.0e-9 / 16384
    t = (np.arange(N) - N // 2) * dt
    M = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(biexp_memory_kernel(t, g1, g2, w1)))) * dt
    f = np.fft.fftshift(np.fft.fftfreq(N, dt))
    Lan = nonmarkovian_lineshape(f, g1, g2, w1)
    sel = np.abs(f) < 4.0 * g2
    relA = float(np.max(np.abs(M.real[sel] - Lan[sel])) / Lan.max())
    g_a = bool(relA < 1e-3)
    ok = ok and g_a
    print("[nm] GATE A: FFT(memory kernel) == lineshape (rel {:.1e}) -> {}".format(
        relA, "PASS" if g_a else "FAIL"), flush=True)

    # ---- GATE B: Markovian limit ----
    b1 = np.allclose(nonmarkovian_lineshape(f, g1, g2, 1.0), lorentzian_area(f, g1))
    b2 = np.allclose(nonmarkovian_lineshape(f, g1, g1, 0.5), lorentzian_area(f, g1))
    g_b = bool(b1 and b2)
    ok = ok and g_b
    print("[nm] GATE B: w1=1 and g1=g2 reduce to a single Lorentzian ({}/{}) -> {}".format(
        b1, b2, "PASS" if g_b else "FAIL"), flush=True)

    # ---- GATE C: model gain reduction ----
    mod = QDGainModel(QDGainParams(n_groups=15).with_detailed_balance_taus())
    st = mod.init_slices(4, 40e-3)
    nu0 = mod.p.nu0_Hz
    g_ref = mod.gain_per_m_slices(st, nu0)
    relC1 = float(np.max(np.abs(mod.gain_per_m_nonmarkovian(st, nu0, gamma2_factor=3.0, w1=1.0) - g_ref)))
    relC2 = float(np.max(np.abs(mod.gain_per_m_nonmarkovian(st, nu0, gamma2_factor=1.0, w1=0.5) - g_ref)))
    g_c = bool(relC1 < 1e-12 and relC2 < 1e-12)
    ok = ok and g_c
    print("[nm] GATE C: gain_per_m_nonmarkovian reduces to gain_per_m_slices (w1=1 {:.1e}, gamma2f=1 "
          "{:.1e}) -> {}".format(relC1, relC2, "PASS" if g_c else "FAIL"), flush=True)

    # ---- GATE D: genuinely non-Lorentzian (sub-Lorentzian heavy wing) ----
    fg = np.linspace(0.0, 20.0 * g2, 400000)
    Lg = nonmarkovian_lineshape(fg, g1, g2, w1)
    hwhm = float(np.interp(0.5 * Lg[0], Lg[::-1], fg[::-1]))   # own half-max half-width
    wing = float(np.interp(2.0 * hwhm, fg, Lg)) / Lg[0]        # value at 2 HWHM / peak
    g_d = bool(wing > 0.20 + 0.01)                             # single Lorentzian would be exactly 0.20
    ok = ok and g_d
    print("[nm] GATE D: super-Lorentzian wing L(2 HWHM)/L(0) = {:.3f} > 0.20 (single Lorentzian) -> "
          "{}".format(wing, "PASS" if g_d else "FAIL"), flush=True)

    # ---- GATE E: area + passivity (wide grid so the heavy Lorentzian wings are captured; the
    # analytic area is exactly w1 + (1-w1) = 1) ----
    fw = np.linspace(0.0, 1500.0 * g2, 3000000)
    area = float(np.trapezoid(nonmarkovian_lineshape(fw, g1, g2, w1), fw)) * 2.0   # even line -> x2
    nonneg = bool(np.all(Lan >= 0.0))
    g_e = bool(abs(area - 1.0) < 1e-2 and nonneg)
    ok = ok and g_e
    print("[nm] GATE E: area-normalized ({:.4f}) + non-negative {} -> {}".format(
        area, nonneg, "PASS" if g_e else "FAIL"), flush=True)

    print("[nm] *** QD-SOA NON-MARKOVIAN LINESHAPE: {} ***".format("PASS" if ok else "FAIL"), flush=True)
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
