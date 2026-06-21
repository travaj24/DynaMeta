"""QD-coupled transverse filamentation -- the 2-D (x-z) beam-propagation model driven by the REAL
group-resolved QD saturable gain (QDGainModel.saturation_curve -> qd_gain_table) instead of the
standalone phenomenological g0/(1+S/Isat). This couples the QD rate-equation physics (WL/ES reservoir +
inhomogeneous broadening) into the transverse field, so the gain saturation that drives gain guiding /
self-focusing / filamentation is the device's actual gain, not a toy. (Research-grade gap from the
2026-06-20 physics-gap audit: the BPM was a standalone toy that did NOT couple the QD gain into 2-D.)

GATE A (phenomenological path byte-identical): a TransverseBPM with NO qd_gain_table reproduces the exact
        g0/(1 + S/Isat) saturable gain -- the default (None) path is unchanged.
GATE B (uniform-beam == 1-D QD saturable-gain ODE): a laterally-uniform beam through the QD-coupled BPM
        integrates dP/dz = (Gamma g_QD(P) - alpha_i) P with the SAME table (diffraction touches only
        k_x != 0) -- the coupling correctly drives the field with the tabulated QD gain.
GATE C (cross-engine == the 1-D marcher): the uniform-beam QD-BPM output power matches the validated
        TravelingWaveSOA.saturation_curve (the dynamic time-domain marcher) at the same input -- the QD
        gain table is consistent with the device's own saturation.
GATE D (the table IS the real QD gain): g_QD(P) is monotone-decreasing (saturation), its low-power limit
        equals the QD small-signal modal gain, and it compresses with power -- the real group-resolved
        QD saturation, not a fitted Lorentzian saturable form.
GATE E (alpha-driven filamentation): with the QD gain and a linewidth-enhancement alpha > 0, a noisy beam
        grows high-spatial-frequency filament-band power; at alpha = 0 (no gain->index coupling) it does
        NOT -- the self-focusing is the gain-index coupling acting on the QD gain, the beam-quality
        feedback the standalone toy could not source from the real device gain.

Run: python -m validation.qd_soa_filament_qd
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dynameta.optics.soa import (QDGainModel, QDGainParams, TransverseBPM, TravelingWaveSOA,
                                 qd_gain_table)


def _nb(bpm, I):
    """NORMALIZED high-spatial-frequency 'filament-band' power: band(I)/total(I), so the UNIFORM gain
    (which amplifies the whole field equally) divides out and only the alpha-driven preferential
    growth of the filament scale (modulational instability) survives. Grid-invariant."""
    F = np.abs(np.fft.rfft(I - I.mean()))
    k = np.fft.rfftfreq(I.size, d=bpm.dx)
    return float(F[(k > 3.0e4) & (k < 2.0e5)].sum()) / I.sum()


def main():
    print("[fil] === QD-coupled transverse filamentation (real QD gain in the 2-D BPM) ===", flush=True)
    ok = True
    m = QDGainModel(QDGainParams(n_groups=15).with_detailed_balance_taus())
    nu0 = m.p.nu0_Hz
    gam, ai, drive = m.gamma_confinement, 150.0, 120e-3
    Pg = np.logspace(-7.0, 0.3, 300)                         # 0.1 uW .. 2 W
    tab = qd_gain_table(m, drive, nu0, Pg)

    # ---- GATE A: phenomenological (no qd_gain_table) path byte-identical ----
    bph = TransverseBPM(300e-6, 256, 1.3e-6, 3.4, g0_per_m=2000.0, Isat_W=1.0e-3)
    Aa = (np.exp(-(bph.x / 40e-6) ** 2) + 0j)
    g_ph = bph.carrier_gain(Aa)
    g_exp = 2000.0 / (1.0 + np.abs(Aa) ** 2 / 1.0e-3)
    g_a = bool(np.array_equal(g_ph, g_exp))
    ok = ok and g_a
    print("[fil] GATE A: None-table gain == g0/(1+S/Isat) exactly -> {}".format(
        "PASS" if g_a else "FAIL"), flush=True)

    # ---- GATE B: uniform-beam QD-BPM == 1-D QD saturable-gain ODE ----
    bpm = TransverseBPM(300e-6, 256, 1.3e-6, 3.4, gamma_confinement=gam, alpha_i_per_m=ai,
                        qd_gain_table=tab)
    P0, Lz, nz = 1.0e-3, 2.0e-3, 4000
    Iout_bpm = float(bpm.propagate(np.full(256, np.sqrt(P0) + 0j), Lz, nz)["I_out"][0])
    gQD = lambda P: np.interp(P, tab[0], tab[1])
    P, dz = P0, Lz / nz
    for _ in range(nz):                                      # RK4 on dP/dz = (Gamma g_QD - alpha_i) P
        k1 = (gam * gQD(P) - ai) * P
        k2 = (gam * gQD(P + 0.5 * dz * k1) - ai) * (P + 0.5 * dz * k1)
        k3 = (gam * gQD(P + 0.5 * dz * k2) - ai) * (P + 0.5 * dz * k2)
        k4 = (gam * gQD(P + dz * k3) - ai) * (P + dz * k3)
        P = P + dz / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    relB = abs(Iout_bpm - P) / P
    g_b = bool(relB < 1e-6)
    ok = ok and g_b
    print("[fil] GATE B: uniform QD-BPM == 1-D QD saturable ODE (rel {:.1e}) -> {}".format(
        relB, "PASS" if g_b else "FAIL"), flush=True)

    # ---- GATE C: cross-engine == the validated 1-D marcher saturation ----
    soa = TravelingWaveSOA(m, Lz, 40, nu_s_Hz=nu0, alpha_i_per_m=ai)
    sc = soa.saturation_curve(drive, [P0, 2.0 * P0], settle_transits=400)
    P_marcher = 10.0 ** (sc["P_out_dBm"][0] / 10.0) * 1.0e-3     # dBm -> W at P0
    relC = abs(Iout_bpm - P_marcher) / P_marcher
    g_c = bool(relC < 2e-2)
    ok = ok and g_c
    print("[fil] GATE C: uniform QD-BPM {:.4f} mW == marcher {:.4f} mW (rel {:.1e}) -> {}".format(
        Iout_bpm * 1e3, P_marcher * 1e3, relC, "PASS" if g_c else "FAIL"), flush=True)

    # ---- GATE D: the table IS the real QD saturable gain ----
    g_ss = float(m.small_signal_gain_per_m(drive, nu0))      # QD unsaturated modal gain (GS+ES)
    mono = bool(np.all(np.diff(tab[1]) <= 1e-9))             # saturates (gain falls with power)
    relD = abs(tab[1][0] - g_ss) / abs(g_ss)                 # low-power limit == small-signal
    compresses = bool(tab[1][-1] < tab[1][0])
    g_d = bool(mono and relD < 1e-2 and compresses)
    ok = ok and g_d
    print("[fil] GATE D: QD table monotone {}, g(P_lo) {:.0f} == small-signal {:.0f} (rel {:.1e}), "
          "compresses {} -> {}".format(mono, tab[1][0], g_ss, relD, compresses,
                                       "PASS" if g_d else "FAIL"), flush=True)

    # ---- GATE E: alpha-driven filamentation, power-thresholded by the QD saturation ----
    # filamentation needs gain SATURATION (intensity ripple -> carrier ripple -> alpha index ripple ->
    # self-focusing); the QD gain saturates only at HIGH power, so filamentation appears at 200 mW and
    # NOT at 5 mW. Metric: normalized filament-band growth alpha=4 vs alpha=0 (uniform gain divides out).
    nx = 512
    rng = np.random.default_rng(3)
    base = TransverseBPM(300e-6, nx, 1.3e-6, 3.4, gamma_confinement=gam, alpha_i_per_m=ai,
                         qd_gain_table=tab)
    shape = np.exp(-(base.x / 60e-6) ** 2) * (1.0 + 0.02 * rng.standard_normal(nx))   # noisy beam

    def alpha_ratio(peak_W):
        """normalized-band growth(alpha=4) / growth(alpha=0) for a beam scaled to peak_W."""
        A0 = (shape * np.sqrt(peak_W / np.max(shape ** 2))) + 0j
        nb_in = _nb(base, np.abs(A0) ** 2)
        out = {}
        for a in (4.0, 0.0):
            b = TransverseBPM(300e-6, nx, 1.3e-6, 3.4, gamma_confinement=gam, alpha_i_per_m=ai,
                              alpha_lef=a, qd_gain_table=tab)
            out[a] = _nb(b, b.propagate(A0, 4.0e-3, 4000)["I_out"]) / nb_in
        return out[4.0] / out[0.0]

    r_hi = alpha_ratio(2.0e-1)                               # 200 mW -> saturates -> filaments
    r_lo = alpha_ratio(5.0e-3)                               # 5 mW -> unsaturated -> no filaments
    g_e = bool(r_hi > 1.3 and r_lo < 1.1)
    ok = ok and g_e
    print("[fil] GATE E: filamentation alpha4/alpha0 -- 200 mW {:.2f}x (>1.3 self-focus) vs 5 mW "
          "{:.2f}x (<1.1 none) -> {}".format(r_hi, r_lo, "PASS" if g_e else "FAIL"), flush=True)

    print("[fil] *** QD-COUPLED FILAMENTATION: {} ***".format("PASS" if ok else "FAIL"), flush=True)
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
