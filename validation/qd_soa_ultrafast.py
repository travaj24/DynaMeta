"""Ultrafast nonlinear gain compression in the QD-SOA (roadmap SOA Phase 5): spectral hole
burning (SHB) + carrier heating (CH), the sub-picosecond gain dynamics on top of the
carrier-density reservoir. optics.soa.traveling_wave.UltrafastCompression folded into the
traveling-wave engine.

GATE A (off-switch, reduces to the carrier-density-only engine): eps_SHB = eps_CH = 0 gives a
        result BYTE-IDENTICAL to amplify() without the ultrafast layer.
GATE B (nonlinear gain compression): the ultrafast SHB + CH terms further suppress the CW
        saturated gain beyond the carrier-density saturation, and more strongly as the
        compression coefficients grow -- the extra gain compression QD-SOAs show at high power.
GATE C (two timescales): the SHB + CH compression depth recovers on a SUB-PICOSECOND-to-~1 ps
        timescale (~ the configured carrier-heating time tau_CH), far faster than the carrier
        reservoir gain recovery (~several ps) -- the hallmark two-timescale QD-SOA gain
        dynamics (fast SHB/CH + slow carrier refill).

Run: python -m validation.qd_soa_ultrafast
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dynameta.optics.soa.qd_gain import QDGainModel, QDGainParams
from dynameta.optics.soa.traveling_wave import TravelingWaveSOA, UltrafastCompression


def main():
    print("[uf] === QD-SOA ultrafast gain compression (SHB + CH) ===", flush=True)
    ok = True
    qd = QDGainModel(QDGainParams(n_groups=1).with_detailed_balance_taus())
    nu0 = qd.p.nu0_Hz
    uf = UltrafastCompression(eps_shb_m3=8e-23, tau_shb_s=8e-14,
                              eps_ch_m3=1.2e-22, tau_ch_s=7e-13)

    # ---- GATE A: off-switch byte-identical ----
    soa = TravelingWaveSOA(qd, 0.6e-3, 50, nu_s_Hz=nu0)
    P = np.full(2000, 2.0e-3)
    a = soa.amplify(P, 40e-3)
    b = soa.amplify(P, 40e-3, ultrafast=UltrafastCompression())   # eps = 0
    g_a = bool(np.array_equal(a["P_out"], b["P_out"]))
    ok = ok and g_a
    print("[uf] GATE A: eps=0 byte-identical to no-ultrafast -> {}".format(
        "PASS" if g_a else "FAIL"), flush=True)

    # ---- GATE B: ultrafast compression reduces CW saturated gain (monotone in eps) ----
    def cw_gain(P0, ufx):
        nt = int(6.0e-9 / soa.dt)
        return soa.amplify(np.full(nt, P0), 40e-3, ultrafast=ufx)["gain_dB"][-1]
    uf_half = UltrafastCompression(eps_shb_m3=4e-23, tau_shb_s=8e-14,
                                   eps_ch_m3=6e-23, tau_ch_s=7e-13)
    G_none = cw_gain(2.0e-2, None)
    G_half = cw_gain(2.0e-2, uf_half)
    G_full = cw_gain(2.0e-2, uf)
    g_b = bool(G_full < G_half < G_none)
    ok = ok and g_b
    print("[uf] GATE B: CW gain @20 mW compresses with eps: {:.2f} (off) > {:.2f} (half) > "
          "{:.2f} dB (full) -> {}".format(G_none, G_half, G_full, "PASS" if g_b else "FAIL"),
          flush=True)

    # ---- GATE C: two timescales (SHB/CH sub-ps vs carrier reservoir ps) ----
    # fast scale: compression-depth recovery in a short lumped section at fine dt
    soa_fast = TravelingWaveSOA(qd, 2.5e-6, 1, nu_s_Hz=nu0)
    dtf = soa_fast.dt
    ntf = int(40e-12 / dtf)
    tf = np.arange(ntf) * dtf
    Pf = 2.0e-4 + 8.0e-2 * np.exp(-0.5 * ((tf - 4e-12) / 3e-13) ** 2)   # 0.3 ps pump
    h = soa_fast.amplify(Pf, 40e-3, return_traces=True, ultrafast=uf)["h_uf"]
    i0 = int(np.argmax(h))
    base = h[-1]
    tgt = base + (h[i0] - base) * np.exp(-1.0)
    ab = np.where(h[i0:] <= tgt)[0]
    tau_uf = float(tf[i0 + ab[0]] - tf[i0]) if ab.size else np.inf
    # slow scale: carrier-reservoir gain recovery (no ultrafast), standard section
    nts = int(0.9e-9 / soa.dt)
    ts = np.arange(nts) * soa.dt
    Ps = 5.0e-4 + 2.0e-2 * np.exp(-0.5 * ((ts - 0.3e-9) / 4e-12) ** 2)
    gm = soa.amplify(Ps, 40e-3, return_traces=True)["g_zt"].mean(axis=1)
    gpre = gm[(ts > 0.2e-9) & (ts < 0.29e-9)].mean()
    j0 = int(np.argmin(gm))
    tc = gm[j0] + (1.0 - np.exp(-1.0)) * (gpre - gm[j0])
    abc = np.where(gm[j0:] >= tc)[0]
    tau_c = float(ts[j0 + abc[0]] - ts[j0]) if abc.size else np.inf
    g_c = bool(tau_uf < 1.5e-12 and tau_uf < 0.3 * tau_c and np.isfinite(tau_c))
    ok = ok and g_c
    print("[uf] GATE C: SHB+CH compression recovery tau_uf={:.2f} ps (~tau_CH=0.70) << carrier "
          "reservoir recovery tau_c={:.1f} ps -> {}".format(
              tau_uf * 1e12, tau_c * 1e12, "PASS" if g_c else "FAIL"), flush=True)

    print("[uf] *** QD-SOA ULTRAFAST COMPRESSION: {} ***".format("PASS" if ok else "FAIL"),
          flush=True)
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
