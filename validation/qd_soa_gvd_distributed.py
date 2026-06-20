"""QD-SOA DISTRIBUTED group-velocity dispersion (multi-segment split-step) vs convergence oracles.
amplify_coherent(gvd_segments=S) refines the single device-scale GVD Strang split into S sub-sections
that INTERLEAVE dispersion and gain S times [D(L/2S) . N(L/S) . D(L/S) . ... . N(L/S) . D(L/2S)], so
the distributed dispersion-gain coupling -- the running FWM phase-matching and in-device pulse
reshaping that the single endpoint split (S=1) only approximates -- is captured and CONTROLLED. This
redeems the earlier honest caveat: the splitting error is now a verifiable 2nd-order (Strang) quantity.

GATE A (reduction): gvd_segments=1 == the single device-scale split (byte-identical); the no-arg
        default is gvd_segments=1.
GATE B (linear-limit S-invariance): gain-free, dispersion commutes with the delay, so S=1,2,4,8 are
        all identical to machine precision AND equal the analytic Gaussian broadening (independent of
        the segmentation -- the coupling exists only with a z-varying saturating gain).
GATE C (2nd-order Strang convergence): with a saturating pulse + dispersion the successive-refinement
        differences ||A_S - A_2S|| fall by ~4x per doubling (error ~ O(1/S^2)) -- the asymptotic
        ratios approach 4, the textbook symmetric-split order. This is the testable order claim.
GATE D (the distributed coupling is REAL): in a strong-coupling regime S=1 differs from the converged
        result by an APPRECIABLE amount (the endpoint split genuinely missed it), while the high-S
        tail is Cauchy (||A_16 - A_32|| << ||A_1 - A_32||) -- so the refinement converges to a
        well-defined distributed limit that the single split did not reach.
GATE E (passivity): every S stays finite (no NaN) and gain-free conserves the pulse energy for ALL S
        (the dispersion sub-steps are unitary).

Run: python -m validation.qd_soa_gvd_distributed
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dynameta.optics.soa.qd_gain import QDGainModel, QDGainParams
from dynameta.optics.soa.traveling_wave import TravelingWaveSOA, TwoLevelSaturableGain


def _rms(t, w):
    W = np.sum(w)
    m1 = np.sum(t * w) / W
    return float(np.sqrt(max(np.sum(t * t * w) / W - m1 * m1, 0.0)))


def main():
    print("[gd] === QD-SOA distributed GVD (multi-segment split-step) vs convergence oracles ===",
          flush=True)
    ok = True

    # ---- GATE A + B: gain-free passive slab (linear limit) ----
    L, nz = 1.0e-3, 256
    eng0 = TravelingWaveSOA(TwoLevelSaturableGain(0.0, 1e-9, 1e-12), L, nz)
    dt, W = eng0.dt, nz * eng0.dt
    nt = 4 * nz
    t = np.arange(nt) * dt
    tc = (nt // 2) * dt
    T0 = W / 16.0
    beta2 = T0 * T0 / L
    Ag = np.exp(-((t - tc) ** 2) / (2.0 * T0 * T0)) + 0j

    base = eng0.amplify_coherent(Ag, None, beta2_s2_per_m=beta2)["A_out"]              # single split
    s1 = eng0.amplify_coherent(Ag, None, beta2_s2_per_m=beta2, gvd_segments=1)["A_out"]
    g_a = bool(np.array_equal(base, s1))
    ok = ok and g_a
    print("[gd] GATE A: gvd_segments=1 == single device-scale split (byte-identical {}) -> {}".format(
        g_a, "PASS" if g_a else "FAIL"), flush=True)

    lin = {S: eng0.amplify_coherent(Ag, None, beta2_s2_per_m=beta2, gvd_segments=S)["A_out"]
           for S in (1, 2, 4, 8)}
    s_inv = max(float(np.max(np.abs(lin[S] - lin[1]))) for S in (2, 4, 8))
    ratio_law = float(np.sqrt(2.0))                                                   # L/L_D = 1
    rms_in = _rms(t, np.abs(Ag) ** 2)
    rel_law = abs(_rms(t, np.abs(lin[8]) ** 2) / rms_in - ratio_law) / ratio_law
    g_b = bool(s_inv < 1e-12 and rel_law < 2e-2)
    ok = ok and g_b
    print("[gd] GATE B: gain-free S-invariant (max|S-1| {:.1e}) AND == NLSE broadening (rel {:.1e}) "
          "-> {}".format(s_inv, rel_law, "PASS" if g_b else "FAIL"), flush=True)

    # ---- GATE C + D + E: strong dispersion-gain coupling regime ----
    m = QDGainModel(QDGainParams(n_groups=15).with_detailed_balance_taus())
    L2, nz2 = 2.0e-3, 128
    eng = TravelingWaveSOA(m, L2, nz2, nu_s_Hz=m.p.nu0_Hz)
    dt2 = eng.dt
    nt2 = 1500
    t2 = np.arange(nt2) * dt2
    tc2 = (nt2 // 2) * dt2
    T0b = 12.0 * dt2
    b2 = 3.0e-21
    f = 4.0 / (nt2 * dt2)
    A2 = (np.sqrt(0.6) * np.exp(-((t2 - tc2) ** 2) / (2.0 * T0b * T0b))
          * np.exp(-1j * 2.0 * np.pi * f * (t2 - tc2))).astype(np.complex128)
    Sset = (1, 2, 4, 8, 16, 32)
    out = {S: eng.amplify_coherent(A2, 40e-3, beta2_s2_per_m=b2, gvd_segments=S)["A_out"]
           for S in Sset}
    nan_free = not any(np.isnan(o).any() for o in out.values())

    def dmax(a, b):
        return float(np.max(np.abs(a - b)))
    diffs = [dmax(out[Sset[i]], out[Sset[i + 1]]) for i in range(len(Sset) - 1)]   # 1-2,2-4,...,16-32
    ratios = [diffs[i] / diffs[i + 1] for i in range(len(diffs) - 1)]              # ->4 (2nd order)
    asymp = ratios[-2:]                                                            # the two cleanest
    g_c = bool(nan_free and all(3.3 < r < 4.7 for r in asymp))
    ok = ok and g_c
    print("[gd] GATE C: 2nd-order Strang convergence, successive-diff ratios {} (->4); asymptotic "
          "{} -> {}".format(["{:.2f}".format(r) for r in ratios],
                            ["{:.2f}".format(r) for r in asymp], "PASS" if g_c else "FAIL"),
          flush=True)

    scale = float(np.max(np.abs(out[32])))
    coupling = dmax(out[1], out[32]) / scale                  # single-split error vs converged
    cauchy = dmax(out[16], out[32]) / scale                   # high-S tail (must be << coupling)
    g_d = bool(coupling > 1e-2 and cauchy < 0.1 * coupling)
    ok = ok and g_d
    print("[gd] GATE D: distributed coupling REAL -- S=1 vs converged rel {:.2e} (>1e-2), high-S "
          "tail {:.2e} << that -> {}".format(coupling, cauchy, "PASS" if g_d else "FAIL"), flush=True)

    # gain-free energy conservation for every S (unitary dispersion sub-steps)
    E_in = float(np.sum(np.abs(Ag) ** 2))
    relE = max(abs(float(np.sum(np.abs(lin[S]) ** 2)) - E_in) / E_in for S in (1, 2, 4, 8))
    g_e = bool(nan_free and relE < 1e-9)
    ok = ok and g_e
    print("[gd] GATE E: all S finite ({}) + gain-free energy conserved for every S (max rel {:.1e}) "
          "-> {}".format(nan_free, relE, "PASS" if g_e else "FAIL"), flush=True)

    print("[gd] *** QD-SOA DISTRIBUTED GVD: {} ***".format("PASS" if ok else "FAIL"), flush=True)
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
