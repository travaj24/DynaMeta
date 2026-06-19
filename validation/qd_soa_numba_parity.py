"""QD-SOA numba carrier-step accelerator: bit-parity with the numpy reference + speedup (SOA
speedup audit). The traveling-wave marcher spends ~70% of its time in the per-step carrier RK4
(qd_gain.step_slices); QDGainModel(fast=True) swaps in a compiled twin (_qd_carrier_rk4_numba)
that mirrors rhs_fields term-for-term. This gate proves the fast path does NOT change the physics
and measures the win, so the default (numpy) and the accelerator stay interchangeable.

GATE A (single-step parity): one step_slices call, fast vs numpy, over a saturating power range
        and both ng = 1 and ng = 41 -- max relative difference at machine precision.
GATE B (full-marcher parity): a complete amplify() and amplify_coherent(line_filter=True) run,
        fast vs numpy -- the nt-step accumulation stays at machine precision (no drift).
GATE C (speedup): the compiled carrier step is materially faster than numpy (report x; require a
        real speedup, not a regression). SKIPS cleanly (PASS) if numba is unavailable.

Run: python -m validation.qd_soa_numba_parity
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dynameta.optics.soa.qd_gain import _HAVE_NUMBA, QDGainModel, QDGainParams
from dynameta.optics.soa.traveling_wave import TravelingWaveSOA


def main():
    print("[nb] === QD-SOA numba carrier-step: parity vs numpy + speedup ===", flush=True)
    if not _HAVE_NUMBA:
        print("[nb] SKIP: numba not installed -- the fast path is unavailable here; the numpy "
              "reference is the only backend. (pip install numba to enable the ~5x accelerator.)",
              flush=True)
        print("[nb] *** QD-SOA NUMBA PARITY: SKIP (PASS) ***", flush=True)
        return True
    ok = True

    # ---- GATE A: single-step parity over a saturating power range, ng in {1, 41} ----
    worst_a = 0.0
    for ng in (1, 41):
        qd0 = QDGainModel(QDGainParams(n_groups=ng).with_detailed_balance_taus())
        qd1 = QDGainModel(QDGainParams(n_groups=ng).with_detailed_balance_taus(), fast=True)
        nu0 = qd0.p.nu0_Hz
        st = qd0.init_slices(50, 40e-3)
        for P_W in (0.0, 1e-4, 1e-3, 1e-2, 5e-2):                # transparency -> deep saturation
            P = np.full(50, P_W)
            a = qd0.step_slices(st, P, 1.4e-13, nu0, 40e-3)
            b = qd1.step_slices(st, P, 1.4e-13, nu0, 40e-3)
            for x, y in zip(a, b):
                denom = max(float(np.max(np.abs(x))), 1e-300)
                worst_a = max(worst_a, float(np.max(np.abs(x - y))) / denom)
    g_a = bool(worst_a < 1e-12)
    ok = ok and g_a
    print("[nb] GATE A: single-step fast==numpy over [0..50 mW], ng in {{1,41}} (max rel "
          "{:.2e}) -> {}".format(worst_a, "PASS" if g_a else "FAIL"), flush=True)

    # ---- GATE B: full-marcher parity (no nt-step drift), incl. the spectral line filter ----
    qd0 = QDGainModel(QDGainParams(n_groups=41).with_detailed_balance_taus())
    qd1 = QDGainModel(QDGainParams(n_groups=41).with_detailed_balance_taus(), fast=True)
    nu0 = qd0.p.nu0_Hz
    s0 = TravelingWaveSOA(qd0, 0.6e-3, 50, nu_s_Hz=nu0)
    s1 = TravelingWaveSOA(qd1, 0.6e-3, 50, nu_s_Hz=nu0)
    nt = int(2.0e-9 / s0.dt)
    P = np.full(nt, 5e-3)
    rel_pw = abs(s0.amplify(P, drive=40e-3)["P_out"][-1]
                 - s1.amplify(P, drive=40e-3)["P_out"][-1]) / s0.amplify(P, drive=40e-3)["P_out"][-1]
    A = np.full(nt, np.sqrt(2e-3)) + 0j
    c0 = s0.amplify_coherent(A, drive=40e-3, alpha_lef=2.0, line_filter=True)["A_out"]
    c1 = s1.amplify_coherent(A, drive=40e-3, alpha_lef=2.0, line_filter=True)["A_out"]
    rel_lf = float(np.max(np.abs(c0 - c1)) / np.max(np.abs(c0)))
    g_b = bool(rel_pw < 1e-11 and rel_lf < 1e-11)
    ok = ok and g_b
    print("[nb] GATE B: full marcher fast==numpy -- amplify rel {:.2e}, line_filter rel {:.2e} "
          "(no nt-step drift) -> {}".format(rel_pw, rel_lf, "PASS" if g_b else "FAIL"), flush=True)

    # ---- GATE C: the compiled step is materially faster ----
    qd0 = QDGainModel(QDGainParams(n_groups=41).with_detailed_balance_taus())
    qd1 = QDGainModel(QDGainParams(n_groups=41).with_detailed_balance_taus(), fast=True)
    st = qd0.init_slices(50, 40e-3)
    Pc = np.full(50, 5e-3)
    qd1.step_slices(st, Pc, 1.4e-13, nu0, 40e-3)                 # warm up the JIT
    reps = 1500

    def bench(qd):
        t0 = time.perf_counter()
        for _ in range(reps):
            qd.step_slices(st, Pc, 1.4e-13, nu0, 40e-3)
        return time.perf_counter() - t0
    t_np = min(bench(qd0) for _ in range(2))
    t_nb = min(bench(qd1) for _ in range(2))
    speedup = t_np / max(t_nb, 1e-12)
    g_c = bool(speedup > 1.5)
    ok = ok and g_c
    print("[nb] GATE C: compiled carrier step {:.1f}x faster ({:.0f} vs {:.0f} us/call) -> "
          "{}".format(speedup, 1e6 * t_np / reps, 1e6 * t_nb / reps, "PASS" if g_c else "FAIL"),
          flush=True)

    print("[nb] *** QD-SOA NUMBA PARITY: {} ***".format("PASS" if ok else "FAIL"), flush=True)
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
