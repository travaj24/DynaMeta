"""REL10 acceleration-factor + system-MTTF oracle. The umbrella over REL1-REL9.

GATE A (reduces-to-closed-form): the Arrhenius AF PARENTHESIZATION is (Ea/kB)(1/Tu - 1/Ts) (the
        mis-grouped Ea/(kB(1/Tu-1/Ts)) overflows -- the audit-flagged bug class); a single mechanism
        reduces the system to itself; immortal (inf) mechanisms drop out; all-immortal -> inf.
GATE B (independent MONTE CARLO, competing risks): the closed form 1/MTTF_sys = sum 1/MTTF_i must
        match the sample mean of min(Exp(m1), Exp(m2), Exp(m3)) over 4e5 draws -- a genuinely
        independent stochastic reference, not algebra.
GATE C (independent MONTE CARLO, array weakest link): the Weibull order statistic
        t63_min = t63 / N^(1/beta) must match the empirical 63.2% quantile of min over N iid Weibull
        draws.
GATE D (chain integration with REL1): extrapolating a stress MTTF to use conditions via
        arrhenius_af * field_af must equal the TDDB model's OWN tBD ratio for the same parameters
        (two independent code paths through the same physics).

Run: python -m validation.reliability_mttf
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dynameta.reliability.mttf import (arrhenius_af, field_af, mttf_use_from_stress, system_mttf,
                                       fit_per_1e9_hours, weibull_earliest_t63, KB_EV_K)
from dynameta.reliability.tddb import tbd_e_model

MV = 1.0e8


def main():
    print("[rm] === REL10 acceleration factors + system MTTF ===", flush=True)
    ok = True

    # ---- GATE A: parenthesization + reductions ----
    af = arrhenius_af(0.5, 358.0, 398.0)
    af_an = float(np.exp((0.5 / KB_EV_K) * (1.0 / 358.0 - 1.0 / 398.0)))
    g_a = (abs(af / af_an - 1) < 1e-13 and 1.0 < af < 1e3                       # O(5), not overflow
           and np.isclose(system_mttf([3.3e5]), 3.3e5, rtol=1e-12)              # float 1/(1/x) round-trip
           and np.isclose(system_mttf([float("inf"), 2.0e5]), 2.0e5, rtol=1e-12)
           and np.isinf(system_mttf([float("inf"), float("inf")]))
           and fit_per_1e9_hours(float("inf")) == 0.0
           and abs(fit_per_1e9_hours(1.0e5 * 3600.0) - 1.0e4) < 1e-6)
    ok = ok and bool(g_a)
    print("[rm] GATE A: AF(0.5 eV, 358<-398 K) = {:.3f} == closed form (no overflow); single/"
          "immortal/FIT reductions exact -> {}".format(af, "PASS" if g_a else "FAIL"), flush=True)

    # ---- GATE B: Monte-Carlo competing risks ----
    m = np.array([2.0e5, 5.0e5, 1.0e6])
    closed = system_mttf(m)                                  # 1/(5+2+1)e-6 = 125000
    rng = np.random.default_rng(7)
    draws = np.min(rng.exponential(m, size=(400000, 3)), axis=1)
    mc = float(np.mean(draws))
    relB = abs(mc - closed) / closed
    g_b = bool(relB < 6e-3)                                  # SE ~ 0.16%; 3.8-sigma headroom
    ok = ok and g_b
    print("[rm] GATE B: MC mean of min-exponential {:.0f} h vs closed {:.0f} h (rel {:.2e}, 4e5 "
          "draws) -> {}".format(mc, closed, relB, "PASS" if g_b else "FAIL"), flush=True)

    # ---- GATE C: Monte-Carlo Weibull weakest link ----
    t63, n_el, beta = 1.0e6, 64, 2.0
    closed_min = weibull_earliest_t63(t63, n_el, beta)       # 1e6/8 = 125000
    mins = np.min(t63 * rng.weibull(beta, size=(200000, n_el)), axis=1)
    mc_t63 = float(np.quantile(mins, 1.0 - np.exp(-1.0)))    # the 63.2% characteristic life
    relC = abs(mc_t63 - closed_min) / closed_min
    g_c = bool(relC < 1.5e-2)
    ok = ok and g_c
    print("[rm] GATE C: MC 63.2% quantile of min-of-{} Weibull {:.0f} h vs t63/N^(1/beta) {:.0f} h "
          "(rel {:.2e}) -> {}".format(n_el, mc_t63, closed_min, relC, "PASS" if g_c else "FAIL"),
          flush=True)

    # ---- GATE D: AF chain == the TDDB model's own ratio (REL10 x REL1 integration) ----
    Ea, gam = 0.7, 3.0
    mttf_stress = 500.0                                      # measured at (8 MV/cm, 398 K)
    af_chain = arrhenius_af(Ea, 358.0, 398.0) * field_af(gam, 5.0, 8.0)
    mttf_use = mttf_use_from_stress(mttf_stress, af_chain)
    ratio_tddb = float(tbd_e_model(5 * MV, 358.0, tau0_s=1.0, gamma_E_per_MV_cm=gam, Ea_eV=Ea)
                       / tbd_e_model(8 * MV, 398.0, tau0_s=1.0, gamma_E_per_MV_cm=gam, Ea_eV=Ea))
    relD = abs(af_chain - ratio_tddb) / ratio_tddb
    g_d = bool(relD < 1e-12)
    ok = ok and g_d
    print("[rm] GATE D: AF chain {:.4e} == TDDB tBD ratio {:.4e} (rel {:.1e}); use MTTF {:.3e} s "
          "-> {}".format(af_chain, ratio_tddb, relD, mttf_use, "PASS" if g_d else "FAIL"), flush=True)

    print("[rm] *** REL10 MTTF UMBRELLA: {} ***".format("PASS" if ok else "FAIL"), flush=True)
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
