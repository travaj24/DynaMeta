"""REL7 stress / thermal-gradient migration oracle (Korhonen PDE + Soret flux).

GATE A (the strong reduces-to gate): at constant kappa and EARLY time (kappa t << L^2, before the
        blocked far end is felt) the Korhonen relaxation from uniform sigma0 with a stress-free via
        has the EXACT semi-infinite solution sigma = sigma0 * erf(x / (2 sqrt(kappa t))) -- the
        method-of-lines PDE must match it.
GATE B (Arrhenius + thresholds): kappa(T2)/kappa(T1) = (T1/T2) exp(-(Q/kB)(1/T2 - 1/T1)) closed form
        (the 1/T prefactor included); sigma_crit = inf -> never nucleates; sigma_crit <= sigma0 ->
        the initial residual already nucleates.
GATE C (Soret): the flux matches the closed form to machine and points DOWN the gradient for
        Q* > 0.

Run: python -m validation.reliability_stressmig
"""
import os
import sys

import numpy as np
from scipy.special import erf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dynameta.reliability.stress_migration import (korhonen_kappa_m2_s, korhonen_relax,
                                                   void_nucleates, soret_flux_per_m2_s,
                                                   KB_EV_K, KB_J_K)


def main():
    print("[rs] === REL7 stress / thermal-gradient migration (Korhonen) ===", flush=True)
    ok = True

    # ---- GATE A: erf reduction ----
    L, n = 20e-6, 401
    x = np.linspace(0.0, L, n)
    kappa, sigma0 = 1.0e-13, 2.0e8
    t = 0.02 * L ** 2 / kappa                               # Fourier number 0.02 (semi-infinite OK)
    sig = korhonen_relax(x, t, sigma0_Pa=sigma0, kappa_m2_s=kappa)
    sig_an = sigma0 * erf(x / (2.0 * np.sqrt(kappa * t)))
    relA = float(np.max(np.abs(sig - sig_an)) / sigma0)
    g_a = bool(relA < 1e-3)
    ok = ok and g_a
    print("[rs] GATE A: Korhonen PDE vs erf(x/2 sqrt(kappa t)) at Fo=0.02: rel-to-sigma0 {:.2e} "
          "-> {}".format(relA, "PASS" if g_a else "FAIL"), flush=True)

    # ---- GATE B: Arrhenius kappa ratio + nucleation thresholds ----
    kw = dict(D0_m2_s=1e-6, Q_eV=0.9, B_Pa=1.0e11, Omega_m3=1.18e-29)
    r = korhonen_kappa_m2_s(400.0, **kw) / korhonen_kappa_m2_s(330.0, **kw)
    r_an = (330.0 / 400.0) * np.exp(-(0.9 / KB_EV_K) * (1.0 / 400.0 - 1.0 / 330.0))
    g_b = bool(abs(r / r_an - 1) < 1e-12
               and not void_nucleates(sig, float("inf"))
               and void_nucleates(np.full(4, sigma0), 0.5 * sigma0))
    ok = ok and g_b
    print("[rs] GATE B: kappa(400)/kappa(330) = {:.1f} == closed form ((T1/T2) exp Arrhenius); "
          "thresholds behave -> {}".format(r, "PASS" if g_b else "FAIL"), flush=True)

    # ---- GATE C: Soret flux closed form + direction ----
    C0, T, gT, Da, Qs = 8.0e28, 350.0, 1.0e7, 1.0e-18, 0.8
    J = soret_flux_per_m2_s(C0, T, gT, D_a_m2_s=Da, Qstar_eV=Qs)
    J_an = -(Da * C0 / (KB_J_K * T)) * (Qs * 1.602176634e-19 / T) * gT
    g_c = bool(abs(J / J_an - 1) < 1e-12 and J < 0.0)        # down the (positive) gradient
    ok = ok and g_c
    print("[rs] GATE C: Soret flux {:.3e} atoms/(m^2 s) == closed form, down-gradient -> {}".format(
        J, "PASS" if g_c else "FAIL"), flush=True)

    print("[rs] *** REL7 STRESS MIGRATION: {} ***".format("PASS" if ok else "FAIL"), flush=True)
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
