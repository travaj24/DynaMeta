"""R12 PCM classical-nucleation + growth oracle (beyond the shipped fixed-exponent JMAK).

GATE A (isothermal Avrami n = 4, reduces-to-closed-form): constant T -> the O(n) moment-scheme
        integrate() must reproduce X = 1 - exp(-(pi/3) I u^3 t^4) (trapezoid-limited, rel < 1e-4);
        the log-log Avrami slope d ln(-ln(1-X))/d ln t == 4.
GATE B (the nucleation NOSE): I(T) peaks STRICTLY BETWEEN Tg and Tm, collapses at Tm (the CNT
        barrier W* ~ 1/dG_v^2 diverges as the driving force vanishes) and is exponentially
        suppressed toward Tg; EXACTLY zero outside (Tg, Tm) (mask, not clip).
GATE C (growth-only n = 3): I0 = 0 with pre-existing nuclei N0 -> X = 1 - exp(-(4 pi/3) N0 u^3 t^3)
        EXACTLY (the cube of the cumulative growth length; Avrami slope == 3).
GATE D (cross-model consistency): the equivalent JMAK rate K_eff = ((pi/3) I u^3)^(1/4) fed to the
        SHIPPED PCMSwitching with avrami_n = 4 reproduces the same isothermal trajectory.
GATE E (state machine + off-switch): melt-quench resets to amorphous; frozen below Tg holds;
        enabled=False returns x0 EXACTLY.

Run: python -m validation.pcm_nucleation_growth
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dynameta.carriers.switching import PCMClassicalNucleation, PCMSwitching

# GST-scale CNT parameters: sigma ~ 0.075 J/m^2, volumetric heat of fusion ~ 6.2e8 J/m^3,
# atomic volume ~ 2.9e-29 m^3 -- W*(700 K) ~ 2.3 eV (the right CNT barrier scale).
P = dict(I0_per_m3_s=1.0e39, sigma_J_m2=0.075, dHf_J_m3=6.2e8, Omega_m3=2.9e-29,
         u0_m_s=1.0e3, Ea_d_J=2.0 * 1.602176634e-19, Ea_g_J=1.5 * 1.602176634e-19,
         T_glass_K=450.0, T_melt_K=900.0)


def _avrami_slope(t, X):
    m = (X > 1e-6) & (X < 0.99)
    y = np.log(-np.log(1.0 - X[m]))
    return float(np.polyfit(np.log(t[m]), y, 1)[0])


def main():
    print("[pn] === R12 PCM classical nucleation + growth (KJMA) ===", flush=True)
    ok = True
    pcm = PCMClassicalNucleation(**P)
    T0 = 700.0
    I = float(pcm.nucleation_rate_I(T0))
    u = float(pcm.growth_velocity_u(T0))
    t_c = (3.0 / (np.pi * I * u ** 3)) ** 0.25                  # X ~ 63% crystallization time

    # ---- GATE A: isothermal n=4 closed form ----
    t = np.linspace(0.0, 1.6 * t_c, 6001)
    x_num = pcm.integrate(t, np.full_like(t, T0))
    x_cf = pcm.fraction_isothermal(t, T0)
    relA = float(np.max(np.abs(x_num - x_cf)))
    slope4 = _avrami_slope(t[1:], x_num[1:])
    g_a = bool(relA < 1e-4 and abs(slope4 - 4.0) < 0.02)
    ok = ok and g_a
    print("[pn] GATE A: isothermal moment scheme vs closed form max|d| = {:.1e}; Avrami slope "
          "{:.3f} == 4 -> {}".format(relA, slope4, "PASS" if g_a else "FAIL"), flush=True)

    # ---- GATE B: the nucleation nose ----
    Tg, Tm = P["T_glass_K"], P["T_melt_K"]
    Ts = np.linspace(Tg + 1.0, Tm - 1.0, 600)
    Is = np.asarray(pcm.nucleation_rate_I(Ts))
    ipk = int(np.argmax(Is))
    interior = 0 < ipk < Ts.size - 1
    collapse_melt = Is[-1] < 1e-30 * Is[ipk]
    suppressed_glass = Is[0] < 1e-2 * Is[ipk] and bool(np.all(np.diff(Is[:ipk + 1]) > 0))
    masked = (float(pcm.nucleation_rate_I(Tg - 10.0)) == 0.0
              and float(pcm.nucleation_rate_I(Tm + 10.0)) == 0.0
              and float(pcm.growth_velocity_u(Tm + 10.0)) == 0.0)
    g_b = bool(interior and collapse_melt and suppressed_glass and masked)
    ok = ok and g_b
    print("[pn] GATE B: nose peak at {:.0f} K (interior); I(Tm-1)/I(peak) = {:.1e}; I(Tg+1)/I(peak) "
          "= {:.1e}; zero outside the window -> {}".format(
              Ts[ipk], Is[-1] / Is[ipk], Is[0] / Is[ipk], "PASS" if g_b else "FAIL"), flush=True)

    # ---- GATE C: growth-only n=3 (pre-existing nuclei) ----
    N0 = 1.0e20
    pcm3 = PCMClassicalNucleation(**{**P, "I0_per_m3_s": 0.0, "N0_per_m3": N0})
    t3c = (3.0 / (4.0 * np.pi * N0 * u ** 3)) ** (1.0 / 3.0)
    t3 = np.linspace(0.0, 1.6 * t3c, 6001)
    x3 = pcm3.integrate(t3, np.full_like(t3, T0))
    x3_cf = 1.0 - np.exp(-(4.0 * np.pi / 3.0) * N0 * u ** 3 * t3 ** 3)
    relC = float(np.max(np.abs(x3 - x3_cf)))
    slope3 = _avrami_slope(t3[1:], x3[1:])
    g_c = bool(relC < 1e-4 and abs(slope3 - 3.0) < 0.02)
    ok = ok and g_c
    print("[pn] GATE C: growth-only vs (4 pi/3) N0 u^3 t^3 max|d| = {:.1e}; Avrami slope {:.3f} "
          "== 3 -> {}".format(relC, slope3, "PASS" if g_c else "FAIL"), flush=True)

    # ---- GATE D: equivalent-JMAK cross-model consistency ----
    K_eff = ((np.pi / 3.0) * I * u ** 3) ** 0.25
    jmak = PCMSwitching(K0_per_s=K_eff, E_a_J=1e-30, T_glass_K=Tg, T_melt_K=Tm, avrami_n=4.0)
    x_jmak = jmak.integrate(t, np.full_like(t, T0))
    relD = float(np.max(np.abs(x_num - x_jmak)))
    g_d = bool(relD < 1e-4)
    ok = ok and g_d
    print("[pn] GATE D: CNT vs equivalent JMAK (K_eff = ((pi/3) I u^3)^(1/4), n=4) max|d| = {:.1e} "
          "-> {}".format(relD, "PASS" if g_d else "FAIL"), flush=True)

    # ---- GATE E: melt-quench / frozen / off-switch ----
    tq = np.linspace(0.0, 1.0 * t_c, 2001)
    Tq = np.full_like(tq, T0)
    Tq[1200:1300] = Tm + 50.0                                  # melt spike
    Tq[1700:] = Tg - 50.0                                    # then freeze
    xq = pcm.integrate(tq, Tq)
    pre = xq[1199]
    during = float(np.max(xq[1200:1300]))
    frozen_flat = float(np.max(np.abs(np.diff(xq[1700:]))))
    off = PCMClassicalNucleation(**{**P, "enabled": False}).integrate(tq, Tq, x0=0.37)
    g_e = bool(pre > 0.05 and during == 0.0 and frozen_flat == 0.0 and np.all(off == 0.37))
    ok = ok and g_e
    print("[pn] GATE E: melt resets (x {:.3f} -> 0), frozen holds exactly, enabled=False == x0 "
          "exactly -> {}".format(pre, "PASS" if g_e else "FAIL"), flush=True)

    print("[pn] *** R12 PCM NUCLEATION + GROWTH: {} ***".format("PASS" if ok else "FAIL"), flush=True)
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
