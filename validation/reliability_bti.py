"""REL2 NBTI/PBTI oracle. dVth = A D^p (E/E_ref)^gamma t^n exp(-Ea/kBT); t_fail = inverse.

GATE A (reduces-to-closed-form): the log-log time slope of dVth(t) equals n to machine over six
        decades; t = 0 -> exactly zero drift; the Arrhenius/field ratios match closed forms; the
        activated-process SIGN is right (hotter degrades FASTER: dVth rises with T, lifetime falls).
GATE B (independent numeric inversion): time_to_dvth vs a scipy.brentq root of dvth(t) - dvth_max
        (numeric root-find vs the closed-form inversion -- two pathways).
GATE C (sane-band anchor): calibrate A on a typical accelerated point (50 mV after 1000 h at 125 degC,
        5 MV/cm) and extrapolate to 10 years at 85 degC / 3.3 MV/cm -- the use-condition drift must
        land in a physically sane 1-200 mV band and be monotonic in T and E.

Run: python -m validation.reliability_bti
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dynameta.reliability.bti import BtiParams, dvth_power_law, time_to_dvth, KB_EV_K

MV = 1.0e8
HOUR = 3600.0
YEAR = 365.25 * 24 * HOUR


def main():
    print("[rb] === REL2 NBTI/PBTI power-law drift ===", flush=True)
    ok = True
    P = dict(A_V=2.0e-3, n_exp=1.0 / 6.0, gamma=0.35, Ea_eV=0.12)

    # ---- GATE A: time slope == n; t=0 -> 0; ratio closed forms; T sign ----
    ts = np.geomspace(1.0, 1.0e6, 13)
    d = dvth_power_law(ts, 5 * MV, 398.0, **P)
    slopes = np.diff(np.log(d)) / np.diff(np.log(ts))
    zero = float(dvth_power_law(0.0, 5 * MV, 398.0, **P))
    rT = float(dvth_power_law(1e5, 5 * MV, 398.0, **P) / dvth_power_law(1e5, 5 * MV, 358.0, **P))
    rT_an = float(np.exp(-P["Ea_eV"] / KB_EV_K * (1.0 / 398.0 - 1.0 / 358.0)))
    rE = float(dvth_power_law(1e5, 6 * MV, 398.0, **P) / dvth_power_law(1e5, 3 * MV, 398.0, **P))
    rE_an = 2.0 ** P["gamma"]
    g_a = (float(np.max(np.abs(slopes - P["n_exp"]))) < 1e-12 and zero == 0.0
           and abs(rT / rT_an - 1) < 1e-12 and abs(rE / rE_an - 1) < 1e-12 and rT > 1.0)
    ok = ok and bool(g_a)
    print("[rb] GATE A: slope == n ({:.6f}), dVth(0) = {}, T-ratio {:.4f} (>1: hotter faster), "
          "E-ratio {:.4f} -> {}".format(float(np.mean(slopes)), zero, rT, rE,
                                        "PASS" if g_a else "FAIL"), flush=True)

    # ---- GATE B: closed-form inversion vs brentq numeric root ----
    from scipy.optimize import brentq
    dmax = 0.05
    t_cf = time_to_dvth(dmax, 5 * MV, 398.0, **P)
    t_num = float(brentq(lambda t: float(dvth_power_law(t, 5 * MV, 398.0, **P)) - dmax,
                         1e-6, 1e3 * t_cf, rtol=1e-13))
    relB = abs(t_num - t_cf) / t_cf
    g_b = bool(relB < 1e-9)
    ok = ok and g_b
    print("[rb] GATE B: t_fail closed-form {:.6e} s vs brentq {:.6e} s (rel {:.1e}) -> {}".format(
        t_cf, t_num, relB, "PASS" if g_b else "FAIL"), flush=True)

    # ---- GATE C: calibrated accelerated point -> 10-year use-condition band ----
    cal = BtiParams.calibrated(t_s=1000 * HOUR, E_ox_V_m=5 * MV, T_K=398.15, dvth_V=0.050)
    anchor = float(cal.dvth_V(1000 * HOUR, 5 * MV, 398.15))
    use = float(cal.dvth_V(10 * YEAR, 3.3 * MV, 358.15))
    g_c = bool(abs(anchor - 0.050) / 0.050 < 1e-12 and 1e-3 < use < 0.2
               and use < float(cal.dvth_V(10 * YEAR, 5 * MV, 398.15)))
    ok = ok and g_c
    print("[rb] GATE C: anchor 50 mV reproduced; 10-yr use (85 degC, 3.3 MV/cm) dVth = {:.1f} mV in "
          "(1, 200) mV, monotonic in stress -> {}".format(use * 1e3, "PASS" if g_c else "FAIL"),
          flush=True)

    print("[rb] *** REL2 BTI: {} ***".format("PASS" if ok else "FAIL"), flush=True)
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
