"""Validate the time-domain nematic director DYNAMICS (Erickson-Leslie relaxation) added in
dynameta/carriers/lc_dynamics.py -- the LC switching-speed axis DynaMeta previously lacked. Independent
oracles: the analytic single-relaxation time tau = gamma1 d^2 / (K pi^2) (1-constant, small field-OFF
perturbation) and the GOLDEN external solver (lc_dynamics_base.py) step-pulse rise/decay times.

GATE A (analytic tau): a small field-OFF perturbation of theta_mid in a 1-constant cell decays as
        exp(-t/tau); the fitted decay constant matches gamma1 d^2/(K pi^2) to < 1%.
GATE B (golden rise/decay): a step pulse (planar uniform, gap 1um, K11=17pN, K33=18pN, eps_para=18.7,
        eps_perp=4.0, theta_b=89.9 deg, V0=2 V, Ton=4 ms, T_end=12 ms, n_o=1.56, n_e=1.92) reproduces the
        external solver's n_eff 10-90 rise = 0.612 ms and 90-10 decay = 0.595 ms to < 3%.
GATE C (turn-on/off physics): the pulse drives n_eff UP toward the homeotropic value (>= 1.76 at the peak,
        the external max) while ON, and it RELAXES back to the planar n_o by T_end (within 1e-3).

Run: python -m validation.lc_director_dynamics
"""
import os
import sys
import math

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dynameta.carriers.lc_dynamics import LCDynamics


def main():
    print("[ld] === Nematic director DYNAMICS (Erickson-Leslie relaxation) ===", flush=True)

    # GATE A: analytic relaxation time tau = gamma1 d^2 / (K pi^2)
    dc = LCDynamics(K11=10e-12, K33=10e-12, gamma1=0.05, eps_para=10.0, eps_perp=5.0,
                    theta_b_rad=0.5 * math.pi, geometry="planar", d_planar=2e-6,
                    field_model="uniform", nz=121)
    tau = dc.tau_1const_s()
    z = dc.geometry_obj().z_m
    th0 = 0.5 * math.pi - math.radians(2.0) * np.sin(math.pi * z / z[-1])   # small field-OFF bump
    t_eval = np.linspace(0.0, 5.0 * tau, 200)
    rr = dc.simulate(t_eval, lambda t: 0.0, theta0_rad=th0)
    amp = 0.5 * math.pi - rr.theta_mid_rad
    m = (t_eval > 0.5 * tau) & (t_eval < 3.0 * tau)
    tau_fit = -1.0 / np.polyfit(t_eval[m], np.log(amp[m]), 1)[0]
    g_a = abs(tau_fit / tau - 1.0) < 1e-2
    print("[ld] A relaxation tau: predicted={:.4f} ms, fitted-decay={:.4f} ms, ratio={:.4f} -> {}"
          .format(tau * 1e3, tau_fit * 1e3, tau_fit / tau, "OK" if g_a else "FAIL"), flush=True)

    # GATE B + C: golden step-pulse rise/decay + turn-on/off physics
    d = LCDynamics(K11=17e-12, K33=18e-12, gamma1=0.085, eps_para=18.7, eps_perp=4.0,
                   theta_b_rad=math.radians(89.9), geometry="planar", d_planar=1e-6,
                   field_model="uniform", n_o=1.56, n_e=1.92, nz=121)
    r = d.simulate_pulse(V0=2.0, Ton=4e-3, T_end=12e-3, n_t=400, waveform="step")
    rise_ms, decay_ms = r.rise_10_90_s * 1e3, r.decay_90_10_s * 1e3
    g_b = (abs(rise_ms / 0.6125 - 1.0) < 0.03) and (abs(decay_ms / 0.5947 - 1.0) < 0.03)
    print("[ld] B golden step pulse: rise_10_90={:.4f} ms (gold 0.6125), decay_90_10={:.4f} ms "
          "(gold 0.5947) -> {}".format(rise_ms, decay_ms, "OK" if g_b else "FAIL"), flush=True)

    n_peak = float(np.nanmax(r.n_eff)); n_end = float(r.n_eff[-1])
    g_c = (n_peak >= 1.76) and (abs(n_end - 1.56) < 1e-3)
    print("[ld] C turn-on/off: n_eff peak={:.5f} (>=1.76 homeotropic-ward), relaxes to end={:.5f} "
          "(~n_o=1.56) -> {}".format(n_peak, n_end, "OK" if g_c else "FAIL"), flush=True)

    ok = g_a and g_b and g_c
    print("[ld] *** LC DIRECTOR DYNAMICS: {} ***".format("PASS" if ok else "FAIL"), flush=True)
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
