"""Validate the LC deeper-physics additions to dynameta/carriers/lc_director.py: FINITE (Rapini-Papoular)
surface anchoring in the two-constant director BVP, and the Haller temperature dependence of the Frank
elastic constants K(T) and the rotational viscosity gamma1(T). Independent analytic oracles only.

GATE A (strong-anchoring recovery): with a very large anchoring strength W the surface director pins to
        the easy axis (theta_surface -> theta_b) and the result equals the strong-anchoring (W=None) solve
        -- and W=None is byte-unchanged vs the validated golden statics (n_eff at V=2 = 1.76198).
GATE B (finite anchoring + extrapolation length b=K/W): a finite W lets the SURFACE tilt toward the field;
        the surface deviation from the easy axis grows MONOTONICALLY as W decreases (b grows), the
        physical signature of weak anchoring.
GATE C (Haller K(T) / gamma1(T)): the order parameter S(T)=(1-T/T_NI)^beta decreases to 0 at T_NI; the
        Frank constant scales K(T)=K_ref (S/S_ref)^2 (so the Freedericksz V_th ~ sqrt(K) FALLS with T);
        gamma1(T) Arrhenius-falls with T; both reduce to the reference value at T_ref.

Run: python -m validation.lc_weak_anchoring_temperature
"""
import os
import sys
import math

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dynameta.carriers.lc_director import (
    director_profile_bvp, freedericksz_threshold_V,
    haller_order_parameter, K_of_temperature, gamma1_of_temperature)

THB = math.radians(89.9)
BKW = dict(K11=17e-12, K33=18e-12, eps_para=18.7, eps_perp=4.0, d_planar=1e-6, theta_b_rad=THB,
           field_model="uniform", nz=151, n_o=1.56, n_e=1.92)


def _surf_ctr_neff(W):
    r = director_profile_bvp(V_app=2.0, W_anchor_J_m2=W, **BKW)
    th = r.theta_field_rad
    return math.degrees(th[0]), math.degrees(th[th.size // 2]), r.n_eff


def main():
    print("[wa] === LC finite surface anchoring + Haller K(T)/gamma1(T) ===", flush=True)

    # GATE A: strong-anchoring recovery + W=None unchanged
    s_surf, _s_ctr, s_neff = _surf_ctr_neff(None)
    h_surf, _h_ctr, h_neff = _surf_ctr_neff(1e2)                  # b = K/W ~ 1e-13 m << d
    g_a = (abs(s_surf - 89.9) < 1e-3) and (abs(s_neff - 1.76198) < 5e-4) \
        and (abs(h_surf - 89.9) < 1e-2) and (abs(h_neff - s_neff) < 1e-3)
    print("[wa] A strong recovery: W=None surface={:.4f} deg n_eff={:.5f} (golden 1.76198); W=1e2 "
          "surface={:.4f} (->theta_b), n_eff match={} -> {}".format(
              s_surf, s_neff, h_surf, abs(h_neff - s_neff) < 1e-3, "OK" if g_a else "FAIL"), flush=True)

    # GATE B: finite anchoring -- surface deviation monotone in 1/W (extrapolation length)
    devs = [89.9 - _surf_ctr_neff(W)[0] for W in (1e-3, 3e-4, 1e-4)]
    g_b = all(d > 0 for d in devs) and devs[0] < devs[1] < devs[2]
    print("[wa] B finite anchoring surface deviation (deg) at W=1e-3/3e-4/1e-4: {:.2f} < {:.2f} < {:.2f} "
          "(monotone) -> {}".format(devs[0], devs[1], devs[2], "OK" if g_b else "FAIL"), flush=True)

    # GATE C: Haller K(T)/gamma1(T)
    T_NI, T_ref = 380.0, 300.0
    S = [haller_order_parameter(T, T_NI) for T in (300, 360, 380)]
    KT = [K_of_temperature(17e-12, T, T_ref_K=T_ref, T_NI_K=T_NI) for T in (300, 360)]
    Vth = [freedericksz_threshold_V(K_of_temperature(17e-12, T, T_ref_K=T_ref, T_NI_K=T_NI), 14.7)
           for T in (300, 360)]
    g1 = [gamma1_of_temperature(0.085, T, T_ref_K=T_ref, E_a_eV=0.4, T_NI_K=T_NI) for T in (300, 360)]
    g_c = (S[0] > S[1] > S[2]) and (abs(S[2]) < 1e-9) \
        and (abs(KT[0] - 17e-12) < 1e-18) and (KT[1] < KT[0]) \
        and (Vth[1] < Vth[0]) and (abs(g1[0] - 0.085) < 1e-9) and (g1[1] < g1[0])
    print("[wa] C Haller: S(300/360/380)={:.3f}/{:.3f}/{:.3f}; K(360)/K_ref={:.3f}; V_th 300->360 "
          "{:.3f}->{:.3f}V; gamma1 {:.4f}->{:.4f} Pa.s -> {}".format(
              S[0], S[1], S[2], KT[1] / 17e-12, Vth[0], Vth[1], g1[0], g1[1], "OK" if g_c else "FAIL"),
          flush=True)

    ok = g_a and g_b and g_c
    print("[wa] *** LC WEAK ANCHORING + K(T)/gamma1(T): {} ***".format("PASS" if ok else "FAIL"), flush=True)
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
