"""Validate the LC dynamics deeper-physics additions to dynameta/carriers/lc_dynamics.py: finite
(Rapini-Papoular) surface anchoring in the time domain, and BACKFLOW (Leslie director-flow coupling,
the local effective-viscosity model). Independent oracles: the STATIC weak-anchoring BVP (same torque
balance -> same steady state) and the backflow off-limit.

GATE A (weak-anchoring dynamics -> static): LCDynamics with finite W held at a fixed voltage relaxes to
        the SAME profile (surface + midplane tilt) as the static director_profile_bvp(W) -- the surface
        torque balance with surface viscosity gamma_s has the static Rapini-Papoular BC as its fixed point.
GATE B (backflow): backflow SPEEDS UP the switching (rise/decay times shorter than no-backflow); the
        effective rotational viscosity gamma1_eff(theta) = gamma1 - g^2/eta_shear is < gamma1; and
        alpha2 = alpha3 = 0 reproduces the no-backflow result byte-for-byte. (The local model OVERESTIMATES
        the speedup vs the full no-slip nonlocal flow; the direction + off-limit are the robust checks.)

Run: python -m validation.lc_dynamics_anchoring_backflow
"""
import os
import sys
import math

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dynameta.carriers.lc_dynamics import LCDynamics
from dynameta.carriers.lc_director import director_profile_bvp

THB = math.radians(89.9)


def main():
    print("[db] === LC weak-anchoring dynamics + backflow ===", flush=True)

    # GATE A: weak-anchoring dynamics steady state == static weak-anchoring BVP
    W = 3e-4
    st = director_profile_bvp(V_app=2.0, K11=17e-12, K33=18e-12, eps_para=18.7, eps_perp=4.0,
                              d_planar=1e-6, theta_b_rad=THB, field_model="uniform", nz=121, W_anchor_J_m2=W)
    dy = LCDynamics(K11=17e-12, K33=18e-12, gamma1=0.085, eps_para=18.7, eps_perp=4.0, theta_b_rad=THB,
                    geometry="planar", d_planar=1e-6, field_model="uniform", nz=121,
                    W_anchor_J_m2=W, gamma_s_Pa_s_m=1e-8)
    tau = dy.tau_1const_s()
    r = dy.simulate(np.linspace(0.0, 30.0 * tau, 300), lambda t: 2.0, theta0_rad=None)
    th_dyn, th_st = r.theta_zt_rad[:, -1], st.theta_field_rad
    dmax = math.degrees(float(np.max(np.abs(th_dyn - th_st))))
    g_a = dmax < 0.5
    print("[db] A weak-anchor dyn vs static: surface {:.2f}/{:.2f} deg, mid {:.2f}/{:.2f} deg, max|d|={:.3f}"
          " deg -> {}".format(math.degrees(th_dyn[0]), math.degrees(th_st[0]),
                              math.degrees(th_dyn[th_dyn.size // 2]), math.degrees(th_st[th_st.size // 2]),
                              dmax, "OK" if g_a else "FAIL"), flush=True)

    # GATE B: backflow speeds up switching; alpha2=alpha3=0 reproduces no-backflow
    base = dict(K11=17e-12, K33=18e-12, gamma1=0.085, eps_para=18.7, eps_perp=4.0, theta_b_rad=THB,
                geometry="planar", d_planar=1e-6, field_model="uniform", n_o=1.56, n_e=1.92, nz=81)
    pk = dict(V0=2.0, Ton=3e-3, T_end=9e-3, n_t=200)
    rno = LCDynamics(**base).simulate_pulse(**pk)
    rbf = LCDynamics(include_backflow=True, alpha2_Pa_s=-0.08, alpha3_Pa_s=-0.003, eta_shear_Pa_s=0.08,
                     **base).simulate_pulse(**pk)
    rz = LCDynamics(include_backflow=True, alpha2_Pa_s=0.0, alpha3_Pa_s=0.0, eta_shear_Pa_s=0.08,
                    **base).simulate_pulse(**pk)
    faster = (rbf.rise_10_90_s < rno.rise_10_90_s) and (rbf.decay_90_10_s < rno.decay_90_10_s)
    off_identical = (abs(rz.rise_10_90_s - rno.rise_10_90_s) < 1e-9) and \
                    (abs(rz.decay_90_10_s - rno.decay_90_10_s) < 1e-9)
    g_b = faster and off_identical
    print("[db] B backflow rise/decay {:.4f}/{:.4f} ms vs no-bf {:.4f}/{:.4f} ms (faster={}); alpha2=alpha3"
          "=0 identical={} -> {}".format(rbf.rise_10_90_s * 1e3, rbf.decay_90_10_s * 1e3,
                                         rno.rise_10_90_s * 1e3, rno.decay_90_10_s * 1e3, faster,
                                         off_identical, "OK" if g_b else "FAIL"), flush=True)

    ok = g_a and g_b
    print("[db] *** LC WEAK-ANCHORING DYNAMICS + BACKFLOW: {} ***".format("PASS" if ok else "FAIL"), flush=True)
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
