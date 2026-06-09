"""Validate the CHIRAL / TWISTED-nematic static director solver added to dynameta/carriers/lc_director.py:
chiral_director_profile_bvp solves the COUPLED tilt theta(z) AND azimuthal twist phi(z) for a three-
constant (K11 splay / K22 twist / K33 bend) planar cell with an optional cholesteric pitch q0. Independent
reduces-to-known-limit oracles only.

GATE A (decouple regression): with phi_top = phi_bottom and q0 = 0 the twist decouples (phi == const) and
        theta(z) must equal the two-constant director_profile_bvp EXACTLY -- the strongest anchor (same
        Euler-Lagrange tilt equation), including the Freedericksz-tilted branch under field.
GATE B (pure twist, no field): a 90 deg twisted cell at V = 0, planar (theta = pi/2), q0 = 0 has the
        elastic minimizer phi(z) = phi_t z/d LINEAR with theta flat, and twist energy (1/2) K22 phi_t^2 / d.
GATE C (cholesteric undistorted helix): with q0 != 0 and the plate azimuths matched to the natural pitch
        the helix is undistorted, phi'(z) = q0 uniform, and the twist free energy = -(1/2) K22 q0^2 d.
GATE D (TN Freedericksz): in a 90 deg TN under field the midplane tilt grows MONOTONICALLY toward
        homeotropic with voltage (the dielectric torque acts on theta; twist preserves the polar angle).
GATE E (Gooch-Tarry optic): the 90 deg TN crossed-polarizer transmission has its first minimum (dark
        state) at the Mauguin number u = sqrt(3), the classic Gooch-Tarry condition.

Run: python -m validation.lc_chiral_twist
"""
import os
import sys
import math

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dynameta.carriers.lc_director import (
    chiral_director_profile_bvp, director_profile_bvp, cholesteric_q0,
    gooch_tarry_transmission, mauguin_number)

THB = math.radians(89.9)
K = dict(K11=11e-12, K22=7e-12, K33=18e-12, eps_para=18.7, eps_perp=4.0, d_planar=4e-6,
         field_model="uniform", nz=161, theta_b_rad=THB, n_o=1.52, n_e=1.74)


def main():
    print("[ct] === LC chiral / twisted-nematic director ===", flush=True)

    # GATE A: decouple -> director_profile_bvp EXACT (incl. Freedericksz-tilted branch at V=2)
    ch = chiral_director_profile_bvp(V_app=2.0, phi_bottom_rad=0.0, phi_top_rad=0.0, q0_rad_m=0.0, **K)
    st = director_profile_bvp(V_app=2.0, K11=K["K11"], K33=K["K33"], eps_para=K["eps_para"],
                              eps_perp=K["eps_perp"], d_planar=K["d_planar"], field_model="uniform",
                              nz=K["nz"], theta_b_rad=THB, n_o=K["n_o"], n_e=K["n_e"])
    dth = math.degrees(float(np.max(np.abs(ch.theta_field_rad - st.theta_field_rad))))
    dphi = math.degrees(float(np.max(np.abs(ch.phi_rad))))
    g_a = ch.success and dth < 1e-3 and dphi < 1e-9 and abs(ch.n_eff - st.n_eff) < 1e-5
    print("[ct] A decouple: max|dtheta vs director_profile_bvp|={:.2e} deg, max|phi|={:.1e} deg, n_eff "
          "{:.5f}=={:.5f} -> {}".format(dth, dphi, ch.n_eff, st.n_eff, "OK" if g_a else "FAIL"), flush=True)

    # GATE B: pure twist, no field -> phi linear, theta flat, energy = (1/2) K22 phi_t^2/d
    ch2 = chiral_director_profile_bvp(V_app=0.0, phi_bottom_rad=0.0, phi_top_rad=0.5 * math.pi,
                                      q0_rad_m=0.0, **{**K, "theta_b_rad": 0.5 * math.pi})
    u = (ch2.z_m - ch2.z_m[0]) / (ch2.z_m[-1] - ch2.z_m[0])
    dlin = math.degrees(float(np.max(np.abs(ch2.phi_rad - 0.5 * math.pi * u))))
    dflat = math.degrees(float(np.max(np.abs(ch2.theta_field_rad - 0.5 * math.pi))))
    w_an = 0.5 * K["K22"] * (0.5 * math.pi) ** 2 / K["d_planar"]
    g_b = dlin < 1e-3 and dflat < 1e-3 and abs(ch2.twist_energy_J_m2 - w_an) / w_an < 1e-6
    print("[ct] B pure twist: max|phi-linear|={:.2e} deg, max|theta-90|={:.2e} deg, twist_E {:.4e} vs "
          "(1/2)K22 phi^2/d {:.4e} -> {}".format(dlin, dflat, ch2.twist_energy_J_m2, w_an,
                                                 "OK" if g_b else "FAIL"), flush=True)

    # GATE C: cholesteric undistorted helix -> phi' = q0, energy = -(1/2) K22 q0^2 d
    q0 = cholesteric_q0(8e-6)
    ch3 = chiral_director_profile_bvp(V_app=0.0, phi_bottom_rad=0.0, phi_top_rad=q0 * K["d_planar"],
                                      q0_rad_m=q0, **{**K, "theta_b_rad": 0.5 * math.pi})
    phz = np.gradient(ch3.phi_rad, ch3.z_m)
    w_ch = -0.5 * K["K22"] * q0 * q0 * K["d_planar"]
    g_c = float(np.max(np.abs(phz - q0))) / q0 < 1e-6 and abs(ch3.twist_energy_J_m2 - w_ch) / abs(w_ch) < 1e-6
    print("[ct] C cholesteric: q0={:.0f} rad/m, max|phi'-q0|/q0={:.1e}, twist_E {:.4e} vs -(1/2)K22 q0^2 d "
          "{:.4e} -> {}".format(q0, float(np.max(np.abs(phz - q0))) / q0, ch3.twist_energy_J_m2, w_ch,
                                "OK" if g_c else "FAIL"), flush=True)

    # GATE D: TN Freedericksz -- midplane tilt grows monotonically with field
    tilts = []
    for V in (0.3, 0.9, 1.5):
        r = chiral_director_profile_bvp(V_app=V, phi_bottom_rad=0.0, phi_top_rad=0.5 * math.pi, **K)
        tilts.append((V, 0.5 * math.pi - float(r.theta_field_rad[r.theta_field_rad.size // 2]), r.success))
    g_d = all(s for _, _, s in tilts) and tilts[0][1] < tilts[1][1] < tilts[2][1] and tilts[2][1] > 0.5
    print("[ct] D TN Freedericksz midplane tilt (rad): " +
          ", ".join("V={:.1f}:{:.3f}".format(v, t) for v, t, _ in tilts) +
          " (monotone, V=1.5 tilted) -> {}".format("OK" if g_d else "FAIL"), flush=True)

    # GATE E: Gooch-Tarry first transmission minimum at u = sqrt(3)
    dn, lam = 0.22, 0.55e-6
    d_min = math.sqrt(3.0) * lam / (2.0 * dn)
    u_min = mauguin_number(d_min, dn, lam)
    T_min = gooch_tarry_transmission(d_min, dn, lam)
    g_e = abs(u_min - math.sqrt(3.0)) < 1e-9 and T_min < 1e-12
    print("[ct] E Gooch-Tarry: u(d_min)={:.4f} (sqrt3={:.4f}), T_min={:.2e} -> {}".format(
        u_min, math.sqrt(3.0), T_min, "OK" if g_e else "FAIL"), flush=True)

    ok = g_a and g_b and g_c and g_d and g_e
    print("[ct] *** LC CHIRAL / TWISTED NEMATIC: {} ***".format("PASS" if ok else "FAIL"), flush=True)
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
