"""Validate the two-constant (K11 splay / K33 bend) static nematic director BVP added to
dynameta/carriers/lc_director.py (director_profile_bvp), plus the electrostatics (Poisson
voltage-division through fixed dielectric layers) and the n_eff optical average. The director module's
new functions use the FIELD-AXIS angle convention (theta = 0 homeotropic, theta = pi/2 planar) of the
external 1-D nematic solver they are ported from; director_to_extra_fields bridges to the plate-plane
'director_angle_rad' the optical LiquidCrystalModel consumes (theta_optic = pi/2 - theta_field).

Independent oracles:
  * GOLDEN external solver (lc_statics_base.py) planar uniform-field sweep at K11=17pN, K33=18pN,
    eps_para=18.7, eps_perp=4.0 (dEps=14.7), gap=1um, theta_b=89.9deg, n_o=1.56, n_e=1.92, extra_k_radial:
      V_app -> (V_lc, n_eff, theta_center_deg): 0->(0,1.5600008,89.9), 1.0->(1.0,1.5600143,89.4382),
      1.5->(1.5,1.6835461,33.6392), 2.0->(2.0,1.7631085,15.5256); and the POISSON case with 100nm/eps7.5
      fixed layers each side: V_app=2 -> (V_lc=1.6141, n_eff=1.6385).
  * the existing 1-constant elliptic-quadrature director_profile() (constant-displacement field) -- the
    new BVP at K11=K33 with field_model='poisson' (no fixed layers) reproduces it THROUGH the pi/2 bridge.
  * the analytic Freedericksz threshold V_th = pi sqrt(K11/(eps0 dEps)).

GATE A (golden statics): director_profile_bvp planar-uniform theta_center + n_eff match the golden tuples
        (< 1.0 deg, < 4e-3 n_eff).
GATE B (1-constant reduction): BVP(K11=K33, poisson) bridged to plate-plane matches director_profile to
        < 2e-2 rad across V (no regression of the existing 1-constant path).
GATE C (Freedericksz + branch): below V_th the cell stays at theta_b (no tilt); above it tilts toward the
        field (theta_center < theta_b - 5 deg by V = 1.5 V_th).
GATE D (Poisson voltage division): with fixed dielectric layers V_lc < V_app and matches the golden
        (< 1%); uniform field returns V_lc = V_app exactly.
GATE E (n_eff limits): n_eff -> n_o at planar (theta=pi/2) and -> n_e at homeotropic (theta=0) for the
        extra_k_radial model.

Run: python -m validation.lc_two_constant_bvp
"""
import os
import sys
import math

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dynameta.carriers.lc_director import (
    director_profile_bvp, director_profile, freedericksz_threshold_V,
    solve_lc_field_profile, compute_lc_geometry, n_eff_from_theta_profile, director_to_extra_fields)

K11, K33 = 17e-12, 18e-12
EPS_PARA, EPS_PERP = 18.7, 4.0
DEPS = EPS_PARA - EPS_PERP
GAP = 1e-6
N_O, N_E = 1.56, 1.92
THB = math.radians(89.9)
GOLD = {0.0: (0.0, 1.5600008, 89.9), 1.0: (1.0, 1.5600143, 89.4382),
        1.5: (1.5, 1.6835461, 33.6392), 2.0: (2.0, 1.7631085, 15.5256)}


def _center_deg(res):
    th = res.theta_field_rad
    return math.degrees(float(th[len(th) // 2]))


def main():
    print("[lc] === Two-constant static director BVP (K11/K33) + electrostatics + n_eff ===", flush=True)
    kw = dict(K11=K11, K33=K33, eps_para=EPS_PARA, eps_perp=EPS_PERP, d_planar=GAP, theta_b_rad=THB,
              field_model="uniform", nz=121, n_o=N_O, n_e=N_E, opt_model="extra_k_radial")

    # GATE A: golden external statics
    dth = dne = 0.0
    for V, (vlc_g, neff_g, thc_g) in GOLD.items():
        r = director_profile_bvp(V_app=V, **kw)
        dth = max(dth, abs(_center_deg(r) - thc_g)); dne = max(dne, abs(r.n_eff - neff_g))
    g_a = (dth < 1.0) and (dne < 4e-3)
    print("[lc] A golden planar-uniform sweep: max|d theta_ctr|={:.3f} deg, max|d n_eff|={:.2e} -> {}"
          .format(dth, dne, "OK" if g_a else "FAIL"), flush=True)

    # GATE B: 1-constant reduction vs the existing elliptic-quadrature director_profile (constant-D)
    dred = 0.0
    for V in (1.3, 1.8, 2.5, 3.5):
        dp = director_profile(K11, DEPS, EPS_PERP, GAP, V)
        bv = director_profile_bvp(V_app=V, K11=K11, K33=K11, eps_para=EPS_PARA, eps_perp=EPS_PERP,
                                  d_planar=GAP, theta_b_rad=math.radians(89.97), field_model="poisson",
                                  nz=301)
        th_bridge = 0.5 * math.pi - bv.theta_field_rad[len(bv.theta_field_rad) // 2]
        dred = max(dred, abs(dp.theta_max_rad - th_bridge))
    g_b = dred < 2e-2
    print("[lc] B 1-constant reduction vs director_profile (pi/2 bridge): max|d theta|={:.2e} rad -> {}"
          .format(dred, "OK" if g_b else "FAIL"), flush=True)

    # GATE C: Freedericksz threshold + branch selection
    Vth = freedericksz_threshold_V(K11, DEPS)
    below = director_profile_bvp(V_app=0.6 * Vth, **kw)
    above = director_profile_bvp(V_app=1.5 * Vth, **kw)
    thb_deg = math.degrees(THB)
    g_c = (abs(_center_deg(below) - thb_deg) < 0.5) and (_center_deg(above) < thb_deg - 5.0)
    print("[lc] C Freedericksz V_th={:.3f} V: theta_ctr below={:.2f} deg (~theta_b), above={:.2f} deg "
          "(tilts) -> {}".format(Vth, _center_deg(below), _center_deg(above), "OK" if g_c else "FAIL"),
          flush=True)

    # GATE D: Poisson voltage division (fixed layers) vs golden + uniform exactness
    geo = compute_lc_geometry(geometry="planar", nz=121, d_planar=GAP, t_in=100e-9, t_out=100e-9)
    res_p = director_profile_bvp(V_app=2.0, K11=K11, K33=K33, eps_para=EPS_PARA, eps_perp=EPS_PERP,
                                 geo=geo, theta_b_rad=THB, field_model="poisson", t_in=100e-9,
                                 t_out=100e-9, eps_in=7.5, eps_out=7.5, nz=121, n_o=N_O, n_e=N_E)
    th_flat = np.full(50, THB)
    geo_u = compute_lc_geometry(geometry="planar", nz=50, d_planar=GAP)
    _Eu, vlc_u = solve_lc_field_profile(th_flat, 2.0, geo_u, eps_para=EPS_PARA, eps_perp=EPS_PERP,
                                        field_model="uniform")
    g_d = (abs(res_p.V_lc - 1.6141) < 0.02) and (res_p.V_lc < 2.0) and (abs(vlc_u - 2.0) < 1e-12)
    print("[lc] D Poisson division: V_app=2 -> V_lc={:.4f} (golden 1.6141, < V_app); uniform V_lc={:.4f}"
          " (==V_app) -> {}".format(res_p.V_lc, vlc_u, "OK" if g_d else "FAIL"), flush=True)

    # GATE E: n_eff limits + the optics bridge
    z = np.linspace(0.0, GAP, 51)
    n_planar = n_eff_from_theta_profile(np.full_like(z, 0.5 * math.pi), z, N_O, N_E, model="extra_k_radial")
    n_homeo = n_eff_from_theta_profile(np.zeros_like(z), z, N_O, N_E, model="extra_k_radial")
    bridged = director_to_extra_fields(np.array([0.5 * math.pi, 0.0]))["director_angle_rad"]
    g_e = (abs(n_planar - N_O) < 1e-6) and (abs(n_homeo - N_E) < 1e-6) and \
          (abs(bridged[0] - 0.0) < 1e-12) and (abs(bridged[1] - 0.5 * math.pi) < 1e-12)
    print("[lc] E n_eff limits: planar={:.4f} (n_o={:.2f}), homeotropic={:.4f} (n_e={:.2f}); bridge "
          "pi/2->0, 0->pi/2 -> {}".format(n_planar, N_O, n_homeo, N_E, "OK" if g_e else "FAIL"), flush=True)

    ok = g_a and g_b and g_c and g_d and g_e
    print("[lc] *** TWO-CONSTANT LC STATIC BVP: {} ***".format("PASS" if ok else "FAIL"), flush=True)
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
