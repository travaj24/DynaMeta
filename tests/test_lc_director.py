"""Unit coverage for the Phase-4 liquid-crystal director driver (carriers.lc_director) -- the
1-constant Frank-elastic Freedericksz transition. Pure numpy/scipy. Run:
python -m pytest tests/test_lc_director.py -q
"""
import numpy as np
import pytest

from dynameta.constants import EPS0
from dynameta.carriers.lc_director import (
    freedericksz_threshold_V, director_profile, director_profile_bvp, solve_lc_field_profile,
    compute_lc_geometry, n_local_from_theta, n_eff_from_theta_profile, eps_along_field,
    flexo_p_along_field, director_to_extra_fields,
    haller_order_parameter, K_of_temperature, gamma1_of_temperature)

K, DEPS, EP, D = 6.5e-12, 11.0, 7.0, 5e-6     # ~5CB nematic cell


def test_threshold_matches_analytic():
    Vth = freedericksz_threshold_V(K, DEPS)
    assert Vth == pytest.approx(np.pi * np.sqrt(K / (EPS0 * DEPS)), rel=1e-12)


def test_threshold_requires_positive_anisotropy():
    with pytest.raises(ValueError):
        freedericksz_threshold_V(K, 0.0)
    with pytest.raises(ValueError):
        freedericksz_threshold_V(K, -2.0)                  # negative anisotropy -> no planar threshold


def test_planar_below_threshold_and_tilt_above():
    Vth = freedericksz_threshold_V(K, DEPS)
    assert director_profile(K, DEPS, EP, D, 0.5 * Vth).theta_max_rad == 0.0      # below -> planar
    assert director_profile(K, DEPS, EP, D, 0.99 * Vth).theta_max_rad == 0.0
    assert director_profile(K, DEPS, EP, D, 1.1 * Vth).theta_max_rad > 0.0       # above -> tilts


def test_tilt_monotonic_in_voltage_and_saturates():
    Vth = freedericksz_threshold_V(K, DEPS)
    tm = [director_profile(K, DEPS, EP, D, r * Vth).theta_max_rad for r in (1.1, 1.5, 2.0, 4.0)]
    assert all(b > a for a, b in zip(tm, tm[1:]))          # monotonic increasing
    assert tm[-1] < 0.5 * np.pi and tm[-1] > np.radians(80)   # saturates toward homeotropic, < 90


def test_threshold_independent_of_thickness():
    # V_th must not depend on cell thickness (classic Freedericksz result)
    Vth = freedericksz_threshold_V(K, DEPS)
    t_thin = director_profile(K, DEPS, EP, 2e-6, 1.5 * Vth).theta_max_rad
    t_thick = director_profile(K, DEPS, EP, 20e-6, 1.5 * Vth).theta_max_rad
    assert t_thin == pytest.approx(t_thick, rel=1e-3)      # same tilt at the same V/V_th


def test_supercritical_sqrt_law_near_threshold():
    Vth = freedericksz_threshold_V(K, DEPS)
    rs = np.array([1.02, 1.05, 1.10])
    tm = np.array([director_profile(K, DEPS, EP, D, r * Vth).theta_max_rad for r in rs])
    ratio = tm ** 2 / (rs - 1.0)                            # theta_max^2 ~ (V/Vth - 1) => ~const
    assert np.all(ratio > 0) and (ratio.max() / ratio.min() < 1.3)


def test_planar_anchoring_boundary_conditions():
    Vth = freedericksz_threshold_V(K, DEPS)
    p = director_profile(K, DEPS, EP, D, 1.6 * Vth, nz=201)
    assert abs(p.theta_rad[0]) < 1e-9 and abs(p.theta_rad[-1]) < 1e-9   # theta=0 at both plates
    assert p.theta_max_rad == pytest.approx(np.max(p.theta_rad))
    assert p.theta_rad.argmax() in (p.theta_rad.size // 2, p.theta_rad.size // 2 - 1,
                                    p.theta_rad.size // 2 + 1)          # peak at the midplane


def test_director_profile_rejects_bad_input():
    with pytest.raises(ValueError):
        director_profile(0.0, DEPS, EP, D, 1.0)            # K <= 0
    with pytest.raises(ValueError):
        director_profile(K, DEPS, EP, D, 1.0, nz=5)        # nz too small


# -------------------------------------------------------------------------
# Two-constant (K11/K33) statics: helpers + BVP reduction (FIELD-AXIS convention)
# -------------------------------------------------------------------------
def test_eps_along_field_limits():
    # FIELD-AXIS: theta=0 (along field) -> eps_para; theta=pi/2 (planar) -> eps_perp
    assert eps_along_field(0.0, 18.7, 4.0) == pytest.approx(18.7)
    assert eps_along_field(0.5 * np.pi, 18.7, 4.0) == pytest.approx(4.0)


def test_n_local_limits_and_models():
    n_o, n_e = 1.56, 1.92
    # extra_k_radial: theta=0 (homeotropic, along field) -> n_e; theta=pi/2 (planar) -> n_o
    assert n_local_from_theta(0.0, n_o, n_e, "extra_k_radial") == pytest.approx(n_e)
    assert n_local_from_theta(0.5 * np.pi, n_o, n_e, "extra_k_radial") == pytest.approx(n_o)
    assert n_local_from_theta(0.3, n_o, n_e, "ordinary") == pytest.approx(n_o)
    with pytest.raises(ValueError):
        n_local_from_theta(0.1, n_o, n_e, "nonsense")


def test_field_profile_uniform_exact_and_poisson_division():
    geo = compute_lc_geometry(geometry="planar", nz=64, d_planar=1e-6)
    th = np.full(64, np.radians(89.9))
    E_u, vlc_u = solve_lc_field_profile(th, 2.0, geo, eps_para=18.7, eps_perp=4.0, field_model="uniform")
    assert vlc_u == pytest.approx(2.0)                     # uniform: V_lc == V_app
    assert np.allclose(E_u, 2.0 / geo.d_lc)                # E == V/d everywhere
    geo_f = compute_lc_geometry(geometry="planar", nz=64, d_planar=1e-6, t_in=100e-9, t_out=100e-9)
    _Ef, vlc_f = solve_lc_field_profile(th, 2.0, geo_f, eps_para=18.7, eps_perp=4.0,
                                        field_model="poisson", t_in=100e-9, t_out=100e-9,
                                        eps_in=7.5, eps_out=7.5)
    assert 0.0 < vlc_f < 2.0                                # series fixed layers drop part of V_app


def test_bvp_reduces_to_one_constant_director_profile():
    # the two-constant BVP at K11==K33 (constant-displacement 'poisson' field, no fixed layers)
    # reproduces the existing 1-constant elliptic-quadrature director_profile THROUGH the pi/2 bridge.
    Kc, dEps, ep, d = 17e-12, 14.7, 4.0, 1e-6
    V = 1.8
    dp = director_profile(Kc, dEps, ep, d, V)
    bv = director_profile_bvp(V_app=V, K11=Kc, K33=Kc, eps_para=ep + dEps, eps_perp=ep, d_planar=d,
                              theta_b_rad=np.radians(89.97), field_model="poisson", nz=201)
    th_bridge = 0.5 * np.pi - bv.theta_field_rad[bv.theta_field_rad.size // 2]
    assert abs(dp.theta_max_rad - th_bridge) < 2e-2


def test_bvp_freedericksz_threshold_and_branch():
    Vth = freedericksz_threshold_V(17e-12, 14.7)
    thb = np.radians(89.9)
    kw = dict(K11=17e-12, K33=18e-12, eps_para=18.7, eps_perp=4.0, d_planar=1e-6,
              theta_b_rad=thb, field_model="uniform", nz=81)
    below = director_profile_bvp(V_app=0.6 * Vth, **kw)
    above = director_profile_bvp(V_app=1.5 * Vth, **kw)
    cen = lambda r: r.theta_field_rad[r.theta_field_rad.size // 2]
    assert abs(cen(below) - thb) < np.radians(0.5)         # below: stays ~planar (theta_b)
    assert cen(above) < thb - np.radians(5.0)              # above: tilts toward the field


def test_flexo_zero_when_coeffs_zero():
    th = np.linspace(0.2, 1.2, 11); dth = np.gradient(th)
    assert np.allclose(flexo_p_along_field(th, dth, 0.0, 0.0), 0.0)


def test_director_to_extra_fields_bridge():
    out = director_to_extra_fields(np.array([0.0, 0.5 * np.pi]))
    assert "director_angle_rad" in out
    assert out["director_angle_rad"][0] == pytest.approx(0.5 * np.pi)   # homeotropic field-axis -> optic pi/2
    assert out["director_angle_rad"][1] == pytest.approx(0.0)           # planar field-axis -> optic 0


def test_bridge_into_liquid_crystal_model_is_uniaxial_with_right_axis():
    # the convention seam end-to-end: a field-axis director -> director_to_extra_fields ->
    # LiquidCrystalModel eps must be uniaxial {n_o^2, n_o^2, n_e^2} with the extraordinary axis at
    # theta_optic = pi/2 - theta_field.
    from dynameta.core.effects import LiquidCrystalModel
    n_o, n_e = 1.56, 1.92
    lcm = LiquidCrystalModel(n_o=n_o, n_e=n_e)
    th_field = np.array([0.0, np.radians(30.0), 0.5 * np.pi])      # homeotropic, 30 deg, planar
    eps = np.asarray(lcm.eps(director_to_extra_fields(th_field), 1.5e-6)).real
    for i, thf in enumerate(th_field):
        w, V = np.linalg.eigh(eps[i])
        assert np.allclose(np.sort(w), np.sort([n_o ** 2, n_o ** 2, n_e ** 2]), atol=1e-9)
        ext = V[:, int(np.argmax(w))]
        th_opt = 0.5 * np.pi - float(thf)
        want = np.array([np.cos(th_opt), 0.0, np.sin(th_opt)])
        assert abs(abs(float(np.dot(ext, want))) - 1.0) < 1e-6


def test_bvp_high_drive_does_not_rail_to_n_o():
    # AUDIT FIX (voltage continuation): at high overdrive the tilted branch has a thin homeotropic
    # boundary layer; without continuation the solver railed n_eff back to n_o. Now it must climb
    # toward n_e and report success.
    thb = np.radians(89.9)
    last = 0.0
    for V in (4.0, 8.0, 14.0):
        r = director_profile_bvp(V_app=V, K11=17e-12, K33=18e-12, eps_para=18.7, eps_perp=4.0,
                                 d_planar=2e-6, theta_b_rad=thb, field_model="uniform", nz=161,
                                 n_o=1.56, n_e=1.92)
        assert r.success and r.n_eff > 1.80 and r.n_eff <= 1.92 + 1e-9   # toward n_e, NOT railed to 1.56
        assert r.n_eff >= last - 1e-6                                    # monotone non-decreasing in V
        last = r.n_eff


def test_bvp_high_splay_contrast_tilts():
    # AUDIT FIX: K11 >> K33 near-planar cells used to return the undistorted branch above threshold.
    r = director_profile_bvp(V_app=1.5, K11=17e-12, K33=4e-12, eps_para=18.7, eps_perp=4.0,
                             d_planar=2e-6, theta_b_rad=np.radians(89.99), field_model="uniform", nz=161)
    mid = r.theta_field_rad[r.theta_field_rad.size // 2]
    assert mid < np.radians(89.99) - np.radians(30.0)                   # genuinely tilted, not ~planar


def test_bvp_negative_anisotropy_van_tilts():
    # AUDIT FIX: negative anisotropy (VAN, dEps<0) used to collapse to the untilted (linearly unstable)
    # root because V_th was forced to inf. The director must now tilt toward planar (theta -> pi/2).
    r = director_profile_bvp(V_app=2.0, K11=10e-12, K33=10e-12, eps_para=4.0, eps_perp=18.7,
                             d_planar=2e-6, theta_b_rad=np.radians(1.0), field_model="uniform", nz=161)
    mid = r.theta_field_rad[r.theta_field_rad.size // 2]
    assert mid > np.radians(60.0)                                       # swung far toward planar (pi/2)


def test_bvp_below_threshold_untilted():
    Vth = freedericksz_threshold_V(17e-12, 14.7)
    r = director_profile_bvp(V_app=0.5 * Vth, K11=17e-12, K33=18e-12, eps_para=18.7, eps_perp=4.0,
                             d_planar=1e-6, theta_b_rad=np.radians(89.9), field_model="uniform", nz=81)
    mid = r.theta_field_rad[r.theta_field_rad.size // 2]
    assert abs(mid - np.radians(89.9)) < np.radians(0.5)                # stays planar below V_th


def test_weak_anchoring_recovers_strong_and_tilts_surface():
    thb = np.radians(89.9)
    kw = dict(K11=17e-12, K33=18e-12, eps_para=18.7, eps_perp=4.0, d_planar=1e-6, theta_b_rad=thb,
              field_model="uniform", nz=121)
    strong = director_profile_bvp(V_app=2.0, W_anchor_J_m2=None, **kw)
    huge = director_profile_bvp(V_app=2.0, W_anchor_J_m2=1e2, **kw)       # b = K/W ~ 1e-13 m
    assert abs(strong.theta_field_rad[0] - thb) < 1e-6                     # strong: surface pinned
    assert abs(huge.theta_field_rad[0] - thb) < np.radians(0.05)          # huge W -> recovers strong
    weak = director_profile_bvp(V_app=2.0, W_anchor_J_m2=3e-4, **kw)
    assert (thb - weak.theta_field_rad[0]) > np.radians(5.0)              # finite W: surface tilts to field


def test_haller_temperature_scaling():
    T_NI = 380.0
    assert haller_order_parameter(300.0, T_NI) > haller_order_parameter(360.0, T_NI) > 0.0
    assert haller_order_parameter(T_NI, T_NI) == 0.0                       # S=0 at the transition
    assert haller_order_parameter(400.0, T_NI) == 0.0                      # and above
    # K ~ S^2 -> reduces to K_ref at T_ref, falls with T
    assert K_of_temperature(17e-12, 300.0, T_ref_K=300.0, T_NI_K=T_NI) == pytest.approx(17e-12)
    assert K_of_temperature(17e-12, 360.0, T_ref_K=300.0, T_NI_K=T_NI) < 17e-12
    # gamma1 Arrhenius -> reduces to gamma_ref at T_ref, falls with T
    assert gamma1_of_temperature(0.085, 300.0, T_ref_K=300.0, E_a_eV=0.4) == pytest.approx(0.085)
    assert gamma1_of_temperature(0.085, 360.0, T_ref_K=300.0, E_a_eV=0.4) < 0.085


def test_cyl_geometry_bvp_tilts_above_threshold():
    # coaxial cell, Poisson voltage division: a moderate drive tilts the midplane toward the field
    # (theta drops well below the planar theta_b).
    thb = np.radians(89.9)
    r = director_profile_bvp(V_app=3.0, K11=17e-12, K33=18e-12, eps_para=18.7, eps_perp=4.0,
                             geometry="cyl", a=51.5e-9, b=181.5e-9, t_in=10e-9, t_out=10e-9,
                             eps_in=7.5, eps_out=7.5, theta_b_rad=thb, field_model="poisson", nz=81)
    assert r.V_lc < 3.0                                            # series fixed layers drop part of V
    assert r.theta_field_rad[r.theta_field_rad.size // 2] < thb - np.radians(20.0)
