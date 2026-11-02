"""Unit coverage for the Phase-4 liquid-crystal director driver (carriers.lc_director) -- the
1-constant Frank-elastic Freedericksz transition. Pure numpy/scipy. Run:
python -m pytest tests/test_lc_director.py -q
"""
import numpy as np
import pytest

from dynameta.constants import EPS0
from dynameta.carriers.lc_director import (
    freedericksz_threshold_V, director_profile, director_profile_bvp, solve_lc_field_profile,
    compute_lc_geometry, LCGeometry, n_local_from_theta, n_eff_from_theta_profile, eps_along_field,
    flexo_p_along_field, director_to_extra_fields,
    haller_order_parameter, K_of_temperature, gamma1_of_temperature,
    chiral_director_profile_bvp, cholesteric_q0, gooch_tarry_transmission, mauguin_number)

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
    # RE-PINNED for AUDIT C-4: the series-capacitance normalisation used to be integrated on
    # solve_bvp's midpoint grid (short one cell), which alone accounted for 89% of the old residual --
    # 1.1861e-2 rad against the 2.0e-2 budget. With the domain-spanning quadrature it is 1.2472e-3.
    Kc, dEps, ep, d = 17e-12, 14.7, 4.0, 1e-6
    V = 1.8
    dp = director_profile(Kc, dEps, ep, d, V)
    bv = director_profile_bvp(V_app=V, K11=Kc, K33=Kc, eps_para=ep + dEps, eps_perp=ep, d_planar=d,
                              theta_b_rad=np.radians(89.97), field_model="poisson", nz=201)
    th_bridge = 0.5 * np.pi - bv.theta_field_rad[bv.theta_field_rad.size // 2]
    assert abs(dp.theta_max_rad - th_bridge) < 3e-3


def test_poisson_normalisation_spans_the_full_cell_on_any_grid():
    # AUDIT C-4 / N2 GATE: the global series-capacitance normalisation must be EXACT for a uniform
    # field on ANY abscissae -- including the interior-only collocation MIDPOINT grid that solve_bvp
    # hands the BVP right-hand side, where the old plain-trapz version was short one full cell
    # (E too large by 1/(1 - h/d_lc): +11.1% at nz=11, +2.04% at nz=51).
    for nz in (11, 13, 51, 64, 201):
        geo = compute_lc_geometry(geometry="planar", nz=nz, d_planar=1e-6)
        span = (float(geo.z_m[0]), float(geo.z_m[0]) + geo.d_lc)
        th = np.full(nz, np.radians(89.9))
        E, vlc = solve_lc_field_profile(th, 2.0, geo, eps_para=18.7, eps_perp=4.0,
                                        field_model="poisson", quad_span=span)
        assert np.max(np.abs(E - 2.0 / geo.d_lc)) / (2.0 / geo.d_lc) < 1e-12
        assert vlc == pytest.approx(2.0, rel=1e-12)
        # the midpoint grid: WITH quad_span exact, WITHOUT it biased by exactly 1/(1 - h/d_lc)
        zm = 0.5 * (geo.z_m[:-1] + geo.z_m[1:])
        gm = LCGeometry("planar", geo.d_lc, zm, zm, geo.r_in, geo.r_out)
        thm = np.full(zm.size, np.radians(89.9))
        Em, _ = solve_lc_field_profile(thm, 2.0, gm, eps_para=18.7, eps_perp=4.0,
                                       field_model="poisson", quad_span=span)
        assert np.max(np.abs(Em - 2.0 / geo.d_lc)) / (2.0 / geo.d_lc) < 1e-12
        Eb, _ = solve_lc_field_profile(thm, 2.0, gm, eps_para=18.7, eps_perp=4.0,
                                       field_model="poisson")
        h = geo.d_lc / (nz - 1)
        assert float(Eb[0]) / (2.0 / geo.d_lc) == pytest.approx(1.0 / (1.0 - h / geo.d_lc), rel=1e-12)


def test_span_weights_reduce_to_trapezoid_on_a_node_grid():
    # the domain-spanning rule is the trapezoid rule when the abscissae already include both ends,
    # and always sums to the domain length (which is what makes the uniform-field case exact).
    from dynameta.carriers.lc_director import _span_weights
    for x in (np.linspace(0.0, 1.0, 7), np.array([0.0, 0.1, 0.35, 0.8, 1.0])):
        w = _span_weights(x, 0.0, 1.0)
        assert w.sum() == pytest.approx(1.0, rel=1e-15)
        y = 3.0 * x + 0.5                                    # trapezoid is exact for a line
        assert float(np.sum(w * y)) == pytest.approx(2.0, rel=1e-14)
    xm = np.array([0.125, 0.375, 0.625, 0.875])              # midpoints only
    assert _span_weights(xm, 0.0, 1.0).sum() == pytest.approx(1.0, rel=1e-15)


def test_chiral_bvp_poisson_normalisation_matches_the_tilt_solver():
    # AUDIT N2 (C-4's defect verbatim in chiral_director_profile_bvp): with the twist decoupled
    # (phi_top == phi_bottom, q0 = 0) the chiral solver must reproduce director_profile_bvp under a
    # POISSON field too.
    #
    # SCOPE (fix-verify W1 item 8). This is a CONSISTENCY check between the two solvers, NOT a
    # regression gate for N2: both solvers carried the identical normalisation defect, so it passed
    # pre-fix as well (measured -- the earlier comment's claim that "their converged midplane tilts
    # differed by ~0.2 deg before the fix" is false). The discriminating gate is
    # test_chiral_bvp_poisson_is_grid_convergent below.
    ck = dict(K11=11e-12, K33=18e-12, eps_para=18.7, eps_perp=4.0, d_planar=4e-6, nz=121,
              theta_b_rad=np.radians(89.9), field_model="poisson")
    ch = chiral_director_profile_bvp(V_app=2.0, K22=7e-12, phi_bottom_rad=0.0, phi_top_rad=0.0,
                                     q0_rad_m=0.0, **ck)
    st = director_profile_bvp(V_app=2.0, **ck)
    assert ch.success
    assert np.max(np.abs(ch.theta_field_rad - st.theta_field_rad)) < 1e-6


def test_chiral_bvp_poisson_is_grid_convergent():
    """AUDIT N2, the DISCRIMINATING gate (fix-verify W1 item 8).

    The defect was the series-capacitance normalisation of the Poisson field: on solve_bvp's
    adapted MIDPOINT abscissae the denominator ``int dz / eps(theta)`` was evaluated with a plain
    trapezoid over the NODES it happened to hold, which spans only ``d_lc - h`` instead of the full
    cell, so E was biased by ``1/(1 - h/d_lc)``. That bias is O(h) and it is what makes the solved
    tilt move with ``nz``. It is invisible to a same-``nz`` comparison of two solvers that share the
    bug, and visible immediately under grid refinement -- so refine.

    Measured midplane tilt (twisted case, phi_top = pi/2, so the chiral solver is genuinely
    exercised), |theta_mid(nz) - theta_mid(nz = 961)| in degrees:

        nz = 121:   PRE-fix 0.757    POST-fix 0.032      (24x)
        nz = 61:    PRE-fix 0.657    POST-fix 0.089

    Pre-fix the sequence is not even monotone in nz (17.499, 17.399, 17.830, 18.156 deg for
    nz = 61, 121, 241, 961) -- the hallmark of an h-dependent forcing rather than a converging
    discretisation.
    """
    ck = dict(K11=11e-12, K33=18e-12, eps_para=18.7, eps_perp=4.0, d_planar=4e-6,
              theta_b_rad=np.radians(89.9), field_model="poisson")
    mids = {}
    for nz in (61, 121, 241):
        r = chiral_director_profile_bvp(V_app=2.0, K22=7e-12, phi_bottom_rad=0.0,
                                        phi_top_rad=0.5 * np.pi, q0_rad_m=0.0, nz=nz, **ck)
        assert r.success, nz
        mids[nz] = np.degrees(float(r.theta_field_rad[r.theta_field_rad.size // 2]))
    # monotone convergence, and the coarse-to-fine drift is an order of magnitude under the
    # pre-fix 0.66-0.76 deg
    assert mids[61] > mids[121] > mids[241], mids
    assert abs(mids[61] - mids[241]) < 0.15, mids
    assert abs(mids[121] - mids[241]) < 0.05, mids
    # ... and the drift SHRINKS with refinement (an O(h) forcing error would not)
    assert abs(mids[121] - mids[241]) < 0.6 * abs(mids[61] - mids[121]), mids


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
        # AUDIT T-6: rtol=0.0 -- eigenvalues are O(3), so numpy's default rtol=1e-5 would have made
        # the "uniaxial" gate 3e-5 absolute instead of the intended 1e-9.
        assert np.allclose(np.sort(w), np.sort([n_o ** 2, n_o ** 2, n_e ** 2]), rtol=0.0, atol=1e-9)
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


# --- chiral / twisted-nematic director (theta + phi coupled) ---------------------------------------
_CK = dict(K11=11e-12, K22=7e-12, K33=18e-12, eps_para=18.7, eps_perp=4.0, d_planar=4e-6,
           field_model="uniform", nz=121, theta_b_rad=np.radians(89.9))


def test_chiral_decouples_to_director_profile_bvp():
    # phi_top == phi_bottom and q0 = 0 -> twist decouples (phi == const) and theta == director_profile_bvp
    # EXACTLY (same tilt Euler-Lagrange), including the Freedericksz-tilted branch under field.
    ch = chiral_director_profile_bvp(V_app=2.0, phi_bottom_rad=0.0, phi_top_rad=0.0, q0_rad_m=0.0,
                                     n_o=1.52, n_e=1.74, **_CK)
    st = director_profile_bvp(V_app=2.0, K11=_CK["K11"], K33=_CK["K33"], eps_para=_CK["eps_para"],
                              eps_perp=_CK["eps_perp"], d_planar=_CK["d_planar"], field_model="uniform",
                              nz=_CK["nz"], theta_b_rad=_CK["theta_b_rad"], n_o=1.52, n_e=1.74)
    assert ch.success
    assert np.max(np.abs(ch.theta_field_rad - st.theta_field_rad)) < 1e-6
    assert np.max(np.abs(ch.phi_rad)) < 1e-9
    assert ch.n_eff == pytest.approx(st.n_eff, abs=1e-5)


def test_chiral_pure_twist_is_linear_with_analytic_energy():
    # 90 deg TN at V=0, planar: phi(z) linear, theta flat, twist energy = (1/2) K22 phi_t^2 / d.
    ch = chiral_director_profile_bvp(V_app=0.0, phi_bottom_rad=0.0, phi_top_rad=0.5 * np.pi, q0_rad_m=0.0,
                                     **{**_CK, "theta_b_rad": 0.5 * np.pi})
    u = (ch.z_m - ch.z_m[0]) / (ch.z_m[-1] - ch.z_m[0])
    assert np.max(np.abs(ch.phi_rad - 0.5 * np.pi * u)) < 1e-4
    assert np.max(np.abs(ch.theta_field_rad - 0.5 * np.pi)) < 1e-4
    w_an = 0.5 * _CK["K22"] * (0.5 * np.pi) ** 2 / _CK["d_planar"]
    assert ch.twist_energy_J_m2 == pytest.approx(w_an, rel=1e-5)


def test_chiral_cholesteric_undistorted_helix():
    # q0 != 0 with matched plate azimuths -> phi' = q0 uniform, energy = -(1/2) K22 q0^2 d.
    q0 = cholesteric_q0(8e-6)
    ch = chiral_director_profile_bvp(V_app=0.0, phi_bottom_rad=0.0, phi_top_rad=q0 * _CK["d_planar"],
                                     q0_rad_m=q0, **{**_CK, "theta_b_rad": 0.5 * np.pi})
    phz = np.gradient(ch.phi_rad, ch.z_m)
    assert np.max(np.abs(phz - q0)) / q0 < 1e-5
    assert ch.twist_energy_J_m2 == pytest.approx(-0.5 * _CK["K22"] * q0 * q0 * _CK["d_planar"], rel=1e-5)


def test_chiral_tn_freedericksz_monotone():
    # midplane tilt of a 90 deg TN grows monotonically with field.
    mids = []
    for V in (0.3, 0.9, 1.5):
        r = chiral_director_profile_bvp(V_app=V, phi_bottom_rad=0.0, phi_top_rad=0.5 * np.pi, **_CK)
        assert r.success
        mids.append(0.5 * np.pi - r.theta_field_rad[r.theta_field_rad.size // 2])
    assert mids[0] < mids[1] < mids[2]
    assert mids[2] > 0.5                                           # clearly tilted at V=1.5


def test_gooch_tarry_first_minimum():
    # 90 deg TN crossed-polarizer transmission -> first dark minimum at Mauguin u = sqrt(3).
    dn, lam = 0.22, 0.55e-6
    d_min = np.sqrt(3.0) * lam / (2.0 * dn)
    assert mauguin_number(d_min, dn, lam) == pytest.approx(np.sqrt(3.0), abs=1e-9)
    assert gooch_tarry_transmission(d_min, dn, lam) < 1e-12
    assert 0.0 <= gooch_tarry_transmission(1e-6, dn, lam) <= 1.0


def test_chiral_rejects_bad_constants():
    with pytest.raises(ValueError):
        chiral_director_profile_bvp(V_app=1.0, K11=1e-12, K22=-1e-12, K33=1e-12, eps_para=10.0,
                                    eps_perp=4.0, d_planar=4e-6)


def test_c14_elliptic_B_is_exact_up_to_the_bisection_bracket():
    """AUDIT C-14 (2026-11-02): the 1201-point uniform-phi trapezoid over-estimated B(theta_m) by
    1.15x at 89.99 deg, 3.56x at 89.999 deg and 42.09x at the bisection's upper bracket
    pi/2 - 1e-6 (endpoint log singularity). The closed form K(m)/sqrt(dEps) is checked against an
    INDEPENDENT reduction -- Carlson's R_F(0, 1-eta, 1+g eta), a different special function -- and
    against a direct high-precision quadrature of the module-docstring integrand."""
    from scipy.special import elliprf
    from scipy.integrate import quad
    from dynameta.carriers.lc_director import _B
    for dEps, ep in ((DEPS, EP), (10.0, 5.0), (0.5, 3.0)):
        g = dEps / ep
        for tm in (np.radians(1.0), np.radians(45.0), np.radians(89.0), np.radians(89.9),
                   np.radians(89.99), np.radians(89.999), 0.5 * np.pi - 1e-6):
            eta = np.sin(tm) ** 2
            eps_m = ep + dEps * eta
            ref = np.sqrt(eps_m / (dEps * ep)) * elliprf(0.0, np.cos(tm) ** 2, 1.0 + g * eta)
            assert _B(tm, dEps, ep) == pytest.approx(ref, rel=1e-13), (dEps, ep, np.degrees(tm))
            # direct quadrature, split so the endpoint singularity sits on an interval boundary
            f = lambda p: (np.sqrt(eps_m / (dEps * (ep + dEps * eta * np.sin(p) ** 2)))
                           / np.sqrt(np.cos(tm) ** 2 + eta * np.cos(p) ** 2))   # 1 - eta sin^2 p
            brk = [0.0, 0.25 * np.pi, 0.5 * np.pi - 1e-2, 0.5 * np.pi - 1e-4, 0.5 * np.pi]
            direct = sum(quad(f, a, b, epsabs=0.0, epsrel=1e-13, limit=200)[0]
                         for a, b in zip(brk, brk[1:]))
            assert _B(tm, dEps, ep) == pytest.approx(direct, rel=1e-10), (dEps, ep, np.degrees(tm))
    # theta_m -> 0 limit: K(0) = pi/2, so V -> V_th exactly
    assert _B(1e-9, DEPS, EP) == pytest.approx(0.5 * np.pi / np.sqrt(DEPS), rel=1e-15)


def test_c14_over_drive_tilt_obeys_deulings_law():
    """AUDIT C-14: at V = 8 V_th the bisection must return the theta_m of Deuling's law
    V/V_th = (2/pi) K(m), 1 - m = cos^2(theta_m) eps_perp/eps(theta_m) (K33 = K11). With the
    biased quadrature the 42x-high V(theta_m) curve put the root at a SMALLER tilt."""
    from scipy.special import ellipkm1
    Vth = freedericksz_threshold_V(K, DEPS)
    for r in (3.0, 8.0):
        tm = director_profile(K, DEPS, EP, D, r * Vth).theta_max_rad
        one_minus_m = np.cos(tm) ** 2 * EP / (EP + DEPS * np.sin(tm) ** 2)
        # rel 1e-7, not tighter: theta_max is arcsin(sin theta_m) of the reconstruction, and arcsin
        # near 1 costs ~eps/cos(theta_m) -- measured 8.3e-9 at 8 V_th (theta_m = 89.9987 deg)
        assert (2.0 / np.pi) * ellipkm1(one_minus_m) == pytest.approx(r, rel=1e-7), r
    # past the bracket's V (~(2/pi) K at pi/2 - 1e-6, ~9.1 V_th here) the tilt saturates at the bracket
    assert director_profile(K, DEPS, EP, D, 50.0 * Vth).theta_max_rad == pytest.approx(
        0.5 * np.pi - 1e-6, abs=1e-9)                      # same arcsin round-off, 4.4e-11 measured


def _theta_form_z_fraction(th, tm, dEps, ep):
    """INDEPENDENT oracle for the half-cell z-fraction of tilt th: the first integral in the THETA
    variable, z(th) ~ int_0^th sqrt(eps(t)) / sqrt(sin^2 tm - sin^2 t) dt, with t = tm - u^2 removing
    the turning point and the near-singular sin(2 tm - u^2) ~ 2 (pi/2 - tm) + u^2 split out by
    breakpoints -- a different variable, integrand and special-function-free route from _z_map's
    Carlson R_F/R_J reduction."""
    from scipy.integrate import quad

    def f(u):
        t = tm - u * u
        return 2.0 * u * np.sqrt(ep + dEps * np.sin(t) ** 2) / np.sqrt(
            np.sin(u * u) * np.sin(2.0 * tm - u * u))

    def area(t_hi):
        u0, u1 = np.sqrt(tm - t_hi), np.sqrt(tm)
        dl = 0.5 * np.pi - tm
        pts = sorted({u0, u1} | {np.sqrt(k * dl) for k in (0.1, 1.0, 10.0, 100.0)
                                 if u0 < np.sqrt(k * dl) < u1})
        return sum(quad(f, a, b, epsabs=0.0, epsrel=1e-11, limit=200)[0]
                   for a, b in zip(pts, pts[1:]))

    return area(th) / area(tm)


def test_lc_profile_z_map_is_exact_up_to_the_bisection_bracket():
    """AUDIT LC-profile (wave-6 follow-up of C-14, 2026-11-02): director_profile reconstructed theta(z)
    with a 4001-point uniform-phi cumulative trapezoid on the log-singular 1/sqrt(1 - eta sin^2 phi);
    its z-map end read 1.67x at theta_m = 89.999 deg and 13.8x at the bracket, and theta(z) was off by
    up to 0.20 rad at 8 V_th / 0.37 rad at 9 V_th (vs 40-digit mpmath). Now a closed form (Carlson
    R_F + R_J) inverted by bisection. Gate: every returned (z_i, theta_i) must satisfy the
    independent theta-form quadrature z-fraction, three material sets, up to the bracket."""
    from dynameta.carriers.lc_director import _V_of_theta_max
    for dEps, ep in ((DEPS, EP), (10.0, 5.0), (0.5, 3.0)):
        for tm in (np.radians(45.0), np.radians(89.9), np.radians(89.999), 0.5 * np.pi - 1e-6):
            V = _V_of_theta_max(tm, K, dEps, ep) * (1.0 - 1e-15)    # just below the bracket's V
            p = director_profile(K, dEps, ep, D, V, nz=21)
            assert p.theta_max_rad == pytest.approx(tm, abs=1e-9)
            tmr = p.theta_max_rad
            zf = np.minimum(p.z_m, D - p.z_m) / (0.5 * D)
            for zi, thi in zip(zf, p.theta_rad):
                if 0.0 < zi < 1.0 - 1e-12:
                    assert _theta_form_z_fraction(thi, tmr, dEps, ep) == pytest.approx(
                        zi, abs=1e-9), (dEps, ep, np.degrees(tm), zi)
            assert p.theta_rad[0] == 0.0 and p.theta_rad[-1] == 0.0
            assert p.theta_rad[10] == pytest.approx(tmr, abs=1e-15)   # midplane node


def test_lc_profile_analytic_limits():
    """LC-profile: (a) weak relative anisotropy (dEps << eps_perp, constant-field limit) -- the
    classical z/(d/2) = F(phi, eta)/K(eta), sin(phi) = sin(theta)/sin(theta_m); (b) just above V_th
    -- the linearised Freedericksz mode theta = theta_m sin(pi z/d)."""
    from scipy.special import ellipkinc, ellipk
    from dynameta.carriers.lc_director import _V_of_theta_max
    ep, dEps = 1e9, 1.0                                             # g = 1e-9
    for tm in (np.radians(30.0), np.radians(80.0), np.radians(89.5)):
        p = director_profile(K, dEps, ep, D, _V_of_theta_max(tm, K, dEps, ep), nz=41)
        eta = np.sin(tm) ** 2
        half = p.z_m <= 0.5 * D
        phi = np.arcsin(np.clip(np.sin(p.theta_rad[half]) / np.sin(tm), 0.0, 1.0))
        assert np.allclose(ellipkinc(phi, eta) / ellipk(eta), p.z_m[half] / (0.5 * D),
                           rtol=0.0, atol=1e-8), np.degrees(tm)
    tm = 1e-3
    p = director_profile(K, DEPS, EP, D, _V_of_theta_max(tm, K, DEPS, EP), nz=41)
    assert np.allclose(p.theta_rad, tm * np.sin(np.pi * p.z_m / D), rtol=0.0, atol=1e-8)
