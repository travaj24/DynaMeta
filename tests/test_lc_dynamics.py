"""Fast unit tests for the nematic director DYNAMICS (Erickson-Leslie relaxation) in
dynameta/carriers/lc_dynamics.py. Pure numpy/scipy; the rigorous golden-oracle checks live in
validation/lc_director_dynamics.py."""
import math

import numpy as np
import pytest

from dynameta.carriers.lc_dynamics import (
    LCDynamics, v_step, v_rc_mirrored, make_three_stage_voltage_func,
    step_rise_10_90, step_decay_90_10, crossing_time)


def test_waveform_step_and_rc():
    assert v_step(-1.0, 2.0, 1.0) == 0.0
    assert v_step(0.5, 2.0, 1.0) == 2.0
    assert v_step(1.5, 2.0, 1.0) == 0.0
    # RC rise approaches V0; tau<=0 falls back to a step
    assert v_rc_mirrored(10.0, 2.0, 1e9, 1.0) == pytest.approx(2.0, abs=1e-3)
    assert v_rc_mirrored(0.5, 2.0, 1.0, 0.0) == 2.0


def test_three_stage_levels():
    vf = make_three_stage_voltage_func(0.5, 2.0, 0.0, 1.0, 1.0, waveform="step")
    assert vf(0.5) == 0.5      # turn stage
    assert vf(1.5) == 2.0      # max stage
    assert vf(2.5) == 0.0      # decay stage


def test_crossing_and_metrics_on_synthetic_trace():
    t = np.linspace(0.0, 10.0, 101)
    y = 1.0 - np.exp(-t)                                  # rising saturation
    assert crossing_time(t, y, 0.5, "rising") == pytest.approx(math.log(2.0), abs=0.05)
    # a rise then decay about Ton=5: build n_eff-like trace
    on = 1.0 - np.exp(-t)
    rise = step_rise_10_90(t, on, 10.0)
    assert math.isfinite(rise) and rise > 0


def test_tau_analytic_matches_decay():
    # 1-constant, small field-OFF perturbation decays with tau = gamma1 d^2/(K pi^2)
    d = LCDynamics(K11=10e-12, K33=10e-12, gamma1=0.05, eps_para=10.0, eps_perp=5.0,
                   theta_b_rad=0.5 * math.pi, geometry="planar", d_planar=2e-6,
                   field_model="uniform", nz=81)
    tau = d.tau_1const_s()
    z = d.geometry_obj().z_m
    th0 = 0.5 * math.pi - math.radians(2.0) * np.sin(math.pi * z / z[-1])
    t_eval = np.linspace(0.0, 4.0 * tau, 120)
    r = d.simulate(t_eval, lambda t: 0.0, theta0_rad=th0)
    amp = 0.5 * math.pi - r.theta_mid_rad
    m = (t_eval > 0.5 * tau) & (t_eval < 3.0 * tau)
    tau_fit = -1.0 / np.polyfit(t_eval[m], np.log(amp[m]), 1)[0]
    assert abs(tau_fit / tau - 1.0) < 2e-2


def test_pulse_turns_on_and_relaxes():
    # above-threshold pulse: midplane tilts toward the field (theta DROPS from ~pi/2) while ON,
    # then relaxes back toward theta_b when OFF.
    d = LCDynamics(K11=17e-12, K33=18e-12, gamma1=0.085, eps_para=18.7, eps_perp=4.0,
                   theta_b_rad=math.radians(89.9), geometry="planar", d_planar=1e-6,
                   field_model="uniform", nz=61)
    r = d.simulate_pulse(V0=2.0, Ton=3e-3, T_end=10e-3, n_t=200, waveform="step")
    thb = math.radians(89.9)
    on_min = float(np.min(r.theta_mid_rad[r.t_s <= 3e-3]))
    assert on_min < thb - math.radians(20.0)             # tilted well toward the field while ON
    assert abs(float(r.theta_mid_rad[-1]) - thb) < math.radians(2.0)   # relaxed back near theta_b


def test_decay_metric_swing_guard():
    # AUDIT FIX: a barely-switching trace (decay swing below the floor / a small fraction of the ON
    # swing) must return NaN, not solver-noise-driven garbage.
    t = np.linspace(0.0, 10.0, 201)
    Ton = 4.0
    # essentially flat (tiny noise) -> NaN
    rng = np.zeros_like(t) + 1.5
    rng += 1e-6 * np.sin(37.0 * t)
    assert math.isnan(step_decay_90_10(t, rng, Ton))
    # a genuine rise-then-decay -> finite
    on = (t <= Ton)
    y = np.where(on, 1.5 + 0.3 * (t / Ton), 1.8 - 0.3 * (1.0 - np.exp(-(t - Ton))))
    d = step_decay_90_10(t, y, Ton)
    assert math.isfinite(d) and d > 0


def test_simulate_rejects_bad_inputs():
    d = LCDynamics(K11=17e-12, K33=18e-12, gamma1=0.085, eps_para=18.7, eps_perp=4.0,
                   geometry="planar", d_planar=1e-6, field_model="uniform", nz=41)
    with pytest.raises(ValueError):
        d.simulate(np.array([0.0, 1.0]), lambda t: 0.0)   # < 5 time points


def test_weak_anchoring_dynamics_surface_tilts():
    # finite W lets the surface director rotate toward the field while ON (strong anchoring pins it).
    base = dict(K11=17e-12, K33=18e-12, gamma1=0.085, eps_para=18.7, eps_perp=4.0,
                theta_b_rad=math.radians(89.9), geometry="planar", d_planar=1e-6,
                field_model="uniform", nz=61)
    strong = LCDynamics(**base)
    weak = LCDynamics(W_anchor_J_m2=3e-4, gamma_s_Pa_s_m=1e-8, **base)
    tau = strong.tau_1const_s()
    t = np.linspace(0.0, 25.0 * tau, 250)
    rs = strong.simulate(t, lambda tt: 2.0, theta0_rad=None)
    rw = weak.simulate(t, lambda tt: 2.0, theta0_rad=None)
    thb = math.radians(89.9)
    # strong stays pinned at the surface; weak rotates measurably away from the easy axis
    assert abs(float(rs.theta_zt_rad[0, -1]) - thb) < math.radians(0.5)
    assert (thb - float(rw.theta_zt_rad[0, -1])) > math.radians(5.0)


def test_weak_anchoring_dynamics_matches_static_bvp():
    # the surface torque balance with surface viscosity relaxes to the STATIC weak-anchoring BVP.
    from dynameta.carriers.lc_director import director_profile_bvp
    W = 3e-4
    st = director_profile_bvp(V_app=2.0, K11=17e-12, K33=18e-12, eps_para=18.7, eps_perp=4.0,
                              d_planar=1e-6, theta_b_rad=math.radians(89.9), field_model="uniform",
                              nz=81, W_anchor_J_m2=W)
    dy = LCDynamics(K11=17e-12, K33=18e-12, gamma1=0.085, eps_para=18.7, eps_perp=4.0,
                    theta_b_rad=math.radians(89.9), geometry="planar", d_planar=1e-6,
                    field_model="uniform", nz=81, W_anchor_J_m2=W, gamma_s_Pa_s_m=1e-8)
    tau = dy.tau_1const_s()
    r = dy.simulate(np.linspace(0.0, 30.0 * tau, 200), lambda t: 2.0, theta0_rad=None)
    dmax = float(np.max(np.abs(r.theta_zt_rad[:, -1] - st.theta_field_rad)))
    assert math.degrees(dmax) < 0.6


# --- BACKFLOW (audit C-2 Leslie coupling swap + N4 theta-dependent Miesowicz eta) -------------------
# 5CB Leslie coefficients. LIT = the literature set (alpha2 = -0.0812, alpha3 = -0.0036, measured
# Miesowicz eta_b = 0.0204 Pa s); MOD = the module's rounded defaults, whose alpha3/alpha2 ratio pins
# the flow-alignment angle at 10.96 deg. gamma1 = alpha3 - alpha2 (the Leslie identity) in both.
_LIT = dict(alpha2_Pa_s=-0.0812, alpha3_Pa_s=-0.0036)
_MOD = dict(alpha2_Pa_s=-0.08, alpha3_Pa_s=-0.003)
_CELL = dict(K11=17e-12, K33=18e-12, eps_para=18.7, eps_perp=4.0, theta_b_rad=math.radians(89.9),
             geometry="planar", d_planar=1e-6, field_model="uniform", n_o=1.56, n_e=1.92, nz=61)


def test_backflow_miesowicz_anchors_5cb():
    # AUDIT C-2 + N4 GATE: gamma1_eff/gamma1 must be ~0.99 PLANAR (theta = pi/2) and ~0.19
    # HOMEOTROPIC (theta = 0) for 5CB -- the shipped code had exactly the complementary profile
    # (0.9987 homeotropic / 0.0588 planar) because alpha2/alpha3 were swapped in the coupling m.
    for coeffs in (_LIT, _MOD):
        g1 = coeffs["alpha3_Pa_s"] - coeffs["alpha2_Pa_s"]
        lc = LCDynamics(gamma1=g1, include_backflow=True, **coeffs, **_CELL)
        assert float(lc.gamma1_eff_of_theta(0.5 * np.pi)) / g1 == pytest.approx(0.99, rel=5e-2)
        assert float(lc.gamma1_eff_of_theta(0.0)) / g1 == pytest.approx(0.19, rel=5e-2)
    # Parodi consistency: eta_c - eta_b = -(alpha2 + alpha3) reproduces the MEASURED 5CB eta_c
    lit = LCDynamics(gamma1=0.0776, include_backflow=True, **_LIT, **_CELL)
    assert lit.eta_c_effective_Pa_s() == pytest.approx(0.1052, abs=1e-6)
    assert float(lit.eta_shear_of_theta(0.5 * np.pi)) == pytest.approx(0.0204, abs=1e-9)


def test_backflow_flow_alignment_angle_5cb():
    # AUDIT C-2 convention pin: the Leslie flow-alignment angle (from the FLOW direction) is
    # atan sqrt(alpha3/alpha2) = 10.96 deg for the module's 5CB set; m(theta) vanishes there.
    lc = LCDynamics(gamma1=0.077, include_backflow=True, **_MOD, **_CELL)
    p_L = math.degrees(lc.flow_alignment_angle_rad())
    assert p_L == pytest.approx(10.96, abs=5e-2)
    assert float(lc.leslie_coupling(0.5 * np.pi - lc.flow_alignment_angle_rad())) == \
        pytest.approx(0.0, abs=1e-15)
    # a tumbling nematic (alpha3/alpha2 < 0) has no aligning solution
    assert math.isnan(LCDynamics(gamma1=0.08, include_backflow=True, alpha2_Pa_s=-0.08,
                                 alpha3_Pa_s=+0.003, **_CELL).flow_alignment_angle_rad())


def test_backflow_near_threshold_planar_cell_has_no_speedup():
    # AUDIT C-2 GATE: a cell whose director stays near PLANAR must show essentially NO backflow
    # enhancement (m -> alpha3, tiny). The shipped swapped coefficients reported x2.44 here.
    g1 = 0.077
    base = dict(gamma1=g1, **_CELL)
    Vth = 1.1354                                     # pi sqrt(K11/(eps0 dEps)), K11 = 17 pN
    pk = dict(V0=1.10 * Vth, Ton=10e-3, T_end=30e-3, n_t=300)
    rno = LCDynamics(**base).simulate_pulse(**pk)
    rbf = LCDynamics(include_backflow=True, **_MOD, **base).simulate_pulse(**pk)
    assert rno.rise_10_90_s / rbf.rise_10_90_s == pytest.approx(1.0, abs=2e-2)


def test_backflow_strongly_driven_cell_speedup_band():
    # AUDIT C-2 re-grade: backflow in a STRONGLY driven cell is REAL, not an artefact of the swap --
    # ~1.34-1.46x with the corrected coefficients. Pinned on the shipped oracle cell's gamma1 = 0.085,
    # which is NOT Leslie-consistent with these alphas and therefore also raises the identity warning.
    base = dict(gamma1=0.085, **_CELL)
    pk = dict(V0=5.0, Ton=1e-3, T_end=5e-3, n_t=1200)
    rno = LCDynamics(**base).simulate_pulse(**pk)
    with pytest.warns(RuntimeWarning, match="Leslie identity"):
        rbf = LCDynamics(include_backflow=True, **_MOD, **base).simulate_pulse(**pk)
    assert 1.34 < rno.rise_10_90_s / rbf.rise_10_90_s < 1.46
    assert 1.34 < rno.decay_90_10_s / rbf.decay_90_10_s < 1.46


def test_backflow_speeds_switching_and_off_is_identical():
    # RE-BASELINED for C-2/N4: same cell, corrected m(theta) and theta-dependent eta(theta).
    base = dict(gamma1=0.077, **_CELL)
    pk = dict(V0=2.0, Ton=3e-3, T_end=9e-3, n_t=160)
    rno = LCDynamics(**base).simulate_pulse(**pk)
    rbf = LCDynamics(include_backflow=True, **_MOD, **base).simulate_pulse(**pk)
    rz = LCDynamics(include_backflow=True, alpha2_Pa_s=0.0, alpha3_Pa_s=0.0,
                    **base).simulate_pulse(**pk)
    # backflow (effective viscosity reduced) speeds both rise and decay
    assert rbf.rise_10_90_s < rno.rise_10_90_s
    assert rbf.decay_90_10_s < rno.decay_90_10_s
    # alpha2 = alpha3 = 0 -> m = 0 -> gamma1_eff = gamma1 -> byte-identical to no-backflow
    assert rz.rise_10_90_s == pytest.approx(rno.rise_10_90_s, abs=1e-12)
    assert rz.decay_90_10_s == pytest.approx(rno.decay_90_10_s, abs=1e-12)


def test_backflow_scalar_eta_override_is_backward_compatible():
    # AUDIT N4: eta_shear_Pa_s now DEFAULTS to None (theta-dependent Miesowicz eta). The old scalar
    # behaviour is still reachable and must pin eta at every theta.
    lc = LCDynamics(gamma1=0.077, include_backflow=True, eta_shear_Pa_s=0.08, **_MOD, **_CELL)
    for th in (0.0, 0.3, 0.5 * np.pi):
        assert float(lc.eta_shear_of_theta(th)) == pytest.approx(0.08, abs=1e-15)
    # the theta-dependent default is NOT the scalar (documented behaviour change)
    dflt = LCDynamics(gamma1=0.077, include_backflow=True, **_MOD, **_CELL)
    assert float(dflt.eta_shear_of_theta(0.0)) != pytest.approx(0.08, abs=1e-6)
    with pytest.raises(ValueError):
        LCDynamics(gamma1=0.077, include_backflow=True, eta_shear_Pa_s=-1.0, **_MOD,
                   **_CELL).simulate_pulse(V0=2.0, Ton=3e-3, T_end=9e-3, n_t=60)


def test_backflow_solver_eta_matches_its_public_twin_when_alpha1_crosses_zero():
    """FIX-VERIFY W1 item 7. ``eta(theta) = eta_c cos^2 + eta_b sin^2 + alpha1 sin^2 cos^2``: the
    validation covers eta_b, eta_c > 0, but ``alpha1_Pa_s`` -- the only sign-free term -- was
    UNVALIDATED, and a large negative alpha1 sends eta(theta) through ZERO at intermediate tilt
    (eta_c = 0.1052, eta_b = 0.0204, alpha1 = -1 gives eta(45 deg) = -0.2186).  The public
    ``gamma1_eff_of_theta`` guards with ``np.maximum(np.abs(eta), 1e-300)``; the solver's inline
    copy did not, so the SAME configuration divided by ~0 inside ``rhs`` and, past the crossing,
    turned the backflow REDUCTION into an increase of gamma1_eff above gamma1.  The two are now
    the same expression."""
    kw = dict(gamma1=0.077, include_backflow=True, alpha2_Pa_s=-0.08, alpha3_Pa_s=-0.003)
    lc = LCDynamics(alpha1_Pa_s=-1.0, **kw, **_CELL)

    # eta(theta) genuinely crosses zero, so the guard is REACHABLE from public kwargs
    th = np.linspace(0.0, 0.5 * np.pi, 401)
    eta = np.asarray(lc.eta_shear_of_theta(th))
    assert eta.min() < 0.0 < eta.max()

    # the public twin never lets gamma1_eff exceed gamma1 and never returns a non-finite value
    g1e = np.asarray(lc.gamma1_eff_of_theta(th))
    assert np.all(np.isfinite(g1e)) and np.all(g1e <= 0.077 + 1e-15)

    # the solver takes the same branch: finite trajectory + the PSD warning, not inf/NaN
    with pytest.warns(RuntimeWarning):
        res = lc.simulate_pulse(V0=3.0, Ton=1e-3, T_end=3e-3, n_t=60)
    assert np.all(np.isfinite(res.theta_zt_rad))

    # a non-finite alpha1 is now rejected up front instead of poisoning the march
    with pytest.raises(ValueError, match="alpha1_Pa_s"):
        LCDynamics(alpha1_Pa_s=float("nan"), **kw, **_CELL).simulate_pulse(
            V0=2.0, Ton=1e-3, T_end=3e-3, n_t=60)

    # and the guard is a NO-OP for a physical set: alpha1 = 0 is byte-identical to before
    ref = LCDynamics(gamma1=0.077, include_backflow=True, **_MOD, **_CELL)
    a = ref.simulate_pulse(V0=2.0, Ton=3e-3, T_end=9e-3, n_t=60)
    b = LCDynamics(gamma1=0.077, include_backflow=True, alpha1_Pa_s=0.0, **_MOD,
                   **_CELL).simulate_pulse(V0=2.0, Ton=3e-3, T_end=9e-3, n_t=60)
    assert np.array_equal(a.theta_zt_rad, b.theta_zt_rad)


def test_backflow_psd_violation_warns_and_floors():
    # AUDIT C-2: positive-definiteness of [[gamma1, m], [m, eta]] requires gamma1*eta >= m^2. The
    # 0.05*gamma1 floor must fire ONLY there -- and say so instead of silently clamping.
    bad = LCDynamics(gamma1=0.077, include_backflow=True, alpha2_Pa_s=-0.08, alpha3_Pa_s=-0.003,
                     eta_shear_Pa_s=0.01, **_CELL)                     # gamma1*eta = 7.7e-4 << m^2
    assert float(bad.gamma1_eff_of_theta(0.0, apply_floor=False)) < 0.0
    assert float(bad.gamma1_eff_of_theta(0.0)) == pytest.approx(0.05 * 0.077, rel=1e-12)
    with pytest.warns(RuntimeWarning, match="positive-definiteness"):
        bad.simulate_pulse(V0=3.0, Ton=1e-3, T_end=3e-3, n_t=60)
    # the Leslie-consistent 5CB set is PSD everywhere -> no floor
    good = LCDynamics(gamma1=0.077, include_backflow=True, **_MOD, **_CELL)
    th = np.linspace(0.0, 0.5 * np.pi, 501)
    assert np.all(good.gamma1_eff_of_theta(th, apply_floor=False) > 0.05 * 0.077)
