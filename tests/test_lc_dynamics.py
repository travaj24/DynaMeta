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


def test_backflow_speeds_switching_and_off_is_identical():
    base = dict(K11=17e-12, K33=18e-12, gamma1=0.085, eps_para=18.7, eps_perp=4.0,
                theta_b_rad=math.radians(89.9), geometry="planar", d_planar=1e-6,
                field_model="uniform", n_o=1.56, n_e=1.92, nz=61)
    pk = dict(V0=2.0, Ton=3e-3, T_end=9e-3, n_t=160)
    rno = LCDynamics(**base).simulate_pulse(**pk)
    rbf = LCDynamics(include_backflow=True, alpha2_Pa_s=-0.08, alpha3_Pa_s=-0.003,
                     eta_shear_Pa_s=0.08, **base).simulate_pulse(**pk)
    rz = LCDynamics(include_backflow=True, alpha2_Pa_s=0.0, alpha3_Pa_s=0.0,
                    eta_shear_Pa_s=0.08, **base).simulate_pulse(**pk)
    # backflow (effective viscosity reduced) speeds both rise and decay
    assert rbf.rise_10_90_s < rno.rise_10_90_s
    assert rbf.decay_90_10_s < rno.decay_90_10_s
    # alpha2 = alpha3 = 0 -> g = 0 -> gamma1_eff = gamma1 -> byte-identical to no-backflow
    assert rz.rise_10_90_s == pytest.approx(rno.rise_10_90_s, abs=1e-12)
    assert rz.decay_90_10_s == pytest.approx(rno.decay_90_10_s, abs=1e-12)
