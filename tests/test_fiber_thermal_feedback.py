"""Thermal-lens + distributed-T(z)-feedback gates (dossier Modules 4-5).

The load-bearing oracle is UNIFORMITY: a constant T(z) profile through the per-z McCumber
machinery (set_temperature_profile) must reproduce the GLOBAL spectroscopy.at_temperature
scaling exactly -- the two code paths implement the same physics and must agree to solver
tolerance. The feedback gates then check the closed loop's direction and self-consistency."""

import numpy as np

from dynameta.optics.fiber_amp.spectroscopy import at_temperature, ytterbium
from dynameta.optics.fiber_amp.steady_state import FiberAmplifier, Pump, Signal
from dynameta.optics.fiber_amp.thermal import (ThermalModel, heat_load_per_m,
                                               solve_with_thermal_feedback,
                                               thermal_guiding_onset_Q_per_m,
                                               thermal_lens_focal_power_per_m,
                                               thermo_optic_phase_rad, total_heat_W)
from dynameta.optics.fiber_amp.waveguide import FiberSpec


def _yb_amp(ion=None, pump_W=60.0, sig_W=0.5):
    fib = FiberSpec(core_radius_m=5.0e-6, na=0.07, n_t_m3=6.0e25, length_m=8.0,
                    clad_radius_m=62.5e-6)
    return FiberAmplifier(ion if ion is not None else ytterbium(), fib,
                          [Pump(pump_W, 0.976e-6, cladding=True)],
                          [Signal(sig_W, 1.03e-6)])


def test_thermal_lens_and_onset_and_phase_pins():
    # dossier Module 4 gates: dT_core = Q/(4 pi k) = 0.0577 K per W/m; lens power carries the
    # extra parabola factor 2; onset ~1.8 kW/m at NA=0.06; phase ~28 rad for 10 W at 1.55 um
    from dynameta.optics.fiber_amp.thermal import peak_temperature_rise
    a = 10e-6
    m = ThermalModel(h_conv_W_m2K=1e12)               # kill convection: pure core conduction
    dT_core = peak_temperature_rise(1.0, a, a * (1.0 + 1e-12), m)
    assert abs(dT_core - 1.0 / (4.0 * np.pi * 1.38)) < 1e-4          # 0.0577 K/(W/m)
    D = thermal_lens_focal_power_per_m(100.0, a)      # 100 W/m in a 10-um core
    D_expect = 1.2e-5 * 100.0 / (2.0 * np.pi * 1.45 * 1.38 * a ** 2)
    assert abs(D / D_expect - 1.0) < 1e-12
    q_on = thermal_guiding_onset_Q_per_m(0.06)
    assert 1.5e3 < q_on < 2.1e3
    assert 26.0 < thermo_optic_phase_rad(10.0, 1.55e-6) < 31.0


def test_uniform_profile_equals_global_at_temperature():
    T1 = 350.0
    ion = ytterbium()
    a_glob = _yb_amp(ion=at_temperature(ion, T1))
    r_glob = a_glob.solve(n_nodes=121)
    a_prof = _yb_amp(ion=ion)
    z = np.linspace(0.0, a_prof.fiber.length_m, 9)
    a_prof.set_temperature_profile(z, np.full(z.size, T1), T_ref_K=300.0)
    r_prof = a_prof.solve(n_nodes=121)
    assert abs(float(r_prof.signal_gain_dB[0]) - float(r_glob.signal_gain_dB[0])) < 1e-6
    assert float(np.max(np.abs(r_prof.nbar2_z - r_glob.nbar2_z))) < 1e-8


def test_feedback_cold_amplifier_is_a_no_op():
    # truly cold: mW pump AND mW seed (a watt-class seed would dump real reabsorption heat)
    amp = _yb_amp(pump_W=5e-3, sig_W=1e-4)
    base = _yb_amp(pump_W=5e-3, sig_W=1e-4).solve(n_nodes=101)
    res, T_z, info = solve_with_thermal_feedback(amp, ThermalModel(), 62.5e-6, n_nodes=101)
    assert info["converged_T"] and info["iterations"] == 1
    assert float(np.max(T_z - 300.0)) < 0.2
    assert abs(float(res.signal_gain_dB[0]) - float(base.signal_gain_dB[0])) < 1e-9


def test_feedback_coolant_differs_from_reference():
    # adversarial-verifier regression: with T_coolant != T_ref the FIRST (unprofiled) solve
    # represents T_ref, so the loop must NOT declare convergence against the coolant on
    # iteration 1 -- a cold 350 K-cooled amplifier must converge to the 350 K physics.
    # NOTE the SHORT fiber: a long one absorbs the 1030 nm seed to ~-124 dB, below the ODE
    # atol (1e-15 W) -- gains there are integrator noise, not physics (probe-verified: the
    # per-z machinery agreed with the global path to 3.5e-10 in nbar2 while the noise-floor
    # 'gain' scattered by 0.05 dB). Keep signals far above atol in gain-comparing tests.
    from dynameta.optics.fiber_amp.spectroscopy import at_temperature, ytterbium

    def mk(ion=None):
        fib = FiberSpec(core_radius_m=5.0e-6, na=0.07, n_t_m3=6.0e25, length_m=0.5,
                        clad_radius_m=62.5e-6)
        return FiberAmplifier(ion if ion is not None else ytterbium(), fib,
                              [Pump(5e-3, 0.976e-6, cladding=True)], [Signal(1e-4, 1.03e-6)])

    model = ThermalModel(T_coolant_K=350.0)
    res, T_z, info = solve_with_thermal_feedback(mk(), model, 62.5e-6, T_ref_K=300.0,
                                                 n_nodes=101)
    assert info["converged_T"] and info["iterations"] >= 2      # no first-iteration escape
    g_fb = float(res.signal_gain_dB[0])
    g300 = float(mk().solve(n_nodes=101).signal_gain_dB[0])
    g350 = float(mk(ion=at_temperature(ytterbium(), 350.0)).solve(n_nodes=101)
                 .signal_gain_dB[0])
    assert abs(g350 - g300) > 5.0 * abs(g_fb - g350)            # lands on the 350 K physics
    assert float(np.min(T_z)) > 349.0                           # the profile is at the coolant


def test_feedback_hot_amplifier_direction_and_self_consistency():
    model = ThermalModel(h_conv_W_m2K=300.0)          # weakly-cooled: a real temperature rise
    amp = _yb_amp(pump_W=60.0)
    base = _yb_amp(pump_W=60.0).solve(n_nodes=121)
    res, T_z, info = solve_with_thermal_feedback(amp, model, 62.5e-6, n_nodes=121, tol_K=0.1)
    assert info["converged_T"]
    dT = T_z - 300.0
    assert float(dT.max()) > 3.0                      # the fiber actually heats
    # T peaks where the heat is deposited (pump end, co-pumped)
    q = info["Q_per_m"]
    assert abs(int(np.argmax(dT)) - int(np.argmax(q))) <= 2
    # direction: hotter fiber -> sigma_e(1030) falls (McCumber, h nu < eps) -> LESS gain
    assert float(res.signal_gain_dB[0]) < float(base.signal_gain_dB[0])
    # self-consistency: T recomputed from the RETURNED heat profile matches the fixed point
    coef = (1.0 / (4.0 * np.pi * model.core_k_W_mK)
            + np.log(62.5e-6 / amp.fiber.core_radius_m) / (2.0 * np.pi * model.clad_k_W_mK)
            + 1.0 / (2.0 * np.pi * 62.5e-6 * model.h_conv_W_m2K))
    T_check = 300.0 + coef * np.maximum(heat_load_per_m(res), 0.0)
    assert float(np.max(np.abs(T_check - T_z))) < 0.5
    # and the heat number itself is physical: bounded by launched pump
    assert 0.0 < total_heat_W(res) < 60.0
