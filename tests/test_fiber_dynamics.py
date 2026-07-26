"""Transient-dynamics gates (audit A-1, A-5). The time-domain march must be the SAME PHYSICS as
the steady-state solve OF THE SAME OBJECT: every opt-in the steady state honours -- the raw
upconversion_C_up spelling and the axial temperature profile solve_with_thermal_feedback leaves
SET -- has to travel into simulate_transient AND into the clone that seeds it.

Independent oracle: the steady-state solver itself. Its quadratic upconversion fixed point
(steady_state._nbar2_c) and its per-z McCumber sigma_e scaling are computed by completely separate
code from the transient's exponential integrator, so agreement in the long-time limit is a real
cross-check rather than a tautology. The failure mode both audit findings describe is silent
(a dropped kwarg on a rebuild), so each gate also pins the SIZE of the effect being carried --
otherwise a re-broken clone would still "agree" with a model that ignores the same term twice."""

import numpy as np

from dynameta.optics.fiber_amp.concentration import ConcentrationModel
from dynameta.optics.fiber_amp.dynamics import simulate_transient
from dynameta.optics.fiber_amp.metrics import gain_compression_curve
from dynameta.optics.fiber_amp.spectroscopy import erbium, ytterbium
from dynameta.optics.fiber_amp.steady_state import AseBand, FiberAmplifier, Pump, Signal
from dynameta.optics.fiber_amp.thermal import ThermalModel, solve_with_thermal_feedback
from dynameta.optics.fiber_amp.waveguide import FiberSpec

ER = erbium()
C_UP = 3e-23                      # the value tests/test_fiber_amp.py already uses


def _amp(*, c_up=0.0, conc=None):
    return FiberAmplifier(ER, FiberSpec(2.0e-6, 0.20, 2.0e25, 6.0),
                          [Pump(120e-3, 0.980e-6, "fwd")], [Signal(50e-6, 1.560e-6)], None,
                          upconversion_C_up=c_up, concentration=conc)


def _hot_yb():
    """A cladding-pumped Yb amplifier hot enough (poor convection, 400 W of pump) that the
    McCumber softening is worth several dB -- the regime where dropping the profile matters."""
    fib = FiberSpec(core_radius_m=5.0e-6, na=0.08, n_t_m3=6.0e25, length_m=5.0,
                    clad_radius_m=62.5e-6)
    return FiberAmplifier(ytterbium(), fib, [Pump(400.0, 0.976e-6, "fwd", cladding=True)],
                          [Signal(5.0, 1.03e-6)], AseBand(1.00e-6, 1.08e-6, n_bins=8))


# ---- A-1: the raw upconversion_C_up opt-in reaches the transient ---------------------------

def test_transient_identical_for_both_upconversion_spellings():
    # FiberAmplifier normalises concentration.c_up_m3_s and the raw upconversion_C_up= kwarg into
    # ONE attribute; the transient gated on the ConcentrationModel OBJECT instead, so the raw
    # spelling was dropped and its march came out bit-equal to the C_up = 0 ideal (audit A-1).
    t = np.linspace(0.0, 40e-3, 400)
    raw = simulate_transient(_amp(c_up=C_UP), t, n_nodes=31, nbar2_0=0.10)
    conc = simulate_transient(_amp(conc=ConcentrationModel(c_up_m3_s=C_UP)), t, n_nodes=31,
                              nbar2_0=0.10)
    ideal = simulate_transient(_amp(), t, n_nodes=31, nbar2_0=0.10)
    assert np.array_equal(raw.nbar2_zt, conc.nbar2_zt)          # same physics, either spelling
    assert not np.array_equal(raw.nbar2_zt, ideal.nbar2_zt)     # and NOT the ideal model
    # the term is worth 4.49 dB of end gain at this (repo-standard) C_up -- pin the size so a
    # future re-break cannot pass by ignoring upconversion consistently everywhere
    assert float(ideal.signal_gain_dB[-1, 0] - raw.signal_gain_dB[-1, 0]) > 4.0


def test_transient_long_time_limit_matches_steady_state_under_raw_cup():
    # the transient must converge onto the steady solver's own quadratic upconversion fixed point
    amp = _amp(c_up=C_UP)
    t = np.linspace(0.0, 40e-3, 800)
    tr = simulate_transient(amp, t, n_nodes=31, nbar2_0=0.10)
    ss = amp.solve(n_nodes=31)
    assert abs(float(tr.signal_gain_dB[-1, 0]) - float(ss.signal_gain_dB[0])) < 0.01
    assert abs(float(tr.nbar2_zt[-1].mean()) - float(ss.nbar2_z.mean())) < 1e-3


def test_transient_steady_state_seed_carries_upconversion():
    # the nbar2_0=None path seeds from _amp_with_boundary(...).solve(); that rebuild dropped the
    # raw C_up too, so the march started from an inversion its own dynamics could never hold
    amp = _amp(c_up=C_UP)
    t = np.linspace(0.0, 5e-3, 60)
    tr = simulate_transient(amp, t, n_nodes=31)
    ss = amp.solve(n_nodes=31)
    start_err = abs(float(tr.nbar2_zt[0].mean()) - float(ss.nbar2_z.mean()))
    assert start_err < 1e-3                                     # seeded ON the fixed point
    drift = float(np.max(np.abs(tr.nbar2_zt - tr.nbar2_zt[0])))
    assert drift < 5e-3                                         # and stays there


# ---- A-5: the axial temperature profile survives every clone -------------------------------

def test_thermal_profile_survives_metrics_clone_and_transient():
    # solve_with_thermal_feedback leaves amp._Tz SET; every metrics clone and the transient used
    # to rebuild a COLD amplifier from it, so metrics.*(amp) silently disagreed with amp.solve()
    amp = _hot_yb()
    G_cold = float(amp.solve(n_nodes=81).signal_gain_dB[0])
    res, T_z, info = solve_with_thermal_feedback(amp, ThermalModel(h_conv_W_m2K=50.0),
                                                 62.5e-6, n_nodes=81)
    assert T_z.max() > 400.0 and info["converged_T"]            # genuinely hot
    G_hot = float(amp.solve(n_nodes=81).signal_gain_dB[0])
    assert G_cold - G_hot > 1.0                                 # the profile is worth >1 dB here
    # the metric clone re-solves the SAME operating point: it must reproduce amp.solve() exactly
    G_metric = float(gain_compression_curve(amp, [5.0], signal_index=0).gain_dB[0])
    assert abs(G_metric - G_hot) < 0.02
    # ... and so must the transient's long-time limit (its own McCumber-scaled march)
    tr = simulate_transient(amp, np.linspace(0.0, 12e-3, 200), n_nodes=81)
    assert abs(float(tr.signal_gain_dB[-1, 0]) - G_hot) < 0.05
    amp.clear_temperature_profile()
    tr_cold = simulate_transient(amp, np.linspace(0.0, 12e-3, 200), n_nodes=81)
    assert abs(float(tr_cold.signal_gain_dB[-1, 0]) - G_cold) < 0.05    # and the cold model still is
