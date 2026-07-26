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

import warnings

import numpy as np
import pytest

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
    assert tr.meta["quasi_static_valid"]                        # a HOT profile stays in regime
    amp.clear_temperature_profile()
    tr_cold = simulate_transient(amp, np.linspace(0.0, 12e-3, 200), n_nodes=81)
    assert abs(float(tr_cold.signal_gain_dB[-1, 0]) - G_cold) < 0.05    # and the cold model still is
    assert tr_cold.meta["quasi_static_valid"]


# ---- A-7: the frozen-inversion step announces when it leaves its valid regime ---------------

def test_transient_flags_ase_runaway_on_a_sub_reference_temperature_profile():
    """audit A-7: the A-5 temperature path also reaches T < T_ref, where McCumber > 1 BOOSTS
    sigma_e. Past ~227 K this amplifier's frozen-inversion step runs the ASE away (the step
    propagates exp(INT g dz) at a FIXED inversion, so in-step ASE never depletes the gain that
    made it) and the march settles ~31 dB below its own solve(). That used to be SILENT. It must
    now raise a RuntimeWarning AND carry meta['quasi_static_valid'] = False -- while the hot side
    of the same sweep, and the same amplifier at 300 K, stay clean and keep agreeing with
    solve()."""
    z = np.linspace(0.0, 5.0, 41)
    t = np.linspace(0.0, 25e-3, 400)

    # --- the broken cold case: the flag fires, the warning names the limitation ---
    cold = _hot_yb()
    cold.set_temperature_profile(z, np.full(z.size, 220.0), T_ref_K=300.0)
    G_ss = float(cold.solve(n_nodes=81).signal_gain_dB[0])
    with np.errstate(all="ignore"), pytest.warns(RuntimeWarning, match="quasi-static"):
        tr = simulate_transient(cold, t, n_nodes=81)
    assert tr.meta["quasi_static_valid"] is False
    assert tr.meta["validity_warning"] and "ASE" in tr.meta["validity_warning"]
    # the flag is not decorative: this run really is ~31 dB away from its own steady state
    assert float(G_ss) - float(tr.signal_gain_dB[-1, 0]) > 20.0
    # and both measured margins are far past their documented limits
    lim = tr.meta["validity_limits"]
    assert tr.meta["max_ase_to_launched"] > 1e3 * lim["ase_to_launched"]
    assert tr.meta["max_ase_gain_integral"] > 2.0 * lim["ase_gain_integral"]

    # --- the hot / in-regime cases stay clean (no warning, flag True, agrees with solve()) ---
    for T in (None, 300.0, 500.0, 800.0):
        amp = _hot_yb()
        if T is not None:
            amp.set_temperature_profile(z, np.full(z.size, T), T_ref_K=300.0)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            ok = simulate_transient(amp, t, n_nodes=81)
        assert ok.meta["quasi_static_valid"] is True, T
        assert ok.meta["validity_warning"] is None
        assert not [w for w in caught if issubclass(w.category, RuntimeWarning)
                    and "quasi-static" in str(w.message)], T
        assert abs(float(ok.signal_gain_dB[-1, 0])
                   - float(amp.solve(n_nodes=81).signal_gain_dB[0])) < 0.05, T


def test_transient_flags_ase_dominance_at_a_tiny_signal_but_not_on_a_healthy_edfa():
    """The pre-existing half of A-7 (no temperature profile involved): the same 400 W Yb booster
    driven at a 1 uW signal lands 54 dB below its own solve(). The threshold is referenced to the
    LAUNCHED power, not to the signal, precisely so a healthy C-band EDFA -- which legitimately
    carries 30-3000x more in-band ASE than signal and agrees with solve() to 1e-3 dB -- does NOT
    trip it."""
    t = np.linspace(0.0, 25e-3, 400)
    tiny = _hot_yb().with_signals([Signal(1e-6, 1.03e-6)])
    G_ss = float(tiny.solve(n_nodes=81).signal_gain_dB[0])
    with np.errstate(all="ignore"), pytest.warns(RuntimeWarning, match="OUT OF ITS VALID REGIME"):
        tr = simulate_transient(tiny, t, n_nodes=81)
    assert tr.meta["quasi_static_valid"] is False
    assert G_ss - float(tr.signal_gain_dB[-1, 0]) > 40.0

    edfa = FiberAmplifier(ER, FiberSpec(1.6e-6, 0.22, 8.0e24, 8.0), [Pump(0.25, 0.98e-6)],
                          [Signal(1e-5, 1.55e-6)], AseBand(1.50e-6, 1.60e-6, n_bins=16))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ok = simulate_transient(edfa, t, n_nodes=81)
    assert ok.meta["quasi_static_valid"] is True
    assert not [w for w in caught if issubclass(w.category, RuntimeWarning)]
    # ASE really does dominate the signal here -- an ASE/SIGNAL threshold would have false-fired
    r = edfa.solve(n_nodes=81)
    ase_out = float(np.sum(r.power_W[[i for i, k in enumerate(r.kind)
                                      if k == "ase" and r.u[i] > 0], -1]))
    sig_out = float(r.power_W[[i for i, k in enumerate(r.kind) if k == "signal"][0], -1])
    assert ase_out > 10.0 * sig_out
    assert ok.meta["max_ase_to_launched"] < ok.meta["validity_limits"]["ase_to_launched"]
    assert abs(float(ok.signal_gain_dB[-1, 0]) - float(r.signal_gain_dB[0])) < 0.05
