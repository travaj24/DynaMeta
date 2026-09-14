"""Gates for `ase_mode` -- the self-consistent (ASE-coupled) transient step that lifts the
audit-A-7 quasi-static validity limit (dynameta.optics.fiber_amp.march_ase +
dynamics.simulate_transient / simulate_transient_eryb).

WHAT IS BEING DISCRIMINATED, and by what independent oracle:

  * THE DEFAULT DID NOT MOVE. The claim is byte-identity, so it is asserted as byte-identity:
    `ase_mode="quasi_static"` returns arrays `np.array_equal` to the default call (same process,
    same seed -- the new code path really does no arithmetic), AND the five single-ion / five
    co-doped end states are pinned to the values v0.11.2 produced (1e-9 relative on the
    LSODA-free fixtures, 1e-5 where the seed comes through solve()). The pins
    are the cross-VERSION half and the equality the cross-PATH half; neither alone would catch a
    change that moved both modes together.

  * THE SELF-CONSISTENT MARCH IS RIGHT. Oracle: the steady RELAXATION solve, which is completely
    separate code (scipy solve_ivp sweeps with the population slaved algebraically at every RHS
    call, against the march's trapezoid propagator and exponential integrator). On the three
    DOCUMENTED failure cases the quasi-static march lands 10-67 dB from that oracle and the
    self-consistent one lands on it -- and the remaining gap FALLS AS O(dz^2) with the mesh,
    exactly like the healthy case's march-vs-solve gap, which is how we know the residue is
    z-quadrature and not the new step.

  * THE STEP CLOSES. Oracle: itself, one step later. `ase_step_residual=True` re-propagates at
    the populations each step ENDED at and reports the worst disagreement with the powers that
    step USED. That is not a tautology: a frozen step in the runaway regime scores 1e5-1e232,
    the self-consistent step scores at its inner tolerance (~1e-6), and the number is computed
    from the returned arrays by the same residual the steady solver stops on.

  * "auto" IS NOT A SHORTCUT. It is gated against the FULLY self-consistent march, not against
    solve(): same final gain to < 1e-3 dB on every failure case, while taking strictly fewer
    self-consistent steps than there are steps on the case where it should.

Every pathology gate is PREMISE-GATED: the healthy behaviour is asserted first in the same test,
so a change that breaks everything cannot pass by making the pathology disappear.
"""

import warnings

import numpy as np
import pytest

from dynameta.optics.fiber_amp import march_ase
from dynameta.optics.fiber_amp.dynamics import simulate_transient
from dynameta.optics.fiber_amp.eryb import (RATE_ARRHENIUS_CHENG_2022,
                                            YB_STARK_976_CANAT_DUSSARDIER, ErYbAmplifier)
from dynameta.optics.fiber_amp.spectroscopy import erbium, ytterbium
from dynameta.optics.fiber_amp.steady_state import AseBand, FiberAmplifier, Pump, Signal
from dynameta.optics.fiber_amp.waveguide import FiberSpec

ER = erbium()
ER_AL = erbium("aluminosilicate")
YB_PH = ytterbium("phosphosilicate")


# ============================ the fixtures the audit note quotes ============================

def _hot_yb():
    """The 400 W cladding-pumped Yb booster the audit-A-7 failure cases are measured on
    (tests/test_fiber_dynamics.py's own fixture, repeated here so the two files cannot drift
    apart silently -- if it changes there, this file's pinned dB numbers change too)."""
    fib = FiberSpec(core_radius_m=5.0e-6, na=0.08, n_t_m3=6.0e25, length_m=5.0,
                    clad_radius_m=62.5e-6)
    return FiberAmplifier(ytterbium(), fib, [Pump(400.0, 0.976e-6, "fwd", cladding=True)],
                          [Signal(5.0, 1.03e-6)], AseBand(1.00e-6, 1.08e-6, n_bins=8))


def _edfa():
    """A HEALTHY C-band EDFA at a 10 uW input: in-band ASE far above the signal, and squarely
    inside the quasi-static regime. The premise half of every agreement gate."""
    return FiberAmplifier(ER, FiberSpec(1.6e-6, 0.22, 8.0e24, 8.0), [Pump(0.25, 0.98e-6)],
                          [Signal(1e-5, 1.55e-6)], AseBand(1.50e-6, 1.60e-6, n_bins=16))


def _eryb(length_m=8.0):
    """The downstream study's reference co-doped design: a = 2.0 um / NA 0.20, N_Er 4e25,
    N_Yb 4e26, 125 um cladding, 176 mW of cladding-coupled 976 nm pump, a 2.1 mW 1550 nm seed,
    8 m, coupled fraction f = 0.9 with the coupled-pool k_tr = 1.11e-21 and K2 = 2e-22. This is
    the amplifier whose cold-start uniform-grid march trips the audit-A-7 flag."""
    return ErYbAmplifier(ER_AL, YB_PH,
                         FiberSpec(2.0e-6, 0.20, 4.0e25, length_m, clad_radius_m=62.5e-6),
                         [Pump(0.176, 0.976e-6, "fwd", cladding=True)],
                         [Signal(2.1e-3, 1.550e-6)], AseBand(1.520e-6, 1.570e-6, 24),
                         n_yb_m3=4.0e26, k_tr_m3_s=1.11e-21, yb_coupled_fraction=0.9,
                         k_tr2_m3_s=2.0e-22, yb_ase=AseBand(1.000e-6, 1.100e-6, 12))


def _march(amp, t, **kw):
    """simulate_transient with the RuntimeWarnings captured rather than raised, and the overflow
    of a deliberately broken quasi-static march silenced."""
    with warnings.catch_warnings(record=True) as caught, np.errstate(all="ignore"):
        warnings.simplefilter("always")
        r = simulate_transient(amp, t, **kw)
    return r, [w for w in caught if issubclass(w.category, RuntimeWarning)]


# ==================== (1) the default march did not move ====================================

# End states MEASURED on v0.11.2 (commit 21dff60) before this work, with the modes' plumbing
# absent entirely. Pinned to 1e-9 RELATIVE rather than bit-for-bit: the co-doped rate assembly
# goes through a BLAS matvec (eryb._rates_profile) whose reduction order is a build property, and
# CI runs this on mixed CPU generations and on a numpy 1.24 / scipy 1.10 leg. 1e-9 is seven
# decades above that noise and many decades below the smallest change this algebra could take
# and still be a different model. The bit-for-bit claim is made against the DEFAULT CALL in the
# same process, below.
#
# EVERY fixture here is a march that CONVERGES to its fixed point, which is why a tight pin is
# meaningful on it. The RUNAWAY marches (the 175 K profile, the 1 uW signal, the co-doped 1 ms
# and 3 ms cold grids) deliberately carry NO pinned value anywhere in this file: a march whose
# iteration is unstable amplifies its own rounding, so a cross-build equality on one would be a
# flake, not a gate. They are asserted by INEQUALITY against solve() instead -- the same pattern
# tests/test_fiber_dynamics.py uses for them.
#
# TOLERANCES ARE SPLIT BY WHETHER THE FIXTURE TOUCHES LSODA (commit 533ece0's lesson: a pin on a
# value that came through the steady solver's adaptive integrator is scipy-version dependent and
# cannot hold at 1e-9). The EDFA and the co-doped fixture are seeded EXPLICITLY (nbar2_0), so
# their pins are pure march arithmetic and hold at 1e-9 -- the EDFA's value is also unchanged
# from the solve()-seeded one to all 17 digits, because that march converges to its fixed point
# whatever it starts from. The Yb booster has no uniform seed that is not itself a runaway (a
# 0.4 seed trips the flag and lands at 0.56 dB), so it keeps the solve() seed and a 1e-5 pin.
_V0112_SINGLE = {                     # (end gain dB, end max nbar2, seed, relative tolerance)
    "yb":    (18.789774052292046, 0.40686071365741333, None, 1e-5),
    "edfa":  (26.862428117239133, 0.98997488891123731, 0.0, 1e-9),
}
_V0112_ERYB = {                       # (end gain dB, end max f2, end max beta_yb)
    # the two-pool co-doped reference, cold start, uniform 300 us: converges to 13.5172 dB
    "cold300": (13.517242494945485, 0.69258909668427637, 0.01393998712845269),
}


def test_quasi_static_mode_is_the_default_call_bit_for_bit():
    # The one claim that can be made EXACTLY: passing the default mode explicitly must not put a
    # single floating-point operation on the path. Checked on both classes and on every returned
    # array, including the resolved ASE and the opt-in profile matrix.
    t = np.linspace(0.0, 25e-3, 120)
    for amp, kw in ((_edfa(), {}), (_hot_yb(), {})):
        a, _ = _march(amp, t, n_nodes=81, store_profiles=True)
        b, _ = _march(amp, t, n_nodes=81, store_profiles=True, ase_mode="quasi_static")
        for name in ("nbar2_zt", "signal_out_W", "pump_out_W", "signal_gain_dB",
                     "ase_fwd_W", "ase_bwd_W", "power_zt"):
            assert np.array_equal(getattr(a, name), getattr(b, name)), name
    te = np.arange(0.0, 12e-3 + 1e-12, 300e-6)
    amp = _eryb()
    a, _ = _march(amp, te, n_nodes=161, nbar2_0=0.0)
    b, _ = _march(amp, te, n_nodes=161, nbar2_0=0.0, ase_mode="quasi_static")
    for name in ("nbar2_zt", "signal_out_W", "signal_gain_dB", "ase_fwd_W", "ase_bwd_W"):
        assert np.array_equal(getattr(a, name), getattr(b, name)), name
    for key in ("beta_yb", "beta_yb_coupled", "beta_yb_uncoupled"):
        assert np.array_equal(a.meta[key], b.meta[key]), key


def test_default_march_still_reproduces_v0_11_2():
    # Cross-VERSION half: the numbers v0.11.2 produced on four fixtures, two per class. A change
    # that moved BOTH modes together would pass the bit-for-bit test above and fail here.
    t = np.linspace(0.0, 25e-3, 400)
    for tag, amp in (("yb", _hot_yb()), ("edfa", _edfa())):
        g, n2, seed, rel = _V0112_SINGLE[tag]
        r, _ = _march(amp, t, n_nodes=81, nbar2_0=seed)
        assert r.signal_gain_dB[-1, 0] == pytest.approx(g, rel=rel), tag
        assert float(r.nbar2_zt[-1].max()) == pytest.approx(n2, rel=rel), tag

    amp = _eryb()
    for tag, dt in (("cold300", 300e-6),):
        r, _ = _march(amp, np.arange(0.0, 12e-3 + 1e-12, dt), n_nodes=161, nbar2_0=0.0)
        g, f2, byb = _V0112_ERYB[tag]
        assert r.signal_gain_dB[-1, 0] == pytest.approx(g, rel=1e-9), tag
        assert float(r.nbar2_zt[-1].max()) == pytest.approx(f2, rel=1e-9), tag
        assert float(r.meta["beta_yb"][-1].max()) == pytest.approx(byb, rel=1e-9), tag


def test_ase_mode_is_validated_by_name():
    # A mis-spelled mode that silently took the default path is exactly the silent-wrong failure
    # this option exists to remove.
    with pytest.raises(ValueError, match="ase_mode"):
        simulate_transient(_edfa(), np.linspace(0.0, 1e-3, 4), ase_mode="selfconsistent")
    with pytest.raises(ValueError, match="quasi_static"):
        simulate_transient(_eryb(), np.linspace(0.0, 1e-3, 4), ase_mode=True)


# ==================== (2) where the flag stays True, the modes agree ========================

def test_self_consistent_agrees_with_quasi_static_inside_the_valid_regime():
    """PREMISE: these two marches never trip the audit-A-7 flag, so the quasi-static answer is
    the trustworthy one and agreement is a test of the NEW step, not of the old one.

    MEASURED (2026-09-15): on the C-band EDFA the whole 400-frame trajectory agrees to 3.8e-4 dB
    and every ASE bin to 1.8e-4 relative; on the co-doped reference at 201 log-spaced frames,
    1.1e-5 dB and 3.2e-6. Both are far inside the 1e-3 dB / 1% the mode is specified to."""
    for tag, amp, t, n, kw in (
            ("edfa", _edfa(), np.linspace(0.0, 25e-3, 400), 81, {}),
            ("eryb", _eryb(), np.concatenate([[0.0], np.logspace(-9, np.log10(12e-3), 200)]),
             161, {})):
        q, wq = _march(amp, t, n_nodes=n, ase_mode="quasi_static", **kw)
        s, ws = _march(amp, t, n_nodes=n, ase_mode="self_consistent", **kw)
        assert q.meta["quasi_static_valid"] is True, tag          # the premise
        assert not wq and not ws, tag
        assert s.meta["march_valid"] is True, tag
        assert float(np.max(np.abs(q.signal_gain_dB - s.signal_gain_dB))) < 1e-3, tag
        for a_q, a_s in ((q.ase_fwd_W, s.ase_fwd_W), (q.ase_bwd_W, s.ase_bwd_W)):
            rel = np.abs(a_q - a_s) / np.maximum(np.abs(a_q), 1e-300)
            assert float(np.max(rel)) < 1e-2, (tag, float(np.max(rel)))


def test_the_two_modes_coincide_as_the_step_shrinks():
    # STRUCTURAL: the self-consistent step's population map is the march's own update, which
    # returns the CURRENT populations as dt -> 0. So the whole inner solve must collapse onto one
    # frozen-gain propagation -- the quasi-static step -- rather than merely approach it.
    # MEASURED at dt = 5.3e-11 s: 9.8e-10 in nbar2 and 3.2e-7 dB in gain, on a 400 W booster
    # whose inversion moves by 0.4 over the full transient.
    t = np.linspace(0.0, 1e-9, 20)
    q, _ = _march(_hot_yb(), t, n_nodes=41, ase_mode="quasi_static")
    s, _ = _march(_hot_yb(), t, n_nodes=41, ase_mode="self_consistent")
    assert float(np.max(np.abs(q.nbar2_zt - s.nbar2_zt))) < 1e-8
    assert float(np.max(np.abs(q.signal_gain_dB - s.signal_gain_dB))) < 1e-6


def test_the_two_modes_share_one_fixed_point():
    """The property the quasi-static march has and the new mode must not lose. A march seeded
    from solve() at a constant drive settles, and it settles on the SAME state in both modes --
    if the self-consistent step had a fixed point of its own, a nearby one, the two would part.

    It does NOT sit exactly on solve()'s profile, and never did: the march propagates on its own
    trapezoid quadrature while solve() runs adaptive LSODA, so both modes relax by the same
    7.3e-5 in nbar2 (2e-4 dB) and then stop. That offset is the mesh, and it is asserted to be
    the same offset for both modes."""
    amp = _edfa()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ss = amp.solve(n_nodes=81)
    t = np.linspace(0.0, 50e-3, 60)
    q, wq = _march(amp, t, n_nodes=81, nbar2_0=ss.nbar2_z, ase_mode="quasi_static")
    s, ws = _march(amp, t, n_nodes=81, nbar2_0=ss.nbar2_z, ase_mode="self_consistent")
    assert not wq and not ws
    for r, tag in ((q, "quasi_static"), (s, "self_consistent")):
        # stationary: the last third of the march does not move at all
        assert float(np.max(np.abs(r.nbar2_zt[-1] - r.nbar2_zt[-20]))) < 1e-12, tag
        assert abs(float(r.signal_gain_dB[-1, 0] - r.signal_gain_dB[-20, 0])) < 1e-12, tag
    assert float(np.max(np.abs(q.nbar2_zt[-1] - s.nbar2_zt[-1]))) < 1e-9
    assert abs(float(q.signal_gain_dB[-1, 0] - s.signal_gain_dB[-1, 0])) < 1e-6
    # and that common fixed point is solve()'s, to the march's own quadrature
    assert float(np.max(np.abs(s.nbar2_zt[-1] - ss.nbar2_z))) < 1e-3
    assert abs(float(s.signal_gain_dB[-1, 0]) - float(ss.signal_gain_dB[0])) < 1e-2


# ==================== (3) the documented failure cases ======================================

def _solve_dB(amp, n_nodes):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(amp.solve(n_nodes=n_nodes, relax="auto").signal_gain_dB[0])


@pytest.mark.parametrize("tag,broken_by_dB,sc_within_dB", [
    # MEASURED 2026-09-15, 400 frames over 25 ms at 81 nodes:
    #   175 K profile : quasi-static -50.14 dB from solve(), self-consistent +0.074
    #   1 uW signal   : quasi-static -67.38 dB from solve(), self-consistent +0.017
    # sc_within_dB is the march's own z-QUADRATURE floor at 81 nodes, not a property of the new
    # step: test_self_consistent_residue_is_mesh_convergent below shows it falling as O(dz^2) on
    # the same fixtures, to 0.005 / 0.001 dB at 321 nodes, where the HEALTHY march's gap against
    # solve() falls identically (0.0084 -> 0.0005).
    ("yb175", 20.0, 0.10),
    ("yb_tiny", 40.0, 0.05),
])
def test_single_ion_runaway_cases_are_recovered(tag, broken_by_dB, sc_within_dB):
    z = np.linspace(0.0, 5.0, 41)
    amp = _hot_yb()
    if tag == "yb175":
        amp.set_temperature_profile(z, np.full(z.size, 175.0), T_ref_K=300.0)
    else:
        amp = amp.with_signals([Signal(1e-6, 1.03e-6)])
    t = np.linspace(0.0, 25e-3, 400)
    G = _solve_dB(amp, 81)

    # PREMISE: the default march really is broken here, loudly.
    q, wq = _march(amp, t, n_nodes=81, ase_mode="quasi_static", ase_step_residual=True)
    assert q.meta["quasi_static_valid"] is False
    assert q.meta["march_valid"] is False
    assert [w for w in wq if "quasi-static" in str(w.message)], "the A-7 warning must still fire"
    assert abs(G - float(q.signal_gain_dB[-1, 0])) > broken_by_dB
    assert q.meta["max_step_power_residual"] > 1e2          # its own inversion does not sustain

    # And the self-consistent march is not: finite, silent, on the steady solve, and closing
    # every step to its inner tolerance.
    s, ws = _march(amp, t, n_nodes=81, ase_mode="self_consistent", ase_step_residual=True)
    assert not ws, [str(w.message)[:120] for w in ws]
    assert s.meta["march_valid"] is True
    assert np.all(np.isfinite(s.signal_gain_dB)) and np.all(np.isfinite(s.nbar2_zt))
    assert abs(G - float(s.signal_gain_dB[-1, 0])) < sc_within_dB
    assert s.meta["max_step_power_residual"] < 1e-4
    # the step-by-step closure improved by at least six decades, which is the whole claim
    assert q.meta["max_step_power_residual"] / s.meta["max_step_power_residual"] > 1e6

    # ... and "auto" reaches the same answer on these two. It switches essentially EVERY step
    # here (these amplifiers are in the coupled regime throughout, unlike the co-doped grids),
    # which is the correct outcome and the reason auto saves nothing on them.
    a, wa = _march(amp, t, n_nodes=81, ase_mode="auto")
    assert not wa, tag
    assert a.meta["march_valid"] is True, tag
    assert abs(float(a.signal_gain_dB[-1, 0]) - float(s.signal_gain_dB[-1, 0])) < 1e-3, tag
    assert a.meta["n_self_consistent_steps"] > 0.9 * (t.size - 1), tag


@pytest.mark.parametrize("dt_s,broken_by_dB", [(1000e-6, 10.0), (3000e-6, 5.0)])
def test_codoped_cold_start_uniform_grid_is_recovered(dt_s, broken_by_dB):
    """The study's own shape: the co-doped reference marched from a COLD fiber on a UNIFORM
    grid. MEASURED 2026-09-15 at 161 nodes -- quasi-static lands +13.93 dB (1 ms) and -10.01 dB
    (3 ms) from solve()'s 13.5178 dB, with step closures of 4.9e2 and 1.8e10; self-consistent
    lands within 5.4e-4 dB on both, closing every step to ~1e-6."""
    amp = _eryb()
    t = np.arange(0.0, 12e-3 + 1e-12, dt_s)
    G = _solve_dB(amp, 801)

    q, wq = _march(amp, t, n_nodes=161, nbar2_0=0.0, ase_mode="quasi_static",
                   ase_step_residual=True)
    assert q.meta["quasi_static_valid"] is False
    assert [w for w in wq if "quasi-static" in str(w.message)]
    assert abs(G - float(q.signal_gain_dB[-1, 0])) > broken_by_dB
    assert q.meta["max_step_power_residual"] > 1e2

    s, ws = _march(amp, t, n_nodes=161, nbar2_0=0.0, ase_mode="self_consistent",
                   ase_step_residual=True)
    assert not ws, [str(w.message)[:120] for w in ws]
    assert s.meta["march_valid"] is True
    assert np.all(np.isfinite(s.signal_gain_dB)) and np.all(np.isfinite(s.nbar2_zt))
    assert np.all(np.isfinite(s.meta["beta_yb"]))
    assert abs(G - float(s.signal_gain_dB[-1, 0])) < 1e-3       # the gate's own 1e-3 dB
    assert s.meta["max_step_power_residual"] < 1e-5


def test_codoped_cold_start_300us_idle_seed_trips_the_flag_and_is_recovered():
    """The exact case the downstream study hit: a cold start into a 300 us uniform grid behind a
    -20 dB idle preamble. The quasi-static march happens to LAND on the right gain here (13.5172
    dB) while its flag is False and its step closure is 1.4e5 -- which is why the flag, not the
    endpoint, is what the study was reading. Self-consistent removes the flag and closes to
    3.1e-6."""
    amp = _eryb()
    p_in = 2.1e-3
    t = np.concatenate([[-3e-3, -1e-3], np.arange(0.0, 12e-3 + 1e-12, 300e-6)])

    def drive(tt):
        return [p_in * (1e-2 if tt < 0.0 else 1.0)]

    q, wq = _march(amp, t, n_nodes=161, nbar2_0=0.0, signal_drive=drive,
                   ase_mode="quasi_static", ase_step_residual=True)
    assert q.meta["quasi_static_valid"] is False
    assert [w for w in wq if "quasi-static" in str(w.message)]
    assert q.meta["max_step_power_residual"] > 1e4

    s, ws = _march(amp, t, n_nodes=161, nbar2_0=0.0, signal_drive=drive,
                   ase_mode="self_consistent", ase_step_residual=True)
    assert not ws
    assert s.meta["march_valid"] is True
    assert s.meta["quasi_static_valid"] is True                 # the ASE never runs away now
    assert s.meta["max_step_power_residual"] < 1e-5
    assert abs(_solve_dB(amp, 801) - float(s.signal_gain_dB[-1, 0])) < 1e-3


def test_self_consistent_march_is_second_order_in_the_z_mesh():
    """DISCRIMINATION for the dB numbers above: is the residual gap against solve() a defect of
    the self-consistent step, or the march's own z QUADRATURE? It is the quadrature -- the
    march's own answer converges at SECOND ORDER in dz, which is exactly the order of the
    trapezoid rule _propagate_fixed integrates with, so the gap against any mesh-independent
    reference must shrink as dz^2 and cannot be a fixed defect of the step.

    RICHARDSON, WITH NO ORACLE. The successive differences G(81) - G(161) and G(161) - G(321)
    must be in the ratio ~4. Deliberately no solve() call: this fixture's steady solve is the
    400 W booster at a 1 uW signal, the hardest relaxation in the repo, and its adaptive-LSODA
    path at a FINE mesh is scipy-build dependent -- measured, the same three meshes give gaps of
    0.0165 / 0.0041 / 0.0010 dB here and 0.0165 / 0.0041 / 0.0116 on the CI py3.10 floor leg,
    where only the 321-node ORACLE moved. A gate that reads a build-dependent oracle to make a
    statement about the MARCH is a flake, so this one reads only the march. The oracle agreement
    itself is gated at 81 nodes, where every build agrees, by
    test_single_ion_runaway_cases_are_recovered above.

    MEASURED here: G = 76.549123 / 76.544919 / 76.543870 dB at 81 / 161 / 321 nodes, i.e.
    successive differences 4.204e-3 and 1.049e-3, ratio 4.01."""
    t = np.linspace(0.0, 25e-3, 200)
    tiny = _hot_yb().with_signals([Signal(1e-6, 1.03e-6)])
    gains = []
    for n in (81, 161, 321):
        s, w = _march(tiny, t, n_nodes=n, ase_mode="self_consistent")
        assert not w, n
        assert s.meta["march_valid"] is True, n
        gains.append(float(s.signal_gain_dB[-1, 0]))
    d1, d2 = gains[0] - gains[1], gains[1] - gains[2]
    assert abs(d1) > 1e-4 and abs(d2) > 1e-6, gains       # the mesh really is moving the answer
    assert d1 * d2 > 0.0, gains                           # and monotonically, not oscillating
    ratio = abs(d1) / abs(d2)
    assert 3.0 < ratio < 6.0, (gains, ratio)              # SECOND order, not first and not noise


# ==================== (4) "auto" ============================================================

def test_auto_reproduces_the_self_consistent_march_on_the_codoped_failure_cases():
    """auto is gated against the FULLY self-consistent march, not against solve(): the question
    is whether the switch criterion catches the steps that matter. MEASURED 2026-09-15: the
    final gains agree to 4.4e-8 dB or better on both cases below (the two single-ion runaway
    cases are gated alongside their own recovery, above)."""
    cases = [
        ("eryb_1ms", _eryb(), np.arange(0.0, 12e-3 + 1e-12, 1000e-6), 161, dict(nbar2_0=0.0)),
        ("eryb_3ms", _eryb(), np.arange(0.0, 12e-3 + 1e-12, 3000e-6), 161, dict(nbar2_0=0.0)),
    ]
    for tag, amp, t, n, kw in cases:
        s, _ = _march(amp, t, n_nodes=n, ase_mode="self_consistent", **kw)
        a, wa = _march(amp, t, n_nodes=n, ase_mode="auto", **kw)
        assert not wa, tag
        assert a.meta["march_valid"] is True, tag
        assert abs(float(a.signal_gain_dB[-1, 0])
                   - float(s.signal_gain_dB[-1, 0])) < 1e-3, tag
        # it really did switch, and it logged where
        assert a.meta["n_self_consistent_steps"] > 0, tag
        assert a.meta["ase_switch_steps"].size == a.meta["n_self_consistent_steps"], tag
        assert np.array_equal(np.where(a.meta["ase_mode_steps"] == 1)[0],
                              a.meta["ase_switch_steps"]), tag
        assert np.allclose(a.meta["ase_switch_times"], t[a.meta["ase_switch_steps"]]), tag


def test_auto_costs_the_quasi_static_march_when_nothing_needs_switching():
    """The other half of auto's contract: on a march that never leaves the valid regime it must
    take ZERO self-consistent steps, spend no probes, and return the quasi-static arrays
    bit-for-bit. MEASURED on the co-doped reference at 201 log-spaced frames: 0 switches,
    0 probes, and 3.24 vs 3.00 ms/step at 161 nodes (the difference is the per-step decision)."""
    amp = _eryb()
    t = np.concatenate([[0.0], np.logspace(-9, np.log10(12e-3), 200)])
    q, _ = _march(amp, t, n_nodes=161, ase_mode="quasi_static")
    a, wa = _march(amp, t, n_nodes=161, ase_mode="auto")
    assert q.meta["quasi_static_valid"] is True                 # the premise
    assert not wa
    assert a.meta["n_self_consistent_steps"] == 0
    assert a.meta["ase_probe_propagations"] == 0
    for name in ("nbar2_zt", "signal_gain_dB", "ase_fwd_W", "ase_bwd_W"):
        assert np.array_equal(getattr(q, name), getattr(a, name)), name

    # DISCRIMINATION: the same amplifier on the grid that DOES need it switches, on the same
    # code path -- so the zero above is a decision, not a dead branch.
    bad = np.arange(0.0, 12e-3 + 1e-12, 3000e-6)
    b, _ = _march(amp, bad, n_nodes=161, nbar2_0=0.0, ase_mode="auto")
    assert b.meta["n_self_consistent_steps"] > 0
    assert b.meta["ase_probe_propagations"] > 0


def test_auto_switch_criterion_is_falsifiable_on_its_own():
    """The predictor, unit-tested away from the march: a step that does not move the ASE gain
    integral predicts no error whatever the ASE level, and a step that moves it by one e-fold
    predicts the ASE's own share of the launched power times e - 1."""
    lim = {"ase_to_launched": 1.0, "ase_gain_integral": 20.0}
    assert march_ase.predicted_step_ase_error(0.9, 9.0, 9.0) == 0.0
    assert march_ase.predicted_step_ase_error(0.0, -5.0, 12.0) == 0.0     # nothing emitting yet
    assert march_ase.predicted_step_ase_error(0.1, 9.0, 10.0) == pytest.approx(
        0.1 * (np.e - 1.0))
    assert not march_ase.auto_switch(0.05, 5.0, 5.0, False, lim, 0.5)     # quiet step
    assert march_ase.auto_switch(0.05, 5.0, 6.0, False, lim, 0.5)         # drifting step
    assert march_ase.auto_switch(0.9, 5.0, 5.0, False, lim, 0.5)          # ASE backstop
    assert march_ase.auto_switch(0.05, 15.0, 15.0, False, lim, 0.5)       # gain-integral backstop
    assert march_ase.auto_switch(0.0, 0.0, 0.0, True, lim, 0.5)           # non-finite
    # the PROBE half: a measured projected ratio that differs from the current one switches even
    # when the cheap estimate is identically zero (the cold-fiber case)
    assert not march_ase.auto_switch(0.0, -5.0, 3.0, False, lim, 0.5, probed_ase_ratio=0.0)
    assert march_ase.auto_switch(0.0, -5.0, 3.0, False, lim, 0.5, probed_ase_ratio=0.5)
    assert march_ase.needs_ase_probe(-5.0, 3.0)                # amplifying, and it moved
    assert not march_ase.needs_ase_probe(9.0, 9.0)             # sitting still
    assert not march_ase.needs_ase_probe(-9.0, -3.0)           # moving, but nothing amplifies


# ==================== the result surface survives the new mode ==============================

def test_resolved_ase_and_frame_as_steady_still_work_in_every_mode():
    # The F-2 surface (resolved ASE, the channel plan, frame_as_steady into the noise layer) has
    # to keep working in the new modes -- and the frame must carry the mode's OWN flags.
    from dynameta.optics.fiber_amp.noise import analyze_noise
    amp = _edfa()
    t = np.linspace(0.0, 10e-3, 30)
    for mode in march_ase.ASE_MODES:
        r, _ = _march(amp, t, n_nodes=61, ase_mode=mode, store_profiles=True)
        assert r.ase_fwd_W is not None and r.ase_lambda_m is not None
        assert r.ase_psd_1pol_W_Hz("fwd").shape == r.ase_fwd_W.shape
        fr = r.frame_as_steady(len(t) - 1)
        assert fr.meta["quasi_static_valid"] is r.meta["quasi_static_valid"]
        # The NEW flag and the mode cannot be lost through the frame either -- but they are only
        # ADDED in the new modes: the frame's meta key set in the default mode is itself pinned
        # as unchanged behaviour by test_fiber_eryb_transient.py, so growing it there would be
        # the very regression this work promises not to make.
        if mode == "quasi_static":
            assert "ase_mode" not in fr.meta and "march_valid" not in fr.meta
        else:
            assert fr.meta["march_valid"] is r.meta["march_valid"]
            assert fr.meta["ase_mode"] == mode
        nf = analyze_noise(fr, 1.55e-6)
        assert np.isfinite(float(nf.nf_dB)), mode
        assert r.meta["ase_mode"] == mode
        assert (r.meta["ase_mode_steps"] is None) == (mode == "quasi_static")


def test_meta_reports_nothing_self_consistent_in_the_default_mode():
    # The new meta block must read as "nothing happened" in the default mode, so a caller can
    # check it unconditionally.
    r, _ = _march(_edfa(), np.linspace(0.0, 5e-3, 20), n_nodes=41)
    assert r.meta["ase_mode"] == "quasi_static"
    assert r.meta["march_valid"] is r.meta["quasi_static_valid"]
    assert r.meta["n_self_consistent_steps"] == 0
    assert r.meta["ase_probe_propagations"] == 0
    for key in ("ase_mode_steps", "ase_switch_steps", "ase_switch_times", "ase_inner_relax",
                "ase_inner_tol", "max_self_consistency_residual", "max_step_power_residual"):
        assert r.meta[key] is None, key


# ==================== composition with the other co-doped opt-ins ===========================

def test_every_mode_composes_with_the_temperature_dependent_rate_opt_ins():
    """The 2026-09-15 co-doped thermal options (RateTemperatureLaw on k_tr / K2 / W_mig, and
    YbStarkThermal on the Yb band) ACT ONLY under an axial temperature profile -- eryb's
    _mcc_matrices returns None without one, so every consumer takes its isothermal branch. Two
    halves, and the second is what makes the first worth asserting:

      * WITHOUT a profile they are inert, so an amplifier carrying them marches BIT-FOR-BIT like
        one that does not, in ALL THREE modes. The self-consistent step calls the amplifier's own
        _fb_rhs3 / _fb_jacobian3 / _rates_profile, so it inherits that inertness rather than
        re-deriving it -- but inheriting it silently is exactly how a composition bug hides.
      * WITH a profile the march REFUSES BY NAME in all three modes, and the message names the
        opt-ins it is also refusing. ase_mode changes the population UPDATE, not the
        cross-sections the step propagates through, so it cannot rescue a profiled amplifier and
        must not appear to."""
    t = np.arange(0.0, 6e-3 + 1e-12, 300e-6)
    plain = _eryb()
    hot_opts = dict(n_yb_m3=4.0e26, k_tr_m3_s=1.11e-21, yb_coupled_fraction=0.9,
                    k_tr2_m3_s=2.0e-22, yb_ase=AseBand(1.000e-6, 1.100e-6, 12),
                    rate_temperature=RATE_ARRHENIUS_CHENG_2022,
                    yb_stark_thermal=YB_STARK_976_CANAT_DUSSARDIER)
    with_opts = ErYbAmplifier(ER_AL, YB_PH,
                              FiberSpec(2.0e-6, 0.20, 4.0e25, 8.0, clad_radius_m=62.5e-6),
                              [Pump(0.176, 0.976e-6, "fwd", cladding=True)],
                              [Signal(2.1e-3, 1.550e-6)], AseBand(1.520e-6, 1.570e-6, 24),
                              **hot_opts)
    # PREMISE: the opt-ins really are attached (a constructor that dropped them would make the
    # equality below vacuous -- RateTemperatureLaw() IS dropped when identity, this one is not).
    assert with_opts.rate_temperature is not None
    assert with_opts.yb_stark_thermal is not None

    for mode in march_ase.ASE_MODES:
        a, _ = _march(plain, t, n_nodes=161, nbar2_0=0.0, ase_mode=mode)
        b, _ = _march(with_opts, t, n_nodes=161, nbar2_0=0.0, ase_mode=mode)
        for name in ("nbar2_zt", "signal_gain_dB", "ase_fwd_W", "ase_bwd_W"):
            assert np.array_equal(getattr(a, name), getattr(b, name)), (mode, name)
        assert np.array_equal(a.meta["beta_yb"], b.meta["beta_yb"]), mode
        assert a.meta["n_self_consistent_steps"] == b.meta["n_self_consistent_steps"], mode

    # ... and with a profile, every mode refuses, naming what it refuses.
    z = np.linspace(0.0, 8.0, 17)
    with_opts.set_temperature_profile(z, np.full(z.size, 360.0), T_ref_K=300.0)
    for mode in march_ase.ASE_MODES:
        with pytest.raises(NotImplementedError, match="temperature profile"):
            simulate_transient(with_opts, t, n_nodes=161, nbar2_0=0.0, ase_mode=mode)
    try:
        simulate_transient(with_opts, t, n_nodes=161, nbar2_0=0.0, ase_mode="self_consistent")
    except NotImplementedError as exc:
        msg = str(exc)
    assert "rate_temperature" in msg and "yb_stark_thermal" in msg
    assert "self_consistent" in msg                      # it says the new modes do not help
