"""Gates for the 2026-08-04 fiber_amp consumer audit
(docs/audit/2026-08-04-fiber-amp-comms-and-efficiency-gap-audit.md).

One test class per finding. Every gate is DISCRIMINATING: it either compares against an
independent computation (an analytic limit, a different code path in the package, a closed form) or
it asserts the specific defect the finding recorded, so that reverting the fix fails the gate.

The opt-in findings additionally gate BYTE-IDENTITY of the default path, because that is the
property that let these fixes land without re-baselining the repo's pinned numbers.
"""

import math

import numpy as np
import pytest

from dynameta.constants import C_LIGHT, H_PLANCK, KB
from dynameta.optics.fiber_amp import (
    AseBand, ChannelPlan, FEC_THRESHOLDS, FiberAmplifier, FiberSpec, PamFormat, Pump, PumpSource,
    Signal, V_MARCUSE_MAX, V_MARCUSE_MIN, analyze_noise, detection_noise, effective_pump_lambda_m,
    energy_per_bit_J, erbium, level_statistics, link_performance, marcuse_validity, max_pam_order,
    ml_thresholds, output_ase_spectrum, pam_levels_W, pam_ser, q_to_ser, ser_to_ber,
    simulate_transient, stokes_limit, v_number, wall_plug_efficiency, ytterbium,
)
from dynameta.optics.fiber_amp.spectroscopy import at_temperature

ER = erbium()
YB = ytterbium()


def _edf(length_m=6.0, n_t=1.0e25):
    return FiberSpec(core_radius_m=1.4e-6, na=0.24, n_t_m3=n_t, length_m=length_m)


def _edfa(length_m=6.0, pump_W=0.1, sig_W=1e-6, bins=10):
    return FiberAmplifier(ER, _edf(length_m), [Pump(pump_W, 0.980e-6)],
                          [Signal(sig_W, 1.560e-6)], AseBand(1.52e-6, 1.575e-6, bins))


def _ydfa(sig_W=20e-3, pump_W=2.0, pump_lam=0.976e-6, length_m=4.0):
    """The fixture that exposed BOTH F-13 and F-14: its 976 nm pump is FULLY absorbed."""
    fib = FiberSpec(3.0e-6, 0.12, 6.0e25, length_m)
    return FiberAmplifier(YB, fib, [Pump(pump_W, pump_lam)], [Signal(sig_W, 1.060e-6)],
                          AseBand(1.02e-6, 1.10e-6, 24))


# ============================ F-13: the converged flag ======================================

class TestF13ConvergedFlag:
    def test_flag_is_true_for_a_fully_absorbed_pump(self):
        """The defect: a fully absorbed pump's endpoint is integrator noise (measured -1.3e-16 W
        after 3003 dB), and the old ABSOLUTE 1e-15 denominator floor made the endpoint residual
        ~0.1 forever, so `converged` could never trip on an efficient amplifier.

        THE FIXTURE IS THE GATE. This used to run at the module's default 20 mW input, where the
        PRE-AUDIT stopping test ALSO converges -- in 4 iterations, measured by re-running the old
        endpoint denominator on this exact fiber -- so the gate could not fail if the fix were
        reverted. 0.243 mW is the audit's own discriminating operating point (its F-13 trace):
        the old test never converges (400 iterations, `converged` False, 34.722740 dB), the new
        one stops at 118 with 34.722738 dB. Both pinned below, since a fix that changed WHERE the
        solve lands would be a different bug."""
        r = _ydfa(sig_W=0.243e-3).solve(n_nodes=201, max_iter=400, relax=1.0)
        pmp = [i for i, k in enumerate(r.kind) if k == "pump"][0]
        # the guard is only exercised if the pump really is absorbed into the noise floor
        assert abs(r.power_W[pmp, -1]) < 1e-12 * r.power_W[pmp, 0]
        assert r.meta["converged"]
        assert r.meta["iterations"] == pytest.approx(118, abs=15)   # measured 118; never, before
        assert float(r.signal_gain_dB[0]) == pytest.approx(34.72274, abs=1e-3)

    def test_stopping_early_does_not_move_the_answer(self):
        """Discrimination: the fix must change WHEN the solve stops, not WHERE it lands. Compare
        against a run forced to iterate far past convergence.

        `tol=1e-11` is UNREACHABLE, deliberately and by construction: the sweeps run LSODA at
        rtol=1e-7, so successive iterates carry ~1e-7 of relative integrator noise and the residual
        bottoms out there (measured on this fiber: 6.405e-07 at tol = 1e-8, 1e-9 and 1e-11 alike,
        all `converged` False at max_iter). So this reference is max_iter-EXHAUSTED, not converged
        -- which is exactly what is wanted from it, and is why the comparison is against the gain
        rather than against the flag."""
        a = _ydfa().solve(n_nodes=201, relax=1.0)
        b = _ydfa().solve(n_nodes=201, tol=1e-11, max_iter=1200, relax=1.0)
        assert not b.meta["converged"] and b.meta["iterations"] == 1200
        assert abs(float(a.signal_gain_dB[0]) - float(b.signal_gain_dB[0])) < 1e-3

    def test_a_channel_carrying_information_is_still_tested(self):
        """The peak-relative floor must not be a blanket loosening: an amplifier whose pump is only
        partly absorbed keeps a substantial endpoint, so its denominator is |out| exactly as
        before and the test is unchanged for it."""
        r = _edfa(pump_W=0.1).solve(n_nodes=161)
        pmp = [i for i, k in enumerate(r.kind) if k == "pump"][0]
        frac = float(r.power_W[pmp, -1] / r.power_W[pmp, 0])
        assert frac > 1e-3                       # residual pump is a real number, not noise
        assert r.meta["converged"]

    def test_all_three_solvers_share_one_residual_implementation(self):
        """The defect existed in TRIPLICATE (steady_state, eryb, transverse held byte-identical
        copies). Consolidating it is what keeps the cross-solver 1e-9 reduction gates comparing
        equal iteration counts."""
        import inspect
        from dynameta.optics.fiber_amp import eryb, steady_state, transverse
        for mod in (steady_state, eryb, transverse):
            src = inspect.getsource(mod)
            assert "_relaxation_residuals(" in src, mod.__name__
            # the old hand-rolled denominator must be gone from all three
            assert "np.maximum(np.abs(out), 1e-15)" not in src, mod.__name__


# ============================ F-14: relaxation robustness ===================================

class TestF14Relaxation:
    def test_default_relax_is_byte_identical(self):
        a = _ydfa().solve(n_nodes=161, relax=1.0)
        b = _ydfa()._solve_once(n_nodes=161, relax=1.0)
        assert np.array_equal(a.power_W, b.power_W)
        assert np.array_equal(a.nbar2_z, b.nbar2_z)

    def test_under_relaxation_recovers_the_physical_branch(self):
        """The defect: at a low signal into a strongly pumped quasi-three-level fiber the plain
        Gauss-Seidel iteration falls onto a spurious, essentially UNPUMPED branch -- measured gain
        -12.6 dB with nbar2 = 0.036, unrecoverable by further iteration. Under-relaxation damps the
        oscillation and finds the real fixed point."""
        bad = _ydfa(sig_W=0.05e-3).solve(n_nodes=201, max_iter=400, relax=1.0)
        good = _ydfa(sig_W=0.05e-3).solve(n_nodes=201, max_iter=400, relax=0.5)
        assert float(bad.signal_gain_dB[0]) < 0.0 and not bad.meta["converged"]
        assert good.meta["converged"]
        assert float(good.signal_gain_dB[0]) > 30.0
        # and an amplifier absorbing 2 W of pump cannot be uninverted
        assert float(good.nbar2_z.mean()) > 0.25

    def test_gain_decreases_monotonically_with_input_power(self):
        """The physical invariant the spurious branch violated, and the cheap sweep gate that would
        have caught both F-13 and F-14 (neither is reachable from a single well-chosen fixture)."""
        p = [0.05e-3, 0.1e-3, 0.243e-3, 1e-3, 5e-3, 20e-3, 100e-3]
        g = [float(_ydfa(sig_W=x).solve(n_nodes=201, max_iter=400, relax=0.5).signal_gain_dB[0])
             for x in p]
        assert np.all(np.diff(g) < 0.0), g

    def test_residuals_are_reported_and_discriminate(self):
        bad = _ydfa(sig_W=0.05e-3).solve(n_nodes=201, max_iter=200, relax=1.0)
        good = _ydfa(sig_W=0.05e-3).solve(n_nodes=201, max_iter=400, relax=0.5)
        assert bad.meta["endpoint_residual"] > 1.0          # measured ~1e9
        assert good.meta["endpoint_residual"] < 1e-5
        assert "min_power_W" in good.meta and "relax" in good.meta

    def test_relax_is_validated(self):
        for bad in (0.0, -0.1, 1.5):
            with pytest.raises(ValueError, match="relax"):
                _edfa().solve(n_nodes=41, relax=bad)

    def test_the_resolved_solver_has_the_same_remedy(self):
        """F-14 shipped `relax` on steady_state ONLY, while transverse.ResolvedFiberAmplifier runs
        the SAME alternating frozen-direction iteration -- so the resolved solver kept the defect
        with no knob to close it. Measured on the counter-pumped Yb reference (3 um core, NA 0.12,
        6e25 m^-3, 6 m, 2 W bwd at 976 nm, 20 mW in at 1060 nm):

            resolved  relax = 1.0   -30.105 dB   converged False, endpoint residual 5.56e+06
            resolved  relax = 0.3   +17.952 dB   converged True   in 177 iterations
            mean field relax = 0.3  +17.891 dB   converged True   in 114 iterations

        i.e. a 48 dB gap that the resolved class could not escape. The under-relaxed answer is
        cross-checked against the mean-field twin, which is also a TSHB-regime statement: a
        CORE-GUIDED pump bleaches its own absorption non-uniformly, so the resolved gain sits
        slightly ABOVE the mean-field one (+0.0615 dB here) rather than below."""
        from dynameta.optics.fiber_amp import ResolvedFiberAmplifier, mean_field_equivalent
        fib = FiberSpec(3.0e-6, 0.12, 6.0e25, 6.0)
        res = ResolvedFiberAmplifier(YB, fib, [Pump(2.0, 0.976e-6, "bwd")],
                                     [Signal(20e-3, 1.060e-6)], AseBand(1.02e-6, 1.10e-6, 6),
                                     n_quad=8)
        bad = res.solve(n_nodes=81, max_iter=50, relax=1.0)      # diverged by iteration 50 already
        assert not bad.meta["converged"] and bad.meta["endpoint_residual"] > 1e3
        good = res.solve(n_nodes=81, max_iter=400, relax=0.3)
        assert good.meta["converged"] and good.meta["endpoint_residual"] < 1e-6
        assert float(good.signal_gain_dB[0]) - float(bad.signal_gain_dB[0]) > 40.0
        mf = mean_field_equivalent(res).solve(n_nodes=81, max_iter=800, relax=0.3)
        assert mf.meta["converged"]
        delta = float(good.signal_gain_dB[0]) - float(mf.signal_gain_dB[0])
        assert abs(delta) < 1.0, delta                 # measured +0.0615 dB: a 16x margin
        assert delta > 0.0, delta                      # core-guided pump -> resolved ABOVE
        # and the four F-14 diagnostics are reported here too
        for k in ("endpoint_residual", "profile_residual", "relax", "min_power_W"):
            assert k in good.meta, k
        assert good.meta["relax"] == 0.3

    def test_the_resolved_default_is_byte_identical_and_relax_is_validated(self):
        from dynameta.optics.fiber_amp import ResolvedFiberAmplifier
        res = ResolvedFiberAmplifier(YB, FiberSpec(3.0e-6, 0.12, 6.0e25, 0.5),
                                     [Pump(1.0, 0.976e-6)], [Signal(1e-3, 1.060e-6)], None,
                                     n_quad=8)
        a, b = res.solve(n_nodes=41, max_iter=20), res.solve(n_nodes=41, max_iter=20, relax=1.0)
        assert np.array_equal(a.power_W, b.power_W)          # relax=1.0 skips the blend entirely
        assert np.array_equal(a.nbar2_rz, b.nbar2_rz)
        for bad in (0.0, -0.1, 1.5):
            with pytest.raises(ValueError, match="relax"):
                res.solve(n_nodes=21, max_iter=2, relax=bad)

    def test_the_eryb_solver_has_the_same_remedy(self):
        """The third copy of the iteration. relax = 1.0 must be BYTE-identical on an existing
        eryb fixture (the blend is skipped, not multiplied by 1.0), and the knob must exist and
        be validated."""
        from dynameta.optics.fiber_amp.eryb import ErYbAmplifier

        def ey():
            return ErYbAmplifier(ER, ytterbium("phosphosilicate"),
                                 FiberSpec(3.0e-6, 0.20, 1.0e25, 4.0, clad_radius_m=62.5e-6),
                                 [Pump(1.0, 0.976e-6, "fwd", cladding=True)],
                                 [Signal(1e-4, 1.550e-6)], AseBand(1.52e-6, 1.575e-6, 6),
                                 n_yb_m3=2e26)

        a, b = ey().solve(n_nodes=61), ey().solve(n_nodes=61, relax=1.0)
        assert np.array_equal(a.power_W, b.power_W) and np.array_equal(a.nbar2_z, b.nbar2_z)
        for k in ("endpoint_residual", "profile_residual", "relax", "min_power_W"):
            assert k in a.meta, k
        assert a.meta["relax"] == 1.0
        # a damped solve finds the SAME fixed point -- the property that makes relax safe
        c = ey().solve(n_nodes=61, max_iter=400, relax=0.4)
        assert c.meta["converged"] and c.meta["relax"] == 0.4
        assert float(c.signal_gain_dB[0]) == pytest.approx(float(a.signal_gain_dB[0]), abs=1e-2)
        for bad in (0.0, -0.1, 1.5):
            with pytest.raises(ValueError, match="relax"):
                ey().solve(n_nodes=21, max_iter=2, relax=bad)

    def test_a_counter_propagating_pump_needs_under_relaxation(self):
        """Found by adversarial review AFTER the audit was written, and a much wider scope than the
        audit's low-signal case: a COUNTER-pumped amplifier fails at relax = 1 at ANY signal level.
        Measured on the Yb reference at 20 mW in -- relax = 1 gives -30.8 dB with an endpoint
        residual of 6.5e6, while relax <= 0.3 converges to +17.9 dB. Co-pumping the SAME fiber
        converges at relax = 1 in 4 iterations, which is what isolates the cause to the
        counter-propagating coupling rather than to a hard amplifier."""
        def yb(pumps, relax):
            a = FiberAmplifier(YB, FiberSpec(3.0e-6, 0.12, 6.0e25, 6.0), pumps,
                               [Signal(20e-3, 1.060e-6)], AseBand(1.02e-6, 1.10e-6, 20))
            return a.solve(n_nodes=201, relax=relax, max_iter=800)

        co = yb([Pump(2.0, 0.976e-6)], 1.0)
        assert co.meta["converged"] and co.meta["iterations"] < 20     # the easy case
        # relax <= 0.3 finds the SAME fixed point from two different paths -> it is the real one
        a, b = yb([Pump(2.0, 0.976e-6, "bwd")], 0.3), yb([Pump(2.0, 0.976e-6, "bwd")], 0.15)
        assert a.meta["converged"] and b.meta["converged"]
        assert float(a.signal_gain_dB[0]) == pytest.approx(float(b.signal_gain_dB[0]), abs=1e-3)
        assert float(a.signal_gain_dB[0]) > 10.0
        # and it is physical: counter-pumping cannot differ from co-pumping by tens of dB
        assert abs(float(a.signal_gain_dB[0]) - float(co.signal_gain_dB[0])) < 3.0
        # relax = 1: whether the marginally-stable Gauss-Seidel oscillation decays is
        # PLATFORM-DEPENDENT (this box: spurious branch at -30.8 dB, endpoint residual 6.5e6;
        # the CI py3.11/py3.13 stacks: it decays and converges -- the original form of this test
        # pinned the local pathology as universal and went red there, first PR-#10 CI round).
        # The contract F-14 actually guards is NO SILENT WRONG ANSWER: either the solve reports
        # its own failure loudly, or it converged -- in which case it must be on the SAME fixed
        # point the doubly-cross-checked under-relaxed solve found, not the spurious branch.
        bad = yb([Pump(2.0, 0.976e-6, "bwd")], 1.0)
        if bad.meta["converged"]:
            assert float(bad.signal_gain_dB[0]) == pytest.approx(
                float(a.signal_gain_dB[0]), abs=0.1)
            assert bad.meta["endpoint_residual"] < 1e-5
        else:
            assert bad.meta["endpoint_residual"] > 1e3
            assert float(bad.signal_gain_dB[0]) < float(a.signal_gain_dB[0]) - 10.0

    def test_the_converged_counter_pumped_solve_conserves_photons(self):
        """Independent physical check on the branch relax<1 selects: the signal power added plus
        the ASE emitted cannot exceed the absorbed pump times the quantum-defect ceiling."""
        amp = FiberAmplifier(YB, FiberSpec(3.0e-6, 0.12, 6.0e25, 6.0),
                             [Pump(2.0, 0.976e-6, "bwd")], [Signal(20e-3, 1.060e-6)],
                             AseBand(1.02e-6, 1.10e-6, 20))
        r = amp.solve(n_nodes=201, relax=0.2, max_iter=2000)
        assert r.meta["converged"]
        si, pi = r.kind.index("signal"), r.kind.index("pump")
        added = float(r.power_W[si, -1] - r.power_W[si, 0])
        absorbed = float(r.power_W[pi, -1] - r.power_W[pi, 0])   # bwd: launched at L, exits at 0
        ase = float(sum(r.power_W[k, -1] if r.u[k] > 0 else r.power_W[k, 0]
                        for k in range(len(r.kind)) if r.kind[k] == "ase"))
        assert added > 0.0 and absorbed > 0.0
        assert (added + ase) <= absorbed * stokes_limit(0.976e-6, 1.060e-6) * 1.001


# ============================ F-3: the public channel plan ==================================

class TestF3ChannelPlan:
    def test_plan_indices_are_result_rows(self):
        """The contract that makes the plan useful: a plan index IS a power_W row index."""
        amp = _edfa(bins=4)
        pl, r = amp.channel_plan(), _edfa(bins=4).solve(n_nodes=41)
        assert isinstance(pl, ChannelPlan)
        assert np.allclose(pl.lambda_m, r.lambda_m) and np.allclose(pl.direction, r.u)
        assert list(pl.kind) == list(r.kind) and np.array_equal(pl.is_ase, r.is_ase)
        # and the forward-ASE spectrum reached through the plan matches noise.py's own extraction
        sp = output_ase_spectrum(r, "fwd", signal_lambda_m=1.560e-6)
        mine = r.power_W[pl.indices("ase", "fwd"), -1]
        assert np.allclose(np.sort(mine), np.sort(sp.power_W), rtol=0, atol=0)

    def test_dnu_is_a_frequency_width_not_a_wavelength_width(self):
        """The specific bookkeeping an external consumer had to re-derive, and the easy way to get
        it wrong. dnu must equal |c/edge_i - c/edge_{i+1}| for the band's own bin edges."""
        lo, hi, n = 1.52e-6, 1.575e-6, 8
        pl = FiberAmplifier(ER, _edf(), [Pump(0.1, 0.98e-6)], [Signal(1e-6, 1.56e-6)],
                            AseBand(lo, hi, n)).channel_plan()
        edges = np.linspace(lo, hi, n + 1)
        want = np.abs(C_LIGHT / edges[:-1] - C_LIGHT / edges[1:])
        got = pl.dnu_hz[pl.indices("ase", "fwd")]
        assert np.allclose(np.sort(got), np.sort(want), rtol=1e-12)

    def test_indices_validates_direction(self):
        with pytest.raises(ValueError, match="direction"):
            _edfa(bins=2).channel_plan().indices("ase", "sideways")

    def test_eryb_implements_the_same_protocol(self):
        from dynameta.optics.fiber_amp import ErYbAmplifier
        ey = ErYbAmplifier(ER, YB, FiberSpec(2.0e-6, 0.2, 1e25, 2.0), [Pump(1.0, 0.976e-6)],
                           [Signal(1e-4, 1.55e-6)], AseBand(1.52e-6, 1.575e-6, 4), n_yb_m3=2e26)
        pl = ey.channel_plan()
        assert isinstance(pl, ChannelPlan)
        # co-doped: every channel carries TWO ions' cross-sections, so there is no single ChannelSet
        assert pl.channels is None
        assert pl.indices("pump").size == 1 and pl.indices("signal").size == 1


# ============================ F-2: transient ASE + the noise bridge =========================

class TestF2TransientAse:
    def test_transient_ase_matches_the_steady_extraction_on_the_same_frame(self):
        """The ASE was already computed and discarded. Gate it against noise.output_ase_spectrum
        applied to the same frame -- a different code path over the same numbers."""
        amp = _edfa(pump_W=0.3, sig_W=5e-3, bins=12)
        t = np.linspace(0.0, 4e-3, 60)
        tr = simulate_transient(amp, t, n_nodes=61, store_profiles=True)
        fr = tr.frame_as_steady(len(t) - 1)
        sp = output_ase_spectrum(fr, "fwd", signal_lambda_m=1.560e-6)
        o = np.argsort(tr.ase_lambda_m)
        assert np.allclose(tr.ase_fwd_W[-1][o], sp.power_W, rtol=0, atol=0)
        assert np.allclose(tr.ase_psd_1pol_W_Hz("fwd")[-1][o], sp.psd_1pol, rtol=0, atol=0)

    def test_capture_does_not_perturb_the_existing_arrays(self):
        amp, t = _edfa(bins=6), np.linspace(0.0, 2e-3, 40)
        a = simulate_transient(amp, t, n_nodes=41)
        b = simulate_transient(amp, t, n_nodes=41, store_profiles=True)
        for f in ("nbar2_zt", "signal_out_W", "pump_out_W", "signal_gain_dB"):
            assert np.array_equal(getattr(a, f), getattr(b, f)), f

    def test_the_noise_layer_runs_on_a_transient_frame(self):
        """The gap: analyze_noise accepts only a SteadyStateResult, so transient noise was
        unreachable. At the march's fixed point the frame must reproduce the steady solve."""
        amp = _edfa(pump_W=0.3, sig_W=5e-3, bins=12)
        t = np.linspace(0.0, 60e-3, 80)                    # >> tau, i.e. at the fixed point
        tr = simulate_transient(amp, t, n_nodes=121, store_profiles=True)
        fr = tr.frame_as_steady(len(t) - 1)
        n_fr, n_ss = analyze_noise(fr, 1.560e-6), analyze_noise(amp.solve(n_nodes=121), 1.560e-6)
        assert abs(n_fr.nf_dB - n_ss.nf_dB) < 0.05
        assert abs(n_fr.osnr_dB - n_ss.osnr_dB) < 0.05
        assert np.isfinite(detection_noise(fr, 1.560e-6, optical_bw_Hz=50e9,
                                          electrical_bw_Hz=10e9).snr_elec_dB)

    def test_the_validity_flag_cannot_be_lost_through_the_frame(self):
        amp, t = _edfa(bins=6), np.linspace(0.0, 2e-3, 30)
        tr = simulate_transient(amp, t, n_nodes=41, store_profiles=True)
        fr = tr.frame_as_steady(3)
        assert fr.meta["quasi_static_valid"] == tr.meta["quasi_static_valid"]
        assert fr.meta["transient_frame"] == 3

    def test_frame_as_steady_refuses_without_profiles(self):
        tr = simulate_transient(_edfa(bins=4), np.linspace(0, 1e-3, 10), n_nodes=41)
        assert tr.ase_fwd_W is not None                    # ASE is always kept (it is free)
        with pytest.raises(ValueError, match="store_profiles"):
            tr.frame_as_steady(0)

    def test_a_driven_frame_reports_the_gain_of_its_own_launch(self):
        """The defect: `frame_as_steady` divided the frame's output by `plan.launched_W`, the
        amplifier's CONFIGURED input -- which is exactly the quantity `signal_drive` overrides. Under
        a 2x drive the reported gain was therefore high by the drive ratio itself, a measured
        +3.010504 dB against 10 log10(2) = 3.010300, and that error propagated into anything
        computed over frames (metrics.gain_flatness read 23.086 dB where the truth was 3.086).

        Two independent oracles, because the frame must agree with BOTH: the march's own
        signal_gain_dB for the same step (exactly -- same numerator, same denominator), and a plain
        steady solve of an amplifier CONFIGURED at the driven power (to 2.04e-4 dB, the transient's
        own residual against its fixed point on this fixture)."""
        sig = 5e-3
        fib = FiberSpec(1.4e-6, 0.24, 1.0e25, 6.0)
        amp = FiberAmplifier(ER, fib, [Pump(0.3, 0.980e-6)], [Signal(sig, 1.560e-6)],
                             AseBand(1.52e-6, 1.575e-6, 12))
        t = np.linspace(0.0, 60e-3, 80)                     # >> tau, i.e. at the fixed point
        tr = simulate_transient(amp, t, n_nodes=61, store_profiles=True,
                                signal_drive=lambda tt: np.array([2.0 * sig]))
        fr = tr.frame_as_steady(len(t) - 1)
        assert float(fr.power_W[fr.kind.index("signal"), 0]) == pytest.approx(2.0 * sig, rel=1e-15)
        assert float(fr.signal_gain_dB[0]) == float(tr.signal_gain_dB[-1, 0])
        driven = FiberAmplifier(ER, fib, [Pump(0.3, 0.980e-6)], [Signal(2.0 * sig, 1.560e-6)],
                                AseBand(1.52e-6, 1.575e-6, 12)).solve(n_nodes=61)
        assert driven.meta["converged"]
        assert float(fr.signal_gain_dB[0]) == pytest.approx(float(driven.signal_gain_dB[0]),
                                                            abs=1e-3)
        # and the pre-fix denominator is exactly what would break it, by the drive ratio
        wrong = 10.0 * math.log10(float(fr.power_W[fr.kind.index("signal"), -1])
                                  / float(tr.plan.launched_W[fr.kind.index("signal")]))
        assert wrong - float(fr.signal_gain_dB[0]) == pytest.approx(10.0 * math.log10(2.0),
                                                                    abs=1e-3)

    def test_store_profiles_refuses_an_oversized_allocation(self):
        """This box's history is silent process deaths from memory pressure, and the (Nt, K, Nz)
        matrix is the product of three innocuous-looking arguments: 2000 steps x 42 channels x 201
        nodes is already 135 MB, and the shapes that reach 6-12 GiB are not exotic. The guard fires
        BEFORE the initial steady solve and before np.empty commits any pages, so this test costs
        nothing and allocates nothing."""
        amp = _edfa(bins=10)
        n_ch = amp.channel_plan().lambda_m.size
        need = 3 * 1024 ** 3                                # ask for ~3 GiB
        nt = int(need / (8 * n_ch * 401)) + 1
        with pytest.raises(ValueError, match="store_profiles") as e:
            simulate_transient(amp, np.linspace(0.0, 1e-3, nt), n_nodes=401, store_profiles=True)
        assert "(%d, %d, %d)" % (nt, n_ch, 401) in str(e.value)     # the shape must be in it
        # and the guard is scoped to the opt-in: a modest request still returns the matrix
        ok = simulate_transient(amp, np.linspace(0.0, 1e-3, 8), n_nodes=41, store_profiles=True)
        assert ok.power_zt is not None and ok.power_zt.shape == (8, n_ch, 41)


# ============================ F-4: multi-pump Stokes ceiling ================================

class TestF4StokesCeiling:
    def test_single_pump_is_the_identity(self):
        pumps = [Pump(0.25, 0.980e-6)]
        assert effective_pump_lambda_m(pumps) == pytest.approx(0.980e-6, rel=1e-15)

    def test_photon_weighted_mean_lies_between_the_two_wavelengths(self):
        """The defect: the ceiling was read from pumps[0] alone, so a 915+976 nm Yb pump reported
        0.8632 while its own measured slope was 0.8660 -- an apparent energy-conservation
        violation that vanished on swapping the pump order."""
        pumps = [Pump(1.0, 0.915e-6), Pump(1.0, 0.976e-6)]
        eff = effective_pump_lambda_m(pumps)
        assert 0.915e-6 < eff < 0.976e-6
        # power-weighted HARMONIC mean, verified against the closed form
        assert eff == pytest.approx(2.0 / (1.0 / 0.915e-6 + 1.0 / 0.976e-6), rel=1e-14)
        ceil = stokes_limit(eff, 1.060e-6)
        assert stokes_limit(0.915e-6, 1.060e-6) < ceil < stokes_limit(0.976e-6, 1.060e-6)

    def test_order_independence(self):
        a = effective_pump_lambda_m([Pump(0.3, 0.915e-6), Pump(0.7, 0.976e-6)])
        b = effective_pump_lambda_m([Pump(0.7, 0.976e-6), Pump(0.3, 0.915e-6)])
        assert a == pytest.approx(b, rel=1e-15)

    def test_zero_pump_is_nan_not_a_crash(self):
        assert math.isnan(effective_pump_lambda_m([Pump(0.0, 0.98e-6)]))


# ============================ F-5 / F-8: receiver electronics ===============================

class TestF5F8Detection:
    def _base(self):
        r = _edfa(bins=10).solve(n_nodes=121)
        return r, dict(optical_bw_Hz=50e9, electrical_bw_Hz=10e9, quantum_efficiency=1.0)

    def test_defaults_are_the_optical_only_budget(self):
        r, kw = self._base()
        b = detection_noise(r, 1.560e-6, **kw)
        assert b.var_thermal == 0.0
        assert b.var_total == b.var_optical
        assert b.var_total == pytest.approx(b.var_shot + b.var_sig_sp + b.var_sp_sp, rel=1e-15)
        assert b.apd_gain == 1.0 and b.apd_excess_noise_F == 1.0

    def test_dark_current_is_reachable_and_correct(self):
        """The gap: amp_noise.beat_noise_variances always accepted I_dark_A but detection_noise
        never forwarded it. Gate the delta against the closed form 2 q I_d B_e."""
        from dynameta.constants import Q_E
        r, kw = self._base()
        i_d = 1e-6
        a, b = detection_noise(r, 1.560e-6, **kw), detection_noise(r, 1.560e-6, dark_current_A=i_d,
                                                                  **kw)
        assert b.var_shot - a.var_shot == pytest.approx(2.0 * Q_E * i_d * 10e9, rel=1e-12)

    def test_thermal_terms_match_their_closed_forms(self):
        r, kw = self._base()
        b = detection_noise(r, 1.560e-6, tia_current_noise_A_rtHz=12e-12, **kw)
        assert b.var_thermal == pytest.approx((12e-12) ** 2 * 10e9, rel=1e-14)
        j = detection_noise(r, 1.560e-6, load_ohm=50.0, temperature_K=300.0, **kw)
        assert j.var_thermal == pytest.approx(4.0 * KB * 300.0 * 10e9 / 50.0, rel=1e-14)

    def test_noise_figure_is_independent_of_the_receiver(self):
        """The amplifier's NF must not depend on what detects it (the audit-S3-10 principle). Only
        snr_elec_dB may move.

        THIS GATE USED TO SWEEP ONLY THE TWO KNOBS THAT WERE ALREADY INVARIANT (the TIA density and
        the load), so it passed while the DIODE's own knobs leaked straight into the reported
        amplifier noise figure. Measured on this fixture before the fix: 3.284766 dB at zero dark
        current, 3.415694 dB at 10 mA of it, and 3.620575 dB at an APD excess-noise factor of 60 --
        the last two are properties of the detector, not of the fiber. The avalanche GAIN M was the
        only one that already cancelled (bit-identical at M = 1, 8, 100). All four are swept here,
        with `==` rather than a tolerance because the fix routes the NF through an arithmetically
        identical dark-free, F-free variance."""
        r, kw = self._base()
        a = detection_noise(r, 1.560e-6, **kw)
        for extra in (dict(tia_current_noise_A_rtHz=50e-12, load_ohm=50.0),
                      dict(dark_current_A=1e-6), dict(dark_current_A=10e-3),
                      dict(apd_gain=8.0, apd_excess_noise_F=3.0),
                      dict(apd_gain=8.0, apd_excess_noise_F=60.0),
                      dict(apd_gain=100.0, apd_excess_noise_F=1.0),
                      dict(dark_current_A=1e-3, apd_gain=8.0, apd_excess_noise_F=6.0,
                           tia_current_noise_A_rtHz=12e-12, load_ohm=50.0)):
            b = detection_noise(r, 1.560e-6, **extra, **kw)
            assert b.nf_beat_dB == a.nf_beat_dB, extra
            assert b.meta["snr_optical_nf_dB"] == a.meta["snr_optical_nf_dB"], extra
        # ... while the RECEIVER's own numbers do move, so the invariance is not a dead branch
        hot = detection_noise(r, 1.560e-6, tia_current_noise_A_rtHz=50e-12, load_ohm=50.0, **kw)
        assert hot.added_rin_per_Hz == a.added_rin_per_Hz     # RIN is optical too
        assert hot.snr_elec_dB < a.snr_elec_dB
        dark = detection_noise(r, 1.560e-6, dark_current_A=10e-3, **kw)
        assert dark.var_shot > a.var_shot                     # the dark current IS in the budget
        assert dark.meta["snr_optical_dB"] < a.meta["snr_optical_dB"]
        assert dark.meta["snr_optical_nf_dB"] == a.meta["snr_optical_nf_dB"]

    def test_receiver_inputs_are_validated(self):
        """Each of these produced a silently wrong number rather than an error: a negative
        temperature or dark current makes a NEGATIVE variance (and then `inf` SNR out of the
        var_total <= 0 branch), load_ohm = 0.0 was falsy so the Johnson term asked for was
        silently dropped, and a non-finite M or F walked past the `M <= 0 or F < 1` test because
        every comparison with NaN is False."""
        r, kw = self._base()
        for bad, pat in ((dict(temperature_K=-1.0), "temperature_K"),
                         (dict(load_ohm=0.0), "load_ohm"),
                         (dict(load_ohm=-50.0), "load_ohm"),
                         (dict(dark_current_A=-1e-9), "dark_current_A"),
                         (dict(apd_gain=float("nan")), "finite"),
                         (dict(apd_gain=float("inf")), "finite"),
                         (dict(apd_excess_noise_F=float("nan")), "finite")):
            with pytest.raises(ValueError, match=pat):
                detection_noise(r, 1.560e-6, **bad, **kw)
        # T = 0 K is legal (a cryogenic front end contributes no Johnson noise), and load_ohm=None
        # still means "omit", which is what the 0.0 rejection above must not have broken
        assert detection_noise(r, 1.560e-6, load_ohm=50.0, temperature_K=0.0,
                               **kw).var_thermal == 0.0
        assert detection_noise(r, 1.560e-6, load_ohm=None, **kw).var_thermal == 0.0

    def test_apd_identity_and_scaling(self):
        r, kw = self._base()
        a = detection_noise(r, 1.560e-6, **kw)
        assert detection_noise(r, 1.560e-6, apd_gain=1.0, apd_excess_noise_F=1.0,
                              **kw).var_total == a.var_total
        M, F = 8.0, 3.0
        b = detection_noise(r, 1.560e-6, apd_gain=M, apd_excess_noise_F=F, **kw)
        assert b.var_shot == pytest.approx(a.var_shot * M * M * F, rel=1e-12)
        assert b.var_sig_sp == pytest.approx(a.var_sig_sp * M * M, rel=1e-12)
        assert b.i_signal_A == pytest.approx(a.i_signal_A * M, rel=1e-12)

    def test_apd_is_validated(self):
        r, kw = self._base()
        with pytest.raises(ValueError, match="apd"):
            detection_noise(r, 1.560e-6, apd_gain=2.0, apd_excess_noise_F=0.5, **kw)

    def test_thermal_can_win_dominant_term_only_when_supplied(self):
        r, kw = self._base()
        assert detection_noise(r, 1.560e-6, **kw).dominant_term in ("shot", "sig-sp", "sp-sp")
        hot = detection_noise(r, 1.560e-6, tia_current_noise_A_rtHz=1e-9, **kw)
        assert hot.dominant_term == "thermal"


# ============================ F-1: the Er C-band refit ======================================

class TestF1ErbiumCBand:
    LAM = np.arange(900.0, 1650.0, 0.5) * 1e-9

    def test_the_refit_is_the_default(self):
        """Made the default on 2026-08-05. erbium() must BE the refit, and cband_refit=False must
        still reproduce the pre-2026-08 spectra exactly for anyone reproducing an old result."""
        assert np.array_equal(erbium().sigma_a.sigma(self.LAM),
                              erbium(cband_refit=True).sigma_a.sigma(self.LAM))
        old = erbium(cband_refit=False)
        assert not np.array_equal(erbium().sigma_a.sigma(self.LAM), old.sigma_a.sigma(self.LAM))
        # the legacy spectra are the two independent Gaussian sums, unchanged
        assert float(old.sigma_a.sigma(1.530e-6)) == pytest.approx(5.9309e-25, rel=1e-4)
        assert float(old.sigma_e.sigma(1.560e-6)) == pytest.approx(3.0400e-25, rel=1e-4)

    def test_the_legacy_model_violates_its_own_mccumber_relation(self):
        """The finding itself, still asserted against cband_refit=False so the defect it records
        cannot be lost now that the default no longer exhibits it: sigma_e disagrees with
        sigma_e_mccumber(sigma_a) by >2x mid-band while agreeing at the anchors."""
        ion = erbium(cband_refit=False)
        lam = np.arange(1534.0, 1563.0, 1.0) * 1e-9
        ratio = ion.sigma_e.sigma(lam) / np.array([float(ion.sigma_e_mccumber(x)) for x in lam])
        assert ratio.max() > 2.0
        for anchor in (1.530e-6, 1.560e-6):
            assert abs(float(ion.sigma_e.sigma(anchor))
                       / float(ion.sigma_e_mccumber(anchor)) - 1.0) < 0.02

    def test_the_refit_is_mccumber_exact(self):
        ion = erbium()
        lam = np.arange(1520.0, 1571.0, 0.5) * 1e-9
        ratio = ion.sigma_e.sigma(lam) / np.array([float(ion.sigma_e_mccumber(x)) for x in lam])
        assert np.allclose(ratio, 1.0, rtol=1e-12)

    def test_the_refit_removes_the_interior_sigma_a_minimum(self):
        c = np.arange(1533.0, 1570.5, 0.5) * 1e-9
        assert not np.all(np.diff(erbium(cband_refit=False).sigma_a.sigma(c)) < 0.0)   # the defect
        assert np.all(np.diff(erbium().sigma_a.sigma(c)) < 0.0)                       # the default

    def test_the_refit_preserves_the_trusted_anchors_and_pump_bands(self):
        old, new = erbium(cband_refit=False), erbium()
        for anchor in (1.530e-6, 1.560e-6):
            assert abs(float(new.sigma_a.sigma(anchor)) / float(old.sigma_a.sigma(anchor))
                       - 1.0) < 0.02
        for pump in (0.976e-6, 0.980e-6, 1.480e-6):
            assert float(new.sigma_a.sigma(pump)) == float(old.sigma_a.sigma(pump))

    def test_the_refit_bleaches_the_1480_nm_in_band_pump(self):
        """A real accuracy gain, not just consistency: 1480 nm pumping genuinely cannot fully
        invert Er, and the default model's sigma_e(1480) ~ 0 said it could."""
        new = erbium(cband_refit=True)
        sa, se = float(new.sigma_a.sigma(1.48e-6)), float(new.sigma_e.sigma(1.48e-6))
        assert se > 1e-27
        assert 0.6 < sa / (sa + se) < 0.85
        assert float(erbium(cband_refit=False).sigma_e.sigma(1.48e-6)) < 1e-29   # old, wrong

    def test_at_temperature_does_not_double_count_mccumber(self):
        """The silent footgun: at_temperature McCumber-scales the ion's EXISTING sigma_e. Applied
        to a McCumber-DERIVED sigma_e it must still apply exactly ONE T-factor."""
        new = erbium()
        assert at_temperature(new, 300.0) is new                      # exact no-op at T_ref
        hot = at_temperature(new, 350.0)
        lam = 1.550e-6
        got = float(hot.sigma_e.sigma(lam)) / float(new.sigma_e.sigma(lam))
        want = math.exp((new.eps_J - H_PLANCK * C_LIGHT / lam)
                        * (1.0 / (KB * 350.0) - 1.0 / (KB * 300.0)))
        assert got == pytest.approx(want, rel=1e-12)

    def test_the_refit_amplifier_still_solves_and_respects_stokes(self):
        from dynameta.optics.fiber_amp import power_conversion_efficiency
        amp = FiberAmplifier(erbium(), FiberSpec(1.4e-6, 0.23, 1e25, 8.0),
                             [Pump(0.3, 0.976e-6)], [Signal(5e-3, 1.550e-6)],
                             AseBand(1.52e-6, 1.575e-6, 16))
        r = amp.solve(n_nodes=161)
        assert r.meta["converged"]
        pce = power_conversion_efficiency(amp, r)
        assert 0.0 < pce < stokes_limit(0.976e-6, 1.550e-6)


# ============================ F-12: the McCumber reference energy ===========================

class TestF12McCumberEps:
    def test_default_eps_is_the_zero_line(self):
        assert ER.mccumber_eps_J is None
        assert ER.eps_J == pytest.approx(H_PLANCK * C_LIGHT / ER.zero_line_m, rel=1e-15)

    def test_a_fitted_eps_is_honoured_and_validated(self):
        from dynameta.optics.fiber_amp import RareEarthIon
        eps = 0.95 * H_PLANCK * C_LIGHT / 0.976e-6
        ion = RareEarthIon("Yb3+fit", YB.sigma_a, YB.sigma_e, tau_s=YB.tau_s,
                           zero_line_m=YB.zero_line_m, mccumber_eps_J=eps)
        assert ion.eps_J == pytest.approx(eps, rel=1e-15)
        # a smaller eps means a smaller (eps - h nu) to the red, i.e. less inflated sigma_e
        lam = 1.060e-6
        assert float(ion.sigma_e_mccumber(lam)) < float(YB.sigma_e_mccumber(lam))
        with pytest.raises(ValueError, match="mccumber_eps_J"):
            RareEarthIon("bad", YB.sigma_a, YB.sigma_e, tau_s=1e-3, zero_line_m=1e-6,
                         mccumber_eps_J=-1.0)

    def test_a_fitted_eps_survives_at_temperature(self):
        from dynameta.optics.fiber_amp import RareEarthIon
        eps = 0.95 * H_PLANCK * C_LIGHT / 0.976e-6
        ion = RareEarthIon("Yb3+fit", YB.sigma_a, YB.sigma_e, tau_s=YB.tau_s,
                           zero_line_m=YB.zero_line_m, mccumber_eps_J=eps)
        assert at_temperature(ion, 350.0).mccumber_eps_J == pytest.approx(eps, rel=1e-15)


# ============================ F-9: the Marcuse validity window ==============================

class TestF9MarcuseValidity:
    def test_v_number_closed_form(self):
        assert v_number(1.4e-6, 0.24, 1.55e-6) == pytest.approx(
            2.0 * math.pi * 1.4e-6 * 0.24 / 1.55e-6, rel=1e-15)

    def test_validity_is_array_safe(self):
        """The reason this is a query and not a warning inside mode_field_radius_m: lambda_m is
        routinely the whole channel array, so a scalar `if` on V would raise."""
        lam = np.array([0.976e-6, 1.03e-6, 1.06e-6])
        ok, vmin, vmax = marcuse_validity(25e-6, 0.054, lam)
        assert isinstance(ok, bool) and vmin <= vmax
        assert not ok and vmax > V_MARCUSE_MAX          # a 25 um LMA core is far out of range

    def test_an_in_range_single_mode_fiber_passes(self):
        ok, vmin, vmax = marcuse_validity(1.4e-6, 0.24, np.array([1.52e-6, 1.56e-6, 1.575e-6]))
        assert ok and V_MARCUSE_MIN < vmin <= vmax < V_MARCUSE_MAX

    def test_no_warning_is_emitted_on_the_solver_hot_path(self):
        """A default warning would fire inside the solve for every LMA/cladding-pumped fixture, and
        the repo runs pytest with filterwarnings=error."""
        import warnings as _w
        fib = FiberSpec(5.0e-6, 0.08, 6.0e25, 2.0, clad_radius_m=62.5e-6)
        amp = FiberAmplifier(YB, fib, [Pump(20.0, 0.976e-6, "fwd", cladding=True)],
                             [Signal(1.0, 1.03e-6)], None)
        with _w.catch_warnings():
            _w.simplefilter("error")
            amp.solve(n_nodes=41, max_iter=20)


# ============================ F-10: the passive fiber =======================================

class TestF10PassiveFiber:
    def test_undoped_fiber_is_pure_beer_lambert(self):
        """The sentinel n_t_m3=1.0 is no longer needed. Gate against the analytic attenuation."""
        alpha, L, p_in = 1e-4, 30.0, 1e-3
        f = FiberSpec(3.0e-6, 0.14, 0.0, L, background_loss_per_m=alpha)
        r = FiberAmplifier(ER, f, [Pump(0.0, 0.98e-6)], [Signal(p_in, 1.55e-6)], None
                           ).solve(n_nodes=81)
        assert float(r.power_W[1, -1]) == pytest.approx(p_in * math.exp(-alpha * L), rel=1e-6)
        assert float(r.signal_gain_dB[0]) == pytest.approx(-10.0 * alpha * L / math.log(10.0),
                                                           rel=1e-6)

    def test_negative_density_is_still_rejected(self):
        with pytest.raises(ValueError, match="n_t_m3"):
            FiberSpec(1e-6, 0.1, -1.0, 1.0)

    def test_upconversion_with_no_ions_takes_the_linear_branch(self):
        """The divide-by-(C_up * n_t) guard: with no ions there is nothing to upconvert."""
        from dynameta.optics.fiber_amp import ChannelSet, metastable_fraction
        f = FiberSpec(3.0e-6, 0.14, 0.0, 1.0)
        ch = ChannelSet.build(ER, f, np.array([1.55e-6]), np.array([1.0]))
        got = metastable_fraction(ch, np.array([1e-3]), f, upconversion_C_up=3e-23)
        assert np.isfinite(got)
        assert got == pytest.approx(metastable_fraction(ch, np.array([1e-3]), f), rel=1e-15)

    def test_giles_calibration_still_requires_ions(self):
        from dynameta.optics.fiber_amp import giles_calibrated_fiber
        with pytest.raises(ValueError, match="n_t_m3"):
            giles_calibrated_fiber("Er", [1.5e-6, 1.6e-6], [1.0, 1.0], [1.0, 1.0], n_t_m3=0.0,
                                   core_radius_m=1.4e-6, na=0.24, length_m=1.0, tau_s=1e-2,
                                   zero_line_m=1.53e-6)

    def _passive(self, bins=6):
        """A 30 m undoped span with a real background loss -- and an AseBand, so the ASE machinery
        is actually exercised rather than skipped."""
        alpha, L, p_in = 1e-4, 30.0, 1e-3
        f = FiberSpec(3.0e-6, 0.14, 0.0, L, background_loss_per_m=alpha)
        amp = FiberAmplifier(ER, f, [Pump(0.0, 0.98e-6)], [Signal(p_in, 1.55e-6)],
                             AseBand(1.52e-6, 1.58e-6, bins))
        return amp, amp.solve(n_nodes=81), alpha, L, p_in

    def test_the_noise_layer_on_a_passive_span_is_a_documented_contract(self):
        """The audit-doc row claimed "18 further entry points confirmed finite" with n_t = 0 and no
        such gate was ever written; the doc row is corrected to what is gated, and this is it.

        On a passive span analyze_noise is NOT finite everywhere, and refusing would be wrong too,
        because the two non-finite fields are the ones that are genuinely undefined while the rest
        are exactly right. Pinned here so the contract is a decision and not an accident."""
        amp, r, alpha, L, _ = self._passive()
        nr = analyze_noise(r, 1.55e-6)
        # NF of a passive attenuator IS 1/G -- the PSD form gives it for free with zero ASE
        assert nr.nf_dB == pytest.approx(10.0 * alpha * L / math.log(10.0), rel=1e-6)
        assert nr.nf_dB == pytest.approx(-float(r.signal_gain_dB[0]), rel=1e-12)
        assert math.isinf(nr.osnr_dB) and nr.osnr_dB > 0    # no ASE -> no noise to beat it down
        assert math.isnan(nr.n_sp)                          # no inversion -> undefined, not zero
        assert math.isnan(nr.n_sp_local_min)
        # every POWER is finite, and the ASE really is identically zero
        assert float(np.max(nr.fwd_ase.power_W)) == 0.0
        assert float(np.max(nr.bwd_ase.power_W)) == 0.0
        assert np.isfinite(nr.meta["P_signal_out_W"]) and nr.meta["P_ase_ref_W"] == 0.0

    def test_the_entry_points_gated_finite_with_no_ions(self):
        """The concrete list the audit-doc row now names -- each called on the SAME passive solve.
        `analyze_noise` is deliberately absent: it has its own contract, gated above."""
        from dynameta.optics.fiber_amp.thermal import heat_load_per_m, total_heat_W
        amp, r, alpha, L, p_in = self._passive()
        assert np.all(np.isfinite(r.power_W)) and np.all(np.isfinite(r.nbar2_z))
        assert np.all(np.isfinite(output_ase_spectrum(r, "fwd", signal_lambda_m=1.55e-6).psd_1pol))
        assert np.all(np.isfinite(heat_load_per_m(r)))
        # a lossy passive span dissipates exactly what it attenuates
        assert float(total_heat_W(r)) == pytest.approx(p_in * (1.0 - math.exp(-alpha * L)),
                                                       rel=1e-6)
        b = detection_noise(r, 1.55e-6, optical_bw_Hz=50e9, electrical_bw_Hz=10e9)
        for v in (b.var_shot, b.var_sig_sp, b.var_sp_sp, b.var_total, b.snr_elec_dB, b.nf_beat_dB):
            assert np.isfinite(v)
        tr = simulate_transient(amp, np.linspace(0.0, 1e-3, 12), n_nodes=41, store_profiles=True)
        assert np.all(np.isfinite(tr.signal_out_W)) and np.all(np.isfinite(tr.nbar2_zt))
        assert float(tr.frame_as_steady(5).signal_gain_dB[0]) == pytest.approx(
            float(r.signal_gain_dB[0]), abs=1e-7)      # 41-node march vs 81-node solve: 9.7e-9 dB


# ============================ F-6: the comms layer ==========================================

class TestF6Comms:
    def test_levels_have_the_exact_mean_and_extinction_ratio(self):
        fmt = PamFormat(order=8, mean_power_W=4e-3, extinction_ratio=9.0)
        lv = pam_levels_W(fmt)
        assert float(np.mean(lv)) == pytest.approx(4e-3, abs=1e-18)
        assert lv[-1] / lv[0] == pytest.approx(9.0, rel=1e-12)
        assert np.all(np.diff(lv) > 0)

    def test_equal_variance_ser_matches_the_closed_form(self):
        """SER = 2(N-1)/N * Q(d / 2 sigma) for equal spacing and equal variances."""
        from scipy.special import erfc
        for n in (2, 4, 8, 16):
            mu, sg = np.arange(n, dtype=float), np.full(n, 0.12)
            want = 2.0 * (n - 1) / n * 0.5 * erfc(1.0 / (2.0 * 0.12) / math.sqrt(2.0))
            assert pam_ser(mu, sg) == pytest.approx(float(want), rel=1e-12)

    def test_ml_threshold_is_the_midpoint_only_for_equal_variances(self):
        assert ml_thresholds(np.array([0.0, 1.0]), np.array([0.2, 0.2]))[0] == pytest.approx(0.5)
        t = ml_thresholds(np.array([0.0, 1.0]), np.array([0.1, 0.4]))[0]
        assert 0.0 < t < 0.5            # leans toward the quieter level

    def test_ml_threshold_solves_the_likelihood_equation(self):
        """Discrimination: verify the returned root actually equalises the two log-likelihoods."""
        m1, m2, s1, s2 = 0.0, 1.0, 0.1, 0.4
        x = ml_thresholds(np.array([m1, m2]), np.array([s1, s2]))[0]
        f = lambda v, m, s: (v - m) ** 2 / (2 * s * s) + math.log(s)   # noqa: E731
        assert f(x, m1, s1) == pytest.approx(f(x, m2, s2), rel=1e-10)

    def test_per_level_variances_are_not_a_rescaling(self):
        """The trap the module exists to prevent: shot and sig-spont scale with the level power,
        spont-spont and thermal do not, so var_total cannot be scaled from one level."""
        fmt = PamFormat(order=4, mean_power_W=1e-3, extinction_ratio=12.0)
        # A THERMALLY dominated front end makes the discrimination unambiguous: with a
        # level-independent term dominating, sigma^2 is nearly constant, so sigma^2/P must vary by
        # nearly the extinction ratio -- whereas a rescaling of one level's var_total would hold it
        # exactly constant by construction.
        st = level_statistics(fmt, rho_sp_W_Hz=1e-18, responsivity_A_W=1.0, optical_bw_Hz=50e9,
                              electrical_bw_Hz=10e9, tia_current_noise_A_rtHz=1e-9)
        assert np.all(np.diff(st.terms_A2["shot"]) > 0)          # linear in the level power
        assert np.all(np.diff(st.terms_A2["sig_spont"]) > 0)     # linear in the level power
        assert np.allclose(st.terms_A2["spont_spont"], st.terms_A2["spont_spont"][0], rtol=0)
        assert np.allclose(st.terms_A2["thermal"], st.terms_A2["thermal"][0], rtol=0)
        ratio = (st.sigma_A ** 2) / st.levels_W
        assert ratio.max() / ratio.min() > 0.5 * fmt.extinction_ratio, ratio

    def test_per_level_variance_matches_a_hand_written_closed_form(self):
        """The exact statement, valid in EVERY regime, against an analytic side written out HERE
        rather than read back from `terms_A2` -- which is what the previous form of this gate did,
        making it circular. Splitting sigma^2(P) into its linear and constant parts:

            var(P) = [2 q R B_e + 4 R^2 rho B_e] P
                   + 2 q (R m rho B_o + I_dark) B_e + m R^2 rho^2 (2 B_o - B_e) B_e + var_thermal

        the CONSTANT part carries a shot term too -- from the detected ASE and the dark current --
        and the earlier gate's `flat` omitted it. Measured on this fixture the omission leaves a
        3.365e-08 relative error, 336x the rel=1e-10 the gate claimed; the assertion survived only
        because pytest.approx's default abs=1e-12 swamps A^2 variances this small. Below: rel=1e-10
        with abs=0 (the hand-written side lands at 1.7e-16, i.e. round-off), and the incomplete
        form asserted to FAIL that same tolerance."""
        from dynameta.constants import Q_E
        fmt = PamFormat(order=4, mean_power_W=1e-3, extinction_ratio=12.0)
        rho, R, b_o, b_e, m, i_d, tia = 1e-18, 1.0, 50e9, 10e9, 2, 5e-9, 1e-9
        st = level_statistics(fmt, rho_sp_W_Hz=rho, responsivity_A_W=R, optical_bw_Hz=b_o,
                              electrical_bw_Hz=b_e, m_pol=m, dark_current_A=i_d,
                              tia_current_noise_A_rtHz=tia)
        lin = 2.0 * Q_E * R * b_e + 4.0 * R ** 2 * rho * b_e
        const = (2.0 * Q_E * (R * m * rho * b_o + i_d) * b_e
                 + m * R ** 2 * rho ** 2 * (2.0 * b_o - b_e) * b_e
                 + tia ** 2 * b_e)
        var = st.sigma_A ** 2
        for k in range(var.size):
            assert var[k] == pytest.approx(lin * st.levels_W[k] + const, rel=1e-10, abs=0.0)

        # ... and the rescaling error is then the CONSTANT share times (P_k/P_0 - 1), exactly.
        scaled = var[0] * (st.levels_W / st.levels_W[0])
        for k in range(1, var.size):
            want = const * (st.levels_W[k] / st.levels_W[0] - 1.0)
            assert scaled[k] - var[k] == pytest.approx(want, rel=1e-10, abs=0.0)
            assert scaled[k] > var[k]        # rescaling OVERSTATES the noise on the upper levels

        # DISCRIMINATION: the pre-fix analytic side dropped 2 q (I_ase + I_dark) B_e from `const`.
        # It must not survive the tolerance this test claims.
        bad = st.terms_A2["spont_spont"][0] + st.terms_A2["thermal"][0]        # the old `flat`
        assert 0.0 < bad < const
        assert (const - bad) / const == pytest.approx(3.365e-8, rel=1e-2)      # the missing share
        with pytest.raises(AssertionError):
            for k in range(1, var.size):
                want = bad * (st.levels_W[k] / st.levels_W[0] - 1.0)
                assert scaled[k] - var[k] == pytest.approx(want, rel=1e-10, abs=0.0)

    def test_ser_degrades_with_order_and_improves_with_power(self):
        kw = dict(rho_sp_W_Hz=1e-19, signal_lambda_m=1.55e-6, quantum_efficiency=0.8,
                  optical_bw_Hz=75e9, tia_current_noise_A_rtHz=12e-12, load_ohm=50.0)
        s = [link_performance(PamFormat(order=o, mean_power_W=2e-5), **kw).ser
             for o in (4, 8, 16, 32)]
        assert np.all(np.diff(s) > 0), s
        lo = link_performance(PamFormat(order=16, mean_power_W=1e-5), **kw).ser
        hi = link_performance(PamFormat(order=16, mean_power_W=4e-5), **kw).ser
        assert hi < lo

    def test_osnr_is_loss_invariant(self):
        """Passive loss attenuates signal and ASE identically, so OSNR cannot depend on it."""
        kw = dict(rho_sp_W_Hz=1e-19, signal_lambda_m=1.55e-6, quantum_efficiency=0.8,
                  optical_bw_Hz=75e9)
        o = [link_performance(PamFormat(mean_power_W=1e-3), loss_dB=x, **kw).osnr_dB
             for x in (0.0, 10.0, 30.0)]
        assert max(o) - min(o) < 1e-9

    def test_max_pam_order_is_monotone_in_received_power(self):
        """RE-PINNED: the operating points now sit where the answer actually MOVES. The old sweep
        (10 / 25 / 40 / 55 dB) reads 16, 0, 0, 0 -- monotone by exhaustion, which cannot fail."""
        kw = dict(rho_sp_W_Hz=1e-19, signal_lambda_m=1.55e-6, quantum_efficiency=0.8,
                  optical_bw_Hz=75e9, tia_current_noise_A_rtHz=12e-12, load_ohm=50.0)
        fmt = PamFormat(mean_power_W=1e-3)
        orders = [max_pam_order(fmt, loss_dB=x, **kw) for x in (0.0, 5.0, 10.0, 15.0, 20.0)]
        assert orders == [128, 64, 16, 8, 2], orders

    def test_max_pam_order_scores_the_BIT_error_rate(self):
        """FEC_THRESHOLDS are pre-FEC BER (G.975.1, OIF-400ZR); scoring the SYMBOL error rate
        against them is wrong by log2(N) -- a whole octave of constellation. Measured over a
        9-point 0-20 dB loss sweep, 4 of the 9 answers moved, every one by exactly one octave."""
        from dataclasses import replace
        kw = dict(rho_sp_W_Hz=1e-19, signal_lambda_m=1.55e-6, quantum_efficiency=0.8,
                  optical_bw_Hz=75e9, tia_current_noise_A_rtHz=12e-12, load_ohm=50.0)
        fmt = PamFormat(mean_power_W=1e-3)
        orders = (2, 4, 8, 16, 32, 64, 128)
        moved = 0
        for x in (0.0, 2.5, 5.0, 7.5, 10.0, 12.5, 15.0, 17.5, 20.0):
            by_ber = max_pam_order(fmt, loss_dB=x, **kw)
            by_ser = 0
            for o in orders:
                r = link_performance(replace(fmt, order=int(o)), loss_dB=x, **kw)
                if np.isfinite(r.ser) and r.ser <= FEC_THRESHOLDS["ofec"]:
                    by_ser = int(o)
            assert by_ber >= by_ser                       # BER <= SER, so the BER answer is >=
            if by_ber != by_ser:
                moved += 1
                assert by_ber == 2 * by_ser, (x, by_ber, by_ser)     # exactly one octave
        assert moved == 4, moved
        # the crossing itself: at 5 dB of loss the two verdicts are 64 and 32
        assert max_pam_order(fmt, loss_dB=5.0, **kw) == 64
        # and the threshold is honoured on the winner but not on the next order up
        r = link_performance(replace(fmt, order=64), loss_dB=5.0, **kw)
        assert r.ber <= FEC_THRESHOLDS["ofec"] < link_performance(
            replace(fmt, order=128), loss_dB=5.0, **kw).ber

    def test_the_symbol_error_rate_has_a_log_domain_route(self):
        """Q(x) underflows to exactly 0.0 just above x = 37.5, so q_to_ser cannot represent the
        deep tail the module's own scope paragraph talks about. q_to_log10_ser must, and must agree
        with q_to_ser wherever q_to_ser is still representable."""
        from dynameta.optics.fiber_amp import q_to_log10_ser
        for q in (2.0, 5.0, 7.0, 20.0, 37.0):
            assert q_to_log10_ser(q) == pytest.approx(math.log10(q_to_ser(q)), rel=1e-12)
        assert q_to_ser(38.0) == 0.0                       # the floor, asserted rather than implied
        assert q_to_log10_ser(38.0) == pytest.approx(-315.539790, abs=1e-5)
        assert q_to_log10_ser(50.0) == pytest.approx(-544.966336, abs=1e-5)
        assert q_to_log10_ser(100.0) == pytest.approx(-2173.871543, abs=1e-4)

    def test_enob_uses_the_sine_rms_convention(self):
        """SNR = 6.02 N + 1.76 dB is defined for a full-scale SINE (rms = swing/(2 sqrt 2)).
        Feeding the PEAK-TO-PEAK swing ratio in directly overstates ENOB by exactly
        20 log10(2 sqrt 2)/6.02 = 1.5001 bits at every operating point."""
        kw = dict(rho_sp_W_Hz=1e-19, signal_lambda_m=1.55e-6, quantum_efficiency=0.8,
                  optical_bw_Hz=75e9, tia_current_noise_A_rtHz=12e-12, load_ohm=50.0)
        for loss in (0.0, 10.0, 25.0):
            r = link_performance(PamFormat(order=4, mean_power_W=1e-3), loss_dB=loss, **kw)
            want = (r.snr_elec_dB - 20.0 * math.log10(2.0 * math.sqrt(2.0)) - 1.76) / 6.02
            assert r.enob == pytest.approx(want, rel=1e-12)
            assert (r.snr_elec_dB - 1.76) / 6.02 - r.enob == pytest.approx(1.5001, abs=1e-4)

    def test_ser_to_ber_gray_coding(self):
        assert ser_to_ber(1e-3, 16) == pytest.approx(1e-3 / 4)
        assert ser_to_ber(1e-3, 16, gray_coded=False) > ser_to_ber(1e-3, 16)

    def test_format_validation(self):
        for bad in dict(order=3), dict(order=1), dict(extinction_ratio=1.0), dict(baud_hz=0.0):
            with pytest.raises(ValueError):
                PamFormat(**bad)


# ============================ F-7: the wall-plug layer ======================================

class TestF7WallPlug:
    def _solved(self):
        amp = FiberAmplifier(ER, FiberSpec(1.4e-6, 0.23, 1e25, 8.0), [Pump(0.3, 0.976e-6)],
                             [Signal(5e-3, 1.560e-6)], AseBand(1.52e-6, 1.575e-6, 16))
        return amp, amp.solve(n_nodes=161)

    def test_efficiency_chain_respects_every_ceiling(self):
        amp, r = self._solved()
        b = wall_plug_efficiency(amp, r, [PumpSource(0.45, 0.90)], aux_electrical_W=0.5)
        assert 0.0 < b.eta_optical_pce <= b.eta_stokes_limit
        # wall-plug cannot exceed the optical PCE times everything upstream of the fiber
        assert b.eta_wallplug <= b.eta_optical_pce * 0.45 * 0.90 + 1e-12
        assert b.eta_wallplug < b.eta_wallplug_gross          # added < output
        assert 0.0 < b.chain["eta_extraction"] <= 1.0 + 1e-9
        assert not b.notes

    def test_pce_agrees_with_the_metrics_module(self):
        from dynameta.optics.fiber_amp import power_conversion_efficiency
        amp, r = self._solved()
        b = wall_plug_efficiency(amp, r, [PumpSource(0.5)])
        assert b.eta_optical_pce == pytest.approx(power_conversion_efficiency(amp, r), rel=1e-15)

    def test_optical_power_balance_is_a_real_closure_and_not_an_identity(self):
        """RE-PINNED, because the previous form could not fail. Closed against `total_heat_W` the
        balance is X - X: total_heat_W IS F(0) - F(L), which expands to (launched - exiting) over
        the same two z-planes, so the residual sat at 5.6e-17 W whatever the solve contained --
        including a deliberately corrupted one. (Integrating heat_load_per_m does not help either:
        np.gradient followed by a trapezoid telescopes back to F(0) - F(L) exactly.)

        The balance now closes against the dissipation recomputed from the amplifier's own RATE
        EQUATIONS on the returned profile, which asks a different question -- is this P(z) actually
        a solution? -- and therefore has an answer that can be wrong. Measured on this fixture at
        n_nodes = 161: 1.183e-06 W of 0.305 W launched (3.88e-06 relative, pure trapezoid
        discretization, falling as O(dz^2): 1.5e-05 / 3.9e-06 / 6.0e-07 / 1.5e-07 at n_nodes =
        81 / 161 / 401 / 801)."""
        amp, r = self._solved()
        b = wall_plug_efficiency(amp, r, [PumpSource(0.45, 0.9)])
        launched = b.p_pump_launched_W + b.p_signal_in_W
        assert abs(b.energy_balance_residual_W) == pytest.approx(1.183e-06, rel=0.2)
        assert abs(b.energy_balance_residual_W) < 1e-5 * launched
        assert not b.notes
        # it really is a discretization floor, not a fixed offset: refine and it falls quadratically
        coarse = wall_plug_efficiency(amp, amp.solve(n_nodes=81), [PumpSource(0.45, 0.9)])
        fine = wall_plug_efficiency(amp, amp.solve(n_nodes=401), [PumpSource(0.45, 0.9)])
        assert abs(coarse.energy_balance_residual_W) > 3.0 * abs(b.energy_balance_residual_W)
        assert abs(fine.energy_balance_residual_W) < 0.5 * abs(b.energy_balance_residual_W)

    def test_a_corrupted_solve_breaks_the_power_balance(self):
        """DISCRIMINATION for the gate above: perturb ONE endpoint by 1% and the residual must
        jump. Measured 1.183e-06 W -> -1.746e-03 W, a factor 1476, and the note fires. Under the
        old total_heat_W form both numbers were 5.551e-17 W -- identical, because the perturbation
        cancels between `exiting` and the heat it is subtracted from."""
        import copy
        amp, r = self._solved()
        clean = wall_plug_efficiency(amp, r, [PumpSource(0.45, 0.9)])
        bad_r = copy.deepcopy(r)
        bad_r.power_W[bad_r.kind.index("signal"), -1] *= 1.01
        bad = wall_plug_efficiency(amp, bad_r, [PumpSource(0.45, 0.9)])
        assert abs(bad.energy_balance_residual_W) > 100.0 * abs(clean.energy_balance_residual_W)
        assert abs(bad.energy_balance_residual_W) == pytest.approx(1.746e-03, rel=0.2)
        assert any("power balance" in n for n in bad.notes), bad.notes
        # the old identity, reproduced here so its degeneracy is on the record rather than asserted
        from dynameta.optics.fiber_amp.thermal import total_heat_W
        for res in (r, bad_r):
            launched = 0.3 + float(res.power_W[res.kind.index("signal"), 0])
            exiting = float(np.sum([res.power_W[i, -1] if res.u[i] > 0 else res.power_W[i, 0]
                                    for i in range(res.power_W.shape[0])]))
            assert abs(launched - exiting - float(total_heat_W(res))) < 1e-15

    def test_electrical_power_is_the_diode_and_coupling_chain(self):
        amp, r = self._solved()
        b = wall_plug_efficiency(amp, r, [PumpSource(0.45, 0.90)], aux_electrical_W=0.25)
        assert b.p_electrical_pumps_W == pytest.approx(0.3 / (0.45 * 0.90), rel=1e-14)
        assert b.p_electrical_total_W == pytest.approx(b.p_electrical_pumps_W + 0.25, rel=1e-14)

    def test_duty_cycle_scales_delivered_power_not_electrical_cost(self):
        amp, r = self._solved()
        full = wall_plug_efficiency(amp, r, [PumpSource(0.45, 0.9)])
        half = wall_plug_efficiency(amp, r, [PumpSource(0.45, 0.9)], duty_cycle=0.5)
        assert half.p_electrical_total_W == pytest.approx(full.p_electrical_total_W, rel=1e-15)
        assert half.eta_wallplug == pytest.approx(0.5 * full.eta_wallplug, rel=1e-12)

    def test_multi_pump_uses_the_photon_weighted_ceiling(self):
        amp = FiberAmplifier(ER, FiberSpec(1.4e-6, 0.23, 1e25, 8.0),
                             [Pump(0.15, 0.976e-6), Pump(0.15, 1.480e-6)],
                             [Signal(5e-3, 1.560e-6)], AseBand(1.52e-6, 1.575e-6, 12))
        r = amp.solve(n_nodes=161)
        b = wall_plug_efficiency(amp, r, [PumpSource(0.45, 0.9), PumpSource(0.30, 0.9)])
        assert (stokes_limit(0.976e-6, 1.560e-6) < b.eta_stokes_limit
                < stokes_limit(1.480e-6, 1.560e-6))
        assert b.p_electrical_pumps_W == pytest.approx(0.15 / (0.45 * 0.9) + 0.15 / (0.30 * 0.9),
                                                       rel=1e-14)

    def test_energy_per_bit_and_per_port(self):
        amp, r = self._solved()
        b = wall_plug_efficiency(amp, r, [PumpSource(0.45, 0.9)], aux_electrical_W=0.5)
        e = energy_per_bit_J(b, 20e9)
        assert e == pytest.approx(b.p_electrical_total_W / 20e9, rel=1e-14)
        assert energy_per_bit_J(b, 20e9, ports=64) == pytest.approx(e / 64, rel=1e-14)

    def test_validation(self):
        amp, r = self._solved()
        with pytest.raises(ValueError, match="PumpSource"):
            wall_plug_efficiency(amp, r, [PumpSource(0.45), PumpSource(0.45)])
        with pytest.raises(ValueError, match="duty_cycle"):
            wall_plug_efficiency(amp, r, [PumpSource(0.45)], duty_cycle=0.0)
        for bad in dict(wallplug_efficiency=0.0), dict(wallplug_efficiency=1.5):
            with pytest.raises(ValueError, match="efficiency"):
                PumpSource(**bad)
