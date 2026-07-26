"""Amplifier-chain gates: the PSD-based cascade must REPRODUCE the classic rules (Friis, the
attenuator asymmetry) rather than assume them, and PDG helpers must pin the Mazurczyk-Zyskind
anchor. Independent oracles: the Friis formula evaluated from the per-stage (G, NF) records,
and the single-amp chain vs the direct solve."""

import numpy as np
import pytest

from dynameta.optics.fiber_amp.chain import AmplifierChain, PassiveElement
from dynameta.optics.fiber_amp.polarization import (TwoPolSaturation, f_from_pdg_slope,
                                                    pdg_cascade_db, pdg_db)
from dynameta.optics.fiber_amp.spectroscopy import erbium
from dynameta.optics.fiber_amp.steady_state import AseBand, FiberAmplifier, Pump, Signal
from dynameta.optics.fiber_amp.waveguide import FiberSpec

LAM_S = 1.55e-6


def _edfa(pump_W=0.25, L=8.0):
    fib = FiberSpec(core_radius_m=1.6e-6, na=0.22, n_t_m3=8.0e24, length_m=L)
    return FiberAmplifier(erbium(), fib, [Pump(pump_W, 0.98e-6)], [Signal(1e-5, LAM_S)],
                          AseBand(1.50e-6, 1.60e-6, n_bins=24))


def test_single_stage_chain_matches_direct_solve():
    from dynameta.optics.fiber_amp.noise import analyze_noise
    amp = _edfa()
    P_in = 1e-5
    ch = AmplifierChain([_edfa()]).solve(P_in, LAM_S, n_nodes=121)
    direct = amp.solve(n_nodes=121)
    nr = analyze_noise(direct, LAM_S)
    assert abs(ch.gain_total_dB - nr.gain_dB) < 1e-6
    assert abs(ch.nf_total_dB - nr.nf_dB) < 1e-6
    assert abs(ch.osnr_dB - nr.osnr_dB) < 1e-6


def test_two_stage_cascade_obeys_friis():
    ch = AmplifierChain([_edfa(0.25), PassiveElement("iso", 0.5), _edfa(0.10, L=6.0)])
    r = ch.solve(1e-5, LAM_S, n_nodes=121)
    amps = [s for s in r.stages if s.kind == "amp"]
    iso = [s for s in r.stages if s.kind == "passive"][0]
    G1 = 10.0 ** (amps[0].gain_dB / 10.0)
    t = 10.0 ** (iso.gain_dB / 10.0)
    F1 = 10.0 ** (amps[0].nf_stage_dB / 10.0)
    F2 = 10.0 ** (amps[1].nf_stage_dB / 10.0)
    # Friis with the mid-stage attenuator folded in: F = F1 + (1/t - 1)/G1 + (F2 - 1)/(G1 t)
    F_friis = F1 + (1.0 / t - 1.0) / G1 + (F2 - 1.0) / (G1 * t)
    F_chain = 10.0 ** (r.nf_total_dB / 10.0)
    assert abs(F_chain / F_friis - 1.0) < 0.05, (F_chain, F_friis)
    # high first-stage gain -> the chain NF sits close to stage 1's
    assert r.nf_total_dB < amps[0].nf_stage_dB + 1.0


def test_attenuator_asymmetry_pre_vs_post():
    loss = PassiveElement("att", 3.0)
    pre = AmplifierChain([loss, _edfa()]).solve(1e-5, LAM_S, n_nodes=121)
    post = AmplifierChain([_edfa(), loss]).solve(1e-5, LAM_S, n_nodes=121)
    # pre-amp loss adds ~dB-for-dB to NF; post-amp loss is nearly free
    assert pre.nf_total_dB > post.nf_total_dB + 2.0
    single = AmplifierChain([_edfa()]).solve(1e-5, LAM_S, n_nodes=121)
    assert abs(post.nf_total_dB - single.nf_total_dB) < 0.3
    # note: the pre-loss amp sees a weaker input (deeper inversion, slightly better NF), so
    # gate the SHIFT against the ideal +3 dB with a tolerant band
    assert 2.0 < pre.nf_total_dB - single.nf_total_dB < 4.0


def test_chain_records_are_consistent():
    ch = AmplifierChain([_edfa(), PassiveElement("gff", 1.5, ase_transmission=0.5), _edfa(0.08)])
    r = ch.solve(2e-5, LAM_S, n_nodes=101)
    g_sum = sum(s.gain_dB for s in r.stages)
    assert abs(g_sum - r.gain_total_dB) < 1e-9
    assert r.P_out_W == pytest.approx(2e-5 * 10.0 ** (r.gain_total_dB / 10.0), rel=1e-9)
    assert all(s.meta.get("converged", True) for s in r.stages if s.kind == "amp")


def test_chain_nf_invariant_under_reference_bandwidth():
    # audit A-2: the chain asked analyze_noise for the ASE reference power at ITS 0.1 nm default
    # and then inverted that power with the CALLER's bandwidth, so every stage's generated PSD
    # picked up a spurious 0.1/ref_bw_nm. The end-to-end NF is a property of the amplifier, not of
    # the bookkeeping bandwidth an OSNR happens to be quoted in, so it must be invariant -- and
    # the un-forwarded version walked 6.78 -> -9.23 dB over 0.05..2.0 nm, i.e. straight through
    # the quantum limit to an implied n_sp of 0.06. OSNR, which IS a per-bandwidth quantity, must
    # still move dB-for-dB with the bandwidth ratio.
    ch = AmplifierChain([_edfa(0.25), PassiveElement("iso", 0.5), _edfa(0.10, L=6.0)])
    rows = {bw: ch.solve(1e-5, LAM_S, ref_bw_nm=bw, n_nodes=121) for bw in (0.05, 0.1, 1.0)}
    ref = rows[0.1]
    for bw, r in rows.items():
        assert r.nf_total_dB == pytest.approx(ref.nf_total_dB, abs=1e-6), bw
        assert r.gain_total_dB == pytest.approx(ref.gain_total_dB, abs=1e-9)
        assert r.rho_out_1pol_W_Hz == pytest.approx(ref.rho_out_1pol_W_Hz, rel=1e-12)
        # implied end-to-end n_sp stays above the quantum limit
        G = 10.0 ** (r.gain_total_dB / 10.0)
        F = 10.0 ** (r.nf_total_dB / 10.0)
        assert (F * G - 1.0) / (2.0 * (G - 1.0)) >= 1.0, bw
    assert rows[0.1].osnr_dB - rows[1.0].osnr_dB == pytest.approx(10.0, abs=1e-6)
    assert rows[0.05].osnr_dB - rows[0.1].osnr_dB == pytest.approx(3.0103, abs=1e-3)


def test_chain_holds_an_eryb_stage():
    # audit A-3: the class docstring advertises ErYbAmplifier stages, but the chain re-seeded
    # every stage through metrics._set_signal, which unconditionally rebuilt a FiberAmplifier from
    # amp.ion -> AttributeError. The type-preserving with_signals protocol keeps the stage's own
    # class (and its Yb sensitizer), so the EYDFA booster must run end-to-end with a finite NF
    # above the 3 dB quantum floor and a gain that tracks its standalone solve.
    from dynameta.optics.fiber_amp.eryb import ErYbAmplifier
    from dynameta.optics.fiber_amp.spectroscopy import ytterbium
    fib = FiberSpec(core_radius_m=3.0e-6, na=0.20, n_t_m3=2.0e25, length_m=3.0,
                    clad_radius_m=50e-6)

    def eydfa(P_sig=1e-3):
        return ErYbAmplifier(erbium(), ytterbium(), fib,
                             [Pump(4.0, 0.976e-6, "fwd", cladding=True)], [Signal(P_sig, LAM_S)],
                             AseBand(1.50e-6, 1.60e-6, n_bins=16), n_yb_m3=4.0e26)

    r = AmplifierChain([_edfa(0.25), PassiveElement("iso", 0.5), eydfa()]).solve(1e-5, LAM_S,
                                                                                n_nodes=121)
    ey = [s for s in r.stages if s.kind == "amp"][-1]
    assert ey.name == "ErYbAmplifier"                       # the stage kept its OWN class
    assert np.isfinite(ey.nf_stage_dB) and 3.0 < ey.nf_stage_dB < 10.0
    assert np.isfinite(r.nf_total_dB) and 3.0 < r.nf_total_dB < 10.0
    assert r.gain_total_dB == pytest.approx(sum(s.gain_dB for s in r.stages), abs=1e-9)
    # the chain re-seeds the booster at ITS actual input power: same power, same standalone solve
    standalone = eydfa(ey.P_in_W).solve(n_nodes=121)
    assert float(standalone.signal_gain_dB[0]) == pytest.approx(ey.gain_dB, abs=1e-9)
    # ... and a single-stage ErYb chain reproduces its own noise analysis
    from dynameta.optics.fiber_amp.noise import analyze_noise
    solo = AmplifierChain([eydfa()]).solve(1e-3, LAM_S, n_nodes=121)
    nr = analyze_noise(eydfa().solve(n_nodes=121), LAM_S)
    assert solo.nf_total_dB == pytest.approx(nr.nf_dB, abs=1e-9)


# ---- PDG (polarization.py) -----------------------------------------------------------------

def test_pdg_anchor_and_cascade():
    assert pdg_db(3.0) == pytest.approx(0.078, abs=1e-9)          # ~0.08 dB at 3 dB compression
    assert pdg_cascade_db(0.08, 16) == pytest.approx(0.32)        # sqrt(N) random walk
    assert pdg_cascade_db(0.08, 16, aligned=True) == pytest.approx(1.28)


def test_two_pol_model_limits():
    m = TwoPolSaturation(g0_dB=30.0, P_sat_W=10e-3, f=f_from_pdg_slope())
    # small-compression signal-dominated limit: PDG/DeltaG -> (1 - f) = eps
    P = 0.4e-3
    ratio = m.pdg_dB(P) / m.compression_dB(P)
    assert abs(ratio - 0.026) < 0.004
    # the orthogonal polarization wins (positive PDG), monotone in drive
    assert 0.0 < m.pdg_dB(0.2e-3) < m.pdg_dB(2e-3)
    # unpolarized ASE alone produces NO PDG but does compress
    assert m.pdg_dB(0.0, P_ase_W=5e-3) == pytest.approx(0.0, abs=1e-12)
    assert m.compression_dB(0.0, P_ase_W=5e-3) > 1.0
    # deep saturation lands in the measured 0.2-0.4 dB band
    deep = None
    for P in np.linspace(1e-3, 50e-3, 200):
        if m.compression_dB(P) >= 9.0:
            deep = m.pdg_dB(P)
            break
    assert deep is not None and 0.15 < deep < 0.45


def test_chain_stage_without_with_signals_raises_a_protocol_error():
    """audit A-3 follow-on: the docstring advertised a "legacy FiberAmplifier-shaped rebuild" for
    a duck-typed stage that exposes .signals but not with_signals. That fallback never worked --
    it routed into metrics._set_signal, which reached for the FiberAmplifier-private `_clone` and
    produced `AttributeError: 'DuckAmp' object has no attribute '_clone'`. The contract is now
    TRUE: with_signals is REQUIRED and its absence raises a TypeError naming it, while a stage
    with NO .signals at all really does pass through unchanged as a pre-configured element."""
    class DuckAmp:                       # .signals + .solve(), no with_signals
        name = "duck"

        def __init__(self, inner):
            self.inner = inner
            self.signals = list(inner.signals)

        def solve(self, **kw):
            return self.inner.with_signals(self.signals).solve(**kw)

    with pytest.raises(TypeError, match="with_signals"):
        AmplifierChain([DuckAmp(_edfa())]).solve(1e-5, LAM_S, n_nodes=81)

    class PreConfigured:                 # no .signals -> pass-through, solved as configured
        name = "pre"

        def __init__(self, inner):
            self.inner = inner

        def solve(self, **kw):
            return self.inner.solve(**kw)

    r = AmplifierChain([PreConfigured(_edfa())]).solve(1e-5, LAM_S, n_nodes=81)
    direct = _edfa().solve(n_nodes=81)
    assert r.gain_total_dB == pytest.approx(float(direct.signal_gain_dB[0]), abs=1e-9)


def test_metrics_name_the_missing_protocol_method_instead_of_leaking_a_private_attribute():
    """The same contract on the metrics side: a third-party amplifier gets a TypeError naming the
    protocol method it is missing, not a bare AttributeError about a private `_clone`."""
    from dynameta.optics.fiber_amp import metrics as M

    class DuckAmp:
        """Implements as much of the protocol as `implements` says, and nothing else. The
        implemented methods return a DuckAmp (a real amplifier would short-circuit the test by
        supplying the rest of the protocol from the next clone on)."""

        name = "duck"

        def __init__(self, inner, implements=()):
            self.inner = inner
            self.signals = list(inner.signals)
            self.pumps = list(inner.pumps)
            self._impl = tuple(implements)
            if "with_signals" in self._impl:
                self.with_signals = lambda s: DuckAmp(inner.with_signals(s), self._impl)
            if "with_pumps" in self._impl:
                self.with_pumps = lambda p: DuckAmp(inner.with_pumps(p), self._impl)

        def solve(self, **kw):
            return self.inner.solve(**kw)

    bare = DuckAmp(_edfa())
    with_sig = DuckAmp(_edfa(), implements=("with_signals",))
    # each metric names the FIRST protocol method it actually needs and cannot find
    cases = [(bare, lambda a: M.gain_compression_curve(a, [1e-5]), r"^DuckAmp does not implement "
              r"with_signals\(\)"),
             (with_sig, lambda a: M.slope_efficiency(a, [0.1, 0.2]),
              r"^DuckAmp does not implement with_pumps\(\)"),
             (with_sig, lambda a: M.gain_spectrum(a, [LAM_S]),
              r"^DuckAmp does not implement without_ase\(\)")]
    for amp, call, pattern in cases:
        with pytest.raises(TypeError, match=pattern):
            call(amp)
        try:
            call(amp)
        except TypeError as e:
            assert "_clone" not in str(e)                 # no private attribute leaks out
            assert "with_signals(signals)" in str(e)      # ... the whole contract is spelled out
