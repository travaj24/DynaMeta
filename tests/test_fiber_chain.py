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
