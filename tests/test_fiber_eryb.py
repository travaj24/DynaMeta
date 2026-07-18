"""Discrimination-proven physics gates for the Er:Yb co-doped fiber amplifier
(dynameta.optics.fiber_amp.eryb.ErYbAmplifier), the sensitized counterpart of the single-ion
FiberAmplifier. Each test is a falsifiable gate kept small so the suite runs fast (< ~60 s).
Bidirectional-adversarial: the sensitization / transfer-efficiency / Yb-clamp claims are proven
out, and the dossier ">95% transfer" figure is NOT forced -- the model reports the actual
value, which the parameters put at ~0.85 (see eryb.py DISCREPANCY NOTE and gate 3).

ErYbAmplifier is not re-exported from the package __init__, so it is imported from its module."""

import numpy as np

from dynameta.optics.fiber_amp import (
    erbium, ytterbium, FiberSpec, Pump, Signal, AseBand, FiberAmplifier,
)
from dynameta.optics.fiber_amp.eryb import ErYbAmplifier

ER = erbium("aluminosilicate")
YB = ytterbium("phosphosilicate")               # phospho Yb: tau_Yb = 1.45 ms (dossier k_tr host)

# analytic low-inversion transfer efficiency (module derivation): the acceptors of a Yb
# excitation are Er GROUND ions, so the per-Yb* transfer rate is k_tr n1 ~ k_tr N_Er.
_TAU_YB = YB.tau_s                                # 1.45e-3 s


def _eta_analytic(n_er, k_tr, tau_yb=_TAU_YB):
    x = k_tr * n_er * tau_yb
    return x / (1.0 + x)


# ============================ Gate 1: Er-only limit (the anchor) ============================

def test_er_only_limit_matches_plain_edfa():
    # k_tr = 0 and a vanishing Yb density must reproduce a plain FiberAmplifier with the same
    # Er / fiber / channels: the EYDFA collapses to the two-level EDFA algebra.
    fib = FiberSpec(1.4e-6, 0.24, 1.0e25, 6.0)
    pumps = [Pump(100e-3, 0.980e-6, "fwd")]
    signals = [Signal(1e-6, 1.560e-6)]
    ase = AseBand(1.52e-6, 1.575e-6, 8)
    g_plain = float(FiberAmplifier(ER, fib, pumps, signals, ase).solve(n_nodes=81)
                    .signal_gain_dB[0])
    r = ErYbAmplifier(ER, YB, fib, pumps, signals, ase, n_yb_m3=1e10, k_tr_m3_s=0.0
                      ).solve(n_nodes=81)
    assert r.meta["converged"]
    assert abs(float(r.signal_gain_dB[0]) - g_plain) < 0.05


# ============================ Gate 2: Yb sensitization ============================

def test_yb_sensitization_raises_1550_gain():
    # a 976 nm cladding-pumped Er fiber has weak DIRECT Er pump absorption (tiny overlap) -> the
    # Er-only case is ~transparent; adding Yb (huge 976 sigma) soaks up the pump and transfers it,
    # inverting Er strongly -> the 1550 gain jumps by tens of dB.
    fib = FiberSpec(3.0e-6, 0.20, 1.0e25, 4.0, clad_radius_m=62.5e-6)
    pumps = [Pump(1.0, 0.976e-6, "fwd", cladding=True)]
    signals = [Signal(1e-4, 1.550e-6)]
    g_er = float(ErYbAmplifier(ER, YB, fib, pumps, signals, None, n_yb_m3=1e10, k_tr_m3_s=0.0
                               ).solve(n_nodes=81).signal_gain_dB[0])
    r_yb = ErYbAmplifier(ER, YB, fib, pumps, signals, None, n_yb_m3=8e25, k_tr_m3_s=2e-22
                         ).solve(n_nodes=81)
    assert r_yb.meta["converged"]
    assert float(r_yb.signal_gain_dB[0]) > g_er + 10.0


# ============================ Gate 3: transfer efficiency ============================

def _eta_model(n_er, pump_W=0.05, k_tr=2e-22):
    fib = FiberSpec(3.0e-6, 0.20, n_er, 2.5, clad_radius_m=125e-6)
    amp = ErYbAmplifier(ER, YB, fib, [Pump(pump_W, 0.976e-6, "fwd", cladding=True)],
                        [Signal(1e-6, 1.550e-6)], None, n_yb_m3=8e25, k_tr_m3_s=k_tr)
    r = amp.solve(n_nodes=81)
    assert r.meta["converged"]
    return amp.transfer_efficiency(r)


def test_transfer_efficiency_matches_low_power_form_and_monotone():
    # k_tr N_Er tau_Yb = 2e-22 * 2e25 * 1.45e-3 = 5.8 -> analytic eta ~ 0.85 (NOT the dossier's
    # >0.95; the model is NOT forced to it -- see eryb.py DISCREPANCY NOTE). At low inversion the
    # model must land within 15% of the analytic branching-ratio form.
    n_er = 2e25
    ana = _eta_analytic(n_er, 2e-22)
    assert abs(ana - 0.853) < 0.01                       # sanity: the "=5.8 -> 0.85" number
    eta = _eta_model(n_er)
    assert abs(eta - ana) / ana < 0.15
    # eta_tr rises monotonically with the Er (acceptor) density
    etas = [_eta_model(n) for n in (1e25, 2e25, 4e25)]
    assert np.all(np.diff(etas) > 0.0)


# ============================ Gate 4: Yb inversion clamp ============================

def _beta_max(n_er, k_tr, pump_W=3.0):
    fib = FiberSpec(3.0e-6, 0.20, n_er, 4.0, clad_radius_m=125e-6)
    amp = ErYbAmplifier(ER, YB, fib, [Pump(pump_W, 0.976e-6, "fwd", cladding=True)],
                        [Signal(50e-3, 1.550e-6)], None, n_yb_m3=8e25, k_tr_m3_s=k_tr)
    r = amp.solve(n_nodes=81)
    assert r.meta["converged"]
    return float(r.meta["beta_yb_z"].max())


def test_yb_inversion_clamped_and_drained_by_transfer():
    # a saturating 1550 signal keeps the Er ground populated, so the transfer drain
    # k_tr N_Er (1 - f2) stays finite: at a few W the Yb inversion is clamped well below 0.15,
    # and it DROPS further when k_tr or N_Er increases (more drain on the Yb reservoir).
    base = _beta_max(2e25, 2e-22)
    assert base < 0.15
    assert _beta_max(2e25, 4e-22) < base                 # more transfer drains the Yb
    assert _beta_max(3e25, 2e-22) < base                 # more acceptors drain the Yb


# ============================ Gate 5: photon / energy bookkeeping ============================

def _multi_w_device(length_m=6.0):
    fib = FiberSpec(3.2e-6, 0.20, 3e25, length_m, clad_radius_m=125e-6)
    return ErYbAmplifier(ER, YB, fib, [Pump(10.0, 0.975e-6, "fwd", cladding=True)],
                         [Signal(100e-3, 1.550e-6)], AseBand(1.53e-6, 1.565e-6, 8),
                         n_yb_m3=2.5e26, k_tr_m3_s=4e-22, yb_ase=AseBand(1.0e-6, 1.08e-6, 6))


def test_energy_bookkeeping_output_below_input_and_heat_positive():
    amp = _multi_w_device()
    r = amp.solve(n_nodes=101)
    assert r.meta["converged"]
    ip, isg = r.kind.index("pump"), r.kind.index("signal")
    tot_in = float(r.power_W[ip, 0] + r.power_W[isg, 0])
    tot_out = float(np.sum(r.power_W[r.u > 0, -1]) + np.sum(r.power_W[r.u < 0, 0]))
    assert tot_out <= tot_in + 1e-9                       # optics out <= optics in (heat is lost)
    pump_abs = float(r.power_W[ip, 0] - r.power_W[ip, -1])
    sig_add = float(r.power_W[isg, -1] - r.power_W[isg, 0])
    ase_out = float(np.sum(r.power_W[(r.u > 0) & r.is_ase, -1])
                    + np.sum(r.power_W[(r.u < 0) & r.is_ase, 0]))
    heat = pump_abs - sig_add - ase_out
    assert heat >= -1e-9                                  # absorbed pump - (gain + ASE) = heat >= 0


# ============================ Gate 6: representative multi-W PCE ============================

def test_multi_watt_pce_and_gain_in_band():
    # 10 W 975 nm cladding pump, 100 mW 1550 seed, L = 6 m: a representative EYDFA. Power
    # conversion efficiency (P_sig_out - P_sig_in)/P_pump in [0.15, 0.45] (dossier device band
    # 25-40%), and C-band gain > 15 dB.
    amp = _multi_w_device()
    r = amp.solve(n_nodes=101)
    assert r.meta["converged"]
    isg = r.kind.index("signal")
    pce = float(r.power_W[isg, -1] - r.power_W[isg, 0]) / 10.0
    assert 0.15 <= pce <= 0.45
    assert float(r.signal_gain_dB[0]) > 15.0


# ============================ Gate 7: convergence flag ============================

def test_convergence_flag_set_across_regimes():
    # every representative regime must report converged=True (the coupled f2/b2 algebra is a
    # bracketed scalar solve, so the relaxation cannot silently fail).
    fib = FiberSpec(3.0e-6, 0.20, 2e25, 4.0, clad_radius_m=125e-6)
    for pumps in ([Pump(3.0, 0.976e-6, "fwd", cladding=True)],
                  [Pump(3.0, 0.976e-6, "bwd", cladding=True)],
                  [Pump(1.5, 0.976e-6, "fwd", cladding=True),
                   Pump(1.5, 0.976e-6, "bwd", cladding=True)]):
        r = ErYbAmplifier(ER, YB, fib, pumps, [Signal(20e-3, 1.550e-6)],
                          AseBand(1.53e-6, 1.565e-6, 6), n_yb_m3=8e25, k_tr_m3_s=2e-22
                          ).solve(n_nodes=81)
        assert r.meta["converged"]
        assert 0.0 <= float(r.nbar2_z.min()) and float(r.nbar2_z.max()) <= 1.0
        assert 0.0 <= float(r.meta["beta_yb_z"].min())
        assert float(r.meta["beta_yb_z"].max()) <= 1.0


# ============================ Diagnostics: parasitic 1-um gain ============================

def test_yb_parasitic_gain_tracks_inversion():
    # yb_parasitic_gain_dB is computed from the b2 profile even with no yb_ase channels; draining
    # the Yb (large k_tr) lowers both the Yb inversion and the 1030 nm parasitic gain.
    fib = FiberSpec(3.0e-6, 0.20, 2e25, 4.0, clad_radius_m=125e-6)

    def run(k_tr):
        amp = ErYbAmplifier(ER, YB, fib, [Pump(3.0, 0.976e-6, "fwd", cladding=True)],
                            [Signal(20e-3, 1.550e-6)], None, n_yb_m3=8e25, k_tr_m3_s=k_tr)
        r = amp.solve(n_nodes=81)
        assert r.meta["converged"]
        return float(r.meta["beta_yb_z"].max()), amp.yb_parasitic_gain_dB(r)

    beta_lo, par_lo = run(5e-22)                          # strong transfer -> low Yb inversion
    beta_hi, par_hi = run(2e-23)                          # weak transfer -> Yb builds up
    assert beta_hi > beta_lo
    assert par_hi > par_lo
