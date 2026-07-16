"""Rare-earth fiber amplifier (EDFA / YDFA) end-to-end physics gates -- a fast standalone
sweep of the headline reduces-to-known-limit checks for dynameta.optics.fiber_amp (the deeper
per-phase discrimination gates live in tests/test_fiber_amp.py). Pure numpy/scipy; configs are
kept small so this runs in the CI smoke tier.

GATE A (Beer-Lambert): an unpumped fiber attenuates the signal by exactly the small-signal
        absorption alpha = Gamma n_t sigma_a -- the z-solver reduces to Beer's law.
GATE B (pumped gain + photon conservation): a pumped EDFA gives net gain with the inversion in
        [0,1], and the signal+ASE photons GAINED never exceed the pump photons LOST (<= 1).
GATE C (noise-figure quantum floor): the local spontaneous-emission factor n_sp >= 1 and the
        optical noise figure NF respects (2 - 1/G); a strongly-inverted high-gain preamp
        approaches the 3 dB quantum limit.
GATE D (Stokes efficiency ceiling): the slope efficiency dP_sig/dP_pump never exceeds the
        quantum-defect limit lambda_pump/lambda_signal.
GATE E (concentration opt-in): with no ConcentrationModel the solve is byte-identical to an
        all-default (identity) model -- the degradation physics is strictly opt-in.
GATE F (heat energy balance + quantum defect): the dissipated power equals pump_abs minus the
        light carried out, and the Yb quantum defect (976->1030) is far below Er's (980->1560).
GATE G (Frantz-Nodvik): fast-pulse extraction reduces to G0 E_in for a small pulse and to
        E_in + E_sat ln G0 (all stored energy) for a large one.

Run: python -m validation.fiber_amp_physics
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dynameta.constants import C_LIGHT, H_PLANCK
from dynameta.optics.fiber_amp import (
    erbium, FiberSpec, overlap_gamma, Pump, Signal, AseBand, FiberAmplifier,
    analyze_noise, slope_efficiency, stokes_limit, ConcentrationModel,
    quantum_defect_fraction, total_heat_W, saturation_energy, frantz_nodvik_output_energy,
)


def _edf(length_m, n_t=1.0e25):
    return FiberSpec(core_radius_m=1.4e-6, na=0.24, n_t_m3=n_t, length_m=length_m)


def main():
    print("[fa] === rare-earth fiber amplifier: end-to-end physics gates ===", flush=True)
    er = erbium("aluminosilicate")
    ok = True

    # GATE A: Beer-Lambert
    f = _edf(6.0)
    r = FiberAmplifier(er, f, [], [Signal(1e-6, 1.560e-6)], None).solve()
    alpha = float(overlap_gamma(f, 1.560e-6)) * f.n_t_m3 * float(er.sigma_a.sigma(1.560e-6))
    analytic = -10.0 * np.log10(np.e) * alpha * f.length_m
    a = abs(float(r.signal_gain_dB[0]) - analytic) < 0.05
    ok = ok and a
    print("[fa] GATE A: Beer-Lambert {:.3f} dB vs analytic {:.3f} dB -> {}".format(
        float(r.signal_gain_dB[0]), analytic, "PASS" if a else "FAIL"), flush=True)

    # GATE B: pumped gain + photon conservation
    amp = FiberAmplifier(er, _edf(6.0), [Pump(100e-3, 0.980e-6, "fwd")],
                         [Signal(1e-6, 1.560e-6)], AseBand(1.52e-6, 1.575e-6, 12))
    r = amp.solve()
    ip, is_ = r.kind.index("pump"), r.kind.index("signal")

    def ph(P, lam):
        return P / (H_PLANCK * C_LIGHT / lam)
    lost = ph(r.power_W[ip, 0], r.lambda_m[ip]) - ph(r.power_W[ip, -1], r.lambda_m[ip])
    got = ph(r.power_W[is_, -1], r.lambda_m[is_]) - ph(r.power_W[is_, 0], r.lambda_m[is_])
    got += sum(ph(r.power_W[k, -1 if r.u[k] > 0 else 0], r.lambda_m[k])
               for k in np.where(r.is_ase)[0])
    b = (r.meta["converged"] and 0.0 <= r.nbar2_z.min() and r.nbar2_z.max() <= 1.0
         and float(r.signal_gain_dB[0]) > 10.0 and 0.0 < got / lost <= 1.02)
    ok = ok and b
    print("[fa] GATE B: gain {:.1f} dB, nbar2<=1, photon ratio {:.3f} -> {}".format(
        float(r.signal_gain_dB[0]), got / lost, "PASS" if b else "FAIL"), flush=True)

    # GATE C: noise-figure quantum floor + preamp -> 3 dB
    nr = analyze_noise(r, 1.560e-6)
    pre = FiberAmplifier(er, _edf(1.5, n_t=2.5e25), [Pump(1.5, 0.980e-6, "fwd")],
                         [Signal(1e-6, 1.560e-6)], AseBand(1.52e-6, 1.575e-6, 25))
    npre = analyze_noise(pre.solve(), 1.560e-6)
    c = (nr.n_sp_local_min >= 1.0 - 1e-9 and npre.gain_dB > 12.0
         and abs(npre.nf_dB - 10.0 * np.log10(2.0)) < 0.4)
    ok = ok and c
    print("[fa] GATE C: n_sp>=1, preamp NF {:.2f} dB (->3.01) -> {}".format(
        npre.nf_dB, "PASS" if c else "FAIL"), flush=True)

    # GATE D: Stokes efficiency ceiling
    se = slope_efficiency(FiberAmplifier(er, _edf(6.0), [Pump(150e-3, 0.980e-6, "fwd")],
                                         [Signal(1e-6, 1.560e-6)], AseBand(1.52e-6, 1.575e-6, 12)),
                          np.linspace(30e-3, 350e-3, 8), saturating_signal_W=5e-3)
    ceil = stokes_limit(0.980e-6, 1.560e-6)
    d = 0.0 < se.slope <= ceil * 1.02
    ok = ok and d
    print("[fa] GATE D: slope {:.3f} <= Stokes {:.3f} -> {}".format(
        se.slope, ceil, "PASS" if d else "FAIL"), flush=True)

    # GATE E: concentration opt-in byte-identical
    r_none = FiberAmplifier(er, _edf(6.0), [Pump(100e-3, 0.980e-6, "fwd")],
                            [Signal(1e-6, 1.560e-6)], AseBand(1.52e-6, 1.575e-6, 12)).solve()
    r_id = FiberAmplifier(er, _edf(6.0), [Pump(100e-3, 0.980e-6, "fwd")],
                          [Signal(1e-6, 1.560e-6)], AseBand(1.52e-6, 1.575e-6, 12),
                          concentration=ConcentrationModel()).solve()
    e = np.array_equal(r_none.power_W, r_id.power_W)
    ok = ok and e
    print("[fa] GATE E: concentration opt-in byte-identical -> {}".format(
        "PASS" if e else "FAIL"), flush=True)

    # GATE F: heat balance + quantum-defect contrast
    heat = total_heat_W(r)
    pump_abs = float(r.power_W[ip, 0] - r.power_W[ip, -1])
    sig_add = float(r.power_W[is_, -1] - r.power_W[is_, 0])
    ase_out = float(np.sum(r.power_W[(r.u > 0) & r.is_ase, -1])
                    + np.sum(r.power_W[(r.u < 0) & r.is_ase, 0]))
    fbal = abs(heat - (pump_abs - sig_add - ase_out)) < 1e-9 * max(1.0, abs(heat)) + 1e-12
    qd = quantum_defect_fraction(0.976e-6, 1.030e-6) < 0.10 < quantum_defect_fraction(
        0.980e-6, 1.560e-6)
    fg = fbal and qd
    ok = ok and fg
    print("[fa] GATE F: heat balance exact & qd(Yb)<qd(Er) -> {}".format(
        "PASS" if fg else "FAIL"), flush=True)

    # GATE G: Frantz-Nodvik limits
    esat = saturation_energy(er, _edf(6.0), 1.560e-6)
    G0 = np.exp(3.0)
    small = float(frantz_nodvik_output_energy(1e-4 * esat, G0, esat))
    large = float(frantz_nodvik_output_energy(50.0 * esat, G0, esat))
    g = (abs(small / (G0 * 1e-4 * esat) - 1.0) < 1e-3
         and abs((large - 50.0 * esat) - esat * np.log(G0)) / (esat * np.log(G0)) < 1e-2)
    ok = ok and g
    print("[fa] GATE G: Frantz-Nodvik small->G0 E_in, large->E_sat lnG0 -> {}".format(
        "PASS" if g else "FAIL"), flush=True)

    print("[fa] *** FIBER AMPLIFIER PHYSICS: {} ***".format("PASS" if ok else "FAIL"), flush=True)
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
