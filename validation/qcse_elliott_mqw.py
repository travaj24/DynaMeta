"""
QCSE depth follow-ons (roadmap Phase-3 REMAINING): the Elliott band-to-band CONTINUUM on top of the
single exciton line, and a MULTI-quantum-well (MQW) stack. Pure numpy/scipy, no FEM.

GATE A (Elliott continuum): core.effects.ElectroAbsorptionModel with continuum_alpha0_per_m > 0 adds a
        step joint-DOS above the unbound edge E_cont = E_T + E_binding with the 2D Sommerfeld factor
        S_2D -> 2 at the edge and -> 1 far above. So the continuum absorption is ~2*alpha0_c just above
        E_cont and ~alpha0_c far above; and continuum_alpha0_per_m = 0 reproduces the pure-exciton
        model exactly.
GATE B (MQW): carriers.qcse.MultiQuantumWell reduces EXACTLY to QuantumWell at n_wells = 1; a THICK
        barrier (uncoupled) keeps the ground subband at the single-well E_1; a THIN barrier (coupled)
        lowers it (miniband hybridization). All grounds stay in-well (not field-ionized) at F = 0.

Run: python -m validation.qcse_elliott_mqw
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dynameta.carriers.qcse import QuantumWell, MultiQuantumWell
from dynameta.core.effects import ElectroAbsorptionModel

EV = 1.602176634e-19
M0 = 9.1093837015e-31
GAAS = dict(well_width_m=10e-9, barrier_e_J=0.20 * EV, barrier_h_J=0.10 * EV,
            m_e_kg=0.067 * M0, m_h_kg=0.45 * M0, E_g_J=1.42 * EV)
E_BIND = 10e-3 * EV
SIGMA = 2e-3 * EV
A0_EXC, A0_CONT = 1.0e7, 5.0e6


def main():
    print("[qe] === QCSE Elliott continuum + MQW ===", flush=True)
    qw = QuantumWell(exciton_binding_J=E_BIND, **GAAS)
    E_T = qw.solve(0.0).E_transition_J
    ov = qw.solve(0.0).overlap
    e_lo, e_hi = E_T - 12 * SIGMA, E_T + E_BIND + 80e-3 * EV
    eam = ElectroAbsorptionModel(qw, eps_bg=complex(12.0, 0.02), alpha0_per_m=A0_EXC,
                                 broadening_J=SIGMA, e_grid_J=(e_lo, e_hi, 4001),
                                 continuum_alpha0_per_m=A0_CONT, continuum_binding_J=E_BIND)

    # GATE A: the continuum reproduces the closed-form 2D Sommerfeld factor S_2D(dE) =
    # 2/(1+exp(-2 pi sqrt(E_b/dE))) above the unbound edge E_cont = E_T + E_b -> 2 at the edge,
    # decaying SLOWLY toward 1 (S_2D(6 E_b) ~ 1.86, not ~1: the factor decays as ~1/sqrt(dE)). The
    # exciton Gaussian is ~e^-12 negligible this far above E_T, so a(E) ~ A0_CONT * S_2D.
    E_cont = E_T + E_BIND

    def s2d(dE):
        return 2.0 / (1.0 + np.exp(-2.0 * np.pi * np.sqrt(E_BIND / dE)))
    eam_off = ElectroAbsorptionModel(qw, eps_bg=complex(12.0, 0.02), alpha0_per_m=A0_EXC,
                                     broadening_J=SIGMA, e_grid_J=(e_lo, e_hi, 4001))
    form_ok = True
    samples = []
    for dE in (0.05e-3 * EV, 20e-3 * EV, 60e-3 * EV):
        a = float(eam._alpha(E_cont + dE, E_T, ov, ov))
        a_pred = A0_CONT * s2d(dE)                                   # model vs closed-form Sommerfeld
        form_ok = form_ok and (abs(a - a_pred) < 1e-2 * A0_CONT + 1e-3 * A0_EXC)
        samples.append((dE / EV * 1e3, a, a_pred))
    a_edge = float(eam._alpha(E_cont + 0.02e-3 * EV, E_T, ov, ov))
    edge_two = abs(a_edge - 2.0 * A0_CONT) < 0.02 * A0_CONT          # edge enhancement = 2x
    a_below = float(eam._alpha(E_cont - 3e-3 * EV, E_T, ov, ov))     # below edge: continuum OFF
    below_ok = abs(a_below - float(eam_off._alpha(E_cont - 3e-3 * EV, E_T, ov, ov))) < 1e-6 * A0_CONT
    off_ok = abs(float(eam_off._alpha(E_cont + 60e-3 * EV, E_T, ov, ov))) < 1e-3 * A0_CONT
    gate_a = bool(form_ok and edge_two and below_ok and off_ok)
    for dEmeV, a, ap in samples:
        print("[qe] continuum dE={:5.2f} meV: a={:.4e}  S2D-pred={:.4e}".format(dEmeV, a, ap),
              flush=True)
    print("[qe] edge a={:.3e} (2*a0c={:.3e}); below-edge continuum off={}; off-model={:.2e}".format(
        a_edge, 2 * A0_CONT, below_ok, float(eam_off._alpha(E_cont + 60e-3 * EV, E_T, ov, ov))),
        flush=True)
    print("[qe] GATE A (continuum == closed-form 2D Sommerfeld; edge=2x; off below/disabled): "
          "{}".format("PASS" if gate_a else "FAIL"), flush=True)

    # GATE B: MQW reduction + coupling
    mqw1 = MultiQuantumWell(n_wells=1, barrier_width_m=0.0, exciton_binding_J=E_BIND, **GAAS)
    d1 = abs(mqw1.solve(0.0).E_transition_J - E_T)
    e1 = qw.solve(0.0).E_e1_J
    mqw_thick = MultiQuantumWell(n_wells=3, barrier_width_m=12e-9, nz=2601,
                                 exciton_binding_J=E_BIND, **GAAS)
    thick = mqw_thick.solve(0.0)
    thin = MultiQuantumWell(n_wells=3, barrier_width_m=1.5e-9, nz=2601,
                            exciton_binding_J=E_BIND, **GAAS).solve(0.0)
    red_ok = bool(d1 < 1e-6 * EV)
    thick_ok = bool(abs(thick.E_e1_J - e1) < 2e-3 * EV and not thick.ionized)
    thin_ok = bool(thin.E_e1_J < e1 - 1e-3 * EV and not thin.ionized)
    gate_b = bool(red_ok and thick_ok and thin_ok)
    print("[qe] MQW: n=1 dE_T={:.2e} eV ; E_e1 single={:.4f} thick(12nm)={:.4f} thin(1.5nm)={:.4f} "
          "eV".format(d1 / EV, e1 / EV, thick.E_e1_J / EV, thin.E_e1_J / EV), flush=True)
    print("[qe] GATE B (MQW n=1 == single well; thick uncoupled ~E_1; thin coupled drops): {}".format(
        "PASS" if gate_b else "FAIL"), flush=True)

    # GATE C (overlap pairing regression): an uncoupled multi-well stack hosts an N-fold near-degenerate
    # ground manifold; the e and h must be paired into the SAME well, so the MQW e-h overlap equals the
    # single-well value -- NOT the spurious ~0 a per-carrier lowest-index pick gives when fp symmetry-
    # breaking localizes them in different wells (audit Finding 3).
    ov_sw = qw.solve(0.0).overlap
    ov_mqw = thick.overlap
    gate_c = bool(abs(ov_mqw - ov_sw) < 0.05 and ov_mqw > 0.5)
    print("[qe] MQW overlap (F=0, uncoupled): single={:.4f} 3-well={:.4f}  (must match, not ~0): {}".format(
        ov_sw, ov_mqw, "PASS" if gate_c else "FAIL"), flush=True)

    overall = gate_a and gate_b and gate_c
    print("[qe] *** QCSE ELLIOTT CONTINUUM + MQW: {} ***".format("PASS" if overall else "FAIL"),
          flush=True)
    return overall


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
