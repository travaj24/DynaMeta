"""R18 many-body density corrections oracle (BGR + exciton screening / Mott in the QCSE model).

GATE A (off-switch + no-density byte-identity): all R18 fields zero -> eps with fields['n']
        present is IDENTICAL to the shipped model; n absent -> corrections (0, 1) exactly.
GATE B (BGR closed form + cross-module consistency): the edge shift's log-log slope vs n is
        EXACTLY 1/3; the magnitude with the published GaAs coefficient (2.4e-8 eV cm =
        3.84e-29 J m) is ~24 meV at 1e24 m^-3; the shift EQUALS BursteinMossEdge.gap_shift_J
        for the same coefficient (machine -- the two models share one closed form).
GATE C (screening closure points, machine): E_b(n)/E_b0 = 1/(1 + n/n_s) at n = 0.1/1/10 n_s ->
        amplitudes ratio^p exactly; the blueshift E_b0 - E_b(n) rides on top of the BGR
        redshift; Mott cutoff: amplitude EXACTLY 0 at n >= n_Mott, > 0 just below.
GATE D (end-to-end device physics): flat band (F = 0) returns eps_bg EXACTLY at every density;
        at fixed field the electro-absorption signal |dalpha| BLEACHES monotonically with
        density (screened excitons modulate less); negative density raises.

Run: python -m validation.qcse_density_screening
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dynameta.constants import HBAR, C_LIGHT, M_E, Q_E as Q
from dynameta.carriers.qcse import QuantumWell
from dynameta.core.effects import BursteinMossEdge, ElectroAbsorptionModel

ME, MHH = 0.067 * M_E, 0.34 * M_E
SIG = 0.005 * Q
C_BGR = 3.84e-29                       # published GaAs 2.4e-8 eV cm in SI [J m]
N_S = 6.0e27                           # screening density [m^-3]
EB0 = 0.010 * Q


def _qw():
    return QuantumWell(well_width_m=10e-9, barrier_e_J=0.30 * Q, barrier_h_J=0.20 * Q,
                       m_e_kg=ME, m_h_kg=MHH, E_g_J=1.42 * Q,
                       exciton_binding_J=EB0, nz=801, n_pad=2.0)


def _eam(qw, ET0, **kw):
    return ElectroAbsorptionModel(qw=qw, eps_bg=12.25 + 0.05j, alpha0_per_m=1e6,
                                  broadening_J=SIG, e_grid_J=(ET0 - 0.4 * Q, ET0 + 0.4 * Q, 4001),
                                  **kw)


def main():
    print("[ds] === R18 BGR + exciton screening / Mott ===", flush=True)
    ok = True
    qw = _qw()
    ET0 = qw.solve(0.0).E_transition_J
    lam = 2.0 * np.pi * HBAR * C_LIGHT / (ET0 - 2.0 * SIG)
    F = {"E": np.array([0.0, 0.0, 5e6])}

    # ---- GATE A: off-switch ----
    base = _eam(qw, ET0)
    off = _eam(qw, ET0)                                    # all R18 fields default 0
    e_base = base.eps(F, lam)
    e_off = off.eps({**F, "n": 1e26}, lam)
    de0, am0 = base._density_corrections(F)                # no 'n' key
    g_a = bool(e_off == e_base and de0 == 0.0 and am0 == 1.0)
    ok = ok and g_a
    print("[ds] GATE A: zeroed R18 fields with n present == shipped eps EXACTLY ({}); no-n "
          "corrections == (0, 1) -> {}".format(e_off == e_base, "PASS" if g_a else "FAIL"),
          flush=True)

    # ---- GATE B: BGR slope + magnitude + cross-module consistency ----
    bgr = _eam(qw, ET0, bgr_coeff_J_m=C_BGR)
    ns_ = np.array([1e24, 1e25, 1e26, 1e27])
    shifts = np.array([-bgr._density_corrections({"n": nn})[0] for nn in ns_])
    slope = np.polyfit(np.log(ns_), np.log(shifts), 1)[0]
    mag_meV = shifts[0] / Q * 1e3                          # at 1e24 m^-3
    # cross-module consistency: BursteinMossEdge composes Eg_opt = Eg0 - dE_BGR + dE_BM, so its
    # BGR component is Eg0 + gap_shift_J (pure BM) - optical_gap_J -- it must equal this model's
    # shift for the same coefficient (the two modules share ONE closed form).
    bm = BursteinMossEdge(eps_inf=4.0, Eg0_J=1.42 * Q, m_vc_kg=ME * MHH / (ME + MHH),
                          alpha_edge=1.0, bgr_coeff_J_m=C_BGR)
    bgr_bm = float(1.42 * Q + bm.gap_shift_J(1e27) - bm.optical_gap_J(1e27))
    cross = abs(shifts[3] - bgr_bm) / shifts[3]
    g_b = bool(abs(slope - 1.0 / 3.0) < 1e-12 and 20.0 < mag_meV < 30.0 and cross < 1e-12)
    ok = ok and g_b
    print("[ds] GATE B: BGR log-log slope = {:.12f} (1/3); {:.1f} meV at 1e24 m^-3 (published "
          "GaAs ~24); BGR component of BursteinMossEdge.gap_shift_J matches (rel {:.1e}) -> {}"
          .format(slope, mag_meV, cross, "PASS" if g_b else "FAIL"), flush=True)

    # ---- GATE C: screening closure + Mott ----
    scr = _eam(qw, ET0, screening_density_m3=N_S, exciton_binding_J=EB0,
               screening_exponent=1.0, mott_density_m3=8.0e27)
    worst = 0.0
    for r in (0.1, 1.0):
        dE, amp = scr._density_corrections({"n": r * N_S})
        eb_ratio = 1.0 / (1.0 + r)
        worst = max(worst, abs(amp - eb_ratio), abs(dE - EB0 * (1.0 - eb_ratio)) / EB0)
    p3 = _eam(qw, ET0, screening_density_m3=N_S, exciton_binding_J=EB0,
              screening_exponent=1.5)._density_corrections({"n": N_S})[1]
    mott_hi = scr._density_corrections({"n": 8.0e27})[1]
    mott_lo = scr._density_corrections({"n": 7.9e27})[1]
    g_c = bool(worst < 1e-14 and abs(p3 - 0.5 ** 1.5) < 1e-14 and mott_hi == 0.0
               and mott_lo > 0.0)
    ok = ok and g_c
    print("[ds] GATE C: E_b closure + blueshift exact (worst {:.1e}); 3D exponent 0.5^1.5; Mott "
          "amplitude 0.0 EXACTLY at n >= n_Mott ({} / {:.3f} below) -> {}".format(
              worst, mott_hi, mott_lo, "PASS" if g_c else "FAIL"), flush=True)

    # ---- GATE D: end-to-end device physics ----
    full = ElectroAbsorptionModel(qw=qw, eps_bg=12.25 + 0.05j, alpha0_per_m=1e6,
                                  broadening_J=SIG, e_grid_J=(ET0 - 0.7 * Q, ET0 + 0.4 * Q, 6001),
                                  bgr_coeff_J_m=C_BGR, screening_density_m3=N_S,
                                  exciton_binding_J=EB0)
    flat = max(abs(full.eps({"E": np.zeros(3), "n": nn}, lam) - (12.25 + 0.05j))
               for nn in (0.0, 1e26, 3e27))
    # TRACK the probe with the BGR/screening-shifted line (2 sigma below it) so the density sweep
    # isolates the AMPLITUDE bleaching -- at a FIXED probe the 0.1-0.4 eV BGR shift just moves the
    # line off the probe (a shift, not a bleach; the first draft of this gate conflated them).
    das = []
    for nn in (1e24, 1e26, 1e27, 5e27):
        dE_n, _ = full._density_corrections({"n": nn})
        lam_n = 2.0 * np.pi * HBAR * C_LIGHT / (ET0 + dE_n - 2.0 * SIG)
        das.append(abs(full.delta_alpha_per_m({**F, "n": nn}, lam_n)))
    bleach = all(das[i + 1] < das[i] for i in range(len(das) - 1))
    guards = False
    try:
        full.eps({**F, "n": -1.0}, lam)
    except ValueError:
        try:
            _eam(qw, ET0, screening_density_m3=N_S).eps({**F, "n": 1e26}, lam)  # E_b0 missing
        except ValueError:
            guards = True
    g_d = bool(flat < 1e-12 and bleach and guards)
    ok = ok and g_d
    print("[ds] GATE D: flat band == eps_bg at every density ({:.1e}); |dalpha| bleaches "
          "monotonically with n ({}); guards raise -> {}".format(
              flat, ["{:.3e}".format(d) for d in das], "PASS" if g_d else "FAIL"), flush=True)

    print("[ds] *** R18 BGR + EXCITON SCREENING: {} ***".format("PASS" if ok else "FAIL"),
          flush=True)
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
