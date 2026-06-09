"""
Quantum intersubband eps_zz oracle (roadmap R7). IntersubbandEffect maps a Schrodinger-Poisson
SubbandResult to a diagonal-anisotropic permittivity: eps_xx=eps_yy = intraband Drude, eps_zz =
Drude + sum_{i<j} Lorentzian at hbar w_ij = E_j - E_i (the growth-axis intersubband transitions).

GATE 1 -- REDUCES TO SCALAR DRUDE (byte-identical off-switch): a SubbandResult with ONE occupied
        sub-band has no i<j pair, so eps -> diag(d,d,d) with d == DrudeOptical(eps_inf, m_opt,
        gamma_intra).eps(lambda, n_m3 = n_s/Leff) to ~1e-13 (the same wp^2 = n q^2/(eps0 m) formula).

GATE 2 -- THOMAS-REICHE-KUHN f-SUM RULE (independent analytic oracle, NO eps formula, NO FEM): for an
        infinite square well the dimensionless oscillator strengths f_1j = (2 m w_1j/hbar)|z_1j|^2 from
        the GROUND state sum to 1. Solve the discretized well (SchrodingerPoisson1D, flat U, Dirichlet
        walls), compute z_1j = <psi_1|z|psi_j> via the same trapz the model uses, and assert
        sum_j f_1j -> 1 to < 2e-2 over ~80 states. Validates the matrix element + the f-strength form
        independent of any permittivity expression (and proves the mass-cancellation: f uses the SAME m).

GATE 3 -- INTERSUBBAND LINE PRESENT, PASSIVE, IN-BAND: a ~1.8 nm well with two occupied sub-bands puts
        a Lorentzian peak in Im(eps_zz) at lambda_12 = 2 pi c / w_12 (assert the swept peak lands within
        2% of it AND in the telecom band, the north-star claim); Im(eps_zz) > 0 at all lambda (exp(-iwt)
        passivity); eps_xx = eps_yy stay the flat intraband Drude with NO peak at w_12.

Run: python -m validation.intersubband_eps_zz
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dynameta.constants import HBAR, C_LIGHT, M_E, Q_E, EPS0
from dynameta.core.effects import IntersubbandEffect, as_tensor
from dynameta.core.numerics import trapz
from dynameta.carriers.schrodinger_poisson import SchrodingerPoisson1D, SubbandResult
from dynameta.materials.optical_model import DrudeOptical

EPS_INF = 4.25
M_OPT = 0.225 * M_E
GAM_INTRA = 1.1e14
GAM_INTER = 1.0e13


def main():
    print("[is] === quantum intersubband eps_zz from sub-band wavefunctions ===", flush=True)
    ok = True

    # ---- GATE 1: single occupied sub-band -> reduces to scalar Drude * I ----
    L1 = 3e-9
    z1 = np.linspace(0.0, L1, 200)
    psi1 = np.sin(np.pi * z1 / L1)[:, None]
    psi1 = psi1 / np.sqrt(np.sum(psi1[:, 0] ** 2) * (z1[1] - z1[0]))     # sum|psi|^2 h = 1
    ns1 = np.array([4.0e17])                                            # m^-2 (one band)
    res1 = SubbandResult(energies_J=np.array([1e-21]), psi=psi1, z_m=z1, sheet_density_m2=ns1)
    model = IntersubbandEffect(EPS_INF, M_OPT, GAM_INTRA, GAM_INTER)
    lam = 1300e-9
    eps_t = model.eps({"subband": res1}, lam)
    n3d = float(ns1.sum()) / (z1[-1] - z1[0])
    eps_d = complex(DrudeOptical(eps_inf=EPS_INF, m_opt_kg=M_OPT, gamma_rad_s=GAM_INTRA).eps(lam, n_m3=n3d))
    ref = as_tensor(np.asarray(eps_d))
    d1 = float(np.max(np.abs(eps_t - ref)))
    g1 = bool(d1 < 1e-13 and abs(eps_t[2, 2] - eps_t[0, 0]) < 1e-13)
    ok = ok and g1
    print("[is] GATE 1 (1 sub-band == Drude*I): max|d|={:.1e}, eps_zz==eps_xx -> {}".format(
        d1, "PASS" if g1 else "FAIL"), flush=True)

    # ---- GATE 2: TRK f-sum rule on an infinite square well ----
    Lw, mw, Nz, n_st = 5e-9, 0.2 * M_E, 400, 80
    zg = np.linspace(0.0, Lw, Nz)
    sp = SchrodingerPoisson1D(zg, mw, T_K=300.0)
    E, psi, zi = sp.solve_schrodinger(np.zeros(Nz), n_states=n_st)
    fsum = 0.0
    for j in range(1, len(E)):
        z1j = trapz(psi[:, 0] * zi * psi[:, j], zi)
        w1j = (E[j] - E[0]) / HBAR
        fsum += (2.0 * mw * w1j / HBAR) * z1j ** 2
    g2 = bool(abs(fsum - 1.0) < 2e-2)
    ok = ok and g2
    print("[is] GATE 2 (TRK f-sum rule, {} states): sum f_1j = {:.5f} (target 1) -> {}".format(
        len(E), fsum, "PASS" if g2 else "FAIL"), flush=True)

    # ---- GATE 3: intersubband line present, passive, in telecom band ----
    Lw3, mw3 = 1.84e-9, 0.35 * M_E
    zg3 = np.linspace(0.0, Lw3, 220)
    sp3 = SchrodingerPoisson1D(zg3, mw3, T_K=300.0)
    E3, psi3, zi3 = sp3.solve_schrodinger(np.zeros(220), n_states=3)
    ns3 = np.array([5.0e17, 1.0e17, 0.0])                              # two occupied bands
    res3 = SubbandResult(energies_J=E3, psi=psi3, z_m=zi3, sheet_density_m2=ns3)
    w12 = (E3[1] - E3[0]) / HBAR
    lam12 = 2.0 * np.pi * C_LIGHT / w12
    lams = np.linspace(1000e-9, 1700e-9, 400)
    im_zz = np.array([model.eps({"subband": res3}, l)[2, 2].imag for l in lams])
    re_xx = np.array([model.eps({"subband": res3}, l)[0, 0].real for l in lams])
    eps_xx = np.array([model.eps({"subband": res3}, l)[0, 0] for l in lams])
    eps_yy = np.array([model.eps({"subband": res3}, l)[1, 1] for l in lams])
    lam_peak = lams[int(np.argmax(im_zz))]
    # eps_xx must be the flat Drude (no intersubband peak): its Im is monotone vs lambda, no resonance
    im_xx = np.array([e.imag for e in eps_xx])
    no_xx_peak = bool(np.argmax(im_xx) in (0, len(lams) - 1))          # Drude Im peaks at a band edge
    passive = bool(np.all(im_zz > 0.0))
    xy_equal = bool(np.max(np.abs(eps_xx - eps_yy)) < 1e-15)
    rel_pos = abs(lam_peak - lam12) / lam12
    in_band = bool(1200e-9 < lam12 < 1600e-9)
    g3 = bool(rel_pos < 2e-2 and passive and xy_equal and no_xx_peak and in_band)
    ok = ok and g3
    print("[is] GATE 3: Im(eps_zz) peak at {:.1f} nm vs lambda_12={:.1f} nm (rel {:.2e}); passive={}, "
          "eps_xx==eps_yy={}, no in-plane peak={}, telecom={} -> {}".format(
              lam_peak * 1e9, lam12 * 1e9, rel_pos, passive, xy_equal, no_xx_peak, in_band,
              "PASS" if g3 else "FAIL"), flush=True)

    print("[is] *** INTERSUBBAND eps_zz: {} ***".format("PASS" if ok else "FAIL"), flush=True)
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
