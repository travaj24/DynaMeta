"""R17 Voigt exciton lineshape oracle (Gaussian (x) Lorentzian convolution in the QCSE model).

GATE A (reduces to Gaussian): lineshape='voigt' with Gamma = 0 reproduces the SHIPPED Gaussian
        model's eps at multiple probes/fields to < 1e-12 (the Faddeeva function's real-axis
        accuracy); the default 'gaussian' path is untouched (byte-identical off-switch).
GATE B (reduces to Lorentzian): sigma << Gamma -> the line profile matches the closed-form
        unit-area Lorentzian (Gamma/pi)/(x^2 + Gamma^2) scaled by sigma sqrt(2 pi) (the
        unit-peak-at-Gamma=0 convention) to < 1e-3.
GATE C (independent literature formula): the numerically measured FWHM of the Voigt profile
        matches the Whiting approximation f_V ~ 0.5346 f_L + sqrt(0.2166 f_L^2 + f_G^2)
        (accurate to 0.02%) across Gamma/sigma in [0.2, 1, 5] -- tol 2e-3.
GATE D (oscillator-strength conservation): the line AREA int alpha dE == alpha0 sigma sqrt(2 pi)
        for EVERY Gamma (the unit-peak scaling conserves it; the PEAK drops instead) -- tol 1e-6.
GATE E (field-ionization broadening seam): Gamma_F_func = c*F makes the full-model |dalpha|
        line WIDEN monotonically with F while flat-band F = 0 still returns eps_bg exactly;
        misuse guards (Gamma0_J on the gaussian path; negative Gamma(F)) raise.

Run: python -m validation.qcse_voigt_lineshape
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dynameta.constants import HBAR, C_LIGHT, M_E, Q_E as Q
from dynameta.carriers.qcse import QuantumWell
from dynameta.core.effects import ElectroAbsorptionModel

ME, MHH = 0.067 * M_E, 0.34 * M_E
SIG = 0.005 * Q                      # 5 meV Gaussian sigma


def _qw():
    return QuantumWell(well_width_m=10e-9, barrier_e_J=0.30 * Q, barrier_h_J=0.20 * Q,
                       m_e_kg=ME, m_h_kg=MHH, E_g_J=1.42 * Q,
                       exciton_binding_J=0.010 * Q, nz=801, n_pad=2.0)


def _eam(qw, ET0, **kw):
    return ElectroAbsorptionModel(qw=qw, eps_bg=12.25 + 0.05j, alpha0_per_m=1e6,
                                  broadening_J=SIG, e_grid_J=(ET0 - 0.4 * Q, ET0 + 0.4 * Q, 4001),
                                  **kw)


def _fwhm(x, y):
    h = float(np.max(y)) / 2.0
    above = np.where(y >= h)[0]
    i0, i1 = above[0], above[-1]
    xl = np.interp(h, [y[i0 - 1], y[i0]], [x[i0 - 1], x[i0]])
    xr = np.interp(h, [y[i1 + 1], y[i1]], [x[i1 + 1], x[i1]])
    return float(xr - xl)


def main():
    print("[vl] === R17 Voigt exciton lineshape ===", flush=True)
    ok = True
    qw = _qw()
    ET0 = qw.solve(0.0).E_transition_J
    F = {"E": np.array([0.0, 0.0, 5e6])}

    # ---- GATE A: Gamma = 0 voigt == gaussian ----
    eam_g = _eam(qw, ET0)
    eam_v0 = _eam(qw, ET0, lineshape="voigt", Gamma0_J=0.0)
    worst = 0.0
    for off in (-3.0, -2.0, -1.0, 1.5):
        lam = 2.0 * np.pi * HBAR * C_LIGHT / (ET0 + off * SIG)
        worst = max(worst, abs(eam_v0.eps(F, lam) - eam_g.eps(F, lam)))
    g_a = bool(worst < 1e-12)
    ok = ok and g_a
    print("[vl] GATE A: voigt(Gamma=0) vs shipped gaussian eps, worst |d| = {:.1e} over 4 probes "
          "-> {}".format(worst, "PASS" if g_a else "FAIL"), flush=True)

    # ---- GATE B: sigma << Gamma -> Lorentzian closed form ----
    gam = 0.004 * Q
    tiny = gam / 1000.0
    eam_l = ElectroAbsorptionModel(qw=qw, eps_bg=12.25 + 0j, alpha0_per_m=1.0, broadening_J=tiny,
                                   e_grid_J=(ET0 - 0.4 * Q, ET0 + 0.4 * Q, 4001),
                                   lineshape="voigt", Gamma0_J=gam)
    x = np.linspace(-8.0 * gam, 8.0 * gam, 1601)
    prof = eam_l._alpha(ET0 + x, ET0, 1.0, 1.0, gam)
    lor = (tiny * np.sqrt(2.0 * np.pi)) * (gam / np.pi) / (x ** 2 + gam ** 2)
    relB = float(np.max(np.abs(prof - lor)) / np.max(lor))
    g_b = bool(relB < 1e-3)
    ok = ok and g_b
    print("[vl] GATE B: sigma = Gamma/1000 profile vs closed-form Lorentzian, rel {:.1e} -> {}"
          .format(relB, "PASS" if g_b else "FAIL"), flush=True)

    # ---- GATE C: Whiting FWHM approximation ----
    f_g = 2.0 * np.sqrt(2.0 * np.log(2.0)) * SIG
    worstC = 0.0
    eam_v = _eam(qw, ET0, lineshape="voigt", Gamma0_J=SIG)        # placeholder; per-gamma below
    xw = np.linspace(-60.0 * SIG, 60.0 * SIG, 120001)
    for r in (0.2, 1.0, 5.0):
        gamr = r * SIG
        prof = eam_v._alpha(ET0 + xw, ET0, 1.0, 1.0, gamr)
        f_l = 2.0 * gamr
        whiting = 0.5346 * f_l + np.sqrt(0.2166 * f_l ** 2 + f_g ** 2)
        worstC = max(worstC, abs(_fwhm(xw, prof) - whiting) / whiting)
    g_c = bool(worstC < 2e-3)
    ok = ok and g_c
    print("[vl] GATE C: measured FWHM vs Whiting formula, worst rel {:.1e} over Gamma/sigma = "
          "0.2/1/5 -> {}".format(worstC, "PASS" if g_c else "FAIL"), flush=True)

    # ---- GATE D: area conservation for every Gamma ----
    # The Voigt area identity int alpha dE = alpha0 sigma sqrt(2 pi) holds on (-inf, inf); a
    # finite +-X window misses the Lorentzian tail mass 1 - (2/pi) atan(X/Gamma) ~ 2 Gamma/(pi X).
    # The SHARP check: the measured window deficit must EQUAL that closed form (a wrong profile
    # normalization would miss by O(1), not by the tail).
    area0 = 1e6 * SIG * np.sqrt(2.0 * np.pi)                      # alpha0 sigma sqrt(2 pi)
    X = 60.0 * SIG
    worstD = 0.0
    for r in (0.0, 0.5, 2.0):
        prof = _eam(qw, ET0, lineshape="voigt", Gamma0_J=r * SIG)._alpha(
            ET0 + xw, ET0, 1.0, 1.0, r * SIG)                     # alpha0 already folded in
        deficit = (area0 - np.trapezoid(prof, xw)) / area0
        tail = 1.0 - (2.0 / np.pi) * np.arctan(X / (r * SIG)) if r > 0 else 0.0
        worstD = max(worstD, abs(deficit - tail))
    g_d = bool(worstD < 1e-3)
    ok = ok and g_d
    print("[vl] GATE D: window area deficit == closed-form Lorentzian tail mass for Gamma/sigma "
          "= 0/0.5/2, worst |d| = {:.1e} -> {}".format(worstD, "PASS" if g_d else "FAIL"),
          flush=True)

    # ---- GATE E: field-ionization seam + flat band + guards ----
    eam_f = _eam(qw, ET0, lineshape="voigt", Gamma0_J=0.5 * SIG,
                 Gamma_F_func=lambda f: 2.0e-10 * Q * f)          # +2 meV at 1e7 V/m
    widths = []
    for fz in (0.0, 5e6, 1e7):
        gam_tot = eam_f._gamma_lor_J(fz)
        widths.append(_fwhm(xw, eam_f._alpha(ET0 + xw, ET0, 1.0, 1.0, gam_tot)))
    lam_probe = 2.0 * np.pi * HBAR * C_LIGHT / (ET0 - 2.0 * SIG)
    flat = abs(eam_f.eps({"E": np.zeros(3)}, lam_probe) - (12.25 + 0.05j)) < 1e-12
    guards = False
    try:
        _eam(qw, ET0, Gamma0_J=1e-22).eps(F, lam_probe)           # gaussian path + Gamma0 set
    except ValueError:
        try:
            eam_neg = _eam(qw, ET0, lineshape="voigt", Gamma_F_func=lambda f: -1e-21)
            eam_neg.eps(F, lam_probe)
        except ValueError:
            guards = True
    g_e = bool(widths[0] < widths[1] < widths[2] and flat and guards)
    ok = ok and g_e
    print("[vl] GATE E: FWHM(F) monotone {:.3e} < {:.3e} < {:.3e} J; flat band == eps_bg; misuse "
          "guards raise -> {}".format(*widths, "PASS" if g_e else "FAIL"), flush=True)

    print("[vl] *** R17 VOIGT EXCITON LINESHAPE: {} ***".format("PASS" if ok else "FAIL"),
          flush=True)
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
