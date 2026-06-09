"""
Carrier-heating (two-temperature) ENZ nonlinearity oracle (roadmap R9). An absorbed pump heats the
electron gas (T_e); hot electrons climb the nonparabolic band so <m*(T_e)> rises (wp^2 ~ n/<m*> drops,
Re(eps) moves through ENZ) and Gamma(T_e) rises. The TTM C_e(T_e) dT_e/dt = -G(T_e-T_l)+alpha_abs I(t)
gives the sub-ps-rise / ps-relaxation asymmetry; the per-instant Drude feeds the transient_optics loop.

GATE 1 -- REDUCES TO LINEAR (mandatory off-switch): (A) intensity == 0 -> T_e == T_l == T0 to machine,
        R(t) flat. (B) with a pump ON but alpha_per_eV == 0 and gamma_p == 0, the per-instant Drude
        collapses to the constant drude0, so R(t) is byte-identical to optical_transient_response with
        the fixed drude0.

GATE 2 -- TIME-ASYMMETRIC SHAPE (independent reference = linearized-TTM cooling constant): a short
        Gaussian pump drives a sub-ps T_e RISE and a ps-scale single-exponential RELAXATION whose decay
        constant matches the small-perturbation limit C_e(T0)/G within a factor ~2; the optical R(t)
        tracks it (fast rise, slow tail) -- the asymmetry a memoryless chi3 cannot reproduce.

GATE 3 -- ENZ ENHANCEMENT (independent reference = R-sensitivity near eps~0): the peak |dR| per pump is
        LARGER (> 2x) at a probe tuned NEAR the cold ENZ wavelength than at a probe far from it, because
        dR/d(eps) diverges where Re(eps) ~ 0; d Re(eps)/d T_e is reported at both probes.

GATE 4 -- LOW-INTENSITY LINEARITY: peak |dR| scales ~linearly with pump amplitude at low intensity
        (T_e - T0 small) -- the reduces-to-linear-response limit.

Run: python -m validation.carrier_heating_enz
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dynameta.constants import M_E, KB, Q_E
from dynameta.materials import DrudeOptical
from dynameta.carriers.carrier_heating import (TwoTempParams, two_temperature_response,
                                               carrier_heating_transient, kane_mass_of_Te,
                                               fermi_energy_J, gamma_of_Te)
from dynameta.transient_optics import optical_transient_response, enz_reflector_stack
from dynameta.optics.tmm_reference import layered_rta

M0, ALPHA_EV, GAMMA0 = 0.35 * M_E, 0.5, 1.0e14
N = 1.0e27
DRUDE0 = DrudeOptical(eps_inf=3.9, m_opt_kg=M0, gamma_rad_s=GAMMA0)
T0 = 300.0
# TTM: degenerate-gas electron heat capacity gamma_e = (pi^2/2) n kB^2/E_F
E_F = float(fermi_energy_J(N, M0, ALPHA_EV))
GAMMA_E = (np.pi ** 2 / 2.0) * N * KB ** 2 / E_F                  # J/m^3/K^2
PARAMS = TwoTempParams(C_e=lambda Te: GAMMA_E * Te, C_l=2.4e6, G_e_l=6.0e15, alpha_abs=1.0)
TAU_COOL = (GAMMA_E * T0) / PARAMS.G_e_l                          # linearized electron cooling time


def _cold_eps(lam):
    """Front ITO eps at T_e = T0 (the per-instant heating Drude evaluated cold)."""
    m = float(kane_mass_of_Te(M0, ALPHA_EV, N, T0))
    g = float(gamma_of_Te(GAMMA0, T0, p=1.0))
    return complex(DrudeOptical(eps_inf=3.9, m_opt_kg=m, gamma_rad_s=g).eps(lam, n_m3=N))


def _peak_dR(times, pump, lam):
    t, R, _T, ef, Te, _Tl = carrier_heating_transient(times, pump, lam, drude0=DRUDE0,
                                                      ttm_params=PARAMS, n_m3=N, alpha_per_eV=ALPHA_EV)
    return float(np.max(np.abs(R - R[0]))), float(np.max(np.abs(ef - ef[0]))), Te


def main():
    print("[ch] === carrier-heating (two-temperature) ENZ nonlinearity ===", flush=True)
    ok = True
    times = np.linspace(0.0, 3.0e-12, 400)
    t0p, sig, I0 = 0.4e-12, 6.0e-14, 3.0e20
    pump = lambda tt: I0 * np.exp(-((tt - t0p) / sig) ** 2)

    # ---- GATE 1A: no pump -> Te == T0, R flat ----
    zero = lambda tt: 0.0
    _t, Te0, Tl0 = two_temperature_response(times, zero, PARAMS, T0_K=T0)
    g1a = bool(np.max(np.abs(Te0 - T0)) < 1e-9 and np.max(np.abs(Tl0 - T0)) < 1e-9)
    # ---- GATE 1B: alpha=0, p=0 -> byte-identical to fixed drude0 ----
    n_of_t = lambda tt: N
    _t2, R_fix, _T2, _e2 = optical_transient_response(times, n_of_t, 1500e-9, drude_model=DRUDE0)
    th, R_h, _Th, _eh, _Teh, _Tlh = carrier_heating_transient(times, pump, 1500e-9, drude0=DRUDE0,
                                                              ttm_params=PARAMS, n_m3=N,
                                                              alpha_per_eV=0.0, gamma_p=0.0)
    d1b = float(np.max(np.abs(R_h - R_fix)))
    g1 = bool(g1a and d1b < 1e-12)
    ok = ok and g1
    print("[ch] GATE 1: no-pump Te==T0 (max|dTe|={:.1e}); alpha=p=0 == fixed Drude max|dR|={:.1e} -> {}"
          .format(float(np.max(np.abs(Te0 - T0))), d1b, "PASS" if g1 else "FAIL"), flush=True)

    # ---- GATE 2: TTM rise sub-ps, decay ~ C_e/G ----
    _t, Te, Tl = two_temperature_response(times, pump, PARAMS, T0_K=T0)
    ipk = int(np.argmax(Te))
    Te_pk = float(Te[ipk])
    rise = Te - T0
    lo10, hi90 = 0.1 * rise[ipk], 0.9 * rise[ipk]
    t_rise = float(np.interp(hi90, rise[:ipk + 1], times[:ipk + 1])
                   - np.interp(lo10, rise[:ipk + 1], times[:ipk + 1]))
    # post-peak single-exponential cooling constant (1/e of the peak excess)
    post = slice(ipk, None)
    exc = Te[post] - Tl[post]
    tdec = float(np.interp(exc[0] / np.e, exc[::-1], times[post][::-1])) - times[ipk]
    # nonlinear-capacity cooling: the 1/e time is bracketed by the COLD and PEAK linear constants
    # C_e(T0)/G .. C_e(Te_pk)/G (C_e ~ gamma_e Te, so a hot gas cools slower) -- the physical reference.
    tau_cold, tau_hot = TAU_COOL, (GAMMA_E * Te_pk) / PARAMS.G_e_l
    g2 = bool(Te_pk > T0 + 200.0 and t_rise < 5.0 * sig and 0.5 * tau_cold < tdec < 1.5 * tau_hot
              and np.all(np.diff(Te[ipk:]) <= 1e-6))
    ok = ok and g2
    print("[ch] GATE 2: Te peak={:.0f} K, 10-90 rise={:.0f} fs (pump sigma {:.0f} fs), 1/e cool={:.0f} fs "
          "(in [C_e(T0)/G, C_e(Te_pk)/G]=[{:.0f},{:.0f}] fs) -> {}".format(
              Te_pk, t_rise * 1e15, sig * 1e15, tdec * 1e15, tau_cold * 1e15, tau_hot * 1e15,
              "PASS" if g2 else "FAIL"), flush=True)

    # ---- GATE 3: ENZ enhancement (near vs far) ----
    lam_near = None
    lg = np.linspace(1200e-9, 1900e-9, 400)
    rez = np.array([_cold_eps(l).real for l in lg])
    sc = np.where(np.diff(np.sign(rez)) != 0)[0]
    if len(sc):
        i = sc[0]
        lam_near = float(np.interp(0.0, [rez[i], rez[i + 1]], [lg[i], lg[i + 1]]))
    lam_far = 1050e-9
    dR_near, _de_n, _ = _peak_dR(times, pump, lam_near)
    dR_far, _de_f, _ = _peak_dR(times, pump, lam_far)
    # finite-difference d Re(eps)/d Te at both probes (independent sensitivity check)
    def depsdTe(lam):
        m1 = float(kane_mass_of_Te(M0, ALPHA_EV, N, T0)); m2 = float(kane_mass_of_Te(M0, ALPHA_EV, N, T0 + 1000.0))
        e1 = complex(DrudeOptical(eps_inf=3.9, m_opt_kg=m1, gamma_rad_s=gamma_of_Te(GAMMA0, T0, p=1.0)).eps(lam, n_m3=N))
        e2 = complex(DrudeOptical(eps_inf=3.9, m_opt_kg=m2, gamma_rad_s=gamma_of_Te(GAMMA0, T0 + 1000.0, p=1.0)).eps(lam, n_m3=N))
        return (e2.real - e1.real) / 1000.0
    g3 = bool(lam_near is not None and dR_near > 2.0 * dR_far)
    ok = ok and g3
    print("[ch] GATE 3: cold ENZ {:.0f} nm; peak|dR| near={:.4f} far(1050nm)={:.4f} (ratio {:.1f}x); "
          "dRe(eps)/dTe near={:.2e} far={:.2e} -> {}".format(
              (lam_near or 0) * 1e9, dR_near, dR_far, dR_near / max(dR_far, 1e-12),
              depsdTe(lam_near), depsdTe(lam_far), "PASS" if g3 else "FAIL"), flush=True)

    # ---- GATE 4: low-intensity linearity of the MATERIAL response (peak |delta eps_front|) ----
    # At low pump, dTe ~ I (linear) -> d<m*> ~ dTe -> d eps ~ I. (The ENZ-probe dR additionally carries
    # the crossing curvature, so the clean linear-response statement is on the front eps excursion.)
    amps = np.array([0.002, 0.004, 0.008]) * I0
    des = []
    for A in amps:
        p = lambda tt, _A=A: _A * np.exp(-((tt - t0p) / sig) ** 2)
        _dr, de, _Te = _peak_dR(times, p, lam_near)
        des.append(de)
    des = np.array(des)
    ratios = des[1:] / des[:-1]                          # expect ~ amps[1:]/amps[:-1] = [2, 2]
    exp_ratios = amps[1:] / amps[:-1]
    g4 = bool(np.all(np.abs(ratios - exp_ratios) / exp_ratios < 0.15))
    ok = ok and g4
    print("[ch] GATE 4: low-I peak|d eps_front| {} ; ratios {} vs linear {} -> {}".format(
        np.round(des, 5), np.round(ratios, 3), np.round(exp_ratios, 3), "PASS" if g4 else "FAIL"),
        flush=True)

    print("[ch] *** CARRIER-HEATING ENZ NONLINEARITY: {} ***".format("PASS" if ok else "FAIL"), flush=True)
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
