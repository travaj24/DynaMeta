"""
Quantum intersubband eps_zz oracle (roadmap R7). IntersubbandEffect maps a Schrodinger-Poisson
SubbandResult to a diagonal-anisotropic permittivity: eps_xx=eps_yy = intraband Drude, eps_zz =
eps_inf + sum_{i<j} Lorentzian at hbar w_ij = E_j - E_i (the growth-axis intersubband transitions;
v0.16: NO Drude term along z -- the intersubband oscillator strengths of a complete set
already exhaust the free-carrier response; the pre-0.16 eps_zz = Drude + sum is the deprecated
z_response='drude_plus_subbands').

GATE 1 -- ONE SUB-BAND: no i<j pair, so eps_xx = eps_yy == DrudeOptical(eps_inf, m_opt,
        gamma_intra).eps(lambda, n_m3 = n_s/Leff) and eps_zz == eps_inf (a confined electron is not
        accelerated along z) to ~1e-13; the deprecated legacy form still gives Drude * I.

GATE 2 -- THOMAS-REICHE-KUHN f-SUM RULE (independent analytic oracle, NO eps formula, NO FEM): for an
        infinite square well the dimensionless oscillator strengths f_1j = (2 m w_1j/hbar)|z_1j|^2 from
        the GROUND state sum to 1. Solve the discretized well (SchrodingerPoisson1D, flat U, Dirichlet
        walls), compute z_1j = <psi_1|z|psi_j> via the same trapz the model uses, and assert
        sum_j f_1j -> 1 to < 2e-2 over ~80 states. Validates the matrix element + the f-strength form
        independent of any permittivity expression (and proves the mass-cancellation: f uses the SAME m).

GATE 5 -- CANONICAL ABSORBER + GAIN DIAGNOSTIC (audit R-1): the standard intersubband absorber is a
        FULL ground sub-band radiating into an EMPTY excited one -- the largest possible population
        difference and therefore the strongest line. The shipped selection drew BOTH the initial and
        the final index from the occupancy-filtered set, so it deleted exactly that configuration
        (bare Drude, eps_zz == eps_xx, no warning) and, for the inverted n_s = [0, n], removed the
        pair before the N_ij < 0 gain warning could fire. Assert (a) n_s = [n, 0, 0] reproduces the
        closed-form Lorentzian peak Im(chi_zz) = S_01/(eps0 w_12 gamma_inter) at lambda_12, and
        (b) n_s = [0, n] raises the gain warning and clamps the line to exactly zero.

GATE 6 -- f-SUM OF A COMPLETE SLAB + DISCRETE IDENTITY (no z Drude term; Riemann dipole): 6 nm flat-band
        slab (n_bg 3e26 at the model E_F, m 0.30 m_e, the generic gated film): last_fsum_ratio within 1e-3
        of 1, and with every interior state kept it equals sum n_i kappa_i / sum n_i, kappa_i = h sum_k
        psi_i,k psi_i,k+1 (the exact finite-difference double commutator) to 1e-12.

GATE 7 -- ORIGIN INVARIANCE (Riemann dipole about the centre): a 100 nm shift of z moves eps_zz by < 1e-12
        (v0.15.1: 1.3e-3, measured on the v0.15.1 export).

GATE 8 -- WIDE-SLAB LIMIT (no z Drude term): abs(eps_zz - eps_xx) for 10/20/40 nm slabs falls strictly,
        20 -> 40 nm ratio in [1.6, 2.4] (1/L), < 0.017 at 40 nm (1.3x the measured value; MEASURED 0.0509
        / 0.0257 / 0.0133, Leff the wall-to-wall slab thickness).

GATE 3 -- INTERSUBBAND LINE PRESENT, PASSIVE, IN-BAND: a ~1.8 nm well with two occupied sub-bands puts
        a Lorentzian peak in Im(eps_zz) at lambda_12 = 2 pi c / w_12 (assert the swept peak lands within
        2% of it AND in the telecom band, the north-star claim); Im(eps_zz) > 0 at all lambda (exp(-iwt)
        passivity); eps_xx = eps_yy stay the flat intraband Drude with NO peak at w_12.

Run: python -m validation.intersubband_eps_zz
"""

import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dynameta.constants import HBAR, C_LIGHT, M_E, Q_E, EPS0
from dynameta.core.effects import IntersubbandEffect, as_tensor
from dynameta.core.numerics import trapz
from dynameta.carriers.schrodinger_poisson import (SchrodingerPoisson1D, SubbandResult,
                                                   sp_bulk_fermi_level_J)
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
    # synthetic 1-3 state wells are incomplete sets by construction: the f-sum advisory is off for them
    model = IntersubbandEffect(EPS_INF, M_OPT, GAM_INTRA, GAM_INTER, fsum_check="off")
    lam = 1300e-9
    eps_t = model.eps({"subband": res1}, lam)
    n3d = float(ns1.sum()) / (z1[-1] - z1[0])
    eps_d = complex(DrudeOptical(eps_inf=EPS_INF, m_opt_kg=M_OPT, gamma_rad_s=GAM_INTRA).eps(lam, n_m3=n3d))
    d1 = max(abs(eps_t[0, 0] - eps_d), abs(eps_t[1, 1] - eps_d))
    dz1 = abs(eps_t[2, 2] - EPS_INF)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")                                   # the deprecation advisory
        leg = IntersubbandEffect(EPS_INF, M_OPT, GAM_INTRA, GAM_INTER,
                                 z_response="drude_plus_subbands").eps({"subband": res1}, lam)
    dleg = float(np.max(np.abs(leg - as_tensor(np.asarray(eps_d)))))
    g1 = bool(d1 < 1e-13 and dz1 < 1e-13 and dleg < 1e-13)
    ok = ok and g1
    print("[is] GATE 1 (1 sub-band): |eps_xx - Drude|={:.1e}, |eps_zz - eps_inf|={:.1e}, legacy "
          "Drude*I max|d|={:.1e} -> {}".format(d1, dz1, dleg, "PASS" if g1 else "FAIL"), flush=True)

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

    # ---- GATE 4: integrated LINE STRENGTH (audit-v2: gates 1-3 checked reduction/sum-rule/position
    # but never the strength normalization). For the Lorentzian form chi = S/(eps0 (w0^2-w^2-i w g))
    # the EXACT sum rule  integral_0^inf w * Im(chi) dw = (pi/2) S / eps0  holds for ANY g, so
    # integrating the MODEL OUTPUT Im(eps_zz - eps_inf) over frequency must recover (pi/2) sum S_ij/eps0
    # computed independently from the SubbandResult ingredients. Catches any wrong prefactor in S_ij.
    w12 = (E3[1] - E3[0]) / HBAR
    wgrid = np.linspace(1e12, 25.0 * w12, 60001)                      # tail correction ~ g/W ~ 3e-4
    lam_w = 2.0 * np.pi * C_LIGHT / wgrid
    im_chi = np.array([(model.eps({"subband": res3}, l)[2, 2] - EPS_INF).imag for l in lam_w])
    integral = trapz(wgrid * im_chi, wgrid)
    Leff = float(zi3[-1] - zi3[0])
    S_sum = 0.0
    for a in range(len(ns3)):
        for b in range(a + 1, len(ns3)):
            # audit R-1: mirror the model's FIXED selection -- a pair radiates on its POPULATION
            # DIFFERENCE, not on both sub-bands being occupied. This mirror previously encoded the
            # defect ("BOTH bands must be occupied"), so it agreed with the model by construction
            # and could not see that the canonical full-ground / empty-excited line was missing.
            if abs(ns3[a] - ns3[b]) <= 0.0:
                continue
            z_ab = float(np.sum(psi3[:, a] * zi3 * psi3[:, b]) * (zi3[1] - zi3[0]))   # Riemann sum (origin-free)
            N_ab = max((ns3[a] - ns3[b]) / Leff, 0.0)
            S_sum += N_ab * Q_E * Q_E * (z_ab * z_ab) * (2.0 * (E3[b] - E3[a]) / HBAR) / HBAR
    expect = (np.pi / 2.0) * S_sum / EPS0
    rel4 = abs(integral - expect) / max(abs(expect), 1e-300)
    g4 = bool(rel4 < 1e-2)
    ok = ok and g4
    print("[is] GATE 4 (integrated line strength): int w*Im(chi_zz) dw = {:.6e} vs (pi/2) S/eps0 = "
          "{:.6e} (rel {:.2e}) -> {}".format(integral, expect, rel4, "PASS" if g4 else "FAIL"),
          flush=True)

    # ---- GATE 5 (audit R-1): the canonical full/empty absorber, and the gain diagnostic ----
    ns_abs = np.array([5.0e17, 0.0, 0.0])                              # ground FULL, excited EMPTY
    res_abs = SubbandResult(energies_J=E3, psi=psi3, z_m=zi3, sheet_density_m2=ns_abs)
    t_abs = model.eps({"subband": res_abs}, lam12)
    chi_im = float((t_abs[2, 2] - EPS_INF).imag)
    z01 = float(np.sum(psi3[:, 0] * zi3 * psi3[:, 1]) * (zi3[1] - zi3[0]))       # Riemann sum (origin-free)
    S01 = (ns_abs[0] / Leff) * Q_E * Q_E * z01 * z01 * (2.0 * w12) / HBAR
    chi_an = S01 / (EPS0 * w12 * GAM_INTER)               # Lorentzian ON resonance: S/(eps0 w gamma)
    rel5a = abs(chi_im - chi_an) / max(abs(chi_an), 1e-300)
    with warnings.catch_warnings(record=True) as wrec:
        warnings.simplefilter("always")
        res_inv = SubbandResult(energies_J=E3[:2], psi=psi3[:, :2], z_m=zi3,
                                sheet_density_m2=np.array([0.0, 5.0e17]))     # fully inverted
        t_inv = model.eps({"subband": res_inv}, lam12)
        gain_warned = any("inverted population" in str(x.message) for x in wrec)
    clamped = bool(abs(t_inv[2, 2] - EPS_INF) < 1e-12)
    g5 = bool(rel5a < 3e-2 and chi_im > 0.0 and gain_warned and clamped)
    ok = ok and g5
    print("[is] GATE 5 (R-1 full-ground/EMPTY-excited absorber): Im(chi_zz) at lambda_12 = {:.5f} vs "
          "closed-form S_01/(eps0 w_12 gamma) = {:.5f} (rel {:.2e}); inverted [0,n] warns={} and "
          "clamps={} -> {}".format(chi_im, chi_an, rel5a, gain_warned, clamped,
                                   "PASS" if g5 else "FAIL"), flush=True)

    # ---- GATES 6-8 (v0.16): flat-band slabs of the generic gated film ----
    m_s = 0.30 * M_E
    ef = sp_bulk_fermi_level_J(3e26, m_s, 300.0)

    def slab(t, n_states=None):
        zz = np.linspace(0.0, t, int(round(t / 0.025e-9)) + 1)
        ns_ = min(zz.size - 3, 600) if n_states is None else n_states
        return SchrodingerPoisson1D(zz, m_s).density(np.zeros_like(zz), ef, n_states=ns_, bound_tol=1e9)

    def slab_model():
        return IntersubbandEffect(4.0, m_s, 1.2e14, 1.2e14)

    r5 = slab(6e-9)
    m5 = slab_model()
    e5 = m5.eps({"subband": r5}, 1200e-9)
    full = slab(6e-9, n_states=239)
    mf = slab_model()
    mf.eps({"subband": full}, 1200e-9)
    hf = float(full.z_m[1] - full.z_m[0])
    kap = hf * np.sum(full.psi[:-1, :] * full.psi[1:, :], axis=0)
    ident = float(np.sum(full.sheet_density_m2 * kap) / np.sum(full.sheet_density_m2))
    g6 = bool(abs(m5.last_fsum_ratio - 1.0) < 1e-3 and abs(mf.last_fsum_ratio - ident) < 1e-12)
    ok = ok and g6
    print("[is] GATE 6 (f-sum, 6 nm slab): ratio = {:.5f} (238 states), {:.12f} (all 239) vs discrete "
          "identity {:.12f} -> {}".format(m5.last_fsum_ratio, mf.last_fsum_ratio, ident,
                                          "PASS" if g6 else "FAIL"), flush=True)
    sh = SubbandResult(energies_J=r5.energies_J, psi=r5.psi, z_m=r5.z_m + 100e-9,
                       sheet_density_m2=r5.sheet_density_m2)
    d7 = abs(slab_model().eps({"subband": sh}, 1200e-9)[2, 2] - e5[2, 2])
    g7 = bool(d7 < 1e-12)
    ok = ok and g7
    print("[is] GATE 7 (100 nm origin shift): |d eps_zz| = {:.2e} -> {}".format(
        d7, "PASS" if g7 else "FAIL"), flush=True)
    dd = []
    for t in (10e-9, 20e-9, 40e-9):
        e = slab_model().eps({"subband": slab(t)}, 1200e-9)
        dd.append(abs(e[2, 2] - e[0, 0]))
        print("[is]   {:.0f} nm: eps_xx = {:.4f}, eps_zz = {:.4f}, |eps_zz - eps_xx| = {:.4f}".format(
            t * 1e9, e[0, 0], e[2, 2], dd[-1]), flush=True)
    g8 = bool(dd[0] > dd[1] > dd[2] and 1.6 <= dd[1] / dd[2] <= 2.4 and dd[2] < 0.017)
    ok = ok and g8
    print("[is] GATE 8 (wide-slab limit): 20->40 nm ratio {:.3f} -> {}".format(
        dd[1] / dd[2], "PASS" if g8 else "FAIL"), flush=True)

    print("[is] *** INTERSUBBAND eps_zz: {} ***".format("PASS" if ok else "FAIL"), flush=True)
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
