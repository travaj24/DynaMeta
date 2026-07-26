"""
Burstein-Moss band-filling + bandgap-renormalization edge oracle (roadmap R8). BursteinMossEdge is a
carrier-density-dependent interband DELTA composed (through DeltaEffect) on top of the bare Drude:
band filling blueshifts the optical gap Eg_opt(n) = Eg0 - dE_BGR(n) + dE_BM(n), shifting the below-gap
refractive index (and thus the exact ENZ crossing) and opening interband absorption above Eg_opt.

GATE A -- REDUCES TO BARE DRUDE: at n = n_ref the DeltaEffect contribution is identically 0, so
        ComposedEffect(Drude, [DeltaEffect(bm, {'n': n_ref})]).eps({'n': n_ref}) == DrudeOptical.eps to
        < 1e-12; with bm.enabled=False the composed eps == bare Drude at ALL n to ~1e-15 (the true
        byte-identical off-switch).

GATE B -- INDEPENDENT ANALYTIC + PUBLISHED-TREND REFERENCE:
        (1) the model's Burstein-Moss shift equals the standalone closed form
            dE_BM(n) = (hbar^2/2)(1/m_vc)(3 pi^2 n)^(2/3) to relative 1e-10 (re-derived here, not the
            model's code path for the comparison value);
        (2) PUBLISHED MAGNITUDE/TREND (the roadmap's "hundreds of meV at n~1e27"): for ITO with a
            reduced joint mass m_vc ~ 0.5 m_e, dE_BM(1e27 m^-3) lands in the 0.3-0.8 eV band and the
            blueshift increases monotonically with n (degenerate-doping trend);
        (3) ENZ crossing: with the delta OFF the composed Re(eps) zero-crossing equals the bare-Drude
            crossing exactly; with the delta ON it shifts (the "corrects the exact ENZ wavelength"
            claim) -- reported, with passivity (Im(eps) >= 0) asserted.

GATE C -- KRAMERS-KRONIG TRUNCATION vs scipy.quad TO INFINITY (audit R-2). The probe sits BELOW the
        optical gap, so the KK integral has no pole in range and the substitution E' = Eg/t lets
        scipy.quad evaluate it EXACTLY, tail to infinity included -- the oracle the audit prescribes
        (not "a wider grid", which cannot see a common truncation). Assert
        (1) the model's absolute dn matches that oracle to < 2% (the shipped default, whose auto grid
            stopped 5 eV above the edge, was 44-50% LOW: the cutoff-free Tauc tail leaves an
            E_hi^-1.5 remainder, so the observed halving-per-doubling IS the convergence law);
        (2) the DeltaEffect DIFFERENCE matches the difference of two separately-converged quad values
            to < 1e-4 of the dn scale. The shipped grid was rebuilt PER eps() CALL from the per-call
            Eg_opt(n), so a DeltaEffect integrated its two halves on DIFFERENT grids, the large
            common truncation did not cancel, and the delta came out 14-16x too large. The grid is
            now a pure function of the configuration, so both halves share it;
        (3) the grid object is literally reused across n and lambda (pinning, by identity);
        (4) a grid of the SHIPPED width now trips the tail test -- the old guard only checked that
            the grid BRACKETED the gap and the probe, and accepted a grid ending 1 meV above Eg.

Run: python -m validation.burstein_moss_blueshift
"""

import os
import sys
import warnings

import numpy as np
from scipy.integrate import quad

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dynameta.constants import HBAR, M_E, Q_E
from dynameta.core.effects.base import _photon_energy_J
from dynameta.core.effects import (BursteinMossEdge, OpticalModelEffect, ComposedEffect, DeltaEffect,
                                   as_tensor)
from dynameta.materials.optical_model import DrudeOptical

EPS_INF, M_OPT, GAMMA = 4.25, 0.225 * M_E, 1.1e14      # reference ITO Drude
EG0 = 3.6 * Q_E                                        # undoped optical gap [J]
M_VC = 0.5 * M_E                                       # reduced joint conduction-valence mass
N_REF = 4.0e26                                         # n_bg (DeltaEffect baseline)
ALPHA_EDGE = 1.5                                       # dimensionless interband edge amplitude (Tauc p=0.5)


def _crossing_nm(eps_re_of_lambda, lams):
    re = np.array([eps_re_of_lambda(l) for l in lams])
    s = np.where(np.diff(np.sign(re)) != 0)[0]
    if not len(s):
        return None
    i = s[0]
    # audit R-2 fallout: this used np.interp(0.0, [re[i], re[i+1]], ...), whose xp is DECREASING
    # here (Re eps falls through zero with lambda). np.interp requires increasing xp and silently
    # returned the left endpoint instead, quantizing the crossing to the lambda grid -- the old
    # "shift -1.224 nm" was exactly one grid spacing (1100 nm / 899), i.e. a quantization artefact,
    # not a measurement. Interpolate explicitly, which is direction-agnostic.
    lo, hi = float(re[i]), float(re[i + 1])
    t = (0.0 - lo) / (hi - lo)
    return float((lams[i] + t * (lams[i + 1] - lams[i])) * 1e9)


def main():
    print("[bm] === Burstein-Moss + bandgap-renormalization interband edge ===", flush=True)
    ok = True

    drude = DrudeOptical(eps_inf=EPS_INF, m_opt_kg=M_OPT, gamma_rad_s=GAMMA)
    bg = OpticalModelEffect(drude)
    bm = BursteinMossEdge(eps_inf=EPS_INF, Eg0_J=EG0, m_vc_kg=M_VC, alpha_edge=ALPHA_EDGE,
                          tauc_exponent=0.5)
    comp = ComposedEffect(background=bg, deltas=[DeltaEffect(bm, baseline_fields={"n": N_REF})])

    # ---- GATE A: reduces to bare Drude ----
    lam = 1300e-9
    at_ref = comp.eps({"n": N_REF}, lam)
    drude_ref = as_tensor(np.asarray(complex(drude.eps(lam, n_m3=N_REF))))
    dA1 = float(np.max(np.abs(at_ref - drude_ref)))
    bm_off = BursteinMossEdge(eps_inf=EPS_INF, Eg0_J=EG0, m_vc_kg=M_VC, alpha_edge=ALPHA_EDGE,
                              enabled=False)
    comp_off = ComposedEffect(background=bg, deltas=[DeltaEffect(bm_off, baseline_fields={"n": N_REF})])
    ns = np.array([2e26, 6e26, 1e27])
    dA2 = max(float(np.max(np.abs(comp_off.eps({"n": float(n)}, lam)
                                  - as_tensor(np.asarray(complex(drude.eps(lam, n_m3=n))))))) for n in ns)
    g_a = bool(dA1 < 1e-12 and dA2 < 1e-14)
    ok = ok and g_a
    print("[bm] GATE A: at n_ref == Drude max|d|={:.1e}; enabled=False == Drude (all n) max|d|={:.1e} "
          "-> {}".format(dA1, dA2, "PASS" if g_a else "FAIL"), flush=True)

    # ---- GATE B(1): analytic Burstein-Moss shift (independent re-derivation) ----
    n_chk = np.geomspace(1e26, 1e27, 6)
    dE_an = (HBAR ** 2 / 2.0) * (1.0 / M_VC) * (3.0 * np.pi ** 2 * n_chk) ** (2.0 / 3.0)
    dE_model = bm.gap_shift_J(n_chk)
    rel = float(np.max(np.abs(dE_model - dE_an) / dE_an))
    g_b1 = bool(rel < 1e-10)
    # ---- GATE B(2): published magnitude + monotone trend ----
    dE_1e27_eV = float((HBAR ** 2 / 2.0) * (1.0 / M_VC) * (3.0 * np.pi ** 2 * 1e27) ** (2.0 / 3.0) / Q_E)
    mono = bool(np.all(np.diff(dE_an) > 0))
    g_b2 = bool(0.3 < dE_1e27_eV < 0.8 and mono)
    ok = ok and g_b1 and g_b2
    print("[bm] GATE B1: dE_BM(model) == analytic rel={:.1e}; B2: dE_BM(1e27)={:.3f} eV in [0.3,0.8], "
          "monotone={} -> {}".format(rel, dE_1e27_eV, mono, "PASS" if (g_b1 and g_b2) else "FAIL"),
          flush=True)
    for nn, ee in zip(n_chk, dE_an / Q_E):
        print("[bm]   n={:.2e} m^-3 -> dE_BM = {:.3f} eV".format(nn, ee), flush=True)

    # ---- GATE B(3): ENZ crossing shift (delta off == Drude; on shifts) + passivity ----
    lams = np.linspace(900e-9, 2000e-9, 900)
    n_acc = 7.0e26                                                  # accumulation density
    enz_drude = _crossing_nm(lambda l: complex(drude.eps(l, n_m3=n_acc)).real, lams)
    enz_off = _crossing_nm(lambda l: comp_off.eps({"n": n_acc}, l)[0, 0].real, lams)
    enz_on = _crossing_nm(lambda l: comp.eps({"n": n_acc}, l)[0, 0].real, lams)
    im_ok = bool(np.all(np.array([comp.eps({"n": n_acc}, l)[0, 0].imag for l in lams]) > -1e-12))
    off_exact = (enz_drude is not None and enz_off is not None and abs(enz_off - enz_drude) < 1e-6)
    shift = (enz_on - enz_off) if (enz_on is not None and enz_off is not None) else None
    g_b3 = bool(off_exact and im_ok and shift is not None and abs(shift) > 1e-6)
    ok = ok and g_b3
    fmt = lambda v: "{:.2f}".format(v) if v is not None else "none"
    print("[bm] GATE B3: ENZ crossing Drude={} nm, delta-off={} nm (==Drude), delta-on={} nm (shift "
          "{:+.3f} nm); passive={} -> {}".format(
              fmt(enz_drude), fmt(enz_off), fmt(enz_on), shift if shift is not None else float("nan"),
              im_ok, "PASS" if g_b3 else "FAIL"), flush=True)

    # ---- GATE C: Kramers-Kronig truncation vs scipy.quad to infinity (audit R-2) ----
    n_re0 = np.sqrt(complex(EPS_INF)).real

    def dn_quad(Eg_J, E_ph_J, p=0.5):
        """EXACT dn = (1/pi) int_Eg^inf E' eps2(E')/(E'^2 - E^2) dE' via E' = Eg/t, t in (0, 1]."""
        def f(t):
            Ep = Eg_J / t
            return ((((Ep - Eg_J) / Eg_J) ** p * (Eg_J / Ep) ** 2 * Ep / (Ep ** 2 - E_ph_J ** 2))
                    * (Eg_J / t ** 2))
        return ALPHA_EDGE / np.pi * quad(f, 1e-14, 1.0, limit=600)[0]

    def dn_model(n_m3, lam_m):
        e = bm.eps({"n": np.asarray(float(n_m3))}, lam_m)
        return float(np.sqrt(complex(e)).real - n_re0)

    worst_abs, worst_delta, dn_scale = 0.0, 0.0, bm._kk_dn_scale()
    for lam_probe in (1300e-9, 1550e-9):
        E_ph = _photon_energy_J(lam_probe)
        d_ref_m, d_ref_q = dn_model(N_REF, lam_probe), dn_quad(
            float(bm.optical_gap_J(np.asarray(N_REF))), E_ph)
        for n in (1e26, 7e26, 1e27):
            Eg_n = float(bm.optical_gap_J(np.asarray(float(n))))
            q, m = dn_quad(Eg_n, E_ph), dn_model(n, lam_probe)
            worst_abs = max(worst_abs, abs(m / q - 1.0))
            worst_delta = max(worst_delta, abs((m - d_ref_m) - (q - d_ref_q)) / dn_scale)
            print("[bm]   lam={:.0f} nm n={:.1e}: dn model={:+.7f} quad={:+.7f} (rel {:.1e}); "
                  "delta vs n_ref model={:+.7f} quad={:+.7f}".format(
                      lam_probe * 1e9, n, m, q, abs(m / q - 1.0), m - d_ref_m, q - d_ref_q), flush=True)
    grid_id = bm._kk_grid()
    bm.eps({"n": np.asarray(2e26)}, 1550e-9)
    pinned = bm._kk_grid() is grid_id
    with warnings.catch_warnings(record=True) as wrec:
        warnings.simplefilter("always")
        BursteinMossEdge(eps_inf=EPS_INF, Eg0_J=EG0, m_vc_kg=M_VC, alpha_edge=ALPHA_EDGE,
                         e_grid_J=(1e-21, 9.0 * Q_E, 3001)).eps({"n": np.asarray(N_REF)}, 1300e-9)
        tail_test = any("Kramers-Kronig grid ends" in str(x.message) for x in wrec)
    g_c = bool(worst_abs < 2e-2 and worst_delta < 1e-4 and pinned and tail_test)
    ok = ok and g_c
    print("[bm] GATE C: |dn - quad|/quad worst = {:.2e} (< 2e-2); |delta - quad delta|/dn_scale worst "
          "= {:.2e} (< 1e-4); grid pinned across n/lambda = {}; shipped-width grid trips the tail "
          "test = {} -> {}".format(worst_abs, worst_delta, pinned, tail_test,
                                   "PASS" if g_c else "FAIL"), flush=True)

    print("[bm] *** BURSTEIN-MOSS EDGE: {} ***".format("PASS" if ok else "FAIL"), flush=True)
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
