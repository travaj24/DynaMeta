"""QD-SOA reduced k-resolved Semiconductor-Bloch-Equation gain vs oracles. The microscopic many-body /
coherent-polarization model (optics.soa.sbe) solves the interband polarization p(k) per frequency; the
free-carrier (no-Coulomb) limit is the exact oracle, Coulomb gives the excitonic enhancement + BGR.

GATE A (free-carrier limit): coulomb_V0 = 0 reproduces the independent free-carrier closed form
        chi(w) = pref sum_k mu^2 (1-f_e-f_h) g_k / (hbar w - e_k + i hbar/T2) (the diagonal SBE).
GATE B (Coulomb enhancement): turning on the screened Coulomb RAISES the peak gain (the excitonic /
        Coulomb-enhancement many-body effect the free-carrier picture misses).
GATE C (transparency + sign): the gain crosses zero near the quasi-Fermi separation Eg + EFe + EFh;
        below it the medium amplifies (g > 0, population-inverted), above it it absorbs (g < 0).
GATE D (Kramers-Kronig): Re chi and Im chi are a Hilbert pair (each k-pole is causal) -- the Hilbert
        transform of Im chi reconstructs the shape of Re chi (central-region correlation ~ 1).
GATE E (BGR red-shift + passivity): the screened-exchange band-gap renormalization red-shifts the gain
        edge (Coulomb gain onset below the free-carrier onset); an UNPUMPED medium (N -> 0) only
        absorbs (g <= 0 everywhere) -- no gain without inversion.

Run: python -m validation.qd_soa_sbe
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dynameta.constants import EPS0, HBAR, KB, Q_E
from dynameta.optics.soa.sbe import reduced_sbe_susceptibility, sbe_gain_per_m


def _fc_oracle(hw_eV, Eg, m_e, m_h, N, T, T2, d, mu, nk, kf):
    me, mh = m_e * 9.1093837015e-31, m_h * 9.1093837015e-31
    mr = me * mh / (me + mh)
    kT = KB * T
    kth = np.sqrt(2.0 * mr * kT) / HBAR
    k = np.linspace(1e-3 * kth, kf * kth, nk)
    dk = k[1] - k[0]
    gk = k * dk / (2.0 * np.pi)
    EFe = kT * np.log(np.expm1(np.pi * HBAR ** 2 * N / (me * kT)))
    EFh = kT * np.log(np.expm1(np.pi * HBAR ** 2 * N / (mh * kT)))
    f_e = 1.0 / (1.0 + np.exp((HBAR ** 2 * k ** 2 / (2 * me) - EFe) / kT))
    f_h = 1.0 / (1.0 + np.exp((HBAR ** 2 * k ** 2 / (2 * mh) - EFh) / kT))
    inv = 1.0 - f_e - f_h
    ek = Eg * Q_E + HBAR ** 2 * k ** 2 / (2 * mr)
    gam = HBAR / T2
    return np.array([(mu / (EPS0 * d)) * np.sum(mu * (-inv * mu) / (w * Q_E - ek + 1j * gam) * gk)
                     for w in hw_eV])


def main():
    print("[sbe] === QD-SOA reduced k-resolved SBE gain vs oracles ===", flush=True)
    ok = True
    P = dict(Eg_eV=0.95, m_e=0.067, m_h=0.45, N_2d_m2=3.0e16, T_K=300.0, T2_s=100e-15,
             eps_r=12.5, d_qw_m=8e-9, mu_Cm=5e-29, nk=200, kmax_factor=6.0)
    hw = np.linspace(0.90, 1.12, 260)

    # ---- GATE A: free-carrier limit == independent oracle ----
    _, chi0 = reduced_sbe_susceptibility(hw, coulomb_V0=0.0, **P)
    chiref = _fc_oracle(hw, P["Eg_eV"], P["m_e"], P["m_h"], P["N_2d_m2"], P["T_K"], P["T2_s"],
                        P["d_qw_m"], P["mu_Cm"], P["nk"], P["kmax_factor"])
    relA = float(np.max(np.abs(chi0 - chiref)) / np.max(np.abs(chi0)))
    g0 = sbe_gain_per_m(hw, chi0, P["eps_r"])
    g_a = bool(relA < 1e-9)
    ok = ok and g_a
    print("[sbe] GATE A: free-carrier == independent oracle (rel {:.1e}) -> {}".format(
        relA, "PASS" if g_a else "FAIL"), flush=True)

    # ---- GATE B: Coulomb enhancement ----
    _, chiC = reduced_sbe_susceptibility(hw, coulomb_V0=3.0e-29, **P)
    gC = sbe_gain_per_m(hw, chiC, P["eps_r"])
    g_b = bool(gC.max() > g0.max() * 1.02)
    ok = ok and g_b
    print("[sbe] GATE B: Coulomb enhances peak gain (free {:.3e} -> Coulomb {:.3e}) -> {}".format(
        g0.max(), gC.max(), "PASS" if g_b else "FAIL"), flush=True)

    # ---- GATE C: transparency + sign ----
    me, mh = P["m_e"] * 9.1093837015e-31, P["m_h"] * 9.1093837015e-31
    kT = KB * P["T_K"]
    EFe = kT * np.log(np.expm1(np.pi * HBAR ** 2 * P["N_2d_m2"] / (me * kT)))
    EFh = kT * np.log(np.expm1(np.pi * HBAR ** 2 * P["N_2d_m2"] / (mh * kT)))
    htr = (P["Eg_eV"] * Q_E + EFe + EFh) / Q_E
    zc = hw[np.argmin(np.abs(g0))]
    below = g0[np.argmin(np.abs(hw - (zc - 0.02)))]
    above = g0[np.argmin(np.abs(hw - (zc + 0.03)))]
    g_c = bool(abs(zc - htr) < 0.02 and below > 0.0 > above)
    ok = ok and g_c
    print("[sbe] GATE C: transparency g~0 at {:.4f} eV vs Eg+EFe+EFh {:.4f}; gain>0 below, <0 above "
          "-> {}".format(zc, htr, "PASS" if g_c else "FAIL"), flush=True)

    # ---- GATE D: Kramers-Kronig (Re/Im Hilbert pair) ----
    from scipy.signal import hilbert
    hw_w = np.linspace(0.5, 1.6, 2048)                       # wide grid for the KK integral
    _, chiw = reduced_sbe_susceptibility(hw_w, coulomb_V0=0.0, **P)
    re_from_im = -np.imag(hilbert(np.imag(chiw)))            # KK: Re = Hilbert transform of Im
    sel = (hw_w > 0.85) & (hw_w < 1.15)                      # central region (window-edge-free)
    a = np.real(chiw)[sel] - np.real(chiw)[sel].mean()
    b = re_from_im[sel] - re_from_im[sel].mean()
    corr = float(np.sum(a * b) / np.sqrt(np.sum(a * a) * np.sum(b * b)))
    g_d = bool(corr > 0.99)
    ok = ok and g_d
    print("[sbe] GATE D: KK Re<->Im Hilbert pair (central correlation {:.4f}) -> {}".format(
        corr, "PASS" if g_d else "FAIL"), flush=True)

    # ---- GATE E: BGR red-shift + passivity ----
    onset0 = hw[np.argmax(g0 > 0.05 * g0.max())]            # free-carrier gain onset
    onsetC = hw[np.argmax(gC > 0.05 * gC.max())]            # Coulomb (BGR red-shifted) onset
    Plow = dict(P); Plow["N_2d_m2"] = 1.0e12                # ~unpumped
    _, chiL = reduced_sbe_susceptibility(hw, coulomb_V0=0.0, **Plow)
    gL = sbe_gain_per_m(hw, chiL, P["eps_r"])
    g_e = bool(onsetC <= onset0 + 1e-9 and gL.max() <= 1e-30)
    ok = ok and g_e
    print("[sbe] GATE E: BGR red-shifts onset ({:.4f} -> {:.4f} eV); unpumped only absorbs (g_max "
          "{:.2e}) -> {}".format(onset0, onsetC, gL.max(), "PASS" if g_e else "FAIL"), flush=True)

    print("[sbe] *** QD-SOA REDUCED SBE: {} ***".format("PASS" if ok else "FAIL"), flush=True)
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
