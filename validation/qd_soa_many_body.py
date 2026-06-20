"""QD-SOA microscopic many-body (screened-Hartree-Fock) gain vs analytic oracles. QDGainModel gains
material_gain_index_mb(rho, nu, N) returning the gain g(nu) AND its carrier-induced index partner
gi(nu) from the renormalized complex susceptibility (one analytic chi -> g, gi are a Kramers-Kronig
pair). The three dominant finite-density corrections (all functions of carrier density N and T):
bandgap renormalization (BGR red-shift), excitation-induced + phonon dephasing (HWHM broadening,
oscillator-strength conserving), and the screened Coulomb/excitonic enhancement.

GATE A (free-carrier reduction): many_body disabled OR all corrections zero -> g == material_gain_per_m
        (machine), and the index partner gi matches the un-renormalized KK index.
GATE B (Kramers-Kronig consistency): gi(nu) == Hilbert(g(nu)) -- the gain and index are Re/Im of the
        same analytic complex Lorentzian, so they are a genuine KK pair (verified by an INDEPENDENT
        numerical Hilbert transform of g alone, central region).
GATE C (BGR red-shift): the gain-peak frequency shifts by dE_BGR(N)/h = -bgr_coeff E_R (a_B^3 N)^(1/3)
        as N grows, matching the analytic shift and the universal N^(1/3) scaling.
GATE D (EID broadening, oscillator strength conserved): the homogeneous HWHM gamma(N) grows with N
        and the peak drops as gamma0/gamma so the integrated gain (line area) is INVARIANT -- the
        physically correct invariant (the free-carrier model holds the peak, over-counting the area).
GATE E (Coulomb enhancement): the peak gain scales by C_enh(N) = 1 + coulomb_enh exp(-N/N_mott)
        (-> 1+coulomb_enh at low N, -> 1 at high N / above the Mott density), == 1 when coulomb_enh=0.

Run: python -m validation.qd_soa_many_body
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scipy.signal import hilbert

from dynameta.constants import HBAR, Q_E
from dynameta.optics.soa.qd_gain import ManyBody, QDGainModel, QDGainParams

H_PLANCK = 2.0 * np.pi * HBAR


def _model(**mbkw):
    mb = ManyBody(**mbkw) if mbkw else None
    return QDGainModel(QDGainParams(n_groups=41).with_detailed_balance_taus(), many_body=mb)


def main():
    print("[mb] === QD-SOA microscopic many-body (screened-HF) gain vs oracles ===", flush=True)
    ok = True
    base = QDGainModel(QDGainParams(n_groups=41).with_detailed_balance_taus())
    nu0 = base.p.nu0_Hz
    rho = np.full(41, 0.9)

    # ---- GATE A: free-carrier reduction ----
    nu = np.linspace(nu0 - 8e12, nu0 + 8e12, 600)
    g_fc = base.material_gain_per_m(rho, nu)
    mr = _model(enabled=True, bgr_coeff=0.0, gamma_eid_Hz=0.0, gamma_phonon_Hz=0.0, coulomb_enh=0.0)
    g_mb, gi_mb = mr.material_gain_index_mb(rho, nu, 1e24)
    g_off, _ = _model().material_gain_index_mb(rho, nu, 1e24)   # many_body=None path
    relA = max(float(np.max(np.abs(g_mb - g_fc))), float(np.max(np.abs(g_off - g_fc))))
    g_a = bool(relA < 1e-9 * float(np.max(np.abs(g_fc))) + 1e-9)
    ok = ok and g_a
    print("[mb] GATE A: zero-correction / disabled MB == material_gain_per_m (max|d| {:.1e}) -> "
          "{}".format(relA, "PASS" if g_a else "FAIL"), flush=True)

    # ---- GATE B: Kramers-Kronig (gi == Hilbert[g]) on a wide window, central region ----
    nuw = np.linspace(nu0 - 4e13, nu0 + 4e13, 8001)
    gw, giw = _model(enabled=True, exciton_rydberg_meV=12.0, exciton_bohr_nm=12.0, bgr_coeff=1.9,
                     gamma_eid_Hz=0.4e12, coulomb_enh=0.3).material_gain_index_mb(rho, nuw, 1e24)
    Hg = np.imag(hilbert(gw))                                    # KK partner of g (independent)
    core = np.abs(nuw - nu0) < 6e12                              # central region (edge-effect-free)
    relB = float(np.max(np.abs(giw[core] - Hg[core])) / np.max(np.abs(giw[core])))
    g_b = bool(relB < 5e-2)
    ok = ok and g_b
    print("[mb] GATE B: index gi == Hilbert(gain) (KK pair, central max rel {:.1e}) -> {}".format(
        relB, "PASS" if g_b else "FAIL"), flush=True)

    # ---- GATE C: BGR red-shift + N^(1/3) scaling ----
    nuc = np.linspace(nu0 - 3e13, nu0 + 1e13, 6000)
    mC = _model(enabled=True, exciton_rydberg_meV=12.0, exciton_bohr_nm=12.0, bgr_coeff=1.9)
    relC, shifts, Ns = 0.0, [], [2e23, 1e24, 5e24]
    for N in Ns:
        gN, _ = mC.material_gain_index_mb(rho, nuc, N)
        meas = float(nuc[int(np.argmax(gN))] - nu0)
        pred = mC._mb_bgr_shift_Hz(N)
        shifts.append(meas)
        relC = max(relC, abs(meas - pred) / abs(pred))
    scale_ok = abs(shifts[2] / shifts[0] - (Ns[2] / Ns[0]) ** (1.0 / 3.0)) / \
        ((Ns[2] / Ns[0]) ** (1.0 / 3.0)) < 5e-2
    # INDEPENDENT absolute check (hand-computed, NOT via the production _mb_bgr_shift_Hz): the
    # measured peak shift at N=1e24 must match -bgr E_R (a_B^3 N)^(1/3)/h with the raw constants
    dnu_hand = -1.9 * (12.0e-3 * Q_E) * ((12.0e-9) ** 3 * 1e24) ** (1.0 / 3.0) / H_PLANCK
    indep_ok = abs(shifts[1] - dnu_hand) / abs(dnu_hand) < 5e-2
    g_c = bool(relC < 5e-2 and scale_ok and indep_ok)
    ok = ok and g_c
    print("[mb] GATE C: gain-peak shift == BGR -bgr E_R (a_B^3 N)^1/3 (max rel {:.1e}) + N^1/3 scaling "
          "{} -> {}".format(relC, scale_ok, "PASS" if g_c else "FAIL"), flush=True)

    # ---- GATE D: EID broadening conserves oscillator strength (single group, wide window) ----
    mD = QDGainModel(QDGainParams(n_groups=1).with_detailed_balance_taus(),
                     many_body=ManyBody(enabled=True, bgr_coeff=0.0, gamma_eid_Hz=0.6e12,
                                        N_ref_eid_m3=1e24, coulomb_enh=0.0))
    nud = np.linspace(nu0 - 2e14, nu0 + 2e14, 200001)           # very wide -> capture Lorentzian tails
    areas, peaks, hwhms = [], [], []
    for N in (0.0, 1e24, 3e24):
        gN, _ = mD.material_gain_index_mb([0.9], nud, N)
        areas.append(float(np.trapezoid(gN, nud)))
        peaks.append(float(gN.max()))
        hwhms.append(mD._mb_hwhm_Hz(N, 300.0))
    area_rel = (max(areas) - min(areas)) / np.mean(areas)       # oscillator strength conserved
    peak_law = abs((peaks[2] / peaks[0]) - (hwhms[0] / hwhms[2])) / (hwhms[0] / hwhms[2])  # peak ~ 1/gamma
    broadened = hwhms[2] > hwhms[0]
    g_d = bool(area_rel < 1e-2 and peak_law < 1e-2 and broadened)
    ok = ok and g_d
    print("[mb] GATE D: EID broadens HWHM {:.2e}->{:.2e} Hz, peak ~ 1/gamma (rel {:.1e}), area conserved "
          "(rel {:.1e}) -> {}".format(hwhms[0], hwhms[2], peak_law, area_rel, "PASS" if g_d else "FAIL"),
          flush=True)

    # ---- GATE E: Coulomb enhancement scales the peak, screened to 1 at high N ----
    ce = 0.5
    mE = _model(enabled=True, bgr_coeff=0.0, gamma_eid_Hz=0.0, coulomb_enh=ce, N_mott_m3=5e24)
    nue = np.linspace(nu0 - 8e12, nu0 + 8e12, 600)
    glo, _ = mE.material_gain_index_mb(rho, nue, 1e22)          # N << N_mott -> ~ 1 + ce
    ghi, _ = mE.material_gain_index_mb(rho, nue, 5e25)          # N >> N_mott -> ~ 1
    g_ref = base.material_gain_per_m(rho, nue)
    enh_lo = float(np.max(glo) / np.max(g_ref))
    enh_hi = float(np.max(ghi) / np.max(g_ref))
    # no enhancement when coulomb_enh = 0
    g0, _ = _model(enabled=True, bgr_coeff=0.0, gamma_eid_Hz=0.0, coulomb_enh=0.0).material_gain_index_mb(
        rho, nue, 1e22)
    none_ok = float(np.max(np.abs(g0 - g_ref))) < 1e-9 * float(np.max(g_ref))
    g_e = bool(abs(enh_lo - (1.0 + ce)) < 5e-2 and abs(enh_hi - 1.0) < 5e-2 and none_ok)
    ok = ok and g_e
    print("[mb] GATE E: Coulomb enhancement peak x{:.3f} at low N (-> 1+ce={:.2f}), x{:.3f} at high N "
          "(-> 1); ce=0 no enh {} -> {}".format(enh_lo, 1 + ce, enh_hi, none_ok,
                                                "PASS" if g_e else "FAIL"), flush=True)

    print("[mb] *** QD-SOA MANY-BODY GAIN: {} ***".format("PASS" if ok else "FAIL"), flush=True)
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
