"""
1D FDTD optical backend oracle (roadmap C9). The time-domain Yee solver optics.fdtd.solve_fdtd_1d
gives the broadband R(f)/T(f) of a layered stack in one run; validated against the analytic
single-slab Airy formula for (A) a non-dispersive slab and (B) a DISPERSIVE Drude slab (the ADE),
and (C) a Kerr sanity check (the chi3 nonlinearity is active and reduces to linear at chi3 = 0).

GATE A: FDTD R(f), T(f) of a non-dispersive slab (n=2, 300 nm) == the analytic Airy R/T, max abs
        error < 1e-2, and R+T = 1 (lossless).
GATE B: FDTD R(f), T(f) of a Drude slab (eps = eps_inf - wp^2/(w^2 + i gamma w)) == the analytic
        complex-n Airy R/T, max abs error < 4e-2 (the ADE dispersion); R+T+A consistent.
GATE C: a Kerr slab at chi3 = 0 (kerr on) reproduces the linear T exactly (reduction); a large chi3
        at finite amplitude CHANGES T (the nonlinearity is active -- the all-optical axis).

Run: python -m validation.fdtd_layered
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dynameta.optics.fdtd import FDTDLayer, solve_fdtd_1d

C = 299792458.0
LMIN, LMAX = 1200e-9, 1800e-9


def airy_slab(f, n, d):
    """vacuum | n(complex) | vacuum single-slab R, T at normal incidence."""
    k0 = 2 * np.pi * np.asarray(f) / C
    beta = n * k0 * d
    r1 = (1.0 - n) / (1.0 + n)
    e2 = np.exp(2j * beta)
    r = r1 * (1.0 - e2) / (1.0 - r1 ** 2 * e2)
    t = (1.0 - r1 ** 2) * np.exp(1j * beta) / (1.0 - r1 ** 2 * e2)
    return np.abs(r) ** 2, np.abs(t) ** 2


def main():
    print("[fd] === 1D FDTD optical backend (broadband R/T, Drude ADE, Kerr) ===", flush=True)

    # GATE A: non-dispersive slab vs analytic
    n, d = 2.0, 300e-9
    rA = solve_fdtd_1d([FDTDLayer(thickness_m=d, eps_inf=n ** 2)],
                       lambda_min_m=LMIN, lambda_max_m=LMAX, resolution=40)
    fA = rA.freqs_Hz[rA.band]
    Ra, Ta = airy_slab(fA, n, d)
    dRA, dTA = float(np.max(np.abs(rA.R[rA.band] - Ra))), float(np.max(np.abs(rA.T[rA.band] - Ta)))
    enA = float(np.max(np.abs(rA.R[rA.band] + rA.T[rA.band] - 1.0)))
    gate_a = bool(dRA < 1e-2 and dTA < 1e-2 and enA < 1e-2)
    print("[fd] A non-disp: max|dR|={:.2e} max|dT|={:.2e} max|R+T-1|={:.2e}".format(dRA, dTA, enA),
          flush=True)

    # GATE B: Drude slab vs analytic complex-n Airy (the ADE)
    eps_inf, wp, gam, dd = 4.0, 1.2e15, 2.0e13, 120e-9
    rB = solve_fdtd_1d([FDTDLayer(thickness_m=dd, eps_inf=eps_inf, drude_wp_rad_s=wp,
                                  drude_gamma_rad_s=gam)],
                       lambda_min_m=LMIN, lambda_max_m=LMAX, resolution=60)
    fB = rB.freqs_Hz[rB.band]
    w = 2 * np.pi * fB
    eps_drude = eps_inf - wp ** 2 / (w ** 2 + 1j * gam * w)
    nB = np.sqrt(eps_drude)                                   # Im(eps)>0 -> Im(n)>0 (passive)
    Rb, Tb = airy_slab(fB, nB, dd)
    dRB, dTB = float(np.max(np.abs(rB.R[rB.band] - Rb))), float(np.max(np.abs(rB.T[rB.band] - Tb)))
    gate_b = bool(dRB < 4e-2 and dTB < 4e-2)
    print("[fd] B Drude: n in [{:.2f},{:.2f}] ; max|dR|={:.2e} max|dT|={:.2e}".format(
        float(nB.real.min()), float(nB.real.max()), dRB, dTB), flush=True)

    # GATE C: Kerr -- reduces at chi3=0, active at large chi3
    base = dict(lambda_min_m=LMIN, lambda_max_m=LMAX, resolution=40, source_amp=3.0)
    lin = solve_fdtd_1d([FDTDLayer(thickness_m=400e-9, eps_inf=4.0)], kerr=False, **base)
    k0_ = solve_fdtd_1d([FDTDLayer(thickness_m=400e-9, eps_inf=4.0, chi3_m2_V2=0.0)], kerr=True, **base)
    kbig = solve_fdtd_1d([FDTDLayer(thickness_m=400e-9, eps_inf=4.0, chi3_m2_V2=0.3)],
                         kerr=True, **base)
    m = lin.band & k0_.band & kbig.band
    reduce_ok = bool(np.max(np.abs(k0_.T[m] - lin.T[m])) < 1e-9)        # chi3=0 -> identical
    active = float(np.max(np.abs(kbig.T[m] - lin.T[m])))
    active_ok = bool(active > 2e-2)                                     # large chi3 -> changes T
    gate_c = reduce_ok and active_ok
    print("[fd] C Kerr: chi3=0 reduces (max|dT|<1e-9)={} ; large-chi3 active max|dT|={:.3e}".format(
        reduce_ok, active), flush=True)

    print("[fd] GATE A (non-dispersive vs analytic): {}".format("PASS" if gate_a else "FAIL"),
          flush=True)
    print("[fd] GATE B (Drude ADE vs analytic complex-n): {}".format("PASS" if gate_b else "FAIL"),
          flush=True)
    print("[fd] GATE C (Kerr active + reduces): {}".format("PASS" if gate_c else "FAIL"), flush=True)
    overall = gate_a and gate_b and gate_c
    print("[fd] *** 1D FDTD OPTICAL BACKEND: {} ***".format("PASS" if overall else "FAIL"),
          flush=True)
    return overall


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
