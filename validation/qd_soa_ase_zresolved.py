"""QD-SOA z-RESOLVED dynamic bidirectional ASE vs the lumped self-consistency + analytic reductions.
ase_self_consistent_zresolved refines the lumped ase_self_consistent so EACH slice's gain is
saturated by its OWN local bidirectional-ASE photon density (not one device-averaged S_ase): the
coupled fixed point g(z,nu) <-> S_f(z,nu), S_b(z,nu) <-> S_ase(z), iterated to convergence. The
forward ASE grows toward z=L and the backward toward z=0, so S_ase(z) and the gain depression carry a
real z-PROFILE the lumped model averages into one number. MAGNITUDE / SCOPE: the QD gain is stiff, so
the ASE back-action is WEAK (local gain depression ~1e-4 relative, spatial spread ~1e-4); the
device-INTEGRATED output therefore depends essentially only on mean(S_ase(z)) and matches the lumped
model (GATE D) -- the refinement is the spatial PROFILE, not the aggregate output. The refinement is
purely LONGITUDINAL; the per-slice ASE saturates via the signal-frequency line filter, so spectral
saturation stays lumped.

GATE A (frozen reduction): ase_saturation=False -> one pass identical to ase_spectrum_bidirectional on
        the signal-only gain (no back-action).
GATE B (self-consistent fixed point): at convergence each slice's returned gain g_sat_z[z] ==
        material_gain(steady_state(I, S_signal + ase_strength S_ase_z[z])) to machine -- the returned
        state IS the converged fixed point.
GATE C (the z-PROFILE is real): the converged S_ase(z) is non-uniform (a real z-profile), and the
        local gain depression is PERFECTLY anti-correlated with it (corr(S_ase_z, g_z) -> -1) -- the
        gain is depressed exactly where the bidirectional ASE flux peaks, the spatial information the
        lumped single-S_ase model cannot represent.
GATE D (consistent with the lumped aggregate): the z-resolved device-OUTPUT ASE spectrum agrees with
        the lumped ase_self_consistent (independent code path, one averaged S_ase) to high precision
        -- the refinement adds the profile WITHOUT changing the device-integrated output.
GATE E (negative feedback + passivity): a larger ASE load lowers the mean saturated gain (monotone),
        every solve converges, and no NaN appears.

Run: python -m validation.qd_soa_ase_zresolved
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dynameta.optics.soa.ase_noise import (ase_self_consistent, ase_self_consistent_zresolved,
                                           ase_spectrum_bidirectional)
from dynameta.optics.soa.qd_gain import QDGainModel, QDGainParams


def main():
    print("[az] === QD-SOA z-resolved dynamic bidirectional ASE vs lumped + reductions ===",
          flush=True)
    ok = True
    m = QDGainModel(QDGainParams(n_groups=21).with_detailed_balance_taus())
    nu0 = m.p.nu0_Hz
    nu = np.linspace(nu0 - 4e12, nu0 + 4e12, 31)
    dnu = np.gradient(nu)
    k0 = int(np.argmin(np.abs(nu - nu0)))
    L, I, nz, strn = 2.0e-3, 80e-3, 30, 200.0

    # GATE A: frozen reduction (no back-action)
    off = ase_self_consistent_zresolved(m, I, 0.0, nu0, nu, dnu, L, n_slices=nz,
                                        ase_saturation=False, m_pol=2)
    y0 = m.steady_state(I, S_conf_m3=0.0, nu_s_Hz=nu0)
    g0 = m.material_gain_per_m(m.rho_GS(y0), nu)
    gsp0 = m.emission_gain_per_m(m.rho_GS(y0), nu)
    ref = ase_spectrum_bidirectional(np.tile(g0, (nz, 1)), np.tile(gsp0, (nz, 1)), L / nz, nu, dnu,
                                     m.p.Gamma, m_pol=2)
    relA = float(np.max(np.abs(off["S_f"] - ref["S_f"]) / np.maximum(np.abs(ref["S_f"]), 1e-300)))
    g_a = bool(relA < 1e-13)
    ok = ok and g_a
    print("[az] GATE A: ase_saturation=False == frozen ase_spectrum_bidirectional (max rel {:.1e}) -> "
          "{}".format(relA, "PASS" if g_a else "FAIL"), flush=True)

    # converged z-resolved solve (used by B, C)
    on = ase_self_consistent_zresolved(m, I, 0.0, nu0, nu, dnu, L, n_slices=nz, ase_saturation=True,
                                       ase_strength=strn, m_pol=2, beta=0.4, max_iter=200)
    Sz = on["S_ase_z"]
    gpk = on["g_sat_z"][:, k0]

    # GATE B: self-consistent fixed point
    recomputed = np.array([m.material_gain_per_m(
        m.rho_GS(m.steady_state(I, S_conf_m3=strn * float(s), nu_s_Hz=nu0)), nu0) for s in Sz])
    relB = float(np.max(np.abs(gpk - recomputed)))
    g_b = bool(on["converged"] and relB < 1e-9)
    ok = ok and g_b
    print("[az] GATE B: converged g_sat_z == material_gain(steady_state(I, signal+ase S_ase_z)) per "
          "slice (max abs {:.1e}, {} iters) -> {}".format(relB, on["n_iter"],
                                                          "PASS" if g_b else "FAIL"), flush=True)

    # GATE C: the z-profile is real + perfectly anti-correlated with the local gain
    var = float((Sz.max() - Sz.min()) / Sz.mean())
    corr = float(np.corrcoef(Sz, gpk)[0, 1])
    rel_dep = float((on["g_unsat"][k0] - gpk.min()) / on["g_unsat"][k0])   # weak (stiff QD gain)
    g_c = bool(var > 0.1 and corr < -0.99)
    ok = ok and g_c
    print("[az] GATE C: S_ase(z) profile var {:.1%} (>10%) AND corr(S_ase_z, g_z) {:.4f} (-> -1, gain "
          "depressed where ASE peaks); back-action is WEAK (peak gain depression {:.1e} rel) -> "
          "{}".format(var, corr, rel_dep, "PASS" if g_c else "FAIL"), flush=True)

    # GATE D: agrees with the lumped aggregate output
    lump = ase_self_consistent(m, I, 0.0, nu0, nu, dnu, L, n_slices=nz, ase_saturation=True,
                               ase_strength=strn, m_pol=2, beta=0.4, max_iter=200)
    relD = float(np.max(np.abs(on["S_f_out"] - lump["S_f_out"])
                        / np.maximum(np.abs(lump["S_f_out"]), 1e-300)))
    g_d = bool(relD < 1e-5)
    ok = ok and g_d
    print("[az] GATE D: z-resolved device output == lumped ase_self_consistent (max rel {:.1e}) -> "
          "{}".format(relD, "PASS" if g_d else "FAIL"), flush=True)

    # GATE E: negative feedback (more ASE -> lower mean gain) + passivity
    res_lo = ase_self_consistent_zresolved(m, I, 0.0, nu0, nu, dnu, L, n_slices=nz,
                                           ase_saturation=True, ase_strength=50.0, m_pol=2,
                                           beta=0.4, max_iter=200)
    mean_lo = float(np.mean(res_lo["g_sat_z"][:, k0]))
    mean_hi = float(np.mean(gpk))
    nan_free = not (np.any(np.isnan(Sz)) or np.any(np.isnan(on["g_sat_z"])))
    g_e = bool(mean_hi < mean_lo and res_lo["converged"] and nan_free)
    ok = ok and g_e
    print("[az] GATE E: more ASE -> lower mean gain ({:.2f} at str200 < {:.2f} at str50 /m); "
          "converged + finite {} -> {}".format(mean_hi, mean_lo, nan_free,
                                               "PASS" if g_e else "FAIL"), flush=True)

    print("[az] *** QD-SOA Z-RESOLVED ASE: {} ***".format("PASS" if ok else "FAIL"), flush=True)
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
