"""REL4 ITO de-doping -> ENZ drift oracle. dn/dt = -lambda(T)(n - n_min) and the ENZ crossing tracked
on the ACTUAL DrudeOptical model.

GATE A (reduces-to-closed-form + off-switch): lambda0 = 0 -> n(t) == n0 EXACTLY (byte-identical off);
        the constant-T closed form matches a scipy solve_ivp integration of the rate ODE.
GATE B (ENZ crossing, constant mass): the numeric brentq Re(eps)=0 crossing on DrudeOptical equals
        the exact relation lambda_ENZ = 2 pi c / sqrt(wp^2/eps_inf - gamma^2) to ~1e-10, and the
        finite-difference sensitivity d(lambda_ENZ)/dn matches the roadmap's -(1/2) lambda/n.
GATE C (Kane reduction factor -- the roadmap REL4 refinement, validated quantitatively): with the
        library's KaneOpticalMass m(n), the finite-difference sensitivity matches
        -(1/2)(lambda/n)(1 - dln m/dln n) and is SMALLER in magnitude than the constant-m value.
INFO: the drift for a 1%/5% carrier loss at the ENZ operating point.

Run: python -m validation.reliability_dedoping
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dynameta.constants import M_E, Q_E, EPS0, C_LIGHT
from dynameta.materials.optical_model import DrudeOptical
from dynameta.materials.scattering import KaneOpticalMass
from dynameta.reliability.dedoping import DedopingParams, carrier_decay, enz_wavelength_m

EPS_INF, GAM = 4.0, 1.0e13                               # small gamma -> clean analytic crossing
N0 = 9.0e26


def main():
    print("[rd] === REL4 ITO de-doping -> ENZ drift ===", flush=True)
    ok = True

    # ---- GATE A: off-switch + closed form vs ODE ----
    t = np.linspace(0.0, 5.0e7, 11)
    off = carrier_decay(t, 350.0, n0_m3=N0, params=DedopingParams(lambda0_per_s=0.0))
    g_off = bool(np.all(off == N0))                       # EXACT (no exp round-trip)
    p = DedopingParams(lambda0_per_s=5.0e21, Ea_eV=2.0, n_min_m3=1.0e26)
    n_cf = carrier_decay(t, 420.0, n0_m3=N0, params=p)
    from scipy.integrate import solve_ivp
    lam = float(p.rate_per_s(420.0))
    sol = solve_ivp(lambda tt, y: [-lam * (y[0] - p.n_min_m3)], (0.0, float(t[-1])), [N0],
                    t_eval=t, rtol=1e-11, atol=1.0)
    rel_ode = float(np.max(np.abs(sol.y[0] - n_cf) / N0))
    g_a = bool(g_off and rel_ode < 1e-8 and n_cf[-1] < N0)
    ok = ok and g_a
    print("[rd] GATE A: off-switch n(t)==n0 exact = {}; closed form vs solve_ivp rel {:.1e} "
          "(decay {:.3e} -> {:.3e}) -> {}".format(g_off, rel_ode, N0, float(n_cf[-1]),
                                                  "PASS" if g_a else "FAIL"), flush=True)

    # ---- GATE B: ENZ crossing + constant-m sensitivity ----
    m0 = 0.30 * M_E
    dr = DrudeOptical(eps_inf=EPS_INF, m_opt_kg=m0, gamma_rad_s=GAM)
    lam_num = enz_wavelength_m(dr, N0)
    wp2 = N0 * Q_E * Q_E / (EPS0 * m0)
    lam_an = 2.0 * np.pi * C_LIGHT / np.sqrt(wp2 / EPS_INF - GAM ** 2)
    rel_cross = abs(lam_num - lam_an) / lam_an
    dn = 1.0e-4 * N0
    sens_fd = (enz_wavelength_m(dr, N0 + dn) - enz_wavelength_m(dr, N0 - dn)) / (2.0 * dn)
    sens_an = -0.5 * lam_num / N0
    rel_sens = abs(sens_fd - sens_an) / abs(sens_an)
    g_b = bool(rel_cross < 1e-9 and rel_sens < 1e-2)
    ok = ok and g_b
    print("[rd] GATE B: crossing {:.2f} nm vs exact {:.2f} nm (rel {:.1e}); d(lam)/dn FD {:.3e} vs "
          "-(1/2)lam/n {:.3e} (rel {:.1e}) -> {}".format(
              lam_num * 1e9, lam_an * 1e9, rel_cross, sens_fd, sens_an, rel_sens,
              "PASS" if g_b else "FAIL"), flush=True)

    # ---- GATE C: Kane n-dependent mass REDUCES the drift sensitivity by (1 - dln m/dln n) ----
    kane = KaneOpticalMass(m0_kg=0.25 * M_E, alpha_eV=0.5)
    drk = DrudeOptical(eps_inf=EPS_INF, m_opt_kg=kane, gamma_rad_s=GAM)
    lam_k = enz_wavelength_m(drk, N0)
    sens_k_fd = (enz_wavelength_m(drk, N0 + dn) - enz_wavelength_m(drk, N0 - dn)) / (2.0 * dn)
    dlnm_dlnn = float((np.log(kane(N0 + dn)) - np.log(kane(N0 - dn)))
                      / (np.log(N0 + dn) - np.log(N0 - dn)))
    sens_k_an = -0.5 * (lam_k / N0) * (1.0 - dlnm_dlnn)
    rel_k = abs(sens_k_fd - sens_k_an) / abs(sens_k_an)
    g_c = bool(rel_k < 1e-2 and abs(sens_k_fd) < abs(-0.5 * lam_k / N0) and dlnm_dlnn > 0.0)
    ok = ok and g_c
    print("[rd] GATE C: Kane dln m/dln n = {:.4f}; d(lam)/dn FD {:.3e} vs -(1/2)(lam/n)(1-dlnm) "
          "{:.3e} (rel {:.1e}); |Kane| < |const-m| = {} -> {}".format(
              dlnm_dlnn, sens_k_fd, sens_k_an, rel_k, abs(sens_k_fd) < abs(-0.5 * lam_k / N0),
              "PASS" if g_c else "FAIL"), flush=True)

    # INFO: drift per carrier loss at the operating point
    for loss in (0.01, 0.05):
        d_nm = (enz_wavelength_m(dr, N0 * (1.0 - loss)) - lam_num) * 1e9
        print("[rd] INFO: {:.0f}% carrier loss -> ENZ red-shift {:+.2f} nm".format(loss * 100, d_nm),
              flush=True)

    print("[rd] *** REL4 DE-DOPING / ENZ DRIFT: {} ***".format("PASS" if ok else "FAIL"), flush=True)
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
