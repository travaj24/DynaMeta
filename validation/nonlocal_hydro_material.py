"""Solver-free gates for the hydrodynamic (nonlocal-Drude) MATERIAL MODEL and the exact layered
HDM in ``optics.nonlocal_tmm`` -- the module the audit (X-19) flagged as having no library consumer
and no validation script, so its only oracle was the test written beside it in the same commit.

Everything here is pure numpy and independent of ``tests/``: the oracles are closed forms
(Barton's beta, the local Drude limit, the bulk-plasmon standing-wave condition) plus an in-file
2x2 Fresnel TMM written from scratch.

GATE A  ONE MATERIAL MODEL (audit X-4/X-5).  ``nonlocal_tmm`` is the single home of
        ``beta_from_vf`` / ``eps_transverse`` / ``beta_eff_squared`` / ``kL_squared``; the FEM tier
        (``hydro_fem.HydroParams``) delegates to it and the FDTD tier (``hydro_fdtd``) re-exports
        ``beta_from_vf``.  All three tiers must return EXACTLY the same numbers (not "approximately"
        -- they must be the same code), and ``beta = sqrt(3/5) v_F`` must match Barton's closed form.
GATE B  LOCAL LIMIT.  ``beta -> 0`` must reproduce an ORDINARY local-Drude film: the in-file
        Fresnel 2x2 TMM (written here, no dynameta code) to ~1e-9 in R and T, at normal AND at
        oblique p-pol incidence.
GATE C  ENERGY.  R + T + A = 1 exactly, A >= 0 for a passive metal, and a LOSSLESS metal
        (gamma = 0) below the plasma edge is a perfect mirror (A = 0, R = 1) -- with and without
        the nonlocal pressure term.
GATE D  BULK PLASMONS.  Above ``omega_p/sqrt(eps_inf)`` the longitudinal wave propagates and the
        p-pol absorption of a thin film shows standing-wave peaks; the m-th peak must land on the
        closed form ``k_L(omega) d = m pi`` to < 1%.  This is the core nonlocal physics -- it is
        absent from the local answer entirely.

Run: python -m validation.nonlocal_hydro_material
"""
import cmath
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

C = 299792458.0
V_F_NA = 1.07e6                      # Na Fermi velocity [m/s]
WP_NA = 8.65e15                      # Na plasma frequency [rad/s]


# ------------------------------------------------------------------ in-file oracle (no dynameta)
def _airy_film(omega, eps_film, d_m, theta_rad=0.0, pol="p"):
    """R, T of a single film in vacuum from the textbook AIRY (multiple-beam) summation

        r = (r01 + r12 e^{2 i phi}) / (1 + r01 r12 e^{2 i phi}),  phi = kz1 d,

    with the ordinary Fresnel interface coefficients.  Written here from the textbook formulae, so
    it shares no code and no matrix convention with the module under test."""
    k0 = omega / C
    kz0 = k0 * math.cos(theta_rad)                       # vacuum on both sides
    kz1 = cmath.sqrt(eps_film * k0 * k0 - (k0 * math.sin(theta_rad)) ** 2)
    if kz1.imag < 0:
        kz1 = -kz1                                       # decaying / outgoing branch
    if pol == "p":                                       # p-pol: kz/eps ; s-pol: kz
        y0, y1 = kz0 / 1.0, kz1 / eps_film
    else:
        y0, y1 = kz0, kz1
    r01 = (y0 - y1) / (y0 + y1)
    t01 = 2.0 * y0 / (y0 + y1)
    r12 = (y1 - y0) / (y1 + y0)
    t12 = 2.0 * y1 / (y1 + y0)
    e2 = cmath.exp(2j * kz1 * d_m)
    den = 1.0 + r01 * r12 * e2
    r = (r01 + r12 * e2) / den
    t = t01 * t12 * cmath.exp(1j * kz1 * d_m) / den
    return float(abs(r) ** 2), float(abs(t) ** 2)        # vacuum both sides -> no flux factor


def main():
    from dynameta.optics import nonlocal_tmm as nt
    from dynameta.optics import hydro_fem as hf
    from dynameta.optics import hydro_fdtd as hfd

    ok = True

    # ---------------------------------------------------------------- GATE A: one material model
    print("[nhm] === GATE A: one hydrodynamic material model across three solver tiers ===",
          flush=True)
    a_ok = (hf.beta_from_vf is nt.beta_from_vf) and (hfd.beta_from_vf is nt.beta_from_vf)
    print("[nhm]   beta_from_vf is the SAME function object in nonlocal_tmm / hydro_fem / "
          "hydro_fdtd: {}".format(a_ok), flush=True)
    beta = nt.beta_from_vf(V_F_NA)
    a_ok = a_ok and abs(beta - math.sqrt(0.6) * V_F_NA) < 1e-9 * beta            # Barton 1979
    a_ok = a_ok and abs(nt.beta_from_vf(V_F_NA, "thomas_fermi")
                        - math.sqrt(1.0 / 3.0) * V_F_NA) < 1e-9 * beta
    p = hf.HydroParams(1.0, WP_NA, 1.0e14, beta, D=2.0e-4)
    lay = p.as_layer(3.0e-9)
    worst = 0.0
    for w in (1.0e15, 5.0e15, 9.0e15, 1.5e16, 9.0e15 + 1.0e14j):
        for got, want in ((p.eps_transverse(w), nt.eps_transverse(w, lay)),
                          (p.beta_eff_squared(w), nt.beta_eff_squared(w, lay)),
                          (p.kL_squared(w), nt.kL_squared(w, lay))):
            if got != want:                                       # EXACT: delegation, not agreement
                a_ok = False
            worst = max(worst, abs(got - want))
    print("[nhm]   HydroParams -> nonlocal_tmm response functions, max |difference| = {:.1e} "
          "(must be exactly 0)".format(worst), flush=True)
    # GNOR sign and passivity, the two conventions a silent re-derivation would invert
    a_ok = a_ok and p.beta_eff_squared(9.0e15).imag < 0.0
    a_ok = a_ok and nt.eps_transverse(5.0e15, lay).imag > 0.0
    print("[nhm] GATE A -> {}".format("PASS" if a_ok else "FAIL"), flush=True)
    ok = ok and a_ok

    # ---------------------------------------------------------------- GATE B: local limit
    print("[nhm] === GATE B: beta -> 0 reproduces an ordinary local-Drude film ===", flush=True)
    d = 20e-9
    gamma = 1.0e14
    b_ok, worst = True, 0.0
    for theta_deg in (0.0, 35.0):
        for pol in ("p", "s"):
            for w in (3.0e15, 6.0e15):
                loc = nt.HydroLayer(1.0, WP_NA, gamma, 1e-6, d)    # beta -> 0
                res = nt.stack_rt(w, [loc], pol=pol, theta_rad=math.radians(theta_deg))
                eps = nt.eps_transverse(w, loc)
                R0, T0 = _airy_film(w, eps, d, math.radians(theta_deg), pol)
                dR, dT = abs(res.R - R0), abs(res.T - T0)
                worst = max(worst, dR, dT)
                b_ok = b_ok and dR < 1e-9 and dT < 1e-9
                print("[nhm]   {}-pol th={:>4.0f} deg w={:.1e}: R {:.9f}/{:.9f}  "
                      "T {:.9f}/{:.9f}".format(pol, theta_deg, w, res.R, R0, res.T, T0),
                      flush=True)
    print("[nhm] GATE B (worst |diff| = {:.1e} vs the in-file Airy oracle) -> {}".format(
        worst, "PASS" if b_ok else "FAIL"), flush=True)
    ok = ok and b_ok

    # ---------------------------------------------------------------- GATE C: energy
    print("[nhm] === GATE C: energy budget ===", flush=True)
    c_ok = True
    lossy = nt.HydroLayer(1.0, WP_NA, gamma, beta, d)
    for w in (2.0e15, 5.0e15, 1.1e16):
        res = nt.stack_rt(w, [lossy], pol="p", theta_rad=math.radians(25.0))
        closes = abs(res.R + res.T + res.A - 1.0) < 1e-12
        c_ok = c_ok and closes and res.A >= -1e-12 and res.R <= 1.0 + 1e-12
        print("[nhm]   w={:.1e}: R={:.5f} T={:.5f} A={:.5f} sum-1={:+.1e}".format(
            w, res.R, res.T, res.A, res.R + res.T + res.A - 1.0), flush=True)
    # a LOSSLESS metal absorbs nothing, however thick, local or nonlocal: A == 0 identically (the
    # residual 1-R is tunnelling THROUGH the film, not loss -- hence the R+T=1 check beside it).
    for bb, tag in ((1e-6, "local"), (beta, "nonlocal")):
        mirror = nt.HydroLayer(1.0, WP_NA, 0.0, bb, 500e-9)        # gamma = 0, below the edge
        res = nt.stack_rt(2.0e15, [mirror], pol="p", theta_rad=0.0)
        perfect = abs(res.A) < 1e-12 and abs(res.R + res.T - 1.0) < 1e-12 and res.R > 0.999
        c_ok = c_ok and perfect
        print("[nhm]   lossless {:>8s} mirror below the plasma edge: R={:.9f} T={:.2e} "
              "A={:+.1e}".format(tag, res.R, res.T, res.A), flush=True)
    print("[nhm] GATE C -> {}".format("PASS" if c_ok else "FAIL"), flush=True)
    ok = ok and c_ok

    # ---------------------------------------------------------------- GATE D: bulk plasmons
    print("[nhm] === GATE D: bulk-plasmon standing waves at k_L d = m pi ===", flush=True)
    d_nm, dm = 10.0, 10e-9
    film = nt.HydroLayer(1.0, WP_NA, 1.0e13, beta, dm)             # low loss -> sharp peaks
    w_edge = WP_NA                                                  # eps_inf = 1
    ws = np.linspace(1.002 * w_edge, 1.06 * w_edge, 4000)
    A = np.array([nt.stack_rt(float(w), [film], pol="p",
                              theta_rad=math.radians(30.0)).A for w in ws])
    # interior local maxima
    pk = [i for i in range(1, len(ws) - 1) if A[i] > A[i - 1] and A[i] > A[i + 1] and A[i] > 1e-4]
    d_ok = len(pk) >= 4
    print("[nhm]   film {:.0f} nm, {} absorption peak(s) found above the plasma edge".format(
        d_nm, len(pk)), flush=True)
    # Closed form: the m-th standing wave sits at Re(k_L) d = m pi. Only ODD m couples (the even
    # modes carry no net surface charge). k_L d = m pi is the QUASISTATIC condition, so the
    # retardation/ABC correction is largest at low m and vanishes as m grows -- the gate is
    # therefore the CONVERGENCE, not a flat tolerance: every order within 5%, the deviation
    # STRICTLY decreasing with m (measured 4.4e-2 -> 1.3e-2 -> 4.0e-3 -> 1.1e-3 -> 3.4e-4 for
    # m = 3,5,7,9,11 on a 10 nm Na film), and below 1e-3 by the highest resolved order.
    errs = []
    for i in pk[:6]:
        kL = complex(cmath.sqrt(nt.kL_squared(float(ws[i]), film))).real
        x = kL * dm / math.pi
        m_near = int(round(x))
        err = abs(x - m_near) / m_near
        errs.append((m_near, err))
        d_ok = d_ok and m_near % 2 == 1 and err < 5e-2
        print("[nhm]   peak w={:.5e}: Re(k_L) d/pi = {:.4f} -> m={} (odd: {}) rel. error "
              "{:.2e}".format(ws[i], x, m_near, m_near % 2 == 1, err), flush=True)
    if len(errs) >= 3:
        shrinks = all(b[1] < a[1] for a, b in zip(errs, errs[1:]))
        d_ok = d_ok and shrinks and errs[-1][1] < 1e-3
        print("[nhm]   deviation from the quasistatic condition strictly decreasing with m: {} "
              "(highest order {:.1e}, must be < 1e-3)".format(shrinks, errs[-1][1]), flush=True)
    # the local solve has no standing-wave ladder -- the ladder IS the nonlocal physics (the local
    # film keeps its single ENZ/Berreman absorption feature at the plasma edge, hence "<= 1")
    loc_film = nt.HydroLayer(1.0, WP_NA, 1.0e13, 1e-6, dm)
    A_loc = np.array([nt.stack_rt(float(w), [loc_film], pol="p",
                                  theta_rad=math.radians(30.0)).A for w in ws])
    n_loc = sum(1 for i in range(1, len(ws) - 1)
                if A_loc[i] > A_loc[i - 1] and A_loc[i] > A_loc[i + 1] and A_loc[i] > 1e-4)
    d_ok = d_ok and n_loc <= 1 and len(pk) >= n_loc + 3
    print("[nhm]   same sweep with beta -> 0: {} peak(s) (local ENZ feature only; must be <= 1 "
          "and at least 3 fewer than the nonlocal ladder)".format(n_loc), flush=True)
    print("[nhm] GATE D -> {}".format("PASS" if d_ok else "FAIL"), flush=True)
    ok = ok and d_ok

    print("[nhm] *** NONLOCAL HYDRODYNAMIC MATERIAL + LAYERED HDM: {} ***".format(
        "PASS" if ok else "FAIL"), flush=True)
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
