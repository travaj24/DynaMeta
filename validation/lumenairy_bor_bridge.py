"""Lumenairy BOR-PMM (axisymmetric / body-of-revolution) bridge vs independent oracles.

BOR-PMM (lumenairy.elements.bor, released in lumenairy 5.16.0) is the CYLINDRICAL peer of the Cartesian
RCWA/PMM solvers: a structure invariant under rotation about an axis is solved on a 1-D RADIAL eigenbasis
per azimuthal order m, cascaded in z. solve_bor returns per-INCIDENT-MODE R/T (each mode = a cylindrical
wave at a discrete polar angle set by the computational radius r_max). The bridge is a SI/OpticalResult
adapter; conventions are identical (exp(-i omega t), Im(eps) > 0).

GATE A (identity): an index-matched stack (super == slab == substrate) reflects nothing -- R[j] ~ 0,
        T[j] ~ 1 for EVERY propagating mode (adapter sanity + trivial energy).
GATE B (uniform stack == planar Fresnel, the INDEPENDENT per-quantity oracle): a uniform interface has
        NO radial structure, so each incident cylindrical mode reflects as a PLANE WAVE at its polar
        angle theta_j (no mode mixing). Its R[j] must equal the analytic Fresnel reflectance at theta_j
        for the mode's TE-or-TM character -- asserted by BRACKETING |R[j]| against BOTH s- and p-pol
        Fresnel (min-distance < tol). This is discriminating where it matters: at large theta the s/p
        Fresnel values DIVERGE, so R[j] must land on one of two DISTINCT curves (energy conservation
        alone, the lossless trap, could not catch a per-mode error). Plus R[j] + T[j] = 1 (lossless).
GATE C (a ring grating DIFFRACTS): a concentric binary ring grating must redistribute power across
        many output orders -- the per-mode R spread is WIDE and the fundamental R DIFFERS materially
        from the same stack with the rings replaced by their volume-average (EMT) uniform index (a build
        that ignored the ring structure would match the uniform one). Plus energy R + T = 1. (The
        rigorous per-order-vs-planar match is lumenairy's own GATE 4, validated upstream.)
GATE D (lossy passivity + sign): a lossy layer (Im(eps) > 0) absorbs -- A = 1 - R - T > 0, energy
        R + T < 1, and 0 <= R, T <= 1 (passivity); a gain sign error would give A < 0 / R+T > 1.
GATE E (complex phase + per-layer absorption budget, AUDIT C1 / B4b): (a) the fundamental reflection
        PHASE is a real physical observable -- it must be GAUGE-STABLE (byte-identical across two
        independent solves of the same spec, via the pinned per_mode_amplitudes gauge / gauge-invariant
        S-diagonal) and NON-TRIVIAL for a finite uniform slab (its two-interface Fresnel-Airy
        interference rotates r off the real axis, so phase differs from 0 and 180 by > 1 deg -- a
        placeholder phase_deg = 0 would fail). (b) per-layer absorption closes the energy budget: a lossy
        uniform slab has A_independent > 0 with |R + T + sum_layers A - 1| < 1e-9 for the fundamental,
        while a LOSSLESS ring grating absorbs ~ 0 (A_independent < 1e-9) -- a sign/normalization error in
        the flux-difference recipe would break the budget or leak a spurious loss.

Honest SKIP (exit 0 + banner) when lumenairy < 5.16.0 / not importable.

Run: python -m validation.lumenairy_bor_bridge
"""
import importlib.util
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

LAM = 1.0e-6


def _fresnel_RsRp(n1, n2, theta_i):
    """Analytic planar power reflectances (s- and p-pol) for n1 -> n2 at incidence theta_i (rad)."""
    n1, n2 = complex(n1), complex(n2)
    c1 = np.cos(theta_i)
    c2 = np.sqrt(1.0 - ((n1 / n2) * np.sin(theta_i)) ** 2 + 0j)     # cos(theta_t) via Snell
    rs = (n1 * c1 - n2 * c2) / (n1 * c1 + n2 * c2)
    rp = (n2 * c1 - n1 * c2) / (n2 * c1 + n1 * c2)
    return abs(rs) ** 2, abs(rp) ** 2


def _phase_off_axis(deg):
    """Angular distance (deg) of a reflection phase from the nearest REAL axis (0 or 180). A lossless
    single interface sits at 0/180 (r real); a finite slab's two-interface interference rotates it off,
    so a genuinely-carried phase reads > 0 here (a placeholder phase_deg = 0 reads exactly 0)."""
    d = abs(float(deg)) % 180.0
    return min(d, 180.0 - d)


def main():
    if importlib.util.find_spec("lumenairy") is None:
        print("[bor] *** SKIP: lumenairy not installed -- BOR bridge gates not run ***", flush=True)
        return True
    import lumenairy
    ver = tuple(int(p) for p in str(lumenairy.__version__).split(".")[:3])
    if ver < (5, 16, 0):
        print("[bor] *** SKIP: lumenairy {} < 5.16.0 (BOR-PMM tier) -- gates not run ***".format(
            lumenairy.__version__), flush=True)
        return True
    from dynameta.optics.lumenairy_bridge import BorLayer, BorStackSpec, solve_bor

    print("[bor] === Lumenairy BOR-PMM (axisymmetric) bridge vs Fresnel / diffraction / passivity ===",
          flush=True)
    ok = True

    # ---- GATE A: index-matched identity ----
    idn = BorStackSpec(layers=[BorLayer(thickness_m=0.6e-6, eps=complex(1.5 ** 2))],
                       azimuthal_order_m=1, r_max_m=40e-6, n_radial=200, n_super=1.5, n_sub=1.5)
    ra = solve_bor(idn, LAM)
    g_a = bool(ra.angles_rad.size > 1 and np.max(ra.R) < 1e-6 and np.max(np.abs(ra.T - 1.0)) < 1e-6)
    ok = ok and g_a
    print("[bor] GATE A: index-matched identity -- {} modes, max R={:.1e}, max|T-1|={:.1e} -> {}".format(
        ra.angles_rad.size, np.max(ra.R), np.max(np.abs(ra.T - 1.0)), "PASS" if g_a else "FAIL"),
        flush=True)

    # ---- GATE B: uniform stack per-mode R == planar Fresnel (s|p bracket) + energy ----
    n1, n2 = 1.0, 1.5
    uni = BorStackSpec(layers=[BorLayer(thickness_m=0.6e-6, eps=complex(n2 ** 2))],   # slab index = sub
                       azimuthal_order_m=1, r_max_m=40e-6, n_radial=220, n_super=n1, n_sub=n2)
    rb = solve_bor(uni, LAM)
    Rs, Rp = _fresnel_RsRp(n1, n2, rb.angles_rad)
    bracket = np.minimum(np.abs(rb.R - Rs), np.abs(rb.R - Rp))      # nearest of the two pol curves
    worst_fresnel = float(np.max(bracket))
    worst_energy = float(np.max(np.abs(rb.energy - 1.0)))
    # discriminability: at the largest angle the two Fresnel pols must DIVERGE (so the bracket is not
    # vacuous -- R has to land on one of two DISTINCT values, which energy conservation cannot enforce)
    sp_split = float(np.max(np.abs(Rs - Rp)))
    g_b = bool(worst_fresnel < 5e-3 and worst_energy < 1e-6 and sp_split > 0.1)
    ok = ok and g_b
    print("[bor] GATE B: uniform == planar Fresnel (s|p bracket) worst |dR|={:.1e} over {} modes "
          "(s/p split up to {:.2f}); energy |R+T-1|={:.1e} -> {}".format(
              worst_fresnel, rb.angles_rad.size, sp_split, worst_energy, "PASS" if g_b else "FAIL"),
          flush=True)

    # ---- GATE C: a ring grating diffracts (differs from the EMT-uniform stack) + energy ----
    period, duty, n_r, n_g = 3.0e-6, 0.5, 2.45, 1.41
    ring = BorStackSpec(layers=[BorLayer(thickness_m=0.5e-6, rings=(period, duty, n_r, n_g))],
                        azimuthal_order_m=1, r_max_m=48e-6, n_radial=256, n_super=1.41, n_sub=1.41)
    rc = solve_bor(ring, LAM)
    eps_emt = duty * n_r ** 2 + (1.0 - duty) * n_g ** 2            # volume-average (a no-diffraction ref)
    emt = BorStackSpec(layers=[BorLayer(thickness_m=0.5e-6, eps=complex(eps_emt))],
                       azimuthal_order_m=1, r_max_m=48e-6, n_radial=256, n_super=1.41, n_sub=1.41)
    re = solve_bor(emt, LAM)
    r_spread = float(rc.R.max() - rc.R.min())                      # diffraction redistributes power
    fund_diff = abs(float(rc.fundamental_result().R) - float(re.fundamental_result().R))
    energy_c = float(np.max(np.abs(rc.energy - 1.0)))
    g_c = bool(rc.angles_rad.size > 1 and r_spread > 0.2 and fund_diff > 5e-3 and energy_c < 1e-6)
    ok = ok and g_c
    print("[bor] GATE C: ring grating diffracts -- {} modes, R spread={:.3f}, |R_ring-R_emt|(fund)="
          "{:.3f}, energy |R+T-1|={:.1e} -> {}".format(rc.angles_rad.size, r_spread, fund_diff,
                                                        energy_c, "PASS" if g_c else "FAIL"), flush=True)

    # ---- GATE D: lossy passivity + sign convention ----
    lossy = BorStackSpec(layers=[BorLayer(thickness_m=0.4e-6, eps=complex(2.5 ** 2, 0.4))],   # Im>0 loss
                         azimuthal_order_m=1, r_max_m=40e-6, n_radial=200, n_super=1.0, n_sub=1.5)
    rd = solve_bor(lossy, LAM)
    A = 1.0 - rd.R - rd.T
    g_d = bool(np.min(A) > 1e-3 and np.max(rd.energy) < 1.0 + 1e-9
               and np.min(rd.R) >= -1e-9 and np.min(rd.T) >= -1e-9 and np.max(rd.R) <= 1.0 + 1e-9)
    ok = ok and g_d
    print("[bor] GATE D: lossy (Im eps>0) absorbs -- min A={:.3f} (>0), max(R+T)={:.4f} (<1), R,T in "
          "[0,1] -> {}".format(float(np.min(A)), float(np.max(rd.energy)), "PASS" if g_d else "FAIL"),
          flush=True)

    # ---- GATE E: fundamental reflection PHASE (gauge-stable + non-trivial) + per-layer absorption ----
    # (a) a finite uniform slab has a real Fresnel-Airy reflection phase -- reproducible run-to-run
    #     (pinned gauge / gauge-invariant S-diagonal) and rotated OFF the real axis by its two-interface
    #     interference (so |phase| is > 1 deg from both 0 and 180; a placeholder phase_deg = 0 would fail).
    ph_spec = BorStackSpec(layers=[BorLayer(thickness_m=0.6e-6, eps=complex(1.8 ** 2), name="film")],
                           azimuthal_order_m=1, r_max_m=40e-6, n_radial=220, n_super=1.0, n_sub=1.0)
    fe1 = solve_bor(ph_spec, LAM).fundamental_result()
    fe2 = solve_bor(ph_spec, LAM).fundamental_result()          # independent re-solve of the same spec
    phase_stable = bool(fe1.phase_deg == fe2.phase_deg)
    phase_off = _phase_off_axis(fe1.phase_deg)
    phase_nontrivial = bool(phase_off > 1.0)
    # (b) lossy uniform slab: A_independent > 0 and R + T + A_ind closes to machine precision for the
    #     fundamental; a LOSSLESS ring grating must absorb ~ 0 (no spurious loss from the flux recipe).
    lossyE = BorStackSpec(layers=[BorLayer(thickness_m=0.4e-6, eps=complex(2.5 ** 2, 0.4), name="absorber")],
                          azimuthal_order_m=1, r_max_m=40e-6, n_radial=200, n_super=1.0, n_sub=1.5)
    le = solve_bor(lossyE, LAM, absorption=True).fundamental_result()
    budget_lossy = abs(float(le.R) + float(le.T) + float(le.A_independent) - 1.0)
    ringE = BorStackSpec(layers=[BorLayer(thickness_m=0.5e-6, rings=(period, duty, n_r, n_g), name="grating")],
                         azimuthal_order_m=1, r_max_m=48e-6, n_radial=256, n_super=1.41, n_sub=1.41)
    lr = solve_bor(ringE, LAM, absorption=True).fundamental_result()
    abs_ok = bool(le.A_independent is not None and le.A_independent > 1e-3 and budget_lossy < 1e-9
                  and lr.A_independent is not None and abs(lr.A_independent) < 1e-9)
    g_e = bool(phase_stable and phase_nontrivial and abs_ok)
    ok = ok and g_e
    print("[bor] GATE E: fund phase={:.3f} deg (stable={}, off-axis={:.1f} deg > 1); lossy A_ind={:.4f} "
          "budget|R+T+A-1|={:.1e}; lossless-ring A_ind={:.1e} -> {}".format(
              fe1.phase_deg, phase_stable, phase_off, float(le.A_independent), budget_lossy,
              float(lr.A_independent), "PASS" if g_e else "FAIL"), flush=True)

    print("[bor] *** LUMENAIRY BOR-PMM BRIDGE: {} ***".format("PASS" if ok else "FAIL"), flush=True)
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
