"""Validate the per-cell time-domain eps hook (roadmap R4): effect_eps_to_fdtd_grid turns a slow-drive
EffectModel's per-cell COMPLEX (lossy) eps into FDTD (eps_inf, wp, gamma) grids that the new
solve_fdtd_2d lateral_wp / lateral_gam seam carries -- the lossy/graded per-cell capability the
eps_inf-only lateral seam could not represent. Independent reference = the coherent-TMM stack_rta.

GATE A (uniform reduction / plumbing): a uniform lossy slab built via the lateral (eps_inf, wp, gam)
        grids reproduces the SAME slab built as an explicit FDTDLayer to < 1e-9 (byte-identical when the
        grid is uniform; same sizing layer so the mesh is identical).
GATE B (graded lossy vs TMM, defeats the lossless trap): a two-sublayer graded LOSSY eps(z) fed through
        the hook + lateral seam reproduces stack_rta R/T AND absorption A = 1-R-T to a few % -- A > 0 from
        a genuinely lossy eps cannot be faked by an energy tripwire (cross-checked vs the independent TMM).

Run: python -m validation.fdtd_effect_seam
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dynameta.optics.fdtd import FDTDLayer
from dynameta.optics.fdtd_nd import solve_fdtd_2d
from dynameta.optics.fdtd_seam import effect_eps_to_fdtd_grid, _eps_to_fdtd_layer
from dynameta.optics.tmm_reference import stack_rta

C = 299792458.0
LAM = 1300e-9
D = 240e-9
EPS_A = 6.25 + 0.6j      # lossy front sublayer
EPS_B = 4.0 + 0.25j      # lossy back sublayer
PERIOD = 300e-9


def _flux_at(r):
    # interpolate to EXACTLY omega0 = 2 pi c / LAM: the per-cell single-Drude-pole inversion reproduces
    # eps only AT omega0, so reading off-frequency would see the pole's (wrong) dispersion.
    f0 = C / LAM
    return float(np.interp(f0, r.freqs_Hz, r.R_flux)), float(np.interp(f0, r.freqs_Hz, r.T_flux))


def _run_lateral(eps_front, eps_back, sizing_layer, *, res=34):
    """solve_fdtd_2d with the structure band split front/back into (eps_front, eps_back), carried via the
    R4 lateral (eps_inf, wp, gam) grids from effect_eps_to_fdtd_grid. `sizing_layer` sets the grid n_max."""
    cache = {}

    def triple(nx, nz, zc, pad, zs):
        key = (nx, nz)
        if key not in cache:
            eg = np.ones((nx, nz), dtype=np.complex128)
            band = (zc >= pad) & (zc < pad + zs)
            front = band & (zc < pad + zs / 2.0)
            back = band & (zc >= pad + zs / 2.0)
            eg[:, front] = eps_front
            eg[:, back] = eps_back
            cache[key] = effect_eps_to_fdtd_grid(eg, LAM)
        return cache[key]

    return _flux_at(solve_fdtd_2d([sizing_layer], period_x_m=PERIOD, nx=4, lambda_min_m=1200e-9,
                                  lambda_max_m=1400e-9, resolution=res,
                                  lateral_eps_inf=lambda *a: triple(*a)[0],
                                  lateral_wp=lambda *a: triple(*a)[1],
                                  lateral_gam=lambda *a: triple(*a)[2]))


def main():
    print("[fe] === FDTD per-cell effect eps(t) hook (lateral wp/gam) vs TMM ===", flush=True)

    # GATE A: uniform lossy slab via lateral grids == explicit FDTDLayer (SAME sizing layer -> same mesh)
    L = _eps_to_fdtd_layer(D, EPS_A, LAM)
    Rl, Tl = _flux_at(solve_fdtd_2d([L], period_x_m=PERIOD, nx=4, lambda_min_m=1200e-9,
                                    lambda_max_m=1400e-9, resolution=34))
    Ra, Ta = _run_lateral(EPS_A, EPS_A, L)
    dA = max(abs(Ra - Rl), abs(Ta - Tl))
    g_a = dA < 1e-9
    print("[fe] A uniform via lateral wp/gam == explicit layer: R {:.6f}/{:.6f} T {:.6f}/{:.6f} max|d|={:.1e}"
          " -> {}".format(Ra, Rl, Ta, Tl, dA, "OK" if g_a else "FAIL"), flush=True)

    # GATE B: graded lossy via hook vs coherent TMM (A > 0, lossless-trap-defeating)
    Rf, Tf = _run_lateral(EPS_A, EPS_B, L)
    Af = 1.0 - Rf - Tf
    Rt, Tt, At = stack_rta(1.0, [(np.sqrt(EPS_A), D / 2.0), (np.sqrt(EPS_B), D / 2.0)], 1.0, LAM)
    dR, dT, dAbs = abs(Rf - Rt), abs(Tf - Tt), abs(Af - At)
    g_b = dR < 3e-2 and dT < 3e-2 and dAbs < 3e-2 and At > 0.1
    print("[fe] B graded lossy vs TMM: R {:.4f}/{:.4f} T {:.4f}/{:.4f} A {:.4f}/{:.4f} (A>0 genuine loss); "
          "|dR|={:.3f} |dT|={:.3f} |dA|={:.3f} -> {}".format(Rf, Rt, Tf, Tt, Af, At, dR, dT, dAbs,
                                                            "OK" if g_b else "FAIL"), flush=True)

    ok = g_a and g_b
    print("[fe] *** FDTD EFFECT eps(t) HOOK: {} ***".format("PASS" if ok else "FAIL"), flush=True)
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
