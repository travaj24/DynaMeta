"""FDTD broadband SWEEP vs TMM. optics.fdtd_seam.fdtd_sweep_spectrum runs ONE broadband FDTD and returns
the whole R/T spectrum (FDTD's native strength) -- vs the per-wavelength OpticalSolver seam that re-solves
each wavelength. On a NON-dispersive (dielectric) stack the single-solve spectrum must reproduce the exact
coherent-TMM R/T at EVERY wavelength across the band, and energy must close. This is the fast path for a
wavelength sweep at a fixed bias (N wavelengths in ~1 solve instead of N).

GATE 1 (spectrum vs TMM): the one-solve R(lambda)/T(lambda), sampled at many wavelengths across the
        well-excited band, == TMM to the FDTD discretization; R+T = 1.
GATE 2 (one solve, many wavelengths): the sweep returns a dense spectrum from a single solve (the N-fold
        win over the per-wavelength seam is structural -- reported here).

Run: python -m validation.fdtd_sweep_vs_tmm
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dynameta.geometry import Design, Layer, Stack, UnitCell
from dynameta.materials import ConstantOptical, Material, MaterialRegistry
from dynameta.optics.fdtd_nd import available_backends
from dynameta.optics.fdtd_seam import fdtd_sweep_spectrum
from dynameta.optics.tmm_reference import layered_rta, layered_stack_from_design


def _design(layer_specs):
    reg = MaterialRegistry()
    reg.add(Material("air", ConstantOptical(1.0 + 0j)))
    layers = []
    for k, (eps, th) in enumerate(layer_specs):
        reg.add(Material("m%d" % k, ConstantOptical(complex(eps))))
        layers.append(Layer("s%d" % k, float(th), "m%d" % k))
    stack = Stack(layers=layers, superstrate_material="air", substrate_material="air")
    return Design(name="diel", unit_cell=UnitCell.square(220e-9), stack=stack, electrodes=[], materials=reg)


LMIN, LMAX = 1100e-9, 1500e-9
RES = 28
TARGETS = np.linspace(1200e-9, 1400e-9, 9)
STACK = [(2.2 ** 2, 250e-9), (1.5 ** 2, 180e-9), (2.2 ** 2, 250e-9)]

# GATE 1 convergence-gate constant, PER GEOMETRY (2026-11-03 decision): the worst second-order
# constant of THIS stack (2.2/1.5/2.2, 250/180/250 nm, band 1100-1500 nm, res 28, the 9 target
# wavelengths) over the 8 sub-cell offsets u = k/8, k = 0..7 (incl. u = 0 and 1/2) of its front face (fdtd_nd.grid.subcell_offset_pads,
# base n_pad_wave 4): C = max of max(|dR|, |dT|) / X^2, X = n_max k0 dz = 2 pi LMIN / (RES lambda)
# (fdtd_nd.grid.second_order_constant). No safety factor. Regenerate:
#   python -m validation.fdtd_sweep_vs_tmm --regen-cgeom        (measured 2026-11-03)
# Part (a) is a REGRESSION bound; the physics gate is part (b), the observed order.
C_GEOM_SWEEP = 0.21488435744194653  # 2026-11-03, u = k/8: 0.214884 0.197138 0.157715 0.101686 0.079942 0.138427 0.185522 0.211053
# The fixture IS the u = 0 member of its own sweep, so (a) sits at ratio 1 by construction; allow
# cross-platform floating-point round-off only (a physics regression moves the ratio by percent).
FP_ROUNDOFF = 1e-6


def _sweep_err(d, n_pad_wave=4.0, z_subcell="average"):
    """Per-target max(|dR|, |dT|) of the one-solve sweep vs TMM, the X per target, and the sweep."""
    sw = fdtd_sweep_spectrum(d, lambda_min_m=LMIN, lambda_max_m=LMAX, dim=2, resolution=RES,
                             n_pad_wave=n_pad_wave, z_subcell=z_subcell)
    err, X = [], []
    for lam in TARGETS:
        Rf = float(np.interp(lam, sw.lambda_m, sw.R)); Tf = float(np.interp(lam, sw.lambda_m, sw.T))
        Rt, Tt, _ = layered_rta(layered_stack_from_design(d, float(lam)), float(lam))
        err.append(max(abs(Rf - Rt), abs(Tf - Tt)))
        X.append(2.0 * np.pi * LMIN / (RES * float(lam)))
    return np.asarray(err), np.asarray(X), sw


def _probe_order(d, z_subcell="average"):
    """Observed order (res RES -> 2 RES) of the same three layers through solve_fdtd_2d (nx = 2)."""
    from dynameta.optics.fdtd_nd import FDTDLayer, solve_fdtd_2d
    lay = [FDTDLayer(thickness_m=th, eps_inf=eps) for eps, th in STACK]
    ep = []
    for res_p in (RES, 2 * RES):
        rp = solve_fdtd_2d(lay, period_x_m=220e-9, nx=2, lambda_min_m=LMIN, lambda_max_m=LMAX, resolution=res_p,
                           n_pad_wave=4.0, backend="numba" if "numba" in available_backends() else "numpy",
                           z_subcell=z_subcell)
        e_p = 0.0
        for lam in TARGETS:
            Rt, Tt, _ = layered_rta(layered_stack_from_design(d, float(lam)), float(lam))
            fq = 299792458.0 / float(lam)
            e_p = max(e_p, abs(float(np.interp(fq, rp.freqs_Hz, rp.R0)) - Rt),
                      abs(float(np.interp(fq, rp.freqs_Hz, rp.T0)) - Tt))
        ep.append(e_p)
    return float(np.log2(ep[0] / ep[1])), ep


def regen_c_geom():
    """The GATE 1 constant from the registration sweep of THIS stack (see C_GEOM_SWEEP)."""
    from dynameta.optics.fdtd_nd import FDTDLayer
    from dynameta.optics.fdtd_nd.grid import layer_band_index_max, second_order_constant, subcell_offset_pads
    d = _design(STACK)
    lay = [FDTDLayer(thickness_m=th, eps_inf=eps) for eps, th in STACK]
    n_max = layer_band_index_max(lay, 2 * np.pi * np.linspace(299792458.0 / LMAX, 299792458.0 / LMIN, 9),
                                 (1.0, 1.0))
    dz = LMIN / (RES * n_max)
    cs = []
    for npw in subcell_offset_pads(dz, LMAX, 8, 4.0):
        err, X, _sw = _sweep_err(d, npw)
        cs.append(second_order_constant(err, X))
    return max(cs), cs


def main():
    print("[fw] === FDTD broadband sweep vs TMM (one solve -> whole spectrum) ===", flush=True)
    d = _design(STACK)
    err_a, X_a, sw = _sweep_err(d)
    print("[fw] one solve ({:.1f}s) -> {} spectral points over [{:.0f},{:.0f}]nm".format(
        sw.solve_time_s, sw.lambda_m.size, sw.lambda_m.min() * 1e9, sw.lambda_m.max() * 1e9), flush=True)

    # GATE 1: sample many wavelengths in the well-excited centre and compare the ONE-solve spectrum to TMM
    targets = TARGETS
    dR = dT = en = 0.0
    for lam in targets:
        Rf = float(np.interp(lam, sw.lambda_m, sw.R))
        Tf = float(np.interp(lam, sw.lambda_m, sw.T))
        Rt, Tt, _ = layered_rta(layered_stack_from_design(d, float(lam)), float(lam))
        dR = max(dR, abs(Rf - Rt)); dT = max(dT, abs(Tf - Tt)); en = max(en, abs(Rf + Tf - 1.0))
    # 2026-11-03: the single-resolution pin (dR, dT < 6e-3) is a CONVERGENCE gate against the
    # true-geometry TMM. (a) REGRESSION bound: per wavelength, max(|dR|, |dT|) <= C_GEOM_SWEEP * X^2
    # with this stack's own registration-sweep constant (see C_GEOM_SWEEP). (b) PHYSICS: observed order
    # >= 1.7 between res 28 and 56 on a cheaper probe of the same engine (solve_fdtd_2d on the same
    # three layers, nx = 2; the 2x-res sweep of THIS config costs ~8x its ~21 s). Under the 2b7a002
    # point rule (b) FAILS here (the staircase does not converge on this near-AR stack: see the doc).
    ratio_a = float(np.max(err_a / (C_GEOM_SWEEP * X_a ** 2)))
    order_1, ep = _probe_order(d)
    gate1 = bool(ratio_a <= 1.0 + FP_ROUNDOFF and order_1 >= 1.7 and en < 6e-3)
    print("[fw] 1 spectrum vs TMM ({} wavelengths): max|dR|={:.2e} max|dT|={:.2e} max|R+T-1|={:.2e}; (a) max "
          "err/(C X^2)={:.3f} (C_geom={}) ; (b) order {:.2f} ({:.2e} -> {:.2e}, nx=2 probe) -> {}".format(
              targets.size, dR, dT, en, ratio_a, C_GEOM_SWEEP, order_1, ep[0], ep[1], "PASS" if gate1 else "FAIL"),
          flush=True)

    # GATE 2: a single solve genuinely yields the whole dense spectrum (the per-wavelength seam would
    # need ~that many solves) -- the structural N-fold speedup.
    gate2 = bool(sw.lambda_m.size >= 20 and np.all(np.isfinite(sw.R)))
    print("[fw] 2 one-solve sweep: {} clean points from 1 FDTD solve (vs {} per-wavelength solves) -> {}".format(
        sw.lambda_m.size, sw.lambda_m.size, "PASS" if gate2 else "FAIL"), flush=True)

    overall = gate1 and gate2
    print("[fw] *** FDTD SWEEP vs TMM (one-solve broadband spectrum matches TMM across the band): {} ***".format(
        "PASS" if overall else "FAIL"), flush=True)
    return overall


if __name__ == "__main__":
    if "--regen-cgeom" in sys.argv[1:]:
        c, per = regen_c_geom()
        print("[fw] C_geom (GATE 1, res 28, 8 offsets k/8 x 9 wavelengths) = {!r}; per offset {}".format(
            c, ", ".join("{:.6f}".format(v) for v in per)))
        raise SystemExit(0)
    raise SystemExit(0 if main() else 1)
