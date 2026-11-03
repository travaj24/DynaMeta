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


def main():
    print("[fw] === FDTD broadband sweep vs TMM (one solve -> whole spectrum) ===", flush=True)
    d = _design([(2.2 ** 2, 250e-9), (1.5 ** 2, 180e-9), (2.2 ** 2, 250e-9)])

    LMIN, LMAX = 1100e-9, 1500e-9
    sw = fdtd_sweep_spectrum(d, lambda_min_m=LMIN, lambda_max_m=LMAX, dim=2, resolution=28)
    print("[fw] one solve ({:.1f}s) -> {} spectral points over [{:.0f},{:.0f}]nm".format(
        sw.solve_time_s, sw.lambda_m.size, sw.lambda_m.min() * 1e9, sw.lambda_m.max() * 1e9), flush=True)

    # GATE 1: sample many wavelengths in the well-excited centre and compare the ONE-solve spectrum to TMM
    targets = np.linspace(1200e-9, 1400e-9, 9)
    dR = dT = en = 0.0
    for lam in targets:
        Rf = float(np.interp(lam, sw.lambda_m, sw.R))
        Tf = float(np.interp(lam, sw.lambda_m, sw.T))
        Rt, Tt, _ = layered_rta(layered_stack_from_design(d, float(lam)), float(lam))
        dR = max(dR, abs(Rf - Rt)); dT = max(dT, abs(Tf - Tt)); en = max(en, abs(Rf + Tf - 1.0))
    # 2026-11-03: the single-resolution pin (dR, dT < 6e-3) is replaced by a CONVERGENCE gate against
    # the true-geometry TMM. (a) At res 28, per wavelength, max(|dR|, |dT|) <= C_REG_2D * X^2 with
    # X = n_max k0 dz = 2 pi lambda_min / (res lambda) and C_REG_2D = 0.0802, the WORST registration-
    # sweep constant of the 2-D normal-incidence engine (max over 8 midpoint sub-cell offsets and the
    # band; 183 nm n = 2 slab and the gain line on it, band 1.2-1.6 um, res 30; docs/audit/2026-11-02-
    # fdtd-registration.md "Convergence gates") -- NOT fitted to this stack. (b) observed order between
    # res 28 and 56 >= 1.7 on a cheaper probe of the same engine (solve_fdtd_2d on the same three
    # layers, nx = 2, numba; the 2x-res sweep of THIS config costs ~8x its ~21 s). Measured 2026-11-03:
    # (a) worst ratio 2.68 -> FAILS: this four-interface near-AR stack (R 0.03) has a ~2.7x larger
    # second-order constant than the registration slab; (b) 6.56e-3 -> 1.59e-3, order 2.05.
    C_REG_2D = 0.0802
    ratio_a = 0.0
    for lam in targets:
        Rf = float(np.interp(lam, sw.lambda_m, sw.R)); Tf = float(np.interp(lam, sw.lambda_m, sw.T))
        Rt, Tt, _ = layered_rta(layered_stack_from_design(d, float(lam)), float(lam))
        X = 2.0 * np.pi * LMIN / (28 * float(lam))
        ratio_a = max(ratio_a, max(abs(Rf - Rt), abs(Tf - Tt)) / (C_REG_2D * X ** 2))
    from dynameta.optics.fdtd_nd import FDTDLayer, solve_fdtd_2d
    lay = [FDTDLayer(thickness_m=250e-9, eps_inf=2.2 ** 2), FDTDLayer(thickness_m=180e-9, eps_inf=1.5 ** 2),
           FDTDLayer(thickness_m=250e-9, eps_inf=2.2 ** 2)]
    ep = []
    for res_p in (28, 56):
        rp = solve_fdtd_2d(lay, period_x_m=220e-9, nx=2, lambda_min_m=LMIN, lambda_max_m=LMAX, resolution=res_p,
                           n_pad_wave=4.0, backend="numba" if "numba" in available_backends() else "numpy")
        e_p = 0.0
        for lam in targets:
            Rt, Tt, _ = layered_rta(layered_stack_from_design(d, float(lam)), float(lam))
            fq = 299792458.0 / float(lam)
            e_p = max(e_p, abs(float(np.interp(fq, rp.freqs_Hz, rp.R0)) - Rt),
                      abs(float(np.interp(fq, rp.freqs_Hz, rp.T0)) - Tt))
        ep.append(e_p)
    order_1 = float(np.log2(ep[0] / ep[1]))
    gate1 = bool(ratio_a <= 1.0 and order_1 >= 1.7 and en < 6e-3)
    print("[fw] 1 spectrum vs TMM ({} wavelengths): max|dR|={:.2e} max|dT|={:.2e} max|R+T-1|={:.2e}; (a) max "
          "err/(C X^2)={:.3f} (C={}) ; (b) order {:.2f} ({:.2e} -> {:.2e}, nx=2 probe) -> {}".format(
              targets.size, dR, dT, en, ratio_a, C_REG_2D, order_1, ep[0], ep[1], "PASS" if gate1 else "FAIL"),
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
    raise SystemExit(0 if main() else 1)
