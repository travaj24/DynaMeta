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
    gate1 = bool(dR < 6e-3 and dT < 6e-3 and en < 6e-3)
    print("[fw] 1 spectrum vs TMM ({} wavelengths): max|dR|={:.2e} max|dT|={:.2e} max|R+T-1|={:.2e} -> {}".format(
        targets.size, dR, dT, en, "PASS" if gate1 else "FAIL"), flush=True)

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
