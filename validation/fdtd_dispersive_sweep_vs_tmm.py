"""FDTD DISPERSIVE broadband sweep vs TMM. optics.fdtd_seam.fdtd_sweep_spectrum(dispersive=True) /
run_fdtd_sweep fit each uniform layer's eps(lambda) to ONE Drude pole the FDTD runs natively, so a SINGLE
broadband solve reproduces a DISPERSIVE material's spectrum across the whole band -- not just a
non-dispersive dielectric (the prior sweep limit). This is what makes the one-solve sweep usable for the
real metal/ITO/Drude layers, closing the per-wavelength seam's repeated-settling-tail cost (audit medium).

GATE 1 (dispersive sweep vs TMM): a stack with a genuinely dispersive (Drude) layer -- the one-solve
        run_fdtd_sweep R(lambda)/T(lambda), at many wavelengths across the band, == coherent TMM; energy
        budget R+T+A=1.
GATE 2 (dispersion matters): the FROZEN-at-centre sweep (dispersive=False) is MUCH worse than the
        dispersive sweep at the band edges (frozen_err >> dispersive_err) -- the fit is doing real work.

Run: python -m validation.fdtd_dispersive_sweep_vs_tmm
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dynameta.geometry import Design, Layer, Stack, UnitCell
from dynameta.materials import ConstantOptical, Material, MaterialRegistry, TabulatedOptical
from dynameta.optics.fdtd_seam import run_fdtd_sweep
from dynameta.optics.tmm_reference import layered_rta, layered_stack_from_design

C = 299792458.0


def _drude_tabulated(eps_inf, wp, gamma, lo=1000e-9, hi=1900e-9, n=200):
    """A genuinely dispersive material: eps(w)=eps_inf - wp^2/(w^2+i*w*gamma), sampled as TabulatedOptical
    (a plain lambda-function both TMM and the FDTD fitter consume identically)."""
    lam = np.linspace(lo, hi, n)
    w = 2.0 * np.pi * C / lam
    eps = eps_inf - wp ** 2 / (w ** 2 + 1j * w * gamma)
    return TabulatedOptical(lambda_m=lam, eps_complex=eps)


def _design():
    # A THICK, strongly-dispersive absorber: eps swings ~1.4 -> 0.5 (Re) across 1.2-1.5um (dielectric-ish
    # to metallic) so dispersion strongly shifts R/T; 700nm + the loss suppress the Fabry-Perot fringe
    # (the single-slab FDTD numerical-dispersion confound), so the dispersion comparison is clean.
    reg = MaterialRegistry()
    reg.add(Material("air", ConstantOptical(1.0 + 0j)))
    reg.add(Material("disp", _drude_tabulated(3.0, 2.0e15, 8.0e13)))
    stack = Stack(layers=[Layer("s0", 700e-9, "disp")],
                  superstrate_material="air", substrate_material="air")
    return Design(name="disp", unit_cell=UnitCell.square(220e-9), stack=stack, electrodes=[], materials=reg)


def main():
    print("[fd] === FDTD dispersive broadband sweep vs TMM (one solve, dispersive absorber) ===", flush=True)
    d = _design()
    targets = np.linspace(1200e-9, 1500e-9, 7)

    # ONE dispersive solve serves all wavelengths (vs the per-wavelength seam = 7 solves)
    disp = run_fdtd_sweep(d, targets, dim=2, resolution=32, band_pad=0.12, dispersive=True)
    froz = run_fdtd_sweep(d, targets, dim=2, resolution=32, band_pad=0.12, dispersive=False)

    dR = dT = 0.0
    fR = 0.0
    Amin = 1.0
    for i, lam in enumerate(targets):
        Rt, Tt, At = layered_rta(layered_stack_from_design(d, float(lam)), float(lam))
        dR = max(dR, abs(disp[i].R - Rt)); dT = max(dT, abs((disp[i].T or 0) - Tt))
        fR = max(fR, abs(froz[i].R - Rt))                   # frozen-at-centre error vs TMM
        Amin = min(Amin, At)
    gate1 = bool(dR < 1e-2 and dT < 1e-2 and Amin > 0.1)    # matches TMM; genuinely absorbing (non-trivial)
    print("[fd] 1 dispersive sweep vs TMM ({} wl, 1 solve {:.1f}s): max|dR|={:.2e} max|dT|={:.2e} "
          "(TMM A>={:.2f}, lossy) -> {}".format(targets.size, disp[0].solve_time_s, dR, dT, Amin,
                                                "PASS" if gate1 else "FAIL"), flush=True)
    gate2 = bool(fR > 2.5 * max(dR, 1e-4))                  # dispersion fit clearly beats frozen-at-centre
    print("[fd] 2 dispersion matters: frozen-at-centre max|dR|={:.2e} vs dispersive {:.2e} "
          "({:.1f}x worse) -> {}".format(fR, dR, fR / max(dR, 1e-9), "PASS" if gate2 else "FAIL"), flush=True)

    overall = gate1 and gate2
    print("[fd] *** FDTD DISPERSIVE SWEEP vs TMM (one-solve, Drude-fit dispersion across the band): {} ***".format(
        "PASS" if overall else "FAIL"), flush=True)
    return overall


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
