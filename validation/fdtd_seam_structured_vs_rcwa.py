"""FDTD STRUCTURED seam vs grcwa (RCWA) -- the gate that the lateral-inclusion RASTERIZATION is correct
(Design inclusions -> the (nx,ny,nz) FDTD eps grid). A 2D-periodic dielectric square-pillar array is
solved two completely independent ways:
  * optics.fdtd_seam.make_fdtd_optical_solver(dim=3) -- time-domain FDTD, geometry rasterized; and
  * grcwa -- frequency-domain RCWA (Fourier-space), the same geometry on an eps grid.
The R/T must agree to a few % (FDTD spatial discretization + RCWA Fourier truncation), and energy must
close. This is where FDTD earns its keep over TMM (which is exact only for UNIFORM stacks). A SUB-
wavelength pillar (0-order only) keeps the 3D grid small/fast while still exercising the full rasterize
path; genuine (kx,ky) diffraction is validated separately in fdtd_3d_reduces.py GATE C.

GATE 0 (oracle sanity): grcwa on a UNIFORM slab == analytic Airy (confirms the RCWA setup is correct).
GATE 1 (the test): FDTD-structured R/T == grcwa-structured, and R+T = 1.

Run: python -m validation.fdtd_seam_structured_vs_rcwa
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import grcwa

from dynameta.geometry import Design, Layer, Stack, UnitCell
from dynameta.geometry.cross_section import Rectangle
from dynameta.geometry.stack import Inclusion
from dynameta.materials import ConstantOptical, Material, MaterialRegistry
from dynameta.optics.fdtd_seam import make_fdtd_optical_solver

C = 299792458.0
LAM = 1300e-9
PERIOD = 600e-9           # sub-wavelength: 0-order only, so the small 3D grid stays run_all-fast while
PILLAR = 300e-9          # still cross-checking the Design->grid RASTERIZATION vs RCWA. (Genuine (kx,ky)
THICK = 250e-9           # diffraction is validated separately by fdtd_3d_reduces GATE C.)
EPS_HI = 6.25            # n = 2.5 dielectric pillar in air


def airy(f, n, d):
    k0 = 2 * np.pi * np.asarray(f) / C
    b = n * k0 * d
    r1 = (1.0 - n) / (1.0 + n)
    e2 = np.exp(2j * b)
    r = r1 * (1.0 - e2) / (1.0 - r1 ** 2 * e2)
    t = (1.0 - r1 ** 2) * np.exp(1j * b) / (1.0 - r1 ** 2 * e2)
    return float(np.abs(r) ** 2), float(np.abs(t) ** 2)


def _rcwa_rt(eps_grid, thick_nm, nG=101):
    """grcwa total R/T for ONE patterned layer (eps_grid, Ng x Ng) in vacuum, normal incidence."""
    lam_nm = LAM * 1e9
    p_nm = PERIOD * 1e9
    freq = 1.0 / lam_nm                                  # grcwa: c = 1, freq = 1/lambda
    obj = grcwa.obj(nG, [p_nm, 0.0], [0.0, p_nm], freq, 0.0, 0.0, verbose=0)
    obj.Add_LayerUniform(lam_nm, 1.0)                    # semi-infinite superstrate (air)
    obj.Add_LayerGrid(thick_nm, eps_grid.shape[0], eps_grid.shape[1])
    obj.Add_LayerUniform(lam_nm, 1.0)                    # semi-infinite substrate (air)
    obj.Init_Setup()
    obj.GridLayer_geteps(eps_grid.flatten().astype(complex))
    obj.MakeExcitationPlanewave(1.0, 0.0, 0.0, 0.0, order=0)   # square pillar = C4v -> pol-independent
    R, T = obj.RT_Solve(normalize=1)
    return float(np.real(R)), float(np.real(T))


def _square_eps_grid(ng):
    p_nm = PERIOD * 1e9
    pil_nm = PILLAR * 1e9
    xs = (np.arange(ng) + 0.5) * p_nm / ng
    X, Y = np.meshgrid(xs, xs, indexing="ij")
    ep = np.ones((ng, ng))
    ep[(np.abs(X - p_nm / 2) <= pil_nm / 2) & (np.abs(Y - p_nm / 2) <= pil_nm / 2)] = EPS_HI
    return ep


def _pillar_design():
    reg = MaterialRegistry()
    reg.add(Material("air", ConstantOptical(1.0 + 0j)))
    reg.add(Material("hi", ConstantOptical(EPS_HI + 0j)))
    pillar = Rectangle(PERIOD / 2, PERIOD / 2, PILLAR, PILLAR)   # centered square
    layer = Layer("pillar", THICK, "air", inclusions=[Inclusion(pillar, "hi")])
    stack = Stack(layers=[layer], superstrate_material="air", substrate_material="air")
    return Design(name="pillar", unit_cell=UnitCell.square(PERIOD), stack=stack, electrodes=[], materials=reg)


def main():
    print("[fr] === FDTD structured seam vs grcwa (RCWA): 2D-periodic dielectric pillar array ===", flush=True)

    # GATE 0: grcwa on a UNIFORM high-index slab == analytic Airy (the RCWA oracle is set up right)
    Ru, Tu = _rcwa_rt(np.full((8, 8), EPS_HI), THICK * 1e9, nG=21)
    Ra, Ta = airy(C / LAM, np.sqrt(EPS_HI), THICK)
    d0 = max(abs(Ru - Ra), abs(Tu - Ta))
    gate0 = bool(d0 < 2e-3)
    print("[fr] 0 grcwa uniform slab: R={:.4f} T={:.4f} | Airy R={:.4f} T={:.4f} | max|d|={:.2e} -> {}".format(
        Ru, Tu, Ra, Ta, d0, "PASS" if gate0 else "FAIL"), flush=True)

    # grcwa structured pillar (the oracle)
    Rr, Tr = _rcwa_rt(_square_eps_grid(96), THICK * 1e9, nG=121)

    # FDTD structured seam (rasterized geometry, time-domain). res kept modest so the persistent gate runs
    # in a few minutes; two utterly different methods agreeing to a few % is the cross-check.
    solver = make_fdtd_optical_solver(dim=3, resolution=16, band_frac=0.16, n_pad_wave=3.0)
    res = solver(_pillar_design(), None, {}, LAM, 1.0 + 0j, 1.0 + 0j)
    Rf, Tf = float(res.R_flux), float(res.T_flux)

    dR, dT = abs(Rf - Rr), abs(Tf - Tr)
    en = abs(Rf + Tf - 1.0)
    gate1 = bool(dR < 4e-2 and dT < 4e-2 and en < 4e-2)
    print("[fr] 1 pillar: FDTD R={:.4f} T={:.4f} (0-order specular R={:.4f}) | RCWA R={:.4f} T={:.4f} | "
          "|dR|={:.2e} |dT|={:.2e} R+T-1={:.2e} -> {}".format(
              Rf, Tf, float(res.R), Rr, Tr, dR, dT, en, "PASS" if gate1 else "FAIL"), flush=True)

    overall = gate0 and gate1
    print("[fr] *** FDTD STRUCTURED SEAM vs RCWA (rasterized pillar array; all-order R/T; energy): {} ***".format(
        "PASS" if overall else "FAIL"), flush=True)
    return overall


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
