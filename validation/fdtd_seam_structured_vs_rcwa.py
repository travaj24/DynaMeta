"""FDTD STRUCTURED seam vs grcwa (RCWA) -- the gate that the lateral-inclusion RASTERIZATION is correct
(Design inclusions -> the (nx,ny,nz) FDTD eps grid). A 2D-periodic dielectric square-pillar array is
solved two completely independent ways:
  * optics.fdtd_seam.make_fdtd_optical_solver(dim=3) -- time-domain FDTD, geometry rasterized; and
  * grcwa -- frequency-domain RCWA (Fourier-space), the same geometry on an eps grid.
The R/T must agree to a few % (FDTD spatial discretization + RCWA Fourier truncation), and energy must
close. This is where FDTD earns its keep over TMM (which is exact only for UNIFORM stacks). A SUB-
wavelength pillar (0-order only) keeps the 3D grid small/fast while still exercising the full rasterize
path; genuine (kx,ky) diffraction is validated separately in fdtd_3d_reduces.py GATE C.

POLARIZATION -- both sides are E along lab y. The 3-D FDTD launches a y-polarized soft plane source
(kernels3d.run_3d: `Eyn[:, :, k_src] += src[n]`) and reads the 0-order from Ey, so the Design must SAY
polarization='y'; the seam's audit C5-7 guard refuses a structured cell whose OpticalSpec says anything
else rather than hand back the y answer under an x label. Drift caught by the first-ever COMPLETED
full-tier nightly (2026-08-30): this script (2026-06-07) predates that guard (2026-07-11), so it never
declared `optical` at all and inherited the OpticalSpec default 'x', while its grcwa oracle was excited
with p_amp=1 -- which at theta=phi=0 is E along lab x (measured: |Ex(G=0)|=1, |Ey(G=0)|=0). The gate was
therefore an x-pol reference against a y-pol solve, and passed only because the square pillar is C4v, so
x and y are degenerate to round-off (measured |R_x - R_y| = 4.8e-15; GATE 0b below pins it). It is no
longer LEANING on that degeneracy: the Design declares 'y' and grcwa is driven s_amp=1 (= E along lab y
at normal incidence).

GATE 0  (oracle sanity): grcwa on a UNIFORM slab == analytic Airy (confirms the RCWA setup is correct).
GATE 0b (C4v degeneracy): the pillar oracle driven on lab x == driven on lab y, so the symmetry the
        pre-guard version silently leaned on is now a MEASURED number, not a comment.
GATE 1  (the test): FDTD-structured R/T == grcwa-structured (both y-polarized), and R+T = 1.

Run: python -m validation.fdtd_seam_structured_vs_rcwa
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# CAPABILITY GUARD (audit C6-6 spelling; see validation/fdtd_numba_cuda.py). `grcwa` is the
# third-party RCWA oracle this gate measures the FDTD seam against. It is NOT a declared
# dependency of this repo, so a bare `import grcwa` crashes with rc=1 -- which run_all counts
# as a FAILURE -- on any machine that lacks it. The 2026-08-30 nightly (run 33312685617, the
# validation tier's first-ever complete execution) is where that showed up: this script died in
# 1 s at this line, indistinguishable in the summary from a physics failure. Exit 42 instead:
# run_all counts that separately and treats it as informational outside the smoke tier.
try:
    import grcwa
except ModuleNotFoundError:
    print("[sr] grcwa is not installed -> SKIP (exit 42; run_all counts it separately, audit "
          "C6-6). This gate compares the structured FDTD seam against grcwa's RCWA solution of "
          "the SAME pillar cell; without it there is no oracle to compare to. "
          "pip install grcwa", flush=True)
    raise SystemExit(42)

from dynameta.geometry import Design, Layer, Stack, UnitCell
from dynameta.geometry.cross_section import Rectangle
from dynameta.geometry.specs import OpticalSpec
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


def _rcwa_rt(eps_grid, thick_nm, nG=101, pol="y"):
    """grcwa total R/T for ONE patterned layer (eps_grid, Ng x Ng) in vacuum, normal incidence, for the
    LAB AXIS `pol` of the incident E ('y' = the axis the FDTD source launches; see the module docstring).

    grcwa's excitation is (p_amp, p_phase, s_amp, s_phase) in the PLANE-OF-INCIDENCE spelling, which at
    theta=phi=0 degenerates onto the lab axes: driving s_amp puts E on lab y and p_amp puts E on lab x
    (verified against Solve_FieldFourier in the incident half-space: s_amp=1 -> |Ey(G=0)|=1, |Ex(G=0)|=0).
    Defaulting to 'y' is what makes this an apples-to-apples comparison with the y-polarized FDTD source
    -- the pre-2026-08-30 call was p_amp=1, i.e. the lab-x reference (audit C5-7)."""
    p_amp, s_amp = (1.0, 0.0) if pol == "x" else (0.0, 1.0)
    lam_nm = LAM * 1e9
    p_nm = PERIOD * 1e9
    freq = 1.0 / lam_nm                                  # grcwa: c = 1, freq = 1/lambda
    obj = grcwa.obj(nG, [p_nm, 0.0], [0.0, p_nm], freq, 0.0, 0.0, verbose=0)
    obj.Add_LayerUniform(lam_nm, 1.0)                    # semi-infinite superstrate (air)
    obj.Add_LayerGrid(thick_nm, eps_grid.shape[0], eps_grid.shape[1])
    obj.Add_LayerUniform(lam_nm, 1.0)                    # semi-infinite substrate (air)
    obj.Init_Setup()
    obj.GridLayer_geteps(eps_grid.flatten().astype(complex))
    obj.MakeExcitationPlanewave(p_amp, 0.0, s_amp, 0.0, order=0)
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
    # polarization='y' is the physics, not a formality: the 3-D FDTD kernel injects the source on Ey.
    # Leaving it at the OpticalSpec default ('x') made this a y-solve wearing an x label, which the
    # audit C5-7 seam guard now (correctly) refuses -- see the module docstring for the 2026-08-30 catch.
    return Design(name="pillar", unit_cell=UnitCell.square(PERIOD), stack=stack, electrodes=[], materials=reg,
                  optical=OpticalSpec(polarization="y", incidence_angle_deg=0.0))


def main():
    print("[fr] === FDTD structured seam vs grcwa (RCWA): 2D-periodic dielectric pillar array ===", flush=True)

    # GATE 0: grcwa on a UNIFORM high-index slab == analytic Airy (the RCWA oracle is set up right)
    Ru, Tu = _rcwa_rt(np.full((8, 8), EPS_HI), THICK * 1e9, nG=21)
    Ra, Ta = airy(C / LAM, np.sqrt(EPS_HI), THICK)
    d0 = max(abs(Ru - Ra), abs(Tu - Ta))
    gate0 = bool(d0 < 2e-3)
    print("[fr] 0 grcwa uniform slab: R={:.4f} T={:.4f} | Airy R={:.4f} T={:.4f} | max|d|={:.2e} -> {}".format(
        Ru, Tu, Ra, Ta, d0, "PASS" if gate0 else "FAIL"), flush=True)

    # grcwa structured pillar (the oracle), driven on the SAME lab axis the FDTD source uses (y).
    grid = _square_eps_grid(96)
    Rr, Tr = _rcwa_rt(grid, THICK * 1e9, nG=121, pol="y")

    # GATE 0b (C4v degeneracy, MEASURED not assumed): a square pillar at normal incidence has no
    # preferred lab axis, so the x-driven and y-driven oracles must agree to round-off. This is the
    # symmetry the pre-audit-C5-7 script leaned on WITHOUT saying so -- it compared an x-pol grcwa
    # reference against the FDTD's y-pol solve and got away with it. Pinning it here means a future
    # edit that breaks the C4v cell (an elongated rectangle, an off-centre inclusion) shows up as a
    # FAIL instead of quietly re-opening the same trap. Sanity of the probe itself: the same
    # comparison on a 420x180 nm rectangle separates by |dR| = 7.2e-2, so it is not blind.
    Rx, Tx = _rcwa_rt(grid, THICK * 1e9, nG=121, pol="x")
    dxy = max(abs(Rr - Rx), abs(Tr - Tx))
    gate0b = bool(dxy < 1e-9)
    print("[fr] 0b pillar oracle x/y degeneracy (C4v): R_y={:.6f} R_x={:.6f} | max|d|={:.2e} -> {}".format(
        Rr, Rx, dxy, "PASS" if gate0b else "FAIL"), flush=True)

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

    overall = gate0 and gate0b and gate1
    print("[fr] *** FDTD STRUCTURED SEAM vs RCWA (rasterized pillar array; all-order R/T; energy): {} ***".format(
        "PASS" if overall else "FAIL"), flush=True)
    return overall


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
