"""Validate STRUCTURED (laterally-patterned) FDTD cells on a NON-VACUUM semi-infinite substrate (e.g. a
metasurface ON GLASS) -- previously deferred because the lateral rasterizer rebuilt the eps grid and
dropped the n_super/n_sub end-media pads. make_structured_lateral now PAINTS the pads (n_super^2/n_sub^2)
into the rasterized grid, so the structured run carries the correct end media (the incident reference,
impedance-matched CPML and Snell T-factor in solve_fdtd_3d already handle non-vacuum).

POLARIZATION -- every side of every gate is E along lab y. The 3-D FDTD launches a y-polarized soft
plane source (kernels3d.run_3d: `Eyn[:, :, k_src] += src[n]`) and reads the 0-order from Ey, so the
Design must SAY polarization='y'; the seam's audit C5-7 guard refuses a structured cell whose OpticalSpec
says anything else rather than hand back the y answer under an x label. Drift caught by the first-ever
COMPLETED full-tier nightly (2026-08-30): this script (2026-06-08) predates that guard (2026-07-11), so
it never declared `optical` and inherited the OpticalSpec default 'x'. GATE A's tmm oracle was already
right ('s' = E perpendicular to the x-z plane of incidence = lab y). GATE B's grcwa oracle was NOT: it
was driven p_amp=1, which at theta=phi=0 is E along lab x (measured: |Ex(G=0)|=1, |Ey(G=0)|=0), so the
gate compared an x-pol reference against a y-pol solve and survived only on the pillar's C4v degeneracy
(measured |R_x - R_y| = 1.8e-15, |T_x - T_y| = 1.7e-14 with the glass substrate). GATE B is now driven
s_amp=1 = lab y and leans on nothing.

GATE A (the fix, direct): a STRUCTURED layer whose inclusion == its background (so it is geometrically a
        UNIFORM slab) on glass (n_sub=1.5) reproduces coherent TMM(air/slab/glass). If the substrate pad
        had reverted to vacuum (the old bug) this would be wrong.
GATE B (genuine structured on glass): a sub-wavelength dielectric pillar array on glass matches grcwa
        (RCWA, substrate eps=n_sub^2) and energy closes (R_flux + T_flux = 1, the flux carries n).

Run: python -m validation.fdtd_structured_nonvacuum
"""
import os
import sys
import math

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tmm

# CAPABILITY GUARD (audit C6-6 spelling; see validation/fdtd_numba_cuda.py). `grcwa` is the
# third-party RCWA oracle GATE B measures against, and it is NOT a declared dependency of this
# repo -- so a bare `import grcwa` crashed this script with rc=1 (a FAILURE in run_all's
# summary, indistinguishable from a physics defect) on the 2026-08-30 nightly, the validation
# tier's first-ever complete execution. `tmm` above needs no such guard: it IS declared (the
# `reference` extra, and [dev]), so every CI leg has it.
try:
    import grcwa
except ModuleNotFoundError:
    print("[sn] grcwa is not installed -> SKIP (exit 42; run_all counts it separately, audit "
          "C6-6). GATE B compares the FDTD pillar-on-glass solve against grcwa's RCWA answer "
          "for the same cell; without it that oracle does not exist. pip install grcwa",
          flush=True)
    raise SystemExit(42)

from dynameta.geometry import Design, Layer, Stack, UnitCell
from dynameta.geometry.cross_section import Rectangle
from dynameta.geometry.specs import OpticalSpec
from dynameta.geometry.stack import Inclusion
from dynameta.materials import ConstantOptical, Material, MaterialRegistry
from dynameta.optics.fdtd_seam import make_fdtd_optical_solver

C = 299792458.0
LAM = 1300e-9
PERIOD = 600e-9
THICK = 250e-9
N_SLAB = 2.0
N_SUB = 1.5            # glass substrate
PILLAR = 300e-9
EPS_HI = 6.25         # n=2.5 pillar (GATE B)


def main():
    print("[nv] === STRUCTURED FDTD on a NON-VACUUM substrate (metasurface-on-glass) ===", flush=True)
    reg = MaterialRegistry()
    reg.add(Material("air", ConstantOptical(1.0 + 0j)))
    reg.add(Material("slab", ConstantOptical(N_SLAB ** 2 + 0j)))
    reg.add(Material("hi", ConstantOptical(EPS_HI + 0j)))
    reg.add(Material("glass", ConstantOptical(N_SUB ** 2 + 0j)))
    cell = UnitCell.square(PERIOD)
    solver = make_fdtd_optical_solver(dim=3, resolution=18, band_frac=0.16, n_pad_wave=3.0)
    # polarization='y' is the physics, not a formality: the 3-D FDTD kernel injects the source on Ey.
    # Leaving it at the OpticalSpec default ('x') made these y-solves wear an x label, which the audit
    # C5-7 seam guard now (correctly) refuses -- see the module docstring for the 2026-08-30 catch.
    opt = OpticalSpec(polarization="y", incidence_angle_deg=0.0)

    # --- GATE A: structured-but-uniform (full-cell inclusion == a uniform slab) on glass vs TMM ---
    full = Rectangle(PERIOD / 2, PERIOD / 2, PERIOD, PERIOD)              # inclusion = whole cell
    dA = Design(name="unif_struct", unit_cell=cell, electrodes=[], materials=reg, optical=opt,
                stack=Stack(layers=[Layer("slab", THICK, "air", inclusions=[Inclusion(full, "slab")])],
                            superstrate_material="air", substrate_material="glass"))
    resA = solver(dA, None, {}, LAM, 1.0 + 0j, complex(N_SUB))
    # tmm's "s" is the PLANE-OF-INCIDENCE spelling of the same mode the FDTD launches: E perpendicular
    # to the x-z plane of incidence, i.e. E along lab y (dynameta.core.polarization, audit V-8). At
    # theta=0 an isotropic laterally-uniform stack is polarization-degenerate anyway (R_s == R_p
    # exactly), so this reference needed no change -- only the Design's declaration did.
    ref = tmm.coh_tmm("s", [1.0, complex(N_SLAB), complex(N_SUB)], [np.inf, THICK * 1e9, np.inf],
                      0.0, LAM * 1e9)
    dR, dT = abs(resA.R - ref["R"]), abs((resA.T if resA.T is not None else 0.0) - ref["T"])
    gA = bool(dR < 2e-2 and dT < 2e-2)
    print("[nv] A uniform-via-structured on glass: FDTD R={:.4f} T={:.4f} | TMM R={:.4f} T={:.4f} | "
          "|dR|={:.2e} |dT|={:.2e} -> {}".format(resA.R, resA.T, ref["R"], ref["T"], dR, dT,
                                                 "PASS" if gA else "FAIL"), flush=True)

    # --- GATE B: genuine pillar array on glass vs grcwa (substrate eps = n_sub^2) ---
    pillar = Rectangle(PERIOD / 2, PERIOD / 2, PILLAR, PILLAR)
    dB = Design(name="pillar_glass", unit_cell=cell, electrodes=[], materials=reg, optical=opt,
                stack=Stack(layers=[Layer("pil", THICK, "air", inclusions=[Inclusion(pillar, "hi")])],
                            superstrate_material="air", substrate_material="glass"))
    resB = solver(dB, None, {}, LAM, 1.0 + 0j, complex(N_SUB))
    Rf, Tf = float(resB.R_flux), float(resB.T_flux)
    # grcwa oracle: same pillar, substrate eps = n_sub^2
    ng = 96
    xs = (np.arange(ng) + 0.5) * PERIOD * 1e9 / ng
    X, Y = np.meshgrid(xs, xs, indexing="ij")
    ep = np.ones((ng, ng))
    ep[(np.abs(X - PERIOD * 1e9 / 2) <= PILLAR * 1e9 / 2) &
       (np.abs(Y - PERIOD * 1e9 / 2) <= PILLAR * 1e9 / 2)] = EPS_HI
    obj = grcwa.obj(121, [PERIOD * 1e9, 0.0], [0.0, PERIOD * 1e9], 1.0 / (LAM * 1e9), 0.0, 0.0, verbose=0)
    obj.Add_LayerUniform(LAM * 1e9, 1.0)
    obj.Add_LayerGrid(THICK * 1e9, ng, ng)
    obj.Add_LayerUniform(LAM * 1e9, N_SUB ** 2)                          # glass substrate
    obj.Init_Setup()
    obj.GridLayer_geteps(ep.flatten().astype(complex))
    # (p_amp, p_phase, s_amp, s_phase): grcwa's plane-of-incidence spelling degenerates onto the lab
    # axes at theta=phi=0, where s_amp drives E along lab y and p_amp drives E along lab x (verified
    # against Solve_FieldFourier in the incident half-space). s_amp=1 therefore matches the FDTD's
    # y-polarized source; the pre-2026-08-30 call was p_amp=1 = the lab-x reference (audit C5-7).
    obj.MakeExcitationPlanewave(0.0, 0.0, 1.0, 0.0, order=0)
    Rr, Tr = obj.RT_Solve(normalize=1)
    Rr, Tr = float(np.real(Rr)), float(np.real(Tr))
    dRb, dTb, en = abs(Rf - Rr), abs(Tf - Tr), abs(Rf + Tf - 1.0)
    gB = bool(dRb < 4e-2 and dTb < 4e-2 and en < 4e-2)
    print("[nv] B pillar on glass: FDTD R={:.4f} T={:.4f} | RCWA R={:.4f} T={:.4f} | |dR|={:.2e} |dT|={:.2e} "
          "R+T-1={:.2e} -> {}".format(Rf, Tf, Rr, Tr, dRb, dTb, en, "PASS" if gB else "FAIL"), flush=True)

    ok = gA and gB
    print("[nv] *** STRUCTURED + NON-VACUUM FDTD: {} ***".format("PASS" if ok else "FAIL"), flush=True)
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
