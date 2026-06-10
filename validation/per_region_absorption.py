"""D2 per-region absorbed-power map oracle (FEM per-region + TMM per-layer).

A 3-layer laterally-uniform stack with ONE lossy middle layer (lossless cladding) is solved by
the FEM and by coherent TMM, and the absorbed-power MAPS are cross-checked:

GATE A (additivity closure, FEM): sum(per_region_absorption.values()) == A_independent EXACTLY
        (the map is the same volumetric loss integral split by material domain) -- < 1e-10.
GATE B (lossless regions identically zero): regions with Im(eps) = 0 contribute EXACTLY 0.0
        (the loss integrand vanishes pointwise); the lossy layer carries ALL of A_independent.
GATE C (independent TMM oracle): tmm.absorp_in_each_layer per-layer fractions match the FEM
        per-region values layer by layer (TMM exact, FEM converges; tol 2.5e-2).
GATE D (TMM closure + solver seam): the TMM per-layer dict sums to A = 1 - R - T to 1e-9, and
        TmmLayeredSolver populates OpticalResult.per_region_absorption.

Run: python -m validation.per_region_absorption
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dynameta.materials import Material, MaterialRegistry, ConstantOptical
from dynameta.geometry import UnitCell, Stack, Layer, Design
from dynameta.geometry.specs import OpticalSpec, Mesh3DSpec
from dynameta.optics.ngsolve_layered import LayeredOpticalBuilder
from dynameta.optics.solver import solve_fem
from dynameta.optics.tmm_reference import (layered_stack_from_design,
                                           layered_per_layer_absorption, TmmLayeredSolver)

LAM = 1300.0
EPS_BY_LAYER = [(1.6 ** 2) + 0j, 4.0 + 0.5j, (1.6 ** 2) + 0j]   # lossless / LOSSY / lossless
DZ = [120.0, 80.0, 120.0]                                       # nm


def _design():
    reg = MaterialRegistry()
    reg.add(Material("air", ConstantOptical(1.0 + 0j)))
    layers = []
    for k, (e, dz) in enumerate(zip(EPS_BY_LAYER, DZ)):
        reg.add(Material("m%d" % k, ConstantOptical(e)))
        layers.append(Layer("s%d" % k, dz * 1e-9, "m%d" % k))
    cell = UnitCell.square(220e-9)
    stack = Stack(layers=layers, superstrate_material="air", substrate_material="air")
    m3 = Mesh3DSpec(pml_thk_m=600e-9, superstrate_buffer_m=900e-9, substrate_buffer_m=900e-9,
                    maxh_superstrate_m=40e-9, maxh_substrate_m=40e-9, maxh_background_m=15e-9)
    return Design(name="absmap", unit_cell=cell, stack=stack, electrodes=[], materials=reg,
                  mesh_3d=m3)


def main():
    print("[pa] === D2 per-region absorbed-power map ===", flush=True)
    ok = True
    d = _design()
    lam_m = LAM * 1e-9

    # ---- FEM solve ----
    geo = LayeredOpticalBuilder(d).build()
    eps_vals = {r: complex(d.materials.get(geo.material_by_region[r]).eps(lam_m))
                for r in geo.mesh.GetMaterials()}
    eps_cf = geo.mesh.MaterialCF(eps_vals, default=1.0)
    opt = OpticalSpec(polarization="y", incidence_angle_deg=0.0, linear_solver="umfpack")
    res = solve_fem(geo, lam_m, eps_cf, opt, order=2, n_super=1.0 + 0j, n_sub=1.0 + 0j)
    pr = res.per_region_absorption
    if pr is None or res.A_independent is None:
        print("[pa] FAIL: per_region_absorption/A_independent not computed", flush=True)
        return False

    # ---- GATE A: additivity closure ----
    gapA = abs(sum(pr.values()) - res.A_independent)
    g_a = bool(gapA < 1e-10)
    ok = ok and g_a
    print("[pa] GATE A: sum(per-region) - A_independent = {:.2e} (A_ind = {:.4f}) -> {}".format(
        gapA, res.A_independent, "PASS" if g_a else "FAIL"), flush=True)

    # ---- GATE B: lossless regions identically zero; lossy layer carries all ----
    lossless = {k: v for k, v in pr.items() if k != "s1"}
    zero_ok = all(v == 0.0 for v in lossless.values())
    lossy_ok = abs(pr["s1"] - res.A_independent) < 1e-12
    g_b = bool(zero_ok and lossy_ok)
    ok = ok and g_b
    print("[pa] GATE B: lossless regions exactly 0.0 ({}); s1 == A_independent ({:.4f}) -> {}"
          .format(zero_ok, pr["s1"], "PASS" if g_b else "FAIL"), flush=True)

    # ---- GATE C: independent TMM per-layer oracle ----
    stack = layered_stack_from_design(d, lam_m)
    per_layer, A_tmm = layered_per_layer_absorption(stack, lam_m, theta_deg=0.0, pol="s")
    dmid = abs(pr["s1"] - per_layer["slab_1"])
    dside = max(abs(pr["s0"] - per_layer["slab_0"]), abs(pr["s2"] - per_layer["slab_2"]))
    g_c = bool(dmid < 2.5e-2 and dside < 1e-9 and abs(res.A_independent - A_tmm) < 2.5e-2)
    ok = ok and g_c
    print("[pa] GATE C: FEM vs TMM -- lossy layer |d| = {:.2e} (TMM {:.4f}), lossless layers "
          "|d| = {:.1e}, total A |d| = {:.2e} -> {}".format(
              dmid, per_layer["slab_1"], dside, abs(res.A_independent - A_tmm),
              "PASS" if g_c else "FAIL"), flush=True)

    # ---- GATE D: TMM closure + solver seam ----
    closure = abs(sum(per_layer.values()) - A_tmm)
    seam = TmmLayeredSolver().solve(stack, lam_m, opt).per_region_absorption
    g_d = bool(closure < 1e-9 and seam is not None
               and abs(seam["slab_1"] - per_layer["slab_1"]) == 0.0)
    ok = ok and g_d
    print("[pa] GATE D: TMM per-layer sums to A (|d| = {:.1e}); TmmLayeredSolver carries the "
          "map -> {}".format(closure, "PASS" if g_d else "FAIL"), flush=True)

    print("[pa] *** D2 PER-REGION ABSORBED POWER: {} ***".format("PASS" if ok else "FAIL"),
          flush=True)
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
