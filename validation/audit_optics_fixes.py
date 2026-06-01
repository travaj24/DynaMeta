"""Verify the optics-solver audit fixes (OPT-1, OPT-2, OPT-3) on a lossy slab.

  OPT-2: the INDEPENDENT volumetric absorption A_independent must agree with the
         energy-budget closure A = 1 - R - T (a genuine, non-tautological energy check).
  OPT-1: oblique incidence with a non-vacuum superstrate (n_super != 1) must RAISE
         NotImplementedError, not return a silently-wrong result.
  OPT-3: an oblique solve must emit a (single) not-angle-aware-PML warning.

Lossy slab air / (n=2+0.1i, 250nm) / air at theta=30 deg, s-pol.
Run: python -m validation.audit_optics_fixes
"""
import sys, os, math, warnings
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dynameta.materials import Material, MaterialRegistry, ConstantOptical
from dynameta.geometry import UnitCell, Stack, Layer, Design
from dynameta.geometry.specs import OpticalSpec, Mesh3DSpec
from dynameta.optics.ngsolve_layered import LayeredOpticalBuilder
from dynameta.optics.solver import solve_fem

LAM, N_SLAB, KAPPA, D, THETA = 1300.0, 2.0, 0.1, 250.0, 30.0
TOL = 0.02


def _geo_eps():
    reg = MaterialRegistry()
    reg.add(Material("air", ConstantOptical(1.0 + 0j)))
    reg.add(Material("slab", ConstantOptical(complex((N_SLAB + 1j * KAPPA) ** 2))))
    cell = UnitCell.square(220e-9)
    stack = Stack(layers=[Layer("slab", D * 1e-9, "slab")], superstrate_material="air",
                   substrate_material="air")
    m3 = Mesh3DSpec(pml_thk_m=700e-9, superstrate_buffer_m=1500e-9, substrate_buffer_m=1500e-9,
                     maxh_superstrate_m=40e-9, maxh_substrate_m=40e-9, maxh_background_m=20e-9)
    d = Design(name="lossy", unit_cell=cell, stack=stack, electrodes=[], materials=reg, mesh_3d=m3)
    geo = LayeredOpticalBuilder(d).build()
    lam_m = LAM * 1e-9
    eps_vals = {r: complex(d.materials.get(geo.material_by_region[r]).eps(lam_m))
                for r in geo.mesh.GetMaterials()}
    return geo, geo.mesh.MaterialCF(eps_vals, default=1.0), lam_m


def main():
    geo, eps_cf, lam_m = _geo_eps()
    ok = True

    # OPT-2 + OPT-3: solve at theta=30 (vacuum superstrate), capture the warning.
    opt = OpticalSpec(polarization="y", incidence_angle_deg=THETA, linear_solver="umfpack")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = solve_fem(geo, lam_m, eps_cf, opt, order=2, n_super=1.0 + 0j, n_sub=1.0 + 0j)
    warned = any("angle-aware" in str(w.message) for w in caught)
    A_budget = res.A
    A_indep = res.A_independent
    print("[t] R={:.4f} T={:.4f}  A(budget=1-R-T)={:.4f}  A_independent={}".format(
        res.R, res.T if res.T is not None else float("nan"), A_budget,
        "None" if A_indep is None else "{:.4f}".format(A_indep)), flush=True)
    if A_indep is None:
        print("[t] OPT-2 FAIL: A_independent not computed", flush=True); ok = False
    else:
        dA = abs(A_indep - A_budget)
        good = dA < TOL and A_budget > 0.02       # genuinely lossy + the two agree
        ok = ok and good
        print("[t] OPT-2 independent-vs-budget |dA|={:.4f} ({})".format(
            dA, "OK" if good else "MISMATCH"), flush=True)
    print("[t] OPT-3 oblique not-angle-aware warning emitted: {}".format(
        "OK" if warned else "MISSING"), flush=True)
    ok = ok and warned

    # OPT-1: a non-vacuum superstrate at oblique must RAISE.
    try:
        solve_fem(geo, lam_m, eps_cf, opt, order=2, n_super=1.5 + 0j, n_sub=1.0 + 0j)
        print("[t] OPT-1 FAIL: dense-superstrate oblique did NOT raise", flush=True); ok = False
    except NotImplementedError:
        print("[t] OPT-1 dense-superstrate oblique correctly raised NotImplementedError", flush=True)

    print("[t] *** OPTICS AUDIT FIXES (OPT-1/2/3): {} ***".format("PASS" if ok else "FAIL"), flush=True)
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
