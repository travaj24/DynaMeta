"""Cross-check the FEM linear solvers (ported from Metasurface_Modulator
stage3_optical/fem/compare_solvers.py, the non-RCWA "iterative scaling" work). The same
Maxwell solve is run with the UMFPACK direct solver and the two BDDC-preconditioned
iterative solvers (GMRes, CG); the reflectance |r|^2 must AGREE to the iterative tolerance.
This regression-guards the iterative path -- the O(n)-memory route that scales to fine
meshes where the direct solver is infeasible (the audit's "HYPRE"/scaling item). It also
reports wall time per solver. Run: python -m validation.solver_comparison
"""
import sys, os, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dynameta.materials import Material, MaterialRegistry, ConstantOptical
from dynameta.geometry import UnitCell, Stack, Layer, Design
from dynameta.geometry.specs import OpticalSpec, Mesh3DSpec
from dynameta.optics.ngsolve_layered import LayeredOpticalBuilder
from dynameta.optics.solver import solve_fem

LAM, N_SLAB, D = 1300.0, 2.0, 250.0
TOL = 5e-4


def main():
    reg = MaterialRegistry()
    reg.add(Material("air", ConstantOptical(1.0 + 0j)))
    reg.add(Material("slab", ConstantOptical(complex(N_SLAB ** 2, 0.0))))
    cell = UnitCell.square(220e-9)
    stack = Stack(layers=[Layer("slab", D * 1e-9, "slab")], superstrate_material="air",
                   substrate_material="air")
    m3 = Mesh3DSpec(pml_thk_m=600e-9, superstrate_buffer_m=900e-9, substrate_buffer_m=900e-9,
                     maxh_superstrate_m=45e-9, maxh_substrate_m=45e-9, maxh_background_m=25e-9)
    d = Design(name="solvers", unit_cell=cell, stack=stack, electrodes=[], materials=reg, mesh_3d=m3)
    geo = LayeredOpticalBuilder(d).build()
    lam_m = LAM * 1e-9
    eps_vals = {r: complex(d.materials.get(geo.material_by_region[r]).eps(lam_m))
                for r in geo.mesh.GetMaterials()}
    eps_cf = geo.mesh.MaterialCF(eps_vals, default=1.0)
    print("[t] FEM solver cross-check: air/slab(n={})/air, ne={}".format(N_SLAB, geo.mesh.ne), flush=True)
    Rs = {}
    for solver in ("umfpack", "bddc_gmres", "bddc_cg"):
        opt = OpticalSpec(polarization="y", incidence_angle_deg=0.0, linear_solver=solver,
                           gmres_rtol=1e-7, gmres_max_iter=400)
        t0 = time.time()
        res = solve_fem(geo, lam_m, eps_cf, opt, order=2, n_super=1.0 + 0j, n_sub=1.0 + 0j)
        Rs[solver] = res.R
        print("[t]   {:11s} |r|^2={:.6f}  ({:.1f}s)".format(solver, res.R, time.time() - t0), flush=True)
    spread = max(Rs.values()) - min(Rs.values())
    ok = spread < TOL
    print("[t]   max |dR| across solvers = {:.2e} (tol {:.0e})".format(spread, TOL), flush=True)
    print("[t] *** FEM SOLVER CROSS-CHECK (direct vs BDDC-iterative agree): {} ***".format(
        "PASS" if ok else "FAIL"), flush=True)
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
