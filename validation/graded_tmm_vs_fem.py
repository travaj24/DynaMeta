"""Cross-check the LAYERED-STACK representation against the FEM on a GRADED (multi-sublayer)
laterally-uniform stack -- the de-risking gate for the future RCWA port (RCWA will consume the
same LayeredStack the graded-TMM does here). A symmetric graded-index slab (8 thin sublayers,
lossless) is solved two ways:
  * the NGSolve FEM (the validated 3D solver), and
  * graded-TMM: layered_stack_from_design(...) -> layered_rta (the LayeredStack consumer).
For a laterally-uniform stack the two MUST agree (TMM exact; FEM converges to it), and both
must conserve energy R+T~1 (lossless). Run: python -m validation.graded_tmm_vs_fem
"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dynameta.materials import Material, MaterialRegistry, ConstantOptical
from dynameta.geometry import UnitCell, Stack, Layer, Design
from dynameta.geometry.specs import OpticalSpec, Mesh3DSpec
from dynameta.optics.ngsolve_layered import LayeredOpticalBuilder
from dynameta.optics.solver import solve_fem
from dynameta.optics.tmm_reference import layered_stack_from_design, layered_rta

LAM = 1300.0
N_PROFILE = [1.6, 1.8, 2.0, 2.2, 2.2, 2.0, 1.8, 1.6]    # symmetric graded bump (lossless)
DZ = 30.0                                               # nm per sublayer
TOL = 0.02


def _design():
    reg = MaterialRegistry()
    reg.add(Material("air", ConstantOptical(1.0 + 0j)))
    layers = []
    for k, n in enumerate(N_PROFILE):
        reg.add(Material("m%d" % k, ConstantOptical(complex(n ** 2, 0.0))))
        layers.append(Layer("s%d" % k, DZ * 1e-9, "m%d" % k))
    cell = UnitCell.square(220e-9)
    stack = Stack(layers=layers, superstrate_material="air", substrate_material="air")
    m3 = Mesh3DSpec(pml_thk_m=600e-9, superstrate_buffer_m=900e-9, substrate_buffer_m=900e-9,
                     maxh_superstrate_m=40e-9, maxh_substrate_m=40e-9, maxh_background_m=15e-9)
    return Design(name="graded", unit_cell=cell, stack=stack, electrodes=[], materials=reg, mesh_3d=m3)


def main():
    d = _design()
    lam_m = LAM * 1e-9
    # --- graded-TMM via the LayeredStack representation ---
    stack = layered_stack_from_design(d, lam_m)
    print("[t] LayeredStack: {} slabs, total {:.0f}nm, unstructured={}".format(
        len(stack.slabs), stack.total_thickness_m * 1e9, stack.is_unstructured), flush=True)
    R_tmm, T_tmm, A_tmm = layered_rta(stack, lam_m, theta_deg=0.0, pol="s")
    # --- FEM ---
    geo = LayeredOpticalBuilder(d).build()
    eps_vals = {r: complex(d.materials.get(geo.material_by_region[r]).eps(lam_m))
                for r in geo.mesh.GetMaterials()}
    eps_cf = geo.mesh.MaterialCF(eps_vals, default=1.0)
    opt = OpticalSpec(polarization="y", incidence_angle_deg=0.0, linear_solver="umfpack")
    res = solve_fem(geo, lam_m, eps_cf, opt, order=2, n_super=1.0 + 0j, n_sub=1.0 + 0j)
    R_fem, T_fem = res.R, (res.T if res.T is not None else float("nan"))
    print("[t] graded-TMM:  R={:.4f} T={:.4f} R+T={:.4f}".format(R_tmm, T_tmm, R_tmm + T_tmm), flush=True)
    print("[t] FEM:         R={:.4f} T={:.4f} R+T={:.4f}".format(R_fem, T_fem, R_fem + T_fem), flush=True)
    dR, dT = abs(R_tmm - R_fem), abs(T_tmm - T_fem)
    print("[t] |dR|={:.4f} |dT|={:.4f} (TOL={:.3f})".format(dR, dT, TOL), flush=True)
    ok = (dR < TOL and dT < TOL and abs(R_tmm + T_tmm - 1.0) < 1e-6
          and abs(R_fem + T_fem - 1.0) < TOL and 0.02 < R_tmm < 0.98)
    print("[t] *** GRADED LAYERED-STACK vs FEM: {} ***".format("PASS" if ok else "FAIL"), flush=True)
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
