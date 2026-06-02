"""Isotropic-reduction gate for the v0.3 tensor-eps path (Phase 0b): a 3x3 eps tensor that is
diagonal-isotropic (eps*I) MUST give the SAME R/T/A as the scalar eps. Builds a uniform LOSSY
slab, assembles eps_by_region two ways through assemble_eps_cf -- a scalar EpsField vs a uniform
3x3 tensor EpsField = eps*I -- solves the FEM both ways, and checks they agree. This exercises
the tensor assembler path, the tensor weak-form matvec branch, AND the tensor A_independent loss
integral, proving each reduces EXACTLY to the validated scalar path. Run:
python -m validation.tensor_isotropic_gate
"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dynameta.materials import Material, MaterialRegistry, ConstantOptical
from dynameta.geometry import UnitCell, Stack, Layer, Design
from dynameta.geometry.specs import OpticalSpec, Mesh3DSpec
from dynameta.core.eps_field import EpsField
from dynameta.optics.ngsolve_layered import LayeredOpticalBuilder
from dynameta.optics.eps_assembler import assemble_eps_cf
from dynameta.optics.solver import solve_fem

LAM = 1300.0
EPS_SLAB = complex(6.0, 0.5)    # lossy dielectric -> A > 0 (also exercises the tensor A_independent)
TOL = 1e-6


def _design():
    reg = MaterialRegistry()
    reg.add(Material("air", ConstantOptical(1.0 + 0j)))
    reg.add(Material("slab", ConstantOptical(EPS_SLAB)))
    cell = UnitCell.square(300e-9)
    stack = Stack(layers=[Layer("s", 200e-9, "slab")],
                  superstrate_material="air", substrate_material="air")
    m3 = Mesh3DSpec(pml_thk_m=600e-9, superstrate_buffer_m=900e-9, substrate_buffer_m=900e-9,
                    maxh_superstrate_m=45e-9, maxh_substrate_m=45e-9, maxh_background_m=25e-9)
    return Design(name="iso", unit_cell=cell, stack=stack, electrodes=[], materials=reg, mesh_3d=m3)


def _solve(geo, eps_by_region, lam_m):
    opt = OpticalSpec(polarization="y", incidence_angle_deg=0.0, linear_solver="umfpack")
    cf = assemble_eps_cf(geo, eps_by_region)
    return solve_fem(geo, lam_m, cf, opt, order=2, n_super=1.0 + 0j, n_sub=1.0 + 0j)


def main():
    d = _design()
    lam_m = LAM * 1e-9
    geo = LayeredOpticalBuilder(d).build()
    mats = list(geo.mesh.GetMaterials())
    eps_vals = {r: complex(d.materials.get(geo.material_by_region[r]).eps(lam_m)) for r in mats}
    res_s = _solve(geo, {r: EpsField(scalar=eps_vals[r]) for r in mats}, lam_m)
    res_t = _solve(geo, {r: EpsField(tensor=eps_vals[r] * np.eye(3, dtype=complex)) for r in mats}, lam_m)
    dR = abs(res_s.R - res_t.R)
    dT = abs((res_s.T or 0.0) - (res_t.T or 0.0))
    dA = abs((res_s.A_independent or 0.0) - (res_t.A_independent or 0.0))
    print("[t] scalar : R={:.6f} T={:.6f} A={:+.6f} A_ind={:.6f}".format(
        res_s.R, res_s.T, res_s.A, res_s.A_independent), flush=True)
    print("[t] tensor : R={:.6f} T={:.6f} A={:+.6f} A_ind={:.6f}".format(
        res_t.R, res_t.T, res_t.A, res_t.A_independent), flush=True)
    print("[t] |dR|={:.2e} |dT|={:.2e} |dA_ind|={:.2e} (TOL={:.0e})".format(dR, dT, dA, TOL), flush=True)
    lossy = (res_s.A_independent is not None) and (res_s.A_independent > 1e-4)
    ok = dR < TOL and dT < TOL and dA < TOL and lossy
    print("[t] *** TENSOR ISOTROPIC-REDUCTION GATE: {} ***".format("PASS" if ok else "FAIL"), flush=True)
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
