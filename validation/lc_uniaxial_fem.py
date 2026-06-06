"""Phase-4 liquid-crystal uniaxial-tensor FEM oracle (roadmap 4b end-to-end): the LiquidCrystalModel
uniaxial permittivity tensor flows through the Phase-0b tensor-eps FEM, checked against the analytic
uniaxial result at the two PRINCIPAL director orientations -- the physically dominant LC switching
states:

  * PLANAR (theta = 0, optic axis along x): eps = diag(n_e^2, n_o^2, n_o^2). A y-polarized
    (ordinary) wave sees n_o; an x-polarized (extraordinary) wave sees n_e.
  * HOMEOTROPIC (theta = 90 deg, optic axis along z): eps = diag(n_o^2, n_o^2, n_e^2). Both the
    y- and the (normal-incidence) x-polarized waves see n_o.

The tensor FEM R/T must match the scalar TMM at the corresponding index. Reuses the tensor
assembler + curl-curl matvec validated isotropically by tensor_isotropic_gate.py.

The INTERMEDIATE-tilt (0 < theta < 90) off-diagonal case -- once a tracked limitation -- is now
SOLVED and has its own end-to-end oracle in validation/lc_tilted_fem.py (the off-diagonal failure was
mesh.SetPML's coordinate stretch being wrong for an anisotropic medium, fixed by the explicit UPML in
solver.solve_fem). This file keeps the two principal (diagonal) states as the focused isotropic-
reduction gate. Run: python -m validation.lc_uniaxial_fem
"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dynameta.materials import Material, MaterialRegistry, ConstantOptical
from dynameta.geometry import UnitCell, Stack, Layer, Design
from dynameta.geometry.specs import OpticalSpec, Mesh3DSpec
from dynameta.core.eps_field import EpsField
from dynameta.core.effects import LiquidCrystalModel
from dynameta.core.layered import LayeredStack, LayeredSlab
from dynameta.optics.ngsolve_layered import LayeredOpticalBuilder
from dynameta.optics.eps_assembler import assemble_eps_cf
from dynameta.optics.solver import solve_fem
from dynameta.optics.tmm_reference import TmmLayeredSolver

LAM = 1550.0
NO, NE = 1.53, 1.71          # nematic LC ordinary / extraordinary indices
L = 500.0                    # slab thickness (nm)
TOL_RT = 0.02


def _design():
    reg = MaterialRegistry()
    reg.add(Material("air", ConstantOptical(1.0 + 0j)))
    reg.add(Material("lc", ConstantOptical(complex(NO ** 2, 0.0))))
    cell = UnitCell.square(300e-9)
    stack = Stack(layers=[Layer("s", L * 1e-9, "lc")],
                  superstrate_material="air", substrate_material="air")
    m3 = Mesh3DSpec(pml_thk_m=600e-9, superstrate_buffer_m=900e-9, substrate_buffer_m=900e-9,
                    maxh_superstrate_m=45e-9, maxh_substrate_m=45e-9, maxh_background_m=22e-9)
    return Design(name="lc", unit_cell=cell, stack=stack, electrodes=[], materials=reg, mesh_3d=m3)


def _fem_rt(eps_tensor, pol):
    # build a FRESH geometry per solve: reusing one geo across solves (esp. mixing polarizations)
    # leaks state on the periodic PML mesh and corrupts later solves (a tracked tensor-FEM follow-on;
    # the thermo/Pockels oracles only ever reused geo within a SINGLE polarization). A fresh mesh
    # per solve is the clean, correct path for this multi-polarization check.
    geo = LayeredOpticalBuilder(_design()).build()
    mats = list(geo.mesh.GetMaterials())
    slab = [r for r in mats if geo.material_by_region[r] == "lc"][0]
    ebr = {rg: EpsField(scalar=complex(1.0, 0.0)) for rg in mats}
    ebr[slab] = EpsField(tensor=np.asarray(eps_tensor, dtype=complex))
    opt = OpticalSpec(polarization=pol, incidence_angle_deg=0.0, linear_solver="umfpack")
    res = solve_fem(geo, LAM * 1e-9, assemble_eps_cf(geo, ebr), opt, order=2,
                    n_super=1.0 + 0j, n_sub=1.0 + 0j)
    return res.R, res.T


def _tmm_rt(eps_scalar, pol):
    opt = OpticalSpec(polarization=pol, incidence_angle_deg=0.0, linear_solver="umfpack")
    stk = LayeredStack(1.0 + 0j, 1.0 + 0j, [LayeredSlab(L * 1e-9, eps=complex(eps_scalar, 0.0))])
    rtm = TmmLayeredSolver().solve(stk, LAM * 1e-9, opt)
    return rtm.R, rtm.T


def main():
    lc = LiquidCrystalModel(n_o=NO, n_e=NE)
    # principal-orientation cases: (label, theta_deg, polarization, expected scalar index)
    cases = [("planar  ord(y)", 0.0, "y", NO), ("planar  ext(x)", 0.0, "x", NE),
             ("homeotr ord(y)", 90.0, "y", NO), ("homeotr ext(x)", 90.0, "x", NO)]
    ok = True
    for label, th_deg, pol, n_exp in cases:
        # take the model's tensor at the principal angle and keep its DIAGONAL (the principal
        # states ARE diagonal; this drops only the ~6e-17 cos(pi/2) numerical residual, not a real
        # off-diagonal -- the intermediate-tilt off-diagonal regime is the deferred+warned one).
        eps = np.diag(np.diag(lc.eps({"director_angle_rad": np.radians(th_deg)}, LAM * 1e-9)))
        Rf, Tf = _fem_rt(eps, pol)
        Rt, Tt = _tmm_rt(n_exp ** 2, pol)
        dR, dT = abs(Rf - Rt), abs(Tf - Tt)
        print("[lc] {} theta={:4.0f} deg: R_fem={:.4f} R_tmm(n={:.2f})={:.4f} dR={:.2e} dT={:.2e}".format(
            label, th_deg, Rf, n_exp, Rt, dR, dT), flush=True)
        ok = ok and dR < TOL_RT and dT < TOL_RT

    print("[lc] *** LC UNIAXIAL TENSOR FEM (planar + homeotropic principal states == TMM; "
          "tilted/off-diagonal now solved -- see lc_tilted_fem): {} ***".format("PASS" if ok else "FAIL"),
          flush=True)
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
