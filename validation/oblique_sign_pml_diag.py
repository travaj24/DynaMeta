"""Diagnose the residual oblique error: is it the TEST-side kcross sign (-1 from the
Galerkin derivation vs +1 in the research sketch) or the PML thickness (500nm < lam)?
Solve the air / n=2 (250nm) / n=1.5 slab at theta=30deg over the 2x2 grid
{sign in -1,+1} x {pml in 500nm, 1300nm} and compare to tmm (R=0.173, T=0.827).
Mesh is rebuilt per PML thickness; the sign only changes the bilinear form (reuse mesh).
Run:  python -m validation.oblique_sign_pml_diag
"""
import sys, os, math
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tmm
import dynameta.optics.solver as solver_mod
from dynameta.materials import Material, MaterialRegistry, ConstantOptical
from dynameta.geometry import UnitCell, Stack, Layer, Design
from dynameta.geometry.specs import OpticalSpec, Mesh3DSpec
from dynameta.optics.ngsolve_layered import LayeredOpticalBuilder
from dynameta.optics.solver import solve_fem

LAM_NM, N_SLAB, N_SUB, D_SLAB_NM, THETA = 1300.0, 2.0, 1.5, 250.0, 30.0

def build(pml_nm):
    reg = MaterialRegistry()
    reg.add(Material("air",  ConstantOptical(1.0 + 0j)))
    reg.add(Material("slab", ConstantOptical(complex(N_SLAB**2, 0.0))))
    reg.add(Material("sub",  ConstantOptical(complex(N_SUB**2, 0.0))))
    cell = UnitCell.square(220e-9)
    stack = Stack(layers=[Layer("slab", D_SLAB_NM*1e-9, "slab")],
                   superstrate_material="air", substrate_material="sub")
    m3 = Mesh3DSpec(pml_thk_m=pml_nm*1e-9, superstrate_buffer_m=1400e-9,
                     substrate_buffer_m=1400e-9, maxh_superstrate_m=45e-9,
                     maxh_substrate_m=45e-9, maxh_background_m=20e-9, fem_order=2)
    return Design(name="diag", unit_cell=cell, stack=stack, electrodes=[],
                    materials=reg, mesh_3d=m3)

def main():
    th = math.radians(THETA)
    ref = tmm.coh_tmm('s', [1.0, complex(N_SLAB), complex(N_SUB)],
                       [np.inf, D_SLAB_NM, np.inf], th, LAM_NM)
    Rt, Tt = ref['R'], ref['T']
    print("[t] oblique sign/PML diagnostic at theta={:.0f}deg  tmm: R={:.4f} T={:.4f} R+T={:.4f}".format(
        THETA, Rt, Tt, Rt + Tt), flush=True)
    print("[t] {:>8s} {:>6s} | {:>8s} {:>8s} {:>8s} | dR     dT".format(
        "pml(nm)", "sign", "R", "T", "R+T"), flush=True)
    lam_m = LAM_NM * 1e-9
    best = None
    for pml_nm in (500.0, 1300.0):
        d = build(pml_nm)
        geo = LayeredOpticalBuilder(d).build()
        eps_vals = {r: complex(d.materials.get(geo.material_by_region[r]).eps(lam_m))
                    for r in geo.mesh.GetMaterials()}
        eps_cf = geo.mesh.MaterialCF(eps_vals, default=1.0)
        for sign in (-1.0, +1.0):
            solver_mod._TEST_KCROSS_SIGN = sign
            opt = OpticalSpec(polarization="y", incidence_angle_deg=THETA,
                               linear_solver="umfpack")
            res = solve_fem(geo, lam_m, eps_cf, opt, order=2,
                             n_super=1.0+0j, n_sub=complex(N_SUB, 0.0))
            Rf = res.R; Tf = res.T if res.T is not None else float('nan')
            dR, dT = abs(Rf - Rt), abs(Tf - Tt)
            score = dR + dT
            tag = ""
            if best is None or score < best[0]:
                best = (score, pml_nm, sign); tag = " <-"
            print("[t] {:8.0f} {:+6.0f} | {:8.4f} {:8.4f} {:8.4f} | {:.3f}  {:.3f}{}".format(
                pml_nm, sign, Rf, Tf, Rf + (Tf if Tf == Tf else 0.0), dR, dT, tag), flush=True)
    print("[t] best: pml={:.0f}nm sign={:+.0f} (dR+dT={:.3f})".format(
        best[1], best[2], best[0]), flush=True)

if __name__ == "__main__":
    main()
