"""Validate P-POLARIZATION oblique incidence vs the `tmm` library. p-pol (TM) has E in
the x-z plane (Ex, Ez); the background reflection/transmission come from the physical
interface BCs (Ex + Hy continuity) solved numerically, and R/T are extracted from the
reconstructed total field's tangential Ex (with the p-pol Poynting factor). Test a
vacuum-exit slab AND a dense-substrate slab at 0/15/30deg vs tmm.coh_tmm('p', ...).
theta=0 must agree with s-pol (degenerate). Run: python -m validation.oblique_ppol_vs_tmm
"""
import sys, os, math
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tmm
from dynameta.materials import Material, MaterialRegistry, ConstantOptical
from dynameta.geometry import UnitCell, Stack, Layer, Design
from dynameta.geometry.specs import OpticalSpec, Mesh3DSpec
from dynameta.optics.ngsolve_layered import LayeredOpticalBuilder
from dynameta.optics.solver import solve_fem

LAM, N_SLAB, D_SLAB = 1300.0, 2.0, 250.0
ANGLES = (0.0, 15.0, 30.0)
TOL = 0.03


def run(label, n_sub):
    reg = MaterialRegistry()
    reg.add(Material("air", ConstantOptical(1.0 + 0j)))
    reg.add(Material("slab", ConstantOptical(complex(N_SLAB ** 2, 0.0))))
    reg.add(Material("sub", ConstantOptical(complex(n_sub ** 2, 0.0))))
    cell = UnitCell.square(220e-9)
    stack = Stack(layers=[Layer("slab", D_SLAB * 1e-9, "slab")],
                   superstrate_material="air", substrate_material="sub")
    m3 = Mesh3DSpec(pml_thk_m=700e-9, superstrate_buffer_m=1500e-9, substrate_buffer_m=1500e-9,
                     maxh_superstrate_m=40e-9, maxh_substrate_m=40e-9, maxh_background_m=20e-9)
    d = Design(name=label, unit_cell=cell, stack=stack, electrodes=[], materials=reg, mesh_3d=m3)
    geo = LayeredOpticalBuilder(d).build()
    lam_m = LAM * 1e-9
    eps_vals = {r: complex(d.materials.get(geo.material_by_region[r]).eps(lam_m))
                for r in geo.mesh.GetMaterials()}
    eps_cf = geo.mesh.MaterialCF(eps_vals, default=1.0)
    ok = True
    for th_deg in ANGLES:
        opt = OpticalSpec(polarization="p", incidence_angle_deg=th_deg, linear_solver="umfpack")
        res = solve_fem(geo, lam_m, eps_cf, opt, order=2, n_super=1.0 + 0j, n_sub=complex(n_sub, 0.0))
        ref = tmm.coh_tmm('p', [1.0, complex(N_SLAB), complex(n_sub)],
                           [np.inf, D_SLAB, np.inf], math.radians(th_deg), LAM)
        Tf = res.T if res.T is not None else float('nan')
        dR, dT = abs(res.R - ref['R']), abs(Tf - ref['T'])
        good = dR < TOL and dT < TOL
        ok = ok and good
        print("[t] {:20s} th={:2.0f}: R {:.4f}/{:.4f}  T {:.4f}/{:.4f}  R+T={:.4f}  {}".format(
            label, th_deg, res.R, ref['R'], Tf, ref['T'], res.R + (Tf if Tf == Tf else 0),
            "OK" if good else "MISMATCH(dR={:.3f},dT={:.3f})".format(dR, dT)), flush=True)
    return ok


def main():
    print("[t] p-pol (TM) oblique validation vs tmm   lam={}nm".format(LAM), flush=True)
    r = [run("air|slab2|air", 1.0), run("air|slab2|sub1.5", 1.5)]
    print("[t] *** P-POL OBLIQUE vs TMM (0-30deg): {} ***".format("PASS" if all(r) else "FAIL"),
           flush=True)


if __name__ == "__main__":
    main()
