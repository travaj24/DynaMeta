"""ADVERSARIAL: oblique was validated only on laterally-UNIFORM layered slabs (vs tmm).
A laterally-STRUCTURED cell (a patch) is the non-separable regime the per-idnr Bloch
detection + demodulated extraction must also handle. tmm doesn't apply to a patch, so
the gate is ENERGY CONSERVATION: a LOSSLESS dielectric patch (no metal) must give
R+T=1 (A=0) at every angle -- exactly what the original (broken) oblique violated. Test
a centered n=3 dielectric pillar in an n=1.5 layer, sub-wavelength cell, s-pol, 0/15/30.
Run:  python -m validation.adversarial_oblique_patch
"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dynameta.materials import Material, MaterialRegistry, ConstantOptical
from dynameta.geometry import UnitCell, Stack, Layer, Inclusion, Design, centered_square
from dynameta.geometry.specs import OpticalSpec, Mesh3DSpec
from dynameta.optics.ngsolve_layered import LayeredOpticalBuilder
from dynameta.optics.solver import solve_fem

LAM = 1300.0
ANGLES = (0.0, 15.0, 30.0)
TOL = 0.03


def main():
    reg = MaterialRegistry()
    reg.add(Material("air", ConstantOptical(1.0 + 0j)))
    reg.add(Material("lowk", ConstantOptical(complex(1.5 ** 2, 0.0))))    # lossless host
    reg.add(Material("highk", ConstantOptical(complex(3.0 ** 2, 0.0))))   # lossless patch
    cell = UnitCell.square(220e-9)                                         # sub-wavelength -> 0-order
    layers = [Layer("pillar", 200e-9, "lowk",
                     inclusions=[Inclusion(centered_square(cell, 110e-9), "highk")])]
    stack = Stack(layers=layers, superstrate_material="air", substrate_material="air")
    m3 = Mesh3DSpec(pml_thk_m=700e-9, superstrate_buffer_m=1500e-9, substrate_buffer_m=1500e-9,
                     maxh_superstrate_m=40e-9, maxh_substrate_m=40e-9, maxh_background_m=25e-9,
                     maxh_inclusion_m=20e-9)
    d = Design(name="oblique_patch", unit_cell=cell, stack=stack, electrodes=[], materials=reg,
                mesh_3d=m3, optical=OpticalSpec(polarization="y", lift="extrude"))
    geo = LayeredOpticalBuilder(d).build()
    lam_m = LAM * 1e-9
    eps_vals = {r: complex(d.materials.get(geo.material_by_region[r]).eps(lam_m))
                for r in geo.mesh.GetMaterials()}
    eps_cf = geo.mesh.MaterialCF(eps_vals, default=1.0)
    print("[t] LOSSLESS dielectric-patch cell, s-pol; energy conservation (R+T=1) at oblique", flush=True)
    print("[t] mesh regions: {}".format(geo.mesh.GetMaterials()), flush=True)
    ok = True
    for th in ANGLES:
        opt = OpticalSpec(polarization="y", incidence_angle_deg=th, linear_solver="umfpack")
        res = solve_fem(geo, lam_m, eps_cf, opt, order=2, n_super=1.0 + 0j, n_sub=1.0 + 0j)
        Tf = res.T if res.T is not None else float("nan")
        RpT = res.R + (Tf if Tf == Tf else 0.0)
        good = abs(RpT - 1.0) < TOL
        ok = ok and good
        print("[t] th={:2.0f}: R={:.4f} T={:.4f} R+T={:.4f} (A={:+.4f})  {}".format(
            th, res.R, Tf, RpT, res.A if res.A is not None else float('nan'),
            "OK" if good else "ENERGY NOT CONSERVED (|R+T-1|={:.3f})".format(abs(RpT - 1.0))), flush=True)
    print("[t] *** ADVERSARIAL OBLIQUE PATCH (energy conservation, non-separable): {} ***".format(
        "PASS" if ok else "FAIL"), flush=True)
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
