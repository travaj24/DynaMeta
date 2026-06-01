"""Validate conical incidence on a LATERALLY-STRUCTURED cell -- the observable case the
layered-stack conical test cannot exercise (audit conical-F1). A layered slab is laterally
uniform, so it cannot 'see' the transverse ky Bloch phase; a square (c4v) dielectric patch
CAN.

At polar angle theta with azimuth phi, the in-plane wavevector is (kx,ky)=k0 sin(theta)
(cos phi, sin phi). For phi=0 the Bloch phase lives on the x-faces (ky=0); for phi=90 it
lives entirely on the y-faces (kx=0). A c4v square patch is invariant under the x<->y
(90 deg) rotation, so the specular R/T at phi=0 and phi=90 MUST be equal -- and a lossless
subwavelength patch MUST conserve energy R+T~1. If the ky Bloch phase were a no-op (the bug
class that hid the original oblique failure), phi=90 would return a different (effectively
normal-in-y) result and BREAK the symmetry. Lossless n=2 patch in air, theta=30 deg, s-pol.
Run: python -m validation.conical_structured
"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dynameta.materials import Material, MaterialRegistry, ConstantOptical
from dynameta.geometry import UnitCell, Stack, Layer, Inclusion, Design, centered_square
from dynameta.geometry.specs import OpticalSpec, Mesh3DSpec
from dynameta.optics.ngsolve_layered import LayeredOpticalBuilder
from dynameta.optics.solver import solve_fem

LAM, P, SIDE, THK, N_HI, THETA = 1300.0, 400.0, 240.0, 200.0, 2.0, 30.0
TOL = 0.02


def main():
    reg = MaterialRegistry()
    reg.add(Material("air", ConstantOptical(1.0 + 0j)))
    reg.add(Material("hi", ConstantOptical(complex(N_HI ** 2, 0.0))))
    cell = UnitCell.square(P * 1e-9)
    patch = centered_square(cell, SIDE * 1e-9)             # c4v square patch (x<->y symmetric)
    stack = Stack(layers=[Layer("patch", THK * 1e-9, "air", inclusions=[Inclusion(patch, "hi")])],
                   superstrate_material="air", substrate_material="air")
    m3 = Mesh3DSpec(pml_thk_m=700e-9, superstrate_buffer_m=1400e-9, substrate_buffer_m=1400e-9,
                     maxh_superstrate_m=45e-9, maxh_substrate_m=45e-9,
                     maxh_background_m=22e-9, maxh_inclusion_m=22e-9)
    d = Design(name="cpatch", unit_cell=cell, stack=stack, electrodes=[], materials=reg, mesh_3d=m3)
    geo = LayeredOpticalBuilder(d).build()
    lam_m = LAM * 1e-9
    eps_vals = {r: complex(d.materials.get(geo.material_by_region[r]).eps(lam_m))
                for r in geo.mesh.GetMaterials()}
    eps_cf = geo.mesh.MaterialCF(eps_vals, default=1.0)
    print("[t] CONICAL on a c4v square patch: lam={:.0f} P={:.0f} side={:.0f} thk={:.0f} n_hi={:.1f} "
          "theta={:.0f} (lambda/P={:.2f})".format(LAM, P, SIDE, THK, N_HI, THETA, LAM / P), flush=True)
    res = {}
    for phi in (0.0, 90.0):
        opt = OpticalSpec(polarization="y", incidence_angle_deg=THETA, azimuth_deg=phi,
                           linear_solver="umfpack")
        r = solve_fem(geo, lam_m, eps_cf, opt, order=3, n_super=1.0 + 0j, n_sub=1.0 + 0j)
        T = r.T if r.T is not None else float("nan")
        res[phi] = (r.R, T)
        print("[t]   phi={:2.0f}d: R={:.4f} T={:.4f} R+T={:.4f}".format(phi, r.R, T, r.R + T), flush=True)
    dR = abs(res[0.0][0] - res[90.0][0])
    dT = abs(res[0.0][1] - res[90.0][1])
    e0 = abs(res[0.0][0] + res[0.0][1] - 1.0)
    e90 = abs(res[90.0][0] + res[90.0][1] - 1.0)
    print("[t] c4v phi=0 vs phi=90: |dR|={:.4f} |dT|={:.4f} (TOL={:.3f})".format(dR, dT, TOL), flush=True)
    print("[t] energy: |R+T-1| phi0={:.4f} phi90={:.4f}".format(e0, e90), flush=True)
    ok = (dR < TOL and dT < TOL and e0 < TOL and e90 < TOL and 0.02 < res[0.0][0] < 0.98)
    print("[t] *** CONICAL STRUCTURED (c4v phi-symmetry + energy, observable ky-phase): {} ***".format(
        "PASS" if ok else "FAIL"), flush=True)
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
