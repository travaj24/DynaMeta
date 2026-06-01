"""Validate the new OCC inclusion shapes (Ellipse, RegularPolygon, Polygon) added to the
default builder (audit BI-4). A lossless subwavelength dielectric inclusion array must
conserve energy R+T ~ 1 at normal incidence (only the 0th order propagates:
lambda/P = 3.25 > n_hi). Builds an ELLIPSE and a HEXAGON inclusion, solves, and checks
energy conservation -- confirming the polygon-prism build meshes + solves correctly.
Run: python -m validation.inclusion_shapes
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dynameta.materials import Material, MaterialRegistry, ConstantOptical
from dynameta.geometry import UnitCell, Stack, Layer, Inclusion, Design
from dynameta.geometry.cross_section import Ellipse, RegularPolygon
from dynameta.geometry.specs import OpticalSpec, Mesh3DSpec
from dynameta.optics.ngsolve_layered import LayeredOpticalBuilder
from dynameta.optics.solver import solve_fem

LAM, P, THK, N_HI = 1300.0, 400.0, 200.0, 2.45     # eps_hi ~ 6; lambda/P = 3.25 > n_hi
TOL = 0.02


def _solve(shape, label):
    reg = MaterialRegistry()
    reg.add(Material("air", ConstantOptical(1.0 + 0j)))
    reg.add(Material("hi", ConstantOptical(complex(N_HI ** 2, 0.0))))
    cell = UnitCell.square(P * 1e-9)
    stack = Stack(layers=[Layer("L", THK * 1e-9, "air", inclusions=[Inclusion(shape, "hi")])],
                   superstrate_material="air", substrate_material="air")
    m3 = Mesh3DSpec(pml_thk_m=700e-9, superstrate_buffer_m=1400e-9, substrate_buffer_m=1400e-9,
                     maxh_superstrate_m=45e-9, maxh_substrate_m=45e-9,
                     maxh_background_m=24e-9, maxh_inclusion_m=24e-9)
    d = Design(name=label, unit_cell=cell, stack=stack, electrodes=[], materials=reg, mesh_3d=m3)
    geo = LayeredOpticalBuilder(d).build()
    n_incl = sum(1 for m in geo.mesh.GetMaterials() if "__incl" in m)
    lam_m = LAM * 1e-9
    eps_vals = {r: complex(d.materials.get(geo.material_by_region[r]).eps(lam_m))
                for r in geo.mesh.GetMaterials()}
    eps_cf = geo.mesh.MaterialCF(eps_vals, default=1.0)
    opt = OpticalSpec(polarization="y", incidence_angle_deg=0.0, linear_solver="umfpack")
    res = solve_fem(geo, lam_m, eps_cf, opt, order=2, n_super=1.0 + 0j, n_sub=1.0 + 0j)
    T = res.T if res.T is not None else float("nan")
    e = abs(res.R + T - 1.0)
    print("[t] {:8s}: incl-subsolids={} R={:.4f} T={:.4f} R+T={:.4f} |R+T-1|={:.4f}".format(
        label, n_incl, res.R, T, res.R + T, e), flush=True)
    return e


def main():
    print("[t] NEW INCLUSION SHAPES (ellipse, hexagon): lam={:.0f} P={:.0f} thk={:.0f} n_hi={:.2f}".format(
        LAM, P, THK, N_HI), flush=True)
    e1 = _solve(Ellipse(cx_m=P / 2 * 1e-9, cy_m=P / 2 * 1e-9, rx_m=150e-9, ry_m=90e-9), "ellipse")
    e2 = _solve(RegularPolygon(cx_m=P / 2 * 1e-9, cy_m=P / 2 * 1e-9, radius_m=140e-9, n_sides=6), "hexagon")
    ok = e1 < TOL and e2 < TOL
    print("[t] *** INCLUSION SHAPES (ellipse + hexagon build/solve + energy): {} ***".format(
        "PASS" if ok else "FAIL"), flush=True)
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
