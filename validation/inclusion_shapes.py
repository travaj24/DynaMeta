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
    # CI FIX (2026-08-31): LINEAR SOLVER ONLY -- umfpack -> bddc_gmres (the library DEFAULT).
    # NOTHING ELSE CHANGES: same mesh, same order, same geometry, same eps, same gate, same TOL.
    # Nightly run 33312685617 (2026-08-30) reported this oracle OVER-BUDGET on the 16 GB runner, so
    # it has never executed on CI. The 12.5 GB in that log is the WATCHDOG KILL POINT (12.4 GB budget
    # + one poll), NOT the peak: with the direct UMFPACK factorization this fixture passed 45 GB on
    # the dev box WITHOUT FINISHING ITS FIRST SOLVE. With bddc_gmres the SAME fixture peaks at
    # 3.2 GB in 142-171 s.
    # WHY NOT SHRINK THE MESH LIKE THE SIBLING ORACLES. It does not work here, and that is MEASURED:
    # at (maxh_bg, maxh_incl, maxh_buf) = (45,40,80)/(50,50,85)/(60,60,100) nm the umfpack peak is
    # still 40.2/38.2/25.6 GB. The floor is upstream of this file -- the builder approximates an
    # Ellipse by an INSCRIBED 72-GON (dynameta/optics/ngsolve_layered.py, `n = 72`), so the
    # perimeter carries ~11 nm facets no matter what maxh_inclusion_m says: at maxh 45 nm the
    # ellipse cell meshes to ~22k elements where the same-size circle or hexagon needs ~7k, and the
    # resulting fine embedded feature also gives the LU pathological fill (~174 kB/DOF vs ~60 for
    # the sibling oracles). Coarsening the mesh here would buy a factor of ~1.6 and cost accuracy;
    # changing the solver buys a factor of >14 and costs nothing.
    # THE SWAP IS VALIDATED AGAINST THE DIRECT SOLVE ON THE SAME MESH: at (45,40,80) umfpack and
    # bddc_gmres both return ellipse R = 0.0272 / T = 0.9726 / |R+T-1| = 0.0002 and hexagon
    # R = 0.0621 / T = 0.9377 / |R+T-1| = 0.0003 -- identical to four decimals -- and on the
    # SHIPPED (unchanged) mesh bddc_gmres returns ellipse 0.0272/0.9726/0.0002 and hexagon
    # 0.0619/0.9379/0.0002, agreeing with the direct solve on three independent coarser meshes.
    # solve_fem raises its own warning if GMRES fails to reach 1e-3 relative residual and declares
    # R/T/A unreliable, so a non-converged solve is LOUD, not a silent pass; none fired here.
    opt = OpticalSpec(polarization="y", incidence_angle_deg=0.0, linear_solver="bddc_gmres")
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
