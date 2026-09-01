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
    # WHY NOT SHRINK THE MESH LIKE THE SIBLING ORACLES (as of 2026-08-31 this reasoning is
    # SUPERSEDED -- kept because it records what was true, and why the conclusion changed).
    # It did not work: at (maxh_bg, maxh_incl, maxh_buf) = (45,40,80)/(50,50,85)/(60,60,100) nm
    # the umfpack peak was still 40.2/38.2/25.6 GB. The floor was UPSTREAM of this file -- the
    # builder approximated an Ellipse by an INSCRIBED 72-GON, so the perimeter carried ~11 nm
    # facets no matter what maxh_inclusion_m said: at maxh 45 nm the ellipse cell meshed to ~22k
    # elements where the same-size circle or hexagon needed ~7k, and that fine embedded feature
    # gave the LU pathological fill (~174 kB/DOF vs ~60 for the sibling oracles).
    # THE 72-GON IS GONE: ngsolve_layered now builds a TRUE OCC ellipse, so the perimeter is an
    # analytic curve the mesher discretizes at maxh like any other. Re-measured at the very same
    # (45,40,80) nm, the ellipse cell is no longer an outlier at all -- 8231 elements against
    # 7910 for the hexagon and 7754 for the circle, a 4% spread where it used to be ~3x. At the
    # SHIPPED (24,24,45) nm all four shapes sit within 1.4% of each other (42.2k/42.6k/42.0k).
    # The cost driver behind the over-budget kill is therefore fixed at the source rather than
    # worked around. bddc_gmres is KEPT regardless: it is the library default, it was validated
    # against the direct solve below, and it is far cheaper than a direct factorization at this
    # size. Whether umfpack now fits the 12.4 GB budget again is UNMEASURED -- deliberately, as
    # the previous attempt reached 45 GB on this box and memory pressure here has historically
    # produced silent process deaths rather than clean failures.
    # THE SWAP IS VALIDATED AGAINST THE DIRECT SOLVE ON THE SAME MESH: at (45,40,80) umfpack and
    # bddc_gmres both return ellipse R = 0.0272 / T = 0.9726 / |R+T-1| = 0.0002 and hexagon
    # R = 0.0621 / T = 0.9377 / |R+T-1| = 0.0003 -- identical to four decimals -- and on the
    # SHIPPED (unchanged) mesh bddc_gmres returns ellipse 0.0272/0.9726/0.0002 and hexagon
    # 0.0619/0.9379/0.0002, agreeing with the direct solve on three independent coarser meshes.
    # Those ellipse figures are the 72-GON's. With the true ellipse the same run gives
    # 0.0270/0.9728/0.0002 -- the hexagon is bit-for-bit unmoved (0.0619/0.9379/0.0002), which is
    # the control: only the shape that changed, changed. A ~0.7% shift in R is the expected size
    # for an inclusion whose area grew by the 0.127% the inscribed polygon was missing, and the
    # gate here is energy conservation (TOL=0.02), which neither geometry threatens.
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
