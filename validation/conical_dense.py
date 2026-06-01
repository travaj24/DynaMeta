"""Validate CONICAL incidence (azimuth phi != 0) on a DENSE (non-vacuum) SUBSTRATE
(audit conical-F4): previously conical was only checked with an air exit. The exit
wavevector kz_sub = sqrt((n_sub k0)^2 - kx^2 - ky^2) must include the ky from the azimuth.
An isotropic layered stack is azimuthally symmetric, so at fixed theta the R/T are
phi-INVARIANT and equal tmm(theta,'s'); here the substrate is n_sub=1.5.
Run: python -m validation.conical_dense
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

LAM, N_SLAB, N_SUB, D, THETA = 1300.0, 2.0, 1.5, 250.0, 30.0
PHIS = (0.0, 45.0)
TOL = 0.03


def main():
    reg = MaterialRegistry()
    reg.add(Material("air", ConstantOptical(1.0 + 0j)))
    reg.add(Material("slab", ConstantOptical(complex(N_SLAB ** 2, 0.0))))
    reg.add(Material("sub", ConstantOptical(complex(N_SUB ** 2, 0.0))))
    cell = UnitCell.square(220e-9)
    stack = Stack(layers=[Layer("slab", D * 1e-9, "slab")], superstrate_material="air",
                   substrate_material="sub")
    m3 = Mesh3DSpec(pml_thk_m=700e-9, superstrate_buffer_m=1500e-9, substrate_buffer_m=1500e-9,
                     maxh_superstrate_m=40e-9, maxh_substrate_m=40e-9, maxh_background_m=20e-9)
    d = Design(name="cdense", unit_cell=cell, stack=stack, electrodes=[], materials=reg, mesh_3d=m3)
    geo = LayeredOpticalBuilder(d).build()
    lam_m = LAM * 1e-9
    eps_vals = {r: complex(d.materials.get(geo.material_by_region[r]).eps(lam_m))
                for r in geo.mesh.GetMaterials()}
    eps_cf = geo.mesh.MaterialCF(eps_vals, default=1.0)
    ref = tmm.coh_tmm("s", [1.0, complex(N_SLAB), complex(N_SUB)], [np.inf, D, np.inf],
                       math.radians(THETA), LAM)
    Rt, Tt = ref["R"], ref["T"]
    print("[t] CONICAL dense substrate (n_sub={:.1f}), theta={:.0f}, vs tmm R={:.4f} T={:.4f}".format(
        N_SUB, THETA, Rt, Tt), flush=True)
    ok, Rs = True, []
    for phi in PHIS:
        opt = OpticalSpec(polarization="y", incidence_angle_deg=THETA, azimuth_deg=phi,
                           linear_solver="umfpack")
        res = solve_fem(geo, lam_m, eps_cf, opt, order=2, n_super=1.0 + 0j, n_sub=complex(N_SUB))
        Tf = res.T if res.T is not None else float("nan")
        dR, dT = abs(res.R - Rt), abs(Tf - Tt)
        good = dR < TOL and dT < TOL
        ok = ok and good
        Rs.append(res.R)
        print("[t]   phi={:2.0f}d: R {:.4f}/{:.4f}  T {:.4f}/{:.4f}  {}".format(
            phi, res.R, Rt, Tf, Tt, "OK" if good else "MISMATCH(dR={:.3f},dT={:.3f})".format(dR, dT)),
            flush=True)
    spread = max(Rs) - min(Rs)
    ok = ok and spread < 5e-3
    print("[t]   phi-spread={:.4f}".format(spread), flush=True)
    print("[t] *** CONICAL DENSE-SUBSTRATE (phi-invariance + tmm): {} ***".format("PASS" if ok else "FAIL"),
          flush=True)
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
