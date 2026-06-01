"""ADVERSARIAL: I claimed oblique is 'tmm-validated' -- but only on LOSSLESS dielectric
slabs. A lossy medium (complex n, nonzero absorption A=1-R-T) is the untested regime, and
absorption is where a wrong field amplitude would show up. Test a LOSSY slab (n=2+0.1i)
vs tmm for BOTH s- and p-pol at 0/30deg, checking R, T, AND A. Run:
python -m validation.adversarial_oblique_lossy
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

LAM, N_SLAB, D = 1300.0, complex(2.0, 0.1), 250.0     # LOSSY slab
TOL = 0.03


def main():
    reg = MaterialRegistry()
    reg.add(Material("air", ConstantOptical(1.0 + 0j)))
    reg.add(Material("slab", ConstantOptical(N_SLAB ** 2)))    # complex eps (lossy)
    cell = UnitCell.square(220e-9)
    stack = Stack(layers=[Layer("slab", D * 1e-9, "slab")], superstrate_material="air",
                   substrate_material="air")
    m3 = Mesh3DSpec(pml_thk_m=700e-9, superstrate_buffer_m=1500e-9, substrate_buffer_m=1500e-9,
                     maxh_superstrate_m=40e-9, maxh_substrate_m=40e-9, maxh_background_m=20e-9)
    d = Design(name="lossy", unit_cell=cell, stack=stack, electrodes=[], materials=reg, mesh_3d=m3)
    geo = LayeredOpticalBuilder(d).build()
    lam_m = LAM * 1e-9
    eps_vals = {r: complex(d.materials.get(geo.material_by_region[r]).eps(lam_m))
                for r in geo.mesh.GetMaterials()}
    eps_cf = geo.mesh.MaterialCF(eps_vals, default=1.0)
    print("[t] LOSSY oblique vs tmm: slab n={}+{}i, d={}nm".format(N_SLAB.real, N_SLAB.imag, D), flush=True)
    ok = True
    for pol, tpol in [("y", "s"), ("p", "p")]:
        for th in (0.0, 30.0):
            opt = OpticalSpec(polarization=pol, incidence_angle_deg=th, linear_solver="umfpack")
            res = solve_fem(geo, lam_m, eps_cf, opt, order=2, n_super=1.0 + 0j, n_sub=1.0 + 0j)
            ref = tmm.coh_tmm(tpol, [1.0, N_SLAB, 1.0], [np.inf, D, np.inf], math.radians(th), LAM)
            Rt, Tt = ref["R"], ref["T"]; At = 1.0 - Rt - Tt
            Tf = res.T if res.T is not None else float("nan")
            Af = res.A if res.A is not None else float("nan")
            dR, dT, dA = abs(res.R - Rt), abs(Tf - Tt), abs(Af - At)
            good = dR < TOL and dT < TOL and dA < TOL
            ok = ok and good
            print("[t] {}-pol th={:2.0f}: R {:.4f}/{:.4f} T {:.4f}/{:.4f} A {:.4f}/{:.4f}  {}".format(
                tpol, th, res.R, Rt, Tf, Tt, Af, At,
                "OK" if good else "MISMATCH(dR={:.3f},dT={:.3f},dA={:.3f})".format(dR, dT, dA)), flush=True)
    print("[t] *** ADVERSARIAL LOSSY OBLIQUE vs tmm (s+p, R/T/A): {} ***".format("PASS" if ok else "FAIL"),
           flush=True)
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
