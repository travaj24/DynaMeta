"""Phase 5b: EXTERNAL validation of oblique incidence vs the `tmm` library (Byrnes).
A purely layered stack (air / lossless slab n=2 / air, NO patch) has only the 0th
diffraction order for a sub-wavelength period, so TMM is exact. Compare dynameta's
FEM R/T (s-pol) to tmm.coh_tmm at several angles. theta=0 also confirms the solver
preserves normal incidence.

The exit medium here is VACUUM (air) -- the clean test of the oblique machinery
(Floquet-Bloch quasi-periodic phase + incidence + PML + demodulated R/T extraction),
all of which must be correct for energy to conserve at angle. Validates to <0.3% in
R and T through 30deg.

NON-vacuum (dense) substrates are now ALSO handled correctly by the layered (Fresnel
two-region) background field in optics/solver.py (eps_bg piecewise + analytic bare
air/substrate E_bg, so the substrate carries no spurious volumetric source) -- see
validation/oblique_isolation.py, which validates a dense (n=1.5) substrate to <0.3%
at 0 and 30deg (it was R=0.26 vs tmm 0.12 at normal under the old uniform background).

Run:  python -m validation.oblique_vs_tmm
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

LAM_NM = 1300.0
N_SLAB, N_SUB = 2.0, 1.0          # vacuum exit medium (see module docstring)
D_SLAB_NM = 250.0
ANGLES = (0.0, 15.0, 30.0)
TOL = 0.03

def build():
    reg = MaterialRegistry()
    reg.add(Material("air",  ConstantOptical(1.0 + 0j)))
    reg.add(Material("slab", ConstantOptical(complex(N_SLAB**2, 0.0))))
    reg.add(Material("sub",  ConstantOptical(complex(N_SUB**2, 0.0))))
    cell = UnitCell.square(220e-9)                 # sub-wavelength -> 0-order only
    stack = Stack(layers=[Layer("slab", D_SLAB_NM*1e-9, "slab")],
                   superstrate_material="air", substrate_material="sub")
    m3 = Mesh3DSpec(pml_thk_m=500e-9, superstrate_buffer_m=1400e-9,
                     substrate_buffer_m=1400e-9, maxh_superstrate_m=45e-9,
                     maxh_substrate_m=45e-9, maxh_background_m=20e-9, fem_order=2)
    return Design(name="slab_oblique", unit_cell=cell, stack=stack, electrodes=[],
                    materials=reg, mesh_3d=m3)

def main():
    d = build()
    geo = LayeredOpticalBuilder(d).build()
    lam_m = LAM_NM * 1e-9
    eps_vals = {r: complex(d.materials.get(geo.material_by_region[r]).eps(lam_m))
                for r in geo.mesh.GetMaterials()}
    eps_cf = geo.mesh.MaterialCF(eps_vals, default=1.0)
    print("[t] slab oblique validation: FEM (s-pol) vs tmm   lam={}nm".format(LAM_NM), flush=True)
    print("[t] {:>6s} | {:>16s} | {:>16s} | {:>8s}".format(
        "theta", "R fem / tmm", "T fem / tmm", "R+T fem"), flush=True)
    ok = True
    for theta_deg in ANGLES:
        opt = OpticalSpec(polarization="y", incidence_angle_deg=theta_deg,
                           linear_solver="umfpack")
        res = solve_fem(geo, lam_m, eps_cf, opt, order=2,
                         n_super=1.0+0j, n_sub=complex(N_SUB, 0.0))
        th = math.radians(theta_deg)
        ref = tmm.coh_tmm('s', [1.0, complex(N_SLAB), complex(N_SUB)],
                           [np.inf, D_SLAB_NM, np.inf], th, LAM_NM)
        Rt, Tt = ref['R'], ref['T']
        Rf = res.R
        Tf = res.T if res.T is not None else float('nan')
        dR, dT = abs(Rf - Rt), abs(Tf - Tt)
        good = dR < TOL and dT < TOL
        ok = ok and good
        print("[t] {:5.0f}d | {:6.4f} / {:6.4f} | {:6.4f} / {:6.4f} | {:7.4f}  {}".format(
            theta_deg, Rf, Rt, Tf, Tt, Rf + (Tf if Tf == Tf else 0.0),
            "OK" if good else "MISMATCH(dR={:.3f},dT={:.3f})".format(dR, dT)), flush=True)
    print("[t] *** OBLIQUE vs TMM (0-30deg, s-pol): {} ***".format("PASS" if ok else "FAIL"),
           flush=True)

if __name__ == "__main__":
    main()
