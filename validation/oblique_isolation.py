"""Isolate the residual oblique R discrepancy. The field is now correctly oblique
(kx + kz_s verified) but |r| is ~0.55x tmm for the air/n2/n1.5 slab. Test progressively:
 (a) single interface air / sub=1.5  (tmm trivial Fresnel)
 (b) air / slab n=2 (250nm) / sub=1.0 (slab in air, symmetric outer media)
 (c) air / slab n=2 (250nm) / sub=1.5 (original)
at theta=0 and 30deg vs tmm. If (a) matches at 30deg, single-interface oblique FEM is
correct and the gap is multilayer/Fabry-Perot; if (a) is off, the formulation is.
Run:  python -m validation.oblique_isolation
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

LAM = 1300.0

def run(label, n_slab, d_slab_nm, n_sub, angles=(0.0, 30.0)):
    reg = MaterialRegistry()
    reg.add(Material("air", ConstantOptical(1.0 + 0j)))
    reg.add(Material("sub", ConstantOptical(complex(n_sub**2, 0.0))))
    layers = []
    if n_slab is not None:
        reg.add(Material("slab", ConstantOptical(complex(n_slab**2, 0.0))))
        layers = [Layer("slab", d_slab_nm * 1e-9, "slab")]
    cell = UnitCell.square(220e-9)
    stack = Stack(layers=layers, superstrate_material="air", substrate_material="sub")
    m3 = Mesh3DSpec(pml_thk_m=700e-9, superstrate_buffer_m=1500e-9, substrate_buffer_m=1500e-9,
                     maxh_superstrate_m=40e-9, maxh_substrate_m=40e-9, maxh_background_m=20e-9)
    d = Design(name=label, unit_cell=cell, stack=stack, electrodes=[], materials=reg, mesh_3d=m3)
    geo = LayeredOpticalBuilder(d).build()
    lam_m = LAM * 1e-9
    eps_vals = {r: complex(d.materials.get(geo.material_by_region[r]).eps(lam_m))
                for r in geo.mesh.GetMaterials()}
    eps_cf = geo.mesh.MaterialCF(eps_vals, default=1.0)
    # tmm stack
    if n_slab is not None:
        ns = [1.0, complex(n_slab), complex(n_sub)]; ds = [np.inf, d_slab_nm, np.inf]
    else:
        ns = [1.0, complex(n_sub)]; ds = [np.inf, np.inf]
    for th_deg in angles:
        opt = OpticalSpec(polarization="y", incidence_angle_deg=th_deg, linear_solver="umfpack")
        res = solve_fem(geo, lam_m, eps_cf, opt, order=2, n_super=1.0 + 0j, n_sub=complex(n_sub, 0.0))
        ref = tmm.coh_tmm('s', ns, ds, math.radians(th_deg), LAM)
        Tf = res.T if res.T is not None else float('nan')
        print("[t] {:24s} th={:2.0f}: R {:.4f}/{:.4f}  T {:.4f}/{:.4f}  R+T={:.4f}".format(
            label, th_deg, res.R, ref['R'], Tf, ref['T'], res.R + (Tf if Tf == Tf else 0)), flush=True)

def main():
    run("(a) air|1.5|sub1.5 (1 iface)", 1.5, 200.0, 1.5)   # slab==sub -> single air/1.5
    run("(b) air|slab2,250|air", 2.0, 250.0, 1.0)
    run("(c) air|slab2,250|sub1.5", 2.0, 250.0, 1.5)

if __name__ == "__main__":
    main()
