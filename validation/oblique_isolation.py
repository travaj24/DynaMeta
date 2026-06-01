"""Validate the LAYERED (Fresnel two-region) background field across vacuum AND dense
substrates, at normal and oblique incidence, vs tmm. This exercises the fix for the
old uniform-background (eps_bg=1) failure on a non-vacuum substrate (which drove a
huge wrong-wavevector source through the substrate and was mesh-fragile even at normal).
 (a) air / 1.5 / sub=1.5  -> single air/1.5 Fresnel interface;
 (b) air / slab n=2 (250nm) / air  -> vacuum exit (reduction check, R0=0,T0=1);
 (c) air / slab n=2 (250nm) / sub=1.5  -> DENSE substrate (previously R=0.26 at normal
     vs tmm 0.12, R=0.08 at 30deg vs 0.17; now exact).
All cases must match tmm R and T to TOL at theta=0 and 30deg, energy-conserving.
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
TOL = 0.03

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
    ok = True
    for th_deg in angles:
        opt = OpticalSpec(polarization="y", incidence_angle_deg=th_deg, linear_solver="umfpack")
        res = solve_fem(geo, lam_m, eps_cf, opt, order=2, n_super=1.0 + 0j, n_sub=complex(n_sub, 0.0))
        ref = tmm.coh_tmm('s', ns, ds, math.radians(th_deg), LAM)
        Tf = res.T if res.T is not None else float('nan')
        dR = abs(res.R - ref['R']); dT = abs(Tf - ref['T'])
        good = dR < TOL and dT < TOL
        ok = ok and good
        print("[t] {:24s} th={:2.0f}: R {:.4f}/{:.4f}  T {:.4f}/{:.4f}  R+T={:.4f}  {}".format(
            label, th_deg, res.R, ref['R'], Tf, ref['T'], res.R + (Tf if Tf == Tf else 0),
            "OK" if good else "MISMATCH(dR={:.3f},dT={:.3f})".format(dR, dT)), flush=True)
    return ok

def main():
    r = []
    r.append(run("(a) air|1.5|sub1.5 (1 iface)", 1.5, 200.0, 1.5))   # slab==sub -> single air/1.5
    r.append(run("(b) air|slab2,250|air", 2.0, 250.0, 1.0))
    r.append(run("(c) air|slab2,250|sub1.5", 2.0, 250.0, 1.5))       # dense substrate (the fix)
    print("[t] *** OBLIQUE NON-VACUUM SUBSTRATE (layered-bg field) vs tmm: {} ***".format(
        "PASS" if all(r) else "FAIL"), flush=True)
    return all(r)

if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
