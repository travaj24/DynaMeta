"""Both oblique formulations under-capture R (~0.05 vs tmm 0.173 at 30deg). Dump the
actual solved scattered field to localize the bug: is the reflected field weak
(solve), x-structured (BC/diffraction), or mis-fit (extraction)? Solve the slab at
30deg (phase_in_space, physical field) and print the demodulated cell-averaged
scattered E_y(z) in the superstrate + the up/down fit coeffs + an x-invariance check.
Run:  python -m validation.oblique_field_dump
"""
import sys, os, math
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tmm
import ngsolve as ng
from dynameta.materials import Material, MaterialRegistry, ConstantOptical
from dynameta.geometry import UnitCell, Stack, Layer, Design
from dynameta.geometry.specs import OpticalSpec, Mesh3DSpec
from dynameta.optics.ngsolve_layered import LayeredOpticalBuilder, S
from dynameta.optics import solver as SM

LAM_NM, N_SLAB, N_SUB, D_SLAB_NM, THETA = 1300.0, 2.0, 1.5, 250.0, 30.0

def build():
    reg = MaterialRegistry()
    reg.add(Material("air",  ConstantOptical(1.0 + 0j)))
    reg.add(Material("slab", ConstantOptical(complex(N_SLAB**2, 0.0))))
    reg.add(Material("sub",  ConstantOptical(complex(N_SUB**2, 0.0))))
    cell = UnitCell.square(220e-9)
    stack = Stack(layers=[Layer("slab", D_SLAB_NM*1e-9, "slab")],
                   superstrate_material="air", substrate_material="sub")
    m3 = Mesh3DSpec(pml_thk_m=700e-9, superstrate_buffer_m=1400e-9,
                     substrate_buffer_m=1400e-9, maxh_superstrate_m=45e-9,
                     maxh_substrate_m=45e-9, maxh_background_m=20e-9, fem_order=2)
    return Design(name="dump", unit_cell=cell, stack=stack, electrodes=[],
                    materials=reg, mesh_3d=m3)

def main():
    d = build(); geo = LayeredOpticalBuilder(d).build(); lam_m = LAM_NM*1e-9
    eps_vals = {r: complex(d.materials.get(geo.material_by_region[r]).eps(lam_m))
                for r in geo.mesh.GetMaterials()}
    eps_cf = geo.mesh.MaterialCF(eps_vals, default=1.0)
    k0 = 2.0*math.pi/(lam_m*S); th = math.radians(THETA)
    kx = k0*math.sin(th); kz_s = k0*math.cos(th)
    th_r = tmm.coh_tmm('s', [1.0, complex(N_SLAB), complex(N_SUB)],
                        [np.inf, D_SLAB_NM, np.inf], th, LAM_NM)
    print("[t] tmm: r={:.4f}{:+.4f}j |r|^2={:.4f}  T={:.4f}".format(
        th_r['r'].real, th_r['r'].imag, abs(th_r['r'])**2, th_r['T']), flush=True)

    # --- inline phase_in_space solve (mirror solver.solve_fem) ---
    mesh = geo.mesh
    mesh.SetPML(ng.pml.HalfSpace(point=(0,0,geo.z_super_interface_nm), normal=(0,0,1), alpha=1j), "pml_top")
    mesh.SetPML(ng.pml.HalfSpace(point=(0,0,geo.z_sub_interface_nm), normal=(0,0,-1), alpha=1j), "pml_bot")
    E_inc = ng.CoefficientFunction((0.0, ng.exp(1j*kx*ng.x - 1j*kz_s*ng.z), 0.0))
    Px = geo.period_x_nm
    phases = SM._bloch_phase_list(geo, kx)   # per-idnr-detected Bloch phase
    print("[t] detected dirs = {}".format(''.join(getattr(geo,'_bloch_dirs',['?']))), flush=True)
    fes = ng.Periodic(ng.HCurl(mesh, order=2, complex=True, dirichlet=""), phase=phases)
    u, v = fes.TnT()
    a = ng.BilinearForm(fes, symmetric=True)
    a += (ng.curl(u)*ng.curl(v) - k0**2*eps_cf*(u*v))*ng.dx
    f = ng.LinearForm(fes); f += (k0**2*(eps_cf-1.0)*(E_inc*v))*ng.dx
    gfu = ng.GridFunction(fes)
    with ng.TaskManager():
        a.Assemble(); f.Assemble()
        gfu.vec.data = a.mat.Inverse(freedofs=fes.FreeDofs(), inverse="umfpack")*f.vec

    # superstrate probes
    z0 = geo.z_intervals_nm["superstrate"][0]; z1 = geo.z_intervals_nm["superstrate"][1]
    zlo, zhi = z0+50.0, z1-50.0
    zs = np.linspace(zlo, zhi, 7)
    xs = np.linspace(0.0, Px, 6, endpoint=False)
    print("[t] superstrate z in [{:.0f},{:.0f}] nm (struct_top={:.0f}, pml@{:.0f})".format(
        zlo, zhi, z0, geo.z_super_interface_nm), flush=True)
    Es = []
    for zv in zs:
        vals = [complex(gfu(mesh(float(xv), float(Px/2), float(zv)))[1])*np.exp(-1j*kx*xv) for xv in xs]
        Es.append(np.mean(vals))
    Es = np.array(Es)
    for zv, e in zip(zs, Es):
        print("[t]   z={:7.1f}  |Es|={:.4f}  arg={:+7.1f}deg".format(zv, abs(e), math.degrees(np.angle(e))), flush=True)
    M = np.column_stack([np.exp(+1j*kz_s*zs), np.exp(-1j*kz_s*zs)])
    c, *_ = np.linalg.lstsq(M, Es, rcond=None)
    print("[t] fit: r(up)={:.4f}{:+.4f}j |r|^2={:.4f} ; down(resid)={:.4f}{:+.4f}j |.|^2={:.4f}".format(
        c[0].real, c[0].imag, abs(c[0])**2, c[1].real, c[1].imag, abs(c[1])**2), flush=True)
    # x-invariance of the demodulated scattered field at mid-superstrate
    zmid = zs[len(zs)//2]
    print("[t] x-invariance (demod E_y) at z={:.0f}:".format(zmid), flush=True)
    for xv in xs:
        e = complex(gfu(mesh(float(xv), float(Px/2), float(zmid)))[1])*np.exp(-1j*kx*xv)
        print("[t]   x={:6.1f}  E_y(demod)={:.4f}{:+.4f}j  |.|={:.4f}".format(xv, e.real, e.imag, abs(e)), flush=True)

if __name__ == "__main__":
    main()
