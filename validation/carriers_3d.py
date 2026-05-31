"""Phase 5c: NATIVE 3D DEVSIM carriers. Builds a 3D MOS-cap (semiconductor + gate
oxide, gate contact on top, body contact on bottom) via gmsh, solves the EXISTING
dimension-agnostic equilibrium physics (single-variable Poisson + Aymerich-Humet
F_1/2) on the 3D mesh, and verifies correctness by SOLVER-INDEPENDENT physics:
  (1) x,y-invariance -- a translationally-invariant gate must give n=n(z) only
      (the 3D solve must recover the 1D symmetry);
  (2) sign + monotonicity -- +Vg accumulates, -Vg depletes;
  (3) Gauss's law -- accumulated sheet charge q*Int(n-n_bg)dz == oxide displacement
      eps_ox*eps0*(Vg - V_surf)/t_ox (independent of the carrier solver).

NOTE: gmsh's OCC kernel cannot build at 1e-9-metre absolute scale, so the geometry
is built in NM and the mesh emitted SCALED to metres (Mesh.ScalingFactor) for
DEVSIM's SI physics; DEVSIM reads MSH 2.2.

Run:  python -m validation.carriers_3d
"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import gmsh
import devsim as ds
from dynameta.carriers import physics_equilibrium as PE
from dynameta.carriers.physics_equilibrium import M_E, EPS0, Q_E

LX_NM = LY_NM = 12.0          # small lateral box (physics is x,y-invariant) keeps
                              # the interface-refined node count tractable
T_SEMI_NM, T_OX_NM = 12.0, 8.0
T_OX = T_OX_NM * 1e-9                  # metres (Gauss check)
N_BG, EPS_SEMI, EPS_OX, DOS = 4e26, 9.5, 18.0, 0.35 * M_E
MSH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_moscap3d.msh")

def build_mesh():
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("moscap3d")
    occ = gmsh.model.occ
    vb = occ.addBox(0, 0, 0, LX_NM, LY_NM, T_SEMI_NM)
    vt = occ.addBox(0, 0, T_SEMI_NM, LX_NM, LY_NM, T_OX_NM)
    occ.synchronize()
    occ.fragment([(3, vb)], [(3, vt)])
    occ.synchronize()
    for dim, tag in gmsh.model.getEntities(3):
        zc = occ.getCenterOfMass(dim, tag)[2]
        gmsh.model.addPhysicalGroup(3, [tag], name=("semi" if zc < T_SEMI_NM else "oxide"))
    gate, body, iface = [], [], []
    for dim, tag in gmsh.model.getEntities(2):
        zc = occ.getCenterOfMass(dim, tag)[2]
        if abs(zc - (T_SEMI_NM + T_OX_NM)) < 1e-4: gate.append(tag)
        elif abs(zc) < 1e-4: body.append(tag)
        elif abs(zc - T_SEMI_NM) < 1e-4: iface.append(tag)
    gmsh.model.addPhysicalGroup(2, gate, name="gate")
    gmsh.model.addPhysicalGroup(2, body, name="body")
    gmsh.model.addPhysicalGroup(2, iface, name="semi_oxide")
    # graded refinement: fine (~0.5nm) near the semi/oxide interface to resolve the
    # ~1nm accumulation layer, coarse (~3nm) away from it.
    fd = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(fd, "SurfacesList", iface)
    ft = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(ft, "InField", fd)
    gmsh.model.mesh.field.setNumber(ft, "SizeMin", 0.5)
    gmsh.model.mesh.field.setNumber(ft, "SizeMax", 3.0)
    gmsh.model.mesh.field.setNumber(ft, "DistMin", 1.0)
    gmsh.model.mesh.field.setNumber(ft, "DistMax", 6.0)
    gmsh.model.mesh.field.setAsBackgroundMesh(ft)
    for opt in ("Mesh.MeshSizeExtendFromBoundary", "Mesh.MeshSizeFromPoints",
                 "Mesh.MeshSizeFromCurvature"):
        gmsh.option.setNumber(opt, 0)
    gmsh.model.mesh.generate(3)
    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)     # DEVSIM reads MSH 2.2
    gmsh.option.setNumber("Mesh.ScalingFactor", 1e-9)     # nm geometry -> metre mesh
    gmsh.write(MSH)
    gmsh.finalize()

def build_device():
    ds.create_gmsh_mesh(mesh="m3", file=MSH)
    ds.add_gmsh_region(mesh="m3", gmsh_name="semi", region="semi", material="ITO")
    ds.add_gmsh_region(mesh="m3", gmsh_name="oxide", region="oxide", material="HfO2")
    ds.add_gmsh_contact(mesh="m3", gmsh_name="gate", region="oxide", name="gate", material="metal")
    ds.add_gmsh_contact(mesh="m3", gmsh_name="body", region="semi", name="body", material="metal")
    ds.add_gmsh_interface(mesh="m3", gmsh_name="semi_oxide", region0="semi", region1="oxide", name="si_ox")
    ds.finalize_mesh(mesh="m3")
    ds.create_device(mesh="m3", device="d3")
    PE.setup_semiconductor_region("d3", "semi", n_bg_m3=N_BG, eps_static=EPS_SEMI, dos_mass_kg=DOS)
    PE.setup_dielectric_region("d3", "oxide", EPS_OX)
    for itf in ds.get_interface_list(device="d3"):
        PE.setup_interface("d3", itf)
    for c in ds.get_contact_list(device="d3"):
        PE.setup_contact("d3", c)

def solve(Vg):
    ds.set_parameter(device="d3", name="gate_bias", value=Vg)
    ds.set_parameter(device="d3", name="body_bias", value=0.0)
    ds.solve(type="dc", solver_type="direct", absolute_error=1e10,
              relative_error=1e-5, maximum_iterations=80)
    g = lambda nm: np.array(ds.get_node_model_values(device="d3", region="semi", name=nm))
    return g("x"), g("y"), g("z"), g("Electrons"), g("Potential")

def xy_invariance(z, n):
    """Lateral (x,y) variation per z-bin. The within-bin LINEAR z-trend is removed
    first: near the interface n(z) is steep, so a finite z-bin otherwise misreads
    that z-gradient as lateral variation. The residual std/mean is the true x,y
    non-invariance (must be ~0 for a translationally-invariant gate)."""
    bins = np.round(z / 1e-9).astype(int)
    worst = 0.0
    for b in np.unique(bins):
        m = bins == b
        if m.sum() >= 6:
            zb, nb = z[m], n[m]
            A = np.column_stack([zb - zb.mean(), np.ones_like(zb)])
            coef, *_ = np.linalg.lstsq(A, nb, rcond=None)     # detrend z within bin
            resid = nb - A @ coef
            worst = max(worst, float(np.std(resid) / max(np.mean(nb), 1e-30)))
    return worst

def main():
    print("[t] building 3D gmsh MOS-cap mesh (nm geometry -> metre mesh)...", flush=True)
    build_mesh()
    build_device()
    nn = len(ds.get_node_model_values(device="d3", region="semi", name="x"))
    print("[t] 3D device built: semi region has {} nodes".format(nn), flush=True)

    x, y, z, n0, V0 = solve(0.0)
    print("[t] zero-bias: n/n_bg mean={:.4f}  xy-invariance(worst std/mean)={:.2e}".format(
        n0.mean() / N_BG, xy_invariance(z, n0)), flush=True)

    results = {}
    for Vg in (+1.0, -1.0):
        x, y, z, n, V = solve(Vg)
        top = n[z > z.max() - 1.0e-9]
        inv = xy_invariance(z, n)
        order = np.argsort(z)
        zs, ns, Vs = z[order], n[order], V[order]
        zu = np.unique(np.round(zs / 2e-10) * 2e-10)
        navg = np.array([ns[np.abs(zs - zz) < 1.5e-10].mean() for zz in zu])
        dn = navg - N_BG
        dQ = Q_E * float(np.sum(0.5 * (dn[1:] + dn[:-1]) * np.diff(zu)))   # C/m^2
        V_surf = Vs[np.abs(zs - zs.max()) < 3e-10].mean()
        D_ox = EPS_OX * EPS0 * (Vg - V_surf) / T_OX                        # C/m^2
        results[Vg] = (top.mean() / N_BG, inv, dQ, D_ox)
        print("[t] Vg={:+.1f}V  n_top/n_bg={:.4f}  xy-inv={:.2e}  "
              "dQ={:+.3e}  D_ox={:+.3e}  ratio={:.3f}".format(
              Vg, top.mean() / N_BG, inv, dQ, D_ox, dQ / D_ox if D_ox else float('nan')),
              flush=True)

    acc = results[+1.0][0]; dep = results[-1.0][0]
    inv_ok = max(results[+1.0][1], results[-1.0][1]) < 0.05   # unstructured-tet noise
    sign_ok = dep < 1.0 < acc
    dQp, Dp = results[+1.0][2], results[+1.0][3]
    gauss_ok = abs(dQp / Dp - 1.0) < 0.15 if Dp else False
    print("[t] CHECKS: xy-invariant={}  sign-correct={}  Gauss(dQ~D_ox to 15%)={}".format(
        inv_ok, sign_ok, gauss_ok), flush=True)
    print("[t] *** 3D DEVSIM CARRIERS: {} ***".format(
        "PASS" if (inv_ok and sign_ok and gauss_ok) else "PARTIAL"), flush=True)

if __name__ == "__main__":
    main()
