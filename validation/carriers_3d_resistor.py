"""ADVERSARIAL: the 3D drift-diffusion "reduces to equilibrium" check is a WEAK transport
test -- a MOS-cap carries no DC current, so DD trivially equals equilibrium. This is the
missing independent test of 3D DRIFT transport: a uniform semiconductor BAR with ohmic
contacts on its two x-end faces, a small applied voltage, and a check that the DD contact
current obeys OHM'S LAW (I = V*sigma*A/L, sigma = q*n*mu).

D2c (v0.14.0): meshed with devsim_3d's tensor-product Kuhn mesh (the builder default), on which DEVSIM's
3-D couplings are exact. The old unstructured gmsh bar read Ohm's law 14 % HIGH (DEVSIM sums the
magnitudes of each tet's circumcentric dual pieces, inflating the couplings -- the same 1.2-1.3 x the
node volumes show; docs/audit/2026-11-03-devsim-3d-mesh.md), and the gate sat at < 0.16 to absorb it.
Run: python -m validation.carriers_3d_resistor [--gmsh  (the legacy mesh, for the comparison)]
"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dynameta.carriers.devsim_env import import_devsim   # D0: the ONE devsim import home (BLAS pin)

ds = import_devsim()
from dynameta.carriers import physics_drift_diffusion as DD
from dynameta.carriers.dc_solve import solve_dc
from dynameta.carriers.physics_equilibrium import M_E

LX, LY, LZ = 200.0, 80.0, 80.0     # nm
N_BG, MU, EPS = 4e26, 0.004, 9.5   # ITO-like: m^-3, m^2/Vs, eps_r
Q = 1.602176634e-19
V_APPLIED = 0.02                    # small -> ohmic


def build_bar_tensor(mesh="bar", dev="bar_dev", h_nm=8.0):
    """The bar on the Kuhn-tensor mesh (devsim_3d's shared helpers), contacts on the two x-end faces."""
    from dynameta.carriers import devsim_3d as D3
    path = os.path.join(os.path.expanduser("~"), ".dynameta", "_resistor_tensor.msh")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ax = tuple(D3._graded_axis([0.0, ln], [], h_nm, h_nm) for ln in (LX, LY, LZ))
    P, T, _cell = D3._kuhn_tets(*ax)
    rid = np.zeros(tuple(len(a) - 1 for a in ax), dtype=int)
    tri = {"left": D3._face_triangles(ax, 0, D3._cell_faces(rid, 0, "bottom", 0), plus=False),
           "right": D3._face_triangles(ax, 0, D3._cell_faces(rid, 0, "top", 0), plus=True)}
    D3._write_msh22(path, P * 1e-9, {"semi": T}, tri)
    ds.create_gmsh_mesh(mesh=mesh, file=path)
    ds.add_gmsh_region(mesh=mesh, gmsh_name="semi", region="semi", material="ITO")
    ds.add_gmsh_contact(mesh=mesh, gmsh_name="left", region="semi", name="left", material="metal")
    ds.add_gmsh_contact(mesh=mesh, gmsh_name="right", region="semi", name="right", material="metal")
    ds.finalize_mesh(mesh=mesh); ds.create_device(mesh=mesh, device=dev)
    return dev


def build_bar(mesh="bar", dev="bar_dev"):
    """The LEGACY unstructured gmsh bar (--gmsh): Ohm's law 14 % high (module docstring)."""
    import gmsh
    path = os.path.join(os.path.expanduser("~"), ".dynameta", "_resistor.msh")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    gmsh.initialize(); gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("bar"); occ = gmsh.model.occ
    occ.addBox(0, 0, 0, LX, LY, LZ); occ.synchronize()
    for dim, tag in gmsh.model.getEntities(3):
        gmsh.model.addPhysicalGroup(3, [tag], name="semi")
    left, right = [], []
    for dim, tag in gmsh.model.getEntities(2):
        xc = occ.getCenterOfMass(dim, tag)[0]
        if abs(xc) < 1e-4: left.append(tag)
        elif abs(xc - LX) < 1e-4: right.append(tag)
    gmsh.model.addPhysicalGroup(2, left, name="left")
    gmsh.model.addPhysicalGroup(2, right, name="right")
    gmsh.option.setNumber("Mesh.MeshSizeMin", 5.0); gmsh.option.setNumber("Mesh.MeshSizeMax", 8.0)
    gmsh.model.mesh.generate(3)
    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2); gmsh.option.setNumber("Mesh.ScalingFactor", 1e-9)
    gmsh.write(path); gmsh.finalize()
    ds.create_gmsh_mesh(mesh=mesh, file=path)
    ds.add_gmsh_region(mesh=mesh, gmsh_name="semi", region="semi", material="ITO")
    ds.add_gmsh_contact(mesh=mesh, gmsh_name="left", region="semi", name="left", material="metal")
    ds.add_gmsh_contact(mesh=mesh, gmsh_name="right", region="semi", name="right", material="metal")
    ds.finalize_mesh(mesh=mesh); ds.create_device(mesh=mesh, device=dev)
    return dev


def main():
    legacy = "--gmsh" in sys.argv
    dev = build_bar() if legacy else build_bar_tensor()
    DD.setup_semiconductor_region_dd(dev, "semi", n_bg_m3=N_BG, eps_static=EPS,
                                      dos_mass_kg=0.35 * M_E, mobility_m2Vs=MU)
    for c in ("left", "right"):
        DD.setup_contact_ohmic_dd(dev, c)
    abs_tol = max(1e10, N_BG * 1e-12)
    ds.set_parameter(device=dev, name="left_bias", value=0.0)
    ds.set_parameter(device=dev, name="right_bias", value=0.0)
    solve_dc(dev, method="newton", abs_tol=abs_tol, rel_tol=1e-10, max_iter=100,
             semiconductor_regions=["semi"])
    n_steps = 4
    for k in range(1, n_steps + 1):
        ds.set_parameter(device=dev, name="right_bias", value=V_APPLIED * k / n_steps)
        solve_dc(dev, method="newton", abs_tol=abs_tol, rel_tol=1e-10, max_iter=100,
                 semiconductor_regions=["semi"])
    # DEVSIM electron current through the contact (A); total = electron (unipolar)
    I_dev = abs(ds.get_contact_current(device=dev, contact="right",
                                        equation="ElectronContinuityEquation"))
    sigma = Q * N_BG * MU                                # S/m
    A = (LY * 1e-9) * (LZ * 1e-9); L = LX * 1e-9
    I_ohm = V_APPLIED * sigma * A / L
    rel = abs(I_dev - I_ohm) / I_ohm
    print("[t] 3D resistor (ITO bar {:.0f}x{:.0f}x{:.0f} nm, V={:.3f} V):".format(LX, LY, LZ, V_APPLIED), flush=True)
    print("[t]   sigma={:.3e} S/m  R={:.2f} ohm".format(sigma, L / (sigma * A)), flush=True)
    print("[t]   I_devsim={:.4e} A   I_ohm={:.4e} A   rel-diff={:.2e}".format(I_dev, I_ohm, rel), flush=True)
    # 3D DD carries current per Ohm's law (I ~ V*sigma*A/L). The FD g-factor cancels in the pure-drift
    # limit, so Boltzmann sigma=q*n*mu is the correct reference. On the Kuhn-tensor mesh the box-method
    # couplings are exact and a uniform bar's SG solution is the exact linear potential, so the residual
    # is solver tolerance only (D2c gate < 1e-6). The legacy gmsh bar (--gmsh) read 0.142 at its 5-8 nm
    # mesh (0.20 at 12-18 nm) -- the inflated-coupling defect, not discretization (it was gated < 0.16).
    ok = (0.0 < rel < 0.16) if legacy else (rel < 1e-6)
    print("[t] *** 3D DD TRANSPORT (Ohm's law, {}): {} ***".format(
        "legacy gmsh mesh" if legacy else "exact Kuhn-tensor mesh", "PASS" if ok else "FAIL"), flush=True)
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
