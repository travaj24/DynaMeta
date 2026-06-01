# DIAGNOSTIC: a developer diagnostic with no PASS/FAIL gate (not run by validation.run_all).
"""Diagnostic for oblique-at-angle: at theta=30deg on the layered slab, solve THREE
ways -- plain periodic (no Bloch phase), phase=+kx, phase=-kx -- and compare R/T/
energy to tmm. Tells us (a) whether ng.Periodic(phase) is applied at all (plain vs
phased differ?), and (b) which sign matches tmm. Run: python -m validation.oblique_phase_diag
"""
import sys, os, math
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ngsolve as ng
import tmm
from dynameta.geometry.specs import OpticalSpec
from dynameta.optics.ngsolve_layered import LayeredOpticalBuilder, S
from dynameta.optics.solver import _reflection, _transmission
from validation.oblique_vs_tmm import build, LAM_NM, N_SLAB, N_SUB, D_SLAB_NM

THETA_DEG = 30.0
d = build()
geo = LayeredOpticalBuilder(d).build()
lam_m = LAM_NM * 1e-9
eps_cf = geo.mesh.MaterialCF(
    {r: complex(d.materials.get(geo.material_by_region[r]).eps(lam_m))
     for r in geo.mesh.GetMaterials()}, default=1.0)
opt = OpticalSpec(polarization="y", incidence_angle_deg=THETA_DEG, linear_solver="umfpack")

k0 = 2.0 * math.pi / (lam_m * S)
th = math.radians(THETA_DEG)
kx = k0 * math.sin(th)
kz_s = k0 * math.cos(th)
kz_sub = complex(np.sqrt(complex((N_SUB * k0) ** 2 - kx ** 2)))
mesh = geo.mesh
try:
    mesh.UnSetPML("pml_top"); mesh.UnSetPML("pml_bot")
except Exception:
    pass
a_pml = 1j / math.cos(th)
mesh.SetPML(ng.pml.HalfSpace(point=(0, 0, geo.z_super_interface_nm), normal=(0, 0, 1), alpha=a_pml), "pml_top")
mesh.SetPML(ng.pml.HalfSpace(point=(0, 0, geo.z_sub_interface_nm), normal=(0, 0, -1), alpha=a_pml), "pml_bot")
E_inc = ng.CoefficientFunction((0.0, ng.exp(1j * kx * ng.x - 1j * kz_s * ng.z), 0.0))
Px = geo.period_x_nm

def solve_with(phases):
    H = ng.HCurl(mesh, order=2, complex=True, dirichlet="")
    fes = ng.Periodic(H) if phases is None else ng.Periodic(H, phase=phases)
    u, v = fes.TrialFunction(), fes.TestFunction()
    a = ng.BilinearForm(fes, symmetric=True)
    a += (ng.curl(u) * ng.curl(v) - k0 ** 2 * eps_cf * (u * v)) * ng.dx
    f = ng.LinearForm(fes)
    f += (k0 ** 2 * (eps_cf - 1.0) * (E_inc * v)) * ng.dx
    gfu = ng.GridFunction(fes)
    with ng.TaskManager():
        a.Assemble(); f.Assemble()
        gfu.vec.data = a.mat.Inverse(freedofs=fes.FreeDofs(), inverse="umfpack") * f.vec
    r = _reflection(mesh, gfu, kx, kz_s, geo, opt)
    t = _transmission(mesh, gfu, kx, kz_s, kz_sub, geo, opt)
    R = float(abs(r) ** 2)
    T = float(abs(t) ** 2 * (kz_sub.real / (k0 * math.cos(th))))
    return R, T

ref = tmm.coh_tmm('s', [1.0, complex(N_SLAB), complex(N_SUB)],
                   [np.inf, D_SLAB_NM, np.inf], th, LAM_NM)
print("[t] theta=30deg  tmm: R={:.4f} T={:.4f}".format(ref['R'], ref['T']), flush=True)
variants = [
    ("plain (no phase)", None),
    ("phase +kx", [ng.exp(+1j * kx * Px)] * geo.n_px + [1.0 + 0j] * geo.n_py),
    ("phase -kx", [ng.exp(-1j * kx * Px)] * geo.n_px + [1.0 + 0j] * geo.n_py),
]
for label, phases in variants:
    R, T = solve_with(phases)
    print("[t] {:18s} R={:.4f} T={:.4f}  R+T={:.4f}".format(label, R, T, R + T), flush=True)
print("[t] (n_px={}, n_py={})".format(geo.n_px, geo.n_py), flush=True)
print("[t] *** OBLIQUE PHASE DIAGNOSTIC DONE ***", flush=True)
