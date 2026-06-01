"""Validate CONICAL incidence (azimuth phi != 0), s-pol. An isotropic LAYERED stack is
azimuthally symmetric, so at a fixed polar angle theta the R/T must be INDEPENDENT of phi
and equal to tmm(theta, 's'). This exercises the new machinery: the 2D Bloch phase
(exp(i kx Px) on x-faces AND exp(i ky Py) on y-faces), the rotated s-pol direction
Es=(-sin phi, cos phi, 0), and the 2D-demodulated extraction. Test air/slab(n=2,250nm)/air
at theta=30deg over phi=0/30/60/90deg vs tmm. Run: python -m validation.oblique_conical_vs_tmm
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

LAM, N_SLAB, D, THETA = 1300.0, 2.0, 250.0, 30.0
PHIS = (0.0, 30.0, 60.0, 90.0)
TOL = 0.03


def main():
    reg = MaterialRegistry()
    reg.add(Material("air", ConstantOptical(1.0 + 0j)))
    reg.add(Material("slab", ConstantOptical(complex(N_SLAB ** 2, 0.0))))
    cell = UnitCell.square(220e-9)
    stack = Stack(layers=[Layer("slab", D * 1e-9, "slab")], superstrate_material="air",
                   substrate_material="air")
    m3 = Mesh3DSpec(pml_thk_m=700e-9, superstrate_buffer_m=1500e-9, substrate_buffer_m=1500e-9,
                     maxh_superstrate_m=40e-9, maxh_substrate_m=40e-9, maxh_background_m=20e-9)
    d = Design(name="conical", unit_cell=cell, stack=stack, electrodes=[], materials=reg, mesh_3d=m3)
    geo = LayeredOpticalBuilder(d).build()
    lam_m = LAM * 1e-9
    eps_vals = {r: complex(d.materials.get(geo.material_by_region[r]).eps(lam_m))
                for r in geo.mesh.GetMaterials()}
    eps_cf = geo.mesh.MaterialCF(eps_vals, default=1.0)
    ref = tmm.coh_tmm('s', [1.0, complex(N_SLAB), 1.0], [np.inf, D, np.inf], math.radians(THETA), LAM)
    Rt, Tt = ref['R'], ref['T']
    print("[t] CONICAL s-pol, theta={:.0f}deg, vs tmm R={:.4f} T={:.4f} (phi-invariant for a layered stack)".format(
        THETA, Rt, Tt), flush=True)
    ok = True
    Rs = []
    for phi in PHIS:
        opt = OpticalSpec(polarization="y", incidence_angle_deg=THETA, azimuth_deg=phi,
                           linear_solver="umfpack")
        res = solve_fem(geo, lam_m, eps_cf, opt, order=2, n_super=1.0 + 0j, n_sub=1.0 + 0j)
        Tf = res.T if res.T is not None else float('nan')
        dR, dT = abs(res.R - Rt), abs(Tf - Tt)
        good = dR < TOL and dT < TOL
        ok = ok and good
        Rs.append(res.R)
        print("[t] phi={:3.0f}d: R {:.4f}/{:.4f}  T {:.4f}/{:.4f}  R+T={:.4f}  {}".format(
            phi, res.R, Rt, Tf, Tt, res.R + (Tf if Tf == Tf else 0),
            "OK" if good else "MISMATCH(dR={:.3f},dT={:.3f})".format(dR, dT)), flush=True)
    phi_spread = max(Rs) - min(Rs)
    # phi-spread is a SAME-MESH azimuthal difference for an isotropic stack -> it should be
    # at quadrature-noise level, far below the per-phi tmm tolerance. Gate it tightly so a
    # transverse-phase/extraction defect cannot hide under the loose 0.03 (audit F3).
    SPREAD_TOL = 5e-3
    print("[t] R phi-spread = {:.4f} (should be ~0 for an isotropic layered stack; tol {:.0e})".format(
        phi_spread, SPREAD_TOL), flush=True)
    ok = ok and phi_spread < SPREAD_TOL
    print("[t] *** CONICAL INCIDENCE (s-pol, phi-invariance + tmm): {} ***".format("PASS" if ok else "FAIL"),
           flush=True)
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
