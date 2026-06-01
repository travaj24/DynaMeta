"""Validate BOUNDARY-SPANNING material inclusions (the connected-grating case).

A periodic structure's specular (0th-order) R/T at NORMAL incidence is invariant under a
lateral translation of the unit-cell origin -- shifting the cell only shifts the fields, and
the cell-averaged reflection/transmission coefficients integrate that shift away. We exploit
that as an oracle-free check of the new boundary-spanning machinery:

  * REFERENCE: a full-y dielectric grating stripe centered at x = Px/2. It is interior in x
    (touches neither x-periodic face), so it builds via the proven interior path.
  * TEST: the SAME stripe centered at x = 0. It now CROSSES the x=0/x=Px boundary, so it only
    builds correctly with the new clip-to-cell + periodic-translate union
    (_inclusion_solids_clipped): the part beyond the cell wraps to the opposite face, and the
    resulting inclusion sub-faces on x=0 and x=Px are paired by _identify_periodic.

By translation invariance the two MUST give the same R and T; and a lossless subwavelength
grating MUST conserve energy R+T ~ 1 (all higher diffraction orders evanescent: lambda/Px > n
for every material, so only the 0th order propagates). A silently-failed periodic Identify on
the boundary case would leak energy (R+T != 1) OR diverge from the interior reference -- either
breaks the test. Run: python -m validation.boundary_inclusion_grating
"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dynameta.materials import Material, MaterialRegistry, ConstantOptical
from dynameta.geometry import UnitCell, Stack, Layer, Inclusion, Design
from dynameta.geometry.cross_section import Rectangle
from dynameta.geometry.specs import OpticalSpec, Mesh3DSpec
from dynameta.optics.ngsolve_layered import LayeredOpticalBuilder
from dynameta.optics.solver import solve_fem

LAM = 1300.0                 # nm
PX, PY = 400.0, 200.0        # nm -- subwavelength (lambda/Px = 3.25 > n_slab=2.5: only 0th order)
W, THK = 160.0, 220.0        # stripe width, grating thickness (nm)
N_BG, N_SLAB = 1.0, 2.5
TOL = 0.02


def _grating(cx_nm):
    """A full-y stripe (1D grating) of n=N_SLAB in n=N_BG, centered at x=cx_nm."""
    reg = MaterialRegistry()
    reg.add(Material("air", ConstantOptical(complex(N_BG ** 2, 0.0))))
    reg.add(Material("hi", ConstantOptical(complex(N_SLAB ** 2, 0.0))))
    cell = UnitCell(period_x_m=PX * 1e-9, period_y_m=PY * 1e-9)
    stripe = Rectangle(cx_m=cx_nm * 1e-9, cy_m=PY / 2 * 1e-9, width_m=W * 1e-9, height_m=PY * 1e-9)
    layers = [Layer("grating", THK * 1e-9, "air", inclusions=[Inclusion(stripe, "hi")])]
    stack = Stack(layers=layers, superstrate_material="air", substrate_material="air")
    m3 = Mesh3DSpec(pml_thk_m=700e-9, superstrate_buffer_m=1400e-9, substrate_buffer_m=1400e-9,
                     maxh_superstrate_m=40e-9, maxh_substrate_m=40e-9,
                     maxh_background_m=18e-9, maxh_inclusion_m=18e-9)
    return Design(name="grat", unit_cell=cell, stack=stack, electrodes=[], materials=reg, mesh_3d=m3)


def _solve(cx_nm, label):
    d = _grating(cx_nm)
    geo = LayeredOpticalBuilder(d).build()
    mats = list(geo.mesh.GetMaterials())
    n_incl_solids = sum(1 for m in mats if "__incl" in m)
    # one region name per inclusion, but it can be backed by >1 solid; report the material map
    incl_regions = sorted({m for m in mats if "__incl" in m})
    print("[t] {}: cx={:.0f}nm  regions={}  inclusion-regions={}".format(
        label, cx_nm, len(mats), incl_regions), flush=True)
    lam_m = LAM * 1e-9
    eps_vals = {r: complex(d.materials.get(geo.material_by_region[r]).eps(lam_m))
                for r in geo.mesh.GetMaterials()}
    eps_cf = geo.mesh.MaterialCF(eps_vals, default=1.0)
    opt = OpticalSpec(polarization="y", incidence_angle_deg=0.0, linear_solver="umfpack")
    res = solve_fem(geo, lam_m, eps_cf, opt, order=3, n_super=complex(N_BG), n_sub=complex(N_BG))
    T = res.T if res.T is not None else float("nan")
    print("[t] {}: R={:.4f}  T={:.4f}  R+T={:.4f}".format(label, res.R, T, res.R + T), flush=True)
    return res.R, T


def main():
    print("[t] BOUNDARY-SPANNING grating: lam={:.0f}nm Px={:.0f} Py={:.0f} w={:.0f} t={:.0f} "
          "n_bg={:.1f} n_slab={:.1f}".format(LAM, PX, PY, W, THK, N_BG, N_SLAB), flush=True)
    print("[t] lambda/Px = {:.2f} (> n_slab={:.1f} -> only 0th order propagates -> R+T=1)".format(
        LAM / PX, N_SLAB), flush=True)
    Ri, Ti = _solve(PX / 2.0, "INTERIOR  (x=Px/2, proven path)")
    Rb, Tb = _solve(0.0,       "BOUNDARY  (x=0, crosses face)")
    dR, dT = abs(Ri - Rb), abs(Ti - Tb)
    ei, eb = abs(Ri + Ti - 1.0), abs(Rb + Tb - 1.0)
    print("[t] translation-invariance: |dR|={:.4f}  |dT|={:.4f}  (TOL={:.3f})".format(dR, dT, TOL),
          flush=True)
    print("[t] energy-conservation:    interior |R+T-1|={:.4f}  boundary |R+T-1|={:.4f}".format(
        ei, eb), flush=True)
    ok = (dR < TOL and dT < TOL and ei < TOL and eb < TOL
          and 0.02 < Ri < 0.98)   # guard against the trivial R~0/R~1 degenerate match
    print("[t] *** BOUNDARY-SPANNING INCLUSION (grating translation-invariance + energy): {} ***".format(
        "PASS" if ok else "FAIL"), flush=True)


if __name__ == "__main__":
    main()
