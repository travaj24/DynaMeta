"""Validate a CORNER-spanning inclusion -- the full 2D boundary case that exercises the
DIAGONAL periodic translates (both +/-Px AND +/-Py at once), not just the 1D x-crossing of
the grating test.

A 2D-periodic array of dielectric disks at NORMAL incidence is subwavelength here
(lambda/P = 3.25 > n_disk = 2.5), so only the 0th order propagates and R+T ~ 1. As with the
grating, the specular R/T are invariant under a lateral shift of the cell origin:

  * REFERENCE: a disk centered at (Px/2, Py/2) -- interior, builds via the proven path.
  * TEST: the SAME disk centered at the CORNER (0, 0). It crosses BOTH the x=0/x=Px and the
    y=0/y=Py boundaries, so within the cell it becomes FOUR quarter-disks at the four corners.
    Only the new clip-to-cell + 9-translate union (_inclusion_solids_clipped, where the
    diagonal (+/-Px, +/-Py) translates now matter) reassembles it, and _identify_periodic must
    pair the quarter-disk sub-faces on x=0<->x=Px AND y=0<->y=Py.

Equal R/T (translation invariance) + R+T ~ 1 (energy) proves the diagonal-translate corner
case. Run: python -m validation.boundary_inclusion_corner_circle
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dynameta.materials import Material, MaterialRegistry, ConstantOptical
from dynameta.geometry import UnitCell, Stack, Layer, Inclusion, Design
from dynameta.geometry.cross_section import Circle
from dynameta.geometry.specs import OpticalSpec, Mesh3DSpec
from dynameta.optics.ngsolve_layered import LayeredOpticalBuilder
from dynameta.optics.solver import solve_fem

LAM = 1300.0
P = 400.0                    # square period (nm); lambda/P = 3.25 > n_disk -> 0th order only
RAD, THK = 130.0, 220.0      # disk radius, thickness (nm); RAD < P/2 so the centered disk is interior
N_BG, N_DISK = 1.0, 2.5
TOL = 0.025


def _design(cx_nm, cy_nm):
    reg = MaterialRegistry()
    reg.add(Material("air", ConstantOptical(complex(N_BG ** 2, 0.0))))
    reg.add(Material("hi", ConstantOptical(complex(N_DISK ** 2, 0.0))))
    cell = UnitCell(period_x_m=P * 1e-9, period_y_m=P * 1e-9)
    disk = Circle(cx_m=cx_nm * 1e-9, cy_m=cy_nm * 1e-9, radius_m=RAD * 1e-9)
    layers = [Layer("disks", THK * 1e-9, "air", inclusions=[Inclusion(disk, "hi")])]
    stack = Stack(layers=layers, superstrate_material="air", substrate_material="air")
    # CI FIXTURE SHRINK (2026-08-31): maxh_background/maxh_inclusion 22 -> 60 nm and
    # maxh_super/substrate 45 -> 100 nm. Nightly run 33312685617 (2026-08-30) reported this oracle
    # OVER-BUDGET on the 16 GB runner, so it has never executed on CI. The 12.5 GB in that log is the
    # WATCHDOG KILL POINT (12.4 GB budget + one poll), NOT the peak: the old fixture is 1.20M order-3
    # HCurl DOFs whose UMFPACK factorization passed 45 GB on the dev box WITHOUT FINISHING ITS FIRST
    # SOLVE. After the shrink the measured dev-box peak is 6.0-6.9 GB (two runs) in 70-80 s.
    # THE CLAIM AND THE TOLERANCE ARE UNCHANGED (diagonal-translate translation-invariance |dR|,|dT|
    # and R+T = 1, both at the SAME TOL = 0.025, and the same 0.02 < R < 0.98 guard; the CORNER case
    # still reassembles as FOUR quarter-disks -- inclusion-subsolids = 4 -- which is the thing under
    # test). Five passing rungs, (maxh_bg/incl, maxh_buf) = (40,80)/(50,90)/(55,90)/(60,90)/(50,110)
    # /(60,100) nm, give interior R = 0.0647/0.0636/0.0631/0.0632/0.0636/0.0632 with the GATE
    # quantity |dR| = 0.0003/0.0005/0.0011/0.0008/0.0005/0.0008 and |R+T-1| <= 0.0002 throughout --
    # 30x inside TOL at every rung. CAUTION: like the sibling grating oracle this fixture has a
    # cliff at a much coarser buffer -- (60,140) makes the probe band stop being a clean two-wave
    # field and the answer goes unphysical (R+T = 1.29). It fails LOUDLY (solve_fem's two-wave-fit /
    # unphysical-R / energy-closure warnings plus a gate FAIL), but do not push maxh_super/substrate
    # past 100 nm here without re-running the ladder.
    m3 = Mesh3DSpec(pml_thk_m=700e-9, superstrate_buffer_m=1400e-9, substrate_buffer_m=1400e-9,
                     maxh_superstrate_m=100e-9, maxh_substrate_m=100e-9,
                     maxh_background_m=60e-9, maxh_inclusion_m=60e-9)
    return Design(name="disk", unit_cell=cell, stack=stack, electrodes=[], materials=reg, mesh_3d=m3)


def _solve(cx, cy, label):
    d = _design(cx, cy)
    geo = LayeredOpticalBuilder(d).build()
    mats = list(geo.mesh.GetMaterials())
    n_incl = sum(1 for m in mats if "__incl" in m)   # # of inclusion sub-solids (4 at the corner)
    print("[t] {}: center=({:.0f},{:.0f})nm  inclusion-subsolids={}  ne={}".format(
        label, cx, cy, n_incl, geo.mesh.ne), flush=True)
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
    print("[t] CORNER-SPANNING disk array: lam={:.0f}nm P={:.0f} r={:.0f} t={:.0f} "
          "n_bg={:.1f} n_disk={:.1f} (lambda/P={:.2f})".format(
              LAM, P, RAD, THK, N_BG, N_DISK, LAM / P), flush=True)
    Ri, Ti = _solve(P / 2, P / 2, "INTERIOR (center, 1 disk)")
    Rc, Tc = _solve(0.0, 0.0,     "CORNER   (4 quarter-disks)")
    dR, dT = abs(Ri - Rc), abs(Ti - Tc)
    ei, ec = abs(Ri + Ti - 1.0), abs(Rc + Tc - 1.0)
    print("[t] translation-invariance: |dR|={:.4f}  |dT|={:.4f}  (TOL={:.3f})".format(dR, dT, TOL),
          flush=True)
    print("[t] energy-conservation:    interior |R+T-1|={:.4f}  corner |R+T-1|={:.4f}".format(
        ei, ec), flush=True)
    ok = (dR < TOL and dT < TOL and ei < TOL and ec < TOL and 0.02 < Ri < 0.98)
    print("[t] *** CORNER-SPANNING INCLUSION (diagonal-translate, translation-invariance + "
          "energy): {} ***".format("PASS" if ok else "FAIL"), flush=True)
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
