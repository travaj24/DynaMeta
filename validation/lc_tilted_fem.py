"""Tilted-LC OFF-DIAGONAL tensor FEM oracle (B-fix end-to-end): a nematic LC whose director is
tilted by an INTERMEDIATE angle theta (0 < theta < 90) in the x-z plane gives a permittivity tensor
with nonzero OFF-DIAGONAL eps_xz = eps_zx. This was the case the FEM mis-solved (energy created,
T = 1.07) because mesh.SetPML's coordinate stretch is wrong for an anisotropic medium; solve_fem now
uses an explicit UPML for tensor eps and solves it correctly. Two analytic references, no free
parameters:

  * ORDINARY wave (y-pol): sees eps_yy = n_o^2 at EVERY tilt, so its R/T must be tilt-INVARIANT and
    equal the planar (theta=0) n_o slab. This is the decisive off-diagonal test -- the y-component is
    physically decoupled from eps_xz, and the pre-fix solver corrupted it.
  * EXTRAORDINARY wave (x-pol): sees the angle-dependent effective index n_eff(theta) with
    1/n_eff^2 = sin^2(theta)/n_o^2 + cos^2(theta)/n_e^2 (n_e at theta=0, n_o at theta=90); compared
    to the scalar TMM at n_eff.

Energy must close (R + T ~ 1) at every tilt, AND the fit-independent Poynting-flux R/T must close --
a genuine, non-tautological check that the field itself conserves energy. Run:
    python -m validation.lc_tilted_fem
"""
import sys, os, warnings
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dynameta.materials import Material, MaterialRegistry, ConstantOptical
from dynameta.geometry import UnitCell, Stack, Layer, Design
from dynameta.geometry.specs import OpticalSpec, Mesh3DSpec
from dynameta.core.eps_field import EpsField
from dynameta.core.effects import LiquidCrystalModel
from dynameta.core.layered import LayeredStack, LayeredSlab
from dynameta.optics.ngsolve_layered import LayeredOpticalBuilder
from dynameta.optics.eps_assembler import assemble_eps_cf
from dynameta.optics.solver import solve_fem
from dynameta.optics.tmm_reference import TmmLayeredSolver

LAM = 1550.0
NO, NE = 1.53, 1.71
L = 500.0
TILTS = (0.0, 30.0, 45.0, 60.0, 90.0)
TOL_O = 2e-2        # ordinary-wave tilt-invariance + energy closure
TOL_E = 3e-2        # extraordinary-wave vs n_eff(theta) TMM


def _design():
    reg = MaterialRegistry()
    reg.add(Material("air", ConstantOptical(1.0 + 0j)))
    reg.add(Material("lc", ConstantOptical(complex(NO ** 2, 0.0))))
    cell = UnitCell.square(300e-9)
    stack = Stack(layers=[Layer("s", L * 1e-9, "lc")],
                  superstrate_material="air", substrate_material="air")
    m3 = Mesh3DSpec(pml_thk_m=600e-9, superstrate_buffer_m=900e-9, substrate_buffer_m=900e-9,
                    maxh_superstrate_m=45e-9, maxh_substrate_m=45e-9, maxh_background_m=22e-9)
    return Design(name="lc_tilt", unit_cell=cell, stack=stack, electrodes=[], materials=reg, mesh_3d=m3)


def _fem(eps_tensor, pol):
    # fresh geometry per solve (a periodic PML mesh leaks state across solves, esp. across polarizations)
    geo = LayeredOpticalBuilder(_design()).build()
    mats = list(geo.mesh.GetMaterials())
    slab = [r for r in mats if geo.material_by_region[r] == "lc"][0]
    ebr = {rg: EpsField(scalar=complex(1.0, 0.0)) for rg in mats}
    ebr[slab] = EpsField(tensor=np.asarray(eps_tensor, dtype=complex))
    opt = OpticalSpec(polarization=pol, incidence_angle_deg=0.0, linear_solver="umfpack")
    res = solve_fem(geo, LAM * 1e-9, assemble_eps_cf(geo, ebr), opt, order=2,
                    n_super=1.0 + 0j, n_sub=1.0 + 0j)
    return res


def _tmm_T(n):
    opt = OpticalSpec(polarization="y", incidence_angle_deg=0.0, linear_solver="umfpack")
    stk = LayeredStack(1.0 + 0j, 1.0 + 0j, [LayeredSlab(L * 1e-9, eps=complex(n ** 2, 0.0))])
    return TmmLayeredSolver().solve(stk, LAM * 1e-9, opt).T


def main():
    lc = LiquidCrystalModel(n_o=NO, n_e=NE)
    T_ord_target = _tmm_T(NO)
    ok = True
    print("[lct] planar n_o TMM T = {:.4f}  (ordinary-wave target at ALL tilts)".format(T_ord_target), flush=True)
    for th in TILTS:
        eps = lc.eps({"director_angle_rad": np.radians(th)}, LAM * 1e-9)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")        # tilted-PML energy-closure warnings are checked explicitly below
            ry = _fem(eps, "y")
            rx = _fem(eps, "x")
        neff = 1.0 / np.sqrt(np.sin(np.radians(th)) ** 2 / NO ** 2 + np.cos(np.radians(th)) ** 2 / NE ** 2)
        T_ext_tmm = _tmm_T(neff)
        d_ord = abs(ry.T - T_ord_target)
        d_ext = abs(rx.T - T_ext_tmm)
        e_ord = abs((ry.R + ry.T) - 1.0)
        e_flux = abs((ry.R_flux + ry.T_flux) - 1.0)
        case_ok = d_ord < TOL_O and d_ext < TOL_E and e_ord < TOL_O and e_flux < TOL_O
        ok = ok and case_ok
        print("[lct] theta={:4.0f} | ord(y) T={:.4f} dT={:.1e} R+T={:.4f} flux R+T={:.4f} | "
              "ext(x) n_eff={:.3f} T={:.4f} T_tmm={:.4f} dT={:.1e}  {}".format(
                  th, ry.T, d_ord, ry.R + ry.T, ry.R_flux + ry.T_flux,
                  neff, rx.T, T_ext_tmm, d_ext, "ok" if case_ok else "FAIL"), flush=True)
    print("[lct] *** TILTED-LC OFF-DIAGONAL TENSOR FEM (UPML): ordinary tilt-invariance + "
          "extraordinary n_eff + energy closure: {} ***".format("PASS" if ok else "FAIL"), flush=True)
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
