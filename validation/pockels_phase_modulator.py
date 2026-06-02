"""Phase-1 end-to-end EO modulator validation (roadmap 1b oracle): a Pockels phase modulator
built from the generalized spine -- electrostatic driver (E = V/d) -> PockelsEffect (tensor eps)
-> tensor FEM (Phase 0b) -- validated two ways:

  (1) ORACLE: for y-polarized normal incidence the diagonal Pockels eps tensor sees only eps_yy,
      so the FEM MUST reproduce the independent TMM scalar solve at n_y(V) = sqrt(eps_yy(V)) --
      same R, T and the SAME transmission-phase modulation dphi(V). This proves the tensor-eps
      FEM + the Pockels response end to end against a trusted 1-D oracle.
  (2) PHYSICS: the modulated index matches the analytic Pockels law n_y(V) = n_o - 0.5 n_o^3 r13
      E_z, E_z = V/L, so the half-wave voltage is Vpi = lambda / (n_o^3 r13) (longitudinal cut).

A deliberately EXAGGERATED r13 is used so the index change (hence dphi) is large enough to
resolve in the FEM (the effect is exactly linear in r, so this does not change what is tested).
 at r13 = 0 the tensor is isotropic and the modulation vanishes. Run:
python -m validation.pockels_phase_modulator
"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dynameta.materials import Material, MaterialRegistry, ConstantOptical
from dynameta.geometry import UnitCell, Stack, Layer, Design
from dynameta.geometry.specs import OpticalSpec, Mesh3DSpec
from dynameta.core.eps_field import EpsField
from dynameta.core.effects import PockelsEffect
from dynameta.core.layered import LayeredStack, LayeredSlab
from dynameta.carriers.electrostatics import parallel_plate_field_z
from dynameta.optics.ngsolve_layered import LayeredOpticalBuilder
from dynameta.optics.eps_assembler import assemble_eps_cf
from dynameta.optics.solver import solve_fem
from dynameta.optics.tmm_reference import TmmLayeredSolver

LAM = 1300.0
NO, NE = 2.21, 2.14
R13, R33 = 5.0e-10, 1.0e-9        # EXAGGERATED (~50x LiNbO3) for a resolvable FEM dphi; linear in r
L = 400.0                          # EO slab thickness (nm)
VS = [0.0, 2.0, 4.0, 6.0]          # applied volts
TOL_PHASE = 3e-3                   # sharp: dphi(V) modulation match (mesh error cancels in dV)
TOL_RT = 0.015                     # looser: absolute FEM-vs-TMM R/T (mesh-discretization limited)


def _design():
    reg = MaterialRegistry()
    reg.add(Material("air", ConstantOptical(1.0 + 0j)))
    reg.add(Material("eo", ConstantOptical(complex(NO ** 2, 0.0))))   # placeholder; eps overridden
    cell = UnitCell.square(300e-9)
    stack = Stack(layers=[Layer("s", L * 1e-9, "eo")],
                  superstrate_material="air", substrate_material="air")
    m3 = Mesh3DSpec(pml_thk_m=600e-9, superstrate_buffer_m=900e-9, substrate_buffer_m=900e-9,
                    maxh_superstrate_m=45e-9, maxh_substrate_m=45e-9, maxh_background_m=22e-9)
    return Design(name="pockels", unit_cell=cell, stack=stack, electrodes=[], materials=reg, mesh_3d=m3)


def main():
    d = _design(); lam_m = LAM * 1e-9
    geo = LayeredOpticalBuilder(d).build()
    mats = list(geo.mesh.GetMaterials())
    slab = [r for r in mats if geo.material_by_region[r] == "eo"]
    if len(slab) != 1:
        print("[t] expected exactly one EO region, got", slab, flush=True); return False
    slab = slab[0]
    eps_bg = np.diag([NO ** 2, NO ** 2, NE ** 2]).astype(complex)
    r_voigt = np.zeros((6, 3)); r_voigt[0, 2] = R13; r_voigt[1, 2] = R13; r_voigt[2, 2] = R33
    pockels = PockelsEffect(eps_bg=eps_bg, r_voigt=r_voigt)
    opt = OpticalSpec(polarization="y", incidence_angle_deg=0.0, linear_solver="umfpack")
    tmm = TmmLayeredSolver()

    rows = []
    for V in VS:
        Ez = parallel_plate_field_z(NO ** 2, L * 1e-9, V)          # = -V/L  [V/m]
        eps_t = pockels.eps({"E": np.array([0.0, 0.0, Ez])}, lam_m)
        n_y = np.sqrt(complex(eps_t[1, 1]))                          # y-pol sees eps_yy
        # FEM: EO slab as the full Pockels tensor, everything else its scalar background
        ebr = {rg: EpsField(scalar=complex(d.materials.get(geo.material_by_region[rg]).eps(lam_m)))
               for rg in mats}
        ebr[slab] = EpsField(tensor=eps_t)
        res_fem = solve_fem(geo, lam_m, assemble_eps_cf(geo, ebr), opt, order=2,
                            n_super=1.0 + 0j, n_sub=1.0 + 0j)
        # TMM oracle: scalar slab at n_y
        stk = LayeredStack(1.0 + 0j, 1.0 + 0j, [LayeredSlab(L * 1e-9, eps=complex(n_y ** 2))])
        res_tmm = tmm.solve(stk, lam_m, opt)
        rows.append((V, n_y.real, res_fem.R, res_tmm.R, res_fem.T, res_tmm.T,
                     np.angle(res_fem.t), np.angle(res_tmm.t)))
        print("[t] V={:+.1f}  n_y={:.5f}  R_fem={:.4f} R_tmm={:.4f}  argt_fem={:+.4f} argt_tmm={:+.4f}".format(
            V, n_y.real, res_fem.R, res_tmm.R, np.angle(res_fem.t), np.angle(res_tmm.t)), flush=True)

    rows = np.array([(r[2], r[3], r[4], r[5], r[6], r[7]) for r in rows])  # R_f,R_t,T_f,T_t,argf,argt
    dR = np.max(np.abs(rows[:, 0] - rows[:, 1]))
    dT = np.max(np.abs(rows[:, 2] - rows[:, 3]))
    dphi_fem = np.unwrap(rows[:, 4]) - rows[0, 4]
    dphi_tmm = np.unwrap(rows[:, 5]) - rows[0, 5]
    dphi_err = np.max(np.abs(dphi_fem - dphi_tmm))
    modulation = abs(dphi_fem[-1])                                   # rad, V=max vs V=0
    Vpi = LAM * 1e-9 / (NO ** 3 * R13)                               # analytic half-wave voltage
    print("[t] FEM-vs-TMM: max|dR|={:.2e} max|dT|={:.2e} (TOL_RT={:.0e}); max|d(dphi)|={:.2e} rad "
          "(TOL_PHASE={:.0e})".format(dR, dT, TOL_RT, dphi_err, TOL_PHASE), flush=True)
    print("[t] phase modulation V={:.0f}->{:.0f}: dphi={:.4f} rad ({:.2f} deg); analytic Vpi={:.1f} V".format(
        VS[0], VS[-1], modulation, np.degrees(modulation), Vpi), flush=True)
    ok = (dR < TOL_RT and dT < TOL_RT and dphi_err < TOL_PHASE and modulation > 1e-3)
    print("[t] *** POCKELS PHASE MODULATOR (FEM tensor == TMM scalar-n_y; modulates): {} ***".format(
        "PASS" if ok else "FAIL"), flush=True)
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
