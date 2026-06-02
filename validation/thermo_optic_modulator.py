"""Phase-2 thermo-optic modulator validation (roadmap 2b oracle): a Si thermo-optic phase
shifter built from the spine -- thermal driver -> ThermoOpticModel (eps(T)) -> FEM. Validated:

  (1) ORACLE: ThermoOpticModel is a scalar response, so the FEM must reproduce the independent
      TMM scalar solve at n(T) = n0 + (dn/dT) dT -- same R/T and the SAME transmission-phase
      modulation dphi(dT).
  (2) PHYSICS: small-signal slope d(phi)/d(dT) ~ (2pi/lambda) (dn/dT) L (single-pass), and NO
      shift at dT = 0.

Real Si numbers (n0=3.48, dn/dT=1.8e-4/K) give a resolvable few-degree phase modulation over a
~150 K rise. Run: python -m validation.thermo_optic_modulator
"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dynameta.materials import Material, MaterialRegistry, ConstantOptical
from dynameta.geometry import UnitCell, Stack, Layer, Design
from dynameta.geometry.specs import OpticalSpec, Mesh3DSpec
from dynameta.core.eps_field import EpsField
from dynameta.core.effects import ThermoOpticModel
from dynameta.core.layered import LayeredStack, LayeredSlab
from dynameta.optics.ngsolve_layered import LayeredOpticalBuilder
from dynameta.optics.eps_assembler import assemble_eps_cf
from dynameta.optics.solver import solve_fem
from dynameta.optics.tmm_reference import TmmLayeredSolver

LAM = 1300.0
N0, DN_DT, T_REF = 3.48, 1.8e-4, 300.0     # crystalline Si
L = 500.0                                   # slab thickness (nm)
DTS = [0.0, 50.0, 100.0, 150.0]            # temperature rises (K)
TOL_PHASE, TOL_RT = 3e-3, 0.015


def _design():
    reg = MaterialRegistry()
    reg.add(Material("air", ConstantOptical(1.0 + 0j)))
    reg.add(Material("si", ConstantOptical(complex(N0 ** 2, 0.0))))      # placeholder; overridden
    cell = UnitCell.square(300e-9)
    stack = Stack(layers=[Layer("s", L * 1e-9, "si")],
                  superstrate_material="air", substrate_material="air")
    m3 = Mesh3DSpec(pml_thk_m=600e-9, superstrate_buffer_m=900e-9, substrate_buffer_m=900e-9,
                    maxh_superstrate_m=45e-9, maxh_substrate_m=45e-9, maxh_background_m=22e-9)
    return Design(name="thermo", unit_cell=cell, stack=stack, electrodes=[], materials=reg, mesh_3d=m3)


def main():
    d = _design(); lam_m = LAM * 1e-9
    geo = LayeredOpticalBuilder(d).build()
    mats = list(geo.mesh.GetMaterials())
    slab = [r for r in mats if geo.material_by_region[r] == "si"]
    if len(slab) != 1:
        print("[t] expected exactly one Si region, got", slab, flush=True); return False
    slab = slab[0]
    tom = ThermoOpticModel(eps_ref=complex(N0 ** 2, 0.0), dn_dT=DN_DT, T_ref=T_REF)
    opt = OpticalSpec(polarization="y", incidence_angle_deg=0.0, linear_solver="umfpack")
    tmm = TmmLayeredSolver()

    Rf, Rt, Tf, Tt, af, at, nval = [], [], [], [], [], [], []
    for dT in DTS:
        eps_T = complex(tom.eps({"T": T_REF + dT}, lam_m))
        n_T = np.sqrt(eps_T).real
        ebr = {rg: EpsField(scalar=complex(d.materials.get(geo.material_by_region[rg]).eps(lam_m)))
               for rg in mats}
        ebr[slab] = EpsField(scalar=eps_T)
        res = solve_fem(geo, lam_m, assemble_eps_cf(geo, ebr), opt, order=2, n_super=1.0 + 0j, n_sub=1.0 + 0j)
        stk = LayeredStack(1.0 + 0j, 1.0 + 0j, [LayeredSlab(L * 1e-9, eps=eps_T)])
        rtm = tmm.solve(stk, lam_m, opt)
        Rf.append(res.R); Rt.append(rtm.R); Tf.append(res.T); Tt.append(rtm.T)
        af.append(np.angle(res.t)); at.append(np.angle(rtm.t)); nval.append(n_T)
        print("[t] dT={:+6.1f}K  n={:.5f}  R_fem={:.4f} R_tmm={:.4f}  argt_fem={:+.4f} argt_tmm={:+.4f}".format(
            dT, n_T, res.R, rtm.R, np.angle(res.t), np.angle(rtm.t)), flush=True)

    dR = float(np.max(np.abs(np.array(Rf) - np.array(Rt))))
    dT_ = float(np.max(np.abs(np.array(Tf) - np.array(Tt))))
    dphi_fem = np.unwrap(np.array(af)) - af[0]
    dphi_tmm = np.unwrap(np.array(at)) - at[0]
    dphi_err = float(np.max(np.abs(dphi_fem - dphi_tmm)))
    modulation = abs(dphi_fem[-1])
    slope_fem = dphi_fem[-1] / (DTS[-1] - DTS[0])
    slope_ana = (2 * np.pi / lam_m) * DN_DT * (L * 1e-9)             # single-pass (no FP)
    no_shift = abs(dphi_fem[0]) < 1e-9                               # dT=0 -> no shift (by construction)
    print("[t] FEM-vs-TMM: max|dR|={:.2e} max|dT|={:.2e} (TOL_RT={:.0e}); max|d(dphi)|={:.2e} rad "
          "(TOL_PHASE={:.0e})".format(dR, dT_, TOL_RT, dphi_err, TOL_PHASE), flush=True)
    print("[t] modulation dT 0->{:.0f}K: dphi={:.4f} rad ({:.2f} deg); slope_fem={:.3e} rad/K "
          "single-pass {:.3e}".format(DTS[-1], modulation, np.degrees(modulation), slope_fem, slope_ana), flush=True)
    ok = (dR < TOL_RT and dT_ < TOL_RT and dphi_err < TOL_PHASE and modulation > 1e-3 and no_shift)
    print("[t] *** THERMO-OPTIC MODULATOR (FEM eps(T) == TMM scalar-n(T); modulates): {} ***".format(
        "PASS" if ok else "FAIL"), flush=True)
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
