"""Non-vacuum superstrate at NORMAL incidence (audit P1 regression guard): the FEM must use the
incidence-medium z-wavevector kz_s = n_super*k0, not the vacuum k0. Two parameter-free checks:

  GATE A: a fully HOMOGENEOUS medium (n_super = n_sub = n_layer) has no interface, so R = 0, T = 1
          for any n. The pre-fix solver used kz_s = k0 and returned a spurious R ~ 0.04 (and
          ~32% phantom loss) for n = 1.5 -- this gate fails on the old code, passes on the fixed.
  GATE B: a bare n1 | n2 interface (dense superstrate over a denser substrate) matches the analytic
          normal-incidence Fresnel R = ((n1 - n2)/(n1 + n2))^2, T = 1 - R.

Run: python -m validation.superstrate_index_fem
"""
import sys, os, warnings
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dynameta.materials import Material, MaterialRegistry, ConstantOptical
from dynameta.geometry import UnitCell, Stack, Layer, Design
from dynameta.geometry.specs import OpticalSpec, Mesh3DSpec
from dynameta.core.eps_field import EpsField
from dynameta.optics.ngsolve_layered import LayeredOpticalBuilder
from dynameta.optics.eps_assembler import assemble_eps_cf
from dynameta.optics.solver import solve_fem

LAM = 1550.0
TOL = 1.5e-2


def _design(n_layer):
    reg = MaterialRegistry()
    reg.add(Material("sup", ConstantOptical(complex(n_layer ** 2, 0.0))))
    cell = UnitCell.square(300e-9)
    stack = Stack(layers=[Layer("s", 600e-9, "sup")], superstrate_material="sup", substrate_material="sup")
    m3 = Mesh3DSpec(pml_thk_m=600e-9, superstrate_buffer_m=900e-9, substrate_buffer_m=900e-9,
                    maxh_superstrate_m=45e-9, maxh_substrate_m=45e-9, maxh_background_m=45e-9)
    return Design(name="sup", unit_cell=cell, stack=stack, electrodes=[], materials=reg, mesh_3d=m3)


def _fem(n_layer, n_super, n_sub):
    geo = LayeredOpticalBuilder(_design(n_layer)).build()
    mats = list(geo.mesh.GetMaterials())
    eps_cf = assemble_eps_cf(geo, {rg: EpsField(scalar=complex(n_layer ** 2, 0.0)) for rg in mats})
    opt = OpticalSpec(polarization="y", incidence_angle_deg=0.0, linear_solver="umfpack")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return solve_fem(geo, LAM * 1e-9, eps_cf, opt, order=2,
                         n_super=complex(n_super), n_sub=complex(n_sub))


def main():
    ok = True
    # GATE A: homogeneous medium -> R=0, T=1 for several n
    print("[sup] GATE A -- homogeneous medium (R=0, T=1):", flush=True)
    for n in (1.5, 2.0):
        res = _fem(n, n, n)
        a_ok = abs(res.R) < TOL and abs(res.T - 1.0) < TOL
        ok = ok and a_ok
        print("[sup]   n_super=n_sub=n_layer={:.2f}: R={:.4f} T={:.4f}  {}".format(
            n, res.R, res.T, "ok" if a_ok else "FAIL (pre-fix bug: spurious R)"), flush=True)

    # GATE B: bare n1|n2 Fresnel interface (n2 layer over n2 substrate, n1 superstrate)
    print("[sup] GATE B -- bare Fresnel interface n1|n2 vs analytic:", flush=True)
    for n1, n2 in ((1.5, 2.0), (2.0, 1.5)):
        # build with the layer+substrate = n2, superstrate = n1
        reg = MaterialRegistry()
        reg.add(Material("n1", ConstantOptical(complex(n1 ** 2, 0.0))))
        reg.add(Material("n2", ConstantOptical(complex(n2 ** 2, 0.0))))
        cell = UnitCell.square(300e-9)
        stack = Stack(layers=[Layer("s", 800e-9, "n2")], superstrate_material="n1", substrate_material="n2")
        m3 = Mesh3DSpec(pml_thk_m=600e-9, superstrate_buffer_m=900e-9, substrate_buffer_m=900e-9,
                        maxh_superstrate_m=45e-9, maxh_substrate_m=45e-9, maxh_background_m=45e-9)
        geo = LayeredOpticalBuilder(Design(name="fr", unit_cell=cell, stack=stack, electrodes=[],
                                           materials=reg, mesh_3d=m3)).build()
        mats = list(geo.mesh.GetMaterials())
        ebr = {rg: EpsField(scalar=complex(n2 ** 2, 0.0)) for rg in mats}
        for rg in mats:
            if geo.material_by_region.get(rg) == "n1":
                ebr[rg] = EpsField(scalar=complex(n1 ** 2, 0.0))
        opt = OpticalSpec(polarization="y", incidence_angle_deg=0.0, linear_solver="umfpack")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = solve_fem(geo, LAM * 1e-9, assemble_eps_cf(geo, ebr), opt, order=2,
                            n_super=complex(n1), n_sub=complex(n2))
        R_an = ((n1 - n2) / (n1 + n2)) ** 2
        T_an = 1.0 - R_an
        b_ok = abs(res.R - R_an) < TOL and abs(res.T - T_an) < TOL
        ok = ok and b_ok
        print("[sup]   n1={:.2f} n2={:.2f}: R_fem={:.4f} R_an={:.4f} T_fem={:.4f} T_an={:.4f}  {}".format(
            n1, n2, res.R, R_an, res.T, T_an, "ok" if b_ok else "FAIL"), flush=True)

    # GATE C (audit C3-7): dense superstrate over a LOSSY layer -- the independent
    # absorption integral must close the energy budget, |A_ind - (1-R-T)| small. The
    # pre-fix normalization omitted Re(n_super), inflating A_ind (and every D2 per-region
    # value) by exactly 1.5x here, breaking closure by ~0.5*A while R/T stayed correct.
    print("[sup] GATE C -- dense superstrate + lossy layer: A_independent closes 1-R-T:", flush=True)
    n1, eps_lossy = 1.5, complex(4.0, 0.6)
    reg = MaterialRegistry()
    reg.add(Material("n1", ConstantOptical(complex(n1 ** 2, 0.0))))
    reg.add(Material("abs", ConstantOptical(eps_lossy)))
    cell = UnitCell.square(300e-9)
    stack = Stack(layers=[Layer("s", 300e-9, "abs")], superstrate_material="n1",
                  substrate_material="n1")
    m3 = Mesh3DSpec(pml_thk_m=600e-9, superstrate_buffer_m=900e-9, substrate_buffer_m=900e-9,
                    maxh_superstrate_m=45e-9, maxh_substrate_m=45e-9, maxh_background_m=45e-9)
    geo = LayeredOpticalBuilder(Design(name="ls", unit_cell=cell, stack=stack, electrodes=[],
                                       materials=reg, mesh_3d=m3)).build()
    mats = list(geo.mesh.GetMaterials())
    ebr = {rg: EpsField(scalar=complex(n1 ** 2, 0.0)) for rg in mats}
    for rg in mats:
        if geo.material_by_region.get(rg) == "abs":
            ebr[rg] = EpsField(scalar=eps_lossy)
    opt = OpticalSpec(polarization="y", incidence_angle_deg=0.0, linear_solver="umfpack")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = solve_fem(geo, LAM * 1e-9, assemble_eps_cf(geo, ebr), opt, order=2,
                        n_super=complex(n1), n_sub=complex(n1))
    budget = 1.0 - res.R - res.T
    d_close = abs(res.A_independent - budget)
    ratio = res.A_independent / budget if budget > 1e-6 else float("nan")
    pra_sum = sum((res.per_region_absorption or {}).values())
    c_ok = (d_close < TOL and budget > 0.05
            and abs(pra_sum - res.A_independent) < 1e-10)
    ok = ok and c_ok
    print("[sup]   A_ind={:.4f} 1-R-T={:.4f} |d|={:.1e} (ratio {:.3f}; pre-fix ~1.5); "
          "sum(pra)==A_ind ({:.1e})  {}".format(
              res.A_independent, budget, d_close, ratio, abs(pra_sum - res.A_independent),
              "ok" if c_ok else "FAIL"), flush=True)

    print("[sup] *** NON-VACUUM SUPERSTRATE (normal incidence) kz_s=n_super*k0: {} ***".format(
        "PASS" if ok else "FAIL"), flush=True)
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
