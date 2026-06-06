"""Graphene conductive-SHEET surface-current FEM oracle (C3): a sheet of surface conductivity sigma
(siemens) on an internal layer interface carries J_s = sigma E_tan, giving the tangential-trace Robin
term in the HCurl solve (solver.solve_fem(sheet_bcs={'iface_z<nm>': sigma}); the interface is named by
the builder pre-OCCGeometry). Checked against the analytic conductive-sheet Fresnel formula
core.graphene.sheet_rt: r = (n1 - n2 - Z0 sigma)/(n1 + n2 + Z0 sigma), t = 2 n1/(n1 + n2 + Z0 sigma).

GATE A: FEM R/T/A == sheet_rt across a range of sigma (a weak graphene sheet AND strong synthetic
        sheets that absorb ~50%), to ~1e-2.
GATE B: sigma -> 0 recovers bare Fresnel (vacuum: R ~ 0, T ~ 1).
GATE C: gate-tunable -- sweeping the Fermi level E_F changes the FEM absorption, tracking the analytic
        graphene_sigma(E_F) -> sheet_rt absorption.
GATE D: energy budget R + T + A = 1 at every sigma (passive sheet; A from the volumetric/flux balance).

Run: python -m validation.graphene_sheet_fem
"""
import sys, os, warnings
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dynameta.materials import Material, MaterialRegistry, ConstantOptical
from dynameta.geometry import UnitCell, Stack, Layer, Design
from dynameta.geometry.specs import OpticalSpec, Mesh3DSpec
from dynameta.core.eps_field import EpsField
from dynameta.core.graphene import sheet_rt, graphene_sigma
from dynameta.optics.ngsolve_layered import LayeredOpticalBuilder
from dynameta.optics.eps_assembler import assemble_eps_cf
from dynameta.optics.solver import solve_fem

LAM = 1550.0
Q_E = 1.602176634e-19
TOL = 1.5e-2


def _design():
    reg = MaterialRegistry()
    reg.add(Material("air", ConstantOptical(1.0 + 0j)))
    reg.add(Material("vac_a", ConstantOptical(1.0 + 0j)))     # distinct names, same eps=1 so the
    reg.add(Material("vac_b", ConstantOptical(1.0 + 0j)))     # a/b interface survives the OCC Glue
    cell = UnitCell.square(300e-9)
    stack = Stack(layers=[Layer("a", 400e-9, "vac_a"), Layer("b", 400e-9, "vac_b")],
                  superstrate_material="air", substrate_material="air")
    m3 = Mesh3DSpec(pml_thk_m=600e-9, superstrate_buffer_m=900e-9, substrate_buffer_m=900e-9,
                    maxh_superstrate_m=45e-9, maxh_substrate_m=45e-9, maxh_background_m=40e-9)
    return Design(name="gr", unit_cell=cell, stack=stack, electrodes=[], materials=reg, mesh_3d=m3)


def _build():
    geo = LayeredOpticalBuilder(_design()).build()
    za, zb = geo.z_intervals_nm["a"], geo.z_intervals_nm["b"]
    shared = [z for z in (za[0], za[1]) if z in (zb[0], zb[1])][0]   # the a/b interface z (nm)
    iface = "iface_z{}".format(int(round(shared)))
    if iface not in set(geo.mesh.GetBoundaries()):
        raise RuntimeError("sheet interface '{}' not named by the builder; got {}".format(
            iface, sorted(b for b in set(geo.mesh.GetBoundaries()) if b.startswith('iface'))))
    mats = list(geo.mesh.GetMaterials())
    eps_cf = assemble_eps_cf(geo, {rg: EpsField(scalar=complex(1.0, 0.0)) for rg in mats})
    return geo, eps_cf, iface


def _fem(geo, eps_cf, iface, sigma):
    opt = OpticalSpec(polarization="y", incidence_angle_deg=0.0, linear_solver="umfpack")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return solve_fem(geo, LAM * 1e-9, eps_cf, opt, order=2, n_super=1.0 + 0j, n_sub=1.0 + 0j,
                         sheet_bcs=({iface: sigma} if sigma != 0 else None))


def main():
    geo, eps_cf, iface = _build()
    print("[gr] sheet interface = '{}'".format(iface), flush=True)
    sig_gr = graphene_sigma(0.4 * Q_E, LAM * 1e-9)

    # GATE A + D: FEM == sheet_rt, energy closes
    gate_a = gate_d = True
    print("[gr] {:>10} | {:>26} | {:>22}".format("sigma", "FEM R/T/A", "analytic R/T/A"), flush=True)
    for label, sigma in [("graphene", sig_gr), ("1e-3", 1e-3 + 0j), ("2e-3", 2e-3 + 0j), ("5e-3", 5e-3 + 0j)]:
        res = _fem(geo, eps_cf, iface, sigma)
        _, _, R_an, T_an, A_an = sheet_rt(1.0, 1.0, sigma)
        dR, dT, dA = abs(res.R - R_an), abs(res.T - T_an), abs(res.A - A_an)
        budget = abs(res.R + res.T + res.A - 1.0)
        gate_a = gate_a and dR < TOL and dT < TOL and dA < TOL
        gate_d = gate_d and budget < 1e-9
        print("[gr] {:>10} | R={:.4f} T={:.4f} A={:+.4f} | R={:.4f} T={:.4f} A={:+.4f}  dT={:.1e}".format(
            label, res.R, res.T, res.A, R_an, T_an, A_an, dT), flush=True)

    # GATE B: sigma -> 0 recovers bare Fresnel (vacuum: R~0, T~1)
    res0 = _fem(geo, eps_cf, iface, 0.0)
    gate_b = abs(res0.R) < 1e-2 and abs(res0.T - 1.0) < 1e-2
    print("[gr] sigma=0: R={:.4f} T={:.4f} (bare Fresnel R~0 T~1): {}".format(
        res0.R, res0.T, "PASS" if gate_b else "FAIL"), flush=True)

    # GATE C: gate-tunable -- sweep E_F, FEM absorption tracks analytic
    gate_c = True
    print("[gr] gate sweep (E_F):", flush=True)
    for ef in (0.2, 0.4, 0.6):
        sig = graphene_sigma(ef * Q_E, LAM * 1e-9)
        res = _fem(geo, eps_cf, iface, sig)
        _, _, _, _, A_an = sheet_rt(1.0, 1.0, sig)
        ok = abs(res.A - A_an) < TOL
        gate_c = gate_c and ok
        print("[gr]   E_F={:.1f} eV: sigma={:.3e} S  A_fem={:+.4f} A_an={:+.4f}  {}".format(
            ef, sig, res.A, A_an, "ok" if ok else "FAIL"), flush=True)

    overall = gate_a and gate_b and gate_c and gate_d
    print("[gr] GATE A (FEM==sheet_rt): {} | B (sigma->0 Fresnel): {} | C (gate-tunable): {} | "
          "D (R+T+A=1): {}".format(*("PASS" if g else "FAIL" for g in (gate_a, gate_b, gate_c, gate_d))),
          flush=True)
    print("[gr] *** GRAPHENE SHEET SURFACE-CURRENT FEM vs analytic sheet_rt: {} ***".format(
        "PASS" if overall else "FAIL"), flush=True)
    return overall


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
