"""Franz-Keldysh + DC-Kerr electro-modulation FEM oracle (audit Finding 9): FranzKeldyshEffect and
KerrEffect were the only field-effect models with NO independent / FEM-level oracle (their unit tests
only checked self-consistent reductions). Here each model's field-dependent permittivity is driven
through the actual solver.solve_fem and checked against the scalar TMM at the SAME eps -- an exact,
independent reference for a homogeneous slab (the FK eps is scalar; the isotropic Kerr eps is a
scalar*I tensor whose y-pol wave sees eps_yy).

GATE A (Franz-Keldysh, electro-ABSORPTION): FEM R/T/A == TMM at eps(E) across fields; E=0 recovers
        the background, and the absorbed fraction A GROWS monotonically with |E| (Im(eps) += beta|E|).
GATE B (DC-Kerr, electro-REFRACTION): FEM R/T and the transmitted PHASE arg(t) == TMM at eps(E)
        across fields; the field lowers Re(eps) (s_kerr>0 raises the impermeability), and in the weak
        regime the FEM index shift matches the perturbative dn = -s_kerr |E|^2 eps_bg^(3/2)/2.

Run: python -m validation.field_effect_em_fem
"""
import sys, os, warnings
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dynameta.materials import Material, MaterialRegistry, ConstantOptical
from dynameta.geometry import UnitCell, Stack, Layer, Design
from dynameta.geometry.specs import OpticalSpec, Mesh3DSpec
from dynameta.core.eps_field import EpsField
from dynameta.core.effects import FranzKeldyshEffect, KerrEffect
from dynameta.core.layered import LayeredStack, LayeredSlab
from dynameta.optics.ngsolve_layered import LayeredOpticalBuilder
from dynameta.optics.eps_assembler import assemble_eps_cf
from dynameta.optics.solver import solve_fem
from dynameta.optics.tmm_reference import TmmLayeredSolver

LAM = 1550.0
L = 300.0
TOL = 1.5e-2


def _design():
    reg = MaterialRegistry()
    reg.add(Material("air", ConstantOptical(1.0 + 0j)))
    reg.add(Material("eo", ConstantOptical(4.0 + 0j)))
    cell = UnitCell.square(300e-9)
    stack = Stack(layers=[Layer("s", L * 1e-9, "eo")], superstrate_material="air", substrate_material="air")
    m3 = Mesh3DSpec(pml_thk_m=600e-9, superstrate_buffer_m=900e-9, substrate_buffer_m=900e-9,
                    maxh_superstrate_m=45e-9, maxh_substrate_m=45e-9, maxh_background_m=30e-9)
    return Design(name="eo", unit_cell=cell, stack=stack, electrodes=[], materials=reg, mesh_3d=m3)


def _fem(eps_slab):
    geo = LayeredOpticalBuilder(_design()).build()
    mats = list(geo.mesh.GetMaterials())
    slab = [r for r in mats if geo.material_by_region[r] == "eo"][0]
    ebr = {rg: EpsField(scalar=complex(1.0, 0.0)) for rg in mats}
    eps_slab = np.asarray(eps_slab)
    ebr[slab] = (EpsField(tensor=eps_slab.astype(complex)) if eps_slab.shape == (3, 3)
                 else EpsField(scalar=complex(eps_slab)))
    opt = OpticalSpec(polarization="y", incidence_angle_deg=0.0, linear_solver="umfpack")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return solve_fem(geo, LAM * 1e-9, assemble_eps_cf(geo, ebr), opt, order=2,
                         n_super=1.0 + 0j, n_sub=1.0 + 0j)


def _tmm(eps_scalar):
    opt = OpticalSpec(polarization="y", incidence_angle_deg=0.0, linear_solver="umfpack")
    stk = LayeredStack(1.0 + 0j, 1.0 + 0j, [LayeredSlab(L * 1e-9, eps=complex(eps_scalar))])
    return TmmLayeredSolver().solve(stk, LAM * 1e-9, opt)


def main():
    ok = True

    # GATE A: Franz-Keldysh electro-absorption
    fk = FranzKeldyshEffect(eps_bg=complex(11.56, 0.0), beta=1.0e-8)   # n~3.4; Im(eps)+=beta|E|
    print("[fe] GATE A -- Franz-Keldysh (FEM == TMM at eps(E); A grows with |E|):", flush=True)
    A_prev, gate_a = -1.0, True
    for Ez in (0.0, 5e6, 1e7):
        eps_E = complex(fk.eps({"E": [0.0, 0.0, Ez]}, LAM * 1e-9))
        rf = _fem(eps_E); rt = _tmm(eps_E)
        dR, dT, dA = abs(rf.R - rt.R), abs(rf.T - rt.T), abs(rf.A - rt.A)
        grow = rf.A > A_prev - 1e-4
        match = dR < TOL and dT < TOL and dA < TOL and grow
        gate_a = gate_a and match
        print("[fe]   E={:.1e} V/m: eps={:.3f}{:+.3f}j  A_fem={:.4f} A_tmm={:.4f} dR={:.1e} dT={:.1e}  {}".format(
            Ez, eps_E.real, eps_E.imag, rf.A, rt.A, dR, dT, "ok" if match else "FAIL"), flush=True)
        A_prev = rf.A
    print("[fe] GATE A: {}".format("PASS" if gate_a else "FAIL"), flush=True)
    ok = ok and gate_a

    # GATE B: DC-Kerr electro-refraction. The ABSOLUTE transmitted phase has a constant FEM-vs-TMM
    # z-origin offset, so compare the FIELD-INDUCED phase SHIFT relative to E=0 (which is the physical
    # modulation signal); also cross-check the index shift vs the perturbative model.
    eps_bg = np.diag([4.0, 4.0, 4.0]).astype(complex)                  # n=2
    kerr = KerrEffect(eps_bg=eps_bg, s_kerr=1.0e-18)                   # eps = (eps_bg^-1 + s|E|^2 I)^-1
    print("[fe] GATE B -- DC-Kerr (FEM == TMM at eps(E); field-induced phase shift; dn matches pert.):",
          flush=True)
    gate_b = True
    arg_f0 = arg_t0 = None
    for Ez in (0.0, 7e7, 1e8):
        eps_t = np.asarray(kerr.eps({"E": [0.0, 0.0, Ez]}, LAM * 1e-9))
        eps_yy = complex(eps_t[1, 1])
        rf = _fem(eps_t); rt = _tmm(eps_yy)
        dR, dT = abs(rf.R - rt.R), abs(rf.T - rt.T)
        af, at = np.degrees(np.angle(rf.t)), np.degrees(np.angle(rt.t))
        if arg_f0 is None:
            arg_f0, arg_t0 = af, at
        # field-induced transmission phase shift relative to E=0; FEM and TMM must agree
        sh_f = (af - arg_f0 + 180.0) % 360.0 - 180.0
        sh_t = (at - arg_t0 + 180.0) % 360.0 - 180.0
        dshift = abs(sh_f - sh_t)
        # perturbative index shift vs the model: dn = -s_kerr |E|^2 eps_bg^(3/2)/2
        dn_pert = -1.0e-18 * Ez ** 2 * 4.0 ** 1.5 / 2.0
        dn_model = np.sqrt(eps_yy).real - 2.0
        match = (dR < TOL and dT < TOL and dshift < 1.0
                 and abs(dn_model - dn_pert) < 0.1 * abs(dn_pert) + 1e-6)
        gate_b = gate_b and match
        print("[fe]   E={:.1e} V/m: eps_yy={:.4f} dn_model={:+.4f} dn_pert={:+.4f} | dR={:.1e} dT={:.1e} "
              "phase-shift fem={:+.2f} tmm={:+.2f} deg  {}".format(
                  Ez, eps_yy.real, dn_model, dn_pert, dR, dT, sh_f, sh_t, "ok" if match else "FAIL"),
              flush=True)
    print("[fe] GATE B: {}".format("PASS" if gate_b else "FAIL"), flush=True)
    ok = ok and gate_b

    print("[fe] *** FIELD-EFFECT (Franz-Keldysh + DC-Kerr) FEM vs TMM: {} ***".format(
        "PASS" if ok else "FAIL"), flush=True)
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
