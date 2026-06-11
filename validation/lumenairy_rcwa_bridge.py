"""Lumenairy RCWA backend bridge (roadmap v0.5 A1) vs the in-repo oracles.

GATE A (unstructured == TMM): a 3-layer lossy stack at normal incidence and 30 deg s/p --
        R/T/phase match tmm_reference to < 1e-9 (both are exact for unstructured stacks;
        Lumenairy itself is Airy-validated to ~1e-16).
GATE B (graded slab chain): a graded ENZ-like eps(z) as a gridded EpsField -> the bridge's
        slice path vs the graded-TMM staircase on the SAME slices -- < 1e-9 (the DEVSIM
        n(z) -> RCWA chain, geometry-side).
GATE C (patterned cell vs FEM + AXIS PIN): an x-y ASYMMETRIC lossless pillar
        (Rectangle 150 x 80 nm, n = 2) in air on glass -- bridge vs the NGSolve FEM for
        x-pol AND y-pol: |dR| < 2e-2 per polarization, the FEM resolves the asymmetry
        (|R_x - R_y| > 2.5e-3, 2x the measured per-pol residual), and the bridge's
        R_x - R_y has the SAME SIGN -- the gate that
        catches a silently transposed (90-degree-rotated) eps_cell, which every
        square-symmetric test would miss.
GATE D (sweep path): solve_sweep == the per-wavelength path identically on a dispersive
        (Drude ITO) stack, and each wavelength matches TMM < 1e-9 (end media re-derived per
        wavelength -- no band-centre freeze).
GATE E (uniform tensor): a planar-LC diagonal tensor diag(n_e^2, n_o^2, n_o^2) -- x-pol
        sees n_e, y-pol sees n_o; each matches the scalar TMM at that index < 1e-6.
GATE F (per-layer absorption): absorption=True fills per_region_absorption keyed by DESIGN
        layer name -- a lossless layer takes ~0, layers close on A_independent == A ==
        1 - R - T (and A == TMM), and a graded layer's slabs aggregate into ONE key.

Honest SKIP (exit 0 + banner) when lumenairy is not importable.

Run: python -m validation.lumenairy_rcwa_bridge
"""
import importlib.util
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dynameta.core.eps_field import EpsField
from dynameta.geometry import Design, Inclusion, Layer, Stack, UnitCell
from dynameta.geometry.cross_section import Rectangle
from dynameta.geometry.specs import Mesh3DSpec, OpticalSpec
from dynameta.materials import ConstantOptical, Material, MaterialRegistry

LAM = 1.31e-6
PER = 400e-9


def _registry():
    reg = MaterialRegistry()
    reg.add(Material("air", ConstantOptical(1.0 + 0j)))
    reg.add(Material("glass", ConstantOptical(complex(1.5 ** 2))))
    reg.add(Material("hi", ConstantOptical(complex(4.0, 0.3))))
    reg.add(Material("lo", ConstantOptical(complex(2.1, 0.0))))
    reg.add(Material("pillar", ConstantOptical(complex(4.0, 0.0))))
    return reg


def _design(layers, *, pol="y", theta=0.0, sub="glass", mesh=False):
    reg = _registry()
    kw = {}
    if mesh:
        kw["mesh_3d"] = Mesh3DSpec(pml_thk_m=500e-9, superstrate_buffer_m=1400e-9,
                                   substrate_buffer_m=1400e-9, maxh_superstrate_m=45e-9,
                                   maxh_substrate_m=45e-9, maxh_background_m=22e-9,
                                   fem_order=2)
    return Design(name="brg", unit_cell=UnitCell.square(PER),
                  stack=Stack(layers=layers, superstrate_material="air",
                              substrate_material=sub),
                  electrodes=[], materials=reg,
                  optical=OpticalSpec(polarization=pol, incidence_angle_deg=theta), **kw)


def main():
    if importlib.util.find_spec("lumenairy") is None:
        print("[lrb] *** SKIP: lumenairy not installed -- bridge gates not run ***",
              flush=True)
        return True
    from dynameta.optics.lumenairy_bridge import make_lumenairy_rcwa_solver
    from dynameta.optics.tmm_reference import make_layered_tmm_solver

    print("[lrb] === Lumenairy RCWA bridge vs TMM / FEM / tensor oracles ===", flush=True)
    ok = True
    tmm = make_layered_tmm_solver()

    # ---- GATE A: unstructured lossy stack, normal + oblique s/p ----
    lays = [Layer("a", 120e-9, "hi"), Layer("b", 200e-9, "lo")]
    worst = 0.0
    for pol, th in (("y", 0.0), ("y", 30.0), ("p", 30.0)):
        d = _design(lays, pol=pol, theta=th)
        r_t = tmm(d, None, {}, LAM, 1.0 + 0j, 1.5 + 0j)
        r_r = make_lumenairy_rcwa_solver(n_orders=3)(d, None, {}, LAM, 1.0 + 0j, 1.5 + 0j)
        worst = max(worst, abs(r_r.R - r_t.R), abs(r_r.T - r_t.T),
                    abs(r_r.r - r_t.r), abs(r_r.t - r_t.t))
    g_a = bool(worst < 1e-9)
    ok = ok and g_a
    print("[lrb] GATE A: unstructured vs TMM (normal + 30deg s/p): worst |d| = {:.2e} -> {}"
          .format(worst, "PASS" if g_a else "FAIL"), flush=True)

    # ---- GATE B: graded eps(z) chain on identical slices ----
    from dynameta.core.layered import LayeredStack, slice_eps_field
    from dynameta.optics.tmm_reference import TmmLayeredSolver
    nz = 17
    z_nm = np.linspace(0.0, 180.0, nz)                   # axes in nm (solver units)
    eps_z = (2.0 + 1.5 * np.sin(np.pi * z_nm / 180.0) ** 2
             + 0.2j * (z_nm / 180.0)).astype(complex)
    ef = EpsField(values_zyx=eps_z.reshape(nz, 1, 1), z_axis_u=z_nm,
                  x_axis_u=np.array([0.0]), y_axis_u=np.array([0.0]))
    slabs = slice_eps_field(ef, 1e-9)
    stk = LayeredStack(1.0 + 0j, 1.5 + 0j, slabs, period_x_m=PER, period_y_m=PER)
    opt = OpticalSpec(polarization="y", incidence_angle_deg=0.0)
    r_t = TmmLayeredSolver().solve(stk, LAM, opt)
    from dynameta.optics.lumenairy_bridge import LumenairyStackSolver
    r_r = LumenairyStackSolver(n_orders=3).solve(stk, LAM, opt)
    dB = max(abs(r_r.R - r_t.R), abs(r_r.T - r_t.T))
    g_b = bool(dB < 1e-9)
    ok = ok and g_b
    print("[lrb] GATE B: graded-slab staircase vs graded-TMM: |d| = {:.2e} -> {}".format(
        dB, "PASS" if g_b else "FAIL"), flush=True)

    # ---- GATE C: asymmetric patterned pillar vs FEM (the axis pin) ----
    # ngsolve is an OPTIONAL extra (dynameta[solvers]); lumenairy being a required core
    # dependency means this validation now RUNS in minimal environments, so the FEM-oracle
    # gate must skip honestly there instead of crashing on the netgen import.
    if importlib.util.find_spec("ngsolve") is None:
        print("[lrb] GATE C: SKIP (ngsolve not installed -- FEM oracle unavailable; "
              "gates A/B/D/E/F still gate the bridge)", flush=True)
    else:
        from dynameta.optics.ngsolve_layered import LayeredOpticalBuilder
        from dynameta.optics.solver import solve_fem
        pil = Inclusion(shape=Rectangle(PER / 2.0, PER / 2.0, 150e-9, 80e-9),
                        material="pillar")
        p_lays = [Layer("slab", 200e-9, "air", inclusions=[pil])]
        R_fem, R_rc = {}, {}
        for pol in ("x", "y"):
            d = _design(p_lays, pol=pol, mesh=True)
            geo = LayeredOpticalBuilder(d).build()
            eps_vals = {r: complex(d.materials.get(geo.material_by_region[r]).eps(LAM))
                        for r in geo.mesh.GetMaterials()}
            eps_cf = geo.mesh.MaterialCF(eps_vals, default=1.0)
            rf = solve_fem(geo, LAM, eps_cf, d.optical, order=2, n_super=1.0 + 0j,
                           n_sub=1.5 + 0j)
            R_fem[pol] = rf.R
            rr = make_lumenairy_rcwa_solver(n_orders=8)(d, None, {}, LAM,
                                                        1.0 + 0j, 1.5 + 0j)
            R_rc[pol] = rr.R
            print("[lrb]   pol {}: R fem {:.4f} / rcwa {:.4f}".format(pol, R_fem[pol],
                                                                      R_rc[pol]),
                  flush=True)
        split_fem = R_fem["x"] - R_fem["y"]
        split_rc = R_rc["x"] - R_rc["y"]
        # discriminability: the FEM split must exceed BOTH backends' agreement scale
        # (~1e-3 measured), so the sign check is meaningful; 2.5e-3 = 2x the worst
        # per-pol residual
        g_c = bool(abs(R_rc["x"] - R_fem["x"]) < 2e-2
                   and abs(R_rc["y"] - R_fem["y"]) < 2e-2
                   and abs(split_fem) > 2.5e-3
                   and np.sign(split_rc) == np.sign(split_fem))
        ok = ok and g_c
        print("[lrb] GATE C: asymmetric pillar vs FEM (dRx {:.1e}, dRy {:.1e}; splits "
              "fem {:+.4f} / rcwa {:+.4f} same-sign) -> {}".format(
                  abs(R_rc["x"] - R_fem["x"]), abs(R_rc["y"] - R_fem["y"]), split_fem,
                  split_rc, "PASS" if g_c else "FAIL"), flush=True)

    # ---- GATE D: dispersive sweep path == per-wavelength + TMM each wavelength ----
    # dispersion enters the PRODUCTION way: through the per-wavelength assemble_at closure
    # (a Drude ITO eps(lambda) as a uniform-scalar EpsField override on the film layer)
    d = _design([Layer("film", 150e-9, "hi")], pol="y")

    def _eps_ito(lam):
        w = 2.0 * np.pi * 2.99792458e8 / lam
        return 3.9 - (2.9e15) ** 2 / (w ** 2 + 1j * w * 1.8e14)

    def _assemble(lam):
        return {"film": EpsField(scalar=complex(_eps_ito(lam)))}

    lams = [1.20e-6, 1.27e-6, 1.31e-6, 1.40e-6, 1.55e-6]
    solver = make_lumenairy_rcwa_solver(n_orders=3)
    rows_sweep = solver.solve_sweep(d, None, _assemble, lams, 1.0 + 0j, 1.5 + 0j)
    worstD = 0.0
    for lam, rs in zip(lams, rows_sweep):
        rp = solver(d, None, _assemble(lam), lam, 1.0 + 0j, 1.5 + 0j)
        rt = tmm(d, None, _assemble(lam), lam, 1.0 + 0j, 1.5 + 0j)
        worstD = max(worstD, abs(rs.R - rp.R), abs(rs.T - rp.T),
                     abs(rs.R - rt.R), abs(rs.T - rt.T))
    g_d = bool(worstD < 1e-9)
    ok = ok and g_d
    print("[lrb] GATE D: dispersive sweep == per-wavelength == TMM over 5 wavelengths: "
          "worst |d| = {:.2e} -> {}".format(worstD, "PASS" if g_d else "FAIL"), flush=True)

    # ---- GATE E: uniform in-plane tensor (planar LC, optic axis x) ----
    n_o, n_e = 1.52, 1.74
    eps_t = np.diag([n_e ** 2, n_o ** 2, n_o ** 2]).astype(complex)
    from dynameta.core.layered import LayeredSlab
    worstE = 0.0
    for pol, n_idx in (("x", n_e), ("y", n_o)):
        stk_t = LayeredStack(1.0 + 0j, 1.5 + 0j,
                             [LayeredSlab(2.0e-6, eps_tensor_cell=np.broadcast_to(
                                 eps_t, (1, 1, 3, 3)).copy())],
                             period_x_m=PER, period_y_m=PER)
        opt = OpticalSpec(polarization=pol, incidence_angle_deg=0.0)
        r_r = LumenairyStackSolver(n_orders=3).solve(stk_t, LAM, opt)
        stk_s = LayeredStack(1.0 + 0j, 1.5 + 0j, [LayeredSlab(2.0e-6,
                                                              eps=complex(n_idx ** 2))])
        r_t = TmmLayeredSolver().solve(stk_s, LAM, opt)
        worstE = max(worstE, abs(r_r.R - r_t.R), abs(r_r.T - r_t.T))
    g_e = bool(worstE < 1e-6)
    ok = ok and g_e
    print("[lrb] GATE E: uniform LC tensor (x-pol -> n_e, y-pol -> n_o) vs scalar TMM: "
          "worst |d| = {:.2e} -> {}".format(worstE, "PASS" if g_e else "FAIL"), flush=True)

    # ---- GATE F: per-layer absorption attribution (absorption=True mapping) ----
    # lossy + lossless 2-layer stack: per_region_absorption must be keyed by the DESIGN
    # layer names, the lossless layer must take ~0, the layers must close on
    # A_independent == A == 1 - R - T (Lumenairy's attribution is energy-conserving by
    # construction), and A must match TMM. Then a GRADED layer (gridded EpsField sliced
    # into many slabs sharing one design name) must aggregate into ONE key, still closing.
    solver_a = make_lumenairy_rcwa_solver(n_orders=2, absorption=True)
    lays = [Layer("lossy", 120e-9, "hi"), Layer("clear", 200e-9, "lo")]
    d = _design(lays, pol="y")
    r_a = solver_a(d, None, {}, LAM, 1.0 + 0j, 1.5 + 0j)
    r_t = tmm(d, None, {}, LAM, 1.0 + 0j, 1.5 + 0j)
    pra = r_a.per_region_absorption or {}
    errF = max(abs(r_a.A - r_t.A),
               abs(r_a.A_independent - r_a.A) if r_a.A_independent is not None else 1.0,
               abs(sum(pra.values()) - r_a.A) if pra else 1.0,
               abs(pra.get("clear", 1.0)))
    keysF = set(pra) == {"lossy", "clear"}
    nz = 40
    z = np.linspace(0.0, 300.0, nz)
    eps_z = 2.0 + 1.5 * (z / z[-1]) + 0.2j * (z / z[-1])  # graded lossy profile
    ef = EpsField(values_zyx=eps_z.reshape(nz, 1, 1), z_axis_u=z,
                  x_axis_u=np.array([0.0]), y_axis_u=np.array([0.0]))
    d_g = _design([Layer("graded", 300e-9, "lo")], pol="y")
    r_g = solver_a(d_g, None, {"graded": ef}, LAM, 1.0 + 0j, 1.5 + 0j)
    pg = r_g.per_region_absorption or {}
    errG = max(abs(sum(pg.values()) - r_g.A) if pg else 1.0,
               abs(r_g.A_independent - r_g.A) if r_g.A_independent is not None else 1.0)
    keysG = set(pg) == {"graded"}
    g_f = bool(keysF and keysG and errF < 1e-9 and errG < 1e-9)
    ok = ok and g_f
    print("[lrb] GATE F: per-layer absorption (keys {}, lossless ~0, closure {:.1e}; "
          "graded slabs -> one key, closure {:.1e}) -> {}".format(
              sorted(pra), errF, errG, "PASS" if g_f else "FAIL"), flush=True)

    print("[lrb] *** LUMENAIRY RCWA BRIDGE: {} ***".format("PASS" if ok else "FAIL"),
          flush=True)
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
