"""Mirror-symmetry domain reduction for the FEM optical solve (Mesh3DSpec.symmetry): a centered,
mirror-symmetric unit cell at NORMAL incidence is meshed on a HALF ('half_x'/'half_y') or 'quarter'
lateral cell with symmetry WALLS (PEC = tangential E = 0 = HCurl Dirichlet; PMC = the natural BC)
replacing the periodic boundary on the reduced axis. The reduction is correctness-neutral: the 0-order
R/T must equal the FULL periodic solve (up to the FEM-vs-FEM mesh-agreement floor), at 1/2 or 1/4 the
DOFs. The wall whose outward normal is PARALLEL to the incident E is PEC, the one PERPENDICULAR is PMC;
at normal incidence pol 'x' -> x-walls PEC, pol 'y' -> y-walls PEC.

GATE A (half-cell == full, both wall types): a c2v centered rectangular pillar (w != h). half_x at pol
        'y' exercises a NATURAL (PMC) x-wall + periodic y; half_y at pol 'y' exercises a Dirichlet (PEC)
        y-wall + periodic x. Both must match the full periodic R/T within the mesh floor.
GATE B (quarter-cell == full, both pols): a c4v centered circular pillar. quarter at pol 'x' and pol
        'y' (PEC on one axis, PMC on the other) must match the full R/T, at ~1/4 the lateral DOFs.
GATE C (the walls are LOAD-BEARING / discrimination): solving the SAME quarter mesh with the symmetry
        flags zeroed (so the solver leaves every lateral wall natural -- the wrong, all-PMC physics)
        must differ GROSSLY from the full solve, while the correct symmetry-walled solve matches. This
        proves the gate is not vacuously passing and that a dropped/mis-assigned wall is caught.

Honest SKIP (exit 0 + banner) when ngsolve is not installed.

Run: python -m validation.symmetry_reduced_fem
"""
import dataclasses
import importlib.util
import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

LAM = 1300e-9
PER = 400e-9
TOL = 5e-3            # FEM-vs-FEM mesh-agreement floor (NOT machine precision: the two meshes differ)


def _design(symmetry, pol, shape_kind):
    from dynameta.geometry import Design, Inclusion, Layer, Stack, UnitCell
    from dynameta.geometry.cross_section import centered_circle, centered_rectangle
    from dynameta.geometry.specs import Mesh3DSpec, OpticalSpec
    from dynameta.materials import ConstantOptical, Material, MaterialRegistry
    reg = MaterialRegistry()
    for nm, e in (("air", 1.0 + 0j), ("glass", complex(1.5 ** 2)), ("pil", complex(2.5 ** 2))):
        reg.add(Material(nm, ConstantOptical(e)))
    cell = UnitCell.square(PER)
    if shape_kind == "circle":
        shape = centered_circle(cell, 120e-9)                 # c4v
    else:
        shape = centered_rectangle(cell, 220e-9, 130e-9)      # c2v (w != h)
    stack = Stack(layers=[Layer("slab", 220e-9, "air", inclusions=[Inclusion(shape=shape, material="pil")])],
                  superstrate_material="air", substrate_material="glass")
    m3 = Mesh3DSpec(pml_thk_m=500e-9, superstrate_buffer_m=1400e-9, substrate_buffer_m=1400e-9,
                    maxh_superstrate_m=60e-9, maxh_substrate_m=60e-9, maxh_background_m=40e-9,
                    maxh_inclusion_m=20e-9, fem_order=2, symmetry=symmetry)
    return Design(name="sym", unit_cell=cell, stack=stack, electrodes=[], materials=reg,
                  optical=OpticalSpec(polarization=pol, incidence_angle_deg=0.0), mesh_3d=m3)


def _solve(d, geo=None):
    from dynameta.core.eps_field import EpsField
    from dynameta.optics.ngsolve_layered import LayeredOpticalBuilder
    from dynameta.optics.solver import solve_fem
    from dynameta.optics.eps_assembler import assemble_eps_cf
    if geo is None:
        geo = LayeredOpticalBuilder(d).build()
    ebr = {r: EpsField(scalar=complex(d.materials.get(geo.material_by_region[r]).eps(LAM)))
           for r in geo.mesh.GetMaterials()}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = solve_fem(geo, LAM, assemble_eps_cf(geo, ebr), d.optical, order=2,
                        n_super=1.0 + 0j, n_sub=1.5 + 0j)
    return res, geo


def _dRT(a, b):
    return max(abs(a.R - b.R), abs((a.T or 0.0) - (b.T or 0.0)))


def main():
    if importlib.util.find_spec("ngsolve") is None:
        print("[sym] *** SKIP: ngsolve not installed -- symmetry-reduced FEM gates not run ***", flush=True)
        return True
    print("[sym] === mirror-symmetry FEM reduction: half/quarter cell == full periodic R/T ===", flush=True)
    ok = True

    # ---- GATE A: c2v rectangle, half cells (PMC-only and PEC-containing) ----
    full_y, gfull = _solve(_design("none", "y", "rect"))
    hx_y, gx = _solve(_design("half_x", "y", "rect"))      # PMC (natural) x-wall + periodic y
    hy_y, gy = _solve(_design("half_y", "y", "rect"))      # PEC (Dirichlet) y-wall + periodic x
    dA = max(_dRT(full_y, hx_y), _dRT(full_y, hy_y))
    g_a = bool(dA < TOL and gx.mesh.nv < gfull.mesh.nv and gy.mesh.nv < gfull.mesh.nv)
    ok = ok and g_a
    print("[sym] GATE A: c2v half-cell==full (half_x PMC R={:.4f} nv={}, half_y PEC R={:.4f} nv={} vs "
          "full R={:.4f} nv={}); worst |dR,dT|={:.1e} -> {}".format(
              hx_y.R, gx.mesh.nv, hy_y.R, gy.mesh.nv, full_y.R, gfull.mesh.nv, dA,
              "PASS" if g_a else "FAIL"), flush=True)

    # ---- GATE B: c4v circle, quarter cell, both polarizations ----
    worst_b, nv_full, nv_q = 0.0, 0, 0
    for pol in ("x", "y"):
        full_p, gf = _solve(_design("none", pol, "circle"))
        q_p, gq = _solve(_design("quarter", pol, "circle"))
        worst_b = max(worst_b, _dRT(full_p, q_p))
        nv_full, nv_q = gf.mesh.nv, gq.mesh.nv
        print("[sym]   pol {}: full R={:.5f} / quarter R={:.5f} (nv {} -> {})".format(
            pol, full_p.R, q_p.R, nv_full, nv_q), flush=True)
        if pol == "y":
            full_circle_y, geo_q_y = full_p, gq          # reuse for GATE C
            q_circle_y = q_p
    g_b = bool(worst_b < TOL and nv_q < nv_full)
    ok = ok and g_b
    print("[sym] GATE B: c4v quarter==full both pols worst |dR,dT|={:.1e}; DOF reduction nv {}->{} "
          "-> {}".format(worst_b, nv_full, nv_q, "PASS" if g_b else "FAIL"), flush=True)

    # ---- GATE C: the symmetry walls are LOAD-BEARING (discrimination) ----
    # Re-solve the SAME quarter mesh with the symmetry flags zeroed: the solver then leaves every lateral
    # wall natural (the wrong, all-PMC physics, = a dropped PEC wall). It MUST diverge from the full
    # solve, while the correctly-walled quarter (GATE B) matched -- so a mis-assigned/dropped wall cannot
    # pass unnoticed.
    geo_nowall = dataclasses.replace(geo_q_y, sym_x=False, sym_y=False)
    bad_y, _ = _solve(_design("quarter", "y", "circle"), geo=geo_nowall)
    d_correct = _dRT(full_circle_y, q_circle_y)           # correct walls (from GATE B)
    d_nowall = _dRT(full_circle_y, bad_y)                 # walls dropped -> wrong
    g_c = bool(d_correct < TOL and d_nowall > 1e-2)
    ok = ok and g_c
    print("[sym] GATE C: walls load-bearing -- correct |d|={:.1e} (<{:.0e}) vs walls-dropped |d|={:.3f} "
          "(>1e-2) -> {}".format(d_correct, TOL, d_nowall, "PASS" if g_c else "FAIL"), flush=True)

    print("[sym] *** SYMMETRY-REDUCED FEM: {} ***".format("PASS" if ok else "FAIL"), flush=True)
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
