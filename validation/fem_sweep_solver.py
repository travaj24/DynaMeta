"""Sweep-aware FEM optical solver (pass-2 audit perf finding): make_fem_optical_solver builds the
wavelength-INDEPENDENT HCurl FESpace ONCE at normal incidence and reuses it across the wavelength
sweep, instead of the per-call default rebuilding it every wavelength. The reuse must be exactly
correctness-neutral -- R/T BYTE-IDENTICAL to a per-call solve_fem (only the redundant space build is
avoided; the eps mass, RHS, and factorization still rebuild per wavelength).

GATE A (normal incidence: sweep == per-call, byte-identical): solve_sweep over several wavelengths
        reproduces R/T/r/phase from the per-call solver to < 1e-12 (the FESpace reuse changes nothing).
GATE B (oblique fallback: sweep == per-call): at 30 deg the Bloch phases are k0-dependent, so the
        sweep MUST fall back to a per-wavelength FESpace build -- and still match the per-call solver
        byte-identically (the fallback path is not broken).
GATE C (oblique refuses an injected reuse-space): a normal-incidence FESpace passed as _reuse_fes into
        an OBLIQUE solve_fem must be IGNORED (the oblique branch precedes the reuse branch), giving the
        byte-identical fresh-build oblique result -- the safety property the sweep fallback relies on.

Honest SKIP (exit 0 + banner) when ngsolve is not installed.

Run: python -m validation.fem_sweep_solver
"""
import importlib.util
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

LAM_NM = 1300.0
N_SLAB = 2.0
D_SLAB_NM = 250.0


def _design(theta_deg):
    from dynameta.geometry import Design, Layer, Stack, UnitCell
    from dynameta.geometry.specs import Mesh3DSpec, OpticalSpec
    from dynameta.materials import ConstantOptical, Material, MaterialRegistry
    reg = MaterialRegistry()
    reg.add(Material("air", ConstantOptical(1.0 + 0j)))
    reg.add(Material("slab", ConstantOptical(complex(N_SLAB ** 2, 0.0))))
    m3 = Mesh3DSpec(pml_thk_m=500e-9, superstrate_buffer_m=1400e-9, substrate_buffer_m=1400e-9,
                    maxh_superstrate_m=45e-9, maxh_substrate_m=45e-9, maxh_background_m=22e-9,
                    fem_order=2)
    return Design(name="fem_sweep", unit_cell=UnitCell.square(220e-9),
                  stack=Stack(layers=[Layer("slab", D_SLAB_NM * 1e-9, "slab")],
                              superstrate_material="air", substrate_material="air"),
                  electrodes=[], materials=reg,
                  optical=OpticalSpec(polarization="y", incidence_angle_deg=theta_deg,
                                      linear_solver="umfpack"), mesh_3d=m3)


def _worst(a, b):
    return max(abs(a.R - b.R), abs((a.T or 0.0) - (b.T or 0.0)), abs(a.r - b.r),
               abs(a.phase_deg - b.phase_deg))


def main():
    if importlib.util.find_spec("ngsolve") is None:
        print("[fsw] *** SKIP: ngsolve not installed -- FEM sweep solver gates not run ***", flush=True)
        return True
    from dynameta.core.eps_field import EpsField
    from dynameta.optics.ngsolve_layered import LayeredOpticalBuilder
    from dynameta.optics.solver import make_fem_optical_solver

    print("[fsw] === sweep-aware FEM solver: FESpace reuse is byte-identical to per-call ===", flush=True)
    ok = True
    solver = make_fem_optical_solver()
    lams = [1.20e-6, 1.30e-6, 1.45e-6]
    _eps = {"air": complex(1.0, 0.0), "slab": complex(N_SLAB ** 2, 0.0)}

    def _eps_by_region(geo):
        # map EVERY mesh region (air buffers + slab + PMLs) to its material eps (constant in lambda)
        return {rg: EpsField(scalar=_eps[geo.material_by_region.get(rg, "air")])
                for rg in geo.mesh.GetMaterials()}

    # ---- GATE A: normal incidence -- sweep (reused FESpace) == per-call (fresh FESpace) ----
    d = _design(0.0)
    geo = LayeredOpticalBuilder(d).build()
    ebr = _eps_by_region(geo)

    def _assemble(_lam):
        return ebr                                           # constant materials: same dict every lambda

    swept = solver.solve_sweep(d, geo, _assemble, lams, 1.0 + 0j, 1.0 + 0j)
    worst_a = 0.0
    for lam, sw in zip(lams, swept):
        per = solver(d, geo, ebr, lam, 1.0 + 0j, 1.0 + 0j)   # per-call (fresh FESpace each time)
        worst_a = max(worst_a, _worst(sw, per))
    distinct = abs(swept[0].R - swept[-1].R) > 1e-4          # the sweep is a real spectrum, not a constant
    g_a = bool(worst_a < 1e-12 and distinct)
    ok = ok and g_a
    print("[fsw] GATE A: normal-incidence sweep==per-call worst |d|={:.1e} (spectrum varies {}) -> {}"
          .format(worst_a, distinct, "PASS" if g_a else "FAIL"), flush=True)

    # ---- GATE B: oblique 30 deg -- sweep falls back to per-wavelength build, still == per-call ----
    d2 = _design(30.0)
    geo2 = LayeredOpticalBuilder(d2).build()
    ebr2 = _eps_by_region(geo2)
    swept2 = solver.solve_sweep(d2, geo2, lambda _lam: ebr2, lams, 1.0 + 0j, 1.0 + 0j)
    worst_b = 0.0
    for lam, sw in zip(lams, swept2):
        per = solver(d2, geo2, ebr2, lam, 1.0 + 0j, 1.0 + 0j)
        worst_b = max(worst_b, _worst(sw, per))
    g_b = bool(worst_b < 1e-12)
    ok = ok and g_b
    print("[fsw] GATE B: oblique-30deg sweep (fallback) ==per-call worst |d|={:.1e} -> {}".format(
        worst_b, "PASS" if g_b else "FAIL"), flush=True)

    # ---- GATE C: at oblique an INJECTED (wrong) normal-incidence FESpace is REFUSED ----
    # The solve_fem oblique branch PRECEDES the _reuse_fes branch, so a normal-incidence space passed as
    # _reuse_fes at oblique MUST be ignored -- the result is byte-identical to the freshly-built oblique
    # fallback (swept2[0]). This is the safety property the sweep relies on: a reused space (only ever
    # built at normal incidence) can never corrupt an oblique solve. (Without it, reuse could silently
    # apply the wrong Bloch phases.)
    import ngsolve as ng
    from dynameta.optics.eps_assembler import assemble_eps_cf
    from dynameta.optics.solver import solve_fem
    od = int(d2.mesh_3d.fem_order)
    eps_cf2 = assemble_eps_cf(geo2, ebr2)
    fes_norm = ng.Periodic(ng.HCurl(geo2.mesh, order=od, complex=True, dirichlet=""))   # NORMAL-inc space
    inj = solve_fem(geo2, lams[0], eps_cf2, d2.optical, order=od, n_super=1.0 + 0j, n_sub=1.0 + 0j,
                    _reuse_fes=fes_norm)                         # oblique -> this space MUST be ignored
    worst_c = _worst(inj, swept2[0])                             # == the fresh-build oblique fallback
    g_c = bool(worst_c < 1e-12)
    ok = ok and g_c
    print("[fsw] GATE C: oblique REFUSES an injected normal-incidence FESpace (==fresh-build worst |d|="
          "{:.1e}) -> {}".format(worst_c, "PASS" if g_c else "FAIL"), flush=True)

    print("[fsw] *** FEM SWEEP SOLVER (FESpace reuse byte-identical): {} ***".format(
        "PASS" if ok else "FAIL"), flush=True)
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
