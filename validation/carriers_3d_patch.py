"""Validate the lateral GATE-PATCH 3D builder (Stacked3DSpec.gate_patch_frac < 1):
the gate contact covers only a centered patch of the oxide top, the rest is a free
surface. The classical 3D DEVSIM solve must then accumulate electrons UNDER the patch
and leave the gap near bulk -- a laterally-VARYING (non-separable) carrier profile the
2D+symmetrization path cannot capture. Confirms the patterned gmsh mesh builds + solves
+ gives the lateral contrast, and bridges to a laterally-varying eps.
Run:  python -m validation.carriers_3d_patch
"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dynameta.carriers.devsim_3d import Devsim3DEquilibrium, Stacked3DSpec
from dynameta.core.carrier_field import ELECTRON_DENSITY
from dynameta.sweep import BiasPoint
from dynameta.materials import Material, MaterialRegistry, DrudeOptical, M_E
from dynameta.core.alignment import GeometryAlignment, RegionAlignment
from dynameta.core import NM, MaterialEpsMap, assemble_eps
from dynameta.core.lift import IdentityLift

LAT = 60e-9
SEMI = 12e-9


def main():
    spec = Stacked3DSpec(lateral_m=LAT, semi_thk_m=SEMI, oxide_thk_m=8e-9,
                          n_bg_m3=4e26, gate_patch_frac=0.5)     # centered 30nm gate patch
    solver = Devsim3DEquilibrium(spec)
    cf = solver.solve(BiasPoint({"gate": 1.0, "body": 0.0}, "patch+1V"))
    v = cf.regions["semi"].grid_fields[ELECTRON_DENSITY]         # (nx, ny, nz)
    xs = cf.regions["semi"].grid_axes_m["x"]
    nx = xs.size
    n_bg = spec.n_bg_m3
    # gate-side (top of semi) density: center (under patch) vs corner (gap)
    ic = nx // 2
    center = v[ic, ic, -1] / n_bg
    corner = v[0, 0, -1] / n_bg
    print("[t] gate-patch 3D (frac=0.5, +1V): n_gate/n_bg  under-patch={:.3f}  gap-corner={:.3f}".format(
        center, corner), flush=True)

    # bridge -> eps; min Re(eps) under the patch vs the gap
    reg = MaterialRegistry()
    reg.add(Material("ITO", DrudeOptical(eps_inf=4.25, m_opt_kg=0.225 * M_E, gamma_rad_s=1.1e14)))
    align = GeometryAlignment(unit_scale=NM, fixed_eps_regions={},
        region_alignments=[RegionAlignment("semi", "semi", (0.0, LAT, 0.0, LAT, 0.0, SEMI), "z")])
    ef = assemble_eps(cf, align, MaterialEpsMap(reg), IdentityLift(), 1300e-9, mesh_regions=["semi"])["semi"]
    re = np.real(ef.values_zyx)                                  # (nz, ny, nx)
    re_patch = float(re[:, ic, ic].min())
    re_gap = float(re[:, 0, 0].min())
    print("[t] min Re(eps): under-patch={:+.3f}  gap-corner={:+.3f}".format(re_patch, re_gap), flush=True)
    solver.teardown()

    lateral_contrast = center > 1.05 * corner                    # accumulation only under the patch
    eps_contrast = re_patch < re_gap - 0.05
    ok = bool(np.isfinite(center) and lateral_contrast and eps_contrast)
    print("[t] *** 3D GATE-PATCH: builds+solves={} lateral_accumulation={} eps_contrast={} -> {} ***".format(
        bool(np.isfinite(center)), bool(lateral_contrast), bool(eps_contrast), "PASS" if ok else "FAIL"), flush=True)


if __name__ == "__main__":
    main()
