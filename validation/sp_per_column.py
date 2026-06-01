"""Validate the PER-COLUMN Schrodinger-Poisson CarrierSolver for a laterally-VARYING
device: a central patch (surface potential +0.4V) over an otherwise ungated ITO layer
(0V in the gap). A 1D SP solve runs per lateral column (cached by psi_s value, so a
~equipotential patch costs only ~2 solves), giving a laterally-varying quantum profile:
accumulation (sub-ENZ) UNDER the patch, flat bulk in the gap. Confirms the CarrierField
is laterally non-uniform and bridges to a laterally-varying eps.
Run:  python -m validation.sp_per_column
"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dynameta.carriers.sp_carrier import SchrodingerPoissonCarrier
from dynameta.sweep import BiasPoint
from dynameta.materials import Material, MaterialRegistry, DrudeOptical, M_E
from dynameta.core.alignment import GeometryAlignment, RegionAlignment
from dynameta.core import NM, MaterialEpsMap, assemble_eps
from dynameta.core.carrier_field import ELECTRON_DENSITY
from dynameta.core.lift import IdentityLift

N_BG = 4e26
LATERAL = 200e-9
T_SEMI = 12e-9


def main():
    # patch covers the central half; +Vg under it, 0 in the gap (a step psi_s)
    def psi_xy(x, y, vg):
        in_patch = abs(x - LATERAL / 2) < LATERAL / 4 and abs(y - LATERAL / 2) < LATERAL / 4
        return vg if in_patch else 0.0

    solver = SchrodingerPoissonCarrier(semi_thk_m=T_SEMI, n_bg_m3=N_BG, lateral_m=LATERAL,
                                        n_lateral=8, surface_potential_xy=psi_xy)
    cf = solver.solve(BiasPoint({"gate": 0.4, "body": 0.0}, "+0.4V"))
    v = cf.regions["semi"].grid_fields[ELECTRON_DENSITY]   # (nx, ny, nz)
    xs = cf.regions["semi"].grid_axes_m["x"]
    nx = xs.size
    ic, ie = nx // 2, 0                                    # center (under patch) vs corner (gap)
    peak_center = v[ic, ic, nx // 2:].max() / N_BG if False else v[ic, ic, :].max() / N_BG
    peak_corner = v[ie, ie, :].max() / N_BG
    print("[t] per-column SP: distinct columns solved = {}  (psi range {} V)".format(
        cf.extras.get("n_distinct_columns"), cf.extras.get("surface_potential_range_V")), flush=True)
    print("[t] peak n/n_bg: under-patch={:.3f}  in-gap={:.3f}".format(peak_center, peak_corner), flush=True)

    # bridge -> eps (ITO Drude); compare min Re(eps) under the patch vs in the gap
    reg = MaterialRegistry()
    reg.add(Material("ITO", DrudeOptical(eps_inf=4.25, m_opt_kg=0.225 * M_E, gamma_rad_s=1.1e14)))
    align = GeometryAlignment(unit_scale=NM, fixed_eps_regions={},
        region_alignments=[RegionAlignment("semi", "semi", (0.0, LATERAL, 0.0, LATERAL, 0.0, T_SEMI), "z")])
    ef = assemble_eps(cf, align, MaterialEpsMap(reg), IdentityLift(), 1300e-9, mesh_regions=["semi"])["semi"]
    re = np.real(ef.values_zyx)                            # (nz, ny, nx)
    re_patch = float(re[:, nx // 2, nx // 2].min())        # under patch, over z
    re_gap = float(re[:, 0, 0].min())
    print("[t] min Re(eps): under-patch={:+.3f}  in-gap={:+.3f}".format(re_patch, re_gap), flush=True)

    lateral_varying = peak_center > 1.1 * peak_corner       # accumulation only under the patch
    few_solves = (cf.extras.get("n_distinct_columns") or 99) <= 3
    eps_varies = re_patch < re_gap - 0.05                    # ENZ deepened under the patch only
    ok = lateral_varying and few_solves and eps_varies
    print("[t] *** PER-COLUMN SP: lateral_varying={} cached_solves={} eps_lateral_contrast={} -> {} ***".format(
        bool(lateral_varying), bool(few_solves), bool(eps_varies), "PASS" if ok else "FAIL"), flush=True)


if __name__ == "__main__":
    main()
