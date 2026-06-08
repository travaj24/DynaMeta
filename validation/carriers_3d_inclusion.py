"""Validate ARBITRARY LATERAL material INCLUSIONS in the 3D DEVSIM carrier mesh (the roadmap-remaining
3D piece, beyond the centered gate patch). A high-permittivity dielectric PILLAR is embedded in the
(buried) lower gate-oxide layer; the builder OCC-fragments it into its own region with adjacency-found
interfaces. Because a higher-eps path drops LESS of the gate voltage, the gate field couples MORE
strongly through the pillar -> the semiconductor accumulates MORE directly under the pillar than in the
surrounding lower-eps oxide. This is a genuinely NON-SEPARABLE lateral carrier topology (a different
MATERIAL laterally, not just a gate-footprint patch).

Stack: semi(ITO) | oxide=Al2O3(eps 9, lower, buried) | diel1=HfO2(eps 18, upper) ; full gate on top.
Inclusion: an HfO2(eps 18) pillar (50% of the cell, centered) inside the lower Al2O3 oxide.

GATE A: the mesh carries the 4 regions (semi, oxide, diel1, the pillar) + converges + accumulates.
GATE B: LATERAL variation -- the gate-side accumulation under the pillar (cell centre) EXCEEDS that in
        the surrounding low-eps oxide (cell edge) by a clear margin (the inclusion modulates the field).

Run: python -m validation.carriers_3d_inclusion
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dynameta.carriers.devsim_3d import Stacked3DSpec, Inclusion3D, Devsim3DEquilibrium
from dynameta.core.carrier_field import ELECTRON_DENSITY
from dynameta.sweep import BiasPoint

N_BG = 4e26
VG = 1.0


def main():
    print("[t] === 3D carriers: lateral material INCLUSION (high-eps pillar in the gate oxide) ===", flush=True)
    spec = Stacked3DSpec(
        semi_material="ITO", lateral_m=16e-9, semi_thk_m=12e-9, n_bg_m3=N_BG, eps_semi=9.5,
        dos_mass_kg=0.35 * 9.1093837015e-31, grid_n=(20, 20, 25), mesh_min_nm=0.5, mesh_max_nm=2.0,
        oxide_material="Al2O3", oxide_thk_m=4e-9, eps_oxide=9.0,        # lower (buried) oxide, low eps
        extra_dielectrics=[("HfO2", 4e-9, 18.0)],                       # upper oxide (gate sits on it)
        inclusions=[Inclusion3D(name="pillar", material="HfO2", role="dielectric", eps=18.0,
                                 in_layer="oxide", x_frac=0.5, y_frac=0.5)])
    import devsim as ds
    b = Devsim3DEquilibrium(spec, mesh_name="inc3_m", device_name="inc3_d")
    cf = b.solve(BiasPoint({"gate": VG, "body": 0.0}, "g"))
    regions = list(ds.get_region_list(device=b.device))
    g = np.asarray(cf.regions["semi"].grid_fields[ELECTRON_DENSITY], float)   # (nx,ny,nz)
    nx, ny, _ = g.shape
    n_top = g[:, :, -1]                                            # gate-side accumulation plane
    n_center = float(n_top[nx // 2, ny // 2])                      # under the pillar
    n_edge = float(n_top[1, 1])                                    # in the surrounding low-eps oxide
    b.teardown()

    print("[t] regions = {} (expect semi, oxide, diel1, pillar)".format(sorted(regions)), flush=True)
    print("[t] gate-side n: under-pillar(centre)={:.3e}  surrounding(edge)={:.3e}  ratio={:.3f}".format(
        n_center, n_edge, n_center / max(n_edge, 1.0)), flush=True)
    g_a = bool(("pillar" in regions) and len(regions) == 4 and n_center > N_BG and n_edge > 0.0)
    g_b = bool(n_center > 1.05 * n_edge)                           # clear lateral enhancement under the pillar
    ok = g_a and g_b
    print("[t] GATE A (4 regions incl pillar + converges + accumulates): {}".format("PASS" if g_a else "FAIL"),
          flush=True)
    print("[t] GATE B (more accumulation under the high-eps pillar -- lateral/non-separable): {}".format(
        "PASS" if g_b else "FAIL"), flush=True)
    print("[t] *** 3D CARRIER LATERAL INCLUSION: {} ***".format("PASS" if ok else "FAIL"), flush=True)
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
