"""Validate the 3D DEVSIM carrier builder on a MULTI-DIELECTRIC gate stack (the roadmap-remaining
3D piece). The stacked builder used to collapse a multi-dielectric gate stack to the single nearest
oxide; it now meshes ALL gate-side dielectric layers as DISTINCT regions (semi | oxide | diel1 | ...
| gate on top), so the gate voltage division is the exact SERIES capacitance.

ORACLE (series-capacitance equivalence): a two-dielectric gate HfO2(t1,eps1) + Al2O3(t2,eps2) must
accumulate the SAME as a SINGLE effective oxide with t_eff/eps_eff = t1/eps1 + t2/eps2 (identical
series capacitance => identical gate coupling => identical accumulation). If the multi-dielectric
stack is meshed/solved correctly the two n_top match to mesh-resolution; collapsing to just HfO2(t1)
(the OLD behavior) would over-accumulate (it drops the Al2O3 series term).

GATE A: the multi-dielectric 3D solve builds 3 regions (semi + 2 dielectrics) and converges +
        accumulates (n_top > n_bg at +Vg).
GATE B: n_top(HfO2+Al2O3) == n_top(single effective oxide, same series C) to < 3%, AND is clearly
        BELOW n_top(HfO2-only) -- proving the second dielectric's series term is actually meshed.

Run: python -m validation.carriers_3d_multidielectric
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dynameta.carriers.devsim_3d import Stacked3DSpec, Devsim3DEquilibrium
from dynameta.core.carrier_field import ELECTRON_DENSITY
from dynameta.sweep import BiasPoint

N_BG = 4e26
T1, E1 = 8e-9, 18.0            # HfO2
T2, E2 = 4e-9, 9.0            # Al2O3
T_EFF = E1 * (T1 / E1 + T2 / E2)   # single oxide (eps=E1) with the SAME series capacitance
VG = 1.0


def _spec(**kw):
    base = dict(semi_material="ITO", lateral_m=12e-9, semi_thk_m=12e-9, n_bg_m3=N_BG,
                eps_semi=9.5, dos_mass_kg=0.35 * 9.1093837015e-31, grid_n=(12, 12, 25),
                mesh_min_nm=0.5, mesh_max_nm=3.0)
    base.update(kw)
    return Stacked3DSpec(**base)


def _n_top(spec, mesh):
    import devsim as ds
    b = Devsim3DEquilibrium(spec, mesh_name=mesh + "_m", device_name=mesh + "_d")
    cf = b.solve(BiasPoint({"gate": VG, "body": 0.0}, "g"))
    g = np.asarray(cf.regions["semi"].grid_fields[ELECTRON_DENSITY], float)  # (nx,ny,nz)
    n_top = float(g[:, :, -1].mean())                                         # gate-side accumulation
    nreg = len(ds.get_region_list(device=b.device))
    b.teardown()
    return n_top, nreg


def main():
    print("[t] === 3D carriers: multi-dielectric gate stack (series-capacitance oracle) ===", flush=True)
    print("[t] HfO2({:.0f}nm,eps{:.0f}) + Al2O3({:.0f}nm,eps{:.0f})  vs  single HfO2({:.1f}nm,eps{:.0f})"
          " (same series C)".format(T1 * 1e9, E1, T2 * 1e9, E2, T_EFF * 1e9, E1), flush=True)

    n_multi, nreg = _n_top(_spec(oxide_material="HfO2", oxide_thk_m=T1, eps_oxide=E1,
                                 extra_dielectrics=[("Al2O3", T2, E2)]), "md3")
    n_eff, _ = _n_top(_spec(oxide_material="HfO2", oxide_thk_m=T_EFF, eps_oxide=E1), "ef3")
    n_hfo2only, _ = _n_top(_spec(oxide_material="HfO2", oxide_thk_m=T1, eps_oxide=E1), "hf3")

    rel = abs(n_multi - n_eff) / n_eff
    print("[t] regions in multi-dielectric device = {} (expect 3: semi+oxide+diel1)".format(nreg), flush=True)
    print("[t] n_top: multi-dielectric={:.4e}  effective-single={:.4e}  rel-diff={:.2e}".format(
        n_multi, n_eff, rel), flush=True)
    print("[t] n_top: HfO2-only (drops Al2O3 series term)={:.4e}  -> multi accumulates {:.1%} of it".format(
        n_hfo2only, n_multi / n_hfo2only), flush=True)

    g_a = bool(nreg == 3 and n_multi > N_BG)
    g_b = bool(rel < 0.03 and n_multi < n_hfo2only * 0.999)
    ok = g_a and g_b
    print("[t] GATE A (3 regions meshed + accumulates): {}".format("PASS" if g_a else "FAIL"), flush=True)
    print("[t] GATE B (== effective series-C oxide <3%, and < HfO2-only): {}".format(
        "PASS" if g_b else "FAIL"), flush=True)
    print("[t] *** 3D MULTI-DIELECTRIC CARRIERS: {} ***".format("PASS" if ok else "FAIL"), flush=True)
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
