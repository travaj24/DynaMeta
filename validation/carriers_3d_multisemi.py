"""Validate the MULTI-SEMICONDUCTOR (heterostack) relaxation of the native 3D gated-cap builder
(Stacked3DSpec.extra_semiconductors + from_design's contiguous gate-ward run). The builder previously
modeled a SINGLE gated semiconductor; it now meshes a stack of semiconductor layers as DISTINCT
equilibrium regions with Potential-continuity interfaces, emitting the gate-adjacent (accumulation) one.

GATE A (homojunction-split reduction, SOLVER): a single semiconductor of thickness T split into two
        stacked sub-layers (T/2 primary + T/2 extra) of the IDENTICAL material/n_bg must reproduce the
        single-layer equilibrium accumulation profile -- the internal semi/semi interface continuity is
        a no-op for a homojunction. The concatenated 2-region electron profile matches the single-region
        profile to mesh resolution (< 5%).
GATE B (from_design heterostack extraction, NO solve): a Design with TWO stacked semiconductor layers
        under the gate yields a spec with the body-most as the primary 'semi', the gate-ward one as an
        extra_semiconductor, and the emitted field region named after the gate-adjacent layer; a Design
        whose two semiconductors are NON-contiguous (separated by a dielectric) is ambiguous -> raises.

Run: python -m validation.carriers_3d_multisemi
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dynameta.materials import (Material, MaterialRegistry, DrudeOptical, ConstantOptical,
                                 TransportModel, M_E)
from dynameta.geometry import UnitCell, Stack, Layer, Electrode, Design, centered_square
from dynameta.geometry.specs import OpticalSpec, Mesh3DSpec
from dynameta.carriers.devsim_3d import Stacked3DSpec, Devsim3DEquilibrium
from dynameta.sweep import BiasPoint

N_BG = 4.0e26
VG = 2.0
TSEMI = 16e-9
COMMON = dict(semi_material="ITO", oxide_material="HfO2", lateral_m=12e-9, oxide_thk_m=8e-9,
              n_bg_m3=N_BG, eps_semi=9.5, eps_oxide=18.0, dos_mass_kg=0.35 * M_E,
              grid_n=(6, 6, 33), mesh_min_nm=0.4, mesh_max_nm=3.0, physics="equilibrium")


def _zprofile(dev, region, zg):
    import devsim as ds
    z = np.asarray(ds.get_node_model_values(device=dev, region=region, name="z"))
    n = np.asarray(ds.get_node_model_values(device=dev, region=region, name="Electrons"))
    return z, n


def _collapse(z, n, zg):
    zs = np.unique(np.round(z, 13))
    vals = np.array([n[np.isclose(z, zz, atol=1e-12)].mean() for zz in zs])
    return np.interp(zg, zs, vals)


def _gate_a():
    import devsim as ds
    zg = np.linspace(0.0, TSEMI, 60)
    # single 16 nm semiconductor
    s1 = Devsim3DEquilibrium(Stacked3DSpec(semi_thk_m=TSEMI, **COMMON),
                             device_name="single", mesh_name="single_m")
    s1.solve(BiasPoint({"gate": VG, "body": 0.0}, "single"))
    z1, n1 = _zprofile("single", "semi", zg)
    p_single = _collapse(z1, n1, zg)
    s1.teardown()
    # split: 8 nm primary + 8 nm extra semiconductor, identical material/n_bg/eps
    s2 = Devsim3DEquilibrium(
        Stacked3DSpec(semi_thk_m=8e-9, extra_semiconductors=[("xs", "ITO", 8e-9, 9.5, N_BG)], **COMMON),
        device_name="split", mesh_name="split_m")
    s2.solve(BiasPoint({"gate": VG, "body": 0.0}, "split"))
    za, na = _zprofile("split", "semi", zg)
    zb, nb = _zprofile("split", "xs", zg)
    s2.teardown()
    zc = np.concatenate([za, zb]); nc = np.concatenate([na, nb])
    p_split = _collapse(zc, nc, zg)
    rel = float(np.max(np.abs(p_split - p_single) / np.maximum(p_single, 1e22)))
    g_a = rel < 5e-2
    print("[ms] A homojunction-split: single peak={:.3e} split peak={:.3e} ; max rel diff={:.3e} -> {}"
          .format(p_single.max(), p_split.max(), rel, "OK" if g_a else "FAIL"), flush=True)
    return g_a


def _design(layer_specs):
    reg = MaterialRegistry()
    reg.add(Material("air", ConstantOptical(1.0 + 0j)))
    reg.add(Material("Si", ConstantOptical(12.0 + 0j)))
    reg.add(Material("HfO2", ConstantOptical(4.0 + 0j), eps_static_dc=18.0))
    reg.add(Material("Al-Nd", ConstantOptical(-180 + 30j), is_metal=True))
    reg.add(Material("ITO", DrudeOptical(eps_inf=4.25, m_opt_kg=0.225 * M_E, gamma_rad_s=1.1e14),
                     transport=TransportModel(n_bg_m3=N_BG, eps_static=9.5,
                                              dos_mass_kg_of_n_m3=lambda n: 0.35 * M_E)))
    cell = UnitCell.square(370e-9)
    layers = [Layer(nm, thk, mat) for (nm, thk, mat) in layer_specs]
    stack = Stack(layers=layers, superstrate_material="air", substrate_material="Si")
    # the gate is a PATCH (CrossSection footprint) so from_design's gate-detection picks it (the
    # metasurface gate IS a patch); the body/back contact on the mirror is a full-plane electrode.
    electrodes = [Electrode("bot", "mirror", "full", role="biased"),
                  Electrode("gate", "patch", centered_square(cell, 300e-9), role="biased")]
    return Design(name="hetero", unit_cell=cell, stack=stack, electrodes=electrodes,
                  materials=reg, mesh_3d=Mesh3DSpec(), optical=OpticalSpec(polarization="x"))


def _gate_b():
    # contiguous heterostack: mirror / lower_hfo2 / ito_bot / ito_top / upper_hfo2 / patch(gate)
    d = _design([("mirror", 70e-9, "Al-Nd"), ("lower_hfo2", 7e-9, "HfO2"),
                 ("ito_bot", 6e-9, "ITO"), ("ito_top", 5e-9, "ITO"),
                 ("upper_hfo2", 7e-9, "HfO2"), ("patch", 50e-9, "air")])
    spec = Stacked3DSpec.from_design(d)
    ok_h = (spec.field_region_name == "ito_top"                      # gate-adjacent emitted
            and spec.semi_material == "ITO" and abs(spec.semi_thk_m - 6e-9) < 1e-12   # body-most primary
            and len(spec.extra_semiconductors) == 1
            and spec.extra_semiconductors[0][0] == "ito_top"
            and abs(spec.extra_semiconductors[0][2] - 5e-9) < 1e-12
            and spec.oxide_material == "HfO2" and abs(spec.oxide_thk_m - 7e-9) < 1e-12)
    print("[ms] B from_design heterostack: primary='semi'({:.0f}nm) extra={} field='{}' oxide={}({:.0f}nm)"
          " -> {}".format(spec.semi_thk_m * 1e9,
                          [(e[0], "{:.0f}nm".format(e[2] * 1e9)) for e in spec.extra_semiconductors],
                          spec.field_region_name, spec.oxide_material, spec.oxide_thk_m * 1e9,
                          "OK" if ok_h else "FAIL"), flush=True)
    # non-contiguous: ito / hfo2 / ito separated -> ambiguous -> raise
    raised = False
    try:
        Stacked3DSpec.from_design(_design(
            [("mirror", 70e-9, "Al-Nd"), ("ito_a", 5e-9, "ITO"), ("mid_hfo2", 4e-9, "HfO2"),
             ("ito_b", 5e-9, "ITO"), ("upper_hfo2", 7e-9, "HfO2"), ("patch", 50e-9, "air")]))
    except ValueError as e:
        raised = "contiguous" in str(e).lower()
    print("[ms] B non-contiguous semiconductors raise: {} -> {}".format(raised, "OK" if raised else "FAIL"),
          flush=True)
    return ok_h and raised


def main():
    print("[ms] === 3D multi-semiconductor heterostack (split-reduction + from_design) ===", flush=True)
    g_b = _gate_b()       # cheap (no solve) first
    g_a = _gate_a()
    ok = g_a and g_b
    print("[ms] *** 3D MULTI-SEMICONDUCTOR: {} ***".format("PASS" if ok else "FAIL"), flush=True)
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
