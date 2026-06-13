"""Validate Stacked3DSpec.from_design on a MULTI-DIELECTRIC stack (the full reference-like
mirror/Al2O3/HfO2/ITO/HfO2/Al2O3/patch). from_design must find the ITO semiconductor +
the nearest GATE-SIDE dielectric (upper HfO2, toward the patch gate above ITO), derive
gate_patch_frac from the patch footprint, name the region "ito", and tolerate the Au
inclusion in the patch layer. No solve -- this checks the Design->spec extraction.
Run:  python -m validation.from_design_multidielectric
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dynameta.materials import (Material, MaterialRegistry, DrudeOptical, ConstantOptical,
                                  TransportModel, M_E)
from dynameta.geometry import UnitCell, Stack, Layer, Inclusion, Electrode, Design, centered_square
from dynameta.geometry.specs import OpticalSpec, Mesh3DSpec
from dynameta.carriers.devsim_3d import Stacked3DSpec

PERIOD = 370e-9
ITO_NBG = 4e26


def build_reference_like():
    reg = MaterialRegistry()
    reg.add(Material("air", ConstantOptical(1.0 + 0j)))
    reg.add(Material("Si", ConstantOptical(12.0 + 0j)))
    reg.add(Material("Al2O3", ConstantOptical(2.756 + 0j), eps_static_dc=9.0))
    reg.add(Material("HfO2", ConstantOptical(4.0 + 0j), eps_static_dc=18.0))
    reg.add(Material("Al-Nd", ConstantOptical(-180 + 30j), is_metal=True))
    reg.add(Material("Au", ConstantOptical(-100 + 8j), is_metal=True))
    reg.add(Material("ITO", DrudeOptical(eps_inf=4.25, m_opt_kg=0.225 * M_E, gamma_rad_s=1.1e14),
                      transport=TransportModel(n_bg_m3=ITO_NBG, eps_static=9.5,
                                                dos_mass_kg_of_n_m3=lambda n: 0.35 * M_E)))
    cell = UnitCell.square(PERIOD)
    layers = [
        Layer("mirror", 70e-9, "Al-Nd"),
        Layer("lower_al2o3", 1e-9, "Al2O3"),
        Layer("lower_hfo2", 7e-9, "HfO2"),
        Layer("ito", 5e-9, "ITO"),
        Layer("upper_hfo2", 7e-9, "HfO2"),
        Layer("upper_al2o3", 1e-9, "Al2O3"),
        Layer("patch", 50e-9, "air", inclusions=[Inclusion(centered_square(cell, 175e-9), "Au")]),
    ]
    stack = Stack(layers=layers, superstrate_material="air", substrate_material="Si")
    electrodes = [
        Electrode("bot_contact", "mirror", "full", role="biased"),
        Electrode("top_contact", "patch", centered_square(cell, 175e-9), role="biased"),
        Electrode("ito_gnd_left", "ito", "x_lo", role="ground", fixed_voltage_V=0.0),
    ]
    return Design(name="reference_like", unit_cell=cell, stack=stack, electrodes=electrodes,
                   materials=reg, mesh_3d=Mesh3DSpec(), optical=OpticalSpec(polarization="x"))


def main():
    d = build_reference_like()
    spec = Stacked3DSpec.from_design(d)
    print("[t] from_design (multi-dielectric reference gate stack):", flush=True)
    print("[t]   region={}  semi_material={}  semi_thk={:.0f}nm".format(
        spec.field_region_name, spec.semi_material, spec.semi_thk_m * 1e9), flush=True)
    print("[t]   gate dielectric={}  oxide_thk={:.0f}nm  eps_oxide={:.1f}".format(
        spec.oxide_material, spec.oxide_thk_m * 1e9, spec.eps_oxide), flush=True)
    print("[t]   gate_patch_frac={:.3f} (175/370={:.3f})  n_bg={:.1e}".format(
        spec.gate_patch_frac, 175.0 / 370.0, spec.n_bg_m3), flush=True)

    ok = (spec.field_region_name == "ito"
          and spec.semi_material == "ITO" and abs(spec.semi_thk_m - 5e-9) < 1e-12
          and spec.oxide_material == "HfO2"            # the GATE-side (upper) HfO2, not lower
          and abs(spec.oxide_thk_m - 7e-9) < 1e-12 and abs(spec.eps_oxide - 18.0) < 1e-9
          and abs(spec.gate_patch_frac - 175.0 / 370.0) < 1e-3
          and abs(spec.n_bg_m3 - ITO_NBG) < 1e18)
    print("[t] *** MULTI-DIELECTRIC from_design: {} ***".format("PASS" if ok else "FAIL"), flush=True)
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
