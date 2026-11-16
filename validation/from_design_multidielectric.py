"""Validate Stacked3DSpec.from_design on a MULTI-DIELECTRIC stack (a generic gated cell:
mirror / oxA / oxB / film / oxB / oxA / patch). from_design must find the film semiconductor +
the nearest GATE-SIDE dielectric (the upper oxB, toward the patch gate above the film), derive
gate_patch_frac from the patch footprint, name the region "film", and tolerate the metal
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

PERIOD = 400e-9
PATCH = 150e-9
FILM_NBG = 3e26


def build_generic_cell():
    reg = MaterialRegistry()
    reg.add(Material("air", ConstantOptical(1.0 + 0j)))
    reg.add(Material("glass", ConstantOptical(1.45 ** 2 + 0j)))
    reg.add(Material("oxA", ConstantOptical(1.70 ** 2 + 0j), eps_static_dc=9.0))
    reg.add(Material("oxB", ConstantOptical(2.00 ** 2 + 0j), eps_static_dc=20.0))
    reg.add(Material("mirror_metal", ConstantOptical(-120.0 + 20.0j), is_metal=True))
    reg.add(Material("metal", ConstantOptical(-60.0 + 4.0j), is_metal=True))
    reg.add(Material("ITO", DrudeOptical(eps_inf=4.0, m_opt_kg=0.30 * M_E, gamma_rad_s=1.2e14),
                      transport=TransportModel(n_bg_m3=FILM_NBG, eps_static=9.0,
                                                dos_mass_kg_of_n_m3=lambda n: 0.30 * M_E)))
    cell = UnitCell.square(PERIOD)
    layers = [
        Layer("mirror", 80e-9, "mirror_metal"),
        Layer("lower_oxa", 2e-9, "oxA"),
        Layer("lower_oxb", 6e-9, "oxB"),
        Layer("film", 6e-9, "ITO"),
        Layer("upper_oxb", 6e-9, "oxB"),
        Layer("upper_oxa", 2e-9, "oxA"),
        Layer("patch", 40e-9, "air", inclusions=[Inclusion(centered_square(cell, PATCH), "metal")]),
    ]
    stack = Stack(layers=layers, superstrate_material="air", substrate_material="glass")
    electrodes = [
        Electrode("bot_contact", "mirror", "full", role="biased"),
        Electrode("top_contact", "patch", centered_square(cell, PATCH), role="biased"),
        Electrode("film_gnd_left", "film", "x_lo", role="ground", fixed_voltage_V=0.0),
    ]
    return Design(name="generic_cell", unit_cell=cell, stack=stack, electrodes=electrodes,
                   materials=reg, mesh_3d=Mesh3DSpec(), optical=OpticalSpec(polarization="x"))


def main():
    d = build_generic_cell()
    spec = Stacked3DSpec.from_design(d)
    print("[t] from_design (multi-dielectric gate stack):", flush=True)
    print("[t]   region={}  semi_material={}  semi_thk={:.0f}nm".format(
        spec.field_region_name, spec.semi_material, spec.semi_thk_m * 1e9), flush=True)
    print("[t]   gate dielectric={}  oxide_thk={:.0f}nm  eps_oxide={:.1f}".format(
        spec.oxide_material, spec.oxide_thk_m * 1e9, spec.eps_oxide), flush=True)
    print("[t]   gate_patch_frac={:.3f} ({:.0f}/{:.0f}={:.3f})  n_bg={:.1e}".format(
        spec.gate_patch_frac, PATCH * 1e9, PERIOD * 1e9, PATCH / PERIOD, spec.n_bg_m3), flush=True)

    ok = (spec.field_region_name == "film"
          and spec.semi_material == "ITO" and abs(spec.semi_thk_m - 6e-9) < 1e-12
          and spec.oxide_material == "oxB"             # the GATE-side (upper) oxB, not the lower one
          and abs(spec.oxide_thk_m - 6e-9) < 1e-12 and abs(spec.eps_oxide - 20.0) < 1e-9
          and abs(spec.gate_patch_frac - PATCH / PERIOD) < 1e-3
          and abs(spec.n_bg_m3 - FILM_NBG) < 1e18)
    print("[t] *** MULTI-DIELECTRIC from_design: {} ***".format("PASS" if ok else "FAIL"), flush=True)
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
