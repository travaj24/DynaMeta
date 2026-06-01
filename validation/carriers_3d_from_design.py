"""Validate the Design-driven 3D builder path: Stacked3DSpec.from_design(design) derives
the stacked 3D carrier spec (semiconductor + gate-dielectric layers, cell period, gate
footprint -> gate_patch_frac, region name) from a single Design, and the emitted
CarrierField region name matches the optics builder's alignment source_region -- so a
3D carrier solve drops into run_pipeline with NO hand-built alignment.
Run:  python -m validation.carriers_3d_from_design
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dynameta.materials import (Material, MaterialRegistry, DrudeOptical, ConstantOptical,
                                  TransportModel, M_E)
from dynameta.geometry import UnitCell, Stack, Layer, Electrode, Design, centered_square
from dynameta.geometry.specs import OpticalSpec, Mesh3DSpec
from dynameta.carriers.devsim_3d import Stacked3DSpec
from dynameta.optics.ngsolve_layered import LayeredOpticalBuilder

PERIOD = 300e-9
N_BG = 4e26


def build_design():
    reg = MaterialRegistry()
    reg.add(Material("air", ConstantOptical(1.0 + 0j)))
    reg.add(Material("HfO2", ConstantOptical(4.0 + 0j), eps_static_dc=18.0))
    reg.add(Material("Si", ConstantOptical(12.0 + 0j)))
    reg.add(Material("ITO", DrudeOptical(eps_inf=4.25, m_opt_kg=0.225 * M_E, gamma_rad_s=1.1e14),
                      transport=TransportModel(n_bg_m3=N_BG, eps_static=9.5,
                                                dos_mass_kg_of_n_m3=lambda n: 0.35 * M_E)))
    cell = UnitCell.square(PERIOD)
    layers = [Layer("ito", 10e-9, "ITO"), Layer("hfo2", 8e-9, "HfO2")]
    stack = Stack(layers=layers, superstrate_material="air", substrate_material="Si")
    electrodes = [
        Electrode("gate", "hfo2", centered_square(cell, 150e-9), role="biased"),  # 150nm patch
        Electrode("gnd", "ito", "x_lo", role="ground", fixed_voltage_V=0.0),
    ]
    return Design(name="gated_ito", unit_cell=cell, stack=stack, electrodes=electrodes,
                   materials=reg, mesh_3d=Mesh3DSpec(), optical=OpticalSpec(polarization="x"))


def main():
    d = build_design()
    spec = Stacked3DSpec.from_design(d)
    print("[t] from_design: region='{}' semi={:.0f}nm oxide={:.0f}nm lateral={:.0f}nm "
          "n_bg={:.1e} eps_ox={:.1f} frac={:.3f} physics={}".format(
        spec.field_region_name, spec.semi_thk_m * 1e9, spec.oxide_thk_m * 1e9,
        spec.lateral_m * 1e9, spec.n_bg_m3, spec.eps_oxide, spec.gate_patch_frac, spec.physics), flush=True)

    derived_ok = (spec.field_region_name == "ito" and abs(spec.semi_thk_m - 10e-9) < 1e-12
                   and abs(spec.oxide_thk_m - 8e-9) < 1e-12 and abs(spec.lateral_m - PERIOD) < 1e-12
                   and abs(spec.eps_oxide - 18.0) < 1e-9 and abs(spec.n_bg_m3 - N_BG) < 1e18
                   and abs(spec.gate_patch_frac - 150e-9 / PERIOD) < 1e-3)
    print("[t] (1) spec derived from Design correctly: {}".format(derived_ok), flush=True)

    # run_pipeline compatibility: the optics builder's alignment must reference the same
    # source_region as the 3D CarrierField (spec.field_region_name) -> no hand-alignment.
    geo = LayeredOpticalBuilder(d)
    geo.build()
    align = geo.alignment()
    src = {ra.source_region for ra in align.region_alignments}
    print("[t] optics alignment source_regions = {}".format(sorted(src)), flush=True)
    compatible = spec.field_region_name in src
    print("[t] (2) carrier region '{}' matches an optics alignment source_region: {}".format(
        spec.field_region_name, compatible), flush=True)

    ok = derived_ok and compatible
    print("[t] *** DESIGN-DRIVEN 3D BUILDER: derived={} run_pipeline_compatible={} -> {} ***".format(
        bool(derived_ok), bool(compatible), "PASS" if ok else "FAIL"), flush=True)


if __name__ == "__main__":
    main()
