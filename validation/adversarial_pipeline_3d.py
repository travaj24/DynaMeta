"""ADVERSARIAL: I claimed the from_design 3D builder is 'run_pipeline-compatible, no
hand-alignment' -- but I only checked the region NAME matched the optics alignment. I
never actually RAN run_pipeline with the 3D carrier. This does: build a gated-ITO Design,
make the 3D carrier via Stacked3DSpec.from_design, and run the FULL pipeline (carriers ->
bridge -> optics) at two biases with NO hand-built alignment. Pass = it completes and
produces finite R that responds to the gate bias. Run: python -m validation.adversarial_pipeline_3d
"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dynameta.materials import (Material, MaterialRegistry, DrudeOptical, ConstantOptical,
                                  TransportModel, M_E)
from dynameta.geometry import UnitCell, Stack, Layer, Electrode, Design, centered_square
from dynameta.geometry.specs import OpticalSpec, Mesh3DSpec
from dynameta.carriers.devsim_3d import Devsim3DEquilibrium, Stacked3DSpec
from dynameta.optics.ngsolve_layered import LayeredOpticalBuilder
from dynameta.pipeline import run_pipeline
from dynameta.sweep import Sweep, BiasPoint

PERIOD = 220e-9


def build_design():
    reg = MaterialRegistry()
    reg.add(Material("air", ConstantOptical(1.0 + 0j)))
    reg.add(Material("HfO2", ConstantOptical(4.0 + 0j), eps_static_dc=18.0))
    reg.add(Material("ITO", DrudeOptical(eps_inf=4.25, m_opt_kg=0.225 * M_E, gamma_rad_s=1.1e14),
                      transport=TransportModel(n_bg_m3=4e26, eps_static=9.5,
                                                dos_mass_kg_of_n_m3=lambda n: 0.35 * M_E)))
    cell = UnitCell.square(PERIOD)
    layers = [Layer("ito", 10e-9, "ITO"), Layer("hfo2", 8e-9, "HfO2")]
    stack = Stack(layers=layers, superstrate_material="air", substrate_material="air")  # vacuum exit
    electrodes = [Electrode("gate", "hfo2", centered_square(cell, 120e-9), role="biased"),
                  Electrode("gnd", "ito", "x_lo", role="ground", fixed_voltage_V=0.0)]
    m3 = Mesh3DSpec(pml_thk_m=500e-9, superstrate_buffer_m=1400e-9, substrate_buffer_m=1400e-9,
                     maxh_superstrate_m=45e-9, maxh_substrate_m=45e-9, maxh_background_m=12e-9)
    return Design(name="gated_ito_3d", unit_cell=cell, stack=stack, electrodes=electrodes,
                   materials=reg, mesh_3d=m3, optical=OpticalSpec(polarization="x", linear_solver="umfpack"))


def main():
    d = build_design()
    spec = Stacked3DSpec.from_design(d)
    print("[t] from_design spec: region={} frac={:.3f} semi={:.0f}nm".format(
        spec.field_region_name, spec.gate_patch_frac, spec.semi_thk_m * 1e9), flush=True)
    carrier = Devsim3DEquilibrium(spec)
    sweep = Sweep(bias_points=[BiasPoint({"gate": 0.0, "body": 0.0}, "0V"),
                                BiasPoint({"gate": 1.0, "body": 0.0}, "+1V")],
                   wavelengths_nm=[1300.0])
    try:
        rows = run_pipeline(d, sweep, carrier_solver=carrier,
                             optical_builder=LayeredOpticalBuilder(d), verbose=True)
    except Exception as e:
        import traceback; traceback.print_exc()
        print("[t] *** ADVERSARIAL run_pipeline 3D: FAILED to run end-to-end: {} ***".format(
            type(e).__name__), flush=True)
        return
    Rs = {r.bias_label: r.R for r in rows}
    print("[t] run_pipeline rows: {}".format({k: round(v, 6) for k, v in Rs.items()}), flush=True)
    finite = all(np.isfinite(list(Rs.values())))
    dR = abs(Rs.get("+1V", 0) - Rs.get("0V", 0))
    print("[t] dR(+1V - 0V) = {:.6f}".format(dR), flush=True)
    ok = finite and len(rows) == 2
    print("[t] *** ADVERSARIAL run_pipeline 3D (no hand-alignment): completes={} finite_R={} "
          "bias_propagates={} -> {} ***".format(
        len(rows) == 2, finite, dR > 1e-7, "PASS" if ok else "FAIL"), flush=True)


if __name__ == "__main__":
    main()
