"""Child-process driver for tests/test_trap_noise.py::test_layered_builder_noise_with_kinetic_traps (v0.15 K3):
LayeredDevsimBuilder.noise_spectrum / terminal_noise make the gate circuit-driven, and DEVSIM circuits are
process-global (tests/_bipolar_calllog.py), so the builder scenario runs here and prints one line
"TRAP_NOISE_RESULT <json>".

    python tests/_trap_noise_child.py kinetic_builder
"""
from __future__ import annotations

import dataclasses
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dynameta.carriers.devsim_env import import_devsim    # noqa: E402

ds = import_devsim()

from dynameta.carriers import noise_analysis as NA           # noqa: E402
from dynameta.carriers import physics_traps as PT            # noqa: E402
from dynameta.carriers.devsim_layered import LayeredDevsimBuilder   # noqa: E402
from dynameta.sweep import BiasPoint                         # noqa: E402

GATE = "top_contact"
FF = [1e6, 1e8, 1e10]


def design():
    """Gated bipolar n-GaAs (Boltzmann: the diffusion source is Nyquist-exact): Al-Nd mirror (bottom gate) /
    7 nm HfO2 / 20 nm n-GaAs 1e23 m^-3 with a KINETIC acceptor level (TransportModel.srh_traps), grounded on
    both x edges / 7 nm HfO2 / Au patch (top gate), 60 nm cell -- tests/_noise_method_child.te_design without
    the AlGaAs layer."""
    from validation import _reference_device as RD
    from validation.devsim_heterojunction import GAAS, transport
    from dynameta.geometry import Design, Electrode, Inclusion, Layer, Stack, UnitCell, centered_square
    from dynameta.materials import ConstantOptical, Material
    reg = RD.build_materials("drift_diffusion")
    lv = PT.TrapLevel(energy_eV=0.6, density_m3=2e22, kind="acceptor", tau_n0_s=1e-9, tau_p0_s=1e-9,
                      dynamic=True)
    tm = dataclasses.replace(transport(GAAS, n_bg=1e23, physics="bipolar_dd"), carrier_statistics="boltzmann",
                             srh_traps=(lv,))
    reg.add(Material("GaAs_n", ConstantOptical(12.0 + 0j), transport=tm))
    cell = UnitCell.square(60e-9)
    layers = [Layer("mirror", 70e-9, "Al-Nd"), Layer("lower_hfo2", 7e-9, "HfO2"), Layer("gaas", 20e-9, "GaAs_n"),
              Layer("upper_hfo2", 7e-9, "HfO2"),
              Layer("patch", 50e-9, "air", inclusions=[Inclusion(centered_square(cell, 30e-9), "Au")])]
    el = [Electrode("bot_contact", "mirror", "full", role="biased"),
          Electrode(GATE, "patch", centered_square(cell, 30e-9), role="biased"),
          Electrode("gnd_left", "gaas", "x_lo", role="ground", fixed_voltage_V=0.0),
          Electrode("gnd_right", "gaas", "x_hi", role="ground", fixed_voltage_V=0.0)]
    return Design(name="trap_gate", unit_cell=cell, stack=Stack(layers=layers, superstrate_material="air",
                                                                substrate_material="Si"),
                  electrodes=el, materials=reg)


def run(scenario):
    if scenario != "kinetic_builder":
        raise SystemExit("unknown scenario " + scenario)
    it = PT.InterfaceTraps(dit_eV_m2=1e16, band_gap_eV=1.42, sigma_n_m2=1e-19, sigma_p_m2=1e-19,
                           dynamic=True, kinetic_bin_eV=0.2)
    b = LayeredDevsimBuilder(design(), mesh_name="tn_m", device_name="tn_d",
                             interface_traps={("gaas", "upper_hfo2"): it})
    b.solve(BiasPoint({}, "eq"))
    kv = PT.kinetic_trap_variables(b.device, "gaas")
    sn = b.noise_spectrum(FF, gate=GATE, per_cell=False)
    tn = b.terminal_noise(FF, gate=GATE)
    return {"kinetic_vars": kv, "mechanisms": sorted(sn.by_mechanism),
            "trap_share": [float(v) for v in (np.asarray(sn.by_mechanism["trap_n"]) + np.asarray(
                sn.by_mechanism["trap_p"])) / np.asarray(sn.S)],
            "default_has_trap": {"trap_n", "trap_p"} <= set(NA.DEFAULT_MECHANISMS),
            "S_I": [float(v) for v in tn.S_I], "ReY": [float(v) for v in tn.Y.real]}


if __name__ == "__main__":
    out = run(sys.argv[1])
    sys.stdout.flush()
    print("TRAP_NOISE_RESULT " + json.dumps(out))
