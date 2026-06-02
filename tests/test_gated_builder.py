"""Unit coverage for the LayeredDevsimBuilder full-edge-ground region planning (the gated-DD fix):
a drift-diffusion semiconductor layer with edge GROUND electrodes gets a thin adjacent edge-metal
strip carved at each grounded edge (so the ground is a region-region interface -> full-line node
capture -> gated DD converges), while the equilibrium path is left untouched. This tests the PURE
region-spec planning (_region_specs); the full build+solve convergence lives in
validation/gated_dd_builder.py. Importing the builder needs DEVSIM, so skip when absent (CI).
Run: python -m pytest tests/test_gated_builder.py -q
"""
import numpy as np
import pytest


def _design(physics):
    from dynameta.materials import (Material, MaterialRegistry, ConstantOptical,
                                    TransportModel, M_E)
    from dynameta.geometry import UnitCell, Stack, Layer, Electrode, Design
    reg = MaterialRegistry()
    reg.add(Material("air", ConstantOptical(1.0 + 0j)))
    reg.add(Material("Au", ConstantOptical(-100 + 8j), is_metal=True))
    mob = (None if physics == "equilibrium"
           else (lambda n: np.full_like(np.asarray(n, float), 30e-4)))
    reg.add(Material("semi", optical=ConstantOptical(4.0 + 0j),
                     transport=TransportModel(n_bg_m3=4e26, eps_static=9.5,
                                              dos_mass_kg_of_n_m3=lambda n: 0.35 * M_E,
                                              physics=physics, mobility_m2Vs_of_n_m3=mob)))
    cell = UnitCell.square(200e-9)
    layers = [Layer("metalpad", 20e-9, "Au"), Layer("ito", 12e-9, "semi")]
    stack = Stack(layers=layers, superstrate_material="air", substrate_material="air")
    electrodes = [Electrode("gl", "ito", "x_lo", role="ground"),
                  Electrode("gr", "ito", "x_hi", role="ground")]
    return Design(name="g", unit_cell=cell, stack=stack, electrodes=electrodes, materials=reg)


def test_dd_edge_grounds_carve_edge_metal_strips():
    pytest.importorskip("devsim")
    from dynameta.carriers.devsim_layered import LayeredDevsimBuilder, _EDGE_METAL_W_M
    b = LayeredDevsimBuilder(_design("drift_diffusion"))
    w, P = _EDGE_METAL_W_M, 200e-9
    # both grounded edges (x_lo and x_hi) get an inert edge-metal strip
    egnd = {s.name: s for s in b._specs if "egnd" in s.name}
    assert sorted(egnd) == ["ito_egnd_hi", "ito_egnd_lo"]
    assert all(s.role == "metal" for s in egnd.values())
    assert abs(egnd["ito_egnd_lo"].x_lo) < 1e-15 and abs(egnd["ito_egnd_lo"].x_hi - w) < 1e-15
    assert abs(egnd["ito_egnd_hi"].x_lo - (P - w)) < 1e-15 and abs(egnd["ito_egnd_hi"].x_hi - P) < 1e-15
    # the semiconductor is narrowed to [w, P-w] (the ground sits at the interfaces)
    ito = next(s for s in b._specs if s.name == "ito")
    assert abs(ito.x_lo - w) < 1e-15 and abs(ito.x_hi - (P - w)) < 1e-15
    assert b._dd_full_edge_grounds() == {"ito": {"x_lo", "x_hi"}}


def test_equilibrium_edge_ground_not_carved():
    pytest.importorskip("devsim")
    from dynameta.carriers.devsim_layered import LayeredDevsimBuilder
    b = LayeredDevsimBuilder(_design("equilibrium"))
    assert not any("egnd" in s.name for s in b._specs)          # equilibrium path untouched
    ito = next(s for s in b._specs if s.name == "ito")
    assert ito.x_lo == 0.0 and abs(ito.x_hi - 200e-9) < 1e-15
    assert b._dd_full_edge_grounds() == {}
