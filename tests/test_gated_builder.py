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


def test_eq_registry_forget_is_targeted():
    """forget(name, loc) drops ONLY the matching recorded entry (and forget(name) drops all by
    name) without touching the live equation -- the bookkeeping half of repointing a contact."""
    R = pytest.importorskip("dynameta.carriers.eq_registry")
    dev = "utest_forget_dev"
    R._REG[dev] = [
        {"scope": "region", "loc": "ito", "name": "PotentialEquation", "kwargs": {}},
        {"scope": "contact", "loc": "gate", "name": "PotentialEquation", "kwargs": {}},
        {"scope": "contact", "loc": "gnd", "name": "PotentialEquation", "kwargs": {}},
        {"scope": "contact", "loc": "gate", "name": "ElectronContinuityEquation", "kwargs": {}},
    ]
    try:
        R.forget(dev, "PotentialEquation", loc="gate")          # only the gate's Potential record
        present = {(e["scope"], e["loc"], e["name"]) for e in R._REG[dev]}
        assert ("contact", "gate", "PotentialEquation") not in present
        assert ("region", "ito", "PotentialEquation") in present        # region kept
        assert ("contact", "gnd", "PotentialEquation") in present       # other contact kept
        assert ("contact", "gate", "ElectronContinuityEquation") in present  # other name kept
        R.forget(dev, "PotentialEquation")                      # no loc -> drop all by name
        names = {(e["loc"], e["name"]) for e in R._REG[dev]}
        assert names == {("gate", "ElectronContinuityEquation")}
        R.forget("no_such_device", "PotentialEquation")         # missing device is a no-op
    finally:
        R.clear(dev)


def test_set_ssac_gate_requires_built_device():
    """set_ssac_gate raises (does not silently no-op) if called before the device is built."""
    pytest.importorskip("devsim")
    from dynameta.carriers.devsim_layered import LayeredDevsimBuilder
    b = LayeredDevsimBuilder(_design("drift_diffusion"), mesh_name="ut_ssac_m",
                             device_name="ut_ssac_d")
    assert not b._built
    with pytest.raises(RuntimeError):
        b.set_ssac_gate("gl")


def test_2d_builder_refuses_lateral_isolation_cases():
    # audit C2-3: non-ambient background WITH inclusions (and laterally touching
    # inclusions) get NO lateral interface in the 2D builder -- the inclusion would be
    # electrostatically isolated silently; must raise (the 3D builder wires these)
    import pytest
    pytest.importorskip("devsim")              # devsim_layered imports devsim at module load
    from dynameta.carriers.devsim_layered import LayeredDevsimBuilder
    from dynameta.geometry import Design, Inclusion, Layer, Stack, UnitCell
    from dynameta.geometry.cross_section import Rectangle
    from dynameta.materials import ConstantOptical, Material, MaterialRegistry, TransportModel
    reg = MaterialRegistry()
    reg.add(Material("air", ConstantOptical(1.0 + 0j)))
    reg.add(Material("hfo2", ConstantOptical(4.0 + 0j)))
    reg.add(Material("ito", ConstantOptical(3.9 + 0j),
                     transport=TransportModel(n_bg_m3=4e26, eps_static=9.5,
                                              dos_mass_kg_of_n_m3=lambda n: 3.2e-31)))
    P = 370e-9
    bar = Inclusion(shape=Rectangle(P / 2, P / 4, 0.4 * P, P), material="ito")
    d = Design(name="t", unit_cell=UnitCell.square(P),
               stack=Stack(layers=[Layer("mix", 10e-9, "hfo2", inclusions=[bar])],
                           superstrate_material="air", substrate_material="air"),
               electrodes=[], materials=reg)
    with pytest.raises(NotImplementedError, match="lateral"):
        LayeredDevsimBuilder(d)
    # two x-touching inclusions in an AMBIENT layer likewise raise
    b1 = Inclusion(shape=Rectangle(P / 4, P / 4, 0.25 * P, P), material="ito")
    b2 = Inclusion(shape=Rectangle(P / 2, P / 4, 0.25 * P, P), material="ito")
    d2 = Design(name="t2", unit_cell=UnitCell.square(P),
                stack=Stack(layers=[Layer("pair", 10e-9, "air", inclusions=[b1, b2])],
                            superstrate_material="air", substrate_material="air"),
                electrodes=[], materials=reg)
    with pytest.raises(NotImplementedError, match="DECOUPLED"):
        LayeredDevsimBuilder(d2)


def test_bipolar3d_expr_requires_body_doping():
    # audit C2-1: net_doping_expr makes the (acceptor, n_bg_m3) scalars dead as doping --
    # phi_bi derived from them silently mis-references the gate (+0.714 V at Vg=0 in the
    # audit probe); the spec-level contract now requires the signed body-side doping
    pytest.importorskip("devsim")              # devsim_3d imports devsim at module load
    from dynameta.carriers.devsim_3d import Stacked3DSpec
    spec = Stacked3DSpec(physics="bipolar_dd", n_i_m3=1e17, net_doping_expr="-1.0e23")
    assert spec.body_net_doping_m3 is None                     # guard fires at solve()
