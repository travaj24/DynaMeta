"""Fast (no FEM) tests for the TMM reference helper -- validates the n_list/d_list/units
wiring against analytic Fresnel + energy conservation, and the Design extractor. Uses the
`tmm` dependency (already required); runs in CI."""
import numpy as np
import pytest

from dynameta.optics.tmm_reference import stack_rta, design_layer_stack


def test_single_interface_fresnel_normal():
    # bare n=1 | n=1.5 interface, normal incidence: R = ((1-1.5)/(1+1.5))^2 = 0.04
    R, T, A = stack_rta(1.0, [], 1.5, 1300e-9, theta_deg=0.0, pol="s")
    assert abs(R - 0.04) < 1e-6
    assert abs(A) < 1e-9                       # lossless -> no absorption
    assert abs(R + T + A - 1.0) < 1e-9
    # T carries the index factor: T = 1 - R for this interface (tmm convention)
    assert abs(T - 0.96) < 1e-6


def test_lossless_slab_energy_conserves():
    R, T, A = stack_rta(1.0, [(2.0, 250e-9)], 1.0, 1300e-9, theta_deg=20.0, pol="p")
    assert abs(A) < 1e-9
    assert abs(R + T + A - 1.0) < 1e-9
    assert 0.0 < R < 1.0


def test_lossy_slab_absorbs():
    R, T, A = stack_rta(1.0, [(2.0 + 0.1j, 250e-9)], 1.0, 1300e-9, pol="s")
    assert A > 0.0                              # a lossy slab absorbs
    assert abs(R + T + A - 1.0) < 1e-9


def test_s_p_differ_at_angle():
    rs = stack_rta(1.0, [(2.0, 250e-9)], 1.5, 1300e-9, theta_deg=45.0, pol="s")
    rp = stack_rta(1.0, [(2.0, 250e-9)], 1.5, 1300e-9, theta_deg=45.0, pol="p")
    assert abs(rs[0] - rp[0]) > 1e-3            # s and p reflectance differ at oblique


def test_design_layer_stack_extract_and_reject_inclusions():
    from dynameta.materials import Material, MaterialRegistry, ConstantOptical
    from dynameta.geometry import UnitCell, Stack, Layer, Inclusion, Design, centered_square
    reg = MaterialRegistry()
    reg.add(Material("air", ConstantOptical(1.0 + 0j)))
    reg.add(Material("hi", ConstantOptical(4.0 + 0j)))   # n=2
    cell = UnitCell.square(300e-9)
    # uniform stack -> extracts and matches a manual stack_rta
    d_uniform = Design(name="u", unit_cell=cell,
                       stack=Stack(layers=[Layer("slab", 250e-9, "hi")],
                                    superstrate_material="air", substrate_material="air"),
                       electrodes=[], materials=reg)
    n_sup, layers, n_sub = design_layer_stack(d_uniform, 1300e-9)
    assert abs(complex(layers[0][0]) - 2.0) < 1e-9 and abs(layers[0][1] - 250e-9) < 1e-18
    R1, _, _ = stack_rta(n_sup, layers, n_sub, 1300e-9)
    R2, _, _ = stack_rta(1.0, [(2.0, 250e-9)], 1.0, 1300e-9)
    assert abs(R1 - R2) < 1e-9
    # a layer with an inclusion is laterally structured -> TMM must refuse
    d_struct = Design(name="s", unit_cell=cell,
                      stack=Stack(layers=[Layer("p", 250e-9, "air",
                                   inclusions=[Inclusion(centered_square(cell, 120e-9), "hi")])],
                                   superstrate_material="air", substrate_material="air"),
                      electrodes=[], materials=reg)
    with pytest.raises(ValueError):
        design_layer_stack(d_struct, 1300e-9)
