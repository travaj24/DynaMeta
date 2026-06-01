"""Fast, dependency-light tests for the data model + n->eps physics (numpy only;
no DEVSIM/NGSolve)."""
import numpy as np

from dynameta import (UnitCell, Stack, Layer, Inclusion, Design,
                       Material, MaterialRegistry, ConstantOptical, DrudeOptical,
                       centered_square, CarrierField)
from dynameta.materials import M_E
from dynameta.core.carrier_field import CarrierRegion, ELECTRON_DENSITY, POTENTIAL


def test_constant_optical():
    assert ConstantOptical(4.0 + 0j).eps(1300e-9) == 4.0 + 0j


def test_drude_density_dependence():
    d = DrudeOptical(eps_inf=4.25, m_opt_kg=0.225 * M_E, gamma_rad_s=1.1e14)
    lam = 1300e-9
    e1 = complex(d.eps(lam, n_m3=4e26))
    e2 = complex(d.eps(lam, n_m3=8e26))
    assert e2.real < e1.real        # more free carriers -> lower Re(eps) (toward ENZ)
    assert e1.imag > 0              # exp(-iwt) lossy convention: Im(eps) > 0


def test_material_dc_permittivity():
    diel = Material("HfO2", ConstantOptical(4.0 + 0j), eps_static_dc=18.0)
    assert diel.dc_permittivity() == 18.0
    assert Material("air", ConstantOptical(1.0 + 0j)).dc_permittivity() is None


def test_unit_cell_and_centered_footprint():
    cell = UnitCell.square(370e-9)
    assert cell.period_x_m == 370e-9 and cell.period_y_m == 370e-9
    xlo, xhi, ylo, yhi = centered_square(cell, 175e-9).bbox_m()
    assert abs((xlo + xhi) / 2 - 185e-9) < 1e-15        # centered in the cell
    assert abs((xhi - xlo) - 175e-9) < 1e-15            # correct side length


def test_device_symmetry_c4v():
    reg = MaterialRegistry()
    reg.add(Material("air", ConstantOptical(1.0 + 0j)))
    reg.add(Material("Au", ConstantOptical(-100 + 8j), is_metal=True))
    cell = UnitCell.square(370e-9)
    stack = Stack(
        layers=[Layer("patch", 50e-9, "air",
                       inclusions=[Inclusion(centered_square(cell, 175e-9), "Au")])],
        superstrate_material="air", substrate_material="air")
    d = Design(name="t", unit_cell=cell, stack=stack, electrodes=[], materials=reg)
    assert d.device_symmetry() == "c4v"


def test_carrier_field_vocab():
    reg = CarrierRegion(name="ito", role="semiconductor", material="ITO",
                         nodes_m=np.zeros((3, 2)),
                         node_fields={ELECTRON_DENSITY: np.ones(3), POTENTIAL: np.zeros(3)})
    cf = CarrierField(bias_label="zero", voltages={}, ndim=2, temperature_K=300.0,
                       regions={"ito": reg}, n_bg_by_region={"ito": 4e26},
                       unit_cell_m=(370e-9, 370e-9))
    assert ELECTRON_DENSITY in cf.field_vocab()
    assert cf.ndim == 2
