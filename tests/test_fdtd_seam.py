"""Fast (no-FDTD-run) unit tests for the FDTD OpticalSolver seam helpers: the complex-eps -> FDTDLayer
Drude inversion, the Design -> layer mapping (order + guards), and the vacuum-end-media guard."""
import math

import pytest

from dynameta.constants import C_LIGHT
from dynameta.geometry import Design, Layer, Stack, UnitCell
from dynameta.geometry.cross_section import Circle
from dynameta.geometry.stack import Inclusion
from dynameta.materials import ConstantOptical, Material, MaterialRegistry
from dynameta.optics.fdtd_seam import (_eps_to_fdtd_layer, design_to_fdtd_layers,
                                       make_fdtd_optical_solver)

LAM = 1300e-9


def _fdtd_layer_eps(layer, lam_m):
    """The analytic eps(lam) the FDTDLayer represents (its convention: eps_inf - wp^2/(w^2 + i gamma w))."""
    w = 2.0 * math.pi * C_LIGHT / lam_m
    e = complex(layer.eps_inf)
    if layer.drude_wp_rad_s > 0.0:
        e = e - layer.drude_wp_rad_s ** 2 / (w ** 2 + 1j * layer.drude_gamma_rad_s * w)
    return e


@pytest.mark.parametrize("eps", [4.0 + 0j, 0.5 + 0j, 3.24 + 0.4j, 3.24 + 1.0j, -5.0 + 2.0j, -20.0 + 0.5j])
def test_drude_inversion_reproduces_eps_at_lambda(eps):
    """The inverted FDTDLayer must reproduce eps EXACTLY at lambda, with a stable background (eps_inf>=1
    except a pure positive-real dielectric, which is represented directly)."""
    L = _eps_to_fdtd_layer(200e-9, eps, LAM)
    assert abs(_fdtd_layer_eps(L, LAM) - eps) < 1e-6 * (abs(eps) + 1.0)
    pure_dielectric = (abs(eps.imag) < 1e-9 and eps.real > 0.0)
    assert pure_dielectric or L.eps_inf >= 1.0 - 1e-9


def test_lossless_dielectric_has_no_drude():
    L = _eps_to_fdtd_layer(100e-9, 4.0 + 0j, LAM)
    assert L.drude_wp_rad_s == 0.0 and abs(L.eps_inf - 4.0) < 1e-12


def _design(layer_specs):
    reg = MaterialRegistry()
    reg.add(Material("air", ConstantOptical(1.0 + 0j)))
    layers = []
    for k, (eps, th, incl) in enumerate(layer_specs):
        reg.add(Material("m%d" % k, ConstantOptical(complex(eps))))
        layers.append(Layer("s%d" % k, float(th), "m%d" % k, inclusions=list(incl)))
    stack = Stack(layers=layers, superstrate_material="air", substrate_material="air")
    return Design(name="t", unit_cell=UnitCell.square(220e-9), stack=stack, electrodes=[], materials=reg)


def test_layers_superstrate_first_order():
    """Stack lists bottom->top; the FDTD layers must come out superstrate-first (incidence order)."""
    d = _design([(4.0, 100e-9, []), (9.0, 200e-9, [])])       # s0 (eps4) bottom, s1 (eps9) top
    layers = design_to_fdtd_layers(d, LAM)
    assert len(layers) == 2
    assert abs(layers[0].eps_inf - 9.0) < 1e-9 and abs(layers[1].eps_inf - 4.0) < 1e-9  # top (s1) first


def test_inclusions_layer_raises():
    incl = [Inclusion(Circle(0.0, 0.0, 30e-9), "m0")]         # a lateral inclusion -> not laterally uniform
    d = _design([(4.0, 100e-9, incl)])
    with pytest.raises(NotImplementedError):
        design_to_fdtd_layers(d, LAM)


def test_non_vacuum_end_media_raises():
    d = _design([(4.0, 100e-9, [])])
    solver = make_fdtd_optical_solver(dim=2)
    with pytest.raises(NotImplementedError):
        solver(d, None, {}, LAM, 1.5 + 0j, 1.0 + 0j)          # glass superstrate -> Phase-0 guard fires


def test_bad_dim_raises():
    with pytest.raises(ValueError):
        make_fdtd_optical_solver(dim=4)
