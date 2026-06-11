"""Lumenairy RCWA bridge: fast unit tests (the rigorous multi-oracle gates live in
validation/lumenairy_rcwa_bridge.py). Skipped wholesale when lumenairy is not installed;
the import-light test runs regardless (the bridge must not drag lumenairy/matplotlib into
a bare dynameta import)."""
import importlib.util
import subprocess
import sys

import numpy as np
import pytest

HAVE_LUM = importlib.util.find_spec("lumenairy") is not None
needs_lum = pytest.mark.skipif(not HAVE_LUM, reason="lumenairy not installed")


def test_dynameta_import_stays_lumenairy_free():
    # the bridge is opt-in: importing dynameta (and the optics package) must not import
    # lumenairy (which hard-depends on matplotlib) -- CI installability contract
    code = ("import dynameta, dynameta.optics, sys; "
            "assert 'lumenairy' not in sys.modules, 'lumenairy leaked into base import'; "
            "print('ok')")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr


def _uniform_design(pol="y", theta=0.0):
    from dynameta.geometry import Design, Layer, Stack, UnitCell
    from dynameta.geometry.specs import OpticalSpec
    from dynameta.materials import ConstantOptical, Material, MaterialRegistry
    reg = MaterialRegistry()
    reg.add(Material("air", ConstantOptical(1.0 + 0j)))
    reg.add(Material("hi", ConstantOptical(complex(4.0, 0.3))))
    reg.add(Material("glass", ConstantOptical(complex(1.5 ** 2))))
    return Design(name="t", unit_cell=UnitCell.square(300e-9),
                  stack=Stack(layers=[Layer("a", 120e-9, "hi")],
                              superstrate_material="air", substrate_material="glass"),
                  electrodes=[], materials=reg,
                  optical=OpticalSpec(polarization=pol, incidence_angle_deg=theta))


@needs_lum
def test_uniform_stack_matches_tmm():
    from dynameta.optics.lumenairy_bridge import make_lumenairy_rcwa_solver
    from dynameta.optics.tmm_reference import make_layered_tmm_solver
    d = _uniform_design()
    lam = 1.31e-6
    r_t = make_layered_tmm_solver()(d, None, {}, lam, 1.0 + 0j, 1.5 + 0j)
    r_r = make_lumenairy_rcwa_solver(n_orders=2)(d, None, {}, lam, 1.0 + 0j, 1.5 + 0j)
    assert r_r.R == pytest.approx(r_t.R, abs=1e-10)
    assert r_r.T == pytest.approx(r_t.T, abs=1e-10)
    assert r_r.phase_deg == pytest.approx(r_t.phase_deg, abs=1e-8)
    assert r_r.R_flux == pytest.approx(r_r.R)            # RCWA R is already a flux quantity


@needs_lum
def test_structured_slab_solver_consumes_eps_cell():
    # the structured LayeredSlab specs get their FIRST consumer here: a lossless 1-D-like
    # grating eps_cell must run and conserve energy
    from dynameta.core.layered import LayeredSlab, LayeredStack
    from dynameta.geometry.specs import OpticalSpec
    from dynameta.optics.lumenairy_bridge import LumenairyStackSolver
    sx = 4 * 5 + 1
    cell = np.full((sx, 1), 1.0 + 0j)
    cell[: sx // 2, 0] = 4.0 + 0j
    stk = LayeredStack(1.0 + 0j, 1.5 + 0j, [LayeredSlab(180e-9, eps_cell=cell)],
                       period_x_m=600e-9, period_y_m=600e-9)
    res = LumenairyStackSolver(n_orders=5).solve(
        stk, 1.31e-6, OpticalSpec(polarization="y", incidence_angle_deg=0.0))
    assert 0.0 <= res.R <= 1.0 and 0.0 <= res.T <= 1.0
    assert res.R + res.T == pytest.approx(1.0, abs=1e-9)  # lossless grating closes


@needs_lum
def test_bad_polarization_raises():
    from dynameta.optics.lumenairy_bridge import make_lumenairy_rcwa_solver
    d = _uniform_design(pol="y")
    object.__setattr__(d.optical, "polarization", "zz") if hasattr(
        d.optical, "__dataclass_fields__") else None
    try:
        d.optical.polarization = "zz"
    except Exception:
        pytest.skip("OpticalSpec immutable; covered by construction-time validation")
    with pytest.raises(ValueError):
        make_lumenairy_rcwa_solver(n_orders=2)(d, None, {}, 1.31e-6, 1.0 + 0j, 1.5 + 0j)


@needs_lum
def test_round_trip_design_geometry():
    from dynameta.optics.lumenairy_bridge import design_to_rcwa_stack, rcwa_stack_to_design
    d0 = _uniform_design()
    stk, _ = design_to_rcwa_stack(d0, 1.31e-6, n_orders=2)
    d1 = rcwa_stack_to_design(stk)
    assert len(d1.stack.layers) == len(d0.stack.layers)
    for a, b in zip(d0.stack.layers, d1.stack.layers):
        assert b.thickness_m == pytest.approx(a.thickness_m, abs=1e-18)
        assert complex(d1.materials.get(b.background_material).eps(1.31e-6)) == \
            pytest.approx(complex(d0.materials.get(a.background_material).eps(1.31e-6)))


@needs_lum
def test_callable_optical_dispersion_chain():
    from dynameta.optics.lumenairy_bridge import (CallableOptical,
                                                  optical_model_to_lumenairy_eps)
    fn = lambda wl: complex(2.0 + 0.1 * (wl / 1e-6))
    model = CallableOptical(fn)
    assert model.eps(1.5e-6) == pytest.approx(fn(1.5e-6))
    back = optical_model_to_lumenairy_eps(model)
    assert back(1.5e-6) == pytest.approx(fn(1.5e-6))
