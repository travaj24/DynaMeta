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
    # lumenairy is a REQUIRED dependency but loaded lazily at solver-call time: importing
    # dynameta (and the optics package) must not import lumenairy (which hard-depends on
    # matplotlib) -- the import-time hygiene contract
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
def test_pmm_uniform_stack_matches_tmm():
    from dynameta.optics.lumenairy_bridge import make_lumenairy_pmm_solver
    from dynameta.optics.tmm_reference import make_layered_tmm_solver
    d = _uniform_design()
    lam = 1.31e-6
    r_t = make_layered_tmm_solver()(d, None, {}, lam, 1.0 + 0j, 1.5 + 0j)
    r_p = make_lumenairy_pmm_solver(degree=10, n_orders=9)(d, None, {}, lam,
                                                           1.0 + 0j, 1.5 + 0j)
    assert r_p.R == pytest.approx(r_t.R, abs=1e-8)
    assert r_p.T == pytest.approx(r_t.T, abs=1e-8)
    assert r_p.r == pytest.approx(r_t.r, abs=1e-8)
    assert r_p.t is None                                  # PMM exposes no transmission Jones


@needs_lum
def test_pmm_lamellar_segments_partition():
    from dynameta.geometry import Inclusion, Layer
    from dynameta.geometry.cross_section import Rectangle
    from dynameta.optics.lumenairy_bridge import layer_to_pmm_segments
    per = 600e-9
    d = _uniform_design()
    lines = Inclusion(shape=Rectangle(per / 2.0, per / 2.0, 0.5 * per, per), material="hi")
    lay = Layer("g", 120e-9, "air", inclusions=[lines])
    segs = layer_to_pmm_segments(lay, d, 1.31e-6, per, per)
    assert sum(w for w, _ in segs) == pytest.approx(1.0, abs=0.0)  # exact unit sum
    assert any(e == pytest.approx(complex(4.0, 0.3)) for _, e in segs)
    assert any(e == pytest.approx(1.0 + 0j) for _, e in segs)


@needs_lum
def test_pmm_partial_y_rectangle_raises():
    from dynameta.geometry import Inclusion, Layer
    from dynameta.geometry.cross_section import Rectangle
    from dynameta.optics.lumenairy_bridge import layer_to_pmm_segments
    per = 600e-9
    d = _uniform_design()
    half = Inclusion(shape=Rectangle(per / 2.0, per / 4.0, 0.3 * per, 0.5 * per),
                     material="hi")
    lay = Layer("bad", 120e-9, "air", inclusions=[half])
    with pytest.raises(ValueError):
        layer_to_pmm_segments(lay, d, 1.31e-6, per, per)


@needs_lum
def test_callable_optical_dispersion_chain():
    from dynameta.optics.lumenairy_bridge import (CallableOptical,
                                                  optical_model_to_lumenairy_eps)
    fn = lambda wl: complex(2.0 + 0.1 * (wl / 1e-6))
    model = CallableOptical(fn)
    assert model.eps(1.5e-6) == pytest.approx(fn(1.5e-6))
    back = optical_model_to_lumenairy_eps(model)
    assert back(1.5e-6) == pytest.approx(fn(1.5e-6))


@needs_lum
def test_berreman_uniform_stack_matches_tmm():
    # the Berreman planar tier reduces EXACTLY to the transfer-matrix coating model on an
    # isotropic stack -- s and p, oblique, complex r/t
    from dynameta.core.layered import LayeredSlab, LayeredStack
    from dynameta.geometry.specs import OpticalSpec
    from dynameta.optics.lumenairy_bridge import BerremanLayeredSolver
    from dynameta.optics.tmm_reference import TmmLayeredSolver
    stk = LayeredStack(1.0 + 0j, 1.5 + 0j,
                       [LayeredSlab(120e-9, eps=complex(4.0, 0.3)),
                        LayeredSlab(200e-9, eps=complex(2.1, 0.0))])
    for pol, th in (("y", 0.0), ("p", 30.0)):
        opt = OpticalSpec(polarization=pol, incidence_angle_deg=th)
        r_b = BerremanLayeredSolver().solve(stk, 1.55e-6, opt)
        r_t = TmmLayeredSolver().solve(stk, 1.55e-6, opt)
        assert r_b.R == pytest.approx(r_t.R, abs=1e-10)
        assert r_b.T == pytest.approx(r_t.T, abs=1e-10)
        assert r_b.r == pytest.approx(r_t.r, abs=1e-9)
        assert r_b.t == pytest.approx(r_t.t, abs=1e-9)


@needs_lum
def test_berreman_uniaxial_decouples_per_axis():
    # an x-axis uniaxial slab decouples into independent n_e (x) / n_o (y) scalar problems --
    # the anisotropic tensor path the scalar TMM cannot represent
    import tmm
    from dynameta.core.layered import LayeredSlab, LayeredStack
    from dynameta.geometry.specs import OpticalSpec
    from dynameta.optics.lumenairy_bridge import BerremanLayeredSolver
    n_o, n_e, d, lam = 1.50, 1.74, 220e-9, 1.55e-6
    eps_t = np.diag([n_e ** 2, n_o ** 2, n_o ** 2]).astype(complex)
    stk = LayeredStack(1.0 + 0j, 1.5 + 0j,
                       [LayeredSlab(d, eps_tensor_cell=np.broadcast_to(eps_t, (1, 1, 3, 3)).copy())],
                       period_x_m=400e-9, period_y_m=400e-9)
    for pol, n_idx in (("x", n_e), ("y", n_o)):
        r_b = BerremanLayeredSolver().solve(stk, lam, OpticalSpec(polarization=pol,
                                                                 incidence_angle_deg=0.0))
        ref = tmm.coh_tmm("s", [1.0, n_idx, 1.5], [np.inf, d * 1e9, np.inf], 0.0, lam * 1e9)
        assert r_b.R == pytest.approx(ref["R"], abs=1e-9)
        assert r_b.T == pytest.approx(ref["T"], abs=1e-9)


@needs_lum
def test_berreman_patterned_layer_raises():
    # the planar-tier scope boundary: a patterned (inclusion) layer routes to RCWA, not Berreman
    from dynameta.geometry import Inclusion, Layer
    from dynameta.geometry.cross_section import Rectangle
    from dynameta.optics.lumenairy_bridge import design_to_berreman_layers
    per = 400e-9
    d = _uniform_design()
    pil = Inclusion(shape=Rectangle(per / 2.0, per / 2.0, 150e-9, 80e-9), material="hi")
    d.stack.layers.append(Layer("grating", 200e-9, "air", inclusions=[pil]))
    with pytest.raises(NotImplementedError):
        design_to_berreman_layers(d, 1.31e-6)


def _grating_design(period, *, pol="x"):
    from dynameta.geometry import Design, Inclusion, Layer, Stack, UnitCell
    from dynameta.geometry.cross_section import Rectangle
    from dynameta.geometry.specs import OpticalSpec
    from dynameta.materials import ConstantOptical, Material, MaterialRegistry
    reg = MaterialRegistry()
    reg.add(Material("air", ConstantOptical(1.0 + 0j)))
    reg.add(Material("glass", ConstantOptical(complex(1.5 ** 2))))
    reg.add(Material("ridge", ConstantOptical(6.0 + 0j)))
    ridge = Inclusion(shape=Rectangle(period / 2.0, period / 2.0, 0.5 * period, period),
                      material="ridge")
    return Design(name="g", unit_cell=UnitCell.square(period),
                  stack=Stack(layers=[Layer("grating", 180e-9, "air", inclusions=[ridge])],
                              superstrate_material="air", substrate_material="glass"),
                  electrodes=[], materials=reg,
                  optical=OpticalSpec(polarization=pol, incidence_angle_deg=0.0))


@needs_lum
def test_emt_rytov_tensor_harmonic_arithmetic():
    # the homogenized tensor is diag(harmonic, arithmetic, arithmetic) of the binary grating
    from dynameta.optics.lumenairy_bridge import rytov_tensor_for_layer
    d = _grating_design(1.55e-6 / 50.0)
    L = d.stack.layers[0]
    t = rytov_tensor_for_layer(L, d, 1.55e-6, d.unit_cell.period_x_m, d.unit_cell.period_y_m)
    eps_par = 0.5 * 6.0 + 0.5 * 1.0
    eps_perp = 1.0 / (0.5 / 6.0 + 0.5 / 1.0)
    assert t[1, 1] == pytest.approx(eps_par, abs=1e-12)
    assert t[0, 0] == pytest.approx(eps_perp, abs=1e-12)
    assert t[0, 0].real < t[1, 1].real                    # form birefringence: perp < par


@needs_lum
def test_emt_screen_converges_to_rcwa_subwavelength():
    # deeply sub-wavelength: the microsecond EMT screen agrees with the rigorous RCWA
    from dynameta.optics.lumenairy_bridge import (make_lumenairy_emt_screen_solver,
                                                  make_lumenairy_rcwa_solver)
    lam = 1.55e-6
    d = _grating_design(lam / 100.0, pol="x")
    r_emt = make_lumenairy_emt_screen_solver()(d, None, {}, lam, 1.0 + 0j, 1.5 + 0j)
    r_rig = make_lumenairy_rcwa_solver(n_orders=20)(d, None, {}, lam, 1.0 + 0j, 1.5 + 0j)
    assert abs(r_emt.R - r_rig.R) < 3e-3
