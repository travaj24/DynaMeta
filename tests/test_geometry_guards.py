"""Stack.features / Inclusion.taper capability guards + the sidewall-taper data model (2026-11-02,
workstream B2 of the lumenairy-5.50 bridge PR).

THE DEFECT (measured on 810086b, 300 nm cell, 120 + 80 nm lossless layers, one 100 x 100 nm Cu via
spanning z 0..200 nm, 1.31 um, normal incidence): the TMM solver / design_layer_stack /
layered_stack_from_design, the RCWA, 1-D PMM, 2-D PMM, Berreman and EMT-screen bridges,
design_to_fdtd_layers and the 2-D FDTD solver ALL ran and returned R/T bit-identical (repr) to the
via-free design; LayeredDevsimBuilder's region plan and Stacked3DSpec.from_design came back via-free
too. Only LayeredOpticalBuilder refused (audit F-14). Every one of them now raises
UnsupportedGeometryError (a NotImplementedError) through dynameta.geometry.guards, and the same for a
tapered inclusion -- except the PMM2D PURE engine, which since 2026-11-02 (workstream B3) BUILDS a
tapered Rectangle pillar via lumenairy's add_tapered_pillar (tests/test_lumenairy_pmm2d_taper_wrap.py);
its hybrid engine keeps refusing (the 'pmm2d_hybrid' row below).

BYTE-IDENTITY: a plain (untapered, feature-free) design is unchanged -- the cache key, the TMM and
RCWA-bridge R/T/phase and the FEM builder's mesh were measured equal (repr) before/after the change;
the values pinned below are the 810086b ones.

Run: python -m pytest -p no:benchmark -q tests/test_geometry_guards.py
"""
import importlib.util
import math
import warnings

import pytest

from dynameta.geometry import (Design, Feature, Inclusion, Layer, Stack, Taper, UnitCell,
                               UnsupportedGeometryError, require_supported_geometry,
                               require_untapered_layer, tapered_inclusions)
from dynameta.geometry.cross_section import Circle, Ellipse, Polygon, Rectangle, RegularPolygon
from dynameta.geometry.specs import Mesh3DSpec, OpticalSpec
from dynameta.materials import ConstantOptical, Material, MaterialRegistry

HAVE_LUM = importlib.util.find_spec("lumenairy") is not None
HAVE_NG = importlib.util.find_spec("ngsolve") is not None
HAVE_DEVSIM = importlib.util.find_spec("devsim") is not None
needs_lum = pytest.mark.skipif(not HAVE_LUM, reason="lumenairy not installed")
needs_ng = pytest.mark.skipif(not HAVE_NG, reason="ngsolve not installed")
needs_devsim = pytest.mark.skipif(not HAVE_DEVSIM, reason="devsim not installed")

P = 400e-9
LAM = 1.31e-6


# ------------------------------------------------------------------------------------ fixtures
def _reg():
    r = MaterialRegistry()
    r.add(Material("air", ConstantOptical(1.0 + 0j)))
    r.add(Material("hi", ConstantOptical(complex(6.0, 0.2))))
    r.add(Material("lo", ConstantOptical(complex(2.1, 0.0))))
    r.add(Material("glass", ConstantOptical(complex(1.5 ** 2))))
    r.add(Material("Cu", ConstantOptical(-100 + 8j), is_metal=True))
    return r


def _mesh():
    return Mesh3DSpec(pml_thk_m=300e-9, superstrate_buffer_m=300e-9, substrate_buffer_m=300e-9,
                      maxh_superstrate_m=250e-9, maxh_substrate_m=250e-9, maxh_pml_m=250e-9,
                      maxh_background_m=70e-9, maxh_inclusion_m=70e-9)


def _pillar(taper=None):
    """60 nm 'hi' film under a 150 nm air layer holding a 160 x 120 nm 'hi' pillar (the byte-identity
    reference of the module docstring); `taper` is attached to the pillar."""
    inc = Inclusion(Rectangle(P / 2, P / 2, 160e-9, 120e-9), "hi", taper=taper)
    return Design(name="pil", unit_cell=UnitCell.square(P),
                  stack=Stack(layers=[Layer("film", 60e-9, "hi"),
                                      Layer("pil", 150e-9, "air", inclusions=[inc])],
                              superstrate_material="air", substrate_material="glass"),
                  electrodes=[], materials=_reg(), mesh_3d=_mesh(),
                  optical=OpticalSpec(polarization="y", incidence_angle_deg=0.0))


def _uniform(features=()):
    return Design(name="uni", unit_cell=UnitCell.square(P),
                  stack=Stack(layers=[Layer("a", 60e-9, "hi"), Layer("b", 90e-9, "glass")],
                              superstrate_material="air", substrate_material="glass",
                              features=list(features)),
                  electrodes=[], materials=_reg(),
                  optical=OpticalSpec(polarization="y", incidence_angle_deg=0.0))


def _via():
    return Feature("via", Rectangle(P / 2, P / 2, 100e-9, 100e-9), "Cu", 0.0, 150e-9)


def _tapered_pillar():
    top = Rectangle(P / 2, P / 2, 160e-9, 120e-9)
    return _pillar(taper=Taper.from_sidewall_angle(top, 150e-9, 2.0))


def _bad_designs():
    return [("feature", _uniform([_via()]), r"Stack\.features.*'via'"),
            ("taper", _tapered_pillar(), r"Inclusion\.taper.*pil\[0\]")]


# =========================================================================== taper data model
def test_taper_defaults_to_none_and_untapered_inclusion_is_unchanged():
    a = Inclusion(Rectangle(0.0, 0.0, 1e-7, 1e-7), "hi")
    assert a.taper is None
    assert a == Inclusion(Rectangle(0.0, 0.0, 1e-7, 1e-7), "hi", 0, None)
    assert tapered_inclusions(_pillar()) == []


def test_from_sidewall_angle_matches_house_convention():
    # memory feedback_taper_not_slant (measured 2026-10-06, exp11 c1): a 160 nm finger, 2.00 deg per
    # face, is 126.5 nm at its base -> the face moves in 16.75 nm, i.e. over 479.65 nm of height.
    top = Rectangle(1e-6, 2e-6, 160e-9, 160e-9)
    L = 16.75e-9 / math.tan(math.radians(2.0))
    t = Taper.from_sidewall_angle(top, L, 2.0)
    assert t.bottom.width_m == pytest.approx(126.5e-9, abs=1e-15)
    assert t.bottom.height_m == pytest.approx(126.5e-9, abs=1e-15)
    assert t.bottom.center_m() == top.center_m()                     # about its OWN centre
    Inclusion(top, "Cu", taper=t)                                    # validates


def test_from_sidewall_angle_other_kinds():
    d = 100e-9 * math.tan(math.radians(5.0))
    c = Taper.from_sidewall_angle(Circle(0.0, 0.0, 80e-9), 100e-9, 5.0).bottom
    assert c.radius_m == pytest.approx(80e-9 - d, rel=1e-14)
    e = Taper.from_sidewall_angle(Ellipse(0.0, 0.0, 80e-9, 50e-9), 100e-9, 5.0).bottom
    assert (e.rx_m, e.ry_m) == pytest.approx((80e-9 - d, 50e-9 - d), rel=1e-14)
    top = RegularPolygon(0.0, 0.0, 80e-9, 6, 10.0)
    h = Taper.from_sidewall_angle(top, 100e-9, 5.0).bottom
    # exact parallel offset: the APOTHEM moves in by d
    assert h.radius_m * math.cos(math.pi / 6) == pytest.approx(80e-9 * math.cos(math.pi / 6) - d, rel=1e-13)
    assert (h.n_sides, h.rotation_deg) == (6, 10.0)
    with pytest.raises(ValueError, match="explicitly"):
        Taper.from_sidewall_angle(Polygon([(0, 0), (1e-7, 0), (0, 1e-7)]), 100e-9, 2.0)
    with pytest.raises(ValueError, match="closes"):
        Taper.from_sidewall_angle(Rectangle(0.0, 0.0, 20e-9, 20e-9), 500e-9, 2.0)   # 2 x 17.5 nm > 20 nm
    with pytest.raises(ValueError, match="thickness"):
        Taper.from_sidewall_angle(Rectangle(0.0, 0.0, 20e-9, 20e-9), 0.0, 2.0)


def test_wider_bottom_refused_unless_opted_in():
    top = Rectangle(0.0, 0.0, 100e-9, 100e-9)
    with pytest.raises(ValueError, match="WIDER"):
        Inclusion(top, "Cu", taper=Taper.from_sidewall_angle(top, 100e-9, -2.0))
    with pytest.raises(ValueError, match="WIDER"):                  # wider in ONE axis is still wider
        Inclusion(top, "Cu", taper=Taper(Rectangle(0.0, 0.0, 90e-9, 101e-9)))
    inc = Inclusion(top, "Cu", taper=Taper.from_sidewall_angle(top, 100e-9, -2.0, allow_wider_bottom=True))
    assert inc.taper.bottom.width_m > top.width_m


def test_slant_twist_and_morph_refused():
    top = Rectangle(200e-9, 200e-9, 100e-9, 100e-9)
    with pytest.raises(ValueError, match="SLANT"):                  # off-centre bottom = slant, ALWAYS
        Inclusion(top, "Cu", taper=Taper(Rectangle(205e-9, 200e-9, 80e-9, 80e-9)))
    with pytest.raises(ValueError, match="SLANT"):
        Inclusion(top, "Cu", taper=Taper(Rectangle(205e-9, 200e-9, 80e-9, 80e-9), allow_wider_bottom=True))
    with pytest.raises(ValueError, match="SAME kind"):
        Inclusion(top, "Cu", taper=Taper(Circle(200e-9, 200e-9, 40e-9)))
    hexa = RegularPolygon(0.0, 0.0, 80e-9, 6, 0.0)
    with pytest.raises(ValueError, match="twist"):
        Inclusion(hexa, "Cu", taper=Taper(RegularPolygon(0.0, 0.0, 70e-9, 6, 10.0)))
    Inclusion(hexa, "Cu", taper=Taper(RegularPolygon(0.0, 0.0, 70e-9, 6, 60.0)))   # one period: same
    with pytest.raises(ValueError, match="sides"):
        Inclusion(hexa, "Cu", taper=Taper(RegularPolygon(0.0, 0.0, 70e-9, 8, 0.0)))
    with pytest.raises(TypeError):
        Inclusion(top, "Cu", taper=Rectangle(200e-9, 200e-9, 80e-9, 80e-9))
    with pytest.raises(TypeError):
        Taper(bottom=(1.0, 2.0))


def test_polygon_taper_rules():
    sq = [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)]
    top = Polygon([(50e-9 * x, 50e-9 * y) for x, y in sq])
    Inclusion(top, "Cu", taper=Taper(Polygon([(40e-9 * x, 40e-9 * y) for x, y in sq])))
    Inclusion(top, "Cu", taper=Taper(Polygon(list(top.points_m))))  # zero-angle point of a sweep
    with pytest.raises(ValueError, match="vertices"):
        Inclusion(top, "Cu", taper=Taper(Polygon([(-4e-8, -4e-8), (4e-8, -4e-8), (0.0, 8e-8)])))
    with pytest.raises(ValueError, match="WIDER"):
        Inclusion(top, "Cu", taper=Taper(Polygon([(60e-9 * x, 60e-9 * y) for x, y in sq])))


def test_device_symmetry_sees_the_bottom_section():
    top = Rectangle(P / 2, P / 2, 120e-9, 120e-9)                  # square (c4v) top ...
    d = Design(name="s", unit_cell=UnitCell.square(P),
               stack=Stack(layers=[Layer("pil", 150e-9, "air",
                                         inclusions=[Inclusion(top, "hi",
                                                               taper=Taper(Rectangle(P / 2, P / 2, 100e-9,
                                                                                     80e-9)))])],
                           superstrate_material="air", substrate_material="glass"),
               electrodes=[], materials=_reg())
    assert d.device_symmetry() == "c2v"                              # ... over a rectangular base
    d.stack.layers[0].inclusions[0].taper = None
    assert d.device_symmetry() == "c4v"


# ================================================================================== cache key
def test_cache_key_untapered_is_byte_identical_and_taper_is_keyed():
    from dynameta.cache import _key
    # measured at 810086b (before Inclusion.taper existed): untapered keys must not move, or every
    # existing on-disk OpticalSolverCache would silently go cold
    assert _key(_pillar(), {}, LAM, 1.0, 1.5, "t") == "keb8125610c11767a53185004333b3b755b13f3ed"
    assert _key(_uniform(), {}, LAM, 1.0, 1.5, "t") == "kdcc22f022f85832787ce5e7e7addbe1a6c6427fc"
    top = Rectangle(P / 2, P / 2, 160e-9, 120e-9)
    k0 = _key(_pillar(), {}, LAM, 1.0, 1.5, "t")
    k2 = _key(_pillar(Taper.from_sidewall_angle(top, 150e-9, 2.0)), {}, LAM, 1.0, 1.5, "t")
    k3 = _key(_pillar(Taper.from_sidewall_angle(top, 150e-9, 3.0)), {}, LAM, 1.0, 1.5, "t")
    k0b = _key(_pillar(Taper(top)), {}, LAM, 1.0, 1.5, "t")         # zero-angle taper: still its own key
    assert len({k0, k2, k3, k0b}) == 4
    assert k2 == _key(_pillar(Taper.from_sidewall_angle(top, 150e-9, 2.0)), {}, LAM, 1.0, 1.5, "t")


# ======================================================================================= helper
def test_helper_contract():
    require_supported_geometry(_pillar(), "X")                       # plain design passes
    require_supported_geometry(_pillar().stack, "X")                 # a bare Stack too
    with pytest.raises(UnsupportedGeometryError, match=r"^MyBackend does not implement Stack\.features"):
        require_supported_geometry(_uniform([_via()]), "MyBackend")
    with pytest.raises(NotImplementedError):                         # subclass: old except-sites hold
        require_supported_geometry(_uniform([_via()]), "MyBackend")
    require_supported_geometry(_uniform([_via()]), "MyBackend", features=True)
    with pytest.raises(UnsupportedGeometryError, match=r"MyBackend does not implement Inclusion\.taper.*pil\[0\]"):
        require_supported_geometry(_tapered_pillar(), "MyBackend")
    require_supported_geometry(_tapered_pillar(), "MyBackend", taper=True)
    assert tapered_inclusions(_tapered_pillar()) == [("pil", 0)]
    require_untapered_layer(_pillar().stack.layers[1], "R")
    with pytest.raises(UnsupportedGeometryError, match=r"R does not implement Inclusion\.taper"):
        require_untapered_layer(_tapered_pillar().stack.layers[1], "R")
    # the guard reads, never mutates
    d = _tapered_pillar()
    before = repr(d)
    with pytest.raises(UnsupportedGeometryError):
        require_supported_geometry(d, "X")
    assert repr(d) == before


# ============================================================================= backend refusals
def _call(fn):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn()


def _backends():
    """(name, callable(design)) for every Design-consuming backend entry point."""
    out = []

    def tmm_solver(d):
        from dynameta.optics.tmm_reference import make_layered_tmm_solver
        return make_layered_tmm_solver()(d, None, {}, LAM, 1.0 + 0j, 1.5 + 0j)

    def tmm_dls(d):
        from dynameta.optics.tmm_reference import design_layer_stack
        return design_layer_stack(d, LAM)

    def tmm_lsfd(d):
        from dynameta.optics.tmm_reference import layered_stack_from_design
        return layered_stack_from_design(d, LAM)

    def fdtd_layers(d):
        from dynameta.optics.fdtd_seam import design_to_fdtd_layers
        return design_to_fdtd_layers(d, LAM)

    def fdtd_disp(d):
        from dynameta.optics.fdtd_seam import _design_to_fdtd_layers_dispersive
        return _design_to_fdtd_layers_dispersive(d, 1.2e-6, 1.4e-6)

    def fdtd_lateral(d):
        from dynameta.optics.fdtd_seam import make_structured_lateral
        return make_structured_lateral(d, LAM)

    def fdtd_solver(d):
        from dynameta.optics.fdtd_seam import make_fdtd_optical_solver
        return make_fdtd_optical_solver(dim=3, resolution=8)(d, None, {}, LAM, 1.0 + 0j, 1.5 + 0j)

    def fdtd_sweep(d):
        from dynameta.optics.fdtd_seam import fdtd_sweep_spectrum
        return fdtd_sweep_spectrum(d, lambda_min_m=1.2e-6, lambda_max_m=1.4e-6, resolution=8)

    out += [("tmm_solver", tmm_solver), ("design_layer_stack", tmm_dls),
            ("layered_stack_from_design", tmm_lsfd), ("design_to_fdtd_layers", fdtd_layers),
            ("fdtd_dispersive", fdtd_disp), ("make_structured_lateral", fdtd_lateral),
            ("fdtd_solver", fdtd_solver), ("fdtd_sweep_spectrum", fdtd_sweep)]
    if HAVE_LUM:
        from dynameta.optics import lumenairy_bridge as lb

        def mk(factory, **kw):
            return lambda d: factory(**kw)(d, None, {}, LAM, 1.0 + 0j, 1.5 + 0j)

        out += [("rcwa", mk(lb.make_lumenairy_rcwa_solver, n_orders=3)),
                ("pmm", mk(lb.make_lumenairy_pmm_solver, degree=6, n_orders=5)),
                ("pmm2d", mk(lb.make_lumenairy_pmm2d_solver, n_modes=4)),
                ("pmm2d_hybrid", mk(lb.make_lumenairy_pmm2d_solver, engine="hybrid", degree=5,
                                    n_orders=3)),
                ("berreman", mk(lb.make_lumenairy_berreman_solver)),
                ("emt_screen", mk(lb.make_lumenairy_emt_screen_solver)),
                ("homogenize_lamellar_layers", lambda d: lb.homogenize_lamellar_layers(d, LAM))]
    if HAVE_NG:
        def fem(d):
            from dynameta.optics.ngsolve_layered import LayeredOpticalBuilder
            return LayeredOpticalBuilder(d).build()
        out.append(("fem_builder", fem))
    return out


@pytest.mark.parametrize("kind", ["feature", "taper"])
@pytest.mark.parametrize("name,fn", _backends(), ids=[b[0] for b in _backends()])
def test_backend_refuses_unsupported_geometry(name, fn, kind):
    _, d, pat = next(b for b in _bad_designs() if b[0] == kind)
    if name == "pmm2d" and kind == "taper":
        # the PMM2D PURE engine BUILDS this tapered Rectangle (2026-11-02, workstream B3: lumenairy
        # add_tapered_pillar, gated in tests/test_lumenairy_pmm2d_taper_wrap.py) -- assert it is
        # translated, not refused (build only: the 8-slice solve is the dedicated test's job)
        from dynameta.optics.lumenairy_bridge import design_to_pmm2d_stack
        _, names = design_to_pmm2d_stack(d, LAM, n_modes=4)
        assert names == ["pil"] * 8 + ["film"]
        return
    with pytest.raises(UnsupportedGeometryError, match=pat):
        _call(lambda: fn(d))


def test_layer_level_helpers_refuse_taper():
    import numpy as np
    from dynameta.optics.rasterize import cell_axes, layer_eps_cell
    d = _tapered_pillar()
    X, Y = np.meshgrid(*cell_axes(8, 8, P, P), indexing="ij")
    with pytest.raises(UnsupportedGeometryError, match="layer_eps_cell"):
        layer_eps_cell(d.stack.layers[1], X, Y, LAM, d.materials, {})
    eps = layer_eps_cell(_pillar().stack.layers[1], X, Y, LAM, d.materials, {})   # untapered paints
    assert np.any(eps != eps[0, 0])
    if HAVE_LUM:
        from dynameta.optics.lumenairy_bridge import rytov_tensor_for_layer
        with pytest.raises(UnsupportedGeometryError, match="rytov_tensor_for_layer"):
            rytov_tensor_for_layer(d.stack.layers[1], d, LAM, P, P)


def test_feature_refusal_keeps_the_fem_wording():
    # tests/test_mesh_periodicity.py::test_f14 matches "Stack.features" on a NotImplementedError; the
    # routed message must keep both (and name the builder)
    pytest.importorskip("ngsolve")
    from dynameta.optics.ngsolve_layered import LayeredOpticalBuilder
    with pytest.raises(NotImplementedError, match=r"LayeredOpticalBuilder does not implement Stack\.features"):
        LayeredOpticalBuilder(_uniform([_via()])).build()


# -------------------------------------------------------------------------------- DEVSIM
def _carrier_design(features=(), taper=None):
    from dynameta.geometry import Electrode
    from dynameta.materials import M_E, TransportModel
    r = _reg()
    r.add(Material("semi", optical=ConstantOptical(4.0 + 0j),
                   transport=TransportModel(n_bg_m3=4e26, eps_static=9.5,
                                            dos_mass_kg_of_n_m3=lambda n: 0.35 * M_E,
                                            physics="equilibrium")))
    r.add(Material("ox", ConstantOptical(2.1 + 0j), eps_static_dc=3.9))
    patch = Rectangle(100e-9, 100e-9, 80e-9, 80e-9)
    layers = [Layer("metalpad", 20e-9, "Cu"), Layer("ito", 12e-9, "semi"), Layer("ox", 10e-9, "ox"),
              Layer("patch", 30e-9, "air", inclusions=[Inclusion(patch, "Cu", taper=taper)])]
    el = [Electrode("gnd", "metalpad", "full", role="ground"),
          Electrode("gate", "patch", patch, role="biased")]
    return Design(name="c", unit_cell=UnitCell.square(200e-9),
                  stack=Stack(layers=layers, superstrate_material="air", substrate_material="air",
                              features=list(features)),
                  electrodes=el, materials=r)


@needs_devsim
@pytest.mark.parametrize("builder", ["layered", "3d"])
def test_devsim_builders_refuse(builder):
    if builder == "layered":
        from dynameta.carriers.devsim_layered import LayeredDevsimBuilder as B
        build = B
    else:
        from dynameta.carriers.devsim_3d import Stacked3DSpec
        build = Stacked3DSpec.from_design
    build(_carrier_design())                                         # the plain design still plans
    via = Feature("via", Rectangle(100e-9, 100e-9, 40e-9, 40e-9), "Cu", 0.0, 32e-9)
    with pytest.raises(UnsupportedGeometryError, match=r"Stack\.features"):
        build(_carrier_design(features=[via]))
    tap = Taper.from_sidewall_angle(Rectangle(100e-9, 100e-9, 80e-9, 80e-9), 30e-9, 2.0)
    with pytest.raises(UnsupportedGeometryError, match=r"Inclusion\.taper.*patch\[0\]"):
        build(_carrier_design(taper=tap))


# ======================================================================= plain-design byte identity
def test_plain_design_results_unchanged_tmm():
    from dynameta.optics.tmm_reference import make_layered_tmm_solver
    r = make_layered_tmm_solver()(_uniform(), None, {}, LAM, 1.0 + 0j, 1.5 + 0j)
    # 810086b: repr-identical before/after on the dev box; pinned to 1e-13 for cross-platform BLAS
    assert (r.R, r.T, r.phase_deg) == pytest.approx(
        (0.21836293389267467, 0.7553376188167926, -163.10821764266944), abs=1e-13)


@needs_lum
def test_plain_design_results_unchanged_rcwa_bridge():
    from dynameta.optics.lumenairy_bridge import make_lumenairy_rcwa_solver
    r = make_lumenairy_rcwa_solver(n_orders=5)(_pillar(), None, {}, LAM, 1.0 + 0j, 1.5 + 0j)
    assert (r.R, r.T) == pytest.approx((0.1910304737839642, 0.7791097782748155), abs=1e-12)
    # the phase is in DEGREES: one shared abs=1e-12 bounded it at 1.7e-14 rad, below the BLAS
    # reduction-order spread (capping BLAS at 2 threads moves it 1.7e-11 deg = 3e-13 rad,
    # measured 2026-11-02, R/T unmoved at 1e-12) -- a units mismatch in the pin, not a drift
    assert r.phase_deg == pytest.approx(-123.94232228780875, abs=1e-9)


@needs_ng
def test_plain_design_fem_mesh_unchanged():
    import ngsolve
    from dynameta.optics.ngsolve_layered import LayeredOpticalBuilder
    g = LayeredOpticalBuilder(_pillar()).build()
    if ngsolve.__version__ != "6.2.2608":
        pytest.skip("element counts pinned on ngsolve 6.2.2608 (got {})".format(ngsolve.__version__))
    assert (g.mesh.ne, g.mesh.nv) == (1073, 279)                     # 810086b, measured
