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
def test_pmm2d_uniform_stack_matches_tmm_both_engines():
    # the 2-D crossed-PMM seam (audit 8.1-4) on a uniform lossy stack: both engines
    # reduce to TMM (R/T and the complex r, pinning layer order + Jones conventions);
    # neither exposes a transmission Jones (t is None, the 1-D PMM precedent); the
    # hybrid absorption surface fills per_region_absorption and closes the budget.
    # The rigorous patch-referee gates live in validation/lumenairy_pmm2d_bridge.py.
    from dynameta.optics.lumenairy_bridge import make_lumenairy_pmm2d_solver
    from dynameta.optics.tmm_reference import make_layered_tmm_solver
    d = _uniform_design()
    lam = 1.31e-6
    r_t = make_layered_tmm_solver()(d, None, {}, lam, 1.0 + 0j, 1.5 + 0j)
    r_h = make_lumenairy_pmm2d_solver(engine="hybrid", degree=7, n_orders=2)(
        d, None, {}, lam, 1.0 + 0j, 1.5 + 0j)
    r_p = make_lumenairy_pmm2d_solver(engine="pure", n_modes=5, n_orders=3)(
        d, None, {}, lam, 1.0 + 0j, 1.5 + 0j)
    for r_b in (r_h, r_p):
        assert r_b.R == pytest.approx(r_t.R, abs=1e-8)
        assert r_b.T == pytest.approx(r_t.T, abs=1e-8)
        assert r_b.r == pytest.approx(r_t.r, abs=1e-8)
        assert r_b.t is None                     # no transmission Jones in 2-D PMM
    r_a = make_lumenairy_pmm2d_solver(engine="hybrid", degree=7, n_orders=2,
                                      absorption=True)(d, None, {}, lam,
                                                       1.0 + 0j, 1.5 + 0j)
    assert set(r_a.per_region_absorption) == {"a"}
    assert r_a.A_independent == pytest.approx(r_a.A, abs=1e-9)


def test_pmm2d_pure_translation_and_guards():
    # lumenairy-FREE surface of the 2-D PMM bridge: union-grid inference + the
    # analytic Rectangle -> (N, N) segment painting, and the construction-time
    # guards (absorption needs the hybrid engine; engine name is validated)
    import pytest as _pytest
    from dynameta.optics.lumenairy_bridge import (layer_to_pure_cell,
                                                  make_lumenairy_pmm2d_solver,
                                                  pure_union_grid_n)
    per = 600e-9
    d = _grating_design(per)                     # ridge walls at per/4 and 3 per/4
    assert pure_union_grid_n(d) == 4
    cell = layer_to_pure_cell(d.stack.layers[0], d, 1.31e-6, per, per, 4)
    assert cell.shape == (4, 4)
    assert np.allclose(cell[1:3, :], 6.0 + 0j)   # ridge spans full y (lamellar)
    assert np.allclose(cell[0, :], 1.0 + 0j) and np.allclose(cell[3, :], 1.0 + 0j)
    with _pytest.raises(NotImplementedError):    # pure retains no internals
        make_lumenairy_pmm2d_solver(engine="pure", absorption=True)
    with _pytest.raises(ValueError):
        make_lumenairy_pmm2d_solver(engine="fmm")


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
def test_bridge_conical_raises_every_polarization():
    # audit C4-2: conical incidence (azimuth != 0) must raise for EVERY polarization -- the
    # bridges' lab-basis rows are phi-dependent s/p mixtures, not the FEM's rotated s/p
    # (probe: 'y' at theta=30/phi=45 on bare glass returned R 32% low, silently; at phi=90
    # exactly the orthogonal polarization). The old guard covered only 'p'. phi=0 solves.
    # BERREMAN is exempt since audit 8.1-6 (commit f0986df): a PLANAR stack is z-rotation
    # covariant, so its bridge synthesizes the exact rotated s/p at any azimuth and SOLVES
    # (pinned in test_berreman_oop_oblique_first_class); RCWA -- whose lattice breaks the
    # covariance -- still raises.
    from dynameta.geometry.specs import OpticalSpec
    from dynameta.optics.lumenairy_bridge import (make_lumenairy_berreman_solver,
                                                  make_lumenairy_rcwa_solver)
    # 'x' cannot reach the bridge at oblique (OpticalSpec rejects it at construction),
    # so 'p' and 'y' are the full reachable set
    for pol in ("p", "y"):
        d = _uniform_design()
        d.optical = OpticalSpec(polarization=pol, incidence_angle_deg=30.0, azimuth_deg=20.0)
        with pytest.raises(NotImplementedError):
            make_lumenairy_rcwa_solver(n_orders=3)(d, None, {}, 1.31e-6, 1.0 + 0j, 1.5 + 0j)
        r = make_lumenairy_berreman_solver()(d, None, {}, 1.31e-6, 1.0 + 0j, 1.5 + 0j)
        assert 0.0 <= r.R <= 1.0 and 0.0 <= r.T <= 1.0    # berreman conical solves (8.1-6)
    d = _uniform_design()
    d.optical = OpticalSpec(polarization="p", incidence_angle_deg=30.0, azimuth_deg=0.0)
    make_lumenairy_rcwa_solver(n_orders=3)(d, None, {}, 1.31e-6, 1.0 + 0j, 1.5 + 0j)  # phi=0 ok


@needs_lum
def test_bridge_bottom_incidence_raises():
    # incidence_side='bottom' is a legal OpticalSpec value the bridges cannot honor (top-only); raise
    from dynameta.geometry.specs import OpticalSpec
    from dynameta.optics.lumenairy_bridge import (make_lumenairy_berreman_solver,
                                                  make_lumenairy_pmm_solver,
                                                  make_lumenairy_rcwa_solver)
    for mk in (lambda: make_lumenairy_rcwa_solver(n_orders=3),
               lambda: make_lumenairy_pmm_solver(degree=8, n_orders=7),
               make_lumenairy_berreman_solver):
        d = _uniform_design()
        d.optical = OpticalSpec(polarization="y", incidence_angle_deg=0.0, incidence_side="bottom")
        with pytest.raises(NotImplementedError):
            mk()(d, None, {}, 1.31e-6, 1.0 + 0j, 1.5 + 0j)


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


# ---- BOR-PMM (axisymmetric) spec construction guards: solver-free, run regardless of lumenairy ----
# The BorLayer/BorStackSpec dataclasses validate at construction (no lumenairy needed); only solve_bor
# lazily imports it. These pin the guards + the lazy-import contract for the BOR backend.

def test_bor_layer_needs_exactly_one_profile():
    from dynameta.optics.lumenairy_bridge import BorLayer
    BorLayer(thickness_m=0.5e-6, eps=2.25 + 0j)                       # uniform: ok
    BorLayer(thickness_m=0.5e-6, rings=(3e-6, 0.5, 2.4 + 0j, 1.4 + 0j))   # ring grating: ok
    BorLayer(thickness_m=0.5e-6, eps_profile=lambda r: np.ones_like(r))   # radial profile: ok
    with pytest.raises(ValueError):                                   # none given
        BorLayer(thickness_m=0.5e-6)
    with pytest.raises(ValueError):                                   # two given
        BorLayer(thickness_m=0.5e-6, eps=2.25 + 0j, rings=(3e-6, 0.5, 2.4 + 0j, 1.4 + 0j))


def test_bor_layer_thickness_and_rings_arity():
    from dynameta.optics.lumenairy_bridge import BorLayer
    with pytest.raises(ValueError):
        BorLayer(thickness_m=0.0, eps=2.25 + 0j)                      # nonpositive thickness
    with pytest.raises(ValueError):
        BorLayer(thickness_m=-1e-6, eps=2.25 + 0j)
    with pytest.raises(ValueError):
        BorLayer(thickness_m=0.5e-6, rings=(3e-6, 0.5, 2.4 + 0j))     # rings must be a 4-tuple


def test_bor_stack_spec_guards():
    from dynameta.optics.lumenairy_bridge import BorLayer, BorStackSpec
    good = BorLayer(thickness_m=0.5e-6, eps=2.25 + 0j)
    BorStackSpec(layers=[good], azimuthal_order_m=0, r_max_m=40e-6)   # m=0 allowed (axisymmetric mode)
    with pytest.raises(ValueError):
        BorStackSpec(layers=[], azimuthal_order_m=1, r_max_m=40e-6)   # empty stack
    with pytest.raises(ValueError):
        BorStackSpec(layers=[good], azimuthal_order_m=-1, r_max_m=40e-6)   # negative azimuthal order
    with pytest.raises(ValueError):
        BorStackSpec(layers=[good], azimuthal_order_m=1, r_max_m=0.0)      # nonpositive radius
    with pytest.raises(ValueError):
        BorStackSpec(layers=[good], azimuthal_order_m=1, r_max_m=40e-6, n_radial=8)   # too few points


def test_bor_backend_import_is_lumenairy_free():
    # constructing the BOR spec/layers must NOT import lumenairy (lazy-import contract): lumenairy is
    # only touched inside solve_bor via _require_bor
    code = ("import sys; "
            "from dynameta.optics.lumenairy_bridge import BorLayer, BorStackSpec; "
            "BorStackSpec(layers=[BorLayer(thickness_m=5e-7, eps=2.25+0j)], "
            "azimuthal_order_m=1, r_max_m=4e-5); "
            "assert 'lumenairy' not in sys.modules, 'lumenairy leaked from BOR spec construction'; "
            "print('ok')")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr


HAVE_JAX = importlib.util.find_spec("jax") is not None
needs_jax = pytest.mark.skipif(not (HAVE_LUM and HAVE_JAX),
                               reason="lumenairy or jax not installed")


@needs_jax
def test_berreman_jax_grad_matches_fd():
    # the JAX twin differentiates through a layer eps tensor: AD == central FD
    import jax
    import jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)
    from dynameta.optics.lumenairy_bridge import berreman_RT
    lam, d, n_o = 1.55e-6, 220e-9, 1.50

    def R_of(ne):
        eps = jnp.asarray([[ne ** 2, 0, 0], [0, n_o ** 2, 0], [0, 0, n_o ** 2]],
                          dtype=jnp.complex128)
        R, _T = berreman_RT([(eps, d)], 1.5 + 0j, 1.0 + 0j, lam, angle=0.0, row=0)
        return jnp.real(R)

    g = float(jax.grad(R_of)(jnp.asarray(1.74)))
    fd = (float(R_of(jnp.asarray(1.74 + 1e-6))) - float(R_of(jnp.asarray(1.74 - 1e-6)))) / 2e-6
    assert abs(g - fd) / (abs(fd) + 1e-12) < 1e-6


@needs_jax
def test_berreman_jax_forward_equals_numpy():
    # the differentiable path is the SAME physics as the concrete numpy forward
    import jax
    import jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)
    from dynameta.optics.lumenairy_bridge import berreman_RT
    lam, d, n_o, ne = 1.55e-6, 220e-9, 1.50, 1.74

    def eps(xp):
        return xp.asarray([[ne ** 2, 0, 0], [0, n_o ** 2, 0], [0, 0, n_o ** 2]],
                          dtype=xp.complex128)

    R_np, T_np = berreman_RT([(eps(np), d)], 1.5 + 0j, 1.0 + 0j, lam, angle=0.2, row=0)
    R_jx, T_jx = berreman_RT([(eps(jnp), jnp.asarray(d))], 1.5 + 0j, 1.0 + 0j,
                             jnp.asarray(lam), angle=jnp.asarray(0.2), row=0)
    assert abs(float(R_np) - float(R_jx)) < 1e-12
    assert abs(float(T_np) - float(T_jx)) < 1e-12


@needs_jax
def test_rcwa_design_twin_eager_grad_finite_nonzero():
    # audit 8.1-5: the RCWAStack design twin builds and one eager gradient (ridge eps through
    # a patterned cell's VALUES) is finite and nonzero; a jax wavelength must raise loudly
    # (the stack twin's static/traced split -- the rigorous gates live in
    # validation/lumenairy_rcwa_jax.py)
    import jax
    import jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)
    from dynameta.optics.lumenairy_bridge import rcwa_stack_RT
    sx = 4 * 3 + 1
    mask = jnp.asarray(np.arange(sx)[:, None] < sx // 2)

    def R_of(e):
        cell = jnp.where(mask, e + 0.0j, 1.0 + 0.0j)
        R, _T = rcwa_stack_RT([(cell, 180e-9)], 1.5 + 0j, 1.0 + 0j, 1.31e-6,
                              period_x=600e-9, n_orders=3, row=0)
        return jnp.real(R)

    g = float(jax.grad(R_of)(jnp.asarray(6.0)))
    assert np.isfinite(g) and g != 0.0
    with pytest.raises(TypeError, match="concrete"):
        rcwa_stack_RT([(jnp.asarray(6.0 + 0j), 180e-9)], 1.5 + 0j, 1.0 + 0j,
                      jnp.asarray(1.31e-6), period_x=600e-9, n_orders=3)


@needs_jax
def test_pmm_design_twin_eager_grad_finite_nonzero():
    # audit 8.1-5: the PMM design twin (1-D lamellar spectral element) builds and one eager
    # segment-eps gradient is finite and nonzero; segment WIDTHS are static and raise on jax
    import jax
    import jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)
    from dynameta.optics.lumenairy_bridge import pmm_stack_RT

    def R_of(e):
        segs = [(0.5, e + 0.0j), (0.5, 1.0 + 0j)]
        R, _T = pmm_stack_RT([(segs, 180e-9)], 1.5 + 0j, 1.0 + 0j, 1.31e-6,
                             period=600e-9, degree=6, n_orders=5, row=0)
        return jnp.real(R)

    g = float(jax.grad(R_of)(jnp.asarray(6.0)))
    assert np.isfinite(g) and g != 0.0
    with pytest.raises(TypeError, match="static"):
        pmm_stack_RT([([(jnp.asarray(0.5), 6.0 + 0j), (0.5, 1.0 + 0j)], 180e-9)],
                     1.5 + 0j, 1.0 + 0j, 1.31e-6, period=600e-9, degree=6, n_orders=5)


def test_drude_eps_jax_matches_material_model():
    # the carrier -> eps closure reproduces DrudeOptical.eps exactly at concrete inputs
    # (same formula, same dynameta.constants) and refuses untraceable callable parameters
    from dynameta.constants import M_E
    from dynameta.materials.optical_model import DrudeOptical
    from dynameta.optics.lumenairy_bridge import drude_eps_jax
    model = DrudeOptical(eps_inf=3.9, m_opt_kg=0.30 * M_E, gamma_rad_s=1.2e14)
    fn = drude_eps_jax(model)
    n, lam = 4.0e26, 1.31e-6
    assert fn(n, lam) == pytest.approx(complex(model.eps(lam, n_m3=n)), rel=1e-14)
    with pytest.raises(NotImplementedError):
        drude_eps_jax(DrudeOptical(eps_inf=3.9, m_opt_kg=lambda x: 0.30 * M_E,
                                   gamma_rad_s=1.2e14))


def _graded_design_and_eps():
    """Asymmetric LOSSY graded layer fixture (audit C5-1 regression).

    slice_eps_field returns ascending-z (substrate-first) slabs; every
    superstrate-first consumer must reverse them. A symmetric or lossless profile
    is blind to the flip (R/T reversal-invariant), so the profile here is both
    asymmetric and lossy -- and the test asserts that discriminating power below.
    """
    from dynameta.core.eps_field import EpsField
    d = _uniform_design()                      # air | 'a' 120nm | glass, normal incidence
    z_nm = np.linspace(0.0, 120.0, 25)         # nm solver units, ascending = substrate-first
    u = z_nm / 120.0
    eps_z = 2.0 + 6.7 * u ** 2 + 1.0j * u ** 3     # eps(top) >> eps(bottom), lossy toward top
    ef = EpsField(z_axis_u=z_nm, y_axis_u=np.zeros(1), x_axis_u=np.zeros(1),
                  values_zyx=eps_z.reshape(-1, 1, 1).astype(complex))
    return d, {"a": ef}


@needs_lum
def test_graded_slab_order_matches_tmm_asymmetric_lossy():
    # audit C5-1: the three bridges inserted slice_eps_field's ascending (substrate-first)
    # slabs UNREVERSED into superstrate-first stacks, vertically flipping every graded
    # profile. TMM (tmm_reference.py, reversed(...)) is the proven-correct side of the
    # seam; all three bridges must agree with it on an asymmetric lossy profile.
    from dynameta.optics.lumenairy_bridge import (make_lumenairy_berreman_solver,
                                                  make_lumenairy_pmm_solver,
                                                  make_lumenairy_rcwa_solver)
    from dynameta.optics.tmm_reference import make_layered_tmm_solver
    d, eps_by_region = _graded_design_and_eps()
    lam = 1.31e-6
    r_t = make_layered_tmm_solver()(d, None, eps_by_region, lam, 1.0 + 0j, 1.5 + 0j)

    # the fixture must actually discriminate the flip: TMM on the reversed profile
    # differs materially (otherwise a future 'simplification' could blind this test)
    d2, ebr2 = _graded_design_and_eps()
    ef = ebr2["a"]
    ebr_flipped = {"a": type(ef)(z_axis_u=ef.z_axis_u,
                                 y_axis_u=ef.y_axis_u, x_axis_u=ef.x_axis_u,
                                 values_zyx=ef.values_zyx[::-1].copy())}
    r_flip = make_layered_tmm_solver()(d2, None, ebr_flipped, lam, 1.0 + 0j, 1.5 + 0j)
    assert abs(r_flip.R - r_t.R) > 1e-3, "fixture lost its asymmetry discrimination"

    for make in (make_lumenairy_rcwa_solver, make_lumenairy_pmm_solver,
                  make_lumenairy_berreman_solver):
        r_b = make()(d, None, eps_by_region, lam, 1.0 + 0j, 1.5 + 0j)
        assert r_b.R == pytest.approx(r_t.R, abs=1e-9), make.__name__
        assert r_b.T == pytest.approx(r_t.T, abs=1e-9), make.__name__
        assert r_b.phase_deg == pytest.approx(r_t.phase_deg, abs=1e-6), make.__name__


@needs_lum
def test_structured_stack_without_period_raises():
    # audit C5-8: a STRUCTURED LayeredStack with defaulted period 0.0 used to solve at a
    # fabricated wavelength-sized period (wrong diffraction geometry, silently)
    import numpy as np
    from dynameta.core.layered import LayeredSlab, LayeredStack
    from dynameta.geometry.specs import OpticalSpec
    from dynameta.optics.lumenairy_bridge import LumenairyStackSolver
    cell = np.where(np.linspace(0, 1, 33)[:, None] < 0.3, 4.0 + 0j, 1.0 + 0j)
    stk = LayeredStack(1.0 + 0j, 1.5 + 0j, [LayeredSlab(180e-9, eps_cell=cell)])
    opt = OpticalSpec(polarization="y", incidence_angle_deg=0.0)
    with pytest.raises(ValueError, match="period"):
        LumenairyStackSolver(n_orders=3).solve(stk, 1.31e-6, opt)


def test_collapse_unclaimed_keys_raise():
    # audit C5-11: a drifted/typo'd region key used to be silently dropped -- the layer
    # silently reverted to nominal material eps in every layered backend
    from dynameta.core.eps_field import EpsField
    from dynameta.core.layered import collapse_regions_to_layers
    d = _uniform_design()
    good = {"a": EpsField(scalar=4.1 + 0j)}
    assert "a" in collapse_regions_to_layers(d, good)
    with pytest.raises(ValueError, match="match no design layer"):
        collapse_regions_to_layers(d, {"ito_top": EpsField(scalar=4.1 + 0j)})
    # superstrate/substrate/pml_* keys remain legitimately ignorable
    out = collapse_regions_to_layers(d, {"a": EpsField(scalar=4.1 + 0j),
                                         "superstrate": EpsField(scalar=1.0 + 0j),
                                         "pml_top": EpsField(scalar=1.0 + 0j)})
    assert set(out) == {"a"}


def test_common_version_floor_and_parse():
    # audit 6.3 / section 8.2-2: ONE bridge floor in _common (bor_backend had a copy-pasted
    # gate at a DIFFERENT floor), and a parse that survives pre/post-release suffixes (the
    # old tuple(int(p) ...) x3 crashed on '5.21.0rc1')
    from dynameta.optics.lumenairy_bridge._common import (VERSION_FLOOR, parse_version,
                                                          require_lumenairy)
    assert parse_version("5.21.0rc1") == (5, 21, 0)
    assert parse_version("5.22.dev0") == (5, 22, 0)
    assert parse_version("5.21") == (5, 21, 0)
    assert parse_version("6.0.0.post1") == (6, 0, 0)
    assert VERSION_FLOOR >= (5, 21, 0)
    lum = pytest.importorskip("lumenairy")
    if parse_version(lum.__version__) >= VERSION_FLOOR:
        assert require_lumenairy() is lum


@needs_lum
def test_translate_reads_layers_via_common():
    # the ONE version-ceilinged private-surface read (stack._layers has no public accessor)
    from dynameta.optics.lumenairy_bridge._common import stack_layer_records
    import lumenairy
    stk = lumenairy.RCWAStack(300e-9, n_superstrate=1.0, n_substrate=1.5, n_orders=3)
    stk.add_layer(100e-9, eps=4.0 + 0j)
    recs = stack_layer_records(stk)
    assert len(recs) == 1 and recs[0].kind == "uniform"


@needs_lum
def test_threaded_sweep_matches_serial():
    # audit 8.1-2: n_workers threads solve_sweep; results stored by index and each
    # wavelength is an independent LAPACK problem, so the output must be byte-identical
    from dynameta.optics.lumenairy_bridge import make_lumenairy_rcwa_solver
    d = _grating_design(600e-9, pol="x")
    lams = [1.2e-6, 1.3e-6, 1.4e-6, 1.5e-6]
    ser = make_lumenairy_rcwa_solver(n_orders=5).solve_sweep(
        d, None, lambda lam: {}, lams, 1.0 + 0j, 1.5 + 0j)
    par = make_lumenairy_rcwa_solver(n_orders=5, n_workers=4).solve_sweep(
        d, None, lambda lam: {}, lams, 1.0 + 0j, 1.5 + 0j)
    for a, b in zip(ser, par):
        assert a.R == b.R and a.T == b.T and a.r == b.r and a.phase_deg == b.phase_deg


@needs_lum
def test_pmm_absorption_parity_surface():
    # audit 8.1-3: absorption=True fills per_region_absorption + A_independent (keyed by
    # design layer) and the budget closes; default stays None (byte-identical off-switch).
    # The cross-engine physics oracle is validation/lumenairy_pmm_bridge GATE E.
    from dynameta.geometry import Layer
    from dynameta.optics.lumenairy_bridge import make_lumenairy_pmm_solver
    d = _uniform_design()          # single lossy layer 'a' (hi = 4.0 + 0.3j)
    lam = 1.31e-6
    r0 = make_lumenairy_pmm_solver()(d, None, {}, lam, 1.0 + 0j, 1.5 + 0j)
    assert r0.per_region_absorption is None and r0.A_independent is None
    r1 = make_lumenairy_pmm_solver(absorption=True)(d, None, {}, lam, 1.0 + 0j, 1.5 + 0j)
    assert set(r1.per_region_absorption) == {"a"}
    assert r1.A_independent == pytest.approx(r1.A, abs=1e-9)
    assert r1.per_region_absorption["a"] == pytest.approx(r1.A_independent, abs=1e-12)
    assert r1.R == r0.R and r1.T == r0.T


@needs_lum
def test_berreman_oop_oblique_first_class():
    # audit 8.1-6: OOP-coupled tensors (exz != 0, tilted-LC director) at OBLIQUE and
    # CONICAL incidence solve through the bridge (lumenairy 5.21 fixed the far field);
    # absorption=True degrades GRACEFULLY there (warn + A_independent None -- the
    # internal-field reconstruction is a documented upstream limitation), never crashes.
    import warnings
    import numpy as np
    from dynameta.core.eps_field import EpsField
    from dynameta.geometry import Design, Layer, Stack, UnitCell
    from dynameta.geometry.specs import OpticalSpec
    from dynameta.materials import ConstantOptical, Material, MaterialRegistry
    from dynameta.optics.lumenairy_bridge import make_lumenairy_berreman_solver
    th_d = np.radians(40.0)
    n_o2, n_e2 = complex(2.30, 0.06), complex(2.89, 0.09)
    d_hat = np.array([np.sin(th_d), 0.0, np.cos(th_d)])
    eps = n_o2 * np.eye(3) + (n_e2 - n_o2) * np.outer(d_hat, d_hat)
    reg = MaterialRegistry()
    reg.add(Material("air", ConstantOptical(1.0 + 0j)))
    reg.add(Material("glass", ConstantOptical(complex(1.5 ** 2))))
    reg.add(Material("lo", ConstantOptical(complex(2.1))))
    d = Design(name="lc", unit_cell=UnitCell.square(300e-9),
               stack=Stack(layers=[Layer("lc", 800e-9, "lo")], superstrate_material="air",
                           substrate_material="glass"),
               electrodes=[], materials=reg,
               optical=OpticalSpec(polarization="y", incidence_angle_deg=30.0,
                                   azimuth_deg=25.0))
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        r = make_lumenairy_berreman_solver(absorption=True)(
            d, None, {"lc": EpsField(tensor=eps)}, 1.55e-6, 1.0 + 0j, 1.5 + 0j)
    assert 0.0 <= r.A <= 1.0 and 0.0 <= r.R <= 1.0 and 0.0 <= r.T <= 1.0
    assert r.A_independent is None
    assert any("absorption unavailable" in str(x.message) for x in w)
