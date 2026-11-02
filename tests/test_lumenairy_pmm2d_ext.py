"""PMM2D bridge extensions (2026-11-02, lumenairy 5.50): tensor eps, curved shapes,
non-commensurate walls (shapes / per-layer routes) and the complex transmission t.

Fast gates only -- every test here is a few seconds on a 4-vCPU leg; the full
convergence ladders (RCWA approach direction, two-sided FFF / Laurent brackets, the
patterned-anisotropic RCWA oracle) live in validation/lumenairy_pmm2d_bridge.py GATES
F-J.  Each physics test carries an INDEPENDENT oracle (Berreman 4x4, TMM, RCWA) AND a
discrimination leg proving it would catch the obvious mapping slip.  The first test is
lumenairy-free (pure-Python translation helpers)."""
import importlib.util

import numpy as np
import pytest

HAVE_LUM = importlib.util.find_spec("lumenairy") is not None
needs_lum = pytest.mark.skipif(not HAVE_LUM, reason="lumenairy not installed")

LAM = 1.31e-6
PER = 600e-9


def _design(layers, *, pol="x", theta=0.0):
    from dynameta.geometry import Design, Stack, UnitCell
    from dynameta.geometry.specs import OpticalSpec
    from dynameta.materials import ConstantOptical, Material, MaterialRegistry
    reg = MaterialRegistry()
    for name, eps in (("air", 1.0), ("glass", 2.25), ("lo", 2.1), ("pil", 4.0),
                      ("lcm", 2.4), ("hi", 4.0 + 0.3j)):
        reg.add(Material(name, ConstantOptical(complex(eps))))
    return Design(name="ext", unit_cell=UnitCell.square(PER),
                  stack=Stack(layers=layers, superstrate_material="air",
                              substrate_material="glass"),
                  electrodes=[], materials=reg,
                  optical=OpticalSpec(polarization=pol, incidence_angle_deg=theta))


def _pillar(shape, *, bg="air", thk=200e-9, pol="x"):
    from dynameta.geometry import Inclusion, Layer
    return _design([Layer("p", thk, bg, inclusions=[Inclusion(shape=shape,
                                                              material="pil")])],
                   pol=pol)


def _uniaxial(no, ne, az_deg, tilt_deg):
    a, t = np.radians(az_deg), np.radians(tilt_deg)
    d = np.array([np.cos(t) * np.cos(a), np.cos(t) * np.sin(a), np.sin(t)])
    return (no ** 2 * np.eye(3) + (ne ** 2 - no ** 2) * np.outer(d, d)).astype(complex)


def _swap_xy(eps):
    p = [1, 0, 2]
    return np.asarray(eps)[p][:, p].copy()


def _solve(sol, d, ebr=None):
    return sol(d, None, ebr or {}, LAM, 1.0 + 0j, 1.5 + 0j)


def test_pmm2d_ext_translation_helpers():
    # lumenairy-FREE: the route choice (resolve_pure_grid), the per-layer strip tile
    # with segment-count equalization, and the TENSOR union cell.
    from dynameta.core.eps_field import EpsField
    from dynameta.geometry import Layer
    from dynameta.geometry.cross_section import Circle, Rectangle
    from dynameta.optics.lumenairy_bridge.pmm2d_backend import (_layer_background,
                                                                _rect_strip_tile,
                                                                layer_to_pure_cell,
                                                                resolve_pure_grid)
    patch = Rectangle(PER / 2.0, PER / 2.0, PER / 2.0, PER / 2.0)     # N = 4, S = 3
    assert resolve_pure_grid(_pillar(patch)) == "union"                 # v1 path kept
    snap8 = Rectangle(0.5625 * PER, 0.4375 * PER, 0.5 * PER, 0.375 * PER)
    assert resolve_pure_grid(_pillar(snap8)) == "shapes"                # N = 16 >> S = 3
    odd = Rectangle(0.53 * PER, 0.48 * PER, 0.46 * PER, 0.37 * PER)
    assert resolve_pure_grid(_pillar(odd)) == "shapes"                  # no N fits
    assert resolve_pure_grid(_pillar(Circle(PER / 2, PER / 2, 0.3 * PER))) == "shapes"
    assert resolve_pure_grid(_pillar(odd), "per-layer") == "per-layer"
    assert resolve_pure_grid(_design([Layer("a", 1e-7, "lo")])) == "union"
    with pytest.raises(ValueError):
        resolve_pure_grid(_pillar(odd), "mortar")
    # strip tile: an x-stripe (walls 0.2 p / 0.55 p, none on y) -> the y axis is
    # BISECTED up to the same segment count (0.5 p, then 0.25 p); stripe in column 1
    stripe = Rectangle(0.375 * PER, PER / 2.0, 0.35 * PER, PER)
    d = _pillar(stripe)
    xw, yw, tile = _rect_strip_tile(d.stack.layers[0], d, LAM, PER, PER, 1.0 + 0j)
    assert np.allclose(np.array(xw) / PER, [0.2, 0.55])
    assert np.allclose(np.array(yw) / PER, [0.25, 0.5])
    assert tile.shape == (3, 3)
    assert np.allclose(tile[1, :], 4.0) and np.allclose(tile[[0, 2], :], 1.0)
    # TENSOR background on the union grid: (4, 4, 3, 3), pillar = 4 I, rest = tensor
    eps_lc = _uniaxial(1.5, 1.7, 30.0, 40.0)
    d = _pillar(patch, bg="lcm")
    bg = _layer_background(d.stack.layers[0], d, LAM, EpsField(tensor=eps_lc))
    cell = layer_to_pure_cell(d.stack.layers[0], d, LAM, PER, PER, 4, bg_eps=bg)
    assert cell.shape == (4, 4, 3, 3)
    assert np.allclose(cell[1:3, 1:3], 4.0 * np.eye(3))
    assert np.allclose(cell[0, 0], eps_lc) and np.allclose(cell[3, 2], eps_lc)
    # a scalar background stays the v1 (N, N) cell, bit for bit
    v1 = layer_to_pure_cell(d.stack.layers[0], d, LAM, PER, PER, 4)
    assert v1.shape == (4, 4) and np.allclose(v1[0, 0], 2.4)


@needs_lum
def test_pmm2d_uniform_tensor_matches_berreman_full_jones():
    # GATE F (fast copy): a uniform asymmetric lossy in-plane tensor and a tilted-
    # director OOP tensor through BOTH engines vs the Berreman 4x4 bridge -- R/T/r/t
    # and the FULL 2x2 reflection + transmission Jones (co-pol is transpose-blind by
    # reciprocity; the TRANSPOSED tensor must move the off-diagonal).  Measured
    # 2026-11-02: pure 1.7e-10 worst (M = 5, oblique), hybrid 7e-15.
    import lumenairy as lum
    from dynameta.core.eps_field import EpsField
    from dynameta.geometry import Layer
    from dynameta.optics.lumenairy_bridge import (design_to_berreman_layers,
                                                  design_to_pmm2d_stack,
                                                  make_lumenairy_berreman_solver,
                                                  make_lumenairy_pmm2d_solver)
    ip = np.array([[3.0 + 0.1j, 0.4 + 0.2j, 0.0], [-0.1 + 0.3j, 2.5 + 0.05j, 0.0],
                   [0.0, 0.0, 2.8 + 0.1j]])
    oop = _uniaxial(1.5, 1.75, 30.0, 40.0) + 0.05j * np.eye(3)
    oop[0, 2] += 0.2
    oop[2, 1] -= 0.15j
    ber = make_lumenairy_berreman_solver()
    kw = {"pure": dict(n_modes=5, n_orders=3), "hybrid": dict(degree=7, n_orders=3)}

    def full(d, ebr, eng):
        st, _ = design_to_pmm2d_stack(d, LAM, engine=eng, eps_by_region=ebr, **kw[eng])
        st.set_source(LAM, theta=np.radians(d.optical.incidence_angle_deg), phi=0.0)
        return np.asarray(st.solve()[3]), np.asarray(st.jones_transmission())

    for eps in (ip, oop):
        ebr = {"a": EpsField(tensor=eps)}
        for pol, th in (("x", 0.0), ("p", 30.0)):
            d = _design([Layer("a", 200e-9, "lo")], pol=pol, theta=th)
            rb = _solve(ber, d, ebr)
            for eng in ("pure", "hybrid"):
                rp = _solve(make_lumenairy_pmm2d_solver(engine=eng, **kw[eng]), d, ebr)
                for a, b in ((rp.R, rb.R), (rp.T, rb.T), (rp.r, rb.r), (rp.t, rb.t)):
                    assert abs(a - b) < 1e-8
            layers, n_sup, n_sb, _ = design_to_berreman_layers(d, LAM, eps_by_region=ebr)
            st = lum.BerremanStack(n_substrate=complex(n_sb), n_superstrate=complex(n_sup))
            for e, t in layers:
                st.add_layer(float(t), eps=e)
            st.set_source(LAM, theta=np.radians(th), phi=0.0)
            jb = (np.asarray(st.solve()[2]), np.asarray(st.jones_transmission()))
            jp = full(d, ebr, "pure")
            assert np.abs(jp[0] - jb[0]).max() < 1e-8
            assert np.abs(jp[1] - jb[1]).max() < 1e-8
            jt = full(d, {"a": EpsField(tensor=eps.T.copy())}, "pure")
            assert max(np.abs(jt[0] - jb[0]).max(), np.abs(jt[1] - jb[1]).max()) > 5e-3
            rs = _solve(make_lumenairy_pmm2d_solver(**kw["pure"]), d,
                        {"a": EpsField(tensor=_swap_xy(eps))})
            assert abs(rs.r - rb.r) > 1e-3                  # x <-> y slip is visible


@needs_lum
def test_pmm2d_t_power_closure_and_per_order_route():
    # GATE H (fast copy): the bridge t is the co-pol zeroth-order transmission in the
    # Byrnes basis -- |t|^2 n_sub cos_t / cos_i == T on a lossless uniform stack at
    # 30 deg (s and p), it matches TMM, and stack.per_order_amplitudes('transmission')
    # at order (0, 0) IS jones_transmission() (the documented per-order route).
    from dynameta.geometry import Layer
    from dynameta.optics.lumenairy_bridge import (design_to_pmm2d_stack,
                                                  make_lumenairy_pmm2d_solver)
    from dynameta.optics.tmm_reference import make_layered_tmm_solver
    th = np.radians(30.0)
    pfac = 1.5 * np.sqrt(1.0 - (np.sin(th) / 1.5) ** 2) / np.cos(th)
    lays = [Layer("a", 120e-9, "lo"), Layer("b", 200e-9, "glass")]
    sol = make_lumenairy_pmm2d_solver(n_modes=5, n_orders=3)
    for pol in ("y", "p"):
        d = _design(lays, pol=pol, theta=30.0)
        r = _solve(sol, d)
        assert abs(abs(r.t) ** 2 * pfac - r.T) < 1e-9
        assert abs(r.t - _solve(make_layered_tmm_solver(), d).t) < 1e-9
    st, _ = design_to_pmm2d_stack(d, LAM, n_modes=5, n_orders=3)
    st.set_source(LAM, theta=th, phi=0.0)
    st.solve()
    po = st.per_order_amplitudes("transmission")
    k0 = int(np.flatnonzero((po["orders"][:, 0] == 0) & (po["orders"][:, 1] == 0))[0])
    jt = np.asarray(st.jones_transmission())
    assert np.abs(po["Ex"][:, k0] - jt[0]).max() == 0.0
    assert np.abs(po["Ey"][:, k0] - jt[1]).max() == 0.0


@needs_lum
def test_pmm2d_circle_pillar_vs_rcwa_fff():
    # GATE I (fast copy): an eps-4 circle (r = 0.3 p) on the curved shapes= map vs the
    # RCWA bridge's FFF formulation.  Measured 2026-11-02: pure M = 5 6.0976e-3 vs
    # FFF n = 12 6.0805e-3 (1.7e-5; the full ladder brackets pure from both sides,
    # validation GATE I); a 2 % larger radius moves R by 1.6e-4.
    from dynameta.geometry.cross_section import Circle
    from dynameta.optics.lumenairy_bridge import (make_lumenairy_pmm2d_solver,
                                                  make_lumenairy_rcwa_solver)
    sol = make_lumenairy_pmm2d_solver(n_modes=5, n_orders=3)
    r_p = _solve(sol, _pillar(Circle(PER / 2, PER / 2, 0.3 * PER)))
    r_f = _solve(make_lumenairy_rcwa_solver(n_orders=12, cell_samples=600,
                                            formulation="fff"),
                 _pillar(Circle(PER / 2, PER / 2, 0.3 * PER)))
    assert abs(r_p.R - r_f.R) < 3e-5
    assert abs(r_p.T - r_f.T) < 3e-5
    r_w = _solve(sol, _pillar(Circle(PER / 2, PER / 2, 0.306 * PER)))
    assert abs(r_w.R - r_p.R) > 1e-4                      # radius mapping is pinned


@needs_lum
def test_pmm2d_noncommensurate_rect_vs_rcwa_fff_and_per_layer():
    # GATE J (fast copy): walls 0.30 / 0.76 p x 0.295 / 0.665 p fit no uniform N <= 32
    # (v1 raised); 'auto' -> 'shapes' places them exactly.  Measured 2026-11-02: pure
    # M = 5 1.04470e-2 vs FFF n = 12 1.05005e-2 (5.3e-5); 6 nm wider moves R 4.9e-4;
    # the per-layer route on the same single layer has the SAME walls -> same answer.
    from dynameta.geometry.cross_section import Rectangle
    from dynameta.optics.lumenairy_bridge import (make_lumenairy_pmm2d_solver,
                                                  make_lumenairy_rcwa_solver)

    def odd(w=0.46 * PER):
        return _pillar(Rectangle(0.53 * PER, 0.48 * PER, w, 0.37 * PER))

    sol = make_lumenairy_pmm2d_solver(n_modes=5, n_orders=3)
    r_p = _solve(sol, odd())
    r_f = _solve(make_lumenairy_rcwa_solver(n_orders=12, cell_samples=600,
                                            formulation="fff"), odd())
    assert abs(r_p.R - r_f.R) < 1e-4
    assert abs(_solve(sol, odd(0.47 * PER)).R - r_p.R) > 2e-4
    r_pl = _solve(make_lumenairy_pmm2d_solver(n_modes=5, n_orders=3,
                                              pure_grid="per-layer"), odd())
    assert abs(r_pl.R - r_p.R) < 1e-10 and abs(r_pl.t - r_p.t) < 1e-10


@needs_lum
def test_pmm2d_patterned_tensor_pure_vs_hybrid():
    # GATE G (fast copy): an in-plane LC background (azimuth 30 deg) around a centred
    # eps-4 square pillar -- the (4, 4, 3, 3) union tensor cell on the pure engine vs
    # the hybrid's (128, 128, 3, 3) eps_tensor_cell raster (Fourier-projected modal --
    # a different basis; the RCWA tensor oracle is the validation-tier gate).
    # Measured 2026-11-02: |dR| 5.1e-5 (x) / 1.1e-5 (y); the x <-> y permuted tensor
    # moves R(x) by 1.6e-2 (it IS the y answer, by the square's symmetry).
    from dynameta.core.eps_field import EpsField
    from dynameta.geometry.cross_section import Rectangle
    from dynameta.optics.lumenairy_bridge import make_lumenairy_pmm2d_solver
    sq = Rectangle(PER / 2.0, PER / 2.0, PER / 2.0, PER / 2.0)
    eps_lc = _uniaxial(1.5, 1.7, 30.0, 0.0)
    ebr = {"p": EpsField(tensor=eps_lc)}
    pure = make_lumenairy_pmm2d_solver(n_modes=5, n_orders=3)
    hyb = make_lumenairy_pmm2d_solver(engine="hybrid", degree=9, n_orders=9)
    for pol in ("x", "y"):
        d = _pillar(sq, bg="lcm", thk=250e-9, pol=pol)
        r_p, r_h = _solve(pure, d, ebr), _solve(hyb, d, ebr)
        assert abs(r_p.R - r_h.R) < 1.5e-4
        assert abs(r_p.T - r_h.T) < 1.5e-4
    d = _pillar(sq, bg="lcm", thk=250e-9, pol="x")
    r_s = _solve(pure, d, {"p": EpsField(tensor=_swap_xy(eps_lc))})
    assert abs(r_s.R - _solve(pure, d, ebr).R) > 5e-3


@needs_lum
def test_pmm2d_ext_guards():
    # loud refusals for what has no exact pure-engine image (or an explicit v1 route
    # asked to do what it cannot): Polygon, a structured TENSOR raster, a Circle under
    # pure_grid='union', a bogus pure_grid.
    from dynameta.core.eps_field import EpsField
    from dynameta.geometry import Layer
    from dynameta.geometry.cross_section import Circle, Polygon
    from dynameta.optics.lumenairy_bridge import make_lumenairy_pmm2d_solver
    pure = make_lumenairy_pmm2d_solver(n_modes=5, n_orders=3)
    tri = Polygon([(100e-9, 100e-9), (500e-9, 120e-9), (300e-9, 480e-9)])
    with pytest.raises(ValueError, match="no EXACT image"):
        _solve(pure, _pillar(tri))
    tv = (np.eye(3) * (2.0 + 0.1 * np.arange(8.0))[:, None, None]).astype(complex)
    tgrid = EpsField(values_zyx=np.broadcast_to(tv, (3, 1, 8, 3, 3)).copy(),
                     z_axis_u=np.array([0.0, 50.0, 100.0]), x_axis_u=np.arange(8.0),
                     y_axis_u=np.array([0.0]))
    with pytest.raises(ValueError, match="TENSOR"):
        _solve(pure, _design([Layer("a", 100e-9, "lo")]), {"a": tgrid})
    with pytest.raises(ValueError):
        _solve(make_lumenairy_pmm2d_solver(n_modes=5, n_orders=3, pure_grid="union"),
               _pillar(Circle(PER / 2, PER / 2, 0.3 * PER)))
    with pytest.raises(ValueError):
        make_lumenairy_pmm2d_solver(pure_grid="mortar")
