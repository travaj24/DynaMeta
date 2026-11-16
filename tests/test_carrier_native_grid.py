"""Gates for the native carrier z grid (v0.16): LayeredDevsimBuilder.solve(grid_n_z=None),
core.resample.resample_tensor_grid, EpsField.voxel_bounds_u on a non-uniform axis, and run_pipeline's
carrier_solve_kwargs / automatic native grid for a layered optical solver.

WHY. solve() returned grid_fields on a uniform grid_n_x x 32 resample of the DEVSIM nodes, and the bridge
(core.bridge.assemble_eps) builds every EpsField from grid_fields; run_pipeline called carrier.solve(bp) with
no grid argument. On the generic gated degenerate film of tests/_gated_film_fixture.py (6 nm, fine 189-node mesh,
+5 V) the 32-point grid (0.194 nm spacing) puts the excess sheet density +7.4e-3 off the raw-node trapezoid,
and it is the layer the optics is most sensitive to (Re eps crosses zero inside the ~1 nm accumulation
layer). The layered optical solvers slice a non-uniform z axis, so the native mesh lines carry the carrier
solution with no resampling error.

FAILS ON v0.15.1: grid_n_z=None is a TypeError in resample_to_grid (int(None)); resample_tensor_grid,
carrier_solve_kwargs and the voxel-axis check do not exist; run_pipeline never passes a grid argument
(the spy sees {}). The 32-point sheet-density error is re-measured below as a consistency check of the
reproduction (MEASURED: +7.40e-3).
"""
import inspect
import warnings

import numpy as np
import pytest

import _gated_film_fixture as G

from dynameta.core.eps_field import AXIS_UNIFORM_RTOL, EpsField
from dynameta.core.numerics import trapz
from dynameta.core.resample import NotATensorGridError, resample_tensor_grid, resample_to_grid
from dynameta.warning_types import AdvisoryWarning


# ---------------------------------------------------------------------------------------------------
# pure: resample_tensor_grid
# ---------------------------------------------------------------------------------------------------

def _tensor_nodes(rng):
    xs = np.array([0.0, 1.0e-9, 4.0e-9, 9.0e-9, 20.0e-9, 40.0e-9])             # non-uniform x
    ys = np.cumsum(np.r_[0.0, np.geomspace(1.25e-11, 5e-10, 20)])              # graded y
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    x, y = X.ravel(), Y.ravel()
    perm = rng.permutation(x.size)                                             # DEVSIM order is arbitrary
    return xs, ys, x[perm], y[perm]


def test_resample_tensor_grid_native_y_exact_x_linear():
    """Oracle: f(x, y) = 3 + 2e9 x + exp(-y / 0.3 nm) is LINEAR in x, so the separable x interpolation on
    each native y line must reproduce f at every (grid x, node y) to round-off, and the two end columns
    must carry the node values bit for bit. The y axis must be the node y lines themselves."""
    rng = np.random.default_rng(7)
    xs, ys, x, y = _tensor_nodes(rng)

    def f(xx, yy):
        return 3.0 + 2e9 * xx + np.exp(-yy / 3e-10)

    vals = f(x, y)
    axes, out = resample_tensor_grid(x, y, {"n": vals, "neg": -vals}, 17)
    assert np.array_equal(axes["y"], ys)
    assert np.array_equal(axes["x"], np.linspace(0.0, 40e-9, 17))
    g = out["n"]
    assert g.shape == (17, ys.size)
    ref = f(axes["x"][:, None], ys[None, :])
    assert np.max(np.abs(g / ref - 1.0)) < 1e-14
    assert np.array_equal(out["neg"], -g)
    # end columns: node values exactly (no interpolation weight off the node)
    for col, xv in ((0, 0.0), (-1, 40e-9)):
        k = x == xv
        o = np.argsort(y[k])
        assert np.array_equal(g[col], vals[k][o])


def test_resample_tensor_grid_refuses_non_tensor_node_set():
    rng = np.random.default_rng(3)
    _, _, x, y = _tensor_nodes(rng)
    with pytest.raises(NotATensorGridError, match="not a tensor grid"):
        resample_tensor_grid(x[1:], y[1:], {"n": np.ones(x.size - 1)}, 8)        # one node missing
    xd, yd = np.r_[x, x[0]], np.r_[y, y[0]]                                    # one node repeated
    with pytest.raises(ValueError):
        resample_tensor_grid(xd, yd, {"n": np.ones(xd.size)}, 8)
    assert issubclass(NotATensorGridError, ValueError)


def test_resample_tensor_grid_agrees_with_resample_to_grid_on_uniform_lines():
    """Consistency check (not an oracle): on a tensor grid with UNIFORM y lines and n_z = Ny the two
    resamplers place the same points; LinearND interpolation along a mesh line is the same linear
    interpolation, so they agree to round-off."""
    xs = np.linspace(0.0, 30e-9, 7)
    ys = np.linspace(0.0, 5e-9, 11)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    v = np.sin(X.ravel() * 1e8) + Y.ravel() * 1e9
    axes, out = resample_tensor_grid(X.ravel(), Y.ravel(), {"v": v}, 13)
    old = resample_to_grid(np.column_stack([X.ravel(), Y.ravel()]), {"v": v}, (13, 11))
    assert np.allclose(axes["y"], old["axis_1"], rtol=0, atol=1e-24)
    assert np.allclose(out["v"], old["v"], rtol=0, atol=1e-13)


# ---------------------------------------------------------------------------------------------------
# pure: EpsField.voxel_bounds_u
# ---------------------------------------------------------------------------------------------------

def _field(z):
    x = np.linspace(0.0, 400.0, 9)
    y = np.array([0.0, 400.0])
    return EpsField(x_axis_u=x, y_axis_u=y, z_axis_u=np.asarray(z, float),
                    values_zyx=np.ones((len(z), 2, 9), complex))


def test_voxel_bounds_refuses_non_uniform_axis():
    """A VoxelCoefficient gets only the two corners: a graded native z axis would be stretched evenly over
    the layer, silently. On v0.15.1 this returned the corners (no error)."""
    z = np.r_[0.0, np.cumsum(np.geomspace(0.0125, 0.5, 12))]
    with pytest.raises(ValueError, match="NON-UNIFORM"):
        _field(z).voxel_bounds_u()
    with pytest.raises(ValueError, match="grid_n_z"):
        _field(z).voxel_bounds_u()


def test_voxel_bounds_uniform_axes_unchanged():
    z = np.linspace(110.0, 115.0, 32)
    s, e = _field(z).voxel_bounds_u()
    assert s == (0.0, 0.0, 110.0) and e == (400.0, 400.0, 115.0)
    # an affine remap of a linspace (what core.bridge does) stays far inside the tolerance
    zr = 1e9 * (2.3e-8 + (np.linspace(0.0, 5e-9, 257) - 0.0) * (7e-9 / 5e-9))
    assert _field(zr).voxel_bounds_u()[1][2] == pytest.approx(zr[-1], rel=0, abs=0)
    d = np.diff(zr)
    assert (d.max() - d.min()) / d.mean() < 1e-3 * AXIS_UNIFORM_RTOL
    # one- and two-point axes are uniform by definition
    assert _field([0.0, 5.0]).voxel_bounds_u()[1][2] == 5.0
    with pytest.raises(ValueError, match="uniform EpsField"):
        EpsField(scalar=2.0 + 0j).voxel_bounds_u()


# ---------------------------------------------------------------------------------------------------
# run_pipeline: which kwargs reach carrier.solve
# ---------------------------------------------------------------------------------------------------

class _Stop(Exception):
    pass


def _sweep():
    from dynameta.sweep import BiasPoint, Sweep
    return Sweep(bias_points=[BiasPoint({"top": 1.0, "bot": 0.0}, "b")], wavelengths_nm=[1200.0])


def _layered_solver(design, geo, eps_by_region, lambda_m, n_super, n_sub):   # pragma: no cover
    raise AssertionError("not reached")


def _layered_solver_native(design, geo, eps_by_region, lambda_m, n_super, n_sub):   # pragma: no cover
    raise AssertionError("not reached")


_layered_solver_native.native_carrier_grid = True


def test_run_pipeline_carrier_solve_kwargs_selection():
    """Spy on carrier.solve. A LayeredDevsimBuilder feeding a solver with native_carrier_grid = True gets
    grid_n_z=None; the FEM default (optical_solver=None), a solver without the attribute, and a
    bring-your-own carrier get the v0.15.1 call solve(bp); carrier_solve_kwargs always wins."""
    pytest.importorskip("devsim")
    from dynameta.carriers.devsim_layered import LayeredDevsimBuilder
    from dynameta.pipeline import run_pipeline

    class SpyLayered(LayeredDevsimBuilder):
        def __init__(self):                                   # no planning, no DEVSIM
            self.calls = []

        def solve(self, bias, **kw):
            self.calls.append(kw)
            raise _Stop

    class SpyBYO:
        def __init__(self):
            self.calls = []

        def solve(self, bias, **kw):
            self.calls.append(kw)
            raise _Stop

    design = G.film_design()
    cases = [
        (SpyLayered(), _layered_solver_native, None, {"grid_n_z": None}),
        (SpyLayered(), None, None, {}),                               # FEM default
        (SpyLayered(), _layered_solver, None, {}),                    # no attribute
        (SpyBYO(), _layered_solver_native, None, {}),                 # not a LayeredDevsimBuilder
        (SpyLayered(), _layered_solver_native, {"grid_n_z": 32}, {"grid_n_z": 32}),
        (SpyLayered(), None, {"grid_n_x": 64, "grid_n_z": 48}, {"grid_n_x": 64, "grid_n_z": 48}),
    ]
    for carrier, solver, kw, want in cases:
        with pytest.raises(_Stop):
            run_pipeline(design, _sweep(), verbose=False, carrier_solver=carrier, optical_solver=solver,
                         carrier_solve_kwargs=kw)
        assert carrier.calls == [want]


# ---------------------------------------------------------------------------------------------------
# DEVSIM: the gated film
# ---------------------------------------------------------------------------------------------------

def _solve_fine(vt):
    from dynameta.carriers.devsim_layered import LayeredDevsimBuilder
    from dynameta.sweep import BiasPoint
    b = LayeredDevsimBuilder(G.film_design(**G.FINE))
    bias = BiasPoint({"top": vt, "bot": 0.0}, "b")
    b.solve(bias, grid_n_x=8)
    return b, bias


def test_native_grid_sheet_density_equals_raw_nodes():
    """Gated film, fine mesh (189 nodes across 6 nm), top gate +5 V, no edge ground. ORACLE: the
    trapezoid of (n - n_bg) over the raw DEVSIM node column (read from DEVSIM directly). The native grid
    must reproduce it to 1e-12 relative (columns differ laterally by round-off only); the v0.15.1 32-point
    grid misses it by +7.40e-3 (re-measured here as a consistency check of the reproduction)."""
    pytest.importorskip("devsim")
    b, bias = _solve_fine(5.0)
    try:
        z, n = G.column(b.device, 20e-9)
        z0, n0 = G.column(b.device, 0.0)
        sheet_raw = trapz(n - G.N_BG, z)
        native = b._to_carrier_field(bias, 8, None).regions["film"]
        uniform = b._to_carrier_field(bias, 8, 32).regions["film"]
    finally:
        b.teardown()
    assert native.grid_axes_m["y"].size == z.size == 189
    assert np.array_equal(native.grid_axes_m["y"], z) and np.array_equal(z0, z)
    ng = native.grid_fields["electron_density_m3"]
    assert np.array_equal(ng[0], n0)                                     # x = 0 column: the nodes, bitwise
    sheet_native = trapz(ng.mean(axis=0) - G.N_BG, native.grid_axes_m["y"])
    assert abs(sheet_native / sheet_raw - 1.0) < 1e-12
    ug = uniform.grid_fields["electron_density_m3"]
    sheet_32 = trapz(ug.mean(axis=0) - G.N_BG, uniform.grid_axes_m["y"])
    assert 6.0e-3 < sheet_32 / sheet_raw - 1.0 < 9.0e-3                  # MEASURED +7.40e-3
    assert sheet_raw == pytest.approx(4.59252366e17, rel=1e-6)           # fine-mesh value of the fixture


def test_grid_n_z_integer_path_bit_identical_and_default_unchanged():
    """Same-tree A/B: solve(grid_n_z=32) -- and the unchanged default -- give exactly what v0.15.1 computed,
    resample_to_grid(nodes, fields, (grid_n_x, 32)) on the region's raw nodes (np.array_equal)."""
    pytest.importorskip("devsim")
    from dynameta.carriers.devsim_layered import LayeredDevsimBuilder
    from dynameta.sweep import BiasPoint
    assert inspect.signature(LayeredDevsimBuilder.solve).parameters["grid_n_z"].default == 32
    assert inspect.signature(LayeredDevsimBuilder.solve).parameters["grid_n_x"].default == 256
    b = LayeredDevsimBuilder(G.film_design(x_lines=dict(x_spacing_edge_m=20e-9, x_spacing_feature_edge_m=20e-9,
                                                       x_spacing_feature_mid_m=20e-9)))
    bias = BiasPoint({"top": 2.0, "bot": 0.0}, "b")
    try:
        cf = b.solve(bias, grid_n_x=16, grid_n_z=32)
        cf_again = b._to_carrier_field(bias, 16, 32)
    finally:
        b.teardown()
    reg = cf.regions["film"]
    old = resample_to_grid(reg.nodes_m, dict(reg.node_fields), (16, 32))
    for got in (reg, cf_again.regions["film"]):
        assert np.array_equal(got.grid_axes_m["x"], old["axis_0"])
        assert np.array_equal(got.grid_axes_m["y"], old["axis_1"])
        for k in ("electron_density_m3", "potential_V"):
            assert np.array_equal(got.grid_fields[k], old[k])


def test_native_grid_fallback_warns_and_uses_uniform_grid(monkeypatch):
    """A region that is not a tensor grid cannot be returned natively: AdvisoryWarning naming the region,
    and the uniform grid_n_x x 32 resample (exactly the integer path) instead."""
    pytest.importorskip("devsim")
    from dynameta.carriers import devsim_layered as DL
    from dynameta.sweep import BiasPoint
    b = DL.LayeredDevsimBuilder(G.film_design(x_lines=dict(x_spacing_edge_m=20e-9, x_spacing_feature_edge_m=20e-9,
                                                          x_spacing_feature_mid_m=20e-9)))
    bias = BiasPoint({"top": 1.0, "bot": 0.0}, "b")

    def refuse(*a, **k):
        raise NotATensorGridError("forced")

    try:
        b.solve(bias, grid_n_x=8)
        ref = b._to_carrier_field(bias, 8, 32).regions["film"]
        monkeypatch.setattr(DL, "resample_tensor_grid", refuse)
        with pytest.warns(AdvisoryWarning, match="'film' is not a tensor grid"):
            got = b._to_carrier_field(bias, 8, None).regions["film"]
    finally:
        b.teardown()
    assert got.grid_axes_m["y"].size == 32
    assert np.array_equal(got.grid_fields["electron_density_m3"], ref.grid_fields["electron_density_m3"])


class _StubGeo:
    class mesh:
        ne = 0
        nv = 0


class _StubOptics:
    """No ngsolve: alignment() / mesh_regions() for the film only (its true bbox)."""
    def __init__(self, design):
        self.design = design

    def build(self):
        return _StubGeo()

    def alignment(self):
        from dynameta.core import NM
        from dynameta.core.alignment import GeometryAlignment, RegionAlignment
        P = self.design.unit_cell.period_x_m
        zlo, zhi = self.design.z_intervals()["film"]
        return GeometryAlignment(unit_scale=NM, region_alignments=[
            RegionAlignment("film", "film", (0.0, P, 0.0, P, zlo, zhi), stack_axis="y")],
            fixed_eps_regions={})

    def mesh_regions(self):
        return ["film"]


def test_run_pipeline_layered_solver_sees_native_z():
    """End to end (default carrier solver, default mesh): an optical solver with native_carrier_grid = True
    receives the film's EpsField on the native z mesh lines (every node y, remapped into the film's z
    interval, non-uniform); without the attribute it receives the 32-point uniform axis as before."""
    pytest.importorskip("devsim")
    from dynameta.core.interfaces import OpticalResult
    from dynameta.core.eps_field import axis_spacing_spread
    from dynameta.pipeline import run_pipeline
    seen = {}

    def make(tag, native):
        def solver(design, geo, eps_by_region, lambda_m, n_super, n_sub):
            seen[tag] = np.asarray(eps_by_region["film"].z_axis_u, float)
            return OpticalResult(r=0j, R=0.0, phase_deg=0.0, solve_time_s=0.0)
        if native:
            solver.native_carrier_grid = True
        return solver

    design = G.film_design(x_lines=dict(x_spacing_edge_m=20e-9, x_spacing_feature_edge_m=20e-9,
                                        x_spacing_feature_mid_m=20e-9))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for tag, native in (("native", True), ("plain", False)):
            run_pipeline(design, _sweep(), verbose=False, optical_builder=_StubOptics(design),
                         optical_solver=make(tag, native))
    zlo, zhi = design.z_intervals()["film"]
    zn = seen["native"]
    assert zn.size > 100 and axis_spacing_spread(zn) > 1.0              # graded native lines
    assert zn[0] == pytest.approx(zlo * 1e9, abs=1e-9) and zn[-1] == pytest.approx(zhi * 1e9, abs=1e-9)
    assert seen["plain"].size == 32 and axis_spacing_spread(seen["plain"]) < AXIS_UNIFORM_RTOL
