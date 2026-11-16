"""v0.16 native carrier grid end to end (docs/audit/2026-11-14-v0.16-carrier-grid-mesh.md): run_pipeline with the
REAL layered optical solver, optics.tmm_reference.make_layered_tmm_solver (which carries native_carrier_grid =
True), hands the solver the carrier film on its native z mesh lines, and the reflection moves against the v0.15.1
32-point uniform resample on a gated thin film.

Device: tests/_wiring_fixture.py (the generic gated-film cell) with an optically transparent top gate (a metal for
the carrier solve, a 10 nm weakly absorbing film, eps 3.8+0.05i, for the optics), laterally uniform, x lines every
20 nm, top gate +5 V and +6.5 V; 1200 nm, p polarization at 60 deg (the film's normal response matters there: its
Drude eps(n(z)) crosses zero inside the accumulation layer). Default Mesh2DSpec: 204 z nodes across the 6 nm film.

ORACLE. A uniform resample with many points converges to the piecewise-linear interpolant of the node values,
which the native grid slices exactly up to the second-order slab rule on its own (finer) intervals: r(native)
must agree with r(grid_n_z = 4096) far better than r(32) does (r(4096) itself is converged to ~5e-8: MEASURED
512 / 1024 / 2048 points 3.4e-6 / 7.9e-7 / 1.5e-7 from it, at +5 V). MEASURED (Windows, DEVSIM 2.11.2), top gate
3, 4, 4.5, 5, 5.25, 5.5, 5.75, 6, 6.5 and 7 V:
  |r(native) - r(4096)| = 3.5e-7 / 2.9e-6 / 9.4e-6 / 7.5e-6 / 4.8e-6 / 3.3e-6 / 2.4e-6 / 1.8e-6 / 1.2e-6 / 8.9e-7,
  |r(32) - r(4096)| / |r(native) - r(4096)| = 52 / 45 / 71 / 285 / 567 / 833 / 987 / 1092 / 1543 / 2640.
The native grid's own error is the second-order slab rule on its intervals; it peaks (9.4e-6 at +4.5 V, 7.5e-6 at
+5 V) where the face density reaches the 1200 nm epsilon-near-zero density (9.3e26 m^-3; 9.9e26 at +5 V), i.e. where
eps ~ 0 over the flat top of the accumulation peak. The 32-point resample's error is set by the accumulation
layer (6.7e-4 at +4.5 V, 2.1e-3 at +5 V, 1.9e-3 at +6.5 V), so from +5 V on it exceeds the native error by >= 285x. Bars (every bias in
3..7 V): native error < 2e-5 (2.1x the largest measured), resample/native > 100 at +5 and +6.5 V (2.85x below the
smallest measured from +5 V on). Against the bias modulation |r(V) - r(0 V)| = 1.02e-2 (+5 V) / 9.8e-3 (+6.5 V) the
v0.15.1 default pipeline was off by 21 % / 19 %. At normal incidence (x polarization) the same film gives 5.0e-6
against a 6.5e-4 modulation (0.8 %, at +5.5 V).

HOW IT IS RUN. A Spy carrier records the keywords run_pipeline passes to solve() (the automatic selection is
the point) and solves with abs_tol=1e-14 V, at the default lateral grid (grid_n_x=256) and the default lift.
Until v0.16's slicer fix this needed grid_n_x=2 and lift="extrude": core.layered.slice_eps_field compared
every sample with the lateral MEAN at rtol 1e-12, and that mean, over 256 x 256 samples of the non-contiguous
values_zyx the bridge builds, carries 1.75e-12 of summation round-off, so the uniform film was called
structured and the layered TMM refused it. It now compares with the first lateral sample against the slice's
max |eps| (tests/test_layered.py); the numbers above are measured at the default grid.

Without the native grid (v0.15.1) run_pipeline passes no grid argument (the spy sees {}) and r(native) is r(32).
"""
import dataclasses
import warnings

import pytest

pytest.importorskip("devsim")

import _wiring_fixture as W                                              # noqa: E402


class _StubGeo:
    class mesh:
        ne = 0
        nv = 0


class _StubOptics:
    """No ngsolve: the alignment of the film only (its true bbox); the other layers are uniform."""

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


def test_run_pipeline_hands_the_layered_tmm_solver_the_native_z_grid():
    from dynameta.carriers.devsim_layered import LayeredDevsimBuilder
    from dynameta.geometry.specs import OpticalSpec
    from dynameta.optics.tmm_reference import make_layered_tmm_solver
    from dynameta.pipeline import run_pipeline
    from dynameta.sweep import BiasPoint, Sweep

    class Spy(LayeredDevsimBuilder):
        calls = []
        z_nodes = []

        def solve(self, bias, **kw):
            Spy.calls.append(dict(kw))
            cf = super().solve(bias, abs_tol=1e-14, **kw)
            Spy.z_nodes.append(cf.regions["film"].grid_axes_m["y"].size)
            return cf

    design = W.film_design(transparent_gate=True,
                           x_lines=dict(x_spacing_edge_m=20e-9, x_spacing_feature_edge_m=20e-9,
                                        x_spacing_feature_mid_m=20e-9))
    design = dataclasses.replace(design, optical=OpticalSpec(polarization="p", incidence_angle_deg=60.0))
    solver = make_layered_tmm_solver()
    assert solver.native_carrier_grid is True

    def r_of(vt, kw):
        sweep = Sweep(bias_points=[BiasPoint({"top": vt, "bot": 0.0}, "b")], wavelengths_nm=[1200.0])
        rows = run_pipeline(design, sweep, verbose=False, carrier_solver=Spy(design),
                            optical_builder=_StubOptics(design), optical_solver=solver, carrier_solve_kwargs=kw)
        return complex(rows[0].result.r)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        r0 = r_of(0.0, None)
        r = {vt: {tag: r_of(vt, kw) for tag, kw in (("native", None), ("u32", {"grid_n_z": 32}),
                                                     ("u4096", {"grid_n_z": 4096}))}
             for vt in (5.0, 6.5)}
    assert Spy.calls[0] == {"grid_n_z": None}
    assert Spy.calls[1:4] == [{"grid_n_z": None}, {"grid_n_z": 32}, {"grid_n_z": 4096}]
    assert Spy.z_nodes[1] > 100 and Spy.z_nodes[2] == 32       # the native graded lines vs the resample
    for vt, rv in r.items():
        d_native = abs(rv["native"] - rv["u4096"])
        d_32 = abs(rv["u32"] - rv["u4096"])
        modulation = abs(rv["native"] - r0)
        assert d_native < 2e-5, (vt, d_native)                 # module docstring: 2.1x the 3..7 V maximum
        assert d_32 > 5e-4 and d_32 > 100.0 * d_native, (vt, d_32, d_native)
        assert abs(rv["u32"] - rv["native"]) > 0.05 * modulation, vt


def test_pipeline_result_is_the_same_with_and_without_each_solver_wrapper(tmp_path):
    """Solver wrappers keep the capability flags. The result cache (and the EMT screen, a wrapper of the Berreman
    bridge) dropped the inner solver's native_carrier_grid flag, so the carriers came back on the 32-point grid:
    same device as above at +5 V, r(TMM) = 0.6092251859+0.7433366229j vs the 32-point answer a wrapper that drops the
    flag returns, 0.6098035783+0.7412784093j (|dr| = 2.14e-3, 21 % of the bias modulation), with no warning (MEASURED
    as the grid_n_z=32 solve above). Now every wrapper gives its inner solver's r bit for bit, with the same
    carrier keywords."""
    from dynameta.cache import cached_optical_solver
    from dynameta.carriers.devsim_layered import LayeredDevsimBuilder
    from dynameta.geometry.specs import OpticalSpec
    from dynameta.optics.lumenairy_bridge.berreman_backend import make_lumenairy_berreman_solver
    from dynameta.optics.lumenairy_bridge.emt_screen import make_lumenairy_emt_screen_solver
    from dynameta.optics.tmm_reference import make_layered_tmm_solver
    from dynameta.pipeline import run_pipeline
    from dynameta.sweep import BiasPoint, Sweep

    class Spy(LayeredDevsimBuilder):
        calls = []

        def solve(self, bias, **kw):
            Spy.calls.append(dict(kw))
            return super().solve(bias, abs_tol=1e-14, **kw)

    design = W.film_design(transparent_gate=True,
                           x_lines=dict(x_spacing_edge_m=20e-9, x_spacing_feature_edge_m=20e-9,
                                        x_spacing_feature_mid_m=20e-9))
    design = dataclasses.replace(design, optical=OpticalSpec(polarization="p", incidence_angle_deg=60.0))
    sweep = Sweep(bias_points=[BiasPoint({"top": 5.0, "bot": 0.0}, "b")], wavelengths_nm=[1200.0])

    def r_of(solver):
        Spy.calls.clear()
        rows = run_pipeline(design, sweep, verbose=False, carrier_solver=Spy(design),
                            optical_builder=_StubOptics(design), optical_solver=solver)
        assert Spy.calls == [{"grid_n_z": None}]
        return complex(rows[0].result.r)

    tmm = make_layered_tmm_solver()
    ber = make_lumenairy_berreman_solver()
    emt = make_lumenairy_emt_screen_solver()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        r_tmm, r_ber = r_of(tmm), r_of(ber)
        pairs = [("cache(tmm)", r_of(cached_optical_solver(tmm, str(tmp_path / "a.h5"))), r_tmm),
                 ("emt_screen", r_of(emt), r_ber),
                 ("cache(emt_screen)", r_of(cached_optical_solver(emt, str(tmp_path / "b.h5"))), r_ber)]
    for name, r, ref in pairs:
        assert r == ref, (name, r, ref)
    assert abs(r_tmm - r_ber) < 1e-12
    # the native-grid answer (regression value), 2.1e-3 from the 32-point one
    assert abs(r_tmm - (0.6092251859 + 0.7433366229j)) < 1e-6
