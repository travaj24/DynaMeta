"""v0.16: optical-solver capability flags survive every wrapper in the library.

The pipeline changes what it hands an optical solver according to flags the solver declares
(core.interfaces.OPTICAL_SOLVER_CAPABILITIES; today native_carrier_grid: a LayeredDevsimBuilder then returns
the carriers on their native z mesh lines). Before the fix the result cache (cache.OpticalSolverCache)
dropped the flag, so the SAME run gave different carriers cached and uncached (|dr| = 2.73e-3 on the generic gated
film of tests/_wiring_fixture, 27 % of the bias modulation; tests/test_pipeline_native_tmm.py measures it with
DEVSIM). The EMT screen, which delegates to the Berreman bridge, dropped it as well. The fix is structural:
the pipeline reads flags through core.interfaces.solver_capability, which follows the standard __wrapped__
link, and every wrapper declares that link (declare_wrapped_solver). This file holds the solver-free gates:

  * each wrapper exposes every capability of its inner solver (FAILED before the fix: the cache and the
    screen reported no native_carrier_grid);
  * run_pipeline picks the same carrier keywords with and without each wrapper (FAILED before the fix: {} vs
    {'grid_n_z': None} for the cache);
  * a cache around a per-wavelength solver no longer advertises solve_sweep (found while fixing: the cache
    always had the method, so run_pipeline took the sweep path for a cached layered TMM and raised
    AttributeError on every multi-wavelength sweep);
  * census: no library module reads a capability flag except through solver_capability.
"""
import os
import re
import sys
import types

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _wrappers(tmp_path):
    """(name, inner, wrapper) for every solver wrapper in the library."""
    from dynameta.cache import cached_optical_solver
    from dynameta.optics.lumenairy_bridge.berreman_backend import make_lumenairy_berreman_solver
    from dynameta.optics.lumenairy_bridge.emt_screen import make_lumenairy_emt_screen_solver
    from dynameta.optics.lumenairy_bridge.rcwa_backend import make_lumenairy_rcwa_solver
    from dynameta.optics.tmm_reference import make_layered_tmm_solver
    tmm = make_layered_tmm_solver()
    rcwa = make_lumenairy_rcwa_solver(n_orders=3)
    emt = make_lumenairy_emt_screen_solver()
    return [
        ("cache(tmm)", tmm, cached_optical_solver(tmm, str(tmp_path / "a.h5"))),
        ("cache(rcwa)", rcwa, cached_optical_solver(rcwa, str(tmp_path / "b.h5"))),
        ("emt_screen", make_lumenairy_berreman_solver(), emt),
        ("cache(emt_screen)", emt, cached_optical_solver(emt, str(tmp_path / "c.h5"))),
        ("cache(cache(tmm))", tmm,
         cached_optical_solver(cached_optical_solver(tmm, str(tmp_path / "d.h5")), str(tmp_path / "e.h5"),
                               tag="outer")),
    ]


def test_every_wrapper_exposes_the_inner_solver_capabilities(tmp_path):
    from dynameta.core.interfaces import OPTICAL_SOLVER_CAPABILITIES, solver_capability
    for name, inner, wrapped in _wrappers(tmp_path):
        for cap in OPTICAL_SOLVER_CAPABILITIES:
            want = solver_capability(inner, cap)
            assert want is True, (name, cap)               # every inner solver here declares it
            assert solver_capability(wrapped, cap) is want, (name, cap)
        # direct attribute reads keep working on the cache (user code, diagnostics)
        if hasattr(wrapped, "inner"):
            assert wrapped.native_carrier_grid is True, name


def test_wrapper_can_override_and_unknown_capability_is_refused():
    from dynameta.core.interfaces import declare_wrapped_solver, solver_capability

    def inner(*a):
        return None
    inner.native_carrier_grid = True

    def plain(*a):
        return None
    assert solver_capability(plain, "native_carrier_grid") is False

    def w(*a):
        return None
    declare_wrapped_solver(w, inner)
    assert solver_capability(w, "native_carrier_grid") is True
    w.native_carrier_grid = False                         # a wrapper that must not inherit says so
    assert solver_capability(w, "native_carrier_grid") is False
    declare_wrapped_solver(inner, w)                      # a cycle terminates
    assert solver_capability(w, "native_carrier_grid") is False
    with pytest.raises(ValueError, match="unknown capability"):
        solver_capability(inner, "native_carier_grid")


def test_pipeline_carrier_keywords_are_the_same_with_and_without_each_wrapper(tmp_path, monkeypatch):
    """_carrier_solve_kwargs (what run_pipeline passes to carrier.solve) with a LayeredDevsimBuilder carrier.
    The DEVSIM module is stood in by a stub in sys.modules (the selection only looks the class up there)."""
    from dynameta.pipeline import _carrier_solve_kwargs

    class LayeredDevsimBuilder:                           # stub with the real class name
        pass
    stub = types.ModuleType("dynameta.carriers.devsim_layered")
    stub.LayeredDevsimBuilder = LayeredDevsimBuilder
    monkeypatch.setitem(sys.modules, "dynameta.carriers.devsim_layered", stub)
    carrier = LayeredDevsimBuilder()
    for name, inner, wrapped in _wrappers(tmp_path):
        want = _carrier_solve_kwargs(carrier, inner, None)
        assert want == {"grid_n_z": None}, name
        assert _carrier_solve_kwargs(carrier, wrapped, None) == want, name


PERIOD = 400e-9


class _UniformCarrier:
    """A solver-free carrier: a 6 nm film whose density grows with the 'top' bias, on a 3 x 3 x 5 grid."""

    def solve(self, bp):
        from dynameta.core.carrier_field import ELECTRON_DENSITY, CarrierField, CarrierRegion
        x = np.linspace(0.0, PERIOD, 3)
        z = np.linspace(0.0, 6e-9, 5)
        n = np.broadcast_to(3e26 * (1.0 + 0.1 * float(bp.voltages["top"])) * (1.0 + 0.2 * z / 6e-9),
                            (3, 3, 5)).copy()
        reg = CarrierRegion(name="semi", role="semiconductor", material="FILM", nodes_m=np.zeros((1, 3)),
                            node_fields={}, grid_axes_m={"x": x, "y": x.copy(), "z": z},
                            grid_fields={ELECTRON_DENSITY: n})
        return CarrierField(bias_label=bp.label, voltages=dict(bp.voltages), ndim=3, temperature_K=300.0,
                            regions={"semi": reg}, n_bg_by_region={"semi": 3e26},
                            unit_cell_m=(PERIOD, PERIOD))


def _film_design():
    from dynameta.constants import M_E
    from dynameta.geometry import Design, Layer, Stack, UnitCell
    from dynameta.geometry.specs import OpticalSpec
    from dynameta.materials import ConstantOptical, DrudeOptical, Material, MaterialRegistry
    reg = MaterialRegistry()
    reg.add(Material("air", ConstantOptical(1.0 + 0j)))
    reg.add(Material("OX", ConstantOptical(2.1 + 0j)))
    reg.add(Material("FILM", DrudeOptical(eps_inf=4.0, m_opt_kg=0.30 * M_E, gamma_rad_s=1.2e14)))
    return Design(name="t", unit_cell=UnitCell.square(PERIOD),
                  stack=Stack(layers=[Layer("film", 6e-9, "FILM"), Layer("ox", 20e-9, "OX")],
                              superstrate_material="air", substrate_material="OX"),
                  electrodes=[], materials=reg,
                  optical=OpticalSpec(polarization="p", incidence_angle_deg=40.0, lift="identity"))


class _Optics:
    def __init__(self, design):
        self.design = design

    def build(self):
        class G:
            class mesh:
                ne = nv = 0
        return G()

    def alignment(self):
        from dynameta.core import NM
        from dynameta.core.alignment import GeometryAlignment, RegionAlignment
        zlo, zhi = self.design.z_intervals()["film"]
        return GeometryAlignment(unit_scale=NM, region_alignments=[
            RegionAlignment("film", "semi", (0.0, PERIOD, 0.0, PERIOD, zlo, zhi), stack_axis="z")],
            fixed_eps_regions={})

    def mesh_regions(self):
        return ["film"]


def test_cached_per_wavelength_solver_runs_a_multi_wavelength_sweep(tmp_path):
    """Before the fix this raised AttributeError ('inner solver has no solve_sweep'): the cache advertised a
    solve_sweep method for a per-wavelength inner solver. Now: no solve_sweep, and the cached rows equal the
    uncached rows bit for bit (cold) and again from the store (warm)."""
    from dynameta.cache import cached_optical_solver
    from dynameta.optics.tmm_reference import make_layered_tmm_solver
    from dynameta.pipeline import run_pipeline
    from dynameta.sweep import BiasPoint, Sweep
    design = _film_design()
    sweep = Sweep(bias_points=[BiasPoint({"top": 0.0}, "a"), BiasPoint({"top": 1.0}, "b")],
                  wavelengths_nm=[1100.0, 1250.0, 1500.0])
    tmm = make_layered_tmm_solver()
    cached = cached_optical_solver(tmm, str(tmp_path / "s.h5"), autosave=False)
    assert not hasattr(cached, "solve_sweep")

    def run(solver):
        rows = run_pipeline(design, sweep, verbose=False, carrier_solver=_UniformCarrier(),
                            optical_builder=_Optics(design), optical_solver=solver)
        return np.array([complex(r.result.r) for r in rows])
    ref = run(tmm)
    assert np.array_equal(run(cached), ref)
    assert cached.misses == 6 and cached.hits == 0
    assert np.array_equal(run(cached), ref)
    assert cached.hits == 6


_FLAG_READ = re.compile(r"""(?:getattr|hasattr)\(\s*[\w.]+\s*,\s*["'](?:{})["']""")


def test_census_capability_flags_are_read_only_through_solver_capability():
    """No library module reads a capability flag with getattr/hasattr (which does not follow wrappers);
    core.interfaces.solver_capability is the one reader."""
    from dynameta.core.interfaces import OPTICAL_SOLVER_CAPABILITIES
    pat = re.compile(_FLAG_READ.pattern.format("|".join(OPTICAL_SOLVER_CAPABILITIES)))
    hits = []
    for root, _dirs, files in os.walk(os.path.join(REPO, "dynameta")):
        for f in files:
            if f.endswith(".py"):
                p = os.path.join(root, f)
                with open(p, encoding="utf-8") as fh:
                    for i, line in enumerate(fh, 1):
                        if pat.search(line):
                            hits.append("{}:{}".format(os.path.relpath(p, REPO), i))
    assert hits == [], hits
