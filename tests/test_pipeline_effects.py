"""Solver-free test that run_pipeline THREADS the new n_to_eps + extra_fields seam (audit Finding 5):
the EffectModel family must be reachable from the orchestrator, not just from a hand-rolled
assemble_eps loop. Uses stub carrier solver / optical builder / optical solver, so no devsim/ngsolve.
Run: python -m pytest tests/test_pipeline_effects.py -q
"""
import numpy as np
import pytest

from dynameta.core import NM, EffectEpsMap
from dynameta.core.effects import ThermoOpticModel
from dynameta.core.alignment import GeometryAlignment, RegionAlignment
from dynameta.core.carrier_field import CarrierField, CarrierRegion, ELECTRON_DENSITY
from dynameta.core.interfaces import OpticalResult
from dynameta.materials import Material, MaterialRegistry, ConstantOptical
from dynameta.geometry import UnitCell, Stack, Layer, Design
from dynameta.geometry.specs import OpticalSpec
from dynameta.sweep import Sweep, BiasPoint
from dynameta.pipeline import run_pipeline

PERIOD = 300e-9
EPS0 = complex(2.5 ** 2, 0.0)
LAM_NM = 1300.0


def _design():
    reg = MaterialRegistry()
    reg.add(Material("air", ConstantOptical(1.0 + 0j)))
    reg.add(Material("mat", ConstantOptical(EPS0)))
    return Design(name="t", unit_cell=UnitCell.square(PERIOD),
                  stack=Stack(layers=[Layer("film", 10e-9, "mat")],
                              superstrate_material="air", substrate_material="air"),
                  electrodes=[], materials=reg,
                  optical=OpticalSpec(polarization="y", incidence_angle_deg=0.0, lift="identity"))


class _StubCarrier:
    """Returns a trivial uniform 3D CarrierField (n = n_bg) for region 'semi' (material 'mat')."""
    def solve(self, bp):
        nx = ny = nz = 3
        x = np.linspace(0.0, PERIOD, nx); y = np.linspace(0.0, PERIOD, ny); z = np.linspace(0.0, 10e-9, nz)
        reg = CarrierRegion(name="semi", role="semiconductor", material="mat",
                            nodes_m=np.zeros((1, 3)), node_fields={},
                            grid_axes_m={"x": x, "y": y, "z": z},
                            grid_fields={ELECTRON_DENSITY: np.full((nx, ny, nz), 4e26)})
        return CarrierField(bias_label=bp.label, voltages=dict(bp.voltages), ndim=3,
                            temperature_K=300.0, regions={"semi": reg},
                            n_bg_by_region={"semi": 4e26}, unit_cell_m=(PERIOD, PERIOD))


class _StubGeo:
    class mesh:
        ne = 0
        nv = 0


class _StubBuilder:
    def build(self):
        return _StubGeo()

    def alignment(self):
        return GeometryAlignment(
            unit_scale=NM,
            region_alignments=[RegionAlignment("semi", "semi",
                               (0.0, PERIOD, 0.0, PERIOD, 0.0, 10e-9), stack_axis="z")],
            fixed_eps_regions={})

    def mesh_regions(self):
        return ["semi"]


def _capturing_solver(store):
    def _solve(design, geo, eps_by_region, lambda_m, n_super, n_sub):
        store.append(eps_by_region)
        return OpticalResult(r=0j, R=0.0, phase_deg=0.0, solve_time_s=0.0)
    return _solve


def _run(**kw):
    store = []
    design = _design()
    sweep = Sweep(bias_points=[BiasPoint(label="b0", voltages={"gate": 0.0})],
                  wavelengths_nm=[LAM_NM])
    run_pipeline(design, sweep, verbose=False, carrier_solver=_StubCarrier(),
                 optical_builder=_StubBuilder(), optical_solver=_capturing_solver(store), **kw)
    return store[0]["semi"]


def test_default_path_is_material_eps():
    # no n_to_eps / no extra_fields -> the carrier/material path (ConstantOptical -> EPS0 everywhere)
    ef = _run()
    assert np.allclose(np.asarray(ef.values_zyx), EPS0)


def test_effect_model_reaches_solver_via_pipeline():
    # an EffectEpsMap(ThermoOptic) + extra_fields={'T':...} must reach the solver with the
    # temperature-shifted eps -- the modulation-mechanism family driven through the orchestrator.
    tom = ThermoOpticModel(eps_ref=EPS0, dn_dT=2.0e-4, T_ref=300.0)
    ef = _run(n_to_eps=EffectEpsMap(_design().materials, effects={"mat": tom}),
              extra_fields={"T": 360.0})
    assert ef.is_uniform and not ef.is_tensor                       # uniform field-effect -> scalar
    assert np.isclose(ef.scalar, complex(tom.eps({"T": 360.0}, LAM_NM * 1e-9)), rtol=1e-12)
    assert ef.scalar != EPS0                                        # genuinely modulated


def test_extra_fields_callable_is_resolved_per_bias():
    # a callable extra_fields(bias_point) -> dict must be evaluated with the bias (so the applied
    # field/temperature can vary with bias -- the usual driver->extra_fields pattern).
    tom = ThermoOpticModel(eps_ref=EPS0, dn_dT=2.0e-4, T_ref=300.0)
    seen = {}

    def _ef(bp):
        seen["label"] = bp.label
        return {"T": 410.0}
    ef = _run(n_to_eps=EffectEpsMap(_design().materials, effects={"mat": tom}), extra_fields=_ef)
    assert seen["label"] == "b0"                                    # callable saw the bias point
    assert np.isclose(ef.scalar, complex(tom.eps({"T": 410.0}, LAM_NM * 1e-9)), rtol=1e-12)


# ---- dispersive end-media: the sweep-aware fast path must NOT freeze n_super/n_sub at band-centre ----
from dynameta.materials import TabulatedOptical                     # noqa: E402
from dynameta.optics.tmm_reference import end_media_indices         # noqa: E402

_LAMS_NM = [1200.0, 1300.0, 1450.0]


def _design_sub(sub_optical):
    """The film design but with a chosen SUBSTRATE optical model (dispersive or constant)."""
    reg = MaterialRegistry()
    reg.add(Material("air", ConstantOptical(1.0 + 0j)))
    reg.add(Material("mat", ConstantOptical(EPS0)))
    reg.add(Material("sub", sub_optical))
    return Design(name="t", unit_cell=UnitCell.square(PERIOD),
                  stack=Stack(layers=[Layer("film", 10e-9, "mat")],
                              superstrate_material="air", substrate_material="sub"),
                  electrodes=[], materials=reg,
                  optical=OpticalSpec(polarization="y", incidence_angle_deg=0.0, lift="identity"))


def _sweep_aware_recorder(rec):
    """A sweep-aware optical solver: BOTH the per-call form and .solve_sweep record the n_sub they were
    handed (per-wavelength vs the one frozen band-centre value)."""
    def _solve(design, geo, eps_by_region, lambda_m, n_super, n_sub):
        rec["per_call"].append((float(lambda_m), complex(n_sub)))
        return OpticalResult(r=0j, R=float(abs(n_sub)), phase_deg=0.0, solve_time_s=0.0)

    def _solve_sweep(design, geo, assemble_at, lams, n_super, n_sub):
        rec["sweep_nsub"].append(complex(n_sub))                    # the ONE frozen value for the band
        return [OpticalResult(r=0j, R=float(abs(n_sub)), phase_deg=0.0, solve_time_s=0.0) for _ in lams]
    _solve.solve_sweep = _solve_sweep
    return _solve


def _run_sweep(design, rec):
    sweep = Sweep(bias_points=[BiasPoint(label="b0", voltages={"gate": 0.0})], wavelengths_nm=_LAMS_NM)
    run_pipeline(design, sweep, verbose=False, carrier_solver=_StubCarrier(),
                 optical_builder=_StubBuilder(), optical_solver=_sweep_aware_recorder(rec))


def test_dispersive_end_media_disables_sweep_fast_path():
    # A DISPERSIVE substrate (eps 4->6 across the band): the band-centre freeze would give the wrong
    # n_sub off-centre AND disagree with the per-wavelength path. run_pipeline must DETECT this, warn,
    # disable the sweep fast path, and solve per-wavelength so n_sub tracks lambda. Discriminator: the
    # PRE-FIX code calls .solve_sweep ONCE with the frozen band-centre n_sub (so all wavelengths share
    # one value) -- this test asserts the OPPOSITE (per-call, n_sub varying per lambda).
    sub = TabulatedOptical(lambda_m=np.array([1.15e-6, 1.50e-6]), eps_complex=np.array([4.0 + 0j, 6.0 + 0j]))
    d = _design_sub(sub)
    rec = {"per_call": [], "sweep_nsub": []}
    with pytest.warns(RuntimeWarning, match="DISPERSIVE"):
        _run_sweep(d, rec)
    assert rec["sweep_nsub"] == []                                  # fast path DISABLED (no frozen solve)
    assert len(rec["per_call"]) == len(_LAMS_NM)                    # solved per wavelength instead
    for (lam_m, n_sub), lam_nm in zip(rec["per_call"], _LAMS_NM):
        _, n_sub_true = end_media_indices(d, lam_nm * 1e-9)         # the correct PER-lambda substrate index
        assert abs(n_sub - n_sub_true) < 1e-12
    seen = [n for _, n in rec["per_call"]]
    assert abs(seen[0] - seen[-1]) > 1e-2                           # n_sub genuinely VARIES (not frozen)


def test_nondispersive_end_media_keeps_sweep_fast_path():
    # A CONSTANT substrate: end media are wavelength-independent, so the band-centre freeze is exact and
    # the one-broadband-solve-per-bias fast path is retained (no regression to the common case).
    d = _design_sub(ConstantOptical(complex(2.0 ** 2, 0.0)))
    rec = {"per_call": [], "sweep_nsub": []}
    _run_sweep(d, rec)
    assert len(rec["sweep_nsub"]) == 1 and rec["per_call"] == []    # fast path used: ONE solve per bias
    assert abs(rec["sweep_nsub"][0] - 2.0) < 1e-12                  # n_sub = sqrt(4) = 2
