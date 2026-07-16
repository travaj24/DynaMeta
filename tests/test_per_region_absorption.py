"""Solver-free tests for the per-region absorbed-power map (driver D2). The FEM side is gated by
validation/per_region_absorption.py (needs NGSolve); here we cover the TMM per-layer path, the
OpticalResult schema, and the cache not-cached contract."""
import numpy as np
import pytest

from dynameta.core.interfaces import OpticalResult
from dynameta.core.layered import LayeredStack, LayeredSlab


def test_optical_result_field_defaults_none():
    r = OpticalResult(r=0.1 + 0j, R=0.01, phase_deg=0.0, solve_time_s=0.0)
    assert r.per_region_absorption is None


def _stack():
    return LayeredStack(n_super=1.0 + 0j, n_sub=1.0 + 0j,
                        slabs=[LayeredSlab(thickness_m=120e-9, eps=(1.6 ** 2) + 0j),
                               LayeredSlab(thickness_m=80e-9, eps=4.0 + 0.5j),
                               LayeredSlab(thickness_m=120e-9, eps=(1.6 ** 2) + 0j)])


def test_tmm_per_layer_map_closure_and_zeros():
    pytest.importorskip("tmm")
    from dynameta.optics.tmm_reference import layered_per_layer_absorption
    per, A = layered_per_layer_absorption(_stack(), 1300e-9)
    assert set(per) == {"slab_0", "slab_1", "slab_2"}
    assert abs(sum(per.values()) - A) < 1e-9            # per-layer fractions sum to 1 - R - T
    assert abs(per["slab_0"]) < 1e-12 and abs(per["slab_2"]) < 1e-12   # lossless slabs
    assert per["slab_1"] > 0.01                          # the lossy slab carries the absorption


def test_tmm_solver_populates_map():
    pytest.importorskip("tmm")
    from dynameta.optics.tmm_reference import TmmLayeredSolver

    class _Opt:
        polarization = "y"
        incidence_angle_deg = 0.0

    res = TmmLayeredSolver().solve(_stack(), 1300e-9, _Opt())
    assert res.per_region_absorption is not None
    assert abs(sum(res.per_region_absorption.values()) - res.A) < 1e-9


def test_cache_drops_per_region_map():
    from dynameta.cache import OpticalSolverCache
    src = OpticalResult(r=0.1 + 0j, R=0.01, phase_deg=0.0, solve_time_s=0.5, t=0.2 + 0j, T=0.04,
                        A=0.95, per_region_absorption={"s1": 0.95})
    back = OpticalSolverCache._unpack(OpticalSolverCache._pack(src))
    assert back.per_region_absorption is None            # documented: diagnostic, never cached
    assert back.R == src.R and back.A == src.A


def test_tmm_pipeline_seam_keys_by_design_layer_name():
    # audit C5-4: run_pipeline's TMM solver used to emit seam-internal 'slab_<i>' keys
    # (one per graded slice) while FEM/RCWA emit design layer names -- a backend swap
    # broke reliability post-processing with a KeyError. The pipeline seam must key by
    # layer name, summing graded slabs into their layer.
    import numpy as np
    from dynameta.core.eps_field import EpsField
    from dynameta.geometry import Design, Layer, Stack, UnitCell
    from dynameta.materials import ConstantOptical, Material, MaterialRegistry
    from dynameta.optics.tmm_reference import make_layered_tmm_solver
    reg = MaterialRegistry()
    reg.add(Material("air", ConstantOptical(1.0 + 0j)))
    reg.add(Material("clear", ConstantOptical(2.25 + 0j)))
    reg.add(Material("lossy", ConstantOptical(4.0 + 0.3j)))
    d = Design(name="t", unit_cell=UnitCell.square(300e-9),
               stack=Stack(layers=[Layer("ito_top", 120e-9, "lossy"),
                                   Layer("cap", 80e-9, "clear")],
                           superstrate_material="air", substrate_material="air"),
               electrodes=[], materials=reg)
    z = np.linspace(0.0, 120.0, 13)
    eps_z = 4.0 + 0.2j + (0.5 + 0.4j) * (z / 120.0) ** 2       # graded lossy profile
    ebr = {"ito_top": EpsField(z_axis_u=z, y_axis_u=np.zeros(1), x_axis_u=np.zeros(1),
                               values_zyx=eps_z.reshape(-1, 1, 1).astype(complex))}
    res = make_layered_tmm_solver()(d, None, ebr, 1.31e-6, 1.0 + 0j, 1.0 + 0j)
    pra = res.per_region_absorption
    assert set(pra) == {"ito_top", "cap"}                      # design names, no slab_<i>
    assert pra["cap"] == pytest.approx(0.0, abs=1e-12)         # lossless layer exactly 0
    assert pra["ito_top"] == pytest.approx(res.A, rel=1e-9)    # graded slabs sum to A
