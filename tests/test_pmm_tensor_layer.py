"""v0.16: the PMM bridge (optics/lumenairy_bridge/pmm_backend.py) accepts a laterally UNIFORM gridded
TENSOR layer (a z-graded uniaxial film under a patterned layer); before, design_to_pmm_stack raised
"carries a gridded TENSOR EpsField ... use the RCWA bridge" for every gridded tensor field (that refusal
is what the solve gates below hit before this change, so each fails there with ValueError).

Oracle: the 2-D periodic TM FEM (optics.fem_2d_tm), which shares no code with the PMM, on the same
piecewise-linear z profile. Cell: a dielectric grating (eps 4, 150 nm of a 400 nm period, 100 nm tall)
over 20 nm oxide (eps 2.1) / a 10 nm graded uniaxial film / glass (n 1.45), 1200 nm.

Error budget (MEASURED on this cell at 1200 nm): the FEM with geometric grading at every corner
(corner_mode='all') is converged to 5e-9 (orders 5 / 6 / 7); the PMM grating-corner convergence at
degree 24 with 3 elements per region is ~1.6e-7 (degree 16, 1 element: 5.6e-6); the PMM slab staircase
(endpoint average per native interval, second order) adds the rest. Gates use 2e-6 at 21 samples.
"""
import importlib.util
import subprocess
import sys
import types

import numpy as np
import pytest

pytest.importorskip("ngsolve")
lum = pytest.importorskip("lumenairy")

from dynameta.core.eps_field import EpsField                                         # noqa: E402
from dynameta.geometry import Design, Inclusion, Layer, Stack, UnitCell              # noqa: E402
from dynameta.geometry.cross_section import Rectangle                                # noqa: E402
from dynameta.geometry.specs import OpticalSpec                                      # noqa: E402
from dynameta.materials import ConstantOptical, Material, MaterialRegistry           # noqa: E402
from dynameta.optics.fem_2d_tm import GriddedEps2D, TMBlock, TMLayer, solve_fem_2d_tm  # noqa: E402
from dynameta.optics.lumenairy_bridge import design_to_pmm_stack, make_lumenairy_pmm_solver  # noqa: E402
from dynameta.optics.tmm_reference import end_media_indices                           # noqa: E402

P, W, LAM = 400e-9, 150e-9, 1200e-9
FAST = dict(degree=10, pmm_kwargs=None)      # consistency checks: no oracle at stake


def _design(pol="x", theta=0.0):
    reg = MaterialRegistry()
    for nm, e in (("air", 1.0), ("glass", 1.45 ** 2), ("ox", 2.1), ("hi", 4.0), ("FILM", 3.0)):
        reg.add(Material(nm, ConstantOptical(complex(e))))
    layers = [Layer("film", 10e-9, "FILM"), Layer("ox", 20e-9, "ox"),
              Layer("grating", 100e-9, "air", inclusions=[Inclusion(Rectangle(P / 2, P / 2, W, P), "hi")])]
    return Design(name="pmmt", unit_cell=UnitCell(period_x_m=P, period_y_m=P),
                  stack=Stack(layers=layers, superstrate_material="air", substrate_material="glass"),
                  electrodes=[], materials=reg, optical=OpticalSpec(polarization=pol, incidence_angle_deg=theta))


def _profile(nz):
    z = np.linspace(0.0, 10e-9, nz)
    exx = 2.6 - 0.9 * np.exp(-(10e-9 - z) / 2e-9) + 0.05j
    ezz = 1.9 + 0.6 * np.exp(-z / 3e-9) + 0.12j
    return z, exx, ezz


def _tensor_field(nz, swap=False, offdiag=0.0):
    z, exx, ezz = _profile(nz)
    if swap:
        exx, ezz = ezz, exx
    v = np.zeros((nz, 1, 1, 3, 3), dtype=np.complex128)
    v[:, 0, 0, 0, 0] = exx
    v[:, 0, 0, 1, 1] = exx
    v[:, 0, 0, 2, 2] = ezz
    v[:, 0, 0, 0, 2] = v[:, 0, 0, 2, 0] = offdiag
    return EpsField(x_axis_u=np.array([0.0, P * 1e9]), y_axis_u=np.array([0.0, P * 1e9]),
                    z_axis_u=z * 1e9, values_zyx=v)


def _fem(nz, theta=0.0, order=6):
    from dynameta.optics.fem_2d_tm import TMMeshSpec
    z, exx, ezz = _profile(nz)
    lays = [TMLayer("film", 10e-9, GriddedEps2D.uniaxial(exx, ezz, z)), TMLayer("ox", 20e-9, 2.1),
            TMLayer("grating", 100e-9, 1.0, (TMBlock(P / 2 - W / 2, P / 2 + W / 2, 4.0, "hi"),))]
    pol = "x" if theta == 0 else "p"
    return solve_fem_2d_tm(P, lays, lambda_m=LAM, theta_deg=theta, n_super=1.0, n_sub=1.45,
                           polarization=pol, order=order, mesh_spec=TMMeshSpec(corner_mode="all"),
                           num_threads=2).r


def _pmm(design, ef, **kw):
    ns, nb = end_media_indices(design, LAM)
    kw.setdefault("pmm_kwargs", {"elements_per_region": 3})
    return make_lumenairy_pmm_solver(degree=kw.pop("degree", 24), **kw)(design, None, {"film": ef}, LAM, ns, nb)


def test_graded_uniaxial_film_under_a_grating_matches_the_fem():
    """Normal incidence ('x') and 30 deg ('p'): PMM bridge (degree 24, 3 elements per region, arithmetic
    slab per native interval) vs the FEM on the same 21-sample profile. MEASURED |dr| at 21 samples
    4.3e-7 / 3.8e-7 (x / p), at 41 9.5e-8 / 1.1e-7, at 11 2.1e-6 / 1.7e-6 (the slab staircase, second order); budget 2e-6 at
    21 samples. Sign-sensitive: the same field with eps_xx and eps_zz exchanged moves r by > 1e-3
    (MEASURED 6.2e-3 / 6.5e-3)."""
    for theta, pol in ((0.0, "x"), (30.0, "p")):
        d = _design(pol, theta)
        r_fem = _fem(21, theta)
        r21 = _pmm(d, _tensor_field(21)).r
        assert abs(r21 - r_fem) < 2e-6, (theta, abs(r21 - r_fem))
        assert abs(_pmm(d, _tensor_field(21, swap=True)).r - r_fem) > 1e-3


def test_laminate_and_merge_options_for_a_diagonal_tensor():
    """z_average='laminate' (zz harmonic per interval) and z_merge_rtol (z-laminate merge) on the
    diagonal tensor field stay within the staircase budget of the FEM (MEASURED at 21 samples:
    laminate 4.2e-7 (bar 2e-6), laminate + merge 3e-2 3.5e-6 (bar 2e-5: the merge replaces slabs within 3 %
    of each other by one laminate, a first-order model change)), the merge reduces the PMM layer count, and the
    partition report records it."""
    d = _design()
    r_fem = _fem(21)
    rl = _pmm(d, _tensor_field(21), z_average="laminate").r
    rm = _pmm(d, _tensor_field(21), z_average="laminate", z_merge_rtol=3e-2).r
    assert abs(rl - r_fem) < 2e-6 and abs(rm - r_fem) < 2e-5
    stack, names = design_to_pmm_stack(d, LAM, eps_by_region={"film": _tensor_field(21)},
                                       z_average="laminate", z_merge_rtol=3e-2)
    rep = stack.dynameta_partition_report["film"]
    assert rep["tensor"] and rep["n_layers"] < rep["n_slabs"] == 20
    assert names.count("film") == rep["n_layers"]


def test_isotropic_tensor_equals_the_scalar_field_and_refusals():
    """An isotropic gridded tensor gives the scalar gridded field's stack (the slab averages agree to
    round-off: 1e-13); an off-diagonal tensor takes the arithmetic staircase only; a laterally
    structured tensor is still refused."""
    d = _design()
    z, exx, _ = _profile(21)
    v = np.zeros((21, 1, 1, 3, 3), dtype=np.complex128)
    for k in range(3):
        v[:, 0, 0, k, k] = exx
    ax = dict(x_axis_u=np.array([0.0, P * 1e9]), y_axis_u=np.array([0.0, P * 1e9]), z_axis_u=z * 1e9)
    rt = _pmm(d, EpsField(values_zyx=v, **ax), **FAST).r
    rs = _pmm(d, EpsField(values_zyx=np.ascontiguousarray(v[..., 0, 0]), **ax), **FAST).r
    assert abs(rt - rs) < 1e-13
    ef_off = _tensor_field(11, offdiag=0.05)
    assert np.isfinite(_pmm(d, ef_off, **FAST).r)
    with pytest.raises(ValueError, match="arithmetic z staircase only"):
        _pmm(d, ef_off, z_average="laminate", **FAST)
    vs = np.repeat(_tensor_field(11).values_zyx, 3, axis=2)
    vs[:, :, 1, 0, 0] *= 1.1
    ef_s = EpsField(x_axis_u=np.array([0.0, 130.0, 260.0]), y_axis_u=np.array([0.0, P * 1e9]),
                    z_axis_u=_profile(11)[0] * 1e9, values_zyx=vs)
    with pytest.raises(ValueError, match="laterally structured gridded TENSOR"):
        _pmm(d, ef_s, **FAST)


def _base_module():
    """The pmm_backend of the v0.15.1 release commit (b8a73c0 on main), loaded side by side (skip when git or
    the history is unavailable)."""
    try:
        src = subprocess.run(["git", "show", "b8a73c0:dynameta/optics/lumenairy_bridge/pmm_backend.py"],
                             capture_output=True, text=True, check=True, encoding="utf-8").stdout
    except Exception:                                                    # pragma: no cover
        pytest.skip("git history of the base commit b8a73c0 unavailable")
    mod = types.ModuleType("_pmm_backend_base")
    mod.__dict__["__name__"] = "_pmm_backend_base"
    exec(compile(src, "pmm_backend_base.py", "exec"), mod.__dict__)
    return mod


def test_previously_accepted_inputs_are_bit_identical_to_the_base_bridge():
    """Same-tree A/B: uniform scalar, uniform tensor and z-graded scalar fields give np.array_equal r, R,
    T through this tree's bridge and the v0.15.1 one."""
    assert importlib.util.find_spec("lumenairy") is not None
    old = _base_module()
    d = _design()
    z, exx, _ = _profile(21)
    ax = dict(x_axis_u=np.array([0.0, P * 1e9]), y_axis_u=np.array([0.0, P * 1e9]), z_axis_u=z * 1e9)
    cases = [EpsField(scalar=2.5 + 0.1j), EpsField(tensor=np.diag([2.5 + 0.1j, 2.5 + 0.1j, 1.8 + 0.3j])),
             EpsField(values_zyx=exx.reshape(-1, 1, 1).astype(np.complex128), **ax)]
    ns, nb = end_media_indices(d, LAM)
    for ef in cases:
        a = make_lumenairy_pmm_solver(degree=10)(d, None, {"film": ef}, LAM, ns, nb)
        b = old.make_lumenairy_pmm_solver(degree=10)(d, None, {"film": ef}, LAM, ns, nb)
        assert np.array_equal([a.r, a.R, a.T], [b.r, b.R, b.T]), ef
    assert "pmm_backend_base" not in sys.modules
