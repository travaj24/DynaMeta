"""Fast (numpy + tmm, no FEM) tests for the layered-stack RCWA-prep spine: the LayeredSlab/
LayeredStack data model, the z-slicer (slice_profile), and the graded-TMM consumer
(layered_rta) + TmmLayeredSolver. These exercise the exact representation + slicing the future
RCWA backend will reuse. Run: python -m pytest tests/test_layered.py -q
"""
import numpy as np
import pytest

from dynameta.core.layered import LayeredSlab, LayeredStack, slice_profile
from dynameta.optics.tmm_reference import (stack_rta, layered_rta, TmmLayeredSolver,
                                           make_layered_tmm_solver, layered_stack_from_design)


def test_layeredslab_requires_exactly_one_spec():
    LayeredSlab(100e-9, eps=4.0)                       # ok
    with pytest.raises(ValueError):
        LayeredSlab(100e-9)                            # zero specs
    with pytest.raises(ValueError):
        LayeredSlab(100e-9, eps=4.0, eps_cell=np.ones((4, 4)))   # two specs
    with pytest.raises(ValueError):
        LayeredSlab(0.0, eps=4.0)                      # non-positive thickness


def test_slice_profile_native_and_resampled():
    z = np.linspace(0.0, 100e-9, 6)                    # 5 intervals
    eps = np.linspace(4.0, 9.0, 6).astype(complex)     # linear profile
    slabs = slice_profile(eps, z)                      # native: one slab per interval
    assert len(slabs) == 5
    assert abs(sum(s.thickness_m for s in slabs) - 100e-9) < 1e-18
    assert np.isclose(slabs[0].eps, 0.5 * (4.0 + 5.0))  # midpoint (trapezoid) eps
    res = slice_profile(eps, z, n_slices=10)           # resample to 10 uniform slabs
    assert len(res) == 10
    assert all(abs(s.thickness_m - 10e-9) < 1e-18 for s in res)
    # descending z gives the same slabs in the same physical order
    assert len(slice_profile(eps[::-1], z[::-1])) == 5


def test_layered_rta_matches_stack_rta_uniform():
    # a single uniform slab as a LayeredStack must reproduce the unstructured stack_rta
    stk = LayeredStack(1.0 + 0j, 1.5 + 0j, [LayeredSlab(250e-9, eps=complex(2.0 ** 2))])
    R, T, A = layered_rta(stk, 1300e-9, theta_deg=20.0, pol="p")
    R2, T2, A2 = stack_rta(1.0, [(2.0, 250e-9)], 1.5, 1300e-9, theta_deg=20.0, pol="p")
    assert abs(R - R2) < 1e-12 and abs(T - T2) < 1e-12


def test_slicing_a_uniform_profile_is_exact():
    # slicing a CONSTANT eps(z) into many slabs must equal the single-slab result
    z = np.linspace(0.0, 250e-9, 41)
    slabs = slice_profile(np.full(41, complex(2.0 ** 2)), z, n_slices=40)
    stk = LayeredStack(1.0 + 0j, 1.0 + 0j, slabs)
    R, T, A = layered_rta(stk, 1300e-9)
    R0, T0, _ = stack_rta(1.0, [(2.0, 250e-9)], 1.0, 1300e-9)
    assert abs(R - R0) < 1e-9 and abs(T - T0) < 1e-9


def test_graded_lossless_conserves_energy_and_converges():
    # a lossless graded slab (n: 1.5 -> 2.5 -> 1.5) sliced finely: R+T ~ 1 and R converges
    z = np.linspace(0.0, 400e-9, 201)
    n_of_z = 2.0 + 0.5 * np.cos(2 * np.pi * (z - z.mean()) / 400e-9)   # smooth, real
    eps = (n_of_z ** 2).astype(complex)
    Rs = []
    for nsl in (50, 200):
        stk = LayeredStack(1.0 + 0j, 1.0 + 0j, slice_profile(eps, z, n_slices=nsl))
        R, T, A = layered_rta(stk, 1300e-9)
        assert abs(R + T - 1.0) < 1e-9                 # lossless -> energy conserved
        Rs.append(R)
    assert abs(Rs[0] - Rs[1]) < 5e-3                   # slab-count converged


def test_tmm_layered_solver_optical_result():
    class _Opt:
        polarization = "y"
        incidence_angle_deg = 0.0
    stk = LayeredStack(1.0 + 0j, 1.0 + 0j, [LayeredSlab(250e-9, eps=complex(2.0 ** 2))])
    res = TmmLayeredSolver().solve(stk, 1300e-9, _Opt())
    assert 0.0 <= res.R <= 1.0 and res.T is not None
    assert abs(res.R + res.T + res.A - 1.0) < 1e-9     # lossless
    assert abs(abs(res.r) ** 2 - res.R) < 1e-9         # |r|^2 == R
    assert -180.0 <= res.phase_deg <= 180.0


def test_make_layered_tmm_solver_seam():
    # make_layered_tmm_solver() must return an optical_solver with the EXACT run_pipeline
    # seam signature fn(design, geo, eps_by_region, lam_m, n_super, n_sub) and reproduce
    # layered_stack_from_design + TmmLayeredSolver on a uniform stack. This is the adapter
    # that lets the layered/TMM (and future RCWA) backend drop into run_pipeline (LTM-4/BLP-1).
    from dynameta.materials import Material, MaterialRegistry, ConstantOptical
    from dynameta.geometry import UnitCell, Stack, Layer, Design
    from dynameta.geometry.specs import OpticalSpec
    reg = MaterialRegistry()
    reg.add(Material("air", ConstantOptical(1.0 + 0j)))
    reg.add(Material("hi", ConstantOptical(complex(2.2 ** 2, 0.0))))
    d = Design(name="u", unit_cell=UnitCell.square(300e-9),
               stack=Stack(layers=[Layer("film", 180e-9, "hi")],
                           superstrate_material="air", substrate_material="air"),
               electrodes=[], materials=reg,
               optical=OpticalSpec(polarization="y", incidence_angle_deg=0.0))
    lam = 1300e-9
    solve = make_layered_tmm_solver()
    res = solve(d, None, {}, lam, 1.0 + 0j, 1.0 + 0j)   # geo/n_super/n_sub unused by TMM
    R, T, A = layered_rta(layered_stack_from_design(d, lam), lam, theta_deg=0.0, pol="s")
    assert abs(res.R - R) < 1e-12 and abs(res.T - T) < 1e-12
    assert abs(res.R + res.T + res.A - 1.0) < 1e-9
    assert abs(abs(res.r) ** 2 - res.R) < 1e-9
