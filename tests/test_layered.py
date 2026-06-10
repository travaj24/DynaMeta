"""Fast (numpy + tmm, no FEM) tests for the layered-stack RCWA-prep spine: the LayeredSlab/
LayeredStack data model, the z-slicer (slice_profile), and the graded-TMM consumer
(layered_rta) + TmmLayeredSolver. These exercise the exact representation + slicing the future
RCWA backend will reuse. Run: python -m pytest tests/test_layered.py -q
"""
import numpy as np
import pytest

from dynameta.core.layered import (LayeredSlab, LayeredStack, slice_profile, slice_eps_field)
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


def test_slice_eps_field_uniform_and_structured():
    # exercise BOTH slice_eps_field branches (the structured/eps_cell path was dead, LTM-3).
    class _EF:                                             # minimal gridded-EpsField stand-in
        def __init__(self, v_zyx, z_u, uniform):
            self.values_zyx = np.asarray(v_zyx, dtype=complex)   # (Nz, Ny, Nx)
            self.z_axis_u = np.asarray(z_u, dtype=float)
            self.is_uniform = uniform
    # laterally-uniform field -> SCALAR slabs (xy-mean, midpoint over z)
    vu = np.zeros((3, 2, 2), complex); vu[0] = 4.0; vu[1] = 6.0; vu[2] = 9.0
    su = slice_eps_field(_EF(vu, [0.0, 50.0, 100.0], False), 1e-9)
    assert len(su) == 2 and all(s.is_uniform for s in su)
    assert np.isclose(su[0].eps, 5.0) and np.isclose(su[1].eps, 7.5)   # midpoints of 4/6, 6/9
    assert np.isclose(su[0].thickness_m, 50e-9)
    # laterally-structured field -> eps_cell slab with the documented (Ny,Nx)->(Nx,Ny) transpose
    vs = np.zeros((2, 2, 3), complex)                      # (Nz=2, Ny=2, Nx=3)
    vs[0] = np.array([[1, 2, 3], [4, 5, 6]]); vs[1] = vs[0] + 1.0
    ss = slice_eps_field(_EF(vs, [0.0, 40.0], False), 1e-9)
    assert len(ss) == 1 and ss[0].eps_cell is not None and not ss[0].is_uniform
    assert ss[0].eps_cell.shape == (3, 2)                  # (Ny,Nx)=(2,3) -> (Nx,Ny)=(3,2)
    stk = LayeredStack(1.0 + 0j, 1.0 + 0j, ss)
    assert stk.is_unstructured is False                    # a structured slab -> TMM must refuse
    with pytest.raises(ValueError):
        layered_rta(stk, 1300e-9)


def test_layered_order_sensitivity_on_asymmetric_lossy_stack():
    # The graded_tmm_vs_fem validation uses a symmetric LOSSLESS profile, for which R is
    # order-INsensitive, so it cannot catch a slab-order / double-reversal regression (LTM-1).
    # Pin the property directly: an ASYMMETRIC LOSSY stack gives a DIFFERENT R when reversed,
    # so layered_rta (hence layered_stack_from_design's ordering) is genuinely order-sensitive.
    slabs = [LayeredSlab(40e-9, eps=complex(n) ** 2) for n in (2.0 + 0.05j, 2.5 + 0.2j, 3.0 + 0.4j)]
    R_f, _, _ = layered_rta(LayeredStack(1.0 + 0j, 1.0 + 0j, slabs), 1300e-9)
    R_r, _, _ = layered_rta(LayeredStack(1.0 + 0j, 1.0 + 0j, slabs[::-1]), 1300e-9)
    assert abs(R_f - R_r) > 0.02                           # order matters -> a reversal is detectable


def test_layered_rta_rejects_lossy_superstrate():
    # R/T/A and the budget A=1-R-T are defined only for a lossless incidence medium (LTM-5,
    # mirroring the FEM OPT-1 guard).
    stk = LayeredStack(1.0 + 0.3j, 1.0 + 0j, [LayeredSlab(100e-9, eps=4.0 + 0j)])
    with pytest.raises(ValueError):
        layered_rta(stk, 1300e-9)


def test_uniform_eps_by_region_entry_reaches_tmm():
    # REGRESSION (drivers/examples seam): a UNIFORM eps_by_region entry (an effect-modulated
    # eps from PCM/LC/thermo via EffectEpsMap, or a uniform carrier region) must override the
    # raw material eps in the TMM stack -- it used to fall through silently. A tensor entry
    # must raise (scalar TMM), not silently drop to the material value.
    from dynameta.core.eps_field import EpsField
    from dynameta.materials import Material, MaterialRegistry, ConstantOptical
    from dynameta.geometry import UnitCell, Stack, Layer, Design
    from dynameta.geometry.specs import OpticalSpec
    reg = MaterialRegistry()
    reg.add(Material("air", ConstantOptical(1.0 + 0j)))
    reg.add(Material("pcm", ConstantOptical(complex(4.0, 0.0))))
    d = Design(name="u", unit_cell=UnitCell.square(300e-9),
               stack=Stack(layers=[Layer("film", 180e-9, "pcm")],
                           superstrate_material="air", substrate_material="air"),
               electrodes=[], materials=reg,
               optical=OpticalSpec(polarization="y", incidence_angle_deg=0.0))
    lam = 1300e-9
    eps_mod = complex(9.0, 0.5)                            # crystallized + lossy: != material 4.0
    stk = layered_stack_from_design(d, lam, eps_by_region={"film": EpsField(scalar=eps_mod)})
    assert stk.slabs[0].eps == eps_mod                     # the modulated value, not 4.0
    stk0 = layered_stack_from_design(d, lam)               # no entry -> material eps unchanged
    assert stk0.slabs[0].eps == complex(4.0, 0.0)
    with pytest.raises(ValueError):                        # anisotropic effect -> loud, not silent
        layered_stack_from_design(d, lam, eps_by_region={
            "film": EpsField(tensor=np.diag([4.0, 4.0, 9.0]).astype(complex))})
