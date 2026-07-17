"""Audit S2-1 gate: the GRADED (gridded) tensor eps assembly path through solve_fem.

Every prior tensor-FEM oracle used a UNIFORM tensor (EpsField(tensor=...)), leaving the graded
VoxelCoefficient tensor branch of eps_assembler (+ the tensor UPML explicit-component sum in the
solver) with zero end-to-end coverage -- even though core/bridge emits gridded (Nz,Ny,Nx,3,3)
tensors onto exactly that path. This gate feeds a CONSTANT-over-grid graded tensor for a tilted
uniaxial (LC-like) slab and pins it against the uniform-tensor solve of the same physics, which
is itself validated against TMM/Berreman elsewhere. ngsolve-gated."""

import numpy as np
import pytest

pytest.importorskip("ngsolve")

from dynameta.materials import Material, MaterialRegistry, ConstantOptical      # noqa: E402
from dynameta.geometry import UnitCell, Stack, Layer, Design                    # noqa: E402
from dynameta.geometry.specs import OpticalSpec, Mesh3DSpec                     # noqa: E402
from dynameta.core.eps_field import EpsField                                    # noqa: E402
from dynameta.optics.ngsolve_layered import LayeredOpticalBuilder               # noqa: E402
from dynameta.optics.eps_assembler import assemble_eps_cf                       # noqa: E402
from dynameta.optics.solver import solve_fem                                    # noqa: E402

LAM_NM = 1550.0
NO, NE = 1.53, 1.71
L_NM = 500.0


def _design():
    reg = MaterialRegistry()
    reg.add(Material("air", ConstantOptical(1.0 + 0j)))
    reg.add(Material("lc", ConstantOptical(complex(NO ** 2, 0.0))))
    cell = UnitCell.square(300e-9)
    stack = Stack(layers=[Layer("s", L_NM * 1e-9, "lc")],
                  superstrate_material="air", substrate_material="air")
    m3 = Mesh3DSpec(pml_thk_m=600e-9, superstrate_buffer_m=900e-9, substrate_buffer_m=900e-9,
                    maxh_superstrate_m=60e-9, maxh_substrate_m=60e-9, maxh_background_m=30e-9)
    return Design(name="graded_tensor_gate", unit_cell=cell, stack=stack, electrodes=[],
                  materials=reg, mesh_3d=m3)


def _tilt_tensor(theta_deg):
    th = np.radians(theta_deg)
    d = np.array([np.sin(th), 0.0, np.cos(th)])
    eps = NO ** 2 * np.eye(3) + (NE ** 2 - NO ** 2) * np.outer(d, d)
    return eps.astype(complex)


def _solve(eps_field, pol):
    geo = LayeredOpticalBuilder(_design()).build()
    mats = list(geo.mesh.GetMaterials())
    slab = [r for r in mats if geo.material_by_region[r] == "lc"][0]
    ebr = {rg: EpsField(scalar=complex(1.0, 0.0)) for rg in mats}
    ebr[slab] = eps_field
    opt = OpticalSpec(polarization=pol, incidence_angle_deg=0.0, linear_solver="umfpack")
    return solve_fem(geo, LAM_NM * 1e-9, assemble_eps_cf(geo, ebr), opt, order=2,
                     n_super=1.0 + 0j, n_sub=1.0 + 0j)


def test_graded_tensor_matches_uniform_tensor():
    # constant-over-grid graded tensor must reproduce the uniform-tensor solve (which is the
    # TMM/Berreman-validated reference) for a 45-deg-tilted uniaxial slab, both polarizations
    eps = _tilt_tensor(45.0)
    nz, ny, nx = 4, 3, 3
    vals = np.broadcast_to(eps, (nz, ny, nx, 3, 3)).copy()
    zax = np.linspace(0.0, L_NM, nz)
    yax = np.linspace(0.0, 300.0, ny)
    xax = np.linspace(0.0, 300.0, nx)
    graded = EpsField(x_axis_u=xax, y_axis_u=yax, z_axis_u=zax, values_zyx=vals)
    uniform = EpsField(tensor=eps)
    for pol in ("x", "y"):
        rg = _solve(graded, pol)
        ru = _solve(uniform, pol)
        assert abs(rg.R - ru.R) < 5e-3, (pol, rg.R, ru.R)
        if rg.T is not None and ru.T is not None:
            assert abs(rg.T - ru.T) < 5e-3, (pol, rg.T, ru.T)
