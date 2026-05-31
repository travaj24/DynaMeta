"""
Build an NGSolve eps CoefficientFunction from a Stage 2 Zarr at one
(bias_label, lambda). Each semiconductor layer's eps(x, z) is loaded as
a complex VoxelCoefficient (optionally symmetrized to (x, y, z)).
Non-semiconductor materials use their Material.eps_at_lambda(lambda_m).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import zarr
import ngsolve as ng

from dynameta.design import Design, OpticalSpec
from dynameta.stage2_drude.symmetrize import symmetrize_n_xz_to_xyz
from dynameta.stage3_optical.ngsolve_build import (
    Cell3DGeometry, UNIT_SCALE,
)


def ngsolve_complex_scalar(z: complex) -> ng.CoefficientFunction:
    return ng.CoefficientFunction(complex(z))


def build_eps_cf_at_bias_lambda(geo: Cell3DGeometry,
                                  design: Design,
                                  stage2_zarr: Path,
                                  lambda_nm: float,
                                  optical: OpticalSpec
                                  ) -> ng.CoefficientFunction:
    """Per-region eps CF for a single (bias, wavelength) point.

    Args:
      geo            : 3D NGSolve cell built by ngsolve_build.build_unit_cell_3d
      design         : the Design instance (gives material name lookup)
      stage2_zarr    : path to a Stage 2 eps_<bias_label>.zarr
      lambda_nm      : wavelength to look up in the Zarr
      optical        : controls use_symmetrization, ny_sym, etc.
    """
    root = zarr.open_group(str(stage2_zarr), mode="r")
    lams_avail = sorted(float(k) for k in root["lambdas"][next(iter(
        root.group_keys())) ].group_keys()) if False else None
    # Build per-layer eps_ito_cf for each semiconductor layer
    sc_eps_cfs: Dict[str, ng.CoefficientFunction] = {}
    for L in design.semiconductor_layers():
        if L.name not in root.group_keys():
            continue
        gp = root[L.name]
        lams = sorted(float(k) for k in gp["lambdas"].group_keys())
        lam_key = "{:.0f}".format(min(lams, key=lambda v: abs(v - lambda_nm)))
        g_lam = gp["lambdas"][lam_key]
        x_axis_m = np.asarray(gp["x_axis_m"][:])
        y_axis_m = np.asarray(gp["y_axis_m"][:])
        n_grid   = np.asarray(gp["n_m3"][:])

        eps_re = np.asarray(g_lam["eps_re"][:])
        eps_im = -np.asarray(g_lam["eps_im"][:])    # paper +iwt -> NGSolve -iwt

        if optical.use_symmetrization:
            cf = _symmetric_voxel_cf(eps_re, eps_im, n_grid,
                                       x_axis_m, y_axis_m, geo, L,
                                       optical.ny_sym, design)
        else:
            cf = _extruded_voxel_cf(eps_re, eps_im, x_axis_m, y_axis_m,
                                       geo, L)
        sc_eps_cfs[L.name] = cf

    # Material lookup for each region in the NGSolve mesh
    region_eps: Dict[str, ng.CoefficientFunction] = {}
    for region_name in geo.mesh.GetMaterials():
        layer_name = geo.layer_name_by_region.get(region_name, region_name)
        # Semiconductor: use spatial eps CF
        if layer_name in sc_eps_cfs:
            region_eps[region_name] = sc_eps_cfs[layer_name]
            continue
        # Non-semiconductor: scalar eps from material lookup
        mat_name = geo.material_by_region.get(region_name, "air")
        # Try the design's MaterialRegistry first
        if mat_name in design.materials:
            eps_at_lambda = design.materials.get(mat_name).optical_eps(
                lambda_nm * 1e-9)
        else:
            # Hardcoded fallbacks for the standard regions (air, Si)
            eps_at_lambda = _fallback_eps(mat_name, lambda_nm * 1e-9)
        region_eps[region_name] = ngsolve_complex_scalar(eps_at_lambda)

    materials = list(geo.mesh.GetMaterials())
    return ng.CoefficientFunction([region_eps[m] for m in materials])


def _symmetric_voxel_cf(eps_re_2d: np.ndarray, eps_im_2d: np.ndarray,
                          n_grid: np.ndarray,
                          x_axis_m: np.ndarray, y_axis_m_devsim: np.ndarray,
                          geo: Cell3DGeometry, layer,
                          ny_out: int, design: Design
                          ) -> ng.CoefficientFunction:
    """Re-symmetrize eps in (x, y) space via the same xy-product formula
    as the carrier-density symmetrization. We symmetrize the eps DEVIATION
    from background; the background eps is what eps would be if n = n_bg
    everywhere.
    """
    n_bg = design.materials.get(layer.material).drude.n_bg_m3
    # Background eps from drude at n_bg at this lambda -- use the eps grid
    # at (x=0, z) as a proxy for background (n is at n_bg there)
    # Better: directly compute by symmetrizing n then re-applying drude --
    # but eps grid is already computed; symmetrize eps deviation directly.
    # n_3d, _ = symmetrize_n_xz_to_xyz(n_grid, x_axis_m, n_bg_m3=n_bg)
    eps_complex_2d = eps_re_2d + 1j * eps_im_2d
    eps_bg = eps_complex_2d[0, :]  # eps at x=0 (outside patch), shape (Nz,)
    # Deviation
    deps = eps_complex_2d - eps_bg[None, :]    # (Nx, Nz)
    # Symmetrize deviation via the xy-product formula
    y_axis_m = np.linspace(0.0, design.period_m, ny_out)
    deps_y = np.zeros((y_axis_m.size, deps.shape[1]), dtype=np.complex128)
    for k in range(deps.shape[1]):
        deps_y[:, k].real = np.interp(y_axis_m, x_axis_m, deps[:, k].real)
        deps_y[:, k].imag = np.interp(y_axis_m, x_axis_m, deps[:, k].imag)
    # Peak per z
    deps_peak = np.empty(deps.shape[1], dtype=np.complex128)
    for k in range(deps.shape[1]):
        idx = int(np.argmax(np.abs(deps[:, k])))
        deps_peak[k] = deps[idx, k] if np.abs(deps[idx, k]) > 0 else 1.0
    deps_3d = np.einsum("ik,jk,k->ijk", deps, deps_y, 1.0 / deps_peak)
    eps_3d = eps_bg[None, None, :] + deps_3d    # (Nx, Ny, Nz)

    return _voxel_cf_from_xyz(eps_3d, x_axis_m, y_axis_m, y_axis_m_devsim,
                                 geo, layer)


def _extruded_voxel_cf(eps_re_2d: np.ndarray, eps_im_2d: np.ndarray,
                         x_axis_m: np.ndarray, y_axis_m_devsim: np.ndarray,
                         geo: Cell3DGeometry, layer
                         ) -> ng.CoefficientFunction:
    """Build a (Ny=2 degenerate) 3D VoxelCoefficient from eps(x, z)."""
    eps_3d = np.repeat((eps_re_2d + 1j * eps_im_2d)[:, None, :], 2,
                         axis=1).astype(np.complex128)
    y_axis_m = np.array([0.0, geo.period_nm / UNIT_SCALE], dtype=np.float64)
    return _voxel_cf_from_xyz(eps_3d, x_axis_m, y_axis_m, y_axis_m_devsim,
                                 geo, layer)


def _voxel_cf_from_xyz(eps_3d: np.ndarray,
                         x_axis_m: np.ndarray, y_axis_m: np.ndarray,
                         y_axis_m_devsim: np.ndarray,
                         geo: Cell3DGeometry, layer
                         ) -> ng.CoefficientFunction:
    """Construct an NGSolve VoxelCoefficient over (x_nm, y_nm, z_nm).

    eps_3d shape: (Nx, Ny, Nz) in (x, y, z_devsim) order.
    NGSolve VoxelCoefficient axis order: (Nz, Ny, Nx).
    """
    S = UNIT_SCALE
    x_fem_nm = x_axis_m * S
    y_fem_nm = y_axis_m * S
    # Find the FEM z interval for this layer (using the in-patch region's z)
    z_intv = None
    for rname, zi in geo.z_intervals.items():
        if geo.layer_name_by_region.get(rname) == layer.name:
            z_intv = zi
            break
    if z_intv is None:
        raise RuntimeError("No 3D region found for layer '{}'".format(layer.name))
    z_lo_nm, z_hi_nm = z_intv
    fem_thk_nm = z_hi_nm - z_lo_nm
    s1_thk_m   = float(y_axis_m_devsim[-1] - y_axis_m_devsim[0])
    z_fem_nm   = z_lo_nm + (y_axis_m_devsim - y_axis_m_devsim[0]) \
                   * (fem_thk_nm / s1_thk_m)
    # Transpose (Nx, Ny, Nz) -> (Nz, Ny, Nx)
    eps_zyx = np.transpose(eps_3d, (2, 1, 0)).astype(np.complex128)
    start = (float(x_fem_nm[0]),  float(y_fem_nm[0]),  float(z_fem_nm[0]))
    end   = (float(x_fem_nm[-1]), float(y_fem_nm[-1]), float(z_fem_nm[-1]))
    return ng.VoxelCoefficient(start=start, end=end,
                                  values=eps_zyx, linear=True)


def _fallback_eps(name: str, lambda_m: float) -> complex:
    """Constant fallback eps for materials not registered in the Design.
    Used only for air and Si (PML / substrate)."""
    if name == "air":
        return complex(1.0, 0.0)
    if name == "Si":
        return complex(12.0, 0.0)
    return complex(1.0, 0.0)
