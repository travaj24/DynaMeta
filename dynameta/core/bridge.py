"""
The bridge: turn a CarrierField + GeometryAlignment into a per-optical-region
EpsField, given an NToEpsMap (n->eps) and a FieldLift (2D->3D reconstruction).

This is the reusable spine. It depends on NEITHER devsim nor ngsolve nor the
Design -- only on the alignment contract + numpy. It reproduces the affine
coordinate placement that used to be buried in eps_loader._voxel_cf_from_xyz
(lines 145-176), but driven explicitly by RegionAlignment:
  - lateral (x, y): the carrier field's SI axes scaled to mesh units (the 2D
    solve spans the cell laterally; the lift synthesizes the 2nd lateral axis)
  - vertical (z):   the source through-stack axis affine-remapped onto the
    region's z-interval (DEVSIM layer thickness need NOT equal the FEM band)
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from dynameta.core.alignment import GeometryAlignment
from dynameta.core.carrier_field import CarrierField, ELECTRON_DENSITY
from dynameta.core.eps_field import EpsField
from dynameta.core.lift import FieldLift
from dynameta.core.n_to_eps import NToEpsMap

# The sign convention the whole library + NGSolve assume (passive loss => Im(eps)>0).
from dynameta.constants import SOLVER_TIME_CONVENTION  # noqa: E402  (single source; re-exported here)


def assemble_eps(field: CarrierField,
                   alignment: GeometryAlignment,
                   n_to_eps: NToEpsMap,
                   lift: FieldLift,
                   lambda_m: float,
                   *,
                   mesh_regions: Optional[List[str]] = None,
                   density_field: str = ELECTRON_DENSITY,
                   extra_fields: Optional[Dict[str, object]] = None) -> Dict[str, EpsField]:
    """Return {mesh_region: EpsField}. Spatial (carrier-derived) EpsFields for
    the aligned semiconductor regions; uniform scalar EpsFields for the rest.

    `extra_fields` (C7): a {field_name: value} bundle merged into the per-region fields dict alongside
    the carrier density 'n', so a field-effect EffectModel (Pockels reading 'E', ThermoOptic reading
    'T', LiquidCrystal reading 'director_angle_rad', ...) is driven through the SAME bridge instead of
    a manual per-region call -- pair it with an n_to_eps that dispatches to those models (EffectEpsMap).
    Values may be grids (broadcast against 'n' per point -> a gridded EpsField) or uniform scalars/
    vectors (a uniform field-effect eps -> a uniform EpsField). None (the default) leaves the carrier-
    only path byte-identical."""
    # The whole library (Drude Im(eps) sign + NGSolve PML) is exp(-iwt). A field
    # carrying a different convention would be fed to the solver with the wrong
    # Im(eps) sign -- fail loudly instead of silently (audit F2/F7).
    if field.time_convention != SOLVER_TIME_CONVENTION:
        raise ValueError(
            "CarrierField.time_convention {!r} != the library/solver convention {!r}; "
            "the Drude Im(eps) sign and the NGSolve PML both assume exp(-iwt). Convert "
            "the field (conjugate eps) before assembling.".format(
                field.time_convention, SOLVER_TIME_CONVENTION))

    if mesh_regions is not None:
        alignment.validate_coverage(mesh_regions)
    else:
        # Even without the mesh-region list we still enforce the internal half of
        # the exactly-once guarantee: no region may be BOTH spatial and fixed
        # (the full mesh-coverage check needs mesh_regions; see assemble_eps_cf).
        _spatial = {ra.mesh_region for ra in alignment.region_alignments}
        _dup = _spatial & set(alignment.fixed_eps_regions)
        if _dup:
            raise ValueError("Regions mapped both spatial and fixed: {}".format(sorted(_dup)))

    mpp = alignment.unit_scale.metres_per_unit
    out: Dict[str, EpsField] = {}

    for ra in alignment.region_alignments:
        if ra.source_region not in field.regions:
            raise ValueError("alignment names source region '{}' absent from the "
                              "carrier field".format(ra.source_region))
        reg = field.regions[ra.source_region]
        if reg.grid_fields is None or reg.grid_axes_m is None:
            raise ValueError("source region '{}' has no resampled grid".format(
                ra.source_region))
        if density_field not in reg.grid_fields:
            raise ValueError("source region '{}' grid missing '{}' (have {})".format(
                ra.source_region, density_field, sorted(reg.grid_fields)))

        n_grid = np.asarray(reg.grid_fields[density_field], dtype=np.float64)
        n_bg = float(field.n_bg_by_region[ra.source_region])

        if n_grid.ndim == 3:
            # Native 3D carrier field (e.g. carriers/devsim_3d): real x/y/z axes,
            # NO lift synthesis. z is the through-stack axis BY THE 3D-FIELD
            # CONVENTION; RegionAlignment.stack_axis applies to 2D fields only
            # (the auto-built alignment may carry its 2D default here -- ignored).
            for _k in ("x", "y", "z"):
                if _k not in reg.grid_axes_m:
                    raise ValueError(
                        "3D source region '{}' grid missing axis '{}' (have {}); the 3D "
                        "bridge requires x/y/z axes with z through-stack".format(
                            ra.source_region, _k, sorted(reg.grid_axes_m)))
            x3_m = np.asarray(reg.grid_axes_m["x"], dtype=np.float64)
            y3_m = np.asarray(reg.grid_axes_m["y"], dtype=np.float64)
            z3_m = np.asarray(reg.grid_axes_m["z"], dtype=np.float64)
            eps_3d = n_to_eps.eps_grid(reg.material, {"n": n_grid, **(extra_fields or {})},
                                       lambda_m)                                # (Nx,Ny,Nz)
        else:
            # 2D (x, v=through-stack) carrier solve -> the FieldLift synthesizes
            # the 2nd lateral axis (SeparableXY / Extrude).
            x_m = np.asarray(reg.grid_axes_m["x"], dtype=np.float64)
            v_m = np.asarray(reg.grid_axes_m[ra.stack_axis], dtype=np.float64)
            n_3d, x3_m, y3_m, z3_m = lift.apply(n_grid, x_m, v_m, n_bg=n_bg)
            # audit C5-5: extra_fields used to merge RAW against the LIFTED n -- a 2D
            # (Nx,Nv) companion grid (thermal/electrostatic driver output on the carrier
            # grid) broadcast its x-axis onto the optical y-axis whenever Nx == Ny (true
            # at the SHIPPED defaults, grid_n_x=256 == ny_sym=256): an exact x<->y
            # TRANSPOSE of the field, silently -- and a cryptic broadcast error
            # otherwise. Matching 2D extras are now extruded y-uniform onto the lifted
            # grid (exact ExtrudeLift semantics; for SeparableXY a 2D companion carries
            # no y-structure information anyway); other multi-D shapes raise.
            extras_lifted = {}
            for _k, _v in (extra_fields or {}).items():
                _arr = np.asarray(_v)
                if _arr.ndim == 2 and _arr.shape == np.shape(n_grid):
                    extras_lifted[_k] = np.broadcast_to(
                        _arr[:, None, :], np.shape(n_3d)).copy()
                elif _arr.ndim >= 2:
                    raise ValueError(
                        "assemble_eps: extra field '{}' has shape {} which matches "
                        "neither the 2D carrier grid {} nor a scalar/(Nz,) profile -- "
                        "a raw merge would silently misplace it (audit C5-5)".format(
                            _k, _arr.shape, np.shape(n_grid)))
                else:
                    extras_lifted[_k] = _v
            eps_3d = n_to_eps.eps_grid(reg.material, {"n": n_3d, **extras_lifted},
                                       lambda_m)                               # (Nx,Ny,Nz)

        # A field-effect EffectModel driven by a UNIFORM extra field (e.g. a uniform gate E for
        # Pockels, or a uniform T) returns a single eps (scalar or 3x3) rather than a per-point grid
        # -> emit a uniform EpsField for the region (C7).
        eps_3d = np.asarray(eps_3d)
        if eps_3d.ndim == 0:
            out[ra.mesh_region] = EpsField(scalar=complex(eps_3d),
                                           time_convention=field.time_convention)
            continue
        if eps_3d.shape == (3, 3):
            out[ra.mesh_region] = EpsField(tensor=eps_3d.astype(np.complex128),
                                           time_convention=field.time_convention)
            continue
        # Affine placement into the region bbox. Lateral axes scale directly
        # (the carrier solve already spans the cell laterally); the vertical
        # axis remaps onto [zlo, zhi].
        _, _, _, _, zlo, zhi = ra.bbox_m
        zspan = z3_m[-1] - z3_m[0]
        z_remap_m = (zlo + (z3_m - z3_m[0]) * ((zhi - zlo) / zspan)
                      if zspan > 0 else np.full_like(z3_m, zlo))
        # eps_grid OUTPUT contract: a gridded response must be (Nx,Ny,Nz) scalar or (Nx,Ny,Nz,3,3)
        # tensor matching the region axes -- the transpose below assumes it, and a wrong-shaped /
        # transposed grid would silently misplace eps in the geometry (audit-v2 finding).
        _want = (x3_m.size, y3_m.size, z3_m.size)
        if not ((eps_3d.ndim == 3 and eps_3d.shape == _want)
                or (eps_3d.ndim == 5 and eps_3d.shape == _want + (3, 3))):
            raise ValueError(
                "n_to_eps.eps_grid for region '{}' returned shape {}; expected scalar (Nx,Ny,Nz)={}, "
                "tensor (Nx,Ny,Nz,3,3), a uniform scalar, or a uniform (3,3) tensor (axis order is "
                "(Nx,Ny,Nz), NOT (Nz,Ny,Nx)).".format(ra.mesh_region, eps_3d.shape, _want))
        if eps_3d.ndim == 5:                                   # tensor (Nx,Ny,Nz,3,3) -> (Nz,Ny,Nx,3,3)
            vals = np.transpose(eps_3d, (2, 1, 0, 3, 4)).astype(np.complex128)
        else:                                                  # scalar (Nx,Ny,Nz) -> (Nz,Ny,Nx)
            vals = np.transpose(eps_3d, (2, 1, 0)).astype(np.complex128)
        out[ra.mesh_region] = EpsField(
            x_axis_u=x3_m / mpp, y_axis_u=y3_m / mpp, z_axis_u=z_remap_m / mpp,
            values_zyx=vals, time_convention=field.time_convention)

    for region, mat_name in alignment.fixed_eps_regions.items():
        out[region] = EpsField(scalar=n_to_eps.scalar_eps(mat_name, lambda_m),
                                  time_convention=field.time_convention)
    return out
