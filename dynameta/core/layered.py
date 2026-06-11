"""
Layered-stack representation -- the solver-agnostic spine for FOURIER-MODAL / TMM
optical backends (a future RCWA port, the present TMM oracle). RCWA and TMM want a stack
of laterally-periodic SLABS (a per-layer in-plane eps + a thickness), NOT the per-mesh-region
voxel eps the FEM consumes. This module names that representation + the z-slicer that turns a
graded eps(z) (a carrier-accumulation layer, a thermal/field gradient) into uniform slabs --
the piece the RCWA adapter and the graded-TMM oracle both reuse. Pure numpy; no devsim/ngsolve.

Convention: public exp(-i omega t), Im(eps) > 0 for absorbers (the library standard), metres.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass
class LayeredSlab:
    """One layer of a LayeredStack. EXACTLY ONE eps specification (mirroring the eventual
    RCWAStack.add_layer surface so the adapter is a 1:1 map; the TMM path uses only `eps`):
      * `eps` (scalar)            -- laterally uniform;
      * `eps_cell` (Sx, Sy)       -- isotropic patterned (in-plane eps grid);
      * `eps_tensor_cell` (Sx,Sy,3,3) -- anisotropic patterned;
      * `shapes` (+ `eps_background`) -- analytic-shape factorization.
    """
    thickness_m: float
    eps: Optional[complex] = None
    eps_cell: Optional[np.ndarray] = None
    eps_tensor_cell: Optional[np.ndarray] = None
    shapes: Optional[list] = None
    eps_background: Optional[complex] = None

    def __post_init__(self):
        n = sum(x is not None for x in
                (self.eps, self.eps_cell, self.eps_tensor_cell, self.shapes))
        if n != 1:
            raise ValueError("LayeredSlab: provide exactly one of eps, eps_cell, "
                             "eps_tensor_cell, shapes (got {}).".format(n))
        if self.shapes is not None and self.eps_background is None:
            raise ValueError("LayeredSlab: shapes requires eps_background.")
        if self.eps_cell is not None and np.asarray(self.eps_cell).ndim != 2:
            raise ValueError("LayeredSlab.eps_cell must be a 2D (Sx, Sy) in-plane grid; got shape "
                             "{}.".format(np.shape(self.eps_cell)))
        if self.eps_tensor_cell is not None:
            sh = np.shape(self.eps_tensor_cell)
            if len(sh) != 4 or sh[-2:] != (3, 3):
                raise ValueError("LayeredSlab.eps_tensor_cell must be (Sx, Sy, 3, 3); got shape "
                                 "{}.".format(sh))
        if not (float(self.thickness_m) > 0.0):
            raise ValueError("LayeredSlab.thickness_m must be > 0.")

    @property
    def is_uniform(self) -> bool:
        return self.eps is not None


@dataclass
class LayeredStack:
    """A laterally-periodic layered stack: superstrate | slabs (in incidence order, the
    superstrate-side slab FIRST) | substrate. period_x/y = 0 means laterally uniform (TMM)."""
    n_super: complex
    n_sub: complex
    slabs: List[LayeredSlab]
    period_x_m: float = 0.0
    period_y_m: float = 0.0

    @property
    def is_unstructured(self) -> bool:
        """True if every slab is laterally uniform -> exactly solvable by TMM."""
        return all(s.is_uniform for s in self.slabs)

    @property
    def total_thickness_m(self) -> float:
        return float(sum(s.thickness_m for s in self.slabs))


def slice_profile(eps_of_z, z_m, *, n_slices: Optional[int] = None) -> List[LayeredSlab]:
    """Slice a sampled scalar permittivity profile eps(z) into uniform LayeredSlabs.

    This is the z-slicer the RCWA/TMM backends need for a graded layer (the carrier ENZ
    accumulation profile, a thermo-optic/field gradient): a continuous eps(z) becomes a
    staircase of uniform slabs. The slabs are returned in the SAME order as `z_m`, so order
    z from the superstrate side to the substrate side before calling.

    Args:
      eps_of_z : complex permittivity sampled at `z_m`.
      z_m      : monotonic z coordinates (m), same length as eps_of_z.
      n_slices : if None, one slab per native interval [z[k], z[k+1]] with eps at the slab
                 midpoint (the trapezoidal average of the endpoints). If given, resample to a
                 uniform staircase of `n_slices` slabs spanning [z[0], z[-1]] (midpoint-sampled)
                 -- use this to study slab-count convergence.
    """
    eps = np.asarray(eps_of_z, dtype=np.complex128).ravel()
    z = np.asarray(z_m, dtype=np.float64).ravel()
    if z.size != eps.size or z.size < 2:
        raise ValueError("slice_profile: eps_of_z and z_m must be 1D, equal length >= 2.")
    if not (np.all(np.diff(z) > 0) or np.all(np.diff(z) < 0)):
        raise ValueError("slice_profile: z_m must be monotonic.")
    if z[0] > z[-1]:                                  # normalize to ascending, keep slab order
        z, eps = z[::-1], eps[::-1]
    if n_slices is None:
        return [LayeredSlab(float(z[k + 1] - z[k]), eps=complex(0.5 * (eps[k] + eps[k + 1])))
                for k in range(z.size - 1)]
    z0, z1 = float(z[0]), float(z[-1])
    dt = (z1 - z0) / int(n_slices)
    slabs = []
    for k in range(int(n_slices)):
        zc = z0 + (k + 0.5) * dt
        em = complex(np.interp(zc, z, eps.real), np.interp(zc, z, eps.imag))
        slabs.append(LayeredSlab(dt, eps=em))
    return slabs


def slice_eps_field(eps_field, metres_per_unit: float, *, n_slices: Optional[int] = None
                    ) -> List[LayeredSlab]:
    """Slice a gridded EpsField (values_zyx (Nz,Ny,Nx), axes in target units) into LayeredSlabs.
    For a laterally-UNIFORM field each slab is a scalar (the xy-mean of the z-slice); a
    laterally-structured field yields an `eps_cell` per slab. Axis lengths are converted to
    metres via `metres_per_unit` (the EpsField axes are in the solver's units, e.g. nm)."""
    if eps_field.is_uniform:
        raise ValueError("slice_eps_field: EpsField is a uniform scalar (nothing to slice).")
    v = np.asarray(eps_field.values_zyx, dtype=np.complex128)        # (Nz, Ny, Nx) or (Nz, Ny, Nx, 3, 3)
    z_m = np.asarray(eps_field.z_axis_u, dtype=np.float64) * float(metres_per_unit)
    is_tensor = (v.ndim == 5 and v.shape[-2:] == (3, 3))             # a graded 3x3 anisotropic eps field
    if not is_tensor and v.ndim != 3:
        raise ValueError("slice_eps_field: values_zyx must be (Nz,Ny,Nx) scalar or (Nz,Ny,Nx,3,3) tensor; "
                         "got shape {}.".format(v.shape))
    laterally_uniform = bool(np.allclose(v, v.mean(axis=(1, 2), keepdims=True)))
    if is_tensor:
        # one eps_tensor_cell (Ny,Nx,3,3)->(Nx,Ny,3,3) per native z-slice; uniform -> a 1x1x3x3 cell.
        slabs = []
        for k in range(z_m.size - 1):
            avg = 0.5 * (v[k] + v[k + 1])                            # (Ny, Nx, 3, 3)
            if laterally_uniform:
                cell = avg.mean(axis=(0, 1)).reshape(1, 1, 3, 3)
            else:
                cell = np.transpose(avg, (1, 0, 2, 3))               # (Nx, Ny, 3, 3) -- explicit, never bare T
            slabs.append(LayeredSlab(float(z_m[k + 1] - z_m[k]), eps_tensor_cell=cell))
        return slabs
    if laterally_uniform:
        return slice_profile(v.mean(axis=(1, 2)), z_m, n_slices=n_slices)
    # structured scalar: one eps_cell (Ny,Nx)->(Nx,Ny) per native z-slice (n_slices resampling NA)
    slabs = []
    for k in range(z_m.size - 1):
        cell = np.transpose(0.5 * (v[k] + v[k + 1]), (1, 0))         # (Nx, Ny) -- explicit, never bare T
        slabs.append(LayeredSlab(float(z_m[k + 1] - z_m[k]), eps_cell=cell))
    return slabs


def _eps_fields_equal(a, b) -> bool:
    """True when two EpsFields carry the same eps content (uniform scalars/tensors compare
    by value; gridded fields by shape + axes + values). Used by collapse_regions_to_layers
    to decide whether lateral mesh subregions of one design layer can merge."""
    ua, ub = getattr(a, "is_uniform", True), getattr(b, "is_uniform", True)
    if ua != ub:
        return False
    if ua:
        ta, tb = a.tensor is not None, b.tensor is not None
        if ta != tb:
            return False
        if ta:
            return bool(np.allclose(np.asarray(a.tensor), np.asarray(b.tensor),
                                    rtol=1e-9, atol=0.0))
        return bool(np.isclose(complex(a.scalar), complex(b.scalar), rtol=1e-9, atol=0.0))
    va, vb = np.asarray(a.values_zyx), np.asarray(b.values_zyx)
    if va.shape != vb.shape:
        return False
    for axa, axb in ((a.z_axis_u, b.z_axis_u), (a.y_axis_u, b.y_axis_u),
                     (a.x_axis_u, b.x_axis_u)):
        if axa.shape != axb.shape or not np.allclose(axa, axb, rtol=1e-9, atol=0.0):
            return False
    return bool(np.allclose(va, vb, rtol=1e-9, atol=0.0))


def collapse_regions_to_layers(design, eps_by_region) -> dict:
    """Collapse a MESH-REGION-keyed eps_by_region (what run_pipeline's FEM bridge emits,
    where a design layer may be split into lateral subregions 'ito_inpatch'/'ito_outside*'
    and inclusion overlays 'grating__incl0'/'grating__bg1') into the DESIGN-LAYER-keyed
    dict the layered extractors (TMM slicer, Lumenairy RCWA/PMM bridges) consume.

    Per design layer: '__incl' overlay entries are DROPPED (the layered rasterizers freeze
    inclusion eps at the material value -- the documented contract); the remaining members
    must carry the SAME eps content (lateral FEM splits of one layer share the full-cell
    lifted field, so identical-by-construction) and merge to one entry; members that
    genuinely differ (a laterally split effect modulation no layered extractor can
    represent) raise. IDENTITY for an already-layer-keyed dict; superstrate / substrate /
    pml_* / other unmatched keys are ignored. Longest layer name claims a key first, so
    overlapping layer-name prefixes cannot mis-assign."""
    if not eps_by_region:
        return {}
    remaining = dict(eps_by_region)
    out = {}
    for L in sorted(design.stack.layers, key=lambda l: -len(l.name)):
        members = {}
        for k in list(remaining):
            if k == L.name or k.startswith(L.name + "_"):
                members[k] = remaining.pop(k)
        members = {k: v for k, v in members.items() if "__incl" not in k[len(L.name):]}
        if not members:
            continue
        keys = sorted(members)
        ref = members[keys[0]]
        for k in keys[1:]:
            if not _eps_fields_equal(ref, members[k]):
                raise ValueError(
                    "collapse_regions_to_layers: layer {!r} subregions {} carry DIFFERENT "
                    "eps fields (a laterally split modulation); the layered extractors "
                    "cannot represent it -- use the FEM solver".format(L.name, keys))
        out[L.name] = ref
    return out
