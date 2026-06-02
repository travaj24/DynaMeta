"""
The NGSolve half of the bridge: turn the bridge's per-region EpsField dict into a single domain
CoefficientFunction over the mesh. Uniform EpsFields become constant CFs; gridded ones become
VoxelCoefficients (axes already in nm, values in (Nz,Ny,Nx) order).

If ANY region is anisotropic (a 3x3 tensor eps), the WHOLE domain CF is built as a 3x3 MATRIX
(scalar regions promoted to the isotropic eps*I), since a single bilinear-form term needs one
tensor rank everywhere. The matrix is built as ONE domain-wise list whose per-region entries are
self-contained 3x3 matrices: ng.CoefficientFunction([region_matrix for each material]).

CRITICAL (verified the hard way): the domain (per-material) dispatch must be the OUTER list. A
per-material domain-wise list CF nested INSIDE a dims=(3,3) matrix CF integrates correctly as a
coefficient but evaluates WRONG in the matvec (eps.u).v form on a periodic multi-material mesh.
Packing VoxelCoefficients/constants into a matrix is fine -- only the per-material *list* must
stay outermost. Solver-specific -- keeps NGSolve out of core/.
"""

from __future__ import annotations

import warnings
from typing import Dict

import numpy as np
import ngsolve as ng

from dynameta.core.eps_field import EpsField
from dynameta.optics.ngsolve_layered import OpticalGeometry

_ID3 = ng.CoefficientFunction((1, 0, 0, 0, 1, 0, 0, 0, 1), dims=(3, 3))


def _warn_if_offdiagonal(T: np.ndarray) -> None:
    """Warn that a tensor with significant OFF-DIAGONAL entries is currently unreliable in the FEM.

    KNOWN LIMITATION (tracked P0b follow-on): the per-region matrix-CF matvec ((eps.u).v) on the
    periodic PML mesh is validated only for DIAGONAL (principal-axis) tensors -- the isotropic gate,
    the Pockels z-cut oracle, and the LC planar/homeotropic states are all diagonal. A tensor with
    nonzero off-diagonal entries (a tilted optic axis, magneto-optic, ...) mis-evaluates in the
    matvec (validation/lc_uniaxial_fem.py: a y-polarized ordinary wave that must see only eps_yy is
    instead corrupted by eps_xz). The constitutive model is correct; the FEM solve of an off-diagonal
    tensor is not yet. Diagonalize the tensor (work in its principal frame) until this is fixed."""
    od = np.abs(T) - np.diag(np.diag(np.abs(T)))
    if float(np.max(od)) > 1e-9 * float(np.max(np.abs(np.diag(T))) or 1.0):
        warnings.warn(
            "assemble_eps_cf: a region eps tensor has significant OFF-DIAGONAL entries; the FEM "
            "tensor matvec is currently validated only for DIAGONAL (principal-axis) tensors and "
            "mis-evaluates off-diagonal ones under PML (tracked P0b follow-on). Use the principal "
            "frame (a diagonal tensor) for a trustworthy FEM result.", RuntimeWarning, stacklevel=3)


def _scalar_region_cf(ef: EpsField):
    """Scalar CF for a SCALAR EpsField: a constant (uniform) or a VoxelCoefficient (gridded)."""
    if ef.is_uniform:
        return ng.CoefficientFunction(complex(ef.scalar))
    start, end = ef.voxel_bounds_u()
    return ng.VoxelCoefficient(start=start, end=end, values=ef.values_zyx, linear=True)


def _region_matrix_cf(ef: EpsField):
    """Self-contained 3x3 matrix CF for ONE region (no per-material list inside). Scalar /
    graded-scalar -> isotropic (scalar or VoxelCoefficient) * I; uniform tensor -> a constant
    matrix; graded tensor -> a matrix of per-component VoxelCoefficients.

    IMPORTANT: zero matrix entries are the literal int 0, NOT complex(0j). NGSolve builds a
    different (sparse vs dense) matrix-CF expression tree for the two, and a dense all-complex
    matrix CF mis-evaluates in the matvec (eps.u).v form on the periodic PML mesh (verified:
    identical coefficient values, but a wrong R), whereas the sparse-zero form is correct."""
    if not ef.is_tensor:
        return _scalar_region_cf(ef) * _ID3
    if ef.tensor is not None:                                  # uniform 3x3
        T = np.asarray(ef.tensor, dtype=np.complex128)
        _warn_if_offdiagonal(T)
        entries = tuple(complex(T[i, j]) if T[i, j] != 0 else 0
                        for i in range(3) for j in range(3))
        return ng.CoefficientFunction(entries, dims=(3, 3))
    v = np.asarray(ef.values_zyx)                              # graded tensor (Nz,Ny,Nx,3,3)
    start, end = ef.voxel_bounds_u()
    diag_max = max(float(np.max(np.abs(v[..., k, k]))) for k in range(3))
    off_max = max(float(np.max(np.abs(v[..., i, j]))) for i in range(3) for j in range(3) if i != j)
    if off_max > 1e-9 * (diag_max or 1.0):
        _warn_if_offdiagonal(np.array([[1.0, off_max, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]))

    def _comp(i, j):
        if not np.any(v[..., i, j]):                           # identically-zero component -> int 0
            return 0
        return ng.VoxelCoefficient(start=start, end=end,
                                   values=np.ascontiguousarray(v[..., i, j]), linear=True)
    return ng.CoefficientFunction(tuple(_comp(i, j) for i in range(3) for j in range(3)),
                                  dims=(3, 3))


def assemble_eps_cf(geo: OpticalGeometry,
                      eps_by_region: Dict[str, EpsField]) -> ng.CoefficientFunction:
    mats = list(geo.mesh.GetMaterials())
    for region in mats:
        if region not in eps_by_region:
            raise ValueError("no EpsField for mesh region '{}' (bridge/alignment "
                              "coverage gap)".format(region))
    if not any(eps_by_region[m].is_tensor for m in mats):
        # scalar path (unchanged): one domain-wise scalar CF keyed by material ordinal
        return ng.CoefficientFunction([_scalar_region_cf(eps_by_region[m]) for m in mats])
    # tensor path: one domain-wise list of self-contained per-region 3x3 matrices
    return ng.CoefficientFunction([_region_matrix_cf(eps_by_region[m]) for m in mats])
