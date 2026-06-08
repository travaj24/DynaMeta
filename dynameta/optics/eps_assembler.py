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

from typing import Dict

import numpy as np
import ngsolve as ng

from dynameta.core.eps_field import EpsField
from dynameta.optics.ngsolve_layered import OpticalGeometry

_ID3 = ng.CoefficientFunction((1, 0, 0, 0, 1, 0, 0, 0, 1), dims=(3, 3))
# |off-diag| / |diag| at or below which an off-diagonal entry is snapped to EXACT int 0 (the
# ~1e-17 cos(pi/2) residual of a homeotropic LC director), keeping the proven sparse matrix-CF path.
# OFF-DIAGONAL tensors are now FULLY SUPPORTED in the FEM solve: the earlier failure was NOT an
# NGSolve assembly defect (docs/ngsolve_offdiag_check.py proves the assembly is exact to ~1e-16) but
# mesh.SetPML's coordinate stretch being wrong for an anisotropic medium. solve_fem now uses an
# explicit UPML (anisotropic PML material tensor) for tensor eps -- see solver.solve_fem -- so a
# tilted-LC / magneto-optic off-diagonal tensor solves correctly (energy-conserving, tilt-invariant
# ordinary wave). No off-diagonal rejection remains.
_OFFDIAG_TOL = 1e-9


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
        diag_max = float(np.max(np.abs(np.diag(T))))
        # Snap sub-tolerance off-diagonals (e.g. the ~1e-17 cos(pi/2) residual of a homeotropic LC
        # director) to EXACT int 0 so they stay on the proven sparse matrix-CF path, not a tiny
        # dense complex entry (gotcha below). GENUINE off-diagonals (|off| > tol) are kept and solved
        # via the UPML tensor path (no rejection); only the numerical-noise residuals are snapped.
        tol = _OFFDIAG_TOL * (diag_max or 1.0)

        def _ent(i, j):
            z = T[i, j]
            if i != j and abs(z) <= tol:
                return 0
            return complex(z) if z != 0 else 0
        entries = tuple(_ent(i, j) for i in range(3) for j in range(3))
        return ng.CoefficientFunction(entries, dims=(3, 3))
    v = np.asarray(ef.values_zyx)                              # graded tensor (Nz,Ny,Nx,3,3)
    start, end = ef.voxel_bounds_u()
    diag_max = max(float(np.max(np.abs(v[..., k, k]))) for k in range(3))      # same metric as uniform

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
    # Reverse coverage: an eps_by_region key with no matching mesh material is a stale region name
    # / bridge-vs-mesh naming drift. The CF is built by iterating `mats`, so the extra entry is
    # SILENTLY dropped -- and the physics it carried (e.g. a biased-carrier eps) never reaches the
    # solve while some other key satisfies that material. Fail loudly (anti-silent-failure).
    extra = [k for k in eps_by_region if k not in set(mats)]
    if extra:
        raise ValueError(
            "assemble_eps_cf: eps_by_region has {} entr{} with no matching mesh material: {} "
            "(stale region name / bridge-vs-mesh naming drift; silently ignored otherwise). Mesh "
            "materials are: {}.".format(len(extra), "y" if len(extra) == 1 else "ies",
                                        sorted(extra), sorted(mats)))
    if not any(eps_by_region[m].is_tensor for m in mats):
        # scalar path (unchanged): one domain-wise scalar CF keyed by material ordinal
        return ng.CoefficientFunction([_scalar_region_cf(eps_by_region[m]) for m in mats])
    # tensor path: one domain-wise list of self-contained per-region 3x3 matrices
    return ng.CoefficientFunction([_region_matrix_cf(eps_by_region[m]) for m in mats])
