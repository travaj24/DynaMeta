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
_OFFDIAG_TOL = 1e-9     # |off-diag| / |diag| above which a tensor is rejected as non-diagonal


def _check_diagonal(off_max: float, diag_max: float) -> None:
    """RAISE if a tensor has significant OFF-DIAGONAL entries (relative to its diagonal).

    CONFIRMED NGSolve LIMITATION (diagnosed 2026-06-02 on NGSolve 6.2.2604, the latest PyPI release):
    the scattered-field HCurl solve MIS-ASSEMBLES a permittivity tensor with nonzero OFF-DIAGONAL
    entries on the periodic PML mesh. Decisive probe: a y-polarized ORDINARY wave through a uniaxial
    slab tilted in x-z (eps_yy unchanged, eps_xz != 0) must be tilt-invariant, but instead it CREATES
    energy (T = 1.07, R+T = 1.11). The defect is in NGSolve's assembly of mixed-component HCurl proxy
    cross-terms (u[j] v[i], i != j) on the complex periodic space; it is INDEPENDENT of how eps is
    expressed -- the matrix-CF matvec, an explicit scalar component sum, a .Compile()'d CF, and
    real-vs-complex entries ALL gave the identical broken result. There is no code-side workaround on
    this version and no newer NGSolve to upgrade to, so this stays a HARD ERROR (a suppressed warning
    would yield a silently-wrong, energy-non-conserving solve -- audit LC-1). The constitutive models
    (tilted LC, magneto-optic) are CORRECT and validated analytically / in their PRINCIPAL FRAME; only
    the off-diagonal FEM solve is deferred until a fixed NGSolve ships. Diagonalize the tensor (use its
    principal frame) to solve in the meantime."""
    if off_max > _OFFDIAG_TOL * (diag_max or 1.0):
        raise NotImplementedError(
            "assemble_eps_cf: eps tensor has significant OFF-DIAGONAL entries (max |off-diag|="
            "{:.3e} vs max |diag|={:.3e}); the FEM tensor matvec is validated ONLY for diagonal "
            "(principal-axis) tensors and mis-evaluates off-diagonal ones under PML (tracked P0b "
            "follow-on). Diagonalize the tensor (use its principal frame).".format(off_max, diag_max))


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
        off = np.abs(T - np.diag(np.diag(T)))
        _check_diagonal(float(np.max(off)), diag_max)
        # Snap sub-tolerance off-diagonals (e.g. the ~1e-17 cos(pi/2) residual of a homeotropic LC
        # director) to EXACT int 0 so they stay on the proven sparse matrix-CF path, not a tiny
        # dense complex entry (gotcha below). Past _check_diagonal every off-diagonal is <= tol, so
        # this yields a clean diagonal matrix for the guarded (diagonal-within-tol) cases.
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
    off_max = max(float(np.max(np.abs(v[..., i, j]))) for i in range(3) for j in range(3) if i != j)
    _check_diagonal(off_max, diag_max)

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
