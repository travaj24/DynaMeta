"""
Minimal self-contained check of whether NGSolve correctly assembles an OFF-DIAGONAL matrix-valued
(anisotropic) coefficient in an HCurl bilinear form -- the construct DynaMeta uses for tensor eps
(tilted liquid crystal, magneto-optic). No DynaMeta, no PML, no scattered field, no R/T extraction:
just the assembled mass matrix M[a,b] = INT (eps . phi_b) . phi_a dx, with two UNIMPEACHABLE checks.

  (1) Assemble M two MATHEMATICALLY IDENTICAL ways and compare:
        matrixCF  : the matrix-CF matvec      (eps_cf . u) . v
        scalarSum : explicit scalar component  sum_ij eps_ij (u[j] v[i])     [plain-complex coeffs]
      They must agree (to round-off).
  (2) HCurl basis functions are real, so for a real-SYMMETRIC eps the assembled M must be SYMMETRIC
      (M = M^T) and for a HERMITIAN eps it must be HERMITIAN (M = M^H) -- forced by the math, with
      no physics. A correct assembly satisfies this.

Run both on PLAIN HCurl and a genuinely PERIODIC HCurl, single-matrix AND a multi-material
domain-list (CoefficientFunction([M_a, M_b, ...]), the exact DynaMeta assembler construct), int-0
sparse AND dense-0j zeros.

RESULT on NGSolve 6.2.2604: every case agrees to ~1e-16 and preserves symmetry/Hermiticity to
~4e-17. NGSolve assembles off-diagonal tensors CORRECTLY -- there is NO matrix-assembly bug. (So if a
higher-level anisotropic solve gives a wrong reflectance, the cause is NOT this assembly; look at the
PML coordinate stretch, the source term, or the R/T extraction.)

Run: python docs/ngsolve_offdiag_check.py
"""
import numpy as np
import scipy.sparse as sp
import ngsolve as ng
import netgen.occ as occ

print("ngsolve", ng.__version__)


def _csr(mat):
    r, c, v = mat.COO()
    return sp.csr_matrix((np.asarray(v, dtype=complex), (np.asarray(r), np.asarray(c))),
                         shape=(mat.height, mat.width))


def _maxabs(A):
    return float(abs(A).max()) if A.nnz else 0.0


def _region_matrix(M, dense):
    if dense:
        ent = tuple(complex(M[i, j]) for i in range(3) for j in range(3))
    else:                                              # int-0 sparse zeros (DynaMeta's construct)
        ent = tuple(complex(M[i, j]) if M[i, j] != 0 else 0 for i in range(3) for j in range(3))
    return ng.CoefficientFunction(ent, dims=(3, 3))


def _eps_cf(Mby, mats, dense):
    # Mby: {material_name: 3x3}. Single material -> uniform matrix; else the per-material domain-list.
    if len(mats) == 1:
        return _region_matrix(Mby[mats[0]], dense)
    return ng.CoefficientFunction([_region_matrix(Mby[m], dense) for m in mats])    # domain-list


def _matrix_cf(Mby, mats, fes, dense):
    u, v = fes.TnT()
    a = ng.BilinearForm(fes, symmetric=False)
    a += (_eps_cf(Mby, mats, dense) * u) * v * ng.dx
    a.Assemble()
    return _csr(a.mat)


def _scalar_sum(Mby, mats, fes):
    u, v = fes.TnT()
    a = ng.BilinearForm(fes, symmetric=False)
    for i in range(3):
        for j in range(3):
            if len(mats) == 1:
                comp = complex(Mby[mats[0]][i, j])
            else:
                comp = ng.CoefficientFunction([complex(Mby[m][i, j]) for m in mats])
            a += comp * (u[j] * v[i]) * ng.dx
    a.Assemble()
    return _csr(a.mat)


def _check(Mby, mats, fes, kind, label):
    A = _matrix_cf(Mby, mats, fes, dense=False)
    A_dense = _matrix_cf(Mby, mats, fes, dense=True)
    B = _scalar_sum(Mby, mats, fes)
    agree = _maxabs(A - B) / max(_maxabs(A), 1e-300)
    agree_d = _maxabs(A_dense - B) / max(_maxabs(A_dense), 1e-300)
    viol = (_maxabs(A - A.conj().T) if kind == "herm" else _maxabs(A - A.T)) / max(_maxabs(A), 1e-300)
    req = "M=M^H" if kind == "herm" else "M=M^T"
    ok = agree < 1e-12 and agree_d < 1e-12 and viol < 1e-12
    print("  [{:22s}] matrixCF-vs-scalarSum int0={:.1e} dense={:.1e} | {} violation={:.1e}  {}".format(
        label, agree, agree_d, req, viol, "OK" if ok else "** MISMATCH **"))
    return ok


def _periodic_mesh(two_material):
    if two_material:
        b0 = occ.Box(occ.Pnt(0, 0, 0), occ.Pnt(1, 1, 0.5)); b0.name = "a"
        b1 = occ.Box(occ.Pnt(0, 0, 0.5), occ.Pnt(1, 1, 1)); b1.name = "b"
        shape = occ.Glue([b0, b1])
    else:
        shape = occ.Box(occ.Pnt(0, 0, 0), occ.Pnt(1, 1, 1))
    shape.faces.Min(occ.X).Identify(shape.faces.Max(occ.X), "px", occ.IdentificationType.PERIODIC)
    shape.faces.Min(occ.Y).Identify(shape.faces.Max(occ.Y), "py", occ.IdentificationType.PERIODIC)
    return ng.Mesh(occ.OCCGeometry(shape).GenerateMesh(maxh=0.6))


def main():
    D = np.diag([2.0, 3.0, 4.0]).astype(complex)
    S = D.copy(); S[0, 2] = S[2, 0] = 0.7                # real symmetric off-diagonal
    H = D.copy(); H[0, 1] = 0.7j; H[1, 0] = -0.7j        # gyrotropic, Hermitian
    ISO = (2.25 + 0j) * np.eye(3)
    ok = True
    # single-material, plain + periodic
    for periodic in (False, True):
        m = _periodic_mesh(False) if periodic else ng.Mesh(
            occ.OCCGeometry(occ.Box(occ.Pnt(0, 0, 0), occ.Pnt(1, 1, 1))).GenerateMesh(maxh=0.6))
        mats = list(m.GetMaterials())
        fes = (ng.Periodic(ng.HCurl(m, order=1, complex=True)) if periodic
               else ng.HCurl(m, order=1, complex=True))
        tag = "periodic" if periodic else "plain"
        print("single-material HCurl ({}, ndof {}, mats {}):".format(tag, fes.ndof, mats))
        ok &= _check({mats[0]: S}, mats, fes, "sym", "offdiag-sym " + tag)
        ok &= _check({mats[0]: H}, mats, fes, "herm", "gyrotropic " + tag)
    # multi-material domain-list (DynaMeta's construct), periodic: region "a" anisotropic, rest iso
    m = _periodic_mesh(True)
    mats = list(m.GetMaterials())
    fes = ng.Periodic(ng.HCurl(m, order=1, complex=True))
    print("multi-material domain-list HCurl (periodic, mats {}, ndof {}):".format(mats, fes.ndof))
    ok &= _check({mt: (S if mt == "a" else ISO) for mt in mats}, mats, fes, "sym",
                 "offdiag-sym domainlist")
    ok &= _check({mt: (H if mt == "a" else ISO) for mt in mats}, mats, fes, "herm",
                 "gyrotropic domainlist")
    print("\n*** NGSolve off-diagonal tensor assembly is CORRECT: {} ***".format(
        "CONFIRMED" if ok else "FAILED -- a real assembly bug"))
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
