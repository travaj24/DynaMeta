# Off-diagonal tensor eps in the FEM optical solver: is it an NGSolve bug?

## Short answer

**No -- it is NOT a confirmed NGSolve bug.** A minimal, self-contained reproducer
(`docs/ngsolve_offdiag_check.py`) shows that NGSolve 6.2.2604 assembles an off-diagonal
(anisotropic) matrix-valued coefficient in an HCurl bilinear form **correctly, to machine
precision**, in every construct DynaMeta uses. An earlier note in this repo attributed the
off-diagonal optical-solve failure to an "NGSolve assembly defect"; that attribution was
**overstated and is retracted here**. The symptom (a wrong, energy-non-conserving reflectance for a
tilted/gyrotropic tensor through `optics/solver.solve_fem`) is real, but its cause is **inside the
DynaMeta pipeline**, not the NGSolve matrix assembly.

## What was claimed, and how it was tested

The claim under test: the scattered-field HCurl solve mis-evaluates a permittivity tensor with
nonzero OFF-DIAGONAL entries, so a y-polarized ordinary wave through a uniaxial slab tilted in x-z
came out with transmission T = 1.07 (energy created) instead of the tilt-invariant T = 0.9998.

`docs/ngsolve_offdiag_check.py` isolates the assembly with two checks that need **no physics** -- the
correct answer is forced by linear algebra:

1. **Two identical assemblies must agree.** The mass matrix `M[a,b] = INT (eps . phi_b) . phi_a dx`
   is built (a) via the matrix-CF matvec `(eps_cf . u) . v` and (b) via an explicit sum of
   scalar-coefficient component forms `sum_ij eps_ij (u[j] v[i])`. These are mathematically the same
   matrix.
2. **Symmetry is forced.** HCurl basis functions are real, so a real-symmetric eps must give a
   symmetric M (`M = M^T`) and a Hermitian eps must give a Hermitian M (`M = M^H`).

Run on: plain HCurl and a genuinely periodic HCurl; a single uniform matrix CF AND the
multi-material **domain-list** `CoefficientFunction([M_a, M_b, ...])` that DynaMeta's assembler
actually emits; with int-0 sparse zeros AND dense `0j` zeros; for a real-symmetric off-diagonal eps
and a gyrotropic (Hermitian) eps.

## Result (NGSolve 6.2.2604)

Every case:

- matrix-CF vs scalar-sum agreement: **~1.6e-16** (machine round-off).
- symmetry / Hermiticity violation: **~4e-17** (machine round-off).
- int-0 and dense zero encodings: identical.
- single matrix and multi-material domain-list: identical.
- plain and periodic HCurl: identical.

So the NGSolve assembly of off-diagonal/anisotropic tensor coefficients in HCurl is correct. There is
no assembly defect, no Periodic-space defect, and no int-0-vs-dense issue at the matrix level.

## Documentation / source / comment check

- NGSolve documents matrix-valued `CoefficientFunction`s and their use in bilinear forms as a
  first-class feature; anisotropic Maxwell problems are an intended use (the Maxwell tutorial + a
  forum thread, "Suitability for anisotropic electrodynamics problem", on exactly off-diagonal
  permittivity tensors with periodic BCs).
- The `nonzero in-out not overloaded for ConstantCoefficientFunctionC` message seen earlier comes
  only from the `.Compile()` path (NGSolve's `NonZeroPattern` sparsity analysis in `fem/coefficient.cpp`)
  and is an info/diagnostic about Compile, NOT an assembly error -- plain (non-Compiled) assembly,
  which is what DynaMeta uses, does not emit it and is correct per the reproducer above.
- The `InnerProduct has been changed and takes now conjugate` notice is a documented NGSolve API
  change; NGSolve detects an already-conjugated second argument and does not double-conjugate
  (confirmed: the lossless slab returns A_independent ~ 0).
- The NGSolve GitHub issue tracker has no matching issue (the project routes user issues to its
  forum). No documented limitation matches the claimed assembly defect.

## So where does the real failure come from?

The off-diagonal optical solve gives a wrong R/T, but the matrix is assembled correctly. The
remaining suspects are all **DynaMeta-side** and still being isolated:

- the PML coordinate stretch (`mesh.SetPML`, a complex z-stretch) combined with the full curl-curl +
  anisotropic mass operator;
- the scattered-field source term `((eps - eps_bg I) . E_bg) . v`;
- the R/T extraction (`_lstsq_2wave` / `_reflection` / `_transmission`), which fits a single-
  polarization plane-wave pair and may misread an off-diagonal-perturbed transmitted field.

Note that `A_independent` (the volumetric Im(eps)|E|^2 loss integral) was ~0 in the failing case, as
it must be for a lossless medium -- which does NOT by itself prove the field is correct, so it does
not yet localize the cause.

## Implications

- **`eps_assembler._check_diagonal` stays a guard** (an off-diagonal tensor still produces a wrong
  R/T end-to-end, so returning one silently would be worse), but its rationale is corrected: the
  block is a DynaMeta-pipeline limitation under investigation, NOT a proven NGSolve assembly defect.
- **Off-diagonal tensor FEM (tilted-LC, magneto-optic) is therefore likely DynaMeta-fixable**, not
  NGSolve-blocked -- a real fix (most likely a two-projection R/T extractor and/or the PML
  interaction) is the path, rather than waiting on an upstream NGSolve release.
- The constitutive models (`LiquidCrystalModel`, `MagnetoOpticModel`) remain correct and are
  validated analytically; only the off-diagonal FEM *solve+extraction* is deferred.

## Reproduce

```
python docs/ngsolve_offdiag_check.py
# -> "*** NGSolve off-diagonal tensor assembly is CORRECT: CONFIRMED ***"
```
