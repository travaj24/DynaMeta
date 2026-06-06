# Off-diagonal tensor eps in the FEM optical solver: is it an NGSolve bug? (RESOLVED)

## Short answer

**No -- it is NOT an NGSolve bug, and it is now FIXED.** A minimal, self-contained reproducer
(`docs/ngsolve_offdiag_check.py`) shows that NGSolve 6.2.2604 assembles an off-diagonal
(anisotropic) matrix-valued coefficient in an HCurl bilinear form **correctly, to machine
precision**, in every construct DynaMeta uses. An earlier note in this repo attributed the
off-diagonal optical-solve failure to an "NGSolve assembly defect"; that attribution was
**overstated and is retracted**. The symptom (a wrong, energy-non-conserving reflectance for a
tilted/gyrotropic tensor through `optics/solver.solve_fem`) was real, and its cause was **inside the
DynaMeta pipeline**: **`mesh.SetPML`'s automatic coordinate stretch is correct only for isotropic
media; for an anisotropic (off-diagonal) eps it perturbs the physically decoupled field component by
a resolution-independent ~3%.** The fix is an explicit **UPML** (anisotropic PML material tensor) for
the tensor path -- see "The fix" below. Off-diagonal tensor FEM (tilted LC, magneto-optic) is now
fully supported and validated.

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

## Where the real failure came from (root cause)

The matrix is assembled correctly, so a sequence of decisive probes localized the failure:

1. **The coefficient is right.** The assembled domain-list `eps_cf` evaluated at a slab point equals
   the intended tensor to **0.0**, and `eps_cf @ e_y = (0, eps_yy, 0)` -- no spurious y-coupling at the
   coefficient level.
2. **The weak-form expansion is irrelevant.** The matrix-CF matvec `(eps_cf*u)*v` and the explicit
   component sum `sum_ij eps_ij u[j] v[i]` give a **bit-identical** (and identically wrong) field --
   so it is not a matvec-assembly footgun.
3. **The error is resolution-INDEPENDENT.** The transmitted ordinary-wave amplitude was 1.0272 at
   order 2 and order 3, and at two mesh densities -- bit-identical. A convergent Galerkin error of a
   *correct* operator would shrink under p- and h-refinement; a fixed error means the discrete operator
   converges to the **wrong continuous operator**.
4. **It is the PML.** With `mesh.SetPML` removed (a uniform absorbing-loss regularization in its
   place), the off-diagonal FULL-tensor field is **identical to the diagonal field to ~1e-6** -- i.e.
   the off-diagonal eps does NOT couple into the decoupled component without the PML. With `SetPML`
   active, it perturbs it by ~3%. So `mesh.SetPML`'s coordinate stretch is wrong for an anisotropic
   medium (it is exact only for the isotropic case, where it equals the UPML tensor -- the
   `tensor_isotropic_gate` confirms UPML and SetPML agree to ~2e-15 for isotropic eps).

`A_independent` (the volumetric Im(eps)|E|^2 loss) was ~0 in the failing case, as it must be for a
lossless medium -- which by itself did not localize the cause; the decisive evidence was the
PML-on/PML-off field comparison.

## The fix

For a **tensor** eps, `solve_fem` no longer calls `mesh.SetPML`; it folds an explicit **UPML** into
the weak form. For a z-stretch PML the UPML material tensor is `Lambda = diag(s_z, s_z, 1/s_z)`
(`s_z = 1 + alpha` inside the PML, `1` outside; `alpha = 1j` matches the old HalfSpace), giving

```
INT  sum_i Lambda^-1_i  curl(u)_i curl(v)_i   -   k0^2  sum_ij (Lambda eps)_ij  u[j] v[i]   dx
```

This is the rigorous stretched-coordinate PML for an arbitrary (anisotropic) medium and reduces to
the `SetPML` answer for isotropic/diagonal eps (so the validated scalar path keeps `SetPML`
untouched; only the tensor path switches to UPML). A fit-independent **Poynting-flux** R/T
(`OpticalResult.R_flux/T_flux`) was also added: it reads the full z-power straight from the field, so
it correctly measures the TOTAL (co + cross) transmission for a gyrotropic medium whose transmitted
wave is elliptical (the single-projection lstsq `R/T` sees only the co-polarized channel).

## Validation

- `validation/lc_tilted_fem.py` -- tilted LC at theta = 0,30,45,60,90 deg: the ordinary (y) wave is
  tilt-invariant to dT ~ 1.6e-4 (was T = 1.07); the extraordinary (x) wave matches the n_eff(theta)
  scalar TMM to dT < 2.4e-3; energy and the flux R/T close to ~1e-4.
- `validation/magneto_optic_faraday.py` GATE D -- the gyrotropic (Hermitian, complex off-diagonal)
  tensor through the FEM matches the circular-eigenmode Jones-TMM (flux T_total and lstsq co-pol T to
  ~1.4e-2) and is lossless (R_flux + T_flux = 1, A_independent ~ 0).
- `validation/tensor_isotropic_gate.py`, `lc_uniaxial_fem.py`, `pockels_phase_modulator.py`,
  `thermo_optic_modulator.py`, `reconfigurable_modulators.py` -- unchanged (no regression).

`eps_assembler._check_diagonal` (the hard off-diagonal guard) is **removed**; the constitutive models
(`LiquidCrystalModel`, `MagnetoOpticModel`) were always correct and are now validated end-to-end
through the FEM, not just analytically.

## Reproduce

```
python docs/ngsolve_offdiag_check.py        # NGSolve assembly is correct (unchanged)
python -m validation.lc_tilted_fem          # off-diagonal FEM fixed (UPML)
python -m validation.magneto_optic_faraday  # gyrotropic FEM vs Jones-TMM
```
