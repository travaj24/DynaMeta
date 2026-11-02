# Tensor eps on symmetry-reduced FEM cells, a direct LDL^T solver, cross-pol reflection (2026-11-02)

Four changes to `optics.solver.solve_fem`. The first three are opt-in; the fourth is a fix to the
iterative-solver residual check, which was **false-alarming on every PEC-walled symmetry cell**.
They were written on 2026-09-25 for the exp31 coax study and carried uncommitted until now. This
pass reviewed them, fixed three defects found in review (one wrong answer, two late refusals),
and added the gates.

Branch `feat/sym-diagonal-tensor`, rebased onto main `95766c0` (v0.11.6). Version
`0.11.6 -> 0.11.7`.

---

## 1. What was built

| Where | What |
| --- | --- |
| `solver.py` `solve_fem` | symmetry-reduced cells accept a tensor eps whose off-diagonal part MEASURES zero; `tensor_mirror_symmetric=` caller assertion; `cross_pol=` -> `OpticalResult.r_cross`; `linear_solver="sparsecholesky"` |
| `solver.py` `_free_dof_relres` (new) | the iterative/LDL^T residual over FREE DOFs only |
| `solver.py` `solve_fem_sourced` | `sparsecholesky` falls back to `umfpack` (advisory) -- it used to reach `pre.mat` on `None` |
| `core/interfaces.py` | `OpticalResult.r_cross: Optional[complex] = None` |
| `geometry/specs.py` | `"sparsecholesky"` in the `linear_solver` literal |
| `tests/test_fem_symmetry_tensor.py` (new) | 11 gates; listed in the `solvers` CI job |

### API

```python
from dynameta.optics.solver import solve_fem

# a uniaxial LC with its director along x, on a quarter cell (Mesh3DSpec.symmetry="quarter")
res = solve_fem(geo, lam, eps_cf, OpticalSpec(polarization="x"), n_sub=1.45)

# a tensor FIELD the caller knows to be mirror-symmetric about every wall (unchecked)
res = solve_fem(geo, lam, eps_cf, opt, tensor_mirror_symmetric=True)

# one Jones column: x-pol -> (res.r, res.r_cross) = (r_xx, r_yx); y-pol -> (r_xy, r_yy)
res = solve_fem(geo, lam, eps_cf, OpticalSpec(polarization="x"), cross_pol=True)

# direct LDL^T for the complex-symmetric normal-incidence matrix
OpticalSpec(polarization="x", linear_solver="sparsecholesky")
```

---

## 2. The physics of each change

**Diagonal tensor on a reduced cell.** The PEC/PMC walls are exact when the structure is
invariant under the wall mirrors. A tensor diagonal in (x, y, z) is invariant under both,
because `M eps M = eps` for `M = diag(-1, 1, 1)` and every diagonal `eps`. An off-diagonal x-y or
x-z entry flips sign under the x-mirror, so it is admitted only if the tensor FIELD is odd about
the wall: `eps(Mx) = M eps(x) M`. A director tilted toward radial about the cell centre is an
example. The solver now integrates `|eps_ij|^2` over the mesh and refuses an off-diagonal L2
norm above `1e-12` of the diagonal's. The odd-field case can be asserted
(`tensor_mirror_symmetric=True`) but not verified, because the reduced mesh holds no mirrored
points to compare against.

**`sparsecholesky`.** At normal incidence with no Bloch phase and a symmetric eps, the
scattered-field curl-curl matrix is complex-SYMMETRIC (not Hermitian). NGSolve's LDL^T is
then a valid direct solve. It does not check symmetry: on a non-symmetric matrix it reads one
triangle and solves a different operator. So the free-DOF residual is measured, and anything
above `1e-8` raises. A genuine solve sits at ~1e-13. A Bloch-phased space (oblique incidence)
is known to be non-symmetric before factorizing, so it falls back to `umfpack` with an advisory
instead of factorizing in order to refuse.

**`cross_pol`.** The 0-order reflection is projected on the lab axis orthogonal to the drive.
The layered background reflects no cross-pol, so `r_cross` is the scattered-field projection
alone. It is defined at normal incidence for `'x'`/`'y'` only, and it is refused **before** the
solve otherwise. `'p'` at normal incidence IS `'x'`, and the p-pol branch has no cross projection.

**Free-DOF residual.** Dirichlet rows (PEC walls) are constrained, never solved. `f` has
entries there that `A u` does not reproduce. Counting them made the "iterative solver did not
converge" warning fire on every converged PEC-walled GMRes solve.

---

## 3. Defects found in review (fixed here)

1. **`r_cross` on a reduced cell was WRONG.** It fitted 5.0e-3 on a quarter cell whose full-cell
   solve gives 2.4e-5. The cross component is odd about the walls. Its evanescent odd orders
   average to zero over the full cell but not over a reduced one. A mirror-symmetric cell
   cannot convert x <-> y at normal incidence, so `r_cross` is now returned as exactly `0j`
   there, and is not fitted.
2. `cross_pol` at oblique incidence raised only AFTER the full solve. It was silently ignored
   (returned `None`) for `'p'`. Both are now refused before the mesh is touched.
3. `sparsecholesky` at oblique incidence factorized and only then refused by residual. In
   `solve_fem_sourced` it crashed on `pre.mat` with `pre = None`. The first now falls back up
   front; the second falls back to `umfpack`.

The duplicated residual computation (once inside the `sparsecholesky` branch, once after) is now
one helper.

---

## 4. Gates -- `tests/test_fem_symmetry_tensor.py`, 11 passed (55 s, 6 threads)

| Claim | Measured |
| --- | --- |
| diagonal tensor: quarter == full | max abs(dr) **3.1e-4** (x), **6.5e-5** (y), gate 2e-3; premise abs(r_x - r_y) = **0.060** (the anisotropy is load-bearing) |
| off-diagonal tensor refused on a reduced cell | `NotImplementedError`; admitted with `tensor_mirror_symmetric=True` |
| `sparsecholesky` == `umfpack` | abs(dr) **1.1e-14** (scalar, full) / **~5e-15** (diagonal tensor, quarter), gate 1e-10 |
| `sparsecholesky` refuses a gyrotropic tensor | residual **3.9e-2** vs the 1e-8 cap |
| `sparsecholesky` at oblique falls back | advisory emitted; abs(dr) vs `umfpack` **3.8e-16** |
| free-DOF residual on a PEC-walled quarter cell | free **2.2e-9 / 4.2e-9** vs all-DOF **1.99 / 2.57** (3744 constrained DOFs, x/y pol); GMRes R == umfpack R to 1e-6; no warning |
| `cross_pol` refused off-regime | `'p'`, 20 deg oblique, 30 deg azimuth: all raise before touching the mesh |
| cross-pol Jones vs an INDEPENDENT eigenframe solve | below |
| mirror-symmetric pillar | full cell abs(r_cross) **2.4e-5**; reduced cell exactly `0j` |

**The cross-pol oracle.** The test structure is a 260 x 100 nm rectangle rotated 45 deg in a
400 nm square lattice. Its Jones matrix is diagonal in the rectangle frame. The same crystal,
rotated -45 deg, is axis-aligned rectangles on a sqrt(2)-P square supercell holding two of them.
Its plain co-pol solves give the eigen-reflections `r_u`, `r_v` with no cross-pol extraction
anywhere. Rotating back gives `r_xx = (r_u + r_v)/2` and `r_yx = (r_u - r_v)/2`:

| | lattice solve | eigenframe oracle | `|diff|` |
| --- | --- | --- | --- |
| `r_xx` | -0.03899 + 0.00884j | -0.03946 + 0.00914j | 5.6e-4 |
| `r_yx` (= `r_cross`) | +0.05614 + 0.01485j | +0.05618 + 0.01496j | **1.2e-4** |

`|r_cross|` = 0.058; the wrong sign would miss by 0.116. Diagonal mirror: `|r_xx - r_yy|` =
2.2e-4. Reciprocity: `|r_yx - r_xy|` = 1.8e-4.

**Why not the Poynting flux.** `R_flux` should close `|r|^2 + |r_cross|^2`. At the test mesh it is
under-resolved: on the plain circle pillar it reads **-3.3e-4** against R = 4.3e-3. It converges
under refinement (rot45: 0.00228 / 0.00531 / 0.00662 at maxh scale 1 / 0.67 / 0.5, against
`|r|^2 + |r_cross|^2` = 0.0063 at every scale), but only at ~45 s per solve. That is a
pre-existing property of the best-effort diagnostic, not a regression. It is recorded here so
nobody reaches for `R_flux` as a cheap cross-pol oracle.

---

## 5. Defaults unchanged

Every new keyword defaults off. On a cell with no Dirichlet DOFs the free-DOF residual equals
the old all-DOF one (periodic minion rows carry `f = 0`, `A u = 0`; measured 1.377e-13 both
ways), so only the warning on PEC-walled cells moves.

**Measured against v0.11.6 (`95766c0`, a separate worktree, separate processes).** Six
default-path solves were run: scalar full/quarter, `umfpack`/`bddc_gmres`, 20-deg oblique, and a
diagonal and a gyrotropic tensor. NGSolve's threaded assembly is NOT bitwise reproducible
run to run. Two runs of unmodified v0.11.6 differ by up to **3.4e-15** in `r`. The branch
differs from either v0.11.6 run by up to **3.7e-15**, which is inside that floor. The one
deterministic case (`umfpack`, tensor) is bit-identical across all three runs.

## 6. Verification

- `tests/test_fem_symmetry_tensor.py`: 11 passed.
- The `solvers` CI module list + `test_graded_tensor_fem`: see the PR.
- `validation.symmetry_reduced_fem` (nightly oracle): **PASS**. GATE A 2.3e-5, GATE B 1.8e-5,
  GATE C walls-dropped 0.036 vs correct 1.8e-5.
- `ruff check .`: clean.

`sparsecholesky` speed, measured here: **2.0x** over `umfpack` (8.9 s vs 17.7 s, scalar full cell,
2223 vertices). The ~3x of 2026-09-25 was measured on the larger exp31 coax cell.
