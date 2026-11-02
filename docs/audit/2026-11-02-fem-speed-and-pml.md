# FEM speed: factorization reuse, solver auto-selection, PARDISO; the PML default (2026-11-02)

Version `0.11.7 -> 0.12.0`. Branch `feat/fem-speed-pml`, stacked on `feat/sym-diagonal-tensor` (PR #31).
Two of the changes alter default results: the PML default (more accurate, by up to 5e-2 in R) and the
default linear solver (a direct solve instead of BDDC-GMRes on small cells: differences ~1e-9).

---

## 1. Where the time goes

Measured on the order-2 HCurl test cell (`tests/test_fem_symmetry_tensor.py` geometry) at 35k / 68k /
136k DOFs, 6 threads, ngsolve 6.2.2604: **assembly 0.05-0.27 s, factorization 1.6-34 s.** A
direct solve is its factorization. So the gains come from factorizing less often or more cheaply;
reusing assembly or splitting the bilinear form cannot help (assembly is <= 2%, the FESpace build is
milliseconds).

## 2. What changed

| Change | Where | Measured |
|---|---|---|
| `FEMFactorCache`: the SAME matrix again reuses the factorization exactly (one back-substitution) | `solver.py` | r_y from the shared factorization = a separate `solve_fem` to <= 9e-16; back-solve 0.02-0.2 s vs factorization 1.6-34 s |
| ... a NEARBY matrix (next wavelength / bias / LC state) is solved by GMRes preconditioned with the stored factorization, accepted only at free-DOF relres <= 1e-8, else refactorized | `_direct_solve` | 68k DOFs: fresh 5.7 s; +1 nm 0.29 s (4 its), +50 nm 0.58 s (9 its), LC +50 deg 1.9 s (25 its); abs(dr) <= 1e-12 |
| `solve_fem_pols(geo, lam, eps, opt, pols)`: several polarizations, one factorization (x/y at normal incidence, s/p at one oblique angle; Jones with `cross_pol=True`) | `solver.py` | ~2x per Jones pair |
| `linear_solver="auto"` (new default): direct up to `auto_direct_max_dofs` = 80k free DOFs (sparsecholesky if complex-symmetric, else PARDISO, else umfpack <= 40k), BDDC-GMRes above | `_auto_linear_solver` | direct 5.6 s vs BDDC 8-11 s at 63k; BDDC 9.3 s vs 25 s at 136k |
| `linear_solver="pardiso"` (MKL; `pip install mkl` / `dynameta[pardiso]`), falls back to umfpack without MKL | `pardiso_available` | 1.2x (6 threads) / 1.6x (12) over umfpack, 34 vs 81 kB/DOF |
| `gmres_rtol` now reaches GMRes as `reltol=` (default 1e-8) | `solve_fem`, `solve_fem_sourced` | see 3 |
| `num_threads` / env `DYNAMETA_FEM_THREADS` pin `ngsolve.SetNumThreads` | `_apply_fem_threads` | umfpack: 10.4 s at 1 thread vs 11.5 s at 6 -- pin when running processes in parallel |
| post-solve diagnostics under the TaskManager | `solve_fem` | ~40% of a 35k-DOF solve was the single-threaded flux diagnostic; ~5x faster threaded |
| `Mesh3DSpec.pml_thk_m` 400 -> 1000 nm | `specs.py` | see 4 |
| `run_pipeline`'s default FEM path and `make_fem_optical_solver` carry one cache per run / solver | `pipeline.py`, `solver.py` | -- |
| `solve_fem_sourced`: `auto`/`pardiso`; `ams`/`hypre` now fall back instead of reaching `pre.mat` on `None` | `solver.py` | -- |
| `ngsolve>=6.2.2608` (was `>=6.2`) | `pyproject.toml` | 70 FEM tests + the full suite pass on 6.2.2608 |

Peak memory per free DOF (98k DOFs): umfpack 81 kB, sparsecholesky 27 kB, PARDISO 34 kB, BDDC 5 kB.
The `auto` thresholds keep a direct solve under ~2.2 GB (sparsecholesky) / ~2.7 GB (PARDISO).

**End to end** (5 wavelengths x 2 polarizations, 63k DOFs, 6 threads, other jobs on the box):

| Path | Wall | max abs(dr) vs umfpack |
|---|---|---|
| 0.11 default: `bddc_gmres` per (lambda, pol) -- already with the relative 1e-8 tolerance | 48.4 s | 7.5e-9 |
| `umfpack` per (lambda, pol) | 211 s | 0 |
| 0.12: `auto` + `solve_fem_pols` + one `FEMFactorCache` (1 factorization, 1 exact reuse, 8 preconditioned) | **26.9 s** | **1.2e-11** |

## 3. A latent bug: `gmres_rtol` was an absolute tolerance

`ngsolve.solvers.GMRes(..., tol=...)` forwards `tol` to the solver's `atol` and `reltol` to its
relative tolerance (same in 6.2.2604 and 6.2.2608). DynaMeta passed `tol=optical.gmres_rtol`, so
the "relative" 1e-6 was an absolute bound on the preconditioned residual, and how tight it was
depended on the scale of the right-hand side. On a 204k-DOF metal-pillar cell it over-converged:

| Setting | its | time | abs(dr) vs direct |
|---|---|---|---|
| old `tol=1e-6` (absolute) | 142 | 16.0 s | 4e-12 |
| `reltol=1e-6` | 73 | 7.2 s | 2e-7 |
| `reltol=1e-8` (new default) | 101 | 10.2 s | 3e-9 |

`CGSolver(tol=)` was already relative. A caller who set `gmres_rtol` explicitly now gets a genuinely
relative tolerance -- looser than before for the same number, still far below discretization error.

## 4. The PML default was too thin

The z-PML is a constant `alpha=1j` HalfSpace stretch ending on a natural (reflecting) face, so the
outgoing wave comes back attenuated by ~exp(-2 k n cos(theta) d). Flat 100 nm metal mirror
(eps -90+10j, R = 0.977) vs exact TMM, maxh_pml 200 nm, x-pol:

| PML | 1060 nm | 1300 nm | 1550 nm |
|---|---|---|---|
| 400 nm (old default) | 6.5e-3 | 3.0e-2 | **5.2e-2** |
| 600 nm | 1.0e-5 | 3.7e-3 | 6.0e-3 |
| 800 nm | 1.2e-3 | 7.5e-4 | 1.4e-3 |
| **1000 nm (new)** | 4.9e-4 | 4.0e-4 | 7.5e-6 |

(Non-monotonic in thickness because the residual reflection interferes with the structure's.) At
45 deg even 800 nm leaves 1.5e-2. Buffers, by contrast, did not matter on a flat stack (30-900 nm,
same error): their job is to let the evanescent diffraction orders of a PATTERNED cell decay before
the probe planes, which depends on period/lambda and is design-specific.

Validation scripts that set their own PML (most use 500-700 nm) are unchanged; any `Design` that
relies on `Mesh3DSpec()` now meshes a thicker (coarse, `maxh_pml_m`) PML.

## 5. Gates -- `tests/test_fem_speed.py` (18; 17 + 1 MKL-gated, 60 s)

- x/y and s/p share one factorization and match separate solves to 1e-12 (umfpack and sparsecholesky);
  a symmetry cell does NOT share (walls follow the polarization) and still matches.
- nearby wavelengths: 1 factorization + 2 preconditioned, abs(dr) <= 1e-9 vs fresh solves; a forced
  stall (`max_its=1`) refactorizes and matches to 1e-12; a different mesh clears the cache;
  `make_fem_optical_solver().solve_sweep` uses it.
- `auto` picks sparsecholesky / BDDC (cap 0) / PARDISO-or-umfpack (oblique, gyrotropic) and matches.
- `pardiso` without MKL falls back with an advisory; with MKL it matches umfpack to 1e-10.
- GMRes receives `reltol=` and no `tol=`; `num_threads` and the env var pin threads (spec wins).
- default PML vs TMM: 1000 nm within 2e-3 at 1550 nm, and the old 400 nm OUTSIDE 1e-2 on the same
  case (the gate discriminates).
- CI: `test_fem_speed.py` joins the `solvers` job; a final step installs MKL and re-runs the
  PARDISO/auto tests (last in the job: MKL makes PARDISO ngsolve's default for untyped `Inverse()`).

Full local suite on this branch: **1780 passed, 1 skipped** (the MKL test), ngsolve 6.2.2608.

## 6. Also resolved

- v0.9.0 audit **F-22** (`make_fem_optical_solver._solve_sweep` built the reused FESpace before any
  `mesh.SetPML`): the space is now built inside `solve_fem`, after the PML is set, and cached there.

## 7. Not done here

An angle-aware / graded PML (would lift the ~50 deg oblique cap), and a buffer-thickness check
(warn when the buffer is < ~3 decay lengths of the first evanescent order).
