# DynaMeta re-validation audit -- 2026-06-01

**Scope.** A second adversarial audit of the whole library, run after the first audit's fixes
plus the minor-upgrade batches, the Metasurface_Modulator ports (analysis bandwidth/C-V,
solver-comparison), and the RCWA-independent prep (`core/layered.py`, `optics/tmm_reference`
layered path, the pluggable `run_pipeline(optical_solver=...)` seam). 9 subsystems, 18 agents
(one adversarial finder + one independent skeptic-verifier per subsystem; the verifier tried to
refute each finding by reading the cited code).

**Method.** find -> independent verify, per subsystem. Each finding carries an attempted
refutation; the verifier independently reproduced or refuted it and corrected the severity.

**Headline.**
- **No regression** of any prior HIGH fix -- all confirmed still in place and correct (list at
  the end).
- **1 new HIGH**: GEO-1 (a clockwise-wound freeform `Polygon` inclusion silently swapped the
  inclusion and background materials).
- The remainder are MEDIUM (silent-failure / overclaim / integration-gap) and LOW/INFO
  (coverage, docstring honesty, BYO-contract robustness).

All actionable findings are **resolved** (commits `5f78d44`, `8bc7868`, `9015b3f`, `31249e4`,
`3370c53`). Verification: **53/53 pytest** (10 new regression tests) + the FEM/SP validations
`bandwidth_cv`, `inclusion_winding` (new), `audit_optics_fixes`, `graded_tmm_vs_fem` re-run
PASS, and a `bddc_gmres` solve exercised the new convergence check.

---

## Findings, verdicts, and resolutions

Severity is the verifier-corrected value. "Fix" gives the commit.

### HIGH

| ID | Subsystem | Finding (verdict: confirmed) | Fix |
|----|-----------|------------------------------|-----|
| GEO-1 | geometry | A clockwise-wound `Polygon` extruded to a negative-volume OCC face whose cell-intersection captured the COMPLEMENT of the footprint -> inclusion and background materials silently SWAPPED (or background vanished). `Polygon` is exported + advertised; `contains_m` is winding-insensitive so the DEVSIM and NGSolve stages modeled different geometry. | `5f78d44`: `_polygon_prism` normalizes winding to CCW (signed shoelace); empty-background guard. New `validation/inclusion_winding.py` drives CW + CCW through the real OCC builder (equal, analytic-correct inclusion volume; non-empty bg). |

### MEDIUM

| ID | Subsystem | Finding | Fix |
|----|-----------|---------|-----|
| OS-1 | optics-solver | `bddc_gmres`/`bddc_cg` can stop at maxsteps without converging and silently return a wrong field (the `printrates` warning is suppressed on the default pipeline path). Bites the ill-conditioned ENZ/metal regime. | `8bc7868`: `solve_fem` measures the relative residual `||b-Ax||/||b||` after the solve and warns if > 1e-3. Verified a clean `bddc_gmres` solve does not false-warn. |
| SP-NEG-1 | schrodinger-poisson | The oxide gate->psi_s bisection had no bracket-validity check; a NEGATIVE (depletion) gate produced a silently sign-flipped density (accumulation at negative gate). | `5f78d44`: bracket is expanded for the depletion branch and a warning is emitted if no root is bracketed (accumulation path unchanged). |
| AN-1 / CC-1 / AD-1 | analysis / 3d / cross-cut | `resonance_dip` referenced undefined `x1`/`y1` -> NameError on a NaN/inf spectrum (e.g. a failed solve point) instead of the documented discrete-min fallback. | `5f78d44`: returns `(lam[i], y[i])`; non-finite samples filtered before argmin/polyfit; clear raise on all-nonfinite. |
| LTM-4 / BLP-1 / CC-4 | layered / pipeline | The advertised `run_pipeline(optical_solver=...)` seam had no shipped adapter and `TmmLayeredSolver.solve(stack, lambda, optical)` did not match the `fn(design, geo, eps_by_region, lam_m, n_super, n_sub)` call signature -- the layered backend could not actually drop in, and the seam had zero coverage. | `9015b3f`: `make_layered_tmm_solver()` returns a callable with the exact seam signature; end-to-end seam test added. |
| BLP-2 | pipeline | `pipeline.py` hard-imported the FEM trio + DEVSIM builder at module top, so a pure layered/TMM path could not import without ngsolve, and `run_pipeline` always built the FEM mesh. | `9015b3f`: FEM trio + default builders lazily imported; verified the pipeline + adapter import and run under a solvers-blocked environment. |
| AN-2 | analysis | `validation/bandwidth_cv.py` ran a no-oxide device (identity-map psi_s -> C ~6-8x too large, f_3dB ~5-6 GHz) yet its END-TO-END header + `..._reproduces_modulator` test name implied device reproduction; the only gate was a degenerate 4-decade band. | `31249e4`: validation now runs WITH an oxide (physical C ~13 mF/m^2, f_3dB ~16 GHz), header reworded to "validates the chain is self-consistent, not the Modulator number", band tightened to [2,100] GHz + a C in [1,50] mF/m^2 gate that catches the identity-map regress. `test_bridge` test renamed `..._formula_with_modulator_C`. |

### LOW / INFO

| ID | Subsystem | Finding | Disposition |
|----|-----------|---------|-------------|
| AN-3 | analysis | `gate_cv` returned a silent NaN C on duplicate gate-bias points. | FIXED `5f78d44` (raise) + test. |
| AN-4 | analysis | `gate_cv` trusted the producer array layout; a transposed BYO grid (equal axis lengths) silently mis-integrated (~28%). | FIXED `3370c53` (shape assert vs axis lengths) + test. |
| AN-5 | analysis | `lumped_rc_bandwidth` gave a silent negative/inf f_3dB for C<=0 (depletion). | FIXED `5f78d44` (warn + NaN; scalar return preserved). |
| AN-6 | analysis | `gate_cv` Q is excess-over-n_bg, so SP hard-wall fields give Q(0)<0 (only C is meaningful). | DOC `5f78d44` (note in docstring). |
| OS-2 | optics-solver | `A_independent` region regex dropped a metacharacter material name (e.g. `ito.n+`). | FIXED `8bc7868` (`re.escape`). |
| OS-3 | optics-solver | `A_independent` excluded any material starting with `pml` (would drop a `pmlayer` structure). | FIXED `8bc7868` (`pml_` prefix). |
| OS-4 | optics-solver | `A_independent` still integrates a lossy super/substrate buffer (only clean for lossless cladding). | DOC `8bc7868` (note); behavior unchanged (validated cases use lossless cladding). |
| OS-5 | optics-solver | Bloch helpers have no fast pytest coverage (solver.py imports ngsolve, so not CI-gateable). | ACCEPTED (info): covered by the conical validations; the ky-phase itself is correct (no regression). |
| LTM-1 | layered | `graded_tmm_vs_fem` uses a symmetric+lossless profile -> R is order-insensitive, so a slab-order regression is invisible there. | FIXED `31249e4` (fast pure-TMM order-sensitivity test on an asymmetric lossy stack). |
| LTM-3 | layered | `slice_eps_field` + the `eps_cell` structured path were dead/untested. | FIXED `31249e4` (both branches unit-tested; structured stack -> `layered_rta` raises). |
| LTM-5 | layered | The layered TMM had no lossy-superstrate guard (the FEM has OPT-1). | FIXED `8bc7868` (`_coh_tmm_stack` raises on `Im(n_super)!=0`) + test. |
| SP-RELAX-2 | schrodinger-poisson | The `relax` docstring overclaimed that under-relaxation tames the isolated-well limit cycle (it does not). | FIXED `31249e4` (docstring corrected). |
| SP-MO0-3 / CC-2 | schrodinger-poisson | `solve_self_consistent(max_outer<1)` raised UnboundLocalError on `dV`. | FIXED `5f78d44` (`dV=inf` init -> reports non-converged). |
| SP-TAUT-4 | schrodinger-poisson | The pytest degenerate-filling check is a normalization tautology. | ACCEPTED (info): the genuine closed-form physics check exists in `validation/schrodinger_poisson.py`. |
| DD-1 / DD-2 / CC-3 | carriers-dd | g-factor docstrings overclaimed "exact in both limits" (degenerate coeff ~15% low) and "<1%" (true ~1.1% peak). | FIXED `31249e4` (docstrings corrected to ~1.1% peak / <0.5% over ITO; degenerate-limit caveat). |
| DD-3 | carriers-dd | The load-bearing g-fit had zero unit coverage and its coefficients were DUPLICATED in two devsim-importing modules. | FIXED `3370c53` (extracted to pure `carriers/einstein.py` single source; `tests/test_carriers_gfactor.py` pins coefficients + accuracy vs exact F_1/2/F_-1/2). |
| DD-4 | carriers-dd | The bipolar Boltzmann-limit gate is degenerate (g->1 at 1e-4 vs a 2% tol), so it never constrains the FD-enhancement magnitude. | MITIGATED `3370c53` (the new unit test gates the g MAGNITUDE directly across eta). |
| DD-5 | carriers-dd | Bipolar holes use the conduction-band N_dos (no valence N_v). | DOC `31249e4` (note); correct for all validated (non-degenerate) bipolar cases. |
| BLP-3 | bridge | A BYO native-3D field with a non-z stack axis is silently mis-axised. | ACCEPTED (low): documented in two places; the auto-built 3D alignment carries `stack_axis='y'` by default, so a `!='z'` raise would false-trigger the shipped path. |
| AD-2 | 3d-devsim | A biased back/body contact whose name was not the resolved `body_name` had its bias silently dropped (the guard only fired when NEITHER gate nor body matched). | FIXED `8bc7868` (also warns on any EXTRA bias key matching neither gate nor body). |
| GEO-2 | geometry | Degenerate shape inputs (n_sides<3, rx/ry<=0, <3 vertices) built silent-garbage solids. | FIXED `5f78d44` (`__post_init__` guards) + test. |
| GEO-3 | geometry | The ellipse 72-gon under-states area by ~0.127% (aspect-INdependent; refutes the feared high-aspect blow-up). | DOC `3370c53` (note at the sampling site); far below tolerance. |
| GEO-4 | geometry | `_identify_periodic` signature rounding (1 pm) vs the boundary-face tol (40 pm) is a latent mis-pairing hazard, now marginally more exposed by polygon-sampled wrapped faces -- but unreachable in every tested case. | ACCEPTED (low): matches the prior BI-5 "currently unreachable" rating; not escalated. |
| GEO-5 | geometry | New OCC inclusion shapes have no CI/pytest coverage, and no test exercised a user `Polygon` (which let GEO-1 ship). | FIXED `31249e4`/`5f78d44` (pure-numpy shape-guard + winding tests; `validation/inclusion_winding.py` drives a `Polygon` through the OCC builder). |

---

## Prior HIGH fixes -- regression check (all HELD)

Independently re-confirmed in place and correct: OPT-1 (oblique raises on a non-vacuum
superstrate), OPT-2 (independent volumetric `A_independent`), OPT-3/4/7, the theta<=60 cap and
the conical ky-Bloch phase; SP-1 (`.converged` + warn on every return path), SP-2 (flat-band
oxide-cap baseline); the FD g-factor accurate rational fit in both DD modules; `from_design`
role=='biased' gate selection + gate-name threading + multi-semi/non-square raises; the bridge
`time_convention` assert; `SeparableXYLift` square + single-sign guards; BI-1 (semiconductor-in-
inclusion raises); the 2D dielectric missing-`eps_static_dc` raise; and all PASS/FAIL
validations exit non-zero on failure.

---

## Verification

- `pytest tests/ -q` -> **53 passed** (was 39; +10 audit regression tests across
  `test_layered`, `test_bridge`, `test_geometry_shapes`, `test_carriers_gfactor`), and the new
  pure modules import under a sys.modules-blocked devsim/ngsolve/netgen/gmsh environment.
- Validations re-run PASS: `bandwidth_cv` (physical C ~13 mF/m^2, f_3dB ~16 GHz),
  `inclusion_winding` (GEO-1 end-to-end, CW==CCW), `audit_optics_fixes` (A_independent + OPT
  guards), `graded_tmm_vs_fem` (graded-TMM vs FEM |dR|=0.0010). A `bddc_gmres` solve exercised
  the OS-1 convergence check without crash or false-warn.
- House rules: ASCII-only `print()`/source, SI units, validations exit-code-gated. DynaMeta
  push policy: commits are local; not pushed unless explicitly requested.
