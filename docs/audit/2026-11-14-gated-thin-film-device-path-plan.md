# Gated thin-film device path: assessment and implementation plan (2026-11-14)

Companion to `docs/audit/2026-11-14-gated-thin-film-device-path-audit.md` (23 findings against v0.15.1,
commit 2e1af87). This document (1) records an independent check of every finding and proposed fix,
(2) gives a verdict per finding, and (3) turns the accepted work into packages that separate engineers can
build in parallel git worktrees without talking to each other. No library code was changed. Every number
marked MEASURED was produced for this assessment (v0.15.1 imported from a read-only clone at 2e1af87,
DEVSIM 2.11.2, Python 3.14, numpy 2.4.4, BLAS pinned to 2 threads, Windows); numbers marked AUDIT are quoted
from the audit and were not re-run. The audit's fixtures were later replaced by generic ones (audit,
"Fixtures"): every number in this plan was then re-measured on them (on an export of the v0.15.1 tree with
lumenairy 5.50.2), taken from the audit's re-measured value, or quoted from the v0.16 package note that
re-measured it (named in place), unless it is marked "(not re-measured on the generic fixture)".

**Terms** (in addition to the audit's "Terms" section):
- **Package (P-n):** one engineer's unit of work, in its own worktree, owning a disjoint set of files.
- **Wave:** packages of one wave run in parallel; a later wave starts from the merged result of the
  earlier ones. Files are disjoint within a wave; a later-wave package may own a file an earlier one
  touched.
- **Staging:** the integration branch `wip/v0160_stage`; packages merge there, never into each other.
- **Gate:** a test with a stated threshold and an independent oracle (house rule, see section 3).
- **L_next:** the Lumenairy release that ships the sister Lumenairy audit's fixes
  (lumenairy's `docs/audits/AUDIT_METAL_STRIP_GRATING_SOLVERS_2026_11_14.md`: geometric
  PMM grading G-1, uniform-layer fast path G-2, `add_layer(formulation='auto')` default G-3, fixed
  `rcwa_convergence` G-5, `n_orders` alias warning G-7). Its number is not fixed yet; 5.51.0 is expected.
- **Fixture A / Fixture B:** the audit's generic strip cell (metal strips 150 nm of a 400 nm period, 40 nm
  tall, over an oxA / oxB / 6 nm film / oxB / oxA spacer on an 80 nm mirror metal, glass, TM, 1200 nm) and
  generic gated film (6 nm, n_bg 3e26 m^-3, DOS mass 0.30 m_e, static eps 9.0, between 2 nm oxA + 6 nm oxB
  on each face, metal gates; top-gate biases +5 / -3 V) (audit, "Fixtures").

There is no `CONTRIBUTING.md` or `CHANGELOG` in this repository. House conventions used below come from
`README.md`, `Makefile`, `pyproject.toml` (`filterwarnings = ["error"]`, ruff `select = F, E9`),
`.github/workflows/ci.yml`, the release notes `docs/audit/*-release.md`, and the maintainer's standing rules
(one implementation per numerical kernel; accuracy over byte identity, with pins re-recorded and the old/new
delta documented; a failure is never "flaky"; bitwise gates are confirmed on a second machine or CI, not on
the main workstation alone; never run two pytest processes of this repo at once on Windows).

---

## 1. Summary

### 1.1 Verdicts

| ID | Audit status | Verdict | Correction to the audit (short) | Package |
|---|---|---|---|---|
| O1 | missing feature | IMPLEMENT WITH CHANGES | average per partition interval as a laminate TENSOR (harmonic eps_xx, arithmetic eps_yy/eps_zz), adaptive partition with a tolerance, one partition per layer | P3 |
| O2 | not a defect | IMPLEMENT WITH CHANGES | also pass `far_field_orders=` instead of the `n_orders=` alias (L_next will warn on the alias) and document that it does not control accuracy | P3 |
| O3 | High | IMPLEMENT WITH CHANGES | 2-D rule must follow lumenairy's Li capability (rectangles); every add_layer call must pass `formulation` explicitly (shapes-Laurent and `LumenairyStackSolver` omit it today); add a solver fingerprint and bump the cache schema | P2 |
| O4 | Medium | IMPLEMENT WITH CHANGES | primary fix = node-registered AREA-weighted raster (+ harmonic companion for 'li'); "choose S so edges fall on pixel boundaries" depends on the sampling convention and fails for incommensurate edges | P2 |
| O5 | not reproduced | REJECT | not library code; N2 covers the real defect | - |
| C1 | High | IMPLEMENT WITH CHANGES | native grid in z only (x stays uniform); FEM VoxelCoefficient silently assumes uniform axes, so non-uniform axes must be refused there; native grid is requested by a capability flag on layered optical solvers | P4 (+flag in P1/P2/P3) |
| C2 | Medium | IMPLEMENT WITH CHANGES | an equilibrium edge electrode becomes the region's Fermi-level reference (exact for any voltage), not merely "no Dirichlet"; explicit `pin_potential=True` keeps the old behaviour | P5 + P11 |
| C3 | as designed | IMPLEMENT (docs) | - | P4 |
| C4 | High | IMPLEMENT WITH CHANGES | the error is NOT thickness-dependent: MEASURED identical on a 20 nm film; the default grading under-resolves every accumulation/depletion layer. Fix must be screening-length-aware, not thickness-aware | P4 |
| C5 | Medium | IMPLEMENT WITH CHANGES | guard now in one shared helper used by both 2-D and 3-D builders; full Kane closure as a later-wave package via a generalized Fermi-integral expansion; refuse the Kane band on DD/bipolar regions in v1 | P5 (guard), P11 (calls), P13 (closure) |
| C6 | Medium | IMPLEMENT WITH CHANGES | charge-only sheets/bulk charge as Poisson node terms independent of the trap machinery; work function requires `chi_eV` (refuse if None) | P5 + P11 |
| C7 | not reproduced | IMPLEMENT (docs only) | DEVSIM behaviour; one sentence in `carriers/devsim_env.py` | P4 |
| Q1 | Medium | IMPLEMENT WITH CHANGES | apply a linear end-correction ramp to the warm start (exact ends, smooth interior, bit-identical when ends already match) instead of overwriting two nodes | P6 |
| Q2 | High | IMPLEMENT WITH CHANGES | the 0 V deficit of a film with oxide on both faces is partly PHYSICAL (both walls are real); what is spurious is the Dirichlet body. Gauss's law at each face must use the conservative face flux | P12 |
| Q3 | Medium | IMPLEMENT WITH CHANGES | also applies at alpha = 0 (T = 0 relation used at 300 K: +0.28 % bulk excess, -1.01 meV, MEASURED); one shared bulk-E_F kernel; closed-form oracle Nc F_1/2 at alpha = 0 | P6 |
| Q4 | High | IMPLEMENT WITH CHANGES | the legacy form is wrong in EVERY regime (it adds the sub-band carriers' own Drude response along z), not "valid for an isolated well"; deprecate it; optional explicit extra free-carrier density instead | P7 |
| Q5 | Medium | IMPLEMENT WITH CHANGES; DEFER item (3) | `hard_wall="both"` + thin-film guard now; in-Newton DG for equilibrium films deferred (needs a degenerate-film calibration) | P8 |
| Q6 | Medium | IMPLEMENT WITH CHANGES | interface condition is continuity of J_z and of beta^2 n1 (the induced pressure), NOT beta^2 div J / wp^2; validate against an independent ODE oracle | P9 |
| Q7 | missing feature | split: Q7.1 IMPLEMENT; Q7.2 DEFER; Q7.3 REJECT as specified | local 2-D periodic TM FEM now; nonlocal graded 2-D FEM is multi-week research (documented limitation); a "local |psi|^2-weighted eps_zz(z)" is not a unique or physical object | P10 |
| N1 | Low | IMPLEMENT WITH CHANGES | resample onto lumenairy's NODE lattice x_j = j P/S (the analytic-shape frame), not onto cell centres | P2 |
| N2 | High | IMPLEMENT WITH CHANGES | one shared `slice_layer_eps_field` at all SIX call sites (incl. FDTD seam); also catches the Nz = 1 structured field that slices to zero slabs; the pipeline path is already safe (bridge remaps z) | P1 (+call sites P2/P3) |
| N3 | Low | IMPLEMENT | vectorize (one matrix product) | P7 |
| N4 | High | IMPLEMENT; plus an open decision | sign fix confirmed; the method's flat-band baseline also omits the gate-face dead-layer field (0.20-0.25 V in psi_s on the audit's film, MEASURED) | P6 |

### 1.2 Packages

| Pkg | Name | Wave | Files owned | Depends on | Size (agent-days) |
|---|---|---|---|---|---|
| P1 | Slab-thickness guard | 1 (first) | `core/layered.py`, `optics/tmm_reference.py`, `optics/fdtd_seam.py`, `optics/lumenairy_bridge/berreman_backend.py`, `optics/lumenairy_bridge/pmm2d_backend.py` | - | 1 |
| P2 | RCWA bridge | 1 | `optics/lumenairy_bridge/rcwa_backend.py`, `optics/lumenairy_bridge/_shapes.py`, new `optics/lumenairy_bridge/_raster.py`, `optics/lumenairy_bridge/_common.py` (FEATURE_FLOORS entry only) | P1 (helper name) | 5-6 |
| P3 | PMM bridge | 1 | `optics/lumenairy_bridge/pmm_backend.py`, new `optics/lumenairy_bridge/_lateral_partition.py` | P1 (helper name) | 3-4 |
| P4 | Carrier grid and mesh | 1 | `carriers/devsim_layered.py`, `pipeline.py`, `core/resample.py`, `core/eps_field.py`, `geometry/specs.py`, `carriers/dc_solve.py` (docs), `carriers/devsim_env.py` (docs) | - | 4-5 |
| P5 | Carrier physics primitives | 1 | `carriers/physics_equilibrium.py`, `carriers/physics_traps.py`, `geometry/electrode.py`, `materials/transport_model.py` | - | 3-4 |
| P6 | Schrodinger-Poisson core | 1 | `carriers/schrodinger_poisson.py`, `carriers/sp_carrier.py` | - | 2 |
| P7 | Intersubband permittivity | 1 | `core/effects/electroabsorption.py` | - | 1.5-2 |
| P8 | Frozen-potential DG guard | 1 | `carriers/density_gradient.py` | - | 1 |
| P9 | Hydrodynamic layer runs | 1 | `optics/nonlocal_tmm.py` | - | 3-4 |
| P10 | 2-D periodic TM FEM | 1 | new `optics/fem_2d_tm.py` | - | 5-6 |
| P11 | Carrier wiring | 2 | `carriers/devsim_layered.py`, `carriers/devsim_3d.py` | P4, P5 | 2-3 |
| P12 | Dual-gate Schrodinger-Poisson | 2 | new `carriers/sp_dual_gate.py`, `carriers/__init__.py` exports | P6 | 4-5 |
| P13 | Kane band in the electrostatics | 3 | new `carriers/kane_band.py`, `carriers/physics_equilibrium.py`, `materials/transport_model.py`, `materials/scattering.py`, `carriers/devsim_layered.py` (wiring lines) | P5, P11 | 5-7 |
| P14 | Lumenairy L_next adoption | after L_next | `optics/lumenairy_bridge/_common.py`, `pyproject.toml` (lumenairy line only) | P2, P3, L_next on PyPI | 1 |
| P15 | Release integration | last | `pyproject.toml` (version), `dynameta/__init__.py` (`__version__`), `cache.py` (`_SCHEMA`), `README.md`, `.github/workflows/ci.yml` (union of test-file entries), `docs/audit/<date>-v0.16.0-release.md` | all | 1-2 |

Total: about 42-54 agent-days; with five concurrent engineers (house cap) about 2.5-3 calendar weeks,
set by the chain P4/P5 -> P11 -> P13.

**Proposed version: 0.16.0** (minor): six defaults change results (RCWA factorization, RCWA raster,
carrier grid on the layered path, DEVSIM mesh grading, intersubband eps_zz, Schrodinger-Poisson E_F and
gate map), and one cache schema bump.

### 1.3 Decisions needed from the owner (details in section 9)

1. Version 0.16.0. *Recommend yes.*
2. RCWA default `formulation='auto'` in 0.16.0 (not a warning-only patch). *Recommend yes.*
3. RCWA raster default `'area'` (node-registered, area-weighted), `'point'` as opt-in legacy. *Recommend yes.*
4. Native z grid automatically on the layered-optics pipeline path. *Recommend yes.*
5. New default DEVSIM interface grading (screening-length-aware). *Recommend yes.*
6. Equilibrium edge electrode = Fermi-level reference; `pin_potential=True` for the old column pin. *Recommend yes.*
7. `IntersubbandEffect` default `z_response='subbands'`; legacy mode deprecated. *Recommend yes.*
8. Schrodinger-Poisson E_F calibrated at finite T also for alpha = 0 (moves default results by ~1 mV). *Recommend yes.*
9. Single-gate Schrodinger-Poisson gate map: switch to the face-field Gauss law (moves psi_s by ~0.2 V). *Recommend yes, with the legacy map kept as an option.*
10. Lumenairy floors: keep VERSION_FLOOR 5.22.0; add feature floors; raise VERSION_VERIFIED_MAX to L_next after re-verification. *Recommend yes.*
11. Ship 0.16.0 without P13 if P13 is late (guard only, closure in 0.16.1). *Recommend yes.*
12. Defer the in-Newton DG for equilibrium films and the nonlocal 2-D FEM. *Recommend yes.*

---

## 2. Corrections to the audit that change the work

These were found while checking the audit; each changes what the engineers must build.

**C4 is not a thin-film problem (MEASURED).** The same gated stack with a 20 nm film instead of 6 nm,
top gate +5 V / -3 V, centre column:

| film | mesh | nodes | n(top) at +5 V | excess sheet +5 V [m^-2] | n(top) at -3 V |
|---|---|---|---|---|---|
| 20 nm | default | 47 | 9.67916e26 | 4.608565e17 | 2.30001e25 |
| 20 nm | fine (0.0125 / 0.05 / 2.25 nm) | 467 | 9.85913e26 | 4.593008e17 | 2.06914e25 |
| 6 nm | default | 19 | 9.67920e26 | 4.608031e17 | 2.29985e25 |
| 6 nm | fine | 189 | 9.85913e26 | 4.592524e17 | 2.06913e25 |

The default-mesh error (-1.8 % in accumulation, +11 % in depletion) is the same at 20 nm as at 6 nm,
because the y mesh lines near each interface are identical (zone 0.8 nm < 0.45 t in both). A
thickness-triggered rule (the audit's "t < 10 nm") would leave every thicker film wrong. Refining only
`interface_ps_m` does nothing (0.0125 nm: n(top) 9.67964e26); what matters is the spacing out to a few
screening lengths. Candidate settings on the 20 nm film (MEASURED, relative to the 467-node row):

| interface_ps / coarse_ps / zone [nm] | nodes | n(top) +5 V, rel. err | n(top) -3 V, rel. err |
|---|---|---|---|
| 0.0125 / 0.5 / 2.25 | 63 | -2.8e-3 | +2.4e-2 |
| 0.0125 / 0.2 / 2.25 | 140 | -4.3e-4 | +3.4e-3 |
| 0.025 / 0.1 / 2.25 | 236 | -1.4e-4 | +1.2e-3 |

(P4 later found that every DEVSIM-graded mesh, the 467-node one included, keeps a ~0.1 nm cell at the
upper face of each layer, so these references are themselves off by a small amount; its gates use an
independent 1-D oracle, `2026-11-14-v0.16-carrier-grid-mesh.md`.)

**N4: the sign fix is right; the method's reference is incomplete (MEASURED).** With the audit's film
(6 nm, n_bg 3e26, m 0.30 m_e, eps_s 9.0, oxide 8 nm / eps 20): the library returns psi_s = -12.19 V at
-1 V and -13.19 V at -2 V with NO warning (the bracket expansion "finds" a spurious sign change). The
sign-fixed residual gives -0.250435 / -0.523020 / -1.141830 V at -1 / -2 / -4 V and +0.457603 V at +2 V,
and brackets on [0, |Vg|] for both signs (f(0) = -|Vg|; f(s|Vg|) has the sign of N_exc, which is s). But
the map measures N_exc against the flat-band (psi_s = 0) profile, which with hard walls carries a face
field D_face(0) = -2.04e-2 C/m^2 (the gate-side dead layer; equivalent to 0.92 V across this oxide). The
physical boundary condition is Gauss's law at the oxide face, eps_ox (Vg - psi_s)/t_ox = D_s(face):

| Vg | fixed flat-band map psi_s | face-field map psi_s |
|---|---|---|
| +2 V | +0.457603 | +0.658488 |
| +1 V | +0.234345 | +0.442191 |
| -1 V | -0.250435 | -0.019089 |
| -2 V | -0.523020 | -0.270639 |

(sign-fixed and face-field values as shipped and re-measured for `2026-11-14-v0.16-sp-core.md`, the face
field with the solver's conservative face flux and the finite-temperature model E_F; the assessment's own
first estimate used a one-sided flux on the original film and was not re-measured on the generic fixture.)
The Dirichlet body absorbs only about 0.5 % of the field change (D_body - D_body0 = -2.7e-4 vs D_face -
D_face0 = 5.01e-2 C/m^2 at +2 V), so the difference is the gate-side dead layer, not the body. Ship the sign
fix in any case; the face-field map is decision 9.

**Q2: what is spurious.** A film gated through oxide on both faces has a real hard wall on both faces, so
part of the 0 V deficit is physical. The spurious ingredient is the Dirichlet phi = 0 at the body, which
absorbs the body-side dead-layer field that in reality ends on the second gate. The dual-gate solver must
therefore not be gated on "zero deficit at 0 V"; it is gated on Gauss's law, symmetry and limits.

**Q3 applies at alpha = 0 too (MEASURED).** The SP model's own bulk relation (parabolic z, Kane in-plane,
finite T) at the carrier's E_F:

| alpha [1/eV] | carrier E_F | n_sep/n_bg | calibrated E_F | shift |
|---|---|---|---|---|
| 0 | 0.54471 eV | 1.00278 (= Nc F_1/2(E_F/kT)/n_bg, closed form) | 0.54370 eV | -1.01 meV |
| 0.4 | 0.46005 eV | 0.89570 | 0.49236 eV | +32.31 meV |

(generic film n_bg 3e26 m^-3, m 0.30 m_e, 300 K; values as re-measured for `2026-11-14-v0.16-sp-core.md`.)
At alpha = 0 the T = 0 relation (3 pi^2 n)^(2/3) is used at 300 K; that is the audit's -1.0 mV and its
1.0029 flat-band ratio. With the calibrated E_F the flat-band centre density of 20 / 40 nm slabs is
0.99928 / 1.00007 n_bg (MEASURED, 0.025 nm grid). The T -> 0 ratio formula (1 + 0.8 aE)/(1 + aE)^(3/2) was
re-derived (integrate the sub-band sheet density pref0 (dE + alpha dE^2) over k_z) and gives 0.8904,
consistent with the finite-T 0.89570.

**Q4: the legacy form is wrong everywhere.** `eps_zz = eps_xx(Drude, sub-band density) + sum ISB` adds
the z free-carrier response of the SAME electrons the sub-band sum already describes. For a complete set
the sum rule makes it a double count (MEASURED f-sum 0.99967-0.99972 for 6-40 nm slabs with the
consistent quadrature); for an isolated well the confined electrons have no free motion along z at all,
so a Drude z term built from their density is also wrong. Only a SEPARATE free population (not in the
sub-band set) has a Drude z response, and it needs its own density. MEASURED trend of the corrected
eps_zz toward the isotropic bulk (calibrated E_F, m = m_opt = 0.30 m_e, n_bg 3e26 m^-3, eps_inf 4.0,
gamma 1.2e14 rad/s, 1200 nm; values as re-measured for `2026-11-14-v0.16-intersubband-permittivity.md`):

| slab | eps_xx | library eps_zz | eps_inf + ISB | abs(eps_zz - eps_xx) | f-sum |
|---|---|---|---|---|---|
| 6 nm | 2.9580+0.0797i | 1.7987+0.2204i | 2.8405+0.1408i | 0.1324 | 0.99967 |
| 10 nm | 2.8549+0.0875i | 1.6838+0.2190i | 2.8289+0.1316i | 0.0511 | 0.99969 |
| 20 nm | 2.7857+0.0928i | 1.5575+0.2073i | 2.7717+0.1146i | 0.0258 | 0.99971 |
| 40 nm | 2.7507+0.0955i | 1.4949+0.2025i | 2.7441+0.1071i | 0.0133 | 0.99972 |

The difference falls as 1/L (quantum size effect), so a fixed "< 0.05 at 20 nm" bar measures physics,
not correctness; the gates in P7 use the sum rule and the 1/L trend instead.

**N3 (MEASURED):** with the Riemann quadrature about the slab centre, a 100 nm origin shift changes
eps_zz by 1.8e-13 on the 6 nm slab (library: 1.2e-3 to 1.3e-3; the shifts of the thicker slabs not
re-measured on the generic fixture).

**O4 and N1: one lattice convention.** lumenairy reads `eps_cell[j]` at the NODE x_j = j P/S; the bridge
paints rasters at cell centres (j + 1/2) P/S (`optics/rasterize.cell_axes`), and analytic shapes sit in
the true frame. A stack mixing an analytic layer with a raster layer is therefore misregistered by P/(2S),
and "edges on pixel boundaries" means different lattices under the two conventions. MEASURED (fixture A,
'li', N = 50, the lumenairy 5.50.0 raster route, manual raster): S = 3200 centre-sampled realises 1200 px =
150.000 nm, r = 0.277415-0.022676i; node-sampled, the edges land exactly on nodes and the raster realises
1201 px = 150.125 nm, r = 0.265818-0.000855i. An area-weighted raster (each pixel the covered fraction of
its own [x_j - P/2S, x_j + P/2S] cell) realises the drawn width exactly for every S and every edge position.

**O3: two latent defaults outside the audit.** (a) `_add_shapes_layer` omits the `formulation` keyword
for Laurent and `LumenairyStackSolver.solve` never passes it, so both inherit lumenairy's default; L_next
flips that default to 'auto', which would silently change DynaMeta results. (b)
`make_lumenairy_rcwa_solver` has no `cache_fingerprint`; `OpticalSolverCache` keys two RCWA solvers with
different settings identically and would also serve v0.15.1 results after the default changes.

**N2 scope.** `slice_eps_field` is called at six sites (PMM, RCWA, Berreman, PMM2D, TMM, FDTD seam). The
pipeline path cannot hit the defect: `core/bridge.assemble_eps` affinely remaps the carrier z axis onto
the region's z interval. The defect is on the bring-your-own seam, and also covers a laterally structured
field with a single z sample, which slices to ZERO slabs (the layer vanishes) in the structured branch.

**Q6 interface condition.** Linearized hydrodynamics per piecewise-uniform sub-layer:
(d/dt + gamma) J = (e^2 n0/m) E + e beta^2 grad n1 (n1 the induced density). Integrating the normal
component across an internal interface, where E_z and J_z stay finite, gives continuity of beta^2 n1
(equivalently beta^2 rho); together with continuity of J_z, E_x and H_y. The audit's "beta^2 div J /
wp^2" form is the P-form divided by wp^2 and is not the right jump condition when n0 (hence wp and beta)
differs between sub-layers, which is exactly the graded case.

---

## 3. Shared rules for every package

1. **Gates.** Independent oracle per gate (closed form re-derived in the test, scipy quadrature written in
   the test, a converged reference from a different method). A gate that compares code with itself is
   labelled a consistency check. Tolerances come from a stated error budget. Every new gate is shown
   FAILING on v0.15.1 by a stated amount (put the number in the test docstring). At least one
   sign-sensitive gate per kernel with a sign or phase (cross-route agreement, or an odd statistic).
2. **No rerun to green.** A failure is a deterministic cause to find (BLAS build, state leak, load).
   Bitwise gates are confirmed on CI or a second machine, not on the main workstation alone.
3. **Warnings.** The test suite runs with `filterwarnings = ["error"]`. Every new warning uses
   `dynameta.warning_types.AdvisoryWarning` (advice) or `UnreliableResultWarning` (the returned number is
   suspect), with `stacklevel`, written as a literal `warnings.warn(` (the warning census matches that
   spelling). Each package lists the existing tests a new default-path warning reaches and updates them
   (`pytest.warns(...)` where the warning is the point, an explicit opt-out argument otherwise). Never add
   a blanket `filterwarnings` exemption.
4. **Behaviour changes.** Every changed default gets (a) a keyword that restores the old behaviour exactly,
   (b) a same-tree A/B test proving the restore is bit-identical to v0.15.1 (`np.array_equal` on the
   outputs), (c) the old -> new delta of every moved pin written into the package note, and (d) a line in
   the release note. Pins are re-recorded, never loosened.
5. **One kernel per formula.** The bulk Fermi level (Q3), the face flux (Q2/N4), the area-weighted
   raster (O4/N1), the lateral partition average (O1) and the slab-thickness check (N2) each get exactly
   one implementation, imported by every user; add a grep census test where the risk of copies is real
   (P6, P2).
6. **Census tests.** Run `tests/test_api_surface.py`, `tests/test_audit_*.py`, `tests/test_numerics.py`,
   `tests/test_ci_solver_coverage.py` and `tests/test_devsim_env.py` before handing in; engineers have
   skipped these before.
7. **DEVSIM.** New modules get DEVSIM via `dynameta.carriers.devsim_env.import_devsim()` (AST gate). Test
   files that need DEVSIM `importorskip("devsim")` BEFORE importing any `devsim_*` module. Every new DEVSIM
   test file is added to the `solvers-devsim` file list in `.github/workflows/ci.yml` (append at the end of
   the list; P15 resolves adjacent-line conflicts by union). Tests over ~60 s are `@pytest.mark.slow`.
   Two DEVSIM solves compared bitwise need `@pytest.mark.devsim_isolated`.
8. **Git hygiene.** One worktree per package, branch `wip/v016_<pkg>` from origin/main (2e1af87 or the
   later merged main). Never `git stash` (shared across worktrees). Never merge into a worktree while a
   test run is going there. Do not edit files you do not own; if you need a change there, write it in
   your package note as a request to the owner package.
9. **Notes.** Each package writes `docs/audit/<date>-v0.16-<pkg-name>.md` (new file, so no conflicts):
   what changed, measured numbers, moved pins with old/new values, plan statements found wrong. P15
   compiles the release note from them. No README edits outside P15.
10. **Machine etiquette.** One pytest process at a time on a Windows box; BLAS threads pinned (2) for
    measurements; check for orphaned python processes and free RAM before diagnosing silent crashes.

---

## 4. Per-finding assessment

### O1. PMM bridge refuses a laterally structured tunable layer

*Check.* Refusal at `pmm_backend.py:205-211` confirmed by reading. lumenairy's `PMMStack.add_layer` accepts
segments whose eps is a scalar or a (3, 3) tensor (read in the installed 5.50.0), so a laterally varying
layer can be represented as tensor segments.

*Assessment of the fix.* The audit's arithmetic mean per interval is the Laurent-type average; it
converges as the partition is refined but at first order near steep variation. For an x-varying film the
leading-order effective medium of a thin interval is the laminate tensor: eps_xx = harmonic mean (E_x is
normal to the variation, D_x continuous), eps_yy = eps_zz = arithmetic mean. That is exact for a
piecewise-constant profile resolved by the partition and second-order otherwise. A fixed "k sub-intervals
near each edge" partition has no error control; use a tolerance-driven bisection.

*Verdict.* IMPLEMENT WITH CHANGES (P3). Default partition: start from the union of every layer's
inclusion edges (and x = 0, P), bisect any interval whose relative eps spread (max |eps - mean| / |mean|
over the interval) exceeds `partition_rtol` (default 1e-2), stop at `max_segments` (default 32) or at a
minimum width of P/4096; one partition per design layer, shared by all its z slabs (fewer walls on the
PMM union grid). Report the largest remaining spread; AdvisoryWarning when it exceeds the tolerance at
the cap.

### O2. `elements_per_region` not exposed

*Check.* True that the bridge cannot pass `elements_per_region`, `grade`, `min_feature`. The convergence
claim does not hold (audit; consistent with the sister audit G-1: degree alone beats Chebyshev h at equal
DOF). Additionally the bridge passes `n_orders=` to `PMMStack`, which is only the far-field order count
(sister audit G-7); L_next will warn on that alias.

*Verdict.* IMPLEMENT WITH CHANGES (P3): `pmm_kwargs: dict | None = None` forwarded verbatim (refuse keys
the bridge sets itself: `degree`, `far_field_orders`, `n_orders`, `n_substrate`, `n_superstrate`,
`layer_grids`), included in `cache_fingerprint`; call `PMMStack(..., far_field_orders=n_orders)` (the
keyword exists far below the 5.22 floor; bit-identical, since `n_orders` is an alias); docstring: accuracy
is set by `degree` (and, on L_next, geometric grading), not `n_orders`.

### O3. RCWA default factorization

*Check.* MEASURED: N = 50 default r = -0.142794+0.150175i (abs(r - r_ref) = 0.496), 'li'
0.303405-0.017257i (0.0469; rasterized as on 5.50.0, default S = 400; AUDIT 0.0704 with analytic shapes on
5.50.2 -- the raster is closer to r_ref here only because its lattice error happens to point toward it).
Matches the audit and the sister audit (G-3), which also measured that Li reduces to Laurent for TE in 1-D
to 4.5e-13.

*Assessment of the fix.* The 1-D rule is right. On a 2-D stack "always 'li'" over-reaches: lumenairy's Li
on analytic shapes covers axis-aligned rectangles only (>= 5.50.1); a disk or ellipse would then be
rasterized, trading exact geometry for a better factorization with no measurement that it is a net gain.
Two latent defaults (section 2) must be closed in the same change.

*Verdict.* IMPLEMENT WITH CHANGES (P2). Rule for `formulation='auto'` (resolved per stack, recorded on
the result):
- a layer is "hard" if it is patterned (inclusions, or a laterally structured gridded field) and contains
  Re(eps) < 0 anywhere or max|eps|/min|eps| > 10 (lumenairy G-3 uses the same thresholds);
- 1-D stack: 'li' if any layer is hard and the incident field has an E_x component (polarization 'x' or
  'p', or any conical incidence); else 'laurent';
- 2-D stack: 'li' if any layer is hard and every hard analytic layer is rectangles-only, or the hard
  layers are rasterized anyway; 'laurent' for hard layers made of curved analytic shapes, with an
  AdvisoryWarning naming `formulation='li'` (and the engineer measures one disk cell both ways against a
  high-N or FEM reference and records it in the note; if 'li' + area raster wins, extend the rule);
- an explicit 'laurent' on a stack the rule would send to 'li' emits one AdvisoryWarning per solver.

### O4. Raster moves strip edges

*Check.* MEASURED: on the generic cell the default S = 400 realises 150 px = 150.000 nm (cell-centred; both
edges happen to fall on pixel boundaries at even N), r = 0.303405-0.017257i; the node-sampled raster at the
same S realises 151 px = 151.000 nm, r = 0.237213+0.081062i; at N = 51 the default S = 408 misses the edges
(150.980 nm, audit O4). Remains on 5.50.0 and on every raster route on any version (audit list confirmed:
`inclusion_model='raster'`, polygons, overlapping inclusions, tensor backgrounds, gridded fields,
absorption below 5.50.1).

*Assessment of the fix.* (a) Edge-aligned S works only for one sampling convention and only when all edges
share a common lattice below the cap; (b) area weighting works for every geometry; lumenairy has the
inverse-rule companion `add_layer(eps_cell_normal=(exx, eyy))` from 5.31.0 (CHANGELOG 5.31.0), so 'li'
rasters can carry the harmonic average too.

*Verdict.* IMPLEMENT WITH CHANGES (P2): new `raster_sampling='area'` default: pixel j represents the cell
[x_j - P/(2S), x_j + P/(2S)] around lumenairy's node x_j = j P/S; eps_cell = arithmetic coverage average;
for 'li' also exx = harmonic along x then arithmetic along y, eyy = harmonic along y then arithmetic along
x (exact for rectangles; curved shapes by 16 x 16 sub-sampling per boundary pixel). Below 5.31 the
companion is omitted (arithmetic only; width still exact). `raster_sampling='point'` reproduces v0.15.1
bit for bit (cell-centred point samples).

### O5. Not reproduced. REJECT (not library code; the real defect is N2).

### C1. 32-point uniform carrier grid, also on the pipeline path

*Check.* Code confirmed: `solve(grid_n_x=256, grid_n_z=32)`; `resample_to_grid` builds `linspace` axes and
interpolates on a Delaunay re-triangulation; `assemble_eps` reads `grid_fields` only; `run_pipeline` calls
`carrier.solve(bp)`. The DEVSIM 2-D mesh is a tensor grid of global mesh lines, so native node axes exist.

*Assessment of the fix.* "Native" in x would make the EpsField x axis non-uniform, which the RCWA raster
(index = position) and the FEM `VoxelCoefficient` (only start/end are passed,
`core/eps_field.voxel_bounds_u`) both misread silently. The FEM path cannot use 0.0125 nm z data anyway
(and a 400 x 256 x 256 voxel grid is 0.4 GB).

*Verdict.* IMPLEMENT WITH CHANGES (P4): `grid_n_z=None` = native z (the region's unique node y), x uniform
(`grid_n_x`), filled by separable 1-D linear interpolation along x on each native z line (exact at nodes;
falls back to the current path with an AdvisoryWarning when the region nodes are not a tensor grid);
`EpsField.voxel_bounds_u` raises on a non-uniform axis (relative spacing spread > 1e-9); `run_pipeline`
gains `carrier_solve_kwargs: dict | None = None` and, when it is None, passes `grid_n_z=None` only if the
carrier is a `LayeredDevsimBuilder` and the optical solver carries `native_carrier_grid = True`. The
audit's "warn when a uniform grid is coarser than 2x the mesh" is dropped on the FEM path (it would fire on
every default run and the FEM cannot use the data); the native path makes it moot on the layered path.

### C2. Equilibrium edge ground pins the edge column

*Check.* Code confirmed (`devsim_layered.py:668-675`, `physics_equilibrium.setup_contact` Potential
Dirichlet). The closure n = N_c F_1/2((Potential - Phi_c0)/V_t) already places the Fermi level at 0, so
the contact adds only a Dirichlet column. The comment about "~2 box-corner nodes" is wrong (AUDIT
deviation 0.696 at the contact, 0.163 at 4 nm).

*Assessment of the fix.* "Drop the Dirichlet" is exact only for 0 V. A lateral contact at voltage V sets
the region's Fermi level to -qV, which in equilibrium mode is exactly n = N_c F_1/2((Potential - V -
Phi_c0)/V_t) everywhere in the region.

*Verdict.* IMPLEMENT WITH CHANGES (P5 primitive, P11 wiring): for an equilibrium region, an edge electrode
creates no DEVSIM contact and sets a region parameter `EF_ref_V` = its voltage (updated per bias); two edge
electrodes on one equilibrium region at different voltages raise (no current flows in equilibrium);
`Electrode(pin_potential=True)` keeps today's contact; the comment is corrected. DD and bipolar regions are
unchanged (their full-edge ohmic grounds are real contacts).

### C3. `rel_tol <= 1e-7` cannot converge on equilibrium devices. IMPLEMENT (P4, docs): `solve()` docstring
states that a Potential-only device sets tight convergence with `abs_tol` in volts; no automatic mapping.

### C4. Default mesh grading. See section 2. IMPLEMENT WITH CHANGES (P4).

### C5. Electrostatics with a constant DOS mass

*Check.* Confirmed at `devsim_layered.py:703` and also `devsim_3d.py:559, 626` (not in the audit). The
audit's Kane table was not re-run; the formula was re-derived (the Kane density at band bending psi with
the same DOS mass at n_bg) and is consistent.

*Assessment of the fix.* A DEVSIM node model cannot do quadrature. Two routes: (a) the custom-equation
hook (`carriers/custom_terms.py`, Python-assembled with an exact Jacobian) with a tabulated spline of
n(eta); (b) a closed form: for the Kane band, sqrt(x(1 + bx))(1 + 2bx) = x^(1/2)(1 + 5/2 bx + 7/8 b^2 x^2 +
O(b^3 x^3)), b = alpha kT, so n = N_c [F_1/2(eta) + (15/4) b F_3/2(eta) + (105/32) b^2 F_5/2(eta) + ...]
with normalized complete Fermi integrals, each available in Aymerich-Humet-type closed form. (b) keeps the
equation in DEVSIM's own Newton (quadratic convergence, extended precision works) and is the recommended
route; the truncation term must be bounded where b eta is large (P13 measures it against quadrature and
adds the third-order term if b eta > 0.3 makes it exceed 1e-3).

*Verdict.* IMPLEMENT WITH CHANGES: guard now (P5 helper `TransportModel.dos_mass_at_bg(where)`: evaluates
the callable on [0.1, 10] n_bg, AdvisoryWarning once per material when it varies by more than 1 %, returns
the value at n_bg; P11 calls it from both builders); full closure in P13 with `TransportModel.band =
"parabolic" | KaneBand(m0_kg, alpha_eV)`, shared with `KaneOpticalMass`; KaneBand on a DD or bipolar region
raises NotImplementedError in v1.

### C6. No work function, flat-band voltage or fixed charge for an equilibrium film

*Check.* `Electrode` has no work function; `InterfaceTraps.fixed_charge_C_m2` is refused unless one side is
bipolar (`physics_traps._semiconductor_side`, measured by the audit). Dielectric regions declare their
Poisson equation without a node model (`physics_equilibrium.setup_dielectric_region`).

*Verdict.* IMPLEMENT WITH CHANGES (P5 primitives, P11 wiring): sheet charge as a node term `ItArea-like
weight x Q/q` on the PotentialEquation of the chosen side (any semiconductor or dielectric region; a
dielectric region gains a node model `PotentialNodeCharge` only when a charge is installed, so devices
without charge keep their equation strings character for character); uniform bulk charge per dielectric
region the same way; `Electrode.work_function_eV` sets the gate contact's Dirichlet offset
-(phi_m - W_ref)/q with W_ref = chi + (E_c - E_F) of the region's frame reference (raise if `chi_eV` is
None, as the heterostack code does). The trap path's `fixed_charge_C_m2` keeps working on bipolar regions;
on other regions it is routed to the new sheet charge instead of raising.

### C7. Not a defect. IMPLEMENT (P4, docs only): `devsim_env` docstring notes that DEVSIM treats
`debug_level="error"` like "verbose".

### Q1. Warm start ignores the end potentials. Confirmed by reading (`schrodinger_poisson.py:333-335`,
Newton updates nodes 1..N-2 only). IMPLEMENT WITH CHANGES (P6): after building phi from `U_init_J`, add the
linear ramp `linspace(phi_left - phi[0], phi_right - phi[-1], N)`; the bulk-buffer branch is covered by
its recursion; Neumann-body branch identical (Poisson stays Dirichlet at both ends).

### Q2. Single-gate carrier. See section 2. IMPLEMENT WITH CHANGES (P12).

### Q3. SP bulk Fermi level. See section 2. IMPLEMENT WITH CHANGES (P6).

### Q4. Intersubband double count. See section 2. IMPLEMENT WITH CHANGES (P7).

### Q5. Frozen-potential DG on a thin film

*Check.* Code confirmed (one Dirichlet wall, the other end pinned to the classical value; uniform rescale
under `conserve_charge`). With gamma = 1, L_q = hbar sqrt(gamma/(6 m kT)) = 1.28 nm for m = 0.30 m_e,
wider than the classical accumulation layer, which is why the peak disappears. The module already states
the closure is Boltzmann and gamma is a fit.

*Verdict.* IMPLEMENT WITH CHANGES (P8): `hard_wall="both"`; UnreliableResultWarning when the profile span
is below 10 L_q or when the frozen-potential deficit exceeds 10 % of |classical excess charge|; an
AdvisoryWarning when `conserve_charge` rescales by more than 10 %; docstring points to
`SchrodingerPoissonCarrier` / the dual-gate solver for degenerate thin films. DEFER the in-Newton DG for
equilibrium films (needs an FD-consistent closure for degenerate films and a calibration against the
Schrodinger-Poisson oracle; v0.15 notes record that gamma on degenerate profiles is not validated): ship as
a documented limitation.

### Q6. Hydrodynamic walls between sub-layers. See section 2. IMPLEMENT WITH CHANGES (P9).

### Q7. No structured nonlocal/quantum cell; no 2-D TM cross-check

*Check.* Confirmed by reading the cited sources (`hydro_fem.hydro_layered_1d`, `scattering_2d`,
`solve_fem` 3-D only, `solve_fdtd_2d` TE only).

*Verdict.* Split. Q7.1 IMPLEMENT (P10): a local 2-D periodic TM FEM (H_y in H1, x-periodic, PML in z,
tensor eps (eps_xx(x, z), eps_zz(x, z)) as a coefficient). It is the independent referee the strip cell
needs (PMM, RCWA and this FEM share no code). Q7.2 DEFER: the nonlocal extension with graded wp^2 and beta
in a structured periodic cell is 2-3 weeks of research (stabilization, interface conditions as in Q6, a
convergence study) -> documented limitation in `optics/fem_2d_tm.py` and `optics/hydro_fem.py` docstrings,
and `fem_2d_tm` refuses hydrodynamic material input with a message naming the gap. Q7.3 REJECT as
specified: a local eps_zz(z) from |psi(z)|^2 weighting is not unique (different weightings give
different profiles with the same slab average) and has no derivation; the intersubband response is
nonlocal along z, and the uniform-slab (3, 3) tensor the module returns is the standard local form for a
film that is one quantum system. Revisit together with Q7.2.

### N1. RCWA bridge ignores EpsField x coordinates. Confirmed by reading (`rcwa_backend.py:395-417`).
IMPLEMENT WITH CHANGES (P2): resample by coordinates onto the node lattice x_j = j P/S, S >= max(the
default raster lattice for N, 4N + 1), dropping a duplicated periodic end point; the resample integrates
the piecewise-linear interpolant over each pixel cell (the same area rule as O4), so gridded fields and
inclusion rasters share one convention and agree with analytic shapes.

### N2. No z-extent check. MEASURED: nm-axis field r = -0.234953-0.460540i, metre-axis field
r = -0.849411+0.035236i with no warning. IMPLEMENT WITH CHANGES (P1 helper, call sites P1/P2/P3).

### N3. Inconsistent dipole quadrature. MEASURED (section 2). IMPLEMENT (P7).

### N4. Depletion sign error. MEASURED (section 2). IMPLEMENT (P6) + decision 9.

---

## 5. Lumenairy dependency and version floors

| DynaMeta change | Needs L_next? | Notes |
|---|---|---|
| O3 auto rule, explicit `formulation` everywhere | no | 'laurent' with `shapes=` is accepted on every version >= 5.22 (only 'li' with shapes needs 5.50.1); making it explicit is REQUIRED before allowing L_next, whose 'auto' default would otherwise change DynaMeta results silently |
| O3 analytic 'li' on shapes | no | existing feature floor `rcwa_shapes_li` 5.50.1; below it the area raster (O4) carries 'li' |
| O4 harmonic companion | no | new feature floor `rcwa_eps_cell_normal` = 5.31.0 (`add_layer(eps_cell_normal=)`, CHANGELOG 5.31.0) |
| O2 `far_field_orders=` | no | keyword predates the 5.22 floor; required before L_next (G-7 alias warning would turn every PMM test red) |
| O2 `pmm_kwargs={'grade': 'geometric', ...}` | yes, for that value only | pass-through needs no gate; P14 adds feature floor `pmm_geometric_grade` = L_next and a bridge docstring recommendation |
| Duplicate Laurent warning | yes | L_next G-3 may warn on an explicit Laurent TM-metal layer; P14 adds a message-scoped filter around the lumenairy call where DynaMeta has already emitted its own advisory (the exact message is only known after L_next) |
| G-5 `rcwa_convergence` | no | DynaMeta does not call it |
| G-2 uniform-layer speed-up | no | transparent; re-time `validation/lumenairy_pmm_bridge.py` |

**Minimum Lumenairy after this plan.** Keep `VERSION_FLOOR = (5, 22, 0)` and `lumenairy>=5.22` in
`pyproject.toml` (the CI floor leg keeps testing it); every younger capability stays behind
`FEATURE_FLOORS`. After P14: `VERSION_VERIFIED_MAX = (L_next major, minor)` once `validation/lumenairy_*.py`
pass on L_next. Recommend in README: lumenairy >= 5.50.2 for the analytic Li route (5.50.1 has the same
library code; 5.50.2 changed records only). If the owner prefers a single supported route, raising
`VERSION_FLOOR` to 5.50.1 would delete the 'li'-rasterize branch and two feature floors; that is decision
10's alternative and is not recommended for 0.16.0 (it drops the floor leg's coverage).

---

## 6. Package briefs

Each brief is self-contained. "Bit-identical" means `np.array_equal` (or equal strings for DEVSIM model
definitions) against v0.15.1 in the same process. All paths are relative to the repository root.

### P1. Slab-thickness guard (N2) -- wave 1, merge first

**Goal.** A gridded EpsField whose z extent does not match its layer can no longer silently change the
optical stack.

**Files.** `core/layered.py` (new function), call sites in `optics/tmm_reference.py` (line ~323),
`optics/fdtd_seam.py` (~254), `optics/lumenairy_bridge/berreman_backend.py` (~149),
`optics/lumenairy_bridge/pmm2d_backend.py` (~828). P2 and P3 switch their own call sites.

**Function (exact signature; P2/P3 code against it):**
```
def slice_layer_eps_field(eps_field, layer, metres_per_unit, *, n_slices=None, rtol=1e-6):
    """slice_eps_field + a check that the slab thicknesses sum to layer.thickness_m."""
```
- calls `slice_eps_field`; if the list is empty, or `abs(sum(t) - L) > rtol * L`, raise ValueError naming
  the layer, both thicknesses, their ratio, and the likely cause (ratio near 1e-9: axes in metres where
  nm were expected; near 1e9: the reverse; a single z sample: the field has no thickness);
- also raise for a laterally structured field with one z sample (today it slices to zero slabs).

**Also in P1:** set `_solve.native_carrier_grid = True` on `make_layered_tmm_solver` and the Berreman
solver closure (contract with P4: the attribute name is `native_carrier_grid`).

**Tests** (`tests/test_slab_thickness_guard.py`, no DEVSIM): the audit's N2 metre-axis field raises at
every one of the six bridges (PMM/RCWA cases are added by P3/P2 in their own test files); the nm-axis field
gives bit-identical r to v0.15.1 through TMM and Berreman; a structured one-z-sample field raises;
`n_slices=7` staircase passes (its total equals the span). Show the metre case passing silently on
v0.15.1 in the docstring (r -0.849411+0.035236i vs -0.234953-0.460540i, PMM degree 24).

**Bit-identical:** every correctly-dimensioned field. **May move:** nothing.

**Size:** 1 day.

### P2. RCWA bridge (O3, O4, N1, N2 call site)

**Goal.** The RCWA bridge's defaults give the converged answer on TM metal cells; rasters realise the
drawn geometry; gridded fields are registered by their coordinates; the solver is cache-safe.

**Files.** `optics/lumenairy_bridge/rcwa_backend.py`, `optics/lumenairy_bridge/_shapes.py`, new
`optics/lumenairy_bridge/_raster.py`, `optics/lumenairy_bridge/_common.py` (only the new FEATURE_FLOORS
row `rcwa_eps_cell_normal: ((5, 31, 0), ...)` and its comment line). Do not change `optics/rasterize.py`
(shared with FDTD).

**Work.**
1. `formulation='auto'` default in `design_to_rcwa_stack` and `make_lumenairy_rcwa_solver`, rule in
   section 4 (O3). One function `_resolve_formulation(design, eps_by_region, pol_row, conical, is_2d)`
   returns ('li' | 'laurent', reason); the result records it (`OpticalResult` has no free-form field: put
   it on the stack object as `stack.dynameta_formulation` and in the solver's `last_formulation`
   attribute; do not change `OpticalResult`).
2. Always pass `formulation=` explicitly: `_add_shapes_layer` (drop the "omit for Laurent" branch),
   `LumenairyStackSolver` (new `formulation='auto'` keyword resolved from the slabs: hard = any eps_cell or
   shapes slab with Re(eps) < 0 or contrast > 10).
3. AdvisoryWarning once per solver when an explicit 'laurent' meets a hard stack with E_x across walls,
   and for curved analytic hard layers on 2-D stacks under 'auto'.
4. `_raster.py`: `area_weighted_cell(layer, S_x, S_y, lambda_m, materials, eps_by_region, px, py,
   companion: bool) -> (eps_cell, (exx, eyy) | None)` on the node lattice x_j = j P/S (pixel cell
   [x_j - P/2S, x_j + P/2S], periodic wrap). Rectangles: exact coverage from 1-D overlaps; circles,
   ellipses, polygons: 16 x 16 sub-samples per pixel only for pixels the shape boundary crosses. Overlap
   priority as in `optics/rasterize` (later inclusion paints over earlier).
   `periodic_layer_eps_cell` in `_shapes.py` stays for `raster_sampling='point'`.
5. New keyword `raster_sampling='area' | 'point'` on both entry points; 'point' reproduces v0.15.1 bit
   for bit. Companion only for 'li' and only when `lumenairy_has('rcwa_eps_cell_normal')`.
6. N1: gridded laterally structured fields: drop a duplicated periodic end sample (`x[-1] - x[0] == P`
   within 1e-9 P and values equal within 1e-12), then area-resample the piecewise-linear interpolant onto
   the same node lattice; S = max(`_default_raster_samples(N)`, `_min_samples(N)`, len(axis) - 1).
7. N2: replace `slice_eps_field` by `slice_layer_eps_field` (P1).
8. `_solve.cache_fingerprint` = "rcwa(n_orders=..,n_orders_y=..,n_slices=..,cell_samples=..,
   formulation=..,inclusion_model=..,raster_sampling=..,absorption=..,stabilize=..,rule=1)" (the `rule`
   integer is bumped whenever the auto rule changes); `_solve.native_carrier_grid = True`.
9. Docstrings: module "FORMULATION RULE" and "RASTER LATTICE" paragraphs rewritten; state the node-lattice
   convention once.

**Tests** (extend `tests/test_lumenairy_bridge.py`; new `tests/test_rcwa_auto_formulation.py`,
`tests/test_rcwa_area_raster.py`):
- fixture A, default: abs(r - r_ref) < 0.02, r_ref = 0.3352997+0.0171262i, at N = 200 (v0.15.1: 0.0226;
  analytic 'li' 0.0089 -- the generic cell needs N = 200 for this bar: at N = 100 'li' is 0.0246 off and
  v0.15.1 0.169, AUDIT; record the value on 5.50.0 (raster 'li') and on >= 5.50.1 (analytic 'li')); the
  default equals the explicit 'li' call bit for bit;
- TE (polarization 'y') fixture A and a dielectric low-contrast grating: bit-identical to v0.15.1;
- explicit 'laurent' on fixture A in TM warns once per solver, not per wavelength;
- area raster: the realised width (sum of coverage x P/S) of every rectangle equals the drawn width to
  1e-12 relative for S in {128, 400, 2960} and edges at arbitrary offsets; fixture A 'li' N = 50 on 5.50.0
  with the default lattice within 1e-3 of the S = 3200 area raster (record both); analytic vs area raster
  converge as O(S^-2) on the existing disk fixture (tests/test_lumenairy_shapes_sem_wrap.py numbers);
- N1: one lamellar field on three x grids (endpoint-inclusive 256, node 256, cell-centred 3200) gives r
  within 2e-3 of each other at N = 50 (v0.15.1 spread: 8.9e-3, AUDIT); N = 100 no longer raises;
- `raster_sampling='point'` + `cell_samples=128` + `inclusion_model='raster'` + `formulation='laurent'`:
  bit-identical to v0.15.1 (the old pre-0.13 promise);
- cache: two solvers differing only in `formulation` produce different fingerprints.

**Existing pins that move:** every TM metal cell through the default RCWA solver; every raster layer under
the default (`validation/lumenairy_rcwa_bridge.py`, `validation/lumenairy_pmm_bridge.py` GATE B,
`tests/test_lumenairy_shapes_sem_wrap.py`, `tests/test_lumenairy_bridge.py`,
`tests/test_lumenairy_bridge_verify_550.py`, `examples/lumenairy_gated_grating.py`). Tests that assert
pre-0.13 or v0.15.1 numbers switch to `raster_sampling='point'`/`formulation='laurent'` explicitly.
`optics/lumenairy_bridge/rcwa_design.py` (differentiable forwards, default 'laurent') is out of scope:
traced eps cannot be inspected for metal; add one docstring sentence pointing to `formulation='li'` for
TM metal cells and leave the default (decision recorded in the note).

**Size:** 5-6 days. **Depends on:** P1's helper (rebase after P1 merges; signature fixed above).

### P3. PMM bridge (O1, O2, N2 call site)

**Goal.** PMM can referee a biased (laterally structured) cell; its knobs are reachable and honestly
documented.

**Files.** `optics/lumenairy_bridge/pmm_backend.py`, new `optics/lumenairy_bridge/_lateral_partition.py`.

**Work.**
1. `_lateral_partition.py`: `partition_layer(ef, period_m, seed_edges, *, rtol=1e-2, max_segments=32,
   min_width_frac=1/4096) -> (edges_frac, report)` and `laminate_segments(ef_slab_profile, x_m, edges_frac,
   mode) -> [(w, eps)]`. Integrate the piecewise-linear interpolant of eps (arithmetic) and of 1/eps
   (harmonic) exactly over each interval using the field's own x coordinates (duplicate periodic end
   dropped). `mode='laminate'` (default): `diag(harm, arith, arith)` tensors; `mode='arithmetic'`:
   scalar means (for users who want scalar segments).
2. `design_to_pmm_stack(..., lateral_partition='auto' | sequence_of_edge_fractions | None,
   partition_rtol=1e-2, max_segments=32, lateral_average='laminate')`: 'auto' seeds with every layer's
   inclusion edges plus 0 and P; None keeps today's refusal. Tensor gridded fields stay refused.
   AdvisoryWarning when the spread at the cap exceeds `partition_rtol`; `stack.dynameta_partition_report`
   = {layer: (n_segments, max_spread)}.
3. `pmm_kwargs` pass-through (O2) and `far_field_orders=` (section 4).
4. N2 call site -> `slice_layer_eps_field`.
5. `cache_fingerprint` gains `pmm_kwargs` (sorted repr), `lateral_partition`, `partition_rtol`,
   `max_segments`, `lateral_average`; `_solve.native_carrier_grid = True`.

**Tests** (`tests/test_pmm_lateral_partition.py`):
- a two-value lamellar field whose edges coincide with the strip edges reproduces the hand-built segment
  stack to 1e-12 at equal degree (AUDIT hand-built value at epr 6, degree 32: 0.068069-0.679514i; use the
  bridge's own degree for the comparison);
- the laminate average of a piecewise-constant field on a partition that resolves its steps is exact
  (tensor entries to 1e-15);
- a field with a linear ramp across each strip edge converges with `partition_rtol` (1e-1, 3e-2, 1e-2,
  3e-3) toward the RCWA 'li' area-raster result at high N, and the laminate average converges faster than
  'arithmetic' (record both sequences);
- laterally uniform and z-graded fields: bit-identical to v0.15.1 (degree 24: -0.2350-0.4605i, AUDIT);
- `pmm_kwargs={'elements_per_region': 3}` equals a direct `PMMStack(..., elements_per_region=3)` to
  1e-14; a forbidden key raises.

**Bit-identical:** every stack v0.15.1 accepted. **May move:** nothing (new capability).

**Size:** 3-4 days. **Depends on:** P1 helper.

### P4. Carrier grid and mesh (C1, C3, C4, C7 docs)

**Goal.** The layered optical path consumes the carrier profile at its native resolution; the default
DEVSIM mesh resolves accumulation and depletion layers.

**Files.** `carriers/devsim_layered.py` (only `solve`, `_to_carrier_field`, `_build_mesh` y-line block
~596-614, docstrings), `pipeline.py`, `core/resample.py`, `core/eps_field.py`, `geometry/specs.py`,
`carriers/dc_solve.py` (docstring), `carriers/devsim_env.py` (docstring).

**Work.**
1. C1 native z: `solve(..., grid_n_z: Optional[int] = 32)`; None -> native z. New
   `core/resample.resample_tensor_grid(x_nodes, y_nodes, fields, grid_n_x) -> axes, fields` (exact when the
   node set is a tensor grid; detect with `len(unique x) * len(unique y) == N` after rounding to 1e-15 m).
   Fallback: the old path plus AdvisoryWarning.
2. `EpsField.voxel_bounds_u` raises ValueError on a non-uniform axis, naming the region and the fix
   (integer `grid_n_z`).
3. `run_pipeline(..., carrier_solve_kwargs=None)`: if given, `carrier.solve(bp, **carrier_solve_kwargs)`;
   if None and `isinstance(carrier, LayeredDevsimBuilder)` and `getattr(optical_solver,
   'native_carrier_grid', False)`, call `carrier.solve(bp, grid_n_z=None)`; else unchanged.
4. C4 grading: new `Mesh2DSpec` fields `semiconductor_refinement: str = "screening"` ("legacy" = today's
   lines exactly), `screening_zone_factor: float = 4.0`, `screening_ps_factor: float = 1/20`. For each
   semiconductor layer compute the screening length at n_bg, L_s^2 = eps_s eps0 / (q^2 dn/dE_F) with
   dn/dE_F = N_c F_{-1/2}(eta_bg)/kT (normalized Fermi integral; the same N_c and eta_bg the builder's
   closure uses; it reduces to the Debye length when non-degenerate and to Thomas-Fermi when
   degenerate), and place,
   at each semiconductor interface: `interface_ps = min(spec.interface_ps_m, L_s/40)`, a line at
   `zone = min(screening_zone_factor * L_s, 0.45 t)` with spacing `max(L_s * screening_ps_factor,
   interface_ps)`, then the existing coarse spacing beyond. Dielectric-only interfaces keep today's lines.
   Measure node counts on every validation device built through `LayeredDevsimBuilder` before/after and
   record them. (For the gated-film material, L_s = 0.777 nm, giving roughly the 0.025 / 0.1 / 2.25 nm row
   of section 2.)
5. Optional post-solve resolution check, `extras['interface_resolution']` = max over semiconductor
   interface edges of |ln(n_k+1 / n_k)| (diagnostic only, no warning).
6. C3 and C7 docstrings.

**Tests** (`tests/test_carrier_native_grid.py`, `tests/test_mesh_screening_grading.py`, DEVSIM, add to the
solvers-devsim leg):
- fixture B fine mesh, +5 V, no ground: sheet density from native `grid_fields` equals the raw-node
  trapezoid to 1e-12 relative (v0.15.1 32-point grid: +7.4e-3, AUDIT);
- `grid_n_z=32` explicit: bit-identical to v0.15.1;
- `EpsField` with a non-uniform z axis: `voxel_bounds_u` raises; uniform: unchanged;
- `run_pipeline` with a TMM solver and a stub optical builder passes `grid_n_z=None`; with the FEM default
  it does not (spy on `carrier.solve`);
- default `Mesh2DSpec`: fixture B (6 nm) and the 20 nm variant at +5 V give n(top) within 1e-3 of the fine
  reference 9.85913e26 (v0.15.1: -1.8e-2), at -3 V within 1e-2 of 2.069e25 (v0.15.1: +1.1e-1);
  `semiconductor_refinement="legacy"` reproduces the v0.15.1 mesh line list exactly. (P4 replaced the fine
  DEVSIM reference by an independent 1-D oracle, against which v0.15.1 is -1.88e-2 / +1.12e-1;
  `2026-11-14-v0.16-carrier-grid-mesh.md`.)

**Existing pins that move:** every test and validation that builds through `LayeredDevsimBuilder` with a
default `Mesh2DSpec` (grep `LayeredDevsimBuilder(` in tests/ and validation/; e.g. `tests/test_gated_builder.py`,
`validation/gated_dd_2d.py`, `validation/carriers_3d_*` are 3-D and do not move). Re-record with old/new
values in the note. The DEVSIM call-log digests hash model/equation definitions, not mesh lines; confirm
they do not move.

**Bit-identical:** explicit integer grids; `semiconductor_refinement="legacy"`; the FEM pipeline path.
**May move:** default-mesh DEVSIM results; layered-optics pipeline results (native z).

**Size:** 4-5 days.

### P5. Carrier physics primitives (C2, C6, C5 guard)

**Goal.** Primitives for a Fermi-level reference, sheet and bulk charges, gate work functions and the DOS
mass guard, each unit-tested on hand-built DEVSIM devices. No builder wiring (that is P11).

**Files.** `carriers/physics_equilibrium.py`, `carriers/physics_traps.py`, `geometry/electrode.py`,
`materials/transport_model.py`.

**Contract (exact names; P11 calls these):**
- `physics_equilibrium.setup_semiconductor_region(..., fermi_ref_param: Optional[str] = None)`: when
  given, eta = (Potential - <param> - Phi_c0)/V_t with the parameter created at 0.0; when None the model
  strings are character-for-character today's.
- `physics_equilibrium.install_sheet_charge(device, interface, region, Q_C_m2) -> str` (adds or extends
  the region's `PotentialNodeCharge`; for a dielectric region re-records the PotentialEquation with a node
  model; returns the node-model name).
- `physics_equilibrium.install_bulk_charge(device, region, rho_C_m3) -> str`.
- `physics_equilibrium.work_function_offset_V(work_function_eV, chi_eV, ec_minus_ef_eV) -> float`
  returning -(phi_m - W_s)/e in volts with W_s = chi + (E_c - E_F) in eV; the gate contact pins
  Potential = bias + offset (+ the existing frame offset). A larger metal work function therefore acts
  like a negative gate bias (electrons leave the semiconductor), which the MOS gate below checks.
- `geometry.electrode.Electrode`: new fields `work_function_eV: Optional[float] = None`,
  `pin_potential: bool = False` (validation: finite; `pin_potential` only meaningful for edge electrodes).
- `TransportModel.dos_mass_at_bg(where: str) -> float` (C5 guard: one AdvisoryWarning per material
  instance when the callable varies by > 1 % over [0.1, 10] n_bg; accepts scalar-returning callables).
- `physics_traps`: an `InterfaceTraps` with only `fixed_charge_C_m2` (no levels, `recombination=False`)
  on a non-bipolar interface routes to `install_sheet_charge` instead of raising.

**Tests** (`tests/test_equilibrium_charges.py`, DEVSIM, 1-D devices built directly):
- 1-D MOS (metal | oxide | equilibrium semiconductor): a sheet charge Q at the oxide/semiconductor
  interface shifts the gate voltage at fixed semiconductor charge by -Q/C_ox to 1e-9 relative (closed
  form); bulk oxide charge rho shifts it by -rho t_ox^2/(2 eps_ox) (centroid formula) to 1e-9;
- a work-function difference equals the same device biased by -delta phi/q, node values bit-identical;
- `fermi_ref_param` set to V: n equals the V = 0 solution with every potential shifted by V (to the solver
  tolerance), and model strings are unchanged when the parameter is not requested;
- `dos_mass_at_bg` warns for `lambda n: m0 (1 + 0.1 log(n/n_bg))`, silent for constants (including the
  scalar-returning lambdas used across tests/).

**Bit-identical:** every device that requests none of the new options (model strings and arrays).

**Size:** 3-4 days.

### P6. Schrodinger-Poisson core (Q1, Q3, N4)

**Goal.** Warm starts honour the boundary potentials; the bulk Fermi level is consistent with the model's
own band; depletion biases map correctly.

**Files.** `carriers/schrodinger_poisson.py`, `carriers/sp_carrier.py`.

**Work.**
1. Q1 ramp (section 4).
2. Q3: `SchrodingerPoisson1D.bulk_fermi_level(n_bg_m3, *, alpha_np_per_eV=0.0) -> float` [J]: solves
   n_sep(E_F) = n_bg with n_sep(E_F) = (1/pi) Int_0^inf dk_z pref0 [kT F_0(eta) + 2 a kT^2 F_1(eta)],
   eta = (E_F - hbar^2 k_z^2/(2 m))/kT, pref0 = g_s g_v m/(2 pi hbar^2) (scipy `quad` to kmax with
   E_F + 60 kT, `brentq` xtol 1e-28 J); reuse the module's `_fermi_log`/`_fd1`. `SchrodingerPoissonCarrier`
   uses it for every alpha (keyword `fermi_level="model"` default, `"bulk3d_T0"` = today's formula).
   Docstrings state that the z dispersion is parabolic.
3. N4: residual `psi + s * Q * n_exc / C_ox - Vg`; root by `brentq` on [0, |Vg|] with xtol 1e-7 V; keep
   the UnreliableResultWarning if the bracket check ever fails. Add `gate_map="flat_band_excess"` (the
   fixed current method) and, per decision 9, `gate_map="face_field"`: residual `psi_s + D_face(psi_s)/C_ox
   - Vg` with the conservative face flux `D_face = eps_s (phi_N - phi_{N-1})/h - q (h/2)(Nd_N - n_N)` (the
   discrete Gauss law of the solver's own Poisson stencil), root by `brentq` with symmetric bracket
   expansion. Put the face-flux formula in ONE function (P12 imports it).

**Tests** (extend `tests/test_schrodinger.py`; no DEVSIM):
- Q1: the audit reproduction gives `phi2[-1] == 1.0` exactly and n2 equal to the cold start n3 to 1e-6
  relative (v0.15.1: 0.388); a warm start whose ends already match is bit-identical to v0.15.1;
- Q3: alpha = 0, `bulk_fermi_level` reproduces N_c F_1/2(E_F/kT) = n_bg to 1e-10 with F_1/2 by
  independent scipy quadrature (MEASURED calibrated E_F 0.54370 eV); alpha = 0.4: 0.49236 eV (+32.31 meV
  vs the T = 0 Kane inversion); flat-band centre n/n_bg of a 40 nm slab (0.025 nm grid) within 1e-3 of 1
  at alpha 0 (MEASURED 1.00007) and within 2e-3 at alpha 0.4; self-consistent centre potential |phi| < 1 mV
  at 0 V for alpha 0.4 (v0.15.1: +32.3 mV, AUDIT);
- N4: for Vg in {-4, -2, -1, +1, +2} V, flat-band map: |psi_s + q N_exc/C_ox - Vg| < 1e-6 V; sign pin
  psi_s < 0 for Vg < 0 and monotone in Vg (MEASURED -1.141830 / -0.523020 / -0.250435 / +0.234345 /
  +0.457603 V, nz 481); face-field map: |psi_s + D_face/C_ox - Vg| < 1e-6 V and psi_s(+2 V) = 0.658 V
  +- 5 mV (0.658488 V with the conservative flux, as re-measured for `2026-11-14-v0.16-sp-core.md`).

**Existing pins that move:** `validation/sp_carrier.py`, `validation/sp_oxide_cap.py`,
`validation/sp_carrier_nonparabolic.py`, `validation/sp_self_consistent_nonparabolic.py`,
`validation/sp_neumann_body.py`, `validation/sp_open_body.py`, `validation/sp_per_column.py`,
`tests/test_schrodinger.py`: ~1 mV-level at alpha 0, tens of mV at alpha > 0, accumulation psi_s by the
bisection tolerance (+2 V: 0.45703 -> 0.457603), depletion entirely. Record each.

**Size:** 2 days.

### P7. Intersubband permittivity (Q4, N3)

**Files.** `core/effects/electroabsorption.py` (class `IntersubbandEffect`).

**Work.**
1. `z_response: str = "subbands"`: eps_zz = eps_inf + chi_ISB + chi_free, chi_free the Drude term of an
   optional `z_free_density_m3: float = 0.0` (a population NOT in the sub-band set; uses `m_opt_kg`,
   `gamma_intra_rad_s`). `"drude_plus_subbands"` keeps v0.15.1 bit for bit and emits a deprecation
   AdvisoryWarning (removal in 0.17).
2. N3: `Z = psi.T @ (psi * (z - z_c)[:, None]) * h`, z_c = midpoint of `z_m`, h the uniform spacing
   (raise if `z_m` is not uniform to rtol 1e-6); vectorize the pair sum.
3. f-sum diagnostic: ratio = sum_{i<j} (n_s,i - n_s,j) f_ij / sum n_s,i with f_ij = (2 m_sp w_ij/hbar)
   z_ij^2, where `m_sp` is a new optional `subband_mass_kg` (default `m_opt_kg`); stored as
   `self.last_fsum_ratio`; `fsum_check: str = "warn"` ("off" silences) -> AdvisoryWarning outside
   [0.9, 1.05] in "subbands" mode.
4. Docstring: the in-plane Drude uses m_opt; the z response uses the Schrodinger mass; they differ for a
   nonparabolic band, so the thick-film limit of eps_zz is the parabolic-mass plasma response.

**Tests** (`tests/test_effects.py` updated + new cases): f-sum of the flat-band 6 nm slab within 1e-3 of 1
(MEASURED 0.99967; v0.15.1 trapezoid quadrature: 1.0865); eps_zz invariant to a 100 nm origin shift to
1e-12 (MEASURED 1.8e-13); abs(eps_zz - eps_xx) for 10/20/40 nm slabs strictly decreasing with the
20 -> 40 nm ratio in [1.6, 2.4] (MEASURED 1.94) and below 0.025 at 40 nm (MEASURED 0.0133); the legacy mode
bit-identical to v0.15.1; the closed-form line-strength gates (canonical absorber, gain clamp) rewritten
on `eps_zz - eps_inf`.

**Existing pins that move:** `tests/test_effects.py` (single-band "reduces to Drude" becomes "reduces to
eps_inf along z"; pair tests compare against eps_inf), `validation/intersubband_eps_zz.py` GATES 1, 4, 5
(same change). Synthetic 2-3 state wells are incomplete sets: pass `fsum_check="off"` there or assert the
warning.

**Size:** 1.5-2 days.

### P8. Frozen-potential DG guard (Q5)

**Files.** `carriers/density_gradient.py`.

**Work.** `hard_wall="both"` (bc: v = 0 at both ends); the two guards and the rescale advisory (section 4);
docstring pointers; the in-Newton DG deferral recorded in the module header.

**Tests** (`tests/test_density_gradient.py`): `hard_wall="both"` on a flat profile is mirror-symmetric to
1e-10 and each half matches the one-wall closed-form deficit (GAMMA_WALL) to the existing tolerance; the
audit's +5 V 6 nm profile warns (UnreliableResultWarning); a 100 L_q flat profile does not;
`hard_wall='left'/'right'` bit-identical. Update `validation/density_gradient_dead_layer.py` /
`validation/dg_dd_in_newton.py` only if they trip the new warning (assert it there).

**Size:** 1 day.

### P9. Hydrodynamic layer runs (Q6)

**Files.** `optics/nonlocal_tmm.py`.

**Work.**
1. `HydroLayer(conductor_id: Optional[str] = None)`. Adjacent HydroLayers with the same non-None id form a
   run; None keeps today's behaviour (each layer walled), but two adjacent HydroLayers with
   `conductor_id=None` raise ValueError unless `allow_walls=True` is passed to `stack_rt` (explicit opt-in
   to the old physics).
2. For a run, propagate the 4-vector (E_x, H_y, J_z, beta^2 n1) through each sub-layer with its 4 x 4
   transfer matrix (2 transverse + 2 longitudinal waves, the same field expressions as
   `_hydro_char_matrix_p`), all four continuous at internal interfaces; impose J_z = 0 at the run's two
   outer faces and reduce to the 2 x 2 characteristic matrix. A run of one layer must use the existing
   `_hydro_char_matrix_p` (bit-identical).
3. s-pol and the local limit (beta = D = 0) unchanged.

**Tests** (`tests/test_nonlocal_tmm.py`): one 6 nm layer equals 2 x 3 nm and 10 x 0.6 nm runs to 1e-12
(v0.15.1: 4.2e-6 / 2.91e-3, AUDIT); a smooth density ramp n0(z) converges with slab count (8, 16, 32, 64)
at second order to an INDEPENDENT oracle written in the test: the 1-D p-pol linearized hydrodynamic
boundary-value problem with continuous n0(z), solved with `scipy.integrate.solve_bvp` and matched to the
cladding by the same r formula; the strongly nonlinear dependence of r on a thin top sub-layer's density
reappears (record the sequence that replaces the AUDIT's linear one); a run with a dielectric between two
hydro layers keeps walls on both sides of the dielectric.

**Size:** 3-4 days.

### P10. 2-D periodic TM FEM (Q7.1)

**Files.** new `optics/fem_2d_tm.py` (NGSolve, lazy import), tests and one validation script.

**Work.** `solve_fem_2d_tm(period_x_m, layers, *, lambda_m, theta_deg=0.0, n_super, n_sub,
eps_xz=None, order=4, maxh_m=..., corner_refinement=...)` -> `OpticalResult` (r referenced to the top
interface, exp(-i omega t), lab E_x tangential, matching the RCWA/PMM bridges). Weak form for H_y with
inverse tensor eps^-1 = diag(1/eps_xx, 1/eps_zz) (x-z plane), Bloch-periodic in x
(`ngsolve.Periodic` plus the Bloch phase for theta != 0), PML in z reusing the helpers of
`optics/solver.py` where they apply (import, do not copy), geometric corner refinement at metal corners.
`layers` accepts the same Design via a small adapter (`solve_design_2d_tm(design, eps_by_region, lambda_m,
...)`) for 1-D lamellar designs, including gridded fields (z-graded and laterally structured, via
NGSolve `VoxelCoefficient` on a uniform resample; non-uniform input is resampled here, not passed raw).
Refuse hydrodynamic input (Q7.2 deferral, message names the gap).

**Tests** (`tests/test_fem_2d_tm.py`, solvers leg; heavy cases `@slow`): laterally uniform stack vs TMM to
1e-6 in r (budget: PML reflection ~exp(-2 k d) with the 1000 nm PML default, plus discretization at
order 4); fixture A strip cell within 1e-3 of r_ref = 0.3352997+0.0171262i (AUDIT reference, good to ~1e-4);
a z-graded film vs PMM with the same slabs to 1e-3; convergence in order (3, 4, 5) recorded.
`validation/fem_2d_tm_strip.py` with the fixture A convergence table.

**Size:** 5-6 days.

### P11. Carrier wiring (C2, C6, C5 guard calls) -- wave 2

**Files.** `carriers/devsim_layered.py`, `carriers/devsim_3d.py`.

**Work.** Equilibrium edge electrodes (C2) per section 4 using P5's `fermi_ref_param` (parameter
`EF_ref_V`, set in `_drive` for that electrode); refuse two edge electrodes at different voltages on one
equilibrium region; `pin_potential=True` keeps the contact; fix the "~2 box-corner nodes" comment.
`LayeredDevsimBuilder(..., fixed_charge: dict | None = None, bulk_charge: dict | None = None)` keyed by
`(region_a, region_b)` interface or region name -> P5 installers; gate electrodes with `work_function_eV`
-> P5 offset added to the existing frame offset. Replace `tr.dos_mass_kg_of_n_m3(tr.n_bg_m3)` by
`tr.dos_mass_at_bg(...)` at `devsim_layered.py:703, 890` and `devsim_3d.py:559, 626`. 3-D builder:
refuse `work_function_eV`/`fixed_charge` with NotImplementedError naming the 2-D builder (or wire them if
the same primitives apply; measure, then decide in the note).

**Tests** (`tests/test_layered_edge_reference.py`, `tests/test_layered_charges.py`, DEVSIM): fixture B with
the edge ground at +5 V: every column equals the centre column to 1e-12 relative (v0.15.1: 0.696 at the
contact, AUDIT); edge electrode at V = 0.3 V equals the no-ground solve with both gate biases shifted by
-0.3 V (to solver tolerance); `pin_potential=True` reproduces v0.15.1 bit for bit
(`@pytest.mark.devsim_isolated`); a fixed sheet charge on the film/oxide interface shifts the gated sheet
density as the 1-D capacitor formula predicts (to 1e-6, mesh-limited) and is laterally uniform; the
audit's `InterfaceTraps(fixed_charge_C_m2=-1.6e-3)` call now builds.

**Size:** 2-3 days. **Depends on:** P4 (same file), P5 (primitives).

### P12. Dual-gate Schrodinger-Poisson (Q2) -- wave 2

**Files.** new `carriers/sp_dual_gate.py`; export in `carriers/__init__.py` (and the API census list).

**Design.** `DualGateSchrodingerPoisson(t_film_m, ox_bottom=(t_m, eps_r), ox_top=(t_m, eps_r), n_bg_m3,
m_eff_kg, eps_static, alpha_np_per_eV=0.0, nz=..., n_states=..., wf_offset_bottom_V=0.0,
wf_offset_top_V=0.0)`; `solve(bias)` with `bias.voltages = {"bottom": Vb, "top": Vt}` (names
configurable). Unknowns the face potentials (phi_b, phi_t) relative to the film Fermi level (fixed, as in
the DEVSIM equilibrium mode); inner solve `SchrodingerPoisson1D.solve_self_consistent` with Dirichlet
(phi_b, phi_t), hard walls on both faces, slab mode (`bound_tol=1e9`), warm-started (needs P6's Q1);
E_F from `bulk_fermi_level` (P6). Outer residuals (signs derived in section 2 of this plan):
R_b = C_b (V_b' - phi_b) + D_s(0), R_t = C_t (V_t' - phi_t) - D_s(t), with V' = V + work-function offset,
C = eps_ox eps0 / t_ox and D_s the conservative face fluxes (P6 function; at z = 0 the mirror formula).
Newton with a finite-difference 2 x 2 Jacobian (two extra inner solves per step) and backtracking; stop at
max |R| < 1e-9 C/m^2. Emit `CarrierField(ndim=3)` exactly like `SchrodingerPoissonCarrier`, with
`extras` = face potentials, gate charges, film sheet charge, iterations.

**Tests** (`tests/test_sp_dual_gate.py`, no DEVSIM): symmetric oxides and biases give a mirror-symmetric
profile to 1e-10; Gauss at both faces to 1e-9 C/m^2 and neutrality Q_b + Q_t + Q_film = 0 to 1e-9 of
|Q_film|; a 40 nm film at 0 V has centre n/n_bg within 1e-3 of 1; a thick film with the bottom oxide made
very thick (C_b -> 0) approaches the single-gate class with `gate_map="face_field"` and
`neumann_body=True` in the top half (record the convergence); exchanging top/bottom (oxides and biases)
mirrors the result exactly (sign pin). Do NOT gate on a zero 0 V deficit (section 2).

**Size:** 4-5 days. **Depends on:** P6.

### P13. Kane band in the electrostatics (C5 closure) -- wave 3

**Files.** new `carriers/kane_band.py`, `carriers/physics_equilibrium.py`, `materials/transport_model.py`,
`materials/scattering.py`, `carriers/devsim_layered.py` (the `setup_semiconductor_region` call only).

**Work.** `KaneBand(m0_kg, alpha_eV)` dataclass in `materials/transport_model.py`;
`TransportModel.band = "parabolic"` default. `kane_band.py`: DEVSIM expression of n(eta) = N_c(m0)
[F_1/2 + (15/4) b F_3/2 + (105/32) b^2 F_5/2] (b = alpha kT) with closed-form F_j approximants (one
function, reused for its derivative via DEVSIM `diff`), and a numpy twin by quadrature for tests. `Phi_c0`
from inverting the closure at n_bg (bisection, as `invert_F12`). `KaneOpticalMass` reads the same
`KaneBand` when the material has one (one band for optics and electrostatics). DD/bipolar + KaneBand
raises NotImplementedError.

**Tests**: `KaneBand(alpha=0)` gives model strings and arrays bit-identical to "parabolic"; the closure
matches the quadrature n(eta) to 1e-3 over eta in [-5, 40] for b up to 0.0104 (alpha 0.4 eV^-1 at 300 K;
record the worst point); at T -> 0 limit it reproduces k_F^3/(3 pi^2) with hbar^2 k_F^2/2m0 = E_F(1 + alpha
E_F) to 1e-3; the audit's table (Kane/library = 1.049 / 1.149 / 1.252 / 1.412 at 0.1 / 0.3 / 0.5 / 0.8 V band
bending, with the same DOS mass at n_bg) is reproduced by the DEVSIM closure on a 1-D device to 2e-3.

**Size:** 5-7 days. **Depends on:** P5, P11.

### P14. Lumenairy L_next adoption -- after L_next is on PyPI

**Files.** `optics/lumenairy_bridge/_common.py`, `pyproject.toml` (the lumenairy comment line only).

**Work.** FEATURE_FLOORS row `pmm_geometric_grade` = L_next; message-scoped filter for L_next's Laurent
warning where DynaMeta already warned (P2 call sites take a `_lumenairy_quiet()` context manager defined
here; coordinate the call-site edit as a one-line follow-up in `rcwa_backend.py`, which P14 may touch
after P2 has merged); re-run `validation/lumenairy_*.py` on L_next and on 5.22 (floor leg), then
`VERSION_VERIFIED_MAX = (5, 51)` (or L_next's minor); verify that no DynaMeta path passes PMMStack
`n_orders=` (P3 removed it) and that no DynaMeta add_layer call relies on lumenairy's default
`formulation` (P2 removed them).

**Size:** 1 day.

### P15. Release integration

**Files.** `pyproject.toml` (version), `dynameta/__init__.py`, `cache.py` (`_SCHEMA = 8` with a history
comment: "VALUE change: RCWA default formulation and raster, carrier native grid, SP E_F/gate map, ISB
eps_zz"), `README.md` (quick start unaffected; add the new knobs and the recommended lumenairy), 
`.github/workflows/ci.yml` (union of the packages' new test files), the release note
`docs/audit/<date>-v0.16.0-release.md` compiled from the package notes, the status line in this plan.

---

## 7. Integration order

1. **P1** (one day; everyone rebases on it).
2. **Wave 1**, at most five engineers at a time: batch A = P2, P4, P6, P7 (+P8 by the same engineer
   after P7), P9; batch B as engineers free up = P3, P5, P10. Merge into staging in this order: P6, P7, P8,
   P9 (independent modules, fast suites) -> P2 -> P3 -> P4 -> P5 -> P10. After each merge run that
   package's fast subset plus the census tests on staging.
3. **Wave 2**: P11 (needs P4 + P5 on staging), P12 (needs P6).
4. **Wave 3**: P13 (needs P11). If late, ship 0.16.0 without it (decision 11).
5. **P14** whenever L_next is on PyPI and P2/P3 are merged; it does not block 0.16.0.
6. **P15**: version bump, schema bump, ci.yml union, release note; then the full verification below, then
   the PR (merge needs the owner's OK).

## 8. Test commands

Per package (fast subset; run from the worktree root; one pytest process at a time on Windows):
```
python -m pytest -q -m "not slow" <the package's test files> tests/test_api_surface.py tests/test_numerics.py tests/test_ci_solver_coverage.py tests/test_devsim_env.py tests/test_audit_*.py
ruff check .
```
- P1/P2/P3/P14: add `tests/test_lumenairy_bridge.py tests/test_lumenairy_shapes_sem_wrap.py
  tests/test_lumenairy_bridge_verify_550.py` and `python -m validation.run_all lumenairy` (the bridge
  validations); P2/P3/P14 also once with lumenairy 5.22 installed in a scratch venv (floor leg:
  `pip install "lumenairy==5.22.*"`).
- P4/P5/P11/P13: the DEVSIM files they touch plus `tests/test_gated_builder.py
  tests/test_devsim_smoke.py tests/test_devsim_geometry.py`; `python -m validation.run_all gated`
  for the gated-device validations.
- P6/P12: `tests/test_schrodinger.py` and `python -m validation.run_all sp`.
- P7: `tests/test_effects.py` and `python -m validation.intersubband_eps_zz`.
- P9: `tests/test_nonlocal_tmm.py tests/test_hydro_fem.py` and `python -m validation.nonlocal_hydro_material`.
- P10: `tests/test_fem_2d_tm.py` (solvers installed).

Before release (on staging, after P15):
```
python -m pytest tests/ -q            # whole tree, @slow included (make test)
ruff check .                          # make lint
python -m validation.run_all          # full validation tier (or --shard K/4 x 4)
```
Run the whole tree once on Windows and once under WSL/OpenBLAS (the CI proxy), and confirm any bitwise
gate on CI or a second machine. Expected moved pins are those listed per package; any other moved number
is a finding to diagnose, not to re-record.

---

## 9. Open decisions (recommendations)

1. **Version.** 0.16.0 (six default changes and a cache schema bump do not belong in a patch). *Recommend.*
2. **RCWA default formulation.** `'auto'` now, with a per-solver advisory for explicit Laurent on hard
   cells; the alternative (warning-only in 0.15.2, flip in 0.16) leaves the 0.17-0.50 error in r (fixture A,
   N = 100-50) on the default path for another release. *Recommend 'auto' in 0.16.0.*
3. **RCWA raster.** `raster_sampling='area'` default, `'point'` legacy. *Recommend.*
4. **Native carrier grid.** Automatic on `LayeredDevsimBuilder` + a layered optical solver; FEM path
   unchanged; `carrier_solve_kwargs` overrides. *Recommend.*
5. **DEVSIM grading.** `semiconductor_refinement="screening"` default; "legacy" restores v0.15.1. It moves
   every default-mesh DEVSIM result by about 2 % in accumulation and 11 % in depletion at the interface.
   *Recommend.*
6. **Equilibrium edge electrode.** Fermi-level reference (no Dirichlet), `pin_potential=True` for the old
   column pin; two edge electrodes at different voltages on one equilibrium region raise. *Recommend.*
7. **Intersubband z response.** `'subbands'` default; legacy deprecated, removed in 0.17. *Recommend.*
8. **SP Fermi level at alpha = 0.** Use the model relation at finite T (moves alpha = 0 results by about
   1 mV of band bending; the old formula stays as `fermi_level="bulk3d_T0"`). *Recommend.*
9. **SP single-gate gate map.** `gate_map="face_field"` default (Gauss's law at the oxide face; psi_s moves
   by 0.20-0.30 V on the audit's film over -4..+2 V), `"flat_band_excess"` (the sign-fixed current method) kept. The sign
   fix ships either way. *Recommend face_field*, because the dual-gate solver uses the same physics and
   the two classes then agree in the thick-film limit.
10. **Lumenairy floor.** Keep VERSION_FLOOR 5.22.0, add `rcwa_eps_cell_normal` (5.31.0) and, with P14,
    `pmm_geometric_grade` (L_next); VERSION_VERIFIED_MAX -> L_next after re-verification. Alternative:
    raise the floor to 5.50.1. *Recommend keeping 5.22.0.*
11. **Release without P13.** Ship the C5 guard in 0.16.0 and the Kane closure in 0.16.1 if P13 is not ready.
    *Recommend.*
12. **Deferrals.** In-Newton DG for equilibrium films (Q5 item 3) and the nonlocal 2-D FEM (Q7.2) as
    documented limitations; Q7.3 rejected as specified. *Recommend.*

---

## 10. Not checked

- C1, C2, C3, C4 optical sizes (the r shifts) and the C2 deviations were taken from the audit; the C4
  thickness study and candidate meshes were re-measured here (section 2), the optical consequence was not.
- O1's hand-built two-zone PMM value, O2's timing table and N1's grid table: AUDIT values, not re-run.
- O3/O4 on lumenairy 5.50.0: on the generic fixture only 5.50.2 was run (the 5.50.0 'li' raster route
  reproduced with `inclusion_model='raster'`); the sister audit reports 5.50.0 and 5.50.2 identical for
  these paths.
- C5's Kane table: formula re-derived, numbers not re-run.
- C6: the refusal was read in code, not re-run.
- Q5, Q6 numbers: AUDIT values; Q6's interface condition was derived here, not prototyped.
- Q2 dual-gate design: not prototyped; the face-field Gauss law was exercised only in the single-gate
  form (section 2).
- The 2-D disk TM case for the O3 auto rule (P2 must measure it).
- Whether the equilibrium builder honours `T_K` (`physics_equilibrium.setup_phi_c0` uses the reference
  temperature): noticed while reading, outside this audit; worth a separate look.
