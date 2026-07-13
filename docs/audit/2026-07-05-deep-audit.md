# DynaMeta deep audit — 2026-07-05

**Git HEAD at audit:** 09e87fc (post BOR-PMM bridge, post FEM symmetry reduction, post QD-SOA Phases 1-34b)
**Scope requested:** feature gaps / seams between library portions, physical correctness,
code-convention adherence, library organization, and speed/memory opportunities that do
NOT sacrifice accuracy.
**Method:** staged multi-agent audit (one focused pass per subsystem, not one giant fan-out),
refute-first finding discipline, bidirectional adversarial verification (refute claimed
defects AND prove out claimed-good paths), findings hand-re-graded before acceptance.
This document is written incrementally as each stage completes.

**Prior-audit context (do not re-report):** the 2026-06-22 two-pass deep audit
(d8bdbdc, 8871b77, 7eae231, 8626360, 61092ec, a6b3987) resolved 1 P1 (cache tensor-fingerprint)
plus ~24 P2 seam/guard findings; earlier rounds are documented in this folder.
Landed SINCE that audit and therefore never audited in-repo: FEM mirror-symmetry
reduction (c638181), the Lumenairy BOR-PMM bridge (09e87fc), and the entire QD-SOA
module (optics/soa/, Phases 1-34b) plus the Berreman/EMT bridge backends.

**Baseline:** fast suite `pytest tests/` — result recorded in §0.3.

---

## Executive summary

**Verdict: the physics cores are sound; the debt lives in the seams, extractors, and
gates.** Across ~130 agents in 8 stages (≤5 concurrent after stage 3), every
foundational solver formula attacked — both Scharfetter-Gummel forms, density-gradient,
S-P, QCSE, LLG, two-temperature, FDTD updates and gain ADEs, FEM weak form and symmetry
walls, TMM conventions, SOA detailed balance, adjoint gradients — verified CORRECT,
many to 1e-15 against independent oracles, and five deliberate attempts to overturn
CORRECT verdicts all failed. The confirmed defects — **3 P1, 23 P2** — cluster where
data crosses subsystem boundaries (slab ordering, seam adapters, caches, unit/convention
tags) and in the *diagnostic/extraction/guard* layers around correct kernels.

The three P1s are all small fixes on flagship paths: (1) all three lumenairy bridges
vertically flip graded eps(z) profiles (one-word fix; fixture: R reads 0.062 where truth
is 0.330, invisible to closure checks); (2) the FDTD pipeline seam silently discards
carrier/effect modulation — every bias returns identical R/T — or crashes blaming the
material; (3) the Norris-Landzberg fatigue acceleration factor is inverted, making
every accelerated-test extrapolation non-conservative by 3-13x, with a gate that
re-implements the same inversion. Highest-blast-radius P2: `KaneOpticalMass`'s default
is the Kane formula with alpha silently halved — λ_ENZ predictions ~4-12% short for
literature ITO parameters — validated only against a copy of itself.

The systemic lesson repeats at every scale: **self-consistency gates (energy closure,
conservation, off-switch, scaling) pass while per-quantity physics is wrong.** A
tautology sweep found the pattern is structural: several validation gates re-implement
the code under test, and four self-consistent-only gates are the sole pins on exactly
the quantities this audit found defective. The single highest-leverage infrastructure
fix is wiring the smoke validation tier into CI with a SKIP category and converting the
tautological gates to independent oracles. Details: findings in §2-§7 (final grades in
§7.4-7.5), feature-gap register in §1.4, performance register in §6.2, residual scope
in §7.6, and the post-audit lumenairy bridge-expansion assessment in §8.

---

## 0. Repo inventory (Stage 0 — inline scout)

### 0.1 Package layout and size (25,441 LOC in `dynameta/`)

| Package | LOC | Contents |
|---|---|---|
| top-level | 1,049 | pipeline, analysis, cache, results, sweep, transient_optics, viz, constants |
| core/ | 1,378 | backend, bridge, carrier_field, eps_field, graphene, interfaces, layered, lift, n_to_eps, numerics, resample, units, alignment |
| core/effects/ | 966 | base, electro, electroabsorption, magneto, reconfigurable, thermo |
| carriers/ | 6,671 | DD stack (equilibrium/DD/bipolar/density-gradient), devsim glue, electrostatics(+FEM), S-P, QCSE, mobility, LC director/dynamics, LLG, thermal, AC/transient/switching |
| carriers/thermal_fem/ | 998 | steady, transient, kirchhoff, twotemp |
| optics/ | 3,315 | solver (FEM), ngsolve_layered, tmm_reference, eps_assembler, fdtd, fdtd_mo, fdtd_seam, laser_gain, inverse_design, topology_opt |
| optics/fdtd_nd/ | 2,595 | 2-D/3-D FDTD: kernels (numpy/numba/jax), CPML, oblique2d, solve2d/3d |
| optics/lumenairy_bridge/ | 1,429 | rcwa, pmm, berreman, bor, emt_screen, translate |
| optics/soa/ | 3,828 | qd_gain, traveling_wave, maxwell_bloch, sbe, ase_noise, lineshape, calibration, thermal, transverse_bpm, metrics |
| materials/ | 989 | db, material, optical_model, transport_model, scattering, mechanical |
| geometry/ | 662 | design, specs, stack, unit_cell, cross_section, electrode |
| reliability/ | 1,142 | bti, corrosion, dedoping, em, fatigue, hci, leakage, lidt, mttf, stress_migration, tddb |
| drivers/ | 303 | reliability_glue, state_glue |
| io/ | 116 | store |

Tests: 54 files (fast, numpy/scipy-only gate). Validation: 205 solver-backed scripts
(`validation/run_all.py` aggregates). Examples: `examples/`.

### 0.2 Audit stage plan

1. Subsystem survey + coverage map (10 readers + seam synthesis)
2. Physics: carriers/
3. Physics: optics core + fdtd_nd
4. Physics: SOA + lumenairy_bridge + reliability
5. Seams & feature gaps (cross-module)
6. Conventions, organization, performance/memory
7. Bidirectional verification sweep + completeness critic + final grading

### 0.3 Baseline test run

`python -m pytest tests/ -q` at 09e87fc: **482 passed, 0 failed** (26 min 15 s wall,
3 numba-cuda occupancy warnings; the CUDA legs genuinely ran on this machine's GPU —
they are among the legs that silently skip in CI, see §1.2).

---

## 1. Codebase map & subsystem survey (Stage 1)

**Method:** 10 subsystem readers (each read every file in its scope) + 1 test/validation
coverage mapper + 1 cross-subsystem synthesizer that reconciled the dependency picture,
spot-verified seam claims against source, and deduped the deep-check target lists for
Stages 2-6. Everything in this section is a *mapped observation or verified-location
candidate*; physics claims are graded in Stages 2-4 and all surviving findings are
re-verified in Stage 7.

### 1.1 Dependency graph & layering

The import graph is a clean DAG that mostly respects the documented layering:
`constants` is a true leaf (~40 importers); `core` depends only on constants +
`materials.MaterialRegistry`; materials/geometry/io are declarative leaves; carriers and
optics sit above core+geometry+materials; `reliability` depends only on
constants/materials; `drivers` is glue over carriers+reliability; `optics/soa` depends
ONLY on constants (fully isolated); orchestration sits on top reaching heavy solvers
lazily. Pure leaves with zero in-package importers (user-injected only): lumenairy_bridge,
drivers, reliability (except drivers), soa, results/cache/viz. The PEP-562 lazy-import
discipline is respected everywhere — `import dynameta` stays numpy-light as designed.

**Layering violations found:**
1. The one genuine upward import: `carriers/carrier_heating.py:145` imports top-level
   `dynameta.transient_optics` (which imports `optics.tmm_reference`) — Stage-1 carriers
   reaching into orchestration glue.
2. Ownership inversion: `optics/fdtd_nd` imports its input spec `FDTDLayer` from the 1-D
   module `optics/fdtd.py` (solve2d.py:13, solve3d.py:13).
3. Knowledge inversions without imports: `core/layered.collapse_regions_to_layers`
   hardcodes geometry/optics region-naming conventions (`__incl`, `_` splits, `pml_*`)
   inside core; `geometry/specs.OpticalSpec` carries NGSolve-only solver knobs;
   `analysis.gate_cv` hardcodes the carriers-side 2D `'y'`==z axis convention.
4. A private-name coupling web across module boundaries: `rcwa_backend.py:51-52` imports
   `fdtd_seam._cell_axes/_layer_eps_cell` and `tmm_reference.S`;
   `carriers/electrothermal.py:31-32` imports thermal_fem underscore helpers;
   `inverse_design`/`fdtd_mo` import fdtd_nd underscore kernels (fdtd_nd/`__init__`
   re-exports ~25 private names as its de-facto API); `lumenairy_bridge/translate.py:92`
   reads the EXTERNAL library's private `_layers`.

### 1.2 Test & validation coverage

Two-gate architecture: fast pytest gate (54 files / ~470 tests, numpy/scipy-only,
per-file self-skip of heavy deps) + exit-code-gated validation tier (~205 scripts,
`validation/run_all.py`, serial, 1800 s/script). Coverage *levels* per package are
strong nearly everywhere (see below), but the *gating* has structural holes:

| Package | Level | Key caveat |
|---|---|---|
| orchestration | strong | cache has a dedicated adversarial validation |
| core, core/effects | strong | `core/units.py`, `core/resample.py` have zero direct coverage |
| carriers | strong (validations) | DEVSIM solver modules (dc_solve, ac_analysis, transient, physics_*, eq_registry) have **zero pytest presence** — no skip-gated smokes, unlike the NGSolve drivers |
| carriers/thermal_fem | strong | all pytest legs skip in CI (no ngsolve) |
| optics core | strong | mirror-symmetry reduction gated ONLY by an NGSolve-required validation, never CI |
| optics/fdtd_nd | strong | `cpml.py` has no direct unit test anywhere |
| lumenairy_bridge | strong | BOR physics leg rests on a single validation script; bridge tests skip wholesale in CI |
| optics/soa | strong (densest in repo) | **all 36 qd_soa validations absent from the smoke tier** — run only in the hours-long full tier |
| materials | moderate | `transport_model.py` never directly asserted |
| geometry, io, drivers, reliability | strong/moderate | reliability is the cleanest 1:1 test+validation pairing |

**Test-infrastructure gaps (verbatim-verified against ci.yml / run_all.py / Makefile):**
- CI never runs the smoke validation tier that run_all.py:8 advertises as "CI-able";
  ci.yml runs only `pytest tests/ -q`.
- lumenairy, ngsolve, devsim, zarr, jax are all absent from CI, so those test legs are
  skip-as-pass dead; skip counts are not surfaced or bounded.
- Validation discovery is a source-string regex (`SystemExit|sys.exit`, run_all.py:66): a
  gated validation that exits non-zero via a bare raise/assert is silently classified as
  a diagnostic and NEVER RUN, with no warning.
- Skip-as-pass semantics for optional capabilities (CUDA/jax): a broken install reads
  green; run_all has no SKIP result category.
- No pytest markers / no coverage measurement in CI; run_all is strictly serial with no
  parallelism or fail-fast.
- Uncovered modules: core/units.py, core/resample.py, carriers/fem_mesh.py,
  fdtd_nd/cpml.py, the DEVSIM solver modules, materials/transport_model.py,
  carriers/eq_registry.py, and run_all.py itself.

### 1.3 Cross-subsystem seam candidates (to be graded in Stages 5/7)

Two candidates were source-verified by the synthesizer at **P1 severity**:

1. **Slab-order flip for graded eps(z)** — `core/layered.slice_eps_field` emits
   ascending-z (substrate-first) slabs (layered.py:85,102); `tmm_reference.py:228`
   `reversed(...)`s them into its superstrate-first stack, but the RCWA/PMM/Berreman
   bridges append them UNREVERSED inside superstrate-first layer loops
   (rcwa_backend.py:198, pmm_backend.py:118, berreman_backend.py:127). For an asymmetric
   graded profile — the ENZ carrier-accumulation layer, the library's core use case — the
   within-layer gradient is vertically flipped in one family. A symmetric test profile
   passes everywhere, hiding it.
2. **FDTD OpticalSolver silently drops carrier modulation** — `fdtd_seam.py:104-108`
   requires `is_uniform AND scalar`, else falls back to nominal `design.materials` eps: a
   graded EpsField (the case `graded_fdtd_layers`/`eps_profile_from_carrier` were built
   for — never called by any pipeline path, fdtd_seam.py:221-244) or a uniform TENSOR
   field is replaced by *unbiased* material eps with no warning. TMM slices graded
   correctly and raises on tensor, so the same Design gives silently different physics
   per backend.

P2-graded candidates (all with file:line evidence, most synthesizer-verified):

3. `layer_absorption` array layout assumed OPPOSITE ways by sibling bridges: RCWA
   heuristic `(2, n_layers)` (rcwa_backend.py:295-297) vs Berreman `(n_layers, 2)`
   (berreman_backend.py:174-175); feeds per_region_absorption → reliability/LIDT while
   total A stays correct (invisible to energy gates).
4. Cache staleness surface: materials hashed by NAME only (cache.py:66) while TMM
   re-derives end media from `design.materials` at solve time (tmm_reference.py:250-252);
   `Feature.priority` not fingerprinted though `Inclusion.priority` is; `repr()`-hashing
   of optical/mesh_3d specs; untagged grid bytes in `_eps_fingerprint` — same family as
   the previously-fixed tensor-collision P1.
5. Cache and sweep-aware fast path mutually exclusive (cache.py:126 `__call__`-only vs
   pipeline.py:123 `hasattr(solve_optics,'solve_sweep')`); per_region_absorption dropped
   on every cache HIT and by SweepResults' fixed schema.
6. per_region_absorption keying is backend-dependent (FEM/RCWA: design layer name;
   TMM/Berreman: `slab_<i>`; BOR: absent) — reliability post-processing must know which
   backend ran (reliability_glue.py:104-123).
7. 2D carrier grid axis contract (`'y'`==z) embedded in three subsystems with no shared
   constant (devsim_layered.py:627, analysis.py:62-68, lift/bridge).
8. `extra_fields` bypass the FieldLift in assemble_eps (bridge.py:103,110-112): 2D T/E
   grids from thermal/electrostatic drivers merge raw — exactly the shape class
   drivers/state_glue emits.
9. EpsField has no axis-unit tag; three bridges hardcode nm axes (`1.0/_S_NM` at
   rcwa:198/pmm:118/berreman:127) — a metre-axes EpsField mis-sizes slabs by 1e9 silently.
10. OpticalResult R/T semantics differ per backend (FEM 0-order specular vs bridge
    order-summed vs FDTD 0-order + flux A) with no convention tag on the result.
11. FDTD seam never reads `design.optical` (fdtd_seam.py:335-360): theta/azimuth/pol
    silently ignored though fdtd_nd ships oblique kernels; compounded by silent backend
    degradations (solve3d.py:33-51 numba-cuda→numpy; solve2d.py:310-320 oblique
    cupy→numpy) and oblique solvers silently dropping chi3/chi2/raman/gain the 1-D entry
    point raises on.
12. LumenairyStackSolver fabricates px=py=lambda_m when stack periods are 0 with no
    uniformity guard (rcwa_backend.py:369-370); LayeredStack period fields exist but no
    slicer populates them.
13. reliability: `arrhenius_af` raises for Ea<0 (mttf.py:30-31) while HCI defaults to a
    legitimately NEGATIVE Ea=-0.1 eV (hci.py:41-43) — REL10 cannot extrapolate REL8.
14. LC dual tilt convention half-bridged: lc_director_2d emits FIELD-AXIS theta with no
    bridge helper to LiquidCrystalModel; state_glue profile mode emits an (nz,) array
    under a key the EffectModel otherwise reads as scalar.
15. carrier_heating.py:147-149 silently replaces a calibrated DrudeOptical's callable
    m_opt_kg with M_E and callable gamma with 1e14 rad/s crossing into the transient loop.
16. lumenairy version floors scattered and stale: pyproject floor 5.14.2 < berreman/emt
    need 5.14.4 < BOR needs 5.16.0; PMM bridge hard-raises on conical although lumenairy
    5.20.0 ships native conical PMM (P3).
17. Sweep-aware fast path has FEWER capabilities than the per-wavelength seam
    (fdtd_seam.py:430-432 vs :352-354); pipeline's two-point end-media dispersion check
    misses in-band resonances (pipeline.py:132-145) (P3).
18. eq_registry process-global dict + fixed default device names collide on a second
    build without teardown(); 2D builder silently drops unknown BiasPoint keys while 3D
    warns (P3).

### 1.4 Capability-matrix gaps (feature asymmetries)

1. **Oblique incidence/polarization:** FEM, TMM, and all lumenairy bridges support it;
   the FDTD OpticalSolver silently solves normal incidence; bottom-side incidence
   rejected by ALL backends; TMM remaps pol 'x'→'s' at oblique where FEM raises.
2. **Graded/tensor eps routing:** TMM slices graded; FDTD seam drops both graded and
   tensor silently; bridges handle graded but drop `__incl` inclusion modulation via
   collapse_regions_to_layers; BOR has no graded auto-slicing; and materials cannot even
   DECLARE an anisotropic OpticalModel although EpsField/cache/effects fully support
   tensors.
3. **Hole density:** CarrierField promises hole_density_m3 and bipolar DD produces it,
   but NO code reads it — the hole-plasma Drude contribution is absent for
   p-type/bipolar devices.
4. **2D-vs-3D carrier builder parity:** velocity-saturation mobility 2D-only; gate
   phi_bi built-in-offset fix 3D-only; ssac/transient wiring 2D-only; unknown-bias-key
   warning 3D-only; bipolar lacks velocity saturation; run_pipeline never dispatches to
   the 3D solver.
5. **Mirror-symmetry reduction** exists on the optics FEM side only (not carriers/thermal
   FEM), and excludes semiconductor layers, tensor eps, and oblique incidence — exactly
   the gated-ITO cases DynaMeta exists for.
6. **Feature (z-spanning via/T-patch) is a dead adapter:** validated by Design,
   fingerprinted by the cache, consumed by NO builder — silently feature-less solves.
7. **solve_sweep parity:** BOR lacks it; no caching path exists for ANY sweep-aware solve.
8. **Absorption diagnostics:** RCWA+Berreman fill per_region_absorption; PMM has no
   absorption path; BOR fills neither and discards all phase (r=sqrt(R), phase_deg=0).
9. **SOA composability:** e/h-split models produce garbage through
   saturation_curve/ASE/MaxwellBloch paths (rho_GS indexes the excitonic layout, no
   guard); ManyBody corrections change nothing in the simulated amplifier; ES-band ASE
   unproducible though calibration reads it; gain_dyn kernel unreachable;
   line_filter/GVD/Langevin/nl_loss single-pol single-band only; numba fast path rejects
   I(z) profiles the numpy path supports.
10. **Reliability glue lag:** impact_ionization→HCI, thermal-transient dT→fatigue,
    leakage Joule→electrothermal, BTI-from-electrothermal all lack glue counterparts;
    miner_time_to_failure_s never passes length_m so the promised Blech-immortality
    check cannot fire.
11. **Thermal FEM:** no Kapitza interface resistance; per-layer k(T) exact only in the
    1-D walk; k(T) transient lacks time-dependent loads; two-temperature results can't
    feed the SOA `.T_at` duck-typed seam; no transient electro-thermal loop despite both
    halves existing.
12. **Test-infra capability matrix:** see §1.2.

### 1.5 Organization assessment (summary; full treatment in Stage 6)

The library is well-layered; organizational debt clusters into five patterns:
1. **Monoliths** due for the repo's own proven split treatment: optics/solver.py (887),
   soa/qd_gain.py (1378), soa/traveling_wave.py (921, six near-duplicate marcher loops —
   the direct cause of the SOA composability gaps), carriers/lc_director.py (876),
   carriers/devsim_3d.py (833), core/effects/electroabsorption.py (428), fdtd_seam.py
   (three unrelated roles).
2. **Misplacements:** FDTDLayer in optics/fdtd.py; transient_optics.py at package top
   level consumed by carriers; core/graphene.py (an effects-family model, unexported,
   stale docstring); OpticalSpec with NGSolve knobs in geometry/; analysis.py mixing
   optical-spectral and electrical-FOM domains; rcwa_backend.py doubling as
   lumenairy_bridge's unofficial _common.py (seeded bor_backend's copy-paste island).
3. **Single-source violations:** KB_EV_K re-declared in 8 reliability modules; H_PLANCK
   in 5 soa files (one raw literal); the exp(-iwt) convention string x3 in core; F12
   Aymerich-Humet python + DEVSIM-string twins with no shared coefficients; Kane E_F(n)
   inversion x3; sbe.py hardcodes m_e/c.
4. **Stale status text** (violates the repo's own re-verify discipline): dynameta
   `__init__` module map omits BOR + symmetry, version 0.5.0 unmoved across three
   feature drops; em.py/hci.py claim shipped drivers as follow-ons; sp_carrier claims
   parabolic-only for nonparabolic code; graphene.py claims the sheet BC is a follow-on
   that solver.py:454 implements.
5. **Naming/export drift:** fdtd_nd re-exports ~25 underscore names;
   solve_fdtd_3d_oblique returns FDTD2DObliqueResult; carriers `__dir__` advertises
   names `__getattr__` cannot deliver.

### 1.6 Stage worklists

The synthesizer produced 78 deduped deep-check targets: 16 carriers physics, 15 optics
physics, 16 SOA/bridge/reliability physics, 16 seams, 15 conventions/perf. These drive
Stages 2-6; each target is recorded with its verdict in the corresponding section.

## 2. Physics correctness — carriers (Stage 2)

**Method:** 16 targeted refute-first deep checks (from §1.6) + 3 open-ended sweeps
(DEVSIM equation strings incl. hand-checking derivative strings, numerics/BCs/mesh,
units/constants). Checkers hand-derived each formula before reading the code and ran
pure-numpy probes where decisive (probe scripts in the session scratchpad). Every claimed
P1/P2 defect was then verified by TWO independent lenses — an adversarial refuter
(instructed to prove the claim wrong, default-refute) and a from-scratch re-deriver —
and is reported CONFIRMED only when both agree. 25 agents total.

**Headline: the carriers physics core is again SOUND on its mainline paths.** All the
classically dangerous formula sites (both Scharfetter-Gummel forms incl. the FD
V_t→g·V_t generalization, density-gradient b-constant, F1/2 twins, QCSE tilt signs, S-P
dilog/Jacobian, two-temperature coupling, LLG torque coefficient, contact-current
dimensionality, transient stepping) verified CORRECT, several to 1e-15 against
independent numerical oracles. The three CONFIRMED defects are all in *wiring/reference-
frame* logic around the solvers, consistent with every prior round's pattern (physics
sound, risk at boundaries). No P0/P1.

### 2.1 Confirmed defects (both verification lenses agree)

**C2-1 (P2, CONFIRMED). Bipolar 3D gate reference offset ignores `net_doping_expr`** —
`devsim_3d.py:545-547` (interacting with `_setup_bipolar_semi` :488-493 and
`physics_bipolar_dd.py:216-219`). The body contact pins psi from the ACTUAL NetDoping
node model (which is `net_doping_expr` when set), but solve() computes the gate offset
phi_bi from the dead scalar fields (`acceptor`, `n_bg_m3`). The two frames cancel only
when the expression's body-side doping matches the scalars. Failure: a uniform p-type
cap expressed via `net_doping_expr='-1.0e23'` with `acceptor=False` silently applies
+0.714 V of spurious gate bias at nominal Vg=0; Newton converges, no diagnostic. Fix:
derive phi_bi from NetDoping evaluated at the body-contact nodes (reduces byte-identically
to the scalar form for uniform doping), or raise when net_doping_expr is set on the
gated cap. Also: the acceptor branch has no validation case.

**C2-2 (P2, CONFIRMED). Kane ⟨m*(Te)⟩ uses the parabolic Sommerfeld coefficient with
the nonparabolic E_F** — `carrier_heating.py:72`. For any DOS the fixed-n Sommerfeld
shift is Δ⟨E⟩=(π²/6)(kT)²g(E_F)/n; for the Kane DOS this carries the factor
(1+2αE_F)/(1+αE_F) that the code omits (=1.274 at the validation's own parameters). The
heating-induced ⟨E⟩/⟨m*⟩ rise — the very effect the module models — is understated
~11-21% (probe: Δ⟨E⟩ at Te=2000 K is 0.0948 eV coded vs 0.1116 eV exact;
ΔRe(eps)@1500 nm +0.242 vs +0.201) while all existing gates pass (they pin limits, not
the coefficient). Fix: multiply the quadratic term by (1+2αE_F)/(1+αE_F) (→1 at α=0,
preserving the off-switch gate) or integrate the Kane DOS directly. Related P3: the
arithmetic occupied-state mass average m0(1+2α⟨E⟩) is first-order inconsistent with the
(-df/dE)-weighted conductivity average used by the library's own KaneOpticalMass.

**C2-3 (P2, CONFIRMED). 2D layered DEVSIM builder never creates lateral interfaces** —
`devsim_layered.py:297-312`. The interface loop's only adjacency test is z-contact, which
can never fire for same-layer regions; a non-ambient background layer with an inclusion
(the patterned-layer class the geometry API advertises) gets coincident DUPLICATED nodes
with no interface — DEVSIM semantics then give each side a natural zero-flux wall, so the
inclusion is electrostatically isolated from its surroundings, silently. The 3D builder
(devsim_3d.py:723) explicitly finds lateral interfaces, so 2D and 3D give silently
different physics for the same Design. Fix: emit vertical `add_2d_interface`s (or raise
NotImplementedError for patterned non-ambient layers until wired, matching the 3D
builder's discipline).

### 2.2 Notable P3 findings (unverified this stage; regraded in Stage 7)

- **Semiconductor-semiconductor interfaces get Potential-only continuity**
  (`devsim_layered.py:394-395`): Electrons/Holes equations get natural zero-flux walls at
  the junction — a silently carrier-blocking heterojunction between two adjacent
  drift-diffusion regions. *(Finder graded P3; flagged for regrade — the 3D twin behaves
  differently.)*
- **Impact-ionization FV quadrature on 3D tet meshes** (`impact_ionization.py:97-114`):
  edge-projected alpha(E‖)·|cosθ| double-counting underestimates I_sub ~2-3x on
  unstructured tets; no dimension guard. *(Flagged for regrade.)*
- **LC branch tie-break energy is neither fixed-V nor fixed-D**
  (`lc_director.py:520-522`, duplicated :810-817): exceeds the true fixed-V potential by
  a branch-dependent term and mis-orders untilted-vs-tilted branches through the
  Freedericksz pitchfork (probe: wrong ordering for V=1.005-2.0 V_th), contradicting the
  docstring. *(Flagged for regrade.)*
- Gummel convergence declared on empty default `semiconductor_regions=()` — one outer
  iteration then "converged" (`dc_solve.py:108-132`).
- `_nearest_nonmetal` looks up LAYER names in a REGION-keyed dict — ambient layer with
  ≥2 inclusions mis-wires or errors the gate contact (`devsim_layered.py:220-235`).
- y-mesh lines only at layer zlo: a meshed layer under a skipped ambient gap gets its top
  snapped to the nearest graded cell (probe: 7 nm → 7.5 nm) (`devsim_layered.py:264-286`).
- `ssac_admittance(v_ac≠1)` silently mis-scales C,G — sources hardcode acreal=1.0
  (`ac_analysis.py:59-80`).
- exp(-eta) in the Electrons node model can trigger a DEVSIM FATAL abort below
  Phi_c0−709.78·V_t (~−18.9 V for ITO) (`physics_equilibrium.py:50,55`).
- Caughey-Thomas driving force is the electrostatic edge-parallel field, not the
  quasi-Fermi gradient — spurious saturation on high-built-in-field zero-current edges
  (accepted TCAD limitation, opt-in path) (`physics_drift_diffusion.py:146-149`).
- Two-temperature: broad `except Exception` NaNs the energy-balance diagnostic silently
  (`electrothermal.py:195-198`); `total_joule_W` "(power in)" excludes top-face flux so
  the advertised balance breaks whenever flux≠0 (:87-88).
- LLG: for a non-symmetric N_demag, H_eff uses −Ms·N@m while the energy gradient implies
  the symmetrized (N+Nᵀ)/2 — Lyapunov guarantee silently breaks (`llg.py:101-102`).
- Docs-only: degenerate-asymptote exponent typo in the g-factor docstring
  (`physics_drift_diffusion.py:23`, mirrored in einstein.py).

### 2.3 Verified-correct highlights (worth recording so they are not re-audited)

- Hole SG current (physics_bipolar_dd.py:158-161): token-for-token match with DEVSIM's
  own simple_dd.py canonical form; probe agreement 3.7e-15 vs exact edge-BVP and an
  independent RK4-shooting oracle; equilibrium current exactly 0.
- FD-enhanced SG: exactly SG with V_t→g·V_t (probe 2e-10 over 2000 random states); drift
  limit g-independent as required; ITO operating point g=14.61 vs fit 14.66 (+0.34%);
  a misplaced g would shift drift/diffusion balance ~14.6x at ITO bulk — existing gates
  (gated_dd GATE B, carriers_3d_resistor, bipolar GATE B, test_carriers_gfactor) pin all
  four failure limbs independently.
- Density-gradient b = γℏ²/(6 m q) verified as the standard Ancona convention (r=3);
  h-vs-ℏ correct.
- ssac C/G sign convention verified correct on a lossy device under DEVSIM's exp(iωt);
  F1/2 python/DEVSIM twins algebraically identical; contact-current dimensionality
  (2D [A/m] vs 3D [A] with Mesh.ScalingFactor=1e-9) handled correctly incl. the em-glue
  sign; transient dcop read returns the current step; QCSE tilt signs and tilt-removal
  referencing correct; S-P spence convention + Trellakis Jacobian sign + psi/z_ij
  normalization consistent; two-temperature G-coupling orientation and G→∞ limit
  correct; LLG H_K=2K_u/(μ0 Ms), γ0μ0/(1+α²) coefficient, and Stoner-Wohlfarth
  H_K/2@45° all reproduced.
- Units/constants sweep across all of carriers/ came back clean (MKS discipline holds;
  hardcoded parameter defaults order-of-magnitude checked against cited sources).

## 3. Physics correctness — optics core + fdtd_nd (Stage 3)

**Method:** as Stage 2 — 15 targeted refute-first checks + 3 sweeps (FDTD kernels/CPML,
FEM weak forms, TMM/adjoint), probes vs the `tmm` oracle where decisive; every claimed
P1/P2 defect verified by a combined adversarial-refuter + independent re-deriver agent.
26 agents, run ≤5 concurrent.

**Headline: this is where the real physics debt lives.** Seven distinct CONFIRMED P2
defects — the densest concentration in the audit — clustered in (a) the FEM solver's
*extraction/diagnostic* layer (not the weak form itself, which verified correct) and
(b) the fdtd_nd *de-embedding/guard* layer around otherwise-correct kernels. The FDTD
update equations, gain ADE identities, sheet-BC Robin term, p-pol lossy-substrate
transmittance, and R_flux construction all verified CORRECT, several to 1e-15 against
independent oracles.

### 3.1 Confirmed defects

**C3-1 (P2). FEM 6×6 lateral probe grid aliases diffraction orders ≡0 (mod 6) into the
"0-order" coefficient** — `solver.py:624-625` (all four call sites: R, T, both p-pol),
contradicting the Fourier-orthogonality docstring at :219-228. The cell-centred 6×6 grid
gives order weight (−1)^(m/6) for m≡0 mod 6 instead of 0. Probe: a substrate order-6
amplitude at 30% of r0 corrupts the fitted r to r0−a6 with fit residual 2.9e-3 — far
below the 5e-2 warn threshold, i.e. SILENT. Bites when Px > 6λ/n_medium (e.g. 3 µm
grating on Si at 1550 nm) or for any 6×1 supercell (the standard supercell-parity
check). All validations use sub-wavelength cells where order 6 is evanescent — fully
gate-blind. Fix: probe-grid size adaptive per medium, N = max(6, ⌊Px(n+|sinθ|)/λ⌋+1).

**C3-2 (P2). Instantaneous-Kerr convention is 3-4x weak, and inconsistent with its own
docstring** — `fdtd.py:106` and all seven nd kernels (kernels2d.py:141, kernels3d.py:88,
+ numba/jax twins). eps_eff = eps_inf + chi3·E(t)² multiplying dE/dt delivers a
fundamental-band shift of chi3|A|²/4 — 3x weaker than the standard χ⁽³⁾ convention
(¾·chi3|A|²) and 4x weaker than the docstring's "eps += chi3|E|²" (probe k≈0.24). A
user entering a literature χ⁽³⁾ gets SPM phase and switching thresholds off 3-4x; every
existing gate checks scaling or backend parity, not the absolute magnitude. Fix: pin ONE
convention (3·chi3·E² for the standard, or fix docs) across all kernels + a magnitude
gate.

**C3-3 (P2). MO grid sizing samples only the +1 circular branch** — `fdtd_mo.py:218-220`
(and the 3-D helper). For reversed magnetization / electron-signed wc<0 (the module's
own signed convention), the resonant branch is never sampled: probe shows n_max=1.28
reported vs 7.51 true → dz 5.9x too coarse with no warning, corrupting near-resonance
R/T/Faraday, while wc>0 with identical physics resolves fine. Fix: max over both
branches (±wc) and floor by the background birefringent indices.

**C3-4 (P2). fdtd_nd t0 phase de-embed is vacuum-only** — `solve2d.py:243` +
`solve3d.py:207` (found independently by T6, T8, and the kernels sweep). The factor
exp(+i·k0·z_struct) omits the n_super traversal leg and the (n_super−n_sub)·d residual
probe-leg; r0 was already fixed to carry n_super, t0 wasn't. For metasurface-on-glass
(n_sub=1.5) the t0 phase is off ~-122° at 1.3 µm and frequency-dependent (−145°..−98°
across ±10% band) — corrupting transmission phase and group delay on the advertised
non-vacuum path; magnitude |t| is untouched so all existing gates pass. Fix (both
files): exp(i·k0·(n_super·z_struct + (n_super−n_sub)·d_res)), byte-identical in vacuum;
add a t-phase gate to fdtd_nonvacuum_vs_tmm.py.

**C3-5 (P2). Oblique band mask admits grazing angles the CPML cannot absorb** —
`solve2d.py:337` (interacts with cpml.py:26). The mask sin θ<0.999 admits in-band
frequencies at θ up to 87°, where the vacuum-CPML round-trip echo reaches 0.1-0.5
field: at the validation's own geometry re-run at 76°, band=True points carry
|R0−TMM|=0.39 and R0+T0−1 up to +0.38. Fix: PML-quality-driven cutoff (predicted echo
< tol → sin θ ≲ 0.95) and/or auto-strengthened z-PML with warning.

**C3-6 (P2). Fixed 200·tau ring-down window truncates high-Q in-band poles** —
`solve2d.py:173-175` (same in solve3d, which also carries Lorentz). The DFT window is
keyed only to source bandwidth (predates the Lorentz/gain ADEs); a loaded-Q~600 line
(ordinary for QW excitons/phonon lines/near-transparency gain) rings past it: probe
shows max|ΔT0|=0.102 vs the shipped TMM oracle, silently (band mask checks excitation
only). Fix: window from material memory (~18/Γ_min) + a residual-energy runtime guard.

**C3-7 (P2). A_independent and per_region_absorption inflated by Re(n_super)** —
`solver.py:778` and :798. The incident-power normalization uses cos θ but omits n_super:
with a dense encapsulant (n_super=1.5, a mainline pipeline path via end_media_indices)
a device absorbing 0.2 reports 0.30, and D2's per-region deposition (feeding
reliability/electro-thermal drivers) is 1.5x high. The energy-closure check then fires a
*misleading* warning blaming diffraction. Fix: normalize by kz_s (reduces to cos θ in
vacuum); add a dense-superstrate case to the absorption validations.

### 3.2 Notable P3s (regrade in Stage 7)

- **CPML end-medium "impedance matching" scales sigma_max BY n instead of 1/n**
  (`cpml.py:32-33`, found by two independent agents): for every non-vacuum end medium
  the feature *strictly degrades* the PML (over-drives sigma by n², raising the
  discrete-reflection floor). Interacts with C3-5.
- CUDA cooperative-launch retry re-runs from step 0 on contaminated device fields
  (`kernels2d_numba.py:233-242`) — the finder graded the *current* code CORRECT only
  because CUDA errors are sticky in practice; the broad `except Exception` makes it
  fragile. Guard + zero-fill on retry.
- 1-D `_n_band_max` samples dispersive |n| at w_min only — under-sizes dz for
  dielectric-side/ENZ Drude layers whose |eps| peaks at the band's high end
  (`fdtd.py:156-167`); the MO sibling's 9-point sampler drops the background
  birefringent bound when drude_wp>0 (`fdtd_mo.py:216-221`).
- Periodic-face partner lookup silently PMC-walls unmatched faces on quantization
  straddling (`ngsolve_layered.py:515-523`).
- UPML tensor mass uses left-multiplied Λ·eps — mis-scales z-mixing off-diagonals inside
  the PML vs the symmetric Λ^½·eps·Λ^½ form (`solver.py:442`); benign for the built-in
  (diagonal-in-PML) cases, wrong for tensor media touching the PML.
- `weighted_objective` unknown 'sense' silently becomes 'max' (`inverse_design.py:48`).
- TMM backend hard-raises on negative theta that OpticalSpec permits and FEM handles
  (`tmm_reference.py:38-44`).
- Graded-tensor branch never wires the sub-tolerance off-diagonal snap its comment
  claims (`eps_assembler.py:77`) — harmless today (see T12 CORRECT below) but the
  comment lies.
- `faraday_deg` sign convention is pol-dependent and pinned only in an inline comment
  (`fdtd_mo.py:264-265`).
- graphene_sigma float64 sinh/cosh overflow → silent NaN at cryogenic T / optical freq
  (`core/graphene.py:59`); sheet-BC docstring states the Robin term sign opposite to the
  (correct) code (`solver.py:239,:456`).
- laser_gain docstring overclaims r=dN0/dN_th as "EXACT" — in the depleted regime the
  disrecommended W_p/W_p_th reproduces omega_RO to 0.5% while the recommended form is
  9-14% low (`laser_gain.py:115-117`).

### 3.3 Verified-correct highlights

- p-pol lossy-substrate transmittance factor (solver.py:738-739): the "missing"
  conjugation is an artifact of tangential-field variables; probe vs tmm over 20
  lossy/metallic oblique cases agrees to 1.1e-15. (Genuinely gate-blind regime though —
  worth adding one lossy-substrate case to oblique_ppol_vs_tmm.py.)
- Sheet-BC Robin term sign/scaling and graphene Im(sigma_inter) vs Falkovsky under
  exp(-iwt): code correct (docstring has the sign typo, P3 above).
- Gain ADE identities (G3 → g0=kappa·dN/(n·c·eps0·dw) at line center; passive-Lorentz
  reduction; S_st with ℏ and emission-positive sign): all verified.
- rfft→exp(-iwt) conjugation consistent across solve2d/solve3d/fdtd_mo; Im(r0) sign
  matches TMM for a lossy Drude slab.
- eps_assembler off-diagonal snap (T12): RELATIVE tolerance 1e-9 — snapped entries are
  provably unobservable for any nm-mesh-representable domain (needs a 0.25 m slab to
  reach even the FEM noise floor); the snap exists to keep NGSolve on the proven sparse
  path.
- R_flux interp-vs-FE asymmetry (T13): no coarse-order bias found.
- CUDA retry double-stepping (T11): current code safe (sticky-error semantics), fragility
  noted above.

## 4. Physics correctness — SOA, Lumenairy bridge, reliability (Stage 4)

**Method:** as Stages 2-3 — 16 targeted checks + 3 sweeps (soa, bridge-vs-installed-
lumenairy-5.20.0-source, reliability-vs-literature), Monte-Carlo/rate-equation probes
where decisive; combined refuter+re-deriver verification per P1/P2 claim; 31 agents,
≤5 concurrent. *(Run was paused 2026-07-05 evening at user request and resumed
2026-07-06.)*

**Headline: the never-audited code carries the audit's two worst findings** — a
sign-of-consequence inversion in reliability fatigue extrapolation and a wrong
polarization mapping at conical incidence in the bridges — plus a cluster of confirmed
P2s in the SOA's composability paths exactly where Stage 1's survey predicted
(traveling_wave's near-duplicate marcher loops). The foundational physics again holds:
detailed-balance, GVD sign, SBE signs, Berreman arg order (the exact 2026-07-04 trap —
clean), PMM per-pol energy bookkeeping (the historical trap — clean), BOR unit scaling,
FN constants all verified CORRECT.

### 4.1 Confirmed P1-class defects (final severity settled in §7)

**C4-1 (P1 finder / P2 verifier, CONFIRMED — found independently by TWO agents).
Norris-Landzberg frequency ratio is INVERTED** — `fatigue.py:72` (and the same inversion
repeated in the module docstring :10-12). Code uses (f_test/f_use)^m; the canonical N-L
model (Nf ∝ f^{+1/3}, JEDEC JEP122) requires (f_use/f_test)^m in AF=Nf_use/Nf_test.
Consequence is systematically NON-CONSERVATIVE for every accelerated test (f_test >
f_use): extrapolating a −40/+125 °C chamber test to a field profile overestimates field
cycle life 3.15-4.0x (13.2x for a 1-cycle/day profile). The existing validation GATE B
re-implements the same inverted expression, so it passes by construction. Fix is
one line + a direction-sensitive gate (AF<1 when only f_test is harsher).

**C4-2 (P1 finder / P2 verifier, CONFIRMED). Conical-incidence 'y' polarization maps to
the wrong physical polarization in the bridges** — `rcwa_backend.py:59,106-116,285-290`
and `berreman_backend.py:216-230`. At azimuth≠0, 'y' is mapped to lumenairy lab row 1
(incident tangential E_y), which is a phi-dependent s/p MIXTURE — not the s-polarization
that OpticalSpec documents and the FEM implements (rotated ŝ=(−sin φ, cos φ)). Probe: at
theta=30°, azimuth=45° on bare air|glass, the bridge returns R=0.0392 vs s-pol truth
0.0578 (32% low, silent); at azimuth=90° it returns exactly the ORTHOGONAL polarization
(R_p). The in-code comment claiming "'s'/'y'/'x' are fine at conical" guarded only 'p'.
Fix: extend the conical guard to 'y'/'x' (minimal), or synthesize true s/p by amplitude
linearity from the two lab rows (correct).

### 4.2 Confirmed P2 defects

**C4-3 (P2, CONFIRMED — two independent finders). Spont-spont beat variance is exactly
2x too large** — `ase_noise.py:130` + docstring :33. The code doubles Olsson 1989's
already-both-polarization form by multiplying by m_pol again (Monte-Carlo:
measured/code = 0.500 for m_pol=1 and 2; the sibling sig-spont term verifies correct
with the same machinery). Any sp-sp-limited budget reads SNDR ~3 dB low / ENOB ~0.5 bit
low and shifts optimal_drive_power.

**C4-4 (P2, CONFIRMED). Langevin marcher ASE has an O(dz) birth-slice deficit while
claiming exactness** — `traveling_wave.py:546-550` (docstring :497-502; FP twin
:785-805). Per-slice noise q·dz injected after slice gain accumulates to
n_sp·hν·(G−1)·[a·dz/(e^{a·dz}−1)] ≈ lnG/(2nz) deficit: 4.3% at 30 dB gain with nz=80
(8.4% at nz=40) → NF from the time-domain engine ~0.2 dB low. The validation gate probes
a 3.3 dB device where the deficit is 0.47% — inside tolerance by construction. Fix:
exact-emit variance scaling (expm1(a·dz)/(a·dz)), mirroring ase_output_psd.

**C4-5 (P2, CONFIRMED). eh_split accessor misindexing** (Stage-1 candidate, now
probe-confirmed) — `qd_gain.py:943-947` + consumers (:1048,:1063,:1072,
ase_noise.py:248-320). rho_GS/rho_ES hardcode the excitonic layout: an eh_split=True
model silently computes gain from misaligned ES-block occupations and a raw WL density
(~1e24) — saturation_curve gain 59% wrong, compression depth 2.5x wrong, no error.
step_slices_wdm raises for eh; these paths don't. Fix: guard or make consumers
layout-aware.

**C4-6 (P2, CONFIRMED). line_filter cancels the entire ES gain band** —
`traveling_wave.py:536-544` with `qd_gain.line_kappa_slices`. The dispersive correction
subtracts the full GS+ES flat gain but the polarization poles re-add only the GS band:
with sigma_pk_ES>0 and line_filter=True, measured gain collapses to −0.004 dB where the
correct value is +3.79 dB (probe), silently. Fix: subtract only the GS-matched flat gain.

**C4-7 (P2, CONFIRMED). Dual-pol TM depletion evaluated at the TE frequency** —
`traveling_wave.py:675`. With tm_peak_shift≠0 the TM field amplifies at nu_tm but
depletes carriers at nu — wrong groups by the lineshape ratio; probe: 97% of TM gain
compression missed (6.52 dB vs 6.17 dB exact-pairing oracle) and photon number not
conserved. Fix: advance carriers with the existing step_slices_wdm per-channel pairing.

**C4-8 (P2, CONFIRMED). Innolume calibration maps the datasheet's NET −3 dB bandwidth
onto the MATERIAL gain FWHM** — `calibration.py:70-84` (+GATE C tautology). The shipped
"device-matched" model has a net amplifier bandwidth of ~16.5 nm vs the datasheet 60 nm
(3.6x narrow): WDM channels at 1280/1340 nm get +12-13 dB where the real BOA delivers
≥32 dB, while report['bandwidth_nm']≈60 falsely claims a match. Fix: report both
quantities distinctly and co-fit ES strength / inhomogeneous profile.

**C4-9 (P2, CONFIRMED). TDDB stress uses |mean Ez| — exactly zero for split-gate
profiles** — `tddb.py:117` (statistic from `electrostatics_fem.py`). For the advertised
patterned-gate case (±V split gate validated in electrostatics_fem GATE B), the adapter
reports E_ox≈0 → time-to-breakdown overstated exponentially; generally understates the
percolation-driving peak field for any nonuniform profile. Fix: per-layer |Ez| max (or
beta-weighted hazard average) instead of the signed mean.

### 4.3 Refuted claim (recorded so it is not re-raised)

- "Graded/tensor eps entries on a layer with inclusions silently delete the lateral
  pattern in the bridge translators" — REFUTED: no built-in emitter can produce that
  state. LayeredOpticalBuilder raises for inclusion+semiconductor layers (audit guard
  BI-1) before any solve, and assemble_eps only emits gridded fields for semiconductor
  region alignments, so a patterned layer's key can never carry a graded field through
  collapse_regions_to_layers. Defensive hardening only (P3 at most).

### 4.4 Notable P3s (regrade in Stage 7)

- SCH transport reservoir advanced by explicit forward Euler — rings for dt>tau_t,
  diverges for dt>2·tau_t, inconsistent with the deliberately-implicit electrical-RC leg
  two lines above (`traveling_wave.py:250`).
- A_eff = π·w_s·w_f is the 1/e²-ellipse-area convention (2x the peak-intensity area) →
  tau_eff 117 vs 58 ps, f_3dB 1.37 vs 2.73 GHz; plus ~1.44x from using output- instead
  of internal-compression power (`calibration.py:212-264`); staged fit has fixed-count
  damped loops with no convergence check (:150-165).
- Black-equation current-exponent regime labels swapped vs literature (n≈2 is
  nucleation-, n≈1 growth-limited) (`em.py:4`); HCI m=2/3 default mis-attributed
  (Takeda-Suzuki has no I_sub power; Hu's lucky-electron m≈2.9-3) (`hci.py:6,41`).
- biaxial_stress docstring sign-case inverted vs its own correct formula
  (`fatigue.py:33`).
- BerremanLayeredSolver(absorption=True) unguarded for OOP-tensor-at-oblique (crashes
  where the sibling entry point degrades gracefully) (`berreman_backend.py`).
- BOR: raw dict units (rad/µm, unsorted arrays) undocumented; per-layer lambda wrappers
  defeat lumenairy's modal-LRU dedup; eps_profile radius conversion has zero
  r-dependent test coverage (`bor_backend.py:140-143` + validation).
- EMT order-0 path drops lumenairy's validity diagnostic by not forwarding
  period/wavelength (`emt_screen.py`).
- Direct-tunneling uses the exponent-only Schuegraf-Hu form (disclosed; 2-32x below
  full S-H sub-barrier) (`leakage.py:65`).
- FEM p-pol r is the raw lab tangential ratio (=−r_p + mesh-z=0 reference phase); only
  magnitudes were ever validated vs tmm — the bridges' r_factor=−1 assumption rests on
  an unvalidated absolute phase (`rcwa_backend.py:259-264` docstring).

### 4.5 Verified-correct highlights

- Detailed-balance tau ratio incl. mu prefactors: dark ES/GS odds ratio = exp(−dE/kT)
  to machine precision; conservation is provably blind to a flipped ratio but GATE D's
  independent Boltzmann oracle would catch it (4x error) — good gate design.
- GVD residual-phase sign: anomalous beta2<0 correctly compresses an alpha-up-chirped
  pulse; SBE +iγ sign and Coulomb-ON Bernard-Duraffourg transparency correct; BPM
  hot-centre focuses (thermal-lens sign correct).
- berreman_jones_1d(layers, n_substrate, n_superstrate) arg order at both call sites
  CORRECT (the 2026-07-04 oracle trap did not recur); functional/class double-solve
  consistent.
- PMM bridge R_eff/T_eff rows include cross-polarized output orders (per-pol
  order-summed, no phantom-A) — the historical per-pol energy trap does not bite here.
- _p_basis_conversion r_xx=−r_p verified for all theta incl. Brewster/TIR/absorbing
  substrate (principal-branch cos_t consistent with lumenairy's convention).
- BOR m→µm scaling single-applied everywhere it must be; FN A/B constants match the
  canonical 1.54e-6/(m_r φ) and 6.83e9·√m_r·φ^1.5 forms (h-vs-ℏ correct).

## 5. Seams & feature gaps (Stage 5)

**Method:** the 16 seam targets from §1.3/§1.6, each proven or refuted with the smallest
*asymmetric* runnable fixture (numpy + tmm + installed lumenairy 5.21; no NGSolve/DEVSIM),
determining which side of each seam is wrong; combined-verifier pass on every P1/P2.
31 agents, ≤5 concurrent. This stage upgraded both Stage-1 P1 candidates to
**fixture-proven CONFIRMED P1s** and confirmed eleven P2s (C5-2(b) plus C5-3..C5-12);
of the remaining Stage-1 candidates, one was refuted outright and three were downgraded
to P3 (§5.3).

### 5.1 Confirmed P1 defects

**C5-1 (P1, CONFIRMED; independently re-proven by two agents). All three lumenairy
bridges vertically FLIP every graded eps(z) profile** — `rcwa_backend.py:198`,
`pmm_backend.py:118`, `berreman_backend.py:127`. The contract was hand-derived from four
independent sources (geometry z-up stacking, assemble_eps ascending-z output, slicer
normalization, and the installed lumenairy 5.21 source: RCWAStack/PMMStack/berreman all
consume superstrate-first): a superstrate-first consumer MUST reverse `slice_eps_field`'s
ascending (substrate-first) slabs. `tmm_reference.py:228` does; the three bridges don't.
Fixture (asymmetric lossy graded layer, air|graded|glass, 1.31 µm): the bridges match
the FLIPPED-profile oracle to <1e-15 — reported R=0.062 where the truth is R=0.330
(another fixture: |ΔR|=4.0e-2) — while A=1−R−T and per-layer absorption closure remain
exactly self-consistent (the flipped stack is itself a valid stack: closure is blind).
This is the flagship ENZ accumulation-layer path via run_pipeline + any bridge solver.
Gate-blindness is total: the existing bridge GATE B feeds both solvers the same
pre-sliced stack (translator never runs); GATE F asserts only closure; graded_tmm_vs_fem
uses a palindromic profile with equal end media (reversal-invariant); and CI never
installs lumenairy anyway. **Fix is one word per bridge** (`reversed(...)`), plus an
asymmetric-lossy regression gate through the design/eps_by_region path. Related P3: the
`slice_profile` docstring (layered.py:85-87) promises "SAME order as z_m" while the code
always normalizes to ascending — the false mental model that plausibly seeded the bug.

**C5-2 (P1 + P2, CONFIRMED). FDTD seam silently zeroes the bias modulation — or crashes
blaming the material — for exactly the inputs the pipeline exists to carry** —
`fdtd_seam.py:104-109`, :254-268, :534-541. (a) P1: gridded (graded) and uniform-tensor
eps_by_region entries are silently replaced by nominal material eps on all three FDTD
seam paths: with an EffectEpsMap + thermal/PCM/LC modulation, every bias returns
IDENTICAL R/T (probe: FDTD |ΔR|=0.0 exactly vs TMM 7.7e-2 on the same dict), no warning.
(b) P2: for the flagship DrudeOptical gated-ITO region the nominal fallback calls
Material.eps(λ) without n_m3 and crashes with "DrudeOptical.eps requires n_m3" — blaming
the material definition while the seam is holding the bias eps it just discarded. Fix:
mirror the TMM peer (raise on tensor, slice graded with lateral-uniformity assert, wire
`graded_fdtd_layers` in).

### 5.2 Confirmed P2 defects

- **C5-3. Persistent cache serves stale physics after a material retune** —
  `cache.py:62-94`: _key hashes material NAMES only while every non-FEM backend
  re-derives eps from design.materials at solve time. Probe: retune sio2 eps 2.25→4.0
  under the same name → cache HIT returns R=0.179 where the truth is 0.059. Fix: hash
  the eps content per material (sampled at the request wavelength band) into _key;
  include Feature.priority; bump _SCHEMA.
- **C5-4. per_region_absorption keying is fragmented across backends at the same seam**
  — TMM emits `slab_<i>`, RCWA/Berreman design-layer names, FEM region labels, PMM/BOR
  nothing; probe: `absorbed_fraction(row,'ito_top')` raises KeyError after a backend
  swap that should be a drop-in. TMM is the wrong side (it has the design and returns
  seam-internal keys). Fix: re-key TMM to design layer names (sum graded slabs) and tag
  the convention on OpticalResult.
- **C5-5. extra_fields bypass the FieldLift and can land TRANSPOSED silently** —
  `bridge.py:103,110-112`: a 2D (Nx,Nv) T/E grid merges raw against the lifted 3-D n;
  with the shipped defaults (grid_n_x=256 == ny_sym=256 on the SeparableXY path) numpy
  broadcasting maps the thermal x-axis onto the optical y-axis — probe: output ==
  exact x↔y transpose of truth, silently. Fix: normalize extras through the lift.
- **C5-6. Cache carries no solver identity or R/T-convention tag** — `cache.py:26,87-94`:
  swapping FEM→RCWA over the same cache path (docstring's own drop-in usage, default
  tag='') serves specular numbers as order-summed ones: probe R=0.107 served vs 0.177
  correct on a diffracting cell. Fix: rt_convention field + solver fingerprint in _key,
  _SCHEMA bump. (Also fixes §1.3-#10.)
- **C5-7. Every FDTD OpticalSolver entry point ignores design.optical** —
  `fdtd_seam.py:335-360` + sweep paths: theta/azimuth/pol/incidence_side silently
  discarded (probe: 30° spec returns the θ=0 answer, R off 10-35%; bottom-incidence
  returns the top answer where every sibling raises). Companion P2: the oblique 2-D/3-D
  solvers silently DROP chi2/raman/gain terms the 1-D path raises on — an amplifying
  stack at 20° returns BIT-IDENTICAL passive R0/T0. Fix: one guard helper mirroring the
  sibling pattern; extend the oblique Lorentz raise to the full term set.
- **C5-8. LumenairyStackSolver fabricates px=py=λ for STRUCTURED stacks** —
  `rcwa_backend.py:369-370`: the "period is irrelevant when uniform" justification only
  holds on the unstructured branch; a 600 nm grating with defaulted period solves at a
  wavelength-sized fake period (R=0.061 vs 0.191 correct, phase off 100°+). Fix: raise
  for structured stacks with period≤0.
- **C5-9. arrhenius_af blocks REL8's negative Ea — and the natural workaround silently
  inverts the AF** — `mttf.py:30-31` vs `hci.py:40-43`: the formula is exact for signed
  Ea (matches hci to 2.2e-16); only the guard blocks it, and passing |Ea| yields the
  reciprocal (0.266 vs 3.77 correct — 14.2x silent). Fix: allow signed Ea with
  documented sign semantics.
- **C5-10. carrier_heating silently swaps calibrated callables for M_E and 1e14 rad/s**
  — `carrier_heating.py:147-149`: with ALL heating knobs off, R(t) differs from the
  calibrated baseline — the module's own documented byte-identical off-switch collapse
  is violated. Fix: require explicit m0_kg/gamma0 when the DrudeOptical carries
  callables (a band-averaged m*(n) callable cannot be inverted safely — evaluating it at
  n_m3 double-counts the Kane filling).
- **C5-11. collapse_regions_to_layers silently drops drifted region keys** —
  `layered.py:202-211`: a renamed/typo'd subregion key makes the layer revert to nominal
  eps in all three bridges, bit-identical to passing no eps dict (probe: A=0.0026 vs
  0.0723 correct); the FEM side of the same seam raises loudly for exactly this drift.
  Fix: third bin — warn/raise on unclaimed non-superstrate/PML keys (keeps the tested
  partial-coverage contract).
- **C5-12. Two-point end-media dispersion guard false-passes on in-band features** —
  `pipeline.py:132-145`: equal band-edge n with a resonance inside freezes a wrong
  band-centre index for the whole sweep (probe: frozen n_sub=2.449 vs true 2.0 at the
  edges). Fix: sample every sweep wavelength + λ_c.

### 5.3 Refuted / downgraded (recorded so they are not re-raised)

- **layer_absorption layout mismatch RCWA-vs-Berreman (§1.3-#3): REFUTED as a defect.**
  Both bridges' reductions are correct against the installed lumenairy 5.21 return
  shapes; the (2,2) ambiguity heuristic in the RCWA bridge is real but currently
  unreachable with wrong results (P3 hardening: replace the shape heuristic with an
  explicit contract).
- Cache×solve_sweep mutual exclusion, 2D 'y'==z axis contract, EpsField unit tag: all
  verified real but graded P3 (defensive/latent; no built-in path produces the failing
  input today). Fix sketches recorded in the Stage-5 result archive.

## 6. Conventions, organization, performance & memory (Stage 6)

**Method:** the 15 target batches from §1.6 — each claimed instance re-verified at
source, perf claims MEASURED with timing probes where cheap, accuracy risk stated per
item; P2-graded (correctness-adjacent) items adversarially verified. 22 agents, ≤5
concurrent. Convention: pure opportunities/hygiene = P3 recommendations; only items that
can mask wrong physics were eligible for P2.

### 6.1 Confirmed P2s (correctness-adjacent)

- **C6-1. lc_director_2d certifies unconverged solutions** — `lc_director_2d.py:203`:
  convergence judged by last-UPDATE size, not error; true error = res/(1−ρ) with
  measured amplification x200 (nz=41) to x4340 (nz=161). The shipped default grid masks
  0.034° at res=3e-6; fine grids certify ~1.5° errors as success='ok'.
- **C6-2. ssac_admittance(v_ac≠1) mis-scales C,G by 1/v_ac** (upgraded from Stage 2's
  P3 by the fail-loud-rule lens): both circuit-source setups hardcode acreal=1.0 while
  the docstring presents v_ac as "the excitation scale" (`ac_analysis.py:59-81`).
- **C6-3. Sweep accepts duplicate bias labels and silently collapses them** —
  `sweep.py:19-22`: the label-keyed fields dict keeps only the LAST duplicate's carrier
  field for all its rows; the repo fail-louds on every sibling degeneracy (wavelength
  collision, duplicate gate biases, empty sweep).
- **C6-4. LC director BVP tol floor** — `lc_director.py:563,:823`:
  tol=max(rtol, 3e-2) — the advertised rtol=1e-6 default is never honored; the knob can
  only loosen. Undocumented.
- **C6-5. Stale docs that would mislead a physical choice**: `sp_carrier.py:27-35`
  claims parabolic-only S-P while the shipped code threads a fully-self-consistent Kane
  Newton; `core/graphene.py:18-20` claims the FEM sheet BC is a follow-on while
  solver.py:454-466 implements and validates it.
- **C6-6. run_all has no SKIP category** — `run_all.py:148`: 22 validation scripts
  deliberately exit 0 with a SKIP banner when CUDA/jax/ngsolve/lumenairy are absent, and
  every rc==0 is tagged PASS — never-executed physics gates read green in the
  machine-checkable summary.

### 6.2 Measured performance opportunities (results-identical; P3 recommendations)

- **Cache autosave is O(N²) full-store rewrite per miss** (`cache.py:134-135` +
  `io/store.py:80-94`, no append path; HDF5 mode-'w' truncate, Zarr rmtree-first).
  Measured: 400-miss HDF5 sweep spends 9.68 s on autosave vs 0.04 s for one final flush
  (240x); 120-miss Zarr = 28.6 s vs 0.40 s (70x; extrapolates to ~28 min pure I/O for a
  1000-point sweep). Packing entries as ONE (N,12) dataset + (N,41) key array is
  100-250x faster on both save and reopen, bit-identical, needs _SCHEMA 3→4.
  **Fixer hazard (P1-class if done naively):** an append-mode fix that doesn't truncate
  after a load-side schema discard resurrects stale mis-keyed entries under a fresh
  schema stamp — extend GATE D with a flush-into-stale-file + reopen leg first. Also:
  the "(cheap, crash-safe)" docstring is inverted at scale — every rewrite is a window
  where a crash destroys the whole store.
- **pipeline retains every bias's CarrierField for the whole run** (`pipeline.py:98-163`,
  sole read is per-bias): free after use → O(n_bias·grid) → O(grid) peak memory.
- **lc_director_2d pure-Python triple-loop Gauss-Seidel**: vectorized red-black or
  sparse Newton ≈ 50-200x (`lc_director_2d.py:64-192`).
- **Burstein-Moss KK**: precompute the fixed-grid parity kernel and matmul the 64 dalpha
  rows (~30-60x on the dominant per-bias cost); cache the bias-independent qw.solve(0.0)
  (`electroabsorption.py:331-342`).
- **topology_opt/inverse_design**: jit with beta as traced arg (removes 4 of 5 XLA
  recompiles); FDTD vacuum-reference caching (~2x on repeated seam solves); kernels3d
  per-step temporary preallocation (~10-25%).
- **LIDT bisection**: memoize absorbed(T) on the fixed grid (~25x fewer TMM solves);
  Korhonen banded Jacobian (10-100x); Miner vectorization (`lidt.py:70-106`, `em.py`).
- **SOA batch**: calibration RK4 vectorize across P_in (~40x, called ~14x per fit);
  dict-keyed lineshape cache (WDM thrash); zresolved Picard → interpolation; warm-start
  saturation sweeps.
- **solve_fem diagnostics opt-out** (three diagnostic integrations + two FE
  interpolations per solve even when only R/T is read) and Berreman absorption
  double-solve consolidation (~2x).
- REFUTED perf claims (no change needed): cache._key per-call eps SHA1 and per-wavelength
  VoxelCoefficient rebuild are NOT hot (T3 verified the existing structure fine).

### 6.3 Convention/organization batches (verified; P3)

- **Constants single-sourcing**: KB_EV_K duplicated in 8 reliability modules, H_PLANCK
  in 5 soa files (one raw literal), inline Q_E literals (electroabsorption.py:284,
  stress_migration.py:90), sbe.py hardcodes m_e/c, exp(-iwt) convention string x3 in
  core, F12 Aymerich-Humet python/DEVSIM twins without shared coefficients.
- **lumenairy floor consolidation**: raise pyproject floor to the max backend
  requirement (BOR 5.16.0); single VERSION_FLOORS + robust parse (crashes on
  '5.21.0rc1', x3 copies); wire lumenairy-native conical PMM (stale raise at
  pmm_backend.py:155-157); stop reading private `_layers` without a version ceiling.
- **Private-API promotion before the next refactor**: fdtd_seam._cell_axes/
  _layer_eps_cell → public rasterizer module (RCWA bridge depends on them); thermal_fem
  underscore helpers → public forms API; stop re-exporting ~25 fdtd_nd underscore names;
  rcwa_backend shared helpers → lumenairy_bridge/_common.py (ends the bor_backend
  copy-paste island); move FDTDLayer out of optics/fdtd.py.
- **Stale-docs batch** (beyond the two P2s): dynameta/__init__ module map omits
  BOR + symmetry (version 0.5.0 unmoved across three feature drops); em.py/hci.py claim
  shipped drivers as follow-ons; reliability/__init__ REL-list stale; solve2d.py
  lateral_eps_inf shape doc; cpml.py:19 false invariant; interfaces.py:93 default-solver
  location.
- **Test-infra recommendations**: wire `--tier smoke` into CI; add skip-gated DEVSIM
  pytest smokes (dc_solve/ac_analysis/transient/physics_* currently have zero pytest);
  fix run_all's gated-script regex discovery (raise/assert-gated scripts silently never
  run) + add a SKIP category (C6-6); curate the 36 qd_soa validations into the smoke
  tier; surface CI skip counts.

## 7. Verification sweep, completeness check, final grading (Stage 7 + 7b)

**Method:** (a) severity REGRADES of the five findings earlier stages flagged; (b)
bidirectional prove-outs — five agents each attacking a load-bearing CORRECT verdict
(the mirror image of defect verification); (c) a completeness critic over the whole
report; (d) a 5-agent gap-fill stage (7b) closing the critic's coverage gaps. 16 agents,
≤5 concurrent. Final severities in this section are the authoritative grades; where they
differ from earlier sections, this section wins.

### 7.1 Regrades (all five re-examined with probes)

| Finding | Was | Final | Basis |
|---|---|---|---|
| Semi-semi interfaces Potential-only (devsim_layered.py:394-395) | P3 | **P2** | Silently carrier-blocking junction on the DEFAULT 2D builder for a plain documented configuration (p-on-n stack); the 3D twin hard-raises for exactly this with a "fail loudly" comment — silent twin-contract violation. Fix: 2D analogue of the 3D guard. |
| Impact-ionization 3D tet quadrature (impact_ionization.py:97-114) | P3 | **P2** | Probe on a real Delaunay tet mesh: I_sub 1.8-3.1x LOW in the low-multiplication HCI regime → REL8 lifetime 1.6-2.1x OPTIMISTIC. Stacked3DSpec(physics='bipolar_dd') is a first-class validated path satisfying the module's documented precondition; no dimension guard while GATE D advertises <5% convergence. |
| LC branch tie-break energy (lc_director.py:520-522) | P3 | **P3 upheld** | Defect real (and worse than reported: prefers untilted for V_th<V<2.88·V_th vs the exact Deuling branch) but *decision-inert on every built-in path*: secondary sort key behind the tilt score (exact float64 ties impossible between distinct branches); vanishes identically where energy is the sole key; ordering-equivalent to the true potential under the default field_model; end-to-end probe emits the correct branch across the whole claimed window. Fix the formula anyway when touched. |
| CPML end-medium sigma×n (cpml.py:32-33) | P3 | **P3 upheld** | Physics claim CONFIRMED (correct scaling is 1/n; probe: echo floor degrades 2-14x for n=1.5-4, and the 1/n law holds the floor flat) but consequence bounded empirically: end-to-end R0/T0 shift ≤2.8e-4 absolute on every reachable case — 50-100x inside shipped validation tolerances. The claimed interaction with C3-5 was refuted. Fix the scaling; it is strictly beneficial. |
| SCH reservoir forward Euler (traveling_wave.py:250) | P3 | **P3 upheld** | Instability confirmed (divergence 1.4e68 in 400 steps at dt=2.5·tau_t) but unreachable: shipped paths run dt/tau_t≈5e-4, and crossing the threshold requires a degenerate configuration (nz=1 on a ≥1.7 mm device). Add the exact-exp update (already in the same file) when touched. |

### 7.2 Bidirectional prove-outs (attacking CORRECT verdicts)

All five verdicts SURVIVED genuinely adversarial re-derivation + probing: the hole
Scharfetter-Gummel current (incl. derivative strings and kahan3 ordering), the PMM
bridge cross-pol energy bookkeeping (re-verified against installed lumenairy 5.21.1
source + a polarization-mixing probe), the fdtd_nd gain-ADE coefficients (z-transform of
the exact discrete recursion, twins byte-identical, saturation path clean), the p-pol
lossy-substrate transmittance (extended to inhomogeneous-wave cases), and the
layer_absorption layouts (lumenairy source re-read + asymmetric 2-layer probe through
both bridges). One prove-out ran without the safety-classifier sidecar; its conclusion
only re-confirms two prior independent verdicts and its cited evidence was spot-checked.
Zero CORRECT verdicts overturned — combined with 21-of-24 defect confirmations upheld in
the two-lens passes, the report's verdicts appear stable in both directions.

### 7.3 Completeness critique → gap-fill (Stage 7b) results

The critic's structural finding was correct: the stage plan had no dedicated physics
pass for the declarative model layer, the top-level consumers, the FEM symmetry
reduction, adjoint gradients, or the validation tier's own gate quality. A 5-agent
gap-fill closed each:

- **materials + core/effects + n_to_eps** — 10 model families hand-verified CORRECT
  (DrudeOptical signs, KK transform, Pockels, gyrotropy signs vs Landau-Lifshitz,
  Bruggeman, LC tensor, Burstein-Moss, intersubband TRK m0-cancellation, QCSE EA
  wiring, composition semantics). **One new P2 (C7b-1): `KaneOpticalMass` default is
  NOT the Kane optical mass** (`scattering.py:33-39`): the exact degenerate Kane result
  is m0·√(1+4αγ_F) = m0·(1+2αE_F); the code returns m0·√(1+2αγ_F) — the standard Kane
  alpha silently HALVED (leading-order enhancement ratio 1.987, hand-verified
  algebraically). With literature ITO α=0.4191 eV⁻¹: m_opt 8.6-22% low over
  n=1e26-2e27 m⁻³ → wp² up to ~24% high → **λ_ENZ ~4-12% short, silently** — and the
  gate + unit test compare against the SAME halved formula (circular, mislabeled "the
  exact bulk Kane DOS-mass closure"). Worse, the same α symbol means true-Kane-α in
  schrodinger_poisson and carrier_heating but 2x-Kane-α here — a mutually inconsistent
  carrier-vs-optics mass model in one pipeline. Fix preserves legacy numbers exactly via
  α_legacy=2α_Kane. Two P3s: MatthiessenGamma ionized-impurity channel scales the RATE
  by the MOBILITY mass law (wrong direction vs Born probe, ~3.5x trend distortion);
  Elliott 2D Sommerfeld exponent doubled (uses E_b where the 3D Rydberg E_b/4 belongs,
  ~20% continuum over-enhancement).
- **analysis/results/transient_optics/viz** — core numerics all CORRECT (gate_cv
  reproduced analytic accumulation to 1.6e-15; store round-trips probed on h5py AND
  zarr; transient t→∞ limit consistent). Two doc P3s (resonance_dip docstring example
  numerically wrong; enz_reflector_stack depth-axis orientation undocumented).
- **FEM mirror-symmetry reduction (c638181)** — the PEC/PMC wall mapping was
  hand-derived (x-pol → PMC on y-normal mirror walls; y-pol swaps; pseudovector H signs)
  and the code's mapping is **CORRECT**; eligibility guards reject the right classes.
  Four P3s: Mesh3DSpec.symmetry string unvalidated (typo silently builds the full
  cell); the sweep fast path builds a dead wrong-BC reuse space under symmetry (silently
  losing the promised amortization); RegularPolygon.rotation_deg ignored by the
  symmetry detector; stale metal-skin comment claims a guard that does not exist.
- **Adjoint gradients (inverse_design + topology_opt)** — **CORRECT**: jax.grad vs
  central finite differences clean on a runnable 2-D probe; no
  stop_gradient/custom_vjp anywhere in the chain; filter/projection/scatter links all
  differentiable as intended.
- **Tautological-gate sweep (47 smoke scripts + reliability tier)** — one new **P2
  (C7b-2): reliability_tddb GATE B is fully tautological** (both the solve_ivp damage
  integration and the "piecewise-analytic reference" derive from the same module
  outputs — the module's contribution cancels identically). Plus a systematic
  classification: 5 more fully/partially tautological gates (fatigue GATE A/C,
  switching GATE B, drude_matthiessen_kane GATE KANE — the C7b-1 enabler,
  qcse part-4, scattering_link A-FIT re-fits the reference's own output), 3
  calibrated-anchor legs that pass by construction, 2 docstring-advertised pins that
  don't exist in code (S-P Gauss integral; sp nonparabolic α=0 byte-identity), and —
  most usefully — confirmation that four SELF-CONSISTENT-ONLY gates are the SOLE pins
  on exactly the quantities this audit found defective (Kerr magnitude C3-2,
  kane_mass_of_Te Sommerfeld C2-2, lc_director_2d convergence C6-1, and the C4-9
  adapter which has ZERO validation coverage).

### 7.4 Final severity decisions (auditor's grading)

- **C4-1 Norris-Landzberg inversion: final P1.** The verifier's P2 rested on "requires
  f_test≠f_use" — but extrapolating between frequencies is the function's entire
  purpose; the path is the shipped fatigue-extrapolation workflow, the error is
  systematically non-conservative, and the gate is tautological (§7.3 confirms the
  script's other gates are too).
- **C4-2 conical 'y' polarization mismap: final P2 (high).** Silent 32%-to-orthogonal-pol
  errors, but azimuth≠0 is a configured minority path, not the default — one step below
  P1 by the rubric, top of the P2 queue.
- **C5-1 bridge graded-profile flip and C5-2 FDTD-seam modulation drop: final P1**
  (fixture-proven, silent, flagship ENZ-modulator path).
- Critic's remaining grading disagreements, adjudicated: CPML ×n and LC tie-break stay
  P3 on measured-consequence grounds (§7.1 — empirical bound beats failure-class
  pattern-matching); 1-D `_n_band_max` w_min-only sampling stays P3-contested (the
  worst case is bounded by √(eps_inf/|eps(w_min)|); recommend the trivial fix — sample
  the band like fdtd_mo — regardless); the "2D y==z axis contract" P3 and C5-5 (P2) are
  distinct findings (general hygiene vs the specific exploitable transposition, which
  IS graded P2).
- §5 tally corrected: eleven P2s counts C5-2(b) + C5-3..C5-12; §5.3 records one
  refutation AND three downgrades.

### 7.5 Final tallies and the priority fix list

**Confirmed defects: 3 P1, 23 P2** (plus ~45 P3 defects/hygiene items and the §6.2
performance register). Verification: every P1/P2 passed adversarial verification
(two-lens in Stages 2-4, combined-lens in 5-6, regrade/prove-out in 7; C7b-1 was
single-lens but its algebra was independently hand-verified and it is corroborated by
two other stages' internal-inconsistency evidence; C7b-2 is source-reading, checkable
by eye).

**P1 (fix first — all are one-line to few-line fixes):**
1. Bridge graded slab-order flip — `reversed(...)` at rcwa:198 / pmm:118 / berreman:127
   (C5-1) + asymmetric-lossy regression gate.
2. FDTD seam graded/tensor drop — mirror the TMM peer's slice/raise (C5-2).
3. Norris-Landzberg frequency-ratio inversion — flip the ratio at fatigue.py:72 +
   direction-sensitive gate (C4-1).

**P2, ranked by blast radius:** KaneOpticalMass halved alpha (C7b-1, flagship ENZ
accuracy); conical 'y' mismap (C4-2); cache staleness pair (C5-3 material-name keying,
C5-6 convention/solver tag); Kane ⟨m*(Te)⟩ Sommerfeld factor (C2-2); FEM probe-grid
aliasing (C3-1); Kerr convention (C3-2); fdtd_nd t0 phase de-embed (C3-4);
A_independent/pra ÷Re(n_super) (C3-7); FDTD seam ignores design.optical + oblique
term drops (C5-7); 2D lateral interfaces (C2-3) + semi-semi guard (§7.1); bipolar-3D
phi_bi vs net_doping_expr (C2-1); II 3D quadrature (§7.1); SOA cluster (C4-3..C4-8);
TDDB |mean Ez| (C4-9) + tautological GATE B (C7b-2); extras transposition (C5-5);
collapse_regions silent drop (C5-11); period fabrication (C5-8); MO grid sizing (C3-3);
CPML grazing mask (C3-5); ring-down window (C3-6); arrhenius signed-Ea (C5-9);
carrier_heating callable swap (C5-10); dispersion false-pass (C5-12); lc_director_2d
convergence certification (C6-1); ssac v_ac (C6-2); Sweep duplicate labels (C6-3); LC
tol floor (C6-4); stale-doc P2 pair (C6-5); run_all SKIP category (C6-6).

### 7.6 Residual scope (what this audit did NOT establish)

- No NGSolve/DEVSIM runtime execution: all FEM/DEVSIM findings are static analysis +
  pure-numpy/tmm probes. Runtime-only classes (DEVSIM string syntax on unexercised
  branches, NGSolve version behavior, exception paths) are unaudited; the `make
  validate` tier remains the coverage instrument there.
- Bridge behavior verified against installed lumenairy 5.21 only; behavior at the
  declared pyproject floor (5.14.2) was never assessed — treat the floor bump (§6.3) as
  correctness work, not hygiene.
- Stages 2-6 audited the Stage-1 worklist plus per-stage open sweeps and the 7b
  gap-fill; coverage completeness is still inherited from that survey — absence of
  findings in un-targeted code is weaker evidence than the targeted verdicts.
- The §1.4 capability-matrix gaps stand as the feature-gap register (items there are
  missing capabilities, not defects, except where confirmed above).

---

## 8. Bridge-expansion opportunities vs current lumenairy (post-audit addendum, 2026-07-11)

**Question:** lumenairy's RCWA/PMM/Berreman engines improved substantially through
v5.20-5.21 — does the DynaMeta bridge need expanding to use them?
**Answer: yes, with sequencing.** Grounded by direct comparison of the bridge's actual
call surface against the installed lumenairy 5.21.1 source (both re-checked on
2026-07-11).

### 8.1 Capabilities lumenairy ships that the bridge does not use

1. **Native conical PMM** — `PMMStack.set_source(theta=, phi=)` with a full conical
   cascade exists (elements/pmm/stack.py:491-555, `_solve_conical`), but the bridge
   still hard-raises for |phi|>0 at `pmm_backend.py:155-157` ("not supported by
   PMMStack; use RCWA or FEM"). Stale since 5.20. Wiring is a few lines — but C4-2
   (conical 'y' polarization mismap, §4.1) must land first since it applies here too.
2. **Threaded RCWAStack** — worker pool with per-worker BLAS pinning
   (elements/rcwa/stack.py:1435-1491, ~8x on sweeps) is not exposed through
   `make_lumenairy_rcwa_solver` / `solve_sweep`. Pure speed, zero physics risk; the
   cheapest win in this section.
3. **PMM per-layer absorption / internal fields** — `RCWAStack.internal_field`
   (stack.py:543+) and the PMMStack v2 internal-field machinery exist; the PMM bridge
   factory has NO absorption kwarg at all and BOR returns neither absorption nor phase
   (§1.4-8). Bringing PMM/BOR to OpticalResult parity feeds the D2
   per-region-absorption → reliability/electro-thermal chain directly, and should ride
   on the C5-4 keying-unification fix.
4. **PMM2DStack / PMM2DStackPure** — 2-D crossed-patterned PMM (elements/pmm/stack2d.py,
   stack2d_pure.py — the pure no-Fourier-floor engine) is not bridged at all. Highest-
   value physics expansion for the library's own device class: metallic patch antennas
   on MIM stacks are exactly where PMM's wall-resolved basis beats RCWA Fourier
   convergence. Moderate effort (`design_to_pmm2d_stack` mirroring the RCWA
   translator) — and MUST bake in the `reversed()` slab-order fix (C5-1) from day one.
5. **JAX-differentiable RCWA/PMM twins** — the bridge already has the wrapper pattern
   (`berreman_design.py` wraps the Berreman JAX twin); the newer differentiable
   `rcwa_jones_2d` / `RCWAStack` / `pmm_jones_2d` are unbridged. Would give
   semi-analytic gradient design of 2-D patch geometry with carrier modulation — far
   cheaper per iterate than the JAX-FDTD topology-opt path for layered/periodic
   structures.
6. **Berreman OOP-tensor-at-oblique** — fixed in the lumenairy 5.21 line; the bridge
   still carries the unguarded `absorption=True` crash path on OOP tensors at oblique
   (§4.4 P3). Lifting that guard makes tilted-LC director stacks at oblique first-class
   through Berreman.

### 8.2 Sequencing (fix before expand)

The bridge currently flips every graded profile (C5-1, P1), mismaps conical 'y'
(C4-2, P2), and sits on stale, never-tested version floors (§6.3; nothing below the
installed 5.21 was ever exercised). Expanding on that foundation propagates wrong
physics into new paths. Order:
1. Land the bridge P1/P2 fixes + asymmetric-profile regression gates (§7.5 list).
2. Bump the pyproject floor to the version actually supported (≥5.20, realistically
   5.21 given items 8.1-1/4/5); single VERSION_FLOORS constant + robust parse; stop
   reading private `_layers`.
3. Quick wins: items 1-3 above (conical PMM, threading, PMM/BOR absorption+phase).
4. Capability campaign: items 4-6 (PMM2D bridge, JAX design twins, Berreman OOP
   oblique), each with an independent-oracle gate per §7.3's tautology findings.

### 8.3 Not worth bridging

The lens/GBD/caustics side of lumenairy (irrelevant to periodic unit cells) and EME
(niche unless waveguide work is planned). EMT screen and BOR are already bridged at
the right scope.

---

## 9. Remediation status (branch `fix/deep-audit-2026-07-05`, updated 2026-07-12)

Every fix below landed with a regression gate verified (or constructed) to discriminate
against the pre-fix code; affected pytest files and NGSolve/DEVSIM/bridge/statistical
validations pass locally after each commit.

**FIXED — P1 (all three):**
- C5-1 bridge graded-slab flip (5c5505c) — `reversed()` x3 + asymmetric-lossy
  bridge-vs-TMM gate (fails on pre-fix code by construction).
- C4-1 Norris-Landzberg inversion (951eb72) — ratio flipped; GATE B de-tautologized to a
  primitive-law Nf ratio + exact frequency-only direction leg; pytest direction pin.
- C5-2 FDTD seam graded/tensor drop (99c11d7) — graded fields sliced like the TMM peer;
  tensor/structured/broadband raise; DrudeOptical crash repro fixed; end-to-end
  modulation-sensitivity FDTD gate.

**FIXED — P2 physics formulas (Wave 2):**
- C7b-1 KaneOpticalMass halved alpha (007d442) — exact √(1+4αγ_F); legacy=True
  back-compat (α_legacy=2α); gate/test de-circularized to a numeric-dispersion reference.
- C2-2 Kane ⟨m*(Te)⟩ Sommerfeld K-factor (2c1e7db) — exact-FD shift gate (1.9% vs
  pre-fix 20%).
- C4-3 spont-spont 2x (7afc83c) — Olsson-identity gate.
- C4-4 Langevin exact-emit (c92053d) — telescoping-identity gate; validation GATE B
  now 1.9e-3.
- C3-7 absorption ÷Re(n_super) (19dad3f) — dense-superstrate lossy closure GATE C
  (ratio 0.999 vs pre-fix 1.5).
- C3-4 t0 de-embed n-aware (b4ad5a2) — hand-derived factor; t0-PHASE legs in all four
  nonvacuum gates (0.2-1.1° vs ~100° pre-fix).
- C3-2 Kerr standard chi3 (3152749) — factor 3 in all eight kernels incl. CUDA;
  ABSOLUTE SPM GATE F vs measured analytic pump (0.992 vs pre-fix 0.33).

**FIXED — P2 seams/guards (Wave 3):**
- C4-2 conical any-pol raise + C5-8 structured-period raise + C5-11 collapse
  unclaimed-key raise (f5cd4f0; Berreman validation conical leg converted to must-raise).
- C5-3 material-content cache keys + C5-6 solver-identity keys, _SCHEMA 4 (8abd725).
- C5-7 seam design.optical guards + oblique chi2/raman/gain raises (663b0eb).
- C5-12 full-band dispersion check + C6-3 Sweep duplicate-label raise (7817409).
- C5-9 signed Arrhenius Ea + C5-10 no silent callable substitution (3946db9).
- C2-1 body_net_doping_m3 phi_bi contract + C2-3 2D lateral-isolation raises +
  semi-semi carrier-blocking raise (69c5ccb; gated_dd_2d/bipolar_diode_2d/
  carriers_3d_bipolar all pass).
- C4-5 eh_split excitonic-accessor raises (77db547).
- C3-3 MO both-branch grid sizing + II-3D dimension guard + C6-4 LC rtol honored +
  C6-2 v_ac threaded through both AC sources (1a2e8fa; ac/mo/ii validations pass).
- C5-5 extras ride the lift — the silent x<->y transpose at shipped defaults (0c48abe;
  discrimination-proven gate).
- C6-1 lc_director_2d certifies on the geometric-tail error estimate, not update size
  (residual now reports the estimate; parity validation passes).
- C4-9 TDDB stress = sign-robust layer-mean |Ez| via new
  ElectrostaticResult.mean_absEz_per_layer (50f2ff0; split-gate probe: signed mean
  1.2e4 vs |Ez| mean 3.0e8 V/m; end-to-end skip-gated NGSolve test).

- C4-6 line_filter ES-band cancellation (f5bd0cc; new gain_per_m_slices_gs so the
  marcher subtracts exactly the band its poles re-add; ES-active ON==OFF carrier gate).
- C4-7 dualpol TM pairing (134c02f; shifted-TM path uses step_slices_wdm per-channel
  lineshapes; single-pol-at-nu_tm oracle gate + unshifted byte-compat pin).
- C4-8 calibration bandwidth honesty (material_fwhm_nm + net_3dB_bw_nm distinct keys,
  module-header caveat; dual-key gate).

**ALL 3 P1 AND ALL 23 CONFIRMED P2 FINDINGS ARE FIXED** (plus both §7.1 regrades and
the C6-class contract fixes), each with a discrimination-proven gate.

- C3-1 adaptive FEM probe grid (_probe_grid_sizes per medium; sub-wavelength cells
  byte-identical; alias-weight + sizing gates) and C3-6 material-memory run window
  (_ring_time_s at all five FDTD front-ends, warning when it dominates) — 9b59daa.
- C3-5 oblique band mask trusts sin_t < 0.95 with an exclusion warning (2D + 3D).
- C5-4 pipeline-seam TMM per_region_absorption re-keyed to design layer names
  (graded slabs sum into their layer; slab_<i> stays for direct LayeredStack use).

- Stale-docs batch (ec3afd7): C6-5 pair ACTUALLY fixed (the ledger had over-claimed it —
  sp_carrier nonparabolic capabilities, graphene sheet-BC shipped) + the §6.3 set
  (em/hci driver notes + Black/HCI attribution corrections, reliability/__init__,
  dynameta/__init__ module map incl. BOR + symmetry, interfaces default location,
  solve2d lateral shape doc) + the CPML matched scaling corrected to 1/n
  (probe-verified strictly beneficial; every non-vacuum gate tightened).
- Cache autosave batching (baf9739): autosave_every=K with atexit dirty-flush;
  default byte-compatible; measured 240x available for sweeps; honest cost model
  replaces the inverted "(cheap, crash-safe)" claim.
- C7b-2 reliability_tddb GATE B de-tautologized (hand-derived Arrhenius reference +
  module cross-pins).

**Hygiene waves (2026-07-12):**
- H1 gate de-tautologization (dd30a20 + follow-ups + H1c): scattering_link A-FIT
  re-fits the material's own eps; switching GATE B cross-pinned vs LCDynamics;
  qcse part-4 exponent independent; S-P Gauss-integral pin ADDED (rel 4.1e-3);
  sp nonparabolic alpha=0 byte-identity ADDED; fatigue GATE A stress reference
  replaced by a numeric isotropic-compliance Hooke solve from (E, nu) primitives
  (+ 2x-dT linearity and CTE-swap antisymmetry pins) and GATE C Weibull leg pinned
  by the distribution-free S(sigma0)=1/e anchor + the ln S ~ s^m functional
  equation. Every §7.3 tautology item now closed (GATE KANE was closed by C7b-1).
- H2 constants single-sourcing (74567f2): KB_EV_K (byte-identical literal,
  KB/Q_E quotient REJECTED — differs at 1.5e-11 rel vs the 1e-13 Arrhenius pins),
  H_PLANCK (exact SI h; replaced the 2*pi*HBAR twins + calibration literal),
  SOLVER_TIME_CONVENTION string, sbe m_e/c, inline Q_E literals, F12
  Aymerich-Humet coefficients hoisted to one tuple (DEVSIM expression verified
  byte-identical); 7 qd_soa validation twins moved to the same source.
  134 affected tests + 23 validations green.
- H4 test infrastructure (948da65): run_all rc==42 SKIP category (C6-6) +
  run-everything-except-SKIP discovery with a drift NOTE (the old regex silently
  never ran raise/assert-gated validations) + 35 qd_soa_* / optical_cache /
  sp_open_body curated into SMOKE; six capability-skips converted from exit-0 to
  SystemExit(42); CI smoke-tier leg wired; tests/test_devsim_smoke.py = first
  pytest presence for the DEVSIM solve path (equilibrium + unipolar DD).
- H3a private-API promotions (1a602d8): optics/rasterize.py owns the
  fdtd_seam rasterizer publics (RCWA bridge off the underscore names);
  thermal_fem build_thermal_forms/add_load_terms/mean_T_per_layer public;
  fem_mesh MESH_SCALE. Underscore aliases retained. (fdtd_nd export cleanup +
  FDTDLayer relocation + lumenairy_bridge/_common.py in flight — _common lands
  with the §8 B1 floor work.)
- H5a perf batch 1, results-identical (3da74bb): lc_director_2d vectorized
  red-black GS (x42 shipped scale, fields agree 1.9e-11 rad); LIDT absorbed(T)
  memo (26x fewer TMM solves, bit-identical); em Miner vectorized (17-45x,
  bit-identical); pipeline frees each CarrierField at its sole read;
  solve_fem diagnostics=False opt-out (~28%/solve, default unchanged;
  True-vs-False delta bounded by repeat-run noise). Korhonen banded Jacobian
  REFUTED as results-identical (LSODA step-sequence fork ~3e-10) — untouched,
  13-119x available if the identity bar is relaxed to solver tolerance.

**Bridge roadmap §8.2 (2026-07-12):**
- B1 / step 2 — floors + _common consolidation: new lumenairy_bridge/_common.py
  owns the SINGLE VERSION_FLOOR (5, 21, 0) + suffix-tolerant parse_version
  ('5.21.0rc1' no longer crashes the x3 copy-pasted gates) + the shared helper
  surface (pol_row/angles_rad/guards/p_basis_conversion, public names;
  rcwa_backend keeps _-aliases) + stack_layer_records, the ONE version-ceilinged
  reader of the private RCWAStack._layers (public attr preferred, warn beyond
  the tested 5.21.x line). bor_backend's copy-pasted gate (floor drifted to
  5.16.0) deleted; berreman's 5.14.4 tightening subsumed; pyproject floor
  5.14.2 -> >=5.21. 27 bridge tests green (2 new: parse/floor + _layers reader).
- B3 + B4a / step 3 quick wins (9d7931b): make_lumenairy_rcwa_solver(n_workers=,
  blas_per_worker=) threads solve_sweep (per-wavelength stack build + solve on a
  bounded pool, BLAS pinned thread-locally; byte-identical to serial, x6.1 on a
  12-wavelength structured sweep); make_lumenairy_pmm_solver(absorption=True)
  fills per_region_absorption/A_independent from PMMStack.layer_absorption --
  PMM at RCWA-bridge parity for the D2 absorption chain. New GATE E cross-engine
  oracle: PMM budget closure 1.4e-13, degree-stability 1.9e-4, and the RCWA
  per-layer split CONVERGES toward PMM's (0.078 -> 0.022 -> 0.010 over n_orders
  32/64/96) -- first run showed RCWA's metal-TM grating-layer absorption 2.7x
  off at n_orders=32 (Gibbs), i.e. PMM referees per-layer absorption, not just R.
- B2 / step 4 Berreman-conical leg (f0986df): conical s/p through the Berreman
  bridge is now EXACT via rotational covariance (layer tensors rotated Rz(-phi),
  solved in-plane -- planar tier only; RCWA/PMM lattices keep their guard). The
  GATE F end-to-end leg pins the bridge against lumenairy's NATIVE conical
  engine rotated to s/p on a LOW-SYMMETRY biaxial tensor -- the first tensor
  tried (optic axis along x) left co-pol r EVEN in phi and the gate BLIND to a
  rotation-sign bug (found by probing the deliberate bug); the Rz(25)Ry(35)
  frame fixed that, and the sign-flip probe now fails the gate. Isotropic
  azimuth-invariance at 1e-12 rides along.
- 8.1-6 closed: OOP-coupled tensors (tilted-LC director) at oblique AND conical
  solve first-class through the Berreman bridge (passivity verified);
  absorption=True on that regime degrades gracefully (upstream raises for
  OOP-oblique internal-field reconstruction -- documented limitation, pinned by
  a no-crash + warn pytest).
- 8.1-1 disposition: conical PMM stays refused at the bridge -- lumenairy's
  native conical PMM (5.20+, pure-nodal 5.21.3) returns lab-basis s/p-mixture
  rows and no per-order Jones, so rotated-s/p totals for a patterned cell are
  not synthesizable (and the covariance shortcut is invalid for lattices). The
  stale "not supported by PMMStack" message now states the real blocker and
  routes planar conical to Berreman, patterned conical to the FEM.
- UPSTREAM FINDING while re-running the bridge validations:
  lumenairy_bor_bridge GATE C (lossless ring-grating energy closure) FAILS at
  |R+T-1| = 2.3e-2. Bisected to lumenairy commit fca4665 ("unit-invariant flux
  normalization + propagating-mode classifiers", its own audit P1-01/P2-06):
  fca4665^ closes at 1.2e-11, fca4665 leaks 2.3e-2 and drops one incident mode
  (319 -> 318). The DynaMeta bridge only relays lumenairy's res["energy"], so
  the defect is upstream; the gate had silently rotted because the smoke tier
  never ran it (fixed by H4's run-everything discovery — this failure is that
  fix WORKING). Left honestly red; handed off as a lumenairy-side task with the
  full repro + bisect.

**PENDING (residual hygiene — optional):**
- §6.2 perf still open: topology_opt jit-beta + FDTD vacuum-ref caching +
  kernels3d prealloc, Berreman absorption double-solve consolidation.
- Refinement follow-ons noted in-code: sampled per-layer peak |Ez| for TDDB; per-order
  Jones synthesis for bridge conical s/p; net-bandwidth co-fit for the Innolume
  calibration; 2D lateral-interface wiring; 3D II element reconstruction.
