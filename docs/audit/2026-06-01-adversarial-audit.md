# DynaMeta -- Independent Adversarial Audit

**Date:** 2026-06-01
**Scope:** the full `dynameta` library (40 modules, ~4700 LOC), its 28 validation scripts, 4 pytest files, and docs.
**Method:** 21 LLM agents under a deterministic workflow -- 8 adversarial *finders* (one per subsystem), each feeding an independent *skeptic* that tried to **refute** every finding by reading the cited code and replicating the math, plus 5 grounded *strategic assessors*. Findings below carry the **post-verification** severity (the skeptic's `corrected_severity`); several finder claims were downgraded or refuted on verification, and the headline conical-incidence concern was **directly refuted** (see below).

This report is a snapshot for planning. Nothing here was changed in the code; it is an assessment only.

> **Resolution (2026-06-01).** The Part-I functional findings were subsequently fixed
> (commits `38a5826`..`1b25aec`): all HIGH items (OPT-1, SP-1, SP-2, from_design role +
> gate-name, the test/CI gaps) and the substantive MEDIUM/LOW items. Highlights: the FD
> g-factor was replaced with an accurate <1% rational fit; the SP self-consistent solve now
> reports convergence + warns; oblique raises on a non-vacuum superstrate and warns on the
> non-angle-aware PML; an independent volumetric absorption `A_independent` was added; the
> solver-free bridge spine is now pytest-covered in CI; and all 27 PASS/FAIL validations exit
> non-zero on failure (`validation/run_all.py` gates the set). The conical ky-Bloch phase was
> adversarially CONFIRMED correct (not a bug) and now has observable structured-cell coverage.
> The Part-II strategic items (RCWA oracle, AC/transient, optimization, field-aware seam)
> remain as the forward roadmap.

---

## Severity legend

| Sev | Meaning |
|---|---|
| **HIGH** | A correctness bug or silent-failure reachable through the public/general API; can return confidently-wrong numbers. |
| **MEDIUM** | A real defect or overclaim with a bounded blast radius (specific mode, BYO path, or quantitative bias). |
| **LOW** | Latent/robustness hazard, loose validation, or a generality gap with a clear workaround. |
| **INFO** | Doc/cosmetic. |

After verification: **~8 HIGH, ~16 MEDIUM, ~18 LOW, ~5 INFO** across 8 subsystems. Seven of eight subsystems are **mostly-solid**; the Schrodinger-Poisson subsystem is **concerning**.

---

## Executive summary

**The static charge -> optics core is genuinely sound and unusually honest.** Verified-correct by code reading + closed-form replication: the Drude sign/loss convention (`exp(-iwt)`, Im(eps)>0), the equilibrium Poisson with the Aymerich-Humet F(1/2) (which correctly routes around DEVSIM's broken Fermi integral, accurate to ~0.1%), the scattered-field complex-HCurl weak form, the layered Fresnel background (reduces exactly to a plane wave at n=1, p-pol BC matrix matches tmm), the Bloch-phase anti-silent assertion, the boundary-inclusion geometry (volume conserved to 1.00000), and the 1D Schrodinger eigensolver (analytic wells to 1e-6). The conical-incidence ky-Bloch phase -- the prime suspect going in -- was **proven correct**: monkeypatching a no-op ky-phase produced a catastrophic energy violation (R+T=0.29), so the feature genuinely enforces the transverse phase; the only gap is test coverage, not correctness.

**But the audit confirmed real bugs the test suite cannot see.** The eight HIGH items cluster into four themes:

1. **Oblique optics is silently wrong for a non-vacuum incidence medium** (OPT-1) -- and `pipeline.py` auto-derives and feeds that medium with no guard.
2. **The Schrodinger-Poisson self-consistent solve does not converge in its documented default mode and returns the non-converged result silently** (SP-1), with a ~1300x swing in the headline accumulation on a cosmetic loop-count parameter; the oxide-cap calibration is biased by a hard-wall artifact (SP-2).
3. **`from_design` mis-detects the gate** -- it ignores `electrode.role` (a ground pad can be taken as the gate) and does not carry the gate's name through, so `run_pipeline` silently solves the whole sweep at Vg=0 for any gate not literally named `"gate"`.
4. **The physics is structurally unguarded:** 0 of 28 validations run in CI, 27 of 28 print PASS/FAIL but `exit 0` on failure, and the solver-free bridge spine has zero pytest coverage. A refactor can break the physics while CI stays green.

**Strategically, the library is cogent-but-narrow.** The alignment keystone is a real decoupling, but the field -> eps seam cannot see the applied E-field (no Pockels/Kerr/electro-absorption), eps is scalar-isotropic only (no birefringent EO crystals), there is exactly one optical solver, and -- despite the name -- the carrier solve is **DC-only**: no AC, transient, bandwidth, RC, or C-V, which is half a modulator's spec. The single highest-leverage addition (and the one missing oracle) is an **independent RCWA solver validated against the FEM on the real lossy, laterally-structured reference cell**.

---

## Part I -- Functional audit

### 1. Optics core (`optics/solver.py`, `eps_assembler.py`) -- mostly-solid

| ID | Sev | Issue | Recommended fix |
|---|---|---|---|
| **OPT-1** | **HIGH** | Oblique incidence is silently wrong for any non-vacuum superstrate. `solver.py:148-150` fixes `kx=k0 sin(theta)` from the **vacuum** dispersion (no `n_super`), while the T-normalization at `:290-291` uses `n_super*k0*cos(theta)` -- internally inconsistent. Replicated: `n_super=1.5, theta=60deg` gives a T-factor 63% off. Reachable in production -- `pipeline.py:84` derives `n_super=sqrt(eps(superstrate))` and passes it (`:91`) with **no guard**; every oblique test uses `n_super=1`. | Assert `|n_super-1|<tol` when `theta>0` and raise `NotImplementedError`, **or** implement the dense-incidence dispersion (`kx=Re(n_super) k0 sin(theta_in_medium)`, `kz_sup=sqrt((n_super k0)^2-kx^2-ky^2)`, normalize T by `Re(kz_sub)/Re(kz_sup_true)`). Add an `n_super=1.5 @ 30deg` vs tmm case. |
| OPT-2 | MED | Absorption `A` is always `1-R-T` (`:277`, `:292`); there is **no** volumetric `Im(eps)|E|^2` integral, so "R+T+A~1 energy conservation" is a tautology and the lossy test's A-check is implied by its R/T checks. | Compute `A` independently (normalized volumetric loss integral or Poynting-flux defect) and report **both** `A` and `1-R-T`; flag disagreement as the real diagnostic. |
| OPT-3 | MED | The HalfSpace z-PML is **not** angle-aware (`pml_alpha=1j` constant, fixed normal, `:153-163`); energy non-conservation reaches ~1.2% at 30deg, masked by `TOL=0.03`. README claims a runtime "warns" that **does not exist**. | Drop the "angle-aware" claim; document a supported angular range; either add the promised warning or correct the README. |
| OPT-4 | MED | `OpticalSpec.incidence_side` and `OpticalSpec.outputs` are validated but **never read** -- top illumination and R/T/A are hardcoded. `incidence_side='bottom'` is silently ignored. | Implement, or raise `NotImplementedError`, or validate-and-reject. |
| OPT-5/6/7 | LOW | README stale (says p-pol/conical oblique unimplemented though both ship + validate); docstring overclaims "<0.3% through 30deg" (actual ~0.9% R, ~1.2% energy at 30deg); `_cell_average` returns `0+0j` if all probes on a z-plane fail (silent zero into the lstsq R/T fit). | Refresh README; quote honest per-angle accuracy; raise/NaN on an all-empty probe plane and offset probes off the x=0/y=0 lines. |

### 2. Oblique + conical Bloch (`solver.py`, `specs.py`) -- mostly-solid; **headline concern refuted**

The central worry -- that a no-op ky-phase would still pass a phi-invariance test on a laterally-uniform slab (the bug class that hid the *original* oblique failure) -- was **directly refuted**: the validated `phase_in_space` route solves the physical field, which genuinely propagates as `exp(i ky y)`, so forcing a no-op ky-phase at phi=90 gave R+T=0.29 (catastrophic), and a sign-flip was also caught. Per-idnr detection is deterministic with a ~1e14 discrimination margin, and the count-mismatch assert is a real backstop. **The conical feature is correct.** The findings are coverage/limits:

| ID | Sev | Issue | Fix |
|---|---|---|---|
| F1 | MED | The conical ky-phase is never validated on a **laterally-structured** (observable) geometry by any committed test -- the one conical test is a featureless slab. | Add a conical (phi=45/90) check on a patch/grating: energy conservation + phi=0 vs phi=90 R/T equality for a square cell. |
| F2 | MED | Conical/oblique has **zero pytest coverage**; only manual `validation/` scripts guard the Bloch path. | Add a fast unit test of `_detect_bloch_dirs` (interleaved x/y split) and `_bloch_phase_list` (non-unit phase on y-idnrs for ky!=0) -- no FEM solve needed. |
| F3/F4 | LOW | Conical `TOL=0.03` is ~15x looser than the achieved error; conical is s-pol + air-substrate only in practice (p-pol conical rejected; dense-substrate conical untested). | Tighten the phi-spread gate to ~1e-3; add a conical dense-substrate point or document the restriction. |
| *(verifier-added)* | MED | **No incidence-angle upper bound** anywhere (`specs.py` guards polarization/azimuth but not theta); the fixed-alpha PML degrades at grazing, so theta=60-80deg returns a silent, likely-wrong answer. | Add a theta bound (or warning) consistent with the validated 0-30deg range. |

### 3. Boundary-spanning inclusions (`ngsolve_layered.py`) -- mostly-solid; geometry verified

The geometry machinery is **solid** (volume conservation 1.00000 across interior/edge/corner/overlap cases, picometre-exact face signatures, corner disk matches centered disk). Two real issues:

| ID | Sev | Issue | Fix |
|---|---|---|---|
| **BI-1** | MED *(from HIGH)* | A **semiconductor in an inclusion layer is silently frozen at nominal eps** -- the `if L.inclusions:` build branch (`:157-179`) never registers `region_align`/`source_by_region`, so the charge->optics bridge is bypassed for that region (it lands in `fixed_eps_regions`). The general case (a patterned active region) silently loses its tuning. Verifier added: a DEVSIM<->optical **region-name asymmetry** makes the single-inclusion case un-patchable by hand. | Register alignment for any semiconductor sub-solid in the inclusion branch (mirror the cavity branch), or raise `NotImplementedError` so it fails loudly. Reconcile the inclusion region naming between the two builders. |
| BI-2 | MED | The validation's stated rigor is an **overclaim**: monkeypatching `_identify_periodic` to identify nothing still PASSES the translation-invariance and energy arms (both degenerate); only the thin `0.02<R<0.98` guard catches it. The Identify *is* load-bearing, but the validation's stated reason it would catch a break is false. | Assert the expected paired-face count directly; tighten energy tolerance; correct the docstring. |
| BI-3 | MED | Boundary inclusions are never tested at **oblique** incidence, exactly where they multiply the interleaved Bloch identifications the solver must resolve (the documented nasty failure mode). | Add an oblique (15-30deg) translation-invariance check on the grating. |
| BI-4/BI-5 | LOW | OCC builder supports only Rectangle/Circle (Ellipse/Polygon/RegularPolygon raise); `_identify_periodic` keys faces by centroid only (latent collision/mis-bucketing hazard, currently unreachable). | Extend shapes or document the limit; make face signatures extent/normal-aware. |

### 4. Bridge / alignment / lift / Drude (`core/*`, `materials/optical_model.py`) -- mostly-solid

The spine is well-built: `validate_coverage` genuinely raises on missing/extra/double-mapped regions, background eps flows from `n_bg` through the same Drude formula (no grid-corner proxy), `resample` is ndim-general, and there is no silent air/default fallback in the mapped path.

| ID | Sev | Issue | Fix |
|---|---|---|---|
| F1 | MED *(from HIGH)* | "SeparableXYLift asserts c4v symmetry" is an **overclaim** -- `apply()` (`lift.py:71-88`) asserts nothing about the field; only `choose_lift` checks a `device_symmetry` **string**. A monotonic-ramp (non-c4v) field lifts with no raise. (Square-cell *is* enforced transitively on the design path; only direct construction bypasses it -- hence the downgrade.) Verifier added a **mixed-sign field bug**: the xy outer product makes the (neg)x(neg) quadrant spuriously positive for a sign-changing lateral profile. | Either downgrade the docstrings (preconditions are the caller's responsibility) or add real even-symmetry / square-cell asserts in `apply()`. |
| F2 | MED | `time_convention` is **write-only** metadata -- never read by the solver/assembler; Drude Im(eps) sign is convention-independent. An `exp(+iwt)` field is labeled `+iwt` but carries `-iwt` values. | Assert `field.time_convention == solver convention` at the bridge/solver boundary (or conjugate Im(eps) on mismatch). |
| F4 | MED | The **3D bridge branch ignores `RegionAlignment.stack_axis`** (hardcodes `z`); a 3D field with the default `stack_axis='y'` is silently mis-axised (gradient placed on the wrong axis, no error). | Honor `stack_axis` in the 3D branch, or assert `=='z'` and raise otherwise. |
| F3/F5/F6 | LOW | `fixed_eps_regions` hardcode `exp(-iwt)` (label inconsistency within one dict); 3D validations use `gate_patch_frac=1.0` (laterally uniform -> can't prove transverse handling, won't catch F4); coverage check is skipped when `mesh_regions is None` (BYO bypass). | Use `field.time_convention` for fixed regions; add a `frac<1` transverse 3D assertion; make coverage validation unconditional or document the BYO requirement. |

### 5. Carriers: equilibrium / DD / contacts (`physics_*`, `dc_solve.py`, `devsim_layered.py`) -- mostly-solid

Equilibrium Poisson, accumulation sign, ohmic dual-contact-model, and the bipolar diode (rectifies 1.8e10, ideality 1.20) are verified correct.

| ID | Sev | Issue | Fix |
|---|---|---|---|
| F1 | MED *(from HIGH)* | The Fermi-Dirac diffusion-enhancement `g`-factor uses the **degenerate asymptote** `g=1+(2/3)(c1 n/Nc)^(2/3)`; vs the true generalized-Einstein ratio `F(1/2)/F(-1/2)` it is **6.6% high at ITO's operating point and 25-35% high in the moderate-degeneracy transition**. (Cancels in pure drift, so it is not a Boltzmann no-op -- hence MED not HIGH.) | Replace with a Pade/rational fit of `F(1/2)/F(-1/2)` (the AH forms are already present), keeping it a simple `pow` so DEVSIM still differentiates it; add a true-FD transition validation. |
| F2/F3 | MED | The 3D MOS-cap "DD reduces to equilibrium" is **mobility-independent** (zero DC current), and its 25% tolerance is explicitly sized to absorb the F1 g-error. The only genuine transport test (3D resistor) passes at a **loose 14% over-prediction** asserted as "mesh-limited" but never tested under refinement in code. | Re-label the MOS-cap test as an equilibrium-limit/sign check; add an in-code Richardson/mesh-refinement check to the resistor and tighten its gate. |
| F4 | MED | The 2D-layered drift-diffusion path is advertised (`_reference_device.py --drift-diffusion`) but **unvalidated**, and the code's own docstring says the gated case **does not converge**; the solve loop has no try/except. | Guard/refuse gated DD with edge-only grounds (point to equilibrium), or make the example converge and validate it. |
| F5/F6/F7 | LOW/INFO | `gummel` solver has zero coverage and admits it does not solve its target case; the bipolar diode prints but never asserts ideality; stale `ElectronQFL` references (no QFL formulation exists). | Validate or mark `gummel` experimental; assert `1<=n<=2`; drop/fix the `ElectronQFL` references. |
| *(verifier-added)* | MED | `_dielectric_eps_static` **falls back to the optical eps** (e.g. HfO2 ~4 vs DC ~18) with only a printed warning for any gate dielectric missing `eps_static_dc` -- a loud-but-continue wrong-physics path on the general API (reference sets it, so not triggered there). | Raise (match the 3D path) or set a hard quality flag on the CarrierField. |

### 6. Schrodinger-Poisson quantum (`schrodinger_poisson.py`, `sp_carrier.py`) -- **concerning**

The 1D eigensolver and the degenerate slab-mode bulk recovery are robust. But the **self-consistent solve and its gate calibration have HIGH-severity defects**:

- **SP-1 (HIGH):** `solve_self_consistent` **does not converge in the documented default isolated-well mode** -- it limit-cycles (max|dphi|=0.76 V for all 60 iterations; Poisson residual ~24 q Nd) because the kept-sub-band set churns across the edge-amplitude threshold, and it **returns the non-converged phi silently** (no flag/warning). The validation's pass test checks an accumulation ratio, not `dV<tol`, and the roadmap falsely claims this case "converges." Verifier added: the returned density depends on the **parity of `max_outer`** -- `60` gives n(gate)=8.96e23, `61` gives 6.79e20, a **~1300x swing** on a cosmetic parameter, and the script prints `converged=True`. *Fix:* return/raise a convergence flag; freeze the kept-state set or under-relax on bound-count change; assert `dV<tol` in the validation.
- **SP-2 (HIGH):** the oxide series-cap calibration is biased by a **hard-wall depletion artifact** -- Dirichlet phi=0 at *both* ends gives a spurious -3.04e17 m^-2 flat-band "excess" (~0.38nm dead layer per wall), so the resolved surface potential is **~444 mV high**, the headline "accumulation 6x smaller" is really ~1.6x, and the self-consistency check is circular. *Fix:* define `N_excess` relative to the self-consistent flat-band baseline (not `2 n_bg`), or use a charge-neutral body boundary; replace the circular check with a Gauss-law check.
- **SP-3/SP-4 (MED):** Kane nonparabolicity is reachable **only** through `density()`, never through `SchrodingerPoissonCarrier` (the carrier that feeds optics is parabolic; ~23% bulk mismatch if mixed) -- advertised "DONE" but unreachable for a device run. The body-side z=0 Dirichlet hard wall injects an unphysical ~0.4nm depletion into the optics-facing density at every bias.
- **SP-5/SP-6 (LOW):** oxide-cap division is silently bypassed when `surface_potential_xy` is supplied; the "degenerate filling consistency" test is a normalization tautology (random unphysical sub-band populations still pass), and the BenDaniel-Duke mass-discontinuity term has no analytic-validation coverage.

### 7. 3D DEVSIM / pipeline / from_design (`devsim_3d.py`, `pipeline.py`, `analysis.py`) -- mostly-solid

The gmsh nm-build, gate-patch OCC imprint (genuine lateral contrast), and the no-hand-alignment region-name matching are verified correct. The generality has silent failure modes:

- **F1 (HIGH):** `from_design` selects the gate as "the first electrode with a CrossSection footprint" **ignoring `electrode.role`** (`:97`) -- a `role='ground'` pad is taken as the gate, silently choosing the wrong dielectric/eps_ox/thickness/patch-frac. *Fix:* filter to `role=='biased'`; raise if ambiguous.
- **Verifier-added (HIGH):** `from_design` does **not carry the gate-electrode name** through; `Devsim3DEquilibrium.solve` hardcodes `bias.voltages.get("gate", 0.0)` while `BiasPoint` is keyed by electrode name, so a gate named anything but `"gate"` makes `run_pipeline` **silently solve the entire sweep at Vg=0**. The from_design docstring explicitly claims "-> run_pipeline, no hand alignment." *Fix:* thread the gate name into `Stacked3DSpec` and use it for the contact + bias lookup.
- **F2/F3 (MED):** with >1 semiconductor layer, `from_design` silently picks the lowest-index one (wrong semiconductor + wrong oxide); a **non-square cell** mis-places the carrier accumulation laterally (carrier eps spans only `[0,min(px,py)]`, the rest is edge-clamped by the VoxelCoefficient). *Fix:* raise on multi-semiconductor / non-square (or remap lateral axes and build a rectangular carrier box).
- **F4/F5/F6 (LOW/INFO):** rectangular/circular gate collapsed to a centered square via `max()`; roadmap stale (says multi-dielectric from_design unsupported though it works for reference ordering); `analysis.resonance_dip` claims "unequally-spaced-safe" but is off by a full grid step on non-uniform grids.

### 8. Cross-cutting rigor (whole library + tests + docs) -- mostly-solid core, fragile guarantees

ASCII-only `print()` output is clean (0 non-ASCII in any print across 40 modules), SI units are canonical, and the bridge fails loudly in the right places (no bare `except:`, no air-default eps). But the **rigor guarantees are thinner than advertised**:

- **F1 (HIGH):** **0 of 28 validations are CI-gated** -- `.github/workflows/ci.yml` installs only numpy/scipy/pytest/tmm and runs the 16 data-model/SP unit tests; all 23 solver-backed validations live outside CI. "No unvalidated physics is shipped" rests entirely on manual runs.
- **F2 (HIGH):** the reusable, **solver-free bridge spine** (`assemble_eps`, all lifts, `validate_coverage`, `MaterialEpsMap`, `EpsField`, `analysis`) has **zero pytest coverage** though it would run in <1s in CI.
- **F3 (HIGH):** **27 of 28 validation scripts** print PASS/FAIL but **never exit non-zero** on failure (only `bipolar_diode.py` is machine-checkable) -- even the manual workflow is not machine-verifiable.
- **F4/F5 (MED):** the 2D gate-dielectric path warns-and-continues with the wrong (optical) eps while the 3D path raises (and a bare `Stacked3DSpec` silently defaults `eps_oxide=18`) -- three inconsistent behaviors for one condition; the README claims an oblique-energy "warns" that does not exist.
- **F6/F7/F8 (LOW/INFO):** `_cell_average` silent `0+0j`; `time_convention` decorative; one non-ASCII character in a `solver.py:218` comment.

---

## Part II -- Strategic assessment

### Is the library cogent and general enough? -- **Cogent-but-narrow, trending leaky**

The alignment keystone (string-key + bbox, no shared object graph; lift the carrier *density* so background eps re-derives through the same formula) is genuinely good engineering. It leaks the moment you leave the reference ITO archetype:

| Gap | Pri | Why it matters |
|---|---|---|
| **eps map cannot see the applied E-field** (`assemble_eps` discards `potential_V`; `eps(lambda, n)` has no field arg) | **CRITICAL** | Structurally excludes **all field-effect EO** -- Pockels (LiNbO3/BTO), Kerr, Franz-Keldysh, QCSE -- arguably the dominant low-loss phase-modulation class. The tool cannot express any of them. |
| **Scalar-isotropic eps only** (no tensor anywhere: EpsField -> VoxelCoefficient -> weak form) | HIGH | Every EO crystal is birefringent at zero bias; graphene is anisotropic. Hits this wall before any dynamics. |
| **One `OpticalSolver`, one device archetype** (all 4 CarrierSolvers model gated free-carrier accumulation; density is the contract field) | MED | Generality is asserted through single impls; a pure-electrostatic (Pockels) or thermo-optic device has no natural home (the seam requires a density). |
| Inclusion geometry limited to rectangle+circle | MED | Huygens/plasmonic metasurfaces need crosses, split-rings, bars, freeform. |

**Recommendation:** widen the seam *now*, before adding physics -- change `NToEpsMap` from `eps(material, n, lambda)` to `eps(material, fields: dict, lambda)` and have the bridge pass the full local grid fields (it already has `potential_V` on a known grid, so it can compute E = -grad(phi)). `DrudeOptical` keeps reading only `electron_density_m3` for exact back-compat. This single change unblocks EO and thermo-optic without touching the solver.

### Which optical solvers to add (RCWA / TMM / FDTD / HYPRE)?

| Solver | Pri | Fits existing Protocol? | Verdict |
|---|---|---|---|
| **RCWA / Fourier-modal** | **CRITICAL** | No -- needs a new `LayeredStackSolver` seam (it wants layered eps(z) slabs + a z-slicer for the graded carrier eps, not per-mesh-region voxels) | The natural fast workhorse for periodic layered ENZ/ITO modulators; you already have `grcwa`/`inkstone` + a native RCWA module from Lumenairy. |
| **TMM fast-path** | HIGH | No (same layered seam) | Exact for unstructured stacks; doubles as a per-bias oracle. Small effort. |
| `incidence_side='bottom'` | MED | Yes | Validated config that the solver never reads (OPT-4); cheap correctness. |
| **FDTD** | LOW | No | Broadband/transient/nonlinear, but RCWA dominates the narrowband CW ENZ regime. Only if you pursue pulsed/nonlinear. |
| **HYPRE / AMG** | LOW | Yes | **Do not.** A known dead-end at ENZ (indefinite near-zero-eps systems; an AMS port already failed). Keep umfpack/BDDC; get scale from RCWA. |

**Recommendation:** build a `LayeredStackSolver` Protocol, prove it with TMM, then add RCWA gated against the FEM on a patterned patch. Offload heavy sweeps to RCWA/TMM; keep the FEM for confirmation.

### Major electrical-side gaps -- the "dyna" is missing

Confirmed by exhaustive grep: **every** carrier solve is `type="dc"`; both DD modules deliberately drop the time term. There is no AC, no transient.

| Gap | Pri | Why it matters |
|---|---|---|
| **AC small-signal / transient carrier solve** -> modulation bandwidth, RC, gate C-V, switching energy | **CRITICAL** | These *are* the modulator figures of merit a "dynameta" tool is expected to deliver. A green run gives static contrast and says nothing about whether the device meets its speed target. |
| **DC gate C-V curve** *(quick win)* | HIGH | Reachable with **zero new solver** -- integrate Q(Vg) over the semiconductor from the per-bias CarrierFields a voltage sweep already produces. Unblocks the bandwidth story. |
| Field-/density-dependent mobility (and making mu spatially resolved) | HIGH | mu sets sheet/access resistance -> the R in RC; in degenerate ITO it is strongly n-dependent (today a single scalar at n_bg). |
| Interface traps + gate tunneling leakage | MED | Shift/stretch C-V, pin charge, set static power and hold voltage. |
| Bandgap narrowing / many-body / quantum capacitance in degenerate ITO | MED | Shift the absolute ENZ depth and co-determine C-V (band_gap_eV/chi_eV are currently inert). |
| Optoelectronic feedback (absorption -> generation -> eps); electro-thermal | LOW | Coupling is strictly one-way; T is hard-coded 300 K. Real but second-order for ITO ENZ at modest intensity. |

**Recommendation:** ship the DC C-V curve next (zero new solver, highest dynamic-adjacent value), then scope the AC/transient solve.

### Optimization / inverse design

There is **no** optimizer today (sweeps are forward-only; the only `scipy.optimize` use is the offline Drude fit). For modulator design the real task is inverse (maximize tuning range / contrast / bandwidth subject to loss + fabrication limits).

- **Add a thin gradient-free layer as two new Protocols over the existing pipeline** (don't touch solver math): an `Objective` (`rows -> scalar`, with 2-3 modulator FOMs in a new `objectives.py`), an `Optimizer` (CMA-ES default), and a `DesignVars <-> Design` encode/decode seam.
- **It is computationally gated on a fast forward solver:** CMA-ES x minutes/FEM-solve x remesh is intractable; geometry optimization needs RCWA. On the FEM, only low-dim bias/Drude tuning at a fixed mesh is realistic.
- Adjoint/gradient is **not** a shortcut as built -- R/T come from Python point-sampling + lstsq outside NGSolve's graph, and the PML is rebuilt per solve. It would need R/T re-expressed as in-graph functionals + a design-constant PML.

### Devil's advocate -- top risks for trusting this on a real design

1. **(CRITICAL) No independent full-wave/RCWA cross-check on a laterally-structured, lossy cell** -- the actual modulator regime. All quantitative tmm checks are featureless slabs; structured tests use only energy/translation-invariance, which a phase/diffraction-partition error passes. The solver is unverified by any independent tool exactly where it must be trusted.
2. **(CRITICAL) No dynamic response** -- zero AC/transient/RC/bandwidth/switching/energy. Half the modulator spec is structurally absent.
3. **(HIGH) The FEM linear solve has no convergence assertion** -- GMRes proceeds past `maxsteps` with whatever vector it holds, and AMS is disabled in the ENZ regime; a non-converged ENZ solve returns a plausible-but-wrong R/T silently.
4. **(HIGH) The physics chain is regression-unguarded** (Part I.8: 0/28 CI-gated, 27/28 exit 0 on failure, spine untested).
5. **(HIGH) SeparableXYLift is an unbounded separable approximation** for the canonical c4v square patch (the reference geometry), never bounded against a true 3D carrier solve.
6. **(MED) Degenerate-ITO approximations bias the ENZ depth** (no BGN, parabolic self-consistent SP, classical-vs-quantum centroid ~1nm, linear resampling smears the ~1nm peak).

**Single change that most increases trust:** add a second `OpticalSolver` backed by an independent RCWA tool (grcwa/inkstone) and validate it against the NGSolve solver on the **actual reference lossy-Au-patch + ENZ-ITO unit cell** -- full R/T/A spectrum *and* the bias-induced resonance shift -- ideally anchored to the reference modulator measured value. This closes the missing oracle, provides the fast solver, and unlocks optimization.

---

## Part III -- Prioritized recommendations

**P0 -- correctness + trust (do first; small-to-medium effort):**
- Fix OPT-1 (guard or implement non-vacuum incidence).
- Fix SP-1 (convergence flag + stop the limit cycle) and SP-2 (baseline-correct oxide-cap); the parity-dependent ~1300x swing is the most alarming single defect.
- Fix `from_design` gate detection (use `role`) and carry the gate **name** through to the bias lookup.
- Harden tests/CI: make all validations `exit(1)` on failure; add a solver-free pytest suite for the bridge spine; add a fast CI smoke. (Cheap; removes the structural fragility.)
- Add a FEM convergence assertion (raise/warn when GMRes hits `maxsteps`).

**P1 -- the linchpin + the missing half-spec (medium-to-large):**
- RCWA as a second `OpticalSolver` behind a `LayeredStackSolver` seam, validated vs FEM on the real reference lossy cell (closes the oracle + fast solver).
- DC gate C-V (zero new solver), then scope the AC/transient carrier solve.

**P2 -- generality (medium-to-large):**
- Field-aware `NToEpsMap` seam (unblocks EO/thermo-optic); register the inclusion-layer semiconductor in the bridge (BI-1).
- Gradient-free `Objective`/`Optimizer` Protocols over the pipeline (gated on RCWA).
- Tensor/anisotropic eps; honor `stack_axis` in the 3D bridge; raise on multi-semiconductor / non-square `from_design`.

**Documentation hygiene:** correct the README/roadmap overclaims surfaced throughout (PML "warns", A as an independent check, p-pol/conical status, multi-dielectric from_design status, "no unvalidated physics is shipped").

---

## Appendix -- methodology

- **Orchestration:** a single background workflow ran 8 finder agents (one per subsystem) -> 8 independent verifier agents (one per subsystem, tasked to *refute*) -> 5 strategic assessors, concurrently. 21 agents, ~2.29M agent tokens, 843 tool calls, ~25 min wall-clock.
- **Evidence standard:** every finding cites `file:line`; verifiers reproduced claims by reading the cited code and replicating math/closed forms (e.g. the OPT-1 kx/kz normalization and the F1 generalized-Einstein ratio were replicated numerically; SP-1/SP-2 were reproduced by re-running the solver). No source files were modified.
- **Adversarial integrity:** verification downgraded BI-1, bridge-F1, and carriers-F1 from HIGH to MEDIUM, and **refuted** the central conical-incidence hypothesis -- evidence the panel was not rubber-stamping.
- **Reproduce:** the validation scripts run as `python -u -m validation.<name>` (minutes each, require the `[solvers]` extra); `python -m pytest tests/ -q` is the ~2s CI gate.
