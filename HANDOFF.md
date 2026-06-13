# Project Handoff: Metasurface Modulator Multi-Physics Simulation

**Prepared:** 2026-05-28
**Purpose:** Comprehensive handoff to a new agent picking up this work.
**Owner:** Andrew Traverso, Neurophos / Metacept
**Working directory:** `D:\Metacept\Neurophos\Python_Test_Scripts\`

---

## 1. PROJECT GOALS

Multi-physics simulation of a **tunable amplitude / phase metasurface
modulator** based on reference et al. 2021 (gap-plasmon nanopatch
antenna over an ITO-loaded MIM cavity).

The simulation predicts:
- DC carrier-density redistribution in the ITO under bias
- Resulting wavelength-dependent permittivity (Drude / ENZ)
- Optical reflection amplitude `|r|^2(λ)` and phase `arg(r)(λ)` at
  normal incidence for each bias point

End goal: feed predictions into a **pixelated SLM design** where each
pixel comprises 10×10 unit cells driven by an ITO ↔ mirror bias,
with the option to flip polarity. The simulation must therefore
support arbitrary device geometries, materials, and electrode layouts
declaratively.

---

## 2. PHYSICAL DEVICE

The unit cell, bottom-to-top:

```
   |-- Au patch antenna (50 nm, square 175 nm x 175 nm)  ← V_top electrode
   |-- Cr adhesion (5 nm, only under patch)
   +-- upper Al2O3 (1 nm)        ← upper dielectric cap
   +-- upper HfO2 (7 nm)         ← upper dielectric cap (high-k)
   +-- ITO (5 nm, active layer, n_bg ≈ 4 × 10^20 cm^-3)
   +-- lower HfO2 (7 nm)         ← lower dielectric cap
   +-- lower Al2O3 (1 nm)        ← lower dielectric cap
   +-- Al-Nd mirror (70 nm, full lateral extent)  ← V_bot electrode
   +-- Si substrate / Si substrate-air interface
```

- Unit-cell period: **P = 370 nm** (square; 2D periodic Bloch lattice)
- Patch side: **L = 175 nm** (square)
- ITO grounded at peripheral pads ~mm away in the real device (the
  exact lateral BC matters — see Section 6)

### Physics

- **Gap-plasmon resonance**: the patch antenna couples to a cavity mode
  whose field is concentrated in the dielectric-cap + ITO sandwich
  directly under the patch. Resonance wavelength ≈ 2 · L · n_eff ≈
  1300 nm for L = 175 nm and the cavity stack above.
- **ITO acts as a tunable ENZ material**: at λ_ENZ, Re(ε_ITO) crosses
  zero. the reference modulator's design point places λ_ENZ near the patch-antenna
  resonance, so that under bias the ITO carrier density modulates,
  shifting λ_ENZ, shifting the effective mode index, and thus
  shifting the resonance dip in `|r|^2(λ)`.
- **Bias scheme (the reference modulator)**: V_top on patch, V_bot grounded (mirror),
  ITO grounded at far peripheral pads. Under +bias: accumulation at
  the TOP of ITO (closest to patch through upper cap); under -bias:
  depletion there.
- **Bias scheme (user's pixel design)**: V applied between ITO and
  mirror, with polarity flip ability. Patch may be floating (driven
  by capacitive coupling) or grounded with the ITO.

---

## 3. PIPELINE ARCHITECTURE

Three stages exchanging Zarr files:

```
Stage 1: DC carriers (DEVSIM)
   ├── 2D Cartesian mesh: x ∈ [0, P], z ∈ stack height
   ├── Drift-diffusion + Poisson with degenerate Fermi-Dirac
   ├── Per-bias outputs: n(x, z), V(x, z) at each electrode
   └── Output: /carrier_field_<bias_label>.zarr

   ↓

Stage 2: Drude permittivity
   ├── For each Zarr, for each wavelength λ:
   │     ε_ITO(x, z, λ) = ε_∞ - ω_p²(n) / (ω² + iωγ(n))
   │     where ω_p²(n) ∝ n, m*(n) is Kane non-parabolic,
   │     γ(n) is Caughey-Thomas mobility
   └── Output: /eps_<bias_label>.zarr (containing per-λ ε grids)

   ↓

Stage 3: 3D FEM optical solve (NGSolve)
   ├── Build 3D unit cell mesh with periodic Bloch BCs in (x, y)
   │   + PML top/bottom
   ├── For each (bias_label, λ):
   │     - Load ε_ITO from Stage 2 Zarr
   │     - Optionally apply xy-product symmetrization (so the
   │       square patch's x↔y symmetry isn't broken by 2D DEVSIM)
   │     - Build complex VoxelCoefficient for ε_ITO(x, y, z)
   │     - Assemble HCurl Helmholtz + Bloch boundaries
   │     - Solve with BDDC + GMRes
   │     - Extract |r|^2 and phase via probe planes
   └── Output: spectra.csv + spectra_overlay.png
```

### Computational characteristics

- Stage 1: ~1-3 min per bias (~100k DEVSIM equations, ~0.3-1 s per Newton step)
- Stage 2: ~1 s per bias (numpy + scipy on grid)
- Stage 3: 3-15 min per (bias, λ) at medium FEM mesh, 30+ min at fine mesh

Total for 4-bias × 13-λ sweep at medium mesh: ~6-8 hours.

### Tooling choices and why

- **DEVSIM 2.10.0** for Stage 1: mature TCAD library, handles Fermi-Dirac
  drift-diffusion, runs natively on Windows (Python 3.14 env).
  *Caveat:* DEVSIM 2.10's built-in `Fermi(eta)` is broken for `eta ≥ 20`
  (degenerate regime). Workaround: Aymerich-Humet 1981 expression
  (called `halen_F_half_expr` in legacy code, misnamed but correct).
- **NGSolve 6.2.2604** for Stage 3: native HCurl Nédélec edge elements,
  Bloch-periodic BCs, free-form OCC geometry. Used UMFPACK direct LU
  initially → switched to BDDC + GMRes for memory (100 GB → 18 GB).
- **Zarr 3.1.6** for inter-stage data: chosen over HDF5 because directory
  stores tolerate partial writes and allow per-bias independent
  pipeline execution.
- **scipy / numpy** for grid resampling and Drude.
- **matplotlib** for all visualization (no JS frontends).

---

## 4. MAJOR MILESTONES (chronological by task)

Tasks #1-9 — **Project scaffold + 1D Stage 1**
- Installed DEVSIM into Python 3.14
- Built a 1D MOS-cap mesh (just z, no lateral structure)
- Configured Fermi-Dirac drift-diffusion in ITO
- DC bias sweep → dump n(z) per bias
- Computed C-V curve at small signal
- Drude eps from n(z), AC bandwidth (3-dB) calculation
- Lumped-R access model for intrinsic electrical bandwidth

Tasks #10-17 — **2D bias grid + RCWA validation**
- 2D (V_top, V_bot) grid sweep over patch and mirror voltages
- Installed grcwa for RCWA cross-check
- Stage 3 RCWA driver, bias-mode comparison
- the reference modulator §8 paper-strip validation
- Rigorous material audit vs the reference paper

Tasks #18-31 — **NGSolve FEM setup**
- Installed NGSolve, set up FEM groundwork
- Built 2D FEM strip solver (TM Helmholtz with PML)
- Built 3D FEM patch solver (HCurl, Bloch-periodic, PML)
- Wavelength scan + bias sweep + RCWA comparison figures
- z-dependent ITO eps (replacing per-layer-average ε)
- Mesh refinement to close gap to paper Fig 2a
- Memory leak fix in `solve_fem_3d` across repeated solves
- BDDC preconditioner (replacing UMFPACK for the FEM)
- TaskManager threading
- Process-level parallel sweep runner

Tasks #33-36 — **HYPRE port** (ultimately limited)
- Built Netgen-PyMFEM bridge library
- pyhypre + nghypre Windows-native HYPRE 2.31 binding
- AMS preconditioner tuned via SetAlphaPoissonMatrix / SetBetaPoissonMatrix
- **Outcome:** AMS doesn't converge for our **indefinite shifted-Helmholtz**
  (the α and β Poisson auxiliary problems become indefinite when ε
  crosses zero at ENZ). AMS works for SPD problems but not ours.
  CSL (Complex Shifted Laplace) preconditioner is the documented fix
  but is a multi-week implementation. Currently using **BDDC + GMRes**
  as the best available iterative.

Tasks #37-40 — **Higher fidelity physics**
- Mesh efficiency: mirror + patch split into skin / bulk sub-layers
  (27% fewer elements with <1% accuracy loss)
- Density-dependent m*(n) (Kane non-parabolic) and γ(n) (Caughey-Thomas)
- ITO deep-acceptor trap states (optional, off by default)
- 2D DEVSIM scaffold for lateral carrier variation under finite-size
  patch

Tasks #42-44 — **2D DEVSIM topology + Zarr port**
- Added explicit ITO ground contacts at x=0 and x=P (the over-pinning
  source — see Section 6)
- Made mirror and patch proper Dirichlet regions
- 3-electrode bias sweeps with color maps + lineouts (patch-swept and
  mirror-swept)
- Port 2D dumps from HDF5 to Zarr

Tasks #45-49 — **Pipeline expansion + symmetric 3D**
- Stage 2 extension to 2D `n(x,z)` → `ε(x, z, λ)`
- Stage 3 FEM consumer of 2D-extruded `ε(x, z)` as `VoxelCoefficient`
- 4-bias × 13-λ wavelength sweep (V_top, V_bot ∈ {0, ±2V})
- Symmetric 3D `ε_ITO(x, y, z)` via xy-product symmetrization (so
  square patch's x ↔ y symmetry isn't broken by 2D DEVSIM)
- Symmetric 3D sweep completed for patch+2V and patch-2V; killed
  before mirror±2V to make room for recalibration

Task #50 — **Declarative library scaffolding**
- Built `Metasurface_Lib/`, a parallel package with dataclass-based
  `Design`, `Layer`, `Electrode`, `Material`, etc. data model
- Replaces ad-hoc scripts. Pipeline: `run_full_pipeline(design, sweep, out_dir)`
- Clean-room rewrites of Stages 1, 2, 3 (Stage 1 not yet end-to-end
  validated — see Section 6)
- Example design `validation/_reference_device.py` reproduces the reference geometry

Task #51 — **Recalibration (today, partially complete)**
- Diagnosed weak bias modulation in the patch+2V FEM result
- Reverted `ITO_N_BG` from 8e20 to 4e20 (the reference-stated value)
- Reverted `ITO_BANDGAP_EV` 3.75 → 3.6, `N_HFO2_IR` 1.91 → 1.95,
  `N_AL2O3_IR` 1.66 → 1.65
- Established experiment directory structure under
  `experiments/2026_05_28_recalibration/`

---

## 5. TODAY'S SESSION (DETAILED)

This was an intensive debugging session that surfaced two errors
I made and one real physics finding. Recording carefully so the next
agent doesn't repeat them.

### Initial state
- Symmetric 3D FEM sweep running (patch±2V completed, mirror±2V queued).
- User noticed bias-induced reflectivity changes were *tiny* (~0.5% on
  `|r|^2`) — vs the reference-modulator ~20% — and the resonance dip wasn't shifting
  visibly with bias.

### Diagnosis 1: F_{1/2} approximation broken (WRONG)
I claimed the production code's `halen_F_half_expr` was underestimating
F_{1/2} by ~50% at degenerate η. **This was wrong.** I was testing a
**different** naive tanh-blended approximation I had written into the
new library (which *was* 50% off). The legacy `halen_F_half_expr`,
despite the name, is actually **Aymerich-Humet (1981)** — verified to
< 0.2% error vs scipy `quad`. The production code's F_{1/2} is fine.

**Lesson for next agent:** if you suspect a numerical bug, test the
*actual* production expression directly (in legacy:
`from stage1_carriers.mos_cap_1d import halen_F_half`), not a
reimplementation.

### Diagnosis 2: "DC ground location doesn't matter" (WRONG)
At one point I argued that since no current flows at DC, the ITO
contact density doesn't affect static accumulation. **This was wrong.**
ITO is not a perfect conductor — it has finite Thomas-Fermi screening
length (~0.5 nm). Past 1-2 nm from any Dirichlet contact, the carrier
population locks the local potential to the contact's Fermi level. So
ITO grounds at x=0, x=P pin the *bulk* V to ~0 across the whole cell.

**Diagnostic evidence:** unstructured DEVSIM data shows V at the top
of the ITO under the patch reaches only **+24 mV** at V_top=+2V; the
back-of-envelope cap calculation predicts ~+100-300 mV if ITO floats.
That's a ~10× suppression of the cap-induced surface ΔV by the lateral
grounds, leading to ~10× suppression of the bias-induced Δn (and thus
of |r|^2 modulation).

**The fix:** floating-ITO BC (Neumann lateral) for interior unit cells.

### Diagnosis 3: n_bg = 8e20 calibration is stale (CORRECT)
The legacy `ITO_N_BG = 8e20 * 1e6` was tuned in the OLD 1D pipeline
to match the reference-modulator 1300 nm dip. In the current 2D-DEVSIM + 3D-FEM
pipeline with the same value, the dip lands at ~1150 nm. The 8e20
calibration didn't transfer because the pipeline changed underneath it,
and it was probably also compensating for the over-pinning we just
identified. Reverting to the reference-modulator stated **n_bg = 4 × 10^20 cm^-3** is
the right choice.

### Recalibration changes applied
- `Metasurface_Modulator/shared/constants.py`:
  - `ITO_N_BG = 4.0e20 * 1e6` (was 8.0e20)
  - `ITO_BANDGAP_EV = 3.6` (was 3.75)
  - `N_HFO2_IR = 1.95` (was 1.91; Hu et al 2018)
  - `N_AL2O3_IR = 1.65` (was 1.66)
- `Metasurface_Lib/validation/_reference_device.py`: `ITO_N_BG = 4.0e20 * 1e6`

### Attempt to remove ITO grounds in new library (failed)
Set up `Metasurface_Lib/validation/_reference_device.py` with no ITO grounds,
then with a single left-edge anchor, then back to two symmetric
grounds. **All three failed in the new library**, for reasons that
are bugs in the library's `stage1_carriers/`, not in the BC choice:
1. `physics.py` had an incorrect `Phi_c0 = -chi_eV + eta_bg * V_T`
   formula. Fixed to `Phi_c0 = -eta_bg * V_T`.
2. The Joyce-Dixon inverse F_{1/2} approximation diverges for x > ~10.
   Replaced with bisection on Aymerich-Humet.
3. The F_half DEVSIM expression overflows `exp(-eta)` when Newton
   over-steps eta strongly negative. Even with corrected Phi_c0,
   Newton fails on the first 0V solve.
4. Without lateral grounds, the Newton matrix is genuinely singular
   (V shift-invariant in the ITO) — UMFPACK correctly diagnoses this.

The **legacy** `Metasurface_Modulator/stage1_carriers/mos_cap_2d.py`
**still works** with the updated constants. The library's stage1 needs
focused debugging.

---

## 6. CURRENT ISSUES (RANKED)

### Issue A — Lateral ITO ground BCs over-pin the bulk ITO
**Severity:** High. Limits patch-bias modulation by ~10×.

The legacy code uses `ito_ground_left` at x=0 and `ito_ground_right`
at x=P, both V=0 Dirichlet. This pins the bulk ITO under the patch to
near 0V. The correct BC for an *interior unit cell* of a periodic
array (which a typical pixel cell is) is **homogeneous Neumann at the
lateral edges** (no E-field crosses the unit-cell boundary, by
periodicity / symmetry).

But removing the grounds in the new library failed (Section 5). To
make the floating-ITO BC work numerically, the next agent needs:
1. An overflow-safe F_half expression that clips η or uses a
   piecewise form (Maxwell-Boltzmann for η < -10, smooth blend, then
   Aymerich-Humet asymptotic for η > 5).
2. A single small Dirichlet anchor on the ITO (one node or one tiny
   edge) to break the V shift degeneracy and stabilize Newton.
3. Better Newton initial conditions: set V = capacitive-divider
   estimate (V_local_init ≈ V_top * C_top / (C_top + C_bot)) in the
   ITO bulk before the first solve.

### Issue B — Resonance position 150 nm blue of the reference-modulator 1300 nm
**Severity:** Medium. Independent of bias.

Our patch±2V dips both at ~1150 nm; the reference-modulator is at ~1300. After the n_bg
revert (which actually moves the ENZ to the red), this should improve,
but we haven't re-run yet to confirm. Other candidates:
- Patch side L = 175 nm — reference might have a slightly different
  fabricated value
- HfO2 optical index n_HfO2 — went 1.91 → 1.95, probably still off
- ITO m* / Kane α — uses constant 0.27 m_e + α = 0.5; reference uses
  constant 0.35 m_e
- FEM mesh — medium (5 nm cavity) isn't fully converged

### Issue C — New library Stage 1 doesn't end-to-end solve
**Severity:** High (blocks library validation).

The library's `dynameta/stage1_carriers/{physics,devsim_build,solver}.py`
have the bugs documented in Section 5. Until fixed, library experiments
must fall back to the legacy pipeline.

Two fixes already applied in `physics.py`:
- `setup_phi_c0`: now `Phi_c0 = -eta_bg * V_T` (was `-chi_eV + ...`)
- `inverse_F12`: now bisection on Aymerich-Humet (was Joyce-Dixon)

Two fixes still needed:
- `F12_aymerich_humet_expr`: needs overflow-safe form for extreme η
- `devsim_build._setup_semiconductor`: needs better initial conditions
  for V, Electrons; possibly Newton damping; possibly a single anchor

### Issue D — Polarization-symmetry asymmetry (mostly resolved)
**Severity:** Low (validated to be < 0.5% on |r|^2).

The 2D-extruded ε(x, z) model treats the patch as a strip (invariant
in y), breaking the square patch's x ↔ y symmetry. We built an
xy-product symmetrization to fix this (in `stage2_drude/symmetrize.py`).
Smoke comparison: symmetric and 2D-extruded results agree to 3-4
decimal places of |r|^2 for patch-bias case. So the asymmetry is real
but doesn't materially affect the resonance.

### Issue E — the reference-modulator top-bias-only-amplitude vs our different result
**Severity:** Medium (raises modeling questions).

the reference modulator reports: patch-bias produces both resonance shift AND
amplitude modulation; mirror-bias produces only phase change. Our
*pre-recalibration* result was the opposite-magnitude (mirror gave
stronger Δn). After recalibration with floating-ITO BC, we expect
patch-bias to dominate (because the mode field is concentrated at the
top of the cavity). Needs verification.

### Issue F — Wavelength sweep step is too coarse to see resonance shift
**Severity:** Low.

50 nm step probably under-resolves the bias-induced shift (predicted
< 50 nm for ±2V). Use ~10 nm step around the dip for finer resolution.

### Issue G — Schrödinger-Poisson missing (Task #41 roadmap)
**Severity:** Low at first pass; matters for quantitative accumulation-layer profile.

DEVSIM uses classical degenerate Fermi-Dirac statistics (Boltzmann +
Joyce-Dixon equivalent). The actual accumulation layer at the
HfO2/ITO interface (~1-2 nm thick) involves sub-band quantization in
the triangular well. This shifts the n(z) profile but conserves the
integrated areal density. Off-roadmap for now.

### Issue H — AMS preconditioner doesn't converge for our indefinite Helmholtz
**Severity:** Low (BDDC + GMRes works fine).

The signed α and β coefficients in the auxiliary Poisson problems
become indefinite when ε crosses zero (ENZ regime). AMS assumes
definite SPD. Documented as a known limitation. CSL (Complex Shifted
Laplace) is the documented academic fix; multi-week to implement.

---

## 7. ROADMAP FORWARD

### Priority 1: Fix new library Stage 1 (~1-2 days)
1. Overflow-safe F_half expression. Three styles to consider:
   - Piecewise: η < -10 → exp(η); -10 ≤ η ≤ 5 → Aymerich-Humet;
     η > 5 → asymptotic (4/3√π)·η^1.5. Blended via tanh.
   - Single expression with `exp(-min(eta, ETA_CAP))` style clamping
     using `0.5 * (eta + ETA_CAP - |eta - ETA_CAP|)` or similar
     min/max via abs.
   - Sub-block: define `eta_safe` as a node model with clipping, then
     use it in F_half.
2. Better initial conditions: set V(z) in the ITO from the
   capacitive-divider estimate at the start of each solve.
3. Single small Dirichlet anchor on the ITO at a corner. Document
   that "this represents the connection to the far-away peripheral
   ground pad."
4. Test: 0V solve should give uniform n = n_bg. +2V patch solve
   should give 30-60% Δn at the top surface under the patch.

### Priority 2: Validate corrected legacy 0V resonance (~3-4 h)
With the constants reverted, run the legacy pipeline at 0V only:
- `Metasurface_Modulator/stage1_carriers/mos_cap_2d.py` → 1 bias point
- `Metasurface_Modulator/stage2_drude/run_eps_xy_sweep.py` → Drude
- One wavelength scan in `Metasurface_Modulator/stage3_optical/fem/`
  at coarse mesh

Outputs to `experiments/2026_05_28_recalibration/01_validation_0V/`.

Verify: where does the resonance dip land? If still 1150 nm, investigate
other candidates (L, n_HfO2, m*). If 1300 nm ± 50 nm, the n_bg revert
was the main culprit.

### Priority 3: Validate floating-ITO BC gives reference-magnitude modulation (~1 day)
Once Priority 1 is done, rerun patch±2V Stage 1 with the new BC.
Compare Δn at the top of ITO to: (a) the pre-recalibration ~4%, (b)
the back-of-envelope ~30-60%. Should land in the 20-50% range.

### Priority 4: Full 4-bias × wavelength sweep with corrections (~6-8 h)
Once Priorities 1-3 confirm physics is right, kick off the full
`run_full_pipeline(reference_modulator_design, 4_bias_sweep, ...)`. Outputs
to `experiments/2026_05_28_recalibration/02_patch_pm2V/` and
`03_mirror_pm2V/`. Will give the dataset to compare against reference
Figure 2a.

### Priority 5: Rename `halen_F_half_expr` → `aymerich_humet_F_half_expr` (~30 min)
Cosmetic but reduces future confusion. Update all call sites.

### Priority 6: Pixel-level cascade (~1 week)
Build a meta-design driver that takes a pixel layout (10×10 unit
cells with shared electrodes) and predicts pixel-level reflectivity
+ phase modulation. This is the user's actual product target.

### Priority 7: 3D DEVSIM (Tasks #41 area) (~2 weeks)
For accurate patch-corner physics (currently approximated by
xy-product symmetrization), a true 3D DEVSIM would be ideal. Significant
work due to memory + Newton convergence on 3D drift-diffusion.

### Priority 8: Schrödinger-Poisson roadmap (Task #41) (~3-4 weeks)
Quantum confinement in the 1-2 nm accumulation layer. Open-source
options include `SesameTK` and `nextnano`, or hand-roll.

### Priority 9: CSL preconditioner for ENZ-regime FEM (~3-4 weeks)
Would unblock AMS / HYPRE for our 3D HCurl indefinite Helmholtz.
Would also help with the medium-vs-fine mesh memory tradeoff.

---

## 8. KEY FILES & LOCATIONS

### Legacy pipeline (works, but with the BC limitation)
- `Metasurface_Modulator/shared/constants.py` — material parameters,
  reverted today
- `Metasurface_Modulator/stage1_carriers/mos_cap_1d.py` — 1D DEVSIM,
  has the correct F_{1/2} expression (`halen_F_half_expr`, misnamed
  but is Aymerich-Humet)
- `Metasurface_Modulator/stage1_carriers/mos_cap_2d.py` — 2D DEVSIM
  with the over-pinning ITO grounds (the main BC issue)
- `Metasurface_Modulator/stage1_carriers/symmetrize_xy.py` —
  xy-product symmetrization
- `Metasurface_Modulator/stage1_carriers/carrier_map_2d.py` — full
  patch-and-mirror sweep driver (the one used today for the pre-
  recalibration runs)
- `Metasurface_Modulator/stage1_carriers/plot_device_heatmap.py` —
  cross-section visualization (both full-stack and cavity-zoom)
- `Metasurface_Modulator/stage2_drude/run_eps_xy_sweep.py` — Stage 2
  on 2D Zarrs
- `Metasurface_Modulator/stage3_optical/fem/geometry_3d.py` — 3D
  OCC + mesh
- `Metasurface_Modulator/stage3_optical/fem/driver_3d.py` — FEM solve
  (BDDC + GMRes)
- `Metasurface_Modulator/stage3_optical/fem/bias_sweep.py` — 2D ε(x,z)
  CoefficientFunction loader + `_ito_eps_cf_3d_xyz_sym` symmetric loader
- `Metasurface_Modulator/stage3_optical/fem/sweep_4biases_3d_sym.py`
  — the 4-bias sweep driver (today's run)

### New library (partially complete)
- `Metasurface_Lib/pyproject.toml` — installable package metadata
- `Metasurface_Lib/README.md` — library docs
- `Metasurface_Lib/RECALIBRATION_PLAN.md` — today's plan
- `Metasurface_Lib/HANDOFF.md` — THIS DOCUMENT
- `Metasurface_Lib/dynameta/design/` — dataclass spec
  (`Design`, `Layer`, `Electrode`, etc.) — VALIDATED, works
- `Metasurface_Lib/dynameta/stage1_carriers/` — DEVSIM build,
  solve, Zarr I/O. **HAS BUGS** (Issue C)
- `Metasurface_Lib/dynameta/stage1_carriers/fermi_dirac.py` —
  high-accuracy F_{1/2} helpers (independent module, works)
- `Metasurface_Lib/dynameta/stage2_drude/` — Drude + xy-product
  symmetrization. Logic looks correct, not yet end-to-end tested
- `Metasurface_Lib/dynameta/stage3_optical/` — NGSolve mesh,
  ε loader, solver. Not yet end-to-end tested
- `Metasurface_Lib/dynameta/pipeline.py` — orchestrator
- `Metasurface_Lib/validation/_reference_device.py` — reference design.
  `ITO_N_BG = 4e20`, ITO grounds present (after rollback)

### Experiment directories (post-recalibration)
- `Metasurface_Modulator/experiments/README.md`
- `Metasurface_Modulator/experiments/2026_05_28_recalibration/README.md`
- `Metasurface_Modulator/experiments/2026_05_28_recalibration/01_validation_0V/`
- `Metasurface_Modulator/experiments/2026_05_28_recalibration/02_patch_pm2V/`
- `Metasurface_Modulator/experiments/2026_05_28_recalibration/03_mirror_pm2V/`
- `Metasurface_Lib/examples/outputs/exp_2026_05_28_recalibration/01_validation_0V/`
- `Metasurface_Lib/examples/outputs/exp_2026_05_28_recalibration/02_reference_modulator_quick/`
- `Metasurface_Lib/examples/outputs/exp_2026_05_28_recalibration/03_reference_modulator_full/`

### Pre-recalibration data (legacy, 8e20 + grounds)
- `Metasurface_Modulator/stage1_carriers/outputs/n_of_xy_2d/` — 10 Zarrs
  (5 patch-swept, 5 mirror-swept)
- `Metasurface_Modulator/stage1_carriers/outputs/device_heatmap_*.png` —
  cross-section visualizations
- `Metasurface_Modulator/stage2_drude/outputs/eps_xy_2d/` — 10 Drude eps
  Zarrs
- `Metasurface_Modulator/stage3_optical/fem/outputs/sweep_4biases_3d_sym/`
  — patch±2V wavelength sweep CSV (mirror sweeps not run; process killed)

---

## 9. CRITICAL GOTCHAS / LESSONS LEARNED

### Numerical / DEVSIM
1. **DEVSIM 2.10's built-in Fermi function is broken for η ≥ 20.** Always
   use the Aymerich-Humet workaround (`halen_F_half_expr` in legacy;
   `F12_aymerich_humet_expr` in the new library).
2. **`halen_F_half_expr` is a misnomer.** The formula is
   Aymerich-Humet (1981), not the original Halen-Pulfrey form.
   Don't rewrite it.
3. **DEVSIM expression syntax** doesn't support `sqrt(x)` directly.
   Pre-compute constants in Python; use `pow(x, 0.5)` if needed.
4. **`exp(-eta)` in F_half overflows** when Newton over-steps eta
   strongly negative. Either bound the Newton step or clip eta in
   the expression.
5. **Newton initial conditions matter.** Setting `Potential = 0` in
   the ITO and `Electrons = n_bg` works at zero bias. Setting V to
   Phi_c0 is WRONG — gives discontinuity at the contacts.
6. **`Phi_c0 = -eta_bg * V_T`**, NOT `-chi_eV + eta_bg * V_T`. The
   chi_eV term doesn't belong; we work in a local-potential frame
   anchored at V=0 = far-away peripheral ground.
7. **DEVSIM "Triangle has no region" warnings** during build are
   typically benign for layers with `patch_footprint` extent — the
   triangles above the stack outside the patch have no region by
   design.

### NGSolve / FEM
8. **VoxelCoefficient axis order is `(Nz, Ny, Nx)`**, not (Nx, Ny, Nz).
   Transpose before passing.
9. **VoxelCoefficient supports complex dtype natively.** Use that
   instead of `Re + 1j * Im` composition (which fails BilinearForm
   assembly).
10. **AMS preconditioner doesn't converge for indefinite shifted-
    Helmholtz.** ε crossing zero (ENZ) makes the α / β Poisson
    auxiliary problems indefinite, violating AMS's SPD assumption.
    Use BDDC + GMRes instead.
11. **UMFPACK is fine for DEVSIM 2D (100k equations) but uses 100 GB
    on NGSolve 3D HCurl (84k tets).** Switch to BDDC + GMRes for
    NGSolve.
12. **Mesh refinement** in NGSolve happens via per-region `maxh` on
    OCC solids before `GenerateMesh`. Skin / bulk splits help (~27%
    fewer elements).

### Physics / BCs
13. **Lateral ITO ground BCs over-pin the bulk ITO** via Thomas-Fermi
    screening. For interior unit cells, use Neumann lateral BCs with
    a single anchor.
14. **"DC ground location doesn't matter" is wrong** for finite-Debye-
    length semiconductors. ITO is not a perfect conductor; bulk V is
    locked to the nearest Dirichlet's Fermi level.
15. **n_bg empirical calibrations don't carry across pipeline changes.**
    The 1D-pipeline `n_bg = 8e20` empirical fit was a fudge that
    didn't transfer to 2D-DEVSIM + 3D-FEM. Use literature values and
    audit when the resonance position changes.

### Process / project
16. **Background tasks in this repo** are launched via `Bash` with
    `run_in_background=true`. Output goes to a temp file. Use the
    notification system; don't poll.
17. **Python module cache** means changes to imported source files
    don't affect a running process. So safe to edit `shared/constants.py`
    while a sweep runs, IF the sweep has already started — but a new
    spawn would pick up the change.
18. **Zarr 3.x** assignment is `group[name] = arr` (the simple form).
    `create_array(name, data=arr)` with `shape` separately conflicts.
19. **The user prefers Windows cp1252-safe printing** — no Unicode
    arrows / Greek letters in `print()`. Use ASCII equivalents.

---

## 10. QUICK-START FOR NEXT AGENT

If you're picking this up, do these in order:

```powershell
# 1. Orient yourself
cd D:\Metacept\Neurophos\Python_Test_Scripts\Metasurface_Lib
Get-Content HANDOFF.md          # this file
Get-Content RECALIBRATION_PLAN.md
cd ..\Metasurface_Modulator
Get-Content experiments\README.md

# 2. Verify legacy pipeline still builds (1 min)
cd ..\Metasurface_Modulator
python -c "from stage1_carriers import mos_cap_2d as M; spec = M.Mesh2DSpec(); import devsim as ds; _, regs = M.build_device_2d(mesh_spec=spec); print('OK:', ds.get_region_list(device=ds.get_device_list()[0]))"

# 3. Confirm Stage 1 in legacy converges at 0V with new constants
#    (target: n in ITO should be uniform at 4e20 m^-3 = 4e26 cm^-3)
python -m stage1_carriers.mos_cap_2d --V-top 0 --V-bot 0 \
    --out experiments\2026_05_28_recalibration\01_validation_0V\stage1.zarr

# 4. Inspect the n profile
python -c "
import zarr, numpy as np
r = zarr.open_group('experiments/2026_05_28_recalibration/01_validation_0V/stage1.zarr', mode='r')
ito = r['regions']['ito']
n = np.asarray(ito['Electrons'][:])
print('n range:', n.min(), n.max(), 'mean:', n.mean(), 'expect ~ 4e26')"

# 5. If that looks right, work on Issue C (library Stage 1) next.
#    See Metasurface_Lib/dynameta/stage1_carriers/{physics,devsim_build}.py
#    Specific fixes needed:
#    - overflow-safe F_half expression
#    - better Newton initial conditions
#    - validate the new physics module's invert_F12 bisection
```

If you need to **continue the user's pixel-design work** (their
ultimate goal):
- They want each pixel = 10×10 unit cells driven by a shared
  ITO ↔ mirror bias, polarity-flippable
- The patch is electrically floating in their design
- Modulation between |r|^2 high and low under polarity flip is the key
  observable
- Requires Issue A (floating-ITO BC) to be resolved before predictions
  are quantitatively meaningful

---

## 11. CONTACT / OWNER

User: Andrew Traverso (Neurophos / Metacept)
Email: travaj@gmail.com
Background: optical physicist on free-space optics TX/RX systems
Active project: this metasurface modulator for an SLM

Working preferences:
- Windows cp1252 — no Unicode arrows, Greek letters in print()
- Numbered subdir per experiment under `experiments/<date>_<theme>/NN_purpose/`
- Descriptive HDF5/Zarr filenames encoding the bias / wavelength / etc.
- Keep experiment-specific code out of the shared library; library is
  for general device descriptions only
