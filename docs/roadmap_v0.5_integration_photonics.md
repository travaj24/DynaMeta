# DynaMeta v0.5 roadmap: Lumenairy integration + silicon photonics

**Date:** 2026-06-10 (owner-directed)
**Scope:** (A) integrate Lumenairy's RCWA and PMM solvers as DynaMeta optical backends with
seamless bidirectional translation between the two libraries; (B) the silicon-photonics
program (waveguide modes, phase shifters, EME components); (C) carry-over items.

---

## A. Lumenairy RCWA + PMM integration

### A0. The architectural decision: BRIDGE, not vendor

DynaMeta takes Lumenairy as an **optional dependency** (`pip install dynameta[lumenairy]`,
floor `lumenairy>=5.14`) and adds a bridge package `dynameta/optics/lumenairy_bridge/`.
The 2026-06-01 wishlist (`docs/lumenairy_rcwa_port_wishlist.md`) framed this as a code copy;
the owner direction that the two libraries remain COMPLEMENTARY with seamless translation
tips the decision to a live dependency:

- Lumenairy is actively maintained (v5.6 -> v5.14.1 since the wishlist, including audited
  P1 fixes, dispersive sweeps, PMM 2-D parity, device-geometry builders). A vendored copy
  forks away from every future fix.
- The conventions are IDENTICAL (`exp(-i omega t)`, `Im(eps) > 0` absorbers, metres,
  radians) -- verified in the wishlist audit -- so the bridge is a thin adapter, not a
  translation layer that would justify owning the code.
- DynaMeta's seam-side prep is DONE and validated: the `LayeredStackSolver` Protocol, the
  `LayeredSlab` spec set that deliberately mirrors `RCWAStack.add_layer`
  (scalar / `eps_cell` / `eps_tensor_cell` / `shapes`), the z-slicers, and the
  `optical_solver`/`solve_sweep` pipeline seam.
- CI/installability: the bridge imports lazily; DynaMeta without Lumenairy is byte-identical
  (the same opt-in pattern as devsim/ngsolve/jax).

### A1. RCWA backend bridge (task #169)

`dynameta/optics/lumenairy_bridge/rcwa_backend.py`:

- `make_lumenairy_rcwa_solver(*, n_orders=..., formulation=..., n_slices=None, ...)` ->
  an `optical_solver` with the exact pipeline seam signature
  `fn(design, geo, eps_by_region, lambda_m, n_super, n_sub) -> OpticalResult`, PLUS
  `solve_sweep` (sweep-aware fast path; Lumenairy's dispersive wavelength sweeps map onto
  it directly).
- Geometry translation `design_to_rcwa_stack(design, lambda_m, eps_by_region=None)`:
  - uniform layers -> `add_layer(eps=...)`; graded EpsFields -> sliced layers (existing
    `slice_eps_field`); uniform tensor EpsFields -> `eps_tensor_cell`;
  - **laterally structured layers (Inclusions) -> Lumenairy patterned layers** -- the piece
    no current DynaMeta frequency-domain backend except the FEM can do. Analytic shapes map
    to Lumenairy shape objects where the vocabulary matches; the general fallback rasterizes
    via the validated fdtd_seam lateral-inclusion rasterizer into `eps_cell`.
- Result mapping -> `OpticalResult`: R/T/A from the order sums + 0-order complex Jones r/t
  (phase), `A_independent` cross-check, per-order data in a documented extras side-channel,
  per-layer absorption -> `per_region_absorption` keyed by layer name (the drivers-glue
  convention).
- ALSO implements the `LayeredStackSolver` Protocol (second concrete backend after TMM) so
  `LayeredStack` consumers get RCWA transparently.

**Validation gates (each an exit-gated script):**
1. Unstructured stacks: bridge == `tmm_reference` (machine-tight; Lumenairy itself is
   Airy-validated to ~1e-16).
2. Graded carrier slab: bridge == graded-TMM == FEM (the existing `graded_tmm_vs_fem`
   triangle extended to a third backend) -- the DEVSIM n(z) -> RCWA chain.
3. Patterned cell: the Park Au-patch/ENZ-ITO cell vs the NGSolve FEM (the wishlist's
   "missing independent oracle") and vs the existing grcwa cross-check machinery.
4. Oblique/conical vs FEM/TMM within the measured envelopes.
5. Sweep path == per-wavelength path; dispersive-material sweep vs DynaMeta Material models.
6. Tensor layer (uniform LC tilt) vs the UPML tensor FEM (`lc_tilted_fem` geometry).

### A2. PMM backend bridge (task #170) -- SHIPPED

Same seam, `pmm_backend.py`: the high-accuracy path for 1-D gratings (incl. TM/metals where
RCWA's factorization is the bottleneck) and the full-3x3 / out-of-plane tensor cases.
Primary roles in DynaMeta:
- **Reference oracle**: 1-D PMM has no Fourier-factorization accuracy floor -- the
  convergence referee for RCWA settings on hard (metallic/tensor) cells.
- **Tensor specialist**: out-of-plane (xz/yz-coupled) tensors that the in-plane RCWA tensor
  path does not carry -- magneto-optic and slanted-LC cells.
Shipped `validation/lumenairy_pmm_bridge.py` (all green): unstructured vs TMM 8.9e-14;
referee ladder -- the RCWA bridge's lamellar 1-D fast path converges to the spectral PMM
reference 8.9e-2 -> 7.3e-4 on a lossy metal TM grating; gyrotropic (3,3) tensor slab vs the
hand-derived circular-eigenmode Jones 4.3e-14; scope guards (partial-y / laterally
structured grids / conical raise loudly). Scope: 1-D lamellar + uniform tensors; no
transmission Jones (OpticalResult.t = None).

Landed alongside (the synergy glue A2 exposed):
- `collapse_regions_to_layers` (core/layered.py): the run_pipeline/FEM bridge emits
  MESH-region-keyed eps_by_region ('ito_inpatch', 'grating__incl0', ...); both Lumenairy
  bridges now collapse those to design-layer keys automatically, so
  `run_pipeline(optical_solver=make_lumenairy_rcwa_solver(...))` works against the DEFAULT
  DEVSIM/NGSolve builders out of the box.
- Lamellar 1-D fast path in the RCWA bridge (y-invariant full-period rectangles and
  y-invariant gridded fields solve as true 1-D RCWA) + `formulation=` plumbed through.
- Per-layer absorption attribution verified (GATE F in `lumenairy_rcwa_bridge.py`):
  per_region_absorption keyed by design layer, graded slabs aggregate, closure to 1e-16.
- Flagship example `examples/lumenairy_gated_grating.py`: DEVSIM gated-ITO accumulation ->
  graded Drude eps(z) -> RCWA bridge (1-D fast path, absorption attribution) with the PMM
  bridge as cross-method referee (|dR| 4e-3 ungated); max gate modulation |dR| = 0.26.

### A3. Bidirectional translation tools (task #171)

`dynameta/optics/lumenairy_bridge/translate.py` -- the "seamless synergy" layer, both ways:

- DynaMeta -> Lumenairy: `design_to_rcwa_stack` (above) exposed as a public utility, plus
  `carrier_field_to_layers(cf, design, n_to_eps, lambda_grid)` -- DEVSIM carrier profiles
  (or thermal/effect-modulated EpsFields) as ready-to-solve Lumenairy graded layers. The
  killer workflow: Stage-1 carriers in DynaMeta, patterned optics in Lumenairy, one call.
- Lumenairy -> DynaMeta: `rcwa_stack_to_design(stack, materials=...)` -- a Lumenairy device
  becomes a DynaMeta `Design` so the multiphysics axes (carriers, thermal, reliability,
  effects) run on the SAME device; `rcwa_result_to_optical_result` for result-level interop.
- Materials: `optical_model_to_lumenairy_material` / back (incl. dispersive tables; both
  sides are exp(-i omega t)).
- Round-trip gates: design -> stack -> design preserves geometry + eps at every wavelength;
  result-level equivalence through both directions.

### A4. Sequencing + Lumenairy-side asks

Order: A1 gates 1-2 (unstructured + graded; lowest risk) -> A1 gates 3-6 (patterned/tensor)
-> A3 translation + round-trips -> A2 PMM. Remaining Lumenairy-side wishlist items (P1 2-D
stack autodiff if still open at v5.14.1, P2 normal-vector 2-D FFF) are tracked as Lumenairy
work, NOT blockers: the bridge ships against what v5.14.1 provides and inherits upgrades.

---

## B. Silicon photonics program

Owner-approved 2026-06-10. Existing assets: `Silicon_Photonics_Sims/` (prior sims, to be
mined for reference cases), the NGSolve FEM infrastructure, DEVSIM bipolar DD, the effects
family, and the reliability axis.

### B1. Waveguide cross-section physics (Phase 1 -- where DynaMeta is unique)

- **Vector mode solver**: NGSolve HCurl/H1 mixed eigenproblem on a waveguide cross-section
  (SOI strip/rib); n_eff, group index, mode profiles, bend loss (perturbative first).
  Oracles: analytic slab modes, published SOI strip n_eff tables, Marcatili.
- **Carrier-driven phase shifter**: DEVSIM 2-D pn junction across the rib (depletion (and
  injection) modulator) -> Soref-Bennett dn(N,P), dalpha(N,P) at 1310/1550 nm as a new
  EffectModel -> mode-solver perturbation -> n_eff(V), loss(V), V_pi*L, C(V)/bandwidth.
  This is the existing carriers->effects->optics spine pointed at a waveguide.
- **Thermo-optic phase shifter**: the thermal FEM + dn/dT (Si) on the same cross-section;
  heater efficiency (mW/pi), thermal bandwidth, crosstalk between neighbors.
- Reliability carry-over: the EM/TDDB/BTI post-processors apply to the modulator drive
  unchanged via the drivers glue.

### B2. Along-the-chip propagation (Phase 2)

- Coupled-mode theory for directional couplers (from Phase-1 mode overlaps).
- **EME** (eigenmode expansion) for tapers, MMIs, splitters: mode solving per cross-section
  + overlap matrices + cascaded S-matrices. MMI self-imaging gives clean analytic oracles.

### B3. Full 3-D component FDTD (Phase 3 -- explicit buy-vs-build)

Arbitrary junctions need mode-source ports, TF/SF injection, and lateral PML -- genuine
engine work on the periodic-Bloch FDTD. Mature tools exist (MEEP, Tidy3D, Lumerical).
Decision deferred until B1/B2 demonstrate the need for in-house multiphysics-coupled
propagation; default is interop, not reimplementation.

---

## C. Carry-over (status pointers)

- DG oxide hard wall: EXPERIMENTAL (`setup_dg_hard_wall`); Newton stalls on the
  log-singular boundary layer at practical tolerances -- continuation plan in
  `validation/_dg_hard_wall_wip.py` + the function docstring. Bipolar DG twin queued after.
- GPU: linear + nonlinear FDTD kernels hardware-validated (CUDA 13.1 / RTX 4070 Ti);
  remaining GPU items are performance, not capability.
- Oblique FEM envelope: validated 0-45 deg; angle-aware PML would extend toward the 60-deg
  cap (open, low priority once RCWA covers oblique periodic cells).
- Gummel: validated on unipolar ohmic transport; gated-accumulation convergence unproven.
- Lasing/cavity, C(T), GPU nonlinear: SHIPPED 2026-06-10 (see physics_depth_roadmap.md).
