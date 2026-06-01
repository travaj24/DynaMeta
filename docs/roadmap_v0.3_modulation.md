# DynaMeta v0.3 -- General modulation-physics roadmap

**Date:** 2026-06-01
**Premise:** broaden DynaMeta from a free-carrier / ENZ (ITO nanopatch) tool into a
general **charge/field-dynamics -> optical-modulation** simulator covering the *typical*
modulation mechanisms, not just plasma dispersion. Confirmed scope (owner): **all four
mechanism families** -- field-effect EO, thermo-optic, QCSE/MQW electro-absorption, and
reconfigurable (PCM/LC/graphene).

This is a planning artifact (companion to `docs/audit/2026-06-01-adversarial-audit.md`).
It supersedes the narrow "big four" (RCWA / AC / field-aware seam / optimization) by
promoting the **field-aware seam + tensor eps** to the keystone and folding in reusable
work from the sibling `Metasurface_Modulator` project (see Part D).

---

## Part A -- The reframe

DynaMeta's real abstraction is **not "ENZ."** It is a three-stage bridge:

> a *driving-physics* solve  ->  a *material-response* map  ->  an *optical* solve

joined by the alignment keystone. Today every stage is hardwired to one mechanism:

| stage | today | needs to become |
|---|---|---|
| driver (Stage 1) | DC carrier density `n` (DEVSIM Poisson/DD) | + electrostatic E-field, temperature T, AC/transient n, QW Stark state, LC director, PCM phase |
| response (`NToEpsMap`) | **scalar, isotropic** Drude(n) | `EffectModel(local_fields, lambda) -> ` **tensor, dispersive** eps (or a surface conductivity) |
| optics | frequency-domain FEM (scalar eps) | tensor eps + surface-current BC; + RCWA / FDTD backends |

Free-carrier/ENZ is actually one of the *harder, more niche* mechanisms; most common
modulators run on effects the library structurally cannot express yet.

---

## Part B -- The modulation-mechanism landscape

| Mechanism | Driving field | eps response | Typical device / material | Status |
|---|---|---|---|---|
| Free-carrier / plasma | carrier density n | scalar, dispersive (Drude) | ITO/ENZ, Si (Soref-Bennett), TCOs | **have it** |
| **Pockels (linear EO)** | applied E | **tensor** r*E | thin-film LiNbO3, BTO, AlScN, EO polymer | needs E + tensor |
| **Kerr / DC Kerr** | E^2 | scalar/tensor | Si, polymers; all-optical | needs E |
| **Franz-Keldysh (EA)** | E | scalar, field-shifted edge | bulk Ge/GaAs EAMs | needs E + band-edge model |
| **QCSE (EA)** | E across a QW | exciton shift -> d(alpha) | III-V MQW EAMs (datacom) | extend the 1D Schrodinger solver |
| **Thermo-optic** | temperature T | scalar, dn/dT | Si photonic heaters (most common tuner) | needs a thermal solve |
| **Liquid crystal** | E -> director | **uniaxial tensor** along n-hat | LCoS SLMs, LC metasurfaces | director PDE + tensor |
| **Phase-change (PCM)** | thermal/electrical pulse | discrete/continuous (n,k) | GST, Sb2S3, VO2 metasurfaces | two-phase / state eps |
| **2D (graphene/TMD)** | gate -> surface sigma | **surface-current BC** | graphene EA/phase modulators | sheet-conductivity BC |
| **MEMS / strain / acousto-optic** | displacement/strain | geometry or photoelastic tensor | tunable-gap metasurfaces, AOMs | geometry sweep / photoelastic tensor |
| **Magneto-optic** | magnetic field | antisymmetric off-diagonal tensor | isolators, MO modulators | non-reciprocal tensor |

Two things block the entire right-hand column: (1) the response map sees only `n`, and
(2) eps is scalar-isotropic.

---

## Part C -- The plan

### Phase 0 -- Architectural keystones (shared by all four families; build ONCE)
- **0a FieldBundle seam.** Generalize `NToEpsMap` -> `EffectModel(fields: dict, lambda) -> eps`;
  the bridge assembles + passes the full local-field dict `{E = -grad(phi), n, T, ...}` on
  the aligned grid. Drude becomes one `EffectModel`. An `EffectModel` registry + composition
  (effects stack: e.g. background birefringence + Pockels, or thermo-optic + free-carrier).
- **0b Tensor/anisotropic eps end to end.** `EpsField` carries a per-point 3x3 eps;
  `VoxelCoefficient`/assembler + the FEM curl-curl weak form handle a tensor coefficient.
  *Gate:* reduces EXACTLY to today's scalar results when the tensor is isotropic.

### Phase 1 -- Field-effect EO (Pockels / Kerr / Franz-Keldysh)  [highest leverage]
- **1a Electrostatics driver:** a pure-dielectric Laplace/Poisson solve for E = -grad(phi)
  (no mobile carriers) -- cheap, a subset of the existing DEVSIM Poisson.
- **1b `PockelsModel`** (EO tensor r + crystal orientation), `KerrModel`, `FranzKeldyshModel`
  (field-shifted band edge -> d(alpha), Kramers-Kronig d(n)).
- *Oracle:* analytic Vpi*L for a thin-film LiNbO3 phase arm; eps -> isotropic at r=0.
  **Delivers a complete non-carrier modulator end to end -> proves the generalized spine.**

### Phase 2 -- Thermo-optic / electro-thermal
- **2a Thermal driver:** steady + transient heat equation (FEM), Joule source from the
  electrical solve (electro-thermal). Emits T into the FieldBundle.
- **2b `ThermoOpticModel`** (dn/dT) + material thermal params (k, Cp, dn/dT).
- *Oracle:* Si-heater d(lambda)/d(power); no shift at dT=0. (Reuses the Phase-0 seam.)

### Phase 3 -- QCSE / MQW electro-absorption
- Extend the **existing `SchrodingerPoisson1D`** with an in-well field -> Stark-shifted
  sub-bands + a simple exciton model -> `ElectroAbsorptionModel` (d(alpha), d(n) via KK).
- *Oracle:* known QCSE redshift vs field; flat-band reduction. (Biggest reuse of what exists.)

### Phase 4 -- Reconfigurable (PCM / LC / graphene)
- `PCMModel`: crystalline-fraction state -> effective (n,k) (GST/Sb2S3/VO2); optional
  thermal-pulse switching driver.
- **LC director driver** (Frank-elastic + field) -> uniaxial tensor along n-hat (reuses
  Phase-0 tensor eps). *Oracle:* Freedericksz threshold.
- **Graphene/2D surface conductivity** sigma(E_F, omega) (Kubo) as a surface-current BC in
  the FEM (not volumetric eps). *Oracle:* analytic sheet reflection.

### Cross-cutting enablers (interleaved; shared by every mechanism)
- **Lumped-RC bandwidth** (near-term; PORT from Modulator, Part D) -- a real modulator FOM now.
- **RCWA backend** -- **PORT Lumenairy's native RCWA (`RCWAStack`, ~v5.6), do NOT rebuild.** It
  already provides multi-layer 1-D/2-D periodic, conical, uniform/`eps_cell`/full-tensor
  `eps_tensor_cell`/analytic-`shapes` layers, R/T/A + 0-order complex Jones, in the SAME
  `exp(-i omega t)`/`Im(eps)>0` convention (no sign bridge). DynaMeta adds a `LayeredStackSolver`
  Protocol + an `RcwaSolver` adapter + a z-slicer (graded/tensor `EpsField` -> `RCWAStack`
  layers). Slot right after Phase 1 (tensor eps) so it serves the EO mechanisms; it is the fast
  forward + independent oracle + the optimization enabler. Lumenairy gaps to close first (P1
  2-D/stack autodiff, P2 normal-vector FFF, P3 2-D patterned-tensor rigor) are itemized in
  `docs/lumenairy_rcwa_port_wishlist.md`. **RCWA-independent prep DONE (2026-06-01):** the
  `LayeredStackSolver` seam + a concrete `TmmLayeredSolver`, the `core.layered` slab model +
  z-slicer, and a pluggable `run_pipeline(optical_solver=...)` are built and validated vs the
  FEM -- so the port reduces to wiring `RCWAStack` to the seam + the structured-slab path.
- **Iterative/scaling solver** -- BDDC is in; a NGSolve-AMS/HYPRE binding is the next rung for
  large 3D (Part D). The non-ENZ mechanisms are well-conditioned, so this is now viable.
- **AC/transient carriers** -- the full dynamics axis (after the lumped-RC stopgap).
- **Optimization / inverse-design** -- rides on a fast forward solver (RCWA).
- **FDTD** -- last, for nonlinear/ultrafast/broadband (Kerr, all-optical).

---

## Part D -- Reuse from the Metasurface_Modulator project

The sibling imperative pipeline (`Python_Test_Scripts/Metasurface_Modulator/`) contains
work that was never fully used for the Park experiment but is directly portable. Audited
2026-06-01:

1. **Lumped-RC -> intrinsic f_3dB bandwidth** (`stage4_system/access_R_f3dB.py`). Model:
   `R_access = rho_sheet * L_path / W_pad`, `rho_sheet = 1/(q n mu t)`,
   `f_3dB = 1/(2 pi R_access C_cell)`; C_cell from the C(V) curve. It produced ~15 GHz for
   the Park cell. **PORTED (2026-06-01).** `analysis.sheet_resistance_ohm_sq`,
   `analysis.lumped_rc_bandwidth` (-> R_access, C_cell, f_3dB) and
   `analysis.switching_energy_per_area` (0.5 C V^2) now compose with `gate_cv` for an
   intrinsic f_3dB **with zero new solver**. Validated: the synthetic pytest reproduces the
   Modulator's ~15.4 GHz Park number; `validation/bandwidth_cv.py` runs a real SP-carrier C-V
   sweep -> f_3dB ~5 GHz (GHz band, monotonic). The on-ramp to the full AC/transient phase.
2. **The "HYPRE"/iterative-solver work** (`stage3_optical/fem/{compare_solvers,
   fine_mesh_with_iterative,smoke_gmres}.py`). *Correction:* there is no literal HYPRE/AMS
   code -- the only preconditioner is NGSolve **BDDC** (`driver_3d.py:179`), which DynaMeta
   already ported (`linear_solver` in {umfpack, bddc_cg, bddc_gmres}). What IS portable:
   (a) a **direct-vs-iterative benchmark harness** (|r|^2 agreement + wall time) --
   **PORTED (2026-06-01)** as `validation/solver_comparison.py` (umfpack/bddc_gmres/bddc_cg
   agree to ~8e-9; bddc_cg fastest); (b) the **scaling lesson**: UMFPACK is infeasible
   at fine meshes (100s of GB) -> BDDC-GMRes is the O(n)-memory path. **Strategic reframe:**
   the audit flagged AMS/HYPRE as an ENZ dead end (indefinite near-zero-eps), but the new
   EO / thermo-optic / dielectric mechanisms are **well-conditioned and physically larger**
   than the deep-subwavelength ENZ patch -- so a real NGSolve-AMS/HYPRE binding (the rung
   above BDDC) becomes worthwhile EXACTLY as we broaden the mechanism set. This is the right
   home for the "HYPRE" item, and it was not viable in the ENZ-only world.
3. **RCWA -- SUPERSEDED by the Lumenairy port.** The Modulator's grcwa pipeline
   (`stage3_optical/rcwa/`) was under-converged for the Park metal patch precisely because grcwa
   lacks Li factorization (erratic ~+/-25 nm resonance noise in 2-D metals). Lumenairy's native
   RCWA already fixes this (correct Li inverse-rule, principal-branch stability, dual-Laurent
   2-D, Wood-anomaly regularization), so the plan is to **port Lumenairy's `RCWAStack`** (Part C
   "RCWA backend" + `docs/lumenairy_rcwa_port_wishlist.md`), NOT grcwa. The Modulator's grcwa
   work is retained only as a cautionary lesson (use a Li-factorized solver for metals) and as a
   3rd-party numerical oracle.
4. **Already ported / present in DynaMeta (no action):** DielectricDB (JARVIS/MP DFPT static
   eps + override/provenance, `materials/db.py`), `fit_drude_params`, prismatic semiconductor
   boundary layers (`Mesh3DSpec.semi_prism_thk_m`), the DD abs-tol-vs-SI-units convergence
   lesson, and the equilibrium/DD division of labor.
5. **Lessons to carry, not code:** (a) the gated degenerate-DD convergence saga resolved to a
   single SI-units **abs-tol floor** (not carrier pinning) -- relevant when the AC/transient DD
   adds the time term; (b) DEVSIM 2-D lateral-edge contacts capture only box-corner nodes ->
   native **3D DEVSIM** (already in `devsim_3d`) is the route for full-edge carrier pinning.

---

## Part E -- Discipline + risks

- **House validation rule** kept throughout: every `EffectModel`/driver **reduces to the
  known limit** (vacuum/background at zero drive) AND is checked against an **independent
  oracle** (analytic Vpi*L, tmm, Freedericksz threshold, analytic sheet reflection, ...).
- **Each phase ships an independently usable, validated device class.** Phases 1-4 become
  "add an `EffectModel` (+ maybe a driver)" once Phase 0 lands -- not four silos.
- **Hard parts to budget for:** tensor-eps FEM conditioning; RCWA-with-anisotropy +
  Li-factorization; the QCSE exciton model; the LC director PDE; the graphene impedance BC;
  a real AMS/HYPRE binding for large 3D.
- **Highest-leverage start:** Phase 0 + Phase 1 (FieldBundle seam + tensor eps +
  electrostatics + Pockels) -- it converts DynaMeta into a general electro-optic modulator
  simulator and proves the spine on a real device. The lumped-RC bandwidth port (Part D.1) is
  a cheap parallel win that delivers a dynamic figure-of-merit immediately.
