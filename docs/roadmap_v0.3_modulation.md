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
- **RCWA backend** -- slot right after Phase 1 so it supports anisotropy from the start; fast
  forward + independent oracle + the optimization enabler. (PORT the integration + lessons.)
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
   the Park cell. **HIGH-VALUE near-term port:** DynaMeta already has the C(V) piece
   (`analysis.gate_cv`, added 2026-06-01); add `analysis.lumped_rc_bandwidth(cv, rho_sheet,
   geometry)` to get an intrinsic f_3dB **with zero new solver**. This is the cheapest real
   modulator figure-of-merit and the on-ramp to the full AC/transient phase.
2. **The "HYPRE"/iterative-solver work** (`stage3_optical/fem/{compare_solvers,
   fine_mesh_with_iterative,smoke_gmres}.py`). *Correction:* there is no literal HYPRE/AMS
   code -- the only preconditioner is NGSolve **BDDC** (`driver_3d.py:179`), which DynaMeta
   already ported (`linear_solver` in {umfpack, bddc_cg, bddc_gmres}). What IS portable:
   (a) a **direct-vs-iterative benchmark harness** (|r|^2 agreement + wall time + peak RSS) --
   port as a solver-regression validation; (b) the **scaling lesson**: UMFPACK is infeasible
   at fine meshes (100s of GB) -> BDDC-GMRes is the O(n)-memory path. **Strategic reframe:**
   the audit flagged AMS/HYPRE as an ENZ dead end (indefinite near-zero-eps), but the new
   EO / thermo-optic / dielectric mechanisms are **well-conditioned and physically larger**
   than the deep-subwavelength ENZ patch -- so a real NGSolve-AMS/HYPRE binding (the rung
   above BDDC) becomes worthwhile EXACTLY as we broaden the mechanism set. This is the right
   home for the "HYPRE" item, and it was not viable in the ENZ-only world.
3. **grcwa RCWA pipeline** (`stage3_optical/rcwa/`) + its hard-won convergence lessons:
   grcwa lacks **Li factorization**, so it is erratic in Fourier order (nG) for 2-D metallic
   patches (~+/-25 nm resonance noise) -- fine for dielectric/low-contrast, unreliable for
   metal gratings. **Port the integration pattern + the lesson** into the RCWA enabler: use a
   Li-factorized RCWA (e.g. inkstone) for metallic/high-contrast, grcwa for dielectric; and
   reuse the eps sign-convention bridge (Stage-2 exp(+iwt) -> grcwa exp(-iwt), conjugate at
   the boundary). Do NOT port grcwa wholesale (it was under-converged for the Park metal patch).
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
