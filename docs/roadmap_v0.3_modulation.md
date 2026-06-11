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

### Phase 0 -- Architectural keystones (shared by all four families; build ONCE)  **[DONE 2026-06-02]**
*Built: `core/effects.py` (`EffectModel` Protocol + `OpticalModelEffect`/`ComposedEffect`); the
bridge now passes a `fields` dict (`{n, E, T, ...}`) to the response (`MaterialEpsMap.eps_grid`);
`EpsField` carries a per-point 3x3 tensor; the assembler builds a domain-wise 3x3 matrix CF and
`solve_fem` uses the matvec `(eps.u).v` weak form + a tensor `A_independent`. Isotropic-reduction
gate `validation/tensor_isotropic_gate.py`: a diagonal `eps*I` reproduces the scalar R/T/A to
~1e-16. (NGSolve gotchas captured in the assembler/solver comments: domain dispatch must be the
OUTER CF list; matrix zero-entries must be literal int 0, not complex 0j.)*
- **0a FieldBundle seam.** Generalize `NToEpsMap` -> `EffectModel(fields: dict, lambda) -> eps`;
  the bridge assembles + passes the full local-field dict `{E = -grad(phi), n, T, ...}` on
  the aligned grid. Drude becomes one `EffectModel`. An `EffectModel` registry + composition
  (effects stack: e.g. background birefringence + Pockels, or thermo-optic + free-carrier).
- **0b Tensor/anisotropic eps end to end.** `EpsField` carries a per-point 3x3 eps;
  `VoxelCoefficient`/assembler + the FEM curl-curl weak form handle a tensor coefficient.
  *Gate:* reduces EXACTLY to today's scalar results when the tensor is isotropic.

### Phase 1 -- Field-effect EO (Pockels / Kerr / Franz-Keldysh)  [highest leverage]  **[DONE 2026-06-02]**
*Built: `core/effects.PockelsEffect` (full 6x3 r-tensor, eps(E)=(B0+dB)^-1), `KerrEffect`
(isotropic quadratic), `FranzKeldyshEffect` (simplified electro-absorption); `carriers/
electrostatics.py` (series-capacitor E-field driver). ORACLE `validation/
pockels_phase_modulator.py`: electrostatics -> Pockels tensor -> tensor FEM reproduces the
independent TMM scalar-n_y solve (|dR|=2e-3, dphi modulation matched to 2.5e-4 rad, 3.9 deg over
0-6 V) -- proves the generalized spine on a real non-carrier modulator. REMAINING: 1a is the
analytic series-cap field (exact for a layered parallel-plate); the full Laplace/Poisson FEM driver
for laterally non-uniform geometries is DONE (C1 92a2b62, `carriers/electrostatics_fem.py` +
`validation/electrostatics_fem.py`); the crystal-orientation rotation and a rigorous
Airy/Kramers-Kronig Franz-Keldysh remain follow-ons.*
- **1a Electrostatics driver:** a pure-dielectric Laplace/Poisson solve for E = -grad(phi)
  (no mobile carriers) -- cheap, a subset of the existing DEVSIM Poisson.
- **1b `PockelsModel`** (EO tensor r + crystal orientation), `KerrModel`, `FranzKeldyshModel`
  (field-shifted band edge -> d(alpha), Kramers-Kronig d(n)).
- *Oracle:* analytic Vpi*L for a thin-film LiNbO3 phase arm; eps -> isotropic at r=0.
  **Delivers a complete non-carrier modulator end to end -> proves the generalized spine.**

### Phase 2 -- Thermo-optic / electro-thermal  **[DONE 2026-06-02]**
*Built: `carriers/thermal.py` (steady series-thermal-resistance driver: `steady_layered_temperature`
mean-T per layer + `uniform_temperature_rise` lumped helper) and `core/effects.ThermoOpticModel`
(eps(T) = (sqrt(eps_ref) + dn_dT*(T - T_ref))^2, scalar/isotropic). ORACLE
`validation/thermo_optic_modulator.py`: a real-Si (n0=3.48, dn/dT=1.8e-4/K) thermo-optic phase
shifter -- the scalar-eps FEM reproduces an independent TMM scalar-n(T) solve (max|dR|=4e-3,
max|d(dphi)|=2e-5 rad over 0-150 K) with a few-degree phase modulation; gated on FEM-vs-TMM dphi
agreement + a Fabry-Perot-aware modulation-slope sanity band + model-level no-shift at T_ref.
Composition for stacking shifts on a region added via `core/effects.DeltaEffect` (subtract a
zero-drive baseline so a ComposedEffect does not double-count the background). Post-phase
adversarial audit (6 agents): all findings fixed -- uniform_temperature_rise shape/finite guard,
ASCII-only docstrings (removed 3 Greek glyphs), honest driver docstrings (the bridge did not yet
auto-assemble {E,T} -- that tracked seam has since SHIPPED, C7 1b082e4), and the validation
no-shift/slope gates made non-tautological.
REMAINING -- ALL DONE since: the volumetric-Joule heat-equation FEM (C2 8a25ca9) + transient
(R5 f3f7477, `validation/thermal_transient_fem.py`); electro-thermal coupling via the Picard loop
(R6 1be7ff1, `validation/electrothermal_picard.py`); the anisotropic dn/dT tensor variant (a12c435).*
- **2a Thermal driver:** steady + transient heat equation (FEM), Joule source from the
  electrical solve (electro-thermal). Emits T into the FieldBundle.
- **2b `ThermoOpticModel`** (dn/dT) + material thermal params (k, Cp, dn/dT).
- *Oracle:* Si-heater d(lambda)/d(power); no shift at dT=0. (Reuses the Phase-0 seam.)

### Phase 3 -- QCSE / MQW electro-absorption  **[DONE 2026-06-02]**
*Built: `carriers/qcse.py` (`QuantumWell` Stark driver: tilts a finite electron + heavy-hole well
by a perpendicular field and solves both ground subbands via the existing BenDaniel-Duke kernel
`SchrodingerPoisson1D.solve_schrodinger`, returning the redshifted interband edge E_T(F) + the e-h
overlap; in-well localization picks the ground state so a strong tilt does not return a field-
ionized edge state; per-carrier confinement energies are referenced to the well-centre band floor
so the linear tilt cancels and the pure quadratic Stark shift remains) and `core/effects.
ElectroAbsorptionModel` (excitonic Gaussian edge alpha(E_photon;F) ~ overlap(F)*line(E_T(F)),
d-alpha = alpha(F)-alpha(0), d-kappa from d-alpha, d-n from a Kramers-Kronig transform of d-alpha
via the Maclaurin alternate-point method `kramers_kronig_dn`; returns a complex scalar eps).
ORACLE `validation/qcse_electroabsorption.py` (pure numpy/scipy, no FEM): (1) deep-well limit
matches the analytic infinite-square-well E1 (0.985) AND the 2nd-order Stark coefficient
dE1 = -beta q^2 m F^2 L^4/hbar^2, beta = (128/pi^6)sum n^2/(n^2-1)^5 = 2.1944e-3 (C_num/C_ana=1.05,
quad-fit R2=0.9998); (2) a physical GaAs well shows a quadratic edge redshift (R2=0.9997) + a
monotonic e-h overlap drop + no shift at F=0; (3) the device turns ON absorption (d-alpha>0, Im(eps)
0.01->0.245) at a probe 2 sigma below the zero-field exciton, and reduces EXACTLY to eps_bg at F=0
(|eps0-eps_bg|=2e-15). REMAINING: the band-to-band continuum (Elliott) on top of the single exciton
line and the MQW stack/coupled-well variant are DONE (a12c435, `validation/qcse_elliott_mqw.py`,
re-run green 2026-06-11); a rigorous 2D exciton binding(F) remains the open follow-on.*
- Extend the **existing `SchrodingerPoisson1D`** with an in-well field -> Stark-shifted
  sub-bands + a simple exciton model -> `ElectroAbsorptionModel` (d(alpha), d(n) via KK).
- *Oracle:* known QCSE redshift vs field; flat-band reduction. (Biggest reuse of what exists.)

### Phase 4 -- Reconfigurable (PCM / LC / graphene)  **[DONE 2026-06-02]**
*Built all three families on the spine: `core/effects.PCMModel` (Bruggeman effective-medium mix of
amorphous/crystalline eps by crystalline fraction; exact end-state reduction, passive branch),
`core/effects.LiquidCrystalModel` (uniaxial tensor eps = n_o^2 I + (n_e^2-n_o^2) n-hat(x)n-hat from
a director tilt -> the Phase-0b tensor FEM) + `carriers/lc_director.py` (1-constant Frank-elastic
Freedericksz driver: exact threshold V_th = pi sqrt(K/(eps0 dEps)), supercritical theta_max(V) by
monotone elliptic-quadrature bisection), and `core/graphene.py` (Kubo surface conductivity
sigma(E_F,omega) intraband Drude + universal interband sigma0=e^2/4hbar + the analytic conductive-
sheet Fresnel r=(n1-n2-Z0 sigma)/(n1+n2+Z0 sigma)). ORACLES: `validation/reconfigurable_modulators.py`
(pure numpy, no FEM) -- PCM endpoint/monotonic/passive/Wiener-bounds; LC Freedericksz threshold +
rotation-invariant uniaxial eigenvalues {n_o^2,n_o^2,n_e^2} + isotropic reduction + n_eff(theta);
graphene universal-sigma0 + Pauli blocking (Re(sigma)->0 once 2E_F>hbar*omega) + gate-tunable
absorption + R+T+A=1 + sigma->0 Fresnel -- and `validation/lc_uniaxial_fem.py` (tensor FEM: the
PLANAR (theta=0) and HOMEOTROPIC (theta=90) principal director states == TMM at n_o/n_e). REMAINING:
(a) **off-diagonal-tensor FEM** -- RESOLVED (fa2b8dc): this LC oracle SURFACED what looked like a
pre-existing P0b gap, but the root cause was mesh.SetPML's coordinate stretch (exact only for
isotropic media), NOT the matrix-CF matvec and NOT an NGSolve defect -- the assembly itself was
proven exact to ~1e-16 (`docs/ngsolve_offdiag_check.py`, 328159c; see
`docs/ngsolve_offdiag_investigation.md`). Fix: an explicit anisotropic UPML
(Lambda=diag(s_z,s_z,1/s_z)) folded into the weak form for tensor eps; the NotImplementedError
guard is REMOVED. Validated: `validation/lc_tilted_fem.py` (theta=0..90) +
`validation/magneto_optic_faraday.py` GATE D (re-run green 2026-06-11). (b) the graphene FEM
surface-current boundary condition is DONE (442ae31, `validation/graphene_sheet_fem.py`); (c) the
LC/PCM thermal-pulse switching drivers are DONE (1064706, `validation/switching_drivers.py`) and
two-constant elasticity (K11!=K33) is DONE (c3f8844, `validation/lc_two_constant_bvp.py`).*
- `PCMModel`: crystalline-fraction state -> effective (n,k) (GST/Sb2S3/VO2); optional
  thermal-pulse switching driver.
- **LC director driver** (Frank-elastic + field) -> uniaxial tensor along n-hat (reuses
  Phase-0 tensor eps). *Oracle:* Freedericksz threshold.
- **Graphene/2D surface conductivity** sigma(E_F, omega) (Kubo) as a surface-current BC in
  the FEM (not volumetric eps). *Oracle:* analytic sheet reflection.

### Cross-cutting enablers (interleaved; shared by every mechanism)
- **Lumped-RC bandwidth** (near-term; PORT from Modulator, Part D) -- a real modulator FOM now.
- **RCWA backend** **[SUPERSEDED by the v0.5 BRIDGE -- Lumenairy is now a live required
  dependency (>=5.14.2); see roadmap_v0.5 A0-A4]** -- **PORT Lumenairy's native RCWA
  (`RCWAStack`, ~v5.6), do NOT rebuild.** It
  already provides multi-layer 1-D/2-D periodic, conical, uniform/`eps_cell`/full-tensor
  `eps_tensor_cell`/analytic-`shapes` layers, R/T/A + 0-order complex Jones, in the SAME
  `exp(-i omega t)`/`Im(eps)>0` convention (no sign bridge). DynaMeta adds a `LayeredStackSolver`
  Protocol + an `RcwaSolver` adapter + a z-slicer (graded/tensor `EpsField` -> `RCWAStack`
  layers). Slot right after Phase 1 (tensor eps) so it serves the EO mechanisms; it is the fast
  forward + independent oracle + the optimization enabler. Lumenairy gaps to close first (P1
  2-D/stack autodiff, P2 normal-vector FFF, P3 2-D patterned-tensor rigor) are itemized in
  `docs/lumenairy_rcwa_port_wishlist.md` (superseded: the v0.5 bridge declares the P-items NOT
  blockers). **RCWA-independent prep DONE (2026-06-01):** the
  `LayeredStackSolver` seam + a concrete `TmmLayeredSolver`, the `core.layered` slab model +
  z-slicer, and a pluggable `run_pipeline(optical_solver=...)` are built and validated vs the
  FEM -- so the port reduces to wiring `RCWAStack` to the seam + the structured-slab path.
- **Iterative/scaling solver** -- BDDC is in; a NGSolve-AMS/HYPRE binding is the next rung for
  large 3D (Part D). The non-ENZ mechanisms are well-conditioned, so this is now viable.
- **AC/transient carriers** **[DONE 2026-06-02]** -- the full dynamics axis shipped: fd5a51a (ssac)
  + ed7531b (transient); validations ac_capacitance / ac_diode / transient_diode /
  transient_metasurface.
- **Optimization / inverse-design** -- rides on a fast forward solver (RCWA). **Groundwork DONE
  (2026-06-02):** a scoped Lumenairy-style array-backend seam (`core/backend.py`: `array_namespace`
  + lazy numpy/cupy/jax dispatch, jax forced to float64) makes the pure-array CONSTITUTIVE maps
  (Pockels/Kerr/FK/thermo-optic/PCM/LC/graphene-sigma) backend-agnostic and JAX-DIFFERENTIABLE -- so
  once the differentiable RCWA optical backend lands, the gradient flows design -> fields -> eps
  through these maps for free (validation/backend_autodiff.py: jax.grad of the Pockels eps matches
  the analytic slope; numpy/jax agree to 0). The NumPy path is bit-identical float64 and never
  imports jax/cupy; the FEM/DEVSIM solvers (C++/host-numpy) are deliberately untouched.
- **FDTD** **[DONE]** -- shipped: 1-D engine C9 e30fa66; 2D/3D fdtd_nd c938b66; nonlinear R15
  28d9e81 (Kerr, all-optical); see `docs/fdtd_engine_roadmap.md`.

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
