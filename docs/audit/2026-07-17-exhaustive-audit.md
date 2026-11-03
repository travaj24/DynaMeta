# DynaMeta exhaustive audit -- 2026-07-17 (v0.7.0, post fiber_amp merge)

Full-codebase audit of DynaMeta at main commit 2f22099 (v0.7.0). Axes: feature gaps and seams
between library portions, physical correctness, code conventions, library organization, and
speed/memory improvements that do not sacrifice accuracy.

**Method.** Staged review, one subsystem cluster at a time, <=5 parallel reviewers per stage
(Opus), each hand-deriving the central formulas against the cited literature and checking
reduce-to-known-limit behaviour -- followed by an adversarial verification pass over every P0/P1
finding (and contentious P2s) before it is accepted into this document. Findings marked
[CONFIRMED] survived refutation; [REFUTED] items are retained in a graveyard section so they are
not re-reported by future audits. Severity: P0 = wrong physics/results in a mainline path;
P1 = wrong in an edge path or a seam mismatch that silently corrupts results; P2 = quality/
robustness/organization defect; P3 = nit/polish.

**Scope survey** (product code, `dynameta/`, ~29.3k LOC):

| Subsystem | LOC | Files | Stage |
|---|---|---|---|
| carriers/ (+thermal_fem) | 7,867 | 36 | 1 |
| optics/ flat (solver, TMM, FDTD-1D, seam, ngsolve, laser_gain, inverse design) | 3,488 | 12 | 2 |
| optics/fdtd_nd/ | 2,943 | 14 | 2 |
| optics/soa/ | 3,940 | 12 | 3 |
| optics/fiber_amp/ | 2,170 | 14 | 3 |
| optics/lumenairy_bridge/ | 2,494 | 11 | 4 |
| core/ (+effects) | 2,490 | 21 | 4 |
| materials/ | 1,024 | 7 | 4 |
| reliability/ | 1,195 | 12 | 4 |
| geometry/, drivers/, io/ | 1,081 | 12 | 4-5 |
| root (pipeline, sweep, analysis, results, cache, viz, transient_optics, constants) | 1,195 | 9 | 5 |
| tests/ (56 files, 8.6k LOC) + validation/ (208 scripts) | -- | -- | 5 |

Prior audits: docs/audit/2026-07-05-deep-audit.md (all findings resolved in v0.6.0). This audit
is fresh; it does not re-litigate resolved items but does check their fixes still hold where a
stage touches them. The fiber_amp subpackage (merged 2026-07-16, PR #2) receives first-time
adversarial review here.

Status legend: [OPEN] awaiting verification, [CONFIRMED], [REFUTED], [DOWNGRADED].

---

## Stage 1: carriers/ -- semiconductor transport, quantum, DEVSIM, LC/LLG, thermal FEM

Five reviewers (transport / quantum / devsim-fem / lc-magnetics / glue-misc), all 36 files read
in full, central formulas hand-derived. **18 findings: 0 P0, 0 P1, 3 P2, 15 P3.** The subsystem
is in strong shape -- the load-bearing discretizations (enhanced Fermi-Dirac Scharfetter-Gummel,
generalized Einstein, Trellakis SP Newton, Frank/Ericksen-Leslie, LLG, Kirchhoff transform,
two-temperature coupling) were all re-derived and confirmed correct.

### Findings

| ID | Sev | Cat | Where | Summary | Status |
|---|---|---|---|---|---|
| S1-1 | P2 | convention | carriers/density_gradient.py:118 | Bare `np.trapezoid` in the `conserve_charge=True` branch breaks on the declared numpy>=1.24 floor (trapezoid is numpy 2.0+); the repo guards this exact call elsewhere with `hasattr`. Untested branch + CI runs numpy 2.x, so the floor is never exercised. | [CONFIRMED] (np.trapezoid is 2.0+; branch uncovered) |
| S1-2 | P2 | test | carriers/physics_equilibrium.py:45 | Load-bearing Aymerich-Humet F_1/2 fit + `invert_F12` (Phi_c0 calibration -- a wrong root silently miscalibrates every downstream carrier solve) are unreachable without devsim (module-top `import devsim`) and have no monotonicity/accuracy test in the eta>=20 regime the module exists to fix (Halen-Pulfrey). Fit itself verified correct by hand. | [CONFIRMED] |
| S1-3 | P2 | test | drivers/__init__.py:39 | The solver-free base-import contract (`import dynameta.carriers/drivers` must not pull devsim/ngsolve) -- the whole point of the PEP-562 lazy machinery -- has no regression test, unlike the analogous jax/cupy and lumenairy contracts. Empirically verified clean today; demonstrably fragile (3 modules already have top-level `import devsim`). | [CONFIRMED] (empirical subprocess check during review) |
| S1-4 | P3 | convention | carriers/__init__.py:11 | `__all__` lists three devsim-requiring submodules, so `from dynameta.carriers import *` eagerly imports devsim -- contradicting the docstring's lazy/solver-free claim (empirically demonstrated). drivers/__init__ applies the correct discipline for the same situation. | [CONFIRMED] **[FIXED wave-7 (2026-11-02), documentation -- reconciled with 2026-07-25 V-13]** Re-verified at HEAD: still eager, and deliberately so -- `__all__` is the declared public surface (test_api_surface walks it) and names the headline DEVSIM builder; drivers/ can drop its lazy name from `__all__` because that name is a convenience re-export, carriers/ cannot without un-declaring the package. V-13 wrote the star caveat into the package docstring; this wave pins the documented behaviour from BOTH sides in a devsim-free child interpreter: `import dynameta.carriers` and a named pure-numpy submodule import work, `from dynameta.carriers import *` raises devsim's ImportError. Gate: `tests/test_audit_2026_07_17_wave7_carriers.py`::test_s1_4_star_import_is_eager_and_the_plain_import_is_not. |
| S1-5 | P3 | seam | carriers/physics_density_gradient.py:105 | DG in-Newton re-points region continuity to ElectronCurrentDG but leaves the ohmic CONTACT equation on classical ElectronCurrent, so extracted terminal current on a DG device is read from the pre-quantum flux on the contact edge (solve itself unaffected). | [CONFIRMED] **[FIXED wave-7 (2026-11-02)]** `setup_contact_dg` now re-binds the contact's ElectronContinuityEquation with `edge_current_model='ElectronCurrentDG'` (same node_model, so the boundary condition and every converged solution are unchanged) and refuses if the region has no DG current yet (the documented order). Measured on a 40 nm bar, 4e26 -> 4e25 m^-3 step 3 nm from the right contact, 10 mV, max Lambda 10 mV: terminal-current conservation |I_L + I_R|/|I_L| 2.95e-2 before (the right contact read the classical flux; 6.0e-3 with the step 6 nm away), 4.4e-16 after, both terminals = the interior DG current to 2e-16. New `eq_registry.recorded()` (read-only view) and an optional `scope=` on `forget()`. Gates: `tests/test_audit_2026_07_17_wave7_carriers.py`::test_s1_5_dg_terminal_current_is_the_density_gradient_flux (fails pre-fix at 2.95e-2), ::test_s1_5_contact_dg_before_the_quantum_correction_refuses. |
| S1-6 | P3 | physics | carriers/density_gradient.py:57 | Diagnostic `quantum_potential_V` uses nested `np.gradient` for u'' -- a 2*dz same-parity stencil that decouples even/odd nodes (sawtooth risk on coarse grids). Load-bearing BVP path unaffected. | [OPEN] |
| S1-7 | P3 | convention | carriers/physics_density_gradient.py:166 | Hard-wall path re-derives V_T locally from KB*T_REF/Q_E instead of importing constants.V_T. | [CONFIRMED] |
| S1-8 | P3 | test | carriers/thermal_fem/transient.py:57 | Time-dependent load paths (flux_of_t/joule_of_t, two-temp source_*_of_t) have zero test/validation coverage; they rely on a subtle ngsolve LinearForm `.vec` lifetime pattern. Reviewer ran a LIVE check (matches constant-flux path to 1.1e-13 K) -- correct today, unguarded against ngsolve upgrades. | [CONFIRMED] **[FIXED wave-7 (2026-11-02), coverage]** Four oracle gates, all passing at HEAD (the paths were correct): single-T `joule_of_t` / `flux_of_t` RAMPS vs the exact slab eigen-series through half a diffusion time (Crank-Nicolson 100 steps 2.3e-5 / 2.6e-5 of the rise, gate 1e-4; backward Euler 200 steps 2.1e-3 / 2.3e-3, gate 5e-3 -- a load applied one step late is ~1e-2); two-temp `source_e_of_t` + `source_l_of_t` in an insulated box vs the EXACT 2x2 theta-method recursion (4.4e-14 CN / 1.2e-14 BE of a 68 K rise, gate 1e-10 -- pins the sampling time of every step's load) and vs the continuous ODE (3.4e-4 CN); two-temp `flux_of_t` energy balance E^{n+1}-E^n = dt[(1-theta)F^n + theta F^{n+1}] (1.1e-10 / 1.9e-10, gate 1e-8, non-uniform field). Gates: `tests/test_audit_2026_07_17_wave7_carriers.py`::test_s1_8_*. |
| S1-9 | P3 | physics | carriers/devsim_3d.py:469 | 3D equilibrium multi-semiconductor heterostack develops no internal built-in potential (per-region Phi_c0 makes psi=0 exact equilibrium; interfaces enforce potential continuity only; no band offsets). Undocumented limitation; DD/bipolar variant already hard-raises. | [CONFIRMED] **[FIXED wave-7 (2026-11-02) for EQUILIBRIUM; DD/bipolar DEFERRED with design]** Root cause is wider than filed: each region's Phi_c0 made THAT region neutral at psi = 0, i.e. an implicit affinity chi = eta_bg kT -- equal work functions, so not even a same-material n+/n HOMOjunction had a built-in potential. New `Stacked3DSpec.band_alignment()` (Anderson rule: Phi_c(r) = Phi_c0(semi) + (chi_semi - chi_r)/q, psi_neutral(r) = (W_semi - W_r)/q); `build_device` installs it on every extra region (primary unchanged -> single-semiconductor solves byte-identical); `extra_semiconductors` entries take optional 6th/7th fields (chi_eV, dos_mass_kg), `chi_semi_eV` added, `from_design` fills both from each layer's TransportModel; a different material without both affinities raises. 4e26 | 4e25 m^-3 ITO: psi_bi = -0.448 V (FD; was 0). Gates: `tests/test_audit_2026_07_17_wave7_carriers.py`::test_s1_9_* + `validation/carriers_3d_multisemi.py` GATE C (3-D solve). DD/bipolar stays refused; thermionic-emission interface design + the 2-D LayeredDevsimBuilder sibling: `docs/audit/2026-11-02-s1-9-heterojunction-built-in-potential.md`. |
| S1-10 | P3 | organization | carriers/devsim_3d.py:62 | In-method `import devsim` lazy ceremony is defeated by module-top import of physics_equilibrium (which imports devsim unconditionally); missing devsim yields a raw ImportError instead of devsim_layered's friendly hint. | [CONFIRMED] **[FIXED wave-7 (2026-11-02)]** Re-verified: physics_equilibrium itself went lazy in wave 3 (S1-2), but devsim_3d still imported physics_drift_diffusion / physics_bipolar_dd / dc_solve (devsim at top), so a devsim-free install got `import of devsim halted` from a sibling. devsim is now imported ONCE at devsim_3d's top with devsim_layered's install hint; the four in-method `import devsim as ds` lines are gone. Gates: `tests/test_audit_2026_07_17_wave7_carriers.py`::test_s1_10_devsim_free_install_gets_the_friendly_hint (both builders), ::test_s1_10_no_in_method_devsim_imports_remain. |
| S1-11 | P3 | convention | carriers/thermal_fem/common.py:43 | ThermalLayerTwoTemp.k_e() defaults electron conductivity to the LATTICE k_thermal, so the documented G->inf single-T reduction carries k=2*k_thermal -- easy to misuse. | [CONFIRMED] **[FIXED wave-7 (2026-11-02), deprecation]** Measured: k_thermal = 1, 200 nm, 1e8 W/m^2, G = 1e20 -> two-temp mean rise 5.00 K vs single-T 10.0 K (exactly half: k = 2 k_thermal). There is no neutral default (metal: electrons carry most heat; dielectric: k_e ~ 0), so `k_electron=None` now emits DynaMetaDeprecationWarning at construction with the legacy value unchanged (results byte-identical; every in-repo caller already passed it); an explicit value must be finite and >= 0. Gate: `tests/test_audit_2026_07_17_wave7_carriers.py`::test_s1_11_* (warning, guards, and k_e + k_l = k reproduces single-T to 1e-5). |
| S1-12 | P3 | convention | carriers/llg.py:40 | Electron gyromagnetic ratio (CODATA) declared as a module literal instead of living in dynameta.constants. | [CONFIRMED] |
| S1-13 | P3 | physics | carriers/lc_director.py:504 | Freedericksz seed/wrong-branch estimate uses mean (K11+K33)/2 instead of the deformation-appropriate constant; for high splay/bend contrast the 1.3*V_th safety net can spuriously flag a correct solution just above the true threshold. Solve itself seed-independent. | [CONFIRMED] **[FIXED wave-7 (2026-11-02)]** New `_onset_threshold_V`: linear stability of the anchored state -- eps0 V_lc^2 (-dEps cos 2theta_b) = pi^2 K_eff(theta_b) + (1/2) f2''(theta_b) Phi^2 - 2 K22 q0 d Phi cos 2theta_b, divided by the undeformed cell's V_lc/V_app -- i.e. splay K11 (planar), bend K33 (VAN), the Leenhouts-Schadt TN threshold in the chiral solver, the series-layer voltage division, and +inf when the field stabilises theta_b. Both solvers seed + safety-net from it. Measured (K11 = 20, K33 = 2 pN, dEps 14.7): old estimate 0.742x of the true 1.232 V onset; at 1.20 V the correct untilted answer was flagged success=False; dEps < 0 planar got a finite 1.64 V threshold and was flagged at 5 V. Gates: `tests/test_audit_2026_07_17_wave7_carriers.py`::test_s1_13_* (closed forms; the helper brackets the BVP's own bifurcation to 0.99/1.01 V_th incl. series layers, and the TN/chiral one to 0.97/1.03; the two false alarms). |
| S1-14 | P3 | physics | carriers/lc_director.py:875 | Reported chiral `twist_energy_J_m2` omits the constant (1/2)K22 q0^2 complete-square baseline -- reporting-semantics offset (can be negative at natural twist); EL equations unaffected. | [CONFIRMED] **[FIXED wave-7 (2026-11-02), convention]** Decided: COMPLETE-SQUARE -- `twist_energy_J_m2` = int[(1/2) K22 (sin^2 th phi' - q0)^2 + (1/2) K33 sin^2 th cos^2 th phi'^2] dz >= 0, zero for an undistorted helix, (1/2) K22 (Phi/d - q0)^2 d for a planar twisted cell (legacy = this - (1/2) K22 q0^2 d; identical at q0 = 0). Documented on LCChiralResult; the existing pin (test_lc_director + validation/lc_chiral_twist GATE C) moved from -(1/2) K22 q0^2 d to 0. Gate: `tests/test_audit_2026_07_17_wave7_carriers.py`::test_s1_14_twist_energy_is_the_complete_square (4 twists incl. both sides of the natural pitch). |
| S1-15 | P3 | convention | carriers/carrier_heating.py:36 | T_REF re-declared locally (=300.0) instead of imported from constants; M_E imported unused. | [CONFIRMED] |
| S1-16 | P3 | convention | carriers/sp_carrier.py:80 | Stale inline comment claims nonparabolicity is a post-hoc parabolic fill, contradicting the code (fully Kane-nonparabolic Newton) and the corrected class docstring -- pre-C6-5 leftover. | [CONFIRMED] |
| S1-17 | P3 | organization | carriers/schrodinger_poisson.py:320 | Dead helper laplacian_matrix(): built and bound at line 327, never used (Newton builds its banded Jacobian directly). | [CONFIRMED] |
| S1-18 | P3 | test | carriers/schrodinger_poisson.py:45 | Sommerfeld large-x (>40) branches of _fd1/_fermi_log and the dF_1/dx=F_0 identity are docstring-claimed but not unit-tested (only the spence branch is exercised). | [CONFIRMED] **[FIXED wave-7 (2026-11-02), coverage + a defect the gate found]** vs 40-digit mpmath polylog over x in [-700, 1e4]: the Sommerfeld (x > 40) branches were exact (<= 9e-17), but the spence branch ran down to x = -700 and its argument 1 + e^x cancels -- 1.0e-3 relative error at x = -30, F_1 == 0 exactly below x = -36.7 (the non-degenerate sub-bands lost their 2 alpha kT Kane share). _fd1 now uses the alternating series for x < -1: <= 1.3e-15 everywhere. Gates: `tests/test_audit_2026_07_17_wave7_carriers.py`::test_s1_18_fermi_integrals_match_40_digit_polylog_on_every_branch, ::test_s1_18_dF1_dx_equals_F0_across_the_branch_edges (integrated identity by adaptive quadrature, intervals straddling x = -1 and 40). |

### Verified correct (do not re-litigate)

- Enhanced Fermi-Dirac SG flux (drift term g-independent, D = mu*g*V_T via kahan3/Bernoulli
  identity) reduces EXACTLY to J = q mu n E + q mu g V_T grad n; equilibrium limits n~exp(psi/(g V_T))
  (electron) and p~exp(-psi/(g V_T)) (hole) both sign-correct.
- Generalized-Einstein g = F_1/2/F_-1/2 fit accuracy claims verified (<=1.3% peak, guarded eta<=32);
  Jacobian bounded as n->0.
- van Overstraeten-de Man SI conversions, Chynoweth masking, I_sub integral; Caughey-Thomas/Canali
  saturation with smoothed |E|; Masetti template's deliberate mu_min2 choice.
- DG sign convention (Lambda<0 at hard wall) consistent between post-hoc BVP and in-Newton psi_eff
  forms; BVP nondimensionalization correct.
- Full SP stack: _fd1 = -Li2(-e^x) via spence identity exact; nonparabolic 2D sheet density
  (Kane in-plane DOS) closed form; BenDaniel-Duke tridiagonal index alignment; Trellakis Newton
  residual/Jacobian (parabolic + Kane branches are exact derivative pairs); Kane bulk E_F inversion.
- Kane Sommerfeld mean-energy factor K=(1+2aE_F)/(1+aE_F) re-derived from d<E>=(pi^2/6)(kT)^2 g(E_F)/n.
- QCSE: infinite-well Stark beta = closed form (1/24 pi^2)(15/pi^2-1); tilt-reference cancellation
  makes E_T reference-independent.
- Kirchhoff transform exactness (unit-k theta problem + pointwise inverse); layered T_at_z closed
  form re-derived character-for-character; two-temperature +G(ue-ul)(ve-vl) is the dissipative
  orientation (flipped sign would be anti-dissipative); MESH_SCALE=1e9 scaling cancels consistently
  across stiffness/mass/flux/source so dt stays SI.
- DEVSIM boundary is SI throughout (no cm/m slip); 2D depth_m seam [A/m]*m -> [A] correct; bipolar
  flat-band phi_bi seed matches the DEVSIM CELEC form.
- LC: elliptic-quadrature Freedericksz solution, two-constant BVP torque balance, flexoelectric
  P/torque mutual consistency (validated against an independent external solver), chiral tilt-twist
  EL system, Rapini-Papoular signs (static + dynamic), backflow effective-viscosity limits, 2-D
  Gauss-Seidel director update; LLG precession/damping signs, H_eff = -(1/mu0 Ms) dU/dm consistency,
  JMAK/CNT moment scheme (Avrami-3/4 reductions).
- Import hygiene: `import dynameta.carriers` / `.drivers` genuinely pull no devsim/ngsolve
  (empirically verified); state_glue/reliability_glue signatures and unit/shape seams all match
  their consumers; TMM slab-ordering seam consistent with tmm_reference.

### Coverage notes

All 36 in-scope files read in full; DEVSIM solves not re-executed (algebra verified by hand against
canonical DEVSIM forms + the cited validation oracles); ngsolve transient path live-checked.
Backflow Leslie-coefficient combination remains an explicitly-flagged upper-bound approximation
(tested for direction + off-limit only). Gated-accumulation SP solve and oxide series-cap map are
validation-script-covered rather than pytest-gated.

## Stage 2: optics core + fdtd_nd

Five reviewers (solver-ngsolve / tmm-lasergain / fdtd-1d / fdtd-nd-solvers / fdtd-nd-kernels),
all 26 files read in full. **18 findings: 0 P0, 0 P1, 5 P2, 13 P3.** All central optics physics
(FEM p-pol BCs and flux normalization, UPML tensor stretch, TMM, four-level gain, Yee/CPML/ADE
discretizations, MO Crank-Nicolson, adjoint gradients) re-derived and confirmed correct.

### Findings

| ID | Sev | Cat | Where | Summary | Status |
|---|---|---|---|---|---|
| S2-1 | P2 | test | optics/eps_assembler.py:75 | Graded (gridded) TENSOR eps is never exercised end-to-end through solve_fem -- all three tensor-FEM oracles use UNIFORM tensors, yet core/bridge emits gridded (Nz,Ny,Nx,3,3) tensors onto this exact path. A component-placement error in the graded UPML mass term would ship undetected. | [CONFIRMED] |
| S2-2 | P3 (was P2) | physics | optics/eps_assembler.py:80 | Graded-tensor branch does NOT snap sub-tolerance off-diagonals to int 0 (uniform branch does). VERIFIED: code asymmetry real, but NO observable effect -- R matches the snapped path to ~1e-17 through solve_fem's explicit component-sum assembly (the matvec pathology the snap guards is unreachable), and docs/ngsolve_offdiag_check.py shows even the original rationale is obsolete. Pure consistency/robustness gap. | [DOWNGRADED -> P3] (executed) |
| S2-3 | P2 | seam | dynameta/transient_optics.py:40 | enz_reflector_stack places eps_ito[0] adjacent to AIR with no documented ordering contract, while every other layered consumer treats depth arrays as substrate-first-and-reverse. A DEVSIM-extracted (ascending-z) ITO profile is silently vertically flipped: the ENZ accumulation layer lands on the wrong side, R(t) wrong while R+T+A still closes (the invisible-flip class). Graded path entirely untested. | [CONFIRMED] |
| S2-4 | P2 | test | optics/fdtd.py:79 | solve_fdtd_1d has ZERO pytest coverage (Drude ADE, Kerr factor-3, Mur ABC, two-run R/T extraction gated only by a manual validation script); sibling MO engine IS pytest-covered. | [CONFIRMED] |
| S2-5 | P2 | performance | optics/fdtd_nd/kernels3d_numba.py:159 | All numba kernels recompute the time-invariant Drude ADE coefficients aJ,bJ per cell per timestep in the innermost loop (LLVM cannot hoist across the outer time loop); numpy/jax kernels precompute once. ~2 extra divisions/cell/step ~= +25% division count in THE hot path; bit-identical fix. | [CONFIRMED] |
| S2-6 | P3 | seam | optics/fdtd_nd/solve3d.py:174 | 3D Raman branch missing the `rw*dt > 1` stability guard the 2D entry has (copy-paste drift): silent NaN blow-up instead of the 2D's clear ValueError. | [CONFIRMED] |
| S2-7 | P3 | seam | optics/fdtd_nd/solve2d.py:235 | Lorentz and gain ADE builders lack the analogous w0*dt stability guard entirely (2D and 3D): a far-above-band pole diverges silently. | [FIXED wave-4 (1154002), GATED wave-7 (2026-11-02)] The `w0*dt > 1` guard was already in solve2d/solve3d for BOTH ADEs (Lorentz + gain, 2D + 3D) but nothing tested it. Gate `tests/test_fdtd_chi2_raman.py::test_s2_7_lorentz_and_gain_ade_w0dt_guard_2d_and_3d`: the undamped recursion's roots stay on the unit circle to w0*dt = 2 and leave it beyond (the guard keeps a 2x margin); the UNGUARDED kernel really diverges at w0*dt = 2.5 (peak > 1e6) and stays bounded at 0.9; all four front-end raises fire. |
| S2-8 | P3 | convention | optics/fdtd_nd/solve3d.py:377 | solve_fdtd_3d_mo._ncell accesses cyclotron_wc_rad_s directly (fill loop uses getattr default): a plain Drude layer in an MO stack crashes during grid sizing on an otherwise-valid input. | [CONFIRMED] |
| S2-9 | P3 | memory | optics/fdtd_nd/solve3d.py:105 | ~10 full (nx,ny,nz) Lorentz/Raman/gain material grids allocated unconditionally even for plain dielectric solves: ~1.3-1.6 GB of pure zeros at 256^3 held live through the solve (~+34% footprint). | [CONFIRMED] |
| S2-10 | P3 | convention | optics/fdtd_nd/backends.py:13 | numba (+numba.cuda) imported eagerly at package-import time (jax/cupy correctly lazy): numpy-only users pay the multi-second LLVM import. | [CONFIRMED] |
| S2-11 | P3 | performance | optics/fdtd_nd/kernels2d.py:98 | 2D numpy/cupy reference kernel allocates full-grid temporaries every step; the 3D kernel was explicitly rewritten to preallocate+out= (audit 6.2) but 2D never got the treatment -- and run_2d_te IS the cupy path (per-step device mallocs). | [FIXED wave-7 (2026-11-02)] run_3d's out= template: every per-step expression is an out= chain into preallocated scratch (same operands, same order) + Ey/Jy ping-pong. BIT-IDENTICAL on 10 paths (passive, Drude, Kerr, Drude+Kerr, Lorentz, Drude+Lorentz+gain, chi2+Raman, dynamic gain, hot carrier fixed-bath / finite-lattice; sha256 pins from the 1b1f394 kernel, `tests/test_fdtd_backend_contracts.py::test_s2_11_*`). numpy, best of 5: 64x400 1.0x (small temporaries are recycled), 256x1600 2.8x linear / 3.4x Drude, 1024x2000 2.1x / 2.3x; traced peak +10 % (scratch held for the run). |
| S2-12 | P3 | gap | optics/fdtd_nd/kernels3d_jax.py:10 | 3D jax kernel omits lor/chi2/raman/gain (2D jax twin carries all): 3D gradient-based inverse design of dispersive/active media is impossible though the forward physics exists. Correctly guarded (raises), so a capability gap not a silent bug. | [FIXED wave-7 (2026-11-02)] Mechanical port of run_3d's per-component Lorentz / gain / diagonal-chi2 and isotropic-Q Raman terms into run_3d_jax (carry extended only when active); the dispatcher no longer raises. Parity vs numpy run_3d 2.2e-15 .. 3.2e-15 relative on all 8 probe planes, each term moving them 5e-3 .. 5e-2; all-off probes bitwise the pre-port kernel's; jax.grad through the Lorentz strength == central FD to 1e-5; solve_fdtd_3d(backend='jax') with a Lorentz layer == numpy to < 1e-9. Gates `test_fdtd_backend_contracts.py::test_s2_12_*`. (numba 3D still carries Lorentz only.) |
| S2-13 | P3 | seam | optics/fdtd_seam.py:41 | _VAC_TOL=1e-3 loss guard on end media is 6 orders looser than the kernel's own 1e-9 guard; a weakly-absorbing substrate (k=1e-3) is silently truncated to lossless, biasing T ~2% high. Stale comment describes a different purpose. | [CONFIRMED] |
| S2-14 | P3 | convention | optics/fdtd_mo.py:241 | fdtd_mo registers materials at Yee E-nodes (arange*dz) while EVERY sibling engine uses cell centers (+0.5): a systematic dz/2 registration offset between the MO 1-D reference and the rest of the family (absorbed today by validation tolerances). | [NO-CHANGE wave-7 (2026-11-02): fdtd_mo is the RIGHT registration; documented + gated] Every engine shares E_k at k*dz (probe_cells rounds z/dz to an E index; the siblings de-embed r0 in that node frame) and eps_k fills E_k's control volume, centred ON the node, so node sampling is unbiased and (k+1/2) sampling shifts the structure -dz/2 on average (r-phase bias -k0*dz). Measured over 9 pad offsets (res 30): node mean dphi_r +0.005 rad, (k+1/2) -0.100 rad, k0*dz 0.090; |R|, |T| cannot tell them apart. Gate `tests/test_fdtd_mo_oblique.py::test_s2_14_*`. FOLLOW-UP (not changed here; it moves every sibling result at the dz/2 level): solve_fdtd_1d's uniform path, 2D, 3D and 3D-MO carry the -k0*dz r0-phase bias that validation/fdtd_seam_vs_tmm.py GATE 3's 15 deg absorbs (measured on solve_fdtd_2d, same slab / 9 pad offsets: mean dphi_r -0.095 rad vs k0*dz 0.090). [ENGINE SIBLINGS FIXED wave-7 (2026-11-02, fdtd_phase)] ONE registration helper (fdtd_nd/grid.py material_z / layer_masks / pad_masks) now feeds solve_fdtd_1d (uniform + time-varying mask), solve_fdtd_2d(_oblique), solve_fdtd_3d(_oblique/_mo), fdtd_mo (bit-identical) and the fdtd_seam structured painter. Two siblings of the defect fixed with it: the nonuniform refine grid (interfaces snapped ON nodes: -0.88 k0*h_fine -> control-volume mean at the tie nodes) and p-pol oblique 2-D (Ez at (k+1/2)*dz now gets its own half-node maps in the numpy/numba/jax TM kernels; node map on Ez too left -0.29/-1.56 k0dz at 30/50 deg). Mean r-phase bias over 8 midpoint sub-cell offsets, res 30: before -0.94..-0.95 k0dz (1-D/2-D/3-D/3-D-MO), -1.32 (n_super 1.5), -0.82 (s 30 deg), -1.15/-2.18 (p 30/50 deg); after +0.04..+0.06 (glass +0.18), shrinking ~4x per dz halving (O((k dz)^2)). Gates `tests/test_fdtd_registration.py` (all bias gates fail on d387eef); validation/fdtd_seam_vs_tmm.py GATE 3 15 deg -> 3a 1.5 k0dz per geometry + 3b offset-mean < 0.25 k0dz (before -0.99, after +0.013 k0dz). OPEN (reported, not re-pinned): point sampling re-rounds layer thicknesses by up to one cell when the sampling positions move dz/2 -- seam GATE 1 dielectric R 0.0307 -> 0.0710 (realized middle layer 10 -> 9 cells; TMM of the realized stacks 0.0326 / 0.0735), validation/fdtd_drude_lorentz_vs_tmm GATE 2 max|dR0| -> 2.82e-2 (33.19-cell slab: 33 -> 34 cells; realized-slab TMM 2.81e-2), validation/fdtd_graded_enz GATE 1 R 0.1072 -> 0.1496 vs TMM 0.1259 (1.35-cell slices: 13 -> 14 cells) and test_fdtd_hot_carrier lat_gate8 (1.87-cell film: 1 -> 2 cells); remedy = control-volume averaging, a scope decision. [Z-SUBCELL AVERAGING DEFAULT wave-7 (2026-11-02, user decision)] every engine above now fill-averages each E node's z control volume through ONE implementation (fdtd_nd/grid.py z_fill_fractions + mix_profile): arithmetic for the tangential E, HARMONIC for the p-pol Ez map (Farjadpour et al. OL 31, 2972 (2006); Oskooi et al. OL 34, 2778 (2009); Hwang-Cangellaris MWCL 11, 158 (2001)), Drude/Lorentz/gain/Raman strengths and chi2/chi3 fill-weighted with shapes kept, two-pole cells as one strength-weighted pole (error O((dgamma/w)^2)), Drude-mixed Ez cells point-sampled, hot-carrier masks = hot fill with p_abs per hot volume, time-varying hooks per-cell weighted; z_subcell='point' = 2b7a002 bit-for-bit. True-geometry max|dR| over 8 offsets, res 30 -> 60: point 7.9e-3 -> 2.3e-3, averaged 1.9e-3 -> 3.9e-4 (1-D; 2-D/3-D/MO/oblique alike), per-offset phase spread 0.073 -> 0.0049 rad. Of the four: seam GATE 1 dR 2.9e-3 PASS, graded_enz 2.0e-4 PASS, drude_lorentz GATE 2 2.82e-2 -> 5.9e-4 PASS, hot_carrier lat_gate8 7.4e-3 STILL FAILS -- diagnosed as the gate's mean-field oracle on a two-cell film (exact per-cell oracle 7.8e-4), reported, not forced. MOVED AWAY at the fixture resolution (converging O((k dz)^2) where point does not, or commensurate grids where point is exact in |R|), reported, NOT re-pinned: fdtd_sweep_vs_tmm GATE 1 (2.9e-3 -> 6.7e-3 at res 28; 4.0e-3 -> 1.6e-3 at res 56), fdtd_3d_reduces GATE A (commensurate slab, faces on E nodes: 2.5e-3 -> 6.3e-3 at res 18, both 4x at res 36), ringdown gate7 Q (-1.16% -> -2.12% vs Fabry-Perot at res 30; -0.37% -> -0.25% at res 60); ringdown gate9 now warns on a noise-floor pole (trace decays to 2e-15, no growth). [2026-11-03] Averaging kept on every interface cell incl. node-on-face (bounds the worst offset, continuous in offset; larger 2nd-order constant on aligned grids). lat_gate8 -> exact per-cell oracle via opt-in hot_out['record_cells'] (7.8e-4; fill-ignored control 6.3e-2) + a mean-field gate on an aligned one-cell film (7.2e-4). ringdown gate7 re-pinned 13.207623 -> 13.078561 + convergence companion (2.121% -> 0.253% vs Fabry-Perot, order 3.07). sweep_vs_tmm GATE 1 / 3d_reduces GATE A -> convergence gates with the registration-sweep constants C_2D 0.0802 / C_3D 0.0795: orders 2.05 / 2.01 PASS but part (a) FAILS (2.68 / 1.10: geometry-dependent constants) -- reported, not fitted. gate9 diagnosed: a zero-frequency drift pole (gamma -3e7 1/s), NOT the noise floor (window ends at 7.7e-7 of peak), latent under point sampling too -- reported, unchanged. [2026-11-03, second round] Per-geometry constants from each gate's own 8-midpoint-offset registration sweep (fdtd_nd.grid.subcell_offset_pads / second_order_constant, `--regen-cgeom`): C_geom 0.0854 (3d_reduces A) / 0.2157 (sweep_vs_tmm 1); (a) 1.023 / 0.996, (b) orders 2.01 / 2.05; under point (b) is -0.40 on the sweep (FAILS), 2.03 on 3d_reduces. 3d_reduces (a) fails by 2.3%: its fixture sits at sub-cell offset u = 0 (face on a node, the worst offset), which the midpoint set never samples -- STOPPED AND REPORTED. Ringdown: poles with |omega| < 2 pi/(N dt) are non-resonant Drift (fixed fit columns, no growing-pole warning, outside (f0, Q), reported); gates drift_i/ii/iii; gate9 warning-free at start_frac 0.10-0.40 under both rules (before: point 4/5, averaged 2/5 warned); default etalon Q 13.0785606 -> 13.0785451. Mur walls' unabsorbed static component noted as the drift source (follow-up). [2026-11-03, third round] Offsets for C_geom now u = k/8 incl. u = 0 and 1/2, anchored bitwise at each fixture's pad (fdtd_nd.grid.subcell_offset_pads; the registration-test midpoint tables are separate and unchanged): C_geom 0.08735397979210943 (3d_reduces A) / 0.21488435744194653 (sweep_vs_tmm 1); (a) 1.000 / 1.000 (attained at u = 0) PASS, (b) 2.01 / 2.05 PASS, point (b) on the sweep -0.40 FAILS. Lateral inclusion edges not smoothed (follow-up). Gates tests/test_fdtd_registration.py::test_true_geom_* (fail on the point rule). |
| S2-15 | P3 | performance | optics/laser_gain.py:167 | FourLevelSystem.evolve calls expm(A*dt) per step; uniform grids (the common case) need one expm + mat-vecs (~100-500x). | [CONFIRMED] |
| S2-16 | P3 | convention | dynameta/transient_optics.py:69 | Returned 'T' from the default lossy-mirror reflector stack is mirror absorption (~3%), not device transmittance (0) -- misleading unless documented. | [CONFIRMED] |
| S2-17 | P3 | performance | optics/fdtd.py:61 | solve_fdtd_1d recomputes loop-invariant coefficients and allocates fresh nz arrays every step even with Kerr off. | [CONFIRMED] |
| S2-18 | P3 | organization | optics/fdtd.py:20 | Unused `Optional` imports in fdtd.py and fdtd_mo.py. | [CONFIRMED] |

### Verified correct (do not re-litigate)

- solve_fem p-pol interface BCs and T = |t|^2 Re((eps_sub/eps_sup)(kz_s/kz_sub)) hand-derived
  exactly; s-pol Fresnel background phase referencing; tensor UPML Lambda-stretch weak form;
  absorbed-fraction normalization incl. the C3-7 Re(n_super) factor; graphene sheet BC S-scaling;
  probe-grid alias bound; Bloch phase vs Identify direction; Poynting R/T diagnostic signs.
- VoxelCoefficient axis order EMPIRICALLY pinned on NGSolve 6.2.2604 ([iz,iy,ix]) and consistent
  with core/bridge's (Nx,Ny,Nz)->(Nz,Ny,Nx) transpose -- the gridded-eps seam is correct end-to-end.
- laser_gain: g0 = kappa dN/(n c eps0 dw) derived from the ADE susceptibility with the correct
  Im(chi)<0 gain sign; relaxation-oscillation closed form re-derived from class-B linearization;
  exact four-level steady state (not the weak-pump approximation); cavity/threshold worked point.
- tmm_reference: per-layer absorption strip (verified against installed tmm source), passive-sqrt
  branch guard, polarization-'x' safety via the OpticalSpec guard.
- FDTD-1D: semi-implicit Drude ADE algebra collapses exactly to -(J^n+J^{n+1})/2; Kerr factor 3 is
  the correct dD/dt; Yee sign pair is a self-consistent gauge; magnetized-Drude gyrotropic ADE is
  internally consistent with its analytic circular-eigenmode oracle; Mur ABC standard.
- fdtd_nd: CPML b/c coefficients match Roden-Gedney with correct E/H staggering; outside-PML
  alpha!=0 is harmless (c=0 exactly); 1/n conductivity matched-scaling physically justified; 3D
  Yee curl stencil verified for all six components (the 'forward-looking' dzb is a correct backward
  diff after index placement); T0 impedance factors correct on both specular and flux paths;
  r0/t0 phase de-embedding + rfft conjugation to exp(-iwt); homogeneous-reference caching sound;
  numba PEC-boundary Jy asymmetry proven harmless; backend parity numpy/jax/numba operand-order
  preserving; adjoint gradients correct by construction (jnp.abs^2 objectives need no manual
  conjugation) with filter/projection chain validated.
- Gain ADE G3's 'missing' EPS0 vs Lorentz C3 is CORRECT (kappa dN = eps0 wp^2 by definition).

### Coverage notes

FEM R/T machinery heavily validated (s/p vs TMM to 45 deg, conical, dense media, graphene,
tensor-UPML); fdtd_nd has strong cross-backend and reduce-to-TMM validation including nonlinear
and autodiff gates. Notable residual gaps beyond the findings: r0/t0 PHASE only validated at
normal incidence; Faraday rotation sign pinned only by self-consistency (oracle aligns handedness
to the FDTD), magnitude + wc=0 reduction are pinned; much of solve2d/solve3d correctness lives in
validation/ scripts rather than pytest gates.

## Stage 3: amplifiers -- optics/soa + optics/fiber_amp

Five reviewers (soa-qd-gain / soa-propagation / soa-noise-metrics / fiber-cw / fiber-dynamic).
fiber_amp received first-time external adversarial review with an explicit attack brief.
**42 findings: 0 P0, 2 P1, 8 P2, 32 P3.** Both P1s are in the newly-merged fiber_amp and were
empirically confirmed by the reviewers (executed reproductions, not just derivations).

### P1 findings

| ID | Sev | Cat | Where | Summary | Status |
|---|---|---|---|---|---|
| S3-1 | P1 | seam | optics/fiber_amp/metrics.py:26 | Metric cloning helper `_with()` (and gain_spectrum's bare rebuild) silently DROPS the ConcentrationModel: every metric that clones (gain_compression_curve, saturation_output_power, slope_efficiency, gain_spectrum) runs the IDEAL model -- dark-pair absorption and photodarkening vanish, upconversion mis-scales (n_t vs active density). Empirically confirmed: `_with(amp).concentration is None`, `_n_dark==0` while the source amp has 1e24. solve() itself unaffected. | [CONFIRMED] (executed) |
| S3-2 | P1 | physics | optics/fiber_amp/detection.py:92 | Spont-spont beat variance is EXACTLY 2x too large: the leading '2' is already the full single-pol coefficient and m_modes multiplies it again. Discriminating limit: B_e=B_o unpolarized thermal light must give var/mean^2 = 1/2 -> sigma^2 = 2(R rho B_o)^2; module returns 4(R rho B_o)^2 (ratio 2.0 verified numerically). The module's own sig-sp term correctly uses one polarization -- the asymmetry is the tell. | [CONFIRMED] (executed) |

### P2 findings

| ID | Sev | Cat | Where | Summary | Status |
|---|---|---|---|---|---|
| S3-3 | P2 | physics | optics/soa/sbe.py:72 | Reduced-SBE susceptibility carries an extra dipole factor (chi ~ mu^3, dimensionally C*m; absolute gain ~28 orders too small; correct prefactor 1/(eps0 d_qw)). VERIFIED by independent derivation + execution: dividing out the spurious mu gives max chi = 0.094 (physical). Held at P2: shape/sign/contrast/KK (the module's claimed content) are unaffected by the constant rescale; sbe_gain_per_m absolute output is nonsense. Fix module + oracle in lockstep. | [CONFIRMED] (executed) |
| S3-4 | P2 | seam | optics/soa/traveling_wave.py:224 | A time-varying drive whose sample count happens to equal n_slices is silently reclassified as a static spatial injection profile -- temporal modulation completely lost, no error. | [CONFIRMED] |
| S3-5 | P2 | physics | optics/soa/metrics.py:175 | ripple_enob_ceiling converts gain-ripple dB with the field /20 where the intensity-encoded photocurrent channel requires /10. VERIFIED: facet_gain_ripple_dB provably returns power-ratio dB; measured offset exactly +1.00 bit (small ripple) to +1.27 bit (3 dB); the docstring's own examples are the inflated values. One-line fix. | [CONFIRMED] (executed) |
| S3-6 | P2 | seam | optics/soa/ase_noise.py:105 | m_pol semantics inconsistent between siblings in the same file: ase_output_psd returns TOTAL (m_pol-scaled) PSD while detector_noise_variances/spectral_noise_figure consume PER-POL and apply m_pol themselves -- natural composition gives 2x-8x errors. | [CONFIRMED] |
| S3-7 | P2 | test | optics/fiber_amp/steady_state.py:171 | Counter-pumped and bidirectionally-pumped solves -- the raison d'etre of the relaxation solver -- have zero coverage (every test/validation pump is 'fwd'). Reviewer ran all three configs: correct today (22.66/22.66/22.68 dB), unguarded against regression in the backward-IVP direction handling. | [CONFIRMED] (executed) |
| S3-8 | P2 | organization | optics/fiber_amp/rare_earth.py:115 | dP_dz/ase_source_per_m are dead exports (solver reimplements inline) AND the two copies have diverged: rare_earth uses n_t where the solver uses active density, omitting dark-pair/photodarkening -- with a false docstring claiming the solver calls them. External code building on dP_dz gets answers inconsistent with solve(). | [CONFIRMED] |
| S3-9 | **P1** (upgraded from P2) | physics | optics/fiber_amp/waveguide.py:97 | cladding_pump_overlap uses (core/clad)^2 where the flux/gain construction requires the pump-DOPANT overlap (b_dope/clad)^2. VERIFIED by derivation (flux = gamma P/(h nu A_dope) is consistent only if gamma is the power fraction inside b) AND execution: confined-doping clad-pumped fiber (core 5um, b 2.5um) shows pump absorption inflated by exactly (core/b)^2 = 4.000x. Upgraded to P1: silently wrong inversion/gain for a standard fiber design through the public path. | [CONFIRMED, UPGRADED P1] (executed) |
| S3-10 | P2 | physics | optics/fiber_amp/detection.py:97 | Beat-noise NF scales with detector quantum efficiency (spurious eta in snr_in): eta=0.7 reports a sub-quantum-limit NF. Standard definition references an IDEAL input detector. Gates only used eta=1, where it is correct. | [CONFIRMED] |
| S3-11 | P2 | convention | optics/fiber_amp/pulse.py:193 | GNLSE dispersion operator is written in numpy's exp(+i w t) reconstruction convention: odd-order dispersion (beta3, TOD; also cpa.apply_spectral_phase) carries the OPPOSITE sign to the repo's exp(-i w t) convention. Even orders immune (why soliton/CPA gates pass); a literature (Agrawal-convention) beta3 gives the mirrored oscillatory tail. Internal CPA self-consistent; external inputs trapped. | [CONFIRMED] (numerically verified ifft convention) |

### P3 findings (condensed)

| ID | Where | Summary |
|---|---|---|
| S3-12 | validation/qd_soa_sbe.py:45 | SBE oracle replicates the mu^3 prefactor -- no absolute-scale gate anywhere (enabler of S3-3). |
| S3-13 | soa/qd_gain.py:41 | Docstring says 'modal gain' for what is MATERIAL gain (consumer applies Gamma); invites double/missing-Gamma. |
| S3-14 | soa/qd_gain.py:82 | fastmath=True vs docstring 'bit-parity' claims (tests correctly assert ~1e-12, not bit). |
| S3-15 | soa/qd_gain.py:566 | ES optical comb reuses the intraband dE_ES_GS as the interband offset; in eh-split mode electron/hole ladders conflated. |
| S3-16 | soa/qd_gain.py:1272 | Non-Markovian two-Lorentzian line is peak-normalized: wing broadening manufactures integrated gain (EID path is correctly area-conserving). |
| S3-17 | soa/qd_gain.py:31 | No confined-state Auger channel (WL Auger only) -- known dominant QD non-radiative loss at high injection. |
| S3-18 | validation/qd_soa_eh_split.py:92 | eh-split dark relaxation with asymmetric hole time has no per-band Boltzmann-ratio gate. |
| S3-19 | soa/qd_gain.py:772 | power_to_photon_density duplicates photon_density with divergent array handling. **[FIXED wave-7 (2026-11-02)]** ONE implementation: power_to_photon_density is now an alias of photon_density (scalar -> float, array -> ndarray; same operation order -> values bit-identical to both old copies; an array power used to raise TypeError). Gate: tests/test_audit_2026_07_17_amps_wave7.py (fails at 1b1f394). |
| S3-20 | soa/qd_gain.py:351 | with_detailed_balance_taus constrains ES<->GS only; WL<->ES ratio thermodynamically unconstrained. |
| S3-21 | soa/traveling_wave.py:492 | eta_in (input-coupling NF penalty) silently ignored on the segmented-GVD path of amplify_coherent. **[FIXED wave-7 (2026-11-02)]** the sqrt(eta_in) facet loss (and its validation) now precede the distributed-GVD branch; both paths report the fiber-referred A_in/P_in. Measured at 1b1f394: A_out(eta 0.5)/A_out(eta 1) = 1.000 on the segmented path (passive and 1500/m gain) vs sqrt(0.5) = 0.7071 now (to 1e-6, linear regime). Rider: the segmented path also silently dropped langevin/seed and state0 -- now NotImplementedError. Gate: tests/test_audit_2026_07_17_amps_wave7.py. |
| S3-22 | soa/thermal.py:57 | Dense (n,n) matrices + dense solve for a tridiagonal fin system (docstring claims tridiagonal solve). **[FIXED wave-7 (2026-11-02)]** three diagonals + LAPACK banded LU (dgbtrf once, dgbtrs per step; dgttrf's scipy wrapper rejects n = 2). 'insulated' vs the old dense solve: 3.5e-15..7.6e-15 (kappaA = 0 and n = 2, 3 bit-identical). 'sunk': the Dirichlet couplings into rows 1/n-2 are eliminated (they multiply theta = 0), so the solve is exact to 1.0e-15 (n 60) / 9.9e-15 (n 300) / 6.1e-14 (n 1000) vs a Fraction (exact rational) oracle, where the OLD dense solve was off by 1.0e-12 / 3.1e-10 / 1.1e-9 (pivoting swapped the |c| ~ 1e7 row above the unit row) -- old-vs-new 'sunk' differs only at that old error level. Time: steady n 2000 79 ms -> 0.22 ms, transient n 2000 x 400 steps 455 -> 35 ms; n = 2e5 now solves in 18 ms (old: 320 GB dense matrix). Gate: tests/test_audit_2026_07_17_amps_wave7.py. |
| S3-23 | soa/traveling_wave.py:307 | saturation_curve settle window sized by transit count with no convergence guard; every validation overrides the default. **[FIXED wave-7 (2026-11-02)]** convergence check rel = |P_out(end) - P_out(mid-window)|/P_out(end) <= rtol (default 1e-3; bounds the remaining error for a monotone exponential approach); unconverged runs are CONTINUED exactly (new private amplify(_P_node0=...) + 'P_node' carry -- chained runs bit-identical to one long run) doubling the window up to max_settle_transits (default 32 x settle_transits), then UnreliableResultWarning; result carries settle_transits_used / settle_rel_change. Measured at 1b1f394: qd_soa_saturation_power device, 100 mW point (tau ~30 transits) left 2.5e-3 (0.011 dB) P_out error at the default 60 transits; two-level gate device (tau_c = 85 transits) 0.32 % / 16.7 % error -> 2.2e-5 / 4.5e-11 now. Points converged at the first check are bit-identical to the old single-window value (all validations, settle_transits=400: unchanged). Gate: tests/test_audit_2026_07_17_amps_wave7.py. |
| S3-24 | soa/calibration.py:194 | report['bandwidth_nm'] aliases MATERIAL FWHM while the identically-named datasheet target is the NET -3 dB width: spurious match hides the documented 3.6x narrowing. |
| S3-25 | soa/ase_noise.py:179 | Dead conditional branch (executes `pass`). |
| S3-26 | soa/ase_noise.py:173 | Docstring claims m_pol applied to *_mean accumulators; code keeps them per-pol. **[FIXED wave-7 (2026-11-02), doc]** the CODE is right: the only consumer, ase_self_consistent, multiplies S_f_mean + S_b_mean by m_pol itself, so scaling the means would double-count polarization. Docstring corrected (m_pol on *_out only; every other PSD per-pol); gate pins S_f/S_b/*_mean m_pol-independent, S_f_out = m_pol x S_f, and the corrected sentence. |
| S3-27 | soa/__init__.py:45 | __all__ omits four imported public ASE/noise functions. |
| S3-28 | soa/ase_noise.py:116 | noise_figure ignores internal loss when alpha_i>0 but Gamma_g is None. **[FIXED wave-7 (2026-11-02)]** alpha_i_per_m > 0 without Gamma_g_per_m now raises ValueError (as does alpha_i < 0) instead of returning the lossless NF (G 100, n_sp 1.5, alpha_i 60/m: 4.74 dB returned vs 5.71 dB at Gamma g 300/m). No in-repo caller hit the silent path. Gate: raise + analytic NF with loss (1e-14) + the PSD route ase_output_psd -> nf_from_psd on a uniform lossy amplifier (1e-10). |
| S3-29 | soa/metrics.py:126 | Distortion path is single-tone harmonics only; no two-tone IMD3 (the spur that actually sets analog-link SFDR). |
| S3-30 | soa/ase_noise.py:101 | Sub-transparency slices contribute ZERO spontaneous emission (n_sp=inf clamped to q=0) though the true source ~rho^2 is finite. |
| S3-31 | fiber_amp/steady_state.py:234 | AseBand.m_modes not stored in result.meta; noise/detection independently default m=2 -- non-default m_modes silently corrupts PSD/n_sp/NF/OSNR. |
| S3-32 | fiber_amp/thermal.py:49 | Untracked out-of-band spontaneous emission counted fully as heat; docstring claims the opposite. |
| S3-33 | fiber_amp/noise.py:50 | local_inversion_factor omits the ESA term from its denominator: under opt-in ESA the reported n_sp is inconsistent with the net gain used. |
| S3-34 | fiber_amp/steady_state.py:220 | Convergence tests endpoint powers only with a 1e-15 W denominator floor; no interior-profile guard. |
| S3-35 | fiber_amp/noise.py:179 | Forward ASE spectrum recomputed 3-4x per noise/detection call. |
| S3-36 | fiber_amp/steady_state.py:29 | _P_FLOOR_W defined with explanatory comment, never used (leftover from the abandoned solve_bvp design). |
| S3-37 | fiber_amp/metrics.py:33 | _set_total_pump collapses all pumps to zero unrecoverably when current total is 0. |
| S3-38 | fiber_amp/dynamics.py:168 | Upconversion is an explicit-Euler bolt-on after the exponential integrator: 'unconditionally stable' claim fails and converged inversion is dt-biased. |
| S3-39 | fiber_amp/pulse.py:158 | SaturableGain 'parabolic' shape 1-x^2 -> -inf in the wings: unbounded parasitic LOSS beyond the band instead of gain->0. |
| S3-40 | fiber_amp/detection.py:76 | detection_noise recomputes the forward ASE spectrum ~3x per call. |
| S3-41 | tests/test_fiber_amp.py:489 | No absolute sp-sp variance pin and no eta<1 test (enablers of S3-2/S3-10). |
| S3-42 | tests/test_fiber_amp.py:298 | Transient dynamics validated only ASE-free/single-signal/co-pumped. |

### Verified correct (do not re-litigate)

- QD gain core: conjugate-flux particle conservation, detailed-balance escape times, group-resolved
  inhomogeneous convolution, transparency at rho=1/2, saturation-density rise with pump (12 items).
- Traveling-wave marcher: dP/dz = (Gamma g - alpha_i)P forms, bidirectional coupling, Agrawal-Olsson
  reduction; BPM paraxial operator; dome thermal profile (15 items).
- SOA ASE: n_sp form, z-resolved bidirectional accumulation, GS/ES two-band bookkeeping, RIN/linewidth
  formulas, NF=(2 n_sp(G-1)+1)/G consistent with fiber_amp's convention at the formula level (20 items).
- fiber_amp CW: Giles model, McCumber + at_temperature composition, ESA threading, relaxation solver
  forward/backward sweeps (reviewer independently re-ran counter- and bidirectional-pumped configs --
  correct), Frantz-Nodvik E_sat = h nu A/(Gamma(sigma_a+sigma_e)) confirmed the correct convention for
  the quasi-3-level scheme, Brown-Hoffman, Giles calibration round-trip (12 items).
- fiber_amp dynamics/pulse: integrating-factor sweeps, exponential integrator core, split-step
  ordering (symmetric Strang), B-integral accumulation, energy bookkeeping of the saturable gain in
  half-steps (correctly exp(gh) total per step), beat-NF reduction at eta=1, CPA compressor matching
  (14 items).

### Coverage notes

soa: strong validation lattice (35 qd_soa_* scripts) but several quantitative claims are gated only
by shape/relative comparisons (the mu^3 lesson: no absolute-scale gate). fiber_amp: 57 pytest gates
give strong forward coverage but are author-written; the found P1s lived exactly in the paths the
gates did not pin (concentration-through-clones, absolute beat-noise scale, eta<1, odd-order
dispersion sign, counter-pumping).

## Stage 4: lumenairy_bridge + core(+effects) + materials + reliability + geometry/drivers/io

Five reviewers (bridge-rcwa / bridge-others / core / effects-materials / reliability-geometry-io).
**29 findings: 0 P0, 0 P1, 7 P2, 22 P3.**

### P2 findings

| ID | Sev | Cat | Where | Summary | Status |
|---|---|---|---|---|---|
| S4-1 | P2 | physics | lumenairy_bridge/_common.py:198 | conical_synthesis propagating-order mask valid only for LOSSLESS end media. VERIFIED empirically (lumenairy 5.24.2): even Im(n_sub)=0.001 collapses T from 0.942 to 0.000 (specular dropped by BOTH mask clauses -- Im(kz)>>tol AND lumenairy's kz.real<0 branch choice); same stack at phi=0 vs phi=45 disagrees by the entire transmitted power. Reachable through the public make_lumenairy_rcwa_solver path; no guard anywhere. | [CONFIRMED] (executed) |
| S4-2 | ~~P2~~ REFUTED -> P3 hygiene | physics | lumenairy_bridge/pmm_backend.py:195 | Code structure read correctly (absorption block not gated on phi; physics reasoning sound) BUT the claimed failure is UNREACHABLE: absorption=True forces retain_internal=True, and lumenairy's PMMStack.solve raises NotImplementedError at conical BEFORE the block (empirically demonstrated). No public path reaches the inconsistency. Residual P3: the bridge relies on lumenairy's internal guard rather than defensively gating like the RCWA bridge -- one-line hardening recommended. | [REFUTED as P2; retained as P3 hardening] (executed) |
| S4-3 | P2 | seam | core/bridge.py:151 | assemble_eps affine-remaps ONLY z onto the alignment bbox; the lateral axes are emitted raw (xlo/xhi/ylo/yhi thrown away, no origin/period check). Safe today only by coincidence of convention (OCC cell at origin); a BYO carrier or centered mesh frame silently shifts the eps laterally. The exact asymmetry that makes z robust leaves x/y fragile. | [CONFIRMED] |
| S4-4 | P2 | test | core/resample.py:16 | resample_to_grid -- the unstructured-node->grid gridder every carrier solve depends on -- has NO fast/CI test (importers are devsim-gated). The NaN nearest-fill ravel-order contract was hand-verified correct, but a regression would silently scramble carrier density at grid edges. | [CONFIRMED] |
| S4-5 | P2 | seam | materials/scattering.py:184 | ScatteringModel feeds MatthiessenGamma's OPTICAL-scaled rate into the DC drift mobility: with optical_dc_ratio != 1 the optical Drude gamma is right but mu_DC is wrong by that factor, flowing straight into the DEVSIM transport solve. The abstraction whose stated purpose is removing the DC-vs-optical inconsistency reintroduces one. | [CONFIRMED] |
| S4-6 | P2 | physics | materials/db.py:170 | JARVIS backend reports dfpt_piezo_MAX_dielectric (max principal tensor component) as the isotropic eps_static, with the ELECTRONIC part built from an AVERAGE -- mixed bases, unlabeled. For the very materials targeted (tetragonal/monoclinic HfO2) max exceeds the orientation average by tens of percent: gate capacitance systematically overestimated. | [CONFIRMED] |
| S4-7 | P2 | seam | io/store.py:110 | HDF5<->Zarr parity break: a 0-d (scalar) numpy array round-trips on HDF5 but raises IndexError on zarr>=3 on BOTH save and load (empirically reproduced). Same public API call succeeds or crashes depending on file extension. Latent for internal callers (all >=1-d). | [CONFIRMED] (executed) |

### P3 findings (condensed)

| ID | Where | Summary |
|---|---|---|
| S4-8 | tests/test_lumenairy_bridge.py:791 | No conical coverage with lossy end media (enabler of S4-1). |
| S4-9 | lumenairy_bridge/rcwa_backend.py:3 | Stale '>= 5.21' floor claims vs enforced VERSION_FLOOR=(5,22,0) (also rcwa_design.py). |
| S4-10 | lumenairy_bridge/translate.py:12 | Stale docstring describes the pre-5.22 private-slot reader; code reads the public layers property. |
| S4-11 [FIXED wave-7 (2026-11-02)] | lumenairy_bridge/rcwa_backend.py:7 | Docstring claims 'no sign translation' but the module performs the validated p-pol lab-basis -> p-hat sign/scale reconciliation. **Fixed wave 7 (2026-11-02):** the rcwa_backend module docstring now says the PHYSICS conventions cross unchanged and names the three translations it does perform (degrees -> radians, label -> lab Jones row, p-pol r -> -r and t -> t cos(theta_i)/cos(theta_t) via _common.p_basis_conversion). Docstring only. |
| S4-12 [FIXED wave-7 (2026-11-02)] | lumenairy_bridge/rcwa_backend.py:127 | Laterally-uniform anisotropic tensor forced through full 2-D RCWA (O((2Nx+1)(2Ny+1)) modes for a planar layer); Berreman is the intended path. **Fixed wave 7 (2026-11-02):** design_to_rcwa_stack clamps n_orders to 1 per axis when the stack is 2-D only because of a uniform tensor and every layer is laterally uniform (orders decouple exactly). Measured (lumenairy 5.50, ITO / 800 nm biaxial lossy LC / ITO, n_orders 11, six x/y/p normal / 20 deg / conical cases): 17.1-24.8 s -> 0.004-0.011 s per solve (~3500x); R/T/r/t vs the old 529-order solve <= 8.4e-13, A_independent <= 5.7e-13, vs Berreman 4x4 <= 1.4e-14 (the clamped solve is the closer one). Kept on RCWA rather than rerouted to Berreman so the per-order / conical / absorption contract is unchanged; a laterally STRUCTURED stack keeps the requested orders. Gate: tests/test_lumenairy_bridge.py::test_rcwa_uniform_tensor_stack_clamps_orders (old code: n_orders_x stays 11). |
| S4-13 [FIXED wave-7 (2026-11-02)] | lumenairy_bridge/pmm_backend.py:16 | Stale scope contract ('PMMStack has no conical source', 't is None') contradicted by the implemented synthesis. **Fixed wave 7 (2026-11-02):** pmm_backend module docstring rewritten: in-plane AND conical incidence (conical_synthesis, lossless end media), t = PMMStack.jones_transmission, per-layer absorption at phi = 0 only, pins re-stated on the 5.22 floor (layer_grids >= 5.32). The module's inline 5.32 layer_grids gate now reads the shared _common.FEATURE_FLOORS table. |
| S4-14 | lumenairy_bridge/bor_backend.py:206 | time.time() for solve timing where every sibling uses perf_counter(). |
| S4-15 [FIXED wave-7 (2026-11-02)] | lumenairy_bridge/bor_backend.py:154 | BorResult.fundamental_result() leaves R_flux/T_flux None; every Cartesian backend sets them -- cross-backend schema inconsistency. **Fixed wave 7 (2026-11-02):** R_flux=R, T_flux=T (the BOR R/T are the order-summed flux fractions, as on RCWA / PMM / PMM2D / Berreman). Gate: tests/test_lumenairy_bridge.py::test_bor_phase_and_absorption (old code: None != R). |
| S4-16 [FIXED wave-7 (2026-11-02)] | lumenairy_bridge/emt_screen.py:69 | order=2 Rytov tensor ORIENTATION (perp vs par components) never validated against a hand oracle; a transposed convention would ship. **Fixed wave 7 (2026-11-02) -- verified CORRECT, oracle added:** hand oracle from Lalanne & Lemercier-Lalanne, J. Mod. Opt. 43, 2063 (1996): eps_par(2) = eps_par(0) + c (e1 - e2)^2, eps_perp(2) = eps_perp(0) + c (1/e1 - 1/e2)^2 eps_perp(0)^3 eps_par(0). Measured (P 300 nm, 1.31 um, f 0.4 of 4+0.2i in air): xx - perp(2) 1.7e-18, yy - par(2) 0, off-diagonals 0; the transposed tensor misses by 0.82 and the order-0 tensor by 0.036 / 0.090. zz = par(2) (lumenairy's choice; not part of the in-plane claim). No code change. Gate: tests/test_lumenairy_bridge.py::test_emt_rytov_order2_orientation_vs_hand_oracle. |
| S4-17 [FIXED wave-7 (2026-11-02)] | lumenairy_bridge/pmm_backend.py:181 | end_media_indices computed twice per solve (also PMM2D, RCWA). **Fixed wave 7 (2026-11-02):** design_to_{rcwa,pmm,pmm2d}_stack attach the pair they computed to the stack (stack.dynameta_end_media, the pmm2d dynameta_origin_shift_m precedent) and the three solver closures read it: one evaluation per solve, the SAME complex values (bit-identical by construction; the bridge test files pass unchanged). Saves two material-model evaluations per solve (microseconds on ConstantOptical; a dispersive / tabulated model pays its lookup once). |
| S4-18 [FIXED wave-7 (2026-11-02)] | core/bridge.py:108 | 2D-lift path hardcodes lateral axis to 'x'; stack_axis=='x' would alias lateral==vertical, unguarded. **Fixed wave 7 (2026-11-02):** the 2-D branch of assemble_eps refuses stack_axis='x' by name (the 2-D grid layout is (Nx, Nv) with lateral 'x') and names a missing vertical axis (was a bare KeyError). Measured on the old code: a 5 x 5 grid with stack_axis='x' RETURNED a (5, 2, 5) gridded EpsField silently (lateral ramp re-used as the z profile); 6 x 4 died on an unrelated eps_grid shape error. Gate: tests/test_bridge.py::test_s4_18_stack_axis_x_aliasing_lateral_is_refused. |
| S4-19 [FIXED wave-7 (2026-11-02)] | core/lift.py:44 | IdentityLift.apply dead (bridge 3D branch bypasses lift) with a non-physical placeholder y-axis. **Fixed wave 7 (2026-11-02):** IdentityLift documented as the 'no lift' marker; apply() keeps its one live job (the R-30 error for a 2-D field) and REFUSES a 3-D array instead of returning the fabricated linspace(0, 1, Ny)-metre y axis. Gate: tests/test_bridge.py::test_s4_19_identity_lift_refuses_to_fabricate_a_y_axis. |
| S4-20 [FIXED wave-7 (2026-11-02)] | core/carrier_field.py:98 | dump writes axis_order attr that load never reads -- dead metadata implying an unhonored guarantee. **Fixed wave 7 (2026-11-02):** load_carrier_field honours axis_order (restores the dumped grid_axes_m order; measured on zarr 3.1.6 the old load returned ['x', 'y'] for a field dumped as ['y', 'x']) and refuses a file whose stored axes disagree with it; files without the attr load as before. Gate: tests/test_carrier_field_io.py::test_s4_20_axis_order_is_honoured_on_load. |
| S4-21 [FIXED wave-7 (2026-11-02)] | core/layered.py:146 | slice_eps_field structured branches do not normalize descending z (slice_profile does): raises instead of handling. **Fixed wave 7 (2026-11-02):** slice_eps_field normalizes a descending z axis (and refuses a non-monotonic one) before branching, so the structured-scalar and tensor branches return the same ascending-z slabs as for the ascending field, bit-identical (np.array_equal), native and n_slices. Gate: tests/test_layered.py::test_s4_21_slice_eps_field_normalizes_descending_z_on_every_branch (old code raises on a negative slab thickness). |
| S4-22 [FIXED wave-7 (2026-11-02)] | core/graphene.py:61 | Interband Kubo terms have no independent numerical oracle (FEM validation consumes the same graphene_sigma). **Fixed wave 7 (2026-11-02) -- oracle added; one bug fixed, one approximation documented:** direct quadrature of the Falkovsky-Varlamov finite-T Kubo integral (G built from Fermi functions, not the module's sinh/cosh). (1) Re: exact, <= 3.6e-15 sigma0 (1.55 um, E_F 0..0.8 eV, 10-300 K). (2) BUG the oracle exposed: the bare sinh/cosh overflowed to NaN below ~6.5 K at 1.55 um (measured T_K = 3: nan + RuntimeWarning); now an exponent-scaled form, <= 4.2e-16 relative to the old values elsewhere (E_F 0..1 eV x 30..600 K x 0.8..10 um). (3) Im is the Falkovsky (2008) thermally SMOOTHED log, exact only as T -> 0 (the old docstring called it 'the matching KK log'): |dIm|/sigma0 <= 0.10 at 300 K (2 E_F within ~0.1 eV of hbar omega), <= 0.075 at 77 K, 0.040 at the resonance point for T <= 30 K, (8 ln2/pi)(kT/hbar omega) ~ 0.058 at E_F = 0; documented in place and pinned. DEFERRED: an exact finite-T Im (quadrature or the Maldague thermal average of the T = 0 log) that stays JAX-traceable in E_F. Gate: tests/test_reconfigurable.py::test_s4_22_interband_kubo_vs_direct_quadrature (old code fails the T = 3 K leg). |
| S4-23 [FIXED wave-7 (2026-11-02)] | core/bridge.py:123 | assemble_eps materializes y-replicated copies of 2D extra fields where a broadcast view suffices. **Fixed wave 7 (2026-11-02):** the 2-D extra is passed as a read-only stride-0 broadcast view (an in-place write would raise, never corrupt). Measured (SeparableXY Ny 256, 256 x 64 grid, ThermoOptic 'T' via EffectEpsMap, 5 runs): tracemalloc peak 201.3 -> 167.8 MB (the 33.5 MB copy), median 0.100 -> 0.071 s, eps grid bit-identical (np.array_equal). Gate: tests/test_bridge.py::test_s4_23_extras_are_broadcast_views. |
| S4-24 [FIXED wave-7 (2026-11-02)] | core/effects/magneto.py:98 | Orphaned section-divider comments from the effects monolith split. **Fixed wave 7 (2026-11-02):** the dividers left at the END of magneto.py ('Intersubband ...') and thermo.py ('QCSE / MQW ...') now head the classes they name in electroabsorption.py (whose stray 'reconfigurable: phase-change + liquid-crystal' divider above IntersubbandEffect is replaced); base.py's 'field-effect electro-optic mechanisms' divider renamed to what follows it (the shared helpers). Comments only. |
| S4-25 | reliability/em.py:49 | Three-way self-contradiction on Black n=2 nucleation-vs-growth labeling (relabel applied only to module docstring). |
| S4-26 | reliability/hci.py:52 | Same incomplete-relabel: 'Takeda' in error message/oracle vs 'Hu lucky-electron' in docstring. |
| S4-27 [FIXED wave-7 (2026-11-02)] | geometry/design.py:85 | device_symmetry()/detect_symmetry_reduction() ignore Stack.features: an off-center via would not demote symmetry. **Fixed wave 7 (2026-11-02):** features demote device_symmetry exactly like inclusions (off-centre -> 'none', else the shape's intrinsic symmetry), and detect_symmetry_reduction applies the inclusion scope rules to them (semiconductor material or a non-rectangle/circle shape -> 'none'). On the old code an off-centre via left 'c4v' and the FEM hint offered 'quarter'. Features stay refused by the solvers (geometry/guards.py); this keeps the lift choice and the FEM hint honest for designs that carry them. Gate: tests/test_geometry_guards.py::test_s4_27_device_symmetry_and_fem_hint_see_stack_features. |
| S4-28 | io/store.py:79 | Metadata json round-trip: tuples silently become lists; numpy scalars crash. |
| S4-29 | reliability/em.py:116 | Miner docstring claims Blech-immortal intervals contribute zero damage; Blech immortality never evaluated in the accumulation. |

### Verified correct (do not re-litigate)

- RCWA bridge: p-pol lab-basis -> p-hat reconciliation (r_xx == -r_p pinned), per-pol energy
  convention handling, version-floor enforcement mechanics, geometry translation, solver caching
  (10 items). Berreman single-solve reuse, BOR complex phase, PMM2D absorption closure (12 items).
- core: CarrierField/EpsField axis contracts incl. the (Nx,Ny,Nz)->(Nz,Ny,Nx) transpose (pinned
  empirically against NGSolve VoxelCoefficient in Stage 2), lift semantics, n_to_eps conventions,
  backend dispatch cleanliness (11 items). resample_to_grid NaN-fill ravel alignment hand-verified.
- effects/materials: Elliott/Voigt/BGR electroabsorption seam (Im(eps)>0), Pockels/Kerr tensor
  conventions, gyrotropy sign, PCM/LC mixing rules, Drude/Lorentz models (14 items).
- reliability: Black EM activation signs, Norris-Landzberg, TDDB E-model, BTI power laws, LIDT
  runaway, eV/kT placements all correct (18 items). Geometry spec validation sound; io h5py path sound.

## Stage 5: seams, conventions, organization, root modules, test/validation infrastructure

Five reviewers (root-pipeline / cross-seams / conventions-sweep / test-infra / organization).
**26 findings: 0 P0, 1 P1, 9 P2, 16 P3** (net of duplicates with earlier stages, which are
cross-referenced rather than re-counted).

### P1

| ID | Sev | Cat | Where | Summary | Status |
|---|---|---|---|---|---|
| S5-1 | P1 | organization/physics | fiber_amp/detection.py:92 vs soa/ase_noise.py:135 | ROOT CAUSE of S3-2, independently rediscovered via duplication analysis: detector beat-noise is implemented twice; the audit-C4-3 fix ('Monte-Carlo confirmed 2x', removed the double-counted polarization factor) was applied to the soa copy only. fiber_amp still has the pre-fix form. Reproduced on identical inputs: sp-sp ratio exactly 2.0, sig-sp terms identical. Two independent reviewers, two methods, same defect => S3-2 is definitively CONFIRMED. | [CONFIRMED] (executed, x2 independent) |

### P2 findings

| ID | Sev | Cat | Where | Summary | Status |
|---|---|---|---|---|---|
| S5-2 | P2 | seam | dynameta/cache.py:177 | Optical-cache solver_id is module+qualname only: two solvers from the SAME factory with different answer-changing kwargs (verified: make_layered_tmm_solver n_slices=10 vs 100 -> identical id) collide under the default tag='' and the cache silently serves the wrong R/T. | [CONFIRMED] (executed) |
| S5-3 | P2 | seam | dynameta/cache.py:272 | Cache HIT reconstructs OpticalResult without per_region_absorption (None) while a MISS returns the populated map: hit/miss-dependent result shape at the pipeline seam; the reliability/LIDT axis consumes exactly this field. | [CONFIRMED] |
| S5-4 | P2 | seam | dynameta/cache.py:232 | flush() is a whole-store truncate-rewrite with no lock/atomic-rename/merge: concurrent writers on one cache path (the natural distributed mesh-sweep pattern) clobber each other's entries; a crash mid-truncate leaves an unreadable store that the bare except silently discards. | [CONFIRMED] |
| S5-5 | P2 | gap | dynameta/pipeline.py:206 | run_pipeline has no partial-failure handling: one solver exception discards every completed SweepRow (carrier solves are uncached and rerun). The NaN-aware downstream containers were hardened for a case the orchestrator can never produce. | [CONFIRMED] |
| S5-6 | P2 | seam | optics/solver.py:845 | per_region_absorption keyed DIFFERENTLY by FEM (mesh subdomain labels: L_skin/L_bulk/L__incl<j>...) vs every layered/Fourier backend (plain design-layer names). The C5-4 comment asserts a convention FEM only matches for plain layers; no normalizer exists. Backend-agnostic absorption queries break across the seam. | [CONFIRMED] |
| S5-7 | P2 | seam | core/n_to_eps.py:64 | EffectEpsMap-registered effect models are SILENTLY DROPPED for any region routed through fixed_eps_regions -- and the default LayeredOpticalBuilder makes only semiconductor layers spatial. An LC/PCM/MO/EO/thermo effect on a non-semi layer is discarded with no warning, defeating the modulation path run_pipeline advertises. Every test/example hand-builds a stub builder that declares the region spatial; the default-builder path is untested. | [CONFIRMED] |
| S5-8 | P2 | performance | tests/test_soa.py:635 | test_soa.py dominates suite runtime: did not finish a 500s cap numba-free (CI-representative) while test_fiber_amp (57 tests) takes 54s -- ~35+ CPU-minutes across the 4 CI legs, from pure-numpy BPM/many-body/MB/SBE integrations at production resolutions. | [CONFIRMED] (measured) |
| S5-9 | P2 | test | tests/test_carrier_field_io.py:9 | CarrierField dump/load (zarr-only, SCHEMA_VERSION=2, the Stage1->bridge serialization contract) is tested ONLY by a file that importorskips zarr -- and CI never installs zarr on any leg. Schema drift would pass CI green. | [CONFIRMED] |
| S5-10 | P2 | test | validation/qd_soa_numba_parity.py:35 | Green-washes the CI smoke tier: with numba absent it returns True (exit 0 = PASS) instead of the rc=42 SKIP convention that audit C6-6 introduced; every sibling capability-gated validation follows the convention. Its entire purpose is unverifiable on CI yet it reports PASS. | [CONFIRMED] (executed) |

### P3 findings (condensed; cross-refs deduplicated)

| ID | Where | Summary |
|---|---|---|
| S5-11 | dynameta/pipeline.py:164 | Verbose _emit crashes (TypeError) when T is not None but A is None ('A={:+.4f}'.format(None)). |
| S5-12 | dynameta/cache.py:212 | Cache wrapper does not proxy solve_sweep: wrapping a sweep-aware solver silently downgrades to per-wavelength solving. |
| S5-13 | (repo) | No .gitattributes EOL policy; committed blobs are uniformly LF -- the stated CRLF convention describes the working tree only -- and one file has mixed EOLs. |
| S5-14 | fiber_amp/spectroscopy.py:81 | 21 implicit-Optional annotations (concrete type + =None) vs the 230x Optional[...] convention. **[FIXED wave-7 (2026-11-02)]** the 21 spanned fiber_amp; 16 were still present at 1b1f394 (calibration 1, detection 3, pulse 6, rare_earth 1, spectroscopy 3, transverse_eryb 2) -> all Optional[...]; AST gate over every fiber_amp module (0 left; dynamics.py/eryb.py already clean). Gate: tests/test_audit_2026_07_17_amps_wave7.py. |
| S5-15 | carriers/devsim_layered.py:55 | _effective_dos_m3 duplicated verbatim in two carriers modules. |
| S5-16 | carriers/devsim_3d.py:358 | warnings.warn drift: 11/50 omit stacklevel, 29/50 omit category. **[FIXED wave-7 (2026-11-02)]** Re-verified: the 2026-07-25 V-17 wave converted every LITERAL `warnings.warn` site (census-gated), but three carriers sites called it through `import warnings as _w; _w.warn(...)` -- no category, no stacklevel, invisible to the census (devsim_3d x1, devsim_layered x2); the gated-bipolar one also sat inside `try: ... except Exception: pass`, which SWALLOWED the warning whenever it was escalated to an error. Now AdvisoryWarning (contact-current extras unavailable) / UnreliableResultWarning (gated bipolar under-accumulates), stacklevel=2, warned outside the try. Gate: `tests/test_audit_2026_07_17_wave7_carriers.py`::test_s5_16_no_aliased_warn_call_escapes_the_v17_census (alias-aware census over dynameta/). |
| S5-17 | pyproject.toml:34 | ruff+mypy are dev deps with no committed config and no CI lint gate: behavior unpinned. |
| S5-18 | .gitignore:13 | .coverage binary artifact tracked in git. |
| S5-19 [FIXED wave-7 (2026-11-02)] | Makefile:4 | `make test` help text understates scope (runs the full suite when solver extras are installed). **Fixed wave 7 (2026-11-02):** the help text says it runs the whole tests/ tree (@slow included; solver / jax / numba / zarr modules where installed), and make test ARGS='-m "not slow"' gives the CI test-leg selection (ARGS forwarded; default empty = the unchanged command). |
| S5-20 | optics/fdtd_seam.py:475 | The two FDTD run_pipeline entry points disagree on capability: per-wavelength supports structured cells on lossless non-vacuum end media; the sweep-aware one (which run_pipeline prefers) raises. **[FIXED wave-7 (2026-11-02)]** Aligned on the per-wavelength envelope: fdtd_sweep_spectrum now paints the end-media pads for a structured cell (the stale 'the rasterizer drops the pads' refusal is gone), AND FDTDSweepOpticalSolver.solve_sweep hands a structured cell THIS bias's eps at the band centre -- it used to rasterize the NOMINAL materials, silently dropping the bias modulation on the path run_pipeline prefers. Gate `tests/test_fdtd_seam.py::test_s5_20_*` (fails on the old code); validation/fdtd_structured_nonvacuum.py GATE C: the sweep path on glass vs TMM |dR| 1.35e-2 (= the per-wavelength GATE A), vs per-wavelength 1.3e-6. |
| S5-21 | optics/tmm_reference.py:202 | TmmLayeredSolver populates per_region_absorption but leaves A_independent=None, breaking the contract that the map sums to A_independent (every other producer sets both). **[FIXED wave-7 (2026-11-02)]** A_independent := sum of the per-slab map -- tmm's in-layer flux differences, a measure independent of the 1 - R - T closure A (|A - A_independent| ~1e-16 on lossy stacks). Gate `tests/test_per_region_absorption.py::test_tmm_solver_populates_map` (fails on the old code). |
| S5-22 [FIXED wave-7 (2026-11-02)] | dynameta/analysis.py:1 | Module docstring describes ~40% of the file; the majority is electrical modulator FOM code. **Fixed wave 7 (2026-11-02):** module docstring rewritten to index both halves (gate_cv, sheet_resistance_ohm_sq, lumped_rc_bandwidth, switching_energy_per_area, modulator_figure_of_merit; resonance_dip / resonance_shift, fano_fit, lorentzian_fit, quasi_bic_scaling). Docstring only. |
| S5-23 | README.md:149 | README version string stale ('v0.5.0') vs pyproject/__init__ 0.7.0. |

Cross-refs: T_REF re-declaration = S1-15; gamma_e = S1-12; resample_to_grid coverage = S4-4.

### Duplication catalogue (organization reviewer, drift-risk verdicts)

| Pair | Verdict |
|---|---|
| Beat-noise: soa/ase_noise vs fiber_amp/detection | **DRIFTED** (P1 S5-1: C4-3 fix applied to one of two copies) |
| Noise figure: soa vs fiber_amp/noise | Consistent today; latent drift pair -- consider a shared reference or cross-pinning test |
| Drude eps x3: materials/optical_model, fdtd ADE, rcwa_design JAX | Justified twins (different evaluation contexts) |
| Thermal x4: carriers planar, soa fin, fiber_amp radial, thermal_fem FEM | Justified (distinct geometries/regimes) |
| Saturation: fiber_amp Frantz-Nodvik vs soa CW P_sat | Justified (pulsed vs CW regimes) |
| McCumber | Single home (fiber_amp/spectroscopy) |

### Conventions-sweep summary

All 160 source files scanned. The conventions layer is in good shape: constants single-sourcing,
exp(-i omega t) discipline, lazy heavy-dep imports, exception hygiene, and no mutable defaults are
all clean apart from the instances already logged. The two substantive residuals are the
unenforced EOL policy (S5-13) and the implicit-Optional cluster in fiber_amp (S5-14). Dataclass
frozen-ness is mixed (112 plain / 36 frozen) but reads intentional (mutable specs vs frozen value
objects).

## Stage 6: performance / memory

Five reviewers with measurement license (micro-benchmarks executed on this host; every finding
states measured cost and why the fix is accuracy-neutral). **18 findings: 1 P1, 5 P2, 12 P3**,
plus quantification of four Stage-2 findings and 39 paths confirmed already-well-optimized.

### P1/P2 findings (all measured)

| ID | Sev | Where | Summary (measured) | Status |
|---|---|---|---|---|
| S6-1 | P1 | fiber_amp/dynamics.py:58 | _propagate_fixed integrates per-channel in a Python loop (K*Nt iterations; 84k _cumtrapz calls at K=42): 95% of ASE-on transient runtime. Vectorized 2D cumtrapz over the channel axis is BIT-IDENTICAL (maxdiff 0.0 verified at K=6/42/82) and gives **5.86x end-to-end** (2205->376 ms). | [CONFIRMED] (measured, bit-identical) |
| S6-2 | P2 | fiber_amp/steady_state.py:195 | scipy interp1d validation wrapper is 43% of solve() runtime (75k scalar evals/solve). Lean uniform-mesh interpolator: 7.1x/call, **1.6x end-to-end**, iteration counts identical, final powers bit-identical. | [CONFIRMED] (measured, bit-identical) |
| S6-3 | P2 | tests/test_soa.py:507 | The >500s runtime (S5-8) is entirely 7 thin-wrapper tests that importlib-run full validation/qd_soa_* suites; the other 24 physics tests pass in seconds. Correct fix is TEST ORGANIZATION (@pytest.mark.slow / a validation leg), NOT grid coarsening -- the fine grids are the point of those gates. | [CONFIRMED] (measured) |
| S6-4 | P2 | carriers/devsim_layered.py:504 | Every solve(bias) re-seeds flat-band and re-ramps 0->V, discarding the device's persisted converged state; run_pipeline drives exactly this per bias point. Solve-count model: 2.6x (5-pt sweep) to 6.6x (13-pt) waste; continuation-from-previous-bias is the standard remedy. | [CONFIRMED] (solve-count model) |
| S6-5 | P2 | dynameta/cache.py:167 | Default autosave_every=1 rewrites the WHOLE store per miss: O(N^2) persistence. Measured 623x (N=200) to 1372x (N=1000) overhead vs one final flush; autosave_every=64 recovers nearly all. Negligible for expensive FEM inner solves (<1%); ~2x total wall for the cheap TMM backend it also wraps. | [CONFIRMED] (measured) |

### P3 findings (condensed; [Q] = quantifies an already-logged Stage-2 item)

| ID | Where | Summary |
|---|---|---|
| S6-6 [Q] | fdtd_nd/kernels3d_numba.py:159 | aJ/bJ hoist (S2-5) measured: bit-identical, 1.18x (2D) and **1.41x (3D)** on the mainline numba kernels (440->620 MC/s). |
| S6-7 | fdtd_nd/kernels3d_numba.py:161 | No has_kerr gate: |E|^2+eps_eff+denom recomputed per cell/step with chi3==0 (default); 3 squares/cell/step in 3D -- a large share of the 1.41x. Bit-identical fix. |
| S6-8 | fdtd_nd/kernels2d_numba.py:30 | All nonlinear polarization-state grids allocated unconditionally even with every has_* flag off: ~157 MB dead at 64x64x800 (numba cannot elide). |
| S6-9 [Q] | fdtd_nd/solve3d.py:105 | The 10-coefficient-grid setup (S2-9) measured: 3.7-4.4x setup time and ~2.5x transient peak (55 vs 21 MB at modest sizes; grows with grid). |
| S6-10 | fdtd_nd/kernels3d_numba.py:120 | CPML psi arrays are full-field but exactly zero outside the two npml slabs (verified c==0 interior): ~95% dead; ~100 MB at 64x64x800. Banded storage is bit-identical (interior stretch is the plain derivative). Compute-side banding measured time-neutral -- memory finding only. **[PARTIAL wave-7 (2026-11-02)]** Banded psi (one slot per c != 0 plane + one shared, forever-zero slot) in `_run_3d_oblique_numba`: 210 -> 6.6 MB complex128 at 64x64x800 and 2.78 -> ~2.2 s (40 steps, 4 threads). Not bitwise under fastmath (1.9e-16 on |E| ~ 0.94; two OLD compiles are bitwise equal, so it is the change); parity vs numpy 8e-16 gated (`test_fdtd_backend_contracts.py::test_s6_10_numba_oblique_banded_cpml`). `_te3d_numba` DEFERRED: the same banding MEASURED +27 % (2.04 -> 2.59 s, 64x64x800 x 60) -- the gathered z index defeats SIMD vectorization of the k loop; needs a split-k-loop rewrite. |
| S6-11 | fdtd_nd/kernels3d.py:409 | _run_3d_mo (the ONLY MO kernel) and numpy _run_3d_oblique never got run_3d's prealloc/roll-free rewrite: ~15-20 temporaries + ~7 np.roll copies per step. **[FIXED wave-7 (2026-11-02)]** Both rewritten on run_3d's template (roll-free derivatives, out= chains, hoisted per-step invariants, E/J ping-pong). BIT-IDENTICAL: MO via sha256 pins of 3 cases from 1b1f394, oblique against its verbatim pre-rewrite loop (3 cases, complex digests are not portable); `test_fdtd_backend_contracts.py::test_s6_11_*`. Best of 3: MO 1.18x (16^2x200) / 3.38x (48^2x400); oblique 1.29x / 2.44x. |
| S6-12 [Q] | fdtd_nd/kernels2d.py:98 | S2-11 quantified: ~12 fresh (nx,nz) temporaries per step in the 2D/cupy reference kernel. **[FIXED wave-7 (2026-11-02)]** with S2-11 (see that row). |
| S6-13 | fiber_amp/steady_state.py:149 | Per-channel invariants (gamma*na*sigma, flux prefactors) recomputed on every RHS call (~25k/solve). |
| S6-14 | fiber_amp/noise.py:178 | analyze_noise computes the forward ASE spectrum twice back-to-back (subset of S3-35). |
| S6-15 | carriers/schrodinger_poisson.py:365 | Loop-invariant |psi|^2 recomputed on each of up to 40 inner Newton iterations. |
| S6-16 | carriers/switching.py:78 | Scalar Arrhenius exp inside the per-step Python loop; vectorizable over the temperature array. |
| S6-17 | optics/solver.py:669 | _cell_average probes the R/T grid point-by-point in a Python double loop (NGSolve supports batched point construction). **[FIXED wave-7 (2026-11-02)]** One batched `mesh(x_arr, y_arr, z_arr)` + `field(points)` per z-plane (off-mesh points: element nr == -1, dropped as before; per-point fallback kept). BIT-IDENTICAL: projection + demodulation written as explicit real arithmetic replicating the scalar complex products (numpy's VECTOR complex multiply differs by 1.7e-16). 3.1-4.3x per call (24x24: 67 -> 16 ms per 7-plane band). Gate `tests/test_solver_guards.py::test_s6_17_*`. |
| S6-18 | optics/solver.py:563 | Default diagnostics path integrates the identical volumetric-loss integrand twice (_absorbed_fraction + _per_region_absorption). **[FIXED wave-7 (2026-11-02)]** ONE `Integrate(..., region_wise=True)` pass gives the per-region map and A_independent := sum(map) (the additivity contract is now EXACT); `_absorbed_fraction` / `_per_region_absorption` are thin views of `_absorption_diagnostics`. Old -> new <= 4.4e-16 relative on every output = the old code's own run-to-run TaskManager noise; 57-97 -> 27 ms of absorption integration per two solves. Gate `tests/test_solver_guards.py::test_s6_18_*` (one Integrate call; values vs the old two-pass integrals to 1e-14). |

### Confirmed well-optimized (39 paths; do not re-litigate)

run_3d's prealloc/out= treatment; FESpace/mesh reuse + _detect_bloch_dirs memoization in
make_fem_optical_solver; homogeneous-reference caching in fdtd_nd; the lumenairy bridge's solver
object reuse; results containers; resample cKDTree usage; thermal_fem assembly patterns;
lc_director ladder; qd_gain steady-state iteration; traveling_wave marcher allocations; et al.
(full lists in the per-reviewer coverage notes).

## Stage 7: verification sweep + summary

Seven adversarial verifiers (each briefed to REFUTE, with execution license) covered the six
findings left [OPEN] plus a batch spot-check of seven open physics P3s. Every verdict is backed
by an independent derivation and/or an executed reproduction/disproof.

### Verdicts

| Finding | Verdict | Key evidence |
|---|---|---|
| S3-3 SBE mu^3 | **CONFIRMED P2** | Independent first-principles derivation (chi = mu^2/(eps0 d_qw) x sum, dimensionless term-by-term); executed: max chi = 4.7e-30, /mu = 0.094 (physical). Fix module + oracle in lockstep. |
| S3-5 ENOB /20 | **CONFIRMED P2** | facet_gain_ripple_dB provably power-ratio dB; measured +1.00..+1.27 bit optimism; docstring examples themselves inflated. |
| S3-9 cladding overlap | **CONFIRMED, UPGRADED P1** | Derivation: flux construction requires gamma = power-fraction inside b_dope; executed: confined-doping fiber pump absorption inflated exactly (core/b)^2 = 4.000x through the public path. |
| S4-1 conical lossy mask | **CONFIRMED P2** | Executed on lumenairy 5.24.2: Im(n_sub)=0.001 collapses T 0.942->0.000; phi=0 vs phi=45 discontinuity = the entire transmitted power; reachable, unguarded. |
| S4-2 PMM conical absorption | **REFUTED (as P2)** | Claimed failure unreachable: absorption=True forces retain_internal=True and lumenairy raises NotImplementedError at conical BEFORE the block (executed). Retained as P3 defensive-hardening note. |
| S2-2 graded-tensor snap | **DOWNGRADED P3** | Executed: R for the un-snapped 1e-17-noise graded tensor matches snapped/uniform to ~1e-17; the matvec pathology is unreachable via solve_fem; original rationale obsolete per docs/ngsolve_offdiag_check.py. |
| S1-5/S1-6/S1-13/S1-14/S3-38/S3-39/S2-7 | **ALL CONFIRMED P3** | Each verified as an accurate code fact at correctly-assessed low impact. |

### Graveyard (do not re-report)

- S4-2 as a live correctness bug: the PMM-conical absorption inconsistency cannot be reached
  through any public entry point on lumenairy >= 5.22 (NotImplementedError fires first). Only the
  defensive-gating hygiene item survives.
- S2-2 as a wrong-R risk: empirically disproven through solve_fem's explicit component-sum
  assembly; the branch asymmetry is cosmetic. (The stale 'dense matrix-CF gives wrong R' docstring
  rationale in eps_assembler should be softened -- it is contradicted by docs/ngsolve_offdiag_check.py.)

---

## Summary

**Final tally (151 findings logged; 150 unique defects -- S5-1 and S3-2 are one defect found
twice independently):**

| Severity | Count | Character |
|---|---|---|
| P0 | 0 | -- |
| P1 | 4 unique | fiber_amp metrics drop ConcentrationModel (S3-1); fiber_amp sp-sp beat noise 2x = un-propagated C4-3 fix (S3-2/S5-1); fiber_amp cladding-pump overlap (core/b)^2 inflation (S3-9, upgraded); fiber_amp transient channel-loop 5.9x bit-identical speedup (S6-1) |
| P2 | 35 | physics edge-paths (SBE mu^3 scale, ENOB +1 bit, conical-lossy T-collapse), seam contracts (cache identity/locking/HIT-shape, per_region_absorption keying, effects-dropped-on-fixed-regions, m_pol semantics, ENZ stack ordering, DEVSIM re-ramp), coverage holes (graded-tensor FEM, counter-pumping, CarrierField zarr IO, solve_fdtd_1d, smoke-tier green-wash) |
| P3 | 111 | conventions, stale docs, dead code, guards, quantified perf/memory items |

**Headline observations:**

1. **The mature subsystems are physically sound.** carriers, optics core, fdtd_nd, and the
   lumenairy bridge -- all multiply-audited before -- produced ZERO P0/P1 and had every central
   discretization/formula re-derivation come back clean. The audit's P1s all live in the
   3-day-old fiber_amp subpackage, and all four were invisible to its author-written 57-gate
   suite. Fresh adversarial eyes on fresh code remain the highest-yield audit investment.
2. **Duplication is where correctness goes to die.** The single most instructive defect: the
   beat-noise C4-3 fix (Monte-Carlo-confirmed, documented in soa/ase_noise) was never propagated
   to fiber_amp's independent reimplementation of the same physics. Two other latent drift pairs
   (noise figure, Giles-vs-solver pointwise physics S3-8) are flagged before they diverge.
3. **Oracles must be independent.** The SBE mu^3 dimensional error survived because its
   validation oracle replicated the same construction (as did the smoke-tier numba-parity
   green-wash). Absolute-scale/dimensional gates are cheap and would have caught both.
4. **The seams need contracts, not conventions.** per_region_absorption keying, m_pol semantics,
   cache-hit result shape, depth-array orientation, effect-on-fixed-region silence -- every one is
   a producer/consumer pair that agrees only by convention today. A small set of asserted
   contracts (or shared helpers) would convert silent divergence into loud errors.
5. **Performance headroom is real and accuracy-free.** Measured, bit-identical wins: 5.9x on
   fiber transient, 1.6x on the CW fiber solve, 1.4x on the 3D numba FDTD kernel, 2.6-6.6x on
   DEVSIM bias sweeps (continuation), ~1000x on default cache autosave, plus ~100-250 MB of dead
   allocations per large 3D FDTD solve.

**Recommended remediation order:**

1. P1 wave: S3-1, S3-2 (+ unify the beat-noise implementations), S3-9, S6-1 (all fiber_amp
   + one shared-noise unification; each has a one-to-few-line core fix plus a pinning gate).
2. P2 physics/seam wave: S3-3 (+oracle), S3-5, S4-1, S3-4, S3-6, S3-10, S3-11, S2-3, S4-3,
   S4-5, S4-6, S5-6, S5-7, S4-7.
3. Infrastructure wave: cache (S5-2/3/4, S6-5), pipeline resilience (S5-5), test-infra
   (S5-8/S6-3 slow-marking, S5-9, S5-10, S2-1, S2-4, S3-7, S4-4), DEVSIM continuation (S6-4).
4. P3 sweep: batched by module, lowest priority; the perf/memory P3 batch (S6-6..S6-18 +
   S2-5/9/11 quantified) is a good standalone 'no-loss speedups' PR.

Audit executed 2026-07-17 against main @ 2f22099 (v0.7.0) in 7 stages, 37 Opus reviewer/verifier
agents, ~5.7M subagent tokens, with every P1/P2 physics claim either executed or independently
re-derived, and adversarial refutation applied to all findings that had not already been
empirically demonstrated during review.

---

## Remediation status (executed 2026-07-17, branch fix/audit-2026-07-17)

Four waves, one commit per wave; every fix carries a pinning gate or an executed verification.

### Wave 1 -- all four P1s FIXED (commit b609ee3)
S3-1, S3-2/S5-1 (formula fixed + soa/fiber term-for-term parity gate + absolute thermal-light
pin), S3-9 (b_dope overlap + confined-doping gate), S6-1 (bit-identical vectorization). Riders:
S3-10, S3-37, S3-38 (semi-implicit upconversion), S3-40, plus a latent shape-preservation crash
in photodarkening_loss_per_m that the new gates caught. New gates: 8.

### Wave 2 -- P2 physics/seams FIXED (commit abc3d24)
S3-3 (+oracle in lockstep +absolute-scale GATE A2), S3-5 (+corrected pins), S3-11 (+analytic
centroid gate), S3-4, S3-6 (per_pol contract), S4-1 (loud guard; verified reachable), S4-2
residual hardening, S4-3 (all-axis affine remap), S4-5, S4-6 (basis labeling + warning), S4-7 +
S4-28 (0-d zarr parity verified both backends), S2-3 + S2-16 (ordering contract + flip test),
S5-6 (collapse_fem_region_keys + fallback), S5-7 (fixed-region effect warning), S3-31 (m_modes
via meta), S3-33, S3-39, S3-8 (dead divergent exports removed), S3-32. Perf riders: S6-2
(1.6x, bit-identical), S6-13, S6-14, S3-34, S3-36. fiber_amp suite 54s -> 23s.

### Wave 3 -- infrastructure FIXED (commit 1039a42)
Cache: S5-2 (fingerprints stamped on tmm/fdtd/fdtd-sweep/berreman/pmm factories + closure
warning), S5-3 (pra persisted + served on HIT), S5-4 (merge-on-flush + atomic HDF5 replace;
GATE D2 still green -- stale schemas are never resurrected), S6-5 (autosave_every default 64;
per-miss harnesses updated to request 1 explicitly), S5-12 (solve_sweep proxy). Pipeline: S5-5
(on_error='skip'), S5-11. DEVSIM: S6-4 (bias continuation with flat-band fallback). Test infra:
S6-3/S5-8 (@slow + CI -m "not slow"), S5-9 (zarr on CI py3.11+), S5-10 (rc=42). New coverage:
S1-2 (F12 devsim-free via lazy shims + monotonicity/accuracy/inverter gates), S1-3 (import
contract), S4-4 (resample), S2-4 (fdtd_1d Airy/Drude), S2-1 (graded-TENSOR FEM gate -- PASSES,
graded == uniform to 5e-3).

### Wave 4 -- no-loss speedups + hygiene (this commit)
Speedups (verified): S2-5/S6-6 + S6-7 numba aJ/bJ + Kerr-gate hoists (2D parity bit-identical
0.0; 3D parity in-band), S6-8 conditional Lorentz-state allocation, S2-15 laser_gain uniform-grid
expm hoist, S2-17 fdtd_1d loop invariants, S6-15 SP |psi|^2 hoist, S6-16 switching Arrhenius
vectorization, S2-13 _VAC_TOL tightened to the kernel's 1e-9 (+honest comment). Guards: S2-6
(3D Raman), S2-7 (Lorentz/gain w0*dt, 2D+3D), S2-8 (_ncell getattr). Hygiene: S1-7, S1-12
(GAMMA_E_RAD_ST -> constants), S1-15, S1-16, S1-17, S2-18, S3-13, S3-14, S3-27, S4-9, S4-14,
S5-13 (.gitattributes codifying LF), S5-17 (ruff F/E9 config; note the F-clean sweep is a
follow-up), S5-18 (.coverage untracked), S5-23 (README 0.7.0).

### DEFERRED (deliberate, with reasons)
- S6-10 CPML psi banding (memory-only; banded-loop rewrite risk outweighs ~100 MB on the sizes in
  use today), S6-11 _run_3d_mo/_run_3d_oblique prealloc (edge kernels; run_3d template exists),
  S2-11/S6-12 2D reference-kernel prealloc (cupy path; medium rewrite), S6-17 batched NGSolve
  probe evaluation, S6-18 duplicate diagnostics integral (diagnostics-path only).
- S2-10 numba eager import (lazy-import restructure of the backend dispatch; do as its own PR).
- S2-12 3D jax nonlinearities (feature work, guarded today). S2-14 fdtd_mo half-cell registration
  (changes MO results at the dz/2 level; needs the MO validation battery re-run alongside).
- S3-15/S3-16/S3-17/S3-20/S3-29/S3-30 (soa model refinements needing physics design, documented
  in place), S3-18/S3-42 partial (deeper eh-split/transient coverage), S4-16 (Rytov orientation
  oracle), S4-22 (graphene quadrature oracle), S4-27 (feature-aware symmetry), S4-12 (planar-
  tensor Berreman rerouting), S4-13/S4-10/S4-11 partial docstring rewrites beyond the floor
  bumps, S4-15 (BorResult flux aliases), S4-17-S4-21/S4-23/S4-24/S4-26/S4-29, S5-14 (implicit-
  Optional sweep), S5-15, S5-16, S5-19-S5-22, S1-4, S1-5, S1-6, S1-8-S1-11, S1-13, S1-14, S1-18,
  S3-19, S3-21-S3-26, S3-28, S3-35 partial, S3-41 partial: catalogued P3 polish, batched for a
  follow-up hygiene PR.
