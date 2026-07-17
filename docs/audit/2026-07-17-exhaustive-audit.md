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
| S1-4 | P3 | convention | carriers/__init__.py:11 | `__all__` lists three devsim-requiring submodules, so `from dynameta.carriers import *` eagerly imports devsim -- contradicting the docstring's lazy/solver-free claim (empirically demonstrated). drivers/__init__ applies the correct discipline for the same situation. | [CONFIRMED] |
| S1-5 | P3 | seam | carriers/physics_density_gradient.py:105 | DG in-Newton re-points region continuity to ElectronCurrentDG but leaves the ohmic CONTACT equation on classical ElectronCurrent, so extracted terminal current on a DG device is read from the pre-quantum flux on the contact edge (solve itself unaffected). | [OPEN] |
| S1-6 | P3 | physics | carriers/density_gradient.py:57 | Diagnostic `quantum_potential_V` uses nested `np.gradient` for u'' -- a 2*dz same-parity stencil that decouples even/odd nodes (sawtooth risk on coarse grids). Load-bearing BVP path unaffected. | [OPEN] |
| S1-7 | P3 | convention | carriers/physics_density_gradient.py:166 | Hard-wall path re-derives V_T locally from KB*T_REF/Q_E instead of importing constants.V_T. | [CONFIRMED] |
| S1-8 | P3 | test | carriers/thermal_fem/transient.py:57 | Time-dependent load paths (flux_of_t/joule_of_t, two-temp source_*_of_t) have zero test/validation coverage; they rely on a subtle ngsolve LinearForm `.vec` lifetime pattern. Reviewer ran a LIVE check (matches constant-flux path to 1.1e-13 K) -- correct today, unguarded against ngsolve upgrades. | [CONFIRMED] |
| S1-9 | P3 | physics | carriers/devsim_3d.py:469 | 3D equilibrium multi-semiconductor heterostack develops no internal built-in potential (per-region Phi_c0 makes psi=0 exact equilibrium; interfaces enforce potential continuity only; no band offsets). Undocumented limitation; DD/bipolar variant already hard-raises. | [OPEN] |
| S1-10 | P3 | organization | carriers/devsim_3d.py:62 | In-method `import devsim` lazy ceremony is defeated by module-top import of physics_equilibrium (which imports devsim unconditionally); missing devsim yields a raw ImportError instead of devsim_layered's friendly hint. | [CONFIRMED] |
| S1-11 | P3 | convention | carriers/thermal_fem/common.py:43 | ThermalLayerTwoTemp.k_e() defaults electron conductivity to the LATTICE k_thermal, so the documented G->inf single-T reduction carries k=2*k_thermal -- easy to misuse. | [OPEN] |
| S1-12 | P3 | convention | carriers/llg.py:40 | Electron gyromagnetic ratio (CODATA) declared as a module literal instead of living in dynameta.constants. | [CONFIRMED] |
| S1-13 | P3 | physics | carriers/lc_director.py:504 | Freedericksz seed/wrong-branch estimate uses mean (K11+K33)/2 instead of the deformation-appropriate constant; for high splay/bend contrast the 1.3*V_th safety net can spuriously flag a correct solution just above the true threshold. Solve itself seed-independent. | [OPEN] |
| S1-14 | P3 | physics | carriers/lc_director.py:875 | Reported chiral `twist_energy_J_m2` omits the constant (1/2)K22 q0^2 complete-square baseline -- reporting-semantics offset (can be negative at natural twist); EL equations unaffected. | [OPEN] |
| S1-15 | P3 | convention | carriers/carrier_heating.py:36 | T_REF re-declared locally (=300.0) instead of imported from constants; M_E imported unused. | [CONFIRMED] |
| S1-16 | P3 | convention | carriers/sp_carrier.py:80 | Stale inline comment claims nonparabolicity is a post-hoc parabolic fill, contradicting the code (fully Kane-nonparabolic Newton) and the corrected class docstring -- pre-C6-5 leftover. | [CONFIRMED] |
| S1-17 | P3 | organization | carriers/schrodinger_poisson.py:320 | Dead helper laplacian_matrix(): built and bound at line 327, never used (Newton builds its banded Jacobian directly). | [CONFIRMED] |
| S1-18 | P3 | test | carriers/schrodinger_poisson.py:45 | Sommerfeld large-x (>40) branches of _fd1/_fermi_log and the dF_1/dx=F_0 identity are docstring-claimed but not unit-tested (only the spence branch is exercised). | [CONFIRMED] |

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
| S2-7 | P3 | seam | optics/fdtd_nd/solve2d.py:235 | Lorentz and gain ADE builders lack the analogous w0*dt stability guard entirely (2D and 3D): a far-above-band pole diverges silently. | [OPEN] |
| S2-8 | P3 | convention | optics/fdtd_nd/solve3d.py:377 | solve_fdtd_3d_mo._ncell accesses cyclotron_wc_rad_s directly (fill loop uses getattr default): a plain Drude layer in an MO stack crashes during grid sizing on an otherwise-valid input. | [CONFIRMED] |
| S2-9 | P3 | memory | optics/fdtd_nd/solve3d.py:105 | ~10 full (nx,ny,nz) Lorentz/Raman/gain material grids allocated unconditionally even for plain dielectric solves: ~1.3-1.6 GB of pure zeros at 256^3 held live through the solve (~+34% footprint). | [CONFIRMED] |
| S2-10 | P3 | convention | optics/fdtd_nd/backends.py:13 | numba (+numba.cuda) imported eagerly at package-import time (jax/cupy correctly lazy): numpy-only users pay the multi-second LLVM import. | [CONFIRMED] |
| S2-11 | P3 | performance | optics/fdtd_nd/kernels2d.py:98 | 2D numpy/cupy reference kernel allocates full-grid temporaries every step; the 3D kernel was explicitly rewritten to preallocate+out= (audit 6.2) but 2D never got the treatment -- and run_2d_te IS the cupy path (per-step device mallocs). | [CONFIRMED] |
| S2-12 | P3 | gap | optics/fdtd_nd/kernels3d_jax.py:10 | 3D jax kernel omits lor/chi2/raman/gain (2D jax twin carries all): 3D gradient-based inverse design of dispersive/active media is impossible though the forward physics exists. Correctly guarded (raises), so a capability gap not a silent bug. | [CONFIRMED] |
| S2-13 | P3 | seam | optics/fdtd_seam.py:41 | _VAC_TOL=1e-3 loss guard on end media is 6 orders looser than the kernel's own 1e-9 guard; a weakly-absorbing substrate (k=1e-3) is silently truncated to lossless, biasing T ~2% high. Stale comment describes a different purpose. | [CONFIRMED] |
| S2-14 | P3 | convention | optics/fdtd_mo.py:241 | fdtd_mo registers materials at Yee E-nodes (arange*dz) while EVERY sibling engine uses cell centers (+0.5): a systematic dz/2 registration offset between the MO 1-D reference and the rest of the family (absorbed today by validation tolerances). | [CONFIRMED] |
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
| S3-19 | soa/qd_gain.py:772 | power_to_photon_density duplicates photon_density with divergent array handling. |
| S3-20 | soa/qd_gain.py:351 | with_detailed_balance_taus constrains ES<->GS only; WL<->ES ratio thermodynamically unconstrained. |
| S3-21 | soa/traveling_wave.py:492 | eta_in (input-coupling NF penalty) silently ignored on the segmented-GVD path of amplify_coherent. |
| S3-22 | soa/thermal.py:57 | Dense (n,n) matrices + dense solve for a tridiagonal fin system (docstring claims tridiagonal solve). |
| S3-23 | soa/traveling_wave.py:307 | saturation_curve settle window sized by transit count with no convergence guard; every validation overrides the default. |
| S3-24 | soa/calibration.py:194 | report['bandwidth_nm'] aliases MATERIAL FWHM while the identically-named datasheet target is the NET -3 dB width: spurious match hides the documented 3.6x narrowing. |
| S3-25 | soa/ase_noise.py:179 | Dead conditional branch (executes `pass`). |
| S3-26 | soa/ase_noise.py:173 | Docstring claims m_pol applied to *_mean accumulators; code keeps them per-pol. |
| S3-27 | soa/__init__.py:45 | __all__ omits four imported public ASE/noise functions. |
| S3-28 | soa/ase_noise.py:116 | noise_figure ignores internal loss when alpha_i>0 but Gamma_g is None. |
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
| S4-11 | lumenairy_bridge/rcwa_backend.py:7 | Docstring claims 'no sign translation' but the module performs the validated p-pol lab-basis -> p-hat sign/scale reconciliation. |
| S4-12 | lumenairy_bridge/rcwa_backend.py:127 | Laterally-uniform anisotropic tensor forced through full 2-D RCWA (O((2Nx+1)(2Ny+1)) modes for a planar layer); Berreman is the intended path. |
| S4-13 | lumenairy_bridge/pmm_backend.py:16 | Stale scope contract ('PMMStack has no conical source', 't is None') contradicted by the implemented synthesis. |
| S4-14 | lumenairy_bridge/bor_backend.py:206 | time.time() for solve timing where every sibling uses perf_counter(). |
| S4-15 | lumenairy_bridge/bor_backend.py:154 | BorResult.fundamental_result() leaves R_flux/T_flux None; every Cartesian backend sets them -- cross-backend schema inconsistency. |
| S4-16 | lumenairy_bridge/emt_screen.py:69 | order=2 Rytov tensor ORIENTATION (perp vs par components) never validated against a hand oracle; a transposed convention would ship. |
| S4-17 | lumenairy_bridge/pmm_backend.py:181 | end_media_indices computed twice per solve (also PMM2D, RCWA). |
| S4-18 | core/bridge.py:108 | 2D-lift path hardcodes lateral axis to 'x'; stack_axis=='x' would alias lateral==vertical, unguarded. |
| S4-19 | core/lift.py:44 | IdentityLift.apply dead (bridge 3D branch bypasses lift) with a non-physical placeholder y-axis. |
| S4-20 | core/carrier_field.py:98 | dump writes axis_order attr that load never reads -- dead metadata implying an unhonored guarantee. |
| S4-21 | core/layered.py:146 | slice_eps_field structured branches do not normalize descending z (slice_profile does): raises instead of handling. |
| S4-22 | core/graphene.py:61 | Interband Kubo terms have no independent numerical oracle (FEM validation consumes the same graphene_sigma). |
| S4-23 | core/bridge.py:123 | assemble_eps materializes y-replicated copies of 2D extra fields where a broadcast view suffices. |
| S4-24 | core/effects/magneto.py:98 | Orphaned section-divider comments from the effects monolith split. |
| S4-25 | reliability/em.py:49 | Three-way self-contradiction on Black n=2 nucleation-vs-growth labeling (relabel applied only to module docstring). |
| S4-26 | reliability/hci.py:52 | Same incomplete-relabel: 'Takeda' in error message/oracle vs 'Hu lucky-electron' in docstring. |
| S4-27 | geometry/design.py:85 | device_symmetry()/detect_symmetry_reduction() ignore Stack.features: an off-center via would not demote symmetry. |
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
| S5-14 | fiber_amp/spectroscopy.py:81 | 21 implicit-Optional annotations (concrete type + =None) vs the 230x Optional[...] convention. |
| S5-15 | carriers/devsim_layered.py:55 | _effective_dos_m3 duplicated verbatim in two carriers modules. |
| S5-16 | carriers/devsim_3d.py:358 | warnings.warn drift: 11/50 omit stacklevel, 29/50 omit category. |
| S5-17 | pyproject.toml:34 | ruff+mypy are dev deps with no committed config and no CI lint gate: behavior unpinned. |
| S5-18 | .gitignore:13 | .coverage binary artifact tracked in git. |
| S5-19 | Makefile:4 | `make test` help text understates scope (runs the full suite when solver extras are installed). |
| S5-20 | optics/fdtd_seam.py:475 | The two FDTD run_pipeline entry points disagree on capability: per-wavelength supports structured cells on lossless non-vacuum end media; the sweep-aware one (which run_pipeline prefers) raises. |
| S5-21 | optics/tmm_reference.py:202 | TmmLayeredSolver populates per_region_absorption but leaves A_independent=None, breaking the contract that the map sums to A_independent (every other producer sets both). |
| S5-22 | dynameta/analysis.py:1 | Module docstring describes ~40% of the file; the majority is electrical modulator FOM code. |
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
| S6-10 | fdtd_nd/kernels3d_numba.py:120 | CPML psi arrays are full-field but exactly zero outside the two npml slabs (verified c==0 interior): ~95% dead; ~100 MB at 64x64x800. Banded storage is bit-identical (interior stretch is the plain derivative). Compute-side banding measured time-neutral -- memory finding only. |
| S6-11 | fdtd_nd/kernels3d.py:409 | _run_3d_mo (the ONLY MO kernel) and numpy _run_3d_oblique never got run_3d's prealloc/roll-free rewrite: ~15-20 temporaries + ~7 np.roll copies per step. |
| S6-12 [Q] | fdtd_nd/kernels2d.py:98 | S2-11 quantified: ~12 fresh (nx,nz) temporaries per step in the 2D/cupy reference kernel. |
| S6-13 | fiber_amp/steady_state.py:149 | Per-channel invariants (gamma*na*sigma, flux prefactors) recomputed on every RHS call (~25k/solve). |
| S6-14 | fiber_amp/noise.py:178 | analyze_noise computes the forward ASE spectrum twice back-to-back (subset of S3-35). |
| S6-15 | carriers/schrodinger_poisson.py:365 | Loop-invariant |psi|^2 recomputed on each of up to 40 inner Newton iterations. |
| S6-16 | carriers/switching.py:78 | Scalar Arrhenius exp inside the per-step Python loop; vectorizable over the temperature array. |
| S6-17 | optics/solver.py:669 | _cell_average probes the R/T grid point-by-point in a Python double loop (NGSolve supports batched point construction). |
| S6-18 | optics/solver.py:563 | Default diagnostics path integrates the identical volumetric-loss integrand twice (_absorbed_fraction + _per_region_absorption). |

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
