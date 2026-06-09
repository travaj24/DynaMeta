# DynaMeta exhaustive audit -- 2026-06-09

**Git HEAD:** 1688913 (R1-R9 north-star set shipped, 295 tests green)
**Method:** 14-unit multi-agent review (10 subsystems + 4 cross-cutting lenses) -> per-finding
adversarial verification -> synthesis (103 agents, 87 raised, 37 survived the agents' own verify).
**Then:** every load-bearing P0/P1 claim was RE-VERIFIED by hand against the actual source. That second
pass overturned all five "P1 physics/bug" alarms as FALSE POSITIVES (the single-skeptic verify was too
lenient). The honest headline is below.

## Headline

**No P0 or P1 defect survived independent re-verification.** The codebase is in good shape: the
physics is correct under the house conventions, the discipline (independent oracles, byte-identical
off-switches) holds. The audit's real value is a backlog of **defensive guards** (against pathological
callable inputs) and **test-coverage gaps** (error paths exercised only by validation/, not pytest).

### The two "P0" items were non-issues
- `xcut-008` (bridge time-convention guard) and `xcut-009` (FEM independent-absorption / lossless-trap
  check) were flagged then CONFIRMED WORKING by the agents themselves ("no action required"). They are
  existing safety mechanisms, not findings. Mislabeled as P0 by the synthesizer.

### The five "P1 physics/bug" alarms -- all REFUTED on hand re-verification

| Alarm | Claim | Verdict after reading the code |
|-------|-------|-------------------------------|
| **lc2d-1** | Neumann BC stencil "doubles" the x-neighbor -> breaks LC dynamics | **REFUTED.** At `i=0`, `ip=im=1`; scipy `csr_matrix` SUMS duplicate `(row,col)` entries, giving `2*ix2` on V[1] with the diagonal unchanged -- *exactly* the correct zero-gradient Neumann stencil (ghost mirror `V[-1]=V[1]` => `(2V[1]-2V[0])/dx^2`). Correct, intentional idiom. (`lc_director_2d.py:80-92`) |
| **CRIT-1** | `SeparableXYLift` sign-inversion bug for depletion | **REFUTED as a bug.** `dn_3d = dn(x)*dn_y(y)/dn_peak` with the *signed* `dn_peak`: at the spine it reduces to `dn(x)` and for pure-depletion `(neg*neg)/neg = neg` preserves sign. Mixed-sign is already rejected at line 93. Residue: a missing *test* for the depletion case (coverage, P3). (`lift.py:86-112`) |
| **DD-2** | Bipolar 3D equilibrium seed uses wrong sign for p-type | **REFUTED.** `n0 = 0.5(nb + sqrt(nb^2+4ni^2))` with `nb=|doping|` is the majority-carrier *magnitude* (correct for either type); the type sign is carried by `phi_bi = (-1 if acceptor else 1)*V_T*ln(n0/ni)` and the node seed uses DEVSIM's `IntrinsicElectrons` equilibrium model. p-type IS handled. Residue: p-type path lacks a validation test (coverage). (`devsim_3d.py:544-555`) |
| **fdtd-1** | Unmasked negative `wp2` in `effect_eps_to_fdtd_grid` | **REFUTED.** The negative `wp2_abs` lanes are never *selected* (`absorber` requires `er<1` => `denom>0`) and are clamped by `sqrt(maximum(wp2,0))` regardless -- standard vectorized compute-then-select. (`fdtd_seam.py:81-90`) |
| **bmoss-kk-1** | KK grid lower-bound clamp silently truncates the integral | **REFUTED.** A photon-energy grid cannot be negative; `e_lo<0` just means "start near 0", the clamp to `1e-21 J` is physically correct, the absorption edge sits at `Eg~3.6 eV` (nothing below it to truncate), and the assertion at line 521 correctly verifies coverage. (`effects.py:510-522`) |

## Genuine, actionable findings (all low-severity)

### A. Defensive guards -- protect against pathological callable returns / unphysical inputs (P3)
These are real: a caller-supplied callable (`m_opt_kg`, `one_over_tau`, ...) or a mutated field could
return a non-physical value and produce a silent `inf`/`NaN`/gain instead of a clear error. On every
*normal* path the code is correct; these just fail loudly instead of silently.

| id | file:line | guard to add |
|----|-----------|--------------|
| mat-4 | `materials/optical_model.py:188` | `DrudeOptical.eps`: raise if `m_opt_kg(n) <= 0` (silent `inf` wp^2 otherwise) |
| mat-2 | `materials/scattering.py:142` | `ScatteringModel.mobility_of_n`: raise if `m_cond<=0` or `1/tau<=0` (silent `inf`/negative mu) |
| mat-1 | `materials/scattering.py:60-77` | `MatthiessenGamma`: validate `T_K>0` (T_K<0 -> negative damping = gain; T_K=0 -> divide warning) |
| GEO-1 | `geometry/stack.py:50-61` | `Feature.__post_init__`: validate `z_lo_m < z_hi_m` (Layer already validates; Feature does not) |
| inv-des-3 | `optics/inverse_design.py:115-116` | `Fdtd2dDesignProblem.spectrum`: guard zero `mL_inc`/`mR_inc` reference amplitude (silent `NaN`) |

**bridge-1 WITHDRAWN.** A guard rejecting gridded `extra_fields` not broadcast-compatible with `n` was
attempted and REVERTED: it broke `test_assemble_eps_extra_fields_pockels_tensor`, because an `E` field
of shape `(3,)` is a per-point field *vector* `[Ex,Ey,Ez]` (the trailing 3-axis is the vector
component, not a spatial axis) and legitimately broadcasts per-point. The per-point broadcast IS the
intended mechanism; field-bundle values carry varying semantics (scalar T, vector E), so a
shape-against-`n` check is wrong. Not a real defect.
| inv-des-2 | `optics/inverse_design.py:24-50` | `weighted_objective`: raise (or document) when both `target` and `sense` keys are present (`sense` is silently ignored) |

### B. Test-coverage gaps -- error/guard paths exercised only by validation/, not pytest (P3)
The guarded behavior is correct; it just isn't locked in by a fast unit test.

| id | target | missing test |
|----|--------|--------------|
| CRIT-1 (residue) | `core/lift.py` SeparableXYLift | depletion (pure-negative) case -> sign preserved (regression lock) |
| xcut-1 | `core/numerics.py` trapz | empty / single-element / shape-mismatch / reversed-x |
| xcut-3 | `materials/optical_model.py` TabulatedOptical | out-of-range wavelength -> ValueError |
| xcut-6 | `geometry/unit_cell.py` | non-positive period -> ValueError |
| xcut-7 | `geometry/design.py` | unregistered material / missing electrode layer / duplicate electrode -> ValueError |
| xcut-10 | `io/store.py` | zarr auto-detect + cleanup path |
| xcut-9 | `carriers/transient.py` transient_step | dt-floor + max_steps error paths (needs DEVSIM -> validation-only is acceptable) |
| inv-des-1 | `optics/inverse_design.py` Fdtd2dDesignProblem.spectrum | reduces-to-TMM oracle at uniform rho (no independent cross-check today) |
| DD-5 (residue) | bipolar p-type | a p-type (acceptor) validation case (production code is correct) |

### C. Minor robustness / documented-limitation items (P3, mostly already known)
- `cache-1` (`cache.py:137`): cached result reports `solve_time_s=0.0`. Arguably intentional ("served
  from cache, no solve happened"); if the *original* solve time is wanted, unpack `d["solve_time_s"]`.
- `qc-2` (`qcse.py:197-205`): cross-well MQW e-h pairing emits a *non-blocking* RuntimeWarning; could
  also store a `cross_well` flag on `StarkState` for downstream models to check.
- `sp-1` (`schrodinger_poisson.py:391`): in isolated-well mode the `converged` flag can read True even
  during the documented kept-state churn; a sub-band-set-stability check would make it strict.
- `ch-2` (`carrier_heating.py:61-64`): the Sommerfeld `<m*(Te)>` expansion is valid for `kTe << E_F`
  (fine in the tested regime, `kTe/E_F ~ 0.32` at the 2785 K peak); add a docstring caveat / guard for
  `kTe/E_F > ~1`. (Uncertain -- no failure in the validation gates.)
- `DD-4`/`DD-9` (`physics_drift_diffusion.py`): the FD g-factor / quasi-Fermi validity is checked only
  at setup (eta_bg), not re-checked per-iteration during a strongly-accumulating gated solve.
- `fdtd-2` (`fdtd_nd.py:1013,1877`): oblique FDTD raises `NotImplementedError` for Lorentz poles -- a
  documented feature-parity gap with the normal-incidence path.

### D. Organization / docs nits (P3)
- `OPT-FEM-3` (`solver.py:262-265`): UPML CFs computed unconditionally even when `use_upml=False`
  (micro-perf; move inside the `if use_upml` block).
- `SLICE-1` (`core/layered.py:145`): comment about `n_slices` applicability is misleading (code correct).
- `xcut-002` (`solver.py:134-135`): bare `except Exception: pass` around caching Bloch directions; narrow
  to `FrozenInstanceError` or document.

## Audit blind spots (completeness critic)
Single-subsystem review cannot see *integration* bugs. Not systematically probed: the `run_pipeline`
orchestration seam (carrier-region <-> mesh-region alignment, per-bias `extra_fields` key stability),
`OpticalResult` optional-field (`None`/`NaN`) propagation through `SweepResults`/`viz`/`analysis`,
empty/degenerate `Sweep` inputs, DEVSIM/JAX teardown + module-global backend state across repeated runs,
and cache correctness *inside* `run_pipeline` (only tested in isolation). These are the recommended
targets for a follow-up integration-test pass.

## Verdict
GREEN on correctness (no surviving high-severity defect), YELLOW on hardening + coverage (a backlog of
defensive guards and unit tests for already-correct error paths). Recommend landing section A (guards)
and section B (tests) as a small hardening batch; the rest are logged.
