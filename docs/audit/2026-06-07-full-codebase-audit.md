# Full codebase audit -- 2026-06-07

Multi-dimension audit of DynaMeta (~9.75k LOC, 60 modules) across **code organization, convention
standardization, physical accuracy (optics + carriers), correctness bugs, compatibility, and
performance**. Method: 8 dimension-lenses fanned out over the tree, then **every candidate finding was
adversarially verified** (default-refute; reject claims that contradict the locked conventions, are
already handled, mis-read the code, or rest only on energy-closure reasoning).

## Result: 31 candidates -> 5 confirmed (0 critical, 0 high, 1 medium, 4 low)

The codebase is in strong shape: **no physics errors, no convention violations of consequence, no
compatibility breakage, and no critical/high-severity bugs.** 26 of 31 candidates were refuted on
concrete, code-grounded grounds (see "Refuted" below) -- the verification pass did its job.

### Confirmed findings

| # | sev | dim | file:line | finding |
|---|-----|-----|-----------|---------|
| 1 | medium | perf | optics/fdtd_seam.py (per-wavelength seam) + fdtd_nd.py:398 | The per-wavelength `make_fdtd_optical_solver` re-runs a full FDTD transient (incl. the `200*tau` settling tail) for EVERY wavelength, while `fdtd_sweep_spectrum` already gets the whole spectrum in ONE solve. Opportunity: an opt-in sweep-aware pipeline path that calls the broadband sweep once per bias and interpolates. NOT a free drop-in -- it freezes eps at band center, so it is approximate for a dispersive (ITO/Drude) active layer (keep the per-wavelength path, or a Drude-Lorentz fit, for accuracy). |
| 2 | low | conventions | optics/solver.py:385 | Hardcoded `_Z0 = 376.730313668` instead of deriving from `constants.py`. **FIXED** this commit (`_Z0 = 1/(EPS0*C_LIGHT)`), matching `core/graphene.py`. |
| 3 | low | org | materials/__init__.py:5-6,16-17 | `EPS0/C_LIGHT/Q_E` re-exported from `materials` (secondary path). **DECLINED**: `M_E` on that path IS used by tests/validations as a convenience for effective-mass material definitions; ripping out the others while keeping `M_E` is churn for ~0 benefit. |
| 4 | low | perf | optics/fdtd_nd.py:603-607 (and 156-157) | The numba kernels recompute the Drude ADE coeffs `aJ/bJ` per cell per step. Hoisting to precomputed `(nx,ny,nz)` arrays saves 2 divisions/cell but ADDS 2 array loads/cell on a memory-bandwidth-bound kernel -- **ambiguous net win**; the finding itself rates it "reasonable to defer/decline." **DEFERRED** (would also require re-proving byte-equivalence of the locked kernels). |
| 5 | low | perf | optics/fdtd_nd.py:603-607 | Duplicate of #4 (low confidence). DEFERRED. |

### Notable refutations (the adversarial pass earned its keep)

- **gate_cv "axis confusion" (x3, claimed high)** -- refuted: the only 2-D carrier-field producer
  (`devsim_layered.py:482`) always keys axes `{"x","y"}` with `y` the through-stack axis; a 2-D field
  keyed `"z"` is not constructible by any shipped code, and `stack_axis` is a bridge-side concept
  `gate_cv` never receives. `analysis.py:64` reading `grid_axes_m["y"]` is correct.
- **FDTD de-embed "phase bug" (claimed medium)** -- refuted: the ~half-cell (`k0*dz ~ 360/res deg`)
  absolute-phase offset is known, bounded, validated (`fdtd_seam_vs_tmm.py` GATE 3, 15 deg tol), and
  cancels in the modulator's bias-relative dphase. The proposed `+0.5*dz` fix accounts for only one of
  two half-cell ambiguities and could worsen it.
- **numpy/scipy "version drift" (x2)** -- refuted: a full grep finds ZERO numpy-2.x-only or removed-1.x
  APIs; the wide floors (`numpy>=1.24`, `scipy>=1.10`) are intentional packaging choices and the code is
  genuinely dual-compatible. Raising them would only reduce install compatibility.
- **Several perf claims (CPML caching, VoxelCF rebuild, vacuum-run caching, eps(lambda) caching, mesh
  Bloch-probe, 3D-vs-2D numba balance)** -- refuted: each mis-modeled the cost (one by ~4-5 orders of
  magnitude: `_cpml_z` is ~2e-7 of solve work, not "~1%") or mis-read the architecture (the broadband
  sweep already does one solve; per-wavelength vacuum fields differ in shape AND content because the
  band edges scale with lambda).
- **Silent `except` handlers (resample Delaunay, db stash)** -- refuted: the Delaunay fallback
  deterministically re-raises the identical error on the next line; the db stash writes an attribute
  nothing reads.

## Actions taken this commit
- **Fixed #2** (Z0 from constants).
- Logged #1 as a roadmap item (sweep-aware pipeline fast path; pairs with the Drude-Lorentz fitter).
- Declined #3 (M_E convenience), deferred #4/#5 (ambiguous micro-opt on locked byte-equivalent kernels).

149 pytest pass; full FDTD/seam validation suite green.
