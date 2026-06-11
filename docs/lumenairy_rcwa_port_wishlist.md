# Lumenairy RCWA -- gaps & wishlist for porting into DynaMeta

> **SUPERSEDED (2026-06-11):** This 2026-06-01 plan framed the integration as a code
> PORT, with Lumenairy P1/P2 gaps to close first. The 2026-06-10 v0.5 decision
> (`docs/roadmap_v0.5_integration_photonics.md`, A0-A4) replaced it with a live
> REQUIRED dependency bridge (`lumenairy>=5.14.2`, `dynameta/optics/lumenairy_bridge/`),
> shipped without P1/P2 (declared not blockers). The structured `eps_cell` path is
> live via the RCWA backend (`validation/lumenairy_rcwa_bridge.py`). The body below
> is kept untouched as the historical record.

**Date:** 2026-06-01
**Purpose:** DynaMeta's v0.3 modulation roadmap (`docs/roadmap_v0.3_modulation.md`) needs a
fast, differentiable, periodic optical solver. Rather than build one, port Lumenairy's
native RCWA (`Free_Space_Optics/Lumenairy/lumenairy/elements/rcwa.py`, ~3380 LOC, ~v5.6).
This write-up is the **Lumenairy-side feature list to add BEFORE the port** + the
DynaMeta-side adapter work, based on an audit of the current module on 2026-06-01.

---

## What Lumenairy RCWA already has (covers most of DynaMeta's needs)

The `RCWAStack` builder/solver is the right entry point:

- **Multi-layer stacks**, 1-D and 2-D (crossed) periodic, conical incidence (theta, phi).
- Per-layer specification via `add_layer(thickness, ...)`:
  - `eps` (scalar) -- uniform spacer;
  - `eps_cell` (Sx, Sy) -- **isotropic numeric** in-plane permittivity (FFT-sampled);
  - `eps_tensor_cell` (Sx, Sy, 3, 3) -- **full-tensor / anisotropic** (Pockels, LC);
  - `shapes` + `eps_background` -- analytic shape Fourier transforms + dual-Laurent.
- **Convention is IDENTICAL to DynaMeta**: `exp(-i omega t)`, `Im(eps) > 0` for absorbers,
  metres, angles in radians. No sign-convention bridge needed at the port boundary.
- `RCWAResult`: per-order R/T efficiencies, **absorptance A**, 0-order complex **Jones
  reflection/transmission** (-> the modulator's complex r/t/phase), `per_order_amplitudes`.
- Convergence machinery: Li inverse-rule (1-D TM/metals), dual-Laurent ('li'/'fff') for 2-D,
  principal-branch eigenvalue stability (`_sqrt_decay`, the high-order blow-up fix),
  Wood-anomaly regularization, Richardson extrapolation (`rcwa_extrapolate`).
- Performance: BLAS thread control, thickness-independent eig caching/reuse (repeated DBR
  layers solved once), CuPy GPU, and a JAX (autodiff) path for the 1-D entry point.
- Oracle-validated: 1-D Airy/TMM ~1e-16, vs grcwa/inkstone <2e-3, 2-D bit-exact vs an
  independent reference, anisotropic vs inkstone ~1e-4, energy conservation ~1e-12.

This already serves: free-carrier/ENZ (numeric `eps_cell`, graded via z-slices), Pockels/LC
(`eps_tensor_cell`), thermo-optic / Franz-Keldysh (numeric `eps_cell`), 2-D metasurfaces, and
lossy metals (1-D exact; 2-D at the dual-Laurent rate).

---

## Gaps to add in Lumenairy BEFORE porting (prioritized)

### P1 -- Differentiable 2-D / `RCWAStack` solve (autodiff + adjoint)  [highest value]
**Today:** the JAX path is validated only for the 1-D entry (`rcwa_efficiency_1d_jax`). The
stable-eig custom-VJP (`_jax_eig_stable`) is dimension-agnostic internally, but
`RCWAStack.solve` routes through eig **caching** (`_cached_homogeneous_eigenmodes`) and is not
a validated differentiable path; there is no 2-D / stack JAX entry.
**DynaMeta needs:** the optimization / inverse-design layer differentiates a **2-D metasurface**
FOM through the solver. Without 2-D/stack autodiff, DynaMeta can only finite-difference (slow)
or optimize 1-D.
**Ask:** make `RCWAStack.solve` JAX-traceable end-to-end (bypass/disable the eig cache on the
traced path; route the per-layer eig through `_eig_for(xp)`'s stable VJP) and add a validated
2-D adjoint example. Stress-test gradient stability near eigenvalue degeneracies (ENZ, metals).

### P2 -- True normal-vector FFF for 2-D metals (Goetz-Schuster)  [medium]
**Today:** 2-D uses the **dual-Laurent** factorization (`'li'`/`'fff'`, the E_z inverse rule).
There is no normal-vector field (Goetz-Schuster / Popov-Neviere) for 2-D metallic corners.
**DynaMeta needs:** plasmonic metal patches (the Park Au nanopatch, gap-plasmon modes) converge
only at the dual-Laurent rate in 2-D -- the same slow/erratic convergence the sibling
Metasurface_Modulator project hit with grcwa (which also lacks it). A true normal-vector FFF
gives Li-rate 2-D convergence for sharp metal features.
**Ask:** add the normal-vector-field 2-D factorization as a `formulation='fff_nv'` option.

### P3 -- Rigorous 2-D PATTERNED-anisotropic factorization (Li-2003)  [medium-low]
**Today:** a 2-D `eps_tensor_cell` is convolved at the Laurent rate (no normal-vector for
tensor discontinuities). UNIFORM tensor layers (an LC cell, a LiNbO3 thin film) are EXACT;
only a **laterally-patterned tensor** (a 2-D EO/LC metasurface with in-plane tensor
discontinuities) converges slowly.
**DynaMeta needs:** patterned Pockels/LC metasurfaces (vs uniform EO layers). Uniform-tensor
EO devices are already fine.
**Ask:** the Li-2003 crossed-grating anisotropic factorization for the 2-D tensor path.

### P4 -- `add_graded_layer(profile, n_slices=..., rule=...)` convenience  [low]
**Today:** a continuous eps(z) (a carrier-accumulation layer, a thermal/field gradient) must be
hand-sliced into many thin `add_layer` calls.
**DynaMeta needs:** the carrier-modulated ENZ layer and thermo-optic/field gradients are
naturally graded in z. A convenience that auto-slices a `profile(z)` (scalar or tensor) into
staircase layers with a convergence-controlled slice count (and a midpoint/trapezoid rule)
would make the DynaMeta z-slicer a one-liner and centralize the slicing-convergence logic.
**Ask:** optional; DynaMeta can also do the slicing on its side (see adapter below).

### P5 -- Fully general 3x3 tensor with z-coupling (exz/eyz/ezx/ezy)  [low / niche]
**Today:** the anisotropic path is an in-plane tensor [[exx,exy],[eyx,eyy]] + ezz (no
xz/yz coupling).
**DynaMeta needs:** only for **magneto-optic** (Faraday/Voigt gyrotropy, antisymmetric
off-diagonal with z-coupling) -- the most niche mechanism, not in the current 4-family scope.
**Ask:** defer unless magneto-optic is prioritized; flag clearly that the tensor path is
in-plane-only so a gyrotropic tensor is rejected rather than silently truncated.

### P6 -- Packaging for clean extraction  [low, porting-ergonomics]
**Today:** `rcwa.py` imports Lumenairy backend helpers (`array_namespace`, `backend_name`,
`to_numpy`, `JAX_AVAILABLE`, `is_jax_array`, GPU shims) and the `JonesField` bridge
(`RCWAResult.to_jones_field` / `apply_reflection`).
**DynaMeta needs:** to copy `rcwa.py` + a minimal backend shim WITHOUT dragging in the whole
Lumenairy package or the JonesField pipeline.
**Ask:** keep the numeric core importable with only `{numpy, optional jax/cupy}` -- isolate the
backend helpers into a small self-contained module and make the `JonesField` bridge an optional
import (DynaMeta uses the raw R/T/A + complex 0-order amplitudes, not JonesField).

---

## DynaMeta-side adapter work (NOT Lumenairy's job -- listed for the port)

**RCWA-independent prep BUILT 2026-06-01 (commits/files noted), so the port is now "wire
RCWAStack to an existing, validated seam":**

1. **`LayeredStackSolver` Protocol** (`core/interfaces.py`) + a concrete first impl
   **`TmmLayeredSolver`** (`optics/tmm_reference.py`) -- DONE. The seam exists and is proven by
   a real solver now; the `RcwaSolver` adapter is the SAME shape (build a `LayeredStack`, solve)
   and its body is the only part that waits for the RCWA code.
2. **z-slicer** -- DONE for the laterally-uniform/graded case: `core/layered.LayeredStack`/
   `LayeredSlab` (slab specs mirror `RCWAStack.add_layer`: scalar/`eps_cell`/`eps_tensor_cell`/
   `shapes`), `slice_profile` (graded eps(z) -> uniform slabs), `slice_eps_field` (gridded
   EpsField -> slabs; scalar if laterally uniform, else `eps_cell`), and
   `tmm_reference.layered_stack_from_design`. VALIDATED vs the FEM on a graded 8-sublayer slab
   (graded-TMM R 0.3372 vs FEM 0.3362, |dR|=0.0010; `validation/graded_tmm_vs_fem.py`) and by
   `tests/test_layered.py`. The structured (in-plane `eps_cell`/tensor per slab) path is stubbed
   in the data model and finalized with the RCWA port.
3. **Result mapping**: `RCWAResult` -> DynaMeta `OpticalResult` (R, T, A; complex r/t and
   phase from the 0-order Jones; `A_independent` cross-check vs 1-R-T).
4. **Validation gate**: RCWA vs the NGSolve FEM on the real Park lossy-Au-patch + ENZ-ITO cell
   (the audit's "missing independent oracle"); and vs `tmm_reference` on unstructured stacks.

---

## Net assessment

Lumenairy's RCWA is **substantially complete for DynaMeta** -- the `RCWAStack` API, the tensor
layer support, and the matching `exp(-i omega t)` convention mean most of the porting work is
already done. The one feature that genuinely blocks a DynaMeta goal is **P1 (2-D/stack
autodiff)**, which gates 2-D inverse design; **P2/P3** improve accuracy/speed for metallic and
patterned-tensor metasurfaces; **P4-P6** are conveniences/ergonomics. Recommend adding **P1**
(and ideally P2) in Lumenairy first, then porting.
