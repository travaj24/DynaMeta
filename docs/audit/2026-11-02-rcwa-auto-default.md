# RCWA inclusion_model default -> auto (v0.13.0 release note section)

Date: 2026-11-02. Scope: `dynameta/optics/lumenairy_bridge/rcwa_backend.py`
(`design_to_rcwa_stack`, `make_lumenairy_rcwa_solver`), lumenairy 5.50.0.

## What changed

1. **Default `inclusion_model='auto'`** (was `'raster'`; user decision). Laterally patterned
   layers whose inclusions are Rectangle / Circle / Ellipse (axis-aligned, disjoint) are passed
   to lumenairy as exact analytic shapes (sinc / Bessel form factors). Polygons, overlapping
   inclusions, tensor-background layers and `absorption=True` solves still rasterize.
2. **Formulation rule.** lumenairy's shapes layers are Laurent (direct rule) by contract:
   `RCWAStack.add_layer(shapes=..., formulation='li')` raises ("Uniform / shapes / tensor layers
   accept only the default"). So:
   - `'auto'` uses analytic shapes **only with `formulation='laurent'`**; with `'li'`/`'fff'` it
     rasterizes every inclusion layer and the requested rule reaches lumenairy (before this fix
     the default `'auto'` silently replaced a requested `'fff'` by Laurent on shapes layers --
     that is what broke the two `test_lumenairy_pmm2d_ext` `*_vs_rcwa_fff` tests);
   - `inclusion_model='analytic'` with `'li'`/`'fff'` raises `ValueError` (in
     `design_to_rcwa_stack` and at `make_lumenairy_rcwa_solver` construction) instead of
     silently solving Laurent.
3. **Raster lattice.** Default point-raster sample count per patterned axis is now
   `S = max(128, 8 N)` (N = that axis' n_orders); it was `max(128, 4 N + 1)`, i.e. lumenairy's
   validity floor. Identical for N <= 16. An explicit `cell_samples` keeps the old formula
   `max(cell_samples, 4 N + 1, 128)`.

## Measured convergence (GATE B fixture of validation/lumenairy_pmm_bridge.py)

600 nm period, 50 % lines centred at P/2, 120 nm thick, air | glass (n 1.5), 1.31 um, normal
incidence. |R - converged reference|. References: TE = PMM degree 48 (self-converged to 1e-11);
TM metal = 0.508768 (see "GATE B correction").

| case | n_orders | 8 | 16 | 24 | 32 | 48 | 64 | 96 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| TE metal (eps -20+2i) | analytic | 2.6e-4 | 3.5e-5 | 1.1e-5 | 4.5e-6 | 1.3e-6 | 5.7e-7 | 1.7e-7 |
| | raster, old S | 1.8e-4 | 4.8e-5 | 7.5e-5 | **4.6e-3** | 3.1e-3 | 2.3e-3 | 1.6e-3 |
| | raster, new S | 1.8e-4 | 4.8e-5 | 2.7e-5 | 1.7e-5 | 8.3e-6 | 4.9e-6 | 2.3e-6 |
| TE dielectric (eps 4+0.3i) | analytic | 8.4e-6 | 1.1e-6 | 3.3e-7 | 1.4e-7 | 4.2e-8 | 1.8e-8 | 5.3e-9 |
| | raster, old S | 3.7e-6 | 3.8e-6 | 4.6e-6 | **8.3e-4** | 5.5e-4 | 4.1e-4 | 2.8e-4 |
| | raster, new S | 3.7e-6 | 3.8e-6 | 1.9e-6 | 1.1e-6 | 5.1e-7 | 3.0e-7 | 1.3e-7 |
| TM dielectric, 'li' | raster, old S | 7.3e-5 | 1.3e-5 | 1.5e-6 | **3.7e-4** | 2.5e-4 | 1.9e-4 | 1.2e-4 |
| | raster, new S | 7.3e-5 | 1.3e-5 | 5.4e-6 | 2.9e-6 | 1.2e-6 | 6.1e-7 | 2.4e-7 |

TM metal (n_orders 32 / 64 / 128): `'li'` raster 4.7e-3 / 1.7e-3 / 8.0e-4 (monotone, algebraic);
Laurent analytic 3.1e-2 / 9.1e-3 / 6.5e-3 (non-monotone; R 0.478 -> 0.518 -> 0.515 ... 0.5095 at
n = 768, approaching from above).

**What the old jump at N = 32 really was.** Not Fourier aliasing: at N >= 32 the old rule
switched to the ODD lattice S = 4N + 1 = 129, on which the edges x = P/4, 3P/4 fall mid-pixel, so
the point raster realised a 65/129 = 0.50388 duty. That is an O(1/S) edge-quantization error, and
it accounts for the number exactly: dR/d(duty) = 1.189 on TE metal (PMM), x 0.00388 = 4.6e-3.
S = 8N + 1 reproduces it (2.3e-3 at N = 32). On S a multiple of 4 the fixture's edges sit between
pixel centres, the realised geometry is exact, and what remains is the genuine aliasing of point
sampling, which S = 8N keeps within ~4x of the analytic truncation error.

**Caveat (measured, same cell, edges NOT on the lattice: duty 0.43 and 0.30).** There the point
raster is quantization-limited under ANY S(N) rule: TE metal, duty 0.43, N = 8..96: new rule
1.1e-2 1.0e-2 4.1e-3 4.6e-4 3.2e-3 4.4e-4 4.4e-4; old rule 1.1e-2 1.0e-2 1.0e-2 5.2e-3 2.8e-5
2.6e-3 2.0e-3 -- erratic, set by where each edge lands (up to half a pixel off). The analytic path
is monotone there (2.3e-4 -> 2.0e-7). This is why `'auto'` prefers analytic shapes; an
area-weighted raster (O(S^-2), with a harmonic companion for `'li'`) would be the raster-side fix
and is NOT part of this change.

**Cost** of the larger lattice: negligible next to the eigensolve. 1-D, N = 32: 11.7 ms (S 129)
vs 11.4 ms (S 256); N = 64: 132.6 ms (S 257) vs 131.0 ms (S 512). In 2-D the raster is S^2 point
tests + one FFT against an O(((2N+1)^2)^3) eigensolve.

## GATE B correction (validation/lumenairy_pmm_bridge.py)

The old GATE B asserted "RCWA (Laurent, raster) converges to the PMM reference on TM metal,
7.3e-4 at n = 32". That was a coincidence: the odd-lattice duty error (0.5039 at S = 129) happened
to cancel most of Laurent's TM error. Properly sampled raster and analytic Laurent both sit at
~3e-2 at n = 32 -- expected: Laurent's rule is the wrong factorization for the field component
normal to a high-contrast wall (Li 1996, JOSA A 13:1870); the inverse rule ('li') fixes it.

The open question "is the ~4.6e-3 TM/'li' floor the PMM reference or lumenairy's RCWA?" is
resolved: **neither is buggy and it is not a floor.** Both methods converge only ALGEBRAICALLY on
this cell because the lossy-metal right-angle corner carries a field singularity (Li & Granet,
JOSA A 28:738 (2011): Type I, Re tau = 0.534, from lumenairy's `classify_from_grating`).

- PMM (bridge, degree 24/32/48/64/96/128): 0.508291 / 0.508511 / 0.508661 / 0.508710 / 0.508744
  / 0.508755; fit R_inf - c d^-2.17 -> 0.508768. lumenairy `pmm_efficiency_1d` with 1 and 6
  elements/region extrapolates to the same 0.508768 (to 1.5e-7). The n_orders of the PMM
  (21 vs 31 vs 41) does not change R at all.
- RCWA 'li' (S = 8N, n_orders 256/384/512/768): 0.508391 / 0.508524 / 0.508587 / 0.508647; fit
  N^-1.15 -> 0.508747 (2e-5 from the PMM limit). Laurent approaches from above (0.50945 at 768).
- Fitted rates: PMM ~ d^-2.17 ~ 4 Re tau; RCWA 'li' ~ N^-1.15 ~ 2 Re tau. Dielectric TM (Re tau
  0.33) and both TE cases converge to 1e-8..1e-10 in both methods.
So the old degree-24 PMM reference was 4.8e-4 low, and 'li' at n = 32 is 5.0e-3 low because it
converges as ~N^-1.15. No lumenairy reproducer is warranted.

New GATE B: PMM references self-converged (TE degree 32 -> 48: 1.1e-11; TM degree 96 -> 128:
1.1e-5, reference = degree 128, 1.3e-5 below the extrapolated limit). TE: analytic AND default
raster decrease monotonically over n_orders 16/32/48 (3.5e-5 / 4.5e-6 / 1.3e-6 and 4.8e-5 /
1.7e-5 / 8.3e-6; bars 3e-6 / 2e-5). TM: 'li' decreases monotonically over 32/64/128 (4.7e-3 /
1.7e-3 / 8.0e-4, bar 1.5e-3, rate N^-1.28 > 0.7); Laurent printed (3.1e-2 / 9.1e-3 / 6.5e-3), not
gated as convergence. Discrimination: Laurent / 'li' at n = 128 = 8.2x (> 5x); a 1 % duty change
moves the TM reference by 2.3e-2 (> 10x the 'li' error).

GATE E (per-layer absorption, same file) rode the same artifact: its "strictly shrinking over
n_orders 32/64/96" RCWA split (Laurent TM, rasterized because absorption=True) came from the odd
lattices S = 129/257/385 (realised duty 0.5039/0.5019/0.5013). On the new lattice the Laurent TM
split is non-monotone (0.020 / 0.049 / 0.033 at 32/64/96, then 5.0e-4 at 256 vs the gate's PMM
split). Rewritten as: TM agreement at n_orders 256 (< 3e-3, measured 5.0e-4) with the n = 64
overshoot printed, plus a tight TE leg (Laurent valid): 4.6e-5 at n = 32 (< 3e-4).

## Pinned numbers that move

- `inclusion_model='auto'` (the default) changes every representable patterned layer from the
  point raster to the analytic form factors. Example: the `test_geometry_guards` rectangle pillar
  (n_orders 5) R 0.1910304737839642 (raster, S 128) -> 0.19094726805155943 (analytic); that test
  pins byte-identity of the raster path and now passes `inclusion_model='raster'` explicitly.
- The new raster lattice moves raster results only for n_orders >= 17 on some axis (S 128 ->
  8N for 17..31; 4N + 1 -> 8N from 32). This reaches the default path wherever the raster is
  still used under `'auto'`: `absorption=True`, `'li'`/`'fff'`, polygons, overlaps, tensor
  backgrounds. In this repo: pmm_bridge GATE B/E (rewritten above) and
  `examples/lumenairy_gated_grating.py` (n_orders 24, absorption=True: S 128 -> 192). No unit-test
  pin moved (tests/test_lumenairy_*.py, test_bridge, test_fdtd_seam, test_geometry_guards all
  pass; every other raster pin is at n_orders <= 16 or passes an explicit `cell_samples`).
- Validation printouts moved by the default flip (32f5545 -> this change; all still PASS, and
  the analytic path lands closer to the independent oracle each time):
  lumenairy_rcwa_bridge GATE C (asymmetric pillar vs FEM) dRx 9.4e-4 -> 7.9e-4, dRy 1.2e-3 ->
  9.8e-4; GATE H cross-engine conical vs PMM 5.0e-4 -> 1.4e-4; lumenairy_pmm2d_bridge GATE B
  RCWA-vs-pure at n_orders 8 / 16: 4.0e-3 / 5.8e-4 -> 3.9e-3 / 6.1e-4 (final bar 1.5e-3).
- lumenairy_pmm2d_bridge GATE I / J request `formulation='fff'`; under the bare default flip
  (c98d3cf) they FAILED because `'auto'` routed the circle / ellipse / rectangle to the
  Laurent-only shapes path (FFF |d| 1.3e-4 / 7.4e-5 / 6.2e-4, no longer bracketing). The
  formulation rule restores the rasterized FFF layer: 1.3e-5 / 1.3e-5 / 4.1e-5 -> PASS, the
  pre-flip numbers.
- validation/lumenairy_rcwa_jax.py GATE A/B compares the bridge against the JAX twin's own
  128-sample raster, so the bridge side now passes `inclusion_model='raster'` explicitly (like
  with like). Its explicit `cell_samples=128` at n_orders 5 is unaffected by the lattice change.

## Opt-out (one line)

`make_lumenairy_rcwa_solver(..., inclusion_model='raster', cell_samples=128)` reproduces the
raster numbers of the commit before the default flip (32f5545) bit for bit at every n_orders: an
explicit `cell_samples` keeps the old lattice formula `max(cell_samples, 4N + 1, 128)`.
`inclusion_model='raster'` ALONE reproduces them only for n_orders <= 16 (identical lattice
S = 128); from n_orders 17 it uses the new `max(128, 8N)` lattice. (Against the v0.12.0 release
itself, raster results additionally differ where the 2026-11-02 bridge fixes apply:
boundary-crossing inclusions now wrap periodically, and inclusions on a uniform-tensor
background are no longer dropped.)
