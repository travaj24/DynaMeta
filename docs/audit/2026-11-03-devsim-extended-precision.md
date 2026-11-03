# Extended-precision (128-bit) DEVSIM solves

DEVSIM phase, release v0.14.0, workstream D2b, branch `wip/dv_extprec` (off `feat/devsim-2.11-v0.14`
at 6baa9a6, after the noise merge). New capability: one opt-in switch that builds a device with
DEVSIM's 128-bit model evaluation and equation assembly, and solves it with DEVSIM's extended Newton.

**Bottom line.**
- **DC: it works.** On a GaN-like junction it turns a current that is pure round-off in double
  (wrong by 1e5x to 1e9x, not even conserved) into one that matches an independent oracle to < 1e-3.
  Transient runs the same mixed-precision Newton. It is gated to run, to move off the double answer
  and not to leak, but it has no oracle of its own here.
- **Heavy doping: it converges at a tolerance double cannot meet.** On a heavily doped p+n junction,
  the equilibrium converges at DynaMeta's tolerance where double fails.
- **ssac and noise: it does NOT fix them.** DEVSIM factorizes and solves that linear system in
  double even in extended mode.
- **Default off is byte-identical.** This was checked by sha256 across commits on DC, ssac, noise,
  transient and the 2-D layered builder.
- **Cost.** 6.3x per Newton iteration in 1-D and 19-22x in 2-D (76k nodes), with identical iteration
  counts.

## What changed

| File | Change |
|---|---|
| `dynameta/carriers/extended_precision.py` (new) | `enable`, `solve`, `extended_solver`, `available`, `is_enabled`, `any_enabled` |
| `dynameta/carriers/devsim_layered.py` | `LayeredDevsimBuilder(..., extended_precision=False)`; `enable` right after `create_device` |
| `dynameta/carriers/devsim_3d.py` | `Stacked3DSpec.extended_precision` (default False), `from_design(..., extended_precision=False)`; `enable` at build; the equilibrium `ds.solve` routed through `XP.solve` |
| `dynameta/carriers/dc_solve.py`, `ac_analysis.py`, `transient.py`, `noise_analysis.py` | `ds.solve` swapped for `XP.solve` (7 sites) |
| `validation/extended_precision.py` (new) | oracles, 1-D builders, gates A-D, `--hashes` (cross-commit fingerprint), `--cost` |
| `tests/test_extended_precision.py` (new) | 15 tests + 1 strict xfail, 74 s at `-n 2` |

## Mechanism (DEVSIM documentation and source)

The DEVSIM 2.11 manual has a "Solver" chapter; its section 9.3 is "Extended precision"
(https://devsim.net/solver.html). The source was read at devsim/devsim e3a25f0, the 2.11.2 line
(CHANGES.md top entry 2.11.2).

**9.3.1 Platform dependence.** "Extended precision is available on all binaries."
- On Linux x86_64 it uses GCC's 128-bit float.
- On the other platforms (x64, arm64, aarch64) it uses boost `cpp_bin_float_quad`, which is "similar
  to 128-bit precision".

Our Windows pip wheel (2.11.2) uses the second. This was confirmed in the binary: the
`cpp_bin_float<113,...>` instantiations carry a 113-bit significand.

`get_parameter(name="info")["extended_precision"]` is `True` on the wheel (measured). The 2.11.1
release note shows the same key.

**9.3.2 How to control.** Three parameters:

| Parameter | Manual's words | Where DEVSIM reads it (source) | Scope |
|---|---|---|---|
| `extended_model` | "extended precision model evaluation" | When a node/edge/element model is **created** (`create_node_model<double, extended_type>(region->UseExtendedPrecisionModels(), ...)` in every model factory) | Region, then device, then global (`GlobalData::GetDBEntryOnRegion`) |
| `extended_equation` | "extended precision equation assembly" | When an equation is **created** (`EquationCommands.cc`) | Region, then device, then global |
| `extended_solver` | "extended precision matrix for Newton and linear solver" | At **every** `solve`, `get_matrix_and_rhs` and `set_initial_condition` call; it selects `Newton<float128>` (`MathCommands.cc`) | Global only |

The manual also notes that "Default geometric models are also calculated with extended precision."
In the source this happens unconditionally. `CreateDefaultModels` instantiates `EdgeCouple`,
`NodeVolume` and the other geometric models as `float128` on any extended-capable build, whatever
the parameters say.

**9.3.3 Kahan summation.** In extended mode, `kahan3`/`kahan4` keep their Kahan compensation
(manual: "better than 128-bit floating precision"). DynaMeta's SG currents and the Poisson charge
already use `kahan3`.

**What the docs do not say: the linear solve stays double.** Each `float128` preconditioner does the
same thing:
1. casts the assembled matrix and RHS to `double`;
2. factorizes and solves in `double`;
3. casts the solution back.

This covers `MKLPardisoPreconditioner<float128>::LUFactorMatrix` / `LUSolve`. It also covers
`ExternalPreconditioner<float128>`, which is the UMFPACK 5.1 `custom` solver this wheel loads.
`GMRES` is refused outright ("GMRES not implemented for extended precision", a string in the
binary).

The "extended solver" is therefore a **mixed-precision Newton**:
- the residual and Jacobian are assembled in 128-bit;
- the correction is solved in 64-bit;
- the next residual is 128-bit again, so the outer loop is self-correcting.

What this means for each solve type:
- **DC and transient** converge to a 128-bit-accurate state. Measured Newton relative updates fall
  to ~1e-33, against ~1e-15 in double. The iteration counts are identical, so the Jacobian is not
  degraded.
- **ssac and noise** are a SINGLE linear solve with no outer loop. They get a correctly rounded
  matrix and a double solve.

**State that is per Newton type.** DEVSIM keeps separate `TimeData<double>` and
`TimeData<float128>` transient histories. A transient must therefore run its `transient_dc` and
every BDF step with the same Newton type.

**Cost.** The model evaluation and assembly run in software quad arithmetic; the factorization is
the same double one. See the measurements below.

## API

One switch, scoped to a device:

```python
LayeredDevsimBuilder(design, extended_precision=True)           # 2-D
Stacked3DSpec(..., extended_precision=True)                      # 3-D
Stacked3DSpec.from_design(design, extended_precision=True)
# hand-built devices (1-D scripts, validation):
from dynameta.carriers import extended_precision as XP
ds.create_device(mesh=..., device=dev)
XP.enable(dev)            # BEFORE any model or equation
```

`dynameta/carriers/extended_precision.py` (new) is the one shared implementation:

- **`enable(device)`** sets DEVICE-level `extended_model` + `extended_equation`.
  - DEVSIM clears device parameters on `delete_device` (2.1 release note), so the opt-in dies with
    the device. A later device of the same name is default (gated).
  - It refuses a device that already carries an equation or the `Potential` solution. DEVSIM fixes a
    model's type at creation, so a late call would leave the existing physics in double without any
    error.
  - It raises on a build without extended precision.
- **`solve(**kw)`** is the drop-in for `ds.solve`. With no opted-in live device, it is exactly
  `ds.solve(**kw)` and writes no parameter, so default off is byte-identical by construction.
  Otherwise it runs inside `extended_solver()`.
- **`extended_solver()`** sets the global `extended_solver` for that one call and restores the
  previous value (or `False` when it was never set) in a `finally`. DEVSIM treats "never set" as
  `False`. The restore also runs after a convergence failure (gated), and the context manager is
  re-entrant.
- **Mixed live devices.** DEVSIM solves all live devices in one Newton. An opted-in device next to a
  default one is therefore solved together with it, and `solve` raises an `AdvisoryWarning` naming
  both.
- **`available()`, `is_enabled(device)`, `any_enabled()`** are helpers.

Every DynaMeta solve site now calls `XP.solve`:

| File | Solve sites |
|---|---|
| `dc_solve.py` | Newton DC, plus both Gummel sub-solves |
| `ac_analysis.py` | ssac |
| `transient.py` | `transient_dc` and `transient_bdf1` |
| `noise_analysis.py` | noise |
| `devsim_3d.py` | the equilibrium DC solve that bypassed `solve_dc` |

The layered and 3-D builders already reach DEVSIM through `solve_dc`.

A raw `ds.solve` in user code on an opted-in device runs the DOUBLE Newton over 128-bit models.
Wrap it in `XP.extended_solver()`, or call `XP.solve`, to get the extended Newton.

## Oracles (independent of the code under test)

Both oracles are in `validation/extended_precision.py` and use only scipy:
- **Ideal-diode (Shockley) current**, with finite neutral widths:
  `J0 = q n_i^2 [Dn/(NA Ln) coth(Wp'/Ln) + Dp/(ND Lp) coth(Wn'/Lp)]`
- **Sah-Noyce-Shockley depletion-region SRH current.** `q int U dx`, with
  `U = n_i^2 (e^{V/V_T} - 1) / (tau (n + p + 2 n_i))` (n1 = p1 = n_i, flat quasi-Fermi levels). It is
  integrated with `scipy.integrate.quad` on the depletion-approximation potential (parabolic, with
  the 2 V_T Debye correction).

Constants come from `scipy.constants`. The zero-bias conductance `G0` is the derivative of that
current at V = 0.

**The oracle reproduces DEVSIM where DEVSIM is resolved.** On the symmetric Si diode, the double DC
`dI/dV` is 5.3868e-4 S/m^2 against an oracle of 5.3837e-4 S/m^2: 6e-4.

| Gate | Device | Quantity | Double | Extended | Gate |
|---|---|---|---|---|---|
| A | GaN-like p-n, n_i 1e-4 m^-3 (1e-10 cm^-3), N_A = N_D = 1e23, tau 1 us, 1 nm junction mesh, 317 nodes | J at +1.0 V (oracle 8.2473e-18 A/m^2) | -1.33e-8: 1.6e9x off, wrong sign | 8.2548e-18: **+9.0e-4** | < 3e-3 |
| A | same | J at +1.5 V (oracle 1.4969e-13) | +2.77e-8: 1.9e5x off | 1.4980e-13: **+7.4e-4** | < 3e-3 |
| A | same | edge-current conservation (max-min)/\|J\| at 1.0 / 1.5 V | 1.05e4 / 4.9e3 | 2.1e-5 / 1.0e-9 | < 1e-3 |
| B | p+n, N_A 1e25 / N_D 1e22, Si-like n_i 1e16, tau 20 us, 734 nodes, absolute tol 1e8 | equilibrium Newton | **Convergence failure** | converges | converges |
| B | same | dI/dV at 0 V, +-0.1 mV (oracle G0 2.5823e-5 S/m^2: SRH 2.52e-5 + diffusion 6.0e-7) | n/a (at tol 1e13: 2.5822950490e-5) | 2.5822950490e-5: **-1.2e-6** | < 1e-4 |
| B | same | ssac Re Y(1 Hz) | -3.5e-5 to -6.8e-5 (state dependent) | **+5.995e-5 = 2.3 G0** | strict xfail |
| C | 1-D Si diode through DC, ssac, noise, transient; 2-D layered metasurface at Vg 0 / +2 V | sha256 of every solved array, base 6baa9a6 vs this branch | | | 7/7 identical |
| D | default, then extended, then default, in one process | sha256 (5 paths) + `extended_solver` after | | | identical; `False` |

**What fails in double, and what does not.** In the GaN junction the densities and the
electrostatics are FINE in double:
- the built-in potential matches V_T ln(N_A N_D / n_i^2) to 0;
- n p = n_i^2 e^{V/V_T} holds at every node to 1e-4 (law of the junction);
- the minority density is 1.0000e-31 m^-3, which is n_i^2/N.

What is lost is the **current**. The SG edge current is a difference of ~1e8 A/m^2 majority drift
and diffusion terms, so it has a ~4e-5 A/m^2 round-off floor, eight to thirteen decades above the
true 1e-17 to 1e-13 A/m^2. Extended moves that floor to ~1e-22.

Wide-bandgap leakage, generation and recombination currents are the use case. Densities alone are
not; n_i^2 does not underflow a double until n_i < 1e-154.

**The heavy-doping gate is a convergence gate, not an accuracy gate.** Loosened to an absolute
tolerance of 1e13, which is what `validation/devsim_noise.py` uses for its shot diode, double
converges and gives the same `dI/dV` to 1e-12. Extended removes the need to loosen the tolerance.
It converges at 1e8, with the same 86 Newton iterations as double at 1e13.

**ssac: extended precision does NOT fix it.** This answers the noise agent's -0.27 S/m^2.

| Configuration | Re Y(1 Hz) [S/m^2] | Oracle G0 |
|---|---|---|
| p+n diode driven at the **p+ end** (`validation/devsim_noise.py` shot diode), double, absolute tol 1e13 | -0.2712 (reproduced) | 2.58e-5 |
| same, extended | -0.0727 (frequency-flat 1 Hz to 10 kHz) | 2.58e-5 |
| same diode driven at the **n end** (this gate), double, absolute tol 1e13 | -3.5e-5 / -6.8e-5 (moves with the DC state) | 2.58e-5 |
| same, extended | +5.995e-5 | 2.58e-5 |
| symmetric Si diode (N 1e22, n_i 1e16, tau 1 us), double | 5.22e-4 (-3.0%) | 5.384e-4 |
| same, extended | 5.49e-4 (+2.0%) | 5.384e-4 |

Three things are established about this floor:
1. **It is not a Jacobian error.** Newton converges quadratically (2e-4, then 1.6e-11, then 8e-24).
2. **It is not a missing equilibration or refinement.** A scipy `splu` of DEVSIM's own
   `get_matrix_and_rhs` AC matrix (static + i w dynamic) reproduces it to the digit: -0.027099609375
   = 111/4096 in double and 0.07275390625 = 298/4096 in extended. These are short binary fractions,
   i.e. the circuit current is quantized at 2^-12 S/m^2. Row and column equilibration and 3 steps of
   iterative refinement change it only within that floor: 1.8e-2, then -1.0e-1, then -1.1e-2.
3. **The cause is the conditioning of the AC system in double.** The circuit-current row
   cancels terms ~1e12 (the p+ contact's dJ/dp at 1e25 m^-3). A double factorization of the
   correctly rounded 128-bit matrix cannot resolve 2.6e-5 out of that. Driving the lightly doped end
   is 3 decades better, but still above G0.

The low-frequency conductance that IS resolved is the DC `dI/dV`. It matches G0 to 1e-6 in either
precision once the DC solve converges.

### Default-off fingerprint (gate C)

The fingerprint is a sha256 (first 16 hex digits) of every solved array, from
`python validation/extended_precision.py --hashes`. The base column was run from a detached
checkout of 6baa9a6 with `--root`. In both runs `extended_solver` was never set ("None").

| Path | Base 6baa9a6 | `wip/dv_extprec` |
|---|---|---|
| 1-D DC at 0 V (`solve_dc`) | 02821f6cf0e27c0c | 02821f6cf0e27c0c |
| 1-D DC, 0.1 V forward ramp + terminal current | dc71a4d2fc3e1591 | dc71a4d2fc3e1591 |
| 1-D ssac C, G at 1 Hz and 1 MHz | 542fc394b3d5134d | 542fc394b3d5134d |
| 1-D noise S at 1 Hz and 1 MHz | a342fdbff9ac27ae | a342fdbff9ac27ae |
| 1-D `transient_step` t, I | 16d9be81d8b52111 | 16d9be81d8b52111 |
| 2-D layered metasurface, Vg 0 (every region's node fields) | 74be34a4f627eaa5 | 74be34a4f627eaa5 |
| 2-D layered metasurface, Vg +2 V | b8d1ff3830876c5e | b8d1ff3830876c5e |

Gate D runs the extended version of the 1-D row set: all five fingerprints move. The default run
after it reproduces the first column.

### Regression on the touched paths

These files were run at `-n 2`, not slow: `test_dc_solve_gummel`, `test_devsim_noise`,
`test_devsim_smoke`, `test_transient_dt_floor`, `test_gated_builder` and `test_geometry_guards`.

- **First run: 94 passed, 5 failed.** All 5 failures were in `test_transient_dt_floor`. Those
  control-flow gates replace `transient.ds` with a stub, and the first cut of `XP.solve` called the
  real devsim, bypassing it.
- **Fix.** `XP.solve(ds_module, ...)` takes the caller's module, and `transient.py` passes its own
  `ds`. A stand-in without `get_device_list` takes the default path.
- **Result.** That file now passes 6/6, and `tests/test_extended_precision.py` passes 15/15 plus
  1 strict xfail (73 s).

## Cost (box A, DEVSIM 2.11.2, UMFPACK 5.1, OpenBLAS)

| Case | Double | Extended | Ratio |
|---|---|---|---|
| 1-D GaN ramp 0 to 1.5 V (16 Newton solves, 129 iterations, 317 nodes) | 0.567 s (4.4 ms/it) | 3.580 s (27.8 ms/it) | **6.3x** (same iteration count) |
| 1-D p+n equilibrium (presolve + coupled, 86 iterations, 734 nodes) | 0.684 s (8.0 ms/it) | 4.427 s (51.5 ms/it) | **6.4x** (same iteration count) |
| 2-D layered reference metasurface, DD, 76,430 nodes, Vg 0 (5 solves, 5 iterations) | 1.81 s (0.363 s/it) | 33.6 s (6.73 s/it) | **18.5x** |
| same, continuation to Vg +2 V (11 solves, 43 iterations) | 11.7 s (0.273 s/it) | 256.2 s (5.96 s/it) | **21.8x** (same iteration count) |

These are solve times only, from `ds.solve(info=True)` iteration counts on one run each (box A
shared with other agents, so expect +-20 %). An earlier run without the iteration count gave wall
times of 3.44 s / 26.5 s (Vg 0, including the build) and 6.87 s / 223.6 s (to +2 V).

The ratio grows from 6x in 1-D to ~20x in 2-D because the 2-D assembly carries more models per edge:
the triangle edge-couple models, the FD g-factor rational fit, and the 9-region interfaces. All of
these are evaluated in software quad, while the double factorization (the same in both modes) is a
smaller share of the iteration.

**Fields on the 2-D metasurface match to round-off.** The degenerate ITO metasurface is
well-conditioned, and its extended and double fields agree to round-off:
- n: max relative difference 4.0e-16 at +2 V, 0 at 0 V;
- psi: max absolute difference 1.7e-16 V.

Extended buys nothing there. Use it only where a current or density spans more decades than a double
can hold.

## Limits

- **ssac and noise are not precision-lifted** (above). The DEVSIM AC and noise solves are one double
  factorization. A 128-bit linear solve would need DEVSIM to ship a quad factorization, or a
  DynaMeta-side refinement with a 128-bit residual. The latter needs matrix-vector products that
  DEVSIM does not expose in 128-bit: `get_matrix_and_rhs` returns doubles.
- **Readout is double.** `get_node_model_values` and `get_edge_model_values` return float64. A
  density below 1e-308 reads as 0, even though DEVSIM holds it in 128-bit (exponent range +-4932).
- **The circuit is double.** DEVSIM assembles circuit elements in double
  (`LoadMatrixAndRHSOnCircuit` casts).
- **Parameters are double.** `set_parameter` values are Python floats, so n_i itself must be
  > 1e-308. Expressions such as `n_i^2` are then evaluated in 128-bit.
- **Raw `ds.solve`** on an opted-in device runs the double Newton over 128-bit models; use
  `XP.solve` / `XP.extended_solver()`.
- **Mixed live devices.** A default device solved alongside an opted-in one is solved by the
  extended Newton. You get a warning, not a refusal. Every DynaMeta builder's `teardown()` deletes
  all devices.
- **3-D.** The switch is gated at build (the opt-in precedes the physics) but not solved here. A
  real 3-D FD region costs minutes of DEVSIM symbolic differentiation even in double.

## Open questions (user decisions)

1. **The ssac and noise floor.** Should D2a's FDT gate (and users) be steered to the DC `dI/dV` for
   the f -> 0 conductance, and to driving the lightly doped contact? Both are cheaper and more
   effective than extended precision.
2. **Builder-level ssac/noise with extended on.** These run, but carry the floor above. Warn when
   `extended_precision=True` meets ssac, or leave it documented only?
3. **Expose on `Design`/`TransportModel`?** Today the switch lives on the builders and the 3-D spec
   only. `run_pipeline` users cannot reach it without constructing the builder.

## Merge notes for the integrator

- **Edits outside the new module.** Six files have one-line `ds.solve` to `XP.solve` swaps plus an
  import, and two builders gain a keyword. `physics_bipolar_dd.py` is untouched.
- **Merge order.** If a later workstream adds a `ds.solve` call, route it through
  `extended_precision.solve` (grep `ds\.solve(` in `dynameta/` must stay empty).
- **Callers that resolve devsim through their own module attribute.** If a module's tests stub that
  attribute (as `transient.py`'s do), pass it: `XP.solve(ds, type=...)`.
- **The cross-commit fingerprint.** `python validation/extended_precision.py --hashes` prints the
  default-off sha256. To compare against another commit, run it from that checkout with
  `--root <that checkout>`.

## Release-note lines

- **Extended precision (opt-in).** `LayeredDevsimBuilder(..., extended_precision=True)`,
  `Stacked3DSpec(extended_precision=True)` / `from_design(..., extended_precision=True)`, or
  `carriers.extended_precision.enable(device)`.
  - The device is built with DEVSIM 128-bit model evaluation and equation assembly, and solved by
    DEVSIM's mixed-precision extended Newton on every DynaMeta solve path.
  - It is device-scoped and restored after every call, and never leaks into a later device.
  - Default off is byte-identical (sha256 across commits on DC, ssac, noise, transient and 2-D).
- **Wide-bandgap currents.** A GaN-like junction's forward current, which is pure round-off in double
  (1e5x to 1e9x off), now matches an independent Sah-Noyce-Shockley oracle to < 1e-3. A 1e25/1e22
  p+n junction converges at the absolute tolerance 1e8, which double cannot reach.
- **Cost.** 6.3x per Newton iteration in 1-D; 19-22x on the 2-D layered metasurface (76k nodes),
  with the same iteration counts.
- **Known limit: ssac and noise are not fixed by extended precision.** DEVSIM solves the AC system
  in double. Re Y(1 Hz) of a p+n diode goes from -0.27 to -0.073 S/m^2 against a true 2.6e-5. Use
  the DC dI/dV for the f -> 0 conductance.
