# Lifting the quasi-static ASE limit on the transient march (2026-09-15)

`simulate_transient` marched the inversion through a FROZEN-population power step. That step is
exact at the march's fixed point and unusable away from it once the ASE stops being a
perturbation: the audit-A-7 monitor (2026-08) could only MEASURE the failure and warn. This work
adds an opt-in step that makes the population change implicit and solves the resulting
ASE-coupled power problem to the steady solver's own tolerance, on BOTH amplifier classes. The
default march is unchanged -- asserted byte-for-byte, not argued.

Branch `feat/march-self-consistent-ase`, merged onto main `1bef664` (v0.11.3: the Er:Yb
migration calibration, the co-doped temperature laws and `eryb_fit`). Version
`0.11.5 -> 0.11.6` -- the slot after PR #30 (0.11.4) and PR #27 (0.11.5), which merge
ahead of this one.

---

## 1. What was built

| Where | What |
| --- | --- |
| `march_ase.py` (new) | `SelfConsistentControl`, `solve_self_consistent_step`, `auto_switch` / `predicted_step_ase_error` / `needs_ase_probe`, `residual_views`, `check_ase_mode` -- the inner fixed-point solver, its warm-start/damping state, and the "auto" switch criterion |
| `dynamics.py` | `ase_mode` / `ase_tol` / `ase_max_iter` / `ase_step_residual` on `simulate_transient` AND `simulate_transient_eryb`; the population update of each march factored into one `advance()` closure that both modes call; the per-step margins the switch criterion reads; the new `meta` block |
| `tests/test_march_self_consistent_ase.py` | 15 gates (17 with parametrisation) |

### API (exact call forms)

```python
from dynameta.optics.fiber_amp import simulate_transient

# UNCHANGED -- the default, bit-for-bit the v0.11.2 march
tr = simulate_transient(amp, t_grid, n_nodes=161)

# every step ASE-coupled and implicit in the populations
tr = simulate_transient(amp, t_grid, n_nodes=161, ase_mode="self_consistent")

# quasi-static until the monitor predicts a violation, then self-consistent for those steps
tr = simulate_transient(amp, t_grid, n_nodes=161, ase_mode="auto")

tr.meta["march_valid"]                 # the flag to test in the new modes
tr.meta["ase_mode_steps"]              # (Nt,) int8: 0 quasi-static, 1 self-consistent
tr.meta["ase_switch_steps"]            # indices, and ["ase_switch_times"] the times
tr.meta["n_self_consistent_steps"]     # and ["ase_inner_iterations_total"/"_max"], ["ase_inner_relax"]
tr.meta["max_self_consistency_residual"]

# opt-in, ANY mode: the step-by-step closure (one extra propagation per step)
tr = simulate_transient(amp, t_grid, ase_mode=..., ase_step_residual=True)
tr.meta["max_step_power_residual"]
```

`ase_tol` (default `1e-6`, the steady solver's own) and `ase_max_iter` (default 120) set the
inner solve. Every other argument, the return type and the resolved-ASE arrays are unchanged.
`frame_as_steady` is unchanged except that it now carries `march_valid` and `ase_mode` onto the
frame alongside the existing `quasi_static_valid` -- the audit-A-7 "the flag cannot be lost
through the frame" rule, applied to the new flag.

---

## 2. Why the obvious formulations do not work

**A literal "frozen-population BVP" is a no-op.** At genuinely fixed populations `g(z)` and
`s(z)` are functions of `z` alone, so `dP/dz = u (g P + s)` is LINEAR and the two directions are
completely decoupled. `_propagate_fixed` is already its exact integrating-factor solution; a
relaxation over it converges in one sweep by construction and returns the same numbers. There is
nothing to make self-consistent while the populations are held.

**Sub-stepping does not help either**, for the same reason -- a finer `z` or `t` grid returns the
same over-amplified ASE from the same exact linear solve. (The module docstring has said so since
audit A-7; this note is where the *alternative* is worked out.)

**What is actually missing** is the coupling between the powers and the population CHANGE across
the step. The frozen step is EXPLICIT in the ASE feedback: ASE generated inside the step never
depletes the inversion that made it, so a population error is multiplied by `exp(INT g dz)`
before it re-enters the update. That is a property of the ITERATION, not of the discretization,
which is why no grid refinement reaches it.

---

## 3. The step

Write `y` for the population state at the start of a step (`nbar2`, or the co-doped `(f2, b2)`
pair / `(f2, b2c, b2nc)` triple), `dt` for the step, and

* `advance(P)` -- the march's OWN population update (the single-ion exponential integrator, or
  the co-doped exponential Rosenbrock step of `docs/audit/2026-09-13-eryb-transient-closure.md`)
  taken from `y` over `dt` with the rates read off a power profile `P`. Unchanged algebra; it is
  now a closure that both modes call, so there is exactly one home for it.
* `gain_source(y)` -- the march's own per-`z` gain and spontaneous source.
* `propagate(g, s)` -- the march's own exact frozen-gain sweep.

The self-consistent step solves

```
P = propagate(gain_source(y_end)),        y_end = advance(P)                            (*)
```

for `(P, y_end)` simultaneously, by the same damped fixed-point iteration the steady solver uses
and against the SAME two residuals (`steady_state._relaxation_residuals`: endpoint and interior
profile, each channel normalised by its own peak), stopping at `ase_tol`. The populations are
then advanced with the converged powers -- i.e. the existing step, evaluated at a converged `P`.

**Limits, and why they are the right ones.**

* `dt -> 0`: `advance(P) -> y` for every `P`, so (*) collapses to ONE call of `propagate` at the
  frozen populations -- the quasi-static step, recovered exactly. MEASURED at `dt = 5.3e-11 s` on
  the 400 W booster: `9.8e-10` in `nbar2`, `3.2e-7 dB` in gain.
* `dt -> infinity`: the exponential integrator returns the root of its local balance, so (*)
  becomes "the powers propagate through the gain the populations they sustain" -- the steady
  relaxation problem itself.
* FIXED POINT: at the amplifier's steady state `advance` returns `y` for ANY `dt` (the increment
  carries the rate right-hand side as a factor), so (*) is satisfied by the pair unchanged. The
  march's fixed point is still exactly the one the DEFAULT march has. Gated as such: a 60-step
  constant-drive march seeded from `solve()` goes stationary in BOTH modes (last 20 frames equal
  to `< 1e-12`) and stops on the SAME state (`< 1e-9` in `nbar2`, `< 1e-6 dB` in gain). Both
  modes relax off `solve()`'s own profile by the same `7.3e-5` in `nbar2` (`2e-4 dB`), which is
  the march's trapezoid quadrature against `solve()`'s adaptive LSODA and predates this work.
* STABILITY: inside a step the gain now DEPLETES as the ASE grows -- a population excursion that
  raises `INT g dz` raises the ASE, the ASE raises the stimulated-emission rate in `advance`, and
  `y_end` comes back down. That negative feedback is what the explicit step lacks.

**What the result arrays mean is unchanged.** The REPORTED powers at `t_i` remain the
instantaneous frozen-population propagation at the populations the same frame reports -- those
two are the exact quasi-static pair at that instant. Only the population UPDATE reads the
self-consistent powers. That is deliberate: it keeps `ase_fwd_W`, `power_zt` and
`frame_as_steady` meaning exactly what they meant, and it makes the two modes differ only through
the population trajectory, by the size of the ASE perturbation rather than by `O(dt)`.

**Affordability.** The inner solve warm-starts from the previous step's converged profile, so a
marching solve costs 2-3 inner iterations per step rather than the 10-30 a cold one needs; the
damping rung that WORKED is carried across steps (downward only) instead of re-walking the ladder
every step; and `auto` pays only its decision on a step that does not need switching -- two gain
evaluations, plus ONE extra propagation on the steps where the gain is both amplifying and moving
(`meta['ase_probe_propagations']` counts those; on a march at its operating point it is zero).

---

## 4. "auto"

Per step, the explicit update is taken first, and the step is handed to the self-consistent
solver if any of:

1. **Predictive (estimate).** ASE leaves the fiber as `exp(INT g dz)`, so a step across which the
   ASE gain integral drifts by `D` mis-states its own ASE by `e^D - 1`; times the ASE's share of
   the launched power, that is the fraction of the launched power the frozen step gets wrong.
   Switch above `1e-2`.
2. **Predictive (probe).** When the estimate is structurally blind -- a COLD fiber emits nothing,
   so the ASE share is exactly 0 whatever the gain does -- and the projected gain integral is
   positive and moved, `auto` spends ONE extra frozen-gain propagation at the projected
   populations and compares the two ASE shares exactly, against the same `1e-2`.
   `meta['ase_probe_propagations']` counts these.
3. **Backstop.** The two audit-A-7 margins, each against HALF its documented limit, at the
   current and the projected populations. This covers a march that STARTS inside the bad regime,
   where there is no drift to detect.
4. **Stickiness.** A trip arms the next 8 steps. The quasi-static instability is a property of
   the iteration: the error a frozen step injects is amplified by the FOLLOWING steps, and by the
   time local quantities show it the march is already ringing. MEASURED on the reference co-doped
   amplifier at a uniform 1 ms grid: with no window `auto` still rings at a GROWING +-0.3 dB and
   lands `0.011 dB` from the fully self-consistent march; with the window it lands `4.4e-8 dB`
   from it.

A march sitting at its operating point trips none of these and takes zero self-consistent steps.

---

## 5. Gates (measured 2026-09-15, numpy 2.4.4 / scipy 1.17.1, Windows / Zen4)

### (1) The default did not move

Two independent halves, both in `tests/test_march_self_consistent_ase.py`:

* **cross-path, EXACT.** `ase_mode="quasi_static"` returns `np.array_equal` arrays to the default
  call -- `nbar2_zt`, `signal_out_W`, `pump_out_W`, `signal_gain_dB`, `ase_fwd_W`, `ase_bwd_W`,
  `power_zt`, and the co-doped `beta_yb` / `beta_yb_coupled` / `beta_yb_uncoupled`.
* **cross-version.** End states pinned to the values v0.11.2 (`21dff60`) produced, on the 400 W
  Yb booster, the C-band EDFA and the co-doped reference marched cold on a uniform 300 us grid.

  Two rules keep those pins from becoming flakes, both learned in this repo. (i) Every pinned
  fixture is a march that CONVERGES to its fixed point; the RUNAWAY marches (the 175 K profile,
  the 1 uW signal, the co-doped 1 ms and 3 ms cold grids) carry no pinned value anywhere in the
  file, because an unstable iteration amplifies its own rounding -- they are asserted by
  INEQUALITY against `solve()`, the pattern `test_fiber_dynamics.py` already uses for them.
  (ii) The tolerance is split by whether the fixture touches LSODA (commit `533ece0`: a pin on a
  value that came through the steady solver's adaptive integrator is scipy-version dependent).
  The EDFA and the co-doped fixture are seeded EXPLICITLY, so their pins are pure march
  arithmetic and hold at `1e-9` -- seven decades above the BLAS-reduction-order noise the
  mixed-CPU and numpy-1.24 legs produce, and many decades below the smallest change this algebra
  could take and still be a different model. (The EDFA's value is bit-identical to the
  `solve()`-seeded one, because that march converges to its fixed point whatever it starts from.)
  The Yb booster has no uniform seed that is not itself a runaway, so it keeps the `solve()` seed
  and a `1e-5` pin.

Independently of the test file, the whole returned surface was diffed between the v0.11.2
checkout and this one over 12 fixtures (both classes; upconversion in both spellings, a 175 K
temperature profile, a 1 uW signal, log and uniform grids, one- and two-Yb-pool amplifiers,
`store_profiles=True`): **every array bit-identical, every meta entry equal.** The existing
transient suites (`test_fiber_dynamics`, `test_fiber_eryb_transient`, `test_transient_dt_floor`,
`test_transient_optics`, `test_fiber_eryb_physics`) pass unchanged, 66/66.

### (2) Where the flag stays True, the modes agree

| case | max over frames of `|dG|` | max relative `dASE` |
| --- | --- | --- |
| C-band EDFA, 10 uW in, 400 frames / 25 ms, 81 nodes | `3.8e-4 dB` | `1.9e-4` |
| co-doped reference, 201 log-spaced frames, 161 nodes | `1.1e-5 dB` | `3.2e-6` |
| 400 W Yb booster, 400 frames / 25 ms, 81 nodes | `1.7e-2 dB` on the PATH, `4e-15 dB` at the end | `3.4e-3` |

The first two are inside the specified `1e-3 dB` / `1%` over the whole trajectory and are what
the gate asserts. The Yb booster's transient is violent enough (inversion swinging 0.4 in 400
steps) that the two population trajectories separate by `1.7e-2 dB` mid-transient while both
endpoints land on the same fixed point; its ASE stays inside 1%.

### (3) The documented failure cases

`step closure` is `meta['max_step_power_residual']`: re-propagate at the populations each step
ENDED at, and take the steady solver's own interior residual against the powers that step USED.

| case | quasi-static vs `solve()` | self-consistent vs `solve()` | step closure QS -> SC |
| --- | --- | --- | --- |
| 400 W Yb, 175 K profile (docstring case 2) | `-50.14 dB` | `+0.074 dB` | `1.3e232` -> `2.6e-6` |
| 400 W Yb, 1 uW signal (docstring case 1) | `-67.38 dB` | `+0.017 dB` | `1.2e26` -> `2.9e-6` |
| co-doped reference, cold start, uniform 1 ms | `+13.93 dB` | `-5.3e-4 dB` | `4.9e2` -> `1.0e-6` |
| co-doped reference, cold start, uniform 3 ms | `-10.01 dB` | `-5.3e-4 dB` | `1.8e10` -> `9.8e-7` |
| co-doped reference, cold start, uniform 300 us behind a -20 dB idle (the study's own case) | `-5.3e-4 dB` but **flag False** | `-5.3e-4 dB`, flag True | `1.4e5` -> `3.1e-6` |

Every self-consistent march is finite throughout, raises no warning, and reports
`march_valid = True`. **Stated tolerance: the step closes to the inner tolerance `ase_tol`
(1e-6); measured worst case over these five, `3.1e-6`.**

The residue against `solve()` on the two Yb cases is the march's own z QUADRATURE, not the new
step. The march converges at SECOND ORDER in `dz` -- the order of the trapezoid rule
`_propagate_fixed` integrates with -- so its distance from any mesh-independent reference must
shrink as `dz^2` and cannot be a fixed defect of the step. Measured on the 1 uW fixture, with no
oracle at all (Richardson on the march alone):

| nodes | march `G` | successive difference |
| --- | --- | --- |
| 81 | `76.549123 dB` | |
| 161 | `76.544919 dB` | `4.204e-3` |
| 321 | `76.543870 dB` | `1.049e-3` (ratio **4.01**) |
| 641 | `76.543608 dB` | `2.62e-4` (ratio 4.00) |

Against `solve()` on the same meshes the gaps are `0.0165 / 0.0041 / 0.0010 / 0.00026 dB`, the
same factor of 4 per doubling, and the healthy amplifier's QUASI-STATIC gap (which nothing here
touched) falls identically at `0.0084 / 0.0021 / 0.0005 dB`. So the `< 1e-3 dB` target is met
outright on the co-doped cases at 161 nodes, and on the single-ion cases at 321; at 81 the mesh,
not the step, is the limit.

A CAVEAT WORTH RECORDING, because it cost a CI failure. The oracle's OWN fine-mesh path on this
fixture is scipy-build dependent: the CI py3.10 floor leg reproduces the march exactly at 81 and
161 nodes (`0.016485` / `0.004119`, four digits) and then reads a gap of `0.0116` at 321, i.e.
its `solve(321)` landed ~0.011 dB elsewhere on the hardest relaxation in the repo (400 W booster,
1 uW signal). The gate was therefore rewritten to read only the MARCH -- a test that consults a
build-dependent oracle to make a statement about the march is a flake, not a gate. Oracle
agreement is gated separately at 81 nodes, where every build agrees.

### (4) "auto" reproduces "self_consistent"

| case | `auto` - `self_consistent` (end gain) | switched steps | cost vs SC |
| --- | --- | --- | --- |
| co-doped, cold, uniform 1 ms | `-4.4e-8 dB` | 9 of 13 | 91% |
| co-doped, cold, uniform 3 ms | `0.0 dB` | 4 of 5 | 87% |
| co-doped, cold, uniform 300 us + idle | `-1.9e-8 dB` | 10 of 43 | 70% |
| 400 W Yb, 175 K | `0.0 dB` | 399 of 400 | 108% |
| 400 W Yb, 1 uW | `0.0 dB` | 399 of 400 | 98% |

`auto` saves nothing on the two Yb cases because those amplifiers are in the coupled regime at
EVERY step -- correctly, it switches all of them and pays a small decision overhead on top. The
saving is on marches that are only transiently coupled (70-91%) and, above all, on marches that
are never coupled: on the co-doped reference at 201 log-spaced frames it takes **0** switches and
**0** probes and returns the quasi-static arrays bit-for-bit.

### (5) Cost report -- reference co-doped fiber, ms per step

Seed `solve()` excluded (the seed is precomputed and passed in), single thread, same box.

| nodes | grid | `quasi_static` | `self_consistent` | `auto` |
| --- | --- | --- | --- | --- |
| 161 | 201 log-spaced frames, warm seed | **3.00** | **11.58** (3.9x) | **3.24** (1.08x, 0 switches) |
| 161 | 41 frames, uniform 300 us, cold seed | **3.60** | **52.30** (14.5x) | **40.88** (11 switches, 7 probes) |
| 801 | 201 log-spaced frames, warm seed | **4.88** | **16.00** (3.3x) | **6.19** (1.27x, 0 switches) |
| 801 | 41 frames, uniform 300 us, cold seed | **6.72** | **90.06** (13.4x) | **64.72** (11 switches, 7 probes) |

The multiplier is set by inner iterations per step: 2.0-2.2 on a warm march (the warm start is
doing its job), 11 on a cold start where the first steps have to find the operating point from
nothing. Note the cost ratio FALLS with mesh (3.9x -> 3.3x at 161 -> 801 nodes): the per-step
bookkeeping the two modes share grows with `n_nodes`, the inner iterations do not grow faster.

READ THE COUNTS, NOT THE SECONDS. The wall-clock column is load-sensitive on a shared box -- the
same table re-measured under contention reads 3.49 / 14.08 / 3.41 and 8.68 / 106.5 / 75.9 ms per
step, i.e. 15-40% slower everywhere in the same proportions. What reproduces EXACTLY between the
two runs is the work done: switches (0, 0, 11, 11), probes (0, 0, 7, 7) and inner iterations
(436, 400, 453, 266 / 268). Those are the numbers to regress against; the seconds are quoted so
the order of magnitude is on the record.

### (6) Composition with the other 2026-09-15 additions

Merged on top of v0.11.4, so two independent options now exist alongside this one. The step
composes with one of them and refuses the other by name:

* **Explicit Er 4I11/2 (`tau32_s`, v0.11.4) -- COMPOSES.** The inner iteration calls the march's
  own `advance()` closure, which builds the exact Jacobian for whatever reservoir tuple the
  amplifier carries and integrates it through the size-parametrized `_phi1_dt_nxn` kernel, so the
  state size is not a parameter of the self-consistent step at all. With `tau32_s` set that tuple
  is the FOUR-vector `(f2, f3, b2c, b2nc)`, and the 4I11/2 population constraint `f2 + f3 <= 1`
  is enforced INSIDE the iteration (in `advance()`, against the already-clipped `f2`) rather than
  outside it, so every inner iterate is a physically admissible state. Measured on the reference
  co-doped fiber with `tau32_s = 7 us`, 161 nodes, cold start, 21 frames at 300 us:

  | mode | end gain | step closure | self-consistent steps |
  | --- | --- | --- | --- |
  | `quasi_static` | 13.5224248741 dB | `1.0` (open) | 0 of 20 |
  | `self_consistent` | 13.5223425238 dB | **`1.0e-6`** | 20 of 20 |
  | `auto` | 13.5223242727 dB | -- | 11 of 20 |
  | `solve()` (oracle) | 13.5213544100 dB | -- | -- |

  `auto` reproduces `self_consistent` to `1.8e-5 dB`, and the self-consistent march lands on
  `solve()` to `9.9e-4 dB` -- which is the SAME `9.9e-4 dB` mesh residue the three-state march
  leaves on the identical fixture with `tau32_s=None` (13.5172341753 vs 13.5162400824). The
  fourth reservoir therefore costs the inner solve nothing in accuracy: the residue is still
  pure z-quadrature. `ase_mode="quasi_static"` on a four-state amplifier is byte-identical to the
  default call. Gate: `test_every_mode_composes_with_the_explicit_4i11_2_level`.

* **Two-population Yb (`yb_coupled_fraction`, v0.11.0) and `yb_sigma_e_scale` (v0.11.4) --
  COMPOSE.** Both are already the three-/four-state path above; `yb_sigma_e_scale` is a
  cross-section, invisible to the march's structure.

* **Axial temperature profile (`RateTemperatureLaw` / `YbStarkThermal`, v0.11.3) -- REFUSED BY
  NAME.** `simulate_transient_eryb` has always refused a profiled amplifier, and it now names
  which of the two thermal objects is attached and states explicitly that `ase_mode` does not
  rescue it: the refusal is about z-dependent RATE COEFFICIENTS, which the transient's z-local
  balances do not carry, and is orthogonal to how the powers are solved. Both thermal objects
  WITHOUT a profile are pure constants and every mode reproduces the unprofiled amplifier
  bit-for-bit. Gate: the last test in `tests/test_march_self_consistent_ase.py`.

---

## 6. Guidance for the downstream burst-PAM study

**Neither co-doped march site is in the unstable regime today.** On the study's own shapes --
`lib/sat56_study.py::droop_transient` (an idle preamble, a 20 dB drive step at t = 0, then
log-spaced burst frames, 801 nodes) and `lib/fiber_burst_pam.py::run_two_timescale` (a uniform
envelope grid capped at 200 ns, four orders below the Er lifetime) -- `meta['quasi_static_valid']`
stays True in every mode and all three modes land on the SAME end gain (13.5178 dB, four
decimals). So the quasi-static numbers the study has are trustworthy as they stand.

**But `auto` is NOT a no-op on the droop shape, and this is the number to know.** Measured on the
droop shape at 801 nodes:

| burst frames | `auto` switches | max path \|auto - QS\| | max path \|SC - QS\| | end gain, all modes |
| --- | --- | --- | --- | --- |
| 100 | 75 of 104 | -- | `0.1147 dB` | 13.5178 |
| 200 | 135 of 204 (steps 0-134, t = -2 us .. +39 us) | `0.0270 dB` | `0.0563 dB` | 13.5178 |
| 400 | 257 of 404 (steps 0-256, t = -2 us .. +28 us) | `0.0098 dB` | `0.0280 dB` | 13.5178 |
| 800 | 502 of 804 | -- | `0.0140 dB` | 13.5178 |

Two things to read off it. (i) `auto` switches exactly the frames just after the drive step,
where the ASE is being re-established -- the right place. (ii) The path difference between modes
HALVES with every doubling of frame density: it is the O(dt) path error of the integrator, not
an ASE effect, and both modes converge to the same path. The study's droop metric is read off
that path, so switching modes would move it by `0.01-0.03 dB` at the study's current frame
density.

**Recommendation.** Switch the co-doped marches to `ase_mode="auto"`, but as a re-run, not a
retrofit: it is the safer default (it removes the whole class of silent failure that produced
the flag in the first place, at 1.3-1.8x the cost on these shapes), and where it differs it is
the better-conditioned side of an O(dt) error the study should be quantifying anyway. Do NOT
paste `auto` numbers next to published `quasi_static` ones. If byte-identical continuity matters
more, keep `"quasi_static"` and keep reading `meta['quasi_static_valid']` exactly as the study
does now -- that flag has not changed meaning.

Two smaller points:

* `meta['march_valid']` is the flag to read in `auto`; `meta['quasi_static_valid']` keeps its old
  meaning (did the frozen step stay in its regime) and will still go False on a switched step --
  which is now information, not a reason to distrust the run.
* Do NOT make `"self_consistent"` the study's default: on a march at its operating point it is
  3-4x the cost for numbers that agree to `1.1e-5 dB`, and on the droop shape it moves the path
  by the full O(dt) difference above rather than only the part `auto` judges necessary.

---

## 7. Limits

* The reported powers are the frozen-population propagation at the reported populations in every
  mode. That is the exact quasi-static pair at each instant, but it means an ASE-coupled march
  still reports an instantaneous ASE, not a step-averaged one. The step-averaged powers are the
  ones the update uses and are not exposed.
* `ase_mode` is first-order on the PATH exactly as the default march is (the ETD1 / exponential
  integrator is unchanged); what it fixes is stability and the fixed point, not the order.
* The z discretization is unchanged -- the inner solve reuses `_propagate_fixed`, i.e. the
  march's trapezoid quadrature, not `solve()`'s adaptive LSODA. Agreement with `solve()` is
  therefore mesh-limited, as the table in gate (3) shows. Refine `n_nodes`, not `ase_tol`, to
  close the last hundredth of a dB.
* `auto`'s stickiness window (8 steps) and switch threshold (1% of launched power) are measured
  defaults, not derived ones. A march that alternates in and out of the coupled regime on a
  cadence faster than 8 steps will simply stay self-consistent, which is the safe direction.
* A co-doped amplifier carrying an axial temperature profile is still refused by the march
  (unchanged from 2026-09-13); the new modes do not lift that.
* `RamanStokes` coupling is still refused (unchanged): the inner solve inherits the same
  per-channel LINEAR gain assumption the propagator makes.
