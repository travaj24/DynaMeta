# Radially resolved Er:Yb co-doped amplifier -- build + audit note (2026-09-15)

Branch `feat/eryb-transverse`, worktree `DynaMeta-eryb4`, branched from `main` at 21dff60
(v0.11.2) and MERGED with `main` at 1bef664 (v0.11.3, PR #29: W_mig calibration, the co-doped
temperature laws and `eryb_fit`). Version 0.11.5 (0.11.4 is reserved for PR #30).
New module `dynameta/optics/fiber_amp/transverse_eryb.py` (`ResolvedErYbAmplifier`); new gate file
`tests/test_fiber_transverse_eryb.py` (24 gates); model spec section 17 + 17a (16 is PR #29's).
`transverse.py` was touched only to EXTRACT its geometry kernel into module-level functions that
both resolved solvers now call (byte-identity re-verified, section 2); `eryb.py` was not touched
at all.

---

## 1. What the feature is, and why the co-dope is where the mean-field closure binds

`steady_state.py` and `eryb.py` solve the Giles-Desurvire model in its MEAN-FIELD reduction:
every channel's intensity is replaced by its area average over the dopant,
`<I_k> = Gamma_k P_k / A_dope`, so each population is one number per z. `transverse.py`
(`ResolvedFiberAmplifier`, the 2026-08 campaign) undoes that for a single ion and finds transverse
spatial hole burning. This note undoes it for the Er:Yb co-dope.

The reason it matters more here is not that the erbium burns a deeper hole. It is that the
sensitization step is a LOCAL, BIMOLECULAR event:

```
R_tr(r) = k_tr n_Yb2(r) n_Er1(r)                                                (E1.1)
```

An excited Yb donor must meet a GROUND Er acceptor at the same point. The mean-field model
evaluates `k_tr <n_Yb2> <n_Er1>`: a product of averages, which differs from the average of the
product by the covariance

```
<n_Yb2 n_Er1> - <n_Yb2> <n_Er1>                                                 (E1.2)
```

In a cladding-pumped EYDFA the two factors are not the same shape. The pump is flat across the
core, so the ytterbium it excites is nearly flat; the C-band signal is a confined LP01 that burns
the erbium inversion hardest on axis. The transfer integral sees that mismatch. Whether it is
LARGE is a measured question, answered in section 5 -- and the honest answer on the study's own
operating points is "small, and smaller than a confound that is easy to mistake for it".

### 1.1 The equations actually solved

Per quadrature node, with `i_k(r)` the channel's normalized transverse intensity profile
(`INT i_k dA = 1`) and `I_k(r) = P_k i_k(r)`:

```
R_{a/e}_Er(r) = SUM_k sigma_{a/e}_Er,k P_k i_k(r) / (h nu_k)                    (E1.3)
R_{a/e}_Yb(r) = SUM_k sigma_{a/e}_Yb,k P_k i_k(r) / (h nu_k)

Er:   R_a_Er (1-f2) - R_e_Er f2 - f2/tau_Er
      + k_tr b2c f n_Yb(r) (1-f2) phi - C_up n_Er(r) f2^2                 = 0   (E1.4)
Ybc:  R_a_Yb (1-b2c) - R_e_Yb b2c - b2c/tau_Yb - k_tr b2c n_Er(r) (1-f2) phi
      - K2 b2c n_Er(r) f2 - W_mig (1-f) (b2c - b2nc)                      = 0
Ybnc: R_a_Yb (1-b2nc) - R_e_Yb b2nc - b2nc/tau_Yb + W_mig f (b2c - b2nc)  = 0
```

The densities are INSIDE the balance. That is the whole content of (E1.1): the transfer-in
coefficient is `k_tr f n_Yb(r)` and the transfer-out coefficient is `k_tr n_Er(r)`, both at the
node, so a ring where the ytterbium is excited but the erbium is already inverted transfers
nothing, and a ring where the erbium is hungry but the ytterbium is dark transfers nothing either.
Propagation is (E1.5) of the module docstring: the same ODE with the four population integrals
`Je_Er,k = INT n_Er(r) f2(r) i_k dA` and friends in place of `Gamma_k n f2`.

`eryb.py`'s reduction survives node by node: at fixed `f2` the two Yb balances are linear in
`(b2c, b2nc)`, the migration cross terms cancel out of the 2x2 determinant
(`det = Dc D + W (f Dc + (1-f) D) > 0` always), and one bracketed scalar equation per node
remains. It is solved by a VECTORIZED safeguarded Newton/bisection stepped in lockstep across
nodes. The relaxation around it -- alternating frozen-direction forward/backward IVP sweeps, the
`relax` ladder, `_relaxation_residuals` -- is imported from `steady_state`, not re-written.

### 1.2 Separate Er and Yb radial profiles

`er_profile` / `yb_profile` each accept `None` (top hat at `fiber.b_dope_m`, the scalar model's
geometry and the DEFAULT), a radius in metres, or `RadialDopant(radius_m, shape)` for a graded
density `n(r) = N shape(r)`. The radius is a quadrature panel edge, so a density step is resolved
spectrally rather than smeared across a panel. The convention is deliberately NOT
number-conserving -- a wider region at the same density holds more ions, which is what
`FiberSpec.n_t_m3` + `dopant_radius_m` already means everywhere else in this package.

This is the only place in the repo where the confined-Er / wide-Yb trade can be computed at all,
because it is exactly a statement about (E1.1).

### 1.3 API

```python
from dynameta.optics.fiber_amp import (ResolvedErYbAmplifier, RadialDopant,
                                       eryb_mean_field_equivalent)

amp = ResolvedErYbAmplifier(er_ion, yb_ion, fiber, pumps, signals, ase,
                            # every ErYbAmplifier keyword, unchanged:
                            n_yb_m3=..., k_tr_m3_s=..., k_back_m3_s=..., a32_per_s=...,
                            yb_ase=..., upconversion_C_up=..., yb_coupled_fraction=...,
                            k_tr2_m3_s=..., yb_migration_rate_per_s=..., concentration=...,
                            # resolved extras, all defaulting to the scalar model's geometry:
                            er_profile=None, yb_profile=None,        # or a radius / RadialDopant
                            signal_modes=None,                       # None / LPMode / "flat"
                            uniform_illumination=False,
                            n_quad=24, n_azimuthal=32, r_max_m=None)

res = amp.solve(n_nodes=201, relax="auto")     # a steady_state.SteadyStateResult
res.nbar2_z, res.meta["beta_yb_z"]             # population-weighted scalar reductions
res.meta["f2_rz"], res.meta["b2c_rz"], res.meta["b2nc_rz"], res.meta["bbar_rz"]   # (M, N) rings
res.meta["grid"], res.meta["doped_mask"], res.meta["n_er_r_m3"], res.meta["n_yb_r_m3"]
amp.closure_residual(res)                      # ring-summed energy closure
amp.energy_terms_from_result(res)              # incl. 'transfer_covariance'
amp.local_transfer_rate_per_m3(powers_W)       # (N,) k_tr n_Yb2(r) n_Er1(r)
amp.state_bytes(n_nodes)                       # the RAM watchdog's own number
eryb_mean_field_equivalent(amp)                # the scalar twin, same plan and opt-ins
```

`transfer_efficiency` / `yb_parasitic_gain_dB` / `channel_plan` / `grid` / `profiles` /
`gamma_quad` / `er_density_r_m3` / `yb_density_r_m3` / `local_populations` / `local_gain_per_m` /
`dP_dz` round out the surface, and `with_signals` / `with_pumps` / `without_ase` implement the
package's re-seed protocol so `metrics.gain_spectrum`, `metrics.slope_efficiency` and
`chain.AmplifierChain` drive this class as they do the other two (gated). `_heat_profile_W_per_m`
is the thermal layer's Q(z) hook -- the same ring-summed dissipation, whose integral agrees with
`thermal.heat_load_per_m`'s independent flux-gradient form to 1.6e-6 relative (gated); the thermal
FEEDBACK loop still cannot run, because it re-solves through `set_temperature_profile`.

---

## 2. What was touched outside the new module, and the byte-identity evidence

`transverse.py` carried five geometry primitives as PRIVATE METHODS of `ResolvedFiberAmplifier`:
the fundamental-mode intensity shape, the synthetic mean-field ("flat") profile, the flat
cladding-pump profile, the default outer quadrature radius, and the panel breakpoints -- plus the
profile builder and the signal-mode validator. The co-doped solver needs every one of them
unchanged. They were EXTRACTED to module level (`fundamental_psi2`, `flat_intensity_profile`,
`cladding_intensity_profile`, `default_r_max_m`, `quadrature_breakpoints`,
`build_normalized_profiles`, `check_signal_modes`) and the class methods now delegate. The
alternative -- a second copy in the new module -- is the defect this repo has already paid for
once (steady_state audit X-3, the duplicated relaxation interpolator).

Two additions to `transverse.py` beyond the extraction: `build_normalized_profiles` takes a
`flat_all` flag (the uniform-illumination limit), and `quadrature_breakpoints` takes `extra_m`
(a co-doped fiber may confine its two ions differently). Both default to the previous behaviour.
The co-doping refusal message now names the new class instead of calling it unsupported.

EVIDENCE. `ResolvedFiberAmplifier` was solved before and after the refactor on five
configurations -- cladding-pumped Yb with an ASE band; core-pumped Er with a backward pump; an LMA
fiber with an explicit LP11 second signal; the same with a `"flat"` signal; a confined-dopant Er
fiber with forward and backward pumps and an ASE band -- comparing `power_W`, `nbar2_z`,
`signal_gain_dB`, `gain_per_m`, `gamma_quad`, `nbar2_rz`, `nt_r_m3`, `nbar2_mode_z`, the grid
nodes and weights, the iteration count and `n_grid`. Every array was BIT-IDENTICAL
(`np.array_equal`, worst absolute difference 0.0), and `tests/test_fiber_transverse.py` (35 gates)
passes unchanged.

`tests/test_fiber_amp.py::test_package_facade_is_exhaustive` pins the fiber_amp submodule COUNT
(a drift guard, audit W5-8); a new module makes it 23, so that one number is bumped. The
substantive half of that gate -- every name in every submodule's `__all__` re-exported from the
package facade -- passed unchanged, which is why the seven extracted kernel functions and the
three new names are in the package `__all__`.

`eryb.py` was not modified. `ResolvedErYbAmplifier` CONSTRUCTS an `ErYbAmplifier` internally and
calls its `_plan()` for the channel table and its validated constructor state for the densities,
rate constants and the active/dark erbium split -- so the two classes cannot drift in the channel
plan, and `eryb_mean_field_equivalent()` hands back that very object rather than a re-built clone
that could drop a keyword.

---

## 3. The gates

`tests/test_fiber_transverse_eryb.py`, 24 gates (19 test functions, 4 parametrized), 20 s.

| # | Gate | Bar | Measured |
|---|------|-----|----------|
| 1 | uniform illumination == `ErYbAmplifier`, one Yb pool | 1e-3 dB | 3.3e-7 dB |
| 1 | the same, two pools + K2 + migration | 1e-3 dB | passes at the same order |
| 1 | `eta_tr` under uniform illumination | 1e-6 | 4.8e-10 |
| 1 | the same with `FiberSpec.overlap_override` set | 1e-3 dB | passes |
| 2 | `N_Yb -> 0`, `k_tr = 0` == `ResolvedFiberAmplifier` | 1e-3 dB | 2.2e-8 dB |
| 2 | ... while that fixture's own TSHB is not zero | > 1e-2 dB | 0.020 dB |
| 3 | ring-summed energy closure, one and two pools | 1e-5 | < 1e-14 |
| 3b | `wall_plug_efficiency` residual is FINITE | finite | -1.6e-5 W (study point A) |
| 4 | RAM: `state_bytes` formula + refusal above 2 GiB | exact | exact |
| 5 | resolved - scalar at 1 nW (low-NA fixture) | < 5e-3 dB | -0.0006 dB |
| 5 | resolved - scalar at 10 mW, SIGN | < -0.05 dB | -0.138 dB |
| - | node kernel vs `_solve_fbb` / `_solve_fb` | 1e-13 | passes, 4 configs x 5 rate decades |
| - | vectorized bracket on a quadratic residual with roots at both ends | 1e-12 | passes |
| - | transfer covariance nonzero saturated / zero flat | 1e-3 / 1e-12 | passes |
| - | quadrature refinement (n_quad 24 -> 48) | 1e-6 dB | passes |

### 3.1 What gate 1 discriminates

Forcing every core channel to its mean-field profile `Gamma_k/A_dope` makes the ring solver
compute the scalar model by a completely different route: `K x N` profile quadratures and a
per-node root find instead of one area-averaged intensity and one scalar root. It fails if the
channel table, the overlap quadrature, the spontaneous prefactor, the transfer coefficients, the
dark-erbium absorption or the population algebra differ anywhere from `eryb.py`'s. It is NOT a
tautology, because nothing in the uniform path reuses `eryb.py`'s RHS.

Gate 2 is the complementary half: a solver that had silently area-averaged its intensities would
pass gate 1 and FAIL gate 2, because gate 2's fixture has 0.020 dB of genuine hole burning.

### 3.2 The node kernel's one deliberate departure from the scalar iteration

`ErYbAmplifier._bracketed_newton` clamps an out-of-bracket Newton step to the bisection midpoint
BEFORE testing convergence. Once Newton has landed on the root, the next pass sets a bracket end
TO that root, so the following step -- of size ~1e-17, below an ulp of `f` -- is no longer
strictly inside `(lo, hi)`, is discarded, and the routine BISECTS about 43 more times to
re-derive an answer it already had. Harmless for one scalar root; ruinous vectorized, where one
straggling node holds the whole vector. MEASURED on the gate-2 fixture: 44.8 residual evaluations
per call. Accepting a step already smaller than `tol` brings it to ~3 and cuts the solve from
31.2 s to 2.5 s. The fixed point is identical and the residual is if anything smaller (Newton
leaves `|H|` at round-off; the bisection path only guarantees the 1e-13 bracket). The two
routines are gated against each other to 1e-13.

This is a latent inefficiency in `eryb.py` too, not a defect in its answers. It is NOT fixed there
(byte-identity of the shipped scalar path, and another agent is editing that file).

### 3.3 RAM

The ring-resolved state is four `(z-node x ring)` population matrices plus `(channel x z-node)`
powers and `(channel x ring)` profiles. `state_bytes(n_nodes)` reports it exactly; `solve`
refuses above `dynamics._STORE_PROFILES_MAX_BYTES` (2 GiB -- the package's EXISTING bar, imported
rather than re-declared) and the message names the ring x z-node x channel shape, because "too
big" without the shape does not tell a caller which axis to coarsen. The study's 8 m point is
**144 rings x 121 z-nodes x 12 channels = 0.56 MiB**; the 0.3 m core-pumped point is
**120 rings x 121 z-nodes x 12 channels = 0.47 MiB**. Four orders of magnitude under the bar.

---

## 4. Scope refusals, and why each is a refusal rather than an answer

* `FiberSpec.overlap_override` unless `uniform_illumination=True`. An override REPLACES the
  overlap this solver computes. With uniform illumination the solver is deliberately not resolving
  anything and the override IS the mean-field Gamma, so it is honoured there and gated.
* `set_temperature_profile`. A z-only McCumber scaling is not meaningful once the inversion is
  resolved; it would have to become `T(r, z)`. The second ion sharpens rather than weakens the
  argument -- the transfer defect is the dominant heat source in an EYDFA, so T is MOST radially
  structured exactly here.
* `ConcentrationModel.pd_loss_per_m != 0`. `eryb.py` subtracts the Yb photodarkening equilibrium
  gray loss from every channel's gain with NO overlap factor -- a channel-uniform background.
  Resolved, the gray loss is a per-node property of the doped glass and a channel picks up
  `INT pd(bbar(r)) i_k(r) dA`, i.e. the confinement factor. Those are two different models and
  answering the second under the name of the first would be silent.
* `rate_temperature` (`RateTemperatureLaw`) and `yb_stark_thermal` (`YbStarkThermal`), the two
  temperature opt-ins `eryb.py` gained in PR #29 (v0.11.3). Both act ONLY through the axial T(z)
  profile this class already refuses -- the Arrhenius law scales `k_tr` / `K2` / `W_mig` at T(z),
  the Stark law depopulates the Yb lower manifold at T(z) -- so they are refused for the same
  reason AND because with no profile set they would be carried completely inert, which is the
  worse failure mode. They are ACCEPTED as constructor keywords so that a port from
  `ErYbAmplifier` fails with a physics message instead of a `TypeError`, and an all-zero
  `RateTemperatureLaw` (the exact identity, which `eryb.py` itself collapses to `None`) is
  accepted and gated.
* Nonzero `sigma_esa`. `eryb.py` drops the ESA term silently; this class refuses rather than
  inherit that. (Neither `erbium()` host nor either ytterbium factory carries ESA by default, so
  no shipped configuration is affected.)

`C_up` and pair-induced quenching are SUPPORTED, not refused: both are strictly local (a quadratic
in the node's own `f2`; an unbleachable absorption at the node's own dark density) and both reduce
to `eryb.py` exactly under uniform illumination -- gated.

---

## 5. The measured transverse effect on the study's two points

Reference co-doped fiber (`calibration.er_yb_phosphosilicate_reference`, Rybaltovsky-class):
`a = 2.0 um`, NA 0.20, `N_Er = 4e25`, `N_Yb = 4e26`, 125 um cladding, background loss 30 dB/km,
`C_up = 1.1e-24`, 1.6% Delevaque pairs. Study parameters: `f = 0.9`, coupled-pool
`k_tr = 1.11e-21`, `K2 = 2e-22`, `k_back = 0`, `W_mig = 0`. Seed 2.1 mW at 1550 nm, pump 176 mW at
976 nm, 4 Er ASE bins (1530-1565 nm) + 3 Yb ASE bins (1000-1080 nm) both directions, 121 z-nodes,
`relax="auto"`. Scalar = `ErYbAmplifier`, resolved = `ResolvedErYbAmplifier`, identical plan.

### Point A -- cladding-pumped, 8 m (the study's reference operating point)

| quantity | scalar | resolved | delta |
|---|---|---|---|
| signal gain | 11.8347 dB | 11.8004 dB | **-0.0343 dB** |
| signal out | 32.040 mW | 31.788 mW | -0.79% |
| wall-plug (45% diode, no coupling loss) | 7.6551% | 7.5906% | **-0.0644 pp, -0.84% relative** |
| `eta_tr` | 0.70324 | 0.70089 | -0.0024 (-0.33%) |
| 1-um parasitic gain | -190.87 dB | -195.14 dB | -4.27 dB on a dead quantity |
| residual pump | 5.977 mW | 5.977 mW | +0.01% |
| closure | -- | 1.3e-15 | -- |

The sign is the documented one for a uniform pump: the resolved gain is LOWER, because a real mode
burns its hole where it is brightest while the area average never sees the peak. The magnitude is
small -- 0.034 dB over 8 m -- for a structural reason worth stating: `V(1550) = 1.62` on this
fiber, so `Gamma = 0.546` and the signal is weakly confined; the pump is a flat cladding field
with nothing to burn; and at 2.1 mW in / 32 mW out the C band is only lightly saturated. The
transfer covariance at mid-fiber is `-5.6e-6` of the local transfer rate. **The transverse
correction does not change any engineering conclusion at point A.**

MESH CONVERGENCE (the delta is small, so it is worth proving it is not a discretization
artifact): at `n_nodes = 201` instead of 121 the same run gives 11.834825 -> 11.800526 dB, a delta
of `-0.0342991 dB` against `-0.0342976 dB` at 121 -- four significant figures unchanged, and the
closure holds at 1.1e-15. The correction is physical.

The 1-um parasitic gain is -191 dB (scalar) / -195 dB (resolved): the ytterbium is essentially
fully de-excited by an erbium acceptor pool 10x more abundant per unit of stored energy than the
Yb inversion it drains, so there is no 1-um risk at this point in either model, and the 4.3 dB
delta is a difference between two numbers that are both zero.

### Point B -- core-pumped, 0.3 m, 976 nm (the bottleneck point)

| quantity | scalar | resolved | delta |
|---|---|---|---|
| signal gain | 0.7230 dB | 0.7982 dB | **+0.0753 dB** |
| **1-um parasitic gain** | **3.1741 dB** | **3.7398 dB** | **+0.5658 dB (+17.8%)** |
| wall-plug | 0.09725% | 0.10834% | +11.4% relative |
| `eta_tr` | 0.002852 | 0.002816 | -1.3% |
| `f2` at the output end | 0.3438 | 0.3421 | -0.5% |
| closure | -- | 4.4e-15 | -- |

The gain sign FLIPS relative to point A, which is the documented core-pump behaviour: a
core-guided pump bleaches its own absorption hardest on axis, so its resolved absorption
coefficient is smaller, more pump survives downstream, and the end-to-end gain exceeds the
mean-field one.

**Does the radial inversion profile change the 1-um parasitic gain materially? NO -- and the
+0.57 dB headline is mostly something else.** Decomposed by re-evaluating the 1030 nm gain
integral with the resolved populations but mean-field weighting:

```
scalar                                     3.1741 dB
resolved populations, mean-field weighting 3.1936 dB   <- radial inversion profile: +0.0196 dB
resolved (module, exact-LP01 weighting)    3.7398 dB   <- 1030 nm mode weighting:   +0.5462 dB
```

`V(1030) = 2.440`, just above the 2.405 cutoff, so the resolved solver takes the EXACT LP01 field
(quadrature overlap 0.8332) while the mean-field one keeps the Marcuse Gaussian (0.8152) -- 2.2%
of overlap, compounded over 0.3 m of strong Yb gain. That is a mode-shape correction, not hole
burning. The radial inversion profile contributes **+0.020 dB, i.e. 0.6%** of the 3.17 dB
parasitic gain, which does not move a parasitic-lasing margin.

The same confound decomposes point B's gain delta (`signal_modes=["flat"]` turns the signal's own
hole burning off while leaving the core pump resolved):

```
uniform illumination (== scalar)  0.72297 dB
signal flat, pump resolved        0.77844 dB   <- pump profile switch (V(976) = 2.575): +0.0555 dB
fully resolved                    0.79823 dB   <- signal hole burning:                  +0.0198 dB
```

So of the +0.0753 dB, 74% is the exact-LP-vs-Marcuse pump overlap (0.8527 against 0.8361) and 26%
is transverse hole burning. Both are real physics the mean-field model gets wrong; only the second
is TSHB, and reporting the sum as TSHB would be wrong. Gate
`test_exact_lp_profile_switch_is_a_separate_effect_from_hole_burning` pins the separation so this
cannot be lost.

### 5.1 Bottom line for the study

* **Point A (8 m, cladding-pumped): the transverse physics is a 0.03 dB / 0.8%-relative
  correction.** Use the scalar solver; re-check with the resolved one only if the seed is raised
  far enough to saturate the C band hard, or if the core is made higher-NA (`V > 2.405` at 1550
  turns the profile switch on as well).
* **Point B (0.3 m, core-pumped): the RADIAL INVERSION PROFILE is not the story.** It moves the
  1-um parasitic gain by 0.020 dB (0.6%) and the C-band gain by 0.020 dB. What DOES move both is
  the exact-LP mode profile above cutoff, +0.55 dB on the parasitic gain and +0.056 dB on the
  signal gain. That is an argument for using the resolved solver (or `lma.solve_lp_modes` +
  `overlap_override`) on any core-pumped point of this fiber, but it is an argument about mode
  shapes, not about hole burning.
* The wall-plug deltas are -0.84% relative (A) and +11.4% relative (B); B's is large only because
  its absolute wall-plug is 0.1%, i.e. the fiber is far too short to absorb its pump.

---

## 6. Cost, and what is not in v1

Solve time on the study points: 1.2 s scalar / 11.2 s resolved at 8 m; 0.4 s / 6.1 s at 0.3 m
(121 z-nodes, 12 channels, 144 / 120 rings). About 10-15x the scalar, which is the cost of a
per-node root find inside every RHS evaluation. The gate file runs in 16 s.

Not in v1, in rough order of how much they would change an answer:
1. **`T(r, z)`.** The transfer defect is the dominant EYDFA heat source and it is radially
   structured; a resolved thermal loop is the natural next feature and is currently refused.
2. **Photodarkening, resolved.** Needs a decision about whether the scalar's channel-uniform gray
   loss or the overlap-weighted one is the intended model; refused rather than guessed.
3. **ESA.** Transcribed in (E1.5) but not wired, and silently dropped by the scalar class.
4. **Ion diffusion / migration IN SPACE.** `W_mig` here is the Dong two-pool exchange, which is a
   population-space transfer, not a radial one. Nothing transports excitation laterally in this
   model -- correct for a glass host, and the reason the balance is strictly pointwise.
5. **A transient ring march.** `dynamics.simulate_transient_eryb` remains mean-field.
6. The degenerate cos/sin LP pair still cannot be SPELLED (inherited from `transverse.py`:
   `LPMode` carries no orientation field), and is refused rather than silently modelled as
   cos + cos.
