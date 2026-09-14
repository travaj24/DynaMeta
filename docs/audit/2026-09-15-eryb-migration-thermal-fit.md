# Er:Yb migration calibration, co-doped temperature laws, and the (f, k_tr) device fit (2026-09-15)

The 2026-09-14 two-population build shipped three things it could not close. Its limit 1 was that
`yb_migration_rate_per_s` had NO SOURCE -- the one parameter in the module with no measurement
behind it, and the one introduced precisely to reconcile the two mutually inconsistent transfer
coefficients the literature extracts from the same data. Its limit 3 was that `k_tr`, `K2` and
`W_mig` are isothermal even under a temperature profile, and that the 976 nm pump line's own
thermal behaviour was absent. Its limit 8 was that a two-population fit is under-determined by one
observable, with no routine that fits two.

This branch closes all three. The migration rate turns out to be READABLE OFF a published decay in
closed form rather than fitted; the temperature laws are calibrated to the one measured co-doped
temperature series that exists; and the device fit is a routine that reports its own degeneracy.

Everything is opt-in. With every new parameter at its default `ErYbAmplifier` reproduces v0.11.2
BITWISE -- including an amplifier already running under a temperature profile.

Branch `feat/eryb-migration-thermal-fit`. Version `0.11.2 -> 0.11.3`.

---

## 1. What was built

| Where | What |
| --- | --- |
| `eryb.py` | `RateTemperatureLaw` (Arrhenius scaling of k_tr / K2 / W_mig), `YbStarkThermal` (976 nm zero-line Boltzmann depopulation), the constants `RATE_ARRHENIUS_CHENG_2022`, `RATE_ARRHENIUS_30PCT_300_480K`, `YB_STARK_976_CANAT_DUSSARDIER`; the two new constructor keywords `rate_temperature` / `yb_stark_thermal` and the `_t_rates` predicate that routes them |
| `eryb.py` | `_mcc_matrices` becomes the 4-slot TEMPERATURE BUNDLE `(mcc_er, mcc_yb, stark, rate_scale)`; the per-z scale is threaded through `_dP`, `_fb_profile`, `_rates_profile`, `_solve_fb`, `_solve_fbb`, `_fbb`, `_fb_rhs3`, `_fb_jacobian3`, `_transfer_efficiency`, `energy_terms` and `_rate_balance_dissipation_W`, every one of them branching on `is None` rather than multiplying by ones |
| `eryb_fit.py` (NEW) | `yb_two_pool_decay` / `YbDecayModes` (the closed-form free decay), `yb_two_pool_from_decay` (the EXACT 3-number inversion), `YbDecayAnchor` / `YbLifetimeAnchor` + the shipped anchors `CHENG_2022_DECAY` / `JEONG_2007_DECAY` / `LAROCHE_2006_LIFETIMES`, `eryb_calibrate_pools` + `PoolCalibration`, `eryb_fit_to_device` + `DeviceTarget` / `DeviceFit` / `device_observables`, and the constants `W_MIG_PHOSPHOSILICATE_PER_S` / `_RANGE_` / `_CEILING_` |
| `tests/test_fiber_eryb_migration_thermal.py` | 19 gates |
| `validation/eryb_migration_thermal_fit.py` | the four-section report (SELF-CONTAINED: no scratchpad folder needed) |
| `docs/fiber_amp_model_spec.md` | section 16 |

### API -- exact call forms

```python
from dynameta.optics.fiber_amp import (
    ErYbAmplifier, RateTemperatureLaw, YbStarkThermal,
    RATE_ARRHENIUS_CHENG_2022, RATE_ARRHENIUS_30PCT_300_480K, YB_STARK_976_CANAT_DUSSARDIER,
    yb_two_pool_decay, yb_two_pool_from_decay, eryb_calibrate_pools, eryb_fit_to_device,
    DeviceTarget, device_observables, YbDecayAnchor, YbLifetimeAnchor,
    CHENG_2022_DECAY, JEONG_2007_DECAY, LAROCHE_2006_LIFETIMES,
    W_MIG_PHOSPHOSILICATE_PER_S, W_MIG_PHOSPHOSILICATE_RANGE_PER_S,
    W_MIG_PHOSPHOSILICATE_CEILING_PER_S,
    F_COUPLED_LITERATURE_BOUNDS, K_TR_FAST_LITERATURE_BOUNDS_M3_S)

# --- 1. the migration rate, read off a published decay in closed form (no fitting) ---------
inv = yb_two_pool_from_decay(tau_fast_s=8.20e-6, tau_slow_s=1333.76e-6, amp_fast=0.864,
                             tau_yb_s=1793.46e-6, n_er_m3=3.96e25)
# -> {'coupled_fraction': 0.8644, 'k_tr_m3_s': 3.065e-21, 'migration_per_s': 222.4,
#     'transfer_yield': 0.8949, 'transfer_rate_per_s': 1.2136e5, 'initial_slope_per_s': 1.0547e5}

modes = yb_two_pool_decay(tau_yb_s=1793.46e-6, n_er_m3=3.96e25, k_tr_m3_s=3.065e-21,
                          coupled_fraction=0.8644, migration_per_s=222.4)
modes.tau_fast_s, modes.tau_slow_s, modes.amp_fast      # the measured triple, round-tripped
modes.mean_lifetime_s          # the amplitude-weighted lifetime a quenched-lifetime paper quotes
modes.transfer_yield           # the ENSEMBLE observable a one-pool model has to match
modes.intensity(t), modes.fraction_relaxed_by(t)

cal = eryb_calibrate_pools(decays=[CHENG_2022_DECAY, JEONG_2007_DECAY],
                           lifetimes=list(LAROCHE_2006_LIFETIMES),
                           f_bounds=F_COUPLED_LITERATURE_BOUNDS,     # (0.10, 0.99)
                           k_bounds=K_TR_FAST_LITERATURE_BOUNDS_M3_S,  # (2e-22, 3e-20)
                           w_bounds=(1.0, 1e8), decay_weight=1.0, lifetime_weight=1.0,
                           w_profile_points=25, profile_cost_factor=2.0)
cal.coupled_fraction, cal.k_tr_m3_s, cal.migration_per_s, cal.effective_f_k_m3_s
cal.exact_inversions          # yb_two_pool_from_decay per full-triple anchor -- NOT fitted
cal.w_profile, cal.w_interval_per_s     # the 1-D profile in W_mig and its x2-cost interval
cal.decay_report, cal.lifetime_report   # per-anchor model values and residuals

# --- 2. the two temperature laws (opt-in; act only under a temperature profile) -------------
amp = ErYbAmplifier(er, yb, fiber, pumps, signals, ase, n_yb_m3=4e26, k_tr_m3_s=1.11e-21,
                    yb_coupled_fraction=0.9, k_tr2_m3_s=2e-22,
                    yb_migration_rate_per_s=W_MIG_PHOSPHOSILICATE_PER_S,   # 2.2e2 1/s
                    rate_temperature=RATE_ARRHENIUS_CHENG_2022,            # default None
                    yb_stark_thermal=YB_STARK_976_CANAT_DUSSARDIER)        # default None
amp.set_temperature_profile(z_m, T_K, T_ref_K=300.0)     # BOTH laws key off THIS T_ref
res, T_z, info = solve_with_thermal_feedback(amp, ThermalModel(h_conv_W_m2K=200.0),
                                             b_outer_m=62.5e-6, n_nodes=161)
RateTemperatureLaw(k_tr_ea_over_k_K=65.2, k_tr2_ea_over_k_K=0.0, w_mig_ea_over_k_K=0.0)
RateTemperatureLaw.ea_over_k_from_ratio(1.30, 300.0, 480.0)           # -> 209.89 K
YbStarkThermal.from_upper_fractions(300.0, 0.07, 400.0, 0.13)         # -> dE/kB 823 K, g 1.17
YB_STARK_976_CANAT_DUSSARDIER.upper_fraction(400.0)                   # 0.13
YB_STARK_976_CANAT_DUSSARDIER.factor(480.0, 300.0)                    # 0.888

# --- 3. the per-fiber (f, k_tr) fit --------------------------------------------------------
fit = eryb_fit_to_device(lambda f, k: build_my_fiber(f, k),
                         DeviceTarget(p_out_W=7.15, one_um_fraction=5e-4,
                                      gain_dB=None, one_um_is_upper_bound=False,
                                      sigma_ln_power=0.05, sigma_ln_one_um=0.693),
                         migration_per_s=W_MIG_PHOSPHOSILICATE_PER_S,
                         f0=0.5, k_tr0_m3_s=1e-21,
                         observe=lambda a: device_observables(a, n_nodes=121, relax=1.0),
                         max_nfev=40)
fit.coupled_fraction, fit.k_tr_m3_s, fit.effective_f_k_m3_s
fit.correlation, fit.sigma_ln_f_k, fit.sigma_ln_ratio, fit.condition_number
fit.degeneracy_direction, fit.covariance, fit.residuals, fit.model
```

---

## 2. The migration rate, derived

### 2.1 The free two-pool decay has a closed form

Switch the pump off and hold the erbium in its ground state (`f2 -> 0`, the low-excitation
condition every published Yb decay is taken under) and `eryb._fb_rhs3` collapses to a LINEAR 2x2:

```
d b2c /dt = -(a + (1-f) W) b2c + (1-f) W b2nc,    a = 1/tau_Yb + k_tr N_Er phi
d b2nc/dt = +f W b2c - (d + f W) b2nc,            d = 1/tau_Yb
```

A pulse excites both pools to the SAME excitation fraction per ion (they absorb identically per
ion), so `b2c(0) = b2nc(0) = 1`, and the measured fluorescence follows the population-weighted
`I(t) = f b2c + (1-f) b2nc`:

```
tr   = a + d + W,   det = a d + W (f a + (1-f) d),   disc = (A - D)^2 + 4 f (1-f) W^2
lam_{1,2} = [tr -/+ sqrt(disc)]/2
I(t) = A1 exp(-lam1 t) + A2 exp(-lam2 t),  A1 + A2 = 1,  A1 lam1 + A2 lam2 = f a + (1-f) d = m1
INT b2c = (d + W)/det,   INT b2nc = (a + W)/det,   yield = f (a - d)(d + W)/det
```

Two structural points. The discriminant is a SUM OF SQUARES, so the decay is always a genuine
bi-exponential and can never be a damped oscillation -- no regime check is needed. And the two
integrals are the SAME closed forms `_solve_fbb` already uses for the steady state (there with
the pump on): one piece of algebra, written once for a root find and once for a decay. The
closed form is gated against a fixed-step RK4 march of `_fb_rhs3` itself -- an oracle that shares
no algebra with the eigen-decomposition -- to better than `1e-9` absolute on `I(t)` over 0.5 ms
and 25 000 steps, and its horizon-truncated integral to `2e-5` relative.

### 2.2 The inversion, and where the number comes from

Three measured numbers plus an Er-free control determine all three parameters, triangularly:

```
W = (lam1 - d)(lam2 - d)/(m1 - d),   R = k_tr N_Er = lam1 + lam2 - 2 d - W,   f = (m1 - d)/R
```

**`W_mig` is the SLOW component's excess rate over the Er-free control, `lam2 - d`, divided by the
coupled fraction.** A genuinely uncoupled ytterbium pool decays at exactly the intrinsic rate;
every `1/s` by which it decays faster is excitation that migrated into the coupled pool and was
transferred there. There is no other channel for it to leave by. `yb_two_pool_from_decay` REFUSES
(rather than clipping) a slow component slower than the control, which would need `W < 0` and
means the control and the sample are not the same host.

Cheng et al., Materials 15(3), 996 (2022) is the only located record that prints a bi-exponential
AND an Er-free control from the same melt: `tau_1 = 8.20 us`, `tau_2 = 1333.76 us`, `A1 = 0.864`,
control `tau_Yb0 = 1793.46 us`, `N_Er = 3.96e25 m^-3`. The inversion gives

| quantity | value |
| --- | --- |
| coupled fraction f | **0.8644** |
| coupled-pool k_tr | **3.065e-21 m^3/s** |
| **W_mig** | **2.22e2 1/s** |

and the anchor folder's own remark that "the slow component is 26% shorter than the Er-free
control, so even the uncoupled 13.6% of the ytterbium is losing excitation somewhere" is exactly
that `lam2 - d = 192 1/s`. The mechanism was in the data; it had not been named.

### 2.3 The reconciliation, which is the point

That ONE parameter set reproduces BOTH coefficients the 2026-09-14 anchor collection extracts
from this single measurement:

| derivation | anchor folder | this model at (f 0.864, k 3.065e-21, W 222) |
| --- | --- | --- |
| fast-pair `k_tr` (from `tau_1`) | 3.065e-21 m^3/s | 3.065e-21 (input) |
| transfer YIELD (from the integral) | 0.8949 | **0.89491** |
| ensemble-yield `k_tr` (the one-pool equivalent of that yield) | 1.199e-22 m^3/s | **1.199e-22** |

A factor of 25.6 between two coefficients from the same data, carried simultaneously. That is the
whole job `yb_migration_rate_per_s` was introduced to do, and it is now done with a number rather
than a hope.

### 2.4 Uncertainty, and the recommended constant

`W_mig` is a difference of two `~600-750 1/s` rates, which sounds fragile and is not:

| perturbation (+/- 5%) | W_mig (1/s) |
| --- | --- |
| `tau_fast` | 222.4 / 222.4 |
| `tau_slow` | 268.0 / 181.1 |
| `amp_fast` | 234.1 / 211.8 |
| `tau_Yb0` (the control) | 188.4 / 253.1 |
| joint +/- 5% on `tau_slow` AND `tau_Yb0` | 147.1 - 298.7 |

```
W_MIG_PHOSPHOSILICATE_PER_S       = 2.2e2 1/s     recommended, phosphosilicate, N_Yb 2-7e26
W_MIG_PHOSPHOSILICATE_RANGE_PER_S = (1.5e2, 3.0e2)  the +/- 5% measurement interval
W_MIG_PHOSPHOSILICATE_CEILING_PER_S = 1.0e4       the loosest value Jeong 2007 still admits
```

DOCUMENTED CONSTANTS, not defaults: `yb_migration_rate_per_s` still defaults to 0 and nothing in
the package reads these unless a caller passes them.

### 2.5 The two other anchor sets, and what each actually constrains

`eryb_calibrate_pools`, five legs, 25-point profile in `W_mig`:

| fit leg | f | k_tr | W_mig | f k_tr | cost | W profile interval (x2 cost) |
| --- | --- | --- | --- | --- | --- | --- |
| Cheng 2022 only | 0.8644 | 3.065e-21 | 222.4 | 2.65e-21 | 5.4e-11 | [215, 215] |
| Jeong 2007 only | 0.9405 | 2.467e-21 | 8250 | 2.32e-21 | 0 | [2.15, 1.0e4] |
| both decays | 0.9092 | 2.758e-21 | 209.9 | 2.51e-21 | 0.0284 | [100, 215] |
| Laroche 2006 only | 0.2186 | 3.446e-22 | 2.98e5 | 7.53e-23 | 0.0469 | [1.0e4, 1.0e8] |
| decays + Laroche | 0.7203 | 2.688e-21 | 169.4 | 1.94e-21 | 1.115 | [1, 2.15e3] |

(The Cheng-only interval is a single grid point because that leg is EXACTLY solvable -- its cost
is `5e-11` at the solution and orders larger one grid step away, so the "twice the minimum"
criterion admits nothing else. Its real uncertainty is the measurement one of 2.4, not this.)

**Jeong 2007** (the SPI 30/600 fiber that made 297 W) states its decay in words, so every
constraint is a band: 50-70% of the ytterbium relaxed within 10 us, a 10-15.5 us fast time
constant, `~2%` still excited at 100 us, an isolated fraction of 2-5%. All four are satisfiable
simultaneously (cost 0), and -- the point -- they are still satisfiable AT the Cheng value
(cost < 1e-6 with `W_mig` pinned to 2.2e2). Jeong therefore CONFIRMS the Cheng number without
sharpening it, and independently rules out `W_mig > 1e5`.

**Laroche 2006's five quenched lifetimes** are a different story, and the profile says so rather
than averaging over it. They are reproduced within 25% ONLY in the fully MIXED limit
(`W_mig >> k_tr N_Er ~ 1e5 1/s`), where the two pools share one excitation and the model is a
single population again:

| fiber (Yb:Er) | model | measured | ratio |
| --- | --- | --- | --- |
| 1 (35) | 652.7 us | 702.0 us | 0.930 |
| 2 (20) | 438.7 us | 406.0 us | 1.080 |
| 3 (18) | 361.9 us | 396.0 us | 0.914 |
| 4 (16) | 248.7 us | 215.0 us | 1.157 |
| 5 (10) | 190.3 us | 206.0 us | 0.924 |

worst deviation **15.7%**. At that fit the effective product is `f k_tr = 7.5e-23 m^3/s`, which is
Laroche's OWN measured `6.4-8.6e-23` recovered by an independent route -- so this dataset
constrains the PRODUCT and is silent about `W_mig` (its profile is flat over four decades,
`[1e4, 1e8]`). A single joint fit over both families (the last row of the table above) hides that
and is reported only to show what it costs: 1.115 against 0.028.

WHY the mixed limit is what Laroche need: their measured lifetimes track `N_Er` (702 -> 206 us
over a 6x Er range), and an UNMIXED two-pool decay's slow component is `d + W f`, which does not
depend on `N_Er` at all. Only migration strong enough to merge the pools makes the whole decay
`1/(1/tau_Yb + f k_tr N_Er)`, the form that does.

A by-product worth recording. At the calibrated `W_mig` and Cheng's fast-pair `k_tr`, each Laroche
fiber ON ITS OWN implies a coupled fraction, and it RISES monotonically as the Yb:Er ratio falls:

| fiber | N_Er (m^-3) | Yb:Er | implied f |
| --- | --- | --- | --- |
| 1 | 1.03e25 | 35 | 0.415 |
| 2 | 2.05e25 | 20 | 0.643 |
| 3 | 2.72e25 | 18 | 0.649 |
| 4 | 4.50e25 | 16 | 0.804 |
| 5 | 6.30e25 | 10 | 0.810 |

-- more erbium per ytterbium, more ytterbium with an erbium neighbour. That is Melkumov's own
reading of what sets the ETE, arrived at here as a consequence rather than an input, and it is
the strongest evidence in this note that `f` is a FIBER property and not a host constant.

---

## 3. The FORC ETE table, before and after

`validation/eryb_migration_thermal_fit.py` section 2 re-runs the 2026-09-14 note's own section 4.2
sweep -- the Exail AG-EY-O-5 at 2 W of 976 nm, `f k_fast` held at 2.7e-21 while `f` is swept, so
only the SPLIT moves -- at four migration rates. Model 1-um fraction, and the factor against the
measured FORC value:

| ETE | measured 1-um | W = 0 (the 2026-09-14 column) | W = 2.2e2 (calibrated) | W = 1e4 (Jeong's ceiling) | W = 1e6 |
| --- | --- | --- | --- | --- | --- |
| 0.20 | 21% | 70.9% (x3.38) | 70.0% (x3.33) | 15.2% (x0.72) | 4.7e-4% (x2.3e-5) |
| 0.25 | 2% | 63.4% (x31.7) | 62.2% (x31.1) | 2.80% (x1.40) | 4.4e-4% (x2.2e-4) |
| 0.38 | 0.5% | 43.4% (x86.9) | 41.2% (x82.5) | 0.063% (x0.127) | 3.9e-4% (x7.8e-4) |
| 0.90 | 0.05% | 0.0043% (x0.086) | 0.0035% (x0.070) | 0.00065% (x0.013) | 0.00033% (x0.0065) |
| **RMS of \|log10(model/measured)\|** | | **1.36 (a typical factor of 23)** | **1.37 (factor 23)** | **1.05 (factor 11)** | **3.52 (factor 3300)** |

The `W = 0` column reproduces the 2026-09-14 note's own 71% / 40% / 0.004% at f = 0.2 / 0.4 / 0.9
exactly, so the two runs are on the same basis.

**READING, and it is not the one the brief expected.** The CALIBRATED `W_mig` changes the FORC
residuals by a few percent and not at all in the aggregate (1.36 -> 1.37 in RMS log). That is not
a surprise once the number is in hand: `2.2e2 1/s` is three orders below the coupled pool's own
transfer rate `k_tr N_Er ~ 1e5 1/s`, so it cannot move an ASE fraction that hangs off a 20-30 dB
single-pass gain. A migration rate at the TOP of what the decay anchors admit (`1e4 1/s`, 45x the
calibrated value) DOES help and substantially -- the ETE 0.25 row moves from x32 to x1.4 and the
ETE 0.38 row from x87 to x0.13, and the aggregate falls from a factor of 23 to a factor of 11 --
but it over-corrects the high-ETE row and it is not the value the measurement gives. Push further
and the two pools merge, the 1-um fraction collapses to a single-population number, and the ETE
dependence the model exists to express is gone.

So **the FORC table does not select `W_mig`**, and reporting it as though it did would be fitting
the parameter to the wrong observable. What the table is dominated by is the 2026-09-14 note's own
limit 6: a factor-2 uncertainty on `sigma_e_Yb` (Morasse 2006 needed 0.4x the McCumber value to
match his measured 1-um ASE), compounded by the exponential sensitivity of an ASE fraction to the
single-pass gain. The arithmetic is brutal and it is why this residual was always going to be the
last one standing: `sigma_e_Yb` multiplies the gain COEFFICIENT, so halving it halves `ln G` --
a 25 dB single-pass 1-um gain becomes 12.5 dB, i.e. the 1-um power falls by 12 dB, a factor of 16,
and Morasse's own 0.4x is 15 dB and a factor of 32. Two to three decades of 1-um fraction from a
factor of two in one cross-section: exactly the size of the residual, from an uncertainty that was
already on the books before this branch.
Section 4 responds to that the right way: it fits `(f, k_tr)` per fiber to BOTH observables
instead of reading `f` off the published ETE.

---

## 4. Per-fiber (f, k_tr)

`validation/eryb_migration_thermal_fit.py` section 3, `W_mig` held at the calibrated `2.2e2 1/s`,
`f` and `k_tr` boxed to the literature ranges, both observables scored (5% on the power, a factor
of 2 on the 1-um fraction), 121 nodes, `relax = 1.0`.

The four FORC fibers are pumped at 976 nm co-propagating with 25 W launched and 0.6 W of 1555 nm
seed. Rows 1 and 2 carry ASSUMED compositions (ASL-116's erbium, ytterbium scaled as
`1/L_optimal`); rows 3 and 4 are printed in full. Bai 2015 stage 1 is a Nufern PM-EYDF-12/130,
3.0 m, 7.5 W of 976 nm, 30 mW of seed, with densities back-computed from the catalogue
absorptions.

| fiber | N_Er (m^-3) | N_Yb (m^-3) | measured out / 1-um | **fitted f** | **fitted k_tr** | f k_tr | k_tr N_Er |
| --- | --- | --- | --- | --- | --- | --- | --- |
| FORC ETE 0.20, 8.0 m *(assumed)* | 8.8e24 | 1.36e25 | 5.450 W / 21% | **0.170** | **1.89e-21** | 3.21e-22 | 1.66e4 1/s |
| FORC ETE 0.25, 5.0 m *(assumed)* | 8.8e24 | 2.18e25 | 5.450 W / 2% | **0.123** | **8.91e-21** | 1.09e-21 | 7.84e4 1/s |
| FORC ASL-116, ETE 0.38, 4.5 m | 8.8e24 | 2.42e25 | 5.400 W / 0.5% | **0.304** | **1.91e-21** | 5.81e-22 | 1.68e4 1/s |
| Nufern 25P/300-HE, ETE 0.90, 1.5 m | 1.1e25 | 1.94e26 | 7.150 W / 0.05% | **0.812** | **1.63e-21** | 1.33e-21 | 1.80e4 1/s |
| Bai 2015 stage 1, 12/130, 3.0 m | 1.79e25 | 1.15e26 | 2.600 W, 19.4 dB, 1-um "not obvious" | **0.631** | **1.98e-21** | 1.25e-21 | 3.55e4 1/s |

and what each fit actually achieved, with the covariance that says how much to believe it:

| fiber | model out | model 1-um | power residual | 1-um residual | corr(logit f, ln k) | sigma[ln f k] | sigma[ln f/k] | cond |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FORC ETE 0.20 | 5.11 W (-6.2%) | 0.102% (x0.0048) | -1.29 | -7.69 | -1.000 | 6.88 | 19.8 | 8.7e3 |
| FORC ETE 0.25 | 5.27 W (-3.3%) | 0.0857% (x0.043) | -0.68 | -4.55 | -0.999 | 4.87 | 11.2 | 6.6e3 |
| FORC ASL-116 | 5.26 W (-2.6%) | 0.0779% (x0.156) | -0.52 | -2.68 | -0.999 | 2.92 | 7.17 | 2.5e3 |
| **Nufern 25P/300-HE** | **7.150 W** | **0.0500%** | **0.00** | **0.00** | **-0.810** | **0.690** | **0.904** | **9.68** |
| **Bai 2015 stage 1** | **2.602 W**, 19.4 dB | 0.964% (under the bound) | **+0.016** | 0 (inactive) | n/a | inf | inf | inf |

**READING.**

*The two fibers with fully specified geometry fit, and one of them is a genuine out-of-sample
success.* The Nufern LMA-EYDF-25P/300-HE fit hits BOTH observables exactly (two parameters, two
residuals, an exact root) with a WELL-CONDITIONED covariance -- condition number 9.7 and a
correlation of only -0.81, meaning `f` and `k_tr` were separately determined rather than trading
off -- and it returns `f = 0.812` against a MEASURED energy-transfer efficiency of **0.90**. The
fit was never told the ETE. A two-population model asked to reproduce a watt-class output power
and a 0.05% 1-um fraction independently recovers the fiber's published coupled fraction to 10%.
That is the strongest single result in this note.

Bai 2015 stage 1 lands at `f = 0.631`, `k_tr = 1.98e-21`, i.e. `f k_tr = 1.25e-21` -- within 25%
of the study's own device-validated effective coefficient of `1.0e-21` -- while reproducing 2.602 W
against 2.600 W measured, 19.4 dB against 19.4 dB, and a 1-um fraction of 0.96%, just under the
1% bound that "no obvious 1-um ASE" was scored as. Its covariance is SINGULAR and the routine says
so (`inf`, not a small number): with the one-sided 1-um bound inactive its Jacobian row is
identically zero, so the split is undetermined by construction and only the product is fitted.
That is the correct report, and it is the behaviour the eigen-decomposition guard was added for --
a plain `inv(J^T J)` on that matrix returns garbage that can come back as a spuriously TIGHT error
bar, an uncertainty that says the opposite of the truth.

*The three 20/125 fibers cannot reach their measured 1-um fractions, and the fit rails rather than
lies.* All three match the output power to 3-6% while landing 6x to 210x LOW on the 1-um fraction,
with `f` driven down to 0.12-0.30 trying to raise it and a condition number of `2.5e3-8.7e3`
saying only the product was pinned. Note the SIGN: section 3's Exail fixture OVER-predicted the
low-ETE 1-um fraction by 3-90x, and these large-core low-`N_Yb` fibers UNDER-predict it by 6-210x.
A quantity that swings two decades in each direction between fixtures is not being set by the
coupled fraction; it is being set by the single-pass 1-um gain, which the factor-2 `sigma_e_Yb`
uncertainty of the 2026-09-14 note's limit 6 moves by 20-30 dB. Two of the three rows also carry
assumed compositions, but the third (ASL-116, fully printed) is 6.4x low on its own, so the
assumption is not the explanation.

*What to take from this for a per-fiber-class calibration.* The fitted `k_tr N_Er` is remarkably
stable across the four independent fibers whose composition is not in doubt or whose absorptions
are catalogued -- `1.7e4`, `1.8e4` and `3.6e4 1/s` -- while `f` spans 0.30 to 0.81 and tracks the
fiber's published ETE where one exists. That is the two-population model behaving as designed: one
transfer rate per coupled pair, one coupled fraction per fiber.

---

## 5. Temperature

### 5.1 What is measured

Cheng et al. 2022, Table 2, Er/Yb/P silica core glass under EXTERNAL heating, 300 -> 480 K -- the
only temperature series on a co-doped phosphosilicate in the 2026-09-14 collection:

| quantity | 300 K | 480 K | change |
| --- | --- | --- | --- |
| Yb decay FAST component `tau_1` | 8.20 us | 7.56 us | -7.8% |
| Yb decay SLOW component `tau_2` | 1333.76 us | 1279.25 us | -4.1% |
| Er `4I13/2` lifetime | 9.12 ms | 8.14 ms | -10.7% |
| `sigma_a_Yb(974 nm)` | 9.43e-25 | 5.86e-25 | -37.9% |
| `sigma_a_Yb(940 nm)` | 1.73e-25 | 1.73e-25 | 0 (stated insensitive) |
| `sigma_a_Yb(915 nm)` | 1.73e-25 | 1.26e-25 | -27.2% |
| `sigma_a_Yb(1018 nm)` | 2.9e-26 | 6.0e-26 | +107% |
| **transfer rate `1/tau_1 - 1/tau_2`** | **1.2120e5 1/s** | **1.3149e5 1/s** | **+8.5%** |

The last row is the TRANSFER INDICATOR this model actually carries, and it is derived from the two
printed decay components rather than quoted. **The frequently-cited "+30% over 300-480 K" figure
is NOT reproducible from the anchor collection**: no measurement supporting it is present there,
and the two decay components give +8.5%. Both laws are shipped, clearly labelled, and the
coefficient is a constructor argument either way.

### 5.2 The simplest law that reproduces it

`RateTemperatureLaw`: `rate(T) = rate(T_ref) exp[-(Ea/kB)(1/T - 1/T_ref)]`, one activation
temperature per coefficient, `T_ref` taken from the amplifier's own temperature profile so a
uniform profile at `T_ref` is an EXACT identity. Fitted rather than transcribed, via
`RateTemperatureLaw.ea_over_k_from_ratio`:

| constant | Ea/kB | k_tr(480 K)/k_tr(300 K) | source |
| --- | --- | --- | --- |
| `RATE_ARRHENIUS_CHENG_2022` | **65.20 K** | **1.0849** | Cheng's two decay components |
| `RATE_ARRHENIUS_30PCT_300_480K` | 209.89 K | 1.3000 | the "+30%" reading; UNSOURCED |

It scales `k_tr` only. `K2` and `W_mig` get their own coefficients in the dataclass but ship at
zero, because Cheng measure no separate indicator for either and assuming the same activation
energy would be an invention. `k_back` is never scaled (no measurement; 0 by default).

### 5.3 The 976 nm Stark depopulation, and why it is not already in the McCumber factor

Canat (2006) / Dussardier (2005) give the 976 nm terminal level's thermal depopulation as **7% at
300 K and 13% at 400 K**. `YbStarkThermal.from_upper_fractions(300, 0.07, 400, 0.13)` is an exact
2-point inversion of the two-level partition function, giving `Delta_E/kB = 822.9 K = 572 cm^-1`
and `g = 1.169` -- the right magnitude for the Yb `2F5/2` Stark splitting, which is a check the
fit could have failed and did not. The multiplier on both Yb cross-sections inside the band:

| T (K) | upper-level fraction | `factor(T, 300 K)` |
| --- | --- | --- |
| 300 | 0.0700 | 1.0000 |
| 350 | 0.1002 | 0.9675 |
| 400 | 0.1300 | 0.9355 |
| 450 | 0.1581 | 0.9053 |
| 480 | 0.1739 | 0.8883 |
| 500 | 0.1840 | 0.8774 |

**DOES THE EXISTING McCUMBER T-SCALING ALREADY CONTAIN IT? No, and that is proven rather than
argued.** `set_temperature_profile` scales `sigma_e` by `exp[(eps_ion - h nu)(1/kT - 1/kT_ref)]`.
At the ytterbium ZERO LINE `h nu = eps_Yb` exactly, so that factor is IDENTICALLY 1 at 976 nm at
every temperature -- gated to `1e-12` on a channel placed at `h c / eps_Yb`, with the same gate
checking that it is NOT 1 at 1030 nm so the test is not vacuous -- and it never touches `sigma_a`
at any wavelength. McCumber moves the RATIO `sigma_e/sigma_a` away from the zero line; the Stark
factor moves the SCALE of both AT it. They are orthogonal by construction and there is no double
count.

Both cross-sections take the SAME factor, and that is forced rather than chosen: McCumber requires
`sigma_e = sigma_a` at the zero line at every temperature, so scaling one without the other would
break the relation the module elsewhere enforces.

The band is deliberately narrow (976 +/- 15 nm). This is a ZERO-LINE model: Cheng measure 940 nm
to be temperature-insensitive and 1018 nm to RISE by 107%, so applying a zero-line factor across
the spectrum would get the 1-um band's sign wrong. SIZE CHECK: at 480 K the factor is 0.888 while
Cheng measure `sigma_a_Yb(974)` at 0.62 of its 300 K value -- the Stark depopulation accounts for
about a third of the measured loss and thermal line broadening (not modelled) for the rest, so
read it as a LOWER BOUND on pump-band softening.

### 5.4 The hot-fiber gate

6 um core, cladding-pumped, `f = 0.5`, uniform profile, both laws on. 1-um ASE fraction against
launched pump, and the THRESHOLD at which it reaches 1% of the output:

| T (K) | laws | 0.25 W | 1 W | 2 W | 4 W | 8 W | 16 W | 1% threshold |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 300 | off | 4.26e-4 | 0.0239 | 0.183 | 0.315 | 0.418 | 0.533 | **0.85 W** |
| 300 | on | 4.26e-4 | 0.0239 | 0.183 | 0.315 | 0.418 | 0.533 | **0.85 W** (EXACT identity at T_ref) |
| 400 | off | 9.45e-5 | 1.99e-4 | 7.60e-4 | 8.91e-3 | 0.0767 | 0.194 | **4.22 W** |
| 400 | on | 9.23e-5 | 1.92e-4 | 7.28e-4 | 8.39e-3 | 0.0723 | 0.185 | **4.33 W** |
| 500 | off | 5.03e-5 | 5.88e-5 | 8.46e-5 | 1.92e-4 | 7.41e-4 | 3.90e-3 | **> 16 W** |
| 500 | on | 4.83e-5 | 4.95e-5 | 5.63e-5 | 8.05e-5 | 1.77e-4 | 2.92e-3 | **> 16 W** |

The threshold rises monotonically with temperature, and turning the two laws ON lowers the 1-um
fraction further at every pump and every temperature ABOVE `T_ref` -- raising the 400 K threshold
from 4.22 to 4.33 W. At `T_ref` itself they are the exact identity, which is the 300 K pair above
and is a gate rather than a coincidence. That is the qualitative result Canat / Dussardier
report and the one Morasse 2007 MEASURED -- a core-temperature rise moved his own 1-um onset from
14 W to 35 W of pump, "which allowed us to reach 10 W output at 1563 nm instead of the 5 W
normally predicted by the theory". Dong 2020's Fig. 3 room-temperature threshold near 200 W on the
Southampton fiber is the same statement at a different scale. The gate is the MONOTONICITY, not
any of these numbers.

A note on what did NOT change. `yb_parasitic_gain_dB` still evaluates the 1030 nm single-pass gain
at the ION's room-temperature cross-sections (it takes `sigma_e_Yb(1030)` straight from the ion,
not from the per-z McCumber matrix), so under a temperature profile it reports the COLD parasitic
gain at the hot inversion. That was true before this branch and is left alone rather than flipped;
the temperature-aware quantity is the resolved `yb_ase` band output, which is what the table above
and `device_observables` use.

---

## 6. Gate numbers

`tests/test_fiber_eryb_migration_thermal.py`, **19 gates, 243-285 s** (two runs on the same box,
under nine and sixteen competing full-core workloads; the three device-fit / threshold gates are
80-90 s each because every residual evaluation inside them is a full relaxation solve, and the
other sixteen are 35 s between them).

| Gate | Measured |
| --- | --- |
| explicit `None` / an identity `RateTemperatureLaw()` == omitting them, BITWISE, COLD and HOT | exact array equality on `power_W`, `nbar2_z`, both pools, the gain, `eta_transfer`, all 13 keys of `energy_terms`, `_heat_profile_W_per_m` and the three-reservoir march -- with and without a 380 K profile set |
| the new branches are structurally unreachable with the defaults | `_t_rates is False`; `_mcc_matrices` returns `None` with no profile and a 4-tuple whose slots 2 and 3 are `None` with one; an identity rate law is DROPPED in the constructor rather than carried |
| `_clone` carries both opt-ins | `with_signals` preserves `rate_temperature`, `yb_stark_thermal` and `_t_rates` |
| a uniform profile at `T_ref` with BOTH laws on == the isothermal solve | bitwise (`array_equal` on the powers and the Yb inversion) |
| the closed-form decay vs a fixed-step RK4 of `_fb_rhs3` itself | worst `|I_closed - I_RK4| < 1e-9` over 0.5 ms / 25 000 steps at `f = 0.62`, `k_tr = 3e-21`, `W = 2.5e3`; the horizon-truncated integral to `2e-5` relative |
| the decay's three analytic limits | `W = 0` -> rates `(a, d)` and amplitudes `(f, 1-f)` to `1e-12`; `f = 1` -> `tau_bar = 1/(1/tau_Yb + k_tr N_Er)` to `1e-12` (Laroche's own inversion formula); `W -> inf` -> slow rate `d + f k_tr N_Er` to `1e-4` |
| the exact inversion round-trips AND carries both Cheng coefficients | `tau_1`, `tau_2`, `A1` back to `1e-10`; `k_tr = 3.065e-21` to 1%; `yield = 0.8949` to `1e-3`; the ensemble-equivalent `1.199e-22` to 2%; their ratio `25.6` to 5%; `W_mig = 222.4 1/s` to 1%, inside `W_MIG_PHOSPHOSILICATE_RANGE_PER_S` |
| the inversion REFUSES an inconsistent control | a slow component longer than the Er-free control raises `ValueError` rather than returning `W < 0` |
| Jeong 2007's stated pattern | fit cost `< 1e-6`; relaxed-by-10 us `0.594` in [0.50, 0.70]; still-excited-at-100 us `0.030` in [0.010, 0.030]; `tau_fast = 10.0 us` in [10, 15.5]; `A1 = 0.930` in [0.93, 0.98]; `k_tr = 2.47e-21` in the fast-pair decade -- and still satisfiable with `W_mig` PINNED to the Cheng value |
| Laroche 2006's five lifetimes | worst deviation **15.7%** (<= 25%); the fit reaches it only at `W_mig = 2.98e5 1/s` (> 1e4) and `f k_tr = 7.53e-23` (inside Laroche's own 6.4-8.6e-23); its `W_mig` profile is flat over `>1e3` in `W` |
| migration raises the transfer yield monotonically | strictly increasing over `W = 0, 1e2, 1e3, 1e4, 1e6`, hitting the analytic `f R/a` at `W = 0` (`1e-12`) and `f R/(d + f R)` at `W = 1e6` (`1e-3`) |
| McCumber is IDENTICALLY 1 at the Yb zero line (the no-double-count proof) | `|mcc_Yb - 1| < 1e-12` at `h c / eps_Yb` at 460 K, and NOT 1 at 1030 nm -- so the gate is not vacuous |
| the shipped temperature constants reproduce their own anchors | `upper_fraction(300) = 0.070` and `(400) = 0.130` to `1e-9`; `factor(400, 300) = 0.87/0.93` to `1e-9`; `Delta_E = 572 cm^-1` inside (500, 650); the band mask excludes 915 / 940 / 1018 / 1030 nm and includes 976; `RATE_ARRHENIUS_CHENG_2022` reproduces Cheng's `1/tau_1 - 1/tau_2` ratio to `1e-12` and the 30% law reproduces 1.30 to `1e-12`; both are exactly 1 at `T_ref` |
| both laws move a hot solve, in the STATED direction | at 460 K the rate law raises `eta_tr` and the gain; the Stark factor LOWERS the gain (a weaker 976 nm line absorbs less pump). A sign flip in either breaks this |
| the 1-um threshold of a hot fiber rises with temperature | the 1-um fraction strictly decreases at 300 -> 400 -> 500 K at every one of 2 / 4 / 8 W, with the laws off AND on |
| thermal feedback with both laws | converges, `heat_source = "rate_balance"`, `T_z >= T_coolant`, finite gain |
| the device fit recovers a synthetic `(f, k_tr)` | `f = 0.45` and `k_tr = 2.0e-21` both back to **5%** from a start at `(0.80, 6.0e-22)` |
| ONE observable leaves the split undetermined, and the fit SAYS so | power-only: condition number `> 1e12` and BOTH sigmas `inf` (the routine refuses an error bar rather than quoting the small one a naive `inv(J^T J)` returns), with a degeneracy direction aligned with neither axis; power + 1-um: condition number `< 1e3` and both sigmas finite. `f k_tr` comes back from either (25% / 10%) |
| an upper-bound target is scored one-sided | a model below the bound gives `residual == 0.0` exactly |

---

## 7. Limits the caller has to know

1. **`W_mig` rests on ONE measurement.** Cheng 2022's bi-exponential plus its own Er-free control
   is the only located record with all four numbers from the same melt. Jeong 2007 confirms the
   value without sharpening it, and Laroche 2006 is silent about it. A second bi-exponential with
   a matched control would be the single most valuable addition to this calibration.
2. **The Cheng `N_Er` is assumed.** `3.96e25 m^-3` from `2 x 0.0009 x 2.2e28` formula units, the
   silica value; at 12.45 mol% P2O5 the molar volume is larger, so this OVERSTATES `N_Er` by
   perhaps 10-20% and understates `k_tr` by the same factor. `f` and `W_mig` are untouched by it
   -- they come from RATES and AMPLITUDES, which are density-free.
3. **Jeong's `N_Er` is assumed too** (the study's own `4.0e25`; the paper does not print it). That
   assumption sets the `k_tr` that leg reports and nothing else.
4. **The two decay anchors and the Laroche lifetimes are not one dataset.** Fitting them jointly
   is offered and reported, and it is worse than either alone by a factor of 40 in cost. Use the
   decays for `W_mig` and the lifetimes for `f k_tr`, or state which you are doing.
5. **The FORC 1-um magnitudes are still 1-2 decades out** and migration does not close them --
   see section 3. The 2026-09-14 note's limit 6 (a factor-2 `sigma_e_Yb`) stands unchanged and is
   the live hypothesis.
6. **The rate law's activation energy is small and its lever is short.** `Ea/kB = 65 K` over a
   180 K span is +8.5%; extrapolating it to 700 K or to a different host is not supported by
   anything measured here.
7. **`YbStarkThermal` is a ZERO-LINE model** applied over a 30 nm window. It says nothing about
   915 / 940 / 1018 nm, where the measured temperature behaviour differs in sign, and it does not
   model thermal line broadening -- which is why it reproduces only about a third of Cheng's
   measured 974 nm loss at 480 K.
8. **The transient march still refuses a temperature profile.** Unchanged from 2026-09-14.
9. **Two of the four FORC fibers have ASSUMED compositions.** Only ASL-116 and the Nufern have
   printed at% and a datasheet; rows 1 and 2 carry ASL-116's erbium and a ytterbium scaled as
   `1/L_optimal`. Their fitted `(f, k_tr)` are conditional on that and are labelled so in the
   report.
10. **The inversion assumes EQUAL initial excitation of the two pools.** `b2c(0) = b2nc(0)` is
    right for a pulse into an unpumped sample -- both pools are the same ion at the same
    wavelength, so their per-ion absorption rates are identical -- which is how Cheng and Jeong
    both measure. It would be WRONG for a decay measured after CW pre-excitation, where the
    steady state already has `b2c < b2nc`; the amplitudes, and therefore the extracted `f`, would
    shift. Check the excitation protocol before inverting someone else's decay.
11. **`eryb_fit_to_device` calls the solver inside an optimiser.** Every residual evaluation is a
    full relaxation solve; `relax = 1.0` converges these fibers in 3-9 iterations where
    `relax = 0.5` needs 26-29 for the same fixed point, which is the difference between a minute
    and an hour. The routine does not choose for you -- pass your own `observe`.

---

## 8. Verification run

```
ruff check .                                                  -> clean (the whole tree)
pytest tests/test_fiber_eryb_migration_thermal.py             -> 19 passed, 243 s
pytest tests/test_fiber_eryb_physics.py tests/test_fiber_eryb.py
       tests/test_fiber_eryb_transient.py tests/test_fiber_thermal_feedback.py
       tests/test_measured_spectra_2026_08_28.py
       tests/test_yb_mccumber_refit.py                        -> 84 passed
pytest tests/test_fiber_amp.py tests/test_audit_2026_08_04_fiber_amp.py
       tests/test_fiber_chain.py tests/test_fiber_dynamics.py
       tests/test_numerics.py                                 -> 194 passed
pytest tests/test_audit_2026_07_25_infra.py tests/test_fiber_bpm.py
       tests/test_fiber_gnlse.py tests/test_fiber_lma.py tests/test_fiber_nonlinear.py
       tests/test_fiber_srs.py tests/test_fiber_transverse.py
       tests/test_validation_runner.py                        -> 147 passed
python -m validation.eryb_migration_thermal_fit --sections 1  -> section 2 of this note
python -m validation.eryb_migration_thermal_fit --sections 2  -> section 3
python -m validation.eryb_migration_thermal_fit --sections 3  -> section 4
python -m validation.eryb_migration_thermal_fit --sections 4  -> section 5
```

Those four `pytest` lines are EVERY test file in the repo that imports `fiber_amp`, which is the
complete blast radius: this change touches `dynameta/optics/fiber_amp/*`, the two version strings,
`validation/run_all.py`'s exclusion table and one count pin in `tests/test_fiber_amp.py` (the
submodule census, 22 -> 23 with `eryb_fit`).

### The out-of-process byte-identity check

The in-repo gate compares an amplifier against one with the new parameters SPELLED OUT at their
defaults, in one process, by exact array equality -- which is reproducible on every runner but
cannot see a change that moves BOTH sides equally. So it is backed by a check the gate cannot
make: a fingerprint script (steady gain, `eta_tr`, the parasitic gain, the summed populations and
powers, all five `energy_terms` channels, the stored energy, the heat profile and a
three-reservoir march) run under `PYTHONPATH` pointed at the **v0.11.2 checkout** and then at this
worktree, over four configurations -- one pool, two pools, `ConcentrationModel`, and two pools
under a 380 K axial temperature profile. The two JSON dumps are IDENTICAL, byte for byte, at full
float repr. That is the claim "v0.11.2 is reproduced bitwise" measured across two working trees
rather than argued.

