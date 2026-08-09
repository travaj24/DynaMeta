# fiber_amp consumer-facing audit -- 2026-08-04

**Scope.** `dynameta.optics.fiber_amp` (v0.9.0, ~7.5 kLOC over 20 modules) plus `optics.amp_noise`,
audited from the position of an EXTERNAL CONSUMER rather than from inside the package: the driving
task was to build a burst-mode PAM-N optical-communication link simulator on top of the library --
pseudo-random symbol bursts through an Er (1550 nm) and a Yb (1060 nm) single-mode amplifier driven
into gain saturation, with an end-to-end wall-plug-efficiency budget.

That vantage point is the point of this audit. The package's own audit trail (S3-*, A-*, X-*, C4-*,
S6-*) is thorough about internal consistency, and this pass found **no errors in the Giles /
Desurvire propagation core** -- an independently written frozen-inversion propagator reproduced
`FiberAmplifier.solve()` to 4e-4 relative on total forward ASE and 1e-5 on the signal and residual
pump (method in sec. "Verification method" below). What it did find is a) one quantified
**physical-accuracy defect in the analytic Er cross-sections**, b) several **API gaps that force a
consumer to reimplement package internals**, and c) two **whole missing layers** that the package's
own roadmap documents assume will exist.

**STATUS: all 14 findings IMPLEMENTED, 2026-08-05; adversarially verified and re-worked
2026-08-08.** This document was written as a report ("no library code was modified"); the fixes
landed the following day, and an adversarial verification pass over the implementation produced a
further 21 corrections (see "Post-implementation verification" below). What records what was
actually done, and where it diverged from the suggestions here, is the **implementation-record
table** in the next section -- there is no per-finding IMPLEMENTED note inside the sections below,
and an earlier version of this sentence claimed there was. Gates live in
`tests/test_audit_2026_08_04_fiber_amp.py` (81 tests in 12 classes).

Two corrections to this document, found while implementing and confirmed by measurement -- recorded
rather than quietly edited away, since the original claims were wrong:

* **F-9 named the wrong fixture.** The claim that `tests/test_fiber_thermal_feedback.py` runs above
  V = 2.4 is FALSE: its fiber (a = 5 um, NA = 0.07) sits at V = 2.2532 at 976 nm and 2.1351 at
  1030 nm, inside the Marcuse window. The genuinely out-of-range fixtures are the 25 um / NA 0.054
  LMA ones in `test_fiber_transverse.py`, `test_fiber_bpm.py` and `validation/fiber_gain_bpm.py`
  (V = 8.69 / 8.22), plus the 3 um / NA 0.20 Er:Yb fixtures (V = 3.86 at 976 nm).
* **F-1 overstated the shape claim.** This document called the resulting C-band gain ripple
  "non-physical". A real EDFA's GAIN spectrum genuinely does have a valley near 1540 nm -- that is
  why gain-flattening filters exist -- so a non-monotone gain is not by itself evidence of a defect,
  and the implemented refit does NOT make the gain monotone. What survives is the part that was
  always airtight: the internal McCumber contradiction (2.058x worst case), the excess depth of the
  `sigma_a` trough, and the measured -6.8 -> +0.9 percentage-point change in the 1540 -> 1545 nm
  power-conversion step.

## Result: 14 findings (0 critical, 2 high, 7 medium, 4 low, 1 informational)

| # | sev | dim | file:line | finding |
|---|-----|-----|-----------|---------|
| F-1 | **high** | physics | `spectroscopy.py:104-118` | The analytic Er cross-sections violate their OWN McCumber relation by up to **2.06x** between the 1530 and 1560 nm anchors, and put a spurious `sigma_a` local minimum at 1543 nm. `EDFA_CBAND_TARGETS` specifies 1550.0 nm -- inside the defect. |
| F-2 | medium | api gap | `dynamics.py:251,271-274,322` | `simulate_transient` computes the full frozen-inversion power matrix `P (K, Nz)` every step and **discards every ASE channel**. `TransientResult` exposes only `signal_out_W` / `pump_out_W`, and `analyze_noise` accepts only a `SteadyStateResult`, so time-resolved ASE / OSNR / NF is unreachable from a transient. |
| F-3 | medium | api gap | `steady_state.py:221` | No public channel-plan accessor. `_plan()` is private; any external consumer needing `(ChannelSet, bc, u, is_ase, kind)` must reimplement it (43 lines, and it must be kept in sync by hand). |
| F-4 | medium | correctness | `metrics.py:202` | `slope_efficiency` reads the Stokes ceiling from `amp.pumps[0].lambda_m` alone. Dual-wavelength pumping (915+976 Yb, 980+1480 Er -- both standard) therefore reports a **wrong ceiling**, and the returned `slope` can legitimately exceed it. |
| F-5 | medium | feature | `detection.py` (whole module) | No receiver-ELECTRONICS noise anywhere in the repo: no Johnson/thermal, no TIA input-referred noise, no APD excess-noise factor. `snr_elec_dB` is therefore an optimistic bound that silently omits the term that DOMINATES at low received power. |
| F-6 | medium | feature | absent | No communications layer at all -- no modulation format, symbol stream, eye, BER, Q-factor, decision threshold, or per-level noise evaluation -- although `docs/DynaMeta_QD_SOA_extension_spec.md:940` explicitly asks for "a true symbol-stream SNDR/EVM". |
| F-7 | medium | feature | absent | No electrical / wall-plug layer. Efficiency stops at optical-optical (`power_conversion_efficiency`, `slope_efficiency`, `stokes_limit`). Zero repo-wide hits for `wall_plug`, `electrical_efficiency`, `diode_efficiency`, pump coupling loss, or energy-per-bit. |
| F-8 | low | api gap | `detection.py:90` | `amp_noise.beat_noise_variances` accepts `I_dark_A` but `detection_noise` never forwards it, so **photodiode dark current is unreachable** through the `fiber_amp` public API. |
| F-9 | low | robustness | `waveguide.py:23,27` | `mode_field_radius_m` states validity `1.2 < V < 2.4` and **nothing checks it**. Every LMA / cladding-pumped fixture in the repo runs outside it, and `fiber_amp_model_spec.md:287-290` quantifies the cost at V=8.2 as a 13% saturation-integral error (0.85 dB/m). Silent. |
| F-10 | low | ergonomics | `waveguide.py:51` | `FiberSpec` rejects `n_t_m3 <= 0`, so modelling a PASSIVE fiber span needs the `n_t_m3=1.0` hack (`tests/test_fiber_srs.py:22`). |
| F-11 | low | docs | `spectroscopy.py:97-107` | The `erbium()` docstring's summary of pump bands omits the **1480 nm** anchor that the code actually carries (`(1.480e-6, 0.040e-6, 0.8e-25)`), so the in-band-pump capability is invisible from the docstring. |
| F-12 | info | physics | `spectroscopy.py:129-143` + `thermal.py` | Yb `sigma_e/sigma_a` over 1030-1080 nm sits **3-11x BELOW** McCumber-from-`sigma_a` with `eps = hc/zero_line_m`. Not a bug -- it is `eps` being taken as the absorption peak rather than a fitted mean transition energy -- but it shares a root cause with the already-documented 3-5x overstatement of the thermal `sigma_e(T)` slope. |
| F-13 | **high** | correctness | `steady_state.py:444-445` | `SteadyStateResult.meta['converged']` is a **permanent false negative on any amplifier whose pump is fully absorbed** -- i.e. on every efficient design. The endpoint test divides an integrator-noise difference by an absolute `1e-15` floor. Measured: the flag never trips through 3000 iterations while the gain is stable to 5 decimals, and the solve burns 200 iterations instead of ~60. The docs instruct users to check this flag. |
| F-14 | medium | robustness | `steady_state.py:384-453` | The relaxation genuinely FAILS (not just the flag) for a low-signal, high-pump quasi-three-level amplifier: at 0.05-0.1 mW into the 2 W Yb reference it returns a **spurious near-unpumped solution** (gain -12.6 dB, `nbar2` 0.036) and more iterations do not help. Bidirectional ASE comparable to the pump makes the Gauss-Seidel iteration's oscillation fail to decay. No under-relaxation control is exposed. |

---

## Implementation record (2026-08-05)

All 14 landed. Gates: `tests/test_audit_2026_08_04_fiber_amp.py` -- **81 tests in 12 classes**, not
"one class per finding" as this line used to say. The map is: F-1, F-2, F-3, F-4, F-6, F-7, F-9,
F-10, F-12, F-13, F-14 have a class each; **F-5 and F-8 SHARE `TestF5F8Detection`** (one signature
change fixed both); **F-11 has NO class** -- it is a docstring-only finding (the `erbium()` summary
now names the 1480 nm in-band pump) and there is nothing behavioural to gate.

Regression control: the failure set of the full suite is **IDENTICAL** with and without these
changes (26 failures in the affected files either way, all `ModuleNotFoundError` on absent optional
extras -- `tmm`, DEVSIM, ngsolve). The fiber suite is 151/151 and the cross-solver 1e-9 reduction
gates are 62/62.

| # | Implemented as | Deviation from the suggestion in this document |
|---|---|---|
| F-1 | `erbium(cband_refit=True)`: one fitted C-band `sigma_a` Gaussian (1.543 um, FWHM 14 nm, peak 1.0e-25) plus `sigma_e` DERIVED from `sigma_a` by McCumber. Consistency exact (1.000000x vs 2.058x); `sigma_a` monotone over 1533-1570 nm; 1530/1560 anchors move +1.54%/+0.99%; pump bands bit-identical; 1480 nm now correctly bleaches (max inversion 0.7425, not 1.0). | **`cband_refit=True` IS THE DEFAULT as shipped** (this row originally recorded the opt-in intermediate state; the commit that landed the audit flipped it and re-pinned the affected goldens -- test_fiber_amp.py and test_fiber_dynamics.py -- in the same change, and the full fast suite passed with the flip on the py3.10/3.12 CI legs of the first PR round). The re-baselining reasoning stands: anchors moved +1.54%/+0.99%, pump bands bit-identical. `cband_refit=False` remains available, and is the documented choice below inversion ~0.5 where the refit's L-band tail is extrapolated. Suggestions (b) and (c) were combined: anchor re-fit AND McCumber closure, which needs no external dataset. |
| F-2 | `TransientResult.ase_fwd_W`/`ase_bwd_W`/`ase_lambda_m`/`ase_dnu_hz`/`plan`, `ase_psd_1pol_W_Hz()`, `frame_as_steady(index)`, `simulate_transient(store_profiles=)`. Transient ASE matches `output_ase_spectrum` on the same frame EXACTLY (0.0); `analyze_noise` on a fixed-point frame matches the steady solve to 0.005 dB NF. | As suggested. ASE is captured unconditionally (free); only the `(Nt, K, Nz)` matrix is opt-in. `meta` also carries `m_modes` and the per-z McCumber matrix, so a frame cannot mix a T_ref `sigma_e` with a hot `nbar2` (the A-6 trap in transient form). |
| F-3 | `ChannelPlan` + `channel_plan()` on `FiberAmplifier` and `ErYbAmplifier`, with `indices(kind, direction)` and the plan-index == `power_W`-row contract gated. | `channels` is `None` for the co-doped case (two ions per channel, no single `ChannelSet`). `ResolvedFiberAmplifier` left alone -- its `_plan` is an 8-tuple with different meaning. |
| F-4 | `metrics.effective_pump_lambda_m` (power-weighted harmonic = photon-flux-weighted); used by `slope_efficiency`. | As suggested. Read after the sweep so a stage lacking the re-seed protocol still fails with the message `test_fiber_chain.py` pins. |
| F-5 | `detection_noise(dark_current_A, tia_current_noise_A_rtHz, load_ohm, temperature_K, apd_gain, apd_excess_noise_F)`; `BeatNoiseResult.var_thermal`/`var_optical`/`apd_*`. Verified against an independently derived APD closed form to 2e-16, with the correct limiting behaviour (thermally limited: SNR 7.19 -> 22.42 dB with M; beat limited: flat to slightly worse). | **`var_total` had to SPLIT** into `var_optical` + `var_thermal`: `nf_beat_dB` is computed from the optical part alone, because folding thermal noise into the noise figure would make the AMPLIFIER's NF depend on its detector -- the same error audit S3-10 removed. `added_rin_per_Hz` likewise stays optical-only. |
| F-6 | New `comms.py`. Equal-variance SER gated against `2(N-1)/N Q(d/2 sigma)` to 1e-12; the per-level-vs-rescaled error gated against its analytic value. | As suggested. |
| F-7 | New `efficiency.py`. Optical power balance closes against `total_heat_W` to 5.6e-17 W. | As suggested, including all three definitional choices; `duty_cycle` is a caller argument. |
| F-8 | Folded into F-5's signature; delta gated against `2 q I_d B_e`. | As suggested. |
| F-9 | `v_number`, `marcuse_validity() -> (ok, V_min, V_max)`, `V_MARCUSE_MIN/MAX`. | **NOT a warning, as this document suggested.** Two confirmed blockers: `lambda_m` is the whole channel array on the `ChannelSet.build -> overlap_gamma` hot path, so a scalar `if` on V raises; and under `filterwarnings=["error"]` a default warning turns ~10 working test files red. A query is the honest form; gated that no warning escapes a cladding-pumped solve. |
| F-10 | `n_t_m3 == 0` accepted, with the three divide-by-density sites guarded. Gated against analytic Beer-Lambert to 1e-6. | As suggested, plus the guards this document listed only as caveats. **The "18 further entry points confirmed finite" claim was unsupported** (2026-08-08 verification): no such gate existed, and `analyze_noise` on a passive span is NOT finite -- it returns `n_sp` NaN and `osnr_dB` +inf. Both are correct rather than broken (no inversion means no inversion factor; no ASE means no finite OSNR), and `nf_dB` correctly reduces to the attenuator's 1/G, so the resolution is a DOCUMENTED contract in `analyze_noise`'s docstring plus `TestF10PassiveFiber::test_the_noise_layer_on_a_passive_span_is_a_documented_contract`. What is now actually gated finite at `n_t = 0`, each on the same passive solve: `FiberAmplifier.solve` (`power_W`, `nbar2_z`), `output_ase_spectrum`, `thermal.heat_load_per_m`, `thermal.total_heat_W` (against the analytic attenuated power), `detection.detection_noise` (all three variances plus both SNRs), `simulate_transient` and `TransientResult.frame_as_steady`, `metastable_fraction` under upconversion, and `giles_calibrated_fiber`'s refusal. |
| F-11 | `erbium()`'s docstring now names the 1480 nm in-band pump and its quantum-defect advantage. | As suggested. |
| F-12 | `RareEarthIon.mccumber_eps_J` + `eps_J`, threaded through `sigma_e_mccumber`, `at_temperature` and `_mcc_matrix`. | **An additional defect was found while implementing:** `at_temperature` rebuilt the ion WITHOUT the new field, silently reverting a fitted eps on every T-scaled copy. Fixed and gated. |
| F-13 | Peak-relative endpoint denominator (`_ENDPOINT_FLOOR_FRAC = 1e-9`) in a shared `_relaxation_residuals()`. Yb reference at 0.243 mW converges in 118 iterations instead of never; gain 34.722738 dB against the old 34.722740. Adversarially attacked with fibers chosen to have a meaningful channel at a small endpoint (long reabsorbing Yb, counter- and bi-pumped, 40 m Er, a 200 nm ASE band): see the corrected bound at right. | **Fixed in all THREE solvers.** `eryb.py` and `transverse.py` held byte-identical copies, so the defect existed in triplicate and a single-solver fix would have desynchronised the iteration counts the cross-solver 1e-9 gates compare. Deduplicating follows X-3's precedent. **The "`tol=1e-11`/2000-iteration reference / worst 8e-5 dB" claim was wrong on both halves** (2026-08-08 verification): `tol=1e-11` is UNREACHABLE, because the sweeps run LSODA at `rtol=1e-7` and the inter-iterate residual therefore bottoms out there -- measured on the Yb reference, `tol` = 1e-8, 1e-9 and 1e-11 all stall at endpoint residual 6.405e-07 and exhaust `max_iter`, so the "reference" was max_iter-exhausted, never converged (`tol=1e-7` does converge, at 74 iterations). Re-measured against a reachable `tol=1e-7`/2000-iteration reference, the worst deviation among solves the flag calls CONVERGED is **1.06e-4 dB, on a 25 m reabsorbing Yb whose signal is dead at -143.4 dB**, and **5.2e-6 dB** across the physically meaningful cases (40 m Er 5.2e-6, 12 m reabsorbing Yb 1.4e-6, bi-pumped Yb 2.2e-7). The two cases that disagree by more (counter-pumped 0.12 dB, 40 m reabsorbing Yb 0.043 dB) are reported `converged=False` by both solves, i.e. they are not false convergence. |
| F-14 | `solve(relax=)` (skipped entirely at the default 1.0, byte-identity verified on three fibers) + `endpoint_residual`/`profile_residual`/`relax`/`min_power_W` on `meta`. | Suggestions 1 and 2 implemented; suggestion 3 (sweep continuation) NOT -- it changes `solve`'s contract and is performance, not correctness. **Scope is WIDER than this document recorded -- see below.** |

### F-14 is broader than recorded: a COUNTER-PROPAGATING PUMP fails at any signal level

Found by adversarial review after this document was written. The original finding blamed a
low-signal, strongly pumped configuration. The real trigger is counter-propagating power comparable
to co-propagating power, and a backward pump does it on its own. Measured on the same 3 um-core /
NA 0.12 / 6e25 m^-3 Yb fiber at 1060 nm with 20 mW of signal -- a perfectly ordinary drive:

```text
 config           relax  gain_dB   nbar2   converged  iters  endpoint_residual
 co-pumped         1.00    18.420  0.2035    True         4   3.8e-07
 bi-directional    1.00    17.438  0.2090    False      800   1.2e+00
 bi-directional    0.50    18.444  0.2043    True        50   8.9e-07
 counter-pumped    1.00   -30.834  0.3168    False      800   6.5e+06
 counter-pumped    0.50    24.007  0.1628    False     2000   1.0e+00
 counter-pumped    0.30    17.922  0.2012    True       128   9.8e-07
 counter-pumped    0.10    17.922  0.2012    True       315   4.3e-07
```

`relax <= 0.3` returns the SAME value at 0.3, 0.2, 0.15 and 0.1, so that is the fixed point; it is
physically consistent with the co-pumped result (17.9 vs 18.4 dB, as counter-pumping should be) and
conserves photons (signal added + ASE = 0.75 of the absorbed pump times the Stokes ceiling).
`relax = 1` is wrong by **49 dB** and `relax = 0.5` merely wanders. Co-pumping the same fiber
converges in 4 iterations, which isolates the cause to the counter-propagating coupling rather than
to a difficult amplifier.

Guidance now in `solve`'s docstring: `relax = 1` for co-pumped, `0.5` for bidirectional or lightly
seeded, `0.2-0.3` for counter-pumped -- and always read `meta['converged']` and
`meta['endpoint_residual']`, because here a non-converged solve is wrong by tens of dB rather than
by a rounding error. The repo's own `co / counter / bi` gate (`test_fiber_amp.py:723`) passes because
its Er fixture is far from this regime; **a counter-pumped high-gain Yb fixture would be a
worthwhile addition to the suite.**

---

## Post-implementation verification (2026-08-08)

An adversarial pass over the SHIPPED implementation confirmed every physics claim above and found
21 further defects in it. All are fixed; the ones that changed a number a user reads are gated.

**Wrong numbers (P1).**

| # | Site | Defect and fix |
|---|---|---|
| K1 | `detection.py` | `nf_beat_dB` moved with the photodiode's DARK CURRENT (+0.13 dB at 10 mA on the reference EDFA) -- an AMPLIFIER noise figure that depends on its detector, the error audit S3-10 removed. The NF's SNR now comes from a dark-free variance. |
| K2 | `detection.py` | Same, for the APD excess-noise factor `F` (+0.34 dB at F = 60; the avalanche GAIN `M` already cancelled bit-identically). `M` and `F` are both dropped from the NF's variance -- `M` cancels in `(M I)^2/(M^2 var)` anyway, `F` does not. The invariance gate now sweeps all four knobs; it previously swept only the two that were already invariant. |
| K3 | `dynamics.py` | `frame_as_steady().signal_gain_dB` divided by `plan.launched_W`, the CONFIGURED input -- exactly what `signal_drive` overrides. Under a 2x drive the reported gain was high by the drive ratio itself (+3.010504 dB against 10 log10 2 = 3.010300), and it propagated into `metrics.gain_flatness` (23.086 dB where the truth is 3.086). Now `P[i, 0]`. |
| K4 | `transverse.py`, `eryb.py` | The F-14 `relax=` remedy shipped in `steady_state` ONLY, while both other solvers run the SAME iteration. Measured counter-pumped Yb: `ResolvedFiberAmplifier` -30.105 dB `converged=False` against the mean-field twin's +17.891 dB -- a 48 dB gap with no knob. `relax` and the four `meta` diagnostics lifted into both; at `relax=0.3` the resolved solver converges to +17.952 dB, 0.0615 dB above the mean-field fixed point (positive, as a core-guided pump's TSHB should be). |

**Verification that did not verify (P2).**

| # | Site | Defect and fix |
|---|---|---|
| K5 | F-13 gate | Non-discriminating: the PRE-AUDIT stopping test also converges at its 20 mW / 201-node fixture, in 4 iterations. Moved to the audit's own 0.243 mW operating point (old: never converges in 400; new: 118 iterations at 34.722738 dB, both pinned). |
| K6 | `efficiency.py` | The power-balance "closure" recomputed the sum `total_heat_W` expands to -- X - X, 5.6e-17 W on a deliberately corrupted solve. Rewritten against the dissipation recomputed from the amplifier's own RATE EQUATIONS: healthy 1.183e-06 W of 0.305 W launched (O(dz^2) discretization), corrupted 1.746e-03 W, a factor 1476. Note threshold tightened 2e-2 -> 1e-3 relative. |
| K7 | F-6 per-level gate | Read both sides from the same `terms_A2` (circular), and its analytic flat term omitted `2 e (I_ase + I_dark) B_e` -- 336x its own stated `rel=1e-10`, surviving only on `pytest.approx`'s default `abs`. Replaced with a hand-written closed form at `rel=1e-10, abs=0` (matches to 1.7e-16), plus an explicit assertion that the incomplete form FAILS it. |
| K8 | F-10 gate / doc | See the corrected F-10 record row above. |
| K9 | `comms.py` | `max_pam_order` compared the SYMBOL error rate against `FEC_THRESHOLDS`, which are pre-FEC BIT error rates (G.975.1, OIF-400ZR). Wrong by log2(N) -- one full octave of constellation; 4 of 9 points on a 0-20 dB loss sweep moved. Now compares `r.ber`; the argument is `target_ber`. |
| K10 | `comms.py` | SER underflows to exactly 0.0 above Q ~ 37.5 while the module docstring advertised 1e-500 at Q = 50. Added `q_to_log10_ser` (`scipy.special.log_ndtr`); Q = 100 is log10 SER = -2173.871543. |
| K11 | `comms.py` | ENOB fed the PEAK-TO-PEAK swing ratio into the full-scale-SINE relation SNR = 6.02 N + 1.76 dB, overstating it by exactly 20 log10(2 sqrt 2)/6.02 = 1.5001 bits at every operating point. |

**Docs contradicted by measurement (P3).** `spectroscopy.py`: the gain-tilt table's nbar2 = 0.60
entry is 1566.3 nm, not 1531.8 -- the C-to-L crossover sits at nbar2 = 0.6028 (0.589-0.598 once a
Gamma(lambda) weighting is folded in), so the safe boundary is nbar2 >~ 0.61 and the previously
declared-safe 0.60 was ON it; the "0.1-0.5 dB" re-baselining figure is saturated-regime only
(1550 nm small-signal measures +2.96 dB, +2.68 dB at 100 uW, +0.03 dB at 5 mW); "pump bands move
< 1e-49 m^2" is exactly 0.0 at 976/980/1480/1490 nm but +4.4e-37 at 1500 and +5.6e-29 at 1520.
`steady_state.py`: the "byte-identical to pre-2026-08" claim covers the BLEND only, not the result,
because F-13 changed the stopping test in the same release; the `set_temperature_profile` docstring
said `eps = hc/zero_line` where the code reads `ion.eps_J`. `thermal.py`: the 3-5x sigma_e(T) slope
caveat is what the DEFAULT eps gives and is now tunable through `mccumber_eps_J` (F-12).
`waveguide.py`, `__init__.py`, `steady_state.py` and `thermal.py`: `Gamma_p = A_core/A_clad` should
read `A_dope/A_clad` in all four (S3-9 fixed the code, not the prose), and the Gaussian's Gamma
error at V = 8.2193 is 1.332% (0.979180 against the exact LP01 0.992394), not ~1.1%.
`transverse.py` and `fiber_amp_model_spec.md` carry the same 1.1% and are fixed with it.

**Latent, fixed with tests.** `detection_noise` now rejects `temperature_K < 0`, `load_ohm <= 0`
(0.0 was falsy and silently dropped the Johnson term asked for), `dark_current_A < 0`, and
non-finite `apd_gain`/`apd_excess_noise_F` (which walked past the `M <= 0 or F < 1` test, since
every comparison with NaN is False) -- each of them produced a negative variance or an `inf` SNR.
`simulate_transient(store_profiles=True)` refuses an `(Nt, K, Nz)` allocation above 2 GiB with the
shape in the message, before the initial solve and before any pages are committed; 6-12 GiB is
reachable from three innocuous-looking arguments, and past a point that stops being an exception.

---

## F-1 (high) -- the Er analytic cross-sections violate their own McCumber relation

### What was measured

`erbium()` builds `sigma_a` and `sigma_e` as independent sums of Gaussians
(`spectroscopy.py:104-115`), anchored at 980 / 1480 / 1530 / 1560 nm (absorption) and 1532 / 1560 nm
(emission). McCumber is an EXACT thermodynamic constraint linking the two for a thermalized
manifold, and the package already implements it (`RareEarthIon.sigma_e_mccumber`,
`spectroscopy.py`). Testing the model against its own McCumber prediction, at T = 300 K, with the
model's own `zero_line_m = 1.530e-6`:

```text
lam_nm   sigma_a       sigma_e(model)  sigma_e(McCumber)  model/McCumber
 1520    6.7162e-26    5.4625e-26      5.4646e-26          0.9996   <- anchor region, consistent
 1530    5.9309e-25    5.9166e-25      5.9309e-25          0.9976   <- anchor, consistent
 1534    ...                                               1.326
 1537    ...                                               1.735
 1540    1.2614e-25    3.1823e-25      1.5462e-25          2.058    <- WORST
 1541    ...                                               2.045
 1545    1.0490e-25    2.2786e-25      1.4221e-25          1.602
 1550    1.3485e-25    2.5675e-25      2.0207e-25          1.271    <- the C-band spec wavelength
 1555    1.5971e-25    2.9113e-25      2.6436e-25          1.101
 1560    1.6900e-25    3.0400e-25      3.0881e-25          0.9844   <- anchor, consistent
 1565    1.5970e-25    2.9111e-25      3.2193e-25          0.9043
```

The model is McCumber-consistent to <=2% **at its anchors** and violates its own relation by up to
**2.06x at 1540-1541 nm**, i.e. across the whole 1534-1556 nm window. This needs no external
datasheet to establish: it is an internal contradiction between two things the package itself
asserts.

Two further consequences follow from the Gaussian-sum construction:

1. **`sigma_a` has a spurious LOCAL MINIMUM at 1543 nm** (9.981e-26 m^2), sitting in the trough
   between the 1530 and 1560 Gaussians. Real Er3+ absorption in an aluminosilicate host decreases
   smoothly from the 1530 nm peak across the rest of the C band; it does not turn back up at
   1543 nm. The interpolation shape, not just the magnitude, is wrong.

2. **A spurious gain / efficiency ripple in the C band.** Measured on the standard 1.4 um-core /
   NA 0.23 / 1e25 m^-3 / 8 m EDFA at 300 mW of 976 nm pump and 5 mW of signal (deep saturation),
   sweeping only the signal wavelength:

   ```
   lam_nm   gain_dB   P_out_W   PCE      NF_dB   OSNR_dB
    1540     15.61     0.1820   0.5902    4.23    60.68
    1545     15.09     0.1616   0.5219    3.67    61.29   <- non-physical local minimum
    1550     15.25     0.1675   0.5415    3.27    61.74
    1555     15.38     0.1725   0.5583    3.27    61.78
    1560     15.39     0.1731   0.5603    3.30    61.80
   ```

   A 6.8 pp PCE dip and a 0.5 dB gain dip at 1545 nm, non-monotonic in wavelength, with no physical
   cause. This lands directly on the functions built to characterize exactly this: `gain_spectrum`,
   `gain_flatness`, and `GainSpectrum.tilt_dB_per_nm`.

3. **The package's own acceptance target sits inside the defect.**
   `calibration.py:101-108` sets `EDFA_CBAND_TARGETS["signal_nm"] = 1550.0` with
   `small_signal_gain_dB = 30.0` and `nf_dB_max = 5.5`, so `calibration_report` is scored at the
   one wavelength where the model is 27% off its own McCumber relation. A fiber calibrated to hit
   30 dB at 1550 nm against this model will be mis-parameterized.

Note the saturated-regime impact on total output power is modest (0.542 vs 0.560 PCE, 1550 vs 1560)
because deep saturation is pump-limited rather than cross-section-limited. The defect matters most
for **small-signal gain shape, gain flatness / tilt, WDM channel-to-channel equalization, and any
calibration anchored at 1550 nm** -- which is most of what the C band is used for.

### Suggested implementations, cheapest first

**(a) Derive `sigma_e` from `sigma_a` by McCumber inside the factory (recommended).** Replace the
independent emission Gaussian sum with the constraint the package already trusts:

```python
# spectroscopy.py, in erbium()
class _McCumberEmission:
    """sigma_e(lambda) = sigma_a(lambda) exp[(eps - h nu)/kT] -- one spectrum, not two."""
    def __init__(self, sigma_a, eps_J, T_K=300.0):
        self._sa, self._eps, self._T = sigma_a, eps_J, T_K
    def sigma(self, lambda_m):
        nu = C_LIGHT / np.asarray(lambda_m, float)
        return self._sa.sigma(lambda_m) * np.exp((self._eps - H_PLANCK * nu) / (KB * self._T))
```
This makes the violation identically zero by construction and removes two free anchors. Cost: it
propagates any `sigma_a` shape error into `sigma_e` with an exponential lever arm, so it is only
safe if (b) is also done. It also changes numbers everywhere, so it must be a versioned change with
the gate values in `fiber_amp_model_spec.md:143-164` re-baselined.

**(b) Add C-band `sigma_a` anchors so the trough disappears.** The minimum-change fix. Two extra
Gaussians at ~1545 and ~1550 nm, with peak values chosen so that `sigma_a` is monotone decreasing
over 1533-1565 nm and `sigma_e/sigma_a` tracks McCumber to within a stated tolerance. This is
purely a re-fit of existing free parameters -- no API change, no new module. A discrimination gate
falls out for free:

```python
def test_er_cross_sections_are_mccumber_consistent():
    """The two spectra are thermodynamically linked; the analytic model must not contradict its own
    sigma_e_mccumber by more than the fit tolerance anywhere in the C band (audit 2026-08-04 F-1:
    the Gaussian-sum model was off by 2.06x at 1540 nm and had a spurious sigma_a minimum at
    1543 nm)."""
    er = erbium()
    lam = np.arange(1.530e-6, 1.566e-6, 1e-9)
    ratio = er.sigma_e.sigma(lam) / np.array([er.sigma_e_mccumber(l) for l in lam])
    assert np.all(np.abs(np.log(ratio)) < np.log(1.15))          # 15% band
    sa = er.sigma_a.sigma(lam)                                    # monotone: no interior trough
    assert np.all(np.diff(sa) < 0.0)
```

**(c) Ship a tabulated measured C-band spectrum.** `CrossSectionTable` and
`ion_from_cross_sections` already exist and duck-type `CrossSectionModel`, but the repo ships **zero
spectra** -- both are empty containers. One published Er:aluminosilicate `alpha(lambda)` /
`g*(lambda)` pair, added as `spectroscopy.erbium_measured()`, would make the accurate path the
default for C-band work and leave `erbium()` as the fast analytic approximation. This is the
highest-value option and the only one that fixes the shape rather than the consistency.

**(d) At minimum, document it.** Add to the `erbium()` docstring: the anchor wavelengths are
980 / 1480 / 1530 / 1560 nm; the model is anchor-exact there and interpolates between; 1535-1555 nm
is interpolation and 1550 nm specifically carries a ~27% McCumber inconsistency; prefer 1560 nm for
gates or calibrate. And either move `EDFA_CBAND_TARGETS["signal_nm"]` to 1560.0 or note why 1550
is kept.

---

## F-2 (medium) -- `simulate_transient` throws away the ASE it already computed

`dynamics.py:251` computes the complete frozen-inversion power matrix for every channel at every
time step:

```python
P = _propagate_fixed(z, g, s, bc, u)          # (K, Nz) -- ALL channels, ASE included
```

The ASE columns are then used only for the read-only validity monitor
(`dynamics.py:255-268`) and discarded. The extraction loop keeps signals and pumps only:

```python
for j, i in enumerate(sig_idx):
    sig_out[it, j] = P[i, -1]
    gain_dB[it, j] = 10.0 * np.log10(P[i, -1] / max(bc[i], 1e-300))
for j, i in enumerate(pmp_idx):
    pmp_out[it, j] = P[i, -1] if u[i] > 0 else P[i, 0]
```

Consequences for a consumer:

* There is no time-resolved ASE, ASE PSD, OSNR, or NF from a transient. `analyze_noise`,
  `output_ase_spectrum`, `noise_figure` and `detection_noise` all take a `SteadyStateResult` and
  there is no adapter, so **noise during a burst transient is simply not obtainable** through the
  public API.
* Building a burst-mode link therefore requires reimplementing `_propagate_fixed`, the spontaneous
  source `s_k = Gamma_k n_t sigma_e,k nbar2 m h nu_k dnu_k`, and the channel plan -- i.e. duplicating
  the package's own solver in consumer code. (This audit did exactly that and validated it back
  against `solve()`; it should not have been necessary.)
* The information is free. It is already in memory and being thrown away.

### Suggested implementation

Two additive fields plus an adapter; no physics changes, and the existing arrays stay
byte-identical.

```python
@dataclass
class TransientResult:
    ...
    ase_fwd_W: np.ndarray = None       # (Nt, n_ase_bins) forward ASE per bin at z = L
    ase_bwd_W: np.ndarray = None       # (Nt, n_ase_bins) backward ASE per bin at z = 0
    ase_lambda_m: np.ndarray = None    # (n_ase_bins,)
    ase_dnu_hz: np.ndarray = None      # (n_ase_bins,) -- needed for the per-pol PSD
    power_zt: np.ndarray = None        # (Nt, K, Nz), OPT-IN via store_profiles=True (memory)
```

filled in the same loop that already reads `P`:

```python
if af.size:                                   # af = np.where(is_ase & (u > 0))[0], hoisted
    ase_fwd[it] = P[af, -1]
if ab.size:
    ase_bwd[it] = P[ab, 0]
```

and, so the whole existing noise layer works per frame without modification:

```python
def frame_as_steady(self, it: int) -> "SteadyStateResult":
    """The it-th frame as a SteadyStateResult, so noise.analyze_noise / detection.detection_noise
    can be applied to a transient frame unchanged. Requires store_profiles=True. This is the
    quasi-static reading of the frame -- exact in the same sense the march itself is (transit time
    << tau), NOT a claim that the frame is a steady state of the drive."""
```

The docstring must carry that last caveat explicitly, and `meta['quasi_static_valid']` should be
propagated onto any derived `SteadyStateResult` so a consumer cannot lose the A-7 flag.

---

## F-3 (medium) -- the channel plan has no public accessor

`FiberAmplifier._plan()` (`steady_state.py:221`) returns `(ChannelSet, bc, u, is_ase, kind)`: the
wavelength grid, the launched powers, the propagation directions, the ASE mask and the channel
kinds. It is the natural interface for anything that wants to consume the amplifier's channel
structure -- and it is private. `ChannelSet` and `ChannelSet.build` ARE public and exported from the
facade, which makes the omission look accidental rather than deliberate: the pieces are public but
the assembly is not.

Within the package, `dynamics.py:162` reaches into it (fine -- same package). Outside, a consumer
must re-derive the ASE bin edges, the wavelength-to-frequency bin-width conversion
(`dnu = |c/edge_i - c/edge_{i+1}|`, easy to get wrong as a wavelength width), the fwd/bwd channel
duplication, and the cladding-pump `gamma` override. That duplicate then silently rots whenever
`_plan` changes.

### Suggested implementation

A thin public alias returning a frozen record, with `_plan` kept as the internal hot path:

```python
@dataclass(frozen=True)
class ChannelPlan:
    channels: ChannelSet
    launched_W: np.ndarray            # (K,) boundary powers: fwd at z=0, bwd at z=L
    direction: np.ndarray             # (K,) +1 / -1
    is_ase: np.ndarray                # (K,) bool
    kind: List[str]                   # (K,) 'pump' | 'signal' | 'ase' | 'stokes'

def channel_plan(self) -> ChannelPlan:
    """The amplifier's channel structure -- PUBLIC. Same content _plan() returns internally;
    exposed because an external consumer that needs the ASE bin frequencies or the fwd/bwd layout
    otherwise has to reimplement the bin-edge and dnu bookkeeping (audit 2026-08-04 F-3)."""
    ch, bc, u, is_ase, kind = self._plan()
    return ChannelPlan(ch, bc, u, is_ase, list(kind))
```

`ErYbAmplifier` and `ResolvedFiberAmplifier` both have their own `_plan` with different arities
(`eryb.py:194`, `transverse.py:617`), so the public method should be added per class or defined by
the same protocol as `with_signals` / `with_pumps` / `without_ase` -- consistent with the audit-A-3
lesson that made those three public in the first place.

---

## F-4 (medium) -- `slope_efficiency`'s Stokes ceiling ignores all but the first pump

`metrics.py:200-204`:

```python
lam_p = amp.pumps[0].lambda_m
...
return SlopeEfficiency(pump, s_out, slope, thr, stokes_limit(lam_p, lam_s))
```

With a single pump this is correct. With two pump wavelengths it is not, and dual-wavelength
pumping is standard practice in both ions this package targets: 915 + 976 nm for Yb (976 for
efficiency, 915 to escape the zero-line inversion clamp that `test_yb_zero_line_pump_caps_inversion`
pins at `nbar2 < 0.55`), and 980 + 1480 nm for Er. The returned `stokes_limit` field then describes
a ceiling that does not apply to the swept configuration, and since `slope_efficiency`'s whole
documented purpose is "the slope tends toward the Stokes ceiling", the comparison silently breaks.
**Measured on a real configuration**, not hypothesised. A Yb amplifier (3 um core, NA 0.12,
6e25 m^-3, 4 m) at 1060 nm, pumped by 915 nm and 976 nm at equal power, swept 0.2-4.0 W with a
saturating 20 mW signal:

```text
reported stokes_limit (from pumps[0] = 915 nm) : 0.8632
correct photon-weighted ceiling                : 0.8911
stokes_limit(976 nm, 1060 nm)                  : 0.9208
measured slope                                 : 0.8660   <- EXCEEDS the reported ceiling
```

The returned `SlopeEfficiency` thus advertises a slope above its own quantum-defect ceiling -- an
apparent violation of energy conservation that is purely a bookkeeping artifact, and one that
would read as a physics bug to anyone checking the invariant. Swapping the pump order changes the
reported ceiling to 0.9208 and the apparent violation disappears, which is the tell.

### Suggested implementation

The correct ceiling for a multi-wavelength pump is set by the PHOTON-weighted mean pump wavelength,
because the defect is per photon converted:

```python
def _effective_pump_lambda_m(amp) -> float:
    """Photon-flux-weighted mean pump wavelength: the Stokes ceiling for a multi-wavelength pump is
    (sum P_i) / (sum P_i / lambda_i), since each pump photon converts to at most one signal photon
    (audit 2026-08-04 F-4: the ceiling used pumps[0] alone, so a 915+976 nm Yb pump reported a
    ceiling its own measured slope could exceed)."""
    num = sum(p.power_W for p in amp.pumps)
    den = sum(p.power_W / p.lambda_m for p in amp.pumps)
    return num / den if den > 0.0 else float("nan")
```

Gate: two pumps at 915 and 976 nm with equal power must give a ceiling strictly between
`stokes_limit(915e-9, lam_s)` and `stokes_limit(976e-9, lam_s)`, and a single-pump amplifier must
return exactly today's value (byte-identity on the existing path).

Worth checking the same pattern elsewhere: `wall-plug`-style aggregate quantities computed from
`pumps[0]` rather than the whole list. `thermal.quantum_defect_fraction` takes an explicit
`pump_lambda_m` so the caller owns the choice there -- that is the right shape.

---

## F-5 (medium) -- no receiver-electronics noise anywhere in the repo

`amp_noise.beat_noise_variances` gives the three OPTICAL variances (shot, signal-spontaneous,
spontaneous-spontaneous) and `BeatNoiseResult.var_total` is documented as exactly their sum. There
is no Johnson/thermal term, no transimpedance-amplifier input-referred current noise, and no APD
excess-noise factor `F(M)` anywhere in the tree (zero hits for `Johnson`, `thermal_noise`, `R_load`,
`transimpedance`, `TIA`, `receiver_sensitivity`).

That makes `snr_elec_dB` an **upper bound presented as a value**. For an amplified link the omission
is often benign -- a preamplified receiver is designed to be signal-spontaneous-beat limited -- but
it is exactly wrong in the two regimes a user is most likely to probe: low received power (after a
span or a 1:N fan-out) and high electrical bandwidth. A 12 pA/rtHz TIA at 7.5 GHz contributes
`sigma_th = 1.04 uA`, which dominates every optical term below roughly -10 dBm received. Nothing in
the API hints that the number is incomplete.

### Suggested implementation

Additive and default-off, so every existing result stays byte-identical:

```python
def detection_noise(result, signal_lambda_m, *, optical_bw_Hz, electrical_bw_Hz,
                    quantum_efficiency=1.0, responsivity_A_W=None, m_modes=None,
                    dark_current_A=0.0,                       # fixes F-8 at the same time
                    tia_current_noise_A_rtHz=0.0, load_ohm=None, temperature_K=T_REF,
                    apd_gain=1.0, apd_excess_noise_F=1.0):
```

with

```python
    var_thermal = 0.0
    if tia_current_noise_A_rtHz > 0.0:
        var_thermal += tia_current_noise_A_rtHz ** 2 * B_e
    if load_ohm:
        var_thermal += 4.0 * KB * temperature_K * B_e / load_ohm
```

and a new `var_thermal` field on `BeatNoiseResult` (0.0 by default, so `dominant_term` keeps its
current three-way behaviour until a thermal term is actually supplied). The APD path multiplies the
shot term by `apd_gain**2 * apd_excess_noise_F` and the beat terms by `apd_gain**2`; leaving both at
1.0 is the PIN case and reproduces today's numbers exactly. `KB` and `T_REF` are already in
`constants.py`.

The docstring should state plainly that `var_total` with all the new arguments at their defaults is
the **optical-noise-limited bound**, not a receiver prediction.

---

## F-6 (medium) -- no communications layer

Confirmed absent as any code identifier, repo-wide: `PAM`, `pam4`, `baud`, `symbol` (as a signal
concept), `eye_diagram`, `BER`, `bit_error`, `prbs`, `Q_factor`, `extinction_ratio`, `OOK`, `NRZ`,
`EVM`, `modulation_format`, `bitrate`, `ISI`. (`analysis.py:599-618`'s `q_factors` are resonator Q;
`validation/fdtd_lasing_cavity.py`'s "burst" is an FDTD seed pulse.) The only place the concept
appears is as a REQUIREMENT: `docs/DynaMeta_QD_SOA_extension_spec.md:940` asks for "a true
symbol-stream SNDR/EVM", and `:41` contrasts the QD-SOA use case against "telecom on/off-keying or
PAM4".

This is a legitimate scope boundary -- but it is worth noting that the amplifier package is
one thin module away from being able to answer link-level questions, and that the thin module has
no physics in it at all. Everything hard is already present.

### Suggested implementation: `fiber_amp/comms.py`, ~150 lines, pure numpy, no new physics

```python
def pam_levels_W(order, mean_power_W, extinction_ratio):
    """N levels equally spaced in OPTICAL POWER with mean exactly mean_power_W and
    P_max/P_min = extinction_ratio: P_min = 2 P_mean/(1+ER), P_max = 2 P_mean ER/(1+ER)."""

def ml_thresholds(mu, sigma):
    """ML decision thresholds between adjacent Gaussian levels of UNEQUAL variance -- the case
    here, since shot and signal-spontaneous beat noise are both linear in the level power. Solves
    (x-mu_k)^2/2 s_k^2 + ln s_k = (x-mu_j)^2/2 s_j^2 + ln s_j and takes the root between the two
    means; degenerates to the midpoint when s_k = s_j."""

def pam_ser(mu, sigma, thresholds=None):
    """Symbol error rate of an equiprobable PAM constellation with per-level Gaussian noise.
    Interior levels err across two thresholds, outer levels across one."""

def link_performance(result, signal_lambda_m, *, order, baud_hz, mean_power_W, ...):
    """Per-level sigma from ONE call to analyze_noise (for rho_sp) plus ONE
    beat_noise_variances call PER LEVEL -- never by scaling var_total, since spont-spont is
    level-independent while shot and sig-spont are not."""
```

The one substantive design note, because it is easy to get wrong: **the per-level variances must be
evaluated per level.** `BeatNoiseResult.var_total` is computed at a single mean output power;
rescaling it by the level power is wrong for `spont_spont` (level-independent) and for the ASE and
dark contributions to `shot`. Call `beat_noise_variances(P_k, rho_sp, ...)` in a loop over levels.

A burst-mode PAM harness of exactly this shape, built on the package from outside, now lives at
`D:\Metacept\Neurophos\Python_Test_Scripts\Fiber_Amplifiers\scripts\fiber_burst_pam.py`; it can be
read as a reference implementation for F-2/F-3/F-5/F-6/F-7 should any of them be taken up.

---

## F-7 (medium) -- no electrical / wall-plug layer

The efficiency story stops at optical-optical. The complete inventory:
`power_conversion_efficiency` = `(P_sig_out - P_sig_in) / P_pump_launched` (`metrics.py:163`);
`slope_efficiency` = `dP_sig_out/dP_pump`; `stokes_limit` = `lambda_p/lambda_s`;
`quantum_defect_fraction` = `1 - lambda_p/lambda_s`; `total_heat_W` = `F(0) - F(L)`. There is no
pump-diode wall-plug efficiency, no pump-to-fiber coupling loss, no controller or TEC overhead, no
electrical power, and no energy-per-bit -- zero repo-wide hits for `wall_plug`, `wallplug`,
`electrical_efficiency`, `eta_wp`, `diode_efficiency`.

For anyone sizing a real amplifier this is the number that matters, and the gap is a genuine
surprise given how carefully the optical side is accounted. It is also the layer where the
quantities the package DOES provide become decisive: `total_heat_W` closes the energy balance
independently, and `stokes_limit` bounds the achievable result.

### Suggested implementation: `fiber_amp/efficiency.py`

```python
@dataclass(frozen=True)
class PumpDiode:
    """Electrical characterization of one pump source. wallplug_efficiency is optical W out of the
    diode facet per electrical W in (0.40-0.55 typical for 9xx nm multimode diodes);
    coupling_efficiency is everything between facet and doped core (combiner, splices)."""
    wallplug_efficiency: float
    coupling_efficiency: float = 1.0

def wall_plug_efficiency(amp, result, diodes, *, controller_W=0.0,
                         tec_fraction_of_pump_heat=0.0, signal_source_wpe=None):
    """Electrical-to-useful-optical efficiency. Reports the whole chain so the binding loss is
    visible: diode WPE -> pump coupling -> pump ABSORPTION (residual pump is a real loss and
    power_conversion_efficiency already keeps it in the denominator) -> quantum defect
    (stokes_limit, the hard ceiling) -> extraction into the signal vs into ASE and unextracted
    inversion. Cross-check: optical in minus optical out must equal total_heat_W(result) to within
    the untracked out-of-band spontaneous fraction."""
```

Three definitional decisions the module should make explicitly rather than leave to the caller,
since each is a place a budget can be quietly flattered:

1. **Numerator: added signal power (`P_out - P_in`), not `P_out`.** Consistent with
   `power_conversion_efficiency`, and it stops the seed laser's power from being counted as
   amplifier output. Report the gross figure separately if wanted.
2. **Residual pump stays in the denominator.** It was paid for electrically. An
   "absorbed-pump" efficiency is a useful diagnostic but is not a wall-plug number.
3. **Duty cycle.** A CW-pumped amplifier serving a bursty load pays for the pump continuously
   while delivering signal only during bursts. The time-average must be over the whole span, with
   an explicit flag for a pump gated to the burst.

`total_heat_W`'s documented bias belongs in the docstring: spontaneous emission outside the
resolved `AseBand` is invisible to the flux balance and is therefore counted AS heat, a conservative
overestimate that tightens as `AseBand` widens.

---

## F-8 (low) -- dark current is unreachable through `detection_noise`

`amp_noise.beat_noise_variances` takes `I_dark_A=0.0` and uses it correctly
(`amp_noise.py:46`: `shot = 2 Q_E (I_sig + I_ase + I_dark) B_e`), but `detection.py:90` calls it
without the argument:

```python
v = beat_noise_variances(P_sig_out, rho_sp, responsivity_A_W=R, electrical_bw_Hz=B_e, ...)
```

so the parameter exists but cannot be reached from the `fiber_amp` public API -- a consumer has to
bypass `detection_noise` and call `amp_noise` directly. Fold the fix into F-5's signature change
(`dark_current_A=0.0`, forwarded as `I_dark_A`); default 0.0 keeps every current result identical.

---

## F-9 (low) -- `mode_field_radius_m`'s stated validity range is never checked

`waveguide.py:23` documents "Valid ~ 1.2 < V < 2.4 (single-mode)" and the only guard in the code is
`V = max(V, 1e-6)` (`waveguide.py:27`), a divide-by-zero clamp. Nothing warns outside the range,
and `overlap_gamma` -- hence every cross-section-times-overlap product and the saturation integral
-- is built on it.

This is not hypothetical: the LMA and cladding-pumped Yb fixtures throughout the repo
(`a = 5 um`, `NA = 0.06-0.08`, e.g. `tests/test_fiber_dynamics.py:36-42`,
`tests/test_fiber_thermal_feedback.py:20-25`, `tests/test_fiber_srs.py:87-95`) all run at V well
above 2.4, and `fiber_amp_model_spec.md:287-290` quantifies the consequence at V = 8.2 as a 1.1%
error in `Gamma` but **up to 13% (0.85 dB/m) in the saturation integral**. The package has the right
answer for that regime -- `transverse.ResolvedFiberAmplifier` and `lma.solve_lp_modes` -- but
nothing points a user from the silent-error path to it.

### Suggested implementation

```python
    if not (1.2 <= V <= 2.405):
        warnings.warn(
            "mode_field_radius_m: V = {:.3f} is outside the Marcuse fit's stated validity "
            "1.2 < V < 2.4; Gamma is ~1% off but the SATURATION INTEGRAL can be off by ~13% "
            "(0.85 dB/m at V = 8.2, fiber_amp_model_spec sec. 'transverse'). Use "
            "transverse.ResolvedFiberAmplifier or lma.solve_lp_modes for a multimode core."
            .format(V), RuntimeWarning, stacklevel=2)
```

Caveat for whoever implements it: `pyproject.toml` sets `filterwarnings = ["error", ...]`, so this
WILL fail every existing LMA test until each one either widens its tolerance intentionally or is
switched to the resolved solver. That is arguably the finding's real value -- it makes an
already-known approximation visible at the call site -- but it is not a one-line change in
practice. A `strict=False` flag, or emitting the warning once per distinct V, are gentler variants.

---

## F-10 (low) -- `FiberSpec` cannot describe a passive fiber

`waveguide.py:51` requires all of `core_radius_m`, `na`, `n_t_m3`, `length_m` to be `> 0`. Modelling
a passive span (a transmission fiber, a pigtail, the fiber between two stages of an
`AmplifierChain`) therefore needs a sentinel dopant density; the repo's own workaround is
`n_t_m3=1.0` (`tests/test_fiber_srs.py:22`). Relaxing the check to `n_t_m3 >= 0` costs nothing --
`n_t = 0` makes every gain and spontaneous-source term identically zero, which is the correct
physics for an undoped fiber -- and removes a hack that currently has to be rediscovered.

---

## F-11 (low) -- the `erbium()` docstring omits the 1480 nm anchor

The code carries `(1.480e-6, 0.040e-6, 0.8e-25)` in `sigma_a` (`spectroscopy.py:106`), i.e. the
model DOES support in-band 1480 nm pumping -- the standard choice for a high-power Er booster,
because its smaller quantum defect (1480/1550 = 0.955 vs 980/1550 = 0.632) raises the efficiency
ceiling by half. The factory docstring's summary of pump bands does not mention it, so the
capability is invisible unless the source is read. One-line docstring fix.

---

## F-12 (informational) -- the Yb McCumber reference energy

Applying the same self-consistency test as F-1 to `ytterbium()`, with `eps = hc/zero_line_m` and
`zero_line_m = 976 nm`:

```text
lam_nm   sigma_a       sigma_e(model)  sigma_e(McCumber)  model/McCumber
  976    2.7034e-24    2.6515e-24      2.7034e-24          0.981
 1000    2.9854e-26    8.6646e-26      9.7091e-26          0.892
 1030    8.1000e-26    3.1109e-25      1.0649e-24          0.292
 1060    2.9854e-26    1.9461e-25      1.4660e-24          0.133
 1080    5.0625e-27    5.3363e-26      5.7463e-25          0.093
```

**This is not the same kind of finding as F-1 and should not be treated as one.** The model's
absolute values at 1060 nm (`sigma_a = 2.99e-26`, `sigma_e = 1.95e-25`) are physically reasonable;
what fails is the McCumber relation with `eps` taken as the 976 nm absorption PEAK. `eps` is
properly the mean transition energy between the thermalized Stark manifolds, which for Yb sits
below the absorption peak; using the peak inflates `(eps - h nu)/kT` and hence the predicted
`sigma_e` exponentially at long wavelengths. Real Yb spectra are not McCumber-consistent against
the peak either.

The reason it is worth logging: **this is the same root cause as an already-documented limitation.**
`thermal.py` notes that "the pure-McCumber rescaling used here gives d ln sigma_e/dT =
-(eps - h nu)/(k T^2) ~ -0.9 to -1.4 %/K at Yb 1030-1064 nm, an UPPER BOUND ~3-5x the measured NET
slopes (~-0.1 to -0.3 %/K)". Both the inflated `sigma_e(T)` slope and the factor-11 ratio mismatch
above come from the same over-large `eps - h nu`. So there is one fix for two symptoms:

```python
@dataclass(frozen=True)
class RareEarthIon:
    ...
    mccumber_eps_J: Optional[float] = None
    """Mean transition energy for the McCumber relation. None -> h c / zero_line_m (today's
    behaviour, byte-identical). Setting it to a FITTED value -- properly the free-energy difference
    between the thermalized manifolds, which lies below the absorption peak -- fixes both the
    sigma_e/sigma_a ratio at long wavelengths and the ~3-5x overstated d ln sigma_e/dT that
    thermal.py documents (audit 2026-08-04 F-12)."""
```

`sigma_e_mccumber` already accepts an `eps_J` override, so the change is to thread the field through
`at_temperature`, `FiberAmplifier._mcc_matrix` (`steady_state.py:307`) and `dynamics`' McCumber
path, all of which currently hard-code `H_PLANCK * C_LIGHT / self.ion.zero_line_m`. With the field
defaulting to `None` every existing result is unchanged.

---

## F-13 (high) -- `meta['converged']` is a permanent false negative on efficient amplifiers

### What was measured

The relaxation loop's stopping test (`steady_state.py:441-451`):

```python
out = np.concatenate([P_fwd[:, -1], (P_bwd[:, 0] if bwd.size else [])])
prof = np.concatenate([P_fwd, P_bwd], axis=0) if bwd.size else P_fwd.copy()
if last_out is not None:
    denom = np.maximum(np.abs(out), 1e-15)
    end_ok = float(np.max(np.abs(out - last_out) / denom)) < tol
    ch_peak = np.maximum(np.max(np.abs(prof), axis=1, keepdims=True), 1e-300)
    prof_ok = float(np.max(np.abs(prof - last_prof) / ch_peak)) < tol
    if end_ok and prof_ok:
        converged = True
```

The two halves of this test are NOT consistent with each other. `prof_ok` normalizes each channel by
**its own peak power** -- correct, scale-aware, and it works. `end_ok` normalizes by
**an absolute 1e-15 W floor** -- and that is what fails.

Instrumenting the loop on the Yb reference (3 um core / NA 0.12 / 6e25 m^-3 / 4 m, 2 W at 976 nm,
0.243 mW at 1060 nm, 24 ASE bins over 1020-1100 nm), printing both residuals and the channel that
maximizes each:

```text
 iter   end_resid   prof_resid   worst_end_ch      worst_prof_ch       gain_dB
    1   1.032e+01   4.154e+00    ase  1031.7nm     ase  1035.0nm      31.07043
    2   8.800e-01   8.042e-01    ase  1031.7nm     ase  1031.7nm      36.23610
    5   1.307e+00   8.358e-01    ase  1031.7nm     ase  1031.7nm      33.68999
   10   2.334e-01   1.849e-01    ase  1031.7nm     ase  1031.7nm      34.99601
   20   2.553e-02   1.960e-02    ase  1031.7nm     ase  1031.7nm      34.75050
   30   2.520e-03   1.927e-03    ase  1031.7nm     ase  1031.7nm      34.72545
   40   2.253e-03   1.839e-04    pump  976.0nm     ase  1031.7nm      34.72300   <-- switches
   50   9.415e-03   1.941e-05    pump  976.0nm     ase  1031.7nm      34.72276   <-- and GROWS
```

The physics converges cleanly: the gain oscillates with decaying amplitude to 34.7227 dB and
`prof_resid` falls monotonically through 1.9e-5, heading for the 1e-6 tolerance. But at iteration
~40 the *endpoint* residual stops falling, starts **growing**, and the channel responsible switches
from an ASE bin to **the 976 nm pump**. Inspecting that channel:

```text
pump at z=0 :  2.000000e+00 W   (launched)
pump at z=L : -1.214900e-16 W   (residual)   <-- NEGATIVE
attenuation :  3003.0 dB
```

The pump is **fully absorbed** -- which is the design goal of an efficient amplifier -- so its
endpoint value is pure integrator noise at the 1e-16 level, and here it is even negative.
`np.abs(-1.215e-16) = 1.215e-16` is *below* the 1e-15 floor, so `denom = 1e-15`, and the test
computes (a ~1e-16 noise difference) / (1e-15) ~ 0.1, permanently, against a tolerance of 1e-6.

Confirmation that iteration count is irrelevant:

```text
max_iter   200 -> gain 34.723 dB, converged=False, iters=200   (10.4 s)
max_iter  1000 -> gain 34.723 dB, converged=False, iters=1000  (52.2 s)
max_iter  3000 -> gain 34.723 dB, converged=False, iters=3000  (156.9 s)
```

Identical answers, `converged` never True, 15x the runtime. And the gain is stable from iteration
60 onward, so **~140 of the 200 default iterations are pure waste** on this problem.

### Why this is high severity

1. It fires precisely on the GOOD case. The better the pump absorption -- the thing every design
   optimizes for -- the more certainly the flag lies. On the Er reference (39 mW of residual pump
   out of 300 mW) the flag is fine; on the fully-absorbed Yb reference it is never True below
   20 mW of signal input.
2. The library instructs users to rely on it: `solve()` does not raise on non-convergence, and the
   documented contract is that the caller checks `meta['converged']`. Every gate in the repo does.
   A flag that is false-negative in the most desirable operating regime trains users to ignore it,
   which then hides the case where it is telling the truth (F-14 below is exactly that case).
3. It silently costs 3-15x the solve time, and `metrics.saturation_output_power` /
   `gain_spectrum` / `gain_compression_curve` each run 25+ full solves.

### Suggested implementation

Make the endpoint test consistent with the profile test that already works -- a per-channel,
peak-relative floor rather than an absolute one:

```python
    ch_peak_end = np.maximum(np.max(np.abs(prof), axis=1), 1e-300)   # same peaks prof_ok uses
    # Floor the denominator at a small fraction of each channel's OWN peak. A channel attenuated
    # far below that (a fully absorbed pump, whose endpoint is integrator noise and can even be
    # negative) then contributes ~0 to the residual instead of ~0.1 forever (audit 2026-08-04
    # F-13: the absolute 1e-15 floor made `converged` a permanent false negative on every
    # amplifier with full pump absorption, at 3-15x the necessary runtime).
    denom = np.maximum(np.abs(out), 1e-9 * ch_peak_end)
    end_ok = float(np.max(np.abs(out - last_out) / denom)) < tol
```

`1e-9` is three decades below `tol = 1e-6`, so a channel carrying real information is still fully
tested; only channels that have decayed nine decades below their own peak are neutralized.

Two gates worth adding:

```python
def test_converged_flag_is_true_for_a_fully_absorbed_pump():
    """A fully-absorbed pump's endpoint is integrator noise; it must not veto convergence
    (audit 2026-08-04 F-13)."""
    fib = FiberSpec(3.0e-6, 0.12, 6.0e25, 4.0)
    amp = FiberAmplifier(ytterbium(), fib, [Pump(2.0, 0.976e-6)],
                         [Signal(20e-3, 1.060e-6)], AseBand(1.02e-6, 1.10e-6, 24))
    r = amp.solve(n_nodes=201)
    assert r.power_W[0, -1] < 1e-12 * r.power_W[0, 0]      # pump really is fully absorbed
    assert r.meta["converged"] and r.meta["iterations"] < 120

def test_converged_flag_tracks_the_actual_fixed_point():
    """The flag must agree with a DIRECT fixed-point measurement: re-propagating the solve's own
    nbar2(z) must reproduce its own channel powers."""
```

An independent, cheap fixed-point measure -- useful as the gate's oracle and as a user-facing
diagnostic -- is to take the solve's converged `nbar2(z)`, re-propagate every channel through it
with the exact integrating-factor solution, and compare endpoints (excluding channels whose
endpoint is a negligible fraction of their own peak). Measured on the Yb reference, this cleanly
separates the false negatives from the real failure of F-14:

```text
 P_in(mW)   gain_dB   pkg_flag   fp_resid_max   gain_resid_dB   verdict
    0.050    -12.56   False      8.496e-01      4.723e+00       genuinely broken (F-14)
    0.100    -10.47   False      6.099e-01      2.653e+00       genuinely broken (F-14)
    0.243     34.72   False      4.994e-03      2.059e-05       CONVERGED, flag is wrong
    1.000     30.71   False      5.339e-03      9.150e-06       CONVERGED, flag is wrong
    5.000     24.47   False      5.452e-03      1.427e-05       CONVERGED, flag is wrong
   20.000     18.72   True       5.480e-03      9.573e-06       converged, flag agrees
  100.000     12.13   True       5.523e-03      9.649e-06       converged, flag agrees
```

(Implemented as `steady_fixed_point_residual` in
`D:\Metacept\Neurophos\Python_Test_Scripts\Fiber_Amplifiers\scripts\fiber_burst_pam.py`; it reuses
only public API.)

A smaller related point: the returned `power_W` retains the negative pump value
(-1.2e-16 W). `_dP_full_c` clamps `P = np.maximum(P, 0.0)` internally for the physics, so this is
cosmetic, but clipping the returned array would spare downstream consumers a negative optical power.

---

## F-14 (medium) -- the relaxation genuinely fails for a low-signal, high-pump quasi-three-level amplifier

Distinct from F-13, and found by the fixed-point measure above. On the same 2 W Yb reference, at
0.05-0.1 mW of signal input, the solve returns a **spurious solution** rather than a slow one:

```text
 P_in(mW)   gain_dB   nbar2_mean   fwd ASE(mW)   bwd ASE(mW)   fp_resid   gain_resid
    0.050    -12.56   0.0357       0.00          0.02          0.85       4.72 dB
    0.100    -10.47   0.0593       0.00          0.05          0.61       2.65 dB
    0.243     34.72   0.3306      45.73        581.73          0.005      2.1e-5 dB
```

At 0.243 mW the amplifier is inverted (`nbar2` 0.33) and gives 34.7 dB. At half that input it
reports `nbar2` 0.036 -- essentially UNPUMPED -- and net absorption. That is not a physical
bifurcation: dropping the signal power cannot de-invert a fiber absorbing 2 W of pump. Extra
iterations do not recover it (checked to 3000).

The mechanism is visible in the healthy-case trace in F-13: the Gauss-Seidel oscillation
(31.1 -> 36.2 -> 32.9 -> 35.7 -> ...) decays only slowly, and its decay rate degrades as the
backward ASE grows. At 0.243 mW the backward ASE is already **582 mW against a 2 W pump**; at
0.05 mW it is larger still relative to the signal, the iteration's amplification factor exceeds 1,
and the sweep falls into the absorbing (uninverted) state and stays there.

This is a real limitation of alternating frozen-direction relaxation, not a bug in the algebra, and
the module docstring is already honest that a single global `solve_bvp` was rejected because it
"overflows on the ASE that grows from the spontaneous floor through tens of dB of gain."

### Suggested implementation

1. **Expose under-relaxation.** The cheapest fix by far. Blend each sweep's result with the
   previous profile, exactly as `thermal.solve_with_thermal_feedback` already does with its
   `relax=0.7` parameter -- so the pattern is established in the package:

   ```python
   def solve(self, *, n_nodes=201, max_iter=200, tol=1e-6, method="LSODA", relax=1.0):
       ...
       P_bwd = relax * sb.y[:, ::-1] + (1.0 - relax) * P_bwd   # relax < 1 damps the oscillation
   ```
   `relax=1.0` keeps every current result byte-identical. A value near 0.5 should both fix the
   0.05 mW case and cut the iteration count in the healthy cases, since the observed failure mode
   is an under-damped oscillation.

2. **Detect it rather than return it.** Whatever the iteration strategy, a solve whose own
   `nbar2(z)` does not reproduce its own channel powers is not a solution. Adding the fixed-point
   residual as a post-solve check (and putting it in `meta`) would turn a silent wrong answer into
   a diagnosable one -- which matters more here than the iteration count, because the returned
   gain was wrong by 47 dB.

3. **Continuation.** For sweeps, seeding each solve from the previous point's profile would both
   accelerate convergence and keep the iteration on the physical branch. `solve()` currently always
   restarts from the flat boundary-power guess (`steady_state.py:395-396`), so a sweep pays full
   price at every point and each point is free to land on a different branch.

---

## Verification method

The propagation core was checked against an independently written implementation rather than read
for plausibility. Using only the public API (`ChannelSet.build`, `gain_coeff_per_m`,
`overlap_gamma`, and the `SteadyStateResult.meta` cross-sections), an integrating-factor
frozen-inversion propagator was handed `solve()`'s own converged `nbar2(z)` and compared channel by
channel on the reference EDFA (1.4 um core, NA 0.23, 1e25 m^-3, 8 m, 300 mW at 976 nm, 5 mW at
1550 nm, 16 ASE bins over 1520-1570 nm):

```text
signal out    : solve 1.674617e-01 W   independent 1.674612e-01 W   ratio 1.00000
residual pump : solve 3.112360e-02 W   independent 3.112353e-02 W   ratio 1.00000
total fwd ASE : solve 5.86982e-04 W    independent 5.87213e-04 W    ratio 1.00039
per-bin ASE   : agreement 1.0000-1.0005 across the band
```

The 4e-4 on ASE is the difference between LSODA on an adaptive mesh and a trapezoidal
integrating-factor sweep on a 201-node uniform mesh -- i.e. discretization, not a discrepancy in
the physics. **The Giles coupled-power core, the spontaneous source normalization (including the
frequency-vs-wavelength bin-width conversion), the fwd/bwd boundary handling and the metastable
balance are all confirmed correct.**

The two-timescale ordering that a burst-mode model rests on was also checked numerically against
the package's own `saturation_energy`, since it justifies treating symbols as seeing a frozen gain:

```text
Er 1550 nm, A_dope 6.158e-12 m^2, Gamma 0.311: E_sat = 6.483 uJ, P_sat = E_sat/tau = 0.648 mW
Yb 1060 nm, A_dope 2.827e-11 m^2, Gamma 0.751: E_sat = 31.44 uJ, P_sat = 37.9 mW
one symbol at 10 mW / 10 Gbaud carries 1 pJ  ->  E_symbol/E_sat = 1.5e-7 (Er), 3.2e-8 (Yb)
```

Seven to eight orders of magnitude, so the inversion integrates the burst envelope and cannot
respond to a symbol. The long-time limit of `simulate_transient` on the envelope reproduced
`amp.solve()` for the same drive to **0.003 dB** on the reference EDFA, and
`meta['quasi_static_valid']` correctly stayed `True` throughout (worst margins: ASE/launched 0.40
against a limit of 1.0, `INT g dz` 12.3 against 20).

**The A-7 monitor works as advertised, including on the failure it was built for.** Sweeping the Yb
reference to a 12 m fiber at 2 W of 976 nm pump drove `_propagate_fixed`'s `exp(INT g dz)` to
numerical overflow; `simulate_transient` set `quasi_static_valid = False` with the reason
"non-finite channel powers appeared during the march" and raised its `RuntimeWarning`, exactly as
documented. Nothing was silently wrong. Two small observations from that encounter, offered as
constructive follow-ups rather than as findings:

* The overflow is avoidable rather than intrinsic. `_propagate_fixed` computes
  `np.exp(G) * (bc + cumtrapz(s * np.exp(-G)))`, which overflows in `exp(G)` and underflows in
  `exp(-G)` separately even when the product is representable. Factoring the running maximum out
  of each row -- the standard log-sum-exp shift, `exp(G - G_max)` paired with
  `exp(G_max - G)` inside the integral -- would extend the usable range by hundreds of decades at
  no accuracy cost, and would let the A-7 monitor's *physical* criteria (the ASE/launched ratio and
  the gain integral) be the thing that trips, rather than IEEE-754.
* Because the NaN arrays are returned (deliberately, per the docstring), a consumer that forgets to
  check the flag gets NaNs propagating into whatever it computes next. That is the documented
  contract and arguably correct, but a `TransientResult.assert_valid()` convenience -- or simply
  mentioning the flag in the `simulate_transient` return annotation -- would make the failure mode
  harder to miss.

## What was NOT found

Stated explicitly, because the absence is the more important result: no error in the Giles
propagation, the relaxation solver, the ASE spontaneous-source normalization, the McCumber
machinery itself, the noise-figure algebra (`nf_from_psd` / `nf_from_nsp` reproduce
`(2 n_sp (G-1) + 1)/G`), the beat-noise variances (Monte-Carlo pinned under C4-3 and reproduced
independently here), the PCE definition, the heat-flux balance, or the transient's exponential
integrator and its semi-implicit upconversion handling. The `m_modes` threading, the audit-A-5
temperature-profile propagation through clones, and the audit-A-1 raw-`upconversion_C_up` spelling
all behave as their comments claim. The unit discipline is genuinely watertight: every optical power
in the audited modules is watts, every `_dB`/`_dBm` name is an output, and the one place a
frequency-vs-wavelength bin width could have gone wrong (`steady_state.py:234-236`) gets it right
and says so.

## Suggested priority

1. **F-13** -- a two-line fix (`denom` floor) to a flag the documentation tells users to trust,
   which is currently wrong on every efficient amplifier and costs 3-15x runtime. Highest
   value-per-line in the list, and it should be done before F-14 because it is what currently
   masks F-14.
2. **F-1** -- a physical-accuracy defect at the package's own C-band specification wavelength.
   Option (b) (re-fit `sigma_a` anchors) plus the McCumber consistency gate is the cheapest
   credible fix; option (c) (ship one measured spectrum) is the highest value.
3. **F-14** -- a silently wrong answer (47 dB of gain error) in a reachable regime. The
   `relax` parameter is ~3 lines and byte-identical at `relax=1.0`; the fixed-point residual in
   `meta` is the durable fix.
4. **F-2, F-3** -- both are small additive API changes that remove the need for consumers to
   reimplement solver internals. F-2's data is already computed and discarded.
5. **F-4** -- a real wrong number on a standard dual-pump configuration; ~10 lines.
6. **F-5 + F-8** -- one signature change fixes both; default-off, byte-identical.
7. **F-7, F-6** -- new modules. F-7 is the one most likely to be wanted, contains no new physics,
   and is where the existing `total_heat_W` and `stokes_limit` become load-bearing.
8. **F-9, F-10, F-11, F-12** -- housekeeping, but F-9 will cascade into the LMA test suite and
   should be scheduled rather than dropped in.

## Note on scope

F-13 and F-14 were found only because the study needed to sweep an amplifier from small-signal into
deep saturation, which walks straight through the regime where both bite. Neither is reachable from
a fixture that solves one well-chosen operating point, which is why the package's own extensive gate
suite does not catch them: `tests/test_fiber_amp.py` asserts `meta["converged"]` on Er fixtures with
substantial residual pump, where the flag behaves. A parameter-sweep gate -- solve across three
decades of input power and assert monotonically decreasing gain -- would catch both, and would be a
cheap addition to the suite.
