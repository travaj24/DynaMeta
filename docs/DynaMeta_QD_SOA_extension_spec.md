# Adding a Quantum-Dot SOA Amplifier Model to DynaMeta

**Purpose.** Specify what the `dynameta` library needs in order to simulate a
quantum-dot semiconductor optical amplifier (QD SOA) amplifying an
**incoherent, intensity-encoded** signal at GHz symbol rates — the gain leg of
the incoherent OVMM — to the fidelity required to predict analog precision
(SFDR / ENOB).

**Scope.** This is the *amplifier* problem. DynaMeta today targets tunable and
active/lasing **metasurface modulators** (the weight/encoding layer). The
amplifier is a distinct physics domain and is not yet modeled. This document is
the gap analysis plus a phased build plan.

> **Assessment note (2026-06-11).** A code-grounded + adversarial-physics + literature
> review of this spec is in **Section 8**. Verdict: the plan is sound and genuinely
> additive, and the Section 3 reuse claims are accurate (two factual fixes below). But the
> Section 6 governing equations need six corrections before Phase 1, and four items the
> spec files under "optional" are actually CORE to hitting SFDR/ENOB. Read Section 8 before
> implementing.

**Acronyms (used throughout):** SOA = semiconductor optical amplifier;
QD = quantum dot; ASE = amplified spontaneous emission; SFDR = spurious-free
dynamic range; ENOB = effective number of bits; SHB = spectral hole burning;
WL/ES/GS = wetting layer / excited state / ground state; NF = noise figure;
FWM = four-wave mixing.

---

## 1. What we are modeling

A QD SOA carries an optical signal whose **intensity** encodes the value — no
phase information, an incoherent scheme. A stream of intensity symbols arrives
at a GHz rate, and the amplifier must reproduce each symbol's amplitude
faithfully and consistently. The governing question is not "how much gain" but
"how linearly and how cleanly," and the figure of merit is:

- **SFDR** — the usable window between the noise floor below and the onset of
  distortion above.
- **ENOB** — the analog precision that window supports.

This differs sharply from telecom on/off-keying or PAM4, which only need to
separate a few discrete levels and therefore tolerate substantial gain
compression and residual pattern effects. Here the analog value *is* the
information, so the **whole transfer curve must stay linear (or be calibrated)
to ~1 part in 2^N for N bits, and be stable symbol-to-symbol.**

---

## 2. Physics established (summary)

- **Pulsing does not dodge CW saturation at these rates.** Gain is set by
  carrier density, which responds on the carrier-recovery timescale (tens to a
  few hundred ps). At GHz–50 GHz the device cannot recover between symbols, so
  it responds to the *average* power and saturates roughly as it would under CW
  at that average. The hard ceiling on average added power is the pump:
  `P_out − P_in ≤ (I/q)·hν·η`. Peak power buys nothing against it.

- **QD is the right amplifier class.** A carrier reservoir (WL + ES) refills the
  dot ground state fast, which (a) lets pump current push the
  carrier-depletion saturation power up substantially — SHB then becomes the
  residual limit — (b) gives fast gain recovery and hence low pattern effects,
  and (c) yields high saturation power and a low noise figure (~5 dB class).

- **Going incoherent removes the hard part.** With intensity encoding, the phase
  penalties of the coherent case — linewidth enhancement factor (α), self-phase
  modulation, phase noise — drop out entirely. What remains is amplitude
  linearity, gain dynamics, and noise.

- **The analog operating point is a window, not a wall.** Distortion rises as
  input power approaches saturation; the ASE/noise floor sits underneath. There
  is an *optimal* drive power that maximizes SFDR — back off from saturation, or
  predistort the known compression curve. This pairs with the downstream
  precision-reconstruction stack (dithering, hierarchical ADC, RNS): the SOA's
  SFDR is the raw analog-channel quality those techniques then refine.

---

## 3. What DynaMeta already provides (reusable)

| Module | What it does | Relevance to the SOA model |
|---|---|---|
| `optics/laser_gain.py` | Four-level laser gain medium: exact matrix-exponential population solver; closed forms for small-signal gain `g₀`, Fabry-Pérot photon lifetime, threshold inversion, pump threshold, relaxation-oscillation frequency. Feeds the metasurface FDTD via a Lorentz-oscillator gain ADE. | Solid laser-physics core and good closed-form sanity checks. **But** the inversion is solved under a *constant pump and clamped* for the optical solve — dynamic field↔population coupling (saturation, lasing) is explicitly flagged as a follow-on. The scheme is four-level *atomic*, not semiconductor band gain. |
| `carriers/carrier_heating.py` | Two-temperature hot-electron model (`T_e` vs `T_l`), Kane non-parabolic mass, Drude damping vs `T_e`; sub-ps rise / ps relaxation. | **Directly reusable** as the SOA carrier-heating ultrafast term — the same physics, currently parameterized for a free-carrier ENZ film rather than an inverted gain medium. |
| `carriers/transient.py`, `dynameta/transient_optics.py` (top-level, **not** under `carriers/`) | Time-domain carrier transport (DEVSIM BDF) and the carrier→Drude→optics transient loop. | Time-domain solver infrastructure and conventions. Aimed at the gated modulator (accumulation, RC, charge storage), not amplifier gain. |
| `carriers/ac_analysis.py` | Small-signal admittance `Y(f)` → capacitance, RC bandwidth. | Bandwidth-analysis pattern; not optical gain dynamics. |
| materials / geometry / FDTD / numerics infra | Declarative material + geometry model, DEVSIM bridge, FDTD engine, solver scaffolding. | Reusable scaffolding and conventions for a new amplifier module. |

---

## 4. Gap analysis — what must be added

In rough priority order:

1. **Semiconductor QD multi-state gain model.** Replace the four-level atomic
   scheme with a quantum-dot population model — WL reservoir → ES → GS, with
   capture / escape / relaxation times and an **injection-current pump** (not a
   constant optical pump rate `W_p`). Material gain `g ∝ N_dot·(2ρ_GS − 1)`,
   summed over the inhomogeneous dot-size distribution for the spectral gain.
   This is what reproduces QD's fast recovery and high saturation power.

2. **Dynamic field↔population coupling.** The signal photon density must deplete
   the inversion locally and in time — the stimulated-emission term that
   `laser_gain.py` currently omits (its "follow-on"). This one addition is what
   produces **gain saturation, the compressed transfer curve, and pattern
   effects** — i.e. the core of this simulation.

3. **Traveling-wave (z-resolved) propagation.** Propagate the signal envelope
   `P(z, t)` along the amplifier length, with the carrier rate equations
   evaluated per z-slice and driven by the local photon density. This replaces
   the clamped single-cell / metasurface-cell treatment. Solve by split-step
   march in `z` and `t`.

4. **ASE noise.** A local spontaneous-emission source into the guided mode (with
   the inversion / `n_sp` factor), propagated to the output, then **detector
   beat-noise statistics** (signal–spontaneous and spontaneous–spontaneous) for
   the SNR. Without this there is no noise floor, hence no SFDR/ENOB.

5. **Analog-channel metrics.** Drive the model with a representative
   intensity-symbol stream and compute: the static transfer curve and its
   distortion (harmonic / intermodulation or deviation-from-linear), the noise
   floor, and from them **SFDR and ENOB**; plus an inter-symbol (pattern)
   penalty on the analog values. Add an optional **predistortion** hook that
   inverts the static compression curve to extend the linear range.

6. **Secondary / optional.** SHB and carrier heating coupled into the gain
   (`carrier_heating.py` supplies the latter); multi-wavelength FWM crosstalk if
   the OVMM is WDM; self-heating / thermal coupling for the operating-point
   drift that ultimately limits predistortion stability.
   **[Reclassified -- see Section 8.5: spectral hole burning is CORE (it is just the
   group-resolved gain saturation of items 1-3, not a separate add-on) and self-heating is
   CORE for any predistortion-based ENOB claim. Carrier heating and FWM remain genuinely
   secondary.]**

---

## 5. Proposed module and build plan

**Placement.** A new `dynameta/optics/soa.py` (or an `amplifier/` subpackage),
declarative in the existing DynaMeta style, reusing the matrix-exponential / ODE
solver pattern from `laser_gain.py` and `TwoTempParams` from
`carrier_heating.py`. Parameterize from datasheet-derived values. (**Correction:** no
Innolume/SOA/BOA parameter set or amplifier model exists in DynaMeta today -- a repo-wide
search finds the term only in this doc. The QD parameter set must be digitized from the
datasheet as new work, not reused; see Section 8.1.)

**Suggested objects**

- `QDGainModel` — multi-state rate equations + spectral gain; steady-state and
  time-domain.
- `TravelingWaveSOA` — z-resolved solver; `.amplify(input_waveform)` → output
  waveform + carrier trajectories; `.saturation_curve()` → static transfer.
- `ASEModel` — spontaneous-emission source + noise figure + detector beat-noise.
- analysis helpers — `sfdr()`, `enob()`, `pattern_penalty()`,
  `optimal_drive_power()`.

**Phases (each independently testable)**

1. **QD gain core** — multi-state rate equations + injection pump; produce the
   static gain-vs-input-power (saturation) curve.
   *Validate:* small-signal limit matches `laser_gain.small_signal_gain_per_m`;
   saturation power rises with pump current as expected.
   **[SHIPPED 2026-06-11: `dynameta/optics/soa/qd_gain.py` (QDGainParams, QDGainModel) +
   `validation/qd_soa_gain_core.py` (5 gates green). Group-resolved WL->ES->GS with the
   Section-8 corrections; standalone package, no edits to laser_gain.py. Per the corrected
   Phase-1 oracle (8.6) the small-signal check is transparency at rho_GS=1/2, NOT a
   cross-quantity equality with the atomic laser_gain g0. Gates: transparency+monotonicity,
   saturation+pump-dependence (S_sat rises 2.2e20->8.8e20 m^-3 over 6-25 mA), particle
   conservation 7e-14, detailed balance exact, SHB hole tracks the drive frequency.]**
2. **Traveling-wave dynamic coupling** — couple `P(z, t)` to the carrier
   dynamics; produce gain recovery and pattern effects.
   *Validate:* gain-recovery time vs literature; near pattern-free behavior at
   high symbol rate with QD parameters.
   **[SHIPPED 2026-06-11: `dynameta/optics/soa/traveling_wave.py` (TravelingWaveSOA, a
   method-of-characteristics z-t marcher with dt = dz/v_g; pluggable slab-gain protocol) +
   `validation/qd_soa_traveling_wave.py` (5 gates green). GATE A verifies the engine against
   the analytic Agrawal-Olsson saturable-gain pulse law to 3e-4 with 1/n_slices convergence
   (TwoLevelSaturableGain + agrawal_olsson_output, the analytic oracle). GATE B: 6.6 ps QD
   gain recovery, completing more with pump (reservoir). GATE C: the QD low-pattern-effect
   advantage (penalty 0.05 vs 0.32 for a throttled reservoir at 80 GHz). GATE D: dynamic
   distortion is frequency-dependent (HD2 0.044 at 2 GHz -> 0.0053 at 150 GHz), the memory
   effect a static IP3 misses. GATE E: added power below the (I/q)h nu one-photon-per-electron
   ceiling + passivity. Standalone engine, NOT wired into the run_pipeline metasurface seam.
   REMAINING for full nonlinear coverage: true 2-colour cross-gain modulation + four-wave
   mixing need a multi-tone field extension (Phase 3); ultrafast spectral-hole + carrier-
   heating dynamics are captured at the group-resolved + reservoir level here and refine
   with the TwoTempParams coupling (Phase 5).]**
3. **ASE + noise** — spontaneous-emission source and beat-noise statistics.
   *Validate:* noise figure lands in the QD-SOA range (~5 dB) and approaches the
   quantum limit (~3 dB) at high gain.
   **[SHIPPED 2026-06-11: `dynameta/optics/soa/ase_noise.py` (inversion_factor_nsp,
   ase_output_psd [z-resolved ASE ODE, exact per-slice, machine-precision reduction to
   n_sp h nu (G-1)], noise_figure [internal-loss + input-coupling corrected],
   detector_noise_variances [shot + sig-spont + spont-spont, Olsson forms]) +
   `validation/qd_soa_noise_metrics.py` GATES A-C. n_sp -> 1 and NF -> 3.01 dB quantum limit
   at full inversion; NF degrades with internal loss; ASE-beat-dominated noise at high gain.
   CORRECTION applied: n_sp = rho^2/(2 rho - 1) (the excitonic f_c(1-f_v)/(f_c-f_v) form,
   consistent with the rho^2 spontaneous term), NOT the spec Section-6 rho/(2 rho - 1).]**
4. **Analog metrics** — symbol-stream drive → SFDR / ENOB / ISI; predistortion
   hook.
   *Validate:* the SFDR-vs-drive-power optimum reproduces the
   distortion-above / noise-below trade-off.
   **[SHIPPED 2026-06-11: `dynameta/optics/soa/metrics.py` (transfer_derivatives,
   harmonic_amplitudes, sndr_db, enob, sndr_vs_drive) + `validation/qd_soa_noise_metrics.py`
   GATE D: combining the gain-compression distortion (transfer-curve curvature) with the ASE
   beat-noise floor, SNDR vs drive has an INTERIOR maximum (ENOB 6.7 at ~3.7 mW for the
   default device) -- the "window, not a wall." Symbol-stream ISI is covered dynamically by
   the Phase-2 pattern/recovery gates. The predistortion hook + analog helpers SHIPPED (see
   below).]**
5. **Extras -- SHIPPED 2026-06-11 (everything left over).**
   - **FWM / multi-lambda + XGM:** the coherent complex-envelope path
     `TravelingWaveSOA.amplify_coherent` (alpha_lef in QDGainParams) +
     `validation/qd_soa_fwm_xgm.py` (6 gates): cross-gain modulation (-5 dB at 30 mW pump),
     four-wave-mixing conjugate sidebands rolling off with detuning, the
     eta_FWM = (1 + alpha^2) carrier-density-pulsation law to 1.2%, the up/down-conversion
     asymmetry that is 0 at alpha=0 and grows with alpha (the index grating breaking
     symmetry); alpha=0 single-tone reduces to the verified power engine (1e-15).
   - **Ultrafast SHB + carrier heating:** `UltrafastCompression` folded into the engine +
     `validation/qd_soa_ultrafast.py` (3 gates): off-switch byte-identical; extra nonlinear
     gain compression beyond carrier-density saturation; the SHB+CH compression recovers
     sub-ps (~tau_CH 0.68 ps) vs the ~6.6 ps carrier reservoir -- the two-timescale QD-SOA
     gain dynamics. (Shares its time constant with carrier_heating.TwoTempParams; folded
     phenomenologically, carriers still see the real photons.)
   - **Analog helpers + thermal budget:** metrics.predistort (linearizes the compression
     ~1500x), optimal_drive_power, sfdr_dB, pattern_penalty_dB, thermal_drift_budget_K
     (dT_max for half-LSB predistortion stability, tighter with bits) +
     `validation/qd_soa_noise_metrics.py` GATES E-F.
   - **Spectral gain DISPERSION -- SHIPPED 2026-06-19** (`amplify_coherent(line_filter=True)` +
     `validation/qd_soa_spectral_dispersion.py`, 5 gates). A Maxwell-Bloch AUXILIARY-DIFFERENTIAL-
     EQUATION line filter (chosen over split-step-Fourier after a design panel: it keeps the exact
     time-march frame, needs no FFT/tone-comb, and resolves the band from one complex polarization
     pole per inhomogeneous group). Each tone at nu_s+f now sees its OWN complex gain
     Gamma_field(nu_s+f) = 0.5 sum_j A_j/(1 - i(nu-nu_j)/hw) -- 2 Re == the existing real gain g,
     the imaginary part is the Kramers-Kronig resonant index. Gates: OFF reduces to the power
     engine (5.6e-16); per-tone gain == analytic Lorentzian ensemble to 0.0077 dB over |f|<=300 GHz;
     ON-OFF transmitted phase == analytic line dispersion to 0.2% with the correct causal sign;
     up/down FWM asymmetry ENLARGED 0.44 dB @20 GHz (vs ~0 flat-gain), monotone in detuning. The
     transit-AVERAGED polarization readout removes the O(dt) ZOH half-sample-delay error; the field
     update is ADDITIVE (the polarization radiates into field nulls -> stable for modulated/nulling
     waveforms, no divide-by-field). Background GVD is now also modeled (see the GVD bullet below).
   - **Speedup -- SHIPPED 2026-06-19**: byte-identical Lorentzian/prefactor caching (1.1-1.3x) +
     an OPT-IN numba carrier-step accelerator `QDGainModel(fast=True)` (bit-parity 1e-16,
     ~7x on the carrier RK4, 3-4.4x on the full marcher; default OFF for byte-stable validations).
     `validation/qd_soa_numba_parity.py`. The remaining lever is a full-marcher JIT (~1.5-2x more,
     deferred for its duplicate-logic maintenance cost); GPU is not worthwhile (cache-resident
     (nz x ng) arrays are launch-bound, same conclusion as the FDTD 'auto' backend).
   - **Electron/hole occupation split -- SHIPPED 2026-06-19** (`QDGainParams(eh_split=True)` +
     `validation/qd_soa_eh_split.py`, 4 gates). The #1 gain-fidelity gap from the Section-8.3/8.5
     reclassification is now an opt-in path: the dots carry SEPARATE electron f_c and hole f_v
     occupations per state, with their OWN capture/escape/relaxation times (holes default to the
     electron times; the physical asymmetry is holes-faster). gain -> N_q w mu_GS sigma_pk L
     (f_c_GS + f_v_GS - 1) [the f_c f_v terms in (downward) - (upward) cancel -> LINEAR inversion],
     spontaneous -> f_c f_v, n_sp -> f_c f_v/(f_c + f_v - 1) (ase_noise.inversion_factor_nsp_eh).
     ONLY stimulated + spontaneous couple the two bands (the SAME scalar into both); WL recomb is
     the pair form B N_w_e N_w_h. Separate electron and hole number are each conserved by internal
     transitions; charge neutrality d(n_tot_e - n_tot_h)/dt = 0 is a global invariant. Gates:
     symmetric e/h reduces to the excitonic model (occupations 1.6e-11, gain 0.0); gain/n_sp closed
     forms exact; separate e/h conservation 1e-12 + closed-box 7e-16; the NEW physics -- with holes
     ~5x faster the saturated-gain recovery differs from excitonic by 3.3% and the GS e/h
     occupations split transiently (the high-speed pattern-effect physics the single-rho excitonic
     model cannot represent), vanishing when the times are restored symmetric. Mirrored into a
     second numba kernel (_qd_carrier_rk4_eh_numba, fast=True parity 1e-16). Default eh_split=False
     keeps the excitonic path byte-identical.
   - **Excited-state optical band / two-state gain -- SHIPPED 2026-06-19**
     (`QDGainParams(sigma_pk_ES_m2>0)` + `validation/qd_soa_es_band.py`, 4 gates). The ES
     transition (nu_ES_j = nu_j + dE_ES_GS*q/h, ~1442 nm, carrying mu_ES) becomes optically active:
     gain g = g_GS + g_ES, an ES stimulated term depletes rho_ES (both f_*_ES in the split), and
     the GS/ES two-state crossover emerges (I_th,ES > I_th,GS, ES gain exceeds the gain-clamped GS
     at high injection). sigma_pk_ES=0 (default) -> GS-only byte-identical; mirrored in both numba
     kernels (parity 0.0). Gates: reduction exact, g_ES == analytic ensemble 1e-16, two-state
     crossover, ES photon-number conservation 1e-16.
   - **Lumped self-heating + facet Airy ripple (ENOB budget) -- SHIPPED 2026-06-19**
     (`SelfHeating` dataclass + `metrics.facet_gain_ripple_dB`/`ripple_enob_ceiling` +
     `validation/qd_soa_enob_budget.py`, 5 gates). set_temperature rigidly red-shifts the combs
     (nu0(T)=nu0-dnu0_dT(T-T0)) and scales the gain (1+dg_dT_frac(T-T0)) on BOTH emission and
     stimulated depletion (photon-safe); steady_gain_self_consistent reaches the thermal fixed
     point T=T0+Rth*P_diss (destabilizing optical-extraction feedback, stable for loop gain<1,
     non-convergence guard raises); dGdT_dB_per_K feeds metrics.thermal_drift_budget_K. The facet
     ripple is the Saitoh-Mukai Airy peak-to-valley (0.17 dB @ R=1e-4, G=20 dB -> ~5.6-bit ENOB
     ceiling). Rth=0 / coefficients 0 -> isothermal byte-identical. Demonstrated ENOB ceiling: a
     13.5 K self-heating drift (dG/dT=-0.037 dB/K) dwarfs the 0.23 K 8-bit predistortion budget.
   - **Bidirectional spectrally-resolved ASE (the final gain ceiling) -- SHIPPED 2026-06-19**
     (`ase_noise.ase_spectrum_bidirectional`/`ase_self_consistent`/`spectral_noise_figure`/
     `inversion_factor_nsp_eh` + `validation/qd_soa_bidir_ase.py`, 5 gates). The forward-only
     single-band `ase_output_psd` is generalized to a frequency grid with the z-resolved gain
     profile and FORWARD + BACKWARD PSD transport. The emission source is pole-free,
     q = Gamma g_sp h nu == Gamma g n_sp h nu (g_sp from emission_gain_per_m's rho^2 vs material
     gain's (2 rho - 1), so g_sp/g = n_sp exactly), and S_f propagates with the NET coefficient
     a = Gamma g - alpha_i. From it: the spectral noise figure NF(nu) (-> 2 n_sp at high gain,
     == the scalar noise_figure at centre), and ASE-induced gain saturation (the integrated
     bidirectional ASE photon density depletes the inversion through a self-consistent lumped
     fixed point). Gates: single-nu forward == ase_output_psd (1.8e-16); uniform spectral sum
     S_f == n_sp h nu (G-1) over the band (4e-14); spectral NF @centre == noise_figure (3.5e-13)
     AND -> 2 n_sp at G>1e3, AND the LOSSY-device NF (alpha_i>0) == noise_figure(bare n_sp, Gg,
     alpha_i) -- proving the loss degradation is counted ONCE (it lives in the net-propagated S_f,
     not multiplied in again); bidirectional symmetry S_f(L) == S_b(0) (0.0); ASE clamps the gain
     monotonically with load, ase_saturation=False reproduces the unsaturated propagator exactly.
     Adversarial pass caught + fixed a P1 (spectral_noise_figure double-counted internal loss when
     alpha_i>0) before ship. This closes every documented gain-leg ceiling from Section 8.
   - **Background group-velocity dispersion (GVD) -- SHIPPED 2026-06-19** (`amplify_coherent(
     beta2_s2_per_m=...)` + `QDGainParams.beta2_s2_per_m` + `validation/qd_soa_gvd.py`, 5 gates).
     The waveguide GVD d2 beta/d omega^2 [s^2/m] adds dA/dz = -(i beta2/2) d2A/dT2 (retarded time,
     exp(-i omega t)): each tone at nu_s + f picks up exp(+i (beta2/2)(2 pi f)^2 L). Applied as a
     SYMMETRIC device-scale (Strang) split D(L/2) . marcher . D(L/2), where D is the EXACT unitary
     spectral phase on the full retarded-time waveform (FFT . phase . iFFT) and the streaming
     nonlinear marcher is untouched -- a per-step dispersion of the spatial node array is invalid
     (it is a snapshot along z, not a fixed retarded-time window, so the shifting-window FFT leaks
     energy; this was caught and rejected during the build). EXACT in the linear/passive limit;
     uncontrolled (single step, no z-refinement) for the distributed dispersion-gain coupling when
     beta2 and gain are both active (a sub-sectioned split is the future refinement). beta2 even in
     omega -> FFT sign immaterial (beta3 would not be). Gates: beta2=0 byte-identical; the split
     assembles to exactly D(L) (4.5e-15) and a windowed CW tone's phase == (beta2/2)(2 pi f)^2 L
     (1.3e-4 rad, both signs); Gaussian RMS broadens to T0 sqrt(1+(L/L_D)^2), L_D=T0^2/|beta2|
     (Agrawal, 7.7e-13); quadratic chirp coefficient C=-(beta2 L)/(2(T0^4+(beta2 L)^2)) sign+
     magnitude (1.5e-16, both signs); gain-free unitarity sum|A|^2 conserved (3.8e-16). Adversarial
     pass confirmed sign/magnitude independently (no code bug); reworded over-claiming docstrings to
     disclose the uncontrolled-coupling limit. GVD was the last explicitly-deferred item -- the
     gain-leg model now spans every physics axis in this spec.

   --- Deeper-realism upgrades (owner-requested, replacing the phenomenological/lumped reductions
   with controlled ones; each opt-in, default-off, oracle-validated) ---
   - **Distributed GVD (multi-segment split-step) -- SHIPPED 2026-06-19** (`amplify_coherent(
     gvd_segments=S)` + `validation/qd_soa_gvd_distributed.py`, 5 gates). Redeems the GVD
     uncontrolled-coupling caveat above: S sub-sections interleave dispersion and gain S times
     [D(L/2S) . N(L/S) . D(L/S) . ... . N(L/S) . D(L/2S)], each N a full streaming sub-marcher with
     its own z-pinned carriers, so the distributed coupling the single endpoint split (S=1) only
     approximated is now CONTROLLED. Gates: gvd_segments=1 == the single split (byte-identical);
     gain-free S-invariant (9.5e-16) + NLSE broadening (1.6e-16); 2nd-order Strang convergence,
     successive-difference ratios plateau at 4.00 out to S=128 (the testable order claim that
     replaces the earlier bare "2nd order" assertion); the distributed coupling is REAL -- in a
     strong regime S=1 differs from the converged result by 4.6% while the high-S tail is Cauchy
     (7e-5); passivity/energy conserved for every S. Adversarial pass (fix-then-ship): composition
     verified correct (dispersion sums to L, no double-count, dz invariant, fresh per-segment
     carriers physically right); the one fix was documenting that the segmented path's leading nz
     samples are an S-dependent multi-fill transient (only the steady tail A_out[nz:] is physical).
   - **Density-dependent alpha(rho) + polarization-dependent gain (PDG) -- SHIPPED 2026-06-19**
     (`QDGainParams.alpha_lef_density_slope` + `QDGainModel.alpha_lef_slices` + `amplify_coherent_
     dualpol` + `validation/qd_soa_alpha_pdg.py`, 6 gates). (1) alpha(rho): the linewidth enhancement
     factor rises with inversion as the gain clamps (dg/dN falls, the carrier-induced index dn/dN
     persists), alpha = alpha_lef + (slope/2)(2 rho_GS - 1) applied PER SLICE from the local carrier
     state (first-order phenomenological expansion; the FREQUENCY dependence of the carrier-induced
     index is the resonant Kramers-Kronig line filter, already shipped). (2) PDG: amplify_coherent_
     dualpol co-propagates TE + TM through ONE shared carrier reservoir with a TM/TE modal-gain ratio
     pdg_ratio (folds confinement ratio + QD material anisotropy), saturated by the modal-weighted
     total |A_TE|^2 + pdg_ratio|A_TM|^2 -- so the pols cross-saturate. Gates: slope=0 byte-identical +
     scalar helper; alpha_lef_slices formula exact (0.0); the DEFINING relation -arg(A_out)/ln|A_out/
     A_in| == the gain-weighted z-average of alpha(rho(z)) (unsat 3e-13, and under saturation the
     gain-weighting beats a plain z-average, |meas-gw|=1.9e-2 << |meas-zavg|=2.3e-1, gradient 0.21);
     PDG pdg_ratio=1 TE-only == single-pol (byte-identical); small-signal PDG == (1-r)Gamma g L*10/ln10
     (1.4e-3); cross-saturation -- a strong TE drops the weak-TM gain 2.0 dB (shared reservoir).
     Adversarial pass (fix-then-ship): all physics confirmed (alpha sign, single-r dualpol fold no
     double-count, alpha_i unscaled); the one fix STRENGTHENED GATE C with the saturating discriminator
     (the unsaturated point alone could not distinguish the gain-weighting from a plain mean).
   - **True Fabry-Perot cavity (facet feedback) -- SHIPPED 2026-06-19** (`amplify_fabry_perot` +
     `validation/qd_soa_fabry_perot.py`, 5 gates). Replaces the single-pass Saitoh-Mukai ripple METRIC
     with a time-domain BIDIRECTIONAL marcher: counter-propagating FORWARD (F) + BACKWARD (B) envelopes
     coupled by the facet power reflectivities R1, R2 (F advances +z, B advances -z, dt=dz/v_g), both
     saturating the SHARED carriers (|F|^2 + |B|^2). Facet BCs F_in = t1 A_in + r1 e^{i phi/2} B(0),
     B_in = r2 e^{i phi/2} F(nz); roundtrip_phase phi is the cavity detuning. Gates: R1=R2=0 == single-
     pass amplify_coherent (byte-identical); phase-swept ripple == Saitoh-Mukai facet_gain_ripple_dB
     (2.5e-13); on-resonance enhancement == Airy (1-R1)(1-R2)/(1-sqrt(R1R2)Gsp)^2 (6e-4); near
     resonance external-seed feedback saturates the gain (round-trip 0.900<=1, G 9.84->9.61 dB, no
     run-away); passivity. SCOPE (adversarial fix): F=B=0 init with only the coherent A_in seed -> no
     spontaneous/ASE seed, so lasing FROM NOISE is out of scope; GATE D is external-seed saturation,
     NOT a self-consistent threshold pin (docstring reworded from the over-claimed "clamps at lasing").
     Marcher otherwise verified correct (advection dirs, facet BCs pre-update, reservoir intensity-sum
     no standing-wave term, phase split, R=0 byte-exact).
   - **Z-resolved dynamic bidirectional ASE -- SHIPPED 2026-06-20** (`ase_noise.
     ase_self_consistent_zresolved` + `ase_spectrum_bidirectional(return_profile=True)` +
     `validation/qd_soa_ase_zresolved.py`, 5 gates). Refines the lumped ase_self_consistent (Phase 8)
     so EACH slice's gain is saturated by its OWN local bidirectional-ASE photon density, not one
     device-averaged S_ase: the coupled fixed point g(z,nu) <-> S_f(z,nu),S_b(z,nu) <-> S_ase(z),
     iterated damped to convergence. ase_spectrum_bidirectional gained return_profile to expose the
     per-slice S_f_z/S_b_z (default off, byte-identical). Gates: ase_saturation=False == frozen
     transport (0.0); fixed point g_sat_z == material_gain(steady_state(I, signal+ase S_ase_z)) per
     slice (3.6e-12); the S_ase(z) profile is real (var 32%) and the local gain is PERFECTLY
     anti-correlated with it (corr -1.000, gain depressed where the ASE flux peaks); the device-output
     reduces to the lumped ase_self_consistent (7e-9, since it depends only on mean(S_ase_z) here);
     negative feedback (more ASE -> lower mean gain) + passivity. SCOPE (adversarial fix-then-ship): the
     ASE back-action is WEAK in the stiff QD gain (~1e-4 relative depression), so the refinement is the
     spatial PROFILE, not the aggregate output (disclosed, not buried); and it is purely LONGITUDINAL --
     the per-slice ASE enters steady_state as a scalar S_conf via the signal-frequency line filter, so
     spectral saturation stays lumped (an 8 THz comb treated as monochromatic at nu_s). PSD->density
     conversion, default-off byte-identical, and the fixed point all verified correct by the adversary.

   --- Heavier-physics build-out (owner-requested deeper realism beyond the 1-D effective-parameter
   layer; each opt-in, oracle-validated, adversarially verified) ---
   - **Closed-form many-body-corrected gain (screened-HF-flavoured) -- SHIPPED 2026-06-20**
     (`ManyBody` dataclass + `QDGainModel(many_body=...)` + `material_gain_index_mb` +
     `validation/qd_soa_many_body.py`, 5 gates). The free-carrier gain is a sum of complex Lorentzians
     whose Re part is the gain and Im part its KK index partner (one analytic chi(nu)); this
     renormalizes that chi with the three dominant finite-density many-body corrections, all functions
     of carrier density N (and T): bandgap renormalization (BGR red-shift, Haug-Koch universal
     dE_BGR = -bgr_coeff E_R (a_B^3 N)^(1/3)), excitation-induced + LO-phonon dephasing (HWHM
     broadening, OSCILLATOR-STRENGTH conserving so the peak drops as gamma0/gamma -- the correct
     invariant vs the free-carrier fixed-peak), and screened Coulomb/excitonic enhancement. gain and
     index are one chi -> KK-consistent and alpha = -gi/g (no separate alpha knob). Gates: disabled /
     zero-correction == material_gain_per_m (3.6e-12); gi == Hilbert(g) (KK pair, 2.6e-2); BGR shift
     == analytic + independent hand value + N^(1/3) scaling (2.1e-4); EID broadens HWHM with peak ~
     1/gamma (1.3e-16) and conserved line area (5.8e-3); Coulomb enhancement 1.5->1 across the Mott
     density. Adversarial pass (fix-then-ship 0.9): ALL physics confirmed correct (BGR sign/prefactor
     hand-verified 1.5e-16, KK causal pole, area conservation, exact reduction) -- the fixes were
     HONESTY: qualified "first-principles" (it is parameterised closed-form Haug-Koch, NOT a solved
     self-consistent k-resolved SBE -- that is the deeper continuum refinement) + disclosed it is a
     STANDALONE chi/alpha accessor NOT yet wired into the marcher (device dynamics unchanged until a
     caller drives the per-slice gain through it with N_w(z)) + hardened a gate with an independent
     BGR value.
   - **Stochastic Langevin spontaneous-emission noise -- SHIPPED 2026-06-20** (`amplify_coherent(
     langevin=True, seed=...)` + `amplify_fabry_perot(langevin=...)` + `emission_gain_per_m_slices`
     + `validation/qd_soa_langevin.py`, 5 gates). Each slice each step adds a complex-Gaussian field
     increment of variance Gamma g_sp(z) h nu v_g -- the fluctuation-dissipation spontaneous source
     (first-principles, no fitted coefficient: the geometric downstream-amplified slice sum reduces to
     the analytic n_sp h nu (G-1)/dt EXACTLY). Gates: langevin=False byte-identical (coherent + FP) +
     seed reproducible; mean ASE <|A|^2> dt == analytic ase_output_psd (2.8e-3); complex-Gaussian ASE
     -> intensity exponential <I^2>/<I>^2 = 2.012; the Henry amplitude-phase coupling DIRECTION (low-f
     phase noise rises with alpha, 1.11/1.25/1.37). SCOPE (honest, the adversary SHIP-ed with no
     fixes): the quantitative (1+alpha^2) laser linewidth is a gain-CLAMPED above-threshold result
     whose Hz-MHz width is below the fs-step time-domain marcher's FFT resolution; the marcher captures
     the coupling direction + the gain clamp (FP, Phase 13), not the absolute laser linewidth -- that
     needs a frequency-domain / clamped-gain treatment. The FP-Langevin path (noise seeds lasing) is
     available for that follow-on.
   - **Carrier transport: reduced SCH stage + DD injection-profile seam -- SHIPPED 2026-06-20**
     (`amplify(transport_tau_s=...)` + per-slice `drive` profile via the array-aware `init_slices` +
     `validation/qd_soa_transport.py`, 5 gates). Two depths beyond the lumped uniform-current
     injection: (1) REDUCED -- an SCH (separate-confinement-heterostructure) reservoir low-passes the
     injection current (dN_sch/dt = I/(qV) - N_sch/tau_t, the WL sees N_sch/tau_t), a first-order pole
     at 1/(2pi tau_t) -> the electrical modulation bandwidth limit; (2) the SEAM -- a non-uniform
     injection PROFILE I(z) (current crowding, or from a DEVSIM drift-diffusion solve) passed as a
     per-slice drive that init_slices seeds and rhs_fields carries. Gates: transport_tau=0 / const
     time-drive / uniform profile all byte-identical to lumped; the SCH rolls off current modulation
     (below the pole ratio 0.93, above 0.27); a non-uniform I(z) -> monotone non-uniform gain; the
     transport leaves the DC gain invariant (1e-16, N_sch -> I tau_t/(qV)); passivity. Adversarial pass
     (2nd clean SHIP, conf 0.93, NO must-fix): the SCH low-pass + pole + DC invariance + byte-identical
     reduction + genuine per-slice threading all verified; honestly labelled -- the SEAM is shipped,
     DEVSIM supplies I(z) through it (no DD solve is run in the gate; the reduced model is a current
     low-pass, not a spatial DD). The full self-consistent DEVSIM coupling drives the SAME per-slice
     seam.
   - **Spatially-resolved thermal: reduced 1-D fin T(z) + thermal-FEM sampling seam -- SHIPPED
     2026-06-20** (`optics/soa/thermal.py`: `thermal_profile_steady_1d`, `sample_T_along_axis`,
     `dome_analytic`; `QDGainModel.gain_per_m_thermal`; `validation/qd_soa_thermal_profile.py`, 5
     gates). Two depths beyond the lumped (single-T) `SelfHeating`/`set_temperature`: (1) REDUCED -- a
     steady 1-D heat-conduction fin `kappa A T'' - (T-T0)/Rth' = -q(z)` (longitudinal conduction +
     distributed sink), tridiagonal solve, `ends='sunk'` (Dirichlet, mounted facets -> the DOME) or
     `ends='insulated'` (Neumann); the per-slice dissipation `q(z)` [W/m] drives a non-uniform `T(z)`;
     (2) the SEAM -- `gain_per_m_thermal(state, nu, T_z)` red-shifts each slice's comb by `dnu0_dT
     (T_z-T0)` and scales gain by `1 + dg_dT_frac (T_z-T0)` (the per-slice `set_temperature`), and
     `sample_T_along_axis` projects ANY external 3-D `T(x,y,z)` (a thermal-FEM `ThermalResult.T_at`)
     onto the SOA axis -> a plain `T(z)` array. Gates: `kappa A -> 0` insulated reduces EXACTLY to the
     lumped per-slice `T0 + q Rth'` (0.0); the sunk-facet dome matches the analytic cosh
     `1 - cosh((z-L/2)/Lc)/cosh(L/2Lc)`, `Lc = sqrt(kappa A Rth')` (4.5e-4, node grid); a ramped `q(z)`
     -> monotone `T(z)`; `gain_per_m_thermal(.,.,T0) == gain_per_m_slices` (0.0) and a hot `T(z)` lowers
     the local gain; the external-FEM sampling seam reproduces the field and stays finite. Scope
     (honestly labelled): STEADY (the transient is the lumped `Cth`'s job), GS band (ES is a
     refinement -- `gain_per_m_thermal` RAISES if the ES band is active rather than silently dropping
     it), 1-D LONGITUDINAL fin (transverse conduction lumped into `Rth'`); the FEM seam is how you
     obtain the full 2-D/3-D heat field. `gain_per_m_thermal` reads the COLD comb and is MUTUALLY
     EXCLUSIVE with the lumped `set_temperature` (no double-count). The reduced fin is the in-gate
     reference; the full `carriers/thermal_fem` NGSolve solve drives the SAME interface (via
     `sample_T_along_axis`, a one-way point-sampler). Adversarial pass (3 lenses + judge, verdict
     fix-then-ship -> all fixes applied): the two disputed P1 "refutations" were both checked against
     source and dismissed -- the insulated Neumann row is already the 2nd-order mirror ghost
     (`-(2c+s)`/`+2c`, manufactured-solution convergence rate 2.00), and the "cell-grid Dirichlet
     mis-placement" was a Gate-D illustrative-dome `dz` bookkeeping choice (the accuracy-backing Gate B
     uses the correct node grid). No P1 code/sign/factor/physics bug survived; the fin signs, lumped
     reduction, analytic-cosh dome, red-shift direction, and GS-only identity are all confirmed. The
     applied fixes were P2/P3 wording + two extra gate pins (uniform-hot == `set_temperature`; ES-band
     guard) + the honest STAND-IN label on the Gate-E sampler smoke-test.

---

## 6. Governing equations (reference)

**QD multi-state rate equations** (occupation probabilities `ρ_ES`, `ρ_GS`;
WL carrier density `N_w`; `V_a` active volume; Pauli-blocking factors `(1−ρ)`):

```
dN_w/dt   = I/(q V_a) − N_w/τ_cap·(1−ρ_ES) + (N_q/V_a)·ρ_ES/τ_esc − N_w/τ_w,sp
dρ_ES/dt  = (N_w τ-coupling)(1−ρ_ES) − ρ_ES/τ_ES→GS·(1−ρ_GS)
              + ρ_GS/τ_GS→ES·(1−ρ_ES) − ρ_ES/τ_esc − ρ_ES/τ_sp
dρ_GS/dt  = ρ_ES/τ_ES→GS·(1−ρ_GS) − ρ_GS/τ_GS→ES·(1−ρ_ES)
              − v_g g S /N_q − ρ_GS/τ_sp        ← stimulated term = field coupling
```

Spectral gain (sum over the inhomogeneous dot-size distribution `D(ν)`):

```
g(ν) = Σ  N_q σ(ν) (2 ρ_GS − 1) · D(ν)
```

The `v_g g S` term in `dρ_GS/dt` is the dynamic field↔population coupling — the
piece to add. Exact bookkeeping follows the standard QD-SOA rate-equation
literature.

**Traveling-wave propagation** (signal power `P`, confinement `Γ`, internal loss
`α_i`; ASE source in the last term, solved both propagation directions):

```
∂P/∂z + (1/v_g) ∂P/∂t = (Γ g − α_i) P + Γ g n_sp h ν Δν_sp
```

**ASE / noise figure** (inversion factor `n_sp`):

```
n_sp = ρ_GS / (2 ρ_GS − 1)
NF   ≈ 2 n_sp (G − 1)/G + 1/G      →  2 n_sp   at high gain
```

**Detection (direct detection), per noise bandwidth `B`** — variances that set
the SNR:

```
shot           : σ²_sh  = 2 q (R P_out) B
signal–spont   : σ²_ssp = 4 R² P_sig S_ASE B
spont–spont    : σ²_spsp ∝ R² S_ASE² Δν B
```

**Analog precision:**

```
SFDR (3rd order) = (2/3)(IP3_out − N_floor)      [dB·Hz^(2/3)]
ENOB             = (SNDR_dB − 1.76) / 6.02
```

where `SNDR` combines the compression-induced distortion (Section 4.2/4.3) with
the total noise floor (shot + ASE beat + source RIN + detector/TIA).

---

## 7. One-line conclusion

DynaMeta has a genuine laser-gain core (`laser_gain.py`) and a reusable
ultrafast carrier-heating model (`carrier_heating.py`), but no
traveling-wave **semiconductor** amplifier with dynamic saturation or ASE. The
missing physics — QD band gain with an injection pump, signal-driven inversion
depletion, z-resolved propagation, and an ASE/noise → SFDR/ENOB chain — is a
focused module (Phases 1–4), not a rewrite, and reuses existing DynaMeta
machinery throughout.


---

## 8. Assessment and required corrections (2026-06-11)

Method: every Section 3 reuse claim was checked against the actual source (file:line); every
Section 6 equation was adversarially reviewed and each finding then re-verified by an
independent skeptic that tried to refute it (so a defensible modeling convention is not
reported as an error); the "full deep physics" bar was anchored in the QD-SOA literature
(Sugawara MST 13:1683 2002; Berg & Mork JQE 40:1527 2004; Berg/Mork/Hvam NJP 6:178 2004;
Uskov; Ben-Ezra/Qasaimeh JLT 21:1690 2003; Loh JQE 47:66 2011 for NF; Olsson JLT 7:1071 1989
for beat noise). Net verdict: the plan is correct, the reuse is real, and it is additive --
but Section 6 as written is a sketch with conservation/normalization gaps, and the SFDR/ENOB
goal forces several "optional" items to be core. Nothing below is a rewrite; all of it is the
same focused module, specified correctly.

### 8.1 Factual corrections (Sections 3, 5)

- **`transient_optics.py` is top-level**, `dynameta/transient_optics.py`, NOT
  `carriers/transient_optics.py`. A builder importing `carriers.transient_optics` verbatim
  hits ImportError. (`transient.py` IS under `carriers/`; behavior of both is as described.)
- **The "existing Innolume-parameterized SOA/BOA model" does not exist** anywhere in the
  repo (the only hit is the substring "boa" in "checkerboard"). The QD parameter set is new
  datasheet-digitization work, not reuse -- scope it explicitly.

Everything else in Section 3 is accurate: `laser_gain.py` is confirmed a four-level ATOMIC
scheme with constant pump, CLAMPED inversion, and the dynamic field<->population coupling
explicitly flagged as the follow-on (header:23-25); `carrier_heating.py` exposes
`TwoTempParams` + `two_temperature_response` by those exact names.

### 8.2 Additivity guardrails (the user's hard requirement)

The extension is **additive-with-care**. To keep it additive:

- **Build a standalone subpackage `dynameta/optics/soa/`** (match the `fdtd_nd/` /
  `carriers/thermal_fem/` package pattern; the spec proposes 4+ objects). The
  traveling-wave SOA is a **time-domain split-step z-t marcher** -- it must NOT plug into
  the `run_pipeline` metasurface `optical_solver` seam (that is a frequency-domain,
  steady-state, once-per-(bias, wavelength) callable returning `OpticalResult`), and must
  NOT route through the normal-incidence metasurface FDTD kernel (`_run_2d_te` /
  `kernels2d.py`). It returns its own waveform/trajectory objects.
- **Do NOT edit `laser_gain.py`, `FourLevelSystem`, or the `kernels2d.py` gain/gain_dyn
  recursion.** Section 4 item 2 ("the stimulated term laser_gain.py omits") must be read as
  "build it fresh in the new module," not "add it to laser_gain." Editing those paths would
  regress, in order of fragility: `validation/fdtd_lasing_cavity.py` (GATES A-E),
  `validation/fdtd_gain_saturation.py` (A-D), `validation/fdtd_gain_medium.py`,
  `tests/test_fdtd_chi2_raman.py` (FourLevelSystem asserts), and the cupy/numba parity gates
  `fdtd_gpu_nonlinear.py` / `fdtd_nonlinear_backends.py`.
- **Reuse, don't modify:** import `small_signal_gain_per_m` and the cavity closed forms as
  Phase-1 sanity oracles (but see 8.6 -- not as a direct equality); import
  `TwoTempParams` / `two_temperature_response` as-is for the optional carrier-heating term,
  feeding SOA absorbed-power density as `intensity_of_t`. Do NOT make `carrier_heating.py`
  gain-aware (it backs `carrier_heating_enz`, a SMOKE-tier gate).
- **Conventions:** `exp(-i omega t)`, Im(eps) > 0 (gain -> Im(chi) < 0); SI units, constants
  from `dynameta.constants`; **ASCII-only source and prints** (this .md uses Unicode, but the
  code must spell Greek out); validations are `main()->bool` + `raise SystemExit(0 if ...)`
  with sequential `[tag] GATE` prints and a reduces-to-known-limit off-switch as an actual
  gate; frozen dataclasses with `__post_init__` ValueError validation; pure numpy/scipy core
  with an optional `xp` GPU seam; keep new names behind `__all__` and out of the lazy ngsolve
  re-export in `optics/__init__.py`.

### 8.3 Gain-model corrections (Section 6 rate equations) -- required before Phase 1

The rate equations are dimensionally and conservation-wise incomplete as written. Required
fixes (rho_X = per-state occupation in [0,1]; mu_GS = 2, mu_ES = 4 the state multiplicities;
N_q = areal/volumetric dot density; group index j over the size distribution):

1. **Particle conservation (blocker).** The WL<->ES capture/escape must appear as ONE
   conjugate flux pair with a consistent per-dot<->per-volume normalization `mu_ES*N_q`
   [m^-3]. As written, the ES-occupation escape loss and the WL density gain do not balance.
   Capture: `drho_ES/dt|cap = +[N_w/(tau_cap*mu_ES*N_q)]*(1-rho_ES)`; conjugate WL loss
   `dN_w/dt|cap = -(N_w/tau_cap)*(1-rho_ES)`. Escape: `drho_ES/dt|esc = -(rho_ES/tau_esc)`,
   conjugate `dN_w/dt|esc = +(mu_ES*N_q/tau_esc)*rho_ES`. The vague `(N_w tau-coupling)`
   placeholder is pinned by this balance, not a free constant.
2. **Stimulated-term normalization.** The per-dot rate is intrinsically per-dot -- there is
   no `/N_q`. The photon density that depletes the dot is the CONFINED density
   `S_conf = Gamma*P(z,t)/(v_g*h*nu*A_mode)`. Use
   `drho_GS,j/dt|stim = -v_g*sigma_pk*L_hom(nu_s - nu_j)*(2*rho_GS,j - 1)*S_conf`, and the
   SAME `sigma_pk*L_hom`, `mu_GS`-weighted `N_q` must build the modal gain g(nu_s) in dP/dz
   (photon-number conservation forces this coupling).
3. **Group-resolved populations = spectral hole burning (promote to CORE).** A single global
   rho_GS makes the inhomogeneous ensemble saturate in lockstep and CANNOT produce SHB.
   Give each size group j its own rho_GS,j (and rho_ES,j) depleted by the LOCAL photon
   density through the homogeneous Lorentzian L_hom; net gain `g(nu) = Sum_j w_j ...`,
   `w_j = D(nu_j) dnu`. This is not an add-on -- it is the gain saturation of items 1-3 done
   correctly, and per the literature it is the dominant high-speed mechanism.
4. **Electron-hole separation (CORE, or document the ceiling).** Net stimulated gain is
   `(f_c + f_v_hole - 1)`, not `(2*rho_GS - 1)`; the latter is the excitonic/charge-neutral
   special case `f_c = f_v = rho`. The field standard for pattern/amplitude fidelity tracks
   separate electron and hole occupations with DIFFERENT capture/relaxation times. Either
   adopt `(f_GS^e, f_GS^h)` per level, or keep the single-rho model and document it as an
   explicit fidelity ceiling on pattern effects.
5. **Spontaneous recombination is nonlinear.** Confined states recombine as `rho^2/tau_sp`
   (under the single-rho convention; `f_e*f_h/tau_r0` under e/h); the WL recombines
   bimolecularly + Auger `-B*N_w^2 - C*N_w^3`, with the linear `-N_w/tau_w,sp` kept only as
   the SRH/defect channel.
6. **Detailed balance + degeneracy.** Make escape a DERIVED (T-dependent) quantity:
   `tau_esc/tau_cap = (mu_ES*N_q/N_c_WL)*exp(E_b/kT)` and
   `tau_GS->ES/tau_ES->GS = (mu_ES/mu_GS)*exp(Delta_E/kT)`, so the dark device relaxes to the
   correct Fermi-Dirac occupancy. Pin `mu_GS = 2`, `mu_ES = 4` once and use them consistently
   in both the rate equations and the gain sum.

### 8.4 Propagation / noise / metric corrections (Section 6)

1. **Noise figure is the lossless ideal.** `NF ~ 2 n_sp` omits internal loss and input
   coupling, which a ~5 dB target depends on:
   `NF = (1/eta_in)*[2*n_sp*(Gamma g)/(Gamma g - alpha_i)*(G-1)/G + 1/G]`.
2. **Facet residual reflectivity caps ENOB and is absent.** A single-pass TWA with facet
   reflectivity R gives Airy gain ripple `dG = 20 log10[(1+G*sqrt(R1 R2))/(1-G*sqrt(R1 R2))]`:
   at G = 20 dB, R = 1e-4 -> 0.17 dB ripple (ENOB ceiling ~5.6 bits); R = 1e-3 -> 1.7 dB
   (~2 bits). The spec MUST add a facet-reflectivity / gain-ripple budget -- it directly
   corrupts the analog transfer curve.
3. **n_sp must track the occupation convention.** `n_sp = rho_GS/(2 rho_GS - 1)` is correct
   AS WRITTEN under the excitonic single-rho convention (verified -- not a bug there). But if
   the e/h upgrade (8.3 item 4) is taken, n_sp must upgrade consistently to
   `f_c(1-f_v)/(f_c - f_v)`, evaluated at the SATURATED, group-local populations.
4. **Distortion at GHz is dynamic, not memoryless (promote to CORE).** Static
   `SFDR = (2/3)(IP3_out - N_floor)` is only the low-frequency/memoryless asymptote. A
   saturating SOA has memory: `dg/dt = (g0 - g)/tau_c - g*P/E_sat`, so IP3 is
   frequency-dependent and rolls off as tone spacing exceeds 1/tau_c -- exactly the regime
   Section 2 says the device lives in. Add: (a) a two-tone `IMD3(delta_f)` swept across tone
   spacing (report the roll-off vs 1/tau_recovery), and (b) a true symbol-stream SNDR/EVM
   from the dynamic solve. Keep the static IP3 only as the DC anchor. This dynamic metric IS
   the SFDR/ENOB the project cares about.
5. **Self-heating is required for any predistortion-ENOB claim (promote to CORE).** Static
   predistortion is only stable if the operating point does not drift: half-LSB retention
   needs `dT_max = 2^-(N+1) / ((1/G)(dG/dT))`. For typical dG/dT this is sub-kelvin at 8-10
   bits -- so a self-heating / thermal model gates the achievable ENOB and cannot be optional
   when predistortion is in the precision budget.
6. **Smaller fixes.** Propagate ASE spectrally-resolved and bidirectionally (per nu slice),
   not lumped; GVD is optional at these bandwidths. Detector shot noise must include the ASE
   and dark terms `2q R (P_sig + P_ASE) B + 2q I_dark B`; the spont-spont coefficient is
   `2 m_pol R^2 S_ASE^2 (2 dnu_o - B) B` (m_pol = ASE polarizations accepted). State the
   SFDR/IP3 explicitly in the DETECTED photocurrent domain with N_floor as a PSD [dBm/Hz] so
   the [dB.Hz^(2/3)] units are self-consistent.

### 8.5 Reclassification: optional -> core (to actually hit SFDR/ENOB)

| Item | Spec class | Correct class | Why |
|---|---|---|---|
| Spectral hole burning (group-resolved populations) | optional (4.6) | **CORE** | It IS the gain saturation of items 1-3 done right; dominant high-speed mechanism |
| Separate electron/hole occupations | implicit single-rho | **CORE** (or documented ceiling) | Required for pattern/amplitude fidelity; literature standard |
| Dynamic (pattern-memory) distortion metric | only static IP3/SFDR | **CORE** | At GHz the distortion is dynamic; static SFDR is the DC asymptote |
| Self-heating / thermal drift | optional (4.6) | **CORE** for predistortion ENOB | Predistortion stability is sub-kelvin at 8-10 bits |
| Carrier heating (TTM, `carrier_heating.py`) | optional (4.6) | genuinely secondary | Ultrafast refinement; reuse TwoTempParams when needed |
| FWM / multi-lambda | optional (4.6) | genuinely optional | Only if the OVMM is WDM |

### 8.6 Corrected Phase-1 oracle

The Phase-1 check "small-signal limit matches `laser_gain.small_signal_gain_per_m`" is
**not a valid equality**: that function returns the line-center intensity gain of a single
classical Lorentz-oscillator ADE (atomic four-level), a different physical quantity from
semiconductor QD band gain summed over an inhomogeneous ensemble. Replace it with: (a)
internal consistency -- QD g(nu) reduces to `sigma_pk*N_eff*(2 rho_GS - 1)` in the
single-group homogeneous limit and integrates to the modal gain; (b) Einstein-relation
closure between the stimulated cross-section and the spontaneous rate; (c) an INDEPENDENT
QD-SOA literature oracle -- saturation power rising with pump current and gain-recovery time
landing in the published ps range (Berg & Mork 2004), and NF approaching 2 n_sp -> ~3 dB at
high gain. Keep `cavity_photon_lifetime_s` / `threshold_inversion_m3` as cold-cavity sanity
checks only.

### 8.7 Bottom line

Additive: **yes**, if built standalone in `dynameta/optics/soa/` with no edits to
`laser_gain.py` or the FDTD gain kernels. Reflects full deep physics: **not yet** -- Section
6 needs the 8.3/8.4 corrections (conservation, stimulated normalization, group-resolved SHB,
e/h or a documented ceiling, internal-loss NF, facet ripple) and the 8.5 reclassifications,
all of which fit the existing phased plan without expanding its scope.
