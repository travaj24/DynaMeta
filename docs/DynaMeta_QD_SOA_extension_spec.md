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
     waveforms, no divide-by-field). GVD (background group-velocity dispersion) remains optional.
   - **Speedup -- SHIPPED 2026-06-19**: byte-identical Lorentzian/prefactor caching (1.1-1.3x) +
     an OPT-IN numba carrier-step accelerator `QDGainModel(fast=True)` (bit-parity 1e-16,
     ~7x on the carrier RK4, 3-4.4x on the full marcher; default OFF for byte-stable validations).
     `validation/qd_soa_numba_parity.py`. The remaining lever is a full-marcher JIT (~1.5-2x more,
     deferred for its duplicate-logic maintenance cost); GPU is not worthwhile (cache-resident
     (nz x ng) arrays are launch-bound, same conclusion as the FDTD 'auto' backend).

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
