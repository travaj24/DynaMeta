# Rare-Earth Fiber Amplifier (EDFA / YDFA) Model Spec

Governing-equation source of truth for `dynameta/optics/fiber_amp/`. Formulation extracted
from the primary literature (deep-research pass, 2026-07-16, verified 3-0 unless noted). SI
units, `exp(-i omega t)` (gain -> Im(chi) < 0), pure numpy/scipy, ASCII.

## 0. Primary references
- **Giles & Desurvire**, "Modeling Erbium-Doped Fiber Amplifiers," *JLT* 9(2):271 (1991) --
  THE two-level coupled-power EDFA model. [OSTI 5843366]
- **Desurvire**, *Erbium-Doped Fiber Amplifiers: Principles and Applications* (Wiley, 1994).
- **Paschotta, Nilsson, Tropper, Hanna**, "Ytterbium-doped fiber amplifiers," *IEEE JQE*
  33(7):1049 (1997) -- YDFA quasi-three-level.
- **Barnard, Myslinski, Chrostowski, Kavehrad**, "Analytical Model for Rare-Earth-Doped Fiber
  Amplifiers and Lasers," *IEEE JQE* 30(8):1817 (1994), DOI 10.1109/3.301646 -- unified
  2/3/4-level closed form; the analytic cross-check for the numerical BVP.
- **Frantz & Nodvik**, "Theory of Pulse Propagation in a Laser Amplifier," *JAP* 34:2346
  (1963) -- saturable-gain pulse energy extraction.
- **McCumber**, *Phys. Rev.* 136:A954 (1964); **Miniscalco & Quimby**, *Opt. Lett.* 16:258
  (1991) -- emission-from-absorption cross-section relation for rare-earth ions.

## 1. The Giles-Desurvire coupled-power model (EDFA; Er = homogeneous two-level)

Each optical channel k (pump, signal, or an ASE spectral bin) carries power `P_k` and
propagates along z with direction `u_k = +1` (forward) or `-1` (backward). Let `nbar2 = N2/n_t`
be the FRACTIONAL upper-level (metastable `4I13/2`) population, `n_t` the ion density.

**Giles parameters** (spectral, directly measurable -- preferred over separate sigma/Gamma/n):
```
alpha(lambda) = sigma_a(lambda) * Gamma(lambda) * n_t      [1/m]  (absorption spectrum)
g*(lambda)    = sigma_e(lambda) * Gamma(lambda) * n_t      [1/m]  (gain spectrum)
```
Gamma(lambda) = mode/dopant overlap integral; sigma_a/sigma_e = absorption/emission
cross-sections [m^2].

**Propagation ODE** (finding [0], verified 3-0):
```
dP_k/dz = u_k * [ (alpha_k + g*_k) * nbar2 - alpha_k - l_k ] * P_k
        + u_k * g*_k * nbar2 * m * h * nu_k * dnu_k
```
- Term 1 (net stimulated): `(alpha_k + g*_k) nbar2 - alpha_k = g*_k nbar2 - alpha_k (1 - nbar2)`
  = `Gamma n_t [sigma_e nbar2 - sigma_a (1 - nbar2)]` -- the local gain coefficient.
- Term 1 also carries `- l_k P_k`, the wavelength-dependent BACKGROUND loss (fiber attenuation).
- Term 2 (ASE spontaneous seeding): `m` = number of modes = **2** (two orthogonal
  polarizations of the fundamental fiber mode); `dnu_k` the bin width [Hz]. Only nonzero for
  ASE channels (a pump/signal channel adds no spontaneous term). [finding 3]

**Steady-state metastable inversion** (finding [1], verified 3-0):
```
nbar2 = ( SUM_k tau * sigma_ak / (h nu_k) * P_k * ibar_k )
        / ( 1 + SUM_k tau * (sigma_ak + sigma_ek) / (h nu_k) * P_k * ibar_k )
```
`tau` = upper-state lifetime; `ibar_k = Gamma_k / A_dope` the overlap-normalized intensity
per unit power [1/m^2] (so `P_k * ibar_k / (h nu_k)` is the modal photon-flux-density rate).
Equivalently `nbar2 = R_a tau / (1 + (R_a + R_e) tau)` with pump-rate `R_a = SUM sigma_a
Gamma P/(h nu A_dope)`, stimulated-emission-rate `R_e = SUM sigma_e Gamma P/(h nu A_dope)`.
Saturation coefficient `zeta = pi * b_eff^2 * n_t / tau` (b_eff = effective doped radius =
core radius for uniform doping).

**Two-point boundary value problem**: forward pump/signal known at z=0, backward ASE (and any
counter-pump) known at z=L (= 0 seed + spontaneous). Solve by relaxation / shooting; nbar2(z)
is algebraic in the local powers, so the ODE set is first-order in P only.

## 2. Yb quasi-three-level (YDFA; Paschotta 1997)

Same two-level coupled-ODE STRUCTURE (Barnard's unified model spans 2/3/4-level), with the
Yb spectroscopy: `2F7/2` ground, `2F5/2` upper, tau ~ 0.8-1.5 ms. The quasi-three-level
character is that the SIGNAL band (1000-1080 nm) has NON-NEGLIGIBLE ground-state absorption
sigma_a(signal) > 0 (unlike an ideal 4-level system), so short-wavelength signals see
reabsorption and the fiber must be pumped hard to reach transparency. Pump 915/940/976 nm.
The same `nbar2` and propagation equations apply with Yb sigma_a/sigma_e.

## 3. McCumber relation (emission from absorption)
```
sigma_e(nu) = sigma_a(nu) * exp( (epsilon - h nu) / (k_B T) )
```
`epsilon` = temperature-dependent excitation chemical potential (net free energy of one
excitation = the zero-phonon-line "zero-line" energy). Ensures detailed balance. Used to
derive sigma_e from a measured sigma_a spectrum (or cross-check the two).

## 4. ASE and noise figure (findings [3], [8])
- Spectrally-resolved forward + backward ASE channels, each seeded by `m h nu dnu` with m=2.
- Single-polarization ASE PSD: `S_ASE(nu) = n_sp * h nu * (G - 1)` (Desurvire).
- Population-inversion (spontaneous-emission) factor:
  `n_sp = sigma_e N2 / (sigma_e N2 - sigma_a N1)` -> n_sp = 1 at full inversion (N1 = 0).
- Amplifier noise figure: `Fn = (2 n_sp (G - 1) + 1) / G ~= 2 n_sp (G - 1)/G`, approaching the
  3 dB (Caves 1982) quantum limit at high gain + full inversion. The `+1/G` is the input
  shot-noise (unamplified) term; keep it for correct low-gain behaviour.

## 5. Representative parameters (literature defaults; findings [5],[6])
Er3+ (aluminosilicate EDF):
- 1560 nm signal: sigma_a = 1.69e-25 m^2, sigma_e = 3.04e-25 m^2.
- 4I13/2 peak ~1530 nm: sigma ~ 5.7e-25 m^2 (Strohhofer-Polman Al2O3).
- 980 nm pump (4I11/2): sigma_a ~ 1.7e-25 m^2. 1480 nm pump: in-band (upper manifold).
- tau(4I13/2) ~ 10 ms (tau = 0.01 s). n_t ~ 1e25 m^-3, core radius ~ 1.5-2 um.
Yb3+ (2F5/2 <-> 2F7/2):
- Peak sigma_abs ~ 2.7e-24 m^2 at 976 nm (aluminosilicate) or 1.4e-24 m^2 at 974.5 nm
  (phosphosilicate); band ~850-1000 nm; peak ~7x the Er 980 nm value.
- tau(2F5/2) = 0.83 ms (aluminosilicate) / 1.45 ms (phosphosilicate).

## 6. Concentration / degradation (opt-in; Phase 5)
- Er cooperative (homogeneous) UPCONVERSION: adds `-C_up * N2^2` to the N2 rate (two excited
  ions -> one higher + one ground), and pair-induced quenching (fast decay of a fraction of
  clustered ions). C_up ~ 1e-24 ... 1e-23 m^3/s host-dependent.
- Yb PHOTODARKENING: a slowly-growing background loss `l_PD(z, t)` (color-center formation),
  scaling super-linearly with inversion; time-dependent.

## 7. Cladding pumping / thermal (Phase 6)
- Double-clad: pump overlap with the CORE `Gamma_p ~= A_core / A_clad` (pump fills the inner
  cladding, only the core fraction is absorbed) -- the key high-power-Yb geometry factor.
- Quantum-defect heat load per unit length `q_heat = (1 - lambda_p/lambda_s) * (dP_p_absorbed/dz)`;
  radial thermal profile from the heat equation. Stokes efficiency ceiling `lambda_p/lambda_s`.

## 8. Pulsed extraction -- Frantz-Nodvik (finding [9])
Fluence form (input fluence -> output fluence through a saturable gain of small-signal gain G0):
```
E_out = E_sat * ln{ 1 + [exp(E_in / E_sat) - 1] * G0 }
```
`E_sat = h nu * A_core / (Gamma (sigma_a + sigma_e))` intrinsic saturation energy; `G0 =
exp(g0 L)` initial (unsaturated) single-pass gain. Extractable energy ~ `E_sat * ln(G0)`.
Validation datapoint: 20 um core aluminosilicate at 1560 nm -> E_IS = 84.5 uJ; 34 dB initial
gain, 22.5 uJ seed -> ~0.57 mJ FNE prediction (measured 0.8 mJ, +1.5 dB, unquenched).

## 9. Validation benchmarks (gate targets; findings [8],[9],[10])
- **NF quantum limit**: fully-inverted high-gain amplifier -> NF -> 3.0 dB (n_sp = 1).
- **Energy conservation**: absorbed pump = signal gain + total ASE + quantum-defect heat
  (lossless host) -- closes to machine precision.
- **Reduce-to-analytic**: unpumped fiber -> pure Beer-Lambert absorption `exp(-alpha L)`;
  fully-inverted, no ASE, small signal -> `G = exp((g* - l) L)`.
- **Barnard closed form**: the numerical BVP must match the Barnard analytic gain on a case
  its assumptions cover (no ASE, uniform inversion) to tight tolerance.
- **McCumber consistency**: sigma_e derived from sigma_a reproduces the measured sigma_e.
- **Frantz-Nodvik**: pulse-extraction module matches the fluence formula (and the 84.5 uJ /
  0.57 mJ datapoint order-of-magnitude).
- **YDFA slope efficiency**: core-pumped Yb near 980 nm -> slope efficiency approaches the
  quantum-defect ceiling `lambda_p/lambda_s` (~92-94%); record 90.7% experimental.
- **Yb reabsorption**: a short-wavelength (~1030 nm) signal in an under-pumped Yb fiber is
  ABSORBED (net loss), turning to gain only above the transparency pump level -- the
  quasi-three-level signature.

## 9a. Achieved (implementation `dynameta/optics/fiber_amp/`, 34 pytest gates + validation.fiber_amp_physics)
The steady state uses a RELAXATION solver (alternating forward/backward IVP sweeps), not
solve_bvp (which overflows on ASE growing from the spontaneous floor through tens of dB).
- Beer-Lambert: unpumped 8 m EDF -> -20.66 dB vs -20.67 dB analytic (0.007 dB).
- Photon conservation: (signal+ASE photons gained)/(pump photons lost) = 0.915 <= 1.
- Gain saturation: 1 uW->2 mW input compresses gain 24.1->14.7 dB, quenches fwd-ASE 20.3->0.14 mW.
- NF: local n_sp >= 1 (exact) all configs; a long high-gain fiber is ASE-clamped at a realistic
  3.66-3.84 dB, while a short heavily-doped preamp reaches NF -> 2.96 dB (n_sp -> 1.00), the
  3.01 dB quantum floor. NF(PSD) == (2 n_sp(G-1)+1)/G to 1e-6.
- Slope efficiency 0.602 <= Stokes 0.628 (96%); PCE 0.597 < ceiling. Gain-tilt peak migrates
  1532.5 -> 1537.5 nm to the red as inversion drops 0.90 -> 0.45.
- Concentration OPT-IN: concentration=None is byte-identical to an all-default ConcentrationModel.
  Upconversion clamps nbar2/gain; 10% PIQ -> 1.65 dB unbleachable penalty; photodarkening ~nbar2^7
  costs 4.99 dB at nbar2=0.996 (915 nm pump) vs 0.13 dB at 0.499 (976 nm zero-line) -- the latter
  re-validating the Yb quasi-3-level 50% inversion cap when pumped on the zero line.
- Cladding overlap ratio 7.67e-3 = Gamma_p/Gamma_core(980) to 0.03%. Heat balance exact
  (pump_abs - sig_add - ASE_out == F(0)-F(L) == integral Q dz). Brown-Hoffman centre rise matches
  an independent FD solve of the cylindrical heat equation. qd(Yb 976->1030)=5% << qd(Er)=37%.
- Transient (nbar2(z,t), quasi-static powers, exp integrator) relaxes to the steady gain to
  0.0019 dB; gain recovery tau_eff = 0.87 ms << bare 10 ms; add/drop XGM +23 dB.
- Frantz-Nodvik: small E_in -> G0 E_in; large -> E_in + E_sat ln G0 (stored) exactly; temporal
  P_out(t) integral matches; leading edge G0, trailing -> 1 in deep saturation.
- Calibration: measured cross-section tables (CrossSectionTable) and vendor Giles alpha/g*
  (giles_calibrated_fiber, overlap_override folds Gamma in) drive the same solver; Giles
  round-trip reproduces alpha(1530) to 8e-5.

## 10. Accuracy extensions (Phases 9-11; all opt-in, byte-identical when off)
- **Excited-state absorption (Phase 9)**: sigma_esa on the ion adds a parasitic beam loss
  -Gamma n_t sigma_esa nbar2 (cycling limit; inversion unchanged). erbium(esa=True) 980 nm pump
  ESA costs ~1.5 dB gain and pushes heat/pump_abs 0.43 -> 0.83; monotonic; localized on the pump.
  Yb is ESA-free (one excited 4f manifold).
- **Temperature (Phase 10)**: at_temperature McCumber-scales sigma_e from a reference T
  (sigma_a held); T=T_ref byte-identical; zero-line crossover T-invariant to 1e-12; captures the
  EDFA gain-tilt-with-T (1560 nm gain 24.0 -> 20.8 dB over 280 -> 360 K). multiphonon_lifetime =
  Miyakawa-Dexter energy-gap law W_nr = coupling exp(-alpha*gap)(nbar+1)^p (large gap ~ radiative,
  small gap quenched, tau falls with T).
- **Detector beat noise (Phase 11)**: detection_noise gives shot / signal-spont / spont-spont
  photocurrent variances, electrical SNR, added RIN, and a beat-noise NF that reduces to the
  optical NF to <0.05 dB in the sig-spont-dominated limit (cross-check). sig-spont dominates at
  high signal, spont-spont at low; an optical filter (smaller B_o) cuts spont-spont.

## 11. Pulsed / chirped-pulse amplification (Phases 12-14; pulse.py, cpa.py)
The envelope A(z,t) [sqrt(W)] evolves by the GNLSE
    dA/dz = i(beta2/2 d2/dt2 + beta3/6 d3/dt3)A + (g-alpha)/2 A + i gamma |A|^2 A,
solved by the symmetric split-step Fourier method (dispersion+gain in frequency, Kerr in time),
accumulating the B-integral gamma INT P_peak dz. Ref: Agrawal, "Nonlinear Fiber Optics";
Strickland-Mourou (CPA).
- **GNLSE core (Phase 12)**: validated against Gaussian dispersive broadening
  T(z)=T0 sqrt(1+(z/L_D)^2), SPM spectral broadening (phi_max=4.5 pi -> 5 peaks, time envelope
  preserved), the fundamental soliton (N=1, beta2<0 -> shape-invariant; pins the sign), energy
  conservation to 9e-13, and flat gain -> exp(gL).
- **Saturable spectral gain (Phase 13)**: SaturableGain g(omega,E) = g_small/(1+E/e_sat)
  shape(omega). GAIN NARROWING (parabolic band) obeys 1/Omega_out^2 = 1/Omega_in^2 + G0/Omega_g^2
  to 0.01% -- the effect that bounds the recompressed pulse. Couple e_sat to
  dynamics.saturation_energy, g_small to the CW inversion.
- **CPA chain (Phase 14)**: seed -> stretch(+GDD) -> amplify(GNLSE) -> compress(-GDD). Linear
  recompression recovers a transform-limited pulse (Strehl 1.0000, exact FWHM); stretching lowers
  the in-fiber peak power and hence the B-integral; and the B-integral is the compression
  killer -- B=1.56 rad -> Strehl 0.75, B=5.2 rad -> 0.24 (the "keep B < ~1-2 rad" design rule).
  Strehl = compressed peak / transform-limited peak.

Still on the table (documented, not built): SRS/SBS and transverse-mode instability (high-power
limiters), thermal lensing / transient heat diffusion, Er:Yb co-doping, full transverse/
polarization-resolved gain, and the delayed Raman response / self-steepening terms of the full
GNLSE (self-frequency shift) -- the current pulse core carries dispersion + Kerr SPM.
