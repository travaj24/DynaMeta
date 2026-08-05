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
- **Frantz-Nodvik temporal reshaping (Phase 13, OPT-IN; audit A-10)**: the law above is the CW
  one -- `E` is the TOTAL pulse energy at that z, one scalar over the whole time window, so it
  cannot reshape a pulse. Correct for a pulse long against the upper-state lifetime; in the
  short-pulse (CPA) regime it under-extracts by up to 27% (measured, mid-regime) and produces no
  edge asymmetry at all. `SaturableGain(frantz_nodvik=True)` applies sec.8's instantaneous law
  G(t) = G0/(G0-(G0-1)exp(-U_in(t)/E_sat)) in the TIME domain each split-step half-step, with
  G0 = exp(g_small dz) the sub-step small-signal gain; the map is linear in u = expm1(U/E_sat)
  with factor G0, so slicing the fiber is exact. A non-flat band composes as a RELATIVE filter
  exp[0.5 g_realized (shape(omega)-1) dz] at the FN-realized (saturated) gain, so narrowing
  stops as the amplifier saturates and a flat band is exactly the identity. Finite
  `recovery_time_s` integrates dg/dt = (g0-g)/tau - g P/E_sat and restores the CW law
  g_small/(1 + P tau/E_sat) for pulses long against tau. Measured: E_out tracks
  frantz_nodvik_output_energy to 2e-6 (2nd-order in dt), a direct method-of-lines ODE integration
  of the PDE pair to 1e-7..2e-6, 1-step == 400-step to 5e-5, leading/trailing gain ratio 141 on a
  flat top (CW: 1.000) with the centroid moving 26 ps EARLIER, and the CPA chain gains +1.09 dB
  over the CW law on the same seed. OFF by default and byte-identical when off. A negative
  `g_small_per_m` -- a saturable ABSORBER -- is legal on all three branches (CW, tau = inf,
  finite tau; audit W5-3 fixed the last, which returned an all-NaN field).
  - *Order (audit W5-1/W5-2).* `recovery_time_s = inf` is the EXACT slab map, so the split keeps
    its O(h^2) (measured 2.04 with beta2 = 1 ps^2/m and gamma = 3e-4 /W/m in the same step). A
    FINITE `recovery_time_s` switches to the thin-slab exponential-Euler gain, which is O(h) and
    drops the WHOLE propagator to first order (measured 1.06 on the same case; the branch-to-
    branch gap converges at order 1.00 over a 64x refinement, not 2). Spell "no recovery" as
    `np.inf`, never 1e30: the huge-but-finite value takes the first-order branch and disagrees
    with the exact one by 0.39% at n_steps = 100, falling only as 1/n_steps.
  - *Window containment (audit W5-4).* The FN factor builds U(t) = INT P dt' from the LEFT EDGE
    of the periodic FFT window, so it is the one operator here that is NOT roll-equivariant: a
    pulse straddling the boundary has its leading part treated as trailing (saturated instead of
    unsaturated), a silent 43.8% energy error on a half-window cyclic shift, while the CW path is
    exact under the same roll. FN mode now measures the edge power each half-step and warns
    (RuntimeWarning) once above 1e-3 of the peak. Keep the pulse centred and the window wide.
  - *Relative band filter accuracy (audit W5-5).* The exact homogeneous system is
    dP_i/dz = s_i n P_i with dn/dt = -(n/E_sat) SUM_j s_j P_j -- one shared inversion depleted by
    the SHAPE-WEIGHTED total power. The filter reproduces the log-gain RATIO s_i/s_j exactly (the
    gain-narrowing physics), but the FN factor saturates on the UNWEIGHTED total power at the
    centre gain, so wing energy over-saturates the model and it UNDER-extracts. Measured against a
    two-line RK4 oracle (parabolic band, Omega = 3e13 rad/s): exact to 5e-5 with all energy at
    the band centre, -0.69% at the shipped CPA example's exposure (mean shape 0.975,
    E_in = 0.5 E_sat -- pinned as a regression in tests/test_fiber_amp.py), -0.91% at the same
    band with E_in = 3 E_sat, and -13.4% in a designed worst case (half the energy at
    shape = 0.305, E_in = 3 E_sat). Use a flat band or the CW path if the wing energetics must be
    better than that.
- **CPA chain (Phase 14)**: seed -> stretch(+GDD) -> amplify(GNLSE) -> compress(-GDD). Linear
  recompression recovers a transform-limited pulse (Strehl 1.0000, exact FWHM); stretching lowers
  the in-fiber peak power and hence the B-integral; and the B-integral is the compression
  killer -- B=1.56 rad -> Strehl 0.75, B=5.2 rad -> 0.24 (the "keep B < ~1-2 rad" design rule).
  Strehl = compressed peak / transform-limited peak.

## 12. Radially-resolved inversion -- transverse spatial hole burning (transverse.py)

Sec. 1 is the MEAN-FIELD reduction of a model that is really transverse: it replaces every
channel's intensity by its area average over the dopant, `<I_k> = Gamma_k P_k / A_dope`, so
`nbar2` is one number per z. `transverse.py` (`ResolvedFiberAmplifier`) solves the
PRE-REDUCTION form, `nbar2(r, phi, z)`. Formulation source:
`docs/fiber_transverse_grounding_2026_07_29.md` sec. 1 (equation numbers below are its).
Primary references: **A. V. Smith & J. J. Smith**, "Increased mode instability thresholds of
fiber amplifiers by gain saturation," *Opt. Express* 21(13):15168 (2013), Eq. (1) [PINNED];
**Giles & Desurvire** (sec. 0 above) for the `m h nu dnu` seed; **Jiang & Marciante**,
*JOSA B* 25(2):247 (2008) as the originating TSHB-in-LMA reference (paywalled -- its equations
were NOT retrieved, so nothing here is transcribed from it).

**Local balance (F1.1) [PINNED verbatim, Smith & Smith Eq. 1, generalized to K channels]**
```
nbar2(r,phi,z) = SUM_k sigma_a_k I_k /(h nu_k)
               / [ 1/tau + SUM_k (sigma_a_k + sigma_e_k) I_k /(h nu_k) ],
I_k = P_k(z) i_k(r,phi),   n2 = nt(r) nbar2,   INT i_k dA = 1   [i_k] = m^-2   (F1.0)
```
Excitation does not move laterally (ion diffusion is nil in glass), so the balance is strictly
pointwise -- there is **no `L_diff` analogue** as there is in the SOA model. Signal-free clamp:
`nbar2 -> sigma_a_p/(sigma_a_p + sigma_e_p)` (= 0.50305 for the Smith & Smith Yb pair).

**Propagation (F1.2) [DERIVED; reduces EXACTLY to the sec.1 Giles ODE]**
```
dP_k/dz = u_k P_k INT [ sigma_e_k n2 - sigma_a_k (nt - n2) ] i_k dA - u_k l_k P_k
        + u_k m h nu_k dnu_k sigma_e_k INT n2 i_k dA            (ASE channels only)
```
With `nbar2` constant over the dopant, `INT n2 i_k dA = nt nbar2 Gamma_k` and this IS
`[(alpha_k + g*_k) nbar2 - alpha_k - l_k] P_k + g*_k nbar2 m h nu dnu`. The spontaneous term
carries the SAME overlap as the stimulated one (one photon per mode). The ODE state stays
`P(z)`, so the solver is the same relaxation (alternating forward/backward IVP sweeps) as
sec. 1's.

**Closed forms (uniform pump + one Gaussian-mode signal; DERIVED, exported as gate oracles)**
```
nbar2(I_s) = n2_inf + (n2_0 - n2_inf)/(1 + I_s/I_sat)                            (F1.3)
J   = INT_dope nbar2 i dA = n2_inf Gamma + (n2_0 - n2_inf) Phi(s0, Gamma)        (F1.5)
Phi = (1/s0) ln[(1 + s0)/(1 + s0 (1 - Gamma))],  s0 = 2 P_s/(pi w^2 I_sat)       (F1.6)
J_MF= Gamma [ n2_inf + (n2_0 - n2_inf)/(1 + s_area) ],  s_area = s0 Gamma/(-ln(1-Gamma))  (F1.7)
kappa(x) = tanh(x^2)/x^2,  x = b/w   ==>  P_sat_resolved = kappa P_sat_meanfield  (F1.8)
```
`tshb_closed_form_J`, `tshb_mean_field_J`, `saturation_correction_kappa`.

**Mode profiles.** Core-guided channels use the EXACT LP field (`lma.mode_field`) whenever
`V > 2.405`, and the Marcuse Gaussian (`waveguide.mode_field_radius_m`) only below it. At
`V = 8.2` the Gaussian's `Gamma` is 1.1% off but its SATURATION integral is off by up to 13%
(0.85 dB/m) -- the dossier's binding correction. Cladding pumps are flat over the inner
cladding, so `waveguide.cladding_pump_overlap` becomes an OUTPUT of the quadrature. A signal may
carry an explicit `LPMode` (multi-mode competition, mutually incoherent, ONE shared
`nbar2(r,phi,z)`) or the string `"flat"`, the mean-field closure written as a profile.

**Quadrature.** Composite Gauss-Legendre, panel edges at `r = min(a,b)`, `max(a,b)`, the inner
cladding, and `r_max`, plus a radius-doubling refinement of the tail (a single wide outer panel
costs 2.9e-8 in every confinement factor -- measured). 24 nodes/panel default. Azimuthal
Gauss-Legendre on `[0, pi/2]` with a symmetry factor 4, activated only when an `l >= 1` mode is
present; 32 nodes default, set by the LP11 + LP21 both-saturating case, where the relative error
against `n_az = 128` runs `8 -> 4.10e-4`, `16 -> 4.98e-7`, `24 -> 2.90e-10`, `32 -> 1.51e-13`
(gate G2; the repo's bar for this feature is 1e-10). NEVER a step mask on a uniform grid (that
is O(h)).

**Scope / refusals (v1, each a loud `ValueError`).** RamanStokes; ConcentrationModel /
`upconversion_C_up != 0` (upconversion makes the per-node balance quadratic and destroys the
Moebius structure the closed-form gates rest on); `set_temperature_profile` (a z-only McCumber
scaling is meaningless once the inversion is resolved -- it would have to become `T(r,z)`); an
ion with nonzero `sigma_esa` at any channel wavelength; Er:Yb co-doping; and
`FiberSpec.overlap_override` (this solver COMPUTES the overlap; an override would replace the
very quantity being resolved). Three consistency refusals join them: an explicit `r_max_m` inside
the inner cladding while a cladding pump is present (the flat pump would be RENORMALIZED over the
truncated disc, inflating `Gamma_p` by `(R_clad/r_max)^2` -- 4x/16x/100x at 100/50/20 um on a
200 um cladding, measured); an `LPMode` whose `V` disagrees with `2 pi a NA / lambda` for this
fiber, i.e. one solved for a different NA (unchecked it shipped +1.00 dB silently); and two
signal modes with the same `(l, m)` for `l >= 1` -- `LPMode` carries no orientation field and the
profile builder hardcodes `cos(l phi)^2`, so the degenerate cos/sin PAIR cannot be spelled and
passing the same mode twice would model cos + cos (-16.5% modal gain at 200 W, measured). The
quarter-plane grid would integrate `sin^2` exactly (6.7e-16 for `l = 1, 2, 3`), so that last one
is a missing spelling, not a quadrature limitation. Plain two-level Er or Yb, core or cladding
pumped, fwd/bwd, with an ASE band, is supported. It is also strictly STEADY-STATE and incoherent, so like every
steady model it computes the index grating in phase with the irradiance grating and therefore
**cannot** predict a TMI threshold (dossier sec. 2.8); what it does deliver is the saturated
resolved inversion every TMI model needs as input.

### 12a. Achieved (`validation/fiber_radial_inversion.py`, 9 gates; `tests/test_fiber_transverse.py`, 35 gates)
- **Clamp**: `nbar2 -> sa_p/(sa_p+se_p) = 0.5030549898` to 3.6e-13 relative (zero-parameter).
- **Closed form == quadrature**: the solver's own `J` matches (F1.5)/(F1.6) to 1.2e-16; the
  25-case `Gamma x s0` sweep to 1.2e-15; the dossier's kW-class Yb LMA benchmark table
  (`g_res` 6.409/2.524/0.828 m^-1, ratios 0.956/0.825/0.890 at 10/200/1000 W) reproduced to
  3.7e-4 -- i.e. to its printed precision.
- **Reduction**: with a 1 pW signal and no ASE the resolved solve equals the mean-field solve to
  4.5e-11 dB. With an ASE band the residual is -3.0e-5 dB, which is the ASE channels' OWN hole
  burning: it is correctly signed and scales roughly as `L x P_ASE` -- i.e. much faster than
  linearly in `L`, because the ASE power the residual is proportional to is itself growing along
  the fiber. MEASURED on the gate fixture at `L = 0.125 / 0.25 / 0.5 / 1.0 m`:
  `-1.371e-6 / -6.032e-6 / -2.964e-5 / -1.891e-4 dB`, so halving `0.5 -> 0.25 m` divides the
  residual by 4.9, not by 2.
- **Degeneracy**: with the `"flat"` profile the two solvers agree to <= 1e-12 dB at EVERY
  saturation level, across a 66x gain compression -- the resolved model contains sec. 1 exactly.
- **Direction (the headline), for a UNIFORM (cladding) pump**: the resolved-minus-mean-field
  bracket is non-positive over 400 x 12 points of `(Gamma, s0)` (max -7.3e-23);
  `kappa = tanh(x^2)/x^2` matches the numerically extracted slope ratio to 2.5e-7 over
  `b/w = 0.25 .. 3`, `kappa(1) = 0.7615942` -- the `w = a` limit, which the Marcuse branch cannot
  quite reach (`w/a >= 1.0990` for `V <= 2.405`, so with `b = a` the correction available there
  is up to ~18% at the `V = 2.405` edge: `x = 0.9099`, `kappa = 0.8205`); and a cladding-pumped
  solve is never above its mean-field twin, the deficit peaking at -0.108 dB mid-saturation and
  vanishing at both ends (gate E(iii)).
- **Sign exception, a CORE-GUIDED pump** (dossier sec. 1.4 caveat (b), which excludes exactly
  this case from the closed forms): the signed statement above is about the LOCAL gain of one
  saturating signal in a uniform pump field, and an end-to-end gain is not that quantity. A
  core-guided pump bleaches its own absorption hardest where it is brightest, so the resolved
  pump absorption coefficient is SMALLER than mean-field, more pump survives downstream, and the
  resolved end-to-end gain EXCEEDS the mean-field gain. MEASURED on a `V = 5.41` core-pumped Yb
  fiber (`a = 7 um`, `NA = 0.12`, `L = 1 m`, `P_s = 1 mW`): `+0.335 / +0.798 / +0.207 dB` at
  `0.05 / 0.2 / 1.0 W` of pump (gate E(iv)).
- **Multi-mode**: on the Smith & Smith LMA fixture (`V = 8.2193`, `L_beat = 9.8953 mm` vs the
  dossier's 9.895) the LP11-LP01 modal gain difference CHANGES SIGN between 1 W (-0.288 dB/m)
  and 10 W (+0.570), peaks at +2.844 dB/m near 100 W, and decays after -- the beam-quality
  mechanism a mean-field model (constant, negative) cannot produce. NOTE: the dossier quotes
  `Gamma_01 = 0.99034` / `Gamma_11 = 0.97467` for this fixture; this module, `lma.dopant_overlap`
  and scipy adaptive quadrature all give 0.9923946 / 0.9798548, and that overlap ratio is the
  ENTIRE difference between the dossier's `g` values and these (+0.21% / +0.53%). The pins are
  therefore re-measured; the physics claim is the dossier's, unchanged.
- **Convergence**: doubling to 48 nodes/panel moves `dP_k/dz` by 1.3e-16 relative in deep
  saturation (bar: 1e-10). On the AZIMUTHAL axis (gate G2, LP11 + LP21 both at 200 W) doubling
  `n_azimuthal` from the shipped default of 32 to 64 moves both the modal gains and `dP_k/dz` by
  1.5e-13; the former default of 16 was 5.0e-7, three orders above the bar, which is why the
  default was raised.
- **Contract**: every profile integrates to 1 to 2.2e-16; the cladding-pump overlap equals
  `cladding_pump_overlap` exactly; the exact-LP01 overlap matches `lma.dopant_overlap` to 3e-7.
- **Two scalar reductions, and which one the noise helpers read**. `ResolvedResult.nbar2_z` is the
  DOPANT-AREA average (the `SteadyStateResult` contract, so `noise.local_inversion_factor` and
  `NoiseResult.n_sp_local_min` / `n_sp_local_in` consume it); `ResolvedResult.nbar2_mode_z` is the
  MODE-weighted `J/Gamma` of the first signal -- the average the gain coefficient of (F1.2)
  actually integrates. They agree in the small-signal limit and separate once the mode burns its
  hole, so `n_sp_local_*` are ADVISORY under saturation: MEASURED on a 0.5 m 3 um-core Yb fixture
  they sit -0.2% below the mode-weighted value at `P_s = 10 mW`, -9.1% at 1 W (worst -10.3% along
  z) and -13.8% at 5 W. `NoiseResult.nf_dB` is unaffected -- it is built from the ASE power
  spectral density the resolved solve itself produced, not from `nbar2_z`.

Still on the table (documented, not built): SRS/SBS and transverse-mode instability (high-power
limiters), transient heat diffusion, Er:Yb co-doping, polarization-resolved gain, and the delayed
Raman response / self-steepening terms of the full GNLSE (self-frequency shift) -- the current
pulse core carries dispersion + Kerr SPM. (The scalar gain-BPM and the quasi-static thermal lens
are now built -- sec. 13.)

## 13. Scalar gain-BPM -- the signal FIELD in the doped core (gain_bpm.py)

Sections 1 and 12 both propagate POWERS: each channel carries a fixed transverse profile and the
gain it sees is an overlap integral. `gain_bpm.py` (`GainBPM`) propagates the signal FIELD
`E(x,y,z)` instead, so the confinement factor is an OUTPUT rather than an input. Formulation
source: `docs/fiber_transverse_grounding_2026_07_29.md` sec. 2 (equation numbers below are its).
Design principle PINNED from **A. V. Smith & J. J. Smith**, *Opt. Express* 21(13):15168 (2013)
sec. 1: "In our BPM model laser gain increases the total signal field locally and is then
apportioned among the modes automatically by diffraction in the presence of the core index step."
Split-step BPM structure after **Feit & Fleck**, *Appl. Opt.* 17(24):3990 (1978) and **Okamoto**,
*Fundamentals of Optical Waveguides* ch. 7 -- cited for STRUCTURE only: both are in the
`exp(+i omega t)` engineering convention, which flips the sign of every `i`, and neither was
re-verified against the primary text in this build (dossier open item 2). The binding convention
anchor is `soa/transverse_bpm.py`.

**Envelope (F2.2)**, repo `exp(-i omega t)` with the carrier `exp(+i k0 n_ref z)` factored out:
```
dE/dz = (i/(2k)) (d2/dx2 + d2/dy2) E + (i k0^2/(2k)) (n^2 - n_ref^2) E + (g/2) E,   k = k0 n_ref
```
**Field normalization contract**: `INT |E|^2 dA = P_signal` [W], i.e. `[E] = sqrt(W)/m` and
`|E|^2` is an irradiance in W/m^2 -- the quantity the local balance consumes.

**Gain convention (a factor-of-2 trap).** `g` is the INTENSITY gain per metre, so `d|E|^2/dz = g
|E|^2` and the per-step AMPLITUDE factor is `exp(g dz/2)`. The sigma-form
`g = sigma_e n2 - sigma_a (nt - n2)` already IS that intensity coefficient (it is `alpha`/`g*`
divided by `Gamma`), so there is no second 2 hiding in it.

**Symmetrized split (F2.3)-(F2.5)**, `D(dz/2) N(dz) D(dz/2)`:
```
D: E_hat *= exp( -i (kx^2 + ky^2) dz / (2 k0 n_ref) )       (exactly unitary)
N: E     *= exp( i k0^2 (n^2 - n_ref^2) dz / (2 k0 n_ref) ) * exp( g dz/2 )
```
with the saturable `g` MIDPOINT-corrected (predictor half-step at the entry gain, full step at the
gain of the half-advanced field, exactly `soa/transverse_bpm._gain`) -- freezing `g` at the step
entry caps the whole scheme at `O(dz)`. The pump is advanced on the same predictor/corrector.

**Local gain (F2.7)** `g = nt[(sa_s+se_s) nbar2 - sa_s] - l_s` with `nbar2` from the local balance
(F1.1) evaluated on the BPM's own per-pixel intensities. That balance is NOT re-implemented: the
module calls `transverse.ResolvedFiberAmplifier._nbar2_nodes`, the package's single home for it,
with intensity rows against a unit power vector (exact, since `I_k = P_k i_k`). **There is no
`Gamma` in (F2.7).**

**Pump co-ODE (F2.8)**, forward pumps, marched on the same z-step:
`dP_p/dz = -P_p { INT [sa_p (nt - n2) - se_p n2] i_p dA + l_p }`. A cladding pump is flat,
`I_p = P_p/A_clad` over `r <= R_clad`, taken ANALYTICALLY -- so unlike sec. 12's quadrature grid,
a window that does not reach the inner cladding does not inflate the pump overlap.

**Heat (F2.9) [PINNED, Smith & Smith Eq. 2]**
`Q = nt (1 - lambda_p/lambda_s) [sa_p - (sa_p+se_p) nbar2] I_p` [W/m^3], i.e. the quantum defect
times the net absorbed pump power density. Background loss and untracked spontaneous emission are
NOT in it; with photodarkening on it would be a serious under-estimate (Jauregui et al. 2020
sec. 4).

**Thermal lens (F2.10)-(F2.15), opt-in `ThermalLoop`, default OFF.** Quasi-static loop
`E -> n2 -> Q -> dT(r) -> dn = (dn/dT) dT -> n(x,y) -> E`, iterated under-relaxed to a `tol_K` bar
on the peak temperature change. `dT(r)` is `thermal.radial_temperature_rise` (Brown & Hoffman) --
the SAME machinery the power solvers use, not a re-derivation -- and because that solution is
linear in the heat load the z-dependent lens is stored as one shape array times the per-step
`Q(z)`. The in-core rise is exactly parabolic, so the duct parameter is
`alpha = sqrt(D')` with `D' = thermal.thermal_lens_focal_power_per_m`, matched radius
`w_m = sqrt(2/(k0 n0 alpha))` and self-imaging period `z_p = 2 pi/alpha`
(`quadratic_duct_radius_m` / `quadratic_duct_period_m`).

**What the thermal coupling throws away.** Only the z-resolved heat survives the hand-off: (F2.9)'s
radial heat density `Q(x,y,z)` [W/m^3] is integrated to a single scalar `Q(z)` [W/m] before it
reaches `radial_temperature_rise`, whose source is UNIFORM over the heated disc, so the RADIAL
SHAPE of the heat is discarded -- and it is far from flat (on the shipped 1 kW-pump / 20 W-signal
fixture the quantum-defect heat density is **2.73x** the disc mean on axis and **0.10x** at the
dopant edge, a factor of 26 across the disc). The heated radius handed to that solution is
`fiber.core_radius_m` regardless of where the dopant is, so a CONFINED dopant (`b_dope < a`) is
still modelled as heating the full core. Both errors live in the shape of `dT` inside the core,
which is exactly where the duct parameter is read, so the lens strength is a core-averaged
estimate rather than a resolved thermal profile. Neither converges away with grid; fixing them
means a radially resolved source in `thermal.py`.

**Boundary and the power-accounting contract (F2.6).** Super-Gaussian amplitude mask
`M = exp(-[(rho - rho_abs)/w_abs]^m_sg)` outside `rho_abs` (defaults 0.85 / 0.10 of the
half-window, order 4), applied after each full step, with the absorbed power ACCUMULATED and
REPORTED: `P(L) = P(0) + medium_exchange - absorbed_boundary` holds to round-off at every sample
and is returned as `energy_residual_W`.

**Numbered scope limitations of the numerics** (module docstring carries the same list):
1. **The absorber is a per-STEP mask, so it is a `dz`-DEPENDENT operator with no `dz -> 0` limit.**
   Halving `dz` doubles how often it acts. MEASURED by parking light in the absorber annulus and
   refining only `dz`: `P_out/P_in` = **0.612 / 0.521 / 0.439 / 0.372 / 0.317 / 0.271** at
   `nz = 2/4/8/16/32/64` on a fixed physical length -- monotone, no limit. That is the worst case,
   not the shipped one: with the defaults every gate's absorbed fraction is `<= 1.2e-4` and
   `dz`-stable, and the order gate runs with the absorber off. A run whose `absorbed_fraction`
   crosses the dossier's 1e-6 converged-window bar now raises a `UserWarning` saying so; the gates
   and tests that mean to cross it CATCH and assert that warning rather than filtering it away.
2. **`dx` and `dz` are not independent knobs.** The highest transverse frequency the grid carries,
   `kx_max = pi/dx`, accumulates a diffraction phase `pi^2 dz/(2 k dx^2)` per step, so the split
   aliases at the corner of the k-grid unless `dz <= 2 k dx^2/pi` -- a bound that QUARTERS when
   `dx` halves. It is a conservative guard, not a fit to an observed failure: on the gate-F fixture
   at a fixed physical taper width, refining `N` at fixed `n_steps` moves the cross-solver
   deviation not at all (**-0.00932 / -0.00957 / -0.00926 / -0.00939 / -0.00931 dB** at
   `N = 64/96/128/192/256`).
3. **The erf taper is a PHYSICAL width, not a cell count.** `index_taper_cells` is `s` in `s*dx`,
   so holding `s` while refining `dx` REMOVES the anti-aliasing smoothing and the discretized index
   step sheds more (absorbed **1.21e-4 -> 1.37e-4 -> 3.23e-3** at `N = 96/128/192`, `s = 2`). Every
   `dx`-refinement study here scales `s` as `1/dx`.

**Grid / step guidance (dossier sec. 2.4).** `n_ref = n_eff(LP01)` by default; `dx <= a/16` is the
rule and `dx > a/8` is REFUSED (the accident it catches is the auto-sized window: with a cladding
pump the default window covers the inner cladding, which around a small core lands on a coarse
`dx` without the caller doing anything wrong); `dz` from the smallest of a 0.3-rad index-phase
criterion, `L_beat/20`, and the transverse-Nyquist bound `2 k dx^2/pi`
(`transverse_nyquist_step_m`, reported by `sampling_report` as `transverse_nyquist_dz_m`), and an
automatic step count above `max_auto_steps` is refused rather than silently run. `solve(dz_m=...)`
refuses a non-positive step (it used to run a single `L`-long step for `dz_m < 0` and raise a bare
`ZeroDivisionError` at `dz_m = 0`).

**Scope / refusals (each a loud `ValueError`).** More than one `Signal` (one scalar field, one
`n_ref`); backward pumps (a one-way marcher, not a two-point BVP); ASE (spontaneous emission seeds
every guided and radiation mode in both directions with random phase -- it stays in the channel
models, and adding it as extra power on the signal field would put spontaneous photons into a
single coherent mode); `RamanStokes`; `ConcentrationModel` / `upconversion_C_up`; nonzero
`sigma_esa`; Er:Yb; `FiberSpec.overlap_override`; `set_temperature_profile`; and passing
`gain_model` together with pumps. Polarization is out of scope by construction (scalar).

**TMI is out of scope, on purpose (dossier sec. 2.8, PINNED from Jauregui/Limpert/Tunnermann,
*Adv. Opt. Photon.* 12(2):429 (2020) sec. 3).** Energy transfer between transverse modes needs a
phase shift between the modal interference pattern and the thermally induced index grating; a
steady-state model computes the grating exactly IN PHASE with the pattern, and an in-phase grating
transfers zero net energy. The missing ingredient is a time axis. This module can deliver the
saturated resolved inversion and heat profile a TMI model needs as input; it cannot deliver a
threshold. `nonlinear_limits.tmi_threshold_W` remains the (scaling-law) estimator.

**Reading TSHB out of a coherent solver -- why the obvious test is the wrong one.** The tempting
test is "launch LP01 + LP11 and watch the LP11 POWER FRACTION grow". It is not valid here. A
coherent field carries a modal-interference cross term `2 Re(c01 c11*) psi01 psi11` in `|E|^2` that
no power model has, and the saturable medium turns it into a gain grating `delta_g`. `psi11` is odd
in the azimuth and `psi01` even, so at FIRST order in the cross-term amplitude `delta_g` has zero
diagonal projection and lives in the off-diagonal element `G_x = <psi01| delta_g |psi11>`. That
element contributes `G_x sqrt(P01 P11)` to `dP01/dz` AND to `dP11/dz` -- the same ABSOLUTE rate for
both modes, fractionally larger for whichever is weaker -- so its effect on the LP11 fraction goes
as `(P01 - P11)`: exactly zero at a 50/50 split, reversed across it (measured `d(f11)/dz` from the
cross term at 200 W total, LP01 fraction 0.1/0.3/0.5/0.7/0.9: **+0.143 / +0.174 / 0 / -0.184 /
-0.158** per metre, with `|G_x|` between 0.60 and 1.08 1/m). The SECOND-order, even part does shift
the diagonals once the mode powers are comparable, and by a lot: at a 70/30 split the coherent
field's modal gains differ from the channel model's by up to 1.1 1/m, against 2.4e-3 for a
single-mode field. The well-posed statement -- the one gate J pins -- is the LOCAL MODAL-GAIN
ORDERING seen by a weak LP11 probe (below).

### 13a. Achieved (`validation/fiber_gain_bpm.py`, 10 gates; `tests/test_fiber_bpm.py`, 30 gates)
- **Unitarity / ledger**: absorber-free power drift **1.0e-13** over 300 steps; on a window
  deliberately narrow enough that the boundary eats 3.6% of the launch, the ledger closes to
  **3.9e-14**. On a saturated GAIN run (50 W in, 53.2 W out) it closes to **3.8e-13** of the
  launched power. Not a tautology: the exchange term is a closed form in `g` and the field
  entering the `N` leg, so closure tests that `D` is unitary and the index leg a pure real phase.
- **Gain convention**: uniform `g = 2.5 m^-1` over 1.3 m gives `P_out/P_in = exp(g L)` to
  **2.7e-15** (dossier measured 2.4e-14).
- **Split order**: on the SMOOTH quadratic duct, errors 3.65e-3 / 9.14e-4 / 2.27e-4 / 5.59e-5 at
  nz = 24 / 48 / 96 / 192, i.e. observed orders **1.998 / 2.012 / 2.019** (bar 2.00 +/- 0.05).
  Run on the step-index fiber instead, the dossier measured a ragged 0.90/0.92/2.03/4.55 -- the
  hard discontinuity contributes an error that does not scale with `dz` at all.
  **That case does NOT discriminate a frozen-gain leg**, contrary to what its docstring used to
  claim: with `gain_model = 0` the midpoint corrector is a no-op and an exponential-Euler leg
  reproduces those errors BITWISE. A second case now runs a SATURABLE gain on the same duct
  (`g = 300/(1 + I/I_sat)`, launch at peak `I/I_sat = 1.05`, 3.5x power gain over the run): errors
  4.48e-3 / 1.12e-3 / 2.77e-4 / 6.82e-5, orders **1.996 / 2.021 / 2.021** (bar 2.00 +/- 0.15),
  against **1.320 / 1.178 / 1.153** measured with a frozen leg patched in. That is the
  discrimination.
- **Quadratic duct** (`D' = 1e5`, `alpha = 316.2278 m^-1`, `w_m = 26.7658 um`,
  `z_p = 19.8692 mm`): matched Gaussian width span [26.765496, 26.765842] um, relative deviation
  **1.29e-5** (dossier 1.4e-5); self-imaging fidelity at `z_p` **1.0000000000**; a 1.5 `w_m` launch
  reaches its first spot-size minimum at **4.967294 mm** vs `z_p/4 = 4.967294 mm` (exact at the
  step resolution) while breathing [17.844, 40.149] um. This one oracle pins the diffraction
  factor, the index-phase factor and both signs at once.
- **Beat length**: LP01+LP11 on the Smith & Smith LMA fixture, relative modal phase fitted vs z ->
  **9.910092 mm** against the exact-mode `lambda/(n_eff01 - n_eff11) = 9.895325 mm`
  (**+1.5e-3** relative), with 0.700/0.300 W launched and 0.650/0.278 W recovered after two beat
  lengths (no spurious inter-mode transfer; the deficit is hard-step radiation).
  **FINDING that refines the dossier**: the (F2.5a) erf taper is described there as preserving
  `n_eff` "to well under the discretization error", which holds for `n_eff` itself but NOT for the
  SPLITTING `n_eff01 - n_eff11` -- a difference of two nearly equal numbers, so a small absolute
  shift is a large relative one. Measured: the taper shortens `L_beat` in proportion to its
  PHYSICAL width `s*dx` (-4.4% at `s = 2`, `dx = 1.17 um`; -2.2% for 1.56 um of smoothing reached
  either as `s=2`/`dx=0.78` or `s=1`/`dx=1.56`). The taper is the right tool for a single-mode
  phase-fidelity problem and the wrong one for modal dispersion.
- **CROSS-SOLVER PARITY** (fundamental-mode launch, 100 W cladding pump, no ASE, 2 cm of the
  Smith & Smith LMA fiber, `N = 96`, `dx = a/16`, 2-cell taper) versus
  `transverse.ResolvedFiberAmplifier` on the identical plan:
  | | channel model | BPM | delta |
  |---|---|---|---|
  | small-signal (`P_s = 1 mW`) | 0.633584 dB | 0.633603 dB | **+0.00002 dB** |
  | saturated (`P_s = 50 W`) | 0.278994 dB | 0.269428 dB | **-0.00957 dB** |
  Step-converged (dz/2 moves the saturated delta by 7e-5 dB). The deviation decomposes into three
  identified pieces, and the first two were **RE-ATTRIBUTED 2026-08-04** after an adversarial
  re-measurement:
  (1) **SPLIT-STEP ERROR ON THE DISCRETIZED INDEX STEP, reported as radiation.** The gate used to
  assert `abs(d_hard) > 0.1` from a HARD-step row at `nz = 400` and call it "the taper doing real
  work". That number is not converged. Over `nz = 200/400/800/1600/3200` the hard step's absorbed
  fraction falls **1.21e-1 -> 6.46e-2 -> 1.78e-2 -> 8.58e-6 -> 7.53e-6** and its deviation follows,
  **-0.535 -> -0.277 -> -0.076 -> -0.00116 -> -0.00109 dB**; at `nz = 1600` the old assertion FAILS.
  The gate now runs the hard step at a converged `dz` and pins the AGREEMENT: **-0.00009 dB**
  small-signal, **-0.00116 dB** saturated. The taper buys resolution, not correctness.
  (2) **TAPER-WIDTH BIAS, which does NOT converge away.** The 2-cell taper leaves a
  `dz`-INDEPENDENT radiation floor of **1.2e-4** of the launch (a launch mismatch: the launch is
  the exact HARD-step LP01 and the tapered guide's mode is a slightly different one). The old claim
  that the saturated residual "converges away with `dx`" came from the series -0.0175 / -0.0096 /
  -0.0060 / -0.0028 dB at `N = 64/96/128/192` -- but that held the taper at 2 CELLS, so refining
  `dx` was physically removing the taper and the number was tracking the taper WIDTH, not the grid.
  Held at a fixed PHYSICAL width (`s` scaled as `1/dx`) the same series is FLAT: saturated
  **-0.00932 / -0.00957 / -0.00926 / -0.00939 dB**, saturation-specific **-0.00961 / -0.00959 /
  -0.00961 / -0.00960 dB** (spread 2.2e-5 dB). The honest attribution is discretization of the
  PHYSICAL problem -- not radiation, not saturation-integral quadrature, and not physics.
  (3) genuine gain guiding, which at this extraction is the SMALLEST term -- the BPM field's dopant
  overlap moves by only 5.5e-4 over the fiber. A fixture with deeper extraction, confined doping,
  or higher contrast would push (3) above (2); this one does not, and the gate says so rather than
  over-claiming.
- **Thermal**: the module's duct parameter IS `thermal.thermal_lens_focal_power_per_m` (bridge
  relative **2.2e-16**). At `Q = 664 W/m` (`D' = 1.014e6`, `w_m = 14.999 um`, `z_p = 6.2396 mm`)
  the matched Gaussian launched into the REAL self-consistent thermal profile -- parabolic core
  plus logarithmic cladding tail, not an idealized parabola -- holds a width span
  [14.9698, 14.9993] um (**2.0e-3** relative) while the same launch with the lens off diffracts to
  69.62 um (**4.6x**): direction and magnitude in one measurement. The quasi-static loop converges
  in **2 iterations** (final max dT 0.019 K), and `INT Q dz` equals the quantum defect times the
  marched pump absorption to **3.0e-4** (the pointwise `d/dz` form measures 1.4e-3 and does not
  tighten with sampling -- that is the half-step offset between a midpoint-recorded `Q` and a
  boundary-sampled `P_p`, not a physics defect).
- **Background loss** (gate I, new -- every other fixture in the suite runs at
  `background_loss_per_m = 0`, so this leg had no gate at all and a real defect hid there).
  `_medium` returns `g` (F2.7) already NET of `l_s` while `_march` folds `exp(-l_s dz/2)` into the
  WHOLE-GRID index phase, so multiplying the gain support by `exp(g dz/2)` on top applied the
  background loss TWICE inside the doped disc -- intensity factor `exp(g_med dz - 2 l_s dz)`. Off
  the support it was right, which is why nothing caught it. FIXED (one term in `_nonlinear`).
  With the gain off, `l_s = 3 m^-1` over 2 cm now gives `P_out/P_in = exp(-l_s L)` to **5.5e-14**
  and a ledger residual of **4.9e-14**; the defect lands on `exp(-2 l_s L)` instead, **-5.82%**
  off, with a ledger residual of 5.7e-2. With a 100 W cladding pump on the same fiber the ledger
  closes to **8.0e-14** and the loss costs the lossless twin **0.260841 dB** against
  `10 log10(exp(l_s L)) = 0.260577 dB` -- the +2.6e-4 dB is the lossy run saturating the medium
  slightly less, not a defect.
- **COHERENT-SOLVER TSHB** (gate J, new -- the headline physics). The BPM's own
  `local_gain_per_m` field projected onto the LP01/LP11 intensities, with a weak LP11 probe on the
  LMA fixture at a 1.5 kW cladding pump, REVERSES the modal-gain ordering as the LP01 power
  saturates the medium: `g01 - g11` = **+0.0920 -> -0.2431 -> -0.6673 1/m** at
  `P_LP01 = 0 / 100 / 1200 W`, against `transverse.ResolvedFiberAmplifier`'s resolved modal gains
  (**+0.0934 / -0.2417 / -0.6659**) to **1.4e-3 1/m** at every point -- two solvers, no shared code
  path, one number. Positive unsaturated because LP01 is better confined to the dopant; negative
  once LP01 has burnt its own hole on axis. The projection is diagonal-only, so the interference
  cross term of sec. 13's TSHB note never enters it, which is exactly why the probe form is the
  well-posed one.
- **Runtime**: the gate script's own body measures **16.1-17.6 s** on a quiet box and
  **61.7-86.4 s** with one competing full-CPU workload (64-192 grids, cm-scale fibers; the work is
  CPU-bound FFT marching, which unlike an import does not amortize against load); 18-23 s / 108-145
  s in `run_all` for the same two regimes, on top of the 14-40 s
  `import dynameta.optics.fiber_amp`. `tests/test_fiber_bpm.py` measures ~4-7 s of test time. The
  loaded figure is 2x the smoke tier's ~60 s per-script bar, so the script sits in
  `run_all.SMOKE_EXCLUDED` with its measured spread rather than being gambled into a tier CI runs
  on shared hardware.
