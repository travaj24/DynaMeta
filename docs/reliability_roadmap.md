# DynaMeta Reliability / Degradation Roadmap

A ranked roadmap for a NEW reliability axis. DynaMeta today models only OPERATING behavior
(carriers -> Drude eps -> optics); it has NO degradation/wear-out physics. This roadmap adds a
reliability layer that turns the device's operating fields into quantitative lifetime / MTTF / FIT
projections for qualification and design trade-offs.

Companion to docs/physics_depth_roadmap.md (which uses R1-R34 ids for operating-physics depth). To
avoid id collision, reliability items are numbered REL1..RELn.

Provenance: the mechanism set was produced by a web-grounded multi-agent sweep (one researcher +
one adversarial verifier per mechanism), THEN hand-checked. The hand check overturned two propagated
errors: (a) the TDDB field-acceleration coefficient had been "corrected" into a form that drives the
activation energy negative at realistic fields -- replaced here with the standard separable E-model;
(b) the Coffin-Manson exponent had been mis-stated (the fatigue ductility exponent c was confused with
the strain exponent 1/c) -- corrected here, with a separate brittle-fracture model for ITO/oxide.

## How to read this

- Effort: S = a few days, M = ~1-2 weeks, L = research-grade / needs new material data or a new driver.
- Value: relevance to the ITO-ENZ modulator north star (High / Med / Low); items marked *general*
  strengthen library breadth more than this device.
- Each item names a validation ORACLE: it MUST (a) reduce to a known closed-form limit AND (b)
  cross-check an INDEPENDENT reference (per the house discipline -- not energy-only, not tautological).
- Builds-on: the EXACT existing DynaMeta driver the model consumes, so each item is a POST-PROCESSOR on
  the closed operating solve (additive, byte-identical when its knob is off), not a from-scratch solve.
- [NEW DRIVER] = needs a quantity the solve does not yet produce (flagged explicitly, never assumed).
- SI units; exp(-i omega t), Im(eps) > 0 for absorbers; cp1252/ASCII-only source + prints (spell out
  omega, sigma, Ea, Delta, degC, ^2).

---

## Executive summary

The gated ITO-ENZ modulator (gate metal | ~5-20 nm gate oxide | ~1-10 nm ITO accumulation layer |
oxide spacer | metal mirror; free-space optics at ~1300-1550 nm) is a multi-stressor reliability
problem, and every stressor is ALREADY a field the operating solve produces:

- gate oxide at ~1-10 MV/cm (the ~1 nm accumulation layer concentrates the field) -> TDDB + BTI;
- drive current in the metal mirror / contacts -> electromigration;
- ENZ absorption + Joule + carrier heating (R9) deposit power -> self-heating / optical damage;
- the ITO carrier density n itself drifts (oxygen-vacancy de-doping) -> the ENZ wavelength drifts
  (the device-specific PARAMETRIC failure -- the modulator silently de-tunes from its design band);
- CTE-mismatched thin films (ITO / oxide / metal / Si) cycle thermally -> fatigue / cracking.

DESIGN PRINCIPLE: every item is a post-processor that consumes the existing ElectroThermalResult
(E-field, per-layer T, Joule Q, and J = sigma*E), the transient heat trace (dT(t)), the Drude n->eps
map + its ENZ crossing, and the optics absorbed power. So the whole axis lives in a new pure-numpy
dynameta/reliability/ module, is additive, and is byte-identical to today when no degradation knob is
enabled. A driver several models need but the solve does NOT yet expose (contact current, mechanical
stress / CTE, substrate current, optical |E|^2 map, humidity) is flagged [NEW DRIVER] -- those are the
real prerequisites.

Recommended first three: REL1 (gate-oxide TDDB) -- the primary categorical risk; REL4 (ITO de-doping
-> ENZ drift) -- the device-specific parametric risk; REL10 (acceleration-factor + system-MTTF
aggregation) -- the umbrella that makes the others comparable and extrapolates stress -> use conditions.

---

## Tier 0 -- primary lifetime limiters (highest leverage; pure post-processors)

### REL1. Gate-oxide TDDB (time-dependent dielectric breakdown)  -- M, High
- Physics: median time-to-breakdown of the gate oxide under the accumulation-layer field. Use the
  standard SEPARABLE E-model (industry default; avoids the negative-activation pathology of folding
  field into the activation energy):
    tBD(E_ox, T) = tau0 * exp(-gamma_E * E_ox) * exp(Ea / (kB * T))
  with E_ox the oxide field (MV/cm), gamma_E ~ 1-4 cm/MV (field acceleration), Ea ~ 0.6-0.9 eV
  (thermal acceleration), kB = 8.617e-5 eV/K. Offer the 1/E (anode-hole-injection) alternative
  tBD = tau0 * exp(G / E_ox), G ~ 300-400 MV/cm, for the high-field regime. Add Weibull AREA scaling
  for a device array: t63(A2) = t63(A1) * (A1/A2)^(1/beta), beta ~ 1-2 (thin-oxide shape factor) ->
  the array's earliest-failure time, not the single-cell median.
  CAVEAT (do NOT use the thermochemical Ea(E)=Ea0 - b*E form unless b*E < Ea0 is enforced; otherwise
  the activation energy goes negative at >~5 MV/cm and tBD is unphysical -- this is the trap the
  adversarial pass + hand-check caught).
- Why: the oxide field is the PRIMARY categorical (catastrophic) lifetime limiter -- a few volts over
  5-20 nm gives 1-10 MV/cm, squarely in the TDDB window; Joule + leakage self-heating shortens tBD
  via the Arrhenius factor. Without it, design optimization is blind to oxide integrity.
- Builds on: ElectrostaticResult.E_cf / mean_Ez_per_layer (oxide field), ElectroThermalResult.T_per_layer
  (Joule-raised T), geometry/stack (oxide layer id + thickness). Pure scipy.
- Oracle: (a) reduces-to-limit: at constant E,T the trap-percolation ODE integrates to a closed
  exponential and tBD matches the analytic E-model value; (b) independent ref: published SiO2 tBD(E,T)
  (e.g. ~100-1000 s for 7 nm at 7 MV/cm, 300 K) and the JEDEC JESD92 extrapolation method; also verify
  the Weibull area-scaling slope 1/beta against a known dataset.
- Effort M / Value High. [NEW DRIVER] none for E,T; needs a per-material {Ea, gamma_E, beta, tau0}
  table (literature defaults). Depends on: ElectroThermalResult, geometry/stack oxide tagging.

### REL2. NBTI / PBTI bias-temperature instability  -- M, High
- Physics: flat-band / threshold-voltage drift of the gated MOS-cap under sustained gate bias + T
  (a PARAMETRIC degradation -- the operating point creeps). Power-law form:
    dVth(t) = A * (E_ox)^gamma * t^n * exp(-Ea / (kB * T))
  with n ~ 1/6 (H2-diffusion-limited reaction-diffusion) to ~0.2, Ea ~ 0.1-0.15 eV, gamma ~ 0.3-0.4.
  Time-to-spec: t_fail = (dVth_max / (A E_ox^gamma exp(-Ea/kT)))^(1/n). Include the AC duty-cycle
  weighting (~D^0.5) and partial recovery during the off phase (split stress/relax bookkeeping).
- Why: a creeping Vth shifts the bias needed to hit the ENZ accumulation point, slowly de-tuning the
  modulator; it does not short the device (unlike TDDB) but erodes the operating margin.
- Builds on: ElectrostaticResult.mean_Ez_per_layer (E_ox), ElectroThermalResult.T_per_layer, applied_V.
  Shares the gate-oxide driver with REL1.
- Oracle: (a) reduces-to-limit: the H2-diffusion-limited reaction-diffusion PDE reduces to the
  sqrt(t) (n=1/6) closed form; (b) independent ref: a JEDEC JESD90-style multi-(V,T)-corner Arrhenius/
  power-law fit on published NBTI data -- NOT a self-inversion of the same power law (that would be
  tautological).
- Effort M (L if the recovery/AC reaction-diffusion PDE is solved) / Value High. [NEW DRIVER] a
  cumulative stress-time accumulator threaded through the bias sweep (pipeline loop-state).

### REL3. Electromigration (Black equation)  -- M, Med
- Physics: void-nucleation/growth failure of current-carrying metal (mirror, contacts, traces):
    MTTF = A * J^(-n) * exp(Ea / (kB * T)),   n = 2 (void-growth, the usual dominant regime),
  Ea ~ 0.55 eV (Cu) to ~0.9 eV (Al); Blech immortality: J < J_Blech = (J*L)_crit / L_wire -> MTTF
  effectively infinite (short contacts are often immortal). NOTE the pre-factor A is geometry-scaled:
  A ~ 1e5-1e6 for um-scale contacts/vias, NOT the ~1e10 long-interconnect value -- using the wrong A
  over-predicts MTTF by ~1e4-1e5x.
- Why: the metal mirror + gate contacts carry the modulation current; for narrow traces / high J this
  competes with TDDB as the lifetime limiter; Blech immortality is a design knob (short, wide contacts).
- Builds on: ElectroThermalResult.T_per_layer + joule_per_layer (J = sqrt(Q*sigma) or Q/E or sigma*E),
  geometry/stack (metal layer + contact area), material role (metal). 
- Oracle: (a) reduces-to-limit: the Arrhenius ratio MTTF(T2)/MTTF(T1) = exp((Ea/kB)(1/T2-1/T1)) and the
  power-law ratio MTTF(J2)/MTTF(J1) = (J1/J2)^n match the closed forms; Blech J<J_Blech -> immortal;
  (b) independent ref: JEDEC JC-15.1 / Black 1969 calibration point (Cu, 300 K, 1e5 A/cm^2 -> MTTF >
  1e5 h). (The originally-proposed void-radius oracle was DROPPED: it failed the Korhonen cross-check
  by ~1000x because it ignored the nucleation pre-factor.)
- Effort M / Value Med. [NEW DRIVER] contact current I_contact [A] -- DEVSIM exports J_n [A/m^2] per
  edge; needs a region-integrated get_contact_current() extractor, OR accept I_contact as an external
  design parameter (simpler MVP). Depends on: ElectroThermalResult, geometry contact area, metal {Ea,
  J_Blech, A} table.

### REL4. ITO thermal de-doping -> ENZ wavelength drift  -- M, High (device-specific)
- Physics: oxygen-vacancy diffusion / re-oxidation slowly reduces the ITO carrier density:
    dn/dt = -lambda(T) (n - n_min),  lambda(T) = lambda0 * exp(-Ea / (kB * T)),  Ea ~ 1.5-2.1 eV.
  Because the Drude plasma frequency wp^2 = n e^2/(eps0 m) and the ENZ crossing sits where Re(eps)~0,
  lambda_ENZ ~ proportional to sqrt(m/n), so the sensitivity is
    d(lambda_ENZ)/d n = -(1/2) * (lambda_ENZ / n) * (1 - dln m/dln n)
  (a factor of 1/2 -- NOT lambda_ENZ/n). The simple -(1/2) lambda/n form assumes CONSTANT m_opt and
  eps_inf; with the library's Kane n-dependent mass (KaneOpticalMass, dln m/dln n > 0) the drift
  sensitivity is REDUCED by the bracketed factor -- use the full expression when the Kane closure is on.
  A few percent carrier loss red-shifts the ENZ point by nm-scale, comparable to the modulation
  bandwidth.
- Why: THE device-specific parametric failure -- the modulator silently de-tunes from its design
  wavelength. Distinct from catastrophic oxide breakdown; in-field-monitorable and bias-compensable.
- Builds on: ThermalResult.mean_T_per_layer (drives lambda(T)), the existing DrudeOptical n->eps map +
  the ENZ-crossing tracker. Byte-identical-off: lambda0 = 0 -> n(t) = n0 exactly.
- Oracle: (a) reduces-to-limit: constant T -> n(t) = n_min + (n0-n_min) exp(-t/tau), tau = 1/lambda(T);
  (b) independent ref: published ITO thermal-aging Arrhenius (resistivity/carrier-loss vs 1/T) -> fit
  Ea and compare (NOT the model's own exp(-t/tau) against itself).
- Effort M / Value High. Depends on: ThermalResult, DrudeOptical, a per-ITO {Ea, lambda0} from
  literature (document as empirical, stoichiometry-dependent).

---

## Tier 1 -- device-specific coupling (need a new driver or transient)

### REL5. Optical / laser-induced damage + thermal runaway (LIDT)  -- M, High
- Physics: the absorbed optical power density is P_abs = (1/2) omega eps0 Im(eps(n,T,lambda)) |E|^2
  [W/m^3] (peaked at ENZ and amplified in the carrier-heating R9 regime); it self-heats the layer
  (dT/dt = (P_abs - heat_loss)/(rho Cp)), and Im(eps) rises with T -> a feedback. Laser-induced damage
  threshold in the thermal-diffusion regime scales as fluence F_th ~ sqrt(tau_pulse) (Stuart 1996).
- Why: a free-space-optics modulator -- absorbed power at the ENZ peak sets the maximum usable optical
  intensity; runaway is a hard ceiling the steady thermal balance cannot see.
- Builds on: the optics solve's |E|^2 / absorbed-A, DrudeOptical Im(eps(T)), the R5 transient heat +
  R9 carrier-heating two-temperature model.
- Oracle: (a) reduces-to-limit: I = 0 (or Im(eps) = 0) -> T = T_ambient byte-identical; CW steady ->
  the existing thermal_fem steady T; (b) independent ref: published ITO LIDT (~0.5-5 J/cm^2 regime) +
  the sqrt(tau_pulse) fluence law.
- Effort M / Value High. [NEW DRIVER] a per-region absorbed-power-density map -- OpticalResult carries
  R/T/A flux, not the local Im(eps)|E|^2; either add P_abs_per_region to the optics result or
  re-evaluate it from the FEM/FDTD field. Depends on: optics |E|^2, thermal props (rho,Cp,k), R5/R9.

### REL6. Thermal-cycling fatigue (CTE mismatch)  -- M, Med
- Physics: cyclic dT (ambient swing or pulsed self-heating) drives biaxial strain via CTE mismatch:
    sigma = (E_film / (1 - nu_film)) * (CTE_sub - CTE_film) * dT.
  DUCTILE metal (Cu/Al traces): Coffin-Manson  Nf = C * (delta_eps_p)^(-1/c), where c is the fatigue
  DUCTILITY exponent ~ 0.5-0.7, so the exponent ON the plastic strain range is 1/c ~ 1.4-2.0 (this is
  the corrected form -- do NOT write Nf ~ delta_eps^(-c) with c~0.6, which inverts the dependence).
  Norris-Landzberg extends it for cycle frequency + Tmax. BRITTLE films (ITO, gate oxide) do NOT follow
  Coffin-Manson -- use a Weibull fracture / critical-stress model (fail when sigma >= sigma_crit, or a
  Paris-law crack-growth + weakest-link ensemble); the brittle oxide typically cracks before the metal.
- Why: ITO (CTE ~ 4-8 ppm/K), SiO2 (~0.5), metal (~16-23), Si substrate -- a 15-20 ppm/K mismatch gives
  100s of MPa residual stress; pulsed/GHz self-heating and ambient cycling accumulate damage.
- Builds on: ThermalTransientResult.t_s + T(t) (extract dT amplitude + cycle frequency), geometry/stack.
- Oracle: (a) reduces-to-limit: dT -> 0 => Nf -> infinity; constant-dT closes to the analytic Nf;
  (b) independent ref: JEDEC JESD22-A104 thermal-cycle TTF curves (ductile) and a brittle-fracture
  Weibull table (ITO/oxide).
- Effort M / Value Med. [NEW DRIVER] mechanical material properties {CTE, E, nu, sigma_crit} -- absent
  from Material/ThermalLayer; a material-schema extension is the prerequisite. Depends on:
  ThermalTransientResult (R5), the new mechanical-property schema.

### REL7. Stress / thermal-gradient migration  -- M, Med
- Physics: atomic flux from mechanical-stress and temperature GRADIENTS (complements electromigration;
  same diffusion species). Korhonen back-stress evolution  d(sigma)/dt = d/dx[ kappa d(sigma)/dx ],
  kappa = D_a B Omega / (kB T),  D_a = D0 exp(-Q / (kB T)); a void nucleates where sigma reaches
  sigma_crit. Thermal-gradient (Soret) flux J_atom ~ (D/kB T)(Q* / T) grad T.
- Why: sharp grad T at the ENZ-absorbing ITO / metal-mirror interface drives migration even below the
  EM current threshold; sets a complementary lower bound on metal-feature lifetime.
- Builds on: thermal_fem grad T, the same mechanical-stress field REL6 introduces.
- Oracle: (a) reduces-to-limit: Q -> 0 (constant diffusivity) -> the linear stress-diffusion PDE has
  the closed-form erfc solution (match to machine precision); (b) independent ref: Korhonen 1993
  back-stress saturation + the EM Blech immortality length as a consistency cross-check.
- Effort M / Value Med. [NEW DRIVER] the mechanical-stress field (shares REL6's schema). Depends on:
  thermal_fem grad T, mechanical schema.

---

## Tier 2 -- general / lower device-relevance (capture for completeness)

### REL8. Hot-carrier injection (HCI)  -- S, Low
- Physics: high-field carriers gain enough energy to generate / inject interface traps. Lucky-electron
  time-to-degradation  t_HCI = C * (I_sub / W)^(-m) * exp(Ea / (kB T)),  m ~ 2/3 (Takeda); the
  interface-trap rate carries the elementary charge: dN_it/dt = (A/q)(I_sub/(W L)) exp(...). HCI often
  WORSENS at LOW T (Ea ~ -0.1 to -0.2 eV) -- opposite to the other Arrhenius mechanisms.
- Why: HONESTLY MARGINAL for a vertical MOS-cap (no lateral channel field); relevant only if in-plane
  fields are large (patterned-gate fringing). Included for completeness.
- Builds on: the DD solve. [NEW DRIVER] substrate current I_sub from a DEVSIM impact-ionization model.
- Oracle: (a) reduces-to-limit: I_sub = 0 -> no degradation (N_it constant); (b) independent ref:
  published Si/SiO2 MOS HCI I-V at matched bias/T -- NOT DEVSIM-Chynoweth against itself (the proposed
  Chynoweth cross-check was tautological: DEVSIM impact ionization IS the Chynoweth model).
- Effort S / Value Low. Depends on: DD + impact-ionization (new), interface-trap model.

### REL9. Environmental corrosion / oxidation / electrochemical migration (CAF)  -- M, Low (general)
- Physics: metal oxidation/corrosion (Deal-Grove oxide growth x^2 + A x = B (t + tau); B parabolic and
  B/A linear rate constants are Arrhenius), and humidity-driven conductive-anodic-filament / electro-
  chemical migration between biased metal features (Peck: t_fail ~ RH^(-n) exp(Ea/kT) f(V), n ~ 2-3).
- Why: GENERAL / packaging-level; low priority for a hermetically-sealed photonic chip, but matters
  for unencapsulated test structures or humid environments.
- Builds on: ElectrostaticResult E-field + metal geometry. [NEW DRIVER] ambient humidity RH and ionic-
  contamination level -- external inputs (not computable from the operating solve).
- Oracle: (a) reduces-to-limit: thin-oxide Deal-Grove linear limit x ~ (B/A) t; CAF zero-field -> no
  filament; (b) independent ref: Massoud Si/SiO2 oxidation data + a Peck humidity-life dataset.
- Effort M / Value Low. Depends on: external RH/contamination inputs, metal geometry.

---

## Tier 3 -- the umbrella: acceleration factors + system MTTF

### REL10. Acceleration factors + system-MTTF aggregation (competing risks)  -- L, High
- Physics: ties every mechanism together. Temperature acceleration
    AF_T = exp( (Ea / kB) * (1/T_use - 1/T_stress) )
  (NOTE the parenthesization: (Ea/kB)*(1/Tu - 1/Ts), NOT Ea/(kB*(1/Tu-1/Ts)) -- the latter overflows;
  this was a flagged-and-fixed error). Voltage/field acceleration AF_V = exp(gamma_V (V_stress -
  V_use)) (or a power law). System reliability as COMPETING RISKS: R_sys(t) = product_i R_i(t), i.e.
  1/MTTF_sys = sum_i 1/MTTF_i for exponential life (weakest-link); for an N-element array use the
  Weibull earliest-failure statistic. Extrapolate accelerated-stress measurements to use conditions
  with the per-mechanism AF and report a device MTTF + FIT.
- Why: makes the per-mechanism MTTFs comparable, converts an accelerated test to a use-condition
  lifetime, and yields the single number qualification needs.
- Builds on: the MTTF outputs of REL1-REL9.
- Oracle: (a) reduces-to-limit: with a single active mechanism, AF reduces to the pure Arrhenius (or
  pure voltage) closed form, and MTTF_sys -> that mechanism's MTTF; (b) independent ref: a JEDEC
  JEP122 / JESD85 worked example (e.g. a published TDDB or EM acceleration table) -- NOT synthetic
  data generated by the same formulas (the originally-proposed self-consistency oracles were
  tautological; the cross-check must use an external published baseline).
- Effort L / Value High. Depends on: REL1-REL9 (consumes their MTTFs).

---

## Cross-cutting design notes

- NEW driver prerequisites, consolidated (these are the real gating work, shared across items):
  (1) contact current I_contact (REL3) -- DEVSIM region-integrated current OR external param;
  (2) a mechanical-property schema {CTE, E, nu, sigma_crit} on Material (REL6, REL7);
  (3) substrate current I_sub from impact ionization (REL8);
  (4) a per-region absorbed-power map P_abs = (1/2) omega eps0 Im(eps) |E|^2 on the optics result (REL5);
  (5) external environment inputs RH / contamination (REL9).
- Module shape: a new pure-numpy dynameta/reliability/ that takes an ElectroThermalResult /
  ThermalTransientResult / OpticalResult + a material reliability-parameter table and returns a
  ReliabilityResult (per-mechanism MTTF, dominant mode, system MTTF/FIT). Mirrors the
  carriers/electrothermal.py post-processor pattern; byte-identical-off (no knob -> no call).
- Validation discipline (mandatory, same as the physics axis): each item ships a reduces-to-known-
  closed-form gate AND an INDEPENDENT published-reference gate; parameters cite their literature source;
  the off-switch (knob disabled -> baseline unchanged) is proven.

## Audit blind spots (completeness critic + hand review)

- Statistical extreme-value / array reliability: a metasurface is an N-element array; the FIRST failing
  element sets system life -> a Weibull/extreme-value layer on top of the per-cell median (partly in
  REL1 area-scaling + REL10, but a full defect-ensemble model is deferred).
- Mechanism COUPLING (not just competing risks): TDDB accelerated by BTI trap build-up; EM accelerated
  by stress-migration back-stress; de-doping feedback into self-heating. The competing-risks product
  assumes independence -- coupled-degradation is a research-grade extension.
- AC / history dependence: trap filling and BTI recovery depend on the bias/stress HISTORY; the
  Arrhenius MTTFs are stateless. A duty-cycle / history-aware state model is deferred.
- Parameter extraction: the per-mechanism Ea / gamma / exponents are material- and process-specific;
  the roadmap ships literature defaults, but a calibration-from-accelerated-test workflow (fit Ea from
  multi-corner data) is itself a deliverable.
- Photon-assisted / ENZ-enhanced barrier lowering: the ENZ field could lower tunneling/reaction
  barriers (Fano/optical-field coupling) -- a novel coupling not in any standard model; research-grade.
