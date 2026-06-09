# DynaMeta Physics-Depth Roadmap

A ranked roadmap of *deeper physics* worth adding to DynaMeta, in the spirit of the recent
liquid-crystal deepening (1-constant static planar director -> full Frank-Oseen 3-constant elasticity +
Erickson-Leslie dynamics with backflow + weak anchoring + chiral/twist + 2-D director + temperature
dependence). It was produced by an 8-domain survey of the *current* code (so every item below is a real
gap, not something already modeled) and ranked by **(physical value for the ITO-ENZ metasurface
modulator north star) x (feasibility within the existing DEVSIM / Drude / FEM / RCWA / FDTD
architecture)**.

## How to read this

- **Effort**: `S` = a few days, `M` = ~1-2 weeks, `L` = research-grade.
- **Value**: relevance to the ITO-ENZ modulator north star (High / Med / Low); items marked *general*
  strengthen library breadth more than the north star (called out honestly).
- **LC-analog flavor** (how it parallels the LC work): `(a)` dynamics / time-evolution of a static
  quantity; `(b)` higher-order / more-complete constitutive law; `(c)` boundary / interface physics;
  `(d)` spatial generalization (1-D -> 2-D/3-D, gradients); `(e)` coupling to another field;
  `(f)` material-property (T / density / frequency) dependence.
- Each item names a **validation oracle** (the library is validation-driven: every feature ships an
  independent reduces-to-known-limit + cross-reference gate).
- `[#n]` cross-references the survey opportunity id for traceability.

Convention reminder for implementers: cp1252-safe ASCII-only source + print(); SI units; exp(-i w t),
Im(eps) > 0 for absorbers; persistent validations under `validation/`; commit incrementally; push only
when asked.

> Companion: this roadmap is OPERATING-physics depth (R1-R34). Reliability / degradation / wear-out
> (electromigration, gate-oxide TDDB, NBTI/PBTI, thermal-cycling fatigue, ITO de-doping / ENZ drift,
> optical damage, system MTTF) is a SEPARATE axis with its own ranked roadmap in
> [docs/reliability_roadmap.md](reliability_roadmap.md) (items REL1-REL10).

## STATUS (2026-06-09): Tier 0 (R1-R9) COMPLETE -- the north-star set is shipped

All nine Tier-0 items are implemented, validated against an independent oracle, unit-tested, committed
and pushed. Each is byte-identical when its new physics is off.

- **R1** field/density mobility mu(E,n) -- `carriers/mobility.py` (Caughey-Thomas + Masetti DEVSIM edge);
  validation `dd_field_mobility.py`. Commit 5921f85.
- **R2** resolved Drude Gamma + Kane m_opt(n) -- `materials/scattering.py` (KaneOpticalMass,
  MatthiessenGamma); `drude_matthiessen_kane.py`. Commit 16ad0f0.
- **R3** shared mobility<->optical-Gamma link -- `ScatteringModel` (one tau drives both); `scattering_link.py`. Commit d22d8ce.
- **R4** per-cell time-domain eps(t) -> FDTD -- `optics/fdtd_seam.effect_eps_to_fdtd_grid` +
  solve_fdtd_2d lateral_wp/gam; `fdtd_effect_seam.py`. Commit a2bce7b.
- **R5** transient heat eqn rho Cp dT/dt -- `thermal_fem.solve_thermal_transient_fem` (theta-method);
  `thermal_transient_fem.py` (Carslaw-Jaeger erfc + steady recovery). Commit f3f7477.
- **R6** self-consistent electro-thermo-optic Picard loop -- `carriers/electrothermal.py`;
  `electrothermal_picard.py`. Commit 1be7ff1.
- **R7** quantum intersubband eps_zz -- `core/effects.IntersubbandEffect`; `intersubband_eps_zz.py`
  (TRK f-sum rule + telecom line). Commit 48b84b4.
- **R8** Burstein-Moss + bandgap-renormalization edge -- `core/effects.BursteinMossEdge`;
  `burstein_moss_blueshift.py`. Commit 4946026.
- **R9** carrier-heating two-temperature ENZ nonlinearity -- `carriers/carrier_heating.py`;
  `carrier_heating_enz.py` (sub-ps rise / ps relaxation, 15.4x ENZ enhancement). Commit 044f0dc.

Test suite: 307 passed (295 at the R1-R9 ship; +12 from the 2026-06-09 audit-guard batch, commit
7a04a37). Tier 1+ below is not yet scheduled.

---

## Tier 0 -- NORTH STAR: ITO-ENZ modulator core fidelity

These directly improve the device the library exists for. Ordered by leverage (value / effort). The
first four are all `S`-effort and individually small -- ship them first.

### R1. Field- and density-dependent mobility mu(E,n)  [#1]  -- S, High
- **Physics**: promote the frozen scalar `mu_n` (evaluated once at `n_bg`) to a real DEVSIM **edge
  model**: Caughey-Thomas/Canali velocity saturation `mu(E) = mu_low/(1+(mu_low|E_par|/v_sat)^b)^(1/b)`
  + Masetti ionized-impurity `mu_low(N)`. `E_par` is already the `ElectricField` edge model; derivatives
  auto-generate via the existing `edge_with_derivs` seam.
- **Why (north star)**: the ~1 nm accumulation sheet sees MV/cm fields and n > 1e27 m^-3; a constant
  low-field mobility over-predicts the charging current and **mis-predicts the RC turn-on/off time and
  the ssac f_3dB the library is built to report**. Single biggest fidelity gap in the DD/transient/AC
  chain -- it changes the *dynamics*, not just a static profile.
- **LC-analog**: (b)+(f). **Feasibility**: pure DEVSIM edge models, no new solver; the `MobilityFn`
  callable on `TransportModel` already exists and is unused. **Oracle**: analytic velocity-saturation
  J-V; C-V stretch-out.

### R2. Resolved Drude damping Gamma(omega,T,n) via Matthiessen decomposition  [#6, #13]  -- S, High
- **Physics**: replace the hand-fit constant `gamma` (1.1e14) with `1/tau = 1/tau_phonon(T) +
  1/tau_ii(n) + 1/tau_gb` (ionized-impurity Brooks-Herring dominant in degenerate TCOs, LO-phonon Bose
  term, grain-boundary residual), plus an optical-vs-DC Gamma distinction. Also supply the n-dependent
  optical mass `m_opt(n)` from the *same* Kane nonparabolicity the SP fill already uses, so
  `wp^2 = n e^2/(eps0 m_opt(n))` is correctly sub-linear in n.
- **Why (north star)**: Gamma sets the *entire* loss (Im eps) at ENZ -- the achievable contrast and the
  absorption penalty -- and `wp(n)` sets the ENZ wavelength the device is tuned to. Today both are
  constants, so the ENZ point is mislocated as the gate sweeps n across the accumulation layer (the very
  sweep the device exploits).
- **LC-analog**: (b)+(f). **Feasibility**: `DrudeOptical` already accepts `gamma_rad_s` and `m_opt_kg`
  as callables of n -- this is supplying the physically-correct closures; pure numpy, no new solver.
  **Oracle**: Park 2021 Fig. S2 / measured ITO n,k vs density; `fit_drude_params` harness exists.

### R3. Self-consistent mobility <-> optical-Gamma scattering link (one tau drives both)  [#4]  -- M, High
- **Physics**: the DC momentum-relaxation time `tau_m = m*_cond mu/q` (transport) and the optical Drude
  `gamma = 1/tau` (dispersion) describe the **same** scattering yet are independent free parameters
  today. A shared `ScatteringModel` makes `gamma(n) = q/(m*_opt(n) mu(n))` (with the standard
  high-frequency-mass and Hall-factor caveats), so one density/temperature law feeds BOTH the carrier
  solve and the n->eps map.
- **Why (north star)**: in ITO the ENZ loss and the DC accumulation transient are governed by the same
  scattering; today you can fit a `gamma` that contradicts the mobility you assumed for the RC time.
  Linking them removes a hidden inconsistency and lets a single parameter set jointly predict modulation
  **depth and speed**. Builds on R1+R2.
- **LC-analog**: (e)+(f). **Feasibility**: lives at the `TransportModel`/`OpticalModel`/`n_to_eps` seam;
  no DEVSIM solver change. **Oracle**: the same tau reproduces Park Im(eps) AND a sane mobility.

### R4. Per-cell time-domain eps(t) hook into FDTD  [#36]  -- S, High
- **Physics**: not new physics -- a wiring gap. `effects.py` computes rich field/T/state-dependent eps
  (Pockels, Franz-Keldysh, QCSE, PCM Bruggeman, thermo-optic, MO) but the FDTD only ingests
  `(eps_inf, wp, gamma, chi3, Lorentz)`. Add an eps-assembler callback so a slow drive (gate E, T, PCM
  fraction f, magnetization) sets the per-cell linear eps the FDTD then propagates broadband.
- **Why (north star)**: the time-domain engine -- where dynamics + broadband + nonlinearity live -- is
  currently blind to the constitutive richness already in `effects.py`. Highest leverage-per-effort: a
  gate- or PCM-tuned metasurface's full spectral reconfiguration in one broadband solve.
- **LC-analog**: (e). **Feasibility**: mirrors the existing `eps_assembler.py`/`fdtd_seam` closures +
  `bridge.py` extra_fields bundle. Scalar/diagonal first; off-diagonal needs the MO/tensor kernel.
  **Oracle**: a static-bias eps reproduces the frequency-domain FEM/TMM result the effect already gives.

### R5. Transient heat equation (rho Cp dT/dt mass term)  [#27]  -- S, High
- **Physics**: `rho Cp dT/dt = div(k grad T) + Q(t)`; add a mass bilinear form alongside the existing
  stiffness in `solve_thermal_fem` + a theta-method/backward-Euler loop. Diffusivity `D = k/(rho Cp)`
  sets `tau ~ L^2/D`.
- **Why (north star)**: thermo-optic / electro-thermal modulators are slow; their headline spec is the
  thermal rise/fall time the steady solver cannot produce. Self-heating from gate leakage/Joule under an
  AC drive sets a thermal floor on modulation rate + a DC bias drift; feeds the existing
  `transient_optics.py` R(t).
- **LC-analog**: (a) -- the direct dynamics move. **Feasibility**: `solve_thermal_fem` already assembles
  K and the Joule load; add an NGSolve mass matrix + theta loop. Templates: `transient.py` BDF1,
  `lc_dynamics` solve_ivp. New inputs: `rho`, `Cp`. **Oracle**: 1-D transient-slab erfc analytic.

### R6. Self-consistent electro-thermo-optic loop (close E <-> n <-> T <-> eps)  [#31]  -- M, High
- **Physics**: Picard/Newton fixed point over the three *existing* solvers: electrostatics/DD ->
  Joule `Q = sigma(n,T)|E|^2` -> thermal -> T feeds back to sigma(T), n(T), Drude/dn-dT eps -> resolve.
- **Why (north star)**: conductivity, density and band edge are all T-dependent, so Joule heating feeds
  back on the fields that produce it -- the one-pass manual chain mis-predicts the operating point under
  sustained bias (thermal runaway / bias drift). Also finally wires T into the field bundle so
  `ThermoOpticModel` becomes pipeline-drivable end-to-end.
- **LC-analog**: (e). **Feasibility**: no new PDE solver -- orchestrates `electrostatics_fem`,
  `thermal_fem`, and the eps assembler + closes the documented `bridge.py` T-seam. **Oracle**: energy
  balance + reduces to the one-pass result at weak coupling.

### R7. Quantum-corrected eps_zz from sub-band wavefunctions (intersubband)  [#10]  -- M, High
- **Physics**: the shipped path collapses the SP solve to a scalar n(z) -> local Drude. Instead build an
  **anisotropic** `eps_zz` with an intersubband Lorentzian at `hbar w_ij = E_j - E_i` from the
  `{E_i, psi_i, n_s,i}` the SP solver already returns (matrix elements `<psi_i|z|psi_j>` are a trapz over
  the existing grid), with sub-band-averaged Kane optical mass and sub-band-specific broadening.
- **Why (north star)**: in a ~1 nm accumulation layer `w_ij` lands in the telecom band -- a gate-tunable
  modulation channel a classical DD + *local* Drude pipeline structurally cannot produce; feeds the
  existing diagonal-anisotropic FEM/RCWA/FDTD path unchanged (Im(eps_zz)).
- **LC-analog**: (b)+(e). **Feasibility**: a new `EffectModel` consuming a `SubbandResult`; the bridge
  already carries tensor eps. **Oracle**: oscillator-strength sum rule; intersubband line vs published
  TCO accumulation-layer data.

### R8. Burstein-Moss band-filling + bandgap renormalization edge  [#7]  -- M, Med
- **Physics**: degenerate ITO blueshifts its optical gap with doping:
  `Eg_opt(n) = Eg0 - dE_BGR(n) + (hbar^2/2)(1/m_vc)(3 pi^2 n)^(2/3)`; the interband edge above
  `Eg_opt(n)` adds an Im(eps) the bare Drude omits, with a Kramers-Kronig partner shifting Re(eps) in the
  NIR. Implement as a `DeltaEffect` (Tauc/parabolic edge + KK).
- **Why (north star)**: at n ~ 1e27 m^-3 Burstein-Moss is hundreds of meV -- it corrects the exact ENZ
  crossing wavelength and adds a doping-tunable interband loss channel the Drude-only model misses.
- **LC-analog**: (b)+(f). **Feasibility**: `ComposedEffect(background=DrudeOptical,
  deltas=[BursteinMossEdge])` -- the compose/delta/passivity + `kramers_kronig_dn` machinery already
  ships. **Oracle**: Park / measured n,k blueshift vs density.

### R9. (Stretch) Carrier-heating (electron-temperature) ENZ nonlinearity  [#8]  -- L, High
- **Physics**: the real origin of ITO's record ENZ nonlinearity is intraband carrier *heating*: a pulse
  raises `T_e`, which (via Kane nonparabolicity) raises `<m*(T_e)>` and shifts `Gamma(T_e)`, so
  `wp^2 ~ n/<m*>` drops and eps moves -- huge, fast (sub-ps), reversible near ENZ where d eps/d wp^2
  diverges. Two-temperature ODE `C_e dT_e/dt = -G(T_e-T_l) + alpha I(t)`.
- **Why (north star)**: this **is** the headline all-optical ENZ physics (Alam/Boyd). The generic chi3
  captures the symptom, not the time-asymmetric dynamics or the zero-crossing enhancement.
- **LC-analog**: (a)+(e). **Feasibility**: a small two-temperature ODE driver -> `m_opt(T_e)`,
  `gamma(T_e)` callables (leans on R2/R3) feeding the existing `transient_optics` TMM loop; the
  fully-self-consistent FDTD version is research-grade. **Oracle**: published ITO ENZ pump-probe rise/
  relaxation; reduces to linear at low intensity.

### R10. (Grand) Self-consistent opto-electronic-thermal FDTD transient  [#32]  -- L, High
- **Physics**: co-evolve carriers (`wp^2(r,t)` updated each step from the live N), the FDTD Drude ADE,
  and heat (`Q = omega eps0 Im(eps)|E|^2`) on one multi-rate time axis -- the optics analog of the full
  Erickson-Leslie LC deepening with back-action.
- **Why (north star)**: the *literal* device physics -- real turn-on waveform, self-heating ENZ drift,
  damage threshold, optical-power-dependent contrast -- all emergent here and unreachable from the
  current static-eps FDTD or the one-way `transient_optics.py`.
- **LC-analog**: (a)+(e). **Feasibility**: a new outer multi-rate loop reusing every single-physics
  solver; the only kernel change is letting `wp^2/eps` vary per outer step (the ADE `aJ,bJ` already
  recompute trivially). DEVSIM<->FDTD<->thermal grid interpolation via `bridge.py` is the integration
  risk. **Oracle**: reduces to each single-physics solver when the coupling is switched off.

---

## Tier 1 -- The most direct LC analogs (order-parameter dynamics)

The clearest "deja vu" of the LC work: a quantity currently a static dial becomes a time-evolved
order-parameter field. (R5 transient heat above is also one of these.)

### R11. Landau-Lifshitz-Gilbert magnetization dynamics M(t)  [#18]  -- M, High (general+MO devices)
- **Physics**: evolve the unit magnetization `m = M/Ms` (the *magnetic director*) by LLG
  `dm/dt = -(gamma_g/(1+a^2))[m x H_eff + a m x (m x H_eff)]`, `H_eff = H_applied(t) + H_anisotropy +
  H_demag`. Precession is the analog of the LC dielectric torque; Gilbert damping `a` is the analog of
  the rotational viscosity gamma1. `m_z(t) -> g(t) -> Faraday(t)`.
- **Why**: the single most direct mirror of the LC deepening -- "magnetization" goes from a [-1,1] dial
  to a true dynamical field, giving FMR-limited (GHz) switching *waveforms*.
- **LC-analog**: (a). **Feasibility**: a new pure-numpy/scipy `magnetization_dynamics.py` mirroring
  `lc_dynamics.py` (solve_ivp, reuse the v_step/v_rc drive helpers) + a `magnetization_to_extra_fields`
  bridge; no new optical solver. Pairs with R13. **Oracle**: analytic FMR frequency + small-angle decay,
  mirroring the lc_dynamics tau self-test.

### R12. PCM nucleation-and-growth crystallization  [#24]  -- M, High (general PCM)
- **Physics**: replace the lumped single-rate JMAK with classical-nucleation-theory two-process
  kinetics: nucleation rate `I_n(T) = I0 exp(-(E_d+E_g*(T))/kT)` (the Gibbs-Thomson barrier `E_g*`
  diverges near melt -> the C-shaped TTT nose) + interface growth velocity `u(T)`, combined in the
  extended-volume Avrami integral.
- **Why**: the direct phase-transition-dynamics counterpart of the LC-dynamics work; sets PCM switching
  energy, speed, and the crystalline/amorphous contrast window. Reduces byte-identically to the current
  model at constant rates.
- **LC-analog**: (a). **Feasibility**: `PCMSwitching.integrate` is already a pure-numpy time loop over a
  T(t) pulse; this is the same forward accumulation with two rate functions. New dataclass /
  `avrami_mode` flag. **Oracle**: the C-shaped TTT nose vs an analytic CNT case; reduces to JMAK.

### R13. Full vector gyrotropic tensor eps_ij = eps_r d_ij - i eps_ijk g_k(m)  [#19]  -- S, High (enables R11)
- **Physics**: generalize `MagnetoOpticModel` from the fixed polar (m||z) tensor to the full first-order
  MO law with gyration vector `g = Q eps_r m` along an arbitrary magnetization (transverse/Voigt MO +
  Cotton-Mouton). Hermitian for real g.
- **Why**: without it an LLG `m(t)` that tilts out of z cannot be expressed -- this is the binding limit
  on coupling magnetization dynamics to optics. Enables transverse-geometry modulators + the full MOKE
  family.
- **LC-analog**: (b). **Feasibility**: a ~30-line rewrite of `MagnetoOpticModel.eps` (Levi-Civita
  contraction), backend-agnostic, reducing byte-exactly to today at m=z; the 3-D FDTD cyclotron ADE
  needs a small generalization for out-of-z m. **Oracle**: extend `magneto_optic_faraday.py` with a
  Voigt/transverse circular-eigenmode TMM.

### R14. Two-temperature model (electron + lattice)  [#29]  -- L, Med (gates the ultrafast-ENZ regime)
- **Physics**: `C_e(T_e) dT_e/dt = div(k_e grad T_e) - G(T_e-T_l) + S`; `C_l dT_l/dt = div(k_l grad T_l)
  + G(T_e-T_l)`. Electrons and lattice are out of equilibrium on sub-ps timescales; reduces to single-T
  Fourier once `T_e=T_l`.
- **Why**: the thermal analog of static-director -> Erickson-Leslie; couples the heat bath to the
  carrier population the optics depends on -- the substrate for the R9 carrier-heating nonlinearity.
- **LC-analog**: (a)+(e). **Feasibility**: a NEW coupled two-field transient solver (two H1 spaces, the
  G term, nonlinear `C_e(T_e)`); builds on R5. **Oracle**: published Au/ITO TTM relaxation curves.

---

## Tier 2 -- High-value mechanism completeness (broader than the north star)

### R15. True chi2 (Pockels/SHG/rectification) + dispersive chi3 ADE in FDTD  [#33]  -- M, High
- Add a nonlinear-polarization ADE the E-update consumes (the Lorentz-pole "extra state" pattern already
  proven across numpy/numba/jax/cuda). `effects.py` has a frequency-domain Pockels the FDTD cannot use;
  ENZ media have giant chi3 + harmonic generation a dispersionless Kerr cannot reach. **LC-analog**: (b).
  **Oracle**: undepleted-pump SHG conversion / Manley-Rowe.

### R16. Gate-oxide tunneling leakage (direct + Fowler-Nordheim)  [#3]  -- M, High
- A new interface model injecting `J_FN = A E_ox^2 exp(-B/E_ox)` (and/or Tsu-Esaki direct tunneling)
  into the semiconductor continuity equation. Converts the modulator from an ideal capacitor to a real
  leaky gate -- sets static power, hold-bias, and the low-f G(f) floor (currently ~0), bounding the
  thin-oxide design space the topology optimizer explores. **LC-analog**: (c)+(e). Needs a genuinely new
  BC type (see Appendix). **Oracle**: analytic FN slope on a uniform MOS cap.

### R17. Field- and lifetime-broadened exciton lineshape (QCSE)  [#14]  -- S, High (general EAM)
- Replace the fixed-sigma Gaussian exciton line with a field-dependent-width Voigt/Lorentzian
  `Gamma_tot(F) = Gamma_0 + hbar/tau(F)`, the WKB tunneling-out rate from the `StarkState` the driver
  returns. Governs the on/off contrast + insertion loss of any electro-absorption modulator -- the EAM
  value proposition is quantitatively wrong without it. **LC-analog**: (b). **Oracle**: field-broadening
  trend vs published GaAs QCSE.

### R18. Many-body BGR + exciton screening vs carrier density (QCSE)  [#15]  -- M, High (general)
- Make exciton binding `E_b(n_2D)` and gap `E_g(n_2D)` callables: gap renormalization
  `dE_g ~ -C n_2D^(1/3)` + Thomas-Fermi/RPA screening collapsing `E_b` to the Mott transition. The
  coupling that ties excitonic electro-absorption to the *same* gate-bias-driven carrier density that
  drives the ENZ effect -- a single device model treating free-carrier and excitonic response on one
  accumulation layer. **LC-analog**: (e)+(f). **Oracle**: Mott density vs published; binding -> 0 guard.

### R19. Density-gradient (quantum-moment) correction to DD  [#12]  -- M, Med
- Add a Bohm/von-Weizsacker quantum potential to the DD closure so the accumulation peak sets back ~1 nm
  from the oxide ("quantum dead layer") WITHOUT a full Schrodinger solve -- bringing the quantum setback
  to the 2-D/3-D metasurface geometries the 1-D eigen-SP cannot reach, at DC and (with the existing
  transient DD) in time. **LC-analog**: (b). **Feasibility**: an extra DEVSIM node/edge PDE
  (sqrt(n)-Laplacian stiffens the Newton). **Oracle**: vs the shipped 1-D SP setback.

### R20. Active gain media: four-level rate-equation ADE in FDTD  [#34]  -- M, Med
- Couple population rate equations to Maxwell; the lasing transition is a Lorentz oscillator whose
  strength is the instantaneous inversion `DeltaN(t)`. Enables loss-compensated/gain-assisted ENZ and
  lasing-threshold studies -- the library can currently only model passive/lossy media (it even has
  gain-as-a-bug tripwires). **LC-analog**: (a)+(e). **Oracle**: small-signal Lorentz-gain limit /
  known steady-state gain coefficient.

### R21. Temperature- (and anisotropy-) dependent thermal conductivity k(T)  [#28]  -- M, Med
- `k(T) = k0 (T0/T)^a` (Umklapp) makes the steady solve nonlinear (Newton or Kirchhoff transform);
  add anisotropic/cross-plane k for thin films. Real k varies 20-50% over a 100-300 K rise -- a
  constant-k solve mis-predicts peak T (hence the dn/dT phase and PCM threshold) in the high-flux regime
  modulators run in. **LC-analog**: (f)+(b). **Oracle**: Kirchhoff-transform analytic.

---

## Tier 3 -- Generality / specialized / research-grade (honest lower north-star priority)

These deepen library breadth or specific subfields but are tangential to the ITO-ENZ north star. Listed
by domain; effort/value in brackets.

- **R22. Spatially-resolved magnetization M(r) -> domain patterns feeding per-cell MO tensor** [#20]
  (M, Med) -- the magnetic analog of `lc_director_2d`; optical assembly is nearly free, but producing
  m(r) by exchange-coupled micromagnetic LLG is research-grade (accept user-supplied textures first).
- **R23. Temperature-dependent magnetization Ms(T), gyration g(T)** [#21] (S, Med) -- Curie/Bloch law;
  the direct analog of the LC order parameter S(T); reads `fields['T']`. Composes with R5/R6.
- **R24. T-dependent QCSE: Varshni gap + phonon broadening** [#16] (S, Med) -- unifies the thermal story
  across effects (composes with ThermoOpticModel); general QCSE, not ENZ-critical.
- **R25. Valence-band Luttinger-Kohn mixing (HH/LH, polarization selectivity)** [#17] (L, Med) -- the
  physically-complete MQW constitutive law (two exciton lines, TE/TM asymmetry); needs a new 4x4
  block-tridiagonal eigen-solver. General III-V, not ITO.
- **R26. Auger + radiative recombination in the bipolar model** [#2] (S, Med) -- one extra additive term
  in the existing SRH node model; completes the recombination physics for the bipolar/diode path.
  Low value for the unipolar ITO device.
- **R27. Bandgap narrowing + incomplete dopant ionization** [#5] (M, Low) -- heavy-doping/low-T
  silicon-device fidelity; ITO is treated as fully-degenerate fixed-N, so low north-star value.
- **R28. Non-equilibrium SP (quasi-Fermi) + SP<->DD Gummel coupling** [#11] (L, Med) -- removes the
  equilibrium-only caveat; a numpy quasi-Fermi SP is tractable, the full two-solver Gummel loop is
  research-grade (two grids + fragile convergence).
- **R29. Graphene magneto-optic Kubo sigma tensor (Landau-quantized) + 2x2 sheet BC** [#22] (M, Med) --
  adds B as a second knob; needs the off-diagonal sheet-current weak form. General MO, tangential.
- **R30. Graphene saturable absorption sigma(E_F,omega,|E|)** [#23] (M, Med) -- the library's first
  nonlinear constitutive law; cleanest as a time-domain ADE surface current in FDTD. General ultrafast
  photonics.
- **R31. PCM Ovshinsky electronic threshold switching** [#25] (L, Med) -- field-driven turn-on +
  Joule -> thermal -> crystallization; connects three existing solvers but the NDR branch is
  research-grade orchestration.
- **R32. Non-Fourier (Cattaneo) heat conduction at nm scale** [#30] (M, Low) -- hyperbolic heat wave +
  ballistic suppression; needs a second-order-in-time scheme on top of R5; narrow payoff for the slow
  tuner.
- **R33. Graphene nonlocal sigma(q,omega)** [#26] (M, Low) -- for deeply sub-wavelength graphene
  plasmons; fits RCWA (q = diffraction order) naturally, not the local FEM Robin BC.
- **R34. Nonlocal / hydrodynamic Drude (spatial dispersion)** [#9, #35] (M-L, Low for ENZ) -- the
  beta^2 grad(div J) pressure term; genuinely deeper for the ~1 nm regime but a *few-percent* correction
  for a low-Fermi-velocity TCO. Highest physical novelty, narrowest ITO payoff -- defer.

---

## Items needing a genuinely NEW solver (vs. natural extensions)

**Natural extensions of existing machinery (no new solver)** -- the whole Tier-0 set plus most of
Tier 2: all DEVSIM edge/node-model items (R1, R16-BC aside, R19, R26, R27) via `edge_with_derivs`/
`node_with_derivs`; all Drude-callable items (R2, R3, R8) via the `gamma_rad_s`/`m_opt_kg` callable
seam + compose/delta/KK; R4 FDTD eps(t) hook; R5 transient heat (mass matrix + theta loop); R6 the
electro-thermo-optic Picard loop; R7 quantum eps_zz (new EffectModel consuming SubbandResult); R11 LLG
(new pure-numpy module mirroring lc_dynamics); R12 PCM nucleation-growth; R13 vector MO tensor; R15
chi2/chi3 (the ADE-extra-state pattern); R17/R18 QCSE lineshape/screening; R20 gain ADE; R21 nonlinear
k(T) (iterate the existing solve).

**Genuinely new solver / operator class**:
- **R16 gate tunneling** -- no new PDE but a new *interface BC type* (reads the oxide-side field, injects
  current into the carrier continuity equation; the oxide carries no carrier variable).
- **R14 two-temperature** -- two coupled H1 spaces, nonlinear C_e(T_e); a new coupled transient solver.
- **R25 valence-band k.p** -- scalar tridiagonal Schrodinger -> 4x4 block-tridiagonal Hermitian eigen.
- **R34 hydrodynamic Drude / R33 nonlocal graphene** -- an extra P-field equation + a new ABC; every
  solver currently assumes eps(r,omega) with no k-dependence (nonlocal graphene fits RCWA, not FEM).
- **R28 non-equilibrium SP<->DD** and **R31 Ovshinsky NDR** -- not new PDEs but new coupled-solver
  orchestration with finicky convergence (Gummel two-grid interpolation; negative-differential-resistance
  branch).
- **R10 coupled opto-electronic-thermal FDTD** -- a new multi-rate outer time loop, but reuses every
  single-physics solver (only the live-eps update hook in the ADE kernel is new).

---

## Recommended sequencing

1. **The S-effort core-fidelity cluster, in order:** R1 mu(E,n) -> R2 Gamma(omega,T,n) + n-dependent
   Drude mass -> R3 shared-scattering link -> R4 FDTD eps(t) hook -> R5 transient heat. Each is small,
   directly serves the north star, and removes a documented inconsistency.
2. **The M-effort coupling items they unlock:** R6 electro-thermo-optic loop, R7 quantum eps_zz, R8
   Burstein-Moss. (R6 and R9/R10 depend on R2/R3/R5 being in place.)
3. **The most-direct LC analog, for breadth + a clean win:** R11 LLG magnetization dynamics (+ R13
   vector MO tensor), then R12 PCM nucleation-growth.
4. **The headline ENZ physics, when ready to invest:** R9 carrier-heating nonlinearity (TMM first),
   then R10 the grand coupled FDTD loop; R14 two-temperature underpins both.
5. Tier 2/3 as application needs dictate.

*Generated by an 8-domain code survey + synthesis (run wf_84bd27a5-b71). Each item is gap-verified
against the current code; effort/value are first-order estimates to be refined at implementation time.*
