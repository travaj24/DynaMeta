# Amplifier generality + realism campaign (2026-07-17/18)

Scope: close every physics-realism gap identified after the v0.7.1 audit for the fiber
amplifier (`dynameta.optics.fiber_amp`) and make the SOA package (`dynameta.optics.soa`)
datasheet-agnostic (no structural dependence on the Innolume sheet). Method: papers-first
grounding (three formulation dossiers with primary citations and numeric benchmark gates),
staged implementation (5 opus subagents on disjoint new modules + orchestrator-inline changes
to every solver-coupled file), every item shipped with discrimination-proven gates.

Branch `feat/amp-generality-2026-07`. Formulation sources are cited inline in each module
docstring; the headline primary references: Smith AO 11:2489 (1972); Kobyakov et al. AOP 2:1
(2010); Gordon OL 11:662 (1986); Blow & Wood IEEE JQE 25:2665 (1989); Lin & Agrawal OL 31:3086
(2006); Jauregui et al. AOP 12:429 (2020); Bromage JLT 22:79 (2004); Karasek IEEE JQE 33:1699
(1997); Kouznetsov & Moloney JOSA B 19:1259 (2002); Marcuse JOSA 66:216 (1976) via Schermer &
Cole IEEE JQE 43:899 (2007); Brown & Hoffman IEEE JQE 37:207 (2001); Brilliant & Lagonik OL
26:1669 (2001); Mazurczyk & Zyskind IPTL 6:616 (1994); Uskov et al. APL 72:58 (1998);
Vurgaftman et al. JAP 89:5815 (2001); Connelly IEEE JQE 37:439 (2001); Agrawal NLFO.

## Fiber amplifier — new physics

| Item | Where | Key gates (all in tests/) |
|---|---|---|
| SBS threshold (passive Smith + active gain-integral) | `fiber_amp/nonlinear_limits.py` | SMF-28 P_th 4.3 mW in [3,7]; linewidth factor x2 at dnu_src=dnu_B; active G_B=20.9 at the 735 W.m LMA point; n_th(11 GHz, 300K)=560 |
| SRS threshold + Stokes wavelength | same | 1550 P_th 1.09 W; 254x the SBS threshold; fwd/bwd 16/20 |
| TMI threshold estimator | same | C0 calibrated to 1 kW @ 20-um core, eta=0.09; exact (lambda/d)^2 scaling; 85-um-rod point within 1.8x of Eidam (documented 2-3x accuracy) |
| Double-Rayleigh MPI + RIN + penalty | same | passive closed form <1%; exponential-gain analytic oracle <0.5%; monotone in gain |
| SRS Stokes channel IN the solver | `steady_state.RamanStokes` | undepleted distributed-seed closed form 2%; Manley-Rowe photon flux conserved 2e-3 in deep depletion; active-amp ceiling (signal drained, Stokes grows); transient refuses raman (clone-drop protection) |
| GNLSE delayed Raman + self-steepening | `pulse.py` (`raman=`, `self_steepening=`) | default path byte-identical to Phase-12; Raman-only conserves energy exactly + red SSFS at 1.2x model-T_R Gordon (in [0.8,1.7]); 1/T0^4 scaling; shock steepens the TRAILING edge, conserves energy+photons; Raman+shock conserves photons, loses energy (Blow-Wood contract) |
| Thermal lens + guiding onset + phase | `thermal.py` | dT_core=0.0577 K/(W/m); lens power carries the parabola factor 2 (2 pi n k a^2); onset 1.8 kW/m @NA 0.06; 28 rad @10 W/1.55um |
| Distributed T(z) feedback | `steady_state.set_temperature_profile` + `thermal.solve_with_thermal_feedback` | UNIFORMITY oracle: constant profile == global at_temperature to 1e-6 dB / 1e-8 in nbar2; hot Yb amp converges, T peaks at the heat peak, gain drops (sigma_e(1030) falls with T); self-consistency residual <0.5 K |
| Er:Yb co-doping (EYDFA) | `eryb.py` (`ErYbAmplifier`) | Er-only limit 2.7e-6 dB vs plain EDFA; sensitization +27.6 dB; eta_tr within 3.7% of the analytic k_tr N_Er tau_Yb form (honest note: ~0.85 at measured phosphosilicate k_tr, NOT the loosely-quoted >95%); beta_Yb clamp; PCE 0.38 in the 25-40% band |
| LMA modes + bend loss + cladding geometry | `lma.py` | LP cutoffs (V=2.405); count ~V^2/2 +8%; LP01 overlap vs Gaussian 4%; Marcuse+1.27 elasto-optic: LP11/LP01 > 1e5 at R=7.5cm, Koplow-Kliner tens-of-dB/m point at R=5cm (honest note: at R=7.5cm WITH the correct 1.27 factor LP11=0.12 dB/m); eta_geo table (octagonal 0.93, centered-circular 0.40); two-population mixing model |
| PDG / PHB | `polarization.py` | Mazurczyk anchor 0.078 dB @3 dB compression; two-pol model slope (1-f)=0.026 in the signal-dominated limit (f is ORIENTATION-AVERAGED ~0.974, NOT the microscopic 2/3 -- documented); ASE-only -> zero PDG; deep-saturation 0.2-0.4 dB band; sqrt(N) cascade |
| Multi-stage chains | `chain.py` (`AmplifierChain`) | PSD-based cascade REPRODUCES Friis to 5% and the pre/post-attenuator asymmetry; single-stage == direct solve to 1e-6 dB |

## SOA — generality

| Item | Where | Key gates |
|---|---|---|
| Datasheet-agnostic calibration | `soa/calibration.py` (`DeviceTargets`, `calibrate_device`, `SOA_PRESETS`) | Innolume is now ONE preset; the fit works from the universal spec set (peak/G0/net-BW/Psat/NF/drive/L) |
| NET-bandwidth co-fit (C4-8 resolved) | same | CONTIGUOUS net -3 dB BW 58.1 nm vs sheet 60 (was 16-17); G0 35.05 dB, Psat 23.2 dBm held at a SPECTRALLY CONVERGED ensemble (n_groups auto-scales so comb spacing <= half the homogeneous FWHM -- the adversarial verifier killed the first fit's comb artifact, where 41 groups at 32.6 THz produced a 27-dB-rippled spike comb whose 'bandwidth' was disjoint fingers); band measured contiguously around nu0 (ripple-proof); sigma-linearity at S=0 verifier-CONFIRMED bit-exact; ES-hijack guard verifier-CONFIRMED (excess 0.07 dB); one S-space saturation scan serves all A_mode iterations (g depends on photon DENSITY; final report re-checks via the unshortcut path); legacy interpretation kept via fit_net_bandwidth=False (verifier-CONFIRMED it reproduces the old behavior) |
| QW/bulk gain core (non-QD devices) | `soa/qw_gain.py` (`BulkGainParams`) | transparency N_tr=1.22e24 with quasi-Fermi separation == h nu at transparency to 1e-5 meV (the sign-trap killer); log-gain g0=1.2e5/m R^2 0.988; Connelly device 26.7 dB @130 mA/600 um; QW T-sensitivity (-30% over 40 K) vs QD flatness; emission-only sub-transparency ASE; found + documented the tau-prefactor magnitude trap (radiative tau, NOT the 0.1 ps dephasing) |
| Auger, capture vs loss SEPARATE (S3-17) | `soa/qd_gain.py` (opt-in params) | tau_cap_eff = tau_cap/(1+tau_cap C_W N_w) exact (rtol 1e-12); loss C_A=4e-41 drops N_w 19-24% at 40-100 mA; ALL-DEFAULT paths float-== pre-edit goldens; numba fast path raises loudly if Auger requested (no silent term drop) |
| WL-ES detailed balance | `soa/qd_gain.py` (`with_full_detailed_balance`) | ES-GS level ratio (mu_GS/mu_ES) exp(dE/kT) = 7.4975 pinned (the dossier's '30.0' had the degeneracy ratio INVERTED -- resolved by per-state exchange balance); equilibrium pinned in the exact Pauli-blocked FERMI form rho_ES(1-rho_GS)/[rho_GS(1-rho_ES)] = exp(-dE/kT) to < 0.5% at two drives (the bare Boltzmann occupancy ratio deviates by ~rho_GS -- probe-refuted as a physics error, it is the blocking correction); WL escape rises as T falls (115/74/56x tau_cap at 250/300/350 K) |
| eh-split ES optical offset (S3-15) | `soa/qd_gain.py` | optical vs kinetic dE decoupled; ES comb moves, GS comb array-equal, None -> byte-identical |
| Area-conserving sech lineshape (S3-16) | `soa/qd_gain.py` (`lineshape='sech'`) | area = pi*hw exact (<1e-6); 5-FWHM sub-gap residual 5e-6 of peak (~2000x below Lorentzian); integrated gain conserved to 0.8%; default bit-identical |
| Sub-transparency ASE (S3-30) | `soa/ase_noise.py` (emission-only `gsp_slices` source) | finite/positive/continuous through g=0; at rho=0.5 source ∝ 0.25 = f_e f_h (legacy guard returned 0); above transparency matches the n_sp form to 2e-16 |
| Two-tone IMD3 + SFDR (S3-29) | `soa/imd.py` | closed form vs the Agrawal-Olsson two-tone FFT oracle: magnitude within 2.0-2.4x at modest G, (P/Psat)^2 slope 1.79, rolloff 11.7 dB/octave-in-power, (1+alpha^2) exact; SFDR 111 dB.Hz^(2/3); (G-1)/4 prefactor's high-gain over-prediction documented |
| Temperature (Varshni + T0) | `soa/temperature.py` | InAs 0.3551 nm/K @1300 (Vurgaftman); QD peak gain moves 0.04% over 293->343 K while the peak tracks Varshni to 2.1% (detailed-balance redistribution IS the T0 mechanism); T_ref no-op byte-equal |
| Shared beat-noise/NF algebra | `optics/amp_noise.py` | ONE implementation consumed by soa + fiber_amp (kills the S3-2/C4-3 duplication-drift class); cross-package parity gates retained |

## Adversarial verification (5 opus refuters, independent derivations + numeric probes)

Every convention-sensitive new-physics area got a dedicated refuter with execution license.
Outcomes:

* SRS solver coupling + GNLSE Raman/shock -- ALL CONFIRMED-CORRECT: Manley-Rowe placement +
  both propagation-direction invariants (5.6e-16 / 1e-6); backward-Stokes closed form ratio
  1.000000; beta3 odd-order flip proven exact against an Agrawal-convention propagator
  (1.1e-13; the un-flipped sign mirrors the pulse); causal-convolution pairing (5.6e-16);
  RK4/exp-phase consistency; Blow-Wood/Lin-Agrawal constants + Gordon T_R.
* Thermal feedback -- one REFUTED defect FIXED: the first-iteration convergence check compared
  against the coolant where the unprofiled solve represents T_ref (silently wrong up to 3.3 dB
  for cryo/hot coolants; masked when both were 300 K) -- now compares against T_ref, regression
  gate added (with a lesson: the gate's first draft used a -124 dB annihilated signal whose
  'gain' was ODE-atol noise; gain-comparing tests must keep powers far above atol). McCumber
  factor/sign/completeness/lens factor-2/onset/phase all CONFIRMED (0.0-rel reproductions);
  docstring now states the pure-McCumber %/K is a ~3-5x UPPER BOUND on the measured net slopes.
* Er:Yb -- ALL CONFIRMED-CORRECT (transfer conservation 3.9e-16 over 2000 randomized solves;
  limits exact; bracket robust over 3000 adversarial parameter sets; both caveats vanish at
  the k_back=0 default).
* qd_gain batch -- one REFUTED defect FIXED: with_full_detailed_balance's WL<->ES escape had
  the state-count ratio INVERTED (suppressing escape ~11x at 300 K); corrected to
  (rho_WL_eff A_dot)/mu_ES and pinned by a number-conserving dark-relaxation oracle gate
  (per-state Boltzmann ratios recovered to 2-5%). ES<->GS 7.4975 ratio, Auger number
  conservation, eh charge neutrality, sech area algebra, sub-transparency source identity,
  IMD3 third-order slope 1.96, Varshni all CONFIRMED at machine precision.
* Calibration co-fit -- one REFUTED artifact FIXED (the comb; see the co-fit row above);
  linearity/guards/legacy/NF/edge-cases CONFIRMED.

## Infrastructure addenda (same branch)

* `dynameta/numba_env.py` + conftest hook: numba threading-layer resilience -- exposes the pip
  tbb wheel's DLLs (python.org installs never search Library/bin; also called from
  dynameta/__init__), then probes numba's DEFAULT layer with a REPRESENTATIVE kernel (the
  small chi2 FDTD solve -- trivial probe kernels provably do not discriminate) in a sacrificial
  timed subprocess, forcing workqueue on failure. Verdict cached 24 h; POSIX/CI exempt.
* jax >= 0.11 REQUIRED (0.11 broke 0.10 APIs): pyproject `jax` extra, CI install (py>=3.11
  legs), and a runtime guard (core.backend.require_jax_011 -- core seam raises, fdtd_nd
  backend disables-with-warning). Verified: autodiff validation scripts (backend_autodiff,
  fdtd_2d_autodiff, fdtd_oblique_jax) + 26 jax-touching tests green under jax 0.11.0.

## Deliberately out of scope (documented, not silent)
Full SBS acoustic dynamics (thresholds + seeding only); TMI full simulation (estimator only,
2-3x absolute); mode-resolved multimode gain competition dynamics (per-mode overlaps provided);
inter-stage ASE saturation injection in chains (ledger carried, documented); vector/PMD
propagation (PDG lumped model only); Er:Yb k_back full 3-level algebra beyond the fast-4I11/2
factor; QW core ASE self-saturation (effective-volume convention documented).
