# Roadmap: ENZ completion, bound-states-in-the-continuum, and subwavelength-gap nonlinear generation

Status ledger for the 2026-07 physics-expansion campaign (branch `feat/enz-bic-nl-2026-07`).
Each item carries: the physics, the formulation source to extract FIRST, the implementation
target, and the validation gates it must ship with. Conventions everywhere: SI units,
exp(-i omega t) (absorber Im eps > 0), pure numpy/scipy cores with lazy heavy deps, ASCII-only,
opt-in physics byte-identical to the legacy path when off.

Inventory baseline (v0.8.0): Drude carrier->eps map + Schroedinger-Poisson accumulation
subbands (nonparabolic); the full Alam-Boyd hot-carrier chain in carriers/carrier_heating.py
(Kane m*(T_e) Fermi-averaged, gamma(T_e), C_e(T_e), two-temperature ODE, drude_of_t) feeding
the QUASI-STATIC transient_optics loop; dispersive FDTD 1/2/3D (per-CELL Drude/Lorentz +
per-CELL chi2/chi3/Raman/gain, uniform grid); linear plane-wave-driven NGSolve FEM (nm-scale
meshing); TMM/Berreman references; lumenairy bridge (RCWA incl. conical Jones); driven-spectrum
resonance_dip/resonance_shift only -- NO eigen/pole analysis anywhere.

---

## PHASE 1 -- Resonance/pole infrastructure (serves BIC + ENZ modes + doubly-resonant NL)

### 1.1 Complex-omega pole finder for layered stacks  [optics/resonance.py]
Physics: resonances/QNMs are poles of the scattering response at complex omega_t = omega_0 -
i gamma/2 (exp(-i omega t) convention => DECAYING modes have Im(omega_t) < 0); Q = omega_0 /
(2|Im omega_t|). BICs appear as poles whose Im -> 0 while remaining inside the continuum.
Implementation: a self-contained complex-omega layered S-matrix evaluator (transfer matrix with
complex k_z branches chosen for outgoing waves; material models Drude/Lorentz/constant are
analytic in omega), pole search via argument principle (contour count) + Newton refinement on
1/S11 or det(S^-1); pole TRACKING vs a swept parameter (angle/k_par/thickness/bias) by
continuation. Q_rad vs Q_abs split by the lossless/lossy two-pass (re-solve the pole with
Im(eps) zeroed: Q_rad; 1/Q_abs = 1/Q - 1/Q_rad).
Gates: Fabry-Perot etalon poles CLOSED FORM (omega_t = (pi c m / (n L)) - i c ln(1/(r1 r2)) /
(2 n L) for the symmetric slab -- derive exactly); real-axis evaluator == tmm_reference to
~1e-12; Q(R) analytic scaling; lossless Q_rad == total Q; Berreman-mode pole of a Drude ENZ
film found near omega_p with the literature thin-film dispersion trend.

### 1.2 Ringdown harmonic inversion  [optics/ringdown.py]
Physics: FDTD time traces after pulsed excitation are sums of damped exponentials; matrix-pencil
/ Prony inversion extracts (omega_0, gamma, Q) far below the FFT resolution limit -- the
geometry-general pole route (works for anything the FDTD can march).
Implementation: matrix-pencil (SVD-thresholded) on uniformly-sampled complex/real traces;
model-order selection; amplitude fit; helpers to pull traces from the FDTD solvers (opt-in
probe return, additive kwarg only).
Gates: synthetic 2-3-mode signals recovered to <1e-6 in omega and <1e-3 in Q incl. close pairs
and noise floors; an FDTD etalon ringdown Q matching the 1.1 pole to a few %.

### 1.3 Fano tooling + quasi-BIC gates  [analysis.py additions]
Physics: a quasi-BIC driven spectrum is a Fano lineshape T(omega) ~ |q + eps_r|^2/(1+eps_r^2),
eps_r = 2(omega-omega_0)/gamma; symmetry-broken quasi-BICs obey Q ~ 1/delta^2 in the asymmetry
parameter delta (THE canonical signature, Koshelev/Kivshar PRL 121:193903 (2018)).
Implementation: fano_fit(omega, T) -> (omega_0, gamma, q, Q) robust least squares with
background; q_bic_scaling helper fitting log Q vs log delta (slope -2 gate).
Gates: synthetic Fano round-trips; TMM etalon driven spectrum -> fano_fit Q == 1.1 pole Q to
~1%; slope -2 recovered from synthetic Q(delta) data.

### 1.4 Far-field polarization vortex charge  [lumenairy_bridge post-processing; PHASE-GATED]
Physics: BICs are winding-number centers (topological charge q = (1/2 pi) closed-integral
d phi, phi = polarization angle of the far-field Jones vector) on the k_par plane (Zhen et al.
PRL 113:257401 (2014)).
Implementation: post-process a conical-RCWA (kx, ky) sweep's zeroth-order Jones vectors into
the polarization-angle field + winding number around candidate BIC points. Depends on the
bridge's conical Jones surface; defer until 1.1-1.3 land.
Gates: synthetic vortex fields (q = +-1, +-2) recovered exactly; charge conservation under
small contour deformation.

## PHASE 2 -- ENZ physics completion

### 2.1 Per-cell hot-carrier two-temperature ADE in fdtd_nd (2D first)
Physics: the ENZ Kerr-like nonlinearity is LOCAL: Gamma_abs |E(r)|^2 heats electrons where the
1/|eps|^2 E_z enhancement concentrates the field; T_e(r,t) drops omega_p via the Kane
Fermi-averaged m*(T_e) and raises gamma(T_e) (carriers/carrier_heating.py already owns the
0-D chain -- REUSE its coefficients, do not duplicate: precompute m*(T_e)/gamma(T_e) lookup
tables per material and interpolate per cell).
Implementation: opt-in per-cell ADE: dU_e/dt = local absorbed power density - G(T_e)(T_e-T_l)
(+ optional lattice bath), T_e from U_e via C_e(T_e); Drude update coefficients re-derived
per cell per step (or every N steps, documented). numpy path first; numba fast path may raise
(the Auger precedent) until kernels are extended.
Gates: uniform-film limit == the 0-D carrier_heating chain (the UNIFORMITY oracle pattern from
the thermal-feedback work); zero-intensity byte-identity; pump-probe transient reproduces the
sub-ps rise / few-ps relax asymmetry; local-enhancement discrimination (structured film heats
at the hot spot, not uniformly).

### 2.2 Intra-march time-varying eps (time refraction / photon acceleration)
Physics: eps changing DURING the optical transit converts frequency (adiabatic frequency
translation; time-boundary reflections). transient_optics is quasi-static by design; this is
the true time-varying tier.
Implementation: time-dependent Drude/ADE coefficients in fdtd (1-D first): drude_of_t hook
evaluated inside the march; output frequency-conversion spectrum diagnostics.
Gates: static drude_of_t == fixed-model byte-identity; the analytic adiabatic limit
omega_out/omega_in = n_in/n_out for a slow index ramp in a uniform medium (photon-number
conservation); a fast step produces the known time-boundary frequency sidebands
(Morgenthaler-style two-frequency split, amplitudes vs analytic).

### 2.3 Mermin / extended-Drude damping  [materials + tmm_reference]
Physics: real TCO scattering is omega-dependent; the Mermin correction keeps local charge
conservation while adding gamma(omega) (Mermin PRB 1:2362 (1970)).
Implementation: MerminDrudeOptical (or an ExtendedDrude with gamma(omega) table/model) usable
by every frequency-domain consumer; FDTD keeps fitted Drude+Lorentz (document the fit path).
Gates: gamma->const limit == plain Drude byte-identical; KK causality spot-check; an ITO
linewidth comparison showing the effect's direction/magnitude.

### 2.4 Hydrodynamic (nonlocal) Drude -- LAYERED TMM tier  [optics/nonlocal_tmm.py]
Physics: the pressure term beta^2 grad(div J) adds a LONGITUDINAL wave; thin films get
thickness-dependent ENZ/Berreman shifts and bulk-plasmon resonances above omega_p. beta^2 =
(3/5) v_F^2 (high-frequency) with the Thomas-Fermi variant documented; GNOR = beta^2 ->
beta^2 + D(gamma - i omega) (Mortensen et al. Nat. Commun. 5:3809 (2014)).
Implementation: 4-wave layered transfer matrix (2 transverse + 2 longitudinal) with the
additional boundary condition J_normal = 0 at metal/dielectric interfaces (Melnyk-Harris /
Sipe ABC); reflectance/transmittance + the 1.1 pole finder running on it.
Gates: beta->0 == local TMM to 1e-10; the analytic thin-film bulk-plasmon standing-wave
resonances at k_L d = m pi (Lindau-Nilsson/ Anderegg-type multipole positions vs closed form);
ENZ-mode blueshift direction + 1/d scaling.

## PHASE 3 -- Subwavelength-gap nonlinear generation

### 3.1 Harmonic diagnostics + slope gates  [FDTD post-processing]
Physics/impl: SH/TH spectral extraction from existing chi2/chi3 FDTD runs (order-resolved for
periodic 2D), conversion-efficiency normalization, P_2w ~ P_w^2 / P_3w ~ P_w^3 slope gates,
undepleted-pump validity check. Route A (gap-loaded nonlinear dielectric) becomes fully
usable in 2D today with this alone.
Gates: slope 2.000/3.000 within 1%; the 1-D chi2 SHG closed-form coupled-wave benchmark
(already a validation oracle) reproduced through the new diagnostics.

### 3.2 Source-driven FEM + surface-SHG two-step solver  [optics/solver.py + optics/shg_fem.py]
Physics: metal SHG = surface Rudnick-Stern sheet P_s(2w) (a,b,d parameters; normal/tangential
E products) + bulk convective dipole terms; undepleted two-step: linear solve at omega ->
build sources -> linear solve at 2 omega -> radiated SH (Sipe/Heinz formalism; Rudnick-Stern
PRB 4:4274 (1971); Ciraci/Scalora reviews for numerics).
Implementation: (a) volume/surface CURRENT-SOURCE excitation in the NGSolve weak form (new
capability, also useful generally); (b) boundary-normal field extraction on metal surfaces;
(c) the two-step driver + Rudnick-Stern source assembly; (d) SH radiated-power extraction.
Gates: flat-surface SHG vs the analytic Rudnick-Stern/Sipe result; small-sphere SHG vs Dadap
multipole closed form (Dadap et al. PRL 83:4045 (1999)); slope-2; reciprocity spot-check.

### 3.3 Hydrodynamic FEM linear ladder (HDM -> GNOR -> QCM)  [2D/3D]
Physics: sub-5-nm gap LINEAR fields are wrong in local Drude (enhancement diverges);
HDM/GNOR bound them; sub-1-nm tunneling gaps need the quantum-corrected effective-conductivity
gap material (Esteban et al. Nat. Commun. 3:825 (2012)).
Implementation: coupled (E, J) weak form with the ABC in NGSolve; GNOR = complex beta^2 knob;
QCM = a gap-material model (easy once parameterized).
Gates: nonlocal Mie/cylinder closed forms (analytic HDM benchmarks); local limit; gap-size
scaling of enhancement saturating instead of diverging.

### 3.4 Deferred/stretch (documented, not scheduled)
Nonuniform/subgridded FDTD mesh for nm gaps in 3D; time-domain hydrodynamic FDTD (derives
surface+bulk SHG self-consistently); pump-depleted coupled-wave option; RCWA complex-omega
poles through the lumenairy bridge (needs complex-frequency support upstream).

---

## PHASE 4 -- Three-wave-mixing generality (SFG / DFG / OPA / SPDC-design)

The instantaneous full-field chi2 FDTD inherently contains ALL second-order mixing (two colors
present -> omega1 +/- omega2 products, parametric gain, idler generation); Phase 4 makes that
usable and extends the FEM two-step beyond degenerate SHG. Spontaneous PDC is quantum
(vacuum-seeded) and is deliberately scoped as a DESIGN tier via the quantum-classical
SFG<->SPDC correspondence -- no photon-pair simulation is pretended.

### 4.1 Three-wave-mixing coupled-wave reference solver  [optics/twm_reference.py]
Physics: the classical three-wave CWEs dA3/dz = i (omega3 d_eff/(n3 c)) A1 A2 e^{i dk z} (+
cyclic, exp(-i omega t) sign convention derived carefully), dk = k3 - k1 - k2; undepleted
closed forms (SFG sinc^2, parametric gain g = sqrt((kappa1 kappa2)|A_p|^2 - (dk/2)^2), OPA
signal cosh^2 / idler sinh^2), Manley-Rowe EXACT in the depleted integrator. Boyd ch. 2.
Implementation: undepleted closed forms + a depleted RK45 integrator; quasi-phase-matching
(sign-flipping d_eff(z)) support. THE oracle for 4.2/4.3.
Gates: Manley-Rowe photon-flux invariants to 1e-12 in the depleted integrator; undepleted
limits recovered; SHG degenerate limit == the existing chi2 coupled-wave oracle; QPM
first-order efficiency = (2/pi)^2 of phase-matched.

### 4.2 Two-color FDTD sources + mixing-band diagnostics  [fdtd sources + optics/harmonics.py]
Implementation: bichromatic source injection (additive opt-in: a second carrier
frequency/amplitude on the existing source machinery, 1-D first then 2D); harmonics.py
generalized: mixing_spectrum(trace, f1, f2) extracting bands at p f1 + q f2 (|p|+|q| <= 2 plus
2f1-f2/2f2-f1), REUSING the pump-bandwidth leakage guard per band (two pumps -> two sigma
checks + cross-leakage).
Gates: SFG bilinear slopes (P(f1+f2) linear in P1 at fixed P2 and vice versa, slope 1.00 each;
total slope 2 vs joint scaling); DFG idler appears at f1-f2; OPA: pump+weak seed -> seed GAIN
and idler growth with idler-photons == signal-photons-gained (Manley-Rowe from the band powers,
few %); parametric gain vs the 4.1 closed form (undepleted pump, phase-matched thin slab,
~10%); zero-chi2 floors; byte-identity of single-color paths.

### 4.3 Nondegenerate two-step FEM (SFG/DFG)  [optics/shg_fem.py extension]
Implementation: sfg_two_step(omega1, omega2, ...): TWO linear fundamental solves -> mixing
source at omega3 = omega1 + omega2 (or omega1 - omega2 for DFG) with the correct
NONDEGENERATE permutation bookkeeping (P(omega3) = eps0 chi2 [E1 E2 + E2 E1] -- factor 2
relative to the degenerate 1/2! SHG convention; state the convention explicitly) -> sourced
solve at omega3; the flat-surface Rudnick-Stern oracle generalized to nondegenerate mixing.
Gates: DEGENERATE LIMIT: sfg_two_step(omega, omega)/4 == shg_two_step(omega) exactly (the
permutation-factor identity -- derive and pin); nondegenerate flat-surface vs the generalized
analytic oracle (~10%, oblique-PML-limited); bilinear pump slopes 1.00/1.00.

### 4.4 SPDC design tier  [optics/spdc_design.py]
Physics: the quantum-classical correspondence -- the SPDC pair-generation rate of a structure
equals a universal prefactor times its CLASSICAL SFG efficiency integrated over the emission
band (Helt, Liscidini, Sipe JOSA B 29:2199 (2012); the 'reversed SFG' relation); the joint
spectral amplitude is the phase-matching function Phi(omega_s, omega_i) x pump envelope.
Implementation: pair_rate_from_sfg (uses 4.1's classical efficiency), jsa(omega_s, omega_i)
from the CWE phase-matching integral (incl. QPM), Schmidt-number/heralding-bandwidth
estimators. DOCUMENTED as design-level: no quantum state is simulated.
Gates: the textbook CW-pumped bulk-crystal pair-rate formula recovered in the uniform limit;
JSA anti-diagonal width == pump bandwidth, diagonal width == phase-matching bandwidth
(constructed cases); Schmidt number -> 1 for a matched-filter case and >> 1 for a long-crystal
CW case; QPM shifts the JSA centre exactly as 4.1 predicts.

## PHASE 5 -- Stretch items (formerly deferred; now scheduled)

### 5.1 Nonuniform-z FDTD grid  [fdtd.py / fdtd_nd]
Physics/need: nm gaps and accumulation layers in wavelength-scale cells (uniform grids explode
in 3D; 2D is painful). Implementation: the standard nonuniform Yee mesh in z (spatially varying
dz with the dual-grid half-cells; single global dt from the smallest cell's Courant bound;
2nd-order on smoothly graded meshes), 1-D first then the 2D solver's z axis (x stays uniform
for periodicity). Grading helper (geometric refinement into designated thin layers).
Gates: uniform-limit byte-identity; a thin-film R/T on a graded mesh == uniform fine mesh
(<0.1%) at a fraction of the cells; convergence order ~2 on smooth grading; a 3-nm gap slab
resolved with ~10x fewer cells than uniform at matched accuracy; Courant stability at the
documented bound.

### 5.2 Time-domain hydrodynamic FDTD (self-consistent metal nonlinearity)  [optics/hydro_fdtd.py]
Physics: Maxwell + electron fluid (n, v): dn/dt = -div(n v); m n (dv/dt + (v.grad)v) =
-e n (E + v x B) - m gamma n v - grad p, p = (beta^2 m/ ...) n^(5/3) closure (document the
Thomas-Fermi/high-frequency choice consistent with nonlocal_tmm's beta). The convective +
pressure + magnetic nonlinearities DERIVE surface+bulk SHG self-consistently (no Rudnick-Stern
parameters). Implementation tiers: 1-D longitudinal (linear validation vs nonlocal_tmm bulk
plasmons) -> 2-D TM with the full nonlinear terms (SHG from a flat metal surface).
Gates: LINEAR tier == nonlocal_tmm (bulk-plasmon peaks <1%, beta->0 local limit); SHG slope 2;
flat-surface SHG scaling vs the Rudnick-Stern oracle TREND (angle dependence + order of
magnitude; the a,b parameters are EMERGENT here -- extracting effective a,b from the
simulation and comparing to the free-electron a~1, b=-1 is the stretch gate); energy bookkeeping.
MINIMUM CORE fallback: the 1-D linear tier + the 2-D linear tier + documented nonlinear status.

### 5.3 Structured-surface (grating) SHG + Dadap check  [shg_fem.py extension]
Need: the flat-surface analytic sampling in shg_two_step cannot handle structured metal
boundaries -- assemble the Rudnick-Stern sheet on ARBITRARY metal boundaries using FEM traces
(local normals via specialcf.normal, E_perp from the field solution just outside, the
verification-fixed SI scaling). Gates: FLAT LIMIT == the analytic-sampling path (<2%); a
shallow sinusoidal grating's SH vs perturbation theory trend (linear in depth for shallow
gratings, enhancement at the SH diffraction anomaly); the Dadap small-cylinder multipole
scaling (SH ~ (a/lambda)^? dipole-forbidden scaling) if the mesh cost is tolerable --
otherwise the grating gates stand alone.

### 5.4 2-D coupled-HDM stabilization  [hydro_fem.py]
Need: the shipped 2-D coupled (E, J) form is indefinite (raises HydroFEMUnstable). Candidate
cures to INVESTIGATE in order: (a) grad-div augmentation on the J block with parameter study;
(b) static condensation / direct factorization with complex shift; (c) reformulation via a
scalar longitudinal potential (J = J_T + grad phi splitting -- the longitudinal physics is a
scalar Helmholtz in the metal); (d) first-order least-squares form. Success = the cylinder
blueshift gate (Raza closed form, 2-3 radii, 15%) and the gap-saturation gate (local/hydro
ratio grows as gap shrinks) running stably. HONEST fallback if all fail: document the
spectral analysis of WHY (the operator's essential spectrum) and keep the 1-D tier + raise.

### 5.5 Rational (AAA) pole extraction from real-axis sweeps  [optics/aaa_poles.py]
Reframed from 'RCWA complex-omega via bridge': instead of forcing complex frequency through
lumenairy, fit an AAA rational approximant (Nakatsukasa-Sete-Trefethen SIAM JSC 40:A1494
(2018)) to REAL-frequency response samples from ANY solver (RCWA bridge, FEM, TMM, measured
data!) and read poles/residues/Q from the approximant -- the established QNM-extraction
route with no upstream changes, and it works on experimental spectra too.
Gates: FP etalon poles from real-axis TMM samples vs optics.resonance exact poles (<1e-6 rel
Re, <1e-3 Q, clean-data limit); noise robustness (0.1% noise -> Q to a few %); spurious-pole
(Froissart doublet) filtering demonstrated (residue threshold + stability-under-sample-count);
a lumenairy-bridge RCWA sweep's resonance Q vs a Fano fit of the same data (consistency few %).

### 5.6 Real-BIC end-to-end capstone  [validation + tests]
Design a symmetry-protected BIC in a 1D-periodic high-index grating slab (at the Gamma point of
the first leaky band -- standard textbook structure), then close the loop with the ENTIRE
Phase-1+5 stack: (i) driven RCWA/FDTD spectra show the resonance VANISHING at normal incidence
and Q ~ 1/delta^2 vs the symmetry-breaking angle (Fano tooling); (ii) AAA/pole tracking shows
Im(omega) -> 0 at Gamma; (iii) the conical Jones map around Gamma carries the +-1 polarization
vortex (bic.py, live lumenairy). This is the integration proof that the BIC stack answers real
design questions. Gates are the three signatures themselves.

### 5.7 Hot-carrier small extensions  [hot_carrier.py]
Finite lattice heat capacity (C_l < inf: coupled T_l(r,t) per cell, bath -> two-temperature
proper) as an additive opt-in; document the numba-path status. Gates: C_l -> inf reproduces
the fixed-bath tier byte-close; energy closure incl. the lattice reservoir.

## Build order and rationale

1 (1.1 -> 1.2 -> 1.3, then 1.4): pure new modules, no solver surgery, unblock BIC work AND
give every later phase its resonance instrumentation (doubly-resonant SHG = poles at omega
and 2 omega; ENZ Berreman dispersion = 1.1 on the Drude stack).
2 (2.1 -> 2.2 -> 2.3 -> 2.4): ENZ dynamics next -- highest product relevance (tunable ITO);
2.1/2.2 are additive opt-ins on owned solvers; 2.4's TMM tier stands alone.
3 (3.1 -> 3.2 -> 3.3): 3.1 is nearly free; 3.2 introduces the one genuinely new solver
capability (source-driven FEM) behind clean gates; 3.3 reuses 2.4's physics in FEM form.

Every item: papers-first (extract the formulation before deriving), independent + absolute-
scale oracles (the audit lessons), adversarial verification before merge.

## Status

| Item | State |
|---|---|
| 1.1 pole finder (layered) | SHIPPED (optics/resonance.py; 9 gates incl. FP closed form to 2e-16, ENZ/Berreman film pole; KEY NUMERICS: the Abeles characteristic matrix is the branch-cut-free pole function -- 1/t and Redheffer cascades inherit the layer sqrt cut and break the argument principle; p-pol has a spurious eps=0 admittance pole to box around) |
| 1.2 ringdown inversion | SHIPPED (optics/ringdown.py + opt-in FDTD trace probe; matrix-pencil to 1e-15 on synthetics, resolves half-linewidth pairs the FFT cannot, etalon Q 1.1% vs closed form; energy-Q convention == pole Q by construction) |
| 1.3 Fano tooling | SHIPPED (analysis.fano_fit/lorentzian_fit/quasi_bic_scaling; VARPRO parameterization stable across all q; slope -2 law gate; NOTE pole-Q vs finesse-Q differ ~6% on low-finesse etalons -- fit windows documented) |
| 1.x cross-gate | SHIPPED (tests/test_resonance_crossgate.py: ONE etalon mode measured by pole finder + FDTD ringdown + driven-spectrum fit agrees on (omega, Q); ringdown excitation-band placement is part of the operating manual -- a mid-gap band rings a skirt mode with a source-transient-contaminated tail) |
| 1.4 vortex charge | SHIPPED (optics/bic.py; RP1 doubled-angle winding, q exact for +-1/+-2, C-point localizer, LIVE conical-RCWA integration test vs lumenairy 5.25.0; charge quantized via round-then-halve) |
| 2.1 hot-carrier ADE | SHIPPED (optics/hot_carrier.py + fdtd_nd numpy kernel; J.E dissipation 2nd-order vs analytic Drude absorption; uniformity oracle ~0.1% vs the 0-D chain (p_abs lives on the HALF-step grid); below-ENZ T 0.42->0.74 under pump, 4.4 fs rise / 366 fs relax; locality corr 0.994; numba guard; NOTE FDTDLayer actually lives in fdtd_nd/spec.py) |
| 2.2 time-varying eps | SHIPPED (fdtd.py _run_tv + run_uniform_time_boundary oracle + frequency_conversion_diagnostic; D-PRESERVING update (E *= eps_old/eps_new); adiabatic omega ratio 0.03% + photon number 1%; Morgenthaler a=(r^2+r)/2, b=(r^2-r)/2 from D/B continuity hit at 0%/2%; NOTE this fdtd.py's Yee sign: forward wave has Hy = -(n/(mu0 c)) Ex) |
| 2.3 Mermin/extended Drude | SHIPPED (materials ExtendedDrudeOptical + gamma_ito_extended + check_kk; PROVED k->0 Mermin == Drude exactly (independent hydrodynamic-Lindhard oracle, residual ~(beta k/omega)^2) so MerminDrudeOptical defers finite-k to 2.4 loudly; ITO preset reproduces optical-mobility>DC (Im eps ratio 0.47-0.66 near-IR)) |
| 2.4 hydrodynamic TMM | SHIPPED (optics/nonlocal_tmm.py; k_L^2 = (omega^2 + i gamma omega - wp^2/eps_inf)/beta^2; ABC J_n=0 folded to an even-in-kz 2x2 Abeles form (branch-cut-free); bulk-plasmon peaks at k_L d = m pi to 0.004-0.09% with the m=1,3,5 SYMMETRY SELECTION RULE (pole finder confirms the dark m=2 at Q~882); GNOR monotone broadening via pole linewidths; ENZ blueshift +0.22% @50nm -> +1.99% @10nm; pole-finder integration 0.0125%) |
| 3.1 harmonic diagnostics | SHIPPED (optics/harmonics.py + opt-in 2D trace kwarg; 1-D SHG closed form reproduced to 2.6e-3; slopes 1.99986 (SHG) / 2.9929 (THG, the Kerr term IS instantaneous full-field and radiates genuine 3w); floors -300 dB; order-resolved Poynting extraction; 2026-07-19 adversarial round: pump-BANDWIDTH guard added -- a broadband pump's own tail into the 2w band mis-reads as SHG (pure-fundamental sigma_f/f0 = 0.21 measures phantom P_2w/P_w ~ 9e-4), so harmonic_spectrum now measures sigma_f over [0.5, 1.5] f0 and warns above ~0.21 (1-2 bw) f0 (= 0.147 f0 default), with the numeric precondition documented + gated) |
| 3.2 source-driven FEM + RS SHG | SHIPPED FULL (solver.solve_fem_sourced + shg_fem: first-principles rudnick_stern_flat_shg oracle independently re-derived to 2e-15, current-sheet closed form 1e-12, reciprocity 1e-9, s-pol == 0 / normal-incidence a-term == 0 / slope 2.0000; 2026-07-19 adversarial round KILLED the earlier deferral story: the 'near-null interior mode ~2-2.5x extraction bias' was NOT reproducible (extraction measured accurate 0.3% normal / 1-5.5% oblique even in an all-vacuum open cell) -- the real defect was an SI-vs-nm units bug in the SH sheet's equivalent vacuum field (E0 low by exactly S=1e9, power 1e18) + tangential-only probe_pol under-counting p-pol power by cos^2 theta; BOTH FIXED, and the previously-deferred quantitative FEM-vs-oracle gate now SHIPPED at 0.5-1.2% measured (gated 10%, tests/test_shg_fem.py); REAL remaining constraint: the power read-out requires a LOSSLESS superstrate (lossy Im eps = 1 inflates p_up ~5x); Dadap/grating stretch pending only the structured-surface source assembly) |
| 3.3 hydrodynamic FEM ladder | SHIPPED (optics/hydro_fem.py: Toscano coupled E-J weak form; 1-D layered tier == nonlocal_tmm to 1e-9 with bulk plasmons to 0.03-0.3%; QCM gap material with the Esteban non-monotonic ~1-1.5 nm enhancement peak + tunneling short; GNOR monotone; the 2-D coupled near-field is numerically UNSTABLE (indefinite J-block) and raises HydroFEMUnstable rather than returning garbage -- gated to raise; 2-D local tier validated (Froehlich 0.2%, 1/gap divergence)) |
| 5.1 nonuniform-z FDTD grid | SHIPPED 1-D (fdtd.py _run_nu + make_graded_z/_refined_full_edges + solve_fdtd_1d z_edges=/refine=; standard nonuniform Yee, E primal / H dual, LOCAL spacing per derivative -- dual for E, primal for H; the Drude ADE + Kerr coeffs are dz-INDEPENDENT so they compose UNCHANGED, only the two derivative denominators go per-cell; single global dt from the smallest cell, S=c*dt/min(dz)<=1 GUARDED. 6 gates green (tests/test_fdtd_nonuniform.py, 37/37 incl. the 3 pre-existing fdtd suites): (1) uniform-limit BYTE-IDENTITY via uniform_z_edges (first np.diff == dz exactly -> detected + routed back through the scalar _run; positional grids carry ULP noise so the diffs must NOT be consumed); (2) lossy Drude film graded-vs-uniform-fine dR=3.1e-4/dT=4.5e-4 at 8.0x fewer cells (238 vs 1914); (3) convergence slope ~2.5 (>=1.7) on smoothly graded slabs; (4) 3-nm eps=12 gap dR=1.1e-3 (<0.5%) at 10.8x fewer cells (344 vs 3704); (5) 50k steps at S=1 bounded (tail 1e-15), courant=1.05 raises; (6) Drude slab absorption A>0.02 on graded mesh. NOT supported on the nonuniform grid (raises loudly): time-varying hooks + bichromatic second_source; trace + static materials ARE supported. KEY NUMERICS: geometric grading caps per-cell ratio ~1.15 (min transition_cells across the largest jump) and SNAPS layer boundaries onto nodes (material interfaces resolved); the accuracy limiter is the node-vs-cell-center staggering offset at the film boundary (~O(dz_film)) -- refine the film ~10x to push both R and T under 0.1%. 2-D fdtd_nd z-grading = documented follow-up tier.) |
| 4.1 TWM coupled-wave reference | SHIPPED (optics/twm_reference.py: TWMSpec + sfg_undepleted/opa_gain/twm_propagate (DOP853) + QPM helpers; CWE signs pinned under exp(-i omega t) -- the A3 phase is exp(-i dk z), the CONJUGATE of the exp(+i omega t) textbook form; d_eff = chi2/2 with the SHG 1/2 degeneracy factor byte-identical to the existing 1-D SHG oracle; 11 gates: undepleted-SFG closed form, Manley-Rowe invariants, OPA sinh gain, sinc^2 phase-mismatch, QPM (2/pi)^2, pump-depletion Jacobi limit) |
| 4.2 two-color FDTD + mixing diagnostics | SHIPPED (solve_fdtd_1d second_source kwarg + harmonics.mixing_spectrum(trace, f1, f2); bands {f1, f2, f1+f2, \|f1-f2\|, 2f1, 2f2, 2f1-f2, 2f2-f1} with per-band pump-leakage guards + degenerate-spacing merge; SFG bilinear slopes 0.9995/0.9997; DFG/OPA idler on/off ratios 1e26-1e29; Manley-Rowe photon closure ~9%; parametric gL_meas/gL_pred = 0.70; PHYSICS FINDING: a non-dispersive phase-matched chi2 slab CANNOT net-gain a seed (SFG coupling sqrt(w_s w_sum) always beats parametric sqrt(w_s w_i)) -- the OPA gate requires dispersively phase-mismatching the SFG channel (Lorentz pole), documented in the test; 47 passed + 1 graceful skip) |
| 4.3 nondegenerate two-step FEM | SHIPPED (shg_fem.sfg_two_step + rudnick_stern_flat_sfg + sfg_field_transverse_kx; nondegenerate D=2 factor (vs degenerate SHG) proven by the degeneracy identity sfg(w,w)/4 == shg(w) at 2.2e-15; DFG = conjugated convention P_z ~ E1 conj(E2) with K_par3 = k_par1 - k_par2; 17 gates green incl. RS analytic oracle + FEM extraction at the 3.2-validated accuracy) |
| 4.4 SPDC design tier | SHIPPED (optics/spdc_design.py: pair_rate_from_sfg via the Helt-Sipe classical<->quantum correspondence (prefactor EXACTLY 1/(2 pi)), jsa/jsi, schmidt_number, heralded_bandwidths; unapodized-sinc purity floor K ~= 1.206 reproduced; rate scales linearly in pump power + L^2 phase-matched) |
| 5.2 time-domain hydrodynamic FDTD | SHIPPED TIERED (optics/hydro_fdtd.py, new self-contained module: p-pol fixed-kx TM reduction (d/dx -> i kx), staggered Yee-z + leapfrog-t, linearized fluid dJ/dt = eps0 wp^2 E - gamma J - beta^2 grad rho with drho/dt = -div J (eliminating (rho,J) reproduces nonlocal_tmm's k_L^2 EXACTLY), pressure as explicit rho source inside the exact fdtd a_J/b_J semi-implicit Drude sub-step, hard-wall ABC J_z=0. TIER 1 (linear) FULLY VALIDATED: beta->0 byte-identity 0.0e0; THE cross-solver gate: confined ring-down eigenfrequencies (matrix-pencil) vs k_L d = m pi closed form AND nonlocal_tmm peaks m=1/3/5 at 0.047/0.069/0.065%; energy 1.2e-4. TIER 2 = CONFINED (kx=0) self-consistent metal SHG: full (n,v) nonlinearity radiates 9.4x the linearized-reference 2w, physical-SH slope 2.125; DEFERRED w/ VERIFIED blockers (documented SCOPE note): radiated flat-surface SHG selection rule + emergent Rudnick-Stern a need a RADIATING OBLIQUE solve -- (i) single-kx mode cannot carry E(kx)^2's 2kx content (needs {0,kx,2kx} multi-mode), (ii) fixed-kx Mur ABC MEASURED unstable obliquely (5000-18000x growth over 2e5 steps) -- PML prerequisite. CAVEATS: collocated (n,v) grid leaks ~0.3-1% linear 2w background (SH excess is measured by PHASE-COHERENT time-domain subtraction of the linearized reference -- wave-2 hardening; the earlier magnitude subtraction was resolution-biased, slope 2.6 at 2x resolution vs 2.1); perturbative drive only (convective term unstable above ~5% density modulation); single-sided exp(-i w t) source lands physical w>0 at NEGATIVE fftfreq bins. WAVE-2 VERIFIED: discrete longitudinal dispersion is clean 2nd-order (m=5 error -0.22%/cpl8 -> -0.01%/cpl32, matching the modified-wavenumber prediction in sign+magnitude; NOT error cancellation -- FDTD, closed form, and nonlocal_tmm agree three ways); kx=0 ring-down == oblique peak positions to ~2e-6 (c kx << beta k_L). 10 gates green, ~3 min) |
| 5.3 grating-surface SHG + Dadap | SHIPPED GRATING TIER (shg_fem.shg_structured_two_step, additive; Dadap = honest roadmap-sanctioned fallback. HONEST FEM FINDING (measured, ngsolve 6.2.2604): the surface-current route CANNOT source a NORMAL sheet in HCurl -- the boundary form (n.v.Trace()) is IDENTICALLY ZERO facet-by-facet (tangential trace orthogonal to facet normal) and the full normal trace is refused as a BND-form; a thin volume band needs sub-element resolution but per-solid maxh is silently IGNORED in this build -> 15-200% quadrature noise. ROUTE USED: the scattered-field method generalized to corrugation -- E_perp extracted via the CONTINUOUS normal-D (D_perp = eps E.n single-valued; E.n is not) by standoff point-sampling + two-wave standing-field fit (0.15-0.65% vs the Sipe closed form), sheet P_perp(x) Fourier-decomposed into orders incl. the linear-in-height surface phase, radiated as a multi-order analytic dipole sheet into the sourced 2w solve. GATES (6 new, 23 total green incl. existing shg/sfg suites): flat limit vs shg_two_step 0.76%/1.24% @ 20/35 deg; slope 2.0000; shallow lamellar gold tooth: specular c0 0.82% of flat @ h=2nm, |c+-1| ~ h (amplitude slope 0.905 -> power ~ h^1.81, derived); s-pol suppression 1.3e-10; full pipeline opens propagating m=+-1 SH channels end-to-end. FLAGS: full-FEM +-1 power carries linear SH re-diffraction off the 500 nm tooth (real, non-perturbative at resolvable h) so the trend gate is on sheet_order_power; _ensure_bloch_dirs pre-caches Bloch idnrs (thin patterned layers misclassify at oblique; grating must be an INTERIOR 2D tooth, not a full-y stripe); Dadap cylinder needs a conformal sheet (radial normal), precluded by the HCurl limitation) |
| 5.4 2-D coupled-HDM stabilization | SOLVED (hydro_fem.py scalar-longitudinal-potential reformulation: psi := div J turns the indefinite vector-J block (essential spectrum accumulating at Omega = beta^2 k^2 over the mesh's longitudinal wavenumbers -- refinement makes it WORSE) into a scalar Helmholtz (P) beta^2 lap psi + Omega psi = i w eps0 wp^2 div E whose wavenumber == nonlocal_tmm.kL_squared, with the transverse free response folded ANALYTICALLY into local Drude eps_T and J recovered pointwise; E in H(curl) + psi in H1(metal), BOTH stiff terms integrated by parts; KEY: the ABC J.n = 0 makes the two surface terms cancel -- a NATURAL condition, no Dirichlet psi; psi discretized order+1 to resolve the 0.1-0.2 nm screening layer; symmetric rescaling -> complex-symmetric O(1e-3..1) matrix. GATES (15 passed): the exact old blow-up case (dimer R=15 gap=3, 600 nm) now bounded enh=5.97 with energy residual 5.8e-4; Raza cylinder blueshift closed form (derived in-module, exact-vs-asymptotic 2-4%) measured 0.94-1.03x pred with the 1/R trend; gap saturation local/hydro strictly monotone 1.0035->1.0178 (12->2 nm); beta->0 local limit machine-exact; all 11 pre-existing gates unchanged; HydroFEMUnstable kept as a live safety net for unresolvable meshes. FLAG: gap=2 nm saturation magnitude (~1.8%) is direction/monotonicity-converged, not absolute-magnitude-converged at h_metal=4. WAVE-2 VERIFIED: the psi reduction proven EXACT (recovered J satisfies the original vector equation identically; div J == psi identically), the ABC surface-term cancellation is pointwise Omega (J.n) w -- curvature-independent, and structural at the discrete level (no surface term assembled, no normal evaluated); Raza closed form re-derived blind, exact match; robustness (gamma x10, eps_inf=4, beta/3) bounded and convergent; + a refinement-CONVERGENCE gate added (the anti-signature of the retired vector-J form, whose failure mode was refinement making it worse)) |
| 5.5 AAA rational pole extraction | SHIPPED + WAVE-2 FIX (optics/aaa_poles.py: aaa (Loewner SVD + arrowhead eigenproblem), find_resonances, sweep_and_extract, q_from_pole; real data -> conjugate mirror poles, keep Im<0; 13 gates incl. LIVE lumenairy GMR sweep: AAA Q = 342.93 vs fano_fit 343.20 (0.1%) with NO upstream lumenairy changes. VERIFICATION KILL FIXED: the original GLOBAL-relative residue floor (1e-3 of max residue) false-killed genuinely weak REAL poles (residue is the wrong Froissart proxy -- weak poles and Froissart doublets both have small residues); replaced with the ACTUAL Froissart signature, pole-zero coincidence (nearest AAA zero within froissart_frac*|Im|, a 13-order-of-magnitude discriminator: genuine >= ~0.5, doublets ~1e-13), residue floor demoted to a LOCAL junk guard 1e-6. Also: clean-data Q is EXACT at any window (analytic continuation, no bias); noise VARIANCE blows up below span ~3x FWHM -> RuntimeWarning when a resonance's FWHM > half the sweep span (broad_warn_frac), envelope documented; gain poles (Im>0) return [] not a wrong-sign Q) |
| 5.6 real-BIC capstone | SHIPPED (validation/bic_capstone.py + tests/test_bic_capstone.py: 700 nm-period suspended Si grating (fill 0.50, t 500 nm), TE 2nd folded band, symmetry-protected BIC at ~1387 nm; Q(theta) = 22226/5557/2468/1391 at 1-4 deg, exponent -2.000 (quasi-BIC law), \|Im omega\| ~ theta^1.993, far-field vortex q = +1 with off-BIC q = 0 control; 5 tests, 5.5 s) |
| 5.7 hot-carrier lattice extension | SHIPPED (HotCarrierParams.c_l_j_m3_k opt-in finite lattice heat capacity + g_sub_w_m3_k substrate out-coupling (0 = adiabatic); per-cell U_l with dU_l/dt = +G(T_e-T_l) - g_sub(T_l-T_l0) -- the SAME per-step coupling array leaves the electrons and enters the lattice, so closure is MACHINE-EXACT (rel_err 3e-15; absorbed == U_e + U_l + substrate outflow, all reservoirs non-trivial); c_l=None bit-identical to the fixed-bath tier; c_l=1e12 matches fixed-bath T_e to 1.2e-8 (limit contract); small-c_l physics: lattice heats 44.8 K and the T_e relaxation tail sits strictly ABOVE fixed-bath (+8.6 K min); 0-D oracle = carrier_heating.two_temperature_response which ALREADY integrates the lattice ODE (relTe 1e-3 / relTl 8e-5 with matched coefficients); numpy kernel only, numba/jax raise as in 2.1; WAVE-2 VERIFIED: closure survives alpha_abs != 1 + hot g_sub + 90% post-pulse tail at ~3e-15; mixed-layer fixed-bath cells byte-identical; slowed relaxation is pure reduced-driving-force (peak T_e coincides to 3e-6; full curves match an independent RK4 2-ODE to 7e-4); 20 passed in tests/test_fdtd_hot_carrier.py) |
| VERIFICATION WAVE 2 (2026-07-20, Phases 4-5) | 5 refuters, all reports in. ONE code kill: aaa_poles Froissart residue floor false-killed weak real poles (FIXED, pole-zero coincidence filter -- see 5.5). ONE gate kill: shg_grating s-pol E_perp threshold leaned on square-mesh symmetry (FIXED, parametrized over non-symmetric meshes; the physical s/p POWER suppression < 1e-4 is mesh-robust). Hardenings: hydro_fdtd phase-coherent SH subtraction + hydro_fem refinement-convergence gate + AAA narrow-window warning + harmonics per-band-guard scope comment. DEEP PHYSICS HELD EVERYWHERE: CWE signs re-derived blind (1e-14), d_eff/degeneracy factors absolute-pinned, Helt-Sipe 1/(2 pi) re-derived (hbar cancels, 1e-16), Schmidt vs Mehler closed form (1e-15), no-net-gain mechanism confirmed by an independent SVEA ODE (disabling ONLY the SFG term flips seed loss to gain), RS-SFG sheet re-derived blind (0.0e0) + fresh dispersive-metal absolute gate (0.58%), psi-reduction proven exact + ABC cancellation curvature-independent, discrete longitudinal dispersion clean 2nd-order 3-way, theta^-2 exponent -2.000 +/- 0.005 with lossless Q -> 358k at 0.25 deg (no saturation), lattice closure ~3e-15 under stress. ~17 new permanent gates | 
