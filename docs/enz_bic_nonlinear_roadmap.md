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
