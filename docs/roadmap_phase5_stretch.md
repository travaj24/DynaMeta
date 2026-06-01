# Phase 5 stretch items: implementation + validation status

Phase 5's **core** deliverable -- the pluggable seams (`CarrierSolver`,
`OpticalGeometryBuilder`) with worked BYO examples and docs -- is **done**
(`examples/byo_carrier_solver.py`, `examples/byo_optical_geometry.py`,
`docs/pluggable.md`). Both **stretch** items below are now **implemented and
externally/physically validated**, each with a clearly characterized remaining
limitation (documented honestly rather than hidden).

---

## Stretch 1 -- oblique incidence  [RESOLVED; tmm-validated 0-30deg, vacuum exit]

**Implemented** in `optics/solver.py` + `geometry/specs.py`:
- transverse wavevector `kx = k0 sin(theta)` (x-z plane, vacuum/air incidence medium);
- physical-field incident plane wave `exp(i kx x - i kz_s z)`, `kz_s = k0 cos(theta)`;
- **Floquet-Bloch quasi-periodic space** `ng.Periodic(HCurl, phase=...)` solving for
  the PHYSICAL field with the GENUINE curl (so the standard stretched-coordinate
  HalfSpace z-PML, `alpha=1j`, transforms it exactly);
- **demodulated R/T fits** (multiply by `exp(-i kx x)`, fit `exp(+-i kz z)` with the
  medium-correct `kz`), Poynting-correct `T = |t|^2 Re(kz_sub)/Re(kz_sup_med)`;
- `OpticalSpec` allows oblique for **s-pol** (`polarization='y'`); p-pol oblique raises.

**Validation (external, vs the `tmm` library)** -- layered slab air / n=2 (250nm) /
**air**, s-pol (`validation/oblique_vs_tmm.py`):

| theta | R fem / tmm | T fem / tmm | energy R+T |
|---|---|---|---|
| 0deg  | **0.198 / 0.198** | 0.795 / 0.802 | 0.993 |
| 15deg | **0.224 / 0.220** | 0.768 / 0.780 | 0.992 |
| 30deg | **0.301 / 0.292** | 0.711 / 0.708 | 1.012 |

R AND T match `tmm` to <0.01 through 30deg and energy conserves -- the full oblique
machinery (Bloch phase, incidence, PML, R/T extraction, T-weighting) is correct.

### The two bugs (this is the interesting part)

The earlier "energy grows with angle" symptom was **two independent bugs**, neither
the PML nor a phase sign:

1. **The Bloch phase was never enforced (the dominant, silent bug).** `ng.Periodic`
   keys its `phase` list per identification in **idnr order**, and netgen does NOT
   number the OCC face identifications in creation order -- for a glued multi-layer
   stack the x- and y-face idnrs come out **interleaved** (`x,y,x,y,...`, one x/y pair
   per z-layer), verified on-machine. The old `[exp(i kx Px)]*n_px + [1]*n_py` list
   therefore put `phase=1` on the actual x-faces -> the cell was **plain-periodic in
   x** -> the solver returned the **normal-incidence field at every angle**. This is
   a vicious silent failure: normal incidence still validates (phase=1 is correct
   there), and a layered field is naturally x-invariant so nothing looked wrong.
   *Diagnosis trail:* the reflected wave's z-phase slope was `k0`, not `k0 cos(theta)`,
   and the demodulated field was x-invariant (kx=0). **Fix:** `solver._detect_bloch_dirs`
   resolves each idnr's axis by toggling a marker phase on it alone and measuring
   whether the x- or y-boundary moves, then asserts the recovered x/y counts (anti-
   silent-failure). With the correct per-idnr phase the field is genuinely oblique
   (kx + `kz_s` both verified) and reflection/transmission match `tmm`.
   - NB: plain periodicity (`phase=1`, normal incidence + the carrier/patch cases)
     WAS always enforced -- only the nontrivial Bloch phase was mis-mapped. Prior
     normal-incidence results stand. `ng.Periodic` keeps `ndof` unchanged and SLAVES
     the minion DOFs (so an ndof check is NOT a periodicity test; a boundary
     enforcement probe is).

2. **Uniform-background scattered field was inaccurate for a non-vacuum substrate
   (a pre-existing Phase-3 issue, NOT an oblique bug) -- NOW FIXED.** With `eps_bg=1`
   everywhere (Phase-3 "Option A"), a dense substrate (`eps != 1`) carried a large
   volumetric source driven at the WRONG (vacuum) wavevector through the whole
   substrate band (mesh-fragile even at normal: case (c) gave R=0.26 vs `tmm` 0.12).
   **Fix (implemented in `optics/solver.py`):** a **layered / Fresnel two-region
   background**. `eps_bg(z)` is piecewise -- superstrate medium above the substrate-top
   interface `z_int`, substrate medium below -- and `E_bg` is the analytic bare
   air/substrate Fresnel field (incident + background reflection `R0` above,
   transmission `T0` below). The scattered source `k0^2 (eps - eps_bg) E_bg` is then
   nonzero ONLY in the structure layers; the substrate carries no spurious source. R/T
   add back the analytic `R0`/`T0`. Reduces exactly to the plain incident wave when
   `n_sub == n_super == 1` (`R0=0, T0=1`). **Validated** (`validation/oblique_isolation.py`,
   all PASS): a single air/1.5 interface, a vacuum-exit slab, and a **dense n=1.5
   substrate** all match `tmm` R and T to **<0.003** at 0 AND 30deg, energy-conserving.
   This also tightens normal-incidence transmission for any device on a substrate.

3. **p-polarization (TM) oblique -- IMPLEMENTED.** p-pol has E in the x-z plane (Ex,
   Ez). The background reflection/transmission E-vector amplitudes come from solving the
   physical interface BCs (tangential Ex + Hy continuity) NUMERICALLY at z_int (no
   Fresnel sign-convention ambiguity; Hy ~ Ex*eps/qz), and R/T are extracted from the
   reconstructed TOTAL field's tangential Ex with the p-pol z-flux factor Sz ~
   |Ex|^2 eps/kz: `R=|Ex_up/Ex_down|^2`, `T=|Ex_t/Ex_i|^2 Re((eps_sub/eps_sup)(kz_s/kz_sub))`.
   **Validated** (`validation/oblique_ppol_vs_tmm.py`, PASS): vs `tmm('p')` for a
   vacuum-exit AND a dense (n=1.5) substrate slab, R and T match to **<0.001** at
   0/15/30deg, energy-conserving (R+T=1.0000). `OpticalSpec.polarization='p'`.

4. **Conical incidence (azimuth phi != 0) -- IMPLEMENTED (s-pol).** `OpticalSpec.azimuth_deg`.
   kx=k0 sin(th) cos(phi), ky=k0 sin(th) sin(phi); a 2D Bloch phase (exp(i kx Px) on x-faces
   AND exp(i ky Py) on y-faces, via the per-idnr detection); the in-plane s-pol direction
   Es=(-sin phi, cos phi, 0); and a 2D-demodulated, projected R/T extraction. **Validated**
   (`validation/oblique_conical_vs_tmm.py`, PASS): an isotropic layered stack is azimuthally
   symmetric, so at theta=30deg the R/T are phi-INVARIANT (spread 0.0000 over phi=0/30/60/90)
   and match `tmm(theta,'s')` to <0.001. p-pol conical is the remaining follow-up.

---

## Stretch 2 -- native 3D DEVSIM carriers  [IMPLEMENTED (equilibrium); validated]

**Implemented + validated** (`validation/carriers_3d.py`): a gmsh-meshed 3D MOS-cap
(semiconductor + gate oxide, gate on top, body on bottom) solved with the EXISTING
dimension-agnostic equilibrium physics (`physics_equilibrium`) on the 3D mesh --
the node/edge models attach to a 3D region unchanged.

Two build gotchas found + fixed:
- gmsh's OCC kernel cannot build at 1e-9-metre absolute scale -> build geometry in
  **nm**, emit the mesh scaled to **metres** via `Mesh.ScalingFactor`;
- DEVSIM reads **MSH 2.2** (gmsh 4.x defaults to 4.1).

**Validation (solver-independent physics):**
- converges cleanly (RelError -> ~1e-8);
- **sign-correct + monotonic**: +1V -> n_top/n_bg = 1.26 (accumulation), -1V -> 0.75
  (depletion);
- **Gauss's law**: accumulated sheet charge `q*Int(n-n_bg)dz` matches the oxide
  displacement `eps_ox*eps0*(Vg-V_surf)/t_ox` to ~12% (ratio 0.88/0.89), and the
  ratio **tightened from 0.65 (coarse) to 0.88 with interface refinement**,
  confirming the residual is mesh resolution of the ~1nm accumulation layer, not
  physics;
- **lateral (x,y) invariance**: machine-precision (~1e-13) at zero bias; ~1% at
  bias (true lateral variation, after detrending the within-z-bin accumulation
  gradient).

So the hard question -- does the carrier physics solve correctly on a true 3D mesh
-- is answered **yes**.

**End-to-end (DONE):** `carriers/devsim_3d.py` `Devsim3DEquilibrium` emits a native
`CarrierField(ndim=3)`; `core/bridge.py` has the ndim=3 branch (real x/y/z axes,
IdentityLift, no lift synthesis) validated by `validation/bridge_3d_field.py` (3D
field -> `assemble_eps` -> `EpsField`, gate accumulation lowers Re(eps) 1.571->0.59
toward ENZ). `validation/pipeline_3d_end_to_end.py` closes the chain: 3D carriers ->
bridge -> `assemble_eps_cf` -> `solve_fem` runs and the gate bias modulates the optical
eps the solver consumes (gate-side Re(eps) 1.57 at 0V -> 0.71 at +1V). NB: a measurable
optical dR needs a RESONANT geometry -- a bare 12 nm ITO layer in air is optically
negligible at 1300 nm (dR~1e-5); the ENZ shift only converts to dR in a patch/cavity.

**3D drift-diffusion (DONE).** `Stacked3DSpec(physics='drift_diffusion')` attaches the
FD-enhanced Scharfetter-Gummel electron continuity + Poisson (`physics_drift_diffusion`,
dimension-agnostic) on the 3D semi region, with the body contact pinning the electron
QFL, an abs_tol scaled to n_bg (the `_dc_abs_tol` lesson), and a zero-bias-seed ->
gate-ramp staged Newton. Validated (`validation/carriers_3d_dd.py`, PASS): converges
(RelError ~1e-11), sign-correct (+1V accumulates, -1V depletes to 0.67), and REDUCES to
the 3D equilibrium accumulation to **0.8%** at +1V (the zero-current MOS-cap limit).
The MOS-cap carries no current, so transport is tested separately by a 3D RESISTOR
(`validation/carriers_3d_resistor.py`, PASS): an ITO bar with ohmic end contacts obeys
Ohm's law `I = V*sigma*A/L` to mesh accuracy (rel-diff 0.20->0.14 on refinement). That
test exposed + fixed a real bug: `setup_contact_ohmic_dd` omitted `edge_current_model`,
so `get_contact_current` read 0 (now added; the Dirichlet pinning is unchanged).

**Design-driven 3D builder + lateral gate patch (DONE).** `Stacked3DSpec.gate_patch_frac`
< 1 imprints a centered gate-patch onto the oxide top (the rest a free surface), so the
classical 3D solve accumulates UNDER the patch and stays near bulk in the gap -- the
laterally-VARYING (non-separable) profile the 2D+symmetrization path cannot capture
(`validation/carriers_3d_patch.py`, PASS: under-patch n=1.33 vs gap 1.02 n_bg; min Re(eps)
0.70 under patch vs 1.53 gap). `Stacked3DSpec.from_design(design)` derives the stacked
spec (semiconductor + gate-dielectric layers, period, gate footprint -> patch frac) AND
names the emitted CarrierField region after the Design's semiconductor layer, so it
matches the optics builder's alignment source_region -> `run_pipeline` with NO hand-built
alignment (`validation/carriers_3d_from_design.py`, PASS: derived spec + region match).

**`from_design` constraints (now enforced, not silent):** it handles a MULTI-dielectric
stack (e.g. the full Park mirror/Al2O3/HfO2/ITO/HfO2/patch -- it walks to the gate-side
dielectric; `validation/from_design_multidielectric.py` PASS), selects the gate by
`electrode.role=='biased'`, and threads the gate/body electrode NAMES into the bias lookup.
It requires a SINGLE semiconductor layer, a SQUARE cell, and an (approximately) centered
square gate footprint -- each now RAISES/WARNS rather than silently mis-modeling (audit
F1-F4). True lateral material inclusions in the semiconductor still need a manual
`Stacked3DSpec` or a further general OCC builder.

---

## Summary

| Item | Status | Validated by | Remaining |
|---|---|---|---|
| Oblique incidence | **resolved, s+p-pol, +conical** | `tmm` s & p, 0-30deg, vacuum/dense/lossy; conical phi-invariant + tmm | p-pol conical |
| 3D DEVSIM carriers | equilibrium + DD + lateral patch + Design-driven | Gauss/sign/invariance, DD reduces-to-eq 0.8%, gate-patch lateral accumulation, from_design run_pipeline-compatible | multi-dielectric stack / arbitrary OCC inclusions |
| Quantum confinement (S-P) | **implemented + CarrierSolver** | analytic wells ~1e-5; ITO bulk-recover + accumulation + ENZ via bridge; nonparabolic reachable via carrier | fully self-consistent nonparabolic solve; Neumann body BC |
| Bipolar DD (holes+SRH) | **implemented** | 1D Si diode J-V (rectify 1.8e10, n=1.20) | wire into 2D builder |
| Boundary-spanning inclusions | **resolved** | grating(1D edge) + disk(2D corner) translation-invariance |dR|<=1e-4 + energy R+T~1 | -- |
| Inclusion shapes | **rect/circle/ellipse/polygon** | ellipse + hexagon build/solve + energy R+T~1 (validation/inclusion_shapes.py) | -- |

Validation scripts live in `validation/`. Both features caught real issues during
verification (the PML angle-limit, the gmsh-scale + MSH-version + NumPy-2 bugs, a
binning-metric artifact) -- exactly what external/physical checks are for.

---

## Further physics extensions (beyond Phase 5)

Larger research efforts. Bipolar DD, Schrodinger-Poisson quantum confinement, and
boundary-spanning inclusions are now all IMPLEMENTED + validated (below).
No unvalidated physics is shipped.

### Bipolar drift-diffusion (holes + recombination)  [IMPLEMENTED; diode-validated]
`carriers/physics_bipolar_dd.py` -- opt-in 3-variable (Potential, Electrons, Holes)
DD in SI, mirroring the electron-only module's FD-enhanced Scharfetter-Gummel and
reusing `physics_equilibrium`/`eq_registry`. Hole current = electron expression with
`q->-q` and the `vdiff` drift term on the `@n0` node (+ a parallel FD g-factor); SRH
`USRH=(np-n_i^2)/(taup(n+n1)+taun(p+p1))` wired as `Gn=-qUSRH`/`Gp=+qUSRH` into the two
continuities; bipolar Poisson charge `-q(p-n+NetDoping)`; charge-neutral ohmic
contacts (n0/p0 + built-in offset). Staged solve: potential-only pre-solve ->
equilibrium carrier seed -> coupled Newton -> bias ramp; the decisive convergence
trick is SI abs_tol scaling (continuity residual ~1e24, so abs=1e18/rel=1e-6 for the
coupled system -- the same lesson as `_dc_abs_tol`).

**Validation (`validation/bipolar_diode.py`, 1D Si p-n diode, both gates PASS):**
- Vbi = 0.9524 V (pre-solve span matches analytic exactly); equilibrium minority =
  `n_i^2/N` (mass-action correct);
- forward J-V rises ~exponentially, reverse saturates (~6e-8 A/m^2 opposite sign):
  monotonic + rectifying, ratio **1.8e10**, ideality **1.20**;
- minority injection ratio **1.6e11** at +0.7 V (the bipolar signature);
- FD path reduces to Boltzmann SG in the non-degenerate limit (FD-on vs off **0.58%**).

Remaining: wire into the 2D `LayeredDevsimBuilder` (extend the Gummel `CARRIER_EQS`/
`_TRACK_VARS` for `Holes`); the pre-existing electron-only "gated-cap DD does not
converge" note is orthogonal (weak 2-node lateral ITO contacts, not the formulation).
NB: unipolar degenerate ITO is correctly electrons-only, so bipolar DD is a generality
feature, not Park-critical.

### Quantum confinement (Schrodinger-Poisson)  [IMPLEMENTED + coupled as CarrierSolver]
`carriers/schrodinger_poisson.py` -- a 1D effective-mass `SchrodingerPoisson1D`:
BenDaniel-Duke tridiagonal Schrodinger (mass at half-nodes, Dirichlet ends, unbound
states discarded), DEGENERATE 2D sub-band filling
`n(z)=sum_i (g_s g_v m* kT/2pi hbar^2) ln(1+exp((E_F-E_i)/kT)) |psi_i|^2` (ITO g_v=1;
overflow-safe `ln(1+e^x)`), and a self-consistent Poisson loop via the **Trellakis
predictor-corrector** (a nonlinear-Poisson Newton inner solve with the exact Fermi-
function Jacobian -- robust where naive Picard sloshes).

**Validation (`validation/schrodinger_poisson.py`, all PASS):**
- infinite square well `E_n = n^2 pi^2 hbar^2/2mL^2`: rel **1e-6 to 2e-5** (n=1..4);
- triangular well `U=qFz` vs the Airy zeros `E_n=|a_n|(qF)^(2/3)(hbar^2/2m)^(1/3)`:
  rel **2e-5 to 7e-5** (n=1..4);
- degenerate filling: `Int n(z) dz` == `sum_i n_s,i` to **1e-7**;
- self-consistent ITO accumulation (0.5 V, 20 nm): converges, forms a ~nm gate-side
  accumulation layer (bound ground state, E0~-0.29 eV).

**Coupled as a CarrierSolver (DONE).** `carriers/sp_carrier.py` `SchrodingerPoissonCarrier`
implements the `CarrierSolver` Protocol: E_F from the bulk degenerate relation
`E_F-E_c=(hbar^2/2m*)(3 pi^2 n_bg)^(2/3)`, self-consistent SP at the gate-induced surface
potential, and a `CarrierField(ndim=3)` (the through-stack quantum profile broadcast over
the cell) the bridge turns into eps. **KEY finding:** for a DEGENERATE-bulk slab the
sub-band rejection (`bound_tol`) must be DISABLED (`bound_tol=1e9`, slab mode) -- it is
right for an isolated well but discards the high sub-bands that carry the bulk continuum,
collapsing the bulk density to ~0 (the default kept 3 of ~8 sub-bands -> 0.64 n_bg; slab
mode -> 1.00 n_bg). Validated (`validation/sp_carrier.py`, PASS): bulk recovers n_bg
(0.996); +0.3 V accumulates (peak 1.42 n_bg); the QUANTUM dead layer / charge-centroid
setback (peak ~1 nm from the interface, n->0 AT it); the bridge shows accumulation deepens
ENZ (min Re(eps) 1.15 -> 0.45). Directly informs the Park tuning: the quantum setback
offsets the sub-ENZ region ~1 nm from the oxide vs the classical (peak-at-interface) model.

**Per-lateral-column SP (DONE)** for laterally-VARYING devices: pass `surface_potential_xy
(x,y,Vg)->V` and the solver runs a 1D SP per lateral column, CACHING by psi_s value (a
~equipotential patch costs only ~2 solves). Validated (`validation/sp_per_column.py`, PASS):
a central patch at +0.4V over an ungated gap gives accumulation only under the patch
(peak 1.53 vs 1.16 n_bg) and a laterally-varying eps -- min Re(eps) **0.16 under the patch
vs 1.15 in the gap** (deep sub-ENZ only where gated). Only 2 cached solves for the step.

**Gate-oxide voltage division (DONE):** pass `oxide_thk_m` + `eps_oxide` and the gate->psi_s
map solves the series-capacitor relation `Vg = psi_s + q*N_excess(psi_s)/C_ox` by bisection
(`validation/sp_oxide_cap.py` PASS: Vg=1V -> psi_s=0.64V, accumulation 6x smaller than the
naive psi_s=Vg, self-consistent) -- the calibrated map.

**ITO band nonparabolicity (DONE in the 2D filling):** `density(..., alpha_np_per_eV=...)`
uses the Kane energy-dependent mass `m*(eps)=m*0(1+2 alpha eps)`, so the 2D sub-band sheet
density is `(g_s g_v m*0/2 pi hbar^2) Int (1+2 alpha eps) f deps`. Validated
(`validation/sp_nonparabolic.py` PASS): matches the analytic T=0 closed form
`pref0*(dE + alpha*dE^2)` to <0.1%, the DOS enhancement is exactly `1+alpha*dE`, and
`m*(E_F)/m*0 = 1.16` for ITO-like alpha=0.5/eV, dE=0.16 eV. Remaining: a fully self-
consistent nonparabolic SOLVE (the Trellakis inner Newton's a-priori density + Jacobian,
and the bulk E_F<->n calibration, must also use the nonparabolic DOS) -- the parabolic
`solve_self_consistent` is unchanged. Nonparabolicity is now ALSO reachable THROUGH the
device path: `SchrodingerPoissonCarrier(alpha_np_per_eV=...)` applies a POST-HOC
nonparabolic 2D fill on the converged (parabolic) potential (`validation/
sp_carrier_nonparabolic.py` PASS: peak-density DOS enhancement ratio 1.25 at alpha=0.5/eV,
Vg=+0.5V). The self-consistent potential + bulk E_F stay parabolic (documented), so the
fully-consistent nonparabolic solve remains the only open piece.

### Boundary-spanning inclusion topologies -- IMPLEMENTED + validated
Phase 3 inclusions were interior-only (the four periodic faces stayed clean
rectangles so the proven face `Identify` works). Features that touch/cross the cell
boundary (connected gratings, wires, corner-shared pillars) are now supported by
`LayeredOpticalBuilder._inclusion_solids_clipped`: each inclusion is intersected
with the unit cell AND unioned with its periodic translates (the 3x3 set of
+/-Px, +/-Py shifts), each also clipped to the cell. A boundary-crossing solid thus
contributes its WRAPPED piece(s) at the opposite face -- a stripe centered on x=0
becomes `[0, w/2] U [Px-w/2, Px]` (2 sub-solids); a disk on a corner becomes 4
quarter-disks. The resulting inclusion-material sub-faces on x=0/x=Px (and
y=0/y=Py) carry matching `(y,z)`/`(x,z)` signatures, so the existing
`_identify_periodic` pairs them with no change. Strictly-interior inclusions keep
only the `(0,0)` translate, so the path reduces exactly to the old build (no
regression). Each sub-solid is named the same region name -> one material.

**Validated** -- the oracle is translation invariance: a periodic structure's
specular (0th-order) R/T at normal incidence is independent of a lateral shift of
the cell origin, so a boundary-spanning inclusion must match the SAME inclusion
built interior, and a lossless subwavelength grating/array must conserve energy.
- `validation/boundary_inclusion_grating.py` (PASS): a full-y n=2.5 stripe in air,
  Px=400nm (subwavelength, lambda/Px=3.25 -> only 0th order). Centered at x=Px/2
  (interior path) vs x=0 (crosses the face): R/T match to |dR|=|dT|=0.0000 and
  R+T=1.0003 for both.
- `validation/boundary_inclusion_corner_circle.py` (PASS): the 2D diagonal-translate
  case -- a disk at the corner (0,0) splits into 4 quarter-disks vs a centered disk;
  R/T match within TOL and R+T~1.
