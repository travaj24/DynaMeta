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

(p-pol oblique -- the in-plane polarization vector -- is the remaining oblique follow-up.)

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

**Remaining (integration, not physics):**
- a `Devsim3DEquilibrium` that meshes an arbitrary `Design` (not just a stacked
  MOS-cap, and sharing the optics builder's lateral extent + region naming so a
  single-Design 3D run needs no hand-built alignment);
- 3D **drift-diffusion** (the equilibrium path is validated; 3D DD is stiffer and
  larger -- the `abs_tol`/seeding lessons + the new `physics_bipolar_dd.py` carry
  over but it needs its own convergence pass).

---

## Summary

| Item | Status | Validated by | Remaining |
|---|---|---|---|
| Oblique incidence | **resolved, s-pol** | `tmm` -- R&T <0.01, 0-30deg, vacuum AND dense substrate | p-pol oblique |
| 3D DEVSIM carriers | equilibrium + end-to-end | Gauss/sign/invariance + bridge ndim=3 + pipeline | general Design->gmsh builder; 3D DD |
| Quantum confinement (S-P) | **implemented** | analytic square/triangular wells (~1e-5) | couple as CarrierSolver; ITO nonparabolic m* |
| Bipolar DD (holes+SRH) | **implemented** | 1D Si diode J-V (rectify 1.8e10, n=1.20) | wire into 2D builder |
| Boundary-spanning inclusions | scoped only | -- | OCC boundary-split + paired Identify |

Validation scripts live in `validation/`. Both features caught real issues during
verification (the PML angle-limit, the gmsh-scale + MSH-version + NumPy-2 bugs, a
binning-metric artifact) -- exactly what external/physical checks are for.

---

## Further physics extensions (beyond Phase 5 -- designed, NOT implemented)

Larger research efforts; the design + hook points are recorded so they can be
picked up cleanly. None is implemented -- no unvalidated physics is shipped.

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

### Quantum confinement (Schrodinger-Poisson)  [IMPLEMENTED; analytically validated]
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

Remaining: couple as an alternative `CarrierSolver` on the semiconductor region
(per-lateral-column SP, or the device-coupled density-gradient route in the notes);
ITO band nonparabolicity (density-dependent m*) for quantitative sub-band spacing.

### Boundary-spanning inclusion topologies
Phase 3 inclusions are interior-only (the four periodic faces stay clean
rectangles so the proven face `Identify` works). For features that touch/cross the
cell boundary (connected gratings, wires), the OCC build must split the boundary-
crossing solid at the cell faces and pair the resulting partial faces in the
periodic `Identify` (matching sub-face signatures across the translation).
**Validation gate:** a boundary-spanning grating's periodic-ndof + energy
conservation match an interior-inclusion or TMM/RCWA reference.
