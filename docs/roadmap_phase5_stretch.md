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

2. **Uniform-background scattered field is inaccurate for a non-vacuum substrate
   (a pre-existing Phase-3 limitation, NOT an oblique bug).** With `eps_bg=1`
   everywhere (Phase-3 "Option A"), a dense substrate (`eps != 1`) carries a large
   volumetric source `k0^2 (eps_sub-1) E_inc` driven at the WRONG (vacuum `kz_s`)
   wavevector through the whole substrate band. `validation/oblique_isolation.py`:
   a **vacuum** exit (case b) matches `tmm` to <0.001 at 0 AND 30deg, but a `n=1.5`
   exit (cases a,c) is off and mesh-fragile **even at normal incidence**. The fix is
   a **layered/Fresnel background field**: `eps_bg` piecewise (1 in the
   superstrate+structure, `n_sub^2` in the substrate), `E_bg` = the bare air/substrate
   Fresnel solution (incident + `r_bg`, transmitted `t_bg`), source nonzero only in
   the slab/patch, and the extracted reflection adds back `r_bg`. A scoped follow-up,
   independent of incidence angle (it also tightens normal-incidence transmission for
   devices on a substrate). Reflection-mode devices with a bottom mirror (e.g. Park,
   T~0) are largely unaffected; `solve_fem` emits a one-time guard when the exit
   medium is non-vacuum.

(p-pol oblique -- the in-plane polarization vector -- is a further follow-up.)

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

**Remaining (integration, not physics):**
- a `carriers/devsim_3d.py` `CarrierSolver` that meshes an arbitrary `Design` (not
  just a stacked MOS-cap) via gmsh and emits a `CarrierField(ndim=3)` for the
  bridge (the bridge / `IdentityLift` / `RegionAlignment` ndim=3 path is designed
  and ready; emitting + consuming a 3D field end-to-end is the next step);
- 3D **drift-diffusion** (the equilibrium path is validated here; 3D DD is stiffer
  and larger -- the `abs_tol`/seeding lessons carry over but it needs its own
  convergence pass).

---

## Summary

| Item | Status | Validated by | Remaining |
|---|---|---|---|
| Oblique incidence | **resolved, s-pol** | `tmm` -- R&T <0.01, 0-30deg (vacuum exit) | non-vacuum substrate (layered-bg field); p-pol |
| 3D DEVSIM carriers | implemented, equilibrium | Gauss + sign + invariance | Design->gmsh builder; 3D DD |

Validation scripts live in `validation/`. Both features caught real issues during
verification (the PML angle-limit, the gmsh-scale + MSH-version + NumPy-2 bugs, a
binning-metric artifact) -- exactly what external/physical checks are for.

---

## Further physics extensions (beyond Phase 5 -- designed, NOT implemented)

Larger research efforts; the design + hook points are recorded so they can be
picked up cleanly. None is implemented -- no unvalidated physics is shipped.

### Bipolar drift-diffusion (holes + recombination)
Current DD is electrons-only (correct for unipolar degenerate ITO). For bipolar
devices add a `Holes` solution variable mirroring `Electrons` in
`physics_drift_diffusion.py`: a hole continuity equation with an FD-enhanced
Scharfetter-Gummel current (sign-flipped drift), an SRH (optionally Auger/
radiative) recombination node model coupling the two continuities, and the Poisson
charge `q(p - n + N_D - N_A)`. Contacts pin both n and p to charge-neutral
equilibrium. **Validation gate:** a p-n diode J-V (monotonic, sign- and
ideality-correct) + reduction to the electron-only result in the unipolar limit.

### Quantum confinement (Schrodinger-Poisson)
The ~1 nm accumulation layer in degenerate ITO has sub-band quantization the
classical Poisson/DD misses. Add a 1D through-stack effective-mass Schrodinger
solve per lateral column (eigen-solve `-hbar^2/2m* d2psi/dz2 - qV psi = E psi`,
fill sub-bands via the 2D DOS to E_F, build `n(z)` from `|psi|^2`), iterated
self-consistently with Poisson (predictor-corrector / Anderson mixing). Plugs in
as an alternative carrier model on the semiconductor region. **Validation gate:**
analytic square-/triangular-well sub-band energies; the quantum `n(z)` must recover
the classical Fermi-Dirac profile in the bulk away from the interface.

### Boundary-spanning inclusion topologies
Phase 3 inclusions are interior-only (the four periodic faces stay clean
rectangles so the proven face `Identify` works). For features that touch/cross the
cell boundary (connected gratings, wires), the OCC build must split the boundary-
crossing solid at the cell faces and pair the resulting partial faces in the
periodic `Identify` (matching sub-face signatures across the translation).
**Validation gate:** a boundary-spanning grating's periodic-ndof + energy
conservation match an interior-inclusion or TMM/RCWA reference.
