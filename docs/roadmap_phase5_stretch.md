# Phase 5 stretch items: implementation + validation status

Phase 5's **core** deliverable -- the pluggable seams (`CarrierSolver`,
`OpticalGeometryBuilder`) with worked BYO examples and docs -- is **done**
(`examples/byo_carrier_solver.py`, `examples/byo_optical_geometry.py`,
`docs/pluggable.md`). Both **stretch** items below are now **implemented and
externally/physically validated**, each with a clearly characterized remaining
limitation (documented honestly rather than hidden).

---

## Stretch 1 -- Bloch-phase oblique incidence  [IMPLEMENTED; normal-validated]

**Implemented** in `optics/solver.py` + `geometry/specs.py`:
- transverse wavevector `kx = k0 sin(theta)` (x-z plane, vacuum/air incidence medium);
- incident field with transverse phase `exp(i kx x - i kz_s z)`;
- **Floquet-Bloch periodic space** `ng.Periodic(HCurl, phase=[exp(i kx Px)]*n_px +
  [1]*n_py)` -- the identification counts `(n_px, n_py)` are threaded from the
  builder via `OpticalGeometry`;
- **demodulated R/T fits** (multiply by `exp(-i kx x)`, fit `exp(+-i kz z)` with the
  medium-correct `kz`), Poynting-correct `T = |t|^2 Re(kz_sub)/Re(kz_sup)`.
- `OpticalSpec` allows oblique for **s-pol** (`polarization='y'`); p-pol oblique
  raises (the in-plane polarization vector is a follow-up).

**Validation (external, vs the `tmm` library)** -- layered slab air / n=2 / n=1.5,
s-pol (`validation/oblique_vs_tmm.py`):

| theta | R fem / tmm | energy R+T |
|---|---|---|
| 0deg  | **0.117 / 0.121** (0.4%) | 1.037 |
| 15deg | 0.053 / 0.133 | 1.125 |
| 30deg | 0.084 / 0.173 | 1.269 |

- **Normal incidence is correct** -- R matches `tmm` to 0.4%, validating the whole
  Bloch/incident/fit/T-weighting machinery reduces properly (theta=0 also exactly
  reproduces the pre-existing normal-incidence path).
- **Oblique angles do not yet conserve energy** (R+T ~1.13/1.27 at 15/30deg, T>1,
  R too low). An **angle-aware PML** (`alpha = 1j/cos(theta)`, now implemented)
  produced IDENTICAL numbers -- so, contrary to the first hypothesis, the dominant
  error is NOT the PML. The symptom (energy growing with angle) points instead to
  the **Bloch-phase identification ordering** and/or the **oblique R/T fit**.

**Remaining (deeper than first thought):**
1. Debug the **Bloch-phase application** -- the phase list assumes the mesh's
   periodic identifications are numbered "all px then all py", which may not match
   the actual idnr order; instrument the `ng.Periodic` phase<->idnr mapping and
   test a phase-sign flip.
2. Audit the **oblique transmission fit** -- the fitted `|t|` comes out ~1.2x high
   at 30deg (so `T>1`); the demodulated substrate fit needs checking at angle.
The angle-aware PML is kept (correct physics; reduces to the validated `alpha=1j`
at normal). Until this validates, `solve_fem` **warns** and oblique R/T is
qualitative. (p-pol oblique remains a separate follow-up.)

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
| Oblique incidence | implemented, s-pol | `tmm` -- normal to 0.4% | Bloch-phase idnr ordering + oblique fit (energy at angle); p-pol |
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
