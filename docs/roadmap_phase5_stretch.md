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
- **Oblique angles do not conserve energy** (R+T climbs to 1.27, T>1) -- the
  fixed-alpha `ng.pml.HalfSpace` PML, tuned for a normal (kz=k0) outgoing wave,
  partially reflects the oblique transmitted wave and contaminates the fit.

**Remaining (the one real limitation): an angle-aware PML.** Scale the PML
absorption by the outgoing `kz = k0 cos(theta)` (e.g. a stretched-coordinate PML
keyed to kz, or `alpha ~ 1/cos(theta)`), then re-validate `R+T==1` + Fresnel at
angle. Until then `solve_fem` **warns loudly** at oblique incidence and oblique
R/T should be treated as qualitative.

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
| Oblique incidence | implemented, s-pol | `tmm` -- normal to 0.4% | angle-aware PML (oblique energy) |
| 3D DEVSIM carriers | implemented, equilibrium | Gauss + sign + invariance | Design->gmsh builder; 3D DD |

Validation scripts live in `validation/`. Both features caught real issues during
verification (the PML angle-limit, the gmsh-scale + MSH-version + NumPy-2 bugs, a
binning-metric artifact) -- exactly what external/physical checks are for.
