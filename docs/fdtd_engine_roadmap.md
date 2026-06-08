# FDTD engine extensions -- status & roadmap

This note tracks the FDTD (time-domain optical) engine extensions requested as the "horizon items" after
the 2D/3D FDTD + topology-optimization arc. Each item is one of: **shipped** (implemented + validated
against an independent oracle), **deferred** (a concrete implementation spec exists; not yet built), or
**blocked** (an environment/dependency limitation outside the code).

The convention throughout: `exp(-i omega t)`, SI units, `Im(eps) > 0` = loss; every shipped item is
validated against an independent oracle (coherent TMM / Airy / analytic reduction), never energy-closure
alone.

## Shipped

### Non-vacuum semi-infinite end media (`n_super` / `n_sub`)
`solve_fdtd_2d` / `solve_fdtd_3d` accept lossless real `n_super` / `n_sub` (default 1 = vacuum,
byte-identical). The z-pads are filled with `n_super^2` / `n_sub^2`, the CFS-CPML conductivity is
impedance-matched per end (`sig_max` scaled by `n`), and the incident reference is a homogeneous-
superstrate run so the reflection subtraction is exact in the incidence medium. `T0` carries the
`n_sub/n_super` Snell power-flux ratio; the Poynting `R_flux`/`T_flux` already carry `n` through H.
The seam (`make_fdtd_optical_solver`, `fdtd_sweep_spectrum`) accepts lossless non-vacuum for uniform
stacks (structured + non-vacuum is deferred -- the rasterizer rebuilds the eps grid and would drop the
pads); a lossy (complex) end medium routes to FEM/TMM.
Oracle: `validation/fdtd_nonvacuum_vs_tmm.py` -- three-medium Airy TMM + lossless energy, 2D & 3D,
`max|dR0|,|dT0| ~ 1e-3`, `|R_flux+T_flux-1| ~ 1e-5`.

### Drude + Lorentz dispersion
`FDTDLayer` gains a Lorentz pole (`lorentz_w0_rad_s`, `lorentz_gamma_rad_s`, `lorentz_delta_eps`); the
2D-TE kernels (numpy / numba / jax) integrate it via the central-difference Lorentz ADE (a second
polarization `PL`), so `eps(w) = eps_inf - wp^2/(w^2 + i w gd) + d_eps w0^2/(w0^2 - w^2 - i w gl)` runs
natively across the band -- a bound-electron / interband resonance the bare Drude cannot represent.
With `d_eps=0` the path is byte-identical (`lor=None`). `fit_drude_lorentz` (seam) fits both poles to
sampled `eps(lambda)` with a scaled, multi-start least-squares (robust to the resonance overshoot).
3D carries a guard (the Lorentz ADE is 2D-only for now).
Oracle: `validation/fdtd_drude_lorentz_vs_tmm.py` -- dispersive coherent TMM (same `eps(w)`), pure-Lorentz
isolation, the fit, and cross-backend (numba == numpy) consistency.

### Combined contrast x bandwidth FOM (design study)
`validation/modulator_design_space.py` -- sweeping the gate-oxide thickness couples both modulator specs
(thinner oxide -> higher C -> more ENZ contrast but lower switching bandwidth). The study shows the two
anti-correlate monotonically (a gain-bandwidth-like trade-off, near-invariant product) so the deliverable
is a constrained design point (max contrast subject to a bandwidth floor), not a magic interior optimum.

### Real DEVSIM n(z) -> topology optimizer
`validation/fdtd_devsim_topology_design.py` -- solves the Park 2-D drift-diffusion metasurface at two gate
biases, extracts the genuine ITO accumulation `n(z)` at each, builds the graded free-carrier-ENZ FDTD
layers from those REAL profiles, and runs the jax.grad topology optimizer to shape the resonator that
maximises the actual device's reflection contrast -- closing the device->design loop with no synthetic
stand-in.

## Deferred (spec ready)

### Per-cell tensor / magneto-optic eps
- **Diagonal anisotropic (uniaxial / birefringent), tractable:** give the 3D kernels a per-component
  `eps_xx / eps_yy / eps_zz` (each E-component uses its own, no coupling); validate by birefringence
  (an x-pol vs y-pol source against an anisotropic-TMM `n_o` / `n_e`). A clean additive change to the
  three 3D kernels + an x-pol source option.
- **Off-diagonal / gyrotropic (magneto-optic Faraday), the valuable case:** this is what the FEM backend
  CANNOT do (its off-diagonal assembly is blocked, see below), so an FDTD path would unblock gyrotropic
  optics. IMPORTANT correction to the design spec: the proposed "E = inv(eps) @ D with a complex
  pre-inverted `eps`" does NOT work for a real-time FDTD -- an imaginary off-diagonal `i*g` is a
  frequency-domain stand-in for a TIME-DERIVATIVE coupling, so a complex algebraic inverse on real fields
  yields complex (unphysical) E. The correct path is a magneto-optic auxiliary-differential-equation
  (the gyration as a `g * dE_perp/dt` antisymmetric polarization-current coupling), then validate vs the
  circular-eigenmode (`n_pm = sqrt(eps +/- g)`) Faraday-rotation oracle (the `magneto_optic_faraday.py`
  pattern). A genuine kernel effort; not the spec's one-liner.

### Oblique Bloch incidence (2D first)
Split-field (`Ey_cos`, `Ey_sin`) Bloch-phase periodic BC in the 2D-TE kernel (numpy + numba), a plane-wave
oblique source, and an m=0-order spatial-DFT probe extraction with `k_z(k_par)` de-embedding. Validate vs
coherent TMM at angles {0,15,30,45} deg (the laterally-uniform slab must reduce to TMM). The full spec
(state, update equations, plumbing, gates) is in the design workflow output. Note the FEM backend already
provides oblique/conical incidence, so this is the time-domain complement, not a gap.

## Blocked (environment)

### Numba-CUDA GPU kernel
A fused `numba.cuda` GPU kernel is the planned large-3D fast path, but `numba.cuda.is_available()` is
`False` here (no CUDA toolkit installed). The hot loop is already a swappable kernel boundary, so this
drops in once a CUDA toolkit is present. The CuPy backend already runs the vectorized loop on a device
when one is available; on Windows a JAX-GPU build is WSL2-only.

### FEM off-diagonal (gyrotropic / tilted-anisotropic) tensor -- "B1"
NGSolve 6.2.2604 mis-assembles a complex periodic HCurl bilinear form with a genuine off-diagonal tensor
coefficient (a spurious mixed-component coupling that only the int-0 sparse pruning neutralizes); the
assembler raises rather than return a wrong answer. This is an NGSolve-version assembly defect, not a
DynaMeta bug. The fix is an NGSolve upgrade (authorized, pending a de-risk probe on a newer build) or the
FDTD magneto-optic ADE above as the alternative path to gyrotropic optical response.
