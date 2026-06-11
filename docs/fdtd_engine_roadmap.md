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
AND structured stacks (shipped 29a580e, 2026-06-08; `validation/fdtd_structured_nonvacuum.py`); a lossy
(complex) end medium still routes to FEM/TMM.
Oracle: `validation/fdtd_nonvacuum_vs_tmm.py` -- three-medium Airy TMM + lossless energy, 2D & 3D,
`max|dR0|,|dT0| ~ 1e-3`, `|R_flux+T_flux-1| ~ 1e-5`.

### Drude + Lorentz dispersion
`FDTDLayer` gains a Lorentz pole (`lorentz_w0_rad_s`, `lorentz_gamma_rad_s`, `lorentz_delta_eps`); the
2D-TE kernels (numpy / numba / jax) integrate it via the central-difference Lorentz ADE (a second
polarization `PL`), so `eps(w) = eps_inf - wp^2/(w^2 + i w gd) + d_eps w0^2/(w0^2 - w^2 - i w gl)` runs
natively across the band -- a bound-electron / interband resonance the bare Drude cannot represent.
With `d_eps=0` the path is byte-identical (`lor=None`). `fit_drude_lorentz` (seam) fits both poles to
sampled `eps(lambda)` with a scaled, multi-start least-squares (robust to the resonance overshoot).
The Lorentz ADE runs in 2D and the full-vector 3D kernels since f768b53
(`validation/fdtd_3d_lorentz_vs_tmm.py`).
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

### Per-cell tensor / magneto-optic eps (1-D)
`dynameta/optics/fdtd_mo.py` -- a normal-incidence, z-propagation, full transverse-polarization (Ex,Ey)
solver with per-cell DIAGONAL anisotropy (`eps_xx`, `eps_yy`) AND a gyrotropic magneto-optic response via
a magnetized-Drude auxiliary-differential-equation (the cyclotron `wc*(zhat x J)` coupling that mixes
Jx<->Jy). This is the physically-correct time-domain route: the off-diagonal `i*g` is a frequency-domain
stand-in for exactly this TIME-DERIVATIVE coupling (a complex algebraic `E = inv(eps) @ D` on real fields
is unphysical -- the design spec was wrong on this point). Implemented as a per-cell 2x2 Crank-Nicolson
solve coupled semi-implicitly to the E-update. Faraday rotation falls out (the +/- circular modes see
`eps_pm = eps_inf - wp^2/(w(w -/+ wc) + i w gamma)`). Validated (`validation/fdtd_mo_vs_tmm.py`):
birefringence vs per-pol TMM (~1e-2), Faraday vs circular-eigenmode Jones-TMM (rotation to 0.04 deg),
reduction at `wc=0`. Complements the frequency-domain FEM gyrotropic path (UPML) with a broadband
time-domain route.

### Oblique Bloch incidence (2D-TE / s-pol)
`solve_fdtd_2d_oblique` in `fdtd_nd.py` -- the complex-envelope (field-transform) Bloch method: the
physical field carries a fixed transverse wavevector `k_par`, so the periodic envelope is solved with
`d/dx -> d/dx + i*k_par` and a zero-phase roll (cleaner than the split cos/sin spec; reduces to the real
normal-incidence kernel at `k_par=0`). A fixed `k_par` makes the physical angle frequency-dependent,
`theta(f) = asin(k_par c/w)`. Validated (`validation/fdtd_2d_oblique_vs_tmm.py`) vs s-pol TMM at
`theta(f)`: reduction at angle 0, tracks TMM to ~2% (the thin-slab discretization floor) at 30/45 deg with
the genuine angle-effect far exceeding that error, energy closes.

### Numba JIT kernels for the MO + oblique solvers
Both `solve_fdtd_mo_1d` and `solve_fdtd_2d_oblique` accept `backend='numba'` (and `'auto'`/`'cpu'` select
it when present), a fused JIT-compiled time loop byte-for-byte equal to the NumPy reference to ~1e-15
(`validation/fdtd_numba_kernels.py`). MO: the per-cell magnetized-Drude 2x2 Crank-Nicolson loop, ~11x (the
O(nz) 2x2-inverse precompute stays in NumPy; only the hot loop is JITed). Oblique: the complex128 Bloch
envelope loop, ~5x -- this kernel is **serial, not prange-threaded**, because the oblique envelope is
laterally smooth so nx is small (~6-8) and threading that tiny x-extent loses to per-step thread overhead
(measured 0.6x with `parallel=True`); serial JIT wins. Mirrors the existing `_te2d_numba` /
`_te3d_numba` normal-incidence kernels.

### Multi-objective / multi-wavelength adjoint inverse design
`Fdtd2dDesignProblem` + `weighted_objective` (`optics/inverse_design.py`): a differentiable 2D-TE FDTD
over a designable density slab whose `spectrum(rho_p)` returns (R,T) at every target wavelength from ONE
adjoint solve, and a combiner that folds per-wavelength goals ({value, weight, sense 'max'/'min' or target})
into one loss for `topology_optimize`. Validated as a dichroic reflector (`validation/fdtd_multiobjective_design.py`):
max R@1500nm + min R@1300nm grows the spectral separation 12x to a binary design.

## Formerly deferred -- shipped 2026-06-08
- Full 3-D / structured tensor FDTD: SHIPPED -- `solve_fdtd_3d_mo` with the magnetized-Drude ADE
  (81869cd, `validation/fdtd_3d_mo_vs_1d.py`) plus the per-cell structured 3D tensor (ba9498d,
  `validation/fdtd_3d_structured_tensor.py`).
- Oblique on the jax backend, p-pol (TM), and structured/3-D oblique: SHIPPED -- TM oblique (3c47729),
  3D oblique (3257ac6), oblique JAX (7dac08c), each with its own validation.

## Resolved (was: blocked on environment)

### Numba-CUDA GPU kernel -- RESOLVED (2026-06-10)
The fused `numba.cuda` GPU kernel shipped f1b72e6 (2026-06-09). CUDA 13.1 and an RTX 4070 Ti are now
installed, and the kernel is hardware-validated: linear GPU == CPU to ~4e-15 (d2ca751), nonlinear worst
2.6e-15 (d646875). The CuPy backend already runs the vectorized loop on a device when one is available;
on Windows a JAX-GPU build is WSL2-only.

## Already resolved (correction)

### FEM off-diagonal (gyrotropic / tilted-anisotropic) tensor -- "B1"
NOT blocked. An earlier note attributed the off-diagonal optical-solve failure to an "NGSolve 6.2.2604
assembly defect"; that attribution was **overstated and retracted** (`docs/ngsolve_offdiag_investigation.md`).
A minimal reproducer (`docs/ngsolve_offdiag_check.py`) proves NGSolve assembles off-diagonal HCurl tensor
coefficients correctly to ~1.6e-16 in every construct DynaMeta uses. The real cause was inside DynaMeta:
`mesh.SetPML`'s coordinate stretch is exact only for ISOTROPIC media; for an anisotropic eps it perturbs
the decoupled field component by a resolution-independent ~3% (a y-pol ordinary wave gave T=1.07). The fix
(shipped) is an explicit anisotropic **UPML** folded into the weak form for the tensor path
(`solver.solve_fem`), plus a Poynting-flux R/T for the elliptical gyrotropic transmission. The
`_check_diagonal` guard is removed. Validated: `lc_tilted_fem.py` (ordinary wave tilt-invariant to ~1.6e-4)
and `magneto_optic_faraday.py` GATE D (gyrotropic FEM == circular-eigenmode Jones-TMM, lossless).
