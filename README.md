# DynaMeta

**DynaMeta** (import `dynameta`) is a declarative multi-physics simulation
library for tunable metasurface modulators.

Combines three stages:

1. **Stage 1 -- DC carriers.** DEVSIM Poisson in the device `(x, z)`
   cross-section. The default `"equilibrium"` model is a single-variable
   nonlinear Poisson with a Fermi-Dirac electron density -- the exact steady
   state of the gated capacitor (no currents); `physics="drift_diffusion"`
   adds the full Scharfetter-Gummel continuity solve. A density-dependent DOS
   effective mass (Kane non-parabolic) sets the conduction-band `Nc`, and a
   self-contained Aymerich-Humet F1/2 approximation replaces DEVSIM's broken
   Fermi integral. The gate-oxide DC permittivity drives the accumulation --
   see [docs/dielectrics.md](docs/dielectrics.md).

2. **Stage 2 -- Drude.** Maps `n(x, z, V_bias)` to complex permittivity
   `eps(x, z, V, lambda)` via the density-dependent `DrudeOptical` model, then lifts
   the `(x, z)` field into the 3D unit cell with the carrier-field lift
   (`OpticalSpec.lift`) -- xy-product symmetrization for square-patch geometries.

3. **Stage 3 -- Optical FEM.** 3D NGSolve HCurl Maxwell solve in a periodic
   unit cell (Bloch boundaries, PML top + bottom). Loads the bias-dependent eps
   and reports the complex reflection coefficient `r` (plus optional
   transmission/absorption) at normal incidence.

## Quick start

A representative end-to-end build is below: a square metal patch over a gated
transparent-conductor film between two oxide layers, on a metal mirror (a gap-plasmon
reflection cell). All numbers are round illustrative values, not a particular device.
The same cell, with a density-dependent (Kane) DOS mass instead of the constant one here, is the
shared validation fixture [validation/_reference_device.py](validation/_reference_device.py):

```python
import numpy as np

from dynameta.materials import (
    Material, MaterialRegistry, ConstantOptical, DrudeOptical, TransportModel, M_E)
from dynameta.geometry import (
    UnitCell, Stack, Layer, Inclusion, Electrode, Design, centered_square)
from dynameta.sweep import Sweep, BiasPoint
from dynameta.pipeline import run_pipeline   # pulls in devsim + ngsolve

# 1. Materials. A material is a *semiconductor* iff it carries a TransportModel;
#    is_metal=True tags a metal; everything else is a dielectric. The physics
#    ROLE is derived here, per material -- it is NOT set on the geometry.
reg = MaterialRegistry()
reg.add(Material("air",   ConstantOptical(1.0 + 0j)))
reg.add(Material("glass", ConstantOptical(1.45 ** 2 + 0j)))
# Dielectrics carry an OPTICAL eps (Stage 2/3) AND a DC eps (eps_static_dc) for
# the Stage-1 gate capacitance. For high-k gate oxides the two differ a lot, and
# eps_static_dc is REQUIRED -- Stage 1 RAISES if it is unset (the optical eps
# would under-predict gate accumulation). See docs/dielectrics.md.
reg.add(Material("oxA", ConstantOptical(1.70 ** 2 + 0j), eps_static_dc=9.0))
reg.add(Material("oxB", ConstantOptical(2.00 ** 2 + 0j), eps_static_dc=20.0))
reg.add(Material("mirror_metal", ConstantOptical(-120 + 20j), is_metal=True))
reg.add(Material("metal",        ConstantOptical(-60 + 4j),   is_metal=True))
# A degenerate transparent-conductor film: Drude optics + the carrier model.
# (dynameta.materials also has literature presets, e.g. materials.tco_presets.)
reg.add(Material("TCO",
    optical=DrudeOptical(eps_inf=4.0, m_opt_kg=0.30 * M_E, gamma_rad_s=1.2e14),
    transport=TransportModel(
        n_bg_m3=3.0e26, eps_static=9.0,
        dos_mass_kg_of_n_m3=lambda n: np.full_like(np.asarray(n, float), 0.30 * M_E),
        band_gap_eV=3.5, chi_eV=4.5)))   # physics defaults to "equilibrium"

# 2. Geometry: a square unit cell + a bottom-to-top Stack of Layers. A Layer is
#    a background material plus optional sub-cell Inclusions. The metal patch is
#    an air layer carrying one centred-square metal inclusion.
cell = UnitCell.square(400e-9)
stack = Stack(
    layers=[
        Layer("mirror", 80e-9, "mirror_metal"),
        Layer("lo_oxa",  2e-9, "oxA"),
        Layer("lo_oxb",  6e-9, "oxB"),
        Layer("film",    6e-9, "TCO"),
        Layer("up_oxb",  6e-9, "oxB"),
        Layer("up_oxa",  2e-9, "oxA"),
        Layer("patch",  40e-9, "air",
              inclusions=[Inclusion(centered_square(cell, 150e-9), "metal")]),
    ],
    superstrate_material="air", substrate_material="glass")

# 3. Electrodes: each attaches to a layer with a footprint that is a CrossSection,
#    an edge selector ("x_lo"/"x_hi"/"y_lo"/"y_hi"), or "full" (the whole face).
electrodes = [
    Electrode("bot_contact", "mirror", "full", role="biased"),
    Electrode("top_contact", "patch", centered_square(cell, 150e-9), role="biased"),
    Electrode("film_gnd_left",  "film", "x_lo", role="ground", fixed_voltage_V=0.0),
    Electrode("film_gnd_right", "film", "x_hi", role="ground", fixed_voltage_V=0.0),
]

design = Design(name="my_cell", unit_cell=cell, stack=stack,
                electrodes=electrodes, materials=reg)
# Mesh/optics defaults are sensible; override via Design(..., mesh_2d=Mesh2DSpec(),
# mesh_3d=Mesh3DSpec(), optical=OpticalSpec(polarization="x", lift="auto"))
# (all three specs live in dynameta.geometry).

# 4. The (bias, wavelength) grid.
sweep = Sweep(
    bias_points=[
        BiasPoint({"top_contact": +3.0}, "patch+3V"),
        BiasPoint({"top_contact": -3.0}, "patch-3V"),
    ],
    wavelengths_nm=[1000, 1050, 1100, 1150, 1200])

# 5. Run carriers -> Drude bridge -> optics. Returns a list of SweepRows held in
#    memory (the pipeline writes nothing to disk; persistence is up to you).
rows = run_pipeline(design, sweep, verbose=True)
for r in rows:
    print("{:9s} lam={:.0f}nm  R={:.4f}  phase={:+.1f} deg".format(
        r.bias_label, r.lambda_nm, r.result.R, r.result.phase_deg))
```

The same builder is exercised end-to-end by the validation suite
(`python -m validation.run_all`, needs the `[solvers]` extra).

## Results

`run_pipeline` returns a `list[SweepRow]` -- one row per `(bias, wavelength)`
solve, in memory. Nothing is written to disk; you choose how to persist.

Each `SweepRow` carries:

| field | meaning |
|---|---|
| `row.bias_label` | the `BiasPoint` label, e.g. `"patch+3V"` |
| `row.lambda_nm` | the wavelength in nm |
| `row.result` | an `OpticalResult` |

`OpticalResult` exposes `r` (complex reflection), `R` (reflectance),
`phase_deg`, `solve_time_s`, and `t` / `T` / `A` (transmission/absorption,
populated when a substrate transmission channel is solved).

Post-process spectra with the helpers in `dynameta.analysis`:

```python
from dynameta.analysis import resonance_shift

lam = sorted({r.lambda_nm for r in rows})
def spectrum(label):
    by_lambda = {r.lambda_nm: r.result.R for r in rows if r.bias_label == label}
    return [by_lambda[L] for L in lam]

shift_nm = resonance_shift(lam, spectrum("patch-3V"), spectrum("patch+3V"))
print("resonance shift, +3V vs -3V: {:+.1f} nm".format(shift_nm))
```

## What's new in v0.16

v0.16.0 extends the gated thin-film path (carriers, Schrodinger-Poisson, quantum and
nonlocal optics) and changes a few defaults. Every changed default has a keyword that
restores the v0.15.1 behaviour; the full list with measured sizes is in
[docs/audit/2026-11-15-v0.16.0-release.md](docs/audit/2026-11-15-v0.16.0-release.md).

Changed defaults (old behaviour in brackets):

- RCWA bridge: `formulation='auto'` (Li's rule on metallic / high-contrast 1-D cells in TM)
  and `raster_sampling='area'` [`formulation='laurent'`, `raster_sampling='point'`].
- `run_pipeline` gives layered optical solvers the carrier solve's native z grid
  [`carrier_solve_kwargs={"grid_n_z": 32}`].
- The 2-D DEVSIM mesh grades semiconductor faces to the screening length -- more accurate,
  slower [`Mesh2DSpec(semiconductor_refinement="legacy")`].
- An edge electrode on an equilibrium film is the film's Fermi-level reference, not a pinned
  column [`Electrode(..., pin_potential=True)`].
- `SchrodingerPoissonCarrier`: `fermi_level="model"`, `gate_map="face_field"`
  [`"bulk3d_T0"`, `"flat_band_excess"`]; the depletion sign error is fixed.
- `IntersubbandEffect(z_response="subbands")` [`"drude_plus_subbands"`, deprecated].
- `SchrodingerPoissonCarrier` (`n_bg_m3`, `m_eff_kg`, `eps_static`; `eps_oxide` with an oxide) and
  `Stacked3DSpec` (`n_bg_m3`, `eps_semi`, `eps_oxide`, `dos_mass_kg`) no longer assume a film: these are
  required [pass them explicitly; `Stacked3DSpec.from_design` reads them from the Design].
- Adjacent unlabelled `HydroLayer`s raise [`conductor_id=`, or `stack_rt(allow_walls=True)`].
- Differentiable RCWA (`rcwa_stack_RT` / `rcwa_stack_jones`): `formulation='auto'` where lumenairy's Li rule
  traces under JAX (5.51) [`formulation='laurent'`].

New, opt-in: `KaneBand` electrostatics; `FilmInterfaceTraps`, fixed / bulk charges and gate
work functions; `DualGateSchrodingerPoisson` (finite barriers, LDA, traps);
`EquilibriumDensityGradient`; `GateLeakage`; `optics.quantum_film` and
`optics.quantum_film_pipeline`; `optics.fem_2d_tm` (2-D periodic TM FEM, with a hydrodynamic
conductor layer); `BoundaryScattering` / `BandFillingBackground`; `dynameta.measurement`.

New material presets for epsilon-near-zero layers, each parameter a cited literature value:
doped oxides (`materials.tco_presets`: `azo`, `gzo`, `in_cdo`, `ito`, `fto`), metallic nitrides,
doped mid-IR InAs / InAsSb and phonon-polariton crystals (`materials.enz_presets`: `tin`, `zrn`,
`inas`, `inassb`, `sic`, `aln`, `sio2`), with the lattice / oscillator permittivity models and oxide
scattering laws behind them. Guide, sources and validity: [docs/enz_presets.md](docs/enz_presets.md).

Lumenairy: the floor stays `lumenairy>=5.22`. Recommended: >= 5.50.1 (analytic Li route on
RCWA shapes) and >= 5.51 for PMM geometric grading
(`degree=8, pmm_kwargs={'grade': 'geometric', 'elements_per_region': 12}`). The bridge is
verified through 5.51 (checked against the 5.51.0 source merged on lumenairy's main; re-check on the published
wheel). On lumenairy 5.50.x the PMM bridge refuses gain layers (that release's PMM gets them wrong).

## Status

v0.9.0 (pyproject) -- general bridge API: the `OpticalModel`/`TransportModel`
materials split, declarative `UnitCell` + `Stack` (`Layer` = background +
`Inclusion`s) + `Electrode` geometry, and `run_pipeline`; plus the
modulation-mechanism family (Pockels/Kerr/FK, thermo-optic, QCSE,
PCM/LC/graphene, magneto-optic), the FDTD engine (1D/2D/3D incl GPU +
nonlinear), the reliability axis (REL1-10 + D1-D4 drivers), the standalone
QD-SOA traveling-wave gain module (`dynameta.optics.soa`), and the required-core
Lumenairy optical backends -- RCWA/PMM plus the Berreman 4x4 anisotropic-planar,
EMT (Rytov) screen, and BOR-PMM axisymmetric tiers
(`dynameta.optics.lumenairy_bridge`).

Shipped since (audit R-9 -- this section stated v0.7.0 while shipping 0.9.0 and
omitted both later releases):

- **v0.8.0, fiber and semiconductor amplifiers** (`dynameta.optics.fiber_amp`):
  Giles steady-state Er/Yb amplifiers with ASE, `AmplifierChain` multi-stage
  cascades with PSD-based noise figure, thermal feedback, LMA/mode-area and
  polarization (PDG) modelling, the nonlinear limit set (SBS/SRS/TMI/MPI),
  GNLSE pulse propagation with Raman and self-steepening, CPA, spectroscopy and
  calibration; plus the SOA quantum-well gain and co-fit extensions
  (`dynameta.optics.soa.qw_gain`) and `dynameta.optics.amp_noise`.
- **v0.9.0, ENZ / BIC / nonlinear optics**: `optics.twm_reference` (SFG/DFG/OPA
  three-wave mixing), `optics.spdc_design` (SPDC / JSA / heralding),
  `optics.resonance` (complex-omega pole finder, `q_budget` Q_rad/Q_abs split,
  `berreman_enz_pole`), `optics.ringdown` (matrix-pencil harmonic inversion),
  `optics.aaa_poles` (AAA rational fit + Froissart filtering),
  `optics.bic` (polarization-vortex topological charge maps),
  `optics.nonlocal_tmm` (hydrodynamic non-local TMM), `optics.hydro_fem` /
  `optics.hydro_fdtd`, `optics.harmonics`, `optics.hot_carrier`, and
  `dynameta.analysis` (Fano / Lorentzian fitting, KK checks).

The generic gated-film metasurface cell (the
[validation/_reference_device.py](validation/_reference_device.py) fixture) is the
validated end-to-end run. Forward plan:
[docs/roadmap_v0.5_integration_photonics.md](docs/roadmap_v0.5_integration_photonics.md).

Known limitations:

- The 2D DEVSIM carrier field is lifted to the 3D optical eps by a *carrier-field
  lift* (`OpticalSpec.lift`). `"auto"` (the default) picks the xy-product
  `SeparableXYLift` for a centred 4-fold (c4v) square device and an extrusion
  otherwise; it captures square symmetry but not full 3D patch-corner
  accumulation. (This replaces the old `use_symmetrization` flag.)
- The FEM optical solve can exploit a mirror-symmetric cell to solve a HALF or
  QUARTER lateral domain (`Mesh3DSpec.symmetry = "half_x"/"half_y"/"quarter"`),
  with PEC/PMC symmetry walls replacing the periodic boundary -- ~1/2 or ~1/4 the
  DOFs for the SAME 0-order R/T. It is OPT-IN (`"none"` is the default; never
  auto-applied); `Design.detect_symmetry_reduction()` reports the best eligible
  mode and the build emits a one-time hint when one is available but unused.
  Scope: a centered, mirror-symmetric (c2v/c4v) cell at normal incidence with a
  scalar eps and rectangle/circle inclusions; the carrier-coupled (semiconductor),
  prismatic, and other-shape paths use the full periodic cell.
- Peripheral film ground contacts at the unit-cell edges (`"x_lo"` / `"x_hi"`)
  over-pin the film potential; physically the ground pads are mm-scale away.
- Stage 1 defaults to the equilibrium Fermi-Dirac Poisson solve (no currents);
  `TransportModel(physics="drift_diffusion", ...)` adds the full
  Scharfetter-Gummel continuity solve. Both defaults are classical, but quantum
  confinement IS available: `SchrodingerPoissonCarrier` (a pluggable Stage-1
  `CarrierSolver`; per-column / oxide-division / open-body variants) and the
  density-gradient correction (post-hoc `carriers/density_gradient.py` or
  in-Newton DG-DD, `validation/dg_dd_in_newton.py`).
- Oblique incidence (`OpticalSpec.incidence_angle_deg != 0`) is implemented for
  s-polarization AND p-polarization (Bloch-Floquet periodicity), plus conical
  incidence (`azimuth_deg != 0`, s-pol AND p-pol;
  `validation/conical_ppol_vs_tmm.py`), each validated against the `tmm` library
  (s/p R&T; conical phi-invariance + structured-cell symmetry). Measured envelope:
  s-pol is gated over 0-45 deg (3% tolerance to 30 deg, 4% at 45 deg; measured
  ~2.6%); p-pol, conical, and lossy cases are gated to 30 deg.
  It REQUIRES a vacuum/air incidence medium (`n_super = 1`; `solve_fem` raises on a
  non-vacuum superstrate at angle) and the polar angle is capped at `|theta| <= 60`
  deg. The fixed-alpha HalfSpace z-PML is not angle-aware: at 60 deg it CREATES
  energy (R+T ~ 1.17, report-only) -- `solve_fem` emits a warning; an angle-aware
  PML is open. Absorption is reported as the budget closure
  `A = 1 - R - T`; `OpticalResult.A_independent` is the independent volumetric-loss
  measurement (their difference is the energy diagnostic).
- The optical linear solve defaults to BDDC + GMRes
  (`OpticalSpec.linear_solver = "bddc_gmres"`); the AMS preconditioner is not the
  default because of sign-changing alpha/beta in the ENZ regime, but is available
  opt-in via `OpticalSpec.linear_solver = "ams"` / `"hypre"` on a HYPRE-built
  NGSolve (falls back to `bddc_gmres`; see
  [docs/installing_hypre_windows.md](docs/installing_hypre_windows.md)).
- The Lumenairy bridge adds a BOR-PMM *axisymmetric* (body-of-revolution) optical
  tier (`BorStackSpec` / `BorLayer` / `solve_bor`; needs `lumenairy >= 5.16.0`) for
  structures invariant under rotation about an axis -- concentric-ring gratings,
  fibers, axisymmetric diffractive lenses. Unlike the Cartesian RCWA/PMM/Berreman
  backends (which plug into the `LayeredStackSolver` seam and return a single 0-order
  `OpticalResult`), the BOR solve returns per-INCIDENT-MODE R/T over a discrete set of
  cylindrical modes (`BorResult`; each mode = a wave at a quantized polar angle);
  `.fundamental_result()` exposes the near-axis mode as an `OpticalResult`. Scope:
  axisymmetric geometry only (no Cartesian patterning), top-incidence, conventions
  identical to the rest of the stack (exp(-i omega t), `Im(eps) > 0`).

**Native 3D carriers** are validated for both the equilibrium solve and full 3D
drift-diffusion (`validation/carriers_3d.py`, `carriers_3d_dd.py`, transport via
`carriers_3d_resistor.py`), and `Stacked3DSpec.from_design` drives `run_pipeline`
from a `Design` with no hand-built alignment (`carriers_3d_from_design.py`).

**Testing & validation.** `python -m pytest tests/` is the fast CI gate
(numpy/scipy only -- data model, dielectric DB, Schrodinger-Poisson, and the
solver-free bridge spine). The heavier solver-backed physics lives in
`validation/*.py`; each ends `raise SystemExit(0 if ok else 1)`, so
`python -m validation.run_all` runs and gates the whole set by exit code (needs the
`[solvers]` extra: ngsolve/devsim/gmsh; budget tens of minutes).

**Roadmap.** The active forward plan is silicon photonics integration (B1-B3):
[docs/roadmap_v0.5_integration_photonics.md](docs/roadmap_v0.5_integration_photonics.md).
Completed phases (linked as records):
[docs/roadmap_v0.3_modulation.md](docs/roadmap_v0.3_modulation.md),
[docs/physics_depth_roadmap.md](docs/physics_depth_roadmap.md),
[docs/reliability_roadmap.md](docs/reliability_roadmap.md), and
[docs/roadmap_phase5_stretch.md](docs/roadmap_phase5_stretch.md); the independent audit:
[docs/audit/](docs/audit/). Material guides: [docs/dielectrics.md](docs/dielectrics.md) (gate
dielectrics), [docs/enz_presets.md](docs/enz_presets.md) (ENZ material presets).
