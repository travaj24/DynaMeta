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

The authoritative, end-to-end example is
[validation/_reference_device.py](validation/_reference_device.py). The snippet below is the same
clean-break API, condensed:

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
reg.add(Material("air", ConstantOptical(1.0 + 0j)))
reg.add(Material("Si",  ConstantOptical(12.0 + 0j)))
# Dielectrics carry an OPTICAL eps (Stage 2/3) AND a DC eps (eps_static_dc) for
# the Stage-1 gate capacitance. For gate oxides the two differ a lot (HfO2 ~4
# optical vs ~18 DC) and eps_static_dc is REQUIRED -- Stage 1 RAISES if it is
# unset (the optical eps would under-predict gate accumulation). See docs/dielectrics.md.
reg.add(Material("Al2O3", ConstantOptical(2.756 + 0j), eps_static_dc=9.0))
reg.add(Material("HfO2",  ConstantOptical(4.0 + 0j),   eps_static_dc=18.0))
reg.add(Material("Al-Nd", ConstantOptical(-180 + 30j), is_metal=True))
reg.add(Material("Au",    ConstantOptical(-100 + 8j),  is_metal=True))
reg.add(Material("ITO",
    optical=DrudeOptical(eps_inf=4.25, m_opt_kg=0.225 * M_E, gamma_rad_s=1.1e14),
    transport=TransportModel(
        n_bg_m3=4.0e20 * 1e6, eps_static=9.5,
        # constant DOS mass here; validation/_reference_device.py uses a Kane m*(n).
        dos_mass_kg_of_n_m3=lambda n: np.full_like(np.asarray(n, float), 0.27 * M_E),
        band_gap_eV=3.6, chi_eV=4.5)))   # physics defaults to "equilibrium"

# 2. Geometry: a square unit cell + a bottom-to-top Stack of Layers. A Layer is
#    a background material plus optional sub-cell Inclusions. The Au nanopatch is
#    an air layer carrying one centred-square Au inclusion.
cell = UnitCell.square(370e-9)
stack = Stack(
    layers=[
        Layer("mirror",      70e-9, "Al-Nd"),
        Layer("lower_al2o3",  1e-9, "Al2O3"),
        Layer("lower_hfo2",   7e-9, "HfO2"),
        Layer("ito",          5e-9, "ITO"),
        Layer("upper_hfo2",   7e-9, "HfO2"),
        Layer("upper_al2o3",  1e-9, "Al2O3"),
        Layer("patch",       50e-9, "air",
              inclusions=[Inclusion(centered_square(cell, 175e-9), "Au")]),
    ],
    superstrate_material="air", substrate_material="Si")

# 3. Electrodes: each attaches to a layer with a footprint that is a CrossSection,
#    an edge selector ("x_lo"/"x_hi"/"y_lo"/"y_hi"), or "full" (the whole face).
electrodes = [
    Electrode("bot_contact", "mirror", "full", role="biased"),
    Electrode("top_contact", "patch", centered_square(cell, 175e-9), role="biased"),
    Electrode("ito_gnd_left",  "ito", "x_lo", role="ground", fixed_voltage_V=0.0),
    Electrode("ito_gnd_right", "ito", "x_hi", role="ground", fixed_voltage_V=0.0),
]

design = Design(name="my_modulator", unit_cell=cell, stack=stack,
                electrodes=electrodes, materials=reg)
# Mesh/optics defaults are sensible; override via Design(..., mesh_2d=Mesh2DSpec(),
# mesh_3d=Mesh3DSpec(), optical=OpticalSpec(polarization="x", lift="auto"))
# (all three specs live in dynameta.geometry).

# 4. The (bias, wavelength) grid.
sweep = Sweep(
    bias_points=[
        BiasPoint({"top_contact": +2.0}, "patch+2V"),
        BiasPoint({"top_contact": -2.0}, "patch-2V"),
    ],
    wavelengths_nm=[1200, 1300, 1400, 1500])

# 5. Run carriers -> Drude bridge -> optics. Returns a list of SweepRows held in
#    memory (the pipeline writes nothing to disk; persistence is up to you).
rows = run_pipeline(design, sweep, verbose=True)
for r in rows:
    print("{:9s} lam={:.0f}nm  R={:.4f}  phase={:+.1f} deg".format(
        r.bias_label, r.lambda_nm, r.result.R, r.result.phase_deg))
```

Run the the reference modulator reference design:

```bash
python -m validation._reference_device --quick            # 1-bias x 3-wavelength smoke test
python -m validation._reference_device                    # 2-bias x 9-wavelength sweep
python -m validation._reference_device --drift-diffusion  # solve Stage 1 with full drift-diffusion
```

## Results

`run_pipeline` returns a `list[SweepRow]` -- one row per `(bias, wavelength)`
solve, in memory. Nothing is written to disk; you choose how to persist.

Each `SweepRow` carries:

| field | meaning |
|---|---|
| `row.bias_label` | the `BiasPoint` label, e.g. `"patch+2V"` |
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

shift_nm = resonance_shift(lam, spectrum("patch-2V"), spectrum("patch+2V"))
print("resonance shift, +2V vs -2V: {:+.1f} nm".format(shift_nm))
```

## Status

v0.4.0 (pyproject) -- general bridge API: the `OpticalModel`/`TransportModel`
materials split, declarative `UnitCell` + `Stack` (`Layer` = background +
`Inclusion`s) + `Electrode` geometry, and `run_pipeline`; plus the
modulation-mechanism family (Pockels/Kerr/FK, thermo-optic, QCSE,
PCM/LC/graphene, magneto-optic), the FDTD engine (1D/2D/3D incl GPU +
nonlinear), the reliability axis (REL1-10 + D1-D4 drivers), and the
required-core Lumenairy RCWA/PMM optical backends. The the reference modulator
reference design ([validation/_reference_device.py](validation/_reference_device.py)) is the
validated end-to-end run. Forward plan:
[docs/roadmap_v0.5_integration_photonics.md](docs/roadmap_v0.5_integration_photonics.md).

Known limitations:

- The 2D DEVSIM carrier field is lifted to the 3D optical eps by a *carrier-field
  lift* (`OpticalSpec.lift`). `"auto"` (the default) picks the xy-product
  `SeparableXYLift` for a centred 4-fold (c4v) square device and an extrusion
  otherwise; it captures square symmetry but not full 3D patch-corner
  accumulation. (This replaces the old `use_symmetrization` flag.)
- Peripheral ITO ground contacts at the unit-cell edges (`"x_lo"` / `"x_hi"`)
  over-pin the ITO potential; physically the ground pads are mm-scale away.
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
[docs/audit/](docs/audit/).
