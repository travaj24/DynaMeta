# Pluggable seams: bring your own DEVSIM device or NGSolve mesh

`dynameta` is a *bridge*: the solver-agnostic `core/` spine (units,
`CarrierField`, the n->eps map, the `GeometryAlignment` keystone) connects a
carrier solver to an optical solver. Either end is replaceable -- you can supply
your own Stage-1 carrier model or your own Stage-3 geometry and still reuse the
Drude bridge + sweep orchestration. The defaults are the layered DEVSIM and
NGSolve builders.

## The Protocols (`core/interfaces.py`)

| Protocol | Methods | Default impl |
|---|---|---|
| `CarrierSolver` | `regions() -> [RegionInfo]`, `solve(bias) -> CarrierField` | `LayeredDevsimBuilder` |
| `OpticalGeometryBuilder` | `build()`, `mesh_regions() -> [str]`, `alignment() -> GeometryAlignment` | `LayeredOpticalBuilder` |
| `OpticalSolver` | `solve(geometry, eps_by_region, lambda_m, optical) -> OpticalResult` | `solve_fem` |
| `NToEpsMap` | `eps(n, lambda)` per region | `MaterialEpsMap` |

`run_pipeline` takes overrides for the two builders:

```python
run_pipeline(design, sweep,
             carrier_solver=MyCarrierSolver(...),     # replaces Stage 1
             optical_builder=MyOpticalBuilder(...))    # replaces Stage 3
```

The Protocols are `runtime_checkable`, so `isinstance(obj, CarrierSolver)` is a
quick conformance smoke-test.

## Bring your own CarrierSolver  (`examples/byo_carrier_solver.py`)

Replace ALL of Stage 1 -- DEVSIM, drift-diffusion, everything -- with any model
that can emit carrier densities: an analytic profile, an ML surrogate, another
TCAD tool, or measured C-V-derived profiles.

The contract:
- `solve(bias)` returns a `CarrierField` whose **region name(s) match the optical
  mesh's carrier-driven subdomains** (e.g. `"ito"`).
- Each semiconductor region provides a **gridded** `electron_density_m3` field on
  `grid_axes_m = {"x": ..., "y": ...}` where **"y" is the through-stack axis**
  (the bridge affinely remaps optical z onto this axis).
- Set `n_bg_by_region` and `unit_cell_m`.

The worked example injects a closed-form Thomas-Fermi accumulation layer and
shows it drives the Drude eps through ENZ -- no DEVSIM imported.

## Bring your own OpticalGeometryBuilder  (`examples/byo_optical_geometry.py`)

Supply your own NGSolve mesh + the bridge contract. Three methods:
1. `build()` -> an `OpticalGeometry` (mesh + z-bookkeeping).
2. `mesh_regions()` -> every subdomain material name (for coverage checks).
3. `alignment()` -> a `GeometryAlignment`: a `RegionAlignment(mesh_region,
   source_region, bbox_m)` for each carrier-driven subdomain, plus
   `fixed_eps_regions` (every other material -> material name). The bridge uses
   ONLY this contract -- it never touches your mesh object.

### The one non-obvious rule: periodicity is a pre-mesh OCC operation

Bloch periodicity comes from **face identifications on the OCC shape, applied
BEFORE meshing**:

```python
f0.Identify(fP, "px_0", IdentificationType.PERIODIC, translation)   # on the OCC face
mesh = ng.Mesh(occ.OCCGeometry(glued).GenerateMesh(...))            # then mesh
```

You **cannot** retrofit periodicity onto a bare `ng.Mesh` after the fact --
without the identifications, `ng.Periodic(fes)` silently returns a non-periodic
space and the unit cell is not Bloch-periodic (a silent physics error). The
example asserts the identifications took effect by counting the x/y pairings
created on the OCC shape (`n_px`, `n_py` > 0). NB: `ng.Periodic(HCurl).ndof` is
NOT smaller than `HCurl.ndof` in current NGSolve -- HCurl periodicity is enforced
by dof *coupling*, not by dropping dofs (verified identical on the validated
layered Park mesh) -- so the identification count, not ndof, is the honest signal;
the physical periodicity is exercised by the FEM solve.

## Validation hooks (use them -- the house style is no silent failure)

- `alignment.validate_coverage(mesh_regions)` -- every mesh material is either
  carrier-driven or fixed-eps, exactly once. Fails loudly on a typo'd region name.
- periodic-ndof reduction -- confirms the OCC `Identify` worked (see above).
- energy conservation `R + T + A ~= 1` on a lossless stack -- confirms the optical
  solve and the eps placement.

## What does NOT need replacing

The bridge (`core/bridge.assemble_eps`), the n->eps map, the lift, and the sweep
orchestration are geometry- and solver-agnostic. Replacing a builder does not
touch them -- that decoupling is the whole point of the architecture.

See also: `docs/dielectrics.md` (Stage-1 DC permittivity sourcing).
