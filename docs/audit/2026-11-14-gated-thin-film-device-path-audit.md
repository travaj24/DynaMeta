# Gated thin-film device path: optical bridges, layered DEVSIM builder, quantum and nonlocal modules (2026-11-14)

Audit of the code a user walks through to model a thin degenerate-semiconductor film that is gated from one
or both sides through thin oxides and read out optically through a laterally structured cell: the lumenairy
PMM and RCWA bridges, `LayeredDevsimBuilder` (equilibrium physics), the Schrodinger-Poisson carrier, the
intersubband permittivity, the frozen-potential density-gradient correction and the hydrodynamic transfer
matrix. 19 candidate findings reported by earlier work were each re-derived from the source and reproduced
(or refuted) with a small neutral script; 4 further defects found on the way are added (N1-N4). No library
code was changed. Every number below was MEASURED unless it is marked otherwise.

## Version, scope, method

- **Audited code:** DynaMeta **v0.15.1**, commit **2e1af87** ("Merge pull request #37"), which equals the
  local `origin/main` on the day of the audit. The package was imported from a read-only clone at that commit
  (`sys.path.insert(0, clone)`, with an assertion that `dynameta.__file__` lies in the clone; the pip-installed
  copy is older and was not used).
- **lumenairy:** 5.50.2 (a frozen tree of that release) for every number below. The RCWA bridge behaves
  differently from lumenairy 5.50.1 on (`_common.FEATURE_FLOORS['rcwa_shapes_li']`): from that version
  `formulation='li'` keeps analytic shapes, while on 5.50.0 it rasterizes; the 5.50.0 raster route is
  reproduced below with `inclusion_model='raster'` (O4).
- **Fixtures (2026-11-15).** The audit was first run on a different strip cell and gated film; both were
  replaced by the generic fixtures of "Fixtures" below and every number in this document was re-measured on
  them on an export of the v0.15.1 tree, unless it is marked "(not re-measured on the generic fixture)" or
  quoted from a v0.16 package note that re-measured it (the note is named in place).
- **DEVSIM** 2.11.2 (UMFPACK 5.1 direct solver), Python 3.14, numpy 2.4.4, Windows, BLAS threads pinned to 2.
- **Method.** For each candidate: read the source, write the smallest reproduction, try to refute it (wrong
  usage, wrong reference, version dependence), then classify as CONFIRMED, NOT REPRODUCED, WORKS AS DESIGNED
  (a documented limitation) or ALREADY FIXED. Proposed fixes are untested unless the text says "prototyped";
  prototypes were run in a scratch folder, never in the library.
- **Out of scope:** drift-diffusion and bipolar physics, the 3-D builder, FDTD, and the NGSolve optical
  solver except where a finding depends on them.

## Terms

- **TM (grating sense).** Electric field in the plane of incidence and across the grating lines (here `x`
  polarization on y-invariant strips). It is the field component that crosses the metal side walls.
- **Laurent / Li rule.** Two ways RCWA builds the Fourier matrix of the permittivity. Laurent's (direct) rule
  is correct for a field component continuous across a wall; Li's inverse rule is the one that converges for
  the normal component (TM on metal walls).
- **PMM.** lumenairy's polynomial modal method: spectral elements per lateral region. `degree` is the
  polynomial order per element, `elements_per_region` (epr) the number of elements per region.
- **Raster.** The point-sampled `eps_cell` the RCWA bridge paints when it does not use analytic shapes. Pixel
  `j` of an `S`-pixel cell is sampled at `(j + 0.5) P / S` (`optics.rasterize.cell_axes`); lumenairy reads
  `eps_cell[j]` as the value at the node `j P / S` (lumenairy PIXEL CELL CONTRACT).
- **Gated film fixture.** A 6 nm degenerate-semiconductor film (material "TCO", region "film"; n_bg =
  3e26 m^-3, static eps 9.0, DOS mass 0.30 m_e, equilibrium physics) between 2 nm oxA (n 1.70, static eps
  9.0) plus 6 nm oxB (n 2.00, static eps 20.0) below and 6 nm oxB plus 2 nm oxA above, an 80 nm metal gate at
  the bottom (the mirror layer) and a 40 nm metal gate at the top (code in "Fixtures").
- **Strip cell fixture.** A 1-D grating of 40 nm tall metal strips (eps -60+4i), 150 nm wide, period 400 nm,
  on the same oxide/film/oxide stack over an 80 nm mirror metal (eps -120+20i) on glass (n 1.45), air above,
  TM, normal incidence, 1200 nm. Film optical model: Drude, eps_inf 4.0, m_opt 0.30 m_e, gamma 1.2e14 rad/s,
  eps(n_bg) = 2.7159+0.0982i. In TM the cell is a gap-mode resonator (reflection dip near 1190 nm, R = 0.113
  at 1200 nm).
- **r.** Zeroth-order co-polarized complex reflection amplitude, exp(-i omega t).
- **Modulation.** |r(biased) - r(unbiased)| for the same cell; used to size an error against the effect being
  modelled.

## Summary

Severity: **High** = a default or a silent path gives a wrong number of the size of the effect under study;
**Medium** = wrong or misleading only in a common but not default usage, or a large missing capability;
**Low** = small error or convenience.

| ID | Title | Class | Severity | Status |
|---|---|---|---|---|
| O1 | PMM bridge refuses a laterally structured tunable layer | missing feature | Medium | WORKS AS DESIGNED (documented) |
| O2 | PMM bridge does not expose `elements_per_region` | usability | Low | NOT REPRODUCED (as a convergence defect) |
| O3 | RCWA bridge default (Laurent) is badly wrong for TM on metal strips, no warning | incorrect default | High | CONFIRMED (Laurent path, every lumenairy version) |
| O4 | RCWA default raster moves a strip edge by up to half a pixel | bug | Medium | CONFIRMED on lumenairy 5.50.0 (generic cell: default lattice misses the edges at odd N only); avoided by the default on >= 5.50.1 |
| O5 | "Graded slab scheme silently gives 0 slabs in metres" | - | - | NOT REPRODUCED (not library code; see N2) |
| C1 | `solve(grid_n_z=32)` resamples the film to 32 uniform z points, also on the `run_pipeline` path | bug | High | CONFIRMED |
| C2 | Equilibrium edge ground pins the potential on the whole edge column | incorrect physics | Medium | CONFIRMED |
| C3 | `rel_tol <= 1e-7` never converges at 0 V (and, measured, at any bias here) | usability | Low | WORKS AS DESIGNED (documented) |
| C4 | Default 2-D mesh grading leaves 19 nodes across a 6 nm film | incorrect default | High | CONFIRMED |
| C5 | Electrostatics uses a parabolic band with the DOS mass frozen at n_bg | incorrect physics / missing feature | Medium | CONFIRMED |
| C6 | No gate work function / flat-band voltage or fixed charge for an equilibrium film | missing feature | Medium | CONFIRMED (interface layers: not a gap) |
| C7 | DEVSIM `debug_level="error"` is more verbose | - | - | NOT REPRODUCED as a DynaMeta defect (DEVSIM behaviour) |
| Q1 | `solve_self_consistent` ignores `phi_left_V` / `phi_right_V` under a warm start | bug | Medium | CONFIRMED |
| Q2 | `SchrodingerPoissonCarrier` is single-gate; 14 % spurious film charge at 0 V | missing feature / incorrect physics | High | CONFIRMED |
| Q3 | Kane SP fill gives 0.90 n_bg in the bulk at the carrier's E_F | incorrect physics | Medium | CONFIRMED (0.896, not 0.85) |
| Q4 | `IntersubbandEffect` counts the free-carrier z response twice | incorrect physics | High | CONFIRMED |
| Q5 | `dg_correct_density_1d` on a thin film: accumulation removed, charge moved to the far side | incorrect physics / usability | Medium | CONFIRMED |
| Q6 | `nonlocal_tmm`: J_z = 0 walls between layers of one conductor | incorrect physics | Medium | CONFIRMED |
| Q7 | No nonlocal/quantum response for a graded layer in a structured cell; no 2-D TM cross-check solver | missing feature | Medium | CONFIRMED |
| N1 | RCWA bridge ignores a gridded EpsField's x coordinates (endpoint-inclusive grid misregistered) | bug | Low | CONFIRMED |
| N2 | Bridges do not check an EpsField's z extent against the layer thickness | bug | High | CONFIRMED |
| N3 | `IntersubbandEffect` dipole quadrature is inconsistent with the eigenvector normalization | bug | Low | CONFIRMED |
| N4 | `SchrodingerPoissonCarrier` depletion: sign error in the oxide charge term | bug | High | CONFIRMED (fix prototyped) |

Counts: 18 CONFIRMED (14 candidates + N1-N4), 2 WORKS AS DESIGNED, 3 NOT REPRODUCED, 0 ALREADY FIXED (O4 is
avoided by the default on lumenairy >= 5.50.1, but remains on 5.50.0 and on every raster route).

## Fixtures

Both fixtures use only public DynaMeta API. Every reproduction below starts from one of them.

```python
# Fixture A: strip cell (optics)
import numpy as np
from dynameta.materials import M_E, ConstantOptical, DrudeOptical, Material, MaterialRegistry, TransportModel
from dynameta.geometry import Design, Inclusion, Layer, Stack, UnitCell, Electrode
from dynameta.geometry.cross_section import Rectangle
from dynameta.geometry.specs import OpticalSpec, Mesh2DSpec
from dynameta.core.eps_field import EpsField

LAM, P, W, N_BG = 1200e-9, 400e-9, 150e-9, 3.0e26
DRUDE = DrudeOptical(eps_inf=4.0, m_opt_kg=0.30 * M_E, gamma_rad_s=1.2e14)
def mats():
    reg = MaterialRegistry()
    reg.add(Material("air", ConstantOptical(1.0 + 0j)))
    reg.add(Material("glass", ConstantOptical(1.45 ** 2 + 0j)))
    reg.add(Material("oxA", ConstantOptical(1.70 ** 2 + 0j), eps_static_dc=9.0))
    reg.add(Material("oxB", ConstantOptical(2.00 ** 2 + 0j), eps_static_dc=20.0))
    reg.add(Material("metal", ConstantOptical(-60.0 + 4.0j), is_metal=True))
    reg.add(Material("mirror_metal", ConstantOptical(-120.0 + 20.0j), is_metal=True))
    reg.add(Material("TCO", optical=DRUDE, transport=TransportModel(
        n_bg_m3=N_BG, eps_static=9.0, physics="equilibrium",
        dos_mass_kg_of_n_m3=lambda n: 0.30 * M_E * np.ones_like(np.asarray(n, float)))))
    return reg
FILM = [("mirror", 80e-9, "mirror_metal"), ("lo_oxa", 2e-9, "oxA"), ("lo_oxb", 6e-9, "oxB"),
        ("film", 6e-9, "TCO"), ("up_oxb", 6e-9, "oxB"), ("up_oxa", 2e-9, "oxA")]
def strip_design(width=W):
    layers = [Layer(n, t, m) for n, t, m in FILM]
    layers.append(Layer("grating", 40e-9, "air",
                        inclusions=[Inclusion(Rectangle(P / 2, P / 2, width, P), "metal")]))
    return Design(name="strip", unit_cell=UnitCell(period_x_m=P, period_y_m=P),
                  stack=Stack(layers=layers, superstrate_material="air", substrate_material="glass"),
                  electrodes=[], materials=mats(),
                  optical=OpticalSpec(polarization="x", incidence_angle_deg=0.0))
eps_film = lambda n: complex(DRUDE.eps(LAM, n_m3=n))
EBR = {"film": EpsField(scalar=eps_film(N_BG))}
# solver call used throughout: solver(design, None, eps_by_region, LAM, None, None)
```

```python
# Fixture B: gated film (carriers); continues from fixture A
from dynameta.carriers.devsim_layered import LayeredDevsimBuilder
from dynameta.carriers.devsim_env import import_devsim
from dynameta.sweep import BiasPoint
ds = import_devsim()
def film_design(ground=False, **mesh):
    layers = [Layer(n, t, m) for n, t, m in FILM] + [Layer("gate", 40e-9, "metal")]
    el = [Electrode("bot", "mirror", "full"), Electrode("top", "gate", "full")]
    if ground:
        el.append(Electrode("film_gnd", "film", "x_lo", role="ground", fixed_voltage_V=0.0))
    m = dict(x_spacing_edge_m=4e-9, x_spacing_feature_edge_m=4e-9, x_spacing_feature_mid_m=4e-9); m.update(mesh)
    return Design(name="film", unit_cell=UnitCell.square(40e-9),
                  stack=Stack(layers=layers, superstrate_material="air", substrate_material="glass"),
                  electrodes=el, materials=mats(), mesh_2d=Mesh2DSpec(**m))
FINE = dict(interface_ps_m=1.25e-11, coarse_ps_m=5e-11, interface_zone_m=2.25e-9)  # 189 nodes across the film
def column(b, x, name="Electrons"):            # raw DEVSIM node values of one x column of the film
    X = np.array(ds.get_node_model_values(device=b.device, region="film", name="x"))
    Y = np.array(ds.get_node_model_values(device=b.device, region="film", name="y"))
    V = np.array(ds.get_node_model_values(device=b.device, region="film", name=name))
    xc = np.unique(np.round(X, 15)); xc = xc[np.argmin(abs(xc - x))]; k = abs(X - xc) < 1e-13
    o = np.argsort(Y[k]); return Y[k][o], V[k][o]
```

Optical reference for fixture A: lumenairy `PMMStack` built directly with the same layers (strip as segments
`[(g, air), (W/P, metal), (g, air)]`, g = (1 - W/P)/2). Convergence of r (epr x degree):

| epr \ degree | 16 | 24 | 32 | 40 |
|---|---|---|---|---|
| 1 | 0.342844-0.001633i | 0.338522+0.009395i | 0.337026+0.013044i | 0.336354+0.014655i |
| 3 | 0.336972+0.013175i | 0.335983+0.015533i | 0.335652+0.016310i | 0.335505+0.016653i |
| 6 | 0.335670+0.016269i | 0.335437+0.016811i | 0.335359+0.016990i | 0.335325+0.017068i |
| 8 | 0.335484+0.016702i | 0.335359+0.016990i | 0.335318+0.017084i | 0.335300+0.017126i |

Reference **r_ref = 0.3352997+0.0171262i** (epr 8, degree 40, 119 s); its last step moved r by 4.6e-5, so
r_ref is good to about 1e-4 (the
converged 2-D TM FEM value of the same cell, `tests/_strip_cell_fixture.R_FEM_CONVERGED` = 0.335270+0.017192i,
is 7.2e-5 away, and a Richardson extrapolation of the PMM sequence in epr lands 5e-6 from it; see
`2026-11-14-v0.16-fem-2d-tm.md`). The corner singularity makes every method converge algebraically on this
cell (as recorded in `2026-11-02-rcwa-auto-default.md`).

---

## O1. PMM bridge refuses a laterally structured tunable layer (WORKS AS DESIGNED, missing feature)

**What.** When the tunable layer's EpsField varies laterally (the film under a strip is gated differently
from the film between strips, so every z slice is lamellar), `design_to_pmm_stack` raises. A z-graded field
that is laterally uniform is accepted. The only way to model the biased cell with PMM is to build a lumenairy
`PMMStack` by hand.

**Where.** `dynameta/optics/lumenairy_bridge/pmm_backend.py:205-211` (the refusal); the module docstring
(lines 12-16) states the 1-D lamellar-inclusions-only scope, so this is documented.

**Reproduction** (fixture A):

```python
from dynameta.optics.lumenairy_bridge.pmm_backend import make_lumenairy_pmm_solver
x = np.linspace(0, P * 1e9, 256)                                   # nm, like assemble_eps
row = np.where(abs(x - P * 5e8) <= W * 5e8, eps_film(6e26), eps_film(N_BG))
lam = EpsField(x_axis_u=x, y_axis_u=np.array([0.0]), z_axis_u=np.array([0.0, 6.0]),
               values_zyx=np.broadcast_to(row, (2, 1, 256)).astype(complex))
make_lumenairy_pmm_solver(degree=24)(strip_design(), None, {"film": lam}, LAM, None, None)
```

Observed: `ValueError: PMM bridge: layer 'film' carries a laterally-structured gridded EpsField; PMM segments
are analytic -- use the RCWA bridge for rasterized cells`. The same call with a laterally uniform z-graded
field (the N2 field) returns r = -0.2350-0.4605i.

**Why it matters.** PMM is the bridge's documented convergence referee for TM metal cells (module docstring);
the biased cell is exactly where a referee is needed, and the RCWA alternative has O3, O4 and N1. For this
two-zone film (eps 1.4317+0.1963i under the strip, 2.7159+0.0982i outside) the hand-built PMM gives r =
0.068069-0.679514i (epr 6, degree 32) against 0.335359+0.016990i unbiased at the same setting.

**Smallest design.** Accept a gridded field whose x variation is resolved onto a lateral partition:

```python
design_to_pmm_stack(design, lam, eps_by_region=..., lateral_partition="inclusions" | "uniform:<k>" | sequence)
```

For each z slab (`slice_eps_field` already produces them), average eps over the intervals of the partition
(default: the union of the stack's inclusion edges, i.e. the strip edges, optionally subdivided into k
sub-intervals near each edge) and emit `segments=[(width_fraction, eps_mean), ...]`. Use the field's own
`x_axis_u` (trapezoid weights on the periodic grid, duplicate endpoint dropped) for the averages. Report the
largest relative deviation of eps from its interval mean so the user sees what the partition throws away.
Builds on `layer_to_pmm_segments` (segment assembly) and `slice_eps_field` (z slabs). Regression test: a
two-value lamellar field must reproduce the hand-built segment stack to 1e-12 (same degree); a field with a
linear ramp across the edge must converge to the RCWA 'li' result as the partition is refined.

## O2. PMM bridge does not expose `elements_per_region` (NOT REPRODUCED as a convergence defect)

**Claim tested.** "Convergence in degree alone is slow at metal corners; `elements_per_region` is the only
effective knob, and the bridge does not expose it."

**Where.** `pmm_backend.py:197-198` builds `PMMStack(px, ..., degree=, n_orders=)`; `make_lumenairy_pmm_solver`
(239-242) has no `elements_per_region`, `grade` or `min_feature` argument. That part is true.

**Reproduction** (fixture A, same cell, |r - r_ref| and wall time on a shared machine):

| route | abs(r - r_ref) | time |
|---|---|---|
| bridge, degree 32 (epr 1) | 4.43e-3 | 0.3 s |
| bridge, degree 48 | 1.77e-3 | 0.8 s |
| bridge, degree 64 | 9.05e-4 | 1.6 s |
| direct, epr 3, degree 32 | 8.89e-4 | 4.0 s |
| direct, epr 6, degree 16 | 9.33e-4 | 4.0 s |

**Conclusion.** Degree alone reaches the same accuracy as h-refinement at well under half the cost here (192
vs 288 unknowns per slice). The convergence claim does not hold for this cell. Exposing the remaining
`PMMStack` keywords is still a reasonable convenience (one `pmm_kwargs: dict | None = None` forwarded verbatim
and included in `cache_fingerprint`), but it is not a correctness issue.

## O3. RCWA bridge default is badly wrong for TM on metal strips, with no warning (CONFIRMED, High)

**What.** `make_lumenairy_rcwa_solver()` defaults to `formulation='laurent'`, `inclusion_model='auto'`. On a
metal-strip cell in TM this converges very slowly and from far away. Nothing warns.

**Where.** `rcwa_backend.py:261` (`design_to_rcwa_stack(..., formulation="laurent")`) and `:551`
(`make_lumenairy_rcwa_solver(..., formulation="laurent")`). The weakness is recorded in
`docs/audit/2026-11-02-rcwa-auto-default.md` ("Laurent's rule is the wrong factorization for the field
component normal to a high-contrast wall"), which kept Laurent as the default; there it cost 3e-2 in R at
n_orders 32 on a 600 nm lamellar grating between air and glass. On a strip over a mirror (a gap-mode resonator)
the error is an order of magnitude larger.

**Reproduction** (fixture A):

```python
from dynameta.optics.lumenairy_bridge.rcwa_backend import make_lumenairy_rcwa_solver
for N in (50, 100, 200):
    for kw in ({}, dict(formulation="li")):
        res = make_lumenairy_rcwa_solver(n_orders=N, **kw)(strip_design(), None, EBR, LAM, None, None)
        print(N, kw, res.r, abs(res.r - (0.3352997 + 0.0171262j)), res.R)
```

Observed (warnings recorded with `warnings.catch_warnings(record=True)`: 0 in every case):

| N | default (Laurent) r | abs(r - r_ref) | R (ref 0.1127) | 'li' r, lumenairy 5.50.2 | abs(r - r_ref) | 'li' R |
|---|---|---|---|---|---|---|
| 50 | -0.142794+0.150175i | 0.496 | 0.0429 | 0.276988-0.022362i | 0.0704 | 0.0772 |
| 100 | 0.203083+0.122269i | 0.169 | 0.0562 | 0.314803+0.003567i | 0.0246 | 0.0991 |
| 200 | 0.315889+0.028712i | 0.0226 | 0.1006 | 0.328568+0.011310i | 0.0089 | 0.1081 |

The default uses Laurent's rule on analytic shapes on every lumenairy version (the Laurent path does not
depend on the 5.50.1 feature floor).

**Why it matters.** At N = 50 the default cuts R to 38 % of its value (0.043 vs 0.113) and puts r in the
other half-plane (Re r < 0); at N = 100 the error (0.17) is more than a third of the bias-induced modulation
this cell shows in C1 (0.457). A user who takes the defaults gets a confidently wrong answer.

**Root cause.** The default factorization is Laurent for every polarization and every material contrast.

**Proposed fix.**
1. New default `formulation="auto"`: 'li' when any patterned layer has a metal (Re eps < 0) or a contrast
   |eps_in - eps_out| > 10 and the incident field has a component across those walls (row 0 for x walls on a
   1-D stack; always on a 2-D stack); 'laurent' otherwise. With lumenairy >= 5.50.1, 'li' keeps the analytic
   shapes (O4); below it, 'li' rasterizes (O4 applies) and the bridge should say so.
2. Whatever the default, emit an `AdvisoryWarning` once per solver when the resolved formulation is
   'laurent' and the condition in (1) holds, naming `formulation='li'`.
3. Regression test: fixture A must be within 0.02 of r_ref with the default at N = 200 (fails today at
   0.0226; 'li' gives 0.0089). The generic cell converges slowly in N, so the bar sits at N = 200: at
   N = 100 'li' is still 0.0246 off (Laurent 0.169, 6.9x).
   A TE cell (polarization 'y') must be bit-identical before and after the change.

## O4. RCWA default raster moves a strip edge by up to half a pixel (CONFIRMED on lumenairy 5.50.0; generic cell: odd N)

**What.** With `formulation='li'` and lumenairy <= 5.50.0, `'auto'` rasterizes every inclusion layer
(`_check_model_formulation`, `rcwa_backend.py:199-225`) on the default lattice `S = max(128, 8 N)`
(`_default_raster_samples`, `:157-186`), cell-centred samples (`rasterize.cell_axes`). An edge that does not
fall on a pixel boundary is moved by up to half a pixel, and r follows it.

**Reproduction** (fixture A, `formulation='li'`; the lumenairy 5.50.0 route reproduced on 5.50.2 with
`inclusion_model='raster'`, which paints the same point raster):

| N | strip (drawn) | lattice | realised width | r |
|---|---|---|---|---|
| 50 | 150.000 nm | default S = 400 (edges on pixel boundaries) | 150.000 nm (150 pixels x 1.000 nm) | 0.303405-0.017257i |
| 50 | 150.000 nm | S = 404 | 150.495 nm (152 pixels x 0.990 nm) | 0.254636+0.026105i |
| 50 | 150.000 nm | S = 3200 (edges on pixel boundaries) | 150.000 nm | 0.277415-0.022676i |
| 51 | 150.000 nm | default S = 408 | 150.980 nm (154 pixels x 0.980 nm) | 0.322357+0.011926i |
| 51 | 150.000 nm | S = 3264 (edges on pixel boundaries) | 150.000 nm | 0.322791+0.006574i |
| 51 | 150.980 nm | S = 3264 | 150.980 nm | 0.290341+0.065523i |

PMM (epr 3, degree 24): width 150.000 nm r = 0.335983+0.015533i, width 150.495 nm r = 0.318855+0.057203i
(|dr| = 0.045 for the 0.495 nm), width 150.980 nm r = 0.299206+0.096982i (|dr| = 0.089).

**Why it matters.** On the generic cell the default lattice at even N (S = 8 N, a multiple of 16) happens to
put both strip edges on pixel boundaries, so there the default raster realises the drawn width and the
width error of this finding does not appear; the default lattice still moves r by 0.027 from the S = 3200
result at N = 50 (the point-sampling aliasing of S = 400 vs 3200 at equal width; 2.8e-3 at N = 100,
S = 800). A lattice that misses the edges (S = 404, 0.495 nm too wide) moves r by 0.054, most of it (0.045
by PMM) the 0.33 % width error. At odd N the default lattice itself misses the edges (N = 51: S = 408,
0.98 nm too wide); there the width error (0.089 by PMM, 0.065 between the two S = 3264 rows) and the
aliasing of S = 408 happen to nearly cancel (5.4e-3 from the S = 3264 result): the default's error is
uncontrolled, not small. The cell's r moves by about 0.09 per nm of strip width, so a sub-nm geometry
error is not negligible.

**Version dependence.** On lumenairy 5.50.2, `formulation='li'` with the default `'auto'` uses analytic
shapes under Li's rule: r = 0.276988-0.022362i at N = 50, equal to the S = 3200 raster (0.277415-0.022676i)
to 5.3e-4 (N = 51: 7.9e-5 from the S = 3264 raster). The defect remains on 5.50.0, and on every version for
`inclusion_model='raster'`,
`absorption=True` below 5.50.1, polygons, overlapping inclusions, tensor backgrounds and gridded EpsFields
(N1). The edge-quantization caveat is already described in `2026-11-02-rcwa-auto-default.md`; what is
missing is a runtime signal.

**Proposed fix.** (a) When the bridge rasterizes a Rectangle inclusion, choose S (>= the current default) so
that every rectangle edge lies on a pixel boundary when such S exists below a cap (here any multiple of 16:
the default S = 400 at N = 50 qualifies, S = 408 at N = 51 does not), else (b) paint an area-weighted raster
(each boundary pixel gets the covered fraction of eps, and for 'li' the harmonic mean, as lumenairy's
`raster='area'` does), else (c) warn with the realised width. Regression test: the realised width of every
rectangle in the painted cell equals the drawn width to 1e-9 relative, and fixture A with 'li' at N = 50
agrees with the S = 3200 result to 1e-3 on lumenairy 5.50.0.

## O5. "Graded slab scheme takes nm and silently gives 0 slabs in metres" (NOT REPRODUCED)

The library has no slab scheme with that interface: `slice_profile` / `slice_eps_field` (`core/layered.py`)
take an explicit `metres_per_unit` and never return zero slabs for a non-empty field. The reported behaviour
belongs to user code outside the library. A related library defect does exist and is reported as N2.

---

## C1. `solve(grid_n_z=32)` resamples the film to 32 uniform z points, also on the pipeline path (CONFIRMED, High)

**What.** `LayeredDevsimBuilder.solve` returns a `CarrierField` whose `grid_fields` are a linear
interpolation of the DEVSIM nodes onto a uniform `grid_n_x x grid_n_z` grid (defaults 256 x 32).
`assemble_eps` builds every EpsField from `grid_fields`, never from `node_fields`. `run_pipeline` calls
`carrier.solve(bp)` with no grid arguments, so the 32-point grid is not even adjustable there.

**Where.** `carriers/devsim_layered.py:1033` (defaults), `:1500-1519` (`resample_to_grid` onto
`(grid_n_x, grid_n_z)`); `core/resample.py:40-43` (uniform `linspace(min, max, n)` axes);
`core/bridge.py:80-87` (`grid_fields` only); `pipeline.py:120` (`carrier.solve(bp)`).

**Reproduction** (fixture B, fine mesh, top gate +5 V, bottom gate 0 V, no ground; then the strip cell with a
laterally uniform film, slabs from the library's own `slice_profile`, PMM epr 3 degree 24):

```python
b = LayeredDevsimBuilder(film_design(**FINE)); b.build_device()
z0 = b.design.z_intervals()["film"][0]
cf = b.solve(BiasPoint({"top": 5.0, "bot": 0.0}, "b"), grid_n_x=8)       # grid_n_z default 32
reg = cf.regions["film"]
zg = reg.grid_axes_m["y"] - z0; ng = reg.grid_fields["electron_density_m3"].mean(axis=0)
z, n = column(b, 20e-9); z = z - z0
# r from slice_profile([eps_film(v) for v in n], z) vs slice_profile(..., zg)
```

Observed: grid spacing 0.194 nm (32 points) vs 189 raw nodes; the interface value is exact (grid end points
are nodes) but the excess sheet density is +0.74 % (4.6265e17 vs 4.5925e17 m^-2). Optically:

| profile fed to the slicer | r | abs(r - r(raw)) |
|---|---|---|
| raw nodes, fine mesh (189) | 0.080680-0.363533i | 0 |
| raw nodes, mesh 2x finer (374) | 0.079391-0.360246i | 3.5e-3 |
| 32-point grid (the default) | 0.065179-0.411211i | **5.01e-2** |
| 128-point grid | 0.093220-0.358687i | 1.34e-2 |

The bias modulation is |r - r(unbiased)| = 0.457 (unbiased r = 0.335983+0.015533i at the same PMM setting).

**Why it matters.** The default grid moves r by 0.050, 11 % of the modulation, on a converged carrier
solution. The face density (9.86e26 m^-3) lies just above the density at which Re eps = 0 at 1200 nm
(9.34e26 m^-3; eps(9.86e26) = -0.22+0.32i), so the film crosses Re eps = 0 within the last 0.013 nm at the
gate face, where the optical response is most sensitive to the profile shape, and a uniform 0.19 nm grid
with linear interpolation plus the slicer's endpoint averaging smears that crossing. The same sensitivity
leaves the raw-node result itself mesh-dependent at the 3.5e-3 level (2x finer mesh), still 14x below the
32-point error.

**Root cause.** A fixed, uniform, coarse resample sits between a graded mesh and a slicer that would accept
the native nodes.

**Proposed fix.**
1. `solve(..., grid_n_z=None)`: `None` means "native": the 2-D DEVSIM mesh is rectilinear (built from mesh
   lines), so return `grid_axes_m` = the unique node x and z coordinates of the region and `grid_fields` on
   that tensor grid without interpolation. Keep an integer `grid_n_z` for consumers that need a uniform grid
   (NGSolve `VoxelCoefficient`), and let the layered bridges (`slice_eps_field` accepts non-uniform z) use the
   native grid.
2. Thread the grid choice through `run_pipeline` (e.g. `carrier_solve_kwargs: dict | None`).
3. Warn when a uniform grid spacing exceeds 2x the smallest mesh spacing inside a semiconductor region.
4. Regression test: fixture B at +5 V; the sheet density from `grid_fields` must equal the raw-node trapezoid
   to 1e-12 with the native grid (today +7.4e-3 at 32 points).

## C2. Equilibrium edge ground pins the potential on the whole edge column (CONFIRMED, Medium)

**What.** An `Electrode(..., "x_lo", role="ground")` on an equilibrium film becomes a DEVSIM contact over the
film's full z range at x = 0, and `PE.setup_contact` imposes Potential = 0 there. In equilibrium mode the
Fermi level is already 0 everywhere (n = N_c F_1/2((psi - Phi_c0)/V_t)), so the contact adds nothing except a
Dirichlet column that distorts the gated profile next to it. In a periodic unit cell this is a grounded line
every period.

**Where.** `devsim_layered.py:668-675` (contact slab over the full layer z range) and `:757-758`
(`PE.setup_contact`, Potential only); `physics_equilibrium.py:252-263`. The comment at `:669-672` says DEVSIM
"captures only ~2 box-corner nodes here"; the measurement below shows the whole column is pinned.

**Reproduction** (fixture B, fine mesh, cell 40 nm):

```python
for ground in (True, False):
    b = LayeredDevsimBuilder(film_design(ground=ground, **FINE)); b.build_device()
    b.solve(BiasPoint({"top": 5.0, "bot": 0.0}, "b"), grid_n_x=8)
    for x in (0, 4e-9, 8e-9, 12e-9, 20e-9): print(ground, x, column(b, x)[1])
```

Observed at +5 V, as max over z of |n_ground - n_no_ground| / max(n_no_ground):

| x from the contact | 0 nm | 4 nm | 8 nm | 12 nm | 20 nm |
|---|---|---|---|---|---|
| deviation | 0.696 | 0.163 | 0.0411 | 0.0124 | 1.3e-3 |
| n at the top interface, with ground (no ground: 9.8591e26) | 3.0000e26 | 1.1465e27 | 1.0264e27 | 9.9811e26 | 9.8719e26 |

Without the ground every column equals the centre column to 2e-15. At 0 V the two agree to 6e-11.

**Why it matters.** A laterally uniform device acquires a 10-nm-wide disturbance at every cell edge; at the
contact the accumulation layer vanishes (n = n_bg at the interface) and 4 nm away it overshoots by 16 %. In a
40 nm cell this is a quarter of the area.

**Proposed fix.** For an equilibrium semiconductor region, an edge ground should be a reference only: do not
install the Potential Dirichlet (or refuse with an explanation, since no ground is needed for well-posedness
in equilibrium mode), and correct the comment. If a pinned edge is wanted (a real lateral contact), make it
explicit (`Electrode(..., pin_potential=True)`). Regression test: fixture B with the ground, +5 V: all
columns equal to 1e-12 relative.

## C3. `rel_tol <= 1e-7` does not converge at 0 V (WORKS AS DESIGNED, documented)

**What.** DEVSIM's relative update test is |dx| / (|x| + 1e-10). At 0 V every potential is 0; and in the
screened interior of a degenerate film the potential stays close to 0 at any bias. There the test becomes
|dx| < rel_tol x 1e-10, which round-off does not meet.

**Where.** `carriers/dc_solve.py:176-182` and `:296-308` (`ZERO_UNKNOWN_NOTE`, printed with the failure).
`LayeredDevsimBuilder.solve` also always runs a zero-bias seed solve at the user's `rel_tol`
(`devsim_layered.py:1083-1091`) unless a previous bias converged.

**Reproduction** (fixture B, default mesh, no ground):

| call | result |
|---|---|
| `solve(Vt=0, rel_tol=1e-5)` | converged |
| `solve(Vt=0, rel_tol=1e-6)` | converged (the lead said it fails; not on this device) |
| `solve(Vt=0, rel_tol=1e-7)` | failed: "film/PotentialEquation (abs 1.62e-17, rel 1.29e-07, relative test on a ~0 unknown: abs(x) <= 2.6e-11)" |
| `solve(Vt=0.5, rel_tol=1e-7)` and `1e-8` | failed at the seed (1e-7: the same numbers; 1e-8: rel 3.58e-7 on the film potential) |
| seed at 1e-5, then continuation to 0.5 V at `rel_tol=1e-8` / `1e-10` | failed (rel 3.57e-7 on the film potential) |
| `solve(Vt=0 or 0.5, rel_tol=1e-5, abs_tol=1e-14)` | converged |

**Conclusion.** Documented, and the failure message explains it. The usability gap is that no
`rel_tol` below a few times 1e-7 can ever be used on such a device, while an absolute criterion in volts
(`abs_tol=1e-14`, stronger than rel 1e-7 x 0.5 V) works. Suggested: document in `solve()` that for an
equilibrium-only device (Potential the only unknown) a tight convergence target is set with `abs_tol` in
volts; optionally map `rel_tol` to `abs_tol = rel_tol x max|V_electrode|` there.

## C4. Default 2-D mesh grading leaves 19 nodes across a 6 nm film (CONFIRMED, High)

**What.** `Mesh2DSpec` defaults (interface 0.05 nm, bulk 0.5 nm, zone 0.8 nm) are tuned for thick films. On a
6 nm film the grading leaves 19 nodes and under-resolves the accumulation and depletion layers.

**Where.** `geometry/specs.py:27-34` (defaults); `devsim_layered.py:597-614` (mesh lines:
`zone = min(interface_zone_m, 0.45 thk)`).

**Reproduction** (fixture B, no ground, centre column):

```python
for mesh in ({}, FINE, dict(interface_ps_m=6.25e-12, coarse_ps_m=2.5e-11, interface_zone_m=2.25e-9)):
    b = LayeredDevsimBuilder(film_design(**mesh)); b.build_device()
    b.solve(BiasPoint({"top": Vt, "bot": 0.0}, "b"), grid_n_x=8); z, n = column(b, 20e-9)
```

| top gate | mesh | nodes in film | n at top interface | excess sheet [m^-2] |
|---|---|---|---|---|
| +5 V | default | 19 | 9.67920e26 | 4.60803e17 |
| +5 V | 0.0125 / 0.05 / 2.25 nm | 189 | 9.85913e26 | 4.59252e17 |
| +5 V | 0.00625 / 0.025 / 2.25 nm | 374 | 9.85939e26 | 4.59250e17 |
| -3 V | default | 19 | 2.29985e25 | -2.69604e17 |
| -3 V | 189 nodes | 189 | 2.06913e25 | -2.68840e17 |
| -3 V | 374 nodes | 374 | 2.06846e25 | -2.68837e17 |

Optical size (strip cell, laterally uniform film, the default-mesh solution linearly interpolated onto the
fine z nodes so that only the carrier error remains): r = 0.101913-0.388075i vs 0.080680-0.363533i,
**|dr| = 0.032**, 7.1 % of the 0.457 modulation. Slicing the 19 default nodes natively gives |dr| = 0.043.

**Why it matters.** 1.8 % low interface density in accumulation, 11 % high in depletion, and 0.03 in r, with
default settings.

**Proposed fix.** Make the refinement thickness-aware: for a semiconductor layer thinner than about 10 nm,
default to `interface_zone_m = 0.45 t`, `coarse_ps_m <= t / 100` and `interface_ps_m <= t / 400` (the 189-node
setting; on this profile, whose face sits at the zero of Re eps, it is 3.5e-3 in r from the 2x finer mesh),
or add a `Mesh2DSpec.auto_thin_film=True` that does it; warn when fewer than about 50 nodes span a
semiconductor layer. Regression test: fixture B at +5 V with default `Mesh2DSpec` must give the
top-interface density within 1e-3 of the 374-node value.

## C5. Electrostatics uses a parabolic band with the DOS mass frozen at n_bg (CONFIRMED, Medium)

**What.** `TransportModel.dos_mass_kg_of_n_m3` is documented as a "callable of n; used for the
conduction-band Nc / Fermi level" (`materials/transport_model.py:76-77`), but the builder evaluates it once,
at n_bg (`devsim_layered.py:703`), and the equilibrium closure is a parabolic F_1/2 with that one N_c
(`physics_equilibrium.py:157-166, 215-217`). The optical side offers a density-dependent Kane mass
(`materials/scattering.py:27`, `KaneOpticalMass`). A nonparabolic band therefore enters the optics but not the
charge distribution, and a density-dependent callable is silently reduced to a constant.

**Reproduction** (library closure vs a Kane band with the same DOS mass at n_bg; band-edge mass 0.22 m_e,
alpha 0.4 eV^-1, T -> 0 for the Kane side):

```python
from dynameta.carriers.physics_equilibrium import effective_dos_m3, F12_aymerich_humet, invert_F12, V_T
# E_F(n_bg) = 0.5992 eV (Kane), DOS mass at n_bg = m0 (1 + alpha E_F) = 0.2727 m_e
Nc = effective_dos_m3(0.2727 * M_E); eta0 = invert_F12(3e26 / Nc)       # eta0 = 23.16
n_lib = Nc * F12_aymerich_humet(eta0 + psi / V_T)                       # library at band bending psi
```

| band bending psi | library n [m^-3] | Kane n [m^-3] | Kane / library |
|---|---|---|---|
| 0.1 V | 3.781e26 | 3.966e26 | 1.049 |
| 0.3 V | 5.513e26 | 6.335e26 | 1.149 |
| 0.5 V | 7.451e26 | 9.329e26 | 1.252 |
| 0.8 V | 1.070e27 | 1.511e27 | 1.412 |

The +5 V interface of C1 (n = 9.859e26 m^-3 with the fixture's 0.30 m_e closure) sits at 0.66 V of band
bending, computed from the same closure.

**Why it matters.** At the accumulation peak the local density-potential relation is off by 15-41 %, which
reshapes the profile (the screening length) even where the gate fixes the total charge; the optics then
applies a Kane mass to a density computed without it.

**Proposed fix.** Add `TransportModel.band = "parabolic" | KaneBand(m0_kg, alpha_eV)`. For the equilibrium
closure, n(eta) for a Kane band at finite T is a 1-D integral that can be tabulated once per material and
passed to DEVSIM as a fitted expression (the code already does this for F_1/2 with the Aymerich-Humet form);
`Phi_c0` follows from inverting it at n_bg. Share the same `KaneBand` with `KaneOpticalMass` so both sides use
one band. Until then, warn when `dos_mass_kg_of_n_m3` is not constant over [0.1, 10] n_bg. Regression tests:
`KaneBand(alpha=0)` bit-identical to today; the closure reproduces the T -> 0 Kane density above to 1e-3 at
eta >> 1.

## C6. No work function / flat-band voltage or fixed charge for an equilibrium film (CONFIRMED, Medium)

**What.** `Electrode` has no work function (`geometry/electrode.py:21-27`); a gate contact pins
Potential = bias + the heterostack frame offset, so the flat-band voltage of every gate is 0 by construction.
A fixed interface charge exists (`physics_traps.InterfaceTraps.fixed_charge_C_m2`) but is refused unless one
side is a bipolar drift-diffusion region. There is no bulk oxide charge. Thin interface dielectric layers are
not a gap: they are ordinary `Layer`s.

**Reproduction** (fixture B):

```python
from dynameta.carriers.physics_traps import InterfaceTraps
LayeredDevsimBuilder(film_design(), interface_traps={("film", "up_oxb"): InterfaceTraps(
    charge="none", recombination=False, fixed_charge_C_m2=-1.6e-3)}).build_device()
```

Observed: `ValueError: install_interface_traps: neither region of interface 'film__up_oxb' (['film',
'up_oxb']) is a bipolar drift-diffusion region ... the SRH occupancy of an interface trap needs both n and p
at the surface.` (`physics_traps.py:1097`).

**Why it matters.** A real gate stack has a metal-semiconductor work-function difference and oxide charge of
order 1e12-1e13 cm^-2; both shift each gate's response by a constant voltage that users currently have to
emulate by shifting biases, and a fixed charge cannot be emulated that way when two gates share the film.

**Smallest design.**
1. `Electrode(work_function_eV: float | None = None)`; for a gate contact, Potential = bias - (phi_m -
   phi_s)/q with phi_s = chi + (E_c - E_F) of the semiconductor it gates (requires `chi_eV`, as the
   heterostack code already does).
2. A charge-only interface sheet for any semiconductor or dielectric interface:
   `LayeredDevsimBuilder(fixed_charge={("film", "up_oxb"): Q_C_m2})`, implemented as an interface node charge
   (DEVSIM `interface_equation` with a constant `PotentialEquation` source), independent of carriers; and an
   optional uniform bulk charge per dielectric region (`node_model` source in the Poisson row).
3. Regression tests: a 1-D MOS with only a work-function difference must equal the same device biased by
   -delta phi / q bit for bit; a fixed sheet Q must shift the 1-D capacitor's threshold by Q / C_ox to 1e-9.

## C7. DEVSIM `debug_level="error"` is more verbose (NOT REPRODUCED as a DynaMeta defect)

`devsim.set_parameter(name="debug_level", value=...)` before building and solving fixture B: "error" and
"verbose" print several times more lines than no call or "info" (the line counts not re-measured on the
generic fixture). DEVSIM treats "error" like "verbose". DynaMeta never
sets or documents `debug_level` (no occurrence in the package), so this is DEVSIM behaviour. A library-level
convenience (a `quiet=True` option that routes DEVSIM's per-iteration report away) would help, but there is
no defect.

---

## Q1. `solve_self_consistent` ignores `phi_left_V` / `phi_right_V` under a warm start (CONFIRMED, Medium)

**What.** With `U_init_J` given, the initial potential is `-U_init_J / q` on the full grid, end nodes
included, and the Newton loop updates only interior nodes, so the Dirichlet values are whatever the warm start
carried. The requested end potentials are silently ignored and the solve reports `converged=True`.

**Where.** `carriers/schrodinger_poisson.py:333-335` (initial phi) and `:381-389` (interior-only update).

**Reproduction:**

```python
from dynameta.carriers.schrodinger_poisson import SchrodingerPoisson1D
from dynameta.constants import HBAR, Q_E
z = np.linspace(0, 6e-9, 241); m = 0.30 * M_E; sp = SchrodingerPoisson1D(z, m)
EF = HBAR**2 / (2 * m) * (3 * np.pi**2 * 3e26) ** (2 / 3)
kw = dict(eps_r=9.0, doping_m3=np.full_like(z, 3e26), E_F_J=EF, bound_tol=1e9, n_states=80, tol_V=1e-6)
phi1, n1, _ = sp.solve_self_consistent(phi_left_V=0.0, phi_right_V=0.5, **kw)
phi2, n2, r2 = sp.solve_self_consistent(phi_left_V=0.0, phi_right_V=1.0, U_init_J=-Q_E * phi1, **kw)
phi3, n3, _ = sp.solve_self_consistent(phi_left_V=0.0, phi_right_V=1.0, **kw)
```

Observed: `phi2[-1] = 0.5000` (1.0 requested), `r2.converged = True`; max|n2 - n3| / max(n3) = 0.388; n2 equals
the 0.5 V solution n1 to 3.7e-7.

**Why it matters.** Any outer loop that warm-starts on changing boundary potentials (a bias sweep, a gate
map, a dual-gate charge-matching loop) silently re-solves the old boundary problem.

**Proposed fix.** After building the initial phi, set `phi[0] = phi_left_V` and `phi[-1] = phi_right_V`
(in the bulk-buffer branch, the far buffer end); if the warm start's end values differ from the requested
ones by more than `tol_V`, ramp or just overwrite. Prototype of the logic: writing the requested ends into
`U_init_J` before the call reproduces the cold-start result. Regression test: the reproduction above must give
`phi2[-1] == 1.0` and n2 equal to n3 to 1e-6.

## Q2. `SchrodingerPoissonCarrier` is single-gate; 14 % spurious film charge at 0 V (CONFIRMED, High)

**What.** The carrier solves one column with phi = 0 at the body (z = 0) and the surface potential at the
gate side (`sp_carrier.py:144-155`), with a hard wall (psi = 0) at the body unless `neumann_body=True`. A thin
film gated from both sides has an oxide wall and a gate on both faces; the class cannot represent the second
gate, and on a 6 nm film the body wall alone removes 14 % of the film charge at 0 V. The class docstring
(`:35-38`) calls the body-side depletion "small (~0.4 nm)", which is true for a thick slab only.

**Reproduction:**

```python
from dynameta.carriers.sp_carrier import SchrodingerPoissonCarrier
for neu in (False, True):
    c = SchrodingerPoissonCarrier(semi_thk_m=6e-9, n_bg_m3=3e26, m_eff_kg=0.30 * M_E, eps_static=9.0, nz=481,
                                  oxide_thk_m=8e-9, eps_oxide=20.0, neumann_body=neu)
    cf = c.solve(BiasPoint({"gate": 0.0, "body": 0.0}, "g"))
    reg = cf.regions["semi"]; n = reg.grid_fields["electron_density_m3"][0, 0]
    print(neu, np.trapezoid(n - 3e26, reg.grid_axes_m["z"]))
```

| body boundary | 0 V sheet excess vs n_bg t | +2 V | -2 V | -4 V |
|---|---|---|---|---|
| Dirichlet (default) | -2.54e17 m^-2 = -2.54e13 cm^-2 (-14.1 %) | -2.3 % | -100 % | -100 % |
| Neumann | -1.41e16 m^-2 (-0.8 %) | +11.0 % | -93.2 % | -99.2 % |

The -100 % at negative bias is not a bracketing failure: the gate map returns psi_s = -13.19 V at -2 V and
-15.20 V at -4 V (Dirichlet body) without a warning; that is a sign error, reported separately as N4.

**Why it matters.** At 0 V the film carries 2.5e13 cm^-2 too few electrons, comparable to the charge a few
volts of gate bias induces (+2 V induces 2.1e13 cm^-2 here), so every comparison with a classical solve
starts from a different film. Neither gate can be represented on the far side.

**Smallest design.** `DualGateSchrodingerPoisson(t_film, ox_bottom=(t, eps), ox_top=(t, eps), ...)`:
Schrodinger with hard walls on both faces (the physical oxide barriers) and Poisson on the film with
Dirichlet values (phi_b, phi_t) that an outer 2-unknown Newton/secant loop adjusts until the displacement
field matches each oxide, eps_ox (V_g - phi_face) / t_ox = eps_s dphi/dz at the face (the gates' work
functions per C6). It reuses `SchrodingerPoisson1D.solve_self_consistent` unchanged once Q1 is fixed (the
outer loop warm-starts on new end potentials). The E_F reference should be the film's own bulk relation of
the SP model (see Q3). Emit `CarrierField(ndim=3)` as today. Regression tests: symmetric biases give a
symmetric profile to 1e-10; Gauss's law at both faces to 1e-9 C/m^2; a 40 nm film at 0 V reproduces n_bg in
the centre to 1e-3; for a thick film with the bottom gate grounded and far, the top-gate response tends to the
present single-gate class (Neumann body).

## Q3. Kane SP fill gives 0.90 n_bg in the bulk at the carrier's E_F (CONFIRMED, 0.896 not 0.85)

**What.** With `alpha_np_per_eV > 0`, the sub-band fill is in-plane Kane with a parabolic mass along z
(`schrodinger_poisson.py:214-226`), but the carrier takes E_F from the full 3-D Kane relation
(`sp_carrier.py:125-136`). The two bands differ, so the flat-band fill at that E_F is not n_bg. At T -> 0 the
ratio is (1 + 0.8 alpha E_F) / (1 + alpha E_F)^(3/2) (derivation: integrate the separable 2-D Kane density
over k_z); for alpha = 0.4 eV^-1, E_F = 0.4600 eV that is 0.890.

**Reproduction** (m 0.30 m_e, n_bg 3e26, 40 nm slab, nz 1601, 400 states; the first three columns as
re-measured on the v0.15.1 export for `2026-11-14-v0.16-sp-core.md`, the last column with the Q2 call at
alpha 0 / 0.4):

| alpha [eV^-1] | carrier E_F | flat-band fill, centre n / n_bg | self-consistent centre potential at 0 V | 6 nm film, 0 V sheet deficit |
|---|---|---|---|---|
| 0 | 0.5447 eV | 1.0029 | -0.0010 V | -14.1 % |
| 0.4 | 0.4600 eV | **0.8958** | **+0.0323 V** | -16.2 % |

In the self-consistent solve the bulk density is restored (centre n -> n_bg; the centre ratio itself not
re-measured on the generic fixture) by a spurious +32 mV of band bending; a thin film keeps 2.1 points more
deficit.

**Why it matters.** A 32 mV built-in offset per film and a different 0 V charge, both artefacts of the band
mismatch, enter every bias.

**Proposed fix.** Calibrate E_F on the SP model's own bulk relation: at finite T, solve
n_sep(E_F) = (1/pi) Int dk_z (m0 / (pi hbar^2)) [kT F_0(eta) + 2 alpha kT^2 F_1(eta)], eta = (E_F - hbar^2 k_z^2 /
2 m0) / kT, equal to n_bg (1-D quadrature + root), and document that the z dispersion is parabolic. A fuller
fix (energy-dependent z mass, a nonlinear eigenproblem) is not needed for the bulk limit. Regression test:
40 nm slab, alpha 0.4: centre potential |phi| < 1 mV and flat-band centre n / n_bg within 5e-3 of 1.

## Q4. `IntersubbandEffect` counts the free-carrier z response twice (CONFIRMED, High)

**What.** `eps_zz = eps_xx(Drude) + sum_{i<j} S_ij / (eps0 (w_ij^2 - w^2 - i w gamma))`
(`core/effects/electroabsorption.py:573-574`, code `:629-653`). When the SubbandResult holds the film's
sub-bands (slab mode, all states kept, as `SchrodingerPoissonCarrier` produces), the intersubband oscillator
strengths already exhaust the f-sum rule: their sum equals the full plasma response along z. Adding the Drude
term adds it a second time.

**Reproduction** (flat-band slab, m = m_opt = 0.30 m_e, n_bg 3e26, E_F from the Q1 expression, eps_inf 4.0,
gamma 1.2e14 both, 1200 nm):

```python
from dynameta.core.effects.electroabsorption import IntersubbandEffect
z = np.linspace(0, t, int(t / 0.025e-9) + 1); sp = SchrodingerPoisson1D(z, 0.30 * M_E)
res = sp.density(np.zeros_like(z), EF, n_states=min(z.size - 3, 600), bound_tol=1e9)
e = IntersubbandEffect(eps_inf=4.0, m_opt_kg=0.30 * M_E, gamma_intra_rad_s=1.2e14,
                       gamma_inter_rad_s=1.2e14).eps({"subband": res}, 1200e-9)
```

| film | eps_xx (Drude) | library eps_zz | eps_inf + intersubband only |
|---|---|---|---|
| 6 nm | 2.9549+0.0799i | 1.7922+0.2212i | 2.8373+0.1413i |
| 20 nm | 2.7822+0.0931i | 1.5505+0.2079i | 2.7683+0.1148i |

Sum rule, computed with the quadrature consistent with the eigenvector normalization (see N3):
sum_pairs (n_i - n_j) f_ij / sum_i n_i = **0.9997** for the 6 nm film.

**Why it matters.** In the thick-film limit eps_zz must approach eps_xx (an isotropic bulk); the library gives
1.551+0.208i against 2.782+0.093i, while eps_inf + intersubband gives 2.768+0.115i. The error is the whole
free-carrier term, about -1.23 in Re eps_zz.

**Proposed fix.** `eps_zz = eps_inf + sum_ISB` when the sub-band set is complete (slab mode), keeping the
current form only for an isolated well whose continuum is not in the sum. Make it explicit:
`IntersubbandEffect(z_response="subbands" | "drude_plus_subbands")`, default "subbands", and check the f-sum
(warn if the weighted sum of f_ij is outside [0.9, 1.05] in "subbands" mode, which flags a truncated state
set). Regression test: a 20 nm flat-band slab must give |eps_zz - eps_xx| < 0.05 at 1200 nm; the f-sum ratio
of a 6 nm slab within 1e-3 of 1 (requires N3).

## Q5. `dg_correct_density_1d` on a thin film: accumulation removed, charge moved to the far side (CONFIRMED, Medium)

**What.** The frozen-potential density-gradient closure puts a hard wall at one end and pins the other end to
the classical value (`density_gradient.py:307-308, 336-337`). On a 6 nm film the far end is also an oxide
wall, which the function cannot represent, and with gamma = 1 the quantum length (1.28 nm, Boltzmann closure)
exceeds the classical accumulation width. `conserve_charge=True` then rescales the whole profile by one
factor (`:348-349`), which spreads the lost interface charge over the film.

**Reproduction** (classical profile from fixture B, fine mesh, top gate +5 V; gate side at z = 6 nm):

```python
from dynameta.carriers.density_gradient import dg_correct_density_1d, GAMMA_WALL
for g in (1.0, GAMMA_WALL):
    for cc in (False, True):
        ndg = dg_correct_density_1d(z, n, 0.30 * M_E, gamma=g, hard_wall="right", conserve_charge=cc)
```

| case | peak position | peak n | centre n | sheet [m^-2] |
|---|---|---|---|---|
| classical | z = 6.00 nm (gate side) | 9.86e26 | 3.11e26 | 2.259e18 |
| gamma 1, no rescale | z = 0 (far side) | 3.00e26 | 2.91e26 | 1.442e18 |
| gamma 1, conserve_charge | z = 0 (far side) | 4.71e26 | 4.56e26 | 2.259e18 |
| GAMMA_WALL, conserve_charge | z = 0 (far side) | 5.26e26 | 4.57e26 | 2.259e18 |

For a symmetric two-sided accumulation profile (constructed as n(z) + n(6 nm - z) - n_bg), the un-walled end
is pinned to its full classical peak (9.86e26; 1.23e27 after the conserve_charge rescale) and the centre
rises to 4.1e26 with conserve_charge.

**Why it matters.** The correction does not move the accumulation peak 1 nm into the film, it deletes it
(36 % of the film charge with gamma = 1) and, with conserve_charge, raises the bulk by 47 % and puts the
maximum at the opposite interface. The module header (audit C-8) already states the closure is Boltzmann,
frozen-potential and that gamma is a fit; what is missing is a guard for the regime where it fails.

**Proposed fix.** (1) `hard_wall="both"` (two Dirichlet u = 0 ends; the BVP is unchanged otherwise);
(2) raise or warn when the film is thinner than about 10 L_q or when the frozen-potential deficit exceeds
about 10 % of the excess charge, pointing to the in-Newton DG (`physics_density_gradient.setup_dg_quantum_correction`,
which exists but is not wired into `LayeredDevsimBuilder`); (3) longer term, wire that in-Newton DG into the
layered builder for equilibrium and unipolar films with insulator walls on both faces. Regression test: the
case above must warn; `hard_wall="both"` on a flat profile gives a symmetric result to 1e-10.

## Q6. `nonlocal_tmm`: J_z = 0 walls between layers of one conductor (CONFIRMED, Medium)

**What.** Every `HydroLayer` imposes the hard-wall condition J_z = 0 at both of its faces
(`optics/nonlocal_tmm.py:81-90`, `_hydro_char_matrix_p` `:355-417`, `_layer_matrix` `:431-444`). Two adjacent
HydroLayers of the same conductor therefore get a wall inside the conductor, and a uniform film split into
identical sub-layers no longer equals itself. A graded density profile (the reason to split) cannot be
represented: a thin sub-layer between two walls has essentially no normal free-carrier current.

**Reproduction** (p-pol, 60 deg, air | 6 nm oxB (eps 4.0) | film | 6 nm oxB | glass (n 1.45), 1200 nm; film:
eps_inf 4.0, wp from n with m 0.30 m_e, gamma 1.2e14, beta = sqrt(3/5) v_F):

```python
from dynameta.optics.nonlocal_tmm import HydroLayer, DielectricLayer, stack_rt, eps_transverse, beta_from_vf
def hydro(n, t):
    wp = np.sqrt(n * 1.602176634e-19**2 / (8.8541878128e-12 * 0.30 * M_E))
    vF = 1.054571817e-34 * (3 * np.pi**2 * n) ** (1 / 3) / (0.30 * M_E)
    return HydroLayer(eps_inf=4.0, wp=wp, gamma=1.2e14, beta=beta_from_vf(vF), thickness_m=t)
```

| film | r | abs(r - r(one 6 nm layer)) |
|---|---|---|
| one HydroLayer 6 nm | 0.045905+0.025745i | 0 |
| two identical 3 nm HydroLayers | 0.045909+0.025745i | 4.19e-6 |
| ten identical 0.6 nm HydroLayers | 0.045885+0.022836i | 2.91e-3 |
| local Drude 6 nm | 0.045770+0.025745i | 1.35e-4 |

Top 0.5 nm sub-layer at density n_sub, remaining 5.5 nm at 3e26:

| n_sub [m^-3] | hydro + hydro | local + local |
|---|---|---|
| 3e26 | 0.045937+0.025716i | 0.045770+0.025745i |
| 6e26 | 0.046001+0.024788i | 0.045926+0.025397i |
| 9e26 | 0.046063+0.023861i | 0.050505+0.026059i |
| 1.2e27 | 0.046124+0.022935i | 0.046597+0.020973i |

**Why it matters.** Splitting a uniform film into ten sub-layers changes r by 21 times the entire nonlocal
correction (2.9e-3 vs 1.35e-4). On the generic film the two-layer split is much milder (4.2e-6, below the
nonlocal correction), so the original claim "any split exceeds the nonlocal correction" holds for the
ten-layer split only; the wall is still a wrong boundary condition in both. With the split, the response to
the sub-layer density is linear in n to three digits (Im r steps -9.28e-4, -9.27e-4, -9.26e-4 per
3e26 m^-3), i.e. it acts only through the tangential permittivity; the strongly nonlinear normal-field
response that the local model shows (Im r 0.0257 -> 0.0254 -> 0.0261 -> 0.0210; the 9e26 m^-3 sub-layer is
near Re eps = 0) is gone.

**Proposed fix.** Interface conditions between hydrodynamic layers of the same conductor: J_z continuous and
the pressure term continuous (beta^2 div J / wp^2, i.e. the induced-charge pressure, for equal-beta layers
div J itself), with J_z = 0 only at conductor/insulator faces. This needs the 4x4 transfer matrix (2
transverse + 2 longitudinal amplitudes) kept through a run of same-conductor layers and collapsed to 2x2 only
at the run's outer faces; mark runs with `HydroLayer(conductor_id=...)`. Until then, raise when two
HydroLayers are adjacent. Regression tests: one 6 nm layer equals two 3 nm and ten 0.6 nm identical layers
to 1e-12; a run of layers with a smooth density ramp converges as the slab count grows.

## Q7. No nonlocal/quantum response for a graded layer in a structured cell; no 2-D TM cross-check solver (CONFIRMED, missing feature)

**What exists** (read from the source):
- `optics/hydro_fem.py:557-560` `hydro_layered_1d(omega, params, d_nm, ...)`: one uniform film in vacuum.
- `hydro_fem.py:889-893` `scattering_2d(mesh, omega, params, ...)`: an isolated uniform scatterer in a host
  with a PML (no periodicity, no density gradient).
- `nonlocal_tmm`: uniform planar layers (Q6).
- `IntersubbandEffect`: one uniform (3,3) tensor per slab ("v1 returns a UNIFORM (3,3) slab response"; a
  z-graded eps_zz(z) is a documented follow-on, `electroabsorption.py:600-604`).
- `optics/solver.py:698` `solve_fem`: 3-D periodic HCurl only.
- `optics/fdtd_nd/solve2d.py:510-520` `solve_fdtd_2d`: periodic 2-D cell, TE only (E along the lines), normal
  incidence; `:889-903` `solve_fdtd_2d_oblique`: has TM but laterally uniform stacks only.

**Why it matters.** For a gated film under strips in TM there is no independent solver of the same 2-D
problem in the library (local or nonlocal), and no way to give the graded layer a nonlocal or quantum
optical response inside the structured cell. Every cross-check has to be built outside the library.

**Smallest design.**
1. A 2-D (x, z) periodic TM FEM: `solve_fem_2d_tm(period_x, layers, eps_xz_callable, lambda, theta)` with
   NGSolve H1 for H_y on a 1-D-periodic mesh (`ng.Periodic` on x), first-order absorbing or PML in z, and
   `eps` given as a (possibly tensor) coefficient so a z-graded anisotropic film (eps_xx(z), eps_zz(z)) fits.
   It shares the mesh/PML/Bloch-phase helpers of `solver.py` and gives an independent check of RCWA/PMM.
2. A nonlocal extension of the same weak form: add the scalar longitudinal potential psi of
   `scattering_2d` (already stabilized, `hydro_fem.py:84-130`) with a position-dependent wp^2(x, z) and
   beta(x, z) in the metal/semiconductor region and J.n = 0 only on its outer boundary (Q6's physics in 2-D).
3. Optionally, `IntersubbandEffect` emitting eps_zz(z) from the local |psi(z)|^2 weighting (its documented
   follow-on) so the layered bridges can slice an anisotropic graded film.
Regression tests: laterally uniform cells reduce to `nonlocal_tmm` / TMM to 1e-8; the strip cell agrees with
the converged PMM reference r_ref to 1e-3.

---

## N1. RCWA bridge ignores a gridded EpsField's x coordinates (CONFIRMED, Low)

**What.** For a laterally structured gridded EpsField the RCWA bridge passes the values to lumenairy as
`eps_cell` without looking at `x_axis_u` (`rcwa_backend.py:395-417`). lumenairy reads `eps_cell[j]` as the
value at x = j P / S. `assemble_eps` produces endpoint-inclusive axes (`core/resample.py:40-43`, `linspace`
from the first to the last node; `core/bridge.py:201-210` remaps them onto [0, P]), so the last sample is the
periodic image of the first, the profile is compressed by (S - 1)/S, and relative to the strip layer
(analytic, or raster at cell centres) it is misregistered by up to P/(2S) (0.78 nm for S = 256). Also, the
default `grid_n_x = 256` caps the usable order count at 63 (lumenairy needs S >= 4N + 1): at N = 100 the solve
raises "the unit-cell sampling (256, 1) is too coarse for n_orders_x=100".

**Reproduction** (fixture A, film lamellar field as in O1 with 256 or 3200 samples, `formulation='li'`,
`n_orders=50`; on lumenairy 5.50.2 the strip layer itself is analytic under 'li'):

| field x samples | r |
|---|---|
| endpoint-inclusive `linspace(0, P, 256)` (what `assemble_eps` produces) | 0.043164-0.664119i |
| node lattice j P / 256 (what lumenairy assumes) | 0.036104-0.669596i |
| cell-centred (j + 0.5) P / 256 | 0.043164-0.664119i |
| node lattice j P / 3200 | 0.042585-0.664572i |
| cell-centred (j + 0.5) P / 3200 | 0.043152-0.664126i |

(For this two-value profile the endpoint-inclusive and the cell-centred 256-sample grids happen to give the
same sample values.)

**Why it matters.** The endpoint-inclusive grid moves r by 8.9e-3 relative to the node-lattice reading of the
same profile (1.3 % of the 0.68 lamellar-split effect, |r(lamellar) - r(uniform)|); smaller than O3/O4, but a
deterministic registration error of a carrier profile against its own gate.

**Proposed fix.** Resample the field onto the RCWA lattice using its coordinates: drop a duplicated periodic
end sample, interpolate periodically onto the same lattice the bridge uses for the inclusion layers (cell
centres, so all raster layers share one convention), with at least `_default_raster_samples(N)` samples.
Regression test: a lamellar EpsField given on three different x grids (endpoint-inclusive 256, node 256,
cell-centred 3200) gives r within 2e-3 of the cell-centred-3200 result, and N = 100 no longer raises.

## N2. Bridges do not check an EpsField's z extent against the layer thickness (CONFIRMED, High)

**What.** `slice_eps_field(ef, metres_per_unit)` multiplies the field's z axis by `metres_per_unit` (the
bridges pass 1e-9: axes in nm, `pmm_backend.py:214`, `rcwa_backend.py:400`). A field whose axes are in
metres becomes slabs 1e9 times too thin, and the layer silently disappears from the optical stack. No bridge
compares the slab total with `Layer.thickness_m`.

**Reproduction** (fixture A, PMM bridge degree 24, a z-graded laterally uniform field on 11 z points):

```python
zn = np.linspace(0, 6, 11)                                    # nm
eg = [eps_film(N_BG * (1 + 3 * np.exp(-(6 - z) / 0.6))) for z in zn]
f = lambda z: EpsField(x_axis_u=np.array([0, P * 1e9]), y_axis_u=np.array([0.0]), z_axis_u=z,
                       values_zyx=np.repeat(np.asarray(eg, complex)[:, None, None], 2, axis=2))
s = make_lumenairy_pmm_solver(degree=24)
s(strip_design(), None, {"film": f(zn)}, LAM, None, None).r          # -0.2350-0.4605i
s(strip_design(), None, {"film": f(zn * 1e-9)}, LAM, None, None).r   # -0.8494+0.0352i, no warning
```

Observed: the metre-axis field gives 10 slabs totalling 6.0e-18 m for a 6e-9 m layer and r =
-0.849411+0.035236i, equal (to 1.6e-7) to the same cell with the film thinned to 1e-15 m.

**Why it matters.** A unit slip in a bring-your-own EpsField (the documented pluggable seam) deletes the
tunable layer with no error; r changes by 0.79 here.

**Proposed fix.** In both bridges (and `tmm_reference` / Berreman, which share `slice_eps_field`), after
slicing a layer: `if abs(sum(s.thickness_m for s in slabs) - L.thickness_m) > 1e-6 * L.thickness_m: raise
ValueError(...)` naming the layer, both thicknesses and the expected axis unit. Regression test: the metre
case above raises; the nm case is bit-identical.

## N3. `IntersubbandEffect` dipole quadrature is inconsistent with the eigenvector normalization (CONFIRMED, Low)

**What.** The sub-band eigenvectors are normalized with a Riemann sum (`sum |psi|^2 h = 1`,
`schrodinger_poisson.py:167-169`), but `z_ij` is integrated with the trapezoid rule on the absolute z
coordinates (`electroabsorption.py:646`). The two quadratures differ at the end nodes, so the discrete states
are not orthogonal under the trapezoid rule and z_ij picks up (z offset) x (overlap error): the result
depends on where z = 0 is.

**Reproduction** (6 nm flat-band slab, 241-point grid, 150 states, as in Q4):

| z origin shift | library eps_zz | f-sum ratio, trapezoid z_ij | f-sum ratio, Riemann z_ij |
|---|---|---|---|
| 0 | 1.79221+0.22118i | 1.0865 | 0.9997 |
| +20 nm | 1.79227+0.22118i | - | - |
| +100 nm | 1.79337+0.22121i | 53.99 | 0.9997 |

**Why it matters.** eps_zz moves by 1.2e-3 for a 100 nm origin shift (the contaminated pairs are mostly
high-energy ones, suppressed at optical frequencies), but the f-sum diagnostic that Q4 needs is unusable.

**Proposed fix.** Use the same quadrature as the normalization (`np.sum(psi_i * z * psi_j) * h`) and
measure z from the slab centre. Regression test: eps_zz invariant to a 100 nm origin shift to 1e-12; f-sum
ratio of the Q4 slab within 1e-3 of 1.

## N4. `SchrodingerPoissonCarrier` depletion: sign error in the oxide charge term (CONFIRMED, High; fix prototyped)

**What.** `_gate_to_psi_s` solves |V_g| = psi + q N_exc / C_ox with psi the surface-potential MAGNITUDE and
N_exc the signed electron excess (`sp_carrier.py:177-181`). Gauss's law is V_g = psi_s + q N_exc / C_ox with
SIGNED quantities; for depletion (s = -1) the charge term must carry s. Without it the residual never
crosses zero in [0, |V_g|], the bracket expands and the bisection converges on a root between -12 and
-15 V here, with no warning.

**Reproduction** (6 nm film, oxide 8 nm / eps 20, as in Q2):

```python
c = SchrodingerPoissonCarrier(semi_thk_m=6e-9, n_bg_m3=3e26, m_eff_kg=0.30 * M_E, eps_static=9.0, nz=481,
                              oxide_thk_m=8e-9, eps_oxide=20.0)
cf = c.solve(BiasPoint({"gate": -1.0, "body": 0.0}, "g")); print(cf.extras["surface_potential_V"])
```

| V_g | library psi_s | Gauss check psi_s + q N_exc / C_ox | prototype psi_s | prototype Gauss check |
|---|---|---|---|---|
| +2 V | +0.4570 V | +1.9978 V | +0.4576 V | +2.0000 V |
| -1 V | **-12.1934 V** | **-23.39 V** | -0.2504 V | -1.0000 V |
| -2 V | **-13.1914 V** | **-24.39 V** | -0.5230 V | -2.0000 V |

(N_exc relative to the flat-band profile, as the method defines it; the +2 V check is off by 0.002 V only
because of the 8-halving bisection, Vg/256. Library columns re-measured on the v0.15.1 export and prototype
columns = the sign-fixed map `gate_map="flat_band_excess"`, both as recorded in
`2026-11-14-v0.16-sp-core.md`.)

**Why it matters.** Every negative gate bias fully depletes the film (the -100 % rows of Q2) instead of
depleting 2.0e17 m^-2 (11 % of n_bg t) at -2 V (C_ox (V_g - psi_s) / q with the prototype psi_s).

**Fix (prototyped in a scratch subclass, not in the library).** In `residual`, use
`psi + s * Q * n_exc / self._C_ox - Vg`. With it the bracket [0, |V_g|] holds for both signs (no expansion
needed for these cases) and Gauss's law is met to the root-finder tolerance (the sign-fixed map whose values
are tabulated above solves to 1e-9 V; 1e-4 V shown). Also tighten the bisection to a voltage tolerance
rather than a fixed 8 steps. Regression test: for V_g in {-2, -1, +1, +2} V, |psi_s + q N_exc / C_ox - V_g| <
1e-3 V.

---

## Prioritised fix order

1. **N4** (one-character sign fix, prototyped; every depletion bias of the SP carrier is wrong).
2. **N2** (one thickness check in the shared slicing path; a unit slip deletes the tunable layer silently).
3. **O3** (TM-on-metal default or warning in the RCWA bridge; the default answer is off by 0.17-0.50 in r at
   N = 100-50).
4. **C4** and **C1** (thin-film mesh default and the native carrier grid, threaded through `run_pipeline`;
   together 0.032 + 0.050 in r on default settings).
5. **Q4** and **N3** (intersubband double count; the f-sum check needs the consistent quadrature).
6. **C2** (equilibrium edge ground as a reference, not a Dirichlet column).
7. **Q1** (warm-start ends), a prerequisite for the dual-gate SP in Q2.
8. **Q3** (SP E_F calibrated on the SP model's own bulk relation).
9. **O4** and **N1** (edge-aligned or area-weighted raster; coordinate-aware resampling of gridded fields).
10. **Q5** (thin-film guard and `hard_wall="both"`), **Q6** (same-conductor hydrodynamic interfaces).
11. Missing features, in order of what they unblock: **Q2** dual-gate Schrodinger-Poisson, **C6** work
    function and fixed charge, **C5** a shared Kane band for electrostatics and optics, **O1** lamellar graded
    layers in the PMM bridge, **Q7** a 2-D TM FEM (local, then nonlocal).
12. Documentation/convenience: **C3** (tight convergence via `abs_tol` in volts), **O2** (`pmm_kwargs`
    pass-through), **C7** (a quiet option for DEVSIM output).
