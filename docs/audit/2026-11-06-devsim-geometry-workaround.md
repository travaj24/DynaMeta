# A DynaMeta-side workaround for DEVSIM's unsigned box-method geometry

DEVSIM phase, release v0.14.0, branch `wip/v014_ecfix` (off `feat/devsim-2.11-v0.14` at 3ba3a2a). DEVSIM 2.11.2
(extended-precision build), Windows dev box. Nothing was filed or posted upstream (the user is holding
the DEVSIM issue). The draft and the source excerpts are in the session scratchpad `devsim_verify/`.

**Bottom line.**
- **Feasible, and done for 2-D and 3-D.** The new module is `dynameta.carriers.devsim_geometry`, with
  public function `apply_exact_geometry(device, region, *, negative=...)`.
- **Now automatic (user decision 2026-11-06, section 7).** Both builders apply it right after
  `create_device` with `negative="warn"` (`exact_geometry=True`; `False` / `"raise"` / `"clip"`).
  Library meshes are only read, so they stay byte-identical. `mesher="gmsh"` stays and is corrected.
- **What it does.** It recomputes DEVSIM's own circumcentric construction with the orientation sign
  of each piece kept, using the node coordinates and the element lists DEVSIM reports. It writes the
  result into DEVSIM's built-in `ElementEdgeCouple`. DEVSIM's dependency chain then rebuilds
  `EdgeCouple`, `ElementNodeVolume`, `EdgeNodeVolume` and `NodeVolume` with its own code. In 3-D the
  module also overwrites `SurfaceArea` / `ContactSurfaceArea` with the signed shares.
- **What it fixes.** Every equation DEVSIM assembles (node, edge, element, contact, interface) uses
  exact control volumes. So does every DynaMeta kernel that reads these models by name.
- **Measured on the legacy gmsh 3-D stack** (`validation/devsim_geometry.py`):
  - series capacitor: 1.203 x before, exact to 6e-16 after;
  - bipolar MOS cap, Gauss residual: 0.228 before, 1.3e-5 after;
  - sum NodeVolume: 1.196 / 1.293 x before, 1 + 2e-16 after.
- **Unchanged where DEVSIM is already exact.** On Kuhn tets and rectilinear right triangles nothing
  moves beyond round-off.
- **Boundary caveat: real, measured, and handled by a policy.**
  - The exact geometry can have negative ASSEMBLED couples and volumes. On the gmsh stack about 1.0 %
    of the edges are negative (124 / 13473 and 130 / 12746, worst -0.053 L^2; the validation table in
    section 4 -- corrected 2026-11-06, an earlier draft said 1.3 % / -0.12 L^2) and no node volumes are. A random-point Delaunay square
    reaches -24 L and -6.7 h^2.
  - Default `raise`: refuse and report, device untouched.
  - **Recommended `warn`**, once the report has been read: exact values plus `UnreliableResultWarning`.
  - `clip`: negatives set to 0, with the clipped fractions measured and returned.
- **Cross-branch finding (the noise row, not this branch).** The per-element Hooge source is biased
  in 3-D on EVERY mesh, including the default Kuhn tensor mesh: **S f N / (alpha I^2) = 1.125**
  (MEASURED). The cause is that its tensor identity does not hold per element for circumcentric
  couples in 3-D (section 5). FIXED in v0.14.0 (merge d08412e, docs/audit/2026-11-06-v0.14-hooge-3d.md:
  P1 element couples; the Kuhn bar now reads 1 - 5.8e-9).

## 1. Feasibility: what DEVSIM lets a caller replace (read in src/, then tested)

| Mechanism | Source | Honoured in 2.11.2? | Used? |
|---|---|---|---|
| Region parameters `node_volume_model`, `edge_couple_model`, `element_edge_couple_model`, `edge_node0/1_volume_model`, `element_node0/1_volume_model` (defaults in `GlobalData.cc:424-430`, read by `Region.cc:1457-1534`) | `Equation.cc:833/912/920/1249/1267`, `ExprContactEquation.cc:61-205` | Yes, 2-D and 3-D. MEASURED: `set_parameter(device, region, ...)` on a node / edge / element *solution* model is returned by `get_parameter` and assembled. | No (see below) |
| `surface_area_model` (interfaces only, `Interface.cc:412`, `InterfaceExprEquation.cc:66`) | | Yes | No |
| `set_element_values` / `set_edge_values` / `set_node_values` on the BUILT-IN models | `ModelCommands.cc:871ff`: works on any existing model. `SetValues` sets `uptodate` and `SignalCallbacks` marks the dependents old | Yes. MEASURED: writing `ElementEdgeCouple` makes DEVSIM rebuild `EdgeCouple`, `ElementNodeVolume`, `EdgeNodeVolume`, `NodeVolume` from it on the next read | **Yes** |

Why the built-ins and not the parameters:
- The parameter route corrects only DEVSIM's assembly.
- DynaMeta reads the geometry by its built-in names in: `noise_analysis` (Hooge per element:
  `ElementEdgeCouple`, `EdgeCouple`, `NodeVolume`), `impact_ionization` (`ElementNodeVolume`),
  `physics_traps` (`ItArea = A / NodeVolume`, computed once at install), `tunneling`
  (`ElementNodeVolume`, `EdgeNodeVolume`, `SurfaceArea`) and the ii2d branch's new
  `element_geometry` check.
- With the parameter route, the discretisation and these post-processors would disagree.

DEVSIM also reads `ElementEdgeCouple` itself in two other places:
- **2-D element-field reconstruction** (`TriangleElementField::GetFieldPairs`, used by
  `element_from_edge_model` / `element_pair_from_edge_model`). It uses the couples as averaging
  weights:
  - With signed couples the two weights of each edge still sum to
    2R sin(theta_i/2) cos((theta_j - theta_k)/2) > 0.
  - For the edge field of a P1 potential the result is unchanged. MEASURED: <= 1e-12 relative on an
    obtuse Delaunay mesh (test). Each node-based vector already equals the element gradient.
- **3-D "COUPLE" node interpolation** (`TetrahedronEdgeModel.cc:199`). This is only for showing
  element values on nodes.

3-D element fields (`TetrahedronElementField`) do not use couples.

DEVSIM's cylindrical mode (`TriangleCylindricalEdgeCouple.cc:153`) takes the same magnitude. It is
not corrected here, and the module refuses a region that already selects non-default geometry
models. No DynaMeta module uses the cylindrical mode (grep).

## 2. The construction and its checks

Kernels (numpy):
- `triangle_dual_pieces`: s_j is the signed distance from the midpoint of edge j to the
  circumcentre, positive toward the opposite vertex. DEVSIM's 2-D couple is |s_j|.
- `tetrahedron_dual_pieces`: per tet edge, 0.5 * sum over its two faces of s1 * s2.
  - s1 is the face's triangle piece (the triangle kernel is reused).
  - s2 is the signed distance from the face circumcentre to the tet circumcentre along the face
    normal, positive toward the fourth vertex.
  - DEVSIM's value is 0.5 * sum |s1| |s2|, which is the same right triangles as
    `TetrahedronEdgeCouple.cc:99`.

The per-element-edge slot layout is not assumed:
- It is read from DEVSIM: the temporary element models `edge_index` and `node_index@en0/1`.
- It is checked against the element node list.

Before anything is written, DEVSIM's current `ElementEdgeCouple` must re-derive from the coordinates
as the unsigned value, to 1e-9 L^(d-1). After the write, `EdgeCouple` and `NodeVolume` are read back
and must equal the signed sums. Either failure raises `RuntimeError`; neither fired anywhere. A
region parameter `dynameta_exact_geometry` marks a corrected region (`exact_geometry_applied()`).
`measure_dual_geometry()` reports without writing.

## 3. Gates (tests/test_devsim_geometry.py, 14 tests, 8 s; independent oracles; all MEASURED)

| Gate | Oracle | DEVSIM's own geometry (shown failing) | Corrected |
|---|---|---|---|
| 2-D kernel = cotangent weight | L cot(theta)/2 with cot = u.v/\|u x v\| (P1 FEM), cross-product area | obtuse triangle: sum \|s\| L/2 = 11.11 x area; > 1 on all 20+ obtuse samples | <= 6.4 eps (budget 64 eps (L + 2\|cot\|)) |
| 3-D pieces partition each tet | \|det\|/6 | corner tet 2.000; 5-tet cube 1.6667 | <= 1e-11 on 300 random tets; exact on both |
| Interior duals = Voronoi | Qhull `Voronoi` + `ConvexHull` | interior cell up to > 1.05 x (asserted) | cells to 1e-10, facets to 1e-9 (>= 20 cells, >= 50 facets) |
| sum NodeVolume = 2 sum ENV = sum EC L/d = V | \|det\|/d! | corner 2.0, 5-tet cube / lattice 1.6667, obtuse triangle 11.11, jittered 2-D > 1.01 | <= 1e-13 |
| Well-centred meshes unchanged (Kuhn 3^3, rectilinear 2-D) | itself (consistency) | n/a | geometry change <= 1.15 eps h^k (budget 64), capacitor potential <= 2.1e-17 V (budget 1e-13) |
| Capacitor: phi = z, C = 1, Gauss Q = 1 | closed form | 5-tet 3^3: phi exact, C = Q = 1.6667; jittered 3-D: phi 3.3e-2 off, C 1.2553, Q 1.2640; jittered 2-D: 1.5e-2, 1.0389, 1.0519; two regions: 3.2e-2, 1.2295, 1.2342 | phi <= 1.1e-16, \|C-1\|, \|Q-1\| <= 2.2e-16 |
| 3-D SurfaceArea / ContactSurfaceArea | facet cross-product areas | 1.0549 (interface), 1.0285 / 1.0681 (plates) | 1 to 1e-13 |
| DD resistor G = q N_D mu A/L (FD-SG, 1 mV) | closed form | 5-tet lattice 1.6667; jittered Delaunay 1.2553 | 1 - 1.1e-16, 1 + 2.2e-16 |
| Policies | recount | random square: 4 negative couples (worst -23.8 L), 4 negative volumes (-6.65 h^2) | raise leaves the device byte-identical; warn exact + warned; re-apply idempotent; clip leaves no negatives, removes 0.284 of sum\|EC\|, adds 14.1 V |
| Consumers | impact_ionization's own condition 2 sum ENV = V | 1.6667, refused | passes |

Each gate carries one deliberate perturbation that must break it:
- writing the unsigned values back restores 5/3;
- scaling the highest-flux EdgeCouple by 1.5 breaks the patch test (> 1e-4);
- DEVSIM's |s| fails the partition;
- the unsigned Voronoi comparison fails.

An observation from the 5-tet lattice capacitor: DEVSIM still reproduces phi = z exactly there,
because the over-count scales the couples of every z-edge uniformly, but C and Q are both 5/3. The
patch test alone does not reveal the defect.

**Validation** (`validation/devsim_geometry.py`, PASS 8/8 via run_all, 86 s). Mesh: the legacy
`Stacked3DSpec(mesher='gmsh')` stack, 12 nm cell, semi 30 nm, oxide 8 nm, roughly 9.4-9.8k tets per
region. The correction is hooked right after the builder's `create_device`.

| | semi | oxide |
|---|---|---|
| non-well-centred tets | 4265 / 9811 | 4200 / 9352 |
| edges with a negative exact couple | 124 / 13473 (worst -0.028 L^2) | 130 / 12746 (worst -0.053 L^2) |
| negative node volumes | 0 | 0 |
| DEVSIM sum NodeVolume / V | 1.1957 | 1.2932 |
| corrected (warn) | 1 + 2.2e-16 | 1 + 2.2e-16 |

| Problem | DEVSIM geometry | `warn` | `clip` |
|---|---|---|---|
| series capacitor C / C_exact | 1.203236 | 1 - 5.6e-16 | 1 + 6.8e-4 |
| max \|phi - piecewise-linear exact\| | 2.98e-2 V | 3.3e-16 V | 2.1e-3 V |
| bipolar MOS cap Gauss residual, 0.5 V | 0.228 | 1.28e-5 | 7.6e-4 |
| psi_s lateral spread | 4.97 mV | 0.21 mV | 0.31 mV |

The bipolar FD-SG Newton converged under `warn` despite the negative couples.

The G3 bound is derived. The geometric gate charge replaces the laterally varying psi_s under the
oxide by its mean, which moves it by at most eps_ox * spread / t_ox * L^2 (the "spread term"):
- `warn`: residual 1.28e-5 <= term 5.25e-4;
- `clip`: 7.63e-4 <= 7.73e-4, a narrow pass;
- DEVSIM's own geometry: 0.228 > 1.23e-2, fails as required.

The spread itself is an unstructured-mesh discretisation error, not a geometry error.

**Why `warn` is the recommendation and `raise` the default.**
- `warn` was the most accurate policy on the realistic mesh, by 60x over `clip`, and the
  M-matrix risk did not materialise.
- `raise` stays the default so that the first call on an unknown mesh prints its negative counts
  instead of choosing for the caller. Pathological meshes do exist: on the random-point square,
  `clip` adds 14x the region volume.
- To make `warn` the default instead, change one keyword.

## 4. Integration with the shared well-centredness check (branch wip/v014_ii2d, read-only)

`dynameta/carriers/element_geometry.check_element_geometry` refuses any element with
|2 sum ENV / |element volume| - 1| > 1e-9.
- It reads DEVSIM's `ElementNodeVolume`.
- The signed pieces partition every element exactly, so after `apply_exact_geometry` the check
  passes with no change on either side.
- MEASURED by loading that file read-only:
  - 5-tet lattice: 32/40 elements inexact (worst 2.0) before, 0 after (max deviation 4.4e-16);
  - jittered 3-D Delaunay: 298/434 inexact (worst 9.24) before, 0 after (2.4e-15);
  - jittered 2-D Delaunay: 24/128 inexact (worst 3.27) before, 0 after (2.4e-15).

The check's refusal message could add one line: "or call
`devsim_geometry.apply_exact_geometry(device, region, negative='warn')` after create_device".

**To make the correction the default for user meshes** (as proposed on 2026-11-06; done, section 7):
1. Call it inside the builders right after `create_device`. For `Devsim3DEquilibrium` that is the
   end of `_create_mesh_device`; the validation's hook is exactly that. On the tensor and Kuhn
   meshes it is a round-off no-op (gated).
2. Decide the default policy: `warn`, or `raise` with a remesh message.
3. Re-record no digests. Library meshes move only by round-off (MEASURED <= 1.2 eps h^k), and,
   INFERRED (not run), the call-log digests do not contain geometry.
4. Retire the `mesher='gmsh'` UnreliableResultWarning only if the user wants that. With the
   correction the gmsh path is exact, so v0.15's removal of it could be revisited.
5. Note in the release notes that 3-D `SurfaceArea` changes on obtuse interface triangles.
   `physics_traps.interface_node_area` keeps barycentric thirds on obtuse triangles, so its
   `SurfaceArea` equality claim then holds only on non-obtuse surfaces. Its circumcentric formula is
   the same kernel as `triangle_dual_pieces` and could call it.

Cost: one pass over the elements in numpy, about 0.35 s for a 10k-tet region (MEASURED), plus one
`set_element_values`.

## 5. Cross-branch finding: the per-element Hooge source is biased in 3-D (not fixed here; fixed in v0.14.0 by d08412e)

`noise_analysis` assembles Hooge per element as S_T ~ |sum_e c_eT a_e dy_e|^2 / V_T. Its docstring
relies on "sum_e c_eT L_e t_e t_e^T = V_T I on a well-centred element".
- **In 3-D that identity does not hold per element for circumcentric couples** (MEASURED).
  - Only its trace does: sum c L / 3 = V_T.
  - Kuhn path tet x, y, z: diag(0.75, 1.5, 0.75) V_T.
  - Well-centred random tets: eigenvalues 0.88-1.11 V_T.
  - It does hold in 2-D, where the circumcentric couple is the cotangent (P1) weight.
- **Consequence.** For an axis-aligned uniform current on a Kuhn mesh, each axis is the middle step
  in 2 of the 6 path tets (m = 1.5) and an end step in 4 (m = 0.75), so S picks up
  <m^2> = (2 * 2.25 + 4 * 0.5625) / 6 = 1.125.
- **MEASURED with noise_analysis itself** (scratch `hooge3d.py`; `validation.devsim_noise.unipolar_3d_resistor`,
  the default Kuhn bar):

  | Bar | S f N / (alpha I^2) | I / (G V) |
  |---|---|---|
  | Kuhn bar | **1.125000** | 1.000000 |
  | 5-tet bar, DEVSIM geometry | 1.000000 | 1.666667 |
  | 5-tet bar, corrected | 1.000000 | 1.000000 |

- **Fix for the noise owner.** Use the per-element P1-FEM edge weights
  c^FEM_eT = -V_T L_e grad(lambda_i) . grad(lambda_j) in the Hooge element sum. They satisfy the
  tensor identity per element in any dimension and equal the circumcentric couples in 2-D. On a
  Kuhn mesh they assemble to the same 7-point weights (derived by hand, not run).
- No current gate catches this: the Hooge gates are 1-D and 2-D, and the ii2d check passes Kuhn
  meshes.

## 6. Files

- `dynameta/carriers/devsim_geometry.py` (new): kernels, `apply_exact_geometry`,
  `measure_dual_geometry`, `exact_geometry_applied`, `ExactGeometryReport`,
  `NegativeDualGeometryError`.
- `tests/test_devsim_geometry.py` (new, 16 tests), added to CI `solvers-devsim` (8 s).
- Section 7 (automatic application): `dynameta/carriers/devsim_layered.py`,
  `dynameta/carriers/devsim_3d.py`, `dynameta/carriers/element_geometry.py`,
  `tests/test_devsim_3d_mesh.py`, `validation/devsim_3d_mesh.py`, `validation/devsim_geometry.py`, docs.
- `validation/devsim_geometry.py` (new, 8 gates).
- `.github/workflows/ci.yml`: the test module, with its measured time.

At first no existing module changed; section 7 wires the builders. No digests moved.

## 7. Automatic application (user decision 2026-11-06; branch `wip/v014_geomdefault` off 432ba47)

The decision: apply the correction automatically, default negative policy `"warn"`, keep the gmsh
mesher (no removal in v0.15).

**Wiring.**
- `devsim_geometry.correct_device_geometry(device, exact_geometry=True)` runs every region through
  `apply_exact_geometry` and merges the regions' negative-couple warnings into ONE
  `UnreliableResultWarning` per device.
  - With `"raise"` it measures every region first, so a refusal leaves the whole device unchanged.
  - `resolve_policy` maps the option: True -> `"warn"`, `"raise"` / `"clip"` as named, False -> off.
- `LayeredDevsimBuilder(..., exact_geometry=True)` calls it right after `create_device`, before
  `XP.enable`.
- `Devsim3DEquilibrium._create_mesh_device` calls it right after `create_device`, through
  `Stacked3DSpec.exact_geometry` (field, default True; `from_design(..., exact_geometry=)`).
- These are the only two `create_device` calls in `dynameta/`.
- The reports are on the builders as `geometry_reports`.
- `apply_exact_geometry` itself now defaults to `negative="warn"`.

**Byte-identity on library meshes: nothing is written.**
- `geometry_already_exact(device, region)` decides first, using devsim `get_*` calls only.
  - It recomputes the pieces per element in local order.
  - It requires DEVSIM's `ElementEdgeCouple` to re-derive per element (as a multiset, to 1e-9
    h_T^(d-1)).
  - It requires no piece to be negative beyond 64 eps h_T^(d-1). The derivation: a zero piece
    measured <= 1.15 eps h^k, and the smallest genuine negative seen anywhere is -1.5e-4 L^2.
  - In 3-D it requires the same of every interface / contact triangle share.
- When it holds, `apply_exact_geometry` returns `written=False` with no write. That means no temporary
  model, no marker parameter and no value write.
- Gates:
  - `tests/test_devsim_geometry.py::test_already_exact_meshes_are_only_read`: a recorder wraps every
    non-`get_` devsim function; zero calls on the 3-D Kuhn lattice with contacts and on the 2-D
    rectilinear mesh. Adversarial: on the 5-tet lattice it records `set_element_values`.
  - `tests/test_devsim_3d_mesh.py::test_builders_only_read_the_library_meshes`: a tripwire on the
    write path stays silent while building three tensor topologies (full gate; patch +
    multi-dielectric; semi inclusion + patch) and the 2-D layered builder. Adversarial: it fires on
    the gmsh mesher.
- MEASURED, `geometry_already_exact` = True on every region of the six 3-D tensor topologies and the
  Kuhn bar:
  - full gate, patch 0.5, multi-dielectric, buried-oxide inclusion, semi inclusion + patch, 60 nm patch
    cell;
  - cost 0.01-0.54 s per region, 1.85 s for the largest (108,864 tets).
- The call-log digests and the SG-flux cross-version gate build raw-API 1-D devices, which the
  builders' hook never sees. Their tests pass unchanged (section 7, tests).

**gmsh mesher: kept, and what moves for its users (MEASURED, `validation/devsim_geometry.py` and
`validation/devsim_3d_mesh.py` M6a/M6b).**

| | `exact_geometry=False` (pre-v0.14 numbers) | default (`"warn"`) | `"clip"` |
|---|---|---|---|
| sum NodeVolume / V, semi / oxide | 1.1957 / 1.2932 | 1 + 2.2e-16 | -- |
| series capacitor C / C_exact | 1.203236 | 1 - 5.6e-16 | 1 + 6.8e-4 |
| max\|phi - exact\| | 2.98e-2 V | 3.3e-16 V | 2.1e-3 V |
| bipolar MOS cap Gauss residual (0.5 V) | 0.228 | 1.28e-5 | 7.6e-4 |
| lateral psi_s spread | 4.97 mV | 0.21 mV | 0.31 mV |

Warnings:
- The mesher's own `UnreliableResultWarning` now fires only with `exact_geometry=False`. Its new
  text names "pre-v0.14" and "exact_geometry=False"; there is no "v0.15".
- By default a gmsh build emits the one merged negative-couple warning instead.

Pins changed, each with a numeric cross-check:
- `test_legacy_gmsh_mesher_keeps_the_defect_and_warns` becomes
  `test_gmsh_mesher_uncorrected_keeps_the_defect_and_warns` (exact_geometry=False: sum NV 1.1957 > 1.1,
  the reworded warning).
- New `test_gmsh_mesher_is_geometry_corrected_by_default`: sum NodeVolume and 2 sum ENV equal L^2 t
  for every layer to 1e-12, and SurfaceArea equals L^2.
- `test_3d_impact_ionization_refuses_an_inexact_mesh` now builds with `exact_geometry=False`. On the
  default gmsh build the element-geometry check passes.
- Validation M6 is split into M6a (False: defect + warning) and M6b (default: 1 to 1e-12).
- No digest moved.

**The element-geometry refusal** (`element_geometry.check_element_geometry`) now says the builders
correct automatically and names `apply_exact_geometry(device, region, negative='warn')` for
raw-API devices.

**Docs** that promised removal now say the mesher is kept, geometry-corrected:
- the `devsim_3d` module doc and the `Stacked3DSpec.mesher` comment;
- the release note (sections 2, 4, 6, 7);
- `2026-11-03-devsim-3d-mesh.md` section 8;
- `2026-11-04-v0.14-user-decisions.md` row 1;
- the v0.15 consolidated plan (rows on the release split and on the effort).

## Release-note lines

- Both DEVSIM builders now correct DEVSIM 2.11's over-counted control volumes and couplings on
  non-well-centred meshes automatically (`exact_geometry=True`; `False` restores the old numbers).
  Library meshes are untouched (byte-identical). On the gmsh 3-D mesher, which is kept, the
  capacitance goes from 1.20x to exact and the MOS-cap Gauss residual from 23 % to 1e-5.
- Raw-API devices: `dynameta.carriers.devsim_geometry.apply_exact_geometry(device, region)` right
  after `create_device`. Negative couples on non-conforming boundaries are kept and warned by default
  (`negative='warn'`); `'raise'` refuses, `'clip'` keeps the M-matrix property at a measured cost.
- (Resolved before release: the per-element 3-D Hooge 1/f source read 1.125x on Kuhn tensor meshes;
  fixed in v0.14.0, docs/audit/2026-11-06-v0.14-hooge-3d.md.)
