# The devsim_3d mesh / node-volume defect (D2c)

DEVSIM phase, release v0.14.0, workstream D2c, branch `wip/dv_mesh3d` (off `feat/devsim-2.11-v0.14` at
e46bd25). DEVSIM 2.11.2, gmsh 4.15.2, box A.

**Bottom line.**
- **Root cause (read in DEVSIM's source, reproduced in numpy to 5 digits).** DEVSIM 2.11's 3-D
  `ElementEdgeCouple` adds the MAGNITUDES of each tetrahedron's circumcentric dual pieces. On a tet whose
  circumcentre lies outside it, the signed pieces partly cancel, but their magnitudes add. gmsh's
  tetrahedra are 42-57 % such tets, so every 3-D volume and coupling was inflated (sum NodeVolume =
  1.196 x the semiconductor, 1.293 x the oxide on the D1c G2 cap).
- **"Make the mesh Delaunay" does not fix it.** A gmsh mesh with 0 non-Delaunay faces still gives
  1.289 x. **Element assembly does not fix it either**: it uses the same couples.
- **The fix is a mesh on which DEVSIM's formula is exact.** `Stacked3DSpec.mesher="tensor"` is now the
  default. It is a graded tensor-product grid through every layer, patch and inclusion boundary, with
  each box split into the 6 right-angled Kuhn tetrahedra. Every circumcentric piece is then >= 0.
  Volumes and couplings are exact to 9e-16 on all five builder topologies, and every equation term
  (node, edge, element, interface) sees the same exact geometry.
- **Gates.** All pass now, and all fail on e46bd25:
  - D1c G2 0.799 -> 0.989;
  - 3-D == 1-D on the same z-lines to 9 digits;
  - carrier-free sheet charge 0.827 -> exact;
  - Gauss residual 23 % -> 3e-16;
  - Ohm's law 14 % -> 8e-16;
  - lateral psi_s spread 5 mV -> 0.
- **Cost.** It is faster wherever it was measured, except on tiny patch or inclusion cells.
  - Full-gate cap: 4482 -> 575 nodes; bipolar solve 25.8 s -> 1.1 s.
  - Equilibrium cap: solve 1.0 s -> 0.1 s. The ~150 s build is DEVSIM's symbolic set-up and does not
    depend on the mesh.
  - 60 nm patch cell: 55397 -> 23273 nodes; DD solve 141 s -> 44 s.
  - Small 12 nm patch or inclusion cells: 2-3x MORE nodes, because the tensor grid carries lateral
    refinement through the whole cell (section 6).
- **Optional item done: 3-D element-based impact ionization.** It is exact on a uniform-field bar and
  equals the 1-D answer to 5e-10 on a laterally uniform diode.

## 1. Diagnosis

### 1.1 What DEVSIM computes (source, devsim/devsim main, `src/GeomModels`)

`TetrahedronEdgeCouple.cc` loops over each tetrahedron and its 6 edges. With `v0` = tet circumcentre
minus edge midpoint, and `v1`, `v2` = the circumcentres of the two faces sharing the edge minus the tet
circumcentre, it computes

    ElementEdgeCouple = 0.5 * ( |v0 x v1| + |v0 x v2| )

This is the area of the edge's circumcentric dual polygon inside the tet, taken as a sum of UNSIGNED
triangle areas. From it:
- `TetrahedronNodeVolume.cc`: `ElementNodeVolume = ElementEdgeCouple * EdgeLength / 6` (a pyramid of
  base = couple and height = L/2);
- `EdgeNodeVolume.cc` (3-D): the per-edge sum of `ElementNodeVolume`;
- `NodeVolume.cc`: the per-node sum of `EdgeNodeVolume`;
- `EdgeCouple`: the per-edge sum of `ElementEdgeCouple`.

So `NodeVolume`, `EdgeNodeVolume`, `EdgeCouple`, `ElementEdgeCouple` and `ElementNodeVolume` all
inherit the magnitude sum.

**Consequence.** The box method needs a SIGNED circumcentric (Voronoi) partition. That partition is
exact for any mesh, because the signed pieces of a tet sum to its volume. DEVSIM's unsigned sum is exact
only when no piece is negative, that is, when every tet is "closed well-centred": its circumcentre lies
in the closed tet, and each face circumcentre lies in its face.
- Delaunay is NOT sufficient. A Delaunay mesh guarantees a non-negative ASSEMBLED Voronoi coupling per
  edge (Bank and Rose, SIAM J. Numer. Anal. 24, 777 (1987); Selberherr, *Analysis and Simulation of
  Semiconductor Devices*, Springer 1984, ch. 6). The per-element pieces can still be negative, and
  DEVSIM takes them in magnitude.
- The DEVSIM manual says nothing on Delaunay or 3-D couples (https://devsim.net/models.html, Tables
  5.1-5.3).
- `element_model` / `ElementEdgeCouple` / `ElementNodeVolume` assembly is therefore NOT an escape: it
  uses the same unsigned couples. A custom signed (P1 cotangent) coupling would be exact in volume, but
  its couplings go negative on 43-74 % of these gmsh tets (table below). That breaks the
  Scharfetter-Gummel M-matrix property and the positivity of the carrier densities.

### 1.2 Measured (`validation/devsim_3d_mesh.py` building blocks; the G2 cap: 12 x 12 nm, Si 30 nm / SiO2 8 nm, mesh_min 0.5 nm)

DEVSIM's formula, re-implemented in numpy from the gmsh tets, reproduces DEVSIM's `NodeVolume` sum to
5 digits. The signed version sums to exactly 1.

| mesh (gmsh option) | region | tets | not well-centred | tets with a negative P1 coupling | non-Delaunay faces | DEVSIM sum NodeVolume / V | numpy unsigned / V | signed / V |
|---|---|---|---|---|---|---|---|---|
| default (Algorithm3D 1 = Delaunay, Optimize 1) | semi | 9811 | 47.4 % | 6810 | 242 / 18272 | **1.19567** | 1.19567 | 1.00000 |
| default | oxide | 9352 | 50.0 % | 6649 | 260 / 17478 | **1.29316** | 1.29316 | 1.00000 |
| Optimize 0 (unoptimised Delaunay) | oxide | 9520 | 50.5 % | 6809 | **0** / 17814 | **1.28917** | | 1.00000 |
| Optimize 0 | semi | 9998 | 48.0 % | 7001 | 12 / 18646 | 1.20751 | | 1.00000 |
| OptimizeNetgen 1 | semi / oxide | | 42.5 / 41.5 % | | 540 / 415 | 1.25766 / 1.24689 | | 1.00000 |
| Algorithm3D 4 (frontal) | semi / oxide | | 55.6 / 51.2 % | | 1174 / 625 | 1.29060 / 1.51703 | | 1.00000 |
| Algorithm3D 10 (HXT) | semi / oxide | | 56.3 / 56.6 % | | 373 / 349 | 1.35318 / 1.46978 | | 1.00000 |
| mesh_min 0.25 nm | semi / oxide | 56118 / 54989 | 44.1 / 44.1 % | | 1399 / 1368 | 1.20722 / 1.33004 | | 1.00000 |
| mesh_min 1.0 nm | semi / oxide | 2550 / 1862 | 51.7 / 53.1 % | | 67 / 73 | 1.22882 / 1.25457 | | 1.00000 |
| **tensor (D2c)** | all | 2016 | **0** | **0** | 0 | **1 + 4e-16** | 1 + 2e-15 | 1.00000 |

Notes on the table:
- The excess is set by the fraction of non-well-centred tets, not by mesh size. Refinement does not
  remove it, which matches D1c's 0.79 / 0.79 / 0.77 G2 ratios.
- No `EdgeCouple` is ever negative (magnitudes), which is why D1c saw "no negative EdgeCouple".
- The P1-cotangent column is the volume-exact element alternative, with its negative couplings.
- Every gmsh variant leaves the volume excess between 1.20 and 1.52.

### 1.3 Why pointwise psi and n survived but integrated charge did not

In Poisson's equation the bulk charge is integrated over the same inflated `NodeVolume` as the flux
couplings. For a laterally uniform 3-D cap both are inflated by similar factors, so their ratio, and
with it the pointwise solution, came out nearly right:
- trap-free psi_s matched 1-D to 0.5 mV;
- a gate-shift test reproduced -Q/C_ox.

A GEOMETRIC quantity breaks that balance:
- **Sheet charge.** A sheet charge with exact interface areas acts against inflated capacitances, which
  gives the 0.80 (D1c G2).
- **Integrated charge.** The volume-integrated charge was compared with the geometric gate charge
  eps_ox dV/t_ox. The Gauss residual was 23 %.
- **Current.** Current through inflated couplings is Ohm's law 14 % high (`validation/carriers_3d_resistor.py`
  had gated it at < 16 % as "mesh discretization").

## 2. The fix and why

**Options weighed.**
- (a) A Delaunay gmsh mesh fails (section 1.2).
- (b) DEVSIM's own element assembly fails, because it uses the same couples.
- (b') A custom signed element coupling is volume-exact but non-M-matrix, so SG loses positivity.
- (c) A mesh on which DEVSIM's unsigned formula IS the signed one.

devsim_3d's geometry is a union of axis-aligned boxes:
- layers;
- a centred gate patch;
- centred rectangular inclusions spanning a full layer.

So (c) is available exactly. That choice is `Stacked3DSpec.mesher="tensor"`, the new default
(`devsim_3d._build_mesh_tensor`). Section 1.1 explains why (a) and (b) fail.

- **Axes.** Each axis passes through every boundary: layer interfaces in z; patch and inclusion edges
  in x and y.
- **Grading.** `_graded_axis` applies the same rule as the gmsh Threshold field it replaces:
  mesh_min_nm within 1 nm of a refinement plane, linear to mesh_max_nm at 6 nm. Each span gets
  ceil(int ds/h) cells at equal steps of int ds/h.
  - z refines at the (last) semiconductor/dielectric interface, as before.
  - x and y refine at patch and inclusion edges. Otherwise they are uniform at <= mesh_max_nm, since a
    full-gate stack needs no lateral resolution.
- **Cells.** `_kuhn_tets` splits every box into the 6 Kuhn tets c -> c+e_p0 -> c+e_p0+e_p1 -> c+(1,1,1).
  - Every face of a Kuhn tet is a right triangle.
  - The tet circumcentre is the box centre, which is the midpoint of the tet's main-diagonal edge.
  - Each face circumcentre is a hypotenuse midpoint.
  - So every circumcentric piece is >= 0 (many are exactly 0), and magnitude equals signed.
  - The assembled couples are the rectilinear box-method ones: an axis edge carries its dual rectangle,
    and face and body diagonals carry 0.
  - The split of each box face along its min -> max diagonal is consistent between neighbours, so the
    mesh is conforming (tested).
- **Regions, contacts, interfaces.** Each box cell is classified by its centre (inclusions first). Each
  region pair gets an interface from the faces between differently labelled cells. Interface names are
  those of the gmsh builders (`if_k` stacked, sorted-pair enumeration with inclusions), and the
  `build_device` hand-off is unchanged.
- **Contacts on their own region only.** Contacts take only faces of their own region's cells. gmsh's
  "gate" and "body" groups also covered the faces of an inclusion reaching the top or bottom. That
  worked only because a coarse gmsh face had no interior nodes; on the tensor mesh DEVSIM refuses such
  nodes ("Contact ... reference coordinate not on region").
- **File.** The mesh is written as MSH 2.2 in metres (`_write_msh22`) and read by the unchanged
  `create_gmsh_mesh` path. gmsh is no longer needed on the default path.
- **Legacy mesher.** `mesher="gmsh"` keeps the old builders to reproduce pre-v0.14 numbers. It raises
  `UnreliableResultWarning` with the measured excess.

**Consistency across ALL equation terms.** With exact geometry, every assembly route integrates over the
same exact control volumes:
- node terms (SRH, traps, Auger/radiative, ionization, the D1c/D1b hook `add_charge_term` /
  `add_recombination_term`) over `NodeVolume`;
- edge terms over `EdgeNodeVolume`;
- element terms (D1d BTBT, the D2a noise element pair weights) over `ElementNodeVolume`;
- fluxes over `EdgeCouple`;
- heterojunction interface terms (D1a) through the interface node pairs.

Laterally uniform 3-D stacks become exactly laterally invariant (spread 0) and reduce to the 1-D system
on the same z-lines.

**One more term had to move: interface trap areas** (`physics_traps.interface_node_area`). D1c split each
3-D interface triangle into barycentric thirds.
- On the Kuhn faces that gives the two nodes on a face diagonal 1/3 of the face each, and the others
  1/6 each: a 2x checkerboard in a laterally uniform D_it. MEASURED on the G2 cap at Vg = 0.5 V:
  psi_s laterally spread by **5.0 mV** (against a 19.5 mV shift), and G3 off by 7.9e-5.
- The split is now circumcentric: node i gets (|e_ij|^2 cot theta_k + |e_ik|^2 cot theta_j)/8. That is
  the interface face of node i's box-method control volume. It equals DEVSIM's `SurfaceArea` node for
  node (tested, rtol 1e-12) and makes the spread 0.
- Obtuse triangles, which occur only on the legacy gmsh mesh, keep thirds, since their circumcentric
  shares go negative.
- 1-D and 2-D are unchanged.

**Validations that built their own gmsh mesh.** `validation/carriers_3d.py` and
`validation/carriers_3d_resistor.py` now build the same Kuhn mesh through the shared helpers.
`--gmsh` reruns the legacy mesh.

## 3. Gates (all MEASURED 2026-11-03; every one FAILS on e46bd25)

`validation/devsim_3d_mesh.py --base <e46bd25 worktree>`: **PASS 8/8**. Gate M7 runs M1-M5 in a
subprocess on the e46bd25 tree, and all 6 fail there.

| gate | oracle | this branch | e46bd25 |
|---|---|---|---|
| M1 volumes | sum NodeVolume, 2 sum ElementNodeVolume and sum EdgeCouple*EdgeLength/3 against the exact region volume; full gate, patch 0.5, multi-dielectric, buried-oxide inclusion, semi inclusion + patch | worst 8.9e-16; min EdgeCouple 0 | worst 0.41 (0.17-0.41) |
| M2 carrier-free sheet charge | both layers dielectric, both contacts 0 V, sigma = 1e-3 C/m^2: psi_if = sigma/(eps_si/d_si + eps_ox/d_ox) = 0.1287076 V at every interface node | 0.0 (every node) | mean 0.827, worst node 0.19 |
| M3 Gauss | bipolar cap at Vg = 0.5 V: q sum (p - n + N) NodeVolume + Q_body + eps_ox (psi_g - psi_s)/t_ox L^2 = 0 (the oxide term is exact for the charge-free planar oxide); DEVSIM's gate contact charge = the geometric one | residual 2.9e-16; gate charge rel 4.4e-16. Q_semi -1.650830e-3, Q_gate 1.746553e-3, Q_body -9.57e-5 C/m^2 | residual 0.23 (Q_semi -2.0326e-3); DEVSIM gate charge 1.23 x the geometric |
| M4 lateral invariance | psi_s max - min over the interface | 0.0 V | 5.0e-3 V |
| M5 / D1c G2 | D_it 1e16 eV^-1 m^-2 psi_s shift at Vg = 0.5 V, 3-D vs the fine 1-D cap (5 %: z-resolution) | -19.525 vs -19.747 mV, **ratio 0.9888** | ratio 0.799 |
| M5 / D1c G3 (new) | the same shift against the 1-D cap on the 3-D mesh's own z-lines (22 nodes); lateral spread | -0.019524920 V both (rel 0.0); spread 0 | no tensor z-lines (G2 already fails) |
| M6 legacy gmsh | `mesher="gmsh"` still over-counts and warns | 1.196, `UnreliableResultWarning` | -- |
| Ohm's law | `validation/carriers_3d_resistor.py`: 200 x 80 x 80 nm ITO bar, 20 mV, I against V sigma A/L | **8.3e-16** (gate now < 1e-6) | 0.142 (gate was < 0.16) |

Notes on the table:
- G2's remaining 1.1 % is entirely z-resolution. G3 against the 1-D cap on the same z-lines is exact,
  and the fine 1-D reference uses 0.05 nm at the interface against the 3-D mesh_min 0.5 nm.
- G2 is kept at 5 %, the D1c definition, because that tolerance is justified by the z-resolution. G3
  is the tight gate.

`tests/test_devsim_3d_mesh.py`: 11 tests, 22 s at `-n 2`. They cover:
- the graded axis;
- Kuhn conformity and closed well-centredness;
- circumcentric shares;
- spec guards;
- exact volumes, couplings and interface areas (== `SurfaceArea`) on the patch builder;
- M2, M3/M4, M6;
- the three impact-ionization tests (section 5).

## 4. Existing tests and validations; moved pins

**Tests.** 130 passed (`-n 2`, 323 s) across these files and tests:
- `test_stacked3d_spec`;
- `test_audit_2026_07_17_wave7_carriers`;
- the 3-D tests of `test_bipolar_recombination_bgn_ionization`, `test_devsim_heterojunction` (3-D DD
  heterostack), `test_devsim_tunneling` (W3 BTBT + oxide tunneling wiring) and `test_extended_precision`
  (the 3-D switch);
- `test_devsim_smoke`, `test_gated_builder`, `test_geometry_guards`, `test_traps`.

`test_impact_ionization` + `test_devsim_3d_mesh`: 15 passed. No test pinned a 3-D number that moved.

**Validations** (sequential, new tree then e46bd25; `[t]` lines quoted):

Every 3-D validation PASSES on both trees. The table lists each value that moved, with the
independent reference where one exists.

| validation | quantity | e46bd25 (gmsh) | D2c (tensor) | reference / why the new value is right |
|---|---|---|---|---|
| `carriers_3d_resistor` | I / I_Ohm - 1 | 0.142 (gate < 0.16) | **8.3e-16** (gate now < 1e-6) | Ohm's law closed form V sigma A/L |
| `carriers_3d` | lateral invariance (worst std/mean) | 8.96e-3 / 1.30e-2 | 1.2e-16 / 2.7e-16 (gate now < 1e-9) | a translation-invariant gate |
| `carriers_3d` | Gauss, box-method dQ/D_ox - 1 (new) | **+0.191 / +0.191** (the volume over-count) | -3.4e-6 / -3.7e-6 (gate < 1e-4; the residual of the script's rel 1e-5 Newton stop) | Gauss's law |
| `carriers_3d` | Gauss, trapezoid dQ/D_ox | 0.878 / 0.889 | 0.985 / 0.989 | the trapezoid differs from the box quadrature by 1.5 % in the 0.5 nm accumulation layer (printed, no longer gated) |
| `carriers_3d` | n_top/n_bg (+1 V) | 1.2629 | 1.2169 | mean over nodes within 1 nm of the top: a window statistic; gmsh's window holds interior nodes |
| `carriers_3d_dd`, `carriers_3d_field` | n_top/n_bg (ITO/HfO2, +1 V) | 1.366 | 1.373 | 1-D cap, same physics: **1.3740** at 0.5 nm interface spacing, 1.3916 converged (0.02 nm) |
| `carriers_3d_patch` (60 nm, frac 0.5) | under-patch n/n_bg; min Re eps | 1.326; 0.697 | 1.368; 0.584 | under the patch centre, 15 nm from its edge (ITO Debye length 0.18 nm), the full-gate 1-D value applies: 1.3740 / 1.3916 |
| `carriers_3d_inclusion` | under-pillar / edge n, ratio | 5.297e26 / 4.944e26, 1.071 | 5.609e26 / 5.016e26, 1.118 | no closed form; the same exactness as the full gate |
| `carriers_3d_multidielectric` | multi-dielectric vs equivalent single oxide (identical series C) | rel 4.63e-4 | **rel 2.9e-16** | the two stacks are the same 1-D problem |
| `carriers_3d_multisemi` | homojunction split (one region vs two identical) | rel 1.46e-2 | rel 4.8e-3 | should be 0 up to the different z-lines (the split adds a breakpoint) |
| `carriers_3d_multisemi` | n+/n built-in, max deviation | 8.0e-4 V | 4.2e-4 V | the Anderson closed form |
| `carriers_3d_bipolar` | areal n (bip / eq); n(ox)/n(body) | 6.618e16 / 6.674e16; 518 | 6.877e16 / 6.927e16; 409 | `n(ox)` is a mean over a 1 nm window (gmsh: interior nodes too); bip == uni unchanged (0.000) |
| `bridge_3d_field` | gate-side Re eps | 0.590 | 0.571 | follows n_top (above) |
| `pipeline_3d_end_to_end` | gate-side Re eps (+1 V); optical dR | 0.714; 2.17e-4 | 0.571; 2.32e-4 | gmsh's interpolation onto the grid's top plane from unstructured nodes; tensor nodes lie on that plane |
| `adversarial_pipeline_3d` | dR(+1 V - 0 V) | 9.4e-5 | 1.06e-4 | follows the accumulation |
| `devsim_heterojunction` gate F (3-D, reported, not gated) | max abs(TE - continuous)/n at Vg 0 / +0.5 V | 1.506e-9 / 1.412e-9 (integration note) | **2.9e-16 / 2.9e-16** | no DC current (gate on an oxide), so both interface models carry zero interface flux and must give the same profile (the gmsh residue of 1.5e-9 is gone) |
| `devsim_heterojunction` gate F | max abs(DD - equilibrium)/n at Vg 0 / +0.5 V | 2.758e-3 / 2.093e-3 | 4.18e-3 / 2.35e-3 | not a geometry measure: the FD g-fit of the DD bulk against the Aymerich-Humet closure of the equilibrium (the script's own caveat, 0.8 % on ITO); it depends on where the nodes sit. No independent reference, recorded only |
| `carriers_3d_from_design`, `from_design_multidielectric` | spec derivation only | -- | unchanged | -- |

Re-pinned gates, all with the evidence above:
- `carriers_3d_resistor`: < 0.16 -> < 1e-6.
- `carriers_3d`: lateral invariance 5 % -> 1e-9; Gauss trapezoid 15 % -> box-method 1e-4.
- `traps_oracles` G: G2 now passes, and G3 is new.

`--gmsh` keeps the old gates on both scripts. No other pin moved: the other validations gate loosely
(signs, ratios, 3 %), and every one still passes.

`validation/traps_oracles.py --3d` (D1c, default `--main` = main 5cbd998): **PASS 12/12**. On the
integration tree it was 10 of 11, with G2 failing.
- A-F are unchanged: A 2.09e-5, B 1.1e-15, C1 7.4e-5 V, C2 2.3e-4, C3 1.9e-6 V, D 7.7e-4; E, E' and F
  (byte identity against main) pass.
- G1 area exact (2e-16).
- G2 ratio 0.989.
- G3 rel 0.0, spread 0.

## 5. Optional: 3-D element-based impact ionization

`impact_ionization.impact_generation_edges` used to raise `NotImplementedError` on any 3-D device. Its
directional edge quadrature had read 1.8-3.1x low on gmsh tets (audit 7.1), which is this defect. In
3-D it now computes, per element edge,

    I_sub = sum (alpha_n(|E|) |J_n| + alpha_p(|E|) |J_p|) * 2 ElementNodeVolume

- **Vectors.** E, J_n and J_p are rebuilt per tetrahedron by `element_from_edge_model`. This is
  isotropic: no directional masking, and it is the same element field D1d's BTBT uses.
- **Geometry check.** It first checks 2 sum ENV against the true tet volumes from the element node list,
  and refuses an inexact mesh (> 1e-9). The legacy gmsh mesher is refused (test).
- **Side effects.** Element models it creates are deleted again. Pre-existing ones (BTBT's
  `ElectricField_x/y/z`, with derivatives) are only read.
- **2-D.** 2-D keeps its validated edge quadrature, and `validation/impact_ionization_isub.py` is
  unchanged.

Gates (MEASURED):

| gate | result |
|---|---|
| uniform-field oracle: 3-D n-Si bar (1e22 m^-3, 100 nm, 20 x 20 nm, 0.1 V), test coefficients a = 1e6 1/m, b = 2e5 V/m, against alpha(E) q N mu E Volume | rel < 1e-6 (passes) |
| 1-D vs 3-D: abrupt p+/n+ 1e24 diode, 200 nm, -3 V, graded z-lines 1 -> 8 nm, van Overstraeten-de Man Si; 3-D element I_sub/A against the 1-D edge quadrature on the same z-lines | 0.1556608697 A/m^2 both, rel **5e-10**; contact J rel 2e-16 |
| inexact mesh refused | `NotImplementedError` naming `mesher='tensor'` |

**Why the 1-D == 3-D identity is exact.**
- In a Kuhn tet of a laterally uniform solution, the P1 field is the z-cell's edge field.
- The SG current on a diagonal edge is (h/L_diag) x the z-edge current, which is an exact projection.
- The tet's 2 ENV sum is the tet volume.

**Tolerance.** With tau = 1e-7 the net reverse current is a ~1e-6 residue of the SG drift and diffusion
terms, and the reconstruction's round-off shows (rel 1.4e-6). The test therefore uses tau = 1e-10.

**Scope note.** The 3-D devices in these tests are two-contact bars built directly with the shared Kuhn
helpers. `Stacked3DSpec` is a gated capacitor with no DC current, so I_sub there is ~0.

## 6. Cost

| case | gmsh (e46bd25) | tensor (this branch) |
|---|---|---|
| G2 cap (12 nm, 0.5 nm), nodes | 4482 | 575 |
| bipolar build + solve at Vg 0.5 V (M3) | 2.75 s + 25.8 s | 3.17 s + 1.13 s |
| D1c gate G (2 x 3-D + 4 x 1-D) | -- | 15-23 s |
| patch 0.5 (12 nm), nodes / DD build | 4464 / 1.68 s | 9200 / 2.31 s |
| buried-oxide inclusion, nodes / DD build | 4657 / 1.70 s | 10688 / 3.04 s |
| semi inclusion + patch, nodes / DD build | 4843 / 1.72 s | 12624 / 3.58 s |
| mesh generation | gmsh 0.1-1.6 s | numpy < 0.05 s |

| 60 nm patch 0.5 (`carriers_3d_patch` geometry), DD: nodes / build / solve at +1 V | 55397 / 11.7 s / 140.8 s | **23273 / 3.1 s / 44.0 s** |
| default ITO cap, equilibrium: nodes / build / solve at +1 V | 4303 / 152.7 s / 1.0 s | 425 / 151.3-153.0 s / 0.1 s |

The validation wall times on the shared box were 1.0-1.6x LONGER on the tensor runs. That comes from box
load, not the mesh: the equilibrium build is DEVSIM's symbolic set-up (simplify of the Aymerich-Humet
F_1/2), which took 151-153 s on both meshers back to back.

- **Laterally uniform stacks** (the common case) get 8-10x fewer nodes and solve 10-20x faster.
- **Patch and inclusion stacks: it depends on the cell size.** A tensor grid carries the lateral
  mesh_min_nm refinement at a patch or inclusion edge through the full height and the full other axis.
  - On a 12 nm cell that means 2-3x more nodes.
  - On a 60 nm cell the tensor grid has 2.4x FEWER nodes and solves 3.2x faster. There gmsh's isotropic
    Threshold field refined the whole interface plane to 0.5 nm in x and y as well.

## 7. Limits

- **Axis-aligned boxes only.** The tensor mesher covers exactly what devsim_3d builds: layers, a centred
  square patch, centred rectangular full-layer inclusions. A future non-box feature, such as a
  cylinder, a slanted wall or a partial-height inclusion, needs either a well-centred unstructured mesher
  or DEVSIM fixing its magnitude sum (upstream: sign the pieces by orientation). Until then such a
  feature must not go back to gmsh silently; the legacy path warns.
- **Lateral cost.** A patch or inclusion edge refines the whole lateral line (section 6). A local
  refinement would need hanging nodes, which DEVSIM does not support.
- **Contacts on an inclusion.** An inclusion reaching the top or bottom face is not contacted. DEVSIM
  requires contact nodes on the contact's region, and the gmsh path got away with it only on coarse
  faces.
- **Shared mesh file.** `msh_path` defaults to `~/.dynameta/_devsim3d.msh`, shared by every concurrent
  process. This is pre-existing and unchanged. Concurrent 3-D builds on one box should pass their own
  `msh_path`.

## 8. Open questions

1. **Keep `mesher="gmsh"`?** It is kept, with a warning, for reproducing pre-v0.14 numbers. It could be
   removed in v0.15, which would also drop gmsh from the default runtime path.
2. **Upstream.** Report the magnitude sum in `TetrahedronEdgeCouple.cc` to DEVSIM. With signed pieces,
   volumes are exact on any mesh, although couplings can still go negative on non-Delaunay meshes.
3. **2-D impact ionization.** Should 2-D move to the same element reconstruction? It would be isotropic
   for 2-D current patterns, but the 2-D pins would move. Left unchanged.

## 9. Release-note lines

- carriers (3-D): **exact 3-D finite-volume geometry.** `devsim_3d` now meshes with a graded
  tensor-product grid of right-angled (Kuhn) tetrahedra (`Stacked3DSpec.mesher="tensor"`, the new
  default). DEVSIM 2.11 sums the magnitudes of each tetrahedron's circumcentric dual pieces, so on the
  previous unstructured gmsh mesh the node volumes and couplings were inflated by 1.20-1.52x. Integrated
  charge, sheet charges and currents were off by about 15-25 %:
  - D1c's 3-D D_it shift 0.80x;
  - Ohm's law +14 %;
  - Gauss residual 23 %.

  All are now exact or equal to 1-D. `mesher="gmsh"` reproduces old numbers, with
  `UnreliableResultWarning`.
- carriers (3-D): laterally uniform 3-D stacks are now exactly laterally invariant and equal the 1-D
  solution on the same z-lines. They have 8-10x fewer nodes and solve 10-20x faster. Patch and
  inclusion stacks have 2.4x fewer nodes on a 60 nm cell, but 2-3x more on a 12 nm cell.
- carriers: `physics_traps.interface_node_area` splits 3-D interface triangles circumcentrically (the
  control-volume face, == DEVSIM `SurfaceArea`) instead of in thirds.
- carriers: 3-D impact ionization (`impact_ionization.substrate_current` on a 3-D device): an
  element-based isotropic |E|, |J| reconstruction weighted by `ElementNodeVolume`. It refuses meshes
  with inexact element volumes.
- validation: `validation/devsim_3d_mesh.py` (gates M1-M7, including `--base` proving every gate fails
  on the pre-fix tree); `traps_oracles --3d` gains G3 and G2 passes. `carriers_3d_resistor` (Ohm < 1e-6)
  and `carriers_3d` (Gauss, invariance) now use the Kuhn mesh.
- tests: `tests/test_devsim_3d_mesh.py` (11 tests).
