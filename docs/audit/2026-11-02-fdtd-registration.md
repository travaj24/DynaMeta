# FDTD material registration (v0.13.0 release note section, 2026-11-02)

Follow-up of audit 2026-07-17 S2-14 (wave-7). Workstream `fdtd_phase`, branch `wip/fdtd_phase`.

## What was wrong

Every DynaMeta FDTD engine uses the same z staggering: the tangential E (the field the plane wave,
the soft source, the probes and the PEC/Mur walls live on) at the integer nodes `k*dz`, the
tangential H at `(k+1/2)*dz`. The probes are E indices and every front end de-embeds the complex
reflection `r0` in that node frame (`exp(-2i n k0 (pad - k_pL*dz))`). The eps that multiplies `E_k`
stands for `E_k`'s control volume `[(k-1/2), (k+1/2)]*dz`, which is centred ON the node (Taflove &
Hagness 3rd ed., Yee material assignment; Hwang & Cangellaris, IEEE MWCL 11:158 (2001): the
second-order effective eps for a tangential E is the control-volume arithmetic mean). Point-sampling
eps at the node puts each effective interface (the H node between the last outside and the first
inside E node) within +-dz/2 of the true one with zero mean over the sub-cell offset.

`solve_fdtd_mo_1d` sampled at the nodes. `solve_fdtd_1d` (uniform path), `solve_fdtd_2d`,
`solve_fdtd_2d_oblique`, `solve_fdtd_3d`, `solve_fdtd_3d_oblique` and `solve_fdtd_3d_mo` sampled at
the cell centres `(k+1/2)*dz`: every structure sat dz/2 early on average, a reflection-phase bias of
`-n_super * k0 * dz * cos(theta)` that |R| and |T| cannot see. validation/fdtd_seam_vs_tmm.py GATE 3
(15 deg, "the half-cell limit") absorbed it. Two more registration errors surfaced while measuring:

* `solve_fdtd_1d(refine=...)` (nonuniform z): its grid builder SNAPS every interface onto a node and
  the half-open rule handed that node wholly to the region starting there (interface moved `-h/2`):
  -0.88 k0*h_fine mean bias.
* `solve_fdtd_2d_oblique(pol='p')`: one eps array fed Ex (node) and Ez (half node). With the node
  map on both, p-pol kept -0.29 k0dz (30 deg) and -1.56 k0dz (50 deg): Ez's interfaces sat dz/2 off.

## The fix (one convention, one implementation)

`dynameta/optics/fdtd_nd/grid.py`:

* `material_z(nz, dz)` = `arange(nz)*dz`, the E-node registration positions;
* `layer_masks(z_m, z_start, layers)` = the half-open `[z, z+t)` slab masks with the exact float
  accumulation of the inline loops it replaces (fdtd_mo is bit-identical);
* `pad_masks(z_m, pad, z_struct)`;
* `material_z_half(nz, dz)` = `(arange(nz)+1/2)*dz`, the NORMAL-component (Ez) positions;
* `interface_node_fractions(...)` = the control-volume fractions of a node lying exactly ON an
  interface (the tie rule, used where the tie is systematic).

Every engine now builds its materials through these: fdtd.py (uniform + time-varying mask +
nonuniform), fdtd_mo.py, solve2d.py (normal + oblique + `_check_lateral_pads`), solve3d.py (normal,
oblique, MO), fdtd_seam.make_structured_lateral (its z slabs; the LATERAL raster
`rasterize.layer_eps_cell` is unchanged). Lateral painters receive `zc` = the E-node positions.

* Nonuniform 1-D: interface-tie nodes take the control-volume arithmetic mean (eps_inf, chi3
  linearly; Drude as one pole with `wp^2 = sum f wp_i^2` and the wp^2-weighted gamma -- exact when
  at most one side is Drude or the gammas agree).
* p-pol oblique 2-D: the TM kernels (numpy / numba / jax) take optional per-component Ez maps
  (`ez=(eps_inf, wp, gam)`; `None` = legacy single array), and the front end passes the half-node
  registration. The 3-D kernels keep one array for Ex/Ey/Ez: Ez is identically 0 on every laterally
  uniform 3-D case they solve (normal incidence; the 3-D oblique engine is s-pol), so their phases
  are exact to O((k dz)^2); in a STRUCTURED 3-D cell Ez is dz/2 off -- the vertical twin of the
  lateral single-array raster, a documented limit.

## Measured table

Slab n = 2, d = 180 nm (|r| ~ 0.6), lambda 1.4 um, band 1.2-1.6 um, vacuum ends unless noted; mean
signed dphi_r vs the exact tangential-field characteristic-matrix TMM over 8 MIDPOINT-uniform
sub-cell pad offsets (pad/dz = m + (k+1/2)/8, which cancels the point-sampling sawtooth exactly);
units of k0*dz (k0*dz = 0.0898 rad at res 30, 0.0449 at res 60). Measured with a one-off probe
script (dev box, 2026-11-02; "before" = the HEAD d387eef package exported by `git archive`); the
same measurement, at 4 offsets, is gated per engine in tests/test_fdtd_registration.py, whose bias
gates all fail on d387eef.

| engine | before res 30 | before res 60 | after res 30 | after res 60 |
| --- | --- | --- | --- | --- |
| solve_fdtd_1d uniform | -0.95 | -0.98 | +0.05 | +0.02 |
| solve_fdtd_1d refine={0:2} (k0*dz units of the BASE dz) | -0.46 | -0.48 | +0.04 | +0.02 |
| solve_fdtd_mo_1d (always node) | +0.05 | +0.02 | +0.05 | +0.02 |
| solve_fdtd_2d numpy (numba, jax identical) | -0.94 | -0.97 | +0.06 | +0.03 |
| solve_fdtd_2d, n_super 1.5 | -1.32 | -1.41 | +0.18 | +0.09 |
| solve_fdtd_3d numpy (numba identical) | -0.94 | -0.97 | +0.06 | +0.03 |
| solve_fdtd_3d_mo | -0.94 | -0.97 | +0.06 | +0.03 |
| solve_fdtd_2d_oblique s, 30 deg | -0.82 | -0.84 | +0.04 | +0.02 |
| solve_fdtd_2d_oblique p, 30 deg | -1.15 | -1.17 | +0.04 | +0.02 |
| solve_fdtd_2d_oblique p, 50 deg | -2.18 | -2.19 | +0.04 | +0.05 |
| (p 30 / 50 deg, node map on Ez too -- the intermediate) | | | -0.29 / -1.56 | -0.31 / -1.57 |
| 3-D oblique kernel s, 30 deg, azimuth 20 | -0.82 | -0.85 | +0.04 | +0.01 |

In rad (res 30 -> 60) the post-fix residual shrinks ~4x per halving of dz (1-D 0.0044 -> 0.0011,
2-D 0.0053 -> 0.0013, glass 0.0164 -> 0.0040, refine 0.0040 -> 0.0011): O((k dz)^2), the Yee
dispersion. Before, it was a constant ~ -1.0 k0dz: a registration offset. The probe-de-embedding
leg (vacuum, ~0.3 pad) alone contributes ~3e-3 rad of O((k dz)^2) at res 30.

validation/fdtd_seam_vs_tmm.py GATE 3b (dielectric 3-layer stack, res 28, 8 offsets): mean dphase
-5.195 deg (-0.99 k0dz) before, +0.066 deg after; bar 0.25 k0dz = 1.315 deg. GATE 3a (single
geometry) tightened from 15 deg to 1.5 k0dz (sanity / convention bar).

## Moved pins

* tests/test_ringdown.py::test_gate7 (n_slab 3.5 etalon Q): 13.227438 -> 13.207623 (-0.15%). The
  slab keeps its 88 cells (pad/dz = 892.5, d/dz = 87.5): a rigid one-cell translation that moves the
  fitted Q through the data-driven window timing; vs Fabry-Perot -1.007% -> -1.155% (gate 3%).
  Re-pinned.
* NOT re-pinned -- |R|/|T|-class moves larger than O((k dz)^2), every one traced to the
  thickness-rounding redraw below (STOP-AND-REPORT per the brief):
  * validation/fdtd_seam_vs_tmm.py GATE 1 dielectric R 0.0307 -> 0.0710 (TMM 0.0305; bar 6e-3).
  * validation/fdtd_drude_lorentz_vs_tmm.py GATE 2 (pure Lorentz, 250 nm, res 40): max|dR0|
    -> 2.82e-2 (bar 2e-2). The slab is 33.19 cells: 33 cells (248.5 nm) before, 34 (256.0 nm)
    after; TMM of the REALIZED 34-cell slab vs the true one gives max|dR| 2.81e-2 (33-cell 6.7e-3),
    i.e. the FDTD reproduces its grid to 1e-4.
  * validation/fdtd_graded_enz.py GATE 1: R 0.1072 -> 0.1496 vs TMM 0.1259 (|d| 1.87e-2 -> 2.37e-2,
    bar 2e-2; the two sit on OPPOSITE sides of TMM). Ten 40 nm slices of 1.348 cells each: realized
    [1,1,2,1,1,2,1,1,2,1] = 13 cells before, [2,1,1,2,1,1,2,1,1,2] = 14 after (true 13.48) -- the
    ENZ-crossing front slice doubled.
  * tests/test_fdtd_hot_carrier.py::test_lat_gate8 (2-ODE trajectory consistency 7.1e-4 -> 7.5e-3,
    bar 5e-3; peak T_e rise 139.8 -> 110.4 K): a 1.87-cell film, 1 cell -> 2 cells.
* Unrelated failure (identical on d387eef): validation/fdtd_gpu_nonlinear.py -- the CUDA 13.1
  headers fail to compile under numba's NVRTC (cuda_fp8/fp6/fp4.hpp syntax errors); environment.

## Open: point sampling re-rounds every layer thickness

Point sampling realizes a layer of thickness t as N whole cells, N = #nodes in [z, z+t). Switching
the sampling positions by dz/2 changes N by up to one cell wherever frac(t/dz) and the faces'
sub-cell offsets straddle 1/2 -- an O(dz) per-geometry change with ZERO mean over offsets (both
rules are thickness-unbiased; only the node rule is position-unbiased). The FDTD solves its realized
grid faithfully: GATE 1's middle 180 nm layer is 10 cells (189.9 nm) under the old rule and 9 cells
(170.9 nm) under the new one; TMM of the two REALIZED stacks gives R = 0.0326 / 0.0735 against the
FDTD's 0.0307 / 0.0710 (true 0.0305). The old rule was lucky on that near-AR design: sweeping the
same stack's front face over the sub-cell offset (1-D march, same res 28 grid, 2026-11-02),
point-sampled R takes 0.0132 / 0.0308 / 0.0711 depending on frac(pad/dz) (0.34-0.88 / 0.52-0.70 /
0.06-0.16) -- GATE 1's 6e-3 bar was an offset-luck pass under either rule -- while control-volume
averaging of the same eps gives 0.0273-0.0280 at every offset (the remaining ~3e-3 is dispersion). The hot-carrier
fixture's 120 nm film is 1.87 cells at resolution 12: 1 cell before, 2 after (closer to 120 nm), so
the single-cell uniformity its mean-field RK4 oracle relied on is gone.

The remedy that removes the redraw is control-volume (sub-cell) averaging -- the Hwang-Cangellaris
arithmetic mean for the tangential E, already used here at the nonuniform grid's interface ties --
which makes position AND thickness second-order. It is exact for non-dispersive linear layers but
has no single-pole equivalent between two different Drude/Lorentz/gain/Raman materials and no
meaning for the per-cell hot-carrier material ids, so it is a design decision (scope: which layer
pairs get averaged), not part of this registration fix.
