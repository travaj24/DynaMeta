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

## Moved pins (2b7a002, E-node POINT sampling -- superseded by the averaging below)

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

## Why point sampling was not enough: it re-rounds every layer thickness

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
DECIDED 2026-11-02 (user): full edge-cell averaging as the DEFAULT -- next section.

## z-subcell averaging (the default since 2026-11-02; `z_subcell="point"` opts out)

### Formulation

Each E node k stands for its control volume V_k = [z_k - h_-, z_k + h_+] (h_- = h_+ = dz/2 on the
uniform grids; the half-cell spacings on the graded 1-D grid). f_r,k is the fraction of V_k inside
region r (superstrate pad, each layer, substrate pad), computed exactly by
`fdtd_nd.grid.z_fill_fractions`. ONE implementation, `fdtd_nd.grid.mix_profile`, turns region
values into node maps for every engine:

* Tangential E (Ex/Ey -- every normal-incidence engine, s-pol, the p-pol Ex): the field is
  continuous across a z-interface, D_k = sum_r f_r,k eps_r E, so eps_k = sum_r f_r,k eps_r (the
  arithmetic mean; Farjadpour et al., Opt. Lett. 31, 2972 (2006); Oskooi et al., Opt. Lett. 34,
  2778 (2009): the parallel component of the subpixel-smoothing tensor; Hwang & Cangellaris, IEEE
  MWCL 11, 158 (2001): the second-order effective eps of a tangential E).
* Normal E (the p-pol Ez half-node map of the oblique 2-D kernels; control volume [k dz, (k+1) dz]):
  D is continuous, so 1/eps_k = sum_r f_r,k / eps_r (the harmonic mean; the same papers'
  perpendicular component).
* Dispersive / nonlinear / gain layers, arithmetic channel: the polarization densities of parallel
  sub-layers add with their volume fractions, so every term is fill-weighted: eps_inf averages and
  each pole's STRENGTH scales (Drude wp^2, Lorentz delta_eps, gain kappa*dN, Raman chi3R) with its
  shape kept (gamma, w0, W_R, w_a, dw_a); chi2 and chi3 scale likewise. Exact whenever a cell holds
  at most one pole of each kind (every layer/dielectric face). Two DIFFERENT poles of one kind in
  one cell (the slices of a graded profile) get one pole with the summed strength and the
  strength-weighted shape (Drude: wp^2 = sum f wp_r^2, gamma = sum f wp_r^2 gamma_r / wp^2). That
  matches sum_r f_r eps_r(w) through first order in gamma/w: measured relative error of the cell's
  Drude term 6.6e-5 for an ITO-like pair (gamma 1e14 / 2e14 at w = 1.35e15), gate
  `test_two_drude_single_pole_rule_error_is_second_order`; end to end, fdtd_graded_enz GATE 1
  (ten 1.35-cell Drude slices) went to |dR| = 2.0e-4.
* Normal channel with dispersion: the mean of 1/eps of a pole is not a pole, so no exact local ADE
  exists. A mixed Ez cell touching any Drude region is POINT-sampled at its half node (every field
  of that cell); non-dispersive mixed Ez cells take the harmonic mean. Measured on a Drude slab at
  p-pol 50 deg (8 offsets, res 30): max|dR| 1.7e-3, max|dT| 1.1e-3 (point everywhere: 4.6e-3 /
  4.1e-3) -- the fallback leaves only the Ez-channel share of the staircase. Gate
  `test_true_geom_ppol_drude_slab_ez_fallback`.
* Hot carriers (2-D): a mixed edge cell carries its hot fill fraction f as the kernel mask; its
  Drude current is already f-weighted, so the kernel takes p_abs = J.E / f (the hot material's own
  dissipation density), and the mask-weighted means / energy closures are over hot VOLUME. A 0/1
  mask keeps the legacy expression bit-for-bit. Thermal parameters come from the cell's majority
  hot layer.
* Time-varying hooks (1-D): the hook layer's fill fraction w per node; a mixed cell's material is
  (its static other-region share) + w x (the hook value), with the D-preserving rescale per cell.
* Lateral painters: fdtd_seam.make_structured_lateral mixes its per-slab lateral rasters along z
  with the same rule; the lateral inclusion edges are NOT smoothed (limit below). User painters
  receive `zc` and paint as they like; the pad check skips the mixed face nodes.

`z_subcell="point"` (every solve_fdtd_* entry point, make_structured_lateral,
make_fdtd_optical_solver, fdtd_sweep_spectrum) reproduces 2b7a002 bit-for-bit. Measured on all
seven engines against an exported 2b7a002 package (numpy backends; the numba 3-D kernel is not
bitwise run-to-run, as before), and gated in-suite on the kernel input maps
(`test_point_opt_out_reproduces_the_2b7a002_material_maps`).

### Measured: true-geometry |R|, |T| and phase at EVERY offset

Slab n = 2, d = 183 nm (9.15 cells at res 30, 18.3 at res 60 -- not commensurate), lambda 1.4 um,
8 midpoint sub-cell offsets; errors against the exact TMM of the TRUE slab, max over the offsets.
"max dphi" = max |dphi_r| in rad (k0*dz = 0.0898 / 0.0449).

| engine | rule | max dR, res 30 / 60 | max dT, res 30 / 60 | max dphi, res 30 / 60 | mean dphi / k0dz, 30 / 60 |
| --- | --- | --- | --- | --- | --- |
| 1-D uniform | point | 7.9e-3 / 2.3e-3 | 7.9e-3 / 2.3e-3 | 0.073 / 0.037 | +0.01 / -0.05 |
| 1-D uniform | average | 1.9e-3 / 3.9e-4 | 1.9e-3 / 3.9e-4 | 0.0049 / 0.0012 | +0.05 / +0.03 |
| 1-D refine={0:2} | both (identical) | 9.8e-4 / 1.7e-4 | 4.7e-4 / 7.1e-5 | 0.0046 / 0.0012 | +0.04 / +0.02 |
| 1-D MO | point | 7.9e-3 / 2.3e-3 | 7.9e-3 / 2.3e-3 | 0.073 / 0.037 | +0.01 / -0.05 |
| 1-D MO | average | 1.9e-3 / 3.9e-4 | 1.9e-3 / 3.9e-4 | 0.0049 / 0.0012 | +0.05 / +0.03 |
| 2-D | point | 8.4e-3 / 2.5e-3 | 8.4e-3 / 2.4e-3 | 0.073 / 0.037 | +0.02 / -0.05 |
| 2-D | average | 2.3e-3 / 4.9e-4 | 2.3e-3 / 4.9e-4 | 0.0055 / 0.0013 | +0.06 / +0.03 |
| 2-D, n_super 1.5 | point | 4.6e-3 / 1.4e-3 | 3.6e-3 / 1.2e-3 | 0.104 / 0.057 | +0.11 / -0.04 |
| 2-D, n_super 1.5 | average | 1.7e-3 / 3.7e-4 | 1.7e-3 / 3.4e-4 | 0.0165 / 0.0040 | +0.18 / +0.09 |
| 2-D Drude+Lorentz (eps_inf 3) | point | 6.8e-3 / 2.9e-3 | 2.1e-2 / 9.4e-3 | 0.055 / 0.024 | +0.01 / +0.02 |
| 2-D Drude+Lorentz (eps_inf 3) | average | 2.9e-4 / 6.0e-5 | 6.1e-4 / 1.1e-4 | 0.0029 / 0.0006 | +0.02 / +0.01 |
| 2-D Drude+Lorentz (eps_inf 1) | point | 3.3e-4 / 2.2e-3 | 4.0e-4 / 2.0e-3 | 0.047 / 0.023 | +0.03 / +0.01 |
| 2-D Drude+Lorentz (eps_inf 1) | average | 8.1e-4 / 2.2e-4 | 8.5e-5 / 2.5e-5 | 0.0019 / 0.0005 | +0.00 / +0.00 |
| 2-D gain line on n = 2 | point | 8.1e-3 / 2.4e-3 | 9.3e-3 / 2.8e-3 | 0.073 / 0.037 | +0.02 / -0.05 |
| 2-D gain line on n = 2 | average | 2.3e-3 / 5.0e-4 | 2.4e-3 / 5.1e-4 | 0.0055 / 0.0013 | +0.06 / +0.03 |
| 2-D gain line (eps_inf 1) | point | 3.6e-3 / 3.1e-3 | 4.4e-2 / 3.5e-2 | 0.118 / 0.069 | +0.23 / -0.02 |
| 2-D gain line (eps_inf 1) | average | 1.3e-4 / 5.4e-5 | 1.6e-3 / 2.3e-4 | 0.018 / 0.0027 | +0.13 / +0.05 |
| 3-D | point | 8.4e-3 / 2.5e-3 | 8.4e-3 / 2.4e-3 | 0.073 / 0.037 | +0.02 / -0.05 |
| 3-D | average | 2.3e-3 / 4.9e-4 | 2.3e-3 / 4.9e-4 | 0.0056 / 0.0013 | +0.06 / +0.03 |
| 3-D MO | point | 8.4e-3 / 2.5e-3 | 8.4e-3 / 2.4e-3 | 0.073 / 0.037 | +0.02 / -0.05 |
| 3-D MO | average | 2.3e-3 / 4.9e-4 | 2.3e-3 / 4.8e-4 | 0.0056 / 0.0013 | +0.06 / +0.03 |
| 2-D oblique s 30 deg | point | 4.6e-3 / 8.9e-4 | 4.6e-3 / 9.4e-4 | 0.064 / 0.033 | +0.00 / -0.06 |
| 2-D oblique s 30 deg | average | 2.1e-3 / 4.2e-4 | 2.2e-3 / 4.5e-4 | 0.0037 / 0.0008 | +0.04 / +0.02 |
| 2-D oblique p 30 deg | point | 4.3e-3 / 9.3e-4 | 4.2e-3 / 8.3e-4 | 0.097 / 0.051 | -0.00 / -0.07 |
| 2-D oblique p 30 deg, harmonic Ez | average | 2.2e-3 / 5.1e-4 | 2.1e-3 / 3.8e-4 | 0.0039 / 0.0008 | +0.04 / +0.02 |
| 2-D oblique p 50 deg | point | 2.7e-3 / 6.0e-4 | 1.8e-3 / 6.0e-4 | 0.198 / 0.104 | -0.01 / -0.06 |
| 2-D oblique p 50 deg, harmonic Ez | average | 1.7e-3 / 5.3e-4 | 1.5e-3 / 4.8e-4 | 0.0022 / 0.0005 | +0.02 / -0.00 |
| 2-D oblique p 50 deg, Drude slab | point | 4.6e-3 / 4.2e-4 | 4.1e-3 / 9.0e-4 | ill-conditioned | -- |
| 2-D oblique p 50 deg, Drude slab, Ez fallback | average | 1.7e-3 / 2.4e-4 | 1.1e-3 / 5.6e-4 | ill-conditioned | -- |
| 3-D oblique s 30 deg (az 20) | point | 4.6e-3 (res 30) | 4.6e-3 | -- (R/T only) | -- |
| 3-D oblique s 30 deg (az 20) | average | 2.1e-3 (res 30) | 2.1e-3 | -- | -- |

Averaged, every engine's |R|, |T| and phase error drops ~4-5x per dz halving (O((k dz)^2), the Yee
dispersion) at EVERY offset; point sampling drops only 2-3.5x (the O(dz) staircase, offset luck at
any single geometry -- the eps_inf-1 Drude/Lorentz slab's point |R| error even GROWS from res 30 to
60). The per-offset phase spread collapses (1-D: 0.073 -> 0.0049 rad at res 30). The mean phase
bias stays at its registration-fix level (+0.05 / +0.06 k0dz at res 30, now halving cleanly as
O((k dz)^2); n_super 1.5: +0.18 -> +0.09). Two single entries are worse at res 30 than the point
rule's lucky value, and both converge (O((k dz)^2)) where point does not: the eps_inf-1
Drude/Lorentz slab's max|dR| (8.1e-4 vs a lucky 3.3e-4; at res 60, 2.2e-4 vs 2.2e-3), and the
fdtd_seam_vs_tmm GATE 3b mean phase (+1.09 deg = 0.21 k0dz at res 28 vs the point mean +0.07 deg,
but +0.25 deg at res 56: a 4.3x drop; that stack's single-geometry 3a error went 5.00 -> 1.03 deg).

Gates (tests/test_fdtd_registration.py; `DYNAMETA_TEST_ZSUBCELL=point` re-runs them on the point
rule, where each discriminating one fails): `test_true_geom_1d_*` (every offset + the 4x drop),
`test_true_geom_2d_matches_tmm_at_every_offset[diel|glass|dl|dl_vac|gain_vac]`,
`test_true_geom_2d_oblique_tangential_and_harmonic_channels[s30|p30|p50]`,
`test_true_geom_ppol_drude_slab_ez_fallback`, `test_true_geom_3d_and_3d_mo`, plus the mixer rules,
the two-Drude error bound and the point opt-out identity.

### The four 2b7a002 failures, against TRUE-geometry references

| gate | bar | 2b7a002 (point) | averaged |
| --- | --- | --- | --- |
| fdtd_seam_vs_tmm GATE 1 (dielectric R/T vs TMM) | 6e-3 | dR 4.05e-2, FAIL | dR 2.92e-3, PASS |
| fdtd_drude_lorentz_vs_tmm GATE 2 (pure Lorentz) | 2e-2 | max dR0 2.82e-2, FAIL | max dR0 5.89e-4, PASS (GATE 1 Drude+Lorentz 4.98e-3 -> 8.80e-4) |
| fdtd_graded_enz GATE 1 (graded ITO vs TMM) | 2e-2 | d 2.37e-2, FAIL | d 2.04e-4, PASS |
| test_fdtd_hot_carrier::test_lat_gate8 (2-ODE) | 5e-3 | 7.53e-3, FAIL | 7.41e-3, FAIL (diagnosed below) |

HOT-CARRIER GATE 8: STOPPED AND REPORTED, NOT FORCED. The film is 120 nm = 1.87 cells at the
fixture's resolution 12; averaged it is two nodes of hot fill 0.934 / 0.936 (hot volume 120 nm,
exact). The gate integrates ONE mean-field 2-ODE driven by the mask-averaged p_abs and compares it
with the mask-averaged T_e. Instrumenting the kernel with per-cell histories: the two cells absorb
5.39e6 / 4.20e6 J/m^3 (a 25% gradient across the ENZ film), and an EXACT per-cell 2-ODE oracle
(each cell driven by its own recorded p_abs) matches the FDTD to 7.8e-4 / 6.8e-4 -- the level of
d387eef's single-cell 7.1e-4 -- in BOTH the point and the averaged runs, while the mean-field
oracle gives 7.5e-3 / 7.4e-3. So the kernel and the fill weighting are right; the gate's mean-field
oracle is exact only for a one-cell (spatially uniform) film, which the fixture had only because
1.87 cells were rounded to 1. Proposed, not done: expose per-cell p_abs histories in hot_out and
integrate the oracle per cell (measured 7.8e-4 < 5e-3), or refine the fixture.

### Moved pins and non-green items under averaging (STOP-AND-REPORT where a value moved away)

Every move below was checked against its TRUE-geometry reference and against resolution. None was
re-pinned: in each case either the averaged value at the fixture's resolution is FARTHER from the
reference than the point value (while converging as O((k dz)^2), which the point value does not),
or the gate's oracle itself is not a true-geometry reference.

* validation/fdtd_sweep_vs_tmm.py GATE 1 (the same near-AR 2.2/1.5/2.2 stack, 1200-1400 nm, res 28,
  bar 6e-3): max|dR| vs TMM -- point 2.88e-3 (res 56: 3.95e-3, no convergence), averaged 6.68e-3
  (res 56: 1.63e-3, a 4.1x drop). MOVED AWAY at res 28 -> reported, not re-pinned, bar not loosened.
* validation/fdtd_3d_reduces.py GATE A (n = 2, 300 nm slab vs Airy, res 18, bar 5e-3): point 2.45e-3
  (res 36: 6.0e-4), averaged 6.26e-3 (res 36: 1.54e-3). BOTH converge 4x: here the grid is
  COMMENSURATE (pad/dz = 216, d/dz = 9 exactly), so both faces sit ON E nodes. Point sampling then
  realizes the exact thickness (a rigid -dz/2 shift |R| cannot see) and carries only the bulk Yee
  dispersion, while averaging gives each face node the 0.5/0.5 mean -- second-order, but with a
  larger constant than a face on an H node. This is the general trade: averaging bounds the WORST
  sub-cell offset (8-offset max|dR| 7.9e-3 -> 1.9e-3 at res 30) but is not better at the best one
  (a face exactly on a node or a cell boundary). MOVED AWAY at both resolutions -> reported.
* tests/test_ringdown.py::test_gate7 (n_slab 3.5 etalon Q vs Fabry-Perot 13.361960): point 13.207623
  (-1.155%; res 60 -0.370%), averaged 13.078561 (-2.121%; res 60 -0.253%). The back face lies
  EXACTLY on node 980 (pad/dz = 892.5, d/dz = 87.5), so averaging gives that node the 0.5/0.5 mean
  and the slab its exact 87.5-cell thickness; the res-30 Q is dispersion-limited (n k dz = 0.17).
  MOVED AWAY at res 30 (closer at res 60) -> reported, not re-pinned.
* tests/test_ringdown.py::test_gate9 (the start_frac = 0.2 path must be warning-free): averaged, the
  pencil drops one GROWING pole (|z| ~ 1) and warns. Q there 13.520 (+1.19%, inside the 3% bar;
  point 13.280, -0.61%). The trace decays to the float floor in both runs (max |E| over the last 5%:
  8.0e-16 point, 1.7e-15 averaged) -- no growth, the dropped pole fits the noise floor that the
  fixed-fraction window includes. Not a pin; reported.
* tests/test_fdtd_hot_carrier.py::test_lat_gate8: 7.41e-3 vs 5e-3 -- the mean-field-oracle
  diagnosis above (exact per-cell oracle 7.8e-4). Reported.
* validation/fdtd_effect_seam.py GATE A (a POINT-band lateral (eps_inf, wp, gam) painter must
  equal the solver's own layer fill cell for cell, max|d| == 0): it compared a point painter with the
  now-averaged fill (5.8e-3). The reference call now passes z_subcell="point" (the gate is about the
  lateral plumbing and the sizing guard); max|d| = 0 again. Not a tolerance change.
* Test-harness updates (no value pinned): test_fdtd_seam S5-20's painter spy now accepts the
  forwarded z_subcell; test_fdtd_mo_oblique's lateral_tensor point painter and its S2-14 node-rule
  gate compare against z_subcell="point" (both are about the point rule by construction).
* D-12 (Kerr + eps_inf hook refusal): under averaging the hook layer's edge cells hold a share of a
  Kerr NEIGHBOUR, which tripped the per-cell test; the refusal is now keyed on the hook layer's own
  chi3 (the neighbour's share in that one cell is left out of the D-preserving rescale -- an
  O(f chi3 E^2) single-cell error). test_d12 passes unchanged.

### Runs (from scratch, 2026-11-03, averaging default)

The 22 FDTD-related test files incl. slow: 555 passed, 3 failed (test_lat_gate8, ringdown gate7,
ringdown gate9 -- all above). The 41 validation/fdtd_* scripts: 38 PASS, 3 FAIL (fdtd_sweep_vs_tmm
GATE 1 and fdtd_3d_reduces GATE A -- moved away at the fixture resolution, above -- and
fdtd_gpu_nonlinear, the pre-existing CUDA 13.1 / numba header compile failure, unchanged).
fdtd_effect_seam passed after its like-for-like reference fix (re-run on its own). ruff clean.

### Limits

* Lateral inclusion edges of structured 2-D/3-D cells are point-rastered (rasterize.layer_eps_cell);
  their staircase is unchanged. Follow-up: the same Farjadpour/Oskooi tensor smoothing laterally.
* The 3-D kernels carry one eps array for Ex/Ey/Ez (arithmetic channel); exact for laterally
  uniform stacks (Ez = 0), approximate for Ez in structured cells (as before).
* The nonuniform refine grid already put interfaces on nodes; averaging there equals the tie rule
  (identical results to 2b7a002). User z_edges with mid-cell interfaces are now averaged too.
* inverse_design's index-based design region is not a layered stack and is not averaged.
