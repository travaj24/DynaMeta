# Bipolar DD: Auger + radiative recombination, bandgap narrowing, incomplete ionization (R26 + R27)

DEVSIM phase, release v0.14.0, workstream D1b, branch `wip/dv_recomb` (off `feat/devsim-2.11-v0.14`
at 2db6a00). Closes roadmap items **R26** (Auger + radiative recombination in the bipolar model) and
**R27** (bandgap narrowing + incomplete dopant ionization) of `docs/physics_depth_roadmap.md`.

## What changed

Everything is **opt-in and off by default**. `physics_bipolar_dd.setup_bipolar_region` takes a new
keyword `options: BipolarOptions | None = None`; `None` (or a bundle whose fields are all `None`)
never calls the new code, so every pre-existing bipolar result is byte-identical (gated, see below).

| File | Change |
|---|---|
| `dynameta/carriers/bipolar_options.py` (new) | parameter dataclasses (`AugerModel`, `RadiativeModel`, `BGNModel`, `IonizationModel`, `BipolarOptions`), cited Si presets, the ONE numpy kernel for every doping-only field (`equilibrium_fields`), the DEVSIM expression builders, `install_bipolar_options` |
| `dynameta/carriers/physics_bipolar_dd.py` | `options=` on `setup_bipolar_region` (installs last, by model name); seed models, `fd_shift_model`, both ohmic contact recipes (DC + circuit) and `ohmic_built_in_offset_V` follow the option equilibrium when BGN/ionization are on |
| `dynameta/materials/transport_model.py` | `TransportModel.bipolar_options` (validated: right type, `physics='bipolar_dd'` only) |
| `dynameta/carriers/devsim_layered.py` | 2-D `LayeredDevsimBuilder` threads `tr.bipolar_options` |
| `dynameta/carriers/devsim_3d.py` | `Stacked3DSpec.bipolar_options` (validated), threaded by `from_design`, into the region setup AND into the gate reference `phi_bi` |
| `tests/test_bipolar_recombination_bgn_ionization.py` (new) | 22 gates (below) |
| `tests/_bipolar_calllog.py` (new) | defaults-off DEVSIM call-log recorder (runs unchanged on v0.13.0) |

One implementation per kernel: the fixed fields (dEg, band-edge split, n_ie, ionization reference
densities, the charge-neutral root n0/p0, the contact potential, the seed shift, the Coulomb-Auger
factors) come from `equilibrium_fields` for the 1-D direct path, the 2-D builder, the 3-D builder and
the 3-D gate frame alike. The solution-dependent strings come from `srh_expr`, `recombination_exprs`
and `ionized_net_doping_expr`.

## Equations

Net recombination (positive = recombination) entering both continuity equations:

    U_net = U_SRH + U_A + U_rad - G
    ElectronGeneration = -q U_net ,  HoleGeneration = +q U_net     (DEVSIM sign convention, unchanged)

### Auger (R26)

    U_A = (C_n n + C_p p) (n p - n_ie^2)

Optional Coulomb enhancement (Altermatt et al. 1997, low-injection parameterization; n0/p0 are the
EQUILIBRIUM densities, so g is a fixed field):

    C_n -> g_eeh(n0) C_n ,  g_eeh = 1 + 13  [1 - tanh((n0/3.3e17 cm^-3)^0.66)]
    C_p -> g_ehh(p0) C_p ,  g_ehh = 1 + 7.5 [1 - tanh((p0/7.0e17 cm^-3)^0.63)]

### Radiative (R26)

    U_rad = B (n p - n_ie^2)

Low-injection lifetimes (n-type): `1/tau = 1/tau_p + C_n n0^2 + B n0`.

### Bandgap narrowing (R27)

Apparent narrowing vs a dopant density chosen by `BGNModel.doping_basis` (corrected 2026-11-06,
see `2026-11-06-v0.14-materials-corrections.md`):

    Slotboom:  dEg = E_ref [ ln(N/N_ref) + sqrt(ln^2(N/N_ref) + C) ],   N = N_D + N_A
    del Alamo: dEg = E_ref ln(N_D/N_ref)   (N_D > N_ref),  0 otherwise   (n-type fit, N_D only)

**Band-edge split and the drift-diffusion quasi-fields** (Lundstrom & Schuelke 1983; Selberherr 1984).
A fraction `gamma` (`conduction_fraction`) of dEg lowers `E_c`, the rest raises `E_v`:

    E_c = E_c0 - q psi - gamma dEg ,     E_v = E_v0 - q psi + (1 - gamma) dEg
    D_c = gamma dEg / q ,  D_v = (1 - gamma) dEg / q                               [V]
    n = n_i exp((psi + D_c - phi_n)/V_t) ,  p = n_i exp((phi_p - psi + D_v)/V_t)
    n p = n_ie^2 exp((phi_p - phi_n)/V_t) ,  n_ie = n_i exp(dEg / 2kT)

The potential stays referenced to the UNNARROWED intrinsic level, so Poisson is untouched; electrons
drift in `psi_n = psi + D_c`, holes in `psi_p = psi - D_v`. In the Scharfetter-Gummel edge flux:

    vdiff_n = vdiff + (D_c@n0 - D_c@n1)/V_t ,   vdiff_p = vdiff - (D_v@n0 - D_v@n1)/V_t

(the FD g-factor divides these exactly as it divided `vdiff`). `gamma = 1/2` is the symmetric
"effective intrinsic density" form. On a uniform region the split is a constant shift and only n_ie
matters; it matters where the doping is graded. With BGN on, SRH uses `n_ie^2` and
`n1 n_ie/n_i, p1 n_ie/n_i` (so `n1 p1 = n_ie^2`): an SRH left on `n_i^2` would recombine AT EQUILIBRIUM
in a narrowed region. Ohmic contact (Dirichlet `Potential - bias - EquilibriumPotential`):

    psi_eq = V_t [ln(n0/n_i) + Delta_FD(n0/N_c)] - D_c   (n-type)
           = -V_t [ln(p0/n_i) + Delta_FD(p0/N_v)] + D_v  (p-type)

with n0 p0 = n_ie^2 and neutrality. In the Boltzmann limit the two branches are equal identically.
The seeded potential uses the same value. `fd_shift_model` returns `SeedPsiShift`, the per-node `s`
for which the unchanged `equilibrium_seed_psi_expr(s)` reproduces `psi_eq` exactly, so the three
staged solvers (1-D validation, 2-D layered, 3-D) seed in the contact frame without a signature change.

### Incomplete ionization (R27)

Boltzmann occupancy of one donor and one acceptor level, each set by its own carrier's quasi-Fermi
level:

    N_D+ = N_D / (1 + g_D n/n1D) ,  n1D = N_c exp(-dE_D/kT) ,  g_D = 2
    N_A- = N_A / (1 + g_A p/p1A) ,  p1A = N_v exp(-dE_A/kT) ,  g_A = 4
    Poisson charge: -q (p - n + N_D+ - N_A-)

DEVSIM differentiates the rational form symbolically (`IonizedNetDoping:Electrons/Holes`), so Newton
stays quadratic. Measured on the 50 K prototype bar kicked 50 mV: Newton relative error 4.9e-2,
4.0e-2, 1.1e-14, 1.7e-16. Optional
Pearson-Bardeen lowering `dE(N) = dE0 - alpha N^(1/3)` is available, with `alpha = 0` (off) by
default. Donors and acceptors default to the net-doping split. A compensated region defines
`Donors` + `Acceptors` node models; defining only one raises an error. The contact/seed neutral root
solves

    f(n) = n - n_ie^2/n - N_D+(n) + N_A-(n_ie^2/n) = 0

by bisection in ln n. f is strictly increasing in n, so the root is unique, and the bracket
`[n_ie^2/(N_A+N_D+n_ie), N_A+N_D+n_ie]` always holds the sign change: at the upper end
`f >= N_A + n_ie - p >= 0`, at the lower end `f <= n_lo - n_ie - N_D <= 0`.

### Temperature and generation

`T_K` sets `V_t = kT/q` on the region (None keeps `V_T` at T_REF bit-for-bit). It does NOT rescale
`n_i`, `N_c`, `N_v`, mobilities or lifetimes: the caller supplies them at `T_K`. `generation_m3s` adds
a uniform G (region parameter `G_ext`, rampable).

## Parameter presets (Si, 300 K; opt-in, never applied per material automatically)

| Preset | Values | Source |
|---|---|---|
| `AUGER_SI_DZIEWIOR_SCHMID_1977` | C_n = 2.8e-31, C_p = 0.99e-31 cm^6/s | Dziewior & Schmid, Appl. Phys. Lett. 31, 346 (1977) |
| `AUGER_SI_ALTERMATT_1997` | same C + g_eeh/g_ehh (13, 3.3e17, 0.66 / 7.5, 7.0e17, 0.63) | Altermatt et al., J. Appl. Phys. 82, 4938 (1997) |
| `RADIATIVE_SI_TRUPKE_2003` | B = 9.5e-15 cm^3/s | Trupke et al., J. Appl. Phys. 94, 4930 (2003) |
| `BGN_SI_SLOTBOOM_1977` | 9 mV, 1e17 cm^-3, C = 0.5 | Slotboom & de Graaff, SSE 19, 857 (1976); IEEE TED 24, 1123 (1977). Fitted on p-type transistor bases; reference n_i about 1.44e10 cm^-3 [inferred] (note added 2026-11-06) |
| `BGN_SI_KLAASSEN_1992` | 6.92 mV, 1.3e17 cm^-3, C = 0.5 | Klaassen, Slotboom & de Graaff, SSE 35, 125 (1992); reference n_i 1.0e10 or 1.08e10 [unverified] |
| `BGN_SI_DEL_ALAMO_1985` | 18.7 mV, 7e17 cm^-3, in N_D | del Alamo & Swanson, IEEE TED ED-34, 1580 (1987), Eq. (11); IEDM 1985 p. 290, Eq. (7). n-type (Si:P) only. **Corrected 2026-11-06:** the fit was cited as SSE 28, 47 and applied to N_D + N_A |
| `IONIZATION_SI_P_B` | dE_D = 45 meV (P), dE_A = 45 meV (B), g_D = 2, g_A = 4 | Sze & Ng, Physics of Semiconductor Devices, 3rd ed. (2007), ch. 1 |
| `SI_DONOR_IONIZATION_EV` / `SI_ACCEPTOR_IONIZATION_EV` | P 45, As 54, Sb 42.74 / B 45 meV | Sze & Ng; Sb from Ramdas & Rodriguez, Rep. Prog. Phys. 44, 1297 (1981), Table 4 (**corrected 2026-11-06** from Sze & Ng's 39 meV) |

The BGN parameter sets are **apparent** narrowings. They were extracted with Boltzmann statistics
against a specific reference n_i, so a result is only meaningful with the n_i it is paired with (see
Open questions). The setup issues an `AdvisoryWarning` when apparent BGN meets the FD g-factor on a
degenerate region (FD built-in correction > 0.05 V_t), because part of the degeneracy is then counted
twice.

## Oracles (all independent of the code under test)

The literature constants are re-typed in CGS inside the test. The BGN formulas are re-typed. The
neutrality roots come from scipy `brentq` on a hand-written residual. k/q come from
`scipy.constants`: the 10-digit `KB_EV_K` literal alone shifts the 50 K freeze-out root by 8.6e-11. The
DEVSIM solution is always what gets measured.

| Gate | Fixture | Oracle | Measured | Gate tol | ON vs off |
|---|---|---|---|---|---|
| R26-a (x3) | n 1e25, SRH 1 us + D-S Auger + Trupke B, 150 um zero-flux slab, G = 1e24 / 1e30 / 1e33 (dp/n0 = 3e-9 .. 0.87) | brentq root of G = U_SRH + U_A + U_rad | 0, 2.2e-16, -1.1e-16 (bit-identical on both OpenBLAS builds) | 1e-9 | SRH-only root 29x / 32x / 229x larger |
| R26-b (n, p) | Auger-only slab, low injection | tau_A = 1/(C n0^2) | -8.4e-9 / -4.9e-8 (= the dp/n0 injection term) | 1e-6 | SRH-only tau 1e3 s, 10 orders longer |
| R26-c | Altermatt Coulomb-enhanced, n0 = 1e17 cm^-3 | 1/(g_eeh C_n n0^2), g_eeh = 8.4646 | -1.39e-7 | 1e-6 | 8.5x shorter than un-enhanced |
| R26-d | radiative-dominated slab (90 % radiative), 3 mm | URad = B(np - n_i^2) at every node; root of G = U | 0 / 0 / 2.2e-16 | 1e-12 / 1e-9 | SRH-only 11.6x larger |
| R27 formulas | Slotboom (old, Klaassen), del Alamo | re-typed formulas; cancellation-free light-doping branch | passes | 1e-12 / 1e-13 | |
| R27-a | uniform n+ 1e19 cm^-3, zero-flux, holes kicked 3x | n p = n_ie^2 (n_ie^2/n_i^2 = 6.845) | 2.2e-16 | 1e-12 | an n_i-based SRH relaxes 5.8 off |
| R27-b (gamma 0.3, 0.5, 1) | graded 1e20 -> 1e17 cm^-3 over 1 um, Slotboom, both contacts 0 V | n p = n_ie(x)^2 everywhere; J = 0 | <= 4.6e-15; max \|J\| 0.03-0.10 A/m^2 = 2.5e-13 of q D N/L_grad | 1e-12 / 1e-12 x scale | negative control (drop the E_c quasi-field): 5e8 - 1.8e9 A/m^2 |
| R27-c | n+ (1e19, 15 um = 13.5 L_p) / p (1e17) long diode, del Alamo, tau 1 ns, V = 0.3 V, Boltzmann | J0p = q n_ie^2 D_p/(N_D L_p); ratio exp(dEg/kT) | -6.3e-4 (BGN), +6.7e-4 (off); ratio -1.3e-3 | 2e-3 | ratio 6.84, not 1 |
| R27-d (x12) | Si:P 1e16 cm^-3, uncompensated and 30 % B-compensated, T = 50/77/100/150/200/300 K, 2 um bar, interior kicked 50 mV | brentq charge-neutrality root | <= 7.1e-15 (mid-bar AND contact pin) | 1e-11 | ionized fraction 0.051 / 0.009 at 50 K (full ionization off 19x / 111x) |
| R27-e | Si:B 1e16 cm^-3, 77 K (g_A = 4 branch) | brentq root | passes 1e-11 (not tabulated separately) | 1e-11 | fraction < 0.5 |
| off-a | v0.13.0 call-log digest (217 DEVSIM calls, two fixtures incl. a circuit contact) | sha256 recorded on main 2db6a00 | identical | exact | |
| off-b | `BipolarOptions()` vs `None`, forward-biased pn diode | bitwise arrays + contact currents | identical | exact | |
| wiring (x4) | 3-D `phi_bi` == pinned contact field (BGN + ionization + 200 K, both polarities); TransportModel/Stacked3DSpec validation; 2-D layered bar builds the option models with n p = n_ie^2; BGN+FD advisory fires once (FD) / never (Boltzmann) | | 1e-14 V | | |

R27-c error budget at 2e-3: SG discretization O((h/L_p)^2) <= 3e-4 at h <= 20 nm; the finite emitter
`tanh(15 um/L_p) - 1` ~ 1e-11; n-side depletion width ~1 nm << L_p, de-rated analytically by
`exp((x - x_j)/L_p)`; low-injection terms ~1e-12.

**Every gate fails on v0.13.0.** Running the file against the untouched main checkout gives
21 failed, 1 passed (the passing one is off-a, which asserts the v0.13.0 digest). Each physics gate
also asserts its own "ON is load-bearing" margin, so a flag that was silently ignored could not pass.

Fixture lessons recorded in the tests:

* **The slab contact (corrected 2026-11-03).** The slabs first used a Potential-only contact, so no
  carrier was pinned anywhere: a floating body. In 1-D steady state with zero flux at both ends, the
  sum of all electron and hole continuity rows is identically zero (charge conservation), so the DC
  Jacobian is rank-deficient by exactly one at every G. Measured on the R26-a ramp: the continuity
  column sums are <= 1.1e-16 of the column norms in all 94 Newton matrices, and the physically scaled
  smallest singular value is 1.3-1.6e-15. The body potential was then set by round-off. A 0.1 V body
  shift re-converged in place; the far-end potential differed by 0.16 mV between scipy's and Elmer's
  OpenBLAS; and R26-a[1e33] hit an exact zero pivot ("umfpack numeric failed") on scipy's OpenBLAS
  only, after the DEVSIM 2.11 BLAS pin. The contact now pins psi and the MAJORITY density and leaves
  the minority row zero-flux. The steady state is unchanged: the minority rows force
  sum vol q (G - U) = 0, so the contact exchanges no net carriers (measured 3e-17 of q G L). The
  scaled smallest singular value is now 1.5e-5, and the Newton matrices and results are bit-identical
  on both OpenBLAS builds. `test_slab_is_well_posed` gates it and fails on the old fixture under both.
* With that contact, a Debye layer one node thick forms under injection. Measured on R26-a at
  G = 1e33: n(0) = 0.53 n_far and p(0) = 1.87 p_far, both within 1.5 % at the next node. In a slab
  much shorter than L, both quasi-Fermi levels are flat, so n p is uniform but n and p are not. The
  first version of R26-a put an Auger root in such a slab and measured a 5e-4 shift. The root gates
  therefore use slabs of 20+ diffusion lengths.
* A radiative-ONLY slab (tau_SRH = 1e3 s) first failed with UMFPACK declaring the Jacobian singular
  mid-ramp. That was put down to its weak `dU/dp = B n0` pinning. It was the floating-body null mode:
  on the old fixture the 200 nm radiative-only ramp still fails under both OpenBLAS builds, and on the
  majority-contact fixture it lands on the brentq root (0, and 2.2e-16 for the 3 mm slab). R26-d
  still adds a weak SRH term (10 %) and keeps the radiative-only statement node-wise.

Other regression checks: `tests/test_devsim_smoke.py`, `test_stacked3d_spec.py`,
`test_dc_solve_gummel.py`, `test_audit_2026_07_17_wave7_carriers.py` and `test_api_surface.py` give
54 passed (pytest -n 2). The new file: 22 passed in ~36 s.

## Limits

* **No Mott transition / impurity-band model.** Above about 3.7e18 cm^-3 (Si:P) real dopants are
  fully ionized. This model still freezes them out, giving 39 % ionization at 1e19 cm^-3 and 300 K.
  Altermatt 2006's non-localized fraction is not implemented. Use incomplete ionization only below
  the Mott density, or switch on Pearson-Bardeen lowering with a user-chosen alpha.
* **Boltzmann occupancy** of the dopant level (`n/N_c`, not `exp(eta)`) even with the FD g-factor on.
  This is accurate where freeze-out matters (non-degenerate) and conservative where it does not.
* **Schenk 1998 BGN is NOT implemented.** It needs carrier-density-dependent band edges (`D_c(n, p)`),
  whose gradients enter the SG flux and the Jacobian through the solution variables, plus about 20
  Pade-fit parameters that could not be verified offline. Deferred.
* **Constant C, B** (300 K values). No Huldt/Dziewior T-dependence, no Altermatt 2005 injection-
  dependent B, no Kerr-Cuevas / Richter ambipolar Auger. Coulomb enhancement uses equilibrium n0/p0
  (the paper's low-injection form).
* `T_K` sets V_t only; the material inputs at T are the caller's job.
* The equilibrium (unipolar ITO, `physics_equilibrium`) and unipolar DD paths are untouched. Auger and
  radiative need holes. Incomplete ionization and BGN are not meaningful for the degenerate,
  fixed-n_bg ITO model, which sits far above the Mott density.
* The 2-D layered builder hardcodes `fd_enhancement=True`, so apparent BGN there is subject to the
  double-counting advisory on degenerate layers. **Superseded 2026-11-06 (v0.15):**
  `TransportModel.carrier_statistics` ("boltzmann" / "fermi_dirac") is threaded through both builders;
  see `2026-11-06-v0.15-si-bundles.md`.
* No 3-D (gmsh) solve with options was run in this workstream. Only the shared `phi_bi` kernel, the
  spec validation and the `from_design` threading are gated.

## Open questions (user decisions)

1. **Default parameter sets per material.** Nothing is auto-applied. Should `materials/` ship Si
   (and later GaAs/InP) `BipolarOptions` defaults, and which BGN model (old Slotboom vs Klaassen vs
   del Alamo) is the house choice?
2. **Reference n_i pairing for apparent BGN.** Each set was fitted against a particular n_i at 300 K.
   **Corrected 2026-11-06:** the earlier statement here, "the commonly quoted modern pairing is
   n_i = 1.0e10 cm^-3 (Sproul & Green)", was wrong. No shipped set was fitted with 1.0e10: old Slotboom
   about 1.44e10 (fitted on p-type bases), del Alamo about 1.36-1.45e10 (both inferred from the
   sources), Klaassen 1992 1.0e10 or 1.08e10 (unverified). The modern consensus value is 9.65e9 cm^-3
   (Altermatt et al., J. Appl. Phys. 93, 1598 (2003)), and it belongs with Fermi-Dirac statistics and
   a real narrowing. Pairing old Slotboom or del Alamo with 9.65e9 under-predicts n_ie^2 by about 2.3x
   (about 21 meV). The presets stay unpaired (nothing checks n_i); see
   `2026-11-06-v0.14-materials-corrections.md`. The shipped validation fixtures use 1.0e10 as a fixture
   value only: their oracles take the same n_i, so no gate depends on the pairing.
3. **III-V radiative/Auger presets** were deliberately NOT shipped. Literature B for GaAs ranges about
   1e-10 to 7e-10 cm^3/s depending on photon recycling, and the user should choose.
4. **Altermatt 1997 constants** (13 / 3.3e17 / 0.66, 7.5 / 7.0e17 / 0.63): VERIFIED 2026-11-03 against
   the published restatement of Altermatt, Schmidt, Heiser & Aberle, J. Appl. Phys. 82, 4938 (1997)
   on pv-manufacturing.org "Recombination processes" (g_eeh(n0) = 1 + 13[1 - tanh((n0/3.3e17)^0.66)],
   g_ehh(p0) = 1 + 7.5[1 - tanh((p0/7.0e17)^0.63)], C_n = 2.8e-31 / C_p = 9.9e-32 cm^6/s Dziewior &
   Schmid). The primary PDF itself was not reachable; resolved, no longer an open question.
5. **Pearson-Bardeen alpha**: no default value is shipped (0 = off). Since 2026-11-06 an alpha that
   zeroes the ionization energy below 1e17 cm^-3 raises `IonizationUnitsError` (the eV*cm vs eV*m
   trap), `IonizationModel.pearson_bardeen(..., alpha_D_eVcm=...)` converts a literature value, and
   regions above the Mott density emit an `UnreliableResultWarning`.

## Merge notes for the integrator

* D1c (traps): with BGN on, `install_bipolar_options` re-defines `USRH` as `srh_expr(True)` (n_ie^2,
  n1/p1 scaled by n_ie/n_i). If D1c generalizes SRH (E_t, Dit), the BGN form must be applied to D1c's
  expression: `n_i^2 -> n_ie^2`, `n1 -> n1*n_ie/n_i`, `p1 -> p1*n_ie/n_i`. Keep that substitution in ONE
  builder.
* D1a (heterojunctions): an affinity/band offset is another band-edge field. It belongs in the same
  `psi_n = psi + D_c`, `psi_p = psi - D_v` quasi-field slots (`vdiff_g`, `vdiff_gp`) and in
  `equilibrium_fields` `psi_eq`, not in a parallel copy.
* `tests/_bipolar_calllog.py` digest: a sibling change that deliberately alters the DEFAULT bipolar
  strings (any `ds.*` model/equation/parameter call made by `setup_bipolar_region`, the seed models, or
  the contact recipes) changes the digest. Re-record with `python tests/_bipolar_calllog.py` from the
  reference checkout. That is a deliberate signal, not a flake.

## Release-note lines

* **Bipolar drift-diffusion: opt-in Auger and radiative recombination (audit R26).**
  `setup_bipolar_region(options=BipolarOptions(auger=..., radiative=...))`. Cited Si presets:
  Dziewior-Schmid 1977, Altermatt 1997 Coulomb-enhanced, Trupke 2003. Uniform-generation slab roots
  match brentq to <= 4.4e-16, and tau_A = 1/(C n0^2) to 3e-9.
* **Bipolar drift-diffusion: opt-in bandgap narrowing (audit R27).** Slotboom (1977, Klaassen 1992)
  and del Alamo models, implemented as an explicit band-edge split (`conduction_fraction`) entering
  both Scharfetter-Gummel currents as quasi-fields, plus n_ie in SRH, Auger, radiative, the contacts
  and the seed. A graded junction is current-free at equilibrium for any split. Long-diode J0 matches
  q n_ie^2 D_p/(N_D L_p) to 6e-4.
* **Bipolar drift-diffusion: opt-in incomplete dopant ionization (audit R27)** with Boltzmann
  occupancy (g_D = 2, g_A = 4), optional Pearson-Bardeen lowering, and a region temperature `T_K`.
  Si:P freeze-out from 50 to 300 K, compensated or not, matches charge neutrality to 7e-15. No Mott
  transition.
* `TransportModel.bipolar_options` and `Stacked3DSpec.bipolar_options` thread the options through
  the 2-D layered and 3-D builders. The 3-D gate reference `phi_bi` uses the same kernel.
* Defaults off: the bipolar DEVSIM physics definition is byte-identical to v0.13.0 (call-log digest gate).
