# One Scharfetter-Gummel flux kernel, sign-selected two-Bernoulli bracket (2026-11-04)

Branch `wip/dv_sgflux` (from `6595bde`, head of `feat/devsim-2.11-v0.14`). User decision 2026-11-04:
replace every `kahan3` B(v) + v flux with an accurate form, accuracy over byte-identity. DEVSIM 2.11.2,
scipy's OpenBLAS (the `devsim_env` pin), Windows. "Measured" = run on this box; "inferred" says so.

## What changed

* **One kernel.** `dynameta/carriers/sg_flux.py` (`install_sg_current`, `sg_bracket`,
  `bernoulli_expr`) builds every Scharfetter-Gummel edge flux: unipolar DD electrons
  (`physics_drift_diffusion`), bipolar DD electrons and holes (`physics_bipolar_dd`), the unipolar DG
  and the bipolar DG fluxes (`physics_density_gradient`), and the validation-side frozen-Lambda flux
  (`validation/dg_dd_in_newton.py` GATE C). `_JN_DG`, `_JP_DG`, `_JN_DG_BIPOLAR`, `_JP_DG_BIPOLAR`,
  `_clamped_bernoulli` and `_upper_clamped_bernoulli` are gone.
* **The form.** Both B(v) and B(-v) are evaluated (DEVSIM's accurate `B`); the flux bracket picks,
  by the sign of v, the spelling that uses the Bernoulli identity only on the non-negative argument
  (section 1.3). Not the plain bracket `n@n1 B(-v) - n@n0 B(v)` of a4f00d9: same large-|v| accuracy,
  but measurably noisier near equilibrium (section 1.3) -- the one deliberate deviation from the
  literal decision; kept by the coordinator's decision (2026-11-04).
* **Clamps.** Classical DD unclamped (as its B(v) always was); DG fluxes B(min(arg, 200)) only; the
  unipolar DG's two-sided clamp removed (section 1.4).
* **Every consumer inherits it** through the edge-model names, which are unchanged
  (`ElectronCurrent`, `HoleCurrent`, `ElectronCurrentDG`, `HoleCurrentDG`): 1-D/2-D/3-D, DC, Gummel,
  transient, ssac, noise (the adjoint uses the same Jacobian), contact-current extraction,
  `impact_ionization` (reads `ElectronCurrent` / `HoleCurrent`), gated DD, `LayeredDevsimBuilder`,
  `devsim_3d`, heterostacks (BGN / band-offset quasi-fields redefine `vdiff_g` / `vdiff_gp`, which
  the kernel references by name). New edge models: `Bern_g_m`, `Bern_gp_m`, `Bern_g_dg_m`
  (unipolar DG), `Bern_g_f_m` (validation) -- each B(-v) with its derivatives.

## 1. Error analysis

### 1.1 kahan3's B(v) + v (measured vs mpmath, 50 digits, DEVSIM 2.11.2)

`kahan3(n@n1 B(v), n@n1 v, -n@n0 B(v))` spells B(-v) as B(v) + v. For v << 0 that is the difference
of two O(|v|) numbers leaving B(-v) ~ |v| e^-|v|: relative error ~ eps |v| e^|v|.

| v | -10 | -15 | -18 | -21 | -25 | -27.6 | -30 | -33 | -36 | -40 |
|---|---|---|---|---|---|---|---|---|---|---|
| B(v) + v vs B(-v) | 1.5e-12 | 1.4e-10 | 8.6e-9 | 1.9e-7 | 2.9e-6 | 6.5e-5 | 1.0e-3 | 1.7e-2 | 0.15 | 1.0 |
| dBdx(v) + 1 vs -B'(-v) | 2.8e-13 | 4.1e-11 | 1.6e-10 | 3.8e-9 | 1.8e-7 | 1.5e-6 | 2.8e-6 | 1.3e-4 | 1.7e-3 | 0.34 |
| DEVSIM B(-v), dBdx(-v) | <= 3.1e-16 at every v in 0.3 .. 60 | | | | | | | | | |

On the actual flux edges (one 1 nm edge per family, each term isolated, `tests/test_devsim_sg_flux.py`)
the 6595bde code measured, worst value / Jacobian error: |v| = 40: 1.0 / 0.18-0.64; 33: 1.1-2.3e-2 /
4.4e-5-9.7e-4; 27.6: 0.7-1.6e-4 / 4.1e-8-4.6e-7; 21: 0.6-1.3e-7 / 3.2e-10-8.0e-9; 10: 0.5-2.3e-12 /
1.1-2.3e-13 (unipolar DD, bipolar DD n and p, unipolar DG). The bipolar DG already used two
Bernoullis (a4f00d9) and measured <= 3.3e-16.

### 1.2 DEVSIM's B and dBdx (measured)

* **Overflow-safe everywhere.** B(+-200) = 2.77e-85 / 200; B(+-700) = 6.9e-302 / 700; B(+-1e5),
  B(+-1e300) = 0 / |x|; dBdx -> -0 / -1. No inf, no NaN, at any finite argument.
* **dBdx is inaccurate near 0** (vs mpmath): 2.1e-14 at |v| = 1e-2, 4.1e-13 at 1e-3, 2.0e-12 at 1e-4,
  2.4e-10 at 1e-6, 1.5-2.6e-7 at 1e-9, 4.8e-5 at 1e-12. B itself is <= 2.2e-16 there. This only
  touches the Jacobian (Newton rate, and the ssac/noise linear systems), and it is what separates the
  two two-Bernoulli brackets below. It is DEVSIM's, not this change's (open question 3).

### 1.3 Which two-Bernoulli bracket: plain vs sign-selected

Plain (a4f00d9): `n@n1 B(-v) - n@n0 B(v)`. Sign-selected (shipped):

    electrons:  v >= 0:  kahan3(n@n1 B(v),  n@n1 v,      -n@n0 B(v))      [B(-v) = B(v) + v]
                v <  0:  kahan3(n@n1 B(-v), -n@n0 B(-v), n@n0 v)        [B(v) = B(-v) - v]
    holes:      v >= 0:  kahan3(p@n1 B(v),  -p@n0 B(v),  -p@n0 v)
                v <  0:  kahan3(p@n1 B(-v), -p@n1 v,     -p@n0 B(-v))

Both are free of the B(v) + v cancellation (the identity is only used where B(|v|) + |v| adds two
positive numbers). They differ near equilibrium: there |v| is small, n@n1 ~ n@n0, and the flux is a
tiny difference of two large terms. With ONE shared Bernoulli per branch its rounding -- and in the
Jacobian the rounding of dBdx, which is poor near 0 (1.2) -- multiplies n@n1 - n@n0 (common mode);
the plain bracket's two independently evaluated Bernoullis multiply n@n1 + n@n0. Measured on
`validation/extended_precision.py`'s 1-D Si diode at 0.1 V forward, 40 extra converged DC passes
with an ssac solve after each (std / mean; DC floor = median of the max relative n, p change per
extra pass):

| form | ssac G(1 Hz) | ssac G(1 MHz) | DC floor |
|---|---|---|---|
| kahan3 (6595bde) | 3.2e-4 | 1.5e-7 | 1.6e-15 |
| plain two-Bernoulli | **2.2e-3** | **1.1e-6** | 3.3e-15 |
| sign-selected (shipped) | 5.7e-4 | 2.6e-7 | 8.9e-16 |

Large-|v| accuracy of the sign-selected form, every family: value <= 3.3e-16, Jacobian <= 8.9e-16 to
|v| = 40 (the plain bracket's numbers). The branches agree analytically, with every derivative, at
v = 0; numerically by ulps. Inferred, not measured: the sign-selected form's residual 1.8x over kahan3
in ssac noise is the ulp-level step between the branches as v changes sign in neutral regions.

### 1.4 Clamping (measured)

* Classical DD: none. Its B(v) was never clamped, and 1.2 shows none is needed.
* DG: `B(min(arg, 200))`, kept from a4f00d9. It moves B by < 2.8e-85 where it binds (a value no-op),
  belt-and-braces for a hard wall's log-singular Lambda in Newton transients.
* Never a lower clamp. Under the plain bracket the unipolar DG's two-sided clamp would freeze both
  Bernoullis for |v| > 200 and zero a wall node's Jacobian column (measured 2026-11-03 on the bipolar
  DG: "Convergence failure!" / "umfpack numeric failed"). The sign-selected branch for |v| > 200
  always carries the linear c v term, so no Jacobian column can vanish.

### 1.5 Jacobian and Newton

Derivatives are DEVSIM's `simplify(diff(...))` of the new string (`eq_registry.edge_with_derivs`; the
derivative of an `ifelse` is the `ifelse` of the branch derivatives). Gated against mpmath (1.1 row,
<= 8.9e-16 for |v| >= 0.1). Newton iteration counts, this tree vs the reference, every solve:

| case | 6595bde | this tree |
|---|---|---|
| `tests/_sgflux_reference` cl_fd (FD diode: equilibrium, 0..+0.4, ..-0.5 V) | 18 7 7 7 7 8 8 9 9 | identical |
| cl_bz (Boltzmann, circuit contact) | 20 8 8 8 8 8 9 9 10 | identical |
| dgcl (unipolar DG hard wall, gamma 0.05..1) | 77 5 9 11 16 | 77 5 9 12 16 |
| unipolar DG + `setup_dg_insulator_wall` (section 6) | 1 10 5 5 5 6 | identical |
| `sg_flux_cross_version` GATE C: 1-D FD diode ramps to +0.3 / -0.5 V, 2-D layered diode +0.3 / -0.4 V (82 solves) | 268 | 268, identical per solve |

Quadratic convergence is unchanged (every DD bias step still ends in 4-5 iterations). The one +1 is a
DG gamma step from a state that differs at round-off.

## 2. Inventory: fluxes changed

| family | module | models (unchanged names) | before | after |
|---|---|---|---|---|
| unipolar DD electrons (1/2/3-D, field mobility, gated DD, layered builder) | `physics_drift_diffusion` | `ElectronCurrent`, `Bern_g`, new `Bern_g_m` | kahan3 | kernel, unclamped |
| bipolar DD electrons | `physics_bipolar_dd` | `ElectronCurrent`, `Bern_g`, new `Bern_g_m` | kahan3 | kernel, unclamped |
| bipolar DD holes | `physics_bipolar_dd` | `HoleCurrent`, `Bern_gp`, new `Bern_gp_m` | kahan3 | kernel, unclamped |
| unipolar DG electrons | `physics_density_gradient.setup_dg_quantum_correction` | `ElectronCurrentDG`, `Bern_g_dg` (two-sided clamp), new `Bern_g_dg_m` | kahan3 | kernel, upper clamp |
| bipolar DG electrons / holes | `physics_density_gradient.setup_bipolar_dg` | `ElectronCurrentDG` / `HoleCurrentDG`, `Bern_g(p)_dg(_m)` | plain two-Bernoulli | kernel, upper clamp |
| frozen-Lambda reference flux | `validation/dg_dd_in_newton.py` GATE C | `ElectronCurrentF`, `Bern_g_f(_m)` | kahan3 | kernel |

Searched and NOT an SG flux (unchanged): heterojunction interface fluxes (thermionic emission /
continuous QFL are interface models, no Bernoulli); BTBT (`tunneling`, a generation term); the
Poisson charge `kahan3(Holes, -Electrons, N)` (a sum, not a Bernoulli); `noise_analysis.bernoulli`
(numpy `x/expm1(x)`, the SG-segment noise weights `n0 B(v) + n1 B(-v)`, a sum of positives with no
cancellation). Consumers inherit the kernel by model name (see "What changed").

## 3. Gates (each fails on 6595bde)

| gate | where | this tree | on 6595bde |
|---|---|---|---|
| value + Jacobian vs mpmath, 6 families, abs(v) <= 40, 1e-13 | `tests/test_devsim_sg_flux.py::test_sg_flux_value_and_jacobian_are_accurate_at_large_drift` | <= 3.3e-16 / 8.9e-16 | 4 families FAIL (1.0 / 0.18-0.64) |
| every family built inside the kernel (run-time spy) | `::test_every_flux_family_is_built_by_the_one_sg_kernel` | pass | FAIL (no `sg_flux`) |
| defaults-off fixtures vs 6595bde, round-off tolerance | `::test_defaults_off_fixtures_equal_6595bde_to_flux_round_off` | pass | trivially equal (it is the reference) |
| the same gate catches mu_n x (1 + 1e-6) | `::test_cross_version_gate_still_catches_physics` | fails as required | -- |
| bipolar DG flux (existing) | `tests/test_bipolar_density_gradient.py::test_dg_edge_flux_is_accurate_at_large_drift` | 3.3e-16 / 2.2e-16 | passed (a4f00d9) |
| validation GATE A / A' / B / C | `validation/sg_flux_cross_version.py` | PASS | A fails (4 families); A' is the old strings failing in this tree |

## 4. Byte-identity guards converted

Every guard now pins (a) the definition, re-recorded with a dated comment, and (b) a numeric
comparison with the pre-change tree on the same fixture, at a tolerance derived from the measured
round-off of the change; each still fails on a real physics change.

| guard | old form | new form | tolerance and derivation |
|---|---|---|---|
| `tests/_bipolar_calllog.py` digest (`test_bipolar_recombination_bgn_ionization`, `test_bipolar_density_gradient`) | sha256 `ac63536e...` (217 calls), v0.13.0 | sha256 `35427315...` (237 calls), re-recorded 2026-11-04. The old and new logs were diffed: they differ ONLY in `ElectronCurrent` / `HoleCurrent` (+ derivatives) and the new `Bern_g_m` / `Bern_gp_m` | definition: exact |
| `tests/_dg_calllog.py` digest (`test_bipolar_density_gradient`) | sha256 `6bcbe925...` (127 calls), e46bd25 | sha256 `bc52c792...` (139 calls); differs only in the flux models, `Bern_g_dg`'s clamp and the new `Bern_g_m` / `Bern_g_dg_m` | definition: exact |
| numeric half of both digests | -- | `tests/test_devsim_sg_flux.py::test_defaults_off_fixtures_equal_6595bde_to_flux_round_off`: the calllog fixtures SOLVED (FD diode, Boltzmann circuit diode, DG hard-wall bar; 21 arrays) vs `tests/data/sgflux_6595bde.json`, recorded on 6595bde (`tests/_sgflux_reference.py --record`) | Densities entry-wise and potentials relative to their max: measured 7.7e-15 / 1.9e-16 -> 1e-13 (13x). Currents: abs(dI) <= 1e-11 abs(I) + 4 eps T, T = q mu V_t c / h of the contact edge. A majority-carrier `get_contact_current` is eps T noise in EITHER spelling (the FD diode's p-contact hole current moved 0.15 / 0.24 eps T = 2.1e-5 / 5.4e-2 RELATIVE); every other current <= 6.5e-13 relative (15x / 9x margins). mu_n x (1 + 1e-6) fails it (`test_cross_version_gate_still_catches_physics`; minority electrons move 4e-7) |
| `validation/traps_oracles.py` gate F "byte-identical vs main" | sha256 of 5 arrays x 3 fixtures, main 5cbd998 vs branch | numeric per array (`cross_version_rows`, `TOL_F`); hashes still printed | Densities entry-wise, Potential / its max, PotentialNodeCharge / q(n+p+abs N), USRH / its term size (n p + n_i^2)/(taup(n+n1)+taun(p+p1)). Measured worst 4.1e-15 (6595bde reproduced main's sha256 exactly) -> 1e-13 (24x) |
| `validation/extended_precision.py` GATE C (manual `--hashes` across commits) | 7 sha256 fingerprints | `--compare DIR` (+ `--dump`): the same arrays, numerically, per class (`TOL_C`) | dc 1e-13 (measured 2.2e-15); small-signal C / noise S / transient 1e-9 (measured 3.5e-11; same-code C spread ~4e-11); ssac G 1e-4 of max G (measured 2.6e-6: Re Y(1 Hz) is round-off in double in both forms -- repeat spread 3-6e-4, 2e-4..3e-3 off the extended DC dI/dV). PASS vs 6595bde |
| integration doc sec. 4 "D1d solution arrays" sha256 (script never committed) | sha256 `a500ca93...`, main vs branch | `validation/sg_flux_cross_version.py` GATE B: `bipolar_diode` FD at +0.3 / -0.5 V and `bipolar_diode_2d` via `LayeredDevsimBuilder` at +0.3 / -0.4 V, 16 arrays, main vs this tree | fields 1e-11 (measured <= 6.5e-15); currents 1e-11 abs(I) + 4 eps T, T an upper bound of the region's edge terms |
| integration doc sec. 4 table; D2b doc fingerprint table | recorded hashes | dated notes pointing here | -- |

Unchanged (within-tree identity, still valid): `test_traps.py` (e),
`test_inactive_options_bundle_is_bitwise_identical_to_none`, `test_devsim_smoke` chunking,
`test_dc_solve_gummel`, `test_devsim_env` reset rebuild, `test_extended_precision` gate D,
`test_devsim_noise` read-only, `test_devsim_tunneling` post-processor, `ac_capacitance`'s reset note.

## 5. Pins and printed values that moved

**Tests.** No assertion changed except the two digests above (call counts 217 -> 237, 127 -> 139).
Every other test passes unchanged. One documented measurement moved:
`test_inversion_equilibrium_flat_qfls_and_np` (bipolar DG GATE E) reads max abs(phi_n) 3.7e-15 V (was
3.3e-16) and n p check 1.5e-13 (was 7.1e-15), against gates 1e-11 / 1e-10. Not the bracket choice: the
plain bracket gives 3.0e-15 / 1.1e-13 on this tree. The classical seed state differs at round-off, and
both values are the double readout floor of psi + Lambda - V_t ln(n/n_i) at abs(Lambda_wall) = 0.71 V
(the n p check is that error / V_t). Docstring updated.

**Validations.** Every printed number compared token by token, 6595bde vs this tree, threshold 1e-12.
A 62c4097 vs 6595bde run gives the run-to-run baseline: only wall times differ.

| validation | quantity | 6595bde | this tree | explanation |
|---|---|---|---|---|
| bipolar_density_gradient GATE E | max abs(phi_n); n p relation | 3.3e-16 V; 7.1e-15 | 3.7e-15 V; 1.5e-13 | double readout floor (above) |
| devsim_noise B | FDT residuals S/(4kT Re Y) - 1, low f | 4.6-7.1e-6 | 0.4-5.4e-6 | Re Y at low f is ssac round-off (section 1.3); gate 1e-4 |
| devsim_noise C (0.5 V, shot) | S/(4kT ReY - 2qI) at 1 Hz / 1 kHz | 1.009 / 1.0088 | 1.010 / 1.0094 | same: Re Y(1 Hz) round-off ~1e-3; gate 1.5 %, so its margin is 1.5x in both trees (the 1 Hz point of this gate is round-off-limited, not physics-limited) |
| devsim_noise G (layered, Vg 0) | FDT residual at 1 Hz | -1.31e-11 | -7.77e-12 | round-off |
| transient_diode | reverse step -0.5 -> -1.0 V: steps, settle_rel | 62, 2.49e-5 | 67, 1.62e-5 | adaptive step control on a 1.8e-4 A/m^2 reverse current whose contact read-back is eps T noise; end current identical, 1.8339e-4 (inferred) |
| impact_ionization_isub GATE C | I_sub(0 V) | 7.890e-24 A | 8.091e-24 A | the zero-bias current is round-off ("negligible" in the gate) |
| dg_dd_in_newton A-D, gummel_vs_newton A | closures, conservation | 1e-16 level | 1e-16 level | round-off |
| traps_oracles F | output lines | hashes | numeric rows, worst 4.1e-15 | the conversion |

No result moved by more than the round-off of its own quantity. Every other printed value of the 33
devsim validations is identical to all printed digits.

## 6. Where the old flux was materially wrong

Max abs(v) per edge over every converged solve of the devsim validations (read-only
instrumentation on the 6595bde run; fields agree with this tree to 1e-14): classical DD <= 7.0
everywhere (impact_ionization_isub 6.99, bipolar_diode_2d 3.09, gated_dd_2d 2.93,
contact_current_drivers 2.52, everything else < 2.1). kahan3's B(v) + v error at abs(v) = 7 is ~ eps e^7
= 2.4e-13 of the B(-v) term: **the old classical flux was never materially wrong on any validation
fixture.** The only edges past -v = 10 are the bipolar DG MOS wall (-v = 24.7,
`bipolar_density_gradient`), fixed by a4f00d9.

Materially wrong (measured):
1. **Bipolar DG insulator walls** (a4f00d9, 2026-11-03): wall edge at -v ~ 27.6, wall Lambda 1.7e-10 V
   off, GATE C in a period-3 Newton cycle. Fixed before this change.
2. **Unipolar DG + `setup_dg_insulator_wall`: CONFIRMED.** 100 nm ITO bar at 4e26, u_floor 1e-6,
   frozen psi, gamma ramp to 1 at REL 1e-5: the wall edge sits at v = -26.15. 6595bde puts the wall
   node's Lambda at -2.567079886 V, this tree at -2.567076685 V; the extended-precision solve gives
   -2.567076685 V (both trees agree there to 3e-28). Old double error 3.2e-6 V (1.25e-6 relative), new
   1.7e-16. Only the wall node moves (Lambda[1], n[1] identical to 1e-16): its Lambda is fixed by that
   edge's flux alone. Same Newton iteration counts.
3. **Unipolar DG + `setup_dg_hard_wall`: REFUTED.** The hard wall pins Lambda, so the wall edge stays
   at abs(v) <= 1.05 (`_dg_calllog` and `test_density_gradient._hard_wall_bar` fixtures); every field
   moved <= 2e-16 of its scale (1.8e-13 entry-wise at a Lambda zero crossing). The coupled-psi variant
   of `_hard_wall_bar` fails to converge on BOTH trees (pre-existing, unrelated).
4. Any classical edge at -v >~ 15 (a coarse mesh across a strong depletion or heterostep): 1.4e-10 at
   15, 1.9e-7 at 21, 100 % at 40 before; exact now. No validation fixture reaches it at convergence
   (Newton transients may; the instrumentation saw converged states only).

## 7. Verification

* `pytest -n 2` over the 53 test files that touch devsim or the carriers modules (the 52 of the
  6595bde count + `test_devsim_sg_flux.py`): **989 passed, 3 xfailed** (the same three strict
  xfails), 35 min. 6595bde: 976 passed, 3 xfailed (52 files).
* `python -m validation.run_all --tier full` over the 33 devsim validations + the new
  `sg_flux_cross_version`: **34/34 PASS** after one environmental rerun. In the first run
  `carriers_3d_dd` failed at `create_device`: the concurrent 6595bde comparison run wrote the same
  hard-coded `~/.dynameta/_devsim3d.msh` at the same moment (shared-path collision; already fixed on the
  integration branch by 4ff95c2, the default tensor mesh goes to DEVSIM in memory -- not touched here).
  The 3-D scripts (`carriers_3d*`, `devsim_3d_mesh`; 11) were rerun serially: 11/11 PASS, every printed
  value identical to 6595bde.
* `ruff check dynameta validation tests`: clean.

## 8. Open questions (user decisions)

1. **Sign-selected vs plain bracket**: DECIDED (coordinator, 2026-11-04) -- keep the sign-selected
   form (section 1.3).
2. **DEVSIM's dBdx near 0** (follow-up, not implemented): vs mpmath 2.1e-14 at abs(v) = 1e-2, 4.1e-13 at
   1e-3, 2.0e-12 at 1e-4, 2.4e-10 at 1e-6, 1.5-2.6e-7 at 1e-9, 4.8e-5 at 1e-12 (B itself <= 2.2e-16).
   It reaches only Jacobians (Newton rate) and the ssac / noise linear systems, where it is the
   residual round-off source in quasi-neutral regions. Options: a series branch in `bernoulli_expr`
   (ifelse(abs(x) < x0, 1 - x/2 + x^2/12 - x^4/720, B(x)), whose DEVSIM derivative is then exact), or
   an upstream report.
3. **ssac Re Y at low frequency** is round-off in double in every flux form (repeat spread 3-6e-4 and
   2e-4..3e-3 off the DC dI/dV oracle on the 0.1 V Si diode; the D2b note's p+n case is worse). Not
   new; recorded so the 0.3 % move of G(1 Hz) is not read as physics.

## Release-note lines

* **One Scharfetter-Gummel flux kernel** (`dynameta.carriers.sg_flux`): every drift-diffusion and
  density-gradient flux (unipolar / bipolar, electrons / holes; 1-D/2-D/3-D; DC / transient / ssac /
  noise / Gummel) is built by `install_sg_current`. It evaluates B(v) and B(-v) and combines them
  sign-selected, replacing DEVSIM's `kahan3(n1 B(v), n1 v, -n0 B(v))` spelling, whose B(v) + v cancels
  for v << 0 (1.9e-7 at v = -21, 100 % at -40). Flux and Jacobian accurate to <= 8.9e-16 at any drift.
* **Results move only at round-off** on every validation fixture (no classical fixture exceeds
  abs(v) = 7). The unipolar DG with `setup_dg_insulator_wall` is now accurate at the wall node (its
  Lambda was 3.2e-6 V off).
* **Defaults-off guards are numeric**: call-log digests re-recorded; the solved fixtures, traps gate
  F, extended-precision gate C (`--compare`) and the D1d arrays (`validation/sg_flux_cross_version.py`)
  compare with the pre-change tree at measured-round-off tolerances.
* The unipolar DG's two-sided Bernoulli clamp is replaced by the upper-only clamp the bipolar DG uses.
