# DEVSIM phase integration: D1b + D1c + D1d + D2a + D1a consolidated (v0.14.0)

Branch `feat/devsim-2.11-v0.14` (worktree `DynaMeta_wt/dv_integ`). Merged on top of D1b (`d3a7012`,
already on the branch):

| Commit | What |
|---|---|
| `71978e9` | merge `wip/dv_tunnel` (D1d: BTBT in the Newton, post-processed oxide tunneling) |
| `2e4b8ee` | merge `wip/dv_traps` (D1c: SRH at E_t, multi-level trapped charge, interface traps) |
| `6baa9a6` | merge `wip/dv_noise` (D2a: impedance-field small-signal noise) |
| `b5e8e47` | consolidation: one term mechanism, one SRH kernel, one contact rule, noise on the same inventory, composition gates |
| `da7614f` | merge `wip/dv_hetero` (D1a: Anderson alignment, thermionic / continuous-QFL heterointerfaces), with the contact-rule and BGN-interface composition (section 2.8) |
| (last commit) | the D1a x D1b / D2a composition gates, the hetero noise builder, this note's final counts |

The first three merge commits are plain unions of the conflicting hunks: keyword lists, dataclass
fields and builder threading, each with its own validation. The design work for D1b/D1c/D1d/D2a is in
`b5e8e47`; the D1a composition is in its merge commit `da7614f`.

## 1. What collided

Before the consolidation the branches used three separate ways to add a generation or charge term to
the bipolar continuity / Poisson equations:

* **D1b** installed its options last. It re-defined `USRH`, `ElectronGeneration`/`HoleGeneration` and
  `PotentialNodeCharge` **by name**.
* **D1c** added the additive hooks `add_recombination_term` / `add_charge_term`. These rebuild the
  composites from `RTerm_*` / `QTerm_*` node models. It also summed its bulk trap rates **into the
  `USRH` string**.
* **D1d** wired BTBT straight into the continuity equations' edge/element volume slots with
  `eq_registry.amend_region_equation`.

Turning on several of these together failed in three ways:

* The D1b options overwrote D1c's trapped charge (`PotentialNodeCharge`).
* The D1b options overwrote every `RTerm_*`, including the interface SRH (`ElectronGeneration`).
* The D1b options overwrote the trap rates (`USRH`).

The D2a noise rebuilt its GR sources from a **parallel list**: `srh_expr` plus `recombination_exprs`,
selected by which parameters were present. So:

* the D1c trap levels and interface SRH were invisible to the noise;
* the D1d BTBT was invisible to the noise;
* the D2a noise refused any equation that carried an `edge_volume_model`.

## 2. Consolidation decisions

### 2.1 One term mechanism: D1c's hooks, extended to edge/element assembly (built on D1d's amend)

`physics_bipolar_dd.add_recombination_term(device, region, name, expr, *, pair=None, derivs=None, assembly="node")`
is now the only way a recombination or generation term enters a bipolar region. `add_charge_term` is
the only way a space charge enters.

**`assembly="node"`** (as in D1c):
* the rate goes into node model `RTerm_<name>`;
* the composite is `ElectronGeneration = -q (USRH + sum RTerm_*)`.

**`assembly="edge"`** (new; used by 1-D BTBT):
* the rate goes into edge model `RTermEdge_<name>`;
* the composite `ElectronGenerationEdge` is wired into the equations' `edge_volume_model`.

**`assembly="element"`** (new; used by 2-D/3-D BTBT):
* the node0 and node1 shares go into `RTermElem0_/RTermElem1_<name>`;
* the composites `ElectronGenerationElem0/1` are wired into `volume_node0_model` /
  `volume_node1_model`.

**Wiring the equation slot.** The slot is wired once, through D1d's `amend_region_equation`. That
function is atomic and refuses to overwrite a slot that a different module wired, so the guard is
kept. After that, each further term only rebuilds the composite. Terms with different names
therefore always sum. Re-adding a name replaces that one term.

**Pair rates.** `pair=` gives the pair rate `G + R` of the same process (`PTerm_*`, `PTermEdge_*`,
`PTermElem*_`); it is the noise GR source (section 2.5). A term with no pair rate is allowed. The
noise then refuses the region, so a term cannot silently go without noise.

**Charge.** `rebuild_poisson_charge` builds `PotentialNodeCharge` from:
* `IonizedNetDoping` if incomplete ionization installed it, otherwise `NetDoping`;
* plus every `QTerm_*`.

D1b's ionization therefore no longer re-defines the composite by name.

**Inventory.** `recombination_terms(device, region)` lists every term and its pair model, read back
from DEVSIM's model lists. There is no module state.

**Why this mechanism and not D1d's amend alone.** Node-local terms (Auger, radiative, trap levels,
interface SRH) have to sum into one node composite. DEVSIM equations have exactly one `node_model`
slot, so amending equation slots cannot compose them. D1c's rebuild-the-composite hook does compose
them. D1d's amend remains the primitive that the hook uses for the edge/element slots, so the code
has one mechanism layered on one primitive, not two parallel mechanisms.

**Who uses it now:**

| Term | Name |
|---|---|
| D1b Auger | `RTerm_auger`; the diagnostic `UAuger` is kept |
| D1b radiative | `RTerm_radiative`; the diagnostic `URad` is kept |
| D1b external G | `RTerm_gext` = `-G_ext` |
| D1c bulk trap levels | `RTerm_trap<k>`, now moved out of the `USRH` string; plus `QTerm_bulk_traps` |
| D1c interface traps | `RTerm_it<j>` / `QTerm_it<j>`, as before |
| D1d BTBT | `RTermEdge_btbt` (1-D) / `RTermElem0/1_btbt` (2-D/3-D); the diagnostics `BTBT_U`, `BTBT_U0/1` and `BTBT_Emag` are kept |

**Install order inside `setup_bipolar_region`:**
1. Base models and currents.
2. `bipolar_options.install_option_fields`: V_t at T_K, `n_ie`, the BGN quasi-fields `vdiff_g` /
   `vdiff_gp` (a transport change, not a term), and the ionization fields.
3. `USRH`, built by the kernel with the BGN substitution.
4. The trap levels.
5. `install_option_terms`: Auger, radiative, G, and the contact equilibrium fields.
6. The continuity equations.
7. BTBT.

Off by default: none of the new calls run, so the defaults-off physics definition is unchanged
(section 4).

### 2.2 One SRH kernel with the BGN substitution: `physics_traps.srh_rate_expr`

The D1b bandgap-narrowing substitution now lives in a single place,
`physics_traps.srh_reference_densities(n1, p1, bgn=...)`:

    n_i^2 -> n_ie^2 ,   n1 -> n1 n_ie/n_i ,   p1 -> p1 n_ie/n_i

`srh_rate_expr` and `srh_occupancy_expr` both take `bgn=`. `srh_rate_expr` also takes `pair=`, which
gives `(np + n_ie^2)/D`. This one kernel builds:
* the built-in channel (`USRH`);
* every `TrapLevel` rate and occupancy (`n1 = n_ie exp(E_t/kT)`, with kT at the region's T_K);
* every interface-trap bin. Its rate and pair use `taun -> 1`, `taup -> c_n/c_p`, scaled by
  `N_k c_n`, which is algebraically the old D1c form. The closed-form Jacobians use the same
  substituted `n1'`, `p1'`, `n_ie^2`.

`bipolar_options.srh_expr(bgn)` is kept as a one-line alias for the kernel. Both defaults are pinned:
* `bgn=False` is the legacy string, character for character (D1c test);
* `bgn=True` is D1b's former string, character for character.

**Trap occupancy charge uses n_ie: decided, and why.** The occupancy
`f_t = (tau_p n + tau_n p1)/(tau_p (n + n1) + tau_n (p + p1))` equals the Fermi function at equilibrium
only if `n1 p1` equals the equilibrium `n p`. With BGN on, that equilibrium product is `n_ie^2`.

If the occupancy kept `n1 p1 = n_i^2` while `n p = n_ie^2`, then the equilibrium occupancy would depend
on `tau_n/tau_p`. That is not a thermal distribution, and Poisson would carry a capture-ratio-dependent
charge at zero bias. So the occupancy, and with it the trapped charge, takes the same substitution as
the rate. Gate `test_trapped_charge_uses_the_narrowed_occupancy` checks it to 1e-12.

**Consequence (RESOLVED by the per-level energy reference, wip/dv_traps2; see the D1c doc section
9):** `E_t` is measured from the **narrowed effective intrinsic level** by default.
Band-referenced levels (`energy_reference="conduction"/"valence"`) are now available. Correction to
the next sentence: in this Boltzmann model gamma does not enter SRH at all (the narrowed E_c - E_ie is
gamma-independent). The references differ by exp(dEg/2kT), and exp(gamma dEg/kT) would apply only to a
level fixed against the UNnarrowed edges. This was already D1b's convention. With an asymmetric band-edge split
(`conduction_fraction != 1/2`), a level pinned to E_c would scale as `exp(gamma dEg/kT)` instead of
`exp(dEg/2kT)`. We ship the symmetric reference that D1b documented, and no other convention is
invented here.

### 2.3 BTBT's Hurkx D uses n_ie when BGN is on

`setup_bipolar_region` passes `n_i="n_ie"` to `setup_btbt` whenever BGN is on. `setup_btbt` then
evaluates the node model at each D node: `n_ie@n0` / `@n1` via `edge_from_node_model` in 1-D, and
`@en0` / `@en1` via `element_from_node_model` in 2-D/3-D. As a result, D = 0 at the narrowed
equilibrium `n p = n_ie^2`. The composition gate asserts that the BTBT edge expression carries
`n_ie@n0` and no `n_i`.

### 2.4 One contact rule (feasible, done)

`bipolar_options.equilibrium_fields`, the D1b numpy kernel behind contacts, seed and the 3-D gate
frame, now takes `traps=`. These are the charged bulk levels as `(E_t, N_t, kind)`, from
`physics_traps.equilibrium_trap_levels`; neutral levels are dropped. They enter the charge-neutral
root:

    f(n) = n - n_ie^2/n - N_D+(n) + N_A-(n_ie^2/n) - Q_t(n)/q = 0,
    Q_t/q = -N_t n/(n + n1) (acceptor-like),  +N_t n1/(n + n1) (donor-like),  n1 = n_ie exp(E_t/kT)

This is the equilibrium SRH occupancy, i.e. the Fermi function. It is the standard neutrality
condition with trapped charge (Shockley & Read 1952; Sze & Ng ch. 1) and exactly the brentq oracle of
D1c's gate (b), so no new physics is introduced.

Why the existing bisection still works:
* Every term is strictly increasing in n, so the root stays unique.
* The bracket widens by `sum N_t`, and the sign argument still holds.
* With no traps the root function is the old one, with the same arithmetic.

Where it is used:
* `setup_bipolar_region` installs `NeutralElectrons` / `EquilibriumPotential` / `SeedPsiShift`
  whenever BGN, ionization or a charged trap level is present. Both ohmic recipes (DC and circuit)
  and the seed already read these fields.
* `ohmic_built_in_offset_V(..., traps=)` gives the 3-D gate reference `phi_bi`. `devsim_3d` passes
  `spec.srh_traps`.

This closes D1c open question 2 and D1c's "trap-free contact" boundary-layer limit, and shares the
rule with D1b's incomplete ionization. Gate `test_charged_trap_contacts_pin_the_trap_aware_root`
checks the contact density against brentq to 1e-11. The trap moves the n-side contact density by
2e-4, so the check is load-bearing.

Interface traps are a sheet charge, not a bulk neutrality term, so they correctly stay out of the
contact rule.

### 2.5 Noise on the same inventory

`noise_analysis._gr_pair_per_node` builds each node's volume-integrated pair rate `P_i [1/s]` from
two sources:
* the base SRH pair from the kernel, `srh_rate_expr(bgn=..., pair=True)`;
* every `PTerm*` from `recombination_terms`.

Each term is assembled with the same weights its rate uses in the residual:

| Term kind | Weight |
|---|---|
| node | `NodeVolume` |
| edge | `EdgeNodeVolume` to both end nodes |
| element | `ElementNodeVolume` to the element edge's node0 / node1 |

The GR source is then `S_i = q^2 |y_p - y_n|^2 2 P_i`. Other details:
* The parallel option-sniffing list (`_pair_rate`, `_gr_pair_expr`) is deleted.
* `external_generation_shot=False` drops the term named `bipolar_options.GEXT_TERM`.
* The `edge_volume_model` refusal now admits the mechanism's own `ElectronGenerationEdge` /
  `HoleGenerationEdge`. A foreign volume model is still refused.
* Test B2 used to monkeypatch `recombination_exprs`. It now filters `PTerm_auger` / `PTerm_radiative`
  out of the noise inventory, with the same assertion. The behaviour is unchanged: 19/19.

**The BTBT pair rate.** The Hurkx split `(np +- n_ie^2)/((n + n_ie)(p + n_ie))` is the same
detailed-balance split as SRH (`hurkx_d_expr(pair=True)`).

* **2-D/3-D (element assembly):** the source is node-local, so the discrete identity is exact.
* **1-D (edge-mean D):** D is the mean of the two end nodes, so it is exact only to O(h^2) inside
  the junction. Measured: S/4kT Re Y - 1 <= 2.3e-6.

### 2.6 A latent D1d bug found by the noise gate (fixed)

`btbt_rate_expr` printed its coefficients with `%.17g`. For an integral float this produces an
**integer** literal, e.g. `1e12 -> "1000000000000"`. DEVSIM 2.11.2 clamps integer literals above
2^31-1 to 2147483647. Measured: `"10000000000000000"` evaluates to 2147483647.0, while `"1e16"`
evaluates to 1e16. So any integral A or B >= 2^31 was silently replaced.

The new BTBT FDT gate exposed it: A = 1e12 moved Re Y by only 8e-4, and the result was independent
of A. The fix is `%.17e`, which always produces a float literal.

The shipped `SILICON_HURKX` was not affected, because A = 4000000000000000.5 is non-integral and
B = 1.9e9 < 2^31. The parsed doubles are identical, so D1d's numbers are unchanged; its 36 tests
still pass.

### 2.7 Smaller fixes made while consolidating

* `Stacked3DSpec.from_design` now threads `TransportModel.btbt`. D1d added the spec field but not the
  threading.
* The kT of the trap levels follows `BipolarOptions.T_K`. Before, `n1_p1` always used `V_T`, which
  was wrong at T != T_REF. The bulk installer uses the region's `V_t`; interface traps read the
  region parameter. For E = 0, or with T_K unset, the arithmetic is unchanged.

### 2.8 D1a heterojunctions x D1b BGN x the contact rule (merge `da7614f`)

**One contact rule.** `physics_bipolar_dd.contact_potential_residual` is used by both the DC and the
circuit bipolar recipes:

    Potential = <bias or circuit node> + s(r) + psi_eq,own(r)

* `s(r)` is D1a's Anderson frame shift (`psi_offset_V`), computed by `anderson_alignment` from the
  UNNARROWED material edges. It is the affinity rule, a property of the material.
* `psi_eq,own(r)` is the contact node's equilibrium potential in the region's own (intrinsic-level)
  frame. It comes from `equilibrium_fields` (BGN band-edge split + incomplete-ionization root +
  charged-trap charge + FD) when the region carries it, otherwise from the legacy closed form.

The two pieces are disjoint: `s` is a per-region constant of the alignment, and every
doping-dependent shift lives in `psi_eq,own`. So nothing is counted twice, and there is no parallel
offset path:
* D1a's own `psi_offset_V` now feeds this one function.
* The circuit recipe gained `psi_offset_V` (0 leaves the string unchanged).
* The builder seed is `equilibrium_seed_psi_expr(SeedPsiShift or FD) + s`, the same composition.

**Decision on D1b's merge note.** D1b's note said band offsets belong in the `vdiff_g` / `vdiff_gp`
quasi-fields and in `psi_eq`.
* The quasi-field part is satisfied trivially. D1a splits regions at every material change, so the
  Anderson offset is constant inside a region; its in-region gradient, which is what the quasi-field
  slots carry, is zero, and the jump sits on the interface, where D1a's interface condition carries
  it.
* The `psi_eq` part is the rule above.

Moving the constant shift into `equilibrium_fields` as well would have been the double-counting
path, so it was not done.

**BGN at the heterointerface.** Lundstrom & Schuelke (IEEE TED 30, 1151 (1983)) treat affinity, gap
and heavy-doping shifts as one effective band-edge field. Following that, the D1a interface
activities read E_F against `E_c - q D_c` and `E_v + q D_v` of each node copy, and the TE barrier
top is the larger effective conduction edge (the smaller effective valence edge for holes). This is
implemented in `physics_heterojunction._bgn_interface_arguments`. Without BGN, D1a's strings are
unchanged.

Gate `test_heterojunction_with_bgn_effective_offsets_at_the_interface[auto|continuous]` checks
`dE_c,eff = dE_c - D_c,A + D_c,G` to 1e-6 eV. With the interface terms removed it misses by
**5.66 meV** on both interface models (0.24310 vs 0.24876 eV).

T_K != T_REF on a heterointerface with BGN was refused here (the interface was written at T_REF), and
without BGN it silently ran at T_REF. Superseded by the D1a temperature follow-up (wip/dv_hetero2): the
interface now runs at the regions' V_t (one source, BipolarOptions.T_K) and a non-isothermal stack is
refused -- docs/audit/2026-11-03-devsim-heterojunctions.md section 4b.

**Noise on heterointerfaces.**
* **Continuous-QFL heterointerface: covered.** On the FDT diode as a heterojunction (n_i 1e19 | 3e18,
  affinity step 0.05 eV, s(r_p) = 18.9 mV through the shared contact rule), S_I = 4kT Re Y holds to
  <= 5.7e-6 over 1 Hz - 1 GHz.
* **Thermionic-emission fluxterm: NOT covered.** It has no noise source: its shot noise
  2q(J_0->1 + J_1->0) is missing. Where the interface resistance matters (A* x 1e-6), the FDT misses
  by -0.69 at 1 Hz. This is pinned by a strict xfail and documented in `noise_analysis` and
  `physics_heterojunction`.
* With the free-electron A*, the same TE device holds the FDT to 5e-6, because the interface is then
  not resistive. This is not evidence of coverage.

**D1a behaviour change (S1-9 fix).** 2-D designs with two or more semiconductor regions of different
doping or affinity now get Fermi-level alignment. No existing test in the files run here pinned the
old zero-built-in-potential result; see section 5.

### 2.9 Consequence of the 2.6 fix: D1d gate L1 now fails (reported, not changed)

`test_devsim_tunneling.py::test_l1_layered_builder_btbt_equals_1d_twin` builds
`BTBTParams(A_SI=4.0e15, B=4e8, P=2.5)`. A = 4.0e15 is an integral float, so before the fix it was
written `"4000000000000000"` and clamped by DEVSIM to 2147483647. The gate therefore compared the 2-D
element-path BTBT with the 1-D edge-path twin at a rate about 1.9e6 times smaller than its docstring
says. In that perturbative regime the two agreed to 3e-9 .. 7e-9.

With the true A, the current at -0.5 V is 2263.76 A/m^2 (1-D) against 2263.12 A/m^2 (2-D), a
relative difference of -2.81e-4. The 1e-6 gate fails. Diagnosis:
* **Not convergence.** The 2-D value is unchanged at rel_tol 1e-5, 1e-8 and 1e-10 (2263.123988991).
* **Not the merge.** Restoring `%.17g` makes the gate pass again with the clamped A.
* **The likely cause is the two quadratures themselves.** At a real generation level, the carriers
  in the junction rise, D varies across each element, and the node-local D of the 2-D element path
  and the edge-mean D of the 1-D path differ at O(h^2) of D's variation.

The tolerance is not loosened and A is not changed here. Two options for the D1d owner or the user:
* gate L1 at a quadrature-consistent tolerance derived from h;
* gate it at the rate its docstring intends with a refined mesh.

## 3. Composition gates: `tests/test_devsim_phase_composition.py` (16 pass + 2 strict xfail)

**Fixture of the first five rows.** A 1-D abrupt p+n+ junction (5e24 m^-3, 0.2 nm at the junction)
with these opt-ins all on:
* BGN (Slotboom);
* Auger (Dziewior-Schmid);
* radiative (Trupke);
* an acceptor `TrapLevel` at E_t - E_i = +0.2 eV (N_t 1e21);
* Si Hurkx BTBT.

It is solved at 0.5 V reverse.

| Gate | Check | Result |
|---|---|---|
| each term exactly once | Every term appears once in the final definitions: `ElectronGeneration` (`USRH`, `RTerm_auger`, `RTerm_radiative`, `RTerm_trap0`), `PotentialNodeCharge` (`QTerm_bulk_traps`) and `ElectronGenerationEdge` (`RTermEdge_btbt`). `USRH` is defined once (never re-defined) and equals `srh_rate_expr(bgn=True)`. The equations assemble exactly these composites. The inventory carries pair rates. BTBT D uses `n_ie@n0`. | pass |
| total = sum of the terms | Each term is re-evaluated in numpy from the SOLVED n, p and n_ie (constants from scipy). The node composite and every term model match to 1e-12 relative. The edge composite matches the BTBT oracle (net generation) to 1e-12. The integrated total equals the sum of the per-term integrals to 1e-12. BTBT carries > 10 % of the total. | pass |
| trapped charge with n_ie | `PotentialNodeCharge` == -q(p - n + N - N_t f_t(n_ie)) | 1e-12, pass |
| trap-aware contacts | `NeutralElectrons` at both contacts == brentq neutrality with Fermi trap occupancy | 1e-11, pass |
| sum / replace / refuse | an extra node term and an extra edge term SUM; re-adding a name REPLACES it; a foreign `edge_volume_model` is refused by `amend`; a term without `pair=` makes the noise raise | pass |
| noise FDT, non-midgap level | the D2a FDT diode, plus a recombination-only level at +0.15 eV, with the built-in channel at -0.1 eV, 1 Hz - 1 GHz | <= 2.2e-6 (gate 1e-4). Re Y x1.49. Dropping the trap's pair breaks the FDT. |
| noise FDT, charged level, quasi-static | the same diode with an acceptor level (N_t 2e21), 1 Hz | 1.4e-6. Re Y x1.57 |
| noise FDT, charged level, 100 kHz / 10 MHz | **strict xfail**: the D1c quasi-static trap charge makes ssac non-charge-conserving | 5.6e-3 / 6.6e-2 (documented D1c limit; turns green when trap kinetics land) |
| noise FDT, BTBT at equilibrium | the D2a FDT diode with BTBT (A 1e12, B 1e7 V/m). The net rate is ~0 (D = 0) but the pair rate and the linear response are not. | <= 2.3e-6 (gate 1e-4). Re Y x1.39. Dropping the BTBT pair breaks the FDT. |
| 2-D element pair weights | uniform field and density on triangles: the element-assembled BTBT + SRH pair reaches node i as pair x NodeVolume_i | 1e-12, pass |
| hetero + BGN: V_bi (x2: `auto` = TE, `continuous`) | 2-D N-Al0.3Ga0.7As / p-GaAs (D1a's gate-B device), BGN on both sides with the test split gamma 0.3 / 0.7 (dEg 14.1 mV). Oracle: q V_bi = W_p - W_n with W_n = chi_A + gamma_A dEg - kT eta_n and W_p = chi_G + E_g - (1 - gamma_G) dEg + kT eta_p (exact FD quadrature). | 2e-4 V gate, pass on both models. BGN moves V_bi by 8.5 mV (> 5 mV asserted). |
| hetero + BGN: flat E_F (x2) | E_Fn / E_Fp from the solved densities against the effective edges, both regions, 2e-4 eV | pass |
| hetero + BGN: effective offsets at the interface (x2) | on the interface node copies, dE_c,eff = dE_c - D_c,A + D_c,G and dE_v,eff = dE_v + D_v,G - D_v,A | 1e-6 eV, pass. **Without the BGN interface terms it misses by 5.66 meV** (the load-bearing gate; the flat-E_F gate moves only to 2.05e-4) |
| noise FDT, continuous-QFL heterointerface | the D2a FDT diode as a heterojunction (n_i 1e19 / 3e18, chi step 0.05 eV, s = 18.9 mV on the circuit contact), 1 Hz - 1 GHz | <= 5.7e-6 (gate 1e-4). Re Y 3.06 vs 6.78 (homojunction). |
| noise FDT, thermionic heterointerface | **strict xfail (LIMIT)**: no noise source for the TE fluxterm | -0.69 at 1 Hz with A* x 1e-6 |

## 4. Defaults-off guards (all against main `5cbd998`, untouched checkout; run on the final tree)

> **2026-11-04: these byte-identity guards were converted on purpose.** The ONE SG flux kernel
> (`dynameta.carriers.sg_flux`, user decision) changed every Scharfetter-Gummel flux string, so the
> defaults-off definition and results can no longer be byte-identical to main. Each guard below now
> pins (a) the re-recorded definition digest and (b) a NUMERIC comparison against the pre-change tree
> at a tolerance derived from the change's measured round-off: the call-log digests ->
> `tests/test_devsim_sg_flux.py` (fixtures solved vs 6595bde); D1d's solution arrays ->
> `validation/sg_flux_cross_version.py` GATE B (vs main); D1c gate F -> numeric (`TOL_F`).
> docs/audit/2026-11-04-sg-flux-two-bernoulli.md section 4. The table records the 2026-11-03 state.

| Guard | Result |
|---|---|
| D1b call-log digest (217 DEVSIM calls, `tests/_bipolar_calllog.py`) | **identical**. It was NOT re-recorded: `test_defaults_off_call_log_is_v013_byte_identical` passes against the digest recorded on main. This covers the circuit contact and the contact-rule refactor. |
| D1d solution arrays | D1d's hashing script was not committed. It was re-implemented over the same fixtures: `validation.bipolar_diode` FD at +0.3 / -0.5 V and the `bipolar_diode_2d` design through `LayeredDevsimBuilder` at +0.3 / -0.4 V. The 16 arrays are Potential / Electrons / Holes / anode current, plus the extras keys. **SHA-256 `a500ca93d99a31b6...` on main AND on the final integration tree**, i.e. after the D1a merge too. |
| D1c cross-version (`validation/traps_oracles.py` gate F, subprocess on main) | diode `f344b6b7...`, layered `792b1278...`, MOS `39553fc4...`: **identical** on main and the branch |
| D1a homojunction limit (gate A in `tests/test_devsim_heterojunction.py`) | pass |

## 5. Validation and regression counts (final tree, `pytest -n 2`)

| Run | Result |
|---|---|
| D1b + D1d + kernels + D1c + D2a + composition (`test_bipolar_recombination_bgn_ionization`, `test_devsim_tunneling`, `test_tunneling_kernels`, `test_traps`, `test_devsim_noise`, `test_devsim_phase_composition`) | **117 passed, 1 failed, 2 xfailed** (strict). The failure is D1d's L1 (section 2.9). Per file: 22 / 17 + L1 / 18 / 25 / 19 / 16 + 2 xfail. |
| `tests/test_devsim_heterojunction.py`, slow gate included | **19 passed** (642 s) |
| Related regression set: devsim_smoke, gated_builder, stacked3d_spec, dc_solve_gummel, audit wave7 carriers / carriers hygiene / conventions / infra, solver_guards, validation_runner, reliability, reliability_ext, density_gradient, geometry_guards, api_surface | **227 passed, 1 failed** (section 5.1) |
| `validation/traps_oracles.py --3d --main <main>` | A-F 9/9 PASS; G1 PASS; **G2 FAIL**, ratio 0.799: the known pre-existing devsim_3d mesh defect, left failing |
| `validation/devsim_noise.py` | **PASS**: 11/11 gate lines, A-G, including G on the 2-D layered builder after the D1a merge |
| `validation/devsim_heterojunction.py` gate F (3-D) | reported, matching D1a's record: TE vs continuous 1.506e-9 / 1.412e-9 (Vg 0 / +0.5 V), DD vs equilibrium 2.758e-3 / 2.093e-3; equilibrium build 554 s, DD builds 3.5 / 3.7 s |

`validation/devsim_heterojunction.py` without `--3d` only re-runs the test file above, so it was run
once, as that file. Gate F was run through its `report_3d()`.

### 5.1 The one regression failure: environmental, not code

`test_audit_2026_07_25_carriers_hygiene.py::test_v20_x21_every_advertised_carriers_name_resolves_in_a_fresh_process`
spawns a fresh Python and imports every name in `dynameta.carriers`. Inside the full `-n 2` run, with
another worker solving DEVSIM and four other agents on box A, that subprocess fails with
`RuntimeError: Issues initializing DEVSIM.`. This is DEVSIM's own math-library initialisation; commit
charge was 79 of 142 GB at the time.

* It passes alone (1.1 s).
* Its whole file passes under `-n 2` (2 passed, 3.7 s).

It was seen twice, both times inside the large parallel run. This is reported as an environment
failure, not rerun to green. No code path touched by this integration changes what that subprocess
imports, except `noise_analysis`, which D2a added to `__all__` and which imports devsim lazily.

### 5.2 D1a behaviour change

No test in the files above pinned the old 2-D zero-built-in-potential result. Nothing was re-pinned.

## 6. Open items carried forward (user decisions)

1. **Trap energy reference under an asymmetric BGN split** (section 2.2): RESOLVED. The user chose
   a per-level `energy_reference` (intrinsic default / conduction / valence); see D1c doc section 9.
   gamma turns out not to enter SRH.
2. **Trap kinetics** (D1c open question 3). The charged-trap FDT xfail pins the consequence for ssac
   and noise.
3. **The devsim_3d mesh is not Delaunay** (D1c open question 4): RESOLVED by D2c (section 7). The
   cause was DEVSIM summing couple-piece magnitudes, not Delaunay-ness; the tensor (Kuhn) mesher is
   exact, and G2 passes (0.989; the rest is z-resolution, G3 = 0).
4. D1b open questions 1-3 and 5, D1c open questions 1 and 3-4, D1d open questions 1-4 and D2a open
   questions 1-3 are unchanged. The D1c contact rule (D1c open question 2) is closed by section 2.4.
5. The 1-D BTBT edge-mean D makes the discrete FDT exact only to O(h^2). Making the 1-D path
   node-local like 2-D/3-D would need a node-model field magnitude, which DEVSIM 1-D does not offer.
   This is recorded, not changed.
6. **D1d gate L1** (section 2.9): RESOLVED by the D1d follow-up (tunneling doc section 8). L1 is
   now a convergence gate at the true A: gap 2.81e-4 (2 nm) / 6.09e-5 (1 nm), order 2.21, gated at
   order >= 1.7 and abs(gap) <= 2 C_inf h^2.
7. **Noise source for thermionic-emission heterointerfaces** (section 2.8): the interface shot noise
   2q(J_0->1 + J_1->0) is not in the source set. The strict xfail pins it.
8. **Heterointerfaces at T_K != T_REF**: RESOLVED by the D1a temperature follow-up (merge `f7c82db`,
   wip/dv_hetero2; heterojunction doc section 4b). The interface now runs at the regions' V_t (one
   source, `BipolarOptions.T_K`): barrier offsets, the BGN-shifted tops (the BGN refusal is gone),
   Richardson A* T^2, the Anderson frame and the band-gap check, with the bipolar N_c/N_v from the
   masses at T_K. A non-isothermal stack is refused ("NON-ISOTHERMAL"), as is an n_i inconsistent with
   band_gap_eV at T_K. Gates T-a..T-d in `tests/test_devsim_heterojunction.py`. (Corrected
   2026-11-06; the item said "refused with BGN, and silently at T_REF without it -- a follow-up".)
9. **D1a open items** (3-D bipolar heterostacks, lateral sidewall interfaces, degenerate TE
   magnitude) are unchanged.

## 7. Late merges: D0 core, D2c 3-D mesh, BLAS-sensitivity fixes (2026-11-03)

| Merge | What | Doc |
|---|---|---|
| `1385d31` wip/dv_core | devsim >= 2.11.2; `carriers.devsim_env` is the one devsim import home (BLAS pinned before the first import; AST gate); `reset_devsim()` in `ac_capacitance`; `DEVSIM_NEW_SYMBOLIC=1` with MKL-PARDISO; Fermi-Jacobian `simplify()` cache | `2026-11-03-devsim-2.11-core.md` |
| `c81cd3f` wip/dv_mesh3d | `Stacked3DSpec.mesher="tensor"` default (Kuhn tets: DEVSIM's magnitude-summed couples are exact, 9e-16); circumcentric 3-D interface trap areas; 3-D element impact ionization; `mesher="gmsh"` warns | `2026-11-03-devsim-3d-mesh.md` |
| `6991e99` wip/dv_blasfix | R26/R27 slab fixture was a floating body (DC Jacobian rank-deficient by one at every G); bipolar DG wall-edge flux by two Bernoulli calls (kahan3 B(v)+v cancelled); DG validation in the extended Newton | recombination doc, DG doc section 1.6 |

Integration fixes made at the merges:
* The D1/D2 modules and validations imported devsim directly; routed through `devsim_env`
  (extended_precision, tunneling, impact_ionization; 6 validations). The one string that is run on
  main too (`traps_oracles._FIXTURE`, gate F) keeps its plain import (`3a56a0d`).
* `validation/devsim_noise.py` integrates through `core.numerics.trapz` (numpy >= 1.24 floor gate).
* `tests/_bipolar_calllog.record_call_log()` records in a CHILD process (`62c4097`). Its circuit-contact
  fixture adds a process-global DEVSIM circuit that the process cannot drop (devsim 2.11.2, MEASURED):
  a second recording died on "cl_n already exists" and left device cl_bz behind (7 unrelated DG /
  Gummel / R26 failures on one xdist worker under `-n 3`); `delete_circuit()` raises an empty error
  and the next circuit solve trips `NodeKeeper.cc:266 CIRCUIT_UNEXPECTED`; and no circuit node may be
  added after an analysis has run with a circuit present. Digests unchanged.

Why two D1/D2 gates flipped with the BLAS pin (both passed on Elmer's OpenBLAS, failed on scipy's):
neither was a BLAS defect; both solves were singular or sub-ulp (R26-a: floating body; DG holes:
wall-edge cancellation plus criteria finer than double). Both now give the same numbers under both
libraries; new gates (`test_slab_is_well_posed`, `test_dg_edge_flux_is_accurate_at_large_drift`)
fail on the old code under both.

**Final tree `3a56a0d`** (devsim 2.11.2, scipy-OpenBLAS pin, Windows):
* tests: every file that touches devsim or carriers (52 files, `-n 3`): **976 passed, 3 xfailed**
  (strict: charged traps ssac at high f, TE heterointerface noise, extended-precision ssac).
* validations: the 33 devsim validations, `run_all --tier full`: **33/33 PASS** (traps_oracles 9/9
  after `3a56a0d`; `impact_ionization_isub` 1463 s is the longest).

Open decisions added by these merges: D0 doc "Open questions" 1-2 and 4; D2c doc section 8; DG doc
section 6 items 5-6.
