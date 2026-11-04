# P3: trap kinetics, thermionic-interface shot noise, and temperature-aware heterointerfaces (v0.15 plan)

Status: PLAN ONLY, for user review. No repository was modified.
Reference tree: `DynaMeta_wt/dv_integ`, branch `feat/devsim-2.11-v0.14` at `6595bde`, devsim 2.11.2.
Author: planning agent, 2026-11-04.

Read before this plan:
* `docs/audit/2026-11-03-devsim-traps.md` (sections 2, 3, 6, 7, 9)
* `docs/audit/2026-11-03-devsim-noise.md`
* `docs/audit/2026-11-03-devsim-heterojunctions.md` (section 4b in particular)
* `docs/audit/2026-11-03-devsim-phase-integration.md` (sections 2.1-2.5, 2.8, 6)

---------------------------------------------------------------------------------------------------

## 0. Before the items: status corrections, ground rules and one measured DEVSIM fact

### 0.1 Item 3's premise is partly out of date (verified against the code)

The brief says heterointerfaces at T_K != T_REF are "refused with BGN, and silently written at T_REF
without it". That was true at `da7614f` (integration note section 2.8 and item 6.8). It was fixed by
the D1a follow-up `f7c82db` (merge of `wip/dv_hetero2`; heterojunction doc section 4b). On `6595bde`:

* `physics_heterojunction.setup_heterointerface_carriers` reads `V_t` from each region
  (`region_V_t`). It refuses two different V_t values (NON-ISOTHERMAL), and uses the region V_t for
  every offset and for `A* T^2` (`interface_models(..., V_t)`, `_bgn_interface_arguments(..., V_t)`).
* `LayeredDevsimBuilder._semiconductor_frames` refuses a mixed-temperature stack. `_own_phi_c`
  and `_hetero_side` use the region kT. `_setup_bipolar_region` builds N_c and N_v from the masses at
  `BipolarOptions.T_K`. The band-gap check runs at T.
* Gates T-a to T-d exist and passed on that merge (heterojunction doc 4b table).

What is still open (found while reading for this plan; section 3.2 lists the evidence):

1. **The affinity is never temperature-dependent.** `chi_eV` is used as given at any T. The band-gap
   check forces `band_gap_eV` to be the gap AT T for a bipolar heterostack. Nothing checks or
   converts chi, so a 300 K affinity silently sets the 250 K Anderson offset.
2. **The 3-D bipolar body runs with a T_REF N_c / N_v under `T_K`.**
   * `devsim_3d.py` lines 816-817 build the region with `effective_dos_m3(s.dos_mass_kg)`, with no
     T.
   * Lines 894-896 do the same for the gate frame phi_bi.
   * Line 989 reports `temperature_K=PE.T_REF`.

   This is not an interface path, but it is the same silent-T_REF defect class.
3. **The default interface-trap bin width is the 300 K kT** (`physics_traps.py` line 425:
   `dmax = ... else V_T`). At T_K = 77 K the bins are 3.9 kT wide, against the documented "<= kT".
4. **BTBT runs at a fixed T_REF** (`tunneling.py` line 41 and line 271). This belongs to D1d and is
   only listed here.

Item 3 below is therefore re-scoped:
* (a) close the remaining silent paths 1-3;
* (b) add an opt-in temperature model for the band data (Varshni E_g(T) and a chi(T) convention), so
  that a user can give 0 K / 300 K material data and simulate at T.

### 0.2 Ground rules applied to every item

* **One kernel per physics quantity.**
  * Every SRH-type rate, occupancy, pair rate and the new capture/emission rates come from
    `physics_traps` (`srh_reference_densities` stays the only place where the energy reference, the
    BGN substitution and the DG scale enter).
  * Every recombination or charge term enters through `physics_bipolar_dd.add_recombination_term` /
    `add_charge_term`. These are extended, not paralleled.
  * Every noise source is assembled in `noise_analysis`, from an inventory read back from DEVSIM.
  * Every heterointerface quantity comes from `physics_heterojunction`.
* **Defaults off.** No new keyword changes a model string, an equation, a parameter or a solved array
  unless it is set. The exceptions are places where the current output is silently wrong; each one is
  a user decision (section 5).
* **Defaults-off guards while the two-Bernoulli SG change is in flight.** Another agent is replacing
  every SG flux. Results move at round-off, and the stored byte digests are being replaced by
  round-off agreement checks. So these guards do not record new digests. Each workstream uses:
  * (i) a SAME-TREE A/B test: the keyword absent against the keyword at its default, in one process.
    The model inventory, the equation commands, every model string and every solution array must be
    bit-identical. This is meaningful whatever the flux is.
  * (ii) a CROSS-VERSION check against the base commit at the agreement tolerance the SG agent
    adopts. The placeholder is <= 1e-12 relative on Potential, Electrons, Holes and terminal current.
    It is not recorded until that agent's change has merged.
  * (iii) the existing call-log digest gate (`test_defaults_off_call_log_is_v013_byte_identical`).
    It counts DEVSIM calls, not floats, so it stays valid.
* **CI budget.** The `test` job is capped at 150 min, and the coverage leg already measures 65-95 min.
  The plan therefore gives each workstream <= 120 s of new non-slow tests, single process, on the box
  A reference. Everything else is `@pytest.mark.slow` (nightly) or a `validation/*.py` script in the
  nightly full tier.
* **Gate discipline.**
  * Every gate states its oracle and its tolerance, with the tolerance derived from the problem.
  * Every gate is shown failing on the base commit before it is accepted.
  * Every passing gate gets one adversarial perturbation that must make it fail (the
    "load-bearing" check of the existing docs).
  * No gate is relaxed to green, and no gate is re-run to green.

### 0.3 A DEVSIM fact this plan relies on (MEASURED for this plan, devsim 2.11.2, box A)

The scratch script `spike_node_eq.py` (in the scratchpad, not the repo) builds a 1-D region with a
Laplace Potential equation. It adds an extra node-only variable `U` with
`equation(node_model=(U - Potential)/tau, time_node_model=U)`. It defines NO contact equation for
`UEq`, puts a grounded contact on one end and a circuit-driven contact (V1, 0.5 V) on the other.
Results:

* DC: `U` equals `Potential` at both contact nodes (0.0 / 0.5). A node-only equation with no
  contact equation KEEPS its bulk row at contact nodes (24 equations = 11 nodes x 2 + 2 circuit).
* `solve(type="ac")` runs, creating `U_Real` / `U_Imag`, so `time_node_model` enters ssac.
* `solve(type="noise", output_node="V1.I")` creates `V1.I_UEq_real/_imag`. The native adjoint field
  therefore exists for any extra equation.

This is what makes the trap-occupancy variable in item 1 cheap to integrate:
* no contact rule is needed for it;
* ssac, transient and the native noise solve pick it up with no special casing.

---------------------------------------------------------------------------------------------------

## 1. ITEM 1: transient trap kinetics (finite capture and emission)

### 1.1 Goal and physics

**Today.** Trapped charge and SRH occupancy are instantaneous functions of (n, p). This is
`srh_occupancy_expr`, the steady-state Shockley-Read occupancy. DC and LF C-V are exact. Transient
and ssac with CHARGED levels are not charge-conserving: the capture current that fills a trap is
missing from the continuity time terms. The measured consequence is the strict xfail
`test_noise_fdt_with_a_charged_trap_level_high_frequency`, where S/4kT Re Y - 1 = 5.6e-3 at 100 kHz
and 6.6e-2 at 10 MHz.

**Goal.**
* A per-level occupancy SOLUTION VARIABLE with its own rate equation, for bulk levels and for
  interface (D_it) bins.
* A DC limit identical to today's model (to solver round-off).
* Charge-conserving transient and ssac.
* The conductance-method C(omega), G(omega) of interface states.
* Trap generation-recombination (g-r) noise as Lorentzians coming out of the impedance-field
  machinery, with the FDT exact at every frequency.

**Rate equations.** Shockley & Read, Phys. Rev. 87, 835 (1952); Hall, Phys. Rev. 87, 387 (1952). The
content is standard. **The equation numbers in the 1952 papers were not verified (paywalled)**, so the
plan cites the papers but no equation numbers. Notation:
* f = electron occupancy of a level of density N_t;
* c_n = sigma_n v_th,n, c_p = sigma_p v_th,p;
* tau_n0 = 1/(c_n N_t), tau_p0 = 1/(c_p N_t).

Emission follows from detailed balance: e_n = c_n n1', e_p = c_p p1'. Here n1' and p1' are EXACTLY
the densities the shipped kernel already produces (`srh_reference_densities`: energy reference, then
BGN, then DG scale). The DLTS emission rate e_n = sigma_n v_th N_c exp(-(E_c - E_t)/kT) [D. V. Lang,
J. Appl. Phys. 45, 3023 (1974)] is this expression with `energy_reference="conduction"`.

Per unit volume:

    R_n = N_t c_n [ n (1 - f) - n1' f ]  = [ n (1 - f) - n1' f ] / tau_n0     (net electron capture)
    R_p = N_t c_p [ p f - p1' (1 - f) ]  = [ p f - p1' (1 - f) ] / tau_p0     (net hole capture)
    N_t df/dt = R_n - R_p
    dn/dt = (1/q) div J_n - U_other - R_n
    dp/dt = -(1/q) div J_p - U_other - R_p
    Poisson: rho += -q N_t f  (acceptor-like) | +q N_t (1 - f) (donor-like) | 0 (neutral, kinetic only)

Two consequences, both checked algebraically in step K1-1:
* **Steady state.** R_n = R_p gives f = (tau_p0 n + tau_n0 p1')/(tau_p0 (n + n1') + tau_n0 (p + p1')),
  the shipped `srh_occupancy_expr`, and R_n = R_p = U_SRH, the shipped `srh_rate_expr`. So the DC
  limit IS the current model.
* **Charge conservation.**
  d/dt[q(p - n) - q N_t f] = -div(J_n + J_p) + q(R_n - R_p) - q(R_n - R_p) = -div J. The quasi-static
  model breaks exactly this identity.

**Interface traps (sheet levels, one per D_it bin).** These are the same equations with N_t replaced by
N_it,k [m^-2]. The bulk n and p are replaced by the surface n_s and p_s, and tau by 1/S with
S = sigma v_th N_it. The sheet terms enter at interface nodes with the shipped weight
`ItArea = A_node/NodeVolume`.

**Conductance method.** E. H. Nicollian and A. Goetzberger, "The Si-SiO2 interface: electrical
properties as determined by the metal-insulator-silicon conductance technique", Bell Syst. Tech. J.
46, 1055 (1967). **Equations VERIFIED from the scanned issue** (bitsavers BSTJ V46N06, pp. 1068-1075;
local copy in the scratchpad `refs/bstj46_6.pdf`). The paper's notation is N_s (single level, cm^-2),
N_ss (continuum, cm^-2 eV^-1), and n_s0, the surface electron density for n-type with majority
electrons.

* (1) R_n(t) = N_s c_n [1 - f(t)] n_s(t);  (2) G_n(t) = N_s e_n f(t);  (3) i_s = q N_s c_n (1 - f) n_s - q N_s e_n f.
* (4) Y_s = j omega (q^2/kT) N_s f0 (1 - f0) / (1 + j omega f0/(c_n n_s0)), so C_s = q^2 N_s f0(1-f0)/kT
  and tau = f0/(c_n n_s0).
* (5) C = C_s/(1 + omega^2 tau^2);  (6) G_p = C_s omega^2 tau/(1 + omega^2 tau^2).
* (12) i_T = (j omega C_D + Y_s) delta psi_s;  (15) delta v_a = i_T (Z_s + 1/(j omega C_ox)), Z_s = (j omega C_D + Y_s)^-1.
  This is the Fig. 6 circuit: C_ox in series with C_D || (C_s + R_s).
* (16) C_p = C_D + C_s/(1 + omega^2 tau^2);  (17) G_p/omega = C_s omega tau/(1 + omega^2 tau^2).
  The peak is at omega tau = 1, with value C_s/2.
* (18) continuum: Y_ss = j omega (q^2/kT) Int N_ss f0 (1 - f0) dpsi / (1 + j omega f0/(c_n n_s0)).
* (19a) Y_ss = (q N_ss/(2 tau_m)) ln(1 + omega^2 tau_m^2) + j (q N_ss/tau_m) arctan(omega tau_m);
  (19b) tau_m = 1/(c_n n_s0).
* (20) G_p/omega = (q N_ss/(2 omega tau_m)) ln(1 + omega^2 tau_m^2). Its peak sits at
  omega tau_m = 1.98, the value marked in Fig. 13. The peak value 0.402 q N_ss is my own arithmetic:
  ln(1 + 1.98^2)/(2 x 1.98).
* (22) tau_m = exp(-(u_s - u_B))/(c_n n_i) for n-type; tau_m = exp(u_s - u_B)/(c_p n_i) for p-type.

Notes on scope:
* N&G derive (4) for majority electrons. The hole (p-type) mirror used below,
  tau = (1 - f0)/(c_p p_s0), is **derived here by the same linearization, not quoted**.
* The statistical surface-potential-fluctuation model (their eqs 26-39) is a 2-D statistical effect
  and is OUT of scope. A laterally uniform DD model cannot produce it.
* E. H. Nicollian and J. R. Brews, *MOS Physics and Technology* (Wiley 1982), ch. 5, is the textbook
  form. **The book was not accessible in this session**, so no equation number from it is cited.

**Trap g-r noise.**
* Each elementary process is a two-way Poisson process with forward and backward event rates. The
  processes are electron exchange (capture + emission) and hole exchange, per level and per node.
* The Langevin source of a process with event rates a and b is 2(a + b). It enters every equation
  touched by the process with that equation's stoichiometric sign.
* Sources for the full set of elementary processes, mutually uncorrelated, are the standard
  master-equation construction. K. M. van Vliet and J. R. Fassett, in *Fluctuation Phenomena in
  Solids* (Burgess ed., 1965), is already cited in the noise doc; Bonani & Ghione 2001, ch. 3.
  **Neither was accessed in this session; no equation numbers are cited.**
* The FDT gate (K3) is the independent check that the construction is right in this discretization.

Per node i and level k, the event-rate sums are

    P_n = [ n (1 - f) + n1' f ] / tau_n0         P_p = [ p f + p1' (1 - f) ] / tau_p0

The shipped fast-trap source 2(G + R) = 2(n p + n_i'^2)/D is what these reduce to only when f follows
(n, p) instantly AND the level carries no charge. That is exactly why the quasi-static charged level
breaks the FDT at high frequency.

**McWhorter 1/f (number fluctuation through oxide border traps).** A. L. McWhorter, in *Semiconductor
Surface Physics* (Kingston ed., 1957); S. Christensson, I. Lundstrom, C. Svensson, Solid-State
Electron. 11, 797 (1968); K. K. Hung, P. K. Ko, C. Hu, Y. C. Cheng, IEEE TED 37, 654 (1990).

* The mechanism: traps at depth z in the oxide exchange with the channel through a tunnelling factor
  exp(-z/lambda). A uniform distribution in z gives a 1/tau distribution of time constants, and so a
  1/f spectrum.
* **UNVERIFIED in this session.** I reached only a TU Wien thesis page (Goes, section 6.1, eqs.
  6.1-6.2) whose equations extracted garbled. It shows the factor exp(x_t/x_0) on the capture and
  emission times. No primary formula is quoted here.
* Scope is a user decision (D-K5).

### 1.2 Current code state and what is reused

| Need | Reused as is | Extended |
|---|---|---|
| n1', p1', n_i'^2 with the energy reference, BGN and DG scale | `physics_traps.srh_reference_densities`, `level_reference_densities`, `n1_p1` | -- |
| steady-state rate, occupancy, pair rate | `srh_rate_expr`, `srh_occupancy_expr` (DC seeding and the K1 identity gate) | -- |
| capture/emission rates and per-process pair rates | -- | NEW `physics_traps.srh_capture_exprs(n1, p1, taun, taup, f, one_minus_f, *, bgn, reference, scale, pair=False)`, built ONLY from `srh_reference_densities` (one conversion) |
| term mechanism | `add_recombination_term` / `add_charge_term` / `recombination_terms` | carrier-specific node terms (`RTermN_`, `RTermP_`) and derivatives with respect to extra solution variables (section 1.3.2) |
| equation records | `eq_registry.record_region_equation` (Gummel freeze/thaw sees it) | -- |
| interface weights | `physics_traps.interface_node_area`, `ItArea<j>` | -- |
| sheet levels | `InterfaceTraps.sheet_levels()` | uses the region kT for the default bin (item 3, path 3) |
| contacts | `contact_potential_residual`, `equilibrium_fields(traps=)`: the equilibrium occupancy is unchanged | none: the trap row stays bulk at contact nodes (section 0.3) |
| ssac, transient, noise solves | `ac_analysis`, `transient`, `noise_analysis` (native adjoint) | noise: the kinetic process sources (K3) |
| extended precision | `extended_precision.enable` / `solve` | none: equations created after `enable` are 128-bit automatically |

### 1.3 Design

#### 1.3.1 API (all defaults OFF)

```python
TrapLevel(..., dynamic: bool = False)        # needs density_m3 > 0 when True (N_t defines the variable)
InterfaceTraps(..., dynamic: bool = False)   # every bin of the spec becomes kinetic
```

* `dynamic=False` is today's model, character for character.
* A kinetic `kind="neutral"` level is allowed: it carries no Poisson charge but delays recombination.
* The builders thread the flags unchanged: `TransportModel(srh_traps=...)`,
  `LayeredDevsimBuilder(interface_traps=...)` and `Stacked3DSpec(srh_traps=..., interface_traps=...)`
  already carry the specs, so no builder keyword is added.
* New diagnostics:
  * `TrapOccupancy<k>` becomes f(u) for a kinetic level;
  * `TrapOccupancySS<k>` is the steady-state expression, for comparison;
  * `ItOccupancy<j>_<k>` per kinetic interface bin.

#### 1.3.2 Term-mechanism extension (the one structural change; step K0)

A kinetic level puts DIFFERENT rates into the electron and hole equations (R_n into one, R_p into the
other). The shipped composite assumes one shared rate: `ElectronGeneration = -q(USRH + sum RTerm)`,
with `+q` for holes. The extension below keeps that one mechanism:

* `add_recombination_term(..., carrier="both" | "electron" | "hole", wrt=None)`:
  * `"electron"` stores `RTermN_<name>`, which enters ElectronGeneration only.
  * `"hole"` stores `RTermP_<name>`, which enters HoleGeneration only.
  * `wrt` lists extra solution variables (for example `TrapU_trap0`) whose derivative models the term
    carries: `RTermN_<name>:TrapU_trap0`. The caller gives them as closed forms in `derivs`.
* The composites become `ElectronGeneration = -q (USRH + sum RTerm_ + sum RTermN_)` and
  `HoleGeneration = +q (USRH + sum RTerm_ + sum RTermP_)`. With no N/P terms the builders return
  today's strings. The pins in `test_traps.py` / `test_devsim_phase_composition.py` cover this.
* **Composite derivatives.** Each composite is differentiated with respect to Electrons, Holes and
  every variable for which some term has a derivative model. That set is read back from DEVSIM's own
  model list (names `<term>:<var>`), so there is no module state, as in the shipped design. DEVSIM's
  `simplify(diff())` of a composite that refers to term models by name only needs the per-term
  derivative models. **To check in K0:** this chain rule through named node models must work for a
  non-standard variable. The 45-bin cost lesson of D1c applies: the closed forms are supplied for
  every kinetic term.
* `add_charge_term(..., wrt=None)`: the same for `QTerm_*`. `rebuild_poisson_charge` differentiates
  with respect to the union of variables.
* `recombination_terms()` gains the keys `node_electron` / `node_hole`, each entry being
  `(rate, pair | None, partner_equation)`. The partner equation is derived from the naming convention
  (term `trap<k>`, variable `TrapU_trap<k>`, equation `TrapEquation_trap<k>`), read from DEVSIM. An
  N/P term without a pair rate, or whose partner equation is missing, makes the noise refuse the region
  (the shipped "never silently noiseless" rule).

#### 1.3.3 The occupancy variable: a logit (trap quasi-Fermi level)

`TrapU_<name>` is u = ln(f/(1 - f)) = (E_Ft - E_t)/kT, the level's own quasi-Fermi level in kT, with
f = 1/(1 + exp(-u)) and 1 - f = 1/(1 + exp(u)). The reasons:

* f stays in (0, 1) under any Newton update. With `variable_update="default"` a variable f can
  overshoot, and DEVSIM has no bounded update.
* There is no cancellation in 1 - f for a nearly full level. A deep level has f ~ 1e-20 or
  1 - f ~ 1e-20 routinely.
* The seed is exact: u0 = logit(`TrapOccupancySS<k>`) at the seed state.
* The equilibrium value is the Fermi level (u = (E_F - E_t)/kT), a direct physical check.

**Limit.** DEVSIM's exp overflows past about 709, so |u| must stay below about 700. At room
temperature |u| < E_g/kT ~ 45. Below about 20 K a deep level can exceed the limit. An installer check
raises a named error when |E_t - E_F|/kT at the seed is above 600.

The alternative, variable f with damping, is the D-K2 decision.

#### 1.3.4 The trap equation (bulk level k)

DEVSIM's residual convention, read from the continuity equations, is
d(time_node_model)/dt + node_model + flux divergence = 0, with NCharge = -q n and
ElectronGeneration = -q U. Accordingly:

    TrapCharge_<name> = -q N_t f(u)                 (time_node_model; d/du = -q N_t f (1 - f))
    TrapNode_<name>   = +q (R_n - R_p)              (node_model; closed-form d/dn, d/dp, d/du)
    RTermN_<name>     = R_n                         (electron row, through ElectronGeneration = -q(...))
    RTermP_<name>     = R_p                         (hole row)
    QTerm_bulk_traps  : -N_t f(u) | N_t (1 - f(u))  (charged kinetic levels; quasi-static levels keep
                                                     their occupancy expression in the same term)

* The equation is `TrapEquation_<name>` with variable `TrapU_<name>`, `variable_update="default"`, and
  no edge model. It is recorded through `eq_registry`.
* Its row scale is q x rate x volume, the same as the continuity rows. Its Newton error is therefore
  commensurate with theirs.

**Interface bins.** Bin k of interface install j uses the region variable `ItU<j>_<k>`. Its node model
is `ItArea<j> * q (R_n,k - R_p,k) + ItMask0<j> * u`, and its time model is
`ItArea<j> * (-q N_it,k f)`. `ItMask0<j>` is 1 off the interface and 0 on it. Off the interface the
row is NodeVolume x u = 0: decoupled, diagonal, and inert in the noise because no source weight sits
there. The interface traps are a region-wide variable because **DEVSIM 2.11 has no interface solution
variable**: `interface_equation` only couples two region rows, and `node_solution` is region-scoped.

The cost is (bins) x (semiconductor-region nodes) extra unknowns:
* about 45 x 300 = 13.5k for a 1-D MOS, which is fine;
* about 3.4 million for the 76k-node 2-D metasurface, which is not.

The design therefore:
* refuses, with the count, above a budget of about 2e5 extra unknowns (the default is to be
  calibrated in K0);
* offers `InterfaceTraps(kinetic_bin_eV=...)` as a coarser bin width for the kinetic variables only;
* records the alternative, interface-only unknowns as circuit nodes assembled by
  `devsim.custom_equation`, as a research spike (D-K3). It is not built here: it is Python-level
  assembly every Newton step, and its ssac, noise and extended-precision behaviour are unverified.

#### 1.3.5 Composition

* **BGN, energy reference, DG scale.** These come in through `srh_capture_exprs`, which only calls
  `srh_reference_densities`, so every composition the shipped kernel has is inherited. The K1-1
  identity gate runs on the full 3 references x BGN on/off x DG scale on/off matrix.
* **Auger, radiative, BTBT, external G.** Untouched; they are separate terms.
* **Contacts.** The neutral root (`equilibrium_fields(traps=...)`) uses the equilibrium occupancy,
  which a kinetic level relaxes to at DC, so there is no change. The trap row is bulk at contact nodes
  (section 0.3). At a pinned contact n and p are fixed, so u is fixed at DC and in ssac.
* **Gummel and staged solves.** `TrapEquation_*` / `ItEquation*` are frozen together with the carrier
  equations (`dc_solve._gummel` uses CARRIER_EQS). The rule is derived from `eq_registry` names
  (prefix list), not hard-coded.
* **1-D, 2-D, 3-D.** The bulk is unchanged in every dimension. The interface uses the weights already
  shipped (half-edge in 2-D, circumcentric in 3-D on the tensor mesher).
* **Extended precision.** Automatic. The ssac/noise round-off floor stays double (documented).

### 1.4 Ordered implementation steps

* **K0 Mechanism and spikes (no physics yet).**
  1. Implement `carrier=` and `wrt=` in `add_recombination_term` / `add_charge_term`, the composite
     derivative union, and the inventory keys.
  2. Same-tree A/B guard: every existing test of the term mechanism passes, and the composite
     strings are unchanged with no N/P terms.
  3. Spike: a toy N/P term pair plus an extra variable, to confirm that DEVSIM's chain rule through
     named models handles the extra variable in DC, ssac and noise. Measure Jacobian build time for
     45 bins.
  4. Calibrate the interface unknown budget on a 2-D layered device (build time, memory, solve time
     per Newton step at 1e4 / 1e5 / 3e5 extra unknowns).
* **K1 Bulk kinetic levels: DC and transient.**
  1. `srh_capture_exprs` plus its symbolic and numeric identity with the shipped steady state.
  2. `install_bulk_traps` handles `dynamic=True`: variable, equation, N/P terms, closed-form
     Jacobians, and the charge term in f(u).
  3. Seeding from `TrapOccupancySS`.
  4. Gummel group membership.
  5. Builders: pass-through, plus the guards (a kinetic level needs N_t > 0 and a bipolar region;
     the |u| range check).
* **K2 Interface kinetic bins and the conductance method.** `install_interface_traps` handles
  `dynamic=True`: per-bin variables, the mask, the budget refusal, `kinetic_bin_eV`, closed-form
  derivatives (the D1c closed forms extended by d/du), and the C-V helpers in validation.
* **K3 Noise sources for kinetic processes.** In `noise_analysis._RegionData`:
  * read `node_electron` / `node_hole` with their partners;
  * per node, add S = q^2 w_i [2 P_n |y_t - y_n|^2 + 2 P_p |y_p - y_t|^2], with w_i = NodeVolume
    (bulk) or ItArea x NodeVolume = A_node (sheet).
  The signs follow from the residual convention in section 1.3.4. They are not assumed: the FDT gate
  pins them, because a wrong sign misses by O(1). Mechanism keys "trap_n" / "trap_p" are added to
  `MECHANISMS` (D-K4). The fast-trap pair `PTerm_trap<k>` is NOT installed for a kinetic level, so it
  is never counted twice.
* **K4 Docs and flips.** Audit note `docs/audit/2026-11-xx-devsim-trap-kinetics.md`, docstrings, the
  xfail flip (D-K1), and release lines.
* **K5 (optional, D-K5) Border traps / McWhorter.** `InterfaceTraps(border=BorderTraps(depth_m, n_depth,
  lambda_m))` scales each bin's c_n and c_p by exp(-z_j/lambda) over depth bins z_j. That gives
  (energy bins) x (depth bins) kinetic sheet variables at the interface nodes. No new mechanism is
  needed: it is K2 with more bins.

### 1.5 Verification gates

All numbers are targets to be measured. Each tolerance is derived; if a measured value misses its
derived tolerance, that is a finding and the tolerance does not move.

**K1 (bulk).**

| # | Gate | Oracle | Tolerance and derivation | Fails on the old code because |
|---|---|---|---|---|
| K1-1 | Kernel identity | Substitute f = `srh_occupancy_expr` into R_n and R_p in numpy (constants from scipy). Over a grid of n, p, E_t, the 3 references, BGN on/off, DG scale on/off: R_n = R_p = `srh_rate_expr` and P_n, P_p > 0. At equilibrium (n p = n_i'^2), u = (E_F - E_t)/kT. | 1e-13 relative. Pure double arithmetic on O(1) ratios: a few ulp times about 10 operations. | `srh_capture_exprs` does not exist (ImportError, asserted with `pytest.fail`, not skip). |
| K1-2 | DC identity | The D1c gates (a) tau(E_t) and (b) compensation, the composition fixture (BGN + Auger + radiative + acceptor level + BTBT at 0.5 V reverse) and the 1-D MOS sweep, each solved twice: `dynamic=False` and `dynamic=True`. Compare Potential, n, p, the terminal current and the occupancy. | <= 1e-10 relative on n and p, <= 1e-12 V on psi. Both are Newton-converged to rel 1e-12 on the same discrete equations; the extra unknown changes only the iteration path. Condition-number amplification is about 1e2 (measure it with a perturbed seed). | The keyword does not exist (TypeError). |
| K1-3 | D1c oracles re-run with kinetics | tau(E_t) against the closed form (D1c measured 2.1e-5); compensation against brentq (1e-15). | The SAME tolerances as the D1c gates: the physics is identical at DC. | n/a (a regression guard) |
| K1-4 | Transient emission (DLTS-like) | p+n diode, n side N_D = 1e21 m^-3, acceptor level at E_c - 0.3 eV (conduction reference), N_t = 1e19. Fill at 0 V, step to -5 V, integrate. Oracle at a node deep in depletion (n/n1 < 1e-6, from the DC solution): f(t) = f_inf + (f0 - f_inf) exp(-(e_n + e_p + c_n n + c_p p) t), with e_n = c_n n1 (Lang 1974 form). | 1e-4 relative in f(t) - f_inf for t <= 5 tau. Derivation: BDF2 local error ~ (dt/tau)^2 / 3 per step at dt = tau/100, about 3e-5 accumulated; plus the c_n n term (< 1e-6). Convergence sub-gate: halving dt divides the error by 4 +- 20 %. | Quasi-static: f follows n instantly, so the trap is empty at t = 0+. The error is O(1). |
| K1-5 | Charge conservation in transient | 1-D, both contacts circuit-driven (one source held at 0 V), same diode with a charged kinetic level, bias step. Check |I_lo(t) + I_hi(t)| / max|I| at every step, and Int I_hi dt = Q_hi(DC final) - Q_hi(DC initial) from two independent DC solves. | Kirchhoff: 1e-9 relative, which is Newton residual times the condition number of the circuit row (measure the floor on the trap-free twin first and gate at 10x that floor). Charge: 1e-4, the BDF quadrature error at the chosen dt, with the same halving-convergence sub-gate. | Quasi-static charged level: measure the miss. Expected O(N_t/N_D), and the gate must fail by > 10x its tolerance (load-bearing). |
| K1-6 | ssac charge conservation | Same 1-D device: Y_lo + Y_hi = 0 at 1 Hz - 1 GHz (both contacts circuit-driven). | 1e-10 relative to |Y|, the ssac round-off floor (D2b documents its floor for Re Y; measure on the trap-free twin and gate at 10x). | Quasi-static charged level: nonzero at high f; measure it. |
| K1-7 | Defaults off | Same-tree A/B on the D1c fixtures, `dynamic` absent against `dynamic=False`: model inventory, equation commands and arrays bit-identical. Cross-version at the SG agreement tolerance (section 0.2). The call-log digest gate passes unchanged. | Exact / round-off. | -- |

**K2 (interface: conductance method).** Device: 1-D p-Si (N_A = 1e17 cm^-3) / 10 nm SiO2 MOS. This is
the D1c (c) device; reuse `_mos`. The gate is circuit-driven (`ac_analysis.setup_circuit_contact`) and
the body is a grounded ohmic contact. SRH tau = 1e3 s suppresses bulk g-r. The bias is in depletion:
psi_s from 0.2 to 0.6 V. Every oracle quantity is computed from the DC solution (p_s0, n_s0, f0 from
the solved surface densities and the band-referenced n1, p1) plus the exact oxide capacitance
C_ox = eps_ox/t_ox. The 1-D FV Laplace in a uniform oxide is exact. Nothing comes from the ssac under
test.

| # | Gate | Oracle | Tolerance and derivation | Fails on the old code because |
|---|---|---|---|---|
| K2-1 | Single discrete level, G_p/omega(omega) | Extract Y_s = (1/Y_m - 1/(j omega C_ox))^-1 (N&G eq 15 inverted). Then G_p/omega = Re Y_s/omega, compared with N&G (17) mirrored for holes: C_s = q^2 N_it f0(1 - f0)/kT, tau = (1 - f0)/(c_p p_s0). Sweep omega tau from 1e-2 to 1e2 and check the peak position omega tau = 1 and the peak value C_s/2. | 1e-3 relative at each point with omega tau in [0.1, 10]. Error sources: (i) minority exchange c_n n_s/(c_p p_s) < 1e-6 by the bias choice; (ii) neutral-body series resistance R_b, giving (omega C_tot)^2 R_b/G_p (computed from the body's conductivity and length, made < 1e-4 by a short, doped body); (iii) the majority dielectric relaxation time, omega tau_d < 1e-6 below 1 MHz. Gate at 10x the computed sum (state it); measure. | Quasi-static occupancy: Y_s is purely capacitive at every f, so G_p = 0 to the ssac floor. The peak is missed by 100 %. |
| K2-2 | Single level, C_p dispersion | N&G (16): C_p - C_D = C_s/(1 + omega^2 tau^2), with C_D = Im Y_s/omega in the omega tau >> 1 limit. A cross-check: the same value as from a trap-free twin at equal psi_s (bias found by brentq on the DC psi_s). | 1e-3, as K2-1. | Quasi-static: C_p is frequency-flat (the LF value). |
| K2-3 | Continuum (uniform D_it, 1e12 cm^-2 eV^-1) | (a) G_p/omega against the binned sum of eq (17) over the simulated bins (exact for the discretized model); (b) against the continuum integral of eq (18) by scipy quad; (c) against the closed form (20) where its assumptions hold (E_F more than 10 kT from the band ends, c_p independent of energy). | (a) 1e-3, as K2-1. (b) bin convergence: the error at bin = kT, kT/2, kT/4 must fall at least as fast as h^2 (midpoint); report the kT value. (c) <= 1e-3, the size of the neglected band-edge tails (exp(-10)). | Quasi-static: G_p = 0. |
| K2-4 | LF limit equals the shipped LF C-V | C_p(omega -> 0) - C_D = C_it of the quasi-static stretch-out gate (D1c (c-i')) at the same psi_s. | 2.3e-4, D1c's measured mesh-limited slope agreement. | Regression guard. |
| K2-5 | Small-step transient = the circuit | A 1 mV gate step. Fig. 6 closed form: the R_s-C_s branch relaxes with tau* = R_s C_s C_th/(C_s + C_th), where C_th = C_ox + C_D and R_s = tau/C_s. For t > 0, i_g(t) = [C_ox/(C_ox + C_D)] (Q_s,inf/tau*) exp(-t/tau*), with Q_s,inf = V_th C_s C_th/(C_s + C_th) and V_th = dV C_ox/(C_ox + C_D). (Derived here from N&G Fig. 6.) | 2e-3 relative over [tau*/10, 5 tau*]: 1 mV keeps the nonlinearity at dV/2V_t ~ 2e-2, and the second-order term is (dV/V_t)^2/8 ~ 2e-4 times the curvature ratio; plus BDF as in K1-4. Halving dt must halve the residual quadratically. | Quasi-static: no exp(-t/tau*) tail (O(1) miss). |
| K2-6 | 2-D and 3-D wiring | A laterally uniform 2-D layered MOS (3 columns) and a 3-D tensor-mesh cap (`Stacked3DSpec`) give the same G_p/omega as 1-D at 3 frequencies. | 1e-6 relative: on the tensor mesh the per-node areas are exact (D2c gate G2 0.989 came from z-resolution, and G3 = 0). Reuse the same z-mesh. | -- |

**K3 (noise).**

| # | Gate | Oracle | Tolerance and derivation | Fails on the old code because |
|---|---|---|---|---|
| K3-1 | The xfail flips | The `gate_b_fdt` diode with `CHARGED_TRAP` kinetic (N_t = 2e21, srh_trap_energy_eV = -0.1). S_I = 4kT Re Y from 1 Hz to 1 GHz. | 1e-4, the D2a FDT gate tolerance (measured 2-6e-6, ssac-limited). | The quasi-static strict xfail: 5.6e-3 at 100 kHz and 6.6e-2 at 10 MHz. |
| K3-2 | Load-bearing checks | (a) drop the hole-process source; (b) flip the trap-row sign; (c) also install the fast-trap PTerm. | Each must break K3-1 by > 1e-2. | -- |
| K3-3 | Neutral kinetic level equals the fast-trap source | Neutral level, N_t -> small (1e15), fast rates: S(f) equals the shipped fast-trap GR spectrum (gate D1's ambipolar Lorentzian device). | O(N_t/n0) relative: compute the bound and gate at 10x. | n/a (consistency) |
| K3-4 | Single-level g-r Lorentzian (trapping noise) | Long n-type resistor (400 L_D, like gate D2), acceptor kinetic level near E_F, majority exchange only, 0.05 V. The test derives the oracle itself: neutrality gives dn = -N_t df, so 1/tau = c_n (n0 + n1 + N_t(1 - f0)); var(dN)/V = [1/n0 + 1/(N_t f0(1 - f0))]^-1; S_I/I^2 = 4 var tau/(N^2 (1 + omega^2 tau^2)). **The variance formula is derived here** (Burgess g-r theorem with the number constraint); the primary source (van Vliet & Fassett) was not accessed. | Plateau and corner S(f_c)/S(0) = 1/2 to 1.5e-2. This is the D2a D2 analogue. Contact shorting over about 2 sqrt(D tau)/L: choose L so that this is < 3e-3 and gate at 1.5e-2 as D2. | Quasi-static: the corner is at the fast-trap (ambipolar) time, not tau_trap. Measure. |
| K3-5 | MOS FDT with kinetic D_it | The K2 MOS: S_I(gate) = 4kT Re Y_m from 10 Hz to 10 MHz. | 1e-4. | Quasi-static: S has no interface contribution at high f while Re Y_m does (or the reverse); measure. |
| K3-6 | Native adjoint on the trap rows | The native `<out>_TrapEquation_*_real/imag` against an equilibrated scipy adjoint of `get_matrix_and_rhs`. | 1e-12, as the D2a adjoint gate. | -- |

**K5 (optional McWhorter).**
* Oracle: depth-uniform border traps with log-spaced tau_j = tau0 exp(z_j/lambda). Over
  [1/(2 pi tau_max), 1/(2 pi tau_min)], S f is flat within the ripple bound of a Lorentzian sum on a
  log grid. Compute it in the test: about 1 % at 4 bins/decade.
* Also check the slope d ln S/d ln f = -1 +- 0.02 in mid-band, and the FDT as K3-5.
* The oracle formula (a uniform N_t(z) with tau ~ exp(z/lambda) gives a 1/tau density) is elementary.
  The McWhorter amplitude is NOT gated, because its source is unverified.

**Adversarial attempts on passing results.**
* K1-2: repeat with a deep level near E_c (n1/n0 = 1e3) and tau_n/tau_p = 1e6, with BGN on and the
  "valence" reference, and under extended precision. The identity must still hold.
* K2-1: move the bias until minority exchange is NOT negligible (weak inversion). K2-1 must fail by
  the computed c_n n_s/(c_p p_s), proving the oracle's assumption is load-bearing. Then the
  two-carrier extension (N&G section 6.5) passes.
* K3-1: repeat at 77 K (T_K) and with BGN, where the different n_i' and V_t would expose any T_REF
  leftover.

### 1.6 Test placement and runtime

* `tests/test_trap_kinetics.py` (non-slow; target <= 120 s single process):
  * K0 A/B and mechanism unit tests (about 5 s);
  * K1-1 (pure numpy, < 1 s);
  * K1-2 on the 1-D diode and MOS (about 20 s);
  * K1-4 at a coarse dt (about 15 s);
  * K1-6;
  * K2-1 at 5 frequencies, single level (about 20 s);
  * K3-1, the flipped xfail (about 15 s; the existing fixture costs about the same);
  * K3-2(a) (about 10 s).
* `@slow` (nightly): K1-5 full charge accounting, K2-3 continuum with bin convergence, K2-5, K2-6
  2-D/3-D, K3-3, K3-4, K3-5, K3-6, K5.
* `validation/trap_kinetics.py`: every gate with its numbers printed. Nightly full tier, estimated
  10-15 min.

### 1.7 Risks and unknowns

1. **Chain rule through named models with an extra variable (K0).** If DEVSIM does not
   differentiate a composite through a `<term>:<var>` model for a non-standard variable, the
   composites need explicit derivative strings for each extra variable. That is more code but not a
   blocker.
2. **Interface unknown count.** In 2-D/3-D the masked region-wide variable is wasteful. If the
   measured budget is too small to be useful, interface kinetics stay 1-D-only and K2-6 is reduced to
   2-D with 3 columns.
3. **Conditioning.** A level with tau_n0 << tau_p0 by 1e10 makes the u-row stiff. Newton should cope
   (the rows are local). Transient BDF should too: DEVSIM's BDF is implicit.
4. **Gummel staging** of trap rows on hard-to-converge devices. The fallback is to solve kinetic
   traps only in the coupled Newton (they are seeded exactly, so the first coupled iterate is the
   quasi-static solution).
5. **The low-frequency floor of Re Y.** D2b shows that ssac Re Y can sit far below its round-off
   floor. Every K2/K3 device is checked: G_p >> floor at all gated frequencies, with the floor
   measured on the trap-free twin.

### 1.8 Dependencies and effort

* Needs: nothing from items 2 and 3. K3 edits `noise_analysis`, as item 2 does, so do item 2 first or
  give both to one owner.
* Coordinate with the SG agent on the cross-version tolerance only.
* Effort:
  * K0: 2-3 days;
  * K1: 4-5 days;
  * K2: 4-6 days (conductance oracles, budget);
  * K3: 3-4 days;
  * K4: 1 day;
  * total 14-19 dev-days;
  * K5 (optional): 4-6 days more.

---------------------------------------------------------------------------------------------------

## 2. ITEM 2: shot noise of thermionic-emission heterointerfaces

### 2.1 Goal and physics

The D1a interface flux is two opposed Richardson currents (Horio & Yanai, IEEE TED 37, 1093 (1990);
D1a doc section 2):

    J(0->1) = J_01 - J_10,   J_01 = A* T^2 exp(w_0),   J_10 = A* T^2 exp(w_1),
    w_i = L_i - k_i = (E_F,i - E_top)/kT  (BGN-effective edges when BGN is on)

Each direction is an independent Poisson stream of carriers over the barrier. The one-sided spectral
density of the net interface current density, averaged over an area A, is

    S_J = 2 q (J_01 + J_10) / A

At zero bias J_01 = J_10 = J_s. The interface conductance is dJ/dV = q J_s/kT (the linearization of
the QFL-form flux), so S_J A = 4 q J_s = 4kT (q J_s/kT). That is exactly Nyquist for the interface
element. In strong forward bias J_10 << J_01, which gives full shot noise 2qJ.

This is the coth law already used as oracle C1 in D2a: 2q(I_f + I_r) = 2qI coth(qV/2kT) (van der
Ziel, Proc. IRE 43, 1639 (1955), cited there). The same law appears for Schottky diodes as
"2qI coth(qV/2k_BT)", with suppression toward thermal noise once Coulomb correlation or series
regions take over [G. Gomila, L. Reggiani, J. M. Rubi, J. Appl. Phys. 88, 3079 (2000). **Only the
abstract was read** (arXiv cond-mat/0009056); no equation number is cited].

**Composition.** Emission events at the interface are microscopically distinct from bulk velocity
fluctuations and from bulk g-r events. The impedance-field sum adds them incoherently. The
equilibrium FDT of the whole discrete device then holds exactly, because every element (SG edge, g-r
node, TE interface pair) is Nyquist-exact in its own linearization. This is the same argument D2a
uses, with gates B and G.

### 2.2 Current code state and reuse

* `physics_heterojunction.setup_heterointerface_carriers` writes the TE `fluxterm` model
  `<iface>_HJ<n|p>` = pref * (exp(w0) - exp(w1)) from strings w0 and w1. These strings are the ONE
  place the barrier, the BGN effective edges and V_t enter.
* `noise_analysis` has no interface sources; a TE interface is silently noiseless. That is the strict
  xfail `test_noise_fdt_on_a_thermionic_heterointerface`, which misses by -0.69 at 1 Hz with
  A* x 1e-6.
* `physics_traps.interface_node_area` gives each node's share of an interface. Whether it equals the
  area DEVSIM uses to integrate a `fluxterm` is NOT yet known: step T0 measures it.

### 2.3 Design

* **Pair model, one kernel.** For model="thermionic", `setup_heterointerface_carriers` also writes
  `interface_model <iface>_HJ<n|p>_pair = |pref| * (exp(w0) + exp(w1))` [A/m^2], from THE SAME
  w0/w1 strings. It writes no derivatives: a source needs only a value. This is the interface analogue
  of the `PTerm_*` pair rate.
* **Noise sources.** A new `noise_analysis._interface_sources(device)`:
  1. List interfaces with `get_interface_equation_list`.
  2. For each continuity equation of type `fluxterm`, require the pair model; otherwise raise
     RuntimeError ("a fluxterm interface without a pair model would be silently noiseless"). A
     foreign fluxterm becomes a refusal, which is a behaviour change (D-T2).
  3. Pair the two region copies of each interface node by coordinate. Get each region's interface
     node indices from `get_element_node_list(region, interface)`, match by (x, y, z) within 1e-12 of
     the device size, and assert a one-to-one match.
  4. Read the pair values with `get_interface_model_values`, which are ordered on the interface; map
     them to node pairs, again by coordinate, asserted.
  5. Per frequency, S_te = sum_i A_i * 2 q * pair_i * |y_r0,i - y_r1,i|^2 per carrier. The fluxterm
     adds to region 0 and subtracts from region 1 (D1a doc section 2), so a current-fluctuation source
     enters +delta / -delta and its Green's function is y_r0 - y_r1. The y come from the native noise
     fields of each region's continuity equation at its own node copy.
* **Area A_i.** Use exactly the weight DEVSIM's fluxterm assembly gives each node (T0 measures it).
  If it equals `interface_node_area` node for node, reuse that function: one source. Otherwise add a
  `fluxterm_node_area` measured from the assembled residual of a unit interface model, and use it.
  **Never** use a geometric area that DEVSIM does not use: the FDT would then break on a non-uniform
  2-D interface mesh.
* **Outputs.**
  * Mechanisms "te_n" and "te_p" are added to `MECHANISMS` and to the default `mechanisms` tuple
    (D-T1).
  * `NoiseSpectrum` gains `by_interface: Dict[str, Dict[str, np.ndarray]]` (default `{}`, so the
    change is backward compatible). The documented invariant becomes
    S = sum(by_region) + sum(by_interface) = sum(by_mechanism).
  * `maps=True` splits an interface contribution half to each node copy.
* **Composition.**
  * The continuous-QFL interface needs no source: it is the A* -> infinity limit, where the
    contribution 4kT G_int |dy|^2 -> 0 as dy ~ 1/G_int. This is checked by T-b.
  * BGN, T_K and holes are inherited through w0/w1.
  * Extended precision: the noise solve stays a double solve (documented floor).
  * 3-D DD heterostacks: electrons only, same code.
  * The shipped `temperature_K` per region already comes from V_t.

### 2.4 Ordered steps

* **T0 (measurement).** On a 2-D z-stacked heterostack with a NON-uniform lateral mesh, and on a 3-D
  tensor-mesh stack:
  1. Set the TE interface model to a constant 1 A/m^2.
  2. Read the per-node interface contribution from `get_matrix_and_rhs` residuals (difference with
     the model at 0).
  3. Compare with `interface_node_area`. Record the result in the doc.
* **T1.** Write the pair model in `physics_heterojunction`, plus a unit test: pair >= |flux|, and at
  zero bias pair = 2 x J_s.
* **T2.** Add `_interface_sources`, the node pairing, the area, the mechanisms, `by_interface` and
  the refusal for a foreign fluxterm.
* **T3.** Flip the xfail. Add the gates, the doc section (heterojunction doc section 4c and noise doc
  "Interfaces"), and remove the LIMIT lines from both module docstrings.

### 2.5 Verification gates

| # | Gate | Oracle | Tolerance and derivation | Fails on the old code because |
|---|---|---|---|---|
| T-a | FDT at zero bias, interface-dominated | `hetero_two_region("thermionic", astar_scale=1e-6)`: S_I = 4kT Re Y from 1 Hz to 1 GHz. | 1e-4, the D2a FDT tolerance. | The strict xfail: -0.69 at 1 Hz. |
| T-b | A* sweep | astar_scale in {1e-6, 1e-4, 1e-2, 1, 1e2}: (i) the FDT holds at every scale to 1e-4; (ii) the TE share by_mechanism[te]/S falls monotonically; (iii) S(A* x 1e6) equals S of the continuous model to 1e-4 (D1a gate E reached -1.1e-5 in DC). | 1e-4. | Without the source, (i) fails at small A*. |
| T-c | Forward bias, interface-dominated isotype barrier | n-GaAs / N-Al0.3Ga0.7As isotype (the D1a gate C device) with A* scaled down until R_int/R_tot >= 0.999 at all biases; sweep V = 0 .. +8 kT/q. Oracle from DC only: J(V) by solve, J_s = (kT/q) G_0 with G_0 from a +-0.1 mV central difference, and S_oracle = 2q(J + 2 J_s). At 4kT/q and beyond, S/2qJ -> 1. | 3e-3. Derivation: the bulk and depletion-region share of the differential resistance, r_b/R <= 1e-3 (computed from the solve), enters the series-chain noise S = sum_k (r_k/R)^2 S_k at about 2 r_b/R. Gate at 3x that value, stated in the test. | No interface source: S is about 4kT/R_bulk-share only. The miss is O(1). |
| T-c' | Crossover (adversarial) | Same device with A* raised until the bulk takes over: the series-chain oracle S = sum (r_k/R)^2 S_k, with r_int from the TE closed form at the solved phi_B and r_b from the DC split. | 5e-3. This shows the source is not a constant fudge: it must track the interface's share. | -- |
| T-d | 2-D area weights | z-stacked 2-D heterostack, lateral mesh deliberately non-uniform (ratio 5), TE interface: FDT 1e-4. Adversarial: replace A_i with equal shares, and the FDT must break (> 1e-3). | 1e-4. | -- |
| T-e | Native adjoint on interface copies | The native y on both copies against an equilibrated scipy adjoint. | 1e-12. | -- |
| T-f | Defaults and refusals | (i) A device with no TE interface: `noise_spectrum` arrays bit-identical same-tree before and after the change (the new code path is inert), and `by_interface == {}`. (ii) A hand-made fluxterm with no pair model raises. (iii) The continuous model is unchanged (the existing 5.7e-6 gate). | Exact / raise. | (ii) Old code silently drops it. |

### 2.6 Placement and runtime

* Non-slow: T1 unit test (< 1 s), T-a (about 15 s, existing fixture), T-f (about 10 s) in
  `tests/test_devsim_phase_composition.py` (the xfail lives there) and `tests/test_devsim_noise.py`.
* `@slow`: T-b, T-c, T-c', T-d, T-e.
* Validation: add gate H to `validation/devsim_noise.py`, about 2-3 min.

### 2.7 Risks and unknowns

* **T0 could show that DEVSIM's fluxterm area differs from `interface_node_area`** (for example, the
  "first interface owns a shared node" rule). The design then uses the measured function. The FDT
  gate T-d is what proves it.
* **T-c needs a device where the TE interface truly dominates.** The depletion region on the AlGaAs
  side is itself a DD barrier with its own shot-like diffusion noise. If r_b/R cannot be pushed below
  1e-3, the gate uses the series-chain oracle (T-c') as primary, with the tolerance re-derived.
* **The degenerate emitting side.** The TE magnitude uses the Maxwell-Boltzmann tail (D1a limit). The
  pair rate inherits it. The equilibrium FDT is still exact (both use the same w), but the off-
  equilibrium shot magnitude on a degenerate side is as approximate as the current. Document it.

### 2.8 Dependencies and effort

* Independent of items 1 and 3. Do it before K3 (both touch `noise_analysis`).
* Effort: T0 0.5 day, T1-T2 2 days, T3 and gates 1.5-2 days; total about 4-5 dev-days.

---------------------------------------------------------------------------------------------------

## 3. ITEM 3: heterointerfaces at T_K != T_REF (re-scoped; see section 0.1)

### 3.1 Goal and physics

Goal: honour T everywhere a heterostack touches temperature, or refuse by name. Never use a T_REF
quantity silently. An opt-in band-temperature model lets 0 K / 300 K tabulated material data be
simulated at T.

**Varshni form.** I. Vurgaftman, J. R. Meyer, L. R. Ram-Mohan, J. Appl. Phys. 89, 5815 (2001),
eq. (2.13), VERIFIED from the extracted text:

    E_g(T) = E_g(T = 0) - alpha T^2 / (T + beta)

The form is due to Y. P. Varshni, Physica 34, 149 (1967) (their ref. 27; not accessed). Recommended
parameters, VERIFIED from the same paper's tables (Gamma gap):

| Material | E_g(0) [eV] | alpha [meV/K] | beta [K] | VBO [eV] | Source |
|---|---|---|---|---|---|
| GaAs | 1.519 | 0.5405 | 204 | -0.80 | Table I |
| AlAs | 3.099 | 0.885 | 530 | -1.33 | Table II |

* AlGaAs Gamma-gap bowing: C = -0.127 + 1.310 x eV (Table XII). The bowing is "independent of
  temperature" (their recommended assumption, in the AlGaAs text). The VBO is linear in x (no bowing
  is listed in Table XII).
* Si, Varshni form: E_g(0) = 1.17 eV, alpha = 4.73e-4 eV/K, beta = 636 K. These come from a
  MINIMOS-NT / Palankovski (TU Wien) table found by search. **The primary attribution is
  unverified.** The Bludau-Onton-Heinke (1974) Si fit is what Sentaurus cites; it was not accessed.

Arithmetic checks, mine, to be re-computed by the gate:
* GaAs: E_g(300 K) = 1.519 - 0.5405e-3 x 9e4/504 = 1.422482 eV. Table I quotes "1.420-1.435 (300 K)"
  as the range.
* AlAs Gamma: E_g(300 K) = 3.003036 eV.

**Affinity convention: there is no single standard.**
* (a) Valence-anchored. Vurgaftman et al. tabulate a temperature-INDEPENDENT VBO on an absolute scale,
  so a gap change moves E_c only: chi(T) = chi(T_ref) + [E_g(T_ref) - E_g(T)].
* (b) Symmetric. Sentaurus Device's default affinity model is chi(T) = chi(0) + alpha T^2/(2(T + beta)):
  half the gap shrinkage goes into the affinity. **This was read from a Sentaurus training mirror (PMI
  section 17.5.4, ghzphy.github.io/Sentaurus_Training/sd/sd_17.html), a SECONDARY source.** The
  licensed manual was not accessed.
* (c) A user fraction gamma_chi in [0, 1]: chi(T) = chi(T_ref) + gamma_chi [E_g(T_ref) - E_g(T)].
  This contains (a) (gamma = 1) and (b) (gamma = 1/2).

The D1b BGN split `conduction_fraction` is the same kind of choice for doping, and it was a user
decision there too (D-H3).

**Why it matters, quantitatively** (my arithmetic, Vurgaftman parameters, Al0.3Ga0.7As against
GaAs): from 250 K to 350 K the alloy gap changes by -47.26 meV and GaAs by -45.11 meV, a difference
of -2.15 meV. So DeltaE_c(T) differs between conventions (a) and (b) by about 1.1 meV over that
range. That is small but far above the 1e-6 eV offset gates. For Si/SiGe-type pairs the mismatch is
larger.

**Thermal factors already handled (keep them).**
* A* T^2 at the region temperature. A* comes from the DOS mass; the mass's temperature dependence is
  not modelled (documented).
* N_c(T), N_v(T) proportional to T^(3/2) through `effective_dos_m3(m, T)`.
* Every kT.

### 3.2 Current state: the remaining silent paths (with evidence)

| # | Path | Evidence | Effect |
|---|---|---|---|
| H-1 | chi used as given at any T | `devsim_layered._semiconductor_frames`: `chi[sp.name] = float(tr.chi_eV)`; `devsim_3d` `band_alignment` likewise; no temperature tag on `TransportModel.chi_eV` | DeltaE_c at T off by gamma_chi x [mismatch of DeltaE_g(T) between the two materials] |
| H-2 | 3-D bipolar body at `T_K` uses N_c, N_v at T_REF | `devsim_3d.py` 816-817 (`effective_dos_m3(s.dos_mass_kg)`), 894-896 (phi_bi), 989 (`temperature_K=PE.T_REF`) | FD normalisation, phi_bi and the reported T are wrong for T_K != 300 K (largest for a degenerate body) |
| H-3 | Interface-trap default bin = the 300 K kT | `physics_traps.py` 425 | at 77 K the bins are 3.9 kT, so the midpoint estimate exp(-2 pi^2 kT/dE) degrades from about 3e-9 to about 6e-3 (estimate) |
| H-4 | BTBT at T_REF | `tunneling.py` 41, 271 | the D1d owner's item; a warning or refusal under T_K is a decision (D-H4) |
| H-5 | Unipolar DD and equilibrium regions have no T_K | by design; a mixed stack is refused | none (documented) |

### 3.3 Design

* **H1 (close H-2, H-3; H-4 by decision).**
  * `devsim_3d._setup_bipolar_semi` and the phi_bi block use
    `effective_dos_m3(m, spec.bipolar_options.T_K)` when T_K is set, the same rule as the 2-D builder
    (one helper `bipolar_options.band_dos_at_T(m_n, m_p, options)` used by both builders).
  * The `CarrierField.temperature_K` comes from the region V_t (`HJ.temperature_of_V_t`).
  * `InterfaceTraps.sheet_levels(kT_eV=None)`: `install_interface_traps` passes the region `V_t`.
    `None` keeps `V_T` for devsim-free callers.
  * All three change results ONLY at T_K != T_REF, and those results are currently silently wrong
    (D-H1).
* **H2 (opt-in band-temperature model; closes H-1).** A new frozen dataclass in `materials` (next to
  `TransportModel`):

  ```python
  @dataclass(frozen=True)
  class BandTemperature:
      Eg0_eV: float                      # Varshni E_g(T = 0)
      alpha_eV_K: float                  # Varshni alpha
      beta_K: float                      # Varshni beta
      affinity_fraction: float           # gamma_chi in [0, 1]; REQUIRED, no default (D-H3)
      chi_reference_T_K: float = T_REF   # the temperature chi_eV is quoted at
      derive_n_i: bool = False           # n_i(T) = sqrt(N_c(T) N_v(T)) exp(-E_g(T)/2kT)
  TransportModel.band_temperature: Optional[BandTemperature] = None
  TransportModel.chi_reference_T_K: Optional[float] = None    # assert-only tag (D-H2)
  ```

  * **One kernel:** `physics_heterojunction.band_data_at_T(tr, T) -> (E_g, chi, n_i | None)`. Both
    builders' frames, the band-gap check and the `_hetero_side` E_g read it.
  * With `band_temperature=None` it returns `(tr.band_gap_eV, tr.chi_eV, tr.n_i_m3)` untouched, which
    keeps the A/B guard bit-identical.
  * With the model, at T = `chi_reference_T_K` it returns the given chi exactly (the delta is
    0.0 x gamma).
* **H2 guard (D-H2).** A bipolar heterostack at T_K != T_REF whose materials carry neither
  `band_temperature` nor `chi_reference_T_K == T_K` is refused. The message names the missing
  temperature of chi and the two remedies. Without this the 300 K affinity is silently used at T,
  which is the literal "never silently use T_REF" rule.
  * This is a behaviour change for existing T_K heterostacks: gates T-a and T-b would need
    `chi_reference_T_K=T_K` added.
  * The alternative is an `AdvisoryWarning`.
* **Composition.**
  * BGN: Varshni is the undoped gap; BGN narrows it further, and the order is the D1b/D1a rule
    (material edge, then BGN effective edge).
  * The trap energy references and n1 already use region N_c, N_v and kT.
  * The noise reads T from V_t.
  * 3-D DD heterostacks run at T_REF only (no T_K for DD), so H2 applies to them only when DD gains
    a T_K (out of scope).

### 3.4 Ordered steps

1. **H0 Status proof.** Run `tests/test_devsim_heterojunction.py -k "temperature or T_"` on the tip.
   Record that T-a to T-d pass, and correct integration-note item 6.8 to "superseded by section 4b".
2. **H1** as above, with the A/B guard at T_REF and the new gates.
3. **H2** dataclass, kernel, builder wiring, guard.
4. **Docs.** Heterojunction doc section 4d; `bipolar_options` docstring item 4 (which inputs are now
   derived at T and which are still the caller's).

### 3.5 Verification gates

| # | Gate | Oracle | Tolerance and derivation | Fails on the old code because |
|---|---|---|---|---|
| H1-a | 3-D bipolar body at T_K = 250 K | A degenerate n body (eta ~ 5). The contact-node electron density equals N_c(250 K) F_1/2(eta) of the FD neutral root, with N_c from the mass at 250 K (independent numpy/scipy quadrature). phi_bi equals the 1-D twin built at 250 K. | 1e-10 relative (the root solver); phi_bi 1e-12 V. | N_c(300)/N_c(250) = 1.31, which moves eta by O(0.3), far above tolerance. |
| H1-b | Reported T | `CarrierField.temperature_K == 250.0` (3-D bipolar, T_K = 250). | Exact. | It reports 300. |
| H1-c | Bin width at 77 K | `sheet_levels` widths <= kT(77 K) at install. The LF C-V slope gate D1c (c-i') at 77 K against its closed form. | The D1c slope tolerance (2.3e-4). | Bins 3.9 kT wide; measure the slope error. |
| H2-a | Varshni numbers | E_g(T) for GaAs, AlAs, Al0.3Ga0.7As at 0 / 250 / 300 / 350 K against hand-coded Varshni + linear interpolation + Table XII bowing in the test (an independent formula, the Table I/II/XII constants typed from the paper). | 1e-12 eV. | No model exists. |
| H2-b | Identity at the reference T | Model on, T = chi_reference_T_K, with `derive_n_i=False`: frames, offsets and the interface strings bit-identical to the model-off device. | Exact. | -- |
| H2-c | V_bi(T), N-Al0.3Ga0.7As / p-GaAs | q V_bi = W_p - W_n at 250 and 350 K. E_g(T) and chi(T) come from the test's own Varshni and gamma_chi, N_c(T) and N_v(T) from the masses, and the FD terms by direct quadrature (as gate T-b). | 2e-4 V (the D1a T-b tolerance; measured 2.5-3.9e-5 there). | No chi(T): misses by gamma_chi x the DeltaE_g mismatch (about 0.5-1.1 meV), which must exceed 5x the tolerance. If not, use a pair with a larger mismatch (an AlAs-rich alloy) so that the gate is load-bearing. |
| H2-d | DeltaE_c(T) from the solved densities | On the interface node copies, against the convention's DeltaE_c(T). | 1e-6 eV (the D1a gate B tolerance). | As H2-c. |
| H2-e | Isotype TE G(T) with an INDEPENDENT barrier | G = 1/(R_bulk + kT/(q A* T^2 e^{-phi_B/kT})), with phi_B from the oracle's band diagram, not from the solve. This is stronger than the shipped T-a, which takes phi_B from the solved densities. | 2e-3 (D1a gate C). | The barrier is off by the chi(T) term: an exp(0.5-1 meV/kT) error is 2-4 %, more than 10x the tolerance. |
| H2-f | Derived n_i | `derive_n_i=True`: the band-gap check passes automatically, and n_i(T) equals sqrt(N_c N_v) exp(-E_g(T)/2kT) computed independently. | 1e-14 relative. | -- |
| H2-g | Refusal (if D-H2 = refuse) | T_K = 250 K heterostack with no tag: NotImplementedError naming chi; with the tag, it builds. | -- | It silently builds. |
| H-adv | Adversarial | Run H2-c with convention (a) against an oracle written in convention (b): it must fail. Run at T = T_REF with the model on: it must equal the model off. | -- | -- |

### 3.6 Placement and runtime

* Non-slow: H1-b, H2-a, H2-b, H2-f, H2-g (< 20 s).
* `@slow`: H1-a, H1-c, H2-c, H2-d, H2-e. The 2-D heterostack fixtures cost 10-30 s each; the slow
  equilibrium fixture is not used.
* Validation: extend `validation/devsim_heterojunction.py` with a `--temperature` section.

### 3.7 Risks and unknowns

* The parameters of the chi(T) convention are not standardised (D-H3). The gates are written to the
  chosen gamma.
* **Varshni at cryogenic T.** The form is an empirical fit, and some parameter sets misbehave below
  about 50 K. The docstring states the validity range. No gate goes below 77 K.
* **A* temperature dependence through m*(T)** is not modelled (documented).

### 3.8 Dependencies and effort

* H1: 1-2 days. H2: 3-4 days. Total about 4-6 dev-days.
* Independent of items 1 and 2.
* H-3 (bins) also touches `physics_traps`. Land it before K2, or let the K2 owner do it.

---------------------------------------------------------------------------------------------------

## 4. Workstreams, order and total effort

| Workstream | Contents | Effort (dev-days) | Order |
|---|---|---|---|
| WS-T | item 2 (T0-T3) | 4-5 | first (small, flips an xfail, touches noise before K3) |
| WS-H | item 3 (H0-H2) | 4-6 | in parallel with WS-T (disjoint files except `physics_traps` line 425: give H-3 to WS-H) |
| WS-K | item 1 (K0-K4) | 14-19 | K0-K2 in parallel with WS-T/WS-H; K3 after WS-T merges |
| WS-K5 | McWhorter border traps (optional) | 4-6 | after WS-K |

* Total: about 22-30 dev-days without K5.
* Every workstream ends with:
  * its audit note;
  * the same-tree A/B guard;
  * the cross-version round-off guard (after the SG agent merges);
  * the regression run of every devsim/carriers test file (`-n 3`);
  * `run_all --tier full` for the touched validations.

---------------------------------------------------------------------------------------------------

## 5. Decisions for the user (each with a recommendation)

| ID | Decision | Options | Recommendation |
|---|---|---|---|
| D-K1 | The charged-trap FDT strict xfail | (a) keep quasi-static as the default; add a passing kinetic twin (K3-1); keep the quasi-static high-f test as a strict xfail renamed "quasi-static model limit"; (b) make charged levels kinetic by default (DC unchanged to round-off; ssac and transient change, and are currently wrong above the quasi-static regime); (c) as (a), plus an `AdvisoryWarning` from ssac / noise / transient when quasi-static CHARGED traps are present | **(c)**: honest default, no result changes, and the user is told when the quasi-static model is out of its regime |
| D-K2 | Occupancy variable | logit u (trap quasi-Fermi level) or f with damping | **logit u**: bounded, no 1 - f cancellation, exact seed; documented |u| < 700 limit |
| D-K3 | Interface kinetics in 2-D/3-D | (a) region-wide masked variables with an unknown budget and refusal; (b) also research `custom_equation` + circuit-node unknowns | **(a) now**; (b) only if a 2-D kinetic D_it study is actually needed |
| D-K4 | Noise mechanism names | new "trap_n" / "trap_p" keys, or fold into "gr" | **new keys** (the split is informative; S still sums) |
| D-K5 | McWhorter / border traps | (a) out of scope; (b) K5 as specified, 1-D only, amplitude not gated against literature | **(a) for v0.15**. Lorentzian g-r per level ships with K3, and Hooge 1/f exists. Revisit when a primary McWhorter source is in hand |
| D-T1 | TE shot noise on by default | on (a missing physical source; the FDT proves the old number wrong) or opt-in | **on by default**, flagged in the release notes: `noise_spectrum` results change for every TE heterointerface device |
| D-T2 | A fluxterm interface without a pair model | refuse or ignore | **refuse** (the shipped "never silently noiseless" rule) |
| D-H1 | Fix the 3-D bipolar T_K N_c / N_v, the reported T, and the trap bin kT | fix (results change only at T_K != T_REF, where they are wrong) or document | **fix** |
| D-H2 | chi at T_K != T_REF without temperature data | refuse (with an assert tag `chi_reference_T_K` or a `band_temperature` model as the remedy) or AdvisoryWarning | **refuse**: the literal "never silently use T_REF"; the existing T_K gates get a one-line tag |
| D-H3 | Affinity convention | valence-anchored (Vurgaftman VBO), symmetric (Sentaurus, secondary source), explicit fraction | **explicit fraction, required, no default**. The docstring explains both conventions |
| D-H4 | BTBT under T_K | refuse, warn, or leave to D1d | **warn now** (AdvisoryWarning: "BTBT parameters at T_REF"); the D1d owner decides on scaling |

---------------------------------------------------------------------------------------------------

## 6. Source ledger (what was verified, and how)

| Source | Used for | Status |
|---|---|---|
| Nicollian & Goetzberger, BSTJ 46, 1055 (1967) | eqs (1)-(6), (12), (15)-(20), (22); peak omega tau = 1.98 (Fig. 13) | VERIFIED: scanned issue (bitsavers BSTJ_V46N06_196707.pdf, pp. 1068-1075) |
| Nicollian & Brews, MOS Physics and Technology (1982) | textbook context | NOT accessed; no equation numbers cited |
| Shockley & Read 1952; Hall 1952 | rate equations | content standard; equation numbers NOT verified |
| Lang, JAP 45, 3023 (1974) | e_n form | used as e_n = c_n n1 (detailed balance); equation number not cited |
| Vurgaftman, Meyer, Ram-Mohan, JAP 89, 5815 (2001) | eq. (2.13); Tables I, II, XII; T-independent bowing | VERIFIED: text extracted from the PDF |
| Varshni, Physica 34, 149 (1967) | origin of the form | cited through Vurgaftman ref. 27; not accessed |
| Si Varshni (1.17 eV, 4.73e-4, 636 K) | Si E_g(T) | search results (MINIMOS-NT / Palankovski table); primary attribution UNVERIFIED |
| Sentaurus chi(T) = chi(0) + alpha T^2/(2(T + beta)) | convention (b) | SECONDARY (training mirror, PMI section 17.5.4) |
| van der Ziel 1955 (coth law) | T-c | already the D2a C1 oracle; equation number not re-verified |
| Gomila, Reggiani, Rubi, JAP 88, 3079 (2000) | shot-to-thermal crossover | abstract only (arXiv cond-mat/0009056) |
| van Vliet & Fassett 1965; Bonani & Ghione 2001 | Langevin sources per process; g-r variance | NOT accessed; K3-4's variance is derived in the test, and the FDT gate is the independent check |
| McWhorter 1957; Christensson et al. 1968; Hung et al. 1990 | border-trap 1/f | NOT verified (only a garbled secondary thesis page); K5 amplitude deliberately ungated |
| DEVSIM 2.11.2 behaviour (node-only equations at contacts; ssac and noise fields for extra equations) | sections 0.3, 1.3.4 | MEASURED for this plan (scratchpad `spike_node_eq.py`) |
