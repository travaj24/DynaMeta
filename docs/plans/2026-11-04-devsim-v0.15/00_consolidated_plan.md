# DynaMeta DEVSIM work after v0.14: one consolidated plan for review

This plan consolidates four detailed plans in this folder:
- **P1**, tunnelling (`P1_tunneling.md`);
- **P2**, materials and API (`P2_materials_api.md`);
- **P3**, traps, noise and heterointerfaces (`P3_traps_noise_hetero.md`);
- **P4**, density gradient, impact ionization and numerics (`P4_dg_ii_numerics.md`).

It also uses the coordinator log and the state of the in-flight branches, both read on 2026-11-04. No repository was modified.

**[C]** marks a point that comes from this consolidation, not from the plans. The evidence is given each time.

**Terms used throughout:**
- **Gate:** an automated test with a stated pass threshold.
- **Oracle:** the independent reference a gate compares against.
- **Agent-day:** one day of one implementation agent. The plans say "agent-days", "dev-days" or "days"; they are treated as equal here.
- **Solvers leg:** the only per-push CI job that installs DEVSIM (45-min cap). The **nightly** jobs run the slow tests and validation scripts.
- **ssac:** small-signal AC analysis, giving the admittance Y(ω).
- **FDT:** fluctuation-dissipation theorem. At equilibrium the current noise must equal 4kT Re Y. It is the strongest available check on any noise source.
- **xfail:** a test marked "expected to fail"; it records a known limitation.

---

## 1. Summary

**What is planned.** Every deferred item from the v0.14 DEVSIM phase:
- **Tunnelling:** gate-oxide tunnelling current fed back into the device equations; cited Si/Ge band-to-band tunnelling (BTBT) sets; nonlocal BTBT.
- **Materials and API:** cited Si parameter bundles with a "house" band-gap-narrowing (BGN) model and a matched intrinsic density n_i; III-V recombination presets; per-material Hooge 1/f coefficients; a one-call noise method; extended precision selectable from the design.
- **Traps and heterointerfaces:** trap occupancy with its own time constant (enabling the conductance method and trap noise); shot noise at thermionic heterointerfaces; band data at temperature.
- **Quantum and numerics:** density gradient through strong inversion; retirement of the old unipolar "hard wall"; correct 2-D impact ionization; automatic detection of ill-posed DC problems; a density-scaled solver tolerance.

All of this rests on three pieces of shared infrastructure, built once.

**Effort.**
- **Before the v0.14.0 tag:** 4-9 agent-days of fixes, beyond the three branches already in flight.
- **v0.15:** about 67-90 agent-days. That is about 4-5 weeks of wall time at 5 concurrent agents, set by two long chains: oxide tunnelling and trap kinetics.
- **"Later" items:** 14-20 agent-days plus unestimated items.

**Release split.**

| Release | Content |
|---|---|
| **v0.14.0** | The in-flight branches, plus fixes to v0.14's own unreleased code: CI wiring, preset corrections, the 3-D temperature defect, trap-bin width, BTBT temperature warning, ionization units guard. Recommended here as well: the 2-D impact-ionization correction, which changes released results. |
| **v0.14.1** | Whatever the user chooses not to hold the tag for. Notably the InGaAs gap row in the SOA module. |
| **v0.15** | The features in section 4. (The gmsh-mesher removal of decision 1, 2026-11-04, is cancelled: user decision 2026-11-06 keeps it, geometry-corrected.) |
| **Later** | 2-D nonlocal BTBT, McWhorter border traps, the Altermatt-2006 ionization model, hole and poly-gate tunnelling, more III-V materials, a DC charge constraint for isolated wells. |

**Decide first** (section 7, group A):
1. Which v0.14 findings hold the tag.
2. Whether the 2-D impact-ionization correction goes into v0.14.0.
3. Whether the temperature-of-affinity guard ships with `chi_eV=None`.
4. Whether to file the DEVSIM upstream issue.

Everything else can wait for your review of this document.

---

## 2. How everything is verified (shared rules)

1. **Independent oracle per gate.** The reference must not share code with the thing tested. Acceptable references: a closed form re-derived in the test, scipy quadrature written in the test, an equilibrated adjoint built from the assembled matrix, or a published value re-typed with its equation or table number. A gate that compares code with itself is labelled a consistency check, not a gate.
2. **Tolerance derived, not tuned.** Each threshold comes from a stated error budget: round-off times a measured condition number, O(h²) with a constant from a clean convergence study, or a paper's quoted uncertainty. A miss is a finding. Nothing is re-run to green, and no tolerance is loosened.
3. **Every new gate is shown failing on the old code**, by a stated amount.
4. **Adversarial refutation.** Each passing gate gets one deliberate perturbation that must break it: flip a sign, drop a source, swap a label, or coarsen an interpolation.
5. **Defaults off, checked three ways.** P3's rule; **[C]** all four plans should use it (see 5.5).
   - (i) Same-tree A/B. In one process, compare the new keyword absent with the keyword present at its default. Model inventory, equation commands, model strings and arrays must be bit-identical.
   - (ii) The call-log digest. It hashes every DEVSIM model and equation definition, and was just re-recorded by the Scharfetter-Gummel (SG) branch.
   - (iii) That branch's numeric cross-version gate against the 6595bde arrays, at measured round-off.
6. **CI budget.**
   - Every new DEVSIM test file goes on the solvers-leg list in the same pull request. A guard test, being added now, fails if a DEVSIM test file is on no leg.
   - Tests over about 60 s run nightly.
   - Physics validations go into the nightly validation tier.
   - Budget arithmetic is in 5.8.

---

## 3. Before v0.14.0 can ship

### 3.1 In-flight branches (read 2026-11-04)

| Branch | State |
|---|---|
| **User decisions** | 9 commits, **merged into the integration branch (4ff95c2) on 2026-11-04**. They cover the gmsh warning, `GAMMA_WALL` guidance, quantum-SRH guidance, the shot-noise flag, `chi_eV`/`band_gap_eV` defaulting to None (a heterostack with a defaulted value now refuses to build), `TrapSpec(enabled=True)` raising, per-process in-memory 3-D meshes, lazy ngsolve, and the audit of non-exact-mesh fixtures. |
| **SG flux** | One commit; documentation edits still uncommitted. One flux kernel for every family, in sign-selected two-Bernoulli form. Accuracy ≤ 3.3e-16 in the value and ≤ 8.9e-16 in the Jacobian up to \|v\| = 40. The old form lost 6.5e-5 at v = −27.6 and was fully wrong at −40. Call-log digests re-recorded; a numeric cross-version gate added. |
| **Release blockers** | **Hooge defect confirmed and fixed.** Off-axis current gave 0.49999999982 of the truth at 45° (exactly cos⁴ + sin⁴); the 3-D gmsh bar gave 0.688. The source is now assembled per element, as a rank-one conductivity-fluctuation tensor: 45° gives 0.99999999964, and 1-D is unchanged to 4e-16. **Stale-solve defect confirmed and fixed.** A solve after the noise-gate setup returned the previous state, 1.0 V off in potential and 0.22 off in ITO electron density; it is now ≤ 6e-16 from a fresh solve. **CI wiring: not yet committed.** |

**[C]** The per-element Hooge fix supersedes two of P2's proposals, because the new assembly is exact on well-centred meshes:
- P2's "refuse misaligned meshes" (P2 D9);
- P2's 2-4 day "correlated source" research item.

On gmsh tetrahedra it is still off: 1.055 instead of 1. The cause is DEVSIM's unsigned circumcentric volumes (the node volume is 1.225x the true volume). That case is covered by row 3.2-13.

### 3.2 Findings in v0.14's own code

**How to read the "Released?" column.** It was checked from git history:
- "v0.14" means the code is unreleased, so fixing it changes no user's results.
- "Released" means the code shipped in v0.13.0 or earlier.

| # | What is wrong | Size | Fix (effort) | Released? | Block the tag? |
|---|---|---|---|---|---|
| 3.2-1 | DEVSIM test files on no CI leg. P2 counted 7; **[C]** a recount on 6595bde finds **12**: recombination/BGN, noise, extended precision, phase composition, traps, heterojunction, tunnelling, bipolar density gradient, 3-D mesh, transient dt floor, trap energy reference, and a carriers audit file. | Every v0.14 physics gate is unprotected. | Add them to the solvers leg; move the heterojunction slow gate (642 s locally) to nightly; add a guard test (0.5 d, in flight). | v0.14 | **Yes** |
| 3.2-2 | Hooge cross terms dropped. | 0.5x at 45°. | Done on the blockers branch. | v0.14 | **Yes:** merge |
| 3.2-3 | Stale operating point after the noise setup. | 1 V off, silently. | Done on the blockers branch. | v0.14 | **Yes:** merge |
| 3.2-4 | **del Alamo BGN preset.** The fit is n-type only, in N_D: 18.7 meV · ln(N_D/7e17). Sources: IEDM 1985 Eq. 7, TED 1987 Eq. 11, read in the primary. The code applies it to N_D + N_A, and the cited SSE 28, 47 is not where the fit appears. | No change in pure n-type material. In p-type or compensated regions it applies a fit with no basis there: about 50 meV at 1e19 cm⁻³. | Evaluate in N_D only; correct the citation; advise or refuse on p-type nodes (0.5 d with the next two rows). | v0.14 | **Yes** (P2 D4) |
| 3.2-5 | **Sb ionization energy 39 meV.** **[C]** This matches its cited table (Sze & Ng), but that table is out of date. The spectroscopic value is 42.74 meV (Ramdas & Rodriguez 1981, Table 4). | Donor level 3.7 meV too shallow, i.e. 0.14 kT at 300 K. | 42.7 meV with the new citation. P, As and B are optional (each moves < 1 meV). | v0.14 | **Yes** |
| 3.2-6 | **Old Slotboom preset** was fitted on p-type bases, with a paired n_i of about 1.44e10. The audit note's claim "the modern pairing is n_i = 1.0e10" is wrong: no shipped set was fitted at 1.0e10, and the modern value is 9.65e9. | Pairing these sets with 9.65e9 under-predicts n_ie² by about 2.3x, which is about 21 meV. | Documentation and source notes only. Legacy presets stay unpaired, so no gate moves. | v0.14 | **Yes** (docs) |
| 3.2-7 | **3-D bipolar model ignores T_K for N_c and N_v.** Both the body and the gate frame use 300 K values (devsim_3d around lines 816-817 and 894-896), and the result reports 300 K (around line 989). Confirmed in code. | At 250 K, N_c changes by 1.31x, which shifts the reduced Fermi level of a degenerate body by about 0.3. The reported T is wrong. | One shared DOS-at-T helper for both builders; report the region T (P3 H1, 1 d). | v0.14 | **Yes.** Results change only where they are currently wrong (D-H1). |
| 3.2-8 | **Interface-trap default bin width uses the 300 K kT** (physics_traps around line 425; confirmed). | At 77 K the bins are 3.9 kT wide, against a documented ≤ kT. The midpoint-rule error goes from about 3e-9 to about 6e-3 (P3's estimate). | Pass the region kT (0.25 d). | v0.14 | **Yes** |
| 3.2-9 | **BTBT always runs at 300 K**, even when T_K is set. | B is silently evaluated with the 300 K gap. | Advisory warning now (D-H4); the scaling itself is v0.15 (0.25 d). | v0.14 | **Yes** (warning) |
| 3.2-10 | **Electron affinity χ is used at any T** as if it had been quoted at that T. | About 1.1 meV in ΔE_c for AlGaAs/GaAs from 250 to 350 K (P3); larger for Si/SiGe. | Refuse a bipolar heterostack at T_K ≠ 300 unless the material states the temperature at which its χ was quoted. The full model is v0.15 (0.5 d). | v0.14 | **Recommended** (decision 3): users then meet one change, together with `chi_eV=None`. |
| 3.2-11 | **Ionization units trap.** The field is in eV·m, but literature quotes eV·cm. An unconverted 3.1e-8 zeroes the ionization energy above about 3e12 cm⁻³, and the clamp hides this as full ionization. Separately, the legacy model freezes out wrongly above the Mott density (39 % ionized at 1e19). | Silent and total. | A guard that raises when the energy would vanish below 1e17 cm⁻³; a converting constructor; a Mott-density advisory (P2 item 4; 0.5 d). | v0.14 | **Recommended:** once released, the trap is permanent. |
| 3.2-12 | **2-D impact ionization is directional.** The edge form converges to ∫[α(\|E_x\|)\|J_x\| + α(\|E_y\|)\|J_y\|] instead of α(\|E\|)\|J\|. Recomputed here: 0.270 of the truth at 45° and b/E = 4. Exact only for mesh-aligned current. | Large at junction corners, current crowding and drain corners; it feeds HCI lifetimes. | Use the 3-D element kernel in 2-D as well, with an exact-mesh check (P4 item 3; 2-3 d). | **Released.** **[C]** Code dates from June 2026, before v0.13.0. | **Decision 2.** Recommend **v0.14.0**: v0.14.0 already moves results through the SG switch, and moving 2-D results in a patch release would be worse. |
| 3.2-13 | **Element-based quantities on meshes that are not well-centred.** 3-D impact ionization checks for this; 2-D/3-D BTBT and the new Hooge source do not. | Example: Hooge 1.055 on a gmsh bar. Library meshes are exact; user meshes and the deprecated gmsh path are not. | One shared exact-volume check, refusing with a named error (0.5 d, with 3.2-12). | v0.14 | With 3.2-12 (decision 6) |
| 3.2-14 | **[C] InGaAs gap in the SOA temperature table.** P1 suspected it; checked here against the Vurgaftman 2001 text already in the scratchpad. The 0 K gap of 0.816 eV matches. But Vurgaftman's alloy rule (binary Varshni laws plus a temperature-independent bowing of 0.477 eV, Table XIII) gives E_g(300) ≈ 0.737 eV, while the repository's (0.816, 2.9e-4, 193 K) gives 0.763 eV. | 26 meV too wide at 300 K, in a shipped module outside DEVSIM. | Use a temperature law consistent with Vurgaftman, gate it against the re-typed tables, and measure the moved SOA numbers (0.5-1 d). | Released (SOA) | Not a DEVSIM blocker (decision 5) |
| 3.2-15 | Stale documentation: integration-note item 6.8 (superseded by commit f7c82db) and the HCI docstring on 3-D quadrature. | — | Fix at close-out. | — | Yes (docs) |

**Effort.** Rows 3.2-1 and 3.2-4 to 3.2-11 take 3.5-4.5 agent-days. Rows 3.2-12/13 add 2.5-3.5, and row 3.2-14 adds 0.5-1. With 2-3 agents this is about 3 calendar days.

**Documented in the v0.14 notes, not blocking:**
- Both builders hard-wire Fermi-Dirac statistics, so a Boltzmann-fitted apparent BGN is double-counted on degenerate layers. An advisory already fires; the fix is v0.15.
- The post-processed one-sided oxide current is nonzero at equilibrium.
- The default absolute tolerance is below one ulp on degenerate devices.

---

## 4. The work, by workstream

Shared infrastructure comes first. Each item below gives:
- what it is and why it matters;
- the chosen approach, and why;
- the key gates;
- the risks;
- the effort, in agent-days ("d");
- where to read the full detail.

### 4.0 Shared infrastructure

**4.0.1 A general "two-point" noise source**

Today every generation-recombination noise source is a same-node electron-hole pair. Four new features instead move one carrier between two places:
- thermionic emission across an interface;
- trap capture and emission;
- oxide tunnelling to the gate;
- nonlocal BTBT, with a hole at the path start and an electron at its end.

The general Langevin source is

S = q²·2P·\|Σ_r c_r y_r\|²

where:
- P is the event rate;
- y_r is the adjoint (impedance-field) value of each affected equation row;
- c_r is that row's stoichiometric sign.

**[C]** Build this first, split out of P1's workstream A, so the thermionic noise does not wait for the custom-equation mechanism.

*Gate:* today's BTBT and SRH noise, re-expressed through the new path on the FDT diode, must be bit-identical (or within 1 ulp if the summation order changes).

*Effort:* 1-1.5 d. *Read:* P1 §0.4.

**4.0.2 Recombination-term bookkeeping, version 2**

All recombination and charge terms enter through one routine. Its inventory, read back from DEVSIM, lets the noise analysis refuse any term without a noise model. Two plans extend it:
- **P3** needs electron-only and hole-only terms (a trap captures the two carriers at different rates), and derivatives with respect to extra unknowns.
- **P1** needs a "custom" kind for Python-assembled terms. Each declares its noise model and whether its Jacobian is exact.

**[C]** One agent does both. A spike first confirms that DEVSIM differentiates through named sub-models with respect to a non-standard variable. If it does not, explicit derivative strings are needed: more code, not a blocker.

*Gate:* same-tree A/B, with the composite strings unchanged when no new terms are present.

*Effort:* 2-3 d. *Read:* P3 §1.3.2 and step K0; P1 §0.3.

**4.0.3 Python-assembled terms (DEVSIM's custom-equation hook)**

Some terms cannot be written in DEVSIM's model language:
- oxide tunnelling at a node depends on the far oxide-face potential;
- nonlocal BTBT depends on the band profile along a path.

DEVSIM's custom-equation callback handles both. P1 probed it on a 1-D MOS:
- it converges to 6 V;
- charge balances to 2e-14 in double precision and exactly in extended precision;
- the entries reach the small-signal (ssac) analysis, matching the DC derivative to 1.8e-8;
- a wrong gate-row sign breaks conservation;
- **a registration can never be removed.**

*Design:*
- one process-wide dispatcher, registered lazily and never at import;
- a device is "live" only if DEVSIM still lists it and it carries a marker parameter;
- local terms are refused;
- explicit structural zeros keep the sparsity pattern fixed.

*Gates:*
- defaults off, including a child process that only *registers* the dispatcher;
- liveness after re-creating a device.

*Cost:* with a custom term, extended-precision Newton converges linearly (about 12 iterations to 7e-30).

*Effort:* 2-3 d. *Read:* P1 §0.

**4.0.4 One implementation per kernel [C]**

- **One exact-volume check**, used by impact ionization, BTBT and Hooge.
- **One Varshni function and table**, used by the band-temperature model, BTBT temperature scaling and the SOA module. Today the SOA module has its own; P1 reuses it, while P3 writes another.
- **The existing closed-form Fermi-Dirac "log activity" model**, ln u + Δ(u). Its arctan is built from half-angle reductions, because DEVSIM has none. P1 reuses it; P4 would rewrite it (see 5.6).

### 4.1 Tunnelling (P1)

**4.1.1 Self-consistent gate-oxide tunnelling**

*What and why.* Today the oxide current is post-processed from a one-sided Fowler-Nordheim (FN) or Schuegraf-Hu (S-H) formula. It never removes electrons from the body or adds current to the gate. That is wrong for:
- deep depletion, where tunnelling is the minority supply;
- heavy injection;
- terminal current balance;
- ssac and noise.

It also gives a nonzero current at equilibrium.

*Approach.*
- **Current:** a two-sided Tsu-Esaki integral with WKB transmission through a dielectric stack, with Anderson-rule barriers.
- **Supply:** the local electron quasi-Fermi level, through the bulk's own Fermi-Dirac activity. The current then vanishes exactly at equilibrium and scales with the surface density n_s at low density, with no fitted factor.
- **Gate barrier:** read from the gate contact's potential offset.
- **Equations:** a sink on each interface node's electron row, plus a source on the gate circuit row when the gate is circuit-driven.
- **Jacobian:** analytic.
- **Energy grid:** fixed per device. The existing adaptive grid steps J(V) by about 1e-10, which breaks finite-difference checks.

*Rejected:*
- n_s-scaled FN/S-H: nonzero equilibrium current and an FDT violation, both kept as proof gates.
- Auto-converting gates to circuits: circuits cannot be removed.
- Image-force lowering: the shipped 3.1 eV barrier is an FN fit that already absorbs it. Adding it would double-count about 0.26 V at 1 MV/cm.

*Gates:*
- **Detailed balance:** at zero bias, \|J_net\|/J_one-way ≤ 1e-10. The old code must give more than 0.5.
- **Supply-limited inversion** (p-Si, 2 nm oxide): I_gate equals the SRH generation integral, and the analytic deep-depletion estimate within 5 %. The old code must exceed it by more than 10x.
- **Conservation, ssac and noise:**
  - I_body + I_gate = 0 at the solver floor;
  - Re Y(f→0) equals the DC dI/dV to 1e-6;
  - FDT to 1e-4, which the one-sided shot formula (up to 2x over) must fail.
- **2-D and 3-D:** equal to 1-D to 1e-6 on tensor meshes. gmsh must warn: a 1.5 % oxide-voltage spread becomes about 37 % in current.

*Risks:*
- The Tsu-Esaki equation number is unchecked.
- The pair-noise formula is derived, not cited.
- Heavy FN may need damping.
- DEVSIM reports zero DC gate current, so its consumers need an audit.
- Electrons only.
- No stack J-V data is in hand (decision 16).

*Effort:* 12-16 d. *Read:* P1 §1.

**4.1.2 More BTBT parameter sets**

*What.* The Kao et al. 2012 Si and Ge sets, direct and indirect. Ge needs both summed, so a region accepts a tuple of sets, each with its own noise pair.

*How.*
- One function computes Kane direct-process sets; a computed preset is always a call of it.
- Opt-in temperature scaling: B scales as (E_g(T)/E_g(300))^1.5 (Sentaurus Eq. 478, read), using the one Varshni function.

*Rejected:*
- vendor default sets;
- Atlas's set (its units are misprinted);
- an untraceable InGaAs fit.

InGaAs ships only as a Kane-computed set. **[C]** Vurgaftman is already in hand, so this is no longer blocked on you.

*Gates:*
- each constant matches the printed value to 1e-15;
- the Kane function matches a CODATA re-coding to 1e-13;
- a 1-D Zener matches the analytic estimate to 1 % + δ_D, which catches a 100x unit error or a swap of the field exponent P;
- direct + indirect equals the sum of separate runs exactly;
- the Kao consistency check is a strict xfail until the primary is read (the Ge-direct A is 10 % off the Kane formula).

*Effort:* 2-3 d once Kao is read. *Read:* P1 §2.

**4.1.3 Nonlocal BTBT (1-D in v0.15)**

*What and why.* The local model generates carriers wherever the field is high, even where no final state exists, and it puts both carriers at one point.

P1's literature check corrects the motivation: nonlocal is **not** generally lower.
- It is about equal in Si homojunctions.
- It is higher in Si-Ge heterojunctions.
- Hurkx under-estimates at high bias.

What nonlocal robustly fixes:
- zero rate where there is no final state;
- a gap that varies along the path;
- separated carriers (tunnelling FETs, GIDL).

*Approach.*
- **Model:** the Sentaurus dynamic-path model, direct process (Eqs. 480-484, read). With an infinite transverse cutoff and m₀ → ∞ it reduces exactly to A F² exp(−B/F) in a uniform field (re-derived by P1).
- **Integrals:** closed-form segment integrals, so there is no quadrature error.
- **Placement:** holes at the start node; electrons spread over the end element.
- **Jacobian:** exact in 1-D. In 2-D with a dynamic direction it is approximate, so ssac and noise refuse.

*Gates:*
- uniform-field identity to 1e-12; finite m₀ must fail it;
- no final state: a 0.8 V step over 2 nm gives exactly 0, where the local model gives more than 1e3x; generation turns on at 1.13 V;
- an independent spline/brentq/quad oracle with convergence order ≥ 1.7;
- FDT to 1e-4, which placing both carriers at one node must fail;
- nightly: digitised Jiao 2013 data, at about the factor-1.5 level.

*Risks:*
- Paths cut by boundaries switch on and off non-smoothly; a 5 % taper is applied and gated.
- 2-D needs numba.

*Effort:* 1-D 8-10 d; 2-D 8-12 d, later (decision 21). *Read:* P1 §3.

### 4.2 Materials and API (P2)

**4.2.1 Si bundles, house BGN, n_i pairing, statistics switch**

*What and why.* An apparent band-gap narrowing (BGN) is fitted together with one intrinsic density n_i and one carrier statistics. Mixing sets moves n_ie² by about 2x without any warning.

*House model: Yan & Cuevas 2013* (read and re-checked). It is a real BGN with Fermi-Dirac (FD) statistics, ΔE = 0.042 meV·[ln(N/1e14)]³, with n_i = 9.65e9, N_c = 2.86e19 and N_v = 3.1e19 cm⁻³. Reasons:
- It is the only set fitted with FD statistics, which both builders run, so it removes the double-count.
- Its n_i is the modern consensus.
- It is primary-verified.

The p-type constants are in the 2014 paper. Until it is read, p-type BGN nodes are refused.

*Pairing check:*
- a mismatch is a hard error at 1e-3 in n_i;
- the message gives the gap equivalent, e.g. "1.84 meV" for 1.0e10 against 9.65e9;
- an explicit override is available;
- legacy presets stay unpaired.

*Bundle details:*
- A statistics switch, whose unset value keeps today's FD statistics byte-identically.
- The bundles use the statistical gap kT ln(N_c N_v/n_i²) = 1.12973 eV. The optical 1.124 eV would fail the heterojunction check.
- Mobilities are required arguments.

*Rejected:* Klaassen 1992 with Boltzmann statistics; its n_i is unverified.

*Gates:*
- FD + real BGN against Boltzmann + apparent BGN, both computed independently (scipy F₁/₂), must agree on n₀p₀ within 0.3, the papers' quoted uncertainty;
- a long-diode J0 against q D_p p₀/L_p to 2e-3, plus the measured FD-fit error;
- mismatches raise, from one check site.

*Effort:* 4-5 d, +1 when the 2014 paper arrives. *Read:* P2 items 1-2.

**4.2.2 III-V presets**

A radiative coefficient B is either internal or effective (after photon recycling); for GaAs the two differ by about 4x.

*Ship:*
- GaAs B_int = 3.3e-10 cm³/s (Niemeyer 2019, read);
- GaAs Auger C_n = C_p = 3.5e-30 cm⁶/s (an equal split of 7 ± 4e-30, from a secondary source);
- InP B_int = 1.2e-10;
- a recycling factor, applied once.

*Defer:* InGaAs, GaN, Ge.

*Gate:* B_int(1 − P_r) = 0.792e-10 must match the measured B_eff of 0.78 ± 0.07e-10. Swapped labels miss by 4.2x.

*Effort:* 1-1.5 d. *Read:* P2 item 3.

**4.2.3 Incomplete ionization**

The units guard and the advisory are covered by row 3.2-11. The Altermatt-2006 model (1.5-2 d) is recommended for later, because its table was seen only via a companion paper. *Read:* P2 item 4.

**4.2.4 Per-material Hooge α**

α_H is not a constant: about 1e-4 in perfect lattices, and 1e-6 to 1e-3 measured (Hooge 1994). The old 2e-3 is a "poor samples" average.

*Approach:*
- an optional per-carrier α on each material, with no defaults;
- a region-to-α mapping in the noise analysis; a plain float stays bit-identical;
- **[C]** built on the new per-element source.

*Gate:* a two-material resistor gives S·f = I²Σα_r/N_r to 1e-4.

*Effort:* 1 d. *Read:* P2 item 5.

**4.2.5 One-call noise spectrum on the 2-D builder**

The hand recipe has three traps: it bypasses extended precision, its circuits are process-global, and its results are per unit depth.

*Approach:*
- circuit-drive the gate once;
- re-settle through the library solver;
- scale results to per cell by the lateral period, which is exact for a laterally invariant device with white sources;
- dielectric gates only in v1.

**[C]** The method's default mechanism list must be inherited from the noise module. Otherwise the thermionic shot noise, which is default-on under decision 43, is silently dropped.

*Gates:*
- bit-identical to the hand recipe;
- FDT to 1e-9;
- the sequence 0 V, +1 V, 0 V gives a third result bitwise equal to the first.

*Effort:* 1.5 d. *Read:* P2 item 6.

**4.2.6 Extended precision from the design**

*Approach:* a carrier-settings block on the design, off by default. An explicit builder argument wins, with one advisory if the two disagree.

*Rejected:*
- on the mesh spec: the 3-D builder would read a "2-D mesh" setting;
- per material: DEVSIM's extended solver is global.

*Gates:*
- fingerprints unchanged when the setting is off;
- the pipeline enables it exactly once;
- nightly: a GaN-like diode matches the Sah-Noyce-Shockley current to 3e-3, while double precision is off by more than 1e3.

*Effort:* 1-1.5 d. *Read:* P2 item 7.

### 4.3 Traps, noise, heterointerfaces (P3)

**4.3.1 Thermionic-interface shot noise (first noise feature)**

*What and why.* A thermionic heterointerface contributes no noise today; the FDT xfail misses by −0.69 at 1 Hz. Each direction of the Richardson current is an independent Poisson stream, so S = 2q(J₀₁ + J₁₀)/A. That is 4kT·G at zero bias and tends to 2qJ in forward bias (the coth law, already an oracle).

*Approach* (P3's design on top of 4.0.1; see 5.1):
- a pair model written from the same strings as the flux;
- the two region copies of each interface node paired by coordinate, asserted one-to-one;
- each pair weighted by the area DEVSIM's flux assembly actually uses, measured first and never assumed;
- a flux without a pair model raises.

*Gates:*
- FDT on the xfail fixture to 1e-4;
- a sweep of the Richardson constant A* from 1e-6 to 1e2: the FDT holds throughout, and the result tends to the continuous model;
- isotype n-GaAs/N-AlGaAs in forward bias: S = 2q(J + 2J_s) to 3e-3;
- on a non-uniform lateral mesh, equal area shares must break the FDT.

*Effort:* 4-5 d. *Read:* P3 §2.

**4.3.2 Kinetic traps: conductance method and trap g-r noise**

*What and why.* Today the occupancy follows n and p instantly. That is exact at DC and in low-frequency C-V. It is not charge-conserving in transient or ssac with charged traps, because the capture current is missing. The charged-trap FDT xfail misses by 6.6e-2 at 10 MHz.

Kinetic occupancy enables:
- the Nicollian-Goetzberger (N&G) conductance method;
- DLTS-like transients;
- trap g-r noise.

*Approach:*
- **Unknown:** one per level or interface bin: the trap's own quasi-Fermi level in kT, u = ln(f/(1−f)). This keeps 0 < f < 1, has no 1−f cancellation, and seeds exactly from the steady state. The \|u\| < 700 limit refuses deep levels below about 20 K.
- **Rates:** from the existing kernel, so the DC limit is the shipped model.
- **DEVSIM support:** P3 measured that such an extra node equation enters contacts, ssac and the native noise solve without special handling.
- **Interface traps** need a masked region-wide unknown, because DEVSIM has no interface unknowns. That is about 13.5k unknowns in 1-D, but 3.4 million on the 76k-node metasurface. Hence a budget refusal at about 2e5 (to be calibrated) and a coarser-bin option.

*Gates:*
- DC identity with the quasi-static model to 1e-10;
- a DLTS transient against the closed-form exponential to 1e-4, where halving dt divides the error by 4;
- charge conservation, where the old model must miss by more than 10x;
- the conductance peak at ωτ = 1 with height C_s/2 (N&G Eqs. 15-17, read from the scan) to 1e-3, which the old model misses by 100 %;
- the continuum peak at ωτ_m = 1.98;
- the FDT xfail flips to 1e-4, and dropping the hole-process source must break it.

*Risks:*
- DEVSIM's chain rule (see 4.0.2).
- Interface kinetics may end up 1-D-only.
- The hole-side conductance formula and the g-r variance are derived, not cited; the FDT is the independent check.

*Effort:* 12-16 d. McWhorter adds 4-6 d, later. *Read:* P3 §1.

**4.3.3 Band temperature model**

Users supply 0 K or 300 K band data and simulate at another T.

*Ingredients:*
- **Gap:** a Varshni E_g(T) from Vurgaftman 2001 (read): GaAs 1.519 eV / 0.5405 meV/K / 204 K; AlAs 3.099 / 0.885 / 530; AlGaAs bowing independent of temperature.
- **Affinity:** χ(T) moves by a required fraction γ_χ of the gap change. There is no standard: 1 is valence-anchored (Vurgaftman); ½ is the Sentaurus convention (secondary source).
- **n_i:** optionally derived.

**[C]** Apply Varshni as a shift anchored at the material's reference temperature (see 5.3). Otherwise P3's Si constants (1.17 eV, 4.73e-4 eV/K, 636 K) give 1.1245 eV at 300 K. That is 5.2 meV off P2's statistical gap, with a derived n_i of 1.067e10 instead of 9.65e9, and every house-bundle device would be refused.

*Gates:*
- Varshni values to 1e-12 eV;
- bitwise identity at the reference T;
- V_bi(T) and ΔE_c(T) of N-AlGaAs/p-GaAs against an independent band diagram, to 2e-4 V and 1e-6 eV (the margin is thin; P3 has a fallback);
- the thermionic conductance with an independently computed barrier, to 2e-3.

*Effort:* 3-4 d. *Read:* P3 §3.

### 4.4 Quantum and numerics (P4)

**4.4.1 Density-scaled absolute tolerance**

DEVSIM's Newton requires both the absolute and the relative criteria (P4 read the source). The absolute error is in practice the largest density update. The default 1e10 m⁻³ falls below the attainable round-off floor once densities exceed about 1e24 m⁻³, and Newton then stalls for 10-18 iterations.

Because the two criteria are combined with AND, raising the absolute tolerance to just above the floor cannot loosen the delivered accuracy.

*New default:* max(1e10, 1024·u·S), where S is DEVSIM's aggregate solution size and u is the unit round-off. Devices below 8.8e22 m⁻³, and all extended-precision devices, are byte-identical.

*Gates:*
- degenerate devices converge within 2 iterations of the extended run;
- results stay within rel·\|x\| + abs of the extended reference, including a 1e8 minority density next to a 1e26 majority.

*Effort:* 1.5-2 d. *Read:* P4 §5.

**4.4.2 Floating-body (ill-posed DC) detection**

A region where no contact pins any carrier has an exactly singular DC Jacobian. Its solution is then set by round-off and depends on the BLAS build. The R26 fixture was such a case and went unnoticed.

*Design:*
- a static topology check proposes candidate null vectors;
- one matrix assembly confirms them numerically;
- an error is raised only when both agree;
- legitimate floating cases can be acknowledged explicitly;
- Newton failure messages get a "near-singular" diagnosis.

**[C]** The candidates must include the kinetic-trap rows. Otherwise a floating body with kinetic traps is missed.

*Gates:*
- the old R26 fixture raises at r ≤ 1e-15;
- a recombination-free slab shows rank deficiency 2;
- no false positives across the whole DEVSIM test set and all 33 validations;
- ≤ 5 % cost.

*Effort:* 3.5-4.5 d. *Read:* P4 §4.

**4.4.3 Strong-inversion density gradient**

*What and why.* A 1-D MOS capacitor fails beyond 0.70 V of band bending with density gradient, and beyond 0.78 V classically. P4 measured the cause: one near-null mode in the minority-electron block. Its condition number grows from 8.6e10 at 0.30 V to 1.7e17 at 0.76 V, because the inversion layer couples to the body only about 1e-17 weakly. Extended precision buys one 20 mV step. Quasi-Fermi or Slotboom variables were measured not to help.

*Approach.* A capacitor's DC state is equilibrium. So replace the minority continuity row with the algebraic statement "its quasi-Fermi level equals the body's", using the existing FD activity.
- The condition number falls to about 1.4e4.
- The zero-current Scharfetter-Gummel (SG) relation gives exactly the same ratio on every edge, so the closed model gives the same discrete solution as the full model wherever the latter converges.
- ssac is allowed: it is the low-frequency C-V.
- Noise is refused, and two-contact devices need an explicit acknowledgement.

*Gates:*
- the smallest singular value rises by ≥ 1e8 (measured 3e11);
- the analytic MOS surface charge is reproduced at order ≥ 1.8;
- the result equals the full model where both converge, to 1e-10;
- the solve reaches 1.3 V in ≤ 12 iterations in double precision;
- a new Schrödinger-Poisson oracle: N_inv within 10 % and centroid within 15 % at γ_wall. These tolerances are registered beforehand; a miss is a model result, not something to retune.

*Effort:* 6-9 d. Needs the SG merge. *Read:* P4 §1.

**4.4.4 Retire the unipolar hard wall**

The old wall pins the carrier and its quantum potential. It acts as a sink and converges to a spurious branch, with a deficit of 3-1000 N₀L_q against the correct 1.187.

*Fix:* use the conserving insulator wall instead, with a shim in v0.15 and removal in v0.16.

*Gates:*
- the closed-form deficit 1.187207 to 2e-4;
- the wall flux ≤ 1e-12.

*Effort:* 1.5-2 d. Needs the SG merge. *Read:* P4 §2.

**4.4.5 2-D impact ionization on elements**

This is row 3.2-12.

*Gates:*
- an exact discrete uniform field at 0°, 30° and 45°, to 1e-12; the old form must equal the componentwise closed form;
- a harmonic potential against 400×400 Gauss-Legendre: order ≥ 1.8, with the old form ≥ 10 % off;
- 2-D against 3-D to 1e-6.

*Effort:* 2-3 d. *Read:* P4 §3.

---

## 5. Conflicts and overlaps, resolved

### 5.1 Thermionic shot noise: P1's shortcut against P3's design

P1 (decision UX-1) says its two-point source closes the thermionic xfail "almost for free". P3 designs a full solution: a pair model, node-copy pairing, a measured area weight, and a refusal.

**Resolution: both, as two layers.**
- The source arithmetic is P1's general source. P3's formula is exactly a two-point source with coefficients (+1, −1).
- The physics and bookkeeping are P3's.

P1 understates the work: the pairing and the area are the substance (4-5 d). The thermionic flux is expressible as a DEVSIM model, so it must **not** go through the custom-equation callback; P1's own rule refuses local terms there.

### 5.2 Temperature across the plans

| Plan | Treatment |
|---|---|
| P1 | BTBT at 300 K; opt-in scaling of B through its own callable, reusing the SOA Varshni table |
| P2 | Refuse paired BGN sets at T ≠ 300 K unless overridden with the user's own n_i(T) |
| P3 | Fix the 3-D DOS and trap bins; refuse heterostacks without a temperature for χ; Varshni E_g(T), χ(T), optional derived n_i(T); warn on BTBT under T_K |

**Resolution:**
1. The P3 fixes, the BTBT warning and the χ guard go into v0.14 (rows 3.2-7 to 3.2-10).
2. **One Varshni function and one table.** BTBT scaling reads E_g(T)/E_g(300) from the region's band-temperature model when present, and otherwise keeps the warning. This replaces P1's separate callable.
3. **P2's refusal stands.** A BGN fit is a 300 K statement even when n_i(T) is derived. P3's derived n_i(T) is documented as the recommended "own n_i(T)" when overriding.
4. **Anchored Varshni** (5.3).

### 5.3 Si band gap: P2's statistical gap against P3's Varshni constants [C]

| Quantity | Value at 300 K |
|---|---|
| P3's Si Varshni gap | 1.1245 eV |
| P2's bundle gap, kT ln(N_c N_v/n_i²) | 1.12973 eV |
| Difference | 5.2 meV (the heterojunction gap check allows 5 meV) |
| n_i derived from P3's gap | 1.067e10 |
| n_i paired in P2's bundle | 9.65e9 (the pairing check allows 1e-3) |

**Resolution: apply Varshni as a change anchored at the material's reference temperature.**
- E_g(T) = E_g,ref + V(T) − V(T_ref), and likewise for χ.
- n_i is derived consistently with that.
- At T_ref the result is bit-identical to the inputs. Extend P3's identity gate from χ to E_g and n_i.
- The absolute form stays available for data quoted at 0 K.

This is decision 32.

### 5.4 `chi_eV=None` and the χ(T) guard; P1's dielectric affinity

On the defaults branch, a heterostack with χ = None refuses to build. With P3's guard the order of checks is:
1. χ is None: refused at any T.
2. χ is given, T_K ≠ 300 K, and no temperature is stated for χ: refused.
3. Otherwise the device builds.

P2's Si bundles set χ = 4.05 eV explicitly. **They should also state 300 K as the temperature of that χ.**

**[C] P1's dielectric affinity hides a dependence.** P1 gives SiO2 a default affinity of 0.95 eV, computed as Si's 4.05 eV minus the fitted 3.1 eV barrier. As a result:
- with Si χ = None (the new default), the oxide barrier is undefined;
- with a user Si χ of 4.17 eV, the barrier silently becomes 3.22 eV instead of the fitted 3.1 eV.

*Recommendation (decision 13):*
- specify a dielectric by its fitted barrier height against a named semiconductor;
- derive its affinity through the Anderson rule only when the semiconductor χ is given;
- refuse when the semiconductor χ is None.

### 5.5 Ordering against the two-Bernoulli switch; stale digest references

The SG branch re-recorded the call-log digests (237 / 139 calls) and replaced array byte-identity with a numeric cross-version gate. That makes some statements in the plans stale:
- P1 and P2 cite "the 16-array SHA a500ca93" and say digests are "rerun, not re-recorded". Both are stale after the merge.
- **[C]** P3 says the digest "counts calls, not floats, so stays valid". That is half right: it hashes the definition strings, which the SG change moved. After re-recording it is a valid structural guard again.

**Resolution:**
- All v0.15 branches start from the post-SG tip.
- All plans use the three-tier rule of section 2.
- P4's density-gradient items (4.4.3, 4.4.4) **hard-depend** on the SG merge.

### 5.6 Fermi-Dirac activity: P4 would re-implement existing code [C]

P4's step 1 plans to:
- "probe DEVSIM atan";
- write a DEVSIM twin of the FD potential shift;
- refuse FD if atan is missing (its decision D1-c).

The heterojunction module already installs ln u + Δ(u) in closed form. It builds arctan from half-angle reductions plus a Taylor tail (DEVSIM indeed lacks atan) and supplies the exact derivative g(u)/c. That derivative is exactly what the quasi-equilibrium row needs.

**Resolution:** reuse it, as P1 does. FD is then supported and D1-c is moot.

### 5.7 Noise-module edit order and mechanism names

Seven changes touch the noise analysis. Merge each before the next starts:

| Order | Change | Section | Plan |
|---|---|---|---|
| 1 | Per-element Hooge | — | v0.14 |
| 2 | Two-point source (bit-identity gated) | 4.0.1 | |
| 3 | Thermionic source | 4.3.1 | P3 |
| 4 | Hooge mapping and noise method | 4.2.4 / 4.2.5 | P2 |
| 5 | Kinetic-trap sources | 4.3.2 | P3 |
| 6 | Oxide-tunnel shot noise | 4.1.1 | P1 |
| 7 | Nonlocal BTBT source | 4.1.3 | P1 |

**Mechanism names.** One registry:
- existing: diffusion_n/p, gr, hooge_n/p;
- added: te_n/p, trap_n/p;
- **[C]** proposed "tunnel_ox" and "btbt_nl" (P1 names none).

**Two fixes follow from this:**
- The noise method takes its default mechanism list from the module.
- The well-posedness candidates include the trap rows.

### 5.8 CI time

**[C]** P1, P3 and P4 budget against the 150-min `test` job (coverage leg 65-95 min). DEVSIM is not installed there, so those tests skip. The binding budgets are:
- **the solvers leg:** 45-min cap, about 10 min today before the 12 modules are added;
- **the nightly slow step;**
- **the nightly validation tier:** 4 shards, 42-89 min each.

Only P2 used the right leg.

| Source | New fast tests (plans' estimates) | Nightly additions |
|---|---|---|
| v0.14 wiring (12 modules) | to measure; about 2-6 min at -n 2 | heterojunction slow gate |
| P1 | 2-3 min | 2-D/3-D tunnelling, nonlocal 2-D, published-data validations |
| P2 | about 2.5 min | GaN-like diode, 3-D noise cross-check, Yan-Cuevas J0 |
| P3 | ≤ 6 min | trap kinetics 10-15 min, noise gate H 2-3 min, slow gates |
| P4 | about 2.2 min | about 28 min (SP inversion 20, impact ionization 5, tolerance floor 2, wall 1) |
| **Total** | **13-20 min single-process, about 7-10 min at -n 2** | **about 45-60 min** |

**Conclusions:**
- The solvers leg should stay under 45 min (about 10 + 5 + 10 min). Measure at each merge.
- **The validation tier needs a fifth shard, or a rebalance by measured time, before the v0.15 validations land.** The largest shard is already 89 min.

### 5.9 Other overlaps

| Overlap | Resolution |
|---|---|
| Term inventory (P1's "custom" kind, P3's single-carrier terms) | One owner (4.0.2). P1's multi-process BTBT threading uses it afterwards. |
| Exact-mesh check (impact ionization, BTBT, Hooge) | One helper (4.0.4). |
| The well-posedness diagnosis is reused by the strong-inversion gate | Build 4.4.2 first. |
| P3's custom-equation alternative for interface unknowns | Would use 4.0.3; stays a research spike. |
| Extended precision with custom terms | Converges linearly. Under 4.4.1 an extended device keeps 1e10, so only the iteration count rises. Document it in the design-flag docstring. |

### 5.10 Contradictions and weak claims (index)

Each item is resolved in the section shown:

| Plan | Claim | Finding | Section |
|---|---|---|---|
| P2 | Seven DEVSIM modules missing from CI | It is 12 | 3.2-1 |
| P2 | Hooge noise should be refused | Superseded by the per-element fix | 3.1 |
| P4 | Fermi-Dirac activity needs re-implementing | It already exists | 5.6 |
| P1, P2, P3 | Digest references are current | Stale after the SG change | 5.5 |
| P1, P3, P4 | CI budget is the `test` job | Wrong job | 5.8 |
| P1 | SiO2 affinity default | Tied to Si's affinity | 5.4 |
| P3 | Si Varshni parameters | Conflict with P2's n_i pairing | 5.3 |
| P1, P3 | Varshni implementation | Two kernels proposed | 4.0.4 |
| P1 | Thermionic noise "almost free" | Understated | 5.1 |
| P1 | InGaAs row suspect | Supported | 3.2-14 |
| P1 | Vurgaftman needed from the user | Already in hand | 8.1 |
| P2 | Sb value "wrong" | Outdated, not mistranscribed | 3.2-5 |
| P4 | 2-D impact-ionization defect | Older than P4 implied: before v0.13.0 | 3.2-12 |
| P3 | H2-c margin | May be under 5x | P3's fallback |
| P4 | Floating-body candidates | Miss trap rows | 4.4.2 |

---

## 6. Sequencing and effort

### 6.1 Schedule (at most 5 agents)

**Wave 0, v0.14.0.** Three slots are busy: defaults, SG, blockers.
- **Slot 4:** P2 corrections and the ionization guard (rows 3.2-4/5/6/11; 1-1.5 d).
- **Slot 5:** temperature fixes (rows 3.2-7 to 3.2-10; 1.5-2 d).
- **Next free slot:** 2-D impact ionization plus the exact-mesh check (3.2-12/13; 2.5-3.5 d), and the InGaAs row if chosen.
- **Close-out:**
  - merge in the order defaults, SG, blockers, fixes;
  - full DEVSIM suite and validations;
  - measure CI times;
  - release notes and documentation rows.

**Wave 1, v0.15 foundations** (from the post-SG tip).

| Slot | Work | Days |
|---|---|---|
| A | Two-point source, then custom-equation dispatcher (4.0.1, 4.0.3) | 3-4.5 |
| B | Term bookkeeping v2 (4.0.2) | 2-3 |
| C | Tolerance, then well-posedness (4.4.1, 4.4.2) | 5-6.5 |
| D | Si bundles and pairing (4.2.1) | 4-5 |
| E | Extended precision on the design, then III-V (4.2.6, 4.2.2) | 2-3 |

**Wave 2.**

| Slot | Work | Days | After |
|---|---|---|---|
| A | Thermionic shot noise (4.3.1) | 4-5 | wave 1 A |
| B | Kinetic traps K1-K2 (4.3.2) | 8-11 | 4.0.2 |
| C | Oxide tunnelling (4.1.1), **critical path** | 12-16 | 4.0.1 and 4.0.3 |
| D | Hard wall, then strong inversion (4.4.4, 4.4.3) | 7.5-11 | SG and 4.4.2 |
| E | Band temperature (4.3.3) | 3-4 | — |

**Wave 3.**

| Work | Days | After |
|---|---|---|
| Hooge mapping and noise method (4.2.4, 4.2.5) | 2.5 | 4.3.1 |
| Trap noise K3 and close-out K4 | 4-5 | 4.2.5 |
| Nonlocal BTBT, 1-D (4.1.3) | 8-10 | 4.0.3 |
| BTBT presets and temperature scaling (4.1.2) | 2-3 | 4.3.3, once Kao is read |

**Close-out:**
- full suite;
- the new nightly shard;
- removal of the gmsh mesher;
- audit notes and release notes.

**Wall time** is about 20-25 working days. The critical path is wave-1 A, then oxide tunnelling, then close-out. The trap chain (about 15-19 d) runs alongside it. This is an estimate.

### 6.2 Effort per item

| Item | Release | Agent-days |
|---|---|---|
| v0.14 fixes, rows 3.2-1 and 3.2-4 to 3.2-11 | v0.14.0 | 3.5-4.5 |
| 2-D impact ionization and exact-mesh check | v0.14.0 (rec.) / v0.14.1 | 2.5-3.5 |
| InGaAs SOA row | v0.14.0 / v0.14.1 | 0.5-1 |
| Shared infrastructure 4.0.1-4.0.3 | v0.15 | 5-7.5 |
| Oxide tunnelling | v0.15 | 12-16 |
| BTBT presets | v0.15 | 2-3 |
| Nonlocal BTBT, 1-D | v0.15 | 8-10 |
| Si bundles and pairing | v0.15 | 4-5 (+1) |
| III-V presets | v0.15 | 1-1.5 |
| Hooge mapping and noise method | v0.15 | 2.5 |
| Extended precision on the design | v0.15 | 1-1.5 |
| Thermionic shot noise | v0.15 | 4-5 |
| Kinetic traps K1-K4 | v0.15 | 12-16 |
| Band temperature model | v0.15 | 3-4 |
| Tolerance and well-posedness | v0.15 | 5-6.5 |
| Hard wall and strong inversion | v0.15 | 7.5-11 |
| ~~gmsh mesher removal~~ cancelled 2026-11-06 (kept, geometry-corrected) | -- | 0 |
| **v0.15 total** | | **about 67-90** |
| Nonlocal 2-D (8-12), McWhorter (4-6), Altermatt 2006 (1.5-2); ohmic-contact noise drive, hole/poly tunnelling, DC charge constraint (not estimated) | later | 14-20 + unestimated |

**Why "later":**
- Nonlocal 2-D: its premise is weaker than first stated, so review the 1-D results first.
- McWhorter: no primary source has been read.
- Altermatt 2006: its table has been seen only via a companion paper.
- The rest need cited inputs or a mechanism that is not yet designed.

---

## 7. Decisions for the user

Original IDs are given in brackets:
- U… are from P1;
- D1-D13 are from P2;
- D-K, D-T and D-H are from P3;
- D1-a to D5-c are from P4;
- [C] marks decisions raised by this consolidation.

### A. Before v0.14.0

| # | Question | Options | Recommendation | Effect | Blocks |
|---|---|---|---|---|---|
| 1 | Which section-3 findings hold the tag? [D4, D13, D-H1, C] | All of 3.2-1 to 3.2-11 / a subset | **All** | About 4 d. Moves del Alamo results only in p-type or compensated regions; Sb freeze-out; 3-D results only at T ≠ 300 K | the tag |
| 2 | Switch 2-D impact ionization to the element form now? [D3-a] | v0.14.0 / v0.14.1 / v0.15 | **v0.14.0** | 2-D substrate current and HCI lifetimes change where current is not mesh-aligned | wave 0 |
| 3 | Refuse heterostacks at T ≠ 300 K unless χ's temperature is stated, alongside `chi_eV=None`? [D-H2, C] | Refuse in v0.14.0 / advise now, refuse in v0.15 / v0.15 only | **Refuse in v0.14.0** | Existing temperature gates gain a one-line tag | the tag |
| 4 | File the drafted DEVSIM upstream issue? [C] | File / hold | **File** (verified, minimal reproducer) | — | — |
| 5 | Fix the InGaAs temperature law in the SOA table? [U2-3, C] | v0.14.0 / v0.14.1 | **v0.14.1**, with measured moved numbers | SOA InGaAs results at 300 K | — |
| 6 | Exact-mesh refusal for element-based quantities? [D3-c, C] | Refuse / warn / none | **Refuse**, with #2 | User gmsh meshes raise | with #2 |

### B. Shared design

| # | Question | Options | Recommendation | Blocks |
|---|---|---|---|---|
| 7 | Build the two-point source first and use it for all new noise? [UX-1] | Yes / per-feature | **Yes** | 4.0.1 |
| 8 | Accept linear extended-precision convergence with custom terms? [UX-2] | Accept and document / restrict | **Accept** | 4.0.3 |
| 9 | Separate mechanism keys (te, trap, tunnel_ox, btbt_nl)? [D-K4, C] | Separate / fold into "gr" | **Separate** | 4.0.1 |

### C. Tunnelling

| # | Question | Options | Recommendation | Blocks |
|---|---|---|---|---|
| 10 | Supply model [U1-1] | Two-sided Tsu-Esaki / scaled FN or S-H | **Tsu-Esaki; refuse FN and S-H** | 4.1.1 |
| 11 | Gate terminal current [U1-2] | Module-reported I_gate, circuit row only when circuit-driven / auto-convert to circuit | **First** | 4.1.1 |
| 12 | Gate work function [U1-3] | Read from the contact offset, optional field / separate parameter | **Read from contact** | 4.1.1 |
| 13 | Dielectric input [C] | Barrier against a named semiconductor / affinity with a default | **Barrier; refuse when the semiconductor χ is None** | 4.1.1 |
| 14 | Carriers [U1-4] | Electrons with a warning / also holes | **Electrons** | 4.1.1 |
| 15 | Image-force lowering [U1-5] | Out of scope / add | **Out** (would double-count the FN fit) | — |
| 16 | Published or team high-k stack J-V? [U1-6] | Supply / none | **Your call**: none means no stack validation | O13 |
| 17 | Poly-Si gates [U1-7] | Refuse / support | **Refuse** | — |
| 18 | BTBT sets to ship [U2-1] | Kao Si/Ge after reading; no vendor sets; InGaAs computed / other | **As stated** | 4.1.2 |
| 19 | Ge-direct A, +10 % vs Kane [U2-2] | As printed with a note / hold | **As printed**; re-check against the primary | 4.1.2 |
| 20 | B temperature scaling [U2-3] | Opt-in via the one Varshni / none | **Opt-in** | 4.1.2 |
| 21 | Nonlocal scope [U3-1] | 1-D now, 2-D after review / 2-D now / drop | **1-D now** | 4.1.3 |
| 22 | Phonon-assisted form [U3-2] | Sentaurus Eq. 490-491 if the transcription verifies / our average-field form | **First, else second, labelled** | 4.1.3 |
| 23 | numba for 2-D nonlocal [U3-3] | Required / pure Python | **Required**; 3-D deferred | later |
| 24 | Path reflection at interfaces [U3-4] | Defer / implement | **Defer** | later |
| 25 | Approximate Jacobian with dynamic direction [U3-5] | DC only, ssac and noise refuse / forbid | **DC only** | 2-D |

### D. Materials and API

| # | Question | Options | Recommendation | Blocks |
|---|---|---|---|---|
| 26 | House Si BGN [D1] | Yan-Cuevas FD + 9.65e9 (n-type until 2014) / Klaassen + Boltzmann | **Yan-Cuevas** | 4.2.1 |
| 27 | Pairing mismatch [D2] | Hard error + override / advisory | **Hard error** | 4.2.1 |
| 28 | Statistics switch [D3] | Add / no | **Add** | 4.2.1 |
| 29 | Paired presets at T ≠ 300 K [D5] | Refuse unless overridden / allow | **Refuse** | 4.2.1 |
| 30 | III-V presets and labels [D6] | GaAs B_int, GaAs Auger (equal split), InP B_int, recycling factor; defer InGaAs, GaN, Ge | **As stated** | 4.2.2 |
| 31 | Ionization [D7] | No default α; guard and advisory now; Altermatt later | **As stated** | 3.2-11 |
| 32 | Varshni anchoring [C] | Anchored at T_ref / absolute E_g(0) | **Anchored**; absolute for 0 K data | 4.3.3 |
| 33 | Hooge per material [D8] | Per-carrier fields, no defaults; region mapping | **As stated**; cite 1e-4, not 2e-3 | 4.2.4 |
| 34 | Hooge cross terms [D9] | — | **Resolved** (per-element fix) | — |
| 35 | Noise method scope [D10/D10b] | Dielectric gates, per cell / also ohmic | **Dielectric**; ohmic later | 4.2.5 |
| 36 | Stale solve [D11] | — | **Resolved** | — |
| 37 | Extended-precision placement [D12] | Carrier block on the design / mesh spec / per material | **Carrier block** | 4.2.6 |
| 38 | CI wiring before the tag [D13] | — | **Yes** (in flight) | the tag |

### E. Traps, noise and heterointerfaces

| # | Question | Options | Recommendation | Blocks |
|---|---|---|---|---|
| 39 | Charged-trap default [D-K1] | Keep quasi-static and add a kinetic twin / kinetic default / as first, plus an advisory when quasi-static charged traps are used in ssac, noise or transient | **The last** | 4.3.2 |
| 40 | Occupancy unknown [D-K2] | Logit / f with damping | **Logit** | 4.3.2 |
| 41 | Interface kinetics in 2-D/3-D [D-K3] | Masked unknowns with a budget / custom-equation spike | **Masked**; spike only if needed | 4.3.2 |
| 42 | McWhorter [D-K5] | Out of scope / 1-D, amplitude ungated | **Out** for v0.15 | later |
| 43 | Thermionic shot noise on by default [D-T1] | On / opt-in | **On**, release-noted (the old number fails the FDT) | 4.3.1 |
| 44 | Interface flux with no pair model [D-T2] | Refuse / ignore | **Refuse** | 4.3.1 |
| 45 | 3-D DOS, reported T, trap bins [D-H1] | Fix / document | **Fix in v0.14.0** | the tag |
| 46 | χ(T) convention [D-H3] | Required fraction / default 1 / default ½ | **Required, no default** | 4.3.3 |
| 47 | BTBT under T_K [D-H4] | Warn now / refuse | **Warn** | the tag |

### F. Quantum and numerics

| # | Question | Options | Recommendation | Blocks |
|---|---|---|---|---|
| 48 | Quasi-equilibrium closure as the strong-inversion route [D1-a] | Yes / pursue quasi-Fermi variables | **Yes** (they were measured not to help) | 4.4.3 |
| 49 | Its home in the drift-diffusion module [D1-b] | Yes / DG module | **Yes** | 4.4.3 |
| 50 | FD without atan [D1-c] | — | **Moot** (5.6) | — |
| 51 | Anisotropic Si SP [D1-d] | Report / gate | **Report** | — |
| 52 | 2-D MOSFET DG transport [D1-e] | Follow-up | **Later** | later |
| 53 | Hard wall [D2-a/b/c] | Shim in v0.15, removal in v0.16, wall-aware seed, keep the name / remove now | **Shim** | 4.4.4 |
| 54 | Keep the edge form of impact ionization as an explicit option [D3-b] | Yes / remove | **Yes** | with #2 |
| 55 | Test-suite hook on raw solves [D4-a] | Yes with an opt-out marker / no | **Yes** | 4.4.2 |
| 56 | Exact singularity = error, near-singularity = diagnosis [D4-b] | Yes / warnings | **Yes**. This also answers the earlier "floating-body warning?" question. | 4.4.2 |
| 57 | DC charge constraint for isolated wells [D4-c] | Defer / build | **Defer** | later |
| 58 | Density-scaled default tolerance [D5-a] | Yes / no | **Yes**: only S > 8.8e22 m⁻³ changes, within rel_tol | 4.4.1 |
| 59 | Migrate the builders' tolerance rules [D5-b] | Separate commit / leave | **Separate commit** | after 4.4.1 |
| 60 | Hard-coded 1e10 in validations [D5-c] | Leave DG; new fixtures use the default | **As stated** | — |

**Routine.** These follow the plans' recommendations without controversy and can be accepted as a block unless you object: 8-9, 11-12, 14-15, 17, 22-25, 27-29, 31, 33, 35, 37, 40-41, 44, 46-47, 49, 51, 53-57 and 59-60.

**Need your judgement:**
- 1-5 (v0.14 scope);
- 13 (dielectric input);
- 16 (stack J-V data);
- 21 (nonlocal scope);
- 26 (house BGN);
- 30 (III-V labels);
- 32 (Varshni anchoring);
- 39 (charged-trap default);
- 43 (thermionic noise default);
- 48 (strong-inversion route);
- 58 (tolerance default).

---

## 8. Literature

### 8.1 Papers you could supply

| Paper | Unblocks |
|---|---|
| Kao et al., IEEE TED 59, 292 (2012), Table III | Si/Ge BTBT presets; the Ge-direct discrepancy (#18-19) |
| Hurkx, Klaassen, Knuvers, IEEE TED 39, 331 (1992) | Confirms the shipped Si BTBT set and the T-scaling attribution |
| Yan & Cuevas, JAP 116, 194505 (2014) | p-type house BGN (#26) |
| Klaassen et al., SSE 35, 125 (1992) | The Klaassen set's n_i (the alternative to #26) |
| Altermatt et al., JAP 100, 113714/113715 (2006) | The later ionization model |
| Sentaurus Device manual (licensed) | Phonon-assisted Eq. 491 (#22); the χ(T) default (#46); BGN and PB defaults |
| Bludau, Onton & Heinke (1974) or Thurmond (1975) | Si Varshni constants |
| van Vliet & Fassett (1965); Bonani & Ghione (2001) | Trap g-r variance citation |
| McWhorter (1957); Christensson (1968); Hung (1990) | Border-trap 1/f (later) |
| Wettstein (2001); Ancona (2002); Garcia-Loureiro (2011) | Density-gradient citations in the release notes |
| Published or team SiO2/high-k J-V | Stack validation (#16) |
| Smets et al., JAP 115, 184503 (2014); Ahrenkiel et al., APL 72, 3470 (1998) | InGaAs nonlocal calibration; InGaAs B and C |

**Not needed from you.** Already in hand: Vurgaftman 2001 (in the scratchpad `refs/`), Nicollian & Goetzberger 1967 (scan), Hooge 1994, Niemeyer 2019, Yan & Cuevas 2013.

### 8.2 Claims marked unverified in the plans

**P1.**
- Equation numbers: Tsu-Esaki.
- Derived rather than cited: the pair-noise formula; the image-force arithmetic.
- Not read: Kane 1961 itself; Thurmond.
- Secondary sources only: Hurkx values and its temperature scaling; all Kao values.
- Ge ε and n_i.
- The Sentaurus Eq. 491 transcription.
- Abstracts only: Pan & Chui; Wong 2020.

**P2.**
- Klaassen's n_i.
- The Yan & Cuevas 2014 constants.
- The Sentaurus BGN and Pearson-Bardeen defaults, and Pearson-Bardeen's own α.
- As and B Mott densities.
- Altermatt Table I (seen via a companion paper).
- Ahrenkiel (OCR only).
- GaN B.
- Si χ = 4.05 eV.
- GaAs n_i = 2.1e6 cm⁻³.
- GaAs Auger split.
- The del Alamo SSE 28, 47 location.
- The Hooge and stale-solve findings are now measured.

**P3.**
- Equation numbers: Shockley-Read, Hall, Lang, van der Ziel.
- Not accessed: Nicollian & Brews; McWhorter; Varshni 1967.
- Derived rather than cited: the hole-side conductance formula; the g-r variance.
- Abstract only: Gomila 2000.
- Si Varshni attribution.
- Sentaurus χ(T) (secondary).
- The low-T bin-error estimate.

**P4.**
- All density-gradient literature claims (abstracts only).
- The MOS F-function citation (re-derived in the test).
- Fang-Howard.
- Whether DEVSIM's Python interface exposes the relative-criterion floor.
- The predicted O(h²) of 2-D element impact ionization (gated).

**Consolidation.**
- The InGaAs 0.737 eV relies on Vurgaftman's alloy rule as read from the extracted text; re-check it when the row is fixed.
- Wall time and CI totals are sums of the plans' estimates, not measurements.
