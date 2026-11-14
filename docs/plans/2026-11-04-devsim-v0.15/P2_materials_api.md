# P2: material presets, BGN/n_i pairing, III-V recombination, Pearson-Bardeen, Hooge, noise wrapper, extended precision on Design

**What this is.** An implementation and verification plan for seven deferred items from DynaMeta's DEVSIM v0.14 work. It is a plan only: no code has been written and nothing in the repository was changed.

**Reference tree.** `DynaMeta_wt/dv_integ`, branch `feat/devsim-2.11-v0.14`, commit ee0ad04. pyproject still says 0.13.0 and there is no v0.14 tag, so **nothing from v0.14 has been released yet**. That matters for item 1: the presets that are already in the tree can still be corrected without breaking any user.

**Concurrent change.** Another agent is changing `TransportModel`: `chi_eV` will default to None, and `TrapSpec(enabled=True)` will raise. Every preset below therefore sets `chi_eV` explicitly, never uses `TrapSpec`, and lands after that change has merged into `transport_model.py`.

## How sources were checked

Three literature passes were run with web access. The extracted text and PDFs are kept under `scratchpad/lit_bgn`, `lit_iiiv` and `lit_pb_hooge`. Each claim carries one of these labels:

- **[P]**: primary text, abstract or equation seen.
- **[S: name]**: restated by the named secondary source.
- **[I]**: inferred by our own arithmetic from primary data.
- **[U]**: unverified.

I re-checked two load-bearing extractions myself in the scratch text:
- Yan & Cuevas 2013, Table I and its n_i input.
- del Alamo, TED 1987, Eq. (11).

**Paywalled or unreachable sources:**
- Slotboom & de Graaff, SSE 1976.
- Klaassen 1992, SSE 35, 125.
- del Alamo, SSE 28, 47.
- Yan & Cuevas 2014, JAP 116, 194505.
- Altermatt 2006 JAP full texts.
- Pearson & Bardeen 1949 full text.
- Hooge 1969 and 1981.
- The Sentaurus manual.

---

## 0. Findings that change the plan

These came out of reading the code and the literature, and they affect what the plan has to do.

1. **The v0.14 DEVSIM tests run nowhere in CI.** Seven new test modules `importorskip("devsim")`:
   - `test_bipolar_recombination_bgn_ionization`
   - `test_devsim_noise`
   - `test_extended_precision`
   - `test_devsim_phase_composition`
   - `test_traps`
   - `test_devsim_heterojunction`
   - `test_devsim_tunneling`

   The `test` legs have no devsim, so these modules skip there. The `solvers` leg runs only an explicit file list, and none of the seven is on it (grep count 0 for each in `.github/workflows/ci.yml`). This is exactly the T-5 failure mode that the workflow comment itself warns about. Workstream 0 below fixes it. Every gate in this plan depends on it.

2. **Two citations on the shipped presets are wrong, and one model is applied to the wrong density.**
   - **del Alamo.** The fit is IEDM 1985 Eq. (7) / IEEE TED ED-34, 1580 (1987) Eq. (11) **[P]**: ΔE_g^app = 18.7 meV · ln(N_D / 7e17) for N_D > 7e17, else 0. It is **n-type only** and uses **N_D**. The code applies it to N_D + N_A. SSE 28, 47 is not where the fit appears **[U]**.
   - **Old Slotboom.** The data come from **p-type** n-p-n transistor bases [S: Swirhun & Swanson 1986 NTRS report].
   - **Sb ionization energy.** The code ships 39 meV; the spectroscopic value is 42.74 meV (Ramdas & Rodriguez 1981, Table 4) **[P]**.

3. **None of the shipped apparent-BGN sets is paired with n_i = 1.0e10.**

   | Set | Paired n_i at 300 K | Basis | Label |
   |---|---|---|---|
   | Old Slotboom | ≈1.42–1.46e10 | 9.565e9 at 295 K | [I from P] |
   | del Alamo | ≈1.36–1.45e10 | TED 1987 Table II | [I] |
   | Klaassen 1992 | "Green's new value": 1.08e10 per the Green 1990 abstract, or about 1.0e10 | — | [U] |

   The modern consensus is n_i = 9.65e9 cm⁻³ at 300 K:
   - Altermatt et al. 2003 **[P]**;
   - Misiakos & Tsamakis 1993 give 9.7 ± 0.1e9 **[P]**;
   - Couderc 2014 **[P abstract]**.

   9.65e9 belongs with Fermi-Dirac statistics and a *real* BGN. The 1.00e10 of Sproul & Green is "influenced by BGN" (Altermatt 2003).

   Pairing old Slotboom or del Alamo with 9.65e9 under-predicts n_ie² by about 2.3x, which is about 21 meV of gap.

   The audit doc's statement that "the modern pairing is n_i = 1.0e10" should be corrected.

4. **A complete modern pairing exists, but half of it is paywalled.** Yan & Cuevas 2013 (JAP 114, 044508), Eq. (3) and Table I **[P, re-checked]**:
   - Form: ΔE = A [ln(N_D/N_ref)]^b + C.
   - **FD with real BGN:** A = 0.042 ± 0.003 meV, N_ref = 1e14, b = 3, C = 0.
   - **Boltzmann with apparent BGN:** A = 12.96 ± 0.70 meV, N_ref = 1e17, b = 1.
   - Inputs: n_i = 9.65e9, N_c = 2.86e19, N_v = 3.1e19, Klaassen mobility; n-type only.
   - The p-type companion is Yan & Cuevas 2014 (JAP 116, 194505). It gives both an FD and a Boltzmann parameterisation, but the **constants were not reached**.

5. **Both builders hard-code `fd_enhancement=True`.** This is true of `devsim_layered.py:678` and `devsim_3d.py:821`. Every Boltzmann-fitted apparent BGN set is therefore double-counted on degenerate layers, which the existing advisory reports. A house BGN choice needs either an FD-native model or a statistics knob.

6. **Hooge noise is likely wrong for currents not aligned with the mesh.** This was found by reading `noise_analysis._RegionData` and is **not yet measured**.
   - The module treats each edge's Hooge source `alpha J_e²/(n f)` as independent.
   - A Hooge source is a **scalar** conductivity fluctuation, δj = j δσ/σ. Its continuum output is ∫ (∇y·j)² α/(n f) dV.
   - The edge sum gives Σ_e c_e L_e (∇y·t_e)² (j·t_e)².
     - It is exact only when every edge with a nonzero couple is parallel or perpendicular to j: 1-D, or the DEVSIM rectangle/Kuhn meshes with axis-aligned current.
     - It **drops the cross terms**. For current at 45° on a rectangular mesh, the predicted result is 0.5x.
     - On an unstructured triangulation of a uniform bar the predicted ratio is Σ c L t_x⁴ / Σ c L t_x² < 1.
   - The existing gate F is 1-D, so it cannot see this. The diffusion source is unaffected, because its velocity fluctuations really are independent per Cartesian component.

7. **`builder.solve()` after `set_ssac_gate` is silently stale.** This was found by reading the code; it is not run yet.
   - `solve()` ramps the `<electrode>_bias` parameter.
   - After `set_ssac_gate`, that electrode's contact equation reads the circuit node, not the parameter.
   - So a later `solve(BiasPoint({gate: V}))` returns the old operating point without an error.
   - A convenience noise wrapper makes this path much more common, so item 6 must fix it.

8. **Pearson-Bardeen.**
   - PB's own boron E_A0 is 80 meV [P abstract], which is obsolete. Their fitted alpha was not seen [U].
   - The only primary-seen alphas are Silvaco ATLAS (2004 manual, Eqs. 3-53/3-54). ATLAS cites no primary source.
     - Donors: 3.6e-8 eV·cm with E0 = 44 meV.
     - Acceptors: 3.037e-8 eV·cm with E0 = 43.8 meV.
   - The modern replacement is Altermatt et al. 2006: E_dop = E_dop0 / (1 + (N/N_ref)^c), plus a non-localised fraction b(N). Its Table I was seen in the SISPAD 2006 companion [P].
   - **Units trap.** The code's `alpha_D_eVm` is in eV·m, but literature quotes eV·cm. Passing 3.1e-8 unconverted zeroes the ionisation energy above about 3e12 cm⁻³, and the clamp then hides it as silent full ionisation.

---

## Workstream 0 (prerequisite): put the DEVSIM gates on CI

**Steps:**
1. Measure each v0.14 DEVSIM module at `-n 2` on the CI-like Linux path (WSL Ubuntu, py3.12, devsim 2.11.2). Published local figures:
   - recombination: 36 s
   - noise: 35 s
   - extended precision with the dt-floor file: 80 s
   - heterojunction: 642 s including its slow gate
2. Add the seven modules to the `solvers` leg list. Any test over about 60 s gets `@pytest.mark.slow` and moves to `nightly` step (a). Keep the leg well under its 45-min cap; it is about 10 min today.
3. Every new module in this plan must be added to the same list in the same PR.

**Verification:** `pytest --collect-only` on the solvers-leg command must show non-zero items for each module.

**Effort:** half a day.

**Decision D13 (recommend yes):** do this before tagging v0.14.0.

---

## Item 1: per-material BipolarOptions presets and a house BGN model

### 1.1 Goal and physics

**Goal.** Give users cited, internally consistent Si parameter bundles (later GaAs and InP) that they opt into explicitly. Nothing is ever applied to an existing `TransportModel` automatically.

**Equations.** The existing kernels are unchanged, with one new BGN form:

    Slotboom   dEg = E_ref [ln(N/N_ref) + sqrt(ln^2(N/N_ref) + C)]            (existing)
    del Alamo  dEg = E_ref ln(N_D/N_ref), N_D > N_ref                          (existing; basis fix)
    power-log  dEg = A [ln(N/N_ref)]^b + C_off   for N > N_ref, else C_off    (NEW: Yan & Cuevas 2013 Eq. 3)
    n_ie = n_i exp(dEg/2kT),  E_c,v split by gamma                             (existing)

**Clamp below N_ref.** Yan & Cuevas use b = 1 (Boltzmann) and b = 3 (FD). With b = 1 the term is negative below N_ref, and with b = 3 it is non-monotone there. Their fits cover only diffused N_D ≥ 1e18. The clamp "0 below N_ref" for b = 1 is our choice and must be stated. For b = 3 with N_ref = 1e14, it never matters in practice.

### 1.2 What is reused

These pieces are reused unchanged:
- `BGNModel.delta_Eg_eV` gets a new `kind` branch inside the same method; there is no new kernel.
- `equilibrium_fields`, the SG quasi-fields, the contact rule, `srh_reference_densities`, and the noise pair rates already consume dEg generically, so a new kind propagates everywhere through one code path.
- `IonizationModel`, `AugerModel` and `RadiativeModel` are unchanged except for additive metadata fields.

### 1.3 Design

**A. Additive fields on the frozen dataclasses** (all defaulted, so existing constructions are unchanged). All of these live in `carriers/bipolar_options.py`; the probe confirmed that importing it does **not** load devsim.

    BGNModel(kind: "slotboom"|"del_alamo"|"power_log", E_ref_eV, N_ref_m3, C=0.5,
             conduction_fraction=0.5, source="",
             exponent_b: float = 1.0, offset_eV: float = 0.0,               # power_log only
             doping_basis: "total"|"donors"|"acceptors" = "total",          # del Alamo -> "donors"
             paired_n_i_m3: Optional[float] = None,                         # item 2
             paired_statistics: Optional["boltzmann"|"fermi_dirac"] = None, # item 2
             paired_T_K: float = 300.0,
             locator: str = "",                                             # "Eq. (3), Table I"
             verification: "primary"|"secondary"|"inferred"|"unverified" = "unverified")

`RadiativeModel` and `AugerModel` gain the same `locator` and `verification` fields. `RadiativeModel` also gains `kind: "internal"|"effective"|None` (see item 3).

**B. A new devsim-free module, `dynameta/materials/semiconductor_presets.py`.** It holds the named bundles and factory functions:

    SI_PAIRING_YC2013_FD        # n_i 9.65e9, Nc 2.86e19, Nv 3.1e19, FD, real BGN (n: YC 2013; p: YC 2014 -- PENDING)
    SI_PAIRING_YC2013_BOLTZMANN # same n_i, Boltzmann, apparent BGN (n: 12.96 meV ln(N/1e17); p: YC 2014 PENDING)
    SI_PAIRING_KLAASSEN1992     # legacy unified n/p apparent, Boltzmann, n_i = PENDING (1.0e10 vs 1.08e10)

    def silicon_bipolar_options(pairing=SI_PAIRING_..., *, auger="altermatt1997"|"dziewior_schmid"|None,
                                radiative=True, bgn=True, ionization=None, conduction_fraction=0.5)
        -> BipolarOptions
    def silicon_transport(*, doping_m3, acceptor=False, net_doping_expr=None, pairing=...,
                          mobility_n=..., mobility_p=..., tau_srh_s=..., options=True) -> TransportModel

**How `silicon_transport` sets its fields.**
- `n_i_m3`, `band_gap_eV` and the DOS masses come from the pairing:
  - m_c = 1.0911 m₀ reproduces N_c = 2.86e19 at T_REF = 300 K.
  - m_v = 1.1513 m₀ reproduces N_v = 3.1e19 (computed here from `effective_dos_m3`).
- `band_gap_eV` is set to the **statistical** gap kT ln(N_c N_v / n_i²):
  - 1.12973 eV for n_i = 9.65e9 (computed);
  - 1.12391 eV for 1.08e10.
  - It is not set to an optical 1.124 eV, because the builder's heterojunction check (`_HJ_EG_TOL_EV` = 5 meV) would refuse a 5.8 meV mismatch, and only the statistical gap is consistent with the bulk model.
- `chi_eV` = 4.05 eV, with its citation to be verified (Sze & Ng) [U].
- The mobilities are required arguments with **no default**. A preset of statistics and recombination must not pretend to calibrate transport.
- The pairing's fit used Klaassen mobility, which DynaMeta does not have. The docstring states this.

**C. Opt-in only.**
- `TransportModel`'s defaults are unchanged.
- `MaterialRegistry` never auto-registers presets.
- `run_pipeline` behaviour depends only on what the user constructs.
- Presets are frozen and named by source and year. A value change gets a **new name**; the old name stays with its old values.

**D. Statistics knob (Decision D3).** Add `TransportModel.carrier_statistics: Optional["fermi_dirac"|"boltzmann"] = None`.
- None keeps today's `fd_enhancement=True` in both builders, byte-identically.
- It is threaded to `setup_bipolar_region(fd_enhancement=...)` in `devsim_layered._setup_bipolar_region` and in `Stacked3DSpec.from_design` / the 3-D region setup.
- Without it, the Boltzmann-paired sets cannot be used correctly in the 2-D or 3-D builders.

**E. Corrections to the existing presets** (Decision D4; recommended before v0.14.0 is tagged):

| Preset | Change |
|---|---|
| `BGN_SI_DEL_ALAMO_1985` | `doping_basis="donors"`. Citation becomes IEDM 1985 Eq. (7) / TED 1987 Eq. (11). `paired_n_i_m3=None` (legacy, unpaired). |
| `BGN_SI_SLOTBOOM_1977` | Source note: p-type bases; paired n_i about 1.44e10 [I]. Stays unpaired. |
| `BGN_SI_KLAASSEN_1992` | Provenance [S]; n_i pending. |
| `SI_DONOR_IONIZATION_EV` | Sb → 0.0427 (R&R Table 4). Optionally P 0.0456, As 0.0538, B 0.0457. |

The legacy presets stay **unpaired**, so they keep the advisory-only behaviour and every existing R27 gate (n_i = 1.0e10 fixtures) is unchanged. Only the new pairing bundles enforce.

### 1.4 House model (Decision D1)

**Recommendation:** the `SI_PAIRING_YC2013_FD` bundle — Yan & Cuevas real BGN with FD statistics and n_i = 9.65e9. Reasons:
- **(a)** It is the only set fitted *with* Fermi-Dirac statistics, which is what both builders run. That removes the double-counting advisory instead of documenting it.
- **(b)** Its n_i is the modern consensus value.
- **(c)** Its constants are primary-verified.

**Blocker:** the p-type constants are in Yan & Cuevas 2014, which could not be reached.
- **User action:** supply the PDF of JAP 116, 194505.
- The bundle ships n-type-only until then. It **refuses** a p-type node with dEg > 0 under the house pairing; it does not silently fall back.
- A Si homojunction needs one n_i on both sides, so mixing Klaassen (≈1.0–1.08e10) on the p side would create a spurious built-in battery.

**Alternative:** `SI_PAIRING_KLAASSEN1992` with `carrier_statistics="boltzmann"` (unified n/p). This needs SSE 35, 125 to pin its n_i.

**Schenk 1998 stays deferred.** It needs carrier-density-dependent band edges (∂D_c/∂n in the SG Jacobian) and Padé tables.

### 1.5 Implementation steps

1. Add the `power_log` kind with its cancellation-safe branch, plus the `doping_basis` and metadata fields to `BGNModel`. In `equilibrium_fields`, select N by `doping_basis`. With the default "total", the arithmetic is byte-identical.
2. Write `semiconductor_presets.py`: the pairing dataclass `SiliconPairing(bgn_n, bgn_p, n_i_m3, N_c_m3, N_v_m3, statistics, T_K, mobility_note, sources)` and the factories.
3. Add `carrier_statistics` to `TransportModel`, validate it, and thread it through both builders.
4. Apply the preset corrections (1.3 E).
5. Write `docs/semiconductor_presets.md`: a table of value → paper → eq/table → verification level. It is generated from the dataclasses by a script, so docs and code cannot drift.

### 1.6 Verification gates

**Gates that need no DEVSIM** (`tests/test_semiconductor_presets.py`, `test` legs, under 2 s):

- **P1 value pins.**
  - Oracle: each constant re-typed in CGS from the paper inside the test, with the eq/table locator in a comment.
  - Tolerance: exact.
  - It fails on old code because the module is missing; the test uses `pytest.fail`, not `importorskip`.
  - Adversarial: a frozen-values test (sha256 of the preset reprs) catches any silent edit under an unchanged name.
- **P2 power-log formula.**
  - Oracle: hand evaluation of A[ln(N/N_ref)]^b at 1e18, 1e19, 1e20.
    - FD: 1e19 → 64.1 meV; 1e20 → 110.8 meV.
    - Boltzmann: 1e19 → 59.7 meV; 1e20 → 89.5 meV.
  - Tolerance 1e-13 (round-off of log and pow).
- **P3 FD vs Boltzmann self-consistency** (adversarial against transcription).
  - Yan & Cuevas state that both of their parameterisations reproduce the same n₀p₀ over the diffused range (their Fig. 3).
  - Independently compute n₀p₀ for N_D = 1e18–5e19:
    - (i) FD neutrality with the real BGN, using F₁/₂ by `scipy.integrate.quad` (not the code's `fd_potential_shift` fit);
    - (ii) Boltzmann n_ie² with the apparent BGN.
  - Tolerance |ratio − 1| < 0.3. Derivation: the ±0.70 meV and ±0.003 meV uncertainties at ln = 4.6–11.5 give ±13% and ±19% in n_ie², about 25% combined.
  - A constant mistyped by 10x would put the ratio off by e^(several).
  - It fails on old code because there is no `power_log` kind.
- **P4 statistics-gap consistency.** kT ln(N_c N_v / n_i²) must equal the preset's `band_gap_eV` to 1e-12, and `effective_dos_m3(m)` must reproduce N_c and N_v to 1e-12.
- **P5 opt-in identity.** `TransportModel(...)` built without presets has `bipolar_options is None` and `carrier_statistics is None`. The registry is unchanged.

**DEVSIM gates** (`tests/test_bipolar_recombination_bgn_ionization.py` additions, `solvers` leg, about 30 s):

- **P6 graded, current-free equilibrium with power_log.** Reuse fixture R27-b: a graded 1e20 → 1e17 bar with γ = 0.3/0.5/1.
  - Oracle: J = 0 and np = n_ie² everywhere.
  - Tolerances are the existing 1e-12 / 1e-12 x scale.
  - Negative control: drop the quasi-field.
- **P7 long-diode J0 under FD + real BGN.** n+ (1e19, about 13 L_p) / p long diode. Oracle:

      J0p = q D_p p0 / L_p

  where p0 comes from the scipy FD neutrality root (quad F₁/₂) with n_ie² = n_i² exp(dEg/kT).
  - **Tolerance 2e-3 plus the FD-fit error.**
    - The R27-c budget is 2e-3, from SG O((h/L)²), the finite emitter and the depletion de-rating.
    - **Step 0 measures** the `fd_potential_shift` rational-fit error against quad at η ≤ 3. An error δ volts maps to J0 error δ/V_t. If it is above 1e-4 V, either the gate widens with the derivation shown or the test asserts on the fit's error budget explicitly.
  - Load-bearing check: the ratio to BGN-off is exp(dEg/kT); at 1e19 that is ≈ 11.9 (computed). It must exceed 5.
- **P8 the double-counting advisory disappears.** A FD-paired preset on an FD region must emit no `AdvisoryWarning`; this is verified under the repo-wide "warnings are errors" policy. A Boltzmann-paired preset on an FD region must raise (item 2).
- **P9 defaults-off.** The existing call-log digest gate (`tests/_bipolar_calllog.py`, 217 calls), the D1d array SHA, and `BipolarOptions()` vs None bitwise must all be unchanged. They are rerun, not re-recorded.
- **P10 statistics knob threading.**
  - `carrier_statistics="boltzmann"` must reach `setup_bipolar_region` in 2-D and 3-D: check the `fd_enhancement` region parameter equals 0 after build. The 3-D check is build-only, about 5 s.
  - None must be byte-identical; check by the D1d SHA.

**Adversarial gate (nightly, optional):** reproduce two of Yan & Cuevas's Table II emitter J0 values (their measured data) through a 1-D emitter.
- Expected disagreement is about 10–30%, because DynaMeta uses constant per-region mobility instead of Klaassen's.
- Report it as a validation, not a hard gate. A >2x miss means a pairing or units error.

### 1.7 Risks

- The p-type constants are blocked on a paywall.
- The Klaassen n_i is unverified.
- The FD-fit error budget is not yet measured.
- The power-log clamp below N_ref is our choice.
- `chi_eV` for Si still needs a cited source.
- The parallel `TransportModel` edit causes merge churn, so sequence after it lands.

**Effort:** 3–4 days, including the docs generator. Add a day once the 2014 PDF arrives.

---

## Item 2: n_i pairing check

### 2.1 Literature status

See section 0, finding 3, and the table below.

| Set | Paired n_i (300 K) | Statistics | Status |
|---|---|---|---|
| Old Slotboom 9 mV / 1e17 / 0.5 | ≈1.44e10 (9.565e9 at 295 K) | Boltzmann | constants [P via Stanford report], n_i [I] |
| del Alamo 18.7 meV / 7e17 (N_D) | ≈1.36–1.45e10 | Boltzmann (TED 1987 Eq. 10) | form [P], n_i [I] |
| Klaassen 6.92 mV / 1.3e17 / 0.5 | 1.08e10 or ≈1.0e10 | Boltzmann | constants [S: Atlas 2006, Sentaurus list], n_i [U] |
| Yan–Cuevas 2013 apparent, n | 9.65e9 | Boltzmann | [P] |
| Yan–Cuevas 2013 real, n | 9.65e9 | FD | [P] |
| Sproul & Green 1990/91 | 1.01e10 ± 3% / 1.00e10 | — | [P] / [S: Altermatt 2003] |
| Misiakos & Tsamakis 1993 | 9.7 ± 0.1e9 | — | [P] |
| Altermatt 2003 | 9.65e9 (Schenk BGN removed from SG data) | FD | [P] |
| Green 1990 | 1.08e10, N_v 3.1e19 (N_c 2.86e19 [S]) | — | [P abstract] |

**Remaining literature tasks** (they need the user's library access):
- SSE 35, 125: Klaassen's n_i.
- JAP 116, 194505: Yan & Cuevas p-type constants.
- Optionally Couderc 2014's revised N_c and N_v. If Couderc changes N_c or N_v, the statistical gap moves; the pairing bundle carries them, so no code change follows.

### 2.2 Encoding

The pairing lives **on the BGN model**: `paired_n_i_m3`, `paired_statistics`, `paired_T_K`. It is checked once, in `install_option_fields`, against the region's `n_i` parameter and its `fd_enhancement`. The same check also goes into `equilibrium_fields`, so the 3-D `phi_bi` path is covered.

The check is a **hard `ValueError`** (Decision D2) when either of these holds:
- **(a)** |n_i / paired_n_i − 1| > 1e-3.
  - Tolerance: n_i values are quoted to three significant figures, and 1e-3 equals 0.05 meV of gap, far below the fits' ±0.7 meV.
  - The message prints the energy-equivalent error kT ln((paired/n_i)²). For 1.0e10 against 9.65e9 that is 1.84 meV, and n_ie² is off by 7.4%.
- **(b)** The paired statistics differ from the region's statistics.
- **(c)** `T_K` is not None and |T_K − paired_T_K| > 0.5 K (Decision D5).
  - There is no n_i(T) model shipped, and the fit is a 300 K fit. The caller must pass `allow_unpaired=True` together with their own n_i(T).
  - Couderc's n_i(T) = 1.541e15 T^1.712 exp(−E_g0/2kT) [P abstract] could become a future n_i(T) model, but E_g(T) needs a Pässler-type law, which is out of scope.

**Override:** `BipolarOptions(allow_unpaired_bgn: bool = False)`. Setting it True turns the raise into the existing `AdvisoryWarning`.

Legacy presets (`paired_n_i_m3=None`) keep the existing heuristic advisory and nothing else, so nothing changes for them.

### 2.3 Gates

All are in the `solvers` leg except Q1, and all take under 10 s.

- **Q1 (pure).** Enumerate every shipped pairing bundle. Its `n_i`, N_c, N_v and gap must satisfy the gap identity (item 1, P4), and each BGN in it must carry `paired_n_i_m3 == bundle.n_i`. This catches a bundle assembled from mismatched sets.
- **Q2.** n_i = 1.0e10 with an FD 9.65e9-paired preset raises, and the message contains "1.84 meV" (to 0.01). With n_i = 9.65e9 it does not raise.
- **Q3.** A Boltzmann-paired preset on an FD region raises. With `carrier_statistics="boltzmann"` it does not raise, and no advisory is emitted.
- **Q4.** `T_K=350` with a paired preset raises. With `allow_unpaired=True` it gives exactly one `AdvisoryWarning`.
- **Q5.** The 3-D `phi_bi` path (`ohmic_built_in_offset_V`) raises identically. This proves there is one check site, not two.
- **Q6 defaults-off.** Legacy presets on n_i = 1.0e10 fixtures: all 22 existing R26/R27 gates pass unchanged.
- **Adversarial.** Construct a bundle with the n-type and p-type sets taken from different pairings. Q1 must catch it.

**Fails on old code:** there are no `paired_*` fields, so construction raises `TypeError`, and the test asserts that.

**Effort:** 1 day, plus the literature closure, which is blocked on the user.

---

## Item 3: III-V radiative and Auger presets

### 3.1 Physics and the key distinction

U_rad = B (np − n_ie²). The kernel is unchanged. In a direct-gap device, a share P_r of the emitted photons is re-absorbed (photon recycling), so the net loss is B_eff = B_int (1 − P_r), and P_r depends on thickness and cladding. DynaMeta has no re-absorption model, so a preset must say which kind of B it is.

**Literature results:**

| Material | Quantity | Value | Source | Label |
|---|---|---|---|---|
| GaAs | B_int | 3.3 ± 0.6e-10 cm³/s, p < 1e18; 2.1e-10 at 1e18 | Niemeyer et al., AIP Adv. 9, 045034 (2019), Eq. 1, Sec. VI | [P] |
| GaAs | B_eff, 2 µm DH | 0.78 ± 0.07e-10; P_r = 0.76 | Niemeyer 2019 | [P] |
| GaAs | B_int, n-type | 3.5e-10 low-n → 1.8e-10 at 3.8e18 | Lush, Sol. Energy Mater. Sol. Cells 93, 1225 (2009) | [P abstract] |
| GaAs | B (VRS calculation) | 7.2e-10 | Varshni 1967 | [S]; calculated, about 2x the measurements |
| GaAs | C_amb (high injection) | 7 ± 4e-30 cm⁶/s | Strauss, Rühle & Köhler, APL 62, 55 (1993) | [S: two papers] |
| InP | B_int | 1.2e-10 · (1 − 0.092 n/1e18) | Semyonov et al., arXiv 1003.6095, Eq. 6 | [P] |
| InP | C | — | paper is internally inconsistent; Ioffe is unsourced | not shippable |
| In0.53Ga0.47As | B (probably effective) | 1.43e-10 | Ahrenkiel et al., APL 72, 3470 (1998) | numbers OCR only; recycling not removed |
| In0.53Ga0.47As | C | 8.1e-29 | Ahrenkiel et al. 1998 | numbers OCR only |
| GaN | B | 2.4e-11 (Im 1997) [S]; 1.5–5e-11 (Dmitriev calc.); 1.1e-8 (Muth, exciton-inflated) | — | 3-decade spread |
| GaN | C | measured values are InGaN (1.4–2e-30, Shen 2007); pure-GaN direct theory < 1e-35 | — | — |
| Ge | B, C | 6.41e-14 / 1e-30 | Ioffe, unsourced | [S, unsourced] |

### 3.2 Design

**RadiativeModel additions:**
- `kind: Optional["internal"|"effective"] = None` (None means legacy).
- `photon_recycling_factor: Optional[float] = None`, valid in [0, 1).
- When both are set, the effective B used is B_int (1 − P_r).
- **One kernel:** the factor is applied once, at the `B_rad` parameter write in `install_option_terms`. There is no second expression.
- `kind="effective"` with a recycling factor raises: it would subtract recycling twice.
- `kind="internal"` with no factor emits one `AdvisoryWarning`: "internal B used as net loss; recycling ignored (overestimates radiative loss in thick direct-gap layers)". Tests must expect it.

**Presets to ship** (Decision D6), placed in `semiconductor_presets.py`:
- `RADIATIVE_GAAS_NIEMEYER_2019_INTERNAL` = 3.3e-10 cm³/s.
- `AUGER_GAAS_STRAUSS_1993` with C_n = C_p = 3.5e-30. The docstring says: "ambipolar sum 7 ± 4e-30, high-injection measurement, split equal by assumption".
- `RADIATIVE_INP_SEMYONOV_2010_INTERNAL` = 1.2e-10. The doping roll-off is documented, not modelled.

**Deferred** until full-text verification: InGaAs (OCR only), GaN (spread, and its Auger is InGaN), Ge (unsourced). Varshni's value is not shipped.

**Noise caveat (document it):** with recycling, an emitted-then-reabsorbed photon is a correlated pair event. Using B_eff in the GR pair rate is an approximation, and the docstring says so.

### 3.3 Gates

- **R1 value pins (pure).** Values re-typed with locators, compared exactly.
- **R2 internal consistency of the source (pure, adversarial against mislabelling).** B_int (1 − P_r) = 3.3e-10 × 0.24 = 0.792e-10. It must match Niemeyer's B_eff of 0.78 ± 0.07e-10 within his stated uncertainty.
  - Swapping the labels gives 3.3/0.78 = 4.2x, which fails.
- **R3 (DEVSIM, about 5 s).** Reuse the R26-d radiative-dominated slab with GaAs-like inputs: n_i = 2.1e6 cm⁻³ (verify the citation), n₀ = 1e17.
  - Oracle: τ_rad = 1/(B_eff n₀) from the brentq root.
  - Tolerances: 1e-12 node-wise and 1e-9 for the root, as in R26-d.
  - This proves the recycling factor reaches the `B_rad` parameter, and that a GaAs-scale n_i² (4.4e24 m⁻⁶) is representable.
  - It fails on old code (`TypeError` on `kind`).
- **R4 refusal and advisory.** Effective plus a factor raises. Internal without a factor warns exactly once.
- **R5 defaults-off.** The call-log digest is unchanged, and legacy `RadiativeModel(B_m3s=...)` writes the same `B_rad` bits.

**Decisions:** D6 (which presets, the label policy, and the equal C_n/C_p split), and whether to model the B(N) roll-off (recommend: document it, do not model it).

**Effort:** 1–1.5 days. Add half a day to verify the deferred materials if the user wants them.

---

## Item 4: Pearson-Bardeen alpha, and the better replacement

### 4.1 Physics

The existing model is dE(N) = dE0 − α N^(1/3), clamped at 0 (Pearson & Bardeen, Phys. Rev. 75, 865 (1949)).

**Literature:**
- PB's own E_A0 for B was 80 meV and vanished above 5e18 [P abstract]. Their α number was not seen [U].
- ATLAS 2004 [P]:
  - E_D = 0.044 − 3.6e-8 N_D^(1/3) eV;
  - E_A = 0.0438 − 3.037e-8 N_A^(1/3) eV;
  - for N < 1e18; linear blend to full ionisation between 1e18 and 3e18.
- The Sentaurus default "3.1e-8" is [U].
- The physically fitted modern model is Altermatt et al., JAP 100, 113714/113715 (2006). Equations from the SISPAD 2006 companion, Table I [P]:

      E_dop = E_dop0 / (1 + (N/N_ref)^c)
      b(N) = 1 / (1 + (N/N_b)^d)
      N+/N = 1 - b n/(n + g n1),   n1 = N_c exp(-E_dop/kT)

  | Dopant | E_dop0 (meV) | N_ref (cm⁻³) | c | N_b (cm⁻³) | d | g |
  |---|---|---|---|---|---|---|
  | P | 45.5 | 2.2e18 | 2 | 6e18 | 2.3 | 1/2 |
  | As | 53.7 | 3e18 | 1.5 | 9e18 | 1.8 | 1/2 |
  | B | 44.39 | 1.3e18 | 1.4 | 4.5e18 | 2.4 | 1/4 |

  - The Mott density for P is N_crit = 3.74e18 [P].
  - The peak non-ionised fraction is about 25% for P and B and about 35% for As [P abstract].
  - g = 1/2 in their form equals g_D = 2 in ours, and 1/4 equals g_A = 4.

### 4.2 Recommendation (Decision D7)

1. **Ship no PB default.** No primary-verified PB α exists. The ATLAS numbers are a code default, not a measurement, so they appear only as documented example values. Keep α = 0.
2. **Units guard.** In `IonizationModel.__post_init__`, if α > 0 and the zero crossing (dE0/α)³ falls below 1e23 m⁻³ (1e17 cm⁻³), raise. The message says: "alpha looks like eV·cm; this field is eV·m = 1e-2 × eV·cm".
   - A legitimate PB α makes the energy vanish at 1.8e18–3e18 cm⁻³, so the guard has 1.3 decades of margin.
   - Add the classmethod `IonizationModel.pearson_bardeen(dE_D_eV, dE_A_eV, alpha_D_eVcm, alpha_A_eVcm)`, which does the conversion explicitly.
3. **Mott-density advisory** for the existing model. Above N_crit (3.74e18 P [P]; As and B [U]; use the Altermatt N_b as the documented proxy) the legacy model freezes out wrongly: 39% ionisation at 1e19. Emit one `AdvisoryWarning` per region. This is a behaviour change only on already opted-in ionisation, which is unreleased.
4. **Follow-on (recommended, a separate PR): `IonizationModel(kind="altermatt2006")`.** E_dop(N) and b(N) become fixed fields in `equilibrium_fields`, and the DEVSIM charge string becomes

       IonDonors*(1 - IonBD*Electrons/(Electrons + IonN1D/IonGD))

   This is the same builder function with one extra factor.
   - The neutral root stays strictly monotone. d/dn [b n/(n + c)] = b c/(n + c)² > 0, so −N_D+ is still increasing, and 0 ≤ N_D+ ≤ N_D keeps the existing bisection bracket valid. The proof in the docstring extends verbatim.
   - The JAP Table I itself still needs to be checked against the SISPAD numbers. Steinkemper 2016 [S] says the two tables match.

### 4.3 Gates

- **I1.** PB algebra: E at N = 1e18 for ATLAS-example α, re-typed in the test, tolerance 1e-15.
- **I2.** Units guard: `alpha_D_eVm = 3.1e-8` (the eV·cm value) raises; `3.1e-10` passes. The classmethod reproduces 3.1e-10 exactly.
- **I3.** Mott advisory fires at P 1e19 and not at 1e18.
- **I4 (follow-on, Altermatt).** Freeze-out root against brentq on a hand-typed residual, at 1e-11 (the existing R27-d tolerance) over 50–300 K and 1e16–1e20.
  - **External oracle:** the maximum non-ionised fraction for Si:P at 300 K must lie in [0.20, 0.30]. The abstract says "up to 25%", read from a curve, hence the ±5% band.
  - Load-bearing: PB or legacy at 1e19 gives 0.61 non-ionised, which fails.
- **I5 defaults-off.** α = 0 must give the R27-d gates bit-identical, and the call-log digest unchanged.

**Effort:** half a day for the guard, advisory and docs. Altermatt 2006 is 1.5–2 days.

---

## Item 5: per-material Hooge alpha

### 5.1 Physics and literature

For each carrier the Hooge source is

    K_H = alpha_H J² / (n f)

which gives S_I/I² = α/(N f) for a uniform bar.

**Literature:**
- **Hooge 1994** (IEEE TED 41, 1926) [P, full text]:
  - α is **not** a constant.
  - It is about 1e-4 in perfect lattices and 1e-6 to 1e-3 measured.
  - Eq. 13: α = (μ/μ_latt)² α_latt.
  - The historical 2e-3 is a "rather poor samples" average.
  - The representative value is α_latt = 1e-4.
- n-MOSFET 1/f noise is number fluctuation (McWhorter), which is not a Hooge source (Vandamme et al. 1994) [S].
- Typical ranges:
  - GaAs about 7e-5 (epitaxial n);
  - InGaAs about 1e-5;
  - GaN 1e-4 to 3e-3;
  - **ITO: no published α_H found.**
- ATLAS uses separate HOOGEN/HOOGEP parameters that **default to 0** [P].

### 5.2 Design (Decision D8)

**TransportModel** gets `hooge_alpha_n: Optional[float] = None` and `hooge_alpha_p: Optional[float] = None`. They are validated finite and > 0 when set. **There are no per-material defaults.** The docs give the cited ranges.

**noise_analysis** keeps `hooge_alpha: float` (both carriers, all regions; unchanged). It additionally accepts a `Mapping[str region, float | (alpha_n, alpha_p)]`:
- regions that are absent get no Hooge source;
- a non-positive entry raises.

The mapping is read in `_RegionData` only. A float keeps today's arithmetic bit-for-bit.

`noise_analysis` stays device-level and knows nothing of `TransportModel`. The builder wrapper (item 6) resolves region → material → alphas. **There is one reader.**

**Step 0, measure the suspected cross-term defect (finding 6):**
- **H-a.** A uniform 2-D bar on a gmsh unstructured triangulation (no 2-D gmsh path exists in DynaMeta, so use a DEVSIM mesh loaded from a hand-written `.msh`). Compare S·f / (α I²/N) against 1.
  - Predicted: Σ c L t⁴ / Σ c L t² < 1.
  - The diffusion source on the same mesh must stay at 1 to round-off; gate A already shows this for 3-D tetra. That is the control.
- **H-b.** DEVSIM rectangle mesh with the current along x: predicted exact. This is the positive control.
- **If H-a confirms** (Decision D9):
  - **Recommended now:** in `_RegionData`, refuse Hooge (`NotImplementedError`) when any edge with |c_e| > 1e-12·max|c| has 1e-6 < |t_e·ĵ_e| < 1 − 1e-6 against the local current direction. The local direction is reconstructed from the node-averaged edge currents.
    - Simpler alternative: refuse whenever the mesh has non-axis-aligned edges with nonzero couple and the region's |J| is non-negligible.
    - The refusal message points to the limitation.
  - **Follow-on:** a correlated scalar source per node or per element. Independent δσ_i/σ per node volume perturbs every incident edge's SG current by ∂J_e/∂μ_i. That gives one source per node with Green's function Σ_{e∋i} c_e Δy_e ∂J_e/∂ln σ_i.
    - Before building it, derive whether it is exact on Voronoi meshes. The node-local box identity is not obvious.
    - Gate the uniform bar on H-a's unstructured mesh to 1e-9.
    - Research effort is 2–4 days.

### 5.3 Gates

- **H1 (existing, kept).** Gate F, 1-D: 7.6e-6 against 1e-4.
- **H2 mapping semantics.**
  - Two-region resistor (two materials in series, uniform J) with α₁ ≠ α₂. Oracle: S·f = I² Σ_r α_r / N_r.
    - Tolerance 1e-4, the gate-F budget; its residual is the contact drift/Debye term derived in the noise doc, below 1e-5 at 10 µm.
  - A region missing from the mapping contributes exactly 0.
  - (α_n, α_p) on a bipolar bar scales each carrier separately. Oracle: per-carrier sums, `by_mechanism["hooge_n"/"hooge_p"]`.
- **H3 float path is bit-identical.** `hooge_alpha=2e-3` gives the same arrays as before; check by sha256 of `by_region_mechanism`.
- **H4 defaults.** `TransportModel` without alphas plus the wrapper gives no Hooge mechanism, and S is bit-identical to the no-Hooge call.
- **H5 (step 0 measurement, H-a/H-b),** then the refusal gate if adopted: unstructured bar → `NotImplementedError`; rectangle bar → passes.
- **Old code fails:** a mapping argument raises `TypeError` in `float(hooge_alpha)`, and the test asserts that.

**Effort:** 1 day for plumbing and gates, plus 1 day for the step-0 measurement and refusal. Add 2–4 days if the correlated source is chosen.

---

## Item 6: `LayeredDevsimBuilder.noise_spectrum(...)` wrapper

### 6.1 The current recipe and its traps

Today the hand recipe (gate G, `validation/devsim_noise.py:625`) is: `solve(bias)`, then `set_ssac_gate`, then `ds.circuit_alter`, then a raw DC re-settle, then `NA.noise_spectrum` on the device.

**Traps found:**
- **(a) Raw `ds.solve` re-settles bypass `XP.solve`.** `validation/modulator_spec_sheet.py` does this. On an extended-precision device that is the double Newton.
- **(b) Circuits are process-global and cannot be deleted** (integration doc §7). A second circuit node after an analysis is refused.
- **(c) Stale `solve()` after repointing** (finding 7).
- **(d) Units.** 2-D results are per unit depth.
- **(e) Scope.** `set_ssac_gate` is for dielectric gates. Ohmic semiconductor contacts warn, and only Potential is repointed.

### 6.2 API (Decision D10)

    def noise_spectrum(self, frequencies, *, gate: str, vg: Optional[float] = None,
                       source_name: str = "V1", node_name: str = "vac",
                       mechanisms=("diffusion_n", "diffusion_p", "gr"),
                       hooge: Union[None, str, float, Mapping] = None,  # "material" -> TransportModel alphas
                       external_generation_shot: bool = True, maps: bool = False,
                       per_cell: bool = True,
                       rel_tol: float = 1e-5, abs_tol: Optional[float] = None,
                       max_iter: int = 60) -> NoiseSpectrum
    def terminal_noise(self, frequencies, *, gate, vg=None, ..., per_cell=True) -> TerminalNoise

**Behaviour:**
1. The builder must be built and solved; otherwise `RuntimeError`.
2. If `gate` is not yet circuit-driven, call `set_ssac_gate` once and record it in `self._circuit_gates`. A second call reuses the gate. A gate repointed under a different source name raises.
3. `vg` defaults to the last converged bias of that electrode.
4. `ds.circuit_alter` is followed by `solve_dc` (the `XP` path) at the builder's tolerances, using the bipolar or DD branch exactly as `solve()` picks it.
5. Call `NA.noise_spectrum`, passing a resolved Hooge mapping when `hooge="material"`.
6. If `per_cell`, scale S, every breakdown and every map by `design.unit_cell.period_y_m`.
   - Why this is exact: the 2-D device is y-invariant, so G is y-independent and spatially white sources add incoherently along y: S_3D = W · S_2D.
   - This is the same convention as `extras["contact_currents_A"]`.
   - The returned object records `units="A^2/Hz per cell"` or `"per unit depth"`.

**Scope v1:** `gate` must be a dielectric (Potential-only) contact. An ohmic semiconductor contact raises `NotImplementedError`, pointing to a follow-on `set_ssac_contact` built on `physics_bipolar_dd.setup_contact_ohmic_bipolar_circuit(psi_offset_V=frame shift)`. The voltage and current drives are both there already; this is Decision D10b.

**Fix (c), Decision D11, recommended:** `solve()` and `_solve_bipolar` drive a circuit-repointed electrode through `ds.circuit_alter(source)` instead of the dead parameter. The alternative is to refuse. The ramp logic is unchanged otherwise.

**Fix (b):** catch DEVSIM's circuit refusals in `set_ssac_gate` and re-raise with an actionable message ("circuit already exists / analysis already run; use a fresh process").

### 6.3 Gates

All are in the `solvers` leg. The first-time gate-G device takes about 20–30 s, so build it once as a module fixture.

- **W1 bitwise equivalence.** Wrapper output with `per_cell=False` must equal the hand recipe on the same device and state, at Vg 0, bit for bit (sha256 of S and the breakdowns). Same calls, so tolerance is exact. **Old code fails** with `AttributeError`.
- **W2 FDT through the wrapper.**
  - At Vg 0: S_I = 4kT Re Y. Re Y comes from `ssac_admittance` (a separate linear solve), scaled by the same period.
  - Tolerance 1e-9 (gate G measured 1.3e-11).
  - At +2 V: 1e-3 (measured 3.7e-5; the FD-degenerate limit is documented).
- **W3 per-cell scaling.** S_cell / S_depth = period_y exactly (1 ulp).
  - Independent check: the 3-D tensor builder on the same flat-band capacitor at Vg 0 gives absolute S = 4kT Re Y_3D. The 2-D per-cell value must agree to the 2-D/3-D model difference, which `validation/devsim_heterojunction` gate F measured at about 1e-3 for DD vs equilibrium.
  - Use 5e-3, or report it rather than gate it if the 3-D build exceeds 60 s (then it is nightly).
- **W4 idempotence and state (adversarial).**
  - Call at Vg 0, then +1 V, then 0. The third result must be bitwise equal to the first.
  - Call at +1 V on a fresh process. It must equal the second call to 1e-12 (different continuation paths give Newton round-off). This refutes leaking continuation state.
- **W5 stale-solve fix.** After the wrapper, `solve(BiasPoint({gate: 1.0}))` must equal a fresh build solved at 1.0 to 1e-9 in n and ψ. **Old code fails:** it returns the Vg-0 state, and the test asserts the difference.
- **W6 refusals.**
  - Ohmic contact → `NotImplementedError`.
  - Unbuilt builder → `RuntimeError`.
  - Second source name → `ValueError`.
- **W7 extended precision.** On an opted-in builder, exactly one `AdvisoryWarning` per call site. Expected with `pytest.warns`.
- **W8 Hooge "material".** A two-material stack with alphas set gives the same per-region Hooge as the explicit mapping, bitwise.

**Effort:** 1.5–2 days, including fix (c).

---

## Item 7: extended precision on Design (reachable from run_pipeline)

### 7.1 Current state

The switch exists in three places only:
- `LayeredDevsimBuilder(extended_precision=False)`;
- `Stacked3DSpec.extended_precision`;
- `from_design(extended_precision=False)`.

`run_pipeline` constructs `LayeredDevsimBuilder(design)` with no knob (`pipeline.py:109`).

`XP` is device-scoped:
- `extended_model` and `extended_equation` are device parameters;
- `extended_solver` is global, set per call.

The ssac/noise floor is documented in `extended_precision.SMALL_SIGNAL_MESSAGE`, which `XP.solve` warns with.

### 7.2 Design (Decision D12)

**Recommended:** a new `CarrierSpec` dataclass in `geometry/specs.py`, attached as `Design.carriers: CarrierSpec = field(default_factory=CarrierSpec)`.

- It is appended **after** `pretty_name`, so positional `Design(...)` calls are unaffected.
- Its first field is `extended_precision: bool = False`. Its docstring carries the cost (6x in 1-D, 18–22x in 2-D), the use case (wide-gap currents below the ~1e-5 relative round-off floor of SG current differences), and the ssac/noise floor guidance (use DC dI/dV; drive the lightly doped contact).

**Alternatives considered:**
- **(a) `Mesh2DSpec.extended_precision`.** There is precedent (`dc_method` lives there), but the 3-D builder would read a "2-D mesh" spec.
- **(b) `TransportModel.extended_precision` per material.** Rejected:
  - DEVSIM's `extended_solver` is global per Newton;
  - per-region `extended_model` is possible but untested;
  - mixed-precision regions inside one Newton have no gate;
  - a per-material flag would invite that.

**Builder kwargs:** `extended_precision: Optional[bool] = None` on both builders and `from_design`. None follows `design.carriers.extended_precision`; an explicit bool wins.
- If an explicit False contradicts a design True, emit one `AdvisoryWarning` ("explicit argument overrides Design.carriers"). This avoids silent disagreement between two sources of truth.

**Provenance:** when the flag is on, `CarrierField.extras["extended_precision"] = True`. With it off, no key is added, so the CarrierField is byte-identical.

**Cache:** `cache._design_fingerprint` hashes only the optical and `mesh_3d` reprs, so the new field does not invalidate existing optical caches. Extended precision changes only the carrier result, which reaches the cache key through `eps_by_region`. That is the correct dependency.

### 7.3 Gates

- **E1 defaults-off.** `python validation/extended_precision.py --hashes` gives 7/7 fingerprints identical to the recorded column, including both 2-D layered rows. Also: the D1d SHA and the call-log digest.
- **E2 equivalence.** On a small bipolar 2-D design (`validation/bipolar_diode_2d`, coarse mesh), `LayeredDevsimBuilder(design_with_flag)` must equal `LayeredDevsimBuilder(design, extended_precision=True)` bitwise (sha256 of all node fields and contact currents). Runtime must be measured; it is expected under 30 s, otherwise nightly.
- **E3 run_pipeline reachability** (cheap, no extended solve). With `carrier_solver=None`, a spy on `XP.enable` records the device and an optical stub skips ngsolve. The spy is called exactly once when the flag is on and never when it is off. **Old code fails:** `Design` has no `carriers` field (`TypeError`).
- **E4 load-bearing (nightly).** A GaN-like 2-D bipolar diode through `LayeredDevsimBuilder` from the design flag. Oracle: the Sah-Noyce-Shockley J at +1.0 / +1.5 V (the existing 1-D gate-A oracle) × the cell width. Tolerance 3e-3, the gate-A budget. Double must be off by more than 1e3, the asserted margin.
- **E5 precedence.** Explicit True on a False design → enabled. Explicit False on a True design → not enabled, plus exactly one advisory.
- **E6 3-D build.** `from_design(design_with_flag)` gives `XP.is_enabled(device)` after build. Build only; no 3-D solve, which is minutes long.
- **E7 small-signal advisory.** The wrapper (item 6) on a design-flagged builder emits one `AdvisoryWarning` with the floor guidance. Gated with `pytest.warns`.
- **Adversarial.** In `run_pipeline` with a flagged design, check `ds.get_parameter(name="extended_solver")` after each bias solve (the spy wraps `solve_dc`): it must be False outside the calls. Teardown must leave no extended device for a following default `run_pipeline`; the second run's fingerprint must equal a fresh default run.

**Effort:** 1–1.5 days.

---

## Workstreams, order, effort

| WS | Items | Depends on | Effort | CI placement |
|---|---|---|---|---|
| **0** CI wiring | — | — | 0.5 d | solvers leg list; slow gates → nightly |
| **A** Si presets + pairing + ionization | 1, 2, 4 (guard/advisory; Altermatt-2006 as A2) | WS0; the parallel TransportModel change merged; user PDFs for the p-type and Klaassen n_i | 4–5 d (+1.5–2 d A2) | pure gates in the test legs (< 2 s); DEVSIM gates in solvers (≈ 40 s) |
| **B** III-V presets | 3 | A's metadata schema (`kind`/`locator`/`verification`) | 1–1.5 d | pure + one 5 s DEVSIM gate |
| **C** Noise wrapper + Hooge | 6, 5 | WS0 | 3–4 d (+2–4 d correlated source) | solvers ≈ 60 s with a module fixture; 3-D cross-check nightly |
| **D** Extended precision on Design | 7 | WS0 (may share C's wrapper for E7) | 1–1.5 d | solvers ≈ 30 s; E4 nightly |

**Order and concurrency:**
- 0 comes first.
- A and C can run in parallel. They touch disjoint files except `transport_model.py`, which both A and C edit with additive fields, so merge A first.
- D is independent.
- B follows A.

**Total:** about 10–13 working days, excluding the optional A2 and the correlated Hooge source.

**Defaults-off guards, rerun at every merge** (never re-recorded):
- the R26/R27 call-log digest;
- the D1d array SHA (`a500ca93…`);
- the D1c cross-version hashes;
- the 7 extended-precision fingerprints.

## Decisions for the user (with recommendations)

- **D1 House Si BGN.** Recommend Yan & Cuevas 2013/2014 real BGN with FD and n_i = 9.65e9, which removes the FD double-counting. Ship n-type only until you supply JAP 116, 194505. Alternative: Klaassen with Boltzmann, which needs SSE 35, 125 for its n_i.
- **D2 Pairing mismatch.** Recommend a hard error, tolerance 1e-3 on n_i, override `allow_unpaired_bgn`. Legacy presets stay unpaired and advisory-only.
- **D3 Statistics knob.** Recommend adding `TransportModel.carrier_statistics` (None = today's FD).
- **D4 Correct the shipped presets before tagging v0.14.0.** Recommend yes: del Alamo citation and N_D basis, Slotboom p-type note, Sb 42.7 meV, and fix the audit doc's "n_i = 1.0e10" statement.
- **D5 T_K ≠ 300 with paired presets.** Recommend refusing unless `allow_unpaired`; no n_i(T) model now.
- **D6 III-V.** Recommend shipping GaAs B_int 3.3e-10 (internal), GaAs Auger 3.5e-30 each (high-injection, split equally) and InP B_int 1.2e-10, plus the `photon_recycling_factor` field. Defer InGaAs, GaN and Ge; never ship Varshni's 7.2e-10 as a measured B.
- **D7 Pearson-Bardeen.** Recommend no default α; add the units guard and the Mott advisory. Implement Altermatt 2006 as the follow-on physics default.
- **D8 Hooge.** Recommend no per-material defaults; per-carrier `TransportModel` fields default to None; the noise module takes a mapping. The docs cite ranges, with 1e-4 as Hooge's own representative value, not 2e-3.
- **D9 Hooge cross-term defect.** Recommend measuring first. If confirmed, refuse non-aligned cases now, and decide later on the correlated-source research item.
- **D10 Wrapper.** Recommend dielectric gates only in v1, per-cell units by default (`per_cell=True`). D10b: an ohmic-contact `set_ssac_contact` as a follow-on.
- **D11 Stale `solve()` after `set_ssac_gate`.** Recommend driving the circuit source (rather than refusing).
- **D12 Extended-precision placement.** Recommend a new `Design.carriers: CarrierSpec`. Reject per-material placement.
- **D13 CI.** Recommend wiring the v0.14 DEVSIM modules into the solvers leg before tagging.

## Unverified items to close before quoting externally

- Klaassen 1992's n_i.
- Yan & Cuevas 2014 p-type constants.
- The Sentaurus defaults for BGN offsets and PB α.
- PB's own α.
- The As and B Mott densities.
- The JAP 100 Table I itself (seen only via SISPAD and Steinkemper).
- The Ahrenkiel 1998 InGaAs numbers (OCR only).
- The GaN B values (Im 1997; Dmitriev 1999, 1.5 vs 5e-11).
- Si χ = 4.05 eV and GaAs n_i = 2.1e6 cm⁻³ (citations still needed).
- Findings 6 and 7, which come from code reading and are not yet measured.
