# P1: DEVSIM tunnelling follow-ons, implementation and verification plan (v0.15 candidate)

Plan only. No code was changed. Reference tree: `DynaMeta_wt/dv_integ`, branch `feat/devsim-2.11-v0.14` at `ee0ad04`.

It covers three deferred items from `docs/audit/2026-11-03-devsim-tunneling.md` section 7:

1. **Self-consistent gate-oxide tunnelling current** (open question 1).
2. **More BTBT parameter sets** (open question 2).
3. **Nonlocal BTBT** (open question 3).

**Status of the claims below.**
* Every claim about current code comes from reading the files named.
* A claim marked **MEASURED** comes from a throw-away probe I ran against the tree on 2026-11-04. The probe is at `scratchpad/probe/probe_custom.py`. It wrote nothing into the tree.
* A literature claim marked **READ** was read in the source named. **SECONDARY** means it was read only as a restatement by another source. **UNVERIFIED** means it was not read at all.
* Two research sub-agents did the literature reads. Their downloads are in `scratchpad/lit_btbt/` and `scratchpad/lit_nonlocal/`.

---

## 0. The enabling mechanism, and what the probe established

All three items need a term that DEVSIM's node/edge/element model system cannot express:

* **Item 1:** the oxide current at a semiconductor interface node depends on the potential at the far (gate) face of the oxide. That is a different region, several elements away. It can also feed a circuit (KCL) row.
* **Item 3:** the rate at a start node depends on the band profile along a path through many elements, and the electrons are deposited at a different node.

DEVSIM 2.11.2 has `devsim.custom_equation(name, procedure)`, a Python callback that adds arbitrary residual entries and Jacobian triplets to the global system (DEVSIM manual section 5.5, READ via devsim.net/models.html):
* The callback signature is `procedure(what, timemode)`, with `what` in {MATRIXONLY, RHS, MATRIXANDRHS} and `timemode` in {DC, TIME}.
* It returns `(rcv, rv, flag)`. `rcv` is a flat list of (row, col, value) triplets, `rv` a flat list of (row, value) pairs.
* `flag` = row permutation: True for bulk rows, False for contact/interface boundary rows.
* Rows come from `get_equation_numbers(device, region, equation=...)` and `get_circuit_equation_number(node=...)`.

**What the probe measured (1-D MOS: n-Si 1e24 m^-3 / 5 nm SiO2, the `mos_cap` fixture geometry).** The probe used a deliberately inflated FN sink (x1e6) with an n_s/(n_s + n_ref) supply factor, added to the semiconductor interface node's electron row.

| Probe | Result (MEASURED) |
|---|---|
| DC Newton with the sink as a custom equation, analytic Jacobian (incl. the column of the gate-face oxide Potential) | converges at every gate step to 6 V |
| Sign | the sink enters the electron row as `-J*Area`; n_s drops (1.7551e27 to 1.7531e27); the body current is -J |
| Charge balance, double | I_body = -1442762019.46034 vs J = 1442762019.46037, i.e. 2e-14 relative |
| Charge balance, extended precision (`extended_precision.enable` + `.solve`) | equal to the last digit. The Newton converges **linearly** (factor ~70 per iteration, rel 7e-30 after ~12 iterations), not quadratically: the callback's values are double |
| `get_matrix_and_rhs` | contains the custom entries (rhs difference = J exactly), so the existing FD-Jacobian gate pattern (J1) applies unchanged |
| Circuit gate (`ac_analysis.setup_circuit_contact`) without a KCL entry | V1.I = 0, so the gate does not see the tunnelling current. Charge is NOT conserved at the terminals |
| Circuit gate with `+J` added to the gate circuit node's row | V1.I = -1442762019.460367 = -I_body; conserved |
| Circuit gate with `-J` | V1.I = +J, a sign error that is not conserved. **A KCL sign test must be a gate** |
| ssac at 1 kHz with the KCL entry | Re Y = -5.551201728e9 vs the DC central-difference dI/dV -5.551201826e9 (1.8e-8). The custom static Jacobian enters ssac |
| Re-registering the same name | allowed. **There is no `delete_custom_equation`**: a registration is process-global and permanent |
| Device deleted while the callback is registered | fine if the callback checks `get_device_list()` itself |

**Consequences for the design. These are shared infrastructure, built once in workstream A (section 4) and used by items 1 and 3.**

1. **One process-wide dispatcher.** A new module, `carriers/custom_terms.py`, registers ONE custom equation named `dynameta_custom_terms`. It does so lazily, on the first opt-in, never at import, so the D1b call-log digest and the D1d/D1c SHA gates stay byte-identical by construction.
   * The dispatcher keeps a registry keyed by device name.
   * Every assembly call loops over live devices and their registered terms, and concatenates the triplets.
   * A device is live only if it is in `get_device_list()` AND carries a DEVSIM marker parameter `_ct_<name>`. Device parameters die with `delete_device`, so a recreated device of the same name cannot inherit a stale term. That is the same "read back from DEVSIM" principle the term inventory follows.
2. **A term is a small protocol object:**
   * `assemble(what, timemode) -> (rcv, rv)`;
   * `rows_live()`, False while its target equations are deleted (Gummel / Poisson-only pre-solve via `eq_registry.delete_by_name`);
   * `noise_sources()`, the two-point shot sources of section 0.3;
   * `jacobian_exact: bool`.
3. **The inventory.** `physics_bipolar_dd.recombination_terms(device, region)` gains a fourth kind, `"custom"`, listing the region's custom terms and their noise capability.
   * The rule "a term without a pair rate makes the noise refuse" carries over: a custom term without `noise_sources` makes `noise_analysis` refuse.
   * So does one whose `jacobian_exact` is False, because ssac/noise ARE the Jacobian.
   * `add_recombination_term` stays the only path for model-expressible terms. A custom term is used only when the term is nonlocal, and that is checked: the constructor refuses a term that declares itself node/edge/element-local.
   * This extends the one mechanism rather than running a parallel one.
4. **Noise needs a two-point source** (new in `noise_analysis`). Today GR sources are same-node pairs, `S_i = q^2 |y_p,i - y_n,i|^2 2P_i`. Both items here move a carrier between two places:
   * oxide tunnelling: electron row at interface node i, to the gate circuit node;
   * nonlocal BTBT: hole row at start node a, to the electron rows of the end element's nodes with barycentric weights w_k.

   The general source is `S = q^2 * 2P * |Σ_r c_r y_r|^2`, a list of (row-adjoint, coefficient) pairs per event. The existing GR pair is the special case {(y_p,i, +1), (y_n,i, -1)}. Gate: re-express the existing BTBT/SRH GR through the new path and require bit-identity with the old one before anything new uses it.
5. **The extended-precision trade-off is now stated.** With a custom term, the 128-bit Newton converges to the double-rounded value of the custom term (MEASURED: conservation then exact). Iterations rise and convergence is linear. `extended_precision.solve` keeps working; document the iteration-count cost, and gate that a converged extended solve exists.
6. **The sparsity pattern must be stable across iterations.** Item 3's paths change with the solution. DEVSIM re-factorises every Newton iteration under UMFPACK. Under MKL-PARDISO, `devsim_env` sets `DEVSIM_NEW_SYMBOLIC=1`, which re-does the symbolic step. A user who overrides that to 0 would reuse a stale symbolic factorisation.
   * Mitigation: emit explicit structural zeros for the candidate pattern (Sentaurus "Strategy 1").
   * Plus a guard that warns if `DEVSIM_NEW_SYMBOLIC` is "0" while a pattern-changing term is active.

---

## 1. Item 1: self-consistent gate-oxide tunnelling current

### 1.1 Goal and physics

Feed the oxide tunnelling current back into the semiconductor electron continuity equation at the oxide interface, and into the gate terminal, so that:
* (a) a floating or deep-depleted body whose minority supply is the tunnelling current is right;
* (b) heavy injection changes the carrier state;
* (c) terminal currents balance;
* (d) ssac/noise see the tunnel conductance and its shot noise.

**The current: two-sided Tsu-Esaki with WKB transmission.**

    J = (q m_s kT / 2 pi^2 hbar^3) Int T(E_x) [ ln(1 + e^{(E_Fs - E_x)/kT}) - ln(1 + e^{(E_Fg - E_x)/kT}) ] dE_x    (1)

* Source: Tsu & Esaki, APL 22, 562 (1973). The form and conventions are as already implemented and documented in `reliability/leakage.py`, gated by K1 (`tests/test_tunneling_kernels.py`) against `scipy.quad` to 6e-11.
* **Caveat:** the primary paper's equation number has not been checked in this round, so it is UNVERIFIED here.
* E_Fs is the semiconductor-side electron quasi-Fermi level at the interface node. E_Fg is the gate Fermi level.
* The existing kernel is the special case E_Fg = E_Fs - qV with an E_F measured from the cathode band bottom.
* The lower integration limit is the higher of the two supply band bottoms (E_c,s at the interface; E_Fg - E_F,metal for a metal gate).
* **(1) vanishes identically when E_Fs = E_Fg**: the integrand is zero pointwise. That is detailed balance by construction, and it is the main reason for the supply decision below.

**Transmission.**
* T(E) = exp(-theta(E)), theta = 2 Int kappa dx, kappa = sqrt(2 m_j (U(x) - E))/hbar, with U piecewise linear across the dielectric layers j.
* Per layer the integral has the closed form already in `wkb_trapezoid_exponent`: `(4 sqrt(2 m_j)/(3 hbar q F_j)) [a^1.5 - b^1.5]`, cancellation-free.
* A stack is the sum over layers, each with its own mass, local barrier height at the layer faces and field F_j = V_j/t_j. The layer drops V_j come from the solved potential at each internal face of the column, so fixed oxide charge, if any, is honoured.
* For a charge-free stack this equals the D-continuity split V_j proportional to t_j/eps_j (a gate, not an assumption).

**Barrier heights by Anderson affinity.**
* The conduction edge of dielectric j along the column is E_c,j(x) = q(Phi_c,s - psi(x)) + (chi_s - chi_j). This is the same affinity rule D1a uses for semiconductor heterointerfaces (`physics_heterojunction` header item 1).
* So a stack's offsets cannot be mutually inconsistent.
* SiO2 keeps the Lenzlinger-Snow fit: phi_b = 3.1 eV against Si, m_ox = 0.42 m0 (J. Appl. Phys. 40, 278 (1969), already cited in code).

**The supply side (the decision): Tsu-Esaki with the local quasi-Fermi level.**
* E_Fs - E_c,s = kT L_n(n_s), with L_n = ln(u) + Delta(u), u = n_s/N_c.
* This is EXACTLY the activity the FD-enhanced SG bulk implies. It is implemented once, in closed form, as a DEVSIM node model by `physics_heterojunction.setup_log_activity`, and read back with its n-derivative. Boltzmann gives Delta = 0.
* In the non-degenerate limit ln(1 + e^x) -> e^x, so J is proportional to n_s: the supply limitation emerges by itself and J -> 0 as n_s -> 0. No ad-hoc n_s/n_ref factor is needed.

**The gate Fermi level.**
* E_Fg = -q V_gate in the library's frame (E_F = 0 at zero bias). V_gate is the bias parameter or the circuit-node value.
* The barrier on the gate side then follows from the gate contact's own Potential offset: phi_Bg = q(Phi_c,s - offset_g) + chi_s - chi_ox. Derivation: E_c,ox at the gate face, from the gate contact's `Potential = V_g + offset_g`, minus E_Fg.
* So the tunnelling barrier is READ from the contact rule, never set separately, and the electrostatics and the tunnelling cannot disagree about the gate work function.
* Today `physics_equilibrium.setup_contact` has offset 0 for gates on a dielectric. That implies a specific W_m, and the module reports it, so the user sees what work function their gate has. (Decision U1-3: a first-class `work_function_eV` on the gate electrode.)

**How the current enters the continuity equation.**
* It is a surface sink at each semiconductor interface node i: `R_i = -J_i * SurfaceArea_i` in the electron continuity row. The sign and units were MEASURED (section 0), with J > 0 for electrons leaving toward the gate.
* The gate side is the circuit row of the gate node: `+J_i * SurfaceArea_i` (KCL sign MEASURED; the opposite sign breaks conservation).
* The hole row is untouched, since this is conduction-band electron tunnelling (ECB) only (Decision U1-4: holes).

**The Jacobian is analytic.** Write V_ox,i = Σ_j w_ij psi_far,j - psi_s,i, where w_ij are the FIXED lateral interpolation weights of the column, computed once from the mesh. For a stack, add the internal-face potentials with their own weights. Then:
* dJ/dpsi_s,i, dJ/dpsi_far,j and dJ/d(internal faces) come through dtheta/dV;
* dJ/dn_s,i comes through dL_n/dn = g(u)/n, read from the activity model's derivative;
* dJ/dV_gate comes through E_Fg (the circuit-node column when circuit-driven, otherwise a parameter, no column).

**Charge conservation.** The dielectric stores no charge from tunnelling in this scope (no oxide trapping). So:
* I_body + I_gate = 0 exactly in steady state;
* in transient, I_body + I_gate + dQ/dt = 0, where the displacement part is DEVSIM's.

**Shot noise.** One-sided S = 2q|J| per area is the T << 1 Landauer-Buttiker limit at large bias. The exact, Nyquist-consistent source for T << 1 is

    S = 2 q^2 (m kT/2 pi^2 hbar^3) Int T(E_x) [N_ss + N_gg - 2 N_sg] dE_x  per area                    (2)

* N_ab = Int_0^inf dE_t f_a f_b over transverse energy, with N_ss = kT ln(1 + e^{(E_Fs - E_x)/kT}), and N_sg in closed form.
* The derivation is from the Landauer-Buttiker f_L(1 - f_R) + f_R(1 - f_L) partition (UNVERIFIED as a citation; derive it in the doc, and gate it by direct 2-D quadrature over E_t).
* At equilibrium (2) = 4kT G_tun. At |qV| >> kT it gives 2q|J|.
* It enters noise as a two-point source: electron row of node i, to the gate circuit row.

**Image-force lowering.**
* For SiO2 it would be sqrt(qE/(4 pi eps_opt eps0)) = 0.26 V at 1 MV/cm and 0.82 V at 10 MV/cm (eps_opt = 2.13; my arithmetic, not a citation).
* That is large. But the shipped barrier parameters are FN-fitted (Lenzlinger-Snow), which absorbs it, so adding image force to fitted parameters double-counts.
* Recommendation: out of scope for v0.15, documented, and refused explicitly (no flag). (Decision U1-5.)

**Quantisation of an accumulation or inversion supply** (2-D subbands; the classical 3-D supply over-estimates) is out of scope and documented. The repo has `schrodinger_poisson`, so it is a possible later coupling.

### 1.2 Current code state, and what is reused (one-kernel rule)

* `carriers/tunneling.py`:
  * `OxideTunnelingParams` (models fn / schuegraf_hu / tsu_esaki; one-sided);
  * `oxide_tunneling_current` (post-processor; planar-interface column drop with lateral interpolation; `SurfaceArea` weights; `NotImplementedError` for non-planar interfaces);
  * `_interface_between`.
* `reliability/leakage.py`: `tsu_esaki_current` / `_tsu_esaki_one` (adaptive composite 12-point GL; breakpoints; theta cut 70) and `wkb_trapezoid_exponent`. These are the ONE home of the WKB/TE math.
* Builders: `LayeredDevsimBuilder(oxide_tunneling=...)` puts the result in `extras["oxide_tunneling"]`; so do `Stacked3DSpec.oxide_tunneling` and `devsim_3d` around line 979.
* Reused, not re-implemented:
  * the column-drop geometry (factored out of `oxide_tunneling_current` into `_column_weights(...)`, which returns sparse weights; the post-processor then uses it, so post and self-consistent share it);
  * the activity L_n (`setup_log_activity`);
  * the frame Phi_c,s (the builders' `band_frames()` / `_own_phi_c`, with the hand-built 1-D fallback `V_t ln(N_c/n_i)`);
  * the contact offset (read from the recorded contact equation kwargs via `eq_registry.recorded`);
  * extended precision (`extended_precision.solve`).

### 1.3 Design (API, defaults OFF)

**Kernel** (`reliability/leakage.py`, additive):
* `wkb_stack_exponent(eps_J, layers, V_faces)` for a stack.
* `wkb_trapezoid_exponent` becomes the one-layer call of it. Gate: bit-identical, or <= 2 ulp with the reason stated.
* `tsu_esaki_two_sided(V_faces, layers, E_Fs, E_Fg, bottom_s, bottom_g, m_s, T_K, *, grid)` returns `(J, S_pair, dJ/dV_faces, dJ/dE_Fs, dJ/dE_Fg)`, vectorised over interface nodes.
* `tsu_esaki_current` is re-expressed through it. Gate: K1-K5 unchanged.
* **Fixed energy grid per device** (decision inside the design, recommended).
  * The current kernel's panel count depends on V, so J(V) has steps of the size of the quadrature error (about 1e-10 relative) where the panel count changes. Newton tolerates that.
  * The FD Jacobian gate and the ssac/DC agreement do not tolerate it.
  * So the self-consistent path uses one energy grid built at setup from the bias range (re-built only when a node leaves it, with a warning). That makes J smooth in V.

**Params.**

```python
@dataclass(frozen=True)
class DielectricBarrier:          # one dielectric of the stack, by Anderson affinity
    chi_eV: float                  # electron affinity; SiO2 default from Lenzlinger-Snow phi_b = 3.1 vs Si chi 4.05 -> 0.95
    m_tunnel_over_m0: float        # SiO2 0.42 (Lenzlinger-Snow)

OxideTunnelingParams(model=..., ..., coupling="post",            # NEW field; "post" = today's behaviour, byte-identical
                     barriers: Mapping[str, DielectricBarrier] | None = None,   # per dielectric region name
                     gate_band_bottom_eV: float | None = None)    # metal gate E_F above its band bottom (no default)
```

* `coupling="self_consistent"` requires `model="tsu_esaki"`. FN and S-H are one-sided and violate detailed balance, so they are refused for self-consistency with a message saying why. (Decision U1-1.)
* `E_F_eV` (the one-sided cathode supply) is not used in self-consistent mode: the semiconductor supply comes from n_s.
* Install: `tunneling.setup_oxide_tunneling(device, semiconductor_region, gate_contact, params, *, dielectric_regions=None)`.
  * It walks the column from the semiconductor interface to the gate contact through the dielectric regions (auto-detected, or given in order).
  * It builds the fixed weights, installs `L_n` if missing, and registers the custom term.
  * It detects whether the gate contact is circuit-driven (`get_contact_equation_command(...)["circuit_node"]`) and only then adds the KCL entries.
* Report: `oxide_tunneling_current(...)` on a self-consistent device returns the same dict. It is evaluated by the SAME kernel call the residual used, plus `"coupling": "self_consistent"` and `"I_gate"`.
* For a DC parameter gate, DEVSIM's `get_contact_current(gate, PotentialEquation)` stays 0 in DC (it carries charge only). The I_gate in the dict is the terminal current, documented. `contact_current.py` and the builder's terminal-current summaries add it, gated.
* Builders:
  * `LayeredDevsimBuilder(oxide_tunneling=OxideTunnelingParams(coupling="self_consistent", ...))` installs at `build_device()` after the contacts.
  * `Stacked3DSpec` likewise.
  * `extras["oxide_tunneling"]` keeps its keys and adds `coupling`.
  * Default `None`: nothing changes.
* Requires `physics="bipolar_dd"` on the semiconductor, otherwise `ValueError`, since there is no electron continuity row to sink into.
* The equilibrium physics mode is refused.
* Refusals: a semiconductor on both sides of a dielectric (a semiconductor gate) is refused in v0.15 (the gate side would need its own continuity row). Poly-Si gates are listed under decisions.

**Composition.**
* *Term mechanism:* custom kind (section 0). It is independent of the volume slots, so it composes with SRH/traps/BTBT/interface traps on the same node.
* *Interface traps (D1c):* their node terms at the same interface nodes sum. A gate re-runs the composition fixture with oxide tunnelling added.
* *Contact rule:* read, not changed (gate offset). The new `work_function_eV` (if approved, U1-3) goes INTO `contact_potential_residual`'s companion for gates. That is the one place.
* *Heterointerfaces:* not affected (semiconductor/dielectric only).
* *Extended precision:* works (MEASURED), at linear convergence.
* *1-D / 2-D / 3-D:* one code path. Only the weight builder differs (1-D: scalar; 2-D: `np.interp` weights; 3-D: barycentric weights from a Delaunay of the far face). The 3-D tensor (Kuhn) mesher is the default since D2c, and is required for the cross-dimension gate.

### 1.4 Ordered implementation steps

1. **Infrastructure (section 0):** `custom_terms.py` dispatcher, marker parameter, inventory kind, noise two-point source, refusal rules. Gate C0 below, before any physics.
2. **Kernel:** `wkb_stack_exponent`; `tsu_esaki_two_sided` (fixed grid; analytic derivatives; pair-noise integrand (2)); re-express the old functions; run K1-K5 plus the new K6-K10.
3. Factor the column geometry out of `oxide_tunneling_current` into `_column_weights`, with the post-processor using it. Then run gates F1-F4/W1/W3 unchanged.
4. The 1-D self-consistent term (electron row only, DC parameter gate). Gates O1-O7.
5. The circuit gate (KCL entries) plus ssac. Gates O8-O9.
6. Noise: the shot source through the two-point path. Gates O10-O11.
7. Multi-dielectric stacks. Gates O12-O13.
8. The 2-D layered builder and the 3-D builder. Gates O14-O15.
9. Docs: tunnelling doc section 9, release notes, and the gate-workfunction report.

### 1.5 Verification gates

Each gate states: its **oracle**; its **tolerance**, with the derivation; **old code**, i.e. how it fails on today's tree; and **refute**, the adversarial attempt to break a passing result.

**C0. Infrastructure.** Placed in `tests/test_custom_terms.py`, fast.
* (a) **Defaults off.**
  * The D1b call-log digest and the 16-array SHA (`a500ca93...`) are unchanged with the new modules imported.
  * The custom equation is NOT registered after import. Check `custom_terms._registered is False`. DEVSIM has no listing call, so also check from a child process that registration happens only on the first `add`.
  * Old code: not applicable (identity gate).
  * Refute: run the SHA fixture in a child process that registers the dispatcher with no terms. It must give the same SHA. Otherwise the mere registration perturbs DEVSIM, and the plan must register per opt-in device only.
* (b) **Liveness.** Delete a device and re-create it with the same name, then solve. The term must not apply: the residual equals the no-term residual bit-for-bit.
  * Refute: same device name, different region names. Must not crash and must not apply.
* (c) **Frozen continuity.** `eq_registry.delete_by_name(ElectronContinuityEquation)` gives a Poisson-only solve that converges with the term registered. The callback returns nothing.
* (d) **The two-point noise path reproduces today's GR noise bit-for-bit** on the D2a FDT diode with BTBT.
  * Oracle: the current `_gr_pair_per_node` result.
  * Tolerance: 0 ulp, or 1 ulp if the summation order changes, with the order stated.

**K6-K10. Kernel**, in `tests/test_tunneling_kernels.py`, fast.

| Gate | Oracle | Tolerance (derivation) | Old code | Refute |
|---|---|---|---|---|
| K6 two-sided TE == scipy.quad of (1), 8 cases: DT/FN, 0/300 K, metal/semi bottoms, E_Fs above/below E_c | independent `quad` with the WKB exponent re-coded in the test (as K1) | 1e-8 (K1's gate; the fixed grid is tuned so that measured <= 1e-10) | no function | grid at 1/2 and 2x density must move J by < the tolerance (else the grid choice is load-bearing) |
| K7 identity with the old one-sided kernel when E_Fg = E_Fs - qV | `tsu_esaki_current` | 1e-13 (same integrand, different grid; the bound is the quadrature-error difference, measured) | n/a | sweep V across panel-count changes of the old kernel: the new one must be smooth (second difference / J < 1e-9) where the old one steps |
| K8 detailed balance: E_Fs = E_Fg gives J == 0.0 exactly; J is odd under swapping the two sides with a symmetric barrier | algebra | exact 0.0 / 1e-14 | n/a | random asymmetric stacks: the zero must remain exact (pointwise zero integrand) |
| K9 analytic derivatives (dV_faces, dE_Fs, dE_Fg) | 4th-order central differences with Richardson; and quad of the differentiated integrand | 1e-8 (h^4 truncation at h = 1e-4 relative is about 1e-14; roundoff eps J/h is about 1e-12) | n/a | at the barrier-top breakpoint (V = phi) the derivative is continuous (left/right agree to 1e-6); a kink there would break Newton and the FD Jacobian gate |
| K10 stack WKB: 1 layer == trapezoid; two identical layers == one of summed thickness; three-layer stack == `quad` of 2 Int kappa dx with `brentq` turning points; charge-free stack drops V_j proportional to t_j/eps_j | closed forms / quad | 1e-14 / 1e-14 / 1e-10 / 1e-13 | no function | a layer whose barrier dips below E (a well) must be handled (theta piece = 0 there), checked against quad |
| K11 pair-noise integrand (2): 2-D quadrature over (E_x, E_t) of f_s(1 - f_g) + f_g(1 - f_s); equilibrium S == 4kT dJ/dV; qV >> kT gives S -> 2qJ | scipy `dblquad`; the K9 derivative | 1e-8 / 1e-8 / 1e-3 at qV = 40 kT (the correction is e^-40, but the T << 1 partition neglect is T(1 - T): the gate prints max T) | n/a | must FAIL if N_sg is dropped (the one-sided S = 2q(J_f + J_b) overcounts at equilibrium by a factor up to 2) |

**O1-O15. Device**, in `tests/test_devsim_oxide_tunneling_sc.py`. The 1-D gates are fast (each fixture < 10 s); 2-D/3-D are marked slow.

* **O1. Detailed balance in the device.**
  * Setup: 1-D n-MOS and p-MOS at zero bias, with the gate offset set so a nonzero oxide field exists at equilibrium (work-function difference), and |E_ox| ~ 5e8 V/m.
  * Pass: |J_net| / J_one-way <= 1e-10.
  * Tolerance derivation: E_Fs carries the discrete-equilibrium residual of L_n, measured about 1e-15 relative for Boltzmann. That gives J/J_one-way ~ dL ~ 1e-13. Gate it at 1e-10.
  * With the FD g-factor, the D1d L1 addendum found the discrete FD equilibrium off mass action at O(h^2). Measure first. If L_n is not flat to 1e-10 under FD, the gate becomes a convergence gate in h (order >= 1.7), stated as such.
  * **Old code (prove-out):** the post-processor gives nonzero J at this equilibrium (FN/S-H one-sided). Assert J_post / J_one-way > 0.5.
  * Also assert that an n_s/n_ref-scaled FN sink gives nonzero equilibrium current. This is the documented reason the alternative was rejected.
  * Refute: flip the gate polarity and change the work function by +-0.3 eV. Still 0.
* **O2. Sign and conservation.**
  * Pass: I_body + I_gate = 0 within the Z3 floor, 32 ulps x q mu V_t N/h of the majority SG term + 1e-4 |J|.
  * Extended precision: <= 1e-25 |J| (MEASURED in the probe: exact).
  * Old code: I_body ~ 0 (post-processed), so the gate asserts |I_body| > 0.5 |J_post|.
  * Refute: reverse the gate. Both currents change sign, conservation still holds.
* **O3. Weak-leak limit = the post-processor.**
  * Setup: scale the barrier mass (or thicken the oxide) so that J/J_transport spans 1e-12 .. 1e-6.
  * Pass: J_sc/J_post - 1 is proportional to the leak, as a first-order perturbation. The fitted log-log slope is 1 +- 0.05, and the intercept extrapolates to 0 within 1e-9.
  * J_post here is the two-sided kernel evaluated on the frozen ideal-capacitor solution.
  * Old code: no SC mode. The test also pins that `coupling="post"` reproduces `oxide_tunneling_current` bit-for-bit.
  * Refute: if the slope is 0 (no feedback) or 2, the coupling is wrong.
* **O4. Supply-limited inversion: the case the post-processor gets wrong.**
  * Setup: p-Si (1e23 m^-3), 2 nm SiO2, inversion bias. SRH lifetime tau set so that the thermal generation current q n_i W/(2 tau) is << J_post.
  * Pass, part 1: in steady state, I_gate equals q Int G_SRH dV over the semiconductor (continuity; Z3-type floor).
  * Pass, part 2: I_gate equals the analytic generation-limited estimate q n_i W_dd(V)/(2 tau_eff), with W_dd the deep-depletion width from the depletion approximation.
  * Tolerance: 5 %, the depletion approximation plus the generation-region width (derive it in the test from the n p << n_i^2 region extent). Measure first, then fix the tolerance from the measured preasymptotic error, as L1 did.
  * Old code: the post-processor gives J_post >> the generation limit (assert > 10x). This is the load-bearing physical gate.
  * Refute: shorten tau by 10x. I_gate rises by 10x until it reaches J_tunnel-limited. Assert the cross-over is monotone and that I_gate <= min(generation, J_post) at every point.
* **O5. Positivity.**
  * Setup: an extreme gate bias that drives n_s down by more than 20 decades.
  * Pass: Newton converges, n_s > 0 everywhere (`variable_update="positive"`), and J -> 0 proportionally to n_s (Boltzmann).
  * Old code: n/a.
  * Refute: start from a seed with n_s 1e10 too large.
* **O6. Jacobian.**
  * Pass: the custom term's share of `get_matrix_and_rhs` (on minus off) equals the central FD of DEVSIM's residual (the J1 pattern) to 1e-6, in 1-D, 2-D and 3-D, including the oxide-face columns and (with a circuit gate) the circuit column and row.
  * Refute: perturb ONLY the far-face Potential (a column that a local model would miss). The FD must see it and match.
* **O7. Newton quality.**
  * Pass: on the O4 sweep, the iteration count with the analytic Jacobian is <= the no-tunnelling count + 3 per bias step. Quadratic convergence is observed: the log of the update norm halves the exponent per iteration over the last 3 iterations.
  * This exists to catch a "converges but with a wrong Jacobian" regression that O6 might miss on a frozen state.
* **O8. ssac.**
  * Setup: circuit gate.
  * Pass: Re Y(f -> 0) == DC dI/dV (central difference +-0.1 mV, Richardson) to 1e-6. MEASURED 1.8e-8 in the probe.
  * Pass: Im Y equals the no-tunnelling C within (G/omega C) effects at 1 MHz on a thick oxide.
  * Old code: Re Y ~ 0 (MEASURED -4.4e-9 without the KCL entry).
  * Refute: KCL sign flipped must fail O2 and O8 (MEASURED: it flips V1.I).
* **O9. Transient charge balance.**
  * Setup: a gate step on the circuit gate.
  * Pass: Int (I_gate + I_body) dt = 0 to the BDF1 truncation, gated by halving dt (order 1).
  * Pass: the final state equals the DC solve to 1e-10.
* **O10. Noise FDT at equilibrium** (circuit gate, 1 Hz - 1 GHz).
  * Pass: S_I = 4kT Re Y with the shot source (2) through the two-point path, to 1e-4 (the D2a gate).
  * Old code: no source.
  * **Prove-out:** dropping the source fails the FDT, and using the one-sided 2q(J_f + J_b) fails it.
* **O11. Shot limit.**
  * Setup: |qV_ox| >> kT, tunnelling-dominated gate current.
  * Pass: S_gate -> 2q|I| within the T(1 - T) and diffusion-noise contributions. Compute both and gate at 1 %.
* **O12. Two-layer stack in 1-D** (SiO2 1 nm / HfO2 3 nm).
  * Pass: J equals the kernel evaluated on the solved face potentials to 1e-12.
  * Pass: per-layer Gauss law eps_j E_j continuous to 1e-8 (as F1).
  * Pass: when HfO2 is replaced by "SiO2-as-two-regions", J equals the single-layer device to 1e-10.
  * Refute: swap the layer order. J changes, and conservation still holds.
* **O13. Published stack data (nightly, validation).**
  * Candidate oracle: a published J-V of a SiO2/high-k stack with full parameters.
  * **No source is in hand.** This gate is listed as an open item (U1-6), not promised.
* **O14. 2-D layered, laterally uniform.**
  * Pass: per-depth J / (cell width) equals the 1-D per-area J to 1e-6.
  * Pass: the conservation O2.
  * Pass: the O6 Jacobian.
  * Use the clean tensor x-mesh. Note: the D1d doc says the layered builder wires the interface only from x = 2.5 nm, so the gate uses the wired width.
* **O15. 3-D tensor mesher, laterally uniform (slow, ~40 s).**
  * Pass: equals 1-D to 1e-6. The tensor mesher is exact to 9e-16 per D2c, and the 1.5 % lateral V_ox spread of D1d was on gmsh tets.
  * With `mesher="gmsh"` the gate must warn, not pass silently. A 1.5 % V_ox spread is a ~37 % J spread at b = B/E ~ 25.
* **Defaults off.**
  * The SHA (16 arrays), the call-log digest, W1's "no key", and `coupling="post"` is bit-for-bit today's post-processor.

**Convergence-order gates.**
* Semiconductor mesh h at the interface: J_sc at h, h/2, h/4. Observed order >= 1.7, since the activity and the SG flux are second order. Gate it like L1, with C_inf from a clean uniform study.
* Oxide mesh: J must be independent of the oxide spacing, because the column drop is exact for a linear potential. Gate: 1e-12 across 3 oxide meshes.

**Runtime.** Fast set (K6-K11, C0, O1-O12 1-D): measure, then budget about 60-90 s total. Slow/nightly: O14, O15, O13.

### 1.6 Risks and unknowns

* **Callback performance in 2-D/3-D.** One vectorised TE evaluation per interface node per assembly. About 300 nodes x a fixed grid of ~2000 energies is ~6e5 integrand evaluations, about 10-50 ms. Acceptable, but measure in step 8.
* **Newton robustness under heavy FN.** The exponential sink near the barrier top may need damping. `variable_update="log_damp"` is on Potential only. Fallback: a bias-continuation helper. Measure on the O4 fixture.
* **The DC gate current is not in DEVSIM's contact current.** Every terminal-current consumer must use the module's I_gate. An audit is needed of `contact_current.py`, the builders' summaries, and any "sum of terminal currents" check.
* **Circuit process-globality** (integration note section 7, MEASURED by others): circuits cannot be removed in devsim 2.11.2. The self-consistent mode must not CREATE circuits. It uses the user's circuit gate only when present.
* **FD statistics vs detailed balance** (O1 note). It may need the convergence-gate form.
* **Semiconductor-gate (poly) and hole tunnelling are excluded.** p-MOS accumulation (HVB) would then be wrong if users enable it there. The module must refuse hole-dominated configurations? A cheaper alternative: warn when p_s > n_s at an interface node with a large |E_ox|.

---

## 2. Item 2: more BTBT parameter sets (cited only)

### 2.1 Goal and physics

Ship additional local-model sets for `U = A |E|^P exp(-B/|E|) D` (Kane: P = 2 direct; P = 2.5 phonon-assisted). Each must carry a citation, and each must have tests that pin every constant to its printed value and units.

The Kane closed form, for computing A and B from cited material constants:

    A = g pi m_r^{1/2} (q F0)^2 / (9 h^2 Eg^{1/2}),   B = pi^2 m_r^{1/2} Eg^{3/2} / (q h) = pi m_r^{1/2} Eg^{3/2}/(2 q hbar)     (Sentaurus UG N-2017.09 Eqs. 495-496, p. 439)

* F0 = 1 V/cm and 1/m_r = 1/m_c + 1/m_v. With g = 2 this is Kane's q^2 m_r^{1/2}/(18 pi hbar^2 Eg^{1/2}).
* READ in the Sentaurus manual (an unofficial hosted copy). Kane, J. Appl. Phys. 32, 83 (1961) itself is UNVERIFIED (not read).
* Also read in Das, Broderick & O'Reilly, arXiv:2108.09328, Eq. (13).
* The phonon-assisted A (Eq. 497) needs D_op, rho and eps_op, so it is not a practical closed form. Phonon-assisted sets ship only as published fits.

**Temperature.**
* Sentaurus Eq. 478 (READ): B scales as (Eg(T)/Eg(300 K))^{3/2} and A is constant.
* Hurkx 1992 itself is SECONDARY: the attribution is via Sentaurus and US patent 10,769,339.
* Today the code fixes T_REF. The plan is to add this scaling opt-in via `BipolarOptions.T_K`. It needs an Eg(T) from a cited Varshni set. The repo already has `optics/soa/temperature.VARSHNI_PARAMS`, which is III-V only, cited to Vurgaftman 2001. Reuse that function; add Si/Ge rows only with a cited source (Thurmond 1975 is the usual one; UNVERIFIED).

### 2.2 Literature table: what was found, and its status

SI conversion: `A_SI = A_cgs 1e6 100^-P`, `B_SI = 100 B` (`BTBTParams.from_cgs`).

| Set | A (as printed) | B | P | Primary | Status |
|---|---|---|---|---|---|
| Si, Hurkx (shipped) | 4e14 cm^-3 s^-1 (F in V/cm) | 1.9e7 V/cm | 2.5 | Hurkx, Klaassen, Knuvers, TED 39, 331 (1992) | **SECONDARY.** Hurkx is paywalled and was unreadable both in D1d and now. Confirmed in Sentaurus UG Table 90 / Eq. 478 text ("to use Hurkx values") and Atlas BBT.KL. D1d open question 4 stands |
| Si direct, Kao 2012 [100] | 1.35e20 cm^-3 s^-1 | 101 MV/cm | 2 | Kao, Verhulst, Vandenberghe, Soree, Groeseneken, De Meyer, IEEE TED 59, 292 (2012), Table III | **SECONDARY**, via Jiao et al., J. Semicond. 34, 092002 (2013) Table 1 (open access, READ). The primary is paywalled |
| Si indirect, Kao | 3.29e15 | 23.8 MV/cm | 2.5 | same | SECONDARY |
| Ge direct, Kao | 1.46e20 | 6.04 MV/cm | 2 | same | SECONDARY. **The Kane formula with Eg = 0.80 eV, m_r = 0.022 gives B = 6.04 (match) but A = 1.61e20 (+10 %).** Unexplained: Kao may use a different Eg, g or m_r. Needs the primary |
| Ge indirect, Kao | 1.67e15 | 6.55 MV/cm | 2.5 | same | SECONDARY. Kao says Ge needs direct + indirect SUMMED |
| Ge, Sentaurus D-2010.03 default | 9.1e16 | 4.9 MV/cm | 2.5 | vendor default (Jiao Table 1) | SECONDARY, vendor. Not recommended |
| Si, Sentaurus E2 / Hurkx default | 3.4e21 cm^-1 s^-1 V^-2 | 22.6 MV/cm | 2 | Sentaurus UG Table 89 (refs Liou, SSE 33, 971 (1990) for E1) | READ (vendor manual). Not recommended unless the user wants vendor parity |
| Atlas BBT.STD | 9.6615e18 (units misprinted in the manual) | 3.0e7 V/cm | 2 | Keeney et al., IEDM 1990 p. 201 | SECONDARY with a units misprint, so **not shippable** |
| In0.53Ga0.47As | none found as a primary local fit | | 2 | Smets et al., JAP 115, 184503 (2014), a NONLOCAL calibration (paywalled) | **No shippable primary.** The 1.3e20 / 5.7 MV/cm in Chen et al., Nanotechnology 33, 225205 (2022) is untraceable. Do not ship |
| In0.53Ga0.47As, computed | from the Kane formula | | 2 | Eg(300 K), m_e, m_lh from Vurgaftman, Meyer, Ram-Mohan, JAP 89, 5815 (2001) | **Computable, UNVERIFIED inputs.** The sub-agent's masses are from memory. [RESOLVED 2026-11-06, v0.14.0: the repo's old Varshni row (0.816 eV, 2.9e-4, 193 K, Eg(300) = 0.763 eV) was wrong; the SOA module now uses Vurgaftman 2001's alloy rule (binary Varshni + C = 0.477 eV), Eg(300) = 0.737 eV -- use that one function (docs/audit/2026-11-06-v0.14-ingaas-gap.md)] |

### 2.3 Design

* `tunneling.py` presets, each a `BTBTParams` with two NEW optional fields: `source: str` (full citation, table/equation, page) and `process: "direct" | "phonon_assisted"`. The fields are excluded from equality/hash use in physics, so `SILICON_HURKX` stays the same object semantically.
* New presets, behind decision U2-1:
  * `SILICON_KAO_DIRECT`, `SILICON_KAO_INDIRECT`, `GERMANIUM_KAO_DIRECT`, `GERMANIUM_KAO_INDIRECT`;
  * the tuple `GERMANIUM_KAO = (direct, indirect)`;
  * optional `IN053GA047AS_KANE` (computed), only after the Vurgaftman inputs are READ.
* `kane_direct_params(Eg_eV, m_r_over_m0, g=2.0, *, source) -> BTBTParams` is the one function that computes Kane sets. A computed preset is a call of it, never a pasted number.
* **Several processes per region.**
  * `setup_btbt(..., name="btbt")`: today's name is the default, so the model names are byte-identical.
  * `setup_bipolar_region(btbt=...)`, `TransportModel.btbt` and `Stacked3DSpec.btbt` accept a `BTBTParams` OR a tuple of them.
  * A tuple installs `btbt_0`, `btbt_1`, ... as separate terms through `add_recombination_term`. They SUM (the mechanism already sums distinct names), each with its pair rate, so noise covers them automatically.
  * The diagnostics get the suffix (`BTBT_U_0`, ...). `btbt_generation_current` sums every BTBT term, or takes `name=`.
* **T scaling:** `BTBTParams.eg_of_T: Callable | None = None`. When set and `BipolarOptions.T_K` differs from T_REF, B -> B (Eg(T)/Eg(300))^{1.5}. Default None: today's behaviour. (Decision U2-3.)

### 2.4 Steps

1. Get the primaries: Kao 2012 (IEEE, needs the user's library access, or the imec repository after any embargo), Vurgaftman 2001 (AIP), and ideally Hurkx 1992.
2. Add the fields, presets and `kane_direct_params`, plus the multi-term threading.
3. Tests and the physics sanity gate.
4. Docs: a parameter table with the exact citation per row, and a units note (the Hurkx "cm^-1/2 V^-5/2 s^-1" and the Sentaurus "cm^-3 s^-1 with F/1 V/cm" forms are the same numbers; the Atlas misprint is explained).

### 2.5 Verification gates

All in `tests/test_tunneling_presets.py`, fast (< 30 s).

* **P1. Citation pin.** For every preset, A_SI, B and P equal the printed value hand-converted in the test (written out as literals with the printed units in a comment) to 1e-15. `source` contains the table/equation id.
  * Old code: the presets do not exist.
  * Refute: a deliberately wrong unit (cm^-1 instead of cm^-3 for A) must change A_SI by 1e4 and fail.
* **P2. Kane function.**
  * `kane_direct_params` equals an independent re-coding in the test from CODATA (scipy.constants) to 1e-13.
  * The uniform-field rate at 1e8 V/m equals Kane's q^2 F^2 m_r^{1/2}/(18 pi hbar^2 Eg^{1/2}) exp(-pi m_r^{1/2} Eg^{3/2}/(2 q hbar F)) to 1e-13.
  * Refute: g = 1 must give exactly half.
* **P3. Kao internal consistency (informational, then a gate once the primary is read).**
  * B(Kane; Eg, m_r as printed in Kao) == Kao's B to 1 %, and A to the precision Kao prints.
  * Today this gives a 10 % A mismatch for Ge direct (SECONDARY numbers). **The gate cannot be finalised until the primary is read.** Recorded as an xfail(strict) with that reason until then.
* **P4. Wiring and units per preset (physics sanity).**
  * Fixture: the existing 1-D Zener (Z1 machinery, `_zener_analytic` generalised to material eps, n_i, P, A, B) with each preset's material: Si eps 11.7, n_i 1e16; Ge eps 16.0, n_i 2.0e19 m^-3 (cite the Ge values, e.g. Sze & Ng Appendix; UNVERIFIED here).
  * Pass: q Int G vs the analytic depletion estimate, within 1 % + delta_D (Z1's derived tolerance).
  * It catches a unit or P mix-up. A 100x unit error misses by 100x; a P = 2 vs 2.5 swap misses by a factor E^0.5 ~ 1e4.
  * Old code: no presets.
* **P5. Composition.** Ge direct + indirect as a tuple equals the sum of the two single-term runs at a FROZEN state (exact), and the solved current is >= each alone. Noise pairs are present for both (`recombination_terms`). The FDT diode with two BTBT terms passes at 1e-4 (the D2a gate).
* **P6. Ordering sanity** at a fixed |E| = 3e8 V/m:
  * U(Ge direct) > U(Si indirect) by more than 10^3;
  * the direct share grows with field for Ge;
  * every preset's rate is monotone in E.
  * Refute: none here. These are coarse sanity checks, labelled as such.
* **P7. Defaults off.** `SILICON_HURKX` unchanged (bit-equal A_SI/B/P). `btbt=SILICON_HURKX` gives byte-identical model strings and names to today (the D1d SHA fixture).
* **P8. T scaling (if U2-3 is approved).**
  * At T = 300 K, B is unchanged bit-for-bit.
  * At 350 K, B equals B (Eg(350)/Eg(300))^1.5 with the cited Varshni Eg, to 1e-14.
  * Refute: A must NOT change.

### 2.6 Risks

* The primary sources are paywalled. Shipping "via Jiao 2013" is honest but weaker.
* The Ge-direct A discrepancy is unexplained.
* The repo's InGaAs Varshni row may be off by about 23 meV in Eg(300). That changes the Kane B by about 5 %. Check it against Vurgaftman before anything uses it for BTBT. If it is wrong, it also affects the SOA module, so that is a separate finding to report rather than fix silently.

---

## 3. Item 3: nonlocal BTBT

### 3.1 Formulation survey (from the primary or vendor texts that were READ)

| Formulation | Local/nonlocal | Read? | Fit for DEVSIM |
|---|---|---|---|
| Hurkx 1992 | local (Sentaurus Eq. 478/479 restatement). The "average field over the tunnelling length" variant is **unconfirmed**: the paper is unread | SECONDARY | already shipped (local) |
| Schenk, SSE 36, 19 (1993) | **local**: Eq. (100) with critical fields Eq. (101), phonon absorption/emission. The nonuniform-field generalisation (action integrals S_c, S_v) is only sketched, Eqs. (85)-(87), (90)-(91) | READ (author copy) | expressible as a node/element model, but it is not the nonlocal item |
| Average field F = Eg/l_tun | quasi-nonlocal: Vandenberghe et al., JAP 109, 124503 (2011) = arXiv:1104.3713, Eq. (102); Pan & Chui JAP 116, 054508 (2014) | READ (Vandenberghe); abstract only (Pan & Chui) | needs the path length l, so the same path machinery as the full model, but a simpler rate |
| **Sentaurus dynamic nonlocal path** | nonlocal. Path along -grad E_V from each vertex to the point where E_C + Delta_C = epsilon; holes at the start, electrons at the end; occupancy f_n(l) - f_p(0); direct Eqs. 480-484 (two-band kappa), phonon-assisted Eqs. 490-493 (Keldysh kappa_V/kappa_C switched at kappa_V = kappa_C); uniform-field limits Eqs. 494-498 | READ (N-2017.09, unofficial copy, pp. 431-443). **Eq. 491's integral structure was transcribed from an image and is uncertain** | needs Python path tracing (custom term) |
| Silvaco Atlas BBT.NONLOCAL | nonlocal: 1-D slices on a separate QT mesh, per-energy Tsu-Esaki (Eqs. 3-379 to 3-383), two-band kappa; Jacobian terms off by default (BBT.NLDERIVS) | READ (2006 manual) | needs a second mesh; heavier than the path model |
| Charon (Sandia) v2.2 | local only; says nonlocal is hard under MPI | READ | n/a |

**Recommendation.** Use the **dynamic nonlocal path model, direct process (Eqs. 480-484)** as the first and fully specified target:

    R^d(x0) = |grad E_V(0)| C_d exp(-2 Int_0^l kappa dx) [f_n(l) - f_p(0)]                         (Eq. 480)
    C_d = (g pi / 36 h) (Int_0^l dx/kappa)^-1 [1 - exp(-k_m^2 Int_0^l dx/kappa)]                     (Eq. 481)
    kappa = (1/hbar) sqrt(m_r Eg,tun (1 - alpha^2))                                                (Eq. 482)

Implementation choices:
* **k_m -> infinity** (the transverse cutoff factor -> 1).
* **The m0 -> infinity form of alpha:** alpha = 2(epsilon - E_V)/Eg - 1, so kappa = (2/hbar) sqrt(m_r (epsilon - E_V)(E_C - epsilon)/Eg).
* With those two choices the uniform-field limit is EXACTLY A F^2 exp(-B/F) with A and B of Eqs. 495-496. I re-derived it: Int over the path of sqrt(u(1 - u)) is pi/8, which gives B; Int dx/kappa gives A.
* The full Eq. 483 (finite m0) and the k_m cutoff of Eqs. 485-487 break that identity. They are later options, each with a gate quantifying the departure.

The **phonon-assisted process** ships second, in one of two ways:
* (a) Eqs. 490-491, after Eq. 491 is re-read against the page image (`lit_nonlocal/img/sd-0485.png`; the sub-agent flagged a stray dx and an outer integral it could not place);
* or (b) an "average-field prefactor" form, R = A F_avg^{P} exp(-2 Int kappa_{V/C} dx) [f_n - f_p] with F_avg = Eg/(q l).

Option (b) reproduces A F^{2.5} exp(-B/F) exactly in a uniform field. That is my derivation: with Keldysh kappa_V up to x0 = m_C l/(m_V + m_C) and kappa_C beyond, 2 Int kappa = (4/3) sqrt(2 m_r) Eg^{3/2}/(q hbar F), which is Eq. 498. But (b) is our own construction, not a published model. (Decision U3-2.)

**A premise check the user should see before funding item 3.** The doc's motivation is "the local field overestimates in abrupt junctions". The literature read is mixed:
* Vandenberghe 2011 (READ): all semiclassical models over-estimate near V ~ 0, and the uniform-field (average-field) model is the best semiclassical one.
* Jiao 2013 (READ): Si homojunctions give nearly the same current for Hurkx-local and nonlocal; nonlocal is HIGHER in Si-Ge heterojunctions above about 2 V.
* Wong et al. 2020 (abstract only): Hurkx UNDER-estimates at large bias.

So nonlocal is not "lower" in general. What it robustly fixes:
* (i) **No final state:** the local model generates wherever |E| is high, even where the total band bending over the tunnelling distance is < Eg. The nonlocal model gives exactly zero there.
* (ii) Heterojunctions, where the gap varies along the path.
* (iii) Generation placed at the path ends (electrons and holes spatially separated), which matters for TFETs and GIDL.

### 3.2 Current code and reuse

* Local BTBT in `tunneling.py`: `btbt_rate_expr`, `hurkx_d_expr`, `setup_btbt`; the element-field machinery.
* The term mechanism and noise pair rates.
* Band edges in the region frame: Phi_c, Phi_v, BGN Delta_C / Delta_V via `_bgn_interface_arguments`, and the Anderson s(r).
* Quasi-Fermi activities L_n, L_p (`setup_log_activity`) for f_n and f_p. These are Boltzmann/FD-consistent with the bulk, so f_n(l) - f_p(0) = 0 at the discrete equilibrium exactly.
* Reused unchanged: `BTBTParams` (A, B, P are the "fitted set"; conversion to m_r and g by inverting Eqs. 495-496 is a new function in the same module); the section 0 dispatcher; the two-point noise source.

### 3.3 Design

* `setup_btbt_nonlocal(device, region, params: BTBTParams, *, process="direct", max_length_m=20e-9, f_min_V_m=None, direction=None, jacobian="exact")`.
  * `NonlocalBTBT` options; defaults OFF. `TransportModel.btbt_nonlocal` / `Stacked3DSpec.btbt_nonlocal`, mutually exclusive with local `btbt` in one region (refused: double counting).
  * `f_min` defaults to B/60, below which exp(-B/F) < e^-60, as the start-node screen.
  * `direction` freezes the tunnelling direction (Sentaurus option); needed for an exact Jacobian in 2-D/3-D (below).
* **Discretisation.**
  * **Start nodes:** semiconductor nodes with |grad E_V| > f_min. The nodal gradient is the volume-weighted mean of the element P1 gradients (2-D/3-D) or the edge mean (1-D).
  * **Path:** a straight ray along -grad E_V. It walks elements by barycentric coordinates.
  * Along each element segment, E_C and E_V are linear (P1). So the end point E_C(x) = epsilon is an exact linear root per segment, and the segment integrals of kappa (sqrt of a quadratic in s) and of 1/kappa (arcsin form) have CLOSED forms. There is no quadrature error, so the uniform-field identity can be gated at 1e-12.
  * **Ray ends:** on reaching an insulator, contact or Neumann boundary, or max_length, the path is discarded and counted in a diagnostic.
  * Specular reflection at semiconductor/insulator interfaces (the Sentaurus default) is a later option. (Decision U3-4. GIDL needs it.)
  * **Holes:** R x NodeVolume at the start node's hole row.
  * **Electrons:** the same amount, deposited into the end element's nodes with barycentric weights. The two integrals are equal by construction, so the discrete continuity is exact (gate).
  * The pair rate for noise is C T [f_p(1 - f_n) + f_n(1 - f_p)]; the source goes through the two-point path.
* **1-D first.** The path is the x-axis, and the walker is a monotone search on the node list (vectorised numpy). 2-D second. 3-D is deferred until the 2-D cost is measured (U3-3).
* **Jacobian.** R depends on:
  * psi at every node of every element the path crosses (through E_C, E_V in the segment integrals);
  * the end-point location (implicit: d l/d psi from E_C(l) = epsilon);
  * epsilon = E_V(x0) (the start node);
  * n at the end element's nodes and p at the start node (Fermi factors);
  * the direction (through grad E_V at the start node).
* **1-D: exact**, with no direction dependence. The analytic derivative of each closed-form segment integral, plus the implicit end point.
* **2-D/3-D:** exact for every dependence EXCEPT the direction. With `direction=` frozen it is exact. With dynamic direction, the missing term is the rotation of the ray. That is an approximate Jacobian: DC Newton still converges to the right solution (the residual is exact), but **ssac/noise would be wrong**. So `jacobian_exact=False` there, and ssac/noise refuse the device with a message (section 0 rule). This is the honest version of Sentaurus' "derivatives off by default".
* **Sparsity:** the candidate pattern is all nodes within max_length of each start node, emitted as structural zeros (section 0 item 6).
* **Performance (to measure; the estimates are mine):**
  * 1-D: about 1e3 start nodes x about 1e2 segments, numpy, < 50 ms per assembly.
  * 2-D: about 2e4 start nodes x about 50 element visits. A Python loop is too slow (seconds per assembly).
  * Write the 2-D walker as a numba kernel with a numpy reference path (numba is an optional extra, and CI has a numba leg). Without numba, 2-D nonlocal raises with an explanation rather than running for minutes.

### 3.4 Ordered steps

1. Fitted to physical conversion `(A, B) -> (m_r, g)` (inverting Eqs. 495-496), plus the closed-form segment integrals and their derivatives. Kernel gates N0.
2. The 1-D path model (direct), as a custom term with an exact Jacobian. Gates N1-N7.
3. Noise and ssac in 1-D (N8-N9).
4. The phonon-assisted process (after U3-2) in 1-D (N10).
5. The 2-D walker (numba plus a numpy reference), frozen and dynamic direction (N11-N13).
6. The nightly validation against published data (N14).
7. 3-D only if approved.

### 3.5 Verification gates

In `tests/test_btbt_nonlocal.py`; 2-D is slow.

* **N0. Kernel.**
  * Closed-form segment integrals Int kappa ds and Int ds/kappa equal `scipy.quad` to 1e-12 for random linear band pieces, including pieces that touch a turning point (kappa -> 0 at an end, with the integrable 1/sqrt singularity handled by the arcsin form).
  * Derivatives equal Richardson FD to 1e-8.
  * The conversion round-trip (A, B) -> (m_r, g) -> (A, B) holds to 1e-14.
* **N1. Uniform-field identity (the consistency the task demands).**
  * Setup: 1-D slab with an imposed linear Potential (the I1 pattern), F in {5e7, 1e8, 3e8} V/m, depleted Fermi factors (f_n = 0, f_p = 1, set via n, p << n_i).
  * Pass: for start nodes whose path stays inside, R / (A F^2 exp(-B/F)) - 1 <= 1e-12. Derivation: closed-form segment integrals, so only roundoff; exp(-B/F) amplifies the relative error of the exponent by B/F ~ 25-100, still < 1e-12.
  * Pass: the integrated hole generation equals the integrated electron generation to 1e-14.
  * Old code: no nonlocal model.
  * **Refute:** turn on the finite-m0 alpha (Eq. 483). The identity must FAIL by the predicted m_r/m0 amount, so the gate is not trivially satisfied.
* **N2. No final state.**
  * Setup: a potential step with total drop 0.8 V < Eg/q = 1.12 V over 2 nm (peak field 4e8 V/m).
  * Pass: nonlocal R == 0.0 exactly at every node.
  * Pass: local Hurkx gives > 1e3 x the integrated nonlocal (i.e. > 0). This is the distinguishing gate.
  * Refute: raise the drop to 1.13 V. R turns on, from paths only in the overlap window.
* **N3. Detailed balance.**
  * Setup: an equilibrium abrupt p+n+ junction (Z0 fixture).
  * Pass: |R| / (one-way rate) <= 1e-12 with Boltzmann.
  * Under FD, a convergence gate as in the L1 addendum (O(h^2)), since the discrete FD equilibrium is off mass action at O(h^2).
* **N4. Independent continuum oracle.**
  * A standalone scipy implementation in the test, which never imports the module. It takes the DEVSIM-solved Potential at a fixed bias, interpolates it with a cubic spline, uses `brentq` for path ends and `quad` for the kappa integrals and the x0 integral, and the same Fermi factors from interpolated L_n, L_p.
  * This gives J_oracle = q Int R dx0.
  * Pass: the module's discrete q Sum R V_i converges to J_oracle at order >= 1.7 over h = 0.4/0.2/0.1 nm. Tolerance on the finest mesh: <= 2 C_inf h^2, with C_inf from a clean uniform study (the L1 method).
  * Refute: break the end-point interpolation (nearest node instead of linear root). The order must drop to about 1 and the gate must fail.
* **N5. Self-consistency.**
  * Pass: terminal current == q Int G_n dV == q Int G_p dV (Z3 floor).
  * Pass: the DC Newton converges on the Zener sweep 0-1.2 V reverse.
  * Pass: the iteration count is within +3 of the local model's.
* **N6. Jacobian (1-D).** The custom share of `get_matrix_and_rhs` equals the FD of the residual to 1e-6 (the J1 pattern), including columns of nodes in the middle of the path.
  * Refute: perturb only one mid-path node's psi. Local models have no entry there, so the entry must exist and match.
* **N7. Local vs nonlocal on the Z1 diode (a measurement, reported, not asserted directionally).**
  * Record I_local / I_nonlocal over the sweep in the doc.
  * Assert only that they agree within the published-model spread at high field (factor 2) and that the low-bias ratio is reported. No direction is gated (section 3.1 premise check).
* **N8. ssac.** Re Y(1 Hz) == DC dI/dV, with the D2b caveats: drive the lightly-doped contact, and use the DC dI/dV as the low-f reference.
* **N9. Noise FDT at equilibrium.**
  * Pass: S = 4kT Re Y to 1e-4 with the two-point source (hole at x0, electron spread over the end element).
  * **Prove-out:** placing both carriers of the source at x0 (a same-node pair, the local-model shortcut) must fail the FDT measurably.
* **N10. Phonon-assisted.**
  * Uniform-field identity as N1 with P = 2.5, form (a) or (b) per U3-2.
  * For form (a): the k_m and finite-m0 limits are gated as departures.
* **N11. 2-D isotropy and cross-dimension (slow).**
  * A laterally uniform 2-D junction equals the 1-D result per area, to the L1-type order-2 convergence gate.
  * An oblique uniform field on triangles (30 deg) gives the N1 identity at interior start nodes to 1e-10. The path walks through many triangles, and the closed forms still hold per segment.
* **N12. 2-D Jacobian.**
  * Frozen direction: FD 1e-6.
  * Dynamic direction: the measured inexactness is reported, and ssac/noise REFUSE (a test asserts the refusal).
* **N13. numba == numpy reference** walker, bit-for-bit or to 1e-15 per path, on random meshes.
* **N14. Published data (nightly validation, `validation/btbt_nonlocal.py`).**
  * Jiao et al. 2013 Fig. 1: Si linearly graded junction, N_A = 2e19, N_D = 1e20 cm^-3, measured J-V 0-4 V reverse (data originally from Hurkx 1992).
  * Tolerance: figure-digitisation error. Measure the digitiser's round-trip on the axis ticks, then add the factor the paper itself shows between its fit and the data.
  * This is honestly a factor-1.5-level check, labelled as such. It is not a precision gate.
  * Vandenberghe 2011 Fig. 5 (symmetric abrupt Si diodes) is a qualitative check of the low-bias ordering.
* **Defaults off.** `btbt_nonlocal=None` changes nothing: SHA, call-log, and no custom registration.

### 3.6 Risks

* **Cost in 2-D/3-D**, and numba as an effective requirement there.
* **Discontinuities.** A start node whose path end crosses a region boundary, or reaches max_length, switches R on or off discontinuously as psi changes. That is non-smooth for Newton, and Sentaurus has the same issue. Mitigation: a smooth taper over the last 5 % of max_length (an ad-hoc choice, documented, gated to change the uniform-field identity by < 1e-12 for paths shorter than 0.95 max_length).
* **Approximate Jacobian in 2-D/3-D with dynamic direction.** ssac/noise are refused. Users wanting AC must freeze the direction.
* The phonon-assisted Eq. 491 transcription.
* The premise (section 3.1): the user may decide the item is not worth the cost beyond 1-D.

---

## 4. Dependencies, effort, workstreams

| Workstream | Content | Depends on | Effort (agent-days, my estimate) |
|---|---|---|---|
| **A: infrastructure** | `custom_terms.py` dispatcher, inventory kind, two-point noise source, refusal rules, C0 gates | none | 3-4 |
| **B: item 2 presets** | fields, presets, `kane_direct_params`, multi-term threading, P1-P8 | the primaries in hand (user); independent of A | 2-3 (+ waiting on sources) |
| **C: item 1** | two-sided TE + stack kernel (K6-K11), 1-D SC term (O1-O7), circuit/ssac/transient (O8-O9), noise (O10-O11), stacks (O12), 2-D/3-D (O14-O15) | A | 12-16 |
| **D: item 3** | segment kernels (N0), 1-D model (N1-N9), phonon (N10), 2-D numba (N11-N13), validation (N14) | A; uses B's `kane_direct_params` inversion | 1-D: 8-10; 2-D: 8-12; 3-D: not estimated |

**Order.** A, then C and D-1-D in parallel. B can run any time. D-2-D comes after D-1-D is accepted.

**Cross-dependencies on other planned work.**
* The D1c trap-kinetics follow-up: the charged-trap ssac xfail is independent, but both touch the noise source set. Coordinate the two-point source API.
* The D1a TE-interface shot noise (strict xfail, integration note 6.7) is the SAME two-point source: a carrier crossing the interface from node copy @r0 to @r1. Workstream A's source closes that xfail almost for free. Suggest bundling it.
* Any gate-work-function work in the contact rule (U1-3).

**CI budget.**
* New fast tests: about 2-3 min total, measured before merge, against a 150-min cap with the coverage leg at 65-95 min.
* Slow (nightly): O14, O15, N11-N13, and the 3-D wiring.
* Validations: O13 (if a source is found) and N14, as `validation/*.py` in `run_all --tier full`.

---

## 5. Decisions for the user

**Item 1**

* **U1-1. Supply model.** Two-sided Tsu-Esaki from the local quasi-Fermi level (recommended: detailed balance exact, J proportional to n_s emerges, and the FDT holds) vs an FN/S-H n_s/n_ref scaling (cheaper, but nonzero current at equilibrium, and the noise cannot satisfy the FDT).
  * Recommend: Tsu-Esaki only, and refuse FN/S-H for self-consistent mode.
* **U1-2. Gate terminal.**
  * Recommend: for a DC parameter gate, report I_gate = -I_tunnel from the module (DEVSIM's DC gate current stays 0, documented, with consumers audited).
  * When the gate is already circuit-driven, add the KCL entries (conservation exact, ssac/noise correct).
  * Do NOT auto-convert gates to circuits: devsim 2.11.2 circuits are process-global and cannot be removed.
* **U1-3. Gate work function.**
  * Recommend: derive the gate barrier from the gate contact's existing Potential offset (one source of truth; the module reports the implied W_m).
  * Optionally add `Electrode.work_function_eV`, which sets that offset in the one contact rule. Default None: unchanged.
* **U1-4. Carriers.** Recommend v0.15 = electrons (ECB) only. Warn when an interface is hole-dominated under high |E_ox|. Hole tunnelling (HVB/EVB) is a follow-on once cited hole masses and offsets are in hand.
* **U1-5. Image-force lowering.** Recommend out of scope, because the shipped barrier is FN-fitted and adding it double-counts. Document the magnitude (0.26 V at 1 MV/cm in SiO2, my arithmetic).
* **U1-6. Stack validation data.** Do you have a published SiO2/high-k J-V with full parameters, or measured data from the team? Without one, O13 does not exist.
* **U1-7. Semiconductor (poly) gates.** Recommend refused in v0.15.

**Item 2**

* **U2-1. Which sets ship.** Recommend Kao 2012 Si/Ge direct + indirect, ONLY after someone reads Kao Table III (library access needed; the sub-agent could not).
  * Not the vendor defaults (Sentaurus E2, Ge 9.1e16/4.9).
  * Not Atlas BBT.STD (units misprinted, primary unread).
  * InGaAs only as a Kane-computed set from Vurgaftman 2001 inputs that have been READ.
  * Can you supply the Kao 2012, Vurgaftman 2001 and Hurkx 1992 PDFs?
* **U2-2. Ge direct-A discrepancy.** The Kane formula gives +10 % vs Kao's printed A. Accept the printed value (recommended, cite as printed) and record the discrepancy, or hold the preset?
* **U2-3. Temperature scaling of B** (Eg(T)^1.5, A constant). Recommend opt-in, only with a cited Varshni set. Also: the repo's old `VARSHNI_PARAMS["InGaAs_LM"]` (Eg(300) = 0.763 eV) was wrong -- RESOLVED 2026-11-06 in v0.14.0 (Vurgaftman alloy rule, 0.737 eV at 300 K; docs/audit/2026-11-06-v0.14-ingaas-gap.md).

**Item 3**

* **U3-1. Is item 3 worth it, given the premise check?** The literature does not support "nonlocal is lower" in general; it supports "no final state, heterojunctions, spatial separation". Recommend 1-D only for v0.15 (gates N0-N10, about 2 weeks), and 2-D after review of the 1-D results.
* **U3-2. Phonon-assisted form.** (a) Sentaurus Eq. 490-491 after re-reading the page image, or (b) our average-field-prefactor form (exact uniform-field limit, not published).
  * Recommend (a) if the transcription verifies, else (b), labelled as our construction.
* **U3-3. 2-D/3-D.** Accept numba as a practical requirement for 2-D nonlocal? Recommend yes, with a refusal message when it is absent. Defer 3-D.
* **U3-4. Interface reflection** (Sentaurus default, needed for GIDL along an oxide). Recommend deferring to after 2-D. v0.15 discards such paths and reports the count.
* **U3-5. Approximate Jacobian.** Recommend: dynamic direction in 2-D/3-D runs DC with an approximate Jacobian and REFUSES ssac/noise; `direction=` frozen gives an exact Jacobian and allows them.

**Cross-cutting**

* **UX-1.** Bundle the two-point noise source with closing the D1a thermionic-interface shot-noise xfail (the same mechanism)? Recommend yes.
* **UX-2.** An extended-precision device with a custom term converges linearly (MEASURED about 12 iterations to 1e-29). Accept, documented? Recommend yes.
