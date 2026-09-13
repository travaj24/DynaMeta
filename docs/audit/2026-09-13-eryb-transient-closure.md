# Er:Yb closure + transient (2026-09-13)

Two gaps stopped `ErYbAmplifier` standing in for `FiberAmplifier` in a downstream burst-PAM
study: the wall-plug layer returned a NaN energy-balance residual for every co-doped result, and
`simulate_transient` refused the class outright. Both are now closed. Nothing about the
single-ion classes changed -- that is asserted byte-for-byte, not argued (gate F).

Branch `feat/eryb-transient-closure`. Version `0.11.0 -> 0.11.1`.

---

## 1. What was built

| Where | What |
| --- | --- |
| `eryb.py` | `_rates_profile`, `_phi`, `_dphi_db`, `_fb_rhs`, `_fb_jacobian`, `_b2_quasi_equilibrium` -- the coupled pair's vectorized RHS and exact Jacobian, the single home of the algebra `_solve_fb` already carried as a root-find residual |
| `eryb.py` | `_rate_balance_dissipation_W(result)` -- the dissipation hook `efficiency` looks for |
| `eryb.py` | `energy_terms(P, f2, b2)` -- the per-z split of that number into background loss / Er / Yb / transfer defect, plus the stored energy and its rate; `stored_energy_J(f2, b2, z)` |
| `efficiency.py` | `_dissipated_power_W` tries `amp._rate_balance_dissipation_W` first, then the existing `_dP_full_c` path, then NaN |
| `dynamics.py` | `simulate_transient_eryb`, the 2x2 `phi_1` kernel (`_phi1_dt_2x2`), `_split_eryb_seed`, the `simulate_transient` dispatch, `frame_as_steady`'s co-doped branch, `amplifier_saturation_energy` |
| `eryb.py` | `solve(relax="auto")` -- the body became `_solve_once` and `solve` gained the same `(1.0, 0.5, 0.25)` ladder wrapper `FiberAmplifier.solve` has, plus `meta['relax_attempts']`. The DEFAULT stays `1.0` (one attempt, unchanged); accepting the string is what lets code written against the single-ion class -- the downstream `solve_closed(amp, relax="auto")` -- run against a co-doped amplifier at all, where it previously raised `ValueError` before the mesh refinement could even start |
| `tests/test_fiber_eryb_transient.py` | 21 gates |

### API (exact call forms)

```python
from dynameta.optics.fiber_amp import PumpSource, wall_plug_efficiency, simulate_transient

# CLOSURE -- unchanged call, now finite for a co-doped result
budget = wall_plug_efficiency(eryb_amp, eryb_amp.solve(n_nodes=641),
                              PumpSource(wallplug_efficiency=1.0))
budget.energy_balance_residual_W          # finite; /budget.p_pump_launched_W is the study's metric

# TRANSIENT -- unchanged call, dispatched on the class
tr = simulate_transient(eryb_amp, t_grid, signal_drive=drive, n_nodes=161)
tr.nbar2_zt          # (Nt, Nz) Er metastable fraction f2
tr.meta["beta_yb"]   # (Nt, Nz) Yb inversion b2         <-- the new array
# nbar2_0 additionally accepts a TUPLE (f2_0, b2_0); a bare scalar/array is f2 alone and b2 is
# seeded from its own quasi-equilibrium at that f2 and the first drive.
```

---

## 2. Derivation of the closure terms

Write `eps_Er`, `eps_Yb` for the two stored excitation quanta (`RareEarthIon.eps_J`, the McCumber
zero lines: 1.2984e-19 J at 1530 nm for Er `4I13/2`, 2.0375e-19 J at 975 nm for Yb `2F5/2`), `A`
for `A_dope`, and `n_a,k = (power absorbed from channel k)/(h nu_k)` for photon rates per unit
length. Per unit length:

```
U(z)     = A (eps_Er N_Er f2 + eps_Yb N_Yb b2)                      stored energy [J/m]
Phi_tr   = A k_tr phi N_Er N_Yb b2 (1 - f2)                         transfers / (m s)
q_opt(z) = -dF/dz = (absorbed + background loss) - (stimulated + guided spontaneous)
```

Each event is charged at the **level's** own energy and the remainder is dissipated:

* an absorbed photon at `h nu` deposits `eps` and sheds `(h nu - eps)` to phonons;
* a stimulated-emission event removes `eps` and hands `h nu` to the field -- the same expression
  with the opposite sign, i.e. anti-Stokes cooling when `h nu > eps`;
* a spontaneous decay removes `eps` of which only the guided part `sum_k s_k` reaches the field --
  the rest is fluorescence into the cladding and non-radiative decay, and is dissipated;
* upconversion removes `eps_Er` outright.

Summing,

```
D_Er   = sum_k (h nu_k - eps_Er)(n_a,k - n_e,k) + (eps_Er Phi_dec_Er - P_sp_Er) + eps_Er Phi_up
D_Yb   = sum_k (h nu_k - eps_Yb)(n_a,k - n_e,k) + (eps_Yb Phi_dec_Yb - P_sp_Yb)
D_tr   = (eps_Yb - eps_Er) Phi_tr
D_loss = sum_k l_k P_k
```

and substituting the two population balances gives the identity `energy_terms` returns:

```
q_opt(z) = dU/dt(z) + D_loss + D_Er + D_Yb + D_tr.
```

### Where each co-doped loss channel sits -- and why nothing is added on top

* **The Yb -> Er transfer defect** is `D_tr`, charged **once**, to the transfer event, with a
  **plus** sign on the dissipation side. It is *not* charged to the Yb absorption (which already
  paid only its own `h nu - eps_Yb`) and *not* to the Er emission. In the fast-`4I11/2` limit the
  model promotes Er straight from `4I15/2` to `4I13/2`, so the 976 nm Yb quantum minus the
  ~1530 nm Er quantum -- the `4I11/2 -> 4I13/2` multiphonon relaxation, **0.4614 eV**, 36% of the
  pump photon -- never appears as light anywhere. Charging it twice breaks the identity by its own
  size (asserted in `test_transfer_defect_is_positive_sized_and_charged_exactly_once`).
* **Yb fluorescence** is inside `D_Yb` as `eps_Yb Phi_dec_Yb - P_sp_Yb`: the excitation left the
  guided field when the pump photon was absorbed and only the guided ASE fraction comes back.
* **Er fluorescence** is the same term in `D_Er`.

Crucially, for the **steady** closure none of this has to be written out: the aggregate
`_rate_balance_dissipation_W` is `-INT dF/dz dz` with `dF/dz` taken from the amplifier's *own*
RHS on the returned profile, and every term above is already inside it. `energy_terms` exists to
*name* the pieces (and to give the transient balance a `dU/dt` to subtract), not to correct the
aggregate. Adding any of them to `_rate_balance_dissipation_W` would double-count.

### Measured -- core-pumped reference point

`a = 2.8 um`, `NA 0.20`, `N_Er = 2e25`, `N_Yb = 2e26`, `k_tr = 2e-22`, 200 mW co-propagating
976 nm pump, 0.5 mW 1550 nm signal, C-band ASE 1520-1570 nm in 24 bins, `yb_ase` 1000-1100 nm in
8 bins, `L = 2 m`. `|residual| / launched pump`:

| n_nodes | 81 | 161 | 321 | 641 | 1281 |
| --- | --- | --- | --- | --- | --- |
| rel. residual | 2.118e-3 | 1.164e-4 | 9.97e-6 | **2.14e-6** | 6.08e-7 |

Falling 4x+ per doubling (`O(dz^2)`, the trapezoid on the rate integral) and **below the study's
1e-5 requirement at 641 nodes**. The gain moves 8.4951 -> 8.5119 dB over the same refinement.

* Identity `q_opt == dU/dt + dissipation` on the solved profile: **4.1e-14** relative.
* Term split at 321 nodes, of 0.2005 W launched: `D_loss` 0 (lossless fiber), `D_Er` 7.571e-3 W,
  `D_Yb` 1.7995e-1 W, `D_tr` 5.046e-3 W, total **0.192571 W**, equal to
  `_rate_balance_dissipation_W` to 2e-15 relative. `D_Yb` dominates because this point is
  *core*-pumped at a 10:1 Yb:Er ratio -- `eta_transfer` is 0.0021 there, so ~90% of the pump ends
  as Yb fluorescence. That is the physics of core-pumping an EYDFA at a 200 mW launch into a
  2.8 um core, not a bug: the pump intensity saturates Yb long before the 10x-scarcer Er
  acceptors can drain it. MEASURED for contrast, the cladding-pumped point of sec. 4 at the same
  doping runs `eta_transfer = 0.516`.
* Discrimination: perturbing the signal output endpoint by 1% moves the residual
  **2.14e-6 -> 1.75e-4** (x82), and 1.75e-4 *is* the injected error. `thermal.total_heat_W` would
  not have moved at all -- it is `X - X` by construction.

---

## 3. The transient march and the integrator choice

### Scheme

Same two-timescale split as the single-ion march: powers **quasi-static** each step (the exact
integrating-factor sweep `_propagate_fixed` through the frozen `g(z; f2, b2)`), populations
advanced by an integrator. The populations are now a **pair** per node.

### Why exponential Rosenbrock (ETD1 with the exact local Jacobian)

```
y' = F(y),  y = (f2, b2);   y_{n+1} = y_n + [dt phi_1(dt J)] F(y_n),   J = dF/dy|_{y_n}
phi_1(X) = (e^X - I) X^{-1} = SUM_{k>=0} X^k/(k+1)!
```

The problem is stiff by a factor ~1e3: with the repo's phosphosilicate numbers the transfer drain
`k_tr N_Er = 4.0e3 /s` sits against `1/tau_Yb = 6.9e2 /s` and `1/tau_Er = 1.0e2 /s`, and a
saturating 976 nm pump adds `R_a_Yb ~ 1e5 /s` on top -- so the Yb reservoir responds ~1e3 faster
than the Er reservoir it feeds, while the transient the caller cares about runs for ~1e-2 s.

Four properties, each of which a cheaper scheme gives up:

1. **It reduces exactly to the existing single-ion update when the coupling vanishes.** With
   `k_tr = 0` the Jacobian is diagonal, `J11 = -(R_a + R_e + 1/tau) = -B`, and the formula
   collapses to `n2_ss + (n2 - n2_ss) exp(-B dt)` term for term. Measured: the Er-only limit
   reproduces `simulate_transient` on the same fiber and drive to **3.6e-14 dB** of gain over a
   200-frame march (gate asks 1e-3), and `nbar2` to 6.4e-16.
2. **The amplifier's own steady state is an exact fixed point**, for any `dt`, because the
   increment carries `F(y_n)` as a factor -- which is what lets a constant-drive march started
   from `solve()` sit still. An implicit Euler on a *linearised* pair would only share it if the
   linearisation were re-centred every step.
3. **Unconditional stability -- the property a SPLIT scheme does not have.** `_fb_jacobian` shows
   `tr(J) < 0`, `det(J) = a d + a k_out (1-f2) + k_in b2 d > 0` (the cross terms cancel out of the
   determinant) and a non-negative discriminant `(J11-J22)^2 + 4 J12 J21`, so both eigenvalues are
   **real and negative at every operating point**: a stable node, never a spiral. `e^{dt J}` is a
   contraction for every `dt`, and as `dt -> inf` the step becomes `y_n - J^{-1}F(y_n)`, a Newton
   step onto the steady state. A semi-implicit split (freeze `b2`, advance `f2`, swap) *shares* the
   fixed point -- both sub-steps vanish only at a root of the pair, so the obvious objection to it
   is not the right one -- but as `dt` grows each sub-step runs to its own local root and the step
   **degenerates into the naive block Gauss-Seidel iteration** whose positive Yb<->Er feedback has
   spectral radius `~ k_tr^2 N_Er N_Yb tau_Er tau_Yb >> 1` and diverges (the failure the steady
   solve's `_solve_fb` was written to avoid). A split march is stability-limited to `dt` well
   inside `1/(k_tr N_Er) ~ 250 us` -- the regime the two-timescale design exists to leave.
4. **No inner nonlinear solve.** A safeguarded implicit Euler with Newton per node is the other
   admissible choice and is equally stable, but it costs an inner loop with a convergence test
   that can fail, and it is the same first order on the path -- so it buys nothing here.

The price: ETD1 is **first order in the path** (the Jacobian and the powers are frozen across the
step). Endpoints and stability are unaffected; the trajectory carries an `O(dt)` error. Same order
as the single-ion march.

### `phi_1` of a 2x2, robustly

The eigen-decomposition route (`f(J) = alpha I + beta J` with `beta` a divided difference) loses
all its digits when the eigenvalues coalesce, which happens routinely here as the pump rises and
the Er and Yb blocks cross. So `phi_1` is evaluated by **scaling and squaring on the matrix
itself**, with no branch: scale `X = dt J` by `2^-m` to `||X||_inf <= 1/2`, Taylor both `e^X` and
`phi_1(X)` there (order 18, truncation `0.5^19/20! ~ 8e-25`), then square up with

```
e^{2X} = (e^X)^2,      phi_1(2X) = (1/2) phi_1(X) (e^X + I)
```

(both from `e^{2X} - I = (e^X - I)(e^X + I)`). Vectorized over `z`; ~10 2x2 matmuls of `(Nz,)`
arrays per step.

The kernel is gated against an INDEPENDENT oracle -- `scipy.linalg.expm` on the augmented matrix,
`expm([[dt J, dt F], [0, 0]])[:2, 2] == dt phi_1(dt J) F`, a different algorithm (Pade
scaling-and-squaring on a 3x3) reaching the same quantity. Worst relative error over 400 random
stiff 2x2s with the sign structure `_fb_jacobian` guarantees, `dt` spanning 1e-9..1e-3 s:
**1.0e-14**. At EXACTLY coalesced eigenvalues (`J12 = J21 = 0`, `J11 = J22`), where the divided
difference is `0/0`: **1.1e-16**.

---

## 4. Measured gate numbers

Cladding-pumped transient point unless stated: `a = 2.8 um`, `NA 0.20`, clad 62.5 um,
`N_Er = 2e25`, `N_Yb = 2e26`, `k_tr = 2e-22`, `L = 4 m`, 1 W 976 nm cladding pump, 1 mW 1550 nm
signal, C-band ASE in 8 bins. Steady gain 24.49151 dB at 161 nodes.

| Gate | Claim | Measured |
| --- | --- | --- |
| (a) Er-only limit | `<= 1e-3` dB vs `FiberAmplifier`'s march | **3.6e-14 dB** max over 200 frames |
| (b) constant drive from `solve()` | `< 1e-3` dB over the whole march | **4.2e-4 dB** at 161 nodes (1.68e-3 / 4.22e-4 / 1.07e-4 / 2.64e-5 at 81/161/321/641 -- `O(dz^2)`) |
| (b) perturbed start | settles onto `solve()` | -47.76 dB -> 24.49180 dB, **2.9e-4 dB** from steady; monotone, no ringing |
| (c) step-drive energy balance | closes step by step | step-averaged residual at `t = 20 us`: **1.77e-3 / 9.50e-4 / 4.92e-4** of launched at `dt = 1e-6 / 5e-7 / 2.5e-7` -- ratios 1.86, 1.93 (**first order**, as the scheme predicts) |
| (c) mesh part of the same residual | `O(dz^2)` | **2.33e-4 / 5.81e-5 / 1.45e-5** at 81/161/321 nodes -- exactly 4x per doubling |
| (d) steady closure | `< 1e-5` of launched pump after refinement | table in sec. 2: **2.14e-6** at 641 nodes |
| (e) stiff regime | no NaN, populations in `[0,1]`, `dt` over 1e-8..1e-4 s | all finite and in range; `||dt J||_inf` spans **3.6e-4 .. 3.63** on the 1 m fixture and **4.0e-3 .. 40.5** on the core-pumped one. Population overshoot before the clip is **0.0** up to `||dt J|| = 12.8` and **8.6e-2** at 40.5 |
| (f) single-ion path unchanged | unchanged | 14 values recorded on `main @ 3496a99` (march gains, `nbar2` sums, ASE sums, the full profile-matrix sum, a `frame_as_steady` frame, and the `wall_plug_efficiency` residual / eta / heat) reproduce to **1e-9 relative** (march-derived) and **1e-5** (solve-derived); `frame_as_steady`'s meta key set is unchanged; and a structural test asserts the three new branches are unreachable on a `FiberAmplifier`. See "A gate this work got wrong first" below |

### A gate this work got wrong first, and what CI proved

Gate (f) originally compared those 14 `float64` values with `==`, on the reasoning that the march
is a fixed sequence of numpy reductions and must therefore be bit-reproducible. **This branch's
first CI run falsified that twice**, while every other test on both legs passed (1362 and 1392
passed respectively):

* **floor leg** (numpy 1.24 / scipy 1.10 / py3.10): all ten MARCH values matched bit for bit, but
  the wall-plug residual came out 1.0193e-4 W against 1.0187e-4 W -- 6.2e-4 relative.
  `amp.solve()` drives scipy's **adaptive LSODA**, whose step sequence differs between scipy
  versions, so it settles on a slightly different point of the same fixed point. In the unit that
  number means -- a fraction of the launched pump -- the shift is 5.2e-7.
* **py3.10 leg** (newest numpy): `gain_last` differed by **one ULP** (`...d80` vs `...d7f`) -- a
  reduction-order / BLAS difference inside the march itself.

So bitwise reproducibility is a property of the BUILD, not of this change, and asserting it was a
false gate: it would go red on any runner whose numpy or scipy differs from the recording box, for
reasons having nothing to do with the code under test. The gate now holds the march values to
1e-9 relative and the solve-derived ones to 1e-5 -- still 7+ and 4+ orders tighter than any
regression this change could cause, since a mis-routed march or a mis-fired efficiency hook moves
gains by dB or returns NaN. The *environment-independent* half of the claim was added alongside
it: `test_the_new_branches_are_structurally_unreachable_for_a_single_ion_amplifier` asserts that
`FiberAmplifier` does not define `_rate_balance_dissipation_W` (so the efficiency hook cannot
fire), is not an `ErYbAmplifier` (so the dispatch cannot fire), and has a `ChannelPlan.channels`
that is not None (so `frame_as_steady`'s co-doped branch is dead code for it). That is the actual
claim, and it does not depend on a numpy build.

The `t = 20 us` reference time in gate (c) is deliberate. Taking the **maximum over the whole
march** instead puts the number at the first step after the drive edge, where the reservoirs move
at `|dU/dt| ~ 1.8x` the launched power and the frozen-power splitting error is
`O(||dt J||) x |dU/dt|`: 1.77e-1 / 1.52e-1 / 1.01e-1 / 5.86e-2 / 3.16e-2 at
`dt = 4e-6 / 2e-6 / 1e-6 / 5e-7 / 2.5e-7`. That is still first order and still converging, but it
is a statement about the sharpest edge in the march, not about the march.

---

## 5. Limits the caller has to know

1. **The closure is mesh-limited by the *pump* absorption length, not the fiber length.** At the
   core-pumped reference point `gamma N_Yb sigma_a(976) ~ 200 /m`, so the pump is gone in ~5 mm
   and 641 nodes over 2 m are needed for 1e-5. A cladding-pumped point at the same doping has
   `~1 /m` and closes at 161. Refine against the residual (`solve_closed`), do not assume a node
   count.
2. **The march's fixed point is `solve()`'s fixed point only to `O(dz^2)`**, because
   `_propagate_fixed` integrates `INT g dz` by cumulative trapezoid where `solve()` uses adaptive
   LSODA. This is inherited from the single-ion march, not new. Measured on the *core*-pumped
   point (the hard case): march-vs-solve gain deviation 5.79e-2 / 9.56e-3 / 5.96e-4 / 3.70e-5 dB
   at 161 / 401 / 1601 / 6401 nodes. On the cladding-pumped point 161 nodes already give 4.2e-4 dB.
3. **The quasi-static (frozen-population) step has the same audit-A-7 ASE limit as the single-ion
   march**, and a co-doped amplifier reaches it more easily because the transfer can invert Er
   hard. Measured: a cold start `(f2, b2) = (0.02, 0.02)` on a **uniform 300 us** grid lets one
   step carry the inversion across the ASE-clamped operating point -- frozen-step ASE hits **342x**
   the launched power, `meta['quasi_static_valid']` goes `False` and a `RuntimeWarning` is raised.
   The same start on a **150 us** uniform grid, or on a log-spaced grid from 1 us, is valid and
   lands on `solve()`. Resolve the fast head of a large population excursion.
4. **The transfer defect's sign and placement.** `D_tr = (eps_Yb - eps_Er) Phi_tr >= 0`, on the
   dissipation side, charged once to the transfer event. It is already inside
   `energy_balance_residual_W`; do not add it.
5. **`meta['max_dt_times_rate']`** is reported, not gated. `>> 1` means the Yb reservoir was
   *slaved* to the Er state within a step rather than resolved -- correct for the endpoints and
   for the steady limit, first order on the path. Read it when a droop-vs-time curve matters at
   sub-`tau_Yb` resolution.
6. **One `m_modes` for both ASE bands.** `ase_psd_1pol_W_Hz` divides the whole spectrum by a
   single mode count, so a C band and a Yb band with different `m_modes` would be mis-scaled in
   one of them. Same limitation `ErYbAmplifier.solve` already has.
7. **`ChannelPlan.channels` is `None` for a co-doped plan** (two ions per channel). The
   cross-sections and this frame's `b2` now travel on `TransientResult.meta` and
   `frame_as_steady` copies them onto the frame, so the noise/thermal layer sees a self-consistent
   `SteadyStateResult`.
8. **One attribute difference is left, deliberately.** `ErYbAmplifier` has no `.ion` -- it has
   `.er_ion` and `.yb_ion`, and two ASE bands (`.ase`, `.yb_ase`). Everything a consumer reaches
   for through the *protocol* (`solve`, `with_pumps`, `with_signals`, `without_ase`,
   `channel_plan`, `pumps`, `signals`, `fiber`) is common, and `saturation_energy`'s dependence on
   the ion is covered by the new `amplifier_saturation_energy(amp, lambda_m)` adapter. A consumer
   that reads `amp.ion.tau_s` directly still has to branch, or read `amp.er_ion.tau_s`. Aliasing
   `.ion` to `.er_ion` was considered and rejected: on a sensitized amplifier "the ion" is not a
   well-posed question, and an alias would quietly answer a 976 nm query with erbium.

---

## 6. Verification run

`pytest tests/test_fiber_eryb_transient.py tests/test_fiber_eryb.py tests/test_fiber_dynamics.py`
-> all green. `ruff check .` clean. Fast suite (`pytest tests/ -q -m "not slow"`): see the PR.

**End-to-end against the consumer, unmodified.** The downstream study's own
`solve_closed(amp, n_nodes=81)` (its defaults: `relax="auto"`, `tol_rel = 1e-5`) and its
`simulate_transient(amp, t_grid, signal_drive=..., n_nodes=161)` call form were run verbatim
against a cladding-pumped `ErYbAmplifier` with a 24-bin C band. `solve_closed` refined
**81 -> 161 -> 321 nodes** (closure 1.03e-4 -> 2.59e-5 -> **6.55e-6**) and returned; the march
produced the idle->data droop curve (32.26 -> 24.45 dB), the 24-bin resolved ASE, an interpolable
`S_ASE(1550 nm)` of 3.88e-16 -> 3.97e-17 W/Hz, `beta_yb` in 0.031-0.245, and
`quasi_static_valid = True` with no warnings. No change was needed on the consumer side. `ruff check .` clean. Fast suite: see the PR.
