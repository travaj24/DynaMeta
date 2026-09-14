# Er:Yb two-population physics (2026-09-14)

`ErYbAmplifier` modelled one ytterbium population, one transfer channel, no concentration
quenching and no temperature. The 2026-09-14 literature validation
(`VALIDATION_SUMMARY.md` section 4) listed six things the model did not contain and tied four of
them to a measured failure: at the device-validated effective coefficient `k_tr = 1e-21 m^3/s`
the single-population model reproduces the vendor PCE and the small-core cladding-pumped stages
but over-predicts the large-core and watt-class points by 1.4-2x, and it has no mechanism at all
for the FORC observation that the 1-um ASE fraction is a property of the fiber's energy-transfer
efficiency. This branch adds the physics those gaps name. Everything is opt-in; with every new
parameter at its default the class reproduces v0.11.1 bit for bit.

Branch `feat/eryb-two-population-physics`. Version `0.11.1 -> 0.11.2`.

---

## 1. What was built

| Where | What |
| --- | --- |
| `eryb.py` | `yb_coupled_fraction` f, `k_tr2_m3_s` K2, `yb_migration_rate_per_s` W, `concentration=ConcentrationModel(...)`; `_solve_fbb` (the three-unknown z-local steady state), `_fbb` (the router), `_bracketed_newton` (the shared root find), `_fb_rhs3` / `_fb_jacobian3` (the 3-state RHS and exact 3x3 Jacobian), `_bbar` |
| `eryb.py` | `set_temperature_profile` / `clear_temperature_profile` / `_mcc_matrices` -- per-z McCumber scaling of BOTH ions about their own zero lines; `_heat_profile_W_per_m` -- the rate-balance heat profile the thermal loop drives on |
| `eryb.py` | `energy_terms(..., b2_uncoupled, z_m=)` gains `k2_defect` / `k2_rate_per_m` / `db2nc_dt` / `beta_yb_mean`; `stored_energy_J(..., b2_uncoupled)` counts both pools |
| `dynamics.py` | `_phi1_dt_nxn` -- the size-parametrized `phi_1` kernel (`_phi1_dt_2x2` is now its n = 2 spelling); the THREE-reservoir march; a 3-tuple `nbar2_0`; `meta['beta_yb_coupled'/'beta_yb_uncoupled']`; an explicit refusal of a temperature profile |
| `thermal.py` | `solve_with_thermal_feedback` accepts `ErYbAmplifier` and prefers the amplifier's own `_heat_profile_W_per_m`; `info['heat_source']` |
| `calibration.py` | `ytterbium_melkumov(host=...)` with the measured P2O5 columns of the same printed table; `er_yb_phosphosilicate_reference()` |
| `spectroscopy.py` | `erbium(host="phosphosilicate")` re-anchored to the measured P-host C band (`p_host_spectra=False` restores the old label-only ion bit-exactly) |
| `tests/test_fiber_eryb_physics.py` | 25 gates |
| `validation/eryb_two_population_anchors.py` | the anchor report (a script, not a gate) |

### API -- exact call forms

```python
from dynameta.optics.fiber_amp import (ErYbAmplifier, ConcentrationModel, ThermalModel,
                                       solve_with_thermal_feedback, simulate_transient,
                                       erbium, ytterbium_melkumov,
                                       er_yb_phosphosilicate_reference)

amp = ErYbAmplifier(er_ion, yb_ion, fiber, pumps, signals, ase,
                    n_yb_m3=4.0e26, k_tr_m3_s=3.0e-21,
                    yb_coupled_fraction=0.9,             # f: 1.0 (default) = the old model
                    k_tr2_m3_s=2.0e-22,                  # K2 secondary transfer (default 0)
                    yb_migration_rate_per_s=5.0e3,       # W_mig (default 0)
                    concentration=ConcentrationModel(c_up_m3_s=1.1e-24, pair_fraction=0.016,
                                                     pair_convention="delevaque",
                                                     pd_loss_per_m=0.0),
                    yb_ase=AseBand(1.00e-6, 1.10e-6, 12))

r = amp.solve(n_nodes=321)
r.meta["beta_yb_z"]              # population-weighted f b2c + (1-f) b2nc -- what the field sees
r.meta["beta_yb_coupled_z"]      # b2c
r.meta["beta_yb_uncoupled_z"]    # b2nc, or None for the one-pool model
r.meta["eta_transfer"]           # now the FORC "ETE": the uncoupled pool is in the denominator

terms = amp.energy_terms(r.power_W, r.nbar2_z,
                         r.meta["beta_yb_coupled_z"], r.meta["beta_yb_uncoupled_z"])
terms["k2_defect"]               # eps_Yb * the K2 rate, the new heat channel
amp.stored_energy_J(r.nbar2_z, r.meta["beta_yb_coupled_z"], r.z_m,
                    r.meta["beta_yb_uncoupled_z"])

tr = simulate_transient(amp, t_grid, n_nodes=81, nbar2_0=(0.02, 0.02, 0.02))   # TRIPLE seed
tr.meta["beta_yb_coupled"], tr.meta["beta_yb_uncoupled"]                       # (Nt, Nz) each

amp.set_temperature_profile(z_m, T_K, T_ref_K=300.0)      # BOTH ions, each about its own line
res, T_z, info = solve_with_thermal_feedback(amp, ThermalModel(h_conv_W_m2K=200.0),
                                             b_outer_m=62.5e-6, n_nodes=161)
info["heat_source"]              # "rate_balance" for ErYb, "flux_gradient" for FiberAmplifier

yb = ytterbium_melkumov("phosphosilicate")                # measured P2O5 table, tau 1.45 ms
er = erbium("phosphosilicate")                            # measured P-host C band, tau 9.0 ms
er2, yb2, fiber2, kwargs = er_yb_phosphosilicate_reference()          # or ("single")
```

---

## 2. Derivations

### 2.1 Two populations (Dong et al., Opt. Express 28(11), 16244 (2020), Eqs. (1)-(12))

A fraction `f` of the ytterbium is COUPLED to the erbium (`n5c`, `n6c`) and `1 - f` is UNCOUPLED
(`n5nc`, `n6nc`). Only the coupled pool transfers; both are pumped, both decay with `tau_Yb`, and
both absorb and emit at every channel, so the propagation sees `n6 = n6c + n6nc`. In this
module's per-ion-fraction variables, `b2c = n6c/(f N_Yb)` and `b2nc = n6nc/((1-f) N_Yb)`:

```
Er:   R_a_Er (1-f2) - R_e_Er f2 - f2/tau_Er + k_tr b2c (f N_Yb)(1-f2) - C_up N_Er f2^2      = 0
Ybc:  R_a_Yb (1-b2c) - R_e_Yb b2c - b2c/tau_Yb - k_tr b2c N_Er (1-f2) phi
      - K2 b2c N_Er f2 - W (1-f)(b2c - b2nc)                                                = 0
Ybnc: R_a_Yb (1-b2nc) - R_e_Yb b2nc - b2nc/tau_Yb + W f (b2c - b2nc)                        = 0
```

Two bookkeeping points that are easy to get wrong and are gated. The transfer-IN coefficient in
the Er equation is `k_tr f N_Yb` (the COUPLED donor density), while the transfer-OUT coefficient
in the coupled-Yb equation stays `k_tr N_Er (1 - f2)` (the acceptor density, per coupled Yb ion).
And the optical field sees the POPULATION-WEIGHTED `bbar = f b2c + (1-f) b2nc`, not `b2c`: that
is what enters the propagation, the 1-um parasitic diagnostic, the photodarkening law and the
stored energy.

### 2.2 The reduction stays a bracketed scalar in f2

At fixed `f2` the two ytterbium balances are LINEAR in `(b2c, b2nc)` -- the transfer drain, the
K2 drain and the migration exchange are each proportional to one of them with `f2`-dependent
coefficients only. With

```
D   = R_a_Yb + R_e_Yb + 1/tau_Yb                              (both pools share it)
Dc  = D + k_tr N_Er (1-f2) phi + K2 N_Er f2                   (the coupled pool's extra drains)
A   = [[Dc + W(1-f),  -W(1-f)], [-W f,  D + W f]],   RHS = (R_a_Yb, R_a_Yb)
```

the `W^2 f (1-f)` terms cancel out of the determinant EXACTLY and

```
det  = Dc D + W (f Dc + (1-f) D)   > 0 for every f2 and every W
b2c  = R_a_Yb (D + W) / det,       b2nc = R_a_Yb (Dc + W) / det
```

-- a closed form with no matrix solve and no possibility of a singular system. Limits, term by
term: `W -> 0` gives `b2c = R_a/Dc` and `b2nc = R_a/D` (an untouched plain-Yb pool); `W -> inf`
gives `b2c = b2nc = R_a/(f Dc + (1-f) D)`, the fully-mixed pool; `K2 -> 0, W -> 0, f -> 1`
recovers the one-pool `b2 = R_a/(D + k_tr N_Er (1-f2) phi)` verbatim.

Substituting `b2c` into the Er residual leaves the SAME single scalar equation `H(f2)` on
`[0, 1]` with `H(0) >= 0` (the absorption and transfer drive) and `H(1) < 0` (decay only), so the
safeguarded Newton/bisection is as unconditionally robust as before. Its slope carries

```
d b2c/d f2 = -b2c (D + W f) dDc/df2 / det,    dDc/df2 = K2 N_Er - k_tr N_Er phi
```

which reduces to the one-pool `R_a s phi / Dc^2` at `W = K2 = 0`. `phi` (the back-transfer
branching factor) is iterated by the same three-pass inner fixed point and now takes the COUPLED
ground density `n5c = f N_Yb (1 - b2c)` as the back-transfer acceptor -- exactly 1.0 when
`k_back = 0`, the default.

The root find itself was extracted into `_bracketed_newton` so the one- and two-pool routines
share it; the closed forms differ, the solver does not. The one-pool routine is RETAINED verbatim
(rather than deleted in favour of the general one) because `b2 = R_a/Dc` and
`b2 = R_a D/(Dc D)` are not the same float; a gate holds the two to `1.3e-13` over 300 random
rate quadruples in the limit where both apply.

### 2.3 Secondary transfer K2 (Sefler et al., JOSA B 21(10), 1740 (2004))

`Yb(2F5/2) + Er(4I13/2) -> Yb(2F7/2) + Er(4F9/2)`, and the `4F9/2` relaxes multiphonon-fast back
to `4I13/2`. The erbium therefore ENDS THE ROUND TRIP IN THE LEVEL IT STARTED IN: the Er rate
equation is untouched (gated: the Er RHS at a fixed state is bit-identical with and without K2),
the coupled ytterbium loses one excitation, and the whole Yb quantum becomes heat. So K2 enters
as a pure drain `-K2 b2c N_Er f2` on `b2c`, joins the non-useful channels of `eta_transfer`, and
adds ONE dissipation term

```
Phi_K2 = A K2 N_Er f2 (f N_Yb b2c),      D_K2 = eps_Yb Phi_K2.
```

Sefler's Table 1 gives `K2 = 2.0 / 1.5 / 4.0 e-22 m^3/s` on three double-clad fibers --
comparable to their own `K1 = 4 / 2 / 2 e-22` and the paper's one genuinely new quantity (it
carries no "seeded from the literature" footnote). Canat's 2006 thesis fits `2e-22` (6.1 um
fiber) and `1e-22` (11 um). Laroche et al. 2006 report the secondary transfer as more efficient
than upconversion. `2e-22` is the value the validation script runs at; the default is 0.

### 2.4 Yb-Yb migration, and the ensemble-versus-pair reconciliation

The exchange term is DERIVED, not posited. Migration hops an excitation from an excited Yb to a
ground-state Yb, so the volumetric coupled -> uncoupled rate is `W' n6c n5nc` and the reverse is
`W' n6nc n5c`. Their difference is

```
J = W' [n6c n5nc - n6nc n5c]
  = W' f (1-f) N_Yb^2 [b2c (1 - b2nc) - b2nc (1 - b2c)]
  = W' f (1-f) N_Yb^2 (b2c - b2nc),
```

because the `(1 - b)` products cancel identically: a bimolecular exchange obeying detailed
balance collapses to a difference of EXCITATION FRACTIONS. Writing `W_mig = W' N_Yb` [1/s],

```
d n6nc/dt |_mig = +J = -W_mig f (n6nc - n6c (1-f)/f),
d b2nc/dt |_mig = +W_mig f (b2c - b2nc),    d b2c/dt |_mig = -W_mig (1-f)(b2c - b2nc),
```

which conserves `f b2c + (1-f) b2nc` exactly (gated to `1e-12` relative over 50 random states)
and has `b2c = b2nc` as its ONLY equilibrium -- equal excitation fractions, never equal
densities. In the Jacobian the block is `W [[-(1-f), (1-f)], [f, -f]]`, a Markov generator with
eigenvalues `0` and `-W`, symmetrizable by `diag(1/sqrt(f), 1/sqrt(1-f))`, so it contributes real
non-positive eigenvalues and cannot introduce a spiral by itself.

WHY IT MATTERS. Cheng et al., Materials 15(3), 996 (2022) measure ONE bi-exponential Yb decay in
Er/Yb/P glass and it yields two mutually inconsistent transfer coefficients: a FAST-PAIR rate
`3.07e-21 m^3/s` (from the 8.20 us fast component, the rate a close-coupled neighbour sees) and
an ENSEMBLE-YIELD rate `1.20e-22 m^3/s` (from the integrated emission ratio, the rate that
reproduces the measured 89.5% transfer yield). A factor of 26, from the same data. Dong et al.
argue explicitly that decay measurements OVERSTATE the coupled fraction because migration feeds
the fast component. With `W_mig > 0` both numbers can be carried at once: a fast-pair `k_tr` on a
small coupled pool, exchanging with a large uncoupled one, delivers the ensemble yield that a
single pool needs a small `k_tr` to match. MEASURED on the gate fixture at `f = 0.5`,
`k_tr = 3e-21`: `eta_transfer` rises `0.535 -> 0.554 -> 0.851 -> 0.912` as
`W = 0 -> 1e2 -> 1e4 -> 1e6 1/s`, while `max|b2c - b2nc|` falls `0.260 -> 0.254 -> 0.0598 ->
7.6e-4`. NO measurement of `W_mig` itself was located; it is the one parameter here with no
source, and its default is 0.

### 2.5 Energy closure

The identity gains one term and the ytterbium terms gain the second pool:

```
U(z)   = A (eps_Er N_Er f2 + eps_Yb N_Yb bbar)                     stored [J/m]
Phi_tr = A k_tr phi N_Er (f N_Yb) b2c (1 - f2)                     transfers/(m s)
Phi_K2 = A K2 N_Er f2 (f N_Yb b2c)                                 K2 events/(m s)

D_Er   = SUM_k (h nu_k - eps_Er)(n_a,k - n_e,k) + (eps_Er Phi_dec_Er - P_sp_Er) + eps_Er Phi_up
D_Yb   = SUM_k (h nu_k - eps_Yb)(n_a,k - n_e,k) + (eps_Yb Phi_dec_Yb - P_sp_Yb)   [both pools]
D_tr   = (eps_Yb - eps_Er) Phi_tr
D_K2   = eps_Yb Phi_K2
D_loss = SUM_k l_k P_k          [incl. the dark-ion absorption and the photodarkening gray loss]

q_opt(z) = dU/dt(z) + D_loss + D_Er + D_Yb + D_tr + D_K2
```

`D_K2` carries the FULL `eps_Yb`, not the `eps_Yb - eps_Er` of the primary transfer, because
nothing of that excitation is stored in the erbium -- it comes straight back down. `dU/dt` uses
`f db2c/dt + (1-f) db2nc/dt`, in which the migration exchange cancels identically; that
cancellation is the numerical signature that the exchange conserves excitation. Pair-induced
quenching and photodarkening need no term of their own: dark erbium is already inside `D_loss`
through the unbleachable absorption `_coeffs` adds, and so is the gray loss, and both are pure
heat. Verifying the eps bookkeeping term by term: every `eps_Er` and `eps_Yb` contribution
cancels between `dU/dt` and the `D_*` sum, leaving exactly
`loss + (p_a_er - p_e_er - p_sp_er) + (p_a_yb - p_e_yb - p_sp_yb) = q_opt`.

### 2.6 The integrator

The march state becomes `y = (f2, b2c, b2nc)` and the update is the SAME exponential Rosenbrock
step (`y <- y + [dt phi_1(dt J)] F(y)`) with the exact 3x3 `_fb_jacobian3`. There is no second
scheme: `_phi1_dt_nxn` is the size-parametrized kernel and `_phi1_dt_2x2` is now its `n = 2`
spelling. The generalization is written as explicit nested lists with the inner products summed
in ascending index order rather than as `np.matmul` over an `(N, n, n)` stack, precisely so the
`n = 2` path performs the same arithmetic in the same order it always did -- `np.matmul`'s
reduction order is a BLAS property, and using it would have made the pinned march values
build-dependent, which the 2026-09-13 note records CI falsifying twice. The one place the
accumulation order had to be stated explicitly is the increment itself: `(y + m0 r0) + m1 r1`,
not `y + (m0 r0 + m1 r1)` -- the two differ by ~1 ULP per node and the second spelling moved the
pinned transient sum on the first try.

Structure of the 3x3: `J[0][2] = J[2][0] = 0` (within a frozen-power step the erbium and the
uncoupled ytterbium meet only through the coupled pool); the diagonal is strictly negative. With
`K2 = 0` the cross terms still cancel out of the determinant as in the 2x2 case. With `K2 > 0`
they do not, so NO theorem is claimed for that corner -- the property the scheme actually needs,
that every eigenvalue keeps a strictly negative real part, is gated numerically instead:
`max Re(eig) = -6.5e2 1/s` over 300 random operating points at `f = 0.6`, `k_tr = 3e-21`,
`K2 = 4e-22`, `W = 1e4`, with `max |Im(eig)| = 0` (the eigenvalues stayed real throughout).

### 2.7 Distributed temperature

`mcc_ion,k(z) = exp[(eps_ion - h nu_k)(1/kT(z) - 1/kT_ref)]` with `eps_ion = RareEarthIon.eps_J`
-- each ion scaled about ITS OWN zero line (Er ~1530 nm, Yb ~974 nm). A single shared `eps` would
be wrong by orders of magnitude in the pump band, where the Yb line sits and the Er line does
not. Two `(K, M)` matrices are built on the solver mesh and interpolated into the IVP exactly as
`steady_state` does for the single ion. `sigma_a` and the two lifetimes are held, and so are
`k_tr`, `K2` and `W_mig`: their measured temperature dependence (Cheng 2022 puts the Yb transfer
component at 8.20 us at 300 K and 7.56 us at 480 K, ~8%) is NOT modelled, and that is stated in
the method docstring rather than buried.

`thermal.solve_with_thermal_feedback` drives `Q(z)` from `_heat_profile_W_per_m` when the
amplifier offers it. For an EYDFA that matters: the dominant heat term is the TRANSFER DEFECT
(the `4I11/2 -> 4I13/2` multiphonon step, 0.46 eV, 36% of every transferred 976 nm photon), and
the second is the fluorescence of an uncoupled pool -- neither is an erbium quantum defect, and a
loop driven by `np.gradient` of the net flux gets the total right only to the mesh error of the
solve while saying nothing about which mechanism deposited it. `FiberAmplifier` defines no such
hook, so the single-ion loop is unchanged; `info['heat_source']` records which ran.

---

## 3. Measured gate numbers

`tests/test_fiber_eryb_physics.py`, 25 gates, 52 s. The fixture is a 6 um-core cladding-pumped
Er:Yb at the study's own densities (N_Er 4e25, N_Yb 4e26), 1 W of 976 nm into a 125 um cladding,
20 mW of 1550 nm seed, 3 m: 13.8 dB, `eta_tr` 0.756, 1-um parasitic gain -24.9 dB.

| Gate | Measured |
| --- | --- |
| defaults == v0.11.1 (steady, closure, transient) | 24 pinned values, all within `1e-9` relative (LSODA-path-sensitive ones within `1e-5`); `k2_defect` and `k2_rate_per_m` identically `0.0` |
| the new branches are unreachable with the defaults | `_two_pop is False`, `_mcc_matrices -> None`, `_fbb` returns `b2nc = None` and `bbar is b2c` (the same object) |
| f = 0 == k_tr = 0 at the same total N_Yb | `1e-5` dB; `eta_transfer == 0.0` exactly; `max b2nc = 0.16` (the ytterbium is still there and still absorbing) |
| coupled fraction, at fixed `f k_tr = 2.7e-21` | `eta_tr` 0.335 / 0.541 / 0.711 / 0.859 and 1-um parasitic gain +20.6 / +5.9 / -17.3 / -37.9 dB at f = 0.3 / 0.5 / 0.7 / 0.9 -- monotone in both, a 58 dB spread |
| K2 drains the ytterbium only | gain 14.237 / 13.850 / 13.461 dB and `eta_tr` 0.859 / 0.797 / 0.730 at K2 = 0 / 1.5e-22 / 4e-22; the Er RHS at a fixed state is bit-identical across all three; `k2_defect == eps_Yb * k2_rate_per_m` to 1e-12 |
| migration | excitation conserved to `1e-12` relative over 50 random states; `max|b2c - b2nc|` 0.260 -> 7.6e-4 and `eta_tr` 0.535 -> 0.912 as W goes 0 -> 1e6 |
| one-pool vs three-unknown algebra | `1.3e-13` worst over 300 random rate quadruples |
| 3x3 eigenvalues | `max Re(eig) = -6.5e2 1/s` over 300 operating points; all real |
| closure, EVERYTHING on | per-z identity `< 2e-15` relative; wall-plug residual `5.7e-5 / 1.4e-5 / 3.5e-6` of the launched pump at 161 / 321 / 641 nodes |
| transient onto solve, EVERYTHING on | `4.8e-3` dB after starting 5+ dB away; `meta['beta_yb']` is bitwise the weighted mean of the two pools |
| concentration vs the single-ion class | `1.5e-6` dB in both conventions, with the Delevaque `2f` == dark `f` identity to `1e-9` |
| thermal, uniform profile vs `at_temperature` on both ions | `1.1e-7` dB at 380 K (the 80 K step itself moves the gain by 0.094 dB) |
| thermal feedback, hot cladding-pumped | 2 iterations, `max_dT = 0.165 K`, peak core 331.3 K, `heat_source = "rate_balance"`; `FiberAmplifier` still reports `"flux_gradient"` |
| Melkumov PhS anchors | 976 nm `1.01e-24 / 1.18e-24`, 1030 nm `1.80e-26 / 3.25e-25`, 1060 nm `2.20e-27 / 1.50e-25` -- all EXACT (printed rows, or the linear interpolation the paper's own reference values require); absorption FWHM 5.59 nm against 7.66 nm for aluminosilicate |
| Melkumov AS branch | the shipped tuples reproduced entry for entry; `ytterbium()` and `ytterbium_melkumov()` unchanged |
| P-host erbium | `sigma_a` exact at 1535 / 1550 / 1560 nm (<1%), peak at 1535.3 nm, `tau` 9.0 ms, `sigma_e(1560)` 0.62x the aluminosilicate fit, emission McCumber-derived to 1e-12 |

---

## 4. Validation residuals

`validation/eryb_two_population_anchors.py`, 241 nodes, against the anchor folder written by the
2026-09-14 literature agents. Four columns so the two changes can be told apart:

* **A** single population, `k_tr = 1e-21`, the validation record's own ion pair. The BEFORE
  column; it reproduces `VALIDATION_SUMMARY.md` section 2.
* **B** two populations at the parameters the brief names, `(f, k_fast, K2) = (0.9, 3e-21,
  2e-22)`, same ions.
* **C** the same, with the measured phosphosilicate spectroscopy that shipped alongside.
* **D** the CONTROL: `f = 0.9` with `k_tr` chosen so `f k_tr` equals A's `1e-21`. Without it, A
  and B differ in TWO things at once -- B's effective `f k_fast` is `2.7e-21`, 2.7x A's -- and
  the pool's own contribution cannot be read off.

### 4.1 Output power against the anchors that state one

`out` in watts, `1um` = the 1-um ASE as a fraction of (signal + 1-um) output. `!` marks a
relaxation that did not meet its tolerance (the deeply absorbed long Morasse cut-backs).

| anchor | measured | A single 1e-21 | B f 0.9 / 3e-21 | C B + PhS ions | D f 0.9, f*k = A |
| --- | --- | --- | --- | --- | --- |
| Exail L2 915 bwd 5 W | PCE > 0.40 | 2.480 W / 0.396 | 2.288 / 0.358 | 2.328 / 0.366 | 2.115 / 0.323 |
| Exail L2 915 bwd 10 W | PCE > 0.40 | 4.482 W / 0.398 | 4.157 / 0.366 | 4.230 / 0.373 | 3.774 / 0.327 |
| Exail L2 915 bwd 20 W | PCE > 0.40 | 8.285 W / 0.389 | 7.849 / 0.367 | 7.880 / 0.369 | 6.526 / 0.301 |
| Exail AG-EY-O-5 976 fwd 1 W | curve ~0.37 | 0.372 W / 0.362 | 0.331 / 0.321 | 0.437 / 0.427 | 0.295 / 0.285 |
| Exail AG-EY-O-5 976 fwd 2 W | curve ~0.37 | 0.773 W / 0.382 | 0.697 / 0.344 | 0.876 / 0.433 | 0.622 / 0.306 |
| Exail AG-EY-O-5 976 fwd 4 W | curve ~0.37 | 1.571 W / 0.390 | 1.430 / 0.355 | 1.760 / 0.438 | 1.274 / 0.316 |
| Bai 2015 stage 1 (12/130, 7.5 W) | 2.6 W | 3.587 W (+38%) | 2.939 (+13%) | 2.325 (-11%) | **2.517 (-3%)** |
| Bai 2015 stage 2 (12/130, 25 W) | 10.0 W | 11.650 W (+17%) | 13.629 (+36%) | 12.026 (+20%) | **10.690 (+7%)** |
| Wei 2020 (10/128, 21 W) | 8.15 W | 12.315 W (+51%) | 12.999 (+59%) | 11.894 (+46%) | **11.077 (+36%)** |
| Hannover SM-EYDF-6/125-HE 5 -> 20 W | ~48% then roll-off | PCE 0.483 -> 0.154 | 0.571 -> 0.351 | 0.557 -> 0.337 | 0.445 -> 0.152 |
| Hannover DCF-EY-6/128 5 -> 20 W | roll-off + 1-um rise | PCE 0.581 -> 0.211 | 0.581 -> 0.464 | 0.551 -> 0.466 | 0.517 -> 0.208 |
| Morasse 2006 cut-back 1.30 m | 0.59 W | 1.391 W (2.36x) | 1.061 (1.80x) | 0.906 (1.54x) | **0.932 (1.58x)** |
| Morasse 2006 cut-back 2.75 m | 0.67 W | 1.244 W (1.86x) | 0.945 (1.41x) | 0.989 (1.48x) | **0.833 (1.24x)** |
| Morasse 2006 cut-back 4.20 m | 0.56 W | 0.994 W (1.78x) | 0.750 (1.34x) | 0.818 (1.46x) | **0.657 (1.17x)** |
| Morasse 2006 cut-back 5.75 m | 0.46 W | 0.763 W (1.66x) | 0.572 (1.24x) | 0.630 (1.37x) | **0.496 (1.08x)** |
| Morasse 2006 cut-back 7.80 m | 0.215 W | 0.481 W (2.24x) | 0.361 (1.68x) | 0.428 (1.99x) | **0.304 (1.41x)** |
| Morasse 2006 cut-back 9.70 m | 0.115 W | 0.173 W (1.50x) | 0.134 (1.17x) | 0.280 (2.44x) | **0.107 (0.93x)** |

Predicted / measured output power over the nine anchors that state one (1.000 is exact):

| column | mean | median | worst |
| --- | --- | --- | --- |
| A single population, `k_tr = 1e-21` | 1.717 | 1.660 | 2.358 |
| B two populations, `f = 0.9`, `k_fast = 3e-21`, `K2 = 2e-22` | 1.414 | 1.363 | 1.798 |
| C as B, with the measured phosphosilicate spectroscopy | 1.536 | 1.460 | 2.436 |
| D two populations at the SAME effective `f k_tr` as A | **1.201** | **1.174** | **1.579** |

**Reading, and the honest version of it.** The 1.4-2x over-prediction the validation record
identified DOES shrink, and the control column says why. At the SAME effective transfer strength
(D) the mean over-prediction falls from 1.72 to 1.20 and the worst from 2.36 to 1.58: the
uncoupled pool takes its share of the pump as fluorescence and 1-um ASE, exactly the mechanism
VALIDATION_SUMMARY section 4 item 1 named. The Bai 2015 stages, which are the cleanest measured
pair, go from +38% / +17% to -3% / +7%.

The B column -- the parameters the brief names -- shrinks the mean to 1.41 but is NOT a clean
demonstration on its own: its effective `f k_fast = 2.7e-21` is 2.7x column A's coefficient, so
it changes two things at once, and on the two large-core watt-class points (Bai stage 2, Wei) the
stronger transfer wins and the over-prediction GROWS. Two mechanisms pull in opposite directions
there and the brief's parameters are on the strong-transfer side of the crossover. Reporting only
B would have hidden that; D is the column that isolates the pool.

The Hannover core-pumped roll-off is a separate check and it is a constraint, not a win: A and D
both reproduce it (PCE 0.48 -> 0.15 and 0.45 -> 0.15 from 5 to 20 W), while B does not (0.57 ->
0.35) because 2.7x the transfer postpones the bottleneck. Any parameter set proposed for this
model has to hold the roll-off AND the output magnitude at once; `f k_tr ~ 1e-21` does, `2.7e-21`
does not.

Column C (the measured phosphosilicate ions) is a mixed result and should not be read as an
improvement on its own: it helps Bai stage 1 and Wei, hurts the Morasse long cut-backs, and
raises the predicted 1-um ASE fraction on the large-core points to 10-16% because the Melkumov
P-host `sigma_e_Yb` is larger relative to `sigma_a_Yb` at 1030-1060 nm than the parametric model
was. See limit 6: Morasse himself needed 0.4x the McCumber `sigma_e_Yb` to match his measured
1-um ASE, so a 2x uncertainty on every 1-um magnitude in this table stands.

Nothing here was tuned: `f = 0.9` and `K2 = 2e-22` are the brief's values, `k_fast = 3e-21` is
Cheng's measured fast-pair rate, and D's `k_tr = 1.11e-21` is fixed by requiring `f k_tr` to
equal column A's own device-validated coefficient.

### 4.2 The FORC ETE claim: the 1-um ASE fraction becomes coupled-fraction dependent

Exail AG-EY-O-5, 2 W of 976 nm, `f k_fast` held at 2.7e-21 while `f` is swept:

| f | eta_tr | signal out (W) | 1-um out (W) | 1-um fraction | 1-um parasitic gain |
| --- | --- | --- | --- | --- | --- |
| 0.20 | 0.218 | 0.291 | 0.708 | 70.9% | +27.4 dB |
| 0.30 | 0.325 | 0.389 | 0.491 | 55.8% | +25.2 dB |
| 0.40 | 0.433 | 0.468 | 0.316 | 40.3% | +22.6 dB |
| 0.60 | 0.639 | 0.587 | 0.062 | 9.5% | +14.1 dB |
| 0.90 | 0.852 | 0.697 | 3.0e-5 | 0.004% | -39.0 dB |
| 1.00 | 0.907 | 0.723 | 2.3e-6 | 0.000% | -56.8 dB |

FORC measure 21% -> 0.5% of the output at 1 um as the ETE rises over 0.2-0.4, and 0.05% at ETE
0.9. The model now reproduces the BEHAVIOUR -- four orders of magnitude of 1-um fraction driven by
the coupled fraction alone, at a fixed effective transfer coefficient -- which the
single-population model cannot express at all (column A's 1-um fraction never moves outside
0-9% across the whole anchor table). The MAGNITUDES do not match: the model gives 71% where FORC
measured 21% at ETE 0.2, and 0.004% where they measured 0.05% at ETE 0.9, i.e. it over-predicts
the low-ETE end by ~3x and under-predicts the high-ETE end by ~10x. Both are inside the stated
factor-2-per-decade uncertainty on `sigma_e_Yb` (limit 6) compounded by the exponential
sensitivity of an ASE fraction to a 20-30 dB single-pass gain, and neither was tuned. The claim
this supports is the ORDERING and the ETE dependence, not the number.

---

## 5. Limits the caller has to know

1. **`W_mig` has no source.** Every other parameter here is measured or fitted in a named paper;
   the migration rate is not. It defaults to 0 and it is the right knob for reconciling a
   decay-measured `k_tr` with a device-fitted one, but a value for it has to come from elsewhere.
2. **Dark erbium accepts no transfer.** A quenched pair is a strong acceptor in reality, and a Yb
   excitation transferred into one is pure loss. Here dark ions only absorb (the single-ion
   convention). At the 1.6-3.2% pair fractions measured on Er:Yb Al:P fiber the omission is
   small; at a large pair fraction it is not.
3. **The transfer coefficients are isothermal.** `k_tr`, `K2` and `W_mig` do not move with
   temperature even when a profile is set, though Cheng 2022 measure ~8% over 300-480 K.
4. **The march refuses a temperature profile** rather than silently running the cold
   cross-sections. Clearing the profile or using `solve()` is the way round it.
5. **The P-host erbium is an ANCHORED model, not a fitted spectrum.** It passes exactly through
   the measured 1535 / 1550 / 1560 nm points and is within 8% at 1530 nm, but it reads 29% low at
   1545 nm and runs high beyond 1565 nm; below 1525 nm the measured P-host absorption has a blue
   shoulder NEITHER host model reproduces. Validity 1530-1565 nm. Its pump bands are deliberately
   the aluminosilicate ones (1-2% of an EYDFA's pump absorption; an untested assumption for a
   P-host Er-only 980-pumped amplifier).
6. **The Yb emission scale carries a factor-2 uncertainty on the 1-um gain.** Morasse 2006 needed
   0.4x the McCumber `sigma_e_Yb` to match his measured 1-um ASE. The Melkumov table shipped here
   is measured rather than McCumber-derived, but the 1-um ASE magnitudes below should still be
   read as order-of-magnitude.
7. **The reference preset is a starting point, not a device model.** Its 0.8 m is the
   length/gain crossing `eryb_reference_parameters.md` derives at `k_tr = 7.5e-23`; at the
   pair-rate coefficient the preset carries, the ytterbium drains ~7x faster, the
   pump-absorption length collapses to ~0.2 m at 1 W, and the length has to be re-optimised.
   That is gated rather than hidden.
8. **A two-population fit is under-determined by one observable.** Dong's own paper fits two
   mutually inconsistent `f = 1` coefficients (1.1e-21 from the Yb parasitic threshold, 2.63e-21
   from the maximum output) to the SAME laser. Adding `f` gives the model the freedom to fit both
   -- and therefore the freedom to be fitted to either one alone and be wrong about the other.
   Fit `f` and `k_tr` to the 1-um fraction AND the output power together, or not at all.

---

## 6. Verification run

```
ruff check dynameta/ tests/ validation/                       -> clean
pytest tests/test_fiber_eryb_physics.py                       -> 25 passed
pytest tests/test_fiber_eryb.py tests/test_fiber_eryb_transient.py
       tests/test_fiber_thermal_feedback.py
       tests/test_measured_spectra_2026_08_28.py
       tests/test_yb_mccumber_refit.py                        -> 58 passed
python -m validation.eryb_two_population_anchors --nodes 241  -> section 4
```
