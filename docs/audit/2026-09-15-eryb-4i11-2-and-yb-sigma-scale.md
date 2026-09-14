# Er:Yb explicit 4I11/2 and the Yb emission scale (2026-09-15)

Two library upgrades to `ErYbAmplifier`, both opt-in, both driven by a named measurement the
previous model could not express.

1. **The Er `4I11/2` level becomes explicit** (`tau32_s`). The model has always eliminated it
   adiabatically -- correct for a high-phosphorus host, where `A_32 ~ 1e6 1/s`, and the assumption
   every one of this repo's Er:Yb numbers rests on. But the measured `4I11/2` lifetimes are 7 us
   (Sefler 2004), 10 us (de Varona 2019 Table 5.1) and 0.1-10 us (Canat 2006), while Dong et al.
   2020 carry `tau_32 = 1 ns` as an admitted *modelling convention* chosen to make back-transfer
   negligible. At 7-50 us the level holds population, the erbium ground state it takes that
   population from is the transfer acceptor, and the back-transfer `k_back n3 n5c` -- which the
   adiabatic model could only approximate through the `phi` branching factor -- becomes a real
   loss channel.
2. **The Yb emission cross-section gains a scale** (`yb_sigma_e_scale`) and a calibration helper
   (`eryb_fit_yb_sigma_e_scale`). Morasse et al. 2006 measured *every* parameter of a CorActive
   Er:Yb fiber and still over-predicted the backward 1-um ASE by a factor 1000 (19.5 mW against
   0.079 mW) until the McCumber-derived Yb `sigma_e` was scaled by 0.4, with the signal output
   insensitive. `VALIDATION_SUMMARY.md` section 4 item 6 carries that as a standing factor-2
   uncertainty on every 1-um magnitude this model reports. It is now a fitted parameter with a
   measurement behind it.

With `tau32_s = None` and `yb_sigma_e_scale = 1.0` -- the defaults -- the class reproduces
v0.11.2 bit for bit, asserted in one process rather than argued.

Branch `feat/eryb-explicit-4i11-2-and-yb-sigma-scale`. Version `0.11.2 -> 0.11.3`.

---

## 1. What was built

| Where | What |
| --- | --- |
| `eryb.py` | `tau32_s`, `upconversion_via_4i11_2`, `er_4i11_2_zero_line_m`, `yb_sigma_e_scale`; `_n3_coeffs`, `_n3_closed_form` (the quadratic inner solve), `_solve_f3fb` (the four-unknown z-local steady state), `_fbb_n3`, `_level_split_J`, `_rates_profile_n3`, `_fb_rhs_n3` / `_fb_jacobian_n3` (the 4-state RHS and exact 4x4 Jacobian), `_n3_quasi_equilibrium`, `_back_transfer_ratio`, `with_yb_sigma_e_scale` |
| `eryb.py` | `energy_terms(..., f3=)` gains `relaxation_32_defect` / `back_transfer_defect` / `relaxation_32_rate_per_m` / `back_transfer_rate_per_m` / `df3_dt`; `stored_energy_J(..., f3=)` counts the 4I11/2 quantum; `_heat_profile_W_per_m` passes `f3` through |
| `eryb.py` | `YbSigmaEScaleFit` + `eryb_fit_yb_sigma_e_scale(amp, measured_1um_ase_W, direction)` -- the bracketed one-parameter fit |
| `dynamics.py` | the FOUR-reservoir march (`_split_eryb_seed_n3`, the n = 4 / n = 3 exponential Rosenbrock step through the existing `_phi1_dt_nxn`), `meta['er_4i11_2']`, the 4I11/2 frame on `frame_as_steady` |
| `__init__.py` | `YbSigmaEScaleFit`, `eryb_fit_yb_sigma_e_scale` exported |
| `tests/test_fiber_eryb_4i11_2.py` | 19 gates |
| `validation/eryb_morasse_yb_sigma_scale.py` | the Morasse 2006 fit + the tau_32 sweep on the same fiber (a reporting script) |

### API -- exact call forms

```python
from dynameta.optics.fiber_amp import (ErYbAmplifier, AseBand, eryb_fit_yb_sigma_e_scale,
                                       simulate_transient)

amp = ErYbAmplifier(er_ion, yb_ion, fiber, pumps, signals, ase,
                    n_yb_m3=4.0e26, k_tr_m3_s=1.0e-21,
                    tau32_s=7.0e-6,                 # None (default) = the adiabatic fast limit
                    k_back_m3_s=1.0e-22,            # now EXPLICIT: k_back n3 n5c
                    upconversion_via_4i11_2=False,  # Dong 2020 Eq. (2) routing; needs tau32_s
                    er_4i11_2_zero_line_m=977e-9,   # sets the level-2/3 channel split and eps_3
                    yb_sigma_e_scale=0.4,           # 1.0 (default) = unchanged
                    yb_ase=AseBand(1.00e-6, 1.10e-6, 10))

r = amp.solve(n_nodes=321)
r.meta["er_4i11_2_z"]          # f3 = n3/N_Er (None in the adiabatic default)
r.meta["back_transfer_ratio"]  # INT k_back n3 n5c dz / INT k_tr n1 n6c dz
r.meta["tau32_s"], r.meta["yb_sigma_e_scale"]

terms = amp.energy_terms(r.power_W, r.nbar2_z, r.meta["beta_yb_z"], f3=r.meta["er_4i11_2_z"])
terms["relaxation_32_defect"], terms["back_transfer_defect"]      # the two new heat channels
amp.stored_energy_J(r.nbar2_z, r.meta["beta_yb_z"], r.z_m, f3=r.meta["er_4i11_2_z"])

tr = simulate_transient(amp, t_grid, n_nodes=161, nbar2_0=(0.02, 0.0, 0.02))  # (f2, f3, b2)
tr.meta["er_4i11_2"]           # (Nt, Nz)

fit = eryb_fit_yb_sigma_e_scale(amp, 0.079e-3, "bwd", n_nodes=201, relax=0.5)
fit.scale, fit.signal_change_dB, fit.ase_1um_unit_scale_W
amp2 = amp.with_yb_sigma_e_scale(fit.scale)
```

---

## 2. Derivations

### 2.1 The explicit 4I11/2 balances

With `f2 = n2/N_Er`, `f3 = n3/N_Er`, `f1 = 1 - f2 - f3` and `N_Er = n1 + n2 + n3`:

```
Er2:  W12 f1 - W21 f2 - f2/tau_Er + f3/tau_32 - c_up C_up N_Er f2^2                      = 0
Er3:  W13 f1 - W31 f3 - f3/tau_32 + k_tr b2c (f N_Yb) f1 - k_back f3 (f N_Yb)(1 - b2c)
      + r_up C_up N_Er f2^2                                                              = 0
Ybc:  R_a_Yb (1-b2c) - R_e_Yb b2c - b2c/tau_Yb - k_tr b2c N_Er f1 - K2 b2c N_Er f2
      + k_back N_Er f3 (1 - b2c) - W (1-f)(b2c - b2nc)                                   = 0
Ybnc: R_a_Yb (1-b2nc) - R_e_Yb b2nc - b2nc/tau_Yb + W f (b2c - b2nc)                      = 0
```

This is Dong et al. 2020 Eqs. (1)-(2) in this module's per-ion-fraction variables, with the
`n5c`/`n6c` coupled-pool bookkeeping of the 2026-09-14 branch kept intact.

**The channel split.** `W12/W21` and `W13/W31` are the same per-ion absorption and emission rate
sums the adiabatic model calls `R_a_Er` / `R_e_Er`, split over the channels by which erbium level
the photon terminates on. The criterion is the photon energy, not a hand-assigned band edge: a
channel is LEVEL-3 when `h nu_k` is closer to the `4I11/2` zero line (`er_4i11_2_zero_line_m`,
977 nm, the model's own pump-band absorption peak) than to the `4I13/2` one
(`RareEarthIon.eps_J`), i.e. below **1192 nm** for the shipped ions. Pumps at 915-1018 nm and a
1000-1100 nm Yb ASE band are level-3; a 1480 nm in-band pump and the C band are level-2 -- the
physically right assignment, and no channel an EYDFA carries sits within 140 nm of the boundary.
`W12` is computed as the DIFFERENCE `R_a_Er - W13` (and `W21` as `R_e_Er - W31`) rather
than as a second masked sum, so no channel can be dropped or counted twice; the round trip
is exact to one float subtraction, gated at `1e-15`.

**What the propagation does with it.** Every channel absorbs from `n1` alone (the ground fraction
becomes `1 - f2 - f3`), a level-2 channel emits from `n2` and a level-3 channel from `n3`, and
the guided spontaneous source follows the same rule. `n3` provides NO C-band gain: the
`4I11/2 -> 4I13/2` mid-infrared transition and `4I11/2` excited-state absorption are not modelled,
which is stated in the module docstring rather than implied. For every Er spectrum this repo
ships the level-3 emission is dead anyway -- `sigma_e_Er(976) = 4.7e-33 m^2`, eight orders below
`sigma_a` there, because the emission is McCumber-derived about the 1530 nm zero line -- so `W31`
is numerically zero; the machinery exists for a tabulated ion with a real 980 nm emission band.

**Back-transfer replaces phi.** With the level explicit, `k_back n3 n5c` is carried as written and
`phi` is forced to exactly 1.0 (`_phi` returns ones whenever `tau32_s` is set). Applying both
would charge the same branching twice. The adiabatic path keeps `phi` unchanged.

**The upconversion routing.** Dong 2020 Eq. (2) writes `-2 C_up n2^2` out of `4I13/2` and
`+C_up n2^2` into `4I11/2`; this module has always written the net `-C_up N_Er f2^2`. Both are the
same model once `n3` relaxes instantly, so `upconversion_via_4i11_2` is (a) default False, (b)
REFUSED when `tau32_s` is None rather than accepted as a silent no-op, and (c) gated to agree
with the default routing to `< 1e-6` dB at `tau_32 = 1 ps` and to DISAGREE at 50 us.

### 2.2 The reduction stays a bracketed scalar -- through a quadratic

At fixed `(f2, f3)` the two ytterbium balances are still linear in `(b2c, b2nc)`: the new
back-transfer source `k_back N_Er f3 (1 - b2c)` contributes a constant `kb_yb f3` to the coupled
right-hand side and a drain `kb_yb f3` to its diagonal. So the 2026-09-14 closed form survives,
with

```
D   = R_a_Yb + R_e_Yb + 1/tau_Yb,        DW  = D + W f
Dc  = D + kout f1 + K2n f2 + kb_yb f3  = P0 + P1 f3,   P1 = kb_yb - kout
Sc  = R_a_Yb + kb_yb f3
det = DW Dc + W (1-f) D                = E0 + E1 f3    > 0 on the whole interval
b2c = [Sc DW + W (1-f) R_a_Yb] / det   = (C0 + C1 f3) / det
b2nc= [(Dc + W (1-f)) R_a_Yb + W f Sc] / det
```

Both the numerator and the denominator of `b2c` are LINEAR in `f3`, so multiplying the `4I11/2`
balance by `det` (positive throughout, so no root is created or destroyed) gives a QUADRATIC:

```
Q(f3) = a2 f3^2 + a1 f3 + a0,     S = W13 g0 + r_up cu f2^2,  T = W13 + W31 + A3,  g0 = 1 - f2
a0 = S E0 + kin g0 C0
a1 = S E1 - T E0 + kin (g0 C1 - C0) - kb_er (E0 - C0)
a2 = -T E1 - kin C1 - kb_er (E1 - C1)
```

`Q(0) = a0 >= 0` (a sum of non-negative products) and `Q(g0) <= 0` whenever the level is pumped
from below, so exactly ONE root lies in `[0, g0]` -- a quadratic cannot return to its starting
sign inside an interval whose endpoints straddle zero. It is taken with the cancellation-free
pair `q = -(a1 + sign(a1) sqrt(disc))/2`, roots `q/a2` and `a0/q`, selecting whichever lands in
`[0, g0]`. That spelling also degenerates correctly as `a2 -> 0` (`a0/q -> -a0/a1`, the linear
root), which matters because `a1` and `a2` both blow up like `A_32` in the stiff limit: at
`tau_32 = 1 ps` the root is ~2.7e-9 and is still resolved to full precision.

Substituting into the `4I13/2` balance leaves the SAME single bracketed scalar the module has
always solved,

```
H(f2) = W12 (1 - f2 - f3(f2)) - W21 f2 - f2/tau_Er + f3(f2)/tau_32 - c_up C_up N_Er f2^2
dH/df2 = -W12 (1 + f3') - W21 - 1/tau_Er + f3'/tau_32 - 2 c_up C_up N_Er f2
f3'   = -(dQ/df2)/(dQ/df3),   dQ/df3 = 2 a2 f3 + a1,   dQ/df2 = da0 + da1 f3   (da2 = 0)
```

with `H(0) >= 0` and `H(1) < 0`, so `_bracketed_newton` is as unconditionally robust here as in
the one- and two-pool cases. **There is no inner iteration anywhere**: the ytterbium pair is a
closed form in `f3`, `f3` is a closed form in `f2`, and only `f2` is solved numerically. The one
safeguard is the clamp `f3 <= 1 - f2` (the population constraint), which is also what keeps
`H(1) < 0` when a ROUTED upconversion would otherwise feed `n3` at `f2 = 1`.

### 2.3 Energy closure

The erbium stores a third quantum and every erbium optical event is charged at the level it
actually terminates on:

```
U      = A (eps_Er N_Er f2 + eps_3 N_Er f3 + eps_Yb N_Yb bbar)
Phi_tr = A k_tr N_Er f1 (f N_Yb) b2c      Phi_32 = A N_Er f3 / tau_32
Phi_bk = A k_back N_Er f3 (f N_Yb)(1 - b2c)

D_Er = SUM_{lvl2} (h nu - eps_Er)(n_a - n_e) + SUM_{lvl3} (h nu - eps_3)(n_a - n_e)
       + (eps_Er Phi_dec_Er - P_sp_lvl2) + (c_up eps_Er - r_up eps_3) Phi_up
D_tr = (eps_Yb - eps_3) Phi_tr
D_32 = (eps_3 - eps_Er) Phi_32 - P_sp_lvl3
D_bk = (eps_3 - eps_Yb) Phi_bk

q_opt(z) = dU/dt(z) + D_loss + D_Er + D_Yb + D_tr + D_K2 + D_32 + D_bk
```

Three readings.

* **`D_tr + D_32` is the adiabatic transfer defect, redistributed.** Once every transferred ion
  relaxes immediately, `Phi_32 -> Phi_tr + (direct 976 nm pumping)` and
  `(eps_Yb - eps_3) Phi_tr + (eps_3 - eps_Er) Phi_32` collapses to `(eps_Yb - eps_Er) Phi_tr` plus
  the part of the direct erbium pumping that used to be charged inside `D_Er`. So the TOTAL does
  not move and only the naming does -- which is the gate, and the reallocation is *exactly* equal
  and opposite between `D_Er` and the pair (measured below).
* **`D_bk` is negative, and that is not a sign error.** `eps_3 = 2.0332e-19 J` (977 nm) sits
  *below* `eps_Yb` (`2.0384e-19` at the parametric ytterbium's 974.5 nm zero line,
  `2.0421e-19` at the Melkumov phosphosilicate table's 972.75 nm), so back-transfer is uphill by
  **3.3-5.5 meV** depending on the ion and ABSORBS a phonon per event. It is a real, tiny
  anti-Stokes term. Its magnitude on the reference fixture at `tau_32 = 7 us`, `k_back = k_tr`, is
  `-0.052 W/m` summed against a `+14.06 W/m` relaxation term.
* **The level-3 guided spontaneous emission is credited against the relaxation** (`- P_sp_lvl3`),
  the same way C-band ASE is credited against the `4I13/2` decay. The model carries no separate
  radiative branching ratio for `4I11/2`, whose relaxation in a phosphosilicate host is
  multiphonon by ~1e3; that is stated rather than buried, and the term is numerically zero for
  every shipped ion.

### 2.4 The integrator

The march state becomes `y = (f2, f3, b2c, b2nc)` -- or `(f2, f3, b2c)` with one ytterbium pool --
advanced by the SAME exponential Rosenbrock step through the SAME size-parametrized `phi_1`
kernel `_phi1_dt_nxn`, now at `n = 4` (or 3), with the exact Jacobian `_fb_jacobian_n3`. No new
scheme, and the 2x2 and 3x3 spellings are untouched.

The exactness matters more here than anywhere else in this file: `1/tau_32` is `1e5-1e9 1/s`
against `1/tau_Er = 1e2`, so the `4I11/2` row is up to seven orders stiffer than the level it
feeds. Structure: `J[0][3] = J[1][3] = J[3][0] = J[3][1] = 0` (the uncoupled ytterbium and the
erbium meet only through the coupled pool) and the diagonal is strictly negative, but the
off-diagonal signs are NOT all positive -- `J[0][1] = A_32 - W12 > 0` while
`J[2][0] = (kout - K2n) b2c` changes sign with `K2` -- so **no determinant theorem is claimed**.
The property the scheme needs is gated numerically instead, and the measurement carries a
finding: over 300 random operating points `max Re(eig) = -1.5e3 1/s` and `max |Im(eig)| = 5.7e4`,
i.e. **complex pairs do occur** at four states, unlike the 2x2 and 3x3 cases where the
eigenvalues stayed real. That is exactly why `phi_1` is evaluated on the matrix by
scaling-and-squaring rather than through an eigen-decomposition's divided difference.

### 2.5 The Yb emission scale

One scalar on `sigma_e_Yb` at every channel, applied in `_plan` -- the single place the channel
spectra are assembled -- so the modal gain, the stimulated-emission rate, the ASE spontaneous
prefactor, the McCumber products and the transfer-efficiency denominator all scale together. The
1030 nm parasitic-gain diagnostic reads the ion directly and is scaled there too, or it would
contradict the ASE it exists to predict. `sigma_a_Yb` is NOT touched.

**The decision, and its cost.** Morasse's 0.4 was on `sigma_e` only and this follows him. The
quantity that pins `sigma_a_Yb` is the fiber's MEASURED pump absorption in dB/m, which is not in
question; the 1-um over-prediction is an emission statement. The price is that the scaled spectra
no longer satisfy McCumber: the inversion clamp at the pump line moves from
`sigma_a/(sigma_a + sigma_e)` to `sigma_a/(sigma_a + s sigma_e)`, which on the Melkumov
phosphosilicate table at 976 nm is **0.461 -> 0.639** at `s = 0.4`. Scaling both spectra would
preserve McCumber but would rescale the pump absorption length, contradicting a measured
`dB/m` -- a worse trade. Note also that in THIS library `sigma_a_Yb` is an independent tabulated
or parametric spectrum rather than a McCumber transform of `sigma_e_Yb`, so there is no derived
absorption tail to keep consistent; that question arises only for an ion whose emission was
derived from its absorption, which is Morasse's case and the reason his correction was needed at
all.

**The fit.** `eryb_fit_yb_sigma_e_scale` runs a bracketed root find on `ln ASE_1um(ln s)` (the
1-um ASE is exponential in the Yb gain integral, so log-log is the natural variable), with the
bracket established by an actual sign change through geometric expansion and closed by Brent's
method. It reports the scale, the 1-um ASE before and after, and -- the number the correction has
to be judged on -- the SIGNAL change it causes. Monotonicity is not assumed; if no scale in
`[scale_min, scale_max]` reaches the target the function raises rather than returning an endpoint.

---

## 3. Measured gate numbers

`tests/test_fiber_eryb_4i11_2.py`, 19 gates. Two fixtures: the **cladding-pumped** reference of
`test_fiber_eryb_physics.py` (6 um core, `N_Er` 4e25, `N_Yb` 4e26, `k_tr` 2e-22, 1 W of 976 nm
into a 125 um cladding, 20 mW seed, 3 m, 13.82 dB) and the **core-pumped** reference of the
2026-09-13 closure note (2.8 um core, `N_Er` 2e25, `N_Yb` 2e26, 200 mW of 976 nm, 0.5 mW seed,
2 m, 8.51 dB), which is the point the brief names for the `tau_32` sensitivity.

| Gate | Claim | Measured |
| --- | --- | --- |
| defaults, bitwise | spelling out `tau32_s=None`, the 4I11/2 zero line and `yb_sigma_e_scale=1.0` changes NO float | exact array equality across the steady solve, `energy_terms` and the march, in one process |
| defaults, structural | the new branches cannot be entered | `_n3 is False`, `_coeffs` carries none of the level-3 keys, `meta['er_4i11_2_z'] is None`, `energy_terms` REFUSES an `f3`, the adiabatic `phi` is still `< 1` when `k_back > 0` |
| `tau_32 -> 0` | reproduces the adiabatic model to 1e-6 dB | **4.2e-9 dB** at `tau_32 = 1 ps` (cladding, 161 nodes), `max f3 = 2.7e-9`. Core-pumped: **1.9e-6 dB**, and that residual is NOT the population -- it does not scale with `f3` at all (`+4.1e-7 / +1.8e-6 / +1.9e-6 / +6.5e-7 / -2.6e-5` dB at `tau_32 = 1e-16 / 1e-14 / 1e-12 / 1e-10 / 1e-8` while `f3` spans 4e-14 to 4e-6). It is the adaptive-LSODA path: the explicit route assembles the same total rate as `(R_a_Er - W13) + W13`, whose float is not bit-identical, and 2e-7 relative is the same integrator noise the 2026-09-13 note measured at 6.2e-4 across environments |
| the split conserves the totals | `W12 + W13 == R_a_Er`, `W21 + W31 == R_e_Er` | equal to `1e-15` relative; the level-3 mask is 1 below 1.15 um and 0 above 1.25 um, crossover 1192 nm |
| closure total, adiabatic limit | unchanged; only the naming moves | total dissipation **3.6e-10** relative; the reallocation is exactly equal and opposite, `(D_tr + D_32) - D_tr(adiabatic) = +0.18267` against `D_Er(adiabatic) - D_Er(explicit) = +0.18267`, agreeing to **1.5e-10 of the total dissipation** (the scale the O(`f3`) error lives on; the moved piece is only 0.7% of the total, so quoted against ITSELF that is 2.3e-8). Evaluating both splits on the SAME profile -- which removes the two independent LSODA paths and leaves only the bookkeeping -- gives 2.4e-9 of the total, i.e. exactly the `f3 = 2.7e-9` the limit carries. The relaxation term is 143x the transfer defect once split -- `eps_Yb - eps_3` is a 3 meV near-resonance and `eps_3 - eps_Er` the 0.46 eV multiphonon drop |
| closure, EVERYTHING on | per-z identity `< 1e-11`, wall-plug `< 1e-5` of the launched pump | per-z **2.0e-14**; residual/pump **5.41e-5 / 1.35e-5 / 3.38e-6** at 161 / 321 / 641 nodes -- 4x per doubling exactly, i.e. the `O(dz^2)` trapezoid on the rate integral, MESH-limited rather than model-limited (explicit level + explicit back-transfer + routed upconversion + two Yb pools at `k_tr = 3e-21` + K2 + migration + concentration; at that transfer coefficient the pump is gone in centimetres). The simpler explicit-level-plus-back-transfer configuration closes at **3.4e-6 already at 161 nodes** |
| `tau_32 = 7 us`, core-pumped | a stated small gain change, `n3/N_Er < 1%` | gain **8.5106 -> 8.4922 dB, -0.0184 dB**; `max f3 = 2.9e-3` (mean 5.5e-4), i.e. **0.29% of N_Er**, under the 1% bar. At 50 us: `-0.129 dB` and `f3 = 2.0%` |
| `tau_32` monotonicity | longer `tau_32` -> more 4I11/2, less gain | cladding: `-1.7e-3 / -1.2e-2 / -9.2e-2` dB and `f3 = 2.7e-3 / 1.9e-2 / 1.2e-1` at 1 / 7 / 50 us |
| back-transfer | measurable at 50 us, monotone in `tau_32` at fixed `k_back` | at `k_back = k_tr = 2e-22` (Dong's own `C36 = C63`): returned fraction **0.071 / 0.345 / 0.596 / 0.784** and gain **13.771 / 13.484 / 12.928 / 11.727 dB** at `tau_32 = 1 / 7 / 20 / 50 us`. At 50 us the penalty against `k_back = 0` (13.729 dB) is **2.00 dB**. With `k_back = 0` the reported ratio is exactly `0.0` |
| upconversion routing | a no-op in the adiabatic limit, refused without `tau32_s` | `< 1e-6` dB at `tau_32 = 1 ps`, `> 1e-4` dB at 50 us; `ValueError` when `tau32_s is None` |
| steady state is the march's fixed point | `F(y*) = 0` on the solved profile | all four components `< 1e-12` of the largest rate in their own balance |
| 4x4 eigenvalues | strictly negative real parts | `max Re(eig) = -1.5e3 1/s` over 300 random operating points; `max |Im| = 5.7e4` -- complex pairs occur (see 2.4) |
| march onto `solve()` | settles | constant drive from `solve()`: `max |gain - steady| = 1.1e-3 dB` over 40 ms; started 88 dB away (`-74.6 dB`): **7.5e-4 dB** from steady |
| stiff `dt` sweep | finite, `f2, f3 >= 0`, `f2 + f3 <= 1`, for `dt` 1e-8 .. 1e-3 s | all finite and inside the simplex; population overshoot before the clip **0.0** at every step, `||dt J||_inf` from 1.9e-3 to 192 |
| `yb_sigma_e_scale = 1.0` | bitwise identical | exact array equality on the solve and on the parasitic diagnostic |
| the scale reaches every consumer | plan spectrum, modal gain, ASE source, flux, diagnostic | all scale by `s` (to 1e-15, the associativity of the product chain); `sigma_a_Yb` and both Er spectra are untouched arrays |
| the asymmetry the fit rests on | 1-um moves by orders, signal by hundredths of a dB | `s = 0.3 -> 1.0` moves the backward 1-um ASE by **14.6x** (9.75 uW -> 142 uW) and the signal by **0.032 dB** (0.4785 -> 0.4821 W) |
| synthetic recovery | a known 0.4 comes back as 0.40 +- 0.02 | **0.39996** (error 4.4e-5) in 6 solves, backward; the forward direction recovers it too |
| the fit's refusals | no `yb_ase`, bad direction, overlapping bands, unreachable target | `ValueError` / `RuntimeError`, each with the reason |

---

## 4. The Morasse 2006 fit

`validation/eryb_morasse_yb_sigma_scale.py`, 201 nodes. Every fiber parameter is Morasse's
measured one: core 10.01 um, NA 0.186, dopant radius 4.9 um, `N_Yb` 8.5e26, `N_Er` 4.7e25,
`tau_Yb` 835 us, `tau_Er` 9.0 ms, cladding 13532 um^2 (65.6 um equivalent radius), background
595 dB/km, 4.31 W at 914.8 nm co-propagating in the cladding, 7.3 mW at 1556.3 nm, `L = 2.75 m`;
measured output 0.670 W and measured BACKWARD 1-um ASE **0.079 mW**. His own model fitted
`k_tr = 1.0e-22` with `tau_32 -> 0` and no upconversion, and needed `sigma_e_Yb x 0.4`.

### 4.1 The failure, reproduced

Backward 1-um ASE predicted at `yb_sigma_e_scale = 1.0`, against the measured 0.079 mW:

| ions | `k_tr = 1e-22` | `3e-22` | `1e-21` |
| --- | --- | --- | --- |
| parametric (the 2026-09-14 validation pair) | 66.6 mW (**843x**) | 11.6 mW (146x) | 4.8 mW (61x) |
| measured phosphosilicate (Melkumov P2O5 + P-host Er) | 130.9 mW (**1657x**) | 44.3 mW (561x) | 22.9 mW (290x) |

Morasse's own number was 19.5 mW against 0.079 mW (247x). Same failure mode, same order: with
every fiber parameter measured, a 1-um ASE prediction two to three orders high.

### 4.2 The fit

At Morasse's own `k_tr = 1e-22`:

| ions | fitted scale | vs Morasse's 0.40 | 1-um ASE | signal output | signal change |
| --- | --- | --- | --- | --- | --- |
| parametric | **0.3482** | -13% | 66.6 mW -> 0.0790 mW | 0.8130 -> 0.8170 W | **+0.021 dB** |
| measured phosphosilicate | **0.3332** | -17% | 130.9 mW -> 0.0790 mW | 0.8003 -> 0.7887 W | **-0.063 dB** |

Both fits converged in 7 solves. **The result is the agreement, and the insensitivity.** Two
independently constructed Yb spectra -- a Gaussian parametric model and a measured P2O5 table --
neither of them the McCumber-derived spectrum Morasse scaled, both need 0.33-0.35 where he needed
0.40. And the signal output moves by 0.02-0.06 dB while the 1-um ASE moves by three orders, which
is precisely the observation that makes a 1-um-only correction admissible rather than a refit of
the amplifier.

What the fit does NOT establish: that 0.35 is a property of ytterbium. The scale absorbs
everything the model gets wrong about the 1-um band at this operating point -- the Yb emission
spectrum itself, the uncoupled-pool fraction (not used here), the Yb ASE bin structure, and any
error in the pump absorption that sets where the inversion sits. It is a CALIBRATION against one
measurement, and the transfer coefficient trades against it directly. MEASURED, parametric ions:

| `k_tr` | fitted scale | signal change |
| --- | --- | --- |
| 1.0e-22 (Morasse's own fit) | 0.3482 | +0.0213 dB |
| 3.0e-22 | **0.4082** | +0.0006 dB |
| 1.0e-21 (the device-validated effective value) | 0.4479 | -0.0012 dB |

One robustness check, because a fitted parameter that moved with the mesh would not be worth
quoting: re-running the whole fit at **81 nodes** instead of 201 returns 0.3484 and 0.3333 against
the 0.3482 and 0.3332 above -- 0.06% and 0.03%, i.e. the fit is mesh-insensitive even where the
unscaled 1-um ASE it fits is not (66.29 mW against 66.63 mW, 0.5%). The script's `--quick` flag
runs that reduced matrix.

A stronger transfer drains the ytterbium into the erbium and leaves less 1-um ASE to remove, so
the fitted scale RISES with `k_tr` -- through Morasse's 0.40 at `k_tr = 3e-22` and past it. The
signal insensitivity holds across the whole range (worst 0.02 dB), which is the property the
calibration rests on; the scale itself is not identified without also fixing `k_tr`. Quote it
with the ion pair AND the transfer coefficient it was fitted with.

### 4.3 The explicit 4I11/2 on the same fiber

The same measured fiber, `k_tr = 1e-22`, is also the cleanest place to read the `tau_32`
sensitivity, because nothing else about it is a guess. Adiabatic gain 20.468 dB:

| `tau_32` | d gain, `k_back = 0` | max `n3/N_Er` | d gain, `k_back = k_tr` | max `n3/N_Er` | back/fwd |
| --- | --- | --- | --- | --- | --- |
| 1 us | -0.0055 dB | 0.31% | **-0.119 dB** | 0.30% | 0.073 |
| 7 us | -0.038 dB | 2.2% | **-0.775 dB** | 1.8% | 0.352 |
| 10 us | -0.055 dB | 3.1% | **-1.072 dB** | 2.3% | 0.436 |
| 50 us | -0.273 dB | 13.7% | **-3.944 dB** | 5.9% | 0.792 |

Three readings. The BOTTLENECK ALONE is small even at 50 us (-0.27 dB): holding 14% of the erbium
in `4I11/2` costs little on this fiber, because what it removes is ground-state acceptor and the
transfer here is not acceptor-limited. The BACK-TRANSFER is what costs dB -- and it is the term
the adiabatic model cannot carry at all, since its `phi` factor is 0.999 at any plausible
`k_back`. And `n3/N_Er` at Sefler's 7 us is **2.2% here** against 0.29% on the core-pumped
reference fixture, so the "under 1%" reading of the gates is a statement about THAT operating
point and not about EYDFAs generally: the fraction tracks `tau_32` x (transfer + direct pump
rate), and 4.31 W into this fiber's 65.6 um cladding drives the erbium harder per ion than the
gate fixture does. Note also that the back-transfer LOWERS `n3` while costing gain (1.8% against
2.2% at 7 us) -- it is draining the level, not filling it.

---

## 5. Limits the caller has to know

1. **`n3` has no gain and no ESA.** The `4I11/2 -> 4I13/2` mid-infrared transition, `4I11/2`
   excited-state absorption (the `4I11/2 -> 4F7/2` channel `spectroscopy.erbium(esa=True)` models
   for the single ion) and any `4I11/2` radiative branching to ground are not represented. `n3`
   absorbs nothing, emits only at level-3 channels, and relaxes entirely to `4I13/2`. Stated for
   completeness because it is the kind of thing an audit should not have to rediscover: when a
   temperature profile is set, the level-3 erbium emission is McCumber-scaled about the `4I13/2`
   zero line like every other erbium channel, which is the wrong line for a 980 nm transition.
   That term is `~5e-33 m^2` for every shipped ion, so the error is unobservable, but it would
   not be for a tabulated ion carrying a real 980 nm emission band.
2. **`tau_32` is one number for one host.** It is not temperature-dependent here even when a
   temperature profile is set, and the same `tau_32` is applied at every `z`.
3. **The `4I11/2` zero line sets the heat SPLIT, not the total.** `er_4i11_2_zero_line_m` divides
   the transfer heat between `D_tr` and `D_32` and sets the `4I11/2` stored energy; their sum and
   the total dissipation do not depend on it. Move it only with a spectroscopic reason.
4. **The level-3 classification is an energy criterion, not a band model.** It is unambiguous for
   every channel an EYDFA carries (nothing within 140 nm of the 1192 nm crossover), but a channel
   placed near 1.2 um would be assigned by a rule that is a convention, not a measurement.
5. **The emission scale breaks McCumber by construction.** See 2.5: the pump-line inversion clamp
   moves 0.461 -> 0.639 at `s = 0.4`. Anything that reads the Yb spectra thermodynamically --
   `spectroscopy.at_temperature`, a McCumber re-derivation, `set_temperature_profile`'s own
   factors -- is being handed spectra that no longer satisfy the relation they assume. The
   temperature scaling still applies its factor about the Yb zero line, so a scaled *and* hot
   amplifier is scaled-then-McCumber-shifted, which is a stated convention rather than a derived
   result.
6. **The fitted scale is not a spectroscopic constant.** It is fitted at ONE operating point, on
   ONE fiber, with ONE `k_tr`, and it absorbs every other 1-um-band error in the model (see 4.2).
   Refit it when any of those change, and never carry it across fibers.
7. **The fit needs a 1-um band that is actually resolved.** With too few `yb_ase` bins the total
   1-um power is a coarse quadrature of a structured spectrum, and the fitted scale inherits that
   error; 10-12 bins over 1000-1100 nm is what the validation script uses.
8. **Back-transfer sees the COUPLED ytterbium only.** The acceptor density is
   `n5c = f N_Yb (1 - b2c)`, so an uncoupled ion can neither donate nor receive. That is the same
   statement the 2026-09-14 branch makes about the forward transfer and it is consistent, but it
   means a fiber modelled with a small `f` has a proportionally small back-transfer as well --
   the two are not independent knobs.
9. **A four-reservoir march is `O(dt)` on the path, as before.** The endpoints, the fixed point
   and the stability are unaffected, but `tau_32` is now the shortest time constant in the system,
   so a trajectory resolved at `dt >> tau_32` has the `4I11/2` slaved rather than resolved.
   `meta['max_dt_times_rate']` reports it.

---

## 6. Verification run

```
ruff check .                                                  -> clean (the whole tree)
pytest tests/test_fiber_eryb_4i11_2.py                        -> 19 passed, 440 s
pytest tests/test_fiber_eryb.py tests/test_fiber_eryb_physics.py
       tests/test_fiber_eryb_transient.py                     -> 57 passed, UNCHANGED
pytest tests/test_fiber_amp.py tests/test_audit_2026_08_04_fiber_amp.py
       tests/test_fiber_chain.py tests/test_fiber_dynamics.py
       tests/test_numerics.py tests/test_fiber_thermal_feedback.py
       tests/test_measured_spectra_2026_08_28.py
       tests/test_yb_mccumber_refit.py                        -> 221 passed
pytest tests/test_audit_2026_07_25_infra.py tests/test_fiber_bpm.py
       tests/test_fiber_gnlse.py tests/test_fiber_lma.py
       tests/test_fiber_nonlinear.py tests/test_fiber_srs.py
       tests/test_fiber_transverse.py                         -> 133 passed
pytest tests/test_validation_runner.py                        -> 14 passed
python -m validation.eryb_morasse_yb_sigma_scale --nodes 201  -> section 4, exit 0
```

The pytest lines above are every test file in the repo that imports `fiber_amp`, which is the
complete blast radius: this change touches `dynameta/optics/fiber_amp/eryb.py`,
`dynamics.py`, the package `__init__`, the two version strings, one validation script,
`validation/run_all.py`'s exclusion table and the docs, and nothing else. The full matrix runs on
the PR.
