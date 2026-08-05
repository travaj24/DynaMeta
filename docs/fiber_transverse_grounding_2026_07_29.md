# Fiber transverse-physics grounding dossier (2026-07-29)

Governing-equation source of truth for two planned `dynameta/optics/fiber_amp/` features:

* **Feature 1 -- radially-resolved inversion** `n2(r, phi, z)` (transverse spatial hole burning,
  TSHB): the pre-reduction form of the Giles-Desurvire model that `steady_state.py` currently
  implements only in its overlap-integral (mean-field) reduction.
* **Feature 2 -- scalar gain-BPM**: a split-step paraxial beam-propagation solver for the signal
  field inside the doped core, gain-coupled to Feature 1's local balance and (quasi-statically)
  to `thermal.py`'s heat/lens model.

Extends `docs/fiber_amp_model_spec.md` (sec. 1, 7, 8). SI units throughout; ASCII only;
`exp(-i omega t)` time convention (the repo-wide convention) unless a quoted source states
otherwise -- see sec. 2.1 for the one place this matters.

House rule applied: formulations are taken FROM the literature. Anything the literature did not
supply is derived here, is labelled **DERIVED**, and is cross-checked against a limit of a
published form and/or against direct numerical quadrature. Every number quoted as "measured"
below was produced by a scratch script during this grounding pass, not copied from a paper.

---

## 0. Provenance summary (read this first)

| Item | Status |
|---|---|
| Local two-level balance `n2(r)/nt` for K channels | **PINNED** verbatim (Smith & Smith 2013 Eq. 1) |
| Quantum-defect heat source `Q(x,y)` from the resolved inversion | **PINNED** verbatim (Smith & Smith 2013 Eq. 2) |
| Signal-free inversion clamp `n2/nt -> sa_p/(sa_p+se_p) ~ 0.5` (Yb @ 976 nm) | **PINNED** (Smith & Smith 2013, sec. 2 text) |
| Yb LMA parameter set + STRS threshold tables | **PINNED** (Smith & Smith 2013, Tables 1-5) |
| "Gain is apportioned among modes by diffraction, not by an overlap integral" (the gain-BPM design principle) | **PINNED** (Smith & Smith 2013, sec. 1) |
| TMI mechanism: MIP -> inversion grating -> thermal grating -> RIG, plus the mandatory MIP/RIG phase shift; steady models cannot get it | **PINNED** (Jauregui/Limpert/Tunnermann 2020, sec. 3) |
| Giles alpha/g*/Gamma reduction and the ASE `m h nu dnu` seed | **PINNED** via the repo's own prior 3-0-verified extraction of Giles & Desurvire 1991 (`docs/fiber_amp_model_spec.md` sec. 1), re-derived here as the `n2` = const limit of the resolved equation |
| Radial-integral propagation equation + radially-weighted ASE source | **DERIVED** (limit cross-check: reduces exactly to the pinned Giles form) |
| TSHB closed form `Phi(s0, Gamma)` for a Gaussian mode + top-hat dopant | **DERIVED** (verified against quadrature to 3.6e-13) |
| Saturation-power correction `kappa(b/w) = tanh(x^2)/x^2` | **DERIVED** (limits `b/w -> 0` gives 1, `b/w -> inf` gives 0; verified numerically) |
| LP01/LP11 TSHB gain-crossover benchmark | **DERIVED** (numeric; qualitatively matches Jiang & Marciante 2008 and Smith & Smith 2013) |
| Radial-quadrature guidance | **DERIVED** (measured convergence table) |
| Quadratic-duct BPM oracle (`w_m`, `z_p`, ABCD, `alpha^2 = D'`) | **DERIVED** from the paraxial equation via the harmonic-oscillator mapping; matches the textbook GRIN ABCD in the thin limit and `thermal.py`'s existing `D'`; verified numerically to 1.4e-5 |
| Feit-Fleck / Okamoto split-step forms | **CONVENTION-CHECKED, NOT RE-READ THIS PASS** -- see sec. 2.1 caveat |
| Jiang & Marciante, JOSA B 25:247 (2008) full model equations | **UNRESOLVED** -- see sec. 1.6 |
| Super-Gaussian absorber parameters from a specific paper | **UNRESOLVED** -- form given as a stated convention, sec. 2.5 |

---

## 1. FEATURE 1 -- radially-resolved inversion (transverse spatial hole burning)

### 1.1 The local two-level steady-state balance (PINNED)

Smith & Smith, "Increased mode instability thresholds of fiber amplifiers by gain saturation",
Opt. Express 21:15168 (2013), arXiv:1304.1064, **Eq. (1)** (transcribed verbatim; their sec. 2 is
titled "Transverse hole burning"):

```
                I_p sa_p/(h nu_p)  +  I_s(x,y) sa_s/(h nu_s)
n_u(x,y) = -----------------------------------------------------------------------
           I_p (sa_p+se_p)/(h nu_p) + I_s(x,y)(sa_s+se_s)/(h nu_s) + 1/tau
```

Generalised to K channels (the repo's channel abstraction; the generalisation is the obvious one
and is exactly the r-resolved form of the pinned mean-field expression in
`docs/fiber_amp_model_spec.md` sec. 1):

```
                    SUM_k  sigma_a_k I_k(r,phi,z) / (h nu_k)
nbar2(r,phi,z) = ------------------------------------------------------------      (F1.1)
                 1/tau + SUM_k (sigma_a_k + sigma_e_k) I_k(r,phi,z) / (h nu_k)
```

with `n2(r,phi,z) = nt(r,phi) * nbar2(r,phi,z)`.

| Symbol | Meaning | SI unit |
|---|---|---|
| `nbar2` (`n_u`) | fractional metastable population `N2/nt`, local | 1 |
| `nt(r,phi)` | dopant ion density (top-hat: `nt` for `r <= b`, else 0) | m^-3 |
| `I_k(r,phi,z)` | local optical intensity (irradiance) of channel k | W m^-2 |
| `sigma_a_k`, `sigma_e_k` | absorption / emission cross-section at `lambda_k` | m^2 |
| `nu_k = c/lambda_k` | optical frequency of channel k | Hz |
| `tau` | metastable lifetime (`RareEarthIon.tau_s`) | s |
| `h` | Planck constant | J s |

**Regime of validity** (the assumptions actually built into (F1.1)):

1. **Steady state**: `dN2/dt = 0`. Valid when everything varies slowly against `tau`
   (Er ~10 ms, Yb ~0.9 ms). The repo's `dynamics.py` is the escape hatch when it is not.
2. **Two manifolds only**, homogeneously broadened, with a single lumped lifetime. Any third
   level is assumed to relax non-radiatively so fast that its population is negligible.
3. **No energy transfer between ions**: no upconversion, no pair-induced quenching, no ASE
   reabsorption bookkeeping beyond the tracked channels. (The repo's `ConcentrationModel` terms
   would enter (F1.1) as extra loss terms in the denominator and would break the linear-fractional
   structure exploited in sec. 1.5 -- see the caveat there.)
4. **No spatial transport of excitation**: `nbar2` at `(r,phi)` depends only on the intensities at
   `(r,phi)`. Ion diffusion is nil in a glass host, so this is exact; this is what makes TSHB a
   *sharp* effect in fibers, unlike the carrier-diffusion-smoothed hole burning in the SOA model
   (`soa/transverse_bpm.py`, `L_diff = sqrt(D tau)`). **There is no `L_diff` analogue here -- the
   fiber gain medium has zero lateral smoothing.**

**Pump-scheme distinction (980 nm vs 1480 nm; three-level vs two-level).**

* **1480 nm (in-band) Er pumping and ALL Yb pumping**: the pump couples the same two manifolds as
  the signal, so `sigma_e(lambda_pump) > 0` and the pump appears in BOTH sums of (F1.1). The
  formula is then *exact* within the two-manifold picture -- this is the two-level model's home
  ground. Consequence: the signal-free inversion is clamped below unity at
  `nbar2_max = sigma_a_p / (sigma_a_p + sigma_e_p)`, the physical reason 1480-pumped EDFAs have a
  worse noise figure than 980-pumped ones.
* **980 nm Er pumping** is genuinely three-level (`4I11/2` -> fast non-radiative -> `4I13/2`). In
  the fast-relaxation limit `N3 -> 0` and it collapses onto (F1.1) with
  `sigma_e(lambda_pump) := 0`, so the pump contributes to the numerator and to the denominator
  only through `sigma_a_p`. That is exactly how `spectroscopy.erbium()` is parametrised (no
  emission peak at 980 nm), so **no code change is needed** -- the distinction is carried entirely
  by the cross-section spectra.
* **ESA** is EXCLUDED from (F1.1) by the repo's cycling-limit convention (`spectroscopy.py`
  docstring): ESA promotes an ion out of the metastable level and it relaxes straight back, so it
  removes beam power without changing the balance. Radially resolved it enters the propagation
  equation (F1.2) only, as an extra `- sigma_esa_k n2(r,phi,z)` inside the integrand.

**Pinned numerical anchor.** Smith & Smith state, for the signal-free region:

> the undepleted population in the region with `I_s = 0` is `n_u = sa_p/(sa_p + se_p) ~ 0.5`

With their Table 1 Yb values (`sa_p = 2.47e-24 m^2`, `se_p = 2.44e-24 m^2` at 976 nm) this is
`0.50305`. This is a zero-parameter gate on any implementation of (F1.1): set `I_s = 0` and a
pump intensity far above `h nu_p/(tau (sa_p+se_p))`, and `nbar2` must equal
`sa_p/(sa_p+se_p)` to round-off.

**Smith & Smith Table 1 (PINNED) -- the reference Yb LMA parameter set for all gates below:**

| Parameter | Value | Parameter | Value |
|---|---|---|---|
| `d_core` | 50 um | `d_dope` | 30-50 um |
| `d_clad` | 100-500 um | `N_Yb` (`nt`) | 3.0e25 m^-3 |
| `lambda_p` | 976 nm | `lambda_s` | 1032 nm |
| `sigma_a_p` | 2.47e-24 m^2 | `sigma_e_p` | 2.44e-24 m^2 |
| `sigma_a_s` | 5.80e-27 m^2 | `sigma_e_s` | 5.0e-25 m^2 |
| `P_p` | varies | `P_s` (seed) | 10 W |
| `dn/dT` | 1.2e-5 K^-1 | `L` | varies |
| `rho` | 2201 kg m^-3 | `C` | 702 J kg^-1 K^-1 |
| `n_core` | 1.451 | `n_clad` | 1.45 |
| `tau` | 901 us | `K` (thermal cond.) | 1.38 W m^-1 K^-1 |
| `NA` | 0.054 | `V` | 8.2 |
| `A_eff(LP01)` | 1175 um^2 | | |

Note `dn/dT`, `K` here match `thermal.DN_DT_SILICA = 1.2e-5` and `ThermalModel.core_k_W_mK = 1.38`
exactly -- the repo's silica constants are already on the published values.

### 1.2 Per-channel propagation with the radial integral (DERIVED; reduces to the pinned Giles form)

The intensity of channel k is written `I_k(r,phi,z) = P_k(z) i_k(r,phi)` with the **normalized
transverse intensity distribution**

```
INT_0^inf INT_0^2pi i_k(r,phi) r dr dphi = 1        [i_k] = m^-2                    (F1.0)
```

(for a circularly symmetric mode, `INT_0^inf i_k(r) 2 pi r dr = 1`). Then

```
dP_k/dz = u_k P_k INT [ sigma_e_k n2(r,phi,z) - sigma_a_k (nt(r,phi) - n2(r,phi,z))
                        - sigma_esa_k n2(r,phi,z) ] i_k(r,phi) dA
        - u_k l_k P_k
        + u_k m h nu_k dnu_k sigma_e_k INT n2(r,phi,z) i_k(r,phi) dA               (F1.2)
```

`dA = r dr dphi`; the integrals run over the whole cross-section (`nt = 0` outside the dopant
kills the doped-region restriction automatically).

| Symbol | Meaning | SI unit |
|---|---|---|
| `P_k(z)` | power in channel k | W |
| `u_k` | `+1` forward, `-1` backward | 1 |
| `i_k(r,phi)` | normalized modal intensity, (F1.0) | m^-2 |
| `l_k` | background (passive) loss at `lambda_k` | m^-1 |
| `m` | modes per ASE bin = 2 (two polarizations) | 1 |
| `dnu_k` | ASE bin width in optical frequency | Hz |

**Spontaneous term.** The `m h nu dnu` seed is the pinned Giles-Desurvire form
(`docs/fiber_amp_model_spec.md` sec. 1, finding [3]); the *radial* statement added here is that it
carries the **same** overlap integral `INT n2 i_k dA` as the stimulated-emission term. That is
required by construction: spontaneous emission into a given guided mode is governed by the same
matrix element as stimulated emission into it (one photon per mode), so the two must share the
weighting. This is only nonzero for `is_ase` channels, exactly as in
`steady_state._coeffs()["s_pref"]`.

**Reduction to the Giles alpha/g*/Gamma form (the required limit check).** Assume `n2` is
independent of `(r,phi)` over the doped region -- the standard "uniform inversion" closure that
Giles & Desurvire make when they replace the transverse integrals by confinement factors. Write
`n2 = nt * nbar2` with `nbar2` constant. Then `INT n2 i_k dA = nt nbar2 Gamma_k` and
`INT nt i_k dA = nt Gamma_k` with

```
Gamma_k = INT_dopant i_k(r,phi) dA        (the usual power confinement factor)
```

and (F1.2) becomes, term by term,

```
dP_k/dz = u_k [ (alpha_k + g*_k) nbar2 - alpha_k - l_k - Gamma_k nt sigma_esa_k nbar2 ] P_k
        + u_k g*_k nbar2 m h nu_k dnu_k
alpha_k = sigma_a_k Gamma_k nt ,   g*_k = sigma_e_k Gamma_k nt
```

which is **exactly** the propagation ODE of `docs/fiber_amp_model_spec.md` sec. 1 and of
`steady_state._dP_full_c()` (`g_e`, `g_a`, `g_esa`, `loss`, `s_pref`). Limit check passes.

Similarly, the closure that makes (F1.1) collapse to `steady_state._nbar2_c()` is: replace
`I_k(r,phi)` by its **area average over the doped region**,

```
<I_k> = (1/A_dope) INT_dopant I_k dA = Gamma_k P_k / A_dope ,   A_dope = pi b^2
```

which is precisely the `flux_a = gamma*sigma_a/(h nu A_dope)` grouping in
`steady_state._coeffs()`. So the current solver = (F1.1)+(F1.2) with **one area-averaged
intensity per channel**. Feature 1 is exactly "stop area-averaging".

**When the closure is NOT valid.** The closure requires the *variation of `nbar2` across the mode*
to be small. From (F1.1), `nbar2` varies on the scale of the local saturation parameter
`I_s/I_sat`; for an LP01-like mode the peak intensity exceeds the doped-area average by a factor
`~ A_dope/A_eff` (2.0 for `b = w` in the Gaussian approximation; 1.87 for the exact LP01 of the
Smith & Smith fiber, measured). So the mean-field closure is good only while
`I_peak/I_sat << 1` -- i.e. in the small-signal / low-extraction regime. Every efficient
high-power amplifier violates it; that is the entire content of Smith & Smith sec. 2.

### 1.3 TSHB in closed form for a Gaussian mode + top-hat dopant (DERIVED)

Take one saturating signal channel plus a pump whose intensity is *uniform* across the doped
region (the cladding-pump case; exactly Smith & Smith's assumption, "the [pump irradiance] is
assumed uniform across the pump cladding"). Then (F1.1) is a **linear-fractional (Moebius)
function of `I_s` alone**:

```
nbar2(I_s) = n2_inf + (n2_0 - n2_inf) / (1 + I_s/I_sat)                            (F1.3)

n2_0   = [I_p sa_p/(h nu_p)] / [I_p (sa_p+se_p)/(h nu_p) + 1/tau]   (signal-free inversion)
n2_inf = sa_s / (sa_s + se_s)                                        (infinite-signal floor)
I_sat  = (h nu_s / (sa_s + se_s)) * [ I_p (sa_p+se_p)/(h nu_p) + 1/tau ]           (F1.4)
```

`I_sat` [W m^-2] is the **pump-broadened** saturation intensity: it reduces to
`h nu_s / (tau (sa_s+se_s))` with the pump off, and grows linearly with pump intensity.

Now take a Gaussian LP01 approximation `i(r) = (2/(pi w^2)) exp(-2 r^2/w^2)` (the `waveguide.py`
Marcuse convention, `w` = 1/e field radius) and a top-hat dopant of radius `b`. Substituting
`v = exp(-2 r^2/w^2)` turns the radial integral into an elementary one:

```
J  ==  INT_0^b nbar2(r) i(r) 2 pi r dr
    =  n2_inf * Gamma  +  (n2_0 - n2_inf) * Phi(s0, Gamma)                          (F1.5)

Phi(s0, Gamma) = (1/s0) * ln[ (1 + s0) / (1 + s0 (1 - Gamma)) ]                     (F1.6)

s0    = I_peak / I_sat = 2 P_s / (pi w^2 I_sat)      (on-axis saturation parameter)
Gamma = 1 - exp(-2 b^2 / w^2)                        (= waveguide.overlap_gamma, unchanged)
```

The **mean-field** value of the same integral is

```
J_MF = Gamma * [ n2_inf + (n2_0 - n2_inf) / (1 + s_area) ] ,
s_area = <I_s>/I_sat = P_s Gamma / (pi b^2 I_sat) = s0 * Gamma / (-ln(1 - Gamma))   (F1.7)
```

so the ENTIRE resolved-vs-mean-field difference is the single bracket
`Phi(s0,Gamma) - Gamma/(1+s_area)`.

**Limit checks (all pass analytically and numerically):**

* `s0 -> 0`: `Phi -> Gamma`, `J -> n2_0 Gamma` = the unsaturated mean-field answer. Resolved and
  mean-field agree exactly in the small-signal limit -- so Feature 1 cannot change any of the
  repo's existing small-signal gates.
* `s0 -> inf`: `Phi -> -ln(1-Gamma)/s0` and `Gamma/(1+s_area) -> -ln(1-Gamma)/s0` too. The two
  models re-converge in the *deep* saturation limit as well (both extract all the stored
  inversion); the disagreement is a bump at intermediate saturation.
* `b -> 0`: `Gamma -> 0`, the intensity is uniform over the dopant and mean-field is exact.

**Verification (measured this pass).** (F1.5)/(F1.6) vs adaptive quadrature of the integral,
over `Gamma in {0.20, 0.50, 0.8647, 0.95, 0.999}` x `s0 in {1e-3, 0.1, 1, 10, 1e3}` (25 cases):
**worst relative error 3.57e-13** (quadrature tolerance limited).

### 1.4 The effective saturation-power correction factor kappa(b/w) (DERIVED)

Expand both models to first order in `s0`:

```
Phi(s0,Gamma)/Gamma        = 1 - s0 (2 - Gamma)/2 + O(s0^2)          (resolved)
[Gamma/(1+s_area)]/Gamma   = 1 - s0 Gamma/(-ln(1-Gamma)) + O(s0^2)   (mean field)
```

The ratio of the two initial saturation slopes is the factor by which the mean-field model
mis-states the saturation power. With `x = b/w` and `Gamma = 1 - exp(-2 x^2)` it collapses to

```
kappa(x) = 2 Gamma / [ (2 - Gamma) * (-ln(1 - Gamma)) ]  ==  tanh(x^2) / x^2 ,  x = b/w   (F1.8)
```

Interpretation: **`P_sat_resolved = kappa * P_sat_meanfield`.** Since `kappa < 1` for all `x > 0`,
the radially-resolved model **saturates earlier and therefore gives LESS saturated gain than the
mean-field model at the same power** (in the weak-to-moderate saturation regime). Direction is
now pinned by construction, not by hand-waving.

Limit cross-checks: `kappa -> 1` as `x -> 0` (uniform intensity over a tiny dopant -- mean-field
exact), `kappa -> 1/x^2 -> 0` as `x -> inf` (dopant much wider than the mode -- mean-field never
saturates at all because its area-average intensity vanishes, while the real mode burns a hole).

Measured table (`kappa_formula` vs the numerically extracted slope ratio; agreement 1e-5, limited
by the finite `s0 = 1e-4` used for the numeric slope):

| `x = b/w` | `Gamma` | `kappa = tanh(x^2)/x^2` |
|---|---|---|
| 0.25 | 0.1175 | 0.998700 |
| 0.50 | 0.3935 | 0.979675 |
| 0.75 | 0.6753 | 0.906364 |
| 1.00 | 0.8647 | 0.761594 |
| 1.25 | 0.9561 | 0.586128 |
| 1.50 | 0.9889 | 0.434678 |
| 2.00 | 0.9997 | 0.249832 |
| 3.00 | 1.0000 | 0.111111 |

For the repo's default (`dopant_radius_m = core_radius_m`, LP01 near cutoff so `w ~ a`, i.e.
`x ~ 1`) the correction is **24%**.

**The deficit is strictly signed and bounded (DERIVED, swept numerically).** Because
`g_res - g_MF = nt (sa_s+se_s) (n2_0 - n2_inf) [ Phi(s0,Gamma) - Gamma/(1+s_area) ]` and all the
prefactors are positive in an inverted medium, the sign is carried entirely by the bracket.
Swept over `Gamma in [0.02, 0.9999]` x `s0 in [1e-6, 1e8]` (400 x 12 points) the bracket is
**non-positive everywhere** (largest value 1.3e-10 = float round-off where the two limits
coincide). The worst-case ratio and where it occurs:

| `Gamma` | min `g_res/g_MF` | at `s0` |
|---|---|---|
| 0.20 | 0.99896 | 1.08 |
| 0.50 | 0.99013 | 1.38 |
| 0.8647 (`b = w`) | 0.92539 | 2.64 |
| 0.99 | 0.72714 | 6.95 |
| 0.9999 | 0.4825 | 19.9 |

So mean-field over-prediction is a **few percent at typical single-mode confinement and tens of
percent for a tightly-confined mode**, always peaking at `s0` of order a few. The sec. 1.5 Yb case
(`Gamma = 0.9646`, worst ratio 0.825 at `s0 = 4.6`) sits exactly on this curve -- an internal
consistency check of the whole derivation.

**Caveat that must go in the module docstring**: (F1.3)-(F1.8) hold only while `nbar2` is a
linear-fractional function of the single saturating intensity. They break if (a) more than one
*signal-band* channel is resolved with different cross-sections, (b) the pump is core-guided
(non-uniform `I_p`), (c) upconversion / PIQ is on (`_nbar2_c` then solves a quadratic). In those
cases the integral must be done by quadrature (sec. 1.7). The closed form remains the *gate*.

### 1.5 Size of the effect on a real amplifier (DERIVED benchmark)

Using the Smith & Smith Table 1 Yb fiber, `b = a_core = 25 um`, Marcuse `w = 19.34 um`
(`Gamma = 0.9646`), a 1 kW pump in a 400 um cladding (`I_p = 7.958e9 W m^-2`, giving
`n2_0 = 0.5002`, `I_sat = 7.348e10 W m^-2`), local modal gain coefficient
`g = nt [(sa_s+se_s) J - sa_s Gamma]`:

| `P_s` | `s0` | `s_area` | `g_resolved` | `g_meanfield` | ratio | difference |
|---|---|---|---|---|---|---|
| 10 W | 0.232 | 0.067 | 6.409 m^-1 | 6.705 m^-1 | 0.956 | -1.29 dB/m |
| 200 W | 4.633 | 1.337 | 2.524 m^-1 | 3.061 m^-1 | 0.825 | -2.33 dB/m |
| 1000 W | 23.16 | 6.686 | 0.828 m^-1 | 0.931 m^-1 | 0.890 | -0.45 dB/m |

So on a kW-class Yb LMA amplifier the mean-field model **over-predicts the local gain by 4-18%**,
peaking around one to a few times `I_sat` -- multi-dB over a metre of fiber. This is the headline
justification for the feature.

**Exact LP01 vs Gaussian.** For this fiber `V = 8.219` -- far outside the Marcuse formula's stated
`1.2 < V < 2.4` validity. Measured: `Gamma_LP01(b=a) = 0.99034` vs Marcuse-Gaussian `0.97918`
(1.1% apart, deceptively good), but the **saturation integral** differs by up to 13% at
`s0 = 30`, i.e. up to **0.85 dB/m of gain**, because the true LP01 at high V is much flatter than
a Gaussian (measured peak intensity ratio 0.817). **Conclusion: Feature 1 must integrate the
EXACT `lma.mode_field` LP profile, not the Gaussian approximation, whenever `V > 2.4`.**
`waveguide.mode_field_radius_m` is the wrong tool in the LMA regime; `lma.solve_lp_modes` +
`lma.mode_field` is the right one.

### 1.6 Multi-mode competition on one shared n2(r,phi,z)

The standard few-mode amplifier model: several transverse modes at (essentially) the same
wavelength share ONE spatially resolved inversion. Treating the modes as mutually incoherent
(power-only; interference terms averaged away -- the assumption a steady-state model must make,
and the one that removes the TMI physics, sec. 2.8):

```
I_s(r,phi,z) = SUM_m P_m(z) i_m(r,phi)                                             (F1.9)
nbar2(r,phi,z) from (F1.1) with this total I_s (plus the pump)
dP_m/dz = u_m P_m nt INT [ (sa+se) nbar2(r,phi,z) - sa ] i_m(r,phi) dA  - u_m l P_m
```

with `i_m` from `lma.mode_field` normalized per (F1.0); for `LP_lm` with `l >= 1` the azimuthal
factor is `cos^2(l phi)` (or `sin^2`), so the normalization is `pi INT psi^2 r dr` rather than
`2 pi INT psi^2 r dr`, and the `phi` integral does NOT factor out of the gain integral once
`nbar2` depends on `phi`.

**Literature status.** Jiang & Marciante, "Impact of transverse spatial-hole burning on beam
quality in large-mode-area Yb-doped fibers", *JOSA B* **25**(2):247-254 (2008) is the canonical
TSHB-in-LMA reference and uses exactly this structure ("a model using spatially resolved gain and
transverse-mode decomposition of the optical field"), reporting that the ASE-source beam quality
optimises when the gain saturates and that a model WITHOUT TSHB fails to reproduce the measured
behaviour. **Its equations could not be retrieved this pass**: the JOSA B article is closed
(Semantic Scholar `openAccessPdf: CLOSED` for DOI 10.1364/JOSAB.25.000247), and the previously
public LLE Review vol. 110 reprint
(`lle.rochester.edu/.../v110/110_06Impact.pdf`) now returns HTTP 404 after the LLE site
restructure (its sitemap no longer lists the LLE Review archive). **UNRESOLVED** -- flagged for a
follow-up pass with institutional access. Nothing below depends on it numerically; the direction
it reports is independently reproduced by the DERIVED gate next.

Also worth pulling in a later pass: Gong, Yuan, Li, Yan, Zhang, Liao, "Numerical modeling of
transverse mode competition in strongly pumped multimode fiber lasers and amplifiers",
*Opt. Express* **15**(6):3236 (2007) -- same equation set, multilayer numerical algorithm. Not
retrieved this pass (Optica full text is JS-gated to `WebFetch`).

**DERIVED benchmark: TSHB reverses the LP01/LP11 gain ordering.** Computed here for the Smith &
Smith fiber (`a = 25 um`, `NA = 0.054`, `lambda_s = 1032 nm`, `V = 8.2193`, `nt = 3.0e25 m^-3`,
`I_p` from 1 kW in a 400 um cladding, uniform doping `b = a`), with LP11 held at a probe level of
1 uW so it does not itself saturate:

| `P_LP01` | `g_01` | `g_11` | `g_11 - g_01` |
|---|---|---|---|
| 1 W | 7.2603 m^-1 | 7.1710 m^-1 | **-0.388 dB/m** |
| 10 W | 6.6005 | 6.7094 | +0.473 dB/m |
| 100 W | 3.6773 | 4.3166 | **+2.776 dB/m** |
| 300 W | 1.9943 | 2.5981 | +2.622 dB/m |
| 1000 W | 0.8215 | 1.1762 | +1.541 dB/m |
| 2000 W | 0.4574 | 0.6797 | +0.966 dB/m |

Mode data: `Gamma_01(b=a) = 0.99034`, `Gamma_11(b=a) = 0.97467`,
`n_eff(LP01) = 1.450937`, `n_eff(LP11) = 1.450833`, `delta n_eff = 1.043e-4`,
`L_beat = 2 pi/(beta_01 - beta_11) = 9.895 mm`.

Physics: at small signal LP01 wins because `Gamma_01 > Gamma_11`. Once the signal burns a hole on
axis, LP11 -- which peaks off-axis, where the inversion is undepleted -- gains MORE. **The
crossover is at ~5 W here and the HOM advantage peaks at +2.8 dB/m near 100 W.** A mean-field
model produces `g_11 - g_01 = const < 0` at every power and can never show this. This IS the beam-
quality degradation mechanism Jiang & Marciante report, and it is the modal-gain half of what
Smith & Smith call transverse hole burning.

Cross-check against a published direction: Smith & Smith show that population saturation
*reduces* the STRS/mode-coupling gain relative to the laser gain (their `chi'` falls with `z` as
saturation deepens, Fig. 5) and that **confining the doping raises the instability threshold**
(50 um core with 30 um doping: threshold 786 W vs 488 W fully doped at `d_clad = 100 um`,
their Tables 2 and 5). Both are consequences of the same shared-`n2` structure. Confined doping
is `FiberSpec.dopant_radius_m < core_radius_m`, already supported.

### 1.7 Radial quadrature guidance (DERIVED, measured)

The integrand of (F1.2) has **two breakpoints**:

* `r = b` (dopant edge): `nt(r)` is a genuine step -- the integrand jumps to zero.
* `r = a` (core radius): the exact LP field switches from `J_l(U r/a)` to `K_l(W r/a)`. `psi` and
  `psi'` are continuous there but `psi''` is not, so the integrand is `C^1` but not `C^2`.

Measured convergence for `INT_0^b nbar2(r) i(r) 2 pi r dr` (Gaussian mode, `Gamma = 0.8647`,
`s0 = 3`, against the exact (F1.5)):

| scheme | nodes | rel. error |
|---|---|---|
| trapezoid on `[0,b]`, node ON the edge | 65 / 257 / 1025 | 1.0e-4 / 6.3e-6 / 4.0e-7 (O(h^2)) |
| trapezoid on `[0,3b]`, `nt` as a step mask | 1025 / 4097 / 16385 | 5.3e-4 / 1.3e-4 / 3.3e-5 (**O(h)**) |
| Gauss-Legendre on `[0,b]` | 8 / 16 / 32 | 4.4e-10 / 9.0e-16 / 9.0e-16 |

**Recommendation (gate-able):**

1. Use **composite Gauss-Legendre with panel breakpoints at `r = min(a,b)`, `r = max(a,b)`** and a
   final panel out to `r_max = max(6a, a(1 + 12/W))` (the existing `lma._overlap_grid` reach).
2. **16 nodes per panel** already reaches machine precision on the smooth pieces; ship 24 as the
   default with 32 as the convergence-check setting.
3. **Never** integrate with `nt(r)` as a step mask on a uniform grid: that is first-order and
   needs >16k nodes for 3e-5. If the existing `numpy.trapezoid` style must be kept for
   consistency with `lma.dopant_overlap`, place a node exactly at `r = b` and use >= 1025 nodes
   for 4e-7.
4. Azimuthal integration is only needed when a mode with `l >= 1` is present. Then use
   Gauss-Legendre in `phi` on `[0, pi/2]` by symmetry, 16 nodes -- `cos^2(l phi)` is trigonometric
   and Gauss converges spectrally.
5. **Convergence criterion for the gate**: doubling the node count per panel must change every
   channel's `dP_k/dz` by less than `1e-10` relative.

### 1.8 Feature 1 validation gates

| # | Gate | Oracle | Expected | Tolerance |
|---|---|---|---|---|
| T1 | Signal-free inversion clamp | Smith & Smith sec. 2 (PINNED) | `nbar2 -> sa_p/(sa_p+se_p)` = 0.50305 for the Table-1 Yb pair at high `I_p`, `I_s = 0` | rel 1e-12 |
| T2 | Closed form vs quadrature | (F1.5)/(F1.6) vs adaptive `quad` | equal | rel 1e-11 (measured 3.6e-13) |
| T3 | Small-signal reduction | resolved solver vs existing `steady_state` | identical gain to the mean-field solver as `P_s -> 0` | rel 1e-9 on `dP/dz`; and gain within 1e-4 dB over a full solve at `s0 <= 1e-3` |
| T4 | Deep-saturation re-convergence | (F1.5) vs (F1.7) at `s0 = 1e4` | ratio -> 1 | within 1e-3 |
| T5 | `kappa(b/w) = tanh(x^2)/x^2` | (F1.8) vs numerically fitted slope ratio | table in sec. 1.4 | rel 1e-4 |
| T6 | Direction + magnitude of the effect | (F1.5) vs (F1.7), swept | `g_resolved <= g_meanfield` for ALL `s0 > 0`, `Gamma in (0,1)`, in an inverted (`n2_0 > n2_inf`) medium; worst-case ratios per the sec. 1.4 table | bracket `<= 1e-9`; worst-case ratios rel 1e-4; sec. 1.5 table rel 1e-6 |
| T7 | TSHB modal-gain crossover | sec. 1.6 table | `g_11 - g_01` changes sign between 1 W and 10 W and peaks at `+2.78 dB/m` near 100 W | sign change: exact; peak value rel 2e-3 |
| T8 | Beat length | LP mode solver | `L_beat(LP01,LP11) = 9.895 mm` for the Table-1 fiber | rel 1e-4 |
| T9 | Quadrature convergence | node doubling | see sec. 1.7 item 5 | rel 1e-10 |
| T10 | Uniform-`nt`, uniform-`I` degeneracy | resolved vs mean-field | with `b -> 0` (or a flat-top mode) the two agree bitwise-close | rel 1e-12 |
| T11 | Photon conservation | existing repo gate | unchanged: (signal+ASE photons gained)/(pump photons lost) <= 1 | as today |

Gate T6 is the one that would catch a sign error in the whole feature; T3 and T10 are the
"do not break what works" gates.

---

## 2. FEATURE 2 -- scalar gain-BPM

### 2.1 Envelope equation and sign convention

Scalar paraxial (slowly-varying-envelope) propagation of a weakly-guiding fiber field. Write the
physical field as `E_phys(x,y,z,t) = E(x,y,z) exp(i k0 n_ref z) exp(-i omega t)` (repo
convention). Then

```
2 i k0 n_ref dE/dz = (d2/dx2 + d2/dy2) E + k0^2 (n^2(x,y,z) - n_ref^2) E
                     + i k0 n_ref g(x,y,z) E                                        (F2.1)
```

Equivalently, dividing by `2 i k = 2 i k0 n_ref`:

```
dE/dz = (i/(2k)) (d2/dx2 + d2/dy2) E + (i k0^2/(2k)) (n^2 - n_ref^2) E + (g/2) E     (F2.2)
```

| Symbol | Meaning | SI unit |
|---|---|---|
| `E(x,y,z)` | slowly varying scalar field envelope, `\|E\|^2` = intensity | sqrt(W)/m (so `INT \|E\|^2 dA` = W) |
| `k0 = 2 pi/lambda` | vacuum wavenumber | m^-1 |
| `n_ref` | reference (paraxial carrier) index | 1 |
| `k = k0 n_ref` | reference wavenumber in the medium | m^-1 |
| `n(x,y,z)` | local refractive index (guide + thermal lens) | 1 |
| `g(x,y,z)` | **INTENSITY** gain coefficient | m^-1 |

**Gain convention (this is a factor-of-2 trap -- state it in the docstring).** With (F2.2),
`d\|E\|^2/dz = g \|E\|^2`, so `g` is the intensity (power) gain per metre and the per-step
amplitude factor is `exp(g dz/2)`. The sigma-form of the gain,
`g = sigma_e n2 - sigma_a (nt - n2)`, is *already* the intensity gain coefficient (it is the same
quantity as `alpha`/`g*` divided by `Gamma`), so **there is no extra factor of 2** in
`g = 2 [sigma_e n2 - sigma_a (nt - n2)]`. Writing the amplitude coefficient as `g/2` is where the
2 belongs. Verified numerically: uniform `g = 2.5 m^-1` over `L = 1.3 m` gives
`P_out/P_in = exp(gL)` to **2.4e-14**.

**Convention caveat (honest).** The canonical BPM references -- M. D. Feit & J. A. Fleck,
"Light propagation in graded-index optical fibers", *Appl. Opt.* **17**(24):3990-3998 (1978), and
K. Okamoto, *Fundamentals of Optical Waveguides*, ch. 7 -- were **not re-read in this pass**
(Optica full text is JS-gated to the fetch tool; Optics Letters items likewise). Their published
form is `dphi/dz = (i/2k)(d2/dx2 + d2/dy2) phi + (i k/2)[(n/n0)^2 - 1] phi` with `k = n0 omega/c`,
which is algebraically identical to (F2.2) once `(k/2)((n/n0)^2 - 1) = k0^2 (n^2 - n0^2)/(2k)` is
used. **Feit & Fleck use the `exp(+i omega t)` engineering convention, which flips the sign of
every `i` relative to (F2.1)/(F2.2).** Do not copy their signs verbatim into this repo; copy the
structure and keep the repo's `exp(-i omega t)`. The consistency anchor is
`soa/transverse_bpm.py`, which is already in the repo's convention:
`dA/dz = (i/2k) d2A/dx2 + ...` with diffraction applied as `A_k *= exp(-i kx^2 dz/(2k))` --
Feature 2 must match that byte-for-byte in its diffraction operator so the two BPMs agree.
**ACTION for the implementer: re-verify the Feit-Fleck/Okamoto equation numbers against the
primary text before citing them in a docstring.**

### 2.2 Symmetrized split-step and the FFT diffraction operator

Split (F2.2) into `D` (diffraction, diagonal in Fourier space) and `N` (index phase + gain,
diagonal in real space). The symmetrized (Strang) step is

```
E(z + dz) = N(dz/2) . D(dz) . N(dz/2) E(z)        (or the D/N/D ordering)            (F2.3)
```

with, in the repo's `exp(-i omega t)` convention,

```
D(dz):  E_hat(kx,ky) -> E_hat(kx,ky) * exp( -i (kx^2 + ky^2) dz / (2 k0 n_ref) )     (F2.4)

N(dz):  E(x,y) -> E(x,y) * exp( i k0^2 (n^2 - n_ref^2) dz / (2 k0 n_ref) )
                         * exp( g(x,y) dz / 2 )                                      (F2.5)
```

`kx = 2 pi fftfreq(Nx, dx)`, `ky` likewise. `D` is exactly unitary (energy conserving); all gain
and loss live in `N`, which is what makes the power-accounting contract of sec. 2.5 auditable.

Local truncation error `O(dz^3)`, global `O(dz^2)`, because the two half-`N` steps cancel the
leading commutator `[D,N]`.

**Measured this pass.** On the SMOOTH quadratic duct (mismatched Gaussian launch, half a
self-imaging period, error against a 16384-step reference) the global order is exactly 2:

| `nz` | 32 | 64 | 128 | 256 | 512 | 1024 |
|---|---|---|---|---|---|---|
| rel. err | 2.046e-3 | 5.115e-4 | 1.279e-4 | 3.197e-5 | 7.992e-6 | 1.998e-6 |
| order | - | **2.0000** | **2.0000** | **2.0000** | **2.0000** | **2.0001** |

On the *step-index* problem the same test came out ragged (0.90, 0.92, 2.03, 4.55) because the
hard circular index discontinuity on a Cartesian grid contributes an error that does not scale
with `dz`. **Order gates must use a smooth test problem (the quadratic duct of sec. 2.7), never
the step-index fiber.**

Note the nonlinear leg: with a *saturable* `g` that depends on `|E|^2`, freezing `g` at the start
of the step drops the whole scheme to first order. `soa/transverse_bpm.py` already solves this
with a midpoint predictor (`_gain`: half-step at the entry gain, full step at the midpoint gain);
**Feature 2 must reuse that pattern**, not re-invent it.

### 2.3 Choice of n_ref

`n_ref` only sets the carrier that is factored out; the *exact* solution is independent of it, but
the *paraxial approximation* is not: the neglected `d2E/dz2` term is small only when the field's
angular content is narrow about `n_ref`. Practical options for a step-index fiber:

* `n_ref = n_clad`: simplest, makes `n^2 - n_ref^2 = 0` in the cladding, but the guided mode then
  carries a residual longitudinal phase `k0 (n_eff - n_clad) z`, which is fastest-varying and
  costs accuracy.
* `n_ref = n_eff(LP01)`: the guided mode's envelope becomes (ideally) z-invariant, the slowest
  possible variation. **This is the recommended default.**
* `n_ref = n_core`: intermediate.

Measured this pass (LP01 launched into its own step-index profile, propagated 1 cm; fidelity
`|<E0|E(z)>|^2 / (<E0|E0><E|E>)`):

| fiber | `n_ref = n_clad` | `n_ref = n_eff` | `n_ref = n_core` |
|---|---|---|---|
| `a = 5 um`, `NA = 0.14`, `V = 4.26` | 0.3537 | **0.6929** | 0.4382 |
| `a = 12.5 um`, `NA = 0.054`, `V = 4.11` | 0.9907 | **0.9954** | 0.9972 |

Two lessons, both of which must be documented:

1. `n_eff` is the best or near-best choice, and the advantage is large at high index contrast.
2. **Absolute fidelity after 1 cm is NOT high when a step-index fiber mode is propagated on a
   Cartesian FFT grid**, and the cause is NOT a launch-mode mismatch. Measured: an
   imaginary-distance BPM run (propagate (F2.2) with `z -> -i z`, renormalizing each step, so the
   highest-`beta` eigenvector survives) returns a field overlapping the continuum LP01 at
   **0.999947** (`NA = 0.14`) and **0.999999** (`NA = 0.054`) -- i.e. the analytic LP01 already IS
   the discrete eigenmode to 5e-5 -- yet propagating that eigenvector gives the *same* 1-cm
   fidelity (0.692544 vs 0.692944; 0.995379 vs 0.995381). Power is conserved to 1e-9 in all cases
   and the intensity correlation stays high (0.9791 / 0.99994), so the residual is an accumulating
   **phase** error, not power loss or mode conversion.

   Diagnosis: the hard index discontinuity has a slowly-decaying transverse spectral tail; those
   high-`|k|` components are propagated with the *parabolic* dispersion `k - K^2/(2k)` instead of
   `sqrt(k^2-K^2)`, and the `N` operator regenerates them (and aliases them on the periodic grid)
   at every step. Over 2000 steps this accumulates. It scales with index contrast, exactly as
   observed. **Consequence for the gate design: do not write a shape-invariance gate with a tight
   complex-fidelity threshold on a hard-step fiber.** Use (a) the smooth quadratic duct for the
   tight accuracy gates (sec. 2.7, fidelity 1.0000000000 there), and (b) intensity correlation +
   power conservation for the step-index self-consistency gate, with the index profile
   anti-aliased (sec. 2.4). See gate B2.

### 2.4 Grid and step guidance

**Transverse sampling.** The FFT grid represents transverse wavenumbers up to the Nyquist limit
`kx_max = pi/dx`, i.e. propagation angles up to
`theta_max = kx_max/(k0 n_ref) = lambda/(2 n_ref dx)` [rad, DERIVED, trivial]. Requirements, in
increasing strictness:

* Represent the fiber's own angular content: `k0 NA <= pi/dx`, i.e. `dx <= lambda/(2 NA)`.
  (For `NA = 0.054`, `lambda = 1.032 um`: `dx <= 9.6 um` -- almost never the binding constraint.)
* Resolve the mode shape and the index step: **`dx <= a/16`** is the practical rule; the gates in
  sec. 2.9 were run at `dx = 10a/256 = a/25.6`.
* The index step is the true sampling driver: the discontinuity error scales with the index
  contrast (measured above: `NA = 0.14` is far worse than `NA = 0.054` at the same `a/dx`).

**Anti-alias the index profile (recommended).** Because the error diagnosed in sec. 2.3 comes from
the spectral tail of a hard `n(x,y)` discontinuity being regenerated and aliased by the `N`
operator every step, the standard remedy is to band-limit the sampled profile rather than to
refine `dx` indefinitely. Replace the hard mask by a one-to-two-cell error-function taper

```
n(r) = n_clad + (n_core - n_clad) * 0.5 * [ 1 - erf( (r - a) / (s dx) ) ] ,   s = 1..2   (F2.5a)
```

This preserves the guided `n_eff` to well under the discretization error while removing the
high-`|k|` content the paraxial operator mis-propagates.

Measured this pass (LP01 launched into its own profile, 2 mm, `dx = a/25.6`, `n_ref = n_eff`;
power conserved to 1e-10 in every row, so this is purely a phase-fidelity effect):

| fiber | hard step | erf `s = 1` | erf `s = 2` |
|---|---|---|---|
| `a=5 um`, `NA=0.14` | fid 0.900655 | 0.969193 | **0.996412** |
| `a=12.5 um`, `NA=0.054` | fid 0.998805 | **0.999947** | 0.999910 |

i.e. a two-cell taper buys ~2 orders of magnitude of phase fidelity at high contrast, confirming
the diagnosis in sec. 2.3. It is a *numerical* smoothing of a *physical* step, so it must be an
explicit, documented, off-by-default option with `s` reported in the solver metadata -- never a
silent default -- and gate B2 should be run both with and without it so the effect is visible.

**Window size.** The FFT grid is periodic, so the cladding tail must decay to numerical
insignificance before the window edge, and any radiated light must be absorbed (sec. 2.5) before
it wraps. The LP tail decays as `exp(-W r/a)`; require
`L_window/2 - a >= 5 a/W` **plus** the absorber margin. The gates used `L_window = 10 a`, which
satisfies this for `W >= 3`.

**Longitudinal step.** Two independent criteria; take the smaller:

1. Phase-per-step: `k0 |n^2 - n_ref^2| dz / (2 n_ref) << pi`, i.e. the `N` operator must not
   alias. In practice keep it below ~0.3 rad.
2. **Beat-length resolution**: the split-step error is a commutator error that accumulates over
   the interference between guided modes, whose scale is
   `L_beat = 2 pi/(beta_1 - beta_2) = lambda/(n_eff1 - n_eff2)`. Rule of thumb: **>= 20 steps per
   shortest beat length of interest**. For the Table-1 Yb LMA fiber `L_beat(LP01,LP11) = 9.895 mm`
   (measured), so `dz <= 0.5 mm` -- on a 4 m fiber that is 8000 steps, which sets the cost scale of
   the whole feature.

### 2.5 Absorbing boundary and the power-accounting contract

**Convention adopted (NOT pinned to a specific paper's parameters -- flagged).** Apply a
super-Gaussian apodization mask after each full step:

```
M(x,y) = 1                                                  , rho <= rho_abs
M(x,y) = exp( - [ (rho - rho_abs) / w_abs ]^m_sg )           , rho >  rho_abs        (F2.6)
```

with `rho = max(|x|,|y|)` (square window) or `rho = sqrt(x^2+y^2)` (circular), `rho_abs` the inner
edge of the absorbing layer, `w_abs` its width, and `m_sg` the super-Gaussian order. Recommended
starting values: absorber occupies the outer 10-15% of the half-window
(`rho_abs = 0.85 * L_window/2`, `w_abs = 0.10 * L_window/2`), `m_sg = 4`. Too abrupt an absorber
reflects; too gentle a one eats guided light. **Both failure modes are caught by the power
contract below.**

The literature alternative is Hadley's **transparent boundary condition** (G. R. Hadley,
"Transparent boundary condition for beam propagation", *Opt. Lett.* **17**:1426 (1992)), which
extrapolates an outgoing plane wave at the boundary. TBC is natural for finite-difference BPM
(tridiagonal) and awkward-to-impossible for a *periodic FFT* BPM, and the review literature notes
that "although transparent boundary conditions are appropriate for highly collimated beams,
properly tailored absorbers are generally better adapted to realistic field distributions"
(paraphrasing the framing of Appl. Opt. **55**:4402 (2016), "Adaptive step-size algorithm for
Fourier beam-propagation method with absorbing boundary layer of auto-determined width").
**For an FFT-based BPM, the absorbing layer is the standard choice.** Hadley's exact update
formula was not retrieved this pass -- **UNRESOLVED**, and not needed if (F2.6) is used.

**Power-accounting contract (mandatory, this is the repo's house style).** The solver must return

```
P_signal(z)   = SUM |E|^2 dA                      (power still in the field)
P_absorbed(z) = cumulative SUM (1 - M^2) |E|^2 dA  (power eaten by the boundary)
P_gain(z)     = cumulative SUM (exp(g dz) - 1) |E|^2 dA   (power added by the medium)
```

and the identity `P_signal(z) - P_signal(0) = P_gain(z) - P_absorbed(z) - P_loss(z)` must hold to
round-off. **Boundary-absorbed power is TRACKED, never silently discarded** -- exactly the
discipline `thermal.heat_load_per_m` already applies to untracked ASE. A converged window has
`P_absorbed(L)/P_signal(0) < 1e-6`; anything larger means the window is too small or the absorber
is biting the mode, and the solver should warn.

### 2.6 Gain coupling and the pump co-ODE

**Local gain from Feature 1.** At each z-plane, with the *local* BPM intensity
`I_s(x,y) = |E(x,y)|^2` [W m^-2] and the pump intensity `I_p(x,y)` (uniform over the pump cladding
for a double-clad fiber):

```
nbar2(x,y) from (F1.1) with these local intensities
g(x,y)  = nt(x,y) [ (sigma_a_s + sigma_e_s) nbar2(x,y) - sigma_a_s ]  -  l_s
        - nt(x,y) sigma_esa_s nbar2(x,y)                                             (F2.7)
```

`g` is the INTENSITY gain per metre (sec. 2.1). **There is no `Gamma` anywhere in (F2.7)**: in a
BPM the confinement factor is not an input, it is an *output*. Smith & Smith state exactly this
distinction (their sec. 1, comparing their model to Ward et al.):

> In the model of Ward et al. the increase of signal power in a mode due to laser gain is found
> from the overlap of the local gain `g(x,y,z)` with the field of that mode. In our BPM model
> laser gain increases the total signal field locally and is then apportioned among the modes
> automatically by diffraction in the presence of the core index step.

This is the design principle of Feature 2 and belongs in the module docstring.

**Pump depletion co-ODE (DERIVED; limit-checked against `waveguide.cladding_pump_overlap`).** The
pump is treated as a *cladding* channel: uniform intensity `I_p = P_p/A_clad` over the inner
cladding, no transverse structure of its own. Its power balance is the transverse integral of the
local pump absorption:

```
dP_p/dz = u_p (P_p / A_clad) INT [ sigma_e_p n2(x,y,z) - sigma_a_p (nt(x,y) - n2(x,y,z)) ] dA
                                                                                     (F2.8)
```

Note the weighting: the pump integral is **unweighted (area) integration**, while the signal is
**intensity-weighted** by its own field. That asymmetry is the whole reason TSHB exists.

*Limit check.* With `n2` uniform over a doped disc of area `A_dope`, (F2.8) becomes
`dP_p/dz = u_p P_p (A_dope/A_clad) nt [sigma_e_p nbar2 - sigma_a_p (1 - nbar2)]`, i.e. exactly the
`Gamma_p = (b_dope/clad_radius)^2` form of `waveguide.cladding_pump_overlap` and
`steady_state._plan()`. Passes.

**Heat source (PINNED).** Smith & Smith Eq. (2), transcribed:

```
Q(x,y) = N_Yb(x,y) [ (nu_p - nu_s)/nu_p ] [ sigma_a_p - (sigma_a_p + sigma_e_p) n_u(x,y) ] I_p
                                                                                     (F2.9)
```

`[W m^-3]`. `(nu_p - nu_s)/nu_p = 1 - lambda_p/lambda_s` is exactly
`thermal.quantum_defect_fraction`, and
`N_Yb [sigma_a_p - (sigma_a_p+sigma_e_p) n_u] I_p = [sigma_a_p (nt - n2) - sigma_e_p n2] I_p` is
the net absorbed pump power density. So (F2.9) is the transverse-resolved version of the repo's
existing quantum-defect heat term. `INT Q dA` must equal `-dP_p/dz` times the quantum defect,
which is a free consistency gate (C3 below).

*Caveat carried from the review:* in Yb the quantum defect converts only 5-10% of the absorbed
pump energy to heat, whereas photodarkening converts ~100% of what it absorbs, so a fiber with
even ~1 dB/m of PD loss can have MORE heat from PD than from the quantum defect (Jauregui et al.
2020, sec. 4, reproducing Opt. Express 23:15265 (2015)). The repo's
`ConcentrationModel.photodarkening_loss_per_m` already models the loss; if it is on, its
dissipated power must be added to (F2.9) or the thermal lens will be badly under-predicted.

### 2.7 Thermal-lens coupling and the parabolic-duct oracle

**Quasi-static loop** (the honest name for what this is):

```
  E(x,y,z)  --(F2.7)-->  n2(x,y,z)  --(F2.9)-->  Q(x,y,z)  [W m^-3]
     ^                                                  |
     |                                                  v
  n(x,y,z) = n_guide + dn_th(r,z)  <---  dn_th = (dn/dT) * dT(r,z)  <---  heat equation
```

The radial temperature rise is already implemented: `thermal.radial_temperature_rise` (the
Brown & Hoffman solution, IEEE JQE 37:207 (2001)) gives, for heat `Q_per_m` [W/m] generated
uniformly in a core of radius `a` inside an outer radius `b_outer` with convection `h`:

```
dT(r <= a) = dT_center - (Q/(4 pi k_core)) (r/a)^2
dT_center  = Q/(4 pi k_core) + Q/(2 pi k_clad) ln(b_outer/a) + Q/(2 pi b_outer h)
```

So **the in-core temperature rise is exactly parabolic** and the induced index change is

```
dn_th(r) = (dn/dT) [dT_center - delta_n0_over_dndt (r/a)^2] ,  delta_n0 = (dn/dT) Q/(4 pi k_core)
n(r) = n_c - (1/2) b_curv r^2 ,   b_curv = 2 delta_n0 / a^2                          (F2.10)
```

which is what `thermal.thermal_lens_focal_power_per_m` already encodes as
`D' = b_curv/n_c = (dn/dT) Q / (2 pi n_c k a^2)` [m^-2].

**The oracle (DERIVED, with a textbook-limit cross-check).** Put the parabolic index into (F2.2)
with `n_ref = n0`. Using `n^2 - n0^2 ~ -n0 b_curv r^2 = -n0^2 alpha^2 r^2`:

```
alpha^2 = b_curv / n0 = D'          ==>   alpha = sqrt(D')      [m^-1]               (F2.11)
```

(F2.2) then becomes, term for term, the 2-D quantum harmonic oscillator with `z <-> t`,
"mass" `m <-> k = k0 n0`, and angular frequency `omega <-> alpha`:

```
i dE/dz = -(1/(2k)) (d2/dx2+d2/dy2) E + (k alpha^2/2) r^2 E
```

Reading off the standard oscillator results:

```
matched (invariant) 1/e FIELD radius:   w_m = sqrt( 2 / (k0 n0 alpha) )              (F2.12)
field self-imaging period:              z_p = 2 pi / alpha                           (F2.13)
spot-size (breathing) period:           z_p/2 = pi / alpha                           (F2.14)
first spot-size extremum of a
mismatched launch:                      z = z_p/4 = pi/(2 alpha)                     (F2.15)
```

*Cross-check against the published GRIN ABCD matrix.* For `n(r) = n0 (1 - (1/2) alpha^2 r^2)` the
standard ray matrix over length `L` (internal angles) is

```
[ cos(alpha L)            sin(alpha L)/alpha ]
[ -alpha sin(alpha L)     cos(alpha L)       ]                                       (F2.16)
```

whose focal power is `1/f = alpha sin(alpha L) -> alpha^2 L` for `alpha L << 1`, i.e.
`1/(f L) -> alpha^2`. That is identically `thermal.thermal_lens_focal_power_per_m`'s `D'`.
So (F2.11) is not a new assumption -- it is the statement that the repo's existing thermal-lens
dioptric power per unit length IS the square of the duct parameter. Limit check passes, and it
means **the BPM and `thermal.py` are gated against each other for free.**

**Measured this pass** (Cartesian FFT BPM, 192^2, `lambda = 1.032 um`, `n0 = 1.45`,
`D' = 1e5 m^-2` so `alpha = 316.228 m^-1`, `w_m = 26.766 um`, `z_p = 19.869 mm`):

| test | result |
|---|---|
| matched Gaussian, one full period, 600 steps | width span `[26.76584, 26.76621] um`, **rel. deviation 1.37e-5**; round-trip fidelity **1.0000000000** |
| mismatched launch `w0 = 1.5 w_m` | first spot-size minimum at `z = 4.967 mm` vs predicted `z_p/4 = 4.967 mm` (**rel. 0.0e0** at the step resolution); width breathes `[17.844, 40.149] um`; fidelity at `z_p` = 1.00000000 |
| mismatched launch `w0 = 0.7 w_m` | first maximum at `z = 4.967 mm` (same); width `[18.736, 38.237] um`; fidelity at `z_p` = 1.00000000 |
| `D' = 1e4 m^-2` | `alpha = 100 m^-1`, `w_m = 47.597 um`, `z_p = 62.83 mm`, breathing 31.42 mm |

This is a *very* tight oracle -- it pins the diffraction operator's factor, the index-phase
operator's factor, and the sign of both simultaneously. A missing `n_ref` in (F2.4), or a factor
2 in (F2.5), moves `z_p` by an easily-detected amount.

### 2.8 Scope boundary: why a steady-state BPM does NOT capture TMI

(PINNED from C. Jauregui, J. Limpert, A. Tunnermann, "Transverse mode instability",
*Adv. Opt. Photon.* **12**(2):429 (2020), arXiv:2004.14752, sec. 3.)

The accepted picture is a four-step chain: two transverse modes beating along the fiber create a
**modal interference pattern (MIP)** with period `L_beat`; because the inversion is more strongly
depleted where the signal is intense, the MIP is imprinted on the inversion profile; the
resulting transversally inhomogeneous, quasi-periodic power extraction gives a quasi-periodic
temperature profile (quantum-defect heat); and the thermo-optic effect turns that into a
**refractive index grating (RIG)** -- a thermally induced long-period grating that automatically
has the right period and symmetry to couple the two modes.

The decisive point for scope:

> a second condition needs to be fulfilled to actually enable energy transfer between the
> transverse modes: **a phase shift between the MIP and the RIG**. This condition implies that the
> intensity maxima/minima of the MIP must be shifted longitudinally with respect to the refractive
> index maxima/minima of the RIG. This, in turn, leads to a movement of the RIG ... Another
> consequence of the movement of the RIG is that a frequency difference between the modes involved
> in the energy exchange appears due to the Doppler effect.

and

> the TMI threshold is determined by the combined strength of the RIG and the phase shift.

A **steady-state** model computes the RIG *exactly in phase* with the MIP (the temperature is the
instantaneous steady solution of the heat equation driven by the instantaneous irradiance
pattern), and an in-phase index grating transfers **zero net energy** between the modes. The
missing ingredient is a *time* axis: the phase shift comes either from an assumed inter-mode
frequency offset (the "steady-periodic" two-frequency models, which give semi-analytic thresholds)
or from the finite thermal response time of the core responding to a fluctuating MIP (the
transient 3-D-BPM + thermal-diffusion + rate-equation models). Consequences the module must
document:

* Feature 2, as specified, is **quasi-static**: it computes the correct *average* thermal lens and
  the correct TSHB modal gain, and can show mode-content evolution driven by *gain* competition.
  It **cannot** predict a TMI threshold, TMI onset frequency, or the chaotic beam fluctuations.
* The relevant timescales are experimentally in the **100 Hz to few-kHz** range for the beam
  fluctuations (larger core -> slower, consistent with core thermal diffusion) with beam-
  fluctuation build-up times of **some ms** -- all of which a z-only model has no representation
  for.
* What Feature 2 *can* legitimately deliver toward TMI is the **input** that every TMI model needs:
  the saturated, radially-resolved inversion and the resulting heat profile. Smith & Smith's whole
  point is that population saturation *reduces* the STRS coupling gain relative to the laser gain
  and thereby raises the threshold -- an effect that is invisible without Feature 1.

Reference threshold numbers for calibration context (Smith & Smith 2013, Tables 2-5; 50 um core,
`NA = 0.054`, `lambda_s = 1032 nm`, `lambda_p = 976 nm`, 1100 Hz LP01-LP11 detuning, LP11 seeded
at 1e-16 W, threshold defined as 1% of signal power in LP11):

| `d_clad` | `L` | co-pumped `P_thres` | counter-pumped `P_thres` |
|---|---|---|---|
| 100 um | 0.8 m | 488 W | 453 W |
| 200 um | 1.6 m | 685 W | 676 W |
| 300 um | 2.6 m | 885 W | 921 W |
| 400 um | 4.0 m | 1101 W | 1220 W |
| 500 um | 6.0 m | 1335 W | 1580 W |

with the no-population-saturation reference threshold at **345 W**, and confined doping
(`d_dope = 30 um` in a 50 um core, co-pumped, `d_clad = 100 um`) raising it to **786 W**. These
are NOT gates for Feature 2 (it cannot compute them); they are the calibration targets for any
future TMI module and the evidence that saturation matters at the factor-of-2-plus level.

### 2.9 Feature 2 validation gates

| # | Gate | Oracle | Expected | Tolerance |
|---|---|---|---|---|
| B1 | Free-space Gaussian diffraction | analytic `w(z) = w0 sqrt(1+(z/zR)^2)`, `zR = pi n0 w0^2/lambda` | match | rel 1e-6 on `w(z)` over 3 `zR` |
| B2 | Guided-mode self-consistency (step-index) | analytic LP01 (= the discrete eigenmode to 5e-5, measured) | power ratio 1; intensity correlation `>= 0.9999` and fidelity `>= 0.999` after 2 mm at `dx = a/25.6` with the `s = 2` erf taper of (F2.5a) | power rel 1e-9; others as stated (measured 0.996412 / 0.999910 fidelity). **Do NOT gate complex fidelity tightly on a HARD step** -- measured 0.9007 (`NA=0.14`) and 0.9988 (`NA=0.054`) at 2 mm, falling to 0.693 / 0.995 at 1 cm (sec. 2.3) |
| B3 | **Quadratic-duct matched Gaussian** | (F2.12) | `w(z)` constant over one `z_p` | rel. width deviation `< 1e-4` (measured 1.4e-5) |
| B4 | **Quadratic-duct self-imaging** | (F2.13) | round-trip fidelity at `z = z_p` | `>= 1 - 1e-8` (measured 1.0000000000) |
| B5 | **Quadratic-duct breathing period** | (F2.15) | first spot-size extremum of a `1.5 w_m` launch at `z_p/4` | rel 1e-3 (measured exact to the step resolution) |
| B6 | Thermal-lens bridge | `alpha^2 == thermal.thermal_lens_focal_power_per_m(Q,a)` | identical to round-off; `z_p = 2 pi/sqrt(D')` | rel 1e-12 |
| B7 | Split-step order | `dz` refinement on the **smooth** duct problem (never the step-index fiber) | global order 2.0 (measured 2.0000 over 5 refinements) | 2.00 +/- 0.05 |
| B8 | Gain convention | uniform `g`, no index structure | `P_out/P_in = exp(g L)` | rel 1e-12 (measured 2.4e-14) |
| B9 | Unitarity of `D` | `g = 0`, `n = const` | power conserved | rel 1e-13 per step |
| B10 | Power-accounting identity | sec. 2.5 | `dP_signal = P_gain - P_absorbed - P_loss` | rel 1e-12 |
| B11 | Window adequacy | sec. 2.5 | `P_absorbed(L)/P_signal(0) < 1e-6` for a guided launch | as stated; warn above |
| B12 | Reduction to the 1-D solver | mean-field `steady_state` | with the dopant profile flat and the mode weakly saturating, BPM `dP_s/dz` matches `Gamma nt [...]` | rel 1e-4 (discretization-limited) |
| B13 | Pump co-ODE limit | (F2.8) with uniform `n2` | equals `Gamma_p` form of `waveguide.cladding_pump_overlap` | rel 1e-12 |
| B14 | Heat-source consistency | (F2.9) | `INT Q dA` = quantum defect x local pump absorption from (F2.8) | rel 1e-12 |
| B15 | TSHB emerges | Feature 1 sec. 1.6 | in a two-mode launch, LP11's fractional power **grows** with saturation depth | monotone trend; sign only (a BPM apportions by diffraction, so no closed-form target) |
| B16 | Scope honesty | n/a | module docstring and `__doc__` state the TMI boundary of sec. 2.8 | doc test |

---

## 3. Formula cheat sheet (transcribe directly)

```
CONSTANTS / SYMBOLS
  h      Planck [J s]        nu_k = c/lambda_k [Hz]      k0 = 2 pi/lambda [1/m]
  nt(r)  ion density [m^-3]  tau [s]   sigma_a, sigma_e, sigma_esa [m^2]
  i_k(r,phi) normalized modal intensity, INT i_k dA = 1  [m^-2]
  I_k = P_k i_k [W/m^2]      u_k = +1 fwd / -1 bwd       m = 2 (ASE modes)

-------- FEATURE 1 --------

(F1.1) LOCAL BALANCE  [PINNED: Smith & Smith, Opt. Express 21:15168 (2013) Eq. 1]
  nbar2(r,phi,z) = SUM_k sigma_a_k I_k /(h nu_k)
                 / [ 1/tau + SUM_k (sigma_a_k+sigma_e_k) I_k /(h nu_k) ]
  n2 = nt * nbar2 ;  980-nm Er pump: set sigma_e(pump)=0 ; 1480 Er / all Yb: keep both.
  Signal-free clamp:  nbar2 -> sigma_a_p/(sigma_a_p+sigma_e_p)   (= 0.50305, Yb 976 nm)

(F1.2) PROPAGATION  [DERIVED; reduces to Giles]
  dP_k/dz = u_k P_k INT [ sigma_e_k n2 - sigma_a_k (nt - n2) - sigma_esa_k n2 ] i_k dA
          - u_k l_k P_k
          + u_k m h nu_k dnu_k sigma_e_k INT n2 i_k dA            (ASE channels only)

  Reduction (n2 r-independent):  Gamma_k = INT_dopant i_k dA ,
      alpha_k = sigma_a_k Gamma_k nt , g*_k = sigma_e_k Gamma_k nt
      dP_k/dz = u_k[(alpha_k+g*_k) nbar2 - alpha_k - l_k] P_k + u_k g*_k nbar2 m h nu_k dnu_k
  Mean-field closure the current solver uses: I_k -> <I_k> = Gamma_k P_k / A_dope.

(F1.3)-(F1.4) MOEBIUS FORM (uniform pump + one signal)
  nbar2(I_s) = n2_inf + (n2_0 - n2_inf)/(1 + I_s/I_sat)
  n2_0   = [I_p sa_p/(h nu_p)] / [I_p (sa_p+se_p)/(h nu_p) + 1/tau]
  n2_inf = sa_s/(sa_s+se_s)
  I_sat  = (h nu_s/(sa_s+se_s)) [ I_p (sa_p+se_p)/(h nu_p) + 1/tau ]      [W/m^2]

(F1.5)-(F1.6) EXACT RADIAL INTEGRAL, Gaussian mode + top-hat dopant  [DERIVED]
  J = INT_0^b nbar2(r) i(r) 2 pi r dr = n2_inf*Gamma + (n2_0-n2_inf)*Phi(s0,Gamma)
  Phi(s0,Gamma) = (1/s0) ln[ (1+s0) / (1 + s0 (1-Gamma)) ]
  s0 = 2 P_s/(pi w^2 I_sat)   (on-axis)     Gamma = 1 - exp(-2 b^2/w^2)
  mean field:  J_MF = Gamma [ n2_inf + (n2_0-n2_inf)/(1+s_area) ],
               s_area = s0 Gamma/(-ln(1-Gamma)) = P_s Gamma/(pi b^2 I_sat)
  modal gain:  g = nt [ (sa_s+se_s) J - sa_s Gamma ]              [1/m]

(F1.8) SATURATION-POWER CORRECTION  [DERIVED]
  kappa(x) = tanh(x^2)/x^2 ,  x = b/w  ==>  P_sat_resolved = kappa * P_sat_meanfield
  kappa(1) = 0.7616 ;  kappa -> 1 as x->0 ;  kappa -> 1/x^2 as x->inf

(F1.9) MULTI-MODE
  I_s = SUM_m P_m i_m ;  one shared nbar2(r,phi,z) ;
  dP_m/dz = u_m P_m nt INT [ (sa+se) nbar2 - sa ] i_m dA - u_m l P_m
  LP_lm (l>=1): i_m ~ psi^2 cos^2(l phi) / (pi INT psi^2 r dr)

QUADRATURE: composite Gauss-Legendre, panel breakpoints at r=min(a,b) and r=max(a,b),
  24 nodes/panel (16 already machine-precision); NEVER a step mask on a uniform grid (O(h)).

-------- FEATURE 2 --------

(F2.2) ENVELOPE (repo exp(-i omega t))
  dE/dz = (i/(2k)) (d2/dx2 + d2/dy2) E + (i k0^2/(2k))(n^2 - n_ref^2) E + (g/2) E
  k = k0 n_ref ;  g = INTENSITY gain [1/m]  (amplitude factor exp(g dz/2); NO extra factor 2)

(F2.3)-(F2.5) SYMMETRIZED SPLIT-STEP  (global O(dz^2))
  E <- N(dz/2) D(dz) N(dz/2) E
  D: E_hat *= exp( -i (kx^2+ky^2) dz / (2 k0 n_ref) )
  N: E    *= exp( i k0^2 (n^2 - n_ref^2) dz / (2 k0 n_ref) ) * exp( g dz/2 )
  Saturable g: midpoint predictor (as soa/transverse_bpm._gain), else the scheme is O(dz).

  n_ref = n_eff(LP01) recommended.
  dx <= a/16 ;  L_window/2 - a >= 5 a/W + absorber ;  dz <= L_beat/20 ,
  L_beat = 2 pi/(beta_1-beta_2) = lambda/(n_eff1 - n_eff2)
  theta_max = lambda/(2 n_ref dx)

(F2.5a) ANTI-ALIASED INDEX PROFILE (opt-in, report s in metadata)
  n(r) = n_clad + (n_core-n_clad)*0.5*[1 - erf((r-a)/(s dx))] ,  s = 1..2
  A hard step's spectral tail is mis-propagated by the parabolic dispersion and aliased by the
  periodic grid; s=2 buys ~2 orders of magnitude of phase fidelity at NA=0.14.

(F2.6) ABSORBER  M = exp(-[(rho-rho_abs)/w_abs]^m_sg) outside rho_abs ; m_sg=4,
  rho_abs = 0.85 (L/2), w_abs = 0.10 (L/2).
  CONTRACT: P_absorbed tracked; dP_signal = P_gain - P_absorbed - P_loss to round-off;
  converged window has P_absorbed(L)/P_signal(0) < 1e-6.

(F2.7) LOCAL GAIN     g(x,y) = nt[(sa_s+se_s) nbar2(x,y) - sa_s] - l_s - nt sigma_esa_s nbar2
  (NO Gamma -- the BPM produces the confinement factor, it does not consume one.)

(F2.8) PUMP CO-ODE (cladding channel, I_p = P_p/A_clad uniform)
  dP_p/dz = u_p (P_p/A_clad) INT [ sigma_e_p n2 - sigma_a_p (nt - n2) ] dA
  (AREA-weighted, unlike the intensity-weighted signal -- this asymmetry IS the TSHB.)

(F2.9) HEAT  [PINNED: Smith & Smith Eq. 2]
  Q(x,y) = nt(x,y) [(nu_p - nu_s)/nu_p] [ sigma_a_p - (sigma_a_p+sigma_e_p) nbar2(x,y) ] I_p
         [W/m^3]

(F2.10)-(F2.15) THERMAL LENS -> QUADRATIC DUCT ORACLE  [DERIVED; matches thermal.py]
  delta_n0 = (dn/dT) Q_per_m/(4 pi k_core) ;  n(r) = n_c - (1/2) b_curv r^2 ,
  b_curv = 2 delta_n0/a^2 ;  D' = b_curv/n_c = (dn/dT) Q/(2 pi n_c k a^2)   [1/m^2]
  alpha = sqrt(D')                                   [1/m]
  matched 1/e field radius   w_m = sqrt( 2/(k0 n0 alpha) )
  field period               z_p = 2 pi/alpha
  breathing period           z_p/2 = pi/alpha ;  first extremum at z_p/4
  GRIN ABCD over L:  [[cos(aL), sin(aL)/a], [-a sin(aL), cos(aL)]] , a = alpha
                     1/f = alpha sin(alpha L) -> alpha^2 L for alpha L << 1

TMI SCOPE: a steady-state BPM puts the index grating IN PHASE with the irradiance grating,
  which transfers ZERO net power between modes. TMI needs a MIP/RIG phase shift, which requires
  a time axis (inter-mode frequency offset / finite thermal response). Document, do not fake.
```

---

## 4. References actually consulted this pass

1. **A. V. Smith and J. J. Smith**, "Increased mode instability thresholds of fiber amplifiers by
   gain saturation," *Opt. Express* **21**(13):15168 (2013); arXiv:1304.1064. **Full text read.**
   Source of (F1.1) [their Eq. 1], (F2.9) [their Eq. 2], Table 1 parameters, Tables 2-5 thresholds,
   the `n_u -> sa_p/(sa_p+se_p) ~ 0.5` anchor, and the gain-BPM design statement.
2. **C. Jauregui, J. Limpert, A. Tunnermann**, "Transverse mode instability," *Adv. Opt. Photon.*
   **12**(2):429 (2020); arXiv:2004.14752. **Sections 2-4 read.** Source of the TMI mechanism, the
   MIP/RIG phase-shift requirement, the timescales, and the photodarkening-heat caveat.
3. **`docs/fiber_amp_model_spec.md`** (this repo), sec. 0-1, 7 -- the prior 3-0-verified extraction
   of **C. R. Giles & E. Desurvire**, "Modeling Erbium-Doped Fiber Amplifiers," *J. Lightwave
   Technol.* **9**(2):271 (1991) and **E. Desurvire**, *Erbium-Doped Fiber Amplifiers* (Wiley,
   1994). Used as the pinned target of the reduction limit in sec. 1.2.
4. **D. C. Brown & H. J. Hoffman**, IEEE JQE **37**:207 (2001) -- via the existing
   `thermal.radial_temperature_rise` implementation and its docstring citation. Not re-read.
5. Cited but **not read this pass** (flagged in-line): Feit & Fleck, *Appl. Opt.* **17**:3990
   (1978); Okamoto, *Fundamentals of Optical Waveguides* ch. 7; Hadley, *Opt. Lett.* **17**:1426
   (1992); *Appl. Opt.* **55**:4402 (2016); Jiang & Marciante, *JOSA B* **25**:247 (2008); Gong
   et al., *Opt. Express* **15**:3236 (2007); Marcuse, *BSTJ* **56**:703 (1977).

## 5. Open items for the implementer

1. **Jiang & Marciante 2008** equations remain unretrieved (paywalled; the LLE Review reprint URL
   is dead). Nothing here depends on them, but the module docstring should cite them as the
   originating TSHB-in-LMA reference and the sec. 1.6 benchmark should be re-checked against their
   published `M^2`-vs-saturation curve when access is available.
2. **Feit-Fleck / Okamoto equation numbers and sign conventions** need a primary-source pass before
   they go into a docstring (sec. 2.1 caveat). The physics is settled by the internal consistency
   anchor (`soa/transverse_bpm.py`) and by gate B3-B5; only the citation precision is open.
3. **Super-Gaussian absorber parameters** (F2.6) are a stated convention, not a pinned published
   set. Gate B11 makes the choice self-validating, so this is low risk.
4. **Hadley TBC update formula** not retrieved; not needed for an FFT BPM.
5. The Moebius closed forms (F1.5)-(F1.8) do **not** cover upconversion / PIQ (quadratic
   `_nbar2_c` branch) or a core-guided (non-uniform) pump. Those need quadrature; the closed forms
   stay as gates on the ideal path.
