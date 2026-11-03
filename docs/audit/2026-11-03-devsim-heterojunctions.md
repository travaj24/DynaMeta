# DEVSIM semiconductor/semiconductor heterojunctions (v0.14.0, workstream D1a)

DEVSIM phase, release v0.14.0. Follows `docs/audit/2026-11-02-s1-9-heterojunction-built-in-potential.md`
(the v0.13.0 3-D equilibrium alignment and its design for the drift-diffusion phase). Code:
`dynameta/carriers/physics_heterojunction.py` (the one kernel), wired into
`LayeredDevsimBuilder` (2-D) and `Devsim3DEquilibrium` (3-D). Gates: `tests/test_devsim_heterojunction.py`;
device factories, oracle helpers and the heavy 3-D gate: `validation/devsim_heterojunction.py`.

## 1. What was wrong / missing

* Both builders REFUSED every interface between two semiconductor regions that solve carrier
  continuity (2-D `NotImplementedError`, audit regrade 7.1; 3-D `ValueError` for DD/bipolar
  `extra_semiconductors`): the interface wired Potential only, so the carriers would have seen a
  zero-flux wall (a silently BLOCKING junction).
* The 2-D builder had the S1-9 zero-built-in-potential defect for adjacent EQUILIBRIUM semiconductors:
  each region's `Phi_c0` made that region neutral at `psi = 0`, i.e. every region carried the same
  work function, and every ohmic contact pinned `psi = bias` in its own region's frame.

## 2. Formulation (from the literature)

**Band alignment -- Anderson's electron-affinity rule** (R. L. Anderson, Solid-State Electron. 5, 341
(1962)). `psi` is continuous (it is the vacuum level up to one constant) and each band edge hangs off
the vacuum level by the affinity:

    E_c(r) = q (Phi_c(r) - psi),   Phi_c(r) = Phi_c,own(ref) + (chi_ref - chi_r)/q
    E_v(r) = E_c(r) - E_g,eff(r)
    s(r)   = Phi_c(r) - Phi_c,own(r)          (frame shift = the neutral-bulk potential of r)

`Phi_c,own(r)` is the conduction edge each physics module already implies in its legacy frame:

| physics | legacy frame | `Phi_c,own` |
|---|---|---|
| equilibrium | `n = N_c F_1/2((psi - Phi_c0)/V_t)`, E_F = 0 | `Phi_c0 = -eta_bg V_t` (Aymerich-Humet) |
| unipolar DD | contact pins `n = N_D` at `psi = bias` | `-V_t [ln(N_D/N_c) + Delta(N_D/N_c)]` |
| bipolar DD | `psi` = intrinsic level, contact pins `V_t[ln(n0/n_i) + Delta]` | `+V_t ln(N_c/n_i)`, `E_g,eff = kT ln(N_c N_v/n_i^2)` |

`Delta(u) = int_0^u (g(x)-1) dx/x` is the Fermi-Dirac correction of the FD-enhanced Scharfetter-Gummel
bulk (audit C-10; 0 in the Boltzmann limit). Every ohmic contact on region r pins
`Potential = bias + s(r)` (+ the bipolar built-in offset); a gate on a dielectric keeps the reference
frame (the reference is the region of the first GROUND electrode on a semiconductor, else the first
semiconductor region -- in 3-D the body `semi`), i.e. flat band of the reference at `Vg = 0`, as in
v0.13.0. Potential and D are continuous: DEVSIM `continuous` interface equation on `PotentialEquation`
(the two region rows are summed -- no interface charge -- and `Potential@r0 = Potential@r1` replaces one).

**Thermionic emission at the interface** -- an interface boundary condition in series with bulk
drift-diffusion (K. Horio & H. Yanai, IEEE TED 37, 1093 (1990); K. Yang, J. R. East & G. I. Haddad,
Solid-State Electron. 36, 321 (1993) -- their thermionic-field-emission condition with zero tunnelling;
D. Schroeder, *Modelling of Interface Carrier Transport for Device Simulation*, Springer 1994). Net
electron particle flux from side 0 to side 1 over the barrier top `E_top = max(E_c,0, E_c,1)`:

    F_n(0->1) = (A* T^2 / q) [ exp((E_Fn,0 - E_top)/kT) - exp((E_Fn,1 - E_top)/kT) ]
              = (A* T^2 / q) exp(-(E_top - E_Fn,1)/kT) [ exp((E_Fn,0 - E_Fn,1)/kT) - 1 ]

(the Horio-Yanai quasi-Fermi-level form). With Boltzmann statistics this is the textbook
`J = q v_n [n_lo exp(-dE_c/kT) N_c,hi/N_c,lo - n_hi]`, `v_n = A* T^2/(q N_c,hi) = sqrt(kT/(2 pi m*_hi))`,
i.e. the Sentaurus / Quay (TU Wien 2001, eq. 3.88) form `J = q[v_2 n_2 - (m_2/m_1) v_1 n_1 exp(-dE/kT)]`.
`A* = 4 pi q m* k^2/h^3` is the effective Richardson constant of the BARRIER side (the region containing
the barrier maximum), from that side's DOS mass or `TransportModel.richardson_n_Am2K2 / _p_`. Holes
mirror it on the valence edge (`E_bot = min(E_v,0, E_v,1)`). Zero bias => equal quasi-Fermi levels =>
zero flux for ANY offset (detailed balance by construction).

**The quasi-Fermi level from the solution variable.** The FD-SG bulk's zero-current limit is
`d psi = g V_t d ln n`, so the quasi-Fermi level the bulk implies is
`(E_Fn - E_c)/kT = L(n) = ln u + Delta(u)`, `dL/dn = g(u)/n`. Using exactly this `L` at the interface
is what keeps the interface consistent with the bulk; any other activity (exact `F_1/2^-1`, or `ln u` on
a degenerate side) is a spurious interface battery. `Delta` has a closed form for the rational g fit:

    Delta(u) = K1 s^2 + K2 s + KLOG ln(1 + b s + d s^2) + KATAN atan(D s/(2 + b s)),  s = u^(1/3),
    D = sqrt(4d - b^2),  K1 = 3c/2d,  K2 = (3a - 3cb/d)/d,  alpha = -3c/d - K2 b,  beta = -K2,
    KLOG = alpha/2d,  KATAN = (beta - alpha b/2d) 2/D           (a, b, c, d = carriers.einstein GA..GD)

DEVSIM has no `atan()`, so it is four half-angle reductions `z -> z/(1 + sqrt(1 + z^2))` plus a 7-term
Taylor tail (|error| < 2e-15 for all z >= 0). MEASURED vs the Gauss-Legendre quadrature
`physics_bipolar_dd.fd_potential_shift` for u in [1e-8, 1e5]: max |difference| = 2.7e-12.

**Continuous quasi-Fermi level** (the drift-diffusion limit; Horio-Yanai's A* -> infinity):
`L_0 - k_0 = L_1 - k_1` (`k_i = (E_top - E_c,i)/kT`) as a DEVSIM `continuous` interface equation on the
continuity equation (rows summed = current continuity; the relation replaces one row).

**DEVSIM mechanism.** `interface_model`s on the node copies `@r0/@r1` with explicit derivative models
(`HJ_L_<carrier>:<carrier>@r0`), recorded through `eq_registry.record_interface_equation` under the
continuity-equation names (`fluxterm` for TE, `continuous` for the QFL model), so Gummel/staged
freeze-and-reapply sees them. Fluxterm sign (conventional current from r0 into r1): confirmed by the 1-D
prototype (zero-offset TE gives the analytic series interface resistance; A* x 1e6 reproduces the
continuous model to 1e-6) and pinned on the builder path by gates C/E.

**Model selection.** `LayeredDevsimBuilder(..., heterointerface="auto"|"thermionic"|"continuous")`
and `Stacked3DSpec.heterointerface`. `auto` = thermionic where the interface has a band offset
(|dE_c| or |dE_v| > 1 nV), continuous where it has none (a same-crystal split such as a p/n homojunction
cut into two layers -- TE there would add a spurious ballistic interface resistance).

## 3. What is implemented, what is refused

| case | status |
|---|---|
| 2-D z-stacked semiconductor heterostack, equilibrium | aligned frame + contact re-reference (S1-9 2-D fix) |
| 2-D z-stacked, unipolar DD / DD | TE or continuous electrons |
| 2-D z-stacked, bipolar / bipolar | TE or continuous electrons AND holes |
| 2-D unipolar DD next to bipolar | REFUSED (`NotImplementedError`: no hole BC on the unipolar side) |
| 2-D bipolar heterojunction with `band_gap_eV` != `kT ln(N_c N_v/n_i^2)` (> 5 meV) | REFUSED (`ValueError`, gives the consistent n_i) |
| 2-D different materials both carrying the TransportModel defaults chi 4.5 / Eg 3.6 | `AdvisoryWarning` (offset zero by default, not by data) |
| 2-D lateral (sidewall) interfaces | still REFUSED: the planner would have to split the background around each inclusion and wire vertical interfaces (Potential and carrier); message now names that |
| 3-D `extra_semiconductors`, equilibrium | unchanged (v0.13.0), `band_alignment` now calls the shared `anderson_alignment` (bit-identical) |
| 3-D `extra_semiconductors`, drift_diffusion | NEW: each extra region solves electron continuity (with the primary's mobility), interfaces TE/continuous |
| 3-D `extra_semiconductors`, bipolar_dd | still REFUSED: the tuple carries no hole statistics (n_i, N_v, hole mobility, gap, doping sign) |
| degenerate emitting side, TE magnitude | Maxwell-Boltzmann Richardson tail (FD emission ~ ln(1+e^x) not used); equilibrium exact |
| tunnelling / thermionic-field emission, interface traps/dipoles | not here (D1d, D1c) |

`geometry/guards.py::require_supported_geometry` needed no change: the semiconductor/semiconductor refusals
were inline in the two builders (removed / narrowed as above); the guard covers Stack.features and tapers.

Other changes: `PE.setup_contact`, `DD.setup_contact_ohmic_dd`, `BP.setup_contact_ohmic_bipolar` take
`psi_offset_V` (0 -> the contact string is character-for-character unchanged);
`TransportModel.richardson_n_Am2K2 / richardson_p_Am2K2` (optional); the 2-D bipolar "gated" warning now
judges an electrode by its contact REGION (a metal-layer electrode landing on a bipolar face is an ohmic
contact, not a gate); the 2-D unipolar "no full-edge ground" advisory is skipped for a face contact.

## 4. Oracles and measured results (2026-11-03, DEVSIM 2.11.2, box A)

Devices: laterally uniform vertical stacks in the 2-D builder (20 nm cell, 3 mesh columns = the 1-D
limit), contacted through metal layers on both faces. GaAs / Al0.3Ga0.7As at 300 K, `dE_c = 0.2431 eV`,
`dE_v = 0.1309 eV`; n_i from each gap. All oracles are independent of the code under test (band-diagram
algebra, Fermi-Dirac by direct quadrature, depletion approximation, Richardson law, Ohm's law integral).

| gate | oracle | measured | gated at |
|---|---|---|---|
| closed-form Delta (node model) | Gauss-Legendre quadrature `fd_potential_shift`, u = 1e-8..1e4 | max abs 2.7e-12; derivative model vs centred FD < 1e-6 | 1e-10 / 1e-6 |
| A homojunction limit (GaAs p/n, 1e17 cm^-3 each; one region with an in-region step vs two regions, `auto` -> continuous) | the existing single-region solve, same 801-node column | potential 1.1e-14 V; n, p 4.4e-13 rel; J(0.7 V) 1.9e-16 rel -- round-off (not bit-identical: the split node's assembly order differs) | 1e-9 V / 1e-8 |
| B built-in potential, N-Al0.3Ga0.7As / p-GaAs 1e17 / 1e17 | `q V_bi = W_p - W_n` (FD quadrature) = 1.502232 V | 1.502262 V (+3.1e-5 V: g fit vs exact FD) | 2e-4 V |
| B depletion widths (charge-equivalent) | Anderson depletion approx. with 2kT/q tail: 99.836 nm each side | 99.815 nm (-2.1e-4) | 1 % |
| B potential split V_n/V_p | eps_p N_A/(eps_n N_D) = 1.0750 (1.0759 with per-side kT/q tails removed) | 1.0732 (-0.17 %; -0.08 % tail-corrected) | 0.5 % / 0.2 % |
| B conduction spike / valence step from SOLVED densities | dE_c = 0.2431, dE_v = 0.1309 eV | errors 6.8e-8 / 3e-16 eV | 1e-6 eV |
| B flat E_Fn, E_Fp at zero bias (carriers > 1 m^-3) | 0 | max 1.0e-4 eV | 2e-4 eV |
| B forward current, short-base electron injection `q D_n (n_i^2/N_A) e^{qV/kT}/W_b` | 0.2848 / 13.43 A/m^2 at 0.7 / 0.8 V | 0.2830 / 13.22 (ratio 0.994 / 0.984); terminal balance 6e-4 / 2.5e-5 | 15 % |
| D neutrality + discrete Gauss law at the interface | total charge 0; D(right) - D(left) = half-box charges | 1.3e-11 of the depletion charge; Gauss residual 1.3e-16 of D_peak (1.6e-3 C/m^2) | 1e-8 / 1e-6 |
| C isotype n-GaAs / N-AlGaAs, low-bias G (+-0.5 mV) | 1/(R_bulk + kT/(q A* T^2 e^{-phi_B/kT})), phi_B from the solved barrier = 0.2177 eV | G_TE/G_oracle - 1 = +9.4e-5 (interface = 92 % of R) | 2e-3 |
| C discriminating: continuous-QFL on the same device | must MISS the oracle | 12x too large (= the bare bulk resistor, G R_bulk = 1.001) | > 5x |
| E barrier sweep phi_B = 0.218 / 0.098 / 0.052 / 0.036 eV | 1/G_TE - 1/G_cont = kT/(q J_s) exactly | identity to 7.9e-6 / 2.2e-5 / 1.0e-3 / 2.4e-3; G_TE/G_cont = 0.082 / 0.285 / 0.688 / 0.804 (monotone to 1) | 1e-2, monotone |
| E A* x 1e6 | TE -> continuous | -1.1e-5 (AlGaAs), -2.4e-7 (zero offset) | 1e-4 |
| 2-D equilibrium n+ GaAs (1e18) / n AlGaAs (1e16) (`slow`) | psi(AlGaAs) - psi(GaAs) = dE_c + kT(eta_A - eta_G) | 0.091516 V vs oracle 0.091511 V (+5e-6 V); discrete Gauss law over the stack to 1.3e-15 of the depletion charge (net charge -3.7e-3: the e^-5 field tail at the 300 nm AlGaAs contact) | 2e-3 V |
| 3-D DD ITO (4e26) / AZO-like (1e26, chi + 0.12 eV), gated | E_Fn = 0 at every node of both regions at Vg = 0 and +0.5 V (no DC path) | max 3.2e-3 / 3.0e-3 eV (eta 17-23: the g fit's Delta vs exact FD, +0.55 % of Delta ~ 16); TE wired ('auto'); oxide-face accumulation x1.52 at +0.5 V | 5e-3 eV |

| F (validation only, `python -m validation.devsim_heterojunction --3d`): same 3-D stack as EQUILIBRIUM vs DD-thermionic vs DD-continuous | no DC path: DD = equilibrium, TE = continuous | TE vs continuous max 1.5e-9 / 1.4e-9 rel (Vg 0 / +0.5 V); DD vs equilibrium 2.8e-3 / 2.1e-3 rel (FD g fit vs Aymerich-Humet closure); equilibrium build 521 s, DD builds 3 s | reported |

1-D prototype (scratch, two same-material unipolar DD regions, 200 nm, 1e16 cm^-3): continuous model
J(10 mV) = 4.005e7 A/m^2 = the ohmic `V/(L/(q mu n))` exactly; zero-offset TE 2.478e7 vs the analytic
series value 2.46e7 (bulk + `kT/(q A* T^2 u e^Delta)`); TE with A* x 1e6 -> 4.005439e7 (1e-6).

Every gate FAILS on v0.13.0 -- checked by running this test file against `git archive 5cbd998`
(18 non-slow tests: 7 failed + 11 errored): `NotImplementedError` "both solve carrier continuity" (9,
every 2-D device fixture), `ModuleNotFoundError: dynameta.carriers.physics_heterojunction` (3),
`TypeError` for the new `heterointerface` / `richardson_n_Am2K2` keywords (4: the 3-D DD heterostack
spec, the isotype fixture, the model-name test). The slow 2-D equilibrium gate returns 0 V on v0.13.0
(every region neutral at psi = 0 in its own frame) against the 0.0915 V oracle.

Run cost (box A, measured --durations): whole file 609 s; the slow equilibrium fixture 457 s, the rest ~150 s
(3-D 53 s, isotype set 32 s, homojunction pair 18 s, p-n 10 s + 15 s of oracle quadrature).

## 5. Limits

* TE magnitude on a degenerate emitting side (MB tail, overestimates); equilibrium exact.
* The FD-SG bulk has an O(h^2) zero-bias discretization "emf" in graded regions (trapezoidal g_avg):
  4e-4 A/m^2 on the p-n gate device, 14 -> 0.9 A/m^2 under 4x refinement in the 1-D prototype. It is a
  bulk property (present in any FD region with a density gradient), not the interface.
* 3-D DD heterostack: extra regions use the primary's mobility (no per-region field in the tuple).
* Equilibrium-mode region setup costs minutes per region (DEVSIM symbolic differentiation of the
  Aymerich-Humet F_1/2), so the equilibrium 2-D gate is marked `slow`.

## 6. Release-note lines

* DEVSIM heterojunctions: semiconductor/semiconductor interfaces with Anderson band offsets
  (`chi_eV`, `band_gap_eV` now load-bearing), thermionic emission (Horio-Yanai / Yang-East-Haddad,
  barrier-side Richardson constant) or continuous quasi-Fermi levels, for unipolar and bipolar
  drift-diffusion in the 2-D `LayeredDevsimBuilder` (`heterointerface=`), and unipolar DD heterostacks in
  3-D (`Stacked3DSpec.heterointerface`). New module `carriers.physics_heterojunction`.
* 2-D builder: adjacent equilibrium semiconductors now carry their built-in potential (S1-9 2-D sibling);
  ohmic contacts are re-referenced to the region's neutral level. Results CHANGE for 2-D designs with two
  or more semiconductor regions of different doping or affinity (single-semiconductor designs unchanged).
* Still refused, by name: 2-D lateral sidewall interfaces, unipolar-next-to-bipolar interfaces, 3-D
  bipolar heterostacks, bipolar heterojunctions whose band gap disagrees with n_i.
