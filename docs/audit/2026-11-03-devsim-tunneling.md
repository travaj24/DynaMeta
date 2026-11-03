# DEVSIM tunneling: band-to-band generation in the bipolar Newton, gate-oxide tunneling current (2026-11-03)

Workstream D1d of the DEVSIM phase (release v0.14.0). Branch `wip/dv_tunnel` off `feat/devsim-2.11-v0.14`
(5cbd998). **Both features are opt-in and default off.** The defaults-off path is byte-identical to main.
That was checked by SHA-256 over 16 solution arrays of the 1-D and 2-D bipolar diode fixtures, giving the
same hash on both trees (section 6).

---

## 1. What was built

| Piece | Where | Coupling |
|---|---|---|
| Local Kane/Hurkx BTBT net-recombination term in both continuity equations | `carriers/tunneling.py` (`BTBTParams`, `SILICON_HURKX`, `setup_btbt`, `btbt_rate_expr`, `hurkx_d_expr`, `btbt_generation_current`) | **self-consistent** (in the coupled Newton, analytic Jacobian) |
| `setup_bipolar_region(..., btbt=None)` | `carriers/physics_bipolar_dd.py` | calls `setup_btbt` when given |
| `TransportModel.btbt` (layered 2-D builder), `Stacked3DSpec.btbt` (3-D builder) | `materials/transport_model.py`, `carriers/devsim_layered.py`, `carriers/devsim_3d.py` | `bipolar_dd` only (else `ValueError`) |
| Gate-oxide tunneling current through a dielectric region, from the converged potential | `carriers/tunneling.py` (`OxideTunnelingParams`, `oxide_tunneling_current`) | **post-processed** (no feedback) |
| `LayeredDevsimBuilder(..., oxide_tunneling=None)`, `Stacked3DSpec.oxide_tunneling` | builders | adds `CarrierField.extras["oxide_tunneling"]` when set |
| Full Schuegraf-Hu closed form; Tsu-Esaki / WKB-trapezoid quadrature | `reliability/leakage.py` (`schuegraf_hu_current`, `wkb_trapezoid_exponent`, `tsu_esaki_current`) | the single kernel home; the existing three functions are unchanged |
| `eq_registry.amend_region_equation` | `carriers/eq_registry.py` | adds terms to a recorded equation (atomic; refuses to overwrite a term) |

### API

```python
from dynameta.carriers.tunneling import SILICON_HURKX, BTBTParams, OxideTunnelingParams
BP.setup_bipolar_region(dev, "si", ..., btbt=SILICON_HURKX)            # or TransportModel(..., btbt=...)
I = btbt_generation_current(dev, "si", depth_m=period_y)               # q Int G dV  (> 0 = generation)

r = oxide_tunneling_current(dev, "oxide", "si", OxideTunnelingParams(model="fn"), depth_m=period_y)
r["I"], r["J_A_m2"], r["V_ox_V"], r["E_ox_V_m"]                        # per-interface-node arrays
OxideTunnelingParams(model="tsu_esaki", E_F_eV=0.1, m_supply_over_m0=0.5)   # E_F has NO default
LayeredDevsimBuilder(design, oxide_tunneling=OxideTunnelingParams())   # -> extras["oxide_tunneling"][region]
Stacked3DSpec(..., physics="bipolar_dd", btbt=SILICON_HURKX, oxide_tunneling=OxideTunnelingParams())
```

Units follow the D1 contact-current convention: 1-D is per unit area (A/m^2). The 2-D layered mesh is per
unit depth, and `depth_m` (period_y) scales it to A. The 3-D mesh is in metres, so it is already in A.

## 2. Equations and citations

### 2.1 Band-to-band tunneling (local field)

    U_BTBT = A |E|^P exp(-B/|E|) * D(n, p),      D = (n p - n_i^2) / ((n + n_i)(p + n_i))

* The field prefactor is Kane's tunneling rate (E. O. Kane, J. Appl. Phys. 32, 83 (1961)). P = 2 is the
  direct transition; P = 2.5 is the phonon-assisted one.
* D is the Hurkx factor (G. A. M. Hurkx, D. B. M. Klaassen, M. P. G. Knuvers, IEEE TED 39, 331 (1992)).
  * D = 0 at equilibrium.
  * D -> -1 in a depleted reverse-biased junction (net generation).
  * D > 0 under forward bias (tunneling-assisted recombination).
* U is a net *recombination*. It enters as `-q U` in the electron equation and `+q U` in the hole
  equation, with the same sign convention as SRH.
* **Si (Hurkx 1992; also the Hurkx defaults of Sentaurus Device):** A = 4e14 cm^-1/2 V^-5/2 s^-1,
  B = 1.9e7 V/cm, P = 2.5. In SI: `A_SI = A_cgs * 1e6 * 100^-P = 4.0e15 m^-3 s^-1 (V/m)^-2.5`,
  B = 1.9e9 V/m (`BTBTParams.from_cgs`). No other material set is shipped.
* The temperature is fixed at T_REF. The Eg(T)^1.5 scaling of B is not applied.
* The expression is built as `A exp(P ln E - B/E) * D`: one exponential, no overflow. A single builder
  (`btbt_rate_expr`) is used in every dimension.

**Discretisation.**

* **2-D/3-D** use the **element field**. DEVSIM's `element_from_edge_model` gives the exact P1
  gradient per triangle or tetrahedron. U is assembled per element edge into each end node's share of
  the element volume (`volume_node0_model`/`volume_node1_model`, weight `ElementNodeVolume`), with D at
  that node.
  * This is *isotropic*. The impact-ionization post-processor's edge-projected field is not, and it
    reads 1.8-3.1x low on tets.
  * DEVSIM requires the derivative with respect to every element node (`en0..en{d}`) of every
    variable. A missing one is FATAL ("...:Electrons@en1 MISSING").
* **1-D** uses the edge field plus `edge_volume_model`, with D the mean of the two end nodes. 1-D has
  no element models, and `element_from_edge_model` raises there.
* |E| has a 1 V/m floor: sqrt(E.E + 1). The floor bias is < 1e-14 relative above 1e7 V/m.
* The nonlocal (path-integrated) model is a follow-up.

### 2.2 Gate-oxide tunneling

* **FN** (Lenzlinger & Snow, J. Appl. Phys. 40, 278 (1969)): `J = A E^2 exp(-B/E)`, with
  `A = q^3 m0/(8 pi h m_ox phi)` and `B = 8 pi sqrt(2 m_ox) phi^1.5/(3 q h)`. For SiO2 (3.1 eV,
  0.42 m0): B = 242 MV/cm. This is the pre-existing kernel.
* **Full Schuegraf-Hu** (Schuegraf, King & Hu, VLSI Symp. 1992): `J = A E^2 exp(-B g(u)/E) / (1 - sqrt(1-u))^2`,
  with `g = 1 - (1-u)^1.5` and `u = min(qV/phi, 1)`.
  * The prefactor is what linearised WKB gives for a trapezoid: the exponent's slope at E_F drops by
    (1 - sqrt(1-u)).
  * It is identical to FN at V >= phi.
  * The existing `direct_tunneling_current` keeps only the exponent. That is exactly the "2-32x below
    full S-H" the 2026-07-05 audit disclosed. The function is left unchanged.
  * S-H fails as V -> 0: it tends to a nonzero constant.
* **Tsu-Esaki / WKB** (Tsu & Esaki, APL 22, 562 (1973)), the reference for both regimes:
  `J = q m_s kT/(2 pi^2 hbar^3) Int T(E) ln[(1+e^((E_F-E)/kT))/(1+e^((E_F-qV-E)/kT))] dE`.
  * The transmission is the closed-form WKB exponent of the trapezoid,
    `theta = 4 sqrt(2 m_ox)/(3 hbar q F)[U0^1.5 - (U0-qV)_+^1.5]`. It is evaluated cancellation-free.
  * The quadrature is composite 12-point Gauss-Legendre. It has breakpoints at the barrier top, E_F,
    qV - phi and qV. The panel width is 1/4 of min(kT, WKB decay length), and the tail is cut at
    theta - theta(E_F) > 70.
  * The supply E_F and mass are parameters with no default for E_F. A metal is 3-11 eV; an
    accumulated Si surface is ~0.1 eV.
  * Image-force lowering and 2-D quantisation of an accumulation-layer supply are not modelled.

**Field and drop at the interface.**

* V_ox at each interface node is the **column drop**: the potential of the opposite (gate-side) oxide
  face, interpolated to the node's lateral position, minus the node's potential.
* E_ox = V_ox/t_ox. Both are exact for a charge-free planar oxide on any mesh.
* The first implementation used the least-squares nodal gradient at the interface. On a laterally
  uniform 3-D Si/SiO2 stack that varied **17.7 %** across the interface, because it amplifies the 3-D
  FV solution's lateral noise (interface psi spread 57 mV). The column drop varies **1.5 %**.
* The current is `Sum J * SurfaceArea` over the interface nodes.
* Only planar interfaces normal to a coordinate axis are supported (`NotImplementedError` otherwise).
* Sign: J > 0 when the gate side is at the higher potential (electrons tunnel semiconductor -> gate).
  The same barrier/supply parameters serve both polarities.
* `model="fn"` with any node below the barrier raises an `AdvisoryWarning`.

## 3. The coupling choice and its limits

* **BTBT is self-consistent.** It must be, because it *is* the Zener current. The Jacobian is
  analytic (DEVSIM symbolic diffs) and checked against finite differences (J1).
* **The oxide current is post-processed.** The carrier profile is that of an ideal capacitor. This is
  exact to first order while the leakage is small next to the transport currents, which covers the
  static-power, hold-bias and low-f G floor questions R16 asks.
  * It is **wrong** for a floating or deep-depleted body whose minority supply is the tunneling current.
  * It is also wrong under heavy FN injection that changes the charge state.
* **A self-consistent interface sink is feasible but is not built here** (see Open questions). It
  would need:
  * a planar single-oxide expression for E_ox in the semiconductor's variables;
  * a supply-limited sink so a depleted surface cannot go negative;
  * a gate circuit term so the terminal currents balance.
* **The local BTBT self-limits at high current.** The generated carriers drift out at mu E, so
  n ~ J_n/(q mu_n E) in the junction. The Hurkx D therefore loses about (n + p)/n_i
  <= J/(q mu_p Emax n_i) = delta_D.
  * At J ~ 1e4 A/m^2 (the 5e18 cm^-3 diode at -3 V) this costs 39 % against a flat-quasi-Fermi
    estimate.
  * That is the local model's own behaviour (constant mobility; velocity saturation would make it
    worse), not a numerical error. It is quantified in the Z1 tolerance below.

## 4. Gates -- `tests/test_devsim_tunneling.py` (18) + `tests/test_tunneling_kernels.py` (18)

All 36 pass (80 s total; the 3-D wiring test is `slow`, 38 s).

**Every gate fails on the old code.** Run against main's tree, the kernel file fails 18/18 and the
DEVSIM file fails at import (`dynameta.carriers.tunneling` does not exist). Z4 shows the physical gap:
the no-BTBT reverse current at -1 V is 4.7e-8 A/m^2 against 38.9 A/m^2.

### Kernels

| Gate | Oracle (coded in the test) | Measured | Tolerance |
|---|---|---|---|
| K1 | Tsu-Esaki vs `scipy.integrate.quad` of the same integral (6 cases: FN/DT, 0/300 K, metal/Si supply) | <= 6e-11 | 1e-8 |
| K2 | FN reduction: `J_TE/J_FN = 1 - 1/b + 19/(9b^2) - 20/(3b^3) + O(b^-4)`, b = B/E, derived in the test | remainder x b^4 = 21-26 over b = 16-60 | 40/b^4 |
| K3 | S-H == FN *identically* at V >= phi; = exponent-only x (1-sqrt(1-u))^-2 below; TE -> K2 series past phi | dev 4.8e-2 (phi+0.2 V), 1.6e-3 (+0.5 V), <= 1.6e-4 (>= +0.9 V) | exact / 1e-13 / 40/b^4 |
| K4 | 300 K / 0 K = Murphy-Good x/sin x, x = pi c kT/E | 0.3-0.6 % | E/B (first-order curvature) |
| K5 | DT: J ~ V at small V (ohmic supply); >> FN through 2 nm | J/V const to 1.1e-3 | 3e-3 |

### BTBT

Zener diode: 1-D abrupt p+n+, Na = Nd = 5e24 m^-3, 0.2 nm at the junction, Si Hurkx, tau = 1 ms.

| Gate | Oracle | Measured | Tolerance |
|---|---|---|---|
| Z0 | equilibrium: max abs(U) / max Kane prefactor (9.9e25 m^-3 s^-1 at the 8.7e7 V/m built-in field) | 1.8e-15 | 1e-12 |
| Z1 | q Int G vs the integrated analytic estimate (depletion profile, exact first-integral Emax with -2Vt, Hurkx D with flat quasi-Fermi levels), V_r = 0.3-1.2 V (0.34 -> 97 A/m^2) | abs(dev) <= 7.9e-3 | 1 % + delta_D |
| Z2 | d ln J/dV vs analytic | <= 0.59 % | 1.5 % |
| Z3 | terminal current == q Int G | 0.3-3e-3 A/m^2 abs | 32 ulps of the 1.4e12 A/m^2 majority SG term + 1e-4 J |
| Z4 | no BTBT at -1 V | 4.7e-8 vs 38.9 A/m^2 | < 1e-6 J_an |
| Z5 | installed rate == numpy A E^P e^(-B/E) D on the solved arrays | 1e-15 | 1e-12 |
| J1 | BTBT share of the assembled Jacobian (on - off) vs central FD of DEVSIM's residual (`get_matrix_and_rhs`) | 6e-9 (2-D), 1e-8 (3-D); share ~1 % | 1e-6 |
| I1 | isotropy: oblique uniform field on 2-D triangles (30 deg) / 3-D Kuhn tets; quadrature identity Sum(U0+U1) ENV = Sum U(E0, D_i) V_i | abs(E) 8.9e-16; identity 5e-15 | 1e-14 / 1e-12 |
| L1 | 2-D layered builder (element path, `TransportModel.btbt`) CONVERGES to the 1-D twin (edge path) on the same x nodes (rewritten 2026-11-03, see section 8) | gap 2.81e-4 (2 nm), 6.09e-5 (1 nm): order 2.21 | order >= 1.7; abs(gap) <= 2 C_inf h^2 |

| V_r [V] | q Int G [A/m^2] | analytic [A/m^2] | dev | delta_D |
|---|---|---|---|---|
| 0.3 | 0.3401 | 0.3408 | -2.1e-3 | 4.5e-5 |
| 0.6 | 3.825 | 3.828 | -7.4e-4 | 4.5e-4 |
| 1.0 | 38.74 | 38.87 | -3.4e-3 | 4.1e-3 |
| 1.2 | 97.30 | 98.07 | -7.9e-3 | 9.8e-3 |
| 2.0 | 1437 | 1579 | -9.0e-2 | 0.13 |
| 3.0 | 9540 | 15693 | -0.39 | 1.16 (bound void) |

The tolerance terms:

* The 1 % is the 1-D edge-mean-D quadrature: 0.2-0.6 % between the 0.2 and 0.1 nm meshes.
* The roundoff floor in Z3 is real. The SI contact current is a difference of ~1e12 A/m^2 majority
  terms at 5e24 m^-3, so it cannot resolve the BTBT current below ~1e-3 A/m^2.

### Oxide tunneling

MOS capacitor: 1-D, n-Si 1e24 / 5 nm SiO2, accumulation Vg = 4-9 V (E_ox 7.7e8-1.76e9 V/m).

| Gate | Oracle | Measured | Tolerance |
|---|---|---|---|
| F1 | post-processed E_ox vs discrete Gauss law eps_ox E = q Sum(n - p - N_D) V_i | 2.4e-8 .. 5.4e-8 | 1e-6 |
| F2 | ln(J/E^2) vs 1/E_Gauss: slope = -B_FN (re-derived), intercept = A_FN; J == closed form | slope to 1e-6 | 1e-6 / 1e-5 / 5e-6 |
| F3 | FN below the barrier warns; tsu_esaki / schuegraf_hu dispatch == kernels; S-H == FN above phi | exact | 1e-12 |
| F4 | sign flips with gate polarity; linear-potential stack gives E, V_ox exactly | 1e-12 | 1e-12 |
| W1 | layered builder extras == direct call; default adds **no** key; potential identical with/without (post-processor) | exact | exact |
| W3 (slow) | 3-D builder: extras == direct; Sum SurfaceArea = lateral^2; lateral V_ox spread; BTBT ~ 0 in the insulated-gate equilibrium | area 6e-4; spread 1.5 %; abs(U)/Kane 2e-12 | 1e-3 / 3 % / 1e-8 |
| D1 | defaults: no BTBT models, legacy equation kwargs term-for-term; `amend` atomic + refuses overwrite; params validation | -- | -- |

## 5. Observations outside this workstream (not changed)

* The 2-D layered builder's bipolar Newton does **not converge** for a 5e24 m^-3 lateral junction with
  a 3 nm tanh grade, with or without BTBT. The validation diode (1e24, 5 nm) converges. A
  quantitative 2-D Zener therefore needs D0/convergence work.
* The layered builder wires the semiconductor/oxide interface from x = 2.5 nm next to a 1.5 nm
  edge-metal strip. The oxide over [0, 2.5] nm is not on the interface, so the post-processor
  integrates the 37.5 nm that are wired (40 nm cell).
* No "Issues initializing DEVSIM" were seen. DEVSIM 2.11.2 loaded libopenblas on every run.

## 6. Verification

* Byte-identity (defaults off): `validation.bipolar_diode` (1-D, +0.3 / -0.5 V) and
  `validation.bipolar_diode_2d`'s design through `LayeredDevsimBuilder` (+0.3 / -0.4 V). Potential,
  Electrons, Holes, the terminal current and the extras keys give SHA-256
  `f04091e800a41a5f...` on **both** main (5cbd998) and `wip/dv_tunnel`.
* Related existing tests (`-n 2`, not slow; 9 files: wave-7 carriers, conventions, Gummel,
  density-gradient, devsim smoke, gated builder, geometry guards, reliability ext, stacked-3D spec):
  155 passed after one fix. The conventions gate X-24 caught two literal `T_K = 300.0` defaults in the
  new code; they are now `T_REF`.
* `ruff check` is clean on every changed file.

## 7. Open questions (decisions for the user)

1. **Oxide current: post-processed (shipped) or self-consistent default?** The self-consistent option
   is feasible on the planar single-oxide stacks. It is not built because it needs three design
   decisions:
   * the supply limitation: J scaled by n_s/n_ref, or the Tsu-Esaki supply from the local E_F;
   * the gate terminal: a circuit term so I_gate = -I_tunnel;
   * the multi-dielectric stacks, where the WKB is a stacked barrier.
2. **Other BTBT parameter sets** (Ge, InGaAs; Kane direct P = 2). None is shipped without a cited
   source in hand. `BTBTParams.from_cgs` takes them.
3. **The nonlocal BTBT model** (tunneling over the band-bending path) for abrupt junctions where the
   local field overestimates.
4. **A published Hurkx-model I-V with full parameters was not available offline.** The Zener gate is
   against the integrated analytic estimate only.
5. **Merge overlap.**
   * `n_i` in D is a parameter symbol (`setup_btbt(..., n_i=...)`). If D1b introduces an effective
     n_ie node model, pass it.
   * `amend_region_equation` refuses a second module wiring the same `volume_node*_model` slot. D1c
     traps that need a volume term must compose with it.

## 8. Addendum (2026-11-03): gate L1 after the integer-literal fix

The integration found that `btbt_rate_expr`'s `%.17g` wrote integral coefficients as integer literals,
which DEVSIM 2.11.2 clamps to 2147483647 (integration note 2.6). The L1 fixture used A = 4.0e15, so the
original L1 compared the two paths at a rate 1.9e6 times lower than intended. With the true A it failed:
a 2.8e-4 gap against a 1e-6 gate.

**Diagnosis.**

* **Not the quadrature.** On identical fields the two integrated quadratures are algebraically equal on
  this y-invariant tensor mesh. Each right-triangle cell's quadrant of node i lies wholly in that cell
  and carries the cell's field. Measured on the solved fields: edge-mean = node-local to 6e-16. The 2-D
  solution is y-invariant to 1e-15.
* **The solutions differ.** Without BTBT, 2-D and 1-D agree to 7.7e-7. With it, n and p differ by
  0.2-0.8 % in the junction and by a constant factor 0.465 in both minority tails (about 20 mV of
  minority quasi-Fermi level).
* **The cause is the nodal distribution of the source.** Node i receives
  Sum_j pre_ij (L_ij/2) D_i on the 2-D element path and Sum_j pre_ij (L_ij/2) (D_i + D_j)/2 on the 1-D
  edge path.
  * The difference is delta_i = Sum_j pre_ij L_ij (D_j - D_i)/4 = (h^2/4) d/dx(pre dD/dx) + O(h^4).
  * It is a conservative redistribution: it sums to zero, so it moves the current only through the
    solution's response, at O(h^2).
  * With the true A the junction is self-limited: the generated n, p ~ 0.1-0.3 n_i and D ~ -0.5. D then
    changes by orders of magnitude per node, which makes the response visible. Under the clamped A,
    D ~ -1 with nothing to redistribute.

**Refinement study.** Clean uniform x mesh: gmsh-array triangles vs 1-D on identical nodes, -0.5 V,
A = 4e15, B = 4e8 V/m.

| h [nm] | I_2D [A/m^2] | I_1D [A/m^2] | gap | order | gap/h^2 [nm^-2] |
|---|---|---|---|---|---|
| 2.0 | 2259.9046 | 2260.6386 | -3.247e-4 | -- | 8.1e-5 |
| 1.0 | 2307.9633 | 2308.1201 | -6.796e-5 | 2.26 | 6.8e-5 |
| 0.5 | 2322.0945 | 2322.1319 | -1.611e-5 | 2.08 | 6.4e-5 |
| 0.25 | 2325.7848 | 2325.7941 | -3.972e-6 | 2.02 | 6.35e-5 |

The `LayeredDevsimBuilder` fixture itself (h = `x_spacing_feature_mid_m`) gives gaps of 2.81e-4,
6.09e-5, 1.45e-5 and 2.56e-6 at 2, 1, 0.5 and 0.25 nm. Its junction spacing is not a clean refinement
below 1 nm (max dx 0.63 nm and 0.44 nm, plus a 0.03 nm sliver), so the order is read from the uniform
study.

**The new gate.**
* The observed order between 2 and 1 nm must be >= 1.7. Measured: 2.21.
* abs(gap) <= 2 C_inf h^2, with C_inf = 6.35e-5 nm^-2 from the finest uniform pair. The factor 2
  covers the preasymptotic excess (1.28 C_inf at 2 nm) and the builder's non-uniform junction spacing.
* The current must be real: > 1e3 A/m^2, where a clamped A gives ~1e-3.
* The equilibrium residual must be <= 1e-4 of the Kane prefactor integral and must shrink with h.

**Mutation checks** (each reverted with `git checkout`):
* Re-introducing `%.17g` fails L1 and both new integral-coefficient I1 cases.
* A 10 % element-weight error fails L1 (gap -3.4e-2 at 2 nm).
* Swapping the node assignment of D passes. It is another conservative O(h^2) redistribution, so a
  consistent discretisation, which is correct behaviour for a convergence gate.

**Equilibrium with the Fermi-Dirac g-factor (finding).** At V = 0 the L1 fixture carries a small BTBT
current, -6.0e-3 A/m^2 at 2 nm, identical in 2-D and 1-D.
* With `fd_enhancement=False` the discrete equilibrium has n p = n_i^2 to 4.6e-15 and BTBT/Kane = 3.2e-16.
* With the FD g-factor, the discrete equilibrium (edge-averaged g) puts n p off n_i^2 at O(h^2) in the
  junction: BTBT/Kane = 6.4e-5 at 2 nm and 1.6e-5 at 1 nm, order 1.97.
* The 1.3 % maximum mismatch in the degenerate bulk does not shrink with h. It is the genuine failure of
  mass action under FD statistics, and the Hurkx D (written with n_i) does not know it. It is harmless
  there because the field and the Kane rate are about zero, but a degenerate high-field junction would
  see it. Recorded as an open question; nothing changed.

**Clamp audit of the other gates.** `SILICON_HURKX` has A_SI = 4000000000000000.5 (non-integral) and
B = 1.9e9 < 2^31, so `%.17g` always printed it as a float literal. Z0-Z5, J1, I1 and W3 were never
clamped and always discriminated. Only L1 used an integral A. I1 now also runs a set with an integral
A AND an integral B above 2^31 (4e15 and 3e9), the direct regression gate for the literal fix.

## Release-note lines

- **BTBT generation (opt-in):** local Kane/Hurkx band-to-band tunneling in the bipolar drift-diffusion
  Newton. Pass `setup_bipolar_region(btbt=...)`, `TransportModel.btbt` or `Stacked3DSpec.btbt`; the Si
  set is `SILICON_HURKX` (Hurkx 1992). It uses the isotropic element field in 2-D/3-D (exact for an
  oblique uniform field) and is zero at equilibrium via the Hurkx D factor. The Zener I-V matches the
  analytic depletion estimate to < 1 % over 2.5 decades.
- **Gate-oxide tunneling current (opt-in, post-processed):** `oxide_tunneling_current` and builder
  `extras["oxide_tunneling"]`, with Fowler-Nordheim, full Schuegraf-Hu or Tsu-Esaki/WKB. The oxide
  field matches Gauss's law to 5e-8 and the FN slope is -B to 1e-6.
- **`reliability.leakage`:** new `schuegraf_hu_current` (the full closed form, prefactor included) and
  `tsu_esaki_current` (the WKB-trapezoid quadrature, == scipy.quad to 6e-11, reduces to FN with the
  derived 1 - E/B + ... series). The existing functions are unchanged.
- **`eq_registry.amend_region_equation`:** atomically adds terms to a recorded region equation.
- Defaults off: byte-identical results (SHA-256 over the 1-D/2-D bipolar fixtures).
