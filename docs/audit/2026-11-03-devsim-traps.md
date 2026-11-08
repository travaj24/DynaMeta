# D1c: SRH trap levels, trapped charge and interface traps (D_it) for bipolar DD (2026-11-03)

Workstream D1c of the v0.14.0 DEVSIM phase. Branch `wip/dv_traps` (from `feat/devsim-2.11-v0.14`
at 5cbd998), DEVSIM 2.11.2. New module `dynameta/carriers/physics_traps.py`, hooks in
`physics_bipolar_dd.py`, wiring in `TransportModel`, `LayeredDevsimBuilder` and `Stacked3DSpec`.
Everything is OPT-IN; with the defaults every model string and every solved array is unchanged
(section 5).

## 1. Before

`physics_bipolar_dd.setup_bipolar_region` had one SRH channel at midgap: parameters `n1 = p1 = n_i`,
`USRH = (n p - n_i^2)/(taup (n + n1) + taun (p + p1))`. No trap energy, no second level, no trapped
charge in Poisson, and nothing at all at a semiconductor/insulator interface: the oxide side of a MOS
stack was Potential continuity only, so no D_it, no fixed oxide charge, no surface recombination.
`materials.TrapSpec` (an `enabled` flag, one acceptor density and a depth) has existed since the
early carrier work but no code path reads it (see section 7).

## 2. Formulation (all SI; trap energies in eV from the intrinsic level E_i, + toward E_c)

**Bulk SRH at E_t** [W. Shockley and W. T. Read, Phys. Rev. 87, 835 (1952); R. N. Hall, Phys. Rev. 87,
387 (1952)]. One level of density N_t in steady state:

    U   = (n p - n_ie^2) / ( tau_p0 (n + n1) + tau_n0 (p + p1) )
    n1  = n_ie exp((E_t - E_i)/kT),   p1 = n_ie exp((E_i - E_t)/kT)
    tau_n0 = 1/(sigma_n v_th,n N_t),  tau_p0 = 1/(sigma_p v_th,p N_t)

E_t = E_i reduces to the legacy channel exactly (`exp(0.0) == 1.0`, so `n1` is `n_i` bit for bit).
A degeneracy factor g shifts E_t by kT ln g and is folded into `energy_eV`.
Low-injection lifetime, n-type [D. K. Schroder, *Semiconductor Material and Device Characterization*,
3rd ed., Wiley 2006, ch. 7]:

    tau = tau_p0 (1 + n1/n0) + tau_n0 (p1/n0)      (n0 >> p0)
    tau = [tau_p0 (n0 + n1) + tau_n0 (p0 + p1)] / (n0 + p0)      (general)

**Several discrete levels.** Independent channels: `U = sum_k U_k`, each with its own
`E_t,k, N_t,k, tau_n0,k, tau_p0,k` (no inter-level transitions, which is the standard multi-level
SRH assumption).

**Occupancy and trapped charge.** The steady-state electron occupancy of a level is

    f_t = (c_n n + c_p p1) / (c_n (n + n1) + c_p (p + p1)),     c = sigma v_th
        = (tau_p0 n + tau_n0 p1) / (tau_p0 (n + n1) + tau_n0 (p + p1))

which has the same denominator as U and equals the Fermi function `1/(1 + exp((E_t - E_F)/kT))` in
equilibrium. The charge it adds to Poisson is `-q N_t f_t` for an acceptor-like level (neutral when
empty) and `+q N_t (1 - f_t)` for a donor-like level (neutral when full). `kind="neutral"` keeps a
level recombination-only, which is the legacy assumption N_t -> 0.

**Interface traps** [E. H. Nicollian and J. R. Brews, *MOS (Metal Oxide Semiconductor) Physics and
Technology*, Wiley 1982]. A density D_it(E) [eV^-1 m^-2] can be uniform over a band,
Gaussian, or a set of discrete levels. It is cut into sheet levels `N_it,k = D_it(E_k) dE_k` by the
midpoint rule. Bins are at most kT wide (`bin_eV`), their edges sit on the band ends and on the neutral
level, and so every bin has one charge type. The integrand is thermally smoothed, so the midpoint rule
converges like `exp(-2 pi^2 kT/dE)`. That bound is an ESTIMATE of about 1e-8 at dE = kT away from
the band ends; the measured C-V gates (about 2e-4) are limited by the mesh, not the bins. Each
sheet level takes the occupancy above, evaluated with the surface densities n_s and p_s, and

    Q_it = q sum_donor N_it,k (1 - f_k) - q sum_acceptor N_it,k f_k  (+ Q_f, a fixed sheet charge)

With `charge="split"` (the default) the levels follow the Nicollian-Brews convention: acceptor-like
above the neutral level E_0 (`neutral_level_eV`, default E_i) and donor-like below it. The other
options are all-acceptor, all-donor, or `"none"` (recombination only). In equilibrium this gives
`C_it = -dQ_it/dpsi_s = q^2 D_it` for a uniform D_it when E_F is away from the band ends; the exact
form is `q D_it [f(E_lo) - f(E_hi)]` with D_it in eV^-1 m^-2. The low-frequency C-V stretch-out
follows as `d psi_s/dV_G = C_ox/(C_ox + C_s + C_it)`. Surface SRH runs through the same levels
[W. Fitzgerald and A. S. Grove, Surf. Sci. 9, 347 (1968); A. G. Aberle, S. Glunz and W. Warta,
J. Appl. Phys. 71, 4422 (1992)]:

    U_s = sum_k (n_s p_s - n_ie^2) / ( (n_s + n1,k)/S_p,k + (p_s + p1,k)/S_n,k )   [m^-2 s^-1]
    S_n,k = sigma_n v_th,n N_it,k,  S_p,k = sigma_p v_th,p N_it,k
    low injection, n-type:  S_eff = (n0 + p0) / ((n0 + n1)/S_p + (p0 + p1)/S_n)  ->  S_p

**Quasi-static only.** Occupancy is the steady-state SRH value, an instantaneous function of (n, p).
A DC sweep, and so the low-frequency C-V, is exact. Transient trap kinetics are a follow-up
(section 6).

## 3. DEVSIM realization

* **One kernel.** `srh_rate_expr(n1, p1, taun, taup)` and `srh_occupancy_expr(...)` are plain string
  builders. With default names, `srh_rate_expr()` returns the legacy `USRH` string character for
  character (pinned). The legacy channel, every bulk level and every interface bin go through them;
  a sheet level substitutes `tau -> 1/S`.
* **Additive terms** (`physics_bipolar_dd.add_charge_term` / `add_recombination_term`). Extra charge
  goes in as node models `QTerm_<name>`, a net positive density in units of +q [m^-3]. Extra
  recombination goes in as `RTerm_<name>` [m^-3 s^-1]; a negative value is generation. The composites
  are rebuilt from whichever terms the region carries, read back from DEVSIM's own model list:
  `PotentialNodeCharge = -q (p - n + NetDoping + sum QTerm)` and
  `ElectronGeneration = -q (USRH + sum RTerm)` (and `+q` for holes). Because the list lives in the
  device, there is no module state to go stale across `delete_device`. The equations reference the
  composites by name and are not touched. With no terms, both composites are the legacy strings.
  D1b (Auger, radiative) can add its rates through the same hook.
* **Interface sheet terms.** These are region node models on the semiconductor side, weighted by
  `ItArea<j> = A_node/NodeVolume`. `A_node` is the interface area each node owns, taken from DEVSIM's
  interface element list: 1 in 1-D, half of each interface edge in 2-D, a third of each interface
  triangle in 3-D. DEVSIM integrates a region node model over NodeVolume, so a node contributes
  exactly (sheet quantity) x A_node. The interface's `continuous` PotentialEquation sums the two
  regions' rows at each interface node, so this charge is the Gauss-law surface charge. MEASURED on a
  two-dielectric 1-D stack: the interface potential equals `sigma/(eps1/d1 + eps2/d2)` to the last
  printed digit. Two other routes were tried and rejected:
  * DEVSIM's `fluxterm` and `hybrid` interface types are flux-transfer terms (added to region 0,
    subtracted from region 1). MEASURED on the same stack, they move the charge ACROSS the interface:
    -2x the potential, with opposite signs on the two sides. They are the right tool for
    semiconductor/semiconductor thermionic emission (D1a), not for a trapped sheet.
  * DEVSIM's built-in `SurfaceArea` node model is the SUM over all of a region's interfaces. MEASURED
    node by node in 2-D, it equals the sum of the per-interface areas. Using it would put a gate-oxide D_it
    on a spacer interface's nodes as well. DEVSIM gives a node shared by two interfaces (or by an
    interface and a contact) to the first one only; the per-interface area follows DEVSIM's element
    list, so it inherits that convention.
* **Closed-form Jacobian.** DEVSIM's `simplify(diff())` of a 45-bin interface sum took about 25 s per
  install (MEASURED, 1-D MOS). The closed forms
  `df/dn = cr (cr n1 + p)/D^2`, `df/dp = -(cr n + p1)/D^2`, `dU/dn = N c_n (p D - cr (np - n_i^2))/D^2`
  and `dU/dp = N c_n (n D - (np - n_i^2))/D^2`, with `cr = c_n/c_p` and `D = cr (n + n1) + p + p1`,
  build in 0.07 s. A test checks them against centred finite differences of the term models.

## 4. API

```python
from dynameta.carriers import physics_traps as PT, physics_bipolar_dd as BP

# bulk: the built-in channel at E_t, plus extra charged/neutral levels
BP.setup_bipolar_region(dev, "si", ..., tau_n_s=1e-7, tau_p_s=1e-7,
                        srh_trap_energy_eV=0.25,                       # default 0.0 = legacy midgap
                        traps=[PT.TrapLevel(energy_eV=0.33, density_m3=6e21, kind="acceptor",
                                            sigma_n_m2=1e-19, sigma_p_m2=1e-19, v_th_n_ms=1e5),
                               PT.TrapLevel(energy_eV=-0.3, kind="neutral",
                                            tau_n0_s=5e-10, tau_p0_s=4e-10)])

# interface (after region physics + PE.setup_interface, before solving)
PT.install_interface_traps(dev, "si_ox", PT.InterfaceTraps(
    dit_eV_m2=1e16, band_gap_eV=1.12,              # uniform over +-E_g/2 (or e_lo_eV/e_hi_eV)
    charge="split", neutral_level_eV=0.0,          # acceptor-like above E_0, donor-like below
    sigma_n_m2=1e-20, sigma_p_m2=1e-20,            # S = sigma v_th N_it ; occupancy ratio c_n/c_p
    fixed_charge_C_m2=0.0, recombination=True))
# also: distribution="gaussian" (e_peak_eV, sigma_eV) | "discrete" (levels=((E, N_it_m2), ...));
# a list of InterfaceTraps on one interface; diagnostics ItCharge<j> [C/m^2], ItRecomb<j> [m^-2/s],
# TrapOccupancy<k>.

# builders
TransportModel(..., physics="bipolar_dd", srh_trap_energy_eV=0.2, srh_traps=(TrapLevel(...),))
LayeredDevsimBuilder(design, interface_traps={("si", "ox"): PT.InterfaceTraps(...)})
Stacked3DSpec(physics="bipolar_dd", ..., srh_traps=(...), interface_traps=PT.InterfaceTraps(...))
Stacked3DSpec.from_design(design, interface_traps=...)   # 3-D: every semi/dielectric interface
```

Guards: a charged `TrapLevel` needs `density_m3 > 0`; every level needs lifetimes or cross sections;
`InterfaceTraps` needs positive capture rates (the occupancy uses their ratio) and a band for
"uniform". Trap fields on a non-bipolar `TransportModel` or `Stacked3DSpec` raise. That covers
equilibrium and unipolar DD, which have no holes for the SRH occupancy. A semiconductor/semiconductor
interface raises `NotImplementedError`, because heterojunctions belong to D1a. An interface with no
bipolar side raises `ValueError`, and so does an unknown `interface_traps` key.

## 5. Oracles and gates (all independent of the code under test; numbers MEASURED 2026-11-03)

`tests/test_traps.py` has 25 tests, which pass in 28 s single-process. `validation/traps_oracles.py`
reuses the test builders and closed forms, then adds the old-physics failure table, the cross-version
byte identity and the 3-D check. Every gate below fails on the pre-D1c code; the numbers that show it
are in the last column. The default validation run (A-F, 9 gates) PASSES. `--3d` adds G1, which
passes, and G2, which FAILS for the pre-existing 3-D mesh reason in section 6.

| gate | oracle | measured | pre-D1c |
|---|---|---|---|
| (a) lifetime vs E_t | n-type slab N_D = 1e21 m^-3, tau_n0 = 3e-10 s, tau_p0 = 1e-10 s, G = 1e24 m^-3 s^-1, L = 60 um, centre dp/G with the 1/cosh(L/2L_p) contact correction, against `tau_p0(1 + n1/n0) + tau_n0 p1/n0`; E_t from -0.4 to +0.4 eV, tau from 1.0e-10 to 5.6e-9 s | max abs rel err 2.1e-5 (largest at +-0.4 eV, the dp/n0 residual) | one tau (1.0e-10 s) at every E_t: -99.4 % at -0.4 eV, -98.1 % at +0.4 eV, -52 % at +0.3 eV |
| (a') multi-level | two neutral levels (+0.35, -0.30 eV), built-in channel at 1e3 s: `1/tau = sum 1/tau_k` | -1.7e-7 | n/a (no second level) |
| (b) trap charge | equilibrium slab; acceptor level (N_t = 6e21 m^-3, +0.33 eV) in N_D = 1e22, donor level (4e21, -0.30 eV) in N_A = 1e22; centre majority density against the scipy brentq root of `p - n + N + Q_trap(E_F)/q = 0` with Fermi f | 0 (acceptor) / 1.1e-15 (donor); compensated majority 6.17e21 / 6.57e21 | centre stays at N_D / N_A: +62 % / +52 % |
| (c-i) LF C-V stretch-out | 1-D p-Si (1e17 cm^-3) / 10 nm SiO2, uniform D_it = 1e12 cm^-2 eV^-1 across the gap (C_it/C_ox = 0.46), V_G from -0.6 to +1.2 V; simulated (V_G, psi_s) against `psi_s - (Q_s + Q_it)/C_ox` (Sze closed-form Q_s, closed-form Fermi integral for Q_it) | max abs dV 7.4e-5 V | 0.24 V |
| (c-i') LF slope | `d psi_s/dV_G` (centred differences) against `C_ox/(C_ox + C_s + C_it)` | max rel 2.3e-4 (median 6.2e-5) | up to 35 % |
| (c-ii) flat-band shift | fixed Q_f = q x 5e15 m^-2; V_FB where psi_s = 0, from a cubic through the six nearest points (linear interpolation of the 20 mV sweep is 1.3e-4 V off on curvature alone); `-Q_f/C_ox = -0.23199 V` | abs err 1.9e-6 V | dV_FB = 0 (miss 0.232 V) |
| (d) surface recombination | n-type slab, ohmic at x = 0, Si/SiO2 at x = L = 20 um, one midgap interface level, sigma = 1e-15 cm^2, v_th = 1e5 m/s, `charge="none"`, G = 1e23 m^-3 s^-1; S from inverting the analytic `dp(L; S)` of `D_p dp'' - dp/tau + G = 0`, `dp(0) = 0`, `-D_p dp'(L) = S dp(L)` | S = 10 / 100 / 1000 m/s, rel err -7.7e-4 / -1.1e-4 / -3.9e-5; profile <= 2.0e-5 of G tau | S_meas = 0 (reflecting) |
| (e) defaults off | forward-biased 1-D Si diode: legacy call against `traps=None`, `()`, `srh_trap_energy_eV=0.0` | model inventory and Potential/Electrons/Holes/USRH/PotentialNodeCharge/ElectronGeneration/contact current BYTE-identical | -- |
| (F) cross-version | the same fixtures (1-D diode with FD on, 1-D MOS swept to +0.6 V, 2-D layered bipolar bar) run on main 5cbd998 and on this branch | sha256 identical on all 3 fixtures (diode f344b6b7..., MOS 39553fc4..., layered 792b1278...) | -- |
| (G1) 3-D wiring | `Stacked3DSpec(interface_traps=...)`, full-gate bipolar n-Si cap (12 nm square, 30 nm, 1e23 m^-3, 8 nm SiO2): interface weights | total = lateral^2 exactly; every weighted node on the semi/oxide face (730 nodes); per-node split differs from DEVSIM's `SurfaceArea` by up to 12 % of a node (barycentric thirds against DEVSIM's own partition, same total) | -- |
| (G2) 3-D vs 1-D | D_it-induced psi_s shift at Vg = 0.5 V against the 1-D cap with the same stack and the same FD/phi_bi frame | **FAIL**: -15.8 mV (3-D) vs -19.7 mV (1-D), ratio 0.80. A pre-existing 3-D mesh defect, not the trap code (section 6) | -- |

Further tests:
* the closed-form Jacobian against centred finite differences: 1.4e-8 relative at the optimal step h = 1e-4, against a 1e-6 gate;
* the 2-D per-interface area (it sums to the interface length, separates two interfaces, and matches
  `SurfaceArea` node by node);
* layered-builder installation (the weights equal `SurfaceArea` on a single-interface region);
* spec validation and guards.

Two tuning notes:
* Gate (d) needs a low G. At G = 1e25 the majority-current field and the injection level bias dp by
  about 1e-4. The weak sensitivity at S = 10 m/s (`d ln dp_s / d ln S ~ 0.03`) amplifies that to a
  0.56 % miss in S. Mesh refinement changes nothing, so the miss is physics, not discretization.
* The slab and MOS solves need DEVSIM's absolute tolerance scaled to about 1e-12 of the density.
  Below the update's precision floor the Newton loop ping-pongs, for example abs 1e6 against N = 1e22.

## 6. Limits

* **Quasi-static only.** Trapped charge is an instantaneous function of (n, p), with no trap state
  variable. DC sweeps and the LF C-V are exact. A transient or `ssac` solve with CHARGED traps is NOT
  charge-conserving, because the capture current that fills a trap is not in the continuity time
  term. The high-frequency / conductance-method `C(omega)` of D_it, and trap kinetics in general,
  need occupancy as a solution variable with `df/dt = c_n n (1 - f) - e_n f - c_p p f + e_p (1 - f)`.
  That is the documented follow-up. Recombination-only levels (`kind="neutral"`, `charge="none"`) are
  unaffected.
* **Ohmic contacts pin the TRAP-FREE neutral densities** (`CELEC`/`CHOLE` from NetDoping). On heavily
  trap-compensated material a contact therefore carries a boundary layer a few Debye lengths thick.
  Gate (b) measures at the slab centre for that reason. Making the contact trap-aware needs a per-node
  neutrality root, and the same applies to D1b's incomplete ionization; it is open question 2.
* **Bipolar regions only.** Interface traps on the equilibrium (`n = N_c F_1/2`) or unipolar-DD (ITO)
  regions would need an E_c-referenced Fermi occupancy. The builders raise instead of dropping traps.
* **Interface area** follows DEVSIM's interface element list. A node shared by two interfaces, or by
  an interface and a contact, belongs to the first one only (section 3). Coordinates are Cartesian;
  cylindrical is not supported.
* **Statistics.** Boltzmann statistics are used for n1/p1. With `fd_enhancement=True` the transport
  uses the FD g-factor while the trap occupancy stays Boltzmann; the difference only matters for traps
  at degenerate surfaces.
* **3-D sheet charges are diluted by a PRE-EXISTING 3-D mesh defect (gate G2, FAILS).** The
  trap-free 3-D cap and the 1-D cap agree on psi_s (0.0955 vs 0.0960 V). The D_it shift does not:
  -15.8 mV against -19.7 mV, and a plain fixed Q_f behaves the same way (ratio 0.79). The cause is
  the gmsh tetrahedral mesh of `devsim_3d`, which is not Delaunay. DEVSIM's Voronoi finite volumes
  then over-count: sum(NodeVolume) is 1.196 x the geometric semiconductor volume and 1.293 x the
  oxide's, with no negative EdgeCouple. The discrete capacitances are inflated along with the
  volumes, so a correctly placed GEOMETRIC sheet charge acts at about 0.8 of its strength. MEASURED
  evidence:
  * the interface weights are exact (G1);
  * a pure-Laplace 3-D test (carriers frozen, only the sheet) gives 0.86 of the analytic shift;
  * on the same 3-D mesh, a gate shift of -Q_f/C_ox reproduces the 1-D answer (0.0755 vs 0.0760 V);
  * mesh_min 0.25 / 0.5 / 1.0 nm gives a ratio of 0.793 / 0.792 / 0.769, so refinement does not
    remove it.

  Bulk charge densities are integrated over the same inflated volumes as the fluxes, which is why the
  trap-free pointwise psi and n survive. Integrated quantities do not: a node-volume sum of the
  carrier charge over-counts by roughly the same factor (MEASURED: the trap-free Gauss balance is
  -2.03e-3 C/m^2 of semiconductor charge against 1.75e-3 C/m^2 on the gate). This affects
  `devsim_3d` beyond traps; open question 4. G2 is kept as a real gate so that it turns green when
  the mesh is fixed. It is not relaxed.
* **2-D gated bipolar.** The layered builder accepts `interface_traps`, but its gated-bipolar solve
  still carries the documented phi_bi gate-offset warning (`UnreliableResultWarning`). For a
  quantitative gated result use `Stacked3DSpec` (gate G) or a 1-D stack.

## 7. Open questions (for the user)

1. **The dead `materials.TrapSpec`.** `TransportModel.traps: Optional[TrapSpec]` (`enabled`,
   `N_trap_acceptor_m3`, `E_trap_depth_eV`, `g_trap`) is exported from `dynameta` but read by no code
   path, so `enabled=True` is silently ignored, as it was before this work. Mapping it onto `TrapLevel`
   needs two decisions: what "depth" is measured from (E_c?), and the E_i convention, since E_i is
   not midgap when N_c != N_v. The options are to map it, deprecate it, or make `enabled=True` raise.
   It is left untouched here.
2. **Trap-aware ohmic contacts** (section 6). This is shared with D1b's incomplete ionization, so the
   two workstreams should settle one contact rule.
3. **Transient trap kinetics** (section 6). Is the conductance-method `C(omega)`/`G(omega)` of D_it
   wanted in v0.14 or later?

4. **The `devsim_3d` mesh is not Delaunay** (section 6, gate G2). The fix belongs to whoever owns
   the 3-D builder, and I did not attempt it here. The options are a Delaunay-conforming tetrahedral
   mesh (gmsh's `Mesh.Algorithm3D` choices, or `Mesh.Optimize`/Netgen optimization), or an
   element-based (non-Voronoi) volume and coupling. Until then, `Stacked3DSpec(interface_traps=...)`
   is correct in placement, but the sheet charge acts at about 0.8 of its strength.

## 9. Per-level trap energy reference (follow-up, user-approved)

**What.** `TrapLevel.energy_reference` and `InterfaceTraps.energy_reference` take one of:
* `"intrinsic"`: the DEFAULT and the shipped behaviour, byte-identical. `energy_eV` is E_t - E_ie.
* `"conduction"`: `energy_eV` is E_c - E_t >= 0.
* `"valence"`: `energy_eV` is E_t - E_v >= 0.

This is the same choice as Sentaurus Device's trap energy references `fromMidBandGap` / `fromCondBand`
/ `fromValBand` (Synopsys, *Sentaurus Device User Guide*, chapter "Traps and Fixed Charges"). **Levels
quoted from DLTS as an activation energy (E_c - E_t, E_t - E_v) should use the band-edge reference**
[D. V. Lang, J. Appl. Phys. 45, 3023 (1974); Schroder, 3rd ed., ch. 5]. Such a level keeps its
distance to the band edge it was measured from. The intrinsic reference instead ties it to E_ie, which
in heavily doped material sits dEg/2 closer to the narrowed E_c.

**One conversion, in the kernel.** `physics_traps.level_reference_densities(E, reference, n_i, V_t,
N_c, N_v)` gives the numbers:

    intrinsic : n1 = n_i exp(E/kT),  p1 = n_i exp(-E/kT)         (n1_p1, the shipped arithmetic)
    conduction: n1 = N_c exp(-E/kT), p1 = n_i^2/n1
    valence   : p1 = N_v exp(-E/kT), n1 = n_i^2/p1

`srh_reference_densities(..., bgn, reference)` then applies BGN:
* intrinsic: n1 and p1 x n_ie/n_i, as before;
* conduction: n1 kept, p1' = n_ie^2/n1, because the narrowing moves the edge, not the DOS;
* valence: the mirror.

So n1' p1' = n_ie^2 on every reference. `srh_rate_expr` / `srh_occupancy_expr` pass `reference`
through, so the bulk levels, every interface bin and the noise pair rate all use it.
`equilibrium_trap_levels` tags a band-referenced charged level as `(E, N_t, kind, reference)`, while
intrinsic levels keep the shipped 3-tuple. The neutrality root (`bipolar_options._neutral_root`, which
serves the contacts, the seed and the 3-D phi_bi) reads it through `physics_traps.equilibrium_n1`, so
contacts and transport use the same reference. N_c / N_v are the region's `n_dos_m3` / `n_dos_p_m3`.

For `InterfaceTraps` with a band reference:
* every energy (levels, `e_lo/e_hi`, `e_peak`, `neutral_level_eV`) is quoted in that coordinate;
* `band_gap_eV` spans [0, E_g];
* "above the neutral level" means toward E_c;
* `charge="split"` requires an explicit `neutral_level_eV`.

The built-in `srh_trap_energy_eV` channel keeps the intrinsic reference.

**Composition with density-gradient (00e2c39).** `srh_reference_densities` / `srh_rate_expr` also take
the D2d `scale=` (s = exp((Lambda_n + Lambda_p)/2V_t)). The order is the reference conversion FIRST,
then the DG scale: n1' -> n1' s, p1' -> p1' s, n_i'^2 -> n_i'^2 s^2. Every reference therefore keeps
n1' p1' = n p|_eq. `scale=None` is the unscaled output character for character. Gate:
`test_reference_composes_with_dg_scale`, symbolic plus numeric, for all 3 references x BGN on/off.
DG applies the scale to the built-in USRH only. A band-referenced level that should follow ONE quantum
band edge (n1 x exp(Lambda_n/V_t) for "conduction") would need a per-band scale, which is not modelled.

**Finding (it corrects the premise of integration open item 1).** In this Boltzmann model, the split
fraction gamma does NOT enter SRH at all, on either reference. The narrowed band diagram has
`E_c - E_ie = kT ln(N_c/n_ie)` and `E_ie - E_v = kT ln(N_v/n_ie)` whatever gamma is: gamma moves both
edges together relative to psi, and SRH is local in (n, p). The two references therefore differ by
the gamma-independent factor `exp(dEg/2kT)` in n1 (and in p1), not by `exp(gamma dEg/kT)`. That
gamma-dependent scaling describes a level held fixed in ABSOLUTE energy against the UNnarrowed edges,
which neither reference (nor the user's choice) does. The gates pin gamma-independence explicitly.

**Gates.** `tests/test_trap_energy_reference.py` has 16 tests. The original 15 all FAIL on f7c82db
(12 failed + 3 errors: the keyword does not exist there). The 16th is the DG-scale composition gate,
added after the rebase onto 00e2c39.
* **(a)** p-type slab, N_A = 1e25 m^-3, Slotboom dEg = 60.5 meV (n_ie = 3.22 n_i), tau_n0 = 1e-10 s,
  tau_p0 = 1e-6 s, level at E_c - 0.25 eV. The measured tau matches the closed form with
  n1 = N_c exp(-0.25/kT), p1 = n_ie^2/n1 to -1.8e-7, for gamma = 0.2 and 0.8.
  * The intrinsic reading of the same DLTS number (E_t - E_i = kT ln(N_c/n_i) - 0.25 eV) has n1 and p1
    3.22x higher and tau = 6.695e-10 s against 2.767e-10 s, a factor **2.419**. Its closed form also
    matches, to -5.7e-7. f7c82db can express only this reading.
  * gamma = 0.2 and 0.8 give identical lifetimes (|ratio - 1| < 1e-8).
* **(b)** With equivalent energies (narrowed diagram, as above), the node-evaluated n1'/p1' agree
  across the three references to 1e-11. This holds with no BGN, with gamma = 0.5, and (finding) with
  gamma = 0.2 and 0.8.
* **(c)** n1' p1' = n_ie^2 node by node to 1e-12 on every reference, plus the numeric product to 1e-13.
* **(d)** The default is byte-identical:
  * the kernel strings with `reference="intrinsic"` are the shipped strings (pinned);
  * the intrinsic number path is `n1_p1` itself;
  * `equilibrium_trap_levels` returns the shipped 3-tuple;
  * the existing `tests/test_traps.py` (25), `tests/test_devsim_phase_composition.py` and
    `tests/test_bipolar_recombination_bgn_ionization.py` pass unchanged (counts in the report).
* **(e)** Slotboom BGN with gamma = 0.2, N = 1e24 m^-3 (dEg = 29 meV). One case is an acceptor at
  E_c - 0.10 eV (N_t = 5e23) in n-type; the other is the mirror, a donor at E_v + 0.10 eV in p-type.
  * The scipy brentq root uses the Fermi occupancy written in the narrowed band diagram. The majority
    density is compensated to 0.724 N.
  * The simulated slab matches it at the centre AND at both contact nodes to 1.1e-15. This is the
    trap-aware contact rule reading the same reference.
  * The intrinsic reading of the same quoted level gives a majority density 8.3 % off.

## 8. Release-note lines

* carriers: SRH recombination at an arbitrary trap energy (`setup_bipolar_region(srh_trap_energy_eV=)`,
  `TransportModel.srh_trap_energy_eV`), following Shockley-Read and Hall; the default midgap level is
  bit-identical to before.
* carriers: multiple discrete bulk trap levels (`physics_traps.TrapLevel`), each with its own E_t,
  lifetimes or cross sections, density, and acceptor/donor/neutral charge. Steady-state SRH occupancy
  feeds the trapped charge into Poisson.
* carriers: interface traps at semiconductor/insulator interfaces (`physics_traps.InterfaceTraps`,
  `install_interface_traps`). D_it can be uniform, Gaussian or discrete, with the Nicollian-Brews
  acceptor/donor split, a fixed oxide charge Q_f, and surface SRH with S = sigma v_th N_it. Wired into
  `LayeredDevsimBuilder(interface_traps=)` and `Stacked3DSpec(interface_traps=)`.
* carriers: `add_charge_term` / `add_recombination_term`, a shared additive hook for extra space
  charge and recombination or generation in bipolar regions.
* validation: `tests/test_traps.py` (25 tests) and `validation/traps_oracles.py`. They cover tau(E_t)
  against the closed form, compensation against a neutrality root, MOS LF C-V stretch-out and slope
  against Nicollian-Brews, dV_FB = -Q_f/C_ox, S = sigma v_th N_it, defaults-off byte identity (also
  against main), and 3-D interface wiring (3-D against 1-D D_it is a documented pre-existing 3-D mesh failure).
* carriers: per-level trap energy reference, `TrapLevel(energy_reference=...)` and
  `InterfaceTraps(energy_reference=...)`. The values are "intrinsic" (default, unchanged),
  "conduction" (E_c - E_t) and "valence" (E_t - E_v), as in Sentaurus' fromMidBandGap / fromCondBand /
  fromValBand. A band-referenced level keeps its distance to the NARROWED band edge under bandgap
  narrowing: n1 = N_c exp(-(E_c - E_t)/kT), p1 = n_ie^2/n1, and n1 p1 = n_ie^2 on every reference.
  Use it for DLTS-quoted levels. The contacts' neutrality root uses the same reference.
