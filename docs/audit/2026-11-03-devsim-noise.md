# Small-signal noise of DEVSIM drift-diffusion devices (impedance-field method)

DEVSIM phase, release v0.14.0, workstream D2a, branch `wip/dv_noise` (off `feat/devsim-2.11-v0.14`
at d3a7012, with D1b merged). New capability: terminal current and voltage noise spectral densities
`S_I(f)`, `S_V(f)` of any DEVSIM drift-diffusion device, split by mechanism, by region and by node.

## What changed

Everything is **opt-in**. Nothing runs unless `noise_analysis` is called, and the call only READS the
device. Its DC solution is bit-identical afterwards and no model is left behind (gated). Existing
AC/DC paths are unchanged:
- The new `drive=` keyword defaults to `"voltage"`, which issues the identical DEVSIM call sequence.
- The R26/R27 call-log digest gate (`test_defaults_off_call_log_is_v013_byte_identical`) still passes. It covers `setup_contact_ohmic_bipolar_circuit`.
- `validation/ac_diode.py` prints the same digits as on the integration branch.

| File | Change |
|---|---|
| `dynameta/carriers/noise_analysis.py` (new) | `noise_spectrum`, `terminal_noise`, `NoiseSpectrum`, `TerminalNoise`, `bernoulli`, `MECHANISMS` |
| `dynameta/carriers/physics_bipolar_dd.py` | `setup_contact_ohmic_bipolar_circuit(..., drive="voltage" \| "current")` plus the `_check_drive` helper |
| `dynameta/carriers/physics_drift_diffusion.py` | new `setup_contact_ohmic_dd_circuit` (unipolar circuit ohmic contact, both drives) |
| `dynameta/carriers/__init__.py` | `noise_analysis` added to `__all__` beside `ac_analysis` |
| `validation/devsim_noise.py` (new) | gates A-G with every device builder (~30 s) |
| `tests/test_devsim_noise.py` (new) | 19 tests (~35 s at `-n 2`) |

## Equations

### Impedance-field method

References: Shockley, Copeland & James (1966); Bonani & Ghione (2001), ch. 3-4.

Linearize the device about its DC point. A local fluctuation `s(r, w)` reaches the output circuit
node linearly:

    delta_out(w) = integral G(r, w) s(r, w) d^3r

For spatially uncorrelated sources with one-sided local spectral density `K(r)`, so that
`<s(r) s*(r')> = K delta(r - r')`, the output noise is:

    S_out(f) = integral |G(r, w)|^2 K(r) d^3r

`G` for every source location comes from ONE adjoint solve per frequency (Bonani, Ghione, Pinto &
Smith 1998):

    (J_static + i w J_dynamic)^T y = e_out

Here `J` is the ssac matrix and `e_out` is the unit vector on the output's circuit equation.
All densities are **one-sided** (unilateral). This is the SOA noise code's convention
(`optics/soa/noise_metrics.py`) and Bonani & Ghione's.

### Local noise sources

**Diffusion (velocity-fluctuation) noise.** One source per carrier; the carriers are mutually
uncorrelated (Bonani & Ghione ch. 3; van der Ziel 1986):

    K_n = 4 q^2 D n = 4 kT q mu_n n ,   K_p = 4 kT q mu_p p      (D = mu kT/q)

In a degenerate (FD g-factor) region the NOISE diffusivity stays `mu kT/q`. That is the
Nyquist-consistent choice (Price 1965), not the generalized `g mu kT/q` of the transport model.
The mobility is the edge model `mu_n_edge` when field-dependent mobility is on. That is the
"Nyquist-like" drift-diffusion approximation: there is no hot-carrier noise temperature.

**Generation-recombination (GR) noise.** Each mechanism is one Poissonian net process. One event
creates or removes an electron AND a hole, so the electron and hole sources are FULLY CORRELATED:

    K_U = 2 (G + R)                                            [per unit volume]
    SRH:        R = n p / D,  G = n_ie^2 / D,  D = tau_p (n + n1') + tau_n (p + p1')
    Auger:      (C_n n + C_p p) (n p +- n_ie^2)
    radiative:  B (n p +- n_ie^2)
    external G: counted as Poisson generation (photocurrent shot noise); off with external_generation_shot=False

For SRH this is the standard fast-trap reduction: the trap occupancy follows the carriers
instantaneously. DEVSIM's SRH has no trap variable.

At equilibrium `2(G + R) = 4 R_eq`, which is `4 kT x` the linear-response GR conductance. The GR
source is therefore exactly Nyquist-consistent.

`G + R` is built from the SAME expression builders the region was set up with
(`bipolar_options.srh_expr` / `recombination_exprs`), with `(np - n_ie^2)` mapped to `(np + n_ie^2)`.
The mapping raises if the factor is absent. There is one implementation per kernel.

**1/f noise (opt-in, `hooge_alpha=`).** Hooge 1969; Hooge 1994:

    K_H = alpha_H J J^T / (n f)    per carrier  (a RANK-ONE tensor along J)

**CORRECTION 2026-11-04 (release blocker).** This record first wrote `alpha_H J^2 / (n f)` and the
code summed it edge by edge like the diffusion source. Hooge noise is a fluctuation of the local
conductivity, `delta j = xi J` with one scalar `xi` per point, so one element's edge currents are
fully correlated; the edge sum dropped their cross terms and returned `cos^4 + sin^4` of the truth
for a uniform field at angle theta to the mesh -- MEASURED 0.5000 at 45 deg on a uniform 2-D DD
resistor with exact (right-isosceles) volumes, 1.0000 aligned; gate F, being 1-D, could not see it.
The source is now assembled per element (`noise_analysis` module doc, item 3): exact at any
orientation on well-centred meshes, identical to the old form in 1-D and for axis-aligned current.
Gate: `tests/test_devsim_noise_hooge.py` (0 and 45 deg to 1e-8, measured 3.6e-10; old code 0.5).

### Discretization (box / Scharfetter-Gummel, DEVSIM's own volumes)

**Edge sources.** A current-density source on edge `e` (nodes 0 -> 1, Voronoi couple `c_e`, length
`L_e`) is the edge-projected fluctuation averaged over the edge volume `c_e L_e`. It enters node 0
with `+c_e dj` and node 1 with `-c_e dj`:

    S_e = (c_e / L_e) |y_0 - y_1|^2 K_e

`Sum_e c_e L_e (dy/L_e)^2` is the box scheme's discrete Dirichlet energy. A uniform bar therefore
gives `4kT G` exactly, on any mesh (gate A).

**Edge carrier density.** Inside an SG edge (linear psi, constant J), a distributed source reaches the
nodes with weight `w(x) = exp(-(psi - psi_0)/V_t)`, and `n(x)` is the SG interpolant. Integrating
`K w^2 / (int w)^2` over the edge gives, in closed form:

    n_eff = 1/2 [ n_0 B(v) + n_1 B(-v) ] ,   p_eff = 1/2 [ p_0 B(-v_p) + p_1 B(v_p) ]
    v = vdiff_g, v_p = vdiff_gp   (the edge's own SG arguments, incl. BGN quasi-fields and g)

At equilibrium `q mu n_eff` is EXACTLY the SG edge conductance with respect to the quasi-Fermi
potential, so the discrete network is Nyquist-exact (gates B and G). A plain average of `n_0` and
`n_1` is not Nyquist-exact.

**GR node sources.** A GR fluctuation perturbs the electron residual by `-q V_i dU` and the hole
residual by `+q V_i dU`:

    S_i = q^2 V_i |y_p,i - y_n,i|^2 2 (G + R)_i

**Units.** The volumes are DEVSIM's `EdgeCouple`, `EdgeLength` and `NodeVolume`, so the units follow
the dimension:
- 1-D: per unit area, A^2/Hz/m^2 (like ssac's S/m^2).
- 2-D: per unit depth, A^2/Hz/m.
- 3-D: absolute, A^2/Hz.

## Native vs custom: what DEVSIM does and what this module adds

**What DEVSIM 2.11 provides.** `solve(type="noise", frequency=f, output_node=out)` (DEVSIM manual,
sec. 9.2.3) computes and stores the impedance field. Two kinds of node model appear:
- `<out>_<Equation>_real` / `_imag`: the scalar fields.
- `_gradx/_grady/_gradz`: element-interpolated gradients.

**The native field is exactly the adjoint above.** An independent scipy solve on
`ds.get_matrix_and_rhs` agrees to 1.6e-15 on the electron field and 2.2e-15 on the minority-hole
field. Equilibration matters: unequilibrated scipy `spsolve` is 2.1e-4 off on the minority-hole rows,
while UMFPACK scales internally.

**What DEVSIM does NOT provide: local noise sources.** So the module is:
- **native:** the Green's function;
- **custom:** sources, assembly and outputs (items 2-3 above).

Why not use DEVSIM's own gradients:
1. They are element interpolations, not the box-scheme edge differences the residual uses, so the
   Johnson identity would hold only approximately, not exactly.
2. They are zeroed on contact nodes, which drops the contact edges.

The module therefore takes edge differences of the scalar field itself.

**One fix-up.** A contact node's bulk row is dropped (permutation row -1), so DEVSIM writes `y = 0`
there.
- **Grounded contact:** 0 is the correct Green's function, because its current is not observed.
- **Circuit contact:** the contact-edge current is assembled into the contact's circuit node. The
  correct `y` is therefore that node's adjoint entry, read from the circuit solution
  `noise_<out>_real/_imag`. It equals 1 for a short-circuit current output.

The module asserts the documented zero before substituting, so a change in DEVSIM's semantics raises
rather than silently misleads.

**Interfaces.** Interface nodes of a carrier-continuous interface are handled by DEVSIM's permuted
rows: both copies carry the merged-row value (gate B3).

## API

```python
from dynameta.carriers import noise_analysis as NA

# after a DC solve, with a circuit contact attached (V source 'V1'):
sn = NA.noise_spectrum(device, freqs, output_node="V1.I",          # short-circuit S_I
                       mechanisms=("diffusion_n", "diffusion_p", "gr"),
                       hooge_alpha=None, external_generation_shot=True, maps=False)
sn.S, sn.by_mechanism, sn.by_region, sn.by_region_mechanism, sn.maps, sn.temperature_K

tn = NA.terminal_noise(device, freqs, source_name="V1")           # S_I, Y (ssac), S_V = S_I/|Y|^2

# open-circuit S_V through its own adjoint: a CURRENT-driven contact
BP.setup_contact_ohmic_bipolar_circuit(dev, "hi", node_name="vhi", source_name="I1", drive="current")
NA.noise_spectrum(dev, freqs, output_node="vhi")                  # V^2/Hz
```

**Outputs.**
- **Per region and per mechanism:** always returned, and they sum exactly to `S`.
- **Per-node maps (`maps=True`):** the local contribution density `|G|^2 K`, with each edge
  contribution split half to each end node.

**Requirements and refusals.**
- `output_node` must be a circuit node.
- Bad frequencies, unknown mechanisms or regions, a non-positive `hooge_alpha` and a bad `drive` raise.
- Cylindrical (`edge_volume_model`) and element-assembled currents raise `NotImplementedError`.

**Device coverage.** Works on any DEVSIM device built by DynaMeta:
- 1-D direct;
- 2-D `LayeredDevsimBuilder` after `set_ssac_gate`;
- 3-D gmsh (gate A4).

**Current-drive DC solves.** DEVSIM refuses an AC part on a current source, and the frozen-carrier
Poisson pre-solve is singular with a current drive. A current-driven contact therefore:
- takes its `Z` from a voltage-driven twin;
- solves its DC point straight from the equilibrium seed with the coupled Newton.

## Oracles (all independent of the code under test)

Constants come from `scipy.constants`. Every reference is a closed form, a solved DC current or
conductance (central difference of DEVSIM terminal currents), or the ssac admittance (a different
linear solve). Nothing compares the module with itself.

| Gate | Device | Oracle | Measured | Gate tolerance |
|---|---|---|---|---|
| A1 | bipolar n-Si bar, 1-D, non-uniform mesh | S_I = 4kT G_dc | 8.9e-16 | 1e-9 |
| A2 | unipolar ITO bar, FD g-factor, `setup_contact_ohmic_dd_circuit` | 4kT G_dc | 1.1e-16 | 1e-9 |
| A3 | bipolar bar, 2-D triangles | 4kT G_dc (G_dc = sigma W/L) | 8.9e-16 | 1e-9 |
| A4 | unipolar ITO bar, 3-D gmsh tetra | 4kT G_dc of that mesh | 1.1e-16 | 1e-9 |
| B1 | long Ge-like p-n diode (n_i 1e19, tau 1 us), 0 V, 1 Hz-10 GHz; Re Z spans 0.148 -> 2.9e-6 Ohm m^2 | S_I = 4kT Re Y(f) | 4.5e-6 | 1e-4 |
| B1 | same diode, current-driven contact (separate adjoint) | open-circuit S_V = 4kT Re(1/Y(f)) | 1.0e-5 | 1e-4 |
| B2 | B1 diode + Auger (C 1e-38) + radiative (B 1e-16) + Slotboom BGN | 4kT Re Y | 2.3e-6 | 1e-4 |
| B3 | B1 diode in two regions + carrier-continuous interface | 4kT Re Y, and S equal to single region | 5.8e-6, 7.5e-6 | 1e-4 |
| C1 | long Si p+n diode (tau 20 us, n side 5 L_p), 1 Hz, V = 0.05-0.55 V | S = 2qI coth(qV/2kT) = 2q(I + 2I_s) | <= 2.0e-4 | 3e-3 |
| C1 | same, V >= 0.4 V | full shot noise 2qI | 0.99997-1.00000 | 2e-3 |
| C2 | same, 0.5 V, 1 Hz-1 MHz (S/2qI rises 1 -> 13.4) | van der Ziel S = 4kT Re Y(f) - 2qI | 1.0092-1.0012 | 1.5e-2 |
| D1 | n-slab N_D 1e22, n_i 1e21, SRH 10 ns (tau_eff 1.196 tau), 50 L_D, 0.05 V, f/f_c 1e-3..100 | GR part = ambipolar Lorentzian with contact cosh shorting | <= 3e-5 | 1e-3 |
| D2 | 400 L_D slab | plain Lorentzian: plateau 2(G+R) tau^2 (q(mu_n+mu_p)E)^2/L; S(f_c)/S(0) = 1/2 | 0.9925; 0.5010 | 1.5e-2; 0.01 |
| E | biased 10 um resistor (0.1 V) vs forward diode | resistor S = 4kT G, not 2qI; diode S = 2qI = (1/2) 4kT G_d | 1 + 6e-6 (S/2qI = 0.517); 0.5043 / 0.5021 / 0.5028 at 0.45 / 0.5 / 0.55 V | 1e-4; 1e-2 |
| F | biased 10 um resistor, alpha_H 2e-3, 1 Hz / 1 kHz / 1 MHz | Hooge alpha I^2/(N f) | 7.6e-6 | 1e-4 |
| F2 | uniform 2-D DD resistor, right-isosceles lattice mesh, current at 0 and 45 deg to every edge (2026-11-04) | Hooge alpha I^2/(N f) | 3.6e-10 both (pre-fix 45 deg: 0.5) | 1e-8 |
| G | 2-D layered reference metasurface (degenerate ITO, FD, 9 regions, edge-metal grounds), 1 GHz-1 PHz | 4kT Re Y at Vg 0 V; at +2 V (accumulation) | 1.3e-11; 3.7e-5 | 1e-9; 1e-3 |
| adj | native noise field vs equilibrated scipy adjoint, 1 GHz | identity | 2.2e-15 | 1e-12 |

### Notes on the oracles

**The coth law (C1).** `S = 2qI coth(qV/2kT)` is the ideal-diode statement `2q(I_forward + I_reverse)`.
It covers the full transition:
- at V -> 0 it equals `4kT G_0`;
- at forward bias it is shot noise.

It holds even where the depletion-region SRH current dominates (0.05-0.2 V): in the quasi-Fermi-flat
depletion region `G/R = exp(-qV/kT)` pointwise. Measured `S/2qI = 1.338` at 0.05 V, against
coth(0.97) = 1.338.

**GR carries a large share of the long-diode noise.** At 0.5 V the neutral-region GR source carries
28 % of S (diffusion share 0.721), so the long-diode shot noise needs both mechanisms.

**Why the FDT diode is Ge-like (B1).** It uses n_i = 1e19 deliberately. On the Si-like
`n_i = 1e16` p+n diode, `G_0 ~ 3e-5 S/m^2`, which is below ssac's round-off floor: the measured
`Re Y(1 Hz)` there is -0.27 S/m^2. The noise sum, which is all positive terms, stays exact there.
This is an ssac limitation, recorded under Limits.

**The B2 options gate is load-bearing.** With the Auger and radiative pair rates dropped from the
noise only, the FDT breaks to `S/(4kT Re Y) = 0.692`.

**Gate E is the diffusion-vs-shot discriminant.**
- A code returning `4kT Re Y` (thermal noise) fails the diode by 2x.
- A code returning `2qI` (shot noise) fails the resistor by 1.93x.

### Failure on the old code

**Every gate fails on the old code.** On d3a7012 (pre-D2a), all 19 tests error at fixture setup
with `ImportError: dynameta.carriers.noise_analysis`. The fixture deliberately uses `pytest.fail`,
not `importorskip`, so a missing module cannot become a skip.

## Limits

- **Source set.** Drift-diffusion sources only:
  - no hot-carrier noise temperature (the diffusion source uses the lattice kT);
  - no trap-occupancy dynamics (the fast-trap SRH reduction);
  - no correlated trap noise (Lorentzians of individual trap levels);
  - no generation-recombination through traps added by other workstreams (D1c) unless they enter
    `USRH`.
- **Degenerate regions.** The diffusion source is Nyquist-exact only for `g == 1`. With the FD
  g-factor it is consistent only to the discretization order: 3.7e-5 FDT residual at +2 V
  accumulation in gate G, and exact at flat band.
- **Biased DD resistor.** It is `4kT G` only up to a PHYSICAL drift/Debye-screening term near the
  ohmic contacts. That term is mesh-independent, scales as ~V^2 L^-2.7, and measured 1.1e-3 at
  1 um / 0.1 V and 2.3e-6 at 10 um. It is not an error, but it means `4kT G` is not an oracle off
  equilibrium on short devices.
- **Tiny admittances.** ssac (not the noise) cannot resolve a junction admittance ~1e-11 below the
  device's bulk conductance scale. FDT checks on such devices need a separately computed Re Y.
- **Unsupported geometry / assembly.** Cylindrical (`edge_volume_model`) and element-assembled
  currents are refused.
- **Contacts and DEVSIM version.** Only DEVSIM's ohmic Dirichlet contacts are covered (Schottky or
  tunnelling contacts with their own current models are not). The contact fix-up assumes DEVSIM
  2.11.2's zero-on-contact semantics, and asserts it.
- **Hooge.** Opt-in, with a single alpha_H for both carriers. There is no mobility-fluctuation vs
  number-fluctuation distinction.

## Open questions (user decisions)

1. **External generation default.** Should external generation `G_ext` count as Poisson
   (photocurrent shot noise, the current default) or be deterministic by default? The current
   default is physical for photodetectors; a modulator user driving `G_ext` as a signal may want
   `external_generation_shot=False`.
2. **Layered-builder convenience method.** Should a `LayeredDevsimBuilder.noise_spectrum(...)`
   wrapper be added? Today the module is called on `builder.device` after `set_ssac_gate`, which
   works (gate G).
3. **Hooge.** Should alpha_H get a per-material default (e.g. 2e-3 for Si) through `TransportModel`,
   or stay call-site only (the current behaviour)?

## Merge notes for the integrator

- **Files touched by other workstreams:**
  - `physics_bipolar_dd.py`: an additive keyword and a helper inserted before
    `setup_contact_ohmic_bipolar_circuit`.
  - `physics_drift_diffusion.py`: an appended function.
  - `carriers/__init__.py`: an added `__all__` name.
- **D1a heterojunctions.** Gate B3 shows carrier-continuous interfaces work. A heterojunction
  interface with a thermionic or jump condition (not `continuous`) is not tested.
- **D1c traps.** If trap rates live outside `USRH` or the BipolarOptions builders, they contribute
  NO GR noise, and FDT will flag it. The pattern for adding them is: give the trap model its own
  pair rate in `_gr_pair_expr`.
- **D0 / DEVSIM.** Any DEVSIM upgrade must keep `test_native_noise_solve_is_the_adjoint` green.

## References

- W. Shockley, J. A. Copeland, R. P. James, "The impedance field method of noise calculation in active
  semiconductor devices," in *Quantum Theory of Atoms, Molecules and the Solid State*, P.-O. Lowdin
  (ed.), Academic Press, 1966, pp. 537-563.
- F. Bonani, G. Ghione, *Noise in Semiconductor Devices: Modeling and Simulation*, Springer, 2001
  (ch. 3 local noise sources, ch. 4 Green's-function / impedance-field techniques).
- F. Bonani, G. Ghione, M. R. Pinto, R. K. Smith, "An efficient approach to noise analysis through
  multidimensional physics-based models," IEEE Trans. Electron Devices 45, 261 (1998).
- DEVSIM manual, sec. 9.2.3 "Noise and sensitivity analysis" (devsim.net/solver.html).
- J. B. Johnson, Phys. Rev. 32, 97 (1928); H. Nyquist, Phys. Rev. 32, 110 (1928); H. B. Callen,
  T. A. Welton, Phys. Rev. 83, 34 (1951).
- P. J. Price, "Fluctuations of hot electrons," in *Fluctuation Phenomena in Solids*, R. E. Burgess
  (ed.), Academic Press, 1965.
- K. M. van Vliet, J. R. Fassett, "Fluctuations due to electronic transitions and transport in solids,"
  ibid. (GR Lorentzian).
- A. van der Ziel, "Theory of shot noise in junction diodes and junction transistors," Proc. IRE 43,
  1639 (1955); *Noise in Solid State Devices and Circuits*, Wiley, 1986.
- C.-T. Sah, R. N. Noyce, W. Shockley, Proc. IRE 45, 1228 (1957) (depletion-region SRH current).
- D. L. Scharfetter, H. K. Gummel, IEEE Trans. Electron Devices 16, 64 (1969).
- F. N. Hooge, Phys. Lett. A 29, 139 (1969); F. N. Hooge, IEEE Trans. Electron Devices 41, 1926 (1994).

## Release-note lines

- **New: small-signal noise analysis** (`dynameta.carriers.noise_analysis`, opt-in) by the
  impedance-field method:
  - Outputs: `noise_spectrum` / `terminal_noise` give one-sided S_I(f) and S_V(f) at any circuit
    node of a DEVSIM drift-diffusion device, in 1-D, 2-D and 3-D, split per mechanism, per region
    and per node.
  - Green's function: DEVSIM's native noise (adjoint) solve.
  - Sources: diffusion noise 4q^2 D n on the exact SG edge density; GR noise 2(G+R) for SRH,
    Auger, radiative and external generation; optional Hooge 1/f.
- **Validated** against:
  - Johnson-Nyquist (1e-15, 1-D/2-D/3-D);
  - the fluctuation-dissipation theorem over 1 Hz-10 GHz (S_I and open-circuit S_V, 1e-5);
  - the ideal-diode law 2qI coth(qV/2kT) (2e-4, full shot noise at forward bias);
  - van der Ziel's long-diode high-frequency law;
  - the ambipolar GR Lorentzian (3e-5);
  - Hooge;
  - the 2-D layered metasurface (FDT 1e-11).
- `physics_bipolar_dd.setup_contact_ohmic_bipolar_circuit(drive="current")` gives a
  current-driven (open-circuit) terminal, and the new
  `physics_drift_diffusion.setup_contact_ohmic_dd_circuit` gives circuit contacts for unipolar DD.
  Defaults are byte-identical.
