# ENZ material presets: doped oxides, nitrides, mid-IR III-V and phonon-polariton crystals

Two modules build ready-to-use materials for **epsilon-near-zero (ENZ)** layers:

| Factory | Material | ENZ mechanism | Gate-tunable? | Typical films / data the numbers come from |
|---|---|---|---|---|
| `tco_presets.azo(n_bg_m3)` | Al-doped ZnO | free carriers, near-IR | yes | sputtered, ALD, PLD films, 1e20-1e21 cm^-3 |
| `tco_presets.gzo(n_bg_m3)` | Ga-doped ZnO | free carriers, near-IR | yes | MBE, PLD, sputtered films, 1e20-1e21 cm^-3 |
| `tco_presets.in_cdo(n_bg_m3)` | In-doped CdO | free carriers, near/mid-IR | yes | HiPIMS epitaxial films, 1e19-1.2e21 cm^-3 |
| `tco_presets.ito(n_bg_m3)` | Sn-doped In2O3 | free carriers, near-IR | yes | sputtered films, 1e20-1.2e21 cm^-3 |
| `tco_presets.fto(n_bg_m3)` | F-doped SnO2 | free carriers, near-IR | yes | spray-pyrolysis / CVD films, 1.5e19-5e20 cm^-3 |
| `enz_presets.inas(n_bg_m3)` | n-InAs | free carriers, mid-IR (4-20 um) | yes, strongly | MBE Si:InAs, 1e18-1e20 cm^-3 |
| `enz_presets.inassb(n_bg_m3)` | n-InAs(0.91)Sb(0.09) | free carriers, mid-IR | yes, strongly | MBE on GaSb, 1e18-1e20 cm^-3 |
| `enz_presets.tin()`, `zrn()` | TiN, ZrN | free carriers, visible (~410-480 nm) | no (n ~ 4e22 cm^-3) | sputtered films |
| `enz_presets.sic(polytype)` | 3C / 4H / 6H-SiC | lattice (LO phonon), ~10.3 um | no | single crystals |
| `enz_presets.aln()` | wurtzite AlN | lattice, ~11 um | no | epitaxial films |
| `enz_presets.sio2()` | fused silica | lattice (glass band), ~8 um | no | bulk glass |

All densities passed to the factories are in **m^-3** (1e20 cm^-3 = 1e26 m^-3); a value that looks like
cm^-3 is refused.

## Two ways to make Re(eps) = 0

- **Free carriers.** A conductor's electron gas gives a negative Drude term that grows toward long
  wavelengths. Where it cancels the positive background, Re(eps) = 0. The ENZ wavelength scales as
  `sqrt(m_opt / n)`, so a gate that adds or removes electrons near a surface moves it. That works in the
  oxides and, strongly, in n-InAs: its light mass gives a large response per electron. It does not work in
  the nitrides: their electron density is so high that the gate charge is screened within 0.06 nm and the
  film's electron count changes by less than 1e-3.
- **Lattice vibrations.** In a polar crystal the ions vibrate against each other. Between the transverse
  (TO) and longitudinal (LO) optical-phonon frequencies, Re(eps) < 0: this is the Reststrahlen band, where
  the crystal reflects almost perfectly. Re(eps) crosses zero again at the LO frequency, the lattice ENZ
  point. With no free carriers there is nothing for a gate to move. The resonances are sharp and low-loss
  in crystals (SiC, AlN) and broad in glass (SiO2). Doping a polar crystal adds a Drude term. The plasma
  and the LO phonon then mix into two coupled modes (L- and L+), and both move with density.

The rest of this page covers the doped oxides first (the gate-tunable near-IR family) and then the other
families.

## The idea in one paragraph (doped oxides)

A heavily doped oxide is a dielectric (background permittivity `eps_inf`) filled with a gas of free
electrons. The electrons give a Drude term `-omega_p^2 / (omega (omega + i gamma))` with the plasma
frequency `omega_p^2 = n e^2 / (eps0 m_opt)`. Where the two cancel, `Re(eps) = 0`: that wavelength is the
**ENZ wavelength** (also called the screened plasma wavelength). A gate adds or removes electrons near the
film surface and moves the ENZ wavelength of that layer -- the effect gate-tunable ENZ devices use. Three
things decide how big and how lossy the effect is: the **optical mass** `m_opt` (heavier mass, longer ENZ
wavelength, smaller response per electron), the **damping** `gamma` (the width of the resonance, i.e. the
loss) and the **background** `eps_inf`.

## What a preset contains

Each factory returns a `Material` with three parts built on **one** conduction band:

- **Band.** A Kane (nonparabolic) band `E (1 + alpha E) = hbar^2 k^2 / (2 m0)`: `m0` is the mass at the band
  bottom and `alpha` says how fast the mass grows with energy. The optical mass of a film at density `n` is
  the mass at the Fermi level, `m_opt = m0 (1 + 2 alpha E_F)`, so it rises with doping (ZnO: 0.29 m_e at
  1e20 cm^-3, 0.42 m_e at 1e21 cm^-3). The same band sets the electron density in the electrostatics
  (`TransportModel(band=KaneBand(...))`), so the charge and the optics cannot disagree about the mass.
- **Optics.** `DrudeOptical(eps_inf, m_opt(n), gamma(n))` with `gamma(n) = e / (m_opt(n) mu_opt)`, where
  `mu_opt` is the **optical mobility** -- the mobility a Drude fit of near-infrared spectra returns. In:CdO
  also carries a density-dependent background (below).
- **Transport and scattering.** A `PolycrystallineScattering` model: the optical damping is the rate inside
  the grains; the DC (Hall) mobility is the intra-grain DC mobility, optionally in series with a
  grain-boundary barrier. The DC mobility is only read by drift-diffusion physics; the default equilibrium
  electrostatics needs only the static permittivity and the band.

`dual_gate_film_kwargs(material)` turns a preset into the keyword arguments of
`carriers.DualGateSchrodingerPoisson` (density, band-edge mass, static permittivity, nonparabolicity).

## Parameters and sources

The full tables, with every source and its status (read in the primary text, abstract only, secondary, or
derived here), are in `tco_presets.PARAMETER_TABLES` and in the module docstring. The defaults:

| Quantity | AZO / GZO | In:CdO | ITO |
|---|---|---|---|
| `eps_inf` | 3.6 (film fits 2.9-3.9) | 5.5 at 1.1e19 cm^-3, falling to 5.1 at 4.1e20 (measured 5.1-5.3) | 4.0 (films 3.5-4.3; bulk In2O3 4.05-4.08) |
| band-edge mass `m0` | 0.24 m_e (0.24-0.28) | 0.10 m_e (0.10-0.16) | 0.21 m_e (0.18-0.30) |
| nonparabolicity `alpha` | 0.33 /eV (0.27-0.33) | 0.735 /eV (0.5-0.74) | 0.52 /eV (0.18-0.52) |
| optical mobility | 38 cm^2/Vs (34-44 good films; ALD 10-33) | 298-365 cm^2/Vs (measured table) | 30 cm^2/Vs (near-IR fits 15-36; Hall 7-43) |
| DC mobility | Masetti law, 54 / 35 / 18 at 1e20 / 5e20 / 1e21 | = optical | = optical |
| static permittivity | 8.7 (along the c axis; 7.8-8.9) | 21.9 | 9.0 (8.9-10.55) |
| band gap | 3.37 eV | 0.9 eV indirect (2.16 eV direct) | 2.8 eV |
| electron affinity | not set (3.5-4.5 eV spread) | 5.8 eV | not set (4.2-5.3 eV) |

Main sources: ZnO -- Romanyuk et al. 2016, Kim et al. 2008, Ruske et al. 2009 (band); Ball et al. 2023,
Kinsey et al. 2015, Knoops et al. 2015, Kim et al. 2013 (optical); Ellmer & Mientus 2008 (mobility laws,
grain barriers); Ashkenov et al. 2003, Pellicer-Porres et al. 2011 (permittivities). CdO -- Nolen et al.
2020, Liu et al. 2016, Mendelsberg et al. 2012 (band, background), Schrecengost et al. 2024 (background),
Cleri et al. 2021 (mobility, optical gap), Jefferson et al. 2008 (direct gap). ITO -- Markwitz et al. 2024,
Feneberg et al. 2016, Fujiwara & Kondo 2005, Liu et al. 2014 (band); Stokey et al. 2021, Hamberg & Granqvist
1986, Blair et al. 2023, Naik et al. 2013 (background, damping); Losego et al. 2009, Korneluk et al. 2023
(Hall mobility, density range).

**Note on the "C" of the TCO literature.** Papers write `m* = m0 sqrt(1 + 2 C hbar^2 (3 pi^2 N)^(2/3) / m0)`;
that `C` is the `alpha` above. Nolen et al. (2020) print this formula, but their fitted `C = 1.47 /eV`
reproduces their own masses only without the factor 2, so the preset uses `alpha = 0.735 /eV`.

## How well the presets reproduce published films

| Reference | Published | Preset |
|---|---|---|
| GZO, MBE, 6.07 / 5.28 / 5.17e20 cm^-3 (Ball 2023) | cross-over 1560 / 1700 / 1710 nm | 1583 / 1676 / 1690 nm (default `eps_inf`); 1560 / 1699 / 1714 nm with the films' own `eps_inf` |
| AZO, PLD, ~4.3e20 cm^-3 (Naik 2012) | metallic beyond 1.84 um | 1.83 um |
| AZO, PLD, 9.8e20 cm^-3 (Kinsey 2015) | eps(1.3 um) = 0.22 + 0.26i | 0.03 + 0.27i |
| In:CdO, HiPIMS, 6.6e19 / 1.4e20 / 4.1e20 (Nolen 2020; from their screened plasma frequencies) | ENZ 3.87 / 2.72 / 1.92 um | 3.91 / 2.89 / 1.91 um |
| In:CdO, 2.8e20 (Yang 2017) | ENZ 2.10 um | 2.21 um |
| In:CdO, sputtered, 5.4 / 6.6 / 8.9 / 11e20 (Liu 2016, independent films) | 1.70 / 1.58 / 1.43 / 1.29 um | 1.72 / 1.59 / 1.42 / 1.31 um |
| In:CdO optical gap, 2.2e19-5e20 (Cleri 2021) | 2.28-2.86 eV | 2.27-2.99 eV |
| ITO optical mass, 2.9e20 / 1.01e21 (Markwitz 2024) | 0.33 / 0.44 m_e | 0.337 / 0.452 m_e |

The tests (`tests/test_tco_presets.py`) pin these comparisons within the literature spread and check that a
wrong band (for example a parabolic one) fails them.

## Validity, and why the presets are only starting points

- **Density range.** A factory refuses a background density outside the range of the films its
  parameters come from (pass `strict=False` to extrapolate with a warning). Inside a gated film the local
  density may leave that range (depleted or accumulated layers); every closure stays finite there.
- **Deposition matters more than the dopant.** For the same density, AZO films span an optical mobility of
  10-45 cm^2/Vs between ALD and PLD, and the fitted `eps_inf` of doped ZnO spans 2.9-3.9. AZO and GZO share
  one implementation: the electrons sit in the same ZnO band; the dopant mainly changes how many electrons a
  film can hold and how good the film is, not the band, within what the literature resolves.
- **Wavelength window.** The damping comes from near-infrared spectra. Ionized-impurity scattering is weaker
  at optical frequencies than at DC (near-IR mobility ~30 % above the DC one in doped ZnO), so for mid-IR or
  THz work set the damping from data in that window. The In:CdO background is evaluated below the
  (renormalized) direct gap only -- wavelengths above about 0.7 um -- and raises otherwise.
- **In:CdO background fall.** Nolen et al. 2020 state that `eps_inf` falls "from 5.6 to 5.1" (abstract) or
  "from 5.5 to 5.1" (text) over their films (1.1e19-4.1e20 cm^-3). Read off their figure, the lowest value
  (5.1) belongs to the ~3.5e20 film; the 4.1e20 film is ~5.2. Liu et al. 2016 tabulate 5.0 at 1e21 cm^-3
  (text: "~5.5 to 4.9" over 1e20-1.2e21, all dopants), and Schrecengost et al. 2024 fit 5.3 at 2.5-4.7e20.
  The default (`edge_amplitude` 7.81) takes Nolen's stated end points: 5.1 at 4.1e20 and 4.87 at 1e21, 0.1-0.2
  below the measured films above ~3e20. It is kept because, with the preset's band, it reproduces the
  measured screened plasma wavelengths best. `edge_amplitude=6.2` matches the measured `eps_inf` (5.18 at
  4.1e20, 5.00 at 1e21) and makes the ENZ wavelengths above 5e20 1-1.6 % longer. Film-to-film scatter is
  about +/-0.1.
- **FDTD.** The single-pole FDTD Drude cannot carry a density-dependent background: build In:CdO with
  `background=None` for that route.

## Using your own film

1. Measure the Hall density `N` and the complex permittivity (spectroscopic ellipsometry) of your film.
2. Fit the Drude parameters:

   ```python
   from dynameta.materials import fit_drude_params
   fit = fit_drude_params(n_m3=N_samples, lambda_m=lam, eps_re=e1, eps_im=e2)
   # fit["eps_inf"], fit["m_eff_ratio"], fit["gamma_rad_s"]
   ```

3. Feed them back. With one film, keep the preset's band shape and set what you measured:

   ```python
   from dynameta.constants import Q_E
   from dynameta.materials import tco_presets as TP
   m_N = float(TP.azo(N).optical.m_opt_kg(N))          # the preset band's optical mass at N
   mu_opt = Q_E / (m_N * fit["gamma_rad_s"])            # reproduces your fitted damping at N
   mat = TP.azo(N, eps_inf=fit["eps_inf"], optical_mobility_m2Vs=mu_opt)
   ```

   With films of several densities, fit `m_opt(N)` with the Kane law and pass `m0_ratio=`, `alpha_per_eV=`.
   A measured `eps_inf(N)` goes in as `background=TabulatedBackground(...)`.
4. For a polycrystalline film whose Hall mobility is below its optical mobility, add a grain-boundary
   barrier: `TP.azo(N, grain_boundary=TP.zno_seto_barrier(trap_density_cm2=..., grain_size_m=...))`.

A different oxide altogether is built the same way the factories do it: `KaneBand(m0, alpha)`, a
`DrudeOptical` with `m_opt_kg=band.optical_mass()` and `gamma_rad_s=OpticalMobilityDamping(mu, band)`, a
`TransportModel(band=band, eps_static=..., physics="equilibrium")`, and `Material(...)`.

## Physics added with the presets (and what was left out)

| Item | Size of the effect | Decision |
|---|---|---|
| Ionized-impurity scattering of a degenerate, nonparabolic band | DC mobility 130 / 79 / 61 cm^2/Vs at 1e20 / 5e20 / 1e21 in ZnO, 20-40 % below the parabolic result | added: `IonizedImpurityScattering`, usable in the Matthiessen sum (`ResolvedMatthiessenGamma`) |
| Empirical intra-grain mobility limit | 54-18 cm^2/Vs in ZnO | added: `MasettiMobility` (ZnO, SnO2 and ITO fits) |
| Grain-boundary barriers; optical vs Hall mobility | Hall mobility 1.5-2.4x below optical in ALD AZO | added: `SetoGrainBarrier`, `PolycrystallineScattering` (optical damping unchanged, DC mobility in series) |
| Density-dependent background (band filling, gap renormalization) | In:CdO: `eps_inf` 5.5 -> 5.1, 4 % in ENZ wavelength | In:CdO default via the existing `BandFillingBackground` (gap and renormalization from the literature; amplitude fitted to one dataset, checked against an independent optical-gap dataset); ZnO: no verified renormalization coefficient, so not shipped (without it the model's ZnO gap at 5e20 cm^-3 is ~0.9 eV too high) |
| Frequency-dependent damping | near-IR mobility ~30 % above DC in ZnO films | not added: the presets take the damping from near-IR data, so the reduction is already in the numbers for that window; only one film's crossover parameters are published |
| Wurtzite anisotropy | `eps_inf` 3.70 vs 3.78 (2 %, ~1 % in ENZ wavelength); mass anisotropy < 10 %, sign inconsistent | below the film-to-film spread; optional tensor `uniaxial=True` (`UniaxialOptical`) |
| Polarization charge at c-axis ZnO faces | 0.057 C/m^2 per face, comparable to a gate (0.041 C/m^2 for 8 nm of eps-25 oxide at 1.5 V) | helper `wurtzite_polarization_charges` for the existing interface-charge inputs; not applied by default (compensation in sputtered / ALD films is unmeasured, mixed polarity cancels it) |

## FTO (F-doped SnO2)

`fto(n_bg_m3)` uses the same machinery as the other oxides. Its defaults are the polycrystalline-film values
of Rey et al. 2013:

- band 0.09 m_e with alpha 0.8 /eV;
- `eps_inf` 3.9;
- optical mobility 35 cm^2/Vs;
- DC mobility from the SnO2 Masetti law (Ellmer & Mientus 2008).

The two published mass laws disagree by about 30 %. The epitaxial SnO2 law (Feneberg et al. 2019, 0.197 m_e
and 0.56 /eV orientation average) is heavier. For epitaxial material pass `m0_ratio=0.197, alpha_per_eV=0.56`.

The preset reproduces the films' screened plasma wavelengths within 0.4 / 3 / 8 %. Those wavelengths are
1639 / 2203 / 2988 nm at 4.3 / 2.2 / 0.9e20 cm^-3.

## Lattice (phonon) permittivity models

`dynameta.materials.lattice_optical` adds three `OpticalModel` classes. None of them depends on density.

**`PolarLatticeOptical`** implements the factorized TO/LO form that infrared ellipsometry fits use:

    eps = eps_inf * prod_j (w_LO,j^2 - w^2 - i w g_LO,j) / (w_TO,j^2 - w^2 - i w g_TO,j)

- Each mode carries its own LO and TO damping.
- Unlike a Lorentz sum, this form gives gain for an unphysical damping pair. One mode is passive only for
  `g_TO <= g_LO <= g_TO (w_LO/w_TO)^2`. The constructor refuses a non-passive set and says why.
- `static_permittivity()` returns the Lyddane-Sachs-Teller value `eps_inf prod (w_LO/w_TO)^2`. `check_lst()`
  compares it with a measured static permittivity.

**`LorentzOscillatorOptical`** is the classical sum of Lorentz oscillators, with an optional Drude term. The
nitrides use it. Its Lorentz term is `lorentz_susceptibility`, the library's single Lorentz kernel.
`optics.resonance.lorentzian_eps` and `optics.fdtd_seam.fit_drude_lorentz` call the same kernel.

**`GaussianOscillatorOptical`** models Gaussian-broadened bands, with their exact Kramers-Kronig partner
(the Dawson function). Glass needs it: its amorphous bands are too broad for Lorentz lines.

Common features:

- **Uniaxial crystals.** 4H/6H-SiC and AlN have different modes for fields perpendicular and parallel to the
  c axis. Build them as `UniaxialOptical(ordinary, extraordinary)`, with the c axis along z.
- **Doped polar crystals.** Every model can serve as the background of a `DrudeOptical`; `with_free_carriers()`
  builds this. The free-carrier term then sits on the full lattice response, so the coupled
  LO-phonon-plasmon modes appear as the zeros of the total permittivity. `coupled_mode_frequencies()` gives
  the undamped closed form for a single mode.
- **Validity window.** `wavelength_range_m` makes `eps()` refuse wavelengths outside the data a model was
  fitted to.
- **Temperature.** Temperature dependence of the phonon frequencies and damping is not included; no clean,
  verified parameterization was found. All values are room-temperature values.

**Which optical solvers can use these models as-is:**

| Route | Scalar lattice / oscillator model | Uniaxial tensor (`UniaxialOptical`) |
|---|---|---|
| planar TMM (`optics.tmm_reference`) | yes | refused (scalar only) |
| RCWA / PMM bridges | yes | yes, for laterally uniform tensor layers |
| Berreman bridge | yes | yes |
| FEM (NGSolve, `fem_2d_tm`) | yes | yes |
| FDTD | one Drude + one Lorentz pole: sample eps over the band and fit `fit_drude_lorentz` (exact for one mode with g_TO = g_LO); the 1-D engine refuses the Lorentz pole | no |

## Phonon-polariton presets

| Preset | Parameters (cm^-1) | Lattice ENZ | Source |
|---|---|---|---|
| `sic("4H")` | E perp c: eps_inf 6.56, TO 796.6, LO 968.7; E par c: 6.78, 783.6, 964.2; damping 2 | 10.32 / 10.37 um | Mutschke et al. 1999 (literature averages) |
| `sic("6H")` | perp 6.56, 797.0, 969.9; par 6.72, 788.1, 965.3 | 10.31 / 10.36 um | same |
| `sic("3C")` | 6.49, 795.9, 972.3 (cubic) | 10.29 um | same |
| `aln()` | perp: 4.08, TO 666.4, LO 909.9, damping 3.1; par: 4.13, TO 610, LO 886.4, damping 9.1 | 10.99 / 11.28 um | Schoeche et al. 2017; A1(TO) from Raman (Iakovenko et al. 2003) |
| `sio2()` | 8 Gaussian bands, eps_inf 2.1232 | 8.02 um (1246 cm^-1) | Kitamura, Pilon, Jonasz 2007 (fused silica) |

**Checks against published data:**

- The SiC small-particle resonance wavelengths (Re eps = -2) of Mutschke et al. match within 0.015 um.
- The 4H-SiC ENZ matches 10.3 um and the AlN ENZ matches 10.9 um (both Caldwell et al. 2015).
- Fused silica matches the thermal-oxide TO-LO pair 1076 / 1256 cm^-1 (Kirk 1988) within 0.5 %.
- Thin film on metal: the Berreman absorption peak sits at the LO frequency for p-polarized light at oblique
  incidence, and no such peak appears for s-polarized light. This is checked through the planar TMM.
- Doped 4H-SiC: the upper coupled mode lands in the 950-1050 cm^-1 band where Raman sees it at 2.6-4.3e18
  cm^-3 (Hergert et al. 2023).

**Limitations:**

- Sputtered, sub-stoichiometric SiO2 sits about 40 cm^-1 below fused silica. Fit your own film.
- The A1(TO) damping of AlN has not been measured. The preset uses the passive equal-damping choice.
- AlN's static permittivity along c is uncertain by ~4 %: the preset's mixed parameter set gives 8.72 (LST),
  Schoeche et al.'s own fit implies 9.11.

## Metallic nitrides (TiN, ZrN)

Both are passive plasmonic / ENZ metals (`Material(is_metal=True)`), modelled as Drude + Lorentz in eV and
valid from 350 to 2000 nm.

**TiN.** The default set is the 200 nm, 23 C film of Reddy et al. 2017.
- Crossover: 481.5 nm, against 479 nm measured for TiN with x = 0.99 (Judek et al. 2021).
- eps(830 nm) = -13.4 + 5.1i. This lies inside the span of three measured films, whose real part runs from
  -11.7 to -19.1 (Guler et al. 2015).
- Two other published film sets are available through `parameter_set=`.
- Stoichiometry dominates: the crossover runs from 392 to 608 nm for x = 0.84-1.21.

**ZrN.** There is a single complete published fit (Naik et al. 2013).
- Crossover: 409 nm.
- eps(830 nm) = -19.6 + 9.2i, against -21.2 + 7.6i measured on a different film.

Gate tuning is negligible for both (see above).

## Heavily doped InAs and InAsSb (mid-IR)

Both presets use a two-band Kane conduction band built from the gap and the Kane matrix element
(`kane_two_band`).

**InAs** (Eg 0.354 eV, P^2 11.9 eV):
- Band-edge mass 0.0218 m_e and alpha 2.70 /eV. The optical mass triples from 1e18 to 1e20 cm^-3.
- The band reproduces the masses stated by Dyksik et al. 2016 within 5 % (1.7 % at 1e18, 4.5 % at 1e19 cm^-3).
- `inas()` includes the InAs lattice (TO 217.4, LO 238.5 cm^-1). The resulting plasmon-LO-phonon coupling
  shifts the ENZ wavelength by about 2.4 % at 1.5e18 cm^-3 and by less than 0.5 % above 1e19 cm^-3.
- An `InterbandOnsetGuard` refuses photon energies at or above the interband onset Eg + E_F(n), where a Drude
  model stops being valid. The onset is about 1.7 um at 1e19 cm^-3 and about 0.93 um at 1e20 cm^-3.
- Damping comes from the fitted values of Wei et al. 2017: 1.4e13 and 1.9e13 rad/s at a nominal 1.25e19 and
  7.35e19 cm^-3 (74 and 101 cm^-1). The higher value sits mid-range of the 0.9-2.9e13 rad/s measured on Si:InAs
  films at 4-5.5e19 cm^-3 (Wei et al. 2020). The lower one is about twice the DC rate implied by the Hall law,
  so it errs on the lossy side. Expect roughly 0.6-1.6e13 rad/s at 1.25e19 and 0.9-3e13 at 7.35e19 between
  films. The DC mobility follows the Hilsum-form law of Petrovic et al. 2024. It gives 1200-1500 cm^2/Vs at
  4-5.5e19 cm^-3, against the 1040-1265 cm^2/Vs Hall data of Wei et al. 2020.
- Published screened plasma wavelengths at equal nominal density disagree by 7-25 %, mostly from how each
  paper calibrates density. The preset lies inside that spread.

**InAsSb** (`inassb()`, x ~ 0.09):
- Uses the alloy gap 0.288 eV with the InAs P^2, and InAs's damping table and DC mobility law. For the alloy the
  InAs damping is an upper-loss estimate: the one fitted InAsSb:Si value found is 5.9e12 rad/s at 1.4e19
  cm^-3 (Moreau et al., preprint), and no measured rate at 3-7e19 cm^-3 was found.
- Reproduces Taliercio et al. 2015 within 4 % at 3.4e19 cm^-3: 0.097 m_e (they report 0.1) and 6.12 um
  (they report 6.25).
- The alloy's phonons are not sourced, so there is no lattice term.

## Does the lattice matter for the oxide ENZ films?

Adding the polar phonon to the oxide presets shifts the ENZ wavelength by these amounts:

| Preset and density | ENZ wavelength | Shift from the phonon |
|---|---|---|
| AZO, 5e20 cm^-3 | ~1.7 um | 0.27 % |
| In:CdO, 2.8e20 cm^-3 | ~2.3 um | 0.54 % |
| In:CdO, 3e19 cm^-3 | ~5.4 um | 3.1 % |
| In:CdO, 1.5e19 cm^-3 | ~7.4 um | 5.6 % |

Phonon parameters used: ZnO TO 408 and LO 592 cm^-1 (Ashkenov et al., secondary source); CdO TO 266 cm^-1,
with LO 531 cm^-1 derived from the Lyddane-Sachs-Teller relation.

The phonon is therefore negligible for near-IR ENZ films but matters for low-density, mid-IR CdO. The CdO
damping is not sourced, so the In:CdO preset does not include the phonon. For mid-IR CdO, add it yourself:
either a composite background, or `background=None` plus a lattice through `with_free_carriers`.

## Not covered, and what each would need

| Family | Status | Reason / what it would need |
|---|---|---|
| amorphous IZO | **not shipped** | No open-access IZO measurement of eps_inf, band mass, nonparabolicity or an ENZ point was found. Only the mobility (~50 cm^2/Vs at 1e20) and the optical gap are verified. The documented stand-in, `ito(..., m0_ratio=0.18, alpha_per_eV=0.5, eps_inf=4.05)` with the IZO mobility, is unvalidated. |
| doped TiO2, SrTiO3-family conductors | not shipped | SrTiO3's very large, field-dependent static permittivity needs a nonlinear dielectric in the electrostatics. |
| hBN (hyperbolic phonon polariton) | not shipped | The machinery is in place (`PolarLatticeOptical` + `UniaxialOptical`), but cited parameters were not collected. |
| temperature-dependent phonons | not included | No clean, verified parameterization was found. |
