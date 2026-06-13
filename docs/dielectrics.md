# Stage-1 dielectric constants (DC..GHz) in dynameta

This note documents how the **static** (DC) relative permittivities used by the
Stage-1 DEVSIM Poisson solve are chosen, and why they differ from the materials'
**optical** permittivities. It also records a measured-vs-computed comparison for
the the reference modulator stack so the choice of values travels with the code.

## Why Stage 1 needs a *static* permittivity (not the optical one)

The gate-induced carrier accumulation is set by the gate-stack capacitance, which
depends on the **static (DC) permittivity** of the gate dielectrics. From DC up
through GHz -- well below the THz lattice (IR phonon) resonances -- the
permittivity of a good insulator is flat at

    eps(0) = eps_electronic + eps_ionic        (static / "DC" value)

The library's optical models (`DrudeOptical`, `ConstantOptical`) carry only the
**electronic** part (e.g. HfO2 ~4, n~2). The **ionic/lattice** part is what raises
it to the DC value (HfO2 ~18). Using the optical eps in Stage 1 therefore
under-predicts the gate capacitance -- for the reference HfO2/Al2O3 stack that is a
**~4x** under-prediction of the accumulation (see the history note below).

How to set it:
- **Dielectrics:** `Material(..., eps_static_dc=<DC value>)`. If unset, Stage 1
  falls back to the optical eps and prints a loud warning (it is almost always
  wrong for a gate oxide).
- **Semiconductors:** `TransportModel(eps_static=<DC value>, ...)`.

## the reference modulator stack: computed vs measured vs the values used here

The three oxides set the gate capacitance. The metals (Au patch, Al-Nd mirror)
are electrodes -- DC permittivity is not applicable.

| Material | JARVIS-DFT (DFPT, live) | Measured / accepted literature | Value used here | Note |
|---|---|---|---|---|
| **HfO2**  | 23.8  (orthorhombic Pca2_1, JVASP-52653; elec 5.11 + ionic 18.70) | monoclinic (stable bulk) ~18-20; **amorphous ALD ~12-24** (as-dep ~14); cubic ~30; tetragonal ~40+; canonical high-k value ~25 | **18.0** | ALD-film value; DFPT picked a higher-k crystalline phase |
| **Al2O3** | 12.0  (corundum R-3c, JVASP-32; elec 3.20 + ionic 8.80) | standard ~9-10; **amorphous ALD ~7-9**; sapphire (alpha) ~9-10 | **9.0** | ALD-film value; DFPT overestimates by ~25% |
| **In2O3** (ITO host) | none (all 3 polymorphs report 'na') | static eps0 ~8.9-10; optical eps_inf ~4.0-4.1 (single-crystal IR ellipsometry) | **9.5** static / **4.25** optical (reference Fig-S2 fit) | within the measured band on both ends |

### What the comparison shows

1. **DFPT runs systematically high vs measured.** Al2O3 12.0 vs measured ~9-10
   (the corundum *crystal itself* measures ~9-10, so the +25% is a genuine DFPT
   overestimate, not a phase artifact); HfO2 23.8 vs ALD-film ~16-20. This is the
   textbook DFT/DFPT behavior: the band-gap underestimate inflates the electronic
   term and the ionic term is often high too. Raw DFPT numbers are therefore not
   what you want to drop straight into a device model.

2. **Polymorph matters as much as the method.** JARVIS returned *orthorhombic*
   HfO2 (Pca2_1, the ferroelectric phase, k~24-30), not the monoclinic ground
   state (~18) or the **amorphous ALD film** the reference device actually uses
   (~16-20). A database returns the most-stable structure that *has* a computed
   dielectric -- which is not necessarily your film.

3. **The values used here are the device-appropriate ones.** 18 / 9 / 9.5 sit on
   the measured ALD-thin-film / accepted numbers, not the inflated crystalline
   DFPT ones. The live database lookup *confirms* this choice rather than
   overturning it.

4. **ITO/In2O3 is anchored independently at both ends:** the optical eps_inf=4.25
   (reference Fig-S2 Drude fit) matches measured In2O3 eps_inf ~4.0-4.1, and the static
   9.5 sits in the measured eps0 ~8.9-10 band.

### Recommendation

Use the **measured / film** values (the defaults above). Treat the DFPT databases
as a provenance-bearing **cross-check** and a **fallback** when nothing better is
available -- not as the device value. For the most accurate result, a **C-V
measurement of your own HfO2/Al2O3 films** supersedes all of these; record it with
provenance via `DielectricRecord.measured(...)` so it always wins.

## Sourcing values programmatically: `DielectricDB`

`dynameta.materials.DielectricDB` pulls static (DC..GHz) permittivities
from open DFPT databases with provenance, caching, and a measured-override path.

```python
from dynameta.materials import DielectricDB, DielectricRecord

# Measured film values always win; the DB only fills what you don't override.
db = DielectricDB(
    backend="jarvis",                         # NIST JARVIS-DFT, no API key
    overrides={"HfO2": DielectricRecord.measured("HfO2", 18.0, "ALD film, lit.")},
)
db.eps_for("HfO2")              # -> float (static eps, cached to disk)
db.eps_static("Al2O3")         # -> DielectricRecord (source/id/kind/elec/ionic/spg)
db.apply(reg.get("HfO2"), "HfO2")   # sets material.eps_static_dc + stashes record
```

Backends:
- `"jarvis"` (default): NIST JARVIS-DFT via `jarvis-tools` (no API key; first call
  downloads + caches the ~1.6 GB dft_3d dataset). Probes the
  `dfpt_piezo_max_dielectric*` keys; if a material has no computed DFPT dielectric
  (e.g. In2O3 -- all polymorphs 'na') it raises a clear, actionable error.
- `"mp"`: Materials Project via `mp-api` (needs `MP_API_KEY`). Uses
  `e_total` / `e_ionic` / `e_electronic`.

Caveats: values are DFT-computed (~10-30% error), polymorph-dependent, and (for
high-k oxides) typically *overestimates* -- see the comparison above. Always
prefer a measured value for a real fabricated film.

## History

The clean-break library rewrite (OpticalModel/TransportModel split) initially left
the gate dielectrics with only their optical eps feeding Stage-1 Poisson (HfO2 4.0,
Al2O3 2.756) instead of their static DC values. That was a ~4.2x too-low gate
capacitance and under-predicted the +2 V accumulation by ~4x (n_top/n_bg 1.09
instead of 1.35). Fixed 2026-05-31 by adding `Material.eps_static_dc` +
`Material.dc_permittivity()`, making Stage 1 use it (warn loudly on fallback), and
setting HfO2=18.0 / Al2O3=9.0 in the reference example. The original (pre-rewrite)
pipeline used the correct static values, so this never affected earlier results.

## References

- HfO2 high-k amorphous films -- phys. status solidi (b), 2013:
  https://onlinelibrary.wiley.com/doi/full/10.1002/pssb.201248520
- ALD HfO2 films (TDMAH) -- MDPI/PMC, 2023:
  https://pmc.ncbi.nlm.nih.gov/articles/PMC10254648/
- HfO2 polymorph / phase stabilization -- J. Cryst. Growth:
  https://www.sciencedirect.com/science/article/abs/pii/S0022024809009361
- ALD Al2O3 thin films, dielectric -- Sci. Rep., 2022:
  https://www.nature.com/articles/s41598-022-09054-7
- Thickness dependence of ALD Al2O3 -- J. Appl. Phys., 2019:
  https://oxides.net.technion.ac.il/files/2020/02/2019_JAP_thickness_dependence_ALD-Al2O3.pdf
- Static & high-frequency dielectric constants of cubic In2O3 -- J. Appl. Phys.
  129, 225102 (2021):
  https://pubs.aip.org/aip/jap/article-abstract/129/22/225102/1080386/
- Materials Project dielectric methodology (DFPT):
  https://docs.materialsproject.org/methodology/materials-methodology/dielectricity
- NIST JARVIS-DFT: https://www.nist.gov/programs-projects/jarvis-dft
