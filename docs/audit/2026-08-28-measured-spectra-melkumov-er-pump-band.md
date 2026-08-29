# 2026-08-28: measured spectra — Yb Melkumov table + Er pump-band refit

Found by a four-way adversarial verification of the SAT56 study (user challenged the
YDFA's low modeled efficiency). The solver, energy bookkeeping, and code paths were
confirmed correct by independent re-simulation; the defects were in two INPUT spectra.

## Finding 1: Yb sigma_e carries 1.62x too little oscillator strength

`spectroscopy.ytterbium()`'s three-Gaussian emission fit implies (Fuchtbauer-Ladenburg,
n = 1.45) tau_rad = 1.345 ms against the asserted tau_s = 0.83 ms — physically
impossible for a ~fully-radiative two-manifold ion. Independently, sigma_e(1060 nm) =
1.95e-25 m^2 sits 1.6x below the measured aluminosilicate value. Four sources converge:

* Melkumov et al. 2004 (FORC RAS Preprint 5; arXiv:1502.02885), Appendix 2, five
  cross-checked methods: sigma_e(1060) = 3.1e-25, sigma_a(1060) = 5.7e-27,
  sigma_a(976) = 2.69e-24, sigma_e(976) = 2.97e-24, tau = 0.83 ms.
* Paschotta et al. 1997 (IEEE JQE 33, 1049), own Fig. 1/2 (preprint figure images,
  pixel-digitized against the paper's marked construction lines): sigma_e(1060) =
  3.25e-25, sigma_a(1060) = 6.6e-27.
* PyFiberAmp's Paschotta-derived dataset: 3.31e-25 (its sigma_a table above ~1045 nm
  is a digitization-floor artifact — do not use).
* nLIGHT LIEKKI Application Designer curves: ~2.9e-25.

The 976 nm pump-band values and tau were already correct; the Gaussian fit lost the
spectral WINGS (900-960 nearly absent, 1000-1020 3x low, 1075-1200 6.5x low, the
1060 operating point ~1.6x low).

**Fix**: `calibration.ytterbium_melkumov()` — RareEarthIon built from the verbatim
Melkumov aluminosilicate table (848-1180 nm, 1 nm grid around the peak), both sigma_a
and sigma_e tabulated, tau = 0.83 ms, zero line 976 nm. FL-consistent (tau_rad/tau_s
~ 0.85). `spectroscopy.ytterbium()` is UNCHANGED (its Gaussian fit remains for
reproducibility); consumers where the Yb signal-band magnitude is load-bearing should
use the Melkumov factory.

## Finding 2: Er 976 nm pump absorption 1.8x low (Yb-width conflation)

The 4I11/2 entry was a 13 nm-FWHM Gaussian at 980 nm peaking at 1.7e-25 — evaluating
to 1.31e-25 at 976 nm. The measured band (nLIGHT LIEKKI; Kir'yanov et al. IEEE JQE 49,
511 (2013): "peak at 978 nm") peaks at 977-978 nm with ~20-25 nm FWHM and sigma_a(peak)
~ 2.55e-25, so 976 nm sits at ~93% of peak (~2.4e-25). The 11-13 nm width belongs to
the NEIGHBORING Yb3+ 976 peak. Closure checks: Er110-4/125 at n_t = 6.6e25 reproduces
110 dB/m @1530; predicted 978 nm absorption within 10% of Kir'yanov's measured 67.5 dB/m.

**Fix**: `erbium(pump_band_refit=True)` (OPT-IN, default False) replaces the pump peak
with (978 nm, 21 nm FWHM, 2.55e-25). Signal band untouched. Affects pump absorption
length only (less unabsorbed residual on long fibers).

## FL gate is Yb-only — deliberately

Barnes et al. (IEEE JQE 27, 1004 (1991), full text recovered) showed FL cross-section
extraction is unreliable for Er (40% disagreement vs the saturation method; the
sigma_e/sigma_a ordering at the 1530 peak flips between methods). Er's measured
1550-band pair in this library (1.85/2.77e-25) matches a measured telecom-grade
aluminosilicate (Facchini/Exail Er#1: 1.88/2.94e-25) — vindicated within host spread.

## Gates

`tests/test_measured_spectra_2026_08_28.py` (8): Yb FL consistency; DISCRIMINATION
(legacy Gaussian must fail FL at ratio > 1.45); Melkumov anchor values; McCumber
zero-line stability 975.8 +- 4 nm across 1030-1080; realistic unpumped 1060 loss at
commercial doping; Er refit values + 976<peak<979 + sigma_e ~ 0 in band; legacy
bit-exact reproduction; signal band untouched by the pump refit.

## Known conservatisms left open

`YB_1060_REFERENCE`-class fibers at n_t = 6e25 m^-3 imply ~560 dB/m @976 — about 2x
commercial 6/125 parts. WPE-neutral (holding cost has no n_t term; transparency depends
on n_t*L), but lengths are ~2x shorter than buyable fiber would need. Left as-is.

## Follow-up (2026-08-28, same day): the corrected spectra are now the DEFAULTS

At the user's direction ("if the defaults were wrong before, please make these the new
defaults"), v0.11.0 flips both:

- `erbium(pump_band_refit=True)` is the DEFAULT. `pump_band_refit=False` reproduces the
  legacy 980 nm / 13 nm band bit-exactly.
- `ytterbium()` (aluminosilicate) returns the measured Melkumov table
  (`calibration.ytterbium_melkumov()`). Escape hatches, both bit-exact: pass
  `melkumov_tables=False` for the parametric Gaussian-sum ion (mccumber_refit applied,
  default True), or pass `mccumber_refit=True/False` EXPLICITLY -- an explicit
  mccumber_refit always selects the parametric ion, so pre-flip call sites that pinned a
  specific parametric variant keep meaning what they said (mccumber_refit's default is now
  the None sentinel). The phosphosilicate host has no measured table and stays parametric.

Gates: +5 default gates in `tests/test_measured_spectra_2026_08_28.py` (bare ytterbium()
IS melkumov; explicit mccumber_refit still parametric and bit-identical to
melkumov_tables=False; phospho stays parametric; bare erbium() carries the refit band;
legacy Er band reproduced bit-exactly). `tests/test_yb_mccumber_refit.py` now requests its
parametric variants explicitly. Consumers: `sat56_study.py` (Fiber_Amplifiers) keeps
explicit overrides on both paths so MEASURED_SPECTRA=False still reproduces the
pre-2026-08-28 study.

Known consequence: any downstream result produced with bare `erbium()` / `ytterbium()`
before this flip (e.g. the fiber_burst_pam 2 W base-study numbers regenerated 2026-08-24)
reflects the parametric-era defaults and will shift when regenerated.
