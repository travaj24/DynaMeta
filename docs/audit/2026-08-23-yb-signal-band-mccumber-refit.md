# 2026-08-23 -- Yb signal-band sigma_a McCumber refit

One finding, one fix. Found while defending the wall-plug-efficiency numbers of the
Fiber_Amplifiers SAT56 study (P_sat-targeted EDFA-vs-YDFA at 56 GBaud) against the study
owner's challenge: "the intrinsic efficiency of Yb should be higher -- I don't believe the
Yb numbers." The challenge was right to the extent that one model input was defective; the
audit trail below records the defect, the evidence, the fix, and the re-pinned gates.

## The defect

`spectroscopy.ytterbium()` built the SIGNAL-BAND absorption cross-section as a hand-placed
Gaussian tail -- `(1.030e-6, 0.050e-6, 0.030 * pk_a)`, i.e. 1030 nm centre, 50 nm FWHM, 3%
of the 976 nm peak -- with no detailed-balance tie to `sigma_e`. Against the ion's OWN
McCumber relation (`eps = h c / zero_line_m`, `zero_line_m = 976 nm`, 300 K) the tabulated
reabsorption is too large by:

| lambda | sigma_a (tab) | sigma_a (McCumber from sigma_e) | ratio |
|---|---|---|---|
| 1010 nm | 5.20e-26 m^2 | 3.29e-26 m^2 | 1.6x |
| 1030 nm | 8.10e-26 | 2.37e-26 | 3.4x |
| 1060 nm | 2.99e-26 | 3.96e-27 | 7.5x |
| 1080 nm | 5.06e-27 | 4.70e-28 | 10.8x |

Same defect class as F-1 (the Er C-band McCumber self-violation fixed 2026-08-05), on the
other ion and the other spectrum.

## Why F-12's reading ("it's the eps, not the spectra") does not hold for Yb

The 2026-08-04 audit logged this ratio table as F-12 (informational) and attributed it to
`eps` being taken at the absorption PEAK, noting that the true excitation potential lies
below the peak. That is the right instinct for Er -- whose 1530 nm absorption peak sits well
above its true zero-phonon line -- but not for Yb:

1. **For Yb the sharp 976 nm peak IS the 0 <-> 0 zero-phonon transition.** The manifold
   partition correction `eps = E_zl + kT ln(Z_l/Z_u)` evaluates to ~ +60 cm^-1 with standard
   Yb-silica Stark splittings -- a 0.76x factor on the 300 K ratio, and in the WRONG
   direction to explain a 3-11x deficit. No admissible eps moves a single scalar factor
   enough to reconcile a violation that varies 1.6x -> 10.8x across the band: the SHAPE of
   one spectrum is wrong.
2. **Forcing consistency the other way is impossible**: keeping the tabulated sigma_a would
   require sigma_e(1060) ~ 1.5e-24 m^2, ~5x above every published Yb-silica value.
   sigma_e in the 1010-1100 nm band is one of the best-measured quantities in fiber optics
   (it sets every Yb laser design); the tiny reabsorption is the poorly measured one.
3. **Independent physical anchors** (not McCumber, not this model): Yb-silica transparency
   inversions are ~0.05 at 1030 nm and ~0.01-0.02 at 1064 nm in every published dataset.
   The tabulated pair gives 0.207 and 0.133. Fiber-level: a ~556 dB/m-at-976 core-pumped
   fiber (Gamma n_t sigma_a with n_t = 6e25 m^-3, a = 3 um, NA 0.12) shows 5.84 dB/m of
   unpumped loss at 1060 nm under the tabulated sigma_a; real fiber of that class measures
   ~0.5-1 dB/m (0.1-0.2% of the 976 peak).

Conclusion: the signal-band sigma_a is the defective spectrum. F-12's `mccumber_eps_J` knob
remains valid and orthogonal (it is the right lever for the residual ~0.76x partition
factor and for the thermal-slope overstatement thermal.py documents).

## The fix

`ytterbium(mccumber_refit=True)` -- THE NEW DEFAULT. The mirror of `erbium(cband_refit)`,
with the trusted/derived roles swapped:

* pump band (lambda <= 1000 nm): the tabulated sigma_a, unchanged bit-for-bit;
* signal band (lambda > 1000 nm): sigma_a DERIVED from sigma_e by detailed balance,
  `sigma_a = sigma_e exp((h nu - eps)/kT)`, `eps = h c / zero_line`, T = 300 K
  (`_McCumberAbsorptionHybrid`, duck-typing CrossSectionModel exactly as
  `_McCumberEmission` does).

The 1000 nm crossover sits in the pump/signal dead zone where the two branches agree within
~11% and both are < 2% of the 976 nm peak; the seam is gated. `mccumber_refit=False`
reproduces the pre-2026-08-23 ion BIT-EXACTLY (gated). Both hosts refit identically
(phospho zero line 974.5 nm).

Refit consequences on the shipped aluminosilicate model:

| quantity | legacy | refit |
|---|---|---|
| transparency inversion, 1030 nm | 0.207 | 0.071 |
| transparency inversion, 1060 nm | 0.133 | 0.020 |
| unpumped 1060 nm loss, 556 dB/m-class fiber | 5.84 dB/m | 0.78 dB/m |
| E_sat (3 um core, 1060 nm) | 31.4 uJ | 35.5 uJ |
| max gain/m at the 976 nm inversion clamp (0.505) | 16.3 dB/m | 18.8 dB/m |

## Gates

`tests/test_yb_mccumber_refit.py` (8 gates): signal-band McCumber identity to 1e-12 on both
hosts; the LEGACY variant proven to violate it (>3x @1030, >7x @1060) so the escape hatch
cannot silently drift; pump band bit-identical between variants; escape-hatch bit-exactness
against the frozen legacy peak list; transparency-inversion anchors (refit inside the
textbook windows, legacy outside both); fiber-level unpumped-loss anchor (refit 0.3-1.5
dB/m, legacy > 4, ratio > 5); crossover-seam bound (< 20% step, < 2% of peak); duck-type /
at_temperature composition smoke.

## Re-pinned numbers (each re-measured, with the mechanism named in the test comment)

* `test_audit_2026_08_04_fiber_amp.py` F-13 fully-absorbed-pump gain: 34.72274 ->
  **34.57521 dB** (the 1030 nm ASE band gains 3.4x less reabsorption than the 1060 nm
  signal gains, so ASE competition lowers the settled signal gain 0.148 dB).
* F-14 mean-field spurious branch: was pinned `< 0 dB` (measured -12.6); under the refit
  the same nbar2 ~ 0.036 branch sits just ABOVE the new 0.020 transparency (measured
  +2.196 dB). Re-gated on the decisive separation from the physical branch (> 15 dB;
  measured > 30).
* F-14 resolved-solver twin: separation pin 40 -> **15 dB** (measured good +19.067 / bad
  -1.830 = 20.9 dB; the reabsorption that dug the -30 dB spurious branch is 7.5x smaller).
* `test_fiber_dynamics.py` thermal-profile-worth pin: > 1.0 -> **> 0.3 dB** (measured
  0.479 dB; the discriminating asserts are the 0.02 / 0.05 dB clone-agreement bounds).
* `test_fiber_dynamics.py` cold-profile ASE-runaway fixture: 220 K -> **175 K** (the trip
  temperature moved ~227 K -> ~185 K because the smaller signal-band sigma_a gives less
  ASE feedstock until a deeper cold boost; at 175 K the runaway is unambiguous:
  ase/launched 2.6e80, gain integral 204 vs the 20 limit, march 40 dB below solve()).
* `test_fiber_transverse.py` unsaturated mode-weighted identity: rel 1e-9 -> **5e-9**
  (asymptotic identity; the fixture's pump/ASE-driven radial structure moved to 1.8e-9,
  still 1e7x tighter than the saturated case the gate discriminates against).

## Downstream note

The Fiber_Amplifiers base study's 2026-08-05 Yb results (YDFA baseline, S-sweeps, the
915-vs-976 comparison) were produced with the legacy ion and overstate Yb reabsorption /
understate Yb WPE at moderate powers; its SAT56 study already carries the refit. High-power
(watt-class, extraction-dominated) Yb conclusions are only mildly affected.
