# Stage 1 Recalibration Plan (2026-05-28)

## Background

The current Park 2021 simulation produces a resonance dip at λ ≈ 1150 nm
but Park reports ≈ 1300 nm. Two independent issues identified:

### Issue 1: Halen-Pulfrey F_{1/2} is ~50% under-valued at degenerate eta

Verified numerically vs scipy.integrate.quad of the exact F_{1/2}:

| eta | F_true | HP / true | dF_HP/dF_true |
|---:|---:|:-:|:-:|
| 0   | 0.765 | 0.76 | 0.96 |
| 5   | 8.84  | **0.62** | 0.52 |
| 10  | 24.1  | **0.55** | 0.49 |
| 15  | 43.9  | **0.51** | **0.49** |
| 18  | 57.7  | **0.51** | **0.48** |
| 20  | 67.5  | **0.51** | **0.48** |

Our ITO operates at eta = 12-20 (depending on n_bg). Halen-Pulfrey
returns half the correct F_{1/2} AND half the correct derivative -
meaning n is half right and dn/dV is half right. Bias-induced
modulation is silently halved by this approximation alone.

### Issue 2: n_bg = 8e20 cm^-3 is a stale calibration

`shared/constants.py` sets `ITO_N_BG = 8e20 * 1e6`. The justification
in the comments cites a material audit that empirically tuned it to
match Park's 1300 nm resonance using the OLD 1D pipeline (uniform ITO
eps, coarse 3D mesh). The current 2D-DEVSIM + 3D-FEM pipeline does
NOT reproduce that calibration -- dip is at 1150 nm, not 1300.

Worse: setting n_bg higher pushes eta_bg further into the regime
where Halen-Pulfrey is most broken. So the calibration knob was
counter-tuning the F_{1/2} bug, not fixing real physics.

Park's STATED value is 4e20 cm^-3. We should revert.

---

## The Fix

### Aymerich-Humet (1981): drop-in F_{1/2} replacement

Already implemented in
`dynameta/stage1_carriers/fermi_dirac.py`. Reference:
Aymerich-Humet, Serra-Mestres, Millan (1981), Solid-State Electronics
24, 981. Form:

```
F_{1/2}(eta) = 1 / (A(eta) + B(eta))
A(eta) = (3 sqrt(pi) / 4) / [eta^4 + 50 + 33.6 * eta * (1 - 0.68 exp(-0.17 (eta+1)^2))]^(3/8)
B(eta) = exp(-eta)
```

Verified accuracy < 0.2 % on F_{1/2} and < 1.3 % on dF/d_eta across
eta = -5..20.

### Park's stated material parameters (use as reset baseline)

| Parameter | Current | Park stated | Revert to |
|---|---|---|---|
| `ITO_N_BG` | 8e20 cm^-3 | **4e20 cm^-3** | 4e20 |
| `ITO_EPS_INF` | 3.9 | 3.9 | unchanged |
| `ITO_EPS_DC` | 9.5 | 9.5 | unchanged |
| `ITO_M_EFF_RATIO` | 0.35 | 0.35 | unchanged (legacy scalar) |
| `ITO_M_EFF_LOW` (Kane m*_0) | 0.27 m_e | -- | unchanged; gives ~0.31 at n=4e20 |
| `ITO_KANE_ALPHA` | 0.5 eV^-1 | -- | unchanged |
| `ITO_BANDGAP_EV` | 3.75 | 3.6 | revert to 3.6 |
| `ITO_CHI_EV` | 4.8 | 4.8 | unchanged |
| `EPS_R_HFO2_DC` | 18.0 | -- (high-k) | unchanged |
| `EPS_R_AL2O3_DC` | 9.0 | -- (high-k) | unchanged |
| `N_HFO2_IR` | 1.91 | 1.95 (Hu et al. 2018) | revert to 1.95 |
| `N_AL2O3_IR` | 1.66 | 1.65 | revert to 1.65 |

---

## Files to modify

### Metasurface_Modulator/ (legacy production)

1. `shared/constants.py`
   - `ITO_N_BG = 4.0e20 * 1e6`  (was 8.0e20)
   - `ITO_BANDGAP_EV = 3.6`  (was 3.75)
   - `N_HFO2_IR = 1.95`  (was 1.91)
   - `N_AL2O3_IR = 1.65`  (was 1.66)
2. `stage1_carriers/mos_cap_1d.py` :: function `halen_F_half_expr`
   - Replace with Aymerich-Humet expression
3. `stage1_carriers/mos_cap_1d.py` :: function `setup_phi_c0` or wherever
   `inv_F12` is used
   - Switch to Joyce-Dixon-based inverse for Phi_c0 calibration

### Metasurface_Lib/ (new library)

1. `dynameta/stage1_carriers/fermi_dirac.py` -- DONE
2. `dynameta/stage1_carriers/physics.py`
   - Import Aymerich-Humet expression
   - Update `setup_phi_c0` to use inverse_F12_joyce_dixon
3. `examples/park_2021.py`
   - `ITO_N_BG = 4.0e20 * 1e6`
   - Other audit updates

---

## Validation runs (after edits applied)

All validation outputs land in:
- Legacy:  `Metasurface_Modulator/experiments/2026_05_28_recalibration/`
- Library: `Metasurface_Lib/examples/outputs/exp_2026_05_28_recalibration/`

NOT in the live `stage1_carriers/outputs/`, `stage2_drude/outputs/`,
`stage3_optical/fem/outputs/` trees -- those contain pre-recalibration
data and we don't want to overwrite or muddle.

### Run 1: Stage 1 only at V = 0
- Build 2D DEVSIM device with new constants
- Solve at zero bias
- Inspect carrier density at ITO. Should be uniform at n_bg = 4e20.
- Out: `experiments/2026_05_28_recalibration/01_validation_0V/`
- Time: ~3 minutes

### Run 2: Stage 1 at V_top = +2V
- Confirm carrier accumulation magnitude
- Expected: ~25-30% at top of ITO (from cap electrostatics + correct F_{1/2})
- Was: ~5% at top of ITO (with broken F_{1/2})
- Out: `experiments/2026_05_28_recalibration/02_patch_pm2V/`
- Time: ~3 minutes

### Run 3: Stage 2 + Stage 3 at V = 0
- Apply Drude with corrected n_bg, then FEM at one wavelength scan
- Expected: dip lands near 1300 nm (Park's value)
- Was: dip at 1150 nm
- Out: `experiments/2026_05_28_recalibration/01_validation_0V/`
- Time: ~25-40 minutes for 13-wavelength scan at coarse mesh

If the resonance dip lands at 1300 nm +/- 50 nm, baseline is restored.
Then re-run the 4-bias x 13-lambda sweep into
`experiments/2026_05_28_recalibration/02_patch_pm2V/` and
`experiments/2026_05_28_recalibration/03_mirror_pm2V/`.
