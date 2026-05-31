# Stage 1 Recalibration — Library-Version Runs — 2026-05-28

Library-version (`metasurface_lib/`) experiments using the recalibrated
physics. Counterpart to
`Metasurface_Modulator/experiments/2026_05_28_recalibration/`.

See `Metasurface_Lib/RECALIBRATION_PLAN.md` for the rationale.

## Subdirectories

| Subdir | Purpose | Status |
|---|---|---|
| `01_validation_0V/` | One-off Stage 1 solve at 0 V to verify the Park 2021 design loads, mesh builds, and carrier density at the ITO is uniform at n_bg = 4e20. | pending |
| `02_park_2021_quick/` | Single-bias 3-wavelength smoke test of full pipeline (Stage 1+2+3). Verifies the library end-to-end. | pending |
| `03_park_2021_full/` | 4 biases × 13 wavelengths sweep matching the legacy Metasurface_Modulator run. Direct comparison. | pending |

## How to launch

```bash
cd Metasurface_Lib

# Quick smoke (recommended first)
python -m examples.park_2021 --quick \
    --out examples/outputs/exp_2026_05_28_recalibration/02_park_2021_quick/

# Full sweep (after smoke validates)
python -m examples.park_2021 \
    --out examples/outputs/exp_2026_05_28_recalibration/03_park_2021_full/
```

Each run writes:
- `stage1_carriers/carrier_field_<bias_label>.zarr`
- `stage2_drude/eps_<bias_label>.zarr`
- `stage3_optical/spectra.csv`, `spectra_overlay.png`
- `run_meta.json`
