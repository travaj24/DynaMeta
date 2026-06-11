# DynaMeta worked examples

End-to-end workflows wiring the validated physics leaves through `dynameta.drivers` (the
glue layer) into the pipeline and the reliability post-processors. Each script is
exit-gated (`python -m examples.<name>` exits 0 on PASS) and carries a chain-equality gate
against a hand-rolled independent path where applicable.

| Example | Chain | Needs |
|---|---|---|
| `electrothermal_reliability` | bias -> electro-thermal Picard FEM -> oxide TDDB t_BD + CW damage threshold (LIDT runaway) | NGSolve |
| `pcm_reflectance_switching` | anneal pulse -> JMAK crystalline fraction -> `PCMModel` blend -> TMM reflectance, through `run_pipeline` | numpy + tmm |
| `lc_voltage_tuning` | drive voltage -> two-constant director BVP -> effective e-wave index (BYO `EffectModel`) -> TMM etalon spectrum, through `run_pipeline` | numpy/scipy + tmm |
| `llg_faraday_rotation` | applied H -> LLG Gilbert relaxation -> `VectorMagnetoOpticModel` gyrotropic tensor -> Faraday rotation (circular eigenmodes) | numpy/scipy |
| `lumenairy_gated_grating` | gate bias -> DEVSIM ITO accumulation -> graded Drude eps(z) -> Lumenairy RCWA bridge (lamellar 1-D fast path + per-layer absorption), PMM bridge as cross-method referee | devsim + ngsolve + lumenairy |

The electromigration glue (DEVSIM contact currents in `CarrierField.extras` -> J -> Black
MTTF) is exercised against fakes in `tests/test_drivers.py`; it consumes any drift-diffusion
sweep's `extras['contact_currents_A']` via `dynameta.drivers.em_mttf_from_carrier_field`.
