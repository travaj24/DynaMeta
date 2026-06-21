"""dynameta.optics.soa: semiconductor quantum-dot optical amplifier (QD-SOA) models.

A STANDALONE time-domain amplifier subpackage -- the traveling-wave semiconductor gain
counterpart to optics.laser_gain (the four-level ATOMIC laser/cavity gain that feeds the
metasurface FDTD). The SOA is injection-pumped (current, not an optical pump rate),
saturates DYNAMICALLY through signal-driven inversion depletion, and is solved as a
z-resolved split-step marcher -- it does NOT plug into the run_pipeline metasurface
optical_solver seam and does NOT reuse the normal-incidence metasurface FDTD kernel.

See docs/DynaMeta_QD_SOA_extension_spec.md (Sections 6 + 8) for the governing equations and
the corrections this implementation applies. Pure numpy/scipy; SI units; exp(-i omega t)
(gain -> Im(chi) < 0), ASCII-only.

Phase 1 (this module): qd_gain.QDGainModel -- group-resolved WL->ES->GS rate equations,
steady-state, small-signal spectral gain, and the static saturation curve. Phases 2-4
(traveling-wave dynamic coupling, ASE/noise, analog SFDR/ENOB metrics) slot in alongside.
"""

from dynameta.optics.soa.ase_noise import (ase_output_psd, ase_self_consistent,
                                           ase_self_consistent_zresolved,
                                           ase_spectrum_bidirectional, detector_noise_variances,
                                           inversion_factor_nsp, inversion_factor_nsp_eh,
                                           noise_figure, single_pass_gain, spectral_noise_figure)
from dynameta.optics.soa.metrics import (enob, facet_gain_ripple_dB, optimal_drive_power,
                                         pattern_penalty_dB, predistort, ripple_enob_ceiling,
                                         sfdr_dB, sndr_db, sndr_vs_drive, thermal_drift_budget_K)
from dynameta.optics.soa.noise_metrics import (henry_factor, linewidth_from_field, rin_spectrum,
                                               schawlow_townes_henry_linewidth)
from dynameta.optics.soa.qd_gain import (Leakage, ManyBody, QDGainModel, QDGainParams,
                                         SelfHeating)
from dynameta.optics.soa.lineshape import (biexp_memory_kernel, lorentzian_area,
                                           nonmarkovian_lineshape)
from dynameta.optics.soa.sbe import reduced_sbe_susceptibility, sbe_gain_per_m
from dynameta.optics.soa.thermal import (dome_analytic, sample_T_along_axis,
                                         thermal_profile_steady_1d, thermal_profile_transient_1d)
from dynameta.optics.soa.transverse_bpm import TransverseBPM, qd_gain_table
from dynameta.optics.soa.maxwell_bloch import MaxwellBlochEnsemble
from dynameta.optics.soa.calibration import (CalibratedDevice, INNOLUME_BOA1310_TARGETS,
                                             InferredDynamic, calibrate_innolume_boa1310,
                                             infer_dynamics_from_cw)
from dynameta.optics.soa.traveling_wave import (NonlinearLoss, TravelingWaveSOA,
                                                TwoLevelSaturableGain, UltrafastCompression,
                                                agrawal_olsson_output)

__all__ = ["QDGainModel", "QDGainParams", "SelfHeating", "ManyBody", "Leakage",
           "thermal_profile_steady_1d",
           "sample_T_along_axis", "dome_analytic", "NonlinearLoss",
           "TravelingWaveSOA", "TwoLevelSaturableGain",
           "UltrafastCompression", "agrawal_olsson_output", "inversion_factor_nsp",
           "inversion_factor_nsp_eh", "single_pass_gain", "ase_output_psd", "noise_figure",
           "detector_noise_variances", "sndr_db", "enob", "sndr_vs_drive", "optimal_drive_power",
           "predistort", "pattern_penalty_dB", "sfdr_dB", "thermal_drift_budget_K",
           "facet_gain_ripple_dB", "ripple_enob_ceiling",
           "rin_spectrum", "linewidth_from_field", "henry_factor",
           "schawlow_townes_henry_linewidth", "TransverseBPM", "qd_gain_table", "MaxwellBlochEnsemble",
           "CalibratedDevice", "INNOLUME_BOA1310_TARGETS", "calibrate_innolume_boa1310",
           "InferredDynamic", "infer_dynamics_from_cw",
           "thermal_profile_transient_1d",
           "nonmarkovian_lineshape", "biexp_memory_kernel", "lorentzian_area",
           "reduced_sbe_susceptibility", "sbe_gain_per_m"]
