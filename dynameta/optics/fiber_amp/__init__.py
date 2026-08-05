"""dynameta.optics.fiber_amp: rare-earth-doped fiber amplifier (EDFA / YDFA) physics models.

A STANDALONE amplifier subpackage -- the RARE-EARTH (atomic) fiber-gain counterpart to the
semiconductor optics.soa (QD-SOA) build and to optics.laser_gain (the four-level ATOMIC
cavity gain that feeds the metasurface FDTD). Unlike the SOA it is OPTICALLY pumped (a pump
photon flux, not an injection current) and it is a SPATIALLY DISTRIBUTED, weakly-guiding
single-mode waveguide amplifier -- so the model is a z-resolved coupled-power propagation
along the fiber, not a lumped cavity. It does NOT plug into the run_pipeline metasurface
optical_solver seam and does NOT reuse the FDTD kernel.

SHARED-CORE DESIGN. Er3+ and Yb3+ are both rare-earth quasi-two/three-level ions modelled by
the SAME Giles-Desurvire coupled-power formalism (the z-resolved pump/signal/ASE ODEs with a
local upper-level fraction set by absorption/emission cross-sections and the mode-doping
overlap). So the engine is ONE `rare_earth` core parameterized by ion-specific SPECTROSCOPY
(cross-section spectra, level lifetimes, excited-state absorption, host) supplied by
spectroscopy_er / spectroscopy_yb -- exactly how soa.qd_gain is the shared semiconductor core.
Er = quasi-two-level (4I15/2 ground, 4I13/2 metastable, tau ~ 10 ms; 980 nm / 1480 nm pump,
1530-1565 nm C-band signal). Yb = quasi-three-level with strong ground-state signal
reabsorption (2F7/2 ground, 2F5/2 upper, tau ~ 0.8-1 ms; 915/940/976 nm pump, 1000-1100 nm
signal; broad cross-sections).

Governing references (extract-the-formulation-first): Giles & Desurvire, JLT 9(2):271 (1991)
[the EDFA two-level coupled-power model]; Desurvire, "Erbium-Doped Fiber Amplifiers" (1994);
Paschotta et al., IEEE JQE 33(7):1049 (1997) [YDFA quasi-three-level]; Barnard et al., IEEE
JQE 30(8):1817 (1994) [analytical rare-earth amplifier]; Frantz & Nodvik, JAP 34:2346 (1963)
[saturable-gain pulse energy extraction]. McCumber relation links emission/absorption
cross-sections. Pure numpy/scipy; SI units; exp(-i omega t) (gain -> Im(chi) < 0); ASCII-only.

Module map (phased build; each phase ships with discrimination-proven validation gates):
  Phase 1  spectroscopy_er / spectroscopy_yb + rare_earth : cross-sections, McCumber, level
           lifetimes, ESA, overlap Gamma, the local N2/N upper-level fraction.
  Phase 2  steady_state          : the z-resolved two-point BVP -- coupled pump / signal /
           forward+backward ASE power ODEs; gain, output power, pump depletion, N2(z).
  Phase 3  ase / noise           : spectral ASE (spontaneous seeding 2 h nu d-nu / mode),
           inversion factor n_sp, noise figure NF = (2 n_sp (G-1) + 1)/G, OSNR.
  Phase 4  metrics               : saturated output power, gain compression/tilt, power
           conversion efficiency, slope + quantum efficiency, gain flatness.
  Phase 5  concentration         : Er pair-induced quenching + cooperative upconversion;
           Yb photodarkening. OPT-IN (off by default -> byte-identical ideal model).
  Phase 6  cladding / thermal    : double-clad pump overlap Gamma_p = A_core/A_clad, quantum-
           defect heat load, radial thermal profile (high-power Yb).
  Phase 7  dynamics              : transient N2(z, t) gain dynamics (add/drop, self-pulsing)
           + Frantz-Nodvik pulse energy extraction.
  Phase 8  calibration           : plug in datasheet cross-section spectra + measured gain/NF
           (mirrors soa.calibration).

Accuracy extensions (all opt-in; byte-identical to the ideal model when off):
  Phase 9  ESA (spectroscopy)     : excited-state absorption sigma_esa -- a parasitic beam loss
           ~nbar2 (Er 980 nm pump ESA; Yb ESA-free).
  Phase 10 temperature (spectroscopy): McCumber-scaled sigma_e(T) (at_temperature) + multiphonon
           energy-gap-law lifetime (multiphonon_lifetime).
  Phase 11 detection              : detector shot / signal-spont / spont-spont beat noise,
           electrical SNR, and a beat-noise NF that reduces to the optical NF.

Pulsed / chirped-pulse amplification (pulse.py, cpa.py):
  Phase 12 pulse                  : the gain-GNLSE envelope model (dispersion + Kerr + gain)
           solved by the symmetric split-step Fourier method (propagate_gnlse).
  Phase 13 pulse (SaturableGain)  : saturable, spectrally-shaped gain -> gain narrowing; opt-in
           Frantz-Nodvik temporal reshaping (frantz_nodvik=True) for the short-pulse regime.
  Phase 14 cpa                    : stretcher/compressor chain, B-integral, Strehl / compression
           metrics (cpa_chain, strehl_ratio, transform_limited).

Realism extensions (2026-07 generality campaign; all opt-in / standalone):
  Phase 15 nonlinear_limits       : SBS/SRS thresholds (passive Smith + active gain-integral
           forms), TMI threshold estimator, double-Rayleigh MPI + RIN.
  Phase 16 steady_state.RamanStokes: the SRS Stokes channel COUPLED into the solve (Manley-Rowe
           exchange + spontaneous seeding); pulse (raman/self_steepening): delayed Raman h_R +
           optical shock in the GNLSE.
  Phase 17 thermal (+ solver hook): thermal lens, thermal-guiding onset, and the SELF-CONSISTENT
           distributed-T(z) feedback loop (set_temperature_profile: per-z McCumber sigma_e).
  Phase 18 eryb                   : Er:Yb co-doped amplifier (Yb-sensitized transfer).
  Phase 19 lma                    : LP-mode solver, per-mode dopant overlaps, Marcuse bend loss,
           cladding-pump geometry efficiency.
  Phase 20 polarization / chain   : PDG/PHB two-pol model + measured anchors; multi-stage chains
           with PSD-based Friis-reproducing noise cascade.

Link-level layers (2026-08-04 consumer audit; both are BOOKKEEPING over the optical core, no new
physics, and neither is reachable from the solvers -- nothing below them imports them):
  Phase 21 efficiency             : the ELECTRICAL (wall-plug) budget the package previously
           stopped short of -- pump-diode WPE, fibre coupling, controller/TEC overhead, the
           electrical -> diode -> coupling -> absorption -> quantum-defect -> extraction chain,
           and energy per bit. metrics.py's PCE is optical-optical only (audit F-7).
  Phase 22 comms                  : IM-DD link metrics for a modulated signal -- PAM-N levels,
           unequal-variance maximum-likelihood decision thresholds, Gaussian SER/BER, Q factor,
           and the largest constellation an operating point supports. The per-level variances are
           EVALUATED per level rather than scaled, because two of the four terms are
           level-independent (audit F-6).

FACADE CONTRACT (audit X-10). This is an EAGER, EXHAUSTIVE facade -- unlike dynameta.optics and
dynameta.carriers, which are deliberate PEP-562 lazy facades whose gaps are by design. Every name
in a submodule's __all__ MUST be re-exported here; drift is a bug, not a design choice (it left
lma's own result type LPMode and its bend-loss / effective-area API invisible from the package).
tests/test_fiber_amp.py::test_package_facade_is_exhaustive enforces it mechanically.
"""

from dynameta.optics.fiber_amp.spectroscopy import (CrossSectionModel, RareEarthIon, erbium,
                                                    ytterbium, at_temperature,
                                                    multiphonon_lifetime)
from dynameta.optics.fiber_amp.waveguide import (V_MARCUSE_MAX, V_MARCUSE_MIN, FiberSpec,
                                                 cladding_pump_overlap, marcuse_validity,
                                                 mode_field_radius_m, overlap_gamma, v_number)
from dynameta.optics.fiber_amp.rare_earth import (ChannelSet, gain_coeff_per_m,
                                                  metastable_fraction)
from dynameta.optics.fiber_amp.steady_state import (AseBand, ChannelPlan, FiberAmplifier, Pump,
                                                   RamanStokes, Signal, SteadyStateResult)
from dynameta.optics.fiber_amp.noise import (AseSpectrum, NoiseResult, analyze_noise,
                                            local_inversion_factor, noise_figure,
                                            output_ase_spectrum)
from dynameta.optics.fiber_amp.metrics import (CompressionCurve, GainSpectrum, SlopeEfficiency,
                                              gain_compression_curve, gain_flatness,
                                              gain_spectrum, power_conversion_efficiency,
                                              saturation_output_power, slope_efficiency,
                                              stokes_limit, effective_pump_lambda_m)
from dynameta.optics.fiber_amp.concentration import (ConcentrationModel, erbium_upconversion,
                                                    ytterbium_photodarkening)
from dynameta.optics.fiber_amp.thermal import (ThermalModel, heat_load_per_m, net_forward_flux,
                                              peak_temperature_rise, quantum_defect_fraction,
                                              radial_temperature_rise, total_heat_W,
                                              thermal_lens_focal_power_per_m,
                                              thermal_guiding_onset_Q_per_m,
                                              thermo_optic_phase_rad,
                                              solve_with_thermal_feedback)
from dynameta.optics.fiber_amp.dynamics import (TransientResult, frantz_nodvik_gain,
                                               frantz_nodvik_instantaneous_gain,
                                               frantz_nodvik_output_energy, frantz_nodvik_pulse,
                                               saturation_energy, simulate_transient)
from dynameta.optics.fiber_amp.calibration import (CrossSectionTable, EDFA_CBAND_TARGETS,
                                                  calibration_report,
                                                  dB_per_m_to_per_m, giles_calibrated_fiber,
                                                  ion_from_cross_sections)
from dynameta.optics.fiber_amp.detection import BeatNoiseResult, detection_noise
from dynameta.optics.fiber_amp.pulse import (Pulse, gaussian_pulse, sech_pulse,
                                            dispersion_length, nonlinear_length, soliton_order,
                                            propagate_gnlse, raman_response_freq, SaturableGain)
from dynameta.optics.fiber_amp.cpa import (apply_spectral_phase, transform_limited, strehl_ratio,
                                          CPAResult, cpa_chain)
from dynameta.optics.fiber_amp.nonlinear_limits import (TMI_C0_DEFAULT, brillouin_shift_hz,
                                                        brillouin_linewidth_hz,
                                                        brillouin_phonon_number,
                                                        effective_length_m,
                                                        raman_gain_coefficient,
                                                        srs_stokes_wavelength_m,
                                                        sbs_threshold_W, sbs_gain_exponent,
                                                        srs_threshold_W, srs_gain_exponent,
                                                        tmi_threshold_W, rayleigh_alpha_per_m,
                                                        capture_fraction, double_rayleigh_mpi,
                                                        mpi_beat_variance_ratio, mpi_rin_per_hz,
                                                        mpi_power_penalty_dB)
from dynameta.optics.fiber_amp.eryb import ErYbAmplifier
from dynameta.optics.fiber_amp.lma import (LPMode, ModeOverlap, solve_lp_modes, dopant_overlap,
                                           cladding_absorption_two_population,
                                           effective_area_m2, marcuse_bend_loss_dB_per_m,
                                           marcuse_bend_loss_per_m, mode_degeneracy, mode_field,
                                           one_over_e_radius_m, pump_absorption_efficiency,
                                           second_moment_radius_m, total_mode_count,
                                           effective_cladding_overlap,
                                           mode_resolved_gain_overlaps)
from dynameta.optics.fiber_amp.transverse import (RadialGrid, ResolvedFiberAmplifier,
                                                  ResolvedResult, mean_field_equivalent,
                                                  saturation_correction_kappa,
                                                  tshb_closed_form_J, tshb_mean_field_J)
from dynameta.optics.fiber_amp.gain_bpm import (BPMResult, GainBPM, ThermalLoop,
                                                quadratic_duct_period_m, quadratic_duct_radius_m)
from dynameta.optics.fiber_amp.polarization import (TwoPolSaturation, f_from_pdg_slope,
                                                    pdg_cascade_db, pdg_db)
from dynameta.optics.fiber_amp.chain import (AmplifierChain, ChainResult, PassiveElement,
                                             StageRecord)
from dynameta.optics.fiber_amp.efficiency import (PumpSource, WallPlugBudget, energy_per_bit_J,
                                                  wall_plug_efficiency)
from dynameta.optics.fiber_amp.comms import (FEC_THRESHOLDS, LevelStatistics, LinkPerformance,
                                             PamFormat, gray_map, level_statistics,
                                             link_performance, max_pam_order, ml_thresholds,
                                             pam_levels_W, pam_ser, q_to_ser, ser_to_ber)

__all__ = ["CrossSectionModel", "RareEarthIon", "erbium", "ytterbium",
           "at_temperature", "multiphonon_lifetime",
           "FiberSpec", "overlap_gamma", "cladding_pump_overlap", "mode_field_radius_m",
           "ChannelSet", "metastable_fraction", "gain_coeff_per_m",
           "Pump", "Signal", "AseBand", "RamanStokes", "FiberAmplifier", "SteadyStateResult",
           "AseSpectrum", "NoiseResult", "output_ase_spectrum", "noise_figure",
           "local_inversion_factor", "analyze_noise",
           "CompressionCurve", "GainSpectrum", "SlopeEfficiency", "gain_compression_curve",
           "gain_flatness", "gain_spectrum", "power_conversion_efficiency",
           "saturation_output_power", "slope_efficiency", "stokes_limit",
           "ConcentrationModel", "erbium_upconversion", "ytterbium_photodarkening",
           "ThermalModel", "quantum_defect_fraction", "net_forward_flux", "heat_load_per_m",
           "total_heat_W", "radial_temperature_rise", "peak_temperature_rise",
           "thermal_lens_focal_power_per_m", "thermal_guiding_onset_Q_per_m",
           "thermo_optic_phase_rad", "solve_with_thermal_feedback",
           "TransientResult", "simulate_transient", "saturation_energy",
           "frantz_nodvik_output_energy", "frantz_nodvik_gain", "frantz_nodvik_pulse",
           "frantz_nodvik_instantaneous_gain",
           "CrossSectionTable", "ion_from_cross_sections", "giles_calibrated_fiber",
           "calibration_report", "dB_per_m_to_per_m", "EDFA_CBAND_TARGETS",
           "BeatNoiseResult", "detection_noise",
           "Pulse", "gaussian_pulse", "sech_pulse", "dispersion_length", "nonlinear_length",
           "soliton_order", "propagate_gnlse", "raman_response_freq", "SaturableGain",
           "apply_spectral_phase", "transform_limited", "strehl_ratio", "CPAResult", "cpa_chain",
           "brillouin_shift_hz", "brillouin_linewidth_hz", "sbs_threshold_W",
           "sbs_gain_exponent", "srs_threshold_W", "srs_gain_exponent", "tmi_threshold_W",
           "rayleigh_alpha_per_m", "capture_fraction", "double_rayleigh_mpi",
           "mpi_beat_variance_ratio", "mpi_rin_per_hz", "mpi_power_penalty_dB",
           "TMI_C0_DEFAULT", "brillouin_phonon_number", "effective_length_m",
           "raman_gain_coefficient", "srs_stokes_wavelength_m",
           "ErYbAmplifier",
           "solve_lp_modes", "dopant_overlap", "marcuse_bend_loss_per_m",
           "pump_absorption_efficiency", "effective_cladding_overlap",
           "mode_resolved_gain_overlaps",
           "LPMode", "ModeOverlap", "cladding_absorption_two_population", "effective_area_m2",
           "marcuse_bend_loss_dB_per_m", "mode_degeneracy", "mode_field", "one_over_e_radius_m",
           "second_moment_radius_m", "total_mode_count",
           "RadialGrid", "ResolvedFiberAmplifier", "ResolvedResult", "mean_field_equivalent",
           "saturation_correction_kappa", "tshb_closed_form_J", "tshb_mean_field_J",
           "GainBPM", "BPMResult", "ThermalLoop", "quadratic_duct_radius_m",
           "quadratic_duct_period_m",
           "TwoPolSaturation", "f_from_pdg_slope", "pdg_cascade_db", "pdg_db",
           "AmplifierChain", "ChainResult", "PassiveElement", "StageRecord",
           # v0.9.1 audit-2026-08-04 additions
           "ChannelPlan", "effective_pump_lambda_m",
           "v_number", "marcuse_validity", "V_MARCUSE_MIN", "V_MARCUSE_MAX",
           "PumpSource", "WallPlugBudget", "wall_plug_efficiency", "energy_per_bit_J",
           "PamFormat", "LevelStatistics", "LinkPerformance", "pam_levels_W", "gray_map",
           "ml_thresholds", "pam_ser", "ser_to_ber", "q_to_ser", "level_statistics",
           "link_performance", "max_pam_order", "FEC_THRESHOLDS"]
