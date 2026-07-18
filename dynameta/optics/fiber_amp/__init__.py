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
  Phase 13 pulse (SaturableGain)  : saturable, spectrally-shaped gain -> gain narrowing.
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
"""

from dynameta.optics.fiber_amp.spectroscopy import (CrossSectionModel, RareEarthIon, erbium,
                                                    ytterbium, at_temperature,
                                                    multiphonon_lifetime)
from dynameta.optics.fiber_amp.waveguide import (FiberSpec, cladding_pump_overlap,
                                                 mode_field_radius_m, overlap_gamma)
from dynameta.optics.fiber_amp.rare_earth import (ChannelSet, gain_coeff_per_m,
                                                  metastable_fraction)
from dynameta.optics.fiber_amp.steady_state import (AseBand, FiberAmplifier, Pump, RamanStokes,
                                                   Signal, SteadyStateResult)
from dynameta.optics.fiber_amp.noise import (AseSpectrum, NoiseResult, analyze_noise,
                                            local_inversion_factor, noise_figure,
                                            output_ase_spectrum)
from dynameta.optics.fiber_amp.metrics import (CompressionCurve, GainSpectrum, SlopeEfficiency,
                                              gain_compression_curve, gain_flatness,
                                              gain_spectrum, power_conversion_efficiency,
                                              saturation_output_power, slope_efficiency,
                                              stokes_limit)
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
                                               frantz_nodvik_output_energy, frantz_nodvik_pulse,
                                               saturation_energy, simulate_transient)
from dynameta.optics.fiber_amp.calibration import (CrossSectionTable, calibration_report,
                                                  dB_per_m_to_per_m, giles_calibrated_fiber,
                                                  ion_from_cross_sections)
from dynameta.optics.fiber_amp.detection import BeatNoiseResult, detection_noise
from dynameta.optics.fiber_amp.pulse import (Pulse, gaussian_pulse, sech_pulse,
                                            dispersion_length, nonlinear_length, soliton_order,
                                            propagate_gnlse, raman_response_freq, SaturableGain)
from dynameta.optics.fiber_amp.cpa import (apply_spectral_phase, transform_limited, strehl_ratio,
                                          CPAResult, cpa_chain)
from dynameta.optics.fiber_amp.nonlinear_limits import (brillouin_shift_hz,
                                                        brillouin_linewidth_hz,
                                                        sbs_threshold_W, sbs_gain_exponent,
                                                        srs_threshold_W, srs_gain_exponent,
                                                        tmi_threshold_W, rayleigh_alpha_per_m,
                                                        capture_fraction, double_rayleigh_mpi,
                                                        mpi_beat_variance_ratio, mpi_rin_per_hz,
                                                        mpi_power_penalty_dB)
from dynameta.optics.fiber_amp.eryb import ErYbAmplifier
from dynameta.optics.fiber_amp.lma import (solve_lp_modes, dopant_overlap,
                                           marcuse_bend_loss_per_m, pump_absorption_efficiency,
                                           effective_cladding_overlap,
                                           mode_resolved_gain_overlaps)
from dynameta.optics.fiber_amp.polarization import (TwoPolSaturation, f_from_pdg_slope,
                                                    pdg_cascade_db, pdg_db)
from dynameta.optics.fiber_amp.chain import AmplifierChain, ChainResult, PassiveElement

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
           "CrossSectionTable", "ion_from_cross_sections", "giles_calibrated_fiber",
           "calibration_report", "dB_per_m_to_per_m",
           "BeatNoiseResult", "detection_noise",
           "Pulse", "gaussian_pulse", "sech_pulse", "dispersion_length", "nonlinear_length",
           "soliton_order", "propagate_gnlse", "raman_response_freq", "SaturableGain",
           "apply_spectral_phase", "transform_limited", "strehl_ratio", "CPAResult", "cpa_chain",
           "brillouin_shift_hz", "brillouin_linewidth_hz", "sbs_threshold_W",
           "sbs_gain_exponent", "srs_threshold_W", "srs_gain_exponent", "tmi_threshold_W",
           "rayleigh_alpha_per_m", "capture_fraction", "double_rayleigh_mpi",
           "mpi_beat_variance_ratio", "mpi_rin_per_hz", "mpi_power_penalty_dB",
           "ErYbAmplifier",
           "solve_lp_modes", "dopant_overlap", "marcuse_bend_loss_per_m",
           "pump_absorption_efficiency", "effective_cladding_overlap",
           "mode_resolved_gain_overlaps",
           "TwoPolSaturation", "f_from_pdg_slope", "pdg_cascade_db", "pdg_db",
           "AmplifierChain", "ChainResult", "PassiveElement"]
