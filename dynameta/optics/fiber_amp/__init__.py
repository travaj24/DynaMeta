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

The imports below are populated as each phase lands.
"""

from dynameta.optics.fiber_amp.spectroscopy import (CrossSectionModel, RareEarthIon, erbium,
                                                    ytterbium, at_temperature,
                                                    multiphonon_lifetime)
from dynameta.optics.fiber_amp.waveguide import (FiberSpec, cladding_pump_overlap,
                                                 mode_field_radius_m, overlap_gamma)
from dynameta.optics.fiber_amp.rare_earth import (ChannelSet, ase_source_per_m, dP_dz,
                                                  gain_coeff_per_m, metastable_fraction)
from dynameta.optics.fiber_amp.steady_state import (AseBand, FiberAmplifier, Pump, Signal,
                                                   SteadyStateResult)
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
                                              radial_temperature_rise, total_heat_W)
from dynameta.optics.fiber_amp.dynamics import (TransientResult, frantz_nodvik_gain,
                                               frantz_nodvik_output_energy, frantz_nodvik_pulse,
                                               saturation_energy, simulate_transient)
from dynameta.optics.fiber_amp.calibration import (CrossSectionTable, calibration_report,
                                                  dB_per_m_to_per_m, giles_calibrated_fiber,
                                                  ion_from_cross_sections)

__all__ = ["CrossSectionModel", "RareEarthIon", "erbium", "ytterbium",
           "at_temperature", "multiphonon_lifetime",
           "FiberSpec", "overlap_gamma", "cladding_pump_overlap", "mode_field_radius_m",
           "ChannelSet", "metastable_fraction", "gain_coeff_per_m", "ase_source_per_m", "dP_dz",
           "Pump", "Signal", "AseBand", "FiberAmplifier", "SteadyStateResult",
           "AseSpectrum", "NoiseResult", "output_ase_spectrum", "noise_figure",
           "local_inversion_factor", "analyze_noise",
           "CompressionCurve", "GainSpectrum", "SlopeEfficiency", "gain_compression_curve",
           "gain_flatness", "gain_spectrum", "power_conversion_efficiency",
           "saturation_output_power", "slope_efficiency", "stokes_limit",
           "ConcentrationModel", "erbium_upconversion", "ytterbium_photodarkening",
           "ThermalModel", "quantum_defect_fraction", "net_forward_flux", "heat_load_per_m",
           "total_heat_W", "radial_temperature_rise", "peak_temperature_rise",
           "TransientResult", "simulate_transient", "saturation_energy",
           "frantz_nodvik_output_energy", "frantz_nodvik_gain", "frantz_nodvik_pulse",
           "CrossSectionTable", "ion_from_cross_sections", "giles_calibrated_fiber",
           "calibration_report", "dB_per_m_to_per_m"]
