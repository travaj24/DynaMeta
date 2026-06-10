"""dynameta.drivers: the glue layer between simulation results and downstream models.

Two seams, both pure numpy/scipy (importable without DEVSIM/NGSolve):

- reliability_glue: result objects (CarrierField, OpticalResult, ElectroThermalResult) ->
  reliability-model inputs (J for electromigration, E_ox/T for TDDB, absorbed fractions for
  LIDT), with the unit conventions pinned at the adapter.
- state_glue: per-bias material-state solvers (LLG macrospin, PCM kinetics, LC director BVP)
  -> run_pipeline extra_fields closures producing exactly the key their partner EffectModel
  reads ('m_vector', 'crystalline_fraction', 'director_angle_rad').

electrothermal_extra_fields (the Joule-heating T closure) lives with its solver in
dynameta.carriers.electrothermal and is re-exported here for discoverability when NGSolve is
installed -- importing THIS package does not require it.
"""

from dynameta.drivers.reliability_glue import (absorbed_fraction, contact_current_A,
                                               contact_current_density_from_field,
                                               cw_damage_threshold_from_stack,
                                               em_mttf_from_carrier_field,
                                               oxide_stress_from_electrothermal,
                                               tddb_tbd_from_electrothermal,
                                               tmm_absorption_by_layer_name)
from dynameta.drivers.state_glue import lc_extra_fields, llg_extra_fields, pcm_extra_fields

__all__ = [
    "absorbed_fraction", "contact_current_A", "contact_current_density_from_field",
    "cw_damage_threshold_from_stack", "em_mttf_from_carrier_field",
    "oxide_stress_from_electrothermal", "tddb_tbd_from_electrothermal",
    "tmm_absorption_by_layer_name",
    "lc_extra_fields", "llg_extra_fields", "pcm_extra_fields",
]
