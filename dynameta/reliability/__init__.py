"""Reliability / degradation post-processors (docs/reliability_roadmap.md). Every model here is a
PURE-NUMPY/scipy POST-PROCESSOR on quantities the operating solve already produces (oxide E-field,
per-layer T, the Drude n->eps map) -- nothing imports the heavy solvers, nothing changes any existing
path (byte-identical-off: the package is opt-in by import, like carriers.thermal_fem).

The FULL REL1-REL10 set is shipped: REL1 TDDB (tddb), REL2 NBTI/PBTI (bti), REL3 electromigration
(em -- drive current from carriers.contact_current via drivers.reliability_glue, or external),
REL4 ITO de-doping -> ENZ drift (dedoping), REL5 optical damage / LIDT + CW thermal runaway (lidt --
a lumped thermal node + absorbed(T) callable; OpticalResult.per_region_absorption is the shipped
driver via drivers.reliability_glue), REL6 thermal-cycling fatigue (fatigue -- ductile Coffin-Manson
vs brittle Weibull; MechanicalProps lives on the MATERIAL schema, materials/mechanical.py, and is
re-exported here), REL7 stress/thermal-gradient migration (stress_migration -- Korhonen PDE + Soret
flux), REL8 HCI (hci -- I_sub from carriers.impact_ionization or external),
REL9 corrosion/oxidation/humidity (corrosion -- Deal-Grove + Peck; ambient is external), and
REL10 acceleration factors + system MTTF (mttf).

SI units; cp1252/ASCII; every model ships a reduces-to-closed-form gate AND an independent reference
gate (validation/reliability_*.py).
"""

from dynameta.reliability.tddb import (TddbParams, tbd_e_model, tbd_one_over_e, weibull_area_scale,
                                       oxide_stress_from_electrothermal)
from dynameta.reliability.bti import BtiParams, dvth_power_law, time_to_dvth
from dynameta.reliability.dedoping import (DedopingParams, carrier_decay, enz_wavelength_m,
                                           enz_drift_m)
from dynameta.reliability.mttf import (arrhenius_af, field_af, mttf_use_from_stress, system_mttf,
                                       fit_per_1e9_hours, weibull_earliest_t63)
from dynameta.reliability.em import (EmParams, black_mttf_s, blech_immortal, current_density_A_m2,
                                     miner_time_to_failure_s)
from dynameta.reliability.lidt import (ThermalNode, lidt_fluence_J_m2, cw_steady_temperature_K,
                                       cw_critical_intensity_W_m2, cw_transient_K,
                                       stack_absorbed_of_T)
from dynameta.reliability.fatigue import (MechanicalProps, biaxial_stress_Pa, coffin_manson_nf,
                                          plastic_strain_range, norris_landzberg_af,
                                          brittle_survival, cycles_to_failure)
from dynameta.reliability.stress_migration import (korhonen_kappa_m2_s, korhonen_relax,
                                                   void_nucleates, soret_flux_per_m2_s)
from dynameta.reliability.hci import trap_generation_rate_per_m2_s, hci_time_to_failure_s
from dynameta.reliability.corrosion import (deal_grove_thickness_m, deal_grove_rate_arrhenius,
                                            peck_time_to_failure_s, peck_af)
from dynameta.reliability.leakage import (OxideLeakageParams, fn_coefficients,
                                          fowler_nordheim_current, direct_tunneling_current)

__all__ = [
    "TddbParams", "tbd_e_model", "tbd_one_over_e", "weibull_area_scale",
    "oxide_stress_from_electrothermal",
    "BtiParams", "dvth_power_law", "time_to_dvth",
    "DedopingParams", "carrier_decay", "enz_wavelength_m", "enz_drift_m",
    "arrhenius_af", "field_af", "mttf_use_from_stress", "system_mttf", "fit_per_1e9_hours",
    "weibull_earliest_t63",
    "EmParams", "black_mttf_s", "blech_immortal", "current_density_A_m2", "miner_time_to_failure_s",
    "ThermalNode", "lidt_fluence_J_m2", "cw_steady_temperature_K", "cw_critical_intensity_W_m2",
    "cw_transient_K", "stack_absorbed_of_T",
    "MechanicalProps", "biaxial_stress_Pa", "coffin_manson_nf", "plastic_strain_range",
    "norris_landzberg_af", "brittle_survival", "cycles_to_failure",
    "korhonen_kappa_m2_s", "korhonen_relax", "void_nucleates", "soret_flux_per_m2_s",
    "trap_generation_rate_per_m2_s", "hci_time_to_failure_s",
    "deal_grove_thickness_m", "deal_grove_rate_arrhenius", "peck_time_to_failure_s", "peck_af",
    "OxideLeakageParams", "fn_coefficients", "fowler_nordheim_current", "direct_tunneling_current",
]
