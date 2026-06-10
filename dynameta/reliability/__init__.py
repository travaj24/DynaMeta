"""Reliability / degradation post-processors (docs/reliability_roadmap.md). Every model here is a
PURE-NUMPY/scipy POST-PROCESSOR on quantities the operating solve already produces (oxide E-field,
per-layer T, the Drude n->eps map) -- nothing imports the heavy solvers, nothing changes any existing
path (byte-identical-off: the package is opt-in by import, like carriers.thermal_fem).

Shipped (the no-new-driver MVP): REL1 gate-oxide TDDB (tddb), REL2 NBTI/PBTI bias-temperature
instability (bti), REL4 ITO thermal de-doping -> ENZ drift (dedoping), REL10 acceleration factors +
system-MTTF aggregation (mttf). Deferred pending new drivers: REL3 electromigration (contact current),
REL5 optical damage (per-region absorbed-power map), REL6/REL7 thermo-mechanical (CTE/stress schema),
REL8 HCI (substrate current), REL9 corrosion (ambient inputs).

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

__all__ = [
    "TddbParams", "tbd_e_model", "tbd_one_over_e", "weibull_area_scale",
    "oxide_stress_from_electrothermal",
    "BtiParams", "dvth_power_law", "time_to_dvth",
    "DedopingParams", "carrier_decay", "enz_wavelength_m", "enz_drift_m",
    "arrhenius_af", "field_af", "mttf_use_from_stress", "system_mttf", "fit_per_1e9_hours",
    "weibull_earliest_t63",
]
