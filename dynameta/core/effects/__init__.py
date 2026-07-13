"""
EffectModel: the generalized material-response seam (the v0.3 keystone).

Where the original NToEpsMap mapped ONLY carrier density n -> a scalar eps, an EffectModel maps
the full local-field bundle {n, E, T, ...} -> eps, which may be a TENSOR (3x3 per point) for
anisotropic effects (Pockels, liquid-crystal, ...). A caller assembles the per-region field bundle
and calls EffectModel.eps(fields, lambda). (Today the bridge auto-assembles only the carrier field
'n'; the field-effect drivers for {E, T} produce their fields for the caller to place in the
bundle -- wiring them through the bridge is a tracked seam. The richer effects are validated
end-to-end at the FEM level by the Phase-1/2 oracles.)

  eps(fields, lambda_m) -> ndarray
      scalar response: shape (...,)            (broadcast of the field grids)
      tensor response: shape (..., 3, 3)       (per-point 3x3 permittivity)

The default `OpticalModelEffect` adapts the existing scalar OpticalModels (Drude / Constant /
Tabulated): it reads fields['n'] (None for a density-independent model) and ignores the rest, so
the free-carrier / ENZ path is unchanged. Richer effects -- a PockelsEffect reading fields['E'],
a ThermoOpticEffect reading fields['T'], ... -- implement the SAME interface and can be COMPOSED
(see `ComposedEffect`: a background response + summed delta-eps contributions).

Pure numpy: no devsim/ngsolve. Convention: exp(-i omega t), Im(eps) > 0 for absorbers.
"""

from __future__ import annotations

from dynameta.core.effects.base import (ComposedEffect, DeltaEffect, EffectModel,
                                        OpticalModelEffect, _E_vec, _photon_energy_J,
                                        _voigt6_to_full, as_tensor, kramers_kronig_dn,
                                        kramers_kronig_dn_rows)
from dynameta.core.effects.electro import (FranzKeldyshEffect, KerrEffect,
                                           PockelsEffect)
from dynameta.core.effects.thermo import (AnisotropicThermoOpticModel,
                                          ThermoOpticModel)
from dynameta.core.effects.electroabsorption import (BursteinMossEdge,
                                                     ElectroAbsorptionModel,
                                                     IntersubbandEffect)
from dynameta.core.effects.reconfigurable import LiquidCrystalModel, PCMModel
from dynameta.core.effects.magneto import (MagnetoOpticModel,
                                           VectorMagnetoOpticModel)
