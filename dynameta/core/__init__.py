"""Bridge core: solver-agnostic spine connecting DEVSIM carriers to NGSolve optics.

Pure numpy/scipy + dataclasses; no devsim/ngsolve imports live here. The bridge
consumes a GeometryAlignment (the keystone identity/coordinate contract), a
CarrierField, an NToEpsMap, and a FieldLift -- not a Design or a mesh.

`core.polarization` is the repo's polarization-VOCABULARY map (audit V-8): which of the five
spellings of "which polarization" means what in which geometry, `normalize_pol` to convert
between them, and the shared error text every consumer module raises through. Stdlib-only.
It also owns ACCEPTANCE: `accept_pol` is the one boundary that normalizes the UNCONDITIONAL
aliases ('te'/'tm' and mixed case for the s/p family -- unification (b)), while the
geometry-DEPENDENT crossings stay strict and go through `normalize_pol` explicitly.
"""

from dynameta.core.units import UnitScale, SI, NM
from dynameta.core.backend import (
    array_namespace, backend_name, to_numpy, to_backend,
    is_numpy_array, is_cupy_array, is_jax_array, CUPY_AVAILABLE, JAX_AVAILABLE,
)
from dynameta.core.eps_field import EpsField
from dynameta.core.carrier_field import (
    CarrierField, CarrierRegion, dump_carrier_field, load_carrier_field,
    SCHEMA_VERSION, ELECTRON_DENSITY, POTENTIAL,
)
from dynameta.core.resample import resample_to_grid
from dynameta.core.lift import (
    FieldLift, IdentityLift, ExtrudeLift, SeparableXYLift, choose_lift,
)
from dynameta.core.alignment import RegionAlignment, GeometryAlignment
from dynameta.core.effects import (EffectModel, OpticalModelEffect, ComposedEffect, DeltaEffect,
                                   as_tensor, PockelsEffect, KerrEffect, FranzKeldyshEffect,
                                   ThermoOpticModel, ElectroAbsorptionModel, kramers_kronig_dn,
                                   PCMModel, LiquidCrystalModel, MagnetoOpticModel,
                                   AnisotropicThermoOpticModel, IntersubbandEffect, BursteinMossEdge,
                                   VectorMagnetoOpticModel)
from dynameta.core.n_to_eps import NToEpsMap, MaterialEpsMap, EffectEpsMap
from dynameta.core.interfaces import (
    RegionInfo, OpticalResult, CarrierSolver, OpticalGeometryBuilder, OpticalSolver,
    LayeredStackSolver,
)
from dynameta.core.layered import LayeredSlab, LayeredStack, slice_profile, slice_eps_field
from dynameta.core.bridge import assemble_eps
from dynameta.core.polarization import (
    VOCABULARIES, PolarizationConversionError, PolarizationVocabularyError, accept_pol,
    normalize_pol,
)

__all__ = [
    "UnitScale", "SI", "NM",
    "array_namespace", "backend_name", "to_numpy", "to_backend",
    "is_numpy_array", "is_cupy_array", "is_jax_array", "CUPY_AVAILABLE", "JAX_AVAILABLE",
    "EpsField",
    "CarrierField", "CarrierRegion", "dump_carrier_field", "load_carrier_field",
    "SCHEMA_VERSION", "ELECTRON_DENSITY", "POTENTIAL",
    "resample_to_grid",
    "FieldLift", "IdentityLift", "ExtrudeLift", "SeparableXYLift", "choose_lift",
    "RegionAlignment", "GeometryAlignment",
    "NToEpsMap", "MaterialEpsMap", "EffectEpsMap",
    "EffectModel", "OpticalModelEffect", "ComposedEffect", "DeltaEffect", "as_tensor",
    "PockelsEffect", "KerrEffect", "FranzKeldyshEffect", "ThermoOpticModel",
    "ElectroAbsorptionModel", "kramers_kronig_dn", "PCMModel", "LiquidCrystalModel",
    "MagnetoOpticModel", "AnisotropicThermoOpticModel", "IntersubbandEffect", "BursteinMossEdge",
    "VectorMagnetoOpticModel",
    "RegionInfo", "OpticalResult", "CarrierSolver", "OpticalGeometryBuilder", "OpticalSolver",
    "LayeredStackSolver",
    "LayeredSlab", "LayeredStack", "slice_profile", "slice_eps_field",
    "assemble_eps",
    "normalize_pol", "accept_pol", "VOCABULARIES", "PolarizationVocabularyError",
    "PolarizationConversionError",
]
