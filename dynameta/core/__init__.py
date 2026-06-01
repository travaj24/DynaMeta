"""Bridge core: solver-agnostic spine connecting DEVSIM carriers to NGSolve optics.

Pure numpy/scipy + dataclasses; no devsim/ngsolve imports live here. The bridge
consumes a GeometryAlignment (the keystone identity/coordinate contract), a
CarrierField, an NToEpsMap, and a FieldLift -- not a Design or a mesh.
"""

from dynameta.core.units import UnitScale, SI, NM
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
from dynameta.core.n_to_eps import NToEpsMap, MaterialEpsMap
from dynameta.core.interfaces import (
    RegionInfo, OpticalResult, CarrierSolver, OpticalGeometryBuilder, OpticalSolver,
    LayeredStackSolver,
)
from dynameta.core.layered import LayeredSlab, LayeredStack, slice_profile, slice_eps_field
from dynameta.core.bridge import assemble_eps

__all__ = [
    "UnitScale", "SI", "NM",
    "EpsField",
    "CarrierField", "CarrierRegion", "dump_carrier_field", "load_carrier_field",
    "SCHEMA_VERSION", "ELECTRON_DENSITY", "POTENTIAL",
    "resample_to_grid",
    "FieldLift", "IdentityLift", "ExtrudeLift", "SeparableXYLift", "choose_lift",
    "RegionAlignment", "GeometryAlignment",
    "NToEpsMap", "MaterialEpsMap",
    "RegionInfo", "OpticalResult", "CarrierSolver", "OpticalGeometryBuilder", "OpticalSolver",
    "LayeredStackSolver",
    "LayeredSlab", "LayeredStack", "slice_profile", "slice_eps_field",
    "assemble_eps",
]
