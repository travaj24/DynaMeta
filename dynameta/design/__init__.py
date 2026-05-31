"""Design-time data model: Material, Layer, Electrode, mesh + optical
specs, and the top-level Design.
"""

from dynameta.design.design import Design
from dynameta.design.electrode import (
    Electrode,
    ElectrodeLocation,
    ElectrodeRole,
)
from dynameta.design.layer import (
    Layer,
    LateralExtent,
    LayerRole,
    full_period,
    patch_footprint,
    rectangle,
)
from dynameta.design.materials import (
    Material,
    MaterialRegistry,
    DrudeSpec,
    TrapSpec,
)
from dynameta.design.mesh_spec import Mesh2DSpec, Mesh3DSpec
from dynameta.design.optical_spec import OpticalSpec, Polarization, LinearSolver
from dynameta.design.sweep import BiasPoint, Sweep

__all__ = [
    "Design",
    "Electrode", "ElectrodeLocation", "ElectrodeRole",
    "Layer", "LateralExtent", "LayerRole",
    "full_period", "patch_footprint", "rectangle",
    "Material", "MaterialRegistry", "DrudeSpec", "TrapSpec",
    "Mesh2DSpec", "Mesh3DSpec",
    "OpticalSpec", "Polarization", "LinearSolver",
    "BiasPoint", "Sweep",
]
