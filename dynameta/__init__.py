"""
dynameta -- a general bridge connecting DEVSIM (DC carrier transport) to
NGSolve (3D FEM optics) through a Drude n->eps map, for tunable metasurface
modulators of arbitrary geometry.

Architecture (clean-break v0.2):
  core/       solver-agnostic bridge spine (CarrierField, EpsField, alignment,
              n->eps, field lift, bridge) -- pure numpy, no devsim/ngsolve
  geometry/   declarative device: UnitCell + Stack(Layer = background +
              Inclusions) + Electrodes + Design
  materials/  OpticalModel (Drude/Constant/Tabulated) + TransportModel
  carriers/   default Stage-1 DEVSIM builder (CarrierSolver)
  optics/     default Stage-3 NGSolve builder + solver (OpticalGeometryBuilder,
              OpticalSolver)

Quick start:
    from dynameta import Design, UnitCell, Stack, Layer, Material, ...
    from dynameta.pipeline import run_pipeline   # imports devsim + ngsolve
    rows = run_pipeline(design, sweep)

The heavy solver pipeline is NOT imported at package top level (so `import
dynameta` stays light); import it explicitly from dynameta.pipeline.
Advanced users can supply their own CarrierSolver / OpticalGeometryBuilder /
OpticalSolver (see dynameta.core.interfaces).
"""

__version__ = "0.2.0"

# Geometry + materials data model (lightweight)
from dynameta.geometry import (
    Design, UnitCell, Stack, Layer, Inclusion, Feature, Electrode,
    CrossSection, Rectangle, Circle, Ellipse, RegularPolygon, Polygon,
    centered_rectangle, centered_square, centered_circle,
    Mesh2DSpec, Mesh3DSpec, OpticalSpec,
)
from dynameta.materials import (
    Material, MaterialRegistry, OpticalModel, ConstantOptical, TabulatedOptical,
    DrudeOptical, TransportModel, TrapSpec, fit_drude_params,
)
from dynameta.sweep import Sweep, BiasPoint
# Bridge core (data + alignment + bridge; still no devsim/ngsolve)
from dynameta.core import (
    CarrierField, EpsField, GeometryAlignment, RegionAlignment,
    MaterialEpsMap, assemble_eps, choose_lift, UnitScale, SI, NM,
)
from dynameta.analysis import resonance_dip, resonance_shift

__all__ = [
    "__version__",
    # geometry
    "Design", "UnitCell", "Stack", "Layer", "Inclusion", "Feature", "Electrode",
    "CrossSection", "Rectangle", "Circle", "Ellipse", "RegularPolygon", "Polygon",
    "centered_rectangle", "centered_square", "centered_circle",
    "Mesh2DSpec", "Mesh3DSpec", "OpticalSpec",
    # materials
    "Material", "MaterialRegistry", "OpticalModel", "ConstantOptical",
    "TabulatedOptical", "DrudeOptical", "TransportModel", "TrapSpec", "fit_drude_params",
    # sweep
    "Sweep", "BiasPoint",
    # bridge core
    "CarrierField", "EpsField", "GeometryAlignment", "RegionAlignment",
    "MaterialEpsMap", "assemble_eps", "choose_lift", "UnitScale", "SI", "NM",
    # analysis
    "resonance_dip", "resonance_shift",
]
