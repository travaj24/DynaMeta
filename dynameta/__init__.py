"""
dynameta -- a general bridge connecting DEVSIM (DC carrier transport) to
NGSolve (3D FEM optics) through a Drude n->eps map, for tunable metasurface
modulators of arbitrary geometry.

Architecture (clean-break; v0.4 adds the modulation-mechanism EffectModel family --
Pockels/Kerr/FK/thermo-optic/QCSE/PCM/LC/graphene/magneto-optic -- the off-diagonal
anisotropic FEM (UPML), and the graphene surface-current BC):
  core/       solver-agnostic bridge spine (CarrierField, EpsField, alignment,
              n->eps + EffectModel response, field lift, bridge) -- pure numpy, no devsim/ngsolve
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

Other modules (import explicitly): dynameta.optics.fdtd / fdtd_nd / fdtd_mo / fdtd_seam
(broadband/dispersive/MO/oblique FDTD + the sweep-aware seam), dynameta.optics.tmm_reference
(coherent-TMM oracle), dynameta.optics.inverse_design + topology_opt (differentiable JAX-FDTD
design), dynameta.transient_optics (coupled carrier->optics transient), dynameta.results
(SweepResults + HDF5/Zarr save/load), dynameta.cache (persistent optical-solver cache),
dynameta.viz (matplotlib spectra/maps), dynameta.carriers.lc_director / lc_dynamics
(nematic LC director statics + Erickson-Leslie dynamics).
"""

__version__ = "0.4.0"

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
    KaneOpticalMass, MatthiessenGamma,
)
from dynameta.sweep import Sweep, BiasPoint
# Bridge core (data + alignment + bridge; still no devsim/ngsolve)
from dynameta.core import (
    CarrierField, EpsField, GeometryAlignment, RegionAlignment,
    MaterialEpsMap, EffectEpsMap, assemble_eps, choose_lift, UnitScale, SI, NM,
)
# Modulation-mechanism EffectModel family (v0.3+) -- the field/temperature/state -> eps response maps
from dynameta.core import (
    EffectModel, as_tensor, ComposedEffect, DeltaEffect, OpticalModelEffect,
    PockelsEffect, KerrEffect, FranzKeldyshEffect, ThermoOpticModel,
    AnisotropicThermoOpticModel, ElectroAbsorptionModel, PCMModel, LiquidCrystalModel,
    MagnetoOpticModel, IntersubbandEffect, BursteinMossEdge,
)
from dynameta.analysis import (
    resonance_dip, resonance_shift, gate_cv, sheet_resistance_ohm_sq,
    lumped_rc_bandwidth, switching_energy_per_area, modulator_figure_of_merit,
)

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
    "KaneOpticalMass", "MatthiessenGamma",
    # sweep
    "Sweep", "BiasPoint",
    # bridge core
    "CarrierField", "EpsField", "GeometryAlignment", "RegionAlignment",
    "MaterialEpsMap", "EffectEpsMap", "assemble_eps", "choose_lift", "UnitScale", "SI", "NM",
    # modulation-mechanism effect models
    "EffectModel", "as_tensor", "ComposedEffect", "DeltaEffect", "OpticalModelEffect",
    "PockelsEffect", "KerrEffect", "FranzKeldyshEffect", "ThermoOpticModel",
    "AnisotropicThermoOpticModel", "ElectroAbsorptionModel", "PCMModel", "LiquidCrystalModel",
    "MagnetoOpticModel", "IntersubbandEffect", "BursteinMossEdge",
    # analysis (resonance + modulator figure-of-merit)
    "resonance_dip", "resonance_shift", "gate_cv", "sheet_resistance_ohm_sq",
    "lumped_rc_bandwidth", "switching_energy_per_area", "modulator_figure_of_merit",
]
