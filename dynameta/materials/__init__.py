"""Materials: optical dispersion + transport models, and a registry."""

from dynameta.materials.optical_model import (
    OpticalModel, ConstantOptical, TabulatedOptical, DrudeOptical,
    RefractiveIndexInfoOptical, fit_drude_params, M_E, Q_E, EPS0, C_LIGHT,
)
from dynameta.materials.transport_model import (
    TransportModel, TrapSpec, CarrierPhysics,
)
from dynameta.materials.material import Material, MaterialRegistry
from dynameta.materials.db import (
    DielectricDB, DielectricRecord, normalize_formula,
)

__all__ = [
    "OpticalModel", "ConstantOptical", "TabulatedOptical", "DrudeOptical",
    "RefractiveIndexInfoOptical", "fit_drude_params", "M_E", "Q_E", "EPS0", "C_LIGHT",
    "TransportModel", "TrapSpec", "CarrierPhysics",
    "Material", "MaterialRegistry",
    "DielectricDB", "DielectricRecord", "normalize_formula",
]
