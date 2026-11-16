"""Materials: optical dispersion + transport models, and a registry."""

from dynameta.materials.optical_model import (
    OpticalModel, ConstantOptical, TabulatedOptical, DrudeOptical,
    ExtendedDrudeOptical, MerminDrudeOptical, gamma_ito_extended, check_kk,
    RefractiveIndexInfoOptical, fit_drude_params, M_E, Q_E, EPS0, C_LIGHT,
    BandFillingBackground, TabulatedBackground, STACK_AXIS_FIELD, FILM_THICKNESS_FIELD,
)
from dynameta.materials.scattering import (
    KaneOpticalMass, MatthiessenGamma, ScatteringModel, BoundaryScattering, fuchs_sondheimer_factor,
    fs_effective_rate,
)
from dynameta.materials.mechanical import MechanicalProps
from dynameta.materials.transport_model import (
    TransportModel, TrapSpec, CarrierPhysics, KaneBand,
)
from dynameta.materials.band_temperature import (
    BandTemperature, BandData, band_data_at_T, band_gap_ratio, algaas_gap_law, ingaasp_lm_gap_law,
    varshni_eg_ev,
)
from dynameta.materials.material import Material, MaterialRegistry
from dynameta.materials.db import (
    DielectricDB, DielectricRecord, normalize_formula,
)

__all__ = [
    "OpticalModel", "ConstantOptical", "TabulatedOptical", "DrudeOptical",
    "ExtendedDrudeOptical", "MerminDrudeOptical", "gamma_ito_extended", "check_kk",
    "RefractiveIndexInfoOptical", "fit_drude_params", "M_E", "Q_E", "EPS0", "C_LIGHT",
    "BandFillingBackground", "TabulatedBackground", "STACK_AXIS_FIELD", "FILM_THICKNESS_FIELD",
    "KaneOpticalMass", "MatthiessenGamma", "ScatteringModel", "MechanicalProps",
    "BoundaryScattering", "fuchs_sondheimer_factor", "fs_effective_rate",
    "TransportModel", "TrapSpec", "CarrierPhysics", "KaneBand",
    "BandTemperature", "BandData", "band_data_at_T", "band_gap_ratio", "algaas_gap_law", "ingaasp_lm_gap_law",
    "varshni_eg_ev",
    "Material", "MaterialRegistry",
    "DielectricDB", "DielectricRecord", "normalize_formula",
]

# The ENZ material presets (materials.tco_presets, materials.enz_presets), the oxide scattering kernels
# (materials.oxide_scattering), the lattice / oscillator models (materials.lattice_optical) and the uniaxial
# composition (materials.uniaxial_optical) are re-exported lazily (PEP 562, as dynameta.carriers does), so
# that `import dynameta.materials` does not import them; dir() and `import *` list them.
_LAZY_NAMES = {
    "azo": "tco_presets", "gzo": "tco_presets", "in_cdo": "tco_presets", "ito": "tco_presets",
    "dual_gate_film_kwargs": "tco_presets", "wurtzite_polarization_charges": "tco_presets",
    "IonizedImpurityScattering": "oxide_scattering", "MasettiMobility": "oxide_scattering",
    "SetoGrainBarrier": "oxide_scattering", "ResolvedMatthiessenGamma": "oxide_scattering",
    "PolycrystallineScattering": "oxide_scattering", "OpticalMobilityDamping": "oxide_scattering",
    "TabulatedMobility": "oxide_scattering", "UniaxialOptical": "uniaxial_optical",
    "fto": "tco_presets",
    "PolarLatticeOptical": "lattice_optical", "LorentzOscillatorOptical": "lattice_optical",
    "GaussianOscillatorOptical": "lattice_optical", "with_free_carriers": "lattice_optical",
    "tin": "enz_presets", "zrn": "enz_presets", "inas": "enz_presets", "inassb": "enz_presets",
    "sic": "enz_presets", "aln": "enz_presets", "sio2": "enz_presets", "FixedDensityOptical": "enz_presets",
}
__all__ += list(_LAZY_NAMES)


def __getattr__(name):
    if name in _LAZY_NAMES:
        import importlib
        return getattr(importlib.import_module("{}.{}".format(__name__, _LAZY_NAMES[name])), name)
    raise AttributeError("module {!r} has no attribute {!r}".format(__name__, name))


def __dir__():
    return sorted(set(globals()) | set(_LAZY_NAMES))
