"""
Material: an optical dispersion model + optional transport model, referenced
by name from the geometry Stack. The single source of truth for what a named
solid means to every stage.

A material is a semiconductor iff it carries a TransportModel (Stage 1 then
solves carriers in it and Stage 2/3 use its OpticalModel with the local n).
Metals/dielectrics have only an OpticalModel.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from dynameta.materials.optical_model import OpticalModel
from dynameta.materials.transport_model import TransportModel


@dataclass
class Material:
    name:         str
    optical:      OpticalModel
    transport:    Optional[TransportModel] = None
    is_metal:     bool = False
    pretty_name:  str = ""
    eps_static_dc: Optional[float] = None   # DC relative permittivity for Stage-1
                                            # Poisson on a (non-semiconductor)
                                            # dielectric. REQUIRED for gate oxides:
                                            # their OPTICAL eps (e.g. HfO2 ~4) is
                                            # NOT the DC value (HfO2 ~18-25), and
                                            # the gate capacitance -> accumulation
                                            # depends on the DC value. Semiconductors
                                            # carry their DC eps on TransportModel.

    def __post_init__(self) -> None:
        if not self.pretty_name:
            self.pretty_name = self.name

    @property
    def is_semiconductor(self) -> bool:
        return self.transport is not None

    def dc_permittivity(self) -> Optional[float]:
        """Static relative permittivity for Stage-1 Poisson, or None if unset."""
        if self.transport is not None:
            return self.transport.eps_static
        return self.eps_static_dc

    def eps(self, lambda_m: float, *, n_m3=None):
        """Optical eps at a wavelength (and, for free-carrier models, density)."""
        return self.optical.eps(lambda_m, n_m3=n_m3)


class MaterialRegistry:
    """Name -> Material. A Design carries its own registry so different
    designs can reuse names without colliding."""

    def __init__(self) -> None:
        self._materials: Dict[str, Material] = {}

    def add(self, material: Material) -> "MaterialRegistry":
        if material.name in self._materials:
            raise ValueError("Material '{}' already registered".format(material.name))
        self._materials[material.name] = material
        return self

    def get(self, name: str) -> Material:
        if name not in self._materials:
            raise KeyError("Material '{}' not in registry. Known: {}".format(
                name, sorted(self._materials)))
        return self._materials[name]

    def __contains__(self, name: str) -> bool:
        return name in self._materials

    def names(self) -> List[str]:
        return sorted(self._materials)

    def __len__(self) -> int:
        return len(self._materials)
