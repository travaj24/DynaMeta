"""
NToEpsMap: carrier density n (and wavelength) -> complex eps. The Drude/optical
formula itself lives in materials/optical_model.py; this is the thin Protocol +
a default implementation that looks materials up in a MaterialRegistry. A
bring-your-own user can supply any object with eps_grid/scalar_eps.

Background eps comes out correctly for free: a region at n = n_bg maps to its
background eps through the SAME formula (no grid-corner proxy).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from dynameta.materials.material import MaterialRegistry


@runtime_checkable
class NToEpsMap(Protocol):
    def eps_grid(self, material_name: str, n_m3: np.ndarray,
                  lambda_m: float) -> np.ndarray: ...
    def scalar_eps(self, material_name: str, lambda_m: float) -> complex: ...


@dataclass
class MaterialEpsMap:
    """Default NToEpsMap: dispatch to each Material's OpticalModel."""
    materials: MaterialRegistry

    def eps_grid(self, material_name, n_m3, lambda_m):
        mat = self.materials.get(material_name)
        return np.asarray(mat.eps(lambda_m, n_m3=n_m3), dtype=np.complex128)

    def scalar_eps(self, material_name, lambda_m):
        return complex(self.materials.get(material_name).eps(lambda_m))
