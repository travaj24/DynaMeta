"""
NToEpsMap: the per-region RESPONSE map -- a local-field bundle (a `fields` dict
{n, E, T, ...}) + wavelength -> complex eps. The default dispatches to each Material's
response (an EffectModel from core.effects); today that is OpticalModelEffect(material), the
scalar Drude/optical formula (materials/optical_model.py) reading fields['n']. A field-
dependent or tensor effect (Pockels, ...) plugs in at the SAME seam. A bring-your-own user can
supply any object with eps_grid(material, fields, lambda) / scalar_eps(material, lambda).

Background eps comes out correctly for free: a region at n = n_bg maps to its background eps
through the SAME formula (no grid-corner proxy).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from dynameta.materials.material import MaterialRegistry
from dynameta.core.effects import OpticalModelEffect


@runtime_checkable
class NToEpsMap(Protocol):
    def eps_grid(self, material_name: str, fields: dict,
                  lambda_m: float) -> np.ndarray: ...
    def scalar_eps(self, material_name: str, lambda_m: float) -> complex: ...


@dataclass
class MaterialEpsMap:
    """Default response map: dispatch to each Material's response via OpticalModelEffect (the
    EffectModel adapter that reads fields['n']). A future field-dependent / tensor effect
    (Pockels, ...) attaches at this seam."""
    materials: MaterialRegistry

    def eps_grid(self, material_name, fields, lambda_m):
        mat = self.materials.get(material_name)
        return np.asarray(OpticalModelEffect(mat).eps(fields, lambda_m), dtype=np.complex128)

    def scalar_eps(self, material_name, lambda_m):
        return complex(self.materials.get(material_name).eps(lambda_m))
