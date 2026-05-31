"""
EpsField: the bridge's per-region optical-permittivity output. Either a
uniform scalar (fixed-eps materials) or a 3D complex grid on axes expressed
in the TARGET solver's length units (e.g. nm for NGSolve). Solver-agnostic:
the optics adapter turns a uniform EpsField into a constant CoefficientFunction
and a gridded one into a VoxelCoefficient.

`values_zyx` is stored in (Nz, Ny, Nx) order -- the order NGSolve's
VoxelCoefficient expects -- so the adapter needs no further transpose.

`time_convention` records the sign convention of Im(eps); the whole library
uses exp(-i*omega*t) (passive loss => Im(eps) > 0), matching NGSolve, so no
buried sign flips (the old eps_loader negated eps_im at load time).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class EpsField:
    scalar:          Optional[complex] = None
    x_axis_u:        Optional[np.ndarray] = None      # target units
    y_axis_u:        Optional[np.ndarray] = None
    z_axis_u:        Optional[np.ndarray] = None
    values_zyx:      Optional[np.ndarray] = None       # complex (Nz, Ny, Nx)
    time_convention: str = "exp(-iwt)"

    @property
    def is_uniform(self) -> bool:
        return self.scalar is not None

    def voxel_bounds_u(self) -> Tuple[Tuple[float, float, float],
                                        Tuple[float, float, float]]:
        """(start, end) in (x, y, z) target units for a VoxelCoefficient."""
        if self.is_uniform:
            raise ValueError("uniform EpsField has no voxel bounds")
        start = (float(self.x_axis_u[0]),  float(self.y_axis_u[0]),  float(self.z_axis_u[0]))
        end   = (float(self.x_axis_u[-1]), float(self.y_axis_u[-1]), float(self.z_axis_u[-1]))
        return start, end
