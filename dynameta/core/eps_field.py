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
buried sign flips (the old eps_loader negated eps_im at load time). It is
ENFORCED, not merely recorded -- see `require_solver_time_convention`, which
every optical backend calls on the `eps_by_region` seam it is handed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from dynameta.constants import SOLVER_TIME_CONVENTION


@dataclass
class EpsField:
    scalar:          Optional[complex] = None
    x_axis_u:        Optional[np.ndarray] = None      # target units
    y_axis_u:        Optional[np.ndarray] = None
    z_axis_u:        Optional[np.ndarray] = None
    values_zyx:      Optional[np.ndarray] = None       # complex (Nz,Ny,Nx) scalar OR (Nz,Ny,Nx,3,3) tensor
    tensor:          Optional[np.ndarray] = None        # uniform anisotropic 3x3 (complex)
    time_convention: str = SOLVER_TIME_CONVENTION

    @property
    def is_uniform(self) -> bool:
        return self.scalar is not None or self.tensor is not None

    @property
    def is_tensor(self) -> bool:
        """True if this region's eps is a 3x3 TENSOR (a uniform `tensor`, or a graded
        `values_zyx` with a trailing (3,3) -> ndim 5) rather than a scalar."""
        return self.tensor is not None or (
            self.values_zyx is not None and np.asarray(self.values_zyx).ndim == 5)

    def voxel_bounds_u(self) -> Tuple[Tuple[float, float, float],
                                        Tuple[float, float, float]]:
        """(start, end) in (x, y, z) target units for a VoxelCoefficient."""
        if self.is_uniform:
            raise ValueError("uniform EpsField has no voxel bounds")
        start = (float(self.x_axis_u[0]),  float(self.y_axis_u[0]),  float(self.z_axis_u[0]))
        end   = (float(self.x_axis_u[-1]), float(self.y_axis_u[-1]), float(self.z_axis_u[-1]))
        return start, end


def require_solver_time_convention(eps_by_region, where: str) -> None:
    """Validate the `time_convention` label of every `EpsField` on the BYO optical seam
    (finding V-5).

    The bridge enforced the convention on its INPUT (`CarrierField`, `core/bridge.py`) and then
    dropped it on its OUTPUT: `EpsField.time_convention` was written by four bridge sites and
    read by nobody (`grep` over `optics/`, `cache.py` and `io/` returned zero hits), while the
    docstring's promise that the field "records the sign convention" reads as a guarantee that
    something checks it. Every optical backend takes `eps_by_region: Dict[str, EpsField]` (the
    documented pluggable seam, `core/interfaces.py`), so a caller assembling `EpsField`s
    themselves -- the whole point of that seam -- could hand any backend an exp(+i omega t)
    permittivity and get a silently AMPLIFYING solve.

    Cheap: one string compare per region, no array touch. Entries that are not `EpsField`-shaped
    (no `time_convention` attribute) pass through -- the seam is duck-typed and this guard is
    about a WRONG label, not a missing one.
    """
    for name, ef in (eps_by_region or {}).items():
        conv = getattr(ef, "time_convention", SOLVER_TIME_CONVENTION)
        if conv != SOLVER_TIME_CONVENTION:
            raise ValueError(
                "{}: eps_by_region[{!r}].time_convention is {!r}, but the whole library -- every "
                "solver, every material model, every validation oracle -- is {!r} (passive loss "
                "=> Im(eps) > 0). Solving an exp(+i omega t) permittivity field under the "
                "exp(-i omega t) engines turns absorption into GAIN silently. Conjugate the "
                "field and relabel it, or build the EpsField through dynameta.core.bridge, "
                "which already enforces the convention on its input.".format(
                    where, name, conv, SOLVER_TIME_CONVENTION))
