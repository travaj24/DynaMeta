"""
The pluggable seams. A user can supply their own DEVSIM device (CarrierSolver)
or their own NGSolve mesh (OpticalGeometryBuilder) and still use the Drude
bridge + sweep orchestration, as long as they satisfy these Protocols. The
library ships default layered implementations of each.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Protocol, Tuple, runtime_checkable

from dynameta.core.alignment import GeometryAlignment
from dynameta.core.carrier_field import CarrierField
from dynameta.core.eps_field import EpsField


@dataclass
class RegionInfo:
    """What a CarrierSolver advertises about each region it solves."""
    name:     str
    role:     str
    material: str
    bbox_m:   Tuple[float, float, float, float, float, float]
    ndim:     int = 2


@dataclass
class OpticalResult:
    """Outcome of one optical solve. t/T/A are None until Phase 3 transmission.

    A is the energy-budget closure 1 - R - T. A_independent (when not None) is the
    INDEPENDENTLY measured absorbed fraction -- the normalized volumetric loss
    integral k0 * Int Im(eps)|E|^2 dV / (cos(theta) * cell_area) over the physical
    (non-PML) domain. |A - A_independent| is then a genuine (non-tautological)
    energy/numerics diagnostic; they agree only if R, T and the in-structure field
    are all consistent."""
    r:             complex
    R:             float
    phase_deg:     float
    solve_time_s:  float
    t:             Optional[complex] = None
    T:             Optional[float] = None
    A:             Optional[float] = None
    A_independent: Optional[float] = None


@runtime_checkable
class CarrierSolver(Protocol):
    def regions(self) -> List[RegionInfo]: ...
    def solve(self, bias) -> CarrierField: ...


@runtime_checkable
class OpticalGeometryBuilder(Protocol):
    def build(self): ...                              # returns an opaque geometry handle
    def mesh_regions(self) -> List[str]: ...          # all subdomain material names
    def alignment(self) -> GeometryAlignment: ...     # the bridge contract


@runtime_checkable
class OpticalSolver(Protocol):
    def solve(self, geometry, eps_by_region: Dict[str, EpsField],
              lambda_m: float, optical) -> OpticalResult: ...


@runtime_checkable
class LayeredStackSolver(Protocol):
    """Solves a LayeredStack (Fourier-modal / TMM family) -- the seam a future RCWA backend
    and the present TMM oracle share. Distinct from OpticalSolver because it consumes the
    layered-slab representation (core.layered.LayeredStack), not the per-mesh-region voxel eps
    the FEM uses. Minimal by design; the RCWA adapter may widen it when ported."""
    def solve(self, stack, lambda_m: float, optical) -> OpticalResult: ...
