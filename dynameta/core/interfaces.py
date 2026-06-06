"""
The pluggable seams. A user can supply their own DEVSIM device (CarrierSolver)
or their own NGSolve mesh (OpticalGeometryBuilder) and still use the Drude
bridge + sweep orchestration, as long as they satisfy these Protocols. The
library ships default layered implementations of each.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Protocol, Tuple, runtime_checkable, TYPE_CHECKING

from dynameta.core.alignment import GeometryAlignment
from dynameta.core.carrier_field import CarrierField
from dynameta.core.eps_field import EpsField

if TYPE_CHECKING:                  # type-only annotations (no runtime core->sweep/geometry dep)
    from dynameta.sweep import BiasPoint
    from dynameta.geometry.specs import OpticalSpec


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
    # Fit-INDEPENDENT R/T from the time-averaged z-Poynting flux of the reconstructed total field
    # (Sz = 0.5 Re(Ex Hy* - Ey Hx*)); these read energy straight from the field, bypassing the
    # up/down least-squares amplitude fit. For a clean propagating 0-order they agree with R/T; a
    # large gap flags an extraction problem -- the flux measure captures the FULL transmitted power
    # (both polarizations), so it is the trustworthy R/T for an off-diagonal / gyrotropic tensor
    # whose transmitted field is elliptical (the single-projection fit cannot see the cross-pol).
    R_flux:        Optional[float] = None
    T_flux:        Optional[float] = None


@runtime_checkable
class CarrierSolver(Protocol):
    """A carrier solver the pipeline drives. `solve` is the workhorse; `regions` is an OPTIONAL
    introspection hook (what the solver advertises about each region) -- the default pipeline
    builds its alignment from the OPTICAL builder, so `regions()` is not on the hot path, but a
    BYO solver may implement it for tooling/inspection."""
    def regions(self) -> List[RegionInfo]: ...
    def solve(self, bias: "BiasPoint") -> CarrierField: ...


@runtime_checkable
class OpticalGeometryBuilder(Protocol):
    def build(self): ...                              # returns an opaque geometry handle
    def mesh_regions(self) -> List[str]: ...          # all subdomain material names
    def alignment(self) -> GeometryAlignment: ...     # the bridge contract


@runtime_checkable
class OpticalSolver(Protocol):
    def solve(self, geometry, eps_by_region: Dict[str, EpsField],
              lambda_m: float, optical: "OpticalSpec") -> OpticalResult: ...


@runtime_checkable
class LayeredStackSolver(Protocol):
    """Solves a LayeredStack (Fourier-modal / TMM family) -- the seam a future RCWA backend
    and the present TMM oracle share. Distinct from OpticalSolver because it consumes the
    layered-slab representation (core.layered.LayeredStack), not the per-mesh-region voxel eps
    the FEM uses. Minimal by design; the RCWA adapter may widen it when ported."""
    def solve(self, stack, lambda_m: float, optical: "OpticalSpec") -> OpticalResult: ...
