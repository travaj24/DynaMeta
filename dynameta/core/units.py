"""
Length-unit conventions for the DEVSIM<->NGSolve bridge.

SI metres are the CANONICAL unit everywhere in the library's data model
(CarrierField, geometry, alignment bboxes). Each *solver* may use a
different length unit for its own coordinates -- DEVSIM works in SI metres,
the NGSolve/OCC geometry historically works in nanometres. A `UnitScale`
records that solver's unit so the bridge can convert without the hardcoded
`1e9` factors that used to live in stage3_optical/ngsolve_build.py and
eps_loader.py.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class UnitScale:
    """The length unit of a solver's coordinate system, in metres per unit.

    Examples:
      UnitScale(1.0)   -> coordinates are in SI metres   (DEVSIM)
      UnitScale(1e-9)  -> coordinates are in nanometres  (OCC/NGSolve)
    """
    metres_per_unit: float = 1.0

    def to_units(self, x_m: float) -> float:
        """Convert an SI-metre length to this solver's coordinate units."""
        return x_m / self.metres_per_unit

    def to_metres(self, x_units: float) -> float:
        """Convert a length in this solver's units back to SI metres."""
        return x_units * self.metres_per_unit


# Convenience singletons
SI = UnitScale(1.0)        # DEVSIM and the canonical data model
NM = UnitScale(1e-9)       # OCC / NGSolve geometry

# Exact m -> nm factor: the single source of the conversion the OCC/NGSolve geometry and the
# TMM d_list use. Kept as the exact literal 1e9 (NOT 1.0/NM.metres_per_unit, which drifts by
# ~1e-16 because 1e-9 is not exactly representable) so coordinates stay byte-identical to the
# old hardcoded `* 1e9`.
NM_PER_M = 1e9
