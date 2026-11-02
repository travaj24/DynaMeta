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

    # audit X-14 (2026-11-02): these two had ZERO callers while core/bridge.py re-typed the same
    # division inline (`x / alignment.unit_scale.metres_per_unit`). The bridge now calls to_units
    # (the identical operation, so the EpsField axes are bit-identical), and both directions are
    # gated in tests/test_audit_2026_07_25_conventions.py. Scalars and numpy arrays alike.
    def to_units(self, x_m):
        """Convert an SI-metre length (scalar or array) to this solver's coordinate units."""
        return x_m / self.metres_per_unit

    def to_metres(self, x_units):
        """Convert a length (scalar or array) in this solver's units back to SI metres."""
        return x_units * self.metres_per_unit


# Convenience singletons
SI = UnitScale(1.0)        # DEVSIM and the canonical data model
NM = UnitScale(1e-9)       # OCC / NGSolve geometry

# Exact m -> nm factor: the single source of the conversion the OCC/NGSolve geometry and the
# TMM d_list use. Kept as the exact literal 1e9 (NOT 1.0/NM.metres_per_unit, which drifts by
# ~1e-16 because 1e-9 is not exactly representable) so coordinates stay byte-identical to the
# old hardcoded `* 1e9`. This is the ONE deliberate exception to "convert through UnitScale"
# (audit X-14): NM.to_units(x) = x / 1e-9 and x * NM_PER_M differ in the last bit for some x
# (measured 1.0/1e-9 = 999999999.9999999), and the mesh builders' coordinates are pinned
# bit-for-bit, so they multiply by this exact factor instead of dividing through NM.
NM_PER_M = 1e9
