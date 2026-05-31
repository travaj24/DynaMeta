"""
Layer: one horizontal layer of the device stack.

A device's electrical and optical structure is described as a vertical
stack of Layers, each with a name, thickness, material reference, and
LateralExtent that tells the geometry builder where it lives in x, y.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple


# ---------------------------------------------------------------------------
# Lateral extent
# ---------------------------------------------------------------------------

LateralExtentKind = Literal["full_period", "patch_footprint", "rectangle"]


@dataclass
class LateralExtent:
    """Where a layer exists in the (x, y) plane of the unit cell.

    Three kinds:
      full_period       : the layer spans [0, P] x [0, P]. Used for
                          stack-wide layers (mirror, oxides, ITO, ...).
      patch_footprint   : the layer is confined to the patch's lateral
                          footprint (size = patch_side_m, centered).
                          Used for adhesion + patch metal.
      rectangle         : explicit (xl, xh, yl, yh) bounding box in m.
                          Used for arbitrary metasurface elements.
    """
    kind: LateralExtentKind = "full_period"
    # Only used when kind == "rectangle":
    xlo_m: Optional[float] = None
    xhi_m: Optional[float] = None
    ylo_m: Optional[float] = None
    yhi_m: Optional[float] = None

    def __post_init__(self) -> None:
        if self.kind == "rectangle":
            for fld in ("xlo_m", "xhi_m", "ylo_m", "yhi_m"):
                if getattr(self, fld) is None:
                    raise ValueError(
                        "LateralExtent(kind='rectangle') requires "
                        "all of xlo_m, xhi_m, ylo_m, yhi_m")

    def bbox(self, period_m: float, patch_side_m: float
              ) -> Tuple[float, float, float, float]:
        """Resolve to a concrete (xlo, xhi, ylo, yhi) for this device."""
        if self.kind == "full_period":
            return (0.0, period_m, 0.0, period_m)
        if self.kind == "patch_footprint":
            lo = (period_m - patch_side_m) / 2.0
            hi = lo + patch_side_m
            return (lo, hi, lo, hi)
        return (self.xlo_m, self.xhi_m, self.ylo_m, self.yhi_m)


# Convenience constructors
def full_period() -> LateralExtent:
    return LateralExtent(kind="full_period")


def patch_footprint() -> LateralExtent:
    return LateralExtent(kind="patch_footprint")


def rectangle(xlo_m: float, xhi_m: float,
                ylo_m: float, yhi_m: float) -> LateralExtent:
    return LateralExtent(kind="rectangle", xlo_m=xlo_m, xhi_m=xhi_m,
                           ylo_m=ylo_m, yhi_m=yhi_m)


# ---------------------------------------------------------------------------
# Layer
# ---------------------------------------------------------------------------

LayerRole = Literal["metal", "dielectric", "semiconductor"]


@dataclass
class Layer:
    """One horizontal layer of the device stack.

    Stacked bottom-to-top in the Design's `layers` list. The Layer's
    z extent is computed by accumulating thicknesses in stack order.

    Args:
      name             : unique identifier, used as the DEVSIM region
                          name and the NGSolve subdomain name
      thickness_m      : layer thickness in metres (vertical extent)
      material         : Material.name string (must exist in the Design's
                          MaterialRegistry)
      role             : "metal" | "dielectric" | "semiconductor".
                          - metal: no Poisson eq, equipotential contact
                            host
                          - dielectric: Poisson eq, no carriers
                          - semiconductor: full drift-diffusion + Drude
      lateral_extent   : where in (x, y) this layer exists. Default
                          full_period().
      eps_static_override : if non-None, override Material.drude.eps_static
                              (or Material's static eps for dielectrics).
                              Use for materials whose static and optical
                              eps differ in ways the Material doesn't
                              encode.
    """
    name:                 str
    thickness_m:          float
    material:             str
    role:                 LayerRole
    lateral_extent:       LateralExtent = None     # type: ignore[assignment]
    eps_static_override:  Optional[float] = None

    def __post_init__(self) -> None:
        if self.lateral_extent is None:
            self.lateral_extent = full_period()
        if self.thickness_m <= 0:
            raise ValueError("Layer '{}' has non-positive thickness {}"
                              .format(self.name, self.thickness_m))
        if self.role not in ("metal", "dielectric", "semiconductor"):
            raise ValueError("Layer '{}' role must be metal|dielectric|"
                              "semiconductor (got {!r})".format(
                                  self.name, self.role))
