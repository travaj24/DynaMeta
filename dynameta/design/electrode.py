"""
Electrode: a DEVSIM contact at a specific location, with a role (biased
or ground) that voltage sweeps reference.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple


ElectrodeRole = Literal["biased", "ground"]
ElectrodeLocation = Literal[
    "full_lateral",      # full x and y of the attached_layer
    "patch_footprint",   # only the patch footprint of the attached_layer
    "x_lo_edge",         # x = 0, full y      (thin strip)
    "x_hi_edge",         # x = period, full y
    "y_lo_edge",
    "y_hi_edge",
    "rectangle",         # explicit bbox
]


@dataclass
class Electrode:
    """A DEVSIM contact attached to a specific layer at a specific
    lateral location.

    Args:
      name              : DEVSIM contact name (also referenced by
                            BiasPoint.voltages keys)
      attached_layer    : name of the Layer the contact sits on
                            (the metal layer for a real electrode,
                             or the semiconductor itself for an ITO ground)
      role              : "biased"  -- voltage settable per BiasPoint
                          "ground"  -- fixed at 0 V (or override)
      location          : where on the attached_layer the contact lives
                            (see ElectrodeLocation)
      bbox_m            : (xlo, xhi, ylo, yhi) -- only used when
                            location == "rectangle"
      fixed_voltage_V   : if role == "ground", this is the fixed voltage
                            (default 0). Ignored if role == "biased".
      pretty_label      : display label for plots / logs
    """
    name:              str
    attached_layer:    str
    role:              ElectrodeRole = "biased"
    location:          ElectrodeLocation = "full_lateral"
    bbox_m:            Optional[Tuple[float, float, float, float]] = None
    fixed_voltage_V:   float = 0.0
    pretty_label:      str = ""

    def __post_init__(self) -> None:
        if not self.pretty_label:
            self.pretty_label = self.name
        if self.location == "rectangle" and self.bbox_m is None:
            raise ValueError(
                "Electrode '{}' uses location='rectangle' but bbox_m is "
                "None".format(self.name))

    def resolve_bbox(self, period_m: float, patch_side_m: float,
                       attached_layer_z_interval: Tuple[float, float]
                       ) -> Tuple[float, float, float, float, float, float]:
        """Compute the (xlo, xhi, ylo, yhi, zlo, zhi) bounding box of
        this electrode in metres. The z-interval is the attached layer's
        z range; the location maps the x/y bounds.
        """
        z_lo, z_hi = attached_layer_z_interval
        p = period_m
        eps_xy = 1e-10
        if self.location == "full_lateral":
            return (0.0, p, 0.0, p, z_lo, z_hi)
        if self.location == "patch_footprint":
            lo = (p - patch_side_m) / 2.0
            hi = lo + patch_side_m
            return (lo, hi, lo, hi, z_lo, z_hi)
        if self.location == "x_lo_edge":
            return (0.0 - eps_xy, 0.0 + eps_xy, 0.0, p, z_lo, z_hi)
        if self.location == "x_hi_edge":
            return (p - eps_xy, p + eps_xy, 0.0, p, z_lo, z_hi)
        if self.location == "y_lo_edge":
            return (0.0, p, 0.0 - eps_xy, 0.0 + eps_xy, z_lo, z_hi)
        if self.location == "y_hi_edge":
            return (0.0, p, p - eps_xy, p + eps_xy, z_lo, z_hi)
        if self.location == "rectangle":
            xlo, xhi, ylo, yhi = self.bbox_m   # type: ignore[misc]
            return (xlo, xhi, ylo, yhi, z_lo, z_hi)
        raise ValueError("Unknown electrode location: {!r}".format(self.location))
