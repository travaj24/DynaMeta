"""
Electrode: a DEVSIM contact attached to a layer. Its lateral footprint is
either a CrossSection (e.g. the patch footprint), an edge selector (a thin
strip on a cell edge, for peripheral grounds), or "full" (the whole layer
face). The DEVSIM builder resolves the footprint to a contact bbox.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Union

from dynameta.geometry.cross_section import CrossSection


EdgeSelector = Literal["x_lo", "x_hi", "y_lo", "y_hi"]
ElectrodeRole = Literal["biased", "ground"]
Footprint = Union[CrossSection, str]   # CrossSection | EdgeSelector | "full"


@dataclass
class Electrode:
    name:             str
    layer:            str                 # attached layer name
    footprint:        Footprint           # CrossSection | "x_lo"/"x_hi"/"y_lo"/"y_hi" | "full"
    role:             ElectrodeRole = "biased"
    fixed_voltage_V:  float = 0.0         # used when role == "ground"

    def __post_init__(self) -> None:
        if isinstance(self.footprint, str):
            if self.footprint not in ("x_lo", "x_hi", "y_lo", "y_hi", "full"):
                raise ValueError(
                    "Electrode '{}' string footprint must be one of "
                    "x_lo/x_hi/y_lo/y_hi/full (got {!r})".format(
                        self.name, self.footprint))
        elif not isinstance(self.footprint, CrossSection):
            raise ValueError(
                "Electrode '{}' footprint must be a CrossSection or a "
                "selector string".format(self.name))
        if self.role not in ("biased", "ground"):
            raise ValueError("Electrode role must be 'biased' or 'ground'")

    @property
    def is_edge(self) -> bool:
        return isinstance(self.footprint, str) and self.footprint != "full"
