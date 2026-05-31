"""
The vertical device structure: a Stack of Layers (bottom-to-top) between a
semi-infinite superstrate and substrate.

A Layer = a background material that fills the cell + a list of Inclusions,
each an (CrossSection shape, material, priority). This single model expresses:
  - uniform films  : background only, no inclusions (mirror, oxide, ITO)
  - patches/pillars : background air + one metal/dielectric inclusion
  - hole arrays     : background metal + an air (or dielectric) inclusion
  - gratings        : background + Rectangle-stripe inclusion(s)
  - dimers/multi-element : several inclusions (resolved by descending priority)

Physics ROLE is NOT stored on the Layer -- it is derived per material from the
MaterialRegistry (metal / semiconductor / dielectric), so a layer can host
regions of mixed role. A Feature spans multiple layers' z (vias, T-patches).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from dynameta.geometry.cross_section import CrossSection


@dataclass
class Inclusion:
    shape:    CrossSection
    material: str
    priority: int = 0          # higher priority wins where inclusions overlap


@dataclass
class Layer:
    name:                str
    thickness_m:         float
    background_material: str
    inclusions:          List[Inclusion] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.thickness_m <= 0:
            raise ValueError("Layer '{}' thickness must be positive".format(self.name))

    def materials_used(self) -> List[str]:
        out = [self.background_material]
        out += [inc.material for inc in self.inclusions]
        return out

    def has_inclusions(self) -> bool:
        return len(self.inclusions) > 0


@dataclass
class Feature:
    """A solid spanning an explicit z-range that may cross layer boundaries
    (via, T-shaped patch stem+cap). Resolved after layers are laid down.
    Forward-looking: the default builders gain full Feature support in a
    later phase; the data model supports it now."""
    name:        str
    shape:       CrossSection
    material:    str
    z_lo_m:      float
    z_hi_m:      float
    priority:    int = 10       # features sit above layer inclusions by default


@dataclass
class Stack:
    layers:                List[Layer]            # bottom-to-top
    superstrate_material:  str                    # semi-infinite medium above the stack
    substrate_material:    str                    # semi-infinite medium below the stack
    features:              List[Feature] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.layers:
            raise ValueError("Stack requires at least one layer")
        names = [L.name for L in self.layers]
        if len(set(names)) != len(names):
            raise ValueError("Duplicate layer names: {}".format(names))

    def z_intervals(self) -> Dict[str, Tuple[float, float]]:
        """{layer_name: (z_lo_m, z_hi_m)}, accumulating thickness bottom-to-top
        with the stack base at z = 0."""
        z = 0.0
        out: Dict[str, Tuple[float, float]] = {}
        for L in self.layers:
            out[L.name] = (z, z + L.thickness_m)
            z += L.thickness_m
        return out

    def total_thickness_m(self) -> float:
        return sum(L.thickness_m for L in self.layers)

    def find_layer(self, name: str) -> Layer:
        for L in self.layers:
            if L.name == name:
                return L
        raise KeyError("No layer named '{}'".format(name))
