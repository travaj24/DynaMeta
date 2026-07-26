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

    def __post_init__(self) -> None:
        if not (self.z_lo_m < self.z_hi_m):       # Layer validates thickness > 0; Feature must validate its span
            raise ValueError("Feature '{}' requires z_lo_m < z_hi_m (got {:.4g}, {:.4g})".format(
                self.name, self.z_lo_m, self.z_hi_m))


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
        # audit R-10: Features were validated only for z_lo < z_hi. Two identically-named
        # features lying ENTIRELY outside [0, total_thickness] were accepted, and combined with
        # the builder's own silence (F-14) the user got no signal at either layer -- the feature
        # simply never appeared in the mesh. Layer names already fail loudly on both counts;
        # apply the same policy to their z-spanning sibling.
        if self.features:
            fnames = [f.name for f in self.features]
            fdupes = sorted({n for n in fnames if fnames.count(n) > 1})
            if fdupes:
                raise ValueError(
                    "Stack: duplicate Feature names {} -- feature names key the geometry and "
                    "per-region diagnostics exactly as layer names do.".format(fdupes))
            total = self.total_thickness_m()
            # relative slack: a feature snapped to the stack top by float arithmetic must not trip
            tol = 1e-9 * max(total, 1.0)
            outside = [(f.name, f.z_lo_m, f.z_hi_m) for f in self.features
                       if f.z_hi_m <= tol or f.z_lo_m >= total - tol]
            if outside:
                raise ValueError(
                    "Stack: Feature(s) {} lie entirely outside the stack z-range [0, {:.6g}] m "
                    "and would silently never be built. Move them inside, or grow the "
                    "stack.".format(outside, total))

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
