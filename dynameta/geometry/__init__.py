"""Geometry: unit cell, cross-sections, stack/inclusions, electrodes, specs, Design."""

from dynameta.geometry.unit_cell import UnitCell
from dynameta.geometry.cross_section import (
    CrossSection, Rectangle, Circle, Ellipse, RegularPolygon, Polygon,
    centered_rectangle, centered_square, centered_circle,
)
from dynameta.geometry.stack import Inclusion, Layer, Feature, Stack, Taper
from dynameta.geometry.guards import (UnsupportedGeometryError, require_supported_geometry,
                                      require_untapered_layer, tapered_inclusions)
from dynameta.geometry.electrode import Electrode
from dynameta.geometry.specs import CarrierSpec, Mesh2DSpec, Mesh3DSpec, OpticalSpec
from dynameta.geometry.design import Design

__all__ = [
    "UnitCell",
    "CrossSection", "Rectangle", "Circle", "Ellipse", "RegularPolygon", "Polygon",
    "centered_rectangle", "centered_square", "centered_circle",
    "Inclusion", "Layer", "Feature", "Stack", "Taper",
    "UnsupportedGeometryError", "require_supported_geometry", "require_untapered_layer",
    "tapered_inclusions",
    "Electrode",
    "CarrierSpec", "Mesh2DSpec", "Mesh3DSpec", "OpticalSpec",
    "Design",
]
