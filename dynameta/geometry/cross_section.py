"""
CrossSection: the lateral (x, y) shape of an inclusion within a layer.

These are pure geometry (SI metres, absolute coordinates in the unit cell);
they carry NO solver code. The DEVSIM builder reads `x_edges_m`/`y_edges_m`
to place mesh-refinement lines; the NGSolve builder reads `kind` + the shape
parameters to construct the OCC face; both can use `contains_m` to rasterize.

A layer's full-cell fill is its BACKGROUND material -- it is NOT a CrossSection.
Inclusions are always sub-cell shapes placed on top of the background. So a
patch = background air + one Rectangle/Circle inclusion of metal; a hole array
= background metal + a Circle inclusion of air; a grating = Rectangle stripes.

`intrinsic_symmetry()` (about the shape's own center) gates the carrier-field
lift: SeparableXYLift requires the device to be 'c4v' (a centered, 4-fold
shape in a square cell).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np


class CrossSection:
    """Base class for lateral inclusion shapes. Coordinates in SI metres."""
    kind: str = "abstract"

    def center_m(self) -> Tuple[float, float]:
        raise NotImplementedError

    def bbox_m(self) -> Tuple[float, float, float, float]:
        """(xlo, xhi, ylo, yhi) lateral bounding box in metres."""
        raise NotImplementedError

    def intrinsic_symmetry(self) -> str:
        """Coarse point-group tag about the shape's own center:
        'c4v' (4-fold), 'c2v' (2 mirror axes), or 'none'."""
        return "none"

    def x_edges_m(self) -> List[float]:
        xlo, xhi, _, _ = self.bbox_m()
        return [xlo, xhi]

    def y_edges_m(self) -> List[float]:
        _, _, ylo, yhi = self.bbox_m()
        return [ylo, yhi]

    def contains_m(self, x, y):
        """Boolean (array-broadcasting) point-in-shape test."""
        raise NotImplementedError


@dataclass
class Rectangle(CrossSection):
    cx_m: float
    cy_m: float
    width_m: float
    height_m: float
    kind: str = field(default="rectangle", init=False)

    def center_m(self):
        return (self.cx_m, self.cy_m)

    def bbox_m(self):
        return (self.cx_m - self.width_m / 2, self.cx_m + self.width_m / 2,
                self.cy_m - self.height_m / 2, self.cy_m + self.height_m / 2)

    def intrinsic_symmetry(self):
        return "c4v" if abs(self.width_m - self.height_m) < 1e-15 * self.width_m else "c2v"

    def contains_m(self, x, y):
        x = np.asarray(x); y = np.asarray(y)
        return ((np.abs(x - self.cx_m) <= self.width_m / 2) &
                (np.abs(y - self.cy_m) <= self.height_m / 2))


@dataclass
class Circle(CrossSection):
    cx_m: float
    cy_m: float
    radius_m: float
    kind: str = field(default="circle", init=False)

    def center_m(self):
        return (self.cx_m, self.cy_m)

    def bbox_m(self):
        r = self.radius_m
        return (self.cx_m - r, self.cx_m + r, self.cy_m - r, self.cy_m + r)

    def intrinsic_symmetry(self):
        return "c4v"   # a circle is c-inf-v, a superset of c4v

    def contains_m(self, x, y):
        x = np.asarray(x); y = np.asarray(y)
        return (x - self.cx_m) ** 2 + (y - self.cy_m) ** 2 <= self.radius_m ** 2


@dataclass
class Ellipse(CrossSection):
    cx_m: float
    cy_m: float
    rx_m: float
    ry_m: float
    kind: str = field(default="ellipse", init=False)

    def center_m(self):
        return (self.cx_m, self.cy_m)

    def bbox_m(self):
        return (self.cx_m - self.rx_m, self.cx_m + self.rx_m,
                self.cy_m - self.ry_m, self.cy_m + self.ry_m)

    def intrinsic_symmetry(self):
        return "c4v" if abs(self.rx_m - self.ry_m) < 1e-15 * self.rx_m else "c2v"

    def contains_m(self, x, y):
        x = np.asarray(x); y = np.asarray(y)
        return (((x - self.cx_m) / self.rx_m) ** 2
                + ((y - self.cy_m) / self.ry_m) ** 2) <= 1.0


@dataclass
class RegularPolygon(CrossSection):
    cx_m: float
    cy_m: float
    radius_m: float          # circumradius
    n_sides: int
    rotation_deg: float = 0.0
    kind: str = field(default="regular_polygon", init=False)

    def vertices_m(self) -> np.ndarray:
        a0 = math.radians(self.rotation_deg)
        ang = a0 + np.arange(self.n_sides) * (2 * math.pi / self.n_sides)
        return np.column_stack([self.cx_m + self.radius_m * np.cos(ang),
                                 self.cy_m + self.radius_m * np.sin(ang)])

    def center_m(self):
        return (self.cx_m, self.cy_m)

    def bbox_m(self):
        v = self.vertices_m()
        return (v[:, 0].min(), v[:, 0].max(), v[:, 1].min(), v[:, 1].max())

    def intrinsic_symmetry(self):
        if self.n_sides % 4 == 0:
            return "c4v"
        return "c2v" if self.n_sides % 2 == 0 else "none"

    def contains_m(self, x, y):
        return _polygon_contains(self.vertices_m(), x, y)


@dataclass
class Polygon(CrossSection):
    points_m: List[Tuple[float, float]]
    kind: str = field(default="polygon", init=False)

    def vertices_m(self) -> np.ndarray:
        return np.asarray(self.points_m, dtype=np.float64)

    def center_m(self):
        v = self.vertices_m()
        return (float(v[:, 0].mean()), float(v[:, 1].mean()))

    def bbox_m(self):
        v = self.vertices_m()
        return (v[:, 0].min(), v[:, 0].max(), v[:, 1].min(), v[:, 1].max())

    def intrinsic_symmetry(self):
        return "none"   # conservative; arbitrary polygons aren't auto-classified

    def contains_m(self, x, y):
        return _polygon_contains(self.vertices_m(), x, y)


def _polygon_contains(verts: np.ndarray, x, y):
    """Ray-cast point-in-polygon (broadcasting over array x, y)."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    inside = np.zeros(np.broadcast(x, y).shape, dtype=bool)
    n = len(verts)
    j = n - 1
    for i in range(n):
        xi, yi = verts[i]; xj, yj = verts[j]
        cond = ((yi > y) != (yj > y)) & (
            x < (xj - xi) * (y - yi) / (yj - yi + 1e-300) + xi)
        inside ^= cond
        j = i
    return inside


# ---- centered-in-cell convenience constructors -------------------------------

def centered_rectangle(cell, width_m: float, height_m: float) -> Rectangle:
    cx, cy = cell.center_m
    return Rectangle(cx, cy, width_m, height_m)


def centered_square(cell, side_m: float) -> Rectangle:
    return centered_rectangle(cell, side_m, side_m)


def centered_circle(cell, radius_m: float) -> Circle:
    cx, cy = cell.center_m
    return Circle(cx, cy, radius_m)
