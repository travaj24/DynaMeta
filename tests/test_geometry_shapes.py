"""Fast (pure-numpy) guards for the inclusion CrossSection shapes: degenerate constructor
inputs must RAISE rather than build silent-garbage OCC solids (audit GEO-2), the DEVSIM
rasterizer contains_m is winding-insensitive, and the signed-shoelace winding the NGSolve
_polygon_prism uses to normalize a Polygon to CCW (the GEO-1 fix) is sign-correct. No
ngsolve/OCC needed. Run: python -m pytest tests/test_geometry_shapes.py -q
"""
import numpy as np
import pytest

from dynameta.geometry.cross_section import (Rectangle, Circle, Ellipse,
                                             RegularPolygon, Polygon)


def test_degenerate_shapes_raise():
    with pytest.raises(ValueError):
        Rectangle(0.0, 0.0, 0.0, 1e-7)                       # zero width
    with pytest.raises(ValueError):
        Circle(0.0, 0.0, -1e-7)                              # negative radius
    with pytest.raises(ValueError):
        Ellipse(0.0, 0.0, 1e-7, 0.0)                         # zero ry
    with pytest.raises(ValueError):
        RegularPolygon(0.0, 0.0, 1e-7, 2)                    # < 3 sides
    with pytest.raises(ValueError):
        Polygon([(0.0, 0.0), (1e-7, 0.0)])                   # < 3 vertices
    with pytest.raises(ValueError):
        Polygon([(0.0, 0.0), (1e-7, 0.0), (2e-7, 0.0)])      # collinear -> zero area


def test_valid_shapes_construct():
    Rectangle(0.0, 0.0, 1e-7, 2e-7)
    Circle(0.0, 0.0, 1e-7)
    Ellipse(0.0, 0.0, 1e-7, 5e-8)
    RegularPolygon(0.0, 0.0, 1e-7, 6)
    Polygon([(0, 0), (1e-7, 0), (1e-7, 1e-7), (0, 1e-7)])    # CCW square
    Polygon([(0, 0), (0, 1e-7), (1e-7, 1e-7), (1e-7, 0)])    # CW square (also valid)


def test_polygon_contains_is_winding_insensitive():
    # the DEVSIM rasterizer must agree for CW and CCW windings of the same shape
    ccw = Polygon([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)])
    cw = Polygon([(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0)])
    x = np.array([0.5, 0.5, -0.5]); y = np.array([0.5, 1.5, 0.5])
    assert np.array_equal(ccw.contains_m(x, y), cw.contains_m(x, y))


def test_polygon_winding_shoelace():
    # mirror of the _polygon_prism winding normalization (signed shoelace): CCW positive,
    # CW negative. The OCC builder reverses a negative-area (CW) ring so the extruded face is
    # positively oriented and the cell-intersection captures the footprint, not its complement
    # (the GEO-1 fix). Pure-logic check (no OCC).
    def area2(pts):
        return sum(x0 * y1 - x1 * y0 for (x0, y0), (x1, y1) in zip(pts, pts[1:] + pts[:1]))
    sq = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    assert area2(sq) > 0.0 and area2(sq[::-1]) < 0.0
