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


def _sym_design(shape, *, pol="y", theta=0.0, azimuth=0.0):
    from dynameta.geometry import Design, UnitCell, Stack, Layer, Inclusion
    from dynameta.geometry.specs import Mesh3DSpec, OpticalSpec
    from dynameta.materials import ConstantOptical, Material, MaterialRegistry
    reg = MaterialRegistry()
    for nm, e in (("air", 1.0 + 0j), ("pil", complex(2.5 ** 2))):
        reg.add(Material(nm, ConstantOptical(e)))
    stack = Stack(layers=[Layer("slab", 200e-9, "air", inclusions=[Inclusion(shape=shape, material="pil")])],
                  superstrate_material="air", substrate_material="air")
    return Design(name="d", unit_cell=UnitCell.square(400e-9), stack=stack, electrodes=[], materials=reg,
                  optical=OpticalSpec(polarization=pol, incidence_angle_deg=theta, azimuth_deg=azimuth),
                  mesh_3d=Mesh3DSpec())


def test_detect_symmetry_reduction():
    # the ADVISORY detector (Design.detect_symmetry_reduction): reports the best ELIGIBLE FEM reduction
    # WITHOUT applying it (mesh_3d.symmetry stays 'none' -- never auto). Pure geometry; no ngsolve.
    from dynameta.geometry import UnitCell
    from dynameta.geometry.cross_section import RegularPolygon, centered_circle, centered_rectangle
    cell = UnitCell.square(400e-9)
    d_c4v = _sym_design(centered_circle(cell, 120e-9))                  # square cell + centered disk
    d_c2v = _sym_design(centered_rectangle(cell, 220e-9, 130e-9))       # w != h -> 2 mirror axes only
    assert d_c4v.device_symmetry() == "c4v" and d_c4v.detect_symmetry_reduction() == "quarter"
    assert d_c2v.device_symmetry() == "c2v" and d_c2v.detect_symmetry_reduction() == "half_x"
    # NEVER auto: the eligible design still has symmetry 'none' until the user opts in
    assert d_c4v.mesh_3d.symmetry == "none"
    # ineligible -> 'none' (each branch): oblique, conical, off-center, unsupported (rotated-able) shape
    assert _sym_design(centered_circle(cell, 120e-9), pol="y", theta=20.0).detect_symmetry_reduction() == "none"
    assert _sym_design(centered_circle(cell, 120e-9), pol="y", azimuth=15.0).detect_symmetry_reduction() == "none"
    # p-pol: the wall type is keyed to a linear x/y E axis, and solve_fem rejects 'p' on a sym mesh,
    # so the detector must NOT advertise a reduction for it (strict subset of the solvable path)
    assert _sym_design(centered_circle(cell, 120e-9), pol="p").detect_symmetry_reduction() == "none"
    from dynameta.geometry import Inclusion  # off-center inclusion -> device_symmetry 'none'
    off = _sym_design(centered_circle(cell, 120e-9))
    off.stack.layers[0].inclusions[0] = Inclusion(shape=__import__("dynameta.geometry.cross_section",
                                                  fromlist=["Circle"]).Circle(50e-9, 50e-9, 120e-9),
                                                  material="pil")
    assert off.detect_symmetry_reduction() == "none"
    # a regular polygon is c2v/c4v by intrinsic symmetry but OUT of the reduced-mesh scope -> 'none'
    assert _sym_design(RegularPolygon(200e-9, 200e-9, 120e-9, 6)).detect_symmetry_reduction() == "none"


def test_polygon_winding_shoelace():
    # mirror of the _polygon_prism winding normalization (signed shoelace): CCW positive,
    # CW negative. The OCC builder reverses a negative-area (CW) ring so the extruded face is
    # positively oriented and the cell-intersection captures the footprint, not its complement
    # (the GEO-1 fix). Pure-logic check (no OCC).
    def area2(pts):
        return sum(x0 * y1 - x1 * y0 for (x0, y0), (x1, y1) in zip(pts, pts[1:] + pts[:1]))
    sq = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    assert area2(sq) > 0.0 and area2(sq[::-1]) < 0.0
