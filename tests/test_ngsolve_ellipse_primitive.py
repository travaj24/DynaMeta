"""Ellipse inclusions are true OCC ellipses, not inscribed polygons.

The builder used to approximate every ellipse by an inscribed 72-gon, which under-filled the area
by a systematic 0.127% -- always in the same direction, so it biased rather than merely blurred
(audit GEO-3). The primitive reproduces pi*rx*ry*h instead.

The gate that earns its keep here is the TALL one. occ.WorkPlane.Ellipse(major, minor) segfaults
-- SIGSEGV, not a Python exception, so it can be neither caught nor reported -- whenever
minor > major. Passing (rx, ry) in their natural order would therefore take the whole process down
for any ellipse taller than it is wide. These gates pin the by-construction choice of major axis
that makes that unreachable, in the same process, where a regression crashes the run outright.
"""

import math

import pytest

occ = pytest.importorskip("netgen.occ", reason="OCC geometry builder needs netgen")

from dynameta.optics.ngsolve_layered import LayeredOpticalBuilder, S      # noqa: E402


class _Ellipse:
    """The minimal duck-type _inclusion_solid consumes for an ellipse (metres in, as always)."""

    kind = "ellipse"

    def __init__(self, cx, cy, rx, ry):
        self._c, self.rx_m, self.ry_m = (cx, cy), rx, ry

    def center_m(self):
        return self._c


def _solid(cx, cy, rx, ry, z_lo=0.0, z_hi=100.0):
    b = LayeredOpticalBuilder.__new__(LayeredOpticalBuilder)          # no Design needed
    return b._inclusion_solid(_Ellipse(cx, cy, rx, ry), z_lo, z_hi)


@pytest.mark.parametrize("rx_nm,ry_nm", [
    (300.0, 120.0),        # wide  -- major already along x
    (120.0, 300.0),        # TALL  -- naive Ellipse(rx, ry) would SIGSEGV here
    (200.0, 200.0),        # equal -- the degenerate boundary between the two branches
    (400.0, 12.0),         # extreme aspect, wide
    (12.0, 400.0),         # extreme aspect, tall
])
def test_the_volume_is_the_analytic_ellipse_whatever_the_aspect(rx_nm, ry_nm):
    """pi*rx*ry*h, to OCC's own integration tolerance. The inscribed 72-gon this replaced was
    0.127% light, which is ~3000x the error budget asserted here."""
    h = 100.0
    rx, ry = rx_nm / S, ry_nm / S                                     # the API takes metres
    sol = _solid(0.0, 0.0, rx, ry, 0.0, h)
    want = math.pi * rx_nm * ry_nm * h
    assert sol.mass > 0.0, "negative volume -- the face extruded to the COMPLEMENT (audit GEO-1)"
    assert abs(sol.mass - want) / want < 1.0e-5, (
        "volume {:.6e} vs analytic {:.6e} (rel {:.2e}) -- not a true ellipse"
        .format(sol.mass, want, abs(sol.mass - want) / want))


@pytest.mark.parametrize("rx_nm,ry_nm", [(300.0, 120.0), (120.0, 300.0), (12.0, 400.0)])
def test_the_bounding_box_carries_the_radii_on_the_right_axes(rx_nm, ry_nm):
    """Guards the branch that swaps the radii for a tall ellipse: swapping them without also
    re-pointing the workplane's x-axis would give the right AREA on the wrong AXES -- a defect no
    volume check can see."""
    lo, hi = _solid(0.0, 0.0, rx_nm / S, ry_nm / S).bounding_box
    for got, want, axis in ((hi[0], rx_nm, "x"), (hi[1], ry_nm, "y")):
        assert abs(got - want) < 1.0e-6 * want, (
            "half-extent along {} is {:.4f} nm, expected {:.4f} (radii landed transposed)"
            .format(axis, got, want))
    assert abs(lo[0] + rx_nm) < 1.0e-6 * rx_nm and abs(lo[1] + ry_nm) < 1.0e-6 * ry_nm


def test_the_ellipse_is_centred_where_it_was_asked_for():
    """The centre comes from the workplane's own origin rather than a MoveTo, and the tall branch
    builds on a DIFFERENT workplane -- so centring is re-checked on both."""
    for rx_nm, ry_nm in ((300.0, 120.0), (120.0, 300.0)):
        cx_nm, cy_nm = 640.0, -220.0
        lo, hi = _solid(cx_nm / S, cy_nm / S, rx_nm / S, ry_nm / S).bounding_box
        assert abs((lo[0] + hi[0]) / 2.0 - cx_nm) < 1.0e-4, "x centre drifted"
        assert abs((lo[1] + hi[1]) / 2.0 - cy_nm) < 1.0e-4, "y centre drifted"


def test_a_round_ellipse_agrees_with_the_circle_branch():
    """rx == ry should give the same solid as `circle`. The 72-gon did not: it was 0.127% light
    while the circle branch built an exact occ.Cylinder, so two spellings of one shape disagreed."""
    r_nm, h = 250.0, 100.0
    ell = _solid(0.0, 0.0, r_nm / S, r_nm / S, 0.0, h)
    cyl = occ.Cylinder(occ.Pnt(0.0, 0.0, 0.0), occ.Z, r=r_nm, h=h)
    assert abs(ell.mass - cyl.mass) / cyl.mass < 1.0e-5, (
        "ellipse(r,r) mass {:.6e} vs cylinder {:.6e}".format(ell.mass, cyl.mass))
