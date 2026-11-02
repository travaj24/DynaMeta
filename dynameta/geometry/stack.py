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
An Inclusion may carry an optional sidewall Taper (its bottom-face section;
widest at the top, narrowing about its own centre toward the base).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from dynameta.geometry.cross_section import (Circle, CrossSection, Ellipse, Polygon, Rectangle,
                                             RegularPolygon)


# ---- sidewall taper (2026-11-02, workstream B2 of the lumenairy-5.50 bridge PR) -----------------
# lumenairy 5.50's PMM2DStackPure.add_tapered_pillar(s) slices a pillar whose bounds interpolate
# LINEARLY from a bottom to a top cross-section ABOUT THE PILLAR'S OWN FIXED CENTRE. The house
# physics rule (memory feedback_taper_not_slant; measured 2026-10-06 on
# exp11_common.coated_pillar_geometry: finger 160 nm at the top -> 126.5 nm at the slot floor,
# 2.00 deg per face) is that EVERY metal feature is WIDEST AT ITS TOP and narrows toward its base,
# symmetrically about its own centre; keeping a gap constant by widening the neighbour is a SLANT
# and is wrong. The model below expresses exactly that, and nothing else by default:
#   * `Inclusion.shape` stays the section at the layer's TOP face (z_hi; the Stack is listed
#     bottom-to-top, superstrate above), so every existing reader of `inc.shape` (bbox mesh
#     refinement, symmetry gating, rasterizers) still sees the WIDEST section -- its bbox still
#     bounds the solid.
#   * `Inclusion.taper = Taper(bottom=<same-kind section at the layer's BOTTOM face>)`; the
#     sidewall is the ruled surface joining the two, linear in z.
#   * validated: same shape kind (no morph), SAME CENTRE as the top (an off-centre bottom is a
#     slant -> always refused), bottom no wider than the top in any extent unless the caller opts
#     in with allow_wider_bottom=True (an undercut / lift-off profile is a legitimate but
#     DIFFERENT device -- the 2026-10-06 note says the sensitivity must be re-checked if the
#     fabricated taper runs the other way, so it is never the silent default).
#   * taper=None (the default) is the ONLY spelling of an untapered inclusion; every existing
#     Design is unchanged (eq/repr gain a trailing `taper=None`; the result-cache key hashes a
#     taper only when one is present -- dynameta/cache.py).
# Backends that cannot honour a taper REFUSE it (dynameta/geometry/guards.py); mapping it onto
# PMM2DStackPure.add_tapered_pillar is a follow-on.
_TAPER_CENTRE_TOL_M = 1e-12      # the same 1 pm absolute centring tolerance Design.device_symmetry uses
_TAPER_SIZE_RTOL = 1e-12         # relative slack on "bottom no wider than top" (float round-off only)


def _taper_size_params(shape: CrossSection) -> Tuple[float, ...]:
    """Per-kind size parameters the no-wider rule compares (the centre is checked separately)."""
    if isinstance(shape, Rectangle):
        return (float(shape.width_m), float(shape.height_m))
    if isinstance(shape, Circle):
        return (float(shape.radius_m),)
    if isinstance(shape, Ellipse):
        return (float(shape.rx_m), float(shape.ry_m))
    if isinstance(shape, RegularPolygon):
        return (float(shape.radius_m),)
    return ()                    # Polygon: vertex containment, below


@dataclass
class Taper:
    """Sidewall TAPER of a layer Inclusion. `bottom` is the inclusion's cross-section at the layer's
    BOTTOM face; the inclusion's own `shape` is the section at the TOP face; the sidewall is linear
    in z between them, about the shape's own fixed centre -- a taper, never a slant.

    `bottom` must be the SAME CrossSection type as the top shape, share its centre (1 pm), and --
    unless allow_wider_bottom=True -- be no wider than the top in any extent: rectangle width and
    height, circle radius, ellipse rx and ry, regular-polygon circumradius (same n_sides AND
    rotation: a rotated bottom is a twist), polygon (same vertex count -- vertex i of the bottom
    joins vertex i of the top -- and every bottom vertex inside-or-on the top). A bottom EQUAL to
    the top is accepted (the zero-angle point of a taper sweep) but is still a TAPERED inclusion to
    the backend guards: spell an untapered inclusion taper=None. Build the house convention
    (N deg per face, widest at the top) with `Taper.from_sidewall_angle(shape, thickness, angle)`."""
    bottom:             CrossSection
    allow_wider_bottom: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.bottom, CrossSection):
            raise TypeError("Taper.bottom must be a CrossSection (got {!r})".format(type(self.bottom)))

    @classmethod
    def from_sidewall_angle(cls, top: CrossSection, thickness_m: float, angle_deg: float, *,
                            allow_wider_bottom: bool = False) -> "Taper":
        """The bottom section of a feature whose every face leans INWARD by `angle_deg` from the
        vertical over a layer `thickness_m` thick: each face moves in by d = thickness * tan(angle),
        so a width w_top -> w_top - 2 d (the exp10/exp11 `wz(w_top, z)` convention; the house value
        is 2 deg per face, but the caller states it). Exact for rectangle, circle and regular
        polygon (apothem - d, i.e. circumradius - d / cos(pi / n)); an ellipse's semi-axes move in
        by d (the true parallel curve of an ellipse is not an ellipse -- exact at its 4 vertices).
        A negative angle (wider bottom) needs allow_wider_bottom=True. A general Polygon has no
        single inward offset (the vertex count can change): give Taper(bottom=...) explicitly."""
        t = float(thickness_m)
        a = float(angle_deg)
        if not (t > 0.0):
            raise ValueError("Taper.from_sidewall_angle: thickness_m must be > 0 (got {})".format(t))
        if not (-90.0 < a < 90.0):
            raise ValueError("Taper.from_sidewall_angle: angle_deg must lie in (-90, 90) (got {})".format(a))
        if not isinstance(top, (Rectangle, Circle, Ellipse, RegularPolygon)):
            raise ValueError(
                "Taper.from_sidewall_angle: no exact inward offset for a {!r} section -- construct "
                "Taper(bottom=...) explicitly".format(getattr(top, "kind", type(top).__name__)))
        d = t * math.tan(math.radians(a))
        try:
            if isinstance(top, Rectangle):
                bot = Rectangle(top.cx_m, top.cy_m, top.width_m - 2.0 * d, top.height_m - 2.0 * d)
            elif isinstance(top, Circle):
                bot = Circle(top.cx_m, top.cy_m, top.radius_m - d)
            elif isinstance(top, Ellipse):
                bot = Ellipse(top.cx_m, top.cy_m, top.rx_m - d, top.ry_m - d)
            else:
                bot = RegularPolygon(top.cx_m, top.cy_m,
                                     top.radius_m - d / math.cos(math.pi / top.n_sides),
                                     top.n_sides, top.rotation_deg)
        except ValueError as e:
            raise ValueError(
                "Taper.from_sidewall_angle: {:.4g} deg per face over {:.4g} nm moves each face by "
                "{:.4g} nm, which closes the {!r} section ({})".format(
                    a, t * 1e9, d * 1e9, top.kind, e)) from None
        return cls(bottom=bot, allow_wider_bottom=allow_wider_bottom)

    def validate_against(self, top: CrossSection, where: str = "Inclusion") -> None:
        """Raise ValueError unless `bottom` is a same-type, same-centre (and, by default, no-wider)
        section of `top` -- the rules in the class docstring."""
        b = self.bottom
        if type(b) is not type(top):
            raise ValueError(
                "{}: taper bottom is a {!r} but the top shape is a {!r} -- a taper joins two sections "
                "of the SAME kind".format(where, getattr(b, "kind", type(b).__name__),
                                          getattr(top, "kind", type(top).__name__)))
        (tcx, tcy), (bcx, bcy) = top.center_m(), b.center_m()
        if abs(bcx - tcx) > _TAPER_CENTRE_TOL_M or abs(bcy - tcy) > _TAPER_CENTRE_TOL_M:
            raise ValueError(
                "{}: taper bottom centre ({:.6g}, {:.6g}) m is not the top shape's centre "
                "({:.6g}, {:.6g}) m -- an off-centre bottom is a SLANT, not a taper (every feature "
                "narrows about its OWN centre)".format(where, bcx, bcy, tcx, tcy))
        if isinstance(top, RegularPolygon):
            if b.n_sides != top.n_sides:
                raise ValueError("{}: taper bottom has {} sides, the top {}".format(
                    where, b.n_sides, top.n_sides))
            period = 360.0 / top.n_sides
            drot = (float(b.rotation_deg) - float(top.rotation_deg)) % period
            if min(drot, period - drot) > 1e-9:
                raise ValueError("{}: taper bottom is rotated {:.6g} deg against the top -- a twist, "
                                 "not a taper".format(where, drot))
        if isinstance(top, Polygon) and b.vertices_m().shape != top.vertices_m().shape:
            raise ValueError("{}: taper bottom has {} vertices, the top {} -- vertex i of the bottom "
                             "joins vertex i of the top".format(
                                 where, len(b.vertices_m()), len(top.vertices_m())))
        if self.allow_wider_bottom:
            return
        wider = [(i, pb, pt) for i, (pb, pt) in enumerate(zip(_taper_size_params(b), _taper_size_params(top)))
                 if pb > pt * (1.0 + _TAPER_SIZE_RTOL)]
        if isinstance(top, Polygon):
            vb, vt = b.vertices_m(), top.vertices_m()
            on_top = np.all(np.abs(vb - vt) <= _TAPER_CENTRE_TOL_M, axis=1)   # coincident vertex (0 deg)
            inside = np.asarray(top.contains_m(vb[:, 0], vb[:, 1]), dtype=bool)
            wider += [("vertex {}".format(int(i)), tuple(vb[i]), "outside the top polygon")
                      for i in np.flatnonzero(~(on_top | inside))]
        if wider:
            raise ValueError(
                "{}: taper bottom is WIDER than the top ({}) -- the house rule is widest at the top, "
                "narrowing toward the base; pass Taper(..., allow_wider_bottom=True) for a deliberate "
                "undercut profile".format(where, wider))


@dataclass
class Inclusion:
    shape:    CrossSection
    material: str
    priority: int = 0          # higher priority wins where inclusions overlap
    # sidewall taper (2026-11-02): None = straight walls -- the ONLY untapered spelling, so every
    # existing Design is unchanged. Otherwise `shape` is the TOP-face section and taper.bottom the
    # BOTTOM-face section (see Taper). Backends that cannot build it refuse (geometry/guards.py).
    taper:    Optional[Taper] = None

    def __post_init__(self) -> None:
        if self.taper is not None:
            if not isinstance(self.taper, Taper):
                raise TypeError("Inclusion.taper must be a Taper or None (got {!r})".format(
                    type(self.taper)))
            self.taper.validate_against(self.shape, where="Inclusion({!r})".format(self.material))


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
    later phase; the data model supports it now. Until then EVERY backend
    refuses a Stack with features (UnsupportedGeometryError, worded once in
    dynameta.geometry.guards; 2026-11-02 -- all but the FEM builder used to
    drop the feature silently)."""
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
