"""Analytic-shape payloads + the PERIODIC (wrap-aware) lateral painter for the lumenairy bridge.

Workstream C of the 2026-11-02 "bridge 5.50 + P3 audit" PR. One module, three consumers:
  * rcwa_backend   -- the FORWARD analytic path (Design inclusions -> RCWAStack shapes=) and the
                      wrap-aware raster fallback;
  * translate      -- the REVERSE path (lumenairy 'shapes' records -> DynaMeta CrossSections);
  * LayeredSlab    -- the first PRODUCER of `LayeredSlab(shapes=...)` (layer_to_shapes_slab); the
                      slab kind was declared in core/layered.py with no producer and no consumer.

THE PAYLOAD CONTRACT (pinned against lumenairy 5.50 RCWAStack.add_layer + _core._validate_shapes,
the single validator that add_layer runs): a LIST of dicts
    {'shape': 'rectangle', 'size': (wx, wy), 'center': (cx, cy), 'eps': complex}
    {'shape': 'disk', 'radius': r, 'center': (cx, cy), 'eps': complex}
    {'shape': 'ellipse', 'semi_axes': (ax, ay), 'center': (cx, cy), 'eps': complex}
in METRES, centre in the ABSOLUTE cell frame [0, Px] x [0, Py] (lumenairy's default centre is
(Px/2, Py/2), i.e. the same frame DynaMeta's CrossSections use), eps in the PUBLIC convention
(Im > 0 lossy), plus a scalar eps_background. Rectangles and ellipses are AXIS-ALIGNED on both
sides (DynaMeta's Rectangle/Ellipse carry no rotation), so Rectangle/Circle/Ellipse map 1:1 and
EXACTLY. RegularPolygon / Polygon have no lumenairy form factor -> not representable (the caller
rasterizes). The form factors are ADDED over the background, so the list must be DISJOINT;
lumenairy rejects an overlap, an over-period extent or a cumulative area fraction > 1 with a
ValueError, and the bridge treats that refusal as "not exactly representable" (DynaMeta's
priority over-painting has no additive equivalent on an overlap).

PERIODIC WRAP -- the convention this module fixes for the bridge. The DynaMeta FEM mesher WRAPS a
boundary-crossing inclusion (optics/ngsolve_layered.py `_inclusion_solids_clipped`: the shape is
unioned with its +-P translates, each clipped to the cell); lumenairy's analytic form factors
wrap by construction (sampled on the reciprocal lattice; measured translation-invariant to 3e-14
in R here). The shared point rasterizer (optics.rasterize.layer_eps_cell) does NOT -- it CLIPS, so
the RCWA raster path used to drop the wrapped piece of a boundary-crossing inclusion silently
(MEASURED 2026-11-02: a 0.4-period ridge centred on x = 0 solved as a 0.2-period ridge). The
painter below unions each inclusion with its +-P translates. FEM ORACLE (out-of-band, 2026-11-02:
400 nm cell, eps-4 disk r = 120 nm centred at x = 32 nm so it CROSSES x = 0, 200 nm, air|glass,
normal incidence, y-pol, NGSolve order 2, maxh 22 nm, 43 s): FEM R = 0.006484; RCWA analytic
|dR| = 3.4e-4 / 2.7e-4 / 2.3e-4 / 2.1e-4 at n_orders 5 / 7 / 9 / 11 (monotone -> the residual is
Laurent truncation, and the crossing disk equals the centred one to 3e-14); the wrapped raster
(S 256, n_orders 9) 2.4e-4; the CLIPPED raster 2.7e-3 -- 11x further from the FEM than the
wrapped one, i.e. the FEM confirms wrap is the right reading. For an inclusion inside the closed
cell it is BYTE-IDENTICAL to layer_eps_cell (the translates are only evaluated for a bbox that
leaves the cell; cell-centred samples are strictly inside (0, P), so a shape touching x = P picks
up no translated sample). optics.rasterize itself (shared with the structured FDTD) is NOT
touched here -- the FDTD clip is reported as an open issue, not silently changed under it.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

__all__ = ["ShapesNotRepresentable", "cross_section_to_shape", "shape_to_cross_section",
           "layer_shapes_payload", "layer_to_shapes_slab", "periodic_layer_eps_cell"]

_FULL_TOL = 1e-12     # relative slack for "this rectangle spans the whole period"


class ShapesNotRepresentable(ValueError):
    """The layer's inclusions have no EXACT lumenairy analytic-shape equivalent (an unsupported
    CrossSection kind, an overlap the additive form factors cannot paint, or a refusal from
    lumenairy's own validator). The message says which; 'auto' callers rasterize instead."""


def cross_section_to_shape(cs, eps: complex, period_x_m: float, period_y_m: float,
                           stack_period_y_m: Optional[float] = None) -> dict:
    """ONE DynaMeta CrossSection -> one lumenairy shape dict (exact), else ShapesNotRepresentable.

    A Rectangle whose extent reaches or exceeds a FULL period on an axis covers that whole axis,
    so it is emitted with size == the period and centre at P/2 on that axis -- the same set of
    points, and the only spelling lumenairy's extent check accepts (a lamellar line is commonly
    written with height == Py, sometimes larger). stack_period_y_m: the RCWAStack's own period_y,
    which differs from the design's Py on a 1-D (lamellar) stack (a 1-D RCWAStack reports
    period_y == period_x); a full-y rectangle is mapped onto THAT period so it stays
    y-invariant in lumenairy's eyes (no RCWAYAverageWarning)."""
    px, py = float(period_x_m), float(period_y_m)
    spy = float(stack_period_y_m) if stack_period_y_m is not None else py
    kind = getattr(cs, "kind", "")
    if kind == "rectangle":
        cx, cy = float(cs.cx_m), float(cs.cy_m)
        wx, wy = float(cs.width_m), float(cs.height_m)
        if wx >= px * (1.0 - _FULL_TOL):
            wx, cx = px, 0.5 * px
        if wy >= py * (1.0 - _FULL_TOL):
            wy, cy = spy, 0.5 * spy
        return {"shape": "rectangle", "size": (wx, wy), "center": (cx, cy), "eps": complex(eps)}
    if kind == "circle":
        return {"shape": "disk", "radius": float(cs.radius_m),
                "center": (float(cs.cx_m), float(cs.cy_m)), "eps": complex(eps)}
    if kind == "ellipse":
        return {"shape": "ellipse", "semi_axes": (float(cs.rx_m), float(cs.ry_m)),
                "center": (float(cs.cx_m), float(cs.cy_m)), "eps": complex(eps)}
    raise ShapesNotRepresentable(
        "CrossSection kind {!r} has no lumenairy analytic form factor (only axis-aligned "
        "rectangle / circle / ellipse do); rasterize this layer instead".format(kind))


def shape_to_cross_section(sh: dict, period_x_m: float, period_y_m: float):
    """ONE lumenairy shape dict -> the DynaMeta CrossSection describing the SAME point set (the
    reverse of cross_section_to_shape). A missing 'center' is lumenairy's default (Px/2, Py/2)."""
    from dynameta.geometry.cross_section import Circle, Ellipse, Rectangle
    kind = sh.get("shape")
    cx, cy = sh.get("center", (0.5 * float(period_x_m), 0.5 * float(period_y_m)))
    cx, cy = float(cx), float(cy)
    if kind == "rectangle":
        wx, wy = sh["size"]
        return Rectangle(cx, cy, float(wx), float(wy))
    if kind == "disk":
        return Circle(cx, cy, float(sh["radius"]))
    if kind == "ellipse":
        ax, ay = sh["semi_axes"]
        return Ellipse(cx, cy, float(ax), float(ay))
    raise ShapesNotRepresentable("unknown lumenairy shape kind {!r}".format(kind))


def layer_shapes_payload(layer, lambda_m: float, materials, bg_eps: complex,
                         period_x_m: float, period_y_m: float,
                         stack_period_y_m: Optional[float] = None) -> Tuple[complex, List[dict]]:
    """A DynaMeta Layer's inclusions as the (eps_background, shapes) pair of
    RCWAStack.add_layer(shapes=, eps_background=). Raises ShapesNotRepresentable when a
    CrossSection kind has no form factor. Disjointness / extent / area are NOT re-implemented
    here: lumenairy's add_layer validator is the single authority, and the caller maps its
    ValueError onto ShapesNotRepresentable (rcwa_backend._add_shapes_layer)."""
    shapes = [cross_section_to_shape(inc.shape, complex(materials.get(inc.material).eps(lambda_m)),
                                     period_x_m, period_y_m, stack_period_y_m)
              for inc in layer.inclusions]
    return complex(bg_eps), shapes


def layer_to_shapes_slab(layer, design, lambda_m: float, *, eps_by_region=None):
    """The analytic-shape LayeredSlab of one patterned Design layer -- the first producer of the
    `LayeredSlab(shapes=, eps_background=)` kind that core/layered.py declared and
    LumenairyStackSolver now consumes. The background follows the rasterizer's contract (an
    eps_by_region uniform-scalar override applies to the BACKGROUND only; graded/tensor entries
    are refused by layer_bg_eps). Raises ShapesNotRepresentable for an unsupported kind; overlap
    is caught when the slab reaches lumenairy (LumenairyStackSolver re-raises it as
    ShapesNotRepresentable)."""
    from dynameta.core.layered import LayeredSlab
    from dynameta.optics.rasterize import layer_bg_eps
    if not layer.inclusions:
        raise ShapesNotRepresentable("layer {!r} has no inclusions (use eps=)".format(layer.name))
    bg = layer_bg_eps(layer, lambda_m, design.materials, eps_by_region)
    bg, shapes = layer_shapes_payload(layer, lambda_m, design.materials, bg,
                                      design.unit_cell.period_x_m, design.unit_cell.period_y_m)
    return LayeredSlab(float(layer.thickness_m), shapes=shapes, eps_background=bg)


def periodic_layer_eps_cell(layer, X, Y, lambda_m: float, materials, eps_by_region,
                            period_x_m: float, period_y_m: float):
    """optics.rasterize.layer_eps_cell with PERIODIC WRAP: each inclusion's mask is the union of
    contains_m over its (dx, dy) in {-Px, 0, Px} x {-Py, 0, Py} translates (only evaluated when
    its bbox leaves the closed cell), painted in ascending priority exactly as layer_eps_cell
    does. Byte-identical to layer_eps_cell for every inclusion inside [0, Px] x [0, Py]."""
    from dynameta.optics.rasterize import layer_bg_eps
    px, py = float(period_x_m), float(period_y_m)
    eps = np.full(X.shape, layer_bg_eps(layer, lambda_m, materials, eps_by_region), dtype=complex)
    for inc in sorted(layer.inclusions, key=lambda i: getattr(i, "priority", 0)):
        xlo, xhi, ylo, yhi = inc.shape.bbox_m()
        dxs = [0.0] + ([px] if xlo < 0.0 else []) + ([-px] if xhi > px else [])
        dys = [0.0] + ([py] if ylo < 0.0 else []) + ([-py] if yhi > py else [])
        mask = np.zeros(X.shape, dtype=bool)
        for dx in dxs:
            for dy in dys:
                # a shape centred at c contains the point p + (dx, dy) iff its translate by
                # -(dx, dy) contains p; evaluating at shifted points needs no shape copy
                mask |= np.asarray(inc.shape.contains_m(X - dx, Y - dy), dtype=bool)
        eps[mask] = complex(materials.get(inc.material).eps(lambda_m))
    return eps
