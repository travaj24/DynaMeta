"""Lateral rasterizer -- paint a Design layer's geometry onto a transverse eps grid.

The ONE rasterization shared by every grid-based backend: cell-centred sample axes
(cell_axes) plus the per-layer complex eps cross-section (layer_eps_cell: background eps
overpainted by each inclusion's CrossSection.contains_m mask in ascending priority). Both
the structured FDTD seam (optics.fdtd_seam.make_structured_lateral) and the lumenairy RCWA
bridge (optics.lumenairy_bridge.rcwa_backend) consume it, so the two backends see
byte-identical geometry. Promoted from fdtd_seam's private helpers (audit 2026-07-05
section 6.3: the bridge imported the underscore names across module boundaries); the old
_cell_axes/_layer_bg_eps/_layer_eps_cell names stay re-exported there for back-compat.
Pure numpy, no solver imports.
"""
from __future__ import annotations

import numpy as np

from dynameta.geometry.guards import require_untapered_layer

__all__ = ["cell_axes", "layer_bg_eps", "layer_eps_cell", "layer_eps_tensor_cell", "inclusion_mask"]


def cell_axes(nx, ny, period_x_m, period_y_m):
    """Cell-centered FDTD lateral sample points (cell frame [0,period], shapes in absolute coords)."""
    xs = (np.arange(nx) + 0.5) * (period_x_m / nx)
    ys = (np.arange(ny) + 0.5) * (period_y_m / ny)
    return xs, ys


def layer_bg_eps(layer, lambda_m, materials, eps_by_region):
    ef = (eps_by_region or {}).get(layer.name)
    if ef is not None:
        if getattr(ef, "is_tensor", False) or not getattr(ef, "is_uniform", True):
            # audit C5-2: the structured (rasterized) path used to silently substitute the
            # NOMINAL material eps for a graded/tensor entry -- the per-layer lateral painter
            # can only carry one uniform scalar per layer, so refuse rather than mis-solve
            raise NotImplementedError(
                "FDTD structured path: layer '{}' carries a {} eps_by_region entry; the lateral "
                "rasterizer represents one uniform scalar per layer -- use the FEM solver (or the "
                "RCWA bridge) for graded/tensor structured cells.".format(
                    layer.name,
                    "TENSOR" if getattr(ef, "is_tensor", False) else "graded (gridded)"))
        if getattr(ef, "scalar", None) is not None:
            return complex(ef.scalar)
    return complex(materials.get(layer.background_material).eps(lambda_m))


def layer_eps_cell(layer, X, Y, lambda_m, materials, eps_by_region, *, period_x_m=None,
                   period_y_m=None):
    """The (nx,ny) COMPLEX eps cross-section of one layer: the background eps, overpainted by each
    inclusion (CrossSection.contains_m mask) in ASCENDING priority so the highest priority wins overlaps.
    A TAPERED inclusion is refused (2026-11-02): one z-invariant section per layer would paint its TOP
    section as a straight wall.

    period_x_m / period_y_m make the paint PERIODIC (2026-11-02): an inclusion whose bbox leaves the
    closed cell [0, Px] x [0, Py] also paints its translates by -/+(Px, Py), i.e. the piece that wraps
    to the opposite edge -- the FEM mesher's convention, and the only reading a periodic solver can
    have. Without them a boundary-crossing inclusion is CLIPPED: a 0.4-period line centred on x = 0
    rasterized as a 0.2-period line (RCWA R 0.0552 vs 0.0951; the wrapped raster is 2.4e-4 from the
    FEM, the clipped one 2.7e-3). Every in-repo periodic caller passes both; an inclusion inside the
    closed cell paints byte-identically either way (its translates are never evaluated)."""
    require_untapered_layer(layer, "layer_eps_cell (lateral rasterizer)")
    eps = np.full(X.shape, layer_bg_eps(layer, lambda_m, materials, eps_by_region), dtype=complex)
    for inc in sorted(layer.inclusions, key=lambda i: getattr(i, "priority", 0)):
        mask = inclusion_mask(inc, X, Y, period_x_m=period_x_m, period_y_m=period_y_m)
        eps[mask] = complex(materials.get(inc.material).eps(lambda_m))
    return eps


def layer_eps_tensor_cell(layer, X, Y, lambda_m, materials, bg_tensor, *, period_x_m=None,
                          period_y_m=None):
    """The (nx, ny, 3, 3) COMPLEX eps_tensor_cell of one layer on a uniform TENSOR background: every
    sample starts as `bg_tensor` (eps[i, j], i, j in (x, y, z) -- the EpsField.tensor layout) and each
    (scalar) inclusion is overpainted as eps * I3 in ASCENDING priority through the shared
    inclusion_mask (periodic when both periods are given) -- layer_eps_cell's painting rule, tensor-
    valued. The ONE tensor painter (2026-11-02): the PMM2D hybrid engine's raster and the RCWA
    bridge's uniform-tensor-plus-inclusions layer both call it. (layer_eps_cell itself keeps refusing
    a tensor eps_by_region entry -- audit C5-2 -- because its consumers carry one scalar per sample.)
    A TAPERED inclusion is refused, as in layer_eps_cell."""
    require_untapered_layer(layer, "layer_eps_tensor_cell (lateral rasterizer)")
    bg = np.asarray(bg_tensor, dtype=complex).reshape(3, 3)
    cell = np.broadcast_to(bg, X.shape + (3, 3)).copy()
    for inc in sorted(layer.inclusions, key=lambda i: getattr(i, "priority", 0)):
        mask = inclusion_mask(inc, X, Y, period_x_m=period_x_m, period_y_m=period_y_m)
        cell[mask] = complex(materials.get(inc.material).eps(lambda_m)) * np.eye(3, dtype=complex)
    return cell


def inclusion_mask(inc, X, Y, *, period_x_m=None, period_y_m=None):
    """Boolean (X.shape) mask of one inclusion's CrossSection -- the ONE painting predicate every
    rasterizer shares (layer_eps_cell; layer_eps_tensor_cell). With both periods it is
    PERIODIC: an inclusion whose bbox leaves the closed cell [0, Px] x [0, Py] also covers its
    translates by -/+(Px, Py); without them it clips (see layer_eps_cell)."""
    if (period_x_m is None) != (period_y_m is None):
        raise ValueError("inclusion_mask/layer_eps_cell: pass both period_x_m and period_y_m, or neither")
    mask = np.asarray(inc.shape.contains_m(X, Y), dtype=bool)
    if period_x_m is not None:
        px, py = float(period_x_m), float(period_y_m)
        xlo, xhi, ylo, yhi = inc.shape.bbox_m()
        dxs = [0.0] + ([px] if xlo < 0.0 else []) + ([-px] if xhi > px else [])
        dys = [0.0] + ([py] if ylo < 0.0 else []) + ([-py] if yhi > py else [])
        for dx in dxs:
            for dy in dys:
                if dx or dy:
                    # the shape contains p + (dx, dy) iff its translate by -(dx, dy) contains p
                    mask |= np.asarray(inc.shape.contains_m(X - dx, Y - dy), dtype=bool)
    return mask
