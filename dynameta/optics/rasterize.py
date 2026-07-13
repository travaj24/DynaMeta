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

__all__ = ["cell_axes", "layer_bg_eps", "layer_eps_cell"]


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


def layer_eps_cell(layer, X, Y, lambda_m, materials, eps_by_region):
    """The (nx,ny) COMPLEX eps cross-section of one layer: the background eps, overpainted by each
    inclusion (CrossSection.contains_m mask) in ASCENDING priority so the highest priority wins overlaps."""
    eps = np.full(X.shape, layer_bg_eps(layer, lambda_m, materials, eps_by_region), dtype=complex)
    for inc in sorted(layer.inclusions, key=lambda i: getattr(i, "priority", 0)):
        mask = np.asarray(inc.shape.contains_m(X, Y), dtype=bool)
        eps[mask] = complex(materials.get(inc.material).eps(lambda_m))
    return eps
