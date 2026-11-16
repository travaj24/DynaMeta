"""Area-weighted rasters on lumenairy's NODE lattice (the RCWA bridge's default raster, v0.16).

THE LATTICE. lumenairy reads an ``eps_cell[j, i]`` as the permittivity at the NODE
``(x_j, y_i) = (j Px / Sx, i Py / Sy)`` (RCWAStack.add_layer, "WHAT A PIXEL MEANS"): the Laurent
table is ``fft2(cell) / (Sx Sy)``, the band-limited profile through those samples. The analytic
shapes sit in the true frame. So pixel j here REPRESENTS the cell
``[x_j - Px / (2 Sx), x_j + Px / (2 Sx)]`` around its node (pixel 0 wraps across x = 0), and its
value is the AREA AVERAGE of the permittivity over that cell. Consequences:

* the drawn geometry is realised exactly in the mean: the covered fraction of every pixel, summed
  and multiplied by Px / Sx, is the drawn width, whatever S and wherever the edges fall (the point
  raster moves each edge to the nearest pixel boundary: up to half a pixel);
* a raster layer and an analytic layer in one stack share one frame (the legacy point raster
  sampled the cell CENTRES (j + 1/2) Px / Sx and was misregistered by Px / (2 Sx) against the
  analytic layers).

INVERSE-RULE COMPANION. Li's rule builds the wall-normal blocks from ``[[1/eps]]^{-1}``. For a
pixel that a wall crosses, the matching subpixel average is HARMONIC across the wall and arithmetic
along it (Farjadpour et al. 2006). lumenairy >= 5.31 takes that pair as
``add_layer(eps_cell_normal=(exx, eyy))`` next to ``eps_cell`` (FEATURE_FLOORS
'rcwa_eps_cell_normal'): exx = harmonic along x, then arithmetic along y; eyy = harmonic along y,
then arithmetic along x. Both are exact for axis-aligned rectangles.

HOW THE AVERAGE IS TAKEN. Each pixel is split into sub-cells whose walls include every rectangle
edge (and its periodic image), so a sub-cell holds one material and the shared painter
(optics.rasterize: background + priority-ordered inclusions, periodic wrap) evaluated at its
midpoint gives it exactly. Rectangles are therefore exact. A pixel column / row that the bounding
box of a curved or polygonal inclusion reaches is further split into ``SUB_SAMPLES`` uniform
sub-cells per axis (16 x 16 sub-samples per pixel; midpoint rule, error O(1/(16 S)) in area).

GRIDDED FIELDS. ``area_resample_cell`` averages the periodic piecewise-LINEAR interpolant of a
sampled lateral profile over the same pixel cells, with the exact interval means of
_lateral_partition (the one implementation of those integrals), so a carrier-graded field and an
inclusion raster share one lattice and one averaging convention.

Pure numpy; no lumenairy import.
"""

from __future__ import annotations

from typing import Callable, Sequence, Tuple

import numpy as np

from dynameta.optics.lumenairy_bridge._lateral_partition import interval_means, periodic_samples

__all__ = ["SUB_SAMPLES", "area_weighted_cell", "area_weighted_tensor_cell", "area_resample_cell",
           "realised_coverage"]

SUB_SAMPLES = 16          # uniform sub-cells per axis in a pixel a curved / polygonal shape reaches
_SNAP = 1e-9              # sub-cell walls closer than this (in pixels) to a pixel wall ARE that wall
_CHUNK = 2_000_000        # sub-cells painted per block (memory bound for 2-D curved shapes)


def _axis_subcells(n: int, period_m: float, edges: Sequence[float],
                   fine_spans: Sequence[Tuple[float, float]]):
    """Sub-cells of one axis over the pixel domain [-h/2, P - h/2] (h = P / n).

    Returns (widths, pixel_index, midpoints_mod_P, starts): widths of the sub-cells, the pixel each
    belongs to, their midpoints mapped into [0, P), and the index of each pixel's first sub-cell.
    `edges` are exact material walls (any real position; mapped periodically); `fine_spans` are
    [lo, hi] intervals (metres, any real; mapped periodically) whose pixels are subdivided into
    SUB_SAMPLES uniform sub-cells."""
    P = float(period_m)
    h = P / n
    lo = -0.5 * h
    walls = (np.arange(n + 1) - 0.5) * h
    pts = [walls]
    if edges:
        e = (np.asarray(edges, dtype=np.float64) - lo) % P + lo
        # an edge within round-off of a pixel wall IS that wall (no sliver sub-cell)
        k = np.rint((e - lo) / h)
        e = np.where(np.abs((e - lo) / h - k) <= _SNAP, walls[np.clip(k.astype(int), 0, n)], e)
        pts.append(e)
    fine = np.zeros(n, dtype=bool)
    for a, b in fine_spans:
        if b - a >= P:
            fine[:] = True
            continue
        # pixels whose cell [x_j - h/2, x_j + h/2] meets [a, b] (periodically)
        ja = int(np.floor((a - lo) / h))
        jb = int(np.floor((b - lo) / h))
        idx = np.arange(ja, jb + 1) % n
        fine[idx] = True
    if np.any(fine):
        sub = (np.arange(1, SUB_SAMPLES) / SUB_SAMPLES) * h
        pts.append((walls[:-1][fine][:, None] + sub[None, :]).ravel())
    B = np.unique(np.concatenate(pts))
    B = B[(B >= walls[0]) & (B <= walls[-1])]
    w = np.diff(B)
    keep = w > _SNAP * h * 1e-3          # drop zero-width duplicates left by unique on floats
    starts_all = B[:-1][keep]
    w = w[keep]
    mid = starts_all + 0.5 * w
    pix = np.clip(np.floor((mid - lo) / h).astype(int), 0, n - 1)
    starts = np.searchsorted(pix, np.arange(n), side="left")
    if np.any(np.diff(starts) <= 0) or starts[0] != 0:
        raise RuntimeError("area raster: a pixel received no sub-cell (internal error)")
    xm = np.mod(mid, P)
    return w, pix, xm, starts


def _layer_walls(layer, axis: int):
    """Exact material walls along `axis` (0 = x, 1 = y) from the layer's rectangles, and the spans
    of its curved / polygonal inclusions (to be sub-sampled)."""
    edges, spans = [], []
    for inc in layer.inclusions:
        xlo, xhi, ylo, yhi = inc.shape.bbox_m()
        lo, hi = (xlo, xhi) if axis == 0 else (ylo, yhi)
        if getattr(inc.shape, "kind", "") == "rectangle":
            edges.extend([lo, hi])
        else:
            spans.append((lo, hi))
            edges.extend([lo, hi])
    return edges, spans


def _average(paint: Callable, sx: int, sy: int, px: float, py: float, layer,
             companion: bool, tensor: bool):
    """Pixel averages of paint(X, Y) -> values (X.shape [+ (3, 3)]) on the node lattice."""
    ex, spx = _layer_walls(layer, 0)
    wx, _, xm, stx = _axis_subcells(sx, px, ex, spx)
    if sy == 1:
        # a 1-D stack (sy = 1): one pixel spanning the whole period in y, sampled at its centre
        # (every inclusion of a 1-D stack is a full-y rectangle)
        wy, ym, sty = np.array([py]), np.array([0.5 * py]), np.array([0])
    else:
        ey, spy = _layer_walls(layer, 1)
        wy, _, ym, sty = _axis_subcells(sy, py, ey, spy)
    hx, hy = px / sx, py / sy
    fx, fy = wx / hx, wy / hy                       # sub-cell weights (sum to 1 per pixel)
    tail = (3, 3) if tensor else ()
    arith = np.zeros((sx, sy) + tail, dtype=complex)
    exx = np.zeros((sx, sy), dtype=complex) if companion else None
    eyy_inv_rows = np.zeros((wx.size, sy), dtype=complex) if companion else None
    # y blocks aligned to pixel rows so each pixel row is reduced in one piece
    rows_per_block = max(1, int(_CHUNK // max(wx.size, 1)))
    jy = 0
    while jy < sy:
        k0 = sty[jy]
        jy1 = jy + 1
        while jy1 < sy and (sty[jy1] - k0) < rows_per_block:
            jy1 += 1
        k1 = sty[jy1] if jy1 < sy else wy.size
        X, Y = np.meshgrid(xm, ym[k0:k1], indexing="ij")
        V = np.asarray(paint(X, Y), dtype=complex)            # (nxs, nys_block[, 3, 3])
        wyb = fy[k0:k1]
        # pixel starts of this block's rows, relative to the block
        rel = sty[jy:jy1] - k0
        Wx = fx.reshape((-1, 1) + (1,) * len(tail))
        Wy = wyb.reshape((1, -1) + (1,) * len(tail))
        a = np.add.reduceat(np.add.reduceat(V * Wx * Wy, stx, axis=0), rel, axis=1)
        arith[:, jy:jy1] = a
        if companion:
            with np.errstate(divide="ignore", invalid="ignore"):
                inv = 1.0 / V
                hx_rows = 1.0 / np.add.reduceat(inv * fx[:, None], stx, axis=0)   # (sx, nys)
                exx[:, jy:jy1] = np.add.reduceat(hx_rows * wyb[None, :], rel, axis=1)
                eyy_inv_rows[:, jy:jy1] = np.add.reduceat(inv * wyb[None, :], rel, axis=1)
        jy = jy1
    eyy = None
    if companion:
        with np.errstate(divide="ignore", invalid="ignore"):
            eyy = np.add.reduceat((1.0 / eyy_inv_rows) * fx[:, None], stx, axis=0)
        if not (np.all(np.isfinite(exx)) and np.all(np.isfinite(eyy))):
            exx = eyy = None                          # a zero permittivity: no harmonic mean
    return arith, ((exx, eyy) if companion and exx is not None else None)


def area_weighted_cell(layer, S_x: int, S_y: int, lambda_m: float, materials, eps_by_region,
                       px: float, py: float, companion: bool = False):
    """(eps_cell, (exx, eyy) or None): the AREA-WEIGHTED raster of one Design layer on the node
    lattice, (S_x, S_y) pixels (axis 0 = x). eps_cell is the arithmetic coverage average of the
    painted permittivity (background + priority-ordered inclusions, periodic wrap, the
    optics.rasterize painter); with companion=True also the inverse-rule pair (exx, eyy) -- None
    when a sub-cell permittivity is exactly zero (no harmonic mean exists). S_y = 1 is a 1-D stack
    (y-invariant layer)."""
    from dynameta.optics.rasterize import layer_eps_cell

    def paint(X, Y):
        return layer_eps_cell(layer, X, Y, lambda_m, materials, eps_by_region,
                              period_x_m=px, period_y_m=py)
    return _average(paint, int(S_x), int(S_y), float(px), float(py), layer, bool(companion),
                    tensor=False)


def area_weighted_tensor_cell(layer, S_x: int, S_y: int, lambda_m: float, materials, bg_tensor,
                              px: float, py: float):
    """The (S_x, S_y, 3, 3) area-weighted eps_tensor_cell of a layer whose BACKGROUND is a uniform
    tensor (scalar inclusions painted over it as eps * I3: optics.rasterize.layer_eps_tensor_cell),
    on the node lattice. Tensor layers have their own factorization (no companion)."""
    from dynameta.optics.rasterize import layer_eps_tensor_cell

    def paint(X, Y):
        return layer_eps_tensor_cell(layer, X, Y, lambda_m, materials, bg_tensor,
                                     period_x_m=px, period_y_m=py)
    cell, _ = _average(paint, int(S_x), int(S_y), float(px), float(py), layer, False,
                       tensor=True)
    return cell


def _pixel_edges(n: int) -> np.ndarray:
    """Period fractions [0, 1/(2n), 3/(2n), ..., 1 - 1/(2n), 1]: the node-lattice pixel walls
    folded into [0, 1] (pixel 0 is the first and the last interval together)."""
    return np.concatenate([[0.0], (np.arange(n) + 0.5) / n, [1.0]])


def _resample_axis(values, x_m, period_m: float, n: int, where: str):
    """Area means over the n node-lattice pixels of the periodic piecewise-linear interpolant
    through `values` (..., Nx) at positions x_m. Returns (arith, harm), each (..., n); harm is
    None when the interpolant passes through zero (no harmonic mean)."""
    v = np.asarray(values, dtype=complex)
    if v.shape[-1] == 1:                    # one sample: the profile is constant along this axis
        rep = np.repeat(v, n, axis=-1)
        return rep, rep.copy()
    xs, vs = periodic_samples(x_m, v, period_m, where=where)
    edges = _pixel_edges(n)
    try:
        a, hm, _ = interval_means(xs, vs, edges, period_m)
    except ValueError:
        a, hm = None, None
    if a is None:
        # the harmonic piece refuses a lossless sign change; the arithmetic mean alone is still
        # exact -- recompute it with the harmonic part on a shifted copy that cannot fail
        a = _arith_only(xs, vs, edges, period_m)
    # fold the two half-pixels at x = 0 and x = P into pixel 0
    arith = a[..., :-1].copy()
    arith[..., 0] = 0.5 * (a[..., 0] + a[..., -1])
    if hm is None:
        return arith, None
    harm = hm[..., :-1].copy()
    harm[..., 0] = 2.0 / (1.0 / hm[..., 0] + 1.0 / hm[..., -1])
    return arith, harm


def _arith_only(xs, vs, edges, period_m):
    """interval_means' arithmetic mean when its harmonic part cannot be formed: the arithmetic
    mean is linear, so mean(v) = mean(v + c) - c for a constant c that moves every sample away
    from zero."""
    c = 1.0 + 2.0 * float(np.max(np.abs(vs)))
    a, _, _ = interval_means(xs, vs + c, edges, period_m)
    return a - c


def area_resample_cell(cell, x_m, y_m, S_x: int, S_y: int, px: float, py: float, *,
                       companion: bool = False, where: str = "gridded EpsField"):
    """Resample a sampled lateral profile onto the node lattice by AREA averaging.

    cell : (Nx, Ny[, 3, 3]) samples (axis 0 = x) at positions x_m (Nx,) and y_m (Ny,) in metres
           inside the cell; Ny == 1 (or S_y == 1) is a y-invariant profile.
    Every axis is read as the periodic piecewise-linear interpolant of its samples (an
    endpoint-inclusive grid's duplicated end sample costs nothing; any grid convention works);
    each pixel takes the exact mean of that interpolant over its cell [x_j - h/2, x_j + h/2].
    Returns (eps_cell (S_x, S_y[, 3, 3]), (exx, eyy) or None). The companion (scalar cells only)
    is exx = harmonic along x of each sampled y row, then the y area mean; eyy = harmonic along y
    of each sampled x column, then the x area mean (exact for a y-invariant profile; second order
    in the y spacing otherwise). None when the interpolant passes through zero."""
    c = np.asarray(cell, dtype=complex)
    tensor = c.ndim == 4
    x = np.asarray(x_m, dtype=np.float64).ravel()
    y = np.asarray(y_m, dtype=np.float64).ravel()
    if c.shape[0] != x.size:
        raise ValueError("{}: {} x samples but {} x positions".format(where, c.shape[0], x.size))
    if tensor:
        flat = c.reshape(c.shape[0], c.shape[1], 9)
        out = np.empty((int(S_x), int(S_y), 9), dtype=complex)
        for k in range(9):
            out[..., k], _ = area_resample_cell(flat[..., k], x, y, S_x, S_y, px, py,
                                                companion=False, where=where)
        return out.reshape(int(S_x), int(S_y), 3, 3), None
    y_inv = c.shape[1] == 1 or int(S_y) == 1
    if y_inv and c.shape[1] > 1:
        c = c[:, :1]
    # x first: rows (Ny, Nx) -> (Ny, S_x)
    ax, hx = _resample_axis(c.T, x, px, int(S_x), where + " (x axis)")
    if y_inv:
        arith = np.broadcast_to(ax.T, (int(S_x), int(S_y))).copy()
        if not companion or hx is None:
            return arith, None
        exx = np.broadcast_to(hx.T, (int(S_x), int(S_y))).copy()
        return arith, (exx, arith.copy())
    if y.size != c.shape[1]:
        raise ValueError("{}: {} y samples but {} y positions".format(where, c.shape[1], y.size))
    arith, _ = _resample_axis(ax.T, y, py, int(S_y), where + " (y axis)")     # (S_x, S_y)
    if not companion or hx is None:
        return arith, None
    exx, _ = _resample_axis(hx.T, y, py, int(S_y), where + " (y axis)")
    ay, hy = _resample_axis(c, y, py, int(S_y), where + " (y axis)")          # (Nx, S_y)
    if hy is None:
        return arith, None
    eyy, _ = _resample_axis(hy.T, x, px, int(S_x), where + " (x axis)")       # (S_y, S_x)
    return arith, (exx, eyy.T)


def realised_coverage(eps_cell, eps_background: complex, eps_inclusion: complex) -> np.ndarray:
    """Per-pixel covered fraction of a two-material raster: (cell - bg) / (incl - bg). Its sum
    times P / S is the realised width (1-D) or area / P (2-D: times Px Py / (Sx Sy))."""
    return ((np.asarray(eps_cell, dtype=complex) - complex(eps_background))
            / (complex(eps_inclusion) - complex(eps_background))).real
