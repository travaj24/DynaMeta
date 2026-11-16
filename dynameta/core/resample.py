"""
Resample an unstructured field (DEVSIM nodes) onto a regular grid
(resample_to_grid), or a 2-D tensor-grid node set onto its native y lines with
a uniform x axis (resample_tensor_grid). Pure numpy + scipy; no solver imports. ndim-general (2D or 3D nodes), generalizing the
old stage1_carriers/io.py:_resample_region_to_grid (which was 2D-only and
baked into Stage 1). The bridge owns resampling now, so a bring-your-own
carrier solver can hand over raw nodes and let the bridge grid them.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

# Node coordinates [m] are binned onto this lattice to decide whether a node set is a tensor grid
# (DEVSIM writes each mesh-line coordinate once, so equal lines agree to the last bit; the bin only
# absorbs a representation difference far below any mesh spacing).
TENSOR_GRID_ROUND_M = 1e-15


class NotATensorGridError(ValueError):
    """resample_tensor_grid: the node set is not the full tensor product of its unique x and y
    coordinates (e.g. a region whose outline is not a rectangle of mesh lines)."""


def resample_tensor_grid(x_nodes, y_nodes, fields: Dict[str, np.ndarray],
                         grid_n_x: int) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Grid scalar node fields of a 2-D TENSOR-GRID node set on its NATIVE y lines, uniform in x.

    Args:
      x_nodes, y_nodes : (N,) node coordinates [m]
      fields           : {name: (N,)} scalar values at the nodes
      grid_n_x         : number of uniform x points spanning [min x, max x]

    Returns (axes, gridded): axes = {"x": (grid_n_x,) linspace, "y": (Ny,) the unique node y, ascending,
    as stored at the nodes}; gridded = {name: (grid_n_x, Ny)} -- the layout of resample_to_grid with
    n_per_axis = (grid_n_x, Ny). Each native y line is interpolated linearly along x (np.interp over that
    line's nodes), so no value is interpolated across y and a grid point on a node column (always the
    two x end points) carries the node value itself, bit for bit. That is what a through-stack (z) consumer needs from a graded mesh: the
    resolution the carrier solver used, with no resampling error (a uniform resample of a 0.0125 nm
    graded mesh onto 32 points misplaces a 1 nm accumulation layer).

    The node set must be the full tensor product of its unique x and unique y coordinates (after binning
    to TENSOR_GRID_ROUND_M): len(unique x) * len(unique y) == N with every pair present once. Otherwise
    NotATensorGridError (a ValueError) is raised; resample_to_grid handles general node sets."""
    x = np.asarray(x_nodes, dtype=np.float64).ravel()
    y = np.asarray(y_nodes, dtype=np.float64).ravel()
    n = x.size
    if y.size != n or n == 0:
        raise ValueError("resample_tensor_grid: x_nodes and y_nodes must be non-empty and the same "
                         "length (got {} and {})".format(x.size, y.size))
    if int(grid_n_x) < 1:
        raise ValueError("resample_tensor_grid: grid_n_x must be >= 1, got {!r}".format(grid_n_x))
    kx = np.round(x / TENSOR_GRID_ROUND_M).astype(np.int64)
    ky = np.round(y / TENSOR_GRID_ROUND_M).astype(np.int64)
    ux, ix0, ix = np.unique(kx, return_index=True, return_inverse=True)
    uy, iy0, iy = np.unique(ky, return_index=True, return_inverse=True)
    flat = ix * uy.size + iy
    if ux.size * uy.size != n or np.unique(flat).size != n:
        raise NotATensorGridError(
            "resample_tensor_grid: {} nodes are not a tensor grid ({} unique x x {} unique y = {}, "
            "or a coordinate pair repeats)".format(n, ux.size, uy.size, ux.size * uy.size))
    xs = x[ix0]                                            # native coordinates, ascending
    ys = y[iy0]
    xg = np.linspace(xs[0], xs[-1], int(grid_n_x))
    out: Dict[str, np.ndarray] = {}
    for name, vals in fields.items():
        v = np.asarray(vals, dtype=np.float64).ravel()
        if v.size != n:
            raise ValueError("resample_tensor_grid: field {!r} has {} values for {} nodes".format(
                name, v.size, n))
        tab = np.empty(ux.size * uy.size, dtype=np.float64)
        tab[flat] = v
        tab = tab.reshape(ux.size, uy.size)                  # (Nx_native, Ny)
        if ux.size == 1:
            out[name] = np.repeat(tab, int(grid_n_x), axis=0)
        else:
            out[name] = np.stack([np.interp(xg, xs, tab[:, j]) for j in range(uy.size)], axis=1)
    return {"x": xg, "y": ys}, out


def resample_to_grid(nodes_m: np.ndarray,
                       fields: Dict[str, np.ndarray],
                       n_per_axis: Sequence[int]) -> Dict[str, np.ndarray]:
    """Interpolate scalar node fields onto a regular grid spanning the node
    bounding box.

    Args:
      nodes_m   : (N, ndim) node coordinates (SI metres), ndim in {2, 3}
      fields    : {name: (N,)} scalar values at the nodes
      n_per_axis: grid resolution per axis, length == ndim

    Returns dict with "axis_0".."axis_{ndim-1}" (1D coordinate arrays) and each
    input field name -> gridded array of shape tuple(n_per_axis). Linear
    interpolation with nearest-neighbour fill for points outside the convex
    hull (matches the old behaviour).
    """
    from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

    nodes = np.asarray(nodes_m, dtype=np.float64)
    n_nodes, ndim = nodes.shape
    if len(n_per_axis) != ndim:
        raise ValueError("n_per_axis length {} != node ndim {}".format(
            len(n_per_axis), ndim))

    axes: List[np.ndarray] = [
        np.linspace(nodes[:, d].min(), nodes[:, d].max(), int(n_per_axis[d]))
        for d in range(ndim)
    ]
    mesh = np.meshgrid(*axes, indexing="ij")
    qpts = np.column_stack([m.ravel() for m in mesh])

    out: Dict[str, np.ndarray] = {}
    for d in range(ndim):
        out["axis_{}".format(d)] = axes[d]

    # Build the Delaunay triangulation ONCE and reuse it for every field (it depends only on the
    # node coordinates, not the values) -- avoids re-triangulating per field. Bit-identical: a
    # LinearNDInterpolator built from this tri interpolates exactly as one built from `nodes`.
    try:
        from scipy.spatial import Delaunay
        tri = Delaunay(nodes)
    except Exception:
        tri = None                                            # degenerate/collinear -> let scipy handle

    # The nearest-neighbour fill is hoisted the same way (audit P-12): NearestNDInterpolator(nodes,
    # vals) built a fresh cKDTree over the SAME node coordinates for every field, and queried it
    # for the SAME outside-the-hull points (the NaN mask depends on the geometry, not the values,
    # unless a field itself carries NaN -- then the mask differs and is re-queried).  One tree,
    # one query per distinct mask; `vals[index]` is exactly what NearestNDInterpolator returns.
    tree = None
    nn_mask = None
    nn_idx = None
    for name, vals in fields.items():
        vals = np.asarray(vals, dtype=np.float64)
        lin = LinearNDInterpolator(tri if tri is not None else nodes, vals)
        grid = lin(qpts).reshape(tuple(int(n) for n in n_per_axis))
        nan = np.isnan(grid)
        if nan.any():
            if tree is None:
                tree = NearestNDInterpolator(nodes, np.zeros(n_nodes)).tree
            mask = nan.ravel()
            if nn_mask is None or not np.array_equal(mask, nn_mask):
                nn_mask = mask
                _, nn_idx = tree.query(qpts[mask])
            grid[nan] = vals[nn_idx]
        out[name] = grid
    return out
