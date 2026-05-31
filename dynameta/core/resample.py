"""
Resample an unstructured field (DEVSIM nodes) onto a regular grid. Pure numpy
+ scipy; no solver imports. ndim-general (2D or 3D nodes), generalizing the
old stage1_carriers/io.py:_resample_region_to_grid (which was 2D-only and
baked into Stage 1). The bridge owns resampling now, so a bring-your-own
carrier solver can hand over raw nodes and let the bridge grid them.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np


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

    for name, vals in fields.items():
        vals = np.asarray(vals, dtype=np.float64)
        lin = LinearNDInterpolator(nodes, vals)
        grid = lin(qpts).reshape(tuple(int(n) for n in n_per_axis))
        nan = np.isnan(grid)
        if nan.any():
            nn = NearestNDInterpolator(nodes, vals)
            grid[nan] = nn(qpts[nan.ravel()])
        out[name] = grid
    return out
