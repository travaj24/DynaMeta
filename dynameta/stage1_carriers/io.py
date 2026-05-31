"""
Stage 1 I/O: dump the carrier field at one bias to a Zarr store, and
load it back. The schema includes both raw unstructured node data and a
regular-grid resample of the ITO (and any other semiconductor) for
downstream Stage 2 / Stage 3 consumption.
"""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import zarr
import devsim as ds

from dynameta.design import BiasPoint, Design
from dynameta.stage1_carriers.devsim_build import Stage1BuildResult


# ---------------------------------------------------------------------------
# Resample helper
# ---------------------------------------------------------------------------

def _resample_region_to_grid(x: np.ndarray, y: np.ndarray,
                                values: Dict[str, np.ndarray],
                                n_x: int = 128, n_y: int = 16,
                                ) -> Dict[str, np.ndarray]:
    """Resample unstructured (x, y) -> regular (n_x, n_y) grid via
    LinearNDInterpolator with NearestNDInterpolator fallback for the
    convex hull's exterior.
    """
    from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
    points = np.column_stack([x, y])
    x_axis = np.linspace(x.min(), x.max(), n_x)
    y_axis = np.linspace(y.min(), y.max(), n_y)
    XX, YY = np.meshgrid(x_axis, y_axis, indexing="ij")
    qpts   = np.column_stack([XX.ravel(), YY.ravel()])

    out: Dict[str, np.ndarray] = {"x_axis_m": x_axis, "y_axis_m": y_axis}
    for key, vals in values.items():
        lin = LinearNDInterpolator(points, vals, fill_value=np.nan)
        grid = lin(qpts).reshape(n_x, n_y)
        nan_mask = np.isnan(grid)
        if np.any(nan_mask):
            nn = NearestNDInterpolator(points, vals)
            grid[nan_mask] = nn(qpts[nan_mask.ravel()])
        out[key] = grid
    return out


# ---------------------------------------------------------------------------
# Dump
# ---------------------------------------------------------------------------

def dump_carrier_field(device: str, design: Design,
                         build: Stage1BuildResult,
                         bp: BiasPoint, out_path: Path,
                         *, grid_n_x: int = 512, grid_n_y: int = 32
                         ) -> Path:
    """Write the full per-region carrier field at this bias to a Zarr
    store. Schema:

        /regions/<L.name>/{x_m, y_m, Potential, Electrons}
        /grid/<L.name>/{x_axis_m, y_axis_m, Electrons, Potential}
            (semiconductor layers only; regular resample for NGSolve)
        .attrs/{
            bias_label, voltages_json,
            design_name, period_m, patch_side_m, n_bg_m3, T_K,
            geometry_dim=2, created_iso,
        }
    """
    import json
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    root = zarr.open_group(str(out_path), mode="w")
    gp_regions = root.create_group("regions")
    gp_grid    = root.create_group("grid")

    for L in design.layers:
        gp = gp_regions.create_group(L.name)
        x = np.asarray(ds.get_node_model_values(device=device, region=L.name,
                                                   name="x"))
        y = np.asarray(ds.get_node_model_values(device=device, region=L.name,
                                                   name="y"))
        gp["x_m"] = np.ascontiguousarray(x, dtype=np.float64)
        gp["y_m"] = np.ascontiguousarray(y, dtype=np.float64)
        gp.attrs["material"] = L.material
        gp.attrs["role"]     = L.role
        # Potential (every layer that has it)
        try:
            V = np.asarray(ds.get_node_model_values(device=device, region=L.name,
                                                       name="Potential"))
            gp["Potential"] = np.ascontiguousarray(V, dtype=np.float64)
        except Exception:
            pass
        # Electrons (semiconductor only)
        if L.role == "semiconductor":
            n = np.asarray(ds.get_node_model_values(device=device,
                                                       region=L.name,
                                                       name="Electrons"))
            gp["Electrons"] = np.ascontiguousarray(n, dtype=np.float64)
            # Regular-grid resample (NGSolve-consumable)
            grid = _resample_region_to_grid(x, y,
                                              {"Potential": V, "Electrons": n},
                                              n_x=grid_n_x, n_y=grid_n_y)
            gp_layer_grid = gp_grid.create_group(L.name)
            for key, arr in grid.items():
                gp_layer_grid[key] = np.ascontiguousarray(arr, dtype=np.float64)
            gp_layer_grid.attrs["Nx"] = int(grid["x_axis_m"].size)
            gp_layer_grid.attrs["Ny"] = int(grid["y_axis_m"].size)

    # Top-level metadata
    root.attrs["design_name"]     = design.name
    root.attrs["bias_label"]      = bp.label
    root.attrs["voltages_json"]   = json.dumps(bp.voltages)
    root.attrs["period_m"]        = float(design.period_m)
    root.attrs["patch_side_m"]    = float(design.patch_side_m)
    semis = [L for L in design.layers if L.role == "semiconductor"]
    if semis:
        root.attrs["n_bg_m3_by_layer"] = json.dumps({
            L.name: float(design.materials.get(L.material).drude.n_bg_m3)
            for L in semis})
    root.attrs["T_K"]             = 300.0
    root.attrs["geometry_dim"]    = 2
    root.attrs["created_iso"]     = datetime.datetime.now().isoformat()
    root.attrs["layer_order"]     = [L.name for L in design.layers]
    return out_path
