"""
CarrierField: the solver-agnostic carrier-solve result that flows from Stage 1
into the bridge. Schema v2 (generalizes the old /regions+/grid Zarr):
  - dimension-agnostic (ndim 2 or 3; axes named x/y[/z])
  - mode discriminated by PRESENCE OF FIELD KEYS, not a flag:
      equilibrium -> {potential_V, electron_density_m3}
      drift-diff  -> + {hole_density_m3, electron_qfl_V, hole_qfl_V,
                        Jn_A_m2, Jp_A_m2, recomb_rate_m3_s}
  - carries unit_cell (period_x/y), NOT patch_side; time_convention on eps store.

Field names are the contract. Stage 2/3 read electron_density_m3 (+ optionally
hole_density_m3) and validate field_vocab on entry, failing loudly on mismatch.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

SCHEMA_VERSION = 2
ELECTRON_DENSITY = "electron_density_m3"
POTENTIAL = "potential_V"


@dataclass
class CarrierRegion:
    name:         str
    role:         str
    material:     str
    nodes_m:      np.ndarray                       # (N, ndim) SI
    node_fields:  Dict[str, np.ndarray]            # name -> (N,) or (N, ndim)
    grid_axes_m:  Optional[Dict[str, np.ndarray]] = None    # {"x":..,"y":..,["z":..]}
    grid_fields:  Optional[Dict[str, np.ndarray]] = None    # name -> gridded array


@dataclass
class CarrierField:
    bias_label:      str
    voltages:        Dict[str, float]
    ndim:            int
    temperature_K:   float
    regions:         Dict[str, CarrierRegion]
    n_bg_by_region:  Dict[str, float]
    unit_cell_m:     Tuple[float, float]            # (period_x, period_y)
    time_convention: str = "exp(-iwt)"
    extras:          Dict[str, object] = field(default_factory=dict)

    def field_vocab(self) -> List[str]:
        vocab = set()
        for r in self.regions.values():
            vocab.update(r.node_fields)
            if r.grid_fields:
                vocab.update(r.grid_fields)
        return sorted(vocab)


def dump_carrier_field(field: CarrierField, path: Path) -> Path:
    import zarr
    path = Path(path)
    root = zarr.open_group(str(path), mode="w")
    root.attrs["schema_version"]      = SCHEMA_VERSION
    root.attrs["ndim"]                = int(field.ndim)
    root.attrs["bias_label"]          = field.bias_label
    root.attrs["voltages_json"]       = json.dumps(field.voltages)
    root.attrs["temperature_K"]       = float(field.temperature_K)
    root.attrs["n_bg_by_region_json"] = json.dumps(field.n_bg_by_region)
    root.attrs["unit_cell_json"]      = json.dumps(
        {"period_x_m": field.unit_cell_m[0], "period_y_m": field.unit_cell_m[1]})
    root.attrs["field_vocab_json"]    = json.dumps(field.field_vocab())
    root.attrs["time_convention"]     = field.time_convention
    root.attrs["extras_json"]         = json.dumps(
        {k: v for k, v in field.extras.items()})

    g_reg = root.create_group("regions")
    g_grid = root.create_group("grid")
    for name, reg in field.regions.items():
        gr = g_reg.create_group(name)
        gr["nodes_m"] = np.ascontiguousarray(reg.nodes_m, dtype=np.float64)
        gf = gr.create_group("fields")
        for fname, vals in reg.node_fields.items():
            gf[fname] = np.ascontiguousarray(vals, dtype=np.float64)
        gr.attrs["material"] = reg.material
        gr.attrs["role"] = reg.role
        if reg.grid_axes_m is not None and reg.grid_fields is not None:
            gg = g_grid.create_group(name)
            ga = gg.create_group("axes")
            for ax, vals in reg.grid_axes_m.items():
                ga[ax] = np.ascontiguousarray(vals, dtype=np.float64)
            gfd = gg.create_group("fields")
            for fname, arr in reg.grid_fields.items():
                gfd[fname] = np.ascontiguousarray(arr, dtype=np.float64)
            gg.attrs["axis_order"] = json.dumps(list(reg.grid_axes_m.keys()))
    return path


def load_carrier_field(path: Path) -> CarrierField:
    import zarr
    root = zarr.open_group(str(Path(path)), mode="r")
    sv = int(root.attrs.get("schema_version", -1))
    if sv != SCHEMA_VERSION:
        raise ValueError("CarrierField schema_version {} != expected {} ({})".format(
            sv, SCHEMA_VERSION, path))
    uc = json.loads(root.attrs["unit_cell_json"])
    regions: Dict[str, CarrierRegion] = {}
    grid_group = root["grid"] if "grid" in root.group_keys() else None
    for name in root["regions"].group_keys():
        gr = root["regions"][name]
        node_fields = {f: np.asarray(gr["fields"][f][:])
                        for f in gr["fields"].array_keys()}
        grid_axes = grid_fields = None
        if grid_group is not None and name in grid_group.group_keys():
            gg = grid_group[name]
            grid_axes = {a: np.asarray(gg["axes"][a][:])
                          for a in gg["axes"].array_keys()}
            grid_fields = {f: np.asarray(gg["fields"][f][:])
                            for f in gg["fields"].array_keys()}
        regions[name] = CarrierRegion(
            name=name, role=str(gr.attrs.get("role", "")),
            material=str(gr.attrs.get("material", "")),
            nodes_m=np.asarray(gr["nodes_m"][:]),
            node_fields=node_fields, grid_axes_m=grid_axes, grid_fields=grid_fields)
    return CarrierField(
        bias_label=str(root.attrs["bias_label"]),
        voltages=json.loads(root.attrs["voltages_json"]),
        ndim=int(root.attrs["ndim"]),
        temperature_K=float(root.attrs["temperature_K"]),
        regions=regions,
        n_bg_by_region=json.loads(root.attrs["n_bg_by_region_json"]),
        unit_cell_m=(uc["period_x_m"], uc["period_y_m"]),
        time_convention=str(root.attrs.get("time_convention", "exp(-iwt)")),
        extras=json.loads(root.attrs.get("extras_json", "{}")),
    )
