"""
Stage 2 driver: read each Stage 1 carrier Zarr, apply Drude at every
wavelength, write a per-bias eps Zarr containing all wavelengths.
"""

from __future__ import annotations

import datetime
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import zarr

from dynameta.design import Design, Sweep
from dynameta.stage2_drude.drude import drude_eps


def run_stage2(design: Design, sweep: Sweep,
                 carrier_zarrs: List[Path],
                 out_dir: Path,
                 *, verbose: bool = True) -> List[Path]:
    """Apply Drude on each Stage 1 Zarr in `carrier_zarrs` for every
    wavelength in `sweep`. Returns the list of output Zarr paths.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []

    semi_layers = design.semiconductor_layers()
    if not semi_layers:
        if verbose:
            print("[stage2] no semiconductor layers in design '{}' -- nothing "
                   "to do".format(design.name))
        return []

    for in_path in carrier_zarrs:
        in_path = Path(in_path)
        root_in = zarr.open_group(str(in_path), mode="r")
        bias_label = str(root_in.attrs["bias_label"])
        if verbose:
            print("[stage2] {} ...".format(in_path.name), flush=True)
        out_path = out_dir / "eps_{}.zarr".format(bias_label)
        root_out = zarr.open_group(str(out_path), mode="w")

        # Top-level metadata mirrored from input
        for key in ("design_name", "bias_label", "voltages_json",
                     "period_m", "patch_side_m", "geometry_dim",
                     "n_bg_m3_by_layer", "layer_order", "T_K"):
            if key in root_in.attrs:
                root_out.attrs[key] = root_in.attrs[key]
        root_out.attrs["lambda_nm_list"] = [float(v) for v in sweep.wavelengths_nm]
        root_out.attrs["created_iso"]    = datetime.datetime.now().isoformat()

        for L in semi_layers:
            if L.name not in root_in["grid"].group_keys():
                # Layer not gridded (e.g. design only has 2D bulk view) -- skip
                continue
            g_in = root_in["grid"][L.name]
            x_axis_m = np.asarray(g_in["x_axis_m"][:])
            y_axis_m = np.asarray(g_in["y_axis_m"][:])
            n_grid   = np.asarray(g_in["Electrons"][:])

            gp_layer = root_out.create_group(L.name)
            gp_layer["x_axis_m"] = np.ascontiguousarray(x_axis_m,
                                                            dtype=np.float64)
            gp_layer["y_axis_m"] = np.ascontiguousarray(y_axis_m,
                                                            dtype=np.float64)
            gp_layer["n_m3"]     = np.ascontiguousarray(n_grid,
                                                            dtype=np.float64)
            gp_lams = gp_layer.create_group("lambdas")
            mat = design.materials.get(L.material)
            for lam_nm in sweep.wavelengths_nm:
                lam_m = float(lam_nm) * 1e-9
                eps = drude_eps(
                    n_m3=n_grid, lambda_m=lam_m,
                    eps_inf=mat.drude.eps_inf,
                    m_eff_kg=mat.drude.optical_mass_fn(),  # optical (not DOS) mass
                    gamma_rad_s=mat.drude.gamma_rad_s_of_n_m3,
                )
                gp = gp_lams.create_group("{:.0f}".format(lam_nm))
                gp["eps_re"] = np.ascontiguousarray(np.real(eps),
                                                       dtype=np.float64)
                gp["eps_im"] = np.ascontiguousarray(np.imag(eps),
                                                       dtype=np.float64)
                gp.attrs["lambda_nm"] = float(lam_nm)
                gp.attrs["lambda_m"]  = lam_m

        if verbose:
            print("[stage2]   -> {}".format(out_path.name), flush=True)
        written.append(out_path)
    return written
