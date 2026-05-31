"""
Stage 3 driver: build the 3D mesh ONCE, loop over BiasPoints and
wavelengths, return per-solve reflectivity / phase / time.
"""

from __future__ import annotations

import csv
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np

from dynameta.design import Design, Sweep
from dynameta.stage3_optical.ngsolve_build import build_unit_cell_3d
from dynameta.stage3_optical.eps_loader import build_eps_cf_at_bias_lambda
from dynameta.stage3_optical.solver import solve_fem, FEMResult


@dataclass
class SweepRow:
    bias_label:   str
    voltages:     dict
    lambda_nm:    float
    r_re:         float
    r_im:         float
    abs_r2:       float
    phase_deg:    float
    solve_time_s: float


def run_stage3(design: Design, sweep: Sweep,
                 eps_zarrs_by_bias: dict,
                 out_dir: Path,
                 *, verbose: bool = True,
                 progress_cb: Optional[Callable[[str, float, float], None]] = None
                 ) -> List[SweepRow]:
    """Stage 3 wavelength sweep across all BiasPoints.

    Args:
      eps_zarrs_by_bias : {bias_label -> Path(eps_<bias_label>.zarr)} from Stage 2
      out_dir           : where to write spectra.csv (incremental, per-bias)

    Returns:
      flat list of SweepRow records (one per (bias, lambda))
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print("[stage3] building 3D NGSolve mesh for design '{}'...".format(
            design.name), flush=True)
    t0 = time.time()
    geo = build_unit_cell_3d(design)
    if verbose:
        print("[stage3]   mesh ok in {:.1f} s (ne = {}, nv = {})".format(
            time.time() - t0, geo.mesh.ne, geo.mesh.nv))

    rows: List[SweepRow] = []
    total_solves = len(sweep.bias_points) * len(sweep.wavelengths_nm)
    n_done = 0
    overall_t0 = time.time()

    csv_path = out_dir / "spectra.csv"
    fieldnames = ["bias_label", "voltages", "lambda_nm",
                   "r_re", "r_im", "abs_r2", "phase_deg", "solve_time_s"]

    for bp in sweep.bias_points:
        if bp.label not in eps_zarrs_by_bias:
            if verbose:
                print("[stage3] WARNING: no Stage-2 file for bias '{}' -- "
                       "skipping".format(bp.label))
            continue
        eps_zarr = eps_zarrs_by_bias[bp.label]
        if verbose:
            print("[stage3] bias '{}' ({}):".format(bp.label,
                    {k: "{:+.2f}V".format(v) for k, v in bp.voltages.items()}),
                   flush=True)
        for lam_nm in sweep.wavelengths_nm:
            lam_m = float(lam_nm) * 1e-9
            t_eps = time.time()
            eps_cf = build_eps_cf_at_bias_lambda(geo, design, eps_zarr,
                                                   float(lam_nm), design.optical)
            t_solve_start = time.time()
            try:
                res: FEMResult = solve_fem(geo, lam_m, eps_cf,
                                              design.optical,
                                              order=design.mesh_3d.fem_order,
                                              verbose=False)
                n_done += 1
                elapsed = time.time() - overall_t0
                eta_s = elapsed * (total_solves - n_done) / n_done \
                          if n_done > 0 else 0
                rows.append(SweepRow(
                    bias_label=bp.label, voltages=bp.voltages,
                    lambda_nm=float(lam_nm),
                    r_re=float(res.r.real), r_im=float(res.r.imag),
                    abs_r2=float(res.R),
                    phase_deg=float(np.degrees(np.angle(res.r))),
                    solve_time_s=float(res.solve_time_s),
                ))
                if verbose:
                    print("[stage3]   lam={:5.0f} nm  |r|^2={:.4f}  "
                           "phase={:+7.2f}  solve={:.1f}s  ETA={:.1f} min"
                           .format(lam_nm, res.R,
                                    np.degrees(np.angle(res.r)),
                                    res.solve_time_s,
                                    eta_s / 60.0), flush=True)
                if progress_cb is not None:
                    progress_cb(bp.label, float(lam_nm), n_done / total_solves)
            except Exception as e:
                if verbose:
                    print("[stage3]   lam={:5.0f} nm  FAILED: {}".format(
                        lam_nm, repr(e)[:80]), flush=True)
        # Incremental CSV write after each bias
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in rows:
                w.writerow({"bias_label": r.bias_label,
                              "voltages": "; ".join("{}={:+.3f}".format(k, v)
                                                      for k, v in r.voltages.items()),
                              "lambda_nm": r.lambda_nm,
                              "r_re": r.r_re, "r_im": r.r_im,
                              "abs_r2": r.abs_r2, "phase_deg": r.phase_deg,
                              "solve_time_s": r.solve_time_s})
    if verbose:
        print("[stage3] done. {} rows -> {}".format(len(rows), csv_path))
    return rows
