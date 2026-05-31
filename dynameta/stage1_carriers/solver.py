"""
Stage 1 DC solver: walk the bias-point list, Newton-ramp each electrode
to its target voltage, dump the carrier field after each.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

import devsim as ds

from dynameta.design import Design, BiasPoint, Sweep
from dynameta.stage1_carriers import io as stage1_io
from dynameta.stage1_carriers.devsim_build import Stage1BuildResult


def _initial_zero_bias(device: str, design: Design,
                         build: Stage1BuildResult) -> None:
    """Initialise carrier density to the equilibrium n_bg, then solve at
    every contact = 0 V."""
    # Set all contact biases to 0 (or to fixed_voltage_V for grounds)
    for E in design.electrodes:
        v = E.fixed_voltage_V if E.role == "ground" else 0.0
        ds.set_parameter(device=device, name="{}_bias".format(E.name), value=v)
    # Electrons is a DERIVED node model (n = N_c*F_1/2(Potential)), not a
    # solution variable, so there is nothing to initialise for it -- the build
    # seeds Potential = 0 (equilibrium, since Phi_c0 is calibrated so n = n_bg
    # at V = 0). Just solve.
    ds.solve(type="dc", absolute_error=1e10, relative_error=1e-5,
              maximum_iterations=60)


def _ramp_to_bias(device: str, design: Design, bp: BiasPoint,
                    current_voltages: Dict[str, float],
                    voltage_step_m: float, rel_tol: float,
                    max_iter: int) -> int:
    """Ramp each electrode from its current value to the target in bp.
    Returns total Newton steps taken."""
    total_steps = 0
    for E in design.electrodes:
        target = bp.voltages.get(E.name)
        if target is None:
            target = E.fixed_voltage_V if E.role == "ground" else 0.0
        v_now = current_voltages.get(E.name, 0.0)
        if abs(target - v_now) < 1e-12:
            continue
        n_steps = max(1, int(abs(target - v_now) / voltage_step_m + 0.5))
        dV = (target - v_now) / n_steps
        for _ in range(n_steps):
            v_now += dV
            ds.set_parameter(device=device, name="{}_bias".format(E.name),
                              value=v_now)
            ds.solve(type="dc", absolute_error=1e10, relative_error=rel_tol,
                      maximum_iterations=max_iter)
            total_steps += 1
        current_voltages[E.name] = target
    return total_steps


def run_stage1(design: Design, sweep: Sweep,
                 out_dir: Path,
                 *, verbose: bool = True,
                 progress_cb: Optional[Callable[[str, float], None]] = None
                 ) -> List[Path]:
    """Solve Stage 1 at every BiasPoint and dump a Zarr per point.

    Returns the list of output Zarr paths.
    """
    from dynameta.stage1_carriers.devsim_build import build_devsim_device
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print("[stage1] building DEVSIM device for design '{}'...".format(design.name),
               flush=True)
    t0 = time.time()
    build = build_devsim_device(design)
    if verbose:
        print("[stage1]   build ok in {:.1f} s ({} regions, {} contacts, {} interfaces)".format(
            time.time() - t0,
            len(ds.get_region_list(device=build.device)),
            len(build.actual_contacts),
            len(build.actual_interfaces)))

    if verbose:
        print("[stage1] zero-bias initial solve...", flush=True)
    _initial_zero_bias(build.device, design, build)

    current_voltages: Dict[str, float] = {
        E.name: (E.fixed_voltage_V if E.role == "ground" else 0.0)
        for E in design.electrodes
    }
    written: List[Path] = []
    n_total = len(sweep.bias_points)
    for i, bp in enumerate(sweep.bias_points):
        t0 = time.time()
        n_steps = _ramp_to_bias(build.device, design, bp,
                                  current_voltages,
                                  sweep.voltage_step_m,
                                  sweep.rel_tol,
                                  sweep.max_iter_dc)
        out_path = out_dir / "carrier_field_{}.zarr".format(bp.label)
        stage1_io.dump_carrier_field(build.device, design, build,
                                        bp, out_path)
        written.append(out_path)
        if verbose:
            print("[stage1]   bias '{}': {} Newton steps, {:.1f} s, dumped {}".format(
                bp.label, n_steps, time.time() - t0, out_path.name), flush=True)
        if progress_cb is not None:
            progress_cb(bp.label, (i + 1) / n_total)

    # Tear down so subsequent calls can rebuild
    for d in list(ds.get_device_list()):
        ds.delete_device(device=d)
    for m in list(ds.get_mesh_list()):
        ds.delete_mesh(mesh=m)
    return written
