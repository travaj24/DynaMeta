"""
Bring-your-own CarrierSolver (Phase 5 pluggable example).

Demonstrates that Stage 1 is a swappable seam: any object satisfying the
`CarrierSolver` Protocol (`regions()` + `solve(bias) -> CarrierField`) can feed
the Drude bridge + NGSolve optics, with NO DEVSIM involved. Here the "solver" is
a closed-form Thomas-Fermi-style accumulation model -- but it could equally be an
ML surrogate, a different TCAD tool, or measured C-V-derived profiles.

The only contract: emit a CarrierField whose region name(s) match the optical
mesh (here "ito") with a gridded `electron_density_m3` field on a (x, y) grid
where "y" is the through-stack axis. The bridge handles the affine z-remap, the
lift to 3D, and n -> eps.

Run:
    python -m examples.byo_carrier_solver           # fast self-check (no FEM)
    python -m examples.byo_carrier_solver --fem     # full pipeline, 1 wavelength
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dynameta.core.carrier_field import (
    CarrierField, CarrierRegion, ELECTRON_DENSITY, POTENTIAL)
from dynameta.core.interfaces import CarrierSolver, RegionInfo
from dynameta.sweep import Sweep, BiasPoint
from examples.park_2021 import build_park_design


class AnalyticAccumulationSolver:
    """A CarrierSolver that injects a closed-form accumulation layer at the
    gate-side ITO interface, with NO drift-diffusion / DEVSIM. n(x, z) =
    n_bg * (1 + delta(V) * exp(-(z_top - z) / lambda_TF)), delta(V) linear in the
    gate bias. Reads only the ITO geometry + n_bg from the Design."""

    def __init__(self, design, *, gate="top_contact", region="ito",
                  screening_nm=1.0, delta_per_volt=0.18, nx=64, nz=41):
        self.design = design
        self.gate = gate
        self.region = region
        self.lambda_m = screening_nm * 1e-9
        self.delta_per_volt = delta_per_volt
        self.nx, self.nz = nx, nz
        zlo, zhi = design.z_intervals()[region]
        self._z_lo, self._z_hi = float(zlo), float(zhi)
        self._P = float(design.unit_cell.period_x_m)
        self._mat = [L.background_material for L in design.stack.layers
                      if L.name == region][0]
        self._n_bg = float(design.materials.get(self._mat).transport.n_bg_m3)

    # ---- CarrierSolver Protocol ----
    def regions(self):
        return [RegionInfo(name=self.region, role="semiconductor", material=self._mat,
                            bbox_m=(0.0, self._P, 0.0, self._P, self._z_lo, self._z_hi),
                            ndim=2)]

    def solve(self, bias) -> CarrierField:
        V = float(bias.voltages.get(self.gate, 0.0))
        delta = self.delta_per_volt * V                       # +V -> accumulation
        x = np.linspace(0.0, self._P, self.nx)
        z = np.linspace(self._z_lo, self._z_hi, self.nz)       # "y" = through-stack
        Z = z[None, :]                                         # broadcast over x
        prof = 1.0 + delta * np.exp(-(self._z_hi - Z) / self.lambda_m)   # (1, nz)
        n_grid = self._n_bg * np.repeat(prof, self.nx, axis=0)           # (nx, nz)
        pot_grid = np.repeat((V * (z - self._z_lo) /
                               max(self._z_hi - self._z_lo, 1e-30))[None, :],
                              self.nx, axis=0)
        XX, ZZ = np.meshgrid(x, z, indexing="ij")
        nodes = np.column_stack([XX.ravel(), ZZ.ravel()])
        reg = CarrierRegion(
            name=self.region, role="semiconductor", material=self._mat,
            nodes_m=nodes,
            node_fields={ELECTRON_DENSITY: n_grid.ravel(), POTENTIAL: pot_grid.ravel()},
            grid_axes_m={"x": x, "y": z},
            grid_fields={ELECTRON_DENSITY: n_grid, POTENTIAL: pot_grid})
        return CarrierField(
            bias_label=bias.label, voltages=dict(bias.voltages), ndim=2,
            temperature_K=300.0, regions={self.region: reg},
            n_bg_by_region={self.region: self._n_bg},
            unit_cell_m=(self._P, self._P))


def _self_check(design):
    """Fast, FEM-free proof of the seam: the analytic n(x,z) flows to the right
    Drude eps -- the accumulation layer pushes the ITO through ENZ (Re(eps)->0)."""
    solver = AnalyticAccumulationSolver(design)
    assert isinstance(solver, CarrierSolver), "must satisfy the CarrierSolver Protocol"
    ito = design.materials.get("ITO")
    lam = 1300e-9
    print("[t] BYO CarrierSolver -- analytic accumulation -> Drude eps @1300nm")
    for V in (0.0, +2.0):
        cf = solver.solve(BiasPoint({"top_contact": V}, "V{}".format(V)))
        n = cf.regions["ito"].grid_fields[ELECTRON_DENSITY]
        n_top = n[:, -1].mean()                       # gate-side interface
        eps_top = complex(ito.eps(lam, n_m3=n_top))
        print("[t]   V={:+.1f}V  n_top/n_bg={:.3f}  Re(eps_ITO,top)={:+.3f}  Im={:+.3f}".format(
            V, n_top / solver._n_bg, eps_top.real, eps_top.imag), flush=True)
    # at +2V the accumulation should drive Re(eps) lower (toward/through ENZ)
    cf0 = solver.solve(BiasPoint({}, "0"))
    cf2 = solver.solve(BiasPoint({"top_contact": 2.0}, "2"))
    re0 = complex(ito.eps(lam, n_m3=cf0.regions["ito"].grid_fields[ELECTRON_DENSITY][:, -1].mean())).real
    re2 = complex(ito.eps(lam, n_m3=cf2.regions["ito"].grid_fields[ELECTRON_DENSITY][:, -1].mean())).real
    assert re2 < re0, "accumulation must lower Re(eps) toward ENZ"
    print("[t] *** BYO CarrierSolver seam OK (no DEVSIM): accumulation lowers Re(eps) "
          "{:+.3f} -> {:+.3f} ***".format(re0, re2))


def main(argv=None):
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fem", action="store_true",
                    help="run the FULL pipeline (carriers->bridge->FEM) at 1 wavelength")
    args = p.parse_args(argv)
    design = build_park_design()
    _self_check(design)
    if args.fem:
        from dynameta.pipeline import run_pipeline
        sweep = Sweep(bias_points=[BiasPoint({"top_contact": +2.0}, "byo+2V")],
                       wavelengths_nm=[1300.0])
        # the WHOLE of Stage 1 is replaced by the analytic solver:
        rows = run_pipeline(design, sweep,
                             carrier_solver=AnalyticAccumulationSolver(design))
        for r in rows:
            print("[t] FEM  {} lam={:.0f}nm  R={:.4f}".format(
                r.bias_label, r.lambda_nm, r.result.R), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
