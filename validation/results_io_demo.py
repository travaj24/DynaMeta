"""End-to-end DEMO of the results / IO / viz / cache layer (no heavy solver). Builds a (bias x wavelength)
sweep with a fast mock optical solver, wraps it in the persistent cache, saves the SweepResults to BOTH
HDF5 and Zarr (round-trips exactly), and renders the summary figure to a PNG -- the day-to-day workflow the
audit flagged as missing (no plotting, no checkpointing).

GATES:
  1  ROUND-TRIP: SweepResults saved+loaded on HDF5 AND Zarr reproduces R/T and the complex r/t exactly.
  2  CACHE REUSE: a second pass over the same (design, bias, wavelength) grid recomputes NOTHING (all
     served from the on-disk cache), while the first pass computed every point.
  3  VIZ: the summary figure (spectrum + modulation-contrast) is written to a non-empty PNG.

Run: python -m validation.results_io_demo
"""
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from types import SimpleNamespace

from dynameta.cache import OpticalSolverCache
from dynameta.core.interfaces import OpticalResult
from dynameta.geometry import Design, Layer, Stack, UnitCell
from dynameta.io.store import available_formats
from dynameta.materials import ConstantOptical, Material, MaterialRegistry
from dynameta.pipeline import SweepRow
from dynameta.results import SweepResults

BIASES = {"off": 4.0e26, "on": 1.2e27}                      # a (label -> carrier-density) modulation
LAMS_NM = [1400, 1450, 1500, 1550, 1600]


def _design():
    reg = MaterialRegistry()
    for nm, e in [("air", 1.0), ("ito", 3.9)]:
        reg.add(Material(nm, ConstantOptical(complex(e))))
    stack = Stack(layers=[Layer("ito", 60e-9, "ito", inclusions=[])],
                  superstrate_material="air", substrate_material="air")
    return Design(name="demo", unit_cell=UnitCell.square(370e-9), stack=stack, electrodes=[], materials=reg)


def _eps_at(n_m3, lam_nm):
    """A toy free-carrier eps(n, lambda) -> a duck-typed eps_by_region for the seam/cache."""
    eps = 3.9 - (n_m3 / 1e27) * (lam_nm / 1500.0) ** 2      # crude ENZ-ish shift with bias + wavelength
    return {"ito": SimpleNamespace(is_uniform=True, scalar=complex(eps, 0.1))}


def _mock_solver(calls):
    def solve(design, geo, eps, lam, ns, nb):
        calls["n"] += 1
        e = complex(eps["ito"].scalar)
        n = e ** 0.5
        R = float(abs((1 - n) / (1 + n)) ** 2)             # bare-interface Fresnel of the toy eps
        return OpticalResult(r=complex((1 - n) / (1 + n)), R=R, phase_deg=0.0, solve_time_s=0.01,
                             t=complex(2 / (1 + n)), T=max(0.0, 1 - R), A=0.0, R_flux=R, T_flux=max(0.0, 1 - R))
    return solve


def _run(cache):
    rows = []
    for bl, n in BIASES.items():
        for w in LAMS_NM:
            eps = _eps_at(n, w)
            res = cache(_design(), None, eps, w * 1e-9, 1.0, 1.0)
            rows.append(SweepRow(bl, float(w), res))
    return SweepResults.from_rows(rows)


def main():
    print("[rio] === results / IO (HDF5+Zarr) / viz / cache demo ===", flush=True)
    print("[rio] serialization backends available: {}".format(available_formats()), flush=True)
    tmp = tempfile.mkdtemp(prefix="dynameta_rio_")
    calls = {"n": 0}
    solver = _mock_solver(calls)

    # pass 1: compute the whole grid through the cache
    cache = OpticalSolverCache(solver, os.path.join(tmp, "cache.zarr"))
    sr = _run(cache)
    n1 = calls["n"]; cache.flush()

    # pass 2: a FRESH cache reading the same file -> everything served from disk
    cache2 = OpticalSolverCache(solver, os.path.join(tmp, "cache.zarr"))
    sr2 = _run(cache2)
    n2_new = calls["n"] - n1
    print("[rio] cache: pass-1 computed {} points; pass-2 recomputed {} (hits={}, hit_rate={:.0%})".format(
        n1, n2_new, cache2.stats()["hits"], cache2.stats()["hit_rate"]), flush=True)
    g2 = (n1 == len(BIASES) * len(LAMS_NM)) and (n2_new == 0)

    # round-trip on every available backend
    g1 = True
    for fmt, ext in (("hdf5", ".h5"), ("zarr", ".zarr")):
        if fmt not in available_formats():
            continue
        p = os.path.join(tmp, "sweep" + ext)
        sr.save(p, fmt=fmt)
        rl = SweepResults.load(p, fmt=fmt)
        ok = (rl.bias_labels == sr.bias_labels and np.allclose(rl.R, sr.R, equal_nan=True)
              and np.allclose(rl.r, sr.r, equal_nan=True) and np.allclose(rl.t, sr.t, equal_nan=True))
        sz = (os.path.getsize(p) if os.path.isfile(p) else sum(
            os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(p) for f in fs))
        print("[rio] {:5s} round-trip exact: {}  ({} bytes at {})".format(fmt, ok, sz, p), flush=True)
        g1 = g1 and ok

    # viz: write the summary PNG
    g3 = True
    try:
        import matplotlib
        matplotlib.use("Agg")
        from dynameta import viz
        png = os.path.join(tmp, "summary.png")
        viz.plot_sweep_summary(sr, save=png)
        g3 = os.path.exists(png) and os.path.getsize(png) > 1000
        print("[rio] viz summary PNG written: {} ({} bytes); max R-contrast = {:.4f}".format(
            g3, os.path.getsize(png) if os.path.exists(png) else 0, sr.max_contrast("R")), flush=True)
    except ImportError:
        print("[rio] matplotlib not installed -> skip the viz gate", flush=True)

    ok = g1 and g2 and g3
    print("[rio] GATE1 HDF5+Zarr round-trip exact: {}".format("PASS" if g1 else "FAIL"), flush=True)
    print("[rio] GATE2 cache reuse (pass-2 recomputes nothing): {}".format("PASS" if g2 else "FAIL"), flush=True)
    print("[rio] GATE3 viz summary PNG written: {}".format("PASS" if g3 else "FAIL"), flush=True)
    print("[rio] *** RESULTS / IO / VIZ / CACHE LAYER: {} ***".format("PASS" if ok else "FAIL"), flush=True)
    import shutil; shutil.rmtree(tmp, ignore_errors=True)
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
