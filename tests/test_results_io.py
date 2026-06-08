"""Fast unit tests for the results/io layer: the HDF5+Zarr store, the SweepResults container, the
persistent optical-solver cache, and the matplotlib viz helpers (all without a real solver)."""
import os
from types import SimpleNamespace

import numpy as np
import pytest

from dynameta.core.interfaces import OpticalResult
from dynameta.io.store import available_formats, load_arrays, save_arrays
from dynameta.pipeline import SweepRow
from dynameta.results import SweepResults

_FORMATS = available_formats()
_EXT = {"hdf5": ".h5", "zarr": ".zarr"}


def _rows():
    rows = []
    for bi, bl in enumerate(["off", "on"]):
        for w in (1400.0, 1500.0, 1600.0):
            R = 0.20 + 0.10 * bi + 1e-4 * (w - 1400.0)
            rows.append(SweepRow(bl, w, OpticalResult(r=complex(R ** 0.5, 0.1), R=R, phase_deg=10.0 * bi,
                                                      solve_time_s=1.0, t=complex((1 - R) ** 0.5, 0.0),
                                                      T=1 - R, A=0.0, R_flux=R, T_flux=1 - R)))
    return rows


@pytest.mark.parametrize("fmt", _FORMATS)
def test_store_roundtrip(fmt, tmp_path):
    p = str(tmp_path / ("s" + _EXT[fmt]))
    arrays = {"R": np.random.rand(2, 4), "wl": np.linspace(1.4, 1.7, 4)}
    attrs = {"bias": ["off", "on"], "n": 2, "x": 3.14}
    save_arrays(p, arrays, attrs, fmt=fmt)
    a, m = load_arrays(p, fmt=fmt)
    assert np.allclose(a["R"], arrays["R"]) and np.allclose(a["wl"], arrays["wl"])
    assert m["bias"] == ["off", "on"] and m["n"] == 2 and abs(m["x"] - 3.14) < 1e-9


def test_sweepresults_pivot_and_contrast():
    sr = SweepResults.from_rows(_rows())
    assert sr.n_bias == 2 and sr.n_wl == 3
    assert sr.R.shape == (2, 3)
    assert abs(float(sr.spectrum("on", "R")[1]) - 0.31) < 1e-9        # on @ 1500nm = 0.20 + 0.10 + 0.01
    assert abs(sr.max_contrast("R") - 0.10) < 1e-9                    # the bias offset


@pytest.mark.parametrize("fmt", _FORMATS)
def test_sweepresults_save_load(fmt, tmp_path):
    sr = SweepResults.from_rows(_rows())
    p = str(tmp_path / ("sweep" + _EXT[fmt]))
    sr.save(p, fmt=fmt)
    sr2 = SweepResults.load(p, fmt=fmt)
    assert sr2.bias_labels == ["off", "on"]
    assert np.allclose(sr.R, sr2.R) and np.allclose(sr.T, sr2.T)
    assert np.allclose(sr.r, sr2.r) and np.allclose(sr.t, sr2.t)      # complex preserved


def _design():
    from dynameta.geometry import Design, Layer, Stack, UnitCell
    from dynameta.materials import ConstantOptical, Material, MaterialRegistry
    reg = MaterialRegistry()
    for nm, e in [("air", 1.0), ("m", 4.0)]:
        reg.add(Material(nm, ConstantOptical(complex(e))))
    stack = Stack(layers=[Layer("s", 100e-9, "m", inclusions=[])],
                  superstrate_material="air", substrate_material="air")
    return Design(name="c", unit_cell=UnitCell.square(220e-9), stack=stack, electrodes=[], materials=reg)


def test_cache_hit_miss_and_persist(tmp_path):
    from dynameta.cache import OpticalSolverCache
    calls = {"n": 0}

    def inner(design, geo, eps, lam, ns, nb):
        calls["n"] += 1
        R = abs(complex(eps["s"].scalar)) * 1e-3 + lam * 1e5
        return OpticalResult(r=complex(R, 0), R=R, phase_deg=0.0, solve_time_s=0.5, T=1 - R, A=0.0)

    d = _design()
    p = str(tmp_path / "cache.zarr")
    eps = {"s": SimpleNamespace(is_uniform=True, scalar=4.0 + 0j)}
    grid = [(b, lam) for b in (4.0, 9.0) for lam in (1.4e-6, 1.5e-6)]

    c1 = OpticalSolverCache(inner, p)
    for b, lam in grid:
        c1(d, None, {"s": SimpleNamespace(is_uniform=True, scalar=complex(b))}, lam, 1.0, 1.0)
    assert calls["n"] == 4 and c1.stats()["misses"] == 4
    # a fresh cache reading the SAME file -> all hits, zero new inner calls
    c2 = OpticalSolverCache(inner, p)
    res = [c2(d, None, {"s": SimpleNamespace(is_uniform=True, scalar=complex(b))}, lam, 1.0, 1.0)
           for b, lam in grid]
    assert calls["n"] == 4 and c2.stats()["hits"] == 4                # nothing recomputed
    assert all(isinstance(r, OpticalResult) for r in res)


def test_viz_saves_png(tmp_path):
    pytest.importorskip("matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    from dynameta import viz
    sr = SweepResults.from_rows(_rows())
    p = str(tmp_path / "summary.png")
    viz.plot_sweep_summary(sr, save=p)
    assert os.path.exists(p) and os.path.getsize(p) > 1000
