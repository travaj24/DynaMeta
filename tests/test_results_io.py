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
    if not _FORMATS:
        pytest.skip("no io backend (h5py/zarr) installed")
    from dynameta.cache import OpticalSolverCache
    calls = {"n": 0}

    def inner(design, geo, eps, lam, ns, nb):
        calls["n"] += 1
        R = abs(complex(eps["s"].scalar)) * 1e-3 + lam * 1e5
        return OpticalResult(r=complex(R, 0), R=R, phase_deg=0.0, solve_time_s=0.5, T=1 - R, A=0.0)

    d = _design()
    p = str(tmp_path / ("cache" + _EXT[_FORMATS[0]]))         # use an installed backend (skip if none)
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


@pytest.mark.parametrize("fmt", _FORMATS)
def test_sweepresults_nan_field_roundtrip(fmt, tmp_path):
    # a row with MISSING (None) fields -> NaN grid cells; the round-trip must PRESERVE the NaN
    # (equal_nan), not zero or drop it -- the documented missing-field contract. The prior test used a
    # fixture that never produced a NaN, so an impl that zeroed/lost missing cells would have passed.
    rows = [SweepRow("off", 1500.0, OpticalResult(r=1 + 0j, R=0.2, phase_deg=0.0, solve_time_s=1.0,
                                                  t=0.9 + 0j, T=0.8, A=0.0, R_flux=0.2, T_flux=0.8)),
            SweepRow("on", 1500.0, OpticalResult(r=1 + 0j, R=0.3, phase_deg=0.0, solve_time_s=1.0,
                                                 t=None, T=None, A=None))]   # T/A/t missing -> NaN
    sr = SweepResults.from_rows(rows)
    assert np.isnan(sr.T[1, 0]) and not np.isnan(sr.T[0, 0])   # missing cell NaN, present cell not
    p = str(tmp_path / ("nan" + _EXT[fmt]))
    sr.save(p, fmt=fmt)
    sr2 = SweepResults.load(p, fmt=fmt)
    assert np.allclose(sr.T, sr2.T, equal_nan=True)            # NaN preserved through the store
    assert np.isnan(sr2.T[1, 0]) and np.isnan(sr2.t[1, 0])     # explicit: the cell survived as NaN


def test_viz_axes_labels_and_orientation():
    # assertive (not just "PNG > 1000 bytes"): pin axis labels, line counts, x-data, and map shape so
    # a transposed/swapped/wrong-bias plot is caught (pass-2 audit -- viz had no assertive test).
    pytest.importorskip("matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    from dynameta import viz
    sr = SweepResults.from_rows(_rows())                       # 2 biases x 3 wavelengths
    ax1 = viz.plot_spectra(sr, "R")                            # the viz helpers return the Axes
    assert ax1.get_xlabel() == "wavelength (nm)"
    assert len(ax1.lines) == sr.n_bias                         # one spectrum line per bias
    assert np.allclose(ax1.lines[0].get_xdata(), sr.wavelengths_nm)
    ax2 = viz.plot_contrast(sr, "R")
    assert len(ax2.lines) == sr.n_bias - 1                     # contrast drops the reference bias
    im = viz.plot_map(sr.R).images[0]
    assert tuple(im.get_array().shape) == sr.R.shape           # (n_bias, n_wl) orientation preserved


def test_max_contrast_all_nan_comparison_raises():
    # Symmetric to the all-NaN-reference guard: if EVERY non-reference (comparison) bias is all-NaN,
    # np.nanmax silently returns 0.0 (the reference's self-contrast) -- reading 'no modulation' when the
    # ON bias actually had NO data. max_contrast must RAISE. A partially-NaN comparison bias (solve
    # failed at some wavelengths) is legitimate and still reduces; a single bias correctly gives 0.0.
    def _r(R):  # R=None -> a missing (NaN) cell
        return OpticalResult(r=1 + 0j, R=R, phase_deg=0.0, solve_time_s=1.0)
    # (a) 2 biases, ON entirely missing -> comparison row all-NaN -> raise (was a silent 0.0)
    sr = SweepResults.from_rows([SweepRow("off", 1500.0, _r(0.2)), SweepRow("on", 1500.0, _r(None))])
    assert np.isnan(sr.R[1, 0]) and not np.isnan(sr.R[0, 0])
    with pytest.raises(ValueError):
        sr.max_contrast("R")
    # (b) partially-NaN ON bias: present cells still reduce (|0.5-0.2| at 1400; 1500 ON NaN skipped)
    sr2 = SweepResults.from_rows([SweepRow("off", 1400.0, _r(0.2)), SweepRow("off", 1500.0, _r(0.2)),
                                  SweepRow("on", 1400.0, _r(0.5)), SweepRow("on", 1500.0, _r(None))])
    assert abs(sr2.max_contrast("R") - 0.3) < 1e-12
    # (c) single bias (no comparison row) -> 0.0, not an error
    sr3 = SweepResults.from_rows([SweepRow("off", 1500.0, _r(0.2))])
    assert sr3.max_contrast("R") == 0.0


def test_cache_material_retune_and_solver_identity_miss(tmp_path):
    # audit C5-3/C5-6: (a) retuning a material's optical constants under an UNCHANGED
    # registry name must MISS (backends re-derive eps from design.materials at solve time;
    # the name-only key served stale R -- probe: HIT returned R=0.179 vs truth 0.059);
    # (b) a DIFFERENT inner solver over the same path with the default tag='' must MISS
    # (FEM specular vs bridge order-summed numbers must not be served across backends).
    if not _FORMATS:
        pytest.skip("no io backend (h5py/zarr) installed")
    from dynameta.cache import OpticalSolverCache
    from dynameta.materials import ConstantOptical, Material

    def mk_inner(val):
        def inner(design, geo, eps, lam, ns, nb):
            return OpticalResult(r=complex(val, 0), R=val, phase_deg=0.0, solve_time_s=0.1)
        return inner

    def _design_with_m(eps_m):
        from dynameta.geometry import Design, Layer, Stack, UnitCell
        from dynameta.materials import MaterialRegistry
        reg = MaterialRegistry()
        reg.add(Material("air", ConstantOptical(1.0 + 0j)))
        reg.add(Material("m", ConstantOptical(complex(eps_m))))
        stack = Stack(layers=[Layer("s", 100e-9, "m", inclusions=[])],
                      superstrate_material="air", substrate_material="air")
        return Design(name="c", unit_cell=UnitCell.square(220e-9), stack=stack,
                      electrodes=[], materials=reg)

    p = str(tmp_path / ("cache2" + _EXT[_FORMATS[0]]))
    eps = {"s": SimpleNamespace(is_uniform=True, scalar=4.0 + 0j)}
    c1 = OpticalSolverCache(mk_inner(0.1), p)
    c1(_design_with_m(4.0), None, eps, 1.4e-6, 1.0, 1.0)
    assert c1.stats()["misses"] == 1
    # (a) an identical design whose 'm' material is RETUNED under the same name -> MISS
    # (the old name-only fingerprint made this a stale HIT)
    c1(_design_with_m(9.0), None, eps, 1.4e-6, 1.0, 1.0)
    assert c1.stats()["misses"] == 2, "material retune under an unchanged name must MISS"
    # (b) same design + eps + wavelength through a DIFFERENTLY-NAMED solver -> MISS
    def other_backend(design, geo, eps_, lam, ns, nb):
        return OpticalResult(r=complex(0.9, 0), R=0.9, phase_deg=0.0, solve_time_s=0.1)
    c2 = OpticalSolverCache(other_backend, p, autosave=False)
    d2 = _design()
    r_other = c2(d2, None, eps, 1.4e-6, 1.0, 1.0)
    assert c2.stats()["misses"] == 1 and r_other.R == pytest.approx(0.9), \
        "a different backend over the same cache path must not be served another's result"


@pytest.mark.parametrize("fmt", _FORMATS)
def test_cache_packed_layout_stale_truncate_and_bit_identity(fmt, tmp_path):
    # audit 6.2: entries persist as ONE (N,12) value matrix + one (N,41) key matrix (schema 5;
    # ~100-250x faster flush/reopen than a dataset per entry, bit-identical). Fixer-hazard leg:
    # after a load-side schema discard the old entries are still PHYSICALLY in the file -- the
    # first flush must TRUNCATE (whole-store rewrite), else a reopen under the fresh schema
    # stamp resurrects the discarded mis-keyed entries (GATE D2 in validation/optical_cache).
    import dynameta.cache as C
    from dynameta.cache import OpticalSolverCache

    def inner(design, geo, eps, lam, ns, nb):
        R = abs(complex(eps["s"].scalar)) * 1e-3 + lam * 1e5
        return OpticalResult(r=complex(R, -0.2), R=R, phase_deg=7.0, solve_time_s=0.5,
                             t=None, T=None, A=None)          # None fields -> NaN lanes round-trip

    d = _design()
    p = str(tmp_path / ("packed" + _EXT[fmt]))
    stale_key = "k" + "f" * 40
    # a store under the PREVIOUS schema (whose layout was one dataset per entry) with a bogus entry
    save_arrays(p, {stale_key: np.zeros(len(C._VEC))},
                {"schema": C._SCHEMA - 1, "tag": "", "layout": list(C._VEC)}, fmt=fmt)
    c1 = OpticalSolverCache(inner, p, fmt=fmt)
    assert c1._mem == {}                                       # load-side schema discard
    grid = [(4.0, 1.4e-6), (9.0, 1.5e-6)]
    for b, lam in grid:
        c1(d, None, {"s": SimpleNamespace(is_uniform=True, scalar=complex(b))}, lam, 1.0, 1.0)
    arrays, meta = load_arrays(p, fmt=fmt)
    assert set(arrays) == {C._PK_KEYS, C._PK_VALS}             # packed layout; stale dataset GONE
    assert arrays[C._PK_VALS].shape == (2, len(C._VEC)) and arrays[C._PK_KEYS].shape == (2, 41)
    assert int(meta["schema"]) == C._SCHEMA
    c2 = OpticalSolverCache(inner, p, fmt=fmt, autosave=False)
    assert set(c2._mem) == set(c1._mem) and stale_key not in c2._mem   # no resurrection
    assert all(c2._mem[k].tobytes() == c1._mem[k].tobytes() for k in c1._mem)  # BIT-identical
    # a reopened HIT returns exactly the fresh compute (incl. the None fields staying None)
    eps0 = {"s": SimpleNamespace(is_uniform=True, scalar=4.0 + 0j)}
    r_hit = c2(d, None, eps0, 1.4e-6, 1.0, 1.0)
    r_ref = inner(d, None, eps0, 1.4e-6, 1.0, 1.0)
    assert c2.hits == 1 and r_hit.R == r_ref.R and r_hit.r == r_ref.r
    assert r_hit.T is None and r_hit.t is None and r_hit.A is None


def test_cache_autosave_batching(tmp_path):
    # audit 6.2: per-miss autosave rewrites the WHOLE store (O(N^2) over a sweep; measured
    # 240x). autosave_every=K batches flushes (default 1 = old behavior byte-compatible);
    # the batched cache flushes every Kth miss and an explicit flush() drains the tail.
    if not _FORMATS:
        pytest.skip("no io backend (h5py/zarr) installed")
    from dynameta.cache import OpticalSolverCache

    flushes = {"n": 0}

    def inner(design, geo, eps, lam, ns, nb):
        return OpticalResult(r=0.1 + 0j, R=0.1, phase_deg=0.0, solve_time_s=0.1)

    d = _design()
    p = str(tmp_path / ("cache3" + _EXT[_FORMATS[0]]))
    c = OpticalSolverCache(inner, p, autosave_every=4)
    orig_flush = c.flush
    c.flush = lambda: (flushes.__setitem__("n", flushes["n"] + 1), orig_flush())[1]
    for i, lam in enumerate([1.3e-6, 1.35e-6, 1.4e-6, 1.45e-6, 1.5e-6, 1.55e-6]):
        c(d, None, {"s": SimpleNamespace(is_uniform=True, scalar=4.0 + 0j)}, lam, 1.0, 1.0)
    assert c.stats()["misses"] == 6 and flushes["n"] == 1     # one flush at miss 4
    c.flush()                                                 # drain the 2-miss tail
    c2 = OpticalSolverCache(inner, p)
    assert len(c2._mem) == 6                                  # everything persisted
