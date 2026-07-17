"""A tiny backend-agnostic array store: write a flat dict of NumPy arrays plus a JSON-able metadata dict
to disk, and read it back, on EITHER HDF5 (h5py) or Zarr (zarr) -- chosen by file extension (.h5/.hdf5 ->
HDF5, .zarr -> Zarr) or an explicit `fmt`. This is the on-disk substrate the result container
(dynameta.results.SweepResults) and the persistent solver cache (dynameta.cache) share, so a saved sweep
or a cache file is portable between the two storage engines and readable by any HDF5/Zarr tool
(xarray, h5py, zarr, MATLAB, ...).

Design notes:
  * Only REAL numeric arrays are stored as datasets; complex data is split into <name>_real/<name>_imag by
    the caller (keeps the backend layer free of complex-dtype portability quirks). Strings / scalars / lists
    go into the metadata dict, which is serialized as ONE JSON attribute ("_meta"), so attribute typing is
    identical across HDF5 and Zarr.
  * h5py and zarr are OPTIONAL dependencies; an unavailable backend raises a clear ImportError naming the
    extra. available_formats() reports which are installed here.
Convention: paths are plain filesystem paths; the Zarr backend uses a directory store (.zarr).
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, Tuple

import numpy as np

_HDF5_EXT = (".h5", ".hdf5", ".he5")
_ZARR_EXT = (".zarr",)
_META_KEY = "_meta"                                         # the single JSON attribute holding the metadata


def _have(mod: str) -> bool:
    try:
        __import__(mod)
        return True
    except Exception:                                       # pragma: no cover - environment-dependent
        return False


def available_formats():
    """The serialization backends runnable here: 'hdf5' if h5py is importable, 'zarr' if zarr is."""
    fmts = []
    if _have("h5py"):
        fmts.append("hdf5")
    if _have("zarr"):
        fmts.append("zarr")
    return fmts


def _detect_fmt(path: str, fmt: str) -> str:
    if fmt != "auto":
        if fmt not in ("hdf5", "zarr"):
            raise ValueError("fmt must be 'hdf5', 'zarr', or 'auto'; got {!r}".format(fmt))
        return fmt
    ext = os.path.splitext(str(path))[1].lower()
    if ext in _HDF5_EXT:
        return "hdf5"
    if ext in _ZARR_EXT:
        return "zarr"
    raise ValueError("cannot infer format from extension {!r}; use a {}/{} suffix or pass fmt=".format(
        ext, "/".join(_HDF5_EXT), "/".join(_ZARR_EXT)))


def _require(mod: str, fmt: str):
    try:
        return __import__(mod)
    except ImportError as e:                                # pragma: no cover - optional dep
        raise ImportError("the '{}' format needs the optional '{}' package (pip install dynameta[io] or "
                          "pip install {}).".format(fmt, mod, mod)) from e


def save_arrays(path: str, arrays: Dict[str, np.ndarray], attrs: Dict[str, Any] = None, *,
                fmt: str = "auto") -> str:
    """Write `arrays` (name -> real NumPy array) and the JSON-able `attrs` dict to `path` on the backend
    chosen by extension (or `fmt`). Returns the path. Parent directories are created."""
    attrs = dict(attrs or {})
    backend = _detect_fmt(path, fmt)
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)

    def _jsonable(o):
        # audit S4-28: numpy scalars crashed json.dumps; convert transparently. (JSON has no
        # tuple type, so tuples round-trip as lists -- documented contract.)
        if isinstance(o, np.generic):
            return o.item()
        raise TypeError("save_arrays attrs value of type {} is not JSON-serializable; use "
                        "plain python / numpy scalar types (tuples become lists)".format(type(o)))

    meta = json.dumps(attrs, default=_jsonable)
    if backend == "hdf5":
        h5py = _require("h5py", "hdf5")
        with h5py.File(path, "w") as f:
            for k, v in arrays.items():
                f.create_dataset(k, data=np.asarray(v))
            f.attrs[_META_KEY] = meta
    else:
        zarr = _require("zarr", "zarr")
        import shutil
        if os.path.isdir(path):                             # a fresh write replaces a prior directory store
            shutil.rmtree(path, ignore_errors=True)
        g = zarr.open_group(path, mode="w")
        for k, v in arrays.items():
            a = np.asarray(v)
            if a.ndim == 0:
                # audit S4-7: `g[k] = a` and `g[k][:]` both raise IndexError for 0-d arrays on
                # zarr>=3, breaking HDF5<->zarr parity; create the 0-d array explicitly
                arr = g.create_array(k, shape=(), dtype=a.dtype)
                arr[...] = a
            else:
                g[k] = a
        g.attrs[_META_KEY] = meta
    return path


def load_arrays(path: str, *, fmt: str = "auto") -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """Read back (arrays, attrs) written by save_arrays. arrays is a name -> NumPy array dict; attrs is the
    metadata dict."""
    backend = _detect_fmt(path, fmt)
    if backend == "hdf5":
        h5py = _require("h5py", "hdf5")
        with h5py.File(path, "r") as f:
            arrays = {k: np.asarray(f[k][()]) for k in f.keys()}
            meta = f.attrs.get(_META_KEY, "{}")
    else:
        zarr = _require("zarr", "zarr")
        g = zarr.open_group(path, mode="r")
        arrays = {k: np.asarray(g[k][...]) for k in g.array_keys()}  # [...] handles 0-d (S4-7)
        meta = g.attrs.get(_META_KEY, "{}")
    return arrays, json.loads(meta)
