"""dynameta.io -- result serialization (HDF5 / Zarr) and the on-disk store backend."""
from dynameta.io.store import available_formats, load_arrays, save_arrays

__all__ = ["save_arrays", "load_arrays", "available_formats"]
