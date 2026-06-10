"""A persistent, content-addressed cache for the optical-solver seam: wrap any optical_solver so that a
repeated (design, bias-eps, wavelength, end-media) solve is served from disk instead of recomputed. This
is the fix for the audit's "expensive (bias x wavelength) FEM/FDTD sweeps re-run from scratch each design
iteration" -- add a wavelength, tweak a bias, restart a notebook, and the unchanged points are free.

The cache key is a content hash of the SOLVE INPUTS -- the design geometry (period + per-layer
name/thickness/material/#inclusions + end materials), the bias-modulated eps_by_region values (uniform
scalar or the gridded array bytes), the wavelength, and n_super/n_sub -- so it is correct across processes
and machines (no reliance on object identity). The store is one HDF5/Zarr file (dynameta.io.store), each
entry a packed 12-float OpticalResult vector. Use OpticalSolverCache(inner, path) as a drop-in
optical_solver in run_pipeline; call .flush() (or rely on autosave) to persist, .stats() for hit/miss.
"""
from __future__ import annotations

import hashlib
import os
import struct
from typing import Optional

import numpy as np

from dynameta.core.interfaces import OpticalResult
from dynameta.io.store import load_arrays, save_arrays

# packed-vector layout (NaN = the field was None): the OpticalResult scalar fields + split complex r/t
_VEC = ("R", "phase_deg", "solve_time_s", "T", "A", "A_independent", "R_flux", "T_flux",
        "r_re", "r_im", "t_re", "t_im")
_OPT = ("T", "A", "A_independent", "R_flux", "T_flux")     # fields that may be None
# OpticalResult.per_region_absorption (D2) is deliberately NOT cached: a variable-length
# per-design diagnostic dict, not a scalar solver output -- a cache HIT returns it as None.


def _eps_fingerprint(eps_by_region) -> bytes:
    h = hashlib.sha1()
    for name in sorted(eps_by_region or {}):
        ef = eps_by_region[name]
        h.update(name.encode("utf-8"))
        sc = getattr(ef, "scalar", None)
        if getattr(ef, "is_uniform", True) and sc is not None:
            z = complex(sc); h.update(struct.pack("<dd", z.real, z.imag))
        else:
            arr = getattr(ef, "values_zyx", None)
            if arr is None:
                arr = getattr(ef, "values", None)
            h.update(np.ascontiguousarray(np.asarray(arr)).tobytes() if arr is not None else b"?")
    return h.digest()


def _design_fingerprint(design) -> bytes:
    parts = [str(getattr(design, "name", "")), str(design.unit_cell.period_x_m),
             str(design.unit_cell.period_y_m)]
    for L in design.stack.layers:
        parts += [L.name, repr(float(L.thickness_m)), str(getattr(L, "background_material", ""))]
        # per-inclusion SHAPE/material/priority -- hashing only the COUNT let a radius/size change
        # silently serve the wrong cached result when eps_by_region was uniform (audit-v2 follow-up;
        # the patterning lives in the GEOMETRY, not in the eps values, so eps cannot save us).
        for inc in (getattr(L, "inclusions", []) or []):
            parts += [repr(getattr(inc, "shape", "")), str(getattr(inc, "material", "")),
                      str(getattr(inc, "priority", 0))]
    for ft in (getattr(design.stack, "features", []) or []):
        parts += ["feat", repr(getattr(ft, "shape", "")), str(getattr(ft, "material", "")),
                  repr(float(getattr(ft, "z_lo_m", 0.0))), repr(float(getattr(ft, "z_hi_m", 0.0)))]
    parts += [str(design.stack.superstrate_material), str(design.stack.substrate_material)]
    # the optical INCIDENCE spec changes R/T but NOT the eps grid -- it MUST be in the key, else an
    # angle/polarization/side sweep silently serves the cached result for a different angle (audit HIGH).
    # Hash the WHOLE spec reprs (deterministic dataclass reprs): any incidence knob or FEM mesh-sizing
    # knob (maxh/buffers/PML/order) changes the solve for the same eps. (mesh_2d is carrier-side only:
    # it cannot change the optical answer GIVEN eps_by_region, which is already a key input.)
    parts += ["opt", repr(getattr(design, "optical", None)),
              "mesh3", repr(getattr(design, "mesh_3d", None))]
    return hashlib.sha1("|".join(parts).encode("utf-8")).digest()


def _key(design, eps_by_region, lambda_m, n_super, n_sub, tag) -> str:
    h = hashlib.sha1()
    h.update(_design_fingerprint(design)); h.update(_eps_fingerprint(eps_by_region))
    h.update(struct.pack("<d", float(lambda_m)))
    for n in (n_super, n_sub):
        z = complex(n); h.update(struct.pack("<dd", z.real, z.imag))
    h.update(str(tag).encode("utf-8"))
    return "k" + h.hexdigest()                              # valid HDF5/Zarr dataset name


class OpticalSolverCache:
    """A drop-in optical_solver that memoizes `inner` to disk (HDF5/Zarr). `tag` namespaces a cache (e.g.
    the solver/resolution config) so different solver settings do not collide. autosave=True persists on
    every miss (cheap, crash-safe); set False and call flush() once for a faster big sweep."""

    def __init__(self, inner_solver, path: str, *, tag: str = "", fmt: str = "auto",
                 autosave: bool = True, verbose: bool = False):
        self.inner = inner_solver
        self.path = path
        self.tag = str(tag)
        self.fmt = fmt
        self.autosave = bool(autosave)
        self.verbose = bool(verbose)
        self.hits = 0
        self.misses = 0
        self._mem = {}
        if os.path.exists(path):
            try:
                arrays, _ = load_arrays(path, fmt=fmt)
                self._mem = {k: np.asarray(v, dtype=float) for k, v in arrays.items()}
            except Exception:                               # pragma: no cover - a corrupt/foreign file
                self._mem = {}

    def __call__(self, design, geometry, eps_by_region, lambda_m, n_super, n_sub) -> OpticalResult:
        key = _key(design, eps_by_region, lambda_m, n_super, n_sub, self.tag)
        if key in self._mem:
            self.hits += 1
            return self._unpack(self._mem[key])
        self.misses += 1
        res = self.inner(design, geometry, eps_by_region, lambda_m, n_super, n_sub)
        self._mem[key] = self._pack(res)
        if self.autosave:
            self.flush()
        return res

    def flush(self) -> str:
        """Write the in-memory cache to disk (HDF5/Zarr)."""
        return save_arrays(self.path, self._mem, {"schema": 2, "tag": self.tag, "layout": list(_VEC)},
                           fmt=self.fmt)                     # schema 2: stronger design fingerprint

    def stats(self) -> dict:
        tot = self.hits + self.misses
        return {"hits": self.hits, "misses": self.misses, "entries": len(self._mem),
                "hit_rate": (self.hits / tot) if tot else 0.0}

    @staticmethod
    def _pack(res: OpticalResult) -> np.ndarray:
        r = complex(res.r) if res.r is not None else complex("nan")
        t = complex(res.t) if res.t is not None else complex("nan")
        def f(x):
            return float(x) if x is not None else float("nan")
        return np.array([f(res.R), f(res.phase_deg), f(res.solve_time_s), f(res.T), f(res.A),
                         f(res.A_independent), f(res.R_flux), f(res.T_flux),
                         r.real, r.imag, t.real, t.imag], dtype=float)

    @staticmethod
    def _unpack(v: np.ndarray) -> OpticalResult:
        d = {k: v[i] for i, k in enumerate(_VEC)}
        opt = (lambda x: None if np.isnan(x) else float(x))
        t = None if np.isnan(d["t_re"]) else complex(d["t_re"], d["t_im"])
        r = None if np.isnan(d["r_re"]) else complex(d["r_re"], d["r_im"])   # mirror t (None round-trips, not nan+nanj)
        # round-trip the ORIGINAL solve time (the field documents the SOLVE cost, not retrieval cost;
        # hardcoding 0.0 here silently discarded it -- audit cache-1, flagged two rounds running).
        st = d["solve_time_s"]
        return OpticalResult(r=r, R=float(d["R"]), phase_deg=float(d["phase_deg"]),
                             solve_time_s=(0.0 if np.isnan(st) else float(st)), t=t,
                             T=opt(d["T"]), A=opt(d["A"]),
                             A_independent=opt(d["A_independent"]), R_flux=opt(d["R_flux"]),
                             T_flux=opt(d["T_flux"]))


def cached_optical_solver(inner_solver, path: str, **kw) -> OpticalSolverCache:
    """Convenience builder: wrap `inner_solver` in a persistent OpticalSolverCache at `path`."""
    return OpticalSolverCache(inner_solver, path, **kw)
