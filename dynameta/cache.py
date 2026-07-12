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
# Store-format schema. BUMP whenever the KEY derivation changes so an on-disk cache written by an
# older (buggy) keying is DISCARDED on load rather than serving collided/mis-keyed entries.
#   2 -> 3: uniform-anisotropic `tensor` eps is now hashed into the key (previously all uniform-tensor
#           states collided -- a Pockels/Kerr/magneto-optic bias sweep served the first point's result).
#   3 -> 4 (audit C5-3/C5-6): material eps CONTENT (sampled at the request wavelength) and the inner
#           SOLVER identity are now keyed, and Feature.priority joined the design fingerprint.
#           Previously (a) retuning a material's optical constants under an unchanged registry name
#           served stale results (backends re-derive eps from design.materials at solve time; probe:
#           HIT returned R=0.179 where the truth was 0.059), and (b) two different backends sharing a
#           cache path with the default tag='' served each other's specular-vs-order-summed numbers.
_SCHEMA = 4
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
            continue
        # A UNIFORM ANISOTROPIC EpsField has is_uniform True but scalar None -- its content lives in
        # `tensor` (a (3,3)), NOT in values_zyx. Hashing it is REQUIRED: a PockelsEffect/KerrEffect
        # under a uniform gate field, or MagnetoOpticModel, emits EpsField(tensor=(3,3)) (core/bridge),
        # so without this every uniform-tensor state in a bias sweep collides to the same key and the
        # cache serves the FIRST point's R/T/phase for all later points (audit P1).
        ten = getattr(ef, "tensor", None)
        if ten is not None:
            h.update(b"T")                                   # tag so a (3,3) tensor cannot alias a grid
            h.update(np.ascontiguousarray(np.asarray(ten, dtype=complex)).tobytes())
            continue
        arr = getattr(ef, "values_zyx", None)
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
                  repr(float(getattr(ft, "z_lo_m", 0.0))), repr(float(getattr(ft, "z_hi_m", 0.0))),
                  str(getattr(ft, "priority", 0))]           # priority resolves overlaps (audit C5-3)
    parts += [str(design.stack.superstrate_material), str(design.stack.substrate_material)]
    # the optical INCIDENCE spec changes R/T but NOT the eps grid -- it MUST be in the key, else an
    # angle/polarization/side sweep silently serves the cached result for a different angle (audit HIGH).
    # Hash the WHOLE spec reprs (deterministic dataclass reprs): any incidence knob or FEM mesh-sizing
    # knob (maxh/buffers/PML/order) changes the solve for the same eps. (mesh_2d is carrier-side only:
    # it cannot change the optical answer GIVEN eps_by_region, which is already a key input.)
    parts += ["opt", repr(getattr(design, "optical", None)),
              "mesh3", repr(getattr(design, "mesh_3d", None))]
    return hashlib.sha1("|".join(parts).encode("utf-8")).digest()


def _materials_fingerprint(design, lambda_m) -> bytes:
    """audit C5-3: the non-FEM backends re-derive eps from design.materials at SOLVE time, so
    the key must carry the material CONTENT, not just names -- retuning a material's optical
    constants under an unchanged registry name used to serve stale cached results. Each
    referenced material's eps is sampled AT THE REQUEST WAVELENGTH (exactly what the backend
    will read; lambda is already a key input). Models whose eps needs runtime state (e.g.
    DrudeOptical without n_m3 -- the carrier value arrives via eps_by_region, which is
    already keyed) fall back to their repr: a deterministic dataclass repr is content-bearing,
    and a non-deterministic repr only causes safe re-solves, never staleness."""
    names = {str(design.stack.superstrate_material), str(design.stack.substrate_material)}
    for L in design.stack.layers:
        names.add(str(getattr(L, "background_material", "")))
        for inc in (getattr(L, "inclusions", []) or []):
            names.add(str(getattr(inc, "material", "")))
    for ft in (getattr(design.stack, "features", []) or []):
        names.add(str(getattr(ft, "material", "")))
    h = hashlib.sha1()
    for name in sorted(n for n in names if n):
        h.update(name.encode("utf-8"))
        try:
            mat = design.materials.get(name)
        except Exception:
            h.update(b"!missing")
            continue
        try:
            z = complex(mat.eps(float(lambda_m)))
            h.update(struct.pack("<dd", z.real, z.imag))
        except Exception:
            h.update(repr(mat).encode("utf-8"))
    return h.digest()


def _key(design, eps_by_region, lambda_m, n_super, n_sub, tag, solver_id="") -> str:
    h = hashlib.sha1()
    h.update(_design_fingerprint(design)); h.update(_eps_fingerprint(eps_by_region))
    h.update(_materials_fingerprint(design, lambda_m))       # audit C5-3
    h.update(struct.pack("<d", float(lambda_m)))
    for n in (n_super, n_sub):
        z = complex(n); h.update(struct.pack("<dd", z.real, z.imag))
    h.update(str(tag).encode("utf-8"))
    h.update(str(solver_id).encode("utf-8"))                 # audit C5-6: backend identity
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
        # audit C5-6: key the inner solver's IDENTITY so two different backends sharing a
        # cache path with the default tag='' cannot serve each other's results (FEM specular
        # vs bridge order-summed R/T; probe: swapped backend served R=0.107 where 0.177 is
        # correct). Function qualnames carry the factory ('make_layered_tmm_solver.<locals>.
        # _solve'); class instances use the class qualname. An explicit `tag` still namespaces
        # solver SETTINGS (resolution, orders) within one backend.
        self._solver_id = "{}:{}".format(
            getattr(inner_solver, "__module__", type(inner_solver).__module__),
            getattr(inner_solver, "__qualname__", type(inner_solver).__qualname__))
        self.fmt = fmt
        self.autosave = bool(autosave)
        self.verbose = bool(verbose)
        self.hits = 0
        self.misses = 0
        self._mem = {}
        if os.path.exists(path):
            try:
                arrays, meta = load_arrays(path, fmt=fmt)
                # DISCARD a cache written under an older key schema -- its keys were derived
                # differently (e.g. the schema-2 uniform-tensor collision), so its entries cannot be
                # trusted against the current _key(). Re-solving is correct; serving stale is not.
                if int((meta or {}).get("schema", -1)) != _SCHEMA:
                    self._mem = {}
                else:
                    self._mem = {k: np.asarray(v, dtype=float) for k, v in arrays.items()}
            except Exception:                               # pragma: no cover - a corrupt/foreign file
                self._mem = {}

    def __call__(self, design, geometry, eps_by_region, lambda_m, n_super, n_sub) -> OpticalResult:
        key = _key(design, eps_by_region, lambda_m, n_super, n_sub, self.tag, self._solver_id)
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
        return save_arrays(self.path, self._mem,
                           {"schema": _SCHEMA, "tag": self.tag, "layout": list(_VEC)},
                           fmt=self.fmt)                     # _SCHEMA (single source): the load-side
                                                            # discard check uses the SAME constant, so
                                                            # a future bump cannot make the cache
                                                            # write-only (flush stamping a stale int)

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
