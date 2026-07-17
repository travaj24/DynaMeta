"""A persistent, content-addressed cache for the optical-solver seam: wrap any optical_solver so that a
repeated (design, bias-eps, wavelength, end-media) solve is served from disk instead of recomputed. This
is the fix for the audit's "expensive (bias x wavelength) FEM/FDTD sweeps re-run from scratch each design
iteration" -- add a wavelength, tweak a bias, restart a notebook, and the unchanged points are free.

The cache key is a content hash of the SOLVE INPUTS -- the design geometry (period + per-layer
name/thickness/material/#inclusions + end materials), the bias-modulated eps_by_region values (uniform
scalar or the gridded array bytes), the wavelength, and n_super/n_sub -- so it is correct across processes
and machines (no reliance on object identity). The store is one HDF5/Zarr file (dynameta.io.store)
holding ALL entries as two datasets: an (N,12) float64 value matrix (one packed OpticalResult vector
per row) and an (N,41) uint8 ASCII key matrix. Use OpticalSolverCache(inner, path) as a drop-in
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
# Store-format schema. BUMP whenever the KEY derivation OR the on-disk layout changes so an on-disk
# cache written by an older format is DISCARDED on load rather than serving collided/mis-keyed entries.
#   2 -> 3: uniform-anisotropic `tensor` eps is now hashed into the key (previously all uniform-tensor
#           states collided -- a Pockels/Kerr/magneto-optic bias sweep served the first point's result).
#   3 -> 4 (audit C5-3/C5-6): material eps CONTENT (sampled at the request wavelength) and the inner
#           SOLVER identity are now keyed, and Feature.priority joined the design fingerprint.
#           Previously (a) retuning a material's optical constants under an unchanged registry name
#           served stale results (backends re-derive eps from design.materials at solve time; probe:
#           HIT returned R=0.179 where the truth was 0.059), and (b) two different backends sharing a
#           cache path with the default tag='' served each other's specular-vs-order-summed numbers.
#   4 -> 5 (audit 6.2): LAYOUT change, keys unchanged -- one small dataset per entry became TWO packed
#           datasets (_PK_VALS (N,12) float64 rows + _PK_KEYS (N,41) uint8 ASCII keys), bit-identical
#           per entry but far faster to flush and reopen (per-dataset metadata churn dominated;
#           measured 25-90x HDF5 / ~130x Zarr at N=400-2000, growing with N). A schema-4 per-key
#           store is discarded on load like any stale schema.
_SCHEMA = 5
_PK_VALS = "packed_vals"                                    # (N, len(_VEC)) float64, one entry per row
_PK_KEYS = "packed_keys"                                    # (N, 41) uint8: "k"+sha1-hex ASCII key rows
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
    the solver/resolution config) so different solver settings do not collide.

    PERSISTENCE COST MODEL (audit 6.2 -- the old '(cheap, crash-safe)' claim was inverted at
    scale): the store has no append path, so EVERY flush rewrites the WHOLE store (HDF5
    mode-'w' truncate; Zarr rmtree-first), and every rewrite is a window where a crash loses
    the accumulated store. Two independent mitigations, still whole-store-rewrite semantics:
      * schema-5 PACKED layout: all entries go into TWO datasets ((N,12) values + (N,41)
        keys) instead of one tiny dataset per entry, whose per-dataset metadata churn
        dominated (the old layout measured 9.68 s of autosave vs 0.04 s for one final
        flush over a 400-miss HDF5 sweep, 240x, and 70x at just 120 Zarr entries); the
        packed rewrite measures 25-90x (HDF5) / ~130x (Zarr) faster per flush AND per
        reopen at N=400-2000, bit-identical entries.
      * autosave_every=K (default 64; audit S6-5 measured the old per-miss default at
        623-1372x persistence overhead on cheap backends, and K=64 recovers nearly all)
        batches the flushes, turning O(N^2) rewrite bytes over a sweep into O(N^2/K)
        while bounding crash loss to K-1 misses; a dirty cache is also flushed at
        interpreter exit (atexit) so a batched tail is never silently dropped.
        autosave=False + one explicit flush() remains the fastest path.
      * flush() MERGES same-schema entries already on disk before an ATOMIC rewrite
        (audit S5-4): concurrent writers with disjoint keys union instead of clobbering,
        and the HDF5 write goes to a temp file + os.replace so a crash mid-write cannot
        leave a half-written store (zarr directory stores replace best-effort). Stale-
        SCHEMA entries are still discarded, never merged (GATE D2 preserved). The race
        window shrinks but is not eliminated: do not share one cache path across
        simultaneous writers producing the SAME keys."""

    def __init__(self, inner_solver, path: str, *, tag: str = "", fmt: str = "auto",
                 autosave: bool = True, autosave_every: int = 64, verbose: bool = False):
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
        # audit S5-2: qualname alone cannot distinguish two solvers from the SAME factory with
        # different answer-changing kwargs (n_slices, resolution, orders...) -- they collided
        # under tag='' and the cache served the wrong R/T. Factories now stamp a
        # `cache_fingerprint` (str or callable) on their closures; fold it into the identity.
        fp = getattr(inner_solver, "cache_fingerprint", None)
        if fp is not None:
            self._solver_id += "|" + str(fp() if callable(fp) else fp)
        elif not self.tag and "<locals>" in self._solver_id:
            import warnings
            warnings.warn(
                "OpticalSolverCache: the inner solver is a factory closure with no "
                "cache_fingerprint and tag='' -- two solvers from the same factory with "
                "different settings would share cache keys. Pass a distinguishing tag= or "
                "stamp solver.cache_fingerprint (audit S5-2).", stacklevel=3)
        self.fmt = fmt
        self.autosave = bool(autosave)
        self.autosave_every = max(1, int(autosave_every))
        self._unsaved = 0
        self.verbose = bool(verbose)
        self.hits = 0
        self.misses = 0
        self._mem = {}
        self._pra = {}                                       # key -> per_region_absorption (S5-3)
        if self.autosave:
            import atexit
            atexit.register(self._flush_if_dirty)            # batched mode: never drop the tail
        if os.path.exists(path):
            try:
                arrays, meta = load_arrays(path, fmt=fmt)
                # DISCARD a cache written under an older key schema or layout -- its keys were derived
                # differently (e.g. the schema-2 uniform-tensor collision), so its entries cannot be
                # trusted against the current _key(). Re-solving is correct; serving stale is not.
                # The discarded entries stay PHYSICALLY on disk until the first flush truncates the
                # file (save_arrays is a whole-store rewrite) -- GATE D2 pins that an append/merge
                # flush cannot resurrect them under the fresh schema stamp.
                if int((meta or {}).get("schema", -1)) != _SCHEMA:
                    self._mem = {}
                else:
                    kmat = np.asarray(arrays[_PK_KEYS], dtype=np.uint8)
                    vmat = np.asarray(arrays[_PK_VALS], dtype=float)
                    # unpack row-wise: each entry gets its OWN (len(_VEC),) float64 copy, bit-identical
                    # to what _pack() produced (ASCII keys and float64 rows round-trip exactly)
                    self._mem = {bytes(kmat[i]).decode("ascii"): vmat[i].copy()
                                 for i in range(kmat.shape[0])}
                    self._pra = {k: dict(v) for k, v in
                                 (meta or {}).get("pra", {}).items() if k in self._mem}
            except Exception:                               # pragma: no cover - a corrupt/foreign file
                self._mem = {}
                self._pra = {}

    def __call__(self, design, geometry, eps_by_region, lambda_m, n_super, n_sub) -> OpticalResult:
        key = _key(design, eps_by_region, lambda_m, n_super, n_sub, self.tag, self._solver_id)
        if key in self._mem:
            self.hits += 1
            return self._unpack(self._mem[key], self._pra.get(key))
        self.misses += 1
        res = self.inner(design, geometry, eps_by_region, lambda_m, n_super, n_sub)
        self._store(key, res)
        return res

    def _store(self, key, res) -> None:
        self._mem[key] = self._pack(res)
        if res.per_region_absorption is not None:            # audit S5-3: a HIT used to drop this
            self._pra[key] = {str(k): float(v) for k, v in res.per_region_absorption.items()}
        self._unsaved += 1
        if self.autosave and self._unsaved >= self.autosave_every:
            self.flush()

    def solve_sweep(self, design, geometry, assemble_at, lambdas, n_super, n_sub):
        """Sweep-aware pass-through (audit S5-12: without this, wrapping a sweep-aware solver
        silently downgraded run_pipeline to per-wavelength solving). Cached wavelengths are
        served from the store; the MISSING subset goes to the inner solve_sweep in one call."""
        if not hasattr(self.inner, "solve_sweep"):
            raise AttributeError("inner solver has no solve_sweep")
        lambdas = list(lambdas)
        results = [None] * len(lambdas)
        keys = []
        miss_lams, miss_idx = [], []
        for i, lm in enumerate(lambdas):
            eps = assemble_at(lm)
            key = _key(design, eps, lm, n_super, n_sub, self.tag, self._solver_id)
            keys.append(key)
            if key in self._mem:
                self.hits += 1
                results[i] = self._unpack(self._mem[key], self._pra.get(key))
            else:
                miss_lams.append(lm)
                miss_idx.append(i)
        if miss_lams:
            self.misses += len(miss_lams)
            solved = list(self.inner.solve_sweep(design, geometry, assemble_at, miss_lams,
                                                 n_super, n_sub))
            if len(solved) != len(miss_lams):
                raise ValueError("inner solve_sweep returned {} results for {} wavelengths".format(
                    len(solved), len(miss_lams)))
            for i, res in zip(miss_idx, solved):
                self._store(keys[i], res)
                results[i] = res
        return results

    def _flush_if_dirty(self) -> None:
        if self._unsaved > 0:
            try:
                self.flush()
            except Exception:                                # atexit must never raise
                pass

    def flush(self) -> str:
        """Write the in-memory cache to disk (HDF5/Zarr; a WHOLE-store rewrite -- see the
        class docstring cost model). The rewrite is deliberate: save_arrays truncates (HDF5
        mode-'w' / Zarr rmtree-first), which is what makes a load-side schema discard
        permanent -- an append/merge flush would resurrect the discarded stale entries
        under the fresh schema stamp (audit 6.2 hazard; GATE D2)."""
        self._unsaved = 0
        # audit S5-4 MERGE step: union same-schema entries already on disk that this process
        # does not hold (disjoint concurrent writers no longer clobber each other). Entries
        # under a DIFFERENT schema are ignored -- never resurrected (GATE D2 semantics).
        if os.path.exists(self.path):
            try:
                arrays, meta = load_arrays(self.path, fmt=self.fmt)
                if int((meta or {}).get("schema", -1)) == _SCHEMA:
                    kmat = np.asarray(arrays[_PK_KEYS], dtype=np.uint8)
                    vmat = np.asarray(arrays[_PK_VALS], dtype=float)
                    disk_pra = (meta or {}).get("pra", {})
                    for i in range(kmat.shape[0]):
                        k = bytes(kmat[i]).decode("ascii")
                        if k not in self._mem:               # memory always wins over disk
                            self._mem[k] = vmat[i].copy()
                            if k in disk_pra:
                                self._pra[k] = dict(disk_pra[k])
            except Exception:                               # unreadable store: overwrite it
                pass
        keys = sorted(self._mem)                            # deterministic on-disk row order
        if keys:
            # keys are uniformly 41 ASCII chars ("k" + sha1 hex) by construction of _key();
            # the reshape fails loudly if that invariant ever breaks (never a silent mis-split)
            kmat = np.frombuffer("".join(keys).encode("ascii"),
                                 dtype=np.uint8).reshape(len(keys), -1)
            vmat = np.stack([self._mem[k] for k in keys])
        else:
            kmat = np.zeros((0, 41), dtype=np.uint8)
            vmat = np.zeros((0, len(_VEC)), dtype=float)
        attrs = {"schema": _SCHEMA, "tag": self.tag, "layout": list(_VEC),
                 "pra": {k: self._pra[k] for k in keys if k in self._pra}}
        backend = self.fmt if self.fmt != "auto" else None
        if (backend or ("zarr" if str(self.path).endswith(".zarr") else "hdf5")) == "hdf5" \
                and not str(self.path).endswith(".zarr"):
            # audit S5-4 ATOMIC step (single-file HDF5): temp write + os.replace
            tmp = str(self.path) + ".tmp-flush"
            save_arrays(tmp, {_PK_KEYS: kmat, _PK_VALS: vmat}, attrs, fmt="hdf5")
            os.replace(tmp, self.path)
            return self.path
        return save_arrays(self.path, {_PK_KEYS: kmat, _PK_VALS: vmat}, attrs,
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
    def _unpack(v: np.ndarray, pra=None) -> OpticalResult:
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
                             T_flux=opt(d["T_flux"]),
                             per_region_absorption=(dict(pra) if pra else None))


def cached_optical_solver(inner_solver, path: str, **kw) -> OpticalSolverCache:
    """Convenience builder: wrap `inner_solver` in a persistent OpticalSolverCache at `path`."""
    return OpticalSolverCache(inner_solver, path, **kw)
