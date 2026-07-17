"""Discriminating validation for the persistent optical-solver cache (dynameta.cache).

The cache is correctness-critical: a wrong key serves a STALE result for a different input. The
prior coverage (results_io_demo GATE2) only counted recomputes (n2_new==0) -- a cache that ignored
the eps tensor entirely would STILL pass that, which is exactly how the uniform-tensor fingerprint
collision (audit P1) shipped. These gates are DISCRIMINATING: each fails for the wrong model.

GATE A (distinct eps states -> distinct fingerprints): scalar, gridded-scalar, uniform-TENSOR, and
        gridded-tensor EpsFields at two different values each must all fingerprint distinctly. The
        load-bearing discriminator is the uniform-tensor pair: the pre-fix impl hashed b"?" for any
        uniform tensor, so a Pockels/Kerr/magneto-optic bias sweep collided -- a tensor-blind
        fingerprint FAILS this gate.
GATE B (HIT round-trips + tensor sweep stays distinct): wrap a deterministic stub solver in
        OpticalSolverCache; a repeat call HITS and returns a result byte-identical to a fresh solve
        (R/phase/T/A/r/t/R_flux/T_flux), and a 2-point UNIFORM-TENSOR sweep returns the two DISTINCT
        per-bias results (a colliding cache would return the first bias's result for both -- the P1
        regression guard, run through the full __call__ path).
GATE C (every key knob moves the key): changing lambda, n_super, n_sub, the optical incidence spec
        (angle / polarization / side) or mesh_3d each changes _key; an unchanged repeat does not.
GATE D (stale-schema cache is discarded on load): a cache file written under an older key schema is
        DROPPED when reopened (re-solve), not served. Skipped honestly when neither h5py nor zarr is
        installed (the on-disk backends).
GATE D2 (flush into a stale file TRUNCATES it -- audit 6.2 fixer hazard): after a load-side schema
        discard, the discarded entries are still physically on disk until the next flush. A flush
        that APPENDS (or read-merges) instead of truncating would rewrite the file with the CURRENT
        schema stamp while the stale mis-keyed entries survive inside it -- the next reopen would
        resurrect and SERVE them. This leg flushes a fresh entry into a stale-schema file, then
        checks the raw on-disk datasets and a reopen: only the fresh entry may exist.

Run: python -m validation.optical_cache
"""
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import dynameta.cache as C
from dynameta.cache import OpticalSolverCache, _eps_fingerprint, _key
from dynameta.core.eps_field import EpsField
from dynameta.core.interfaces import OpticalResult
from dynameta.geometry import Design, Layer, Stack, UnitCell
from dynameta.geometry.specs import Mesh3DSpec, OpticalSpec
from dynameta.io.store import available_formats, load_arrays, save_arrays
from dynameta.materials import ConstantOptical, Material, MaterialRegistry

PER = 400e-9
LAM = 1.31e-6


def _design(*, pol="y", theta=0.0, side="top", mesh=None):
    reg = MaterialRegistry()
    reg.add(Material("air", ConstantOptical(1.0 + 0j)))
    reg.add(Material("glass", ConstantOptical(complex(1.5 ** 2))))
    reg.add(Material("film", ConstantOptical(2.0 + 0j)))
    opt = OpticalSpec(polarization=pol, incidence_angle_deg=theta, incidence_side=side)
    kw = {"mesh_3d": mesh} if mesh is not None else {}
    return Design(name="cache", unit_cell=UnitCell.square(PER),
                  stack=Stack(layers=[Layer("film", 150e-9, "film")],
                              superstrate_material="air", substrate_material="glass"),
                  electrodes=[], materials=reg, optical=opt, **kw)


def _uniform_tensor(tr):
    """A uniform anisotropic EpsField with trace-encoding diagonal (is_uniform True, scalar None)."""
    return EpsField(tensor=np.diag([2.0, 2.0, tr - 4.0]).astype(complex))


def _grid_scalar(v):
    return EpsField(values_zyx=np.full((4, 1, 1), v, complex), z_axis_u=np.arange(4.0),
                    x_axis_u=np.array([0.0]), y_axis_u=np.array([0.0]))


def _stub_solver(design, geo, ebr, lam, n_sup, n_sub):
    """Deterministic stub: encodes the eps tensor + wavelength into R so distinct inputs give
    distinct results (and a wrong cached value is detectable)."""
    ef = ebr["film"]
    ten = np.asarray(ef.tensor, dtype=complex)
    val = (float(np.real(np.trace(ten))) * 0.01 + 1e6 * float(lam)) % 1.0
    return OpticalResult(r=complex(val, -0.1 * val), R=val, phase_deg=val * 90.0,
                         solve_time_s=0.123, t=complex(1.0 - val, 0.05), T=1.0 - val, A=0.0,
                         R_flux=val, T_flux=1.0 - val)


def _same_result(a, b):
    fields = [("R", a.R, b.R), ("phase_deg", a.phase_deg, b.phase_deg), ("T", a.T, b.T),
              ("A", a.A, b.A), ("R_flux", a.R_flux, b.R_flux), ("T_flux", a.T_flux, b.T_flux)]
    ok = all(abs(complex(x) - complex(y)) < 1e-12 for _n, x, y in fields)
    return ok and abs(a.r - b.r) < 1e-12 and abs(a.t - b.t) < 1e-12


def main():
    print("[cache] === optical-solver cache discriminating gates ===", flush=True)
    ok = True

    # ---- GATE A: distinct eps states -> distinct fingerprints ----
    states = {
        "scalar_lo": {"film": EpsField(scalar=2.1 + 0j)},
        "scalar_hi": {"film": EpsField(scalar=2.5 + 0j)},
        "grid_lo": {"film": _grid_scalar(2.1)},
        "grid_hi": {"film": _grid_scalar(2.5)},
        "tensor_lo": {"film": _uniform_tensor(6.1)},
        "tensor_hi": {"film": _uniform_tensor(6.5)},
        "gtensor": {"film": EpsField(values_zyx=np.broadcast_to(
            np.diag([2.0, 2.0, 2.3]).astype(complex), (3, 1, 1, 3, 3)).copy(),
            z_axis_u=np.arange(3.0), x_axis_u=np.array([0.0]), y_axis_u=np.array([0.0]))},
    }
    fps = {k: _eps_fingerprint(v).hex() for k, v in states.items()}
    all_distinct = len(set(fps.values())) == len(fps)
    tensor_pair_distinct = fps["tensor_lo"] != fps["tensor_hi"]   # the P1 discriminator
    g_a = bool(all_distinct and tensor_pair_distinct)
    ok = ok and g_a
    print("[cache] GATE A: {} eps states all-distinct {}, uniform-tensor pair distinct {} -> {}".format(
        len(fps), all_distinct, tensor_pair_distinct, "PASS" if g_a else "FAIL"), flush=True)

    # ---- GATE B: HIT byte-round-trips + uniform-tensor sweep stays distinct (P1 regression) ----
    d = _design()
    with tempfile.TemporaryDirectory() as td:
        # in-memory only (no flush): the HIT path reads self._mem; autosave off avoids needing a backend
        cache = OpticalSolverCache(_stub_solver, os.path.join(td, "c.h5"), autosave=False)
        e_lo = {"film": _uniform_tensor(6.1)}
        e_hi = {"film": _uniform_tensor(6.5)}
        r_lo_miss = cache(d, None, e_lo, LAM, 1.0 + 0j, 1.5 + 0j)        # miss
        r_lo_hit = cache(d, None, e_lo, LAM, 1.0 + 0j, 1.5 + 0j)         # hit
        r_hi = cache(d, None, e_hi, LAM, 1.0 + 0j, 1.5 + 0j)            # miss (distinct tensor)
        fresh_lo = _stub_solver(d, None, e_lo, LAM, 1.0 + 0j, 1.5 + 0j)
        hit_roundtrips = _same_result(r_lo_hit, fresh_lo) and cache.hits == 1
        sweep_distinct = abs(r_lo_hit.R - r_hi.R) > 1e-9                # colliding cache would tie these
        g_b = bool(hit_roundtrips and sweep_distinct)
    ok = ok and g_b
    print("[cache] GATE B: HIT==fresh {} (hits {}), tensor-sweep R distinct {:.3f} vs {:.3f} -> {}".format(
        hit_roundtrips, cache.hits, r_lo_hit.R, r_hi.R, "PASS" if g_b else "FAIL"), flush=True)

    # ---- GATE C: each key knob moves the key; unchanged repeat matches ----
    base = _design()
    e = {"film": _uniform_tensor(6.1)}
    k0 = _key(base, e, LAM, 1.0 + 0j, 1.5 + 0j, "")
    moves = {
        "repeat": _key(base, e, LAM, 1.0 + 0j, 1.5 + 0j, "") == k0,             # must be SAME
        "lambda": _key(base, e, 1.55e-6, 1.0 + 0j, 1.5 + 0j, "") != k0,
        "n_super": _key(base, e, LAM, 1.001 + 0j, 1.5 + 0j, "") != k0,
        "n_sub": _key(base, e, LAM, 1.0 + 0j, 1.6 + 0j, "") != k0,
        "angle": _key(_design(theta=20.0), e, LAM, 1.0 + 0j, 1.5 + 0j, "") != k0,
        "pol": _key(_design(pol="p"), e, LAM, 1.0 + 0j, 1.5 + 0j, "") != k0,
        "side": _key(_design(side="bottom"), e, LAM, 1.0 + 0j, 1.5 + 0j, "") != k0,
        "mesh3d": _key(_design(mesh=Mesh3DSpec(pml_thk_m=600e-9)), e, LAM, 1.0 + 0j, 1.5 + 0j, "") != k0,
        "eps": _key(base, {"film": _uniform_tensor(6.5)}, LAM, 1.0 + 0j, 1.5 + 0j, "") != k0,
    }
    g_c = bool(all(moves.values()))
    ok = ok and g_c
    print("[cache] GATE C: key knobs {} -> {}".format(
        {k: ("ok" if v else "MISS") for k, v in moves.items()}, "PASS" if g_c else "FAIL"), flush=True)

    # ---- GATE D: a stale-schema cache file is discarded on load ----
    fmts = available_formats()
    if not fmts:
        print("[cache] GATE D: SKIP (no h5py/zarr backend installed -- on-disk schema check not run)",
              flush=True)
        g_d = True
    else:
        ext = ".h5" if "hdf5" in fmts else ".zarr"
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "stale" + ext)
            # write a store under the OLD schema (2) carrying a bogus packed entry
            save_arrays(path, {"kBOGUS": np.zeros(len(C._VEC))},
                        {"schema": 2, "tag": "", "layout": list(C._VEC)})
            stale = OpticalSolverCache(_stub_solver, path, autosave=False)
            discarded = len(stale._mem) == 0                       # schema-2 entries must be dropped
            # a CURRENT-schema file (packed layout: (N,12) values + (N,41) keys) must load normally
            path2 = os.path.join(td, "fresh" + ext)
            kk = "k" + "0" * 40
            save_arrays(path2, {C._PK_KEYS: np.frombuffer(kk.encode("ascii"),
                                                          dtype=np.uint8).reshape(1, -1),
                                C._PK_VALS: np.zeros((1, len(C._VEC)))},
                        {"schema": C._SCHEMA, "tag": "", "layout": list(C._VEC)})
            loaded = kk in OpticalSolverCache(_stub_solver, path2, autosave=False)._mem
            g_d = bool(discarded and loaded)
        print("[cache] GATE D: stale schema-2 discarded {}, current schema-{} loads {} -> {}".format(
            discarded, C._SCHEMA, loaded, "PASS" if g_d else "FAIL"), flush=True)
    ok = ok and g_d

    # ---- GATE D2: flush into a stale-schema file truncates it (no resurrection on reopen) ----
    if not fmts:
        print("[cache] GATE D2: SKIP (no h5py/zarr backend -- stale-truncate check not run)", flush=True)
        g_d2 = True
    else:
        ext = ".h5" if "hdf5" in fmts else ".zarr"
        stale_key = "k" + "f" * 40                             # a plausible mis-keyed old-schema entry
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "resurrect" + ext)
            # a store written under the PREVIOUS schema, carrying one bogus packed entry
            save_arrays(path, {stale_key: np.zeros(len(C._VEC))},
                        {"schema": C._SCHEMA - 1, "tag": "", "layout": list(C._VEC)})
            c1 = OpticalSolverCache(_stub_solver, path, autosave=True, autosave_every=1)
            discarded2 = len(c1._mem) == 0                     # load-side discard (GATE D semantics)
            d = _design()
            e = {"film": _uniform_tensor(6.2)}
            r_fresh = c1(d, None, e, LAM, 1.0 + 0j, 1.5 + 0j)  # miss -> autosave flush INTO the stale file
            # raw on-disk check: the stale entry must be physically GONE, not just unread
            raw, meta = load_arrays(path)
            stale_gone = stale_key not in raw
            restamped = int(meta.get("schema", -1)) == C._SCHEMA
            # reopen under the fresh stamp: exactly the fresh entry, served as a HIT
            c2 = OpticalSolverCache(_stub_solver, path, autosave=False)
            only_fresh = (set(c2._mem) == set(c1._mem) and len(c2._mem) == 1
                          and stale_key not in c2._mem)
            r_hit = c2(d, None, e, LAM, 1.0 + 0j, 1.5 + 0j)
            served_fresh = c2.hits == 1 and _same_result(r_hit, r_fresh)
            g_d2 = bool(discarded2 and stale_gone and restamped and only_fresh and served_fresh)
        print("[cache] GATE D2: discard {}, stale entry gone after flush {}, restamped {}, reopen "
              "fresh-only {}, HIT==fresh {} -> {}".format(
                  discarded2, stale_gone, restamped, only_fresh, served_fresh,
                  "PASS" if g_d2 else "FAIL"), flush=True)
    ok = ok and g_d2

    # ---- GATE E: flush -> reopen round-trip (pins flush()/load _SCHEMA agreement) ----
    # GATE B is in-memory (autosave off) and GATE D hand-writes via save_arrays, so neither exercises
    # the cache's OWN flush() write path. A flush() that stamped a schema the load-discard rejects would
    # make the cache write-only (re-solve every reopen); this round-trip catches that.
    if not fmts:
        print("[cache] GATE E: SKIP (no h5py/zarr backend -- flush round-trip not run)", flush=True)
        g_e = True
    else:
        ext = ".h5" if "hdf5" in fmts else ".zarr"
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "rt" + ext)
            d = _design()
            e = {"film": _uniform_tensor(6.3)}
            c1 = OpticalSolverCache(_stub_solver, path, autosave=True, autosave_every=1)  # per-miss flush
            r_miss = c1(d, None, e, LAM, 1.0 + 0j, 1.5 + 0j)               # miss -> compute + flush
            c1(d, None, {"film": _uniform_tensor(6.7)}, LAM, 1.0 + 0j, 1.5 + 0j)   # 2nd packed row
            c2 = OpticalSolverCache(_stub_solver, path, autosave=False)    # reopen FRESH from disk
            r_hit = c2(d, None, e, LAM, 1.0 + 0j, 1.5 + 0j)
            survived = (c2.hits == 1 and c2.misses == 0 and _same_result(r_hit, r_miss))
            # BIT-identity across the store: the reopened _mem must be BYTE-identical to what was
            # flushed (the packed (N,12)+(N,41) layout must round-trip float64 bits + keys exactly)
            bit_identical = (set(c2._mem) == set(c1._mem) and all(
                c1._mem[k].dtype == c2._mem[k].dtype and c1._mem[k].tobytes() == c2._mem[k].tobytes()
                for k in c1._mem))
        g_e = bool(survived and bit_identical)
        print("[cache] GATE E: flush->reopen survives as HIT {} (hits {}, misses {}), _mem "
              "byte-identical {} -> {}".format(survived, c2.hits, c2.misses, bit_identical,
                                               "PASS" if g_e else "FAIL"), flush=True)
    ok = ok and g_e

    print("[cache] *** OPTICAL CACHE: {} ***".format("PASS" if ok else "FAIL"), flush=True)
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
