"""THE single `import devsim` home: pin DEVSIM's BLAS/LAPACK to a pip-installed library BEFORE the
first import (D0, DEVSIM 2.11.2 core groundwork, 2026-11-03).

Every dynameta module that needs devsim gets it from `import_devsim()` (or the lazy shim in
eq_registry, which calls it), and so do the in-repo validation scripts; tests/conftest.py calls
`configure_math_libs()` first. tests/test_devsim_env.py gates that no other module imports devsim
directly.

WHY (MEASURED on this Windows box, devsim 2.11.2). DEVSIM loads its math library ONCE, at the first
`import devsim` (2.11.2 CHANGES, issue #169), searching in this order: the libraries named in
DEVSIM_MATH_LIBS (';'-separated on Windows, ':' elsewhere); else the Intel MKL (mkl_rt.N -- on
Windows devsim/__init__.py appends <sys.prefix>/Library/bin, where pip `mkl` puts it, to PATH for
that search only); else the bare names "libopenblas.dll;liblapack.dll;libblas.dll", resolved by
the OS loader. On Windows that last step is a PATH search, so WHICHEVER program put a
libopenblas.dll on PATH supplies devsim's BLAS -- here the OpenBLAS 0.3.7 (2019, DYNAMIC_ARCH
detecting this Zen CPU as "Barcelona", MAX_THREADS=64) of an Elmer 8.4 install. That library is
also the BLAS of devsim's UMFPACK 5.1 direct solver (devsim/umfpack/umfpack_loader.py loads
`info['math_libraries']`), so it touches every Newton step. Under concurrent load
(24 copies of tests/test_audit_2026_07_25_carriers_hygiene.py::test_v20_x21_* at once, 3 rounds)
the fresh-process import crashed 9/72 times with 0xC0000005 right after
`Loading "libopenblas.dll": ALL BLAS/LAPACK LOADED`: every OpenBLAS instance COMMITS ~32 MB per
thread at load (Elmer's +995 MB / 31 threads; scipy's +738 MB / 23 threads; numpy's ILP64 copy
+738 MB; ngsolve_openblas +3979 MB), a hygiene child holds ~6.9 GB of private commit with four of
them, and the 2019 build dereferences a failed allocation where current builds return an error.

THE PIN (when the user has NOT set DEVSIM_MATH_LIBS -- a user value always wins, untouched):
  1. pip `mkl` installed in this environment -> DEVSIM_MATH_LIBS = that mkl_rt by FULL path. This
     is what devsim already picks when mkl is present (mkl_rt first, then MKL-PARDISO as the
     direct solver instead of UMFPACK -- measured: info['direct_solver'] == 'mkl_pardiso'); the
     pin keeps that choice and only removes the PATH lookup.
  2. else the LP64 OpenBLAS bundled in the installed scipy wheel (scipy.libs/libscipy_openblas-
     <hash>.dll|.so; scipy is a core dependency), with DEVSIM_BLAS_PREFIX='scipy_' and
     DEVSIM_BLAS_SUFFIX='_' (devsim 2.11.1 issue #167 -- the documented way to use scipy's
     OpenBLAS). The file is located through scipy's installed RECORD (importlib.metadata), never
     PATH, and its LP64 symbols are checked (ctypes) before it is chosen. scipy loads this same
     file on `import dynameta.carriers`, so devsim then reuses an already-initialised library
     (one OpenBLAS thread pool and ~1 GB less commit per process) instead of loading another.
     numpy's bundled copy is NOT a candidate: it is ILP64 (USE64BITINT, symbols *_64_), devsim
     needs 32-bit integers.
  3. else nothing found -> devsim's own search, unchanged.
PLATFORMS: the pin is applied by default on Windows only (where the fallback is a PATH search).
On Linux/macOS devsim resolves bare names through the dynamic loader (RPATH $ORIGIN/../lib, the
ld.so cache -- e.g. the CI solvers leg's apt libopenblas), which is deterministic, so the default
there is unchanged; DYNAMETA_DEVSIM_BLAS=pin applies the same resolution on any platform (the .so
/ .dylib names are handled), DYNAMETA_DEVSIM_BLAS=off disables it everywhere.

The choice is exported through os.environ, so child processes that import devsim inherit it (and
treat it as a user setting). Stdlib-only at import time; nothing happens until
configure_math_libs() / import_devsim() is called.
"""

from __future__ import annotations

import os
import re
import sys

__all__ = ["configure_math_libs", "import_devsim", "math_libs_decision", "MODE_ENV"]

MODE_ENV = "DYNAMETA_DEVSIM_BLAS"           # 'auto' (default) | 'pin' | 'off'

# Installed-file name patterns (basename). mkl: mkl_rt.dll / mkl_rt.3.dll (Windows),
# libmkl_rt.so[.N] (Linux), libmkl_rt[.N].dylib (macOS). scipy: the LP64 bundle
# libscipy_openblas-<hash>.{dll,so} (Windows/Linux, scipy.libs/) or libscipy_openblas.dylib
# (macOS, scipy/.dylibs/); the ILP64 libscipy_openblas64_* is excluded by the pattern.
_MKL_RE = re.compile(r"^(lib)?mkl_rt(\.(\d+))?\.(dll|dylib|so(\.\d+)*)$|^libmkl_rt\.so\.(\d+)$")
_SCIPY_OB_RE = re.compile(r"^libscipy_openblas(-[0-9a-f]+)?\.(dll|dylib|so(\.\d+)*)$")

# LP64 BLAS + LAPACK entry points (scipy_ prefix, _ suffix) that must resolve before the bundled
# scipy OpenBLAS is handed to devsim -- a guard against a future scipy changing its symbol naming
# (devsim would otherwise fail its import with "MISSING SYMBOLS" / "Issues initializing DEVSIM").
_SCIPY_PROBE_SYMBOLS = ("scipy_dgemm_", "scipy_dgetrf_", "scipy_dgetrs_", "scipy_zgetrf_",
                        "scipy_zgetrs_")

_DECISION: dict | None = None


def _dist_file(dist_name: str, pattern: "re.Pattern[str]") -> list:
    """Absolute paths of the files that pip installed for `dist_name` whose basename matches
    `pattern` (from the distribution's RECORD -- never a PATH search). [] when not installed."""
    try:
        from importlib.metadata import PackageNotFoundError, distribution
        try:
            dist = distribution(dist_name)
        except PackageNotFoundError:
            return []
        out = []
        for f in dist.files or ():
            if pattern.match(f.name):
                p = os.path.abspath(str(dist.locate_file(f)))
                if os.path.isfile(p):
                    out.append(p)
        return sorted(set(out))
    except Exception:
        return []


def _mkl_rt() -> str | None:
    """pip `mkl`'s mkl_rt (highest ABI number when several are recorded)."""
    hits = _dist_file("mkl", _MKL_RE)

    def _abi(p):
        m = re.search(r"mkl_rt\.(\d+)", os.path.basename(p))
        return int(m.group(1)) if m else -1
    return max(hits, key=_abi) if hits else None


def _scipy_openblas() -> str | None:
    """The LP64 OpenBLAS bundled in the installed scipy wheel, if exactly one is recorded and its
    scipy_-prefixed LP64 symbols resolve."""
    hits = _dist_file("scipy", _SCIPY_OB_RE)
    if len(hits) != 1:                       # absent (conda / Accelerate builds) or ambiguous
        return None
    try:
        import ctypes
        lib = ctypes.CDLL(hits[0])           # the very file scipy loads: same module, no new pool
        for sym in _SCIPY_PROBE_SYMBOLS:
            getattr(lib, sym)
    except (OSError, AttributeError):
        return None
    return hits[0]


def _is_windows() -> bool:
    return os.name == "nt"


def _devsim_honours_blas_prefix() -> bool:
    """True when the INSTALLED devsim reads DEVSIM_BLAS_PREFIX / _SUFFIX (>= 2.11.1, issue #167).
    An older devsim would look up the bare dgemm_ in scipy's scipy_dgemm_ library and fail its
    import with MISSING SYMBOLS, so the scipy option is skipped there (read from the installed
    metadata, without importing devsim)."""
    try:
        from importlib.metadata import version
        nums = re.findall(r"\d+", version("devsim"))[:3]
        return tuple(int(x) for x in nums) + (0,) * (3 - len(nums)) >= (2, 11, 1)
    except Exception:
        return False


def math_libs_decision() -> dict | None:
    """What configure_math_libs() decided in this process (None before the first call):
    {'source': 'user' | 'mkl' | 'scipy-openblas' | 'devsim-default' | 'off' | 'posix-default'
               | 'already-imported', 'math_libs': str | None, 'prefix': str | None,
     'suffix': str | None}."""
    return None if _DECISION is None else dict(_DECISION)


def configure_math_libs() -> dict:
    """Set DEVSIM_MATH_LIBS (+ DEVSIM_BLAS_PREFIX / _SUFFIX) to a pip-installed library before the
    first `import devsim` -- see the module docstring for the order and the evidence. Idempotent:
    the first call decides, later calls return the same decision. Never overrides a
    DEVSIM_MATH_LIBS the user set; a no-op once devsim is imported (its libraries are loaded)."""
    global _DECISION
    if _DECISION is not None:
        return dict(_DECISION)
    mode = os.environ.get(MODE_ENV, "auto").strip().lower() or "auto"
    if mode not in ("auto", "pin", "off"):
        raise ValueError("{}={!r}: expected 'auto', 'pin' or 'off'".format(MODE_ENV, mode))

    def _decide(source, libs=None, prefix=None, suffix=None):
        global _DECISION
        _DECISION = {"source": source, "math_libs": libs, "prefix": prefix, "suffix": suffix}
        return dict(_DECISION)

    if os.environ.get("DEVSIM_MATH_LIBS"):
        return _decide("user", os.environ["DEVSIM_MATH_LIBS"],
                       os.environ.get("DEVSIM_BLAS_PREFIX"), os.environ.get("DEVSIM_BLAS_SUFFIX"))
    if "devsim" in sys.modules:
        return _decide("already-imported")
    if mode == "off":
        return _decide("off")
    if mode == "auto" and not _is_windows():
        return _decide("posix-default")

    mkl = _mkl_rt()
    if mkl is not None:
        os.environ["DEVSIM_MATH_LIBS"] = mkl
        return _decide("mkl", mkl)
    ob = _scipy_openblas() if _devsim_honours_blas_prefix() else None
    if ob is not None:
        os.environ["DEVSIM_MATH_LIBS"] = ob
        os.environ["DEVSIM_BLAS_PREFIX"] = "scipy_"
        os.environ["DEVSIM_BLAS_SUFFIX"] = "_"
        return _decide("scipy-openblas", ob, "scipy_", "_")
    return _decide("devsim-default")


def import_devsim():
    """`configure_math_libs()` then `import devsim`; returns the devsim module. Raises the original
    ImportError when devsim is not installed."""
    configure_math_libs()
    import devsim
    return devsim
