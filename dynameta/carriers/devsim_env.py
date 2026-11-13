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
them, and with the 2019 build that pressure ends in an access violation where the current builds
raise an ordinary error. scipy's build also factorises 1.31x faster inside UMFPACK (18k-node 2-D
bipolar diode: 72 s against Elmer's 94 s).

THE PIN (when the user has NOT set DEVSIM_MATH_LIBS -- a user value always wins, untouched):
  1. pip `mkl` installed in this environment -> DEVSIM_MATH_LIBS = that mkl_rt by FULL path. This
     is what devsim already picks when mkl is present (mkl_rt first, then MKL-PARDISO as the
     direct solver instead of UMFPACK -- measured: info['direct_solver'] == 'mkl_pardiso'); the
     pin keeps that choice and only removes the PATH lookup. It ALSO sets DEVSIM_NEW_SYMBOLIC=1
     unless the user set it: DEVSIM (>= 2.6.0) otherwise reuses the PARDISO symbolic
     factorisation from the second Newton iteration on, and with it
     validation/bipolar_diode_2d.py did not converge at ANY bias (3/3 "Convergence failure!",
     MEASURED in a pip-mkl venv, D0 2026-11-03). With a fresh symbolic factorisation per
     iteration it converges, the currents match UMFPACK to <= 2e-16 relative, and DEVSIM's solve
     time on that 18k-node device is 57 s against UMFPACK's 73 s (66 s vs 132 s on a loaded box;
     docs/audit/2026-11-03-devsim-2.11-core.md, section 4).
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

__all__ = ["configure_math_libs", "import_devsim", "math_libs_decision", "MODE_ENV",
           "UMFPACK_2BY2_ENV", "umfpack_2by2_guard"]

MODE_ENV = "DYNAMETA_DEVSIM_BLAS"           # 'auto' (default) | 'pin' | 'off'
UMFPACK_2BY2_ENV = "DYNAMETA_UMFPACK_2BY2"  # 'symmetric' (default) | 'keep'

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
        os.environ.setdefault("DEVSIM_NEW_SYMBOLIC", "1")   # PARDISO symbolic reuse: see docstring
        return _decide("mkl", mkl)
    ob = _scipy_openblas() if _devsim_honours_blas_prefix() else None
    if ob is not None:
        os.environ["DEVSIM_MATH_LIBS"] = ob
        os.environ["DEVSIM_BLAS_PREFIX"] = "scipy_"
        os.environ["DEVSIM_BLAS_SUFFIX"] = "_"
        return _decide("scipy-openblas", ob, "scipy_", "_")
    return _decide("devsim-default")


def import_devsim():
    """`configure_math_libs()` then `import devsim`, then `umfpack_2by2_guard()`; returns the devsim
    module. Raises the original ImportError when devsim is not installed."""
    configure_math_libs()
    import devsim
    umfpack_2by2_guard()
    return devsim


# ---------------------------------------------------------------------------------------------------
# UMFPACK 5.1: the obsolete 2-by-2 strategy (v0.15.1, 2026-11-10;
# docs/audit/2026-11-10-v0.15.1-dg-iteration-cost.md)
# ---------------------------------------------------------------------------------------------------
# UMFPACK 5.1 umfpack.h indices (the version devsim/umfpack/umfpack_lgpl ships: "UMFPACK V5.1.0 (May 31,
# 2007)" in its own report). Control[UMFPACK_STRATEGY] = 5. Info, each index checked against
# umfpack_di_report_info on the same factorization (the eight DG matrices of the note and the 2-D diode):
# [18] UMFPACK_STRATEGY_USED: "strategy used: symmetric 2-by-2" exactly when == 2 ("symmetric" 3,
# "unsymmetric" 1); [23] / [24] UMFPACK_LNZ_ESTIMATE / UMFPACK_UNZ_ESTIMATE: the "upper bound" column of
# "nz in L (incl diagonal)" / "nz in U (incl diagonal)" (e.g. 9601722 / 9875001 = Info[23] / Info[24] on the
# 2-D electron + hole DG matrix under 2-by-2, 6910000 / 7085696 under symmetric).
_UMF_CONTROL_STRATEGY = 5
_UMF_INFO_STRATEGY_USED = 18
_UMF_INFO_LNZ_ESTIMATE, _UMF_INFO_UNZ_ESTIMATE = 23, 24
_UMF_STRATEGY_AUTO, _UMF_STRATEGY_2BY2, _UMF_STRATEGY_SYMMETRIC = 0, 2, 3
# Keep AUTO's 2-by-2 unless its symbolic upper bound on nz(L+U) exceeds the symmetric strategy's by more than
# this factor (docs/audit/2026-11-10-v0.15.1-dg-iteration-cost.md, section 12: measured <= 1.070 wherever
# 2-by-2 factorized as sparsely as symmetric, >= 1.291 wherever it filled in 4-45x).
_UMF_2BY2_BOUND_RATIO = 1.15
_GUARD: dict | None = None


def umfpack_2by2_guard() -> dict:
    """Keep devsim's bundled UMFPACK 5.1 off its obsolete 2-by-2 strategy WHERE THAT STRATEGY FILLS IN.
    Returns the (live) record {'installed': bool, 'mode': 'symmetric' | 'keep', 'reason': str,
    'symbolic': int, 'auto_2by2': int, 'replaced': int, 'kept': int}: 'symbolic' counts the fresh symbolic
    factorizations seen since install, 'auto_2by2' how many of them AUTO sent to the 2-by-2 strategy,
    'replaced' how many of those the guard redid with the symmetric strategy and 'kept' how many it left on
    2-by-2 (replaced + kept == auto_2by2). Idempotent; called by import_devsim().

    WHAT. devsim 2.11.2 (the Windows and the Linux pip wheel alike) factorizes every Newton step with the
    UMFPACK 5.1 of its Python shim devsim/umfpack/umfshim.py, under UMFPACK's default Control, i.e.
    strategy AUTO. On many matrices 5.1's AUTO picks the "symmetric 2-by-2" strategy (measured: every
    density-gradient + quasi-equilibrium-closure Jacobian, the heterojunction suites, the 2-D bipolar
    diode). UMFPACK itself disabled that strategy in 5.4.0 (May 2009; ChangeLog: "disabled 2-by-2
    strategy"). On the DG + closure Jacobians it fills in (4-45x the nonzeros of the symmetric strategy on
    the same ordering, up to 150x the numeric time); elsewhere it factorizes as sparsely as the symmetric
    strategy and its solves are MORE accurate (2-D bipolar diode: componentwise backward error 9.0e-11
    against 5.0e-8 under symmetric; 1-D electron DG Newton steps: 9e-14 against up to 3.3e-5).

    RULE (v0.15.1 v2). When AUTO's symbolic step picks 2-by-2, the guard also runs the symbolic step with
    UMFPACK_STRATEGY_SYMMETRIC and compares the two symbolic upper bounds on nz(L) + nz(U)
    (Info[UMFPACK_LNZ_ESTIMATE] + Info[UMFPACK_UNZ_ESTIMATE]). It keeps AUTO's 2-by-2 factorization (Info
    restored, so devsim's call sees exactly its own symbolic step) unless the 2-by-2 bound exceeds the
    symmetric one by more than _UMF_2BY2_BOUND_RATIO = 1.15; then the symmetric Symbolic object replaces it.
    Measured (docs/audit/2026-11-10-v0.15.1-dg-iteration-cost.md, section 12; Windows, unguarded runs that
    also factor every 2-by-2 matrix with the symmetric strategy): bound ratio 0.60-1.070 on every matrix
    where 2-by-2's actual nz(L+U) stayed within 0.99-1.15x the symmetric one (2-D diode 1.03, valley-DG and
    quasi-equilibrium suites <= 1.070, heterojunction 1.000); 1.291-117 on every matrix where it filled in
    (actual 4.2-45x). The bound is computed from the pattern before any numeric pivoting, and the decision
    costs one extra symbolic step per fresh symbolic factorization that AUTO sent to 2-by-2, as before.
    v1 (5ee2c8d .. e29fbd3) replaced EVERY 2-by-2 choice; on the 2-D bipolar diode (18,357 nodes) that cost
    ~1000x backward error (9.6e-8 against 9.0e-11) for nothing (the factor time was the same within the
    run-to-run spread).

    WHERE. The guard wraps devsim.umfpack.umfpack_loader.umf_control.symbolic, the one call through which
    every symbolic step of devsim's UMFPACK shim goes (umfshim.dsobject.factor; DEVSIM reuses that Symbolic
    object for the is_same_symbolic numeric steps of a Newton solve, so each decision holds for them too).
    It is a class attribute, so it also survives ds.reset_devsim() (which re-registers umfshim's callback).
    Not installed (reason says why) when devsim did not load its UMFPACK shim (e.g. pip `mkl` present:
    MKL-PARDISO is then the direct solver; not measured here) or the loader lacks the expected names.
    DYNAMETA_UMFPACK_2BY2=keep leaves devsim's AUTO choice untouched everywhere (the pre-v0.15.1
    factorization, for A/B comparisons); the variable is read once, at the first call."""
    global _GUARD
    if _GUARD is not None:
        return _GUARD
    mode = os.environ.get(UMFPACK_2BY2_ENV, "symmetric").strip().lower() or "symmetric"
    if mode not in ("symmetric", "keep"):
        raise ValueError("{}={!r}: expected 'symmetric' or 'keep'".format(UMFPACK_2BY2_ENV, mode))
    guard = {"installed": False, "mode": mode, "reason": "", "symbolic": 0, "auto_2by2": 0, "replaced": 0,
             "kept": 0}
    loader = sys.modules.get("devsim.umfpack.umfpack_loader")
    ctl_cls = getattr(loader, "umf_control", None) if loader is not None else None
    if mode == "keep":
        guard["reason"] = "{}=keep".format(UMFPACK_2BY2_ENV)
    elif "devsim.umfpack.umfshim" not in sys.modules or ctl_cls is None:
        guard["reason"] = "devsim did not load its UMFPACK 5.1 shim (another direct solver)"
    elif not callable(getattr(ctl_cls, "symbolic", None)):
        guard["reason"] = "devsim.umfpack.umfpack_loader has an unexpected layout"
    else:
        original = ctl_cls.symbolic

        def symbolic(self, matrix):
            sym = original(self, matrix)
            guard["symbolic"] += 1
            if (self.Control[_UMF_CONTROL_STRATEGY] != _UMF_STRATEGY_AUTO
                    or self.Info[_UMF_INFO_STRATEGY_USED] != _UMF_STRATEGY_2BY2):
                return sym
            guard["auto_2by2"] += 1
            info_2by2 = self.Info[:]
            bound_2by2 = info_2by2[_UMF_INFO_LNZ_ESTIMATE] + info_2by2[_UMF_INFO_UNZ_ESTIMATE]
            self.Control[_UMF_CONTROL_STRATEGY] = _UMF_STRATEGY_SYMMETRIC
            try:
                sym_s = original(self, matrix)
            finally:
                self.Control[_UMF_CONTROL_STRATEGY] = _UMF_STRATEGY_AUTO
            bound_s = self.Info[_UMF_INFO_LNZ_ESTIMATE] + self.Info[_UMF_INFO_UNZ_ESTIMATE]
            if bound_2by2 > _UMF_2BY2_BOUND_RATIO * bound_s:
                del sym                     # di_symbolic.__del__ frees the AUTO one
                guard["replaced"] += 1
                return sym_s
            del sym_s
            self.Info[:] = info_2by2        # devsim's call sees its own (AUTO) symbolic step
            guard["kept"] += 1
            return sym

        symbolic.dynameta_original = original
        ctl_cls.symbolic = symbolic
        guard["installed"] = True
        guard["reason"] = ("AUTO's 2-by-2 choice is redone with the symmetric strategy where its symbolic "
                           "nz(L+U) bound exceeds the symmetric one by more than {}x".format(_UMF_2BY2_BOUND_RATIO))
    _GUARD = guard
    return _GUARD
