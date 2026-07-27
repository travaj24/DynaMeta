"""Numba threading-layer resilience: expose the pip-installed TBB runtime to numba, PROBE the
default layer selection with a REPRESENTATIVE kernel out-of-process, and force the built-in
'workqueue' layer when the selected layer is broken at runtime.

Why a subprocess probe: a broken parallel runtime does NOT raise -- the first parallel-kernel
launch either LIVELOCKS (all cores spinning forever) or hard-crashes the interpreter with no
traceback, and neither is catchable in-process. Numba's own 'default' selection only skips a
layer that fails to LOAD, not one that wedges at launch.

Why the REAL kernel: the 2026-07-18 Windows incident showed the failure is KERNEL-DEPENDENT --
trivial prange loops (with or without fastmath/transcendentals) ran fine under the OpenMP
layer while the repo's large fused FDTD kernel died silently every time. A toy probe therefore
proves nothing; the probe below JIT-compiles and runs a tiny case of the actual
optics.fdtd_nd.solve_fdtd_2d numba backend (the smallest member of the failing class -- the
header used to name solve_fdtd_1d, which _PROBE_SRC has never called) in a sacrificial child
under a timeout.

Why the DLL exposure: the pip 'tbb' wheel drops tbb12.dll into <sys.prefix>/Library/bin -- a
conda-style path a python.org install never searches, so numba's tbbpool binding fails with
DLL-not-found and numba silently falls through to OpenMP. _expose_tbb_dlls() adds that
directory via os.add_dll_directory when present, which makes TBB the default layer again
(numba's preference order: tbb > omp > workqueue).

Cost control: (1) an explicit NUMBA_THREADING_LAYER always wins -- no probe; (2) POSIX skips
by default (the breakage class is Windows DLL/runtime rot; CI pays nothing -- pass
windows_only=False to probe anywhere); (3) no numba -> nothing to do; (4) a CONCLUSIVE verdict is
CACHED per (python, numba version, tbb version, probe-source hash, dynameta version) key with a
TTL, so even the broken machine pays the probe once a day and a healthy box pays one small-kernel
JIT (~15-30 s) once a day.

AUDIT T-11 -- the verdict is no longer indiscriminate. Only two child outcomes are EVIDENCE
about the threading layer, and they are the two documented failure modes:
  * the child never finished (`TimeoutExpired`)               -> 'workqueue-fallback:timeout'
  * the child died with no Python traceback (hard crash)      -> 'workqueue-fallback:crash'
Everything else tells us nothing about the layer and must NOT force workqueue for the session
and all its children: a failed import in the child ('probe-error:import'), any other Python
exception ('probe-error:exception'), a child that could not be launched at all
('probe-error:launch', e.g. FileNotFoundError on sys.executable) and a clean exit with no
LAYER_OK marker ('probe-error:no-verdict') leave numba's own selection untouched, print the
child's stderr tail instead of the misleading "livelock/crash" line, and are NOT cached -- so a
source fix is re-probed on the next session rather than serialising every parallel kernel for a
day. The cache key carries the numba AND tbb versions (plus the probe source and dynameta
version), so a runtime upgrade invalidates a stale negative verdict.

No import-time side effects: call ensure_working_threading_layer() explicitly (tests/conftest
does) BEFORE the first parallel kernel launch of the process."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time

__all__ = ["ensure_working_threading_layer"]

# Sacrificial-child source: expose TBB DLLs, run the SMALLEST member of the failing kernel
# class (a tiny chi2 solve through the numba FDTD backend), report the layer that served it.
_PROBE_SRC = (
    "import os, sys\n"
    "d = os.path.join(sys.prefix, 'Library', 'bin')\n"
    "if os.path.isdir(d):\n"
    "    os.add_dll_directory(d)\n"
    "from dynameta.optics.fdtd import FDTDLayer\n"
    "from dynameta.optics.fdtd_nd import solve_fdtd_2d\n"
    "res = solve_fdtd_2d([FDTDLayer(150e-9, eps_inf=2.0, chi2_m_V=1e-13)],\n"
    "                    period_x_m=100e-9, lambda_min_m=1.0e-6, lambda_max_m=1.4e-6,\n"
    "                    resolution=10, backend='numba')\n"
    "import numba\n"
    "try:\n"
    "    layer = numba.threading_layer()\n"
    "except Exception:\n"
    "    layer = 'unknown'\n"
    "print('LAYER_OK:' + layer)\n"
)


def _cache_path() -> str:
    return os.path.join(tempfile.gettempdir(),
                        "dynameta_numba_layer_probe_py{}{}.json".format(*sys.version_info[:2]))


def _tbb_version() -> str:
    """Version of the installed pip 'tbb' runtime, or 'dll-only' / 'absent' (AUDIT T-11: the
    cache key must invalidate on a runtime UPGRADE, not just on present/absent)."""
    try:
        from importlib.metadata import PackageNotFoundError, version
        try:
            return str(version("tbb"))
        except PackageNotFoundError:
            pass
    except Exception:
        pass
    d = os.path.join(sys.prefix, "Library", "bin")
    return "dll-only" if os.path.isfile(os.path.join(d, "tbb12.dll")) else "absent"


def _dynameta_version() -> str:
    try:
        from dynameta import __version__               # fully imported by the time we are called
        return str(__version__)
    except Exception:
        return "unknown"


def _cache_key(numba_version: str) -> str:
    """AUDIT T-11: key the verdict on everything that can legitimately flip it -- the numba and
    tbb versions, the probe source itself and the dynameta version whose kernel it runs."""
    return "numba-{}-tbb-{}-probe-{}-dm-{}".format(
        numba_version, _tbb_version(),
        hashlib.sha1(_PROBE_SRC.encode("utf-8")).hexdigest()[:8], _dynameta_version())


def _stderr_tail(err: str, n_lines: int = 3, n_chars: int = 400) -> str:
    lines = [ln for ln in (err or "").strip().splitlines() if ln.strip()]
    return " | ".join(lines[-n_lines:])[:n_chars] if lines else "(no stderr)"


def _classify(returncode: int, stdout: str, stderr: str) -> tuple[str, str]:
    """Map a COMPLETED probe child onto (verdict, diagnostic). AUDIT T-11: the three outcomes the
    old code collapsed -- a hard crash, a Python-level failure in the probe body, and a clean exit
    that printed nothing -- are distinguished here, and only the crash is evidence about the
    threading layer."""
    if returncode == 0 and "LAYER_OK:" in (stdout or ""):
        return "default-ok:{}".format(stdout.split("LAYER_OK:", 1)[1].strip()), ""
    if returncode == 0:
        return "probe-error:no-verdict", "child exited 0 without printing LAYER_OK"
    if "Traceback (most recent call last)" in (stderr or ""):
        # a Python-level failure (bad signature, missing dependency, guard raise, JIT error):
        # says nothing about whether the threading layer wedges.
        kind = ("import" if ("ModuleNotFoundError" in stderr or "ImportError" in stderr)
                else "exception")
        return "probe-error:{}".format(kind), _stderr_tail(stderr)
    # non-zero exit with NO Python traceback == the interpreter died where it stood, which is the
    # module header's second documented failure mode (hard crash, e.g. 0xC0000005 / a signal).
    return "workqueue-fallback:crash", "child exited {} with no Python traceback: {}".format(
        returncode, _stderr_tail(stderr))


def _expose_tbb_dlls() -> bool:
    """Make the pip 'tbb' wheel's runtime findable (module header). Returns True when the
    directory exists and was added (idempotent; harmless when absent)."""
    d = os.path.join(sys.prefix, "Library", "bin")
    if os.name == "nt" and os.path.isdir(d) and os.path.isfile(os.path.join(d, "tbb12.dll")):
        try:
            os.add_dll_directory(d)
            return True
        except Exception:
            return False
    return False


def ensure_working_threading_layer(*, timeout_s: float = 180.0, ttl_s: float = 86400.0,
                                   windows_only: bool = True, verbose: bool = True) -> str:
    """Select a WORKING numba threading layer (module header for the rationale). Returns the
    decision: 'explicit' (NUMBA_THREADING_LAYER already set -- untouched), 'posix-skip',
    'no-numba', 'default-ok:<layer>' (probe passed; numba's own selection kept),
    'workqueue-fallback:timeout' / 'workqueue-fallback:crash' (the selected layer livelocked or
    hard-crashed on the representative kernel -> NUMBA_THREADING_LAYER=workqueue exported for
    this process and its children), or 'probe-error:{import,exception,launch,no-verdict}' (the
    probe itself failed, so nothing is known about the layer: numba's selection is left alone and
    the verdict is NOT cached -- AUDIT T-11). Safe to call repeatedly; conclusive verdicts cached
    for ttl_s."""
    if os.environ.get("NUMBA_THREADING_LAYER"):
        _expose_tbb_dlls()                           # explicit tbb choice still needs the DLLs
        return "explicit"
    if windows_only and os.name != "nt":
        return "posix-skip"
    try:
        import numba
    except Exception:
        return "no-numba"
    _expose_tbb_dlls()                               # presence/version is folded into _tbb_version()

    cache = _cache_path()
    key = _cache_key(str(numba.__version__))

    def _apply(verdict: str, why: str = "") -> str:
        if verdict.startswith("default-ok"):
            return verdict
        if verdict.startswith("probe-error"):
            # AUDIT T-11: no evidence about the layer -> do NOT serialise every parallel kernel.
            if verbose:
                print("[dynameta.numba_env] the threading-layer probe did not run to a verdict "
                      "({}): {}. numba's own layer selection is UNCHANGED and this result is not "
                      "cached.".format(verdict, why or "no diagnostic"), flush=True)
            return verdict
        os.environ["NUMBA_THREADING_LAYER"] = "workqueue"
        if verbose:
            mode = ("LIVELOCKED (probe timed out after {:.0f} s)".format(timeout_s)
                    if verdict.endswith(":timeout") else "HARD-CRASHED the probe child")
            print("[dynameta.numba_env] the default numba threading layer {} on the "
                  "representative kernel; forcing NUMBA_THREADING_LAYER=workqueue{}".format(
                      mode, (" [" + why + "]") if why else ""), flush=True)
        return verdict

    try:
        with open(cache, "r") as fh:
            d = json.load(fh)
        if d.get("key") == key and (time.time() - d.get("t", 0.0)) < ttl_s:
            cached = str(d.get("verdict", ""))
            if cached.startswith("default-ok") or cached.startswith("workqueue-fallback"):
                return _apply(cached, str(d.get("why", "cached verdict")))
    except Exception:
        pass

    # child must resolve `import dynameta` exactly like this process
    env = dict(os.environ)
    pkg_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env["PYTHONPATH"] = pkg_root + os.pathsep + env.get("PYTHONPATH", "")
    try:
        r = subprocess.run([sys.executable, "-c", _PROBE_SRC], capture_output=True,
                           text=True, timeout=timeout_s, env=env)
        verdict, why = _classify(r.returncode, r.stdout or "", r.stderr or "")
    except subprocess.TimeoutExpired:
        verdict, why = "workqueue-fallback:timeout", "no output within {:.0f} s".format(timeout_s)
    except Exception as exc:                        # e.g. FileNotFoundError on sys.executable
        verdict, why = "probe-error:launch", "{}: {}".format(type(exc).__name__, exc)
    if not verdict.startswith("probe-error"):       # AUDIT T-11: cache CONCLUSIVE verdicts only
        try:
            with open(cache, "w") as fh:
                json.dump({"key": key, "t": time.time(), "verdict": verdict, "why": why}, fh)
        except Exception:
            pass
    return _apply(verdict, why)
