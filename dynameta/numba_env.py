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
optics.fdtd.solve_fdtd_1d numba backend (the smallest member of the failing class) in a
sacrificial child under a timeout.

Why the DLL exposure: the pip 'tbb' wheel drops tbb12.dll into <sys.prefix>/Library/bin -- a
conda-style path a python.org install never searches, so numba's tbbpool binding fails with
DLL-not-found and numba silently falls through to OpenMP. _expose_tbb_dlls() adds that
directory via os.add_dll_directory when present, which makes TBB the default layer again
(numba's preference order: tbb > omp > workqueue).

Cost control: (1) an explicit NUMBA_THREADING_LAYER always wins -- no probe; (2) POSIX skips
by default (the breakage class is Windows DLL/runtime rot; CI pays nothing -- pass
windows_only=False to probe anywhere); (3) no numba -> nothing to do; (4) the verdict is
CACHED per (python, numba, tbb-present) key with a TTL, so even the broken machine pays the
probe once a day and a healthy box pays one small-kernel JIT (~15-30 s) once a day.

No import-time side effects: call ensure_working_threading_layer() explicitly (tests/conftest
does) BEFORE the first parallel kernel launch of the process."""

from __future__ import annotations

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
    'no-numba', 'default-ok:<layer>' (probe passed; numba's own selection kept), or
    'workqueue-fallback' (the selected layer wedged on the representative kernel ->
    NUMBA_THREADING_LAYER=workqueue exported for this process and its children). Safe to call
    repeatedly; verdicts cached for ttl_s."""
    if os.environ.get("NUMBA_THREADING_LAYER"):
        _expose_tbb_dlls()                           # explicit tbb choice still needs the DLLs
        return "explicit"
    if windows_only and os.name != "nt":
        return "posix-skip"
    try:
        import numba
    except Exception:
        return "no-numba"
    have_tbb = _expose_tbb_dlls()

    cache = _cache_path()
    key = "numba-{}-tbb-{}".format(numba.__version__, int(have_tbb))

    def _apply(verdict: str) -> str:
        if verdict.startswith("default-ok"):
            return verdict
        os.environ["NUMBA_THREADING_LAYER"] = "workqueue"
        if verbose:
            print("[dynameta.numba_env] the default numba threading layer failed the "
                  "representative-kernel probe (livelock/crash); forcing "
                  "NUMBA_THREADING_LAYER=workqueue", flush=True)
        return "workqueue-fallback"

    try:
        with open(cache, "r") as fh:
            d = json.load(fh)
        if d.get("key") == key and (time.time() - d.get("t", 0.0)) < ttl_s:
            return _apply(d.get("verdict", "workqueue-fallback"))
    except Exception:
        pass

    # child must resolve `import dynameta` exactly like this process
    env = dict(os.environ)
    pkg_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env["PYTHONPATH"] = pkg_root + os.pathsep + env.get("PYTHONPATH", "")
    verdict = "workqueue-fallback"
    try:
        r = subprocess.run([sys.executable, "-c", _PROBE_SRC], capture_output=True,
                           text=True, timeout=timeout_s, env=env)
        if r.returncode == 0 and "LAYER_OK:" in r.stdout:
            layer = r.stdout.split("LAYER_OK:", 1)[1].strip()
            verdict = "default-ok:{}".format(layer)
    except subprocess.TimeoutExpired:
        pass                                        # livelock: the observed failure mode
    except Exception:
        pass
    try:
        with open(cache, "w") as fh:
            json.dump({"key": key, "t": time.time(), "verdict": verdict}, fh)
    except Exception:
        pass
    return _apply(verdict)
