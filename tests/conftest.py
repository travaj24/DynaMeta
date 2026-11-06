"""Make `import dynameta` work when running pytest from anywhere, and select a WORKING numba
threading layer before any parallel kernel launches (dynameta.numba_env: prefer TBB, probe it
in a sacrificial subprocess, fall back to workqueue when the TBB runtime is livelocked or
hard-crashed by it -- those are the two failure modes that do not raise; a probe that merely
FAILS is reported as 'probe-error:*' and changes nothing, audit T-11).

Also completes the audit-T-13 warnings policy: pyproject's `filterwarnings = ["error", ...]`
makes every warning a test failure, and this file adds the one exemption that cannot be spelled
there (see pytest_configure)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dynameta.numba_env import ensure_working_threading_layer   # noqa: E402

ensure_working_threading_layer()

# D0 (2026-11-03): pin DEVSIM's BLAS before ANY test module's `importorskip("devsim")` / bare
# `import devsim` can load it through a PATH search (dynameta.carriers.devsim_env docstring). It does
# not import devsim; it exports DEVSIM_MATH_LIBS, so the fresh-process children the suite spawns
# inherit the same library. A user-set DEVSIM_MATH_LIBS is left alone.
from dynameta.carriers.devsim_env import configure_math_libs   # noqa: E402

configure_math_libs()


def pytest_configure(config):
    """audit T-13: exempt the FEM solver's ADVISORY warning category from the `error` policy.

    `FEMDiagnosticWarning` (optics/solver.py:116) exists precisely so a caller can silence the
    regime advisories and fit/energy-closure quality flags without silencing NumPy/NGSolve --
    its own docstring documents `warnings.filterwarnings('ignore', category=FEMDiagnosticWarning)`
    as the supported recipe. It cannot go in pyproject's `filterwarnings` list, because pytest
    imports the named category at startup and `dynameta.optics.solver` imports ngsolve at module
    scope: the four CI legs without the [solvers] extra would fail to START. Registering it here,
    behind a try/except, keeps those legs working and degrades to "no exemption" when the FEM
    stack is absent (in which case nothing can raise the category anyway).
    """
    try:
        from dynameta.optics.solver import FEMDiagnosticWarning   # noqa: F401
    except Exception:
        return
    config.addinivalue_line("filterwarnings",
                            "ignore::dynameta.optics.solver.FEMDiagnosticWarning")


# v0.15 (plan 4.4.2, decision 55): every RAW devsim.solve(type="dc") a test makes runs the DC well-
# posedness check first (dynameta.carriers.well_posedness.ensure_dc_well_posed, cached per equation-set
# fingerprint), so a fixture that builds a FLOATING BODY -- the R26 slab with a Potential-only contact,
# whose singular Jacobian let round-off and the BLAS build pick the answer until 2026-11-03 -- fails
# with IllPosedDCError instead of passing by luck. dc_solve.solve_dc checks on its own; this covers the
# fixtures that call devsim.solve / extended_precision.solve directly. Read-only: it never creates a
# device or circuit (the process-global DEVSIM state, tests/_bipolar_calllog.py). Transient / ssac /
# noise solves are not checked. Opt out per test with @pytest.mark.floating_ok("reason") (the device is
# deliberately ill-posed, e.g. the gate that shows the old R26 fixture fails). The wrapper is installed
# lazily, once devsim has been imported by a test module, so the solver-free legs never import it.
_WELLPOSED = {"opt_out": False}


def _install_dc_well_posedness_hook():
    mod = sys.modules.get("devsim")
    if mod is None or getattr(mod.solve, "_dynameta_wellposed", False):
        return
    from dynameta.carriers import well_posedness as _WP
    orig = mod.solve

    def solve(*args, **kwargs):
        if kwargs.get("type") == "dc" and not _WELLPOSED["opt_out"]:
            _WP.ensure_dc_well_posed()
        return orig(*args, **kwargs)

    solve._dynameta_wellposed = True
    solve.__wrapped__ = orig
    solve.__doc__ = orig.__doc__
    mod.solve = solve


def pytest_runtest_setup(item):
    _install_dc_well_posedness_hook()
    _WELLPOSED["opt_out"] = item.get_closest_marker("floating_ok") is not None


def pytest_runtest_teardown(item, nextitem):
    _WELLPOSED["opt_out"] = False


def pytest_sessionfinish(session, exitstatus):
    """With DYNAMETA_WELLPOSED_STATS=<path>, append this process's well-posedness hook counters (calls,
    seconds in the static / numeric passes, assemblies, raises, refuted candidates) as one JSON line --
    the CI-time measurement of the audit note (each xdist worker appends its own line)."""
    path = os.environ.get("DYNAMETA_WELLPOSED_STATS")
    wp = sys.modules.get("dynameta.carriers.well_posedness")
    if not path or wp is None:
        return
    import json
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(dict(wp.STATS, pid=os.getpid())) + "\n")
