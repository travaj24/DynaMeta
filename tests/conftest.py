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


# v0.15 (2026-11-06): @pytest.mark.devsim_isolated("reason") runs ONE test in a fresh interpreter (its
# own `python -m pytest <nodeid>`), so it sees no DEVSIM device left live by another test of the worker.
# WHY: a DC solve iterates EVERY live DEVSIM device in one Newton (dc_solve.solve_dc docstring), so a
# device that a module-scoped fixture keeps alive both affects the convergence of a later solve and is
# itself moved by it (by ulps). A test that compares two SEPARATE solves BITWISE is therefore only
# meaningful in a process with no unrelated live device: measured, test_devsim_tunneling.py::test_w1's
# off/on potentials differed by 1.1e-16 V with two converged leftovers live, and agreed bitwise alone.
# Under xdist which devices are live is a scheduling accident. The child runs the test unchanged (its
# fixtures are rebuilt there, so it costs one interpreter + those fixtures); its outcome and output are
# reported here. Inside the child (DYNAMETA_DEVSIM_ISOLATED_CHILD=1) the marker is a no-op.
_ISOLATED_ENV = "DYNAMETA_DEVSIM_ISOLATED_CHILD"


def pytest_runtest_protocol(item, nextitem):
    if item.get_closest_marker("devsim_isolated") is None or os.environ.get(_ISOLATED_ENV):
        return None
    import subprocess
    import time
    from _pytest.reports import TestReport
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # an xdist worker's PYTEST_XDIST_* variables would make the child believe it is a worker too
    env = {k: v for k, v in os.environ.items()
           if not k.startswith("PYTEST_XDIST") and k != "PYTEST_CURRENT_TEST"}
    env[_ISOLATED_ENV] = "1"
    env["PYTHONPATH"] = os.pathsep.join([repo] + [p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p])
    item.ihook.pytest_runtest_logstart(nodeid=item.nodeid, location=item.location)
    t0 = time.time()
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-p", "no:xdist", "-rfEs",
                        item.nodeid], cwd=repo, env=env, capture_output=True, text=True, timeout=3600)
    t1 = time.time()
    tail = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else ""
    status = ("passed" if r.returncode == 0 and " passed" in tail else
              "skipped" if r.returncode == 0 and " skipped" in tail else "failed")
    longrepr = None
    if status == "failed":
        longrepr = "devsim_isolated child (exit {}):\n{}\n{}".format(r.returncode, r.stdout[-6000:],
                                                                    r.stderr[-3000:])
    elif status == "skipped":
        longrepr = (str(item.location[0]), int(item.location[1] or 0), "devsim_isolated child: " + tail)
    # The session's setup stack still holds what the PREVIOUS item left for this one (its module
    # fixtures, say): tear it down to what `nextitem` needs, as pytest's own teardown phase would --
    # skipping this leaves the stack inconsistent ("previous item was not torn down properly" at the
    # next module's setup, measured). Not the teardown HOOKS: this item never ran setup, and plugins'
    # teardown hooks read stash keys their setup hooks write.
    teardown_err = None
    try:
        item.session._setupstate.teardown_exact(nextitem)
    except Exception as e:                                    # noqa: BLE001 (reported, not swallowed)
        teardown_err = "devsim_isolated: teardown of the parent's fixtures failed: {!r}".format(e)
    for when in ("setup", "call", "teardown"):
        outcome = status if when == "call" else ("failed" if when == "teardown" and teardown_err else "passed")
        lr = longrepr if when == "call" else (teardown_err if when == "teardown" else None)
        rep = TestReport(item.nodeid, item.location, {k: 1 for k in item.keywords}, outcome, lr, when,
                         sections=[("devsim_isolated child stdout", r.stdout[-3000:])] if when == "call" else (),
                         duration=(t1 - t0) if when == "call" else 0.0, start=t0, stop=t1)
        item.ihook.pytest_runtest_logreport(report=rep)
    item.ihook.pytest_runtest_logfinish(nodeid=item.nodeid, location=item.location)
    return True
