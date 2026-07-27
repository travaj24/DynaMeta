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
