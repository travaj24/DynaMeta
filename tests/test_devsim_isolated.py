"""Gates for @pytest.mark.devsim_isolated (tests/conftest.py, v0.15 2026-11-06): a marked test runs in a
fresh interpreter, so no DEVSIM device left live by another test of the worker takes part in its solves.

WHY. A DC solve iterates every live DEVSIM device in one Newton (dc_solve.solve_dc docstring): a leftover
device both affects a later solve's convergence and is itself moved by it. A BITWISE comparison of two
separate solves is meaningful only with no unrelated device live. MEASURED (scipy OpenBLAS): with a
converged ITO gate (4e26) and a 1e22 slab left live, test_devsim_tunneling.py::test_w1's off / on
potentials differed by 1.1e-16 V on one run and agreed on the next; in a clean process they are
bitwise equal. Which run differs is a round-off accident (one leftover did not make two light-slab solves
differ in three tries), so this file gates the deterministic part: the leftover is MOVED by an unrelated
solve, and a marked test runs where no leftover exists.

FAILS ON THE OLD CODE (c6aa0a4: no marker, no hook): the marker is unknown (--strict-markers: collection
error), and with the marker merely registered the isolated test sees the leftover device and fails.
Runtime ~25 s (two child interpreters).
"""
import os

import numpy as np
import pytest

devsim = pytest.importorskip("devsim")

import _abs_tol_fixtures as F                                             # noqa: E402

LEFTOVER = "iso_leftover"


@pytest.fixture(scope="module")
def leftover():
    """A converged 4e26 ITO gate at 1 V kept live for the module, as a module-scoped fixture of another
    file is under xdist."""
    from dynameta.carriers import dc_solve as DC
    steps, _ = F.ito_gate(LEFTOVER)
    assert all(ok for _, ok, _ in F.run_steps(steps, abs_tol=DC.default_abs_tol, rel_tol=1e-5))
    yield LEFTOVER
    F.drop(LEFTOVER)


def _two_solves_bitwise():
    """Two SEPARATE solves of the same light slab (built twice under different names, each dropped after
    its solve): their arrays, which in a clean process are bitwise equal."""
    import _wellposed_fixtures as W
    from dynameta.carriers import dc_solve as DC
    out = []
    for tag in ("a", "b"):
        dev = "iso_slab_" + tag
        W.slab(dev, nd=1e22, contact="ohmic", hi_contact="ohmic")
        try:
            devsim.set_node_values(device=dev, region="bulk", name="Holes",
                                   values=list(3.0 * W._v(dev, "bulk", "Holes")))
            with F.quiet():
                DC.solve_dc(dev, rel_tol=1e-10, max_iter=60)
            out.append({v: W._v(dev, "bulk", v) for v in ("Potential", "Electrons", "Holes")})
        finally:
            W.drop(dev)
    return out


def test_leftover_device_is_live_and_is_moved_by_a_later_solve(leftover):
    """The premise, in-process: the leftover is live, and a solve of an UNRELATED device iterates it
    too (its node values change -- by round-off -- although nothing about it was touched)."""
    assert LEFTOVER in devsim.get_device_list()
    before = np.asarray(devsim.get_node_model_values(device=LEFTOVER, region="ito", name="Electrons"))
    _two_solves_bitwise()
    after = np.asarray(devsim.get_node_model_values(device=LEFTOVER, region="ito", name="Electrons"))
    assert not np.array_equal(before, after)                 # moved by a solve it took no part in
    assert np.max(np.abs(after / before - 1.0)) < 1e-12      # ... by round-off only


@pytest.mark.devsim_isolated("asserts it runs with no leftover device and in the child")
def test_isolated_test_runs_in_a_clean_child_process():
    """GATE: a marked test runs in the child (DYNAMETA_DEVSIM_ISOLATED_CHILD=1) with no device live,
    although `leftover` is live in the parent when this runs serially after the test above; and two
    separate solves there are bitwise equal."""
    assert os.environ.get("DYNAMETA_DEVSIM_ISOLATED_CHILD") == "1"
    assert LEFTOVER not in devsim.get_device_list()
    a, b = _two_solves_bitwise()
    assert all(np.array_equal(a[k], b[k]) for k in a)


def test_a_failing_isolated_test_is_reported_as_a_failure(tmp_path):
    """Refutation: the hook must not turn a child failure into a pass, and must leave pytest's setup
    stack consistent. Probe files run by a nested pytest: tests/_isolated_probe.py (a plain test using a
    module fixture, then one marked passing and one marked failing test) and tests/_isolated_probe2.py
    (a plain test in the NEXT module): '1 failed, 3 passed', no error, and the failure carries the
    child's output. (The first version of the hook skipped the teardown phase and the next module's
    setup errored with 'previous item was not torn down properly': measured under -n 3.)"""
    import subprocess
    import sys
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = {k: v for k, v in os.environ.items() if k != "DYNAMETA_DEVSIM_ISOLATED_CHILD"
           and not k.startswith("PYTEST_XDIST") and k != "PYTEST_CURRENT_TEST"}
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-p", "no:xdist", "-rf",
                        os.path.join("tests", "_isolated_probe.py"), os.path.join("tests", "_isolated_probe2.py")],
                       cwd=repo, env=env, capture_output=True, text=True, timeout=600)
    tail = r.stdout.strip().splitlines()[-1]
    assert r.returncode == 1 and "1 failed, 3 passed" in tail and "error" not in tail, r.stdout[-3000:]
    assert "devsim_isolated child" in r.stdout and "PROBE_FAILS_IN_CHILD" in r.stdout
