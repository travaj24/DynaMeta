"""Probe cases for tests/test_devsim_isolated.py::test_a_failing_isolated_test_is_reported_as_a_failure
(run only by that test, through a nested pytest; the leading underscore keeps it out of collection)."""
import os

import pytest


@pytest.fixture(scope="module")
def module_state():
    return {"set_up_in": os.getpid()}


def test_probe_plain_first(module_state):
    """An ordinary test with a module fixture BEFORE the marked ones: the marked tests must tear the
    session's setup stack down correctly for the next module (tests/_isolated_probe2.py)."""
    assert module_state["set_up_in"] == os.getpid()


@pytest.mark.devsim_isolated("probe")
def test_probe_passes_in_child():
    assert os.environ.get("DYNAMETA_DEVSIM_ISOLATED_CHILD") == "1"


@pytest.mark.devsim_isolated("probe")
def test_probe_fails_in_child():
    assert os.environ.get("DYNAMETA_DEVSIM_ISOLATED_CHILD") == "1"
    raise AssertionError("PROBE_FAILS_IN_CHILD")
