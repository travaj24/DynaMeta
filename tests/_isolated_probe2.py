"""Second probe module for tests/test_devsim_isolated.py (run after tests/_isolated_probe.py, whose last
tests are devsim_isolated): its setup fails with "previous item was not torn down properly" if the
marker's protocol skipped pytest's teardown phase."""


def test_probe_after_the_isolated_module():
    assert True
