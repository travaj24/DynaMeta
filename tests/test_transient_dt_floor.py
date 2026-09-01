"""The dt floor must sit where corruption actually begins -- i.e. it must scale with dt0.

Driven to dt = 1.78e-19 s by a halving cascade, DEVSIM returned converged=True together with a
terminal current of -35.8 A/m^2 where the true value was 1.8e-4 A/m^2 (nightly triage, 2026-08-31).
The floor meant to stop that was an absolute 1.0e-19 s, so 1.78e-19 sailed straight over it and the
garbage was integrated as an ordinary sample. No absolute constant can be right here: the dt at
which dQ/dt degenerates into round-off is a property of the device, so the floor is now derived from
dt0, the caller's own declared timescale.

The solver is stubbed rather than driven through DEVSIM so the pathological step is reached
deterministically on every runner, and so the "converged" report can be forced to be a LIE -- which
is the whole point, since a reported convergence is not evidence that a step is physical.
"""

import sys
import types

import pytest


class _StubError(Exception):
    """Stands in for ds.error; deliberately NOT RuntimeError, so the guard's own RuntimeError can
    never be swallowed by the module's `except ds.error` clause."""


# carriers.transient imports devsim at module scope, but NOTHING here needs the real solver -- every
# gate replaces `tr.ds` with the stub below. Rather than strand these gates on the one CI leg that
# ships devsim (they are control-flow gates, and the control flow is identical on every leg), stand
# a shim in for the import and withdraw it immediately afterwards. Withdrawing matters: leaving a
# fake `devsim` in sys.modules would silently defeat `importorskip("devsim")` in any module imported
# after this one, turning correct skips into failures somewhere else entirely.
try:
    import devsim as _real_devsim                                          # noqa: F401
    _shim = None
except ImportError:
    _shim = types.ModuleType("devsim")
    _shim.error = _StubError
    sys.modules["devsim"] = _shim
try:
    from dynameta.carriers import transient as tr
finally:
    if _shim is not None and sys.modules.get("devsim") is _shim:
        del sys.modules["devsim"]


# class (1): the DEVICE equations are far from converged, so transient_step halves and retries
_UNCONVERGED = {"converged": False, "iterations": [{"devices": [{"regions": [{"equations": [
    {"relative_error": 1.0e3, "absolute_error": 1.0e30}]}]}]}]}


class _StubDS:
    """Minimal devsim stand-in. Every transient_bdf1 solve fails as class (1) -- forcing the
    halving cascade -- except one attempted below `lie_below`, which reports success and hands back
    the garbage current, exactly as DEVSIM did at 1.78e-19 s."""

    error = _StubError

    def __init__(self, lie_below=0.0, garbage_current=-35.8):
        self.lie_below, self.garbage_current = float(lie_below), float(garbage_current)
        self.attempted, self.accepted_garbage = [], []

    def solve(self, **kw):
        if kw.get("type") != "transient_bdf1":
            return None                                  # the transient_dc initial condition
        dt = float(kw["tdelta"])
        self.attempted.append(dt)
        if dt < self.lie_below:
            return {"converged": True}                   # the lie
        return _UNCONVERGED

    def circuit_alter(self, **kw):
        return None

    def get_circuit_node_value(self, **kw):
        if self.attempted and self.attempted[-1] < self.lie_below:
            self.accepted_garbage.append(self.attempted[-1])       # a garbage sample was recorded
        return self.garbage_current


@pytest.fixture
def stub(monkeypatch):
    def _install(**kw):
        s = _StubDS(**kw)
        monkeypatch.setattr(tr, "ds", s)
        return s
    return _install


def test_an_explicit_floor_above_dt0_is_refused_before_any_solve(stub):
    """A floor above dt0 means the very first step is already unphysical -- an argument error,
    caught at the door like its siblings, before the device is touched at all."""
    s = stub()
    with pytest.raises(ValueError, match=r"dt_floor_s"):
        tr.transient_step(0.5, t_end=1.0e-9, dt0=1.0e-20, dt_floor_s=1.0e-18)
    assert s.attempted == [], "a sub-floor dt0 reached the solver instead of being refused"


def test_a_halving_cascade_raises_at_the_floor_instead_of_integrating(stub):
    """Class (1) forever -> dt halves without bound. The run must die at the derived floor."""
    s = stub()
    with pytest.raises(RuntimeError, match=r"dt_floor_s"):
        tr.transient_step(0.5, t_end=1.0e-9, dt0=1.0e-14)
    assert s.attempted, "the cascade never ran"
    assert min(s.attempted) >= 1.0e-18, (            # dt0 * dt_floor_frac = 1e-14 * 1e-4
        "a step BELOW the derived floor was handed to the solver: {:.3e}".format(min(s.attempted)))


def test_the_measured_1p78e_19_garbage_step_is_refused_not_recorded(stub):
    """THE REGRESSION, with the measured numbers. dt0=1e-14 is the diode's default, and the solver
    starts reporting converged=True (with I = -35.8 A/m^2 against a true 1.8e-4) once the cascade
    drops under 2e-19 -- which is where it really did. The OLD absolute floor of 1e-19 sat below
    that and let it through; the derived floor (1e-14 * 1e-4 = 1e-18) stops the run first, so the
    garbage is never sampled."""
    s = stub(lie_below=2.0e-19, garbage_current=-35.8)
    with pytest.raises(RuntimeError, match=r"dt_floor_s"):
        tr.transient_step(0.5, t_end=1.0e-9, dt0=1.0e-14)
    assert s.accepted_garbage == [], (
        "garbage current accepted at dt={:.3e} s -- the silent-corruption path is still open"
        .format(s.accepted_garbage[0] if s.accepted_garbage else float("nan")))


def test_the_old_absolute_floor_would_still_let_the_garbage_through(stub):
    """Proof that the gate above can FAIL, and that the fix is the thing that closes it: restoring
    the old constant via the explicit override reproduces the original corruption exactly -- the
    cascade walks straight past 1.78e-19 and the -35.8 sample is recorded.

    The run is then killed by a DIFFERENT guard (the anti-silent-failure max_steps check, since
    sub-floor steps advance t by almost nothing), which is why the assertion is on what the solver
    was asked to record rather than on the return value. The distinction matters: the old floor did
    not prevent the corruption, it merely failed to notice it."""
    s = stub(lie_below=2.0e-19, garbage_current=-35.8)
    with pytest.raises(RuntimeError, match=r"INCOMPLETE"):
        tr.transient_step(0.5, t_end=1.0e-9, dt0=1.0e-14, dt_floor_s=1.0e-19, max_steps=50)
    assert s.accepted_garbage, "the old floor was expected to admit the garbage step"
    assert min(s.accepted_garbage) < 2.0e-19, "the admitted step should be the sub-2e-19 one"


def test_the_floor_is_tunable_and_validated(stub):
    """A raised floor must bite earlier (the guard reads the parameter, not a hardcoded constant),
    and nonsensical floors are rejected."""
    s = stub()
    with pytest.raises(RuntimeError, match=r"dt_floor_s"):
        tr.transient_step(0.5, t_end=1.0e-9, dt0=1.0e-14, dt_floor_s=1.0e-16)
    assert min(s.attempted) >= 1.0e-16, "dt_floor_s was ignored in favour of the derived value"
    with pytest.raises(ValueError, match=r"dt_floor_s"):
        tr.transient_step(0.5, t_end=1.0e-9, dt0=1.0e-14, dt_floor_s=0.0)
    with pytest.raises(ValueError, match=r"dt_floor_frac"):
        tr.transient_step(0.5, t_end=1.0e-9, dt0=1.0e-14, dt_floor_frac=1.0)


def test_the_derived_floor_tracks_dt0(stub):
    """The point of deriving the floor: a caller who declares a slower timescale gets a
    proportionally higher floor, with no absolute constant involved."""
    for dt0, want in ((1.0e-14, 1.0e-18), (1.0e-11, 1.0e-15)):
        s = stub()
        with pytest.raises(RuntimeError, match=r"dt_floor_s"):
            tr.transient_step(0.5, t_end=1.0e-6, dt0=dt0)
        assert min(s.attempted) >= want, (
            "dt0={:.0e} should floor at {:.0e}, but {:.3e} was attempted"
            .format(dt0, want, min(s.attempted)))
