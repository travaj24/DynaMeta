"""Guard + light smoke tests for the contact-current extractor (driver D1). The quantitative
Ohm's-law / bipolar / off-switch oracles live in validation/contact_current_drivers.py (full
DEVSIM solves); here we cover the argument guards and the empty-device behaviour."""
import pytest

pytest.importorskip("devsim")

from dynameta.carriers.contact_current import extract_contact_currents, CONTINUITY_EQUATIONS


def test_depth_guard():
    with pytest.raises(ValueError):
        extract_contact_currents("any_device", depth_m=0.0)
    with pytest.raises(ValueError):
        extract_contact_currents("any_device", depth_m=-1e-6)


def test_equation_names_are_devsim_continuity_pair():
    assert CONTINUITY_EQUATIONS == ("ElectronContinuityEquation", "HoleContinuityEquation")


def test_unknown_device_raises():
    import devsim as ds
    with pytest.raises(Exception):
        extract_contact_currents("no_such_device_xyz", depth_m=1e-6)
