"""
Equation registry: record every region/contact/interface equation as it is
created, so a Gummel (decoupled) solve can DELETE equations to freeze a variable
and RE-ADD them afterwards. DEVSIM's ds.solve is coupled-Newton only and has no
equation-subset flag, so "freeze variable X" == "delete the equations whose
variable is X, solve, re-add them".

Keyed by device. Physics setup calls record_region/contact/interface_equation
instead of the raw ds.* so the specs are captured. Newton solves ignore the
registry entirely; only Gummel uses it.
"""

from __future__ import annotations

from typing import Dict, List

import devsim as ds

# device -> list of {scope, loc, name, kwargs}
_REG: Dict[str, List[dict]] = {}


def record_region_equation(device: str, region: str, **kwargs) -> None:
    ds.equation(device=device, region=region, **kwargs)
    _REG.setdefault(device, []).append(
        {"scope": "region", "loc": region, "name": kwargs["name"], "kwargs": kwargs})


def record_contact_equation(device: str, contact: str, **kwargs) -> None:
    ds.contact_equation(device=device, contact=contact, **kwargs)
    _REG.setdefault(device, []).append(
        {"scope": "contact", "loc": contact, "name": kwargs["name"], "kwargs": kwargs})


def record_interface_equation(device: str, interface: str, **kwargs) -> None:
    ds.interface_equation(device=device, interface=interface, **kwargs)
    _REG.setdefault(device, []).append(
        {"scope": "interface", "loc": interface, "name": kwargs["name"], "kwargs": kwargs})


def delete_by_name(device: str, eq_name: str) -> None:
    """Delete every recorded equation with this name (freezes that variable)."""
    for e in _REG.get(device, []):
        if e["name"] != eq_name:
            continue
        if e["scope"] == "region":
            ds.delete_equation(device=device, region=e["loc"], name=eq_name)
        elif e["scope"] == "contact":
            ds.delete_contact_equation(device=device, contact=e["loc"], name=eq_name)
        else:
            ds.delete_interface_equation(device=device, interface=e["loc"], name=eq_name)


def reapply_by_name(device: str, eq_name: str) -> None:
    """Re-create every recorded equation with this name (models persist, so this
    just re-binds the equation)."""
    for e in _REG.get(device, []):
        if e["name"] != eq_name:
            continue
        if e["scope"] == "region":
            ds.equation(device=device, region=e["loc"], **e["kwargs"])
        elif e["scope"] == "contact":
            ds.contact_equation(device=device, contact=e["loc"], **e["kwargs"])
        else:
            ds.interface_equation(device=device, interface=e["loc"], **e["kwargs"])


def equation_names(device: str) -> set:
    return {e["name"] for e in _REG.get(device, [])}


def clear(device: str) -> None:
    _REG.pop(device, None)
