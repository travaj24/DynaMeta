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

# devsim resolves lazily (audit S1-2): registry bookkeeping is importable devsim-free


class _DevsimShim:
    _mod = None

    def __getattr__(self, name):
        if _DevsimShim._mod is None:
            import devsim as _devsim
            _DevsimShim._mod = _devsim
        return getattr(_DevsimShim._mod, name)


ds = _DevsimShim()

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


def forget(device: str, eq_name: str, *, loc: str = None) -> None:
    """Drop recorded equation entries matching `eq_name` (and `loc`, if given) from the registry
    WITHOUT touching the live DEVSIM equation. Use when REPOINTING a contact -- e.g. swapping a
    bias-Dirichlet gate for a circuit-driven one: the caller ds.delete_contact_equation's the stale
    equation, forget()s its record here, then records the replacement, so a later Gummel/staged
    delete_by_name -> reapply_by_name does NOT resurrect the deleted equation. (delete_by_name both
    deletes the live equation AND re-adds it on reapply; forget only removes the bookkeeping, leaving
    the live equation as the caller left it.)"""
    reg = _REG.get(device)
    if not reg:
        return
    _REG[device] = [e for e in reg
                    if not (e["name"] == eq_name and (loc is None or e["loc"] == loc))]


def edge_with_derivs(device: str, region: str, name: str, eq: str, wrt) -> None:
    """Create an edge model + its @n0/@n1 derivatives w.r.t. each variable in `wrt` (the
    Jacobian entries DEVSIM's coupled Newton consumes). Shared by the unipolar and bipolar
    drift-diffusion modules (was duplicated verbatim in both)."""
    ds.edge_model(device=device, region=region, name=name, equation=eq)
    for w in wrt:
        for nd in ("n0", "n1"):
            ds.edge_model(device=device, region=region,
                          name="{}:{}@{}".format(name, w, nd),
                          equation="simplify(diff({}, {}@{}))".format(eq, w, nd))


def node_with_derivs(device: str, region: str, name: str, eq: str, wrt) -> None:
    """Create a node model + its derivatives w.r.t. each variable in `wrt`."""
    ds.node_model(device=device, region=region, name=name, equation=eq)
    for w in wrt:
        ds.node_model(device=device, region=region, name="{}:{}".format(name, w),
                      equation="simplify(diff({}, {}))".format(eq, w))


def equation_names(device: str) -> set:
    return {e["name"] for e in _REG.get(device, [])}


def clear(device: str) -> None:
    _REG.pop(device, None)
