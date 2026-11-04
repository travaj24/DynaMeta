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
            # D0: through the ONE devsim import home (BLAS pinned before the first import)
            from dynameta.carriers.devsim_env import import_devsim
            _DevsimShim._mod = import_devsim()
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


def amend_region_equation(device: str, region: str, eq_name: str, **extra_kwargs) -> None:
    """ADD terms to an already-recorded REGION equation (D1d, 2026-11-03): merge `extra_kwargs`
    (e.g. edge_volume_model / volume_node0_model / volume_node1_model) into its recorded kwargs, then
    delete + re-create the live DEVSIM equation from the merged record, so a later Gummel/staged
    delete_by_name -> reapply_by_name re-creates it WITH the new terms. Refuses to silently REPLACE a
    term the record already carries with a different model (two physics modules wiring the same
    equation slot would otherwise drop one of them without a trace)."""
    hits = [e for e in _REG.get(device, [])
            if e["name"] == eq_name and e["scope"] == "region" and e["loc"] == region]
    if len(hits) != 1:
        raise ValueError("amend_region_equation: expected one recorded {!r} on region {!r}, found {}"
                         .format(eq_name, region, len(hits)))
    kw = hits[0]["kwargs"]
    clash = {k: (kw[k], v) for k, v in extra_kwargs.items() if k in kw and kw[k] != v}
    if clash:
        raise ValueError("amend_region_equation: {!r} on {!r} already carries {} -- refusing to "
                         "overwrite (compose the terms into one model instead)".format(
                             eq_name, region, clash))
    merged = dict(kw, **extra_kwargs)
    ds.delete_equation(device=device, region=region, name=eq_name)
    try:
        ds.equation(device=device, region=region, **merged)
    except Exception:
        ds.equation(device=device, region=region, **kw)      # put the legacy equation back, then raise
        raise
    kw.update(extra_kwargs)                                  # record only what DEVSIM accepted


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


def recorded(device: str, eq_name: str, *, scope: str = None, loc: str = None) -> List[dict]:
    """COPIES of the recorded entries for `eq_name` (optionally filtered by scope in {'region',
    'contact', 'interface'} and by loc). Read-only view for callers that must re-point an existing
    equation (e.g. physics_density_gradient.setup_contact_dg, audit S1-5) without reaching into _REG."""
    return [{"scope": e["scope"], "loc": e["loc"], "name": e["name"], "kwargs": dict(e["kwargs"])}
            for e in _REG.get(device, [])
            if e["name"] == eq_name and (scope is None or e["scope"] == scope)
            and (loc is None or e["loc"] == loc)]


def forget(device: str, eq_name: str, *, loc: str = None, scope: str = None) -> None:
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
    # `scope` (optional, wave-7 2026-11-02) disambiguates a contact and a region that share a name
    _REG[device] = [e for e in reg
                    if not (e["name"] == eq_name and (loc is None or e["loc"] == loc)
                            and (scope is None or e["scope"] == scope))]


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


def element_with_derivs(device: str, region: str, name: str, eq: str, wrt) -> None:
    """Create an element (element-edge) model + its derivatives w.r.t. each NODE-SUFFIXED variable in
    `wrt` (e.g. "Potential@en0" .. "Holes@en3"). DEVSIM's element assembler demands the derivative
    w.r.t. EVERY element node of EVERY solution variable of the equation (a missing one is FATAL:
    "Model: ...:Electrons@en1 MISSING"), so callers pass the full list; most simplify to 0."""
    ds.element_model(device=device, region=region, name=name, equation=eq)
    for v in wrt:
        ds.element_model(device=device, region=region, name="{}:{}".format(name, v),
                         equation="simplify(diff({}, {}))".format(eq, v))


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
