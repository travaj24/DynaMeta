"""
Geometry-capability guards: the ONE place that words the refusal when a backend is handed a Design
whose geometry it cannot build.

WHY (2026-11-02, workstream B2 of the lumenairy-5.50 bridge PR): `Stack.features` (vias, T-patch
stems) and `Inclusion.taper` (sidewall tapers) are VALIDATED data-model fields, so a Design carrying
them constructs cleanly -- and then every backend that lays down `stack.layers` only simply never
reads them. Measured on 810086b with a 300 nm cell, 120 nm + 80 nm lossless layers and one 100 x 100 nm
Cu via (z 0..200 nm) at 1.31 um, normal incidence: the TMM solver, design_layer_stack,
layered_stack_from_design, the RCWA / 1-D PMM / 2-D PMM / Berreman / EMT-screen bridges,
design_to_fdtd_layers and the 2-D FDTD solver ALL ran and returned R/T bit-identical (repr) to the
via-free design (R = 0.16891680139775, a dielectric slab ignoring a metal post), and the DEVSIM 2-D
region planner (LayeredDevsimBuilder._region_specs) and Stacked3DSpec.from_design returned the
via-free specs. Only LayeredOpticalBuilder refused (audit F-14). A validated Design that silently
loses a solid is worse than one that refuses: each such backend now calls
`require_supported_geometry` at its entry, and the message is single-sourced here.

API:
  require_supported_geometry(design_or_stack, backend, *, features=False, taper=False)
      raise UnsupportedGeometryError (a NotImplementedError) naming the backend and the offending
      features / tapered inclusions; features=True / taper=True declare that the backend DOES
      implement that field (taper=True: only the PMM2D bridge's PURE engine, which maps ONE tapered
      Rectangle per layer onto lumenairy's add_tapered_pillar staircase -- 2026-11-02, workstream B3;
      every other backend, and the hybrid engine, still refuses).
  require_untapered_layer(layer, backend)
      the same taper refusal for LAYER-level helpers (rasterizers, per-layer EMT) that never see
      the Design.
"""

from __future__ import annotations

from typing import List, Tuple


class UnsupportedGeometryError(NotImplementedError):
    """A backend was handed a Design geometry field (Stack.features, Inclusion.taper) it does not
    implement and would otherwise silently drop. Subclasses NotImplementedError, so existing
    `except NotImplementedError` / pytest.raises(NotImplementedError) sites keep working."""


def _stack_of(obj):
    return getattr(obj, "stack", obj)


def tapered_inclusions(design_or_stack) -> List[Tuple[str, int]]:
    """[(layer_name, inclusion_index)] of every inclusion with a non-None taper."""
    st = _stack_of(design_or_stack)
    return [(L.name, i) for L in (getattr(st, "layers", ()) or ())
            for i, inc in enumerate(getattr(L, "inclusions", ()) or ())
            if getattr(inc, "taper", None) is not None]


def _features_message(backend: str, feats) -> str:
    return ("{} does not implement Stack.features yet, and would build the layer stack WITHOUT the {} "
            "feature(s) ({}) -- a device different from the one the Design describes. Model the "
            "feature as a Layer inclusion (Stack.layers[].inclusions), or use a backend that builds "
            "feature solids. (Design validates Feature materials and z-spans, so this cannot be "
            "caught at construction.)".format(backend, len(feats), ", ".join(repr(f.name) for f in feats)))


def _taper_message(backend: str, where) -> str:
    return ("{} does not implement Inclusion.taper (sidewall tapers) yet, and would build the tapered "
            "inclusion(s) {} as STRAIGHT walls at their TOP section -- a device different from the one "
            "the Design describes. Set taper=None for straight walls, or approximate the taper as a "
            "z-staircase of untapered layers. (Inclusion validates the taper, so this cannot be caught "
            "at construction.)".format(backend, ", ".join("{}[{}]".format(n, i) for n, i in where)))


def require_supported_geometry(design_or_stack, backend: str, *, features: bool = False,
                               taper: bool = False) -> None:
    """Raise UnsupportedGeometryError if `design_or_stack` carries a geometry field `backend` does
    not implement: Stack.features unless features=True, a tapered Inclusion unless taper=True.
    A plain (feature-free, untapered) design passes untouched -- the guard reads, never mutates."""
    st = _stack_of(design_or_stack)
    if not features:
        feats = list(getattr(st, "features", ()) or ())
        if feats:
            raise UnsupportedGeometryError(_features_message(backend, feats))
    if not taper:
        where = tapered_inclusions(st)
        if where:
            raise UnsupportedGeometryError(_taper_message(backend, where))


def require_untapered_layer(layer, backend: str) -> None:
    """Layer-level twin of the taper refusal, for helpers that never see the Design."""
    where = [(getattr(layer, "name", "?"), i) for i, inc in enumerate(getattr(layer, "inclusions", ()) or ())
             if getattr(inc, "taper", None) is not None]
    if where:
        raise UnsupportedGeometryError(_taper_message(backend, where))
