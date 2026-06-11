"""Bidirectional DynaMeta <-> Lumenairy translation (roadmap v0.5 A3).

FORWARD (DynaMeta -> Lumenairy) lives in rcwa_backend.design_to_rcwa_stack. This module adds
the REVERSE direction and the materials mapping -- the synergy layer that lets a
Lumenairy-born device gain DynaMeta's multiphysics axes (carriers, thermal, reliability,
effects) and lets DynaMeta materials drive Lumenairy's dispersive solves.

Conventions are identical on both sides (exp(-i omega t), Im(eps) > 0, metres); the ONE
mapping trap is index-vs-permittivity: Lumenairy REGION media (n_superstrate/n_substrate)
are refractive INDICES while every layer spec is a PERMITTIVITY -- both handled here.

Version pin: the reverse translator reads RCWAStack's public attributes (period_x, period_y,
is_1d, n_superstrate, n_substrate) plus the slotted per-layer record
(_layers[i].thickness/.kind/.data/.dispersive -- the stable 5.14 surface). A Lumenairy
release changing that record bumps the bridge floor."""

from __future__ import annotations

from typing import Callable, List, Optional, Union

import numpy as np

from dynameta.geometry import Design, Layer, Stack, UnitCell
from dynameta.geometry.specs import OpticalSpec
from dynameta.materials import ConstantOptical, Material, MaterialRegistry

__all__ = ["CallableOptical", "optical_model_to_lumenairy_eps",
           "lumenairy_eps_to_optical_model", "rcwa_stack_to_design"]


class CallableOptical:
    """OpticalModel adapter around a wavelength -> PERMITTIVITY callable (the Lumenairy
    dispersive-spec shape). Satisfies DynaMeta's OpticalModel duck type:
    eps(lambda_m, n_m3=None) -> complex (density-independent)."""

    def __init__(self, eps_of_wl: Callable[[float], complex]):
        if not callable(eps_of_wl):
            raise TypeError("CallableOptical needs a wavelength -> eps callable")
        self._fn = eps_of_wl

    def eps(self, lambda_m: float, *, n_m3=None):
        return complex(self._fn(float(lambda_m)))


def optical_model_to_lumenairy_eps(model, *, n_m3: Optional[float] = None
                                   ) -> Callable[[float], complex]:
    """A DynaMeta OpticalModel (or Material) as a Lumenairy dispersive spec: a
    wavelength -> PERMITTIVITY callable accepted by every RCWAStack layer slot. Free-carrier
    models (DrudeOptical) need the density: pass n_m3 explicitly (raises otherwise, the same
    contract as model.eps itself). For the REGION-index slots (n_superstrate/n_substrate)
    take sqrt: lambda wl: np.sqrt(fn(wl)) -- Lumenairy region media are INDICES."""
    def _eps(wl: float) -> complex:
        return complex(model.eps(float(wl), n_m3=n_m3))
    return _eps


def lumenairy_eps_to_optical_model(spec) -> object:
    """A Lumenairy layer PERMITTIVITY spec (complex scalar or wl -> eps callable) as a
    DynaMeta OpticalModel (ConstantOptical / CallableOptical). NOTE: a lumenairy.Material
    instance IS a wl -> eps callable (its __call__ returns (n+ik)^2), so it passes through
    the callable branch unchanged -- but it raises outside its tabulated range (no
    extrapolation), which the resulting model inherits."""
    if callable(spec):
        return CallableOptical(spec)
    return ConstantOptical(complex(spec))


def _index_to_eps_model(n_spec) -> object:
    """A Lumenairy REGION medium (refractive INDEX, scalar or callable) as an eps-based
    OpticalModel (the index-vs-permittivity trap, handled once here)."""
    if callable(n_spec):
        return CallableOptical(lambda wl: complex(n_spec(float(wl))) ** 2)
    return ConstantOptical(complex(n_spec) ** 2)


def rcwa_stack_to_design(stack, *, name: str = "lumenairy_import",
                         layer_names: Optional[List[str]] = None,
                         polarization: str = "y") -> Design:
    """Translate a Lumenairy RCWAStack into a DynaMeta Design, so the SAME device runs
    through DynaMeta's carriers/thermal/reliability/effects axes (and back through the RCWA
    bridge for optics -- the round-trip is gated in validation/lumenairy_translate.py).

    v1 scope: UNIFORM (scalar or dispersive-callable) layers. Patterned layers ('iso',
    'tensor', 'shapes') raise NotImplementedError -- a rasterized eps_cell has no faithful
    inverse into DynaMeta's analytic Inclusion vocabulary; reconstructing shapes from grids
    is a documented follow-on. Identical non-dispersive layer eps values share one material;
    dispersive layers get one material each (callables are not comparable).

    Lumenairy stacks are built SUPERSTRATE-side first; DynaMeta Stacks are bottom-to-top --
    the layer list is reversed here, and layer_names (when given) follows the LUMENAIRY
    (top-first) order to match how the stack was written."""
    layers_rec = list(getattr(stack, "_layers"))
    bad = [i for i, L in enumerate(layers_rec) if L.kind != "uniform"]
    if bad:
        raise NotImplementedError(
            "rcwa_stack_to_design v1 translates UNIFORM layers only; layers {} have kinds "
            "{} (a rasterized cell has no faithful inverse into Inclusion shapes -- "
            "follow-on)".format(bad, [layers_rec[i].kind for i in bad]))
    if layer_names is not None and len(layer_names) != len(layers_rec):
        raise ValueError("layer_names must match the stack's {} layers".format(
            len(layers_rec)))

    reg = MaterialRegistry()
    reg.add(Material("superstrate", _index_to_eps_model(stack.n_superstrate)))
    reg.add(Material("substrate", _index_to_eps_model(stack.n_substrate)))

    const_pool = {}                                       # eps value -> material name
    dm_layers: List[Layer] = []
    for i, rec in enumerate(layers_rec):                  # lumenairy order: top-first
        lname = layer_names[i] if layer_names else "layer_{}".format(i)
        if rec.dispersive or callable(rec.data):
            mname = "mat_" + lname
            reg.add(Material(mname, lumenairy_eps_to_optical_model(rec.data)))
        else:
            key = complex(rec.data)
            if key not in const_pool:
                mname = "mat_eps_{}".format(len(const_pool))
                const_pool[key] = mname
                reg.add(Material(mname, ConstantOptical(key)))
            mname = const_pool[key]
        dm_layers.append(Layer(lname, float(rec.thickness), mname))

    dm_layers.reverse()                                   # DynaMeta wants bottom-to-top
    cell = (UnitCell.square(stack.period_x) if stack.is_1d
            or stack.period_y == stack.period_x
            else UnitCell(period_x_m=stack.period_x, period_y_m=stack.period_y))
    return Design(name=name, unit_cell=cell,
                  stack=Stack(layers=dm_layers, superstrate_material="superstrate",
                              substrate_material="substrate"),
                  electrodes=[], materials=reg,
                  optical=OpticalSpec(polarization=polarization, incidence_angle_deg=0.0))
