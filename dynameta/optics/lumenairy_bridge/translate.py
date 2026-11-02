"""Bidirectional DynaMeta <-> Lumenairy translation (roadmap v0.5 A3).

FORWARD (DynaMeta -> Lumenairy) lives in rcwa_backend.design_to_rcwa_stack. This module adds
the REVERSE direction and the materials mapping -- the synergy layer that lets a
Lumenairy-born device gain DynaMeta's multiphysics axes (carriers, thermal, reliability,
effects) and lets DynaMeta materials drive Lumenairy's dispersive solves.

The SIGN conventions are identical on both sides (exp(-i omega t), Im(eps) > 0, metres); the
mapping traps this module handles are index-vs-permittivity -- Lumenairy REGION media
(n_superstrate/n_substrate) are refractive INDICES while every layer spec is a PERMITTIVITY --
and layer ORDER. See the package header for the full list of translated quantities (audit V-7).

Version pin (audit V-7 -- this paragraph described machinery `_common.py` had already retired):
the reverse translator reads RCWAStack's public attributes (period_x, period_y, is_1d,
n_superstrate, n_substrate) plus the per-layer record (thickness/.kind/.data/.dispersive)
through `_common.stack_layer_records`. Since lumenairy 5.22 that helper reads the PUBLIC
read-only `RCWAStack.layers` tuple -- there is no private-slot access and no version CEILING
left here; the single bridge-wide FLOOR is `_common.VERSION_FLOOR`, and the soft "newer than
verified" notice is `_common.VERSION_VERIFIED_MAX` (finding Q-16). A Lumenairy release that
changes the record shape still bumps the floor.

POLARIZATION VOCABULARY (audit V-8): this module speaks {'x', 'y', 'p'} -- the LAB AXIS of the
incident E ('y' = s-pol, 'p' = p-pol, 'x' = E along lab x, transverse only at normal incidence). It
is one of five spellings in the repo -- {'s','p'} is the PLANE-OF-INCIDENCE spelling (tmm_reference,
resonance, nonlocal_tmm, shg_fem's closed forms, the oblique 2-D FDTD), {'te','tm'} the lumenairy
grating bridge's, the integer `row` 0/1 the differentiable Berreman/RCWA/PMM forwards', and
`pol_axis` hydro_fem's 2-D in-plane axis. The map, the `normalize_pol` converter and the
normal-incidence / azimuth caveats live in `dynameta.core.polarization`. The set ACCEPTED here is
UNCHANGED; acceptance unification (b), the V-8 follow-on, widened only the two PLANE-OF-INCIDENCE
families ({'s','p'} and {'te','tm'}), whose aliases name the same physical mode in every geometry
they cover; this vocabulary's crossings depend on the azimuth (or have no image at all), so they
stay STRICT and are made explicitly, through normalize_pol.
"""

from __future__ import annotations

from typing import Callable, List, Optional

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


def _uniform_cell_value(cell):
    """The single value of a laterally-UNIFORM (Sx, Sy[, 3, 3]) cell, else None (exact compare:
    a cell that varies by one ulp is patterned -- no tolerance is a faithful inverse)."""
    c = np.asarray(cell)
    flat = c.reshape((c.shape[0] * c.shape[1],) + c.shape[2:])
    return flat[0].copy() if bool(np.all(flat == flat[:1])) else None


def rcwa_stack_to_design(stack, *, name: str = "lumenairy_import",
                         layer_names: Optional[List[str]] = None,
                         polarization: str = "y", return_eps_by_region: bool = False):
    """Translate a Lumenairy RCWAStack into a DynaMeta Design, so the SAME device runs
    through DynaMeta's carriers/thermal/reliability/effects axes (and back through the RCWA
    bridge for optics -- the round-trip is gated in validation/lumenairy_translate.py).

    SCOPE (v2, 2026-11-02 -- v1 translated UNIFORM layers only), per lumenairy layer kind:
      * 'uniform' (scalar or dispersive callable) -> a plain Layer (unchanged);
      * 'shapes' (analytic form factors) -> a Layer whose background is eps_background and whose
        Inclusions are the EXACT CrossSections (rectangle -> Rectangle, disk -> Circle, ellipse ->
        Ellipse; same absolute cell frame, _shapes contract). lumenairy guarantees the list is
        DISJOINT, so priority order is immaterial and DynaMeta's over-painting reproduces the
        additive form factors point for point. Dispersive shape / background eps -> CallableOptical.
        On a 1-D stack only full-period_y rectangles are faithful (lumenairy y-AVERAGES anything
        else there, a structure no Inclusion list describes) -- others are refused;
      * 'tensor' that is laterally UNIFORM (the add_layer(eps=(3,3)) entry, or a constant cell) ->
        a Layer + a uniform TENSOR EpsField in the returned eps_by_region (DynaMeta materials are
        scalar, so a tensor has no Material image). Needs return_eps_by_region=True, which changes
        the return to (design, eps_by_region); without it the layer is refused with that hint.
        The layer's background_material is a scalar PLACEHOLDER (the diagonal mean) named
        'mat_tensor_placeholder_<layer>' -- the EpsField is the source of truth;
      * 'iso' (eps_cell) that is laterally UNIFORM -> a plain Layer (a constant cell IS a uniform
        slab, exactly).
    STILL REFUSED (NotImplementedError, with the reason): a PATTERNED eps_cell / eps_tensor_cell
    -- a rasterized grid has no faithful inverse into DynaMeta's analytic Inclusion vocabulary
    (any shape fitted to pixels is a guess about sub-pixel geometry the grid never stored) -- and
    a dispersive (callable) tensor. Identical non-dispersive eps values share one material;
    dispersive ones get one material each (callables are not comparable).

    Lumenairy stacks are built SUPERSTRATE-side first; DynaMeta Stacks are bottom-to-top --
    the layer list is reversed here, and layer_names (when given) follows the LUMENAIRY
    (top-first) order to match how the stack was written.

    `polarization` is the OpticalSpec LAB-AXIS vocabulary {'x','y','p'} (audit V-8): it is handed
    straight to the OpticalSpec this builds, which validates it and names the sibling family on a
    wrong label. Note the asymmetry with the SAME package's rcwa_grating_RT, which speaks
    {'te','tm'} -- see dynameta.core.polarization."""
    from dynameta.optics.lumenairy_bridge._common import stack_layer_records
    layers_rec = stack_layer_records(stack)
    if layer_names is not None and len(layer_names) != len(layers_rec):
        raise ValueError("layer_names must match the stack's {} layers".format(
            len(layers_rec)))
    px = float(stack.period_x)
    py = float(stack.period_y)

    # ---- refusals FIRST (nothing is built for a stack that cannot be translated) ----
    plan = []                                             # (kind, payload) per record
    bad = []
    for i, rec in enumerate(layers_rec):
        disp = bool(rec.dispersive) or callable(rec.data)
        if rec.kind == "uniform":
            plan.append(("uniform", rec.data))
        elif rec.kind == "shapes":
            bg, shapes = rec.data
            if stack.is_1d:
                partial = [k for k, sh in enumerate(shapes)
                           if sh.get("shape") != "rectangle"
                           or float(sh["size"][1]) < py * (1.0 - 1e-12)]
                if partial:
                    bad.append((i, "1-D stack shapes {} are not full-period_y rectangles; "
                                   "lumenairy solves their y-AVERAGE, which no Inclusion "
                                   "list describes".format(partial)))
                    continue
            plan.append(("shapes", (bg, shapes)))
        elif rec.kind == "iso" and not disp and _uniform_cell_value(rec.data) is not None:
            plan.append(("uniform", complex(_uniform_cell_value(rec.data))))
        elif rec.kind == "tensor" and not disp and _uniform_cell_value(rec.data) is not None:
            if not return_eps_by_region:
                bad.append((i, "a uniform TENSOR layer has no scalar-Material image; pass "
                               "return_eps_by_region=True to receive it as a tensor EpsField "
                               "(the return becomes (design, eps_by_region))"))
                continue
            plan.append(("tensor", np.asarray(_uniform_cell_value(rec.data), dtype=complex)))
        elif rec.kind == "tensor" and disp:
            bad.append((i, "a dispersive (callable) tensor layer has no DynaMeta image (an "
                           "EpsField is concrete at one wavelength)"))
        else:
            bad.append((i, "a PATTERNED rasterized {} cell has no faithful inverse into "
                           "DynaMeta's analytic Inclusion shapes (a shape fitted to pixels "
                           "guesses sub-pixel geometry the grid never stored)".format(
                               "eps_tensor" if rec.kind == "tensor" else "eps")))
    if bad:
        raise NotImplementedError(
            "rcwa_stack_to_design: {} layer(s) cannot be translated faithfully -- {}".format(
                len(bad), "; ".join("layer {} ({}): {}".format(i, layers_rec[i].kind, why)
                                   for i, why in bad)))

    from dynameta.core.eps_field import EpsField
    from dynameta.geometry import Inclusion
    from dynameta.optics.lumenairy_bridge._shapes import shape_to_cross_section
    reg = MaterialRegistry()
    reg.add(Material("superstrate", _index_to_eps_model(stack.n_superstrate)))
    reg.add(Material("substrate", _index_to_eps_model(stack.n_substrate)))

    const_pool = {}                                       # eps value -> material name

    def _material(spec, disp_name):
        """A pooled ConstantOptical for a concrete eps; one CallableOptical per dispersive spec."""
        if callable(spec):
            reg.add(Material(disp_name, lumenairy_eps_to_optical_model(spec)))
            return disp_name
        key = complex(spec)
        if key not in const_pool:
            mname = "mat_eps_{}".format(len(const_pool))
            const_pool[key] = mname
            reg.add(Material(mname, ConstantOptical(key)))
        return const_pool[key]

    dm_layers: List[Layer] = []
    eps_by_region = {}
    for i, (rec, (kind, payload)) in enumerate(zip(layers_rec, plan)):   # lumenairy: top-first
        lname = layer_names[i] if layer_names else "layer_{}".format(i)
        if kind == "uniform":
            # (the v1 naming, unchanged: 'mat_' + layer for a dispersive layer)
            dm_layers.append(Layer(lname, float(rec.thickness),
                                   _material(payload, "mat_" + lname)))
        elif kind == "shapes":
            bg, shapes = payload
            incs = [Inclusion(shape=shape_to_cross_section(sh, px, py),
                              material=_material(sh["eps"], "mat_{}_shape{}".format(lname, k)))
                    for k, sh in enumerate(shapes)]
            dm_layers.append(Layer(lname, float(rec.thickness),
                                   _material(bg, "mat_{}_bg".format(lname)), inclusions=incs))
        else:                                             # uniform tensor
            mname = "mat_tensor_placeholder_{}".format(lname)
            reg.add(Material(mname, ConstantOptical(complex(np.trace(payload) / 3.0))))
            dm_layers.append(Layer(lname, float(rec.thickness), mname))
            eps_by_region[lname] = EpsField(tensor=payload.copy())

    dm_layers.reverse()                                   # DynaMeta wants bottom-to-top
    cell = (UnitCell.square(stack.period_x) if stack.is_1d
            or stack.period_y == stack.period_x
            else UnitCell(period_x_m=stack.period_x, period_y_m=stack.period_y))
    design = Design(name=name, unit_cell=cell,
                    stack=Stack(layers=dm_layers, superstrate_material="superstrate",
                                substrate_material="substrate"),
                    electrodes=[], materials=reg,
                    optical=OpticalSpec(polarization=polarization, incidence_angle_deg=0.0))
    return (design, eps_by_region) if return_eps_by_region else design
