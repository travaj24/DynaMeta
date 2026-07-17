"""Reliability glue: adapters from simulation RESULTS to reliability-model INPUTS.

The reliability models (dynameta.reliability) are validated post-processors that take plain
physical quantities (J [A/m^2], E_ox [V/m], absorbed fraction, T [K]). The simulators produce
richer objects (CarrierField, OpticalResult, ElectroThermalResult). This module is the missing
seam between them -- every adapter here consumes a result object and returns exactly what one
reliability model wants, with the unit conventions pinned in the docstrings.

Pure numpy: importable without DEVSIM/NGSolve. All adapters are total functions of their
inputs (no hidden state) so they compose into sweep post-processing loops over List[SweepRow]
or List[CarrierField].
"""

from __future__ import annotations

import re
from typing import Callable, Dict, List, Optional

import numpy as np

from dynameta.reliability.em import EmParams, current_density_A_m2
from dynameta.reliability.lidt import ThermalNode, cw_critical_intensity_W_m2, stack_absorbed_of_T
from dynameta.reliability.tddb import TddbParams, oxide_stress_from_electrothermal

__all__ = [
    "contact_current_A", "contact_current_density_from_field", "em_mttf_from_carrier_field",
    "absorbed_fraction", "tmm_absorption_by_layer_name",
    "oxide_stress_from_electrothermal", "tddb_tbd_from_electrothermal",
    "cw_damage_threshold_from_stack",
]


# ---- D1 contact current -> REL3 electromigration --------------------------------------------

def contact_current_A(carrier_field, contact_name: str) -> Optional[float]:
    """Terminal current [A] flowing into `contact_name`, from CarrierField.extras
    ['contact_currents_A'] (threaded by the DEVSIM builders; already scaled to amperes on both
    the 2D-layered and 3D paths). Returns None when the key is absent -- an equilibrium /
    Poisson-only solve carries no terminal current, and extraction failures are warn-only
    upstream -- so callers must treat None as 'no current available', not zero."""
    cc = getattr(carrier_field, "extras", {}).get("contact_currents_A")
    if cc is None:
        return None
    if contact_name not in cc:
        raise KeyError("contact {!r} not in contact_currents_A (have {})".format(
            contact_name, sorted(cc)))
    return float(cc[contact_name])


def contact_current_density_from_field(carrier_field, contact_name: str, *,
                                       width_m: float, thickness_m: float) -> Optional[float]:
    """|J| [A/m^2] through a trace of cross-section width_m x thickness_m carrying the terminal
    current of `contact_name`. Uses ONE contact's magnitude (the sum over both contacts of a
    two-terminal device is ~0 by conservation). None when no current is available (see
    contact_current_A)."""
    I = contact_current_A(carrier_field, contact_name)
    if I is None:
        return None
    return float(current_density_A_m2(abs(I), width_m, thickness_m))


def em_mttf_from_carrier_field(carrier_field, contact_name: str, *, width_m: float,
                               thickness_m: float, T_K: float, params: EmParams,
                               length_m: Optional[float] = None) -> Optional[float]:
    """Black-equation MTTF [s] for the interconnect fed by `contact_name` at temperature T_K.
    Anchor `params` with EmParams.calibrated() at a qualification point first -- the A_s
    prefactor is geometry/process-scaled. Folds the Blech immortality check when length_m is
    given (inf = immortal). None when no terminal current is available."""
    J = contact_current_density_from_field(carrier_field, contact_name,
                                           width_m=width_m, thickness_m=thickness_m)
    if J is None:
        return None
    return float(params.mttf_s(J, T_K, length_m=length_m))


# ---- D2 per-region absorption -> REL5 LIDT / thermal loads ----------------------------------

_FEM_DECOR = re.compile(r"(__incl\d*|__bg\d*|_skin|_bulk|_inpatch|_outside\w*)$")


def collapse_fem_region_keys(pra: dict) -> dict:
    """Collapse the FEM mesh-subdomain decorations (L_skin/L_bulk, L_inpatch/L_outside*,
    L__incl<j>/L__bg<k>) back onto plain design-layer names by summing (audit S5-6: the FEM
    backend keys per_region_absorption by mesh labels while every layered/Fourier backend uses
    design-layer names, so a backend-agnostic query broke across the seam). Idempotent for maps
    that are already design-layer-keyed."""
    out = {}
    for k, v in pra.items():
        base = _FEM_DECOR.sub("", k)
        while _FEM_DECOR.search(base):
            base = _FEM_DECOR.sub("", base)
        out[base] = out.get(base, 0.0) + float(v)
    return out


def absorbed_fraction(optical_result, region: Optional[str] = None) -> float:
    """Absorbed FRACTION of incident power (never watts) from an OpticalResult.

    region=None -> total absorption: A_independent when available (the loss-integral path,
    exactly the sum of the per-region map), else A = 1 - R - T.
    region=<name> -> that region's fraction from per_region_absorption. Backend-agnostic: FEM
    mesh-decorated keys (L_skin/L_bulk, L__incl<j>/L__bg<k>, L_inpatch/L_outside*) are collapsed
    onto the design-layer name and summed when the raw key is absent (audit S5-6), so the same
    design-layer query works across FEM and the layered/Fourier backends (TMM 'slab_<i>' keys:
    see tmm_absorption_by_layer_name). Raises if the map or key is missing rather than silently
    returning 0 -- a missing region name is layer-name drift, not zero absorption."""
    if region is None:
        if optical_result.A_independent is not None:
            return float(optical_result.A_independent)
        if optical_result.A is not None:
            return float(optical_result.A)
        raise ValueError("OpticalResult carries neither A_independent nor A")
    pra = optical_result.per_region_absorption
    if pra is None:
        raise ValueError("OpticalResult.per_region_absorption was not computed (FEM fills it "
                         "when A_independent is available; TMM always fills it)")
    if region in pra:
        return float(pra[region])
    collapsed = collapse_fem_region_keys(pra)
    if region in collapsed:
        return float(collapsed[region])
    raise KeyError("region {!r} not in per_region_absorption, raw ({}) or collapsed ({})".format(
        region, sorted(pra), sorted(collapsed)))


def tmm_absorption_by_layer_name(optical_result, design) -> Dict[str, float]:
    """Re-key a TMM per_region_absorption ('slab_<i>', indexed TOP-FIRST from the superstrate
    side) by the design's LAYER names (L.name; design.stack.layers is ordered bottom -> top,
    so the slab index walks the reversed list). Layer names are unique by Stack contract, so
    the result is a clean bijection -- this matches the layer-name addressing of the FEM
    region labels and of oxide_stress_from_electrothermal, NOT the material names (several
    layers may share one material). Raises on a non-TMM (already name-keyed FEM) map and on
    a slab/layer count mismatch (e.g. a graded n_slices stack, which has no 1:1 layer map)."""
    pra = optical_result.per_region_absorption
    if pra is None:
        raise ValueError("per_region_absorption was not computed")
    if not all(k.startswith("slab_") for k in pra):
        raise ValueError("not a TMM map (keys {}); FEM maps are already keyed by the "
                         "layer-derived region labels".format(sorted(pra)))
    names_top_first = [L.name for L in reversed(design.stack.layers)]
    if len(names_top_first) != len(pra):
        raise ValueError("design has {} layers but the TMM map has {} slabs (graded/sliced "
                         "stacks have no 1:1 layer mapping)".format(
                             len(names_top_first), len(pra)))
    return {name: float(pra["slab_{}".format(i)]) for i, name in enumerate(names_top_first)}


def cw_damage_threshold_from_stack(build_stack_at_T: Callable[[float], object],
                                   lambda_m: float, node: ThermalNode, *,
                                   T_max_K: float = 2000.0) -> float:
    """Critical CW irradiance [W/m^2] (thermal-runaway bifurcation) of a temperature-dependent
    stack: build_stack_at_T(T_K) -> LayeredStack (e.g. ITO Drude damping at Gamma(T)). This is
    the composed REL5 workflow -- absorbed(T) from the stack, then the runaway bisection. The
    temperature FEEDBACK is the point; passing a fixed-T absorption removes the bifurcation."""
    absorbed = stack_absorbed_of_T(build_stack_at_T, lambda_m)
    return float(cw_critical_intensity_W_m2(absorbed, node, T_max_K=T_max_K))


# ---- ElectroThermalResult -> REL1 TDDB -------------------------------------------------------

def tddb_tbd_from_electrothermal(et_result, oxide_layer_name: str,
                                 params: TddbParams) -> float:
    """Time-to-breakdown [s] of the named oxide layer under the converged electro-thermal
    stress: (|mean E_z|, mean T) of that layer via oxide_stress_from_electrothermal, into
    params.tbd_s (E-model). Calibrate `params` with TddbParams.calibrated() first."""
    E_ox, T_K = oxide_stress_from_electrothermal(et_result, oxide_layer_name)
    return float(params.tbd_s(E_ox, T_K))
