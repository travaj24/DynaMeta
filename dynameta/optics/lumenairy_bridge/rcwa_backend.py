"""Lumenairy RCWA as a DynaMeta optical backend (roadmap v0.5 A1).

BRIDGE, not vendor: Lumenairy (>= 5.22, the single floor in _common.VERSION_FLOOR) is a REQUIRED dependency of dynameta but is
imported lazily inside the functions -- importing this module (and base dynameta) stays
fast and matplotlib-free; calling against a broken/outdated environment raises with an
install hint. The PHYSICS conventions are IDENTICAL on both sides (public exp(-i omega t),
Im(eps) > 0 for absorbers, metres) -- verified in docs/lumenairy_rcwa_port_wishlist.md -- so eps,
lengths and the time convention cross unchanged. Three translations DO happen here (audit S4-11,
wave-7 2026-11-02: this paragraph used to say "no sign translation happens here"): (1) incidence
angles degrees -> radians (_common.angles_rad); (2) the polarization label -> lumenairy's lab Jones
ROW (_common.pol_row); (3) for p-pol, the lab-basis Jones x-components are mapped onto the
Byrnes-tmm p-hat-basis Fresnel amplitudes DynaMeta's FEM phases were validated against
(_common.p_basis_conversion): r -> -r (lumenairy's r_xx is EXACTLY -r_p, a pi phase) and
t -> t * cos(theta_i) / cos(theta_t) (the lab-x projection of the p-hat field). R/T need no
conversion. 'x' / 'y' pass through (the bases coincide).

Pinned cross-library contracts (first pinned at lumenairy 5.14.2, re-verified on the 5.22 floor; file:line refs in the roadmap):
- RCWAStack(period, period_y=, n_superstrate=, n_substrate=, n_orders=, n_orders_y=):
  region media are complex refractive INDICES (or callables wl -> n); a stack is 1-D iff
  period_y/n_orders_y are omitted; the lattice is rectangular.
- add_layer(thickness, eps=|eps_cell=|eps_tensor_cell=|shapes=+eps_background=,
  formulation=): eps_cell is (Sx, Sy) with axis 0 = x; patterned cells need
  Sx >= 4 n_orders_x + 1 (and y alike); 'li'/'fff' formulation is isotropic-eps_cell-only.
- set_source(wavelength, theta=, phi=) in metres/radians; solve(retain_internal=,
  stabilize=) -> RCWAResult; efficiencies() rows are keyed INCIDENT E_x (row 0) /
  E_y (row 1) -- never relabel as TE/TM; jones_reflection()/jones_transmission() are the
  ZEROTH-order (2, 2) lab-basis Jones, columns = incident Ex/Ey.
- A stack holding any dispersive callable must be solved via solve_vs_wavelength; this
  bridge instead rebuilds a CONCRETE stack per wavelength (the pipeline's assemble_at
  closure is the dispersive source of truth, including carrier-modulated eps), which also
  fixes the band-centre end-media freeze of the generic sweep path: end media are
  re-derived per wavelength via end_media_indices.

Polarization mapping (DynaMeta OpticalSpec 'x'|'y'|'p' -> result row): 'x' -> row 0,
'y' -> row 1, 'p' -> row 0 (plane of incidence is x-z at phi = 0, so the in-plane TM field
is x-polarized at normal/oblique non-conical incidence). OpticalResult.R/T are the TOTAL
(co- + cross-polarized) order-summed efficiencies (also exposed as R_flux/T_flux -- for
RCWA they are the same Poynting quantities); r/t are the CO-polarized zeroth-order complex
amplitudes (phase-bearing, the modulator observable).

CONICAL incidence (azimuth != 0) is supported for the 2-D lattice (audit 8.1-1, the last
conical gap): the native lab rows are s/p MIXTURES at phi != 0, so R/T + the co-pol r/t are
SYNTHESIZED from the per-order complex amplitudes for the rotated s/p eigen-polarization
(the shared _common.conical_synthesis, from res.per_order_amplitudes). Per-layer absorption
is left unset at conical (layer_absorption gives the per-lab-pol diagonal only; the rotated-
pol absorbed power is a quadratic form whose cross term it does not expose -- A = 1 - R - T
stays exact).

Inclusion handling: laterally structured layers are RASTERIZED onto a (Sx, Sy) cell with
the validated shared rasterizer (optics.rasterize, also the structured-FDTD painter:
background + priority-ordered inclusion overpainting).
Inclusion eps comes from the material registry at lambda; an eps_by_region override applies
to the layer BACKGROUND only (rasterizer contract) -- a uniform TENSOR override too (2026-11-02:
the inclusions are painted over the tensor background as eps * I3 by
optics.rasterize.layer_eps_tensor_cell, the PMM2D bridge's reading; this branch used to drop them). The raster is PERIODIC (wrap-aware,
_shapes.periodic_layer_eps_cell, 2026-11-02): a boundary-crossing inclusion contributes its
wrapped piece at the opposite edge -- the FEM mesher's convention. The shared rasterizer used to
clip it, which solved a different (narrower) structure; byte-identical for inclusions inside the
closed cell.

ANALYTIC SHAPES (2026-11-02, the follow-on this paragraph used to defer; lumenairy 5.50
add_layer(shapes=, eps_background=); FLOOR lumenairy 5.31 -- below it the shape validator that
refuses overlaps is missing, so 'auto' rasterizes and 'analytic' raises
LumenairyFeatureUnavailable, _common.FEATURE_FLOORS): `inclusion_model` = 'auto' (THE DEFAULT since v0.13.0 --
analytic where exact, raster otherwise), 'raster' (the pre-0.13 default) or 'analytic' (every
inclusion layer goes through lumenairy's exact form factors -- raises ShapesNotRepresentable when a
layer cannot). Why 'auto' became the default (user decision, 2026-11-02): it removes the point
raster's edge-quantization error (each edge up to half a pixel off: ~1e-3 in R on a 0.4-period
line at S = 128, up to 1.1e-2 on a 0.43-duty metal line, erratic in n_orders --
_default_raster_samples) wherever the geometry is exactly representable, and never refuses what
the raster accepts. SHAPES UNDER A FORMULATION (2026-11-02; relaxed for lumenairy >= 5.50.1):
lumenairy <= 5.50.0 shapes layers are Laurent by contract (add_layer raises for
formulation='li' with shapes=), so there 'auto' uses them ONLY with formulation='laurent' and
rasterizes every inclusion layer under 'li'/'fff' (honouring the rule the caller asked for --
the convergent one for TM on metal walls); 'analytic' + a non-Laurent formulation raises
ValueError rather than silently solving Laurent. POINT-RASTER LATTICE (2026-11-02):
default S = max(128, 8 N) per axis (was max(128, 4 N + 1); identical for N <= 16), measured in
_default_raster_samples; 'raster' + cell_samples=128 reproduces the pre-0.13 numbers bit for bit.
Rectangle / Circle / Ellipse map 1:1 (both sides axis-aligned; frame pin: absolute cell frame
[0, P], lumenairy's default centre P/2 -- _shapes module docstring); polygons and OVERLAPPING
inclusions (priority painting has no additive form-factor equivalent; lumenairy's own validator
decides) stay rasterized. Same Laurent direct-rule factorization as an eps_cell layer, so the
analytic layer is the S -> infinity limit of an AREA-AVERAGED raster at fixed n_orders: MEASURED
(disk r = 0.32 P, eps 6+0.4i in 2.1, n_orders 5x5, theta 0.2 rad): |dR| 1.1e-4 / 2.8e-5 / 6.4e-6 /
1.7e-6 at S = 64/128/256/512 (O(S^-2)), while a 1 % radius error moves R by 1.9e-3 and a 1 %
background error by 1.8e-3 (tests/test_lumenairy_shapes_sem_wrap.py). The bridge's POINT-sampled
raster at its default S = 128 is 3.8e-5 off the exact geometry on the same disk (Gauss-circle
area noise, O(S^-4/3)) -- the analytic path is the more accurate one. LUMENAIRY 5.50.1 (2026-11-03, the
fixes for this bridge's L-3 / L-4 findings) lifts both Laurent-era limits, each gated by
_common.FEATURE_FLOORS: shapes layers take formulation='li' (1-D y-invariant shapes, 2-D
axis-aligned rectangles; 'auto' rasterizes only the layers lumenairy refuses) and
layer_absorption() works on them, so absorption=True keeps the analytic layers. Below 5.50.1 the
old rules stand: 'li'/'fff' rasterizes every inclusion layer under 'auto' (refused under
'analytic'), and absorption=True rasterizes every layer under 'auto' (refused under 'analytic').

FORMULATION RULE (v0.16.0; formulation='auto' is the default of design_to_rcwa_stack,
make_lumenairy_rcwa_solver and LumenairyStackSolver; 'laurent' restores v0.15.1). Every lumenairy
add_layer call names its factorization, so a result never depends on lumenairy's own default
(lumenairy 5.51 changes it to 'auto'). A patterned isotropic layer (inclusions on a scalar
background, or a laterally structured scalar gridded field) is HARD when it holds Re(eps) < 0
anywhere or max|eps| / min|eps| > 10. 'auto' (_auto_formulation, versioned by AUTO_RULE_VERSION)
resolves per stack: no hard layer -> 'laurent'; a 1-D stack -> 'li' when the incident field has
an E_x component (polarization 'x' / 'p', or conical incidence), 'laurent' for E along the lines;
a 2-D stack -> 'laurent' (measured: lumenairy's 2-D Li rule drifted away from two
converged references on a metal-patch array; see _auto_formulation; pass 'li' to use it).
Why: Laurent's rule is the wrong factorization for the field component normal to a
high-contrast wall; on the generic metal strip over a mirror in TM (tests/_strip_cell_fixture) it
is 0.50 off in r at n_orders 50, 0.17 at 100 and 0.023 at 200, where 'li' (analytic shapes,
lumenairy >= 5.50.1) is 0.070 / 0.025 / 8.9e-3 off (reference: the 2-D TM FEM, optics.fem_2d_tm). TE
cells and laterally uniform stacks are bit-identical to v0.15.1. An explicit 'laurent' on a stack
the rule sends to 'li' warns once per solver. The resolved rule is recorded on the stack
(stack.dynameta_formulation / _requested / _reason) and on the solver (last_formulation). The
sweep path resolves once per spectrum. With 'li' and absorption=True the per-layer split is the
face-flux difference (_face_flux_split).

RASTER LATTICE (v0.16.0; raster_sampling='area' is the default, 'point' restores v0.15.1).
lumenairy reads eps_cell[j, i] as the value at the NODE (j Px/Sx, i Py/Sy); the analytic shapes
sit in the true frame. The default raster (_raster.area_weighted_cell) gives pixel j the AREA
AVERAGE of the permittivity over [x_j - Px/(2 Sx), x_j + Px/(2 Sx)] (exact coverage for
rectangles, 16 x 16 sub-samples per pixel for curved / polygonal shapes), so every drawn width is
realised exactly whatever the lattice, and raster and analytic layers share one frame (the point
raster sampled cell CENTRES and moved each edge by up to half a pixel: the generic cell's 150 nm
strip realised as 150.495 nm at S = 404, r off by 0.053 at n_orders 50; on S = 400 both of its edges
happen to fall on pixel boundaries and the width is exact). Under 'li' (lumenairy >= 5.31) the raster
carries its inverse-rule companion (harmonic across the wall: add_layer(eps_cell_normal=)). A
laterally structured gridded EpsField is resampled BY ITS OWN x (and y) COORDINATES onto the same
lattice (_raster.area_resample_cell: exact pixel means of its periodic piecewise-linear
interpolant; endpoint-inclusive, endpoint-exclusive and cell-centred grids all read correctly),
with at least as many pixels as distinct samples, so the default carrier grid no longer caps
n_orders. Default lattice: max(128, 8 N) per axis, and at least _AREA_SAMPLES_1D on a 1-D stack.

POLARIZATION VOCABULARY (audit V-8): this module speaks the integer lab `row` 0/1 (0 = incident E_x,
1 = incident E_y) -- an INDEX, not a label. It is one of five spellings in the repo -- {'s','p'} is
the PLANE-OF-INCIDENCE spelling, {'x','y','p'} OpticalSpec's LAB AXIS ({'x','y','p'} -> row is
_common.pol_row, and the INVERSE is refused: row 0 is ambiguous), {'te','tm'} the grating bridge's,
and `pol_axis` hydro_fem's. The map, the `normalize_pol` converter and the normal-incidence /
azimuth caveats live in `dynameta.core.polarization`. The set ACCEPTED here is UNCHANGED; acceptance
unification (b), the V-8 follow-on, widened only the two PLANE-OF-INCIDENCE families ({'s','p'} and
{'te','tm'}), whose aliases name the same physical mode in every geometry they cover; this
vocabulary's crossings depend on the azimuth (or have no image at all), so they stay STRICT and are
made explicitly, through normalize_pol.
"""

from __future__ import annotations

from dynameta.core.eps_field import require_solver_time_convention as _require_eps_convention

import os
import time
import warnings
from typing import List, Optional

import numpy as np

from dynameta.core.interfaces import OpticalResult
from dynameta.core.lateral import (is_laterally_uniform, is_uniform_cell, is_y_invariant,
                                   tensor_couples_y)
from dynameta.core.layered import (LayeredStack, collapse_regions_to_layers,
                                   slice_layer_eps_field)
from dynameta.optics.rasterize import cell_axes, layer_eps_tensor_cell
from dynameta.optics.lumenairy_bridge._raster import (area_resample_cell, area_weighted_cell,
                                                      area_weighted_tensor_cell)
from dynameta.optics.lumenairy_bridge._shapes import (ShapesNotRepresentable,
                                                      layer_shapes_payload,
                                                      periodic_layer_eps_cell)
from dynameta.optics.tmm_reference import S as _S_NM
from dynameta.optics.tmm_reference import end_media_indices
from dynameta.geometry.guards import require_supported_geometry

__all__ = ["design_to_rcwa_stack", "make_lumenairy_rcwa_solver", "LumenairyStackSolver",
           "rcwa_result_to_optical_result"]

# Shared plumbing now lives in _common.py (audit 6.3: rcwa_backend had become the bridge's
# unofficial common module). Underscore aliases keep the historical import surface working.
from dynameta.optics.lumenairy_bridge._common import (angles_rad as _angles_rad,
                                                      conical_synthesis as _conical_synthesis,
                                                      guard_incidence_side as _guard_incidence_side,
                                                      lumenairy_has as _lumenairy_has,
                                                      lumenairy_version as _lumenairy_version,
                                                      p_basis_conversion as _p_basis_conversion,
                                                      pol_row as _pol_row,
                                                      refuse_gain_half_space as _refuse_gain_half_space,
                                                      require_lumenairy as _require_lumenairy,
                                                      require_lumenairy_feature as _require_feature)
from dynameta.warning_types import AdvisoryWarning


def _min_samples(n_orders: int) -> int:
    """lumenairy's HARD sampling bound for a patterned cell (S >= 4 N + 1 per axis: the Laurent
    table spans difference orders -2N..2N). A validity floor, NOT an accuracy target -- see
    _default_raster_samples for what the bridge's own raster uses."""
    return 4 * int(n_orders) + 1


_RASTER_FLOOR = 128      # the historical default floor (pre-0.13: S = max(128, 4 N + 1))
_RASTER_PER_ORDER = 8    # 2026-11-02: default S = max(128, 8 N)
# v0.16: the default lattice floor of the AREA raster on a 1-D stack. An area-averaged pixel that
# a wall crosses smooths the wall over one pixel, an O(h^2) error whose constant depends on where
# the wall falls inside its pixel; on a 1-D stack the raster and its FFT cost nothing next to the
# eigensolve, so the default lattice is simply made fine. MEASURED (the generic metal-strip cell
# of tests/_strip_cell_fixture in TM, 'li' with the companion, n_orders 50, |r - analytic 'li'|):
# S = 400 (8 N) 3.7e-2, 800 1.6e-2, 1600 4.3e-3, 2048 2.6e-3, 2960 1.2e-3, 4096 6.5e-4, 6000 3.0e-4,
# 8192 1.6e-4, 12000 7.5e-5 (O(S^-2)). The point raster depends on where the edges fall: on these
# lattices both edges of this cell sit on pixel boundaries (exact width) and it is 2.7e-2 / 8.3e-3 /
# 2.1e-3 / 1.3e-3 / 6.2e-4 at 400 / 800 / 1600 / 2048 / 2960; on non-aligned lattices it carries the
# width error, 5.3e-2 / 1.3e-2 / 4.8e-3 at S = 404 / 2964 / 8196, where the area raster is 3.3e-2 /
# 7.9e-4 / 1.0e-4. ~0.07 s per solve at every S.
_AREA_SAMPLES_1D = 8192


def _default_raster_samples(n_orders: int) -> int:
    """Default point-raster sample count per patterned axis: S = max(128, 8 N) (2026-11-02).

    The pre-0.13 rule was max(128, 4 N + 1), i.e. lumenairy's validity floor. It is byte-identical
    to the new rule for N <= 16 (both 128) and differs from N = 17 up. MEASURED 2026-11-02 on the
    validation/lumenairy_pmm_bridge GATE B fixture (600 nm period, 50 % lines centred at P/2,
    120 nm, air | glass, 1.31 um, normal incidence; |R - converged reference| at N = 8 / 16 / 24 /
    32 / 48 / 64 / 96):
        TE metal (eps -20+2i), Laurent:  old 1.8e-4 4.8e-5 7.5e-5 4.6e-3 3.1e-3 2.3e-3 1.6e-3
                                          new 1.8e-4 4.8e-5 2.7e-5 1.7e-5 8.3e-6 4.9e-6 2.3e-6
        TE dielectric (eps 4+0.3i):      old 3.7e-6 3.8e-6 4.6e-6 8.3e-4 5.5e-4 4.1e-4 2.8e-4
                                          new 3.7e-6 3.8e-6 1.9e-6 1.1e-6 5.1e-7 3.0e-7 1.3e-7
        TM dielectric, 'li':             old 7.3e-5 1.3e-5 1.5e-6 3.7e-4 2.5e-4 1.9e-4 1.2e-4
                                          new 7.3e-5 1.3e-5 5.4e-6 2.9e-6 1.2e-6 6.1e-7 2.4e-7
    The old jump at N = 32 is NOT Fourier aliasing in the first place: at N >= 32 the old rule
    switched to the ODD lattice S = 4 N + 1 = 129, on which the fixture's edges (x = P/4, 3P/4) fall
    mid-pixel, so the point raster realised a 65/129 = 0.5039 duty instead of 0.5 -- an O(1/S)
    EDGE-QUANTIZATION error (S = 8 N + 1 reproduces it: 2.3e-3 at N = 32). On S a multiple of 4 the
    edges sit between pixel centres, the realised geometry is exact, and the residual is the genuine
    aliasing of point sampling, which S = 8 N keeps below the analytic truncation error x ~4.
    CAVEAT, measured on the same cell with MISALIGNED edges (duty 0.43 and 0.30): there the point
    raster is quantization-limited at ANY S rule -- |dR| on TE metal is erratic in N (duty 0.43,
    new rule: 1.1e-2 1.0e-2 4.1e-3 4.6e-4 3.2e-3 4.4e-4 4.4e-4; old: 1.1e-2 1.0e-2 1.0e-2 5.2e-3
    2.8e-5 2.6e-3 2.0e-3), set by where the edges land on the lattice (each edge up to half a pixel
    off). Tying S to N bounds that error by O(1/N) alongside the truncation error but cannot make it
    monotone; the cure is the analytic path ('auto' / 'analytic', 2.3e-4 -> 2.0e-7 monotone there)
    or an area-weighted raster. COST: raster + FFT are negligible next to the eigensolve -- 1-D
    N = 32: 11.7 ms (S 129) vs 11.4 ms (S 256); N = 64: 132.6 ms (S 257) vs 131.0 ms (S 512); a
    2-D cell is S^2 point tests + one FFT against an O(((2N+1)^2)^3) eigensolve."""
    return max(_RASTER_FLOOR, _RASTER_PER_ORDER * int(n_orders))


_INCLUSION_MODELS = ("raster", "analytic", "auto")
_FORMULATIONS = ("auto", "laurent", "li", "fff")
_RASTER_SAMPLINGS = ("area", "point")
# Version of the formulation='auto' rule (_auto_formulation). It is part of the solver's
# cache_fingerprint: bump it whenever the rule can return a different answer for some stack, so a
# cache never serves a result computed under the old rule.
AUTO_RULE_VERSION = 1
# a patterned layer is HARD (Laurent's rule converges slowly and from far away for the field
# component across its walls) when it holds Re(eps) < 0 anywhere or its max|eps| / min|eps|
# exceeds this (the thresholds of lumenairy's own 'auto' rule)
_HARD_CONTRAST = 10.0


def _check_inclusion_model(inclusion_model: str, where: str) -> str:
    if str(inclusion_model) not in _INCLUSION_MODELS:
        raise ValueError("{}: inclusion_model must be one of {}, got {!r}".format(
            where, _INCLUSION_MODELS, inclusion_model))
    return str(inclusion_model)


def _check_formulation(formulation, where: str) -> str:
    f = str(formulation).lower() if isinstance(formulation, str) else formulation
    if f not in _FORMULATIONS:
        raise ValueError("{}: formulation must be one of {}, got {!r}".format(
            where, _FORMULATIONS, formulation))
    return f


def _check_raster_sampling(raster_sampling, where: str) -> str:
    if raster_sampling not in _RASTER_SAMPLINGS:
        raise ValueError("{}: raster_sampling must be one of {}, got {!r}".format(
            where, _RASTER_SAMPLINGS, raster_sampling))
    return str(raster_sampling)


def _check_model_formulation(inclusion_model: str, formulation: str, where: str) -> bool:
    """True when analytic shapes may be used for this (inclusion_model, formulation) pair.
    `formulation` is the RESOLVED rule ('laurent' | 'li' | 'fff'), never 'auto'.

    lumenairy >= 5.50.1 (FEATURE_FLOORS['rcwa_shapes_li'], lumenairy audit L-4) applies Li's
    inverse rule ('li' / 'fff' -- the convergent factorization for TM on metal / high-contrast
    walls, Li 1996 JOSA A 13:1870) to analytic shapes layers itself: 1-D stacks with y-invariant
    shapes, 2-D stacks with axis-aligned rectangles. Anything else (a disk / ellipse under 'li' on
    a 2-D stack) is refused by lumenairy's add_layer with ValueError, which _add_shapes_layer turns
    into the usual per-layer decision: 'auto' rasterizes THAT layer (the raster's 'li' path takes
    any pixel pattern), 'analytic' raises ShapesNotRepresentable. The requested rule is honoured
    on every route.

    Below 5.50.1 lumenairy's shapes layers are LAURENT BY CONTRACT (5.50.0 add_layer raises for
    formulation='li' with shapes=), so a non-Laurent request can only be honoured on a RASTERIZED
    layer: 'auto' rasterizes every inclusion layer then (2026-11-02; it used to hand them to the
    Laurent-only shapes path and silently drop the requested rule), and 'analytic' refuses the
    combination outright instead of silently solving Laurent."""
    f = str(formulation).lower()
    if f == "laurent" or _lumenairy_has("rcwa_shapes_li"):
        return inclusion_model != "raster"
    if inclusion_model == "analytic":
        raise ValueError(
            "{}: inclusion_model='analytic' requires formulation='laurent' -- lumenairy's analytic "
            "shapes layers are Laurent (direct rule) by contract and cannot apply the requested "
            "formulation={!r}; use inclusion_model='auto' or 'raster' (both rasterize the "
            "inclusions and honour the factorization rule)".format(where, formulation))
    return False


def _add_shapes_layer(stack, L, design, lambda_m, eps_by_region, px, py, inclusion_model,
                      formulation):
    """Try to add design layer L as an ANALYTIC-shape RCWA layer. True on success; False when
    the layer is not exactly representable and inclusion_model == 'auto' (the caller
    rasterizes); ShapesNotRepresentable when inclusion_model == 'analytic'. lumenairy's add_layer
    validator (disjointness, extent <= period, cumulative area <= 1) is the single authority on
    what the additive form factors can paint; it raises BEFORE appending, so a refused layer
    leaves the stack untouched. `formulation` is ALWAYS passed (v0.16: the Laurent call used to
    omit it and so inherited lumenairy's default, which lumenairy 5.51 changes to 'auto')."""
    from dynameta.optics.rasterize import layer_bg_eps
    try:
        bg = layer_bg_eps(L, lambda_m, design.materials, eps_by_region)
        bg, shapes = layer_shapes_payload(L, lambda_m, design.materials, bg, px, py,
                                          stack_period_y_m=float(stack.period_y))
        try:
            stack.add_layer(L.thickness_m, shapes=shapes, eps_background=bg,
                            formulation=formulation)
        except ValueError as exc:
            raise ShapesNotRepresentable(
                "layer {!r}: lumenairy refused the analytic shapes ({})".format(L.name, exc)
            ) from exc
    except ShapesNotRepresentable as exc:
        if inclusion_model == "analytic":
            raise ShapesNotRepresentable(
                "design_to_rcwa_stack(inclusion_model='analytic'): layer {!r} is not exactly "
                "representable by lumenairy analytic shapes -- {} (inclusion_model='auto' "
                "rasterizes such layers)".format(L.name, exc)) from exc
        return False
    return True


def _is_hard(values) -> bool:
    """Re(eps) < 0 anywhere, or max|eps| / min|eps| > _HARD_CONTRAST (inf for a zero sample)."""
    v = np.asarray(values, dtype=complex).ravel()
    if v.size == 0:
        return False
    if bool(np.any(v.real < 0.0)):
        return True
    a = np.abs(v)
    lo = float(np.min(a))
    return lo == 0.0 or float(np.max(a)) / lo > _HARD_CONTRAST


def _hard_layers(design, eps_by_region, lambda_m):
    """Names of the HARD isotropic patterned layers -- the ones a factorization applies to
    (inclusions on a scalar background, laterally structured scalar gridded fields) holding
    Re(eps) < 0 anywhere or a contrast max|eps| / min|eps| above _HARD_CONTRAST."""
    from dynameta.optics.rasterize import layer_bg_eps
    hard = []
    for L in design.stack.layers:
        ef = eps_by_region.get(L.name)
        if ef is not None and not getattr(ef, "is_uniform", True):
            if getattr(ef, "is_tensor", False):
                continue                                  # tensor cells: own factorization
            v = np.asarray(ef.values_zyx, dtype=complex)
            if is_laterally_uniform(v):
                continue                                  # laterally uniform: no walls
            if _is_hard(v):
                hard.append(L.name)
            continue
        if not L.inclusions or (ef is not None and getattr(ef, "is_tensor", False)):
            continue
        vals = [layer_bg_eps(L, lambda_m, design.materials, eps_by_region)]
        vals += [design.materials.get(inc.material).eps(lambda_m) for inc in L.inclusions]
        if _is_hard(vals):
            hard.append(L.name)
    return hard


def _auto_formulation(hard, *, is_2d: bool, pol_row: int, conical: bool):
    """THE FORMULATION RULE (formulation='auto'; versioned by AUTO_RULE_VERSION).
    Returns (rule, reason) with rule 'li' | 'laurent'.
      * no hard patterned layer -> 'laurent' (both rules converge; Laurent keeps every dielectric
        result as it was);
      * 1-D stack: 'li' when the incident field has an E_x component (polarization 'x' / 'p', or
        conical incidence), i.e. a field component ACROSS the walls; 'laurent' for E along the
        lines (TE), where Laurent is the right (direct) rule and v0.15.1 stays bit for bit;
      * 2-D stack: 'laurent' (the v0.15.1 rule) -- MEASURED, not assumed (an earlier draft of
        the rule sent hard 2-D rectangle layers to 'li'). On a 2-D array of 300 nm square
        metal patches (eps -3+1i, P 600 nm, 60 nm on a 150 nm spacer, 1.31 um, x) lumenairy's 2-D
        Li rule -- analytic rectangles and eps_cell alike, which agree to 5e-5 -- drifts AWAY from
        the reference as n_orders grows (R 0.07441 / 0.07428 / 0.07392 / 0.07373 / 0.07365 at
        4 / 8 / 12 / 16 / 20) while Laurent closes in (0.07987 ... 0.07441), against two
        independent references that agree with each other: the 3-D FEM (R 0.074469, order 3)
        and the 2-D spectral-element PMM (0.074485, n_modes 8). On a metal-DISK array (eps
        -3+1i, against the 3-D FEM) 'li' on the area raster matched R (7e-6 at n_orders 24) but
        not the phase (|dr| 3.3e-3, Laurent 2.4e-3). Pass formulation='li' explicitly to use
        Li's 2-D rule; the rule's version is bumped when 2-D Li is admitted."""
    if not hard:
        return "laurent", "no metallic or high-contrast patterned layer"
    names = ", ".join(repr(n) for n in hard)
    if not is_2d:
        if pol_row == 0 or conical:
            return "li", ("1-D stack, hard patterned layer(s) {} with the incident E across "
                          "the walls".format(names))
        return "laurent", ("1-D stack, incident E along the walls (TE) on hard layer(s) "
                           "{}".format(names))
    return "laurent", ("2-D stack (hard patterned layer(s) {}): Laurent's rule -- lumenairy's "
                       "2-D Li rule measured off a converged reference on a metal-patch array "
                       "(see _auto_formulation)".format(names))


_EXPLICIT_LAURENT_ADVICE = (
    "{where}: formulation='laurent' on a stack whose patterned layer(s) {names} are metallic or "
    "high-contrast with the incident E across their walls. Laurent's rule converges slowly and "
    "from far away there (measured on a metal-strip cell in TM: 0.50 in r at n_orders 50, 0.17 at "
    "100); formulation='auto' (the default) or 'li' applies Li's inverse rule.")


def _stack_lattice(nx_o, ny_o, *, is_2d, structured, cell_samples, raster_sampling="point"):
    """(sx, sy, s_min_x, s_min_y): the raster lattice per patterned axis -- default S = max(128,
    8 N) (_default_raster_samples), and on a 1-D stack under the area raster at least
    _AREA_SAMPLES_1D; an EXPLICIT cell_samples keeps the pre-0.13 formula
    max(cell_samples, 4 N + 1, 128) exactly, so cell_samples=128 reproduces the pre-0.13 lattice."""
    s_min_x = _min_samples(nx_o)
    s_min_y = _min_samples(ny_o) if is_2d else 1
    if cell_samples is None:
        sx = _default_raster_samples(nx_o) if structured else s_min_x
        if structured and not is_2d and raster_sampling == "area":
            sx = max(sx, _AREA_SAMPLES_1D)
        sy = _default_raster_samples(ny_o) if is_2d else 1
    else:
        sx = max(int(cell_samples), s_min_x, _RASTER_FLOOR if structured else s_min_x)
        sy = max(int(cell_samples), s_min_y, _RASTER_FLOOR) if is_2d else 1
    return sx, sy, s_min_x, s_min_y


def _distinct_samples(axis_m, period_m: float) -> int:
    """Number of distinct periodic samples of a lateral axis: an endpoint-inclusive grid (first
    and last sample one period apart) carries its first sample twice."""
    a = np.asarray(axis_m, dtype=np.float64).ravel()
    if a.size >= 2 and abs((a[-1] - a[0]) - float(period_m)) <= 1e-9 * float(period_m):
        return int(a.size - 1)
    return int(a.size)


def design_to_rcwa_stack(design, lambda_m: float, *, eps_by_region=None, n_orders: int = 11,
                         n_orders_y: Optional[int] = None, n_slices: Optional[int] = None,
                         cell_samples: Optional[int] = None, formulation: str = "auto",
                         inclusion_model: str = "auto", raster_sampling: str = "area"):
    """Translate a DynaMeta Design (+ optionally the bridge's eps_by_region) into a CONCRETE
    Lumenairy RCWAStack at one wavelength. Returns (stack, layer_names_top_first) where the
    name list is aligned with the Lumenairy layers (graded fields contribute several slabs
    per design layer) -- the key for mapping per-layer absorption back to design layers.

    formulation: 'auto' (default since v0.16.0; the module's FORMULATION RULE), 'laurent', 'li'
    or 'fff' (lumenairy's alias of 'li'). Every lumenairy add_layer call names its factorization
    explicitly, so the result never depends on lumenairy's own default. The stack records what
    ran: stack.dynameta_formulation ('li' | 'laurent'), .dynameta_formulation_requested,
    .dynameta_formulation_reason. An explicit 'laurent' on a stack the rule sends to 'li' emits
    one AdvisoryWarning per call (once per solver through make_lumenairy_rcwa_solver).

    raster_sampling: 'area' (default since v0.16.0) paints every rasterized layer as an
    AREA-WEIGHTED raster on lumenairy's node lattice (the module's RASTER LATTICE paragraph; with
    'li' and lumenairy >= 5.31 also the inverse-rule companion eps_cell_normal), and resamples a
    laterally structured gridded EpsField by ITS OWN x (and y) coordinates onto that lattice;
    'point' is the v0.15.1 raster bit for bit (cell-centred point samples; gridded values passed as
    they come).

    inclusion_model: 'auto' (default since v0.13.0: analytic where exact -- and, with 'li',
    where lumenairy >= 5.50.1 takes Li's rule on that layer -- raster otherwise), 'raster' or
    'analytic' (lumenairy analytic shapes for EVERY inclusion layer, raises
    ShapesNotRepresentable otherwise; ValueError with 'li' below lumenairy 5.50.1) -- see the
    module docstring. The resolved factorization rule is honoured on every route, never dropped.

    cell_samples (the raster lattice, per patterned axis): None (default) = max(128, 8 N), N the
    axis' n_orders (2026-11-02, _default_raster_samples -- byte-identical to the pre-0.13
    max(128, 4 N + 1) for N <= 16), and at least _AREA_SAMPLES_1D = 8192 for the area raster on a
    1-D stack (v0.16); an explicit value = max(cell_samples, 4 N + 1, 128), the
    pre-0.13 formula unchanged, so inclusion_model='raster', raster_sampling='point',
    formulation='laurent', cell_samples=128 reproduces the pre-0.13 raster numbers at every
    n_orders. A gridded field under 'area' uses at least as many pixels as it has distinct x
    samples, so its default grid never caps the usable order count.

    Layer dispatch, in order: a structured (gridded) EpsField -> sliced eps_cell /
    eps_tensor_cell slabs (slice_layer_eps_field: the slab total must equal the layer thickness);
    a uniform tensor EpsField -> a tiled eps_tensor_cell (on a layer WITH inclusions: the tensor
    is the BACKGROUND and the inclusions are painted over it as eps * I3); a uniform scalar
    EpsField -> eps; a graded laterally-uniform EpsField -> sliced uniform slabs; inclusions -> an
    analytic shapes layer or a raster; else the material eps at lambda. Design layers are
    bottom -> top, the RCWA stack wants superstrate-side first -> reversed()."""
    stack, names = _build_rcwa_stack(
        design, lambda_m, eps_by_region=eps_by_region, n_orders=n_orders, n_orders_y=n_orders_y,
        n_slices=n_slices, cell_samples=cell_samples, formulation=formulation,
        inclusion_model=inclusion_model, raster_sampling=raster_sampling,
        where="design_to_rcwa_stack")
    if stack.dynameta_formulation_advice is not None:
        warnings.warn(stack.dynameta_formulation_advice, AdvisoryWarning, stacklevel=2)
    return stack, names


def _stack_topology(design, eps_by_region):
    """(is_2d, structured, laterally_uniform_only, has_tensor) of a Design under a layer-keyed eps dict.

    A TENSOR layer does not make the stack 2-D by itself. It does only when it
    varies along y or couples y to x / z (eps_xy, eps_yx, eps_yz, eps_zy); a y-invariant tensor without
    y coupling (a diagonal or x-z tensor: a uniaxial graded film, a boundary-scattering Drude film) is
    1-D-compatible, exactly like a y-invariant scalar grid, so the factorization rule is decided by the
    hard patterned layers themselves (before this rule such a film on a metal-strip cell went to a 2-D
    Laurent stack, far less accurate than the same film as a scalar field, at (2N+1)^2 cost). The lateral
    tests are
    core.lateral's, shared by every optical route."""
    py = float(design.unit_cell.period_y_m)

    def _y_invariant(ef):
        v = ef.values_zyx
        return v is None or v.shape[1] == 1 or is_y_invariant(v)

    def _couples_y(ef):
        if getattr(ef, "is_uniform", True):
            return tensor_couples_y(ef.tensor)
        return tensor_couples_y(ef.values_zyx)

    def _lamellar(L):
        """All inclusions are full-y rectangles (y-invariant grating lines) -> the layer is
        1-D-compatible and the stack can stay 1-D (a genuinely cheaper and
        better-conditioned solve than a y-degenerate 2-D one)."""
        for inc in L.inclusions:
            if getattr(inc.shape, "kind", "") != "rectangle":
                return False
            _, _, ylo, yhi = inc.shape.bbox_m()
            if ylo > 1e-12 * py or yhi < py * (1.0 - 1e-12):
                return False
        return True

    def _structured(L):
        ef = eps_by_region.get(L.name)
        grid = ef is not None and not getattr(ef, "is_uniform", True)
        utensor = ef is not None and getattr(ef, "is_tensor", False)
        return bool(L.inclusions) or grid or utensor

    def _needs_2d(L):
        ef = eps_by_region.get(L.name)
        grid_2d = (ef is not None and not getattr(ef, "is_uniform", True)
                   and ef.values_zyx is not None and not _y_invariant(ef))
        tensor_2d = ef is not None and getattr(ef, "is_tensor", False) and _couples_y(ef)
        return (bool(L.inclusions) and not _lamellar(L)) or grid_2d or tensor_2d

    def _laterally_uniform(L):
        if L.inclusions:
            return False
        ef = eps_by_region.get(L.name)
        if ef is None or getattr(ef, "is_uniform", True):
            return True
        v = ef.values_zyx
        return v is None or is_laterally_uniform(v)

    structured = any(_structured(L) for L in design.stack.layers)
    is_2d = any(_needs_2d(L) for L in design.stack.layers)
    uniform_only = all(_laterally_uniform(L) for L in design.stack.layers)
    has_tensor = any(getattr(eps_by_region.get(L.name), "is_tensor", False) for L in design.stack.layers)
    return is_2d, structured, uniform_only, has_tensor


def _build_rcwa_stack(design, lambda_m: float, *, eps_by_region, n_orders, n_orders_y,
                      n_slices, cell_samples, formulation, inclusion_model, raster_sampling,
                      where):
    """design_to_rcwa_stack without the warning (the solver warns once per solver). An 'auto' request
    is resolved for THIS (design, eps, wavelength) point alone (so a result never depends on the rest of
    a sweep or on a cache)."""
    require_supported_geometry(design, where)   # 2026-11-02: refuse Stack.features / Inclusion.taper
    lum = _require_lumenairy()
    n_super, n_sub = end_media_indices(design, lambda_m)
    _refuse_gain_half_space(where, n_super, n_sub)
    px = float(design.unit_cell.period_x_m)
    py = float(design.unit_cell.period_y_m)
    # mesh-region-keyed dicts (the run_pipeline/FEM bridge output: 'film_inpatch',
    # 'grating__incl0', ...) collapse to design-layer keys; identity when already layer-keyed
    eps_by_region = collapse_regions_to_layers(design, eps_by_region or {})
    inclusion_model = _check_inclusion_model(inclusion_model, where)
    requested = _check_formulation(formulation, where)
    raster_sampling = _check_raster_sampling(raster_sampling, where)

    is_2d, structured, uniform_only, has_tensor = _stack_topology(design, eps_by_region)
    if uniform_only and has_tensor:
        # audit S4-12 (wave-7, 2026-11-02): a stack that is 2-D ONLY because it holds a uniform
        # TENSOR layer (no inclusion, no laterally varying grid anywhere) is laterally UNIFORM, so
        # its diffraction orders decouple exactly and only the specular order carries power; the
        # caller's n_orders bought nothing but cost: (2N+1)^2 coupled orders for a planar layer.
        # Clamp to lumenairy's minimum (1 per axis: 3 x 3 orders). MEASURED 2026-11-02 (lumenairy
        # 5.50; air | ITO 30 nm | 800 nm biaxial-lossy LC (n_o 1.5, n_e 1.7, azimuth 30 deg, tilt
        # 25 deg, +0.01i) | ITO 30 nm | glass, 1.31 um, n_orders 11), six x/y/p normal / 20 deg /
        # conical (phi 35) cases: 17.1-24.8 s -> 0.004-0.011 s per solve (~3500x); R/T/r/t vs the
        # old 23 x 23-order solve <= 8.4e-13, A_independent <= 5.7e-13, and vs the Berreman 4x4
        # oracle (phi = 0) <= 1.4e-14 -- the clamped solve is the CLOSER one (the old residual is
        # round-off accumulated over 529 coupled orders). Gate: tests/test_lumenairy_bridge.py::
        # test_rcwa_uniform_tensor_stack_clamps_orders (LumenairyStackSolver clamps the same way:
        # 0.007-0.011 s, <= 1.4e-14 from Berreman). Not rerouted to Berreman: the RCWA result
        # keeps its own per-order / conical / absorption contract.
        n_orders = n_orders_y = 1

    # ---- the factorization (v0.16) ----------------------------------------------------------
    theta, phi = _angles_rad(design.optical)
    hard = _hard_layers(design, eps_by_region, lambda_m)
    rule, reason = _auto_formulation(hard, is_2d=is_2d, pol_row=_pol_row(design.optical),
                                     conical=abs(phi) > 1e-12)
    advice = None
    if requested == "auto":
        resolved = rule
        if (inclusion_model == "analytic" and resolved == "li"
                and not _lumenairy_has("rcwa_shapes_li")):
            # below lumenairy 5.50.1 analytic shapes are Laurent-only: the explicit 'analytic'
            # request wins over the rule, and the user is told what that costs
            resolved = "laurent"
            reason = ("inclusion_model='analytic' below lumenairy 5.50.1 (analytic shapes are "
                      "Laurent-only); the rule asked for 'li': " + reason)
            advice = _EXPLICIT_LAURENT_ADVICE.format(where=where, names=", ".join(
                repr(n) for n in hard))
        f_lum = resolved
    else:
        resolved = "laurent" if requested == "laurent" else "li"
        f_lum = requested                       # the caller's own spelling ('fff' stays 'fff')
        reason = "explicit formulation={!r}".format(requested)
        if requested == "laurent" and rule == "li":
            advice = _EXPLICIT_LAURENT_ADVICE.format(where=where, names=", ".join(
                repr(n) for n in hard))

    use_shapes = _check_model_formulation(inclusion_model, f_lum, where)
    if use_shapes and not _lumenairy_has("rcwa_analytic_shapes"):
        # 2026-11-02: below lumenairy 5.31 add_layer(shapes=) exists but does NOT refuse
        # overlapping shapes (measured on 5.22.0: two overlapping disks accepted, form factors
        # ADDED), so the "lumenairy refused -> rasterize" decision below would silently build a
        # wrong layer. 'auto' rasterizes every inclusion layer there; 'analytic' is refused.
        if inclusion_model == "analytic":
            _require_feature("rcwa_analytic_shapes",
                             "design_to_rcwa_stack(inclusion_model='analytic')")
        use_shapes = False
    # the inverse-rule companion of an area raster (lumenairy >= 5.31, 'li' only)
    companion = (raster_sampling == "area" and resolved == "li"
                 and _lumenairy_has("rcwa_eps_cell_normal"))

    if is_2d:
        stack = lum.RCWAStack(px, period_y=py, n_superstrate=complex(n_super),
                              n_substrate=complex(n_sub), n_orders=int(n_orders),
                              n_orders_y=int(n_orders_y if n_orders_y is not None
                                             else n_orders))
        nx_o, ny_o = stack.n_orders_x, stack.n_orders_y
    else:
        stack = lum.RCWAStack(px, n_superstrate=complex(n_super),
                              n_substrate=complex(n_sub), n_orders=int(n_orders))
        nx_o, ny_o = stack.n_orders_x, 0

    # audit S4-17 (wave-7, 2026-11-02): the end media ride on the stack so the solver closure
    # does not evaluate end_media_indices a second time per solve (same values, bit for bit)
    stack.dynameta_end_media = (n_super, n_sub)
    stack.dynameta_formulation = resolved
    stack.dynameta_formulation_requested = requested
    stack.dynameta_formulation_reason = reason
    stack.dynameta_formulation_advice = advice
    stack.dynameta_raster_sampling = raster_sampling
    sx, sy, s_min_x, s_min_y = _stack_lattice(nx_o, ny_o, is_2d=is_2d, structured=structured,
                                              cell_samples=cell_samples,
                                              raster_sampling=raster_sampling)

    no_companion: List[str] = []                       # 'li' area-raster layers without the companion

    def _iso(t, cell, pair=None, area_name=None):
        kw = {"eps_cell_normal": pair} if pair is not None else {}
        if area_name is not None and resolved == "li" and pair is None and area_name not in no_companion:
            no_companion.append(area_name)
        stack.add_layer(t, eps_cell=cell, formulation=f_lum, **kw)

    names: List[str] = []
    for L in reversed(design.stack.layers):              # superstrate side first
        ef = eps_by_region.get(L.name)
        if ef is not None and not getattr(ef, "is_uniform", True):
            # gridded EpsField (carrier/effect-graded; axes in nm solver units like the
            # TMM extractor) -> slice_layer_eps_field returns slabs in ASCENDING z
            # (substrate-side first) after checking that they sum to the layer thickness (the
            # slab-thickness guard); this stack is superstrate-first, so reverse (unreversed slabs
            # vertically flip the graded profile)
            slabs = slice_layer_eps_field(ef, L, 1.0 / _S_NM, n_slices=n_slices)
            grid = None
            if raster_sampling == "area" and any(_laterally_structured(s) for s in slabs):
                grid = _grid_lattice(ef, L, px, py, sx=sx, sy=sy, s_min_y=s_min_y, is_2d=is_2d)
            for slab in reversed(slabs):
                if slab.eps is not None:
                    stack.add_layer(slab.thickness_m, eps=complex(slab.eps),
                                    formulation="laurent")
                elif slab.eps_cell is not None:
                    cell = np.asarray(slab.eps_cell, dtype=complex)
                    if grid is not None:
                        x_m, y_m, gx, gy = grid
                        cell, pair = area_resample_cell(
                            cell, x_m, y_m, gx, gy, px, py, companion=companion,
                            where="layer {!r} gridded EpsField".format(L.name))
                        _iso(slab.thickness_m, cell, pair, area_name=L.name)
                    else:
                        if not is_2d and cell.shape[1] > 1:
                            cell = cell[:, :1]            # y-invariant grid on a 1-D stack
                        _iso(slab.thickness_m, _meet_sampling(cell, s_min_x, s_min_y))
                else:
                    cell = np.asarray(slab.eps_tensor_cell, dtype=complex)
                    if grid is not None and _laterally_structured(slab):
                        x_m, y_m, gx, gy = grid
                        cell, _ = area_resample_cell(
                            cell, x_m, y_m, gx, gy, px, py,
                            where="layer {!r} gridded EpsField".format(L.name))
                    else:
                        if not is_2d and cell.shape[1] > 1:
                            cell = cell[:, :1]
                        cell = _meet_sampling(cell, s_min_x, s_min_y)
                    stack.add_layer(slab.thickness_m, eps_tensor_cell=cell,
                                    formulation="laurent")
                names.append(L.name)
            continue
        if ef is not None and getattr(ef, "is_tensor", False):
            if L.inclusions:
                # a uniform TENSOR entry on a layer WITH inclusions is the layer BACKGROUND (the
                # rasterizer's scalar-override contract, the FEM region semantics and the PMM2D
                # bridge's reading); each scalar inclusion is painted over it as eps * I3 by the
                # shared periodic tensor painter. Until 2026-11-02 this branch tiled the bare
                # tensor over the WHOLE layer and DROPPED the inclusions: measured on a 600 nm
                # cell, 250 nm LC layer (n_o 1.5 / n_e 1.7, azimuth 30 deg) holding a centred
                # 300 nm eps-4 square, n_orders 6, x-pol: R 0.07647258774158677 with AND without
                # the pillar, bit for bit (r too). lumenairy's analytic shapes take a SCALAR
                # eps_background only, so a tensor-background layer is always rasterized
                # ('analytic' refuses it, 'auto' rasterizes).
                if inclusion_model == "analytic":
                    raise ShapesNotRepresentable(
                        "design_to_rcwa_stack(inclusion_model='analytic'): layer {!r} has a "
                        "TENSOR background (uniform tensor eps_by_region entry) around its "
                        "inclusions; lumenairy's analytic shapes take a scalar eps_background "
                        "only (inclusion_model='auto' rasterizes such layers)".format(L.name))
                if raster_sampling == "area":
                    cell = area_weighted_tensor_cell(L, sx, sy, lambda_m, design.materials,
                                                     ef.tensor, px, py)
                else:
                    x, y = cell_axes(sx, sy, px, py)
                    X, Y = np.meshgrid(x, y, indexing="ij")
                    cell = layer_eps_tensor_cell(L, X, Y, lambda_m, design.materials, ef.tensor,
                                                 period_x_m=px, period_y_m=py)
                stack.add_layer(L.thickness_m, eps_tensor_cell=cell, formulation="laurent")
                names.append(L.name)
                continue
            cell = np.broadcast_to(np.asarray(ef.tensor, dtype=complex),
                                   (s_min_x, max(s_min_y, 1), 3, 3)).copy()
            stack.add_layer(L.thickness_m, eps_tensor_cell=cell, formulation="laurent")
            names.append(L.name)
            continue
        if L.inclusions:
            if use_shapes and _add_shapes_layer(
                    stack, L, design, lambda_m, eps_by_region, px, py, inclusion_model, f_lum):
                names.append(L.name)
                continue
            if raster_sampling == "area":
                cell, pair = area_weighted_cell(L, sx, sy, lambda_m, design.materials,
                                                eps_by_region, px, py, companion=companion)
                _iso(L.thickness_m, cell, pair, area_name=L.name)
            else:
                x, y = cell_axes(sx, sy, px, py)
                X, Y = np.meshgrid(x, y, indexing="ij")
                cell = periodic_layer_eps_cell(L, X, Y, lambda_m, design.materials,
                                               eps_by_region, px, py)
                _iso(L.thickness_m, np.asarray(cell, dtype=complex))
            names.append(L.name)
            continue
        if ef is not None:                               # uniform scalar (effect-modulated)
            stack.add_layer(L.thickness_m, eps=complex(ef.scalar), formulation="laurent")
        else:
            stack.add_layer(L.thickness_m,
                            eps=complex(design.materials.get(L.background_material)
                                        .eps(lambda_m)),
                            formulation="laurent")
        names.append(L.name)
    if no_companion:
        # Li's inverse rule on an area raster needs its harmonic companion (eps_cell_normal); without
        # it the partially filled edge pixels are wrong for the wall-normal field. MEASURED on the
        # generic strip cell (TM, inclusion_model='raster', default 8192-pixel lattice): 3.6e-3 /
        # 1.0e-2 / 4.1e-3 in r at n_orders 11 / 30 / 50 against the analytic layer, vs 2.9e-5 /
        # 1.0e-4 / 1.6e-4 with the companion. Said once per solver.
        why = ("lumenairy {} has no eps_cell_normal (needs >= 5.31: pip install -U lumenairy)".format(
            ".".join(str(p) for p in (_lumenairy_version() or ())) or "?")
            if not _lumenairy_has("rcwa_eps_cell_normal") else
            "the permittivity passes exactly through zero inside a pixel (a lossless sign change, or a "
            "zero sample), where no harmonic mean exists -- add a little loss")
        msg = ("{}: Li's rule runs on the area raster of layer(s) {} WITHOUT its inverse-rule companion: {}. "
               "Expect ~1e-3 in r on metal cells in TM (measured 1.3e-3 .. 2.2e-3 at the default lattice, "
               "vs ~1e-5 with the companion).".format(where, ", ".join(repr(n) for n in no_companion), why))
        stack.dynameta_formulation_advice = (msg if stack.dynameta_formulation_advice is None
                                             else stack.dynameta_formulation_advice + " " + msg)
    return stack, names


def _laterally_structured(slab) -> bool:
    """A sliced slab with more than one lateral sample (an eps_cell, or a tensor cell that is not
    the 1 x 1 cell slice_eps_field emits for a laterally uniform tensor field)."""
    if slab.eps_cell is not None:
        return True
    c = slab.eps_tensor_cell
    return c is not None and np.asarray(c).shape[0] * np.asarray(c).shape[1] > 1


def _grid_lattice(ef, L, px, py, *, sx, sy, s_min_y, is_2d):
    """(x_m, y_m, gx, gy) for area-resampling a laterally structured gridded field: its own
    coordinates in metres and the node lattice it lands on -- at least the stack lattice and at
    least as many pixels as distinct samples per axis (so the default carrier grid never caps the
    usable n_orders; lumenairy needs S >= 4 N + 1)."""
    if ef.x_axis_u is None:
        raise ValueError(
            "layer {!r}: a laterally structured gridded EpsField needs x_axis_u (its lateral "
            "coordinates, in nm) to be registered on the RCWA lattice; raster_sampling='point' "
            "passes the values as they come (the v0.15.1 reading)".format(L.name))
    x_m = np.asarray(ef.x_axis_u, dtype=np.float64) / _S_NM
    v = np.asarray(ef.values_zyx)
    ny = v.shape[1]
    y_inv = (ny == 1 or is_y_invariant(v))
    gx = max(int(sx), _distinct_samples(x_m, px))
    if not is_2d:
        return x_m, np.zeros(1), gx, 1
    if y_inv:
        return x_m, np.zeros(1), gx, int(s_min_y)
    if ef.y_axis_u is None:
        raise ValueError("layer {!r}: a y-varying gridded EpsField needs y_axis_u".format(L.name))
    y_m = np.asarray(ef.y_axis_u, dtype=np.float64) / _S_NM
    return x_m, y_m, gx, max(int(sy), _distinct_samples(y_m, py))


def _meet_sampling(cell, s_min_x: int, s_min_y: int):
    """Tile a too-coarse UNIFORM-pattern axis up to Lumenairy's sampling bound. Genuinely
    patterned axes that violate the bound raise downstream with Lumenairy's own message
    (refine the source grid or lower n_orders)."""
    cell = np.asarray(cell, dtype=complex)
    reps = [max(1, int(np.ceil(s_min_x / cell.shape[0]))),
            max(1, int(np.ceil(s_min_y / max(cell.shape[1], 1))))]
    if reps[0] > 1 or reps[1] > 1:
        uniform_x = bool(np.all(cell == cell[:1, ...]))
        uniform_y = bool(np.all(cell == cell[:, :1, ...]))
        reps[0] = reps[0] if uniform_x else 1
        reps[1] = reps[1] if uniform_y else 1
        tile = reps + [1] * (cell.ndim - 2)
        cell = np.tile(cell, tile)
    return cell


def _face_flux_split(res, stack, row: int, theta: float, phi: float, n_super) -> np.ndarray:
    """Per-layer absorbed fraction for incident lab row `row` as the POYNTING-FLUX DIFFERENCE
    between each layer's two faces (Poynting's theorem: the lateral flux cancels over a period),
    from the retained modal fields -- the tangential E/H are the RCWA state vector, so the face
    fluxes are exact and close on 1 - R - T to round-off, for any factorization. Used for every
    layer of a stack solved with Li's rule, where the Im(eps)|E|^2 volume integral of lumenairy's
    layer_absorption (before 5.51) is not consistent with the inverse-rule field: the truncated
    series of the wall-normal E_x jumps at every wall and is a poor |E_x|^2 inside a metal (the
    defect lumenairy 5.51 fixes the same way for its 'li' layers).

    Normalized by the incident z-flux of a unit-tangential plane wave, kz_inc |E_inc|^2 / k0 with
    kz_inc = n_super cos(theta) and |E_inc|^2 = 1 + (k_t . e / kz)^2 (lumenairy's efficiency
    normalization), so the fractions are an INDEPENDENT measurement of the absorptance (their
    sum is A_independent; 1 - R - T comes from the far field)."""
    recs = list(stack.layers)
    nox = int(stack.n_orders_x)
    noy = int(getattr(stack, "n_orders_y", 0) or 0)
    is_1d = bool(getattr(stack, "is_1d", noy == 0))
    fnx = 2 * nox + 2
    fny = 1 if (is_1d or noy == 0) else 2 * noy + 2
    inc = (1.0, 0.0) if row == 0 else (0.0, 1.0)
    ns = float(np.real(complex(n_super)))
    kz_inc = ns * np.cos(theta)
    kt = ns * np.sin(theta) * np.array([np.cos(phi), np.sin(phi)])
    long_inc = kt[0] * inc[0] + kt[1] * inc[1]
    einc_sq = 1.0 + (long_inc / kz_inc) ** 2
    out = np.zeros(len(recs))
    for i, rec in enumerate(recs):
        fl = res.internal_field([0.0, float(rec.thickness)], component="all", nx=fnx, ny=fny,
                                layer=i, incident=inc)
        sz = np.real(fl["Ex"] * np.conj(fl["Hy"]) - fl["Ey"] * np.conj(fl["Hx"]))
        sz = sz.reshape(2, -1).mean(axis=1)
        out[i] = float(sz[0] - sz[1]) / (kz_inc * einc_sq)
    return out


def rcwa_result_to_optical_result(res, row: int, *, t0: float,
                                  layer_names: Optional[List[str]] = None,
                                  absorption: bool = False,
                                  r_factor: complex = 1.0,
                                  t_factor: complex = 1.0,
                                  conical=None,
                                  absorption_split=None) -> OpticalResult:
    """Map an RCWAResult to a DynaMeta OpticalResult for ONE incident polarization row.
    R/T = TOTAL order-summed efficiencies (== the Poynting R_flux/T_flux); r/t = co-polarized
    zeroth-order complex Jones, multiplied by the p-basis conversion factors when applicable
    (_p_basis_conversion); A = 1 - R - T (budget); with absorption=True, per-layer absorption
    (independent volumetric loss) fills per_region_absorption keyed by DESIGN layer name
    (slabs of a graded layer are summed) and its total fills A_independent.

    conical=(pol, theta, phi, n_super) (azimuth != 0): the native lab rows are s/p MIXTURES, so
    R/T + the co-pol r/t are SYNTHESIZED from res.per_order_amplitudes for the rotated s/p
    eigen-polarization (consumer-gap B, the 2-D-lattice analogue of the PMM path -- one shared
    _common.conical_synthesis). Per-layer absorption at conical is left unset: layer_absorption()
    gives the per-LAB-pol diagonal only, and the rotated-pol absorbed power is a quadratic form
    whose cross term it does not expose (A = 1 - R - T stays exact).

    absorption_split (v0.16): a precomputed per-layer absorbed fraction for `row` (aligned with
    layer_names) used instead of res.layer_absorption() -- the solver passes the face-flux split
    (_face_flux_split) for a stack solved with Li's rule."""
    if isinstance(row, bool) or not isinstance(row, int) or row not in (0, 1):
        # audit V-8: `row` is the integer lab-`row` vocabulary (0 = incident E_x, 1 = E_y).  A
        # sibling label ('x'/'y'/'p', 's'/'p', 'te'/'tm') used to reach the fancy-index and die
        # opaquely.  Accepted set unchanged; LAZY import, failure path only.
        from dynameta.core.polarization import pol_vocabulary_error
        raise pol_vocabulary_error(row, "row", where="rcwa_result_to_optical_result", param="row")
    if conical is not None and abs(float(conical[2])) > 1e-12:
        pol, theta, phi, n_super = conical
        R, T, r, t = _conical_synthesis(res, pol, float(theta), float(phi), n_super)
        return OpticalResult(r=r, R=R, phase_deg=float(np.degrees(np.angle(r)))
                             if r != 0 else 0.0, solve_time_s=time.perf_counter() - t0,
                             t=t, T=T, A=1.0 - R - T, R_flux=R, T_flux=T)
    orders, R2, T2 = res.efficiencies()
    R = float(np.sum(R2[row]))
    T = float(np.sum(T2[row]))
    jr = np.asarray(res.jones_reflection())
    jt = np.asarray(res.jones_transmission())
    r = complex(jr[row, row]) * complex(r_factor)
    t = complex(jt[row, row]) * complex(t_factor)
    A = 1.0 - R - T
    a_ind, pra = None, None
    if absorption and layer_names is not None:
        try:
            if absorption_split is not None:
                la = np.asarray(absorption_split, dtype=float)
            else:
                la = np.asarray(res.layer_absorption())
            if la.ndim == 2:                              # (2, n_layers) per incident pol
                la = la[row] if la.shape[0] == 2 else la[:, row]
            pra = {}
            for name, val in zip(layer_names, la):
                pra[name] = pra.get(name, 0.0) + float(val)
            a_ind = float(np.sum(la))
        except Exception as exc:                          # pragma: no cover - defensive
            warnings.warn("lumenairy bridge: per-layer absorption unavailable ({}); "
                          "A_independent left unset".format(exc), AdvisoryWarning, stacklevel=2)
    return OpticalResult(r=r, R=R, phase_deg=float(np.degrees(np.angle(r))),
                         solve_time_s=time.perf_counter() - t0, t=t, T=T, A=A,
                         A_independent=a_ind, R_flux=R, T_flux=T,
                         per_region_absorption=pra)


def make_lumenairy_rcwa_solver(*, n_orders: int = 11, n_orders_y: Optional[int] = None,
                               n_slices: Optional[int] = None,
                               cell_samples: Optional[int] = None,
                               absorption: bool = False, stabilize: bool = False,
                               formulation: str = "auto",
                               n_workers: Optional[int] = 1, blas_per_worker: int = 1,
                               inclusion_model: str = "auto", raster_sampling: str = "area"):
    """Build an `optical_solver` for run_pipeline backed by Lumenairy RCWA, with the exact
    seam signature fn(design, geo, eps_by_region, lambda_m, n_super, n_sub) -> OpticalResult
    PLUS the sweep-aware `solve_sweep` fast path (one result per wavelength, in order; end
    media re-derived PER WAVELENGTH, fixing the generic band-centre freeze). geo and the
    passed n_super/n_sub are accepted for seam compatibility but unused (the stack re-derives
    end media from the Design, the make_layered_tmm_solver precedent).

    formulation ('auto' default since v0.16.0 | 'laurent' | 'li' | 'fff') and raster_sampling
    ('area' default since v0.16.0 | 'point'): see design_to_rcwa_stack and the module's
    FORMULATION RULE / RASTER LATTICE paragraphs. formulation='laurent', raster_sampling='point'
    reproduces v0.15.1 bit for bit. The solver records what ran in `last_formulation` (and
    `last_formulation_reason`). 'auto' is resolved per (design, eps, wavelength) point on EVERY path,
    so a result is a function of its own point only -- not of which other wavelengths share the sweep or
    which are already in a result cache (solve_sweep used to pick one rule for the whole spectrum, so r
    at one wavelength moved between a single solve, a sweep and a pre-warmed cache). A material crossing the hardness threshold inside a band therefore
    switches the factorization there; pass formulation='li' (or 'laurent') explicitly to keep one rule
    across a band. An explicit 'laurent' on a stack the rule sends to 'li' warns once per solver
    (AdvisoryWarning), not once per wavelength.

    absorption=True: per_region_absorption / A_independent come from lumenairy's
    layer_absorption() on a Laurent solve (unchanged), and from the POYNTING-FLUX difference
    across each layer's faces on a Li solve (_face_flux_split: lumenairy's volume integral is not
    consistent with the inverse-rule field before 5.51); the face fluxes are an independent
    measurement that closes on 1 - R - T to round-off.

    n_workers threads the solve_sweep wavelengths (audit 8.1-2): each wavelength builds its
    OWN concrete stack (the bridge's dispersive policy) and solves on a bounded thread pool,
    with the BLAS cap applied ONCE around the pool on the calling thread -- applying it is
    PROCESS-GLOBAL on OpenBLAS, so a per-worker cap would race (lumenairy audit M4; see the
    comment at the call site). Keep n_workers * blas_per_worker within the core count.

    Results are stored by index, so the output ORDER is identical to serial. THE NUMBERS ARE
    BYTE-IDENTICAL ONLY AT MATCHED BLAS THREADING: each wavelength is an independent LAPACK
    problem, so threading itself changes nothing, but a different BLAS thread count changes
    the reduction order and hence the last bits (measured ~5e-14 on R/T at n_orders=21). The
    serial branch below deliberately does NOT apply the cap -- capping the default path to
    blas_per_worker=1 would be a silent single-thread performance regression -- so serial and
    parallel agree exactly when blas_per_worker equals the ambient BLAS thread count and to
    ~1e-13 otherwise. No physics claim rests on those bits. Default n_workers=1 = the serial
    path, byte-identical to the pre-threading bridge; None = min(cpu_count, n_wavelengths).

    inclusion_model ('auto' default since v0.13.0 | 'raster' | 'analytic'): forwarded to
    design_to_rcwa_stack. With 'li'/'fff', analytic shapes need lumenairy >= 5.50.1 (Li's rule on
    shapes layers -- 1-D y-invariant shapes / 2-D axis-aligned rectangles; other layers fall back
    to the raster under 'auto'); below it 'auto' rasterizes every inclusion layer and 'analytic'
    with an explicit 'li' is refused here, at construction. absorption=True needs
    layer_absorption() on analytic-shape layers, which lumenairy has from 5.50.1
    (FEATURE_FLOORS['rcwa_shapes_absorption']); below it 'auto' rasterizes every layer when
    absorption=True and 'analytic' is refused here too. cell_samples: None = the default raster
    lattice max(128, 8 N) per axis; an explicit value keeps the pre-0.13 formula
    max(cell_samples, 4 N + 1, 128) -- see design_to_rcwa_stack.

    The solver carries `cache_fingerprint` (every control that can change a number, the auto
    rule's version and the lumenairy version: OpticalSolverCache keys on it) and
    `native_carrier_grid = True` (a gridded field on a native non-uniform carrier z grid is
    consumed interval by interval through slice_layer_eps_field)."""
    where = "make_lumenairy_rcwa_solver"
    inclusion_model = _check_inclusion_model(inclusion_model, where)
    formulation = _check_formulation(formulation, where)
    raster_sampling = _check_raster_sampling(raster_sampling, where)
    if formulation != "auto":
        _check_model_formulation(inclusion_model, formulation, where)
    shapes_absorb = _lumenairy_has("rcwa_shapes_absorption")
    if absorption and inclusion_model == "analytic" and not shapes_absorb:
        raise ValueError(
            "make_lumenairy_rcwa_solver: absorption=True needs RCWAResult.layer_absorption() on "
            "analytic-shape layers, which lumenairy provides from 5.50.1; use "
            "inclusion_model='auto' (rasterizes every layer when absorption=True) or 'raster', "
            "or pip install -U lumenairy")
    if inclusion_model == "analytic":            # wave-7: refused at construction below 5.31
        _require_feature("rcwa_analytic_shapes",
                         "make_lumenairy_rcwa_solver(inclusion_model='analytic')")
    model_eff = "raster" if (absorption and not shapes_absorb) else inclusion_model
    advised = []                                 # the explicit-'laurent' advisory: once per solver

    def _advise(stack):
        msg = stack.dynameta_formulation_advice
        if msg is not None and not advised:
            advised.append(msg)
            warnings.warn(msg, AdvisoryWarning, stacklevel=4)

    def _solve_at(design, eps_by_region, lambda_m):
        t0 = time.perf_counter()
        theta, phi = _angles_rad(design.optical)
        pol = getattr(design.optical, "polarization", "y")
        _guard_incidence_side(design.optical)
        stack, names = _build_rcwa_stack(
            design, lambda_m, eps_by_region=eps_by_region, n_orders=n_orders,
            n_orders_y=n_orders_y, n_slices=n_slices, cell_samples=cell_samples,
            formulation=formulation, inclusion_model=model_eff,
            raster_sampling=raster_sampling, where=where)
        _advise(stack)
        _solve.last_formulation = stack.dynameta_formulation
        _solve.last_formulation_reason = stack.dynameta_formulation_reason
        stack.set_source(lambda_m, theta=theta, phi=phi)
        res = stack.solve(retain_internal=absorption, stabilize=stabilize)
        n_sup, n_sb = stack.dynameta_end_media      # audit S4-17: set by design_to_rcwa_stack
        rf, tf = _p_basis_conversion(pol, theta, n_sup, n_sb)
        row = _pol_row(design.optical)
        split = None
        if absorption and stack.dynameta_formulation == "li" and abs(phi) <= 1e-12:
            split = _face_flux_split(res, stack, row, theta, phi, n_sup)
        return rcwa_result_to_optical_result(res, row, t0=t0,
                                             layer_names=names, absorption=absorption,
                                             r_factor=rf, t_factor=tf,
                                             conical=(pol, theta, phi, n_sup),
                                             absorption_split=split)

    def _solve(design, geo, eps_by_region, lambda_m, n_super, n_sub):
        _require_eps_convention(eps_by_region, where)   # refuses an exp(+i omega t) field
        return _solve_at(design, eps_by_region, lambda_m)

    def _solve_sweep(design, geo, assemble_at, lams, n_super, n_sub):
        eps_list = [assemble_at(lam) for lam in lams]
        workers = (min(os.cpu_count() or 1, len(lams)) if n_workers is None
                   else max(1, int(n_workers)))
        if workers == 1 or len(lams) < 2:
            return [_solve_at(design, e, lam) for lam, e in zip(lams, eps_list)]
        import contextlib
        from concurrent.futures import ThreadPoolExecutor

        def _one(item):
            lam, e = item
            return _solve_at(design, e, lam)

        # ONE PROCESS-GLOBAL CAP AROUND THE POOL -- not lumenairy's rcwa_blas_threads, and not
        # one cap per worker. Both of those are wrong here, for DIFFERENT reasons, and the
        # 2026-08-31 bridge audit (vs lumenairy 5.42.1) measured both:
        #   * PER WORKER (what this used to do, citing a "thread-local, so each worker pins its
        #     own" rationale): lumenairy's audit M4 REFUTED that rationale and its own
        #     _blas_threads_quiet now says the opposite outright -- "Applying a cap is
        #     process-global on OpenBLAS, so per-worker enter/exit pairs race". Measured on a
        #     24-core box at n_orders=21: nondeterministic last-bit divergence, 1 run in 12,
        #     max|dT| 5.5e-14.
        #   * rcwa_blas_threads AROUND THE POOL: inert. It only sets a THREAD-LOCAL request
        #     (_BLAS_STATE = threading.local()); the apply happens later inside each solve via
        #     _blas_limit(), which reads the CALLING thread's request -- so a worker sees None
        #     and runs uncapped while the serial path runs capped. Measured: the two paths then
        #     differ systematically (rel ~1e-14 on R), which is exactly what the repaired
        #     threaded-sweep test caught when this was tried.
        # threadpool_limits caps the pool PROCESS-GLOBALLY for the whole parallel section with
        # a single set/restore: every worker is covered, and there is nothing to race. Absent
        # threadpoolctl the cap is inert -- the same graceful degradation lumenairy documents.
        try:
            from threadpoolctl import threadpool_limits
            cap = threadpool_limits(limits=max(1, int(blas_per_worker)), user_api="blas")
        except ImportError:                        # optional dep; cap inert, solve unaffected
            cap = contextlib.nullcontext()
        with cap:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                return list(pool.map(_one, list(zip(lams, eps_list))))

    _solve.solve_sweep = _solve_sweep
    _solve.last_formulation = None
    _solve.last_formulation_reason = None
    lv = _lumenairy_version()
    _solve.cache_fingerprint = (
        "rcwa(n_orders={},n_orders_y={},n_slices={},cell_samples={},formulation={},"
        "inclusion_model={},raster_sampling={},absorption={},stabilize={},rule={},"
        "lumenairy={})".format(
            n_orders, n_orders_y, n_slices, cell_samples, formulation, inclusion_model,
            raster_sampling, bool(absorption), stabilize, AUTO_RULE_VERSION,
            ".".join(str(p) for p in lv) if lv is not None else None))
    # consumes a non-uniform native z grid interval by interval (slice_layer_eps_field)
    _solve.native_carrier_grid = True
    return _solve


class LumenairyStackSolver:
    """LayeredStackSolver backed by Lumenairy RCWA -- the SECOND concrete implementation of
    the layered seam (TmmLayeredSolver is the first) and the first consumer of the
    structured LayeredSlab specs (eps_cell / eps_tensor_cell / shapes), which were stubbed in
    the data model awaiting exactly this backend. A `shapes` slab carries the lumenairy
    analytic-shape payload VERBATIM (a list of {'shape', 'eps', geometry, 'center'} dicts in
    metres, absolute cell frame, + eps_background -- the contract pinned in _shapes; produced
    from a Design layer by _shapes.layer_to_shapes_slab). Wired 2026-11-02; it used to raise.

    formulation (v0.16): 'auto' (default) | 'laurent' | 'li' | 'fff', passed EXPLICITLY on every
    add_layer call (v0.15.1 passed none and so inherited lumenairy's default, which lumenairy 5.51
    changes to 'auto'). A structured LayeredStack is solved on a 2-D lattice, where the 'auto' rule
    is Laurent's (see _auto_formulation), so the default reproduces v0.15.1 bit for bit; 'li'
    applies Li's rule to every eps_cell / shapes slab (lumenairy refuses a shapes list it cannot
    take under 'li': anything but axis-aligned rectangles, or lumenairy < 5.50.1).
    `last_formulation` records what ran."""

    def __init__(self, *, n_orders: int = 11, n_orders_y: Optional[int] = None,
                 stabilize: bool = False, formulation: str = "auto"):
        self.n_orders = int(n_orders)
        self.n_orders_y = n_orders_y
        self.stabilize = bool(stabilize)
        self.formulation = _check_formulation(formulation, "LumenairyStackSolver")
        self.last_formulation = None

    def _resolve(self, stack: LayeredStack):
        """(resolved, keyword): a structured LayeredStack is solved on a 2-D lattice, where the
        'auto' rule is Laurent's (_auto_formulation: lumenairy's 2-D Li rule was measured off a
        converged reference), i.e. what v0.15.1 ran."""
        if self.formulation == "auto":
            return "laurent", "laurent"
        return ("laurent" if self.formulation == "laurent" else "li"), self.formulation

    def solve(self, stack: LayeredStack, lambda_m: float, optical) -> OpticalResult:
        lum = _require_lumenairy()
        t0 = time.perf_counter()
        _guard_incidence_side(optical)
        structured = not stack.is_unstructured
        # audit C5-8: the lambda-sized fallback period is only harmless when the stack is
        # laterally UNIFORM (0-order-only physics); a STRUCTURED stack solved at a fabricated
        # period gets wrong diffraction geometry silently (probe: R=0.061 vs 0.191 correct)
        if structured and (float(stack.period_x_m or 0.0) <= 0.0
                           or float(stack.period_y_m or 0.0) <= 0.0):
            raise ValueError(
                "LumenairyStackSolver: the stack holds structured (eps_cell/eps_tensor_cell) "
                "slabs but period_x_m/period_y_m are unset (LayeredStack defaults 0.0; the "
                "slicers do not populate them) -- set the real lattice periods on the "
                "LayeredStack (audit C5-8; the old code silently substituted px=py=lambda).")
        _refuse_gain_half_space("LumenairyStackSolver", stack.n_super, stack.n_sub)
        resolved, f_lum = self._resolve(stack)
        self.last_formulation = resolved
        px = stack.period_x_m or float(lambda_m)          # period is irrelevant when uniform
        py = stack.period_y_m or float(lambda_m)
        nox = self.n_orders
        noy = int(self.n_orders_y if self.n_orders_y is not None else self.n_orders)
        if structured and all(
                slab.eps is not None or (slab.eps_tensor_cell is not None
                                         and is_uniform_cell(slab.eps_tensor_cell))
                for slab in stack.slabs):
            nox = noy = 1        # audit S4-12: laterally uniform tensor slabs only (see design_to_rcwa_stack)
        if structured:
            rs = lum.RCWAStack(px, period_y=py, n_superstrate=complex(stack.n_super),
                               n_substrate=complex(stack.n_sub), n_orders=nox, n_orders_y=noy)
        else:
            rs = lum.RCWAStack(px, n_superstrate=complex(stack.n_super),
                               n_substrate=complex(stack.n_sub), n_orders=self.n_orders)
        s_min_x = _min_samples(rs.n_orders_x)
        s_min_y = _min_samples(rs.n_orders_y) if structured else 1
        for slab in stack.slabs:                          # already superstrate-side first
            if slab.eps is not None:
                rs.add_layer(slab.thickness_m, eps=complex(slab.eps), formulation="laurent")
            elif slab.eps_cell is not None:
                cell = _meet_sampling(slab.eps_cell, s_min_x, s_min_y)
                rs.add_layer(slab.thickness_m, eps_cell=cell, formulation=f_lum)
            elif slab.eps_tensor_cell is not None:
                rs.add_layer(slab.thickness_m,
                             eps_tensor_cell=_meet_sampling(slab.eps_tensor_cell,
                                                            s_min_x, s_min_y),
                             formulation="laurent")
            else:
                _require_feature("rcwa_analytic_shapes", "LumenairyStackSolver (shapes slab)")
                # analytic-shape slab (2026-11-02): the payload is lumenairy's own contract, so
                # it passes through; lumenairy's validator rejects an overlap / over-period
                # extent / malformed dict with a named ValueError before anything is solved; an
                # explicit 'li' is passed as asked (lumenairy refuses a shapes list its Li rule
                # cannot take: anything but axis-aligned rectangles, or lumenairy < 5.50.1)
                rs.add_layer(slab.thickness_m, shapes=list(slab.shapes),
                             eps_background=complex(slab.eps_background), formulation=f_lum)
        theta, phi = _angles_rad(optical)
        pol = getattr(optical, "polarization", "y")
        rs.set_source(lambda_m, theta=theta, phi=phi)
        res = rs.solve(stabilize=self.stabilize)
        rf, tf = _p_basis_conversion(pol, theta, stack.n_super, stack.n_sub)
        return rcwa_result_to_optical_result(res, _pol_row(optical), t0=t0,
                                             r_factor=rf, t_factor=tf,
                                             conical=(pol, theta, phi, stack.n_super))
