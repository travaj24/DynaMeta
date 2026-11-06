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
the raster accepts. FORMULATION RULE (2026-11-02; relaxed for lumenairy >= 5.50.1, below):
lumenairy <= 5.50.0 shapes layers are Laurent by contract (add_layer raises for
formulation='li' with shapes=), so there 'auto' uses them ONLY with formulation='laurent' and
rasterizes every inclusion layer under 'li'/'fff' (honouring the rule the caller asked for --
the convergent one for TM on metal walls); 'analytic' + a non-Laurent formulation raises
ValueError rather than silently solving Laurent. RASTER LATTICE (2026-11-02):
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
from dynameta.core.layered import (LayeredStack, collapse_regions_to_layers,
                                   slice_eps_field)
from dynameta.optics.rasterize import cell_axes, layer_eps_tensor_cell
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
                                                      p_basis_conversion as _p_basis_conversion,
                                                      pol_row as _pol_row,
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


def _check_inclusion_model(inclusion_model: str, where: str) -> str:
    if str(inclusion_model) not in _INCLUSION_MODELS:
        raise ValueError("{}: inclusion_model must be one of {}, got {!r}".format(
            where, _INCLUSION_MODELS, inclusion_model))
    return str(inclusion_model)


def _check_model_formulation(inclusion_model: str, formulation: str, where: str) -> bool:
    """True when analytic shapes may be used for this (inclusion_model, formulation) pair.

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
                      formulation="laurent"):
    """Try to add design layer L as an ANALYTIC-shape RCWA layer. True on success; False when
    the layer is not exactly representable and inclusion_model == 'auto' (the caller
    rasterizes); ShapesNotRepresentable when inclusion_model == 'analytic'. lumenairy's add_layer
    validator (disjointness, extent <= period, cumulative area <= 1) is the single authority on
    what the additive form factors can paint; it raises BEFORE appending, so a refused layer
    leaves the stack untouched. A non-Laurent `formulation` is forwarded (lumenairy >= 5.50.1,
    _check_model_formulation); the Laurent call is the pre-5.50.1 call verbatim."""
    from dynameta.optics.rasterize import layer_bg_eps
    try:
        bg = layer_bg_eps(L, lambda_m, design.materials, eps_by_region)
        bg, shapes = layer_shapes_payload(L, lambda_m, design.materials, bg, px, py,
                                          stack_period_y_m=float(stack.period_y))
        kw = {} if str(formulation).lower() == "laurent" else {"formulation": formulation}
        try:
            stack.add_layer(L.thickness_m, shapes=shapes, eps_background=bg, **kw)
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


def design_to_rcwa_stack(design, lambda_m: float, *, eps_by_region=None, n_orders: int = 11,
                         n_orders_y: Optional[int] = None, n_slices: Optional[int] = None,
                         cell_samples: Optional[int] = None, formulation: str = "laurent",
                         inclusion_model: str = "auto"):
    """Translate a DynaMeta Design (+ optionally the bridge's eps_by_region) into a CONCRETE
    Lumenairy RCWAStack at one wavelength. Returns (stack, layer_names_top_first) where the
    name list is aligned with the Lumenairy layers (graded fields contribute several slabs
    per design layer) -- the key for mapping per-layer absorption back to design layers.

    inclusion_model: 'auto' (default since v0.13.0: analytic where exact -- and, with
    'li'/'fff', where lumenairy >= 5.50.1 takes Li's rule on that layer -- raster otherwise),
    'raster' (the periodic point raster, the pre-0.13 default) or 'analytic' (lumenairy analytic
    shapes for EVERY inclusion layer, raises ShapesNotRepresentable otherwise; ValueError with a
    non-Laurent formulation below lumenairy 5.50.1) -- see the module docstring. The requested
    factorization rule is honoured on every route, never dropped.

    cell_samples (the raster lattice, per patterned axis): None (default) = max(128, 8 N), N the
    axis' n_orders (2026-11-02, _default_raster_samples -- byte-identical to the pre-0.13
    max(128, 4 N + 1) for N <= 16); an explicit value = max(cell_samples, 4 N + 1, 128), the
    pre-0.13 formula unchanged, so inclusion_model='raster', cell_samples=128 reproduces the
    pre-0.13 raster numbers at every n_orders.

    Layer dispatch, in order: a structured (gridded) EpsField -> sliced eps_cell /
    eps_tensor_cell slabs; a uniform tensor EpsField -> a tiled eps_tensor_cell (on a layer WITH
    inclusions: the tensor is the BACKGROUND and the inclusions are painted over it as eps * I3,
    optics.rasterize.layer_eps_tensor_cell -- 2026-11-02, they used to be dropped); a uniform
    scalar EpsField -> eps; a graded laterally-uniform EpsField -> sliced uniform slabs;
    inclusions -> a rasterized eps_cell; else the material eps at lambda. Design layers are
    bottom -> top, the RCWA stack wants superstrate-side first -> reversed()."""
    require_supported_geometry(design, "design_to_rcwa_stack")   # 2026-11-02: refuse Stack.features / Inclusion.taper
    lum = _require_lumenairy()
    n_super, n_sub = end_media_indices(design, lambda_m)
    px = float(design.unit_cell.period_x_m)
    py = float(design.unit_cell.period_y_m)
    # mesh-region-keyed dicts (the run_pipeline/FEM bridge output: 'ito_inpatch',
    # 'grating__incl0', ...) collapse to design-layer keys; identity when already layer-keyed
    eps_by_region = collapse_regions_to_layers(design, eps_by_region or {})
    inclusion_model = _check_inclusion_model(inclusion_model, "design_to_rcwa_stack")
    use_shapes = _check_model_formulation(inclusion_model, formulation, "design_to_rcwa_stack")
    if use_shapes and not _lumenairy_has("rcwa_analytic_shapes"):
        # wave-7 (2026-11-02): below lumenairy 5.31 add_layer(shapes=) exists but does NOT refuse
        # overlapping shapes (measured on 5.22.0: two overlapping disks accepted, form factors
        # ADDED), so the "lumenairy refused -> rasterize" decision below would silently build a
        # wrong layer. 'auto' rasterizes every inclusion layer there; 'analytic' is refused.
        if inclusion_model == "analytic":
            _require_feature("rcwa_analytic_shapes",
                             "design_to_rcwa_stack(inclusion_model='analytic')")
        use_shapes = False

    def _y_invariant(ef):
        v = ef.values_zyx
        return v is None or v.shape[1] == 1 or bool(
            np.allclose(v, v[:, :1], rtol=1e-12, atol=0.0))

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
        utensor = ef is not None and getattr(ef, "is_tensor", False)
        return (bool(L.inclusions) and not _lamellar(L)) or grid_2d or utensor

    def _laterally_uniform(L):
        if L.inclusions:
            return False
        ef = eps_by_region.get(L.name)
        if ef is None or getattr(ef, "is_uniform", True):
            return True
        v = ef.values_zyx
        return v is None or bool(np.allclose(v, v[:, :1, :1], rtol=1e-12, atol=0.0))

    structured = any(_structured(L) for L in design.stack.layers)
    is_2d = any(_needs_2d(L) for L in design.stack.layers)
    if is_2d and all(_laterally_uniform(L) for L in design.stack.layers):
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
    s_min_x = _min_samples(nx_o)
    s_min_y = _min_samples(ny_o) if is_2d else 1
    # raster lattice (2026-11-02): default S = max(128, 8 N) per patterned axis
    # (_default_raster_samples -- measured there); an EXPLICIT cell_samples keeps the pre-0.13
    # formula max(cell_samples, 4 N + 1, 128) exactly, so cell_samples=128 reproduces the
    # pre-0.13 raster lattice (max(128, 4 N + 1)) at every n_orders
    if cell_samples is None:
        sx = _default_raster_samples(nx_o) if structured else s_min_x
        sy = _default_raster_samples(ny_o) if is_2d else 1
    else:
        sx = max(int(cell_samples), s_min_x, _RASTER_FLOOR if structured else s_min_x)
        sy = max(int(cell_samples), s_min_y, _RASTER_FLOOR) if is_2d else 1

    names: List[str] = []
    for L in reversed(design.stack.layers):              # superstrate side first
        ef = eps_by_region.get(L.name)
        if ef is not None and not getattr(ef, "is_uniform", True):
            # gridded EpsField (carrier/effect-graded; axes in nm solver units like the
            # TMM extractor) -> slice_eps_field returns slabs in ASCENDING z
            # (substrate-side first); this stack is superstrate-first, so reverse
            # (audit C5-1: unreversed slabs vertically flip the graded profile)
            for slab in reversed(slice_eps_field(ef, 1.0 / _S_NM, n_slices=n_slices)):
                if slab.eps is not None:
                    stack.add_layer(slab.thickness_m, eps=complex(slab.eps))
                elif slab.eps_cell is not None:
                    cell = np.asarray(slab.eps_cell, dtype=complex)
                    if not is_2d and cell.shape[1] > 1:
                        cell = cell[:, :1]                # y-invariant grid on a 1-D stack
                    stack.add_layer(slab.thickness_m,
                                    eps_cell=_meet_sampling(cell, s_min_x, s_min_y),
                                    formulation=formulation)
                else:
                    cell = np.asarray(slab.eps_tensor_cell, dtype=complex)
                    if not is_2d and cell.shape[1] > 1:
                        cell = cell[:, :1]
                    stack.add_layer(slab.thickness_m,
                                    eps_tensor_cell=_meet_sampling(cell, s_min_x, s_min_y))
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
                x, y = cell_axes(sx, sy, px, py)
                X, Y = np.meshgrid(x, y, indexing="ij")
                cell = layer_eps_tensor_cell(L, X, Y, lambda_m, design.materials, ef.tensor,
                                             period_x_m=px, period_y_m=py)
                stack.add_layer(L.thickness_m, eps_tensor_cell=cell)
                names.append(L.name)
                continue
            cell = np.broadcast_to(np.asarray(ef.tensor, dtype=complex),
                                   (s_min_x, max(s_min_y, 1), 3, 3)).copy()
            stack.add_layer(L.thickness_m, eps_tensor_cell=cell)
            names.append(L.name)
            continue
        if L.inclusions:
            if use_shapes and _add_shapes_layer(
                    stack, L, design, lambda_m, eps_by_region, px, py, inclusion_model,
                    formulation):
                names.append(L.name)
                continue
            x, y = cell_axes(sx, sy, px, py)
            X, Y = np.meshgrid(x, y, indexing="ij")
            cell = periodic_layer_eps_cell(L, X, Y, lambda_m, design.materials, eps_by_region,
                                           px, py)
            stack.add_layer(L.thickness_m, eps_cell=np.asarray(cell, dtype=complex),
                            formulation=formulation)
            names.append(L.name)
            continue
        if ef is not None:                               # uniform scalar (effect-modulated)
            stack.add_layer(L.thickness_m, eps=complex(ef.scalar))
        else:
            stack.add_layer(L.thickness_m,
                            eps=complex(design.materials.get(L.background_material)
                                        .eps(lambda_m)))
        names.append(L.name)
    return stack, names


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


def rcwa_result_to_optical_result(res, row: int, *, t0: float,
                                  layer_names: Optional[List[str]] = None,
                                  absorption: bool = False,
                                  r_factor: complex = 1.0,
                                  t_factor: complex = 1.0,
                                  conical=None) -> OpticalResult:
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
    whose cross term it does not expose (A = 1 - R - T stays exact)."""
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
                               formulation: str = "laurent",
                               n_workers: Optional[int] = 1, blas_per_worker: int = 1,
                               inclusion_model: str = "auto"):
    """Build an `optical_solver` for run_pipeline backed by Lumenairy RCWA, with the exact
    seam signature fn(design, geo, eps_by_region, lambda_m, n_super, n_sub) -> OpticalResult
    PLUS the sweep-aware `solve_sweep` fast path (one result per wavelength, in order; end
    media re-derived PER WAVELENGTH, fixing the generic band-centre freeze). geo and the
    passed n_super/n_sub are accepted for seam compatibility but unused (the stack re-derives
    end media from the Design, the make_layered_tmm_solver precedent).

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
    is refused here, at construction. absorption=True needs layer_absorption() on analytic-shape
    layers, which lumenairy has from 5.50.1 (FEATURE_FLOORS['rcwa_shapes_absorption']); below it
    'auto' rasterizes every layer when absorption=True and 'analytic' is refused here too.
    cell_samples: None = the default raster lattice max(128, 8 N) per axis; an explicit value
    keeps the pre-0.13 formula max(cell_samples, 4 N + 1, 128) -- see design_to_rcwa_stack."""
    inclusion_model = _check_inclusion_model(inclusion_model, "make_lumenairy_rcwa_solver")
    _check_model_formulation(inclusion_model, formulation, "make_lumenairy_rcwa_solver")
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

    def _solve_at(design, eps_by_region, lambda_m):
        t0 = time.perf_counter()
        theta, phi = _angles_rad(design.optical)
        pol = getattr(design.optical, "polarization", "y")
        _guard_incidence_side(design.optical)
        stack, names = design_to_rcwa_stack(design, lambda_m, eps_by_region=eps_by_region,
                                            n_orders=n_orders, n_orders_y=n_orders_y,
                                            n_slices=n_slices, cell_samples=cell_samples,
                                            formulation=formulation,
                                            inclusion_model=model_eff)
        stack.set_source(lambda_m, theta=theta, phi=phi)
        res = stack.solve(retain_internal=absorption, stabilize=stabilize)
        n_sup, n_sb = stack.dynameta_end_media      # audit S4-17: set by design_to_rcwa_stack
        rf, tf = _p_basis_conversion(pol, theta, n_sup, n_sb)
        return rcwa_result_to_optical_result(res, _pol_row(design.optical), t0=t0,
                                             layer_names=names, absorption=absorption,
                                             r_factor=rf, t_factor=tf,
                                             conical=(pol, theta, phi, n_sup))

    def _solve(design, geo, eps_by_region, lambda_m, n_super, n_sub):
        _require_eps_convention(eps_by_region, "make_lumenairy_rcwa_solver")   # audit V-5
        return _solve_at(design, eps_by_region, lambda_m)

    def _solve_sweep(design, geo, assemble_at, lams, n_super, n_sub):
        workers = (min(os.cpu_count() or 1, len(lams)) if n_workers is None
                   else max(1, int(n_workers)))
        if workers == 1 or len(lams) < 2:
            return [_solve_at(design, assemble_at(lam), lam) for lam in lams]
        import contextlib
        from concurrent.futures import ThreadPoolExecutor

        def _one(lam):
            return _solve_at(design, assemble_at(lam), lam)

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
                return list(pool.map(_one, lams))

    _solve.solve_sweep = _solve_sweep
    return _solve


class LumenairyStackSolver:
    """LayeredStackSolver backed by Lumenairy RCWA -- the SECOND concrete implementation of
    the layered seam (TmmLayeredSolver is the first) and the first consumer of the
    structured LayeredSlab specs (eps_cell / eps_tensor_cell / shapes), which were stubbed in
    the data model awaiting exactly this backend. A `shapes` slab carries the lumenairy
    analytic-shape payload VERBATIM (a list of {'shape', 'eps', geometry, 'center'} dicts in
    metres, absolute cell frame, + eps_background -- the contract pinned in _shapes; produced
    from a Design layer by _shapes.layer_to_shapes_slab). Wired 2026-11-02; it used to raise."""

    def __init__(self, *, n_orders: int = 11, n_orders_y: Optional[int] = None,
                 stabilize: bool = False):
        self.n_orders = int(n_orders)
        self.n_orders_y = n_orders_y
        self.stabilize = bool(stabilize)

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
        px = stack.period_x_m or float(lambda_m)          # period is irrelevant when uniform
        py = stack.period_y_m or float(lambda_m)
        nox = self.n_orders
        noy = int(self.n_orders_y if self.n_orders_y is not None else self.n_orders)
        if structured and all(
                slab.eps is not None or (slab.eps_tensor_cell is not None and bool(np.all(
                    np.asarray(slab.eps_tensor_cell) == np.asarray(slab.eps_tensor_cell)[:1, :1])))
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
                rs.add_layer(slab.thickness_m, eps=complex(slab.eps))
            elif slab.eps_cell is not None:
                rs.add_layer(slab.thickness_m,
                             eps_cell=_meet_sampling(slab.eps_cell, s_min_x, s_min_y))
            elif slab.eps_tensor_cell is not None:
                rs.add_layer(slab.thickness_m,
                             eps_tensor_cell=_meet_sampling(slab.eps_tensor_cell,
                                                            s_min_x, s_min_y))
            else:
                _require_feature("rcwa_analytic_shapes", "LumenairyStackSolver (shapes slab)")
                # analytic-shape slab (2026-11-02): the payload is lumenairy's own contract, so
                # it passes through; lumenairy's validator rejects an overlap / over-period
                # extent / malformed dict with a named ValueError before anything is solved
                rs.add_layer(slab.thickness_m, shapes=list(slab.shapes),
                             eps_background=complex(slab.eps_background))
        theta, phi = _angles_rad(optical)
        pol = getattr(optical, "polarization", "y")
        rs.set_source(lambda_m, theta=theta, phi=phi)
        res = rs.solve(stabilize=self.stabilize)
        rf, tf = _p_basis_conversion(pol, theta, stack.n_super, stack.n_sub)
        return rcwa_result_to_optical_result(res, _pol_row(optical), t0=t0,
                                             r_factor=rf, t_factor=tf,
                                             conical=(pol, theta, phi, stack.n_super))
