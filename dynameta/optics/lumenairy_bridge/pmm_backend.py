"""Lumenairy PMM as a DynaMeta optical backend (roadmap v0.5 A2).

PMM's roles in DynaMeta (complementing the RCWA backend, not replacing it):
- CONVERGENCE REFEREE: the 1-D PMM family is a subsectional spectral element with NO
  Fourier-factorization accuracy floor -- the referee for RCWA truncation settings on hard
  (metallic TM) 1-D cells. (The no-floor claim is 1-D-only; Lumenairy's 2-D PMM is itself
  Fourier-floored, so this bridge is deliberately 1-D.)
- TENSOR SPECIALIST: pmm_jones_1d / PMMStack carry FULL (3,3) tensors including
  out-of-plane (exz/eyz/ezx/ezy) couplings -- gyrotropic / slanted-LC cells that the RCWA
  in-plane tensor path cannot represent.

Scope contract (enforced loudly): 1-D devices only -- every patterned layer's inclusions must
be Rectangles spanning the FULL y period (a lamellar grating; a line crossing the cell edge
wraps into two segments), translated into PMM segments [(width_fraction, eps)].

GRIDDED EpsFields (a carrier-modulated layer). Each is first cut into z slabs by the shared
slab-thickness guard (core.layered.slice_layer_eps_field: one slab per native z interval, so a
NON-UNIFORM native carrier grid is consumed as is; the solver therefore sets
native_carrier_grid = True). Then:
  * z-graded, laterally uniform field: one uniform PMM layer per slab. No lateral averaging is
    involved even under a patterned layer -- this is exact up to the z staircase (unchanged
    from v0.15.1, bit for bit).
  * laterally structured field (y-invariant, varying along x: the film under a strip differs
    from the film between strips; v0.16): each slab's lateral profile is averaged onto one
    ADAPTIVE PARTITION per layer (_lateral_partition.partition_edges), seeded with every wall
    the other layers own and refined until each interval's relative spread is within
    `partition_rtol`, as a laminate tensor diag(harm, arith, arith) per interval (the field
    across the variation sees the series mean) for an interval narrower than the layer and the
    plain mean for a wider one (in a thin film E_x is pinned by its neighbours); each native z
    interval is itself a z-laminate (E_z sees the series mean along z). Consecutive slabs that
    are nearly identical are merged (`z_merge_rtol`) as a z-laminate diag(arith, arith, harm), which keeps the solve
    affordable on a fine native grid (each PMM layer pays an eigensolve on the union grid).
    `stack.dynameta_partition_report` records what each layer's partition left behind.
  * laterally uniform gridded TENSOR field (v0.16; e.g. a z-graded uniaxial film): one uniform
    tensor PMM layer per slab (the endpoint average); a diagonal tensor also takes
    z_average='laminate' and z_merge_rtol (_uniform_tensor_layer_plan). A uniform (3, 3) tensor
    EpsField (e.g. the equivalent layer of optics.quantum_film_pipeline) is one PMM layer.
  Laterally STRUCTURED gridded tensor fields and fields varying along y still raise (use the RCWA
  bridge).
[audit S4-13, wave-7 2026-11-02: this paragraph used to say "In-plane incidence only
(azimuth = 0; PMMStack has no conical source). PMM returns NO transmission Jones and no
per-order complex amplitudes: OpticalResult.t is None" -- all three false since lumenairy 5.22,
whose PMMStack gained set_source(phi=), jones_transmission() and per_order_amplitudes().]
Incidence: in-plane AND conical (azimuth != 0; the rotated s/p R/T/r/t are synthesized from
the per-order complex amplitudes by the shared _common.conical_synthesis, lossless end media
only -- see make_lumenairy_pmm_solver). r = the zeroth-order co-pol reflection Jones, t = the
zeroth-order co-pol TRANSMISSION Jones (PMMStack.jones_transmission), both with the same
lab-basis -> Byrnes p-pol conversion as the RCWA bridge at phi = 0; R/T = the order-summed
efficiencies (gated in validation/lumenairy_pmm_bridge.py). absorption=True fills the per-layer
split at phi = 0 only (audit S4-2).

Cross-library pins (first pinned at lumenairy 5.14.2, re-verified on the 5.22 floor):
PMMStack(period, n_substrate=, n_superstrate= [INDICES], degree=, far_field_orders=[, layer_grids=
>= 5.32][, **pmm_kwargs]) -- far_field_orders is the far-field order COUNT (the `n_orders` alias
of it is deprecated upstream); it does NOT set the accuracy, which is controlled by `degree`
(and the element controls in pmm_kwargs); add_layer(thickness, segments=[(w_frac, EPS)] | eps=
[scalar or (3,3)], slant_angle=); set_source(wavelength, theta=, phi=); solve() ->
(orders, R_eff(2, M), T_eff(2, M), jones_reflection(2, 2)) with rows/columns keyed
INCIDENT E_x (0) / E_y (1) -- identical to the RCWA result layout. Dispersive callables
force solve_vs_wavelength; this bridge rebuilds CONCRETE stacks per wavelength instead
(same policy as the RCWA bridge)."""

from __future__ import annotations

from dynameta.core.eps_field import require_solver_time_convention as _require_eps_convention

import time
from typing import List, Optional, Tuple

import numpy as np

from dynameta.core.interfaces import OpticalResult
from dynameta.core.lateral import is_laterally_uniform, is_y_invariant
from dynameta.core.layered import collapse_regions_to_layers, slice_layer_eps_field
from dynameta.optics.lumenairy_bridge import _lateral_partition as _lp
from dynameta.optics.lumenairy_bridge._common import (FEATURE_FLOORS as _FEATURE_FLOORS,
                                                      LumenairyFeatureUnavailable,
                                                      angles_rad as _angles_rad,
                                                      conical_synthesis as _conical_synthesis,
                                                      guard_incidence_side as _guard_incidence_side,
                                                      lumenairy_has as _lumenairy_has,
                                                      p_basis_conversion as _p_basis_conversion,
                                                      pol_row as _pol_row,
                                                      refuse_gain_half_space as _refuse_gain_half_space,
                                                      require_lumenairy as _require_lumenairy,
                                                      require_pmm_gain_support as _require_pmm_gain_support,
                                                      require_lumenairy_feature as _require_feature)
from dynameta.optics.tmm_reference import S as _S_NM
from dynameta.optics.tmm_reference import end_media_indices
from dynameta.geometry.guards import require_supported_geometry
from dynameta.warning_types import AdvisoryWarning

__all__ = ["design_to_pmm_stack", "make_lumenairy_pmm_solver", "layer_to_pmm_segments"]

_REL_TOL = 1e-9

# pmm_kwargs: PMMStack keywords this bridge sets itself (refused in pmm_kwargs).
_PMM_RESERVED = ("period", "degree", "far_field_orders", "n_orders", "n_substrate",
                 "n_superstrate", "layer_grids")
# The lumenairy release that introduced a PMMStack keyword (for the error message when the installed
# release does not know it). Keywords present at the bridge floor (5.22.0): elements_per_region,
# grade (bool), factorization, min_feature.
_PMM_KW_SINCE = {"window_halfwidth": (5, 33, 0), "allow_small_elements": (5, 51, 0)}
# Values of an existing keyword that need a newer release: geometric element grading (feature row
# 'pmm_geometric_grade' of _common.FEATURE_FLOORS, DETECTED from the installed API). OLDER releases
# coerce `grade` with bool(), so grade='geometric' would be read as grade=True there without any
# error -- the reason this is checked here rather than left to lumenairy.

# z_merge_rtol='auto': no merge for a laterally uniform graded layer (bit-identical to v0.15.1),
# this tolerance for a laterally structured one (measured: docs/audit/2026-11-14-v0.16-pmm-bridge.md).
_Z_MERGE_AUTO_STRUCTURED = 3e-2


def _version_str(v) -> str:
    return ".".join(str(p) for p in v)


def _resolve_pmm_kwargs(lum, pmm_kwargs) -> dict:
    """Validate the pass-through PMMStack keywords against the INSTALLED lumenairy: refuse the
    keywords the bridge sets itself, refuse a keyword the installed PMMStack does not accept
    (naming the release that introduced it when known), and refuse a geometric `grade` on a
    release that would silently coerce it to bool."""
    if pmm_kwargs is None:
        return {}
    if not isinstance(pmm_kwargs, dict):
        raise TypeError("pmm_kwargs must be a dict of PMMStack keywords or None, got {!r}"
                        .format(type(pmm_kwargs).__name__))
    kw = dict(pmm_kwargs)
    bad = sorted(k for k in kw if k in _PMM_RESERVED)
    if bad:
        raise ValueError(
            "pmm_kwargs: {} set by the bridge itself (degree= and n_orders= are bridge keywords; "
            "n_orders is passed as PMMStack far_field_orders; the end media come from the "
            "Design; layer_grids is a bridge keyword) -- remove {} from pmm_kwargs".format(
                ", ".join(repr(k) for k in bad), "it" if len(bad) == 1 else "them"))
    import inspect
    try:
        params = inspect.signature(lum.PMMStack.__init__).parameters
    except (TypeError, ValueError):                         # pragma: no cover - defensive
        params = None
    installed = getattr(lum, "__version__", "?")
    if params is not None and not any(p.kind == p.VAR_KEYWORD for p in params.values()):
        unknown = sorted(k for k in kw if k not in params)
        if unknown:
            hints = []
            for k in unknown:
                since = _PMM_KW_SINCE.get(k)
                hints.append("{!r} (needs lumenairy >= {})".format(k, _version_str(since))
                             if since else repr(k))
            raise LumenairyFeatureUnavailable(
                "pmm_kwargs: PMMStack of the installed lumenairy {} does not accept {}; "
                "accepted keywords: {}".format(
                    installed, ", ".join(hints),
                    ", ".join(sorted(k for k in params if k != "self"
                                     and k not in _PMM_RESERVED))))
    if "grade" in kw and not isinstance(kw["grade"], (bool, np.bool_)):
        if not _lumenairy_has("pmm_geometric_grade"):
            raise LumenairyFeatureUnavailable(
                "pmm_kwargs: grade={!r} (geometric element grading) needs lumenairy >= {} "
                "(installed {}); this release only accepts grade=True/False and would read any "
                "other value as True".format(
                    kw["grade"], _version_str(_FEATURE_FLOORS["pmm_geometric_grade"][0]),
                    installed))
    return kw


def _require_gain_support(where: str, layer_name, layer: dict) -> None:
    """_common.require_pmm_gain_support on the add_layer keywords `layer` (eps= or segments=)."""
    values = ([e for _w, e in layer["segments"]] if "segments" in layer else [layer.get("eps")])
    _require_pmm_gain_support(where, values, layer_name=layer_name)


def _segment_walls(segs) -> List[float]:
    """Interior wall positions (period fractions) of a [(width_fraction, eps)] list."""
    return [float(x) for x in np.cumsum([w for w, _ in segs])[:-1]]


def _pmm_eps(c):
    """(xx, yy, zz) -> the PMM eps: the scalar when the three agree to 1e-14 (bit-for-bit the
    value PMMStack would promote), else the diagonal tensor."""
    c = np.asarray(c, dtype=np.complex128)
    a = complex(c[1])
    if abs(c[0] - a) <= 1e-14 * abs(a) and abs(c[2] - a) <= 1e-14 * abs(a):
        return a
    return np.diag(c)


def _z_harmonic_rows(L, ef, n_slices, z_average, lateral):
    """The per-native-interval harmonic z means of the field (ascending z) for
    z_average='laminate': (Nslab,) for a laterally uniform field (xy-mean column, as the slicer
    uses), (Nslab, Nx) for a y-invariant structured one; None for z_average='arithmetic'."""
    if z_average == "arithmetic":
        return None
    if n_slices is not None:
        raise ValueError("PMM bridge layer {!r}: z_average='laminate' integrates the field's own "
                         "z intervals; it cannot be combined with n_slices={!r} (pass "
                         "n_slices=None)".format(L.name, n_slices))
    v = np.asarray(ef.values_zyx, dtype=np.complex128)
    col = v[:, 0, :] if lateral else v.mean(axis=(1, 2))
    try:
        return _lp.z_interval_harmonic(col, ef.z_axis_u)
    except ValueError as exc:
        raise ValueError("PMM bridge layer {!r}: {} (z_average='laminate')".format(L.name, exc)) \
            from None


def _structured_layer_plan(L, ef, px, seeds, *, n_slices, lateral_partition, partition_rtol,
                           max_segments, lateral_average, z_merge_rtol, z_average):
    """PMM layers (bottom first) for a laterally structured gridded field: [(thickness,
    [(w, eps)])], the partition edges, and the report entry."""
    v = np.asarray(ef.values_zyx)
    if v.shape[1] > 1 and not is_y_invariant(v):
        raise ValueError("PMM bridge: layer {!r} carries a gridded EpsField that varies along y; "
                         "the 1-D PMM needs y-invariant lines -- use the RCWA bridge".format(L.name))
    slabs = slice_layer_eps_field(ef, L, 1.0 / _S_NM, n_slices=n_slices)    # ascending z
    nx = v.shape[2]
    prof = np.empty((len(slabs), nx), dtype=np.complex128)
    for k, s in enumerate(slabs):
        prof[k] = (np.full(nx, complex(s.eps)) if s.eps is not None
                   else np.asarray(s.eps_cell)[:, 0])
    if z_average == "auto":           # laminate on the field's own z intervals
        z_average = "laminate" if n_slices is None else "arithmetic"
    hz = _z_harmonic_rows(L, ef, n_slices, z_average, lateral=True)
    thick = np.array([s.thickness_m for s in slabs], dtype=np.float64)
    x_m = np.asarray(ef.x_axis_u, dtype=np.float64) / _S_NM
    where = "PMM bridge layer {!r}".format(L.name)
    rows = prof if hz is None else np.concatenate([prof, hz], axis=0)
    try:
        if isinstance(lateral_partition, str):
            edges, rep = _lp.partition_edges(rows, x_m, px, seeds, rtol=partition_rtol,
                                             max_segments=max_segments, refine=True)
        else:
            user = np.asarray(lateral_partition, dtype=np.float64).ravel()
            if user.size and (np.any(user < -1e-12) or np.any(user > 1.0 + 1e-12)):
                raise ValueError("lateral_partition edges must be period fractions in [0, 1]")
            edges, rep = _lp.partition_edges(rows, x_m, px, user, rtol=partition_rtol,
                                             max_segments=max_segments, refine=False)
        widths, comps = _lp.laminate_segments(prof, x_m, px, edges, lateral_average,
                                              layer_thickness_m=float(L.thickness_m),
                                              zz_profiles=hz)
    except ValueError as exc:
        raise ValueError("{}: {}".format(where, exc)) from None
    zr = _Z_MERGE_AUTO_STRUCTURED if z_merge_rtol == "auto" else z_merge_rtol
    groups = _lp.z_merge_groups(comps, thick, zr)
    t_out, c_out = _lp.laminate_z(comps, thick, groups)
    wsum = float(np.sum(widths))
    plan = [(float(t), [(float(w) / wsum, _pmm_eps(c)) for w, c in zip(widths, cs)])
            for t, cs in zip(t_out, c_out)]
    rep = dict(rep, n_slabs=len(slabs), n_layers=len(plan), z_merge_rtol=zr,
               lateral_average=lateral_average, z_average=z_average,
               edges=tuple(float(e) for e in edges))
    return plan, edges, rep


def _uniform_layer_plan(L, ef, *, n_slices, z_merge_rtol, z_average):
    """PMM layers (bottom first) for a laterally uniform graded field: [(thickness, eps)].
    z_merge_rtol None/'auto' with z_average='arithmetic' -> one scalar layer per slab (v0.15.1,
    bit for bit)."""
    slabs = slice_layer_eps_field(ef, L, 1.0 / _S_NM, n_slices=n_slices)    # ascending z
    z_average = "arithmetic" if z_average == "auto" else z_average
    hz = _z_harmonic_rows(L, ef, n_slices, z_average, lateral=False)
    if z_merge_rtol in (None, "auto") and hz is None:
        return [(s.thickness_m, complex(s.eps)) for s in slabs], None
    e = np.array([complex(s.eps) for s in slabs], dtype=np.complex128)
    thick = np.array([s.thickness_m for s in slabs], dtype=np.float64)
    comps = np.stack([e, e, e if hz is None else hz], axis=1)
    zr = None if z_merge_rtol == "auto" else z_merge_rtol
    groups = _lp.z_merge_groups(comps, thick, zr)
    t_out, c_out = _lp.laminate_z(comps, thick, groups)
    return ([(float(t), _pmm_eps(c)) for t, c in zip(t_out, c_out)],
            {"n_slabs": len(slabs), "n_layers": len(groups), "z_merge_rtol": zr,
             "z_average": z_average})


def _uniform_tensor_layer_plan(L, ef, *, n_slices, z_merge_rtol, z_average):
    """PMM layers (bottom first) for a laterally uniform GRIDDED TENSOR field (v0.16; refused
    before): [(thickness, eps)] with eps the (3, 3) slab tensor (or its scalar when the three
    diagonal components agree, as _pmm_eps does). One slab per native z interval (the
    slab-thickness guard, as for scalar fields), each the endpoint average of the tensor -- the
    tensor counterpart of the laterally uniform scalar staircase. A DIAGONAL tensor additionally
    accepts z_average='laminate' (zz the harmonic mean of the z-linear interpolant over each
    interval, xx and yy arithmetic) and z_merge_rtol (a z-laminate merge of near-identical slabs),
    with the same kernels as the structured scalar path; a tensor with off-diagonal components takes
    the arithmetic staircase only. Returns (plan, report or None)."""
    v = np.asarray(ef.values_zyx, dtype=np.complex128)                  # (Nz, Ny, Nx, 3, 3)
    slabs = slice_layer_eps_field(ef, L, 1.0 / _S_NM, n_slices=n_slices)    # ascending z
    cells = [np.asarray(s.eps_tensor_cell, dtype=np.complex128).reshape(-1, 3, 3) for s in slabs]
    tens = np.stack([c[0] for c in cells], axis=0)                       # (Nslab, 3, 3)
    scale = np.max(np.abs(np.diagonal(tens, axis1=1, axis2=2)))
    offd = tens - np.einsum("kii->ki", tens)[:, :, None] * np.eye(3)[None]
    diagonal = bool(np.max(np.abs(offd)) <= 1e-12 * max(scale, 1e-300))
    z_average = "arithmetic" if z_average == "auto" else z_average
    zr = None if z_merge_rtol == "auto" else z_merge_rtol
    if not diagonal:
        if z_average != "arithmetic" or zr not in (None, 0, 0.0):
            raise ValueError("PMM bridge layer {!r}: a gridded tensor with off-diagonal components "
                             "takes the arithmetic z staircase only (z_average='arithmetic', "
                             "z_merge_rtol=None); got z_average={!r}, z_merge_rtol={!r}".format(
                                 L.name, z_average, z_merge_rtol))
        return [(float(s.thickness_m), t.copy()) for s, t in zip(slabs, tens)], None
    comps = np.einsum("kii->ki", tens).copy()                            # (Nslab, 3): xx, yy, zz
    if z_average == "laminate":
        if n_slices is not None:
            raise ValueError("PMM bridge layer {!r}: z_average='laminate' integrates the field's own "
                             "z intervals; it cannot be combined with n_slices={!r} (pass "
                             "n_slices=None)".format(L.name, n_slices))
        zz_col = v[:, 0, 0, 2, 2]
        try:
            comps[:, 2] = _lp.z_interval_harmonic(zz_col, ef.z_axis_u)
        except ValueError as exc:
            raise ValueError("PMM bridge layer {!r}: {} (z_average='laminate')".format(L.name, exc)) \
                from None
    thick = np.array([s.thickness_m for s in slabs], dtype=np.float64)
    if zr is None and z_average == "arithmetic":
        return [(float(t), _pmm_eps(c)) for t, c in zip(thick, comps)], None
    groups = _lp.z_merge_groups(comps, thick, zr)
    t_out, c_out = _lp.laminate_z(comps, thick, groups)
    return ([(float(t), _pmm_eps(c)) for t, c in zip(t_out, c_out)],
            {"n_slabs": len(slabs), "n_layers": len(groups), "z_merge_rtol": zr,
             "z_average": z_average, "tensor": True})


def layer_to_pmm_segments(layer, design, lambda_m: float, period_x_m: float,
                          period_y_m: float, *, bg_eps=None) -> List[Tuple[float, complex]]:
    """Translate a lamellar (1-D) DynaMeta layer into PMM segments [(width_fraction, eps)].
    Every inclusion must be a Rectangle spanning the FULL y period (y-invariant grating
    lines), at most one period wide (a line crossing x = 0 or x = period_x WRAPS into two
    segments, 2026-11-02) and mutually non-overlapping in x; gaps are filled
    with the background eps. Anything else raises -- the honest 1-D scope."""
    px, py = float(period_x_m), float(period_y_m)
    bg = complex(bg_eps if bg_eps is not None
                 else design.materials.get(layer.background_material).eps(lambda_m))
    spans = []
    for inc in layer.inclusions:
        if getattr(inc.shape, "kind", "") != "rectangle":
            raise ValueError("PMM bridge: layer {!r} inclusion kind {!r} is not a lamellar "
                             "Rectangle (1-D scope); use the RCWA bridge".format(
                                 layer.name, getattr(inc.shape, "kind", "?")))
        xlo, xhi, ylo, yhi = inc.shape.bbox_m()
        if ylo > _REL_TOL * py or yhi < py * (1.0 - _REL_TOL):
            raise ValueError("PMM bridge: layer {!r} Rectangle does not span the full y "
                             "period (y-invariant grating lines required; got y in "
                             "[{:.3e}, {:.3e}] of {:.3e})".format(layer.name, ylo, yhi, py))
        eps_inc = complex(design.materials.get(inc.material).eps(lambda_m))
        if xlo < -_REL_TOL * px or xhi > px * (1.0 + _REL_TOL):
            # PERIODIC WRAP (2026-11-02; this used to raise "wrap-around is a follow-on"): a
            # boundary-crossing line is the SAME periodic structure as its two in-cell pieces
            # [xlo', Px] + [0, xhi' - Px] after shifting xlo into [0, Px) -- the convention of the
            # FEM mesher (_inclusion_solids_clipped) and of the RCWA bridge's periodic raster /
            # analytic shapes. A line WIDER than the period would overlap its own image, which
            # no segment list can paint -> refused; exactly one period wide = the whole cell.
            # Gate (tests/test_lumenairy_shapes_sem_wrap.py): shifting every line by the same
            # dx leaves R/T unchanged to ~1e-13 and agrees with the RCWA analytic path.
            width = xhi - xlo
            if width > px * (1.0 + _REL_TOL):
                raise ValueError("PMM bridge: layer {!r} Rectangle x-width {:.3e} exceeds the "
                                 "period {:.3e} -- it would overlap its own periodic image"
                                 .format(layer.name, width, px))
            if width >= px * (1.0 - _REL_TOL):
                spans.append((0.0, px, eps_inc))
                continue
            lo = xlo - np.floor(xlo / px) * px          # in [0, Px)
            hi = lo + width
            if hi <= px * (1.0 + _REL_TOL):
                spans.append((lo, min(hi, px), eps_inc))
            else:
                if px - lo > _REL_TOL * px:
                    spans.append((lo, px, eps_inc))
                spans.append((0.0, hi - px, eps_inc))
            continue
        spans.append((max(xlo, 0.0), min(xhi, px), eps_inc))
    spans.sort(key=lambda s: s[0])
    for (a_lo, a_hi, _), (b_lo, b_hi, _) in zip(spans, spans[1:]):
        if b_lo < a_hi - _REL_TOL * px:
            raise ValueError("PMM bridge: layer {!r} Rectangles overlap in x (segments "
                             "must partition the period)".format(layer.name))
    segs, x = [], 0.0
    for lo, hi, eps in spans:
        if lo > x + _REL_TOL * px:
            segs.append(((lo - x) / px, bg))
        segs.append(((hi - lo) / px, eps))
        x = hi
    if x < px * (1.0 - _REL_TOL):
        segs.append(((px - x) / px, bg))
    total = sum(w for w, _ in segs)
    return [(w / total, e) for w, e in segs]              # exact unit sum for PMM's check


# DEFAULT `degree` ACROSS THE THREE PMM ENTRY POINTS (audit V-15, measured 2026-11-02). The same
# lumenairy spectral-element engine is reached at three different default polynomial degrees:
# design_to_pmm_stack / make_lumenairy_pmm_solver 16 (this file), the differentiable
# rcwa_design.pmm_stack_RT / pmm_stack_jones 12, and the 2-D pmm2d_backend 11 (hybrid engine only;
# its default pure engine converges in n_modes instead). The difference is
# deliberate (a jit-traced solve and a 2-D tensor-product basis pay superlinearly per degree) but
# it is NOT free, so a convergence study through one entry point does not transfer to its sibling.
# Measured through pmm_stack_RT (period 0.8 um, 1310 nm, superstrate 1, substrate 1.45, 21 orders)
# against degree 32: the E_x row (field ACROSS the walls, edge-singular) converges only
# algebraically -- |dR| = 1.5e-4 / 6.4e-5 / 1.4e-5 at degree 12 / 16 / 24 on a 400 nm Si
# lamellar grating, 6.8e-4 / 3.0e-4 / 6.6e-5 on a 100 nm Au one -- while the E_y row is at
# 9e-8 / 1e-8 / 4e-10 (Si). So degree 12 vs 16 moves R by ~1e-4 (dielectric) to ~4e-4 (metal):
# pin `degree` explicitly when comparing results across entry points.
def design_to_pmm_stack(design, lambda_m: float, *, eps_by_region=None,
                        degree: int = 16, n_orders: int = 21,
                        n_slices: Optional[int] = None,
                        layer_grids: str = "shared",
                        lateral_partition="auto", partition_rtol: float = 0.1,
                        max_segments: int = 24, lateral_average: str = "auto",
                        z_merge_rtol="auto", z_average: str = "auto",
                        pmm_kwargs: Optional[dict] = None):
    """Translate a 1-D-compatible DynaMeta Design into a Lumenairy PMMStack at one
    wavelength. Returns (stack, layer_names_top_first). Same layer dispatch as the RCWA
    bridge except: patterned layers go through layer_to_pmm_segments and a laterally
    structured gridded EpsField is averaged onto a lateral partition (below).

    degree      : polynomial degree per spectral element -- THE accuracy control (the TM channel
                  converges algebraically at metal corners; check convergence in degree).
    n_orders    : far-field order count (PMMStack far_field_orders); it does not change accuracy.
    n_slices    : None = one PMM layer per native z interval of a gridded field (the field's own,
                  possibly non-uniform, z grid); an int resamples to a uniform staircase.
    pmm_kwargs  : extra PMMStack keywords forwarded verbatim (elements_per_region, grade,
                  min_feature, factorization, window_halfwidth, and on newer lumenairy geometric
                  grading grade='geometric' / ('geometric', ratio[, min_element_m])). Keywords
                  the bridge sets itself raise ValueError; a keyword (or a grade value) the
                  installed lumenairy does not support raises LumenairyFeatureUnavailable naming
                  the release that adds it. Defaults: none (lumenairy's own defaults).
                  RECOMMENDED where the installed lumenairy has it (feature row
                  'pmm_geometric_grade', lumenairy >= 5.51): degree=8, pmm_kwargs={'grade':
                  'geometric', 'elements_per_region': 12}. MEASURED on the generic metal strip
                  cell of tests/_strip_cell_fixture.py (TM, 1200 nm): 4.0e-6 from the converged
                  2-D TM FEM reference (itself uncertain by ~3e-6), against 2.0e-2 for the default
                  degree 16 and 4.5e-3 for degree 32 (Chebyshev, one element per region; the same
                  numbers on lumenairy 5.22.0, 5.50.2 and 5.51.0).
                  On lumenairy >= 5.51 PMMStack also refuses a LAYOUT element smaller than 1e-6
                  (Chebyshev / uniform) or 1e-5 (geometric) wavelengths (ValueError; way back
                  pmm_kwargs={'allow_small_elements': True}), warns (UserWarning) for a
                  Chebyshev / uniform element between 1e-6 and 2e-6 wavelengths, and, at an exact
                  Rayleigh anomaly, solves at a wavelength nudged by ~1e-7 relative with a
                  WoodNudgeWarning (older releases solve the ill-posed point silently); the bridge
                  passes all three through.

    GAIN MEDIA (Im eps < 0). A gain LAYER (segment, uniform scalar or tensor) is solved where the
    installed lumenairy splits a non-passive layer's modes correctly -- decided by a probe solve
    (feature row 'pmm_gain_media'): true on 5.22.0 and on 5.51.0, false on 5.50.x, which returns
    R + T of 1e1 .. 1e4. Where it is false the bridge raises LumenairyFeatureUnavailable naming
    the layer (gate: the gain strip against the 2-D TM FEM, tests/test_lumenairy_feature_detection.py).
    A stack with net gain (R + T > 1) can draw lumenairy's "energy not conserved" UserWarning, whose
    premise is a passive stack; the bridge passes it through. A gain SUPERSTRATE or SUBSTRATE is
    refused on every release (ValueError, _common.refuse_gain_half_space).

    Laterally structured gridded fields (y-invariant, varying along x; refused before v0.16).
    The field is the periodic piecewise-linear interpolant of its samples (x in nm on [0, P]):
    lateral_partition : 'auto' (default) = adaptive partition seeded with every wall the other
                  layers own (and 0, P), refined while an interval's spread
                  max|eps - mean| / max(|mean|, 1) over any z slab exceeds partition_rtol (each
                  cut placed at the sample that best separates where the profile varies), up to
                  max_segments intervals (minimum width P/4096); a sequence of edge fractions =
                  exactly that partition; None = refuse the field (the v0.15.1 behaviour).
                  COST: every PMM layer of the stack pays an eigensolve on the union of all
                  walls, ~(segments x degree)^3; measured (2 BLAS threads, one structured layer
                  of S equal metal / dielectric segments, degree 16) ~0.05 / 0.27 / 1.7 s per
                  layer at 8 / 16 / 32 segments, the same within ~10 % on lumenairy 5.22.0,
                  5.50.2 and 5.51.0 (5.51's fast path speeds up laterally UNIFORM layers only).
                  Defaults partition_rtol=0.1, max_segments=24 (measurements in
                  docs/audit/2026-11-14-v0.16-pmm-bridge.md; a profile through Re eps = 0 needs
                  more).
    lateral_average : how an interval's lateral variation is averaged. 'laminate' = diag(harm,
                  arith, arith) (E_x across the variation sees the series mean; right for an
                  interval narrower than the layer: tall lamellae); 'arithmetic' = scalar means
                  (right for an interval wider than the layer: in a thin film E_x is pinned by
                  tangential continuity with its neighbours); 'auto' (default) = per interval by
                  that aspect-ratio rule against the layer thickness. Second order once the
                  partition resolves the variation (docs/audit/2026-11-14-v0.16-pmm-bridge.md).
    z_average   : 'auto' (default) = 'laminate' for a laterally structured field (each native z
                  interval as diag(arith, arith, harm) of the z-linear interpolant: E_z sees the
                  series mean; 12x smaller staircase error at 41 samples on a film through
                  Re eps = 0), 'arithmetic' (the endpoint average for every component, the
                  v0.15.1 slab) for a laterally uniform field and whenever n_slices is set.
                  'laminate' needs n_slices=None.
    z_merge_rtol : merge consecutive z slabs whose components agree within this relative
                  tolerance into one PMM layer, as a z-laminate diag(arith, arith, harm). 'auto'
                  (default) = 3e-2 for a laterally structured field and NO merge for a
                  laterally uniform graded field (its v0.15.1 result is kept bit for bit; pass a
                  float to merge it too); None/0 = never merge.
    For a laterally uniform field, z_average='laminate' with z_merge_rtol=3e-2 is the more
    accurate and faster choice (the defaults keep the v0.15.1 numbers).
    An AdvisoryWarning is emitted when an 'auto' partition stops at max_segments (or the minimum
    width) with a spread still above partition_rtol; stack.dynameta_partition_report =
    {layer: {'n_segments', 'max_spread', 'converged', 'capped', 'n_slabs', 'n_layers', ...}}
    (also for a laterally uniform layer when z_average/z_merge_rtol change its slabs).

    layer_grids (lumenairy >= 5.32, OPT-IN, default unchanged): 'shared' builds ONE union
    spectral-element grid carrying every layer's walls -- so a stack of layers with different
    wall positions pays, in every layer, for the walls of all the others. 'per-layer' gives
    each layer only its own walls and couples them with a non-conforming mortar.

    WHY IT IS OPT-IN RATHER THAN THE DEFAULT. The payoff is large and it grows with degree --
    MEASURED here on a 4-layer lamellar stack with distinct walls at 30 deg: 14.7x at
    degree 8, 73x at degree 12, 3.2x at degree 16, both paths converging to the same R (gap
    1.07e-6 -> 9.9e-8). But the mortar is NON-CONFORMING, so it trades the union grid's exact
    energy closure (|R+T-1| ~ 1e-13) for 4.9e-7 at degree 8, tightening to 2.2e-8 by degree
    16. This backend's job in DynaMeta is to be the CONVERGENCE REFEREE for the RCWA path --
    the one that has no Fourier-factorization accuracy floor -- and a referee that quietly
    stopped closing energy at 1e-13 would be a worse referee even while being 73x faster. So
    the caller asks for the speed explicitly and knows what it costs. (Bridge audit
    2026-08-31 vs lumenairy 5.42.1; conical and layer_absorption() verified unchanged on the
    per-layer path.)"""
    require_supported_geometry(design, "design_to_pmm_stack")   # 2026-11-02: refuse Stack.features / Inclusion.taper
    lum = _require_lumenairy()
    n_super, n_sub = end_media_indices(design, lambda_m)
    _refuse_gain_half_space("design_to_pmm_stack", n_super, n_sub)
    px = float(design.unit_cell.period_x_m)
    py = float(design.unit_cell.period_y_m)
    # mesh-region-keyed dicts (the run_pipeline/FEM bridge output) collapse to design-layer
    # keys; identity when already layer-keyed
    eps_by_region = collapse_regions_to_layers(design, eps_by_region or {})
    if str(layer_grids) not in ("shared", "per-layer"):
        raise ValueError("design_to_pmm_stack: layer_grids must be 'shared' or 'per-layer', "
                         "got {!r}".format(layer_grids))
    # PASS THE KEYWORD ONLY WHEN IT IS ASKED FOR. `layer_grids` arrived in lumenairy 5.32, and
    # this bridge's declared floor is VERSION_FLOOR = 5.22 -- so forwarding it unconditionally
    # (even as the default 'shared') breaks the PMM backend outright on every supported version
    # below 5.32 with "PMMStack.__init__() got an unexpected keyword argument". That is exactly
    # what the floor CI leg caught on the first push of this feature: not the new opt-in path,
    # but the DEFAULT one, on the oldest pins the library promises to support. Omitting the
    # kwarg at the default reproduces pre-5.32 behaviour bit-for-bit (lumenairy's own default is
    # 'shared'); asking for 'per-layer' on a floor install raises with the version named,
    # because there the speed-up genuinely does not exist.
    kw = {}
    if str(layer_grids) != "shared":
        # the per-layer spectral-element grid and its non-conforming mortar do not exist below
        # 5.32; use the default 'shared' grid there. (wave-7: the floor lives in the shared
        # _common.FEATURE_FLOORS table; LumenairyFeatureUnavailable is a NotImplementedError.)
        _require_feature("pmm_layer_grids",
                         "design_to_pmm_stack(layer_grids={!r})".format(layer_grids))
        kw["layer_grids"] = str(layer_grids)
    if isinstance(lateral_partition, str) and lateral_partition != "auto":
        raise ValueError("design_to_pmm_stack: lateral_partition must be 'auto', a sequence of "
                         "edge fractions or None, got {!r}".format(lateral_partition))
    if lateral_average not in ("auto", "laminate", "arithmetic"):
        raise ValueError("design_to_pmm_stack: lateral_average must be 'auto', 'laminate' or "
                         "'arithmetic', got {!r}".format(lateral_average))
    if not (z_merge_rtol is None or z_merge_rtol == "auto"
            or (isinstance(z_merge_rtol, (int, float)) and float(z_merge_rtol) >= 0.0)):
        raise ValueError("design_to_pmm_stack: z_merge_rtol must be 'auto', None or a float "
                         ">= 0, got {!r}".format(z_merge_rtol))
    if z_average not in ("auto", "arithmetic", "laminate"):
        raise ValueError("design_to_pmm_stack: z_average must be 'auto', 'arithmetic' or "
                         "'laminate', "
                         "got {!r}".format(z_average))
    kw.update(_resolve_pmm_kwargs(lum, pmm_kwargs))
    # far_field_orders= (the PMMStack keyword since before the 5.22 floor); `n_orders=` is only
    # its alias, so this is bit-identical and avoids the upstream alias deprecation.
    stack = lum.PMMStack(px, n_substrate=complex(n_sub), n_superstrate=complex(n_super),
                         degree=int(degree), far_field_orders=int(n_orders), **kw)
    # audit S4-17 (wave-7, 2026-11-02): the end media ride on the stack so the solver closure
    # does not evaluate end_media_indices a second time per solve (same values, bit for bit)
    stack.dynameta_end_media = (n_super, n_sub)
    # Seeds of every lateral partition: the walls every patterned layer owns. A structured layer's
    # partition then contains them (and the partitions built before it), so its own new walls
    # never form a cross-layer near-collision on the shared union grid.
    seeds: List[float] = []
    if lateral_partition is not None and any(
            ef is not None and not getattr(ef, "is_uniform", True) for ef in eps_by_region.values()):
        for L in design.stack.layers:
            ef = eps_by_region.get(L.name)
            if L.inclusions and not (ef is not None and (not getattr(ef, "is_uniform", True)
                                                         or getattr(ef, "is_tensor", False))):
                seeds.extend(_segment_walls(layer_to_pmm_segments(
                    L, design, lambda_m, px, py,
                    bg_eps=(ef.scalar if ef is not None else None))))
    report = {}
    names: List[str] = []

    def _add(name, thickness, **layer):
        # every layer passes the gain check on its way in (_require_gain_support)
        _require_gain_support("design_to_pmm_stack", name, layer)
        stack.add_layer(thickness, **layer)

    for L in reversed(design.stack.layers):              # superstrate side first
        ef = eps_by_region.get(L.name)
        if ef is not None and not getattr(ef, "is_uniform", True):
            v = ef.values_zyx
            if getattr(ef, "is_tensor", False):
                # v0.16: a laterally UNIFORM gridded tensor (a z-graded uniaxial film, e.g. a
                # Drude film with boundary scattering) is a z staircase of uniform tensor layers;
                # a laterally structured tensor field is still refused
                if not is_laterally_uniform(v):
                    raise ValueError("PMM bridge: layer {!r} carries a laterally structured "
                                     "gridded TENSOR EpsField; the bridge averages scalar "
                                     "structured fields only (a laterally uniform tensor field "
                                     "is accepted) -- use the RCWA bridge".format(L.name))
                plan, rep = _uniform_tensor_layer_plan(L, ef, n_slices=n_slices,
                                                       z_merge_rtol=z_merge_rtol,
                                                       z_average=z_average)
                if rep is not None:
                    report[L.name] = rep
                for t, eps in reversed(plan):            # superstrate side first
                    _add(L.name, t, eps=eps)
                    names.append(L.name)
                continue
            if not is_laterally_uniform(v):
                if lateral_partition is None:
                    raise ValueError("PMM bridge: layer {!r} carries a laterally-structured "
                                     "gridded EpsField and lateral_partition=None; pass "
                                     "lateral_partition='auto' (or use the RCWA bridge for "
                                     "rasterized cells)".format(L.name))
                plan, edges, rep = _structured_layer_plan(
                    L, ef, px, seeds, n_slices=n_slices, lateral_partition=lateral_partition,
                    partition_rtol=partition_rtol, max_segments=max_segments,
                    lateral_average=lateral_average, z_merge_rtol=z_merge_rtol,
                    z_average=z_average)
                seeds.extend(float(e) for e in edges[1:-1])
                report[L.name] = rep
                if isinstance(lateral_partition, str) and not rep["converged"]:
                    import warnings
                    warnings.warn(
                        "PMM bridge: layer {!r} lateral partition stopped at {} segments with a "
                        "relative eps spread of {:.3g} > partition_rtol={:g} (max_segments={}); "
                        "raise max_segments or partition_rtol, and check the answer's "
                        "convergence".format(L.name, rep["n_segments"], rep["max_spread"],
                                             float(partition_rtol), int(max_segments)),
                        AdvisoryWarning, stacklevel=2)
                for t, segs in reversed(plan):           # superstrate side first
                    _add(L.name, t, segments=segs)
                    names.append(L.name)
                continue
            # z-graded, laterally uniform: one uniform layer per z slab. The slices are
            # ascending-z (substrate-first); this stack is superstrate-first, so reverse
            # (audit C5-1). slice_layer_eps_field = slice_eps_field + the slab-thickness guard.
            plan, rep = _uniform_layer_plan(L, ef, n_slices=n_slices, z_merge_rtol=z_merge_rtol,
                                            z_average=z_average)
            if rep is not None:
                report[L.name] = rep
            for t, eps in reversed(plan):
                _add(L.name, t, eps=eps)
                names.append(L.name)
            continue
        if ef is not None and getattr(ef, "is_tensor", False):
            _add(L.name, L.thickness_m, eps=np.asarray(ef.tensor, dtype=complex))
            names.append(L.name)
            continue
        if L.inclusions:
            _add(L.name, L.thickness_m,
                            segments=layer_to_pmm_segments(
                                L, design, lambda_m, px, py,
                                bg_eps=(ef.scalar if ef is not None else None)))
            names.append(L.name)
            continue
        if ef is not None:
            _add(L.name, L.thickness_m, eps=complex(ef.scalar))
        else:
            _add(L.name, L.thickness_m,
                            eps=complex(design.materials.get(L.background_material)
                                        .eps(lambda_m)))
        names.append(L.name)
    stack.dynameta_partition_report = report
    return stack, names


def make_lumenairy_pmm_solver(*, degree: int = 16, n_orders: int = 21,
                              n_slices: Optional[int] = None,
                              stabilize=None, absorption: bool = False,
                              layer_grids: str = "shared",
                              lateral_partition="auto", partition_rtol: float = 0.1,
                              max_segments: int = 24, lateral_average: str = "auto",
                              z_merge_rtol="auto", z_average: str = "auto",
                              pmm_kwargs: Optional[dict] = None):
    """Build an `optical_solver` for run_pipeline backed by Lumenairy PMM (the exact seam
    signature + solve_sweep, mirroring the RCWA bridge incl. per-wavelength end media).
    1-D lamellar devices only. R/T are total order-summed efficiencies.

    The layer keywords (degree, n_orders, n_slices, layer_grids, lateral_partition,
    partition_rtol, max_segments, lateral_average, z_merge_rtol, z_average, pmm_kwargs) are those of
    design_to_pmm_stack; all enter cache_fingerprint. The solver carries native_carrier_grid =
    True: a gridded field on its native (non-uniform) carrier z grid is consumed interval by
    interval, so run_pipeline hands it the carrier solve's own z mesh lines.

    absorption=True (audit 8.1-3, PMM at RCWA-bridge parity): runs
    a second, retain_internal=True solve (R/T/r/t stay those of the plain solve) and fills
    per_region_absorption keyed by DESIGN layer name (slabs
    of a graded layer sum into their layer) + A_independent from PMMStack.layer_absorption
    -- the internal z-Poynting flux difference per layer, an independent volumetric
    measurement (lumenairy's own invariant closes it against 1 - sum R - sum T).

    CONICAL incidence (azimuth != 0) is now supported (audit 8.1-1 / consumer-gap B): the
    rotated s/p totals R/T and the co-pol amplitude r are SYNTHESIZED from the per-order
    complex amplitudes (lumenairy 5.22 per_order_amplitudes), since the native result rows
    are lab-basis s/p MIXTURES at phi != 0. OpticalResult.t (the zeroth-order transmission
    Jones, phase-bearing) is filled from PMMStack.jones_transmission on BOTH the in-plane
    and conical paths -- it was None before 5.22."""

    if pmm_kwargs is not None and not isinstance(pmm_kwargs, dict):
        raise TypeError("make_lumenairy_pmm_solver: pmm_kwargs must be a dict or None")
    layer_kw = dict(lateral_partition=lateral_partition, partition_rtol=partition_rtol,
                    max_segments=max_segments, lateral_average=lateral_average,
                    z_merge_rtol=z_merge_rtol, z_average=z_average,
                    pmm_kwargs=(dict(pmm_kwargs) if pmm_kwargs is not None else None))

    def _solve_at(design, eps_by_region, lambda_m):
        t0 = time.perf_counter()
        _guard_incidence_side(design.optical)
        theta, phi = _angles_rad(design.optical)
        pol = getattr(design.optical, "polarization", "y")
        stack, names = design_to_pmm_stack(design, lambda_m, eps_by_region=eps_by_region,
                                           degree=degree, n_orders=n_orders,
                                           layer_grids=layer_grids,
                                           n_slices=n_slices, **layer_kw)
        stack.set_source(lambda_m, theta=theta, phi=phi)
        # R / T / r / t always come from the plain solve, so asking for the absorption split can
        # never move them: lumenairy 5.51 routes a retain_internal solve through its dense cascade
        # and a plain one through the uniform-layer fast path, which differ in the last bit (R
        # measured 1.2e-16 relative apart on a single lossy layer; bit-identical up to 5.50). The
        # retained solve the absorption block needs is a second solve (only when absorption=True).
        orders, R2, T2, jones = stack.solve(stabilize=stabilize)
        row = _pol_row(design.optical)
        n_sup, n_sb = stack.dynameta_end_media      # audit S4-17: set by design_to_pmm_stack
        rf, tf = _p_basis_conversion(pol, theta, n_sup, n_sb)
        if abs(phi) > 1e-12:
            # rotated s/p eigen-polarization: synthesize R/T + the co-pol zeroth-order r/t from
            # the per-order amplitudes (the native lab rows R2/T2/jones are s/p mixtures at
            # phi != 0). The p-basis conversion is already baked into the eigen-projection, so
            # rf/tf are NOT re-applied here.
            R, T, r, t = _conical_synthesis(stack, pol, theta, phi, n_sup)
        else:
            R = float(np.sum(R2[row]))
            T = float(np.sum(T2[row]))
            r = complex(np.asarray(jones)[row, row]) * complex(rf)
            t = complex(np.asarray(stack.jones_transmission())[row, row]) * complex(tf)
        a_ind, pra = None, None
        # Audit S4-2 (defensive gate): at conical incidence the reported R/T/r/t are the ROTATED
        # s/p eigen-polarization while layer_absorption()[:, row] is the per-LAB-pol diagonal --
        # the true absorbed power is a quadratic form whose cross term lumenairy does not expose.
        # Today lumenairy itself raises (retain_internal unsupported at conical) before this
        # block, but do not rely on that: mirror the RCWA bridge's explicit conical early-out so
        # a future lumenairy relaxation cannot silently produce the inconsistent number.
        if absorption and abs(phi) <= 1e-12:
            # retain_internal only here (phi == 0): at conical the per-lab-pol diagonal is not the
            # eigen-pol absorption (audit S4-2) and lumenairy's own conical retain_internal raises.
            stack.solve(stabilize=stabilize, retain_internal=True)
            try:
                la = np.asarray(stack.layer_absorption())     # (n_layers, 2) per incident pol
                la = la[:, row] if la.ndim == 2 and la.shape[1] == 2 else la[row]
                pra = {}
                for name, val in zip(names, la):
                    pra[name] = pra.get(name, 0.0) + float(val)
                a_ind = float(np.sum(la))
            except Exception as exc:                          # pragma: no cover - defensive
                import warnings
                warnings.warn("lumenairy PMM bridge: per-layer absorption unavailable ({}); "
                              "A_independent left unset".format(exc), AdvisoryWarning, stacklevel=2)
        return OpticalResult(r=r, R=R, phase_deg=float(np.degrees(np.angle(r))),
                             solve_time_s=time.perf_counter() - t0, t=t, T=T,
                             A=1.0 - R - T, A_independent=a_ind, R_flux=R, T_flux=T,
                             per_region_absorption=pra)

    def _solve(design, geo, eps_by_region, lambda_m, n_super, n_sub):
        _require_eps_convention(eps_by_region, "make_lumenairy_pmm_solver")   # audit V-5
        return _solve_at(design, eps_by_region, lambda_m)

    def _solve_sweep(design, geo, assemble_at, lams, n_super, n_sub):
        return [_solve_at(design, assemble_at(lam), lam) for lam in lams]

    _solve.solve_sweep = _solve_sweep
    lp = (lateral_partition if lateral_partition is None or isinstance(lateral_partition, str)
          else tuple(float(e) for e in lateral_partition))
    pk = ("None" if pmm_kwargs is None
          else "{" + ",".join("{}={!r}".format(k, pmm_kwargs[k]) for k in sorted(pmm_kwargs)) + "}")
    _solve.cache_fingerprint = (
        "pmm(degree={},n_orders={},n_slices={},stabilize={},absorption={},layer_grids={},"
        "lateral_partition={!r},partition_rtol={!r},max_segments={},lateral_average={},"
        "z_merge_rtol={!r},z_average={},pmm_kwargs={})".format(
            degree, n_orders, n_slices, stabilize, absorption, layer_grids, lp,
            float(partition_rtol), int(max_segments), lateral_average, z_merge_rtol, z_average,
            pk))
    # consumes a non-uniform native z grid interval by interval (slice_layer_eps_field)
    _solve.native_carrier_grid = True
    return _solve
