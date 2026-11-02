"""Lumenairy 2-D crossed-patterned PMM as a DynaMeta optical backend (audit 8.1-4).

Two engines behind one seam (``engine=``):
- ``'pure'`` (PMM2DStackPure, the default): the NO-FLOOR staggered modified-Legendre
  PMM -- every region is solved in one shared modal basis, interfaces are square modal
  matches, and the Rayleigh projection happens ONCE at the far field, so accuracy is
  ``n_orders``-INDEPENDENT and tracks only the modal degree ``n_modes``.  This is the
  2-D convergence REFEREE: on a metallic patch its wall-resolved basis pins the value
  RCWA converges toward as n_orders grows (gated in validation/lumenairy_pmm2d_bridge
  GATE B, the referee pattern of the 1-D PMM bridge).  How a patterned layer reaches
  it is the ``pure_grid=`` choice (2026-11-02, lumenairy 5.50):
    'union'     -- the v1 path: ANALYTIC Rectangle inclusions painted onto the smallest
                   COMMENSURATE uniform (N, N) segment grid shared by every layer
                   (layer_to_pure_cell); raises when no N <= pure_max_cells fits.
    'shapes'    -- every patterned layer as add_layer(shapes=[Rect | Circle | Ellipse],
                   background_eps=): the stack merges every outline into ONE wall grid
                   and ONE coordinate map, so walls sit at their EXACT physical
                   positions (no commensurability needed) and circles / axis-aligned
                   ellipses are EXACT grid lines of a curved map (lumenairy's curved-
                   cell Phases B-E1), not staircases.  Rectangles only give the
                   identity map (the shipped unmapped solver on non-uniform walls).
    'per-layer' -- PMM2DStackPure(layer_grids='per-layer'): a Rectangle-only layer
                   becomes an eps_cell STRIP TILE on its OWN walls (x_walls/y_walls),
                   a curved layer its own map, joined by lumenairy's L2 / curved
                   mortar -- outlines of DIFFERENT layers may then cross in plan view
                   (which the 'shapes' merge refuses).
    'auto'      -- (default) 'union' when every patterned layer is Rectangle-only and
                   commensurate (BIT-IDENTICAL to v1), else 'shapes'.
  LUMENAIRY FLOORS per route (wave-7, 2026-11-02; measured per release, _common.FEATURE_FLOORS):
  'union' with scalar eps runs on the bridge floor 5.22; a TENSOR eps on the pure engine needs
  5.43, 'per-layer' (and every taper) 5.45, 'shapes' 5.50. Below them the bridge raises
  LumenairyFeatureUnavailable naming the route and both versions, before lumenairy is called.
- ``'hybrid'`` (PMM2DStackHybrid): the Fourier-projected 2-D PMM (per-layer exact
  spectral-element walls, no union-grid constraint, but the FMM ``n_orders`` floor).
  It accepts RASTERIZED cells (the shared optics.rasterize painter, byte-identical
  geometry to the RCWA bridge).
BOTH engines expose per-layer absorption (``layer_absorption``, wired via
``absorption=True`` at 1-D-PMM/RCWA parity -- the pure engine gained internal
retention in lumenairy 5.22, audit C3; before that only the hybrid engine had it).

TENSOR permittivity (2026-11-02; lumenairy >= 5.43 pure / hybrid eps_tensor_cell).
DynaMeta's EpsField.tensor is eps[i, j] with i, j in (x, y, z), D_i = eps_ij E_j --
the SAME (3, 3) layout lumenairy's Berreman / RCWA / PMM2D entries take, so it passes
through UNCHANGED (no transpose, no index permutation; measured 2026-11-02 at bring-up:
a uniform asymmetric lossy tensor through the pure engine matches BerremanStack's full
2x2 reflection AND transmission Jones to 6e-15 at normal / 2.8e-10 at 30 deg [M=5],
while the TRANSPOSED tensor moves the off-diagonal Jones by 8e-2 -- GATE F).  Both
engines take the FULL tensor: a block-form (in-plane, e_xz = e_yz = e_zx = e_zy = 0)
tensor rides the 2q^2 in-plane pencil, OUT-OF-PLANE coupling (a tilted LC director)
routes that layer to lumenairy's first-order 4q^2 generator + the generalized cascade;
e_zz == 0 is refused by lumenairy (both E_z eliminations divide by it).  What
lumenairy refuses under a CURVED map (an out-of-plane tensor on a per-layer curved
map) it raises itself (NotImplementedError naming the phase) -- the bridge does not
catch or re-route it.
Semantics on a layer WITH inclusions: a uniform tensor EpsField is the layer's
BACKGROUND (the scalar-override contract of optics.rasterize, and the FEM region
semantics: collapse_regions_to_layers drops the '__incl' overlays because inclusion
eps is frozen at the material value) -- so an LC background with scalar metal /
dielectric pillars is a patterned TENSOR cell ((N, N, 3, 3) union cell, a (3, 3)
background_eps under shapes=, a strip tile per-layer, an (Sx, Sy, 3, 3) raster on the
hybrid).  The RCWA bridge reads it the same way since 2026-11-02 (workstream B3; it
used to REPLACE the whole layer with the tensor and drop the inclusions): the hybrid
raster and the RCWA cell are both painted by optics.rasterize.layer_eps_tensor_cell,
and the RCWA bridge equals the direct-RCWAStack oracle of GATE G exactly
(validation/lumenairy_pmm2d_taper_wrap GATE O).  Gridded tensor EpsFields slice to (1, 1, 3, 3) cells when laterally uniform
(both engines) and to (Nx, Ny, 3, 3) eps_tensor_cell rasters otherwise (hybrid only;
the pure engine refuses rasters).

TRANSMISSION (2026-11-02): OpticalResult.t is the zeroth-order CO-polarized complex
transmission from lumenairy's jones_transmission() (PerOrderAmplitudesMixin, both
engines, LAB-referenced) times the shared p-basis t factor cos_i / cos_t -- the RCWA
bridge's mapping, so 'p' t is the Byrnes-tmm t_p (gated vs TMM in GATE A, and
|t|^2 Re(n_sub cos_t) / Re(n_sup cos_i) == T on non-diffracting isotropic cells, GATE
H).  Per-order complex amplitudes are NOT copied into OpticalResult (it has no slot for
them): build the stack with design_to_pmm2d_stack, set_source, solve, then read
stack.per_order_amplitudes('reflection' | 'transmission') -- the documented route,
consistent with the solver's r / t (tests/test_lumenairy_pmm2d_ext.py).

Scope contract (enforced loudly -- the honest-scope convention):
- 'pure': inclusions must be Rectangle / Circle / Ellipse (axis-aligned; lumenairy's
  shape primitives are EXACT for these); a seam-crossing inclusion is WRAPPED exactly
  (2026-11-02, _canonical_pure_cell: rectangles split into in-cell pieces, curved /
  tapered features moved by a rigid origin shift -- refused only when no origin puts
  them all inside); a TAPERED Rectangle (one per layer, scalar background) becomes
  lumenairy's add_tapered_pillar staircase (taper_slices / taper_rule; every other
  taper raises -- the hybrid engine refuses all of them);
  Polygon / RegularPolygon raise (no exact image -> engine='hybrid' or the RCWA
  bridge); laterally-structured gridded EpsFields raise (rasters have no exact
  walls).  Per-layer absorption IS supported (lumenairy 5.22 added internal
  retention to PMM2DStackPure -- absorption=True at hybrid parity).
- 'hybrid': patterned layers rasterize onto a (cell_samples, cell_samples) pixel grid
  (walls sit on pixel boundaries -- lumenairy's own cost guard rejects
  pathologically-many-wall rasters with its refine/route-elsewhere message);
  laterally-structured gridded EpsFields pass through as per-slab eps_cell /
  eps_tensor_cell.
- Both: in-plane incidence only (conical azimuth raises, audit C4-2); TOP incidence
  only.

Cross-library pins (verified against the installed 5.50 source, stack2d.py /
stack2d_pure.py / shapes2d.py): PMM2DStackPure(period_x, period_y, n_superstrate=,
n_substrate= [INDICES], n_modes=, n_orders=, layer_grids=); add_layer(thickness,
eps= [scalar | (3, 3)] | eps_cell= [(N, N) | (N, N, 3, 3)] | shapes=+background_eps=,
x_walls=, y_walls= [per-layer only]) where eps_cell is a SQUARE per-SEGMENT grid on
uniform walls i*period/N (Granet staggered basis) unless x_walls / y_walls place them;
shape primitives Rect(cx, cy, w, h, eps) / Circle(cx, cy, r, eps, core=) /
Ellipse(cx, cy, a, b, eps) in metres; PMM2DStackHybrid(period_x, period_y,
n_superstrate=, n_substrate=, degree= [ODD], n_orders=, formulation=);
add_layer(thickness, eps=|eps_cell=|eps_tensor_cell=) where eps_cell is a PIXEL grid
(walls extracted at value changes).  Both: set_source(wavelength, theta=, phi=)
[m/rad]; solve() -> (orders(N, 2), R(2, N), T(2, N), jones_reflection(2, 2)) with
rows keyed INCIDENT E_x (0) / E_y (1) in the PUBLIC gauge -- the rcwa_jones_2d
layout, so the pol-row mapping and the lab-basis -> Byrnes p-pol r / t conversion are
IDENTICAL to the RCWA bridge (gated vs TMM in GATE A); jones_transmission() is the
(2, 2) zeroth-order transmission Jones in the same layout.  Dispersive callables force
solve_vs_wavelength; this bridge rebuilds CONCRETE stacks per wavelength (the
bridge-wide policy).
"""

from __future__ import annotations

from dynameta.core.eps_field import require_solver_time_convention as _require_eps_convention

import time
from typing import List, Optional, Tuple

import numpy as np

from dynameta.core.interfaces import OpticalResult
from dynameta.core.layered import collapse_regions_to_layers, slice_eps_field
from dynameta.optics.lumenairy_bridge._common import (angles_rad as _angles_rad,
                                                      guard_conical_ppol as _guard_conical_ppol,
                                                      guard_incidence_side as _guard_incidence_side,
                                                      p_basis_conversion as _p_basis_conversion,
                                                      pol_row as _pol_row,
                                                      require_lumenairy as _require_lumenairy,
                                                      require_lumenairy_feature as _require_feature)
from dynameta.optics.rasterize import cell_axes, layer_eps_cell, layer_eps_tensor_cell
from dynameta.optics.tmm_reference import S as _S_NM
from dynameta.optics.tmm_reference import end_media_indices
from dynameta.geometry.guards import require_supported_geometry
from dynameta.warning_types import AdvisoryWarning

__all__ = ["design_to_pmm2d_stack", "make_lumenairy_pmm2d_solver", "layer_to_pure_cell",
           "pure_union_grid_n", "resolve_pure_grid"]

_REL_TOL = 1e-9
# commensurability slack for wall fractions (|f*N - round(f*N)|): generous vs float
# noise (~1e-13) yet far below any physical wall placement
_COMM_TOL = 1e-6
_ENGINES = ("pure", "hybrid")
_PURE_GRIDS = ("auto", "union", "shapes", "per-layer")
# inclusion kinds lumenairy's shape primitives represent EXACTLY (shapes2d: Rect is
# the identity map, Circle / Ellipse are curved-map grid lines); every other
# CrossSection (Polygon, RegularPolygon) has no exact pure-engine image
_PURE_KINDS = ("rectangle", "circle", "ellipse")


def _check_engine(engine: str) -> str:
    if engine not in _ENGINES:
        raise ValueError("PMM2D bridge: engine must be one of {} (got {!r})".format(
            _ENGINES, engine))
    return engine


def _check_pure_grid(pure_grid: str) -> str:
    if pure_grid not in _PURE_GRIDS:
        raise ValueError("PMM2D bridge: pure_grid must be one of {} (got {!r})".format(
            _PURE_GRIDS, pure_grid))
    return pure_grid


def _is_tensor(eps) -> bool:
    """True for a (3, 3) permittivity (the EpsField.tensor / lumenairy layout)."""
    return np.ndim(eps) == 2


def _painted(eps, tensor: bool):
    """A scalar inclusion eps as a cell entry: eps * I3 inside a TENSOR cell (an
    isotropic medium is the diagonal tensor), the scalar itself otherwise."""
    return complex(eps) * np.eye(3, dtype=complex) if tensor else complex(eps)


def _layer_background(layer, design, lambda_m: float, ef):
    """Background permittivity of an INCLUSION layer: a uniform EpsField override
    wins (scalar -> complex; tensor -> its (3, 3), the background-tensor semantics of
    the module docstring), else the background material eps at lambda."""
    if ef is not None and getattr(ef, "is_tensor", False):
        return np.asarray(ef.tensor, dtype=complex).reshape(3, 3)
    if ef is not None and getattr(ef, "scalar", None) is not None:
        return complex(ef.scalar)
    return complex(design.materials.get(layer.background_material).eps(lambda_m))


def _check_in_cell(layer, inc, px: float, py: float, engine_tag: str = "pure"):
    """Every pure-engine inclusion must lie inside [0, px] x [0, py] once
    _canonical_pure_cell has wrapped the design (lumenairy's shape layouts refuse a
    seam-crossing shape) -- an internal pin, unreachable through design_to_pmm2d_stack."""
    xlo, xhi, ylo, yhi = inc.shape.bbox_m()
    if xlo < -_REL_TOL * px or xhi > px * (1.0 + _REL_TOL) \
            or ylo < -_REL_TOL * py or yhi > py * (1.0 + _REL_TOL):
        raise ValueError(
            "PMM2D bridge ({}): layer {!r} {} bbox x [{:.3e}, {:.3e}] y [{:.3e}, "
            "{:.3e}] leaves the cell [0, {:.3e}] x [0, {:.3e}] (not wrapped: "
            "build through design_to_pmm2d_stack, which splits / shifts it)".format(
                engine_tag, layer.name, getattr(inc.shape, "kind", "?"), xlo, xhi,
                ylo, yhi, px, py))


def _require_pure_kind(layer, inc):
    kind = getattr(inc.shape, "kind", "")
    if kind not in _PURE_KINDS:
        raise ValueError(
            "PMM2D bridge (pure): layer {!r} inclusion kind {!r} has no EXACT image "
            "in the pure staggered basis (lumenairy's shape primitives cover {}: a "
            "polygon would be a staircase, which is what the referee engine exists "
            "to avoid); use engine='hybrid' (rasterized, pixel-snapped walls) or the "
            "RCWA bridge".format(layer.name, kind, ", ".join(_PURE_KINDS)))
    return kind


def _pure_wall_fractions(layer, period_x_m: float,
                         period_y_m: float) -> Tuple[List[float], List[float]]:
    """Wall positions of one patterned layer as PERIOD FRACTIONS (x list, y list).
    Every inclusion must be an axis-aligned Rectangle lying inside the cell --
    the pure engine's analytic vocabulary (exact walls, Eq. 26 of Granet 2023);
    anything else raises with a pointer (the honest scope)."""
    px, py = float(period_x_m), float(period_y_m)
    fx, fy = [], []
    for inc in layer.inclusions:
        if getattr(inc.shape, "kind", "") != "rectangle":
            raise ValueError(
                "PMM2D bridge (pure): layer {!r} inclusion kind {!r} is not an "
                "axis-aligned Rectangle -- the pure staggered basis needs exact "
                "rectangular walls; use engine='hybrid' (rasterized, pixel-snapped "
                "walls) or the RCWA bridge".format(
                    layer.name, getattr(inc.shape, "kind", "?")))
        if getattr(inc, "taper", None) is not None:
            raise ValueError(
                "PMM2D bridge (pure): layer {!r} holds a TAPERED inclusion -- a "
                "z-staircase with different walls per slice has no single union "
                "grid; tapers take pure_grid='per-layer' (the 'auto' choice)".format(
                    layer.name))
        _check_in_cell(layer, inc, px, py)
        xlo, xhi, ylo, yhi = inc.shape.bbox_m()
        fx += [max(xlo, 0.0) / px, min(xhi, px) / px]
        fy += [max(ylo, 0.0) / py, min(yhi, py) / py]
    return fx, fy


def pure_union_grid_n(design, *, max_cells: int = 32) -> Optional[int]:
    """The smallest uniform square grid size N such that EVERY patterned layer's
    Rectangle walls land on the segment boundaries i/N of BOTH axes -- the pure
    engine's union-grid (all patterned layers share ONE (N, N) segmentation, and
    lumenairy's staggered Basis1D uses UNIFORM segments only).  None when the
    design has no patterned layer; raises when no N <= max_cells fits (each extra
    segment costs modal DOF ~ (N*(n_modes-1))^2, so a large commensurate N is a
    cost problem, not just a validity one).  A seam-crossing inclusion counts by
    its wrapped in-cell pieces (_canonical_pure_cell)."""
    design, _ = _canonical_pure_cell(design)
    px = float(design.unit_cell.period_x_m)
    py = float(design.unit_cell.period_y_m)
    fracs: List[float] = []
    for L in design.stack.layers:
        if L.inclusions:
            fx, fy = _pure_wall_fractions(L, px, py)
            fracs += fx + fy
    if not fracs:
        return None
    for n in range(1, int(max_cells) + 1):
        if all(abs(f * n - round(f * n)) <= _COMM_TOL for f in fracs):
            return n
    raise ValueError(
        "PMM2D bridge (pure): no uniform (N, N) segmentation with N <= {} puts "
        "every Rectangle wall on a segment boundary (wall fractions {}); the pure "
        "staggered basis needs commensurate walls.  Raise pure_max_cells (cost "
        "grows ~ N^2 in modal DOF) or use engine='hybrid' / the RCWA bridge, "
        "which have no union-grid constraint.".format(
            int(max_cells), sorted({round(f, 6) for f in fracs})))


def layer_to_pure_cell(layer, design, lambda_m: float, period_x_m: float,
                       period_y_m: float, n_cells: int, *, bg_eps=None):
    """Paint one patterned layer onto the pure engine's (n_cells, n_cells)
    per-SEGMENT eps grid (segment i spans [i, i+1]*period/N): background eps
    (eps_by_region scalar override wins, the rasterizer contract) overpainted by
    each Rectangle in ASCENDING priority via segment-CENTER membership --
    unambiguous because pure_union_grid_n guaranteed the walls sit on segment
    boundaries, never inside a segment.

    A (3, 3) TENSOR bg_eps (a uniform tensor EpsField on the layer = its
    background, 2026-11-02) returns the (n, n, 3, 3) block-form / full tensor cell
    lumenairy's PMM2DStackPure.add_layer(eps_cell=) takes, with every (scalar)
    inclusion painted as eps * I3; the scalar path is unchanged byte for byte."""
    n = int(n_cells)
    px, py = float(period_x_m), float(period_y_m)
    _pure_wall_fractions(layer, px, py)          # re-validate shape kinds/bounds
    tensor = bg_eps is not None and _is_tensor(bg_eps)
    cx = (np.arange(n) + 0.5) * (px / n)         # segment centres, axis 0 = x
    cy = (np.arange(n) + 0.5) * (py / n)
    X, Y = np.meshgrid(cx, cy, indexing="ij")
    if tensor:
        cell = np.broadcast_to(np.asarray(bg_eps, dtype=complex).reshape(3, 3),
                               (n, n, 3, 3)).copy()
    else:
        bg = complex(bg_eps if bg_eps is not None
                     else design.materials.get(layer.background_material).eps(lambda_m))
        cell = np.full((n, n), bg, dtype=complex)
    for inc in sorted(layer.inclusions, key=lambda i: getattr(i, "priority", 0)):
        xlo, xhi, ylo, yhi = inc.shape.bbox_m()
        mask = (X >= xlo) & (X <= xhi) & (Y >= ylo) & (Y <= yhi)
        cell[mask] = _painted(design.materials.get(inc.material).eps(lambda_m), tensor)
    return cell


def _n_distinct_interior(fracs: List[float]) -> int:
    """Count of DISTINCT interior wall fractions (0 < f < 1, merged at _COMM_TOL)."""
    out: List[float] = []
    for f in sorted(fracs):
        if f <= _COMM_TOL or f >= 1.0 - _COMM_TOL:
            continue
        if out and f - out[-1] <= _COMM_TOL:
            continue
        out.append(f)
    return len(out)


def resolve_pure_grid(design, pure_grid: str = "auto", *, max_cells: int = 32) -> str:
    """The pure-engine patterned-layer route actually taken for ``pure_grid``
    ('union' | 'shapes' | 'per-layer'; see the module docstring).  Explicit choices
    pass through ('union' on a design it cannot represent raises later, from
    pure_union_grid_n / _pure_wall_fractions -- the v1 messages).

    'auto' keeps the v1 UNION grid -- BIT-IDENTICAL results -- when every patterned
    layer is Rectangle-only, commensurate with some N <= max_cells, AND that N is
    affordable next to the exact-wall alternative: N^2 <= 2 S^2, S = the merged strip
    count per axis the 'shapes' route would solve (the union of every layer's walls,
    segment counts equalized).  Otherwise 'shapes'.  Why the cost clause (measured
    2026-11-02): a commensurate N is NOT a cheap N -- a 0.46 x 0.37-period pillar
    snapped to eighths needs N = 16 where its walls need S = 3, and the uniform
    grid's dense pencil 2 (N (M-1))^2 = 12 800 at M = 6 trips lumenairy's own
    max_pencil_dof = 12 000 refusal (~29 GB projected), while 'shapes' solves the
    SAME walls on a 450-dof pencil in ~4 s.  N^2 <= 2 S^2 keeps every fixture the v1
    path ran (the gated centred patch: N = 4, S = 3) on the union grid.

    A TAPERED inclusion (2026-11-02) needs per-layer grids (lumenairy's
    add_tapered_pillar staircase): 'auto' -> 'per-layer', and an explicit 'union' /
    'shapes' raises.  Seam-crossing inclusions are judged by their wrapped in-cell
    image (_canonical_pure_cell)."""
    _check_pure_grid(pure_grid)
    if _has_taper(design):
        if pure_grid in ("union", "shapes"):
            raise ValueError(
                "PMM2D bridge (pure): pure_grid={!r} cannot build a TAPERED inclusion -- "
                "lumenairy's add_tapered_pillar slices it into a z-staircase whose "
                "slices have DIFFERENT walls, which needs layer_grids='per-layer'; pass "
                "pure_grid='per-layer' or 'auto'".format(pure_grid))
        return "per-layer"
    if pure_grid != "auto":
        return pure_grid
    design, _ = _canonical_pure_cell(design)
    patterned = [L for L in design.stack.layers if L.inclusions]
    if not patterned:
        return "union"
    if all(getattr(inc.shape, "kind", "") == "rectangle"
           for L in patterned for inc in L.inclusions):
        try:
            n = pure_union_grid_n(design, max_cells=max_cells)
        except ValueError:
            n = None                             # non-commensurate / out-of-cell
        if n is not None:
            px = float(design.unit_cell.period_x_m)
            py = float(design.unit_cell.period_y_m)
            fx_all: List[float] = []
            fy_all: List[float] = []
            for L in patterned:
                fx, fy = _pure_wall_fractions(L, px, py)
                fx_all += fx
                fy_all += fy
            s = max(_n_distinct_interior(fx_all), _n_distinct_interior(fy_all)) + 1
            if n * n <= 2 * s * s:
                return "union"
    return "shapes"


def _layer_lum_shapes(layer, design, lambda_m: float, px: float, py: float, *,
                      circle_core=None) -> list:
    """The layer's inclusions as lumenairy shape primitives (shapes2d), in ASCENDING
    priority -- lumenairy paints a later shape over an earlier one, the rasterizer's
    overpainting order.  DynaMeta Rectangle(cx, cy, width, height) -> Rect(cx, cy, w,
    h) (the bbox CLIPPED to the cell exactly as the union path clips its walls; a
    side on the cell boundary is the boundary itself in lumenairy's layout);
    Circle(cx, cy, radius) -> Circle(cx, cy, r, core=circle_core); axis-aligned
    Ellipse(cx, cy, rx, ry) -> Ellipse(cx, cy, a=rx, b=ry).  All lengths in metres
    on both sides; eps is the inclusion material at lambda (scalar)."""
    _require_lumenairy()
    from lumenairy.elements.pmm import Circle as _LCircle
    from lumenairy.elements.pmm import Ellipse as _LEllipse
    from lumenairy.elements.pmm import Rect as _LRect
    out = []
    for inc in sorted(layer.inclusions, key=lambda i: getattr(i, "priority", 0)):
        kind = _require_pure_kind(layer, inc)
        _check_in_cell(layer, inc, px, py)
        e = complex(design.materials.get(inc.material).eps(lambda_m))
        sh = inc.shape
        if kind == "rectangle":
            xlo, xhi, ylo, yhi = sh.bbox_m()
            xlo, xhi = max(xlo, 0.0), min(xhi, px)
            ylo, yhi = max(ylo, 0.0), min(yhi, py)
            out.append(_LRect(0.5 * (xlo + xhi), 0.5 * (ylo + yhi), xhi - xlo,
                              yhi - ylo, eps=e))
        elif kind == "circle":
            out.append(_LCircle(float(sh.cx_m), float(sh.cy_m), float(sh.radius_m),
                                eps=e, core=circle_core))
        else:                                    # ellipse (axis-aligned in DynaMeta)
            out.append(_LEllipse(float(sh.cx_m), float(sh.cy_m), float(sh.rx_m),
                                 float(sh.ry_m), eps=e))
    return out


def _merge_walls(edges, period: float) -> List[float]:
    """Sorted INTERIOR walls (0 < w < period) from rectangle edge positions, with
    edges closer than _COMM_TOL * period merged (float noise of coincident edges) and
    edges on the cell boundary dropped (a boundary is not an interior wall)."""
    tol = _COMM_TOL * period
    walls: List[float] = []
    for w in sorted(edges):
        if w <= tol or w >= period - tol:
            continue
        if walls and w - walls[-1] <= tol:
            continue
        walls.append(float(w))
    return walls


def _equalize_segments(xw: List[float], yw: List[float], px: float,
                       py: float) -> Tuple[List[float], List[float]]:
    """lumenairy's staggered tensor-product basis needs the SAME segment COUNT on
    both axes (positions may differ; PMM2DStackPure.add_layer docstring).  Pad the
    axis with fewer segments by bisecting its WIDEST segment (an extra wall inside a
    homogeneous strip is a finer partition of the same device, not a new wall)."""
    xw, yw = list(xw), list(yw)

    def _split(w, per):
        b = [0.0] + w + [per]
        k = int(np.argmax(np.diff(b)))
        return sorted(w + [0.5 * (b[k] + b[k + 1])])

    while len(xw) != len(yw):
        if len(xw) < len(yw):
            xw = _split(xw, px)
        else:
            yw = _split(yw, py)
    return xw, yw


def _rect_strip_tile(layer, design, lambda_m: float, px: float, py: float, bg):
    """A Rectangle-only layer as ITS OWN wall grid (the pure engine's per-layer route,
    PMM2DStackPure(layer_grids='per-layer').add_layer(eps_cell=tile, x_walls=,
    y_walls=)): interior walls = the clipped rectangle edges, segment counts
    equalized, and the STRIP TILE painted by segment-centre membership (unambiguous:
    every rectangle edge IS a wall).  Returns (x_walls, y_walls, tile) with tile
    (len(xw) + 1, len(yw) + 1) or (..., 3, 3) for a tensor background."""
    ex, ey = [], []
    for inc in layer.inclusions:
        _require_pure_kind(layer, inc)
        _check_in_cell(layer, inc, px, py)
        xlo, xhi, ylo, yhi = inc.shape.bbox_m()
        ex += [max(xlo, 0.0), min(xhi, px)]
        ey += [max(ylo, 0.0), min(yhi, py)]
    xw, yw = _equalize_segments(_merge_walls(ex, px), _merge_walls(ey, py), px, py)
    bx = np.array([0.0] + xw + [px])
    by = np.array([0.0] + yw + [py])
    X, Y = np.meshgrid(0.5 * (bx[1:] + bx[:-1]), 0.5 * (by[1:] + by[:-1]),
                       indexing="ij")
    tensor = _is_tensor(bg)
    if tensor:
        tile = np.broadcast_to(np.asarray(bg, dtype=complex), X.shape + (3, 3)).copy()
    else:
        tile = np.full(X.shape, complex(bg), dtype=complex)
    for inc in sorted(layer.inclusions, key=lambda i: getattr(i, "priority", 0)):
        xlo, xhi, ylo, yhi = inc.shape.bbox_m()
        mask = (X >= xlo) & (X <= xhi) & (Y >= ylo) & (Y <= yhi)
        tile[mask] = _painted(design.materials.get(inc.material).eps(lambda_m), tensor)
    return xw, yw, tile


# ---- periodic wrap on the pure engine (2026-11-02, workstream B3) ------------------------------
# The pure engine's vocabulary (union grid, strip tiles, lumenairy shapes2d, add_tapered_pillar)
# is IN-CELL only: lumenairy's shape layouts refuse a shape crossing the periodic seam ("shapes do
# not wrap ... Shift the lattice origin", shapes2d._check_inside -- curved shapes must sit at least
# the 1e-3-period sliver from the edges) and add_tapered_pillar needs 0 < lo < hi < period. The
# FEM mesher, the hybrid raster and the RCWA bridge all WRAP a boundary-crossing inclusion
# (optics.rasterize.inclusion_mask). _canonical_pure_cell maps a design onto that vocabulary
# EXACTLY, never by approximation:
#   * a RECTANGLE crossing the seam splits into its in-cell pieces (up to 4: the x and y
#     intervals reduced mod the period) -- the same point set of the periodic medium;
#   * a CIRCLE / ELLIPSE or a TAPERED pillar cannot be split into lumenairy primitives (a circular
#     segment is no Circle; add_tapered_pillar takes ONE in-cell box), so the WHOLE lattice
#     origin is shifted (every inclusion of every layer by the same (sx, sy), mod the period) to
#     put the seam in the widest gap between those features. A rigid lateral translation of a
#     periodic structure leaves every order's power and the ZEROTH-order r / t unchanged (the
#     m-th order amplitude gains exp(-i G_m s)): OpticalResult carries order-summed R / T, the
#     (0, 0) r / t and per-layer absorption only, so the solver result is the unshifted one.
#     NB a caller reading stack.per_order_amplitudes() of design_to_pmm2d_stack directly sees
#     those exp(-i G_m s) phases on m != 0 orders; the shift is published on the returned
#     stack as `dynameta_origin_shift_m` ((0.0, 0.0) when none was needed).
#   * no shift when every curved / tapered feature already sits inside (and the design object
#     itself is used, UNCHANGED, when nothing crosses the seam -- every in-cell design builds
#     byte for byte as before); a precise refusal when those features' extents cover the whole
#     period on an axis (no origin puts them all inside one cell).
# GATE K of validation/lumenairy_pmm2d_taper_wrap.py pins it (translation invariance + RCWA).
_SEAM_MARGIN = 1e-3     # lumenairy shapes2d strict-inside sliver (_STAG_MIN_SEG_FRAC), in periods


def _unsplittable(inc) -> bool:
    """Inclusions the wrap cannot split into in-cell primitives: curved shapes and tapers."""
    return (getattr(inc.shape, "kind", "") in ("circle", "ellipse")
            or getattr(inc, "taper", None) is not None)


def _extent(inc, axis: int) -> Tuple[float, float]:
    """(lo, hi) of an inclusion along x (0) / y (1): the top section's bbox, widened by the
    taper bottom's when that is larger (an allow_wider_bottom undercut)."""
    b = inc.shape.bbox_m()
    lo, hi = b[2 * axis], b[2 * axis + 1]
    if getattr(inc, "taper", None) is not None:
        bb = inc.taper.bottom.bbox_m()
        lo, hi = min(lo, bb[2 * axis]), max(hi, bb[2 * axis + 1])
    return float(lo), float(hi)


def _seam_shift(arcs, per: float, axis_name: str) -> float:
    """The origin shift along one axis: 0.0 when every unsplittable extent already lies inside
    [m, per - m] (m = the lumenairy sliver), else the midpoint of the widest gap the extents
    (reduced mod per) leave on the periodic axis. Raises when no gap of at least 2 m exists."""
    m = _SEAM_MARGIN * per
    if all(lo >= m and hi <= per - m for lo, hi in arcs):
        return 0.0
    segs: Optional[list] = []
    for lo, hi in arcs:
        if hi - lo >= per - 2.0 * m:
            segs = None
            break
        a = lo % per
        segs += [(a, a + hi - lo), (a - per, a - per + hi - lo)]
    best = None
    if segs is not None:
        segs.sort()
        cover_hi = segs[0][1]
        for a, b in segs[1:]:
            if a > cover_hi:
                if best is None or a - cover_hi > best[1] - best[0]:
                    best = (cover_hi, a)
            cover_hi = max(cover_hi, b)
    if best is None or best[1] - best[0] < 2.0 * m:
        raise ValueError(
            "PMM2D bridge (pure): the curved / tapered inclusions' {}-extents (mod the period "
            "{:.4e} m) {} leave no seam gap of at least 2 x {:.0e} period -- no lattice origin "
            "puts them all inside one cell, which lumenairy's shape layouts and "
            "add_tapered_pillar require (shapes do not wrap; a circular segment has no exact "
            "primitive). Use engine='hybrid' or the RCWA bridge (both wrap by "
            "rasterization).".format(axis_name, per, [(lo, hi) for lo, hi in arcs], _SEAM_MARGIN))
    return 0.5 * (best[0] + best[1]) % per


def _wrap_intervals(lo: float, hi: float, per: float) -> List[Tuple[float, float]]:
    """[lo, hi] reduced onto the periodic cell [0, per]: one interval, or two when it crosses the
    seam, or the whole period; slivers below the merge tolerance are dropped."""
    tol = _REL_TOL * per
    if hi - lo >= per - tol:
        return [(0.0, per)]
    if lo >= -tol and hi <= per + tol:
        return [(max(lo, 0.0), min(hi, per))]
    a = lo % per
    b = a + (hi - lo)
    if b <= per + tol:
        return [(a, min(b, per))]
    return [iv for iv in ((a, per), (0.0, b - per)) if iv[1] - iv[0] > tol]


def _translated(cs, dx: float, dy: float, px: float, py: float):
    """A Rectangle / Circle / Ellipse with its centre moved by (-dx, -dy), reduced mod the period."""
    from dynameta.geometry.cross_section import Circle, Ellipse, Rectangle
    cx, cy = (float(cs.cx_m) - dx) % px, (float(cs.cy_m) - dy) % py
    if isinstance(cs, Rectangle):
        return Rectangle(cx, cy, cs.width_m, cs.height_m)
    if isinstance(cs, Circle):
        return Circle(cx, cy, cs.radius_m)
    if isinstance(cs, Ellipse):
        return Ellipse(cx, cy, cs.rx_m, cs.ry_m)
    raise ValueError("PMM2D bridge (pure): cannot translate a {!r} section (no exact pure-engine "
                     "image)".format(getattr(cs, "kind", type(cs).__name__)))


def _canonical_pure_cell(design):
    """(design', (sx, sy)): the pure engine's in-cell image of `design` (block comment above).
    `design` itself, unchanged, with (0.0, 0.0) when nothing crosses the seam."""
    import dataclasses

    from dynameta.geometry import Taper
    from dynameta.geometry.cross_section import Rectangle
    px = float(design.unit_cell.period_x_m)
    py = float(design.unit_cell.period_y_m)
    incs = [inc for L in design.stack.layers for inc in L.inclusions]
    if any(getattr(inc.shape, "kind", "") not in _PURE_KINDS for inc in incs):
        return design, (0.0, 0.0)               # refused downstream (_require_pure_kind)
    hard = [inc for inc in incs if _unsplittable(inc)]
    sx = _seam_shift([_extent(i, 0) for i in hard], px, "x") if hard else 0.0
    sy = _seam_shift([_extent(i, 1) for i in hard], py, "y") if hard else 0.0
    tol_x, tol_y = _REL_TOL * px, _REL_TOL * py

    def crosses(inc):
        xlo, xhi, ylo, yhi = inc.shape.bbox_m()
        return xlo < -tol_x or xhi > px + tol_x or ylo < -tol_y or yhi > py + tol_y

    if sx == 0.0 and sy == 0.0 and not any(crosses(i) for i in incs):
        return design, (0.0, 0.0)
    layers = []
    for L in design.stack.layers:
        out = []
        for inc in L.inclusions:
            shp = inc.shape
            if sx or sy:
                shp = _translated(shp, sx, sy, px, py)
            if _unsplittable(inc):
                tap = inc.taper
                if tap is not None and (sx or sy):
                    tap = Taper(bottom=_translated(tap.bottom, sx, sy, px, py),
                                allow_wider_bottom=tap.allow_wider_bottom)
                out.append(dataclasses.replace(inc, shape=shp, taper=tap))
                continue
            xlo, xhi, ylo, yhi = shp.bbox_m()
            for ax0, ax1 in _wrap_intervals(xlo, xhi, px):
                for ay0, ay1 in _wrap_intervals(ylo, yhi, py):
                    out.append(dataclasses.replace(
                        inc, shape=Rectangle(0.5 * (ax0 + ax1), 0.5 * (ay0 + ay1),
                                             ax1 - ax0, ay1 - ay0)))
        layers.append(dataclasses.replace(L, inclusions=out))
    return (dataclasses.replace(design, stack=dataclasses.replace(design.stack, layers=layers)),
            (sx, sy))


# ---- tapered pillars on the pure engine (2026-11-02, workstream B3) ------------------------------
# lumenairy 5.50 PMM2DStackPure.add_tapered_pillar(thickness, eps_pillar=, eps_host=,
# x_bounds_bottom=, y_bounds_bottom=, x_bounds_top=, y_bounds_top=, n_slices=, rule=) -- pinned
# against the installed source (stack2d_pure.py): ONE axis-aligned rectangular pillar, SCALAR
# eps_pillar / eps_host (both cast with complex()), bounds strictly inside the cell
# (0 < lo < hi < period at every slice), layer_grids='per-layer' REQUIRED (each slice its own
# 3-segment non-uniform grid, slices joined by the L2 mortar). The bounds interpolate LINEARLY
# from bottom to top; slices are appended TOP-down (superstrate-first, as this bridge builds),
# slice s of n sampled at z-fraction 1 - (s + 1/2) / n above the bottom ('midpoint', O(1/n^2))
# or 1 - (s + 1) / n ('bottom', O(1/n)). That is a STAIRCASE of exact-wall slices -- the taper
# converges in taper_slices (and, per slice, in n_modes); the bridge default 8 is lumenairy's.
# The DynaMeta Taper is the same object: Inclusion.shape = the TOP-face section, Taper.bottom the
# BOTTOM-face section, linear in z about the shape's own centre (memory feedback_taper_not_slant).
# What lumenairy does not build is REFUSED with the specific reason, never approximated: a
# tapered Circle / Ellipse, a tapered pillar sharing its layer with any other inclusion, a TENSOR
# background around it (eps_host is scalar), and pure_grid='union' / 'shapes' (stack-wide
# single-grid routes; the staircase needs per-layer grids -- 'auto' takes 'per-layer' whenever a
# taper is present). add_tapered_pillars (several pillars per slice) has no `rule` and an
# equal-wall-count restriction; it is not wired (follow-on). GATES L-N of
# validation/lumenairy_pmm2d_taper_wrap.py pin the mapping (zero-angle identity, taper_slices
# convergence vs a stair-step DynaMeta stack through this bridge AND the RCWA bridge,
# reversed-taper / wrong-angle discrimination).
_TAPER_RULES = ("midpoint", "bottom")


def _check_taper_args(taper_slices, taper_rule) -> Tuple[int, str]:
    if isinstance(taper_slices, bool) or int(taper_slices) != taper_slices or taper_slices < 1:
        raise ValueError("PMM2D bridge: taper_slices must be an integer >= 1 (got {!r})".format(
            taper_slices))
    if taper_rule not in _TAPER_RULES:
        raise ValueError("PMM2D bridge: taper_rule must be one of {} (got {!r})".format(
            _TAPER_RULES, taper_rule))
    return int(taper_slices), taper_rule


def _has_taper(design) -> bool:
    return any(getattr(inc, "taper", None) is not None
               for L in design.stack.layers for inc in L.inclusions)


def _tapered_pillar_kwargs(L, design, lambda_m: float, px: float, py: float, bg) -> dict:
    """The add_tapered_pillar keyword set for a layer holding ONE tapered Rectangle (the scope
    lumenairy builds), or a ValueError naming what is outside it."""
    if len(L.inclusions) != 1:
        raise ValueError(
            "PMM2D bridge (pure): layer {!r} holds a tapered inclusion next to {} other "
            "inclusion(s) (counting the in-cell pieces of a wrapped rectangle); lumenairy's "
            "add_tapered_pillar builds ONE pillar per layer (the multi-pillar "
            "add_tapered_pillars is not wired) -- put each tapered pillar in its own layer, or "
            "use straight walls (taper=None)".format(L.name, len(L.inclusions) - 1))
    inc = L.inclusions[0]
    if getattr(inc.shape, "kind", "") != "rectangle":
        raise ValueError(
            "PMM2D bridge (pure): layer {!r} tapered {!r} -- lumenairy's add_tapered_pillar "
            "tapers an axis-aligned RECTANGLE only (a tapered circle / ellipse would be a "
            "staircase of curved maps it does not build); use taper=None or a z-staircase of "
            "untapered layers".format(L.name, getattr(inc.shape, "kind", "?")))
    if _is_tensor(bg):
        raise ValueError(
            "PMM2D bridge (pure): layer {!r} has a TENSOR background (uniform tensor "
            "eps_by_region entry) around a tapered pillar; lumenairy's add_tapered_pillar takes a "
            "SCALAR eps_host only".format(L.name))
    t, b = inc.shape.bbox_m(), inc.taper.bottom.bbox_m()
    for nm, (xlo, xhi, ylo, yhi) in (("top", t), ("bottom", b)):
        if not (0.0 < xlo < xhi < px and 0.0 < ylo < yhi < py):
            raise ValueError(           # unreachable after _canonical_pure_cell; kept as a pin
                "PMM2D bridge (pure): layer {!r} tapered pillar {} bounds x [{:.4e}, {:.4e}] y "
                "[{:.4e}, {:.4e}] must lie strictly inside the cell (add_tapered_pillar)".format(
                    L.name, nm, xlo, xhi, ylo, yhi))
    return dict(eps_pillar=complex(design.materials.get(inc.material).eps(lambda_m)),
                eps_host=complex(bg), x_bounds_bottom=(b[0], b[1]),
                y_bounds_bottom=(b[2], b[3]), x_bounds_top=(t[0], t[1]),
                y_bounds_top=(t[2], t[3]))


# `degree=11` here vs 16 for the 1-D design_to_pmm_stack (audit V-15): it reaches ONLY
# engine='hybrid' (a 2-D tensor-product basis, so lower by design). The DEFAULT engine='pure' never
# reads `degree` -- its resolution knob is `n_modes` (measured 2026-11-02 on a 400 nm Rectangle
# patch cell: R identical to every printed digit at degree 6 / 8). The 1-D spread is quantified in
# pmm_backend's V-15 note.
def design_to_pmm2d_stack(design, lambda_m: float, *, engine: str = "pure",
                          eps_by_region=None, n_modes: int = 8, degree: int = 11,
                          n_orders: Optional[int] = None, formulation: str = "li",
                          n_slices: Optional[int] = None,
                          cell_samples: Optional[int] = None,
                          pure_max_cells: int = 32, pure_grid: str = "auto",
                          pure_circle_core: Optional[float] = None,
                          taper_slices: int = 8, taper_rule: str = "midpoint"):
    """Translate a DynaMeta Design into a CONCRETE Lumenairy PMM2DStackPure
    ('pure') or PMM2DStackHybrid ('hybrid') at one wavelength.  Returns
    (stack, layer_names_top_first), names aligned with the Lumenairy layers
    (graded fields contribute several slabs per design layer).

    Layer dispatch (design layers are bottom -> top; both 2-D PMM stacks are
    superstrate-side first -> reversed(), audit C5-1 -- and so is every
    slice_eps_field consumer: the slicer returns ASCENDING-z substrate-first
    slabs, gated by the asymmetric-profile GATE C):
    laterally-uniform gridded EpsField (scalar OR tensor) -> reversed uniform slabs
    (a tensor slab is eps=(3, 3) on pure / a (1, 1, 3, 3) eps_tensor_cell on
    hybrid); laterally-STRUCTURED gridded EpsField -> reversed eps_cell /
    eps_tensor_cell slabs (hybrid) / raise (pure: rasters have no exact walls);
    uniform tensor EpsField on a layer without inclusions -> a uniform tensor
    layer (both); inclusions over the background (material, or the uniform scalar
    / TENSOR EpsField override, module docstring) -> per pure_grid
    (resolve_pure_grid): shapes= / a per-layer strip tile / layer_to_pure_cell on the union
    grid (pure) / the shared rasterizer (hybrid, byte-identical geometry to the
    RCWA bridge); uniform scalar -> eps.

    TAPERED inclusions (pure engine only, 2026-11-02): a layer holding ONE tapered
    Rectangle on a scalar background becomes lumenairy's add_tapered_pillar staircase
    of `taper_slices` exact-wall slices (`taper_rule` 'midpoint' | 'bottom'; the
    block comment above _TAPER_RULES), contributing taper_slices names; everything
    else tapered is refused.  The hybrid engine refuses every taper.  SEAM-CROSSING
    inclusions on the pure engine are wrapped exactly (_canonical_pure_cell: split
    rectangles, or a rigid origin shift for curved / tapered ones, published as
    stack.dynameta_origin_shift_m)."""
    _check_engine(engine)
    # 2026-11-02: refuse Stack.features (both engines) / Inclusion.taper (hybrid only)
    require_supported_geometry(
        design, "design_to_pmm2d_stack" if engine == "pure" else
        "design_to_pmm2d_stack(engine='hybrid'; engine='pure' builds a tapered Rectangle)",
        taper=(engine == "pure"))
    taper_slices, taper_rule = _check_taper_args(taper_slices, taper_rule)
    lum = _require_lumenairy()
    n_super, n_sub = end_media_indices(design, lambda_m)
    px = float(design.unit_cell.period_x_m)
    py = float(design.unit_cell.period_y_m)
    # mesh-region-keyed dicts (the run_pipeline/FEM bridge output) collapse to
    # design-layer keys; identity when already layer-keyed
    eps_by_region = collapse_regions_to_layers(design, eps_by_region or {})
    mode = None
    shift = (0.0, 0.0)
    if engine == "pure":
        # wrap seam-crossing inclusions exactly (layer names are kept, so the
        # layer-keyed eps_by_region still applies)
        design, shift = _canonical_pure_cell(design)
        mode = resolve_pure_grid(design, pure_grid, max_cells=pure_max_cells)
        # wave-7 (2026-11-02): the pure-engine routes newer than the bridge floor (5.22) are
        # refused with the version they need, BEFORE lumenairy is called -- they used to die
        # inside it (TypeError layer_grids / ImportError Circle / a numpy broadcast error from a
        # (3, 3) eps on <= 5.42). Floors measured per release: _common.FEATURE_FLOORS.
        where = "design_to_pmm2d_stack(engine='pure', pure_grid={!r} -> {!r})".format(
            pure_grid, mode)
        if mode == "per-layer":
            _require_feature("pmm2d_pure_per_layer", where)
        elif mode == "shapes" and any(L.inclusions for L in design.stack.layers):
            for L in design.stack.layers:        # a USER error (polygon) outranks the version
                for inc in L.inclusions:
                    _require_pure_kind(L, inc)
            _require_feature("pmm2d_pure_shapes", where)
        # (a TENSOR on the pure engine is gated where it is added -- _pure_tensor_gate -- so the
        # bridge's own refusals of what the pure basis cannot represent still come first)
        stack = lum.PMM2DStackPure(px, py, n_superstrate=complex(n_super),
                                   n_substrate=complex(n_sub), n_modes=int(n_modes),
                                   **({} if n_orders is None
                                      else dict(n_orders=int(n_orders))),
                                   **({} if mode != "per-layer"
                                      else dict(layer_grids="per-layer")))
        n_pure = (pure_union_grid_n(design, max_cells=pure_max_cells)
                  if mode == "union" else None)
    else:
        _check_pure_grid(pure_grid)              # validated on both engines
        stack = lum.PMM2DStackHybrid(px, py, n_superstrate=complex(n_super),
                                     n_substrate=complex(n_sub), degree=int(degree),
                                     formulation=formulation,
                                     **({} if n_orders is None
                                        else dict(n_orders=int(n_orders))))
        n_pure = None
    sx = sy = int(cell_samples or 128)           # hybrid raster (pixel walls)

    names: List[str] = []
    for L in reversed(design.stack.layers):      # superstrate side first
        ef = eps_by_region.get(L.name)
        if ef is not None and not getattr(ef, "is_uniform", True):
            v = ef.values_zyx
            lat_uniform = bool(np.allclose(v, v[:, :1, :1], rtol=1e-12, atol=0.0))
            if not lat_uniform and engine == "pure":
                raise ValueError(
                    "PMM2D bridge (pure): layer {!r} carries a laterally-"
                    "structured gridded {}EpsField; the pure staggered basis "
                    "needs exact analytic walls, not a raster -- use "
                    "engine='hybrid' or the RCWA bridge".format(
                        L.name, "TENSOR " if getattr(ef, "is_tensor", False) else ""))
            # slice_eps_field returns ascending-z (substrate-first) slabs; this
            # stack is superstrate-first, so reverse (audit C5-1)
            for slab in reversed(slice_eps_field(ef, 1.0 / _S_NM, n_slices=n_slices)):
                if slab.eps is not None:
                    stack.add_layer(slab.thickness_m, eps=complex(slab.eps))
                elif slab.eps_cell is not None:  # hybrid only (pure raised above)
                    stack.add_layer(slab.thickness_m,
                                    eps_cell=np.asarray(slab.eps_cell, dtype=complex))
                else:
                    # TENSOR slab (2026-11-02): (1, 1, 3, 3) when laterally uniform
                    # (slice_eps_field's own collapse), else an (Nx, Ny, 3, 3) raster
                    cell = np.asarray(slab.eps_tensor_cell, dtype=complex)
                    if engine == "pure":
                        if cell.shape[:2] != (1, 1):
                            # the bridge's first-sample test and the slicer's
                            # lateral-MEAN test disagree only at the 1e-12 edge
                            raise ValueError(
                                "PMM2D bridge (pure): layer {!r} sliced to a "
                                "laterally-STRUCTURED tensor raster {}; the pure "
                                "engine needs exact walls -- use engine='hybrid' "
                                "or the RCWA bridge".format(L.name, cell.shape))
                        _pure_tensor_gate(L)
                        stack.add_layer(slab.thickness_m, eps=cell[0, 0])
                    else:
                        stack.add_layer(slab.thickness_m, eps_tensor_cell=cell)
                names.append(L.name)
            continue
        if L.inclusions:
            bg = _layer_background(L, design, lambda_m, ef)
            if engine == "pure":
                if _is_tensor(bg):
                    _pure_tensor_gate(L)
                if any(getattr(inc, "taper", None) is not None for inc in L.inclusions):
                    stack.add_tapered_pillar(
                        L.thickness_m, n_slices=taper_slices, rule=taper_rule,
                        **_tapered_pillar_kwargs(L, design, lambda_m, px, py, bg))
                    names.extend([L.name] * taper_slices)
                    continue
                _add_pure_patterned(stack, L, design, lambda_m, px, py, mode=mode,
                                    n_pure=n_pure, ef=ef, bg=bg,
                                    circle_core=pure_circle_core)
            else:
                x, y = cell_axes(sx, sy, px, py)
                X, Y = np.meshgrid(x, y, indexing="ij")
                if _is_tensor(bg):
                    stack.add_layer(L.thickness_m, eps_tensor_cell=layer_eps_tensor_cell(
                        L, X, Y, lambda_m, design.materials, bg, period_x_m=px,
                        period_y_m=py))
                else:
                    cell = layer_eps_cell(L, X, Y, lambda_m, design.materials,
                                          eps_by_region, period_x_m=px, period_y_m=py)
                    stack.add_layer(L.thickness_m,
                                    eps_cell=np.asarray(cell, dtype=complex))
            names.append(L.name)
            continue
        if ef is not None and getattr(ef, "is_tensor", False):
            # a uniform TENSOR layer (2026-11-02): the (3, 3) passes through
            # unchanged -- DynaMeta's EpsField.tensor and lumenairy's add_layer
            # (3, 3) are the same eps[i, j], i, j in (x, y, z) layout (GATE F vs
            # Berreman, full Jones; a transpose is caught there)
            t33 = np.asarray(ef.tensor, dtype=complex).reshape(3, 3)
            if engine == "pure":
                _pure_tensor_gate(L)
                stack.add_layer(L.thickness_m, eps=t33)
            else:
                stack.add_layer(L.thickness_m, eps_tensor_cell=t33[None, None].copy())
            names.append(L.name)
            continue
        if ef is not None:                       # uniform scalar (effect-modulated)
            stack.add_layer(L.thickness_m, eps=complex(ef.scalar))
        else:
            stack.add_layer(L.thickness_m,
                            eps=complex(design.materials.get(L.background_material)
                                        .eps(lambda_m)))
        names.append(L.name)
    stack.dynameta_origin_shift_m = shift
    # audit S4-17 (wave-7, 2026-11-02): the end media ride on the stack so the solver closure
    # does not evaluate end_media_indices a second time per solve (same values, bit for bit)
    stack.dynameta_end_media = (n_super, n_sub)
    return stack, names


def _pure_tensor_gate(L) -> None:
    """A (3, 3) permittivity on PMM2DStackPure needs lumenairy >= 5.43 (wave-7, 2026-11-02:
    5.42.1 dies with a numpy broadcast error at solve / an eps_cell shape error)."""
    _require_feature("pmm2d_pure_tensor",
                     "design_to_pmm2d_stack(engine='pure'): TENSOR eps on layer {!r}".format(
                         L.name))


def _add_pure_patterned(stack, L, design, lambda_m: float, px: float, py: float, *,
                        mode: str, n_pure, ef, bg, circle_core):
    """One inclusion layer onto a PMM2DStackPure along the resolved pure_grid route
    (module docstring).  'union' keeps the v1 call byte for byte on scalar cells
    (bg_eps = the scalar override or None -> the material eps)."""
    if mode == "union":
        cell = layer_to_pure_cell(L, design, lambda_m, px, py, n_pure,
                                  bg_eps=(bg if _is_tensor(bg)
                                          else (ef.scalar if ef is not None
                                                else None)))
        stack.add_layer(L.thickness_m, eps_cell=cell)
        return
    curved = any(getattr(inc.shape, "kind", "") != "rectangle" for inc in L.inclusions)
    if mode == "per-layer" and not curved:
        xw, yw, tile = _rect_strip_tile(L, design, lambda_m, px, py, bg)
        if not xw and not yw:                    # rectangles cover the whole cell
            stack.add_layer(L.thickness_m, eps=tile[0, 0])
            return
        stack.add_layer(L.thickness_m, eps_cell=tile, x_walls=xw or None,
                        y_walls=yw or None)
        return
    shapes = _layer_lum_shapes(L, design, lambda_m, px, py, circle_core=circle_core)
    try:
        stack.add_layer(L.thickness_m, shapes=shapes,
                        background_eps=(np.asarray(bg, dtype=complex) if _is_tensor(bg)
                                        else complex(bg)))
    except ValueError as exc:
        if mode != "shapes":
            raise
        # the stack-wide merge refuses outlines that cross in plan view, two
        # curves on one cell edge, or a folding map -- per-layer maps are the
        # lumenairy route for exactly that (Phase E2, the curved mortar)
        raise ValueError(
            "PMM2D bridge (pure, pure_grid='shapes'): layer {!r} cannot join the "
            "stack-wide shape map ({}); pass pure_grid='per-layer' (each layer its "
            "own wall grid / map, joined by lumenairy's mortar) or use "
            "engine='hybrid' / the RCWA bridge".format(L.name, exc)) from exc


def make_lumenairy_pmm2d_solver(*, engine: str = "pure", n_modes: int = 8,
                                degree: int = 11, n_orders: Optional[int] = None,
                                formulation: str = "li",
                                n_slices: Optional[int] = None,
                                cell_samples: Optional[int] = None,
                                pure_max_cells: int = 32,
                                absorption: bool = False, pure_grid: str = "auto",
                                pure_circle_core: Optional[float] = None,
                                taper_slices: int = 8, taper_rule: str = "midpoint"):
    """Build an `optical_solver` for run_pipeline backed by the Lumenairy 2-D
    crossed-patterned PMM (the exact seam signature + solve_sweep, mirroring the
    RCWA/1-D-PMM bridges incl. per-wavelength end media).  engine='pure' is the
    no-Fourier-floor referee (converge via n_modes; n_orders only needs to cover
    the propagating orders); engine='hybrid' is the Fourier-projected stack
    (converge via degree [ODD] + n_orders).  In-plane incidence only; TOP
    incidence only; R/T are total order-summed efficiencies; r / t are the
    zeroth-order co-polarized reflection / transmission Jones entries with the
    shared lab-basis -> Byrnes p-pol conversion (t since 2026-11-02, from
    lumenairy's jones_transmission(); all gated in
    validation/lumenairy_pmm2d_bridge).  pure_grid / pure_circle_core select the
    pure engine's patterned-layer route (resolve_pure_grid, module docstring;
    validated on both engines, used by 'pure').

    absorption=True (BOTH engines): solves with retain_internal=True and fills
    per_region_absorption keyed by DESIGN layer name (slabs of a graded layer sum
    into their layer) + A_independent from PMM2DStack{Hybrid,Pure}.layer_absorption
    -- the internal z-flux difference per layer ((n_layers, 2) per incident pol),
    an independent volumetric measurement (lumenairy's own invariant closes it
    against 1 - sum R - sum T).  The pure engine gained internal retention in
    lumenairy 5.22 (audit C3); before that absorption raised for engine='pure'.

    taper_slices / taper_rule (2026-11-02): the z-staircase of a TAPERED Rectangle
    pillar on the pure engine (lumenairy add_tapered_pillar; design_to_pmm2d_stack).
    The taper is a staircase of exact-wall slices: converge in taper_slices as well
    as n_modes ('midpoint' error O(1/taper_slices^2), measured in validation/
    lumenairy_pmm2d_taper_wrap GATE M)."""
    _check_engine(engine)
    _check_pure_grid(pure_grid)
    _check_taper_args(taper_slices, taper_rule)

    def _solve_at(design, eps_by_region, lambda_m):
        t0 = time.perf_counter()
        _guard_incidence_side(design.optical)
        theta, phi = _angles_rad(design.optical)
        _guard_conical_ppol(design.optical, phi)
        stack, names = design_to_pmm2d_stack(
            design, lambda_m, engine=engine, eps_by_region=eps_by_region,
            n_modes=n_modes, degree=degree, n_orders=n_orders,
            formulation=formulation, n_slices=n_slices, cell_samples=cell_samples,
            pure_max_cells=pure_max_cells, pure_grid=pure_grid,
            pure_circle_core=pure_circle_core, taper_slices=taper_slices,
            taper_rule=taper_rule)
        stack.set_source(lambda_m, theta=theta, phi=phi)
        # both PMM2DStackHybrid.solve and PMM2DStackPure.solve take the same
        # keyword surface (retain_internal retains the internals layer_absorption
        # integrates -- pure since lumenairy 5.22, audit C3)
        orders, R2, T2, jones = stack.solve(retain_internal=absorption)
        row = _pol_row(design.optical)
        R = float(np.sum(R2[row]))
        T = float(np.sum(T2[row]))
        n_sup, n_sb = stack.dynameta_end_media      # audit S4-17: set by design_to_pmm2d_stack
        rf, tf = _p_basis_conversion(getattr(design.optical, "polarization", "y"),
                                     theta, n_sup, n_sb)
        r = complex(np.asarray(jones)[row, row]) * complex(rf)
        # zeroth-order transmission Jones (PerOrderAmplitudesMixin, both engines;
        # columns = incident lab E_x / E_y, rows = [E_x; E_y]) -> co-pol entry with
        # the RCWA bridge's p-basis t factor cos_i / cos_t (GATE A vs TMM)
        t = complex(np.asarray(stack.jones_transmission())[row, row]) * complex(tf)
        a_ind, pra = None, None
        if absorption:
            try:
                la = np.asarray(stack.layer_absorption())[:, row]  # (n_layers, 2)
                pra = {}
                for name, val in zip(names, la):
                    pra[name] = pra.get(name, 0.0) + float(val)
                a_ind = float(np.sum(la))
            except Exception as exc:              # pragma: no cover - defensive
                import warnings
                warnings.warn("lumenairy PMM2D bridge: per-layer absorption "
                              "unavailable ({}); A_independent left unset".format(exc), AdvisoryWarning,
                              stacklevel=2)
        return OpticalResult(r=r, R=R, phase_deg=float(np.degrees(np.angle(r))),
                             solve_time_s=time.perf_counter() - t0, t=t, T=T,
                             A=1.0 - R - T, A_independent=a_ind, R_flux=R, T_flux=T,
                             per_region_absorption=pra)

    def _solve(design, geo, eps_by_region, lambda_m, n_super, n_sub):
        _require_eps_convention(eps_by_region, "make_lumenairy_pmm2d_solver")   # audit V-5
        return _solve_at(design, eps_by_region, lambda_m)

    def _solve_sweep(design, geo, assemble_at, lams, n_super, n_sub):
        return [_solve_at(design, assemble_at(lam), lam) for lam in lams]

    _solve.solve_sweep = _solve_sweep
    return _solve
