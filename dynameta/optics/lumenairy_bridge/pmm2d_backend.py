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
hybrid).  NB the RCWA bridge (rcwa_backend, 2026-11-02) instead REPLACES the whole
layer with the uniform tensor and drops the inclusions; the two bridges differ on
that input until the RCWA side is aligned (GATE G therefore uses a direct RCWAStack
oracle).  Gridded tensor EpsFields slice to (1, 1, 3, 3) cells when laterally uniform
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
  shape primitives are EXACT for these) lying inside the cell (no wrap-around);
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
                                                      require_lumenairy as _require_lumenairy)
from dynameta.optics.rasterize import cell_axes, layer_eps_cell
from dynameta.optics.tmm_reference import S as _S_NM
from dynameta.optics.tmm_reference import end_media_indices
from dynameta.geometry.guards import require_supported_geometry

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
    """Every pure-engine inclusion must lie inside [0, px] x [0, py] (no wrap-around
    across the periodic seam -- lumenairy's shape layouts refuse it too)."""
    xlo, xhi, ylo, yhi = inc.shape.bbox_m()
    if xlo < -_REL_TOL * px or xhi > px * (1.0 + _REL_TOL) \
            or ylo < -_REL_TOL * py or yhi > py * (1.0 + _REL_TOL):
        raise ValueError(
            "PMM2D bridge ({}): layer {!r} {} bbox x [{:.3e}, {:.3e}] y [{:.3e}, "
            "{:.3e}] leaves the cell [0, {:.3e}] x [0, {:.3e}] (wrap-around is a "
            "follow-on; shift the unit-cell origin)".format(
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
        xlo, xhi, ylo, yhi = inc.shape.bbox_m()
        if xlo < -_REL_TOL * px or xhi > px * (1.0 + _REL_TOL) \
                or ylo < -_REL_TOL * py or yhi > py * (1.0 + _REL_TOL):
            raise ValueError(
                "PMM2D bridge (pure): layer {!r} Rectangle bbox x [{:.3e}, {:.3e}] "
                "y [{:.3e}, {:.3e}] leaves the cell [0, {:.3e}] x [0, {:.3e}] "
                "(wrap-around is a follow-on; shift the unit-cell origin)".format(
                    layer.name, xlo, xhi, ylo, yhi, px, py))
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
    cost problem, not just a validity one)."""
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
    path ran (the gated centred patch: N = 4, S = 3) on the union grid."""
    _check_pure_grid(pure_grid)
    if pure_grid != "auto":
        return pure_grid
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


def _tensor_raster(layer, X, Y, lambda_m: float, materials, bg):
    """The hybrid engine's (Sx, Sy, 3, 3) eps_tensor_cell for an inclusion layer on a
    TENSOR background: the shared rasterizer's painting rule (contains_m masks in
    ASCENDING priority, so the highest priority wins overlaps) with every scalar
    inclusion as eps * I3.  (optics.rasterize.layer_eps_cell itself refuses a
    tensor eps_by_region entry -- audit C5-2 -- so the tensor variant lives here.)"""
    cell = np.broadcast_to(np.asarray(bg, dtype=complex), X.shape + (3, 3)).copy()
    for inc in sorted(layer.inclusions, key=lambda i: getattr(i, "priority", 0)):
        mask = np.asarray(inc.shape.contains_m(X, Y), dtype=bool)
        cell[mask] = _painted(materials.get(inc.material).eps(lambda_m), True)
    return cell


def design_to_pmm2d_stack(design, lambda_m: float, *, engine: str = "pure",
                          eps_by_region=None, n_modes: int = 8, degree: int = 11,
                          n_orders: Optional[int] = None, formulation: str = "li",
                          n_slices: Optional[int] = None,
                          cell_samples: Optional[int] = None,
                          pure_max_cells: int = 32, pure_grid: str = "auto",
                          pure_circle_core: Optional[float] = None):
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
    RCWA bridge); uniform scalar -> eps."""
    require_supported_geometry(design, "design_to_pmm2d_stack")   # 2026-11-02: refuse Stack.features / Inclusion.taper
    lum = _require_lumenairy()
    _check_engine(engine)
    n_super, n_sub = end_media_indices(design, lambda_m)
    px = float(design.unit_cell.period_x_m)
    py = float(design.unit_cell.period_y_m)
    # mesh-region-keyed dicts (the run_pipeline/FEM bridge output) collapse to
    # design-layer keys; identity when already layer-keyed
    eps_by_region = collapse_regions_to_layers(design, eps_by_region or {})
    mode = None
    if engine == "pure":
        mode = resolve_pure_grid(design, pure_grid, max_cells=pure_max_cells)
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
                        stack.add_layer(slab.thickness_m, eps=cell[0, 0])
                    else:
                        stack.add_layer(slab.thickness_m, eps_tensor_cell=cell)
                names.append(L.name)
            continue
        if L.inclusions:
            bg = _layer_background(L, design, lambda_m, ef)
            if engine == "pure":
                _add_pure_patterned(stack, L, design, lambda_m, px, py, mode=mode,
                                    n_pure=n_pure, ef=ef, bg=bg,
                                    circle_core=pure_circle_core)
            else:
                x, y = cell_axes(sx, sy, px, py)
                X, Y = np.meshgrid(x, y, indexing="ij")
                if _is_tensor(bg):
                    stack.add_layer(L.thickness_m, eps_tensor_cell=_tensor_raster(
                        L, X, Y, lambda_m, design.materials, bg))
                else:
                    cell = layer_eps_cell(L, X, Y, lambda_m, design.materials,
                                          eps_by_region)
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
    return stack, names


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
                                pure_circle_core: Optional[float] = None):
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
    lumenairy 5.22 (audit C3); before that absorption raised for engine='pure'."""
    _check_engine(engine)
    _check_pure_grid(pure_grid)

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
            pure_circle_core=pure_circle_core)
        stack.set_source(lambda_m, theta=theta, phi=phi)
        # both PMM2DStackHybrid.solve and PMM2DStackPure.solve take the same
        # keyword surface (retain_internal retains the internals layer_absorption
        # integrates -- pure since lumenairy 5.22, audit C3)
        orders, R2, T2, jones = stack.solve(retain_internal=absorption)
        row = _pol_row(design.optical)
        R = float(np.sum(R2[row]))
        T = float(np.sum(T2[row]))
        n_sup, n_sb = end_media_indices(design, lambda_m)
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
                              "unavailable ({}); A_independent left unset".format(exc),
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
